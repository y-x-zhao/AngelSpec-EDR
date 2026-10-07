#!/usr/bin/env python3
"""Download and normalize Open-PerfectBlend for the Qwen3 EDR recipes.

1. Stream ``mlabonne/open-perfectblend`` and write local conversation JSONL,
   using the normalization rules of ``angelspec.data.preprocessing``.
2. Unless ``--skip-tokenization`` is given, run ``load_conversation_dataset``
   with the training YAML to populate or validate its tokenized cache. Configs
   that set ``dataset.target_sampling`` require the regenerated target cache
   (``tools/regenerate_perfectblend.py``) for this step.

Model/tokenizer assets must be downloaded first (e.g. with ``hf download``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "dataset" / "Open-PerfectBlend"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "open-perfectblend.jsonl"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "vllm_qwen3_8b_dfly_edr.yaml"

# Keep the Hugging Face download cache inside the dataset tree unless HF_HOME or
# HF_DATASETS_CACHE is already set.
os.environ.setdefault("HF_HOME", str(DEFAULT_DATA_DIR / "cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(DEFAULT_DATA_DIR / "cache" / "huggingface" / "datasets"),
)

from angelspec.config.train_config import config_to_flat_args, load_config  # noqa: E402
from angelspec.data.dataset import load_conversation_dataset  # noqa: E402
from angelspec.data.preprocessing import _normalize_conversation  # noqa: E402
from angelspec.data.utils import estimate_row_count, load_hf_dataset  # noqa: E402

_ROLE_ALIASES = {
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def normalize_sample(
    row: Mapping[str, Any],
    index: int,
    *,
    min_turns: int = 2,
) -> dict[str, Any] | None:
    """Normalize one source row to the conversation schema consumed by AngelSpec."""

    raw_conversation = (
        row.get("conversations") or row.get("conversation") or row.get("messages")
    )
    if not isinstance(raw_conversation, list):
        return None

    messages = _normalize_conversation(raw_conversation)
    if len(messages) < min_turns:
        return None

    normalized_messages = []
    has_assistant = False
    for message in messages:
        if not isinstance(message, Mapping):
            return None
        role = _ROLE_ALIASES.get(str(message.get("role", "")), message.get("role"))
        content = message.get("content")
        if not role or content is None:
            return None
        normalized = dict(message)
        normalized["role"] = role
        normalized_messages.append(normalized)
        has_assistant = has_assistant or (role == "assistant" and bool(content))

    if not has_assistant:
        return None

    sample: dict[str, Any] = {
        "id": str(row.get("id", f"perfectblend_{index}")),
        "conversations": normalized_messages,
    }
    for key in ("tools", "reasoning_effort"):
        value = row.get(key)
        if value:
            sample[key] = value
    return sample


def _reservoir_sample(
    rows: Iterable,
    sample_size: int,
    *,
    seed: int,
    min_turns: int,
    total: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Select a deterministic uniform sample without materializing the raw dataset."""

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    valid_seen = 0
    skipped = 0
    for index, row in enumerate(tqdm(rows, desc="Downloading and normalizing", total=total)):
        sample = normalize_sample(row, index, min_turns=min_turns)
        if sample is None:
            skipped += 1
            continue
        valid_seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(sample)
            continue
        replacement = rng.randrange(valid_seen)
        if replacement < sample_size:
            reservoir[replacement] = sample
    rng.shuffle(reservoir)
    return reservoir, skipped


def prepare_jsonl(
    source: str,
    output: Path,
    *,
    force: bool,
    sample_size: int | None,
    seed: int,
    min_turns: int,
) -> int:
    """Download, normalize, and atomically write the local training JSONL."""

    if output.exists() and not force:
        print(f"Reusing existing normalized dataset: {output}")
        count = estimate_row_count(str(output))
        return int(count or 0)

    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if min_turns < 1:
        raise ValueError("min_turns must be at least 1")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_hf_dataset(source)
    total = estimate_row_count(source)
    temp_name: str | None = None
    written = 0
    skipped = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            if sample_size is not None:
                selected, skipped = _reservoir_sample(
                    rows,
                    sample_size,
                    seed=seed,
                    min_turns=min_turns,
                    total=total,
                )
                for sample in selected:
                    temp_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written = len(selected)
            else:
                for index, row in enumerate(
                    tqdm(rows, desc="Downloading and normalizing", total=total)
                ):
                    sample = normalize_sample(row, index, min_turns=min_turns)
                    if sample is None:
                        skipped += 1
                        continue
                    temp_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1

        if written == 0:
            raise RuntimeError("Open-PerfectBlend preprocessing produced no valid samples")
        os.replace(temp_name, output)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)

    print(f"Wrote {written:,} normalized samples to {output} ({skipped:,} invalid rows skipped)")
    return written


def warm_tokenized_cache(config_path: Path, output: Path, *, num_proc: int) -> int:
    """Populate the exact cache that ``load_conversation_dataset`` uses in training."""

    if num_proc < 1:
        raise ValueError("num_proc must be at least 1")

    original_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        config = load_config(config_path=str(config_path))
        args = config_to_flat_args(config)
        args.num_proc = num_proc

        configured_dataset = Path(args.train_data_path).resolve()
        if configured_dataset != output.resolve():
            raise ValueError(
                f"Config resolves dataset.train_data_path to {configured_dataset}, "
                f"but --output is {output.resolve()}. Update the YAML or use its default output."
            )

        target_path = Path(args.target_model_path).expanduser()
        if not target_path.is_absolute():
            target_path = (REPO_ROOT / target_path).resolve()
        if not (target_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Target tokenizer/model is missing at {target_path}. Download "
                "it (e.g. with `hf download`) before dataset preprocessing."
            )

        prompts = load_conversation_dataset(args)
        print(f"Tokenized cache is ready for {len(prompts):,} samples under {args.cache_dir}")
        return len(prompts)
    finally:
        os.chdir(original_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, normalize, and pre-tokenize Open-PerfectBlend for AngelSpec"
    )
    parser.add_argument("--source", default="mlabonne/open-perfectblend")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="CPU tokenizer worker count (default: min(32, available CPUs))",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Uniformly sample this many valid conversations instead of using all rows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing normalized JSONL; the tokenized cache remains hash-versioned",
    )
    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Only download/normalize JSONL; do not populate AngelSpec's tokenized cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    prepare_jsonl(
        args.source,
        output,
        force=args.force,
        sample_size=args.sample_size,
        seed=args.seed,
        min_turns=args.min_turns,
    )
    if not args.skip_tokenization:
        warm_tokenized_cache(config_path, output, num_proc=args.num_proc)


if __name__ == "__main__":
    main()
