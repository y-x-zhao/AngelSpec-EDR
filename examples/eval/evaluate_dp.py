#!/usr/bin/env python3
"""Offline pathwise EDR evaluation over reusable target trajectories.

The first evaluation samples equally many trajectories per prompt, with at least
1,500 per dataset, through one batched vLLM call and caches the target residual
streams and LM-head inputs needed for offline scoring. Later draft checkpoints
reuse those fixed trajectories and
reconstruct target logits from the cached hidden rows. AngelSpec's NumPy
Bellman recurrence then produces two pathwise target-round estimators.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from angelspec.controller.eval_datasets import (  # noqa: E402
    DEEPSPEC_DATASETS,
    load_dataset_prompts,
)
from angelspec.utils.sampling import (  # noqa: E402
    sampling_cache_key,
    target_model_cache_id,
    validate_sampling_parameters,
    validate_vllm_sampling_parameters,
)
from examples.eval.common import (  # noqa: E402
    HFTargetRunner,
    ResolvedCheckpoint,
    _validate_target_geometry,
    checkpoint_output_name,
    load_draft_model,
    resolve_checkpoint,
)
from examples.eval.dp_resume import DPResultStore, validate_records  # noqa: E402

DEFAULT_PROMPTS_DIR = REPO_ROOT / "angelspec" / "data" / "eval_prompts"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval_outputs"
DFLY_QUERY_BLOCK_SIZE = 8
NUM_DFLY_PROPOSALS = DFLY_QUERY_BLOCK_SIZE - 1
MIN_TARGET_SEQUENCES_PER_DATASET = 1500
TARGET_CACHE_FORMAT_VERSION = 4
TARGET_CACHE_DIRNAME = "eval_dp_cache"
SUPPORTED_DATASETS = DEEPSPEC_DATASETS
DATASET_ALIASES = {
    "AIME25": "aime25", "LCB": "livecodebench", "lcb": "livecodebench",
    "Alpaca": "alpaca", "Arena-Hard": "arena-hard-v2", "arena-hard": "arena-hard-v2",
}
DP_TOKEN_COUNTING = "sampled_tokens_including_terminal"
# Version of the target-distribution reconstruction (vLLM's FP32 PyTorch
# temperature/top-k/top-p arithmetic). It is part of the evaluation identity, so
# results scored under a different version are rejected instead of reused.
DP_SCORING_VERSION = 2


def resolve_output_root(
    checkpoint_path: str | Path, explicit_root: str | Path | None = None,
) -> Path:
    """Group known training recipes without guessing the objective of loose weights."""
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()
    recipes = {}
    for family, output_family in (
        ("qwen3-4b-dspark", "qwen3-4b-spark"),
        ("qwen3-4b-spark", "qwen3-4b-spark"),
        ("qwen3-8b-dfly", "qwen3-8b-dfly"),
    ):
        for objective in ("e2e", "edr", "lk"):
            for infix in ("", "-cpt"):
                recipes[f"{family}{infix}-{objective}"] = f"{output_family}-{objective}"
    path = Path(checkpoint_path).expanduser().absolute()
    # Keep a descriptive caller-side symlink name, but also recognize a recipe
    # in its destination when the caller uses a generic alias such as "latest".
    for candidate in (path, path.resolve()):
        for ancestor in (candidate, *candidate.parents):
            recipe = recipes.get(ancestor.name.lower())
            if recipe is not None:
                return DEFAULT_OUTPUT_ROOT / recipe
    return DEFAULT_OUTPUT_ROOT


@dataclass(frozen=True)
class PromptSpec:
    dataset: str
    dataset_index: int
    global_index: int
    prompt: str


@dataclass(frozen=True)
class TargetCacheRequest:
    dataset: str
    cache_dir: Path
    prompt_specs: tuple[PromptSpec, ...]


@dataclass(frozen=True)
class TargetTrajectory:
    dataset: str
    dataset_index: int
    global_index: int
    prompt: str
    prompt_token_ids: tuple[int, ...]
    ordinary_token_ids: tuple[int, ...]
    boundary_token_id: int
    target_text: str
    finish_reason: str
    stop_reason: int | str | None
    boundary_source: str
    sample_index: int = 0
    trajectory_index: int = 0

    @property
    def ordinary_tokens(self) -> int:
        """DP horizon length, excluding the sampled terminal boundary."""
        return len(self.ordinary_token_ids)

    @property
    def generated_tokens(self) -> int:
        """Actual sampled output length, including EOS/stop/length boundary."""
        return self.ordinary_tokens + 1


def target_samples_per_prompt(prompt_count: int) -> int:
    if prompt_count <= 0:
        raise ValueError("A target cache dataset must contain at least one prompt")
    return (MIN_TARGET_SEQUENCES_PER_DATASET + prompt_count - 1) // prompt_count


@dataclass(frozen=True)
class DPSequenceResult:
    dataset: str
    dataset_index: int
    global_index: int
    prompt_tokens: int
    generated_tokens: int
    expected_rounds: float
    weighted_cost_to_go: float
    sample_index: int = 0
    trajectory_index: int = 0

    @property
    def round_start_mal(self) -> float:
        return self.generated_tokens / self.expected_rounds

    @property
    def weighted_cost_mal(self) -> float:
        return self.generated_tokens / self.weighted_cost_to_go


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache equal Qwen3 target rollouts per prompt (at least 1500 per dataset), "
            "then evaluate DSpark or DFly "
            "checkpoints against those fixed paths with the exact EDR NumPy dynamic "
            "program on one GPU."
        )
    )
    parser.add_argument("--draft-checkpoint", help="HF export or AngelSpec checkpoint")
    parser.add_argument(
        "--target-model",
        default=os.environ.get("TARGET_MODEL") or None,
        help="Target model path (required; default: TARGET_MODEL)",
    )
    parser.add_argument(
        "--draft-config",
        type=Path,
        default=None,
        help="Draft architecture config JSON (required)",
    )
    parser.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS_DIR))
    parser.add_argument(
        "--output-root",
        default=os.environ.get("EVAL_OUTPUT_ROOT") or os.environ.get("EVAL_OUTPUT_DIR"),
        help="Report root override; default: eval_outputs/<recognized model-objective>",
    )
    parser.add_argument(
        "--target-cache-root",
        default=os.environ.get(
            "EVAL_DP_CACHE_ROOT", str(DEFAULT_OUTPUT_ROOT / TARGET_CACHE_DIRNAME)
        ),
        help=(
            "Shared target-trajectory/feature cache root. By default this is "
            "./eval_outputs/eval_dp_cache under the repository, independent of --output-root."
        ),
    )
    parser.add_argument("--checkpoint-name")
    parser.add_argument(
        "--resolve-output-root", metavar="CHECKPOINT",
        help="Print the inferred report root and exit without loading models",
    )
    parser.add_argument(
        "--resolve-checkpoint-name",
        metavar="CHECKPOINT",
        help="Print the resolved output name and exit without importing vLLM",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        type=lambda name: DATASET_ALIASES.get(name, name),
        default=list(DEEPSPEC_DATASETS),
        help=(
            "Prompt sets to evaluate "
            "(default: all nine DeepSpec sets). Completed matching datasets are skipped."
        ),
    )
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Deterministically sample at most this many prompts per dataset; 0 means all",
    )
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=NUM_DFLY_PROPOSALS,
        help="Seven proposals for DSpark Block7 or DFly Block8",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Target sampling temperature, shared by the draft (required); "
            "0 uses greedy one-hot distributions"
        ),
    )
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=None,
        help="Draft temperature; when supplied, it must equal --temperature",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Target top-p (required; 1 disables it)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Target top-k (required; -1 disables it)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--edr-chunk-size", type=int, default=2048)
    parser.add_argument("--edr-vocab-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=32,
        help="Maximum cached trajectories scored together (default: 32)",
    )
    parser.add_argument(
        "--score-max-batch-tokens",
        type=int,
        default=65536,
        help="Maximum batch_size * padded full-sequence length during DP scoring",
    )
    parser.add_argument(
        "--score-max-target-tokens",
        type=int,
        default=8192,
        help="Maximum generated-token rows projected through the target LM head per batch",
    )
    parser.add_argument(
        "--score-dp-workers",
        type=int,
        default=8,
        help="CPU workers used by the batched NumPy Bellman recurrence",
    )
    parser.add_argument(
        "--score-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load and pin the next cached feature batch while the GPU scores this one",
    )
    parser.add_argument(
        "--target-attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
        help="Hugging Face target attention used during offline trajectory scoring",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resolve_checkpoint_name or args.resolve_output_root:
        return
    if not args.draft_checkpoint:
        parser.error("--draft-checkpoint is required")
    missing = [
        option for option, value in (
            ("--target-model", args.target_model),
            ("--draft-config", args.draft_config),
            ("--temperature", args.temperature),
            ("--top-p", args.top_p),
            ("--top-k", args.top_k),
        ) if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    args.output_root = str(resolve_output_root(args.draft_checkpoint, args.output_root))
    try:
        validate_sampling_parameters(args.temperature, args.top_k, args.top_p)
    except ValueError as exc:
        parser.error(str(exc))
    if args.draft_temperature is not None and args.draft_temperature != args.temperature:
        parser.error("--draft-temperature must equal --temperature; draft top-k/top-p are disabled")
    args.draft_temperature = args.temperature
    for name in (
        "max_new_tokens",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "edr_chunk_size",
        "edr_vocab_chunk_size",
        "score_batch_size",
        "score_max_batch_tokens",
        "score_max_target_tokens",
        "score_dp_workers",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.score_batch_size > args.edr_chunk_size:
        parser.error("--score-batch-size cannot exceed --edr-chunk-size")
    if args.max_model_len <= args.max_new_tokens + 1:
        parser.error("--max-model-len must leave room for a prompt and terminal boundary")
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        parser.error("--limit-per-dataset must be positive")
    if args.sample_size < 0:
        parser.error("--sample-size cannot be negative")
    if args.log_every < 0:
        parser.error("--log-every cannot be negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.num_speculative_tokens != NUM_DFLY_PROPOSALS:
        parser.error("DSpark Block7 / DFly Block8 requires --num-speculative-tokens 7")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("--datasets must not contain duplicate dataset names")


def load_prompt_specs(
    args: argparse.Namespace,
    datasets: Sequence[str] | None = None,
) -> tuple[list[PromptSpec], dict[str, int]]:
    specs: list[PromptSpec] = []
    counts: dict[str, int] = {}
    for dataset in args.datasets if datasets is None else datasets:
        prompts = load_dataset_prompts(
            dataset,
            sample_size=args.sample_size,
            seed=args.dataset_seed,
            limit=args.limit_per_dataset,
            prompts_dir=args.prompts_dir,
        )
        if not prompts:
            raise ValueError(f"Dataset {dataset!r} contains no usable prompts")
        counts[dataset] = len(prompts)
        for dataset_index, prompt in enumerate(prompts):
            specs.append(
                PromptSpec(
                    dataset=dataset,
                    dataset_index=dataset_index,
                    global_index=len(specs),
                    prompt=prompt,
                )
            )
    return specs, counts


def _normalize_token_ids(*values: Any) -> set[int]:
    token_ids: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            token_ids.add(int(value))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    return token_ids


def _json_stop_reason(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | str):
        return value
    return str(value)


def _resolve_stop_token_id(stop_reason: Any, tokenizer: Any) -> int | None:
    if isinstance(stop_reason, bool):
        return None
    if isinstance(stop_reason, int):
        return int(stop_reason)
    if not isinstance(stop_reason, str):
        return None
    token_id = tokenizer.convert_tokens_to_ids(stop_reason)
    if token_id is None:
        return None
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if unknown_id is not None and int(token_id) == int(unknown_id):
        return None
    return int(token_id)


def split_target_tokens(
    generated_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    finish_reason: str,
    stop_token_id: int | None,
) -> tuple[tuple[int, ...], int, str]:
    """Split target samples into ordinary tokens plus one positional boundary."""
    generated = [int(token_id) for token_id in generated_token_ids]
    eos_index = next(
        (index for index, token_id in enumerate(generated) if token_id in eos_token_ids),
        None,
    )
    if eos_index is not None:
        ordinary = generated[:eos_index]
        boundary = generated[eos_index]
        source = "generated_eos"
    elif stop_token_id is not None:
        try:
            stop_index = generated.index(stop_token_id)
        except ValueError:
            stop_index = -1
        if stop_index >= 0:
            ordinary = generated[:stop_index]
            boundary = generated[stop_index]
            source = "generated_stop_token"
        else:
            ordinary = generated
            boundary = stop_token_id
            source = "appended_stop_token"
    elif len(generated) >= max_new_tokens + 1:
        ordinary = generated[:max_new_tokens]
        boundary = generated[max_new_tokens]
        source = "length_lookahead"
    elif finish_reason == "length" and generated:
        # A request can hit max_model_len before its requested token budget. The
        # final sampled token is still a valid positional boundary for the
        # preceding path.
        ordinary = generated[:-1]
        boundary = generated[-1]
        source = "model_length_boundary"
    else:
        raise RuntimeError(
            "Target generation returned no recoverable terminal token; request one extra "
            "token or use a token-id stop condition"
        )
    if len(ordinary) > max_new_tokens:
        raise RuntimeError(
            f"Target trajectory has {len(ordinary)} ordinary tokens, exceeding "
            f"--max-new-tokens={max_new_tokens}"
        )
    return tuple(ordinary), int(boundary), source


def _fallback_prompt_token_ids(tokenizer: Any, prompt: str, enable_thinking: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _shutdown_vllm(llm: Any) -> None:
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def generate_target_trajectories(
    args: argparse.Namespace,
    prompt_specs: Sequence[PromptSpec],
    tokenizer: Any,
    eos_token_ids: set[int],
) -> tuple[list[TargetTrajectory], float]:
    """Generate every target path in one vLLM continuous-batching call."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=1,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        generation_config="vllm",
    )
    try:
        prompt_counts = Counter(spec.dataset for spec in prompt_specs)
        samples_by_dataset = {
            dataset: target_samples_per_prompt(count)
            for dataset, count in prompt_counts.items()
        }
        sampling_params = [
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                # vLLM requires n=1 for greedy decoding. Identical deterministic
                # copies below preserve the same per-prompt trajectory count.
                n=1 if args.temperature == 0 else samples_by_dataset[spec.dataset],
                stop_token_ids=sorted(eos_token_ids),
                # The extra sample is the positional terminal boundary when the
                # ordinary output reaches --max-new-tokens without EOS.
                max_tokens=args.max_new_tokens + 1,
            )
            for spec in prompt_specs
        ]
        for params in sampling_params:
            validate_vllm_sampling_parameters(
                params,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        messages = [[{"role": "user", "content": spec.prompt}] for spec in prompt_specs]
        started = time.perf_counter()
        request_outputs = llm.chat(
            messages,
            sampling_params,
            use_tqdm=not args.disable_progress,
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
        )
        generation_seconds = time.perf_counter() - started
        if len(request_outputs) != len(prompt_specs):
            raise RuntimeError(
                f"vLLM returned {len(request_outputs)} outputs for {len(prompt_specs)} prompts"
            )

        trajectories = []
        for spec, request_output in zip(prompt_specs, request_outputs):
            completions = getattr(request_output, "outputs", None) or []
            samples_per_prompt = samples_by_dataset[spec.dataset]
            expected_completions = 1 if args.temperature == 0 else samples_per_prompt
            if len(completions) != expected_completions:
                raise RuntimeError(
                    f"Target request {spec.global_index} returned {len(completions)} "
                    f"completions, expected {expected_completions}"
                )
            if args.temperature == 0:
                completions = completions * samples_per_prompt
            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
            if prompt_token_ids is None:
                prompt_token_ids = _fallback_prompt_token_ids(
                    tokenizer,
                    spec.prompt,
                    args.enable_thinking,
                )
            prompt_token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
            if not prompt_token_ids:
                raise RuntimeError(f"Target request {spec.global_index} has an empty prompt")
            for sample_index, completion in enumerate(completions):
                finish_reason = str(getattr(completion, "finish_reason", "") or "")
                raw_stop_reason = getattr(completion, "stop_reason", None)
                ordinary, boundary, boundary_source = split_target_tokens(
                    getattr(completion, "token_ids", ()),
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos_token_ids,
                    finish_reason=finish_reason,
                    stop_token_id=_resolve_stop_token_id(raw_stop_reason, tokenizer),
                )
                if len(prompt_token_ids) + len(ordinary) + 1 > args.max_model_len:
                    raise RuntimeError(
                        f"Target request {spec.global_index}, sample {sample_index} exceeds "
                        "--max-model-len after adding its positional boundary"
                    )
                trajectories.append(
                    TargetTrajectory(
                        dataset=spec.dataset,
                        dataset_index=spec.dataset_index,
                        global_index=spec.global_index,
                        prompt=spec.prompt,
                        prompt_token_ids=prompt_token_ids,
                        ordinary_token_ids=ordinary,
                        boundary_token_id=boundary,
                        target_text=str(getattr(completion, "text", "")),
                        finish_reason=finish_reason,
                        stop_reason=_json_stop_reason(raw_stop_reason),
                        boundary_source=boundary_source,
                        sample_index=sample_index,
                        trajectory_index=(
                            spec.dataset_index * samples_per_prompt + sample_index
                        ),
                    )
                )
        return trajectories, generation_seconds
    finally:
        _shutdown_vllm(llm)
        del llm
        gc.collect()


def aggregate_sequence_results(results: Sequence[DPSequenceResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate an empty DP result set")
    total_tokens = sum(result.generated_tokens for result in results)
    total_expected_rounds = math.fsum(result.expected_rounds for result in results)
    total_weighted_cost = math.fsum(result.weighted_cost_to_go for result in results)
    if total_expected_rounds <= 0 or total_weighted_cost <= 0:
        raise RuntimeError("EDR aggregate denominators must be positive")
    prompt_count = len(
        {(result.dataset, result.dataset_index) for result in results}
    )
    return {
        "prompts": prompt_count,
        "sequences": len(results),
        "total_generated_tokens": total_tokens,
        "total_expected_rounds": total_expected_rounds,
        "total_weighted_cost_to_go": total_weighted_cost,
        "round_start_mal": total_tokens / total_expected_rounds,
        "weighted_cost_mal": total_tokens / total_weighted_cost,
        "mean_generated_tokens": total_tokens / len(results),
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _serialize_dp_result(result: DPSequenceResult) -> dict[str, Any]:
    return {
        **asdict(result),
        "token_counting": DP_TOKEN_COUNTING,
        "round_start_mal": result.round_start_mal,
        "weighted_cost_mal": result.weighted_cost_mal,
    }


def _load_dp_results(
    path: Path, allowed_datasets: Sequence[str] = SUPPORTED_DATASETS,
) -> list[DPSequenceResult]:
    field_names = tuple(DPSequenceResult.__dataclass_fields__)
    results: list[DPSequenceResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                missing = [name for name in field_names if name not in record]
                if missing:
                    raise ValueError(f"missing fields {missing}")
                result = DPSequenceResult(
                    **{name: record[name] for name in field_names}
                )
                token_counting = record.get("token_counting")
                if token_counting is None:
                    # Records without token_counting exclude the sampled terminal
                    # token; add it so every row counts that token.
                    result = replace(result, generated_tokens=result.generated_tokens + 1)
                elif token_counting != DP_TOKEN_COUNTING:
                    raise ValueError(f"unknown token counting convention: {token_counting!r}")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Invalid DP result at {path}:{line_number}: {exc}"
                ) from exc
            if result.dataset not in allowed_datasets:
                raise RuntimeError(
                    f"Unexpected dataset {result.dataset!r} found in {path}"
                )
            results.append(result)
    if not results:
        raise RuntimeError(f"Existing DP result file is empty: {path}")
    aggregate_sequence_results(results)
    return results


def _sampling_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_model_cache_id": target_model_cache_id(args.target_model),
        "sampling_cache_key": sampling_cache_key(
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
        ),
        "draft_temperature": float(args.temperature),
    }


def resolve_target_cache_directory(
    args: argparse.Namespace,
    dataset: str,
) -> Path:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported target-cache dataset: {dataset!r}")
    cache_root = (
        Path(args.target_cache_root).expanduser()
        if args.target_cache_root
        else DEFAULT_OUTPUT_ROOT / TARGET_CACHE_DIRNAME
    )
    return (
        cache_root.resolve()
        / target_model_cache_id(args.target_model)
        / sampling_cache_key(
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
        )
        / dataset
    )


def _target_feature_path(cache_dir: Path, trajectory: TargetTrajectory) -> Path:
    return cache_dir / "features" / f"trajectory_{trajectory.trajectory_index:08d}.pt"


def _trajectory_from_json(record: dict[str, Any]) -> TargetTrajectory:
    converted = dict(record)
    converted["prompt_token_ids"] = tuple(int(value) for value in record["prompt_token_ids"])
    converted["ordinary_token_ids"] = tuple(
        int(value) for value in record["ordinary_token_ids"]
    )
    return TargetTrajectory(**converted)


def load_target_cache(
    cache_dir: Path,
    *,
    dataset: str,
) -> tuple[list[TargetTrajectory], dict[str, Any]]:
    if not cache_dir.is_dir() or cache_dir.name != dataset:
        raise RuntimeError(
            f"Target cache folder must be named for dataset {dataset!r}: {cache_dir}"
        )
    manifest_path = cache_dir / "manifest.json"
    trajectories_path = cache_dir / "target_outputs.jsonl"
    if not manifest_path.is_file() or not trajectories_path.is_file():
        raise RuntimeError(
            f"Target cache exists but is incomplete: {cache_dir}. Remove or move this exact "
            "cache directory before regenerating it."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Target cache is not complete: {cache_dir}")
    prompt_count = manifest.get("prompt_count")
    trajectory_count = manifest.get("trajectory_count")
    samples_per_prompt = manifest.get("samples_per_prompt")
    if (
        isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count <= 0
        or isinstance(trajectory_count, bool)
        or not isinstance(trajectory_count, int)
        or isinstance(samples_per_prompt, bool)
        or not isinstance(samples_per_prompt, int)
        or samples_per_prompt <= 0
        or trajectory_count != prompt_count * samples_per_prompt
    ):
        raise RuntimeError(f"Target cache manifest has invalid counts: {cache_dir}")
    for name in ("generation_seconds", "feature_extraction_seconds"):
        value = manifest.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise RuntimeError(f"Target cache manifest has invalid {name}: {cache_dir}")

    trajectories = []
    with trajectories_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            try:
                trajectories.append(_trajectory_from_json(json.loads(raw_line)))
            except Exception as exc:
                raise ValueError(
                    f"Invalid target cache trajectory at {trajectories_path}:{line_number}"
                ) from exc

    if len(trajectories) != trajectory_count:
        raise RuntimeError(
            f"Target cache has {len(trajectories)} trajectories, expected {trajectory_count}"
        )
    for trajectory_index, trajectory in enumerate(trajectories):
        dataset_index = trajectory_index // samples_per_prompt
        sample_index = trajectory_index % samples_per_prompt
        expected = (
            dataset,
            dataset_index,
            sample_index,
            trajectory_index,
        )
        actual = (
            trajectory.dataset,
            trajectory.dataset_index,
            trajectory.sample_index,
            trajectory.trajectory_index,
        )
        if actual != expected:
            raise RuntimeError(
                f"Target cache trajectory {trajectory_index} has invalid dataset ordering"
            )
        if not trajectory.prompt_token_ids:
            raise RuntimeError(f"Target cache trajectory {trajectory_index} has an empty prompt")
        if trajectory.ordinary_tokens and not _target_feature_path(
            cache_dir, trajectory
        ).is_file():
            raise RuntimeError(
                f"Target cache feature file is missing for trajectory {trajectory_index}"
            )
    return trajectories, manifest


@dataclass(frozen=True)
class CachedTargetFeatures:
    hidden_states: tuple[Any, ...]
    output_hidden: Any


@dataclass(frozen=True)
class DPScoreBatchPlan:
    """Length-bucketed trajectories that fit the configured scoring budgets."""

    indexed_trajectories: tuple[tuple[int, TargetTrajectory], ...]
    padded_sequence_tokens: int
    padded_proposal_blocks: int
    target_tokens: int


@dataclass(frozen=True)
class PreparedDPScoreBatch:
    """Pinned CPU tensors ready for one asynchronous device transfer."""

    indexed_trajectories: tuple[tuple[int, TargetTrajectory], ...]
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    loss_mask: Any
    hidden_states: tuple[Any, ...]
    normalized_target_hidden: Any


def load_cached_target_features(
    cache_dir: Path,
    trajectory: TargetTrajectory,
    *,
    expected_layers: int,
    expected_hidden_size: int,
) -> CachedTargetFeatures:
    import torch

    path = _target_feature_path(cache_dir, trajectory)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Target feature cache {path} is not a tensor dictionary")
    stacked = payload.get("hidden_states")
    output_hidden = payload.get("output_hidden")
    sequence_length = (
        len(trajectory.prompt_token_ids) + trajectory.ordinary_tokens + 1
    )
    expected_stacked_shape = (expected_layers, sequence_length, expected_hidden_size)
    expected_output_shape = (trajectory.ordinary_tokens, expected_hidden_size)
    if not isinstance(stacked, torch.Tensor) or tuple(stacked.shape) != expected_stacked_shape:
        raise ValueError(
            f"Target feature cache {path} hidden_states has shape "
            f"{getattr(stacked, 'shape', None)}, expected {expected_stacked_shape}"
        )
    if not isinstance(output_hidden, torch.Tensor) or tuple(
        output_hidden.shape
    ) != expected_output_shape:
        raise ValueError(
            f"Target feature cache {path} output_hidden has shape "
            f"{getattr(output_hidden, 'shape', None)}, expected {expected_output_shape}"
        )
    if not stacked.is_floating_point() or not output_hidden.is_floating_point():
        raise TypeError(f"Target feature cache {path} must contain floating-point tensors")
    return CachedTargetFeatures(
        hidden_states=tuple(stacked[index] for index in range(expected_layers)),
        output_hidden=output_hidden,
    )


def plan_dp_score_batches(
    trajectories: Sequence[TargetTrajectory],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
    max_proposal_blocks: int,
    max_target_tokens: int,
) -> list[DPScoreBatchPlan]:
    """Bucket nonempty trajectories by length under context and LM-head budgets."""
    if (
        max_batch_size < 1
        or max_batch_tokens < 1
        or max_proposal_blocks < 1
        or max_target_tokens < 1
    ):
        raise ValueError("DP scoring batch limits must be positive")

    indexed = [
        (index, trajectory)
        for index, trajectory in enumerate(trajectories)
        if trajectory.ordinary_tokens
    ]
    # The draft backbone pads proposal blocks to the longest horizon in a batch,
    # while its context cache pads to the longest full sequence.  Bucket first
    # by proposal length, then by context length, to keep both rectangles dense.
    # With the 2,048-block / 32-row defaults this creates 64-block bands.
    proposal_bucket_size = max(1, max_proposal_blocks // max_batch_size)
    indexed.sort(
        key=lambda item: (
            (item[1].ordinary_tokens - 1) // proposal_bucket_size,
            len(item[1].prompt_token_ids) + item[1].ordinary_tokens + 1,
            item[1].ordinary_tokens,
            item[0],
        )
    )

    plans: list[DPScoreBatchPlan] = []
    current: list[tuple[int, TargetTrajectory]] = []
    current_max_length = 0
    current_max_generated = 0
    current_target_tokens = 0

    def publish() -> None:
        nonlocal current, current_max_length, current_max_generated, current_target_tokens
        if not current:
            return
        plans.append(
            DPScoreBatchPlan(
                indexed_trajectories=tuple(current),
                padded_sequence_tokens=len(current) * current_max_length,
                padded_proposal_blocks=len(current) * current_max_generated,
                target_tokens=current_target_tokens,
            )
        )
        current = []
        current_max_length = 0
        current_max_generated = 0
        current_target_tokens = 0

    for indexed_trajectory in indexed:
        _, trajectory = indexed_trajectory
        sequence_length = (
            len(trajectory.prompt_token_ids) + trajectory.ordinary_tokens + 1
        )
        if sequence_length > max_batch_tokens:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_index} needs {sequence_length} padded "
                f"tokens, exceeding --score-max-batch-tokens={max_batch_tokens}"
            )
        if trajectory.ordinary_tokens > max_target_tokens:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_index} needs "
                f"{trajectory.ordinary_tokens} target rows, exceeding "
                f"--score-max-target-tokens={max_target_tokens}"
            )
        if trajectory.ordinary_tokens > max_proposal_blocks:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_index} needs "
                f"{trajectory.ordinary_tokens} proposal starts, exceeding "
                f"--edr-chunk-size={max_proposal_blocks}"
            )

        candidate_size = len(current) + 1
        candidate_max_length = max(current_max_length, sequence_length)
        candidate_max_generated = max(
            current_max_generated,
            trajectory.ordinary_tokens,
        )
        candidate_target_tokens = current_target_tokens + trajectory.ordinary_tokens
        if current and (
            candidate_size > max_batch_size
            or candidate_size * candidate_max_length > max_batch_tokens
            or candidate_size * candidate_max_generated > max_proposal_blocks
            or candidate_target_tokens > max_target_tokens
        ):
            publish()
            candidate_size = 1
            candidate_max_length = sequence_length
            candidate_max_generated = trajectory.ordinary_tokens
            candidate_target_tokens = trajectory.ordinary_tokens

        current.append(indexed_trajectory)
        current_max_length = candidate_max_length
        current_max_generated = candidate_max_generated
        current_target_tokens = candidate_target_tokens

    publish()
    # Largest pinned buffers run first. With one batch prefetched, the host
    # allocator can then recycle the two largest blocks for every smaller batch
    # instead of retaining a ladder of progressively larger pinned allocations.
    plans.sort(
        key=lambda plan: (plan.padded_sequence_tokens, plan.target_tokens),
        reverse=True,
    )
    return plans


def prepare_dp_score_batch(
    plan: DPScoreBatchPlan,
    *,
    cache_directories: Mapping[str, Path],
    expected_layers: int,
    expected_hidden_size: int,
    pin_memory: bool,
) -> PreparedDPScoreBatch:
    """Load mmap-backed feature files into one dense, optionally pinned CPU batch."""
    import torch

    if not plan.indexed_trajectories:
        raise ValueError("A DP score batch cannot be empty")
    batch_size = len(plan.indexed_trajectories)
    sequence_lengths = [
        len(trajectory.prompt_token_ids) + trajectory.ordinary_tokens + 1
        for _, trajectory in plan.indexed_trajectories
    ]
    sequence_length = max(sequence_lengths)
    tensor_options = {"device": "cpu", "pin_memory": bool(pin_memory)}
    input_ids = torch.zeros(
        (batch_size, sequence_length), dtype=torch.long, **tensor_options
    )
    attention_mask = torch.zeros(
        (batch_size, sequence_length), dtype=torch.bool, **tensor_options
    )
    position_ids = torch.empty(
        (batch_size, sequence_length), dtype=torch.long, **tensor_options
    )
    position_ids.copy_(
        torch.arange(sequence_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    loss_mask = torch.zeros(
        (batch_size, sequence_length), dtype=torch.float32, **tensor_options
    )
    stacked_hidden = torch.empty(
        (expected_layers, batch_size, sequence_length, expected_hidden_size),
        dtype=torch.bfloat16,
        **tensor_options,
    )
    normalized_target_hidden = torch.zeros(
        (batch_size, sequence_length, expected_hidden_size),
        dtype=torch.bfloat16,
        **tensor_options,
    )

    for row_index, (_, trajectory) in enumerate(plan.indexed_trajectories):
        cache_dir = cache_directories[trajectory.dataset]
        features = load_cached_target_features(
            cache_dir,
            trajectory,
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden_size,
        )
        full_token_ids = (
            trajectory.prompt_token_ids
            + trajectory.ordinary_token_ids
            + (trajectory.boundary_token_id,)
        )
        row_length = len(full_token_ids)
        output_start = len(trajectory.prompt_token_ids)
        output_end = output_start + trajectory.ordinary_tokens
        input_ids[row_index, :row_length].copy_(torch.tensor(full_token_ids))
        attention_mask[row_index, :row_length] = True
        loss_mask[row_index, output_start : output_end + 1] = 1.0
        for layer_index, hidden_state in enumerate(features.hidden_states):
            stacked_hidden[layer_index, row_index, :row_length].copy_(hidden_state)
        if row_length < sequence_length:
            stacked_hidden[:, row_index, row_length:].zero_()
        normalized_target_hidden[
            row_index,
            output_start - 1 : output_end - 1,
        ].copy_(features.output_hidden)

    return PreparedDPScoreBatch(
        indexed_trajectories=plan.indexed_trajectories,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        loss_mask=loss_mask,
        hidden_states=tuple(stacked_hidden[layer] for layer in range(expected_layers)),
        normalized_target_hidden=normalized_target_hidden,
    )


def iter_prepared_dp_score_batches(
    plans: Sequence[DPScoreBatchPlan],
    *,
    cache_directories: Mapping[str, Path],
    expected_layers: int,
    expected_hidden_size: int,
    pin_memory: bool,
    prefetch: bool,
) -> Iterator[PreparedDPScoreBatch]:
    """Prepare batches serially or with one CPU batch loaded ahead of the GPU."""

    def prepare(plan: DPScoreBatchPlan) -> PreparedDPScoreBatch:
        return prepare_dp_score_batch(
            plan,
            cache_directories=cache_directories,
            expected_layers=expected_layers,
            expected_hidden_size=expected_hidden_size,
            pin_memory=pin_memory,
        )

    if not prefetch:
        for plan in plans:
            yield prepare(plan)
        return
    if not plans:
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-dp-cache") as executor:
        pending = executor.submit(prepare, plans[0])
        for next_plan in plans[1:]:
            prepared = pending.result()
            pending = executor.submit(prepare, next_plan)
            yield prepared
        yield pending.result()


class TargetFeatureCacheWriter:
    """Load the full target only while materializing reusable trajectory features."""

    def __init__(self, args: argparse.Namespace, target_config: Any, layer_ids: Sequence[int]):
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        self.torch = torch
        self.device = torch.device("cuda:0")
        # vLLM has already shut down at this point. Collect its Python objects
        # and return cached blocks before loading the full Hugging Face target.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Loading Hugging Face target to populate DP cache: {args.target_model}", flush=True)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            config=target_config,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation=args.target_attn_implementation,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        self.target_runner = HFTargetRunner(self.target_model, layer_ids)
        lm_head = self.target_model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("Target model does not expose an output embedding module")
        self._lm_head_input = None

        def capture_lm_head_input(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Target LM head did not receive a hidden-state tensor")
            self._lm_head_input = inputs[0]

        self._lm_head_handle = lm_head.register_forward_pre_hook(capture_lm_head_input)

    def write(self, trajectory: TargetTrajectory, path: Path, max_model_len: int) -> None:
        torch = self.torch
        if trajectory.ordinary_tokens == 0:
            return
        full_token_ids = (
            trajectory.prompt_token_ids
            + trajectory.ordinary_token_ids
            + (trajectory.boundary_token_id,)
        )
        if len(full_token_ids) > max_model_len:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_index} exceeds max_model_len={max_model_len}"
            )
        input_ids = torch.tensor(full_token_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1], device=self.device, dtype=torch.long
        ).unsqueeze(0)
        logits_to_keep = trajectory.ordinary_tokens + 2
        self._lm_head_input = None
        with torch.inference_mode():
            target_output, hidden_states = self.target_runner.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=logits_to_keep,
                return_dict=True,
            )
        if target_output.logits.shape[1] != logits_to_keep:
            raise RuntimeError("Target logits_to_keep returned an unexpected trajectory slice")
        output_hidden = self._lm_head_input
        if output_hidden is None or output_hidden.ndim != 3:
            raise RuntimeError("Target LM-head input hidden states were not captured")
        if output_hidden.shape[1] != target_output.logits.shape[1]:
            if output_hidden.shape[1] < target_output.logits.shape[1]:
                raise RuntimeError("Captured target output hidden state is shorter than its logits")
            output_hidden = output_hidden[:, -target_output.logits.shape[1] :, :]

        cached_hidden = torch.stack(
            [state[0].detach().to(device="cpu", dtype=torch.bfloat16) for state in hidden_states]
        ).contiguous()
        cached_output_hidden = (
            output_hidden[0, : trajectory.ordinary_tokens]
            .detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden_states": cached_hidden,
                "output_hidden": cached_output_hidden,
            },
            path,
        )
        self._lm_head_input = None

    def close(self) -> None:
        if self._lm_head_handle is not None:
            self._lm_head_handle.remove()
            self._lm_head_handle = None
        if self.target_runner is not None:
            self.target_runner.close()
            self.target_runner = None
        self.target_model = None
        gc.collect()
        self.torch.cuda.empty_cache()


def populate_target_caches(
    args: argparse.Namespace,
    *,
    requests: Sequence[TargetCacheRequest],
    tokenizer: Any,
    eos_token_ids: set[int],
    target_config: Any,
    draft_config: dict[str, Any],
) -> tuple[
    dict[str, tuple[list[TargetTrajectory], dict[str, Any]]],
    float,
    float,
]:
    """Populate every absent dataset shard in one target-model lifecycle."""
    if not requests:
        return {}, 0.0, 0.0
    datasets = [request.dataset for request in requests]
    if len(datasets) != len(set(datasets)):
        raise ValueError("Target cache population requests must have unique datasets")

    temporary_dirs: dict[str, Path] = {}
    writer: TargetFeatureCacheWriter | None = None
    try:
        for request in requests:
            request.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary_dirs[request.dataset] = Path(
                tempfile.mkdtemp(
                    prefix=f".{request.dataset}.tmp-",
                    dir=request.cache_dir.parent,
                )
            )

        all_prompt_specs = [
            prompt_spec
            for request in requests
            for prompt_spec in request.prompt_specs
        ]
        all_trajectories, generation_seconds = generate_target_trajectories(
            args,
            all_prompt_specs,
            tokenizer,
            eos_token_ids,
        )
        trajectories_by_dataset = {request.dataset: [] for request in requests}
        for trajectory in all_trajectories:
            try:
                trajectories_by_dataset[trajectory.dataset].append(trajectory)
            except KeyError as exc:
                raise RuntimeError(
                    f"Generated an unexpected dataset trajectory: {trajectory.dataset!r}"
                ) from exc
        for request in requests:
            dataset_trajectories = trajectories_by_dataset[request.dataset]
            expected_count = len(request.prompt_specs) * target_samples_per_prompt(
                len(request.prompt_specs)
            )
            if len(dataset_trajectories) != expected_count:
                raise RuntimeError(
                    f"Generated {len(dataset_trajectories)} {request.dataset} trajectories, "
                    f"expected {expected_count}"
                )
            _atomic_write_jsonl(
                temporary_dirs[request.dataset] / "target_outputs.jsonl",
                [asdict(trajectory) for trajectory in dataset_trajectories],
            )

        feature_started = time.perf_counter()
        nonempty = [
            trajectory for trajectory in all_trajectories if trajectory.ordinary_tokens
        ]
        if nonempty:
            writer = TargetFeatureCacheWriter(
                args,
                target_config,
                draft_config["target_layer_ids"],
            )
            from tqdm.auto import tqdm

            iterator = tqdm(
                nonempty,
                desc="Target feature cache",
                unit="sequence",
                disable=args.disable_progress,
            )
            for trajectory in iterator:
                writer.write(
                    trajectory,
                    _target_feature_path(
                        temporary_dirs[trajectory.dataset],
                        trajectory,
                    ),
                    args.max_model_len,
                )
        feature_extraction_seconds = time.perf_counter() - feature_started
        if writer is not None:
            writer.close()
            writer = None

        populated: dict[str, tuple[list[TargetTrajectory], dict[str, Any]]] = {}
        population_datasets = [request.dataset for request in requests]
        for request in requests:
            dataset_trajectories = trajectories_by_dataset[request.dataset]
            manifest = {
                "format_version": TARGET_CACHE_FORMAT_VERSION,
                "status": "complete",
                "dataset": request.dataset,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prompt_count": len(request.prompt_specs),
                "trajectory_count": len(dataset_trajectories),
                "samples_per_prompt": target_samples_per_prompt(len(request.prompt_specs)),
                "target_model": args.target_model,
                "sampling_identity": _sampling_identity(args),
                "target_layer_ids": [
                    int(index) for index in draft_config["target_layer_ids"]
                ],
                "sampling": {
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "seed": int(args.seed),
                    "enable_thinking": bool(args.enable_thinking),
                    "max_new_tokens": int(args.max_new_tokens),
                    "max_model_len": int(args.max_model_len),
                    "eos_token_ids": sorted(int(token_id) for token_id in eos_token_ids),
                },
                "population_batch_datasets": population_datasets,
                "generation_seconds": generation_seconds,
                "feature_extraction_seconds": feature_extraction_seconds,
            }
            temporary_dir = temporary_dirs[request.dataset]
            _atomic_write_json(temporary_dir / "manifest.json", manifest)
            try:
                temporary_dir.replace(request.cache_dir)
            except OSError:
                # Another evaluator may have published this immutable dataset
                # shard while the common target pass was running.
                if not request.cache_dir.exists():
                    raise
                populated[request.dataset] = load_target_cache(
                    request.cache_dir,
                    dataset=request.dataset,
                )
            else:
                populated[request.dataset] = (dataset_trajectories, manifest)
        return populated, generation_seconds, feature_extraction_seconds
    finally:
        if writer is not None:
            writer.close()
        for temporary_dir in temporary_dirs.values():
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)


class OfflineEDRScorer:
    """Own the draft, lightweight target head, and cached scoring path."""

    def __init__(
        self,
        args: argparse.Namespace,
        checkpoint: ResolvedCheckpoint,
        edr_stop_token_ids: Sequence[int],
    ) -> None:
        import torch
        import transformers

        from angelspec.models.dfly import DFlyModel
        from angelspec.models.dspark import DSparkModel
        from angelspec.models.target.target_utils import TargetLMHead

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"Expected exactly one visible GPU, but PyTorch sees {torch.cuda.device_count()}. "
                "Set CUDA_VISIBLE_DEVICES to a single GPU."
            )
        self.torch = torch
        self.device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.edr_stop_token_ids = tuple(sorted(set(int(i) for i in edr_stop_token_ids)))
        if not self.edr_stop_token_ids:
            raise ValueError("The target model does not define an EOS/stopping token ID")
        print(f"Loading target LM head for cached DP scoring: {args.target_model}", flush=True)
        self.target_lm_head = TargetLMHead.from_pretrained(
            model_path=args.target_model,
            device="cuda",
            dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
        self.lm_head_weight = self.target_lm_head.lm_head.weight

        print(f"Loading draft for DP scoring: {checkpoint.source_path}", flush=True)
        self.draft_model, self.draft_config = load_draft_model(
            checkpoint, self.device, args.draft_config
        )
        is_dspark = self.draft_config.get("model_type") == "dspark"
        self.wrapper = (DSparkModel if is_dspark else DFlyModel)(
            draft_model=self.draft_model,
            # block_size counts learned proposals (7 for both drafters).
            block_size=NUM_DFLY_PROPOSALS,
            loss_objective="edr",
            edr_chunk_size=args.edr_chunk_size,
            edr_vocab_chunk_size=args.edr_vocab_chunk_size,
            edr_dp_workers=args.score_dp_workers,
            query_includes_input_anchor=not is_dspark,
            edr_stop_token_ids=self.edr_stop_token_ids,
            edr_temperature=args.temperature,
            edr_top_k=args.top_k,
            edr_top_p=args.top_p,
        )
        self.wrapper.eval()
        if self.lm_head_weight.device != self.device:
            raise RuntimeError(
                f"Target LM head is on {self.lm_head_weight.device}, expected {self.device}"
            )
        self.draft_temperature = float(args.temperature)
        self.target_top_k = int(args.top_k)
        self.target_top_p = float(args.top_p)
        self.max_model_len = int(args.max_model_len)
        self.expected_target_layers = len(self.draft_config["target_layer_ids"])
        self.expected_hidden_size = int(self.draft_config["target_hidden_size"])
        self.versions = {
            "gpu": torch.cuda.get_device_name(self.device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }

    def score(
        self,
        trajectory: TargetTrajectory,
        cached_features: CachedTargetFeatures | None,
    ) -> DPSequenceResult:
        torch = self.torch
        ordinary_length = trajectory.ordinary_tokens
        if ordinary_length == 0:
            return DPSequenceResult(
                dataset=trajectory.dataset,
                dataset_index=trajectory.dataset_index,
                global_index=trajectory.global_index,
                prompt_tokens=len(trajectory.prompt_token_ids),
                generated_tokens=trajectory.generated_tokens,
                expected_rounds=1.0,
                weighted_cost_to_go=1.0,
                sample_index=trajectory.sample_index,
                trajectory_index=trajectory.trajectory_index,
            )
        if cached_features is None:
            raise ValueError("A nonempty target trajectory requires cached target features")

        from angelspec.models.ops.edr import (
            exact_edr_dynamic_program,
            prepare_edr_target_distribution,
        )
        from angelspec.models.ops.flex_attention import isolated_flex_attention_fallback

        full_token_ids = (
            trajectory.prompt_token_ids
            + trajectory.ordinary_token_ids
            + (trajectory.boundary_token_id,)
        )
        if len(full_token_ids) > self.max_model_len:
            raise ValueError(
                f"Trajectory {trajectory.global_index} has {len(full_token_ids)} tokens, "
                f"exceeding max_model_len={self.max_model_len}"
            )
        output_start = len(trajectory.prompt_token_ids)
        learned_target_count = ordinary_length
        input_ids = torch.tensor(
            full_token_ids,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1],
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        loss_mask[:, output_start:] = 1.0

        with torch.inference_mode():
            hidden_states = [
                state.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
                for state in cached_features.hidden_states
            ]
            output_hidden = cached_features.output_hidden.to(
                device=self.device,
                dtype=self.lm_head_weight.dtype,
            )
            target_logits = self.target_lm_head.lm_head(output_hidden)
            target_distribution = prepare_edr_target_distribution(
                target_logits,
                self.wrapper.edr_vocab_chunk_size,
                stopping_token_ids=getattr(self, "edr_stop_token_ids", ()),
                temperature=self.draft_temperature,
                top_k=self.target_top_k,
                top_p=self.target_top_p,
            )
            context_cache = self.wrapper._prepare_edr_context_cache(
                input_ids,
                list(hidden_states),
                position_ids,
            )
            prefixes = torch.arange(
                learned_target_count,
                device=self.device,
                dtype=torch.long,
            )
            anchors = prefixes + output_start - 1
            cost_chunks = []
            acceptance_chunks = []
            with isolated_flex_attention_fallback():
                for chunk_start in range(
                    0,
                    learned_target_count,
                    self.wrapper.edr_chunk_size,
                ):
                    chunk_slice = slice(
                        chunk_start,
                        chunk_start + self.wrapper.edr_chunk_size,
                    )
                    chunk_prefixes = prefixes[chunk_slice]
                    chunk_costs, chunk_acceptance, chunk_valid = (
                        self.wrapper._edr_chunk_statistics(
                            input_ids=input_ids,
                            hidden_states_list=list(hidden_states),
                            loss_mask=loss_mask,
                            lm_head_weight=self.lm_head_weight,
                            target_distribution=target_distribution,
                            target_probability_start=output_start - 1,
                            anchors=anchors[chunk_slice],
                            prefixes=chunk_prefixes,
                            horizon_lengths=torch.full_like(
                                chunk_prefixes,
                                ordinary_length,
                            ),
                            target_distribution_offsets=0,
                            target_probability_counts=learned_target_count,
                            attention_mask=attention_mask,
                            ctx_doc_ids=None,
                            base_position_ids=position_ids,
                            draft_context_cache=context_cache,
                            draft_temperature=self.draft_temperature,
                        )
                    )
                    cost_chunks.append(
                        torch.where(
                            chunk_valid,
                            chunk_costs,
                            torch.zeros_like(chunk_costs),
                        )
                    )
                    acceptance_chunks.append(
                        torch.where(
                            chunk_valid,
                            chunk_acceptance,
                            torch.zeros_like(chunk_acceptance),
                        )
                    )

            # Transfer only compact [L, 7] statistics. Passing CPU tensors to
            # exact_edr_dynamic_program keeps all NumPy recurrence outputs on CPU.
            costs = torch.cat(cost_chunks, dim=0).float().cpu()
            acceptance = torch.cat(acceptance_chunks, dim=0).float().cpu()

        dynamic_program = exact_edr_dynamic_program(
            costs,
            acceptance,
            num_proposals=NUM_DFLY_PROPOSALS,
        )
        expected_rounds = float(dynamic_program.round_start_probabilities.sum().item())
        weighted_cost_to_go = float(dynamic_program.expected_passes.item())
        if expected_rounds < 1.0 or weighted_cost_to_go < 1.0:
            raise RuntimeError("EDR pathwise round estimates must be at least one")
        return DPSequenceResult(
            dataset=trajectory.dataset,
            dataset_index=trajectory.dataset_index,
            global_index=trajectory.global_index,
            prompt_tokens=len(trajectory.prompt_token_ids),
            generated_tokens=trajectory.generated_tokens,
            expected_rounds=expected_rounds,
            weighted_cost_to_go=weighted_cost_to_go,
            sample_index=trajectory.sample_index,
            trajectory_index=trajectory.trajectory_index,
        )

    def score_prepared_batch(
        self,
        prepared: PreparedDPScoreBatch,
    ) -> list[tuple[int, DPSequenceResult]]:
        """Score independent cached trajectories in one bounded GPU batch."""
        torch = self.torch
        from angelspec.models.ops.edr import EDRHorizon, exact_edr_dynamic_programs
        from angelspec.models.ops.flex_attention import isolated_flex_attention_fallback

        indexed_trajectories = prepared.indexed_trajectories
        if not indexed_trajectories:
            return []
        if any(not trajectory.ordinary_tokens for _, trajectory in indexed_trajectories):
            raise ValueError("Prepared DP score batches must contain only nonempty trajectories")
        padded_proposal_blocks = len(indexed_trajectories) * max(
            trajectory.ordinary_tokens for _, trajectory in indexed_trajectories
        )
        if padded_proposal_blocks > self.wrapper.edr_chunk_size:
            raise ValueError(
                "Prepared DP batch would split each row across repeated context scans "
                f"({padded_proposal_blocks} > edr_chunk_size={self.wrapper.edr_chunk_size})"
            )
        if prepared.input_ids.shape[1] > self.max_model_len:
            raise ValueError(
                f"Prepared DP batch has {prepared.input_ids.shape[1]} tokens, exceeding "
                f"max_model_len={self.max_model_len}"
            )

        with torch.inference_mode():
            input_ids = prepared.input_ids.to(self.device, non_blocking=True)
            attention_mask = prepared.attention_mask.to(self.device, non_blocking=True)
            position_ids = prepared.position_ids.to(self.device, non_blocking=True)
            loss_mask = prepared.loss_mask.to(self.device, non_blocking=True)
            hidden_states = [
                state.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                for state in prepared.hidden_states
            ]
            normalized_target_hidden = prepared.normalized_target_hidden.to(
                device=self.device,
                dtype=self.lm_head_weight.dtype,
                non_blocking=True,
            )
            horizons_by_row = []
            for row_index, (_, trajectory) in enumerate(indexed_trajectories):
                output_start = len(trajectory.prompt_token_ids)
                horizons_by_row.append(
                    [
                        EDRHorizon(
                            batch_index=row_index,
                            start=output_start,
                            boundary=output_start + trajectory.ordinary_tokens,
                        )
                    ]
                )

            context_cache = self.wrapper._prepare_edr_context_cache(
                input_ids,
                hidden_states,
                position_ids,
            )
            with isolated_flex_attention_fallback():
                statistics_by_row, target_distribution = (
                    self.wrapper._edr_all_row_statistics(
                        input_ids=input_ids,
                        hidden_states_list=hidden_states,
                        loss_mask=loss_mask,
                        lm_head_weight=self.lm_head_weight,
                        normalized_target_hidden=normalized_target_hidden,
                        horizons_by_row=horizons_by_row,
                        attention_mask=attention_mask,
                        ctx_doc_ids=None,
                        base_position_ids=position_ids,
                        draft_context_cache=context_cache,
                        draft_temperature=self.draft_temperature,
                    )
                )
            del target_distribution
            if any(len(row_statistics) != 1 for row_statistics in statistics_by_row):
                raise RuntimeError("Each cached trajectory must produce exactly one EDR horizon")
            flat_statistics = [row_statistics[0] for row_statistics in statistics_by_row]
            dynamic_programs = exact_edr_dynamic_programs(
                [
                    (statistics.costs, statistics.acceptance)
                    for statistics in flat_statistics
                ],
                num_proposals=NUM_DFLY_PROPOSALS,
                max_workers=self.wrapper.edr_dp_workers,
                return_on_cpu=True,
            )
            # One synchronization for all metrics in the batch, rather than two
            # scalar synchronizations per trajectory.
            metric_rows = torch.stack(
                [
                    torch.stack(
                        (
                            dynamic_program.round_start_probabilities.sum(),
                            dynamic_program.expected_passes,
                        )
                    )
                    for dynamic_program in dynamic_programs
                ]
            ).cpu()

        results: list[tuple[int, DPSequenceResult]] = []
        for (original_index, trajectory), metric_row in zip(
            indexed_trajectories,
            metric_rows.tolist(),
            strict=True,
        ):
            expected_rounds, weighted_cost_to_go = (float(value) for value in metric_row)
            if (
                not math.isfinite(expected_rounds)
                or not math.isfinite(weighted_cost_to_go)
                or expected_rounds < 1.0
                or weighted_cost_to_go < 1.0
            ):
                raise RuntimeError("EDR pathwise round estimates must be finite and at least one")
            results.append(
                (
                    original_index,
                    DPSequenceResult(
                        dataset=trajectory.dataset,
                        dataset_index=trajectory.dataset_index,
                        global_index=trajectory.global_index,
                        prompt_tokens=len(trajectory.prompt_token_ids),
                        generated_tokens=trajectory.generated_tokens,
                        expected_rounds=expected_rounds,
                        weighted_cost_to_go=weighted_cost_to_go,
                        sample_index=trajectory.sample_index,
                        trajectory_index=trajectory.trajectory_index,
                    ),
                )
            )
        return results

    def close(self) -> None:
        self.wrapper = None
        self.draft_model = None
        self.target_lm_head = None
        gc.collect()
        self.torch.cuda.empty_cache()


def build_dataset_summaries(
    results: Sequence[DPSequenceResult],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    summaries = []
    for dataset in datasets:
        dataset_results = [result for result in results if result.dataset == dataset]
        summary = aggregate_sequence_results(dataset_results)
        summary["name"] = dataset
        summaries.append(summary)
    return summaries


def render_report(payload: dict[str, Any]) -> str:
    run = payload["run"]
    target_cache_rows = [
        "| Dataset | Status this run | Prompts | Samples/prompt | Trajectories | Cache directory |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        *[
            f"| {cache['dataset']} | {cache['status']} | {cache['prompts']} | "
            f"{cache['samples_per_prompt']} | {cache['trajectories']} | `{cache['cache_dir']}` |"
            for cache in run["target_caches"]
        ],
    ]
    rows = [
        "| Dataset | Prompts | Sequences | Tokens | Σ expected rounds | Round-start MAL | "
        "Σ weighted cost | Weighted-cost MAL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in payload["datasets"]:
        rows.append(
            f"| {dataset['name']} | {dataset['prompts']} | "
            f"{dataset['sequences']} | "
            f"{dataset['total_generated_tokens']} | "
            f"{dataset['total_expected_rounds']:.4f} | "
            f"{dataset['round_start_mal']:.4f} | "
            f"{dataset['total_weighted_cost_to_go']:.4f} | "
            f"{dataset['weighted_cost_mal']:.4f} |"
        )
    aggregate = payload["aggregate"]
    rows.append(
        f"| **ALL** | **{aggregate['prompts']}** | "
        f"**{aggregate['sequences']}** | "
        f"**{aggregate['total_generated_tokens']}** | "
        f"**{aggregate['total_expected_rounds']:.4f}** | "
        f"**{aggregate['round_start_mal']:.4f}** | "
        f"**{aggregate['total_weighted_cost_to_go']:.4f}** | "
        f"**{aggregate['weighted_cost_mal']:.4f}** |"
    )
    return "\n".join(
        [
            f"# Offline EDR DP evaluation: {run['checkpoint_name']}",
            "",
            "> Equally many fixed target trajectories per prompt are loaded from independent "
            "dataset cache shards, then scored with exact draft statistics and "
            "AngelSpec's NumPy recurrence.",
            "",
            "## Configuration",
            "",
            f"- Draft: `{run['draft_checkpoint_resolved']}`",
            f"- Target: `{run['target_model']}`",
            f"- GPU: {run['gpu']}",
            f"- Sampling: target temperature {run['target_temperature']}, "
            f"top-k {run['top_k']}, top-p {run['top_p']}; draft temperature "
            f"{run['draft_temperature']} "
            f"({'greedy/one-hot' if run['draft_temperature'] == 0 else 'full softmax'}), "
            "no draft top-k/top-p filtering",
            "- At temperature 0, repeated trajectories for a prompt are identical "
            "deterministic copies, not independent stochastic samples.",
            f"- Thinking: {'enabled' if run['enable_thinking'] else 'disabled'}",
            f"- Target cache root: `{run['target_cache_root']}`",
            f"- Datasets scored this run: "
            f"{', '.join(run.get('evaluated_datasets_this_run', [])) or 'none'}",
            f"- Completed datasets skipped this run: "
            f"{', '.join(run.get('skipped_completed_datasets', [])) or 'none'}",
            f"- Datasets included in this report: "
            f"{', '.join(run.get('report_datasets', []))}",
            f"- Dataset caches this run: "
            f"{len(run['target_cache_datasets_reused'])} reused / "
            f"{len(run['target_cache_datasets_populated'])} populated",
            f"- Minimum target sequences per newly populated dataset: "
            f"{run['min_target_sequences_per_dataset']} (equal samples per prompt)",
            f"- EDR chunks: {run['edr_chunk_size']} starts / "
            f"{run['edr_vocab_chunk_size']} vocabulary entries",
            f"- Cached scoring batches: up to {run['score_batch_size']} trajectories / "
            f"{run['score_max_batch_tokens']} padded context tokens / "
            f"{run['score_max_target_tokens']} target rows",
            f"- Cached scoring pipeline: {run['score_batches']} length-bucketed batches, "
            f"{run['score_dp_workers']} DP workers, "
            f"prefetch {'enabled' if run['score_prefetch'] else 'disabled'}",
            f"- Effective scoring batch: at most "
            f"{run['score_effective_max_batch_size']} rows; context/proposal padding "
            f"{100.0 * run['score_context_padding_fraction']:.1f}% / "
            f"{100.0 * run['score_proposal_padding_fraction']:.1f}%",
            f"- vLLM generation batch limits: {run['max_num_seqs']} sequences / "
            f"{run['max_num_batched_tokens']} tokens",
            f"- Target generation this run: {run['generation_seconds']:.2f} seconds",
            f"- Target feature extraction this run: "
            f"{run['feature_extraction_seconds']:.2f} seconds",
            f"- Offline DP scoring: {run['scoring_seconds']:.2f} seconds",
            "- D-Cut: not used; all seven draft proposal states are evaluated",
            "- Round semantics: seven proposals, one target "
            "bonus, and at most eight tokens advanced per round",
            "",
            "## Target cache shards",
            "",
            *target_cache_rows,
            "",
            "## MAL",
            "",
            *rows,
            "",
            f"Global round-start MAL: **{aggregate['round_start_mal']:.4f}**",
            "",
            f"Global weighted-cost MAL: **{aggregate['weighted_cost_mal']:.4f}**",
            "",
            "## Definitions",
            "",
            "For a trajectory with `L` ordinary generated tokens, the terminal EOS/length "
            "lookahead token is positional boundary context and is excluded from `L` "
            "for the DP recurrence only. The MAL numerator counts all `L + 1` sampled "
            "output tokens, including that EOS, stopping token, or length boundary. "
            "An immediate EOS therefore contributes one token and one round.",
            "",
            "- Round-start expected rounds: `Σ_{n=0}^{L} ω_{n,n+1}`. The initial "
            "term is already `ω_{0,1}=1`, so no second leading one is added.",
            "- Weighted-cost target passes: `1 + U_{0,1}` from the EDR Bellman recurrence.",
            "- Each MAL is a corpus ratio: total generated tokens divided by the "
            "corresponding denominator summed over all trajectories. It is not a mean "
            "of per-sequence MAL values.",
            "",
        ]
    )


def _evaluation_identity(
    args: argparse.Namespace, checkpoint: ResolvedCheckpoint,
) -> dict[str, Any]:
    """Fingerprint inputs, not batch sizes or other performance-only settings.

    Checkpoint sizes/mtimes avoid rereading multi-GB weights just to skip scoring.
    Completed results must use a new output name if those files are replaced.
    """
    source = checkpoint.source_path.resolve()
    files = [source] if source.is_file() else sorted(
        path for path in source.rglob("*")
        if path.is_file() and (
            path.suffix in {".safetensors", ".bin", ".pt", ".distcp"}
            or path.name == ".metadata" or path.name.endswith(".json")
        )
    )
    if not files:
        raise RuntimeError(f"No checkpoint files to fingerprint in {source}")
    weights = []
    for path in files:
        stat = path.stat()
        weights.append({
            "name": path.name if source.is_file() else str(path.relative_to(source)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    config = json.loads(Path(args.draft_config).read_text(encoding="utf-8"))
    return {
        "checkpoint": str(source),
        "checkpoint_kind": checkpoint.kind,
        "checkpoint_files": weights,
        "draft_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sampling": _sampling_identity(args),
        "enable_thinking": bool(args.enable_thinking),
        "num_speculative_tokens": args.num_speculative_tokens,
        "token_counting": DP_TOKEN_COUNTING,
        "scoring_version": DP_SCORING_VERSION,
    }


def _migrate_completed_results(
    run_dir: Path, args: argparse.Namespace, store: DPResultStore,
    identity: dict[str, Any], datasets: Sequence[str],
) -> None:
    """Import report-only results with a matching scoring identity."""
    candidates = [
        (run_dir / "metrics.json", run_dir / "dp_results.jsonl"),
    ]
    for metrics_path, results_path in candidates:
        if not metrics_path.exists() or not results_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run = metrics.get("run", {})
        if run.get("status") != "complete":
            continue
        previous_results = _load_dp_results(results_path, SUPPORTED_DATASETS)
        names = [name for name in datasets if any(r.dataset == name for r in previous_results)]
        names = [name for name in names if store.load(name) is None]
        if not names:
            continue
        mismatch = (
            run.get("draft_checkpoint_resolved") != identity["checkpoint"]
            or run.get("checkpoint_kind") != identity["checkpoint_kind"]
            or run.get("enable_thinking") != identity["enable_thinking"]
        )
        recorded_identity = run.get("evaluation_identity")
        if recorded_identity is not None:
            mismatch = mismatch or recorded_identity != identity
        else:
            # Without a recorded identity the scoring version is unknown, so the
            # report's DP denominators are rejected; its target cache stays usable.
            mismatch = True
        if mismatch:
            raise RuntimeError(
                f"Existing DP results at {results_path} have a different or unknown "
                "checkpoint/sampling/scoring configuration. Use a different --checkpoint-name "
                "or --output-root with the same --target-cache-root to rescore; "
                "existing results have not been overwritten."
            )
        # Report-only results have no per-dataset cache snapshot. Import them
        # only when the weights and the sampled cache predate their completion time.
        try:
            completed_ns = int(datetime.fromisoformat(run["completed_at"]).timestamp() * 1e9)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot verify completion time of {metrics_path}") from exc
        if any(row["mtime_ns"] > completed_ns for row in identity["checkpoint_files"]):
            raise RuntimeError(
                "Checkpoint files changed after existing DP results were written. "
                "Use a different --checkpoint-name or --output-root."
            )
        cache_rows = {row["dataset"]: row for row in run.get("target_caches", [])}
        pending = []
        for name in names:
            cache_dir = resolve_target_cache_directory(args, name)
            cache_info = cache_rows.get(name)
            if cache_info is None or Path(cache_info["cache_dir"]).resolve() != cache_dir:
                raise RuntimeError(f"Existing {name} results refer to a different/unknown target cache")
            for filename in ("manifest.json", "target_outputs.jsonl"):
                if (cache_dir / filename).stat().st_mtime_ns > completed_ns:
                    raise RuntimeError(
                        f"Target cache for {name} changed after existing results were written; "
                        "use a different --checkpoint-name or --output-root."
                    )
            records = [_serialize_dp_result(r) for r in previous_results if r.dataset == name]
            validate_records(cache_dir, name, records)
            cache_info = dict(cache_info)
            cache_info.setdefault("samples_per_prompt", len(records) // cache_info["prompts"])
            pending.append((name, records, cache_info))
        for name, records, cache_info in pending:
            store.save(name, records, cache_info, run)
            print(f"Imported completed {name} DP results from {results_path}", flush=True)


def _publish_incremental_results(
    args: argparse.Namespace, run_dir: Path, entries: Mapping[str, dict[str, Any]],
    payload: dict[str, Any], pending: Sequence[str], skipped: Sequence[str],
) -> dict[str, Any]:
    report_datasets = list(entries)
    results = []
    prompt_offset = 0
    fields = DPSequenceResult.__dataclass_fields__
    cache_rows = []
    for name, entry in entries.items():
        for row in sorted(entry["results"], key=lambda r: r["trajectory_index"]):
            result = DPSequenceResult(**{field: row[field] for field in fields})
            results.append(replace(result, global_index=prompt_offset + result.dataset_index))
        cache_info = dict(entry["cache_info"])
        cache_info["status"] = "populated" if name in payload["run"].get(
            "target_cache_datasets_populated", [],
        ) else "reused"
        cache_rows.append(cache_info)
        prompt_offset += cache_info["prompts"]
    report_filename = "report_dp.md"
    results_path = run_dir / "dp_results.jsonl"
    payload["datasets"] = build_dataset_summaries(results, report_datasets)
    payload["aggregate"] = aggregate_sequence_results(results)
    payload["run"].update({
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "report_filename": report_filename,
        "requested_datasets": list(args.datasets),
        "evaluated_datasets_this_run": list(pending),
        "skipped_completed_datasets": list(skipped),
        "report_datasets": report_datasets,
        "dp_results": str(results_path), "current_dp_results": str(results_path),
        "target_caches": cache_rows, "prompts": prompt_offset, "trajectories": len(results),
        "dataset_prompt_counts": {row["dataset"]: row["prompts"] for row in cache_rows},
        "target_samples_per_prompt": {
            row["dataset"]: row["samples_per_prompt"] for row in cache_rows
        },
        "target_outputs": {row["dataset"]: row["target_outputs"] for row in cache_rows},
    })
    _atomic_write_jsonl(results_path, [_serialize_dp_result(result) for result in results])
    _atomic_write_json(run_dir / "metrics.json", payload)
    report_path = run_dir / report_filename
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(render_report(payload), encoding="utf-8")
    temporary.replace(report_path)
    return payload


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root = str(resolve_output_root(args.draft_checkpoint, args.output_root))
    checkpoint = resolve_checkpoint(args.draft_checkpoint)
    checkpoint_name = checkpoint_output_name(checkpoint, args.checkpoint_name)
    run_dir = Path(args.output_root).expanduser().resolve() / checkpoint_name
    identity = _evaluation_identity(args, checkpoint)
    # Reports retain the other completed DeepSpec datasets even when a rerun
    # explicitly selects only one new dataset.
    report_order = list(DEEPSPEC_DATASETS)
    store = DPResultStore(run_dir, identity, dataset_cache_dirs={
        name: resolve_target_cache_directory(args, name) for name in SUPPORTED_DATASETS
    })
    # Validate and import report-only results before metrics.json is rewritten.
    _migrate_completed_results(run_dir, args, store, identity, SUPPORTED_DATASETS)
    entries = {name: entry for name in report_order if (entry := store.load(name)) is not None}
    skipped = [name for name in args.datasets if name in entries]
    pending = [name for name in args.datasets if name not in entries]
    for name in skipped:
        print(f"Skipping completed {name} DP results ({len(entries[name]['results'])} sequences)",
              flush=True)

    def save_dataset(name, results, cache_info, run_info):
        entries[name] = store.save(
            name, [_serialize_dp_result(result) for result in results], cache_info, run_info,
        )
        print(f"Saved completed {name} DP results ({len(results)} sequences)", flush=True)

    if pending:
        pending_args = argparse.Namespace(**vars(args))
        pending_args.datasets = pending
        payload = _run_pending_evaluation(
            pending_args, on_dataset_complete=save_dataset, evaluation_identity=identity,
        )
    else:
        # A completed rerun requires neither a GPU nor model/tokenizer loading.
        payload = {"run": dict(next(iter(entries.values()))["run_info"])}
        payload["run"].update({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_seconds": 0.0, "feature_extraction_seconds": 0.0,
            "scoring_seconds": 0.0, "score_batches": 0,
            "score_effective_max_batch_size": 0,
            "score_context_padding_fraction": 0.0, "score_proposal_padding_fraction": 0.0,
            "target_cache_hit": True,
            "target_cache_datasets_reused": list(args.datasets),
            "target_cache_datasets_populated": [],
        })
    payload["run"].update({
        "evaluation_identity": identity, "checkpoint_name": checkpoint_name,
        "draft_checkpoint_input": args.draft_checkpoint,
        "draft_checkpoint_resolved": str(checkpoint.source_path),
        "target_cache_root": str(resolve_target_cache_directory(args, args.datasets[0]).parent),
        "min_target_sequences_per_dataset": MIN_TARGET_SEQUENCES_PER_DATASET,
    })
    ordered_entries = {name: entries[name] for name in report_order if name in entries}
    return _publish_incremental_results(args, run_dir, ordered_entries, payload, pending, skipped)


def _run_pending_evaluation(
    args: argparse.Namespace, *, on_dataset_complete: Callable,
    evaluation_identity: dict[str, Any],
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    # Freeze the concrete checkpoint resolved by the coordinator: a concurrently
    # training job may advance latest_checkpointed_iteration.txt during evaluation.
    checkpoint = ResolvedCheckpoint(
        args.draft_checkpoint, Path(evaluation_identity["checkpoint"]),
        evaluation_identity["checkpoint_kind"],
    )
    checkpoint_name = checkpoint_output_name(checkpoint, args.checkpoint_name)
    run_dir = Path(args.output_root).expanduser().resolve() / checkpoint_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_filename = "report_dp.md"
    payload: dict[str, Any] = {
        "run": {
            "status": "initializing",
            "evaluation_identity": evaluation_identity,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "draft_checkpoint_input": args.draft_checkpoint,
            "draft_checkpoint_resolved": str(checkpoint.source_path),
            "checkpoint_kind": checkpoint.kind,
            "checkpoint_name": checkpoint_name,
            "target_model": args.target_model,
            "prompts_dir": str(Path(args.prompts_dir).expanduser().resolve()),
            "temperature": args.temperature,
            "target_temperature": args.temperature,
            "draft_temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "sampling_identity": _sampling_identity(args),
            "seed": args.seed,
            "dataset_seed": args.dataset_seed,
            "report_filename": report_filename,
            "enable_thinking": args.enable_thinking,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "edr_chunk_size": args.edr_chunk_size,
            "edr_vocab_chunk_size": args.edr_vocab_chunk_size,
            "score_batch_size": args.score_batch_size,
            "score_max_batch_tokens": args.score_max_batch_tokens,
            "score_max_target_tokens": args.score_max_target_tokens,
            "score_dp_workers": args.score_dp_workers,
            "score_prefetch": args.score_prefetch,
            "score_batches": 0,
            "target_samples_per_prompt": {},
            "min_target_sequences_per_dataset": MIN_TARGET_SEQUENCES_PER_DATASET,
            "token_counting": DP_TOKEN_COUNTING,
            "dcut_enabled": False,
        },
        "datasets": [],
    }
    _atomic_write_json(run_dir / "metrics.json", payload)

    scorer: OfflineEDRScorer | None = None
    try:
        draft_config = json.loads(args.draft_config.read_text(encoding="utf-8"))
        target_config = AutoConfig.from_pretrained(
            args.target_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        _validate_target_geometry(target_config, draft_config)
        target_geometry = getattr(target_config, "text_config", target_config)
        target_max_positions = int(target_geometry.max_position_embeddings)
        if args.max_model_len > min(
            target_max_positions,
            int(draft_config["max_position_embeddings"]),
        ):
            raise ValueError(
                f"--max-model-len={args.max_model_len} exceeds target/draft positional capacity"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        eos_token_ids = _normalize_token_ids(
            getattr(tokenizer, "eos_token_id", None),
            getattr(target_geometry, "eos_token_id", None),
        )
        try:
            generation_config = GenerationConfig.from_pretrained(
                args.target_model,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
        except OSError:
            generation_config = GenerationConfig.from_model_config(target_config)
        eos_token_ids.update(
            _normalize_token_ids(
                getattr(generation_config, "eos_token_id", None),
                getattr(generation_config, "stop_token_ids", None),
            )
        )
        if not eos_token_ids:
            raise ValueError("The target model does not define an EOS/stopping token ID")
        payload["run"]["eos_token_ids"] = sorted(eos_token_ids)

        cache_requests: dict[str, TargetCacheRequest] = {}
        trajectories_by_dataset: dict[str, list[TargetTrajectory]] = {}
        cache_manifests: dict[str, dict[str, Any]] = {}
        cache_status: dict[str, str] = {}
        missing_datasets: list[str] = []
        for dataset in args.datasets:
            target_cache_dir = resolve_target_cache_directory(args, dataset)
            request = TargetCacheRequest(
                dataset=dataset,
                cache_dir=target_cache_dir,
                prompt_specs=(),
            )
            cache_requests[dataset] = request
            if target_cache_dir.exists():
                print(
                    f"Reusing fixed {dataset} target cache: {target_cache_dir}",
                    flush=True,
                )
                trajectories_by_dataset[dataset], cache_manifests[dataset] = (
                    load_target_cache(
                        target_cache_dir,
                        dataset=dataset,
                    )
                )
                cache_status[dataset] = "reused"
            else:
                missing_datasets.append(dataset)

        generation_seconds = 0.0
        feature_extraction_seconds = 0.0
        missing_requests: list[TargetCacheRequest] = []
        if missing_datasets:
            missing_prompt_specs, _ = load_prompt_specs(args, missing_datasets)
            for dataset in missing_datasets:
                request = replace(
                    cache_requests[dataset],
                    prompt_specs=tuple(
                        spec
                        for spec in missing_prompt_specs
                        if spec.dataset == dataset
                    ),
                )
                cache_requests[dataset] = request
                missing_requests.append(request)
            missing_prompt_count = len(missing_prompt_specs)
            sampling_plan = {
                request.dataset: target_samples_per_prompt(len(request.prompt_specs))
                for request in missing_requests
            }
            print(
                f"Target cache absent for {missing_datasets}; samples per prompt "
                f"by dataset: {sampling_plan}; {missing_prompt_count} prompts "
                f"in one vLLM call "
                f"(max_num_seqs={args.max_num_seqs}, "
                f"max_num_batched_tokens={args.max_num_batched_tokens})",
                flush=True,
            )
            populated, generation_seconds, feature_extraction_seconds = (
                populate_target_caches(
                    args,
                    requests=missing_requests,
                    tokenizer=tokenizer,
                    eos_token_ids=eos_token_ids,
                    target_config=target_config,
                    draft_config=draft_config,
                )
            )
            for request in missing_requests:
                trajectories_by_dataset[request.dataset], cache_manifests[
                    request.dataset
                ] = populated[request.dataset]
                cache_status[request.dataset] = "populated"

        trajectories = []
        global_prompt_offset = 0
        for dataset in args.datasets:
            dataset_trajectories = [
                replace(
                    trajectory,
                    global_index=global_prompt_offset + trajectory.dataset_index,
                )
                for trajectory in trajectories_by_dataset[dataset]
            ]
            trajectories_by_dataset[dataset] = dataset_trajectories
            trajectories.extend(dataset_trajectories)
            global_prompt_offset += int(cache_manifests[dataset]["prompt_count"])
        cache_rows = [
            {
                "dataset": dataset,
                "status": cache_status[dataset],
                "cache_dir": str(cache_requests[dataset].cache_dir),
                "target_outputs": str(
                    cache_requests[dataset].cache_dir / "target_outputs.jsonl"
                ),
                "prompts": int(cache_manifests[dataset]["prompt_count"]),
                "samples_per_prompt": int(cache_manifests[dataset]["samples_per_prompt"]),
                "trajectories": len(trajectories_by_dataset[dataset]),
                "cached_generation_seconds": float(
                    cache_manifests[dataset]["generation_seconds"]
                ),
                "cached_feature_extraction_seconds": float(
                    cache_manifests[dataset]["feature_extraction_seconds"]
                ),
            }
            for dataset in args.datasets
        ]
        target_cache_root = cache_requests[args.datasets[0]].cache_dir.parent
        payload["run"].update(
            {
                "status": "target_cached",
                "trajectories": len(trajectories),
                "prompts": global_prompt_offset,
                "dataset_prompt_counts": {
                    dataset: int(cache_manifests[dataset]["prompt_count"])
                    for dataset in args.datasets
                },
                "target_samples_per_prompt": {
                    dataset: int(cache_manifests[dataset]["samples_per_prompt"])
                    for dataset in args.datasets
                },
                "target_cache_hit": not missing_requests,
                "target_cache_root": str(target_cache_root),
                "target_cache_datasets_reused": [
                    dataset
                    for dataset in args.datasets
                    if cache_status[dataset] == "reused"
                ],
                "target_cache_datasets_populated": [
                    dataset
                    for dataset in args.datasets
                    if cache_status[dataset] == "populated"
                ],
                "target_caches": cache_rows,
                "generation_seconds": generation_seconds,
                "feature_extraction_seconds": feature_extraction_seconds,
                "target_outputs": {
                    row["dataset"]: row["target_outputs"] for row in cache_rows
                },
            }
        )
        _atomic_write_json(run_dir / "metrics.json", payload)

        scorer = OfflineEDRScorer(args, checkpoint, sorted(eos_token_ids))
        payload["run"].update(scorer.versions)
        payload["run"]["status"] = "scoring"
        _atomic_write_json(run_dir / "metrics.json", payload)
        torch = scorer.torch
        torch.cuda.reset_peak_memory_stats(scorer.device)
        scoring_started = time.perf_counter()
        ordered_results: list[DPSequenceResult | None] = [None] * len(trajectories)
        completed_results: list[DPSequenceResult] = []
        score_plans = plan_dp_score_batches(
            trajectories,
            max_batch_size=args.score_batch_size,
            max_batch_tokens=args.score_max_batch_tokens,
            max_proposal_blocks=args.edr_chunk_size,
            max_target_tokens=args.score_max_target_tokens,
        )
        proposal_blocks = sum(
            trajectory.ordinary_tokens
            for trajectory in trajectories
            if trajectory.ordinary_tokens
        )
        sequence_tokens = sum(
            len(trajectory.prompt_token_ids) + trajectory.ordinary_tokens + 1
            for trajectory in trajectories
            if trajectory.ordinary_tokens
        )
        padded_proposal_blocks = sum(
            plan.padded_proposal_blocks for plan in score_plans
        )
        padded_sequence_tokens = sum(
            plan.padded_sequence_tokens for plan in score_plans
        )
        payload["run"].update(
            {
                "score_batches": len(score_plans),
                "score_effective_max_batch_size": max(
                    (len(plan.indexed_trajectories) for plan in score_plans),
                    default=0,
                ),
                "score_proposal_blocks": proposal_blocks,
                "score_padded_proposal_blocks": padded_proposal_blocks,
                "score_proposal_padding_fraction": (
                    1.0 - proposal_blocks / padded_proposal_blocks
                    if padded_proposal_blocks
                    else 0.0
                ),
                "score_sequence_tokens": sequence_tokens,
                "score_padded_sequence_tokens": padded_sequence_tokens,
                "score_context_padding_fraction": (
                    1.0 - sequence_tokens / padded_sequence_tokens
                    if padded_sequence_tokens
                    else 0.0
                ),
            }
        )
        _atomic_write_json(run_dir / "metrics.json", payload)
        cache_directories = {
            dataset: cache_requests[dataset].cache_dir for dataset in args.datasets
        }
        dataset_results: dict[str, list[DPSequenceResult]] = {
            dataset: [] for dataset in args.datasets
        }
        cache_info_by_dataset = {row["dataset"]: row for row in cache_rows}

        def record_result(index: int, result: DPSequenceResult) -> None:
            if not 0 <= index < len(trajectories) or ordered_results[index] is not None:
                raise RuntimeError(f"Invalid or duplicate DP scoring result index: {index}")
            trajectory = trajectories[index]
            if (result.dataset, result.trajectory_index) != (
                trajectory.dataset, trajectory.trajectory_index,
            ):
                raise RuntimeError("DP scoring result does not match its cached trajectory")
            ordered_results[index] = result
            completed_results.append(result)
            rows = dataset_results[result.dataset]
            rows.append(result)
            if len(rows) == len(trajectories_by_dataset[result.dataset]):
                on_dataset_complete(
                    result.dataset, sorted(rows, key=lambda r: r.trajectory_index),
                    cache_info_by_dataset[result.dataset], payload["run"],
                )

        from tqdm.auto import tqdm

        progress = tqdm(
            total=len(trajectories),
            desc="EDR DP",
            unit="sequence",
            disable=args.disable_progress,
        )
        for index, trajectory in enumerate(trajectories):
            if trajectory.ordinary_tokens:
                continue
            result = scorer.score(trajectory, None)
            record_result(index, result)
            progress.update(1)

        prepared_batches = iter_prepared_dp_score_batches(
            score_plans,
            cache_directories=cache_directories,
            expected_layers=scorer.expected_target_layers,
            expected_hidden_size=scorer.expected_hidden_size,
            pin_memory=True,
            prefetch=args.score_prefetch,
        )
        next_log = args.log_every if args.log_every else 0
        try:
            for prepared_batch in prepared_batches:
                batch_results = scorer.score_prepared_batch(prepared_batch)
                for original_index, result in batch_results:
                    record_result(original_index, result)
                progress.update(len(batch_results))
                if next_log and len(completed_results) >= next_log:
                    running = aggregate_sequence_results(completed_results)
                    print(
                        f"EDR DP: {len(completed_results)}/{len(trajectories)} sequences, "
                        f"round_start_MAL={running['round_start_mal']:.4f}, "
                        f"weighted_cost_MAL={running['weighted_cost_mal']:.4f}",
                        flush=True,
                    )
                    next_log = (
                        len(completed_results) // args.log_every + 1
                    ) * args.log_every
        finally:
            prepared_batches.close()
            progress.close()

        if any(result is None for result in ordered_results):
            raise RuntimeError("DP scoring did not produce a result for every trajectory")
        results = [result for result in ordered_results if result is not None]
        torch.cuda.synchronize(scorer.device)
        scoring_seconds = time.perf_counter() - scoring_started
        peak_allocated_gib = torch.cuda.max_memory_allocated(scorer.device) / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved(scorer.device) / 1024**3

        payload["datasets"] = build_dataset_summaries(results, args.datasets)
        payload["aggregate"] = aggregate_sequence_results(results)
        payload["run"].update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "scoring_seconds": scoring_seconds,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                "evaluated_datasets_this_run": list(args.datasets),
                "report_datasets": list(args.datasets),
            }
        )
        # The outer coordinator publishes old and new datasets together. Every
        # completed dataset is already durable if the process stops before that.
        return payload
    except Exception as exc:
        payload["run"].update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(run_dir / "metrics.json", payload)
        raise
    finally:
        if scorer is not None:
            scorer.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    if args.resolve_output_root:
        print(resolve_output_root(args.resolve_output_root, args.output_root))
        return 0
    if args.resolve_checkpoint_name:
        checkpoint = resolve_checkpoint(args.resolve_checkpoint_name)
        print(checkpoint_output_name(checkpoint, args.checkpoint_name))
        return 0
    try:
        payload = run_evaluation(args)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report = (
        Path(args.output_root).expanduser().resolve()
        / payload["run"]["checkpoint_name"]
        / payload["run"]["report_filename"]
    )
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
