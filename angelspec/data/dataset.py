# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
import json
import logging as _logging
import multiprocessing as mp
import os
from pathlib import Path

import torch
from tqdm import tqdm

from angelspec.data.parse import create_parser, has_thinking_content
from angelspec.data.preprocessing import (
    _normalize_conversation,
    preprocess_conversations,
)
from angelspec.data.template import TEMPLATE_REGISTRY
from angelspec.data.utils import (
    estimate_row_count,
    extract_media_urls,
    flatten_multimodal_content,
    load_hf_dataset,
)
from angelspec.utils.logging import logger
from angelspec.utils.processing import load_tokenizer

_logging.getLogger("transformers_modules").setLevel(_logging.ERROR)

_worker_state = {}


def _find_single_tokenized_cache(cache_dir: str) -> str | None:
    """Return the sole tokenized ``.pt`` cache, rejecting ambiguity."""
    if not os.path.isdir(cache_dir):
        return None

    cache_files = sorted(
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith(".pt") and os.path.isfile(os.path.join(cache_dir, name))
    )
    if len(cache_files) > 1:
        raise RuntimeError(
            f"Expected at most one tokenized dataset cache in {cache_dir}, "
            f"found {len(cache_files)}: {cache_files}"
        )
    return cache_files[0] if cache_files else None


def target_rollout_sampling(args) -> dict | None:
    """Resolve an explicit rollout policy, or EDR's required on-policy policy."""
    from angelspec.utils.sampling import validate_sampling_parameters

    is_edr = str(getattr(args, "dflash_loss_objective", "decay")).lower() == "edr"
    configured = getattr(args, "target_sampling", None)
    if configured is None and not is_edr:
        return None
    edr_sampling = {
        "temperature": getattr(args, "dflash_edr_temperature", 1.0),
        "top_k": getattr(args, "dflash_edr_top_k", -1),
        "top_p": getattr(args, "dflash_edr_top_p", 1.0),
    }
    sampling = dict(configured) if configured is not None else dict(edr_sampling)
    sampling.setdefault("min_p", 0.0)
    sampling.setdefault("enable_thinking", False)
    validate_sampling_parameters(sampling["temperature"], sampling["top_k"], sampling["top_p"])
    if sampling["min_p"] != 0.0 or sampling["enable_thinking"] is not False:
        raise ValueError("Target rollout caches currently require min_p=0 and enable_thinking=false")
    if is_edr and any(sampling[key] != value for key, value in edr_sampling.items()):
        raise ValueError("dataset.target_sampling must match the EDR temperature/top-k/top-p")
    return sampling


def find_tokenized_cache_for_training(args) -> str | None:
    """Return the tokenized cache to train on, or None if none matches.

    Without a target sampling policy, the single ``.pt`` cache in cache_dir is
    used. With a policy, a cache matches when its provenance file (the
    ``.pt.json`` sidecar, or else ``target_rollout_work/<stem>/manifest.json``)
    is complete and records the same target model and sampling. Filenames alone
    do not establish provenance. The ``.pt`` payload is not read or hashed.
    """
    cache_root = Path(getattr(args, "cache_dir", "./cache"))
    cache_dir = cache_root / "tokenized_dataset"
    policy = target_rollout_sampling(args)
    if policy is None:
        return _find_single_tokenized_cache(str(cache_dir))

    from angelspec.utils.sampling import target_model_cache_id

    expected_model = target_model_cache_id(args.target_model_path)
    expected_sampling = {
        **{key: value for key, value in policy.items() if key != "min_p"},
        "n": 1,
        "max_total_tokens": args.max_seq_length,
    }

    matching = []
    for cache_path in sorted(cache_dir.glob("*.pt")):
        if not cache_path.is_file():
            continue
        metadata_path = cache_path.with_suffix(".pt.json")
        if not metadata_path.is_file():
            metadata_path = (
                cache_root / "target_rollout_work" / cache_path.stem / "manifest.json"
            )
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read EDR cache provenance: {metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid EDR cache provenance: {metadata_path}")
        sampling = metadata.get("sampling")
        model = metadata.get("target_model")
        if (
            metadata.get("status") != "complete"
            or metadata.get("artifact_name") != cache_path.name
            or not isinstance(model, str)
            or target_model_cache_id(model) != expected_model
            or not isinstance(sampling, dict)
            # A missing min_p means 0, the only value the regeneration tool uses.
            or sampling.get("min_p", 0.0) != policy["min_p"]
            or any(sampling.get(key) != value for key, value in expected_sampling.items())
        ):
            continue
        matching.append(str(cache_path))

    if len(matching) > 1:
        raise RuntimeError(
            f"Multiple target-generated EDR caches match model/sampling in {cache_dir}: "
            f"{matching}. Select a dedicated cache_dir containing one matching artifact."
        )
    return matching[0] if matching else None


def _init_tokenize_worker(
    tokenizer_path,
    trust_remote_code,
    chat_template_name,
    last_turn_loss_only=False,
    min_loss_tokens=0,
    drop_overlength=False,
):
    """Initializer for each worker process — loads tokenizer once."""
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    _worker_state["tokenizer"] = load_tokenizer(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["preprocess"] = preprocess_conversations
    _worker_state["last_turn_loss_only"] = last_turn_loss_only
    _worker_state["min_loss_tokens"] = min_loss_tokens
    _worker_state["drop_overlength"] = drop_overlength


def _resolve_last_turn_loss_only(messages):
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        return has_thinking_content(messages)
    return bool(ltlo)


def _tokenize_single(args):
    """Worker function — tokenize one sample."""
    messages, max_length, train_with_decode, extra = args
    # `extra` carries per-sample top-level fields (e.g. tools, reasoning_effort)
    # consumed by the chat template's jinja. preprocess_conversations expects
    # column-wise kwargs (one list per key, aligned with the conversations
    # list), so wrap each value in a length-1 list for this single sample.
    extra_kwargs = {k: [v] for k, v in (extra or {}).items()}
    processed = _worker_state["preprocess"](
        _worker_state["tokenizer"],
        [messages],
        _worker_state["template"],
        max_length=max_length,
        is_preformatted=False,
        include_attention_mask=False,
        use_packed_loss_mask=True,
        add_generation_prompt=train_with_decode,
        return_formatted_text=True,
        last_turn_loss_only=_resolve_last_turn_loss_only(messages),
        min_loss_tokens=_worker_state.get("min_loss_tokens", 0),
        drop_overlength=_worker_state.get("drop_overlength", False),
        **extra_kwargs,
    )
    if not processed["input_ids"]:
        return None
    # Return plain lists instead of tensors to avoid shared memory mmap
    # exhaustion when transferring results across process boundaries.
    input_ids = processed["input_ids"][0]
    return {
        "input_ids": input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
        "packed_loss_mask": processed["packed_loss_mask"][0],
        "formatted_prompt": processed["formatted_text"][0],
    }


def _init_format_worker(
    tokenizer_path, trust_remote_code, chat_template_name, last_turn_loss_only=False
):
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["parser"] = create_parser(tokenizer, _worker_state["template"])
    _worker_state["last_turn_loss_only"] = last_turn_loss_only


def _format_single(args):
    """
    Worker function — format only, skip tokenization.
    """
    messages, _, train_with_decode, _ = args
    messages = _normalize_conversation(messages)

    result = {}
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        result["has_thinking"] = has_thinking_content(messages)

    parser = _worker_state["parser"]
    formatted = parser.format(
        messages, add_generation_prompt=train_with_decode, expand_media_tokens=False
    )
    if not formatted:
        return None
    result["formatted_prompt"] = formatted
    return result


def load_conversation_dataset(args):
    """Load conversation dataset and optionally tokenize for training.

    When defer_tokenization=True, only applies the chat template to produce
    formatted text — no tokenizer is loaded and no input_ids/loss_mask are
    generated. The inference engine handles tokenization and media token
    expansion; loss mask is computed at training time from the engine's
    actual input_ids.

    When defer_tokenization=False (default), fully tokenizes and produces
    input_ids + packed_loss_mask for the input_ids engine path.

    Returns list of dicts. Fields depend on mode:
        defer_tokenization=True:  data_id, formatted_prompt, multimodal_inputs, metadata
        defer_tokenization=False: data_id, input_ids, packed_loss_mask, formatted_prompt, multimodal_inputs, metadata
    """
    prompt_key = getattr(args, "prompt_key", "text")
    chat_template_name = getattr(args, "chat_template", None)
    # By default the limit is max_seq_length - 1, reserving vLLM's feature-extraction
    # output slot. With allow_full_length_cached_sequences, cached sequences may use
    # max_seq_length tokens and vLLM gets that slot outside the training limit.
    full_length_cache = bool(getattr(args, "allow_full_length_cached_sequences", False))
    max_length = args.max_seq_length - int(not full_length_cache)
    defer_tokenization = getattr(args, "defer_tokenization", False)

    logger.info(f"Max sequence length allowed for training: {max_length}")

    if not chat_template_name:
        raise ValueError("chat_template must be set for load_conversation_dataset")

    dataset_name = os.path.basename(args.train_data_path)
    file_stat = ""
    if os.path.isfile(args.train_data_path):
        st = os.stat(args.train_data_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    last_turn_loss_only_flag = getattr(args, "last_turn_loss_only", False)
    train_with_decode = getattr(args, "train_with_decode", False)
    min_loss_tokens_val = getattr(args, "min_loss_tokens", 0)
    drop_overlength_flag = getattr(args, "drop_overlength", False)
    cache_params = (
        f"{dataset_name}-{args.train_data_path}{file_stat}-{args.target_model_path}"
        f"-{max_length}-{chat_template_name}-ltlo={last_turn_loss_only_flag}"
        f"-defer={defer_tokenization}-decode={train_with_decode}"
        f"-mlt={min_loss_tokens_val}-drop={drop_overlength_flag}"
    )
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "tokenized_dataset")
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    policy = target_rollout_sampling(args)
    existing_cache_path = find_tokenized_cache_for_training(args)
    if existing_cache_path is not None:
        if policy is not None:
            logger.info("Loading matching target-generated cache: %s", existing_cache_path)
        else:
            logger.info(
                "Loading sole tokenized dataset cache without validating its hash: "
                f"{existing_cache_path}"
            )
        prompts = torch.load(existing_cache_path, weights_only=False)
        logger.info(f"Loaded {len(prompts)} cached samples")
        return prompts

    if policy is not None:
        raise FileNotFoundError(
            f"No completed target-generated cache in {cache_dir} matches target model "
            f"{args.target_model_path!r}, sampling={policy}, "
            f"max_total_tokens={args.max_seq_length}. Run tools/regenerate_perfectblend.py "
            "with matching --target-model, --temperature, --top-k, --top-p, "
            "--max-total-tokens and --cache-dir. Keep its .pt.json sidecar (or a completed "
            "legacy rollout manifest). This recipe never falls back to source assistant responses."
        )

    if full_length_cache:
        raise FileNotFoundError(
            "dataset.allow_full_length_cached_sequences requires an existing "
            f"tokenized cache in {cache_dir}; it never falls back to raw-source tokenization"
        )

    custom_template = TEMPLATE_REGISTRY.get(chat_template_name)
    hf_dataset = load_hf_dataset(args.train_data_path)

    mode_label = "Formatting" if defer_tokenization else "Tokenizing"
    logger.info(f"{mode_label} dataset (cache will be saved to {cache_path})")

    total_estimate = estimate_row_count(args.train_data_path)
    num_proc = getattr(args, "num_proc", 64)

    # Pass 1: collect and normalize raw samples (fast I/O, no tokenization)
    raw_samples = []
    for idx, sample in enumerate(tqdm(hf_dataset, desc="Loading samples", total=total_estimate)):
        raw_prompt = sample.get(prompt_key, "")

        if not isinstance(raw_prompt, list):
            raise ValueError(
                f"Expected conversation format (list of messages) for sample {idx}, got {type(raw_prompt)}"
            )

        messages = _normalize_conversation(raw_prompt)
        multimodal_inputs = extract_media_urls(messages)
        flatten_multimodal_content(messages, custom_template.image_placeholder)
        data_id = sample.get("id", f"sample_{idx}")
        # Top-level fields consumed by the chat template's jinja (e.g. agent
        # tool definitions, reasoning effort). Kept per-sample and threaded
        # through to parser.format via preprocess_conversations kwargs.
        extra = {}
        for key in ("tools", "reasoning_effort"):
            val = sample.get(key)
            if val:
                extra[key] = val
        raw_samples.append((data_id, messages, multimodal_inputs, extra))

    logger.info(
        f"Loaded {len(raw_samples)} samples, {mode_label.lower()} with {num_proc} workers..."
    )

    # Pass 2: process in parallel
    work_items = [
        (messages, max_length, train_with_decode, extra) for _, messages, _, extra in raw_samples
    ]

    last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
    if defer_tokenization:
        worker_init = _init_format_worker
        worker_initargs = (args.target_model_path, True, chat_template_name, last_turn_loss_only)
        worker_fn = _format_single
        desc = "Formatting dataset"
    else:
        if last_turn_loss_only:
            logger.info(
                f"last_turn_loss_only={last_turn_loss_only}: loss mask will only cover the last assistant turn"
            )
        min_loss_tokens = getattr(args, "min_loss_tokens", 0)
        drop_overlength = getattr(args, "drop_overlength", False)
        worker_init = _init_tokenize_worker
        worker_initargs = (
            args.target_model_path,
            True,
            chat_template_name,
            last_turn_loss_only,
            min_loss_tokens,
            drop_overlength,
        )
        worker_fn = _tokenize_single
        desc = "Tokenizing dataset"

    if num_proc <= 1:
        worker_init(*worker_initargs)
        results = [worker_fn(item) for item in tqdm(work_items, desc=desc)]
    else:
        with mp.Pool(num_proc, initializer=worker_init, initargs=worker_initargs) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_fn, work_items, chunksize=64),
                    total=len(work_items),
                    desc=desc,
                )
            )

    # Collect results
    prompts = []
    skipped = 0
    for (data_id, _, multimodal_inputs, _), result in zip(raw_samples, results):
        if result is None:
            skipped += 1
            continue
        metadata = {}
        if "has_thinking" in result:
            metadata["has_thinking"] = result["has_thinking"]

        entry = {
            "data_id": data_id,
            "metadata": metadata,
            "multimodal_inputs": multimodal_inputs,
            "formatted_prompt": result["formatted_prompt"],
        }

        if not defer_tokenization:
            input_ids = result["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            entry["input_ids"] = input_ids
            entry["packed_loss_mask"] = result["packed_loss_mask"]

        prompts.append(entry)

    if skipped:
        reasons = "empty source, zero loss mask"
        if getattr(args, "drop_overlength", False):
            reasons += f", or > {max_length} tokens (drop_overlength)"
        logger.warning(f"Skipped {skipped} samples ({reasons})")

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(prompts, cache_path)
    logger.info(f"Saved {len(prompts)} samples to cache: {cache_path}")

    return prompts
