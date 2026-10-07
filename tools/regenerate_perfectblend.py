#!/usr/bin/env python3
"""Regenerate one reply to each Open-PerfectBlend first user message by default.

Leading system instructions are retained; old answers and later turns are discarded.
Use --turn-policy all to regenerate every reply, conditioning on newly generated earlier replies.
Durable prompt shards and rollout progress support resuming each training epoch.
Normally stopped responses end at their sampled EOS, never a template newline.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the angelspec package loads GPU training dependencies, so load the
# dependency-free sampling/cache-key helpers directly from their file.
_sampling_spec = importlib.util.spec_from_file_location(
    "angelspec_cpu_sampling",
    ROOT / "angelspec/utils/sampling.py",
)
if _sampling_spec is None or _sampling_spec.loader is None:
    raise ImportError("Cannot load AngelSpec's local sampling helpers")
_sampling = importlib.util.module_from_spec(_sampling_spec)
_sampling_spec.loader.exec_module(_sampling)
sampling_cache_key = _sampling.sampling_cache_key
target_model_cache_id = _sampling.target_model_cache_id
validate_sampling_parameters = _sampling.validate_sampling_parameters

TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_stamp(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _local_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _portable_model_reference(path):
    # The loader resolves this relative to the training repository. Do not bake
    # the CPU node's mount point into a cache that will move to a GPU node.
    if path.is_relative_to(ROOT):
        return "./" + path.relative_to(ROOT).as_posix()
    return str(path)


@contextmanager
def _atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            # Only remove this invocation's uncommitted temporary file.
            temporary.unlink(missing_ok=True)


def _save_json(value, path):
    with _atomic_file(path) as handle:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _save_torch(value, path):
    import torch

    with _atomic_file(path) as handle:
        torch.save(value, handle)


def _load_torch(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)

FORMAT = "angelspec-cached-multiturn-regeneration"
VERSION = 1
DEFAULT_CONFIG = ROOT / "configs/vllm_qwen3_4b_dspark_edr.yaml"


@dataclass
class QwenLayout:
    start: int
    end: int
    newline: list[int]
    headers: dict[str, list[int]]
    nonthinking: list[int]
    stop_ids: list[int]
    header_newline_ids: list[int] | None = None

    @classmethod
    def from_tokenizer(cls, tokenizer, stop_ids):
        def encode(text):
            return list(tokenizer.encode(text, add_special_tokens=False))

        start, end = encode("<|im_start|>"), encode("<|im_end|>")
        if len(start) != 1 or len(end) != 1 or end[0] not in stop_ids:
            raise ValueError("Expected a Qwen tokenizer with an independently stopping <|im_end|>")
        headers = {
            role: encode(f"<|im_start|>{role}\n") for role in ("system", "user", "assistant")
        }
        prefix = encode("<think>\n\n</think>\n\n")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        suffix = headers["assistant"] + prefix
        if not prefix or list(prompt[-len(suffix) :]) != suffix:
            raise ValueError("Expected the Qwen3 non-thinking assistant generation prefix")
        merged_newlines = {
            encoded[0] for size in range(1, 33) if len(encoded := encode("\n" * size)) == 1
        }
        return cls(
            start[0],
            end[0],
            encode("\n"),
            headers,
            prefix,
            sorted(stop_ids),
            sorted(merged_newlines),
        )


def unpack_mask(packed: str, length: int) -> list[int]:
    runs = [int(value) for value in packed.split(",")] if packed else []
    if any(value < 0 for value in runs) or sum(runs) != length:
        raise ValueError("Packed loss mask does not match input token length")
    return [index % 2 for index, size in enumerate(runs) for _ in range(size)]


def pack_mask(mask: list[int]) -> str:
    runs, current = [0], 0
    for value in mask:
        if value not in (0, 1):
            raise ValueError("Loss mask must be binary")
        if value != current:
            runs.append(0)
            current = value
        runs[-1] += 1
    return ",".join(map(str, runs))


def extract_conversation(entry, index: int, layout: QwenLayout, limit: int, *, turn_policy="first"):
    """Extract only a verified prompt skeleton and deterministic empty-think prefix.

    No decoding/re-tokenization, and no arbitrary old assistant content survives.
    A complete final user turn can get a new assistant header; a partial user
    turn cannot safely be reconstructed and is rejected (or explicitly skipped).
    """
    if turn_policy not in {"first", "all"}:
        raise ValueError(f"Unsupported turn policy: {turn_policy!r}")
    tokens = (
        entry["input_ids"].tolist()
        if hasattr(entry["input_ids"], "tolist")
        else list(entry["input_ids"])
    )
    if not tokens or any(not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError("Expected a nonempty 1-D token sequence")
    mask = unpack_mask(entry["packed_loss_mask"], len(tokens))
    # A sampled answer can contain a literal message-start token. Only
    # unsupervised template headers delimit retained source messages.
    starts = [pos for pos, token in enumerate(tokens) if token == layout.start and mask[pos] == 0]
    if not starts or starts[0] != 0:
        raise ValueError("Conversation must begin with a Qwen message header")
    starts.append(len(tokens))
    turns, pending, pending_mask = [], [], []
    last_role = None
    inserted_prefixes = 0
    for begin, finish in zip(starts[:-1], starts[1:], strict=True):
        message, message_mask = tokens[begin:finish], mask[begin:finish]
        # Match role IDs without the newline, as the normal Qwen loss-mask
        # parser does: BPE may merge it with content-leading newlines. User
        # tokens are still copied exactly; assistant formatting is canonical.
        bare_headers = {
            role: header[: -len(layout.newline)] for role, header in layout.headers.items()
        }
        role = next(
            (role for role, header in bare_headers.items() if message[: len(header)] == header),
            None,
        )
        if role is None:
            raise ValueError("Incomplete or unsupported Qwen role header")
        header = layout.headers[role]
        body_start = len(bare_headers[role])
        if len(message) <= body_start:
            raise ValueError("Incomplete Qwen role header newline")
        if message[body_start : body_start + len(layout.newline)] == layout.newline:
            body_start += len(layout.newline)
        elif message[body_start] in (layout.header_newline_ids or []):
            body_start += 1
        else:
            raise ValueError("Unsupported Qwen header newline encoding")
        if any(message_mask[:body_start]):
            raise ValueError("Role header must be outside the assistant loss mask")
        body, body_mask = message[body_start:], message_mask[body_start:]
        if role != "assistant":
            if any(message_mask):
                raise ValueError("User/system tokens must not be supervised")
            if layout.end not in body:
                raise ValueError("Incomplete user/system message at source length boundary")
            closing = body.index(layout.end)
            if body[closing + 1 :] not in ([], layout.newline):
                raise ValueError("Unsupported tokens after user/system end-of-turn")
            pending.extend(message)
            pending_mask.extend(message_mask)
            last_role = role
            if role == "user" and turn_policy == "first":
                # Do not inspect old answers or validate/reserve later turns.
                # Build a fresh, unsupervised assistant prefix below.
                break
            continue
        if last_role != "user":
            raise ValueError("Assistant reply must follow a retained user turn")
        stop = next((pos for pos, token in enumerate(body) if token in layout.stop_ids), None)
        suffix = [] if stop is None else body[stop + 1 :]
        if suffix not in ([], layout.newline):
            raise ValueError("Unsupported tokens after assistant EOS")
        if stop is None and finish != len(tokens):
            raise ValueError("Unclosed source assistant turn before another message")
        answer = body if stop is None else body[: stop + 1]
        answer_mask = body_mask[: len(answer)]
        if stop is not None and answer_mask[-1] == 0:
            reasons = entry.get("metadata", {}).get("finish_reasons", [])
            if (
                body[stop] != layout.end
                or finish == len(tokens)
                or len(reasons) <= len(turns)
                or reasons[len(turns)] != "length"
            ):
                raise ValueError(
                    "Unsupervised assistant EOS is not a verified intermediate length closure"
                )
            # All-turn generation closes a length-capped intermediate reply with
            # an unsupervised <|im_end|> to format the next user turn; drop it.
            answer, answer_mask = answer[:-1], answer_mask[:-1]
        prefix = layout.nonthinking
        if answer[: len(prefix)] == prefix:
            prefix_mask = answer_mask[: len(prefix)]
            answer, answer_mask = answer[len(prefix) :], answer_mask[len(prefix) :]
        elif len(answer) < len(prefix) and prefix[: len(answer)] == answer:
            # A cache can end partway through the fixed nonthinking header.
            # Rebuild it as prompt context, not a disconnected supervised
            # prefix horizon whose last token would be a fabricated boundary.
            prefix_mask = [0] * len(prefix)
            answer, answer_mask = [], []
            inserted_prefixes += 1
        else:
            if answer and answer[0] == prefix[0]:
                raise ValueError(
                    "Source contains a nonempty thinking prefix, not a non-thinking reply"
                )
            prefix_mask = [0] * len(prefix)
            inserted_prefixes += 1
        if any(value != 1 for value in answer_mask):
            raise ValueError(
                "Unsupervised old assistant content cannot be reused as prompt context"
            )
        if suffix and any(body_mask[-len(suffix) :]):
            raise ValueError("Post-EOS template newline must be outside the loss mask")
        turns.append(
            {
                "tokens": pending + header + prefix,
                "mask": pending_mask + [0] * len(header) + prefix_mask,
            }
        )
        # Preserve the exact separator between turns, but never append it to
        # the final regenerated answer. Supply one if the source omitted it.
        pending = suffix or list(layout.newline)
        pending_mask = [0] * len(pending)
        last_role = role
    if last_role == "user":
        # The user message is complete, even if truncation removed its reply.
        header = layout.headers["assistant"] + layout.nonthinking
        turns.append({"tokens": pending + header, "mask": pending_mask + [0] * len(header)})
        inserted_prefixes += 1
    elif last_role != "assistant":
        raise ValueError("Conversation has no final user/assistant turn")
    fixed_tokens = sum(len(turn["tokens"]) for turn in turns)
    # One real sampled token per reply; reserve a masked closure in every
    # intermediate turn in case that reply reaches its allocated length cap.
    if fixed_tokens + len(turns) + len(turns) - 1 > limit:
        raise ValueError(
            "Preserved user turns leave no response capacity under the total token limit"
        )
    return {
        "conversation_index": index,
        "data_id": str(entry["data_id"]),
        "source_index": entry.get("metadata", {}).get("source_index", index),
        "turns": turns,
        "inserted_nonthinking_prefixes": inserted_prefixes,
    }


def extract_json_conversation(
    row, index: int, layout: QwenLayout, tokenizer, limit: int, *, turn_policy="first"
):
    """Keep selected user/system messages, never source assistant answer text."""
    if turn_policy not in {"first", "all"}:
        raise ValueError(f"Unsupported turn policy: {turn_policy!r}")
    messages = row.get("conversations") or row.get("conversation") or row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected a conversation list")
    turns, pending = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Expected conversation message objects")
        role = message.get("role", message.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role == "assistant":
            continue  # Never inspect, retain, or tokenize the old answer.
        if role not in {"user", "system"}:
            raise ValueError(f"Unsupported source role: {role!r}")
        content = message.get("content", message.get("value"))
        if not isinstance(content, str):
            raise ValueError("Only text user/system messages are supported")
        pending.append({"role": role, "content": content})
        if role == "user":
            tokens = list(
                tokenizer.apply_chat_template(
                    pending,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
            )
            suffix = layout.headers["assistant"] + layout.nonthinking
            if tokens[-len(suffix) :] != suffix:
                raise ValueError(
                    "Tokenizer did not produce the verified non-thinking assistant header"
                )
            if turns:
                tokens = layout.newline + tokens
            turns.append({"tokens": tokens, "mask": [0] * len(tokens)})
            pending = []
            if turn_policy == "first":
                break
    if pending or not turns:
        raise ValueError("Source needs a user turn after any system-only prefix/suffix")
    if sum(len(turn["tokens"]) for turn in turns) + 2 * len(turns) - 1 > limit:
        raise ValueError(
            "Preserved user turns leave no response capacity under the total token limit"
        )
    return {
        "conversation_index": index,
        "data_id": str(row.get("id", f"opb_{index:09d}")),
        "source_index": index,
        "turns": turns,
        "inserted_nonthinking_prefixes": len(turns),
    }


_JSON_TOKENIZER = None
_JSON_LAYOUT = None
_JSON_LIMIT = None
_JSON_TURN_POLICY = None


def _init_json_worker(model, layout, limit, turn_policy):
    global _JSON_TOKENIZER, _JSON_LAYOUT, _JSON_LIMIT, _JSON_TURN_POLICY
    from transformers import AutoTokenizer

    _JSON_TOKENIZER = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=True
    )
    _JSON_LAYOUT, _JSON_LIMIT = layout, limit
    _JSON_TURN_POLICY = turn_policy


def _extract_json_line(task, tokenizer=None, layout=None, limit=None, turn_policy=None):
    index, line = task
    try:
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Expected a JSON object")
        return extract_json_conversation(
            row, index, layout or _JSON_LAYOUT, tokenizer or _JSON_TOKENIZER, limit or _JSON_LIMIT,
            turn_policy=turn_policy or _JSON_TURN_POLICY,
        ), None
    except (ValueError, KeyError, TypeError) as exc:
        return None, {"conversation_index": index, "reason": str(exc)}


def initial_state(record):
    return {
        "conversation_index": record["conversation_index"],
        "next_turn": 0,
        "tokens": [],
        "mask": [],
        "finish_reasons": [],
        "sampled_tokens": 0,
    }


def next_prompt(record, state, limit: int):
    turn = state["next_turn"]
    prompt = state["tokens"] + record["turns"][turn]["tokens"]
    future = record["turns"][turn + 1 :]
    # Each future turn needs >=1 sampled token; reserve one possible masked
    # closure for THIS reply and each intermediate future reply.
    reserve = sum(len(item["tokens"]) for item in future) + 2 * len(future)
    budget = limit - len(prompt) - reserve
    if budget < 1:
        raise RuntimeError("Response budget exhausted despite prompt-skeleton reservation")
    return prompt, budget


def accept_completion(record, state, output, layout: QwenLayout, limit: int):
    prompt, budget = next_prompt(record, state, limit)
    if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
        raise RuntimeError("vLLM changed prompt IDs or returned multiple completions")
    completion = output.outputs[0]
    generated = list(completion.token_ids)
    if not generated or len(generated) > budget:
        raise RuntimeError("Empty or oversized sampled response")
    stop = next((pos for pos, token in enumerate(generated) if token in layout.stop_ids), None)
    reason = completion.finish_reason
    if stop is not None:
        if stop != len(generated) - 1:
            raise RuntimeError("vLLM returned tokens after sampled EOS")
        reason = "stop"
    elif reason != "length" or len(generated) != budget:
        # Token-ID vLLM output retains EOS. Do not manufacture a stop token
        # from a generic finish_reason or text-only result.
        raise RuntimeError(f"Expected sampled EOS or a length-capped response, got {reason!r}")
    turn = state["next_turn"]
    new_state = {
        **state,
        "tokens": prompt + generated,
        "mask": state["mask"] + record["turns"][turn]["mask"] + [1] * len(generated),
        "next_turn": turn + 1,
        "finish_reasons": state["finish_reasons"] + [reason],
        "sampled_tokens": state["sampled_tokens"] + len(generated),
    }
    if stop is None and turn + 1 < len(record["turns"]):
        # Structural closure for the next user turn, NOT a sampled EOS and
        # NOT part of the previous loss span. Its boundary stays the last
        # actually generated token.
        new_state["tokens"].append(layout.end)
        new_state["mask"].append(0)
    if len(new_state["tokens"]) > limit:
        raise RuntimeError("Regenerated conversation exceeded total token limit")
    return new_state


def cache_entry(record, state, seed):
    import torch

    if state["next_turn"] != len(record["turns"]):
        raise RuntimeError("Cannot publish an unfinished conversation")
    return {
        "data_id": record["data_id"],
        "input_ids": torch.tensor(state["tokens"], dtype=torch.long),
        "packed_loss_mask": pack_mask(state["mask"]),
        "formatted_prompt": None,
        "multimodal_inputs": None,
        "metadata": {
            "source_index": record["source_index"],
            "conversation_index": record["conversation_index"],
            "generation_seed": seed + record["conversation_index"],
            "finish_reasons": state["finish_reasons"],
        },
    }


def resolve_source(value: Path):
    if value.is_dir():
        directory = (
            value / "tokenized_dataset" if (value / "tokenized_dataset").is_dir() else value
        )
        paths = sorted(directory.glob("*.pt"))
        if len(paths) != 1:
            raise ValueError(
                f"Expected exactly one epoch-1 .pt cache in {directory}, found {len(paths)}"
            )
        value = paths[0]
    metadata = _json(value.with_suffix(".pt.json"))
    if metadata.get("status") != "complete" or metadata.get("artifact_name") != value.name:
        raise ValueError(f"Source cache is missing complete provenance: {value}")
    if not isinstance(metadata.get("cached_samples"), int) or metadata["cached_samples"] < 1:
        raise ValueError("Source metadata must contain a positive cached_samples count")
    return value, metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--epoch",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=5,
        help="Destination epoch; same first-reply policy for all epochs (default: 5)",
    )
    parser.add_argument(
        "--turn-policy",
        choices=("first", "all"),
        default="first",
        help="Generate one reply to the first user message (default), or regenerate all replies",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--source",
        type=Path,
        help="Raw OPB JSONL: select user turns using --turn-policy, discard the source assistant answers "
        "(default for all epochs: config dataset.train_data_path)",
    )
    sources.add_argument(
        "--source-cache",
        type=Path,
        help="Explicit alternative source: AngelSpec .pt, tokenized_dataset directory or epoch root",
    )
    parser.add_argument("--target-model", type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Destination epoch cache root")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float, choices=[0.0])
    parser.add_argument(
        "--seed", type=int,
        help="Base seed + zero-based source index (default by --epoch: 1:42, 2:749, 3:1456, 4:2163, 5:2870)",
    )
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=min(24, os.cpu_count() or 1),
        help="CPU tokenizer workers for JSONL input (cached IDs need no tokenization)",
    )
    parser.add_argument(
        "--invalid-source",
        choices=("error", "skip"),
        default="error",
        help="Stop on incomplete/malformed source rows, or explicitly skip and report them",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument(
        "--attention-backend",
        type=str.upper,
        choices=("AUTO", "FLASH_ATTN", "TRITON_ATTN"),
        default="AUTO",
        help="vLLM attention backend (default: auto, selected by vLLM)",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--stop-after-preprocessing", action="store_true")
    return parser


def parse_args(argv=None):
    from omegaconf import OmegaConf

    parser = build_parser()
    initial = parser.parse_args(argv)
    config_path = _local_path(initial.training_config)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    epochs = config["dataset"].get("epoch_cache_dirs") or [
        str(Path(config["cache_dir"]) / f"epoch{i}") for i in (1, 2, 3, 4, 5)
    ]
    policy = config["dataset"].get("target_sampling")
    if not policy or policy.get("enable_thinking") is not False:
        parser.error("Training config must specify non-thinking dataset.target_sampling")
    destination = (
        epochs[initial.epoch - 1]
        if len(epochs) >= initial.epoch
        else str(Path(config["cache_dir"]) / f"epoch{initial.epoch}")
    )
    parser.set_defaults(
        target_model=_local_path(config["model"]["target_model_path"]),
        cache_dir=_local_path(destination),
        temperature=policy["temperature"],
        top_p=policy["top_p"],
        top_k=policy["top_k"],
        min_p=policy.get("min_p", 0.0),
        max_total_tokens=config["training"]["max_seq_length"],
        seed={1: 42, 2: 749, 3: 1456, 4: 2163, 5: 2870}[initial.epoch],
    )
    args = parser.parse_args(argv)
    if args.source is None and args.source_cache is None:
        args.source = (config_path.parent / config["dataset"]["train_data_path"]).resolve()
    return parser, args


def validate_args(args, parser):
    args.training_config = _local_path(args.training_config)
    args.source_cache = _local_path(args.source_cache) if args.source_cache is not None else None
    args.source = _local_path(args.source) if args.source is not None else None
    args.cache_dir = _local_path(args.cache_dir)
    args.target_model = _local_path(args.target_model)
    args.work_dir = _local_path(args.work_dir) if args.work_dir is not None else None
    args.policy = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "enable_thinking": False,
    }
    args.prepare_only = args.stop_after_preprocessing
    try:
        validate_sampling_parameters(args.temperature, args.top_k, args.top_p)
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_p != 0:
        parser.error("Only min_p=0 is supported")
    if not (args.target_model / "config.json").is_file():
        parser.error("--target-model must be a complete local model directory")
    if (
        args.shard_size < 1
        or args.max_num_seqs < 1
        or args.max_num_batched_tokens < 1
        or args.num_proc < 1
    ):
        parser.error("Shard and engine batch sizes must be positive")
    if (
        args.seed < 0
        or not 0 < args.gpu_memory_utilization < 1
        or not 2 <= args.max_total_tokens <= 4096
    ):
        parser.error("Require seed>=0, 0<GPU utilization<1, and 2<=total tokens<=4096")
    if args.source is not None and not args.source.is_file():
        parser.error(f"--source does not exist: {args.source}")
    if args.source_cache is not None and not args.source_cache.exists():
        parser.error(f"--source-cache does not exist: {args.source_cache}")


def make_signature(args, source, metadata, layout):
    policy = {**args.policy, "n": 1, "max_total_tokens": args.max_total_tokens}
    if metadata is not None and any(
        metadata.get("sampling", {}).get(key) != value for key, value in policy.items()
    ):
        raise ValueError("Epoch-1 sampling provenance does not match the configured target policy")
    if metadata is not None and (
        not metadata.get("target_model")
        or target_model_cache_id(_local_path(metadata["target_model"]))
        != target_model_cache_id(args.target_model)
    ):
        raise ValueError("Epoch-1 cache target model does not match the configured target")
    return {
        "format": FORMAT,
        "version": VERSION,
        "target_model": _portable_model_reference(args.target_model),
        "target_metadata_sha256": {
            name: _file_hash(args.target_model / name)
            for name in TOKENIZER_FILES
            if (args.target_model / name).is_file()
        },
        "sampling": policy,
        "seed": args.seed,
        "seed_policy": "base_seed_plus_zero_based_source_index_same_seed_each_turn",
        "source_kind": "token_cache" if metadata is not None else "jsonl",
        "source_artifact": source.name,
        "source_stamp": _file_stamp(source),
        "source_metadata_sha256": _file_hash(source.with_suffix(".pt.json"))
        if metadata is not None
        else None,
        "source_rows": metadata["cached_samples"] if metadata is not None else args.source_rows,
        "shard_size": args.shard_size,
        "invalid_source": args.invalid_source,
        "layout": asdict(layout),
        "response_policy": "first_user_turn_one_new_assistant_reply"
        if args.turn_policy == "first"
        else "all_retained_user_turns_with_new_earlier_assistant_context",
        "boundary_policy": "sampled_EOS_or_last_sampled_token_no_final_template_newline",
        "budget_policy": "single_reply_uses_remaining_total_token_capacity"
        if args.turn_policy == "first"
        else "reserve_later_prompt_chunks_one_sample_each_and_masked_intermediate_closures",
        "prefix_mask_policy": "nonthinking_prefix_is_unsupervised"
        if args.turn_policy == "first"
        else "preserve_existing_empty_think_prefix_mask_inserted_prefix_is_unsupervised",
    }


def artifact_name(args):
    policy = sampling_cache_key(
        **{key: args.policy[key] for key in ("temperature", "top_k", "top_p")}
    )
    replies = "firstAssistant" if args.turn_policy == "first" else "allAssistant"
    return f"{target_model_cache_id(args.target_model)}__nonthink__{policy}__minP0__{replies}__maxTotal{args.max_total_tokens}__seed{args.seed}__n1.pt"


def validate_part(payload, signature_hash, index, start, end):
    if (
        payload.get("signature") != signature_hash
        or payload.get("shard_index") != index
        or payload.get("source_start") != start
        or payload.get("source_end") != end
    ):
        raise ValueError(f"Mismatched durable shard {index}; refusing to mix generations")


def open_manifest(work, signature):
    path = work / "manifest.json"
    signature_hash = _digest(signature)
    if path.exists():
        manifest = _json(path)
        if (
            manifest.get("signature_hash") != signature_hash
            or manifest.get("signature") != signature
        ):
            raise ValueError(
                "Resume settings/source changed; use matching settings or a new --work-dir. "
                "Reply policy, shard size and other saved settings must match; "
                "incompatible generation shards cannot be mixed."
            )
        return manifest
    work.mkdir(parents=True, exist_ok=True)
    manifest = {"signature": signature, "signature_hash": signature_hash}
    _save_json(manifest, path)
    return manifest


def prepare_prompts(args, source, signature, work, manifest, layout, tokenizer=None):
    from tqdm import tqdm

    count = signature["source_rows"]
    number = (count + args.shard_size - 1) // args.shard_size
    paths = [work / "prompts" / f"part-{index:06d}.pt" for index in range(number)]
    # Once preparation is durable, restarting generation need not load the
    # multi-gigabyte source cache (or tokenize the source JSONL) again.
    if manifest.get("prompts_complete") and all(path.is_file() for path in paths):
        return paths
    raw_json = signature.get("source_kind") == "jsonl"
    entries = None
    if not raw_json:
        print(f"Loading source cache for token-only prompt extraction: {source}", flush=True)
        entries = _load_torch(source)
        if not isinstance(entries, list) or len(entries) != count:
            raise ValueError("Source cache row count does not match its metadata")
    total, rejected = 0, 0
    with ExitStack() as stack:
        lines = stack.enter_context(source.open(encoding="utf-8")) if raw_json else None
        pool = None
        if raw_json and args.num_proc > 1:
            pool = stack.enter_context(
                mp.get_context("spawn").Pool(
                    args.num_proc,
                    initializer=_init_json_worker,
                    initargs=(str(args.target_model), layout, args.max_total_tokens, args.turn_policy),
                )
            )
        for index, path in enumerate(
            tqdm(
                paths, desc="Preparing retained user turns",
                disable=args.disable_progress,
            )
        ):
            start, end = index * args.shard_size, min((index + 1) * args.shard_size, count)
            tasks = None
            if raw_json:
                tasks = list(zip(range(start, end), islice(lines, end - start), strict=True))
            if path.exists():
                payload = _load_torch(path)
                validate_part(payload, manifest["signature_hash"], index, start, end)
            else:
                records, errors = [], []
                if raw_json:
                    results = (
                        pool.imap(_extract_json_line, tasks, chunksize=8)
                        if pool
                        else (
                            _extract_json_line(
                                task, tokenizer, layout, args.max_total_tokens, args.turn_policy
                            )
                            for task in tasks
                        )
                    )
                    for record, error in results:
                        if error is not None:
                            errors.append(error)
                        else:
                            records.append(record)
                else:
                    for ordinal in range(start, end):
                        try:
                            records.append(
                                extract_conversation(
                                    entries[ordinal], ordinal, layout, args.max_total_tokens,
                                    turn_policy=args.turn_policy,
                                )
                            )
                        except (ValueError, KeyError, TypeError) as exc:
                            errors.append(
                                {
                                    "conversation_index": ordinal,
                                    "reason": str(exc),
                                }
                            )
                if errors and args.invalid_source == "error":
                    error = errors[0]
                    raise ValueError(
                        f"Cannot reconstruct conversation index {error['conversation_index']}: "
                        f"{error['reason']}. Inspect it or explicitly use --invalid-source skip."
                    )
                payload = {
                    "signature": manifest["signature_hash"],
                    "shard_index": index,
                    "source_start": start,
                    "source_end": end,
                    "records": records,
                    "rejected": errors,
                }
                _save_torch(payload, path)
            if payload["rejected"]:
                # Also recover a kill between publishing .pt and its audit sidecar.
                _save_json({"rejected": payload["rejected"]}, path.with_suffix(".rejected.json"))
            total += len(payload["records"])
            rejected += len(payload["rejected"])
    if _file_stamp(source) != signature["source_stamp"]:
        raise ValueError("Source changed during extraction")
    if total == 0:
        raise ValueError("No eligible conversations remain after prompt extraction")
    manifest.update(
        prompts_complete=True, eligible_conversations=total, rejected_conversations=rejected
    )
    _save_json(manifest, work / "manifest.json")
    del entries
    gc.collect()
    print(f"Prepared {total:,}/{count:,} conversations; rejected={rejected:,}", flush=True)
    return paths


def sampling_params(record, state, args, stop_ids, factory):
    _, budget = next_prompt(record, state, args.max_total_tokens)
    return factory(
        n=1,
        temperature=args.policy["temperature"],
        top_p=args.policy["top_p"],
        top_k=args.policy["top_k"],
        min_p=0.0,
        seed=args.seed + record["conversation_index"],
        max_tokens=budget,
        stop_token_ids=stop_ids,
        detokenize=False,
    )


def generate_part(args, payload, path, layout, engine_factory, params_factory):
    """Persist a whole turn wave before submitting any subsequent turn.

    If interrupted inside generate(), only that unfinished wave is replayed.
    Earlier sampled replies, including those for still-open conversations, are
    restored from disk. No response text is ever decoded or tokenized again.
    """
    records = payload["records"]
    if path.exists():
        result = _load_torch(path)
        validate_part(
            result,
            payload["signature"],
            payload["shard_index"],
            payload["source_start"],
            payload["source_end"],
        )
        if result.get("complete"):
            return result
        states = result["states"]
    else:
        states = [initial_state(record) for record in records]
    if len(states) != len(records) or any(
        state["conversation_index"] != record["conversation_index"]
        for record, state in zip(records, states, strict=True)
    ):
        raise ValueError("Rollout state does not correspond to the durable prompt shard")
    while True:
        active = [
            index
            for index, (record, state) in enumerate(zip(records, states, strict=True))
            if state["next_turn"] < len(record["turns"])
        ]
        if not active:
            break
        params = [
            sampling_params(records[index], states[index], args, layout.stop_ids, params_factory)
            for index in active
        ]
        prompts = [
            {
                "prompt_token_ids": next_prompt(
                    records[index], states[index], args.max_total_tokens
                )[0]
            }
            for index in active
        ]
        outputs = engine_factory().generate(
            prompts, sampling_params=params, use_tqdm=not args.disable_progress
        )
        if len(outputs) != len(active):
            raise RuntimeError("vLLM returned the wrong number of completions")
        for index, output in zip(active, outputs, strict=True):
            states[index] = accept_completion(
                records[index], states[index], output, layout, args.max_total_tokens
            )
        _save_torch(
            {
                key: payload[key]
                for key in ("signature", "shard_index", "source_start", "source_end")
            }
            | {"states": states, "complete": False},
            path,
        )
    stats = Counter()
    for record, state in zip(records, states, strict=True):
        stats.update(
            {
                "conversations": 1,
                "assistant_turns": len(record["turns"]),
                "tokens": len(state["tokens"]),
                "sampled_tokens": state["sampled_tokens"],
                "inserted_nonthinking_prefixes": record["inserted_nonthinking_prefixes"],
            }
        )
        stats.update(
            {
                f"finish_{reason}": state["finish_reasons"].count(reason)
                for reason in set(state["finish_reasons"])
            }
        )
    result = {
        key: payload[key] for key in ("signature", "shard_index", "source_start", "source_end")
    }
    result.update(
        complete=True,
        entries=[
            cache_entry(record, state, args.seed)
            for record, state in zip(records, states, strict=True)
        ],
        statistics=dict(stats),
    )
    _save_torch(result, path)
    return result


def assemble_cache(paths, output, signature, signature_hash, manifest):
    from tqdm import tqdm

    entries, statistics = [], Counter()
    for index, path in enumerate(tqdm(
        paths, desc="Assembling AngelSpec cache", unit="shard",
    )):
        payload = _load_torch(path)
        validate_part(
            payload,
            signature_hash,
            index,
            index * signature["shard_size"],
            min((index + 1) * signature["shard_size"], signature["source_rows"]),
        )
        if not payload.get("complete"):
            raise RuntimeError(f"Unfinished rollout shard: {path}")
        entries.extend(payload["entries"])
        statistics.update(payload["statistics"])
    if not entries or len(entries) != manifest["eligible_conversations"]:
        raise RuntimeError("Final cache count differs from validated prompt count")
    # A crash here never publishes a partial .pt. Shards remain available for
    # reassembly even if the final file is durable but its sidecar is not yet.
    _save_torch(entries, output)
    _save_json(
        {
            **signature,
            "signature_hash": signature_hash,
            "status": "complete",
            "artifact_name": output.name,
            "cached_samples": len(entries),
            "cached_tokens": statistics["tokens"],
            "statistics": dict(statistics),
            "rejected_conversations": manifest["rejected_conversations"],
            "output_bytes": output.stat().st_size,
        },
        output.with_suffix(".pt.json"),
    )
    return len(entries)


def run(args):
    from transformers import AutoTokenizer

    from angelspec.config.edr import resolve_edr_stop_token_ids

    if args.source_cache is not None:
        source, metadata = resolve_source(args.source_cache)
    else:
        source, metadata = args.source, None
        with source.open(encoding="utf-8") as handle:
            args.source_rows = sum(1 for _ in handle)
        if args.source_rows == 0:
            raise ValueError("Source JSONL is empty")
    output = args.cache_dir / "tokenized_dataset" / artifact_name(args)
    if source == output or source.parent == output.parent:
        raise ValueError("Output must not replace or share the source tokenized_dataset")
    other_outputs = [path for path in output.parent.glob("*.pt") if path != output]
    if other_outputs:
        raise ValueError(
            f"Destination already has another .pt cache: {other_outputs[0]}; use a separate --cache-dir"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.target_model), local_files_only=True, trust_remote_code=True
    )
    layout = QwenLayout.from_tokenizer(
        tokenizer, resolve_edr_stop_token_ids(str(args.target_model), local_files_only=True)
    )
    signature = make_signature(args, source, metadata, layout)
    signature_hash = _digest(signature)
    work = (
        _local_path(args.work_dir)
        if args.work_dir
        else args.cache_dir / "target_rollout_work" / output.stem
    )
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another generator owns {work}") from exc
        manifest_path = work / "manifest.json"
        if not manifest_path.exists() and (
            output.exists() or output.with_suffix(".pt.json").exists()
        ):
            raise ValueError(
                "Output already exists without this work manifest; refusing to overwrite"
            )
        manifest = open_manifest(work, signature)
        sidecar = output.with_suffix(".pt.json")
        if output.exists() and sidecar.exists():
            complete = _json(sidecar)
            if (
                complete.get("signature_hash") != signature_hash
                or complete.get("status") != "complete"
                or complete.get("output_bytes") != output.stat().st_size
            ):
                raise ValueError("Existing final cache does not match this generation")
            print(f"Already complete: {output} ({complete['cached_samples']:,} conversations)")
            return output
        print(
            f"Epoch-{args.epoch} regeneration\n  source: {source}\n  target: {args.target_model}\n  turn policy: {args.turn_policy}\n  policy: {args.policy}\n  seed: {args.seed} + zero-based source conversation index\n  limit: {args.max_total_tokens} total tokens; no separate response cap\n  shard size: {args.shard_size}\n  work: {work}\n  output: {output}",
            flush=True,
        )
        prompt_paths = prepare_prompts(args, source, signature, work, manifest, layout, tokenizer)
        if args.prepare_only:
            return None
        from tqdm import tqdm
        from vllm import LLM, SamplingParams

        from angelspec.utils.sampling import validate_vllm_sampling_parameters

        engine = None

        def get_engine():
            nonlocal engine
            if engine is None:
                attention_kwargs = (
                    {} if args.attention_backend == "AUTO"
                    else {"attention_backend": args.attention_backend}
                )
                print(f"vLLM attention backend request: {args.attention_backend}", flush=True)
                engine = LLM(
                    model=str(args.target_model),
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    generation_config="vllm",
                    seed=args.seed,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_total_tokens,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    max_num_seqs=args.max_num_seqs,
                    enforce_eager=args.enforce_eager,
                    disable_log_stats=True,
                    **attention_kwargs,
                )
            return engine

        def params_factory(**kwargs):
            params = SamplingParams(**kwargs)
            validate_vllm_sampling_parameters(
                params, **{key: args.policy[key] for key in ("temperature", "top_k", "top_p")}
            )
            return params

        rollouts = [work / "rollouts" / path.name for path in prompt_paths]
        completed = 0
        for index, path in enumerate(rollouts):
            if path.exists():
                payload = _load_torch(path)
                validate_part(
                    payload,
                    signature_hash,
                    index,
                    index * args.shard_size,
                    min((index + 1) * args.shard_size, signature["source_rows"]),
                )
                if payload.get("complete"):
                    completed += len(payload["entries"])
        print(
            f"Resuming {completed:,}/{manifest['eligible_conversations']:,} completed conversations",
            flush=True,
        )
        try:
            with tqdm(
                total=manifest["eligible_conversations"],
                initial=completed,
                desc="Regenerating conversations",
                unit="conversation",
                disable=args.disable_progress,
            ) as progress:
                for index, (prompt, rollout) in enumerate(
                    zip(prompt_paths, rollouts, strict=True)
                ):
                    if rollout.exists() and _load_torch(rollout).get("complete"):
                        continue
                    payload = _load_torch(prompt)
                    validate_part(
                        payload,
                        signature_hash,
                        index,
                        index * args.shard_size,
                        min((index + 1) * args.shard_size, signature["source_rows"]),
                    )
                    result = generate_part(
                        args, payload, rollout, layout, get_engine, params_factory
                    )
                    progress.update(len(result["entries"]))
        finally:
            if engine is not None:
                _shutdown_vllm(engine)
                del engine
                gc.collect()
        count = assemble_cache(rollouts, output, signature, signature_hash, manifest)
        manifest.update(status="complete", cached_samples=count)
        _save_json(manifest, manifest_path)
        print(f"Complete: {count:,} conversations in {output}", flush=True)
        return output


def _shutdown_vllm(engine):
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        shutdown = getattr(getattr(engine, "llm_engine", None), "shutdown", None)
    if callable(shutdown):
        shutdown()


def main(argv=None):
    parser, args = parse_args(argv)
    validate_args(args, parser)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
