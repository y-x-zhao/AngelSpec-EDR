"""Convert a local published Qwen3 DSpark checkpoint to AngelSpec on CPU.

No model is downloaded or instantiated. The three renamed weights follow the
Qwen3DSparkModel forward in deepseek-ai/DeepSpec at
005e03b81cec38b7da6399833d609ee89a2587f2:
``hidden_norm(fc(target_hidden_states))`` and ``norm(draft_hidden_states)``.
The checkpoint's frozen LM head is omitted only after an exact comparison with
the local target's LM head (or its embedding when the target ties these weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT_CONFIG = ROOT / "angelspec/config/dspark_qwen3_4b_draft_config.json"
WEIGHTS_NAME = "pytorch_model.bin"
MANIFEST_NAME = "import_manifest.json"
IMPORT_FORMAT = "angelspec-dspark-import-v1"
RENAMED_WEIGHTS = {
    "fc.weight": "context_proj.weight",
    "hidden_norm.weight": "context_norm.weight",
    "norm.weight": "final_norm.weight",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require(config: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if key not in config or config[key] != expected:
        raise ValueError(f"{label}: {key} must be {expected!r}, got {config.get(key)!r}")


def _rope_theta(config: Mapping[str, Any], label: str) -> float:
    parameters = config.get("rope_parameters")
    if parameters is not None:
        if not isinstance(parameters, dict) or parameters.get("rope_type") != "default":
            raise ValueError(f"{label}: only default RoPE is supported")
        if set(parameters) - {"rope_type", "rope_theta"}:
            raise ValueError(f"{label}: unsupported RoPE parameters")
        theta = parameters.get("rope_theta")
        if config.get("rope_theta", theta) != theta:
            raise ValueError(f"{label}: inconsistent rope_theta and rope_parameters")
    else:
        theta = config.get("rope_theta")
    if config.get("rope_scaling") is not None:
        raise ValueError(f"{label}: scaled RoPE is not supported")
    if not isinstance(theta, (int, float)) or theta <= 0:
        raise ValueError(f"{label}: missing or invalid rope_theta")
    return float(theta)


def validate_configs(
    source: Mapping[str, Any], target: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject architectural changes that a weight rename cannot represent."""
    _require(expected, "model_type", "dspark", "AngelSpec config")
    _require(expected, "architectures", ["DSparkDraftModel"], "AngelSpec config")
    _require(source, "model_type", "qwen3", "Published draft")
    _require(source, "architectures", ["Qwen3DSparkModel"], "Published draft")
    _require(source, "block_size", 7, "Published draft")
    _require(source, "num_anchors", 512, "Published draft")
    _require(target, "model_type", "qwen3", "Target")
    _require(target, "architectures", ["Qwen3ForCausalLM"], "Target")
    common_fields = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
        "max_position_embeddings",
    )
    draft_fields = (
        "num_hidden_layers",
        "target_layer_ids",
        "mask_token_id",
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "tie_word_embeddings",
    )
    for name in common_fields:
        _require(source, name, expected[name], "Published draft")
        _require(target, name, expected[name], "Target")
    for name in draft_fields:
        _require(source, name, expected[name], "Published draft")
    _require(source, "num_target_layers", expected["target_num_hidden_layers"], "Published draft")
    _require(target, "num_hidden_layers", expected["target_num_hidden_layers"], "Target")
    _require(expected, "target_hidden_size", expected["hidden_size"], "AngelSpec config")
    _require(expected, "num_target_layers", len(expected["target_layer_ids"]), "AngelSpec config")
    _require(expected, "markov_head_type", "vanilla", "AngelSpec config")
    if expected.get("fusion_type", "concat_fc") != "concat_fc":
        raise ValueError("AngelSpec config: only concat_fc fusion is supported")
    for label, config in (("Published draft", source), ("Target", target)):
        _require(config, "hidden_act", "silu", label)
        _require(config, "attention_bias", False, label)
        _require(config, "attention_dropout", 0.0, label)
        if config.get("use_sliding_window", False) or config.get("sliding_window") is not None:
            raise ValueError(f"{label}: sliding-window attention is not supported")
        layer_types = config.get("layer_types")
        if (
            layer_types is not None
            and layer_types != ["full_attention"] * config["num_hidden_layers"]
        ):
            raise ValueError(f"{label}: expected full attention in every layer")
        if _rope_theta(config, label) != float(expected["rope_theta"]):
            raise ValueError(f"{label}: rope_theta does not match the AngelSpec config")
        dtype = config.get("dtype", config.get("torch_dtype"))
        if dtype != "bfloat16":
            raise ValueError(f"{label}: expected bfloat16 checkpoint, got {dtype!r}")
    if not isinstance(target.get("tie_word_embeddings"), bool):
        raise ValueError("Target: tie_word_embeddings must be explicit")
    converted = dict(expected)
    converted.update(block_size=7, num_anchors=512, dtype="bfloat16")
    return converted


def expected_weight_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the DSpark parameter shapes that DSparkDraftModel builds from ``config``.

    Defined here because importing the training package can initialize CUDA,
    which a CPU-only conversion node may lack.
    """
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    rank, head = config["markov_rank"], config["head_dim"]
    shapes = {
        "context_proj.weight": (
            hidden,
            config["num_target_layers"] * config["target_hidden_size"],
        ),
        "context_norm.weight": (hidden,),
        "embed_tokens.weight": (vocab, hidden),
        "final_norm.weight": (hidden,),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"layers.{layer}."
        for name in ("q", "k", "v"):
            heads = config["num_attention_heads" if name == "q" else "num_key_value_heads"]
            shapes[f"{prefix}self_attn.{name}_proj.weight"] = (heads * head, hidden)
        shapes[f"{prefix}self_attn.o_proj.weight"] = (hidden, config["num_attention_heads"] * head)
        for name in ("q", "k"):
            shapes[f"{prefix}self_attn.{name}_norm.weight"] = (head,)
        for name in ("gate", "up"):
            shapes[f"{prefix}mlp.{name}_proj.weight"] = (config["intermediate_size"], hidden)
        shapes[f"{prefix}mlp.down_proj.weight"] = (hidden, config["intermediate_size"])
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{prefix}{name}.weight"] = (hidden,)
    if rank > 0:
        shapes["markov_head.markov_w1.weight"] = (vocab, rank)
        shapes["markov_head.markov_w2.weight"] = (vocab, rank)
    if config["enable_confidence_head"]:
        width = hidden + (rank if config["confidence_head_with_markov"] else 0)
        shapes["confidence_head.proj.weight"] = (1, width)
        shapes["confidence_head.proj.bias"] = (1,)
    return shapes


class _LocalSafeTensors:
    """Open only needed local safetensor shards; never load a target model."""

    def __init__(self, directory: Path, stack: ExitStack):
        self.directory = directory
        self.stack = stack
        self.open_files: dict[str, Any] = {}
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            weight_map = _read_json(index).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Invalid safetensors weight_map in {index}")
            self.weight_map = weight_map
            for filename in weight_map.values():
                if (
                    not isinstance(filename, str)
                    or Path(filename).name != filename
                    or not filename.endswith(".safetensors")
                ):
                    raise ValueError(f"Unsafe safetensors shard name in {index}")
        else:
            filename = "model.safetensors"
            handle = self._open(filename)
            self.weight_map = dict.fromkeys(handle.keys(), filename)

    def _open(self, filename: str):
        if filename not in self.open_files:
            self.open_files[filename] = self.stack.enter_context(
                safe_open(self.directory / filename, framework="pt", device="cpu")
            )
        return self.open_files[filename]

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise ValueError(f"Missing {key} in local checkpoint {self.directory}")
        return self._open(self.weight_map[key]).get_tensor(key)

    def validate_index(self) -> None:
        """Ensure a source index cannot hide an unexpected tensor in a shard."""
        actual = {}
        for filename in set(self.weight_map.values()):
            for key in self._open(filename).keys():
                if key in actual:
                    raise ValueError(f"Duplicate tensor {key} in published checkpoint shards")
                actual[key] = filename
        if actual != self.weight_map:
            raise ValueError(
                "Published safetensors index does not match the tensor keys in its shards"
            )


def _equal_rows(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Exact comparison without large temporary buffers or a GPU allocation."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return torch.equal(left, right)
    return all(
        torch.equal(left[start : start + 4096], right[start : start + 4096])
        for start in range(0, left.shape[0], 4096)
    )


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _validate_completed(
    output: Path, config: dict[str, Any], manifest: dict[str, Any], state: dict[str, torch.Tensor]
) -> None:
    if not output.is_dir() or not (output / MANIFEST_NAME).is_file():
        raise ValueError(f"Refusing to overwrite unrelated or incomplete output: {output}")
    if (
        _read_json(output / MANIFEST_NAME) != manifest
        or _read_json(output / "config.json") != config
    ):
        raise ValueError(
            f"Existing import does not match source, target, or configuration: {output}"
        )
    saved = torch.load(output / WEIGHTS_NAME, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or set(saved) != set(state):
        raise ValueError(f"Invalid converted state dictionary in {output}")
    for key, value in state.items():
        if not isinstance(saved[key], torch.Tensor) or not _equal_rows(saved[key], value):
            raise ValueError(f"Existing converted weight does not match source: {key}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def import_checkpoint(
    source: Path,
    target_model: Path,
    output_dir: Path,
    *,
    draft_config: Path = DEFAULT_DRAFT_CONFIG,
) -> Path:
    """Validate and atomically publish a CPU-only AngelSpec initialization.

    A repeat run verifies and reuses a completed identical import. An interrupted
    run leaves a separate staging directory; rerunning retries the conversion
    without modifying either input.
    """
    source, target_model, output_dir = (
        path.expanduser().resolve() for path in (source, target_model, output_dir)
    )
    for root in (source, target_model):
        if not root.is_dir():
            raise ValueError(f"Local checkpoint directory does not exist: {root}")
        if output_dir == root or root in output_dir.parents:
            raise ValueError(
                "--output-dir must be separate from the source and target directories"
            )
    source_config, target_config = (
        _read_json(source / "config.json"),
        _read_json(target_model / "config.json"),
    )
    converted_config = validate_configs(source_config, target_config, _read_json(draft_config))
    expected_shapes = expected_weight_shapes(converted_config)
    with ExitStack() as stack:
        published, target = (
            _LocalSafeTensors(source, stack),
            _LocalSafeTensors(target_model, stack),
        )
        published.validate_index()
        source_to_output = {
            key: RENAMED_WEIGHTS.get(key, key)
            for key in published.weight_map
            if key != "lm_head.weight"
        }
        actual_keys = list(source_to_output.values())
        if len(set(actual_keys)) != len(actual_keys):
            raise ValueError("Published checkpoint has duplicate weights after renaming")
        missing, extra = (
            set(expected_shapes) - set(actual_keys),
            set(actual_keys) - set(expected_shapes),
        )
        if missing or extra or "lm_head.weight" not in published.weight_map:
            raise ValueError(
                f"Published checkpoint keys mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}; frozen lm_head.weight is required"
            )
        state = {}
        for original, renamed in source_to_output.items():
            value = published.tensor(original)
            if tuple(value.shape) != expected_shapes[renamed]:
                raise ValueError(
                    f"Shape mismatch for {original}: got {tuple(value.shape)}, expected {expected_shapes[renamed]}"
                )
            if value.dtype != torch.bfloat16:
                raise ValueError(f"Expected bfloat16 weight for {original}, got {value.dtype}")
            state[renamed] = value
        embedding_key = "model.embed_tokens.weight"
        target_embedding = target.tensor(embedding_key)
        lm_head_key = "lm_head.weight"
        if target_config["tie_word_embeddings"]:
            # Qwen3-4B saves only the embedding, but some local exports store both.
            if lm_head_key in target.weight_map and not _equal_rows(
                target.tensor(lm_head_key), target_embedding
            ):
                raise ValueError("Target declares tied embeddings, but its saved LM head differs")
            lm_head_key = embedding_key
        if not _equal_rows(state["embed_tokens.weight"], target_embedding):
            raise ValueError(
                "Published DSpark embed_tokens.weight differs from the local target; refusing a different initialization"
            )
        if not _equal_rows(published.tensor("lm_head.weight"), target.tensor(lm_head_key)):
            raise ValueError(
                "Published DSpark lm_head.weight differs from the local target; it cannot be discarded safely"
            )
        manifest = {
            "format": IMPORT_FORMAT,
            "source_config_sha256": _json_fingerprint(source_config),
            "target_config_sha256": _json_fingerprint(target_config),
            "renamed_weights": RENAMED_WEIGHTS,
            "omitted_weight": "lm_head.weight",
            "target_embedding_key": embedding_key,
            "target_lm_head_key": lm_head_key,
            "frozen_weights_verified": "exact dtype and value equality",
            "converted_tensor_count": len(state),
        }
        if output_dir.exists():
            _validate_completed(output_dir, converted_config, manifest, state)
            print(f"Verified existing AngelSpec DSpark import: {output_dir}", flush=True)
            return output_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.", suffix=".incomplete", dir=output_dir.parent
            )
        )
        try:
            _write_json(stage / "config.json", converted_config)
            print(f"Writing {len(state)} CPU tensors to {stage / WEIGHTS_NAME}", flush=True)
            with (stage / WEIGHTS_NAME).open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            _write_json(stage / MANIFEST_NAME, manifest)
            _validate_completed(stage, converted_config, manifest, state)
            _fsync_directory(stage)
            if output_dir.exists():
                raise FileExistsError(
                    f"Output appeared while importing; refusing to overwrite {output_dir}"
                )
            stage.rename(output_dir)
            _fsync_directory(output_dir.parent)
        except BaseException:
            print(
                f"Import was not published. Recoverable staging directory: {stage}",
                file=sys.stderr,
                flush=True,
            )
            raise
    print(f"AngelSpec DSpark initialization ready: {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("./draft_checkpoints/dspark_qwen3_4b_block7")
    )
    parser.add_argument("--target-model", type=Path, default=Path("./target_models/Qwen3-4B"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./draft_checkpoints/dspark_qwen3_4b_block7-angelspec"),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="CPU tensor comparison threads (default: up to 8)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    try:
        import_checkpoint(args.source, args.target_model, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DSpark import failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
