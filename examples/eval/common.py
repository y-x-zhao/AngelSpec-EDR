"""Checkpoint resolution and HF target helpers shared by the offline DP evaluator."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_DRAFT_CONFIG = REPO_ROOT / "angelspec" / "config" / "dfly_qwen3_8b_draft_config.json"


_PRIMARY_WEIGHT_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
_AUXILIARY_WEIGHT_NAMES = {"mask_embedding.pt"}
_SAFE_CHECKPOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_weight_shard(name: str) -> bool:
    return (name.startswith("model-") and name.endswith(".safetensors")) or (
        name.startswith("pytorch_model-") and name.endswith(".bin")
    )


@dataclass(frozen=True)
class ResolvedCheckpoint:
    input_path: str
    source_path: Path
    kind: str


def _has_hf_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        child.is_file() and (child.name in _PRIMARY_WEIGHT_NAMES or _is_weight_shard(child.name))
        for child in path.iterdir()
    )


def _resolve_dcp_root(root: Path) -> Path:
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        iteration_dirs = sorted(path.name for path in root.glob("iter_*") if path.is_dir())
        suffix = f" Found: {', '.join(iteration_dirs)}." if iteration_dirs else ""
        raise FileNotFoundError(
            f"No latest-checkpoint tracker at {tracker}.{suffix} "
            "Pass a specific iter_XXXXXXX directory or restore the tracker."
        )
    raw_step = tracker.read_text(encoding="utf-8").strip()
    if not raw_step.isdigit():
        raise ValueError(f"Checkpoint tracker {tracker} must contain a non-negative integer")
    model_dir = root / f"iter_{int(raw_step):07d}" / "model"
    if not (model_dir / ".metadata").is_file():
        raise FileNotFoundError(
            f"Latest model checkpoint is incomplete: {model_dir / '.metadata'}"
        )
    return model_dir


def resolve_checkpoint(raw_path: str | os.PathLike[str]) -> ResolvedCheckpoint:
    """Resolve a direct HF export, final export, or AngelSpec DCP checkpoint."""
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Draft checkpoint does not exist: {path}")
    path = path.resolve()

    if path.is_file():
        if _is_weight_shard(path.name):
            raise ValueError(
                f"{path} is one shard of a checkpoint. Pass its parent directory so all "
                "shards and the index file are loaded together."
            )
        if path.name in _AUXILIARY_WEIGHT_NAMES:
            raise ValueError(f"{path} is an auxiliary tensor, not a complete draft checkpoint")
        if path.name.endswith((".bin", ".pt", ".safetensors")):
            return ResolvedCheckpoint(str(raw_path), path, "huggingface_export")
        raise ValueError(
            f"Unsupported checkpoint file {path}; expected .bin, .pt, or .safetensors"
        )

    # A completed serving export is preferred only when the caller passes its
    # parent training-output directory. DCP markers in one directory still win
    # over loose weights so resumable checkpoints are not misidentified.
    if _has_hf_weights(path / "hf_final"):
        return ResolvedCheckpoint(str(raw_path), (path / "hf_final").resolve(), "hf_final")
    if (path / "model" / ".metadata").is_file():
        return ResolvedCheckpoint(str(raw_path), (path / "model").resolve(), "dcp_iteration")
    if (path / ".metadata").is_file():
        return ResolvedCheckpoint(str(raw_path), path, "dcp_model")
    if (path / "latest_checkpointed_iteration.txt").is_file() or any(path.glob("iter_*")):
        return ResolvedCheckpoint(str(raw_path), _resolve_dcp_root(path), "dcp_latest")
    if _has_hf_weights(path):
        return ResolvedCheckpoint(str(raw_path), path, "huggingface_export")
    checkpoint_root = path / "checkpoints"
    if checkpoint_root.is_dir():
        return ResolvedCheckpoint(str(raw_path), _resolve_dcp_root(checkpoint_root), "dcp_latest")
    raise FileNotFoundError(
        f"Could not find model weights or an AngelSpec distributed checkpoint under {path}"
    )


def checkpoint_output_name(
    checkpoint: ResolvedCheckpoint,
    explicit_name: str | None = None,
) -> str:
    """Return a path-safe output namespace for one checkpoint."""
    if explicit_name is not None:
        name = explicit_name
    elif checkpoint.kind.startswith("dcp_"):
        model_dir = checkpoint.source_path
        name = model_dir.parent.name if model_dir.name == "model" else model_dir.name
    elif checkpoint.source_path.is_dir():
        name = checkpoint.source_path.name
    elif checkpoint.source_path.name in {"pytorch_model.bin", "model.safetensors"}:
        name = checkpoint.source_path.parent.name
    else:
        name = checkpoint.source_path.stem
    if not _SAFE_CHECKPOINT_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            f"Unsafe checkpoint output name {name!r}; use only letters, digits, '.', '_', and '-'"
        )
    return name


def _load_tensor_state(path: Path) -> dict[str, Any]:
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError(f"Draft checkpoint {path} did not contain a non-empty state dictionary")
    return state


def _load_export_into_model(model: Any, source: Path) -> None:
    if source.is_file():
        state = _load_tensor_state(source)
        model.load_state_dict(state, strict=True)
        del state
        return

    safe_index = source / "model.safetensors.index.json"
    bin_index = source / "pytorch_model.bin.index.json"
    if safe_index.is_file() or bin_index.is_file():
        from transformers.trainer_utils import load_sharded_checkpoint

        load_sharded_checkpoint(model, str(source), strict=True, prefer_safe=safe_index.is_file())
        return
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = source / filename
        if candidate.is_file():
            state = _load_tensor_state(candidate)
            model.load_state_dict(state, strict=True)
            del state
            return
    shards = sorted(path.name for path in source.iterdir() if _is_weight_shard(path.name))
    if shards:
        raise FileNotFoundError(
            f"Found weight shards under {source}, but no matching checkpoint index JSON"
        )
    raise FileNotFoundError(f"No supported draft weights found under {source}")


def load_draft_model(
    checkpoint: ResolvedCheckpoint,
    device: Any,
    config_path: Path = CANONICAL_DRAFT_CONFIG,
) -> tuple[Any, dict[str, Any]]:
    """Strictly load draft weights into the DSpark or DFly model described by ``config_path``."""
    import torch

    from angelspec.models.draft.dfly import DFlyConfig, DFlyDraftModel

    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if raw_config.get("model_type") == "dspark":
        from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel

        model = DSparkDraftModel(DSparkConfig(**raw_config))
    else:
        if int(raw_config.get("block_size", 0)) != 8 or not raw_config.get("dspark_bonus_anchor"):
            raise ValueError("Canonical Qwen3 DFly config must describe the released Block8 layout")
        model = DFlyDraftModel(DFlyConfig(**raw_config))
    if checkpoint.kind.startswith("dcp_"):
        from tools.convert_to_hf import _extract_model_weights, _load_fsdp_state_dict

        full_state = _load_fsdp_state_dict(str(checkpoint.source_path))
        state = _extract_model_weights(full_state)
        del full_state
        if not state:
            raise ValueError(
                f"No draft_model tensors found in DCP checkpoint {checkpoint.source_path}"
            )
        model.load_state_dict(state, strict=True)
        del state
    else:
        _load_export_into_model(model, checkpoint.source_path)
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, raw_config


class HFTargetRunner:
    """Capture only the target-layer residual streams consumed by the draft."""

    def __init__(self, model: Any, layer_ids: Sequence[int]):
        self.model = model
        self.layer_ids = tuple(int(index) for index in layer_ids)
        # Forward hooks run in the Python thread that invoked the model; per-thread
        # storage keeps concurrent forwards from overwriting each other's captures.
        self._thread_state = threading.local()
        self._handles = []
        layers = self._transformer_layers(model)
        for layer_id in self.layer_ids:
            if not 0 <= layer_id < len(layers):
                raise ValueError(
                    f"Target layer {layer_id} is out of bounds for {len(layers)} layers"
                )
            self._handles.append(layers[layer_id].register_forward_hook(self._hook(layer_id)))

    @staticmethod
    def _transformer_layers(model: Any) -> Any:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Could not locate target transformer layers for DFly hidden-state hooks")

    def _hook(self, layer_id: int):
        def capture(_module, _inputs, output):
            captured = getattr(self._thread_state, "captured", None)
            if captured is None:
                captured = {}
                self._thread_state.captured = captured
            captured[layer_id] = output[0] if isinstance(output, tuple) else output

        return capture

    def forward(self, **kwargs) -> tuple[Any, tuple[Any, ...]]:
        captured: dict[int, Any] = {}
        self._thread_state.captured = captured
        output = self.model(**kwargs)
        missing = [index for index in self.layer_ids if index not in captured]
        if missing:
            raise RuntimeError(f"Target forward did not capture DFly layers: {missing}")
        states = tuple(captured[index] for index in self.layer_ids)
        expected_length = int(kwargs["input_ids"].shape[1])
        if any(state.ndim != 3 or state.shape[1] != expected_length for state in states):
            raise RuntimeError(
                "Captured target hidden states do not match the target input length"
            )
        expected_device = kwargs["input_ids"].device
        if any(state.device != expected_device for state in states):
            raise RuntimeError("Captured target hidden states are not on the target input device")
        return output, states

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if hasattr(self._thread_state, "captured"):
            self._thread_state.captured.clear()


def _validate_target_geometry(config: Any, draft_config: dict[str, Any]) -> None:
    target = getattr(config, "text_config", config)
    expected = {
        "hidden_size": int(draft_config["target_hidden_size"]),
        "num_hidden_layers": int(draft_config["target_num_hidden_layers"]),
        "vocab_size": int(draft_config["vocab_size"]),
    }
    mismatches = {
        key: (getattr(target, key, None), value)
        for key, value in expected.items()
        if getattr(target, key, None) != value
    }
    if mismatches:
        raise ValueError(f"Target model is incompatible with the DFly checkpoint: {mismatches}")
