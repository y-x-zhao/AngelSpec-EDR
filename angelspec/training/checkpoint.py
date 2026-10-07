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

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from angelspec.utils.logging import logger

HF_EXPORT_WEIGHTS_NAME = "pytorch_model.bin"


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None
        )


def _bf16_optimizer_state_for_load(optimizer, checkpoint_dir: Path) -> dict:
    """Return an optimizer state dict with the Adam slots listed in the checkpoint.

    DCP loads only keys present in the destination state dict, and a fresh AdamW
    has no slots. The BF16 wrapper keys state by integer parameter ID rather than
    by the model FQNs of ``get_state_dict``. Slots are allocated from the saved
    metadata, so parameters without saved state stay empty. Moment tensors follow
    each master parameter's placement, including DTensor shards; scalar Adam
    steps are loaded on CPU.
    """
    state_dict = optimizer.state_dict()
    is_muon = getattr(optimizer, "optimizer_type", "adamw") == "muon"
    adam_state = state_dict["adamw"] if is_muon else state_dict
    adam_state["state"] = {}
    params = {
        param_id: param
        for saved_group, group in zip(adam_state["param_groups"], optimizer.param_groups)
        for param_id, param in zip(saved_group["params"], group["params"])
    }
    prefix = "optim_state.optim." + ("adamw." if is_muon else "") + "state."
    metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
    for key, value in metadata.state_dict_metadata.items():
        if not key.startswith(prefix):
            continue
        param_id_text, slot = key[len(prefix) :].split(".", 1)
        param_id = int(param_id_text)
        if param_id not in params:
            raise ValueError(f"Checkpoint Adam parameter {param_id} is not in the current optimizer")
        if slot not in {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            raise ValueError(f"Unsupported checkpoint Adam state: {key}")
        if isinstance(value, TensorStorageMetadata):
            if slot == "step":
                tensor = torch.empty(value.size, dtype=value.properties.dtype, device="cpu")
            else:
                param = params[param_id]
                if value.size != param.shape:
                    raise ValueError(
                        f"Checkpoint Adam shape mismatch for {key}: "
                        f"saved {tuple(value.size)}, current {tuple(param.shape)}"
                    )
                tensor = torch.empty_like(param, dtype=value.properties.dtype)
        elif slot == "step":
            tensor = 0  # The Adam step may be saved as a Python scalar.
        else:
            raise ValueError(f"Checkpoint Adam moment is not a tensor: {key}")
        adam_state["state"].setdefault(param_id, {})[slot] = tensor
    for param_id, slots in adam_state["state"].items():
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(slots):
            raise ValueError(f"Incomplete checkpoint Adam state for parameter {param_id}")
    return state_dict


class OptimizerState(Stateful):
    """Wrapper for optimizer + fp32 master params."""

    def __init__(self, model, optimizer, *, checkpoint_dir: Path | None = None):
        self.model = model
        self.optimizer = optimizer
        self._is_bf16_optimizer = hasattr(optimizer, "sync_fp32_params_from_model")
        self._load_state = (
            _bf16_optimizer_state_for_load(optimizer, checkpoint_dir)
            if self._is_bf16_optimizer and checkpoint_dir is not None
            else None
        )

    def state_dict(self):
        if self._is_bf16_optimizer:
            return {
                "optim": (
                    self._load_state if self._load_state is not None else self.optimizer.state_dict()
                ),
                "fp32_params": {str(i): p.data for i, p in enumerate(self.optimizer.fp32_params)},
            }
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=[self.optimizer])
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        if self._is_bf16_optimizer:
            self.optimizer.load_state_dict(state_dict["optim"])
            if "fp32_params" in state_dict:
                with torch.no_grad():
                    for i, mp in enumerate(self.optimizer.fp32_params):
                        mp.data.copy_(state_dict["fp32_params"][str(i)])
                logger.info("Restored fp32 master params from checkpoint")
            return
        set_state_dict(
            self.model,
            optimizers=[self.optimizer],
            model_state_dict=None,
            optim_state_dict=state_dict["optim"],
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler, *, override_config: bool = False):
        self.lr_scheduler = lr_scheduler
        self._configured_policy = None
        self._configured_groups = None
        if override_config:
            from angelspec.training.lr_scheduler import LRSchedulerWithWarmup

            if not isinstance(lr_scheduler, LRSchedulerWithWarmup):
                raise TypeError("override_lr_scheduler requires LRSchedulerWithWarmup")
            # Capture before optimizer loading, which overwrites each group's
            # LR/initial_lr and optional max_lr/min_lr with the saved values.
            policy_keys = (
                "max_lr", "min_lr", "init_lr", "warmup_steps", "total_steps",
                "decay_style", "wsd_decay_steps", "wsd_decay_style", "base_lrs",
            )
            self._configured_policy = {
                key: copy.deepcopy(getattr(lr_scheduler, key)) for key in policy_keys
            }
            self._configured_groups = [
                {key: copy.deepcopy(group[key]) for key in ("initial_lr", "max_lr", "min_lr") if key in group}
                for group in lr_scheduler.optimizer.param_groups
            ]

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        if self._configured_policy is None:
            return
        # Keep the checkpoint's step counters (last_epoch/_step_count) and apply
        # the configured policy at that step.
        self.lr_scheduler.load_state_dict(self._configured_policy)
        groups = self.lr_scheduler.optimizer.param_groups
        if len(groups) != len(self._configured_groups):
            raise ValueError("Cannot override LR schedule: optimizer group count changed")
        for group, configured in zip(groups, self._configured_groups, strict=True):
            for key in ("initial_lr", "max_lr", "min_lr"):
                if key in configured:
                    group[key] = configured[key]
                else:
                    group.pop(key, None)
        # Loading scheduler state leaves optimizer.param_groups as restored from
        # the optimizer checkpoint; set the LR so the first update uses the policy.
        for group, lr in zip(groups, self.lr_scheduler.get_lr(), strict=True):
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(lr)
            else:
                group["lr"] = lr
        self.lr_scheduler._last_lr = [group["lr"] for group in groups]
        logger.info(
            "Overrode checkpoint LR policy from config: style=%s, lr=%s, "
            "scheduler_step=%d, total_steps=%d, warmup_steps=%d; optimizer state preserved",
            self.lr_scheduler.decay_style, self.lr_scheduler.get_last_lr(),
            self.lr_scheduler.last_epoch, self.lr_scheduler.total_steps,
            self.lr_scheduler.warmup_steps,
        )


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _resolve_hf_export_weights(
    load_path: str | Path | None, cache_dir: str | None = None
) -> Path | None:
    """Resolve a local or Hub-hosted Hugging Face ``pytorch_model.bin`` export.

    Returns None for DCP checkpoint roots and for local directories without an
    export file; those are handled by the DCP loader.
    """
    if load_path is None:
        return None

    raw_path = str(load_path)
    local_path = Path(raw_path).expanduser()
    if local_path.is_dir():
        tracker_path = local_path / "latest_checkpointed_iteration.txt"
        has_iteration_dir = any(
            child.is_dir() and child.name.startswith("iter_") for child in local_path.iterdir()
        )
        if tracker_path.is_file() or has_iteration_dir:
            return None
        weights_path = local_path / HF_EXPORT_WEIGHTS_NAME
        return weights_path if weights_path.is_file() else None
    if local_path.is_file():
        return local_path if local_path.suffix == ".bin" else None

    # Missing absolute, "./" or "~" paths and .bin names are local paths for the
    # DCP loader, not Hub repository IDs.
    if local_path.is_absolute() or raw_path.startswith((".", "~")) or local_path.suffix == ".bin":
        return None

    try:
        validate_repo_id(raw_path)
    except HFValidationError:
        return None

    resolved = hf_hub_download(
        repo_id=raw_path,
        filename=HF_EXPORT_WEIGHTS_NAME,
        cache_dir=cache_dir,
    )
    return Path(resolved)


def load_hf_export(
    draft_model: torch.nn.Module,
    load_path: str | Path | None,
    cache_dir: str | None = None,
) -> bool:
    """Strictly preload draft weights from a Hugging Face export when present.

    The caller is responsible for invoking this only on rank 0 and broadcasting
    the resulting model state. ``False`` means the path was not an HF export and
    should be passed to the normal DCP loader.
    """
    weights_path = _resolve_hf_export_weights(load_path, cache_dir=cache_dir)
    if weights_path is None:
        return False

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Hugging Face export {weights_path} did not contain a model state dictionary"
        )
    draft_model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded Hugging Face draft weights from {weights_path}")
    return True


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Normal training resume requires the actor's optimizer and scheduler state.
    ``continual_training=True`` explicitly requests model initialization instead.
    """
    load_root = getattr(actor.args, "load_path", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"Model checkpoint {model_dir} not found; skipping load.")
        return None

    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    continual_training = getattr(actor.args, "continual_training", False)

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"Loaded model from {model_dir}")
    except (Exception, CheckpointException) as e:
        if not continual_training:
            raise RuntimeError(f"Failed to resume model from {model_dir}") from e
        # Continual training keeps the fresh initialization after a local load
        # error. CheckpointException reports a distributed failure and is re-raised.
        if isinstance(e, CheckpointException):
            raise
        logger.error(f"Failed to load model from {model_dir}: {e}")
        return None

    # Capture the configured LR policy before optimizer.load_state_dict can
    # replace the freshly configured optimizer-group LR metadata.
    load_lr_scheduler = (
        not continual_training and getattr(actor, "lr_scheduler", None) is not None
    )
    lr_scheduler_state = (
        LRSchedulerState(
            actor.lr_scheduler,
            override_config=getattr(actor.args, "override_lr_scheduler", False),
        )
        if load_lr_scheduler else None
    )

    # Keep optimizer/LR state out of continual training so it starts fresh.
    load_optimizer = not continual_training and getattr(actor, "optimizer", None) is not None
    if load_optimizer:
        if not (optimizer_dir / ".metadata").is_file():
            raise FileNotFoundError(f"Optimizer checkpoint required for resume: {optimizer_dir}")
        try:
            optimizer_state = OptimizerState(
                actor.model, actor.optimizer, checkpoint_dir=optimizer_dir
            )
            optim_state_dict = {"optim_state": optimizer_state}
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"Loaded optimizer from {optimizer_dir}")
        except (Exception, CheckpointException) as e:
            raise RuntimeError(f"Failed to resume optimizer from {optimizer_dir}") from e

    # Resume requires the scheduler checkpoint. override_lr_scheduler replaces
    # only its policy; counters and optimizer state are restored.
    if load_lr_scheduler:
        if not (lr_scheduler_dir / ".metadata").is_file():
            raise FileNotFoundError(f"LR scheduler checkpoint required for resume: {lr_scheduler_dir}")
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"Loaded LR scheduler from {lr_scheduler_dir}")
        except (Exception, CheckpointException) as e:
            raise RuntimeError(f"Failed to resume LR scheduler from {lr_scheduler_dir}") from e

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
        "optimizer_dir": optimizer_dir,
    }


def _restore_fp32_master_params(actor: Any, optim_dir: Path) -> None:
    """Sync BF16Optimizer's fp32 master params after model-only checkpoint load.

    When optimizer state is skipped for continual training, the fp32 master copies
    still hold pre-checkpoint (random init) values.  The first optimizer step
    would copy these back to the model, overwriting the loaded weights.

    Strategy: temporarily load the optimizer checkpoint to recover the fp32 master
    params, then restore the freshly configured param_groups and clear Adam state
    so continual training still uses the new optimizer hyperparameters. Falls
    back to copying from the bf16 model weights if the optimizer checkpoint is
    unavailable.
    """
    opt = actor.optimizer
    if not hasattr(opt, "fp32_params"):
        return

    if optim_dir.exists() and (optim_dir / ".metadata").exists():
        try:
            fresh_param_groups = [
                {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
                for group in opt.optimizer.param_groups
            ]
            # Continual training only needs the masters, not newly allocated Adam slots.
            optim_state = OptimizerState(actor.model, opt)
            optim_sd = {"optim_state": optim_state}
            dcp.load(state_dict=optim_sd, checkpoint_id=str(optim_dir))
            for group, fresh_group in zip(opt.optimizer.param_groups, fresh_param_groups):
                params = group["params"]
                group.clear()
                group.update(copy.deepcopy(fresh_group))
                group["params"] = params
            opt.optimizer.state.clear()
            logger.info(f"Loaded fp32 master params from {optim_dir}")
            return
        except (Exception, CheckpointException) as e:
            logger.warning(f"Failed to load fp32 params from optimizer checkpoint: {e}")

    if hasattr(opt, "sync_fp32_params_from_model"):
        opt.sync_fp32_params_from_model()
        logger.info("Synced optimizer fp32 master params from bf16 model weights (lossy)")


def _restore_cuda_rng_state(saved_states: list[torch.Tensor], seed: int) -> None:
    """Restore CUDA RNG states by visible-device index; seed additional devices.

    Checkpoints store rank 0's visible-device RNG list. With the same device count
    all states are restored. Otherwise the common prefix is restored by device
    index and additional devices are seeded with the training seed.
    """
    device_count = torch.cuda.device_count()
    if len(saved_states) == device_count:
        torch.cuda.set_rng_state_all(saved_states)
        return
    logger.warning(
        f"CUDA RNG device count changed from {len(saved_states)} to {device_count}; "
        "restoring matching visible-device indices and seeding additional devices "
        f"with training seed {seed}. Exact RNG continuity is not guaranteed."
    )
    torch.cuda.set_rng_state_all(saved_states[:device_count])
    for device_index in range(len(saved_states), device_count):
        with torch.cuda.device(device_index):
            torch.cuda.manual_seed(seed)


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    continual_training = getattr(actor.args, "continual_training", False)

    if checkpoint_payload.get("rng") is not None and not continual_training:
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            _restore_cuda_rng_state(rng_state["cuda"], seed=getattr(actor.args, "seed", 42))

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata and not continual_training:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        next_step = metadata.get("next_step") or metadata.get("next_inference_id")
        if next_step is not None:
            actor.args.start_step = next_step
    elif iteration is not None and not continual_training:
        if getattr(actor.args, "start_step", None) is None:
            actor.args.start_step = iteration

    if continual_training and hasattr(actor, "optimizer"):
        _restore_fp32_master_params(actor, checkpoint_payload["optimizer_dir"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, step: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    base_dir = Path(actor.args.checkpoint_dir).expanduser()
    step_id = step + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    if hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    if hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "step": step,
            "inference_id": step,  # compat: old checkpoints use this key
            "next_step": step + 1,
            "next_inference_id": step + 1,  # compat: old checkpoints use this key
            "global_step": actor.global_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
