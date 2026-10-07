"""Single-GPU EDR and DFlash-family baseline training for Qwen3 DFly and DSpark drafts.

Runs in one process without Ray or Mooncake: a frozen HF target on the same GPU
computes the target features. Training reads the cached token IDs and response
masks; it does not regenerate responses. The recipe run.sh launchers set the
paper's global optimizer batch of 96.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tqdm import tqdm

from angelspec.config.distillation import configure_dflash_distillation
from angelspec.config.edr import configure_dflash_edr
from angelspec.config.train_config import config_to_flat_args, load_config, print_config
from angelspec.data.dataset import find_tokenized_cache_for_training, load_conversation_dataset
from angelspec.data.epoch_cache import EpochCachedDataset
from angelspec.data.utils import DataCollatorWithPadding, resolve_loss_mask
from angelspec.models.draft.auto import AutoDraftModelConfig
from angelspec.training.dfly_trainer import DFlyTrainer
from angelspec.training.dspark_trainer import DSparkTrainer
from angelspec.training.local_target import LocalTargetFeatures
from angelspec.training.schedule import auto_calculate_training_steps, epoch_cursor
from angelspec.utils.checkpoint_policy import _cleanup_old_checkpoints, _is_save_interval_step
from angelspec.utils.distributed import init_gloo_group
from angelspec.utils.logging import get_tb_writer, init_tracking, logger


def _local_row_group(args) -> int:
    key = (
        "dflash_edr_cross_row_batch_size"
        if getattr(args, "dflash_loss_objective", "decay") == "edr"
        else "dflash_distill_cross_row_batch_size"
    )
    return int(getattr(args, key))


def _local_target_prefill_limits(args, group: int) -> tuple[int, int]:
    """Return (target prefill rows, target token cap); the cap applies only across groups."""
    values = []
    for name in ("single_gpu_target_batch_size", "single_gpu_target_max_tokens"):
        value = getattr(args, name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        values.append(value)
    target_rows, max_tokens = values
    if group < 1 or args.draft_accumulation_steps < 1 or args.draft_accumulation_steps % group:
        raise ValueError("draft_accumulation_steps must be positive and divisible by cross-row group size")
    if target_rows and (
        target_rows % group or target_rows > min(96, args.draft_accumulation_steps)
    ):
        raise ValueError(
            "single_gpu_target_batch_size must be a multiple of the cross-row group "
            "and at most min(96, draft_accumulation_steps), or zero"
        )
    return target_rows or group, max_tokens


def validate_single_gpu_args(args, draft_config) -> None:
    """Raise on settings that single-GPU training does not support and set derived batch fields."""
    from angelspec.models.draft.dfly import DFlyConfig
    from angelspec.models.draft.dspark import DSparkConfig

    if not isinstance(draft_config, (DFlyConfig, DSparkConfig)):
        raise ValueError("Single-GPU training requires a DFly or DSpark draft config")
    configure_dflash_distillation(args)
    is_dspark = isinstance(draft_config, DSparkConfig)
    if args.training_num_nodes != 1 or args.training_num_gpus_per_node != 1:
        raise ValueError("Single-GPU training requires one training node and one training GPU")
    if args.fsdp_strategy.upper() != "REPLICATE":
        raise ValueError("Single-GPU training requires fsdp_strategy=REPLICATE")
    if args.target_model_backend != "hf":
        raise ValueError("Single-GPU training requires model.target_model_backend=hf")
    for name in (
        "defer_tokenization", "train_with_decode", "dflash_packing", "mtp_packing",
        "dflash_opd_enabled", "online_eval_enabled", "debug_train_only", "debug_inference_only",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Single-GPU training does not support {name}=true")
    if args.eval_data_path:
        raise ValueError("Use the separate evaluation scripts; single-GPU training is training-only")
    if args.attention_backend == "usp" or args.sp_ring_size != 1 or args.sp_ulysses_size != 1:
        raise ValueError("Single-GPU training does not support sequence parallelism")
    if args.micro_batch_size != 1 or args.prefetch_depth != 0:
        raise ValueError("Single-GPU training requires micro_batch_size=1 and prefetch_depth=0")
    if not args.last_hidden_states_prenorm:
        raise ValueError("Local target features are pre-final-norm; last_hidden_states_prenorm must be true")
    if args.dflash_loss_objective not in ("decay", "dpace", "edr"):
        raise ValueError("Single-GPU training supports the decay, dpace and edr objectives")
    if args.dflash_loss_objective != "edr" and args.dflash_edr_full_anchor_backprop:
        raise ValueError("Full-anchor backpropagation requires the EDR objective")
    group = _local_row_group(args)
    if args.dflash_loss_objective != "edr" and group > 1:
        # A cross-row group shares one loss scaled by group/draft_accumulation_steps,
        # so each active term must be a per-row mean (CE, L1, LK and e2e-TV are).
        # The terms below reduce over all tokens of the grouped rows.
        token_mean_terms = ["dflash_kl_loss_weight"]
        if is_dspark:
            token_mean_terms.append("dspark_confidence_head_alpha")
        active = [key for key in token_mean_terms if getattr(args, key, 0) > 0]
        if os.environ.get("ANGELSPEC_DFLASH_LOSS_CHUNK", "0") not in ("", "0"):
            active.append("ANGELSPEC_DFLASH_LOSS_CHUNK")
        if active:
            raise ValueError(
                f"{', '.join(active)} reduce over all tokens of a cross-row group; "
                "set training.dflash_distill_cross_row_batch_size=1 to use them"
            )
    if group < 1 or args.draft_accumulation_steps < 1 or args.draft_accumulation_steps % group:
        raise ValueError("draft_accumulation_steps must be positive and divisible by cross-row group size")
    _local_target_prefill_limits(args, group)
    # dflash_block_size counts learned proposals. A draft checkpoint's
    # block_size is its query width, which includes the input-anchor slot when
    # enabled. DSpark configs may omit block_size; the check then applies to DFly only.
    block_size = getattr(args, "dflash_block_size", 7)
    query_width = block_size + int(args.dflash_query_includes_input_anchor)
    checkpoint_block_size = getattr(draft_config, "block_size", None)
    if block_size < 1 or (
        (not is_dspark or checkpoint_block_size is not None)
        and query_width != checkpoint_block_size
    ):
        raise ValueError("dflash_block_size must match the draft checkpoint block size")
    num_anchors = args.dspark_num_anchors if is_dspark else args.dflash_num_anchors
    if num_anchors < 1 or args.max_seq_length < 2:
        raise ValueError("Anchor count and maximum sequence length must be positive")
    proposal_width = block_size
    if args.min_loss_tokens < 2 * proposal_width:
        raise ValueError("min_loss_tokens must be at least twice the proposal width")
    layer_ids = getattr(draft_config, "target_layer_ids", None)
    layer_count_key = "dspark_num_target_layers" if is_dspark else "dflash_num_target_layers"
    if not layer_ids or len(layer_ids) != getattr(args, layer_count_key):
        raise ValueError(f"Draft config must specify all {layer_count_key} target_layer_ids")
    for key in (
        "dflash_edr_chunk_size", "dflash_edr_vocab_chunk_size",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    args.aux_hidden_states_layers = list(layer_ids)
    args.dp_size = args.world_size = 1
    args.per_dp_rank_batch_size = 1
    args.global_batch_size = args.draft_accumulation_steps


def validate_local_checkpoint(args) -> int:
    """Check the draft initialization or checkpoint before loading models; return resume step."""
    root = Path(args.load_path).expanduser() if args.load_path else None
    if root is None or not root.exists():
        raise FileNotFoundError("training.load_path must point to a local draft initialization or checkpoint")
    hf_weights = root / "pytorch_model.bin" if root.is_dir() else root
    if args.continual_training and hf_weights.is_file() and hf_weights.suffix == ".bin":
        return 0
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError(f"Expected a DCP checkpoint ROOT containing {tracker.name}: {root}")
    iteration_text = tracker.read_text().strip()
    if not iteration_text.isdecimal():
        raise ValueError(f"Invalid checkpoint iteration in {tracker}")
    directory = root / f"iter_{int(iteration_text):07d}"
    required = ["model/.metadata", "meta.json"]
    if not args.continual_training:
        required.extend(["optimizer/.metadata", "lr_scheduler/.metadata", "rng.pt"])
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {directory / name}")
    metadata = json.loads((directory / "meta.json").read_text())
    step = metadata.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"Checkpoint metadata has no valid global_step: {directory}")
    return 0 if args.continual_training else step


class LocalTrainingBatches:
    """Yield one optimizer step's draft microbatches with target features computed on the GPU.

    With single_gpu_target_batch_size=0, each cross-row group gets its own target
    prefill. Positive row/token limits let consecutive groups with the same collator
    padding width share one prefill; draft batch shapes are the same in both cases.
    The token cap applies only when combining groups.
    """

    def __init__(self, args, target_features, *, device):
        self.args = args
        self.target_features = target_features
        self.device = torch.device(device)
        self.microbatches_per_item = _local_row_group(args)
        self.rows_per_step = args.draft_accumulation_steps
        self.target_batch_size, self.target_max_tokens = _local_target_prefill_limits(
            args, self.microbatches_per_item,
        )
        self.collator = DataCollatorWithPadding()
        self._rows = []

    def set_step(self, records):
        if len(records) != self.rows_per_step:
            raise ValueError(f"An optimizer step requires exactly {self.rows_per_step} cached sequences")
        rows = []
        max_cached_length = self.args.max_seq_length - int(
            not getattr(self.args, "allow_full_length_cached_sequences", False)
        )
        # Validate the entire step on CPU before any gradient is accumulated.
        for record in records:
            ids = torch.as_tensor(record["input_ids"], dtype=torch.long).reshape(-1)
            if not 1 <= ids.numel() <= max_cached_length:
                raise ValueError(
                    f"Cached sequence exceeds the {max_cached_length}-token input limit "
                    "or is empty; do not silently truncate"
                )
            if record.get("multimodal_inputs"):
                raise ValueError("Single-GPU training supports text-only cached sequences")
            if record.get("packed_loss_mask") is None:
                raise ValueError("Cached sequence must contain packed_loss_mask; refusing to supervise prompt tokens")
            row = {"input_ids": ids, "packed_loss_mask": record["packed_loss_mask"]}
            mask = resolve_loss_mask(row)
            if mask is None or int(mask.sum()) < self.args.min_loss_tokens:
                # As in MooncakeDataset, rows below min_loss_tokens keep their
                # batch slot but contribute zero loss.
                mask = torch.zeros_like(ids)
            rows.append({"input_ids": ids.unsqueeze(0), "loss_mask": mask.unsqueeze(0)})
        if self.args.length_balance_optimizer_step:
            rows.sort(key=lambda row: row["input_ids"].numel())
        self._rows = rows

    def __iter__(self):
        if len(self._rows) != self.rows_per_step:
            raise RuntimeError("set_step must provide a complete optimizer batch before training")
        rows, self._rows = self._rows, []
        group = self.microbatches_per_item
        if self.target_batch_size > group:
            yield from self._iter_coalesced_target_prefills(rows)
            return
        for start in range(0, len(rows), group):
            batch = self.collator(rows[start : start + group])
            batch["_token_counts"] = (
                int(batch["attention_mask"].sum()), int(batch["loss_mask"].sum()),
                batch["attention_mask"].numel(),
            )
            for key in ("input_ids", "attention_mask", "loss_mask"):
                batch[key] = batch[key].to(self.device)
            batch.update(self.target_features(batch["input_ids"], batch["attention_mask"]))
            # The EDR loss is a sum over the group's rows; the other objectives
            # return a row mean. Scale both to the optimizer-batch sum divided by
            # draft_accumulation_steps.
            batch["_loss_scale"] = (
                1 if getattr(self.args, "dflash_loss_objective", "decay") == "edr" else group
            ) / self.rows_per_step
            yield batch

    def _target_prefill_groups(self, rows):
        """Group collated CPU batches into shared target prefills, keeping their padding widths."""
        group = self.microbatches_per_item
        pending = []
        for start in range(0, len(rows), group):
            batch = self.collator(rows[start : start + group])
            batch["_token_counts"] = (
                int(batch["attention_mask"].sum()), int(batch["loss_mask"].sum()),
                batch["attention_mask"].numel(),
            )
            width = batch["input_ids"].shape[1]
            candidate_rows = (len(pending) + 1) * group
            if pending and (
                candidate_rows > self.target_batch_size
                # Combine only equal widths so padding matches separate prefills.
                or width != pending[0]["input_ids"].shape[1]
                or (self.target_max_tokens and candidate_rows * width > self.target_max_tokens)
            ):
                yield pending
                pending = []
            pending.append(batch)
        if pending:
            yield pending

    def _iter_coalesced_target_prefills(self, rows):
        group = self.microbatches_per_item
        loss_scale = (
            1 if getattr(self.args, "dflash_loss_objective", "decay") == "edr" else group
        ) / self.rows_per_step
        for batches in self._target_prefill_groups(rows):
            prefill = dict(batches[0])
            for key in ("input_ids", "attention_mask", "loss_mask"):
                values = [batch[key] for batch in batches]
                prefill[key] = (
                    values[0] if len(values) == 1 else torch.cat(values, dim=0)
                ).to(self.device)
            prefill.update(self.target_features(prefill["input_ids"], prefill["attention_mask"]))
            for key, value in prefill.items():
                if isinstance(value, torch.Tensor) and value.shape[:2] != prefill["input_ids"].shape:
                    raise ValueError(f"Target prefill field {key!r} must be token-aligned with input_ids")
            del value
            for index, batch in enumerate(batches):
                start = index * group
                width = batch["input_ids"].shape[1]
                # Slices are views into the prefill outputs; target features stay on the GPU.
                yield {
                    **{
                        key: value[start : start + group, :width]
                        if isinstance(value, torch.Tensor) else value
                        for key, value in prefill.items()
                    },
                    "_token_counts": batch["_token_counts"],
                    "_loss_scale": loss_scale,
                }
            del prefill


class _LocalTargetTrainerMixin:
    """Trainer mixin that takes target features and the target norm/LM head from a local target."""

    def __init__(self, args, target_features):
        self._local_target = target_features
        super().__init__(args)

    def _build_training_wrapper(self, draft_model):
        model = super()._build_training_wrapper(draft_model)
        model.edr_sparse_target = (
            model.loss_objective == "edr" and model.edr_temperature > 0
            and 1 <= model.edr_top_k <= 128
        )
        if model.edr_sparse_target:
            logger.info(
                "Single-GPU EDR: compact target support enabled (top_k=%d, top_p=%g, T=%g); "
                "full-vocabulary draft normalizer/gradient, no dense rejection cache; "
                "oversized tied support falls back to dense statistics",
                model.edr_top_k, model.edr_top_p, model.edr_temperature,
            )
        return model

    def _init_target_lm_head(self, target_model_path):
        # Reuse the frozen target modules already on the GPU; they are excluded
        # from the draft optimizer and checkpoint.
        self.target_lm_head = SimpleNamespace(
            lm_head=self._local_target.lm_head, norm=self._local_target.norm,
        )

    def _prepare_training_batches(self, batches, num_batches):
        # LocalTrainingBatches already yields one item per cross-row group.
        group = self.data_fetcher.microbatches_per_item
        expected = (
            self.edr_cross_row_batch_size if self.loss_objective == "edr"
            else self.distill_cross_row_batch_size
        )
        if group != expected or num_batches % group:
            raise ValueError("local batches do not match the configured cross-row grouping")
        return batches, num_batches // group

class LocalDFlyTrainer(_LocalTargetTrainerMixin, DFlyTrainer):
    """Local DFly trainer using local target features."""


class LocalDSparkTrainer(_LocalTargetTrainerMixin, DSparkTrainer):
    """Local DSpark trainer using the same batch/checkpoint pipeline as DFly."""

    def _build_training_wrapper(self, draft_model):
        model = super()._build_training_wrapper(draft_model)
        device = self._local_target.lm_head.weight.device
        pure_baseline = (
            model.loss_objective in {"decay", "dpace"}
            and (
                (model.e2e_tv_loss_weight > 0 and model.lk_loss_weight == 0)
                or (model.lk_loss_weight > 0 and model.e2e_tv_loss_weight == 0)
            )
            and all(weight == 0 for weight in (
                model.ce_loss_alpha, model.l1_loss_alpha, model.kl_loss_weight,
                model.gate_entropy_weight, model.confidence_head_alpha,
            ))
        )
        capability = (
            torch.cuda.get_device_capability(device)
            if device.type == "cuda" and (model.loss_objective == "edr" or pure_baseline)
            else None
        )
        model.edr_fused_markov_projection = (
            model.loss_objective == "edr" and capability in {(9, 0), (10, 0)}
        )
        model.distill_fused_markov_projection = pure_baseline and capability == (9, 0)
        if model.edr_fused_markov_projection:
            logger.info(
                "Single-GPU EDR (SM%d%d): fused in-place DSpark Markov logit projection; "
                "full-vocabulary draft distribution, no separate dense bias tensor",
                *capability,
            )
        elif model.distill_fused_markov_projection:
            logger.info(
                "Single-GPU E2E/LK (SM90): fused in-place DSpark Markov logit projection; "
                "unchanged full-vocabulary T=1 loss, no separate dense bias tensor"
            )
        return model


def _local_trainer_class(draft_config):
    from angelspec.models.draft.dfly import DFlyConfig
    from angelspec.models.draft.dspark import DSparkConfig

    if isinstance(draft_config, DSparkConfig):
        return LocalDSparkTrainer
    if isinstance(draft_config, DFlyConfig):
        return LocalDFlyTrainer
    raise ValueError("Single-GPU training requires a DFly or DSpark draft config")


def _progress_metrics(metrics):
    if "train/edr_mal" in metrics:
        return {
            "tokens": f"{metrics['train/edr_generated_tokens']:.0f}",
            "cost": f"{metrics['train/edr_weighted_cost']:.2f}",
            "MAL": f"{metrics['train/edr_mal']:.3f}",
        }
    return {
        "acc": f"{metrics.get('train/avg_acc', 0):.3f}",
        "acc_len": f"{metrics.get('train/simulated_acc_len', 0):.2f}",
    }


def run_local_training(args, dataset, trainer) -> None:
    """Run the training loop; checkpoints are saved only after complete optimizer steps."""
    if len(dataset) < args.global_batch_size:
        raise ValueError("Dataset must contain at least one complete global optimizer batch")
    if trainer.global_step > args.num_train_steps:
        raise ValueError("Resumed global_step exceeds num_train_steps; check the dataset and schedule")
    last_saved = None
    epoch_index = None
    order = []

    def save():
        nonlocal last_saved
        if last_saved != trainer.global_step:
            logger.info("Saving normal checkpoint after completed step %d", trainer.global_step)
            trainer.save_model(trainer.global_step, force_sync=True)
            last_saved = trainer.global_step
            _cleanup_old_checkpoints(args.checkpoint_dir, args.max_checkpoints)

    with tqdm(total=args.num_train_steps, initial=trainer.global_step, desc="Training") as progress:
        while trainer.global_step < args.num_train_steps:
            step = trainer.global_step
            epoch, epoch_step, epoch_steps = epoch_cursor(args, step)
            if epoch != epoch_index:
                if isinstance(dataset, EpochCachedDataset):
                    dataset.activate(epoch)
                # Shuffle with seed + epoch, as the distributed controller does.
                order = list(range(len(dataset)))
                if args.shuffle_dataset:
                    random.Random(args.seed + epoch).shuffle(order)
                epoch_index = epoch
            offset = epoch_step * args.global_batch_size
            trainer.data_fetcher.set_step(
                [dataset[index] for index in order[offset : offset + args.global_batch_size]]
            )
            started = time.perf_counter()
            metrics = trainer.train_from_queue(step=step, num_batches=args.draft_accumulation_steps)
            elapsed = time.perf_counter() - started
            if trainer.global_step != step + 1:
                raise RuntimeError("Trainer did not complete exactly one optimizer step; refusing checkpoint")
            progress.update(1)
            if (
                _is_save_interval_step(trainer.global_step, args.save_interval)
                or (args.save_per_epoch and epoch_step + 1 == epoch_steps)
                or trainer.global_step == args.num_train_steps
            ):
                save()
            progress.set_postfix(
                loss=f"{metrics.get('train/avg_loss', 0):.3f}",
                **_progress_metrics(metrics),
                thru=f"{args.global_batch_size / elapsed:.1f}",
                epoch=f"{epoch + 1}/{args.num_epochs}", refresh=False,
            )
            if args.use_wandb:
                import wandb

                wandb.log(metrics)
            writer = get_tb_writer()
            if writer is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, trainer.global_step)
        save()
    trainer.save_draft_model_for_serving(str(Path(args.output_dir) / "hf_final"))


def train_single_gpu(args) -> None:
    draft_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    validate_single_gpu_args(args, draft_config)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES; use run.sh for eight GPUs")
    if dist.is_initialized():
        raise RuntimeError("Run single-GPU training directly, not inside torchrun or a Ray worker")
    # The target is loaded with local_files_only=True and must already exist locally.
    if not Path(args.target_model_path).is_dir():
        raise FileNotFoundError(f"Local target checkpoint not found: {args.target_model_path}")
    expected_step = validate_local_checkpoint(args)
    # Same stop-token, auxiliary-loss and EDR configuration as train_entry.py.
    configure_dflash_edr(args)
    if not args.checkpoint_dir:
        raise ValueError("output_dir is required for resumable training")
    epoch_caches = bool(getattr(args, "epoch_cache_dirs", None))
    cache_dir = str(Path(args.cache_dir) / "tokenized_dataset")
    if not epoch_caches and find_tokenized_cache_for_training(args) is None:
        raise FileNotFoundError(
            f"No matching tokenized training cache in {cache_dir}. Generate target responses "
            "with tools/regenerate_perfectblend.py using the configured target and EDR "
            "temperature/top-k/top-p. Copy the .pt and its .pt.json provenance together; "
            "do not fall back to the OPB source assistant responses."
        )
    dataset = EpochCachedDataset(args) if epoch_caches else load_conversation_dataset(args)
    if len(dataset) < args.global_batch_size:
        raise ValueError("Dataset must contain at least one complete global optimizer batch")
    auto_calculate_training_steps(args, len(dataset))
    if epoch_caches and expected_step < args.num_train_steps:
        dataset.activate(epoch_cursor(args, expected_step)[0])
    if args.num_train_steps < 1 or args.lr_total_steps < 1:
        raise ValueError("Training and LR schedules must contain at least one optimizer step")
    torch.cuda.set_device(0)
    trainer = target_features = None
    with tempfile.TemporaryDirectory(prefix="angelspec-single-gpu-") as rendezvous:
        try:
            dist.init_process_group(
                backend="nccl", rank=0, world_size=1,
                init_method=Path(rendezvous, "rendezvous").as_uri(),
                timeout=timedelta(minutes=args.distributed_timeout_minutes),
            )
            init_gloo_group()
            from transformers import AutoModelForCausalLM

            target_rows, target_max_tokens = _local_target_prefill_limits(args, _local_row_group(args))
            logger.info(
                "Single GPU: frozen HF target + %s trainer, direct GPU features; "
                "no Ray, Mooncake or vLLM engines. Global batch=%d; cross-row group=%d; "
                "vocab chunk=%d; objective=%s; teacher statistics reuse loss projections; "
                "target prefill rows=%d; target token cap=%d (extra coalescing only)",
                type(draft_config).__name__.removesuffix("Config"),
                args.global_batch_size, _local_row_group(args),
                args.dflash_edr_vocab_chunk_size,
                args.dflash_loss_objective,
                target_rows, target_max_tokens,
            )
            target = AutoModelForCausalLM.from_pretrained(
                args.target_model_path, dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation="sdpa", trust_remote_code=args.trust_remote_code,
                local_files_only=True,
            )
            target_features = LocalTargetFeatures(target, args.aux_hidden_states_layers)
            trainer = _local_trainer_class(draft_config)(args, target_features)
            trainer.init_model(draft_config, args.target_model_path)
            if trainer.global_step != expected_step:
                raise RuntimeError(
                    f"Checkpoint resume did not restore global_step={expected_step}; "
                    f"got {trainer.global_step}. Refusing to start a fresh schedule."
                )
            trainer.data_fetcher = LocalTrainingBatches(args, target_features, device="cuda:0")
            init_tracking(args)
            logger.info(
                "Single-GPU training ready at completed step %d; checkpoints=%s",
                trainer.global_step, args.checkpoint_dir,
            )
            run_local_training(args, dataset, trainer)
        finally:
            if trainer is not None:
                trainer.close()
            if target_features is not None:
                target_features.close()
            writer = get_tb_writer()
            if writer is not None:
                writer.close()
            if dist.is_initialized():
                dist.destroy_process_group()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--print-config-only", action="store_true")
    options, overrides = parser.parse_known_args(argv)
    config = load_config(options.config, cli_args=overrides, save_snapshot=False)
    if options.print_config_only:
        print_config(config)
        return
    train_single_gpu(config_to_flat_args(config))


if __name__ == "__main__":
    main()
