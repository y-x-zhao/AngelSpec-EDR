# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Optimizer-step scheduling shared by single-GPU and distributed training."""

import math
from bisect import bisect_right

from angelspec.utils.logging import logger


def auto_calculate_training_steps(args, dataset_size: int):
    """Auto-calculate num_train_steps and lr_total_steps based on dataset size if not explicitly set.

    All step counts are in optimizer steps (not dispatches).
    steps_per_epoch = dataset_size // global_batch_size
    where global_batch_size = per_dp_rank_batch_size * dp_size * draft_accumulation_steps.

    If num_train_steps is set by user, num_epochs is calculated from it.
    Otherwise: lr_total_steps = steps_per_epoch * num_epochs
    """

    global_batch_size = args.global_batch_size
    epoch_sizes = getattr(args, "epoch_dataset_sizes", None)
    if epoch_sizes is not None:
        _configure_epoch_cache_schedule(args, epoch_sizes)
        return
    steps_per_epoch = dataset_size // global_batch_size

    if steps_per_epoch == 0:
        logger.warning(
            f"Dataset size ({dataset_size}) < global_batch_size ({global_batch_size}). Setting steps_per_epoch to 1."
        )
        steps_per_epoch = 1

    args.steps_per_epoch = steps_per_epoch

    current_num_train_steps = getattr(args, "num_train_steps", None)
    current_lr_total_steps = getattr(args, "lr_total_steps", None)

    if current_num_train_steps is not None:
        args.num_epochs = math.ceil(current_num_train_steps / steps_per_epoch)
        logger.info(
            f"Setting num_epochs to {args.num_epochs} based on num_train_steps={current_num_train_steps}!"
        )
        if current_lr_total_steps is None:
            args.lr_total_steps = current_num_train_steps
    else:
        num_epochs = getattr(args, "num_epochs", 1)
        calculated_total_steps = num_epochs * steps_per_epoch
        args.num_train_steps = calculated_total_steps
        if current_lr_total_steps is None:
            args.lr_total_steps = calculated_total_steps

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    logger.info(
        f"Training steps (optimizer steps): num_train_steps={args.num_train_steps}, "
        f"lr_total_steps={args.lr_total_steps} "
        f"(dataset_size={dataset_size}, global_batch_size={global_batch_size}, "
        f"per_dp_rank_batch_size={args.per_dp_rank_batch_size}, "
        f"accumulation_steps={accumulation_steps}, "
        f"steps_per_epoch={steps_per_epoch}, num_epochs={args.num_epochs})"
    )


def _configure_epoch_cache_schedule(args, sizes: list[int]) -> None:
    if len(sizes) != args.num_epochs or any(size < args.global_batch_size for size in sizes):
        raise ValueError("Each configured epoch cache must provide at least one full global batch")
    counts = [size // args.global_batch_size for size in sizes]
    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + count)
    requested = getattr(args, "num_train_steps", None)
    if requested is not None and not 0 < requested <= boundaries[-1]:
        raise ValueError(
            f"num_train_steps={requested} exceeds configured epoch-cache coverage "
            f"({boundaries[-1]} optimizer steps), or is not positive"
        )
    args.epoch_steps = counts
    args.epoch_step_boundaries = boundaries
    args.steps_per_epoch = counts[0]  # Reporting only; epoch_cursor handles per-epoch lengths.
    args.num_train_steps = requested if requested is not None else boundaries[-1]
    if getattr(args, "lr_total_steps", None) is None:
        args.lr_total_steps = args.num_train_steps
    # Warmup is computed from epoch 1's step count and applies once.
    args.epoch_cache_warmup_steps = int(getattr(args, "warmup_ratio", 0.0) * counts[0])
    logger.info(
        "Epoch-cache schedule: samples=%s optimizer_steps=%s boundaries=%s "
        "global_batch_size=%d num_train_steps=%d lr_total_steps=%d warmup_steps=%d "
        "(warmup applies once, against epoch 1)",
        sizes, counts, boundaries, args.global_batch_size, args.num_train_steps,
        args.lr_total_steps, args.epoch_cache_warmup_steps,
    )


def epoch_cursor(args, completed_steps: int) -> tuple[int, int, int]:
    """Return zero-based epoch, completed steps in it, and its optimizer length."""
    boundaries = getattr(args, "epoch_step_boundaries", None)
    if boundaries is None:
        epoch, offset = divmod(completed_steps, args.steps_per_epoch)
        return epoch, offset, args.steps_per_epoch
    if not 0 <= completed_steps <= boundaries[-1]:
        raise ValueError(f"Completed step {completed_steps} is outside epoch cache coverage {boundaries[-1]}")
    # At the final boundary return the last epoch, so a resumed finished run can
    # save/export without loading a cache beyond the configured epochs.
    epoch = min(bisect_right(boundaries, completed_steps) - 1, len(boundaries) - 2)
    return epoch, completed_steps - boundaries[epoch], boundaries[epoch + 1] - boundaries[epoch]
