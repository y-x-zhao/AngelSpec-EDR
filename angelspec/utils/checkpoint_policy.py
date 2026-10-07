# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Checkpoint cadence and retention shared without importing the controller."""

import re
import shutil
from pathlib import Path

from angelspec.utils.logging import logger


def _is_save_interval_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _cleanup_old_checkpoints(checkpoint_dir: str | None, max_checkpoints: int) -> None:
    """Delete old checkpoints, keeping only the most recent `max_checkpoints`.

    Checkpoint directories are named ``iter_NNNNNNN`` where N is the step number.
    The ``latest_checkpointed_iteration.txt`` and ``best_*`` files are preserved.
    """
    if not checkpoint_dir or max_checkpoints <= 0:
        return

    base_dir = Path(checkpoint_dir).expanduser()
    if not base_dir.exists():
        return

    # Find all iter_* directories, sorted by step number
    iter_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and re.match(r"iter_\d+", d.name)),
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if len(iter_dirs) <= max_checkpoints:
        return

    # Delete oldest checkpoints, keep the newest max_checkpoints
    to_delete = iter_dirs[: len(iter_dirs) - max_checkpoints]
    for old_dir in to_delete:
        logger.info(f"Removing old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except OSError as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")
