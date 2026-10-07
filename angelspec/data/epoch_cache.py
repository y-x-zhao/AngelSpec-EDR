"""Validated, lazily loaded target-response caches for successive training epochs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from angelspec.data.dataset import find_tokenized_cache_for_training, load_conversation_dataset
from angelspec.utils.logging import logger


class EpochCachedDataset:
    """One target-response cache per epoch; only the active epoch's cache is loaded.

    Each epoch's cache is selected by ``find_tokenized_cache_for_training``. Its
    sidecar's ``cached_samples`` gives the epoch size, so the step schedule is
    built without loading the large .pt files.
    """

    def __init__(self, args):
        directories = getattr(args, "epoch_cache_dirs", None)
        if not directories:
            raise ValueError("No data cache detected!")
        if len(directories) != args.num_epochs:
            raise ValueError(f"len(num_epoches): {len(directories)}, num_epochs: {args.num_epochs}, dataset.epoch_cache_dirs must contain one cache directory per num_epochs")
        if any(getattr(args, key, False) for key in (
            "defer_tokenization", "train_with_decode", "dflash_packing", "mtp_packing",
        )):
            raise ValueError("Per-epoch caches require cached, non-packed, prefill-only training")
        self.args = copy.copy(args)
        self.paths: list[Path] = []
        self.sizes: list[int] = []
        self._signatures: list[tuple] = []
        self.epoch: int | None = None
        self.rows: list | None = None
        for epoch, directory in enumerate(directories):
            epoch_args = copy.copy(args)
            epoch_args.cache_dir = str(directory)
            path = find_tokenized_cache_for_training(epoch_args)
            if path is None:
                raise FileNotFoundError(
                    f"Epoch {epoch + 1}: no completed target/sampling-matching cache in "
                    f"{Path(directory) / 'tokenized_dataset'}. Finish generation and retain "
                    "both the .pt and its .pt.json provenance before starting training."
                )
            path = Path(path)
            sidecar = path.with_suffix(".pt.json")
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Epoch {epoch + 1} requires a readable .pt.json sidecar: {sidecar}") from exc
            count = metadata.get("cached_samples") if isinstance(metadata, dict) else None
            if (
                not isinstance(metadata, dict)
                or metadata.get("status") != "complete"
                or metadata.get("artifact_name") != path.name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < args.global_batch_size
            ):
                raise ValueError(
                    f"Epoch {epoch + 1} cache must be complete with integer cached_samples "
                    f">= global_batch_size ({args.global_batch_size}): {sidecar}"
                )
            self.paths.append(path)
            self.sizes.append(count)
            self._signatures.append(self._signature(path))
        args.epoch_dataset_sizes = list(self.sizes)

    @staticmethod
    def _signature(path: Path) -> tuple:
        # Size and mtime detect a cache replaced between startup and loading
        # without reading or hashing the large tensor payload.
        return tuple(
            (stat.st_size, stat.st_mtime_ns)
            for stat in (path.stat(), path.with_suffix(".pt.json").stat())
        )

    def activate(self, epoch: int) -> None:
        if not 0 <= epoch < len(self.paths):
            raise ValueError(f"No dataset cache configured for epoch {epoch + 1}")
        if self.epoch == epoch:
            return
        path = self.paths[epoch]
        if self._signature(path) != self._signatures[epoch]:
            raise RuntimeError(f"Epoch {epoch + 1} cache changed after startup: {path}; restart to rebuild the schedule")
        epoch_args = copy.copy(self.args)
        epoch_args.cache_dir = str(path.parent.parent)
        if find_tokenized_cache_for_training(epoch_args) != str(path):
            raise RuntimeError(f"Epoch {epoch + 1} selected cache changed after startup: {path}")
        # Release the previous epoch's rows before loading the next cache.
        self.rows = None
        self.epoch = None
        rows = load_conversation_dataset(epoch_args)
        if len(rows) != self.sizes[epoch]:
            raise ValueError(
                f"Stale cached_samples for epoch {epoch + 1}: sidecar says {self.sizes[epoch]}, "
                f"but {path} contains {len(rows)}; regenerate the cache/sidecar together"
            )
        self.rows = rows
        self.epoch = epoch
        logger.info("Activated epoch %d cache: %s (%d samples)", epoch + 1, path, len(rows))

    def __len__(self) -> int:
        return self.sizes[self.epoch if self.epoch is not None else 0]

    def __getitem__(self, index):
        if self.rows is None:
            raise RuntimeError("Activate an epoch cache before reading training samples")
        return self.rows[index]
