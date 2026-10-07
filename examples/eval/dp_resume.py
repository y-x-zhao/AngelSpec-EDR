"""Durable, CPU-only per-dataset results for incremental DP evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

FORMAT_VERSION = 1
TOKEN_COUNTING = "sampled_tokens_including_terminal"
_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _dataset_name(dataset: str) -> str:
    if not isinstance(dataset, str) or not _DATASET_NAME.fullmatch(dataset):
        raise ValueError(f"Unsafe DP dataset name: {dataset!r}")
    return dataset


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite(value: Any, *, minimum: float) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _cache_snapshot(cache_dir: Path, dataset: str) -> tuple[str, list[dict[str, int]]]:
    """Hash cache metadata and validate identities, without opening feature tensors."""
    _dataset_name(dataset)
    cache_dir = Path(cache_dir)
    if cache_dir.name != dataset or not cache_dir.is_dir():
        raise RuntimeError(f"Expected the target cache directory for {dataset!r}: {cache_dir}")
    digest = hashlib.sha256()
    manifest_path = cache_dir / "manifest.json"
    trajectories_path = cache_dir / "target_outputs.jsonl"
    try:
        raw_manifest = manifest_path.read_bytes()
        digest.update(b"manifest.json\0" + raw_manifest + b"\0target_outputs.jsonl\0")
        manifest = json.loads(raw_manifest)
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise ValueError("manifest is not complete")
        if manifest.get("format_version") != 4 or manifest.get("dataset") != dataset:
            raise ValueError("manifest has an unsupported format or different dataset")
        prompts = manifest.get("prompt_count")
        samples = manifest.get("samples_per_prompt")
        count = manifest.get("trajectory_count")
        if not (
            _integer(prompts, minimum=1)
            and _integer(samples, minimum=1)
            and _integer(count, minimum=1)
            and count == prompts * samples
        ):
            raise ValueError("manifest has invalid prompt/trajectory counts")
        expected: list[dict[str, int]] = []
        with trajectories_path.open("rb") as handle:
            for index, line in enumerate(handle):
                digest.update(line)
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("dataset") != dataset:
                    raise ValueError(f"trajectory {index} has a different dataset")
                ids = {
                    "dataset_index": index // samples,
                    "sample_index": index % samples,
                    "trajectory_index": index,
                }
                if any(not _integer(row.get(k)) or row[k] != v for k, v in ids.items()):
                    raise ValueError(f"trajectory {index} has invalid dataset/sample ordering")
                prompt_ids = row.get("prompt_token_ids")
                ordinary_ids = row.get("ordinary_token_ids")
                if (
                    not isinstance(prompt_ids, list)
                    or not prompt_ids
                    or not isinstance(ordinary_ids, list)
                    or not all(_integer(token) for token in (*prompt_ids, *ordinary_ids))
                    or not _integer(row.get("boundary_token_id"))
                ):
                    raise ValueError(f"trajectory {index} has invalid prompt/generated token IDs")
                expected.append(
                    {
                        **ids,
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(ordinary_ids) + 1,
                    }
                )
        if len(expected) != count:
            raise ValueError(f"cache contains {len(expected)} trajectories, expected {count}")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot validate target cache {cache_dir}: {exc}") from exc
    return digest.hexdigest(), expected


def cache_signature(cache_dir: Path) -> str:
    """Return a SHA256 signature covering the complete manifest and sampled tokens."""
    cache_dir = Path(cache_dir)
    signature, _ = _cache_snapshot(cache_dir, cache_dir.name)
    return signature


def _validate_records(dataset: str, records: Any, expected: list[dict[str, int]]) -> None:
    if not isinstance(records, list) or len(records) != len(expected):
        raise RuntimeError(f"Incomplete DP results for {dataset}: expected {len(expected)} rows")
    seen: set[int] = set()
    for index, record in enumerate(records):
        prefix = f"Invalid DP result for {dataset} at row {index}"
        if not isinstance(record, dict) or record.get("dataset") != dataset:
            raise RuntimeError(f"{prefix}: different dataset")
        trajectory_index = record.get("trajectory_index")
        if not _integer(trajectory_index) or trajectory_index >= len(expected):
            raise RuntimeError(f"{prefix}: invalid trajectory_index")
        if trajectory_index in seen:
            raise RuntimeError(f"{prefix}: duplicate trajectory_index {trajectory_index}")
        seen.add(trajectory_index)
        for field, value in expected[trajectory_index].items():
            if not _integer(record.get(field)) or record[field] != value:
                raise RuntimeError(f"{prefix}: {field} does not match the cached trajectory")
        # Global indices depend on which dataset combination was selected for a
        # run; the dataset-local identity above is the durable sample identity.
        if not _integer(record.get("global_index")):
            raise RuntimeError(f"{prefix}: invalid global_index")
        if record.get("token_counting", TOKEN_COUNTING) != TOKEN_COUNTING:
            raise RuntimeError(f"{prefix}: incompatible token_counting convention")
        for field in ("expected_rounds", "weighted_cost_to_go"):
            if not _finite(record.get(field), minimum=1.0):
                raise RuntimeError(f"{prefix}: {field} must be finite and at least one")
        for field, denominator in (
            ("round_start_mal", "expected_rounds"),
            ("weighted_cost_mal", "weighted_cost_to_go"),
        ):
            if field in record and (
                not _finite(record[field], minimum=0.0)
                or not math.isclose(
                    record[field], record["generated_tokens"] / record[denominator], rel_tol=1e-9
                )
            ):
                raise RuntimeError(f"{prefix}: {field} does not match the EOS-inclusive token count")


def validate_records(cache_dir: Path, dataset: str, records: list[dict[str, Any]]) -> None:
    """Require exactly one valid result for every cached prompt/sample pair."""
    _, expected = _cache_snapshot(Path(cache_dir), dataset)
    _validate_records(dataset, records, expected)


class DPResultStore:
    """Immutable complete dataset artifacts, compatible only with the same run/cache.

    Construction and loading are read-only. Saving fsyncs a unique temporary
    file before atomic publication, then fsyncs its parent directory. A per-
    dataset advisory lock prevents concurrent writers from replacing results.
    """

    def __init__(
        self,
        run_dir: Path,
        identity: dict[str, Any],
        *,
        dataset_cache_dirs: Mapping[str, Path],
    ) -> None:
        if not isinstance(identity, dict):
            raise TypeError("DP result identity must be a dictionary")
        self.directory = Path(run_dir) / "dp_completed"
        self.identity = _json_copy(identity)
        self.dataset_cache_dirs = {
            _dataset_name(name): Path(path) for name, path in dataset_cache_dirs.items()
        }

    def _paths(self, dataset: str) -> tuple[Path, Path]:
        dataset = _dataset_name(dataset)
        if dataset not in self.dataset_cache_dirs:
            raise ValueError(f"No target cache configured for DP dataset {dataset!r}")
        return self.directory / f"{dataset}.json", self.dataset_cache_dirs[dataset]

    def load(self, dataset: str) -> dict[str, Any] | None:
        path, cache_dir = self._paths(dataset)
        if path.is_symlink():
            raise RuntimeError(f"Refusing a symlink as a completed DP result: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"Cannot read completed DP result {path}: {exc}") from exc
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("expected an object")
            if type(payload.get("format_version")) is not int or payload["format_version"] != FORMAT_VERSION:
                raise ValueError("unsupported format_version")
            if payload.get("status") != "complete" or payload.get("dataset") != dataset:
                raise ValueError("incomplete status or different dataset")
            if payload.get("identity") != self.identity:
                raise ValueError(
                    "checkpoint/config/sampling identity changed; choose another output directory"
                )
            if not isinstance(payload.get("cache_info"), dict) or not isinstance(payload.get("run_info"), dict):
                raise ValueError("missing cache_info or run_info")
            signature, expected = _cache_snapshot(cache_dir, dataset)
            if payload.get("cache_signature") != signature:
                raise ValueError("target cache changed; choose another output directory")
            _validate_records(dataset, payload.get("results"), expected)
            _json_copy(payload)  # Reject non-finite values anywhere in persisted metadata.
        except (ValueError, TypeError, RuntimeError) as exc:
            raise RuntimeError(
                f"Cannot reuse completed DP result {path}: {exc}. Existing results were not overwritten."
            ) from exc
        return payload

    def save(
        self,
        dataset: str,
        records: list[dict[str, Any]],
        cache_info: dict[str, Any],
        run_info: dict[str, Any],
    ) -> dict[str, Any]:
        path, cache_dir = self._paths(dataset)
        signature, expected = _cache_snapshot(cache_dir, dataset)
        _validate_records(dataset, records, expected)
        if not isinstance(cache_info, dict) or not isinstance(run_info, dict):
            raise TypeError("DP result cache_info and run_info must be dictionaries")
        payload = _json_copy(
            {
                "format_version": FORMAT_VERSION,
                "status": "complete",
                "dataset": dataset,
                "identity": self.identity,
                "cache_signature": signature,
                "results": sorted(records, key=lambda row: row["trajectory_index"]),
                "cache_info": cache_info,
                "run_info": run_info,
            }
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f".{dataset}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            previous = self.load(dataset)
            if previous is not None:
                if previous != payload:
                    raise RuntimeError(f"Refusing to overwrite completed DP results: {path}")
                return previous
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.directory,
                    prefix=f".{dataset}.", suffix=".tmp", delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return payload
