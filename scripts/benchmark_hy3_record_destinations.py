#!/usr/bin/env python3
"""Compare Hy3 sidecar reads into a record arena and component rows.

This benchmark isolates destination/layout efficiency. Both arms read the
same byte ranges with ``os.preadv`` and ``F_NOCACHE``; it does not claim to
remove a syscall. All MLX resources are materialized before the first external
write, and the harness performs no Metal reads while writes are outstanding.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_manifest import (  # noqa: E402
    load_expert_manifest,
    resolve_artifact_member,
)
from mtplx.kernels.moe_record_q4 import HY3_RECORD_Q4_LAYOUT  # noqa: E402


HY3_RECORD_BYTES = HY3_RECORD_Q4_LAYOUT.record_bytes
HY3_COMPONENT_LENGTHS = tuple(
    component.length for component in HY3_RECORD_Q4_LAYOUT.components
)
_ARMS = ("record-arena", "component-rows")
_ALLOWED_QUEUE_DEPTHS = (1, 32)
_PINNED_SOURCE_REPO = "pipenetwork/Hy3-4bit"
_PINNED_SOURCE_REVISION = "160619d3f96c8470350b6dac0ef033a8381551e3"


class BenchmarkContractError(ValueError):
    """Raised when a benchmark arm would no longer be comparable."""


@dataclass(frozen=True)
class RecordRange:
    offset: int
    length: int
    identity: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("offset", self.offset, 0),
            ("length", self.length, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise BenchmarkContractError(f"record {name} must be an integer")
            if value < minimum:
                raise BenchmarkContractError(
                    f"record {name} must be at least {minimum}"
                )
        if not isinstance(self.identity, str) or not self.identity:
            raise BenchmarkContractError("record identity must be non-empty")


@dataclass(frozen=True)
class Workload:
    individual: tuple[RecordRange, ...]
    adjacent_batches: tuple[tuple[RecordRange, ...], ...]


def manifest_record_range(record: Any) -> RecordRange:
    """Copy one manifest record without treating offset zero as missing."""

    if record.sidecar_offset is None or record.sidecar_length is None:
        raise BenchmarkContractError("manifest sidecar record is incomplete")
    return RecordRange(
        offset=record.sidecar_offset,
        length=record.sidecar_length,
        identity=f"layer-{record.layer}-expert-{record.expert}",
        sha256=record.sha256,
    )


def _exact_positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkContractError(f"{name} must be an integer")
    if value <= 0:
        raise BenchmarkContractError(f"{name} must be positive")
    return value


def release_views(views: Iterable[memoryview]) -> None:
    for view in views:
        try:
            view.release()
        except Exception:
            pass


def _resource_bytes(resource: Any) -> memoryview:
    try:
        view = memoryview(resource)
    except TypeError as exc:
        raise BenchmarkContractError(
            "destination resource must expose writable bytes"
        ) from exc
    if view.readonly:
        view.release()
        raise BenchmarkContractError("destination resource must be writable")
    try:
        return view.cast("B")
    except TypeError as exc:
        view.release()
        raise BenchmarkContractError(
            "destination resource must be C-contiguous"
        ) from exc


class _Destination:
    def __init__(
        self,
        resources: tuple[Any, ...],
        *,
        capacity: int,
        component_lengths: tuple[int, ...],
    ) -> None:
        self._resources: tuple[Any, ...] | None = resources
        self._base_views: tuple[memoryview, ...] | None = None
        self.capacity = capacity
        self.component_lengths = component_lengths

    def _ensure_base_views(self) -> tuple[memoryview, ...]:
        resources = self._resources
        if resources is None:
            raise BenchmarkContractError("destination is closed")
        if self._base_views is not None:
            return self._base_views
        views: list[memoryview] = []
        try:
            for resource, length in zip(resources, self.component_lengths, strict=True):
                resource_view = _resource_bytes(resource)
                expected = self.capacity * length
                if resource_view.nbytes != expected:
                    resource_view.release()
                    raise BenchmarkContractError(
                        "destination resource byte coverage differs from its shape"
                    )
                views.append(resource_view)
        except BaseException:
            release_views(views)
            raise
        self._base_views = tuple(views)
        return self._base_views

    def record_views(self, slot: int) -> tuple[memoryview, ...]:
        base_views = self._ensure_base_views()
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise BenchmarkContractError("slot must be an integer")
        if slot < 0 or slot >= self.capacity:
            raise BenchmarkContractError("slot is outside destination capacity")
        views: list[memoryview] = []
        try:
            for resource_view, length in zip(
                base_views, self.component_lengths, strict=True
            ):
                start = slot * length
                views.append(resource_view[start : start + length])
        except BaseException:
            release_views(views)
            raise
        return tuple(views)

    def record_bytes_at(self, slot: int) -> bytes:
        views = self.record_views(slot)
        try:
            return b"".join(bytes(view) for view in views)
        finally:
            release_views(views)

    def close(self) -> None:
        if self._base_views is not None:
            release_views(self._base_views)
            self._base_views = None
        self._resources = None


def allocate_destinations(
    *,
    capacity: int,
    record_bytes: int,
    component_lengths: Sequence[int],
    allocate: Callable[[tuple[int, ...]], Any],
    finalize: Callable[..., None],
) -> tuple[_Destination, _Destination]:
    """Allocate exactly one arena plus one resource per component."""

    capacity = _exact_positive("capacity", capacity)
    record_bytes = _exact_positive("record bytes", record_bytes)
    lengths = tuple(
        _exact_positive("component length", length) for length in component_lengths
    )
    if not lengths or sum(lengths) != record_bytes:
        raise BenchmarkContractError(
            "component byte coverage must equal the record stride"
        )
    arena_resource = allocate((capacity, record_bytes))
    component_resources = tuple(allocate((capacity, length)) for length in lengths)
    # MLX is lazy. This one call finalizes every allocation before any preadv.
    finalize(arena_resource, *component_resources)
    return (
        _Destination(
            (arena_resource,),
            capacity=capacity,
            component_lengths=(record_bytes,),
        ),
        _Destination(
            component_resources,
            capacity=capacity,
            component_lengths=lengths,
        ),
    )


def _validate_coverage(record: RecordRange, views: Sequence[memoryview]) -> None:
    if not views or sum(view.nbytes for view in views) != record.length:
        raise BenchmarkContractError(
            f"destination byte coverage differs from {record.identity}"
        )
    if any(view.readonly or not view.c_contiguous for view in views):
        raise BenchmarkContractError(
            "preadv destinations must be writable and contiguous"
        )


def _advance_views(views: Sequence[memoryview], consumed: int) -> list[memoryview]:
    pending: list[memoryview] = []
    for view in views:
        if consumed >= view.nbytes:
            consumed -= view.nbytes
            continue
        if consumed:
            view = view[consumed:]
            consumed = 0
        pending.append(view)
    return pending


def _preadv_exact(fd: int, offset: int, views: Sequence[memoryview]) -> None:
    pending = list(views)
    read_total = 0
    expected = sum(view.nbytes for view in views)
    while pending:
        try:
            read_now = int(os.preadv(fd, pending, offset + read_total))
        except InterruptedError:
            continue
        except OSError as exc:
            raise BenchmarkContractError(f"positional read failed: {exc}") from exc
        if read_now <= 0:
            raise BenchmarkContractError(
                f"short read at {offset + read_total}; wanted {expected - read_total} bytes"
            )
        read_total += read_now
        pending = _advance_views(pending, read_now)
    if read_total != expected:  # pragma: no cover - defensive kernel contract
        raise BenchmarkContractError(
            f"short read: received {read_total} of {expected} bytes"
        )


def preadv_record_exact(
    fd: int,
    record: RecordRange,
    views: Sequence[memoryview],
) -> None:
    _validate_coverage(record, views)
    _preadv_exact(fd, record.offset, views)


def preadv_adjacent_batch_exact(
    fd: int,
    records: Sequence[RecordRange],
    record_views: Sequence[Sequence[memoryview]],
) -> None:
    if not records or len(records) != len(record_views):
        raise BenchmarkContractError(
            "adjacent batch must pair every record with one destination"
        )
    for record, views in zip(records, record_views, strict=True):
        _validate_coverage(record, views)
    for left, right in zip(records, records[1:], strict=False):
        if right.offset != left.offset + left.length:
            raise BenchmarkContractError("batch records must be physically adjacent")
    flattened = tuple(view for views in record_views for view in views)
    _preadv_exact(fd, records[0].offset, flattened)


def _hash_views(views: Sequence[memoryview]) -> str:
    digest = hashlib.sha256()
    for view in views:
        digest.update(view)
    return digest.hexdigest()


def run_parity_preflight(
    fd: int,
    ranges: Sequence[RecordRange],
    arena: _Destination,
    rows: _Destination,
) -> list[dict[str, Any]]:
    if not ranges:
        raise BenchmarkContractError("parity preflight requires records")
    results: list[dict[str, Any]] = []
    for index, record in enumerate(ranges):
        slot = index % min(arena.capacity, rows.capacity)
        arena_views = arena.record_views(slot)
        row_views = rows.record_views(slot)
        try:
            preadv_record_exact(fd, record, arena_views)
            preadv_record_exact(fd, record, row_views)
            arena_hash = _hash_views(arena_views)
            row_hash = _hash_views(row_views)
            if arena_hash != row_hash:
                raise BenchmarkContractError(
                    f"destination parity failed for {record.identity}"
                )
            if record.sha256 is not None and arena_hash != record.sha256:
                raise BenchmarkContractError(
                    f"trusted record hash failed for {record.identity}"
                )
            results.append(
                {
                    "identity": record.identity,
                    "offset": record.offset,
                    "length": record.length,
                    "sha256": arena_hash,
                    "arena_sha256": arena_hash,
                    "component_sha256": row_hash,
                    "trusted_sha256": record.sha256,
                }
            )
        finally:
            release_views(arena_views)
            release_views(row_views)
    return results


def validate_config(
    *,
    operations: int,
    queue_depths: Sequence[int],
    batch_records: int,
    repeats: int,
) -> None:
    operations = _exact_positive("operations", operations)
    batch_records = _exact_positive("batch records", batch_records)
    if batch_records > 32:
        raise BenchmarkContractError("batch records must not exceed 32")
    _exact_positive("repeats", repeats)
    depths = tuple(queue_depths)
    if not depths:
        raise BenchmarkContractError("queue depths must not be empty")
    if any(
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or depth not in _ALLOWED_QUEUE_DEPTHS
        for depth in depths
    ):
        raise BenchmarkContractError("queue depth must be 1 or 32")
    if len(set(depths)) != len(depths):
        raise BenchmarkContractError("queue depths must be unique")
    if operations % max(depths) or operations % batch_records:
        raise BenchmarkContractError(
            "operations must divide evenly across queue depth and batch records"
        )


def _require_adjacent(records: Sequence[RecordRange]) -> None:
    for left, right in zip(records, records[1:], strict=False):
        if right.offset != left.offset + left.length:
            raise BenchmarkContractError("record corpus is not physically adjacent")


def precompute_workload(
    ranges: Sequence[RecordRange],
    *,
    operations: int,
    batch_records: int,
    seed: int,
) -> Workload:
    operations = _exact_positive("operations", operations)
    batch_records = _exact_positive("batch records", batch_records)
    ordered = tuple(sorted(ranges, key=lambda record: record.offset))
    if len(ordered) < batch_records:
        raise BenchmarkContractError("record corpus is smaller than one batch")
    _require_adjacent(ordered)
    if operations % batch_records:
        raise BenchmarkContractError("operations must divide into adjacent batches")
    rng = random.Random(seed)
    individual = tuple(ordered[rng.randrange(len(ordered))] for _ in range(operations))
    windows: list[tuple[RecordRange, ...]] = []
    last_start = len(ordered) - batch_records
    for _ in range(operations // batch_records):
        start = rng.randrange(last_start + 1)
        windows.append(ordered[start : start + batch_records])
    return Workload(individual=individual, adjacent_batches=tuple(windows))


def arm_order(repeat: int) -> tuple[str, str]:
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise BenchmarkContractError("repeat index must be non-negative")
    return _ARMS if repeat % 2 == 0 else tuple(reversed(_ARMS))


def summarize_latencies(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise BenchmarkContractError("latency summary requires observations")
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _workload_sha256(workload: Workload) -> str:
    canonical = {
        "individual": [record.identity for record in workload.individual],
        "adjacent_batches": [
            [record.identity for record in batch] for batch in workload.adjacent_batches
        ],
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_payload(
    *,
    command: Sequence[str],
    config: dict[str, Any],
    provenance: dict[str, Any],
    workload_sha256: str,
    parity: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    argv = list(command)
    return {
        "schema": "mtplx-hy3-record-destination-benchmark-v1",
        "mechanism": "destination/layout efficiency",
        "command": {"argv": argv, "shell": shlex.join(argv)},
        "configuration": config,
        "provenance": provenance,
        "workload_sha256": workload_sha256,
        "parity_preflight": parity,
        "io_contract": {
            "syscall": "os.preadv",
            "cache_mode": "F_NOCACHE",
            "claims_syscall_elimination": False,
            "writes_complete_before_metal_reads": True,
        },
        "results": results,
    }


@contextmanager
def open_sidecar_nocache(path: Path) -> Iterator[int]:
    command = getattr(fcntl, "F_NOCACHE", None)
    if command is None:
        raise BenchmarkContractError("F_NOCACHE is unavailable on this platform")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BenchmarkContractError(f"could not open sidecar: {exc}") from exc
    try:
        try:
            fcntl.fcntl(fd, command, 1)
        except OSError as exc:
            raise BenchmarkContractError(f"could not enable F_NOCACHE: {exc}") from exc
        yield fd
    finally:
        os.close(fd)


def _measure_individual(
    fd: int,
    records: Sequence[RecordRange],
    destination: _Destination,
    *,
    queue_depth: int,
) -> tuple[list[float], list[float], float]:
    record_latencies: list[float] = []
    batch_latencies: list[float] = []

    def read_one(pair: tuple[int, RecordRange]) -> float:
        slot, record = pair
        views: tuple[memoryview, ...] = ()
        started = time.perf_counter_ns()
        try:
            views = destination.record_views(slot)
            preadv_record_exact(fd, record, views)
            return (time.perf_counter_ns() - started) / 1e6
        finally:
            release_views(views)

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=queue_depth) as pool:
        for start in range(0, len(records), queue_depth):
            batch = records[start : start + queue_depth]
            batch_started = time.perf_counter_ns()
            record_latencies.extend(pool.map(read_one, tuple(enumerate(batch))))
            batch_latencies.append((time.perf_counter_ns() - batch_started) / 1e6)
    return record_latencies, batch_latencies, time.perf_counter() - wall_started


def _measure_adjacent(
    fd: int,
    batches: Sequence[Sequence[RecordRange]],
    destination: _Destination,
) -> tuple[list[float], list[float], float]:
    record_latencies: list[float] = []
    batch_latencies: list[float] = []
    wall_started = time.perf_counter()
    for batch in batches:
        views: tuple[tuple[memoryview, ...], ...] = ()
        started = time.perf_counter_ns()
        try:
            views = tuple(
                destination.record_views(slot) for slot, _record in enumerate(batch)
            )
            preadv_adjacent_batch_exact(fd, batch, views)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            # All records in one preadv complete at the batch completion time.
            record_latencies.extend([elapsed_ms] * len(batch))
            batch_latencies.append(elapsed_ms)
        finally:
            for record_views in views:
                release_views(record_views)
    return record_latencies, batch_latencies, time.perf_counter() - wall_started


def _warm_individual(
    fd: int,
    records: Sequence[RecordRange],
    destination: _Destination,
    *,
    queue_depth: int,
) -> None:
    _measure_individual(fd, records, destination, queue_depth=queue_depth)


def _result(
    *,
    repeat: int,
    arm: str,
    lane: str,
    io_queue_depth: int,
    batch_records: int,
    record_count: int,
    record_bytes: int,
    record_latencies: list[float],
    batch_latencies: list[float],
    elapsed: float,
) -> dict[str, Any]:
    total_bytes = record_count * record_bytes
    return {
        "repeat": repeat,
        "arm": arm,
        "lane": lane,
        "io_queue_depth": io_queue_depth,
        "batch_records": batch_records,
        "records": record_count,
        "bytes": total_bytes,
        "elapsed_seconds": elapsed,
        "gib_per_second": total_bytes / 1024**3 / elapsed,
        "records_per_second": record_count / elapsed,
        "record_latency_ms": summarize_latencies(record_latencies),
        "batch_latency_ms": summarize_latencies(batch_latencies),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_ROOT,
            text=True,
            timeout=5,
        )
    except Exception:
        return True
    return bool(status.strip())


def harness_provenance() -> dict[str, Any]:
    """Identify the exact harness and environment that produced a result."""

    source = Path(__file__).resolve()
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_dirty": _git_dirty(),
        "cwd": str(Path.cwd()),
        "mlx_version": importlib.metadata.version("mlx"),
    }


def validate_manifest_contract(manifest: Any) -> tuple[int, ...]:
    """Fail closed unless every record matches the pinned Hy3 v1 geometry."""

    pinned = (
        manifest.model_key == "hy3-q4",
        manifest.source_repo == _PINNED_SOURCE_REPO,
        manifest.source_revision == _PINNED_SOURCE_REVISION,
        manifest.quant_bits == 4,
        manifest.quant_group_size == 64,
        manifest.quant_mode == "affine",
        manifest.sidecar is not None,
        bool(manifest.records),
    )
    if not all(pinned):
        raise BenchmarkContractError("benchmark requires the pinned Hy3 Q4 sidecar")

    expected_segments = tuple(
        (component.name, component.length, component.dtype, component.shape)
        for component in HY3_RECORD_Q4_LAYOUT.components
    )
    for record in manifest.records:
        actual_segments = tuple(
            (segment.component, segment.length, segment.dtype, tuple(segment.shape))
            for segment in record.segments
        )
        if (
            record.logical_bytes != HY3_RECORD_BYTES
            or record.sidecar_offset is None
            or record.sidecar_length != HY3_RECORD_BYTES
            or actual_segments != expected_segments
        ):
            raise BenchmarkContractError(
                "manifest record geometry differs from pinned Hy3 v1"
            )
    return HY3_COMPONENT_LENGTHS


def _positive_argument(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--operations", type=_positive_argument, default=96)
    parser.add_argument("--warmup-operations", type=_positive_argument, default=32)
    parser.add_argument("--queue-depths", default="1,32")
    parser.add_argument("--batch-records", type=_positive_argument, default=32)
    parser.add_argument("--repeats", type=_positive_argument, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def _parse_queue_depths(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise BenchmarkContractError(
            "queue depths must be comma-separated integers"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queue_depths = _parse_queue_depths(args.queue_depths)
    validate_config(
        operations=args.operations,
        queue_depths=queue_depths,
        batch_records=args.batch_records,
        repeats=args.repeats,
    )
    if args.warmup_operations % max(queue_depths):
        raise BenchmarkContractError(
            "warmup operations must divide evenly across the largest queue depth"
        )

    root = args.model_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_expert_manifest(manifest_path)
    component_lengths = validate_manifest_contract(manifest)

    ranges = tuple(manifest_record_range(record) for record in manifest.records)
    workload = precompute_workload(
        ranges,
        operations=args.operations,
        batch_records=args.batch_records,
        seed=args.seed,
    )

    import mlx.core as mx

    capacity = max((*queue_depths, args.batch_records))
    arena, rows = allocate_destinations(
        capacity=capacity,
        record_bytes=HY3_RECORD_BYTES,
        component_lengths=component_lengths,
        allocate=lambda shape: mx.zeros(shape, dtype=mx.uint8),
        finalize=mx.eval,
    )
    destinations = {"record-arena": arena, "component-rows": rows}
    sidecar_path = resolve_artifact_member(root, manifest.sidecar.file)
    results: list[dict[str, Any]] = []
    try:
        with open_sidecar_nocache(sidecar_path) as fd:
            sentinels = (
                ranges[0],
                ranges[len(ranges) // 2],
                ranges[-1],
            )
            parity = run_parity_preflight(fd, sentinels, arena, rows)
            warm_records = workload.individual[: args.warmup_operations]
            for repeat in range(args.repeats):
                for arm in arm_order(repeat):
                    destination = destinations[arm]
                    for depth in queue_depths:
                        _warm_individual(
                            fd,
                            warm_records,
                            destination,
                            queue_depth=depth,
                        )
                        record_ms, batch_ms, elapsed = _measure_individual(
                            fd,
                            workload.individual,
                            destination,
                            queue_depth=depth,
                        )
                        results.append(
                            _result(
                                repeat=repeat,
                                arm=arm,
                                lane="individual",
                                io_queue_depth=depth,
                                batch_records=1,
                                record_count=len(workload.individual),
                                record_bytes=HY3_RECORD_BYTES,
                                record_latencies=record_ms,
                                batch_latencies=batch_ms,
                                elapsed=elapsed,
                            )
                        )
                    # One QD1 preadv spanning one physically adjacent record batch.
                    warm_batch = workload.adjacent_batches[:1]
                    _measure_adjacent(fd, warm_batch, destination)
                    record_ms, batch_ms, elapsed = _measure_adjacent(
                        fd,
                        workload.adjacent_batches,
                        destination,
                    )
                    results.append(
                        _result(
                            repeat=repeat,
                            arm=arm,
                            lane="adjacent-batch",
                            io_queue_depth=1,
                            batch_records=args.batch_records,
                            record_count=args.operations,
                            record_bytes=HY3_RECORD_BYTES,
                            record_latencies=record_ms,
                            batch_latencies=batch_ms,
                            elapsed=elapsed,
                        )
                    )
    finally:
        arena.close()
        rows.close()

    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    config = {
        "operations": args.operations,
        "warmup_operations": args.warmup_operations,
        "queue_depths": list(queue_depths),
        "batch_records": args.batch_records,
        "repeats": args.repeats,
        "seed": args.seed,
        "record_bytes": HY3_RECORD_BYTES,
        "component_lengths": list(component_lengths),
    }
    payload = build_payload(
        command=command,
        config=config,
        provenance={
            "git_commit": _git_commit(),
            "harness": harness_provenance(),
            "model_key": manifest.model_key,
            "source_repo": manifest.source_repo,
            "source_revision": manifest.source_revision,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest.manifest_sha256,
            "sidecar": str(sidecar_path),
            "sidecar_sha256": manifest.sidecar.sha256,
            "sidecar_bytes": manifest.sidecar.size,
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
        },
        workload_sha256=_workload_sha256(workload),
        parity=parity,
        results=results,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
