#!/usr/bin/env python3
"""Benchmark exact 1/2/4/8-way reads of one Hy3 sidecar record."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import hashlib
import json
import os
import platform
import random
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


ALLOWED_STRIPE_COUNTS = (1, 2, 4, 8)
ALIGNMENT_BYTES = 16 * 1024
HY3_RECORD_BYTES = 10_616_832
PAGE_SIZE = os.sysconf("SC_PAGESIZE")
_PROT_READ = 0x01
_MAP_SHARED = 0x0001
_MAP_FAILED = ctypes.c_void_p(-1).value
_MADV_DONTNEED = 4
_MS_INVALIDATE = 0x0002
_MINCORE_INCORE = 0x1
_MINCORE_REFERENCED = 0x2
_COLD_RESIDENT_FRACTION_LIMIT = 0.10


class BenchmarkContractError(RuntimeError):
    """The experiment would no longer compare equivalent record reads."""


@dataclass(frozen=True)
class LogicalStripe:
    logical_offset: int
    source_offset: int
    length: int
    destination: memoryview


@dataclass(frozen=True)
class RecordRange:
    identity: str
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class IOAccounting:
    host_calls: int = 0
    returned_bytes: int = 0

    def __add__(self, other: IOAccounting) -> IOAccounting:
        return IOAccounting(
            host_calls=self.host_calls + other.host_calls,
            returned_bytes=self.returned_bytes + other.returned_bytes,
        )


@dataclass(frozen=True)
class RecordOutcome:
    accounting: IOAccounting
    service_time_ms: float
    sha256: str
    logical_publications: int


@dataclass(frozen=True)
class ResidencySnapshot:
    pages: int
    resident_pages: int
    referenced_pages: int


def _libc() -> ctypes.CDLL:
    name = ctypes.util.find_library("c")
    if name is None:
        raise BenchmarkContractError("cannot locate libc for mincore verification")
    lib = ctypes.CDLL(name, use_errno=True)
    lib.mmap.restype = ctypes.c_void_p
    lib.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
    ]
    lib.munmap.restype = ctypes.c_int
    lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mincore.restype = ctypes.c_int
    lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    lib.madvise.restype = ctypes.c_int
    lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.msync.restype = ctypes.c_int
    lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return lib


class FileResidency:
    """Verify and evict sidecar page residency through mincore/madvise."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.size = self.path.stat().st_size
        self._libc = _libc()
        self._fd = os.open(
            self.path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        address = self._libc.mmap(None, self.size, _PROT_READ, _MAP_SHARED, self._fd, 0)
        if address in (None, 0) or address == _MAP_FAILED:
            error = ctypes.get_errno()
            os.close(self._fd)
            raise BenchmarkContractError(
                f"mmap failed for residency verification: {os.strerror(error)}"
            )
        self._address = address
        self._closed = False

    def _range(self, offset: int, length: int) -> tuple[int, int, int]:
        if offset < 0 or length <= 0 or offset + length > self.size:
            raise BenchmarkContractError("residency range is outside the sidecar")
        start = (offset // PAGE_SIZE) * PAGE_SIZE
        end = -(-(offset + length) // PAGE_SIZE) * PAGE_SIZE
        span = end - start
        return self._address + start, span, span // PAGE_SIZE

    def residency(self, offset: int, length: int) -> ResidencySnapshot:
        address, span, pages = self._range(offset, length)
        vector = ctypes.create_string_buffer(pages)
        if self._libc.mincore(ctypes.c_void_p(address), span, vector) != 0:
            error = ctypes.get_errno()
            raise BenchmarkContractError(f"mincore failed: {os.strerror(error)}")
        raw = vector.raw[:pages]
        return ResidencySnapshot(
            pages=pages,
            resident_pages=sum(1 for byte in raw if byte & _MINCORE_INCORE),
            referenced_pages=sum(1 for byte in raw if byte & _MINCORE_REFERENCED),
        )

    def try_evict(self, offset: int, length: int) -> bool:
        address, span, _pages = self._range(offset, length)
        invalidated = (
            self._libc.msync(ctypes.c_void_p(address), span, _MS_INVALIDATE) == 0
        )
        discarded = (
            self._libc.madvise(ctypes.c_void_p(address), span, _MADV_DONTNEED) == 0
        )
        return invalidated and discarded

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._libc.munmap(ctypes.c_void_p(self._address), self.size)
        os.close(self._fd)

    def __enter__(self) -> FileResidency:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def release_stripes(stripes: Sequence[LogicalStripe]) -> None:
    for stripe in stripes:
        stripe.destination.release()


def partition_record(
    destination: memoryview,
    *,
    source_offset: int,
    logical_bytes: int,
    stripe_count: int,
    alignment: int,
) -> tuple[LogicalStripe, ...]:
    """Return equal, aligned, disjoint source and destination ranges."""

    if isinstance(stripe_count, bool) or stripe_count not in ALLOWED_STRIPE_COUNTS:
        raise BenchmarkContractError("stripe count must be one of 1,2,4,8")
    for name, value, minimum in (
        ("source offset", source_offset, 0),
        ("logical bytes", logical_bytes, 1),
        ("alignment", alignment, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise BenchmarkContractError(f"{name} must be an integer >= {minimum}")
    if destination.readonly or not destination.c_contiguous:
        raise BenchmarkContractError("destination must be writable and contiguous")
    if destination.nbytes != logical_bytes:
        raise BenchmarkContractError("destination coverage differs from logical bytes")
    if logical_bytes % stripe_count:
        raise BenchmarkContractError("record cannot be evenly striped")
    if source_offset % alignment:
        raise BenchmarkContractError("source offset must preserve alignment")
    stripe_bytes = logical_bytes // stripe_count
    if stripe_bytes % alignment:
        raise BenchmarkContractError("every stripe must preserve alignment")

    return tuple(
        LogicalStripe(
            logical_offset=index * stripe_bytes,
            source_offset=source_offset + index * stripe_bytes,
            length=stripe_bytes,
            destination=destination[index * stripe_bytes : (index + 1) * stripe_bytes],
        )
        for index in range(stripe_count)
    )


def preadv_exact(
    fd: int,
    *,
    source_offset: int,
    destination: memoryview,
    preadv: Callable[[int, Sequence[memoryview], int], int] = os.preadv,
) -> IOAccounting:
    """Fill one range and return actual call/returned-byte accounting."""

    read_total = 0
    host_calls = 0
    while read_total < destination.nbytes:
        target = destination[read_total:]
        try:
            host_calls += 1
            read_now = int(preadv(fd, [target], source_offset + read_total))
        except InterruptedError:
            continue
        finally:
            target.release()
        if read_now <= 0:
            raise BenchmarkContractError(
                f"short read at {source_offset + read_total}; "
                f"wanted {destination.nbytes - read_total} bytes"
            )
        if read_now > destination.nbytes - read_total:
            raise BenchmarkContractError("preadv returned more bytes than requested")
        read_total += read_now
    return IOAccounting(host_calls=host_calls, returned_bytes=read_total)


def _drain(
    futures: Sequence[Future[IOAccounting]],
) -> tuple[IOAccounting, BaseException | None]:
    accounting = IOAccounting()
    first_error: BaseException | None = None
    for future in futures:
        while True:
            try:
                accounting += future.result()
                break
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                if future.done():
                    break
    return accounting, first_error


@contextmanager
def _blocked_termination_signals():
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return
    previous = pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous)


def read_record_exact(
    fd: int,
    record: RecordRange,
    destination: bytearray | memoryview,
    *,
    stripe_count: int,
    executor: Any | None,
) -> IOAccounting:
    """Read one record; accepted stripes always settle before an error escapes."""

    root = memoryview(destination)
    stripes = partition_record(
        root,
        source_offset=record.offset,
        logical_bytes=record.length,
        stripe_count=stripe_count,
        alignment=ALIGNMENT_BYTES,
    )
    try:
        if stripe_count == 1:
            stripe = stripes[0]
            return preadv_exact(
                fd,
                source_offset=stripe.source_offset,
                destination=stripe.destination,
            )
        if executor is None:
            raise BenchmarkContractError("striped candidates require an executor")

        futures: list[Future[IOAccounting]] = []
        submit_error: BaseException | None = None
        for stripe in stripes:
            try:
                with _blocked_termination_signals():
                    future = executor.submit(
                        preadv_exact,
                        fd,
                        source_offset=stripe.source_offset,
                        destination=stripe.destination,
                    )
                    futures.append(future)
            except BaseException as exc:
                submit_error = exc
                break
        accounting, read_error = _drain(futures)
        if submit_error is not None:
            raise submit_error
        if read_error is not None:
            raise read_error
        return accounting
    finally:
        release_stripes(stripes)
        root.release()


def read_and_validate_record(
    fd: int,
    record: RecordRange,
    destination: bytearray | memoryview,
    *,
    stripe_count: int,
    executor: Any | None,
) -> RecordOutcome:
    """Read, fully hash, and publish exactly one complete logical record."""

    started_ns = time.perf_counter_ns()
    accounting = read_record_exact(
        fd,
        record,
        destination,
        stripe_count=stripe_count,
        executor=executor,
    )
    service_time_ms = (time.perf_counter_ns() - started_ns) / 1e6
    digest = hashlib.sha256(destination).hexdigest()
    if digest != record.sha256:
        raise BenchmarkContractError(
            f"record hash mismatch for {record.identity}: {digest} != {record.sha256}"
        )
    return RecordOutcome(
        accounting=accounting,
        service_time_ms=service_time_ms,
        sha256=digest,
        logical_publications=1,
    )


def balanced_order(
    repeat: int, stripe_counts: Sequence[int] = ALLOWED_STRIPE_COUNTS
) -> tuple[int, ...]:
    """Return a balanced eight-repeat order, repeated independently per phase."""

    widths = tuple(stripe_counts)
    if set(widths) != set(ALLOWED_STRIPE_COUNTS) or len(widths) != 4:
        raise BenchmarkContractError("stripe counts must contain exactly 1,2,4,8")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise BenchmarkContractError("repeat must be a non-negative integer")
    phase_repeat = repeat % 8
    cycle = phase_repeat % 4
    order = widths[cycle:] + widths[:cycle]
    return order if phase_repeat < 4 else tuple(reversed(order))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise BenchmarkContractError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def result_row(
    *,
    repeat: int,
    order_position: int,
    stripe_count: int,
    record_bytes: int,
    outcomes: Sequence[RecordOutcome],
    expected_sha256: Sequence[str],
    validated_wall_seconds: float,
) -> dict[str, object]:
    if not outcomes or validated_wall_seconds <= 0:
        raise BenchmarkContractError(
            "a result requires outcomes and positive wall time"
        )
    if len(expected_sha256) != len(outcomes):
        raise BenchmarkContractError("expected hashes must cover every outcome")
    logical_records = len(outcomes)
    logical_bytes = logical_records * record_bytes
    service_times = [outcome.service_time_ms for outcome in outcomes]
    service_total_ms = sum(service_times)
    host_calls = sum(outcome.accounting.host_calls for outcome in outcomes)
    returned_bytes = sum(outcome.accounting.returned_bytes for outcome in outcomes)
    publications = sum(outcome.logical_publications for outcome in outcomes)
    observed_sha256 = [outcome.sha256 for outcome in outcomes]
    expected_hashes = list(expected_sha256)
    return {
        "repeat": repeat,
        "phase": "selection" if repeat < 8 else "confirmation",
        "order_position": order_position,
        "stripe_count": stripe_count,
        "logical_records": logical_records,
        "logical_bytes": logical_bytes,
        "logical_record_publications": publications,
        "host_calls": host_calls,
        "returned_bytes": returned_bytes,
        "byte_amplification": returned_bytes / logical_bytes,
        "host_calls_per_record": host_calls / logical_records,
        "validated_wall_seconds": validated_wall_seconds,
        "useful_gib_per_second": logical_bytes / 1024**3 / (service_total_ms / 1e3),
        "validated_wall_gib_per_second": logical_bytes
        / 1024**3
        / validated_wall_seconds,
        "service_time_ms": {
            "mean": statistics.fmean(service_times),
            "p50": _percentile(service_times, 0.50),
            "p95": _percentile(service_times, 0.95),
            "max": max(service_times),
            "total": service_total_ms,
        },
        "expected_sha256": expected_hashes,
        "observed_sha256": observed_sha256,
        "hash_parity": observed_sha256 == expected_hashes
        and publications == logical_records,
    }


def paired_bootstrap_interval(
    values: Sequence[float], *, seed: int, resamples: int
) -> dict[str, float | int]:
    """Return a deterministic paired percentile-bootstrap interval."""

    sample = [float(value) for value in values]
    if len(sample) < 2:
        raise BenchmarkContractError("paired intervals require at least two repeats")
    if resamples < 100:
        raise BenchmarkContractError("bootstrap resamples must be at least 100")
    rng = random.Random(seed)
    count = len(sample)
    means = sorted(
        statistics.fmean(sample[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    return {
        "mean": statistics.fmean(sample),
        "low": _percentile(means, 0.025),
        "high": _percentile(means, 0.975),
        "pairs": count,
        "positive_pairs": sum(value > 0 for value in sample),
        "zero_pairs": sum(value == 0 for value in sample),
        "negative_pairs": sum(value < 0 for value in sample),
    }


def _service_metric(row: dict[str, object], name: str) -> float:
    service = row.get("service_time_ms")
    if not isinstance(service, dict):
        raise BenchmarkContractError("result is missing service_time_ms")
    value = service.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkContractError(f"result is missing service_time_ms.{name}")
    return float(value)


def _paired_summary(
    pairs: Sequence[tuple[dict[str, object], dict[str, object]]],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    bandwidth = [
        float(candidate["useful_gib_per_second"])
        - float(control["useful_gib_per_second"])
        for control, candidate in pairs
    ]
    relative_bandwidth = [
        (
            float(candidate["useful_gib_per_second"])
            / float(control["useful_gib_per_second"])
            - 1.0
        )
        * 100.0
        for control, candidate in pairs
    ]
    saved_ms = [
        _service_metric(control, "mean") - _service_metric(candidate, "mean")
        for control, candidate in pairs
    ]
    p95_delta = [
        _service_metric(candidate, "p95") - _service_metric(control, "p95")
        for control, candidate in pairs
    ]
    return {
        "useful_gib_per_second_delta": paired_bootstrap_interval(
            bandwidth, seed=seed + 1, resamples=bootstrap_resamples
        ),
        "relative_useful_gib_per_second_delta_pct": paired_bootstrap_interval(
            relative_bandwidth, seed=seed + 2, resamples=bootstrap_resamples
        ),
        "milliseconds_saved_per_record": paired_bootstrap_interval(
            saved_ms, seed=seed + 3, resamples=bootstrap_resamples
        ),
        "p95_service_time_delta_ms": paired_bootstrap_interval(
            p95_delta, seed=seed + 4, resamples=bootstrap_resamples
        ),
    }


def analyze_results(
    results: Sequence[dict[str, object]], *, seed: int, bootstrap_resamples: int
) -> dict[str, object]:
    """Apply the paired correctness/performance gate without a fixed gain cutoff."""

    indexed: dict[tuple[int, int], dict[str, object]] = {}
    for row in results:
        repeat = row.get("repeat")
        width = row.get("stripe_count")
        if not isinstance(repeat, int) or not isinstance(width, int):
            raise BenchmarkContractError(
                "every result needs integer repeat/stripe_count"
            )
        key = (repeat, width)
        if key in indexed:
            raise BenchmarkContractError(
                f"duplicate result for repeat={repeat}, width={width}"
            )
        indexed[key] = row
    repeats = sorted({repeat for repeat, _width in indexed})
    if repeats != list(range(16)):
        raise BenchmarkContractError("analysis requires exactly repeats 0 through 15")
    for repeat in repeats:
        for width in ALLOWED_STRIPE_COUNTS:
            if (repeat, width) not in indexed:
                raise BenchmarkContractError(
                    f"missing result for repeat={repeat}, width={width}"
                )

    candidates: dict[str, dict[str, object]] = {}
    selection_passed: list[int] = []
    for candidate_width in ALLOWED_STRIPE_COUNTS[1:]:
        phase_pairs = {
            "selection": [
                (indexed[(repeat, 1)], indexed[(repeat, candidate_width)])
                for repeat in range(8)
            ],
            "confirmation": [
                (indexed[(repeat, 1)], indexed[(repeat, candidate_width)])
                for repeat in range(8, 16)
            ],
            "full": [
                (indexed[(repeat, 1)], indexed[(repeat, candidate_width)])
                for repeat in range(16)
            ],
        }
        summaries = {
            name: _paired_summary(
                pairs,
                seed=seed + candidate_width * 100 + index * 10,
                bootstrap_resamples=bootstrap_resamples,
            )
            for index, (name, pairs) in enumerate(phase_pairs.items())
        }
        candidate_rows = [indexed[(repeat, candidate_width)] for repeat in repeats]
        failed: list[str] = []
        selection_failed: list[str] = []
        for phase in ("selection", "confirmation"):
            phase_summary = summaries[phase]
            bandwidth = phase_summary["useful_gib_per_second_delta"]
            saved = phase_summary["milliseconds_saved_per_record"]
            if float(bandwidth["low"]) <= 0:
                failed.append(f"{phase}_useful_gib_per_second")
                if phase == "selection":
                    selection_failed.append(f"{phase}_useful_gib_per_second")
            if float(saved["low"]) <= 0:
                failed.append(f"{phase}_milliseconds_saved_per_record")
                if phase == "selection":
                    selection_failed.append(f"{phase}_milliseconds_saved_per_record")
            p95 = phase_summary["p95_service_time_delta_ms"]
            if float(p95["high"]) > 0:
                failed.extend((f"{phase}_p95_regression", "p95_regression"))
                if phase == "selection":
                    selection_failed.extend(
                        (f"{phase}_p95_regression", "p95_regression")
                    )
        if any(float(row["byte_amplification"]) != 1.0 for row in candidate_rows):
            failed.append("byte_amplification")
            if any(
                float(indexed[(repeat, candidate_width)]["byte_amplification"]) != 1.0
                for repeat in range(8)
            ):
                selection_failed.append("byte_amplification")
        if any(
            row.get("hash_parity") is not True
            or row.get("logical_record_publications") != row.get("logical_records")
            for row in candidate_rows
        ):
            failed.append("exact_hash_or_publication_parity")
            if any(
                indexed[(repeat, candidate_width)].get("hash_parity") is not True
                or indexed[(repeat, candidate_width)].get("logical_record_publications")
                != indexed[(repeat, candidate_width)].get("logical_records")
                for repeat in range(8)
            ):
                selection_failed.append("exact_hash_or_publication_parity")
        if any(
            row.get("returned_bytes") != row.get("logical_bytes")
            for row in candidate_rows
        ):
            failed.append("returned_byte_parity")
            if any(
                indexed[(repeat, candidate_width)].get("returned_bytes")
                != indexed[(repeat, candidate_width)].get("logical_bytes")
                for repeat in range(8)
            ):
                selection_failed.append("returned_byte_parity")
        failed = list(dict.fromkeys(failed))
        selection_failed = list(dict.fromkeys(selection_failed))
        if not selection_failed:
            selection_passed.append(candidate_width)
        candidates[str(candidate_width)] = {
            **summaries,
            "maintenance_cost": {
                "assessment": "acceptable for an isolated benchmark",
                "runtime_integration_requires_separate_review": True,
                "preference": "choose the narrowest passing width",
            },
            "failed_gates": failed,
            "selection_failed_gates": selection_failed,
            "decision": "pending_confirmation",
        }

    preselected = min(selection_passed) if selection_passed else None
    selected: int | None = None
    for candidate_width in ALLOWED_STRIPE_COUNTS[1:]:
        candidate = candidates[str(candidate_width)]
        failed = candidate["failed_gates"]
        assert isinstance(failed, list)
        if candidate_width != preselected:
            failed.append("not_preselected")
        failed[:] = list(dict.fromkeys(failed))
        if candidate_width == preselected and not failed:
            candidate["decision"] = "accept"
            selected = candidate_width
        else:
            candidate["decision"] = "reject"
    return {
        "interval_method": "paired percentile bootstrap, 95%",
        "bootstrap_resamples": bootstrap_resamples,
        "repeatability_rule": "selection and confirmation intervals must both pass",
        "fixed_percent_gain_cutoff": None,
        "selection": {
            "passing_stripe_counts": selection_passed,
            "selected_stripe_count": preselected,
            "policy": "freeze the narrowest selection-pass width before confirmation",
        },
        "candidates": candidates,
        "selected_stripe_count": selected,
        "decision": "accept" if selected is not None else "reject",
        "authorization": "microbenchmark evidence only; runtime integration remains unauthorized",
    }


def build_payload(
    *,
    command: Sequence[str],
    configuration: dict[str, object],
    workload: Sequence[dict[str, object]],
    provenance: dict[str, object],
    results: Sequence[dict[str, object]],
    analysis: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "mtplx-hy3-record-stripe-benchmark-v2",
        "mechanism": "within-record exact-demand preadv striping",
        "command": {"argv": list(command), "shell": shlex.join(command)},
        "configuration": configuration,
        "workload": list(workload),
        "provenance": provenance,
        "io_contract": {
            "control": "direct synchronous preadv",
            "candidates": "persistent executor with disjoint aligned preadv ranges",
            "host_call_semantics": "actual preadv attempts",
            "returned_byte_semantics": "actual positive bytes",
            "full_record_hash_per_operation": True,
            "one_logical_publication_per_validated_record": True,
            "cache_mode": "F_NOCACHE",
            "device_queue_depth_measured": False,
            "physical_nand_bytes_measured": False,
        },
        "scope": {
            "runtime_integration_authorized": False,
            "scheduler_change": False,
            "cache_policy_change": False,
            "mlx_or_metal_work": False,
        },
        "results": list(results),
        "analysis": analysis,
    }


def validate_config(
    *,
    stripe_counts: Sequence[int],
    operations: int,
    warmup_operations: int,
    repeats: int,
    bootstrap_resamples: int,
) -> tuple[int, ...]:
    widths = tuple(stripe_counts)
    if set(widths) != set(ALLOWED_STRIPE_COUNTS) or len(widths) != 4:
        raise BenchmarkContractError("stripe counts must contain exactly 1,2,4,8")
    for name, value, minimum in (
        ("operations", operations, 1),
        ("warmup operations", warmup_operations, 0),
        ("bootstrap resamples", bootstrap_resamples, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise BenchmarkContractError(f"{name} must be an integer >= {minimum}")
    if repeats != 16:
        raise BenchmarkContractError(
            "repeats must be exactly 16: eight balanced selection and eight confirmation"
        )
    return widths


def manifest_record_ranges(manifest: Any) -> tuple[RecordRange, ...]:
    """Fail closed unless every manifest record has the pinned Hy3 geometry."""

    if getattr(manifest, "model_key", None) != "hy3-q4":
        raise BenchmarkContractError("manifest model_key must be hy3-q4")
    sidecar = getattr(manifest, "sidecar", None)
    if sidecar is None:
        raise BenchmarkContractError("manifest must contain a sidecar")
    if getattr(sidecar, "alignment", None) != ALIGNMENT_BYTES:
        raise BenchmarkContractError("sidecar alignment must be 16 KiB")
    converted: list[RecordRange] = []
    for record in getattr(manifest, "records", ()):
        offset = getattr(record, "sidecar_offset", None)
        length = getattr(record, "sidecar_length", None)
        logical_bytes = getattr(record, "logical_bytes", None)
        digest = getattr(record, "sha256", None)
        if length != HY3_RECORD_BYTES or logical_bytes != HY3_RECORD_BYTES:
            raise BenchmarkContractError(
                "every Hy3 record must be exactly 10,616,832 bytes"
            )
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise BenchmarkContractError(
                "every record needs a non-negative sidecar offset"
            )
        if offset % ALIGNMENT_BYTES:
            raise BenchmarkContractError("every sidecar record offset must be aligned")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BenchmarkContractError("every record needs a lowercase SHA-256 hash")
        converted.append(
            RecordRange(
                identity=f"layer-{record.layer}-expert-{record.expert}",
                offset=offset,
                length=length,
                sha256=digest,
            )
        )
    if not converted:
        raise BenchmarkContractError("manifest contains no records")
    expected_size = max(record.offset + record.length for record in converted)
    if getattr(sidecar, "size", None) != expected_size:
        raise BenchmarkContractError(
            "sidecar size does not match contiguous record ranges"
        )
    return tuple(converted)


def measure_records(
    fd: int,
    records: Sequence[RecordRange],
    destination: bytearray,
    *,
    stripe_count: int,
    executor: Any | None,
) -> tuple[list[RecordOutcome], float]:
    outcomes: list[RecordOutcome] = []
    started = time.perf_counter()
    for record in records:
        outcomes.append(
            read_and_validate_record(
                fd,
                record,
                destination,
                stripe_count=stripe_count,
                executor=executor,
            )
        )
    return outcomes, time.perf_counter() - started


def observe_record_residency(
    residency: Any, records: Sequence[RecordRange]
) -> dict[str, object]:
    if not records:
        raise BenchmarkContractError("residency verification requires records")
    pages = 0
    resident_pages = 0
    referenced_pages = 0
    for record in records:
        snapshot = residency.residency(record.offset, record.length)
        pages += int(snapshot.pages)
        resident_pages += int(snapshot.resident_pages)
        referenced_pages += int(snapshot.referenced_pages)
    fraction = resident_pages / pages if pages else 1.0
    return {
        "pages": pages,
        "resident_pages": resident_pages,
        "referenced_pages": referenced_pages,
        "resident_fraction": fraction,
        "page_size": PAGE_SIZE,
        "verified_cold": fraction < _COLD_RESIDENT_FRACTION_LIMIT,
        "cold_resident_fraction_limit": _COLD_RESIDENT_FRACTION_LIMIT,
    }


def prepare_cold_records(
    residency: Any, records: Sequence[RecordRange]
) -> dict[str, object]:
    eviction_calls_succeeded = 0
    for record in records:
        if residency.try_evict(record.offset, record.length):
            eviction_calls_succeeded += 1
    snapshot = observe_record_residency(residency, records)
    snapshot["eviction_calls"] = len(records)
    snapshot["eviction_calls_succeeded"] = eviction_calls_succeeded
    if snapshot["verified_cold"] is not True:
        raise BenchmarkContractError(
            "could not establish verified-cold sidecar records before measurement"
        )
    return snapshot


def measure_schedule(
    fd: int,
    warm_records: Sequence[RecordRange],
    records: Sequence[RecordRange],
    destination: bytearray,
    *,
    stripe_counts: Sequence[int],
    repeats: int,
    executors: dict[int, Any | None],
    residency: Any,
) -> list[dict[str, object]]:
    if not records:
        raise BenchmarkContractError("measured workload cannot be empty")
    record_bytes = records[0].length
    results: list[dict[str, object]] = []
    for repeat in range(repeats):
        for position, stripe_count in enumerate(balanced_order(repeat, stripe_counts)):
            executor = executors[stripe_count]
            if warm_records:
                prepare_cold_records(residency, warm_records)
                measure_records(
                    fd,
                    warm_records,
                    destination,
                    stripe_count=stripe_count,
                    executor=executor,
                )
            cache_before = prepare_cold_records(residency, records)
            outcomes, wall_seconds = measure_records(
                fd,
                records,
                destination,
                stripe_count=stripe_count,
                executor=executor,
            )
            cache_after = observe_record_residency(residency, records)
            if cache_after["verified_cold"] is not True:
                raise BenchmarkContractError(
                    "F_NOCACHE arm populated the buffer cache; refusing SSD claim"
                )
            results.append(
                result_row(
                    repeat=repeat,
                    order_position=position,
                    stripe_count=stripe_count,
                    record_bytes=record_bytes,
                    outcomes=outcomes,
                    expected_sha256=[record.sha256 for record in records],
                    validated_wall_seconds=wall_seconds,
                )
            )
            row = results[-1]
            row["cache_before"] = cache_before
            row["cache_after"] = cache_after
            print(
                f"repeat={repeat:02d} position={position} stripes={stripe_count} "
                f"useful={float(row['useful_gib_per_second']):.3f} GiB/s "
                f"p95={float(row['service_time_ms']['p95']):.3f} ms",
                file=sys.stderr,
                flush=True,
            )
    return results


def select_workload(
    records: Sequence[RecordRange],
    *,
    operations: int,
    warmup_operations: int,
    seed: int,
) -> tuple[tuple[RecordRange, ...], tuple[RecordRange, ...]]:
    if operations + warmup_operations > len(records):
        raise BenchmarkContractError(
            "operations plus warmup operations exceed available unique records"
        )
    selected = random.Random(seed).sample(list(records), operations + warmup_operations)
    warm = tuple(selected[:warmup_operations])
    measured = tuple(selected[warmup_operations:])
    return warm, measured


@contextmanager
def open_sidecar_nocache(path: Path, *, expected_size: int):
    resolved = path.expanduser().resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    try:
        actual_size = os.fstat(fd).st_size
        if actual_size != expected_size:
            raise BenchmarkContractError(
                f"sidecar size changed: {actual_size} != {expected_size}"
            )
        nocache = getattr(fcntl, "F_NOCACHE", None)
        if nocache is None:
            raise BenchmarkContractError("F_NOCACHE is unavailable on this host")
        fcntl.fcntl(fd, nocache, 1)
        yield fd
    finally:
        os.close(fd)


def _parse_stripe_counts(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "stripe counts must be comma-separated integers"
        ) from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--stripe-counts", type=_parse_stripe_counts, default=ALLOWED_STRIPE_COUNTS
    )
    parser.add_argument("--operations", type=_positive_int, default=96)
    parser.add_argument("--warmup-operations", type=_nonnegative_int, default=32)
    parser.add_argument("--repeats", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=20_000)
    parser.add_argument("--output-json", type=Path)
    return parser


def git_provenance() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, timeout=5
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_ROOT,
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        raise BenchmarkContractError(
            f"git provenance could not be verified: {exc}"
        ) from exc
    if not commit:
        raise BenchmarkContractError("git provenance returned an empty commit")
    return {"git_commit": commit, "git_dirty": bool(status), "git_status": status}


def verify_final_provenance(
    *, initial_git: dict[str, object], initial_harness_sha256: str, source: Path
) -> dict[str, object]:
    final_git = git_provenance()
    final_harness_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
        final_git["git_commit"] != initial_git["git_commit"]
        or final_git["git_dirty"] is not False
        or final_git["git_status"] != initial_git["git_status"]
        or final_harness_sha256 != initial_harness_sha256
    ):
        raise BenchmarkContractError(
            "git state or benchmark harness changed during measurement"
        )
    return {
        "post_run_git_commit": final_git["git_commit"],
        "post_run_git_dirty": final_git["git_dirty"],
        "post_run_git_status": final_git["git_status"],
        "post_run_harness_sha256": final_harness_sha256,
    }


def _host_provenance() -> dict[str, object]:
    def sysctl(name: str) -> str | None:
        try:
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", name], text=True, timeout=2
            ).strip()
        except Exception:
            return None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": sysctl("hw.memsize"),
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    from mtplx.expert_manifest import load_expert_manifest, resolve_artifact_member

    args = build_parser().parse_args(argv)
    widths = validate_config(
        stripe_counts=args.stripe_counts,
        operations=args.operations,
        warmup_operations=args.warmup_operations,
        repeats=args.repeats,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    model_root = args.model_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if not model_root.is_dir():
        raise BenchmarkContractError(f"model root does not exist: {model_root}")
    manifest = load_expert_manifest(manifest_path)
    ranges = manifest_record_ranges(manifest)
    warm_records, records = select_workload(
        ranges,
        operations=args.operations,
        warmup_operations=args.warmup_operations,
        seed=args.seed,
    )
    sidecar = resolve_artifact_member(model_root, manifest.sidecar.file)
    source = Path(__file__).resolve()
    initial_harness_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    git = git_provenance()
    if git["git_dirty"] is not False:
        raise BenchmarkContractError(
            f"benchmark requires a clean committed worktree:\n{git['git_status']}"
        )
    destination = bytearray(HY3_RECORD_BYTES)
    executors: dict[int, Any | None] = {1: None}
    executors.update(
        {
            width: ThreadPoolExecutor(
                max_workers=width, thread_name_prefix=f"hy3-record-stripe-{width}"
            )
            for width in widths
            if width != 1
        }
    )
    try:
        with (
            FileResidency(sidecar) as residency,
            open_sidecar_nocache(sidecar, expected_size=manifest.sidecar.size) as fd,
        ):
            results = measure_schedule(
                fd,
                warm_records,
                records,
                destination,
                stripe_counts=widths,
                repeats=args.repeats,
                executors=executors,
                residency=residency,
            )
    finally:
        for executor in executors.values():
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    analysis = analyze_results(
        results,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    workload = [
        {
            "identity": record.identity,
            "offset": record.offset,
            "length": record.length,
            "expected_sha256": record.sha256,
        }
        for record in records
    ]
    workload_sha256 = hashlib.sha256(
        json.dumps(workload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    final_provenance = verify_final_provenance(
        initial_git=git,
        initial_harness_sha256=initial_harness_sha256,
        source=source,
    )
    command = [sys.executable, str(source), *(argv or sys.argv[1:])]
    payload = build_payload(
        command=command,
        configuration={
            "stripe_counts": list(widths),
            "operations": args.operations,
            "warmup_operations": args.warmup_operations,
            "repeats": args.repeats,
            "selection_repeats": list(range(8)),
            "confirmation_repeats": list(range(8, 16)),
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "record_bytes": HY3_RECORD_BYTES,
            "alignment_bytes": ALIGNMENT_BYTES,
            "workload_sha256": workload_sha256,
        },
        workload=workload,
        provenance={
            "created_at_utc": datetime.now(UTC).isoformat(),
            **git,
            **final_provenance,
            "harness_path": str(source.relative_to(_ROOT)),
            "harness_sha256": initial_harness_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest.manifest_sha256,
            "sidecar_path": str(sidecar),
            "sidecar_sha256": manifest.sidecar.sha256,
            "sidecar_size": manifest.sidecar.size,
            "model_key": manifest.model_key,
            "source_repo": manifest.source_repo,
            "source_revision": manifest.source_revision,
            "host": _host_provenance(),
        },
        results=results,
        analysis=analysis,
    )
    if args.output_json is None:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        output = args.output_json.expanduser().resolve()
        _atomic_write_json(output, payload)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
