#!/usr/bin/env python3
"""Guarded conformance and latency probe for the PR #391 K20 Metal selector.

This module intentionally has no import-time MLX or MTPLX dependency.  The
canonical guard attestation, source hashes, and captured production rows are
validated before the GPU implementation is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
EXPECTED_ROWS = 3338
EXPECTED_WIDTH = 20
EXPECTED_TOP_P = 0.95
EXPECTED_NUMPY_VERSION = "2.4.4"
EXPECTED_DRIVER_SHA256 = (
    "0ae20c7c4028cea83d9b9084d29067925d6dca08ff0ca2ce5a4ea9d73b9bb7d0"
)
CAPTURE_SOURCE_COMMIT = "340c153375de864740151c8c7a4c6368fd4af745"
REVIEWED_CAPTURE_SHA256 = (
    "8b43a1734a756627790d8e4ac033731dc6b3705d690b400dc0800365c9782a6f"
)
EXPECTED_COUNTERS = {
    "drafted_tokens": 3338,
    "accepted_drafts": 1656,
    "verify_calls": 1146,
    "correction_tokens": 789,
    "bonus_tokens": 342,
}
CAPTURE_ARRAYS = {
    "candidate_ids",
    "candidate_values",
    "candidate_probs",
    "uniforms",
    "selected_tokens",
    "rng_pre_sha256",
    "rng_post_sha256",
    "metadata_json",
}
REQUIRED_EXPECTED_FILES = {
    "mtplx/kernels/qwen4_frspec_k20_float32_choice.py",
    "scripts/pr391_float32_choice_drift.py",
    "scripts/pr391_metal_float32_selector_bench.py",
}
SELECTOR_MODULE = "mtplx.kernels.qwen4_frspec_k20_float32_choice"
DESCRIPTOR_BUILDER = "build_pcg64_midpoint_descriptors"
SELECTOR_BINDER = "bind_qwen4_frspec_k20_float32_choice"
MAX_MISMATCH_INDICES = 64
MAX_GUARD_ATTESTATION_NS = 60_000_000_000


class BenchmarkContractError(RuntimeError):
    """Base class for a fail-closed benchmark contract violation."""


class GuardAttestationError(BenchmarkContractError):
    """The process is not the child currently owned by the canonical guard."""


class SourceContractError(BenchmarkContractError):
    """The source revision or a required file hash does not match."""


class CaptureHashMismatch(BenchmarkContractError):
    """The captured production NPZ is not the expected immutable artifact."""


class CaptureContractError(BenchmarkContractError):
    """The capture does not describe the reviewed K20 production workload."""


def _validate_attestation_window(receipt: Mapping[str, Any], *, now: int) -> None:
    required_ints = (
        "schema_version",
        "guard_pid",
        "child_pid",
        "lock_device",
        "lock_inode",
        "issued_monotonic_ns",
        "expires_monotonic_ns",
    )
    if any(
        isinstance(receipt.get(key), bool) or not isinstance(receipt.get(key), int)
        for key in required_ints
    ):
        raise GuardAttestationError("guard attestation integer fields are malformed")
    issued = receipt["issued_monotonic_ns"]
    expires = receipt["expires_monotonic_ns"]
    if (
        receipt["schema_version"] != 1
        or not issued <= now <= expires
        or expires - issued > MAX_GUARD_ATTESTATION_NS
    ):
        raise GuardAttestationError("guard attestation is stale or has an invalid window")


@dataclass(frozen=True)
class CaptureRows:
    candidate_ids: np.ndarray
    candidate_values: np.ndarray
    candidate_probs: np.ndarray
    uniforms: np.ndarray
    selected_tokens: np.ndarray
    rng_pre_sha256: np.ndarray
    rng_post_sha256: np.ndarray
    metadata: Mapping[str, Any]
    path: Path
    sha256: str


@dataclass(frozen=True)
class ExpectedTokens:
    reduced_exact: np.ndarray
    reduced_float32: np.ndarray


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def consume_guard_attestation() -> dict[str, Any]:
    """Consume and verify the canonical guard's one-use in-process receipt."""

    raw_fd = os.environ.pop("MTPLX_GUARD_ATTEST_FD", None)
    nonce = os.environ.pop("MTPLX_GUARD_ATTEST_NONCE", None)
    if raw_fd is None or nonce is None:
        raise GuardAttestationError("canonical GPU guard attestation is required")
    try:
        descriptor = int(raw_fd)
    except ValueError as exc:
        raise GuardAttestationError("guard attestation descriptor is invalid") from exc
    payload = bytearray()
    try:
        while len(payload) <= 16 * 1024:
            chunk = os.read(descriptor, 16 * 1024 + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
    finally:
        os.close(descriptor)
    if len(payload) > 16 * 1024:
        raise GuardAttestationError("guard attestation payload is oversized")
    try:
        receipt = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GuardAttestationError("guard attestation payload is invalid") from exc

    if not isinstance(receipt, dict):
        raise GuardAttestationError("guard attestation payload must be an object")
    _validate_attestation_window(receipt, now=time.monotonic_ns())

    lock_path = LOCK.resolve(strict=True)
    lock_status = lock_path.lstat()
    expected = {
        "nonce": nonce,
        "child_pid": os.getpid(),
        "guard_pid": os.getppid(),
        "lock_path": str(lock_path),
        "lock_device": lock_status.st_dev,
        "lock_inode": lock_status.st_ino,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise GuardAttestationError(f"guard attestation mismatch: {key}")
    if (
        not stat.S_ISREG(lock_status.st_mode)
        or lock_status.st_nlink != 1
        or stat.S_IMODE(lock_status.st_mode) != 0o600
        or lock_status.st_uid != os.getuid()
    ):
        raise GuardAttestationError("canonical GPU lock identity is unsafe")
    with lock_path.open("r+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            raise GuardAttestationError(
                "guard claims the lane but the canonical GPU lock is free"
            )
    return dict(receipt)


def _git_head(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceContractError("could not resolve source git revision") from exc


def _validate_hex(value: str, *, width: int, label: str) -> str:
    if len(value) != width or any(character not in "0123456789abcdef" for character in value):
        raise SourceContractError(f"{label} must be {width} lowercase hexadecimal characters")
    return value


def validate_reviewed_capture(*, expected_capture_sha256: str) -> None:
    """Prevent callers from substituting a self-chosen capture artifact."""

    if expected_capture_sha256 != REVIEWED_CAPTURE_SHA256:
        raise CaptureHashMismatch(
            "expected-capture-sha256 does not identify the reviewed capture"
        )


def verify_source(
    source: Path,
    expected_benchmark_source: str,
    expected_files: Mapping[str, str],
) -> dict[str, Any]:
    """Pin the current benchmark revision and files independently of capture."""

    source = source.resolve(strict=True)
    expected_benchmark_source = _validate_hex(
        str(expected_benchmark_source),
        width=40,
        label="expected-benchmark-source",
    )
    actual_source = _git_head(source)
    if actual_source != expected_benchmark_source:
        raise SourceContractError(
            "source commit mismatch: expected benchmark HEAD "
            f"{expected_benchmark_source}, got {actual_source}"
        )
    missing = sorted(REQUIRED_EXPECTED_FILES - set(expected_files))
    if missing:
        raise SourceContractError(
            f"required expected-file hashes missing: {', '.join(missing)}"
        )

    verified: dict[str, str] = {}
    for relative, expected_hash in expected_files.items():
        expected_hash = _validate_hex(
            str(expected_hash), width=64, label=f"expected-file {relative}"
        )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SourceContractError(f"expected-file path is unsafe: {relative}")
        path = (source / relative_path).resolve(strict=True)
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise SourceContractError(
                f"expected-file escapes source tree: {relative}"
            ) from exc
        actual_hash = _sha256_path(path)
        if actual_hash != expected_hash:
            raise SourceContractError(
                f"source file SHA-256 mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        verified[str(relative_path)] = actual_hash
    return {
        "benchmark_commit": actual_source,
        "files": verified,
        "path": str(source),
    }


def _metadata_json(value: np.ndarray) -> dict[str, Any]:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise CaptureContractError("metadata_json must be one scalar string")
    try:
        metadata = json.loads(str(value.item()))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CaptureContractError("metadata_json is invalid") from exc
    if not isinstance(metadata, dict):
        raise CaptureContractError("metadata_json must contain an object")
    return metadata


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    value = arrays[name]
    if value.shape != shape:
        raise CaptureContractError(
            f"{name} must have shape {shape}; exact capture requires 3338 rows"
        )
    if value.dtype != dtype:
        raise CaptureContractError(f"{name} must have dtype {dtype}")
    return value


def load_capture(
    path: Path,
    *,
    expected_sha256: str,
    expected_capture_source: str,
    top_p: float,
) -> CaptureRows:
    """Hash first, then load and validate the exact 3338-row NPZ contract."""

    path = path.resolve(strict=True)
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise CaptureHashMismatch(
            "expected capture SHA-256 must be 64 lowercase hexadecimal characters"
        )
    actual_sha256 = _sha256_path(path)
    if actual_sha256 != expected_sha256:
        raise CaptureHashMismatch(
            "capture SHA-256 mismatch: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )
    if float(top_p) != EXPECTED_TOP_P:
        raise CaptureContractError("isolated selector benchmark requires top_p=0.95")

    try:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != CAPTURE_ARRAYS:
                missing = sorted(CAPTURE_ARRAYS - set(saved.files))
                extra = sorted(set(saved.files) - CAPTURE_ARRAYS)
                raise CaptureContractError(
                    f"capture array schema mismatch: missing={missing} extra={extra}"
                )
            arrays = {name: saved[name].copy() for name in saved.files}
    except CaptureContractError:
        raise
    except (OSError, ValueError) as exc:
        raise CaptureContractError("capture NPZ could not be loaded safely") from exc

    ids = _require_array(
        arrays,
        "candidate_ids",
        shape=(EXPECTED_ROWS, EXPECTED_WIDTH),
        dtype=np.dtype(np.int64),
    )
    values = _require_array(
        arrays,
        "candidate_values",
        shape=(EXPECTED_ROWS, EXPECTED_WIDTH),
        dtype=np.dtype(np.float32),
    )
    probs = _require_array(
        arrays,
        "candidate_probs",
        shape=(EXPECTED_ROWS, EXPECTED_WIDTH),
        dtype=np.dtype(np.float32),
    )
    uniforms = _require_array(
        arrays,
        "uniforms",
        shape=(EXPECTED_ROWS,),
        dtype=np.dtype(np.float64),
    )
    selected = _require_array(
        arrays,
        "selected_tokens",
        shape=(EXPECTED_ROWS,),
        dtype=np.dtype(np.int64),
    )
    rng_pre = _require_array(
        arrays,
        "rng_pre_sha256",
        shape=(EXPECTED_ROWS,),
        dtype=np.dtype("S64"),
    )
    rng_post = _require_array(
        arrays,
        "rng_post_sha256",
        shape=(EXPECTED_ROWS,),
        dtype=np.dtype("S64"),
    )
    metadata = _metadata_json(arrays["metadata_json"])

    if np.any(ids < 0) or np.any(ids > np.iinfo(np.uint32).max):
        raise CaptureContractError("candidate_ids must round-trip through uint32")
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise CaptureContractError("candidate_ids must be unique within each row")
    if not np.all(np.isfinite(values)):
        raise CaptureContractError("candidate_values must be finite")
    if (
        not np.all(np.isfinite(probs))
        or np.any(probs < np.float32(0.0))
        or np.any(probs > np.float32(1.0))
    ):
        raise CaptureContractError("candidate_probs must be finite values in [0, 1]")
    if np.any(np.sum(probs, axis=1, dtype=np.float64) <= 0.0):
        raise CaptureContractError("each candidate_probs row must have positive mass")
    if not np.all(np.isfinite(uniforms)) or np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise CaptureContractError("uniforms must be finite values in [0, 1)")
    grid = np.asarray(uniforms * float(1 << 53), dtype=np.uint64)
    if not np.array_equal(np.ldexp(grid.astype(np.float64), -53), uniforms):
        raise CaptureContractError("uniforms must lie on the exact PCG64 53-bit grid")
    if np.any(selected < 0) or np.any(selected > np.iinfo(np.uint32).max):
        raise CaptureContractError("selected_tokens must round-trip through uint32")
    if not np.all(np.any(ids == selected[:, None], axis=1)):
        raise CaptureContractError("selected_tokens must belong to their candidate row")
    for name, hashes in (("rng_pre_sha256", rng_pre), ("rng_post_sha256", rng_post)):
        if any(
            len(value) != 64
            or any(character not in b"0123456789abcdef" for character in value)
            for value in hashes.tolist()
        ):
            raise CaptureContractError(f"{name} must contain lowercase SHA-256 values")

    required_metadata = {
        "capture_kind": "diagnostic_pre_top_p_draft_choices",
        "driver_sha256": EXPECTED_DRIVER_SHA256,
        "expected_counters": EXPECTED_COUNTERS,
        "float32_policy": "benchmark_experiment_only_not_retainable",
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "observed_counters": EXPECTED_COUNTERS,
        "row_count": EXPECTED_ROWS,
        "source_commit": expected_capture_source,
        "support_width": EXPECTED_WIDTH,
    }
    for key, expected_value in required_metadata.items():
        if metadata.get(key) != expected_value:
            raise CaptureContractError(
                f"capture metadata mismatch for {key}: "
                f"expected={expected_value!r} observed={metadata.get(key)!r}"
            )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise CaptureContractError(
            f"benchmark requires NumPy {EXPECTED_NUMPY_VERSION}; found {np.__version__}"
        )

    return CaptureRows(
        candidate_ids=ids,
        candidate_values=values,
        candidate_probs=probs,
        uniforms=uniforms,
        selected_tokens=selected,
        rng_pre_sha256=rng_pre,
        rng_post_sha256=rng_post,
        metadata=metadata,
        path=path,
        sha256=actual_sha256,
    )


def build_expected_tokens(capture: CaptureRows, analyzer: Any) -> ExpectedTokens:
    """Compute both reviewed midpoint schedules for every captured row."""

    row_count = int(capture.uniforms.shape[0])
    reduced_exact = np.empty(row_count, dtype=np.uint32)
    reduced_float32 = np.empty(row_count, dtype=np.uint32)
    for index in range(row_count):
        exact_row = analyzer.prepare_reduced_exact_row(
            capture.candidate_ids[index],
            capture.candidate_values[index],
            capture.candidate_probs[index],
            top_p=EXPECTED_TOP_P,
        )
        float32_row = analyzer.prepare_reduced_float32_row(
            capture.candidate_ids[index],
            capture.candidate_values[index],
            capture.candidate_probs[index],
            top_p=EXPECTED_TOP_P,
        )
        uniform = float(capture.uniforms[index])
        reduced_exact[index] = analyzer.select_reduced_exact_token(exact_row, uniform)
        reduced_float32[index] = analyzer.select_reduced_float32_token(
            float32_row,
            uniform,
            cast_uniform=False,
        )
    return ExpectedTokens(
        reduced_exact=reduced_exact,
        reduced_float32=reduced_float32,
    )


def _mismatch(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "mismatches": max(int(left.size), int(right.size)),
            "indices": [],
            "shape_mismatch": [list(left.shape), list(right.shape)],
        }
    indices = np.flatnonzero(left != right)
    return {
        "mismatches": int(indices.size),
        "indices": [int(value) for value in indices[:MAX_MISMATCH_INDICES]],
        "shape_mismatch": None,
    }


def _timing_summary(seconds: list[float]) -> dict[str, Any]:
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean_s": float(np.mean(values)),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def _memory_snapshot(mx: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _raw_output_report(
    raw_ids: Any,
    raw_values: Any,
    raw_probs: Any,
    *,
    expected_ids: np.ndarray,
    expected_values: np.ndarray,
    expected_probs: np.ndarray,
) -> tuple[dict[str, Any], int]:
    ids = np.asarray(raw_ids)
    values = np.asarray(raw_values)
    probs = np.asarray(raw_probs)
    ids_contract = ids.shape == expected_ids.shape and ids.dtype == np.dtype(np.uint32)
    values_contract = (
        values.shape == expected_values.shape and values.dtype == np.dtype(np.float32)
    )
    probs_contract = (
        probs.shape == expected_probs.shape and probs.dtype == np.dtype(np.float32)
    )
    ids_exact = bool(ids_contract and np.array_equal(ids, expected_ids))
    values_exact = bool(
        values_contract
        and np.array_equal(values.view(np.uint32), expected_values.view(np.uint32))
    )
    probs_exact = bool(
        probs_contract
        and np.array_equal(probs.view(np.uint32), expected_probs.view(np.uint32))
    )
    nonfinite = int(np.count_nonzero(~np.isfinite(values))) + int(
        np.count_nonzero(~np.isfinite(probs))
    )
    failures = sum(not value for value in (ids_contract, values_contract, probs_contract))
    failures += sum(not value for value in (ids_exact, values_exact, probs_exact))
    failures += int(nonfinite > 0)
    return (
        {
            "ids_shape_dtype": bool(ids_contract),
            "values_shape_dtype": bool(values_contract),
            "probs_shape_dtype": bool(probs_contract),
            "ids_bit_exact": ids_exact,
            "values_bit_exact": values_exact,
            "probs_bit_exact": probs_exact,
        },
        failures,
    )


def run_benchmark(
    *,
    mx: Any,
    selector: Any,
    descriptor_builder: Any,
    capture: CaptureRows,
    expected: ExpectedTokens,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    """Run batch conformance, isolated B1, and a dependent compiled D3 chain."""

    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    rows = int(capture.candidate_ids.shape[0])
    ids_host = capture.candidate_ids.astype(np.uint32, copy=True)
    if not np.array_equal(ids_host.astype(np.int64), capture.candidate_ids):
        raise CaptureContractError("candidate IDs do not round-trip through uint32")
    descriptor_host = np.asarray(descriptor_builder(capture.uniforms))
    if descriptor_host.shape != (rows, 5) or descriptor_host.dtype != np.dtype(np.uint32):
        raise BenchmarkContractError(
            "midpoint descriptor builder must return uint32[rows,5]"
        )

    memory_before = _memory_snapshot(mx)
    mx.reset_peak_memory()
    ids_device = mx.array(ids_host, dtype=mx.uint32)
    values_device = mx.array(capture.candidate_values, dtype=mx.float32)
    probs_device = mx.array(capture.candidate_probs, dtype=mx.float32)
    descriptor_device = mx.array(descriptor_host, dtype=mx.uint32)

    conformance_outputs = selector(
        ids_device, values_device, probs_device, descriptor_device
    )
    if not isinstance(conformance_outputs, (tuple, list)) or len(conformance_outputs) != 4:
        raise BenchmarkContractError(
            "selector must return selected plus three raw pass-through arrays"
        )
    mx.eval(*conformance_outputs)
    selected_device, raw_ids, raw_values, raw_probs = conformance_outputs
    selected = np.asarray(selected_device)
    selected_contract = selected.shape == (rows,) and selected.dtype == np.dtype(np.uint32)
    selected_compare = (
        selected
        if selected_contract
        else np.empty(0, dtype=np.uint32)
    )
    selected_mismatch = _mismatch(selected_compare, expected.reduced_float32)
    raw_report, raw_failures = _raw_output_report(
        raw_ids,
        raw_values,
        raw_probs,
        expected_ids=ids_host,
        expected_values=capture.candidate_values,
        expected_probs=capture.candidate_probs,
    )
    nonfinite_output_count = int(np.count_nonzero(~np.isfinite(np.asarray(raw_values))))
    nonfinite_output_count += int(np.count_nonzero(~np.isfinite(np.asarray(raw_probs))))

    for _ in range(warmups):
        outputs = selector(ids_device, values_device, probs_device, descriptor_device)
        mx.eval(*outputs)
    batch_seconds: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = selector(ids_device, values_device, probs_device, descriptor_device)
        mx.eval(*outputs)
        batch_seconds.append(time.perf_counter() - started)

    for index in range(warmups):
        row = index % rows
        outputs = selector(
            ids_device[row : row + 1],
            values_device[row : row + 1],
            probs_device[row : row + 1],
            descriptor_device[row : row + 1],
        )
        mx.eval(*outputs)
    b1_seconds: list[float] = []
    for index in range(repeats):
        row = index % rows
        started = time.perf_counter()
        outputs = selector(
            ids_device[row : row + 1],
            values_device[row : row + 1],
            probs_device[row : row + 1],
            descriptor_device[row : row + 1],
        )
        mx.eval(*outputs)
        b1_seconds.append(time.perf_counter() - started)

    def dependent_d3(start_index: Any):
        index = start_index
        selected_tokens: list[Any] = []
        for _ in range(3):
            row_ids = mx.take(ids_device, index, axis=0)
            row_values = mx.take(values_device, index, axis=0)
            row_probs = mx.take(probs_device, index, axis=0)
            row_descriptor = mx.take(descriptor_device, index, axis=0)
            selected_token, _, _, _ = selector(
                row_ids, row_values, row_probs, row_descriptor
            )
            selected_tokens.append(selected_token)
            index = selected_token % rows
        return tuple(selected_tokens)

    compile_started = time.perf_counter()
    compiled_d3 = mx.compile(dependent_d3)
    compiled_smoke = compiled_d3(mx.array([0], dtype=mx.uint32))
    mx.eval(*compiled_smoke)
    compile_seconds = time.perf_counter() - compile_started
    for index in range(warmups):
        outputs = compiled_d3(mx.array([index % rows], dtype=mx.uint32))
        mx.eval(*outputs)
    d3_seconds: list[float] = []
    for index in range(repeats):
        started = time.perf_counter()
        outputs = compiled_d3(mx.array([index % rows], dtype=mx.uint32))
        mx.eval(*outputs)
        d3_seconds.append(time.perf_counter() - started)

    batch = _timing_summary(batch_seconds)
    batch["rows"] = rows
    batch["rows_per_second"] = float(rows / batch["mean_s"])
    b1 = _timing_summary(b1_seconds)
    d3 = _timing_summary(d3_seconds)
    d3.update(
        {
            "depth": 3,
            "evals_per_sample": 1,
            "compile_and_smoke_s": float(compile_seconds),
            "mean_s_per_selector": float(d3["mean_s"] / 3.0),
        }
    )

    captured_selected = capture.selected_tokens.astype(np.uint32)
    drift_f32_exact = _mismatch(expected.reduced_float32, expected.reduced_exact)
    drift_f32_capture = _mismatch(expected.reduced_float32, captured_selected)
    failure_count = raw_failures
    failure_count += int(not selected_contract)
    failure_count += int(selected_mismatch["mismatches"] > 0)
    failure_count += int(nonfinite_output_count > 0)
    return {
        "schema_version": 1,
        "status": "pass" if failure_count == 0 else "fail",
        "policy": {
            "float32_test_only": True,
            "retention_eligible": False,
            "timing_is_diagnostic_not_e2e_tps": True,
        },
        "conformance": {
            "rows_checked": rows,
            "selected_shape_dtype": bool(selected_contract),
            "selected_vs_reduced_float32": selected_mismatch,
            "raw_passthrough": raw_report,
            "candidate_id_uint32_roundtrip": True,
            "nonfinite_output_count": nonfinite_output_count,
        },
        "drift": {
            "reduced_float32_vs_reduced_exact": drift_f32_exact,
            "reduced_float32_vs_captured_selected": drift_f32_capture,
        },
        "descriptor": {
            "dtype": "uint32",
            "shape": [rows, 5],
            "words": [
                "pcg64_grid_hi21",
                "pcg64_grid_lo32",
                "midpoint_significand",
                "midpoint_exponent_int32_bits",
                "upper_endpoint_even",
            ],
            "uniform_cast_to_float32": False,
        },
        "timing": {
            "batch": batch,
            "b1": b1,
            "dependent_d3_single_eval": d3,
        },
        "memory": {
            "before": memory_before,
            "after": _memory_snapshot(mx),
            "peak_bytes": int(mx.get_peak_memory()),
            "host_capture_bytes": int(
                capture.candidate_ids.nbytes
                + capture.candidate_values.nbytes
                + capture.candidate_probs.nbytes
                + capture.uniforms.nbytes
                + capture.selected_tokens.nbytes
                + capture.rng_pre_sha256.nbytes
                + capture.rng_post_sha256.nbytes
            ),
            "host_descriptor_bytes": int(descriptor_host.nbytes),
            "device_input_static_bytes": int(
                ids_host.nbytes
                + capture.candidate_values.nbytes
                + capture.candidate_probs.nbytes
                + descriptor_host.nbytes
            ),
            "device_output_static_bytes": int(
                rows * np.dtype(np.uint32).itemsize
                + ids_host.nbytes
                + capture.candidate_values.nbytes
                + capture.candidate_probs.nbytes
            ),
        },
    }


def _parse_expected_files(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        relative, separator, digest = value.partition("=")
        if not separator or not relative or relative in parsed:
            raise SourceContractError(
                "expected-file must be one unique RELATIVE_PATH=SHA256 pair"
            )
        parsed[relative] = digest
    return parsed


def _import_hash_gated_module(source: Path, name: str, relative: str) -> Any:
    source = source.resolve(strict=True)
    source_string = str(source)
    sys.path[:] = [entry for entry in sys.path if entry != source_string]
    sys.path.insert(0, source_string)
    if name in sys.modules:
        raise SourceContractError(f"hash-gated module was preloaded: {name}")
    module = importlib.import_module(name)
    observed_file = getattr(module, "__file__", None)
    expected_file = (source / relative).resolve(strict=True)
    try:
        observed_path = Path(observed_file).resolve(strict=True)
    except (TypeError, OSError) as exc:
        sys.modules.pop(name, None)
        raise SourceContractError(f"hash-gated module has no valid file: {name}") from exc
    if observed_path != expected_file:
        sys.modules.pop(name, None)
        raise SourceContractError(
            f"hash-gated module came from {observed_path}, expected {expected_file}"
        )
    return module


def _load_gpu_api(source: Path, *, top_p: float) -> tuple[Any, Any, Any]:
    """Import MLX and the hash-gated selector only after guard/source/capture gates."""

    source = source.resolve(strict=True)
    mx = importlib.import_module("mlx.core")
    kernel = _import_hash_gated_module(
        source,
        SELECTOR_MODULE,
        "mtplx/kernels/qwen4_frspec_k20_float32_choice.py",
    )
    try:
        descriptor_builder = getattr(kernel, DESCRIPTOR_BUILDER)
        binder = getattr(kernel, SELECTOR_BINDER)
    except AttributeError as exc:
        raise BenchmarkContractError("selector module API does not match the reviewed contract") from exc
    selector = binder(top_p=top_p)
    if not callable(selector) or not callable(descriptor_builder):
        raise BenchmarkContractError("selector binder did not return callable APIs")
    return mx, selector, descriptor_builder


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-benchmark-source", required=True)
    parser.add_argument("--expected-file", action="append", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--expected-capture-sha256", required=True)
    parser.add_argument("--top-p", type=float, default=EXPECTED_TOP_P)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    guard = consume_guard_attestation()
    validate_reviewed_capture(
        expected_capture_sha256=args.expected_capture_sha256,
    )
    expected_files = _parse_expected_files(args.expected_file)
    source_receipt = verify_source(
        args.source,
        args.expected_benchmark_source,
        expected_files,
    )
    capture = load_capture(
        args.capture,
        expected_sha256=args.expected_capture_sha256,
        expected_capture_source=CAPTURE_SOURCE_COMMIT,
        top_p=args.top_p,
    )
    source = args.source.resolve(strict=True)
    analyzer = _import_hash_gated_module(
        source,
        "scripts.pr391_float32_choice_drift",
        "scripts/pr391_float32_choice_drift.py",
    )
    expected = build_expected_tokens(capture, analyzer)
    mx, selector, descriptor_builder = _load_gpu_api(source, top_p=args.top_p)
    report = run_benchmark(
        mx=mx,
        selector=selector,
        descriptor_builder=descriptor_builder,
        capture=capture,
        expected=expected,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    report["provenance"] = {
        "guard": guard,
        "benchmark_source": source_receipt,
        "capture": {
            "path": str(capture.path),
            "sha256": capture.sha256,
            "source_commit": CAPTURE_SOURCE_COMMIT,
        },
        "numpy_version": np.__version__,
        "top_p": float(args.top_p),
        "rows": EXPECTED_ROWS,
        "width": EXPECTED_WIDTH,
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded, flush=True)
    if args.output_json is not None:
        _write_json(args.output_json, report)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
