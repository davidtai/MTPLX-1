#!/usr/bin/env python3
"""Gate issue #31's same-byte packed Hy3 expert layout.

The control and candidate are checksum-producing Metal stream floors. They do
the same integer work and touch the same logical Q4/group-64 bytes; only the
addresses differ. The control reads split weight/scale/bias buffers. The
candidate reads four groups at a time from a 144-byte packed chunk.

This benchmark cannot authorize runtime integration. A win only authorizes a
later bit-exact packed-QMV experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    import mlx.core as mx
except (ImportError, OSError):  # pragma: no cover - exercised by non-Metal CI
    mx = None


GROUP_SIZE = 64
GROUPS_PER_CHUNK = 4
WEIGHT_BYTES_PER_GROUP = 32
PARAMETER_BYTES_PER_GROUP = 2
WEIGHT_BYTES_PER_CHUNK = GROUPS_PER_CHUNK * WEIGHT_BYTES_PER_GROUP
PARAMETER_BYTES_PER_CHUNK = GROUPS_PER_CHUNK * PARAMETER_BYTES_PER_GROUP
CHUNK_BYTES = WEIGHT_BYTES_PER_CHUNK + 2 * PARAMETER_BYTES_PER_CHUNK
TOP_K = 8
MINIMUM_PERCENT = 5.0
PROCESS_REPEATS = 4
GATE_N = 1536
GATE_K = 4096
DOWN_N = 4096
DOWN_K = 1536
_ARMS = ("split", "packed")
_LOCK_PATH = Path.home() / ".cache" / "mtplx" / "hy3-hardware-benchmark.lock"


class BenchmarkContractError(ValueError):
    """Raised when benchmark arms are no longer physically comparable."""


def _require_mlx() -> None:
    if mx is None:
        raise BenchmarkContractError(
            "MLX Metal is required for the packed-stream hardware benchmark"
        )


@dataclass(frozen=True)
class ProjectionLayout:
    k: int
    groups: int
    chunks: int
    split_row_bytes: int
    packed_row_bytes: int


@dataclass(frozen=True)
class ProjectionArrays:
    weight: Any
    scales: Any
    biases: Any
    packed: Any
    packing_ms: float


def projection_layout(k: int) -> ProjectionLayout:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise BenchmarkContractError("K must be a positive integer")
    chunk_values = GROUP_SIZE * GROUPS_PER_CHUNK
    if k % chunk_values:
        raise BenchmarkContractError("K must be divisible by 256")
    groups = k // GROUP_SIZE
    chunks = groups // GROUPS_PER_CHUNK
    split_row_bytes = groups * (WEIGHT_BYTES_PER_GROUP + 2 * PARAMETER_BYTES_PER_GROUP)
    packed_row_bytes = chunks * CHUNK_BYTES
    if split_row_bytes != packed_row_bytes:
        raise BenchmarkContractError("packed layout must not add or remove bytes")
    if packed_row_bytes % 16:
        raise BenchmarkContractError("packed row stride must be 16-byte aligned")
    return ProjectionLayout(
        k=k,
        groups=groups,
        chunks=chunks,
        split_row_bytes=split_row_bytes,
        packed_row_bytes=packed_row_bytes,
    )


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkContractError(f"{name} must be a positive integer")
    return value


def projection_bytes(*, capacity: int, n: int, k: int) -> int:
    capacity = _positive_int("capacity", capacity)
    n = _positive_int("N", n)
    return capacity * n * projection_layout(k).packed_row_bytes


def routed_layer_bytes(*, top_k: int) -> int:
    top_k = _positive_int("top_k", top_k)
    return top_k * (
        2 * projection_bytes(capacity=1, n=GATE_N, k=GATE_K)
        + projection_bytes(capacity=1, n=DOWN_N, k=DOWN_K)
    )


def _expected_component_lengths(
    *, capacity: int, n: int, k: int
) -> tuple[int, int, int, int]:
    layout = projection_layout(k)
    rows = _positive_int("capacity", capacity) * _positive_int("N", n)
    weight = rows * layout.groups * WEIGHT_BYTES_PER_GROUP
    parameter = rows * layout.groups * PARAMETER_BYTES_PER_GROUP
    packed = rows * layout.packed_row_bytes
    return weight, parameter, parameter, packed


def pack_projection_bytes(
    weight: bytes,
    scales: bytes,
    biases: bytes,
    *,
    capacity: int,
    n: int,
    k: int,
) -> bytes:
    expected_weight, expected_scale, expected_bias, expected_packed = (
        _expected_component_lengths(capacity=capacity, n=n, k=k)
    )
    for name, value, expected in (
        ("weight", weight, expected_weight),
        ("scale", scales, expected_scale),
        ("bias", biases, expected_bias),
    ):
        if len(value) != expected:
            raise BenchmarkContractError(
                f"{name} byte length must be exactly {expected}, got {len(value)}"
            )

    layout = projection_layout(k)
    output = bytearray(expected_packed)
    rows = capacity * n
    weight_row = layout.groups * WEIGHT_BYTES_PER_GROUP
    parameter_row = layout.groups * PARAMETER_BYTES_PER_GROUP
    for row in range(rows):
        weight_base = row * weight_row
        scale_base = row * parameter_row
        bias_base = row * parameter_row
        packed_base = row * layout.packed_row_bytes
        for chunk in range(layout.chunks):
            group = chunk * GROUPS_PER_CHUNK
            weight_start = weight_base + group * WEIGHT_BYTES_PER_GROUP
            scale_start = scale_base + group * PARAMETER_BYTES_PER_GROUP
            bias_start = bias_base + group * PARAMETER_BYTES_PER_GROUP
            chunk_start = packed_base + chunk * CHUNK_BYTES
            output[chunk_start : chunk_start + WEIGHT_BYTES_PER_CHUNK] = weight[
                weight_start : weight_start + WEIGHT_BYTES_PER_CHUNK
            ]
            output[
                chunk_start + WEIGHT_BYTES_PER_CHUNK : chunk_start
                + WEIGHT_BYTES_PER_CHUNK
                + PARAMETER_BYTES_PER_CHUNK
            ] = scales[scale_start : scale_start + PARAMETER_BYTES_PER_CHUNK]
            output[
                chunk_start
                + WEIGHT_BYTES_PER_CHUNK
                + PARAMETER_BYTES_PER_CHUNK : chunk_start + CHUNK_BYTES
            ] = biases[bias_start : bias_start + PARAMETER_BYTES_PER_CHUNK]
    return bytes(output)


def unpack_projection_bytes(
    packed: bytes,
    *,
    capacity: int,
    n: int,
    k: int,
) -> tuple[bytes, bytes, bytes]:
    weight_length, scale_length, bias_length, packed_length = (
        _expected_component_lengths(capacity=capacity, n=n, k=k)
    )
    if len(packed) != packed_length:
        raise BenchmarkContractError(
            f"packed byte length must be exactly {packed_length}, got {len(packed)}"
        )
    layout = projection_layout(k)
    weight = bytearray(weight_length)
    scales = bytearray(scale_length)
    biases = bytearray(bias_length)
    rows = capacity * n
    weight_row = layout.groups * WEIGHT_BYTES_PER_GROUP
    parameter_row = layout.groups * PARAMETER_BYTES_PER_GROUP
    for row in range(rows):
        weight_base = row * weight_row
        scale_base = row * parameter_row
        bias_base = row * parameter_row
        packed_base = row * layout.packed_row_bytes
        for chunk in range(layout.chunks):
            group = chunk * GROUPS_PER_CHUNK
            weight_start = weight_base + group * WEIGHT_BYTES_PER_GROUP
            scale_start = scale_base + group * PARAMETER_BYTES_PER_GROUP
            bias_start = bias_base + group * PARAMETER_BYTES_PER_GROUP
            chunk_start = packed_base + chunk * CHUNK_BYTES
            weight[weight_start : weight_start + WEIGHT_BYTES_PER_CHUNK] = packed[
                chunk_start : chunk_start + WEIGHT_BYTES_PER_CHUNK
            ]
            scales[scale_start : scale_start + PARAMETER_BYTES_PER_CHUNK] = packed[
                chunk_start + WEIGHT_BYTES_PER_CHUNK : chunk_start
                + WEIGHT_BYTES_PER_CHUNK
                + PARAMETER_BYTES_PER_CHUNK
            ]
            biases[bias_start : bias_start + PARAMETER_BYTES_PER_CHUNK] = packed[
                chunk_start
                + WEIGHT_BYTES_PER_CHUNK
                + PARAMETER_BYTES_PER_CHUNK : chunk_start + CHUNK_BYTES
            ]
    return bytes(weight), bytes(scales), bytes(biases)


def pack_projection_mlx(weight: Any, scales: Any, biases: Any) -> Any:
    _require_mlx()
    if len(weight.shape) != 3 or weight.dtype != mx.uint8:
        raise BenchmarkContractError("weight must be a rank-3 uint8 array")
    capacity, n, weight_row_bytes = map(int, weight.shape)
    if weight_row_bytes % WEIGHT_BYTES_PER_GROUP:
        raise BenchmarkContractError("weight row must contain whole Q4 groups")
    groups = weight_row_bytes // WEIGHT_BYTES_PER_GROUP
    k = groups * GROUP_SIZE
    layout = projection_layout(k)
    expected_parameters = (capacity, n, groups, PARAMETER_BYTES_PER_GROUP)
    for name, value in (("scales", scales), ("biases", biases)):
        if value.dtype != mx.uint8 or tuple(value.shape) != expected_parameters:
            raise BenchmarkContractError(
                f"{name} must have uint8 shape {expected_parameters}"
            )
    chunk_shape = (capacity, n, layout.chunks)
    packed = mx.concatenate(
        (
            weight.reshape((*chunk_shape, WEIGHT_BYTES_PER_CHUNK)),
            scales.reshape((*chunk_shape, PARAMETER_BYTES_PER_CHUNK)),
            biases.reshape((*chunk_shape, PARAMETER_BYTES_PER_CHUNK)),
        ),
        axis=-1,
    )
    return packed.reshape((capacity, n, layout.packed_row_bytes))


_KERNEL_HEADER = r"""
    using namespace metal;
    constant constexpr int GROUP_SIZE = 64;
    constant constexpr int VALUES_PER_THREAD = 16;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    inline uint fold_loaded_bits(const device ushort* words, ushort scale_bits,
                                 ushort bias_bits, uint state) {
      uint loaded = uint(words[0])
                  ^ (uint(words[1]) << 1)
                  ^ (uint(words[2]) << 2)
                  ^ (uint(words[3]) << 3)
                  ^ (uint(scale_bits) << 4)
                  ^ (uint(bias_bits) << 5);
      return state ^ loaded;
    }
"""


_KERNEL_PROLOGUE = r"""
    uint r_idx = threadgroup_position_in_grid.x;
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    int R = int(R_size);
    int K = int(K_size);
    int N = int(N_size);
    if (int(r_idx) >= R) return;
    int out_row = int(n_tile) * BN
                + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    long slot = long(slot_index[r_idx]);
    uint acc[RESULTS_PER_SIMDGROUP];
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      acc[row] = 2166136261u ^ uint(row) ^ simd_lid;
    }
"""


_SPLIT_BODY = r"""
    int weight_row_bytes = K / 2;
    int parameter_row_bytes = (K / GROUP_SIZE) * 2;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
      int weight_k = k / 2 + int(simd_lid) * 8;
      int parameter_k = (k / GROUP_SIZE + int(simd_lid) / 4) * 2;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          long weight_base = slot * long(N) * long(weight_row_bytes)
                           + long(n) * long(weight_row_bytes);
          long parameter_base = slot * long(N) * long(parameter_row_bytes)
                              + long(n) * long(parameter_row_bytes);
          const device ushort* words =
              (const device ushort*)(w + weight_base + weight_k);
          ushort scale_bits =
              *((const device ushort*)(scales + parameter_base + parameter_k));
          ushort bias_bits =
              *((const device ushort*)(biases + parameter_base + parameter_k));
          acc[row] = fold_loaded_bits(words, scale_bits, bias_bits, acc[row]);
        }
      }
    }
"""


_PACKED_BODY = r"""
    int packed_row_bytes = (K / GROUP_SIZE / 4) * 144;
    int lane_chunk = int(simd_lid) / 16;
    int lane_in_chunk = int(simd_lid) % 16;
    int group_in_chunk = lane_in_chunk / 4;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
      int block_base = (k / (GROUP_SIZE * 4)) * 144;
      int chunk_base = block_base + lane_chunk * 144;
      int weight_offset = chunk_base + lane_in_chunk * 8;
      int scale_offset = chunk_base + 128 + group_in_chunk * 2;
      int bias_offset = chunk_base + 136 + group_in_chunk * 2;
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          long packed_base = slot * long(N) * long(packed_row_bytes)
                           + long(n) * long(packed_row_bytes);
          const device ushort* words =
              (const device ushort*)(packed + packed_base + weight_offset);
          ushort scale_bits =
              *((const device ushort*)(packed + packed_base + scale_offset));
          ushort bias_bits =
              *((const device ushort*)(packed + packed_base + bias_offset));
          acc[row] = fold_loaded_bits(words, scale_bits, bias_bits, acc[row]);
        }
      }
    }
"""


_KERNEL_EPILOGUE = r"""
    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        uint reduced = simd_sum(acc[row]);
        if (simd_lid == 0) out[long(r_idx) * long(N) + long(n)] = reduced;
      }
    }
"""


def build_split_floor_kernel() -> Any:
    _require_mlx()
    return mx.fast.metal_kernel(
        name="hy3_split_stream_floor_issue31",
        input_names=[
            "w",
            "scales",
            "biases",
            "slot_index",
            "R_size",
            "K_size",
            "N_size",
        ],
        output_names=["out"],
        header=_KERNEL_HEADER,
        source=_KERNEL_PROLOGUE + _SPLIT_BODY + _KERNEL_EPILOGUE,
        ensure_row_contiguous=True,
    )


def build_packed_floor_kernel() -> Any:
    _require_mlx()
    return mx.fast.metal_kernel(
        name="hy3_packed_stream_floor_issue31",
        input_names=[
            "packed",
            "slot_index",
            "R_size",
            "K_size",
            "N_size",
        ],
        output_names=["out"],
        header=_KERNEL_HEADER,
        source=_KERNEL_PROLOGUE + _PACKED_BODY + _KERNEL_EPILOGUE,
        ensure_row_contiguous=True,
    )


def _dispatch(kernel: Any, inputs: list[Any], *, n: int) -> Any:
    return kernel(
        inputs=inputs,
        output_shapes=[(TOP_K, n)],
        output_dtypes=[mx.uint32],
        grid=(32 * TOP_K, 2 * math.ceil(n / 8), 1),
        threadgroup=(32, 2, 1),
    )[0]


def dispatch_split(
    kernel: Any,
    arrays: ProjectionArrays,
    slots: Any,
    *,
    n: int,
    k: int,
) -> Any:
    return _dispatch(
        kernel,
        [arrays.weight, arrays.scales, arrays.biases, slots, TOP_K, k, n],
        n=n,
    )


def dispatch_packed(
    kernel: Any,
    arrays: ProjectionArrays,
    slots: Any,
    *,
    n: int,
    k: int,
) -> Any:
    return _dispatch(kernel, [arrays.packed, slots, TOP_K, k, n], n=n)


def allocate_projection_arrays(
    *, capacity: int, n: int, k: int, seed: int
) -> ProjectionArrays:
    _require_mlx()
    if capacity < TOP_K + 1:
        raise BenchmarkContractError(
            f"capacity must be at least {TOP_K + 1} for scattered/repeated parity"
        )
    layout = projection_layout(k)
    mx.random.seed(seed)
    weight = mx.random.randint(
        0,
        256,
        shape=(capacity, n, layout.groups * WEIGHT_BYTES_PER_GROUP),
        dtype=mx.uint8,
    )
    scales = mx.random.randint(
        0,
        256,
        shape=(capacity, n, layout.groups, PARAMETER_BYTES_PER_GROUP),
        dtype=mx.uint8,
    )
    biases = mx.random.randint(
        0,
        256,
        shape=(capacity, n, layout.groups, PARAMETER_BYTES_PER_GROUP),
        dtype=mx.uint8,
    )
    mx.eval(weight, scales, biases)
    started = time.perf_counter()
    packed = pack_projection_mlx(weight, scales, biases)
    mx.eval(packed)
    packing_ms = (time.perf_counter() - started) * 1e3
    return ProjectionArrays(weight, scales, biases, packed, packing_ms)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise BenchmarkContractError("cannot summarize empty samples")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _t_critical_95(sample_count: int) -> float:
    if sample_count != PROCESS_REPEATS:
        raise BenchmarkContractError(
            f"issue #31 requires exactly {PROCESS_REPEATS} process repeats"
        )
    return 3.182  # Student t, two-sided 95%, df=3


def _mean_ci95(values: Sequence[float]) -> tuple[float, list[float]]:
    numbers = [float(value) for value in values]
    mean = statistics.mean(numbers)
    half = (
        _t_critical_95(len(numbers))
        * statistics.stdev(numbers)
        / math.sqrt(len(numbers))
    )
    return mean, [mean - half, mean + half]


def summarize_process_pairs(
    *,
    control_ms: Sequence[float],
    packed_ms: Sequence[float],
    control_p95_ms: Sequence[float] | None = None,
    packed_p95_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    control = [float(value) for value in control_ms]
    packed = [float(value) for value in packed_ms]
    if len(control) != len(packed) or len(control) < 2:
        raise BenchmarkContractError(
            "control and packed must have equal process-repeat counts of at least two"
        )
    speedups = [(left / right - 1) * 100 for left, right in zip(control, packed)]
    speedup, speedup_ci = _mean_ci95(speedups)
    result: dict[str, Any] = {
        "process_repeats": len(control),
        "control_ms": statistics.mean(control),
        "packed_ms": statistics.mean(packed),
        "control_ms_by_process": control,
        "packed_ms_by_process": packed,
        "speedup_percent": speedup,
        "speedup_percent_ci95": speedup_ci,
    }
    if control_p95_ms is not None and packed_p95_ms is not None:
        control_p95 = [float(value) for value in control_p95_ms]
        packed_p95 = [float(value) for value in packed_p95_ms]
        if len(control_p95) != len(control) or len(packed_p95) != len(control):
            raise BenchmarkContractError("p95 repeat counts must match medians")
        regressions = [
            (candidate / baseline - 1) * 100
            for baseline, candidate in zip(control_p95, packed_p95)
        ]
        result.update(
            {
                "control_p95_ms_by_process": control_p95,
                "packed_p95_ms_by_process": packed_p95,
                "packed_p95_regression_percent": statistics.mean(regressions),
                "packed_p95_regression_percent_worst_process": max(regressions),
            }
        )
    else:
        result["packed_p95_regression_percent"] = 0.0
        result["packed_p95_regression_percent_worst_process"] = 0.0
    return result


def weighted_projection_summary(
    gate: dict[str, Any], down: dict[str, Any]
) -> dict[str, Any]:
    gate_control = gate["control_ms_by_process"]
    gate_packed = gate["packed_ms_by_process"]
    down_control = down["control_ms_by_process"]
    down_packed = down["packed_ms_by_process"]
    if not (
        len(gate_control) == len(gate_packed) == len(down_control) == len(down_packed)
    ):
        raise BenchmarkContractError("projection process-repeat counts must match")
    control = [
        2 * gate_value + down_value
        for gate_value, down_value in zip(gate_control, down_control)
    ]
    packed = [
        2 * gate_value + down_value
        for gate_value, down_value in zip(gate_packed, down_packed)
    ]
    result = summarize_process_pairs(control_ms=control, packed_ms=packed)
    result["formula"] = "2 * gate_up_geometry + down_geometry"
    result["packed_p95_regression_percent"] = max(
        float(gate.get("packed_p95_regression_percent", 0.0)),
        float(down.get("packed_p95_regression_percent", 0.0)),
    )
    result["packed_p95_regression_percent_worst_process"] = max(
        float(gate.get("packed_p95_regression_percent_worst_process", 0.0)),
        float(down.get("packed_p95_regression_percent_worst_process", 0.0)),
    )
    return result


def classify_floor_gate(summary: dict[str, Any], *, minimum_percent: float) -> str:
    lower, upper = map(float, summary["speedup_percent_ci95"])
    p95_regression = float(
        summary.get(
            "packed_p95_regression_percent_worst_process",
            summary.get("packed_p95_regression_percent", 0.0),
        )
    )
    if p95_regression > 2.0:
        return "no-go"
    if lower >= minimum_percent:
        return "go-exact-qmv"
    if upper <= minimum_percent:
        return "no-go"
    return "inconclusive"


def _precompute_indices(
    *, capacity: int, per_eval: int, iterations: int, seed: int
) -> list[list[Any]]:
    rng = random.Random(seed)
    result = [
        [
            mx.array(rng.sample(range(capacity), TOP_K), dtype=mx.int32)
            for _ in range(per_eval)
        ]
        for _ in range(iterations)
    ]
    mx.eval(*(item for batch in result for item in batch))
    return result


def _parity_preflight(
    *,
    arrays: ProjectionArrays,
    split_kernel: Any,
    packed_kernel: Any,
    n: int,
    k: int,
    capacity: int,
) -> dict[str, Any]:
    slots = mx.array(
        [capacity - 1, 1, capacity - 1, 0, capacity - 2, 2, 3, 4],
        dtype=mx.int32,
    )
    split = dispatch_split(split_kernel, arrays, slots, n=n, k=k)
    packed = dispatch_packed(packed_kernel, arrays, slots, n=n, k=k)
    mx.eval(split, packed)
    equal = bool(mx.array_equal(split, packed))
    return {
        "equal": equal,
        "slots": [int(value) for value in slots.tolist()],
        "output_values": TOP_K * n,
    }


def _measure_projection(
    *,
    arrays: ProjectionArrays,
    split_kernel: Any,
    packed_kernel: Any,
    n: int,
    k: int,
    capacity: int,
    per_eval: int,
    warmup: int,
    iters: int,
    repeat_index: int,
) -> dict[str, Any]:
    batches = _precompute_indices(
        capacity=capacity,
        per_eval=per_eval,
        iterations=warmup + iters,
        seed=91_000 + repeat_index * 101 + n,
    )
    samples: dict[str, list[float]] = {arm: [] for arm in _ARMS}
    for iteration, slots_batch in enumerate(batches):
        graphs = {
            "split": [
                dispatch_split(split_kernel, arrays, slots, n=n, k=k)
                for slots in slots_batch
            ],
            "packed": [
                dispatch_packed(packed_kernel, arrays, slots, n=n, k=k)
                for slots in slots_batch
            ],
        }
        order = _ARMS if (iteration + repeat_index) % 2 == 0 else _ARMS[::-1]
        for arm in order:
            started = time.perf_counter()
            mx.eval(*graphs[arm])
            elapsed_ms = (time.perf_counter() - started) * 1e3 / per_eval
            if iteration >= warmup:
                samples[arm].append(elapsed_ms)
    logical_bytes = projection_bytes(capacity=TOP_K, n=n, k=k)
    arms: dict[str, Any] = {}
    for arm in _ARMS:
        median_ms = statistics.median(samples[arm])
        arms[arm] = {
            "samples_ms": samples[arm],
            "median_ms": median_ms,
            "p95_ms": _percentile(samples[arm], 0.95),
            "logical_gb_per_second": logical_bytes / (median_ms / 1e3) / 1e9,
        }
    return {
        "n": n,
        "k": k,
        "logical_bytes": logical_bytes,
        "arms": arms,
    }


def _git_output(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if completed.returncode:
        return None
    return completed.stdout.strip()


def _worker_result(args: argparse.Namespace) -> dict[str, Any]:
    refuse_if_busy(allow=args.allow_busy)
    split_kernel = build_split_floor_kernel()
    packed_kernel = build_packed_floor_kernel()
    projections = {
        "gate_up": (GATE_N, GATE_K),
        "down": (DOWN_N, DOWN_K),
    }
    arrays = {
        name: allocate_projection_arrays(
            capacity=args.capacity,
            n=n,
            k=k,
            seed=31_000 + args.repeat_index * 17 + n,
        )
        for name, (n, k) in projections.items()
    }
    parity = {
        name: _parity_preflight(
            arrays=arrays[name],
            split_kernel=split_kernel,
            packed_kernel=packed_kernel,
            n=n,
            k=k,
            capacity=args.capacity,
        )
        for name, (n, k) in projections.items()
    }
    if not all(item["equal"] for item in parity.values()):
        raise BenchmarkContractError(
            "split and packed checksums differ; timing withheld"
        )

    order = list(projections)
    if args.repeat_index % 2:
        order.reverse()
    measurements: dict[str, Any] = {}
    for name in order:
        n, k = projections[name]
        measurements[name] = _measure_projection(
            arrays=arrays[name],
            split_kernel=split_kernel,
            packed_kernel=packed_kernel,
            n=n,
            k=k,
            capacity=args.capacity,
            per_eval=args.per_eval,
            warmup=args.warmup,
            iters=args.iters,
            repeat_index=args.repeat_index,
        )
    return {
        "repeat_index": args.repeat_index,
        "projection_order": order,
        "packing_ms": {name: value.packing_ms for name, value in arrays.items()},
        "packing_ms_per_expert": {
            name: value.packing_ms / args.capacity for name, value in arrays.items()
        },
        "parity": parity,
        "measurements": measurements,
    }


def _aggregate(
    args: argparse.Namespace, process_results: list[dict[str, Any]]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for projection in ("gate_up", "down"):
        control = [
            result["measurements"][projection]["arms"]["split"]["median_ms"]
            for result in process_results
        ]
        packed = [
            result["measurements"][projection]["arms"]["packed"]["median_ms"]
            for result in process_results
        ]
        control_p95 = [
            result["measurements"][projection]["arms"]["split"]["p95_ms"]
            for result in process_results
        ]
        packed_p95 = [
            result["measurements"][projection]["arms"]["packed"]["p95_ms"]
            for result in process_results
        ]
        summaries[projection] = summarize_process_pairs(
            control_ms=control,
            packed_ms=packed,
            control_p95_ms=control_p95,
            packed_p95_ms=packed_p95,
        )
    weighted = weighted_projection_summary(summaries["gate_up"], summaries["down"])
    decision = classify_floor_gate(weighted, minimum_percent=args.minimum_percent)
    return {
        "schema": "mtplx.hy3-packed-stream-floor.v1",
        "issue": 31,
        "decision": decision,
        "claim_boundary": (
            "A go result authorizes only an exact packed-QMV microbenchmark; "
            "it does not authorize a codec, sidecar, or runtime integration."
        ),
        "config": {
            "capacity": args.capacity,
            "top_k": TOP_K,
            "group_size": GROUP_SIZE,
            "chunk_bytes": CHUNK_BYTES,
            "per_eval": args.per_eval,
            "warmup": args.warmup,
            "iters": args.iters,
            "process_repeats": args.repeats,
            "minimum_percent": args.minimum_percent,
            "p95_regression_limit_percent": 2.0,
            "flush_subtracted": False,
        },
        "layout": {
            "chunk": "128 weight bytes + 8 scale bytes + 8 bias bytes",
            "gate_up_row_bytes": projection_layout(GATE_K).packed_row_bytes,
            "down_row_bytes": projection_layout(DOWN_K).packed_row_bytes,
            "bytes_added": 0,
        },
        "projections": summaries,
        "weighted_three_projection": weighted,
        "process_results": process_results,
        "provenance": {
            "command": shlex.join(sys.argv),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_status": _git_output("status", "--porcelain"),
            "harness_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            "python": sys.version,
            "mlx": importlib.metadata.version("mlx"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def refuse_if_busy(*, allow: bool) -> None:
    if allow:
        return
    output = subprocess.run(
        ["ps", "-axo", "pid=,rss=,comm="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in output.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            rss_kib = int(parts[1])
        except ValueError:
            continue
        if rss_kib > 8 * 1024 * 1024 and "python" in parts[2].lower():
            raise BenchmarkContractError(
                f"pid {parts[0]} holds {rss_kib / 1_048_576:.1f} GiB; "
                "the machine is not quiet"
            )


@contextmanager
def exclusive_benchmark_lock(path: Path = _LOCK_PATH) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BenchmarkContractError(
                f"another hardware benchmark holds {path}"
            ) from exc
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _worker_command(
    args: argparse.Namespace, *, repeat_index: int, output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repeat-index",
        str(repeat_index),
        "--capacity",
        str(args.capacity),
        "--per-eval",
        str(args.per_eval),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--repeats",
        str(args.repeats),
        "--minimum-percent",
        str(args.minimum_percent),
        "--output",
        str(output),
    ]
    if args.allow_busy:
        command.append("--allow-busy")
    return command


def _run_parent(args: argparse.Namespace) -> dict[str, Any]:
    refuse_if_busy(allow=args.allow_busy)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mtplx-issue31-packed-") as temp:
        temp_path = Path(temp)
        for repeat_index in range(args.repeats):
            output = temp_path / f"repeat-{repeat_index}.json"
            completed = subprocess.run(
                _worker_command(args, repeat_index=repeat_index, output=output),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                raise BenchmarkContractError(
                    f"worker {repeat_index} failed:\n{completed.stdout}{completed.stderr}"
                )
            results.append(json.loads(output.read_text()))
    return _aggregate(args, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=102)
    parser.add_argument("--per-eval", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=PROCESS_REPEATS)
    parser.add_argument("--minimum-percent", type=float, default=MINIMUM_PERCENT)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.capacity < 102:
        raise BenchmarkContractError(
            "capacity must be at least 102 for a production-like DRAM gate"
        )
    for name in ("per_eval", "warmup", "iters", "repeats"):
        _positive_int(name, getattr(args, name))
    if args.repeats != PROCESS_REPEATS:
        raise BenchmarkContractError(
            f"issue #31 requires exactly {PROCESS_REPEATS} process repeats"
        )
    if args.minimum_percent != MINIMUM_PERCENT:
        raise BenchmarkContractError(
            f"issue #31 requires exactly a {MINIMUM_PERCENT:.1f}% promotion gate"
        )
    if args.repeat_index < 0:
        raise BenchmarkContractError("repeat index cannot be negative")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.worker:
        result = _worker_result(args)
    else:
        with exclusive_benchmark_lock():
            result = _run_parent(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not args.worker:
        weighted = result["weighted_three_projection"]
        print(f"decision: {result['decision']}")
        print(
            "weighted packed speedup: "
            f"{weighted['speedup_percent']:.3f}% "
            f"(95% CI {weighted['speedup_percent_ci95'][0]:.3f}% to "
            f"{weighted['speedup_percent_ci95'][1]:.3f}%)"
        )
        print(f"result: {args.output}")


if __name__ == "__main__":
    main()
