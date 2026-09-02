"""Exact device top-K selection for the stock native-MTP K20 sites.

What this is
------------
A shape-parameterised port of the parked PR391 selector
(``experiments/pr391-target-lmhead-top20``, ``mtplx/kernels/pr391_target_topk.py``,
receipts in that branch's ``docs/perf/pr391-target-topk-radix-result.md``:
0.927 ms vs 1.559 ms isolated on ``[4, 248320]``).  The original is frozen at
``ROW_COUNT=4`` / ``VOCAB_SIZE=248320``; the stock lane also needs ``[1, V]``
for each draft depth and, when the FRSpec ranked table is handed in, a
``[1, 65536]`` compact row whose tie keys and outputs are the *real* vocabulary
ids.  Everything else -- the two-stage tile/radix schedule, the composite key,
the signed-zero canonicalisation, the NaN ordering -- is the parked kernel's,
unchanged.

Why it was neutral there and may not be here
--------------------------------------------
On the PR391 lane the K20 fed a device decision kernel that ran concurrently
with the lookahead D3 and the 3-append MTP replay, so the selector sat behind
other GPU work and its isolated 40.5% win did not reach the stream (-0.26%
end to end).  On the stock lane every K20 is followed by a hard
``np.asarray`` (``mtplx/fast_sampling.py`` ``_device_serial_support_arrays``),
so nothing hides it.  That is a claim about the *lane*, not about this file;
``scripts/fable/micro_k20_select.py`` is the falsifier.

The ordering contract (identical to the stock selector's)
---------------------------------------------------------
Descending float32 value, then ASCENDING real token id.  ``-0.0`` and ``+0.0``
are one value.  Non-NaN outranks NaN; NaN ties break on id.  That is exactly
``mtplx/fast_sampling.py:_deterministic_mlx_top_k_support`` composed with
``_order_bounded_mlx_top_k_support``, and it is the same rule
``_device_serial_support_arrays`` reaches on its ``np.lexsort((ids, -values))``
hot path.  :func:`reference_top_k` is the kernel-free NumPy oracle for it.

NO execution happens at import.  Kernels are built lazily, per shape.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


TOP_K = 20
TILE_SIZE = 256
MERGE_WIDTH = 256
RADIX_PASSES = 8


def tile_count(vocab_size: int, tile_size: int = TILE_SIZE) -> int:
    """Tiles needed to cover ``vocab_size`` (the tail tile is masked)."""

    return (int(vocab_size) + int(tile_size) - 1) // int(tile_size)


_ORDERING_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline bool fdk_value_before(
    float candidate_value,
    uint candidate_index,
    bool candidate_valid,
    float other_value,
    uint other_index,
    bool other_valid) {
    if (candidate_valid != other_valid) {
        return candidate_valid;
    }
    if (!candidate_valid) {
        return candidate_index < other_index;
    }
    const bool candidate_nan = metal::isnan(candidate_value);
    const bool other_nan = metal::isnan(other_value);
    if (candidate_nan || other_nan) {
        if (candidate_nan != other_nan) {
            return !candidate_nan;
        }
        return candidate_index < other_index;
    }
    if (candidate_value > other_value) {
        return true;
    }
    if (other_value > candidate_value) {
        return false;
    }
    return candidate_index < other_index;
}

inline uint fdk_float_order_key(float value) {
    // The stock deterministic tie repair treats signed zero as one value.
    if (value == 0.0f) {
        value = 0.0f;
    }
    if (metal::isnan(value)) {
        return 0u;
    }
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong fdk_composite_key(float value, uint token_id) {
    // Larger values win; exact value ties prefer the lower real token id.
    return (ulong(fdk_float_order_key(value)) << 32) |
           ulong(0xffffffffu - token_id);
}
"""


_TILE_SOURCE = r"""
        const uint tile = threadgroup_position_in_grid.x;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_position_in_threadgroup.x;
        const uint slot = tile * TILE_SIZE + lane;
        const bool valid = row < ROW_COUNT && slot < VOCAB_SIZE;

        threadgroup float exchange_values[TILE_SIZE];
        threadgroup uint exchange_indices[TILE_SIZE];
        threadgroup uchar exchange_valid[TILE_SIZE];

        float my_value = valid
            ? rows[(size_t)row * VOCAB_SIZE + slot]
            : -INFINITY;
        uint my_index = valid ? TOKEN_ID_EXPR : 0xffffffffu;
        bool my_valid = valid;

        for (uint sequence = 2; sequence <= TILE_SIZE; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_values[lane] = my_value;
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const float other_value = exchange_values[partner];
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const float a_value = is_lower ? my_value : other_value;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const float b_value = is_lower ? other_value : my_value;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;
                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = fdk_value_before(
                    b_value, b_index, b_valid, a_value, a_index, a_valid);
                const bool a_before_b = fdk_value_before(
                    a_value, a_index, a_valid, b_value, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_value = is_lower ? b_value : a_value;
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }

        if (lane < TOP_K) {
            const size_t out =
                ((size_t)row * TILE_COUNT + tile) * TOP_K + lane;
            tile_values[out] = my_value;
            tile_indices[out] = my_index;
        }
"""


_MERGE_SOURCE = r"""
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_position_in_threadgroup.x;
        const size_t candidate_base = (size_t)row * CANDIDATE_COUNT;

        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;
        threadgroup float exchange_values[MERGE_WIDTH];
        threadgroup uint exchange_indices[MERGE_WIDTH];
        threadgroup uchar exchange_valid[MERGE_WIDTH];

        if (lane == 0) {
            radix_prefix = 0ul;
            radix_rank = TOP_K - 1;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pass = 0; pass < 8; ++pass) {
            atomic_store_explicit(
                &radix_histogram[lane], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint shift = 56u - pass * 8u;
            const ulong prefix = radix_prefix;
            for (uint candidate = lane;
                 candidate < CANDIDATE_COUNT;
                 candidate += MERGE_WIDTH) {
                const size_t at = candidate_base + candidate;
                const ulong key = fdk_composite_key(
                    tile_values[at], tile_indices[at]);
                const bool prefix_matches = pass == 0
                    || (key >> (shift + 8u)) == prefix;
                if (prefix_matches) {
                    const uint digit = uint((key >> shift) & 0xfful);
                    atomic_fetch_add_explicit(
                        &radix_histogram[digit], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                uint rank = radix_rank;
                uint chosen = 0;
                for (int digit = 255; digit >= 0; --digit) {
                    const uint count = atomic_load_explicit(
                        &radix_histogram[uint(digit)], memory_order_relaxed);
                    if (rank < count) {
                        chosen = uint(digit);
                        break;
                    }
                    rank -= count;
                }
                radix_prefix = (radix_prefix << 8) | ulong(chosen);
                radix_rank = rank;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            threshold_key = radix_prefix;
            atomic_store_explicit(
                &selected_count, 0u, memory_order_relaxed);
        }
        exchange_values[lane] = -INFINITY;
        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ulong threshold = threshold_key;
        for (uint candidate = lane;
             candidate < CANDIDATE_COUNT;
             candidate += MERGE_WIDTH) {
            const size_t at = candidate_base + candidate;
            const float value = tile_values[at];
            const uint token_id = tile_indices[at];
            if (fdk_composite_key(value, token_id) >= threshold) {
                const uint slot = atomic_fetch_add_explicit(
                    &selected_count, 1u, memory_order_relaxed);
                if (slot < TOP_K) {
                    exchange_values[slot] = value;
                    exchange_indices[slot] = token_id;
                    exchange_valid[slot] = 1;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float my_value = exchange_values[lane];
        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0;
        for (uint sequence = 2; sequence <= MERGE_WIDTH; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_values[lane] = my_value;
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const float other_value = exchange_values[partner];
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const float a_value = is_lower ? my_value : other_value;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const float b_value = is_lower ? other_value : my_value;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;
                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = fdk_value_before(
                    b_value, b_index, b_valid, a_value, a_index, a_valid);
                const bool a_before_b = fdk_value_before(
                    a_value, a_index, a_valid, b_value, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_value = is_lower ? b_value : a_value;
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }

        if (lane < TOP_K) {
            const size_t out = (size_t)row * TOP_K + lane;
            top_values[out] = my_value;
            top_indices[out] = my_index;
        }
"""


def _shape_header(
    *,
    rows: int,
    vocab_size: int,
    top_k: int,
    tiles: int,
) -> str:
    return _ORDERING_HEADER + (
        f"\nconstant constexpr uint TOP_K = {int(top_k)};\n"
        f"constant constexpr uint TILE_SIZE = {TILE_SIZE};\n"
        f"constant constexpr uint TILE_COUNT = {int(tiles)};\n"
        f"constant constexpr uint VOCAB_SIZE = {int(vocab_size)};\n"
        f"constant constexpr uint ROW_COUNT = {int(rows)};\n"
        f"constant constexpr uint CANDIDATE_COUNT = {int(tiles) * int(top_k)};\n"
        f"constant constexpr uint MERGE_WIDTH = {MERGE_WIDTH};\n"
        "constant constexpr uint RADIX_BINS = 256;\n"
    )


@lru_cache(maxsize=None)
def _tile_kernel(rows: int, vocab_size: int, top_k: int, mapped: bool):
    import mlx.core as mx

    tiles = tile_count(vocab_size)
    source = _TILE_SOURCE.replace(
        "TOKEN_ID_EXPR",
        "uint(id_map[slot < VOCAB_SIZE ? slot : 0u])" if mapped else "slot",
    )
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_fable_device_k20_tile_f32_r{rows}_v{vocab_size}"
            f"_k{top_k}_t{TILE_SIZE}{'_mapped' if mapped else ''}"
        ),
        input_names=["rows", "id_map"] if mapped else ["rows"],
        output_names=["tile_values", "tile_indices"],
        header=_shape_header(
            rows=rows, vocab_size=vocab_size, top_k=top_k, tiles=tiles
        ),
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _merge_kernel(rows: int, vocab_size: int, top_k: int):
    import mlx.core as mx

    tiles = tile_count(vocab_size)
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_fable_device_k20_merge_f32_r{rows}"
            f"_c{tiles * top_k}_k{top_k}"
        ),
        input_names=["tile_values", "tile_indices"],
        output_names=["top_values", "top_indices"],
        header=_shape_header(
            rows=rows, vocab_size=vocab_size, top_k=top_k, tiles=tiles
        ),
        source=_MERGE_SOURCE,
        ensure_row_contiguous=True,
    )


def device_top_k(
    rows: Any,
    *,
    top_k: int = TOP_K,
    id_map: Any = None,
) -> tuple[Any, Any]:
    """Exact ``(ids uint32 [R, k], values float32 [R, k])``, value desc / id asc.

    ``rows`` is ``[R, V]`` float32.  ``id_map`` (optional, ``int32 [V]``) maps a
    compact column to the real vocabulary id used for BOTH the tie key and the
    emitted id -- that is what makes a selection over the FRSpec 65,536-wide
    head identical to one over its 248,320-wide scatter.
    """

    import mlx.core as mx

    if rows.ndim != 2:
        raise ValueError("device_top_k expects a 2-D [rows, vocab] block")
    if rows.dtype != mx.float32:
        raise ValueError("device_top_k expects float32 rows")
    row_count = int(rows.shape[0])
    vocab_size = int(rows.shape[1])
    k = int(top_k)
    if not 0 < k <= TILE_SIZE:
        raise ValueError(f"device_top_k supports 1 <= top_k <= {TILE_SIZE}")
    if vocab_size < k:
        raise ValueError("device_top_k requires vocab_size >= top_k")
    tiles = tile_count(vocab_size)
    mapped = id_map is not None
    if mapped and (id_map.ndim != 1 or int(id_map.shape[0]) != vocab_size):
        raise ValueError("device_top_k id_map must be [vocab_size]")

    tile_values, tile_indices = _tile_kernel(row_count, vocab_size, k, mapped)(
        inputs=[rows, id_map] if mapped else [rows],
        grid=(tiles * TILE_SIZE, row_count, 1),
        threadgroup=(TILE_SIZE, 1, 1),
        output_shapes=[(row_count, tiles, k), (row_count, tiles, k)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    top_values, top_indices = _merge_kernel(row_count, vocab_size, k)(
        inputs=[tile_values, tile_indices],
        grid=(MERGE_WIDTH, row_count, 1),
        threadgroup=(MERGE_WIDTH, 1, 1),
        output_shapes=[(row_count, k), (row_count, k)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    return top_indices, top_values


#: The kernel-free oracle lives in ``mtplx.fable_device_k20`` so the CPU tests
#: can import it without pulling ``mtplx.kernels`` (and therefore MLX) in.
from ..fable_device_k20 import reference_top_k  # noqa: E402  (re-export)


__all__ = [
    "TOP_K",
    "TILE_SIZE",
    "MERGE_WIDTH",
    "device_top_k",
    "reference_top_k",
    "tile_count",
]
