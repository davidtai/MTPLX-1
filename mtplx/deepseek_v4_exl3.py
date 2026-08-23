"""Exact EXL3-Trellis operators for the pinned MiaAI DeepSeek V4 artifact.

This module is deliberately format-specific.  The target archive stores each
16x16 weight tile as 48 signed int16 words: a three-bit Trellis stream in the
tensor-core order used by ExLlamaV3 revision
``787d1582267117d6ee83c90014f03b525b14754f`` with the MCG codebook.  It is not
an MLX affine-quantized matrix and must never be passed through ``mx.dequantize``
or requantized during installation.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn


EXL3_BITS = 3
EXL3_TILE = 16
EXL3_PACKED_WORDS = 48
EXL3_HADAMARD = 128
EXL3_MCG_MULTIPLIER = 0xCBAC1FED

_EXPERT_KEY = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w1|w2|w3)\.rank0\."
    r"(?P<field>trellis|suh|svh|mcg)$"
)
_PROJECTION_NAMES = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
_DSPARK_EXPERT_KEY = re.compile(
    r"^mtp\.(?P<stage>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w1|w2|w3)\.(?P<field>weight|scale)$"
)


def _tensor_core_permutation() -> tuple[int, ...]:
    """Return ExLlamaV3's encoded-index to row-major tile permutation."""

    permutation: list[int] = []
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        col0 = thread // 4
        col1 = col0 + 8
        permutation.extend(
            (
                row0 * 16 + col0,
                row1 * 16 + col0,
                row2 * 16 + col0,
                row3 * 16 + col0,
                row0 * 16 + col1,
                row1 * 16 + col1,
                row2 * 16 + col1,
                row3 * 16 + col1,
            )
        )
    return tuple(permutation)


EXL3_TENSOR_CORE_PERMUTATION = _tensor_core_permutation()
EXL3_TENSOR_CORE_INVERSE = tuple(
    sorted(range(256), key=EXL3_TENSOR_CORE_PERMUTATION.__getitem__)
)


def decode_mcg_trellis_tile(packed: Any):
    """Decode one authentic ``[48]`` MCG/K3 tile to row-major float16.

    This is the installation-time numeric oracle, transcribed from the pinned
    ExLlamaV3 ``unpack_trellis_kernel`` / ``decode_3inst<1>`` pair.  It is kept
    off the execution path; the Metal operator consumes the packed words
    directly.
    """

    import numpy as np

    source = np.asarray(packed)
    if source.shape != (EXL3_PACKED_WORDS,):
        raise ValueError(
            f"EXL3 K3 tile must have shape ({EXL3_PACKED_WORDS},), "
            f"got {source.shape}"
        )
    if source.dtype not in (np.dtype(np.int16), np.dtype(np.uint16)):
        raise TypeError(f"EXL3 packed tile must be int16/uint16, got {source.dtype}")

    words = np.ascontiguousarray(source).view(np.uint16).view(np.uint32)
    decoded_tc = np.empty(256, dtype=np.float16)
    word_count = EXL3_BITS * 256 // 32
    for offset in range(256):
        bit0 = offset * EXL3_BITS + EXL3_BITS - 16 + 256 * EXL3_BITS
        bit1 = bit0 + 16
        index0 = bit0 // 32
        index1 = (bit1 - 1) // 32
        shift = (index1 + 1) * 32 - bit1
        low = int(words[index0 % word_count])
        high = int(words[index1 % word_count])
        state = (((low << 32) | high) >> shift) & 0xFFFF

        product = (state * EXL3_MCG_MULTIPLIER) & 0xFFFFFFFF
        # PTX lop3(a, b, c, 0x6a) is c XOR (a AND b).
        half_pair_bits = 0x3B603B60 ^ (product & 0x8FFF8FFF)
        pair = np.array(
            [half_pair_bits & 0xFFFF, half_pair_bits >> 16], dtype=np.uint16
        ).view(np.float16)
        decoded_tc[offset] = np.float16(pair[0] + pair[1])

    row_major = np.empty(256, dtype=np.float16)
    row_major[list(EXL3_TENSOR_CORE_PERMUTATION)] = decoded_tc
    return row_major.reshape(EXL3_TILE, EXL3_TILE)


@lru_cache(maxsize=None)
def _mcg_qmv_kernel(
    size_k: int,
    size_n: int,
    experts: int = 1,
    topk: int = 0,
    routed_input: bool = False,
):
    if size_k % EXL3_HADAMARD or size_n % EXL3_HADAMARD:
        raise ValueError("EXL3 projection dimensions must be divisible by H128")
    inverse = ",".join(str(value) for value in EXL3_TENSOR_CORE_INVERSE)
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint NTILES_N = {size_n // 16};
        constant constexpr uint KBLOCKS = {size_k // 128};
        constant constexpr uint EXPERTS = {experts};
        constant constexpr uint TOPK = {topk};
        constant constexpr uint HAD = 128;
        constant constexpr uint TILE_WORDS = 48;
        constant constexpr uint BLOCK_TILES = 8;
        constant constexpr float HAD_SCALE = 0.088388347648f;
        constant ushort TC_INV[256] = {{ {inverse} }};

        inline half decode_mcg(
            threadgroup const ushort* packed,
            uint tensor_core_offset
        ) {{
            threadgroup const uint* words =
                reinterpret_cast<threadgroup const uint*>(packed);
            uint bit0 = tensor_core_offset * 3u + 755u;
            uint bit1 = bit0 + 16u;
            uint index0 = bit0 / 32u;
            uint index1 = (bit1 - 1u) / 32u;
            uint shift = (index1 + 1u) * 32u - bit1;
            uint low = words[index0 % 24u];
            uint high = words[index1 % 24u];
            uint state = ((high >> shift) | (low << (32u - shift))) & 0xffffu;
            uint product = state * 0xCBAC1FEDu;
            uint half_pair_bits = 0x3B603B60u ^ (product & 0x8FFF8FFFu);
            half2 pair = as_type<half2>(half_pair_bits);
            return pair.x + pair.y;
        }}
    """
    grouped_setup = (
        """
        uint task = threadgroup_position_in_grid.z;
        uint row = task / TOPK;
        uint expert = uint(expert_ids[task]);
        size_t x_row = task;
        """
        if routed_input
        else """
        uint task = threadgroup_position_in_grid.z;
        uint row = task / TOPK;
        uint expert = uint(expert_ids[task]);
        size_t x_row = row;
        """
    )
    if not topk:
        grouped_setup = """
        uint row = threadgroup_position_in_grid.z;
        uint expert = 0u;
        size_t x_row = row;
        """
    expert_trellis_offset = (
        "(size_t)expert * (SIZE_K / 16u) * NTILES_N * TILE_WORDS + "
        if topk
        else ""
    )
    expert_suh_offset = "(size_t)expert * SIZE_K + " if topk else ""
    expert_svh_offset = "(size_t)expert * SIZE_N + " if topk else ""
    load_hadamard = """
            half scaled = half(
                x[x_row * SIZE_K + k]
                * suh[__EXPERT_SUH_OFFSET__k]
            );
            had_values[lane] = float(scaled);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 1u; stride < HAD; stride <<= 1u) {
                float own = had_values[lane];
                float peer = had_values[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                had_values[lane] =
                    (lane & stride) ? (peer - own) : (own + peer);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            x_had[lane] = half(had_values[lane] * HAD_SCALE);
        """
    source = """
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        __GROUPED_SETUP__
        uint n = n_block * HAD + lane;

        threadgroup float had_values[HAD];
        threadgroup half x_had[HAD];
        threadgroup ushort packed_tiles[
            BLOCK_TILES * BLOCK_TILES * TILE_WORDS
        ];

        float accumulator = 0.0f;
        for (uint k_block = 0; k_block < KBLOCKS; ++k_block) {
            uint k = k_block * HAD + lane;
            __LOAD_HADAMARD__

            for (
                uint packed_index = lane;
                packed_index < BLOCK_TILES * BLOCK_TILES * TILE_WORDS;
                packed_index += HAD
            ) {
                uint tile_k = packed_index / (BLOCK_TILES * TILE_WORDS);
                uint remainder = packed_index % (BLOCK_TILES * TILE_WORDS);
                uint tile_n = remainder / TILE_WORDS;
                uint word = remainder % TILE_WORDS;
                size_t source_index =
                    __EXPERT_TRELLIS_OFFSET__
                    ((size_t)(k_block * BLOCK_TILES + tile_k) * NTILES_N
                     + n_block * BLOCK_TILES + tile_n) * TILE_WORDS + word;
                packed_tiles[packed_index] =
                    reinterpret_cast<const device ushort*>(trellis)[source_index];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint tile_n = lane / 16u;
            uint local_n = lane & 15u;
            for (uint local_k = 0; local_k < HAD; ++local_k) {
                uint tile_k = local_k / 16u;
                uint local_row = local_k & 15u;
                uint row_major = local_row * 16u + local_n;
                uint tensor_core = uint(TC_INV[row_major]);
                threadgroup const ushort* tile =
                    packed_tiles
                    + (tile_k * BLOCK_TILES + tile_n) * TILE_WORDS;
                half weight = decode_mcg(tile, tensor_core);
                accumulator += float(x_had[local_k]) * float(weight);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ExLlamaV3's half-output GEMV rounds before its output H128 epilogue.
        had_values[lane] = float(half(accumulator));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = had_values[lane];
            float peer = had_values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            had_values[lane] =
                (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        half rotated = half(had_values[lane] * HAD_SCALE);
        y[(size_t)threadgroup_position_in_grid.z * SIZE_N + n] =
            half(rotated * svh[__EXPERT_SVH_OFFSET__n]);
    """
    source = (
        source.replace("__GROUPED_SETUP__", grouped_setup)
        .replace("__LOAD_HADAMARD__", load_hadamard)
        .replace("__EXPERT_SUH_OFFSET__", expert_suh_offset)
        .replace("__EXPERT_TRELLIS_OFFSET__", expert_trellis_offset)
        .replace("__EXPERT_SVH_OFFSET__", expert_svh_offset)
    )
    input_names = ["x", "trellis", "suh", "svh"]
    if topk:
        input_names.append("expert_ids")
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_mcg_qmv_k{size_k}_n{size_n}"
            f"_e{experts}_t{topk}_r{int(routed_input)}_v3"
        ),
        input_names=input_names,
        output_names=["y"],
        header=header,
        source=source,
    )


def exl3_mcg_qmv(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
) -> mx.array:
    """Run one pinned-format MCG/K3 projection with fused sign/H128 stages."""

    if x.ndim != 2 or x.dtype != mx.float16:
        raise ValueError("EXL3 MCG QMV requires a two-dimensional float16 input")
    if trellis.ndim != 3 or trellis.dtype != mx.int16:
        raise ValueError("EXL3 MCG QMV requires an int16 [K/16,N/16,48] trellis")
    size_k = int(trellis.shape[0]) * 16
    size_n = int(trellis.shape[1]) * 16
    if int(trellis.shape[2]) != EXL3_PACKED_WORDS:
        raise ValueError("EXL3 MCG QMV requires exactly 48 packed words per tile")
    if int(x.shape[1]) != size_k:
        raise ValueError("EXL3 MCG QMV input width does not match its trellis")
    if tuple(suh.shape) != (size_k,) or suh.dtype != mx.float16:
        raise ValueError("EXL3 MCG QMV suh does not match its input width")
    if tuple(svh.shape) != (size_n,) or svh.dtype != mx.float16:
        raise ValueError("EXL3 MCG QMV svh does not match its output width")
    rows = int(x.shape[0])
    kernel = _mcg_qmv_kernel(size_k, size_n)
    (output,) = kernel(
        inputs=[
            mx.contiguous(x),
            mx.contiguous(trellis),
            mx.contiguous(suh),
            mx.contiguous(svh),
        ],
        grid=(128, size_n // 128, rows),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, size_n)],
        output_dtypes=[mx.float16],
    )
    return output


def exl3_mcg_grouped_qmv(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    expert_ids: mx.array,
) -> mx.array:
    """Project router-selected rows through one packed K216-style expert bank."""

    if trellis.ndim != 4 or trellis.dtype != mx.int16:
        raise ValueError(
            "grouped EXL3 QMV requires int16 [E,K/16,N/16,48] trellis"
        )
    experts = int(trellis.shape[0])
    size_k = int(trellis.shape[1]) * 16
    size_n = int(trellis.shape[2]) * 16
    if int(trellis.shape[3]) != EXL3_PACKED_WORDS:
        raise ValueError("grouped EXL3 QMV requires 48 packed words per tile")
    if tuple(suh.shape) != (experts, size_k) or suh.dtype != mx.float16:
        raise ValueError("grouped EXL3 QMV suh bank has the wrong geometry")
    if tuple(svh.shape) != (experts, size_n) or svh.dtype != mx.float16:
        raise ValueError("grouped EXL3 QMV svh bank has the wrong geometry")
    if expert_ids.ndim != 2 or expert_ids.dtype not in (mx.int32, mx.uint32):
        raise ValueError("grouped EXL3 QMV expert IDs must be a 2-D int32 array")
    rows, topk = (int(value) for value in expert_ids.shape)
    routed_input = x.ndim == 3
    if routed_input:
        if tuple(x.shape[:2]) != (rows, topk) or int(x.shape[2]) != size_k:
            raise ValueError("routed EXL3 QMV input does not match router geometry")
        x_rows = mx.contiguous(x.reshape(rows * topk, size_k))
    else:
        if x.ndim != 2 or tuple(x.shape) != (rows, size_k):
            raise ValueError("grouped EXL3 QMV input does not match router rows")
        x_rows = mx.contiguous(x)
    if x_rows.dtype != mx.float16:
        raise ValueError("grouped EXL3 QMV requires float16 activations")

    tasks = rows * topk
    kernel = _mcg_qmv_kernel(size_k, size_n, experts, topk, routed_input)
    (output,) = kernel(
        inputs=[
            x_rows,
            mx.contiguous(trellis),
            mx.contiguous(suh),
            mx.contiguous(svh),
            mx.contiguous(expert_ids.reshape(tasks)),
        ],
        grid=(128, size_n // 128, tasks),
        threadgroup=(128, 1, 1),
        output_shapes=[(tasks, size_n)],
        output_dtypes=[mx.float16],
    )
    return output.reshape(rows, topk, size_n)


@lru_cache(maxsize=None)
def _route_hadamard_kernel(
    size_k: int,
    experts: int,
    topk: int,
    routed_input: bool,
):
    source_row = "task" if routed_input else "task / TOPK"
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint TOPK = {topk};
        constant constexpr uint HAD = 128;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = f"""
        uint lane = thread_position_in_threadgroup.x;
        uint k_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        size_t source_row = {source_row};
        uint k = k_block * HAD + lane;
        threadgroup float values[HAD];
        half scaled = half(
            x[source_row * SIZE_K + k]
            * suh[(size_t)expert * SIZE_K + k]
        );
        values[lane] = float(scaled);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {{
            float own = values[lane];
            float peer = values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[lane] = (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        y[(size_t)task * SIZE_K + k] = half(values[lane] * HAD_SCALE);
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_route_h128_k{size_k}_e{experts}"
            f"_t{topk}_r{int(routed_input)}_v1"
        ),
        input_names=["x", "suh", "expert_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


def _route_hadamard(
    x: mx.array,
    suh: mx.array,
    expert_ids: mx.array,
) -> mx.array:
    rows, topk = (int(value) for value in expert_ids.shape)
    tasks = rows * topk
    size_k = int(suh.shape[1])
    routed_input = x.ndim == 3
    kernel = _route_hadamard_kernel(
        size_k,
        int(suh.shape[0]),
        topk,
        routed_input,
    )
    (output,) = kernel(
        inputs=[
            mx.contiguous(x.astype(mx.float16)),
            mx.contiguous(suh),
            mx.contiguous(expert_ids.reshape(tasks)),
        ],
        grid=(128, size_k // 128, tasks),
        threadgroup=(128, 1, 1),
        output_shapes=[(tasks, size_k)],
        output_dtypes=[mx.float16],
    )
    return output


@lru_cache(maxsize=None)
def _mma_route_pack_kernel(experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint EXPERTS = {experts};
        constant constexpr uint BM = 8;
    """
    source = """
        uint lane = thread_position_in_threadgroup.x;
        threadgroup atomic_uint total;
        if (lane == 0u) atomic_store_explicit(&total, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < EXPERTS) {
            uint count = uint(row_count[lane]);
            uint blocks = (count + BM - 1u) / BM;
            uint destination = atomic_fetch_add_explicit(
                &total, blocks, memory_order_relaxed
            );
            uint start = uint(row_start[lane]);
            for (uint block = 0u; block < blocks; ++block) {
                uint offset = block * BM;
                block_expert[destination + block] = lane;
                block_row[destination + block] = start + offset;
                block_size[destination + block] = min(BM, count - offset);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) packed_count[0] = atomic_load_explicit(
            &total, memory_order_relaxed
        );
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_exl3_mma_route_pack_e{experts}_v1",
        input_names=["row_start", "row_count"],
        output_names=["block_expert", "block_row", "block_size", "packed_count"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _trellis_route_pack_kernel(experts: int, topk: int, block_m: int):
    """Source-owned histogram/prefix/route pack with no generic sort."""
    header = f"""
        using namespace metal;
        constant constexpr uint EXPERTS = {experts}u;
        constant constexpr uint TOPK = {topk}u;
        constant constexpr uint BM = {block_m}u;
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        threadgroup atomic_uint counts[EXPERTS];
        threadgroup atomic_uint cursors[EXPERTS];
        threadgroup uint offsets[EXPERTS + 1u];
        threadgroup uint total_blocks;

        for (uint expert = tid; expert < EXPERTS; expert += 256u) {
            atomic_store_explicit(&counts[expert], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint task = tid; task < uint(n_tasks); task += 256u) {
            uint expert = uint(expert_ids[task]);
            atomic_fetch_add_explicit(
                &counts[expert], 1u, memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint offset = 0u;
            uint block = 0u;
            for (uint expert = 0u; expert < EXPERTS; ++expert) {
                offsets[expert] = offset;
                uint count = atomic_load_explicit(
                    &counts[expert], memory_order_relaxed
                );
                atomic_store_explicit(
                    &cursors[expert], offset, memory_order_relaxed
                );
                for (uint first = 0u; first < count; first += BM) {
                    block_expert[block] = expert;
                    block_row[block] = offset + first;
                    block_size[block] = min(BM, count - first);
                    block += 1u;
                }
                offset += count;
            }
            offsets[EXPERTS] = offset;
            total_blocks = block;
            packed_count[0] = block;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint task = tid; task < uint(n_tasks); task += 256u) {
            uint expert = uint(expert_ids[task]);
            uint position = atomic_fetch_add_explicit(
                &cursors[expert], 1u, memory_order_relaxed
            );
            packed_tasks[position] = task;
            inverse[task] = position;
            sorted_ids[position] = expert;
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_route_e{experts}_t{topk}"
            f"_bm{block_m}_v1"
        ),
        input_names=["expert_ids", "n_tasks"],
        output_names=[
            "packed_tasks",
            "inverse",
            "sorted_ids",
            "block_expert",
            "block_row",
            "block_size",
            "packed_count",
        ],
        header=header,
        source=source,
    )


def _pack_trellis_routes(
    expert_ids: mx.array,
    *,
    experts: int,
    topk: int,
    block_m: int,
    kernel,
):
    tasks = int(expert_ids.size)
    route_blocks = _trellis_route_block_capacity(tasks, experts, block_m)
    return kernel(
        inputs=[
            mx.contiguous(expert_ids.reshape(tasks).astype(mx.uint32)),
            tasks,
        ],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tasks,)] * 3 + [(route_blocks,)] * 3 + [(1,)],
        output_dtypes=[mx.uint32] * 7,
    )


def _trellis_route_block_capacity(tasks: int, experts: int, block_m: int) -> int:
    """Exact maximum populated blocks for compact, expert-grouped routes.

    Give one route to each active expert first: each creates one block.  Once
    all experts are active, every additional block requires another ``block_m``
    routes assigned to some expert.  This shape-only bound is proven before
    Metal execution and therefore needs neither ``packed_count`` readback nor
    a task-count launch padded with inactive threadgroups.
    """
    active_experts = min(int(tasks), int(experts))
    extra_blocks = max(int(tasks) - int(experts), 0) // int(block_m)
    return max(active_experts + extra_blocks, 1)


@lru_cache(maxsize=None)
def _packed_route_hadamard_kernel(size_k: int, experts: int, topk: int):
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k}u;
        constant constexpr uint TOPK = {topk}u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint k_block = threadgroup_position_in_grid.y;
        uint sorted_task = threadgroup_position_in_grid.z;
        uint original_task = uint(packed_tasks[sorted_task]);
        uint source_row = original_task / TOPK;
        uint expert = uint(sorted_ids[sorted_task]);
        uint k = k_block * HAD + lane;
        threadgroup float values[HAD];
        half scaled = half(
            float(x[size_t(source_row) * SIZE_K + k])
            * float(suh[size_t(expert) * SIZE_K + k])
        );
        values[lane] = float(scaled);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = values[lane];
            float peer = values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[lane] = (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        y[size_t(sorted_task) * SIZE_K + k] = half(
            values[lane] * HAD_SCALE
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_packed_h128_k{size_k}_e{experts}"
            f"_t{topk}_v1"
        ),
        input_names=["x", "suh", "packed_tasks", "sorted_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _mcg_trellis_mma_kernel(
    size_k: int,
    size_n: int,
    experts: int,
    block_m: int,
):
    if block_m not in (8, 64):
        raise ValueError("Mia Trellis block_m must be 8 or 64")
    inverse = ",".join(str(value) for value in EXL3_TENSOR_CORE_INVERSE)
    simdgroups = block_m // 8 * 2
    threads = simdgroups * 32
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k}u;
        constant constexpr uint SIZE_N = {size_n}u;
        constant constexpr uint NTILES_N = {size_n // 16}u;
        constant constexpr uint BM = {block_m}u;
        constant constexpr uint BN = 32u;
        constant constexpr uint BK = 32u;
        constant constexpr uint THREADS = {threads}u;
        constant constexpr uint TILE_WORDS = 48u;
        constant ushort TC_INV[256] = {{ {inverse} }};

        inline half decode_mcg_device(
            device const ushort* packed,
            uint tensor_core_offset
        ) {{
            device const uint* words = reinterpret_cast<device const uint*>(packed);
            uint bit0 = tensor_core_offset * 3u + 755u;
            uint bit1 = bit0 + 16u;
            uint index0 = bit0 / 32u;
            uint index1 = (bit1 - 1u) / 32u;
            uint shift = (index1 + 1u) * 32u - bit1;
            uint low = words[index0 % 24u];
            uint high = words[index1 % 24u];
            uint state = ((high >> shift) | (low << (32u - shift))) & 0xffffu;
            uint product = state * 0xCBAC1FEDu;
            uint bits = 0x3B603B60u ^ (product & 0x8FFF8FFFu);
            half2 pair = as_type<half2>(bits);
            return pair.x + pair.y;
        }}
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint packed_block = threadgroup_position_in_grid.z;
        if (packed_block >= packed_count[0]) return;
        uint n0 = threadgroup_position_in_grid.y * BN;
        uint expert = block_expert[packed_block];
        uint first_row = block_row[packed_block];
        uint active_rows = block_size[packed_block];
        uint sg_m = sg / 2u;
        uint sg_n = (sg & 1u) * 16u;

        threadgroup half A_tile[BM * BK];
        threadgroup half B_tile[BK * BN];
        threadgroup half C_tile[BM * BN];
        simdgroup_matrix<half, 8, 8> a, b_left, b_right;
        simdgroup_matrix<float, 8, 8> c_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_right =
            simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0u; k0 < SIZE_K; k0 += BK) {
            for (uint index = tid; index < BM * BK; index += THREADS) {
                uint row = index / BK;
                uint local_k = index - row * BK;
                A_tile[index] = row < active_rows
                    ? x[size_t(first_row + row) * SIZE_K + k0 + local_k]
                    : half(0.0h);
            }
            for (uint index = tid; index < BK * BN; index += THREADS) {
                uint local_k = index / BN;
                uint local_n = index - local_k * BN;
                uint k = k0 + local_k;
                uint n = n0 + local_n;
                uint tile_k = k / 16u;
                uint tile_n = n / 16u;
                uint row_major = (k & 15u) * 16u + (n & 15u);
                uint tensor_core = uint(TC_INV[row_major]);
                size_t tile_index =
                    ((size_t)expert * (SIZE_K / 16u) * NTILES_N
                     + size_t(tile_k) * NTILES_N + tile_n) * TILE_WORDS;
                B_tile[index] = decode_mcg_device(
                    reinterpret_cast<const device ushort*>(trellis) + tile_index,
                    tensor_core
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0u; ks < BK; ks += 8u) {
                simdgroup_load(a, A_tile + sg_m * 8u * BK + ks, BK);
                simdgroup_load(b_left, B_tile + ks * BN + sg_n, BN);
                simdgroup_load(b_right, B_tile + ks * BN + sg_n + 8u, BN);
                simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
                simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<half, 8, 8> out_left, out_right;
        out_left.thread_elements()[0] = half(c_left.thread_elements()[0]);
        out_left.thread_elements()[1] = half(c_left.thread_elements()[1]);
        out_right.thread_elements()[0] = half(c_right.thread_elements()[0]);
        out_right.thread_elements()[1] = half(c_right.thread_elements()[1]);
        simdgroup_store(out_left, C_tile + sg_m * 8u * BN + sg_n, BN);
        simdgroup_store(out_right, C_tile + sg_m * 8u * BN + sg_n + 8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < active_rows * BN; index += THREADS) {
            uint row = index / BN;
            uint local_n = index - row * BN;
            y[size_t(first_row + row) * SIZE_N + n0 + local_n] = C_tile[index];
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_mma_k{size_k}_n{size_n}"
            f"_e{experts}_bm{block_m}_v1"
        ),
        input_names=[
            "x",
            "trellis",
            "block_expert",
            "block_row",
            "block_size",
            "packed_count",
        ],
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _trellis_activation_down_hadamard_kernel(
    intermediate_size: int,
    experts: int,
    limit: float,
):
    limit_literal = format(float(limit), ".9g")
    if "." not in limit_literal and "e" not in limit_literal.lower():
        limit_literal += ".0"
    header = f"""
        using namespace metal;
        constant constexpr uint INTERMEDIATE = {intermediate_size}u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;
        constant constexpr float LIMIT = {limit_literal}f;
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(sorted_ids[task]);
        uint column = block * HAD + lane;
        threadgroup float gate_values[HAD];
        threadgroup float up_values[HAD];

        gate_values[lane] = float(gate_inner[size_t(task) * INTERMEDIATE + column]);
        up_values[lane] = float(up_inner[size_t(task) * INTERMEDIATE + column]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float gate_own = gate_values[lane];
            float gate_peer = gate_values[lane ^ stride];
            float up_own = up_values[lane];
            float up_peer = up_values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gate_values[lane] = (lane & stride)
                ? gate_peer - gate_own
                : gate_own + gate_peer;
            up_values[lane] = (lane & stride)
                ? up_peer - up_own
                : up_own + up_peer;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        half gate_half = half(
            gate_values[lane] * HAD_SCALE
            * float(gate_svh[size_t(expert) * INTERMEDIATE + column])
        );
        half up_half = half(
            up_values[lane] * HAD_SCALE
            * float(up_svh[size_t(expert) * INTERMEDIATE + column])
        );
        float gate = float(gate_half);
        float up = float(up_half);
        if (LIMIT > 0.0f) {
            gate = min(gate, LIMIT);
            up = clamp(up, -LIMIT, LIMIT);
        }
        half activated = half((gate / (1.0f + exp(-gate))) * up);
        gate_values[lane] = float(half(
            float(activated)
            * float(down_suh[size_t(expert) * INTERMEDIATE + column])
        ));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = gate_values[lane];
            float peer = gate_values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gate_values[lane] = (lane & stride) ? peer - own : own + peer;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        down_h[size_t(task) * INTERMEDIATE + column] = half(
            gate_values[lane] * HAD_SCALE
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_swiglu_down_h_i{intermediate_size}"
            f"_e{experts}_l{int(round(limit * 1000.0))}_v1"
        ),
        input_names=[
            "gate_inner",
            "up_inner",
            "gate_svh",
            "up_svh",
            "down_suh",
            "sorted_ids",
        ],
        output_names=["down_h"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _trellis_final_reduce_kernel(
    hidden_size: int,
    experts: int,
    topk: int,
):
    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {hidden_size}u;
        constant constexpr uint TOPK = {topk}u;
        constant constexpr uint HAD = 128u;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = r"""
        uint lane = thread_position_in_threadgroup.x;
        uint block = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.z;
        uint column = block * HAD + lane;
        threadgroup float values[HAD];
        float routed_sum = 0.0f;
        for (uint route = 0u; route < TOPK; ++route) {
            uint original_task = row * TOPK + route;
            uint sorted_task = uint(inverse[original_task]);
            uint expert = uint(expert_ids[original_task]);
            values[lane] = float(
                down_inner[size_t(sorted_task) * HIDDEN + column]
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1u; stride < HAD; stride <<= 1u) {
                float own = values[lane];
                float peer = values[lane ^ stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                values[lane] = (lane & stride) ? peer - own : own + peer;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            half projected = half(
                values[lane] * HAD_SCALE
                * float(down_svh[size_t(expert) * HIDDEN + column])
            );
            routed_sum += float(projected)
                * float(route_weights[original_task]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        output[size_t(row) * HIDDEN + column] = T(
            routed_sum + float(shared[size_t(row) * HIDDEN + column])
        );
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_dsv4_exl3_trellis_final_reduce_h{hidden_size}"
            f"_e{experts}_t{topk}_v1"
        ),
        input_names=[
            "down_inner",
            "down_svh",
            "inverse",
            "expert_ids",
            "route_weights",
            "shared",
        ],
        output_names=["output"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _mcg_grouped_mma_kernel(size_k: int, size_n: int, experts: int):
    inverse = ",".join(str(value) for value in EXL3_TENSOR_CORE_INVERSE)
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_K = {size_k};
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint NTILES_N = {size_n // 16};
        constant constexpr uint BM = 8;
        constant constexpr uint BN = 32;
        constant constexpr uint BK = 32;
        constant constexpr uint TILE_WORDS = 48;
        constant ushort TC_INV[256] = {{ {inverse} }};

        inline half decode_mcg_device(
            device const ushort* packed,
            uint tensor_core_offset
        ) {{
            device const uint* words = reinterpret_cast<device const uint*>(packed);
            uint bit0 = tensor_core_offset * 3u + 755u;
            uint bit1 = bit0 + 16u;
            uint index0 = bit0 / 32u;
            uint index1 = (bit1 - 1u) / 32u;
            uint shift = (index1 + 1u) * 32u - bit1;
            uint low = words[index0 % 24u];
            uint high = words[index1 % 24u];
            uint state = ((high >> shift) | (low << (32u - shift))) & 0xffffu;
            uint product = state * 0xCBAC1FEDu;
            uint bits = 0x3B603B60u ^ (product & 0x8FFF8FFFu);
            half2 pair = as_type<half2>(bits);
            return pair.x + pair.y;
        }}
    """
    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint sg = tid / 32u;
        uint packed_block = threadgroup_position_in_grid.z;
        if (packed_block >= packed_count[0]) return;
        uint n0 = threadgroup_position_in_grid.y * BN;
        uint expert = block_expert[packed_block];
        uint first_row = block_row[packed_block];
        uint active_rows = block_size[packed_block];

        threadgroup half A_tile[BM * BK];
        threadgroup half B_tile[BK * BN];
        threadgroup half C_tile[BM * BN];
        simdgroup_matrix<half, 8, 8> a, b_left, b_right;
        simdgroup_matrix<float, 8, 8> c_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_right =
            simdgroup_matrix<float, 8, 8>(0.0f);
        uint sg_n = sg * 16u;

        for (uint k0 = 0u; k0 < SIZE_K; k0 += BK) {
            for (uint index = tid; index < BM * BK; index += 64u) {
                uint row = index / BK;
                uint local_k = index % BK;
                A_tile[index] = row < active_rows
                    ? x[(size_t)(first_row + row) * SIZE_K + k0 + local_k]
                    : half(0.0h);
            }
            for (uint index = tid; index < BK * BN; index += 64u) {
                uint local_k = index / BN;
                uint local_n = index % BN;
                uint k = k0 + local_k;
                uint n = n0 + local_n;
                uint tile_k = k / 16u;
                uint tile_n = n / 16u;
                uint row_major = (k & 15u) * 16u + (n & 15u);
                uint tensor_core = uint(TC_INV[row_major]);
                size_t tile_index =
                    ((size_t)expert * (SIZE_K / 16u) * NTILES_N
                     + (size_t)tile_k * NTILES_N + tile_n) * TILE_WORDS;
                B_tile[index] = decode_mcg_device(
                    reinterpret_cast<const device ushort*>(trellis) + tile_index,
                    tensor_core
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0u; ks < BK; ks += 8u) {
                simdgroup_load(a, A_tile + ks, BK);
                simdgroup_load(b_left, B_tile + ks * BN + sg_n, BN);
                simdgroup_load(b_right, B_tile + ks * BN + sg_n + 8u, BN);
                simdgroup_multiply_accumulate(c_left, a, b_left, c_left);
                simdgroup_multiply_accumulate(c_right, a, b_right, c_right);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_matrix<half, 8, 8> out_left, out_right;
        out_left.thread_elements()[0] = half(c_left.thread_elements()[0]);
        out_left.thread_elements()[1] = half(c_left.thread_elements()[1]);
        out_right.thread_elements()[0] = half(c_right.thread_elements()[0]);
        out_right.thread_elements()[1] = half(c_right.thread_elements()[1]);
        simdgroup_store(out_left, C_tile + sg_n, BN);
        simdgroup_store(out_right, C_tile + sg_n + 8u, BN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < active_rows * BN; index += 64u) {
            uint row = index / BN;
            uint local_n = index % BN;
            y[(size_t)(first_row + row) * SIZE_N + n0 + local_n] = C_tile[index];
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_exl3_mcg_mma_k{size_k}_n{size_n}_e{experts}_v1",
        input_names=[
            "x",
            "trellis",
            "block_expert",
            "block_row",
            "block_size",
            "packed_count",
        ],
        output_names=["y"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _route_output_hadamard_kernel(size_n: int, experts: int):
    header = f"""
        using namespace metal;
        constant constexpr uint SIZE_N = {size_n};
        constant constexpr uint HAD = 128;
        constant constexpr float HAD_SCALE = 0.088388347648f;
    """
    source = """
        uint lane = thread_position_in_threadgroup.x;
        uint n_block = threadgroup_position_in_grid.y;
        uint task = threadgroup_position_in_grid.z;
        uint expert = uint(expert_ids[task]);
        uint n = n_block * HAD + lane;
        threadgroup float values[HAD];
        values[lane] = float(x[(size_t)task * SIZE_N + n]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1u; stride < HAD; stride <<= 1u) {
            float own = values[lane];
            float peer = values[lane ^ stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            values[lane] = (lane & stride) ? (peer - own) : (own + peer);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        half rotated = half(values[lane] * HAD_SCALE);
        y[(size_t)task * SIZE_N + n] = half(
            rotated * svh[(size_t)expert * SIZE_N + n]
        );
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_exl3_route_output_h128_n{size_n}_e{experts}_v1",
        input_names=["x", "svh", "expert_ids"],
        output_names=["y"],
        header=header,
        source=source,
    )


def _mma_route_arena(expert_ids: mx.array, experts: int):
    flat_ids = expert_ids.reshape(-1).astype(mx.uint32)
    order = mx.argsort(flat_ids)
    inverse = mx.argsort(order)
    sorted_ids = mx.contiguous(flat_ids[order])
    expert_range = mx.arange(experts, dtype=mx.uint32)
    starts = mx.searchsorted(sorted_ids, expert_range, side="left").astype(mx.int32)
    ends = mx.searchsorted(sorted_ids, expert_range, side="right").astype(mx.int32)
    tasks = int(flat_ids.shape[0])
    blocks = _mma_route_pack_kernel(experts)(
        inputs=[mx.contiguous(starts), mx.contiguous(ends - starts)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tasks,), (tasks,), (tasks,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
    )
    return order, inverse, sorted_ids, blocks


def _mma_project_sorted(
    transformed: mx.array,
    trellis: mx.array,
    svh: mx.array,
    sorted_ids: mx.array,
    route_blocks,
) -> mx.array:
    tasks, size_k = (int(value) for value in transformed.shape)
    experts = int(trellis.shape[0])
    size_n = int(trellis.shape[2]) * 16
    block_expert, block_row, block_size, packed_count = route_blocks
    (inner,) = _mcg_grouped_mma_kernel(size_k, size_n, experts)(
        inputs=[
            transformed,
            mx.contiguous(trellis),
            block_expert,
            block_row,
            block_size,
            packed_count,
        ],
        grid=(64, size_n // 32, tasks),
        threadgroup=(64, 1, 1),
        output_shapes=[(tasks, size_n)],
        output_dtypes=[mx.float16],
    )
    (output,) = _route_output_hadamard_kernel(size_n, experts)(
        inputs=[inner, mx.contiguous(svh), sorted_ids],
        grid=(128, size_n // 128, tasks),
        threadgroup=(128, 1, 1),
        output_shapes=[(tasks, size_n)],
        output_dtypes=[mx.float16],
    )
    return output


def exl3_mcg_grouped_mma(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    expert_ids: mx.array,
) -> mx.array:
    """Pinned EXL3 M-tiled path using Metal simdgroup matrix accumulation."""

    rows, topk = (int(value) for value in expert_ids.shape)
    experts = int(trellis.shape[0])
    order, inverse, sorted_ids, route_blocks = _mma_route_arena(
        expert_ids, experts
    )
    transformed = _route_hadamard(x, suh, expert_ids)[order]
    sorted_output = _mma_project_sorted(
        transformed,
        trellis,
        svh,
        sorted_ids,
        route_blocks,
    )
    return sorted_output[inverse].reshape(rows, topk, int(svh.shape[1]))


class EXL3LinearBank(nn.Module):
    """One construction-qualified bank of MCG/K3 expert projections."""

    def __init__(
        self,
        experts: int,
        input_dims: int,
        output_dims: int,
        topk: int,
        *,
        routed_input: bool,
    ) -> None:
        super().__init__()
        if input_dims % 128 or output_dims % 128:
            raise ValueError("EXL3 expert dimensions must be divisible by H128")
        self.experts = int(experts)
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.topk = int(topk)
        self.routed_input = bool(routed_input)
        self.trellis = mx.zeros(
            (self.experts, self.input_dims // 16, self.output_dims // 16, 48),
            dtype=mx.int16,
        )
        self.suh = mx.zeros((self.experts, self.input_dims), dtype=mx.float16)
        self.svh = mx.zeros((self.experts, self.output_dims), dtype=mx.float16)
        self._kernel = _mcg_qmv_kernel(
            self.input_dims,
            self.output_dims,
            self.experts,
            self.topk,
            self.routed_input,
        )
        self._had_kernel = _route_hadamard_kernel(
            self.input_dims,
            self.experts,
            self.topk,
            self.routed_input,
        )
        self._mma_kernel = _mcg_grouped_mma_kernel(
            self.input_dims,
            self.output_dims,
            self.experts,
        )
        self._output_had_kernel = _route_output_hadamard_kernel(
            self.output_dims,
            self.experts,
        )

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        rows = int(expert_ids.shape[0])
        tasks = rows * self.topk
        x_rows = (
            mx.contiguous(x.reshape(tasks, self.input_dims))
            if self.routed_input
            else mx.contiguous(x)
        )
        (output,) = self._kernel(
            inputs=[
                x_rows,
                self.trellis,
                self.suh,
                self.svh,
                mx.contiguous(expert_ids.reshape(tasks)),
            ],
            grid=(128, self.output_dims // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.output_dims)],
            output_dtypes=[mx.float16],
        )
        return output.reshape(rows, self.topk, self.output_dims)

    def transform_routes(self, x: mx.array, flat_ids: mx.array) -> mx.array:
        tasks = int(flat_ids.shape[0])
        (output,) = self._had_kernel(
            inputs=[mx.contiguous(x), self.suh, mx.contiguous(flat_ids)],
            grid=(128, self.input_dims // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.input_dims)],
            output_dtypes=[mx.float16],
        )
        return output

    def mma_sorted(self, x: mx.array, sorted_ids: mx.array, route_blocks) -> mx.array:
        tasks = int(x.shape[0])
        block_expert, block_row, block_size, packed_count = route_blocks
        (inner,) = self._mma_kernel(
            inputs=[
                mx.contiguous(x),
                self.trellis,
                block_expert,
                block_row,
                block_size,
                packed_count,
            ],
            grid=(64, self.output_dims // 32, tasks),
            threadgroup=(64, 1, 1),
            output_shapes=[(tasks, self.output_dims)],
            output_dtypes=[mx.float16],
        )
        (output,) = self._output_had_kernel(
            inputs=[inner, self.svh, sorted_ids],
            grid=(128, self.output_dims // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.output_dims)],
            output_dtypes=[mx.float16],
        )
        return output

class _InstalledTrellisPlan(NamedTuple):
    block_m: int
    route_pack: Any
    hidden_to_intermediate: Any
    intermediate_to_hidden: Any


class EXL3SwitchGLU(nn.Module):
    """DeepSeek routed SwiGLU over the exact Mia K216 EXL3 expert banks."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        experts: int,
        topk: int,
        *,
        limit: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.experts = int(experts)
        self.topk = int(topk)
        self.limit = float(limit or 0.0)
        self._route_pack = _mma_route_pack_kernel(self.experts)
        self.gate_proj = EXL3LinearBank(
            experts,
            hidden_size,
            intermediate_size,
            topk,
            routed_input=False,
        )
        self.up_proj = EXL3LinearBank(
            experts,
            hidden_size,
            intermediate_size,
            topk,
            routed_input=False,
        )
        self.down_proj = EXL3LinearBank(
            experts,
            intermediate_size,
            hidden_size,
            topk,
            routed_input=True,
        )
        self._trellis_installed = False
        self._trellis_plans = ()
        self._trellis_input_hadamard = None
        self._trellis_activation_down = None
        self._trellis_final_reduce = None

    def install_trellis_runtime(self, *, max_tokens: int) -> None:
        """Install the pinned decode/prefill plans before request execution."""
        if int(max_tokens) < 1:
            raise ValueError("EXL3 Trellis max_tokens must be positive")
        self._trellis_max_tokens = int(max_tokens)
        plans = []
        for block_m in (8, 64):
            plans.append(
                _InstalledTrellisPlan(
                    block_m=block_m,
                    route_pack=_trellis_route_pack_kernel(
                        self.experts, self.topk, block_m
                    ),
                    hidden_to_intermediate=_mcg_trellis_mma_kernel(
                        self.hidden_size,
                        self.gate_proj.output_dims,
                        self.experts,
                        block_m,
                    ),
                    intermediate_to_hidden=_mcg_trellis_mma_kernel(
                        self.down_proj.input_dims,
                        self.hidden_size,
                        self.experts,
                        block_m,
                    ),
                )
            )
        self._trellis_plans = tuple(plans)
        self._trellis_input_hadamard = _packed_route_hadamard_kernel(
            self.hidden_size, self.experts, self.topk
        )
        self._trellis_activation_down = _trellis_activation_down_hadamard_kernel(
            self.gate_proj.output_dims, self.experts, self.limit
        )
        self._trellis_final_reduce = _trellis_final_reduce_kernel(
            self.hidden_size, self.experts, self.topk
        )
        self._trellis_installed = True

    def _trellis_mma(
        self,
        bank: EXL3LinearBank,
        transformed: mx.array,
        route_blocks,
        *,
        block_m: int,
        kernel,
    ) -> mx.array:
        tasks = int(transformed.shape[0])
        block_expert, block_row, block_size, packed_count = route_blocks
        route_blocks_capacity = int(block_expert.shape[0])
        threads = int(block_m) * 8
        return kernel(
            inputs=[
                mx.contiguous(transformed),
                bank.trellis,
                block_expert,
                block_row,
                block_size,
                packed_count,
            ],
            grid=(threads, bank.output_dims // 32, route_blocks_capacity),
            threadgroup=(threads, 1, 1),
            output_shapes=[(tasks, bank.output_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def fused(
        self,
        x: mx.array,
        expert_ids: mx.array,
        route_weights: mx.array,
        shared: mx.array,
    ) -> mx.array:
        """Run the installed W4A16 Trellis MoE and final weighted reduction."""
        original_dtype = x.dtype
        rows = int(expert_ids.shape[0])
        tasks = rows * self.topk
        plan = self._trellis_plans[0 if rows <= 127 else 1]
        block_m = plan.block_m
        (
            packed_tasks,
            inverse,
            sorted_ids,
            block_expert,
            block_row,
            block_size,
            packed_count,
        ) = _pack_trellis_routes(
            expert_ids,
            experts=self.experts,
            topk=self.topk,
            block_m=block_m,
            kernel=plan.route_pack,
        )
        route_blocks = (block_expert, block_row, block_size, packed_count)
        x_half = mx.contiguous(x.astype(mx.float16))
        transform = self._trellis_input_hadamard
        gate_h = transform(
            inputs=[x_half, self.gate_proj.suh, packed_tasks, sorted_ids],
            grid=(128, self.hidden_size // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.hidden_size)],
            output_dtypes=[mx.float16],
        )[0]
        up_h = transform(
            inputs=[x_half, self.up_proj.suh, packed_tasks, sorted_ids],
            grid=(128, self.hidden_size // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, self.hidden_size)],
            output_dtypes=[mx.float16],
        )[0]
        gate_inner = self._trellis_mma(
            self.gate_proj,
            gate_h,
            route_blocks,
            block_m=block_m,
            kernel=plan.hidden_to_intermediate,
        )
        up_inner = self._trellis_mma(
            self.up_proj,
            up_h,
            route_blocks,
            block_m=block_m,
            kernel=plan.hidden_to_intermediate,
        )
        intermediate = self.gate_proj.output_dims
        down_h = self._trellis_activation_down(
            inputs=[
                gate_inner,
                up_inner,
                self.gate_proj.svh,
                self.up_proj.svh,
                self.down_proj.suh,
                sorted_ids,
            ],
            grid=(128, intermediate // 128, tasks),
            threadgroup=(128, 1, 1),
            output_shapes=[(tasks, intermediate)],
            output_dtypes=[mx.float16],
        )[0]
        down_inner = self._trellis_mma(
            self.down_proj,
            down_h,
            route_blocks,
            block_m=block_m,
            kernel=plan.intermediate_to_hidden,
        )
        return self._trellis_final_reduce(
            inputs=[
                down_inner,
                self.down_proj.svh,
                inverse,
                mx.contiguous(expert_ids.reshape(tasks).astype(mx.uint32)),
                mx.contiguous(route_weights.reshape(tasks)),
                mx.contiguous(shared),
            ],
            template=[("T", original_dtype)],
            grid=(128, self.hidden_size // 128, rows),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, self.hidden_size)],
            output_dtypes=[original_dtype],
        )[0]

    def direct_qmv(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        """Run the authentic direct-QMV MoE arithmetic unconditionally."""

        original_dtype = x.dtype
        x_half = x.astype(mx.float16)
        gate = self.gate_proj(x_half, expert_ids)
        up = self.up_proj(x_half, expert_ids)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        activated = (nn.silu(gate) * up).astype(mx.float16)
        return self.down_proj(activated, expert_ids).astype(original_dtype)

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        rows = int(expert_ids.shape[0])
        # Preserve the explicit stock/oracle compatibility route.  Production
        # binds ``direct_qmv`` and ``fused`` separately at construction.
        if rows <= 6:
            return self.direct_qmv(x, expert_ids)

        original_dtype = x.dtype
        x_half = x.astype(mx.float16)
        tasks = int(expert_ids.size)
        flat_ids = expert_ids.reshape(tasks).astype(mx.uint32)
        order = mx.argsort(flat_ids)
        inverse = mx.argsort(order)
        sorted_ids = mx.contiguous(flat_ids[order])
        expert_range = mx.arange(self.experts, dtype=mx.uint32)
        starts = mx.searchsorted(sorted_ids, expert_range, side="left").astype(
            mx.int32
        )
        ends = mx.searchsorted(sorted_ids, expert_range, side="right").astype(
            mx.int32
        )
        route_blocks = self._route_pack(
            inputs=[mx.contiguous(starts), mx.contiguous(ends - starts)],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(tasks,), (tasks,), (tasks,), (1,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
        )
        gate_h = self.gate_proj.transform_routes(x_half, flat_ids)[order]
        up_h = self.up_proj.transform_routes(x_half, flat_ids)[order]
        gate = self.gate_proj.mma_sorted(gate_h, sorted_ids, route_blocks)
        up = self.up_proj.mma_sorted(up_h, sorted_ids, route_blocks)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        activated = (nn.silu(gate) * up).astype(mx.float16)
        down_h = self.down_proj.transform_routes(activated, sorted_ids)
        down = self.down_proj.mma_sorted(down_h, sorted_ids, route_blocks)
        return down[inverse].reshape(
            int(expert_ids.shape[0]), self.topk, self.hidden_size
        ).astype(
            original_dtype
        )


def _map_mia_target_name(name: str) -> str:
    top_level = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "hc_head_fn": "model.hc_head.fn",
        "hc_head_base": "model.hc_head.base",
        "hc_head_scale": "model.hc_head.scale",
    }
    if name in top_level:
        return top_level[name]
    if not name.startswith("layers."):
        return name
    mapped = "model." + name
    replacements = (
        (".ffn.shared_experts.w1", ".ffn.shared_experts.gate_proj"),
        (".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"),
        (".ffn.shared_experts.w3", ".ffn.shared_experts.up_proj"),
        (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
    )
    for source, target in replacements:
        mapped = mapped.replace(source, target)
    return mapped


def _expand_mia_fp8_block_scales(
    scales: mx.array,
    output_dims: int,
    input_dims: int,
) -> mx.array:
    expected = ((output_dims + 127) // 128, (input_dims + 127) // 128)
    if tuple(scales.shape) != expected or scales.dtype != mx.uint8:
        raise ValueError(
            f"Mia FP8 block scales {tuple(scales.shape)}/{scales.dtype} "
            f"do not match {expected}/uint8"
        )
    expanded = mx.repeat(mx.repeat(scales, 128, axis=0), 4, axis=1)
    return mx.contiguous(expanded[:output_dims, : input_dims // 32])


def sanitize_mia_exl3_target_weights(
    weights: dict[str, mx.array],
    *,
    layers: int,
    experts: int,
) -> dict[str, mx.array]:
    """Map the exact Mia target storage onto the installed MLX module tree.

    FP8 weights remain byte-identical and are merely viewed as the uint32 words
    MLX's native ``mxfp8`` operator expects.  Their 128x128 E8M0 scale grid is
    repeated into the equivalent per-row, group-32 grid.  EXL3 experts are
    stacked expert-major without decoding or requantizing their payload.
    ``mtp.*`` belongs to the separately loaded K64 draft and is excluded here.
    """

    source = dict(weights)
    mapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    expert_fields: dict[tuple[int, str, str], dict[int, mx.array]] = {}

    for name, value in source.items():
        if name.startswith("mtp."):
            consumed.add(name)
            continue
        match = _EXPERT_KEY.match(name)
        if match is None:
            continue
        consumed.add(name)
        if match.group("field") == "mcg":
            continue
        key = (
            int(match.group("layer")),
            match.group("projection"),
            match.group("field"),
        )
        expert_fields.setdefault(key, {})[int(match.group("expert"))] = value

    for name, value in source.items():
        if name in consumed:
            continue
        if name.endswith(".scale"):
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name in source and source[weight_name].dtype == mx.uint8:
                continue
        if name.endswith(".weight"):
            scale_name = name.removesuffix(".weight") + ".scale"
            scales = source.get(scale_name)
            if scales is not None and value.dtype == mx.uint8:
                if value.ndim != 2 or int(value.shape[1]) % 128:
                    raise ValueError(f"unsupported Mia FP8 weight geometry: {name}")
                output_dims, input_dims = (int(dim) for dim in value.shape)
                target = _map_mia_target_name(name)
                mapped[target] = mx.contiguous(value).view(mx.uint32)
                mapped[target.removesuffix(".weight") + ".scales"] = (
                    _expand_mia_fp8_block_scales(
                        scales,
                        output_dims,
                        input_dims,
                    )
                )
                consumed.update((name, scale_name))
                continue
        target = _map_mia_target_name(name)
        if target.endswith(".ffn.gate.tid2eid") and value.dtype == mx.int64:
            value = value.astype(mx.int32)
        mapped[target] = value
        consumed.add(name)

    expected_ids = set(range(experts))
    for layer in range(layers):
        for projection, target_projection in _PROJECTION_NAMES.items():
            for field in ("trellis", "suh", "svh"):
                values = expert_fields.get((layer, projection, field), {})
                if set(values) != expected_ids:
                    raise ValueError(
                        f"Mia EXL3 layer {layer} {projection}.{field} has "
                        f"{len(values)} experts, expected {experts}"
                    )
                target = (
                    f"model.layers.{layer}.ffn.switch_mlp."
                    f"{target_projection}.{field}"
                )
                mapped[target] = mx.stack(
                    [values[expert] for expert in range(experts)], axis=0
                )

    return mapped


def _open_file_identity(stream) -> tuple[int, int, int, int, int]:
    observed = os.fstat(stream.fileno())
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("Mia shard must be a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _load_verified_safetensors(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_canonical_sha256: str | None = None,
) -> dict[str, mx.array]:
    """Hash and load one stable descriptor with an optional semantic header pin."""

    path = Path(path)
    if len(expected_sha256) != 64:
        raise ValueError(f"pinned Mia shard checksum is invalid: {path.name}")
    if (
        expected_canonical_sha256 is not None
        and len(expected_canonical_sha256) != 64
    ):
        raise ValueError(
            f"pinned Mia shard canonical checksum is invalid: {path.name}"
        )
    with path.open("rb", buffering=0) as stream:
        identity = _open_file_identity(stream)
        if identity[2] != int(expected_bytes):
            raise ValueError(f"pinned Mia shard size changed: {path.name}")

        digest = hashlib.sha256()
        canonical_digest = None
        if expected_canonical_sha256 is not None:
            from mtplx.deepseek_v4_mia_engine import (
                _SAFETENSORS_CANONICAL_PREFIX,
                _canonical_safetensors_header,
            )

            encoded_header_length = stream.read(8)
            if len(encoded_header_length) != 8:
                raise ValueError(f"invalid Mia safetensors header: {path.name}")
            header_length = struct.unpack("<Q", encoded_header_length)[0]
            if header_length == 0 or header_length > identity[2] - 8:
                raise ValueError(f"invalid Mia safetensors header: {path.name}")
            encoded_header = stream.read(header_length)
            if len(encoded_header) != header_length:
                raise ValueError(f"truncated Mia safetensors header: {path.name}")
            try:
                canonical_header = _canonical_safetensors_header(encoded_header)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Mia safetensors JSON header: {path.name}"
                ) from exc
            digest.update(encoded_header_length)
            digest.update(encoded_header)
            canonical_digest = hashlib.sha256()
            canonical_digest.update(_SAFETENSORS_CANONICAL_PREFIX)
            canonical_digest.update(encoded_header_length)
            canonical_digest.update(struct.pack("<Q", len(canonical_header)))
            canonical_digest.update(canonical_header)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if canonical_digest is not None:
                canonical_digest.update(chunk)
        if _open_file_identity(stream) != identity:
            raise ValueError(
                f"pinned Mia shard changed while validating: {path.name}"
            )
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"pinned Mia shard checksum changed: {path.name} "
                f"observed={observed_sha256}, expected={expected_sha256}"
            )
        if canonical_digest is not None and (
            canonical_digest.hexdigest() != expected_canonical_sha256
        ):
            observed_canonical_sha256 = canonical_digest.hexdigest()
            raise ValueError(
                f"pinned Mia shard canonical checksum changed: {path.name} "
                f"observed={observed_canonical_sha256}, "
                f"expected={expected_canonical_sha256}"
            )

        stream.seek(0)
        weights = mx.load(stream, format="safetensors")
        if _open_file_identity(stream) != identity:
            raise ValueError(
                f"pinned Mia shard changed while loading: {path.name}"
            )
    if not isinstance(weights, dict):
        raise ValueError(f"invalid Mia safetensors shard: {path}")
    return dict(weights)


def load_indexed_safetensors(
    root: Path | str,
    *,
    weight_map: dict[str, str] | None = None,
    shard_pins: tuple[Any, ...] | None = None,
) -> dict[str, mx.array]:
    """Load exactly the tensors named by one local safetensors index."""

    root = Path(root)
    if weight_map is None:
        weight_map = _indexed_weight_map(root)
    if not weight_map:
        raise ValueError(f"invalid safetensors index: {root}")
    pins_by_name = (
        {str(pin.name): pin for pin in shard_pins}
        if shard_pins is not None
        else None
    )
    filenames = set(weight_map.values())
    if pins_by_name is not None and set(pins_by_name) != filenames:
        raise ValueError("Mia shard pins do not match the safetensors index")
    expected = set(weight_map)
    weights: dict[str, mx.array] = {}
    for filename in sorted(filenames):
        shard = root / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)
        if pins_by_name is None:
            weights.update(mx.load(str(shard)))
        else:
            pin = pins_by_name[filename]
            weights.update(
                _load_verified_safetensors(
                    shard,
                    expected_bytes=int(pin.bytes),
                    expected_sha256=str(pin.sha256),
                    expected_canonical_sha256=str(pin.canonical_sha256),
                )
            )
    observed = set(weights)
    if observed != expected:
        raise ValueError(
            f"safetensors index mismatch in {root}: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    return weights


def _indexed_weight_map(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    return {str(name): str(filename) for name, filename in weight_map.items()}


def _install_quantized_modules(
    model: nn.Module,
    expected: dict[str, str],
    *,
    prefix: str,
) -> dict[str, str]:
    selected: set[str] = set()

    def predicate(path: str, module: nn.Module):
        if not path.startswith(prefix) or not hasattr(module, "to_quantized"):
            return False
        mode = expected.get(path)
        if mode == "mxfp4":
            selected.add(path)
            return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        if mode == "mxfp8":
            selected.add(path)
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return False

    nn.quantize(model, class_predicate=predicate)
    if selected != set(expected):
        raise ValueError(
            f"Mia quantized module ownership mismatch under {prefix!r}: "
            f"missing={sorted(set(expected) - selected)!r}, "
            f"extra={sorted(selected - set(expected))!r}"
        )
    installed = dict(model.named_modules())
    for path, mode in expected.items():
        module = installed.get(path)
        bits = 4 if mode == "mxfp4" else 8
        if (
            module is None
            or int(getattr(module, "group_size", 0)) != 32
            or int(getattr(module, "bits", 0)) != bits
            or str(getattr(module, "mode", "")) != mode
            or getattr(getattr(module, "weight", None), "dtype", None)
            != mx.uint32
            or getattr(getattr(module, "scales", None), "dtype", None)
            != mx.uint8
            or getattr(module, "biases", None) is not None
        ):
            raise ValueError(
                f"Mia module {path!r} did not install group-32 {mode} ownership"
            )
    return dict(expected)


def _target_quantized_modules_from_index(
    weight_map: dict[str, str],
) -> dict[str, str]:
    expected: dict[str, str] = {}
    for scale_name in weight_map:
        if scale_name.startswith("mtp.") or not scale_name.endswith(".scale"):
            continue
        weight_name = scale_name.removesuffix(".scale") + ".weight"
        if weight_name not in weight_map:
            continue
        target_weight = _map_mia_target_name(weight_name)
        expected[target_weight.removesuffix(".weight")] = "mxfp8"
    return expected


def _map_mia_target_carried_shard(
    source: dict[str, mx.array],
    *,
    fp8_geometries: dict[str, tuple[int, int]],
) -> dict[str, mx.array]:
    """Map one bounded non-expert shard without retaining the other shards.

    The TP1 package splits two FP8 weight/scale pairs across adjacent carried
    shards.  Their installed module geometry is fixed before streaming starts,
    so it supplies both cross-shard ownership and the exact scale expansion
    dimensions without retaining either source shard.
    """

    mapped: dict[str, mx.array] = {}
    for name, value in source.items():
        if name.startswith("mtp."):
            continue
        if _EXPERT_KEY.match(name) is not None:
            raise ValueError("an EXL3 expert tensor reached a carried Mia shard")
        target = _map_mia_target_name(name)
        if name.endswith(".scale"):
            module_name = target.removesuffix(".scale")
            geometry = fp8_geometries.get(module_name)
            if geometry is not None:
                output_dims, input_dims = geometry
                mapped[module_name + ".scales"] = _expand_mia_fp8_block_scales(
                    value,
                    output_dims,
                    input_dims,
                )
                continue
        if name.endswith(".weight"):
            module_name = target.removesuffix(".weight")
            geometry = fp8_geometries.get(module_name)
            if geometry is not None:
                if value.dtype != mx.uint8 or value.ndim != 2:
                    raise ValueError(f"unsupported Mia FP8 weight geometry: {name}")
                output_dims, input_dims = geometry
                if tuple(value.shape) != (output_dims, input_dims):
                    raise ValueError(
                        f"unsupported Mia FP8 weight geometry: {name} owns "
                        f"{tuple(value.shape)}, expected {(output_dims, input_dims)}"
                    )
                mapped[target] = mx.contiguous(value).view(mx.uint32)
                continue
        if target.endswith(".ffn.gate.tid2eid") and value.dtype == mx.int64:
            value = value.astype(mx.int32)
        if ".attn.wo_a." in target and value.ndim == 3:
            value = value.reshape(
                int(value.shape[0]) * int(value.shape[1]),
                int(value.shape[2]),
            )
        mapped[target] = value
    return mapped


def _map_mia_target_expert_shard(
    source: dict[str, mx.array],
    *,
    layer: int,
    experts: int,
) -> dict[str, mx.array]:
    """Stack one layer-local EXL3 shard and discard its unused MCG mirrors."""

    fields: dict[tuple[str, str], dict[int, mx.array]] = {}
    observed: set[str] = set()
    for name, value in source.items():
        match = _EXPERT_KEY.match(name)
        if match is None or int(match.group("layer")) != int(layer):
            raise ValueError(
                f"EXL3 shard for layer {layer} owns unexpected tensor {name!r}"
            )
        observed.add(name)
        if match.group("field") == "mcg":
            continue
        key = (match.group("projection"), match.group("field"))
        fields.setdefault(key, {})[int(match.group("expert"))] = value

    expected_ids = set(range(int(experts)))
    mapped: dict[str, mx.array] = {}
    for projection, target_projection in _PROJECTION_NAMES.items():
        for field in ("trellis", "suh", "svh"):
            values = fields.get((projection, field), {})
            if set(values) != expected_ids:
                raise ValueError(
                    f"Mia EXL3 layer {layer} {projection}.{field} has "
                    f"{len(values)} experts, expected {experts}"
                )
            target = (
                f"model.layers.{int(layer)}.ffn.switch_mlp."
                f"{target_projection}.{field}"
            )
            mapped[target] = mx.stack(
                [values[expert] for expert in range(int(experts))],
                axis=0,
            )
    if observed != set(source):
        raise ValueError(f"Mia EXL3 layer {layer} shard was not fully consumed")
    return mapped


def _install_mia_weight_batch(
    model: nn.Module,
    mapped: dict[str, mx.array],
    installed_names: set[str],
) -> None:
    overlap = installed_names.intersection(mapped)
    if overlap:
        raise ValueError(f"Mia target parameters were loaded twice: {sorted(overlap)!r}")
    if mapped:
        mx.eval(*mapped.values())
        model.load_weights(list(mapped.items()), strict=False)
        installed_names.update(mapped)


def load_mia_exl3_target_streaming(
    model: nn.Module,
    root: Path | str,
    *,
    layers: int,
    experts: int,
    weight_map: dict[str, str] | None = None,
    shard_pins: tuple[Any, ...] | None = None,
) -> dict[str, str]:
    """Install the 106 GB target with one source shard live at a time.

    The rank-sliced package owns five carried shards and one complete EXL3 shard
    per target layer.  Keeping that boundary avoids the former all-shards source
    dictionary and bounds conversion scratch to one carried shard or one 2 GB
    expert layer while preserving the exact destination tensors.
    """

    from mlx.utils import tree_flatten

    root = Path(root)
    if weight_map is None:
        weight_map = _indexed_weight_map(root)
    pins_by_name = (
        {str(pin.name): pin for pin in shard_pins}
        if shard_pins is not None
        else None
    )
    if pins_by_name is not None and set(pins_by_name) != set(weight_map.values()):
        raise ValueError("Mia target shard pins do not match its weight index")

    def load_shard(filename: str) -> dict[str, mx.array]:
        shard = root / filename
        if pins_by_name is None:
            return dict(mx.load(str(shard)))
        pin = pins_by_name[filename]
        return _load_verified_safetensors(
            shard,
            expected_bytes=int(pin.bytes),
            expected_sha256=str(pin.sha256),
            expected_canonical_sha256=str(pin.canonical_sha256),
        )

    quantized = _target_quantized_modules_from_index(weight_map)
    model_quantized = {
        path: mode for path, mode in quantized.items() if path.startswith("model.")
    }
    head_quantized = {
        path: mode for path, mode in quantized.items() if path.startswith("lm_head")
    }
    receipt: dict[str, str] = {}
    receipt.update(
        _install_quantized_modules(model, model_quantized, prefix="model.")
    )
    receipt.update(
        _install_quantized_modules(model, head_quantized, prefix="lm_head")
    )
    installed_modules = dict(model.named_modules())
    fp8_geometries: dict[str, tuple[int, int]] = {}
    for path in quantized:
        module = installed_modules[path]
        output_dims = int(module.scales.shape[0])
        input_dims = int(module.scales.shape[1]) * 32
        if tuple(module.weight.shape) != (output_dims, input_dims // 4):
            raise ValueError(f"Mia module {path!r} has invalid MXFP8 geometry")
        fp8_geometries[path] = (output_dims, input_dims)

    files: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        files.setdefault(filename, set()).add(name)
    installed_names: set[str] = set()

    carried_files = {
        name for name in files if name.startswith("carried-")
    }
    for filename in sorted(carried_files):
        source = load_shard(filename)
        if set(source) != files[filename]:
            raise ValueError(f"Mia carried shard index mismatch: {filename}")
        mapped = _map_mia_target_carried_shard(
            source,
            fp8_geometries=fp8_geometries,
        )
        _install_mia_weight_batch(model, mapped, installed_names)
        del mapped, source
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    expert_files = {
        name for name in files if name.startswith("exl3-layer-")
    }
    if len(expert_files) != int(layers):
        raise ValueError(
            f"Mia target owns {len(expert_files)} EXL3 layer shards, "
            f"expected {layers}"
        )
    for layer in range(int(layers)):
        filename = f"exl3-layer-{layer:03d}-tp1-rank0.safetensors"
        if filename not in expert_files:
            raise ValueError(f"Mia target is missing {filename}")
        source = load_shard(filename)
        if set(source) != files[filename]:
            raise ValueError(f"Mia EXL3 shard index mismatch: {filename}")
        mapped = _map_mia_target_expert_shard(
            source,
            layer=layer,
            experts=experts,
        )
        _install_mia_weight_batch(model, mapped, installed_names)
        del mapped, source
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    unexpected_files = set(files) - carried_files - expert_files
    if unexpected_files:
        raise ValueError(
            f"Mia target index owns unexpected shards: {sorted(unexpected_files)!r}"
        )
    installed_parameters = {
        name for name, _value in tree_flatten(model.parameters())
    }
    if installed_names != installed_parameters:
        raise ValueError(
            "Mia streaming target parameter mismatch: "
            f"missing={len(installed_parameters - installed_names)}, "
            f"extra={len(installed_names - installed_parameters)}"
        )
    model._mia_target_load_receipt = {
        "mode": "bounded_one_shard",
        "artifact_identity": (
            "raw_canonical_sha256_same_fd"
            if pins_by_name is not None
            else "unverified_path"
        ),
        "source_shards": len(files),
        "carried_shards": len(carried_files),
        "exl3_layer_shards": len(expert_files),
        "installed_parameters": len(installed_names),
    }
    return receipt


def _map_mia_dspark_name(name: str) -> str:
    mapped = name
    replacements = (
        (".ffn.shared_experts.w1", ".ffn.shared_experts.gate_proj"),
        (".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"),
        (".ffn.shared_experts.w3", ".ffn.shared_experts.up_proj"),
        (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
        (".hc_head_fn", ".hc_head.fn"),
        (".hc_head_base", ".hc_head.base"),
        (".hc_head_scale", ".hc_head.scale"),
    )
    for source, target in replacements:
        mapped = mapped.replace(source, target)
    return mapped


def sanitize_mia_dspark_weights(
    weights: dict[str, mx.array],
    *,
    stages: int,
    experts: int,
) -> dict[str, mx.array]:
    """Map the exact Mia K64 draft onto the installed three-stage owner.

    Routed weights remain byte-identical OCP FP4 and dense projections remain
    byte-identical E4M3.  Only their array views and scale-grid ownership change
    to the native MLX ``mxfp4``/``mxfp8`` module contracts.
    """

    source = dict(weights)
    mapped: dict[str, mx.array] = {}
    consumed: set[str] = set()
    expert_fields: dict[tuple[int, str, str], dict[int, mx.array]] = {}

    for name, value in source.items():
        match = _DSPARK_EXPERT_KEY.match(name)
        if match is None:
            continue
        consumed.add(name)
        key = (
            int(match.group("stage")),
            match.group("projection"),
            match.group("field"),
        )
        expert_fields.setdefault(key, {})[int(match.group("expert"))] = value

    for name, value in source.items():
        if name in consumed:
            continue
        if name.endswith(".scale"):
            weight_name = name.removesuffix(".scale") + ".weight"
            if weight_name in source and source[weight_name].dtype == mx.uint8:
                continue
        if name.endswith(".weight"):
            scale_name = name.removesuffix(".weight") + ".scale"
            scales = source.get(scale_name)
            if scales is not None and value.dtype == mx.uint8:
                if value.ndim != 2 or int(value.shape[1]) % 128:
                    raise ValueError(f"unsupported Mia DSpark FP8 geometry: {name}")
                output_dims, input_dims = (int(dim) for dim in value.shape)
                target = _map_mia_dspark_name(name)
                mapped[target] = mx.contiguous(value).view(mx.uint32)
                mapped[target.removesuffix(".weight") + ".scales"] = (
                    _expand_mia_fp8_block_scales(scales, output_dims, input_dims)
                )
                consumed.update((name, scale_name))
                continue
        mapped[_map_mia_dspark_name(name)] = value
        consumed.add(name)

    expected_ids = set(range(experts))
    for stage in range(stages):
        for projection, target_projection in _PROJECTION_NAMES.items():
            weights_by_expert = expert_fields.get((stage, projection, "weight"), {})
            scales_by_expert = expert_fields.get((stage, projection, "scale"), {})
            if set(weights_by_expert) != expected_ids or set(scales_by_expert) != expected_ids:
                raise ValueError(
                    f"Mia DSpark stage {stage} {projection} has incomplete K{experts} storage"
                )
            stem = f"mtp.{stage}.ffn.switch_mlp.{target_projection}"
            mapped[f"{stem}.weight"] = mx.stack(
                [
                    mx.contiguous(weights_by_expert[expert]).view(mx.uint32)
                    for expert in range(experts)
                ],
                axis=0,
            )
            mapped[f"{stem}.scales"] = mx.stack(
                [scales_by_expert[expert] for expert in range(experts)], axis=0
            )

    if consumed != set(source):
        raise ValueError(
            "unmapped Mia DSpark tensors: " + ", ".join(sorted(set(source) - consumed))
        )
    return mapped


def _quantize_loaded_modules(
    model: nn.Module,
    weights: dict[str, mx.array],
    *,
    prefix: str,
) -> dict[str, str]:
    expected = {
        name.removesuffix(".scales"): (
            "mxfp4" if ".ffn.switch_mlp." in name else "mxfp8"
        )
        for name in weights
        if name.startswith(prefix) and name.endswith(".scales")
    }
    return _install_quantized_modules(model, expected, prefix=prefix)


def _default_mia_dspark_root(target_root: Path) -> Path:
    configured = json.loads((target_root / "config.json").read_text()).get(
        "dspark_draft_model"
    )
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else target_root / candidate
    suffix = "-tp1"
    if target_root.name.endswith(suffix):
        return target_root.with_name(
            target_root.name.removesuffix(suffix) + "-dspark-k64"
        )
    return target_root / "dspark-k64"


def load_mia_exl3_dspark_model(
    target_root: Path | str,
    *,
    draft_root: Path | str | None = None,
    artifact_validation=None,
    lazy: bool = False,
    context_capacity_tokens: int = 384_000,
    max_batch_tokens: int = 8_224,
):
    """Construct the exact split Mia K216 target plus K64 DSpark owner."""

    from mlx.utils import tree_flatten

    from mtplx.models.deepseek_v4 import Model, ModelArgs
    from mtplx.models.deepseek_v4_dspark import build_deepseek_v4_dspark

    target_root = Path(target_root).resolve()
    resolved_draft = (
        Path(draft_root).resolve()
        if draft_root is not None
        else _default_mia_dspark_root(target_root).resolve()
    )
    from mtplx.deepseek_v4_mia_engine import validate_pinned_mia_artifacts

    if artifact_validation is None:
        artifact_validation = validate_pinned_mia_artifacts(
            target_root,
            resolved_draft,
        )
    elif (
        artifact_validation.target_root != target_root
        or artifact_validation.draft_root != resolved_draft
    ):
        raise ValueError(
            "pinned Mia artifact validation does not own the requested roots"
        )
    target_config = dict(artifact_validation.target_config)
    # Qualify the source metadata before clearing only the separately-owned
    # draft signature for target construction.
    ModelArgs.from_dict(target_config)
    target_only = dict(target_config)
    target_only.update(
        {
            "dspark_block_size": None,
            "dspark_markov_rank": None,
            "dspark_noise_token_id": None,
            "dspark_target_layer_ids": None,
            "num_nextn_predict_layers": 0,
        }
    )
    model = Model(ModelArgs.from_dict(target_only))
    quantized_modules = load_mia_exl3_target_streaming(
        model,
        target_root,
        layers=int(model.args.num_hidden_layers),
        experts=int(model.args.n_routed_experts),
        weight_map=artifact_validation.target_weight_map,
        shard_pins=artifact_validation.target_shards,
    )
    model._mia_target_load_receipt["small_file_sha256"] = dict(
        artifact_validation.target_small_file_sha256
    )
    model.eval()
    for layer in model.layers:
        layer.ffn.install_mia_exl3_runtime(max_tokens=8224)

    draft_config = dict(artifact_validation.draft_config)
    draft_experts = int(draft_config.get("n_routed_experts", 0))
    draft_config["hybrid_tr3_tail"] = None
    draft_args = ModelArgs.from_dict(draft_config)
    if draft_experts != 64:
        raise ValueError(f"Mia DSpark draft must own K64, got K{draft_experts}")
    owner = build_deepseek_v4_dspark(draft_args)
    model.install_dspark_owner(owner)

    draft_source = load_indexed_safetensors(
        resolved_draft,
        weight_map=artifact_validation.draft_weight_map,
        shard_pins=artifact_validation.draft_shards,
    )
    draft_weights = sanitize_mia_dspark_weights(
        draft_source,
        stages=3,
        experts=64,
    )
    del draft_source
    model._mia_draft_load_receipt = {
        "mode": "single_shard",
        "artifact_identity": "raw_canonical_sha256_same_fd",
        "source_shards": len(artifact_validation.draft_shards),
        "source_tensors": len(artifact_validation.draft_weight_map),
        "small_file_sha256": dict(
            artifact_validation.draft_small_file_sha256
        ),
    }
    quantized_modules.update(
        _quantize_loaded_modules(model, draft_weights, prefix="mtp.")
    )
    installed = {
        name: value
        for name, value in tree_flatten(model.parameters())
        if name.startswith("mtp.")
    }
    if set(installed) != set(draft_weights):
        raise ValueError(
            "Mia DSpark installed parameter mismatch: "
            f"missing={len(set(installed) - set(draft_weights))}, "
            f"extra={len(set(draft_weights) - set(installed))}"
        )
    for name, value in installed.items():
        if value.shape != draft_weights[name].shape:
            raise ValueError(
                f"Mia DSpark shape mismatch for {name}: "
                f"installed={value.shape}, source={draft_weights[name].shape}"
            )
    model.load_weights(list(draft_weights.items()), strict=False)
    model._mia_quantized_modules = dict(quantized_modules)

    # Replace the layer-local mHC chain with the pinned carried state machine
    # for both the 43-layer target and the three-stage K64 draft.
    model.install_mia_mhc_runtime(max_tokens=8224)

    # Bind the source-derived B12X WO owner directly against the native MXFP8
    # tensors after both target and draft weights exist.  Each attention owns a
    # distinct prebound plan; generation never enters the generic o-LoRA route.
    from mtplx.models.deepseek_v4 import (
        install_mia_qkv_prologue_routes,
        install_mia_tp1_wo_projection_routes,
        install_mia_stacked_projections,
    )

    wo_projection = install_mia_tp1_wo_projection_routes(
        model,
        max_prefill_rows=max_batch_tokens,
    )
    if (
        wo_projection["route"] != "mia_tp1_b12x_wo_mxfp8"
        or wo_projection["target_attention"] != 43
        or wo_projection["draft_attention"] != 3
        or wo_projection["plan_count"] != 46
        or wo_projection["unique_plan_count"] != 46
        or wo_projection["plan_type"] != "MiaTP1WOMXFP8Plan"
        or wo_projection["max_prefill_rows"] != int(max_batch_tokens)
    ):
        raise RuntimeError(
            f"Mia TP1 WO projection route is incomplete: {wo_projection}"
        )

    stacked_projections = install_mia_stacked_projections(model)
    if stacked_projections != {
        "target_attention": 43,
        "draft_attention": 3,
        "main_compressor": 41,
        "indexer_compressor": 21,
    }:
        raise RuntimeError(
            f"Mia stacked projection installation is incomplete: "
            f"{stacked_projections}"
        )

    qkv_prologue = install_mia_qkv_prologue_routes(model)
    if (
        qkv_prologue["route"] != "mia_fused_qkv_stock432"
        or qkv_prologue["target_attention"] != 43
        or qkv_prologue["draft_attention"] != 3
        or qkv_prologue["plan_count"] != 46
        or qkv_prologue["unique_plan_count"] != 46
        or qkv_prologue["plan_type"] != "MiaBoundQKVPrologue"
        or qkv_prologue["prefill_cutoff"] != 1024
        or qkv_prologue["proposal_rows"] != 5
        or qkv_prologue["context_rows"] != 128
    ):
        raise RuntimeError(
            f"Mia fused Q/KV prologue installation is incomplete: {qkv_prologue}"
        )

    from mtplx.deepseek_v4_mia_engine import build_mia_engine_plan

    engine_plan = build_mia_engine_plan(
        model,
        target_root=target_root,
        draft_root=resolved_draft,
        context_capacity_tokens=context_capacity_tokens,
        max_batch_tokens=max_batch_tokens,
    )
    model.install_mia_engine_plan(engine_plan)
    if not lazy:
        mx.eval(model.parameters())
        model._mia_prewarm_receipt = engine_plan.prewarm(model)
    return model
