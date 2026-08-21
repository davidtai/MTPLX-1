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
import json
from pathlib import Path
import re
from typing import Any

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

    def __call__(self, x: mx.array, expert_ids: mx.array) -> mx.array:
        original_dtype = x.dtype
        x_half = x.astype(mx.float16)
        rows = int(expert_ids.shape[0])
        tasks = int(expert_ids.size)
        # Packaged DSpark verifies one primary plus five future tokens.  Keep
        # every M1..M6 decode/verify call on the authentic direct QMV kernel;
        # route only wider prefill work to the sorted M-tiled MMA path.
        mma = rows > 6
        if not mma:
            gate = self.gate_proj(x_half, expert_ids)
            up = self.up_proj(x_half, expert_ids)
            if self.limit > 0:
                gate = mx.minimum(gate, self.limit)
                up = mx.clip(up, -self.limit, self.limit)
            activated = (nn.silu(gate) * up).astype(mx.float16)
            return self.down_proj(activated, expert_ids).astype(original_dtype)

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


def load_indexed_safetensors(root: Path | str) -> dict[str, mx.array]:
    """Load exactly the tensors named by one local safetensors index."""

    root = Path(root)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    expected = set(weight_map)
    weights: dict[str, mx.array] = {}
    for filename in sorted(set(weight_map.values())):
        shard = root / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)
        weights.update(mx.load(str(shard)))
    observed = set(weights)
    if observed != expected:
        raise ValueError(
            f"safetensors index mismatch in {root}: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    return weights


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
) -> None:
    def predicate(path: str, module: nn.Module):
        if not path.startswith(prefix) or not hasattr(module, "to_quantized"):
            return False
        if ".ffn.switch_mlp." in path:
            return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        if f"{path}.scales" in weights:
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return False

    nn.quantize(model, class_predicate=predicate)


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
    lazy: bool = False,
):
    """Construct the exact split Mia K216 target plus K64 DSpark owner."""

    from mlx.utils import tree_flatten

    from mtplx.models.deepseek_v4 import Model, ModelArgs
    from mtplx.models.deepseek_v4_dspark import build_deepseek_v4_dspark

    target_root = Path(target_root).resolve()
    target_config = json.loads((target_root / "config.json").read_text())
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
    target_weights = model.sanitize(load_indexed_safetensors(target_root))
    _quantize_loaded_modules(model, target_weights, prefix="model.")
    _quantize_loaded_modules(model, target_weights, prefix="lm_head")
    model.eval()
    model.load_weights(list(target_weights.items()), strict=True)

    resolved_draft = (
        Path(draft_root).resolve()
        if draft_root is not None
        else _default_mia_dspark_root(target_root).resolve()
    )
    draft_config = json.loads((resolved_draft / "config.json").read_text())
    draft_experts = int(draft_config.get("n_routed_experts", 0))
    draft_config["hybrid_tr3_tail"] = None
    draft_args = ModelArgs.from_dict(draft_config)
    if draft_experts != 64:
        raise ValueError(f"Mia DSpark draft must own K64, got K{draft_experts}")
    owner = build_deepseek_v4_dspark(draft_args)
    model.install_dspark_owner(owner)

    # The exact Mia route inherits the independently measured DeepSeek-V4
    # fixed-HC Sinkhorn win.  Bind it once to all 43 target layers and all three
    # DSpark stages; the token path sees only the installed callable.
    sinkhorn_owners = tuple(model.layers) + tuple(owner.stages)
    for layer in sinkhorn_owners:
        layer.attn_hc.install_sinkhorn_kernel()
        layer.ffn_hc.install_sinkhorn_kernel()

    draft_weights = sanitize_mia_dspark_weights(
        load_indexed_safetensors(resolved_draft),
        stages=3,
        experts=64,
    )
    _quantize_loaded_modules(model, draft_weights, prefix="mtp.")
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

    # Reuse the existing direct grouped o-LoRA route against Mia's native MXFP8
    # packing.  All target and draft attention owners are validated and bound
    # here; generation never materializes or consults a fallback lane.
    from mtplx.models.deepseek_v4 import install_deepseek_v4_o_lora_routes

    o_lora = install_deepseek_v4_o_lora_routes(
        model,
        mode="gather_qmm",
        canonical_mixed_route=False,
    )
    if (
        o_lora["module_count"] != 46
        or o_lora["trunk_module_count"] != 43
        or o_lora["mtp_module_count"] != 3
        or not o_lora["all_direct"]
        or not o_lora["all_mode_matches"]
    ):
        raise RuntimeError(f"Mia direct o-LoRA route is incomplete: {o_lora}")
    if not lazy:
        mx.eval(model.parameters())
    return model
