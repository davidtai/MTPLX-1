"""Mia-compatible paged FP8 indexer cache for DeepSeek V4 ratio-4 layers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math

import mlx.core as mx

from mtplx.attention_context import current_attention_phase
from mtplx.paged_cache import PagedCachePlan, PagedCachePool


INDEXER_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_TOPK = 512
INDEXER_RECORD_BYTES = 132
# SparkInfer's production paged prefill route owns one 32K K supertile and
# folds it into a fixed top-512 carry.  The scorer's physical tile remains
# 16-wide on Metal; this constant is the bounded selector workspace extent.
INDEXER_PREFILL_SCORE_CHUNK_ROWS = 32768
INDEXER_DECODE_SLICE_ROWS = 4096


_FP8_HEADER = r"""
    using namespace metal;

    inline uchar mtplx_indexer_e4m3_encode(float value) {
        uint sign = as_type<uint>(value) >> 31;
        float magnitude = min(abs(value), 448.0f);
        uchar code;
        constexpr float MIN_NORMAL = 0.015625f;
        constexpr float SUB_STEP = 0.001953125f;
        if (!(magnitude > 0.0f)) {
            code = uchar(0);
        } else if (magnitude < MIN_NORMAL) {
            uint mantissa = uint(rint(magnitude / SUB_STEP));
            code = mantissa >= 8u ? uchar(0x08) : uchar(mantissa);
        } else {
            int exponent = int(floor(log2(magnitude)));
            float step = exp2(float(exponent - 3));
            uint significand = uint(rint(magnitude / step));
            if (significand >= 16u) {
                exponent += 1;
                significand = 8u;
            }
            uint stored_exponent = uint(exponent + 7);
            if (stored_exponent >= 15u) {
                stored_exponent = 15u;
                significand = min(significand, 14u);
            }
            code = uchar((stored_exponent << 3) | (significand - 8u));
        }
        return uchar(code | uchar(sign << 7));
    }

    inline float mtplx_indexer_e4m3_decode(uchar raw) {
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        float magnitude = exponent == 0u
            ? float(mantissa) * 0.001953125f
            : (1.0f + float(mantissa) * 0.125f)
                * exp2(float(int(exponent) - 7));
        return (uint(raw) & 0x80u) != 0u ? -magnitude : magnitude;
    }

    inline float mtplx_indexer_record_scale(const device uchar* record) {
        uint scale_bits = uint(record[128u])
            | (uint(record[129u]) << 8u)
            | (uint(record[130u]) << 16u)
            | (uint(record[131u]) << 24u);
        return as_type<float>(scale_bits);
    }
"""


@lru_cache(maxsize=1)
def _pack_kernel():
    source = r"""
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        const device T* source_row = rows + size_t(row) * 128u;
        device uchar* record = records + size_t(row) * 132u;

        uint dim0 = lane * 4u;
        float local_amax = 0.0f;
        for (uint element = 0u; element < 4u; ++element) {
            local_amax = max(
                local_amax,
                abs(float(source_row[dim0 + element]))
            );
        }
        float amax = simd_max(local_amax);
        float scale = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            record[dim] = mtplx_indexer_e4m3_encode(
                float(source_row[dim]) / scale
            );
        }
        if (lane < 4u) {
            uint scale_bits = as_type<uint>(scale);
            record[128u + lane] = uchar((scale_bits >> (8u * lane)) & 0xffu);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_indexer_fp8_pack",
        input_names=["rows"],
        output_names=["records"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _pack_indexer132(rows: mx.array) -> mx.array:
    if rows.ndim < 2 or int(rows.shape[-1]) != INDEXER_HEAD_DIM:
        raise ValueError("Mia indexer rows must end in width 128")
    row_count = math.prod(int(dim) for dim in rows.shape[:-1])
    return _pack_kernel()(
        inputs=[mx.contiguous(rows)],
        template=[("T", rows.dtype)],
        grid=(row_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*rows.shape[:-1], INDEXER_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]


@lru_cache(maxsize=1)
def _query_rope_quant_kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint head = group % 64u;
        uint query = group / 64u;
        const device T* q_row = queries
            + (size_t(query) * 64u + size_t(head)) * 128u;
        device uchar* record = records
            + (size_t(query) * 64u + size_t(head)) * 132u;
        threadgroup float rotated[128];

        uint dim0 = lane * 4u;
        float local_max = 0.0f;
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            float value = float(q_row[dim]);
            if (dim >= 64u) {
                uint rope_dim = dim - 64u;
                uint pair = rope_dim / 2u;
                float angle = float(positions[query]) * float(inv_freq[pair]);
                float c = cos(angle);
                float s = sin(angle);
                uint even_dim = 64u + pair * 2u;
                float even = float(q_row[even_dim]);
                float odd = float(q_row[even_dim + 1u]);
                value = (rope_dim & 1u) == 0u
                    ? even * c - odd * s
                    : odd * c + even * s;
                value = float(bfloat(value));
            }
            rotated[dim] = value;
            local_max = max(local_max, abs(value));
        }
        float amax = simd_max(local_max);
        float scale = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint element = 0u; element < 4u; ++element) {
            uint dim = dim0 + element;
            record[dim] = mtplx_indexer_e4m3_encode(rotated[dim] / scale);
        }
        if (lane < 4u) {
            uint one_bits = as_type<uint>(1.0f);
            record[128u + lane] = uchar((one_bits >> (8u * lane)) & 0xffu);
        }
        if (lane == 0u) {
            scaled_weights[size_t(query) * 64u + head] =
                float(weights[size_t(query) * 64u + head])
                * scale * float(weight_scale);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_indexer_q_rope_fp8",
        input_names=[
            "queries",
            "weights",
            "positions",
            "inv_freq",
            "weight_scale",
        ],
        output_names=["records", "scaled_weights"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def fused_indexer_query_records(
    queries: mx.array,
    weights: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    *,
    weight_scale: float,
) -> tuple[MiaIndexerQueryRecords, mx.array]:
    """Fuse source-order Q RoPE/FP8 quantization and Q-scale weight folding."""
    batch, query_count, heads, head_dim = (int(dim) for dim in queries.shape)
    if (batch, heads, head_dim) != (1, INDEXER_HEADS, INDEXER_HEAD_DIM):
        raise ValueError("Mia indexer Q requires [1, rows, 64, 128]")
    return _run_fused_indexer_query_records(
        queries,
        weights,
        positions,
        inv_freq,
        weight_scale=weight_scale,
    )


def _run_fused_indexer_query_records(
    queries: mx.array,
    weights: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    *,
    weight_scale: float,
) -> tuple[MiaIndexerQueryRecords, mx.array]:
    """Validated scalar wrapper for the generic/reference entry point."""
    return _run_installed_indexer_query_records(
        queries,
        weights,
        positions.astype(mx.int32),
        inv_freq,
        weight_scale=mx.array(float(weight_scale), dtype=mx.float32),
    )


def _run_installed_indexer_query_records(
    queries: mx.array,
    weights: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    *,
    weight_scale: mx.array,
) -> tuple[MiaIndexerQueryRecords, mx.array]:
    """Direct query finalizer with the construction-bound scalar operand."""
    batch, query_count = (int(dim) for dim in queries.shape[:2])
    records, scaled_weights = _query_rope_quant_kernel()(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(weights),
            mx.contiguous(positions),
            mx.contiguous(inv_freq),
            weight_scale,
        ],
        template=[("T", queries.dtype)],
        grid=(query_count * INDEXER_HEADS * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch, query_count, INDEXER_HEADS, INDEXER_RECORD_BYTES),
            (batch, query_count, INDEXER_HEADS),
        ],
        output_dtypes=[mx.uint8, mx.float32],
    )
    return MiaIndexerQueryRecords(records), scaled_weights


def install_indexer_query_records(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    weight_scale: float,
):
    observed = (int(heads), int(head_dim), int(rope_dim))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM, 64)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia indexer Q geometry: {observed} != {expected}"
        )
    _query_rope_quant_kernel()
    installed_scale = mx.array(float(weight_scale), dtype=mx.float32)
    mx.eval(installed_scale)
    return partial(
        _run_installed_indexer_query_records,
        weight_scale=installed_scale,
    )


def _decode_e4m3(raw_bytes: mx.array) -> mx.array:
    raw = raw_bytes.astype(mx.uint32)
    negative = (raw & 0x80) != 0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    subnormal = mantissa.astype(mx.float32) * (2.0**-9)
    normal = (1.0 + mantissa.astype(mx.float32) / 8.0) * mx.power(
        mx.array(2.0, dtype=mx.float32), exponent.astype(mx.float32) - 7.0
    )
    magnitude = mx.where(exponent == 0, subnormal, normal)
    return mx.where(negative, -magnitude, magnitude)


def decode_indexer132(records: mx.array) -> mx.array:
    if records.dtype != mx.uint8 or int(records.shape[-1]) != INDEXER_RECORD_BYTES:
        raise ValueError("Mia indexer records must end in 132 uint8 bytes")
    scales = mx.contiguous(records[..., 128:132]).view(mx.float32)
    return _decode_e4m3(records[..., :128]) * scales


@dataclass(frozen=True)
class PagedMiaIndexerRecords:
    records: mx.array
    block_table: mx.array
    length: int
    block_size: int
    record_bytes: int = INDEXER_RECORD_BYTES

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, int(self.length), INDEXER_HEAD_DIM)


@dataclass(frozen=True)
class MiaTopKSelection:
    """Compact Mia sparse-indexer interchange consumed by sparse MLA."""

    indices: mx.array
    lengths: mx.array


@dataclass(frozen=True)
class MiaIndexerQueryRecords:
    """Post-RoPE FP8 query records produced by the fused Mia Q boundary."""

    records: mx.array

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.records.shape[0]),
            int(self.records.shape[1]),
            INDEXER_HEADS,
            INDEXER_HEAD_DIM,
        )


@dataclass(frozen=True)
class MiaIndexerWorkspace:
    """Construction-owned seed storage shared by every ratio-4 layer.

    MLX Metal kernels return functional output arrays, so candidate buffers are
    allocator-owned results.  The large, repeatedly identical carry seeds are
    different: they are allocated once for the installed launcher envelope and
    sliced by the phase route.  ``sentinel`` is the fixed ratio-4 capacity and
    is therefore outside every live logical row range.
    """

    max_query_rows: int
    topk: int
    sentinel: int
    empty_scores: mx.array
    empty_indices: mx.array

    @classmethod
    def allocate(
        cls,
        *,
        max_query_rows: int,
        topk: int,
        sentinel: int,
    ) -> "MiaIndexerWorkspace":
        max_query_rows = int(max_query_rows)
        topk = int(topk)
        sentinel = int(sentinel)
        if max_query_rows <= 0 or topk <= 0 or sentinel <= 0:
            raise ValueError("Mia indexer workspace geometry must be positive")
        shape = (1, max_query_rows, topk)
        return cls(
            max_query_rows=max_query_rows,
            topk=topk,
            sentinel=sentinel,
            empty_scores=mx.full(shape, -float("inf"), dtype=mx.float32),
            empty_indices=mx.full(shape, sentinel, dtype=mx.int32),
        )

    def seeds(self, query_count: int) -> tuple[mx.array, mx.array]:
        stop = int(query_count)
        return self.empty_scores[:, :stop], self.empty_indices[:, :stop]


class PagedMiaIndexerRows:
    """Fixed pages for the 132-byte FP8+scale indexer records Mia uses."""

    mode = "fp8_e4m3_ue8m0_scale132_paged"
    record_bytes = INDEXER_RECORD_BYTES

    def __init__(self, *, capacity_rows: int, block_size: int = 64) -> None:
        capacity_rows = int(capacity_rows)
        if capacity_rows <= 0:
            raise ValueError("paged Mia indexer capacity_rows must be positive")
        self._capacity_rows = capacity_rows
        plan = PagedCachePlan.contiguous(
            block_size=int(block_size),
            num_blocks=math.ceil(capacity_rows / int(block_size)),
            array_names=("records",),
        )
        self._pool = PagedCachePool(plan)
        self._pages = self._pool.bind(
            "records", row_shape=(self.record_bytes,), dtype=mx.uint8
        )

    def __len__(self) -> int:
        return int(self._pool.offset)

    @property
    def capacity(self) -> int:
        return self._capacity_rows

    @property
    def pages(self) -> mx.array:
        return self._pages

    @property
    def block_table(self) -> mx.array:
        return self._pool.block_table

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return (1, len(self), INDEXER_HEAD_DIM)

    @property
    def paged_records(self) -> PagedMiaIndexerRecords:
        return PagedMiaIndexerRecords(
            records=self.pages,
            block_table=self.block_table,
            length=len(self),
            block_size=self.block_size,
        )

    @property
    def records(self) -> mx.array:
        return self._pool.active("records")[None]

    @property
    def state(self):
        return self.pages, self.block_table, len(self)

    def append(self, rows: mx.array) -> None:
        if rows.ndim != 3 or tuple(int(dim) for dim in rows.shape[::2]) != (1, 128):
            raise ValueError("paged Mia indexer rows must be [1, rows, 128]")
        self.append_records(_pack_indexer132(rows))

    def append_records(self, records: mx.array) -> None:
        """Insert records already finalized by the fused Mia compressor."""
        if (
            records.dtype != mx.uint8
            or records.ndim != 3
            or tuple(int(dim) for dim in records.shape[:1]) != (1,)
            or int(records.shape[-1]) != self.record_bytes
        ):
            raise ValueError(
                "paged Mia indexer records must be [1, rows, 132] uint8"
            )
        count = int(records.shape[1])
        if len(self) + count > self.capacity:
            raise ValueError(
                f"paged Mia indexer capacity exceeded: {len(self) + count} "
                f"> {self.capacity}"
            )
        self._append_installed_records(records)

    def _append_installed_records(self, records: mx.array) -> None:
        """Insert records emitted by the installed Mia132 finalizer."""
        self._pool._write_installed_tail(
            {"records": records[0]},
            count=int(records.shape[1]),
        )

    def decode(self) -> mx.array:
        return decode_indexer132(self.records)

    def truncate(self, length: int) -> None:
        length = max(0, int(length))
        if length < len(self):
            self._pool.truncate(length)

    def clear(self) -> None:
        self._pool.clear()

    def replace_state(self, state) -> None:
        if state is None:
            self.clear()
            return
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("invalid paged Mia indexer state")
        pages, block_table, length = state
        if tuple(int(value) for value in block_table.shape) != (
            self._pool.num_blocks,
        ):
            raise ValueError("paged Mia indexer block table shape changed")
        self._pool.block_table = block_table
        self._pool.replace_state({"records": pages}, int(length))
        self._pages = self._pool.buffer("records")


@lru_cache(maxsize=1)
def _score_kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint row = group % uint(n_rows);
        uint query = group / uint(n_rows);
        uint logical_row = uint(row_start) + row;
        uint logical_block = logical_row / uint(block_size);
        uint row_in_block = logical_row - logical_block * uint(block_size);
        uint physical_block = uint(block_table[logical_block]);
        uint physical_row = physical_block * uint(block_size) + row_in_block;
        const device uchar* k_record = k_records + size_t(physical_row) * 132u;
        uint k_scale_bits = uint(k_record[128u])
            | (uint(k_record[129u]) << 8u)
            | (uint(k_record[130u]) << 16u)
            | (uint(k_record[131u]) << 24u);
        float k_scale = as_type<float>(k_scale_bits);

        float score = 0.0f;
        for (uint head = 0u; head < 64u; ++head) {
            const device uchar* q_record = q_records
                + (size_t(query) * 64u + size_t(head)) * 132u;
            uint q_scale_bits = uint(q_record[128u])
                | (uint(q_record[129u]) << 8u)
                | (uint(q_record[130u]) << 16u)
                | (uint(q_record[131u]) << 24u);
            float q_scale = as_type<float>(q_scale_bits);
            float partial = 0.0f;
            uint dim0 = lane * 4u;
            for (uint element = 0u; element < 4u; ++element) {
                uint dim = dim0 + element;
                partial += mtplx_indexer_e4m3_decode(q_record[dim])
                    * mtplx_indexer_e4m3_decode(k_record[dim]);
            }
            float dot = simd_sum(partial) * q_scale * k_scale;
            if (lane == 0u) {
                score += max(dot, 0.0f)
                    * weights[size_t(query) * 64u + size_t(head)];
            }
        }
        if (lane == 0u) {
            scores[size_t(query) * size_t(n_rows) + size_t(row)] = score;
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_paged_indexer_scores",
        input_names=[
            "q_records",
            "weights",
            "k_records",
            "block_table",
            "row_start",
            "n_rows",
            "block_size",
        ],
        output_names=["scores"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _tiled_score_kernel():
    header = _FP8_HEADER + r"""
        constant constexpr int MTPLX_INDEX_Q_TILE = 16;
        constant constexpr int MTPLX_INDEX_K_TILE = 16;
        constant constexpr int MTPLX_INDEX_DIM = 128;
        constant constexpr int MTPLX_INDEX_HEADS = 64;
        constant constexpr int MTPLX_INDEX_THREADS = 64;
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint tile = threadgroup_position_in_grid.x;
        uint k_tiles = (uint(n_rows) + MTPLX_INDEX_K_TILE - 1u)
            / MTPLX_INDEX_K_TILE;
        uint q_tile = tile / k_tiles;
        uint k_tile = tile - q_tile * k_tiles;
        uint q0 = q_tile * MTPLX_INDEX_Q_TILE;
        uint k0 = k_tile * MTPLX_INDEX_K_TILE;

        threadgroup half q_values[
            MTPLX_INDEX_Q_TILE * MTPLX_INDEX_DIM
        ];
        threadgroup half k_values[
            MTPLX_INDEX_DIM * MTPLX_INDEX_K_TILE
        ];
        threadgroup float q_scales[MTPLX_INDEX_Q_TILE];
        threadgroup float k_scales[MTPLX_INDEX_K_TILE];
        threadgroup float head_dot[
            MTPLX_INDEX_Q_TILE * MTPLX_INDEX_K_TILE
        ];
        threadgroup float scores[
            MTPLX_INDEX_Q_TILE * MTPLX_INDEX_K_TILE
        ];

        for (uint index = tid;
             index < MTPLX_INDEX_Q_TILE * MTPLX_INDEX_K_TILE;
             index += MTPLX_INDEX_THREADS) {
            scores[index] = 0.0f;
        }
        if (tid < MTPLX_INDEX_K_TILE) {
            uint local_k = tid;
            uint logical_row = uint(row_start) + k0 + local_k;
            if (k0 + local_k < uint(n_rows)) {
                uint logical_block = logical_row / uint(block_size);
                uint row_in_block = logical_row
                    - logical_block * uint(block_size);
                uint physical_block = uint(block_table[logical_block]);
                uint physical_row = physical_block * uint(block_size)
                    + row_in_block;
                const device uchar* record = k_records
                    + size_t(physical_row) * 132u;
                k_scales[local_k] = mtplx_indexer_record_scale(record);
            } else {
                k_scales[local_k] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = tid;
             index < MTPLX_INDEX_DIM * MTPLX_INDEX_K_TILE;
             index += MTPLX_INDEX_THREADS) {
            uint dim = index / MTPLX_INDEX_K_TILE;
            uint local_k = index - dim * MTPLX_INDEX_K_TILE;
            uint logical_row = uint(row_start) + k0 + local_k;
            half value = half(0.0f);
            if (k0 + local_k < uint(n_rows)) {
                uint logical_block = logical_row / uint(block_size);
                uint row_in_block = logical_row
                    - logical_block * uint(block_size);
                uint physical_block = uint(block_table[logical_block]);
                uint physical_row = physical_block * uint(block_size)
                    + row_in_block;
                const device uchar* record = k_records
                    + size_t(physical_row) * 132u;
                // Stage the exact E4M3 value.  Its power-of-two grid is
                // exactly representable in FP16; the row scales are applied
                // to the FP32 accumulator below, in the same order as the
                // source FP8 logits kernel.
                value = half(mtplx_indexer_e4m3_decode(record[dim]));
            }
            k_values[index] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint head = 0u; head < MTPLX_INDEX_HEADS; ++head) {
            if (tid < MTPLX_INDEX_Q_TILE) {
                uint local_q = tid;
                uint query = q0 + local_q;
                if (query < uint(n_queries)) {
                    const device uchar* record = q_records
                        + (size_t(query) * MTPLX_INDEX_HEADS + size_t(head))
                            * 132u;
                    q_scales[local_q] = mtplx_indexer_record_scale(record);
                } else {
                    q_scales[local_q] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint index = tid;
                 index < MTPLX_INDEX_Q_TILE * MTPLX_INDEX_DIM;
                 index += MTPLX_INDEX_THREADS) {
                uint local_q = index / MTPLX_INDEX_DIM;
                uint dim = index - local_q * MTPLX_INDEX_DIM;
                uint query = q0 + local_q;
                half value = half(0.0f);
                if (query < uint(n_queries)) {
                    const device uchar* record = q_records
                        + (size_t(query) * MTPLX_INDEX_HEADS + size_t(head))
                            * 132u;
                    value = half(mtplx_indexer_e4m3_decode(record[dim]));
                }
                q_values[index] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<half, 8, 8> q_matrix;
            simdgroup_matrix<half, 8, 8> k_left;
            simdgroup_matrix<half, 8, 8> k_right;
            simdgroup_matrix<float, 8, 8> dot_left =
                simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_matrix<float, 8, 8> dot_right =
                simdgroup_matrix<float, 8, 8>(0.0f);
            uint q_base = simd_group * 8u * MTPLX_INDEX_DIM;
            for (uint dim0 = 0u; dim0 < MTPLX_INDEX_DIM; dim0 += 8u) {
                simdgroup_load(
                    q_matrix,
                    q_values + q_base + dim0,
                    MTPLX_INDEX_DIM
                );
                simdgroup_load(
                    k_left,
                    k_values + dim0 * MTPLX_INDEX_K_TILE,
                    MTPLX_INDEX_K_TILE
                );
                simdgroup_load(
                    k_right,
                    k_values + dim0 * MTPLX_INDEX_K_TILE + 8u,
                    MTPLX_INDEX_K_TILE
                );
                simdgroup_multiply_accumulate(
                    dot_left, q_matrix, k_left, dot_left
                );
                simdgroup_multiply_accumulate(
                    dot_right, q_matrix, k_right, dot_right
                );
            }
            uint score_base = simd_group * 8u * MTPLX_INDEX_K_TILE;
            simdgroup_store(
                dot_left,
                head_dot + score_base,
                MTPLX_INDEX_K_TILE
            );
            simdgroup_store(
                dot_right,
                head_dot + score_base + 8u,
                MTPLX_INDEX_K_TILE
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint index = tid;
                 index < MTPLX_INDEX_Q_TILE * MTPLX_INDEX_K_TILE;
                 index += MTPLX_INDEX_THREADS) {
                uint local_q = index / MTPLX_INDEX_K_TILE;
                uint local_k = index - local_q * MTPLX_INDEX_K_TILE;
                uint query = q0 + local_q;
                if (query < uint(n_queries)
                    && k0 + local_k < uint(n_rows)) {
                    float dot = head_dot[index]
                        * q_scales[local_q] * k_scales[local_k];
                    scores[index] += max(dot, 0.0f)
                        * weights[
                            size_t(query) * MTPLX_INDEX_HEADS + size_t(head)
                        ];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint index = tid;
             index < MTPLX_INDEX_Q_TILE * MTPLX_INDEX_K_TILE;
             index += MTPLX_INDEX_THREADS) {
            uint local_q = index / MTPLX_INDEX_K_TILE;
            uint local_k = index - local_q * MTPLX_INDEX_K_TILE;
            uint query = q0 + local_q;
            uint row = k0 + local_k;
            if (query < uint(n_queries) && row < uint(n_rows)) {
                output[size_t(query) * size_t(n_rows) + size_t(row)] =
                    scores[index];
            }
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_paged_indexer_tiled_scores",
        input_names=[
            "q_records",
            "weights",
            "k_records",
            "block_table",
            "row_start",
            "n_rows",
            "block_size",
            "n_queries",
        ],
        output_names=["output"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _radix_fold_kernel():
    """Exact SparkInfer-style four-pass MSD radix fold into top-512."""
    header = r"""
        using namespace metal;

        inline uint mtplx_indexer_ordered_key(float value) {
            uint bits = as_type<uint>(value);
            return (bits & 0x80000000u) != 0u
                ? ~bits
                : bits | 0x80000000u;
        }
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint query = threadgroup_position_in_grid.x;
        constexpr uint TOPK = 512u;
        constexpr uint THREADS = 256u;

        threadgroup atomic_uint histogram[256];
        threadgroup atomic_uint output_counter;
        threadgroup uint prefix;
        threadgroup uint remaining;
        threadgroup uint pivot;

        int causal = causal_lengths[query];
        int available = causal - int(row_start);
        uint local_count = uint(max(0, min(available, int(n_local))));
        uint previous_count = uint(has_carry)
            * uint(min(min(causal, int(row_start)), int(TOPK)));
        uint total = local_count + previous_count;
        device float* output_value_row = output_values
            + size_t(query) * TOPK;
        device int* output_index_row = output_indices
            + size_t(query) * TOPK;
        const device float* score_row = scores
            + size_t(query) * size_t(n_local);
        const device int* score_index_row = score_indices
            + size_t(query) * size_t(n_local);
        const device float* carry_value_row = carry_values
            + size_t(query) * TOPK;
        const device int* carry_index_row = carry_indices
            + size_t(query) * TOPK;

        for (uint i = tid; i < TOPK; i += THREADS) {
            output_value_row[i] = -INFINITY;
            output_index_row[i] = int(sentinel);
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if (total <= TOPK) {
            for (uint i = tid; i < total; i += THREADS) {
                bool local = i < local_count;
                output_value_row[i] = local
                    ? score_row[i]
                    : carry_value_row[i - local_count];
                output_index_row[i] = local
                    ? (use_score_indices
                        ? score_index_row[i]
                        : int(row_start) + int(i))
                    : carry_index_row[i - local_count];
            }
            return;
        }

        if (tid == 0u) {
            prefix = 0u;
            remaining = TOPK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint round = 0u; round < 4u; ++round) {
            if (tid < 256u) {
                atomic_store_explicit(
                    &histogram[tid], 0u, memory_order_relaxed
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shift = 24u - round * 8u;
            uint locked = prefix;
            uint mask = round == 0u
                ? 0u
                : 0xffffffffu << (32u - round * 8u);
            for (uint i = tid; i < total; i += THREADS) {
                float value = i < local_count
                    ? score_row[i]
                    : carry_value_row[i - local_count];
                uint key = mtplx_indexer_ordered_key(value);
                if (round == 0u || (key & mask) == locked) {
                    uint bucket = (key >> shift) & 0xffu;
                    atomic_fetch_add_explicit(
                        &histogram[bucket], 1u, memory_order_relaxed
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint need = remaining;
                uint chosen = 0u;
                for (int bucket = 255; bucket >= 0; --bucket) {
                    uint count = atomic_load_explicit(
                        &histogram[uint(bucket)], memory_order_relaxed
                    );
                    if (need <= count) {
                        chosen = uint(bucket);
                        break;
                    }
                    need -= count;
                }
                prefix = locked | (chosen << shift);
                remaining = need;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            pivot = prefix;
            atomic_store_explicit(
                &output_counter, 0u, memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint equality = 0u; equality < 2u; ++equality) {
            for (uint i = tid; i < total; i += THREADS) {
                bool local = i < local_count;
                uint carry_slot = i - local_count;
                float value = local
                    ? score_row[i]
                    : carry_value_row[carry_slot];
                uint key = mtplx_indexer_ordered_key(value);
                bool selected = equality == 0u ? key > pivot : key == pivot;
                if (selected) {
                    uint position = atomic_fetch_add_explicit(
                        &output_counter, 1u, memory_order_relaxed
                    );
                    if (position < TOPK) {
                        output_value_row[position] = value;
                        output_index_row[position] = local
                            ? (use_score_indices
                                ? score_index_row[i]
                                : int(row_start) + int(i))
                            : carry_index_row[carry_slot];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_indexer_radix_top512_fold",
        input_names=[
            "scores",
            "score_indices",
            "carry_values",
            "carry_indices",
            "causal_lengths",
            "row_start",
            "n_local",
            "use_score_indices",
            "has_carry",
            "sentinel",
        ],
        output_names=["output_values", "output_indices"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_radix_fold(
    scores: mx.array,
    carry_values: mx.array,
    carry_indices: mx.array,
    causal_lengths: mx.array,
    *,
    row_start: int,
    score_indices: mx.array | None = None,
    has_carry: bool,
    sentinel: int,
) -> tuple[mx.array, mx.array]:
    query_count = int(scores.shape[1])
    n_local = int(scores.shape[2])
    explicit_indices = score_indices is not None
    if score_indices is None:
        score_indices = carry_indices
    return tuple(
        _radix_fold_kernel()(
            inputs=[
                mx.contiguous(scores),
                mx.contiguous(score_indices),
                mx.contiguous(carry_values),
                mx.contiguous(carry_indices),
                mx.contiguous(causal_lengths),
                int(row_start),
                n_local,
                bool(explicit_indices),
                bool(has_carry),
                int(sentinel),
            ],
            template=[],
            grid=(query_count * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, query_count, INDEXER_TOPK)] * 2,
            output_dtypes=[mx.float32, mx.int32],
        )
    )


@lru_cache(maxsize=1)
def _fused_decode_candidates_kernel():
    """Fused paged FP8 score plus exact local radix top-512."""
    header = _FP8_HEADER + r"""
        inline uint mtplx_indexer_ordered_key(float value) {
            uint bits = as_type<uint>(value);
            return (bits & 0x80000000u) != 0u
                ? ~bits
                : bits | 0x80000000u;
        }
    """
    source = r"""
        uint tid = thread_position_in_threadgroup.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint query = group / uint(n_slices);
        uint slice = group - query * uint(n_slices);
        constexpr uint TOPK = 512u;
        constexpr uint SLICE_ROWS = 4096u;
        constexpr uint THREADS = 256u;
        constexpr uint SIMD_GROUPS = 8u;

        threadgroup float local_scores[SLICE_ROWS];
        threadgroup atomic_uint histogram[256];
        threadgroup atomic_uint output_counter;
        threadgroup uint prefix;
        threadgroup uint remaining;
        threadgroup uint pivot;

        int causal = causal_lengths[query];
        uint row_start = slice * SLICE_ROWS;
        uint local_count = uint(max(
            0,
            min(causal - int(row_start), int(SLICE_ROWS))
        ));
        device float* output_value_row = candidate_values
            + size_t(group) * TOPK;
        device int* output_index_row = candidate_indices
            + size_t(group) * TOPK;

        for (uint i = tid; i < TOPK; i += THREADS) {
            output_value_row[i] = -INFINITY;
            output_index_row[i] = int(sentinel);
        }

        for (uint local_row = sg;
             local_row < local_count;
             local_row += SIMD_GROUPS) {
            uint logical_row = row_start + local_row;
            uint logical_block = logical_row / uint(block_size);
            uint row_in_block = logical_row
                - logical_block * uint(block_size);
            uint physical_block = uint(block_table[logical_block]);
            uint physical_row = physical_block * uint(block_size) + row_in_block;
            const device uchar* k_record = k_records
                + size_t(physical_row) * 132u;
            float k_scale = mtplx_indexer_record_scale(k_record);
            float score = 0.0f;
            for (uint head = 0u; head < 64u; ++head) {
                const device uchar* q_record = q_records
                    + (size_t(query) * 64u + size_t(head)) * 132u;
                float q_scale = mtplx_indexer_record_scale(q_record);
                float partial = 0.0f;
                uint dim0 = lane * 4u;
                for (uint element = 0u; element < 4u; ++element) {
                    uint dim = dim0 + element;
                    partial += mtplx_indexer_e4m3_decode(q_record[dim])
                        * mtplx_indexer_e4m3_decode(k_record[dim]);
                }
                float dot = simd_sum(partial) * q_scale * k_scale;
                if (lane == 0u) {
                    score += max(dot, 0.0f)
                        * weights[size_t(query) * 64u + size_t(head)];
                }
            }
            if (lane == 0u) {
                local_scores[local_row] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if (local_count <= TOPK) {
            for (uint i = tid; i < local_count; i += THREADS) {
                output_value_row[i] = local_scores[i];
                output_index_row[i] = int(row_start + i);
            }
            return;
        }

        if (tid == 0u) {
            prefix = 0u;
            remaining = TOPK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint round = 0u; round < 4u; ++round) {
            atomic_store_explicit(
                &histogram[tid], 0u, memory_order_relaxed
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint shift = 24u - round * 8u;
            uint locked = prefix;
            uint mask = round == 0u
                ? 0u
                : 0xffffffffu << (32u - round * 8u);
            for (uint i = tid; i < local_count; i += THREADS) {
                uint key = mtplx_indexer_ordered_key(local_scores[i]);
                if (round == 0u || (key & mask) == locked) {
                    atomic_fetch_add_explicit(
                        &histogram[(key >> shift) & 0xffu],
                        1u,
                        memory_order_relaxed
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                uint need = remaining;
                uint chosen = 0u;
                for (int bucket = 255; bucket >= 0; --bucket) {
                    uint count = atomic_load_explicit(
                        &histogram[uint(bucket)], memory_order_relaxed
                    );
                    if (need <= count) {
                        chosen = uint(bucket);
                        break;
                    }
                    need -= count;
                }
                prefix = locked | (chosen << shift);
                remaining = need;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            pivot = prefix;
            atomic_store_explicit(
                &output_counter, 0u, memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint equality = 0u; equality < 2u; ++equality) {
            for (uint i = tid; i < local_count; i += THREADS) {
                float value = local_scores[i];
                uint key = mtplx_indexer_ordered_key(value);
                bool selected = equality == 0u ? key > pivot : key == pivot;
                if (selected) {
                    uint position = atomic_fetch_add_explicit(
                        &output_counter, 1u, memory_order_relaxed
                    );
                    if (position < TOPK) {
                        output_value_row[position] = value;
                        output_index_row[position] = int(row_start + i);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_fused_decode_indexer_top512",
        input_names=[
            "q_records",
            "weights",
            "k_records",
            "block_table",
            "causal_lengths",
            "n_slices",
            "block_size",
            "sentinel",
        ],
        output_names=["candidate_values", "candidate_indices"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_fused_decode_candidates(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    causal_lengths: mx.array,
) -> tuple[mx.array, mx.array]:
    query_count = int(q_records.shape[1])
    n_slices = max(
        1,
        (int(rows.length) + INDEXER_DECODE_SLICE_ROWS - 1)
        // INDEXER_DECODE_SLICE_ROWS,
    )
    return tuple(
        _fused_decode_candidates_kernel()(
            inputs=[
                mx.contiguous(q_records),
                mx.contiguous(weights),
                rows.records,
                rows.block_table,
                mx.contiguous(causal_lengths),
                n_slices,
                int(rows.block_size),
                int(rows.length),
            ],
            template=[],
            grid=(query_count * n_slices * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, query_count, n_slices, INDEXER_TOPK)] * 2,
            output_dtypes=[mx.float32, mx.int32],
        )
    )


def _run_paged_indexer_score_slice_oracle(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    row_start: int,
    row_count: int,
) -> mx.array:
    batch, query_count = (int(dim) for dim in q_records.shape[:2])
    (scores,) = _score_kernel()(
        inputs=[
            q_records,
            weights,
            rows.records,
            rows.block_table,
            int(row_start),
            int(row_count),
            int(rows.block_size),
        ],
        template=[],
        grid=(batch * query_count * int(row_count) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, query_count, int(row_count))],
        output_dtypes=[mx.float32],
    )
    return scores


def _run_paged_indexer_score_slice(
    q_records: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
    row_start: int,
    row_count: int,
) -> mx.array:
    batch, query_count = (int(dim) for dim in q_records.shape[:2])
    q_tiles = (query_count + 15) // 16
    k_tiles = (int(row_count) + 15) // 16
    (scores,) = _tiled_score_kernel()(
        inputs=[
            q_records,
            weights,
            rows.records,
            rows.block_table,
            int(row_start),
            int(row_count),
            int(rows.block_size),
            int(query_count),
        ],
        template=[],
        grid=(q_tiles * k_tiles * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(batch, query_count, int(row_count))],
        output_dtypes=[mx.float32],
    )
    return scores


def _run_paged_indexer_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    return _run_paged_indexer_score_slice_oracle(
        _pack_indexer132(queries),
        weights,
        rows,
        0,
        int(rows.length),
    )


def _run_paged_indexer_tiled_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    return _run_paged_indexer_score_slice(
        _pack_indexer132(queries),
        weights,
        rows,
        0,
        int(rows.length),
    )


def _installed_query_records(
    queries: mx.array | MiaIndexerQueryRecords,
) -> tuple[mx.array, int, int]:
    if isinstance(queries, MiaIndexerQueryRecords):
        return queries.records, int(queries.records.shape[0]), int(
            queries.records.shape[1]
        )
    records = _pack_indexer132(queries)
    return records, int(queries.shape[0]), int(queries.shape[1])


def _run_paged_indexer_topk(
    queries: mx.array | MiaIndexerQueryRecords,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int = INDEXER_PREFILL_SCORE_CHUNK_ROWS,
) -> MiaTopKSelection:
    """Stream fixed-width score tiles into Mia's compact top-k buffer.

    SparkInfer's paged prefill indexer scores one bounded K supertile at a time
    and folds each tile's candidates into a fixed ``[queries, topk]`` carry.
    Metal keeps the scorer's 16x16 physical tile, folds one source-sized 32K K
    supertile at a time, and applies the source's exact four-pass radix select.
    No whole-context score, boolean selection, or generic sort enters the graph.
    """
    q_records, _, query_count = _installed_query_records(queries)
    return _run_paged_indexer_records_topk(
        q_records,
        weights,
        positions,
        rows,
        topk=topk,
        compress_ratio=compress_ratio,
        workspace=workspace,
        score_chunk_rows=score_chunk_rows,
        query_count=query_count,
    )


def _run_paged_indexer_records_topk(
    q_records: mx.array,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int,
    query_count: int,
) -> MiaTopKSelection:
    """Direct installed prefill selector over already-qualified Q records."""
    n_rows = int(rows.length)
    topk = int(topk)
    score_chunk_rows = int(score_chunk_rows)
    causal_lengths = mx.minimum(
        (positions + 1) // int(compress_ratio),
        n_rows,
    )[None]
    output_lengths = mx.minimum(causal_lengths, topk).astype(mx.int32)
    carry_scores, carry_indices = workspace.seeds(query_count)
    has_carry = False

    for row_start in range(0, n_rows, score_chunk_rows):
        row_count = min(score_chunk_rows, n_rows - row_start)
        scores = _run_paged_indexer_score_slice(
            q_records,
            weights,
            rows,
            row_start,
            row_count,
        )
        carry_scores, carry_indices = _run_radix_fold(
            scores,
            carry_scores,
            carry_indices,
            causal_lengths,
            row_start=row_start,
            has_carry=has_carry,
            sentinel=n_rows,
        )
        has_carry = True

    return MiaTopKSelection(
        indices=carry_indices,
        lengths=output_lengths,
    )


def _run_paged_indexer_decode_topk(
    queries: mx.array | MiaIndexerQueryRecords,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
) -> MiaTopKSelection:
    """Fused score/local-select followed by one bounded candidate merge."""
    q_records, _, query_count = _installed_query_records(queries)
    return _run_paged_indexer_records_decode_topk(
        q_records,
        weights,
        positions,
        rows,
        topk=topk,
        compress_ratio=compress_ratio,
        workspace=workspace,
        query_count=query_count,
    )


def _run_paged_indexer_records_decode_topk(
    q_records: mx.array,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    query_count: int,
) -> MiaTopKSelection:
    """Direct installed decode selector over already-qualified Q records."""
    n_rows = int(rows.length)
    causal_lengths = mx.minimum(
        (positions + 1) // int(compress_ratio),
        n_rows,
    )[None]
    output_lengths = mx.minimum(causal_lengths, int(topk)).astype(mx.int32)
    candidate_values, candidate_indices = _run_fused_decode_candidates(
        q_records,
        weights,
        rows,
        causal_lengths,
    )
    candidate_width = int(candidate_values.shape[2]) * int(topk)
    values = candidate_values.reshape(1, query_count, candidate_width)
    indices = candidate_indices.reshape(1, query_count, candidate_width)
    if candidate_width > int(topk):
        empty_values, empty_indices = workspace.seeds(query_count)
        merge_lengths = mx.full(
            (1, query_count), candidate_width, dtype=mx.int32
        )
        _, indices = _run_radix_fold(
            values,
            empty_values,
            empty_indices,
            merge_lengths,
            row_start=0,
            score_indices=indices,
            has_carry=False,
            sentinel=n_rows,
        )
    return MiaTopKSelection(indices=indices, lengths=output_lengths)


def _run_paged_indexer_phase_topk(
    queries: mx.array | MiaIndexerQueryRecords,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int,
) -> MiaTopKSelection:
    if current_attention_phase() == "prefill":
        return _run_paged_indexer_topk(
            queries,
            weights,
            positions,
            rows,
            topk=topk,
            compress_ratio=compress_ratio,
            workspace=workspace,
            score_chunk_rows=score_chunk_rows,
        )
    return _run_paged_indexer_decode_topk(
        queries,
        weights,
        positions,
        rows,
        topk=topk,
        compress_ratio=compress_ratio,
        workspace=workspace,
    )


def _run_installed_paged_indexer_phase_topk(
    queries: MiaIndexerQueryRecords,
    weights: mx.array,
    positions: mx.array,
    rows: PagedMiaIndexerRecords,
    *,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
    score_chunk_rows: int,
) -> MiaTopKSelection:
    """Phase-only route for the installed Mia record type."""
    q_records = queries.records
    query_count = int(q_records.shape[1])
    if current_attention_phase() == "prefill":
        return _run_paged_indexer_records_topk(
            q_records,
            weights,
            positions,
            rows,
            topk=topk,
            compress_ratio=compress_ratio,
            workspace=workspace,
            score_chunk_rows=score_chunk_rows,
            query_count=query_count,
        )
    return _run_paged_indexer_records_decode_topk(
        q_records,
        weights,
        positions,
        rows,
        topk=topk,
        compress_ratio=compress_ratio,
        workspace=workspace,
        query_count=query_count,
    )


def paged_indexer_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    """Validated construction/oracle boundary for the direct paged indexer."""
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer requires Metal")
    if tuple(int(dim) for dim in queries.shape[:1]) != (1,) or tuple(
        int(dim) for dim in queries.shape[2:]
    ) != (INDEXER_HEADS, INDEXER_HEAD_DIM):
        raise ValueError("Mia indexer queries must be [1, rows, 64, 128]")
    if tuple(int(dim) for dim in weights.shape) != tuple(
        int(dim) for dim in queries.shape[:3]
    ):
        raise ValueError("Mia indexer weights must be [1, rows, 64]")
    if rows.record_bytes != INDEXER_RECORD_BYTES or rows.records.dtype != mx.uint8:
        raise ValueError("invalid Mia paged indexer records")
    return _run_paged_indexer_scores(queries, weights, rows)


def paged_indexer_tiled_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    """Validated oracle boundary for Mia's bounded tiled prefill scorer."""
    if not mx.metal.is_available():
        raise RuntimeError("Mia tiled paged indexer requires Metal")
    if tuple(int(dim) for dim in queries.shape[:1]) != (1,) or tuple(
        int(dim) for dim in queries.shape[2:]
    ) != (INDEXER_HEADS, INDEXER_HEAD_DIM):
        raise ValueError("Mia indexer queries must be [1, rows, 64, 128]")
    if tuple(int(dim) for dim in weights.shape) != tuple(
        int(dim) for dim in queries.shape[:3]
    ):
        raise ValueError("Mia indexer weights must be [1, rows, 64]")
    if rows.record_bytes != INDEXER_RECORD_BYTES or rows.records.dtype != mx.uint8:
        raise ValueError("invalid Mia paged indexer records")
    return _run_paged_indexer_tiled_scores(queries, weights, rows)


def install_paged_indexer_scores(*, heads: int, head_dim: int):
    observed = (int(heads), int(head_dim))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM)
    if observed != expected:
        raise ValueError(f"unsupported Mia paged indexer geometry: {observed} != {expected}")
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer installation requires Metal")
    _pack_kernel()
    _score_kernel()
    return _run_paged_indexer_scores


def install_paged_indexer_topk(
    *,
    heads: int,
    head_dim: int,
    topk: int,
    compress_ratio: int,
    workspace: MiaIndexerWorkspace,
):
    observed = (int(heads), int(head_dim), int(topk), int(compress_ratio))
    expected = (INDEXER_HEADS, INDEXER_HEAD_DIM, INDEXER_TOPK, 4)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia paged indexer top-k geometry: {observed} != {expected}"
        )
    if (
        int(workspace.topk) != INDEXER_TOPK
        or int(workspace.max_query_rows) <= 0
        or int(workspace.sentinel) <= 0
    ):
        raise ValueError("the Mia paged indexer workspace geometry is invalid")
    if not mx.metal.is_available():
        raise RuntimeError("Mia paged indexer top-k installation requires Metal")
    _pack_kernel()
    _tiled_score_kernel()
    _radix_fold_kernel()
    _fused_decode_candidates_kernel()

    return partial(
        _run_installed_paged_indexer_phase_topk,
        topk=int(topk),
        compress_ratio=int(compress_ratio),
        workspace=workspace,
        score_chunk_rows=INDEXER_PREFILL_SCORE_CHUNK_ROWS,
    )
