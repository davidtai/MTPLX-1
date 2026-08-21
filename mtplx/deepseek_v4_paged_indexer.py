"""Mia-compatible paged FP8 indexer cache for DeepSeek V4 ratio-4 layers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import mlx.core as mx

from mtplx.paged_cache import PagedCachePlan, PagedCachePool


INDEXER_HEADS = 64
INDEXER_HEAD_DIM = 128
INDEXER_RECORD_BYTES = 132


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
"""


@lru_cache(maxsize=1)
def _pack_kernel():
    source = r"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / 132u;
        uint byte = gid - row * 132u;
        const device T* source_row = rows + size_t(row) * 128u;
        device uchar* record = records + size_t(row) * 132u;

        float amax = 0.0f;
        for (uint dim = 0u; dim < 128u; ++dim) {
            amax = max(amax, abs(float(source_row[dim])));
        }
        float scale = exp2(ceil(log2(max(amax, 1.0e-4f) / 448.0f)));
        if (byte < 128u) {
            record[byte] = mtplx_indexer_e4m3_encode(
                float(source_row[byte]) / scale
            );
            return;
        }
        uint scale_bits = as_type<uint>(scale);
        record[byte] = uchar((scale_bits >> (8u * (byte - 128u))) & 0xffu);
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
        grid=(row_count * INDEXER_RECORD_BYTES, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*rows.shape[:-1], INDEXER_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]


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
        count = int(rows.shape[1])
        if len(self) + count > self.capacity:
            raise ValueError(
                f"paged Mia indexer capacity exceeded: {len(self) + count} "
                f"> {self.capacity}"
            )
        self._pool.write_tail({"records": _pack_indexer132(rows)[0]})

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
        uint logical_block = row / uint(block_size);
        uint row_in_block = row - logical_block * uint(block_size);
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
            "n_rows",
            "block_size",
        ],
        output_names=["scores"],
        header=_FP8_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_paged_indexer_scores(
    queries: mx.array,
    weights: mx.array,
    rows: PagedMiaIndexerRecords,
) -> mx.array:
    batch, query_count, heads, width = (int(dim) for dim in queries.shape)
    q_records = _pack_indexer132(queries)
    (scores,) = _score_kernel()(
        inputs=[
            q_records,
            weights,
            rows.records,
            rows.block_table,
            int(rows.length),
            int(rows.block_size),
        ],
        template=[],
        grid=(batch * query_count * int(rows.length) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, query_count, int(rows.length))],
        output_dtypes=[mx.float32],
    )
    return scores


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
