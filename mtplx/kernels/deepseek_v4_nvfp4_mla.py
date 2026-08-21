"""Direct sparse MLA over Mia DeepSeek-V4 ``stock432`` NVFP4 records."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_HEADS = 64
_HEAD_DIM = 512
_WINDOW = 128
_RECORD_BYTES = 432
_LANES = 32
_VALUES_PER_LANE = 16


_HEADER = f"""
    using namespace metal;

    constant constexpr int MTPLX_H = {_HEADS};
    constant constexpr int MTPLX_D = {_HEAD_DIM};
    constant constexpr int MTPLX_WINDOW = {_WINDOW};
    constant constexpr int MTPLX_RECORD = {_RECORD_BYTES};
    constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};

    inline float mtplx_dsv4_e4m3(uchar raw) {{
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        float magnitude = exponent == 0u
            ? float(mantissa) * 0.001953125f
            : (1.0f + float(mantissa) * 0.125f)
                * exp2(float(int(exponent) - 7));
        return (uint(raw) & 0x80u) != 0u ? -magnitude : magnitude;
    }}

    inline float mtplx_dsv4_e2m1(uchar raw) {{
        constexpr float values[8] = {{0.0f, 0.5f, 1.0f, 1.5f,
                                      2.0f, 3.0f, 4.0f, 6.0f}};
        float magnitude = values[uint(raw) & 0x07u];
        return (uint(raw) & 0x08u) != 0u ? -magnitude : magnitude;
    }}

    inline float mtplx_dsv4_latent(
        const device uchar* record,
        uint dim,
        float scale
    ) {{
        uchar packed = record[dim >> 1];
        uchar code = (dim & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
        return mtplx_dsv4_e2m1(code) * scale;
    }}

    inline void mtplx_dsv4_consume(
        const device uchar* record,
        thread const float* query,
        thread float* accumulator,
        thread float& running_max,
        thread float& running_sum,
        uint lane
    ) {{
        uint dim0 = lane * MTPLX_ELEMS;
        float latent_scale = mtplx_dsv4_e4m3(record[256u + lane]);
        const device bfloat* rope = reinterpret_cast<const device bfloat*>(
            record + 304u
        );
        float values[MTPLX_ELEMS];
        float partial = 0.0f;
        for (uint element = 0u; element < MTPLX_ELEMS; ++element) {{
            uint dim = dim0 + element;
            float value = mtplx_dsv4_latent(record, dim, latent_scale);
            float key = dim < 448u ? value : float(rope[dim - 448u]);
            values[element] = value;
            partial += query[element] * key;
        }}
        float score = simd_sum(partial);
        float next_max = max(running_max, score);
        float correction = fast::exp(running_max - next_max);
        float probability = fast::exp(score - next_max);
        running_max = next_max;
        running_sum = running_sum * correction + probability;
        for (uint element = 0u; element < MTPLX_ELEMS; ++element) {{
            accumulator[element] = accumulator[element] * correction
                + probability * values[element];
        }}
    }}
"""


@lru_cache(maxsize=1)
def _kernel():
    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint query_count = uint(n_queries);
        uint groups_per_batch = uint(MTPLX_H) * query_count;
        uint batch = group / groups_per_batch;
        uint within_batch = group - batch * groups_per_batch;
        uint head = within_batch / query_count;
        uint query_row = within_batch - head * query_count;

        size_t query_base = (
            (size_t(batch * uint(MTPLX_H) + head) * size_t(query_count)
             + size_t(query_row)) * size_t(MTPLX_D)
        );
        const device T* query_ptr = queries + query_base + lane * MTPLX_ELEMS;
        float query[MTPLX_ELEMS];
        float accumulator[MTPLX_ELEMS];
        for (uint element = 0u; element < MTPLX_ELEMS; ++element) {
            query[element] = float(scale) * float(query_ptr[element]);
            accumulator[element] = 0.0f;
        }

        // The learned sink is a logit in the denominator with a zero V row.
        float running_max = sinks[head];
        float running_sum = 1.0f;
        int query_position = query_positions[query_row];

        size_t window_batch = size_t(batch) * size_t(n_window_records);
        for (uint row = 0u; row < uint(n_window_records); ++row) {
            int absolute_position = int(window_start) + int(row);
            bool visible = absolute_position <= query_position
                && absolute_position > query_position - MTPLX_WINDOW;
            if (visible) {
                const device uchar* record = window_records
                    + (window_batch + size_t(row)) * size_t(MTPLX_RECORD);
                mtplx_dsv4_consume(
                    record,
                    query,
                    accumulator,
                    running_max,
                    running_sum,
                    lane
                );
            }
        }

        uint length = uint(compressed_lengths[
            batch * query_count + query_row
        ]);
        size_t selected_base = (
            size_t(batch * query_count + query_row) * size_t(selected_width)
        );
        size_t compressed_batch = size_t(batch) * size_t(n_compressed_records);
        for (uint slot = 0u; slot < length; ++slot) {
            uint row = use_indices != 0
                ? uint(compressed_indices[selected_base + size_t(slot)])
                : slot;
            if (row < uint(n_compressed_records)) {
                const device uchar* record = compressed_records
                    + (compressed_batch + size_t(row)) * size_t(MTPLX_RECORD);
                mtplx_dsv4_consume(
                    record,
                    query,
                    accumulator,
                    running_max,
                    running_sum,
                    lane
                );
            }
        }

        float inverse_sum = 1.0f / running_sum;
        device T* output_ptr = out + query_base + lane * MTPLX_ELEMS;
        for (uint element = 0u; element < MTPLX_ELEMS; ++element) {
            output_ptr[element] = T(accumulator[element] * inverse_sum);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_stock432_sparse_mla",
        input_names=[
            "queries",
            "window_records",
            "window_start",
            "query_positions",
            "compressed_records",
            "compressed_indices",
            "compressed_lengths",
            "sinks",
            "scale",
            "n_queries",
            "n_window_records",
            "n_compressed_records",
            "selected_width",
            "use_indices",
        ],
        output_names=["out"],
        header=_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    batch, _heads, query_count, _width = (int(value) for value in queries.shape)
    window_count = int(window_records.shape[1])
    if compressed_records is None:
        compressed_records = mx.zeros(
            (batch, 0, _RECORD_BYTES),
            dtype=mx.uint8,
        )
    compressed_count = int(compressed_records.shape[1])
    if compressed_indices is None:
        compressed_indices = mx.zeros((batch, query_count, 1), dtype=mx.int32)
        selected_width = 1
        use_indices = 0
    else:
        selected_width = int(compressed_indices.shape[2])
        use_indices = 1
    if compressed_lengths is None:
        compressed_lengths = mx.zeros((batch, query_count), dtype=mx.int32)

    (output,) = _kernel()(
        inputs=[
            queries,
            window_records,
            int(window_start),
            query_positions,
            compressed_records,
            compressed_indices,
            compressed_lengths,
            sinks,
            float(scale),
            int(query_count),
            int(window_count),
            int(compressed_count),
            int(selected_width),
            int(use_indices),
        ],
        template=[("T", mx.bfloat16)],
        grid=(batch * _HEADS * query_count * _LANES, 1, 1),
        threadgroup=(_LANES, 1, 1),
        output_shapes=[(batch, _HEADS, query_count, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Validate and run the fixed Mia sparse-MLA contract.

    Exact-model execution uses :func:`install_nvfp4_sparse_mla` once and calls
    its returned function directly; this checked entry point is the codec oracle
    and construction boundary.
    """
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 sparse MLA requires Metal")
    if tuple(queries.shape[:2]) != (1, _HEADS) or int(queries.shape[-1]) != _HEAD_DIM:
        raise ValueError("Mia sparse MLA queries must have shape [1, 64, rows, 512]")
    if queries.dtype != mx.bfloat16:
        raise ValueError("Mia sparse MLA queries must be BF16")
    if (
        window_records.dtype != mx.uint8
        or tuple(window_records.shape[:1]) != (1,)
        or int(window_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError("Mia sparse MLA window records must be [1, rows, 432] uint8")
    query_count = int(queries.shape[2])
    if tuple(query_positions.shape) != (query_count,):
        raise ValueError("Mia sparse MLA query positions must match query rows")
    if tuple(sinks.shape) != (_HEADS,):
        raise ValueError("Mia sparse MLA sinks must have shape [64]")
    if compressed_records is not None and (
        compressed_records.dtype != mx.uint8
        or tuple(compressed_records.shape[:1]) != (1,)
        or int(compressed_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError(
            "Mia sparse MLA compressed records must be [1, rows, 432] uint8"
        )
    if compressed_indices is not None and tuple(compressed_indices.shape[:2]) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse MLA selected indices must be [1, queries, K]")
    if compressed_lengths is not None and tuple(compressed_lengths.shape) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse MLA compressed lengths must be [1, queries]")
    return _run_nvfp4_sparse_mla(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
    )


def install_nvfp4_sparse_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
):
    """Validate Mia's fixed geometry once and return the direct hot callable."""
    observed = (int(heads), int(head_dim), int(rope_dim), int(window_size))
    expected = (_HEADS, _HEAD_DIM, 64, _WINDOW)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia stock432 sparse MLA geometry: {observed!r} != {expected!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 sparse MLA installation requires Metal")
    _kernel()
    return _run_nvfp4_sparse_mla
