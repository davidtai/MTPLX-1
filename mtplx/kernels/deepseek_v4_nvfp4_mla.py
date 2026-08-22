"""Direct sparse MLA over Mia DeepSeek-V4 ``stock432`` NVFP4 records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from functools import partial

import mlx.core as mx

from mtplx.deepseek_v4_nvfp4_kv import PagedMiaNVFP4Records


_HEADS = 64
_HEAD_DIM = 512
_WINDOW = 128
_RECORD_BYTES = 432
_LANES = 32
_VALUES_PER_LANE = 16
_PREFILL_HEADS_PER_GROUP = 16
_PREFILL_CANDIDATE_TILE = 32
_PREFILL_QK_GROUPS = 4
_PREFILL_NAX_GROUPS = 8
_PREFILL_NAX_THREADS = _PREFILL_NAX_GROUPS * _LANES
_PREFILL_NAX_SCRATCH_BYTES = 28 * 1024
_MAX_QUERY_ROWS = 8_224
_DSPARK_ROWS = 5


@dataclass(frozen=True)
class MiaMLAWorkspace:
    """Shared fixed inputs for invariant empty MLA operands."""

    max_query_rows: int
    dummy_record: mx.array
    dummy_block_table: mx.array
    dummy_indices: mx.array
    empty_lengths: mx.array

    def indices(self, query_count: int) -> mx.array:
        return self.dummy_indices[:, : int(query_count)]

    def lengths(self, query_count: int) -> mx.array:
        return self.empty_lengths[:, : int(query_count)]


@lru_cache(maxsize=1)
def mia_mla_workspace() -> MiaMLAWorkspace:
    return MiaMLAWorkspace(
        max_query_rows=_MAX_QUERY_ROWS,
        dummy_record=mx.zeros((1, 1, _RECORD_BYTES), dtype=mx.uint8),
        dummy_block_table=mx.zeros((1,), dtype=mx.int32),
        dummy_indices=mx.zeros((1, _MAX_QUERY_ROWS, 1), dtype=mx.int32),
        empty_lengths=mx.zeros((1, _MAX_QUERY_ROWS), dtype=mx.int32),
    )


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
        uint lane,
        float attention_scale
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
        // SparkInfer applies the attention scale to the completed FP32 QK
        // accumulator.  Keeping it here also avoids rounding a separately
        // scaled copy of every BF16 query element on the decode route.
        float score = simd_sum(partial) * attention_scale;
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


_PREFILL_HEADER = _HEADER.replace(
    f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};",
    f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};\n"
    f"    constant constexpr int MTPLX_LANES = {_LANES};\n"
    f"    constant constexpr int MTPLX_PREFILL_HEADS = "
    f"{_PREFILL_HEADS_PER_GROUP};\n"
    f"    constant constexpr int MTPLX_PREFILL_TILE = {_PREFILL_CANDIDATE_TILE};\n"
    f"    constant constexpr int MTPLX_PREFILL_QK_GROUPS = "
    f"{_PREFILL_QK_GROUPS};\n"
    f"    constant constexpr int MTPLX_PREFILL_NAX_GROUPS = "
    f"{_PREFILL_NAX_GROUPS};\n"
    f"    constant constexpr int MTPLX_PREFILL_NAX_THREADS = "
    f"{_PREFILL_NAX_THREADS};\n"
    f"    constant constexpr int MTPLX_PREFILL_NAX_SCRATCH = "
    f"{_PREFILL_NAX_SCRATCH_BYTES};",
) + r"""
    inline bfloat mtplx_dsv4_device_value_bf16(
        const device uchar* record,
        uint dim,
        float latent_scale
    ) {
        return bfloat(mtplx_dsv4_latent(record, dim, latent_scale));
    }
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
            query[element] = float(query_ptr[element]);
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
                    lane,
                    float(scale)
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
                uint physical_row = row;
                if (use_paged_compressed != 0) {
                    uint logical_block = row / uint(compressed_block_size);
                    uint row_in_block = row - logical_block * uint(compressed_block_size);
                    uint physical_block = uint(compressed_block_table[logical_block]);
                    physical_row = physical_block * uint(compressed_block_size)
                        + row_in_block;
                } else {
                    physical_row += uint(compressed_batch);
                }
                const device uchar* record = compressed_records
                    + size_t(physical_row) * size_t(MTPLX_RECORD);
                mtplx_dsv4_consume(
                    record,
                    query,
                    accumulator,
                    running_max,
                    running_sum,
                    lane,
                    float(scale)
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
            "compressed_block_table",
            "compressed_indices",
            "compressed_lengths",
            "sinks",
            "scale",
            "n_queries",
            "n_window_records",
            "n_compressed_records",
            "selected_width",
            "use_indices",
            "compressed_block_size",
            "use_paged_compressed",
        ],
        output_names=["out"],
        header=_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _dspark_k5_kernel():
    """Direct noncausal K5 attention over a fixed context ring and five rows.

    The packaged DSpark path never promotes its five neural rows into persistent
    cache ownership. Keep the 128-row target-context ring and the five
    proposal-local records as separate Metal inputs and consume them as one
    online-softmax union. This is the fixed-K5 specialization of the decode
    kernel above, not a second attention implementation.
    """

    source = r"""
        uint lane = thread_index_in_simdgroup;
        uint group = threadgroup_position_in_grid.x;
        uint query_count = uint(MTPLX_DSPARK_ROWS);
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
            query[element] = float(query_ptr[element]);
            accumulator[element] = 0.0f;
        }

        float running_max = sinks[head];
        float running_sum = 1.0f;
        int prefix = int(prefix_length);
        int context_start = max(0, prefix - MTPLX_WINDOW);
        size_t ring_batch = size_t(batch) * size_t(MTPLX_WINDOW);
        for (int absolute_position = context_start;
             absolute_position < prefix;
             ++absolute_position) {
            uint physical_row = uint(absolute_position) % uint(MTPLX_WINDOW);
            const device uchar* record = context_records
                + (ring_batch + size_t(physical_row)) * size_t(MTPLX_RECORD);
            mtplx_dsv4_consume(
                record,
                query,
                accumulator,
                running_max,
                running_sum,
                lane,
                float(scale)
            );
        }

        size_t draft_batch = size_t(batch) * size_t(MTPLX_DSPARK_ROWS);
        for (uint row = 0u; row < uint(MTPLX_DSPARK_ROWS); ++row) {
            const device uchar* record = draft_records
                + (draft_batch + size_t(row)) * size_t(MTPLX_RECORD);
            mtplx_dsv4_consume(
                record,
                query,
                accumulator,
                running_max,
                running_sum,
                lane,
                float(scale)
            );
        }

        float inverse_sum = 1.0f / running_sum;
        device T* output_ptr = out + query_base + lane * MTPLX_ELEMS;
        for (uint element = 0u; element < MTPLX_ELEMS; ++element) {
            output_ptr[element] = T(accumulator[element] * inverse_sum);
        }
    """
    header = _HEADER.replace(
        f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};",
        f"constant constexpr int MTPLX_ELEMS = {_VALUES_PER_LANE};\n"
        f"    constant constexpr int MTPLX_DSPARK_ROWS = {_DSPARK_ROWS};",
    )
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_stock432_dspark_k5",
        input_names=[
            "queries",
            "context_records",
            "draft_records",
            "prefix_length",
            "sinks",
            "scale",
        ],
        output_names=["out"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _run_dspark_k5_nvfp4_mla(
    queries: mx.array,
    context_records: mx.array,
    draft_records: mx.array,
    prefix_length: int,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    batch = int(queries.shape[0])
    (output,) = _dspark_k5_kernel()(
        inputs=[
            queries,
            context_records,
            draft_records,
            int(prefix_length),
            sinks,
            float(scale),
        ],
        template=[("T", mx.bfloat16)],
        grid=(batch * _HEADS * _DSPARK_ROWS * _LANES, 1, 1),
        threadgroup=(_LANES, 1, 1),
        output_shapes=[(batch, _HEADS, _DSPARK_ROWS, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )
    return output


@lru_cache(maxsize=1)
def _prefill_nax_mg16_kernel():
    """Mia/SparkInfer NVFP4 prefill mapped to the M5 NAX tile geometry.

    SparkInfer owns sixteen query heads together, dequantizes each selected KV
    record once per tensor operand, runs BF16 QK and P.V tensor operations, and
    carries one FP32 online softmax over the SWA/indexed-cache union.  Metal's
    native NAX primitive is M16xN32xK16, so this implementation keeps the same
    ownership with a 32-candidate tile, four 128-wide QK splits, and eight PV
    SIMD groups that each own two 32-wide output fragments.
    """
    source = r"""
        using namespace mpp::tensor_ops;

        uint lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint thread_index = simd_group * MTPLX_LANES + lane;
        uint group = threadgroup_position_in_grid.x;
        uint query_count = uint(n_queries);
        uint head_groups = uint(MTPLX_H / MTPLX_PREFILL_HEADS);
        uint groups_per_batch = head_groups * query_count;
        uint batch = group / groups_per_batch;
        uint within_batch = group - batch * groups_per_batch;
        uint head_group = within_batch / query_count;
        uint query_row = within_batch - head_group * query_count;
        uint head_base = head_group * uint(MTPLX_PREFILL_HEADS);
        int query_position = query_positions[query_row];

        // The query is invariant across all candidate tiles.  Load its sixteen
        // BF16 head rows once, matching SparkInfer's S0 ownership.
        threadgroup uchar scratch[MTPLX_PREFILL_NAX_SCRATCH];
        threadgroup bfloat* q_shared =
            reinterpret_cast<threadgroup bfloat*>(scratch);
        for (uint index = thread_index;
             index < uint(MTPLX_PREFILL_HEADS * MTPLX_D);
             index += uint(MTPLX_PREFILL_NAX_THREADS)) {
            uint local_head = index / uint(MTPLX_D);
            uint dim = index - local_head * uint(MTPLX_D);
            size_t query_index = (
                (size_t(batch * uint(MTPLX_H) + head_base + local_head)
                 * size_t(query_count) + size_t(query_row))
                * size_t(MTPLX_D) + size_t(dim)
            );
            q_shared[index] = bfloat(queries[query_index]);
        }

        threadgroup float running_max[MTPLX_PREFILL_HEADS];
        threadgroup float running_sum[MTPLX_PREFILL_HEADS];
        threadgroup float row_correction[MTPLX_PREFILL_HEADS];
        threadgroup uint tile_rows[MTPLX_PREFILL_TILE];
        threadgroup uchar tile_kinds[MTPLX_PREFILL_TILE];
        if (thread_index < uint(MTPLX_PREFILL_HEADS)) {
            running_max[thread_index] = sinks[head_base + thread_index];
            running_sum[thread_index] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr auto qk_desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        constexpr auto pv_desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate
        );
        matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
        matmul2d<pv_desc, metal::execution_simdgroup> pv_op;

        auto pv_acc_lo = pv_op.template get_destination_cooperative_tensor<
            tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup bfloat, extents<int, 32, 16>, tensor_inline>,
            float
        >();
        auto pv_acc_hi = pv_op.template get_destination_cooperative_tensor<
            tensor<threadgroup bfloat, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup bfloat, extents<int, 32, 16>, tensor_inline>,
            float
        >();
        for (uint16_t index = 0;
             index < pv_acc_lo.get_capacity(); ++index) {
            pv_acc_lo[index] = 0.0f;
            pv_acc_hi[index] = 0.0f;
        }

        int first_window = max(
            0,
            query_position - int(window_start) - MTPLX_WINDOW + 1
        );
        int window_end = min(
            int(n_window_records),
            query_position - int(window_start) + 1
        );
        uint visible_window = uint(window_end - first_window);
        uint compressed_length = uint(compressed_lengths[
            batch * query_count + query_row
        ]);
        uint total_candidates = visible_window + compressed_length;
        size_t window_batch = size_t(batch) * size_t(n_window_records);
        size_t selected_base = (
            size_t(batch * query_count + query_row) * size_t(selected_width)
        );
        size_t compressed_batch = size_t(batch) * size_t(n_compressed_records);

        for (uint tile_start = 0u; tile_start < total_candidates;
             tile_start += uint(MTPLX_PREFILL_TILE)) {
            uint tile_count = min(
                uint(MTPLX_PREFILL_TILE), total_candidates - tile_start
            );

            // Resolve the dual-cache union once for this candidate tile.  Every
            // QK/PV operand thereafter consumes the same physical-row table.
            if (thread_index < uint(MTPLX_PREFILL_TILE)) {
                uint candidate = thread_index;
                uint physical_row = 0u;
                uchar kind = 0u;
                if (candidate < tile_count) {
                    uint global_candidate = tile_start + candidate;
                    if (global_candidate < visible_window) {
                        physical_row = uint(
                            window_batch + size_t(first_window)
                            + size_t(global_candidate)
                        );
                    } else {
                        kind = 1u;
                        uint slot = global_candidate - visible_window;
                        uint row = use_indices != 0
                            ? uint(compressed_indices[
                                selected_base + size_t(slot)
                            ])
                            : slot;
                        physical_row = row;
                        if (use_paged_compressed != 0) {
                            uint logical_block =
                                row / uint(compressed_block_size);
                            uint row_in_block = row
                                - logical_block * uint(compressed_block_size);
                            uint physical_block = uint(
                                compressed_block_table[logical_block]
                            );
                            physical_row = physical_block
                                * uint(compressed_block_size) + row_in_block;
                        } else {
                            physical_row += uint(compressed_batch);
                        }
                    }
                }
                tile_rows[candidate] = physical_row;
                tile_kinds[candidate] = kind;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // QK: four NAX SIMD groups split K=512 into four K=128 ranges.
            // Each group builds the native BF16 K operand directly from the
            // stock432 record, then stores one FP32 M16xN32 partial matrix.
            threadgroup bfloat* qk_tiles =
                reinterpret_cast<threadgroup bfloat*>(scratch + 16384);
            threadgroup float* qk_partials =
                reinterpret_cast<threadgroup float*>(scratch + 20480);
            if (simd_group < uint(MTPLX_PREFILL_QK_GROUPS)) {
                threadgroup bfloat* qk_values = qk_tiles
                    + simd_group * uint(MTPLX_PREFILL_TILE * 16);
                threadgroup float* qk_partial = qk_partials
                    + simd_group * uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE);

                tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> Q(
                    q_shared,
                    dextents<int, 2>{MTPLX_D, MTPLX_PREFILL_HEADS},
                    array<int, 2>{1, MTPLX_D}
                );
                tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> K(
                    qk_values,
                    dextents<int, 2>{MTPLX_PREFILL_TILE, 16},
                    array<int, 2>{1, MTPLX_PREFILL_TILE}
                );
                tensor<threadgroup float, dextents<int, 2>, tensor_inline> S(
                    qk_partial,
                    dextents<int, 2>{MTPLX_PREFILL_TILE, MTPLX_PREFILL_HEADS},
                    array<int, 2>{1, MTPLX_PREFILL_TILE}
                );
                auto qk_acc = qk_op.template get_destination_cooperative_tensor<
                    tensor<threadgroup bfloat,
                           extents<int, 16, 16>, tensor_inline>,
                    tensor<threadgroup bfloat,
                           extents<int, 32, 16>, tensor_inline>,
                    float
                >();
                for (uint16_t index = 0;
                     index < qk_acc.get_capacity(); ++index) {
                    qk_acc[index] = 0.0f;
                }

                uint k_begin = simd_group * 128u;
                for (uint k0 = k_begin; k0 < k_begin + 128u; k0 += 16u) {
                    uint candidate = lane;
                    const device uchar* record = tile_kinds[candidate] == 0u
                        ? window_records
                            + size_t(tile_rows[candidate]) * size_t(MTPLX_RECORD)
                        : compressed_records
                            + size_t(tile_rows[candidate]) * size_t(MTPLX_RECORD);
                    if (candidate < tile_count) {
                        if (k0 < 448u) {
                            float latent_scale = mtplx_dsv4_e4m3(
                                record[256u + (k0 >> 4)]
                            );
                            for (uint element = 0u; element < 16u; ++element) {
                                uint dim = k0 + element;
                                qk_values[element * uint(MTPLX_PREFILL_TILE)
                                          + candidate] = bfloat(
                                    mtplx_dsv4_latent(
                                        record, dim, latent_scale
                                    )
                                );
                            }
                        } else {
                            const device bfloat* rope = reinterpret_cast<
                                const device bfloat*>(record + 304u);
                            for (uint element = 0u; element < 16u; ++element) {
                                uint dim = k0 + element;
                                qk_values[element * uint(MTPLX_PREFILL_TILE)
                                          + candidate] = rope[dim - 448u];
                            }
                        }
                    } else {
                        for (uint element = 0u; element < 16u; ++element) {
                            qk_values[element * uint(MTPLX_PREFILL_TILE)
                                      + candidate] = bfloat(0.0f);
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    auto q_tile = Q.template slice<16, 16>(k0, 0);
                    auto k_tile = K.template slice<32, 16>(0, 0);
                    qk_op.run(q_tile, k_tile, qk_acc);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                auto score_tile = S.template slice<32, 16>(0, 0);
                qk_acc.store(score_tile);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Sum the four K partials in FP32 and apply the attention scale
            // after QK, matching SparkInfer's S3 ordering.
            threadgroup float* tile_scores =
                reinterpret_cast<threadgroup float*>(scratch + 16384);
            for (uint offset = thread_index;
                 offset < uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE);
                 offset += uint(MTPLX_PREFILL_NAX_THREADS)) {
                float sum01 = qk_partials[offset]
                    + qk_partials[
                        uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE) + offset
                    ];
                float sum23 = qk_partials[
                        2u * uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE)
                        + offset
                    ]
                    + qk_partials[
                        3u * uint(MTPLX_PREFILL_HEADS * MTPLX_PREFILL_TILE)
                        + offset
                    ];
                tile_scores[offset] = (sum01 + sum23) * float(scale);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // S4/S5: one online-softmax state per head; probabilities are BF16
            // because the native stock432 P.V path is BF16 tensor math.
            threadgroup bfloat* probabilities =
                reinterpret_cast<threadgroup bfloat*>(scratch + 18432);
            if (thread_index < uint(MTPLX_PREFILL_HEADS)) {
                uint row = thread_index;
                uint row_base = row * uint(MTPLX_PREFILL_TILE);
                float old_max = running_max[row];
                float next_max = old_max;
                for (uint candidate = 0u; candidate < tile_count; ++candidate) {
                    next_max = max(next_max, tile_scores[row_base + candidate]);
                }
                float correction = fast::exp(old_max - next_max);
                float next_sum = running_sum[row] * correction;
                for (uint candidate = 0u;
                     candidate < uint(MTPLX_PREFILL_TILE); ++candidate) {
                    float probability = candidate < tile_count
                        ? fast::exp(
                            tile_scores[row_base + candidate] - next_max
                        )
                        : 0.0f;
                    probabilities[row_base + candidate] = bfloat(probability);
                    next_sum += probability;
                }
                row_correction[row] = correction;
                running_max[row] = next_max;
                running_sum[row] = next_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint matrix_row_base = ((lane & 7u) >> 1)
                + ((lane >> 4) * 4u);
            for (uint16_t index = 0;
                 index < pv_acc_lo.get_capacity(); ++index) {
                uint matrix_row = matrix_row_base
                    + (((uint(index) >> 2) & 1u) << 3);
                float correction = row_correction[matrix_row];
                pv_acc_lo[index] *= correction;
                pv_acc_hi[index] *= correction;
            }

            // S6: eight groups cover 64 V dimensions each.  Each group reuses
            // one M16xN32xK16 B tile for its low/high 32-dimension fragments.
            threadgroup bfloat* value_tiles =
                reinterpret_cast<threadgroup bfloat*>(scratch + 19456);
            threadgroup bfloat* value_base = value_tiles
                + simd_group * 16u * 32u;
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> P(
                probabilities,
                dextents<int, 2>{MTPLX_PREFILL_TILE, MTPLX_PREFILL_HEADS},
                array<int, 2>{1, MTPLX_PREFILL_TILE}
            );
            tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> V(
                value_base,
                dextents<int, 2>{32, 16},
                array<int, 2>{1, 32}
            );

            for (uint candidate_base = 0u;
                 candidate_base < uint(MTPLX_PREFILL_TILE);
                 candidate_base += 16u) {
                auto probability_tile = P.template slice<16, 16>(
                    candidate_base, 0
                );
                for (uint output_half = 0u; output_half < 2u; ++output_half) {
                    uint candidate = lane >> 1;
                    uint half_group = lane & 1u;
                    uint dim_base = simd_group * 64u + output_half * 32u
                        + half_group * 16u;
                    uint tile_candidate = candidate_base + candidate;
                    const device uchar* record = tile_kinds[tile_candidate] == 0u
                        ? window_records
                            + size_t(tile_rows[tile_candidate])
                                * size_t(MTPLX_RECORD)
                        : compressed_records
                            + size_t(tile_rows[tile_candidate])
                                * size_t(MTPLX_RECORD);
                    if (tile_candidate < tile_count) {
                        float latent_scale = mtplx_dsv4_e4m3(
                            record[256u + (dim_base >> 4)]
                        );
                        for (uint element = 0u; element < 16u; ++element) {
                            value_base[candidate * 32u
                                       + half_group * 16u + element] =
                                mtplx_dsv4_device_value_bf16(
                                    record, dim_base + element, latent_scale
                                );
                        }
                    } else {
                        for (uint element = 0u; element < 16u; ++element) {
                            value_base[candidate * 32u
                                       + half_group * 16u + element] =
                                bfloat(0.0f);
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    auto value_tile = V.template slice<32, 16>(0, 0);
                    if (output_half == 0u) {
                        pv_op.run(
                            probability_tile, value_tile, pv_acc_lo
                        );
                    } else {
                        pv_op.run(
                            probability_tile, value_tile, pv_acc_hi
                        );
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint matrix_row_base = ((lane & 7u) >> 1)
            + ((lane >> 4) * 4u);
        uint matrix_col_base = ((lane & 1u) << 2)
            + (((lane >> 3) & 1u) << 3);
        for (uint16_t index = 0;
             index < pv_acc_lo.get_capacity(); ++index) {
            uint matrix_row = matrix_row_base
                + (((uint(index) >> 2) & 1u) << 3);
            uint matrix_col = matrix_col_base + (uint(index) & 3u)
                + (uint(index) >> 3) * 16u;
            uint output_head = head_base + matrix_row;
            uint output_dim_lo = simd_group * 64u + matrix_col;
            uint output_dim_hi = output_dim_lo + 32u;
            size_t output_base = (
                (size_t(batch * uint(MTPLX_H) + output_head)
                 * size_t(query_count) + size_t(query_row))
                * size_t(MTPLX_D)
            );
            float inverse_sum = 1.0f / running_sum[matrix_row];
            out[output_base + size_t(output_dim_lo)] =
                T(pv_acc_lo[index] * inverse_sum);
            out[output_base + size_t(output_dim_hi)] =
                T(pv_acc_hi[index] * inverse_sum);
        }
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_stock432_prefill_nax_mg16_tile32_v1",
        input_names=[
            "queries",
            "window_records",
            "window_start",
            "query_positions",
            "compressed_records",
            "compressed_block_table",
            "compressed_indices",
            "compressed_lengths",
            "sinks",
            "scale",
            "n_queries",
            "n_window_records",
            "n_compressed_records",
            "selected_width",
            "use_indices",
            "compressed_block_size",
            "use_paged_compressed",
        ],
        output_names=["out"],
        header=(
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            + _PREFILL_HEADER
        ),
        source=source,
        ensure_row_contiguous=True,
    )


def _run_nvfp4_sparse_mla_storage(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_block_table: mx.array,
    compressed_count: int,
    compressed_block_size: int,
    use_paged_compressed: int,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    batch, _heads, query_count, _width = (int(value) for value in queries.shape)
    window_count = int(window_records.shape[1])
    # MLX assigns zero-sized array inputs to Metal's constant address space.
    # Keep the logical counts at zero, but pass one real device-backed record so
    # the fixed kernel signature is identical for every layer and phase.
    if window_count == 0:
        window_records = workspace.dummy_record
    if compressed_records is None:
        compressed_records = window_records[:, :1]
    if compressed_indices is None:
        compressed_indices = workspace.indices(query_count)
        selected_width = 1
        use_indices = 0
    else:
        selected_width = int(compressed_indices.shape[2])
        use_indices = 1
    if compressed_lengths is None:
        compressed_lengths = workspace.lengths(query_count)

    (output,) = _kernel()(
        inputs=[
            queries,
            window_records,
            int(window_start),
            query_positions,
            compressed_records,
            compressed_block_table,
            compressed_indices,
            compressed_lengths,
            sinks,
            float(scale),
            int(query_count),
            int(window_count),
            int(compressed_count),
            int(selected_width),
            int(use_indices),
            int(compressed_block_size),
            int(use_paged_compressed),
        ],
        template=[("T", mx.bfloat16)],
        grid=(batch * _HEADS * query_count * _LANES, 1, 1),
        threadgroup=(_LANES, 1, 1),
        output_shapes=[(batch, _HEADS, query_count, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def _run_nvfp4_prefill_mla_storage(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_block_table: mx.array,
    compressed_count: int,
    compressed_block_size: int,
    use_paged_compressed: int,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    batch, _heads, query_count, _width = (int(value) for value in queries.shape)
    window_count = int(window_records.shape[1])
    if window_count == 0:
        window_records = workspace.dummy_record
    if compressed_records is None:
        compressed_records = window_records[:, :1]
    if compressed_indices is None:
        compressed_indices = workspace.indices(query_count)
        selected_width = 1
        use_indices = 0
    else:
        selected_width = int(compressed_indices.shape[2])
        use_indices = 1
    if compressed_lengths is None:
        compressed_lengths = workspace.lengths(query_count)

    (output,) = _prefill_nax_mg16_kernel()(
        inputs=[
            queries,
            window_records,
            int(window_start),
            query_positions,
            compressed_records,
            compressed_block_table,
            compressed_indices,
            compressed_lengths,
            sinks,
            float(scale),
            int(query_count),
            int(window_count),
            int(compressed_count),
            int(selected_width),
            int(use_indices),
            int(compressed_block_size),
            int(use_paged_compressed),
        ],
        template=[("T", mx.bfloat16)],
        grid=(
            batch
            * (_HEADS // _PREFILL_HEADS_PER_GROUP)
            * query_count
            * _PREFILL_NAX_THREADS,
            1,
            1,
        ),
        threadgroup=(_PREFILL_NAX_THREADS, 1, 1),
        output_shapes=[(batch, _HEADS, query_count, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )
    return output


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
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    compressed_count = (
        0 if compressed_records is None else int(compressed_records.shape[1])
    )
    return _run_nvfp4_sparse_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        workspace.dummy_block_table,
        compressed_count,
        1,
        0,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def _run_nvfp4_prefill_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    compressed_count = (
        0 if compressed_records is None else int(compressed_records.shape[1])
    )
    return _run_nvfp4_prefill_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        workspace.dummy_block_table,
        compressed_count,
        1,
        0,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def _run_paged_nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: PagedMiaNVFP4Records | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    if compressed_records is None:
        pages = None
        block_table = workspace.dummy_block_table
        compressed_count = 0
        block_size = 1
    else:
        pages = compressed_records.records
        block_table = compressed_records.block_table
        compressed_count = int(compressed_records.length)
        block_size = int(compressed_records.block_size)
    return _run_nvfp4_sparse_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        pages,
        block_table,
        compressed_count,
        block_size,
        1,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


def _run_paged_nvfp4_prefill_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: PagedMiaNVFP4Records | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
    *,
    workspace: MiaMLAWorkspace,
) -> mx.array:
    if compressed_records is None:
        pages = None
        block_table = workspace.dummy_block_table
        compressed_count = 0
        block_size = 1
    else:
        pages = compressed_records.records
        block_table = compressed_records.block_table
        compressed_count = int(compressed_records.length)
        block_size = int(compressed_records.block_size)
    return _run_nvfp4_prefill_mla_storage(
        queries,
        window_records,
        window_start,
        query_positions,
        pages,
        block_table,
        compressed_count,
        block_size,
        1,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=workspace,
    )


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
    paged_compressed = isinstance(compressed_records, PagedMiaNVFP4Records)
    if paged_compressed:
        if (
            compressed_records.records.dtype != mx.uint8
            or int(compressed_records.records.shape[-1]) != _RECORD_BYTES
            or int(compressed_records.block_size) <= 0
        ):
            raise ValueError("invalid paged Mia sparse MLA compressed records")
    elif compressed_records is not None and (
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
    runner = _run_paged_nvfp4_sparse_mla if paged_compressed else _run_nvfp4_sparse_mla
    return runner(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=mia_mla_workspace(),
    )


def nvfp4_prefill_mla(
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
    """Validated oracle boundary for the measured large-M head-group route."""
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 sparse prefill MLA requires Metal")
    if tuple(queries.shape[:2]) != (1, _HEADS) or int(queries.shape[-1]) != _HEAD_DIM:
        raise ValueError("Mia sparse prefill queries must have shape [1, 64, rows, 512]")
    if queries.dtype != mx.bfloat16:
        raise ValueError("Mia sparse prefill queries must be BF16")
    if (
        window_records.dtype != mx.uint8
        or tuple(window_records.shape[:1]) != (1,)
        or int(window_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError(
            "Mia sparse prefill window records must be [1, rows, 432] uint8"
        )
    query_count = int(queries.shape[2])
    if tuple(query_positions.shape) != (query_count,):
        raise ValueError("Mia sparse prefill positions must match query rows")
    if tuple(sinks.shape) != (_HEADS,):
        raise ValueError("Mia sparse prefill sinks must have shape [64]")
    paged_compressed = isinstance(compressed_records, PagedMiaNVFP4Records)
    if paged_compressed:
        if (
            compressed_records.records.dtype != mx.uint8
            or int(compressed_records.records.shape[-1]) != _RECORD_BYTES
            or int(compressed_records.block_size) <= 0
        ):
            raise ValueError("invalid paged Mia sparse prefill records")
    elif compressed_records is not None and (
        compressed_records.dtype != mx.uint8
        or tuple(compressed_records.shape[:1]) != (1,)
        or int(compressed_records.shape[-1]) != _RECORD_BYTES
    ):
        raise ValueError(
            "Mia sparse prefill records must be [1, rows, 432] uint8"
        )
    if compressed_indices is not None and tuple(compressed_indices.shape[:2]) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse prefill indices must be [1, queries, K]")
    if compressed_lengths is not None and tuple(compressed_lengths.shape) != (
        1,
        query_count,
    ):
        raise ValueError("Mia sparse prefill lengths must be [1, queries]")
    runner = (
        _run_paged_nvfp4_prefill_mla
        if paged_compressed
        else _run_nvfp4_prefill_mla
    )
    return runner(
        queries,
        window_records,
        window_start,
        query_positions,
        compressed_records,
        compressed_indices,
        compressed_lengths,
        sinks,
        scale,
        workspace=mia_mla_workspace(),
    )


def install_nvfp4_sparse_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    paged_compressed: bool = False,
    workspace: MiaMLAWorkspace | None = None,
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
    runner = (
        _run_paged_nvfp4_sparse_mla
        if paged_compressed
        else _run_nvfp4_sparse_mla
    )
    return partial(runner, workspace=workspace or mia_mla_workspace())


def install_dspark_k5_nvfp4_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    block_size: int,
):
    """Install the packaged fixed-K5, 128-row stock432 DSpark attention."""

    observed = (
        int(heads),
        int(head_dim),
        int(rope_dim),
        int(window_size),
        int(block_size),
    )
    expected = (_HEADS, _HEAD_DIM, 64, _WINDOW, _DSPARK_ROWS)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia stock432 DSpark geometry: {observed!r} != {expected!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 DSpark K5 installation requires Metal")
    _dspark_k5_kernel()
    return _run_dspark_k5_nvfp4_mla


def install_nvfp4_prefill_mla(
    *,
    heads: int,
    head_dim: int,
    rope_dim: int,
    window_size: int,
    paged_compressed: bool = False,
    workspace: MiaMLAWorkspace | None = None,
):
    """Install Mia's M5 NAX prefill engine for the fixed stock432 geometry."""
    observed = (int(heads), int(head_dim), int(rope_dim), int(window_size))
    expected = (_HEADS, _HEAD_DIM, 64, _WINDOW)
    if observed != expected:
        raise ValueError(
            f"unsupported Mia stock432 prefill geometry: {observed!r} != {expected!r}"
        )
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 prefill installation requires Metal")
    from mtplx.nax_verify import nax_available

    if not nax_available():
        raise RuntimeError(
            "Mia stock432 prefill requires Apple G17 NAX on macOS 26.2 or newer"
        )
    runner = (
        _run_paged_nvfp4_prefill_mla
        if paged_compressed
        else _run_nvfp4_prefill_mla
    )
    return partial(runner, workspace=workspace or mia_mla_workspace())
