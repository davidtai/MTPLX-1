"""Retained target-shaped kernels from the pinned Qwen 3.8 challenge."""

from __future__ import annotations

import weakref
from typing import Any

_DUAL_RMS_CONCAT_KERNEL = None
_ROW9_PAIRED_QMV_KERNEL = None
qwen38_dual_norm_calls = 0
qwen38_row9_qmv_calls = 0
_ROW9_PATCH: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "modules": {},
}


def qwen38_row9_paired_qmv_g32_m4(
    x: Any,
    weight: Any,
    scales: Any,
    biases: Any,
) -> Any:
    """Pair adjacent target-verify rows while streaming each G32 tile once.

    Source row 9 implemented the same two-row sharing for affine-4/group-64.
    The fixed MTPLX D3 verifier is four target rows over the group-32 trunk, so
    this preserves the source ownership and arithmetic while changing only the
    live group geometry.
    """

    import mlx.core as mx

    global _ROW9_PAIRED_QMV_KERNEL, qwen38_row9_qmv_calls
    if x.ndim != 2 or tuple(x.shape[:1]) != (4,):
        raise ValueError("row 9 QMV requires the fixed M=4 target shape")
    hidden = int(x.shape[1])
    output = int(weight.shape[0])
    if (
        x.dtype != mx.bfloat16
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or hidden % 512
        or output < 4096
        or output % 8
        or int(weight.shape[1]) != hidden // 8
        or tuple(scales.shape) != (output, hidden // 32)
        or tuple(biases.shape) != tuple(scales.shape)
    ):
        raise ValueError("unsupported row 9 affine-4/group-32 QMV contract")
    if _ROW9_PAIRED_QMV_KERNEL is None:
        _ROW9_PAIRED_QMV_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_row9_paired_qmv_g32_m4_v1",
            input_names=["w", "scales", "biases", "x"],
            output_names=["y"],
            source=r"""
                constexpr int M = 4;
                constexpr int inputs_per_group = 2;
                constexpr int rows_per_simd = 4;
                constexpr int values_per_thread = 16;
                constexpr int block_size = values_per_thread * 32;
                constexpr int group_size = 32;

                const int in_vec_size = int(x_shape[1]);
                const int out_vec_size = int(w_shape[0]);
                const int first_m = int(threadgroup_position_in_grid.x)
                    * inputs_per_group;
                const int out_row = int(threadgroup_position_in_grid.y) * 8
                    + int(simdgroup_index_in_threadgroup) * rows_per_simd;
                const int lane = int(thread_index_in_simdgroup);
                const int in_vec_size_w = in_vec_size / 2;
                const int in_vec_size_g = in_vec_size / group_size;

                float2 result[rows_per_simd];
                for (int r = 0; r < rows_per_simd; ++r) {
                    result[r] = float2(0.0f);
                }
                for (int k = 0; k < in_vec_size; k += block_size) {
                    ushort packed[rows_per_simd][4];
                    float scale_local[rows_per_simd];
                    float bias_local[rows_per_simd];
                    for (int r = 0; r < rows_per_simd; ++r) {
                        const int row = out_row + r;
                        const device ushort* ws =
                            reinterpret_cast<const device ushort*>(
                                reinterpret_cast<const device uchar*>(w)
                                + row * in_vec_size_w + k / 2 + lane * 8);
                        for (int i = 0; i < 4; ++i) {
                            packed[r][i] = ws[i];
                        }
                        const int group_index = row * in_vec_size_g
                            + k / group_size + lane / 2;
                        scale_local[r] = float(scales[group_index]);
                        bias_local[r] = float(biases[group_index]);
                    }

                    float x0[values_per_thread];
                    float x1[values_per_thread];
                    float2 sums = float2(0.0f);
                    const device bfloat16_t* a0 = x + first_m * in_vec_size
                        + k + lane * values_per_thread;
                    const device bfloat16_t* a1 = a0 + in_vec_size;
                    for (int i = 0; i < values_per_thread; i += 4) {
                        float4 v0 = float4(a0[i], a0[i + 1], a0[i + 2], a0[i + 3]);
                        float4 v1 = float4(a1[i], a1[i + 1], a1[i + 2], a1[i + 3]);
                        sums += float2(v0[0] + v0[1] + v0[2] + v0[3],
                                       v1[0] + v1[1] + v1[2] + v1[3]);
                        x0[i] = v0[0];
                        x0[i + 1] = v0[1] / 16.0f;
                        x0[i + 2] = v0[2] / 256.0f;
                        x0[i + 3] = v0[3] / 4096.0f;
                        x1[i] = v1[0];
                        x1[i + 1] = v1[1] / 16.0f;
                        x1[i + 2] = v1[2] / 256.0f;
                        x1[i + 3] = v1[3] / 4096.0f;
                    }
                    for (int r = 0; r < rows_per_simd; ++r) {
                        float2 partial = float2(0.0f);
                        for (int i = 0; i < 4; ++i) {
                            partial +=
                                float2(x0[4 * i], x1[4 * i])
                                    * float(packed[r][i] & 0x000f)
                                + float2(x0[4 * i + 1], x1[4 * i + 1])
                                    * float(packed[r][i] & 0x00f0)
                                + float2(x0[4 * i + 2], x1[4 * i + 2])
                                    * float(packed[r][i] & 0x0f00)
                                + float2(x0[4 * i + 3], x1[4 * i + 3])
                                    * float(packed[r][i] & 0xf000);
                        }
                        result[r] += scale_local[r] * partial
                            + bias_local[r] * sums;
                    }
                }
                for (int r = 0; r < rows_per_simd; ++r) {
                    float reduced0 = simd_sum(result[r][0]);
                    float reduced1 = simd_sum(result[r][1]);
                    if (lane == 0) {
                        y[first_m * out_vec_size + out_row + r]
                            = bfloat16_t(reduced0);
                        y[(first_m + 1) * out_vec_size + out_row + r]
                            = bfloat16_t(reduced1);
                    }
                }
            """,
            ensure_row_contiguous=True,
        )
    qwen38_row9_qmv_calls += 1
    (result,) = _ROW9_PAIRED_QMV_KERNEL(
        inputs=[weight, scales, biases, x],
        template=[],
        grid=(2 * 32, (output // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(4, output)],
        output_dtypes=[mx.bfloat16],
    )
    return result


def _row9_route_active() -> bool:
    from .attention_context import (
        current_attention_phase,
        current_model_forward_kind,
    )

    return (
        current_attention_phase() == "decode_verify"
        and current_model_forward_kind() == "target_verify"
    )


def _row9_paired_qmv_dispatch(self: Any, x: Any, width: int) -> Any | None:
    import mlx.core as mx

    if not _ROW9_PATCH["enabled"] or not _row9_route_active() or int(width) != 4:
        return None
    owner_ref = _ROW9_PATCH["modules"].get(id(self))
    if owner_ref is None or owner_ref() is not self:
        return None
    if (
        int(getattr(self, "bits", 0) or 0) != 4
        or int(getattr(self, "group_size", 0) or 0) != 32
        or str(getattr(self, "mode", "")).lower() != "affine"
        or x.dtype != mx.bfloat16
    ):
        return None
    weight = self["weight"]
    hidden = int(x.shape[-1])
    output = int(weight.shape[0])
    if (
        hidden % 512
        or output < 4096
        or output % 8
        or int(weight.shape[1]) != hidden // 8
    ):
        return None
    y = qwen38_row9_paired_qmv_g32_m4(
        x.reshape(4, hidden),
        weight,
        self["scales"],
        self["biases"],
    ).reshape(*x.shape[:-1], output)
    if "bias" in self:
        y = y + self["bias"]
    return y


def configure_qwen38_row9_paired_qmv(*, active: bool, model: Any) -> dict[str, Any]:
    """Bind row 9 to one loaded Qwen target and make ABBA toggling reversible."""

    if active and not _ROW9_PATCH["installed"]:
        from .nax_verify import register_qwen38_qmv_dispatch

        register_qwen38_qmv_dispatch(_row9_paired_qmv_dispatch)
        _ROW9_PATCH["installed"] = True
    if _ROW9_PATCH["installed"]:
        _ROW9_PATCH["enabled"] = bool(active)
        _ROW9_PATCH["modules"] = (
            {
                id(module): weakref.ref(module)
                for _, module in model.named_modules()
            }
            if active
            else {}
        )
    return {
        "installed": bool(_ROW9_PATCH["installed"]),
        "active": bool(_ROW9_PATCH["enabled"]),
        "bound_modules": len(_ROW9_PATCH["modules"]),
    }


def reset_qwen38_row9_qmv_calls() -> None:
    global qwen38_row9_qmv_calls

    qwen38_row9_qmv_calls = 0


def qwen38_row9_qmv_counter_snapshot() -> int:
    return int(qwen38_row9_qmv_calls)


def qwen38_dual_rms_norm_concat(
    a: Any,
    b: Any,
    a_weight: Any,
    b_weight: Any,
    eps: float,
) -> Any:
    """Normalize two hidden-5120 BF16 rows and emit one contiguous concat."""

    import mlx.core as mx

    global _DUAL_RMS_CONCAT_KERNEL, qwen38_dual_norm_calls
    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != mx.bfloat16
        or b.dtype != mx.bfloat16
        or int(a.shape[-1]) != 5120
        or tuple(a_weight.shape) != (5120,)
        or tuple(b_weight.shape) != (5120,)
        or a_weight.dtype != mx.bfloat16
        or b_weight.dtype != mx.bfloat16
    ):
        raise ValueError("unsupported Qwen 3.8 dual RMSNorm contract")
    rows = int(a.size) // 5120
    if _DUAL_RMS_CONCAT_KERNEL is None:
        _DUAL_RMS_CONCAT_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_dual_rms_norm_concat_bf16_v1",
            input_names=["a", "b", "a_weight", "b_weight", "eps"],
            output_names=["concat_out"],
            source=r"""
                constexpr uint n_reads = 4;
                constexpr uint simd_size = 32;
                constexpr uint lsize = 1024;

                uint row = threadgroup_position_in_grid.x;
                uint thread_id = thread_position_in_threadgroup.x;
                uint simd_thread = thread_index_in_simdgroup;
                uint simd_group = simdgroup_index_in_threadgroup;
                uint axis_size = uint(a_shape[a_ndim - 1]);
                uint a_rows = 1;
                for (uint i = 0; i + 1 < a_ndim; ++i) {
                    a_rows *= uint(a_shape[i]);
                }
                bool is_a = row < a_rows;
                uint local_row = is_a ? row : row - a_rows;
                ulong in_off = ulong(local_row) * ulong(axis_size);
                ulong out_off = ulong(local_row) * ulong(axis_size * 2)
                    + (is_a ? 0 : ulong(axis_size));

                threadgroup float local_inv_mean[1];
                threadgroup float local_sums[simd_size];
                float acc = 0.0f;
                for (uint r_start = 0; r_start < axis_size;
                     r_start += lsize * n_reads) {
                    uint elem = r_start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        if (elem + i < axis_size) {
                            float xi = is_a
                                ? float(a[in_off + elem + i])
                                : float(b[in_off + elem + i]);
                            acc += xi * xi;
                        }
                    }
                }
                acc = simd_sum(acc);
                if (simd_group == 0) {
                    local_sums[simd_thread] = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_thread == 0) {
                    local_sums[simd_group] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group == 0) {
                    acc = simd_sum(local_sums[simd_thread]);
                    if (simd_thread == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(
                            acc / float(axis_size) + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float inv_mean = local_inv_mean[0];
                for (uint r_start = 0; r_start < axis_size;
                     r_start += lsize * n_reads) {
                    uint elem = r_start + thread_id * n_reads;
                    for (uint i = 0; i < n_reads; ++i) {
                        if (elem + i < axis_size) {
                            float xi = is_a
                                ? float(a[in_off + elem + i])
                                : float(b[in_off + elem + i]);
                            bfloat wi = is_a
                                ? a_weight[elem + i] : b_weight[elem + i];
                            concat_out[out_off + elem + i] =
                                wi * bfloat(xi * inv_mean);
                        }
                    }
                }
            """,
            ensure_row_contiguous=True,
        )
    qwen38_dual_norm_calls += 1
    (output,) = _DUAL_RMS_CONCAT_KERNEL(
        inputs=[a, b, a_weight, b_weight, float(eps)],
        template=[],
        grid=(2 * rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(*a.shape[:-1], 10240)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def reset_qwen38_dual_norm_calls() -> None:
    global qwen38_dual_norm_calls

    qwen38_dual_norm_calls = 0


def qwen38_dual_norm_counter_snapshot() -> int:
    return int(qwen38_dual_norm_calls)
