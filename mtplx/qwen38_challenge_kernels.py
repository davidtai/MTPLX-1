"""Retained target-shaped kernels from the pinned Qwen 3.8 challenge."""

from __future__ import annotations

import weakref
from typing import Any

_QMV_INPUTS_PER_GROUP = {
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 3,
    7: 4,
    8: 4,
    9: 3,
}
_QMV_REPLICA_KERNEL = None
_QMV_TABLE_KERNEL = None
_QMV_XSUMS_KERNEL = None
_DUAL_RMS_CONCAT_KERNEL = None
_QMV_PATCH: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "modules": {},
}
qwen38_qmv_counts: dict[str, int] = {}
qwen38_dual_norm_calls = 0


def qwen38_active_input_groups(width: int) -> int:
    """Return the exact live threadgroup count for a supported verify width."""

    try:
        inputs_per_group = _QMV_INPUTS_PER_GROUP[int(width)]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen 3.8 QMV width: {width}") from exc
    return (int(width) + inputs_per_group - 1) // inputs_per_group


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


_QMV_HEADER = r"""
    template <int NA, bool USE_TABLE>
    inline void qwen38_qmv_wide(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int group_size,
        const int sums_stride,
        int first_m,
        int out_row,
        uint simd_lid
    ) {
        typedef vec<float, NA> VF;
        constexpr int rows_per_simd = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int bytes_per_lane = 8;
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / group_size;

        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }
        for (int k = 0; k < in_vec_size; k += block_size) {
            thread uint16_t packed[rows_per_simd][4];
            thread float scale_local[rows_per_simd];
            thread float bias_local[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) {
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * in_vec_size_w + k / 2 +
                        simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) {
                    packed[r][i] = ws[i];
                }
                const int group_index = row * in_vec_size_g
                    + k / group_size
                    + int(simd_lid) / (group_size / values_per_thread);
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
            }

            VF sums = VF(0.0f);
            if (USE_TABLE) {
                const device float* st =
                    xsums + ((k / block_size) * 32 + int(simd_lid)) *
                    sums_stride + first_m;
                for (int m = 0; m < NA; m++) {
                    sums[m] = st[m];
                }
            }
            VF partial[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) {
                partial[r] = VF(0.0f);
            }
            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const vec<bfloat16_t, 4> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 4>*>(xm);
                    a0[m] = static_cast<float>(xv[0]);
                    a1[m] = static_cast<float>(xv[1]);
                    a2[m] = static_cast<float>(xv[2]);
                    a3[m] = static_cast<float>(xv[3]);
                    if (!USE_TABLE) {
                        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                    }
                }
                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] += (a0 * (packed[r][i] & 0x000f) +
                                   a1 * ((packed[r][i] >> 4) & 0x000f) +
                                   a2 * ((packed[r][i] >> 8) & 0x000f) +
                                   a3 * ((packed[r][i] >> 12) & 0x000f));
                }
            }
            for (int r = 0; r < rows_per_simd; r++) {
                acc[r] += scale_local[r] * partial[r]
                    + sums * bias_local[r];
            }
        }
        for (int r = 0; r < rows_per_simd; r++) {
            for (int m = 0; m < NA; m++) {
                const float reduced = simd_sum(acc[r][m]);
                if (simd_lid == 0) {
                    y[(first_m + m) * out_vec_size + out_row + r] =
                        static_cast<bfloat16_t>(reduced);
                }
            }
        }
    }

    template <int M, int IPG, bool USE_TABLE>
    inline void qwen38_qmv_m(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int group_size,
        const int sums_stride,
        int group_x,
        int out_row,
        uint simd_lid
    ) {
        static_assert(M % IPG != 1, "a one-input tail group is not built");
        constexpr int TAIL = M % IPG;
        const int first_m = group_x * IPG;
        if (first_m >= M) {
            return;
        }
        if (TAIL == 0 || M - first_m >= IPG) {
            qwen38_qmv_wide<IPG, USE_TABLE>(
                w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
                group_size, sums_stride, first_m, out_row, simd_lid);
        } else {
            qwen38_qmv_wide<(TAIL >= 2 ? TAIL : 2), USE_TABLE>(
                w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
                group_size, sums_stride, first_m, out_row, simd_lid);
        }
    }
"""


def _qmv_source(*, table: bool) -> str:
    sums = "xsums" if table else "qmv_null_sums"
    flag = "true" if table else "false"
    null_declaration = (
        "" if table else "const device float* qmv_null_sums = nullptr;"
    )
    cases = "\n".join(
        f"""
            case {width}:
                qwen38_qmv_m<{width}, {inputs_per_group}, {flag}>(
                    w, scales, biases, x, {sums}, y,
                    qmv_k, qmv_n, qmv_group_size, qmv_stride,
                    qmv_gx, qmv_out_row, qmv_lid);
                break;
        """
        for width, inputs_per_group in _QMV_INPUTS_PER_GROUP.items()
    )
    return f"""
        const int qmv_m = x_shape[x_ndim - 2];
        const int qmv_k = x_shape[x_ndim - 1];
        const int qmv_n = w_shape[0];
        const int qmv_stride = qmv_m <= 8 ? 8 : 16;
        const int qmv_group_size = group_size;
        const uint3 qmv_tid = threadgroup_position_in_grid;
        const uint qmv_lid = thread_index_in_simdgroup;
        const uint qmv_sgid = simdgroup_index_in_threadgroup;
        const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
        const int qmv_gx = int(qmv_tid.x);
        {null_declaration}
        switch (qmv_m) {{
            {cases}
            default:
                break;
        }}
    """


def _qmv_xsums(x: Any) -> Any:
    import mlx.core as mx

    global _QMV_XSUMS_KERNEL
    width, hidden = map(int, x.shape)
    stride = 8 if width <= 8 else 16
    k_blocks = hidden // 512
    if _QMV_XSUMS_KERNEL is None:
        _QMV_XSUMS_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_affine4_xsums_v2",
            input_names=["x"],
            output_names=["xsums"],
            source=r"""
                const int xs_m = x_shape[x_ndim - 2];
                const int xs_k = x_shape[x_ndim - 1];
                const int xs_stride = xs_m <= 8 ? 8 : 16;
                const uint3 xs_gid = thread_position_in_grid;
                const int xs_lane = int(xs_gid.x);
                const int xs_kb = int(xs_gid.y);
                const int xs_row = int(xs_gid.z);
                const device bfloat16_t* xm =
                    x + xs_row * xs_k + xs_kb * 512 + xs_lane * 16;
                float s = 0.0f;
                for (int i = 0; i < 4; i++) {
                    const vec<bfloat16_t, 4> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
                            xm + 4 * i);
                    s += xv[0] + xv[1] + xv[2] + xv[3];
                }
                xsums[(xs_kb * 32 + xs_lane) * xs_stride + xs_row] = s;
            """,
            ensure_row_contiguous=True,
        )
    (xsums,) = _QMV_XSUMS_KERNEL(
        inputs=[x],
        template=[],
        grid=(32, k_blocks, width),
        threadgroup=(32, 1, 1),
        output_shapes=[(k_blocks * 32 * stride,)],
        output_dtypes=[mx.float32],
    )
    return xsums


def qwen38_affine4_qmv(
    x: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    *,
    group_size: int,
) -> Any:
    """Run the final affine-4 QMV for group 32/64 and widths 2 through 9."""

    import mlx.core as mx

    global _QMV_REPLICA_KERNEL, _QMV_TABLE_KERNEL
    if x.ndim != 2:
        raise ValueError("Qwen 3.8 QMV input must be a two-dimensional row batch")
    width, hidden = map(int, x.shape)
    output = int(weight.shape[0])
    group_size = int(group_size)
    qwen38_active_input_groups(width)
    if (
        x.dtype != mx.bfloat16
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or group_size not in (32, 64)
        or int(weight.shape[1]) != hidden // 8
        or tuple(scales.shape) != (output, hidden // group_size)
        or tuple(biases.shape) != tuple(scales.shape)
        or hidden % 512
        or output < 4096
        or output % 8
    ):
        raise ValueError("unsupported Qwen 3.8 affine-4 QMV contract")
    use_table = width >= 4
    if use_table:
        if _QMV_TABLE_KERNEL is None:
            _QMV_TABLE_KERNEL = mx.fast.metal_kernel(
                name="mtplx_qwen38_affine4_qmv_sums_v2",
                input_names=[
                    "w", "scales", "biases", "x", "xsums", "group_size"
                ],
                output_names=["y"],
                header=_QMV_HEADER,
                source=_qmv_source(table=True),
                ensure_row_contiguous=True,
            )
        kernel = _QMV_TABLE_KERNEL
        inputs = [weight, scales, biases, x, _qmv_xsums(x), group_size]
    else:
        if _QMV_REPLICA_KERNEL is None:
            _QMV_REPLICA_KERNEL = mx.fast.metal_kernel(
                name="mtplx_qwen38_affine4_qmv_wide_v2",
                input_names=["w", "scales", "biases", "x", "group_size"],
                output_names=["y"],
                header=_QMV_HEADER,
                source=_qmv_source(table=False),
                ensure_row_contiguous=True,
            )
        kernel = _QMV_REPLICA_KERNEL
        inputs = [weight, scales, biases, x, group_size]
    (result,) = kernel(
        inputs=inputs,
        template=[],
        grid=(qwen38_active_input_groups(width) * 32, (output // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(width, output)],
        output_dtypes=[mx.bfloat16],
    )
    return result


def reset_qwen38_qmv_counts() -> None:
    qwen38_qmv_counts.clear()


def qwen38_qmv_counter_snapshot() -> dict[str, int]:
    return dict(sorted(qwen38_qmv_counts.items()))


def reset_qwen38_dual_norm_calls() -> None:
    global qwen38_dual_norm_calls

    qwen38_dual_norm_calls = 0


def qwen38_dual_norm_counter_snapshot() -> int:
    return int(qwen38_dual_norm_calls)


def _qmv_route_active() -> bool:
    from .attention_context import (
        current_attention_phase,
        current_model_forward_kind,
    )

    return (
        current_attention_phase() == "decode_verify"
        and current_model_forward_kind() == "target_verify"
    )


def _qwen38_qmv_dispatch(self: Any, x: Any, width: int) -> Any | None:
    import mlx.core as mx

    if not _QMV_PATCH["enabled"] or not _qmv_route_active():
        return None
    owner_ref = _QMV_PATCH["modules"].get(id(self))
    if owner_ref is None or owner_ref() is not self:
        return None
    group_size = int(getattr(self, "group_size", 0) or 0)
    if (
        not 2 <= int(width) <= 9
        or int(getattr(self, "bits", 0) or 0) != 4
        or group_size not in (32, 64)
        or str(getattr(self, "mode", "")).lower() != "affine"
        or x.dtype != mx.bfloat16
    ):
        return None
    weight = self["weight"]
    hidden = int(x.shape[-1])
    output = int(weight.shape[0])
    if (
        hidden % 512 != 0
        or output < 4096
        or output % 8 != 0
        or int(weight.shape[1]) != hidden // 8
    ):
        return None
    key = f"g{group_size}_m{int(width)}"
    qwen38_qmv_counts[key] = qwen38_qmv_counts.get(key, 0) + 1
    y = qwen38_affine4_qmv(
        x.reshape(int(width), hidden),
        weight,
        self["scales"],
        self["biases"],
        group_size=group_size,
    ).reshape(*x.shape[:-1], output)
    if "bias" in self:
        y = y + self["bias"]
    return y


def configure_qwen38_final_qmv(*, active: bool, model: Any) -> dict[str, Any]:
    """Register once and bind dispatch to one exact Qwen model instance."""

    if active and not _QMV_PATCH["installed"]:
        from .nax_verify import register_qwen38_qmv_dispatch

        try:
            register_qwen38_qmv_dispatch(_qwen38_qmv_dispatch)
        except RuntimeError:
            pass
        else:
            _QMV_PATCH["installed"] = True
    if _QMV_PATCH["installed"]:
        _QMV_PATCH["enabled"] = bool(active)
        _QMV_PATCH["modules"] = (
            {
                id(module): weakref.ref(module)
                for _, module in model.named_modules()
            }
            if active
            else {}
        )
    return {
        "installed": bool(_QMV_PATCH["installed"]),
        "active": bool(_QMV_PATCH["enabled"]),
        "bound_modules": len(_QMV_PATCH["modules"]),
    }
