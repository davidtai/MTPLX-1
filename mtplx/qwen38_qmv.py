"""Rows 70/78/80 Q4-group64 wide-QMV candidate for Qwen 3.8 MTP."""

from __future__ import annotations

from typing import Any

_QMV_REPLICA_KERNEL = None
_QMV_TABLE_KERNEL = None
_QMV_XSUMS_KERNEL = None
_ORIGINAL_QUANTIZED_LINEAR_CALL = None
QWEN38_QMV_CALLS = {width: 0 for width in range(2, 10)}

_INPUTS_PER_GROUP = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}


def qwen38_qmv_active_input_groups(width: int) -> int:
    inputs_per_group = _INPUTS_PER_GROUP[int(width)]
    return (int(width) + inputs_per_group - 1) // inputs_per_group


def qwen38_qmv_counter_snapshot() -> dict[str, int]:
    return {f"m{width}": int(calls) for width, calls in QWEN38_QMV_CALLS.items()}


_QMV_HEADER = r"""
    template <int NA, bool USE_TABLE>
    inline void qwen_qmv_wide(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
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
        const int in_vec_size_g = in_vec_size / 64;

        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) acc[r] = VF(0.0f);

        for (int k = 0; k < in_vec_size; k += block_size) {
            thread uint16_t packed[rows_per_simd][4];
            thread float scale_local[rows_per_simd];
            thread float bias_local[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) {
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w)
                        + row * in_vec_size_w + k / 2
                        + simd_lid * bytes_per_lane);
                for (int i = 0; i < 4; i++) packed[r][i] = ws[i];
                const int group_index =
                    row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
            }

            VF sums = VF(0.0f);
            if (USE_TABLE) {
                const device float* st =
                    xsums + ((k / block_size) * 32 + int(simd_lid))
                    * sums_stride + first_m;
                for (int m = 0; m < NA; m++) sums[m] = st[m];
            }
            VF partial[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) partial[r] = VF(0.0f);
            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k
                        + simd_lid * values_per_thread + 4 * i;
                    const vec<bfloat16_t, 4> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 4>*>(xm);
                    a0[m] = static_cast<float>(xv[0]);
                    a1[m] = static_cast<float>(xv[1]);
                    a2[m] = static_cast<float>(xv[2]);
                    a3[m] = static_cast<float>(xv[3]);
                    if (!USE_TABLE) sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                }
                for (int r = 0; r < rows_per_simd; r++) {
                    partial[r] +=
                        a0 * (packed[r][i] & 0x000f)
                        + a1 * ((packed[r][i] >> 4) & 0x000f)
                        + a2 * ((packed[r][i] >> 8) & 0x000f)
                        + a3 * ((packed[r][i] >> 12) & 0x000f);
                }
            }
            for (int r = 0; r < rows_per_simd; r++) {
                acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
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
    inline void qwen_qmv_m(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        const int in_vec_size,
        const int out_vec_size,
        const int sums_stride,
        int group_x,
        int out_row,
        uint simd_lid
    ) {
        static_assert(M % IPG != 1, "a one-input tail group is not built");
        constexpr int TAIL = M % IPG;
        const int first_m = group_x * IPG;
        if (first_m >= M) return;
        if (TAIL == 0 || M - first_m >= IPG) {
            qwen_qmv_wide<IPG, USE_TABLE>(
                w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
                sums_stride, first_m, out_row, simd_lid);
        } else {
            qwen_qmv_wide<(TAIL >= 2 ? TAIL : 2), USE_TABLE>(
                w, scales, biases, x, xsums, y, in_vec_size, out_vec_size,
                sums_stride, first_m, out_row, simd_lid);
        }
    }
"""


def _qmv_source(*, use_table: bool) -> str:
    sums = "xsums" if use_table else "qmv_null_sums"
    flag = "true" if use_table else "false"
    null_decl = "" if use_table else "const device float* qmv_null_sums = nullptr;"
    cases = "\n".join(
        f"""
            case {width}:
                qwen_qmv_m<{width}, {inputs_per_group}, {flag}>(
                    w, scales, biases, x, {sums}, y, qmv_k, qmv_n,
                    qmv_stride, qmv_gx, qmv_out_row, qmv_lid);
                break;
        """
        for width, inputs_per_group in _INPUTS_PER_GROUP.items()
    )
    return f"""
        const int qmv_m = x_shape[x_ndim - 2];
        const int qmv_k = x_shape[x_ndim - 1];
        const int qmv_n = w_shape[0];
        const int qmv_stride = qmv_m <= 8 ? 8 : 16;
        const uint3 qmv_tid = threadgroup_position_in_grid;
        const uint qmv_lid = thread_index_in_simdgroup;
        const uint qmv_sgid = simdgroup_index_in_threadgroup;
        const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
        const int qmv_gx = int(qmv_tid.x);
        {null_decl}
        switch (qmv_m) {{
            {cases}
            default: break;
        }}
    """


def _kernels() -> tuple[Any, Any, Any]:
    import mlx.core as mx

    global _QMV_REPLICA_KERNEL, _QMV_TABLE_KERNEL, _QMV_XSUMS_KERNEL
    if _QMV_REPLICA_KERNEL is None:
        _QMV_REPLICA_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_q4_g64_qmv_wide_v1",
            input_names=["w", "scales", "biases", "x"],
            output_names=["y"],
            source=_qmv_source(use_table=False),
            header=_QMV_HEADER,
            ensure_row_contiguous=False,
        )
        _QMV_TABLE_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_q4_g64_qmv_wide_sums_v1",
            input_names=["w", "scales", "biases", "x", "xsums"],
            output_names=["y"],
            source=_qmv_source(use_table=True),
            header=_QMV_HEADER,
            ensure_row_contiguous=False,
        )
        _QMV_XSUMS_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_q4_g64_xsums_v1",
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
            ensure_row_contiguous=False,
        )
    return _QMV_REPLICA_KERNEL, _QMV_TABLE_KERNEL, _QMV_XSUMS_KERNEL


def qwen38_qmv(linear: Any, x: Any) -> Any | None:
    """Return the routed result or ``None`` when the stock QMM must run."""

    import mlx.core as mx

    if not bool(getattr(linear, "_mtplx_qwen38_qmv_active", False)):
        return None
    min_width = int(getattr(linear, "_mtplx_qwen38_qmv_min_width", 3))
    active_groups = bool(getattr(linear, "_mtplx_qwen38_qmv_active_groups", False))
    allowed_widths = tuple(
        int(width)
        for width in getattr(linear, "_mtplx_qwen38_qmv_allowed_widths", ())
    )
    use_sum_table = bool(
        getattr(linear, "_mtplx_qwen38_qmv_use_table", True)
    )
    min_output_size = int(
        getattr(linear, "_mtplx_qwen38_qmv_min_output_size", 4096)
    )
    weight = linear.weight
    scales = linear.scales
    biases = linear.biases
    k = int(x.shape[-1])
    n = int(weight.shape[0])
    m = int(x.size) // k
    if (
        int(linear.bits) != 4
        or int(linear.group_size) != 64
        or str(linear.mode) != "affine"
        or getattr(linear, "bias", None) is not None
        or biases is None
        or x.dtype != mx.bfloat16
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or m < min_width
        or (allowed_widths and m not in allowed_widths)
        or m not in _INPUTS_PER_GROUP
        or int(x.shape[-2]) != m
        or tuple(weight.shape[1:]) != (k // 8,)
        or k % 512 != 0
        or n % 8 != 0
        or n < min_output_size
    ):
        return None

    # MLX Python does not expose strides. Keep these copies explicit so an
    # enclosing mx.compile graph captures them instead of asking metal_kernel
    # to inject an implicit row-contiguous primitive.
    x = mx.contiguous(x)
    weight = mx.contiguous(weight)
    scales = mx.contiguous(scales)
    biases = mx.contiguous(biases)
    replica, table, xsums_kernel = _kernels()
    output_shape = (*x.shape[:-1], n)
    grid_groups = qwen38_qmv_active_input_groups(m) if active_groups else m
    common = {
        "template": [],
        "grid": (grid_groups * 32, (n // 8) * 2, 1),
        "threadgroup": (32, 2, 1),
        "output_shapes": [output_shape],
        "output_dtypes": [mx.bfloat16],
    }
    if m >= 4 and use_sum_table:
        stride = 8 if m <= 8 else 16
        (xsums,) = xsums_kernel(
            inputs=[x],
            template=[],
            grid=(32, k // 512, m),
            threadgroup=(32, 1, 1),
            output_shapes=[(k // 512 * 32 * stride,)],
            output_dtypes=[mx.float32],
        )
        (result,) = table(inputs=[weight, scales, biases, x, xsums], **common)
    else:
        (result,) = replica(inputs=[weight, scales, biases, x], **common)
    QWEN38_QMV_CALLS[m] += 1
    return result


def configure_qwen38_qmv(
    model: Any,
    *,
    active: bool,
    min_width: int = 3,
    active_groups: bool = False,
) -> dict[str, int]:
    """Toggle the candidate only on Q4/group64 linears inside the MTP block."""

    import mlx.nn as nn

    global _ORIGINAL_QUANTIZED_LINEAR_CALL
    if _ORIGINAL_QUANTIZED_LINEAR_CALL is None:
        _ORIGINAL_QUANTIZED_LINEAR_CALL = nn.QuantizedLinear.__call__

        def routed_call(self, x):
            result = qwen38_qmv(self, x)
            if result is not None:
                return result
            return _ORIGINAL_QUANTIZED_LINEAR_CALL(self, x)

        nn.QuantizedLinear.__call__ = routed_call

    text = getattr(model, "language_model", model)
    mtp = getattr(text, "mtp", None)
    eligible = 0
    for module in list(getattr(mtp, "modules", lambda: [])()):
        if not isinstance(module, nn.QuantizedLinear):
            continue
        is_eligible = bool(
            int(getattr(module, "bits", 0)) == 4
            and int(getattr(module, "group_size", 0)) == 64
            and str(getattr(module, "mode", "")) == "affine"
        )
        module._mtplx_qwen38_qmv_active = bool(active and is_eligible)
        module._mtplx_qwen38_qmv_min_width = int(min_width)
        module._mtplx_qwen38_qmv_allowed_widths = ()
        module._mtplx_qwen38_qmv_active_groups = bool(active_groups)
        module._mtplx_qwen38_qmv_use_table = True
        module._mtplx_qwen38_qmv_min_output_size = 4096
        eligible += int(is_eligible)
    return {
        "eligible_modules": eligible,
        "active_modules": eligible if active else 0,
        "min_width": int(min_width) if active else 0,
        "active_groups": int(bool(active and active_groups)),
    }


def configure_qwen38_dflash_qmv(
    draft_model: Any,
    *,
    active: bool,
    allowed_widths: tuple[int, ...],
) -> dict[str, Any]:
    """Route only selected DFlash draft block widths through the source QMV."""

    import mlx.nn as nn

    global _ORIGINAL_QUANTIZED_LINEAR_CALL
    if _ORIGINAL_QUANTIZED_LINEAR_CALL is None:
        _ORIGINAL_QUANTIZED_LINEAR_CALL = nn.QuantizedLinear.__call__

        def routed_call(self, x):
            result = qwen38_qmv(self, x)
            if result is not None:
                return result
            return _ORIGINAL_QUANTIZED_LINEAR_CALL(self, x)

        nn.QuantizedLinear.__call__ = routed_call

    widths = tuple(int(width) for width in allowed_widths)
    if tuple(sorted(set(widths))) != widths or any(
        width not in _INPUTS_PER_GROUP for width in widths
    ):
        raise ValueError("DFlash QMV widths must be unique, chronological, and in 2..9")
    eligible = 0
    for module in list(getattr(draft_model, "modules", lambda: [])()):
        if not isinstance(module, nn.QuantizedLinear):
            continue
        is_eligible = bool(
            int(getattr(module, "bits", 0)) == 4
            and int(getattr(module, "group_size", 0)) == 64
            and str(getattr(module, "mode", "")) == "affine"
        )
        module._mtplx_qwen38_qmv_active = bool(active and is_eligible)
        module._mtplx_qwen38_qmv_min_width = min(widths, default=10)
        module._mtplx_qwen38_qmv_allowed_widths = widths
        module._mtplx_qwen38_qmv_active_groups = bool(active)
        module._mtplx_qwen38_qmv_use_table = False
        module._mtplx_qwen38_qmv_min_output_size = 0
        private_dispatch = getattr(module, "_call_fn", None)
        if callable(private_dispatch) and not hasattr(
            module, "_mtplx_qwen38_qmv_original_call_fn"
        ):
            object.__setattr__(
                module,
                "_mtplx_qwen38_qmv_original_call_fn",
                private_dispatch,
            )

            def routed_private_call(x, *, _module=module, _stock=private_dispatch):
                result = qwen38_qmv(_module, x)
                return _stock(x) if result is None else result

            object.__setattr__(module, "_call_fn", routed_private_call)
        eligible += int(is_eligible)
    return {
        "eligible_modules": eligible,
        "active_modules": eligible if active else 0,
        "allowed_widths": list(widths) if active else [],
    }
