"""Retained target-shaped kernels from the pinned Qwen 3.8 challenge."""

from __future__ import annotations

from typing import Any

_DUAL_RMS_CONCAT_KERNEL = None
qwen38_dual_norm_calls = 0
qwen38_row18_mlp_gate_up_calls = 0
qwen38_row24_eval_ladder_calls = 0
qwen38_row26_prefill_ladder_calls = 0
_QWEN38_MLP_ORIGINAL_CALL = None


def _qwen38_pack_gate_up(mlp: Any) -> dict[str, Any] | None:
    import mlx.core as mx
    import mlx.nn as nn

    gate, up = mlp.gate_proj, mlp.up_proj
    if isinstance(gate, nn.QuantizedLinear) and isinstance(up, nn.QuantizedLinear):
        if not (
            int(gate.bits) == int(up.bits)
            and int(gate.group_size) == int(up.group_size)
            and str(gate.mode) == str(up.mode) == "affine"
            and tuple(gate.weight.shape[1:]) == tuple(up.weight.shape[1:])
            and tuple(gate.scales.shape[1:]) == tuple(up.scales.shape[1:])
            and tuple(gate.biases.shape[1:]) == tuple(up.biases.shape[1:])
        ):
            return None
        payload = {
            "kind": "quantized",
            "weight": mx.concatenate([gate.weight, up.weight], axis=0),
            "scales": mx.concatenate([gate.scales, up.scales], axis=0),
            "biases": mx.concatenate([gate.biases, up.biases], axis=0),
            "split_at": int(gate.weight.shape[0]),
            "group_size": int(gate.group_size),
            "bits": int(gate.bits),
            "mode": str(gate.mode),
        }
    elif (
        isinstance(gate, nn.Linear)
        and isinstance(up, nn.Linear)
        and "bias" not in gate
        and "bias" not in up
        and tuple(gate.weight.shape[1:]) == tuple(up.weight.shape[1:])
    ):
        payload = {
            "kind": "dense",
            "weight": mx.concatenate([gate.weight, up.weight], axis=0),
            "split_at": int(gate.weight.shape[0]),
        }
    else:
        return None
    mx.eval(*(value for value in payload.values() if isinstance(value, mx.array)))
    return payload


def configure_qwen38_row18_mlp_gate_up(model: Any, *, active: bool) -> dict[str, int]:
    """Toggle the row-18 S<=9 packed gate/up MLP projection."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import MLP

    global _QWEN38_MLP_ORIGINAL_CALL, qwen38_row18_mlp_gate_up_calls
    if _QWEN38_MLP_ORIGINAL_CALL is None:
        _QWEN38_MLP_ORIGINAL_CALL = MLP.__call__

        def packed_mlp_call(self, x):
            global qwen38_row18_mlp_gate_up_calls
            payload = getattr(self, "_mtplx_qwen38_row18_gate_up", None)
            if (
                not bool(getattr(self, "_mtplx_qwen38_row18_gate_up_active", False))
                or payload is None
                or int(x.shape[-2]) > 9
            ):
                return _QWEN38_MLP_ORIGINAL_CALL(self, x)
            if payload["kind"] == "quantized":
                fused = mx.quantized_matmul(
                    x,
                    payload["weight"],
                    scales=payload["scales"],
                    biases=payload["biases"],
                    transpose=True,
                    group_size=payload["group_size"],
                    bits=payload["bits"],
                    mode=payload["mode"],
                )
            else:
                fused = x @ payload["weight"].T
            gate, up = mx.split(fused, [payload["split_at"]], axis=-1)
            qwen38_row18_mlp_gate_up_calls += 1
            return self.down_proj(nn.silu(gate) * up)

        MLP.__call__ = packed_mlp_call

    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", text)
    layers = list(getattr(inner, "layers", ()) or ())
    mtp = getattr(text, "mtp", None)
    layers.extend(list(getattr(mtp, "layers", ()) or ()))
    eligible = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, MLP):
            continue
        payload = getattr(mlp, "_mtplx_qwen38_row18_gate_up", None)
        if payload is None:
            payload = _qwen38_pack_gate_up(mlp)
            if payload is not None:
                mlp._mtplx_qwen38_row18_gate_up = payload
        is_eligible = payload is not None
        mlp._mtplx_qwen38_row18_gate_up_active = bool(active and is_eligible)
        eligible += int(is_eligible)
    return {"eligible_modules": eligible, "active_modules": eligible if active else 0}


def qwen38_row18_mlp_counter_snapshot() -> int:
    return int(qwen38_row18_mlp_gate_up_calls)


def qwen38_row24_async_eval(value: Any, *, row26: bool = False) -> None:
    import mlx.core as mx

    global qwen38_row24_eval_ladder_calls, qwen38_row26_prefill_ladder_calls
    qwen38_row24_eval_ladder_calls += 1
    if row26:
        qwen38_row26_prefill_ladder_calls += 1
    mx.async_eval(value)


def qwen38_row24_eval_ladder_counter_snapshot() -> int:
    return int(qwen38_row24_eval_ladder_calls)


def qwen38_row26_prefill_ladder_counter_snapshot() -> int:
    return int(qwen38_row26_prefill_ladder_calls)


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
