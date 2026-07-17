"""Standalone precise/fast Qwen SwiGLU kernels for experimental A/Bs."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _qwen_swiglu_source(*, fast: bool) -> str:
    exp_call = "metal::fast::exp" if fast else "metal::exp"
    return f"""
        using namespace metal;

        uint vec_idx = thread_position_in_grid.x;
        int vec_count = int(size) / 8;
        if (int(vec_idx) >= vec_count) {{
            return;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *gate_v = (const device Vec8*)gate;
        const device Vec8 *up_v = (const device Vec8*)up;
        device Vec8 *out_v = (device Vec8*)out;
        Vec8 gate_values = gate_v[vec_idx];
        Vec8 up_values = up_v[vec_idx];
        Vec8 result;
        _Pragma("unroll")
        for (int i = 0; i < 8; ++i) {{
            T gate_value = gate_values[i];
            T up_value = up_values[i];
            auto sigmoid_tail = 1 / (
                1 + {exp_call}(metal::abs(gate_value))
            );
            T sigmoid_value = (gate_value < T(0))
                ? T(sigmoid_tail)
                : T(1 - sigmoid_tail);
            result[i] = T((gate_value * sigmoid_value) * up_value);
        }}
        out_v[vec_idx] = result;
    """


@lru_cache(maxsize=None)
def _qwen_swiglu_kernel(dtype: mx.Dtype, fast: bool):
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    mode = "fast" if fast else "precise"
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen_swiglu_vec8_{mode}_{dtype_tag}",
        input_names=["gate", "up", "size"],
        output_names=["out"],
        source=_qwen_swiglu_source(fast=fast),
    )


def qwen_swiglu(gate: mx.array, up: mx.array, *, fast: bool) -> mx.array:
    """Apply standalone Qwen SwiGLU, falling back when Vec8 is ineligible."""
    from mlx_lm.models.qwen3_next import swiglu

    if (
        not mx.metal.is_available()
        or gate.dtype not in (mx.bfloat16, mx.float16)
        or gate.dtype != up.dtype
        or tuple(gate.shape) != tuple(up.shape)
    ):
        return swiglu(gate, up)
    size = int(gate.size)
    if size <= 0 or size % 8:
        return swiglu(gate, up)

    gate_c = mx.contiguous(gate)
    up_c = mx.contiguous(up)
    kernel = _qwen_swiglu_kernel(gate.dtype, bool(fast))
    vectors = size // 8
    threads = ((vectors + 255) // 256) * 256
    (out,) = kernel(
        inputs=[gate_c, up_c, size],
        template=[("T", gate.dtype)],
        grid=(threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(gate.shape)],
        output_dtypes=[gate.dtype],
    )
    return out
