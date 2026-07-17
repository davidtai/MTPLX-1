"""Experimental no-copy Qwen small-M gate/up fusion on ``vk_k`` geometry.

The default runtime remains unchanged.  When explicitly selected, this keeps
the two original Q4 affine weight arrays and the two-way split-K ownership used
by the current Qwen verifier, compiling one kernel per exact verify tile
(M2-M6).  Owning the true row count keeps the whole gate/up + SwiGLU chain in
registers and threadgroup memory, so no zero-padded activation copy is
materialized on any call.  The only fused boundary is gate projection + up
projection + precise SwiGLU; down projection remains on the existing path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _fused_q4_pack_block(m: int, suffix: str, tile_cols: int) -> str:
    """Emit one shared-x Q4 pack for the gate and up projections."""
    pack = f"pack{suffix}"
    lines = [
        f"int k_base{suffix} = {pack} * 8;",
        f"int gi{suffix} = k_base{suffix} / GS;",
    ]
    for row in range(m):
        lines.append(f"Vec8 v{suffix}_{row} = xv[({row} * K + k_base{suffix}) / 8];")
    for col in range(tile_cols):
        lines.extend(
            [
                f"uint32_t gate_p{suffix}_{col} = "
                f"gate_w_q[(n0 + {col}) * K_by_p + {pack}];",
                f"uint32_t up_p{suffix}_{col} = "
                f"up_w_q[(n0 + {col}) * K_by_p + {pack}];",
                f"float gate_s{suffix}_{col} = float("
                f"gate_scales[(n0 + {col}) * K_by_gs + gi{suffix}]);",
                f"float gate_b{suffix}_{col} = float("
                f"gate_biases[(n0 + {col}) * K_by_gs + gi{suffix}]);",
                f"float up_s{suffix}_{col} = float("
                f"up_scales[(n0 + {col}) * K_by_gs + gi{suffix}]);",
                f"float up_b{suffix}_{col} = float("
                f"up_biases[(n0 + {col}) * K_by_gs + gi{suffix}]);",
            ]
        )
    for col in range(tile_cols):
        lines.extend(
            [
                "{",
                f"    uint32_t gate_packed = gate_p{suffix}_{col};",
                f"    uint32_t up_packed = up_p{suffix}_{col};",
                f"    float gate_scale = gate_s{suffix}_{col};",
                f"    float gate_bias = gate_b{suffix}_{col};",
                f"    float up_scale = up_s{suffix}_{col};",
                f"    float up_bias = up_b{suffix}_{col};",
                '    _Pragma("unroll")',
                "    for (int ki = 0; ki < 8; ++ki) {",
                "        float gate_w = "
                "float((gate_packed >> (ki * 4)) & 0xFu) * gate_scale + gate_bias;",
                "        float up_w = "
                "float((up_packed >> (ki * 4)) & 0xFu) * up_scale + up_bias;",
            ]
        )
        for row in range(m):
            acc_idx = col * m + row
            lines.append(f"        float x_value_{row} = float(v{suffix}_{row}[ki]);")
            lines.append(f"        gate_acc[{acc_idx}] += x_value_{row} * gate_w;")
            lines.append(f"        up_acc[{acc_idx}] += x_value_{row} * up_w;")
        lines.extend(["    }", "}"])
    return "\n            ".join(lines)


def _fused_gate_up_q4_ksplit_source(
    *,
    m: int,
    group_size: int,
    k_parts: int,
    kconst: int,
    tile_cols: int = 4,
) -> str:
    """Return the Metal body so source-shape invariants stay testable."""
    if m not in {2, 3, 4, 5, 6}:
        raise ValueError("the Qwen fusion supports M2-M6 compute tiles")
    if group_size not in {32, 64, 128}:
        raise ValueError("unsupported group size")
    if k_parts != 2:
        raise ValueError("the Qwen gate/up prototype requires two K parts")
    if tile_cols not in {2, 4}:
        raise ValueError("the Qwen gate/up prototype supports BN=2 or BN=4")
    if kconst <= 0 or kconst % 64:
        raise ValueError("K must be a positive multiple of 64")

    n_acc = tile_cols * m
    pack_block = _fused_q4_pack_block(m, "A", tile_cols)
    return f"""
        using namespace metal;
        constexpr int GS = {group_size};
        constexpr int K_PARTS = {k_parts};
        constexpr int K = {kconst};
        constexpr int K_by_p = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int per_part = K_by_p / K_PARTS;

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * {tile_cols};
        int p_start = int(part) * per_part;
        int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;

        float gate_acc[{n_acc}];
        float up_acc[{n_acc}];
        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            gate_acc[i] = 0.0f;
            up_acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int packA = p_start + int(lane); packA < p_end; packA += 32) {{
            {pack_block}
        }}

        _Pragma("unroll")
        for (int i = 0; i < {n_acc}; ++i) {{
            gate_acc[i] = simd_sum(gate_acc[i]);
            up_acc[i] = simd_sum(up_acc[i]);
        }}

        threadgroup float gate_partials[K_PARTS * {n_acc}];
        threadgroup float up_partials[K_PARTS * {n_acc}];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < {n_acc}; ++i) {{
                gate_partials[int(part) * {n_acc} + i] = gate_acc[i];
                up_partials[int(part) * {n_acc} + i] = up_acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < {n_acc}) {{
            float gate_total = 0.0f;
            float up_total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                gate_total += gate_partials[p * {n_acc} + int(lane)];
                up_total += up_partials[p * {n_acc} + int(lane)];
            }}
            int j = int(lane) / {m};
            int row = int(lane) - j * {m};
            T gate_value = T(gate_total);
            T up_value = T(up_total);
            auto sigmoid_tail = 1 / (1 + metal::exp(metal::abs(gate_value)));
            T sigmoid_value = (gate_value < T(0))
                ? T(sigmoid_tail)
                : T(1 - sigmoid_tail);
            y[row * N + n0 + j] = T((gate_value * sigmoid_value) * up_value);
        }}
    """


@lru_cache(maxsize=None)
def _fused_gate_up_q4_ksplit_kernel(
    m: int,
    group_size: int,
    dtype: mx.Dtype,
    kconst: int,
    tile_cols: int,
):
    source = _fused_gate_up_q4_ksplit_source(
        m=m,
        group_size=group_size,
        k_parts=2,
        kconst=kconst,
        tile_cols=tile_cols,
    )
    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qwen_gateup_vk_k_m{m}_bn{tile_cols}_q4_k{kconst}"
            f"_gs{group_size}_{dtype_tag}"
        ),
        input_names=[
            "x",
            "gate_w_q",
            "gate_scales",
            "gate_biases",
            "up_w_q",
            "up_scales",
            "up_biases",
            "N_size",
        ],
        output_names=["y"],
        source=source,
    )


def is_qwen_gate_up_swiglu_vk_k_eligible(
    x: mx.array,
    gate_module: Any,
    up_module: Any,
) -> bool:
    """Return whether the internal M2-M6 Qwen fusion can run."""
    if not mx.metal.is_available():
        return False
    if not isinstance(gate_module, nn.QuantizedLinear) or not isinstance(
        up_module, nn.QuantizedLinear
    ):
        return False
    if x.dtype not in (mx.bfloat16, mx.float16) or len(x.shape) < 2:
        return False
    if not 2 <= int(x.shape[-2]) <= 6:
        return False
    if any(
        int(getattr(module, "bits", 0) or 0) != 4 for module in (gate_module, up_module)
    ):
        return False
    group_size = int(getattr(gate_module, "group_size", 0) or 0)
    if group_size not in {32, 64, 128} or group_size != int(
        getattr(up_module, "group_size", 0) or 0
    ):
        return False
    if any(
        str(getattr(module, "mode", "affine")) != "affine"
        for module in (gate_module, up_module)
    ):
        return False
    if any("bias" in module for module in (gate_module, up_module)):
        return False
    if tuple(gate_module.weight.shape) != tuple(up_module.weight.shape):
        return False
    if tuple(gate_module.scales.shape) != tuple(up_module.scales.shape):
        return False
    if tuple(gate_module.biases.shape) != tuple(up_module.biases.shape):
        return False
    if any(
        module.scales.dtype != x.dtype or module.biases.dtype != x.dtype
        for module in (gate_module, up_module)
    ):
        return False

    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    if k != int(gate_module.weight.shape[1]) * 8:
        return False
    if k % 64 or n < 4096 or n % 4:
        return False
    batch_count = 1
    for dim in x.shape[:-2]:
        batch_count *= int(dim)
    return batch_count == 1


def qwen_gate_up_swiglu_vk_k(
    x: mx.array,
    gate_module: nn.QuantizedLinear,
    up_module: nn.QuantizedLinear,
    *,
    tile_cols: int = 4,
) -> mx.array:
    """Compute Qwen gate/up + SwiGLU with the internal small-M prototype."""
    from mlx_lm.models.qwen3_next import swiglu

    if not is_qwen_gate_up_swiglu_vk_k_eligible(x, gate_module, up_module):
        return swiglu(gate_module(x), up_module(x))
    if tile_cols not in {2, 4}:
        return swiglu(gate_module(x), up_module(x))

    leading = x.shape[:-2]
    m = int(x.shape[-2])
    k = int(x.shape[-1])
    n = int(gate_module.weight.shape[0])
    # Compile one kernel per exact verify tile (M2-M6).  Owning the true row
    # count keeps the gate/up + SwiGLU chain in registers and removes the
    # zero-padded activation copy a fixed M4/M6 tile would materialize on every
    # call.  Each output row reduces over an independent accumulator, so the
    # live rows stay bit-identical to the padded tile.
    x2 = mx.contiguous(x.reshape(m, k))
    kernel = _fused_gate_up_q4_ksplit_kernel(
        m,
        int(gate_module.group_size),
        x.dtype,
        k,
        tile_cols,
    )
    (y,) = kernel(
        inputs=[
            x2,
            gate_module.weight,
            gate_module.scales,
            gate_module.biases,
            up_module.weight,
            up_module.scales,
            up_module.biases,
            n,
        ],
        template=[("T", x.dtype)],
        grid=(64, n // tile_cols, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*leading, m, n)
