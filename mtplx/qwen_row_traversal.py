"""Experimental exact-M verify QuantizedLinear routing (Qwen row traversal).

The turbo NAX verify patch routes 4-bit ``QuantizedLinear`` calls with M in
4..16 through the MTPLX verify kernels, but the M2/M3 speculative verify tiles
(depths 1-2) stay on stock MLX ``qmv``, which re-reads the full weight matrix
once per live row, and M5 (depth 4) runs zero-padded into the M6 tile.

When the experimental Qwen row-traversal variant is active this module wraps
``QuantizedLinear.__call__`` once more and routes eligible non-prefill
M in {2, 3, 5} calls through the proven vk_k split-K codegen
(:func:`mtplx.verify_kernels._build_ksplit_kernel`) compiled at the true row
count: one weight read for all rows and no zero-padded activation copy.

Default off.  Enabled implicitly by ``--experimental-qwen-row-traversal``
(the ``fused_gateup_vk_k*`` MLP variants) and explicitly overridable either
way with ``MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR``.  Numerics are the established
turbo vk-family class (fp32 accumulate, lane-strided reduction; not bit-exact
vs stock), and every lane self-validates at model load through
``kernel_selfcheck`` exactly like the shipped NAX lanes.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

_EXACT_M_ROWS = (2, 3, 5)

_PATCH: dict[str, Any] = {"installed": False, "original": None}

_STATS = {
    "enabled": False,
    "installed": False,
    "calls_m2": 0,
    "calls_m3": 0,
    "calls_m5": 0,
    "fallback_calls": 0,
}


def _fused_variant_active() -> bool:
    variant = os.environ.get("MTPLX_MLP_CALL_VARIANT", "").strip().lower()
    return variant.replace("-", "_") in {
        "fused_gateup_vk_k",
        "fused_gateup_vkk",
        "fused_gateup_vk_k_bn2",
    }


def qwen_row_traversal_qlinear_enabled() -> bool:
    """Explicit env wins; otherwise follow the row-traversal MLP variant."""
    raw = os.environ.get("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", "").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    return _fused_variant_active()


def qwen_exact_m_eligible(
    m: int, k: int, n: int, bits: int, group_size: int, dtype
) -> bool:
    """Mirror of ``vk_eligible_ksplit`` at the exact M2/M3/M5 tiles.

    Floors from the 2026-07-17 queued-lane microbench on the live census
    shapes: tiny-N projections (N=48 GDN b/a, N=1024 kv) lose to stock qmv at
    every M (launch-bound), and M2 loses on lm_head-class N (its two stock
    qmv passes stay bandwidth-competitive at 62k tiny threadgroups), so both
    stay stock. Every trunk shape with N >= 2048 wins: M2 +8..19%, M3
    +25..34%, M5 +32..45% vs stock qmv.
    """
    return (
        int(bits) == 4
        and int(group_size) in (32, 64, 128)
        and dtype in (mx.bfloat16, mx.float16)
        and int(m) in _EXACT_M_ROWS
        and int(k) % 64 == 0
        and int(n) % 4 == 0
        and int(n) >= 2048
        and not (int(m) == 2 and int(n) >= 100000)
    )


def qwen_qmm_exact_m(
    x2: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int = 64,
) -> mx.array:
    """Exact-M split-K verify matmul; x2 must be (M in {2,3,5}, K).

    Same morphology and numerics family as ``vk_qmm_m4_ksplit`` (vk_k: K baked
    into the specialization), compiled at the true row count so no zero-padded
    activation copy is materialized.
    """
    from .verify_kernels import _build_ksplit_kernel

    m = int(x2.shape[0])
    if m not in _EXACT_M_ROWS:
        raise ValueError("qwen_qmm_exact_m supports M in {2, 3, 5}")
    k = int(x2.shape[1])
    n = int(w_q.shape[0])
    k_parts = 2 if n >= 4096 else 4
    kernel = _build_ksplit_kernel(
        m,
        4,
        int(group_size),
        x2.dtype,
        k_parts=k_parts,
        dual=False,
        kconst=k,
    )
    (y,) = kernel(
        inputs=[mx.contiguous(x2), w_q, scales, biases, k, n],
        template=[("T", x2.dtype)],
        grid=(32 * k_parts, n // 4, 1),
        threadgroup=(32 * k_parts, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x2.dtype],
    )
    return y


def install_qwen_exact_m_qlinear_patch() -> dict[str, object]:
    """Wrap the current ``QuantizedLinear.__call__`` (post-NAX) once.

    Must be installed after the NAX patch so exact-M5 takes precedence over
    the padded-M6 NAX route; everything else falls through unchanged.
    Idempotent.
    """
    import mlx.nn as nn

    _STATS["enabled"] = True
    if _PATCH["installed"]:
        return {"installed": True, "already": True}

    original = nn.QuantizedLinear.__call__

    from .attention_context import current_attention_phase
    from .kernel_selfcheck import lane_disabled

    def patched(self, x: mx.array) -> mx.array:  # type: ignore[no-untyped-def]
        bits = int(getattr(self, "bits", 0) or 0)
        if (
            bits == 4
            and x.ndim >= 2
            and str(getattr(self, "mode", "affine")) == "affine"
            and current_attention_phase() != "prefill"
        ):
            m = 1
            for d in x.shape[:-1]:
                m *= int(d)
            if m in _EXACT_M_ROWS and not lane_disabled(f"qwen_rt_qmm_m{m}"):
                group_size = int(getattr(self, "group_size", 0) or 0)
                w_q = self["weight"]
                k = int(x.shape[-1])
                n = int(w_q.shape[0])
                scales = self["scales"]
                biases = self["biases"]
                if (
                    qwen_exact_m_eligible(m, k, n, bits, group_size, x.dtype)
                    and scales.dtype == x.dtype
                    and biases.dtype == x.dtype
                ):
                    y = qwen_qmm_exact_m(
                        x.reshape(m, k),
                        w_q,
                        scales,
                        biases,
                        group_size=group_size,
                    )
                    _STATS[f"calls_m{m}"] = int(_STATS[f"calls_m{m}"]) + 1
                    y = y.reshape(*x.shape[:-1], n)
                    if "bias" in self:
                        y = y + self["bias"]
                    return y
                _STATS["fallback_calls"] = int(_STATS["fallback_calls"]) + 1
        return original(self, x)

    nn.QuantizedLinear.__call__ = patched
    _PATCH["installed"] = True
    _PATCH["original"] = original
    _STATS["installed"] = True
    return {"installed": True, "already": False}


def uninstall_qwen_exact_m_qlinear_patch() -> None:
    import mlx.nn as nn

    if _PATCH["installed"] and _PATCH["original"] is not None:
        nn.QuantizedLinear.__call__ = _PATCH["original"]
    _PATCH["installed"] = False
    _PATCH["original"] = None
    _STATS["installed"] = False


def qwen_row_traversal_stats() -> dict[str, object]:
    return dict(_STATS)
