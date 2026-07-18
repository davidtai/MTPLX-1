"""Experimental exact-M2/M3/M5 verify QuantizedLinear routing."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.qwen_row_traversal import (
    install_qwen_exact_m_qlinear_patch,
    qwen_exact_m_eligible,
    qwen_qmm_exact_m,
    qwen_row_traversal_qlinear_enabled,
    qwen_row_traversal_stats,
    uninstall_qwen_exact_m_qlinear_patch,
)


def _q4_linear(k: int, n: int, *, group_size: int = 64) -> nn.QuantizedLinear:
    module = nn.QuantizedLinear(
        k,
        n,
        bias=False,
        group_size=group_size,
        bits=4,
        mode="affine",
    )
    module.scales = module.scales.astype(mx.bfloat16)
    module.biases = module.biases.astype(mx.bfloat16)
    return module


def test_enablement_follows_variant_with_env_override(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", raising=False)
    monkeypatch.delenv("MTPLX_MLP_CALL_VARIANT", raising=False)
    assert not qwen_row_traversal_qlinear_enabled()

    monkeypatch.setenv("MTPLX_MLP_CALL_VARIANT", "fused_gateup_vk_k_bn2")
    assert qwen_row_traversal_qlinear_enabled()
    monkeypatch.setenv("MTPLX_MLP_CALL_VARIANT", "fused-gateup-vk-k")
    assert qwen_row_traversal_qlinear_enabled()

    # Explicit env wins in both directions.
    monkeypatch.setenv("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", "0")
    assert not qwen_row_traversal_qlinear_enabled()
    monkeypatch.setenv("MTPLX_MLP_CALL_VARIANT", "stock")
    monkeypatch.setenv("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", "1")
    assert qwen_row_traversal_qlinear_enabled()


def test_eligibility_is_exact_m235_q4() -> None:
    for rows in (2, 3, 5):
        assert qwen_exact_m_eligible(rows, 512, 4096, 4, 64, mx.bfloat16)
    for rows in (1, 4, 6, 7):
        assert not qwen_exact_m_eligible(rows, 512, 4096, 4, 64, mx.bfloat16)
    assert not qwen_exact_m_eligible(2, 512, 4096, 8, 64, mx.bfloat16)
    assert not qwen_exact_m_eligible(2, 500, 4096, 4, 64, mx.bfloat16)
    assert not qwen_exact_m_eligible(2, 512, 4098, 4, 64, mx.bfloat16)
    assert not qwen_exact_m_eligible(2, 512, 4096, 4, 48, mx.bfloat16)
    assert not qwen_exact_m_eligible(2, 512, 4096, 4, 64, mx.float32)
    # Measured floors: tiny-N projections and M2 x lm_head-class N stay stock.
    assert not qwen_exact_m_eligible(3, 512, 1024, 4, 64, mx.bfloat16)
    assert not qwen_exact_m_eligible(5, 512, 48, 4, 64, mx.bfloat16)
    assert not qwen_exact_m_eligible(2, 512, 248320, 4, 64, mx.bfloat16)
    assert qwen_exact_m_eligible(3, 512, 248320, 4, 64, mx.bfloat16)
    assert qwen_exact_m_eligible(5, 512, 248320, 4, 64, mx.bfloat16)
    assert qwen_exact_m_eligible(2, 512, 2048, 4, 64, mx.bfloat16)


def test_exact_m_matches_stock_within_vk_band() -> None:
    mx.random.seed(31)
    for rows in (2, 3, 5):
        for n in (4096, 1024):  # exercises k_parts = 2 and 4
            module = _q4_linear(512, n)
            x = (mx.random.normal((rows, 512), dtype=mx.float32) * 0.5).astype(
                mx.bfloat16
            )
            candidate = qwen_qmm_exact_m(
                x,
                module.weight,
                module.scales,
                module.biases,
                group_size=64,
            )
            reference = mx.quantized_matmul(
                x,
                module.weight,
                scales=module.scales,
                biases=module.biases,
                transpose=True,
                group_size=64,
                bits=4,
            )
            mx.eval(candidate, reference)
            dmax = float(
                mx.abs(
                    candidate.astype(mx.float32) - reference.astype(mx.float32)
                ).max()
            )
            assert candidate.shape == (rows, n)
            assert dmax <= 0.25, f"M={rows} N={n} drift too large: {dmax}"


def test_exact_m_rejects_other_row_counts() -> None:
    module = _q4_linear(512, 4096)
    x = mx.zeros((4, 512), dtype=mx.bfloat16)
    with pytest.raises(ValueError):
        qwen_qmm_exact_m(
            x,
            module.weight,
            module.scales,
            module.biases,
            group_size=64,
        )


def test_patch_routes_m235_and_falls_through(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", "1")
    original = nn.QuantizedLinear.__call__
    try:
        report = install_qwen_exact_m_qlinear_patch()
        assert report["installed"]
        again = install_qwen_exact_m_qlinear_patch()
        assert again["already"]

        module = _q4_linear(512, 4096)
        before = qwen_row_traversal_stats()
        mx.random.seed(37)
        for rows in (2, 3, 5):
            x = (
                mx.random.normal((1, rows, 512), dtype=mx.float32) * 0.5
            ).astype(mx.bfloat16)
            routed = module(x)
            monkeypatch.setenv("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", "0")
            # The patch decision is per-call via lane state, not env; the
            # numeric contract is what matters here.
            monkeypatch.setenv("MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR", "1")
            stock = original(module, x)
            mx.eval(routed, stock)
            assert routed.shape == stock.shape
            dmax = float(
                mx.abs(
                    routed.astype(mx.float32) - stock.astype(mx.float32)
                ).max()
            )
            assert dmax <= 0.25, f"patched M={rows} drift too large: {dmax}"
        after = qwen_row_traversal_stats()
        assert int(after["calls_m2"]) == int(before["calls_m2"]) + 1
        assert int(after["calls_m3"]) == int(before["calls_m3"]) + 1
        assert int(after["calls_m5"]) == int(before["calls_m5"]) + 1

        # M1/M4 fall through to the wrapped call (stock here).
        x1 = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        x4 = mx.zeros((1, 4, 512), dtype=mx.bfloat16)
        mx.eval(module(x1), module(x4))
        final = qwen_row_traversal_stats()
        assert int(final["calls_m2"]) == int(after["calls_m2"])
        assert int(final["calls_m5"]) == int(after["calls_m5"])
    finally:
        uninstall_qwen_exact_m_qlinear_patch()
        assert nn.QuantizedLinear.__call__ is original
