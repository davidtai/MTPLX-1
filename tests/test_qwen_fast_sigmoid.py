"""Standalone experimental fast-sigmoid Qwen SwiGLU path."""

from __future__ import annotations

import mlx.core as mx

from mtplx.kernels.qwen_fast_sigmoid import (
    _qwen_swiglu_source,
    qwen_swiglu,
)


def test_fast_source_changes_only_the_exp_intrinsic() -> None:
    precise = _qwen_swiglu_source(fast=False)
    fast = _qwen_swiglu_source(fast=True)

    assert "metal::exp(metal::abs(gate_value))" in precise
    assert "metal::fast::exp(metal::abs(gate_value))" in fast
    assert "metal::fast" not in precise
    assert precise.replace("metal::exp", "metal::fast::exp") == fast
    assert "using Vec8 = vec<T, 8>;" in precise


def test_precise_and_fast_swiglu_are_deterministic_and_numerically_close() -> None:
    from mlx_lm.models.qwen3_next import swiglu

    mx.random.seed(31)
    gate = (mx.random.normal((4, 4096), dtype=mx.float32) * 2.0).astype(mx.bfloat16)
    up = (mx.random.normal((4, 4096), dtype=mx.float32) * 0.5).astype(mx.bfloat16)
    reference = swiglu(gate, up)
    precise = qwen_swiglu(gate, up, fast=False)
    fast = qwen_swiglu(gate, up, fast=True)
    replay = qwen_swiglu(gate, up, fast=True)
    mx.eval(reference, precise, fast, replay)

    precise_dmax = float(
        mx.abs(precise.astype(mx.float32) - reference.astype(mx.float32)).max()
    )
    fast_dmax = float(
        mx.abs(fast.astype(mx.float32) - precise.astype(mx.float32)).max()
    )
    replay_dmax = float(
        mx.abs(fast.astype(mx.float32) - replay.astype(mx.float32)).max()
    )
    assert precise_dmax <= 0.03125
    assert fast_dmax <= 0.125
    assert replay_dmax == 0.0


def test_internal_native_mlp_selector_accepts_fast_sigmoid(monkeypatch) -> None:
    from mtplx import native_mlp

    monkeypatch.setenv("MTPLX_MLP_CALL_VARIANT", "fast-sigmoid")
    assert native_mlp._normalized_variant() == "fast_sigmoid"
