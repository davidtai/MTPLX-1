from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.hy3_verify_router import (
    reset_hy3_verify_router_stats,
    hy3_verify_router_stats,
)
from mtplx.models.hy3_mlx import Model, ModelArgs


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _args() -> ModelArgs:
    return ModelArgs(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _router():
    router = Model(_args()).model.layers[1].mlp.router
    router.route_norm = False
    router.gate.weight = mx.array(
        [
            [0.1] * 64,
            [-0.2] * 64,
        ],
        dtype=mx.bfloat16,
    )
    router.expert_bias = mx.zeros((2,), dtype=mx.float32)
    return router


def test_fixed_m4_verify_router_reuses_one_compiled_trace_and_matches_stock(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    stock_indices, stock_weights = router._forward_stock(hidden)
    mx.eval(stock_indices, stock_weights)

    reset_hy3_verify_router_stats()
    with attention_phase("decode_verify"):
        first_indices, first_weights = router(hidden)
        mx.eval(first_indices, first_weights)
        second_indices, second_weights = router(hidden)
        mx.eval(second_indices, second_weights)

    assert mx.array_equal(first_indices, stock_indices).item()
    assert mx.array_equal(first_weights, stock_weights).item()
    assert mx.array_equal(second_indices, stock_indices).item()
    assert mx.array_equal(second_weights, stock_weights).item()
    stats = hy3_verify_router_stats()
    assert stats["eligible_calls"] == 2
    assert stats["compiled_calls"] == 2
    assert stats["compiled_router_count"] == 1
    assert stats["traces"] == 1
    assert stats["retraces"] == 0
    assert stats["failures"] == 0


def test_verify_router_compile_is_inactive_outside_fixed_decode_verify(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()
    reset_hy3_verify_router_stats()

    with attention_phase("prefill"):
        prefill = router(mx.zeros((1, 4, 64), dtype=mx.bfloat16))
    with attention_phase("decode_verify"):
        wrong_rows = router(mx.zeros((1, 3, 64), dtype=mx.bfloat16))
    mx.eval(*prefill, *wrong_rows)

    stats = hy3_verify_router_stats()
    assert stats["eligible_calls"] == 0
    assert stats["compiled_calls"] == 0
    assert stats["traces"] == 0
    assert stats["fallback_reasons"] == {
        "attention_phase:prefill": 1,
        "rows:3": 1,
    }


def test_fixed_m4_router_retained_stats_reset_keeps_prewarmed_graph(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)

    reset_hy3_verify_router_stats()
    with attention_phase("decode_verify"):
        warmup = router(hidden)
        mx.eval(*warmup)
    assert hy3_verify_router_stats()["traces"] == 1

    reset_hy3_verify_router_stats()
    with attention_phase("decode_verify"):
        retained = router(hidden)
        mx.eval(*retained)

    stats = hy3_verify_router_stats()
    assert stats["compiled_calls"] == 1
    assert stats["compiled_router_count"] == 1
    assert stats["traces"] == 0
    assert stats["initial_traces"] == 0
    assert stats["retraces"] == 0


def test_verify_router_parity_fails_closed_on_changed_route_weights(
    monkeypatch,
) -> None:
    import mtplx.hy3_verify_router as verify_router

    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "parity")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    real_compile = mx.compile

    def changed_compile(function, *args, **kwargs):
        compiled = real_compile(function, *args, **kwargs)

        def changed(value):
            indices, weights = compiled(value)
            return indices, weights + mx.array(0.125, dtype=weights.dtype)

        return changed

    monkeypatch.setattr(verify_router.mx, "compile", changed_compile)
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        with pytest.raises(RuntimeError, match="route-weight parity"):
            router(hidden)

    stats = hy3_verify_router_stats()
    assert stats["parity_checks"] == 1
    assert stats["parity_failures"] == 1
    assert stats["failures"] == 1
