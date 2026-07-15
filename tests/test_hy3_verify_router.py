from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.attention_context import attention_phase
from mtplx.hy3_verify_router import (
    reset_hy3_verify_router_stats,
    hy3_verify_router_stats,
)
from mtplx.models.hy3_mlx import Model, ModelArgs, Router


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


def test_fixed_m4_verify_router_uses_immutable_model_load_config_on_hot_path(
    monkeypatch,
) -> None:
    import mtplx.hy3_verify_router as verify_router

    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    original_mode = verify_router._mode
    original_rows = verify_router._target_rows
    config_reads = {"mode": 0, "rows": 0}

    def counted_mode():
        config_reads["mode"] += 1
        return original_mode()

    def counted_rows():
        config_reads["rows"] += 1
        return original_rows()

    monkeypatch.setattr(verify_router, "_mode", counted_mode)
    monkeypatch.setattr(verify_router, "_target_rows", counted_rows)
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        first = router(hidden)
        mx.eval(*first)
        second = router(hidden)
        mx.eval(*second)

    assert config_reads == {"mode": 0, "rows": 0}


def test_fixed_m4_linear_routers_share_one_weight_parameterized_graph(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    args = _args()
    args.num_hidden_layers = 3
    args.mlp_layer_types = ["dense", "sparse", "sparse"]
    args.num_experts = 4
    args.num_experts_per_tok = 2
    model = Model(args)
    first_router = model.model.layers[1].mlp.router
    second_router = model.model.layers[2].mlp.router
    first_router.gate.weight = mx.array(
        [[0.1] * 64, [-0.2] * 64, [0.3] * 64, [-0.4] * 64],
        dtype=mx.bfloat16,
    )
    second_router.gate.weight = mx.array(
        [[-0.3] * 64, [0.4] * 64, [-0.1] * 64, [0.2] * 64],
        dtype=mx.bfloat16,
    )
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    expected_first = first_router._forward_stock(hidden)
    expected_second = second_router._forward_stock(hidden)
    mx.eval(*expected_first, *expected_second)
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        actual_first = first_router(hidden)
        mx.eval(*actual_first)
        actual_second = second_router(hidden)
        mx.eval(*actual_second)

    assert mx.array_equal(actual_first[0], expected_first[0]).item()
    assert mx.array_equal(actual_first[1], expected_first[1]).item()
    assert mx.array_equal(actual_second[0], expected_second[0]).item()
    assert mx.array_equal(actual_second[1], expected_second[1]).item()
    stats = hy3_verify_router_stats()
    assert stats["compiled_router_count"] == 2
    assert stats["compiled_graph_count"] == 1
    assert stats["traces"] == 1
    assert stats["retraces"] == 0
    assert stats["shared_graph_calls"] == 2
    assert stats["per_router_graph_calls"] == 0


def test_fixed_m4_router_caches_shared_record_after_first_layer_dispatch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")

    class CountingGroup(dict):
        def __init__(self):
            super().__init__()
            self.lookups = 0

        def get(self, key, default=None):
            self.lookups += 1
            return super().get(key, default)

    group = CountingGroup()
    routers = [
        Router(_args(), verify_router_group=group),
        Router(_args(), verify_router_group=group),
    ]
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        for router in routers:
            for _ in range(2):
                output = router(hidden)
                mx.eval(*output)

    assert group.lookups == 2
    stats = hy3_verify_router_stats()
    assert stats["compiled_calls"] == 4
    assert stats["compiled_graph_count"] == 1
    assert stats["traces"] == 1


def test_fixed_m4_per_router_topology_uses_distinct_captured_graphs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_TOPOLOGY", "per-router")
    args = _args()
    args.num_hidden_layers = 3
    args.mlp_layer_types = ["dense", "sparse", "sparse"]
    model = Model(args)
    routers = [
        model.model.layers[1].mlp.router,
        model.model.layers[2].mlp.router,
    ]
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    expected = [router._forward_stock(hidden) for router in routers]
    mx.eval(*(item for output in expected for item in output))
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        actual = [router(hidden) for router in routers]
        mx.eval(*(item for output in actual for item in output))

    for observed, stock in zip(actual, expected, strict=True):
        assert mx.array_equal(observed[0], stock[0]).item()
        assert mx.array_equal(observed[1], stock[1]).item()
    stats = hy3_verify_router_stats()
    assert stats["topology"] == "per-router"
    assert stats["compiled_router_count"] == 2
    assert stats["compiled_graph_count"] == 2
    assert stats["shared_graph_calls"] == 0
    assert stats["per_router_graph_calls"] == 2
    assert stats["traces"] == 2
    assert stats["retraces"] == 0


def test_verify_router_rejects_unknown_topology_at_model_load(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_TOPOLOGY", "unknown")

    with pytest.raises(ValueError, match="TOPOLOGY.*shared.*per-router"):
        Model(_args())


def test_fixed_m4_wrapped_router_preserves_wrapper_on_per_router_graph(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()

    class CountingGate(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return self.base(value)

    wrapper = CountingGate(router.gate)
    router.gate = wrapper
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    expected = router._forward_stock(hidden)
    mx.eval(*expected)
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        actual = router(hidden)
        mx.eval(*actual)

    assert mx.array_equal(actual[0], expected[0]).item()
    assert mx.array_equal(actual[1], expected[1]).item()
    assert wrapper.calls == 2
    stats = hy3_verify_router_stats()
    assert stats["compiled_graph_count"] == 1
    assert stats["shared_graph_calls"] == 0
    assert stats["per_router_graph_calls"] == 1


def test_fixed_m4_biased_linear_preserves_bias_on_per_router_graph(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_COMPILE", "1")
    monkeypatch.setenv("MTPLX_HY3_VERIFY_ROUTER_ROWS", "4")
    router = _router()
    router.gate = nn.Linear(64, 2, bias=True)
    router.gate.weight = mx.array(
        [[0.1] * 64, [-0.2] * 64],
        dtype=mx.bfloat16,
    )
    router.gate.bias = mx.array([0.75, -0.25], dtype=mx.float32)
    hidden = mx.full((1, 4, 64), 0.125, dtype=mx.bfloat16)
    expected = router._forward_stock(hidden)
    mx.eval(*expected)
    reset_hy3_verify_router_stats()

    with attention_phase("decode_verify"):
        actual = router(hidden)
        mx.eval(*actual)

    assert mx.array_equal(actual[0], expected[0]).item()
    assert mx.array_equal(actual[1], expected[1]).item()
    stats = hy3_verify_router_stats()
    assert stats["shared_graph_calls"] == 0
    assert stats["per_router_graph_calls"] == 1


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

        def changed(*values):
            indices, weights = compiled(*values)
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
