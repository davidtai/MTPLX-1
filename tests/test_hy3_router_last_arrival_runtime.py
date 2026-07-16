from __future__ import annotations

import os
import inspect
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.hy3_router_last_arrival as last_arrival
import mtplx.models.hy3_mlx as hy3_mlx
from mtplx.attention_context import attention_phase, current_attention_phase
from mtplx.hy3_router_fp32 import (
    Hy3RouterFP32Ineligible,
    hy3_router_fp32_route,
    prepare_hy3_router_fp32_weight,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime


_SELECTOR = "mpp-r1-last-arrival-fused-r2"


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    hardware_parity = os.environ.get("MTPLX_RUN_ISSUE58_HARDWARE_PARITY") == "1"
    if not hardware_parity:
        mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _router_args() -> hy3_mlx.ModelArgs:
    return hy3_mlx.ModelArgs(
        model_type="hy_v3",
        hidden_size=4096,
        num_hidden_layers=1,
        intermediate_size=8192,
        moe_intermediate_size=1536,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts=192,
        num_experts_per_tok=8,
        num_shared_experts=1,
        first_k_dense_replace=0,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=128,
        route_norm=True,
        router_scaling_factor=2.826,
    )


def _router() -> hy3_mlx.Router:
    router = hy3_mlx.Router(_router_args())
    router.gate.weight = mx.zeros((192, 4096), dtype=mx.bfloat16)
    return router


def _runtime_inputs(rows: int = 4):
    return (
        mx.zeros((1, rows, 4096), dtype=mx.float32),
        mx.zeros((4096, 192), dtype=mx.bfloat16),
        mx.zeros((192,), dtype=mx.float32),
    )


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_last_arrival_source_specializes_logical_rows_over_one_physical_m8_tile(
    rows: int,
) -> None:
    source = last_arrival.hy3_router_last_arrival_source(
        rows=rows,
        scaling_factor=2.826,
        sigmoid_mode="precise",
    )

    assert f"constexpr int ROWS = {rows};" in source
    assert "constexpr int PADDED_ROWS = 8;" in source
    assert f"constexpr int R2_WAVES = {1 if rows <= 4 else 2};" in source
    assert "constexpr int P = 16;" in source
    assert "constexpr int SGPTG = 4;" in source
    assert "constexpr int THREADGROUPS = 48;" in source
    assert "threadgroup float A_tile[PADDED_ROWS * KS];" in source
    assert "row < ROWS" in source
    assert "partials[15 * STRIDE + index]" in source
    assert "uint row = simd_gid + uint(wave) * SGPTG;" in source
    assert "if (row >= uint(ROWS))" in source
    assert "atomic_store_explicit(&ready[tg], tag" in source
    assert "atomic_store_explicit(&checks[tg], ~tag" in source
    assert "atomic_compare_exchange_weak_explicit(" in source
    assert "memory_order_seq_cst" in source
    assert "thread_scope_device" in source
    assert "atomic_fetch_add" not in source
    assert "while (atomic_load" not in source


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_last_arrival_runtime_calls_one_kernel_and_returns_logical_m_contract(
    rows: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, weight, expert_bias = _runtime_inputs(rows)
    captured: dict[str, object] = {}
    build_rows: list[int] = []

    class FakeKernel:
        def __call__(self, **kwargs: object):
            assert not captured
            captured.update(kwargs)
            return (
                mx.zeros((rows, 8), dtype=mx.int32),
                mx.ones((rows, 8), dtype=mx.float32),
                mx.zeros((24_672,), dtype=mx.float32),
            )

    def fake_build(logical_rows: int, *_args, **_kwargs):
        build_rows.append(logical_rows)
        return FakeKernel()

    monkeypatch.setattr(
        last_arrival,
        "_build_hy3_router_last_arrival_kernel",
        fake_build,
    )
    assert not hasattr(last_arrival, "_next_router_epoch")
    epoch = mx.array(73, dtype=mx.uint32)

    output = last_arrival.hy3_router_last_arrival_route(
        value,
        weight,
        expert_bias,
        available=True,
        sigmoid_mode="precise",
        epoch=epoch,
    )

    assert tuple(output.expert_ids.shape) == (1, rows, 8)
    assert output.expert_ids.dtype == mx.int32
    assert tuple(output.route_weights.shape) == (1, rows, 8)
    assert output.route_weights.dtype == mx.float32
    assert output.dispatch_count == 1
    assert output.batch_shape == (1,)
    assert output.rows == rows
    assert output.top_k == 8
    assert output.assignment_count == rows * 8
    assert build_rows == [rows]
    assert captured["grid"] == (48 * 128, 1, 1)
    assert captured["threadgroup"] == (128, 1, 1)
    assert captured["output_shapes"] == [(rows, 8), (rows, 8), (24_672,)]
    assert captured["output_dtypes"] == [mx.int32, mx.float32, mx.float32]
    assert "init_value" not in captured
    inputs = captured["inputs"]
    assert isinstance(inputs, list)
    assert len(inputs) == 4
    assert tuple(inputs[0].shape) == (rows, 4096)
    assert inputs[1] is weight
    assert inputs[2] is expert_bias
    assert inputs[3] is epoch


def test_last_arrival_kernel_cache_key_includes_logical_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_metal_kernel(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(mx.fast, "metal_kernel", fake_metal_kernel)
    last_arrival._build_hy3_router_last_arrival_kernel.cache_clear()
    try:
        m1_first = last_arrival._build_hy3_router_last_arrival_kernel(
            1,
            2.826,
            "precise",
        )
        m1_second = last_arrival._build_hy3_router_last_arrival_kernel(
            1,
            2.826,
            "precise",
        )
        m8 = last_arrival._build_hy3_router_last_arrival_kernel(
            8,
            2.826,
            "precise",
        )
    finally:
        last_arrival._build_hy3_router_last_arrival_kernel.cache_clear()

    assert m1_first is m1_second
    assert m8 is not m1_first
    assert len(calls) == 2
    assert calls[0]["name"].startswith("mtplx_hy3_router_last_arrival_m1_")
    assert calls[1]["name"].startswith("mtplx_hy3_router_last_arrival_m8_")
    assert "constexpr int ROWS = 1;" in calls[0]["source"]
    assert "constexpr int ROWS = 8;" in calls[1]["source"]


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_resident_callable_structurally_satisfies_logical_m_contract(
    rows: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, weight, expert_bias = _runtime_inputs(rows)
    expected = last_arrival.Hy3RouterLastArrivalOutput(
        expert_ids=mx.zeros((1, rows, 8), dtype=mx.int32),
        route_weights=mx.ones((1, rows, 8), dtype=mx.float32),
    )
    calls: list[tuple[object, object, object, dict[str, object]]] = []

    def fake_route(hidden_rows, resident_weight, resident_bias, **kwargs):
        calls.append((hidden_rows, resident_weight, resident_bias, kwargs))
        return expected

    monkeypatch.setattr(last_arrival, "hy3_router_last_arrival_route", fake_route)
    router = last_arrival.Hy3RouterLastArrival(
        weight=weight,
        expert_bias=expert_bias,
        available=True,
    )

    epoch = mx.array(91, dtype=mx.uint32)
    observed = router(value, epoch=epoch)

    assert observed is expected
    assert calls == [
        (
            value,
            weight,
            expert_bias,
            {
                "available": True,
                "top_k": 8,
                "route_norm": True,
                "scaling_factor": 2.826,
                "sigmoid_mode": "precise",
                "epoch": epoch,
            },
        )
    ]


def test_last_arrival_runtime_fails_before_dispatch_outside_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("an ineligible route must not build or dispatch a kernel")

    monkeypatch.setattr(
        last_arrival,
        "_build_hy3_router_last_arrival_kernel",
        forbidden_kernel,
    )
    value, weight, expert_bias = _runtime_inputs()
    epoch = mx.array(1, dtype=mx.uint32)

    for invalid_shape in ((1, 0, 4096), (1, 9, 4096), (2, 2, 4096)):
        with pytest.raises(Hy3RouterFP32Ineligible, match=r"\[1, M, 4096\]"):
            last_arrival.hy3_router_last_arrival_route(
                mx.zeros(invalid_shape, dtype=mx.float32),
                weight,
                expert_bias,
                available=True,
                epoch=epoch,
            )
    with pytest.raises(Hy3RouterFP32Ineligible, match=r"\[1, M, 4096\]"):
        last_arrival.hy3_router_last_arrival_route(
            mx.zeros((4, 4096), dtype=mx.float32),
            weight,
            expert_bias,
            available=True,
            epoch=epoch,
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="K-major BF16"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            mx.zeros((192, 4096), dtype=mx.bfloat16),
            expert_bias,
            available=True,
            epoch=epoch,
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="precise"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            weight,
            expert_bias,
            available=True,
            sigmoid_mode="fast",
            epoch=epoch,
        )
    with pytest.raises(Hy3RouterFP32Ineligible, match="top-8"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            weight,
            expert_bias,
            available=True,
            top_k=4,
            epoch=epoch,
        )


@pytest.mark.parametrize(
    "epoch",
    (
        mx.array([1], dtype=mx.uint32),
        mx.array(1, dtype=mx.int32),
    ),
)
def test_last_arrival_runtime_rejects_non_scalar_u32_epoch_before_dispatch(
    epoch: mx.array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        last_arrival,
        "_build_hy3_router_last_arrival_kernel",
        lambda *_args, **_kwargs: pytest.fail("invalid epoch reached Metal dispatch"),
    )
    value, weight, expert_bias = _runtime_inputs()

    with pytest.raises(Hy3RouterFP32Ineligible, match="scalar uint32 epoch"):
        last_arrival.hy3_router_last_arrival_route(
            value,
            weight,
            expert_bias,
            available=True,
            epoch=epoch,
        )


def test_last_arrival_runtime_epochs_do_not_repeat_with_reused_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(last_arrival, "_ROUTER_EPOCH", 580_000)

    first = last_arrival.new_hy3_router_forward_epoch(3)
    second = last_arrival.new_hy3_router_forward_epoch(2)
    mx.eval(first, second)

    assert first.tolist() == [580_001, 580_002, 580_003]
    assert second.tolist() == [580_004, 580_005]


def test_last_arrival_epoch_reservation_fails_before_uint32_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(last_arrival, "_ROUTER_EPOCH", 0xFFFFFFFF - 2)
    final = last_arrival.new_hy3_router_forward_epoch(2)
    mx.eval(final)
    assert final.tolist() == [0xFFFFFFFF - 1, 0xFFFFFFFF]

    with pytest.raises(RuntimeError, match="epoch space is exhausted"):
        last_arrival.new_hy3_router_forward_epoch(1)


@pytest.mark.skipif(
    os.environ.get("MTPLX_RUN_ISSUE58_HARDWARE_PARITY") != "1",
    reason="Issue #58 parity requires an explicitly locked Metal hardware gate",
)
@pytest.mark.parametrize("rows", tuple(range(1, 9)))
def test_last_arrival_runtime_matches_issue59_authoritative_m1_to_m8_routes(
    rows: int,
) -> None:
    mx.random.seed(58_590_000 + rows)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    resident_weight = prepare_hy3_router_fp32_weight(source_weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    value = mx.random.normal((1, rows, 4096)).astype(mx.float32)

    expected_ids, expected_weights = hy3_router_fp32_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
        n_tile=16,
        grid_k_parts=16,
        operand_mode="grouped-direct",
        simd_groups_per_threadgroup=4,
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode="simd",
        simd_groups=1,
        sigmoid_mode="precise",
    )
    first = last_arrival.hy3_router_last_arrival_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
        epoch=mx.array(1, dtype=mx.uint32),
    )
    second = last_arrival.hy3_router_last_arrival_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
        epoch=mx.array(2, dtype=mx.uint32),
    )
    mx.eval(
        expected_ids,
        expected_weights,
        first.expert_ids,
        first.route_weights,
        second.expert_ids,
        second.route_weights,
    )

    assert bool(mx.array_equal(first.expert_ids, expected_ids).item())
    assert bool(mx.array_equal(first.route_weights, expected_weights).item())
    assert bool(mx.array_equal(second.expert_ids, first.expert_ids).item())
    assert bool(mx.array_equal(second.route_weights, first.route_weights).item())


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
@pytest.mark.parametrize(
    "phase", ("prefill", "ar_decode", "mtp_draft", "decode_verify", "unknown")
)
def test_issue58_selector_reuses_one_issue59_weight_and_dispatches_m1_to_m8(
    rows: int,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router()
    source_weight = router.gate.weight
    calls: list[tuple[object, object, object, dict[str, object]]] = []

    def fake_route(value, weight, expert_bias, epoch, *, scaling_factor):
        calls.append(
            (
                value,
                weight,
                expert_bias,
                {"epoch": epoch, "scaling_factor": scaling_factor},
            )
        )
        return last_arrival.Hy3RouterLastArrivalOutput(
            expert_ids=mx.zeros((1, rows, 8), dtype=mx.int32),
            route_weights=mx.ones((1, rows, 8), dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "_dispatch_hy3_router_last_arrival", fake_route)

    report = router.configure_kernel(_SELECTOR, available=True)
    epoch_block = mx.array([117], dtype=mx.uint32)
    with attention_phase(phase), last_arrival.hy3_router_forward_epoch(epoch_block):
        indices, weights = router(mx.zeros((1, rows, 4096), dtype=mx.bfloat16))

    assert report["selector"] == _SELECTOR
    assert report["dispatch_count"] == 1
    assert report["supported_rows"] == "1-8"
    assert report["physical_rows"] == 8
    assert report["sigmoid_mode"] == "precise"
    assert report["topology"] == "n16-p16-sg4-in-kernel-pad"
    assert report["threadgroups"] == 48
    assert report["authority_phases"] == "all"
    assert report["prepared_weight_bytes"] == 192 * 4096 * 2
    assert report["incremental_bytes"] == 192 * 4096 * 2
    assert router.gate.weight is source_weight
    assert tuple(indices.shape) == (1, rows, 8)
    assert tuple(weights.shape) == (1, rows, 8)
    assert len(calls) == 1
    value, resident_weight, resident_bias, kwargs = calls[0]
    assert value.dtype == mx.float32
    assert resident_weight is router._mtplx_router_kernel_state.prepared_weight
    assert tuple(resident_weight.shape) == (4096, 192)
    assert resident_weight.dtype == mx.bfloat16
    assert resident_bias is router.expert_bias
    assert kwargs == {
        "scaling_factor": 2.826,
        "epoch": epoch_block[0],
    }


def test_issue58_selector_uses_large_m_bulk_router_without_hybrid_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingGate(nn.Linear):
        def __init__(self) -> None:
            super().__init__(4096, 192, bias=False)
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return mx.zeros((*value.shape[:-1], 192), dtype=mx.float32)

    router = _router()
    gate = RecordingGate()
    gate.weight = router.gate.weight
    router.gate = gate
    router.configure_kernel(_SELECTOR, available=True)

    def forbidden_route(*_args, **_kwargs):
        raise AssertionError("input outside [1,M,4096] M1-M8 must use stock")

    monkeypatch.setattr(hy3_mlx, "_dispatch_hy3_router_last_arrival", forbidden_route)
    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_fp32_route",
        lambda *_args, **_kwargs: pytest.fail("#59 checker must not double-run"),
    )

    indices, weights = router(mx.zeros((1, 9, 4096), dtype=mx.bfloat16))

    assert gate.calls == 1
    assert tuple(indices.shape) == (1, 9, 8)
    assert tuple(weights.shape) == (1, 9, 8)


@pytest.mark.parametrize("shape", ((2, 2, 4096), (4, 4096)))
def test_issue58_selector_flattens_small_prefix_shapes_into_the_fused_kernel(
    shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router()
    calls: list[tuple[int, ...]] = []
    router.configure_kernel(_SELECTOR, available=True)

    def fake_route(value, *_args, **_kwargs):
        calls.append(tuple(value.shape))
        rows = int(value.shape[1])
        return last_arrival.Hy3RouterLastArrivalOutput(
            expert_ids=mx.zeros((1, rows, 8), dtype=mx.int32),
            route_weights=mx.ones((1, rows, 8), dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "_dispatch_hy3_router_last_arrival", fake_route)
    with last_arrival.hy3_router_forward_epoch(mx.array([151], dtype=mx.uint32)):
        indices, weights = router(mx.zeros(shape, dtype=mx.bfloat16))

    assert calls == [(1, 4, 4096)]
    assert tuple(indices.shape) == (*shape[:-1], 8)
    assert tuple(weights.shape) == (*shape[:-1], 8)


@pytest.mark.parametrize("rows", tuple(range(1, 9)))
@pytest.mark.parametrize(
    "phase", ("prefill", "ar_decode", "mtp_draft", "decode_verify", "unknown")
)
def test_issue58_selector_never_downgrades_an_eligible_call_without_an_epoch(
    rows: int,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router()
    router.configure_kernel(_SELECTOR, available=True)
    monkeypatch.setattr(
        hy3_mlx,
        "_dispatch_hy3_router_last_arrival",
        lambda *_args, **_kwargs: pytest.fail("missing epoch reached fused dispatch"),
    )
    monkeypatch.setattr(
        hy3_mlx,
        "hy3_router_fp32_route",
        lambda *_args, **_kwargs: pytest.fail("eligible R1 call silently downgraded"),
    )

    with attention_phase(phase):
        with pytest.raises(RuntimeError, match="requires a forward epoch block"):
            router(mx.zeros((1, rows, 4096), dtype=mx.bfloat16))


def test_issue58_runtime_allocates_one_epoch_per_forward_shared_by_all_routers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routers = (_router(), _router())
    for epoch_slot, router in enumerate(routers):
        router.configure_kernel(
            _SELECTOR,
            available=True,
            epoch_slot=epoch_slot,
        )
    observed_epochs: list[mx.array] = []
    allocated_blocks: list[mx.array] = []

    real_allocate = last_arrival.new_hy3_router_forward_epoch

    def recording_allocate(slots: int) -> mx.array:
        block = real_allocate(slots)
        allocated_blocks.append(block)
        return block

    monkeypatch.setattr(
        last_arrival,
        "new_hy3_router_forward_epoch",
        recording_allocate,
    )

    def fake_route(value, _weight, _bias, epoch, *, scaling_factor):
        del scaling_factor
        observed_epochs.append(epoch)
        rows = int(value.shape[1])
        return last_arrival.Hy3RouterLastArrivalOutput(
            expert_ids=mx.zeros((1, rows, 8), dtype=mx.int32),
            route_weights=mx.ones((1, rows, 8), dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "_dispatch_hy3_router_last_arrival", fake_route)

    class TwoRouterModel:
        _mtplx_hy3_router_kernel_report = {
            "selector": _SELECTOR,
            "router_count": 2,
            "target_router_count": 2,
            "mtp_router_count": 0,
        }

        def __call__(self, value, cache=None):
            del cache
            hidden = mx.zeros((1, int(value.shape[1]), 4096), dtype=mx.bfloat16)
            first = routers[0](hidden)
            second = routers[1](hidden)
            return first[1] + second[1]

    model = TwoRouterModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=object(),
        model_path=Path("tiny-hy3"),
        mtp_enabled=False,
        contract=MTPContract(),
    )
    # Runtime metadata is a load-time cache. Mutating the diagnostic report
    # after construction must not add report parsing to the forward hot path.
    model._mtplx_hy3_router_kernel_report = {
        "selector": "stock",
        "router_count": 99,
    }
    value = mx.zeros((1, 3), dtype=mx.int32)

    with attention_phase("decode_verify"):
        runtime.forward_ar(value)
        runtime.forward_ar(value)

    epoch_values = [int(epoch.item()) for epoch in observed_epochs]
    assert len(allocated_blocks) == 2
    assert [tuple(block.shape) for block in allocated_blocks] == [(2,), (2,)]
    assert len(set(epoch_values[:2])) == 2
    assert len(set(epoch_values[2:])) == 2
    assert set(epoch_values[:2]).isdisjoint(epoch_values[2:])


def test_last_arrival_hot_router_path_has_no_per_router_host_epoch_work() -> None:
    router_source = inspect.getsource(hy3_mlx.Router.__call__)
    last_arrival_branch = 'if state.selector == "mpp-r1-last-arrival-fused-r2":'
    prefix = router_source.split(last_arrival_branch, maxsplit=1)[0]

    assert last_arrival_branch in router_source
    assert "current_attention_phase" not in prefix
    assert "current_hy3_router_forward_epoch" not in prefix
    assert "next_epoch" not in router_source
    assert "cursor" not in router_source
    assert "_ROUTER_EPOCH_LOCK" not in router_source
    assert "_next_router_epoch" not in router_source
    assert not hasattr(last_arrival, "_next_router_epoch")


def test_issue58_configured_router_uses_only_the_prevalidated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router()
    router.configure_kernel(_SELECTOR, available=True)
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_dispatch(value, _weight, _bias, epoch, *, scaling_factor):
        calls.append((tuple(value.shape), int(epoch.item())))
        rows = int(value.shape[1])
        return last_arrival.Hy3RouterLastArrivalOutput(
            expert_ids=mx.zeros((1, rows, 8), dtype=mx.int32),
            route_weights=mx.ones((1, rows, 8), dtype=mx.float32),
        )

    monkeypatch.setattr(
        hy3_mlx,
        "_dispatch_hy3_router_last_arrival",
        fake_dispatch,
        raising=False,
    )
    monkeypatch.setattr(
        last_arrival,
        "hy3_router_last_arrival_route",
        lambda *_args, **_kwargs: pytest.fail(
            "configured Router re-entered the checked qualification wrapper"
        ),
    )

    with last_arrival.hy3_router_forward_epoch(mx.array([401], dtype=mx.uint32)):
        indices, weights = router(mx.zeros((1, 3, 4096), dtype=mx.bfloat16))

    assert calls == [((1, 3, 4096), 401)]
    assert tuple(indices.shape) == (1, 3, 8)
    assert tuple(weights.shape) == (1, 3, 8)


@pytest.mark.parametrize(
    "expert_bias",
    (
        mx.zeros((191,), dtype=mx.float32),
        mx.zeros((192,), dtype=mx.bfloat16),
    ),
)
def test_issue58_rejects_invalid_residents_during_configuration(
    expert_bias: mx.array,
) -> None:
    router = _router()
    router.expert_bias = expert_bias

    with pytest.raises(Hy3RouterFP32Ineligible, match="expert bias"):
        router.configure_kernel(_SELECTOR, available=True)


def test_issue58_configuration_assigns_phase_local_static_epoch_slots() -> None:
    class MTPContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = _router()

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target_a = _router()
            self.target_b = _router()
            self.mtp = MTPContainer()

    root = Root()
    report = hy3_mlx.configure_hy3_router_kernels(
        root,
        _SELECTOR,
        available=True,
    )

    assert report["target_router_count"] == 2
    assert report["mtp_router_count"] == 1
    assert {
        root.target_a._mtplx_router_kernel_state.epoch_slot,
        root.target_b._mtplx_router_kernel_state.epoch_slot,
    } == {0, 1}
    assert root.mtp.router._mtplx_router_kernel_state.epoch_slot == 0


def test_issue58_runtime_rejects_overlapping_target_and_mtp_router_counts() -> None:
    class InvalidReportModel:
        _mtplx_hy3_router_kernel_report = {
            "selector": _SELECTOR,
            "router_count": 2,
            "target_router_count": 2,
            "mtp_router_count": 2,
        }

    with pytest.raises(RuntimeError, match="invalid phase counts"):
        MTPLXRuntime(
            model=InvalidReportModel(),
            tokenizer=object(),
            model_path=Path("invalid-hy3-router-report"),
            mtp_enabled=True,
            contract=MTPContract(),
        )


def test_issue58_runtime_rejects_reentrant_target_forward_epoch_reuse() -> None:
    class ReentrantModel:
        _mtplx_hy3_router_kernel_report = {
            "selector": _SELECTOR,
            "router_count": 1,
            "target_router_count": 1,
            "mtp_router_count": 0,
        }

        def __init__(self) -> None:
            self.runtime = None
            self.reentered = False

        def __call__(self, value, cache=None):
            del cache
            if not self.reentered:
                self.reentered = True
                return self.runtime.forward_ar(value)
            return value

    model = ReentrantModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=object(),
        model_path=Path("reentrant-hy3-target"),
        mtp_enabled=False,
        contract=MTPContract(),
    )
    model.runtime = runtime

    with pytest.raises(RuntimeError, match="nested Hy3 router forward"):
        runtime.forward_ar(mx.zeros((1, 1), dtype=mx.int32))


def test_issue58_runtime_rejects_target_to_mtp_epoch_domain_reuse() -> None:
    class ReentrantModel:
        mtp = type("MTPRoot", (), {})()
        _mtplx_hy3_router_kernel_report = {
            "selector": _SELECTOR,
            "router_count": 2,
            "target_router_count": 1,
            "mtp_router_count": 1,
        }

        def __init__(self) -> None:
            self.runtime = None

        def __call__(self, value, cache=None):
            del cache
            return self.runtime.draft_mtp(value, mx.zeros((1, 1), dtype=mx.int32))

        def mtp_forward(self, hidden, _tokens, **_kwargs):
            return hidden

    model = ReentrantModel()
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=object(),
        model_path=Path("reentrant-hy3-target-mtp"),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    model.runtime = runtime

    with pytest.raises(RuntimeError, match="nested Hy3 router forward"):
        runtime.forward_ar(mx.zeros((1, 1), dtype=mx.int32))


def test_runtime_marks_every_mtp_model_forward_as_mtp_draft() -> None:
    observed_phases: list[str] = []

    class MTPModel:
        mtp = type("MTPRoot", (), {})()

        def mtp_forward(self, _hidden, _tokens, **_kwargs):
            observed_phases.append(current_attention_phase())
            return mx.zeros((1, 1, 8), dtype=mx.float32)

        def mtp_update_cache(self, _hidden, _tokens, **_kwargs):
            observed_phases.append(current_attention_phase())
            return mx.zeros((1, 1, 8), dtype=mx.float32)

    runtime = MTPLXRuntime(
        model=MTPModel(),
        tokenizer=object(),
        model_path=Path("tiny-hy3-mtp"),
        mtp_enabled=True,
        contract=MTPContract(),
    )

    assert current_attention_phase() == "unknown"
    runtime.draft_mtp(
        mx.zeros((1, 1, 8), dtype=mx.float32),
        mx.zeros((1, 1), dtype=mx.int32),
    )
    runtime.update_mtp_cache(
        mx.zeros((1, 1, 8), dtype=mx.float32),
        mx.zeros((1, 1), dtype=mx.int32),
    )

    assert observed_phases == ["mtp_draft", "mtp_draft"]
    assert current_attention_phase() == "unknown"


def test_issue58_runtime_supplies_one_epoch_block_to_mtp_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router()
    router.configure_kernel(_SELECTOR, available=True)
    observed: list[tuple[str, int]] = []

    def fake_route(value, _weight, _bias, epoch, *, scaling_factor):
        del scaling_factor
        observed.append((current_attention_phase(), int(epoch.item())))
        return last_arrival.Hy3RouterLastArrivalOutput(
            expert_ids=mx.zeros((1, 1, 8), dtype=mx.int32),
            route_weights=mx.ones((1, 1, 8), dtype=mx.float32),
        )

    monkeypatch.setattr(hy3_mlx, "_dispatch_hy3_router_last_arrival", fake_route)

    class MTPRouterModel:
        mtp = type("MTPRoot", (), {})()
        _mtplx_hy3_router_kernel_report = {
            "selector": _SELECTOR,
            "router_count": 1,
            "target_router_count": 0,
            "mtp_router_count": 1,
        }

        def mtp_forward(self, hidden, _tokens, **_kwargs):
            router(hidden)
            return mx.zeros((1, 1, 8), dtype=mx.float32)

    runtime = MTPLXRuntime(
        model=MTPRouterModel(),
        tokenizer=object(),
        model_path=Path("tiny-hy3-mtp-router"),
        mtp_enabled=True,
        contract=MTPContract(),
    )

    hidden = mx.zeros((1, 1, 4096), dtype=mx.bfloat16)
    tokens = mx.zeros((1, 1), dtype=mx.int32)
    runtime.draft_mtp(hidden, tokens)
    runtime.draft_mtp(hidden, tokens)

    assert [phase for phase, _epoch in observed] == ["mtp_draft", "mtp_draft"]
    assert observed[0][1] != observed[1][1]


def test_issue58_runtime_skips_epoch_allocation_for_large_m_bulk_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, mx.array | None]] = []
    allocations: list[int] = []
    real_allocate = last_arrival.new_hy3_router_forward_epoch

    def recording_allocate(slots: int) -> mx.array:
        allocations.append(int(slots))
        return real_allocate(slots)

    monkeypatch.setattr(
        last_arrival,
        "new_hy3_router_forward_epoch",
        recording_allocate,
    )

    class Model:
        mtp = type("MTPRoot", (), {})()
        _mtplx_hy3_router_kernel_report = {
            "selector": _SELECTOR,
            "router_count": 2,
            "target_router_count": 1,
            "mtp_router_count": 1,
        }

        def __call__(self, input_ids, cache=None):
            del cache
            observed.append(("target", last_arrival.current_hy3_router_forward_epoch()))
            return input_ids

        def mtp_forward(self, hidden, _tokens, **_kwargs):
            observed.append(("mtp", last_arrival.current_hy3_router_forward_epoch()))
            return hidden

    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=object(),
        model_path=Path("hy3-large-m-bulk"),
        mtp_enabled=True,
        contract=MTPContract(),
    )

    runtime.forward_ar(mx.zeros((1, 9), dtype=mx.int32))
    runtime.forward_ar(mx.zeros((1, 8), dtype=mx.int32))
    runtime.draft_mtp(
        mx.zeros((1, 9, 4), dtype=mx.float32),
        mx.zeros((1, 9), dtype=mx.int32),
    )
    runtime.draft_mtp(
        mx.zeros((1, 8, 4), dtype=mx.float32),
        mx.zeros((1, 8), dtype=mx.int32),
    )

    assert [kind for kind, _epoch in observed] == ["target", "target", "mtp", "mtp"]
    assert observed[0][1] is None
    assert observed[1][1] is not None
    assert observed[2][1] is None
    assert observed[3][1] is not None
    assert allocations == [1, 1]


@pytest.mark.skipif(
    os.environ.get("MTPLX_RUN_ISSUE58_HARDWARE_PARITY") != "1",
    reason="Issue #58 parity requires an explicitly locked Metal hardware gate",
)
@pytest.mark.parametrize("rows", (1, 3, 5, 6, 7, 8))
@pytest.mark.parametrize(
    "fixture_name", ("full-tie", "near-monotonic", "selected-ties")
)
def test_last_arrival_matches_issue59_on_adversarial_ties(
    rows: int,
    fixture_name: str,
) -> None:
    value = mx.zeros((1, rows, 4096), dtype=mx.float32)
    source_weight = mx.zeros((192, 4096), dtype=mx.bfloat16)
    resident_weight = prepare_hy3_router_fp32_weight(source_weight)
    if fixture_name == "full-tie":
        expert_bias = mx.zeros((192,), dtype=mx.float32)
    elif fixture_name == "near-monotonic":
        expert_bias = mx.arange(192, dtype=mx.float32) * 1e-7
    else:
        selected = {3, 17, 29, 61, 97, 133, 171, 190}
        expert_bias = mx.array(
            [0.25 if index in selected else 0.0 for index in range(192)],
            dtype=mx.float32,
        )

    expected_ids, expected_weights = hy3_router_fp32_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
        n_tile=16,
        grid_k_parts=16,
        operand_mode="grouped-direct",
        simd_groups_per_threadgroup=4,
        top_k=8,
        route_norm=True,
        scaling_factor=2.826,
        finalizer_mode="simd",
        simd_groups=1,
        sigmoid_mode="precise",
    )
    observed = last_arrival.hy3_router_last_arrival_route(
        value,
        resident_weight,
        expert_bias,
        available=True,
        epoch=mx.array(900 + rows, dtype=mx.uint32),
    )
    mx.eval(expected_ids, expected_weights, observed.expert_ids, observed.route_weights)

    assert bool(mx.array_equal(observed.expert_ids, expected_ids).item())
    assert bool(mx.array_equal(observed.route_weights, expected_weights).item())


def test_issue58_selector_does_not_admit_the_issue60_fast_mode() -> None:
    router = _router()

    with pytest.raises(ValueError, match="Hy3 router kernel"):
        router.configure_kernel("mpp-r1-last-arrival-fast-fused-r2", available=True)
