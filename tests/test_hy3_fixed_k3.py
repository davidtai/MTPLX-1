from __future__ import annotations

from dataclasses import dataclass

import pytest

from mtplx.hy3_fixed_k3 import (
    HY3_FIXED_K3_DRAFT_ROWS,
    HY3_FIXED_K3_TARGET_ROWS,
    HY3_FIXED_K3_TOP_K,
    FixedK3ContractError,
    FixedK3LayerCapture,
    FixedK3SweepPayload,
    Rows4ExpertWaveOutput,
    Rows4ExpertWaveRequest,
    Rows4RouterOutput,
    capture_fixed_k3_target_sweep,
)


@dataclass(frozen=True, slots=True)
class Shaped:
    shape: tuple[int, ...]
    label: str = ""


def _routes(*, dispatch_count: int = 1) -> Rows4RouterOutput:
    return Rows4RouterOutput(
        expert_ids=Shaped((1, 4, 8), "expert_ids"),
        route_weights=Shaped((1, 4, 8), "route_weights"),
        dispatch_count=dispatch_count,
    )


def _expert_output() -> Rows4ExpertWaveOutput:
    return Rows4ExpertWaveOutput(
        hidden_rows=Shaped((1, 4, 4096), "expert_hidden"),
        stage_capture={"gate_up_swiglu": "opaque-device-result"},
    )


def test_fixed_k3_geometry_is_one_current_plus_three_drafts() -> None:
    assert HY3_FIXED_K3_DRAFT_ROWS == 3
    assert HY3_FIXED_K3_TARGET_ROWS == 4
    assert HY3_FIXED_K3_TOP_K == 8


def test_rows4_router_contract_is_one_m4_top8_dispatch() -> None:
    routes = _routes()

    assert routes.batch_shape == (1,)
    assert routes.rows == 4
    assert routes.top_k == 8
    assert routes.assignment_count == 32
    assert routes.dispatch_count == 1


@pytest.mark.parametrize(
    ("ids_shape", "weights_shape", "dispatch_count", "match"),
    [
        ((1, 3, 8), (1, 3, 8), 1, "Rows4"),
        ((1, 4, 7), (1, 4, 7), 1, "top-8"),
        ((1, 4, 8), (4, 8), 1, "batch prefixes"),
        ((1, 4, 8), (1, 4, 8), 2, "one dispatch"),
    ],
)
def test_rows4_router_contract_fails_closed_outside_issue58_59_seam(
    ids_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    dispatch_count: int,
    match: str,
) -> None:
    with pytest.raises(FixedK3ContractError, match=match):
        Rows4RouterOutput(
            expert_ids=Shaped(ids_shape),
            route_weights=Shaped(weights_shape),
            dispatch_count=dispatch_count,
        )


def test_rows4_expert_wave_contract_preserves_all_32_routed_assignments() -> None:
    request = Rows4ExpertWaveRequest(
        hidden_rows=Shaped((1, 4, 4096), "router_input"),
        routes=_routes(),
    )
    output = _expert_output()

    assert request.batch_shape == output.batch_shape == (1,)
    assert request.assignment_count == 4 * 8
    assert output.rows == 4
    assert output.hidden_size == 4096


def test_rows4_expert_wave_rejects_row_or_batch_shape_drift() -> None:
    with pytest.raises(FixedK3ContractError, match="hidden rows"):
        Rows4ExpertWaveRequest(
            hidden_rows=Shaped((1, 3, 4096)),
            routes=_routes(),
        )
    with pytest.raises(FixedK3ContractError, match="batch prefixes"):
        Rows4ExpertWaveOutput(hidden_rows=Shaped((4, 4096)))


def test_one_captured_m4_sweep_builds_rejection_commit_without_target_rerun() -> None:
    calls = {"capture": 0, "ordinary_forward": 0}
    cache = object()
    cache_capture = object()
    layer = FixedK3LayerCapture(
        layer_index=7,
        routes=_routes(),
        expert_wave=_expert_output(),
    )

    def target_capture(input_ids: Shaped, live_cache: object) -> FixedK3SweepPayload:
        calls["capture"] += 1
        assert input_ids.shape == (1, 4)
        assert live_cache is cache
        return FixedK3SweepPayload(
            logits=Shaped((1, 4, 128), "logits"),
            hidden_rows=Shaped((1, 4, 4096), "hidden"),
            cache_capture=cache_capture,
            layer_captures=(layer,),
        )

    capture = capture_fixed_k3_target_sweep(
        Shaped((1, 4), "tokens"),
        cache=cache,
        target_capture=target_capture,
    )
    request = capture.build_commit_request(
        accepted_drafts=1,
        correction_token=93,
    )

    assert calls == {"capture": 1, "ordinary_forward": 0}
    assert capture.target_forward_calls == 1
    assert request.verified_tokens == 4
    assert request.keep_tokens == 2
    assert request.selected_row == 1
    assert request.correction_token == 93
    assert request.cache_capture is cache_capture
    assert request.logits is capture.payload.logits
    assert request.hidden_rows is capture.payload.hidden_rows

    with pytest.raises(FixedK3ContractError, match="already consumed"):
        capture.build_commit_request(accepted_drafts=1, correction_token=93)
    assert calls == {"capture": 1, "ordinary_forward": 0}


def test_one_captured_m4_sweep_builds_accept_all_commit_without_correction() -> None:
    payload = FixedK3SweepPayload(
        logits=Shaped((1, 4, 128)),
        hidden_rows=Shaped((1, 4, 4096)),
        cache_capture=object(),
        layer_captures=(),
    )
    capture = capture_fixed_k3_target_sweep(
        Shaped((1, 4)),
        cache=object(),
        target_capture=lambda _input_ids, _cache: payload,
    )

    request = capture.build_commit_request(
        accepted_drafts=3,
        correction_token=None,
    )

    assert request.keep_tokens == 4
    assert request.selected_row == 3
    assert request.correction_token is None


def test_fixed_k3_capture_and_commit_fail_closed_on_non_m4_semantics() -> None:
    payload = FixedK3SweepPayload(
        logits=Shaped((1, 4, 128)),
        hidden_rows=Shaped((1, 4, 4096)),
        cache_capture=object(),
        layer_captures=(),
    )
    with pytest.raises(FixedK3ContractError, match="input ids"):
        capture_fixed_k3_target_sweep(
            Shaped((1, 3)),
            cache=object(),
            target_capture=lambda _input_ids, _cache: payload,
        )

    capture = capture_fixed_k3_target_sweep(
        Shaped((1, 4)),
        cache=object(),
        target_capture=lambda _input_ids, _cache: payload,
    )
    with pytest.raises(FixedK3ContractError, match="requires a correction"):
        capture.build_commit_request(accepted_drafts=0, correction_token=None)


def test_fixed_k3_payload_requires_unique_aligned_layer_captures() -> None:
    layer = FixedK3LayerCapture(
        layer_index=3,
        routes=_routes(),
        expert_wave=_expert_output(),
    )
    with pytest.raises(FixedK3ContractError, match="unique"):
        FixedK3SweepPayload(
            logits=Shaped((1, 4, 128)),
            hidden_rows=Shaped((1, 4, 4096)),
            cache_capture=object(),
            layer_captures=(layer, layer),
        )
