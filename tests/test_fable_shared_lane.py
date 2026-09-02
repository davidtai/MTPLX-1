"""MTPLX_FABLE_SHARED_LANE — the gate, the contract, the wiring, the micro.

Everything here runs on the CPU stream with stub objects: no Metal, no kernel
dispatch, no model, no array evaluation.  The lane has no kernel of its own —
it re-homes three existing dispatches onto a second ``mx.gpu`` stream — so the
things worth proving on the CPU are:

1. THE GATE.  Off by default; memoized once; refuses to arm outside the paired
   routed-GLU lane and outside M4 stage3.  A flag that installs on a lane it
   was not measured against produces a meaningless A/B.

2. THE CONTRACT.  The lane's justification is the measured dispatch anatomy of
   a q8/group-64 shared expert (see the module docstring of
   ``mtplx/kernels/qwen4_m4_shared_lane``).  Every field that anatomy depends
   on is checked at install and RAISES with the offending field named — there
   is no silent fallback, because arming against a different pack means the arm
   measured something else.

3. THE WIRING.  ``shared_lane=True`` must actually reach the branch.  A lane
   that reads flat in an A/B because it never engaged is the failure mode the
   counters law exists to prevent, so the counters are asserted too.

4. THE BYTE MODEL, pinned against the retained-stack census so a shape drift
   shows up here rather than as a quietly wrong ms/cycle claim in a writeup.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.kernels.qwen4_m4_shared_lane as lane
import mtplx.qwen4_m4_stage3 as stage3

HIDDEN = lane.HIDDEN
INTERMEDIATE = lane.INTERMEDIATE
ROWS = lane.ROWS


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Default every test to the shipped state, whatever the session env is."""

    monkeypatch.delenv(stage3.FABLE_SHARED_LANE_ENV, raising=False)
    stage3.reset_fable_shared_lane_cache()
    lane.reset_counters()
    lane.reset_stream_cache()
    yield
    stage3.reset_fable_shared_lane_cache()
    lane.reset_counters()
    lane.reset_stream_cache()


class _ArraySpec:
    """Shape/dtype only: the contract check reads nothing else."""

    def __init__(self, shape, dtype=mx.uint32) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


def _shared_expert(
    *,
    bits=lane.SHARED_BITS,
    group_size=lane.SHARED_GROUP_SIZE,
    mode=lane.SHARED_MODE,
    gu_weight_shape=None,
    gu_meta_shape=None,
    down_weight_shape=None,
    down_meta_shape=None,
    gu_weight_dtype=mx.uint32,
    down_weight_dtype=mx.uint32,
    down_bias=None,
):
    gu_weight_shape = gu_weight_shape or (2 * INTERMEDIATE, HIDDEN * 8 // 32)
    gu_meta_shape = gu_meta_shape or (2 * INTERMEDIATE, HIDDEN // 64)
    down_weight_shape = down_weight_shape or (HIDDEN, INTERMEDIATE * 8 // 32)
    down_meta_shape = down_meta_shape or (HIDDEN, INTERMEDIATE // 64)
    down = SimpleNamespace(
        bits=bits,
        group_size=group_size,
        mode=mode,
        weight=_ArraySpec(down_weight_shape, down_weight_dtype),
        scales=_ArraySpec(down_meta_shape, mx.bfloat16),
        biases=_ArraySpec(down_meta_shape, mx.bfloat16),
        bias=down_bias,
    )
    return SimpleNamespace(
        bits=bits,
        group_size=group_size,
        mode=mode,
        gu_weight=_ArraySpec(gu_weight_shape, gu_weight_dtype),
        gu_scales=_ArraySpec(gu_meta_shape, mx.bfloat16),
        gu_biases=_ArraySpec(gu_meta_shape, mx.bfloat16),
        down_proj=down,
    )


def _block(**kwargs):
    return SimpleNamespace(shared_expert=_shared_expert(**kwargs))


# --------------------------------------------------------------------------
# 1. the gate
# --------------------------------------------------------------------------


def test_gate_defaults_off():
    assert stage3.fable_shared_lane_enabled() is False


def test_gate_reads_the_environment_once(monkeypatch):
    monkeypatch.setenv(stage3.FABLE_SHARED_LANE_ENV, "1")
    assert stage3.fable_shared_lane_enabled() is True
    # Memoized: a later unset must not change the answer mid-process, or two
    # halves of one decode could disagree about which lane they are on.
    monkeypatch.delenv(stage3.FABLE_SHARED_LANE_ENV)
    assert stage3.fable_shared_lane_enabled() is True
    stage3.reset_fable_shared_lane_cache()
    assert stage3.fable_shared_lane_enabled() is False


def test_gate_rejects_an_unknown_spelling(monkeypatch):
    monkeypatch.setenv(stage3.FABLE_SHARED_LANE_ENV, "yes-please")
    with pytest.raises(ValueError):
        stage3.fable_shared_lane_enabled()


def test_lane_requires_the_paired_routed_glu():
    with pytest.raises(ValueError, match="MTPLX_QWEN4_M4_ROUTED_GLU"):
        stage3._validate_feature_combination(
            routed_down_reduce_enabled=True,
            routed_down_residual_tail_enabled=True,
            routed_glu_enabled=False,
            shared_lane_enabled=True,
        )


def test_lane_is_accepted_on_the_paired_routed_glu_lane():
    stage3._validate_feature_combination(
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
        shared_lane_enabled=True,
    )


def test_lane_requires_stage3(monkeypatch):
    monkeypatch.setenv(stage3.FABLE_SHARED_LANE_ENV, "1")
    monkeypatch.delenv("MTPLX_QWEN4_M4_STAGE3", raising=False)
    with pytest.raises(ValueError, match="require M4 stage3"):
        stage3.qwen4_m4_stage3_flags()


# --------------------------------------------------------------------------
# 2. the contract
# --------------------------------------------------------------------------


def test_contract_accepts_the_shipped_pack():
    lane.check_contract(_block(), index=0)
    assert lane.COUNTERS["contract_checks"] == 1


def test_contract_rejects_a_block_with_no_shared_expert():
    with pytest.raises(lane.SharedLaneContractError, match="no shared_expert"):
        lane.check_contract(SimpleNamespace(shared_expert=None), index=3)


def test_contract_rejects_an_unfused_shared_expert():
    shared = _shared_expert()
    del shared.gu_weight
    with pytest.raises(lane.SharedLaneContractError, match="gu_weight"):
        lane.check_contract(SimpleNamespace(shared_expert=shared), index=0)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"bits": 4}, "bits=4"),
        ({"group_size": 32}, "group_size=32"),
        ({"mode": "mxfp4"}, "mxfp4"),
    ],
)
def test_contract_rejects_a_different_pack(kwargs, needle):
    with pytest.raises(lane.SharedLaneContractError, match=needle):
        lane.check_contract(_block(**kwargs), index=7)


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"gu_weight_shape": (1281, 640)}, "gu_weight"),
        ({"gu_meta_shape": (1280, 41)}, "gu_scales"),
        ({"down_weight_shape": (2560, 161)}, "down_proj.weight"),
        ({"down_meta_shape": (2560, 11)}, "down_proj.scales"),
    ],
)
def test_contract_rejects_a_wrong_shape(kwargs, needle):
    with pytest.raises(lane.SharedLaneContractError, match=needle):
        lane.check_contract(_block(**kwargs), index=11)


def test_contract_rejects_unpacked_weights():
    with pytest.raises(lane.SharedLaneContractError, match="expected uint32"):
        lane.check_contract(_block(gu_weight_dtype=mx.bfloat16), index=0)


def test_contract_rejects_a_biased_down_projection():
    with pytest.raises(lane.SharedLaneContractError, match="carries a bias"):
        lane.check_contract(_block(down_bias=_ArraySpec((HIDDEN,))), index=0)


def test_contract_names_the_layer():
    with pytest.raises(lane.SharedLaneContractError, match="layer 41"):
        lane.check_contract(_block(bits=4), index=41)


# --------------------------------------------------------------------------
# 3. the wiring
# --------------------------------------------------------------------------


class _RowsStub:
    def __init__(self, tag):
        self.tag = tag

    def reshape(self, *_shape):
        return self


def _paired_forward(monkeypatch, *, shared_lane):
    """Drive the retained forward with every kernel stubbed out.

    Only the branch selection is under test, so the route head, the paired GLU
    and the residual tail are recorded rather than run.
    """

    calls = []
    monkeypatch.setattr(
        lane,
        "shared_branch",
        lambda block, x: calls.append("lane") or "shared_down",
    )
    monkeypatch.setattr(
        lane,
        "stock_shared_branch",
        lambda block, x: calls.append("stock") or "shared_down",
    )
    monkeypatch.setattr(stage3._census, "enabled", False, raising=False)

    seen = {}

    def _tail(routed_h, w, s, b, ids, scores, shared_down, factor, hyper, inject):
        seen["shared_down"] = shared_down
        return "tail-out"

    routed = SimpleNamespace(
        gu_weight=None, gu_scales=None, gu_biases=None,
        down_proj=SimpleNamespace(weight=None, scales=None, biases=None),
    )
    gate = SimpleNamespace(weight=None, scales=None, biases=None)
    block = SimpleNamespace(
        switch_mlp=routed,
        gate=gate,
        shared_expert_gate=gate,
        shared_expert=_shared_expert(),
        _mtplx_m4_layer_index=0,
    )
    out = stage3._m4_paired_routed_glu_residual_tail_forward(
        block,
        _RowsStub("x"),
        lambda rows, w, s, b, ids: "routed_h",
        _tail,
        "hyper",
        "inject",
        route=lambda *args: ("ids", SimpleNamespace(astype=lambda _d: "scores"),
                             SimpleNamespace(astype=lambda _d: "factor")),
        shared_lane=shared_lane,
    )
    return out, calls, seen


def test_forward_takes_the_stock_branch_by_default(monkeypatch):
    out, calls, seen = _paired_forward(monkeypatch, shared_lane=False)
    assert out == "tail-out"
    assert calls == ["stock"]
    assert seen["shared_down"] == "shared_down"


def test_forward_takes_the_lane_when_armed(monkeypatch):
    out, calls, seen = _paired_forward(monkeypatch, shared_lane=True)
    assert out == "tail-out"
    assert calls == ["lane"]
    assert seen["shared_down"] == "shared_down"


def test_the_two_branches_share_one_definition(monkeypatch):
    """Stock and lane must emit the same ops, or exactness is a coincidence."""

    seen = []
    monkeypatch.setattr(
        lane, "_emit_branch", lambda block, x: seen.append((block, x)) or "out"
    )
    monkeypatch.setattr(lane, "stream", lambda: mx.cpu)
    assert lane.stock_shared_branch("b", "x") == "out"
    assert lane.shared_branch("b", "x") == "out"
    assert seen == [("b", "x"), ("b", "x")]
    assert lane.COUNTERS["stock_calls"] == 1
    assert lane.COUNTERS["branch_calls"] == 1


def test_the_stream_is_created_once(monkeypatch):
    created = []
    monkeypatch.setattr(
        mx, "new_stream", lambda device: created.append(device) or "stream"
    )
    assert lane.stream() == "stream"
    assert lane.stream() == "stream"
    assert len(created) == 1
    assert lane.COUNTERS["streams_created"] == 1
    lane.reset_stream_cache()
    assert lane.stream() == "stream"
    assert len(created) == 2


def test_counters_reset():
    lane.COUNTERS["branch_calls"] = 5
    lane.reset_counters()
    assert set(lane.COUNTERS.values()) == {0}


# --------------------------------------------------------------------------
# 4. the byte model and the receipts
# --------------------------------------------------------------------------


def test_byte_model_matches_the_census():
    # q8/group-64: one byte per weight plus a bf16 scale and bias per 64.
    assert lane.GU_BYTES_PER_LAYER == 2 * INTERMEDIATE * HIDDEN + 2 * 2 * (
        2 * INTERMEDIATE
    ) * (HIDDEN // 64)
    assert lane.DOWN_BYTES_PER_LAYER == HIDDEN * INTERMEDIATE + 2 * 2 * HIDDEN * (
        INTERMEDIATE // 64
    )
    assert lane.BYTES_PER_LAYER == 5_222_400
    # 48 MoE layers; the census's "MoE shared 251.7 MB/cyc" row, less the
    # 2.7 kB/layer scalar gate that the lane does not move.
    assert lane.BYTES_PER_LAYER * 48 == 250_675_200


def test_engagement_line_distinguishes_off_from_on():
    assert lane.engagement_line(installed_layers=0, enabled=False) == (
        "[fable] shared-lane: off"
    )
    line = lane.engagement_line(installed_layers=48, enabled=True)
    assert "layers=48" in line
    assert "exactness_failures=0" in line
    assert str(lane.BYTES_PER_LAYER) in line


def test_installation_report_separates_armed_from_installed():
    off = stage3._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
    )
    assert off["shared_lane"]["armed"] is False
    assert off["shared_lane"]["installed"] is False
    assert off["shared_lane"]["layers"] == 0
    assert off["shared_lane"]["fence_dispatches_per_layer"] == 0

    on = stage3._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
        shared_lane_enabled=True,
        shared_lane_layers=48,
    )
    assert on["shared_lane"]["armed"] is True
    assert on["shared_lane"]["installed"] is True
    assert on["shared_lane"]["layers"] == 48
    assert on["shared_lane"]["branch_dispatches_per_layer"] == 3
    assert on["shared_lane"]["fence_dispatches_per_layer"] == 4
    assert on["shared_lane"]["weight_bytes_per_layer"] == lane.BYTES_PER_LAYER


def test_a_disabled_lane_is_reported_as_armed_but_not_installed():
    """The A/B must be able to tell 'ran and did nothing' from 'never ran'."""

    report = stage3._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
        shared_lane_enabled=True,
        shared_lane_layers=0,
    )
    assert report["shared_lane"]["armed"] is True
    assert report["shared_lane"]["installed"] is False


def test_install_validates_the_gate_before_binding_anything():
    """Regression: the gate must reach install's OWN validation call.

    ``install_qwen4_m4_stage3`` re-validates rather than trusting
    ``qwen4_m4_stage3_flags``.  If the lane's gate is not in that call, an
    install reached directly with the lane armed and the paired routed-GLU lane
    off installs nothing at all -- ``_install_validated_plans`` only sets
    ``_mtplx_m4_shared_lane`` on the routed-GLU branch -- and the resulting flat
    A/B reads as "the lane did not help" instead of "the lane never ran".
    """

    import inspect
    import re

    source = inspect.getsource(stage3.install_qwen4_m4_stage3)
    call = re.search(
        r"_validate_feature_combination\((.*?)\n    \)", source, re.S
    )
    assert call is not None
    assert "shared_lane_enabled=shared_lane_enabled" in call.group(1)
    # ...and the gate must be read before that call, not beside its first use.
    assert source.index("shared_lane_enabled = fable_shared_lane_enabled()") < (
        source.index("_validate_feature_combination(")
    )


def test_install_only_arms_the_lane_on_the_paired_glu_branch():
    """``_install_validated_plans`` must not set the attribute off-lane."""

    import inspect

    source = inspect.getsource(stage3._install_validated_plans)
    glu_branch = source.index("if routed_glu_enabled:")
    next_branch = source.index("elif routed_down_residual_tail_enabled:")
    assignment = source.index("layer._mtplx_m4_shared_lane")
    assert glu_branch < assignment < next_branch
