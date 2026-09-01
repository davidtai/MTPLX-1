"""Contract tests for sorted-adjacent routed gathers (MTPLX_FABLE_MOE_SORTED).

Everything here runs on the CPU stream against stub routed/shared owners: the
gate is a pure permutation of forty independent M=1 gather rows, so what has to
be proven is that the permutation and its inverse cancel exactly and that every
downstream consumer still sees the original ``[4, 10, ...]`` slot order.  No
Metal kernel is dispatched -- the three fused kernels are injected as callables
by the forwards under test, so the stubs stand in for them directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.qwen4_m4_stage3 import (
    FABLE_MOE_SORTED_ENV,
    HIDDEN,
    INTERMEDIATE,
    ROWS,
    TOP_K,
    _m4_forward,
    _m4_paired_routed_glu_residual_tail_forward,
    _m4_routed_down_reduce_forward,
    _m4_routed_down_residual_tail_forward,
    _routed_gather_plan,
    fable_moe_sorted_enabled,
    reset_fable_moe_sorted_cache,
)

NUM_EXPERTS = 512
PAIRS = ROWS * TOP_K


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Pin every op in this module to the CPU stream, then put it back.

    Restoring the previous device keeps this module from changing the device
    for any other test that shares the process.
    """

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    """Never let a memoized gate leak between cases (or in from the caller)."""

    monkeypatch.delenv(FABLE_MOE_SORTED_ENV, raising=False)
    reset_fable_moe_sorted_cache()
    try:
        yield
    finally:
        reset_fable_moe_sorted_cache()


def _arm(monkeypatch, value: str) -> None:
    monkeypatch.setenv(FABLE_MOE_SORTED_ENV, value)
    reset_fable_moe_sorted_cache()


# ---------------------------------------------------------------------------
# stub owners
# ---------------------------------------------------------------------------


def _emulate_gather_qmm(x: mx.array, idx: mx.array, n: int) -> mx.array:
    """Mimic ``mx.gather_qmm(x[..., 1, K], rhs_indices=idx) -> [..., 1, N]``.

    Every output element is a deterministic function of the input row's
    contents, the expert id and the output column, and of nothing else --
    which is exactly the invariant a pure row permutation must preserve.  The
    arithmetic is elementwise, so reordering the rows reorders the results
    bitwise and nothing more.
    """

    k = int(x.shape[-1])
    row = mx.squeeze(x, -2).astype(mx.float32)
    reps = -(-n // k)
    base = mx.concatenate([row] * reps, axis=-1)[..., :n] if reps > 1 else row[..., :n]
    base = mx.broadcast_to(base, tuple(idx.shape) + (n,))
    ids = idx.astype(mx.float32)[..., None]
    column = mx.arange(n, dtype=mx.float32) * 0.00048828125
    out = base * (ids * 0.5 + 1.0) + column + ids * 0.0078125
    return mx.expand_dims(out.astype(mx.bfloat16), -2)


class _StubSwitchGLU:
    """Records the routed gather calls and answers them elementwise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], tuple[int, ...], bool]] = []
        self.gu_weight = mx.zeros((1,), dtype=mx.uint32)
        self.gu_scales = mx.zeros((1,), dtype=mx.bfloat16)
        self.gu_biases = mx.zeros((1,), dtype=mx.bfloat16)
        self.down_proj = _StubDownProj(self)

    def _gu(self, x, idx, sorted_indices=False):
        self.calls.append(
            ("gu", tuple(x.shape), tuple(idx.shape), bool(sorted_indices))
        )
        gu = _emulate_gather_qmm(x, idx, 2 * INTERMEDIATE)
        assert tuple(gu.shape) == tuple(idx.shape) + (1, 2 * INTERMEDIATE)
        return mx.split(gu, 2, axis=-1)

    def _down(self, h, idx, sorted_indices=False):
        self.calls.append(
            ("down", tuple(h.shape), tuple(idx.shape), bool(sorted_indices))
        )
        out = _emulate_gather_qmm(h, idx, HIDDEN)
        assert tuple(out.shape) == tuple(idx.shape) + (1, HIDDEN)
        return out


class _StubDownProj:
    def __init__(self, owner: _StubSwitchGLU) -> None:
        self._owner = owner
        self.weight = mx.zeros((1,), dtype=mx.uint32)
        self.scales = mx.zeros((1,), dtype=mx.bfloat16)
        self.biases = mx.zeros((1,), dtype=mx.bfloat16)

    def __call__(self, h, idx, sorted_indices=False):
        return self._owner._down(h, idx, sorted_indices=sorted_indices)


def _router_logits() -> mx.array:
    """Logits whose per-row top-10 overlap heavily, like the real census."""

    experts = mx.arange(NUM_EXPERTS, dtype=mx.float32)
    # Every 13th expert is strongly attractive to every row, so the four rows
    # draw their top-10 from a shared pool of ~40 and duplicate hard.
    pool = (experts % 13 == 0).astype(mx.float32) * 4.0
    rows = mx.arange(ROWS, dtype=mx.float32).reshape(ROWS, 1)
    jitter = mx.sin(experts.reshape(1, NUM_EXPERTS) * 0.37 + rows * 1.1) * 0.9
    return (pool.reshape(1, NUM_EXPERTS) + jitter).reshape(1, ROWS, NUM_EXPERTS)


def _quantized_shared() -> SimpleNamespace:
    dense = mx.sin(
        mx.arange(8 * HIDDEN, dtype=mx.float32) * 0.0011
    ).reshape(8, HIDDEN).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(dense, group_size=64, bits=8)

    def down(h):
        folded = mx.sum(h.astype(mx.float32), axis=-1, keepdims=True)
        column = mx.arange(HIDDEN, dtype=mx.float32) * 0.00390625
        return (folded + column).astype(mx.bfloat16)

    return SimpleNamespace(
        gu_weight=weight,
        gu_scales=scales,
        gu_biases=biases,
        group_size=64,
        bits=8,
        mode="affine",
        down_proj=down,
    )


def _stub_block() -> SimpleNamespace:
    logits = _router_logits()
    return SimpleNamespace(
        gate=lambda x: logits,
        top_k=TOP_K,
        norm_topk_prob=True,
        switch_mlp=_StubSwitchGLU(),
        shared_expert=_quantized_shared(),
        shared_expert_gate=lambda x: mx.sum(
            x.astype(mx.float32), axis=-1, keepdims=True
        ).astype(mx.bfloat16),
    )


def _hidden() -> mx.array:
    return mx.sin(
        mx.arange(ROWS * HIDDEN, dtype=mx.float32) * 0.0009765625
    ).reshape(1, ROWS, HIDDEN).astype(mx.bfloat16)


def _hyper_inject() -> tuple[mx.array, mx.array]:
    hyper = mx.sin(
        mx.arange(ROWS * 4 * HIDDEN, dtype=mx.float32) * 0.000244140625
    ).reshape(1, ROWS, 4 * HIDDEN).astype(mx.bfloat16)
    inject = mx.cos(
        mx.arange(ROWS * 4, dtype=mx.float32) * 0.0625
    ).reshape(1, ROWS, 4).astype(mx.bfloat16)
    return hyper, inject


def _stage3(routed_down, shared_down, route_scores, shared_factor):
    """Stand in for the stage3 combine tail; sensitive to slot alignment."""

    assert tuple(routed_down.shape) == (ROWS, TOP_K, HIDDEN)
    assert tuple(route_scores.shape) == (ROWS, TOP_K)
    weighted = routed_down.astype(mx.float32) * route_scores.astype(mx.float32)[
        ..., None
    ]
    combined = mx.sum(weighted, axis=1) + shared_down.astype(
        mx.float32
    ) * shared_factor.astype(mx.float32)[:, None]
    return combined.astype(mx.bfloat16)


def _routed_reduce_core(routed_h, expert_ids, route_scores, shared_down, shared_factor):
    assert tuple(routed_h.shape) == (ROWS, TOP_K, INTERMEDIATE)
    assert tuple(expert_ids.shape) == (ROWS, TOP_K)
    tiled = mx.concatenate([routed_h] * (HIDDEN // INTERMEDIATE), axis=-1)
    ids = expert_ids.astype(mx.float32)[..., None]
    weighted = (
        tiled.astype(mx.float32)
        * (ids + 1.0)
        * route_scores.astype(mx.float32)[..., None]
    )
    return mx.sum(weighted, axis=1) + shared_down.astype(
        mx.float32
    ) * shared_factor.astype(mx.float32)[:, None]


def _routed_down_reduce(
    routed_h, weight, scales, biases, expert_ids, route_scores, shared_down, shared_factor
):
    return _routed_reduce_core(
        routed_h, expert_ids, route_scores, shared_down, shared_factor
    ).astype(mx.bfloat16)


def _routed_down_residual_tail(
    routed_h,
    weight,
    scales,
    biases,
    expert_ids,
    route_scores,
    shared_down,
    shared_factor,
    hyper,
    inject,
):
    block_out = _routed_reduce_core(
        routed_h, expert_ids, route_scores, shared_down, shared_factor
    )
    streams = hyper.reshape(ROWS, 4, HIDDEN).astype(mx.float32)
    scaled = block_out[:, None, :] * inject.reshape(ROWS, 4)[:, :, None].astype(
        mx.float32
    )
    return (streams + scaled).astype(mx.bfloat16).reshape(*hyper.shape)


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------


def test_gate_defaults_off():
    assert fable_moe_sorted_enabled() is False


def test_gate_is_read_once(monkeypatch):
    _arm(monkeypatch, "1")
    assert fable_moe_sorted_enabled() is True
    monkeypatch.delenv(FABLE_MOE_SORTED_ENV, raising=False)
    assert fable_moe_sorted_enabled() is True
    reset_fable_moe_sorted_cache()
    assert fable_moe_sorted_enabled() is False


def test_gate_rejects_an_unknown_spelling(monkeypatch):
    _arm(monkeypatch, "sorta")
    with pytest.raises(ValueError):
        fable_moe_sorted_enabled()


# ---------------------------------------------------------------------------
# the permutation
# ---------------------------------------------------------------------------


def _sample_ids(seed: int = 7) -> mx.array:
    """Forty routed ids with real duplicates across the four rows."""

    key = mx.random.key(seed)
    ids = mx.random.randint(0, 24, (1, ROWS, TOP_K), key=key).astype(mx.uint32)
    return ids


def test_plan_off_is_the_historical_layout():
    x = _hidden()
    ids = _sample_ids()
    routed_input, gather_ids, inverse = _routed_gather_plan(x, ids)
    assert tuple(routed_input.shape) == (1, ROWS, 1, 1, HIDDEN)
    assert gather_ids is ids
    assert inverse is None


def test_plan_on_sorts_duplicates_adjacent(monkeypatch):
    _arm(monkeypatch, "1")
    x = _hidden()
    ids = _sample_ids()
    routed_input, gather_ids, inverse = _routed_gather_plan(x, ids)

    assert tuple(routed_input.shape) == (PAIRS, 1, HIDDEN)
    assert tuple(gather_ids.shape) == (PAIRS,)
    assert tuple(inverse.shape) == (PAIRS,)

    flat = ids.reshape(PAIRS)
    sorted_ids = gather_ids.tolist()
    assert sorted_ids == sorted(flat.tolist())
    # A real overlap, otherwise the case proves nothing.
    assert len(set(sorted_ids)) < PAIRS

    # The inverse restores both the ids and the gathered hidden rows.
    assert gather_ids[inverse].tolist() == flat.tolist()
    rows = x.reshape(ROWS, 1, HIDDEN)
    restored = routed_input[inverse].reshape(ROWS, TOP_K, HIDDEN)
    expected = mx.broadcast_to(
        rows.reshape(ROWS, 1, HIDDEN), (ROWS, TOP_K, HIDDEN)
    )
    assert bool(mx.array_equal(restored, expected).item())


def test_plan_on_pairs_each_row_with_its_own_hidden(monkeypatch):
    _arm(monkeypatch, "1")
    x = _hidden()
    ids = _sample_ids()
    routed_input, _, _ = _routed_gather_plan(x, ids)
    order = mx.argsort(ids.reshape(PAIRS))
    rows = x.reshape(ROWS, 1, HIDDEN)
    assert bool(mx.array_equal(routed_input, rows[order // TOP_K]).item())


# ---------------------------------------------------------------------------
# the variants
# ---------------------------------------------------------------------------


def _run_m4_forward(block):
    return _m4_forward(block, _hidden(), _stage3)


def _run_reduce(block):
    return _m4_routed_down_reduce_forward(block, _hidden(), _routed_down_reduce)


def _run_residual_tail(block):
    hyper, inject = _hyper_inject()
    return _m4_routed_down_residual_tail_forward(
        block, _hidden(), _routed_down_residual_tail, hyper, inject
    )


VARIANTS = {
    "m4_forward": _run_m4_forward,
    "routed_down_reduce": _run_reduce,
    "routed_down_residual_tail": _run_residual_tail,
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_sorted_matches_unsorted_bitwise(monkeypatch, name):
    run = VARIANTS[name]

    unsorted_block = _stub_block()
    reference = run(unsorted_block)

    _arm(monkeypatch, "1")
    sorted_block = _stub_block()
    candidate = run(sorted_block)

    assert tuple(candidate.shape) == tuple(reference.shape)
    assert candidate.dtype == reference.dtype
    assert bool(mx.array_equal(reference, candidate).item())


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_flag_off_keeps_the_historical_call_sequence(name):
    block = _stub_block()
    VARIANTS[name](block)

    calls = block.switch_mlp.calls
    expected = [("gu", (1, ROWS, 1, 1, HIDDEN), (1, ROWS, TOP_K), False)]
    if name == "m4_forward":
        expected.append(
            ("down", (1, ROWS, TOP_K, 1, INTERMEDIATE), (1, ROWS, TOP_K), False)
        )
    assert calls == expected


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_flag_on_gathers_forty_flat_rows(monkeypatch, name):
    _arm(monkeypatch, "1")
    block = _stub_block()
    VARIANTS[name](block)

    calls = block.switch_mlp.calls
    expected = [("gu", (PAIRS, 1, HIDDEN), (PAIRS,), False)]
    if name == "m4_forward":
        expected.append(("down", (PAIRS, 1, INTERMEDIATE), (PAIRS,), False))
    assert calls == expected


def test_reduce_variants_hand_the_kernel_the_original_slot_order(monkeypatch):
    """The routed-down kernel indexes routed_h and expert_ids alike."""

    seen: dict[str, mx.array] = {}

    def recording(
        routed_h, weight, scales, biases, expert_ids, route_scores, *rest
    ):
        seen["routed_h"] = routed_h
        seen["expert_ids"] = expert_ids
        seen["route_scores"] = route_scores
        return _routed_down_reduce(
            routed_h, weight, scales, biases, expert_ids, route_scores, *rest
        )

    reference = _m4_routed_down_reduce_forward(_stub_block(), _hidden(), recording)
    unsorted = dict(seen)

    _arm(monkeypatch, "1")
    candidate = _m4_routed_down_reduce_forward(_stub_block(), _hidden(), recording)

    for key in ("routed_h", "expert_ids", "route_scores"):
        assert bool(mx.array_equal(unsorted[key], seen[key]).item()), key
    assert bool(mx.array_equal(reference, candidate).item())


def test_paired_glu_lane_is_untouched_by_the_gate(monkeypatch):
    """The retained lane gathers inside its kernels; the gate must not move it.

    Pins the documented limitation: both gathers live in Metal, so there is no
    ``mx.gather_qmm`` for the permutation to reorder and the ids handed to the
    kernels stay in routed slot order under either gate setting.
    """

    seen: list[mx.array] = []

    def routed_gu_activation(value, weights, scales, biases, expert_ids):
        assert tuple(value.shape) == (ROWS, HIDDEN)
        assert tuple(expert_ids.shape) == (ROWS, TOP_K)
        seen.append(expert_ids)
        ids = expert_ids.astype(mx.float32)[..., None]
        base = mx.broadcast_to(
            value.astype(mx.float32)[:, None, :INTERMEDIATE],
            (ROWS, TOP_K, INTERMEDIATE),
        )
        return (base * (ids + 1.0)).astype(mx.bfloat16)

    hyper, inject = _hyper_inject()
    reference = _m4_paired_routed_glu_residual_tail_forward(
        _stub_block(),
        _hidden(),
        routed_gu_activation,
        _routed_down_residual_tail,
        hyper,
        inject,
    )

    _arm(monkeypatch, "1")
    candidate = _m4_paired_routed_glu_residual_tail_forward(
        _stub_block(),
        _hidden(),
        routed_gu_activation,
        _routed_down_residual_tail,
        hyper,
        inject,
    )

    assert len(seen) == 2
    assert bool(mx.array_equal(seen[0], seen[1]).item())
    assert bool(mx.array_equal(reference, candidate).item())
