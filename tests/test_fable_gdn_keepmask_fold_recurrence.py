"""W66d: the keep-mask fold's prefix must actually REACH the step kernel.

The first fold-alone ABBA bracket (2026-09-02, receipts
``fable-w66b-gdn-fold-alone-*-1788400662..1788400673``) reported every
engagement counter exactly as the ring policy predicts -- installed, 35 folded
layers, ``windows == compiled_calls``, flushes 0.16-0.21/window, ring depth
never above the max -- and three seeds of DIFFERENT TEXT, diverging by token
~10.

The cause was a lookup.  ``MTPLX_COMPILED_GDN=1`` is a family default, so
``Qwen4ExpTextModel._forward`` routes every ``S <= 4`` decode -- the M4 verify
included -- through ``_decode_layers_compiled``, whose ``_compiled_run_fn``
body hands each GDN layer a THROWAWAY ``ArraysCache``, not the compiled
verify's shadow entry.  The fold's ``id(cache entry)`` lookup missed all 35
foldable layers, and a miss is NOT a decline: ``_fold_state_in`` had already
substituted the ring's base into state slot 1, so each layer ran the stock
recurrence from a state missing one or two committed windows.

Two classes of test here, both CPU-only (no Metal, no model):

1. the scope itself -- that the layer-index key, the alias hook and the
   trace-time ``assert_prefix_consumed`` guard behave, driven through the REAL
   ``Qwen4ExpTextModel._compiled_run_fn`` so a future re-wrap of the cache
   container is caught at this level;
2. the ring protocol -- the multi-window / flush / all-accept / decline /
   bypass sequence replayed with the production ``advance_ring``,
   ``masked_replay_state`` and ``folded_gated_delta_update``, asserted
   BITWISE against the shipped eager replay.  On the CPU stream both sides
   route to ``gated_delta_ops``, so this pins the bookkeeping (which rows,
   which mask, which base, when to flush) rather than the Metal kernel; the
   kernel's split/merge exactness is what
   ``scripts/fable/micro_gdn_keepmask_fold.py`` and the install-time probe
   measure under the flock.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from mtplx import fable_gdn_keepmask_fold as fold
from mtplx.kernels import gdn_keepmask_fold as foldk


@pytest.fixture(autouse=True)
def _cpu_device():
    # set_default_device leaks into every later-collected module (pytest
    # imports share one process), so restore it on the way out.  It is also
    # what keeps this file off the GPU: `gated_delta_update` routes to
    # `gated_delta_ops` whenever the default device is not the GPU, so no
    # Metal kernel is ever built here.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# --------------------------------------------------------------------------
# 1. The prefix scope and the container re-wrap
# --------------------------------------------------------------------------


class _Entry:
    """Stand-in for a cache entry: identity is all the scope uses."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


def _trailing(layers: int, mask: str = "MASK") -> list[str]:
    return [f"r{index}" for index in range(5 * layers)] + [mask]


def test_make_prefix_scope_binds_both_keys():
    entries = {0: _Entry("l0"), 4: _Entry("l4")}
    scope = fold.make_prefix_scope((0, 4), _trailing(2), entries.__getitem__)
    assert scope.by_layer[0] == ("r0", "r1", "r2", "r3", "r4", "MASK")
    assert scope.by_layer[4] == ("r5", "r6", "r7", "r8", "r9", "MASK")
    assert scope.by_entry[id(entries[0])] == scope.by_layer[0]
    assert scope.by_entry[id(entries[4])] == scope.by_layer[4]
    assert scope.consumed == 0


def test_make_prefix_scope_refuses_a_mismatched_arity():
    with pytest.raises(ValueError, match="keep-mask fold leaves"):
        fold.make_prefix_scope((0, 4), _trailing(1), lambda index: _Entry("x"))


def test_make_prefix_scope_is_none_without_folded_layers():
    assert fold.make_prefix_scope((), [], lambda index: None) is None


def test_bind_fold_alias_points_a_rewrapped_container_at_its_layer():
    entries = {0: _Entry("l0"), 1: _Entry("l1")}
    scope = fold.make_prefix_scope((0, 1), _trailing(2), entries.__getitem__)
    rewrapped = _Entry("throwaway")
    with fold.fold_prefix_scope(scope):
        # This is the measured defect: the layer sees a container the scope
        # was never built from.
        assert fold.fold_prefix_for(rewrapped) is None
        fold.bind_fold_alias(1, rewrapped)
        assert fold.fold_prefix_for(rewrapped) == scope.by_layer[1]


def test_bind_fold_alias_removes_a_recycled_id_for_an_unfolded_layer():
    entries = {0: _Entry("l0")}
    scope = fold.make_prefix_scope((0,), _trailing(1), entries.__getitem__)
    container = _Entry("throwaway")
    with fold.fold_prefix_scope(scope):
        fold.bind_fold_alias(0, container)
        assert fold.fold_prefix_for(container) == scope.by_layer[0]
        # CPython recycles id()s; a container that lands on a freed address
        # must not inherit the previous layer's rows.
        fold.bind_fold_alias(7, container)
        assert fold.fold_prefix_for(container) is None


def test_bind_fold_alias_is_a_noop_outside_a_traced_forward():
    fold.bind_fold_alias(0, _Entry("l0"))  # must not raise


def test_assert_prefix_consumed_raises_when_a_layer_missed_its_prefix():
    entries = {0: _Entry("l0"), 1: _Entry("l1")}
    scope = fold.make_prefix_scope((0, 1), _trailing(2), entries.__getitem__)
    with fold.fold_prefix_scope(scope):
        fold.fold_prefix_for(entries[0])
    with pytest.raises(
        fold.GdnKeepMaskFoldContractError, match="1 of 2 folded GDN layers"
    ):
        fold.assert_prefix_consumed(scope, label="unit")


def test_assert_prefix_consumed_passes_when_every_layer_took_its_prefix():
    entries = {0: _Entry("l0"), 1: _Entry("l1")}
    scope = fold.make_prefix_scope((0, 1), _trailing(2), entries.__getitem__)
    with fold.fold_prefix_scope(scope):
        fold.fold_prefix_for(entries[0])
        fold.fold_prefix_for(entries[1])
    fold.assert_prefix_consumed(scope, label="unit")


def test_assert_prefix_consumed_ignores_an_absent_scope():
    fold.assert_prefix_consumed(None, label="unit")


# --------------------------------------------------------------------------
# 1b. The real call site: MTPLX_COMPILED_GDN's per-layer container
# --------------------------------------------------------------------------


def _drive_compiled_run(monkeypatch, *, armed: bool):
    """Run ``Qwen4ExpTextModel._compiled_run_fn`` over three fake GDN layers.

    Returns ``(scope, seen)`` where ``seen[layer_index]`` is the prefix that
    layer's ``__call__`` could actually look up from the container it was
    handed -- i.e. exactly what ``GatedDeltaNet.__call__`` does.
    """

    from mtplx.models import qwen4_exp as qm

    seen: dict[int, object] = {}

    class _Layer:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, h, *, input_ids, ssm_mask, cache):
            seen[self.index] = fold.fold_prefix_for(cache)
            cache[0] = cache[0] + 1
            cache[1] = cache[1] + 1
            return h + 1

    class _Model:
        layers = [_Layer(0), _Layer(1), _Layer(2)]

    monkeypatch.setattr(qm, "_GDN_KEEPMASK_FOLD_ARMED", armed)
    run = qm.Qwen4ExpTextModel._compiled_run_fn(_Model(), (0, 1, 2), capture=False)

    shadow = {index: _Entry(f"shadow{index}") for index in range(3)}
    scope = fold.make_prefix_scope((0, 2), _trailing(2), shadow.__getitem__)
    flat = [mx.zeros((1, 2)) for _ in range(6)]
    with fold.fold_prefix_scope(scope):
        out = run(mx.zeros((1, 2)), *flat)
        mx.eval(out)
    return scope, seen


def test_compiled_gdn_run_reaches_the_fold_prefix_through_the_rewrap(monkeypatch):
    scope, seen = _drive_compiled_run(monkeypatch, armed=True)
    assert seen[0] == scope.by_layer[0]
    assert seen[2] == scope.by_layer[2]
    # Layer 1 is not in the fold plan (the production analogue is the
    # PLE-carrying GDN layer), so it takes the stock recurrence.
    assert seen[1] is None
    assert scope.consumed == 2
    fold.assert_prefix_consumed(scope, label="unit")


def test_compiled_gdn_run_without_the_alias_is_the_measured_defect(monkeypatch):
    """Reproduces W66b's silent divergence, and proves the guard catches it."""

    scope, seen = _drive_compiled_run(monkeypatch, armed=False)
    assert seen == {0: None, 1: None, 2: None}
    assert scope.consumed == 0
    with pytest.raises(
        fold.GdnKeepMaskFoldContractError, match="0 of 2 folded GDN layers"
    ):
        fold.assert_prefix_consumed(scope, label="unit")


# --------------------------------------------------------------------------
# 2. The ring protocol, bitwise against the shipped eager replay
# --------------------------------------------------------------------------

#: Production topology at a size the CPU ops reference can carry.  Only the
#: pad builders read these; the recurrence itself is shape-generic.
K_HEADS = 2
V_HEADS = 4
DIM = 8
MAX_WINDOWS = 2


@pytest.fixture
def small_geometry(monkeypatch):
    monkeypatch.setattr(foldk, "NUM_K_HEADS", K_HEADS)
    monkeypatch.setattr(foldk, "NUM_V_HEADS", V_HEADS)
    monkeypatch.setattr(foldk, "HEAD_DIM", DIM)
    foldk.reset_prefix_caches()
    try:
        yield
    finally:
        foldk.reset_prefix_caches()


def _window_rows(key):
    keys = mx.random.split(key, 5)
    return (
        mx.random.normal((1, fold.VERIFY_WIDTH, K_HEADS, DIM), key=keys[0]),
        mx.random.normal((1, fold.VERIFY_WIDTH, K_HEADS, DIM), key=keys[1]),
        mx.random.normal((1, fold.VERIFY_WIDTH, V_HEADS, DIM), key=keys[2]),
        mx.random.normal((1, fold.VERIFY_WIDTH, V_HEADS), key=keys[3]),
        mx.random.normal((1, fold.VERIFY_WIDTH, V_HEADS), key=keys[4]),
    )


def _eager_replay(rows, keep, A_log, dt_bias, state):
    """``commit_verified_window``'s shipped GDN branch, verbatim."""

    from mlx_lm.models.gated_delta import gated_delta_update

    _y, out = gated_delta_update(
        *(tensor[:, :keep] for tensor in rows),
        A_log,
        dt_bias,
        state,
        None,
        use_kernel=True,
    )
    return out


def _stock_window(rows, A_log, dt_bias, state):
    """The compiled verify's own four-row step, from the committed state."""

    from mlx_lm.models.gated_delta import gated_delta_update

    return gated_delta_update(
        *rows, A_log, dt_bias, state, None, use_kernel=True
    )


#: ``(kind, keep)`` per cycle.  ``fold`` is a partial accept whose verify was
#: this window's compiled graph; ``accept`` is the all-accept branch that
#: returns before ``commit_verified_window``; ``decline`` is a foldable window
#: whose commit did not match the lane (the measured ``verify_width_9`` rows);
#: ``bypass`` is a commit from a non-M4 round.  The sequence deliberately
#: reaches ring depth 2 and flushes twice.
_SEQUENCE = [
    ("fold", 2),      # depth 0 -> 1
    ("fold", 1),      # depth 1 -> 2
    ("fold", 3),      # depth 2 -> FLUSH -> 1
    ("fold", 2),      # depth 1 -> 2
    ("accept", 4),    # all-accept: ring resets, graph state authoritative
    ("fold", 3),      # depth 0 -> 1
    ("decline", 1),   # today's replay, ring resets
    ("fold", 1),      # depth 0 -> 1
    ("bypass", 2),    # today's replay from a non-M4 round, ring resets
    ("fold", 2),      # depth 0 -> 1
    ("fold", 3),      # depth 1 -> 2
    ("fold", 1),      # depth 2 -> FLUSH -> 1
    ("fold", 2),      # depth 1 -> 2
]


def test_fold_ring_is_bit_exact_across_windows_flushes_and_bypasses(
    small_geometry,
):
    key = mx.random.key(20260902)
    parts = mx.random.split(key, 3)
    A_log = mx.random.normal((V_HEADS,), key=parts[0]).astype(mx.float32)
    dt_bias = mx.random.normal((V_HEADS,), key=parts[1]).astype(mx.float32)
    state0 = mx.random.normal(
        (1, V_HEADS, DIM, DIM), key=parts[2]
    ).astype(mx.float32)
    mx.eval(A_log, dt_bias, state0)

    windows = []
    row_key = mx.random.key(4242)
    for _ in _SEQUENCE:
        row_key, sub = mx.random.split(row_key)
        rows = _window_rows(sub)
        mx.eval(*rows)
        windows.append(rows)

    control = state0
    base, ring, keeps = state0, [], ()
    depths_seen: set[int] = set()
    flushes = 0

    for (kind, keep), rows in zip(_SEQUENCE, windows):
        # What `entry.cache[1]` holds at window entry under the fold: the
        # lazy masked replay of the ring over the base, or the base itself.
        pre_state = (
            foldk.masked_replay_state(ring, keeps, A_log, dt_bias, base)
            if ring
            else base
        )
        mx.eval(pre_state)
        assert mx.array_equal(pre_state, control), (
            f"{kind}/keep={keep}: committed state drifted from the shipped path"
        )

        # The compiled window: stock runs four rows from the committed state,
        # the fold runs the padded ring plus those four rows from the base.
        stock_y, stock_state = _stock_window(rows, A_log, dt_bias, control)
        fold_y, fold_state = foldk.folded_gated_delta_update(
            ring, keeps, *rows, A_log, dt_bias, base, max_windows=MAX_WINDOWS
        )
        mx.eval(stock_y, stock_state, fold_y, fold_state)
        assert mx.array_equal(stock_y, fold_y), f"{kind}: window y diverged"
        assert mx.array_equal(
            stock_state, fold_state
        ), f"{kind}: window state output diverged"

        depths_seen.add(len(keeps))

        if kind == "accept":
            control = stock_state
            base, ring, keeps = fold_state, [], ()
            continue

        control = _eager_replay(rows, keep, A_log, dt_bias, control)
        mx.eval(control)

        if kind == "fold":
            pending = fold.FoldPending(
                base=base, rows=list(ring), keeps=keeps, state=pre_state
            )
            base, ring, keeps, flushed = fold.advance_ring(
                pending, rows, keep, max_windows=MAX_WINDOWS
            )
            flushes += int(flushed)
        else:
            # A decline or a bypass takes today's exact replay and leaves no
            # descriptor, so the next window enters at ring depth 0.
            base, ring, keeps = control, [], ()

    final = (
        foldk.masked_replay_state(ring, keeps, A_log, dt_bias, base)
        if ring
        else base
    )
    mx.eval(final)
    assert mx.array_equal(final, control)
    # The sequence must actually have exercised the deep ring and the flush,
    # or this test would pass on a lane that never folds anything.
    assert depths_seen == {0, 1, 2}
    assert flushes == 2


def test_prefix_mask_and_padded_rows_agree_on_the_ring_order(small_geometry):
    """The pad is at the FRONT and the mask must line up with it."""

    key = mx.random.key(7)
    key, a = mx.random.split(key)
    rows_a = _window_rows(a)
    key, b = mx.random.split(key)
    rows_b = _window_rows(b)
    mx.eval(*rows_a, *rows_b)

    padded = foldk.padded_prefix_leaves(
        [rows_a, rows_b], (2, 3), max_windows=MAX_WINDOWS, dtype=mx.float32
    )
    assert [tuple(t.shape) for t in padded[:3]] == [
        (1, 8, K_HEADS, DIM),
        (1, 8, K_HEADS, DIM),
        (1, 8, V_HEADS, DIM),
    ]
    assert fold.prefix_mask_rows((2, 3), max_windows=MAX_WINDOWS) == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    # A one-window ring pads the FRONT, so the live rows sit adjacent to the
    # new window's four.
    assert fold.prefix_mask_rows((2,), max_windows=MAX_WINDOWS) == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_nested_mx_compile_is_inlined_into_the_outer_trace():
    """The premise the alias hook rests on.

    ``_compiled_run_fn`` returns an ``mx.compile``d step, and the fixed-M4
    verify calls it from inside its OWN ``mx.compile`` trace.  The alias is
    bound in the step's Python body, so it only reaches the graph if MLX
    inlines a nested compile into the outer trace instead of replaying a
    separately cached inner graph.  Pin that, because the fold's correctness
    now depends on it.
    """

    flag = {"on": False}
    executions: list[bool] = []

    def inner(a, b):
        executions.append(flag["on"])
        return a * b if flag["on"] else a + b

    compiled_inner = mx.compile(inner)

    def outer_a(a, b):
        return compiled_inner(a, b) + 0

    def outer_b(a, b):
        return compiled_inner(a, b) + 0

    x, y = mx.array([3.0]), mx.array([4.0])
    first = mx.compile(outer_a)(x, y)
    mx.eval(first)
    flag["on"] = True
    second = mx.compile(outer_b)(x, y)
    mx.eval(second)

    assert executions == [False, True]
    assert first.item() == 7.0
    assert second.item() == 12.0
