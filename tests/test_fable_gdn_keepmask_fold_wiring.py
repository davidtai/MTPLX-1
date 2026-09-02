"""Pure-Python tests for the WIRED keep-mask fold (W66b).

``tests/test_fable_gdn_keepmask_fold.py`` pins the policy in isolation; this
file pins the protocol the three wiring sites run between them:

    compiled window  ->  `_fold_window_prefix`   (base + fixed-shape prefix)
    commit           ->  `_gdn_keepmask_fold_plan` + `advance_ring`
    every other site ->  the lazy leaf on `cache[1]`

Nothing here imports ``mlx`` for arithmetic and nothing evaluates an array.
The recurrence is modelled symbolically -- a state is the ORDERED TUPLE of
rows applied to it -- which is exactly the property the fold's exactness rests
on: the ``t`` loop may be split or merged anywhere, and a masked row is an
exact no-op.  Under that model the folded protocol either produces the same
row sequence as the shipped eager replay or it does not, at every accept
pattern, which is a stronger statement than any counter check.  The float
arithmetic that model abstracts away is pinned on a GPU by
``scripts/fable/micro_gdn_keepmask_fold.py``'s parity arm and by
``mtplx.kernels.gdn_keepmask_fold.default_exactness_probe`` at install.
"""

from __future__ import annotations

import pytest

from mtplx import fable_gdn_keepmask_fold as fold


# --------------------------------------------------------------------------
# Symbolic recurrence
# --------------------------------------------------------------------------


class _State:
    """A recurrence state: the ORDERED tuple of rows applied to it.

    A distinct object per construction so ``is`` means what it means on an
    ``mx.array`` leaf, with ``==`` comparing the row sequence.  ``base`` and
    ``evaluated`` model MLX's laziness: an unevaluated masked replay keeps its
    input alive, and ``mx.async_eval`` on any node cuts the chain there.
    ``lazy_depth`` is how many unevaluated replays stand between this value
    and a materialised one -- the fold's memory bound, and the reason a flush
    is safe.
    """

    __slots__ = ("rows", "base", "evaluated")

    def __init__(self, rows, base=None, evaluated=False):
        self.rows = tuple(rows)
        self.base = base
        self.evaluated = bool(evaluated)

    def mark_evaluated(self):
        """What `mx.async_eval` on this leaf does to the chain behind it."""

        self.evaluated = True
        return self

    def lazy_depth(self):
        if self.evaluated or self.base is None:
            return 0
        return 1 + self.base.lazy_depth()

    def __eq__(self, other):
        return isinstance(other, _State) and self.rows == other.rows

    def __hash__(self):
        return hash(self.rows)

    def __repr__(self):  # pragma: no cover - debugging only
        return f"_State({self.rows!r}, lazy={self.lazy_depth()})"


def _apply(state, rows, mask, *, lazy=False):
    """``step(state, rows, mask)`` as the ordered sequence of LIVE rows.

    A masked step takes the stock kernel's ``else`` branch: it writes ``y = 0``
    and never touches the register state.  So the state after a masked pass is
    the state before it plus exactly the unmasked rows, in order -- which is
    all this model needs to be faithful.
    """

    live = tuple(row for row, keep in zip(rows, mask) if keep)
    return _State(
        state.rows + live, base=state if lazy else None, evaluated=not lazy
    )


def _window_rows(window_id):
    return tuple((window_id, index) for index in range(fold.VERIFY_WIDTH))


class _Entry:
    """The two GDN cache leaves the fold touches, and nothing else."""

    def __init__(self, state):
        self.cache = [None, state]


class _Sim:
    """One layer running the full verify -> accept -> commit protocol.

    Drives the SHIPPED functions (``pending_for``, ``set_active``,
    ``active_for``, ``advance_ring``, ``prefix_mask_rows``, ``clear_pending``)
    rather than a transcription of them, so a change to the policy fails here.
    """

    def __init__(self, *, max_windows):
        self.max_windows = int(max_windows)
        self.entry = _Entry(_State(("S0",)))
        self.eager = _State(("S0",))       # the shipped replay's answer
        self.next_window = 0
        self.ring_depths: list[int] = []
        self.flushes = 0
        self.max_lazy_depth = 0

    # -- the compiled window ------------------------------------------------

    def verify(self):
        """What ``_forward_installed_fixed_m4`` does, in this model."""

        window_id = self.next_window
        self.next_window += 1
        rows = _window_rows(window_id)

        pending = fold.pending_for(self.entry)
        if pending is None:
            leaf = self.entry.cache[1]
            pending = fold.FoldPending(
                base=leaf, rows=[], keeps=(), state=leaf
            )
        seq = fold.next_window_seq()
        fold.set_active(self.entry, pending, seq)
        self.ring_depths.append(len(pending.keeps))

        # The pre-boundary `mx.async_eval(*state_in)` materialises the base.
        self.max_lazy_depth = max(
            self.max_lazy_depth, pending.base.lazy_depth()
        )
        pending.base.mark_evaluated()

        # The graph starts from the BASE and runs the padded ring then the
        # window's four rows.  Pad slots are masked off and are no-ops.
        mask = fold.prefix_mask_rows(pending.keeps, max_windows=self.max_windows)
        flat: list[object] = []
        for _ in range(self.max_windows - len(pending.keeps)):
            flat.extend(("pad", index) for index in range(fold.VERIFY_WIDTH))
        for group in pending.rows:
            flat.extend(group)
        assert len(flat) == len(mask)
        state_out = _apply(pending.base, flat, mask)
        state_out = _apply(state_out, rows, [True] * fold.VERIFY_WIDTH)

        self.entry.cache[1] = state_out
        fold.clear_pending(self.entry)
        return window_id, rows, seq

    # -- the commit ---------------------------------------------------------

    def cycle(self, accepted_count):
        """One decode cycle.  ``accepted_count`` is 0..3 (3 = all-accept)."""

        _window_id, rows, seq = self.verify()
        keep = 1 + int(accepted_count)
        self.eager = _apply(self.eager, rows, [True] * keep)
        if keep == fold.VERIFY_WIDTH:
            # generation.py returns before commit_verified_window: the graph's
            # own state output IS the committed state and the ring resets for
            # free.
            assert self.entry.cache[1] == self.eager
            return
        active = fold.active_for(self.entry, seq)
        assert active is not None
        base, ring_rows, ring_keeps, flushed = fold.advance_ring(
            active, rows, keep, max_windows=self.max_windows
        )
        replay_mask = [
            index < k for k in ring_keeps for index in range(fold.VERIFY_WIDTH)
        ]
        flat = [row for group in ring_rows for row in group]
        state = _apply(base, flat, replay_mask, lazy=True)
        self.max_lazy_depth = max(self.max_lazy_depth, state.lazy_depth())
        self.entry.cache[1] = state
        self.entry._mtplx_fold_pending = fold.FoldPending(
            base=base, rows=ring_rows, keeps=ring_keeps, state=state
        )
        fold.clear_active(self.entry)
        self.flushes += int(flushed)

    def copy_round(self, *, block=2, keep=1):
        """A context-copy block round: an EAGER forward on the live cache.

        It reads ``cache[1]`` -- forcing the deferred leaf, which is the flush
        -- runs its own rows, and its commit finds no window stamp, so it takes
        the shipped replay and rebinds the leaf.  The descriptor is dropped the
        moment something else owns the leaf.
        """

        forced = self.entry.cache[1]
        assert forced == self.eager
        forced.mark_evaluated()
        rows = tuple(("copy", self.next_window, index) for index in range(block))
        self.next_window += 1
        assert fold.active_for(self.entry, fold.current_window_seq()) is None
        self.eager = _apply(self.eager, rows, [True] * keep)
        self.entry.cache[1] = _apply(forced, rows, [True] * keep)
        assert fold.pending_for(self.entry) is None
        assert self.entry.cache[1] == self.eager

@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(fold.ENV_FLAG, raising=False)
    monkeypatch.delenv(fold.ENV_WINDOWS, raising=False)
    monkeypatch.setenv(fold.ENV_LOG, "0")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.reset_stats()
    yield
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.reset_stats()


# --------------------------------------------------------------------------
# Flush-order equivalence: the folded state IS the eager replay's state
# --------------------------------------------------------------------------


@pytest.mark.parametrize("max_windows", fold.MAX_WINDOWS_CHOICES)
@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param([0], id="single-reject"),
        pytest.param([3], id="single-all-accept"),
        pytest.param([0, 0, 0, 0, 0, 0], id="never-accepts"),
        pytest.param([3, 3, 3, 3], id="always-accepts"),
        pytest.param([0, 1, 2, 0, 1, 2, 0, 1, 2], id="k0-k1-k2-cycle"),
        pytest.param([2, 2, 3, 0, 0, 3, 1, 1, 1, 3, 2], id="mixed"),
        pytest.param([1, 3, 1, 3, 1, 3], id="alternating-bonus"),
    ],
)
def test_folded_state_matches_the_eager_replay_at_every_step(
    pattern, max_windows
):
    sim = _Sim(max_windows=max_windows)
    for accepted in pattern:
        sim.cycle(accepted)
        assert sim.entry.cache[1] == sim.eager


@pytest.mark.parametrize("max_windows", fold.MAX_WINDOWS_CHOICES)
def test_a_copy_round_between_windows_is_exact(max_windows):
    """A block round forces the leaf, gets today's answer, and resets the ring."""

    sim = _Sim(max_windows=max_windows)
    for index, accepted in enumerate([0, 1, 2, 0, 1, 2]):
        sim.cycle(accepted)
        if index % 2 == 1:
            sim.copy_round()
        assert sim.entry.cache[1] == sim.eager


@pytest.mark.parametrize("max_windows", fold.MAX_WINDOWS_CHOICES)
def test_an_all_accept_resets_the_ring_for_free(max_windows):
    sim = _Sim(max_windows=max_windows)
    sim.cycle(0)
    sim.cycle(1)
    assert fold.pending_for(sim.entry) is not None
    sim.cycle(3)                                  # all-accept: no commit at all
    assert fold.pending_for(sim.entry) is None
    assert sim.entry.cache[1] == sim.eager
    sim.cycle(0)
    assert sim.ring_depths[-1] == 0
    assert sim.entry.cache[1] == sim.eager


def test_request_end_leaves_a_leaf_anything_can_force():
    """Generation can stop on any cycle; the leaf is always the real state."""

    sim = _Sim(max_windows=2)
    for accepted in [0, 1, 0, 2]:
        sim.cycle(accepted)
        pending = fold.pending_for(sim.entry)
        assert pending is None or pending.state is sim.entry.cache[1]
        assert sim.entry.cache[1] == sim.eager


@pytest.mark.parametrize("max_windows", fold.MAX_WINDOWS_CHOICES)
def test_the_ring_never_exceeds_its_depth_and_flushes_exactly_when_full(
    max_windows,
):
    sim = _Sim(max_windows=max_windows)
    pattern = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    for accepted in pattern:
        sim.cycle(accepted)
    assert max(sim.ring_depths) <= max_windows
    # A commit flushes iff the ring it entered was already full; with no
    # all-accepts in the pattern the two counts are the same thing.
    assert sim.flushes == sum(
        1 for depth in sim.ring_depths if depth == max_windows
    )


@pytest.mark.parametrize("max_windows", fold.MAX_WINDOWS_CHOICES)
def test_the_lazy_chain_never_stacks_more_than_two_unevaluated_replays(
    max_windows,
):
    """A flush's new base is the PREVIOUS pending leaf -- unevaluated.

    It becomes the next window's ``state_in``, and that window's pre-boundary
    ``mx.async_eval`` materialises it, so the chain is cut before a third
    level can form.  Without that the fold would build an ever-deeper lazy
    graph across a 1,024-token generation and pay for all of it at once.
    """

    sim = _Sim(max_windows=max_windows)
    for accepted in [0, 1, 2, 0, 0, 1, 3, 0, 2, 1, 0, 0, 2, 1, 0]:
        sim.cycle(accepted)
    assert sim.max_lazy_depth <= 2


# --------------------------------------------------------------------------
# Pending / active descriptor protocol
# --------------------------------------------------------------------------


def test_active_is_only_honoured_for_its_own_window():
    entry = _Entry(("S0",))
    pending = fold.FoldPending(base=("S0",), state=("S0",))
    seq = fold.next_window_seq()
    fold.set_active(entry, pending, seq)
    assert fold.active_for(entry, seq) is pending
    later = fold.next_window_seq()
    assert fold.active_for(entry, later) is None


def test_active_is_absent_before_any_window():
    assert fold.active_for(_Entry(("S0",)), 1) is None


def test_clear_active_is_idempotent():
    entry = _Entry(("S0",))
    fold.clear_active(entry)
    fold.set_active(entry, fold.FoldPending(base=("S0",)), 7)
    fold.clear_active(entry)
    assert fold.active_for(entry, 7) is None
    fold.clear_active(entry)


def test_advance_ring_refuses_a_flush_with_no_materialisable_state():
    pending = fold.FoldPending(base=("S0",), rows=[()], keeps=(1,), state=None)
    with pytest.raises(fold.GdnKeepMaskFoldContractError):
        fold.advance_ring(pending, (), 2, max_windows=1)


def test_advance_ring_extends_then_flushes():
    base = ("S0",)
    pending = fold.FoldPending(base=base, rows=[], keeps=(), state=base)
    new_base, rows, keeps, flushed = fold.advance_ring(
        pending, "w0", 2, max_windows=2
    )
    assert (new_base, rows, keeps, flushed) == (base, ["w0"], (2,), False)
    pending = fold.FoldPending(base=new_base, rows=rows, keeps=keeps, state="L1")
    new_base, rows, keeps, flushed = fold.advance_ring(
        pending, "w1", 1, max_windows=2
    )
    assert (new_base, rows, keeps, flushed) == (base, ["w0", "w1"], (2, 1), False)
    pending = fold.FoldPending(base=new_base, rows=rows, keeps=keeps, state="L2")
    new_base, rows, keeps, flushed = fold.advance_ring(
        pending, "w2", 3, max_windows=2
    )
    assert (new_base, rows, keeps, flushed) == ("L2", ["w2"], (3,), True)


# --------------------------------------------------------------------------
# Prefix scope
# --------------------------------------------------------------------------


def test_prefix_scope_is_empty_outside_a_traced_forward():
    assert fold.fold_prefix_for(_Entry(("S0",))) is None


def test_prefix_scope_is_keyed_by_entry_identity():
    folded, unfolded = _Entry(("S0",)), _Entry(("S0",))
    leaves = ("q", "k", "v", "a", "b", "mask")
    with fold.fold_prefix_scope({id(folded): leaves}):
        assert fold.fold_prefix_for(folded) == leaves
        # The PLE-carrying GDN layer is never in the map, so it takes the
        # stock `gated_delta_update` it takes today.
        assert fold.fold_prefix_for(unfolded) is None
    assert fold.fold_prefix_for(folded) is None


def test_prefix_scope_restores_an_outer_scope():
    entry = _Entry(("S0",))
    with fold.fold_prefix_scope({id(entry): "outer"}):
        with fold.fold_prefix_scope(None):
            assert fold.fold_prefix_for(entry) is None
        assert fold.fold_prefix_for(entry) == "outer"


# --------------------------------------------------------------------------
# Graph arity: one fixed shape, whatever the ring holds
# --------------------------------------------------------------------------


def test_prefix_leaf_count_is_five_rows_a_layer_plus_one_shared_mask():
    assert fold.prefix_leaf_count(0) == 0
    assert fold.prefix_leaf_count(1) == 6
    assert fold.prefix_leaf_count(fold.FOLDABLE_LAYERS) == 176


def test_prefix_leaf_count_rejects_a_negative_layer_count():
    with pytest.raises(ValueError):
        fold.prefix_leaf_count(-1)


@pytest.mark.parametrize("max_windows", fold.MAX_WINDOWS_CHOICES)
@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_the_padded_mask_is_one_fixed_width_at_every_ring_depth(
    max_windows, depth
):
    if depth > max_windows:
        pytest.skip("deeper than the ring")
    keeps = tuple(1 + (index % 3) for index in range(depth))
    mask = fold.prefix_mask_rows(keeps, max_windows=max_windows)
    assert len(mask) == fold.VERIFY_WIDTH * max_windows
    assert sum(mask) == sum(keeps)


# --------------------------------------------------------------------------
# Decline / bypass accounting
# --------------------------------------------------------------------------


def test_note_decline_counts_by_reason():
    fold.note_decline("ring_depth_disagreement")
    fold.note_decline("ring_depth_disagreement")
    fold.note_decline("keep_tokens_4")
    snapshot = fold.stats_snapshot()
    assert snapshot["declines"] == 3
    assert snapshot["decline_reasons"] == {
        "ring_depth_disagreement": 2,
        "keep_tokens_4": 1,
    }


def test_stats_snapshot_copies_the_decline_map():
    fold.note_decline("x")
    snapshot = fold.stats_snapshot()
    snapshot["decline_reasons"]["x"] = 99
    assert fold.STATS["decline_reasons"]["x"] == 1


def test_reset_stats_rewinds_the_window_epoch():
    fold.next_window_seq()
    fold.next_window_seq()
    assert fold.current_window_seq() == 2
    fold.reset_stats()
    assert fold.current_window_seq() == 0


# --------------------------------------------------------------------------
# Receipt gate
# --------------------------------------------------------------------------


def _engaged_snapshot(windows=1000, **overrides):
    p = fold.CENSUS_P_ALL_ACCEPT
    flushes = round(
        fold.expected_state_passes_per_cycle(p, max_windows=2) * windows
    )
    snapshot = {
        "installed": True,
        "install_status": "installed",
        "install_error": None,
        "folded_layers": fold.FOLDABLE_LAYERS,
        "max_windows": 2,
        "windows": windows,
        "folded_windows": round((1 - p) * windows),
        "deferred_commits": round((1 - p) * windows),
        "flushes": flushes,
        "declines": 0,
        "decline_reasons": {},
        "ring_depth_hist": {"0": 300, "1": 500, "2": 200},
    }
    snapshot.update(overrides)
    return snapshot


def test_receipt_gate_passes_an_engaged_arm():
    report = fold.receipt_gate(_engaged_snapshot(), compiled_windows=1000)
    assert report["ok"] is True, report["checks"]
    assert report["observed_flushes_per_cycle"] == pytest.approx(0.206, abs=5e-3)


def test_receipt_gate_fails_a_disabled_lane():
    snapshot = _engaged_snapshot(
        installed=False, install_status="exactness_failed", install_error="boom"
    )
    report = fold.receipt_gate(snapshot, compiled_windows=1000)
    assert report["ok"] is False
    failed = {item["check"] for item in report["checks"] if not item["ok"]}
    assert {"installed", "no_install_error"} <= failed


def test_receipt_gate_fails_on_any_decline():
    snapshot = _engaged_snapshot(declines=1, decline_reasons={"keep_tokens_4": 1})
    report = fold.receipt_gate(snapshot, compiled_windows=1000)
    assert report["ok"] is False
    assert any(
        item["check"] == "fold_declined_zero" and not item["ok"]
        for item in report["checks"]
    )


def test_receipt_gate_fails_when_windows_miss_compiled_calls():
    """A window on the shipped route dilutes the delta by exactly its share."""

    report = fold.receipt_gate(_engaged_snapshot(windows=900), compiled_windows=1000)
    assert report["ok"] is False
    assert any(
        item["check"] == "windows_cover_compiled_calls" and not item["ok"]
        for item in report["checks"]
    )


def test_receipt_gate_fails_when_the_ring_flushes_every_cycle():
    """Something forcing the deferred leaf shows up here and nowhere else."""

    snapshot = _engaged_snapshot(flushes=705)
    report = fold.receipt_gate(snapshot, compiled_windows=1000)
    assert report["ok"] is False
    assert any(
        item["check"] == "flushes_per_cycle" and not item["ok"]
        for item in report["checks"]
    )


def test_receipt_gate_fails_a_ring_deeper_than_its_depth():
    snapshot = _engaged_snapshot(ring_depth_hist={"0": 1, "3": 1})
    report = fold.receipt_gate(snapshot, compiled_windows=1000)
    assert report["ok"] is False
    assert any(
        item["check"] == "ring_depth_within_max" and not item["ok"]
        for item in report["checks"]
    )


def test_receipt_gate_expectation_matches_the_ring_policy():
    for windows in fold.MAX_WINDOWS_CHOICES:
        snapshot = _engaged_snapshot(max_windows=windows)
        report = fold.receipt_gate(snapshot, compiled_windows=1000)
        assert report["expected_flushes_per_cycle"] == pytest.approx(
            fold.expected_state_passes_per_cycle(
                fold.CENSUS_P_ALL_ACCEPT, max_windows=windows
            )
        )


# --------------------------------------------------------------------------
# Graphbank side: install resolution and the per-window prefix
# --------------------------------------------------------------------------
#
# `mlx` is imported (graphbank imports it) but no array is created and nothing
# is evaluated: the kernel helpers that would build tensors are stubbed, and
# the geometry gate is fed stub modules exactly as
# tests/test_fable_gdn_keepmask_fold.py feeds it.

GDN_INDICES = tuple(index for index in range(48) if index % 4 != 3)
PLE_INDEX = GDN_INDICES[2]


class _StubState:
    shape = (1, fold.NUM_V_HEADS, fold.HEAD_DIM, fold.HEAD_DIM)
    dtype = "mlx.core.float32"


class _StubWeight:
    dtype = "bfloat16-stub"


class _StubConv:
    weight = _StubWeight()


class _StubGdn:
    def __init__(self):
        self.num_v_heads = fold.NUM_V_HEADS
        self.num_k_heads = fold.NUM_K_HEADS
        self.head_v_dim = fold.HEAD_DIM
        self.head_k_dim = fold.HEAD_DIM
        self.training = False
        self.conv1d = _StubConv()


class _StubLayer:
    def __init__(self, linear_attn=None):
        self.linear_attn = linear_attn


def _stub_layers():
    return [
        _StubLayer(_StubGdn() if index in GDN_INDICES else None)
        for index in range(48)
    ]


def _capture_layout():
    return tuple(
        (
            index,
            ("qkv", "q", "k", "v", "a", "b")
            + (
                ("ple_hidden", "ple_ids", "ple_conv_rows")
                if index == PLE_INDEX
                else ()
            ),
        )
        for index in GDN_INDICES
    )


def _bank(monkeypatch, *, layout=None):
    from mtplx import graphbank as module
    from mtplx.kernels import gdn_keepmask_fold as kernels

    bank = module.CompiledVerifyBank.__new__(module.CompiledVerifyBank)
    bank._extra_capture_layout = _capture_layout() if layout is None else layout
    bank._fold_layer_indices = ()
    bank._fold_entries = ()
    bank._fold_windows = 0
    bank._fold_dtype = None
    layers = _stub_layers()
    monkeypatch.setattr(
        module.CompiledVerifyBank,
        "_fold_text_layers",
        lambda self: tuple(layers),
        raising=True,
    )
    monkeypatch.setattr(
        kernels, "default_exactness_probe", lambda **_: (True, "stubbed")
    )
    cache = [_Entry(_StubState()) for _ in range(48)]
    return bank, cache


def test_install_resolution_is_inert_when_the_flag_is_off(monkeypatch):
    bank, cache = _bank(monkeypatch)
    bank._resolve_gdn_keepmask_fold(cache)
    assert bank._fold_layer_indices == ()
    assert bank._fold_entries == ()
    assert bank._fold_windows == 0


def test_install_resolution_excludes_the_ple_layer(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    bank, cache = _bank(monkeypatch)
    bank._resolve_gdn_keepmask_fold(cache)
    assert len(bank._fold_layer_indices) == fold.FOLDABLE_LAYERS == 35
    assert PLE_INDEX not in bank._fold_layer_indices
    assert set(bank._fold_layer_indices) < set(GDN_INDICES)
    assert bank._fold_windows == fold.DEFAULT_MAX_WINDOWS
    assert bank._fold_dtype is _StubConv.weight.dtype
    assert fold.STATS["installed"] is True


def test_install_resolution_raises_on_a_bf16_recurrent_state(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    bank, cache = _bank(monkeypatch)

    class _Bf16:
        shape = (1, fold.NUM_V_HEADS, fold.HEAD_DIM, fold.HEAD_DIM)
        dtype = "bfloat16"

    cache[GDN_INDICES[0]].cache[1] = _Bf16()
    with pytest.raises(fold.GdnKeepMaskFoldContractError, match="bit-exact"):
        bank._resolve_gdn_keepmask_fold(cache)


def test_install_resolution_raises_without_exactly_one_ple_layer(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    layout = tuple(
        (index, ("qkv", "q", "k", "v", "a", "b")) for index in GDN_INDICES
    )
    bank, cache = _bank(monkeypatch, layout=layout)
    with pytest.raises(
        fold.GdnKeepMaskFoldContractError, match="exactly one PLE"
    ):
        bank._resolve_gdn_keepmask_fold(cache)


def test_a_disabling_exactness_probe_leaves_the_lane_unarmed(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    bank, cache = _bank(monkeypatch)
    from mtplx.kernels import gdn_keepmask_fold as kernels

    monkeypatch.setattr(
        kernels, "default_exactness_probe", lambda **_: (False, "ulp drift")
    )
    bank._resolve_gdn_keepmask_fold(cache)
    assert bank._fold_layer_indices == ()
    assert fold.STATS["install_status"] == "exactness_failed"


def _stub_prefix_kernels(monkeypatch):
    from mtplx.kernels import gdn_keepmask_fold as kernels

    monkeypatch.setattr(
        kernels,
        "empty_prefix_leaves",
        lambda *, max_windows, dtype: tuple(
            ("pad", name, max_windows) for name in "qkvab"
        ),
    )
    monkeypatch.setattr(
        kernels,
        "padded_prefix_leaves",
        lambda rows, keeps, *, max_windows, dtype: tuple(
            ("rows", name, tuple(keeps)) for name in "qkvab"
        ),
    )
    monkeypatch.setattr(
        kernels,
        "prefix_mask_array",
        lambda keeps, *, max_windows: ("mask", tuple(keeps), max_windows),
    )


def _dispatch(entries, windows=2):
    return {
        "fold_entries": tuple(entries),
        "fold_windows": windows,
        "fold_dtype": "bf16-stub",
    }


def test_window_prefix_is_one_fixed_arity_at_every_ring_depth(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    _stub_prefix_kernels(monkeypatch)
    bank, cache = _bank(monkeypatch)
    bank._resolve_gdn_keepmask_fold(cache)
    entries = bank._fold_entries
    dispatch = _dispatch(entries)

    from mtplx import graphbank as module

    arities = set()
    for keeps in [(), (2,), (2, 1)]:
        for entry in entries:
            if keeps:
                leaf = object()
                entry.cache[1] = leaf
                entry._mtplx_fold_pending = fold.FoldPending(
                    base=object(),
                    rows=[object() for _ in keeps],
                    keeps=keeps,
                    state=leaf,
                )
            else:
                fold.clear_pending(entry)
        leaves, depth, seq = module.CompiledVerifyBank._fold_window_prefix(
            bank, dispatch
        )
        arities.add(len(leaves))
        assert depth == len(keeps)
        assert leaves[-1] == ("mask", keeps, 2)
        assert fold.active_for(entries[0], seq) is not None
    assert arities == {fold.prefix_leaf_count(fold.FOLDABLE_LAYERS)}
    assert fold.STATS["windows"] == 3
    assert fold.STATS["ring_depth_hist"] == {"0": 1, "1": 1, "2": 1}
    assert fold.STATS["folded_windows"] == 2
    assert fold.STATS["declines"] == 0


def test_a_ring_the_layers_disagree_on_declines_to_todays_path(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    _stub_prefix_kernels(monkeypatch)
    bank, cache = _bank(monkeypatch)
    bank._resolve_gdn_keepmask_fold(cache)
    entries = bank._fold_entries
    for index, entry in enumerate(entries):
        leaf = object()
        entry.cache[1] = leaf
        entry._mtplx_fold_pending = fold.FoldPending(
            base=object(),
            rows=[object()] * (1 if index else 2),
            keeps=(2,) if index else (2, 1),
            state=leaf,
        )

    from mtplx import graphbank as module

    leaves, depth, seq = module.CompiledVerifyBank._fold_window_prefix(
        bank, _dispatch(entries)
    )
    assert depth == 0
    assert leaves[-1] == ("mask", (), 2)
    assert fold.STATS["declines"] == 1
    assert fold.STATS["decline_reasons"] == {"ring_depth_disagreement": 1}
    # Declining still stamps every entry, and every base is the entry's own
    # leaf -- the correct state, forced.  That is today's answer at today's
    # cost, with the graph's shape unchanged.
    for entry in entries:
        active = fold.active_for(entry, seq)
        assert active is not None and active.base is entry.cache[1]


# --------------------------------------------------------------------------
# Commit side: which windows may defer
# --------------------------------------------------------------------------


def _commit_plan(cache, plan, *, keep, verified, armed=True, monkeypatch=None):
    from mtplx.models import qwen4_exp

    monkeypatch.setattr(qwen4_exp, "_GDN_KEEPMASK_FOLD_ARMED", armed)
    return qwen4_exp.Qwen4ExpTextModel._gdn_keepmask_fold_plan(
        object(), cache, plan, keep_tokens=keep, verified_tokens=verified
    )


def _stamped_cache(indices, seq, keeps=(2,)):
    cache = {}
    for index in indices:
        entry = _Entry(object())
        fold.set_active(
            entry,
            fold.FoldPending(base=object(), rows=[object()], keeps=keeps),
            seq,
        )
        cache[index] = entry
    return cache


def test_commit_plan_folds_a_stamped_partial_accept(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    seq = fold.next_window_seq()
    cache = _stamped_cache([0, 1, 2], seq)
    plan = [("gdn", index, None) for index in (0, 1, 2)]
    actives, windows = _commit_plan(
        cache, plan, keep=2, verified=4, monkeypatch=monkeypatch
    )
    assert set(actives) == {0, 1, 2}
    assert windows == fold.DEFAULT_MAX_WINDOWS
    assert fold.STATS["declines"] == 0


def test_commit_plan_bypasses_an_unstamped_round(monkeypatch):
    """A copy round's `forward_ar` leaves no stamp: bypass, not decline."""

    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    fold.next_window_seq()
    cache = {index: _Entry(object()) for index in (0, 1, 2)}
    plan = [("gdn", index, None) for index in (0, 1, 2)]
    actives, windows = _commit_plan(
        cache, plan, keep=2, verified=6, monkeypatch=monkeypatch
    )
    assert actives == {} and windows == 0
    assert fold.STATS["declines"] == 0
    assert fold.STATS["bypassed_commits"] == 1


def test_commit_plan_declines_a_stale_stamp_on_some_layers(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    seq = fold.next_window_seq()
    cache = _stamped_cache([0, 1], seq)
    cache[2] = _Entry(object())
    plan = [("gdn", index, None) for index in (0, 1, 2)]
    actives, _windows = _commit_plan(
        cache, plan, keep=2, verified=4, monkeypatch=monkeypatch
    )
    assert actives == {}
    assert fold.STATS["decline_reasons"] == {"partial_window_stamp": 1}


@pytest.mark.parametrize("keep", [0, 4, 5])
def test_commit_plan_declines_an_impossible_keep(monkeypatch, keep):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    seq = fold.next_window_seq()
    cache = _stamped_cache([0, 1], seq)
    plan = [("gdn", index, None) for index in (0, 1)]
    actives, _windows = _commit_plan(
        cache, plan, keep=keep, verified=4, monkeypatch=monkeypatch
    )
    assert actives == {}
    assert fold.STATS["decline_reasons"] == {f"keep_tokens_{keep}": 1}


def test_commit_plan_declines_a_non_four_row_window(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    seq = fold.next_window_seq()
    cache = _stamped_cache([0], seq)
    plan = [("gdn", 0, None)]
    actives, _windows = _commit_plan(
        cache, plan, keep=2, verified=6, monkeypatch=monkeypatch
    )
    assert actives == {}
    assert fold.STATS["decline_reasons"] == {"verify_width_6": 1}


def test_commit_plan_declines_when_layers_carry_different_rings(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    seq = fold.next_window_seq()
    cache = _stamped_cache([0], seq, keeps=(2,))
    cache.update(_stamped_cache([1], seq, keeps=(2, 1)))
    plan = [("gdn", index, None) for index in (0, 1)]
    actives, _windows = _commit_plan(
        cache, plan, keep=1, verified=4, monkeypatch=monkeypatch
    )
    assert actives == {}
    assert fold.STATS["decline_reasons"] == {"ring_depth_disagreement": 1}


def test_commit_plan_is_inert_when_the_lane_is_not_installed(monkeypatch):
    fold.reset_stats()
    seq = fold.next_window_seq()
    cache = _stamped_cache([0], seq)
    plan = [("gdn", 0, None)]
    actives, windows = _commit_plan(
        cache, plan, keep=2, verified=4, armed=False, monkeypatch=monkeypatch
    )
    assert (actives, windows) == ({}, 0)
    assert fold.STATS["declines"] == 0
    assert fold.STATS["bypassed_commits"] == 0


def test_commit_plan_ignores_a_window_with_no_gdn_layers(monkeypatch):
    monkeypatch.setenv(fold.ENV_FLAG, "1")
    fold.reset_fable_gdn_keepmask_fold_cache()
    fold.STATS["installed"] = True
    actives, windows = _commit_plan(
        {}, [("trim", 3, None)], keep=2, verified=4, monkeypatch=monkeypatch
    )
    assert (actives, windows) == ({}, 0)
    assert fold.STATS["bypassed_commits"] == 0
