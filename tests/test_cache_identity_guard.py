"""A compiled run's container re-wrap must not be able to hide a lane.

``Qwen4ExpTextModel._compiled_run_fn`` hands each GDN layer a THROWAWAY
``ArraysCache(size=2)``, not the container its caller owns.  One decode lane
measured what that costs: its ``id(cache entry)`` lookup missed every layer,
the recurrence silently ran from a state the dispatch had already replaced,
and every host-side counter read perfect for hours.

These tests pin the GENERIC guard, driven through the REAL
``_compiled_run_fn``:

1. a synthetic lane that keys per-layer state by the identity of the container
   it is handed resolves correctly through ``resolve_cache_entry``;
2. the same lane, with the alias never bound, raises at trace time naming the
   lane and the layer index -- rather than declining, or degrading silently.

CPU-only: the default device is pinned to the CPU for the whole file, so no
Metal kernel is ever built.
"""

from __future__ import annotations

import pytest

import mlx.core as mx

from mtplx import cache_identity as ci


@pytest.fixture(autouse=True)
def _cpu_device():
    # set_default_device leaks into every later-collected module (pytest
    # imports share one process), so restore it on the way out.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _Entry:
    """Stand-in for a real cache entry: identity is all the lanes use."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


# --------------------------------------------------------------------------
# The alias
# --------------------------------------------------------------------------


def test_resolve_is_the_identity_without_a_rewrap():
    entry = _Entry("real")
    assert ci.resolve_cache_entry(entry) is entry
    assert ci.real_entry_for(entry) is None
    assert ci.rewrapped_layer_index(entry) is None


def test_bind_points_a_container_at_the_real_entry():
    real = [_Entry("l0"), _Entry("l1"), _Entry("l2")]
    container = _Entry("throwaway")
    with ci.rewrap_scope(real):
        assert ci.bind_rewrapped_entry(container, 1) is real[1]
    assert ci.resolve_cache_entry(container) is real[1]
    assert ci.real_entry_for(container) is real[1]
    assert ci.rewrapped_layer_index(container) == 1


def test_bind_outside_a_scope_removes_a_stale_alias():
    """CPython recycles addresses; a reused container must not inherit."""

    real = [_Entry("l0")]
    container = _Entry("throwaway")
    with ci.rewrap_scope(real):
        ci.bind_rewrapped_entry(container, 0)
    assert ci.real_entry_for(container) is real[0]
    # No scope: the container is standing in for nothing.
    assert ci.bind_rewrapped_entry(container, 0) is None
    assert ci.real_entry_for(container) is None
    assert ci.resolve_cache_entry(container) is container


def test_bind_for_a_layer_the_source_does_not_cover_is_a_removal():
    real = [_Entry("l0")]
    container = _Entry("throwaway")
    with ci.rewrap_scope(real):
        ci.bind_rewrapped_entry(container, 0)
        assert ci.bind_rewrapped_entry(container, 7) is None
    assert ci.real_entry_for(container) is None


# --------------------------------------------------------------------------
# The expectations registry
# --------------------------------------------------------------------------


def test_assert_is_a_noop_when_no_lane_registered_anything():
    real = [_Entry("l0")]
    container = _Entry("throwaway")
    with ci.rewrap_scope(real):
        ci.bind_rewrapped_entry(container, 0)
    ci.assert_satisfied((0,), label="unit")  # must not raise
    assert ci.current_expectations() is None


def test_assert_ignores_a_layer_that_was_never_rewrapped():
    """An eager forward's layers belong to the lane's own end-of-forward check."""

    with ci.expectations_scope():
        ci.expect("synthetic", 3)
        ci.assert_satisfied((3,), label="unit")  # nothing was re-wrapped


def test_assert_raises_for_a_rewrapped_layer_no_lane_resolved():
    real = [_Entry("l0"), _Entry("l1")]
    with ci.expectations_scope():
        ci.expect("synthetic", 1, real[1])
        with ci.rewrap_scope(real):
            ci.bind_rewrapped_entry(_Entry("throwaway"), 1)
        with pytest.raises(ci.CacheIdentityContractError) as excinfo:
            ci.assert_satisfied((1,), label="unit")
    assert "synthetic @ layer 1" in str(excinfo.value)


def test_note_resolved_attributes_through_the_alias():
    real = [_Entry("l0"), _Entry("l1")]
    container = _Entry("throwaway")
    with ci.expectations_scope():
        ci.expect("synthetic", 1, real[1])
        with ci.rewrap_scope(real):
            ci.bind_rewrapped_entry(container, 1)
        ci.note_resolved("synthetic", container)
        ci.assert_satisfied((1,), label="unit")


def test_note_resolved_attributes_a_real_entry_that_was_never_rewrapped():
    real = [_Entry("l0"), _Entry("l1")]
    with ci.expectations_scope():
        ci.expect("synthetic", 1, real[1])
        with ci.rewrap_scope(real):
            ci.bind_rewrapped_entry(_Entry("throwaway"), 1)
        ci.note_resolved("synthetic", real[1])
        ci.assert_satisfied((1,), label="unit")


def test_assert_scopes_itself_to_the_run_it_is_given():
    real = [_Entry("l0"), _Entry("l1")]
    with ci.expectations_scope():
        ci.expect("synthetic", 0, real[0])
        ci.expect("synthetic", 1, real[1])
        with ci.rewrap_scope(real):
            ci.bind_rewrapped_entry(_Entry("t0"), 0)
            ci.bind_rewrapped_entry(_Entry("t1"), 1)
        ci.note_resolved_index("synthetic", 0)
        ci.assert_satisfied((0,), label="unit")  # run 0 is clean
        with pytest.raises(ci.CacheIdentityContractError, match="layer 1"):
            ci.assert_satisfied((1,), label="unit")


def test_expectations_scope_does_not_leak():
    with ci.expectations_scope():
        assert ci.current_expectations() is not None
    assert ci.current_expectations() is None


# --------------------------------------------------------------------------
# The real call site: a synthetic lane through _compiled_run_fn
# --------------------------------------------------------------------------

#: A synthetic lane's per-layer state, keyed by the identity of the cache
#: container the layer is handed.
_LANE = "synthetic_identity_lane"


def _drive_compiled_run(monkeypatch, *, resolve: bool, declare: bool = True):
    """Run the REAL ``_compiled_run_fn`` over three fake layers.

    ``resolve`` picks whether the synthetic lane looks its state up through
    ``resolve_cache_entry`` (the guard's contract) or straight off the
    container it was handed (the defect).  Returns
    ``(real, seen, run, flat)``.
    """

    from mtplx.models import qwen4_exp as qm

    real = [_Entry(f"real{index}") for index in range(3)]
    state = {id(real[0]): "rows0", id(real[2]): "rows2"}
    seen: dict[int, object] = {}

    class _Layer:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, h, *, input_ids, ssm_mask, cache):
            key = ci.resolve_cache_entry(cache) if resolve else cache
            rows = state.get(id(key))
            seen[self.index] = rows
            if rows is not None:
                ci.note_resolved(_LANE, cache)
            cache[0] = cache[0] + 1
            cache[1] = cache[1] + 1
            return h + 1

    class _Model:
        layers = [_Layer(0), _Layer(1), _Layer(2)]

    run = qm.Qwen4ExpTextModel._compiled_run_fn(_Model(), (0, 1, 2), capture=False)
    flat = [mx.zeros((1, 2)) for _ in range(6)]

    expectations = ci.CacheIdentityExpectations()
    if declare:
        # The dispatch has already committed layers 0 and 2 to consuming this
        # lane's rows.
        expectations.expect(_LANE, 0, real[0])
        expectations.expect(_LANE, 2, real[2])
    error = None
    with ci.expectations_scope(expectations):
        with ci.rewrap_scope(real):
            out = run(mx.zeros((1, 2)), *flat)
            mx.eval(out)
        try:
            ci.assert_satisfied((0, 1, 2), label="unit compiled run")
        except ci.CacheIdentityContractError as failure:
            error = failure
    return real, seen, error


def test_a_resolving_lane_sees_through_the_rewrap(monkeypatch):
    _real, seen, error = _drive_compiled_run(monkeypatch, resolve=True)
    assert error is None
    assert seen == {0: "rows0", 1: None, 2: "rows2"}


def test_a_lane_hidden_by_the_rewrap_raises_at_trace_time(monkeypatch):
    """The failure mode, generically: a miss, not a decline."""

    _real, seen, error = _drive_compiled_run(monkeypatch, resolve=False)
    assert seen == {0: None, 1: None, 2: None}
    assert error is not None
    message = str(error)
    assert _LANE in message
    assert "layer 0" in message and "layer 2" in message
    assert "layer 1" not in message  # layer 1 declared nothing
    assert "resolve_cache_entry" in message


def test_an_undeclared_lane_costs_nothing(monkeypatch):
    """Zero-cost when no lane registers: the assertion cannot fire."""

    _real, seen, error = _drive_compiled_run(
        monkeypatch, resolve=False, declare=False
    )
    assert seen == {0: None, 1: None, 2: None}
    assert error is None


# --------------------------------------------------------------------------
# The other half: a stash on the throwaway that nobody forwards
# --------------------------------------------------------------------------


def _drive_stashing_run(monkeypatch, *, attr: str, capture: bool):
    from mtplx.models import qwen4_exp as qm

    class _Layer:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, h, *, input_ids, ssm_mask, cache):
            # `GatedDeltaNet.__call__` does exactly this with
            # `_mtplx_verify_rows` under the capture scope.
            setattr(cache, attr, (h,) * 6)
            cache[0] = cache[0] + 1
            cache[1] = cache[1] + 1
            return h + 1

    class _Model:
        layers = [_Layer(0), _Layer(1), _Layer(2)]

    run = qm.Qwen4ExpTextModel._compiled_run_fn(
        _Model(), (0, 1, 2), capture=capture
    )
    flat = [mx.zeros((1, 2)) for _ in range(6)]
    return run, flat


def test_a_forwarded_stash_is_allowed(monkeypatch):
    run, flat = _drive_stashing_run(
        monkeypatch, attr="_mtplx_verify_rows", capture=True
    )
    mx.eval(run(mx.zeros((1, 2)), *flat))


def test_a_stash_the_run_never_reads_back_raises(monkeypatch):
    run, flat = _drive_stashing_run(
        monkeypatch, attr="_mtplx_some_new_lane", capture=False
    )
    with pytest.raises(ci.CacheIdentityContractError) as excinfo:
        run(mx.zeros((1, 2)), *flat)
    message = str(excinfo.value)
    assert "_mtplx_some_new_lane" in message
    assert "layer 0" in message


def test_capture_rows_stashed_outside_the_capture_run_raise(monkeypatch):
    """capture=False forwards nothing, so even the known stash is dropped."""

    run, flat = _drive_stashing_run(
        monkeypatch, attr="_mtplx_verify_rows", capture=False
    )
    with pytest.raises(ci.CacheIdentityContractError, match="_mtplx_verify_rows"):
        run(mx.zeros((1, 2)), *flat)


def test_the_alias_itself_is_never_reported_as_a_dropped_stash():
    container = _Entry("throwaway")
    with ci.rewrap_scope([_Entry("real")]):
        ci.bind_rewrapped_entry(container, 0)
    ci.assert_no_dropped_stash(container, 0, forwarded=(), label="unit")


def test_a_container_without_a_dict_is_ignored():
    class _Slotted:
        __slots__ = ()

    ci.assert_no_dropped_stash(_Slotted(), 0, forwarded=(), label="unit")


