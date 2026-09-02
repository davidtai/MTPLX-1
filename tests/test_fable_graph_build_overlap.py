"""W63/W67 gates for MTPLX_FABLE_GRAPH_BUILD_OVERLAP[_LAYERS].

Pure Python.  Nothing here evaluates an MLX array: ``mtplx.graphbank``'s module
level ``mx`` is replaced by a recorder, and every "array" is a tagged sentinel
that raises if anything tries to read it on the host.  The point of the central
test is not that the split path runs -- it is that the split path leaves the
cache in **exactly** the state the shipped monolithic path leaves it in, entry
by entry, slot by slot, so the unchanged device/host commit downstream cannot
tell them apart.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Production census (config: 48 layers, 36 linear / 12 full attention,
# ple_layer_ids == [2] one-indexed, i.e. ONE PLE layer at index 1).
QSA_INDICES = tuple(range(3, 48, 4))
GDN_INDICES = tuple(index for index in range(48) if index not in QSA_INDICES)
PLE_INDEX = 1
STATE_LEAVES = 134  # 35*2 GDN + 1*4 PLE-GDN + 12*5 QSA
CAPTURE_LEAVES = 219  # 36*6 GDN rows + 3 PLE rows
PREFIX_STATE_LEAVES = 2
PREFIX_CAPTURE_LEAVES = 6
#: Depths W67 prices.  4 is the first that puts a QSA layer in the prefix
#: (layers 3, 7, ... are full attention), 2 the first that puts the PLE layer
#: there.  1 is W63's partition and the flag's default.
DEPTHS = (1, 2, 3, 4)


def _state_leaves_of(index: int) -> int:
    if index in QSA_INDICES:
        return 5
    return 4 if index == PLE_INDEX else 2


def _capture_leaves_of(index: int) -> int:
    return 9 if index == PLE_INDEX else 6


def prefix_census(layer_count: int) -> tuple[int, int, int]:
    """(state leaves, capture leaves, capture-plan entries) in layers 0..N-1.

    Spelled out from the production geometry rather than read back off the
    implementation, so a partition that drifts fails here.

        N=1 -> (2, 6, 1)    layer 0, GDN, no PLE
        N=2 -> (6, 15, 2)   + layer 1, the PLE layer (4 state, 9 capture)
        N=3 -> (8, 21, 3)   + layer 2, GDN
        N=4 -> (13, 21, 3)  + layer 3, QSA (5 state, no capture rows)
    """

    state = sum(_state_leaves_of(index) for index in range(layer_count))
    capture = sum(
        _capture_leaves_of(index) for index in GDN_INDICES if index < layer_count
    )
    plan_len = sum(1 for index in GDN_INDICES if index < layer_count)
    return state, capture, plan_len


class Sentinel:
    """A stand-in for one mx.array that refuses every host read."""

    __slots__ = ("tag",)

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.tag}>"

    def __int__(self):
        raise AssertionError(f"host read of {self.tag}")

    def __iter__(self):
        raise AssertionError(f"host iteration of {self.tag}")

    def __array__(self, *_args, **_kwargs):
        raise AssertionError(f"numpy materialization of {self.tag}")


class RecorderMX:
    """Records async_eval / eval calls instead of running them."""

    def __init__(self) -> None:
        self.async_eval_calls: list[tuple] = []
        self.eval_calls: list[tuple] = []

    def async_eval(self, *values):
        self.async_eval_calls.append(tuple(values))

    def eval(self, *values):
        self.eval_calls.append(tuple(values))


class GDNEntry:
    def __init__(self, index: int, slots: int) -> None:
        self.index = index
        self.cache = [Sentinel(f"gdn{index}.pre{slot}") for slot in range(slots)]
        self._mtplx_verify_rows = None
        self._mtplx_verify_ple = None
        self._mtplx_verify_compiled_aux = None

    def census(self):
        return (
            tuple(self.cache),
            self._mtplx_verify_rows,
            self._mtplx_verify_ple,
            self._mtplx_verify_compiled_aux,
        )


class QSAKV:
    def __init__(self, index: int) -> None:
        self.cache = [Sentinel(f"qsa{index}.kv{slot}") for slot in range(3)]
        self.rollback_state = [Sentinel(f"qsa{index}.rb{slot}") for slot in range(3)]


class QSAEntry:
    def __init__(self, index: int) -> None:
        self.index = index
        self.kv = QSAKV(index)
        self.raw_keys = Sentinel(f"qsa{index}.raw")
        self.pooled = Sentinel(f"qsa{index}.pooled")

    def census(self):
        return (
            tuple(self.kv.cache),
            tuple(self.kv.rollback_state),
            self.raw_keys,
            self.pooled,
        )


def _build_cache():
    entries = []
    for index in range(48):
        if index in QSA_INDICES:
            entries.append(QSAEntry(index))
        else:
            entries.append(GDNEntry(index, 4 if index == PLE_INDEX else 2))
    return entries


def _state_plan(graphbank, entries):
    plan = []
    for index, entry in enumerate(entries):
        if index in QSA_INDICES:
            plan.append((graphbank.VERIFY_SPEC_KIND_QSA, entry, 5))
        else:
            plan.append(
                (
                    graphbank.VERIFY_SPEC_KIND_GDN,
                    entry,
                    4 if index == PLE_INDEX else 2,
                )
            )
    return tuple(plan)


def _capture_plan(entries):
    plan = []
    position = 0
    for index in GDN_INDICES:
        count = 9 if index == PLE_INDEX else 6
        plan.append((entries[index], position, count))
        position += count
    assert position == CAPTURE_LEAVES
    return tuple(plan)


def _outputs(prefix_tag: str, count: int, start: int = 0):
    return tuple(Sentinel(f"{prefix_tag}{start + i}") for i in range(count))


@pytest.fixture()
def graphbank(monkeypatch):
    module = importlib.import_module("mtplx.graphbank")
    recorder = RecorderMX()
    monkeypatch.setattr(module, "mx", recorder)
    monkeypatch.setattr(
        module, "_expert_census", SimpleNamespace(end_cycle=lambda: None)
    )
    module._recorder = recorder
    yield module
    del module._recorder


def _shared_outputs(layer_count: int = 1):
    """One set of graph outputs, so two banks can be compared leaf by leaf.

    Split at the depth's boundary, so ``prefix_* + suffix_*`` concatenates to
    exactly the monolithic graph's flat capture and state tuples in the plans'
    own (layer) order -- which is what makes the census comparison a real gate
    and not a restatement of the implementation.
    """

    state_leaves, capture_leaves, _plan_len = prefix_census(layer_count)
    return {
        "logits": Sentinel("logits"),
        "hidden": Sentinel("hidden"),
        "prefix_hidden": Sentinel("prefix_hidden"),
        "prefix_captures": _outputs("cap", capture_leaves),
        "suffix_captures": _outputs(
            "cap", CAPTURE_LEAVES - capture_leaves, start=capture_leaves
        ),
        "prefix_state": _outputs("state", state_leaves),
        "suffix_state": _outputs(
            "state", STATE_LEAVES - state_leaves, start=state_leaves
        ),
        "aux": Sentinel("compiled_aux"),
    }


def _make_bank(
    graphbank,
    *,
    entries,
    donate=True,
    boundary="both",
    shared=None,
    layer_count=1,
):
    bank = graphbank.CompiledVerifyBank.__new__(graphbank.CompiledVerifyBank)
    bank._shadow = []
    bank._held_state_refs = []
    bank._held_aux_refs = []
    bank._held_fixed_m4_split_refs = []
    bank._fixed_m4_overlap_prefix = None
    bank._fixed_m4_split_generation = -1
    bank._fixed_m4_overlap_layers = int(layer_count)
    bank.stats = {
        "calls": 0,
        "compiled_calls": 0,
        "buckets": {},
        "fixed_m4_capacity_transitions": 0,
        "fixed_m4_route_transitions": 0,
    }

    state_leaves, capture_leaves, capture_plan_len = prefix_census(layer_count)
    needs_aux = PLE_INDEX < layer_count
    shared = _shared_outputs(layer_count) if shared is None else shared
    logits = shared["logits"]
    hidden = shared["hidden"]
    prefix_hidden = shared["prefix_hidden"]
    prefix_captures = shared["prefix_captures"]
    suffix_captures = shared["suffix_captures"]
    prefix_state = shared["prefix_state"]
    suffix_state = shared["suffix_state"]

    calls: dict[str, list] = {"prefix": [], "suffix": [], "monolithic": []}
    aux_calls: list = []

    if needs_aux:

        def prefix_fn(input_ids, compiled_aux, *state_in):
            calls["prefix"].append((input_ids, compiled_aux, state_in))
            return (prefix_hidden, *prefix_captures, *prefix_state)

    else:

        def prefix_fn(input_ids, *state_in):
            calls["prefix"].append((input_ids, None, state_in))
            return (prefix_hidden, *prefix_captures, *prefix_state)

    def suffix_fn(prefix_h, input_ids, compiled_aux, *state_in):
        calls["suffix"].append((prefix_h, input_ids, compiled_aux, state_in))
        return (logits, hidden, *suffix_captures, *suffix_state)

    def monolithic_fn(input_ids, compiled_aux, *state_in):
        calls["monolithic"].append((input_ids, compiled_aux, state_in))
        return (
            logits,
            hidden,
            *prefix_captures,
            *suffix_captures,
            *prefix_state,
            *suffix_state,
        )

    aux = shared["aux"]

    def prepare_aux(*args):
        aux_calls.append(args)
        return aux

    bank._fixed_m4_dispatch = {
        "fn": monolithic_fn,
        "prepare_aux": prepare_aux,
        "state_plan": _state_plan(graphbank, entries),
        "state_leaves": STATE_LEAVES,
        "capture_plan": _capture_plan(entries),
        "capture_leaves": CAPTURE_LEAVES,
        "returns_aux": False,
        "aux_contract": "materialized",
        "graph_aux": None,
        "boundary": boundary,
        "donate": donate,
        "overlap_split": {
            "prefix_fn": prefix_fn,
            "suffix_fn": suffix_fn,
            "layer_count": int(layer_count),
            "needs_aux": needs_aux,
            "ple_layer_index": PLE_INDEX,
            "prefix_plan_len": int(layer_count),
            "prefix_capture_plan_len": capture_plan_len,
            "prefix_state_leaves": state_leaves,
            "prefix_capture_leaves": capture_leaves,
            "suffix_capture_leaves": CAPTURE_LEAVES - capture_leaves,
        },
    }
    bank._transition_fixed_m4_generation = lambda *_a, **_k: None
    bank.install_fixed_m4_overlap_split = lambda _n: None
    calls["aux"] = aux_calls
    return bank, calls, aux


def _enqueue(bank, input_ids, *, committed_count, host_input_ids=(1, 2, 3, 4),
             completion_tokens=()):
    """Enqueue with the arguments the decode loop passes."""

    return bank.enqueue_fixed_m4_overlap_prefix(
        input_ids,
        committed_count=committed_count,
        cache=object(),
        host_input_ids=list(host_input_ids),
        completion_tokens=completion_tokens,
    )


def _census(entries):
    return tuple(entry.census() for entry in entries)


# --------------------------------------------------------------------------
# flag surface
# --------------------------------------------------------------------------
def test_flag_is_off_by_default_and_arms_nothing(monkeypatch):
    module = importlib.reload(importlib.import_module("mtplx.graph_build_overlap"))
    monkeypatch.delenv(module.ENV_FLAG, raising=False)
    module.enabled.cache_clear()
    module.items.cache_clear()
    module.timing_enabled.cache_clear()
    assert module.enabled() is False
    assert module.items() == frozenset()
    assert module.engagement_line() is None
    assert module.TIMING is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_flag_accepts_the_shared_truthy_spellings(monkeypatch, value):
    module = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(module.ENV_FLAG, value)
    monkeypatch.delenv(module.ITEMS_ENV, raising=False)
    module.enabled.cache_clear()
    module.items.cache_clear()
    module.timing_enabled.cache_clear()
    try:
        assert module.enabled() is True
        assert module.items() == frozenset(module.DEFAULT_ITEMS)
        assert module.timing_enabled() is False  # timing is opt-in by name
        line = module.engagement_line()
        assert line is not None and module.ENV_FLAG in line
    finally:
        module.enabled.cache_clear()
        module.items.cache_clear()
        module.timing_enabled.cache_clear()


def test_flag_refuses_a_value_it_cannot_parse(monkeypatch):
    module = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(module.ENV_FLAG, "maybe")
    module.enabled.cache_clear()
    try:
        with pytest.raises(ValueError, match=module.ENV_FLAG):
            module.enabled()
    finally:
        module.enabled.cache_clear()


def test_items_refuses_an_unknown_name(monkeypatch):
    module = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(module.ENV_FLAG, "1")
    monkeypatch.setenv(module.ITEMS_ENV, "timing,not_a_thing")
    module.enabled.cache_clear()
    module.items.cache_clear()
    try:
        with pytest.raises(ValueError, match="not_a_thing"):
            module.items()
    finally:
        module.enabled.cache_clear()
        module.items.cache_clear()


def test_receipt_counters_are_a_closed_set():
    module = importlib.import_module("mtplx.graph_build_overlap")
    module.reset_receipt()
    with pytest.raises(KeyError):
        module.bump("prefix_enqueud")  # typo must fail loudly, not vanish
    module.bump("prefix_enqueued")
    assert module.last_receipt()["prefix_enqueued"] == 1
    module.reset_receipt()
    assert module.last_receipt()["prefix_enqueued"] == 0


# --------------------------------------------------------------------------
# the central equivalence gate, at every depth
# --------------------------------------------------------------------------
@pytest.mark.parametrize("depth", DEPTHS)
def test_split_join_publishes_exactly_the_monolithic_census(graphbank, depth):
    """Same leaves, same slots, same capture attributes, same order."""

    input_ids = Sentinel("input_ids")

    shared = _shared_outputs(depth)
    split_entries = _build_cache()
    bank, split_calls, aux = _make_bank(
        graphbank, entries=split_entries, shared=shared, layer_count=depth
    )
    _enqueue(bank, input_ids, committed_count=17, host_input_ids=[5, 6, 7, 8],
             completion_tokens=(1, 2, 3))
    split_logits, split_hidden, split_captures = bank.forward_fixed_m4_overlap(
        Sentinel("unused_host_built_ids"),
        host_input_ids=[5, 6, 7, 8],
        completion_tokens=(1, 2, 3),
        committed_count=17,
        cache=object(),
    )

    mono_entries = _build_cache()
    mono_bank, mono_calls, _mono_aux = _make_bank(
        graphbank, entries=mono_entries, shared=shared, layer_count=depth
    )
    mono_logits, mono_hidden, mono_captures = (
        mono_bank._forward_installed_fixed_m4(
            input_ids,
            [5, 6, 7, 8],
            (1, 2, 3),
            17,
            object(),
        )
    )

    assert _census(split_entries) == _census(mono_entries)
    assert (split_logits, split_hidden, split_captures) == (
        mono_logits,
        mono_hidden,
        mono_captures,
    )
    # and the split really did split.
    assert len(split_calls["prefix"]) == 1
    assert len(split_calls["suffix"]) == 1
    assert split_calls["monolithic"] == []
    assert len(mono_calls["monolithic"]) == 1


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_prefix_owns_exactly_layers_0_to_n_minus_1(graphbank, depth):
    """The partition itself: leaf counts, plan lengths, and who publishes."""

    state_leaves, capture_leaves, plan_len = prefix_census(depth)
    entries = _build_cache()
    bank, calls, _aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    split = bank._fixed_m4_dispatch["overlap_split"]
    assert split["prefix_state_leaves"] == state_leaves
    assert split["prefix_capture_leaves"] == capture_leaves
    assert split["suffix_capture_leaves"] == CAPTURE_LEAVES - capture_leaves
    assert split["prefix_capture_plan_len"] == plan_len

    prefix = _enqueue(bank, Sentinel("input_ids"), committed_count=0)
    assert len(prefix.state_out) == state_leaves
    assert len(prefix.captures) == capture_leaves

    # the prefix graph was handed exactly the prefix layers' state leaves
    _ids, _aux_in, prefix_state_in = calls["prefix"][0]
    assert len(prefix_state_in) == state_leaves

    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    # ... and the suffix graph exactly the rest.
    _h, _sids, _saux, suffix_state_in = calls["suffix"][0]
    assert len(suffix_state_in) == STATE_LEAVES - state_leaves

    # every prefix layer's live state and capture rows came from the prefix
    # tuple, every suffix layer's from the suffix tuple.
    prefix_pos = 0
    for index in range(depth):
        entry = entries[index]
        n_leaves = _state_leaves_of(index)
        if index in QSA_INDICES:
            published = (
                *entry.kv.cache,
                entry.raw_keys,
                entry.pooled,
            )
        else:
            published = tuple(entry.cache[:n_leaves])
        assert published == prefix.state_out[prefix_pos : prefix_pos + n_leaves]
        prefix_pos += n_leaves
    assert prefix_pos == state_leaves

    capture_pos = 0
    for index in GDN_INDICES:
        if index >= depth:
            break
        count = _capture_leaves_of(index)
        assert entries[index]._mtplx_verify_rows == tuple(
            prefix.captures[capture_pos : capture_pos + 6]
        )
        capture_pos += count
    assert capture_pos == capture_leaves

    # the suffix graph was handed the prefix's rooted hidden and its ids
    prefix_hidden, suffix_ids, _aux_in, _state_in = calls["suffix"][0]
    assert prefix_hidden is prefix.hidden
    assert suffix_ids is prefix.input_ids


@pytest.mark.parametrize("depth", DEPTHS)
def test_ple_layer_keeps_its_three_extra_rows_and_the_aux(graphbank, depth):
    """The PLE layer's 9 rows survive whichever side of the seam it lands on."""

    entries = _build_cache()
    bank, _calls, aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    _enqueue(bank, Sentinel("input_ids"), committed_count=0)
    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    ple = entries[PLE_INDEX]
    assert len(ple._mtplx_verify_rows) == 6
    assert len(ple._mtplx_verify_ple) == 3
    assert ple._mtplx_verify_compiled_aux is aux
    # every other GDN layer keeps rows only
    for index in GDN_INDICES:
        if index == PLE_INDEX:
            continue
        assert entries[index]._mtplx_verify_ple is None
        assert entries[index]._mtplx_verify_compiled_aux is None


# --------------------------------------------------------------------------
# W67: the aux hoist
# --------------------------------------------------------------------------
@pytest.mark.parametrize("depth", DEPTHS)
def test_the_auxiliary_is_built_exactly_once_per_window(graphbank, depth):
    entries = _build_cache()
    bank, calls, aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    _enqueue(bank, Sentinel("ids"), committed_count=7,
             host_input_ids=[9, 8, 7, 6], completion_tokens=(4, 5))
    hoisted = PLE_INDEX < depth
    assert len(calls["aux"]) == (1 if hoisted else 0)
    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[9, 8, 7, 6],
        completion_tokens=(4, 5),
        committed_count=7,
        cache=object(),
    )
    assert len(calls["aux"]) == 1
    # and it was built from the window's HOST ids, never from the device array
    (_ids, host_input_ids, completion_tokens, committed) = calls["aux"][0]
    assert host_input_ids == [9, 8, 7, 6]
    assert completion_tokens == (4, 5)
    assert committed == 7
    # the suffix graph got that same object
    _h, _sids, suffix_aux, _state_in = calls["suffix"][0]
    assert suffix_aux is aux


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_hoist_happens_before_the_prefix_submission(graphbank, depth):
    """At depth > 1 the prefix CONSUMES the aux, so it must exist first."""

    entries = _build_cache()
    bank, calls, aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    prefix = _enqueue(bank, Sentinel("ids"), committed_count=0)
    hoisted = PLE_INDEX < depth
    assert (prefix.compiled_aux is aux) is hoisted
    _ids, prefix_aux, _state_in = calls["prefix"][0]
    assert (prefix_aux is aux) is hoisted
    assert prefix.layer_count == depth


@pytest.mark.parametrize("depth", DEPTHS)
def test_a_refused_prefix_reuses_the_auxiliary_it_already_paid_for(
    graphbank, depth
):
    entries = _build_cache()
    bank, calls, aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    _enqueue(bank, Sentinel("stale"), committed_count=11)
    bank.stats["fixed_m4_capacity_transitions"] += 1  # the plan regrew
    bank.forward_fixed_m4_overlap(
        Sentinel("ids"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=11,
        cache=object(),
    )
    assert len(calls["monolithic"]) == 1
    assert calls["suffix"] == []
    # exactly one prepare_aux for the window, on both sides of the fallback
    assert len(calls["aux"]) == 1
    _input_ids, monolithic_aux, _state_in = calls["monolithic"][0]
    assert monolithic_aux is aux


def test_a_window_with_no_prefix_at_all_still_builds_its_own_aux(graphbank):
    entries = _build_cache()
    bank, calls, aux = _make_bank(graphbank, entries=entries, layer_count=3)
    bank.forward_fixed_m4_overlap(
        Sentinel("ids"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    assert len(calls["aux"]) == 1
    _input_ids, monolithic_aux, _state_in = calls["monolithic"][0]
    assert monolithic_aux is aux


def test_the_receipt_counts_the_hoist_and_the_installed_depth(graphbank):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    lane.reset_receipt()
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries, layer_count=3)
    for step in range(3):
        _enqueue(bank, Sentinel(f"ids{step}"), committed_count=step)
        bank.forward_fixed_m4_overlap(
            Sentinel("unused"),
            host_input_ids=[1, 2, 3, 4],
            completion_tokens=(),
            committed_count=step,
            cache=object(),
        )
    receipt = lane.last_receipt()
    assert receipt["aux_hoisted"] == 3
    assert receipt["suffix_joined"] == 3
    assert receipt["monolithic_windows"] == 0
    lane.reset_receipt()


# --------------------------------------------------------------------------
# submission order and array lifetimes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("depth", DEPTHS)
def test_prefix_is_rooted_immediately_and_never_read_on_the_host(
    graphbank, depth
):
    entries = _build_cache()
    bank, _calls, aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    input_ids = Sentinel("input_ids")
    prefix = _enqueue(bank, input_ids, committed_count=3)
    recorder = graphbank._recorder
    state_leaves, _capture, _plan = prefix_census(depth)
    if PLE_INDEX < depth:
        # the hoisted aux crosses the materialization boundary first, with
        # the prefix layers' state, then the prefix outputs are rooted.
        assert len(recorder.async_eval_calls) == 2
        assert recorder.async_eval_calls[0][0] is aux
        assert len(recorder.async_eval_calls[0]) == 1 + state_leaves
        assert recorder.async_eval_calls[1] == prefix.outputs
    else:
        # depth 1 reads no auxiliary: exactly one submission, W63's.
        assert recorder.async_eval_calls == [prefix.outputs]
    assert recorder.eval_calls == []
    # Sentinel raises on __int__ / __iter__ / __array__, so reaching here
    # proves the enqueue never materialized the drafted ids.
    assert prefix.input_ids is input_ids


@pytest.mark.parametrize("depth", DEPTHS)
def test_prefix_does_not_retain_its_input_state_leaves(graphbank, depth):
    """The FixedM4Prefix shape that pins state_in is what defeats donation."""

    entries = _build_cache()
    bank, _calls, _aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    prefix = _enqueue(bank, Sentinel("input_ids"), committed_count=0)
    assert not hasattr(prefix, "state_in")
    assert bank._held_fixed_m4_split_refs == []
    # one slot, not a growing list
    assert bank._fixed_m4_overlap_prefix is prefix
    # and nothing anywhere in the object graph is one of the INPUT leaves
    inputs = set(
        id(leaf)
        for kind, entry, n in bank._fixed_m4_dispatch["state_plan"]
        for leaf in (
            (*entry.kv.cache, entry.raw_keys, entry.pooled)
            if kind == graphbank.VERIFY_SPEC_KIND_QSA
            else entry.cache[:n]
        )
    )
    held = set(id(value) for value in (*prefix.outputs, prefix.hidden))
    assert not (held & inputs)


def test_only_one_prefix_is_ever_held_and_overwrites_are_counted(graphbank):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    lane.reset_receipt()
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    for step in range(5):
        _enqueue(bank, Sentinel(f"ids{step}"), committed_count=step)
    assert bank._fixed_m4_overlap_prefix.input_ids.tag == "ids4"
    # four windows queued a prefix and never joined it; the receipt says so
    assert lane.last_receipt()["prefix_discarded"] == 4
    assert lane.last_receipt()["prefix_enqueued"] == 5
    bank.discard_fixed_m4_overlap_prefix()
    assert bank._fixed_m4_overlap_prefix is None
    bank.discard_fixed_m4_overlap_prefix()  # idempotent
    assert lane.last_receipt()["prefix_discarded"] == 5
    lane.reset_receipt()


@pytest.mark.parametrize("boundary", ["both", "pre"])
@pytest.mark.parametrize("depth", DEPTHS)
def test_the_aux_still_crosses_the_materialization_boundary(
    graphbank, boundary, depth
):
    """_prepare_compiled_verify_aux's contract: aux is rooted before the graph.

    Whichever graph consumes it FIRST is the one it must precede: the suffix
    at depth 1, the prefix at any depth that contains the PLE layer.
    """

    entries = _build_cache()
    bank, _calls, aux = _make_bank(
        graphbank, entries=entries, boundary=boundary, layer_count=depth
    )
    state_leaves, _capture, _plan = prefix_census(depth)
    recorder = graphbank._recorder
    _enqueue(bank, Sentinel("input_ids"), committed_count=0)
    if PLE_INDEX < depth:
        assert recorder.async_eval_calls[0][0] is aux
        assert len(recorder.async_eval_calls[0]) == 1 + state_leaves
    recorder.async_eval_calls.clear()
    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    pre_submission = recorder.async_eval_calls[0]
    if PLE_INDEX < depth:
        # already across; the join roots only the suffix's own state
        assert aux not in pre_submission
        assert len(pre_submission) == STATE_LEAVES - state_leaves
    else:
        assert pre_submission[0] is aux
        assert len(pre_submission) == 1 + (STATE_LEAVES - state_leaves)


@pytest.mark.parametrize("depth", DEPTHS)
def test_donation_drops_state_refs_before_rooting_the_outputs(graphbank, depth):
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(
        graphbank, entries=entries, donate=True, layer_count=depth
    )
    _enqueue(bank, Sentinel("input_ids"), committed_count=0)
    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    assert bank._held_state_refs == []


# --------------------------------------------------------------------------
# the fallbacks
# --------------------------------------------------------------------------
def test_no_queued_prefix_runs_the_shipped_monolithic_route(graphbank):
    entries = _build_cache()
    bank, calls, _aux = _make_bank(graphbank, entries=entries)
    bank.forward_fixed_m4_overlap(
        Sentinel("input_ids"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    assert len(calls["monolithic"]) == 1
    assert calls["prefix"] == [] and calls["suffix"] == []


@pytest.mark.parametrize("depth", DEPTHS)
def test_a_prefix_from_another_window_is_refused(graphbank, depth):
    entries = _build_cache()
    bank, calls, _aux = _make_bank(
        graphbank, entries=entries, layer_count=depth
    )
    _enqueue(bank, Sentinel("stale_ids"), committed_count=11)
    bank.forward_fixed_m4_overlap(
        Sentinel("this_windows_ids"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=12,  # the window moved on; the prefix did not
        cache=object(),
    )
    assert len(calls["monolithic"]) == 1
    assert calls["suffix"] == []
    assert bank._fixed_m4_overlap_prefix is None


def test_a_capacity_transition_under_a_queued_prefix_is_refused(graphbank):
    entries = _build_cache()
    bank, calls, _aux = _make_bank(graphbank, entries=entries)
    _enqueue(bank, Sentinel("ids"), committed_count=4)
    bank.stats["fixed_m4_capacity_transitions"] += 1  # the plan regrew
    bank.forward_fixed_m4_overlap(
        Sentinel("ids"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=4,
        cache=object(),
    )
    assert len(calls["monolithic"]) == 1
    assert calls["suffix"] == []


def test_split_is_recompiled_when_the_generation_moves(graphbank):
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries, layer_count=3)
    rebuilds = []
    bank.install_fixed_m4_overlap_split = lambda count: rebuilds.append(
        (bank._fixed_m4_generation(), count)
    )
    _enqueue(bank, Sentinel("a"), committed_count=0)
    _enqueue(bank, Sentinel("b"), committed_count=1)
    assert rebuilds == [(0, 3)]  # compiled once, not per window
    bank.stats["fixed_m4_route_transitions"] += 1
    _enqueue(bank, Sentinel("c"), committed_count=2)
    assert rebuilds == [(0, 3), (1, 3)]


def test_demote_drops_a_queued_prefix(graphbank):
    """A demoted cache stops mirroring the shadow the prefix was traced on."""

    lane = importlib.import_module("mtplx.graph_build_overlap")
    lane.reset_receipt()
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries, layer_count=3)
    bank.stats["demotions"] = 0
    bank._compiled = {"key": object()}
    bank._shadow_signature = ("sig",)
    bank._spec = [(0, graphbank.VERIFY_SPEC_KIND_GDN, 2)]
    _enqueue(bank, Sentinel("ids"), committed_count=0)
    assert bank._fixed_m4_overlap_prefix is not None

    class _Promoted(graphbank.TensorOffsetQSACache):
        def __init__(self):
            pass

        def demote(self):
            return object()

    cache = [_Promoted()]
    assert bank.demote(cache) == 1
    assert bank._fixed_m4_overlap_prefix is None
    assert lane.last_receipt()["prefix_discarded"] == 1
    # and the pair is forced to recompile against the new shadow
    assert bank._fixed_m4_split_generation == -1
    lane.reset_receipt()


def test_arming_without_an_installed_fixed_m4_plan_raises(graphbank):
    bank = graphbank.CompiledVerifyBank.__new__(graphbank.CompiledVerifyBank)
    bank._fixed_m4_dispatch = None
    with pytest.raises(RuntimeError, match="installed fixed-M4"):
        bank.arm_fixed_m4_graph_build_overlap()


# --------------------------------------------------------------------------
# call-site structure (source-level; no import of the decode loop needed)
# --------------------------------------------------------------------------
def _generation_source() -> str:
    return (ROOT / "mtplx/generation.py").read_text()


def test_prefix_is_queued_at_the_first_statement_that_owns_the_window():
    """After the ids exist, before anything else touches them."""

    source = _generation_source()
    enqueue = source.index("            _graph_overlap_enqueue(\n")
    # every branch that can produce a 4-row fixed-M4 window assigns
    # verify_input_array before the hook ...
    last_assign = source.rindex(
        "verify_input_array = mx.array([verify_input])", 0, enqueue
    )
    # ... and the verify itself comes after it.
    verify = source.index("_fixed_m4_verify(\n", enqueue)
    assert last_assign < enqueue < verify


def test_the_hook_passes_the_same_array_the_monolithic_route_would():
    source = _generation_source()
    block = source[
        source.index("if (\n            _graph_overlap_enqueue is not None") : source.index(
            "        if lazy_bonus_verify:"
        )
    ]
    assert "_graph_overlap_enqueue(\n                verify_input_array," in block
    assert "committed_count=len(tokens) - 1," in block
    # W67's hoist rides the SAME three arguments the verify site passes, so
    # the auxiliary the prefix builds is the one the window would have built.
    assert "host_input_ids=verify_input," in block
    assert "completion_tokens=tokens," in block
    # the same guard the fixed-M4 verify site uses
    assert "verified_token_count == 4" in block
    assert "a3b_target_prefix_route is None" in block
    code = "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#")
    )
    for forbidden in ("np.asarray", "mx.eval", "int("):
        assert forbidden not in code, forbidden


def test_the_verify_site_passes_the_hook_the_same_window_arguments():
    """host_input_ids / completion_tokens must not drift between the two."""

    source = _generation_source()
    hook = source.index("            _graph_overlap_enqueue(\n")
    verify = source.index("                    _fixed_m4_verify(\n", hook)
    verify_block = source[verify : verify + 600]
    assert "host_input_ids=verify_input," in verify_block
    assert "completion_tokens=tokens," in verify_block
    assert "committed_count=len(tokens) - 1," in verify_block
    # and nothing rebinds `tokens` between the two sites
    between = source[hook:verify]
    for mutation in ("tokens.append", "tokens.extend", "tokens = "):
        assert mutation not in between, mutation


def test_the_control_arm_keeps_the_shipped_monolithic_verify():
    source = _generation_source()
    assert "_fixed_m4_verify = (\n        compiled_verify_bank.forward_fixed_m4\n" in source
    assert "verify_logits, verify_hidden, captures = (\n                    _fixed_m4_verify(" in source
    # the lane is resolved once, outside the decode loop
    tree = ast.parse(source)
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_graph_overlap_enqueue"
            for target in node.targets
        )
    ]
    assert len(assigns) == 2  # the None default and the armed binding


def test_an_armed_flag_on_an_unsupported_route_refuses():
    source = _generation_source()
    assert (
        "requires the installed \"\n            \"physical-M4 compiled verify"
        in source
        or "requires the installed " in source
    )
    assert "elif _graph_build_overlap.enabled():" in source


def test_pr391s_layer0_split_is_left_alone():
    """W67 adds a pair; it does not edit the device-committed lane's."""

    source = (ROOT / "mtplx/graphbank.py").read_text()
    tree = ast.parse(source)
    bank = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in bank.body
        if isinstance(node, ast.FunctionDef)
    }
    install = methods["install_fixed_m4_split"]
    assert install.count("mx.compile(") == 2
    assert "prefix_state_leaves != 2" in install
    assert "prefix_capture_leaves != 6" in install
    assert "!= 132" in install and "!= 213" in install
    assert "layer_count" not in install
    # and the runtime's layer-0 forwards keep their hard-wired range
    verifier = ast.parse((ROOT / "mtplx/qwen4_fixed_verify.py").read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in verifier.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "inner.layers[0]" in functions["_forward_fixed_m4_prefix"]
    assert "range(1, len(inner.layers))" in functions["_forward_fixed_m4_suffix"]
    # the W67 pair exists and is range-parameterized
    assert "range(count)" in functions["_forward_fixed_m4_overlap_prefix"]
    assert (
        "range(first, len(inner.layers))"
        in functions["_forward_fixed_m4_overlap_suffix"]
    )


# --------------------------------------------------------------------------
# install gate: the real method, on the production geometry
# --------------------------------------------------------------------------
def _install_ready_bank(graphbank, *, returns_aux=False, layers=48):
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    del bank.install_fixed_m4_overlap_split  # use the real method
    bank._fixed_m4_dispatch["returns_aux"] = returns_aux
    if returns_aux:
        bank._fixed_m4_dispatch["aux_contract"] = "raw_q4"
    bank._spec = [
        (
            index,
            graphbank.VERIFY_SPEC_KIND_QSA
            if index in QSA_INDICES
            else graphbank.VERIFY_SPEC_KIND_GDN,
            _state_leaves_of(index),
        )
        for index in range(layers)
    ]
    bank._extra_capture_layout = tuple(
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
    bank.runtime = SimpleNamespace()
    graphbank._recorder.compile = lambda fn: fn
    return bank


def test_install_fixed_m4_split_accepts_the_production_census(graphbank):
    """PR391's layer-0 install, unchanged, on the real geometry."""

    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    bank._spec = [
        (
            index,
            graphbank.VERIFY_SPEC_KIND_QSA
            if index in QSA_INDICES
            else graphbank.VERIFY_SPEC_KIND_GDN,
            _state_leaves_of(index),
        )
        for index in range(48)
    ]
    bank._extra_capture_layout = tuple(
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
    bank.runtime = SimpleNamespace()
    graphbank._recorder.compile = lambda fn: fn

    bank.install_fixed_m4_split()

    split = bank._fixed_m4_dispatch["split"]
    assert split["prefix_state_leaves"] == PREFIX_STATE_LEAVES
    assert split["prefix_capture_leaves"] == PREFIX_CAPTURE_LEAVES
    assert split["suffix_capture_leaves"] == CAPTURE_LEAVES - PREFIX_CAPTURE_LEAVES


@pytest.mark.parametrize("depth", DEPTHS)
def test_install_fixed_m4_overlap_split_partitions_the_real_census(
    graphbank, depth
):
    """The one path that would raise at request setup, at every depth."""

    state_leaves, capture_leaves, plan_len = prefix_census(depth)
    bank = _install_ready_bank(graphbank)
    bank.install_fixed_m4_overlap_split(depth)

    split = bank._fixed_m4_dispatch["overlap_split"]
    assert split["layer_count"] == depth
    assert split["ple_layer_index"] == PLE_INDEX
    assert split["needs_aux"] is (PLE_INDEX < depth)
    assert split["prefix_state_leaves"] == state_leaves
    assert split["prefix_capture_leaves"] == capture_leaves
    assert split["prefix_capture_plan_len"] == plan_len
    assert split["suffix_capture_leaves"] == CAPTURE_LEAVES - capture_leaves
    lane = importlib.import_module("mtplx.graph_build_overlap")
    assert lane.last_receipt()["prefix_layers"] == depth
    lane.reset_receipt()


@pytest.mark.parametrize("depth", [0, -1, 48, 49])
def test_install_refuses_a_depth_outside_the_plan(graphbank, depth):
    bank = _install_ready_bank(graphbank)
    with pytest.raises(RuntimeError, match="GRAPH_BUILD_OVERLAP_LAYERS"):
        bank.install_fixed_m4_overlap_split(depth)


def test_install_refuses_the_raw_q4_aux_past_the_ple_layer(graphbank):
    """Depth 1 is fine on raw_q4; depth 2 puts the PLE layer in the prefix."""

    bank = _install_ready_bank(graphbank, returns_aux=True)
    bank.install_fixed_m4_overlap_split(1)
    with pytest.raises(RuntimeError, match="materialized auxiliary contract"):
        bank.install_fixed_m4_overlap_split(2)


def test_install_refuses_a_drifted_production_census(graphbank):
    bank = _install_ready_bank(graphbank)
    bank._fixed_m4_dispatch["capture_leaves"] = 220
    with pytest.raises(RuntimeError, match="production census changed"):
        bank.install_fixed_m4_overlap_split(3)


def test_install_refuses_more_than_one_ple_layer(graphbank):
    bank = _install_ready_bank(graphbank)
    bank._extra_capture_layout = tuple(
        (
            index,
            ("qkv", "q", "k", "v", "a", "b")
            + (
                ("ple_hidden", "ple_ids", "ple_conv_rows")
                if index in (PLE_INDEX, 2)
                else ()
            ),
        )
        for index in GDN_INDICES
    )
    with pytest.raises(RuntimeError, match="exactly one PLE layer"):
        bank.install_fixed_m4_overlap_split(3)


def test_arm_reads_the_flag_and_returns_the_installed_depth(
    graphbank, monkeypatch
):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(lane.ENV_FLAG, "1")
    monkeypatch.setenv(lane.LAYERS_ENV, "3")
    lane.enabled.cache_clear()
    lane.layers.cache_clear()
    try:
        bank = _install_ready_bank(graphbank)
        assert bank.arm_fixed_m4_graph_build_overlap() == 3
        assert bank._fixed_m4_dispatch["overlap_split"]["layer_count"] == 3
        assert bank._fixed_m4_overlap_layers == 3
        line = lane.engagement_line(3)
        assert "layer 2/3" in line
    finally:
        lane.enabled.cache_clear()
        lane.layers.cache_clear()
        lane.reset_receipt()


# --------------------------------------------------------------------------
# W67 flag surface
# --------------------------------------------------------------------------
def test_layers_defaults_to_w63s_partition(monkeypatch):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.delenv(lane.LAYERS_ENV, raising=False)
    lane.layers.cache_clear()
    try:
        assert lane.layers() == 1 == lane.DEFAULT_LAYERS
    finally:
        lane.layers.cache_clear()


@pytest.mark.parametrize("value", ["1", "2", "3", "4", "8"])
def test_layers_accepts_the_priced_range(monkeypatch, value):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(lane.LAYERS_ENV, value)
    lane.layers.cache_clear()
    try:
        assert lane.layers() == int(value)
    finally:
        lane.layers.cache_clear()


@pytest.mark.parametrize("value", ["0", "-1", "9", "48", "three", "2.5", ""])
def test_layers_refuses_a_value_it_cannot_use(monkeypatch, value):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(lane.LAYERS_ENV, value)
    lane.layers.cache_clear()
    try:
        if value == "":
            assert lane.layers() == lane.DEFAULT_LAYERS  # unset spelling
        else:
            with pytest.raises(ValueError, match=lane.LAYERS_ENV):
                lane.layers()
    finally:
        lane.layers.cache_clear()


def test_engagement_line_names_the_seam_and_any_mismatch(monkeypatch):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(lane.ENV_FLAG, "1")
    monkeypatch.setenv(lane.LAYERS_ENV, "4")
    lane.enabled.cache_clear()
    lane.items.cache_clear()
    lane.layers.cache_clear()
    try:
        assert "layer 3/4" in lane.engagement_line()
        mismatched = lane.engagement_line(2)
        assert "layer 1/2" in mismatched and "requested 4" in mismatched
    finally:
        lane.enabled.cache_clear()
        lane.items.cache_clear()
        lane.layers.cache_clear()


def test_the_new_receipt_counters_are_in_the_closed_set():
    lane = importlib.import_module("mtplx.graph_build_overlap")
    lane.reset_receipt()
    lane.note_prefix_layers(4)
    lane.note_aux_hoisted()
    receipt = lane.last_receipt()
    assert receipt["prefix_layers"] == 4
    assert receipt["aux_hoisted"] == 1
    with pytest.raises(KeyError):
        lane.bump("aux_hoised")  # typo must fail loudly
    lane.reset_receipt()
    assert lane.last_receipt()["prefix_layers"] == 0


# --------------------------------------------------------------------------
# the micro bench's pure-Python reduce (it only ever runs inside a guarded
# GPU window, so a mistake here would first surface after the window is spent)
# --------------------------------------------------------------------------
def _micro():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "micro_graph_build_overlap",
        ROOT / "scripts/fable/micro_graph_build_overlap.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _micro_row(arm, layers, ms, digest, installed=None):
    return {
        "arm": arm,
        "layers": layers,
        "installed_layers": layers if installed is None else installed,
        "aux_hoisted": 4 if layers > 1 else 0,
        "windows": 4,
        "ms_per_window": ms,
        "decode_tok_s": 1000.0 / ms,
        "host_build_ms": 1.9,
        "monolithic_build_ms": 1.9 if layers == 0 else None,
        "prefix_build_ms": None if layers == 0 else 0.1,
        "suffix_build_ms": None if layers == 0 else 1.8,
        "response_token_sha256": digest,
        "graph_build_overlap": {
            "monolithic_windows": 0,
            "prefix_discarded": 0,
        },
    }


def test_micro_summary_prices_every_depth_against_its_own_ceiling():
    micro = _micro()
    rows = [
        _micro_row("stock", 0, 37.40, "aaa"),
        _micro_row("overlap@1", 1, 36.90, "aaa"),
        _micro_row("overlap@3", 3, 35.85, "aaa"),
    ]
    summary = micro._summarize(rows)
    assert summary["arms"] == ["stock", "overlap@1", "overlap@3"]
    by_depth = {entry["arm"]: entry for entry in summary["by_depth"]}
    assert by_depth["overlap@1"]["predicted_saving_ms"] == 0.53
    assert by_depth["overlap@3"]["predicted_saving_ms"] == round(
        min(3 * 0.53, 45 / 48 * 1.934), 3
    )
    assert abs(by_depth["overlap@1"]["measured_saving_ms"] - 0.50) < 1e-9
    assert abs(by_depth["overlap@3"]["measured_saving_ms"] - 1.55) < 1e-9
    assert by_depth["overlap@3"]["aux_hoisted"] == 4
    assert by_depth["overlap@1"]["aux_hoisted"] == 0
    assert summary["token_digest_match"] is True
    assert summary["ms_per_window"]["overlap@3_delta"] == pytest.approx(-1.55)


def test_micro_summary_flags_a_diverged_arm():
    micro = _micro()
    rows = [
        _micro_row("stock", 0, 37.40, "aaa"),
        _micro_row("overlap@2", 2, 36.00, "bbb"),
    ]
    summary = micro._summarize(rows)
    assert summary["token_digest_match"] is False
    assert summary["by_depth"][0]["token_digest_matches_stock"] is False


def test_micro_summary_surfaces_a_depth_that_did_not_install():
    micro = _micro()
    rows = [
        _micro_row("stock", 0, 37.40, "aaa"),
        _micro_row("overlap@4", 4, 36.00, "aaa", installed=1),
    ]
    summary = micro._summarize(rows)
    assert summary["by_depth"][0]["installed_layers"] == [1]
    assert summary["by_depth"][0]["layers"] == 4


def test_the_join_refuses_a_prefix_whose_aux_contract_disagrees(graphbank):
    """A prefix carrying (or missing) an auxiliary the split does not expect."""

    entries = _build_cache()
    bank, _calls, aux = _make_bank(graphbank, entries=entries, layer_count=3)
    prefix = _enqueue(bank, Sentinel("ids"), committed_count=0)
    assert prefix.compiled_aux is aux
    # forge the disagreement the enqueue could never produce
    import dataclasses

    bank._fixed_m4_overlap_prefix = dataclasses.replace(
        prefix, compiled_aux=None
    )
    with pytest.raises(RuntimeError, match="auxiliary contract disagree"):
        bank.forward_fixed_m4_overlap(
            Sentinel("unused"),
            host_input_ids=[1, 2, 3, 4],
            completion_tokens=(),
            committed_count=0,
            cache=object(),
        )


def test_arm_only_retraces_when_the_depth_changes(graphbank, monkeypatch):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    monkeypatch.setenv(lane.ENV_FLAG, "1")
    lane.enabled.cache_clear()
    lane.layers.cache_clear()
    try:
        entries = _build_cache()
        bank, _calls, _aux = _make_bank(graphbank, entries=entries)
        rebuilds = []
        bank.install_fixed_m4_overlap_split = lambda count: rebuilds.append(count)
        monkeypatch.setenv(lane.LAYERS_ENV, "1")
        lane.layers.cache_clear()
        assert bank.arm_fixed_m4_graph_build_overlap() == 1
        assert rebuilds == [1]
        # re-arming the SAME depth on the same bank must not retrace
        assert bank.arm_fixed_m4_graph_build_overlap() == 1
        assert rebuilds == [1]
        monkeypatch.setenv(lane.LAYERS_ENV, "3")
        lane.layers.cache_clear()
        assert bank.arm_fixed_m4_graph_build_overlap() == 3
        assert rebuilds == [1, 3]
    finally:
        lane.enabled.cache_clear()
        lane.layers.cache_clear()
        lane.reset_receipt()


def test_the_depth_knob_alone_refuses():
    """A LAYERS knob with the lever off would label an arm it did not run."""

    source = _generation_source()
    assert "elif _graph_build_overlap.layers() != _graph_build_overlap.DEFAULT_LAYERS:" in source
    assert "the depth knob does nothing " in source
