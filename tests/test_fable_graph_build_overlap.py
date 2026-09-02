"""W63 gates for MTPLX_FABLE_GRAPH_BUILD_OVERLAP.

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


def _shared_outputs():
    """One set of graph outputs, so two banks can be compared leaf by leaf."""

    return {
        "logits": Sentinel("logits"),
        "hidden": Sentinel("hidden"),
        "layer0_hidden": Sentinel("layer0_hidden"),
        "prefix_captures": _outputs("cap", PREFIX_CAPTURE_LEAVES),
        "suffix_captures": _outputs(
            "cap", CAPTURE_LEAVES - PREFIX_CAPTURE_LEAVES,
            start=PREFIX_CAPTURE_LEAVES,
        ),
        "prefix_state": _outputs("state", PREFIX_STATE_LEAVES),
        "suffix_state": _outputs(
            "state", STATE_LEAVES - PREFIX_STATE_LEAVES,
            start=PREFIX_STATE_LEAVES,
        ),
        "aux": Sentinel("compiled_aux"),
    }


def _make_bank(graphbank, *, entries, donate=True, boundary="both", shared=None):
    bank = graphbank.CompiledVerifyBank.__new__(graphbank.CompiledVerifyBank)
    bank._shadow = []
    bank._held_state_refs = []
    bank._held_aux_refs = []
    bank._held_fixed_m4_split_refs = []
    bank._fixed_m4_overlap_prefix = None
    bank._fixed_m4_split_generation = -1
    bank.stats = {
        "calls": 0,
        "compiled_calls": 0,
        "buckets": {},
        "fixed_m4_capacity_transitions": 0,
        "fixed_m4_route_transitions": 0,
    }

    shared = _shared_outputs() if shared is None else shared
    logits = shared["logits"]
    hidden = shared["hidden"]
    layer0_hidden = shared["layer0_hidden"]
    prefix_captures = shared["prefix_captures"]
    suffix_captures = shared["suffix_captures"]
    prefix_state = shared["prefix_state"]
    suffix_state = shared["suffix_state"]

    calls: dict[str, list] = {"prefix": [], "suffix": [], "monolithic": []}

    def prefix_fn(input_ids, *state_in):
        calls["prefix"].append((input_ids, state_in))
        return (layer0_hidden, *prefix_captures, *prefix_state)

    def suffix_fn(layer0, input_ids, compiled_aux, *state_in):
        calls["suffix"].append((layer0, input_ids, compiled_aux, state_in))
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
    bank._fixed_m4_dispatch = {
        "fn": monolithic_fn,
        "prepare_aux": lambda *_args: aux,
        "state_plan": _state_plan(graphbank, entries),
        "state_leaves": STATE_LEAVES,
        "capture_plan": _capture_plan(entries),
        "capture_leaves": CAPTURE_LEAVES,
        "returns_aux": False,
        "aux_contract": "materialized",
        "graph_aux": None,
        "boundary": boundary,
        "donate": donate,
        "split": {
            "prefix_fn": prefix_fn,
            "suffix_fn": suffix_fn,
            "prefix_state_leaves": PREFIX_STATE_LEAVES,
            "prefix_capture_leaves": PREFIX_CAPTURE_LEAVES,
            "suffix_capture_leaves": CAPTURE_LEAVES - PREFIX_CAPTURE_LEAVES,
        },
    }
    bank._transition_fixed_m4_generation = lambda *_a, **_k: None
    bank.install_fixed_m4_split = lambda: None
    return bank, calls, aux


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
# the central equivalence gate
# --------------------------------------------------------------------------
def test_split_join_publishes_exactly_the_monolithic_census(graphbank):
    """Same leaves, same slots, same capture attributes, same order."""

    input_ids = Sentinel("input_ids")

    shared = _shared_outputs()
    split_entries = _build_cache()
    bank, split_calls, aux = _make_bank(
        graphbank, entries=split_entries, shared=shared
    )
    bank.enqueue_fixed_m4_overlap_prefix(
        input_ids, committed_count=17, cache=object()
    )
    split_logits, split_hidden, split_captures = bank.forward_fixed_m4_overlap(
        Sentinel("unused_host_built_ids"),
        host_input_ids=[5, 6, 7, 8],
        completion_tokens=(1, 2, 3),
        committed_count=17,
        cache=object(),
    )

    mono_entries = _build_cache()
    mono_bank, mono_calls, _mono_aux = _make_bank(
        graphbank, entries=mono_entries, shared=shared
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
    # and the split really did split: layer 0 came from the prefix graph.
    assert len(split_calls["prefix"]) == 1
    assert len(split_calls["suffix"]) == 1
    assert split_calls["monolithic"] == []
    assert len(mono_calls["monolithic"]) == 1


def test_layer0_capture_and_state_come_from_the_prefix_graph(graphbank):
    entries = _build_cache()
    bank, calls, _aux = _make_bank(graphbank, entries=entries)
    prefix = bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("input_ids"), committed_count=0, cache=object()
    )
    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    layer0 = entries[0]
    assert layer0._mtplx_verify_rows == prefix.captures
    assert tuple(layer0.cache) == prefix.state_out
    # the suffix graph was handed the prefix's rooted hidden and its ids
    layer0_hidden, suffix_ids, _aux_in, _state_in = calls["suffix"][0]
    assert layer0_hidden is prefix.hidden
    assert suffix_ids is prefix.input_ids


def test_ple_layer_keeps_its_three_extra_rows_and_the_aux(graphbank):
    entries = _build_cache()
    bank, _calls, aux = _make_bank(graphbank, entries=entries)
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("input_ids"), committed_count=0, cache=object()
    )
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
# submission order and array lifetimes
# --------------------------------------------------------------------------
def test_prefix_is_rooted_immediately_and_never_read_on_the_host(graphbank):
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    input_ids = Sentinel("input_ids")
    prefix = bank.enqueue_fixed_m4_overlap_prefix(
        input_ids, committed_count=3, cache=object()
    )
    recorder = graphbank._recorder
    # exactly one submission, and it roots the whole prefix output tuple: the
    # GPU has layer 0 to run before the caller's D3 sync.
    assert recorder.async_eval_calls == [prefix.outputs]
    assert recorder.eval_calls == []
    # Sentinel raises on __int__ / __iter__ / __array__, so reaching here
    # proves the enqueue never materialized the drafted ids.
    assert prefix.input_ids is input_ids


def test_prefix_does_not_retain_its_input_state_leaves(graphbank):
    """The FixedM4Prefix shape that pins state_in is what defeats donation."""

    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    prefix = bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("input_ids"), committed_count=0, cache=object()
    )
    assert not hasattr(prefix, "state_in")
    assert bank._held_fixed_m4_split_refs == []
    # one slot, not a growing list
    assert bank._fixed_m4_overlap_prefix is prefix


def test_only_one_prefix_is_ever_held_and_overwrites_are_counted(graphbank):
    lane = importlib.import_module("mtplx.graph_build_overlap")
    lane.reset_receipt()
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    for step in range(5):
        bank.enqueue_fixed_m4_overlap_prefix(
            Sentinel(f"ids{step}"), committed_count=step, cache=object()
        )
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
def test_the_aux_still_crosses_the_materialization_boundary(graphbank, boundary):
    """_prepare_compiled_verify_aux's contract: aux is rooted before the graph."""

    entries = _build_cache()
    bank, _calls, aux = _make_bank(graphbank, entries=entries, boundary=boundary)
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("input_ids"), committed_count=0, cache=object()
    )
    recorder = graphbank._recorder
    recorder.async_eval_calls.clear()
    bank.forward_fixed_m4_overlap(
        Sentinel("unused"),
        host_input_ids=[1, 2, 3, 4],
        completion_tokens=(),
        committed_count=0,
        cache=object(),
    )
    pre_submission = recorder.async_eval_calls[0]
    assert pre_submission[0] is aux
    assert len(pre_submission) == 1 + (STATE_LEAVES - PREFIX_STATE_LEAVES)


def test_donation_drops_state_refs_before_rooting_the_outputs(graphbank):
    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries, donate=True)
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("input_ids"), committed_count=0, cache=object()
    )
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


def test_a_prefix_from_another_window_is_refused(graphbank):
    entries = _build_cache()
    bank, calls, _aux = _make_bank(graphbank, entries=entries)
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("stale_ids"), committed_count=11, cache=object()
    )
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
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("ids"), committed_count=4, cache=object()
    )
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
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    rebuilds = []
    bank.install_fixed_m4_split = lambda: rebuilds.append(
        bank._fixed_m4_generation()
    )
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("a"), committed_count=0, cache=object()
    )
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("b"), committed_count=1, cache=object()
    )
    assert rebuilds == [0]  # compiled once, not per window
    bank.stats["fixed_m4_route_transitions"] += 1
    bank.enqueue_fixed_m4_overlap_prefix(
        Sentinel("c"), committed_count=2, cache=object()
    )
    assert rebuilds == [0, 1]


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
    # the same guard the fixed-M4 verify site uses
    assert "verified_token_count == 4" in block
    assert "a3b_target_prefix_route is None" in block
    code = "\n".join(
        line for line in block.splitlines() if not line.strip().startswith("#")
    )
    for forbidden in ("np.asarray", "mx.eval", "int("):
        assert forbidden not in code, forbidden


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


def test_install_fixed_m4_split_accepts_the_production_census(graphbank):
    """The one path that would raise at request setup, on the real geometry.

    ``install_fixed_m4_split`` is PR391's, and its four census assertions
    (2 layer-0 state leaves, 6 layer-0 capture leaves, 132 / 213 for the rest)
    are exactly what ``arm_fixed_m4_graph_build_overlap`` runs into at the
    request boundary.  Checked against the production shape -- 48 layers,
    36 linear / 12 full attention, one PLE layer at index 1 -- so a drift in
    either census fails here instead of in a benchmark window.
    """

    entries = _build_cache()
    bank, _calls, _aux = _make_bank(graphbank, entries=entries)
    del bank.install_fixed_m4_split  # use the real method
    bank._spec = [
        (
            index,
            graphbank.VERIFY_SPEC_KIND_QSA
            if index in QSA_INDICES
            else graphbank.VERIFY_SPEC_KIND_GDN,
            5 if index in QSA_INDICES else (4 if index == PLE_INDEX else 2),
        )
        for index in range(48)
    ]
    bank._extra_capture_layout = tuple(
        (
            index,
            ("qkv", "q", "k", "v", "a", "b")
            + (("ple_hidden", "ple_ids", "ple_conv_rows") if index == PLE_INDEX else ()),
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
