"""The fold parity probe's construction environment.

2026-09-02: the probe died on the GPU box before any model load with
``construction environment drifted: {'MTPLX_DROP_EVENTS': ('1', '0')}``.  The
turbo profile sets ``MTPLX_DROP_EVENTS=1``; the probe reports
``stats.events`` as the cross-check that its own per-cycle hooks covered the
whole run, that list is populated only at 0, and it was writing that 0
straight to ``os.environ`` after building the expected environment -- so its
own intentional override read as drift.

The fix routes the override through ``abba_driver.build_family_overrides``
(``--retain-events``, which the driver already supports precisely so "the
effective-environment drift check below still compares equal"), which means
the expected environment carries the probe's value and the drift check stays
strict over every key with NO exclusion list.  These tests pin both halves:
the override no longer trips the check, and a real drift of a key the probe
does not own still does.

Pure Python: no MLX array is evaluated, no model is loaded, no GPU.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "fable" / "probe_gdn_fold_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("_gdn_fold_parity_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


probe = _load()


@pytest.fixture(autouse=True)
def _restore_environ():
    """``apply_environment`` writes ``os.environ`` directly; put it back."""

    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_the_probe_asks_the_driver_for_retained_events():
    args = probe.retained_stack_args()
    assert args.retain_events is True
    family, _candidate = probe.driver.build_family_overrides(args)
    # The driver's own supported route, not a raw os.environ write.
    assert family["MTPLX_DROP_EVENTS"] == "0"


def test_apply_environment_accepts_the_retained_events_override():
    """The 2026-09-02 startup failure, reproduced: ambient 1, probe needs 0."""

    os.environ["MTPLX_DROP_EVENTS"] = "1"
    args = probe.retained_stack_args()
    report = probe.apply_environment(args)
    assert os.environ["MTPLX_DROP_EVENTS"] == "0"
    assert report["family_overrides"]["MTPLX_DROP_EVENTS"] == "0"
    assert report["extra_environment"] == probe.EXTRA_ENV
    # Every key the probe writes raw is reported, so a receipt records the
    # whole construction environment and not just the family half.
    assert "MTPLX_FUSED_HC" in report["probe_raw_environment"]


def test_apply_environment_still_catches_drift_of_a_key_it_does_not_own(
    monkeypatch,
):
    """A profile key stomped after the profile is applied must still raise."""

    from mtplx import profiles

    real = profiles.apply_profile_env

    def stomping_apply_profile_env(*args, **kwargs):
        result = real(*args, **kwargs)
        # MTPLX_QSA_GATHER is a family override the probe never writes raw.
        os.environ["MTPLX_QSA_GATHER"] = "0"
        return result

    monkeypatch.setattr(profiles, "apply_profile_env", stomping_apply_profile_env)
    args = probe.retained_stack_args()
    with pytest.raises(RuntimeError, match="construction environment drifted"):
        probe.apply_environment(args)


def test_the_retained_events_key_is_still_compared_not_excluded(monkeypatch):
    """The override is not an exclusion: a THIRD value still trips the check."""

    from mtplx import profiles

    real = profiles.apply_profile_env

    def stomping_apply_profile_env(*args, **kwargs):
        result = real(*args, **kwargs)
        os.environ["MTPLX_DROP_EVENTS"] = "7"
        return result

    monkeypatch.setattr(profiles, "apply_profile_env", stomping_apply_profile_env)
    args = probe.retained_stack_args()
    with pytest.raises(RuntimeError, match="MTPLX_DROP_EVENTS"):
        probe.apply_environment(args)


def test_a_raw_write_to_an_expected_key_is_refused_loudly(monkeypatch):
    """The rule that replaces an exclusion list.

    Excluding a probe-owned key from the drift check would silently stop
    checking it for everyone.  Instead a raw write to a key the profile or the
    family overrides already own is a construction error naming the key.
    """

    original = probe._probe_raw_env
    monkeypatch.setattr(
        probe,
        "_probe_raw_env",
        # MTPLX_QSA_GATHER is a family override, so writing it raw is exactly
        # the mistake MTPLX_DROP_EVENTS made.
        lambda: {**original(), "MTPLX_QSA_GATHER": "0"},
    )
    args = probe.retained_stack_args()
    with pytest.raises(RuntimeError, match="route them through"):
        probe.apply_environment(args)


def test_the_probe_refuses_a_driver_that_stops_retaining_events(monkeypatch):
    """If --retain-events stopped reaching the family env, say so."""

    original = probe.driver.build_family_overrides

    def family_without_retention(args):
        family, candidate = original(args)
        family.pop("MTPLX_DROP_EVENTS", None)
        return family, candidate

    monkeypatch.setattr(
        probe.driver, "build_family_overrides", family_without_retention
    )
    args = probe.retained_stack_args()
    with pytest.raises(RuntimeError, match="the probe needs retained events"):
        probe.apply_environment(args)


def test_no_raw_key_collides_with_the_expected_environment():
    """The invariant the collision guard enforces, checked directly."""

    from mtplx.profiles import get_profile

    args = probe.retained_stack_args()
    family, _candidate = probe.driver.build_family_overrides(args)
    expected = get_profile("turbo").env_dict()
    expected.update(family)
    raw = probe._probe_raw_env()
    assert set(raw) & set(expected) == set()
    # And the one key that DID collide is now owned by the family half.
    assert "MTPLX_DROP_EVENTS" not in raw
    assert expected["MTPLX_DROP_EVENTS"] == "0"


# --------------------------------------------------------------------------
# The per-window recorder, and the verdict that must refuse a short run
# --------------------------------------------------------------------------
#
# 2026-09-02, receipt gdn-fold-parity-1788400389: a healthy run -- 512 tokens,
# 177 compiled M4 windows, identical text on both arms -- was reported as
# `{"cycles": 1, "first_state_divergence": null, "first_token_divergence_cycle":
# null}`.  `close_cycle()` was only ever called from `finish()`, so every
# window overwrote the pending record and one survived.  The comparison the
# probe exists to make never happened, and only the `--min-cycles` floor
# caught it.


class _FoldStub:
    """Just the counter surface ``ArmRecorder._stats`` reads."""

    def __init__(self) -> None:
        self.STATS = {
            "windows": 0,
            "folded_windows": 0,
            "deferred_commits": 0,
            "flushes": 0,
            "declines": 0,
            "bypassed_commits": 0,
            "ring_depth_hist": {},
        }

    def run_window(self, depth: int) -> None:
        self.STATS["windows"] += 1
        key = str(int(depth))
        hist = self.STATS["ring_depth_hist"]
        hist[key] = hist.get(key, 0) + 1


def _recorder(fold: _FoldStub):
    rec = probe.ArmRecorder(label="unit", state_dir=None, keep_states=0)
    rec._fold = fold
    rec._cache = None  # skips the MLX digest; the bookkeeping is what is under test
    return rec


def _drive(rec, fold: _FoldStub, windows):
    """Replay the hook order: close previous, sample, run window, open, commit."""

    for depth, commit in windows:
        rec.close_cycle()
        before = rec._stats()
        fold.run_window(depth)
        rec.open_window([1, 2, 3, 4], before)
        if commit is not None:
            keep, kind = commit
            after_before = rec._stats()
            if kind == "deferred":
                fold.STATS["deferred_commits"] += 1
            elif kind == "flushed":
                fold.STATS["deferred_commits"] += 1
                fold.STATS["flushes"] += 1
            elif kind == "declined":
                fold.STATS["declines"] += 1
            elif kind == "bypassed":
                fold.STATS["bypassed_commits"] += 1
            rec.note_commit(
                keep_tokens=keep,
                verified_tokens=4,
                committed=True,
                before=after_before,
                after=rec._stats(),
            )
    rec.finish()


def test_the_recorder_keeps_one_record_per_window():
    fold = _FoldStub()
    rec = _recorder(fold)
    plan = [
        (0, (2, "deferred")),
        (1, (1, "deferred")),
        (2, (3, "flushed")),
        (1, None),            # all-accept: no commit
        (0, (1, "declined")),
        (0, (2, "bypassed")),
    ]
    _drive(rec, fold, plan)

    assert len(rec.cycles) == len(plan) == fold.STATS["windows"]
    assert [c["cycle"] for c in rec.cycles] == list(range(len(plan)))
    assert [c["ring_depth_at_entry"] for c in rec.cycles] == [0, 1, 2, 1, 0, 0]


def test_the_recorder_attributes_each_commit_to_its_own_window():
    fold = _FoldStub()
    rec = _recorder(fold)
    _drive(
        rec,
        fold,
        [
            (0, (2, "deferred")),
            (1, (3, "flushed")),
            (1, None),
            (0, (1, "declined")),
            (0, (2, "bypassed")),
        ],
    )
    commits = [c["commit"] for c in rec.cycles]
    assert commits[0]["keep_tokens"] == 2 and commits[0]["deferred"] is True
    assert commits[1]["flushed"] is True and commits[1]["keep_tokens"] == 3
    assert commits[2] is None                      # the all-accept window
    assert commits[3]["declined"] is True
    assert commits[4]["bypassed"] is True


def test_a_commit_with_no_open_window_is_dropped_not_misattributed():
    fold = _FoldStub()
    rec = _recorder(fold)
    # A copy-round commit before any compiled window has run.
    rec.note_commit(
        keep_tokens=1,
        verified_tokens=9,
        committed=True,
        before=rec._stats(),
        after=rec._stats(),
    )
    assert rec.cycles == []
    _drive(rec, fold, [(0, (2, "deferred"))])
    assert len(rec.cycles) == 1
    assert rec.cycles[0]["commit"]["verified_tokens"] == 4


def test_entered_ring_depth_is_none_when_the_histogram_is_ambiguous():
    before = {"ring_depth_hist": {"0": 1}}
    assert probe.ArmRecorder._entered_ring_depth(
        before, {"ring_depth_hist": {"0": 1, "2": 1}}
    ) == 2
    # Two keys moved: the record cannot say which window entered at which.
    assert (
        probe.ArmRecorder._entered_ring_depth(
            before, {"ring_depth_hist": {"0": 2, "1": 1}}
        )
        is None
    )
    # A disarmed arm never records a depth at all.
    assert (
        probe.ArmRecorder._entered_ring_depth(
            {"ring_depth_hist": {}}, {"ring_depth_hist": {}}
        )
        is None
    )


# -- the verdict -----------------------------------------------------------


def _arm(*, cycles: int, compiled: int, traced: int, consumed, tokens):
    return {
        "cycles": [{"cycle": i} for i in range(cycles)],
        "compiled_calls": compiled,
        "prefix_kernel_traced": traced,
        "prefix_consumed": list(consumed),
        "tokens": list(tokens),
    }


def _healthy(min_cycles=150):
    tokens = list(range(512))
    control = _arm(
        cycles=177, compiled=177, traced=0, consumed=[], tokens=tokens
    )
    candidate = _arm(
        cycles=177, compiled=177, traced=36, consumed=[35], tokens=tokens
    )
    return control, candidate


def test_assess_passes_a_healthy_run():
    control, candidate = _healthy()
    assert probe.assess(control, candidate, min_cycles=150) == []


def test_assess_refuses_the_one_window_run():
    """The 2026-09-02 receipt, exactly: identical tokens, one recorded cycle."""

    control, candidate = _healthy()
    for arm in (control, candidate):
        arm["cycles"] = [{"cycle": 0}]
    problems = probe.assess(control, candidate, min_cycles=150)
    assert any("under --min-cycles 150" in p for p in problems)
    # And the instrument's own check fires too: 1 record for 177 windows.
    assert any("dropped records" in p for p in problems)
    assert any(p.startswith("control") for p in problems)
    assert any(p.startswith("candidate") for p in problems)


def test_assess_refuses_a_recorder_that_dropped_cycles_above_the_floor():
    """Even a long run is void if the records do not cover the windows."""

    control, candidate = _healthy()
    candidate["cycles"] = [{"cycle": i} for i in range(160)]
    problems = probe.assess(control, candidate, min_cycles=150)
    assert problems == [
        "candidate recorded 160 cycles for 177 compiled M4 windows: the "
        "probe's per-window hook dropped records, so any per-cycle verdict "
        "below is vacuous"
    ]


def test_assess_refuses_a_candidate_whose_prefix_never_entered_the_graph():
    control, candidate = _healthy()
    candidate["prefix_kernel_traced"] = 0
    candidate["prefix_consumed"] = []
    problems = probe.assess(control, candidate, min_cycles=150)
    assert any("did not enter the compiled verify graph" in p for p in problems)
    assert any("no traced verify reported" in p for p in problems)


def test_assess_refuses_a_partially_bound_prefix():
    """W66d's defect at half strength: some layers took a prefix, some did not."""

    control, candidate = _healthy()
    candidate["prefix_consumed"] = [34]
    problems = probe.assess(control, candidate, min_cycles=150)
    assert any("only [34] of 35 folded GDN layers" in p for p in problems)


def test_assess_refuses_a_control_that_ran_the_fold_kernel():
    control, candidate = _healthy()
    control["prefix_kernel_traced"] = 3
    problems = probe.assess(control, candidate, min_cycles=150)
    assert any("did not actually clear between" in p for p in problems)


def test_assess_refuses_divergent_tokens():
    control, candidate = _healthy()
    candidate["tokens"] = candidate["tokens"][:-1] + [999999]
    assert "the two arms produced different tokens" in probe.assess(
        control, candidate, min_cycles=150
    )


def test_assess_reports_the_banks_window_count_in_the_floor_message():
    """The message must name both numbers, or the next reader repeats the hunt."""

    control, candidate = _healthy()
    control["cycles"] = [{"cycle": 0}]
    control["compiled_calls"] = 177
    problems = probe.assess(control, candidate, min_cycles=150)
    assert "control recorded 1 windows, under --min-cycles 150 (the bank ran 177)" in problems
