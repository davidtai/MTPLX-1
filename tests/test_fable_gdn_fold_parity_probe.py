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
