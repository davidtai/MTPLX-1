from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qwen38_challenge_dflash_gate as gate


def test_dflash_survivors_are_unique_chronological_and_dependency_closed() -> None:
    assert gate._parse_dflash_survivors("") == ()
    assert gate._parse_dflash_survivors("21,24,26,48") == (
        21,
        24,
        26,
        48,
    )

    for invalid in ("24", "21,18", "18", "17"):
        with pytest.raises(ValueError):
            gate._parse_dflash_survivors(invalid)


def test_dflash_flat_counter_delta_tracks_only_current_arm() -> None:
    assert gate._flat_counter_delta(
        {"memo": 11, "qk": 3},
        {"memo": 18, "qk": 3, "boundary": 4},
    ) == {"boundary": 4, "memo": 7, "qk": 0}


def test_row21_engagement_requires_candidate_only_fused_calls() -> None:
    args = SimpleNamespace(candidate_label="r21")
    by_variant = {
        "control": [
            {"engagement": {"r21_qk_rms_rope": {"calls": 0}}},
            {"engagement": {"r21_qk_rms_rope": {"calls": 0}}},
        ],
        "candidate": [
            {"engagement": {"r21_qk_rms_rope": {"calls": 32}}},
            {"engagement": {"r21_qk_rms_rope": {"calls": 31}}},
        ],
    }

    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1]["engagement"]["r21_qk_rms_rope"]["calls"] = 0
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_optimized_speed_dflash_target_never_constructs_native_mtp(monkeypatch) -> None:
    from mtplx import runtime as runtime_module

    runtime = SimpleNamespace()
    load_calls = []

    def fake_load(path, *, mtp):
        load_calls.append((path, mtp))
        return runtime

    def fake_stack_loader(
        model_path,
        runtime_contract,
        *,
        load_runtime_fn,
        install_draft_head_fn,
    ):
        loaded = load_runtime_fn(model_path, mtp=True)
        head = install_draft_head_fn(loaded, bits=4, group_size=64, mode="affine")
        return loaded, {"contract": runtime_contract, "draft_lm_head_report": head}

    monkeypatch.setattr(runtime_module, "load", fake_load)
    monkeypatch.setattr(gate, "_load_optimized_speed_stack", fake_stack_loader)

    loaded, report = gate._load_optimized_speed_target_stack(
        Path("speed"),
        {"profile": "turbo"},
    )

    assert loaded is runtime
    assert load_calls == [(Path("speed"), False)]
    assert report["native_mtp_loaded"] is False
    assert report["draft_lm_head_report"] == {
        "installed": False,
        "reason": "replaced_by_dflash2",
    }
