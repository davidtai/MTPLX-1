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


def test_dflash_adaptive_rows_are_unique_chronological_and_dependency_closed() -> None:
    assert gate._parse_dflash_adaptive_rows("") == ()
    assert gate._parse_dflash_adaptive_rows("11,15,18") == (11, 15, 18)

    for invalid in ("15", "11,18,15", "11,11", "11,17"):
        with pytest.raises(ValueError):
            gate._parse_dflash_adaptive_rows(invalid)


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


def test_row24_engagement_requires_candidate_ladder_and_qk_fallback() -> None:
    args = SimpleNamespace(candidate_label="r24")

    def arm(ladder: int, fallback: int):
        return {
            "engagement": {
                "r24_eval_ladder": {"calls": ladder},
                "r24_qk_length_limit": {"fallback_calls": fallback},
            }
        }

    by_variant = {
        "control": [arm(0, 0), arm(0, 0)],
        "candidate": [arm(144, 176), arm(144, 176)],
    }

    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(144, 0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_row26_engagement_requires_candidate_prefill_stride_calls() -> None:
    args = SimpleNamespace(candidate_label="r26")

    def arm(calls: int):
        return {"engagement": {"r26_prefill_ladder_3": {"calls": calls}}}

    by_variant = {
        "control": [arm(0), arm(0)],
        "candidate": [arm(176), arm(176)],
    }

    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_row48_engagement_requires_candidate_boundary_fusion() -> None:
    args = SimpleNamespace(candidate_label="r48")

    def arm(calls: int, merged: int):
        return {
            "engagement": {
                "r48_boundary_fused": {
                    "calls": calls,
                    "merged_boundaries": merged,
                }
            }
        }

    by_variant = {
        "control": [arm(0, 0), arm(0, 0)],
        "candidate": [arm(151, 9513), arm(151, 9513)],
    }

    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(151, 0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_row11_engagement_requires_candidate_position_ema_cycles() -> None:
    args = SimpleNamespace(
        candidate_label="a11",
        control_adaptive_rows="",
        candidate_adaptive_rows="11",
    )

    def arm(rows, cycles):
        return {
            "adaptive_metrics": (
                {}
                if not rows
                else {
                    "kind": "qwen38_position_ema",
                    "proposal_rows": list(rows),
                    "cycles": cycles,
                    "cycles_by_block": {"5": cycles},
                }
            )
        }

    by_variant = {
        "control": [arm((), 0), arm((), 0)],
        "candidate": [arm((11,), 190), arm((11,), 191)],
    }

    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm((11,), 0)
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
