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


def test_dflash_custom_rows_are_dependency_closed() -> None:
    assert gate._parse_dflash_custom_rows("") == ()
    assert gate._parse_dflash_custom_rows("34,40,47") == (34, 40, 47)
    for invalid in ("40", "34,47", "34,34", "34,38"):
        with pytest.raises(ValueError):
            gate._parse_dflash_custom_rows(invalid)


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


def test_custom_row34_engagement_requires_m6_draft_qmv_calls() -> None:
    args = SimpleNamespace(
        candidate_label="c34",
        control_custom_rows="",
        candidate_custom_rows="34",
    )

    def arm(m6):
        return {"engagement": {"r70_qmv_sumtable": {"m6": m6}}}

    by_variant = {
        "control": [arm(0), arm(0)],
        "candidate": [arm(64), arm(63)],
    }

    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_command_buffer_candidate_isolated_environment() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(
        control_max_mb_per_buffer=512,
        candidate_max_mb_per_buffer=1024,
        control_max_ops_per_buffer=50,
        candidate_max_ops_per_buffer=50,
    )

    control = stack_gate._variant_environment(args, "control", {"KEEP": "1"})
    candidate = stack_gate._variant_environment(args, "candidate", {"KEEP": "1"})

    assert control["KEEP"] == candidate["KEEP"] == "1"
    assert control["MLX_MAX_MB_PER_BUFFER"] == "512"
    assert candidate["MLX_MAX_MB_PER_BUFFER"] == "1024"
    assert control["MLX_MAX_OPS_PER_BUFFER"] == "50"
    assert candidate["MLX_MAX_OPS_PER_BUFFER"] == "50"


def test_command_buffer_engagement_requires_exact_variant_caps() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="cb1024")

    def arm(max_mb: int):
        return {
            "feature_receipt": {
                "r53_command_buffers": {
                    "max_mb_per_buffer": max_mb,
                    "max_ops_per_buffer": 50,
                }
            }
        }

    by_variant = {
        "control": [arm(512), arm(512)],
        "candidate": [arm(1024), arm(1024)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(512)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m8_linear_z_engagement_requires_all_64_live_projections() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_linear_z")

    def arm(*, linear: bool, eligible: int, output_calls: int, linear_calls: int):
        shapes = [[6144, 5120]]
        if linear:
            shapes.insert(0, [5120, 6144])
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "active": True,
                    "width": 8,
                    "include_linear_z": linear,
                    "shapes": shapes,
                    "eligible_attention_modules": 16,
                    "eligible_linear_z_projections": 48 if linear else 0,
                    "eligible_projections": eligible,
                }
            },
            "engagement": {
                "nax_verify": {
                    "m8_nax": output_calls + linear_calls,
                    "m8_nax_k6144_n5120": output_calls,
                    "m8_nax_k5120_n6144": linear_calls,
                }
            },
            "adaptive_metrics": {"cycles_by_block": {"8": 66}},
        }

    by_variant = {
        "control": [
            arm(linear=False, eligible=16, output_calls=1056, linear_calls=0)
            for _ in range(2)
        ],
        "candidate": [
            arm(linear=True, eligible=64, output_calls=1056, linear_calls=3168)
            for _ in range(2)
        ],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(
        linear=True,
        eligible=64,
        output_calls=1056,
        linear_calls=0,
    )
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m8_output_engagement_requires_actual_kernel_routes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_nax_island")

    def arm(*, active: bool, calls: int):
        return {
            "feature_receipt": (
                {
                    "dflash_m8_nax_island": {
                        "active": True,
                        "width": 8,
                        "shapes": [[6144, 5120]],
                        "eligible_attention_modules": 16,
                        "validated_projections": 32,
                        "eligible_projections": 16,
                    }
                }
                if active
                else {}
            ),
            "engagement": {
                "nax_verify": {"m8_nax_k6144_n5120": calls}
            },
            "adaptive_metrics": {"cycles_by_block": {"8": 66}},
        }

    by_variant = {
        "control": [arm(active=False, calls=0) for _ in range(2)],
        "candidate": [arm(active=True, calls=1056) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(active=True, calls=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m7_output_engagement_requires_actual_padded_m8_routes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m7_nax_output")

    def arm(*, active: bool, calls: int):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "active": True,
                    "include_m7_output": active,
                    "m7_shapes": [[6144, 5120]] if active else [],
                    "eligible_m7_projections": 16 if active else 0,
                }
            },
            "engagement": {
                "nax_verify": {"m7_to_m8_nax_k6144_n5120": calls}
            },
            "adaptive_metrics": {"cycles_by_block": {"7": 62}},
        }

    by_variant = {
        "control": [arm(active=False, calls=0) for _ in range(2)],
        "candidate": [arm(active=True, calls=992) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(active=True, calls=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m7_linear_z_engagement_requires_incremental_live_routes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m7_nax_linear_z")

    def arm(*, linear: bool, linear_calls: int):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "active": True,
                    "include_m7_output": True,
                    "include_m7_linear_z": linear,
                    "m7_shapes": (
                        [[5120, 6144], [6144, 5120]]
                        if linear
                        else [[6144, 5120]]
                    ),
                    "eligible_m7_projections": 64 if linear else 16,
                    "eligible_m7_linear_z_projections": 48 if linear else 0,
                }
            },
            "engagement": {
                "nax_verify": {
                    "m7_to_m8_nax_k6144_n5120": 992,
                    "m7_to_m8_nax_k5120_n6144": linear_calls,
                }
            },
            "adaptive_metrics": {"cycles_by_block": {"7": 62}},
        }

    by_variant = {
        "control": [arm(linear=False, linear_calls=0) for _ in range(2)],
        "candidate": [arm(linear=True, linear_calls=2976) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(linear=True, linear_calls=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_expanded_m8_engagement_requires_every_winning_shape() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_nax_expanded")
    expected = {
        "m8_nax_k5120_n1024": 2112,
        "m8_nax_k5120_n10240": 3168,
        "m8_nax_k5120_n17408": 7392,
    }

    def arm(*, active: bool, calls: dict[str, int]):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "active": True,
                    "include_m8_expanded": active,
                    "m8_expanded_shapes": (
                        [
                            [5120, 1024],
                            [5120, 10240],
                            [5120, 17408],
                        ]
                        if active
                        else []
                    ),
                    "eligible_m8_expanded_projections": 192 if active else 0,
                }
            },
            "engagement": {"nax_verify": calls},
            "adaptive_metrics": {"cycles_by_block": {"8": 66}},
        }

    zero = {key: 0 for key in expected}
    by_variant = {
        "control": [arm(active=False, calls=zero) for _ in range(2)],
        "candidate": [arm(active=True, calls=expected) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(active=True, calls={**expected, "m8_nax_k5120_n1024": 0})
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m8_kv_engagement_requires_live_subgroup_only() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_nax_kv")

    def arm(*, active: bool, calls: int):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "include_m8_kv": active,
                    "m8_expanded_shapes": [[5120, 1024]] if active else [],
                    "eligible_m8_expanded_projections": 32 if active else 0,
                }
            },
            "engagement": {"nax_verify": {"m8_nax_k5120_n1024": calls}},
        }

    by_variant = {
        "control": [arm(active=False, calls=0) for _ in range(2)],
        "candidate": [arm(active=True, calls=2112) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(active=True, calls=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m8_qkv_engagement_requires_incremental_subgroup() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_nax_qkv")

    def arm(*, qkv: bool, qkv_calls: int):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "include_m8_kv": True,
                    "include_m8_qkv": qkv,
                    "m8_expanded_shapes": (
                        [[5120, 1024], [5120, 10240]]
                        if qkv
                        else [[5120, 1024]]
                    ),
                    "eligible_m8_expanded_projections": 80 if qkv else 32,
                }
            },
            "engagement": {
                "nax_verify": {
                    "m8_nax_k5120_n1024": 2112,
                    "m8_nax_k5120_n10240": qkv_calls,
                }
            },
        }

    by_variant = {
        "control": [arm(qkv=False, qkv_calls=0) for _ in range(2)],
        "candidate": [arm(qkv=True, qkv_calls=3168) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(qkv=True, qkv_calls=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m8_mlp_engagement_requires_incremental_subgroup() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_nax_mlp")

    def arm(*, mlp: bool, mlp_calls: int):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "include_m8_qkv": True,
                    "include_m8_mlp": mlp,
                    "m8_expanded_shapes": (
                        [[5120, 1024], [5120, 10240], [5120, 17408]]
                        if mlp
                        else [[5120, 1024], [5120, 10240]]
                    ),
                    "eligible_m8_expanded_projections": 192 if mlp else 80,
                }
            },
            "engagement": {
                "nax_verify": {
                    "m8_nax_k5120_n10240": 3168,
                    "m8_nax_k5120_n17408": mlp_calls,
                }
            },
        }

    by_variant = {
        "control": [arm(mlp=False, mlp_calls=0) for _ in range(2)],
        "candidate": [arm(mlp=True, mlp_calls=7392) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(mlp=True, mlp_calls=0)
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
