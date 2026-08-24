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


def test_exact_m5_engagement_requires_every_live_shape() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m5_exact")
    shapes = (
        (5120, 1024), (5120, 6144), (5120, 10240), (5120, 12288),
        (5120, 17408), (5120, 48), (6144, 5120), (17408, 5120),
    )

    def arm(*, exact: bool, missing: tuple[int, int] | None = None):
        counters = {}
        for k, n in shapes:
            counters[f"m5_padded_m6_ksplit_kp2_k{k}_n{n}"] = (
                0 if exact else 64
            )
            counters[f"m5_exact_ksplit_k{k}_n{n}"] = (
                64 if exact and (k, n) != missing else 0
            )
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {"include_m5_exact": exact}
            },
            "engagement": {"nax_verify": counters},
        }

    by_variant = {
        "control": [arm(exact=False) for _ in range(2)],
        "candidate": [arm(exact=True) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(exact=True, missing=shapes[-1])
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m6_kp1_engagement_requires_both_selected_shapes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m6_kp1")
    selected = ((5120, 10240), (5120, 17408))

    def arm(*, kp1: bool, missing: tuple[int, int] | None = None):
        counters = {}
        for k, n in selected:
            counters[f"m6_ksplit_kp2_k{k}_n{n}"] = 0 if kp1 else 64
            counters[f"m6_ksplit_kp1_k{k}_n{n}"] = (
                64 if kp1 and (k, n) != missing else 0
            )
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "include_m5_exact": True,
                    "include_m6_kp1": kp1,
                    "m6_kp1_shapes": (
                        [[5120, 10240], [5120, 17408]] if kp1 else []
                    ),
                }
            },
            "engagement": {"nax_verify": counters},
        }

    by_variant = {
        "control": [arm(kp1=False) for _ in range(2)],
        "candidate": [arm(kp1=True) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(kp1=True, missing=selected[0])
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_row24_phase_removal_requires_other_phase_to_remain_live() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="row24_no_decode")

    def arm(*, decode_active: bool, prefill_calls: int = 176):
        return {
            "feature_receipt": {
                "r24_eval_ladder": {
                    "prefill_active": 1,
                    "decode_active": int(decode_active),
                }
            },
            "engagement": {
                "r24_eval_ladder": {
                    "prefill_calls": prefill_calls,
                    "decode_calls": 1616 if decode_active else 0,
                }
            },
        }

    by_variant = {
        "control": [arm(decode_active=True) for _ in range(2)],
        "candidate": [arm(decode_active=False) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0] = arm(decode_active=False, prefill_calls=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_row48_phase_removal_requires_stock_fallback_and_other_fusion() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="row48_no_prefill")

    def arm(*, prefill_active: bool, decode_fused: int = 12726):
        return {
            "feature_receipt": {
                "r48_boundary_fused": {
                    "prefill_active": int(prefill_active),
                    "decode_active": 1,
                }
            },
            "engagement": {
                "r48_boundary_fused": {
                    "prefill_fused_boundaries": 567 if prefill_active else 0,
                    "prefill_stock_boundaries": 0 if prefill_active else 567,
                    "decode_fused_boundaries": decode_fused,
                    "decode_stock_boundaries": 0,
                }
            },
        }

    by_variant = {
        "control": [arm(prefill_active=True) for _ in range(2)],
        "candidate": [arm(prefill_active=False) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][1] = arm(prefill_active=False, decode_fused=0)
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m8_output_removal_keeps_m7_output_and_other_m8_routes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_no_output")

    def arm(*, output: bool):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {"include_m8_output": output}
            },
            "engagement": {
                "nax_verify": {
                    "m8_nax_k6144_n5120": 1056 if output else 0,
                    "m7_to_m8_nax_k6144_n5120": 992,
                    "m8_nax_k5120_n17408": 7392,
                }
            },
        }

    by_variant = {
        "control": [arm(output=True) for _ in range(2)],
        "candidate": [arm(output=False) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True


def test_m8_qkv_removal_keeps_kv_and_mlp_routes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m8_no_qkv")

    def arm(*, qkv: bool):
        return {
            "feature_receipt": {
                "dflash_m8_nax_island": {
                    "include_m8_qkv": qkv,
                    "include_m8_mlp": True,
                }
            },
            "engagement": {
                "nax_verify": {
                    "m8_nax_k5120_n10240": 3168 if qkv else 0,
                    "m8_nax_k5120_n17408": 7392,
                    "m8_nax_k5120_n1024": 2112,
                }
            },
        }

    by_variant = {
        "control": [arm(qkv=True) for _ in range(2)],
        "candidate": [arm(qkv=False) for _ in range(2)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True


def test_final_phase_stack_requires_every_retained_kernel_and_no_m8_output() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="final_phase_stack_v2")

    control = {"feature_receipt": {}, "engagement": {"nax_verify": {}}}
    candidate = {
        "feature_receipt": {
            "dflash_gqa_widths": {"active": True, "widths": [6, 7, 8]},
            "dflash_m8_nax_island": {
                "include_m8_output": False,
                "include_m7_output": True,
                "include_m7_linear_z": True,
                "include_m8_kv": True,
                "include_m8_qkv": True,
                "include_m8_mlp": True,
                "include_m5_exact": True,
                "include_m6_kp1": True,
            },
        },
        "engagement": {
            "nax_verify": {
                "m8_nax_k6144_n5120": 0,
                "m7_to_m8_nax_k6144_n5120": 912,
                "m7_to_m8_nax_k5120_n6144": 2736,
                "m8_nax_k5120_n1024": 2368,
                "m8_nax_k5120_n10240": 3552,
                "m8_nax_k5120_n17408": 8288,
                "m5_exact_ksplit": 13144,
                "m6_ksplit_kp1": 4960,
            }
        },
    }
    by_variant = {
        "control": [control, control],
        "candidate": [candidate, candidate],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True

    missing_m6 = {
        **candidate,
        "engagement": {
            "nax_verify": {
                **candidate["engagement"]["nax_verify"],
                "m6_ksplit_kp1": 0,
            }
        },
    }
    by_variant["candidate"][0] = missing_m6
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_nax_split_candidate_flags_are_isolated_per_arm() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(
        control_max_mb_per_buffer=512,
        candidate_max_mb_per_buffer=512,
        control_max_ops_per_buffer=50,
        candidate_max_ops_per_buffer=50,
        control_m7_linear_z_nsg4=False,
        candidate_m7_linear_z_nsg4=True,
        control_m8_kv_nsg16=False,
        candidate_m8_kv_nsg16=True,
        control_m8_qkv_nsg4=False,
        candidate_m8_qkv_nsg4=True,
        control_m56_partition_v2=False,
        candidate_m56_partition_v2=True,
        control_m5_partition_v2=False,
        candidate_m5_partition_v2=True,
        control_m6_partition_v2=False,
        candidate_m6_partition_v2=True,
        control_m6_barrier_free_kp1=False,
        candidate_m6_barrier_free_kp1=True,
        control_m56_kconst=False,
        candidate_m56_kconst=True,
    )
    control = stack_gate._variant_environment(args, "control", {})
    candidate = stack_gate._variant_environment(args, "candidate", {})
    assert control["MTPLX_QWEN38_M7_LINEAR_Z_NSG4"] == "0"
    assert candidate["MTPLX_QWEN38_M7_LINEAR_Z_NSG4"] == "1"
    assert control["MTPLX_QWEN38_M8_KV_NSG16"] == "0"
    assert candidate["MTPLX_QWEN38_M8_KV_NSG16"] == "1"
    assert control["MTPLX_QWEN38_M8_QKV_NSG4"] == "0"
    assert candidate["MTPLX_QWEN38_M8_QKV_NSG4"] == "1"
    assert control["MTPLX_QWEN38_M56_PARTITION_V2"] == "0"
    assert candidate["MTPLX_QWEN38_M56_PARTITION_V2"] == "1"
    assert control["MTPLX_QWEN38_M5_PARTITION_V2"] == "0"
    assert candidate["MTPLX_QWEN38_M5_PARTITION_V2"] == "1"
    assert control["MTPLX_QWEN38_M6_PARTITION_V2"] == "0"
    assert candidate["MTPLX_QWEN38_M6_PARTITION_V2"] == "1"
    assert control["MTPLX_QWEN38_M6_BARRIER_FREE_KP1"] == "0"
    assert candidate["MTPLX_QWEN38_M6_BARRIER_FREE_KP1"] == "1"
    assert control["MTPLX_QWEN38_M56_KCONST"] == "0"
    assert candidate["MTPLX_QWEN38_M56_KCONST"] == "1"


def test_m7_linear_z_nsg4_engagement_requires_exact_shape_counter() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m7_linear_z_nsg4")

    def arm(nsg):
        return {
            "feature_receipt": {
                "dflash_nax_split_tuning": {
                    "active": True,
                    "m7_nsg_by_shape": ({"5120x6144": 4} if nsg == 4 else {}),
                    "m8_nsg_by_shape": {},
                }
            },
            "engagement": {
                "nax_verify": {f"m7_to_m8_nax_nsg{nsg}_k5120_n6144": 2736}
            },
        }

    by_variant = {
        "control": [arm(8), arm(8)],
        "candidate": [arm(4), arm(4)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0]["engagement"]["nax_verify"].clear()
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m56_partition_v2_engagement_checks_every_selected_shape() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m56_partition_v2")
    m5 = {"5120x48": 4, "5120x1024": 1, "5120x17408": 1, "17408x5120": 1}
    m6 = {"5120x1024": 1, "5120x10240": 1, "5120x12288": 4, "5120x17408": 2, "17408x5120": 4}

    def arm(active):
        counters = {}
        if active:
            for shape, kp in m5.items():
                k, n = shape.split("x")
                counters[f"m5_exact_ksplit_kp{kp}_k{k}_n{n}"] = 1
            for shape, kp in m6.items():
                k, n = shape.split("x")
                counters[f"m6_ksplit_kp{kp}_k{k}_n{n}"] = 1
        return {
            "feature_receipt": {
                "dflash_m56_partition_tuning": {
                    "active": active,
                    "m5_kparts_by_shape": m5 if active else {},
                    "m6_kparts_by_shape": m6 if active else {},
                }
            },
            "engagement": {"nax_verify": counters},
        }

    by_variant = {
        "control": [arm(False), arm(False)],
        "candidate": [arm(True), arm(True)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0]["engagement"]["nax_verify"].pop(
        "m6_ksplit_kp4_k17408_n5120"
    )
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m6_barrier_free_engagement_requires_both_retained_shapes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m6_barrier_free_kp1")

    def arm(active):
        calls = 1 if active else 0
        return {
            "feature_receipt": {
                "dflash_m6_barrier_free_kp1": {"active": active}
            },
            "engagement": {
                "nax_verify": {
                    "m6_ksplit_kp1_direct_k5120_n10240": calls,
                    "m6_ksplit_kp1_direct_k5120_n17408": calls,
                }
            },
        }

    by_variant = {
        "control": [arm(False), arm(False)],
        "candidate": [arm(True), arm(True)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0]["engagement"]["nax_verify"].pop(
        "m6_ksplit_kp1_direct_k5120_n17408"
    )
    assert stack_gate._engagement_exact(args, by_variant) is False


def test_m56_kconst_engagement_requires_all_three_selected_routes() -> None:
    from scripts import qwen38_challenge_dflash_stack_gate as stack_gate

    args = SimpleNamespace(candidate_label="m56_kconst")

    def arm(active):
        calls = 1 if active else 0
        return {
            "feature_receipt": {
                "dflash_m56_kconst": {
                    "active": active,
                    "m5_shapes": [[5120, 48], [5120, 10240]] if active else [],
                    "m6_shapes": [[5120, 10240]] if active else [],
                }
            },
            "engagement": {
                "nax_verify": {
                    "m5_ksplit_kconst_k5120_n48": calls,
                    "m5_ksplit_kconst_k5120_n10240": calls,
                    "m6_ksplit_kconst_k5120_n10240": calls,
                }
            },
        }

    by_variant = {
        "control": [arm(False), arm(False)],
        "candidate": [arm(True), arm(True)],
    }
    assert stack_gate._engagement_exact(args, by_variant) is True
    by_variant["candidate"][0]["engagement"]["nax_verify"].pop(
        "m5_ksplit_kconst_k5120_n48"
    )
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
