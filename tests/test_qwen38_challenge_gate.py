from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_challenge_port_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_challenge_port_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_metrics_include_prefill_decode_and_peak_memory() -> None:
    gate = _module()
    stats = SimpleNamespace(
        new_prefill_tokens=512,
        prompt_target_prefill_time_s=0.25,
        prompt_target_prefill_tok_s=2048.0,
        decode_tok_s=40.0,
        peak_memory_bytes=24 * 2**30,
        capture_commit_time_s=0.125,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        events=[
            {"capture_repair": "captured_prefix_commit"},
            {"capture_repair": "captured_prefix_pending_correction"},
            {"capture_repair": "standard_reforward"},
        ],
    )

    metrics = gate._generation_metrics(
        stats,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert metrics == {
        "prefill_tokens": 512,
        "prefill_time_s": 0.25,
        "prefill_tok_s": 2048.0,
        "decode_tok_s": 40.0,
        "peak_memory_bytes": 24 * 2**30,
        "peak_memory_gib": 24.0,
        "capture_commit_time_s": 0.125,
        "capture_commit_events": 2,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }


def test_optimized_speed_stack_applies_turbo_before_load_and_installs_q4_head() -> None:
    gate = _module()
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(model=object())
    contract = {
        "recommended_draft_sampler": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        }
    }

    def apply_profile(name, *, runtime_env_overrides):
        calls.append(("profile", name, runtime_env_overrides))

    def load_runtime(path, *, mtp):
        calls.append(("load", path, mtp))
        return runtime

    def install_draft_head(loaded, *, bits, group_size, mode):
        calls.append(("draft_head", loaded, bits, group_size, mode))
        return {"installed": True}

    loaded, stack = gate._load_optimized_speed_stack(
        Path("/model"),
        contract,
        apply_profile_env_fn=apply_profile,
        load_runtime_fn=load_runtime,
        install_draft_head_fn=install_draft_head,
    )

    assert loaded is runtime
    assert calls == [
        ("profile", "turbo", {}),
        ("load", Path("/model"), True),
        ("draft_head", runtime, 4, 64, "affine"),
    ]
    assert stack["profile"] == "turbo"
    assert stack["runtime_profile"] == "native_mtp_turbo"
    assert stack["draft_lm_head"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert stack["draft_sampler"] == contract["recommended_draft_sampler"]
    assert stack["verify_strategy"] == "capture_commit"
    assert stack["verify_core"] == "linear-gdn-from-conv-tape"
    assert stack["base_stack"] == {
        "id": "upstream_main_qwen38_optimized_speed",
        "commit": "bd4421567f9e16ce957c6ef97708b072dcd73937",
        "internal_control_route": "control",
    }


def test_expand_prompt_hits_exact_token_budget() -> None:
    gate = _module()

    class CharacterTokenizer:
        @staticmethod
        def encode(text):
            return [ord(character) for character in text]

        @staticmethod
        def decode(tokens):
            return "".join(chr(token) for token in tokens)

    prompt, token_ids = gate._expand_prompt_to_token_count(
        CharacterTokenizer(),
        "ab",
        9,
    )

    assert prompt == "ab\nab\nab\n"
    assert token_ids == [ord(character) for character in prompt]


def test_context_prompt_preserves_one_tail_instruction_at_exact_budget() -> None:
    gate = _module()

    class CharacterTokenizer:
        @staticmethod
        def encode(text):
            return [ord(character) for character in text]

        @staticmethod
        def decode(tokens):
            return "".join(chr(token) for token in tokens)

    prompt, token_ids = gate._context_prompt_to_token_count(
        CharacterTokenizer(),
        context="0123456789",
        instruction="WRITE-LONG",
        target_tokens=32,
    )

    assert len(token_ids) == 32
    assert prompt.endswith("WRITE-LONG")
    assert prompt.count("WRITE-LONG") == 1


def test_gate_defaults_to_the_16k_generation_python_prompt(monkeypatch) -> None:
    gate = _module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", "/tmp/out.json"])

    args = gate._parse_args()

    assert args.prompt_tokens == 16_384
    assert args.context_file == gate.ROOT / "mtplx/generation.py"
    assert args.max_tokens == 1_024


def test_correctness_requires_exact_cross_route_replay() -> None:
    gate = _module()
    arms = [
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
    ]

    correctness = gate._correctness_summary(
        arms,
        route_ids=["control", "kv_only_history"],
        max_tokens=1024,
    )

    assert correctness["passed"] is True
    assert correctness["full_output"] is True
    assert correctness["cross_route_token_exact"] is True
    assert correctness["per_route_deterministic"] == {
        "control": True,
        "kv_only_history": True,
    }


def test_deterministic_cross_route_drift_is_recorded_without_rejection() -> None:
    gate = _module()
    arms = [
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "candidate-hash",
            "attempted_depth_schedule": [3, 2],
            "accepted_depth_schedule": [1, 2],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "candidate-hash",
            "attempted_depth_schedule": [3, 2],
            "accepted_depth_schedule": [1, 2],
        },
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
    ]
    correctness = gate._correctness_summary(
        arms,
        route_ids=["control", "kv_only_history"],
        max_tokens=1024,
    )

    assert correctness["passed"] is True
    assert correctness["mode"] == "deterministic_drift"
    assert correctness["cross_route_token_exact"] is False
    assert correctness["cross_route_schedule_exact"] is False


def test_empty_event_schedules_do_not_create_vacuous_exactness() -> None:
    gate = _module()
    arms = [
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "same",
            "attempted_depth_schedule": [],
            "accepted_depth_schedule": [],
            "drafted_by_depth": [10, 9, 8],
            "accepted_by_depth": [9, 8, 7],
        },
        {
            "route_id": "candidate",
            "generated_tokens": 1024,
            "token_hash": "same",
            "attempted_depth_schedule": [],
            "accepted_depth_schedule": [],
            "drafted_by_depth": [10, 9, 8],
            "accepted_by_depth": [9, 8, 6],
        },
    ]

    correctness = gate._correctness_summary(
        arms,
        route_ids=["control", "candidate"],
        max_tokens=1024,
    )

    assert correctness["cross_route_schedule_exact"] is False
    assert correctness["schedule_capture"] == "depth_histograms"


def test_route_validation_accepts_the_single_cumulative_winner_stack() -> None:
    gate = _module()

    assert gate._validate_route_id("control") == {"control"}
    assert gate._validate_route_id("kv_only_history") == {"kv_only_history"}
    assert gate._validate_route_id(
        "kv_only_history+dual_norm+source_proposal"
    ) == {
        "kv_only_history",
        "dual_norm",
        "source_proposal",
    }
    assert gate._validate_route_id("r08_device_draft") == {"r08_device_draft"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab"
    ) == {"r08_device_draft", "r10_compact_vocab"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block"
    ) == {"r08_device_draft", "r10_compact_vocab", "r17_q4_mtp_block"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
        "r28_q4_mtp_block"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
        "r36_qkv_islands"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r17_q4_mtp_block",
        "r36_qkv_islands",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo"
    ) == {"r08_device_draft", "r10_compact_vocab", "r18_gdn_decay_memo"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r24_eval_ladder",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r61_dual_norm_concat"
    ) == {"r08_device_draft", "r61_dual_norm_concat"}
    assert gate._validate_route_id(
        "r08_device_draft+r48_boundary_fused"
    ) == {"r08_device_draft", "r48_boundary_fused"}
    assert gate._validate_route_id(
        "r08_device_draft+r50_wired_residency"
    ) == {"r08_device_draft", "r50_wired_residency"}
    assert gate._validate_route_id(
        "r08_device_draft+r63_q8_embedding_dual_norm"
    ) == {"r08_device_draft", "r63_q8_embedding_dual_norm"}
    assert gate._validate_route_id(
        "r08_device_draft+r70_qmv_sumtable+r78_qmv_active_groups+r80_qmv_m2"
    ) == {
        "r08_device_draft",
        "r70_qmv_sumtable",
        "r78_qmv_active_groups",
        "r80_qmv_m2",
    }

    with pytest.raises(ValueError, match="unknown route features"):
        gate._validate_route_id("kv_only_history+dual_norm+qmv_final")
    with pytest.raises(ValueError, match="unknown route features"):
        gate._validate_route_id("packed_qkv")
    with pytest.raises(ValueError, match="unknown route features"):
        gate._validate_route_id("gdn_projection_pairs")


def test_route_validation_rejects_control_combinations() -> None:
    gate = _module()

    with pytest.raises(ValueError, match="control cannot be combined"):
        gate._validate_route_id("control+dual_norm")


def test_row_8_adapts_device_resident_draft_chaining_to_the_fixed_d3_route() -> None:
    gate = _module()

    control = gate._route_execution_options("control")
    row_8 = gate._route_execution_options("r08_device_draft")

    assert control["draft_core"] == "stock"
    assert row_8 == {
        "cache_route": "control",
        "adaptive_policy": "none",
        "speculative_depth": 3,
        "adaptive_depth_cap": 0,
        "dual_norm": False,
            "source_proposal": False,
            "row10_compact_vocab": False,
            "mtp_block_variant": None,
            "row18_gdn_decay_memo": False,
            "row21_qk_rms_rope": False,
        "row24_eval_ladder": False,
        "row26_prefill_ladder_3": False,
        "row48_boundary_fused": False,
        "row50_wired_residency": False,
        "row63_q8_embedding_dual_norm": False,
        "row70_qmv_sumtable": False,
        "row78_qmv_active_groups": False,
        "row80_qmv_m2": False,
        "draft_core": "device",
        "source_rows": (8,),
    }


def test_row_10_extends_retained_row_8_with_compact_proposal_vocabulary() -> None:
    gate = _module()

    row_10 = gate._route_execution_options(
        "r08_device_draft+r10_compact_vocab"
    )

    assert row_10["draft_core"] == "device"
    assert row_10["row10_compact_vocab"] is True
    assert row_10["source_rows"] == (8, 10)


def test_row11_enables_position_ema_depth_four_candidate() -> None:
    gate = _module()

    options = gate._route_execution_options(
        "r08_device_draft+r10_compact_vocab+r11_position_ema"
    )

    assert options["adaptive_policy"] == "position_ema"
    assert options["speculative_depth"] == 4
    assert options["adaptive_depth_cap"] == 4
    assert options["source_rows"] == (8, 10, 11)


def test_row11_promotion_requires_position_ema_policy_events() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r11_position_ema"
    zero = {"route_id": route, "adaptive_policy_events": []}
    engaged = {
        "route_id": route,
        "adaptive_policy_events": [
            {"kind": "position_ema", "attempted_depth": 4, "next_depth": 4}
        ],
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 11 position-EMA adaptive policy did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row_18_decay_memo_extends_rows_8_and_10() -> None:
    gate = _module()
    row_18 = gate._route_execution_options(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo"
    )

    assert row_18["draft_core"] == "device"
    assert row_18["row18_gdn_decay_memo"] is True
    assert row_18["source_rows"] == (8, 10, 18)


def test_row18_promotion_requires_memoized_decay_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo"
    zero = {
        "route_id": route,
        "engagement": {"r18_gdn_decay_memo": {"memo_calls": 0}},
    }
    engaged = {
        "route_id": route,
        "engagement": {"r18_gdn_decay_memo": {"memo_calls": 48}},
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 18 GDN decay memo did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row_20_kv_only_history_extends_rows_8_10_and_18() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history"
    )

    options = gate._route_execution_options(route)

    assert options["cache_route"] == "kv_only_history"
    assert options["source_rows"] == (8, 10, 18, 20)


def test_row20_promotion_requires_kv_only_history_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history"
    )
    zero = {
        "route_id": route,
        "engagement": {
            "r18_gdn_decay_memo": {"memo_calls": 48},
            "r20_kv_only_history": {"calls": 0, "packed_calls": 0},
        },
    }
    engaged = {
        "route_id": route,
        "engagement": {
            "r18_gdn_decay_memo": {"memo_calls": 48},
            "r20_kv_only_history": {"calls": 48, "packed_calls": 48},
        },
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 20 K/V-only history path did not execute",
        "row 20 packed K/V projection did not execute",
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row50_conditions_cache_clearing_candidate_before_control() -> None:
    gate = _module()
    control = "r08_device_draft"
    candidate = control + "+r50_wired_residency"

    assert gate._conditioning_order(
        [control, candidate, candidate, control],
        candidate_id=candidate,
    ) == [candidate, control]
    assert gate._conditioning_order(
        [control, control + "+r48_boundary_fused"],
        candidate_id=control + "+r48_boundary_fused",
    ) == [control, control + "+r48_boundary_fused"]


def test_row21_promotion_requires_qk_fusion_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
        "r18_gdn_decay_memo+r20_kv_only_history+r21_qk_rms_rope"
    )
    block = {
        "r17_q4_mtp_block": {
            "installed": True,
            "active": True,
            "variant": "r17",
            "bits": 4,
            "group_size": 64,
        }
    }
    cumulative = {
        "r18_gdn_decay_memo": {"memo_calls": 48},
        "r20_kv_only_history": {"calls": 48, "packed_calls": 48},
    }
    zero = {
        "route_id": route,
        "feature_receipt": block,
        "engagement": {**cumulative, "r21_qk_rms_rope": {"calls": 0}},
    }
    engaged = {
        "route_id": route,
        "feature_receipt": block,
        "engagement": {**cumulative, "r21_qk_rms_rope": {"calls": 48}},
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 21 fused Q/K RMSNorm+RoPE did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row48_promotion_requires_boundary_fusion_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r48_boundary_fused"
    zero = {
        "route_id": route,
        "engagement": {"r48_boundary_fused": {"calls": 0, "merged_boundaries": 0}},
    }
    engaged = {
        "route_id": route,
        "engagement": {
            "r48_boundary_fused": {"calls": 48, "merged_boundaries": 3024}
        },
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 48 boundary-fused residual/RMSNorm path did not execute",
        "row 48 fused no-copy layer boundaries did not execute",
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row63_promotion_requires_q8_embedding_fusion_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r61_dual_norm_concat+r63_q8_embedding_dual_norm"
    zero = {
        "route_id": route,
        "engagement": {"r63_q8_embedding_dual_norm": {"calls": 0}},
    }
    engaged = {
        "route_id": route,
        "engagement": {"r63_q8_embedding_dual_norm": {"calls": 48}},
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 63 fused Q8 embedding/dual RMSNorm did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row17_promotion_requires_the_pinned_q4_mtp_block() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block"
    wrong = {
        "route_id": route,
        "feature_receipt": {
            "r17_q4_mtp_block": {"installed": True, "active": True, "variant": "r28"}
        },
    }
    engaged = {
        "route_id": route,
        "feature_receipt": {
            "r17_q4_mtp_block": {
                "installed": True,
                "active": True,
                "variant": "r17",
                "bits": 4,
                "group_size": 64,
            }
        },
    }

    assert gate._candidate_engagement_errors(route, [wrong], []) == [
        "row 17 pinned Q4/group-64 MTP block was not active"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [wrong]) == []


def test_row_24_eval_ladder_extends_the_retained_stack() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder"
    )

    options = gate._route_execution_options(route)

    assert options["row24_eval_ladder"] is True
    assert options["source_rows"] == (8, 10, 18, 20, 24)


def test_row24_promotion_requires_eval_ladder_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder"
    )
    cumulative = {
        "r18_gdn_decay_memo": {"memo_calls": 48},
        "r20_kv_only_history": {"calls": 48, "packed_calls": 48},
    }
    zero = {
        "route_id": route,
        "engagement": {**cumulative, "r24_eval_ladder": {"calls": 0}},
    }
    engaged = {
        "route_id": route,
        "engagement": {**cumulative, "r24_eval_ladder": {"calls": 48}},
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 24 target evaluation ladder did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []


def test_row_26_prefill_cadence_extends_retained_row_24() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3"
    )

    options = gate._route_execution_options(route)

    assert options["row26_prefill_ladder_3"] is True
    assert options["source_rows"] == (8, 10, 18, 20, 24, 26)


def test_row26_promotion_requires_prefill_cadence_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3"
    )
    cumulative = {
        "r18_gdn_decay_memo": {"memo_calls": 48},
        "r20_kv_only_history": {"calls": 48, "packed_calls": 48},
        "r24_eval_ladder": {"calls": 48},
    }
    zero = {
        "route_id": route,
        "engagement": {**cumulative, "r26_prefill_ladder_3": {"calls": 0}},
    }
    engaged = {
        "route_id": route,
        "engagement": {**cumulative, "r26_prefill_ladder_3": {"calls": 48}},
    }

    assert gate._candidate_engagement_errors(route, [zero], []) == [
        "row 26 every-third-layer prefill ladder did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [engaged], [zero]) == []
def test_promotion_gate_is_strictly_above_point_zero_five_and_clean() -> None:
    gate = _module()
    order = ["kv_only_history", "kv_only_history+dual_norm"] * 2
    order[2:] = ["kv_only_history+dual_norm", "kv_only_history"]
    kwargs = {
        "order": order,
        "control_id": "kv_only_history",
        "candidate_id": "kv_only_history+dual_norm",
        "correctness": {"passed": True},
        "source_status": [],
    }

    passed = gate._promotion_decision(improvement_pct=0.050453818, **kwargs)
    tied = gate._promotion_decision(improvement_pct=0.05, **kwargs)
    dirty = gate._promotion_decision(
        improvement_pct=0.050453818,
        **{**kwargs, "source_status": [" M mtplx/runtime.py"]},
    )

    assert passed == {"passed": True, "threshold_pct": 0.05, "errors": []}
    assert tied["passed"] is False
    assert any("strictly greater" in error for error in tied["errors"])
    assert dirty["passed"] is False
    assert any("source tree" in error for error in dirty["errors"])


def test_promotion_gate_allows_row28_to_replace_retained_row17_artifact() -> None:
    gate = _module()
    prefix = (
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
        "r18_gdn_decay_memo+r20_kv_only_history+r21_qk_rms_rope+"
        "r24_eval_ladder+r26_prefill_ladder_3"
    )
    candidate = prefix.replace("r17_q4_mtp_block", "r28_q4_mtp_block")

    result = gate._promotion_decision(
        order=[prefix, candidate, candidate, prefix],
        control_id=prefix,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
    )

    assert result == {"passed": True, "threshold_pct": 0.05, "errors": []}
