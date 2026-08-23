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
    )

    metrics = gate._generation_metrics(stats)

    assert metrics == {
        "prefill_tokens": 512,
        "prefill_time_s": 0.25,
        "prefill_tok_s": 2048.0,
        "decode_tok_s": 40.0,
        "peak_memory_bytes": 24 * 2**30,
        "peak_memory_gib": 24.0,
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
        "dual_norm": False,
        "source_proposal": False,
        "draft_core": "device",
        "source_rows": (8,),
    }


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
