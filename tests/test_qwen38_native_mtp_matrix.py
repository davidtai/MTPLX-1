from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import subprocess
import sys
from typing import Any


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_native_mtp_matrix.py"
ARM_SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_native_mtp_matrix_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_native_mtp_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_module():
    spec = importlib.util.spec_from_file_location(
        "qwen38_native_mtp_matrix_arm", ARM_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _optimized_stack_receipt() -> dict[str, object]:
    return {
        "profile": "turbo",
        "runtime_profile": "native_mtp_turbo",
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
        "draft_sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "runtime_env": {
            "MTPLX_COMPILED_VERIFY": "1",
            "MTPLX_DROP_EVENTS": "1",
            "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
            "MTPLX_MTP_HISTORY_POLICY": "committed",
        },
    }


def test_matrix_has_four_fresh_lanes_and_no_historical_pr335_lane() -> None:
    matrix = _module()

    assert matrix.LANE_IDS == (
        "v2.9.2-mlx0322",
        "full-fixed-k3",
        "full-adaptive",
        "full-q4-adaptive",
    )
    assert matrix.PAIRED_ORDER == (
        "v2.9.2-mlx0322",
        "full-fixed-k3",
        "full-adaptive",
        "full-q4-adaptive",
        "full-q4-adaptive",
        "full-adaptive",
        "full-fixed-k3",
        "v2.9.2-mlx0322",
    )
    assert matrix.ONE_PASS_ORDER == matrix.LANE_IDS
    assert "pr335" not in " ".join(matrix.LANE_IDS).lower()


def test_matrix_workload_contract_redoes_every_requested_context() -> None:
    matrix = _module()

    assert matrix.CONTEXT_TOKENS == (1_024, 16_384, 65_536, 131_072)
    assert matrix.CONDITIONER_OUTPUT_TOKENS == 1_024
    assert matrix.LOW_OUTPUT_TOKENS == 1_024
    assert matrix.XHIGH_OUTPUT_TOKENS == 16_384
    assert matrix.VANITY_PROMPT_TOKENS == 100
    assert matrix.VANITY_TEMPERATURE == 0.0
    assert matrix.VANITY_PROMPT_FILE.name == "qwen38_palindrome_vanity.jsonl"
    assert matrix.PYTHON_PROMPT_FILE.name == "python_modules_long.jsonl"
    assert matrix.PYTHON_CONTEXT_MANIFEST.name == "qwen38-pr335-python-context.json"


def test_frozen_input_artifact_hashes_match_repository_bytes() -> None:
    matrix = _module()

    assert matrix._sha256(matrix.VANITY_PROMPT_FILE) == (
        matrix.PROMPT_ARTIFACT_SHA256["vanity"]
    )
    assert matrix._sha256(matrix.PYTHON_PROMPT_FILE) == (
        matrix.PROMPT_ARTIFACT_SHA256["python"]
    )
    manifest = json.loads(matrix.PYTHON_CONTEXT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest == {
        "path": "mtplx/generation.py",
        "sha256": matrix.PYTHON_CONTEXT_SHA256,
        "source_commit": "9a6f48e69f9c8c6932d0f005c364844b2bf33e9c",
        "source_pr": 335,
    }


def test_campaign_accepts_the_external_pr335_context_by_hash(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    context = tmp_path / "pr335-generation.py"
    row28 = tmp_path / "row28.safetensors"
    args = type(
        "Args",
        (),
        {
            "baseline_root": tmp_path / "baseline",
            "candidate_root": tmp_path / "candidate",
            "output_root": tmp_path / "receipts",
            "workload": "low",
            "prompt_file": matrix.PYTHON_PROMPT_FILE,
            "context_file": context,
            "row28_artifact": row28,
        },
    )()
    monkeypatch.setattr(matrix, "_git_status", lambda root: [])
    hashes = {
        matrix.PYTHON_PROMPT_FILE: matrix.PROMPT_ARTIFACT_SHA256["python"],
        context: matrix.PYTHON_CONTEXT_SHA256,
        row28: matrix.ROW28_ARTIFACT_SHA256,
    }
    monkeypatch.setattr(matrix, "_sha256", hashes.__getitem__)

    matrix._assert_campaign_inputs(args)


def test_lane_specs_keep_source_and_head_changes_separate(tmp_path: Path) -> None:
    matrix = _module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    specs = matrix.lane_specs(
        baseline_root=baseline,
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=candidate,
        candidate_commit="c" * 40,
        workload="low",
    )

    assert specs["v2.9.2-mlx0322"].source_root == baseline
    assert specs["v2.9.2-mlx0322"].source_commit == matrix.V292_COMMIT
    assert specs["v2.9.2-mlx0322"].route_id == "control"
    assert specs["full-fixed-k3"].source_root == candidate
    assert specs["full-fixed-k3"].route_id == matrix.FULL_FIXED_NATIVE_ROUTE
    assert specs["full-adaptive"].route_id == matrix.FULL_ADAPTIVE_NATIVE_ROUTE
    assert specs["full-q4-adaptive"].route_id == (
        matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
    )


def test_full_fixed_k3_uses_the_complete_bf16_stack_without_adaptive_depth(
    tmp_path: Path,
) -> None:
    matrix = _module()

    assert matrix.FULL_FIXED_NATIVE_ROUTE == matrix.gate.FULL_ADAPTIVE_SHARED_ROUTE
    features = matrix.gate._validate_route_id(matrix.FULL_FIXED_NATIVE_ROUTE)
    assert features == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r21_qk_rms_rope",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
        "r48_boundary_fused",
        "r50_wired_residency",
        "r61_dual_norm_concat",
    }
    options = matrix.gate._route_execution_options(matrix.FULL_FIXED_NATIVE_ROUTE)
    assert options["draft_core"] == "device"
    assert options["speculative_depth"] == 3
    assert options["adaptive_policy"] == "none"
    assert options["adaptive_depth_cap"] == 0
    assert options["mtp_block_variant"] is None

    for workload in ("vanity", "low", "xhigh"):
        specs = matrix.lane_specs(
            baseline_root=tmp_path / "baseline",
            baseline_commit=matrix.V292_COMMIT,
            candidate_root=tmp_path / "candidate",
            candidate_commit="c" * 40,
            workload=workload,
        )
        assert specs["full-fixed-k3"].route_id == matrix.FULL_FIXED_NATIVE_ROUTE


def test_only_v292_is_unoptimized(tmp_path: Path) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="low",
    )

    assert specs["v2.9.2-mlx0322"].route_id == "control"
    assert all(
        specs[lane_id].route_id != "control"
        for lane_id in ("full-fixed-k3", "full-adaptive", "full-q4-adaptive")
    )


def test_full_fixed_receipt_requires_bf16_kernels_features_and_no_policy() -> None:
    matrix = _module()
    route = matrix.FULL_FIXED_NATIVE_ROUTE
    receipt = {
        "route_id": route,
        "installed_route_id": (
            "kv_only_history+r18_gdn_decay_memo+r21_qk_rms_rope+"
            "r24_eval_ladder+r26_prefill_ladder_3+r48_boundary_fused+"
            "r50_wired_residency+dual_norm+r10_compact_vocab"
        ),
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "adaptive_policy_receipt": None,
        "adaptive_policy_events": [],
        "kernel_ids": list(matrix.BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            "r10_compact_vocab": {"active": True, "installed": True},
            "r18_gdn_decay_memo": {"active_modules": 48},
            "r20_kv_only_history": {"installed": True},
            "r21_qk_rms_rope": {"active_modules": 16},
            "r24_eval_ladder": {"active": 1},
            "r26_prefill_ladder_3": {"active": 1},
            "r48_boundary_fused": {"active": 1, "construction_bound": 1},
            "r50_wired_residency": {"active": True, "installed": True},
            "dual_norm": {"active": 1},
        },
        "draft_core": "device",
        "device_core_receipt": {
            "requested": "device",
            "device_calls": 8,
            "device_fallbacks": 0,
        },
        "compiled_verify_receipt": {
            "mode": "on",
            "compiled_calls": 8,
            "fallback_reasons": {},
            "permanent_eager": False,
            "permanent_eager_reason": None,
        },
    }

    assert matrix.full_fixed_receipt_errors(receipt, expected_route=route) == []

    receipt["adaptive_policy_receipt"] = {"kind": "position_ema", "executed": True}
    assert "optimized fixed K3 executed an adaptive policy" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )
    receipt["adaptive_policy_receipt"] = None
    receipt["feature_receipt"]["r28_q4_mtp_block"] = {"active": True}
    assert "optimized fixed K3 installed a Q4 MTP block" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )
    del receipt["feature_receipt"]["r28_q4_mtp_block"]
    receipt["device_core_receipt"]["device_fallbacks"] = 1
    assert "optimized fixed K3 device draft fallback occurred" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )


def test_adaptive_receipts_require_the_exact_shared_stack_and_q4_delta() -> None:
    matrix = _module()
    shared_features = {
        key: {"active": True}
        for key in matrix.BF16_OPTIMIZED_FEATURE_KEYS
    }
    bf16 = {
        "route_id": matrix.FULL_ADAPTIVE_NATIVE_ROUTE,
        "installed_route_id": matrix.BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "kernel_ids": list(matrix.BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": shared_features,
    }

    assert matrix.adaptive_optimized_receipt_errors(
        bf16, expected_route=matrix.FULL_ADAPTIVE_NATIVE_ROUTE
    ) == []
    bf16["kernel_ids"] = list(matrix.BF16_OPTIMIZED_KERNEL_IDS[:-1])
    assert "adaptive BF16 kernel stack mismatch" in matrix.adaptive_optimized_receipt_errors(
        bf16, expected_route=matrix.FULL_ADAPTIVE_NATIVE_ROUTE
    )

    q4 = {
        "route_id": matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE,
        "installed_route_id": matrix.Q4_OPTIMIZED_INSTALLED_ROUTE_ID,
        "kernel_ids": list(matrix.Q4_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            **shared_features,
            "r28_q4_mtp_block": {"active": True},
        },
    }
    assert matrix.adaptive_optimized_receipt_errors(
        q4, expected_route=matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
    ) == []
    del q4["feature_receipt"]["r28_q4_mtp_block"]
    assert "adaptive Q4 MTP block is inactive" in matrix.adaptive_optimized_receipt_errors(
        q4, expected_route=matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
    )


def test_arm_policy_contract_distinguishes_fixed_from_adaptive() -> None:
    matrix = _module()
    arm = _arm_module()

    arm._assert_route_policy_contract(
        matrix.FULL_FIXED_NATIVE_ROUTE,
        {"adaptive_policy_receipt": None, "adaptive_policy_events": []},
    )
    try:
        arm._assert_route_policy_contract(
            matrix.FULL_FIXED_NATIVE_ROUTE,
            {
                "adaptive_policy_receipt": {
                    "kind": "position_ema",
                    "executed": True,
                }
            },
        )
    except RuntimeError as exc:
        assert "fixed optimized route executed an adaptive policy" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("fixed optimized route accepted an adaptive policy")

    arm._assert_route_policy_contract(
        matrix.FULL_ADAPTIVE_NATIVE_ROUTE,
        {
            "adaptive_policy_receipt": {
                "kind": "position_ema",
                "executed": True,
            }
        },
    )


def test_vanity_lane_specs_keep_the_complete_optimized_native_mtp_stack(
    tmp_path: Path,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="vanity",
    )

    bf16 = specs["full-adaptive"].route_id
    q4 = specs["full-q4-adaptive"].route_id
    assert bf16 == matrix.FULL_ADAPTIVE_NATIVE_ROUTE
    assert q4 == matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
    assert matrix.gate._route_execution_options(bf16)["draft_core"] == "device"
    assert matrix.gate._route_execution_options(q4)["draft_core"] == "device"
    for optimized in (
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r21_qk_rms_rope",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
        "r48_boundary_fused",
        "r50_wired_residency",
        "r61_dual_norm_concat",
        "r11_position_ema",
    ):
        assert optimized in bf16
    assert "r28_q4_mtp_block" in q4


def test_stochastic_lane_specs_keep_the_measured_device_draft_stack(
    tmp_path: Path,
) -> None:
    matrix = _module()

    for workload in ("low", "xhigh"):
        specs = matrix.lane_specs(
            baseline_root=tmp_path / "baseline",
            baseline_commit=matrix.V292_COMMIT,
            candidate_root=tmp_path / "candidate",
            candidate_commit="c" * 40,
            workload=workload,
        )
        assert specs["full-adaptive"].route_id == matrix.FULL_ADAPTIVE_NATIVE_ROUTE
        assert (
            specs["full-q4-adaptive"].route_id
            == matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
        )
        assert (
            matrix.gate._route_execution_options(
                specs["full-adaptive"].route_id
            )["draft_core"]
            == "device"
        )


def test_128k_uses_one_pass_but_shorter_contexts_use_symmetric_pairs() -> None:
    matrix = _module()

    assert matrix.order_for_context(1_024) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(16_384) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(65_536) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(131_072) == matrix.ONE_PASS_ORDER


def test_aggregate_uses_the_renamed_full_fixed_lane_as_wall_baseline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="low",
    )
    wall_by_lane = {
        "v2.9.2-mlx0322": 12.0,
        "full-fixed-k3": 10.0,
        "full-adaptive": 8.0,
        "full-q4-adaptive": 9.0,
    }
    receipts = [
        {
            "lane_id": lane_id,
            "wall_s": wall_by_lane[lane_id],
            "prefill_tok_s": 800.0,
            "decode_tok_s": 20.0,
            "peak_memory_gib": 40.0,
            "token_hash": lane_id,
        }
        for lane_id in matrix.ONE_PASS_ORDER
    ]
    monkeypatch.setattr(matrix, "receipt_errors", lambda *args, **kwargs: [])

    result = matrix.aggregate(
        workload="low",
        context_tokens=131_072,
        order=matrix.ONE_PASS_ORDER,
        receipts=receipts,
        specs=specs,
    )

    assert result["summary"]["full-fixed-k3"]["wall_faster_vs_fixed_k3_pct"] == 0.0
    assert result["summary"]["full-adaptive"]["wall_faster_vs_fixed_k3_pct"] == 25.0


def test_child_command_attests_source_workload_and_custom_head(tmp_path: Path) -> None:
    matrix = _module()
    lane = matrix.LaneSpec(
        lane_id="full-q4-adaptive",
        source_root=tmp_path / "source",
        source_commit="d" * 40,
        route_id=matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE,
    )
    command = matrix.child_command(
        lane=lane,
        workload="xhigh",
        context_tokens=16_384,
        output=tmp_path / "arm.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row28_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    joined = " ".join(map(str, command))

    for expected in (
        f"--source-root {lane.source_root}",
        f"--source-commit {lane.source_commit}",
        "--lane-id full-q4-adaptive",
        f"--route {matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE}",
        "--workload xhigh",
        "--prompt-tokens 16384",
        "--max-tokens 16384",
        "--warmup-tokens 1024",
        "--target-temperature 1.0",
        "--top-p 0.95",
        "--top-k 20",
    ):
        assert expected in joined
    assert "--record-depth-usage" not in command

    command_128k = matrix.child_command(
        lane=lane,
        workload="xhigh",
        context_tokens=131_072,
        output=tmp_path / "arm-128k.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row28_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    assert "--record-depth-usage" in command_128k

    fixed_command_128k = matrix.child_command(
        lane=matrix.LaneSpec(
            lane_id="full-fixed-k3",
            source_root=tmp_path / "source",
            source_commit="d" * 40,
            route_id=matrix.FULL_FIXED_NATIVE_ROUTE,
        ),
        workload="xhigh",
        context_tokens=131_072,
        output=tmp_path / "fixed-128k.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row28_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    assert "--record-depth-usage" not in fixed_command_128k

    vanity_command = matrix.child_command(
        lane=lane,
        workload="vanity",
        context_tokens=100,
        output=tmp_path / "vanity.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row28_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    vanity_joined = " ".join(map(str, vanity_command))
    assert "--warmup-tokens 0" in vanity_joined


def test_campaign_rejects_dirty_sources_before_entering_gpu_window(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    args = type(
        "Args",
        (),
        {
            "baseline_root": tmp_path / "baseline",
            "candidate_root": tmp_path / "candidate",
            "output_root": tmp_path / "receipts",
            "workload": "low",
            "prompt_file": matrix.PYTHON_PROMPT_FILE,
            "context_file": tmp_path / "pr335-generation.py",
        },
    )()
    monkeypatch.setattr(
        matrix,
        "_git_status",
        lambda root: [" M mtplx/generation.py"] if root == args.candidate_root else [],
    )

    try:
        matrix._assert_campaign_inputs(args)
    except RuntimeError as exc:
        assert "candidate source tree must be clean" in str(exc)
    else:  # pragma: no cover - assertion message is more useful than pytest.raises here
        raise AssertionError("dirty candidate source was accepted")


def test_campaign_rejects_receipt_output_inside_either_source(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    args = type(
        "Args",
        (),
        {
            "baseline_root": baseline,
            "candidate_root": candidate,
            "output_root": candidate / "bench" / "results",
            "workload": "low",
            "prompt_file": matrix.PYTHON_PROMPT_FILE,
            "context_file": tmp_path / "pr335-generation.py",
        },
    )()
    monkeypatch.setattr(matrix, "_git_status", lambda root: [])

    try:
        matrix._assert_campaign_inputs(args)
    except RuntimeError as exc:
        assert "outside the candidate source tree" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("in-tree receipt output was accepted")


def test_receipt_validation_requires_exact_source_and_route_engagement() -> None:
    matrix = _module()
    lane = matrix.LaneSpec(
        lane_id="full-adaptive",
        source_root=Path("candidate"),
        source_commit="e" * 40,
        route_id=matrix.FULL_ADAPTIVE_NATIVE_ROUTE,
    )
    receipt = {
        "lane_id": lane.lane_id,
        "source_commit": lane.source_commit,
        "source_status": [],
        "route_id": lane.route_id,
        "prompt_tokens": 16_384,
        "conditioner_output_tokens": 1_024,
        "max_tokens": 1_024,
        "generated_tokens": 1_024,
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "gpu_lock_scope": "attested_parent",
        "source_import_attested": True,
        "model_id": matrix.MODEL_ID,
        "model_artifact_hashes": {
            "config.json": "a" * 64,
            "mtp.safetensors": "b" * 64,
        },
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "optimized_stack": _optimized_stack_receipt(),
        "workload": "low",
        "sampler": {
            "target_temperature": 1.0,
            "draft_temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
        "prompt_token_sha256": matrix.PROMPT_TOKEN_SHA256["low"][16_384],
        "prompt_artifact_sha256": matrix.PROMPT_ARTIFACT_SHA256["python"],
        "context_artifact_sha256": matrix.PYTHON_CONTEXT_SHA256,
        "row28_artifact_sha256": matrix.ROW28_ARTIFACT_SHA256,
        "source_rows": [8, 10, 18, 20, 21, 24, 26, 48, 50, 61, 11],
        "installed_route_id": matrix.BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "kernel_ids": list(matrix.BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            key: {"active": True}
            for key in (
                "r10_compact_vocab",
                "r18_gdn_decay_memo",
                "r20_kv_only_history",
                "r21_qk_rms_rope",
                "r24_eval_ladder",
                "r26_prefill_ladder_3",
                "r48_boundary_fused",
                "r50_wired_residency",
                "dual_norm",
            )
        },
        "adaptive_policy_receipt": {"kind": "position_ema", "executed": True},
        "device_core_receipt": {
            "requested": "device",
            "device_calls": 400,
            "device_fallbacks": 0,
        },
        "compiled_verify_receipt": {
            "mode": "on",
            "compiled_calls": 400,
            "fallback_reasons": {},
            "permanent_eager": False,
            "permanent_eager_reason": None,
        },
        "history_route_receipt": {
            "route_id": "kv_only_history",
            "prompt_tokens": 16_384,
            "row20_engaged": True,
        },
        "draft_core": "device",
        "drafted_by_depth": [1, 1, 1],
        "accepted_by_depth": [1, 1, 1],
        "verify_calls": 1,
        "attempted_depth_schedule": [0] * 1_020 + [3],
        "depth_usage": matrix.depth_usage(
            decode_cycles=1_021,
            verify_calls=1,
            drafted_by_depth=[1, 1, 1],
            accepted_by_depth=[1, 1, 1],
        ),
    }
    receipt["adaptive_policy_receipt"].update(
        {
            "initial_accept_ema": [0.5, 0.5, 0.5],
            "final_accept_ema": [0.7, 0.6, 0.5],
            "initial_depth": 3,
            "final_depth": 2,
            "max_depth": 3,
            "depth_cap": 3,
        }
    )

    assert matrix.receipt_errors(
        receipt,
        lane=lane,
        context_tokens=16_384,
        output_tokens=1_024,
    ) == []
    receipt["kernel_ids"] = []
    assert "optimized route reported no installed kernels" in matrix.receipt_errors(
        receipt,
        lane=lane,
        context_tokens=16_384,
        output_tokens=1_024,
    )
    receipt["kernel_ids"] = list(matrix.BF16_OPTIMIZED_KERNEL_IDS)
    receipt["model_artifact_hashes"] = {}
    assert "model artifact attestation is missing" in matrix.receipt_errors(
        receipt,
        lane=lane,
        context_tokens=16_384,
        output_tokens=1_024,
    )


class _Tokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        content = messages[0]["content"]
        rendered = "<chat>" + content + "</chat>"
        if kwargs.get("tokenize"):
            return self.encode(rendered)
        return rendered


def test_arm_builds_exact_low_and_xhigh_prompt_budgets() -> None:
    arm = _arm_module()
    tokenizer = _Tokenizer()

    _, low_ids = arm.build_prompt(
        tokenizer,
        workload="low",
        instruction="solve",
        context="abc",
        target_tokens=1_024,
    )
    _, xhigh_ids = arm.build_prompt(
        tokenizer,
        workload="xhigh",
        instruction="solve",
        context="abc",
        target_tokens=16_384,
    )

    assert len(low_ids) == 1_024
    assert len(xhigh_ids) == 16_384
    assert tokenizer.template_calls[0]["enable_thinking"] is True
    assert tokenizer.template_calls[0]["reasoning_effort"] == "low"
    assert tokenizer.template_calls[-1]["enable_thinking"] is True
    assert tokenizer.template_calls[-1]["reasoning_effort"] == "xhigh"


def test_arm_requires_exact_100_token_non_thinking_vanity_prompt() -> None:
    arm = _arm_module()
    tokenizer = _Tokenizer()
    prompt = "x" * (100 - len("<chat></chat>"))

    _, token_ids = arm.build_prompt(
        tokenizer,
        workload="vanity",
        instruction=prompt,
        context="ignored",
        target_tokens=100,
    )

    assert len(token_ids) == 100
    assert tokenizer.template_calls[-1]["enable_thinking"] is False


def test_arm_module_imports_no_mlx_or_mtplx_runtime() -> None:
    source = ARM_SCRIPT.read_text(encoding="utf-8")
    prefix = source.split("def _activate_source_root", maxsplit=1)[0]

    assert "import mlx" not in prefix
    assert "from mtplx" not in prefix


def test_arm_uses_v292_native_internal_history_when_construction_binding_is_absent() -> None:
    arm = _arm_module()

    receipt = arm._history_route_receipt(object(), 16_384)

    assert receipt == {
        "route_id": "native_internal_committed_history",
        "prompt_tokens": 16_384,
        "row20_engaged": False,
        "construction_binding_available": False,
    }


def test_depth_usage_derives_attempted_and_accepted_d0_through_d3() -> None:
    matrix = _module()

    usage = matrix.depth_usage(
        decode_cycles=100,
        verify_calls=80,
        drafted_by_depth=[80, 50, 20],
        accepted_by_depth=[60, 30, 10],
    )

    assert usage["decode_cycles"] == 100
    assert usage["attempted_counts"] == {"D0": 20, "D1": 30, "D2": 30, "D3": 20}
    assert usage["accepted_counts"] == {"D0": 40, "D1": 30, "D2": 20, "D3": 10}
    assert usage["attempted_tokens_by_position"] == {"D1": 80, "D2": 50, "D3": 20}
    assert usage["accepted_tokens_by_position"] == {"D1": 60, "D2": 30, "D3": 10}
    assert usage["acceptance_rate_pct_by_position"] == {
        "D1": 75.0,
        "D2": 60.0,
        "D3": 50.0,
    }
    assert sum(usage["attempted_share_pct"].values()) == 100.0
    assert sum(usage["accepted_share_pct"].values()) == 100.0
    assert usage["mean_attempted_depth"] == 1.5
    assert usage["mean_accepted_depth"] == 1.0


def test_adaptive_receipt_rejects_missing_or_mismatched_depth_telemetry() -> None:
    matrix = _module()
    lane = matrix.LaneSpec(
        lane_id="full-adaptive",
        source_root=Path("candidate"),
        source_commit="e" * 40,
        route_id=matrix.FULL_ADAPTIVE_NATIVE_ROUTE,
    )
    base = {
        "lane_id": lane.lane_id,
        "source_commit": lane.source_commit,
        "source_status": [],
        "route_id": lane.route_id,
        "workload": "low",
        "prompt_tokens": 1_024,
        "conditioner_output_tokens": 1_024,
        "max_tokens": 1_024,
        "generated_tokens": 1_024,
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "gpu_lock_scope": "attested_parent",
        "source_import_attested": True,
        "model_id": matrix.MODEL_ID,
        "model_artifact_hashes": {
            "config.json": "a" * 64,
            "mtp.safetensors": "b" * 64,
        },
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "optimized_stack": _optimized_stack_receipt(),
        "sampler": {
            "target_temperature": 1.0,
            "draft_temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
        "prompt_token_sha256": matrix.PROMPT_TOKEN_SHA256["low"][1_024],
        "prompt_artifact_sha256": matrix.PROMPT_ARTIFACT_SHA256["python"],
        "context_artifact_sha256": matrix.PYTHON_CONTEXT_SHA256,
        "row28_artifact_sha256": matrix.ROW28_ARTIFACT_SHA256,
        "source_rows": [8, 10, 18, 20, 21, 24, 26, 48, 50, 61, 11],
        "installed_route_id": matrix.BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "kernel_ids": list(matrix.BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            key: {"active": True}
            for key in (
                "r10_compact_vocab",
                "r18_gdn_decay_memo",
                "r20_kv_only_history",
                "r21_qk_rms_rope",
                "r24_eval_ladder",
                "r26_prefill_ladder_3",
                "r48_boundary_fused",
                "r50_wired_residency",
                "dual_norm",
            )
        },
        "adaptive_policy_receipt": {
            "kind": "position_ema",
            "executed": True,
            "initial_accept_ema": [0.5, 0.5, 0.5],
            "final_accept_ema": [0.6, 0.5, 0.4],
            "initial_depth": 3,
            "final_depth": 2,
            "max_depth": 3,
            "depth_cap": 3,
        },
        "device_core_receipt": {
            "requested": "device",
            "device_calls": 400,
            "device_fallbacks": 0,
        },
        "compiled_verify_receipt": {
            "mode": "on",
            "compiled_calls": 400,
            "fallback_calls": 0,
            "fallback_reasons": {},
            "permanent_eager": False,
            "permanent_eager_reason": None,
        },
        "history_route_receipt": {
            "route_id": "stock_history",
            "prompt_tokens": 1_024,
            "row20_engaged": False,
        },
        "draft_core": "device",
        "drafted_by_depth": [500, 300, 100],
        "accepted_by_depth": [300, 150, 50],
        "verify_calls": 500,
        "attempted_depth_schedule": [0] * 24
        + [1] * 200
        + [2] * 200
        + [3] * 100,
    }
    base["depth_usage"] = matrix.depth_usage(
        decode_cycles=len(base["attempted_depth_schedule"]),
        verify_calls=base["verify_calls"],
        drafted_by_depth=base["drafted_by_depth"],
        accepted_by_depth=base["accepted_by_depth"],
    )
    assert matrix.receipt_errors(
        base, lane=lane, context_tokens=1_024, output_tokens=1_024
    ) == []

    broken = json.loads(json.dumps(base))
    broken["depth_usage"]["attempted_counts"]["D3"] += 1
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive depth usage does not match raw histograms" in errors

    broken = json.loads(json.dumps(base))
    del broken["adaptive_policy_receipt"]["final_accept_ema"]
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive policy state receipt is incomplete" in errors

    broken = json.loads(json.dumps(base))
    broken["device_core_receipt"]["device_fallbacks"] = 1
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive device draft core did not engage without fallback" in errors

    broken = json.loads(json.dumps(base))
    broken["compiled_verify_receipt"]["fallback_reasons"] = {
        "exception:ValueError": 1
    }
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive compiled verification did not engage cleanly" in errors


def test_parent_preflight_reads_both_distribution_versions_from_selected_python(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    observed: dict[str, object] = {}

    def fake_check_output(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return '{"mlx": "0.32.2", "mlx_metal": "0.32.2"}\n'

    monkeypatch.setattr(matrix.subprocess, "check_output", fake_check_output)
    versions = matrix._interpreter_versions(tmp_path / "python")

    assert versions == {"mlx": "0.32.2", "mlx_metal": "0.32.2"}
    assert observed["command"][0] == str((tmp_path / "python").resolve())
    assert "mlx-metal" in observed["command"][2]


def test_parent_preflight_does_not_dereference_virtualenv_python_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    selected_python = tmp_path / "venv" / "bin" / "python"
    selected_python.parent.mkdir(parents=True)
    selected_python.symlink_to(sys.executable)
    observed: dict[str, object] = {}

    def fake_check_output(command, **kwargs):
        observed["command"] = command
        return '{"mlx": "0.32.2", "mlx_metal": "0.32.2"}\n'

    monkeypatch.setattr(matrix.subprocess, "check_output", fake_check_output)
    matrix._interpreter_versions(selected_python)

    assert observed["command"][0] == str(selected_python.absolute())


def test_matrix_parent_accepts_direct_or_delegated_guard_ownership() -> None:
    matrix = _module()

    assert matrix._validated_parent_guard_scope("direct") == "direct"
    assert matrix._validated_parent_guard_scope("attested_parent") == "attested_parent"


def test_matrix_entrypoint_imports_from_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
