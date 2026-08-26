from __future__ import annotations

import importlib.util
from pathlib import Path
import json


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


def test_matrix_has_four_fresh_lanes_and_no_historical_pr335_lane() -> None:
    matrix = _module()

    assert matrix.LANE_IDS == (
        "v2.9.2-mlx0322",
        "fixed-k3",
        "full-adaptive",
        "full-q4-adaptive",
    )
    assert matrix.PAIRED_ORDER == (
        "v2.9.2-mlx0322",
        "fixed-k3",
        "full-adaptive",
        "full-q4-adaptive",
        "full-q4-adaptive",
        "full-adaptive",
        "fixed-k3",
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
    assert matrix.PYTHON_CONTEXT_FILE.name == "generation.py"


def test_lane_specs_keep_source_and_head_changes_separate(tmp_path: Path) -> None:
    matrix = _module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    specs = matrix.lane_specs(
        baseline_root=baseline,
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=candidate,
        candidate_commit="c" * 40,
    )

    assert specs["v2.9.2-mlx0322"].source_root == baseline
    assert specs["v2.9.2-mlx0322"].source_commit == matrix.V292_COMMIT
    assert specs["v2.9.2-mlx0322"].route_id == "control"
    assert specs["fixed-k3"].source_root == candidate
    assert specs["fixed-k3"].route_id == "control"
    assert specs["full-adaptive"].route_id == matrix.FULL_ADAPTIVE_NATIVE_ROUTE
    assert specs["full-q4-adaptive"].route_id == (
        matrix.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
    )


def test_128k_uses_one_pass_but_shorter_contexts_use_symmetric_pairs() -> None:
    matrix = _module()

    assert matrix.order_for_context(1_024) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(16_384) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(65_536) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(131_072) == matrix.ONE_PASS_ORDER


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
            "context_file": matrix.PYTHON_CONTEXT_FILE,
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
            "context_file": matrix.PYTHON_CONTEXT_FILE,
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
        "kernel_ids": ["row21", "row61"],
        "feature_receipt": {
            key: {"active": True}
            for key in (
                "r10_compact_vocab",
                "r18_gdn_decay_memo",
                "r21_qk_rms_rope",
                "r24_eval_ladder",
                "r26_prefill_ladder_3",
                "r48_boundary_fused",
                "r50_wired_residency",
                "dual_norm",
            )
        },
        "adaptive_policy_receipt": {"kind": "position_ema", "executed": True},
        "history_route_receipt": {
            "route_id": "kv_only_history",
            "prompt_tokens": 16_384,
            "row20_engaged": True,
        },
        "draft_core": "device",
        "drafted_by_depth": [1, 1, 1],
        "accepted_by_depth": [1, 1, 1],
        "depth_usage": matrix.depth_usage(
            generated_tokens=1_024,
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
        generated_tokens=200,
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
        "kernel_ids": ["engaged"],
        "feature_receipt": {
            key: {"active": True}
            for key in (
                "r10_compact_vocab",
                "r18_gdn_decay_memo",
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
        },
        "history_route_receipt": {
            "route_id": "stock_history",
            "prompt_tokens": 1_024,
            "row20_engaged": False,
        },
        "draft_core": "device",
        "drafted_by_depth": [500, 300, 100],
        "accepted_by_depth": [300, 150, 50],
    }
    base["depth_usage"] = matrix.depth_usage(
        generated_tokens=1_024,
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
