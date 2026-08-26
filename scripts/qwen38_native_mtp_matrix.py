#!/usr/bin/env python3
"""Run the four-lane Qwen3.8 native-MTP comparison matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, NamedTuple

try:
    from scripts import qwen38_challenge_port_gate as gate
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    import qwen38_challenge_port_gate as gate  # type: ignore[no-redef]


V292_COMMIT = "bbc67427e88288001e4b90ecb44708dc0222154c"
FULL_ADAPTIVE_NATIVE_ROUTE = gate.FULL_ADAPTIVE_NATIVE_ROUTE
FULL_Q4_ADAPTIVE_NATIVE_ROUTE = gate.FULL_Q4_ADAPTIVE_NATIVE_ROUTE
GREEDY_ADAPTIVE_NATIVE_ROUTE = gate.GREEDY_ADAPTIVE_NATIVE_ROUTE
GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE = gate.GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE

LANE_IDS = (
    "v2.9.2-mlx0322",
    "fixed-k3",
    "full-adaptive",
    "full-q4-adaptive",
)
PAIRED_ORDER = (
    "v2.9.2-mlx0322",
    "fixed-k3",
    "full-adaptive",
    "full-q4-adaptive",
    "full-q4-adaptive",
    "full-adaptive",
    "fixed-k3",
    "v2.9.2-mlx0322",
)
ONE_PASS_ORDER = LANE_IDS
CONTEXT_TOKENS = (1_024, 16_384, 65_536, 131_072)
CONDITIONER_OUTPUT_TOKENS = 1_024
LOW_OUTPUT_TOKENS = 1_024
XHIGH_OUTPUT_TOKENS = 16_384
VANITY_PROMPT_TOKENS = 100
VANITY_TEMPERATURE = 0.0
REQUIRED_MLX_VERSION = "0.32.2"
REQUIRED_MLX_METAL_VERSION = "0.32.2"
MODEL_HASHES_ENV = gate.MODEL_ARTIFACT_HASHES_ENV
ROOT = Path(__file__).resolve().parents[1]
ARM_SCRIPT = ROOT / "scripts/qwen38_native_mtp_matrix_arm.py"
ISOLATED_SCRIPT = ROOT / "scripts/qwen38_challenge_port_isolated_gate.py"
VANITY_PROMPT_FILE = ROOT / "mtplx/benchmarks/prompts/qwen38_palindrome_vanity.jsonl"
PYTHON_PROMPT_FILE = ROOT / "mtplx/benchmarks/prompts/python_modules_long.jsonl"
PYTHON_CONTEXT_FILE = ROOT / "mtplx/generation.py"
PROMPT_ARTIFACT_SHA256 = {
    "vanity": "878a98fe36e5d62566b093b77d11d11bd502fb31e6d2caf7309ea71a9a79bb02",
    "python": "ca2054913c5c27c24c983ed27e3ee4eff1d01d456a73e71377fdaea3cbf8c140",
}
PYTHON_CONTEXT_SHA256 = "454a8d33d514456c0e1dc6dfccbff2e473e3068f601a54514e964d2b21b17751"
ROW28_ARTIFACT_SHA256 = "c934b40f1254858425cc0b5fdfe62b6ae13d1a4aff74da9d81606e92fdcf41ee"
PROMPT_TOKEN_SHA256 = {
    "vanity": {
        100: "94a188b7cacc378c60a6503feea97429c59f6dab3980635eaa5e35da1e6b767b",
    },
    "low": {
        1_024: "3015401ec3e421502b1a23f18d0a6e5d53004b189fdbab0e2e3ba27802fcd7e6",
        16_384: "af141694261c1d3c4d8aa6e36e903fa55fae08e2fc3ad21ad78ebcde213f6954",
        65_536: "0a042777fa323cbf0304270d72bf84990aa326653170333594ba45365f0e1fda",
        131_072: "1fba20935b8828ab480208e7bbb882aeaeaca6b5031f752962c7a488daa4699e",
    },
    "xhigh": {
        1_024: "27d46cfc472799e56283a42c9f2abfa0811ba0591b7846694ea43a11e9b79cb4",
        16_384: "f604437b2a74a7dbfc3fd9c51acd5fb5e301e2dcea12b29587c9c738182deb12",
        65_536: "01a9e6f1ebc8000bc5480c0bdfd593358e737afafb7502b0d64cea29bfbffe61",
        131_072: "9651a2343f9aee2c538f581bec0115b2c215abeeb0f15ea4d140a31e8800272e",
    },
}


class LaneSpec(NamedTuple):
    lane_id: str
    source_root: Path
    source_commit: str
    route_id: str


def lane_specs(
    *,
    baseline_root: Path,
    baseline_commit: str,
    candidate_root: Path,
    candidate_commit: str,
    workload: str,
) -> dict[str, LaneSpec]:
    if workload not in {"vanity", "low", "xhigh"}:
        raise ValueError(f"unknown workload: {workload}")
    adaptive_route = (
        GREEDY_ADAPTIVE_NATIVE_ROUTE
        if workload == "vanity"
        else FULL_ADAPTIVE_NATIVE_ROUTE
    )
    q4_adaptive_route = (
        GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE
        if workload == "vanity"
        else FULL_Q4_ADAPTIVE_NATIVE_ROUTE
    )
    return {
        "v2.9.2-mlx0322": LaneSpec(
            "v2.9.2-mlx0322", baseline_root, baseline_commit, "control"
        ),
        "fixed-k3": LaneSpec(
            "fixed-k3", candidate_root, candidate_commit, "control"
        ),
        "full-adaptive": LaneSpec(
            "full-adaptive",
            candidate_root,
            candidate_commit,
            adaptive_route,
        ),
        "full-q4-adaptive": LaneSpec(
            "full-q4-adaptive",
            candidate_root,
            candidate_commit,
            q4_adaptive_route,
        ),
    }


def order_for_context(context_tokens: int) -> tuple[str, ...]:
    if context_tokens not in CONTEXT_TOKENS:
        raise ValueError(f"unsupported context size: {context_tokens}")
    return ONE_PASS_ORDER if context_tokens == 131_072 else PAIRED_ORDER


def _workload_values(workload: str) -> tuple[int, float, float, int]:
    if workload == "low":
        return LOW_OUTPUT_TOKENS, 1.0, 0.95, 20
    if workload == "xhigh":
        return XHIGH_OUTPUT_TOKENS, 1.0, 0.95, 20
    if workload == "vanity":
        return LOW_OUTPUT_TOKENS, VANITY_TEMPERATURE, 1.0, 0
    raise ValueError(f"unknown workload: {workload}")


def child_command(
    *,
    lane: LaneSpec,
    workload: str,
    context_tokens: int,
    output: Path,
    model: Path,
    prompt_file: Path,
    context_file: Path,
    row28_artifact: Path,
    python: Path,
    lock: Path,
) -> list[str]:
    output_tokens, temperature, top_p, top_k = _workload_values(workload)
    command = [
        str(python.absolute()),
        str(ARM_SCRIPT),
        "--source-root", str(lane.source_root.resolve()),
        "--source-commit", lane.source_commit,
        "--lane-id", lane.lane_id,
        "--route", lane.route_id,
        "--workload", workload,
        "--model", str(model.resolve()),
        "--prompt-file", str(prompt_file.resolve()),
        "--context-file", str(context_file.resolve()),
        "--prompt-tokens", str(context_tokens),
        "--max-tokens", str(output_tokens),
        "--warmup-tokens", str(CONDITIONER_OUTPUT_TOKENS),
        "--seed", "42",
        "--target-temperature", str(temperature),
        "--draft-temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--row28-artifact", str(row28_artifact.resolve()),
        "--lock", str(lock.resolve()),
        "--output", str(output.resolve()),
    ]
    return command


def receipt_errors(
    receipt: dict[str, Any],
    *,
    lane: LaneSpec,
    context_tokens: int,
    output_tokens: int,
) -> list[str]:
    errors: list[str] = []
    exact = {
        "lane_id": lane.lane_id,
        "source_commit": lane.source_commit,
        "route_id": lane.route_id,
        "prompt_tokens": context_tokens,
        "conditioner_output_tokens": CONDITIONER_OUTPUT_TOKENS,
        "max_tokens": output_tokens,
        "mlx_version": REQUIRED_MLX_VERSION,
        "mlx_metal_version": REQUIRED_MLX_METAL_VERSION,
        "gpu_lock_scope": "attested_parent",
        "source_import_attested": True,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "draft_core": str(gate._route_execution_options(lane.route_id)["draft_core"]),
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            errors.append(f"{key} mismatch: {receipt.get(key)!r} != {expected!r}")
    if receipt.get("source_status"):
        errors.append("source tree is not clean")
    workload = str(receipt.get("workload") or "")
    expected_prompt_hash = PROMPT_TOKEN_SHA256.get(workload, {}).get(context_tokens)
    if receipt.get("prompt_token_sha256") != expected_prompt_hash:
        errors.append("prompt token hash does not match the frozen workload")
    expected_artifact_hash = PROMPT_ARTIFACT_SHA256[
        "vanity" if workload == "vanity" else "python"
    ]
    if receipt.get("prompt_artifact_sha256") != expected_artifact_hash:
        errors.append("prompt artifact hash does not match the frozen workload")
    if receipt.get("context_artifact_sha256") != PYTHON_CONTEXT_SHA256:
        errors.append("Python context artifact hash does not match the frozen workload")
    if receipt.get("row28_artifact_sha256") != ROW28_ARTIFACT_SHA256:
        errors.append("row28 artifact hash does not match the frozen custom head")
    expected_sampler = {
        "target_temperature": _workload_values(receipt.get("workload", ""))[1],
        "draft_temperature": _workload_values(receipt.get("workload", ""))[1],
        "top_p": _workload_values(receipt.get("workload", ""))[2],
        "top_k": _workload_values(receipt.get("workload", ""))[3],
    }
    if receipt.get("sampler") != expected_sampler:
        errors.append("sampler receipt does not match the frozen workload")
    optimized_stack = receipt.get("optimized_stack") or {}
    expected_stack = {
        "profile": "turbo",
        "runtime_profile": "native_mtp_turbo",
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
        "draft_sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }
    for key, expected in expected_stack.items():
        if optimized_stack.get(key) != expected:
            errors.append(f"optimized stack {key} mismatch")
    required_runtime_env = {
        "MTPLX_COMPILED_VERIFY": "1",
        "MTPLX_DROP_EVENTS": "1",
        "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
    }
    runtime_env = optimized_stack.get("runtime_env") or {}
    for key, expected in required_runtime_env.items():
        if runtime_env.get(key) != expected:
            errors.append(f"optimized stack runtime env {key} mismatch")
    if receipt.get("workload") != "vanity" and int(receipt.get("generated_tokens", -1)) != output_tokens:
        errors.append("timed output token count is not exact")
    if lane.route_id != "control":
        expected_options = gate._route_execution_options(lane.route_id)
        if tuple(receipt.get("source_rows") or ()) != tuple(
            expected_options["source_rows"]
        ):
            errors.append("optimized source-row receipt mismatch")
        if not receipt.get("kernel_ids"):
            errors.append("optimized route reported no installed kernels")
        errors.extend(gate._candidate_engagement_errors(lane.route_id, [], [receipt]))
        try:
            expected_usage = depth_usage(
                generated_tokens=int(receipt["generated_tokens"]),
                drafted_by_depth=list(receipt["drafted_by_depth"]),
                accepted_by_depth=list(receipt["accepted_by_depth"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"adaptive depth usage is invalid: {exc}")
        else:
            if receipt.get("depth_usage") != expected_usage:
                errors.append("adaptive depth usage does not match raw histograms")
        policy = receipt.get("adaptive_policy_receipt") or {}
        if not (
            policy.get("kind") == "position_ema"
            and policy.get("executed") is True
            and len(policy.get("initial_accept_ema") or ()) == 3
            and len(policy.get("final_accept_ema") or ()) == 3
            and 0 <= int(policy.get("initial_depth", -1)) <= 3
            and 0 <= int(policy.get("final_depth", -1)) <= 3
            and int(policy.get("max_depth", -1)) == 3
            and int(policy.get("depth_cap", -1)) == 3
        ):
            errors.append("adaptive policy state receipt is incomplete")
        expected_draft_core = str(
            gate._route_execution_options(lane.route_id)["draft_core"]
        )
        device_core = receipt.get("device_core_receipt") or {}
        if expected_draft_core == "device" and not (
            device_core.get("requested") == "device"
            and int(device_core.get("device_calls", 0)) > 0
            and int(device_core.get("device_fallbacks", -1)) == 0
        ):
            errors.append("adaptive device draft core did not engage without fallback")
        compiled_verify = receipt.get("compiled_verify_receipt") or {}
        exception_fallbacks = sum(
            int(value)
            for reason, value in (compiled_verify.get("fallback_reasons") or {}).items()
            if str(reason).startswith("exception:")
        )
        if not (
            compiled_verify.get("mode") == "on"
            and int(compiled_verify.get("compiled_calls", 0)) > 0
            and compiled_verify.get("permanent_eager") is False
            and not compiled_verify.get("permanent_eager_reason")
            and exception_fallbacks == 0
        ):
            errors.append("adaptive compiled verification did not engage cleanly")
    elif receipt.get("source_rows") or receipt.get("kernel_ids"):
        errors.append("control route unexpectedly installed candidate features")
    return errors


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"invalid {key} values")
    return statistics.fmean(values)


def depth_usage(
    *,
    generated_tokens: int,
    drafted_by_depth: list[int],
    accepted_by_depth: list[int],
) -> dict[str, Any]:
    drafted = ([int(value) for value in drafted_by_depth] + [0, 0, 0])[:3]
    accepted = ([int(value) for value in accepted_by_depth] + [0, 0, 0])[:3]
    cycles = int(generated_tokens) - sum(accepted)
    if not (cycles >= drafted[0] >= drafted[1] >= drafted[2] >= 0):
        raise ValueError("drafted-depth histogram contradicts generated work")
    if not (
        cycles >= accepted[0] >= accepted[1] >= accepted[2] >= 0
        and all(left <= right for left, right in zip(accepted, drafted))
    ):
        raise ValueError("accepted-depth histogram contradicts drafted work")
    attempted_exact = (
        cycles - drafted[0],
        drafted[0] - drafted[1],
        drafted[1] - drafted[2],
        drafted[2],
    )
    accepted_exact = (
        cycles - accepted[0],
        accepted[0] - accepted[1],
        accepted[1] - accepted[2],
        accepted[2],
    )

    def keyed(values: tuple[int, int, int, int]) -> dict[str, int]:
        return {f"D{depth}": value for depth, value in enumerate(values)}

    def shares(values: tuple[int, int, int, int]) -> dict[str, float]:
        return {
            f"D{depth}": value / cycles * 100.0 if cycles else 0.0
            for depth, value in enumerate(values)
        }

    return {
        "unit": "speculative_decode_cycles",
        "decode_cycles": cycles,
        "attempted_tokens_by_position": {
            f"D{depth + 1}": drafted[depth] for depth in range(3)
        },
        "accepted_tokens_by_position": {
            f"D{depth + 1}": accepted[depth] for depth in range(3)
        },
        "acceptance_rate_pct_by_position": {
            f"D{depth + 1}": (
                accepted[depth] / drafted[depth] * 100.0 if drafted[depth] else 0.0
            )
            for depth in range(3)
        },
        "attempted_counts": keyed(attempted_exact),
        "attempted_share_pct": shares(attempted_exact),
        "accepted_counts": keyed(accepted_exact),
        "accepted_share_pct": shares(accepted_exact),
        "mean_attempted_depth": (
            sum(depth * value for depth, value in enumerate(attempted_exact)) / cycles
            if cycles else 0.0
        ),
        "mean_accepted_depth": (
            sum(depth * value for depth, value in enumerate(accepted_exact)) / cycles
            if cycles else 0.0
        ),
    }


def _aggregate_depth_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    drafted = [0, 0, 0]
    accepted = [0, 0, 0]
    for row in rows:
        for index, value in enumerate((row.get("drafted_by_depth") or ())[:3]):
            drafted[index] += int(value)
        for index, value in enumerate((row.get("accepted_by_depth") or ())[:3]):
            accepted[index] += int(value)
    return depth_usage(
        generated_tokens=sum(int(row["generated_tokens"]) for row in rows),
        drafted_by_depth=drafted,
        accepted_by_depth=accepted,
    )


def aggregate(
    *,
    workload: str,
    context_tokens: int,
    order: tuple[str, ...],
    receipts: list[dict[str, Any]],
    specs: dict[str, LaneSpec],
) -> dict[str, Any]:
    output_tokens, temperature, top_p, top_k = _workload_values(workload)
    errors: list[str] = []
    if len(receipts) != len(order):
        errors.append(f"expected {len(order)} arms, found {len(receipts)}")
    for index, (lane_id, receipt) in enumerate(zip(order, receipts)):
        errors.extend(
            f"arm {index}: {error}"
            for error in receipt_errors(
                receipt,
                lane=specs[lane_id],
                context_tokens=context_tokens,
                output_tokens=output_tokens,
            )
        )
    for key in (
        "prompt_token_sha256",
        "prompt_artifact_sha256",
        "context_artifact_sha256",
        "model_artifact_hashes",
        "row28_artifact_sha256",
    ):
        values = {json.dumps(receipt.get(key), sort_keys=True) for receipt in receipts}
        if len(values) != 1:
            errors.append(f"{key} changed across arms")

    summary: dict[str, dict[str, Any]] = {}
    fixed_rows = [row for row in receipts if row.get("lane_id") == "fixed-k3"]
    fixed_wall = _mean(fixed_rows, "wall_s") if fixed_rows else math.nan
    for lane_id in LANE_IDS:
        rows = [row for row in receipts if row.get("lane_id") == lane_id]
        expected_arms = order.count(lane_id)
        if len(rows) != expected_arms:
            errors.append(f"{lane_id} has {len(rows)} arms, expected {expected_arms}")
            continue
        wall = _mean(rows, "wall_s")
        usage = _aggregate_depth_usage(rows)
        summary[lane_id] = {
            "arms": len(rows),
            "source_commit": specs[lane_id].source_commit,
            "route_id": specs[lane_id].route_id,
            "prefill_tok_s_mean": _mean(rows, "prefill_tok_s"),
            "decode_tok_s_mean": _mean(rows, "decode_tok_s"),
            "wall_s_mean": wall,
            "wall_faster_vs_fixed_k3_pct": (fixed_wall / wall - 1.0) * 100.0,
            "peak_memory_gib_max": max(float(row["peak_memory_gib"]) for row in rows),
            "per_lane_token_deterministic": (
                len({row["token_hash"] for row in rows}) == 1
                if len(rows) > 1
                else None
            ),
            "depth_usage": usage,
        }
    return {
        "schema_version": 1,
        "kind": "qwen38_native_mtp_four_lane_matrix",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": workload,
        "context_tokens": context_tokens,
        "conditioner_output_tokens": CONDITIONER_OUTPUT_TOKENS,
        "timed_output_tokens": output_tokens,
        "order": list(order),
        "sampler": {
            "temperature": temperature,
            "draft_temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": 42,
        },
        "software": {"mlx": REQUIRED_MLX_VERSION, "mlx_metal": REQUIRED_MLX_METAL_VERSION},
        "invariant_errors": errors,
        "summary": summary,
        "arms": receipts,
    }


def _load_isolated() -> Any:
    spec = importlib.util.spec_from_file_location("qwen38_matrix_isolated", ISOLATED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ISOLATED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_parent_guard_scope(scope: str) -> str:
    if scope not in {"direct", "attested_parent"}:
        raise RuntimeError(f"matrix parent has invalid execution guard scope: {scope}")
    return scope


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _git_status(root: Path) -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).splitlines()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _interpreter_versions(python: Path) -> dict[str, str]:
    program = (
        "import importlib.metadata,json;"
        "print(json.dumps({'mlx':importlib.metadata.version('mlx'),"
        "'mlx_metal':importlib.metadata.version('mlx-metal')},sort_keys=True))"
    )
    output = subprocess.check_output(
        [str(python.absolute()), "-c", program], text=True
    ).strip()
    return {str(key): str(value) for key, value in json.loads(output).items()}


def _assert_campaign_inputs(args: argparse.Namespace) -> None:
    for label, root in (
        ("baseline", args.baseline_root),
        ("candidate", args.candidate_root),
    ):
        if _git_status(root):
            raise RuntimeError(f"{label} source tree must be clean before the campaign")
    output_root = args.output_root.resolve()
    for label, root in (
        ("baseline", args.baseline_root.resolve()),
        ("candidate", args.candidate_root.resolve()),
    ):
        if output_root == root or output_root.is_relative_to(root):
            raise RuntimeError(
                f"output root must be outside the {label} source tree: {output_root}"
            )
    expected_prompt = VANITY_PROMPT_FILE if args.workload == "vanity" else PYTHON_PROMPT_FILE
    if args.prompt_file.resolve() != expected_prompt.resolve():
        raise RuntimeError(
            f"{args.workload} requires frozen prompt artifact {expected_prompt}"
        )
    if args.context_file.resolve() != PYTHON_CONTEXT_FILE.resolve():
        raise RuntimeError(f"matrix requires frozen Python context {PYTHON_CONTEXT_FILE}")
    expected_hashes = {
        args.prompt_file: PROMPT_ARTIFACT_SHA256[
            "vanity" if args.workload == "vanity" else "python"
        ],
        args.context_file: PYTHON_CONTEXT_SHA256,
        args.row28_artifact: ROW28_ARTIFACT_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"frozen artifact hash mismatch for {path.name}: "
                f"{observed_hash} != {expected_hash}"
            )


def run(args: argparse.Namespace) -> int:
    _assert_campaign_inputs(args)
    observed_versions = _interpreter_versions(args.python)
    required_versions = {
        "mlx": REQUIRED_MLX_VERSION,
        "mlx_metal": REQUIRED_MLX_METAL_VERSION,
    }
    if observed_versions != required_versions:
        raise RuntimeError(
            f"benchmark interpreter versions mismatch: {observed_versions} "
            f"!= {required_versions}"
        )
    candidate_commit = _git_commit(args.candidate_root)
    baseline_commit = _git_commit(args.baseline_root)
    if baseline_commit != V292_COMMIT:
        raise RuntimeError(
            f"baseline must be exact v2.9.2: {baseline_commit} != {V292_COMMIT}"
        )
    specs = lane_specs(
        baseline_root=args.baseline_root,
        baseline_commit=baseline_commit,
        candidate_root=args.candidate_root,
        candidate_commit=candidate_commit,
        workload=args.workload,
    )
    contexts = (
        (VANITY_PROMPT_TOKENS,)
        if args.workload == "vanity"
        else tuple(args.contexts)
    )
    isolated = _load_isolated()
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    with isolated._gpu_lock_scope(args.lock) as lock_scope:
        _validated_parent_guard_scope(lock_scope)
        model_hashes = gate._model_artifact_hashes(args.model)
        for context_tokens in contexts:
            order = PAIRED_ORDER if args.workload == "vanity" else order_for_context(context_tokens)
            context_root = args.output_root / f"{args.workload}-{context_tokens}"
            context_root.mkdir(parents=True, exist_ok=True)
            receipts: list[dict[str, Any]] = []
            for index, lane_id in enumerate(order):
                lane = specs[lane_id]
                output = context_root / f"arm-{index}-{lane_id}.json"
                command = child_command(
                    lane=lane,
                    workload=args.workload,
                    context_tokens=context_tokens,
                    output=output,
                    model=args.model,
                    prompt_file=args.prompt_file,
                    context_file=args.context_file,
                    row28_artifact=args.row28_artifact,
                    python=args.python,
                    lock=args.lock,
                )
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(lane.source_root)
                environment[MODEL_HASHES_ENV] = json.dumps(
                    model_hashes, sort_keys=True, separators=(",", ":")
                )
                result = isolated._run_attested_child(
                    command,
                    environment=isolated._environment_for_route(lane.route_id, environment),
                    lock_path=args.lock,
                    owns_process_group=True,
                )
                log = output.with_suffix(".log")
                log.write_text(result.stdout or "", encoding="utf-8")
                if result.returncode != 0 or not output.is_file():
                    raise RuntimeError(
                        f"{args.workload} {context_tokens} arm {index} failed; see {log}"
                    )
                receipt = json.loads(output.read_text(encoding="utf-8"))
                receipts.append(receipt)
                print(json.dumps({
                    "event": "arm_complete",
                    "workload": args.workload,
                    "context_tokens": context_tokens,
                    "arm": index + 1,
                    "lane": lane_id,
                    "wall_s": receipt["wall_s"],
                }), flush=True)
            combined = aggregate(
                workload=args.workload,
                context_tokens=context_tokens,
                order=order,
                receipts=receipts,
                specs=specs,
            )
            combined_path = context_root / "combined.json"
            combined_path.write_text(
                json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if combined["invariant_errors"]:
                raise RuntimeError(
                    f"{args.workload} {context_tokens} invariant errors: "
                    f"{combined['invariant_errors']}"
                )
            completed.append({
                "workload": args.workload,
                "context_tokens": context_tokens,
                "receipt_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
                "summary": combined["summary"],
            })
            (args.output_root / "index.json").write_text(
                json.dumps({
                    "kind": "qwen38_native_mtp_four_lane_campaign",
                    "completed": completed,
                }, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("vanity", "low", "xhigh"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXT_TOKENS))
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--row28-artifact", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.workload != "vanity" and any(value not in CONTEXT_TOKENS for value in args.contexts):
        raise ValueError(f"contexts must be selected from {CONTEXT_TOKENS}")
    return args


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
