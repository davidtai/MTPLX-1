#!/usr/bin/env python3
"""Matched main-vs-PR Qwen3.8 cold-prefill benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, NamedTuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL = Path.home() / ".mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
DEFAULT_DRAFT = Path.home() / (
    ".cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/"
    "snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
)
DEFAULT_MAIN_ROOT = Path("/tmp/mtplx-qwen38-main-control.1Z0Lm7/main")
DEFAULT_CONTEXT = ROOT / "mtplx/generation.py"
DEFAULT_PROMPT = ROOT / "mtplx/benchmarks/prompts/python_modules_long.jsonl"
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
ORDER = ("main_native_mtp", "pr_dflash2", "pr_dflash2", "main_native_mtp")


class Scenario(NamedTuple):
    name: str
    prompt_tokens: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    prompt_kind: str


SCENARIOS = (
    Scenario("burst_is_palindrome", 101, 1_024, 0.0, 1.0, 0, "is_palindrome"),
    Scenario("coding_cold_prefill_1k", 1_024, 1_024, 0.6, 0.95, 20, "coding"),
    Scenario("coding_cold_prefill_16k", 16_384, 1_024, 0.6, 0.95, 20, "coding"),
    Scenario("coding_cold_prefill_64k", 65_536, 1_024, 0.6, 0.95, 20, "coding"),
    Scenario("coding_cold_prefill_128k", 131_072, 1_024, 0.6, 0.95, 20, "coding"),
)


def build_exact_coding_prompt(
    tokenizer: Any,
    *,
    target_tokens: int,
    context: str,
    instruction: str,
) -> tuple[str, list[int]]:
    """Fill the whole cold-prefill budget with code context and one task tail."""

    if target_tokens <= 0:
        raise ValueError("prompt token target must be positive")
    tail_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
    if len(tail_ids) >= target_tokens:
        raise ValueError("instruction does not fit inside prompt token target")
    context_ids = list(tokenizer.encode(context.rstrip() + "\n"))
    if not context_ids:
        raise ValueError("context must encode to at least one token")
    context_budget = target_tokens - len(tail_ids)
    repeats = (context_budget + len(context_ids) - 1) // len(context_ids)
    token_ids = (context_ids * repeats)[:context_budget] + tail_ids
    return str(tokenizer.decode(token_ids)), token_ids


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def child_command(
    args: argparse.Namespace,
    *,
    engine: str,
    source_root: Path,
    source_commit: str,
    scenario: Scenario,
    output: Path,
) -> list[str]:
    headline = scenario.prompt_kind == "is_palindrome"
    command = [
        sys.executable,
        str(ROOT / "scripts/qwen38_final_benchmark_arm.py"),
        "--engine", engine,
        "--source-root", str(source_root),
        "--source-commit", source_commit,
        "--model", str(args.model),
        "--draft", str(args.draft),
        "--prompt-file", str(args.prompt_file),
        "--context-file", str(args.context_file),
        "--prompt-kind", scenario.prompt_kind,
        "--prompt-tokens", str(scenario.prompt_tokens),
        "--max-tokens", str(scenario.max_tokens),
        "--temperature", str(scenario.temperature),
        "--top-p", str(scenario.top_p),
        "--top-k", str(scenario.top_k),
        "--conditioner-tokens", str(scenario.max_tokens if headline else args.conditioner_tokens),
        "--conditioner-mode", "same_prompt" if headline else "unrelated_prompt",
        "--seed", str(args.seed),
        "--lock", str(args.lock),
        "--output", str(output),
    ]
    if engine == "pr_dflash2":
        command.append("--dflash2-adaptive")
    return command


def aggregate_scenario(
    scenario: Scenario,
    child_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    arms = [dict(receipt["arm"]) for receipt in child_receipts]
    by_engine = {
        engine: [arm for arm in arms if arm["engine"] == engine]
        for engine in ORDER[:2]
    }
    summary = {
        engine: {
            key: _mean(rows, key)
            for key in (
                "prefill_tps",
                "decode_tps",
                "wall_s",
                "prefill_s",
                "decode_elapsed_s",
                "peak_memory_gib",
                "generated_tokens",
            )
        }
        for engine, rows in by_engine.items()
    }
    control_wall = summary["main_native_mtp"]["wall_s"]
    candidate_wall = summary["pr_dflash2"]["wall_s"]
    output_limit_respected = all(
        0 < int(arm["generated_tokens"]) <= scenario.max_tokens for arm in arms
    )
    exact_counts = all(
        int(arm["prompt_tokens"]) == scenario.prompt_tokens for arm in arms
    ) and (
        output_limit_respected
        if scenario.prompt_kind == "is_palindrome"
        else all(int(arm["generated_tokens"]) == scenario.max_tokens for arm in arms)
    )
    no_prefix_cache = all(
        not bool(arm.get("prefix_cache_used", False))
        and int(arm.get("cached_tokens", 0)) == 0
        for arm in arms
    )
    dflash_adaptive = {
        "requested": all(
            bool(arm.get("requested_adaptive"))
            for arm in by_engine["pr_dflash2"]
        ),
        "effective": all(
            bool(arm.get("effective_adaptive"))
            for arm in by_engine["pr_dflash2"]
        ),
        "observed_widths": sorted(
            {
                int(width)
                for arm in by_engine["pr_dflash2"]
                for width in arm.get("effective_widths", [])
            }
        ),
    }
    return {
        "scenario": scenario.name,
        "prompt_kind": scenario.prompt_kind,
        "prompt_tokens": scenario.prompt_tokens,
        "generated_tokens": scenario.max_tokens,
        "temperature": scenario.temperature,
        "top_p": scenario.top_p,
        "top_k": scenario.top_k,
        "cold_prefill": True,
        "prefix_cache_used": not no_prefix_cache,
        "timed_order": list(ORDER),
        "timed_arm_count": len(arms),
        "summary": summary,
        "wall_time_improvement_pct": (control_wall / candidate_wall - 1.0) * 100.0,
        "dflash_adaptive": dflash_adaptive,
        "correctness": {
            "exact_prompt_and_output_counts": exact_counts,
            "output_limit_respected": output_limit_respected,
            "no_prefix_cache_or_session_restore": no_prefix_cache,
            "per_engine_token_replay": {
                engine: len({row["token_sha256"] for row in rows}) == 1
                for engine, rows in by_engine.items()
            },
            "dflash_no_ar_fallback": all(
                not bool(row.get("fallback_ar", False))
                for row in by_engine["pr_dflash2"]
            ),
            "dflash_adaptive_effective": dflash_adaptive["effective"],
        },
        "arms": arms,
        "child_receipts": child_receipts,
    }


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--pr-root", type=Path, default=ROOT)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--conditioner-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    main_commit = _git_commit(args.main_root)
    pr_commit = _git_commit(args.pr_root)
    if subprocess.check_output(["git", "status", "--short"], cwd=args.main_root, text=True).strip():
        raise RuntimeError("main control worktree is dirty")

    from scripts.qwen38_challenge_port_isolated_gate import (
        _gpu_lock_scope,
        _run_attested_child,
    )

    results: list[dict[str, Any]] = []
    with _gpu_lock_scope(args.lock) as lock_scope:
        with tempfile.TemporaryDirectory(prefix="qwen38-final-matrix-") as temp:
            temp_root = Path(temp)
            for scenario in selected:
                receipts: list[dict[str, Any]] = []
                for index, engine in enumerate(ORDER):
                    source_root = args.main_root if engine == "main_native_mtp" else args.pr_root
                    source_commit = main_commit if engine == "main_native_mtp" else pr_commit
                    output = temp_root / f"{scenario.name}-{index}.json"
                    command = child_command(
                        args,
                        engine=engine,
                        source_root=source_root,
                        source_commit=source_commit,
                        scenario=scenario,
                        output=output,
                    )
                    environment = dict(os.environ)
                    environment["MLX_MAX_MB_PER_BUFFER"] = "512"
                    environment["MLX_MAX_OPS_PER_BUFFER"] = "50"
                    result = _run_attested_child(
                        command, environment=environment, lock_path=args.lock
                    )
                    if result.returncode != 0 or not output.is_file():
                        raise RuntimeError(
                            f"{scenario.name} {engine} arm {index} failed "
                            f"({result.returncode}):\n{result.stdout}"
                        )
                    print(result.stdout, end="", flush=True)
                    receipts.append(json.loads(output.read_text(encoding="utf-8")))
                scenario_result = aggregate_scenario(scenario, receipts)
                results.append(scenario_result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                partial = {
                    "kind": "qwen38_final_main_pr_cold_prefill_matrix",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "main_commit": main_commit,
                    "pr_commit": pr_commit,
                    "gpu_lock_scope": lock_scope,
                    "cold_prefill_definition": (
                        "fresh prompt/KV state; exact total prompt length; no prefix cache "
                        "or session restore; same-prompt decode conditioner for the burst "
                        "and unrelated short conditioner for coding loads"
                    ),
                    "results": results,
                }
                args.output.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
                print(json.dumps(scenario_result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
