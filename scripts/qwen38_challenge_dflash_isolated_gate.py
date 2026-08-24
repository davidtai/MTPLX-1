#!/usr/bin/env python3
"""Four-process ABBA gate for the Qwen 3.8 DFlash2 replacement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen38_challenge_dflash_gate as arm_gate  # noqa: E402
from scripts.qwen38_challenge_port_isolated_gate import (  # noqa: E402
    _gpu_lock_scope,
    _run_attested_child,
)


ORDER = ("mtp_fixed_d3", "dflash2", "dflash2", "mtp_fixed_d3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=arm_gate.DEFAULT_MODEL)
    parser.add_argument("--draft", type=Path, default=arm_gate.DEFAULT_DFLASH_SNAPSHOT)
    parser.add_argument(
        "--row36-artifact",
        type=Path,
        default=arm_gate.DEFAULT_ROW36_ARTIFACT,
    )
    parser.add_argument("--prompt-file", type=Path, default=arm_gate.DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=arm_gate.DEFAULT_CONTEXT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, default=arm_gate.DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _child_command(
    args: argparse.Namespace,
    *,
    engine: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/qwen38_challenge_dflash_gate.py"),
        "--engine",
        engine,
        "--model",
        str(args.model),
        "--draft",
        str(args.draft),
        "--row36-artifact",
        str(args.row36_artifact),
        "--prompt-file",
        str(args.prompt_file),
        "--context-file",
        str(args.context_file),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--seed",
        str(args.seed),
        "--lock",
        str(args.lock),
        "--output",
        str(output),
    ]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _aggregate(
    args: argparse.Namespace,
    child_receipts: list[dict[str, Any]],
    *,
    lock_scope: str,
) -> dict[str, Any]:
    arms = [receipt["arm"] for receipt in child_receipts]
    warmups = [receipt["warmup"] for receipt in child_receipts]
    by_engine = {
        engine: [arm for arm in arms if arm["engine"] == engine]
        for engine in ("mtp_fixed_d3", "dflash2")
    }
    deterministic = {
        engine: len({arm["token_hash"] for arm in rows}) == 1
        for engine, rows in by_engine.items()
    }
    generated_exact = all(
        int(arm["generated_tokens"]) == args.max_tokens for arm in arms
    )
    dflash_contract = all(
        int(arm["requested_width"]) == arm_gate.STATIC_WIDTH
        and int(arm["effective_width"]) == arm_gate.STATIC_WIDTH
        and not bool(arm["fallback_ar"])
        for arm in by_engine["dflash2"]
    )
    mean_wall = {
        engine: _mean(rows, "wall_s") for engine, rows in by_engine.items()
    }
    improvement_pct = (
        mean_wall["mtp_fixed_d3"] / mean_wall["dflash2"] - 1.0
    ) * 100.0
    summary = {
        engine: {
            "prefill_tps": _mean(rows, "prefill_tps"),
            "decode_tps": _mean(rows, "decode_tps"),
            "peak_memory_gb": _mean(rows, "peak_memory_gb"),
            "wall_s": mean_wall[engine],
        }
        for engine, rows in by_engine.items()
    }
    exact = bool(generated_exact and dflash_contract and all(deterministic.values()))
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    promoted = bool(
        exact
        and not source_status
        and improvement_pct > arm_gate.PROMOTION_THRESHOLD_PCT
    )
    first = child_receipts[0]
    return {
        "kind": "qwen38_challenge_dflash2_item55_isolated_abba",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": first["model"],
        "dflash": first["dflash"],
        "workload": {
            **first["workload"],
            "timed_order": list(ORDER),
            "timed_arm_count": 4,
        },
        "optimized_speed_stack": first["optimized_speed_stack"],
        "retained_route": first["retained_route"],
        "mlx_version": first["mlx_version"],
        "dflash_mlx_version": first["dflash_mlx_version"],
        "python": first["python"],
        "platform": first["platform"],
        "git_commit": first["git_commit"],
        "isolation_reason": (
            "DFlash construction and retained MTP tensor-offset cache cannot "
            "coexist safely in one MLX process"
        ),
        "conditioning_scope": (
            "one 1024-token conditioner in each isolated timed-arm process"
        ),
        "gpu_lock_scope": lock_scope,
        "warmups": warmups,
        "arms": arms,
        "summary": summary,
        "correctness": {
            "per_engine_deterministic": deterministic,
            "generated_count_exact": generated_exact,
            "dflash_width_and_fallback_exact": dflash_contract,
            "cross_engine_token_exact": (
                by_engine["mtp_fixed_d3"][0]["token_hash"]
                == by_engine["dflash2"][0]["token_hash"]
            ),
            "cross_engine_token_exact_required": False,
            "exact": exact,
        },
        "source_status": source_status,
        "candidate_improvement_pct": improvement_pct,
        "promotion": {
            "threshold_pct": arm_gate.PROMOTION_THRESHOLD_PCT,
            "passed": promoted,
            "reason": (
                "strict wall improvement above threshold"
                if promoted
                else "correctness, clean-source, or strict wall threshold failed"
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    if args.prompt_tokens != 16_384 or args.max_tokens != 1024:
        raise ValueError("item 55 requires exactly 16K input and 1024 output tokens")
    child_receipts: list[dict[str, Any]] = []
    with _gpu_lock_scope(args.lock) as lock_scope:
        with tempfile.TemporaryDirectory(prefix="qwen38-dflash-item55-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, engine in enumerate(ORDER):
                child_output = temp_root / f"arm-{index}.json"
                result = _run_attested_child(
                    _child_command(
                        args,
                        engine=engine,
                        output=child_output,
                    ),
                    environment=os.environ,
                    lock_path=args.lock,
                )
                if result.returncode != 0 or not child_output.is_file():
                    raise RuntimeError(
                        f"isolated {engine} arm {index} failed "
                        f"({result.returncode}):\n{result.stdout}"
                    )
                child_receipts.append(
                    json.loads(child_output.read_text(encoding="utf-8"))
                )

    receipt = _aggregate(args, child_receipts, lock_scope=lock_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    print(f"candidate_improvement_pct={receipt['candidate_improvement_pct']:.6f}")
    print(f"promotion_passed={receipt['promotion']['passed']}")
    return 0 if receipt["correctness"]["exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
