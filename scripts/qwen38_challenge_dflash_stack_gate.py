#!/usr/bin/env python3
"""Four-process ABBA gate for one cumulative DFlash2 stack improvement."""

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

ORDER = ("control", "candidate", "candidate", "control")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--control-survivors", default="")
    parser.add_argument("--candidate-survivors", default="")
    parser.add_argument("--control-adaptive-rows", default="")
    parser.add_argument("--candidate-adaptive-rows", default="")
    parser.add_argument("--control-custom-rows", default="")
    parser.add_argument("--candidate-custom-rows", default="")
    parser.add_argument("--control-gqa-widths", default="")
    parser.add_argument("--candidate-gqa-widths", default="")
    parser.add_argument("--control-m8-nax-island", action="store_true")
    parser.add_argument("--candidate-m8-nax-island", action="store_true")
    parser.add_argument("--control-m8-linear-z", action="store_true")
    parser.add_argument("--candidate-m8-linear-z", action="store_true")
    parser.add_argument("--control-m7-nax-output", action="store_true")
    parser.add_argument("--candidate-m7-nax-output", action="store_true")
    parser.add_argument("--control-cost-aligned-widths", action="store_true")
    parser.add_argument("--candidate-cost-aligned-widths", action="store_true")
    parser.add_argument("--control-release-native-mtp", action="store_true")
    parser.add_argument("--candidate-release-native-mtp", action="store_true")
    parser.add_argument("--control-max-mb-per-buffer", type=int, default=512)
    parser.add_argument("--candidate-max-mb-per-buffer", type=int, default=512)
    parser.add_argument("--control-max-ops-per-buffer", type=int, default=50)
    parser.add_argument("--candidate-max-ops-per-buffer", type=int, default=50)
    parser.add_argument("--model", type=Path, default=arm_gate.DEFAULT_MODEL)
    parser.add_argument("--draft", type=Path, default=arm_gate.DEFAULT_DFLASH_SNAPSHOT)
    parser.add_argument("--prompt-file", type=Path, default=arm_gate.DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=arm_gate.DEFAULT_CONTEXT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, default=arm_gate.DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _variant_environment(
    args: argparse.Namespace,
    variant: str,
    base: dict[str, str] | os._Environ[str],
) -> dict[str, str]:
    environment = dict(base)
    prefix = "control" if variant == "control" else "candidate"
    environment["MLX_MAX_MB_PER_BUFFER"] = str(
        int(getattr(args, f"{prefix}_max_mb_per_buffer"))
    )
    environment["MLX_MAX_OPS_PER_BUFFER"] = str(
        int(getattr(args, f"{prefix}_max_ops_per_buffer"))
    )
    return environment


def _variant_config(
    args: argparse.Namespace,
    variant: str,
) -> tuple[str, str, str, str, bool, bool, bool, bool, bool]:
    if variant == "control":
        return (
            args.control_survivors,
            args.control_adaptive_rows,
            args.control_custom_rows,
            args.control_gqa_widths,
            bool(args.control_m8_nax_island),
            bool(args.control_m8_linear_z),
            bool(args.control_m7_nax_output),
            bool(args.control_cost_aligned_widths),
            bool(args.control_release_native_mtp),
        )
    return (
        args.candidate_survivors,
        args.candidate_adaptive_rows,
        args.candidate_custom_rows,
        args.candidate_gqa_widths,
        bool(args.candidate_m8_nax_island),
        bool(args.candidate_m8_linear_z),
        bool(args.candidate_m7_nax_output),
        bool(args.candidate_cost_aligned_widths),
        bool(args.candidate_release_native_mtp),
    )


def _child_command(
    args: argparse.Namespace,
    *,
    variant: str,
    output: Path,
) -> list[str]:
    (
        survivors,
        adaptive_rows,
        custom_rows,
        gqa_widths,
        m8_nax_island,
        m8_linear_z,
        m7_nax_output,
        cost_aligned_widths,
        release_native_mtp,
    ) = _variant_config(args, variant)
    command = [
        sys.executable,
        str(ROOT / "scripts/qwen38_challenge_dflash_gate.py"),
        "--engine",
        "dflash2",
        "--model",
        str(args.model),
        "--draft",
        str(args.draft),
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
        "--dflash-survivors",
        survivors,
        "--dflash-adaptive-rows",
        adaptive_rows,
        "--dflash-custom-rows",
        custom_rows,
        "--dflash-gqa-widths",
        gqa_widths,
        "--lock",
        str(args.lock),
        "--output",
        str(output),
    ]
    if release_native_mtp:
        command.append("--release-native-mtp")
    if cost_aligned_widths:
        command.append("--dflash-cost-aligned-widths")
    if m8_nax_island:
        command.append("--dflash-m8-nax-island")
    if m8_linear_z:
        command.append("--dflash-m8-linear-z")
    if m7_nax_output:
        command.append("--dflash-m7-nax-output")
    return command


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _engagement_exact(
    args: argparse.Namespace,
    by_variant: dict[str, list[dict[str, Any]]],
) -> bool:
    if args.candidate_label.startswith("c") and args.candidate_label[1:].isdigit():
        expected_row = int(args.candidate_label[1:])
        expected_width = {34: 6, 40: 7, 47: 8}.get(expected_row)
        control_rows = arm_gate._parse_dflash_custom_rows(args.control_custom_rows)
        candidate_rows = arm_gate._parse_dflash_custom_rows(args.candidate_custom_rows)
        if expected_width is None or not candidate_rows or candidate_rows[-1] != expected_row:
            return False

        def calls(row: dict[str, Any], width: int) -> int:
            return int(row["engagement"]["r70_qmv_sumtable"][f"m{width}"])

        control_expected = expected_row in control_rows
        return all(
            (calls(row, expected_width) > 0) == control_expected
            for row in by_variant["control"]
        ) and all(
            calls(row, expected_width) > 0 for row in by_variant["candidate"]
        )
    if args.candidate_label.startswith("a"):
        expected_row = int(args.candidate_label[1:])
        control_rows = list(
            arm_gate._parse_dflash_adaptive_rows(args.control_adaptive_rows)
        )
        candidate_rows = list(
            arm_gate._parse_dflash_adaptive_rows(args.candidate_adaptive_rows)
        )
        if not candidate_rows or candidate_rows[-1] != expected_row:
            return False

        def matches(row: dict[str, Any], expected: list[int]) -> bool:
            metrics = row.get("adaptive_metrics", {})
            if not expected:
                return not metrics
            return (
                metrics.get("kind") == "qwen38_position_ema"
                and metrics.get("proposal_rows") == expected
                and int(metrics.get("cycles", 0)) > 0
            )

        return all(matches(row, control_rows) for row in by_variant["control"]) and all(
            matches(row, candidate_rows) for row in by_variant["candidate"]
        )
    if args.candidate_label == "release_native_mtp":
        return all(
            not bool(row["feature_receipt"]["native_mtp_release"]["native_mtp_released"])
            for row in by_variant["control"]
        ) and all(
            bool(row["feature_receipt"]["native_mtp_release"]["native_mtp_released"])
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "r21":
        def calls(row: dict[str, Any]) -> int:
            return int(row["engagement"]["r21_qk_rms_rope"]["calls"])

        return all(calls(row) == 0 for row in by_variant["control"]) and all(
            calls(row) > 0 for row in by_variant["candidate"]
        )
    if args.candidate_label == "r24":
        def counts(row: dict[str, Any]) -> tuple[int, int]:
            engagement = row["engagement"]
            return (
                int(engagement["r24_eval_ladder"]["calls"]),
                int(engagement["r24_qk_length_limit"]["fallback_calls"]),
            )

        return all(counts(row) == (0, 0) for row in by_variant["control"]) and all(
            ladder > 0 and fallback > 0
            for ladder, fallback in map(counts, by_variant["candidate"])
        )
    if args.candidate_label == "r26":
        def calls(row: dict[str, Any]) -> int:
            return int(row["engagement"]["r26_prefill_ladder_3"]["calls"])

        return all(calls(row) == 0 for row in by_variant["control"]) and all(
            calls(row) > 0 for row in by_variant["candidate"]
        )
    if args.candidate_label == "r48":
        def counts(row: dict[str, Any]) -> tuple[int, int]:
            report = row["engagement"]["r48_boundary_fused"]
            return int(report["calls"]), int(report["merged_boundaries"])

        return all(counts(row) == (0, 0) for row in by_variant["control"]) and all(
            calls > 0 and merged > 0
            for calls, merged in map(counts, by_variant["candidate"])
        )
    if args.candidate_label == "gqa678":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row.get("feature_receipt", {}).get("dflash_gqa_widths", {}))

        return all(not route(row) for row in by_variant["control"]) and all(
            bool(route(row).get("active"))
            and route(row).get("widths") == [6, 7, 8]
            and all(
                int(row.get("adaptive_metrics", {}).get("cycles_by_block", {}).get(str(width), 0)) > 0
                for width in (6, 7, 8)
            )
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "cost_aligned":
        def alignment(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("adaptive_metrics", {}).get("cost_alignment", {})
            )

        return all(not alignment(row).get("active") for row in by_variant["control"]) and all(
            bool(alignment(row).get("active"))
            and set(alignment(row).get("promoted_widths", ())) == {"5->6", "7->8"}
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_nax_island":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any]) -> int:
            return int(
                row.get("engagement", {})
                .get("nax_verify", {})
                .get("m8_nax_k6144_n5120", 0)
            )

        return all(
            not route(row) and calls(row) == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("active"))
            and int(route(row).get("width", 0)) == 8
            and route(row).get("shapes") == [[6144, 5120]]
            and int(route(row).get("eligible_attention_modules", 0)) == 16
            and int(route(row).get("validated_projections", 0)) == 32
            and int(route(row).get("eligible_projections", 0)) == 16
            and int(
                row.get("adaptive_metrics", {})
                .get("cycles_by_block", {})
                .get("8", 0)
            )
            > 0
            and calls(row) > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "cb1024":
        def limits(row: dict[str, Any]) -> tuple[int, int]:
            report = row.get("feature_receipt", {}).get(
                "r53_command_buffers", {}
            )
            return (
                int(report.get("max_mb_per_buffer", 0)),
                int(report.get("max_ops_per_buffer", 0)),
            )

        return all(limits(row) == (512, 50) for row in by_variant["control"]) and all(
            limits(row) == (1024, 50) for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_linear_z":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], shape: str) -> int:
            return int(
                row.get("engagement", {})
                .get("nax_verify", {})
                .get(shape, 0)
            )

        return all(
            bool(route(row).get("active"))
            and not bool(route(row).get("include_linear_z"))
            and route(row).get("shapes") == [[6144, 5120]]
            and int(route(row).get("eligible_projections", 0)) == 16
            and calls(row, "m8_nax_k6144_n5120") > 0
            and calls(row, "m8_nax_k5120_n6144") == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("active"))
            and bool(route(row).get("include_linear_z"))
            and route(row).get("shapes") == [[5120, 6144], [6144, 5120]]
            and int(route(row).get("eligible_attention_modules", 0)) == 16
            and int(route(row).get("eligible_linear_z_projections", 0)) == 48
            and int(route(row).get("eligible_projections", 0)) == 64
            and int(
                row.get("adaptive_metrics", {})
                .get("cycles_by_block", {})
                .get("8", 0)
            )
            > 0
            and calls(row, "m8_nax_k6144_n5120") > 0
            and calls(row, "m8_nax_k5120_n6144") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m7_nax_output":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any]) -> int:
            return int(
                row.get("engagement", {})
                .get("nax_verify", {})
                .get("m7_to_m8_nax_k6144_n5120", 0)
            )

        return all(
            not bool(route(row).get("include_m7_output"))
            and route(row).get("m7_shapes", []) == []
            and calls(row) == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("active"))
            and bool(route(row).get("include_m7_output"))
            and route(row).get("m7_shapes") == [[6144, 5120]]
            and int(route(row).get("eligible_m7_projections", 0)) == 16
            and int(
                row.get("adaptive_metrics", {})
                .get("cycles_by_block", {})
                .get("7", 0)
            )
            > 0
            and calls(row) > 0
            for row in by_variant["candidate"]
        )
    return True


def _aggregate(
    args: argparse.Namespace,
    children: list[dict[str, Any]],
    *,
    lock_scope: str,
) -> dict[str, Any]:
    arms = []
    warmups = []
    for variant, child in zip(ORDER, children, strict=True):
        arms.append({**child["arm"], "variant": variant})
        warmups.append({**child["warmup"], "variant": variant})
    by_variant = {
        variant: [arm for arm in arms if arm["variant"] == variant]
        for variant in ("control", "candidate")
    }
    deterministic = {
        variant: len({arm["token_hash"] for arm in rows}) == 1
        for variant, rows in by_variant.items()
    }
    generated_exact = all(
        int(arm["generated_tokens"]) == args.max_tokens for arm in arms
    )
    width_exact = all(
        int(arm["requested_width"]) == arm_gate.STATIC_WIDTH
        and int(arm["effective_width"]) == arm_gate.STATIC_WIDTH
        and not bool(arm["fallback_ar"])
        for arm in arms
    )
    engagement_exact = _engagement_exact(args, by_variant)
    mean_wall = {
        variant: _mean(rows, "wall_s") for variant, rows in by_variant.items()
    }
    improvement_pct = (mean_wall["control"] / mean_wall["candidate"] - 1.0) * 100.0
    summary = {
        variant: {
            "prefill_tps": _mean(rows, "prefill_tps"),
            "decode_tps": _mean(rows, "decode_tps"),
            "peak_memory_gb": max(float(row["peak_memory_gb"]) for row in rows),
            "wall_s": mean_wall[variant],
        }
        for variant, rows in by_variant.items()
    }
    exact = bool(generated_exact and width_exact and engagement_exact and all(deterministic.values()))
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    promoted = bool(
        exact
        and not source_status
        and improvement_pct > arm_gate.PROMOTION_THRESHOLD_PCT
    )
    first = children[0]
    return {
        "kind": "qwen38_challenge_dflash2_cumulative_stack_abba",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_label": args.candidate_label,
        "model": first["model"],
        "dflash": first["dflash"],
        "workload": {
            **first["workload"],
            "timed_order": list(ORDER),
            "timed_arm_count": 4,
        },
        "stack": {
            "control_survivors": list(arm_gate._parse_dflash_survivors(args.control_survivors)),
            "candidate_survivors": list(arm_gate._parse_dflash_survivors(args.candidate_survivors)),
            "control_adaptive_rows": list(
                arm_gate._parse_dflash_adaptive_rows(args.control_adaptive_rows)
            ),
            "candidate_adaptive_rows": list(
                arm_gate._parse_dflash_adaptive_rows(args.candidate_adaptive_rows)
            ),
            "control_custom_rows": list(
                arm_gate._parse_dflash_custom_rows(args.control_custom_rows)
            ),
            "candidate_custom_rows": list(
                arm_gate._parse_dflash_custom_rows(args.candidate_custom_rows)
            ),
            "control_gqa_widths": list(
                arm_gate._parse_dflash_gqa_widths(args.control_gqa_widths)
            ),
            "candidate_gqa_widths": list(
                arm_gate._parse_dflash_gqa_widths(args.candidate_gqa_widths)
            ),
            "control_m8_nax_island": bool(args.control_m8_nax_island),
            "candidate_m8_nax_island": bool(args.candidate_m8_nax_island),
            "control_m8_linear_z": bool(args.control_m8_linear_z),
            "candidate_m8_linear_z": bool(args.candidate_m8_linear_z),
            "control_m7_nax_output": bool(args.control_m7_nax_output),
            "candidate_m7_nax_output": bool(args.candidate_m7_nax_output),
            "control_cost_aligned_widths": bool(args.control_cost_aligned_widths),
            "candidate_cost_aligned_widths": bool(args.candidate_cost_aligned_widths),
            "control_release_native_mtp": bool(args.control_release_native_mtp),
            "candidate_release_native_mtp": bool(args.candidate_release_native_mtp),
            "control_command_buffers": {
                "max_mb_per_buffer": int(args.control_max_mb_per_buffer),
                "max_ops_per_buffer": int(args.control_max_ops_per_buffer),
            },
            "candidate_command_buffers": {
                "max_mb_per_buffer": int(args.candidate_max_mb_per_buffer),
                "max_ops_per_buffer": int(args.candidate_max_ops_per_buffer),
            },
        },
        "mlx_version": first["mlx_version"],
        "dflash_mlx_version": first["dflash_mlx_version"],
        "git_commit": first["git_commit"],
        "gpu_lock_scope": lock_scope,
        "warmups": warmups,
        "arms": arms,
        "summary": summary,
        "correctness": {
            "per_variant_deterministic": deterministic,
            "generated_count_exact": generated_exact,
            "dflash_width_and_fallback_exact": width_exact,
            "candidate_engagement_exact": engagement_exact,
            "cross_variant_token_exact": (
                by_variant["control"][0]["token_hash"]
                == by_variant["candidate"][0]["token_hash"]
            ),
            "cross_variant_token_exact_required": False,
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
        raise ValueError("DFlash2 stack gates require exactly 16K input and 1024 output")
    arm_gate._parse_dflash_survivors(args.control_survivors)
    arm_gate._parse_dflash_survivors(args.candidate_survivors)
    arm_gate._parse_dflash_adaptive_rows(args.control_adaptive_rows)
    arm_gate._parse_dflash_adaptive_rows(args.candidate_adaptive_rows)
    arm_gate._parse_dflash_custom_rows(args.control_custom_rows)
    arm_gate._parse_dflash_custom_rows(args.candidate_custom_rows)
    arm_gate._parse_dflash_gqa_widths(args.control_gqa_widths)
    arm_gate._parse_dflash_gqa_widths(args.candidate_gqa_widths)
    for prefix in ("control", "candidate"):
        if bool(getattr(args, f"{prefix}_m8_linear_z")) and not bool(
            getattr(args, f"{prefix}_m8_nax_island")
        ):
            raise ValueError("DFlash M8 linear-Z requires the M8 NAX island")
        if bool(getattr(args, f"{prefix}_m7_nax_output")) and not bool(
            getattr(args, f"{prefix}_m8_nax_island")
        ):
            raise ValueError("DFlash M7 output route requires the M8 NAX island")
    for value in (
        args.control_max_mb_per_buffer,
        args.candidate_max_mb_per_buffer,
        args.control_max_ops_per_buffer,
        args.candidate_max_ops_per_buffer,
    ):
        if int(value) <= 0:
            raise ValueError("command-buffer limits must be positive")
    children: list[dict[str, Any]] = []
    with _gpu_lock_scope(args.lock) as lock_scope:
        with tempfile.TemporaryDirectory(prefix="qwen38-dflash-stack-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, variant in enumerate(ORDER):
                child_output = temp_root / f"arm-{index}.json"
                result = _run_attested_child(
                    _child_command(args, variant=variant, output=child_output),
                    environment=_variant_environment(args, variant, os.environ),
                    lock_path=args.lock,
                )
                if result.returncode != 0 or not child_output.is_file():
                    raise RuntimeError(
                        f"isolated {variant} arm {index} failed ({result.returncode}):\n"
                        f"{result.stdout}"
                    )
                children.append(json.loads(child_output.read_text(encoding="utf-8")))

    receipt = _aggregate(args, children, lock_scope=lock_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    print(f"candidate_improvement_pct={receipt['candidate_improvement_pct']:.6f}")
    print(f"promotion_passed={receipt['promotion']['passed']}")
    return 0 if receipt["correctness"]["exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
