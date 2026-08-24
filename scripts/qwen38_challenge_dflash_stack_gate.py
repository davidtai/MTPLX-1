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
    parser.add_argument("--control-disable-m8-output", action="store_true")
    parser.add_argument("--candidate-disable-m8-output", action="store_true")
    parser.add_argument("--control-m8-linear-z", action="store_true")
    parser.add_argument("--candidate-m8-linear-z", action="store_true")
    parser.add_argument("--control-m7-nax-output", action="store_true")
    parser.add_argument("--candidate-m7-nax-output", action="store_true")
    parser.add_argument("--control-m7-nax-linear-z", action="store_true")
    parser.add_argument("--candidate-m7-nax-linear-z", action="store_true")
    parser.add_argument("--control-m8-nax-expanded", action="store_true")
    parser.add_argument("--candidate-m8-nax-expanded", action="store_true")
    parser.add_argument("--control-m8-nax-kv", action="store_true")
    parser.add_argument("--candidate-m8-nax-kv", action="store_true")
    parser.add_argument("--control-m8-nax-qkv", action="store_true")
    parser.add_argument("--candidate-m8-nax-qkv", action="store_true")
    parser.add_argument("--control-m8-nax-mlp", action="store_true")
    parser.add_argument("--candidate-m8-nax-mlp", action="store_true")
    parser.add_argument("--control-m5-exact", action="store_true")
    parser.add_argument("--candidate-m5-exact", action="store_true")
    parser.add_argument("--control-m6-kp1", action="store_true")
    parser.add_argument("--candidate-m6-kp1", action="store_true")
    parser.add_argument("--control-m7-linear-z-nsg4", action="store_true")
    parser.add_argument("--candidate-m7-linear-z-nsg4", action="store_true")
    parser.add_argument("--control-m8-kv-nsg16", action="store_true")
    parser.add_argument("--candidate-m8-kv-nsg16", action="store_true")
    parser.add_argument("--control-m8-qkv-nsg4", action="store_true")
    parser.add_argument("--candidate-m8-qkv-nsg4", action="store_true")
    parser.add_argument("--control-m56-partition-v2", action="store_true")
    parser.add_argument("--candidate-m56-partition-v2", action="store_true")
    parser.add_argument("--control-m5-partition-v2", action="store_true")
    parser.add_argument("--candidate-m5-partition-v2", action="store_true")
    parser.add_argument("--control-m6-partition-v2", action="store_true")
    parser.add_argument("--candidate-m6-partition-v2", action="store_true")
    parser.add_argument("--control-disable-row24-prefill-ladder", action="store_true")
    parser.add_argument("--candidate-disable-row24-prefill-ladder", action="store_true")
    parser.add_argument("--control-disable-row24-decode-ladder", action="store_true")
    parser.add_argument("--candidate-disable-row24-decode-ladder", action="store_true")
    parser.add_argument("--control-disable-row48-prefill-fusion", action="store_true")
    parser.add_argument("--candidate-disable-row48-prefill-fusion", action="store_true")
    parser.add_argument("--control-disable-row48-decode-fusion", action="store_true")
    parser.add_argument("--candidate-disable-row48-decode-fusion", action="store_true")
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
    environment["MTPLX_QWEN38_M7_LINEAR_Z_NSG4"] = (
        "1" if bool(getattr(args, f"{prefix}_m7_linear_z_nsg4", False)) else "0"
    )
    environment["MTPLX_QWEN38_M8_KV_NSG16"] = (
        "1" if bool(getattr(args, f"{prefix}_m8_kv_nsg16", False)) else "0"
    )
    environment["MTPLX_QWEN38_M8_QKV_NSG4"] = (
        "1" if bool(getattr(args, f"{prefix}_m8_qkv_nsg4", False)) else "0"
    )
    environment["MTPLX_QWEN38_M56_PARTITION_V2"] = (
        "1" if bool(getattr(args, f"{prefix}_m56_partition_v2", False)) else "0"
    )
    environment["MTPLX_QWEN38_M5_PARTITION_V2"] = (
        "1" if bool(getattr(args, f"{prefix}_m5_partition_v2", False)) else "0"
    )
    environment["MTPLX_QWEN38_M6_PARTITION_V2"] = (
        "1" if bool(getattr(args, f"{prefix}_m6_partition_v2", False)) else "0"
    )
    return environment


def _variant_config(
    args: argparse.Namespace,
    variant: str,
) -> tuple[Any, ...]:
    if variant == "control":
        return (
            args.control_survivors,
            args.control_adaptive_rows,
            args.control_custom_rows,
            args.control_gqa_widths,
            bool(args.control_m8_nax_island),
            bool(args.control_disable_m8_output),
            bool(args.control_m8_linear_z),
            bool(args.control_m7_nax_output),
            bool(args.control_m7_nax_linear_z),
            bool(args.control_m8_nax_expanded),
            bool(args.control_m8_nax_kv),
            bool(args.control_m8_nax_qkv),
            bool(args.control_m8_nax_mlp),
            bool(args.control_m5_exact),
            bool(args.control_m6_kp1),
            bool(args.control_disable_row24_prefill_ladder),
            bool(args.control_disable_row24_decode_ladder),
            bool(args.control_disable_row48_prefill_fusion),
            bool(args.control_disable_row48_decode_fusion),
            bool(args.control_cost_aligned_widths),
            bool(args.control_release_native_mtp),
        )
    return (
        args.candidate_survivors,
        args.candidate_adaptive_rows,
        args.candidate_custom_rows,
        args.candidate_gqa_widths,
        bool(args.candidate_m8_nax_island),
        bool(args.candidate_disable_m8_output),
        bool(args.candidate_m8_linear_z),
        bool(args.candidate_m7_nax_output),
        bool(args.candidate_m7_nax_linear_z),
        bool(args.candidate_m8_nax_expanded),
        bool(args.candidate_m8_nax_kv),
        bool(args.candidate_m8_nax_qkv),
        bool(args.candidate_m8_nax_mlp),
        bool(args.candidate_m5_exact),
        bool(args.candidate_m6_kp1),
        bool(args.candidate_disable_row24_prefill_ladder),
        bool(args.candidate_disable_row24_decode_ladder),
        bool(args.candidate_disable_row48_prefill_fusion),
        bool(args.candidate_disable_row48_decode_fusion),
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
        disable_m8_output,
        m8_linear_z,
        m7_nax_output,
        m7_nax_linear_z,
        m8_nax_expanded,
        m8_nax_kv,
        m8_nax_qkv,
        m8_nax_mlp,
        m5_exact,
        m6_kp1,
        disable_row24_prefill_ladder,
        disable_row24_decode_ladder,
        disable_row48_prefill_fusion,
        disable_row48_decode_fusion,
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
    if disable_m8_output:
        command.append("--disable-dflash-m8-output")
    if m8_linear_z:
        command.append("--dflash-m8-linear-z")
    if m7_nax_output:
        command.append("--dflash-m7-nax-output")
    if m7_nax_linear_z:
        command.append("--dflash-m7-nax-linear-z")
    if m8_nax_expanded:
        command.append("--dflash-m8-nax-expanded")
    if m8_nax_kv:
        command.append("--dflash-m8-nax-kv")
    if m8_nax_qkv:
        command.append("--dflash-m8-nax-qkv")
    if m8_nax_mlp:
        command.append("--dflash-m8-nax-mlp")
    if m5_exact:
        command.append("--dflash-m5-exact")
    if m6_kp1:
        command.append("--dflash-m6-kp1")
    if disable_row24_prefill_ladder:
        command.append("--disable-dflash-row24-prefill-ladder")
    if disable_row24_decode_ladder:
        command.append("--disable-dflash-row24-decode-ladder")
    if disable_row48_prefill_fusion:
        command.append("--disable-dflash-row48-prefill-fusion")
    if disable_row48_decode_fusion:
        command.append("--disable-dflash-row48-decode-fusion")
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
    split_labels = {
        "m7_linear_z_nsg4": ("m7_nsg_by_shape", "5120x6144", 4, "m7_to_m8_nax"),
        "m8_kv_nsg16": ("m8_nsg_by_shape", "5120x1024", 16, "m8_nax"),
        "m8_qkv_nsg4": ("m8_nsg_by_shape", "5120x10240", 4, "m8_nax"),
    }
    if args.candidate_label in split_labels:
        route_key, shape_key, candidate_nsg, counter_prefix = split_labels[
            args.candidate_label
        ]
        k_text, n_text = shape_key.split("x", 1)

        def matches(row: dict[str, Any], nsg: int) -> bool:
            report = row.get("feature_receipt", {}).get(
                "dflash_nax_split_tuning", {}
            )
            configured = int(report.get(route_key, {}).get(shape_key, 8))
            counter = (
                f"{counter_prefix}_nsg{nsg}_k{k_text}_n{n_text}"
            )
            calls = int(
                row.get("engagement", {}).get("nax_verify", {}).get(counter, 0)
            )
            return configured == nsg and calls > 0

        return all(matches(row, 8) for row in by_variant["control"]) and all(
            matches(row, candidate_nsg) for row in by_variant["candidate"]
        )
    if args.candidate_label in {
        "m5_partition_v2",
        "m6_partition_v2",
        "m56_partition_v2",
    }:
        expected_families = {
            "m5_partition_v2": {"m5"},
            "m6_partition_v2": {"m6"},
            "m56_partition_v2": {"m5", "m6"},
        }[args.candidate_label]

        def matches(row: dict[str, Any], families: set[str]) -> bool:
            report = row.get("feature_receipt", {}).get(
                "dflash_m56_partition_tuning", {}
            )
            if bool(report.get("active")) != bool(families):
                return False
            if not families:
                return not report.get("m5_kparts_by_shape") and not report.get(
                    "m6_kparts_by_shape"
                )
            counters = row.get("engagement", {}).get("nax_verify", {})
            for family_key, counter_family, receipt_key in (
                ("m5", "m5_exact_ksplit", "m5_kparts_by_shape"),
                ("m6", "m6_ksplit", "m6_kparts_by_shape"),
            ):
                routes = report.get(receipt_key, {})
                if (family_key in families) != bool(routes):
                    return False
                for shape, parts in routes.items():
                    k_text, n_text = shape.split("x", 1)
                    counter = f"{counter_family}_kp{parts}_k{k_text}_n{n_text}"
                    if int(counters.get(counter, 0)) <= 0:
                        return False
            return True

        return all(matches(row, set()) for row in by_variant["control"]) and all(
            matches(row, expected_families) for row in by_variant["candidate"]
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
    if args.candidate_label == "m7_nax_linear_z":
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
            bool(route(row).get("include_m7_output"))
            and not bool(route(row).get("include_m7_linear_z"))
            and route(row).get("m7_shapes") == [[6144, 5120]]
            and calls(row, "m7_to_m8_nax_k6144_n5120") > 0
            and calls(row, "m7_to_m8_nax_k5120_n6144") == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m7_output"))
            and bool(route(row).get("include_m7_linear_z"))
            and route(row).get("m7_shapes") == [[5120, 6144], [6144, 5120]]
            and int(route(row).get("eligible_m7_projections", 0)) == 64
            and int(route(row).get("eligible_m7_linear_z_projections", 0)) == 48
            and calls(row, "m7_to_m8_nax_k6144_n5120") > 0
            and calls(row, "m7_to_m8_nax_k5120_n6144") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_nax_expanded":
        expected_shapes = [
            [5120, 1024],
            [5120, 10240],
            [5120, 17408],
        ]
        counter_keys = [
            f"m8_nax_k{k}_n{n}" for k, n in expected_shapes
        ]

        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], key: str) -> int:
            return int(
                row.get("engagement", {}).get("nax_verify", {}).get(key, 0)
            )

        return all(
            not bool(route(row).get("include_m8_expanded"))
            and route(row).get("m8_expanded_shapes", []) == []
            and all(calls(row, key) == 0 for key in counter_keys)
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m8_expanded"))
            and route(row).get("m8_expanded_shapes") == expected_shapes
            and int(route(row).get("eligible_m8_expanded_projections", 0)) == 192
            and all(calls(row, key) > 0 for key in counter_keys)
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_nax_kv":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any]) -> int:
            return int(
                row.get("engagement", {})
                .get("nax_verify", {})
                .get("m8_nax_k5120_n1024", 0)
            )

        return all(
            not bool(route(row).get("include_m8_kv")) and calls(row) == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m8_kv"))
            and route(row).get("m8_expanded_shapes") == [[5120, 1024]]
            and int(route(row).get("eligible_m8_expanded_projections", 0)) == 32
            and calls(row) > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_nax_qkv":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], shape: str) -> int:
            return int(
                row.get("engagement", {}).get("nax_verify", {}).get(shape, 0)
            )

        return all(
            bool(route(row).get("include_m8_kv"))
            and not bool(route(row).get("include_m8_qkv"))
            and calls(row, "m8_nax_k5120_n1024") > 0
            and calls(row, "m8_nax_k5120_n10240") == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m8_kv"))
            and bool(route(row).get("include_m8_qkv"))
            and route(row).get("m8_expanded_shapes")
            == [[5120, 1024], [5120, 10240]]
            and int(route(row).get("eligible_m8_expanded_projections", 0)) == 80
            and calls(row, "m8_nax_k5120_n1024") > 0
            and calls(row, "m8_nax_k5120_n10240") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_nax_mlp":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], shape: str) -> int:
            return int(
                row.get("engagement", {}).get("nax_verify", {}).get(shape, 0)
            )

        return all(
            bool(route(row).get("include_m8_qkv"))
            and not bool(route(row).get("include_m8_mlp"))
            and calls(row, "m8_nax_k5120_n10240") > 0
            and calls(row, "m8_nax_k5120_n17408") == 0
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m8_qkv"))
            and bool(route(row).get("include_m8_mlp"))
            and route(row).get("m8_expanded_shapes")
            == [[5120, 1024], [5120, 10240], [5120, 17408]]
            and int(route(row).get("eligible_m8_expanded_projections", 0)) == 192
            and calls(row, "m8_nax_k5120_n10240") > 0
            and calls(row, "m8_nax_k5120_n17408") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_no_output":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], key: str) -> int:
            return int(
                row.get("engagement", {}).get("nax_verify", {}).get(key, 0)
            )

        return all(
            bool(route(row).get("include_m8_output"))
            and calls(row, "m8_nax_k6144_n5120") > 0
            and calls(row, "m7_to_m8_nax_k6144_n5120") > 0
            for row in by_variant["control"]
        ) and all(
            not bool(route(row).get("include_m8_output"))
            and calls(row, "m8_nax_k6144_n5120") == 0
            and calls(row, "m7_to_m8_nax_k6144_n5120") > 0
            and calls(row, "m8_nax_k5120_n17408") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m8_no_qkv":
        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], key: str) -> int:
            return int(
                row.get("engagement", {}).get("nax_verify", {}).get(key, 0)
            )

        return all(
            bool(route(row).get("include_m8_qkv"))
            and bool(route(row).get("include_m8_mlp"))
            and calls(row, "m8_nax_k5120_n10240") > 0
            and calls(row, "m8_nax_k5120_n17408") > 0
            for row in by_variant["control"]
        ) and all(
            not bool(route(row).get("include_m8_qkv"))
            and bool(route(row).get("include_m8_mlp"))
            and calls(row, "m8_nax_k5120_n10240") == 0
            and calls(row, "m8_nax_k5120_n17408") > 0
            and calls(row, "m8_nax_k5120_n1024") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m5_exact":
        shapes = (
            (5120, 1024),
            (5120, 6144),
            (5120, 10240),
            (5120, 12288),
            (5120, 17408),
            (5120, 48),
            (6144, 5120),
            (17408, 5120),
        )

        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], kind: str, shape: tuple[int, int]) -> int:
            k, n = shape
            return int(
                row.get("engagement", {})
                .get("nax_verify", {})
                .get(f"{kind}_k{k}_n{n}", 0)
            )

        return all(
            not bool(route(row).get("include_m5_exact"))
            and all(calls(row, "m5_padded_m6_ksplit_kp2", shape) > 0 for shape in shapes)
            and all(calls(row, "m5_exact_ksplit", shape) == 0 for shape in shapes)
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m5_exact"))
            and all(calls(row, "m5_padded_m6_ksplit_kp2", shape) == 0 for shape in shapes)
            and all(calls(row, "m5_exact_ksplit", shape) > 0 for shape in shapes)
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "m6_kp1":
        selected = ((5120, 10240), (5120, 17408))

        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def calls(row: dict[str, Any], kind: str, shape: tuple[int, int]) -> int:
            k, n = shape
            return int(
                row.get("engagement", {})
                .get("nax_verify", {})
                .get(f"{kind}_k{k}_n{n}", 0)
            )

        return all(
            bool(route(row).get("include_m5_exact"))
            and not bool(route(row).get("include_m6_kp1"))
            and all(calls(row, "m6_ksplit_kp2", shape) > 0 for shape in selected)
            and all(calls(row, "m6_ksplit_kp1", shape) == 0 for shape in selected)
            for row in by_variant["control"]
        ) and all(
            bool(route(row).get("include_m5_exact"))
            and bool(route(row).get("include_m6_kp1"))
            and route(row).get("m6_kp1_shapes") == [[5120, 10240], [5120, 17408]]
            and all(calls(row, "m6_ksplit_kp2", shape) == 0 for shape in selected)
            and all(calls(row, "m6_ksplit_kp1", shape) > 0 for shape in selected)
            for row in by_variant["candidate"]
        )
    if args.candidate_label in {"row24_no_prefill", "row24_no_decode"}:
        disabled_phase = (
            "prefill" if args.candidate_label == "row24_no_prefill" else "decode"
        )
        other_phase = "decode" if disabled_phase == "prefill" else "prefill"

        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row.get("feature_receipt", {}).get("r24_eval_ladder", {}))

        def calls(row: dict[str, Any], phase: str) -> int:
            return int(
                row.get("engagement", {})
                .get("r24_eval_ladder", {})
                .get(f"{phase}_calls", 0)
            )

        return all(
            bool(route(row).get(f"{disabled_phase}_active"))
            and bool(route(row).get(f"{other_phase}_active"))
            and calls(row, disabled_phase) > 0
            and calls(row, other_phase) > 0
            for row in by_variant["control"]
        ) and all(
            not bool(route(row).get(f"{disabled_phase}_active"))
            and bool(route(row).get(f"{other_phase}_active"))
            and calls(row, disabled_phase) == 0
            and calls(row, other_phase) > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label in {"row48_no_prefill", "row48_no_decode"}:
        disabled_phase = (
            "prefill" if args.candidate_label == "row48_no_prefill" else "decode"
        )
        other_phase = "decode" if disabled_phase == "prefill" else "prefill"

        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("r48_boundary_fused", {})
            )

        def calls(row: dict[str, Any], phase: str, kind: str) -> int:
            return int(
                row.get("engagement", {})
                .get("r48_boundary_fused", {})
                .get(f"{phase}_{kind}_boundaries", 0)
            )

        return all(
            bool(route(row).get(f"{disabled_phase}_active"))
            and bool(route(row).get(f"{other_phase}_active"))
            and calls(row, disabled_phase, "fused") > 0
            and calls(row, disabled_phase, "stock") == 0
            for row in by_variant["control"]
        ) and all(
            not bool(route(row).get(f"{disabled_phase}_active"))
            and bool(route(row).get(f"{other_phase}_active"))
            and calls(row, disabled_phase, "fused") == 0
            and calls(row, disabled_phase, "stock") > 0
            and calls(row, other_phase, "fused") > 0
            for row in by_variant["candidate"]
        )
    if args.candidate_label == "final_phase_stack_v2":
        retained_counters = (
            "m7_to_m8_nax_k6144_n5120",
            "m7_to_m8_nax_k5120_n6144",
            "m8_nax_k5120_n1024",
            "m8_nax_k5120_n10240",
            "m8_nax_k5120_n17408",
            "m5_exact_ksplit",
            "m6_ksplit_kp1",
        )

        def route(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_m8_nax_island", {})
            )

        def gqa(row: dict[str, Any]) -> dict[str, Any]:
            return dict(
                row.get("feature_receipt", {}).get("dflash_gqa_widths", {})
            )

        def calls(row: dict[str, Any], key: str) -> int:
            return int(
                row.get("engagement", {}).get("nax_verify", {}).get(key, 0)
            )

        return all(
            not route(row)
            and not bool(gqa(row).get("active"))
            and calls(row, "m8_nax_k6144_n5120") == 0
            and all(calls(row, key) == 0 for key in retained_counters)
            for row in by_variant["control"]
        ) and all(
            bool(gqa(row).get("active"))
            and gqa(row).get("widths") == [6, 7, 8]
            and not bool(route(row).get("include_m8_output"))
            and bool(route(row).get("include_m7_output"))
            and bool(route(row).get("include_m7_linear_z"))
            and bool(route(row).get("include_m8_kv"))
            and bool(route(row).get("include_m8_qkv"))
            and bool(route(row).get("include_m8_mlp"))
            and bool(route(row).get("include_m5_exact"))
            and bool(route(row).get("include_m6_kp1"))
            and calls(row, "m8_nax_k6144_n5120") == 0
            and all(calls(row, key) > 0 for key in retained_counters)
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
            "control_disable_m8_output": bool(args.control_disable_m8_output),
            "candidate_disable_m8_output": bool(args.candidate_disable_m8_output),
            "control_m8_linear_z": bool(args.control_m8_linear_z),
            "candidate_m8_linear_z": bool(args.candidate_m8_linear_z),
            "control_m7_nax_output": bool(args.control_m7_nax_output),
            "candidate_m7_nax_output": bool(args.candidate_m7_nax_output),
            "control_m7_nax_linear_z": bool(args.control_m7_nax_linear_z),
            "candidate_m7_nax_linear_z": bool(args.candidate_m7_nax_linear_z),
            "control_m8_nax_expanded": bool(args.control_m8_nax_expanded),
            "candidate_m8_nax_expanded": bool(args.candidate_m8_nax_expanded),
            "control_m8_nax_kv": bool(args.control_m8_nax_kv),
            "candidate_m8_nax_kv": bool(args.candidate_m8_nax_kv),
            "control_m8_nax_qkv": bool(args.control_m8_nax_qkv),
            "candidate_m8_nax_qkv": bool(args.candidate_m8_nax_qkv),
            "control_m8_nax_mlp": bool(args.control_m8_nax_mlp),
            "candidate_m8_nax_mlp": bool(args.candidate_m8_nax_mlp),
            "control_m5_exact": bool(args.control_m5_exact),
            "candidate_m5_exact": bool(args.candidate_m5_exact),
            "control_m6_kp1": bool(args.control_m6_kp1),
            "candidate_m6_kp1": bool(args.candidate_m6_kp1),
            "control_m7_linear_z_nsg4": bool(args.control_m7_linear_z_nsg4),
            "candidate_m7_linear_z_nsg4": bool(args.candidate_m7_linear_z_nsg4),
            "control_m8_kv_nsg16": bool(args.control_m8_kv_nsg16),
            "candidate_m8_kv_nsg16": bool(args.candidate_m8_kv_nsg16),
            "control_m8_qkv_nsg4": bool(args.control_m8_qkv_nsg4),
            "candidate_m8_qkv_nsg4": bool(args.candidate_m8_qkv_nsg4),
            "control_m56_partition_v2": bool(args.control_m56_partition_v2),
            "candidate_m56_partition_v2": bool(args.candidate_m56_partition_v2),
            "control_m5_partition_v2": bool(args.control_m5_partition_v2),
            "candidate_m5_partition_v2": bool(args.candidate_m5_partition_v2),
            "control_m6_partition_v2": bool(args.control_m6_partition_v2),
            "candidate_m6_partition_v2": bool(args.candidate_m6_partition_v2),
            "control_disable_row24_prefill_ladder": bool(
                args.control_disable_row24_prefill_ladder
            ),
            "candidate_disable_row24_prefill_ladder": bool(
                args.candidate_disable_row24_prefill_ladder
            ),
            "control_disable_row24_decode_ladder": bool(
                args.control_disable_row24_decode_ladder
            ),
            "candidate_disable_row24_decode_ladder": bool(
                args.candidate_disable_row24_decode_ladder
            ),
            "control_disable_row48_prefill_fusion": bool(
                args.control_disable_row48_prefill_fusion
            ),
            "candidate_disable_row48_prefill_fusion": bool(
                args.candidate_disable_row48_prefill_fusion
            ),
            "control_disable_row48_decode_fusion": bool(
                args.control_disable_row48_decode_fusion
            ),
            "candidate_disable_row48_decode_fusion": bool(
                args.candidate_disable_row48_decode_fusion
            ),
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
        if bool(getattr(args, f"{prefix}_m7_nax_linear_z")) and not bool(
            getattr(args, f"{prefix}_m7_nax_output")
        ):
            raise ValueError("DFlash M7 linear-Z route requires the M7 output route")
        if bool(getattr(args, f"{prefix}_m8_nax_expanded")) and not bool(
            getattr(args, f"{prefix}_m8_nax_island")
        ):
            raise ValueError("DFlash expanded M8 route requires the M8 NAX island")
        if bool(getattr(args, f"{prefix}_m8_nax_kv")) and not bool(
            getattr(args, f"{prefix}_m8_nax_island")
        ):
            raise ValueError("DFlash M8 K/V route requires the M8 NAX island")
        if bool(getattr(args, f"{prefix}_disable_m8_output")) and not bool(
            getattr(args, f"{prefix}_m8_nax_island")
        ):
            raise ValueError("Disabling DFlash M8 output requires the NAX verify route")
        if (
            bool(getattr(args, f"{prefix}_m5_exact"))
            or bool(getattr(args, f"{prefix}_m6_kp1"))
        ) and not bool(getattr(args, f"{prefix}_m8_nax_island")):
            raise ValueError("DFlash M5/M6 tuning requires the NAX verify route")
        if bool(getattr(args, f"{prefix}_m6_kp1")) and not bool(
            getattr(args, f"{prefix}_m5_exact")
        ):
            raise ValueError("DFlash M6 tuning requires the retained exact-M5 route")
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
