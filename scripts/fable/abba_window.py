#!/usr/bin/env python3
"""Run one matched A/B/B/A decode bracket inside a single guarded GPU window.

This process is the guard's *direct child*: it consumes the one-shot guard
attestation exactly once, publishes the DeepSeek-style verified window receipt
(``scripts/deepseek_v4_guard_window.issue_guard_window``), and then spawns one
``scripts/fable/abba_driver.py`` subprocess per arm.  Each arm re-verifies that
receipt against the still-live process ancestry and the still-held flock before
it imports MLX, so every grandchild is attested without ever touching the
already-consumed pipe.

Each arm is a fresh process because each arm loads the model.

Run it THROUGH ``bench/laguna/run_guarded.py`` -- see ``--help`` for the exact
outer command line.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "fable" / "abba_driver.py"
OUT_DIR = ROOT / ".benchmark-artifacts" / "fable"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_GUARDED = Path(
    "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py"
)
QWEN_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")

PRODUCTION_SEEDS = (20260829, 20260830, 20260831)

# The retained 67.818 tok/s "paired routed GLU" arm
# (docs/perf/pr391-m4-paired-routed-glu-result.md, receipt
# rebench3-1788287001-paired-routed-glu-candidate-seeds-16k-1k-seeds-16k-1k.json)
# is the control.  These are exactly the flags and construction-time overrides
# that receipt records.
CONTROL_FLAGS: tuple[str, ...] = (
    "--target-mode",
    "batched",
    "--require-compiled-verify",
    "--m4-stage3",
    "--qsa-fused-kv-gather",
    "--full-frspec",
    "--compiled-mtp-prepare",
    "--max-tokens",
    "1024",
)
CONTROL_CANDIDATE_ENV: tuple[str, ...] = (
    "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE=1",
    "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL=1",
    "MTPLX_QWEN4_M4_ROUTED_GLU=1",
)

ORDERS: dict[str, tuple[str, ...]] = {
    "ABBA": ("A", "B", "B", "A"),
    "BAAB": ("B", "A", "A", "B"),
    "AB": ("A", "B"),
}
ARM_NAMES = {"A": "control", "B": "candidate"}

GUARD_WINDOW_PATH_ENV = "MTPLX_DSV4_GUARD_WINDOW_PATH"
GUARD_WINDOW_SHA256_ENV = "MTPLX_DSV4_GUARD_WINDOW_SHA256"


# --------------------------------------------------------------------------
# Pure planning / argv construction (unit-tested)
# --------------------------------------------------------------------------


def arm_sequence(order: str) -> tuple[str, ...]:
    """The per-seed arm order."""

    try:
        return ORDERS[order]
    except KeyError as exc:
        raise ValueError(
            f"unknown order {order!r}; expected one of {sorted(ORDERS)}"
        ) from exc


def plan_runs(
    seeds: Sequence[int], order: str, sequence: int
) -> list[dict[str, Any]]:
    """Expand seeds x order into the ordered list of arm runs.

    Every seed's full arm order runs before the next seed starts, so each seed
    contributes a self-contained matched bracket.
    """

    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct: {list(seeds)}")
    arms = arm_sequence(order)
    runs: list[dict[str, Any]] = []
    index = 0
    for seed in seeds:
        for position, arm in enumerate(arms):
            runs.append(
                {
                    "index": index,
                    "position_in_seed": position,
                    "arm": arm,
                    "arm_name": ARM_NAMES[arm],
                    "seed": int(seed),
                    "sequence": sequence + index,
                }
            )
            index += 1
    return runs


def arm_label(prefix: str, run: Mapping[str, Any]) -> str:
    return (
        f"{prefix}-{run['arm_name']}-{run['arm']}"
        f"{run['position_in_seed']}-s{run['seed']}"
    )


def receipt_name(prefix: str, run: Mapping[str, Any]) -> str:
    return f"{arm_label(prefix, run)}-{run['sequence']}.json"


def build_arm_argv(
    run: Mapping[str, Any],
    *,
    python: str,
    driver: str,
    label_prefix: str,
    receipt_dir: str,
    common_flags: Sequence[str],
    arm_flags: Sequence[str],
    candidate_env: Sequence[str],
    extra_env: Sequence[str],
) -> list[str]:
    """Build one arm's complete driver command line."""

    argv = [
        python,
        driver,
        "--label",
        arm_label(label_prefix, run),
        "--sequence",
        str(run["sequence"]),
        "--seed",
        str(run["seed"]),
        "--receipt-path",
        str(Path(receipt_dir) / receipt_name(label_prefix, run)),
        "--guard-mode",
        "window",
    ]
    argv.extend(common_flags)
    argv.extend(arm_flags)
    for setting in candidate_env:
        argv.extend(["--candidate-env", setting])
    for setting in extra_env:
        argv.extend(["--env", setting])
    return argv


def merge_candidate_env(
    base: Sequence[str], overrides: Sequence[str]
) -> list[str]:
    """Later KEY=VALUE settings replace earlier ones with the same KEY."""

    merged: dict[str, str] = {}
    for setting in [*base, *overrides]:
        if "=" not in setting:
            raise ValueError(f"expected KEY=VALUE, got {setting!r}")
        key, value = setting.split("=", 1)
        if not key or not value:
            raise ValueError(f"expected KEY=VALUE, got {setting!r}")
        merged[key] = value
    return [f"{key}={value}" for key, value in sorted(merged.items())]


def extract_run_row(
    receipt: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Any]:
    """Flatten one arm receipt into the summary row for the table."""

    rows = receipt.get("rows") or []
    if len(rows) != 1:
        raise ValueError(
            f"arm receipt must hold exactly one measured row, got {len(rows)}"
        )
    row = rows[0]
    compiled_calls = int(row.get("compiled_m4_calls") or 0)
    decode_s = float(row["decode_elapsed_s"])
    generated = int(row["generated_tokens"])
    thermal = row.get("thermal_gate") or {}
    parity = row.get("reference_token_parity") or {}
    return {
        "index": int(run["index"]),
        "position_in_seed": int(run["position_in_seed"]),
        "arm": run["arm"],
        "arm_name": run["arm_name"],
        "seed": int(run["seed"]),
        "sequence": int(run["sequence"]),
        "decode_tok_s": float(row["decode_tok_s"]),
        "decode_s": decode_s,
        "wall_s": float(row["wall_s"]),
        "generated_tokens": generated,
        "compiled_m4_calls": compiled_calls,
        "ms_per_compiled_window": (
            decode_s * 1000.0 / compiled_calls if compiled_calls else None
        ),
        "tokens_per_window": (
            generated / compiled_calls if compiled_calls else None
        ),
        "accepted_by_depth": list(row["accepted_by_depth"]),
        "drafted_by_depth": list(row["drafted_by_depth"]),
        "verify_forward_s": float(row["verify_forward_time_s"]),
        "draft_s": float(row["draft_time_s"]),
        "digest": str(row["response_token_sha256"]),
        "peak_bytes": int(row["peak_memory_bytes"]),
        "ready_c": thermal.get("ready_c"),
        "page_cache_regime": row.get("page_cache_regime"),
        "reference_token_parity": parity.get("status"),
        "ple_hot_rows": row.get("ple_hot_rows"),
        "per_cycle_available": bool((row.get("per_cycle") or {}).get("available")),
    }


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return statistics.fmean(materialized) if materialized else None


def _median(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return statistics.median(materialized) if materialized else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-arm aggregates, per-seed paired deltas, and adjacent-pair deltas."""

    if not rows:
        raise ValueError("summary needs at least one measured row")
    by_arm: dict[str, list[Mapping[str, Any]]] = {"A": [], "B": []}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)

    arms: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        arms[arm] = {
            "arm": arm,
            "arm_name": ARM_NAMES.get(arm, arm),
            "runs": len(arm_rows),
            "mean_decode_tok_s": _mean(r["decode_tok_s"] for r in arm_rows),
            "median_decode_tok_s": _median(r["decode_tok_s"] for r in arm_rows),
            "mean_decode_s": _mean(r["decode_s"] for r in arm_rows),
            "median_decode_s": _median(r["decode_s"] for r in arm_rows),
            "mean_verify_forward_s": _mean(
                r["verify_forward_s"] for r in arm_rows
            ),
            "max_peak_bytes": (
                max(int(r["peak_bytes"]) for r in arm_rows) if arm_rows else None
            ),
            "digests": sorted({str(r["digest"]) for r in arm_rows}),
        }

    seeds = sorted({int(row["seed"]) for row in rows})
    paired: list[dict[str, Any]] = []
    for seed in seeds:
        control = [
            r for r in rows if int(r["seed"]) == seed and r["arm"] == "A"
        ]
        candidate = [
            r for r in rows if int(r["seed"]) == seed and r["arm"] == "B"
        ]
        if not control or not candidate:
            continue
        control_mean = _mean(r["decode_tok_s"] for r in control)
        candidate_mean = _mean(r["decode_tok_s"] for r in candidate)
        delta = candidate_mean - control_mean
        paired.append(
            {
                "seed": seed,
                "control_runs": len(control),
                "candidate_runs": len(candidate),
                "control_mean_decode_tok_s": control_mean,
                "candidate_mean_decode_tok_s": candidate_mean,
                "delta_decode_tok_s": delta,
                "delta_pct": (
                    100.0 * delta / control_mean if control_mean else None
                ),
                "control_mean_decode_s": _mean(r["decode_s"] for r in control),
                "candidate_mean_decode_s": _mean(
                    r["decode_s"] for r in candidate
                ),
                "digests_match": (
                    len({str(r["digest"]) for r in (*control, *candidate)}) == 1
                ),
            }
        )

    adjacent: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda r: int(r["index"]))
    for first, second in zip(ordered, ordered[1:]):
        if int(first["seed"]) != int(second["seed"]):
            continue
        if first["arm"] == second["arm"]:
            continue
        control_row, candidate_row = (
            (first, second) if first["arm"] == "A" else (second, first)
        )
        adjacent.append(
            {
                "seed": int(first["seed"]),
                "control_index": int(control_row["index"]),
                "candidate_index": int(candidate_row["index"]),
                "delta_decode_tok_s": (
                    float(candidate_row["decode_tok_s"])
                    - float(control_row["decode_tok_s"])
                ),
            }
        )

    control_mean = arms.get("A", {}).get("mean_decode_tok_s")
    candidate_mean = arms.get("B", {}).get("mean_decode_tok_s")
    control_median = arms.get("A", {}).get("median_decode_tok_s")
    candidate_median = arms.get("B", {}).get("median_decode_tok_s")
    overall = {
        "control_mean_decode_tok_s": control_mean,
        "candidate_mean_decode_tok_s": candidate_mean,
        "delta_mean_decode_tok_s": (
            None
            if control_mean is None or candidate_mean is None
            else candidate_mean - control_mean
        ),
        "delta_mean_pct": (
            None
            if not control_mean or candidate_mean is None
            else 100.0 * (candidate_mean - control_mean) / control_mean
        ),
        "control_median_decode_tok_s": control_median,
        "candidate_median_decode_tok_s": candidate_median,
        "delta_median_decode_tok_s": (
            None
            if control_median is None or candidate_median is None
            else candidate_median - control_median
        ),
        "paired_delta_mean_decode_tok_s": _mean(
            entry["delta_decode_tok_s"] for entry in paired
        ),
        "paired_delta_median_decode_tok_s": _median(
            entry["delta_decode_tok_s"] for entry in paired
        ),
        "adjacent_delta_mean_decode_tok_s": _mean(
            entry["delta_decode_tok_s"] for entry in adjacent
        ),
        "all_digests_match": len({str(row["digest"]) for row in rows}) == 1,
    }
    return {
        "arms": arms,
        "per_seed": paired,
        "adjacent_pairs": adjacent,
        "overall": overall,
    }


def _format(value: Any, spec: str) -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def render_markdown(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> str:
    """Render the per-run table plus the paired-delta table."""

    lines = [
        "| # | Arm | Seed | Decode tok/s | Decode s | ms/window | tok/window "
        "| Accepted by depth | Verify fwd s | Digest | Peak bytes | Ready C "
        "| Page cache |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- "
        "| ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda r: int(r["index"])):
        lines.append(
            "| {index} | {arm_name} ({arm}) | {seed} | {tok_s} | {decode_s} "
            "| {ms} | {tpw} | {depths} | {vf} | {digest} | {peak:,} | {ready} "
            "| {regime} |".format(
                index=int(row["index"]),
                arm_name=row["arm_name"],
                arm=row["arm"],
                seed=int(row["seed"]),
                tok_s=_format(row["decode_tok_s"], ".6f"),
                decode_s=_format(row["decode_s"], ".6f"),
                ms=_format(row["ms_per_compiled_window"], ".4f"),
                tpw=_format(row["tokens_per_window"], ".4f"),
                depths=",".join(str(int(v)) for v in row["accepted_by_depth"]),
                vf=_format(row["verify_forward_s"], ".6f"),
                digest=str(row["digest"])[:12],
                peak=int(row["peak_bytes"]),
                ready=_format(row["ready_c"], ".4f"),
                regime=row["page_cache_regime"] or "n/a",
            )
        )

    lines.append("")
    lines.append(
        "| Seed | Control tok/s | Candidate tok/s | Delta tok/s | Delta % "
        "| Digests match |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    for entry in summary["per_seed"]:
        lines.append(
            "| {seed} | {control} | {candidate} | {delta} | {pct} | {match} |".format(
                seed=entry["seed"],
                control=_format(entry["control_mean_decode_tok_s"], ".6f"),
                candidate=_format(entry["candidate_mean_decode_tok_s"], ".6f"),
                delta=_format(entry["delta_decode_tok_s"], "+.6f"),
                pct=_format(entry["delta_pct"], "+.4f"),
                match="yes" if entry["digests_match"] else "NO",
            )
        )

    overall = summary["overall"]
    lines.append("")
    lines.append(
        "Control mean {control} tok/s, candidate mean {candidate} tok/s, "
        "delta {delta} tok/s ({pct}%).".format(
            control=_format(overall["control_mean_decode_tok_s"], ".6f"),
            candidate=_format(overall["candidate_mean_decode_tok_s"], ".6f"),
            delta=_format(overall["delta_mean_decode_tok_s"], "+.6f"),
            pct=_format(overall["delta_mean_pct"], "+.4f"),
        )
    )
    lines.append(
        "Median control {control} tok/s, median candidate {candidate} tok/s, "
        "median delta {delta} tok/s.".format(
            control=_format(overall["control_median_decode_tok_s"], ".6f"),
            candidate=_format(overall["candidate_median_decode_tok_s"], ".6f"),
            delta=_format(overall["delta_median_decode_tok_s"], "+.6f"),
        )
    )
    lines.append(
        "Paired (per-seed) delta mean {mean} tok/s, median {median} tok/s; "
        "adjacent-pair delta mean {adjacent} tok/s.".format(
            mean=_format(overall["paired_delta_mean_decode_tok_s"], "+.6f"),
            median=_format(overall["paired_delta_median_decode_tok_s"], "+.6f"),
            adjacent=_format(
                overall["adjacent_delta_mean_decode_tok_s"], "+.6f"
            ),
        )
    )
    lines.append(
        "Every arm produced the same response-token digest: "
        + ("yes" if overall["all_digests_match"] else "NO")
    )
    return "\n".join(lines) + "\n"


def outer_command_line(child_argv: Sequence[str] | None = None) -> str:
    """The exact copy-pasteable guarded invocation for this window."""

    child = list(child_argv or [str(VENV_PYTHON), str(Path(__file__).resolve())])
    parts = [
        f"PYTHONPATH={ROOT}",
        str(VENV_PYTHON),
        str(RUN_GUARDED),
        "--plist",
        str(QWEN_PLIST),
        "--lock-timeout-seconds",
        "1800",
        "--child-timeout-seconds",
        "36000",
        "--",
        *child,
    ]
    return " ".join(shlex.quote(part) if " " in part else part for part in parts)


# --------------------------------------------------------------------------
# Window execution
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Launch this THROUGH the guarded window, never directly:\n\n"
        "  " + outer_command_line() + " \\\n"
        "      --sequence 1788400001\n\n"
        "The window process consumes the guard attestation once and re-attests "
        "each arm subprocess through the DeepSeek verified-window receipt.\n"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument(
        "--order", choices=sorted(ORDERS), default="ABBA"
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        default=None,
        help=f"Repeatable. Default: {' '.join(str(s) for s in PRODUCTION_SEEDS)}",
    )
    parser.add_argument("--label-prefix", default="fable-abba")
    parser.add_argument("--python", default=str(VENV_PYTHON))
    parser.add_argument("--driver", type=Path, default=DRIVER)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--expected-source", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--thermal-gate-max-c", default=None, metavar="CELSIUS")
    parser.add_argument("--prewarm-ngram-table", action="store_true")
    parser.add_argument("--retain-events", action="store_true")
    parser.add_argument("--d3-softfloat64-route", action="store_true")
    parser.add_argument("--require-reference-token-parity", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--control-flag",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "Extra raw driver flag for arm A (repeatable). Use the '=' form "
            "so argparse does not eat the leading dashes: "
            "--control-flag=--nax-verify"
        ),
    )
    parser.add_argument(
        "--candidate-flag",
        action="append",
        default=[],
        metavar="FLAG",
        help=(
            "Extra raw driver flag for arm B (repeatable). Use the '=' form: "
            "--candidate-flag=--nax-verify (values too: "
            "--candidate-flag=--frspec-n --candidate-flag=32768)"
        ),
    )
    parser.add_argument(
        "--control-env",
        action="append",
        default=[],
        metavar="MTPLX_KEY=VALUE",
        help="Override a control construction-time MTPLX_* setting.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="MTPLX_KEY=VALUE",
        help="Candidate construction-time MTPLX_* setting (adds to control).",
    )
    parser.add_argument(
        "--control-extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Non-MTPLX process env for arm A (e.g. MLX_MAX_OPS_PER_BUFFER=...).",
    )
    parser.add_argument(
        "--candidate-extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Non-MTPLX process env for arm B.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned arm command lines and exit without the GPU.",
    )
    return parser


def common_driver_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--source", str(args.source)]
    if args.expected_source:
        flags.extend(["--expected-source", args.expected_source])
    if args.allow_dirty_source:
        flags.append("--allow-dirty-source")
    if args.thermal_gate_max_c is not None:
        flags.extend(["--thermal-gate-max-c", str(args.thermal_gate_max_c)])
    if args.prewarm_ngram_table:
        flags.append("--prewarm-ngram-table")
    if args.retain_events:
        flags.append("--retain-events")
    if args.d3_softfloat64_route:
        flags.append("--d3-softfloat64-route")
    if args.require_reference_token_parity:
        flags.append("--require-reference-token-parity")
    return flags


# Flags the window supplies itself; a user-supplied arm flag that repeats one
# would silently produce two conflicting values on the driver command line.
RESERVED_ARM_FLAGS = frozenset(
    {
        "--label",
        "--sequence",
        "--seed",
        "--receipt-path",
        "--guard-mode",
        "--source",
        "--expected-source",
        "--candidate-env",
        "--env",
        "--thermal-gate-max-c",
    }
)


def check_arm_flags(flags: Sequence[str]) -> None:
    """Reject arm flags that collide with what the window already supplies."""

    for flag in flags:
        name = flag.split("=", 1)[0]
        if name in RESERVED_ARM_FLAGS:
            raise ValueError(
                f"{name} is set by the window; do not pass it as an arm flag"
            )


def arm_specification(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    check_arm_flags(args.control_flag)
    check_arm_flags(args.candidate_flag)
    return {
        "A": {
            "flags": [*CONTROL_FLAGS, *args.control_flag],
            "candidate_env": merge_candidate_env(
                CONTROL_CANDIDATE_ENV, args.control_env
            ),
            "extra_env": list(args.control_extra_env),
        },
        "B": {
            "flags": [*CONTROL_FLAGS, *args.candidate_flag],
            "candidate_env": merge_candidate_env(
                CONTROL_CANDIDATE_ENV, [*args.control_env, *args.candidate_env]
            ),
            "extra_env": list(args.candidate_extra_env),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else PRODUCTION_SEEDS
    runs = plan_runs(seeds, args.order, args.sequence)
    specs = arm_specification(args)
    common = common_driver_flags(args)
    out_dir = args.out_dir.resolve()

    commands = {
        int(run["index"]): build_arm_argv(
            run,
            python=args.python,
            driver=str(args.driver),
            label_prefix=args.label_prefix,
            receipt_dir=str(out_dir),
            common_flags=common,
            arm_flags=specs[run["arm"]]["flags"],
            candidate_env=specs[run["arm"]]["candidate_env"],
            extra_env=specs[run["arm"]]["extra_env"],
        )
        for run in runs
    }

    if args.dry_run:
        print("[fable-abba-window] outer command:")
        print("  " + outer_command_line())
        for run in runs:
            print(f"[arm {run['index']}] " + shlex.join(commands[run["index"]]))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.deepseek_v4_guard_window import issue_guard_window

    window_path, window_digest = issue_guard_window()
    print(
        f"[fable-abba-window] verified guard window {window_path} "
        f"({window_digest})",
        flush=True,
    )

    arm_environment = dict(os.environ)
    arm_environment.pop("MTPLX_GUARD_ATTEST_FD", None)
    arm_environment.pop("MTPLX_GUARD_ATTEST_NONCE", None)
    arm_environment[GUARD_WINDOW_PATH_ENV] = str(window_path)
    arm_environment[GUARD_WINDOW_SHA256_ENV] = window_digest
    arm_environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *(p for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p)]
    )

    started = time.time()
    rows: list[dict[str, Any]] = []
    arm_records: list[dict[str, Any]] = []
    for run in runs:
        command = commands[int(run["index"])]
        print(
            f"[fable-abba-window] arm {run['index']} "
            f"{run['arm_name']}({run['arm']}) seed {run['seed']}: "
            + shlex.join(command),
            flush=True,
        )
        arm_started = time.time()
        completed = subprocess.run(command, env=arm_environment, check=False)
        arm_wall = time.time() - arm_started
        record = {
            **run,
            "command": command,
            "returncode": completed.returncode,
            "arm_wall_s": arm_wall,
            "receipt_path": str(
                out_dir / receipt_name(args.label_prefix, run)
            ),
        }
        arm_records.append(record)
        if completed.returncode != 0:
            record["status"] = "failed"
            print(
                f"[fable-abba-window] arm {run['index']} FAILED with exit "
                f"{completed.returncode}; stopping the bracket",
                flush=True,
            )
            break
        receipt = json.loads(Path(record["receipt_path"]).read_text())
        record["status"] = "measured"
        rows.append(extract_run_row(receipt, run))

    summary: dict[str, Any] | None = None
    table = ""
    if rows:
        summary = summarize(rows)
        table = render_markdown(rows, summary)
        print(table, flush=True)

    payload = {
        "schema": "mtplx-fable-abba-window-v1",
        "sequence": args.sequence,
        "order": args.order,
        "seeds": list(seeds),
        "label_prefix": args.label_prefix,
        "started_epoch_s": started,
        "elapsed_s": time.time() - started,
        "outer_command": outer_command_line(),
        "guard_window": {
            "receipt_path": str(window_path),
            "receipt_sha256": window_digest,
        },
        "arm_specification": {
            arm: {
                "flags": spec["flags"],
                "candidate_env": spec["candidate_env"],
                "extra_env": spec["extra_env"],
            }
            for arm, spec in specs.items()
        },
        "common_driver_flags": common,
        "arms": arm_records,
        "rows": rows,
        "summary": summary,
        "markdown": table,
    }
    stem = f"abba-window-{args.sequence}-{args.label_prefix}"
    summary_path = out_dir / f"{stem}.json"
    markdown_path = out_dir / f"{stem}.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(table or "no measured arms\n")
    print(f"[fable-abba-window] wrote {summary_path}", flush=True)
    print(f"[fable-abba-window] wrote {markdown_path}", flush=True)
    return 0 if rows and len(rows) == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
