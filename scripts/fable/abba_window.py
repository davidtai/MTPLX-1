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
#: ``--max-tokens`` inside :data:`CONTROL_FLAGS`.  Both arms always carry the
#: same value, so it belongs to the shared baseline rather than either arm.
DEFAULT_MAX_TOKENS = 1024

#: ``--prefill-only``: enough decode to produce a real first token and a
#: non-degenerate ``decode_tok_s``, few enough that the arm is dominated by
#: prefill.  ``prefill_tok_s`` / ``prompt_eval_time_s`` / ``ttft_s`` are
#: recorded per row either way; this just stops paying 1,024 tokens of decode
#: for a prefill-only question.
PREFILL_ONLY_MAX_TOKENS = 64

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
# Context-copy cost model (the corrected cycle-time statistic)
# --------------------------------------------------------------------------
#
# ``mtplx/context_copy.py`` prompt-lookup decoding is ON BY DEFAULT and emits
# ~9.4% of the output tokens through *block rounds* -- variable-width
# ``forward_ar`` calls that are NOT the fixed-M4 compiled window.  Those rounds
# are cheap per token (8.33 ms/token vs 15.13 ms/token for an M4 window) and
# their COUNT is trajectory luck: it depends on how often the model happened to
# quote the prompt, not on the kernel under test.  Reading ``decode_tok_s`` or
# raw ``decode_s / compiled_m4_calls`` therefore mixes a 4.5%-noise retrieval
# yield into a 0.5%-noise kernel measurement.
#
# Fitting ``decode_s = m4*C4 + rounds*C0 + block_tokens*C1`` jointly over the
# 39 complete control runs and the HC_M4 stack arms (6 trajectory points, 4 free
# parameters, residual 111 ms = 0.7%) gives C0 = 21.0 ms per copy round and
# C1 = 3.636 ms per verified copy row.  Subtracting that fitted copy budget and
# dividing by the compiled M4 calls collapses the cross-seed spread from 7.8%
# (tok/s) to 1.1%, with a 0.3-0.7% within-seed noise floor:
#
#     ms_per_m4_window_net =
#         (decode_s - rounds*C0 - accepted_tokens*C1) / compiled_m4_calls
#
# This is the PRIMARY cycle-time metric for every paired delta below.  Raw
# ms/window and tok/s stay in the receipt so a window is still comparable to
# the pre-correction ledger, but they are secondary.
#
# Token basis, stated because it matters for a re-fit: the subtraction uses
# ``context_copy.accepted_tokens``, not the wider ``drafted_tokens`` (the rows
# the block round actually verified, which is what C1 was fitted against).
# Accepted is the stable, receipt-native quantity and the two move together, so
# the default C1 below absorbs the difference.  A window that re-fits the cost
# model against verified ROWS must pass the re-fitted value through
# ``--copy-token-cost-s``; ``drafted_tokens`` is recorded on every row so the
# re-fit can be done from the receipts alone.
#
#: Fitted fixed cost of one context-copy block round, seconds (C0).
DEFAULT_COPY_ROUND_COST_S = 0.0210
#: Fitted marginal cost of one verified context-copy row, seconds (C1).
DEFAULT_COPY_TOKEN_COST_S = 0.003636


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


def resolve_max_tokens(
    max_tokens: int | None, prefill_only: bool
) -> int:
    """The generated-token budget both arms carry.

    ``--max-tokens`` always wins, so ``--prefill-only --max-tokens 256`` is a
    coherent request rather than a silently ignored one.
    """

    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError(f"--max-tokens must be >= 1, got {max_tokens}")
        return int(max_tokens)
    return PREFILL_ONLY_MAX_TOKENS if prefill_only else DEFAULT_MAX_TOKENS


def control_flags(max_tokens: int) -> list[str]:
    """:data:`CONTROL_FLAGS` with its ``--max-tokens`` value replaced.

    Replaced, never appended: the driver's parser would take the last of two
    ``--max-tokens`` and the receipt would disagree with the command line the
    window printed.
    """

    flags = list(CONTROL_FLAGS)
    index = flags.index("--max-tokens")
    flags[index + 1] = str(int(max_tokens))
    return flags


def merge_env_settings(
    base: Sequence[str], overrides: Sequence[str]
) -> list[str]:
    """Later KEY=VALUE settings replace earlier ones with the same KEY.

    Used for both ``--candidate-env`` and the raw ``--env`` passthrough. The
    de-duplication matters on arm B, which now carries the control settings as
    well: the driver's ``parse_key_values`` refuses a repeated key outright, so
    a candidate override of a control setting has to collapse here rather than
    reach the command line twice.
    """

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
    receipt: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    copy_round_cost_s: float = DEFAULT_COPY_ROUND_COST_S,
    copy_token_cost_s: float = DEFAULT_COPY_TOKEN_COST_S,
) -> dict[str, Any]:
    """Flatten one arm receipt into the summary row for the table.

    ``copy_round_cost_s`` / ``copy_token_cost_s`` are the fitted C0/C1 of the
    context-copy cost model documented at the top of this module; they define
    ``decode_s_net`` and ``ms_per_m4_window_net``.  A receipt with no
    ``context_copy`` block (context-copy disabled, or an older driver) reads as
    zero rounds and zero accepted tokens, so the net metric equals the raw one.
    """

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
    context_copy = row.get("context_copy") or {}
    copy_rounds = int(context_copy.get("rounds") or 0)
    copy_accepted = int(context_copy.get("accepted_tokens") or 0)
    copy_drafted = int(context_copy.get("drafted_tokens") or 0)
    # The fitted copy budget this run spent OUTSIDE the compiled M4 windows.
    copy_cost_s = (
        copy_rounds * float(copy_round_cost_s)
        + copy_accepted * float(copy_token_cost_s)
    )
    decode_s_net = decode_s - copy_cost_s
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
        # -- context-copy corrected cycle time (PRIMARY) -------------------
        "context_copy_rounds": copy_rounds,
        "context_copy_accepted_tokens": copy_accepted,
        "context_copy_drafted_tokens": copy_drafted,
        "context_copy_active": bool(context_copy.get("active", bool(copy_rounds))),
        "context_copy_cost_s": copy_cost_s,
        "decode_s_net": decode_s_net,
        "ms_per_m4_window_net": (
            decode_s_net * 1000.0 / compiled_calls if compiled_calls else None
        ),
        "tokens_per_m4_window": (
            (generated - copy_accepted) / compiled_calls
            if compiled_calls
            else None
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
    materialized = [float(value) for value in values if value is not None]
    return statistics.fmean(materialized) if materialized else None


def _median(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return statistics.median(materialized) if materialized else None


def _delta(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control is None:
        return None
    return candidate - control


def _delta_pct(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or not control:
        return None
    return 100.0 * (candidate - control) / control


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    copy_round_cost_s: float = DEFAULT_COPY_ROUND_COST_S,
    copy_token_cost_s: float = DEFAULT_COPY_TOKEN_COST_S,
) -> dict[str, Any]:
    """Per-arm aggregates, per-seed paired deltas, and adjacent-pair deltas.

    The PRIMARY cycle-time metric is ``ms_per_m4_window_net`` -- decode seconds
    with the fitted context-copy budget removed, per compiled M4 window (see the
    cost-model note at the top of this module).  It is a COST, so a negative
    delta is the candidate winning.  Raw ``ms_per_compiled_window`` and
    ``decode_tok_s`` deltas are kept alongside it as secondary readings.

    The two cost constants are recorded in the returned summary so a receipt
    always states the model its net numbers were computed under.
    """

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
            "mean_ms_per_m4_window_net": _mean(
                r.get("ms_per_m4_window_net") for r in arm_rows
            ),
            "median_ms_per_m4_window_net": _median(
                r.get("ms_per_m4_window_net") for r in arm_rows
            ),
            "mean_ms_per_compiled_window": _mean(
                r.get("ms_per_compiled_window") for r in arm_rows
            ),
            "mean_tokens_per_m4_window": _mean(
                r.get("tokens_per_m4_window") for r in arm_rows
            ),
            "mean_context_copy_rounds": _mean(
                r.get("context_copy_rounds") for r in arm_rows
            ),
            "mean_context_copy_accepted_tokens": _mean(
                r.get("context_copy_accepted_tokens") for r in arm_rows
            ),
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
        control_net = _mean(r.get("ms_per_m4_window_net") for r in control)
        candidate_net = _mean(r.get("ms_per_m4_window_net") for r in candidate)
        control_raw_ms = _mean(r.get("ms_per_compiled_window") for r in control)
        candidate_raw_ms = _mean(
            r.get("ms_per_compiled_window") for r in candidate
        )
        paired.append(
            {
                "seed": seed,
                "control_runs": len(control),
                "candidate_runs": len(candidate),
                # PRIMARY: copy-corrected cycle time (lower is better).
                "control_mean_ms_per_m4_window_net": control_net,
                "candidate_mean_ms_per_m4_window_net": candidate_net,
                "delta_ms_per_m4_window_net": _delta(candidate_net, control_net),
                "delta_ms_per_m4_window_net_pct": _delta_pct(
                    candidate_net, control_net
                ),
                # Secondary: raw cycle time and throughput.
                "control_mean_ms_per_compiled_window": control_raw_ms,
                "candidate_mean_ms_per_compiled_window": candidate_raw_ms,
                "delta_ms_per_compiled_window": _delta(
                    candidate_raw_ms, control_raw_ms
                ),
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
                # Trajectory context: how much retrieval each arm drew.
                "control_mean_context_copy_rounds": _mean(
                    r.get("context_copy_rounds") for r in control
                ),
                "candidate_mean_context_copy_rounds": _mean(
                    r.get("context_copy_rounds") for r in candidate
                ),
                "control_mean_context_copy_accepted_tokens": _mean(
                    r.get("context_copy_accepted_tokens") for r in control
                ),
                "candidate_mean_context_copy_accepted_tokens": _mean(
                    r.get("context_copy_accepted_tokens") for r in candidate
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
                "delta_ms_per_m4_window_net": _delta(
                    candidate_row.get("ms_per_m4_window_net"),
                    control_row.get("ms_per_m4_window_net"),
                ),
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
    control_net = arms.get("A", {}).get("mean_ms_per_m4_window_net")
    candidate_net = arms.get("B", {}).get("mean_ms_per_m4_window_net")
    control_net_median = arms.get("A", {}).get("median_ms_per_m4_window_net")
    candidate_net_median = arms.get("B", {}).get("median_ms_per_m4_window_net")
    control_raw_ms = arms.get("A", {}).get("mean_ms_per_compiled_window")
    candidate_raw_ms = arms.get("B", {}).get("mean_ms_per_compiled_window")
    overall = {
        # -- PRIMARY: copy-corrected cycle time, ms per compiled M4 window.
        # A COST: negative delta = the candidate is faster.
        "control_mean_ms_per_m4_window_net": control_net,
        "candidate_mean_ms_per_m4_window_net": candidate_net,
        "delta_mean_ms_per_m4_window_net": _delta(candidate_net, control_net),
        "delta_mean_ms_per_m4_window_net_pct": _delta_pct(
            candidate_net, control_net
        ),
        "control_median_ms_per_m4_window_net": control_net_median,
        "candidate_median_ms_per_m4_window_net": candidate_net_median,
        "delta_median_ms_per_m4_window_net": _delta(
            candidate_net_median, control_net_median
        ),
        "paired_delta_mean_ms_per_m4_window_net": _mean(
            entry["delta_ms_per_m4_window_net"] for entry in paired
        ),
        "paired_delta_median_ms_per_m4_window_net": _median(
            entry["delta_ms_per_m4_window_net"] for entry in paired
        ),
        "paired_delta_mean_ms_per_m4_window_net_pct": _mean(
            entry["delta_ms_per_m4_window_net_pct"] for entry in paired
        ),
        "adjacent_delta_mean_ms_per_m4_window_net": _mean(
            entry["delta_ms_per_m4_window_net"] for entry in adjacent
        ),
        # -- Secondary: raw (uncorrected) cycle time.
        "control_mean_ms_per_compiled_window": control_raw_ms,
        "candidate_mean_ms_per_compiled_window": candidate_raw_ms,
        "delta_mean_ms_per_compiled_window": _delta(
            candidate_raw_ms, control_raw_ms
        ),
        "delta_mean_ms_per_compiled_window_pct": _delta_pct(
            candidate_raw_ms, control_raw_ms
        ),
        # -- Secondary: throughput (contaminated by retrieval yield).
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
        "copy_cost_model": {
            "copy_round_cost_s": float(copy_round_cost_s),
            "copy_token_cost_s": float(copy_token_cost_s),
            "statistic": (
                "ms_per_m4_window_net = (decode_elapsed_s "
                "- context_copy.rounds * copy_round_cost_s "
                "- context_copy.accepted_tokens * copy_token_cost_s) "
                "/ compiled_m4_calls * 1000"
            ),
            "primary_metric": "ms_per_m4_window_net",
        },
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
        "| # | Arm | Seed | Decode tok/s | Decode s | ms/window | ms/M4win net "
        "| tok/window | tok/M4win | ccopy rounds | ccopy accepted "
        "| Accepted by depth | Verify fwd s | Digest | Peak bytes | Ready C "
        "| Page cache |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda r: int(r["index"])):
        lines.append(
            "| {index} | {arm_name} ({arm}) | {seed} | {tok_s} | {decode_s} "
            "| {ms} | {ms_net} | {tpw} | {tpw_net} | {rounds} | {accepted} "
            "| {depths} | {vf} | {digest} | {peak:,} | {ready} "
            "| {regime} |".format(
                index=int(row["index"]),
                arm_name=row["arm_name"],
                arm=row["arm"],
                seed=int(row["seed"]),
                tok_s=_format(row["decode_tok_s"], ".6f"),
                decode_s=_format(row["decode_s"], ".6f"),
                ms=_format(row["ms_per_compiled_window"], ".4f"),
                ms_net=_format(row.get("ms_per_m4_window_net"), ".4f"),
                tpw=_format(row["tokens_per_window"], ".4f"),
                tpw_net=_format(row.get("tokens_per_m4_window"), ".4f"),
                rounds=_format(row.get("context_copy_rounds"), "d"),
                accepted=_format(row.get("context_copy_accepted_tokens"), "d"),
                depths=",".join(str(int(v)) for v in row["accepted_by_depth"]),
                vf=_format(row["verify_forward_s"], ".6f"),
                digest=str(row["digest"])[:12],
                peak=int(row["peak_bytes"]),
                ready=_format(row["ready_c"], ".4f"),
                regime=row["page_cache_regime"] or "n/a",
            )
        )

    cost_model = summary.get("copy_cost_model") or {}
    lines.append("")
    lines.append(
        "PRIMARY cycle-time metric: ms/M4win net = (decode_s - rounds*{c0} "
        "- accepted*{c1}) / compiled_m4_calls. It is a COST, so a NEGATIVE "
        "delta is the candidate winning.".format(
            c0=_format(cost_model.get("copy_round_cost_s"), ".6g"),
            c1=_format(cost_model.get("copy_token_cost_s"), ".6g"),
        )
    )
    lines.append("")
    lines.append(
        "| Seed | Control ms/M4win net | Candidate ms/M4win net "
        "| Delta ms | Delta % | Control tok/s | Candidate tok/s | Delta tok/s "
        "| Delta % | ccopy rounds A/B | Digests match |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| --- | --- |"
    )
    for entry in summary["per_seed"]:
        lines.append(
            "| {seed} | {cnet} | {knet} | {dnet} | {dnetpct} | {control} "
            "| {candidate} | {delta} | {pct} | {rounds} | {match} |".format(
                seed=entry["seed"],
                cnet=_format(
                    entry.get("control_mean_ms_per_m4_window_net"), ".4f"
                ),
                knet=_format(
                    entry.get("candidate_mean_ms_per_m4_window_net"), ".4f"
                ),
                dnet=_format(entry.get("delta_ms_per_m4_window_net"), "+.4f"),
                dnetpct=_format(
                    entry.get("delta_ms_per_m4_window_net_pct"), "+.4f"
                ),
                control=_format(entry["control_mean_decode_tok_s"], ".6f"),
                candidate=_format(entry["candidate_mean_decode_tok_s"], ".6f"),
                delta=_format(entry["delta_decode_tok_s"], "+.6f"),
                pct=_format(entry["delta_pct"], "+.4f"),
                rounds="{}/{}".format(
                    _format(
                        entry.get("control_mean_context_copy_rounds"), ".1f"
                    ),
                    _format(
                        entry.get("candidate_mean_context_copy_rounds"), ".1f"
                    ),
                ),
                match="yes" if entry["digests_match"] else "NO",
            )
        )

    overall = summary["overall"]
    lines.append("")
    lines.append(
        "PRIMARY: control {control} ms/M4win net, candidate {candidate} "
        "ms/M4win net, delta {delta} ms ({pct}%); paired delta mean {paired} "
        "ms, median {median} ms; adjacent-pair delta mean {adjacent} ms.".format(
            control=_format(
                overall.get("control_mean_ms_per_m4_window_net"), ".4f"
            ),
            candidate=_format(
                overall.get("candidate_mean_ms_per_m4_window_net"), ".4f"
            ),
            delta=_format(
                overall.get("delta_mean_ms_per_m4_window_net"), "+.4f"
            ),
            pct=_format(
                overall.get("delta_mean_ms_per_m4_window_net_pct"), "+.4f"
            ),
            paired=_format(
                overall.get("paired_delta_mean_ms_per_m4_window_net"), "+.4f"
            ),
            median=_format(
                overall.get("paired_delta_median_ms_per_m4_window_net"), "+.4f"
            ),
            adjacent=_format(
                overall.get("adjacent_delta_mean_ms_per_m4_window_net"), "+.4f"
            ),
        )
    )
    lines.append(
        "Secondary raw cycle time: control {control} ms/window, candidate "
        "{candidate} ms/window, delta {delta} ms ({pct}%).".format(
            control=_format(
                overall.get("control_mean_ms_per_compiled_window"), ".4f"
            ),
            candidate=_format(
                overall.get("candidate_mean_ms_per_compiled_window"), ".4f"
            ),
            delta=_format(
                overall.get("delta_mean_ms_per_compiled_window"), "+.4f"
            ),
            pct=_format(
                overall.get("delta_mean_ms_per_compiled_window_pct"), "+.4f"
            ),
        )
    )
    lines.append(
        "Secondary throughput (contaminated by how often the model quoted the "
        "prompt): control mean {control} tok/s, candidate mean {candidate} "
        "tok/s, delta {delta} tok/s ({pct}%).".format(
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
            "Extra raw driver flag for BOTH arms -- this moves the shared "
            "baseline (repeatable). Use the '=' form so argparse does not eat "
            "the leading dashes: --control-flag=--nax-verify"
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
        help=(
            "Override a construction-time MTPLX_* setting on BOTH arms "
            "(moves the shared baseline)."
        ),
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
        help=(
            "Non-MTPLX process env for BOTH arms "
            "(e.g. MLX_MAX_OPS_PER_BUFFER=...)."
        ),
    )
    parser.add_argument(
        "--candidate-extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Non-MTPLX process env for arm B only (adds to --control-extra-env).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Generated tokens per arm on BOTH arms "
            f"(default {DEFAULT_MAX_TOKENS}; "
            f"{PREFILL_ONLY_MAX_TOKENS} under --prefill-only)."
        ),
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help=(
            "Measure prefill cheaply: drop --max-tokens to "
            f"{PREFILL_ONLY_MAX_TOKENS} on both arms. The driver still records "
            "prefill_tok_s, prompt_eval_time_s and ttft_s per row; decode_tok_s "
            "from such a short window is diagnostic only. An explicit "
            "--max-tokens still wins."
        ),
    )
    parser.add_argument(
        "--copy-round-cost-s",
        type=float,
        default=DEFAULT_COPY_ROUND_COST_S,
        metavar="SECONDS",
        help=(
            "C0: fitted fixed cost of one context-copy block round, subtracted "
            "from decode_elapsed_s before the per-M4-window cycle time "
            f"(default {DEFAULT_COPY_ROUND_COST_S}). Re-fit and override this "
            "when a window changes the copy round's cost (e.g. a compiled copy "
            "round, or MTPLX_CONTEXT_COPY_K)."
        ),
    )
    parser.add_argument(
        "--copy-token-cost-s",
        type=float,
        default=DEFAULT_COPY_TOKEN_COST_S,
        metavar="SECONDS",
        help=(
            "C1: fitted marginal cost of one verified context-copy row "
            f"(default {DEFAULT_COPY_TOKEN_COST_S})."
        ),
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
        # The window owns this one now (--max-tokens / --prefill-only); an arm
        # flag repeating it would put two values on the driver command line.
        "--max-tokens",
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
    """Build both arms: control is the shared baseline, candidate adds to it.

    The ``--control-*`` options move the baseline for *both* arms; only the
    ``--candidate-*`` options separate arm B from arm A. Arm B taking the
    control flags too is what makes the comparison matched: a control flag
    that landed on arm A alone would be measured as a candidate difference
    with the opposite sign.
    """

    check_arm_flags(args.control_flag)
    check_arm_flags(args.candidate_flag)
    base = control_flags(
        resolve_max_tokens(args.max_tokens, args.prefill_only)
    )
    return {
        "A": {
            "flags": [*base, *args.control_flag],
            "candidate_env": merge_env_settings(
                CONTROL_CANDIDATE_ENV, args.control_env
            ),
            "extra_env": merge_env_settings((), args.control_extra_env),
        },
        "B": {
            "flags": [*base, *args.control_flag, *args.candidate_flag],
            "candidate_env": merge_env_settings(
                CONTROL_CANDIDATE_ENV, [*args.control_env, *args.candidate_env]
            ),
            "extra_env": merge_env_settings(
                args.control_extra_env, args.candidate_extra_env
            ),
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
        rows.append(
            extract_run_row(
                receipt,
                run,
                copy_round_cost_s=args.copy_round_cost_s,
                copy_token_cost_s=args.copy_token_cost_s,
            )
        )

    summary: dict[str, Any] | None = None
    table = ""
    if rows:
        summary = summarize(
            rows,
            copy_round_cost_s=args.copy_round_cost_s,
            copy_token_cost_s=args.copy_token_cost_s,
        )
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
        # The cost model every ``ms_per_m4_window_net`` in this receipt was
        # computed under, so a re-fit never silently reinterprets old numbers.
        "copy_cost_model": {
            "copy_round_cost_s": float(args.copy_round_cost_s),
            "copy_token_cost_s": float(args.copy_token_cost_s),
            "defaults": {
                "copy_round_cost_s": DEFAULT_COPY_ROUND_COST_S,
                "copy_token_cost_s": DEFAULT_COPY_TOKEN_COST_S,
            },
            "primary_metric": "ms_per_m4_window_net",
        },
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
