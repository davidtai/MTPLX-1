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

#: ``--prompt-tokens``: the measured cell's prompt length on BOTH arms.
#: Mirrors ``abba_driver.DEFAULT_PROMPT_TOKENS`` /
#: ``abba_driver.PROMPT_TOKEN_CHOICES``; checked here so a bad value fails
#: while the window is still planning rather than after an arm has taken
#: the GPU lock.  Only the prompt moves -- labels, receipt paths and the
#: summary table are identical at every length.
DEFAULT_PROMPT_TOKENS = 16_384
PROMPT_TOKEN_CHOICES = (
    1_024,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
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
# Acceptance model (the MTP accept coin and its conditional expectation)
# --------------------------------------------------------------------------
#
# ``accepted_by_depth`` is the REALISED accept coin: one Bernoulli draw per
# drafted token per depth.  Its per-window spread is the spread of a few
# hundred coin flips, which is wide enough to swamp the sub-percent effects
# this harness is trying to resolve.  ``generate_mtpk`` also accumulates
# ``accept_probability_sum_by_depth`` (``mtplx/generation.py``) -- the
# conditional expectation of that same coin, summed over the same draws.
# Divided by ``drafted_by_depth`` it is the mean accept PROBABILITY per depth:
# the same acceptance the counter estimates, with the coin-flip variance taken
# out, and therefore the lower-variance reading of the acceptance term.
#
# Both readings are carried on every row:
#
#     accept_rate_by_depth[d]               = accepted[d] / drafted[d]   (realised)
#     mean_accept_probability_by_depth[d]   = prob_sum[d] / drafted[d]   (expected)
#
# with the conditional ratios ``d2|d1`` and ``d3|d2`` -- how much of a depth's
# acceptance survives to the next depth -- as the RATIO OF RATES,
# ``rate[d+1] / rate[d]``, for each.  Ratio of rates, not of raw counts:
# ``drafted_by_depth`` is not identical across depths (the last cycle of a run
# routinely drafts one fewer at the deepest position), and the rate ratio stays
# the conditional continuation probability when it is not.
#
# A receipt written before the driver carried the sums has no expected reading
# at all.  It stays ``None`` and renders ``n/a``; it is never zero-filled, so
# "not observed" can never be read as "observed to be zero".

#: A primary-metric delta smaller than this, in per cent of the control, is
#: inside the noise this harness can resolve -- the "rounding class".  A
#: candidate that lands in it AND reproduces the control's exact token stream
#: on every seed did not engage; saying so on the window's own summary is what
#: this constant is for (2026-09-02, where that proof was three by-hand hashes
#: of ``response_text_head`` + ``response_text_tail``).
DEFAULT_ROUNDING_CLASS_PCT = 1.0


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


def _safe_ratio(
    numerator: float | None, denominator: float | None
) -> float | None:
    """``numerator / denominator``.

    ``None`` when either side was not recorded or the denominator is zero --
    an acceptance nothing was drafted for has no rate, and calling it 0.0
    would drag every aggregate that reads it toward zero.
    """

    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def per_depth_rates(
    numerators: Sequence[Any] | None, denominators: Sequence[Any] | None
) -> list[float | None] | None:
    """Per-depth ``numerator/denominator`` (acceptance rate or probability).

    ``None`` when the numerator was never recorded -- an unobserved acceptance
    is not an acceptance of zero.  Individual depths read ``None`` when nothing
    was drafted there.
    """

    if numerators is None or denominators is None:
        return None
    return [
        _safe_ratio(numerator, denominator)
        for numerator, denominator in zip(numerators, denominators)
    ]


def conditional_depth_ratios(
    rates: Sequence[float | None] | None,
) -> list[float | None] | None:
    """``[rate1/rate0, rate2/rate1, ...]`` -- the d2|d1, d3|d2 ratios.

    Entry 0 is ``d2|d1``, entry 1 is ``d3|d2``: the share of one depth's
    acceptance that survives into the next.
    """

    if rates is None:
        return None
    return [
        _safe_ratio(later, earlier)
        for earlier, later in zip(rates, list(rates)[1:])
    ]


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

    The acceptance term is carried in both of its readings: the realised
    ``accepted_by_depth`` coin and, when the receipt has it, the driver's
    ``accept_probability_sum_by_depth`` -- the conditional expectation of the
    same coin, which is the lower-variance estimate of the same acceptance (see
    the acceptance-model note at the top of this module).  A receipt without
    the sums keeps every expected field at ``None``.
    """

    rows = receipt.get("rows") or []
    if len(rows) != 1:
        raise ValueError(
            f"arm receipt must hold exactly one measured row, got {len(rows)}"
        )
    row = rows[0]
    # An arm LABEL is not unique across attempts, so a stale receipt sitting
    # at the expected path reads as this run's evidence unless the sequence is
    # checked.  The sequence IS unique per arm within a window.
    receipt_sequence = row.get("sequence", receipt.get("sequence"))
    if receipt_sequence is not None and int(receipt_sequence) != int(
        run["sequence"]
    ):
        raise ValueError(
            f"receipt is sequence {int(receipt_sequence)} but this arm is "
            f"sequence {int(run['sequence'])}: refusing to read another run's "
            "receipt as this one"
        )
    # sha256 over the raw uint32 id bytes (mtplx/fable_token_source.py).
    # Receipts written before that field existed fall back to the older
    # comma-joined digest; the comparison is only ever within one window, so
    # a consistent fallback is still a valid identity test.
    ids_digest = row.get("output_ids_sha256")
    ids_digest_source = "output_ids_sha256"
    if not ids_digest:
        ids_digest = row["response_token_sha256"]
        ids_digest_source = "response_token_sha256"
    token_sources = row.get("token_sources") or {}
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
    # -- acceptance: the realised coin and its conditional expectation ------
    accepted_by_depth = list(row["accepted_by_depth"])
    drafted_by_depth = list(row["drafted_by_depth"])
    prob_sums_raw = row.get("accept_probability_sum_by_depth")
    accept_probability_sum_by_depth = (
        None if prob_sums_raw is None else [float(value) for value in prob_sums_raw]
    )
    accept_rate_by_depth = per_depth_rates(accepted_by_depth, drafted_by_depth)
    mean_accept_probability_by_depth = per_depth_rates(
        accept_probability_sum_by_depth, drafted_by_depth
    )
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
        # -- acceptance (realised coin + lower-variance expectation) -------
        "accepted_by_depth": accepted_by_depth,
        "drafted_by_depth": drafted_by_depth,
        "accept_probability_sum_by_depth": accept_probability_sum_by_depth,
        # False on a receipt written before the driver carried the sums: the
        # expected columns are unobserved, not observed-and-zero.
        "accept_probability_recorded": accept_probability_sum_by_depth is not None,
        "accept_rate_by_depth": accept_rate_by_depth,
        "mean_accept_probability_by_depth": mean_accept_probability_by_depth,
        "conditional_accept_rate_by_depth": conditional_depth_ratios(
            accept_rate_by_depth
        ),
        "conditional_accept_probability_by_depth": conditional_depth_ratios(
            mean_accept_probability_by_depth
        ),
        "verify_forward_s": float(row["verify_forward_time_s"]),
        "draft_s": float(row["draft_time_s"]),
        "digest": str(row["response_token_sha256"]),
        # -- output identity + provenance ----------------------------------
        "output_ids_sha256": str(ids_digest),
        "output_ids_digest_source": ids_digest_source,
        "token_sources_available": bool(token_sources.get("available")),
        "token_sources_complete": bool(token_sources.get("complete")),
        "token_source_counts": dict(token_sources.get("counts") or {}),
        # -- which write produced this row ---------------------------------
        "run_id": row.get("run_id") or receipt.get("run_id"),
        "attempt": int(row.get("attempt") or receipt.get("attempt") or 1),
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


def _mean_by_depth(
    sequences: Iterable[Sequence[float | None] | None],
) -> list[float | None] | None:
    """Element-wise mean of per-depth vectors across runs.

    ``None`` when no run recorded the vector at all; a depth reads ``None``
    when no run recorded a value there.  Ragged vectors are tolerated (a run
    that stopped at a shallower depth contributes only the depths it has).
    """

    materialized = [list(seq) for seq in sequences if seq is not None]
    if not materialized:
        return None
    width = max((len(seq) for seq in materialized), default=0)
    result: list[float | None] = []
    for depth in range(width):
        values = [
            float(seq[depth])
            for seq in materialized
            if depth < len(seq) and seq[depth] is not None
        ]
        result.append(statistics.fmean(values) if values else None)
    return result


def _delta_by_depth(
    candidate: Sequence[float | None] | None,
    control: Sequence[float | None] | None,
) -> list[float | None] | None:
    """Element-wise ``candidate - control`` over two per-depth vectors."""

    if candidate is None or control is None:
        return None
    width = max(len(candidate), len(control))
    return [
        _delta(
            candidate[depth] if depth < len(candidate) else None,
            control[depth] if depth < len(control) else None,
        )
        for depth in range(width)
    ]


def _delta(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control is None:
        return None
    return candidate - control


def _delta_pct(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or not control:
        return None
    return 100.0 * (candidate - control) / control


def row_output_digest(row: Mapping[str, Any]) -> str:
    """The identity of a row's generated token stream.

    ``output_ids_sha256`` (sha256 over the raw uint32 ids) when the receipt
    carries it, the older comma-joined ``digest`` otherwise.  Never the text
    head/tail: on 2026-09-02 a duplicated subword sat outside both 600-char
    windows and two different outputs hashed the same by hand.
    """

    return str(row.get("output_ids_sha256") or row["digest"])


def output_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_delta_pct: float | None,
    rounding_class_pct: float = DEFAULT_ROUNDING_CLASS_PCT,
) -> dict[str, Any]:
    """Per-seed output identity and the non-engagement verdict.

    Two questions the window could not answer without opening receipts:

    * did every arm of a seed emit the SAME tokens (the exactness gate), and
    * did the candidate reproduce the control on every seed while its primary
      delta stayed inside the rounding class -- which is what an arm that
      never engaged looks like, and is indistinguishable from a real bit-exact
      win on timings alone.
    """

    seeds = sorted({int(row["seed"]) for row in rows})
    per_seed: list[dict[str, Any]] = []
    matches = 0
    paired_seeds = 0
    for seed in seeds:
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        control = [r for r in seed_rows if r["arm"] == "A"]
        candidate = [r for r in seed_rows if r["arm"] == "B"]
        digests = {row_output_digest(r) for r in seed_rows}
        control_digests = {row_output_digest(r) for r in control}
        candidate_digests = {row_output_digest(r) for r in candidate}
        matched = bool(
            control_digests
            and candidate_digests
            and control_digests == candidate_digests
            and len(control_digests) == 1
        )
        if control and candidate:
            paired_seeds += 1
            matches += int(matched)
        per_seed.append(
            {
                "seed": seed,
                "identical": len(digests) == 1,
                "candidate_matches_control": matched,
                "digests": sorted(digests),
            }
        )
    in_rounding_class = (
        primary_delta_pct is not None
        and abs(float(primary_delta_pct)) < float(rounding_class_pct)
    )
    return {
        "digest_key": "output_ids_sha256",
        "digest_sources": sorted(
            {
                str(row.get("output_ids_digest_source") or "digest")
                for row in rows
            }
        ),
        "per_seed": per_seed,
        "identical_per_seed": all(entry["identical"] for entry in per_seed),
        "candidate_matches_control_seeds": matches,
        "paired_seeds": paired_seeds,
        "rounding_class_pct": float(rounding_class_pct),
        "primary_delta_pct": (
            None if primary_delta_pct is None else float(primary_delta_pct)
        ),
        "in_rounding_class": bool(in_rounding_class),
        "non_engagement": bool(
            paired_seeds > 0 and matches == paired_seeds and in_rounding_class
        ),
        "token_sources_available": all(
            bool(row.get("token_sources_available")) for row in rows
        ),
        "token_sources_complete": all(
            bool(row.get("token_sources_complete")) for row in rows
        ),
    }


#: The three window headlines, as ``(name, label, overall delta key, overall
#: pct key, higher-is-better)``.  ``ms_per_m4_window_net`` is a COST; the
#: acceptance yield and tok/s are the other way up.
HEADLINES: tuple[tuple[str, str, str, str, bool], ...] = (
    (
        "cost",
        "ms/M4win net",
        "delta_mean_ms_per_m4_window_net",
        "delta_mean_ms_per_m4_window_net_pct",
        False,
    ),
    (
        "acceptance",
        "tok/M4win",
        "delta_mean_tokens_per_m4_window",
        "delta_mean_tokens_per_m4_window_pct",
        True,
    ),
    (
        "throughput",
        "tok/s",
        "delta_mean_decode_tok_s",
        "delta_mean_pct",
        True,
    ),
)


def headline_agreement(overall: Mapping[str, Any]) -> dict[str, Any]:
    """Which arm each headline favours, and whether they point the same way.

    Three readings of one window: the copy-corrected cycle time (a COST, so a
    negative delta is the candidate winning), the copy-corrected acceptance
    yield ``tok/M4win``, and raw ``tok/s``.  tok/s carries the retrieval yield
    -- how often the trajectory happened to quote the prompt -- that the two
    M4 headlines subtract out, so tok/s pointing the other way is the normal
    signature of trajectory luck rather than a contradiction.  A window whose
    headlines disagree has to say so on its own summary.
    """

    entries: dict[str, Any] = {}
    for name, label, delta_key, pct_key, higher_is_better in HEADLINES:
        delta = overall.get(delta_key)
        if delta is None:
            favours = None
        elif float(delta) == 0.0:
            favours = "tie"
        else:
            favours = (
                "candidate"
                if (float(delta) > 0.0) == higher_is_better
                else "control"
            )
        entries[name] = {
            "label": label,
            "metric": delta_key,
            "delta": None if delta is None else float(delta),
            "delta_pct": overall.get(pct_key),
            "higher_is_better": higher_is_better,
            "favours": favours,
        }
    resolved = {
        entry["favours"]
        for entry in entries.values()
        if entry["favours"] not in (None, "tie")
    }
    m4_verdicts = {
        entries["cost"]["favours"],
        entries["acceptance"]["favours"],
    } - {None, "tie"}
    throughput = entries["throughput"]["favours"]
    return {
        **entries,
        "agree": len(resolved) <= 1,
        "m4_headlines_agree": len(m4_verdicts) <= 1,
        "throughput_disagrees_with_m4_headlines": bool(
            throughput not in (None, "tie")
            and len(m4_verdicts) == 1
            and throughput not in m4_verdicts
        ),
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    copy_round_cost_s: float = DEFAULT_COPY_ROUND_COST_S,
    copy_token_cost_s: float = DEFAULT_COPY_TOKEN_COST_S,
    rounding_class_pct: float = DEFAULT_ROUNDING_CLASS_PCT,
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
        arm_accept_rate = _mean_by_depth(
            r.get("accept_rate_by_depth") for r in arm_rows
        )
        arm_accept_probability = _mean_by_depth(
            r.get("mean_accept_probability_by_depth") for r in arm_rows
        )
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
            "median_tokens_per_m4_window": _median(
                r.get("tokens_per_m4_window") for r in arm_rows
            ),
            # -- acceptance: realised coin and its expectation -------------
            "mean_accepted_by_depth": _mean_by_depth(
                r.get("accepted_by_depth") for r in arm_rows
            ),
            "mean_drafted_by_depth": _mean_by_depth(
                r.get("drafted_by_depth") for r in arm_rows
            ),
            "mean_accept_rate_by_depth": arm_accept_rate,
            "mean_accept_probability_by_depth": arm_accept_probability,
            "conditional_accept_rate_by_depth": conditional_depth_ratios(
                arm_accept_rate
            ),
            "conditional_accept_probability_by_depth": conditional_depth_ratios(
                arm_accept_probability
            ),
            "accept_probability_recorded": all(
                bool(r.get("accept_probability_recorded")) for r in arm_rows
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
        # The acceptance headline: copy-corrected tokens per compiled M4
        # window.  Higher is better -- it is the yield the accept coin buys.
        control_tok_win = _mean(r.get("tokens_per_m4_window") for r in control)
        candidate_tok_win = _mean(
            r.get("tokens_per_m4_window") for r in candidate
        )
        control_accept_rate = _mean_by_depth(
            r.get("accept_rate_by_depth") for r in control
        )
        candidate_accept_rate = _mean_by_depth(
            r.get("accept_rate_by_depth") for r in candidate
        )
        control_accept_prob = _mean_by_depth(
            r.get("mean_accept_probability_by_depth") for r in control
        )
        candidate_accept_prob = _mean_by_depth(
            r.get("mean_accept_probability_by_depth") for r in candidate
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
                # PRIMARY: copy-corrected acceptance yield (higher is better).
                "control_mean_tokens_per_m4_window": control_tok_win,
                "candidate_mean_tokens_per_m4_window": candidate_tok_win,
                "delta_tokens_per_m4_window": _delta(
                    candidate_tok_win, control_tok_win
                ),
                "delta_tokens_per_m4_window_pct": _delta_pct(
                    candidate_tok_win, control_tok_win
                ),
                # The acceptance term itself, in both readings.
                "control_mean_accept_rate_by_depth": control_accept_rate,
                "candidate_mean_accept_rate_by_depth": candidate_accept_rate,
                "delta_mean_accept_rate_by_depth": _delta_by_depth(
                    candidate_accept_rate, control_accept_rate
                ),
                "control_mean_accept_probability_by_depth": control_accept_prob,
                "candidate_mean_accept_probability_by_depth": (
                    candidate_accept_prob
                ),
                "delta_mean_accept_probability_by_depth": _delta_by_depth(
                    candidate_accept_prob, control_accept_prob
                ),
                "control_conditional_accept_rate_by_depth": (
                    conditional_depth_ratios(control_accept_rate)
                ),
                "candidate_conditional_accept_rate_by_depth": (
                    conditional_depth_ratios(candidate_accept_rate)
                ),
                "control_conditional_accept_probability_by_depth": (
                    conditional_depth_ratios(control_accept_prob)
                ),
                "candidate_conditional_accept_probability_by_depth": (
                    conditional_depth_ratios(candidate_accept_prob)
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
                "delta_tokens_per_m4_window": _delta(
                    candidate_row.get("tokens_per_m4_window"),
                    control_row.get("tokens_per_m4_window"),
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
    control_tok_win = arms.get("A", {}).get("mean_tokens_per_m4_window")
    candidate_tok_win = arms.get("B", {}).get("mean_tokens_per_m4_window")
    control_tok_win_median = arms.get("A", {}).get(
        "median_tokens_per_m4_window"
    )
    candidate_tok_win_median = arms.get("B", {}).get(
        "median_tokens_per_m4_window"
    )
    control_accept_rate = arms.get("A", {}).get("mean_accept_rate_by_depth")
    candidate_accept_rate = arms.get("B", {}).get("mean_accept_rate_by_depth")
    control_accept_prob = arms.get("A", {}).get(
        "mean_accept_probability_by_depth"
    )
    candidate_accept_prob = arms.get("B", {}).get(
        "mean_accept_probability_by_depth"
    )
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
        # -- PRIMARY: copy-corrected acceptance yield, tokens per compiled
        # M4 window.  A YIELD: a positive delta is the candidate winning.
        "control_mean_tokens_per_m4_window": control_tok_win,
        "candidate_mean_tokens_per_m4_window": candidate_tok_win,
        "delta_mean_tokens_per_m4_window": _delta(
            candidate_tok_win, control_tok_win
        ),
        "delta_mean_tokens_per_m4_window_pct": _delta_pct(
            candidate_tok_win, control_tok_win
        ),
        "control_median_tokens_per_m4_window": control_tok_win_median,
        "candidate_median_tokens_per_m4_window": candidate_tok_win_median,
        "delta_median_tokens_per_m4_window": _delta(
            candidate_tok_win_median, control_tok_win_median
        ),
        "paired_delta_mean_tokens_per_m4_window": _mean(
            entry["delta_tokens_per_m4_window"] for entry in paired
        ),
        "paired_delta_median_tokens_per_m4_window": _median(
            entry["delta_tokens_per_m4_window"] for entry in paired
        ),
        "paired_delta_mean_tokens_per_m4_window_pct": _mean(
            entry["delta_tokens_per_m4_window_pct"] for entry in paired
        ),
        "adjacent_delta_mean_tokens_per_m4_window": _mean(
            entry["delta_tokens_per_m4_window"] for entry in adjacent
        ),
        # -- The acceptance term behind that yield, in both readings.  The
        # expected one (accept_probability_sum_by_depth / drafted) is the same
        # acceptance without the coin-flip variance; it is None on receipts
        # written before the driver carried the sums.
        "control_mean_accept_rate_by_depth": control_accept_rate,
        "candidate_mean_accept_rate_by_depth": candidate_accept_rate,
        "delta_mean_accept_rate_by_depth": _delta_by_depth(
            candidate_accept_rate, control_accept_rate
        ),
        "control_mean_accept_probability_by_depth": control_accept_prob,
        "candidate_mean_accept_probability_by_depth": candidate_accept_prob,
        "delta_mean_accept_probability_by_depth": _delta_by_depth(
            candidate_accept_prob, control_accept_prob
        ),
        "control_conditional_accept_rate_by_depth": conditional_depth_ratios(
            control_accept_rate
        ),
        "candidate_conditional_accept_rate_by_depth": conditional_depth_ratios(
            candidate_accept_rate
        ),
        "control_conditional_accept_probability_by_depth": (
            conditional_depth_ratios(control_accept_prob)
        ),
        "candidate_conditional_accept_probability_by_depth": (
            conditional_depth_ratios(candidate_accept_prob)
        ),
        "accept_probability_recorded": all(
            bool(row.get("accept_probability_recorded")) for row in rows
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
    identity = output_identity(
        rows,
        primary_delta_pct=overall.get("delta_mean_ms_per_m4_window_net_pct"),
        rounding_class_pct=rounding_class_pct,
    )
    return {
        "arms": arms,
        "per_seed": paired,
        "adjacent_pairs": adjacent,
        "overall": overall,
        "output_identity": identity,
        "headline_agreement": headline_agreement(overall),
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
            "primary_cost_metric": "ms_per_m4_window_net",
            "primary_acceptance_metric": "tokens_per_m4_window",
        },
    }


def _format(value: Any, spec: str) -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def _format_by_depth(values: Sequence[Any] | None, spec: str) -> str:
    """A per-depth vector as ``d1,d2,d3``; ``n/a`` when it was not recorded."""

    if values is None:
        return "n/a"
    if not values:
        return "-"
    return ",".join(_format(value, spec) for value in values)


def render_markdown(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> str:
    """Render the per-run table plus the paired-delta table."""

    lines = [
        "| # | Seq | Arm | Seed | Decode tok/s | Decode s | ms/window "
        "| ms/M4win net "
        "| tok/window | tok/M4win | ccopy rounds | ccopy accepted "
        "| Accepted by depth | Accept rate by depth "
        "| Mean accept prob by depth | Cond accept realised "
        "| Cond accept expected "
        "| Verify fwd s | Output ids sha256 | Peak bytes "
        "| Ready C | Page cache |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | --- | --- | --- | --- | --- "
        "| ---: | --- | ---: | ---: | --- |",
    ]
    # Keyed by SEQUENCE, not label: an arm label repeats on every attempt at
    # that arm, and reading the wrong attempt's row is exactly the failure
    # this column exists to make impossible.
    for row in sorted(rows, key=lambda r: int(r["sequence"])):
        lines.append(
            "| {index} | {sequence} | {arm_name} ({arm}) | {seed} | {tok_s} "
            "| {decode_s} "
            "| {ms} | {ms_net} | {tpw} | {tpw_net} | {rounds} | {accepted} "
            "| {depths} | {rates} | {probs} | {cond_real} | {cond_exp} "
            "| {vf} | {digest} | {peak:,} | {ready} "
            "| {regime} |".format(
                index=int(row["index"]),
                sequence=int(row["sequence"]),
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
                rates=_format_by_depth(row.get("accept_rate_by_depth"), ".4f"),
                probs=_format_by_depth(
                    row.get("mean_accept_probability_by_depth"), ".4f"
                ),
                cond_real=_format_by_depth(
                    row.get("conditional_accept_rate_by_depth"), ".4f"
                ),
                cond_exp=_format_by_depth(
                    row.get("conditional_accept_probability_by_depth"), ".4f"
                ),
                vf=_format(row["verify_forward_s"], ".6f"),
                digest=row_output_digest(row)[:12],
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
    lines.append(
        "PRIMARY acceptance metric: tok/M4win = (generated_tokens - ccopy "
        "accepted) / compiled_m4_calls -- the copy-corrected yield of the "
        "accept coin. It is a YIELD, so a POSITIVE delta is the candidate "
        "winning. Mean accept prob by depth is that same coin's conditional "
        "expectation (accept_probability_sum_by_depth / drafted_by_depth), "
        "the lower-variance reading; n/a on receipts written before the "
        "driver carried the sums."
    )
    lines.append("")
    lines.append(
        "| Seed | Control ms/M4win net | Candidate ms/M4win net "
        "| Delta ms | Delta % "
        "| Control tok/M4win | Candidate tok/M4win | Delta tok/M4win "
        "| Delta % "
        "| Control tok/s | Candidate tok/s | Delta tok/s "
        "| Delta % | ccopy rounds A/B | Digests match |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | ---: "
        "| --- | --- |"
    )
    for entry in summary["per_seed"]:
        lines.append(
            "| {seed} | {cnet} | {knet} | {dnet} | {dnetpct} "
            "| {ctok} | {ktok} | {dtok} | {dtokpct} | {control} "
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
                ctok=_format(
                    entry.get("control_mean_tokens_per_m4_window"), ".4f"
                ),
                ktok=_format(
                    entry.get("candidate_mean_tokens_per_m4_window"), ".4f"
                ),
                dtok=_format(entry.get("delta_tokens_per_m4_window"), "+.4f"),
                dtokpct=_format(
                    entry.get("delta_tokens_per_m4_window_pct"), "+.4f"
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
    # One line, three readings: the cost headline (ms/M4win net), the
    # acceptance headline (tok/M4win) and raw tok/s.  Reading only the third
    # is what made a real cycle-time win look like nothing on 2026-09-02.
    lines.append(
        "PRIMARY: control {control} ms/M4win net, candidate {candidate} "
        "ms/M4win net, delta {delta} ms ({pct}%); paired delta mean {paired} "
        "ms, median {median} ms; adjacent-pair delta mean {adjacent} ms. "
        "ACCEPTANCE: control {ctok} tok/M4win, candidate {ktok} tok/M4win, "
        "delta {dtok} tok ({tokpct}%); paired delta mean {ptok} tok, median "
        "{mtok} tok; adjacent-pair delta mean {atok} tok. THROUGHPUT: delta "
        "{dtoks} tok/s ({tokspct}%).".format(
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
            ctok=_format(
                overall.get("control_mean_tokens_per_m4_window"), ".4f"
            ),
            ktok=_format(
                overall.get("candidate_mean_tokens_per_m4_window"), ".4f"
            ),
            dtok=_format(overall.get("delta_mean_tokens_per_m4_window"), "+.4f"),
            tokpct=_format(
                overall.get("delta_mean_tokens_per_m4_window_pct"), "+.4f"
            ),
            ptok=_format(
                overall.get("paired_delta_mean_tokens_per_m4_window"), "+.4f"
            ),
            mtok=_format(
                overall.get("paired_delta_median_tokens_per_m4_window"), "+.4f"
            ),
            atok=_format(
                overall.get("adjacent_delta_mean_tokens_per_m4_window"), "+.4f"
            ),
            dtoks=_format(overall.get("delta_mean_decode_tok_s"), "+.6f"),
            tokspct=_format(overall.get("delta_mean_pct"), "+.4f"),
        )
    )
    agreement = summary.get("headline_agreement") or headline_agreement(overall)
    if not agreement.get("agree", True):
        lines.append(
            "HEADLINES DISAGREE: {readings}{tail}".format(
                readings="; ".join(
                    "{label} favours {favours} ({delta}{unit}, {pct}%)".format(
                        label=agreement[name]["label"],
                        favours=agreement[name]["favours"] or "n/a",
                        delta=_format(agreement[name]["delta"], "+.4f"),
                        unit=unit,
                        pct=_format(agreement[name]["delta_pct"], "+.4f"),
                    )
                    for name, unit in (
                        ("cost", " ms"),
                        ("acceptance", " tok"),
                        ("throughput", " tok/s"),
                    )
                ),
                tail=(
                    " -- tok/s carries the retrieval yield (how often the "
                    "trajectory quoted the prompt) that both M4 headlines "
                    "subtract out, so read the M4 headlines."
                    if agreement.get("throughput_disagrees_with_m4_headlines")
                    else " -- the cost and acceptance headlines split, so "
                    "this arm moved cycle time and accepted yield in "
                    "opposite directions."
                ),
            )
        )
    lines.append(
        "Acceptance by depth: control realised {crate} (expected {cprob}), "
        "candidate realised {krate} (expected {kprob}); conditional "
        "d2|d1,d3|d2 realised {ccond} vs {kcond}, expected {cpcond} vs "
        "{kpcond}.".format(
            crate=_format_by_depth(
                overall.get("control_mean_accept_rate_by_depth"), ".4f"
            ),
            cprob=_format_by_depth(
                overall.get("control_mean_accept_probability_by_depth"), ".4f"
            ),
            krate=_format_by_depth(
                overall.get("candidate_mean_accept_rate_by_depth"), ".4f"
            ),
            kprob=_format_by_depth(
                overall.get("candidate_mean_accept_probability_by_depth"),
                ".4f",
            ),
            ccond=_format_by_depth(
                overall.get("control_conditional_accept_rate_by_depth"), ".4f"
            ),
            kcond=_format_by_depth(
                overall.get("candidate_conditional_accept_rate_by_depth"),
                ".4f",
            ),
            cpcond=_format_by_depth(
                overall.get("control_conditional_accept_probability_by_depth"),
                ".4f",
            ),
            kpcond=_format_by_depth(
                overall.get(
                    "candidate_conditional_accept_probability_by_depth"
                ),
                ".4f",
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

    # -- the two lines that would have saved 2026-09-02 --------------------
    # Both read output_ids_sha256 (sha256 over the raw uint32 generated ids),
    # never response_text_head/tail: 600 characters at each end of a 1,024
    # token completion is not the completion.
    identity = summary.get("output_identity") or output_identity(
        rows, primary_delta_pct=overall.get("delta_mean_ms_per_m4_window_net_pct")
    )
    mismatched = [
        entry["seed"] for entry in identity["per_seed"] if not entry["identical"]
    ]
    lines.append(
        "outputs identical per seed: {verdict} ({key}{fallback}{detail})".format(
            verdict="yes" if identity["identical_per_seed"] else "no",
            key=identity["digest_key"],
            fallback=(
                ""
                if identity["digest_sources"] == ["output_ids_sha256"]
                else " via " + "/".join(identity["digest_sources"])
            ),
            detail=(
                "; seeds that differ: "
                + ",".join(str(seed) for seed in mismatched)
                if mismatched
                else ""
            ),
        )
    )
    matched = int(identity["candidate_matches_control_seeds"])
    paired_seeds = int(identity["paired_seeds"])
    engagement = (
        "candidate == control on {matched}/{total} seeds "
        "(identical output ids)".format(matched=matched, total=paired_seeds)
    )
    if identity["non_engagement"]:
        engagement += (
            "; primary delta {delta} is inside the +/-{band:.2f}% rounding "
            "class -- NON-ENGAGEMENT: this arm reproduced the control exactly "
            "and moved nothing measurable.".format(
                delta=_format(identity["primary_delta_pct"], "+.4f") + "%",
                band=identity["rounding_class_pct"],
            )
        )
    elif paired_seeds and matched == paired_seeds:
        engagement += (
            "; primary delta {delta} is OUTSIDE the +/-{band:.2f}% rounding "
            "class, so this is a bit-exact change, not an inert arm.".format(
                delta=_format(identity["primary_delta_pct"], "+.4f") + "%",
                band=identity["rounding_class_pct"],
            )
        )
    else:
        engagement += "."
    lines.append(engagement)
    if not identity["token_sources_available"]:
        lines.append(
            "Per-token source column: NOT recorded on every row -- provenance "
            "for this window is unknown, not observed-and-empty."
        )
    elif not identity["token_sources_complete"]:
        lines.append(
            "Per-token source column: recorded but INCOMPLETE on at least one "
            "row -- a lane committed tokens through a site the recorder does "
            "not cover (mtplx/fable_token_source.py)."
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
            "Raw process env for BOTH arms, applied before the mlx import "
            "(e.g. MLX_MAX_OPS_PER_BUFFER=...). MTPLX_* keys are refused here "
            "except MTPLX_FABLE_* and the raw-environment allowlist "
            "(MTPLX_CONTEXT_COPY_K, MTPLX_CONTEXT_COPY_PROBATION_K, "
            "MTPLX_SESSION_BANK_MAX_BYTES), which are "
            "read straight off os.environ and are NOT profile overrides."
        ),
    )
    parser.add_argument(
        "--candidate-extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Raw process env for arm B only (adds to --control-extra-env); "
            "same namespace rules. This is the channel for the block-cap "
            "recipe: --candidate-extra-env MTPLX_CONTEXT_COPY_K=48"
        ),
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
        "--prompt-tokens",
        type=int,
        choices=PROMPT_TOKEN_CHOICES,
        default=DEFAULT_PROMPT_TOKENS,
        metavar="N",
        help=(
            "Prompt length of the measured cell on BOTH arms, in tokens "
            f"(default {DEFAULT_PROMPT_TOKENS}; one of "
            f"{', '.join(str(value) for value in PROMPT_TOKEN_CHOICES)}). "
            "The default reproduces the pinned production prompt byte for "
            "byte; any other value is built to exactly N tokens from the "
            "same SHA-pinned fixtures. Labels, receipt paths and the "
            "summary table are unchanged. The warm-up cell that "
            "--prefill-only implies uses the same N."
        ),
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help=(
            "Measure prefill cheaply: drop --max-tokens to "
            f"{PREFILL_ONLY_MAX_TOKENS} on both arms, and run the driver's "
            "unmeasured graph warm-up cell first (--warm-graph) so the cold "
            "first prefill chunk does not land inside the measurement. The "
            "driver still records prefill_tok_s, prompt_eval_time_s and ttft_s "
            "per row; decode_tok_s from such a short window is diagnostic "
            "only. An explicit --max-tokens still wins."
        ),
    )
    parser.add_argument(
        "--warm-graph",
        action="store_true",
        help=(
            "Force the driver's unmeasured graph warm-up cell on every arm "
            "(implied by --prefill-only)."
        ),
    )
    parser.add_argument(
        "--no-warm-graph",
        action="store_true",
        help=(
            "Suppress the warm-up cell that --prefill-only otherwise implies, "
            "e.g. to measure the cold first chunk on purpose."
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
        "--rounding-class-pct",
        type=float,
        default=DEFAULT_ROUNDING_CLASS_PCT,
        metavar="PCT",
        help=(
            "Primary-metric delta below which a candidate that reproduced the "
            "control's exact token stream on every seed is reported as "
            f"NON-ENGAGEMENT (default {DEFAULT_ROUNDING_CLASS_PCT}%%)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned arm command lines and exit without the GPU.",
    )
    return parser


def common_driver_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--source",
        str(args.source),
        # Always explicit, on every arm, even at the default: a window
        # whose printed command line omits the prompt length is a window
        # whose receipts cannot be re-run from what it recorded.
        "--prompt-tokens",
        str(int(args.prompt_tokens)),
    ]
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
    if warm_graph_enabled(args):
        flags.append("--warm-graph")
    return flags


def warm_graph_enabled(args: argparse.Namespace) -> bool:
    """Whether the arms run the driver's unmeasured graph warm-up cell.

    ``--prefill-only`` implies it.  A prefill-only arm measures ONE prefill
    of ``--prompt-tokens`` tokens in a fresh process (the warm-up cell is
    the same cell, so it is the same length), and on 2026-09-01 that
    first chunk was bimodal -- ~1.9 s or ~4.4 s -- on control and
    candidate alike, in lockstep
    with the throughput of the driver's own 29.8 GiB ``--prewarm-ngram-table``
    read (12/12 arms in the w22 window; +2.2 s median on prompt_eval_time_s
    across 20 window-arm groups).  That is residency state left by the
    prewarm, not the candidate, and a +-2.4 s term on a single measured chunk
    swamps the effects these windows look for.  The unmeasured cell pays it
    first; the cold number survives as ``first_chunk_cold_s``.
    ``--no-warm-graph`` opts back out.
    """

    if args.no_warm_graph:
        return False
    return bool(args.warm_graph or args.prefill_only)


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
        # Ditto: the window decides warm-up from --prefill-only / --warm-graph
        # / --no-warm-graph, and store_true repeated is a silent no-op that
        # would make the printed command line disagree with the receipt.
        "--warm-graph",
        # The window owns the prompt length too (--prompt-tokens): both
        # arms must measure the same prompt or the pairing is void.
        "--prompt-tokens",
    }
)


#: Mirrors ``scripts/fable/abba_driver.RAW_ENV_MTPLX_KEYS``.  These MTPLX_*
#: settings are read with a bare ``os.environ.get`` at their use site rather
#: than through ``mtplx.profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS``, so they ride
#: ``--control-extra-env`` / ``--candidate-extra-env`` (the driver's raw
#: ``--env``) and NOT ``--control-env`` / ``--candidate-env``.
#:
#: Checked here as well as in the driver so a mis-routed key fails while the
#: window is still planning, instead of after an arm has taken the GPU lock and
#: loaded the model.
RAW_ENV_MTPLX_KEYS = frozenset(
    {
        "MTPLX_CONTEXT_COPY_K",
        "MTPLX_CONTEXT_COPY_PROBATION_K",
        "MTPLX_SESSION_BANK_MAX_BYTES",
    }
)


def _is_raw_env_mtplx_key(key: str) -> bool:
    return key.startswith("MTPLX_FABLE_") or key in RAW_ENV_MTPLX_KEYS


def check_env_settings(settings: Sequence[str], *, flag: str, mtplx: bool) -> None:
    """Reject KEY=VALUE settings the driver would refuse after model load.

    ``mtplx=True`` is the construction-time override channel (``--*-env``):
    keys must be MTPLX_* and must not be one of the raw-environment settings.
    ``mtplx=False`` is the process-environment channel (``--*-extra-env``):
    MTPLX_* keys are refused unless they are MTPLX_FABLE_* or allowlisted.
    """

    for setting in settings:
        if "=" not in setting:
            raise ValueError(f"expected KEY=VALUE, got {setting!r}")
        key, value = setting.split("=", 1)
        if not key or not value:
            raise ValueError(f"expected KEY=VALUE, got {setting!r}")
        raw = _is_raw_env_mtplx_key(key)
        if mtplx:
            if raw:
                raise ValueError(
                    f"{key} is a raw process-environment setting; pass it with "
                    f"--control-extra-env / --candidate-extra-env, not {flag}"
                )
            if not key.startswith("MTPLX_"):
                raise ValueError(f"{flag} keys must start with MTPLX_: {setting!r}")
        elif key.startswith("MTPLX_") and not raw:
            raise ValueError(
                f"{key} is a construction-time override; pass it with "
                f"--control-env / --candidate-env, not {flag} "
                f"(raw-environment allowlist: "
                f"{', '.join(sorted(RAW_ENV_MTPLX_KEYS))})"
            )


def check_prompt_tokens(args: argparse.Namespace) -> None:
    """Mirror the driver's ``--prompt-tokens`` fail-closed rule, at plan time.

    The PR391 reference rows were recorded against the pinned 16,384-token
    production prompt, so token parity at any other length is a guaranteed
    false drift.  Catching it here means the window never takes the GPU.
    """

    prompt_tokens = int(getattr(args, "prompt_tokens", DEFAULT_PROMPT_TOKENS))
    if prompt_tokens not in PROMPT_TOKEN_CHOICES:
        raise ValueError(
            f"--prompt-tokens must be one of "
            f"{', '.join(str(value) for value in PROMPT_TOKEN_CHOICES)}, "
            f"got {prompt_tokens}"
        )
    if prompt_tokens != DEFAULT_PROMPT_TOKENS and getattr(
        args, "require_reference_token_parity", False
    ):
        raise ValueError(
            "--require-reference-token-parity is only defined at "
            f"--prompt-tokens {DEFAULT_PROMPT_TOKENS}; the reference rows "
            "were recorded against the pinned production prompt"
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

    check_prompt_tokens(args)
    check_arm_flags(args.control_flag)
    check_arm_flags(args.candidate_flag)
    check_env_settings(args.control_env, flag="--control-env", mtplx=True)
    check_env_settings(args.candidate_env, flag="--candidate-env", mtplx=True)
    check_env_settings(
        args.control_extra_env, flag="--control-extra-env", mtplx=False
    )
    check_env_settings(
        args.candidate_extra_env, flag="--candidate-extra-env", mtplx=False
    )
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
        receipt_path = out_dir / receipt_name(args.label_prefix, run)
        # A receipt already at this path is a previous ATTEMPT at this arm.
        # The driver replaces it and stamps `attempt`; recording the fact here
        # means the window says so too, rather than the reader discovering it
        # from an mtime.
        preexisting = receipt_path.exists()
        arm_started = time.time()
        completed = subprocess.run(command, env=arm_environment, check=False)
        arm_wall = time.time() - arm_started
        record = {
            **run,
            "command": command,
            "returncode": completed.returncode,
            "arm_wall_s": arm_wall,
            "receipt_preexisting": preexisting,
            "receipt_path": str(receipt_path),
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
        record["run_id"] = receipt.get("run_id")
        record["attempt"] = receipt.get("attempt")
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
            rounding_class_pct=args.rounding_class_pct,
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
        "prompt_tokens": int(args.prompt_tokens),
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
        "rounding_class_pct": float(args.rounding_class_pct),
        "arms": arm_records,
        "rows": rows,
        # The same rows, keyed by the one field that is unique per arm across
        # re-runs.  ``rows`` stays a list so existing readers keep working.
        "rows_by_sequence": {str(int(row["sequence"])): row for row in rows},
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
