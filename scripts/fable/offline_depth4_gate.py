#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# NO GPU.  Pure NumPy.  This script never imports mlx and never touches the
# GPU exclusive lock -- it scores logged K20 rows on the CPU.
# ---------------------------------------------------------------------------
"""Score L Sec.D's confidence-gated depth 4 from a ``MTPLX_FABLE_DEPTH4_PROBE`` log.

The one number this exists to produce
-------------------------------------
``L-fable-decode-ideas.md`` Sec.D sizes a 4th draft step dispatched only on
windows where the drafter is confident.  Every term in that table is measured
except one, which was **extrapolated** by continuing the observed depth decay::

    a4est = max(0, a3 - (a2 - a3))

and the whole program hangs off it: at ``q(x_3) > 0.8`` the gate fires on 30%
of windows, 52% of those accepted all three drafts, and the extra row is worth
``+0.147 tok/window`` for ``+0.89 ms`` -- **+3.5%** -- *if* ``alpha_4`` on the
gated windows is at least 0.75.  Ungated depth 4 is -0.5%, exactly as H found.

This script replaces ``a4est`` with a measurement.

Why no M=5 verify graph is needed
---------------------------------
After a normal M4 cycle whose three drafts were all accepted, the target's
**bonus row** is ``p(. | primary, d_1, d_2, d_3)`` -- which is precisely the
distribution a fourth draft would have been verified against.  So the probe
(``mtplx/fable_depth4_probe.py``) runs one extra ``draft_mtp`` at depth 4 on
exactly those windows, shapes it with the draft sampler, and logs the row.
Here that row is paired with target row 3 and scored::

    alpha_4 = sum_x min(p_3(x), q_4(x))

``sum min(p, q)`` is ``E_{x ~ q}[min(1, p(x)/q(x))]`` -- the expected
Leviathan-Chen acceptance of a draw from ``q`` -- so it estimates the same
quantity as the per-token ``min(1, p/q)`` the receipts report, with the draw
integrated out.  The ladder for depths 1..3 is reported in the **same** form,
so the four columns are comparable and the printed table lines up column for
column with ``L_gate_out.txt``.

What the estimator does and does not include
--------------------------------------------
``Delta T = P(G) . P(all 3 accepted | G) . alpha_4|G``.

* ``P(G)`` and ``P(all 3 | G)`` come from **every** scoreable window, probed or
  not -- which is why the log records ``gate_q`` everywhere and not only where
  the probe fired.
* ``alpha_4|G`` comes from the probed windows, which are by construction the
  all-accept ones.  So the conditional is ``alpha_4 | G, all 3 accepted``,
  which is the term the product needs.
* A 5th token is worth nothing unless the 4th is accepted, and depth 5 is not
  drafted, so ``Delta T`` stops at one extra expected token.  It is a
  lower bound on the gated arm in that narrow sense and an upper bound in a
  wider one: it assumes the M=5 verify row is exact and free of acceptance
  drift beyond what ``alpha_4`` already shows.

Cost model
----------
``Delta cost = P(G) . (draft_step_ms + row_ms)``, charged only on gated
windows.  Two row costs are reported because the ledger has two:

``--row-ms 1.8``
    H Sec.2.4's marginal verify row, the number L Sec.D's table used.
``--row-ms 1.4``
    ``K-novel-decode-ideas.md``'s fit for a **compiled** fixed-width row
    (620 MB/row of PLE + dense at 614 GB/s = 1.01 ms, plus ~0.3 ms of GDN
    recurrence and ~0.17 ms of QSA indexer).  The eager 3.64 ms/row is what an
    uncompiled M=5 lane would pay, which is why widening the kernels is on the
    critical path of the program and not an afterthought.

Neither is measured on an M=5 graph, because no M=5 graph exists yet.  That is
the point: this script decides whether building one is worth 5-8 days.

Usage::

    MTPLX_FABLE_DEPTH4_PROBE=1 MTPLX_FABLE_K20_LOG=/path/d4.npz <benchmark>
    python scripts/fable/offline_depth4_gate.py /path/d4.npz --ms-per-window 38.7
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import numpy as np

from scripts.fable.offline_block_verification import (
    LAYOUT_STOCK,
    LAYOUT_STOCK_BV,
    load_log,
    prepared_row,
)

#: The L Sec.D go/no-go: the gate it recommends and the bar alpha_4 must clear
#: on it for the M=5 lane to be worth building.
GO_GATE = 0.8
GO_ALPHA = 0.75

#: Defaults from L Sec.3 (the retained 72.0 tok/s frame in M4-window units) and
#: L Sec.D's cost row.
MS_PER_WINDOW = 38.7
DRAFT_STEP_MS = 1.2
ROW_MS = (1.8, 1.4)
THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


def overlap(
    p_ids: np.ndarray, p_probs: np.ndarray, q_ids: np.ndarray, q_probs: np.ndarray
) -> float:
    """``sum_x min(p(x), q(x))`` over the union of two prepared rows.

    Both rows arrive from :func:`prepared_row`: id-ascending, unique, positive,
    renormalised.  ``np.intersect1d`` is therefore exact, and every id present
    in only one row contributes ``min(., 0) = 0``.
    """

    _shared, p_at, q_at = np.intersect1d(
        p_ids, q_ids, assume_unique=True, return_indices=True
    )
    if p_at.size == 0:
        return 0.0
    return float(np.sum(np.minimum(p_probs[p_at], q_probs[q_at]), dtype=np.float64))


def log_spec(log: dict[str, np.ndarray]) -> dict[str, Any]:
    """Depth, row count and layout, read off the arrays rather than assumed."""

    layout = str(log["layout"]) if "layout" in log else LAYOUT_STOCK
    return {
        "layout": layout,
        "cycles": int(log["draft_tokens"].shape[0]),
        "depth": int(log["draft_tokens"].shape[1]),
        "target_rows": int(log["target_ids"].shape[1]),
    }


def require_probe_columns(log: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fail loudly and specifically when the log was not written by a probe run."""

    spec = log_spec(log)
    if spec["layout"] not in {LAYOUT_STOCK, LAYOUT_STOCK_BV}:
        raise SystemExit(
            f"depth-4 gate: layout {spec['layout']!r} is not a stock-lane log. "
            "The probe hooks the stock native-MTP accept loop's all-accept "
            "branch; the PR391 device lane has no hook."
        )
    missing = [
        key
        for key in ("gate_q", "probe_valid", "probe_ids", "probe_probs")
        if key not in log
    ]
    if missing:
        raise SystemExit(
            f"depth-4 gate: this log has no {missing}. Re-run the capture with "
            "MTPLX_FABLE_DEPTH4_PROBE=1 alongside MTPLX_FABLE_K20_LOG; the "
            "probe columns are written only when the probe recorded a window."
        )
    if spec["target_rows"] <= spec["depth"]:
        raise SystemExit(
            "depth-4 gate: this log has no bonus target row, which is the "
            "distribution alpha_4 is measured against."
        )
    return spec


def score(
    log: dict[str, np.ndarray],
    *,
    thresholds: Sequence[float] = THRESHOLDS,
    limit: int | None = None,
) -> dict[str, Any]:
    """Per-window alphas and the gate table, from the logged rows alone."""

    spec = require_probe_columns(log)
    depth = spec["depth"]
    count = spec["cycles"] if limit is None else min(spec["cycles"], int(limit))

    draft_ids = log["draft_ids"]
    draft_values = log["draft_values"]
    draft_probs = log["draft_probs"]
    draft_valid = log["draft_valid"].astype(bool)
    target_ids = log["target_ids"]
    target_values = log["target_values"]
    target_probs = log["target_probs"]
    target_valid = log["target_valid"].astype(bool)
    accepted = log["accepted"].astype(np.int64)
    greedy = log["greedy"].astype(bool)
    gate_q = log["gate_q"].astype(np.float64)
    probe_valid = log["probe_valid"].astype(bool)
    probe_ids = log["probe_ids"]
    probe_values = log["probe_values"]
    probe_probs = log["probe_probs"]
    probe_trimmed = log.get("probe_trimmed")

    # "Scoreable" mirrors the L Sec.D analysis exactly: a window with every
    # draft row and every per-depth target row on the host.  Greedy windows
    # carry no distributions at all and are excluded by the same mask.
    ok = (
        (~greedy[:count])
        & draft_valid[:count, :depth].all(axis=1)
        & target_valid[:count, :depth].all(axis=1)
    )
    all_accepted = accepted[:count] == depth

    alpha = np.full((count, depth + 1), np.nan, dtype=np.float64)
    for index in range(count):
        if not ok[index]:
            continue
        for level in range(depth):
            p_ids, p_probs = prepared_row(
                target_ids[index, level],
                target_values[index, level],
                target_probs[index, level],
            )
            q_ids, q_probs = prepared_row(
                draft_ids[index, level],
                draft_values[index, level],
                draft_probs[index, level],
            )
            alpha[index, level] = overlap(p_ids, p_probs, q_ids, q_probs)

    probe_scored = np.zeros(count, dtype=bool)
    for index in np.flatnonzero(probe_valid[:count]):
        if not target_valid[index, depth]:
            # The bonus target row was never materialised on the host (the
            # lazy per-row path), so there is nothing to score q_4 against.
            continue
        p_ids, p_probs = prepared_row(
            target_ids[index, depth],
            target_values[index, depth],
            target_probs[index, depth],
        )
        q_ids, q_probs = prepared_row(
            probe_ids[index], probe_values[index], probe_probs[index]
        )
        alpha[index, depth] = overlap(p_ids, p_probs, q_ids, q_probs)
        probe_scored[index] = True

    base_tokens = float(np.mean(accepted[:count][ok] + 1)) if ok.any() else float("nan")
    gates: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = ok & (gate_q[:count, depth - 1] > float(threshold))
        probed = selected & probe_scored
        gates.append(
            {
                "threshold": float(threshold),
                "windows": int(np.count_nonzero(selected)),
                "p_gate": (
                    float(np.count_nonzero(selected) / np.count_nonzero(ok))
                    if ok.any()
                    else float("nan")
                ),
                "p_all_accepted": (
                    float(np.mean(all_accepted[:count][selected]))
                    if selected.any()
                    else float("nan")
                ),
                "alpha": [
                    float(np.nanmean(alpha[selected, level]))
                    if selected.any()
                    else float("nan")
                    for level in range(depth)
                ],
                "probed": int(np.count_nonzero(probed)),
                "alpha4": (
                    float(np.mean(alpha[probed, depth])) if probed.any() else float("nan")
                ),
                "alpha4_se": (
                    float(
                        np.std(alpha[probed, depth]) / np.sqrt(np.count_nonzero(probed))
                    )
                    if np.count_nonzero(probed) > 1
                    else float("nan")
                ),
            }
        )

    return {
        "layout": spec["layout"],
        "depth": depth,
        "cycles": count,
        "cycles_scoreable": int(np.count_nonzero(ok)),
        "greedy_cycles": int(np.count_nonzero(greedy[:count])),
        "all_accepted_cycles": int(np.count_nonzero(all_accepted & ok)),
        "probe_cycles": int(np.count_nonzero(probe_valid[:count])),
        "probe_cycles_scored": int(np.count_nonzero(probe_scored)),
        "probe_cycles_trimmed": (
            0 if probe_trimmed is None else int(np.sum(probe_trimmed[:count]))
        ),
        "base_tokens_per_window": base_tokens,
        "alpha_ungated": [
            float(np.nanmean(alpha[ok, level])) if ok.any() else float("nan")
            for level in range(depth)
        ],
        "alpha4_ungated": (
            float(np.mean(alpha[probe_scored, depth]))
            if probe_scored.any()
            else float("nan")
        ),
        "alpha4_ungated_se": (
            float(
                np.std(alpha[probe_scored, depth])
                / np.sqrt(np.count_nonzero(probe_scored))
            )
            if np.count_nonzero(probe_scored) > 1
            else float("nan")
        ),
        "gates": gates,
    }


def project(
    result: dict[str, Any],
    *,
    ms_per_window: float,
    draft_step_ms: float,
    row_ms: float,
) -> list[dict[str, Any]]:
    """Tokens/window and tok/s per gate under one row cost."""

    base = result["base_tokens_per_window"]
    baseline = 1000.0 * base / ms_per_window
    rows = []
    for gate in result["gates"]:
        delta_tokens = gate["p_gate"] * gate["p_all_accepted"] * gate["alpha4"]
        delta_cost = gate["p_gate"] * (draft_step_ms + row_ms)
        tok_s = 1000.0 * (base + delta_tokens) / (ms_per_window + delta_cost)
        rows.append(
            {
                **gate,
                "delta_tokens": float(delta_tokens),
                "delta_cost_ms": float(delta_cost),
                "tok_s": float(tok_s),
                "delta_pct": float(100.0 * (tok_s / baseline - 1.0)),
            }
        )
    return rows


def report(
    result: dict[str, Any],
    *,
    ms_per_window: float,
    draft_step_ms: float,
    row_costs: Sequence[float],
    go_gate: float,
    go_alpha: float,
) -> str:
    depth = result["depth"]
    base = result["base_tokens_per_window"]
    lines = [
        f"layout {result['layout']}  depth {depth}  cycles {result['cycles']}",
        f"scoreable {result['cycles_scoreable']}  greedy {result['greedy_cycles']}  "
        f"all-accepted {result['all_accepted_cycles']}",
        f"depth-4 probe: {result['probe_cycles']} windows recorded, "
        f"{result['probe_cycles_scored']} scored against a bonus target row"
        + (
            f", {result['probe_cycles_trimmed']} shaped wider than K20 and were trimmed"
            if result["probe_cycles_trimmed"]
            else ""
        ),
        "",
        f"base {base:.4f} tok/window at {ms_per_window:.2f} ms "
        f"-> {1000.0 * base / ms_per_window:.2f} tok/s",
        "alpha (sum min(p, q)) ungated: "
        + " ".join(f"a{level + 1}={value:.3f}" for level, value in
                   enumerate(result["alpha_ungated"]))
        + f"  a{depth + 1}={result['alpha4_ungated']:.3f}"
        + (
            f" +-{result['alpha4_ungated_se']:.3f}"
            if np.isfinite(result["alpha4_ungated_se"])
            else ""
        ),
    ]

    for row_ms in row_costs:
        lines.append("")
        lines.append(
            f"-- gate on q(x_{depth}); extra step {draft_step_ms:.2f} ms + "
            f"M={depth + 2} row {row_ms:.2f} ms, charged on gated windows only --"
        )
        header = (
            f"{'gate':>10} {'P(G)':>6} {'n':>5} {'P(all|G)':>9} "
            + " ".join(f"{'a' + str(level + 1) + '|G':>7}" for level in range(depth))
            + f" {'probed':>7} {'a' + str(depth + 1) + '|G':>9} {'dT/win':>8} "
            f"{'dcost':>6} {'tok/s':>7} {'delta':>8}"
        )
        lines.append(header)
        for row in project(
            result,
            ms_per_window=ms_per_window,
            draft_step_ms=draft_step_ms,
            row_ms=row_ms,
        ):
            lines.append(
                f"{'>' + format(row['threshold'], '.2f'):>10} "
                f"{row['p_gate']:6.3f} {row['windows']:5d} "
                f"{row['p_all_accepted']:9.3f} "
                + " ".join(f"{value:7.3f}" for value in row["alpha"])
                + f" {row['probed']:7d} {row['alpha4']:9.3f} "
                f"{row['delta_tokens']:+8.4f} {row['delta_cost_ms']:6.2f} "
                f"{row['tok_s']:7.2f} {row['delta_pct']:+7.2f}%"
            )

    verdict = next(
        (gate for gate in result["gates"] if abs(gate["threshold"] - go_gate) < 1e-9),
        None,
    )
    lines.append("")
    if verdict is None or not np.isfinite(verdict["alpha4"]):
        lines.append(
            f"NO-GO (undetermined): no scored probe window at the q(x_{depth}) > "
            f"{go_gate} gate. The measurement did not happen; do not read the "
            "table above as evidence either way."
        )
    elif verdict["alpha4"] >= go_alpha:
        lines.append(
            f"GO: alpha_{depth + 1} | q(x_{depth}) > {go_gate} = "
            f"{verdict['alpha4']:.3f} +-{verdict['alpha4_se']:.3f} on "
            f"{verdict['probed']} probed windows, at or above the {go_alpha} bar "
            "L Sec.D set. The M=5 lane is worth building."
        )
    else:
        lines.append(
            f"NO-GO: alpha_{depth + 1} | q(x_{depth}) > {go_gate} = "
            f"{verdict['alpha4']:.3f} +-{verdict['alpha4_se']:.3f} on "
            f"{verdict['probed']} probed windows, below the {go_alpha} bar "
            "L Sec.D set. Gated depth 4 does not pay; do not build the M=5 lane."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="path written by MTPLX_FABLE_K20_LOG")
    parser.add_argument(
        "--ms-per-window",
        type=float,
        default=MS_PER_WINDOW,
        help="verify-window wall time; L Sec.3's retained frame is 38.7 "
        "(read it off an UNARMED arm -- a probed run is a data run)",
    )
    parser.add_argument(
        "--draft-step-ms",
        type=float,
        default=DRAFT_STEP_MS,
        help="cost of the extra depth-4 draft step (L Sec.D: 1.2)",
    )
    parser.add_argument(
        "--row-ms",
        type=float,
        action="append",
        default=[],
        help="marginal verify-row cost; repeatable. Default reports both "
        "1.8 (H Sec.2.4, eager-adjacent) and 1.4 (K's compiled fit).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[],
        help="gate on q(x_D); repeatable. Default 0.6 0.7 0.8 0.9.",
    )
    parser.add_argument("--go-gate", type=float, default=GO_GATE)
    parser.add_argument("--go-alpha", type=float, default=GO_ALPHA)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    args = parser.parse_args(argv)

    thresholds = tuple(sorted(set(args.threshold))) if args.threshold else THRESHOLDS
    if args.go_gate not in thresholds:
        thresholds = tuple(sorted({*thresholds, float(args.go_gate)}))
    row_costs = tuple(args.row_ms) if args.row_ms else ROW_MS

    log = load_log(args.npz)
    try:
        result = score(log, thresholds=thresholds, limit=args.limit)
    except SystemExit as unscoreable:
        # A log with no probe columns is a real, common outcome (the probe was
        # never armed, or armed but no window accepted all three drafts). Say
        # so once, in the same shape as every other failure here, instead of
        # letting the raise print twice.
        print(f"FAIL: {unscoreable}", file=sys.stderr)
        return 1
    print(
        report(
            result,
            ms_per_window=args.ms_per_window,
            draft_step_ms=args.draft_step_ms,
            row_costs=row_costs,
            go_gate=args.go_gate,
            go_alpha=args.go_alpha,
        )
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if not result["probe_cycles_scored"]:
        print(
            "\nFAIL: no probed window could be scored. Either the probe never "
            "fired (no all-accept window, or MTPLX_FABLE_DEPTH4_PROBE was not "
            "set) or every probed window took the lazy target path and has no "
            "bonus row on the host.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
