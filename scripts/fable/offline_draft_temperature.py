#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# NO GPU.  Pure NumPy.  This script never imports mlx and never touches the
# GPU exclusive lock -- it replays logged K20 rows on the CPU.
# ---------------------------------------------------------------------------
"""Sweep the draft proposal's temperature (and top-p / top-k) offline against
the exact acceptance objective, on the K20 rows captured by
``MTPLX_FABLE_K20_LOG``.

This is H §Option D.  It is the cheapest of the four options and carries the
least risk, for one reason that is worth restating precisely:

    The accept law ``min(1, p/q)`` with residual ``(p - q)+`` is exact for
    **any** proposal ``q``, provided the verifier tests against the same ``q``
    the drafter sampled from.  So the draft distribution is a free
    hyperparameter: reshaping it cannot change the output distribution, only
    the acceptance rate.  Exact by construction -- there is no exactness
    experiment to run, only an efficiency one.

The objective is exact too.  With ``x ~ q``,

    E_x[ min(1, p(x)/q(x)) ] = sum_y min(p(y), q(y)) = beta

so ``beta`` -- the total-variation overlap of the prepared rows -- *is* the
per-depth acceptance probability, computed in closed form with no sampling.
Nothing in this script draws a random number.

Re-temperaturing, and the one thing the log cannot give
-------------------------------------------------------
A tempered softmax is a power of the original: ``exp(v/T) = exp(v)^(1/T)``, so
``q_T(y) ∝ q_1(y)^(1/T)`` over the **whole** vocabulary, and the top-20 set is
invariant (the map is monotone).  The log stores the top-20 raw logits and the
full-vocabulary probabilities, so everything about the retained support is
exact.  What is *not* in the log is the shape of the truncated tail, and the
tail contributes to the tempered normaliser.  Two tail models bracket it:

``--tail drop`` (optimistic)
    Renormalise over the top-20 only, i.e. assume the tail contributes nothing.
    Since ``sum_i x_i^a <= (sum_i x_i)^a`` for ``a = 1/T >= 1``, dropping the
    tail *under*-states the normaliser, so this over-states the retained
    probabilities.
``--tail lump`` (pessimistic, default)
    Treat the whole tail as ONE atom of mass ``1 - sum(probs)`` and temper it
    with everything else.  By the same inequality this *over*-states the
    tail's tempered mass.

The truth lies between, and with top-k 20 / top-p 0.95 the tail is a few
percent, so the two arms land close together.  ``lump`` is the identity at
``T = 1`` -- it reproduces the logged row exactly -- while ``drop`` renormalises
the head even at ``T = 1``, so its whole sweep is measured against its own
``T = 1`` row.  Run both and read the interval, not either endpoint.

What this measures, and the assumptions it rests on
----------------------------------------------------
Per logged cycle and per depth ``d`` the script recomputes the prepared draft
row ``q_d`` under the candidate shaping, keeps the target row ``p_d`` exactly
as logged, and evaluates ``beta_d = sum_y min(p_d(y), q_d(y))``.  Then, since
the three accept coins are independent given the rows,

    E[l]  =  E_rows[ b1 + b1*b2 + b1*b2*b3 ]        (per-cycle products)
    T_m4  =  E[l] + 1                                (every window emits l + 1)

**Assumption, stated plainly: the rows are held fixed.**  Changing the
proposal changes which token the drafter picks, which changes the MTP chain's
next hidden state, which changes every subsequent row -- including the target
rows, because the M4 verify window is conditioned on the drafted tokens.  The
logged rows are the ones the *shipped* proposal produced.  So this sweep
answers "how much overlap would the target have had with a re-shaped proposal
on the realised row distribution", not "what would a re-tempered run score".
It is a first-order estimate.  It is also the same first-order estimate that
H §5.4 makes with a surrogate, only with the real rows instead of a fit -- and
the direction it recommends (sharpen ``q``; the draft is flatter than the
target) is a robust conclusion even if the magnitude is not.

Two secondary numbers are printed as a cross-check on that assumption:

* ``E[l] (product of means)`` -- the same chain with the per-depth means
  multiplied instead of the per-cycle products.  A large gap between the two
  means the per-cycle ``beta`` values are strongly correlated across depths;
* ``P(alpha = 0)`` per depth, i.e. the fraction of drafted tokens the target
  gives zero mass.  H §1.2 reads its rise with depth as the MTP chain drifting
  off-distribution; a temperature that lowers it is doing real work, one that
  only raises ``beta`` at fixed ``P(alpha = 0)`` is trimming the tail.

Usage::

    python scripts/fable/offline_draft_temperature.py rows.npz
    python scripts/fable/offline_draft_temperature.py rows.npz \
        --temperatures 0.6 0.7 0.8 0.9 1.0 1.1 1.2 --ms-per-window 37.47
    python scripts/fable/offline_draft_temperature.py rows.npz --top-p 0.9 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - import shim, exercised both ways in practice
    from scripts.fable.offline_block_verification import (
        DEPTH,
        TOP_P,
        ZERO,
        load_log,
        lookup,
        lookup_many,
        prepare_batched_row,
        prepare_row,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from offline_block_verification import (  # type: ignore[no-redef]
        DEPTH,
        TOP_P,
        ZERO,
        load_log,
        lookup,
        lookup_many,
        prepare_batched_row,
        prepare_row,
    )

DEFAULT_TEMPERATURES = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


def temper_row(
    probs: np.ndarray,
    *,
    temperature: float,
    tail: str = "lump",
) -> np.ndarray:
    """Re-temperature one logged top-20 row of full-vocabulary probabilities.

    ``q_T(y) ∝ q_1(y)**(1/T)``.  Returns probabilities on the same 20 ids,
    still normalised against the *whole* vocabulary under the chosen tail
    model, because the kernel's top-p cut (``cumulative_before < top_p``,
    ``pr391_softfloat64_verifier_decision.py:60-66``) reads pre-truncation
    mass and would otherwise see the wrong scale.
    """

    values = np.asarray(probs, dtype=np.float64)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    power = 1.0 / float(temperature)
    tempered = np.where(values > ZERO, np.power(values, power), ZERO)
    total = np.sum(tempered, dtype=np.float64)
    if tail == "lump":
        head = float(np.sum(values, dtype=np.float64))
        remainder = max(0.0, 1.0 - head)
        total = total + np.power(np.float64(remainder), power)
    elif tail != "drop":
        raise ValueError("tail must be 'lump' or 'drop'")
    if not np.isfinite(total) or total <= ZERO:
        raise ValueError("tempered row lost all mass")
    return tempered / total


def restrict_top_k(
    ids: np.ndarray, values: np.ndarray, probs: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the ``top_k`` highest-scoring entries, kernel tie-break.

    Ranking is ``lexsort((ids.astype(uint64), -values.astype(float64)))`` --
    score descending, id ascending on a tie -- the same order
    ``_prepare_batched_candidate_row`` uses (kernel line 60).  Temperature does
    not reorder a row, so this can be applied to the logged logits directly.
    """

    if top_k >= ids.size:
        return ids, values, probs
    rank = np.lexsort((ids.astype(np.uint64), -values.astype(np.float64)))[:top_k]
    return ids[rank], values[rank], probs[rank]


def overlap(
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
) -> float:
    """``beta = sum_y min(p(y), q(y))`` over the union of both supports.

    This is the exact expected acceptance at that depth: for ``x ~ q``,
    ``E[min(1, p(x)/q(x))] = sum_y min(p(y), q(y))`` (H §1.2).
    """

    union = np.union1d(
        np.asarray(target_ids, dtype=np.uint32), np.asarray(draft_ids, dtype=np.uint32)
    )
    return float(
        np.sum(
            np.minimum(
                lookup_many(target_ids, target_probs, union),
                lookup_many(draft_ids, draft_probs, union),
            ),
            dtype=np.float64,
        )
    )


def score_arm(
    log: dict[str, np.ndarray],
    *,
    temperature: float,
    tail: str,
    top_p: float,
    top_k: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Per-depth beta and the implied chain, for one proposal shaping."""

    cycles = int(log["draft_tokens"].shape[0])
    if limit is not None:
        cycles = min(cycles, int(limit))
    betas = np.zeros((cycles, DEPTH), dtype=np.float64)
    zero_alpha = np.zeros((cycles, DEPTH), dtype=bool)

    for index in range(cycles):
        for depth in range(DEPTH):
            target_ids, target_probs = prepare_batched_row(
                log["target_ids"][index, depth],
                log["target_values"][index, depth],
                log["target_probs"][index, depth],
            )
            ids, values, probs = restrict_top_k(
                np.asarray(log["draft_ids"][index, depth], dtype=np.uint32),
                np.asarray(log["draft_values"][index, depth], dtype=np.float32),
                np.asarray(log["draft_probs"][index, depth], dtype=np.float32),
                top_k,
            )
            tempered = temper_row(probs, temperature=temperature, tail=tail)
            draft_ids, draft_probs = prepare_row(
                ids, values, tempered.astype(np.float32), top_p=top_p
            )
            betas[index, depth] = overlap(
                target_ids, target_probs, draft_ids, draft_probs
            )
            # P(alpha = 0) is a property of the drafted token, which a
            # re-shaped proposal would not have drawn.  Reported for the
            # logged token only, i.e. it is a T = 1 diagnostic.
            token = int(log["draft_tokens"][index, depth])
            zero_alpha[index, depth] = (
                lookup(target_ids, target_probs, token) <= ZERO
            )

    chain = betas[:, 0] + betas[:, 0] * betas[:, 1] + (
        betas[:, 0] * betas[:, 1] * betas[:, 2]
    )
    means = betas.mean(axis=0)
    product_of_means = means[0] + means[0] * means[1] + means[0] * means[1] * means[2]
    expected_length = float(chain.mean())
    return {
        "temperature": float(temperature),
        "tail": tail,
        "top_p": float(top_p),
        "top_k": int(top_k),
        "cycles": cycles,
        "beta": [float(value) for value in means],
        "beta_sem": [
            float(betas[:, depth].std(ddof=1) / np.sqrt(cycles)) if cycles > 1 else float("nan")
            for depth in range(DEPTH)
        ],
        "zero_alpha": [float(zero_alpha[:, depth].mean()) for depth in range(DEPTH)],
        "expected_length": expected_length,
        "expected_length_product_of_means": float(product_of_means),
        "tokens_per_window": expected_length + 1.0,
    }


def sweep(
    log: dict[str, np.ndarray],
    *,
    temperatures: Sequence[float],
    tail: str,
    top_p: float,
    top_k: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return [
        score_arm(
            log,
            temperature=temperature,
            tail=tail,
            top_p=top_p,
            top_k=top_k,
            limit=limit,
        )
        for temperature in temperatures
    ]


def report(rows: Sequence[dict[str, Any]], *, ms_per_window: float | None) -> str:
    if not rows:
        return "no arms scored"
    lines: list[str] = []
    first = rows[0]
    lines.append(
        f"cycles {first['cycles']}   tail model {first['tail']}   "
        f"draft top-p {first['top_p']}   draft top-k {first['top_k']}"
    )
    lines.append("")
    header = (
        f"{'T':>6}{'beta1':>9}{'beta2':>9}{'beta3':>9}"
        f"{'E[l]':>10}{'tok/win':>10}{'vs T=1':>9}"
    )
    if ms_per_window:
        header += f"{'tok/s':>10}"
    lines.append(header)
    base = next(
        (row for row in rows if abs(row["temperature"] - 1.0) < 1e-12), rows[0]
    )
    for row in rows:
        delta = row["tokens_per_window"] / base["tokens_per_window"] - 1.0
        line = (
            f"{row['temperature']:>6.2f}"
            f"{row['beta'][0]:>9.4f}{row['beta'][1]:>9.4f}{row['beta'][2]:>9.4f}"
            f"{row['expected_length']:>10.4f}{row['tokens_per_window']:>10.4f}"
            f"{delta * 100:>8.2f}%"
        )
        if ms_per_window:
            line += f"{row['tokens_per_window'] / (ms_per_window / 1000.0):>10.2f}"
        lines.append(line)
    lines.append("")
    lines.append(
        f"beta sem (T=1)             "
        + "  ".join(f"{value:.4f}" for value in base["beta_sem"])
    )
    lines.append(
        f"P(alpha = 0) at T=1        "
        + "  ".join(f"{value:.4f}" for value in base["zero_alpha"])
        + "   (drafted-token diagnostic; not a function of T)"
    )
    lines.append(
        f"E[l] product-of-means T=1  {base['expected_length_product_of_means']:.4f}"
        f"  vs per-cycle {base['expected_length']:.4f}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="path written by MTPLX_FABLE_K20_LOG")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPERATURES),
        help="draft temperatures to sweep (the target is never re-shaped)",
    )
    parser.add_argument(
        "--tail",
        choices=("lump", "drop"),
        default="lump",
        help="model for the truncated tail's tempered mass; run both to bracket",
    )
    parser.add_argument(
        "--top-p", type=float, default=TOP_P, help="draft top-p (shipped: 0.95)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="draft top-k (shipped: 20, the log's width)"
    )
    parser.add_argument("--ms-per-window", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    log = load_log(args.npz)
    if "draft_values" not in log:
        print(
            "FAIL: this log has no draft logits, so it cannot be "
            "re-temperatured.",
            file=sys.stderr,
        )
        return 1
    rows = sweep(
        log,
        temperatures=args.temperatures,
        tail=args.tail,
        top_p=args.top_p,
        top_k=args.top_k,
        limit=args.limit,
    )
    print(report(rows, ms_per_window=args.ms_per_window))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
