#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# NO GPU.  Pure NumPy.  This script never imports mlx and never touches the
# GPU exclusive lock -- it replays logged K20 rows on the CPU.
# ---------------------------------------------------------------------------
"""Score the block-verification law (H §Option B) against the shipped law,
offline, on the K20 rows captured by ``MTPLX_FABLE_K20_LOG``.

Why offline, and which number to read
-------------------------------------
``H-tokens-per-window-design.md`` §1.4: ``Var(l) ~ 1.50`` at ``n ~ 385`` gives
``SE(E[l]) = 0.062``, i.e. **+-4.2% (1 sigma)**.  Both candidate laws move
``E[l]`` by a few percent, so a live A/B cannot resolve them.  A replay on
logged rows fixes the rows, which removes most of that; two estimators are
reported, and they are not interchangeable:

**``E[tok/win]`` -- the one to quote.**  ``1 + sum_d w_d``, where ``w_d`` is the
probability the window accepts through depth ``d`` given the rows and the
drafted tokens.  No uniform is consulted, so the accept-coin variance is
integrated out analytically and the only noise left is the drafted-row
sampling -- which is *identical* for both arms.  The reported ``delta`` is the
paired per-cycle difference with its own standard error, and that interval is
several times tighter than either arm's own.

**``replay`` -- the exactness proof.**  The full emission simulation, driven by
the four **logged PCG64 uniforms**, so it reproduces the token the device
would have emitted.  This is what compares each law's decision against the
device kernel's own logged output and what verifies the ``c = 1`` identity.
Its arm-to-arm difference is noisy (the two laws' coins land differently on
the same uniform), so read it for correctness, not for magnitude.

What is replayed, and what is not
---------------------------------
Each logged cycle carries the exact seven prepared K20 rows and the four
decision uniforms of one verify window.  This script re-decides **that window**
under each law.  It does **not** re-run the model: under the block law a
window can accept a different number of tokens, which in production would
change every subsequent window's rows.  So the number reported here is a
**per-window counterfactual on the realised row distribution** -- the standard
and correct estimator for ``E[l]``, and the same framing H §3.4 uses for its
surrogate.  It is not a full-trajectory simulation and must not be sold as one.

The two laws
------------
**Current** -- ``mtplx/kernels/pr391_softfloat64_verifier_decision.py:220-345``,
mirrored here in float64::

    for d in 1..3:
        rho_d = p_d(x_d) / q_d(x_d)
        alpha_d = min(1, rho_d)                       # kernel line 280-287
        if u_d <= alpha_d: accept                     # kernel line 289, `<=`
        else: emit sample(normalise((p_d - q_d)+))    # kernel lines 296-306
    emit bonus ~ p_4                                  # kernel lines 335-342

**Block verification** -- H §3.2.  One clip moves: the running product is
clipped at 1 instead of each factor being clipped and then multiplied, and
depth ``d``'s accept coin is allowed to look at the depth ``d+1`` rows::

    c_0 = 1 ; w_0 = 1
    for d in 1..3:
        rho_d = p_d(x_d) / q_d(x_d)
        A_d   = min(1, c_{d-1} * rho_d)               # reach BUDGET
        if d < 3:
            base(y) = min(1, A_d * rho_{d+1}(y))      # y over the d+1 draft support
            lam_d   = solve_lambda:  sum_y q_{d+1}(y) * min(cap, base(y) + lam) = A_d
            w_d     = min(cap, base(x_{d+1}) + lam_d) # realised reach probability
        else:
            w_d     = A_d
        a_d = w_d / w_{d-1}                           # CONDITIONAL accept coin
        if u_d <= a_d: c_d = A_d ; continue
        emit sample(normalise((c_{d-1} * p_d - q_d)+))    # SCALED residual
        stop
    emit bonus ~ p_4

``lam_d`` is the water-filling level that keeps ``E_{x_{d+1}~q_{d+1}}[w_d]``
exactly at the budget ``A_d`` -- which is what preserves exactness: the
position-``d`` accept probability, averaged over the *next* drafted token,
still equals ``min(1, rho_d)``, so ``q_d(y) * P(accept | x_d = y) <= p_d(y)``
holds pointwise and the residual stays non-negative.  Within that budget the
mass is shifted toward the realisations whose next drafted token the target
likes, which is the entire gain.

**One correction to H's pseudo-code, made explicit.**  H writes the water-fill
cap as a literal ``min(1, ...)``.  With that cap ``w_d`` can exceed ``w_{d-1}``
(a good ``x_{d+1}`` can ask to reach depth ``d+1`` more often than depth ``d``
was itself reached), and then ``a_d = w_d / w_{d-1} > 1`` is not a probability.
Two caps are implemented:

``--cap reach`` (default)
    Cap the water-fill at ``w_{d-1}``.  ``w_d <= w_{d-1}`` by construction,
    ``E[w_d] = A_d`` still holds exactly (feasible because
    ``w_{d-1} >= A_d``), and the coin is always a probability.  This is the
    smallest change that makes H's law well defined and budget-exact.
``--cap one``
    H's literal cap, with ``a_d`` clipped at 1.  The clip wastes budget, so
    this arm is a lower bound; the report counts how often the clip binds.

Both caps coincide whenever ``w_{d-1} = 1``, so both reproduce the current law
exactly on the ``c = 1`` windows (H §3.2: "when c = 1 the law is bit-identical
to today's"), which the report verifies rather than assumes -- and fails on if
it ever stops holding.  Holding it needs one piece of float64 care: when the
budget reaches the cap the water-fill has exactly one solution (everything
saturates), and computing it as ``lam = cap - min(base)`` and adding it back
returns ``0.9999999999999999`` often enough to break the identity on a couple
of windows per four hundred.  :func:`_block_realised_reach` short-circuits that
case to ``cap`` instead.

Arithmetic and tie ownership (mirrored, with citations)
-------------------------------------------------------
All of it is float64, from ``pr391_softfloat64_verifier_decision.py``:

* row preparation ``_prepare_batched_candidate_row`` (lines 33-77):
  rank by ``lexsort((ids.astype(uint64), -values.astype(float64)))`` -- score
  descending, **id ascending on a score tie**; keep ``probs > 0``; top-p by
  ``cumulative_before < top_p`` (**strict**, and computed on the raw
  full-vocabulary probabilities, i.e. the mass *before* the entry); normalise
  once by ``first_total``; return **sorted by token id ascending**;
* ``_prepare_candidate_row`` (lines 88-105) adds the second normalisation
  ``_renormalize_sparse_probabilities`` (lines 80-86).  **Draft rows get the
  double normalisation, target rows only the single one** (lines 246-265);
* ``alpha`` uses the *single*-normalised target row (line 281) while the
  residual uses the *double*-normalised one (line 297).  Both are mirrored;
* accept on ``u <= alpha`` -- line 289, ``<=``, so a draw exactly equal to the
  accept probability **accepts**;
* ``_sample_prepared`` (lines 138-149): ``cdf = cumsum(p); cdf /= sum(p);``
  ``index = min(searchsorted(cdf, u, side="right"), n - 1)`` -- ``side="right"``
  means a draw exactly equal to a CDF breakpoint lands in the **next** bucket,
  and the final clip owns the ``u`` above the last breakpoint;
* ``_prepare_residual`` (lines 165-197): ``union1d`` of both supports,
  ``max(p - q, 0)``, drop non-positive, normalise twice; when nothing survives,
  fall back to the double-normalised target row (lines 303-306);
* a stop token accepted at depth ``d`` ends the window with
  ``draws_used = d + 1`` and no selected token (lines 290-303);
* ``q <= 0`` for a drafted token yields ``alpha = 1 if p > 0 else 0``
  (lines 282-284) -- unreachable in practice, mirrored anyway.

Self-check
----------
Before scoring anything, the current-law replay is compared against the
decision the **device kernel actually returned** for every logged cycle
(``accepted`` / ``first_reject`` / ``selected_token`` / ``selected_kind`` /
``draws_used``).  A single mismatch is a hard failure: it means this file no
longer mirrors the kernel, and every number below it would be worthless.
``--allow-replay-mismatch`` downgrades it to a warning for debugging.

Usage::

    python scripts/fable/offline_block_verification.py rows.npz
    python scripts/fable/offline_block_verification.py rows.npz --ms-per-window 37.47
    python scripts/fable/offline_block_verification.py rows.npz --cap one --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable, Sequence

import numpy as np

DEPTH = 3
TARGET_ROWS = DEPTH + 1
TOP_P = 0.95

SELECTED_NONE = 0
SELECTED_CORRECTION = 1
SELECTED_BONUS = 2

ZERO = np.float64(0.0)
ONE = np.float64(1.0)


# ---------------------------------------------------------------------------
# Row preparation -- exact mirror of the kernel's NumPy reference.
# ---------------------------------------------------------------------------


def prepare_batched_row(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float = TOP_P,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``_prepare_batched_candidate_row`` (kernel lines 33-77)."""

    ids = np.asarray(ids, dtype=np.uint32)
    values = np.asarray(values, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    probabilities = probs.astype(np.float64)
    rank = np.lexsort((ids.astype(np.uint64), -values.astype(np.float64)))
    ranked_ids = ids[rank]
    ranked_probs = probabilities[rank]
    cumulative_before = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(ranked_probs[:-1], dtype=np.float64))
    )
    retained = ranked_probs > ZERO
    bounded = np.float64(top_p)
    if bounded < ONE:
        retained &= cumulative_before < bounded
    retained_ids = ranked_ids[retained]
    retained_probs = ranked_probs[retained]
    first_total = np.sum(retained_probs, dtype=np.float64)
    if not np.isfinite(first_total) or first_total <= ZERO:
        raise ValueError("candidate row must retain positive finite mass")
    token_order = np.argsort(retained_ids)
    return (
        retained_ids[token_order].astype(np.uint32, copy=False),
        retained_probs[token_order] / first_total,
    )


def renormalize_sparse(probabilities: np.ndarray) -> np.ndarray:
    """Mirror ``_renormalize_sparse_probabilities`` (kernel lines 80-86)."""

    sanitized = np.where(
        np.isfinite(probabilities) & (probabilities > ZERO), probabilities, ZERO
    )
    return sanitized / np.sum(sanitized, dtype=np.float64)


def prepare_row(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float = TOP_P,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``_prepare_candidate_row`` (kernel lines 88-105)."""

    prepared_ids, once = prepare_batched_row(ids, values, probs, top_p=top_p)
    return prepared_ids, renormalize_sparse(once)


def lookup(token_ids: np.ndarray, probabilities: np.ndarray, token: int) -> np.float64:
    """Mirror ``_lookup_prepared`` (kernel lines 152-162)."""

    hits = np.nonzero(token_ids == np.uint32(token))[0]
    return ZERO if hits.size == 0 else np.float64(probabilities[int(hits[0])])


def lookup_many(
    token_ids: np.ndarray, probabilities: np.ndarray, wanted: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`lookup` for a prepared (id-ascending, unique) row.

    Prepared rows come out of :func:`prepare_batched_row` sorted by token id
    with no duplicates (kernel lines 73-77), so a binary search is exact.
    """

    wanted = np.asarray(wanted, dtype=np.uint32)
    out = np.zeros(wanted.size, dtype=np.float64)
    if token_ids.size == 0 or wanted.size == 0:
        return out
    position = np.searchsorted(token_ids, wanted)
    clipped = np.minimum(position, token_ids.size - 1)
    hit = token_ids[clipped] == wanted
    out[hit] = np.asarray(probabilities, dtype=np.float64)[clipped[hit]]
    return out


def sample_prepared(
    token_ids: np.ndarray, probabilities: np.ndarray, uniform: float
) -> int:
    """Mirror ``_sample_prepared`` (kernel lines 138-149)."""

    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf /= np.sum(probabilities, dtype=np.float64)
    index = min(
        int(np.searchsorted(cdf, np.float64(uniform), side="right")),
        int(token_ids.size) - 1,
    )
    return int(token_ids[index])


def prepare_residual(
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
    *,
    scale: np.float64 = ONE,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mirror ``_prepare_residual`` (kernel lines 165-197).

    ``scale`` is the block law's reach credit ``c_{d-1}`` multiplying the target
    term (H §3.2, "one scalar multiply on the target term").  ``scale = 1``
    reproduces the kernel byte-for-byte.
    """

    union_ids = np.union1d(target_ids, draft_ids).astype(np.uint32, copy=False)
    residual = np.array(
        [
            max(
                np.float64(scale) * lookup(target_ids, target_probs, int(token))
                - lookup(draft_ids, draft_probs, int(token)),
                ZERO,
            )
            for token in union_ids
        ],
        dtype=np.float64,
    )
    residual = np.where(np.isfinite(residual) & (residual > ZERO), residual, ZERO)
    keep = residual > ZERO
    first_total = np.sum(residual[keep], dtype=np.float64)
    if not np.isfinite(first_total) or first_total <= ZERO:
        return None
    once = residual[keep] / first_total
    sanitized = np.where(np.isfinite(once) & (once > ZERO), once, ZERO)
    return union_ids[keep], sanitized / np.sum(sanitized, dtype=np.float64)


# ---------------------------------------------------------------------------
# One window's prepared rows.
# ---------------------------------------------------------------------------


class Window:
    """The seven prepared K20 rows plus the draws of one verify cycle."""

    __slots__ = (
        "draft_tokens",
        "draft",
        "target_single",
        "target_double",
        "uniforms",
        "bonus_allowed",
        "stops",
    )

    def __init__(
        self,
        *,
        draft_tokens: Sequence[int],
        draft_rows: Sequence[tuple[np.ndarray, np.ndarray]],
        target_rows: Sequence[tuple[np.ndarray, np.ndarray]],
        uniforms: np.ndarray,
        bonus_allowed: bool,
        stops: frozenset[int],
    ) -> None:
        self.draft_tokens = [int(token) for token in draft_tokens]
        self.draft = list(draft_rows)
        self.target_single = list(target_rows)
        self.target_double = [
            (ids, renormalize_sparse(probs)) for ids, probs in target_rows
        ]
        self.uniforms = np.asarray(uniforms, dtype=np.float64)
        self.bonus_allowed = bool(bonus_allowed)
        self.stops = stops

    def rho(self, depth: int, token: int) -> np.float64:
        """``p_d(token) / q_d(token)`` on the prepared rows, kernel-style.

        Kernel lines 280-287: a drafted token with ``q <= 0`` scores 1 when the
        target keeps it and 0 otherwise; ``rho`` is otherwise the raw ratio and
        is clipped by the *caller*, which is the only difference between the
        two laws at depth 1.
        """

        target_ids, target_probs = self.target_single[depth]
        draft_ids, draft_probs = self.draft[depth]
        p_value = lookup(target_ids, target_probs, token)
        q_value = lookup(draft_ids, draft_probs, token)
        if q_value <= ZERO:
            return ONE if p_value > ZERO else ZERO
        return np.float64(p_value / q_value)


def build_window(log: dict[str, np.ndarray], index: int, stops: frozenset[int]) -> Window:
    return Window(
        draft_tokens=log["draft_tokens"][index],
        draft_rows=[
            prepare_row(
                log["draft_ids"][index, depth],
                log["draft_values"][index, depth],
                log["draft_probs"][index, depth],
            )
            for depth in range(DEPTH)
        ],
        target_rows=[
            prepare_batched_row(
                log["target_ids"][index, row],
                log["target_values"][index, row],
                log["target_probs"][index, row],
            )
            for row in range(TARGET_ROWS)
        ],
        uniforms=log["decision_uniforms"][index],
        bonus_allowed=bool(log["bonus_allowed"][index]),
        stops=stops,
    )


# ---------------------------------------------------------------------------
# Decision outcome.
# ---------------------------------------------------------------------------


class Outcome:
    __slots__ = (
        "accepted",
        "first_reject",
        "selected_token",
        "selected_kind",
        "selected_present",
        "draws_used",
        "accept_probability",
        "ladder_all_one",
        "clipped_depths",
    )

    def __init__(self) -> None:
        self.accepted = 0
        self.first_reject = -1
        self.selected_token = 0
        self.selected_kind = SELECTED_NONE
        self.selected_present = False
        self.draws_used = 0
        self.accept_probability = [0.0] * DEPTH
        self.ladder_all_one = True
        self.clipped_depths = 0

    @property
    def tokens(self) -> int:
        """Tokens this window emits: the accepted prefix plus at most one."""

        return self.accepted + (1 if self.selected_present else 0)

    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.accepted,
            self.first_reject,
            self.selected_token,
            self.selected_kind,
            int(self.selected_present),
            self.draws_used,
        )


def _finish_bonus(window: Window, out: Outcome) -> Outcome:
    """Kernel lines 333-343: full accept, then the optional bonus."""

    out.accepted = DEPTH
    out.draws_used = DEPTH
    if window.bonus_allowed:
        bonus_ids, bonus_probs = window.target_single[DEPTH]
        out.selected_token = sample_prepared(
            bonus_ids, bonus_probs, window.uniforms[DEPTH]
        )
        out.selected_kind = SELECTED_BONUS
        out.selected_present = True
        out.draws_used = DEPTH + 1
    return out


def _emit_correction(
    window: Window, out: Outcome, depth: int, credit: np.float64
) -> Outcome:
    """Kernel lines 292-320, with H's ``c`` scaling on the target term."""

    out.first_reject = depth
    target_ids, target_double = window.target_double[depth]
    draft_ids, draft_probs = window.draft[depth]
    residual = prepare_residual(
        target_ids, target_double, draft_ids, draft_probs, scale=credit
    )
    correction_ids, correction_probs = (
        (target_ids, target_double) if residual is None else residual
    )
    out.selected_token = sample_prepared(
        correction_ids, correction_probs, window.uniforms[depth + 1]
    )
    out.selected_kind = SELECTED_CORRECTION
    out.selected_present = True
    out.draws_used = depth + 2
    return out


def decide_current(window: Window) -> Outcome:
    """The shipped law -- kernel lines 265-345, in float64."""

    out = Outcome()
    for depth in range(DEPTH):
        token = window.draft_tokens[depth]
        alpha = min(ONE, window.rho(depth, token))
        out.accept_probability[depth] = float(alpha)
        if window.uniforms[depth] <= alpha:
            out.accepted = depth + 1
            if token in window.stops:
                out.draws_used = depth + 1
                return out
            continue
        return _emit_correction(window, out, depth, ONE)
    return _finish_bonus(window, out)


def water_fill_lambda(
    q: np.ndarray, base: np.ndarray, cap: np.float64, target: np.float64
) -> np.float64:
    """Smallest ``lam >= 0`` with ``sum q * min(cap, base + lam) == target``.

    ``f(lam)`` is continuous, non-decreasing and piecewise linear with
    breakpoints at ``cap - base``.  ``f(0) <= target`` always holds here
    (``sum_y min(q(y), A * p(y)) <= A``), and ``f(inf) = cap * sum q >= target``
    whenever ``cap >= target``, so a root exists.  The walk is over the
    breakpoints sorted ascending with a stable sort, so the result is a
    deterministic function of the (id-ordered) input rows.
    """

    q = np.asarray(q, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    cap = np.float64(cap)
    target = np.float64(target)
    value = np.sum(q * np.minimum(cap, base), dtype=np.float64)
    if value >= target:
        return ZERO
    threshold = cap - base
    order = np.argsort(threshold, kind="stable")
    threshold = threshold[order]
    weights = q[order]
    active = np.sum(weights[threshold > ZERO], dtype=np.float64)
    level = ZERO
    index = int(np.searchsorted(threshold, ZERO, side="right"))
    size = int(threshold.size)
    while index < size:
        edge = np.float64(threshold[index])
        if active <= ZERO:
            return level
        gain = active * (edge - level)
        if value + gain >= target:
            return level + (target - value) / active
        value += gain
        level = edge
        while index < size and np.float64(threshold[index]) == edge:
            active -= np.float64(weights[index])
            index += 1
    if active > ZERO:
        return level + (target - value) / active
    return level


def alpha_by_depth(window: Window) -> np.ndarray:
    """``alpha_d = min(1, rho_d)`` at **every** depth, past the first rejection.

    The kernel returns as soon as a depth rejects (kernel lines 289-320), so
    the receipts' ``drafts[].accept_probability`` is censored exactly where
    block verification pays -- H §3.2.  The logged rows are not: the drafter
    always runs the full three-deep chain and the M4 window always carries all
    four target rows, so the uncensored ladder is recoverable here.  This is
    H §1.2's table without its truncation.
    """

    return np.array(
        [
            min(ONE, window.rho(depth, window.draft_tokens[depth]))
            for depth in range(DEPTH)
        ],
        dtype=np.float64,
    )


def reach_ladder_current(window: Window) -> np.ndarray:
    """``w_d`` for the shipped law: ``prod_{j<=d} min(1, rho_j)``.

    ``w_d`` is the probability, given the rows and the drafted tokens, that the
    window accepts through depth ``d``.  It consults **no uniform**, so
    ``E[l] = sum_d w_d`` is the same quantity the coin-driven replay estimates
    but with the coin noise integrated out.  Both laws are evaluated on the
    same drafted tokens, so the paired difference is a common-random-numbers
    estimator with a far tighter interval than the replay's.
    """

    ladder = np.zeros(DEPTH, dtype=np.float64)
    credit = ONE
    for depth in range(DEPTH):
        credit = credit * min(ONE, window.rho(depth, window.draft_tokens[depth]))
        ladder[depth] = credit
    return ladder


def reach_ladder_block(window: Window, *, cap_mode: str = "reach") -> np.ndarray:
    """``w_d`` for block verification, on the same drafted tokens."""

    ladder = np.zeros(DEPTH, dtype=np.float64)
    credit = ONE
    reach = ONE
    for depth in range(DEPTH):
        rho = window.rho(depth, window.draft_tokens[depth])
        budget = min(ONE, credit * rho)
        realised = _block_realised_reach(
            window, depth, budget=budget, reach=reach, cap_mode=cap_mode
        )
        realised = min(realised, reach)
        ladder[depth] = realised
        credit = budget
        reach = realised
    return ladder


def _block_realised_reach(
    window: Window,
    depth: int,
    *,
    budget: np.float64,
    reach: np.float64,
    cap_mode: str,
) -> np.float64:
    """``w_d`` at one depth: the water-filled look-ahead, or the raw budget."""

    if depth + 1 >= DEPTH:
        return budget
    cap = ONE if cap_mode == "one" else reach
    if budget >= cap:
        # The row's mass sums to 1, so a budget at or above the cap has exactly
        # one solution: every realisation saturates.  Short-circuiting it is not
        # an optimisation -- solving for `lam = cap - min(base)` and adding it
        # back reintroduces float64 rounding, and at `cap = budget = 1` that
        # rounding is precisely what would break the c = 1 identity with the
        # shipped law (H §3.2).
        return cap
    draft_ids, draft_probs = window.draft[depth + 1]
    target_ids, target_probs = window.target_single[depth + 1]
    next_p = lookup_many(target_ids, target_probs, draft_ids)
    # Prepared rows keep only probs > 0 (kernel line 64), so the mask is
    # belt-and-braces; it also keeps NumPy from warning.
    next_rho = np.divide(
        next_p, draft_probs, out=np.zeros_like(next_p), where=draft_probs > ZERO
    )
    base = np.minimum(ONE, budget * next_rho)
    level = water_fill_lambda(draft_probs, base, cap, budget)
    next_token = window.draft_tokens[depth + 1]
    hits = np.nonzero(draft_ids == np.uint32(next_token))[0]
    return min(cap, base[int(hits[0])] + level) if hits.size else min(cap, level)


def decide_block(window: Window, *, cap_mode: str = "reach") -> Outcome:
    """Block verification -- H §Option B, §3.2.

    ``cap_mode='reach'`` caps the water-fill at ``w_{d-1}`` (budget-exact and
    always a probability); ``cap_mode='one'`` is H's literal ``min(1, .)`` with
    the coin clipped at 1, which wastes budget and is a lower bound.
    """

    if cap_mode not in {"reach", "one"}:
        raise ValueError("cap_mode must be 'reach' or 'one'")
    out = Outcome()
    credit = ONE  # c_{d-1}: the reach budget entering this depth
    reach = ONE  # w_{d-1}: the probability this depth was reached at all
    for depth in range(DEPTH):
        token = window.draft_tokens[depth]
        rho = window.rho(depth, token)
        budget = min(ONE, credit * rho)  # A_d
        # The c = 1 identity (H §3.2).  With credit = reach = 1 the final
        # depth's coin is exactly min(1, rho) and the residual scale is 1, so
        # it always agrees; an earlier depth agrees only when its budget is
        # also 1, because otherwise the water-fill redistributes it.
        if credit != ONE or reach != ONE:
            out.ladder_all_one = False
        elif depth + 1 < DEPTH and budget != ONE:
            out.ladder_all_one = False
        realised = _block_realised_reach(
            window, depth, budget=budget, reach=reach, cap_mode=cap_mode
        )
        # reach == 0 is a measure-zero branch (it needs an earlier coin of 0
        # AND a uniform of exactly 0.0); the conditional is arbitrary there.
        coin = ONE if reach <= ZERO else np.float64(realised / reach)
        if coin > ONE:
            out.clipped_depths += 1
            coin = ONE
        out.accept_probability[depth] = float(coin)
        if window.uniforms[depth] <= coin:
            out.accepted = depth + 1
            if token in window.stops:
                out.draws_used = depth + 1
                return out
            credit = budget
            reach = realised if realised < reach else reach
            continue
        return _emit_correction(window, out, depth, credit)
    return _finish_bonus(window, out)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def load_log(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        log = {key: handle[key] for key in handle.files}
    required = (
        "draft_ids",
        "draft_values",
        "draft_probs",
        "target_ids",
        "target_values",
        "target_probs",
        "draft_tokens",
        "decision_uniforms",
        "accepted",
        "first_reject",
        "selected_token",
        "selected_kind",
        "selected_present",
        "draws_used",
        "bonus_allowed",
    )
    missing = [key for key in required if key not in log]
    if missing:
        raise KeyError(f"K20 log is missing {missing}; was it written by fable_k20_log?")
    return log


def logged_key(log: dict[str, np.ndarray], index: int) -> tuple[int, ...]:
    return (
        int(log["accepted"][index]),
        int(log["first_reject"][index]),
        int(log["selected_token"][index]),
        int(log["selected_kind"][index]),
        int(log["selected_present"][index]),
        int(log["draws_used"][index]),
    )


def score(
    log: dict[str, np.ndarray],
    *,
    cap_mode: str = "reach",
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay both laws over every logged window."""

    stops = frozenset(int(token) for token in log.get("stop_ids", ()))
    cycles = int(log["draft_tokens"].shape[0])
    if limit is not None:
        cycles = min(cycles, int(limit))

    current_tokens: list[int] = []
    block_tokens: list[int] = []
    current_accepted: list[int] = []
    block_accepted: list[int] = []
    current_ladder = np.zeros((cycles, DEPTH), dtype=np.float64)
    block_ladder = np.zeros((cycles, DEPTH), dtype=np.float64)
    alpha = np.zeros((cycles, DEPTH), dtype=np.float64)
    agree = 0
    ladder_one = 0
    ladder_one_agree = 0
    replay_mismatch: list[int] = []
    clipped = 0

    for index in range(cycles):
        window = build_window(log, index, stops)
        current = decide_current(window)
        block = decide_block(window, cap_mode=cap_mode)
        if current.key() != logged_key(log, index):
            replay_mismatch.append(index)
        current_tokens.append(current.tokens)
        block_tokens.append(block.tokens)
        current_accepted.append(current.accepted)
        block_accepted.append(block.accepted)
        current_ladder[index] = reach_ladder_current(window)
        block_ladder[index] = reach_ladder_block(window, cap_mode=cap_mode)
        alpha[index] = alpha_by_depth(window)
        clipped += block.clipped_depths
        same = current.key() == block.key()
        agree += int(same)
        if block.ladder_all_one:
            ladder_one += 1
            ladder_one_agree += int(same)

    current_length = current_ladder.sum(axis=1)
    block_length = block_ladder.sum(axis=1)
    paired = block_length - current_length
    return {
        "cycles": cycles,
        "cap_mode": cap_mode,
        "replay_mismatch_cycles": replay_mismatch,
        "current": _summarise(
            np.asarray(current_tokens, dtype=np.float64),
            current_accepted,
            current_ladder,
        ),
        "block": _summarise(
            np.asarray(block_tokens, dtype=np.float64), block_accepted, block_ladder
        ),
        "paired_delta_tokens_per_window": (
            float(np.mean(paired)) if cycles else float("nan")
        ),
        "paired_delta_sem": (
            float(np.std(paired, ddof=1) / np.sqrt(cycles))
            if cycles > 1
            else float("nan")
        ),
        "alpha_uncensored": {
            "mean": [float(np.mean(alpha[:, d])) for d in range(DEPTH)],
            "p_zero": [float(np.mean(alpha[:, d] <= 0.0)) for d in range(DEPTH)],
            "p_one": [float(np.mean(alpha[:, d] >= 1.0)) for d in range(DEPTH)],
        },
        "agree_fraction": (agree / cycles) if cycles else float("nan"),
        "ladder_all_one_cycles": ladder_one,
        "ladder_all_one_agree_fraction": (
            (ladder_one_agree / ladder_one) if ladder_one else float("nan")
        ),
        "clipped_coin_depths": clipped,
    }


def _summarise(
    tokens: np.ndarray, accepted: Iterable[int], ladder: np.ndarray
) -> dict[str, Any]:
    """Both estimators of tokens/window, plus the per-depth reach ladder.

    ``tokens_per_window`` is the coin-driven replay -- the emission simulation,
    which is what verifies exactness and agreement.  ``tokens_per_window_e``
    integrates the coins out (``1 + sum_d w_d``) and is the number to quote:
    it removes the accept-coin variance entirely, leaving only the drafted-row
    sampling noise, which is *shared* between the two laws.
    """

    accepted_arr = np.asarray(list(accepted), dtype=np.float64)
    size = int(tokens.size)
    mean = float(np.mean(tokens)) if size else float("nan")
    std = float(np.std(tokens, ddof=1)) if size > 1 else float("nan")
    length = ladder.sum(axis=1)
    return {
        "tokens_per_window": mean,
        "tokens_per_window_sem": (std / np.sqrt(size)) if size > 1 else float("nan"),
        "tokens_per_window_e": (float(np.mean(length)) + 1.0) if size else float("nan"),
        "tokens_per_window_e_sem": (
            float(np.std(length, ddof=1) / np.sqrt(size)) if size > 1 else float("nan")
        ),
        "reach_by_depth": [
            float(np.mean(ladder[:, depth])) if size else float("nan")
            for depth in range(DEPTH)
        ],
        "accepted_mean": float(np.mean(accepted_arr)) if size else float("nan"),
        "full_accept_fraction": (
            float(np.mean(accepted_arr == DEPTH)) if size else float("nan")
        ),
    }


def report(result: dict[str, Any], *, ms_per_window: float | None) -> str:
    lines: list[str] = []
    lines.append(f"cycles                     {result['cycles']}")
    lines.append(f"water-fill cap             {result['cap_mode']}")
    mismatches = result["replay_mismatch_cycles"]
    lines.append(
        "current-law replay         "
        + (
            f"EXACT on all {result['cycles']} logged decisions"
            if not mismatches
            else f"MISMATCH on {len(mismatches)} cycles {mismatches[:8]}"
        )
    )
    lines.append("")
    lines.append(
        "E[tokens/window] = 1 + sum_d w_d, with the accept coins integrated "
        "out.\nThis is the number to quote; the coin-driven replay below is "
        "the same\nquantity with the coin noise left in, and is there to prove "
        "exactness."
    )
    lines.append("")
    lines.append(
        f"{'':12}{'E[tok/win]':>12}{'+-sem':>9}"
        f"{'w1':>9}{'w2':>9}{'w3':>9}{'replay':>9}{'+-sem':>8}"
    )
    for name in ("current", "block"):
        row = result[name]
        reach = row["reach_by_depth"]
        lines.append(
            f"{name:12}{row['tokens_per_window_e']:>12.4f}"
            f"{row['tokens_per_window_e_sem']:>9.4f}"
            f"{reach[0]:>9.4f}{reach[1]:>9.4f}{reach[2]:>9.4f}"
            f"{row['tokens_per_window']:>9.4f}{row['tokens_per_window_sem']:>8.4f}"
        )
    delta = result["paired_delta_tokens_per_window"]
    sem = result["paired_delta_sem"]
    base = result["current"]["tokens_per_window_e"]
    percent = (delta / base * 100.0) if base else float("nan")
    lines.append(
        f"{'delta':12}{delta:>12.4f}{sem:>9.4f}"
        f"   paired (common random rows), {percent:+.2f}%"
    )
    lines.append("")
    alpha = result["alpha_uncensored"]
    lines.append(
        "alpha = min(1, rho) at EVERY depth (H §1.2, uncensored -- the "
        "receipts stop\nat the first rejection):"
    )
    lines.append(
        f"{'':12}{'d1':>9}{'d2':>9}{'d3':>9}"
    )
    for label, key in (("E[alpha]", "mean"), ("P(alpha=0)", "p_zero"), ("P(alpha=1)", "p_one")):
        row = alpha[key]
        lines.append(f"{label:12}{row[0]:>9.4f}{row[1]:>9.4f}{row[2]:>9.4f}")
    lines.append("")
    lines.append(
        f"laws agree on              {result['agree_fraction'] * 100:.2f}% of windows"
    )
    lines.append(
        f"  of which c = 1 windows   {result['ladder_all_one_cycles']} "
        f"(agreement {result['ladder_all_one_agree_fraction'] * 100:.2f}%, "
        "must be 100.00%)"
    )
    lines.append(f"coin clipped at depths     {result['clipped_coin_depths']}")
    if ms_per_window is not None and ms_per_window > 0:
        lines.append("")
        lines.append(f"at {ms_per_window:.3f} ms/window:")
        for name in ("current", "block"):
            rate = result[name]["tokens_per_window_e"] / (ms_per_window / 1000.0)
            lines.append(f"  {name:24}{rate:>10.2f} tok/s")
        rate_base = base / (ms_per_window / 1000.0)
        rate_gain = (base + delta) / (ms_per_window / 1000.0)
        lines.append(
            f"  {'delta':24}{rate_gain - rate_base:>10.2f} tok/s "
            f"(+-{sem / (ms_per_window / 1000.0):.2f}, {percent:+.2f}%)"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="path written by MTPLX_FABLE_K20_LOG")
    parser.add_argument(
        "--cap",
        choices=("reach", "one"),
        default="reach",
        help="water-fill cap; 'reach' is budget-exact, 'one' is H's literal form",
    )
    parser.add_argument(
        "--ms-per-window",
        type=float,
        default=None,
        help="verify-window wall time, to convert tokens/window into tok/s "
        "(H/cost.py calibrates the M4 baseline at 37.47)",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N cycles")
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    parser.add_argument(
        "--allow-replay-mismatch",
        action="store_true",
        help="warn instead of failing when the current-law replay disagrees "
        "with the logged device decision",
    )
    args = parser.parse_args(argv)

    log = load_log(args.npz)
    result = score(log, cap_mode=args.cap, limit=args.limit)
    print(report(result, ms_per_window=args.ms_per_window))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if result["replay_mismatch_cycles"] and not args.allow_replay_mismatch:
        print(
            "\nFAIL: the current-law replay no longer mirrors the device "
            "kernel, so every number above is unreliable.",
            file=sys.stderr,
        )
        return 1
    if result["ladder_all_one_cycles"] and result["ladder_all_one_agree_fraction"] < 1.0:
        print(
            "\nFAIL: block verification diverged on a c = 1 window, where it "
            "is provably identical to the shipped law.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
