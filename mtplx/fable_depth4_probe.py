"""Opt-in depth-4 draft probe: the go/no-go measurement for gated M=5.

Why
---
``L-fable-decode-ideas.md`` §D sizes *confidence-gated* depth 4.  ``H`` killed
adaptive depth from **history** (acceptance is memoryless across windows), but
within a window the drafter's own probability of the token it drafted predicts
the target's acceptance strongly (``L_gate_out.txt``, 1,113 logged windows):

===============  =====  =====  =====
``q(x_d)`` bin   a1     a2     a3
===============  =====  =====  =====
< 0.20 (28%)     0.54   0.46   0.43
0.80-0.95        0.77   0.79   0.77
>= 0.95 (22%)    0.96   0.91   0.89
===============  =====  =====  =====

Gating a 4th draft step on ``q(x_3) > 0.8`` fires on 30% of windows, of which
52% accepted all three drafts, and is worth **+0.147 tok/window for +0.89 ms**
-- *if* ``alpha_4`` on those windows is at least 0.75.  Ungated depth 4 is
-0.5%, exactly as H found.  So the whole program turns on one number that
nothing in the ledger measures: **alpha_4 conditional on the gate.**

The measurement needs no M=5 verify graph
-----------------------------------------
After a normal M4 cycle whose three drafts were **all accepted**, the target's
bonus row is ``p(. | primary, d1, d2, d3)`` -- which is *exactly* the
distribution a fourth draft would have been verified against.  So:

1. run one extra ``rt.draft_mtp(..., mtp_depth=4)`` from the d3 hidden state
   and the d3 token,
2. shape its row with the same draft sampler the real drafts used,
3. log that row (``q_4``) next to the bonus target row (``p_3``) the K20 log
   already captures,
4. score ``alpha_4 = sum_x min(p_3(x), q_4(x))`` offline
   (``scripts/fable/offline_depth4_gate.py``).

``sum min(p, q)`` is the expected acceptance of a draw from ``q`` under the
Leviathan-Chen law, so this is the honest estimator and it integrates out the
accept coin -- which is why **nothing here samples and nothing here draws a
uniform.**  The probe is a pure read.

What the probe must never do, and does not
------------------------------------------
* **Change a token.**  The row it computes is written to the K20 log and to
  nothing else.  No commit, no ``pending_primary``, no bonus.
* **Consume RNG.**  ``_distribution_from_mlx_logits`` takes no generator; the
  probe never calls ``rng``.  ``tests/test_fable_depth4_probe.py`` asserts
  this by source inspection of the hook *and* by driving :func:`run_probe`
  with a generator stub that fails on any draw.
* **Move the MTP history.**  ``rt.draft_mtp`` appends one speculative row to
  the QSA cache it is handed.  :func:`run_probe` records the offset first and
  restores it in a ``finally``, so the offset the production commit sees is
  the one it would have seen.  (The all-accept path then trims to
  ``cycle_offset + 1`` regardless, so this is belt *and* braces -- the belt is
  what the test checks.)

Cost
----
An armed cycle pays one MTP-layer forward (~1.2 ms) plus one full-vocab K20
selection (~0.4 ms) -- ~1.6 ms on the ~31% of windows that accept all three
drafts.  **An armed run is a data run, not a timing run.**  The probe's own
seconds land in ``event["timing_s"]["fable_depth4_probe"]`` so they are
visible rather than folded into ``draft_time``.

``MTPLX_FABLE_DEPTH4_PROBE`` is read exactly once, at import.  When unset the
hook is behind a module-level constant in ``generation.py`` and costs one
predicted-not-taken branch on the all-accept path.  The probe writes through
``mtplx.fable_k20_log``, so it also needs ``MTPLX_FABLE_K20_LOG`` pointed at a
path; armed without it, it is inert (there is nowhere to put the row).

Usage::

    MTPLX_FABLE_DEPTH4_PROBE=1 MTPLX_FABLE_K20_LOG=/path/d4.npz <benchmark>
    python scripts/fable/offline_depth4_gate.py /path/d4.npz
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

_ENV_VAR = "MTPLX_FABLE_DEPTH4_PROBE"

#: Log width.  Mirrors ``fable_k20_log.K20`` without importing it -- the
#: dependency runs the other way (the log imports :func:`gate_feature`), and a
#: back-import here would close the cycle.
K20 = 20


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_ENABLED = _env_truthy(_ENV_VAR)


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_DEPTH4_PROBE`` was set at import."""

    return _ENABLED


def _configure_for_test(enabled: bool) -> None:
    """Flip the module gate (tests only); never a hot path."""

    global _ENABLED
    _ENABLED = bool(enabled)


def support_from_distribution(
    distribution: Any, *, width: int = K20
) -> tuple[np.ndarray, np.ndarray]:
    """``(ids, probs)`` for one already-shaped host row, trimmed to ``width``.

    ``distribution`` is whatever ``_distribution_from_mlx_logits`` returned for
    the probe's logits: a ``SparseDistribution`` (the sparse top-k path, which
    is what the production cell's ``top_k=20`` sampler takes) or a dense
    ``np.ndarray`` over the vocabulary (the fallback path).

    The row arrives with temperature, top-p and top-k already applied and
    renormalised, exactly like the three real draft rows the K20 log stores --
    so it needs no further shaping and gets none.  The only thing that can
    happen here is a **trim**: the log's fixed ``K20`` columns cannot hold a
    wider support, so a row with more than ``width`` positive entries is cut to
    its ``width`` largest, ranked ``(probability desc, id asc)`` -- the same
    tie-break ``fast_sampling._deterministic_mlx_top_k_support`` uses.  On the
    production cell (``top_k=20``) nothing is ever trimmed; when something is,
    the caller records it so the offline scorer can say so.

    The trimmed row is **not** renormalised here.  The offline scorer's
    ``prepared_row`` renormalises every stock row on load, which is exactly
    what the three real draft rows get; doing it twice would be the only way
    these four rows could stop being comparable.
    """

    ids = getattr(distribution, "token_ids", None)
    probs = getattr(distribution, "probs", None)
    if ids is None or probs is None:
        dense = np.asarray(distribution, dtype=np.float64).reshape(-1)
        keep = np.flatnonzero(dense > 0.0)
        ids = keep.astype(np.int64)
        probs = dense[keep]
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if ids.size != probs.size:
        raise ValueError("depth-4 probe row has mismatched ids/probs")
    keep = probs > 0.0
    ids = ids[keep]
    probs = probs[keep]
    if ids.size == 0:
        raise ValueError("depth-4 probe row retained no mass")
    if ids.size > int(width):
        # (probability desc, id asc), then back to the caller in that order.
        rank = np.lexsort((ids, -probs))[: int(width)]
        ids = ids[rank]
        probs = probs[rank]
    return ids, probs


def run_probe(
    *,
    draft_step: Callable[[], Any],
    shape_row: Callable[[Any], Any],
    mtp_cache: Any,
    read_offset: Callable[[Any], int],
    rollback: Callable[[Any, int], None],
    remap_ids: Callable[[np.ndarray], np.ndarray] | None = None,
    width: int = K20,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """One depth-4 draft step, shaped, with the MTP cache offset restored.

    Every device-touching piece is injected, so this function -- the part that
    owns the offset contract -- is exercised by the pure-python tests with no
    MLX in the process.  ``generation.py`` binds:

    ``draft_step``
        ``rt.draft_mtp(d3_hidden, [[d3_token]], mtp_cache=..., mtp_depth=4,
        position_offset=...)`` reduced to its last-position logits row.
    ``shape_row``
        ``_distribution_from_mlx_logits(row, draft_sampler)`` -- the *draft*
        sampler, so ``q_4`` is shaped exactly like ``q_1..q_3``.
    ``read_offset`` / ``rollback``
        ``_mtp_cache_offset`` / ``_rollback_mtp_cache``.
    ``remap_ids``
        the ``_frspec_legacy_ids`` gather the real draft applies, so probe ids
        land in the target's id space and the offline scorer can intersect
        them with the bonus target row.

    Returns ``(ids, probs, trimmed)``.  ``trimmed`` is True when the shaped row
    was wider than ``width`` and had to be cut (never on the production cell).

    The rollback runs in a ``finally``: a raising draft step must not leave a
    speculative row staged in the production MTP history.
    """

    offset = int(read_offset(mtp_cache))
    try:
        distribution = shape_row(draft_step())
    finally:
        rollback(mtp_cache, offset)
    ids, probs = support_from_distribution(distribution, width=width)
    trimmed = False
    raw_size = int(getattr(distribution, "token_ids", ids).size)
    if raw_size > int(width):
        trimmed = True
    if remap_ids is not None:
        ids = np.asarray(remap_ids(ids), dtype=np.int64).reshape(-1)
    return ids, probs, trimmed


def gate_feature(
    draft_ids: Any, draft_probs: Any, draft_token: int
) -> float:
    """``q(x_d)`` -- the drafter's own probability of the token it drafted.

    The gate feature of ``L`` §D.  Computed on the host from a row the accept
    loop already owns; zero when the token is not in the (shaped) support,
    which cannot happen for a token *sampled* from that row but can for one
    substituted by the correction cache or the top-k reranker -- and a zero
    there correctly excludes that window from every gate.
    """

    ids = np.asarray(draft_ids, dtype=np.int64).reshape(-1)
    probs = np.asarray(draft_probs, dtype=np.float64).reshape(-1)
    hits = np.nonzero(ids == int(draft_token))[0]
    return 0.0 if hits.size == 0 else float(probs[int(hits[0])])


__all__ = [
    "K20",
    "gate_feature",
    "is_enabled",
    "run_probe",
    "support_from_distribution",
]
