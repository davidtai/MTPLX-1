"""Opt-in per-cycle capture of the K20 rows a verify decision holds.

Why
---
``H-tokens-per-window-design.md`` §7 names this the one instrumentation gap
that gates three of its four acceptance options:

* **Option B (block verification).**  Its gain lives entirely in the cycles
  whose reach credit ``c`` has dropped below 1, and the retained receipts
  truncate ``drafts[].accept_probability`` at the first rejection -- exactly
  the cycles B pays in.  With the rows on disk the gain is an *exact* offline
  number instead of a surrogate, which matters because the per-run acceptance
  spread is +-4.2% (1 sigma) and a live A/B cannot resolve a +4% effect.
* **Option D (draft-proposal recalibration).**  The whole temperature /
  top-p / top-k / tilt sweep becomes a pure-CPU replay against the exact
  objective ``sum(min(p, q))``, with no GPU lock and no exactness risk.
* **The draft's rank of the selected token** -- the question the receipts
  cannot answer at all (H §1.3).

Two lanes, two layouts
----------------------
The decision that turns K20 rows into an accepted count lives in two different
places, and this module captures both.  Which one a run uses is a property of
the run, not a setting here: the log records whichever fires.

``pr391_raw`` -- the opt-in softfloat64 D3/M4 device lane
    (``abba_driver --d3-softfloat64-route``).  The decision is a Metal kernel
    (``kernels/pr391_softfloat64_verifier_decision.py``) fed **raw** rows: the
    top-20 ids, the top-20 raw logits, and ``exp(v - logsumexp(full row))``,
    the softmax over the *whole* vocabulary.  Top-p 0.95 and the double
    renormalisation happen *inside* the kernel, so the log holds the kernel's
    input and an offline scorer must re-run that preparation itself.  The
    draft ids have already been mapped through the 65,536-row frspec rank
    table (``generation.py:5606-5609``), so draft and target ids share one id
    space.  Sources: ``generation.py:5591-5605`` (draft), ``:5715-5741``
    (target).

``stock_prepared`` -- the retained stock native-MTP lane, the default
    The draft and the accept/correct decision run on the **host** in NumPy,
    in ``generate_mtpk``'s accept loop.  The rows reaching that decision are
    ``SparseDistribution`` / ``BatchedSparseDistributions`` objects that are
    **already fully shaped**: temperature applied, top-p then top-k applied
    (``sampling.py:122-176``), and renormalised twice.  Consequences that an
    offline scorer must respect, and that this docstring is the contract for:

    * **No raw logits reach the host.**  ``has_raw_logits`` is 0 and
      ``values`` carries ``log(probs)``.  That ranks identically (the shaping
      is monotone in the logit) and is exactly what a re-temperature consumes,
      but it is *not* a logit and must not be read as one.
    * **No truncated tail survives.**  The mass outside top-p/top-k is gone
      before the host sees the row, so an offline sweep can only ever *shrink*
      the support, never recover what shaping dropped.
    * **A temperature is already baked in.**  ``temperature`` /
      ``draft_temperature`` / ``top_p`` / ``top_k`` are recorded so that
      ``p' ∝ p ** (T / T')`` re-tempers the retained support exactly.

    Everything logged here is a copy of an array the lane already built on the
    host for its own decision.  No device work, no new ``mx.eval``.

    A **greedy** run (``temperature <= 0``) builds no distributions at all --
    acceptance is argmax equality -- so its windows are recorded with
    ``greedy = 1`` and no rows.  Neither Option B nor Option D is defined on a
    greedy lane; :meth:`flush` says so rather than writing a silently empty
    file.

What is recorded, per verify window
-----------------------------------
Both layouts normalise to one on-disk schema, so the offline scorers load
either file through the same path and branch only where the two genuinely
differ -- whether the rows still need the kernel's preparation.

===========================  ==========================  ===================
key                          dtype / shape               meaning
===========================  ==========================  ===================
``layout``                   str (0-d)                   ``pr391_raw`` /
                                                         ``stock_prepared``
``has_raw_logits``           uint8 (0-d)                 1 iff values are logits
``draft_ids``                uint32  ``[C, D, K]``       draft row support
``draft_values``             float32 ``[C, D, K]``       logits, or log(prob)
``draft_probs``              float64 ``[C, D, K]``       see the layout notes
``draft_valid``              uint8   ``[C, D]``          row present
``target_ids``               uint32  ``[C, R, K]``       target row support
``target_values``            float32 ``[C, R, K]``
``target_probs``             float64 ``[C, R, K]``
``target_valid``             uint8   ``[C, R]``
``draft_tokens``             uint32  ``[C, D]``          selected draft chain
``primary``                  uint32  ``[C]``             window's first token
``decision_uniforms``        float64 ``[C, D+1]``        accept coins, then the
                                                         correction/bonus draw
``decision_uniforms_valid``  uint8   ``[C]``             real entries of the above
``draft_uniforms``           float64 ``[C, D]``          draft-select draws (pr391)
``accepted``                 uint32  ``[C]``             accepted depth count
``first_reject``             int32   ``[C]``             -1 when none
``selected_token``           uint32  ``[C]``             correction or bonus
``selected_kind``            uint32  ``[C]``             0 none / 1 corr / 2 bonus
``selected_present``         uint8   ``[C]``
``draws_used``               uint32  ``[C]``
``accept_probability``       float64 ``[C, D]``          alpha, truncated
``accept_probability_valid`` uint8   ``[C]``             real entries of the above
``bonus_allowed``            uint8   ``[C]``
``greedy``                   uint8   ``[C]``
``descriptor_offset``        int64   ``[C]``             PCG64 tape offset (pr391)
``rng_state``                uint64  ``[C, 4]``          PCG64 state/inc (stock)
``stop_ids``                 uint32  ``[S]``             request's stop set
``temperature``              float64 (0-d)
``draft_temperature``        float64 (0-d)
``top_p`` / ``top_k``        float64 / int64 (0-d)
===========================  ==========================  ===================

**alpha beyond the first rejection is not in the log, because neither lane
computes it.**  The kernel returns as soon as a depth rejects (kernel lines
289-320); the stock accept loop ``break``s.  ``accept_probability_valid``
records how many entries are real.  The *rows* are not truncated -- the draft
chain always runs to full depth, and the batched target support is
materialised for the whole window -- so an offline scorer recomputes the full
per-depth alpha ladder exactly.  That is the whole point of logging rows.

Likewise ``decision_uniforms_valid``: the stock lane draws an accept coin only
for the depths it actually reaches, so a counterfactual that accepts deeper has
no logged draw to use.  ``rng_state`` is the window's PCG64 state, from which a
scorer derives a deterministic auxiliary stream for those depths -- the same
stream for every law it compares, so the comparison stays paired.

Cost
----
``MTPLX_FABLE_K20_LOG`` is read exactly once, at import.  When unset,
:data:`_ENABLED` is ``False``, every call site is behind a module-level
constant in ``generation.py``, and this costs one global load plus one
predicted-not-taken branch per verify cycle.

When on:

* **pr391 lane** -- one small device-to-host copy per cycle and *no extra
  synchronisation*.  The six row arrays are already inputs to the verifier
  kernel, so adding them to the decision's own ``mx.eval`` only retains them.
  ``7 x 20 x (uint32 + 2 x float32) = 1,680 B`` per cycle, ~0.66 MB over a
  1,024-token request.
* **stock lane** -- no device work at all.  Every array copied is one the host
  already built for its own decision; the cost is ``np.asarray`` copies and a
  list append.

Either way the copies sit on the critical path between the decision and the
commit, so **an instrumented run is a data run, not a timing run.**  Read
tok/s off an un-instrumented arm.

Guard
-----
If the log is armed and the process exits without ever writing a row, the
atexit hook prints why and forces a **non-zero exit status**.  A run that
silently produced nothing is the failure this instrumentation exists to
prevent.  ``sys.exit`` inside an ``atexit`` callback is swallowed by CPython
(the traceback prints and the status stays 0), so the hook uses ``os._exit``
after flushing the streams; it is registered at import and ``atexit`` runs
LIFO, so it fires at the end of the chain.

Usage::

    MTPLX_FABLE_K20_LOG=/path/rows.npz <benchmark command>
    python scripts/fable/offline_block_verification.py /path/rows.npz
    python scripts/fable/offline_draft_temperature.py /path/rows.npz

A ``.json`` path is accepted too; the arrays always land in the ``.npz``
sibling and the ``.json`` gets a small pointer document.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any, Sequence

import numpy as np

_ENV_VAR = "MTPLX_FABLE_K20_LOG"

K20 = 20
DEPTH = 3
TARGET_ROWS = DEPTH + 1
DECISION_DRAWS = DEPTH + 1

LAYOUT_PR391 = "pr391_raw"
LAYOUT_STOCK = "stock_prepared"

SELECTED_NONE = 0
SELECTED_CORRECTION = 1
SELECTED_BONUS = 2

#: Padding ids for a support narrower than ``K20``.  Well above any real
#: vocabulary id and distinct from each other, so a prepared row stays unique;
#: their probability is 0, so every consumer drops them.
_PAD_ID_BASE = 0xFFFFFFFF
_PAD_VALUE = np.float32(-3.0e38)

EMPTY_MESSAGE = (
    "[fable-k20-log] nothing recorded for {path!r}; neither the PR391 "
    "float32 D3 route nor the stock native-MTP accept loop ran a verify "
    "decision."
)


def _env_path() -> str | None:
    raw = (os.environ.get(_ENV_VAR) or "").strip()
    return raw or None


def _npz_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return path if ext == ".npz" else root + ".npz"


def _row_arrays(
    ids: Any, values: Any, probs: Any, width: int = K20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad one support to ``width`` with zero-probability sentinel ids."""

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if values is None:
        # No logits on this lane: log(prob) ranks identically under any
        # monotone shaping and is exactly what a re-temperature consumes.
        with np.errstate(divide="ignore"):
            values = np.where(probs > 0.0, np.log(probs), float(_PAD_VALUE))
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    size = int(ids.size)
    if size > width:
        raise ValueError(f"K20 log: support of {size} exceeds width {width}")
    padding = width - size
    out_ids = np.concatenate(
        (
            ids.astype(np.uint32, copy=False),
            np.array(
                [_PAD_ID_BASE - offset for offset in range(padding)], dtype=np.uint32
            ),
        )
    )
    out_values = np.concatenate(
        (values, np.full(padding, _PAD_VALUE, dtype=np.float32))
    )
    out_probs = np.concatenate((probs, np.zeros(padding, dtype=np.float64)))
    return out_ids, out_values, out_probs


def _distribution_rows(distribution: Any) -> tuple[Any, Any, Any] | None:
    """``(ids, values, probs)`` from a SparseDistribution-like host object."""

    if distribution is None:
        return None
    ids = getattr(distribution, "token_ids", None)
    probs = getattr(distribution, "probs", None)
    if ids is not None and probs is not None:
        return ids, None, probs
    array = np.asarray(distribution, dtype=np.float64).reshape(-1)
    keep = np.flatnonzero(array > 0.0)
    return keep.astype(np.int64), None, array[keep]


class K20RowLog:
    """Buffer one window's K20 rows per verify cycle; write once at the end."""

    def __init__(self, path: str | None = None) -> None:
        self.configure(path)

    # -- lifecycle ---------------------------------------------------
    def configure(self, path: str | None) -> None:
        """(Re)point the log.  Also the test seam; never a hot path."""

        self.path = path
        self.enabled = path is not None
        self.layout: str | None = None
        self._rows: list[dict[str, Any]] = []
        self._open: dict[str, Any] | None = None
        self._stop_ids = np.zeros(0, dtype=np.uint32)
        self._meta: dict[str, Any] = {}
        self._written: str | None = None

    def set_stop_ids(self, stop_ids: Any) -> None:
        """Record the request's stop-token set once, for exact replay.

        Per *request*, not per cycle: a second request through the same process
        overwrites it.
        """

        if not _ENABLED or not self.enabled:
            return
        self._stop_ids = np.asarray(stop_ids, dtype=np.uint32).reshape(-1).copy()

    def set_sampler(self, *, sampler: Any, draft_sampler: Any) -> None:
        """Record the shaping the rows were produced under.

        The stock lane's rows arrive already shaped, so an offline
        re-temperature needs the temperature that shaped them.
        """

        if not _ENABLED or not self.enabled:
            return
        temperature = float(getattr(sampler, "temperature", 1.0))
        self._meta = {
            "temperature": temperature,
            "draft_temperature": float(
                getattr(draft_sampler, "temperature", temperature)
            ),
            "top_p": float(getattr(sampler, "top_p", 1.0)),
            "top_k": int(getattr(sampler, "top_k", 0)),
        }

    def _claim_layout(self, layout: str) -> None:
        if self.layout is None:
            self.layout = layout
        elif self.layout != layout:
            raise RuntimeError(
                f"K20 log already holds {self.layout!r} rows; refusing to mix "
                f"in {layout!r}. One lane per file."
            )

    # -- pr391 softfloat64 device lane --------------------------------
    def record(
        self,
        *,
        draft_result: Sequence[Any],
        target_support: Sequence[Any],
        uniform_tape: Any,
        reservation: Any,
        primary: int,
        bonus_allowed: int,
        decision: Sequence[Any],
    ) -> None:
        """Copy one PR391 D3/M4 cycle's rows to the host.

        Called from ``_pr391_decode_float32_verifier_decision`` *after* its own
        ``mx.eval``, which the caller has already widened to include these six
        row arrays.  Nothing here evaluates, syncs, or touches the device
        beyond the ``np.asarray`` copies themselves.
        """

        if not _ENABLED or not self.enabled:
            return
        self._claim_layout(LAYOUT_PR391)

        offset = int(reservation.offset)
        tape = np.asarray(uniform_tape.values, dtype=np.float64)
        decision_uniforms = tape[offset : offset + DECISION_DRAWS]
        if decision_uniforms.size != DECISION_DRAWS:
            raise RuntimeError("K20 log: decision reservation is short")
        draft_start = offset - DEPTH
        if draft_start < 0:
            raise RuntimeError("K20 log: draft reservation precedes the tape")

        (
            accepted,
            first_reject,
            selected_token,
            selected_kind,
            selected_present,
            draws_used,
            accept_probs,
        ) = decision
        draft_ids = np.asarray(draft_result[1], dtype=np.int64).reshape(DEPTH, K20)
        draft_values = np.asarray(draft_result[2], dtype=np.float32).reshape(
            DEPTH, K20
        )
        draft_probs = np.asarray(draft_result[3], dtype=np.float64).reshape(DEPTH, K20)
        target_ids = np.asarray(target_support[0], dtype=np.int64).reshape(
            TARGET_ROWS, K20
        )
        target_values = np.asarray(target_support[1], dtype=np.float32).reshape(
            TARGET_ROWS, K20
        )
        target_probs = np.asarray(target_support[2], dtype=np.float64).reshape(
            TARGET_ROWS, K20
        )

        self._rows.append(
            {
                "draft_tokens": [
                    int(token)
                    for token in np.asarray(
                        draft_result[0], dtype=np.uint32
                    ).reshape(DEPTH)
                ],
                "draft_rows": [
                    (draft_ids[d], draft_values[d], draft_probs[d])
                    for d in range(DEPTH)
                ],
                "target_rows": [
                    (target_ids[r], target_values[r], target_probs[r])
                    for r in range(TARGET_ROWS)
                ],
                "primary": int(primary),
                "decision_uniforms": [float(value) for value in decision_uniforms],
                "decision_uniforms_valid": DECISION_DRAWS,
                "draft_uniforms": [float(value) for value in tape[draft_start:offset]],
                "accepted": int(accepted),
                "first_reject": int(first_reject),
                "selected_token": int(selected_token),
                "selected_kind": int(selected_kind),
                "selected_present": bool(selected_present),
                "draws_used": int(draws_used),
                "accept_probability": [float(value) for value in accept_probs],
                "accept_probability_valid": (
                    DEPTH if int(first_reject) < 0 else int(first_reject) + 1
                ),
                "bonus_allowed": int(bool(bonus_allowed)),
                "greedy": 0,
                "descriptor_offset": offset,
                "rng_state": [0, 0, 0, 0],
            }
        )

    # -- stock native-MTP host lane -----------------------------------
    def stock_open(
        self,
        *,
        primary: int,
        draft_tokens: Sequence[Any],
        draft_probs: Sequence[Any],
        target_batch: Any = None,
        target_list: Any = None,
        bonus_allowed: bool,
        greedy: bool,
        rng: Any = None,
    ) -> None:
        """Open one stock verify window and copy every row already on the host.

        The draft chain always runs to full depth, so ``draft_probs`` is
        complete here even for the depths the accept loop will never reach.
        The batched target support is materialised for the whole window up
        front on the retained lane, so those rows are complete too; the lazy
        per-row path fills them in through :meth:`stock_depth` instead and
        leaves ``target_valid`` at 0 for rows it never built.

        Every array copied is one the lane already owns.  Nothing here touches
        the device.  An already-open window is closed first, so no accept-loop
        ``break`` or ``continue`` needs its own hook.
        """

        if not _ENABLED or not self.enabled:
            return
        self._claim_layout(LAYOUT_STOCK)
        self._close_open()

        depth = len(draft_tokens)
        rows: list[tuple[Any, Any, Any] | None] = []
        for index in range(depth):
            source = draft_probs[index] if index < len(draft_probs) else None
            rows.append(_distribution_rows(source))

        target_rows: list[tuple[Any, Any, Any] | None] = []
        if target_batch is not None:
            ids = np.asarray(target_batch.token_ids, dtype=np.int64)
            probs = np.asarray(target_batch.probs, dtype=np.float64)
            for row in range(int(ids.shape[0])):
                keep = probs[row] > 0.0
                target_rows.append((ids[row][keep], None, probs[row][keep]))
        elif target_list is not None:
            for distribution in target_list:
                target_rows.append(_distribution_rows(distribution))
        while len(target_rows) < depth + 1:
            target_rows.append(None)

        self._open = {
            "draft_tokens": [int(token) for token in draft_tokens],
            "draft_rows": rows,
            "target_rows": target_rows,
            "primary": int(primary),
            "decision_uniforms": [float("nan")] * (depth + 1),
            "decision_uniforms_valid": 0,
            "draft_uniforms": [float("nan")] * depth,
            "accepted": 0,
            "first_reject": -1,
            "selected_token": 0,
            "selected_kind": SELECTED_NONE,
            "selected_present": False,
            "draws_used": 0,
            "accept_probability": [0.0] * depth,
            "accept_probability_valid": 0,
            "bonus_allowed": int(bool(bonus_allowed)),
            "greedy": int(bool(greedy)),
            "descriptor_offset": -1,
            "rng_state": _pcg64_state(rng),
        }

    def stock_depth(
        self,
        depth_index: int,
        *,
        target_p: Any = None,
        accept_prob: float,
        coin: float | None,
        accepted: bool,
        correction: int,
    ) -> None:
        """Record one depth's decision inside the stock accept loop."""

        if not _ENABLED or not self.enabled or self._open is None:
            return
        window = self._open
        index = int(depth_index)
        if index >= len(window["accept_probability"]):
            return
        if target_p is not None and window["target_rows"][index] is None:
            window["target_rows"][index] = _distribution_rows(target_p)
        window["accept_probability"][index] = float(accept_prob)
        window["accept_probability_valid"] = index + 1
        if coin is not None:
            window["decision_uniforms"][index] = float(coin)
            window["decision_uniforms_valid"] = index + 1
        if accepted:
            window["accepted"] = index + 1
            window["draws_used"] = index + 1
            return
        window["first_reject"] = index
        window["selected_token"] = int(correction)
        window["selected_kind"] = SELECTED_CORRECTION
        window["selected_present"] = True
        window["draws_used"] = index + 2

    def stock_bonus(self, token: int) -> None:
        """Record the all-accept bonus token for the open stock window."""

        if not _ENABLED or not self.enabled or self._open is None:
            return
        window = self._open
        if window["first_reject"] >= 0:
            return
        window["selected_token"] = int(token)
        window["selected_kind"] = SELECTED_BONUS
        window["selected_present"] = True
        window["draws_used"] = len(window["draft_tokens"]) + 1

    def stock_close(self) -> None:
        """Close the open stock window.  Idempotent; safe on every path."""

        if not _ENABLED or not self.enabled:
            return
        self._close_open()

    def _close_open(self) -> None:
        if self._open is not None:
            self._rows.append(self._open)
            self._open = None

    # -- output ------------------------------------------------------
    @property
    def cycles(self) -> int:
        return len(self._rows) + (1 if self._open is not None else 0)

    def _pack(self) -> dict[str, np.ndarray]:
        """Normalise every buffered window into the one on-disk schema."""

        rows = self._rows
        count = len(rows)
        depth = max(len(row["draft_tokens"]) for row in rows)
        target_count = max(len(row["target_rows"]) for row in rows)
        blank = _row_arrays(np.zeros(0), None, np.zeros(0))

        draft_ids = np.zeros((count, depth, K20), dtype=np.uint32)
        draft_values = np.zeros((count, depth, K20), dtype=np.float32)
        draft_probs = np.zeros((count, depth, K20), dtype=np.float64)
        draft_valid = np.zeros((count, depth), dtype=np.uint8)
        target_ids = np.zeros((count, target_count, K20), dtype=np.uint32)
        target_values = np.zeros((count, target_count, K20), dtype=np.float32)
        target_probs = np.zeros((count, target_count, K20), dtype=np.float64)
        target_valid = np.zeros((count, target_count), dtype=np.uint8)
        draft_tokens = np.zeros((count, depth), dtype=np.uint32)
        uniforms = np.full((count, depth + 1), np.nan, dtype=np.float64)
        draft_uniforms = np.full((count, depth), np.nan, dtype=np.float64)
        accept_probability = np.zeros((count, depth), dtype=np.float64)

        for index, row in enumerate(rows):
            for slot, entry in enumerate(row["draft_rows"][:depth]):
                packed = blank if entry is None else _row_arrays(*entry)
                draft_ids[index, slot] = packed[0]
                draft_values[index, slot] = packed[1]
                draft_probs[index, slot] = packed[2]
                draft_valid[index, slot] = 0 if entry is None else 1
            for slot, entry in enumerate(row["target_rows"][:target_count]):
                packed = blank if entry is None else _row_arrays(*entry)
                target_ids[index, slot] = packed[0]
                target_values[index, slot] = packed[1]
                target_probs[index, slot] = packed[2]
                target_valid[index, slot] = 0 if entry is None else 1
            tokens = row["draft_tokens"]
            draft_tokens[index, : len(tokens)] = np.asarray(tokens, dtype=np.uint32)
            drawn = row["decision_uniforms"]
            uniforms[index, : len(drawn)] = np.asarray(drawn, dtype=np.float64)
            selects = row["draft_uniforms"]
            draft_uniforms[index, : len(selects)] = np.asarray(
                selects, dtype=np.float64
            )
            alpha = row["accept_probability"]
            accept_probability[index, : len(alpha)] = np.asarray(
                alpha, dtype=np.float64
            )

        def column(key: str, dtype: Any) -> np.ndarray:
            return np.asarray([row[key] for row in rows], dtype=dtype)

        return {
            "layout": np.asarray(self.layout or LAYOUT_STOCK),
            "has_raw_logits": np.asarray(
                1 if self.layout == LAYOUT_PR391 else 0, dtype=np.uint8
            ),
            "draft_ids": draft_ids,
            "draft_values": draft_values,
            "draft_probs": draft_probs,
            "draft_valid": draft_valid,
            "target_ids": target_ids,
            "target_values": target_values,
            "target_probs": target_probs,
            "target_valid": target_valid,
            "draft_tokens": draft_tokens,
            "primary": column("primary", np.uint32),
            "decision_uniforms": uniforms,
            "decision_uniforms_valid": column("decision_uniforms_valid", np.uint8),
            "draft_uniforms": draft_uniforms,
            "accepted": column("accepted", np.uint32),
            "first_reject": column("first_reject", np.int32),
            "selected_token": column("selected_token", np.uint32),
            "selected_kind": column("selected_kind", np.uint32),
            "selected_present": column("selected_present", np.uint8),
            "draws_used": column("draws_used", np.uint32),
            "accept_probability": accept_probability,
            "accept_probability_valid": column("accept_probability_valid", np.uint8),
            "bonus_allowed": column("bonus_allowed", np.uint8),
            "greedy": column("greedy", np.uint8),
            "descriptor_offset": column("descriptor_offset", np.int64),
            "rng_state": np.asarray(
                [row["rng_state"] for row in rows], dtype=np.uint64
            ),
            "stop_ids": self._stop_ids,
            "temperature": np.asarray(
                self._meta.get("temperature", float("nan")), dtype=np.float64
            ),
            "draft_temperature": np.asarray(
                self._meta.get("draft_temperature", float("nan")), dtype=np.float64
            ),
            "top_p": np.asarray(
                self._meta.get("top_p", float("nan")), dtype=np.float64
            ),
            "top_k": np.asarray(int(self._meta.get("top_k", 0)), dtype=np.int64),
        }

    def flush(self) -> str | None:
        """Write the log out.  Safe to call repeatedly; no-op when off."""

        if not _ENABLED or not self.enabled:
            return None
        self._close_open()
        if not self._rows:
            print(EMPTY_MESSAGE.format(path=self.path), file=sys.stderr)
            return None

        arrays = self._pack()
        out = _npz_path(self.path)
        directory = os.path.dirname(os.path.abspath(out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(out, **arrays)
        if self.path.endswith(".json"):
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "npz": out,
                        "layout": self.layout,
                        "cycles": int(len(self._rows)),
                        "depth": int(arrays["draft_tokens"].shape[1]),
                        "target_rows": int(arrays["target_ids"].shape[1]),
                        "top_k": K20,
                    },
                    handle,
                    indent=2,
                )
        self._written = out
        greedy = int(np.sum(arrays["greedy"]))
        note = ""
        if greedy:
            note = (
                f" WARNING: {greedy}/{len(self._rows)} windows are greedy "
                "(temperature <= 0): acceptance is argmax equality and those "
                "windows carry no distributions, so neither the block-"
                "verification nor the draft-temperature scorer applies."
            )
        print(
            f"[fable-k20-log] wrote {out} layout={self.layout} "
            f"cycles={len(self._rows)}{note}",
            file=sys.stderr,
        )
        return out


def _pcg64_state(rng: Any) -> list[int]:
    """Split a PCG64 generator's 128-bit state/inc into four uint64 words.

    Returned as zeros for anything that is not a PCG64 ``Generator`` (the
    PR391 lane's uniform tape, or a test stub).  An offline scorer uses this to
    build the deterministic auxiliary stream it needs for depths the stock lane
    never drew a coin for.
    """

    bit_state = getattr(getattr(rng, "bit_generator", None), "state", None)
    if not isinstance(bit_state, dict):
        return [0, 0, 0, 0]
    inner = bit_state.get("state")
    if not isinstance(inner, dict):
        return [0, 0, 0, 0]
    value = int(inner.get("state", 0))
    increment = int(inner.get("inc", 0))
    mask = 0xFFFFFFFFFFFFFFFF
    return [
        (value >> 64) & mask,
        value & mask,
        (increment >> 64) & mask,
        increment & mask,
    ]


_LOG_PATH = _env_path()
_ENABLED = _LOG_PATH is not None

k20_log = K20RowLog(_LOG_PATH)


def _atexit_flush() -> None:
    """Flush at exit and fail the process when an armed run captured nothing.

    ``sys.exit`` inside an ``atexit`` callback is swallowed by CPython (the
    traceback prints and the status stays 0), so the hard failure goes through
    ``os._exit`` after the streams are flushed.  This handler is registered at
    import and ``atexit`` runs LIFO, so it fires at the end of the chain.
    """

    if not _ENABLED or not k20_log.enabled:
        return
    k20_log.flush()
    if k20_log._written is not None:
        return
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)


if _ENABLED:
    atexit.register(_atexit_flush)


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_K20_LOG`` named a path at import."""

    return _ENABLED


def _configure_for_test(path: str | None) -> None:
    """Re-point the module-level singleton and enable flag (tests only)."""

    global _ENABLED, _LOG_PATH
    _LOG_PATH = path
    _ENABLED = path is not None
    k20_log.configure(path)


__all__ = [
    "K20RowLog",
    "LAYOUT_PR391",
    "LAYOUT_STOCK",
    "is_enabled",
    "k20_log",
]
