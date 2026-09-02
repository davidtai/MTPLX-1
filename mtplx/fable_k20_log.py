"""Opt-in per-cycle capture of the seven prepared K20 rows the PR391 D3/M4
verifier decision already holds.

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

What is recorded, per verify cycle
----------------------------------
The decision ABI (``kernels/pr391_softfloat64_verifier_decision.py:200-235``)
is 3 draft rows ``[3, 20]`` and 4 target rows ``[4, 20]``, each carrying
``ids`` / ``values`` / ``probs``.  All seven rows are logged verbatim, in the
kernel's own input form -- **before** the kernel's ``_prepare_candidate_row``
top-p-0.95 truncation and double renormalisation (lines 33-105).  That is
deliberate: the offline scorers re-run that preparation themselves, so the
log is the kernel's *input*, not a partially-cooked intermediate.

Concretely, for both sides (``generation.py:5591-5605`` for the draft,
``generation.py:5715-5741`` for the target):

``values``
    The **top-20 raw logits** of the full row, ordered by (score desc, id asc).
    These are pre-truncation logits, so re-temperaturing offline is exact on
    the retained support -- see ``scripts/fable/offline_draft_temperature.py``
    for the one tail assumption it forces.
``probs``
    ``exp(values - logsumexp(full_row))`` -- the softmax over the *whole*
    vocabulary, restricted to the top-20 ids.  So ``sum(probs)`` is the mass
    the top-20 covers and ``1 - sum(probs)`` is the truncated tail; and
    ``logsumexp(full_row) = values - log(probs)`` is recoverable per row.
``ids``
    Real vocabulary ids.  The draft side has already been mapped through the
    65,536-row frspec rank table (``generation.py:5606-5609``), so draft and
    target ids live in one id space.

Alongside the rows, per cycle:

* ``draft_tokens`` -- the three tokens the draft chain actually selected;
* ``primary`` -- the token the window verifies from;
* ``decision_uniforms`` ``[4]`` and ``draft_uniforms`` ``[3]`` -- the exact
  PCG64 float64 draws the decision and the three draft selects consumed
  (``pcg64_tape.py:116-134``).  Logging both means an offline replay can
  reproduce the emitted stream bit-for-bit, and Option D can re-select the
  drafted token under a re-shaped ``q`` from the same uniform;
* the decision outputs: ``accepted``, ``first_reject``, ``selected_token``,
  ``selected_kind``, ``selected_present``, ``draws_used``, and
  ``accept_probability`` per depth;
* ``bonus_allowed`` and ``descriptor_offset`` (the reservation offset, so the
  rows can be ordered and de-duplicated against the RNG tape).

**alpha beyond the first rejection is not in the log, because the kernel does
not compute it** -- ``reference_pr391_softfloat64_verifier_decision`` returns
as soon as a depth rejects (kernel lines 289-320), leaving the remaining
``accept_probability_bits`` entries at 0.0.  ``accept_probability_valid``
records how many entries are real (``first_reject + 1``, or 3 on a full
accept).  The offline scorers recompute the full per-depth alpha ladder from
the logged rows exactly; that is the whole point of logging the rows.

Cost
----
``MTPLX_FABLE_K20_LOG`` is read exactly once, at import.  When unset,
:data:`_ENABLED` is ``False``, every call site passes ``k20_capture=None``
(a conditional expression, so nothing is even built), and this module costs
one global load plus one predicted-not-taken branch per verify cycle.

When on, the cost is deliberately **one small device-to-host copy per cycle
and no extra synchronisation**.  The six row arrays are already inputs to the
verifier kernel, so they are materialised before the decision outputs exist;
adding them to the decision's own ``mx.eval`` in
``_pr391_decode_float32_verifier_decision`` only retains them, it schedules no
new GPU work and adds no new sync point.  The D2H copy is
``7 x 20 x (uint32 + 2 x float32) = 1,680 B`` per cycle -- about 0.66 MB over a
1,024-token request, three orders of magnitude under the census JSONLs already
in ``.benchmark-artifacts/pr391/``.  Host-side it is six ``np.asarray`` calls
and one list append.

That still perturbs the wall clock a little (the copies are on the critical
path between the decision sync and the commit), so **an instrumented run is a
data run, not a timing run**.  Read tok/s off an un-instrumented arm.

Output
------
``flush()`` -- registered with ``atexit`` when enabled, and safe to call
explicitly and repeatedly -- writes ``numpy.savez_compressed`` with, for
``C`` recorded cycles:

===========================  ==========================  ===================
key                          dtype / shape               meaning
===========================  ==========================  ===================
``draft_ids``                uint32  ``[C, 3, 20]``      draft row support
``draft_values``             float32 ``[C, 3, 20]``      draft top-20 logits
``draft_probs``              float32 ``[C, 3, 20]``      full-vocab softmax
``target_ids``               uint32  ``[C, 4, 20]``      target row support
``target_values``            float32 ``[C, 4, 20]``      target top-20 logits
``target_probs``             float32 ``[C, 4, 20]``      full-vocab softmax
``draft_tokens``             uint32  ``[C, 3]``          selected draft chain
``primary``                  uint32  ``[C]``             window's first token
``decision_uniforms``        float64 ``[C, 4]``          decision draws
``draft_uniforms``           float64 ``[C, 3]``          draft-select draws
``accepted``                 uint32  ``[C]``             accepted depth count
``first_reject``             int32   ``[C]``             -1 when none
``selected_token``           uint32  ``[C]``             correction or bonus
``selected_kind``            uint32  ``[C]``             0/1/2 per kernel
``selected_present``         uint8   ``[C]``
``draws_used``               uint32  ``[C]``
``accept_probability``       float64 ``[C, 3]``          alpha, truncated
``accept_probability_valid`` uint8   ``[C]``             real entries of the above
``bonus_allowed``            uint8   ``[C]``
``descriptor_offset``        int64   ``[C]``             PCG64 tape offset
``stop_ids``                 uint32  ``[S]``             request's stop set
===========================  ==========================  ===================

``stop_ids`` is per *request*, not per cycle: a second request through the
same process overwrites it.  The PR391 D3 route is one request per process
in practice; if that ever changes, an offline replay of the early-stop
branch would be reading the last request's set.

Usage::

    MTPLX_FABLE_K20_LOG=/path/rows.npz <the usual PR391 D3 benchmark command>
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


def _env_path() -> str | None:
    raw = (os.environ.get(_ENV_VAR) or "").strip()
    return raw or None


def _npz_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return path if ext == ".npz" else root + ".npz"


class K20RowLog:
    """Buffer the decision's seven K20 rows per cycle; write once at the end."""

    def __init__(self, path: str | None = None) -> None:
        self.configure(path)

    # -- lifecycle ---------------------------------------------------
    def configure(self, path: str | None) -> None:
        """(Re)point the log.  Also the test seam; never a hot path."""

        self.path = path
        self.enabled = path is not None
        self._rows: list[dict[str, np.ndarray]] = []
        self._stop_ids = np.zeros(0, dtype=np.uint32)
        self._written: str | None = None

    def set_stop_ids(self, stop_ids: Any) -> None:
        """Record the request's stop-token set once, for exact replay."""

        if not _ENABLED or not self.enabled:
            return
        self._stop_ids = np.asarray(stop_ids, dtype=np.uint32).reshape(-1).copy()

    # -- recording ---------------------------------------------------
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
        """Copy one cycle's rows to the host.

        Called from ``_pr391_decode_float32_verifier_decision`` *after* its own
        ``mx.eval``, which the caller has already widened to include these six
        row arrays.  Nothing here evaluates, syncs, or touches the device
        beyond the ``np.asarray`` copies themselves.
        """

        if not _ENABLED or not self.enabled:
            return

        offset = int(reservation.offset)
        tape = np.asarray(uniform_tape.values, dtype=np.float64)
        decision_uniforms = tape[offset : offset + DECISION_DRAWS]
        if decision_uniforms.size != DECISION_DRAWS:
            raise RuntimeError("K20 log: decision reservation is short")
        draft_start = offset - DEPTH
        if draft_start < 0:
            raise RuntimeError("K20 log: draft reservation precedes the tape")
        draft_uniforms = tape[draft_start:offset]

        (
            accepted,
            first_reject,
            selected_token,
            selected_kind,
            selected_present,
            draws_used,
            accept_probs,
        ) = decision
        valid = DEPTH if int(first_reject) < 0 else int(first_reject) + 1

        self._rows.append(
            {
                "draft_tokens": np.asarray(
                    draft_result[0], dtype=np.uint32
                ).reshape(DEPTH),
                "draft_ids": np.asarray(
                    draft_result[1], dtype=np.uint32
                ).reshape(DEPTH, K20),
                "draft_values": np.asarray(
                    draft_result[2], dtype=np.float32
                ).reshape(DEPTH, K20),
                "draft_probs": np.asarray(
                    draft_result[3], dtype=np.float32
                ).reshape(DEPTH, K20),
                "target_ids": np.asarray(
                    target_support[0], dtype=np.uint32
                ).reshape(TARGET_ROWS, K20),
                "target_values": np.asarray(
                    target_support[1], dtype=np.float32
                ).reshape(TARGET_ROWS, K20),
                "target_probs": np.asarray(
                    target_support[2], dtype=np.float32
                ).reshape(TARGET_ROWS, K20),
                "primary": np.uint32(int(primary)),
                "decision_uniforms": decision_uniforms.astype(
                    np.float64, copy=True
                ),
                "draft_uniforms": draft_uniforms.astype(np.float64, copy=True),
                "accepted": np.uint32(int(accepted)),
                "first_reject": np.int32(int(first_reject)),
                "selected_token": np.uint32(int(selected_token)),
                "selected_kind": np.uint32(int(selected_kind)),
                "selected_present": np.uint8(1 if selected_present else 0),
                "draws_used": np.uint32(int(draws_used)),
                "accept_probability": np.asarray(
                    accept_probs, dtype=np.float64
                ).reshape(DEPTH),
                "accept_probability_valid": np.uint8(valid),
                "bonus_allowed": np.uint8(1 if bonus_allowed else 0),
                "descriptor_offset": np.int64(offset),
            }
        )

    # -- output ------------------------------------------------------
    @property
    def cycles(self) -> int:
        return len(self._rows)

    def flush(self) -> str | None:
        """Write the log out.  Safe to call repeatedly; no-op when off."""

        if not _ENABLED or not self.enabled:
            return None
        if not self._rows:
            print(
                f"[fable-k20-log] nothing recorded for {self.path!r}; the "
                "PR391 float32 D3 route never ran a verify decision.",
                file=sys.stderr,
            )
            return None

        arrays = {
            key: np.stack([row[key] for row in self._rows])
            for key in self._rows[0]
        }
        arrays["stop_ids"] = self._stop_ids
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
                        "cycles": int(len(self._rows)),
                        "depth": DEPTH,
                        "target_rows": TARGET_ROWS,
                        "top_k": K20,
                    },
                    handle,
                    indent=2,
                )
        self._written = out
        print(
            f"[fable-k20-log] wrote {out} cycles={len(self._rows)}",
            file=sys.stderr,
        )
        return out


_LOG_PATH = _env_path()
_ENABLED = _LOG_PATH is not None

k20_log = K20RowLog(_LOG_PATH)

if _ENABLED:
    atexit.register(k20_log.flush)


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_K20_LOG`` named a path at import."""

    return _ENABLED


def _configure_for_test(path: str | None) -> None:
    """Re-point the module-level singleton and enable flag (tests only)."""

    global _ENABLED, _LOG_PATH
    _LOG_PATH = path
    _ENABLED = path is not None
    k20_log.configure(path)


__all__ = ["K20RowLog", "k20_log", "is_enabled"]
