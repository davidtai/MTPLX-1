"""MTP QSA indexer-selection reuse across the draft chain -- row K-D2.

``MTPLX_FABLE_INDEXER_REUSE=1``, default off.

The cost this removes
---------------------
Every depth of the 3-step MTP draft chain runs the MTP head's ONE QSA layer,
and that layer's indexer re-derives the sparse block selection from scratch for
a single query row: ``index_qk_proj`` -> q norm -> partial RoPE -> a
``[1, 1, H, nb]`` score GEMM over EVERY pooled block (~4,352 of them at 16K)
-> relu -> head sum -> scale -> validity mask -> tie-break -> ``argpartition``
to top-512 -> blocks-to-tokens -> the K/V gather.  On the order of 40 dependent
dispatches, three times per cycle, on a sync-terminated chain where nothing
overlaps: the depth-2 draft cannot start until the depth-1 token exists, so the
host encode lag and the kernel latency both land on the critical path.

Depths 2 and 3 pay that in full to re-rank a history that grew by ONE token.

What is reused, exactly
-----------------------
Depth 1 computes its selection normally and its block ids are kept.  Depths 2
and 3 skip the query preparation, the score GEMM and the top-k entirely and are
handed::

    S_d  =  S_1  union  {b : nb_1 <= b < nb_d}

where ``nb_d = (pos_start_d + 1) // ratio`` is the number of COMPLETE blocks
visible at depth ``d`` (block ``b`` covers tokens ``[b*ratio, (b+1)*ratio)``).
Nothing else about the layer changes: the raw-key write and the pooled-block
bank update still happen at every depth, so the cache the verifier and the next
cycle read is byte-for-byte what the stock chain leaves behind.

"plus the newest block", at the pooled-block boundary
-----------------------------------------------------
``nb_d - nb_1 = (P+d)//ratio - (P+1)//ratio <= ceil((d-1)/ratio)``, so for a
chain of depth ``D`` at most ``ceil((D-1)/ratio)`` blocks can complete inside
one cycle -- for the shipped ``ratio=4``, ``D=3`` that is **exactly one**.  The
reuse therefore appends ONE slot to the depth-1 id row:

* its id is ``nb_1`` -- the block that was depth 1's incomplete tail and became
  complete when the depth-1 (or depth-2) token landed in it;
* it is live iff ``nb_1 < nb_d``, which is precisely "a block completed";
* when no block completed the slot repeats ``S_1[0]`` and is marked invalid, so
  the emitted width is a constant ``k_eff + 1`` for depths 2..3 and the slot
  costs nothing in either consumer: the dense-mask lane's scatter is idempotent
  and the rows-gather lane scores invalid slots at ``-inf``.

That keeps every property the selection contract needs.  **Causal**: every id
is ``< nb_d``, and blocks below ``nb_d`` are complete and strictly below the
query's own tail.  **Valid**: validity is taken from the depth-``d`` mask
``blk < nb_d``, never carried over from depth 1.  **Superset-consistent**:
``S_1 subset S_2 subset S_3``, so a block the chain has already attended to is
never dropped mid-cycle, and the most recent complete tokens -- the ones a
draft is most sensitive to -- are never silently lost to the moving tail.

``depth - 1 <= ratio`` is required and raises when violated; past that bound a
single extra slot would no longer be exact and the lane would quietly drop a
completed block.

What it does and does not change
--------------------------------
This moves the draft PROPOSAL ``q`` only.  Different attention over a different
(coarser) visible set gives different draft logits.  It does **not** touch the
target ``p``, the verify graph, or the acceptance law, so exact speculative
sampling still holds and the output distribution is unchanged -- a worse ``q``
can only cost acceptance, never correctness.

It is therefore judged on exactly two numbers:

* acceptance, offline, through ``scripts/fable/shadow_draft_harness.py``
  (``--variant indexer-reuse=MTPLX_FABLE_INDEXER_REUSE=1``), which pairs the
  candidate's rows against the SAME logged target rows; and
* cycle time, through ``scripts/fable/abba_window.py``
  (``--candidate-extra-env MTPLX_FABLE_INDEXER_REUSE=1``).  Output digests will
  differ between the arms -- a different ``q`` draws different tokens -- so a
  digest mismatch there is the expected result, not a defect.

Why the gate is read per call
-----------------------------
Unlike ``fable_mtp_kv_only`` (read once at import) this reads ``os.environ``
each time it is asked.  The shadow harness arms a variant by setting its env
around a single draft-chain call, in one process, so that both arms run from an
identical hidden state; a gate cached at import or at runtime construction
would make the candidate bit-identical to stock and the harness would report
``DID NOT ARM``.  The read happens at most once per draft call -- three dict
lookups per decode cycle -- which is nothing beside the ~40 dispatches it
decides.

Arming it
---------
``MTPLX_FABLE_INDEXER_REUSE`` is not a member of
``profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS``; like every other ``MTPLX_FABLE_*``
key it rides the ABBA driver's raw ``--env`` / ``--candidate-extra-env``
passthrough rather than ``family_overrides``.

An armed flag that meets a selection lane it cannot serve RAISES.  It never
reverts to the stock chain, because a silent revert would put a stock number in
a receipt labelled with the flag.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

ENV_FLAG = "MTPLX_FABLE_INDEXER_REUSE"

_TRUTHY = {"1", "true", "yes", "on"}

#: The depth of the draft step currently executing, or ``None`` outside a
#: draft.  A ContextVar rather than a module global so a nested or concurrent
#: runtime cannot leak one request's chain position into another's.
_DRAFT_DEPTH: ContextVar[Optional[int]] = ContextVar(
    "mtplx_fable_draft_depth", default=None
)

_COUNTERS = {"cycles": 0, "steps_reused": 0}


def indexer_reuse_enabled() -> bool:
    """Read ``MTPLX_FABLE_INDEXER_REUSE`` now -- never cached.

    See the module docstring: the shadow-draft harness arms a proposal variant
    by scoping the env around one draft-chain call, so a cached gate would make
    the candidate identical to stock and be reported as ``DID NOT ARM``.
    """

    raw = os.environ.get(ENV_FLAG)
    return bool(raw) and raw.strip().lower() in _TRUTHY


@contextmanager
def draft_depth_scope(depth: Optional[int]) -> Iterator[None]:
    """Mark the body as draft step ``depth`` (1-based); restore on exit.

    ``MTPLXRuntime.draft_mtp`` wraps its ``model.mtp_forward`` call in this.
    ``None`` (and the ``0`` the adapter-ensemble probe passes for its
    unadapted base draft) mean "not a chain step": no anchor is taken and no
    reuse is served, so those calls keep the stock selection exactly.
    """

    token = _DRAFT_DEPTH.set(None if depth is None else int(depth))
    try:
        yield
    finally:
        _DRAFT_DEPTH.reset(token)


def current_draft_depth() -> Optional[int]:
    """The 1-based draft depth in flight, or ``None`` outside a draft step."""

    return _DRAFT_DEPTH.get()


def note_cycle() -> None:
    """One depth-1 selection was anchored for later depths to reuse."""

    _COUNTERS["cycles"] += 1


def note_step_reused() -> None:
    """One depth>=2 selection was served from the anchor instead of scored."""

    _COUNTERS["steps_reused"] += 1


def indexer_reuse_counters() -> dict[str, int]:
    """Snapshot of ``{cycles, steps_reused}`` for a receipt.

    ``steps_reused`` should be ``(depth - 1) * cycles`` on an armed run whose
    every cycle took the reuse lane; a shortfall means some cycles fell back
    (a re-anchor after a cache identity change, or a chain that never reached
    depth 2), and it is the number to look at before believing a ms/window
    delta came from this lane.
    """

    return dict(_COUNTERS)


def reset_indexer_reuse_counters() -> None:
    """Zero the counters -- call between measured runs, not inside one."""

    _COUNTERS["cycles"] = 0
    _COUNTERS["steps_reused"] = 0
