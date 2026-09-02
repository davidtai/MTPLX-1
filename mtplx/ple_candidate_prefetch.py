"""Speculative PLE row prefetch over the draft chain's K20 candidates (K-P1).

Report M §A'.3.  Decode reads 16 n-gram rows x 100 B per token out of a 32 GB
memory-mapped table.  The rows are 85-93 % novel, so no LRU covers them, and
the table cannot go resident next to ~85 GB of wired weights -- the ledger's
resident-PLE arm collapsed to ~8 tok/s.  The runtime-only answer is prediction
instead of residency.

The row ids are a PURE FUNCTION of the token ids (``_ngram_rows_np``), and the
draft chain publishes a 20-token candidate support per depth before it samples
from it.  So the moment depth *d*'s K20 row exists on the host, the 16 rows for
each of its 20 candidates are known -- 320 rows x 100 B = 32 KB per depth -- and
a worker thread can read them while the GPU runs depth *d+1*.  When the sampled
token turns out to be one of the 20 (it always is: it was sampled FROM them),
its rows are already in a host buffer and the pages are warm, so the per-cycle
row assembly and the page-fault stall leave the critical path.

**Exactness is structural, not compared.**  The buffer is CONTENT-ADDRESSED by
table row id, and every payload in it is a literal read of the same three
memmaps the shipped gather reads.  A row id names the same table row for the
life of the process, so a hit returns the bytes the shipped gather would have
returned -- including a hit on a row left over from an earlier cycle.  Only
timing changes.  Nothing here decides anything: the sampled token still comes
from the sampler, the row arithmetic still comes from ``_ngram_rows_np``, and
a row this lane does not hold falls back to the shipped gather whole.

Off by default and construction-bound (:data:`ENV_FLAG` resolved once).

What the lane costs, and why the read formulation is not a free choice
---------------------------------------------------------------------
A candidate bucket is 320 rows across three maps.  Read through ``os.pread``
that is 960 syscalls at ~5 us of GIL-CONTENDED Python each (W46) -- ~4.8 ms of
GIL per depth against a 37 ms cycle, two orders above the ~50 us/cycle gather
this lane is trying to move.  Read through the memmap fancy index on WARM
pages it is ~13 ns per row: ~12 us for the whole bucket.  So this lane is
worth having only where the table's pages are already resident, and
:func:`ple_row_gather.warm_decision` is what decides that per bucket rather
than assuming it.  A cold bucket over :data:`PREAD_ROW_BUDGET` rows is
DECLINED, not preaded: the window then resolves as a miss and the shipped
gather runs exactly as it does today.  Under a cold page cache the right lane
is the pre-touch (``MTPLX_FABLE_PLE_FIRST_GATHER_EARLY``), which this one
composes with rather than duplicates.

Where the hook can and cannot fire
----------------------------------
The *speculative* form needs the K20 support on the HOST.  The stock serial
draft loop puts it there once per depth (``_sample_draft_from_logits`` ->
``SparseDistribution.token_ids``).  The retained PR391 joint-D3 core does NOT:
it runs all three depths inside one compiled graph and syncs only the packed
token vector, leaving ``raw_ids_by_depth`` verifier-resident by design.  On
that lane this module still runs -- with the three RESOLVED tokens as
one-candidate depths, which moves the same memmap read off the critical path
into the pool -- but the speculative overlap is only available where the
candidates cross the boundary.  Removing the sync itself (device-side
selection out of a ``[4,20,16,100]`` tensor) is phase 2; see
``docs/perf/ple-candidate-prefetch-phase2.md``.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

__all__ = [
    "ENV_FLAG",
    "CandidateRowPrefetch",
    "candidate_rows",
    "enabled",
    "last_receipt",
    "reset_receipt",
]

ENV_FLAG = "MTPLX_FABLE_PLE_CANDIDATE_PREFETCH"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Qwen4's coding sampler is top_k=20 and the fixed-M4 window is 4 rows wide,
#: so one cycle's worst case is 4 depths x 20 candidates x 16 heads = 1,280
#: table rows (~128 KB).  The buffer is sized for that once and reused, so a
#: cycle allocates nothing.
DEFAULT_WINDOW = 4
DEFAULT_TOP_K = 20

#: mincore draws per map for one decode bucket.  ``ple_row_gather``'s default
#: of 256 is calibrated for a 32,768-row prefill gather; at ~1.9 us a probe it
#: would cost 1.5 ms to decide a ~12 us read here.  Residency has no realistic
#: middle -- a prewarmed table probes 1.00 and a cold one 0.00 -- so a handful
#: of draws separates the two regimes.
PROBE_ROWS = 8

#: Largest bucket the cold path will ``pread`` rather than decline.  One row
#: costs three syscalls at ~5 us of GIL-contended Python each, so 16 rows is
#: ~0.25 ms -- the same order the shipped fixed-M4 primary prefetch already
#: pays every cycle.  A 320-row candidate bucket would be ~4.8 ms, which is
#: two orders above the ~50 us gather this lane is moving.
PREAD_ROW_BUDGET = 16


@lru_cache(maxsize=1)
def enabled() -> bool:
    """Resolve :data:`ENV_FLAG` once.  Unparseable values raise."""

    raw = (os.environ.get(ENV_FLAG) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{ENV_FLAG} must be one of {accepted}, got {os.environ.get(ENV_FLAG)!r}"
    )


_RECEIPT_ZERO: dict[str, float] = {
    # Candidate buckets submitted (one per draft depth, plus the primary).
    "depths": 0,
    # Distinct table rows the lane read into the buffer.
    "candidate_rows": 0,
    # Window resolves served entirely from the buffer / that fell back.
    "hits": 0,
    "misses": 0,
    # Bytes read into the buffer (all three maps).
    "bytes": 0,
    # Owner-thread time blocked joining the worker futures.
    "worker_wait_ms": 0.0,
    # Detail behind hits/misses: window ROWS served vs missing at resolve.
    "rows_served": 0,
    "rows_missing": 0,
    # Which read formulation the worker took, per bucket (see ple_row_gather).
    "vectorized_buckets": 0,
    "pread_buckets": 0,
    # Buckets the worker DECLINED because mincore said the pages were cold and
    # the bucket was too big to pread without costing more than it saves.
    "cold_declines": 0,
}

_RECEIPT: dict[str, float] = dict(_RECEIPT_ZERO)

#: Worker outcome -> receipt key.  Explicit so a new outcome has to be named
#: in the receipt before it can be counted.
_BUCKET_COUNTER = {
    "vectorized": "vectorized_buckets",
    "pread": "pread_buckets",
    "cold_decline": "cold_declines",
    "empty": "cold_declines",
    "overflow": "cold_declines",
}


def last_receipt() -> dict[str, float]:
    """The lane's cumulative engagement receipt for this process."""

    return dict(_RECEIPT)


def reset_receipt() -> None:
    _RECEIPT.update(_RECEIPT_ZERO)


def _bump(name: str, value: float = 1) -> None:
    _RECEIPT[name] = _RECEIPT[name] + value


def candidate_rows(rows, previous, prefix, candidates):
    """The 16 n-gram row ids for each candidate at ONE window position.

    ``rows`` is the bound ``_ngram_rows_np`` partial the aux already owns,
    ``previous`` the two cache-owned tokens ahead of the window, ``prefix``
    the window tokens already fixed (``[]`` for the primary's own position),
    and ``candidates`` the ids competing for the next position.

    The hist handed to ``rows`` is the SAME token sequence, at the SAME
    offsets from index 0, that the full-window call builds -- 2 history tokens
    then the prefix then the position -- so the EOS segment scan
    (``pos_in_seg``, which governs the ``shift`` masks) lands on the same
    values and the returned ids are identical to
    ``window_rows[0, len(prefix), :]`` for whichever candidate is sampled.
    Truncating the history instead would break exactly that: an EOS inside the
    last two tokens makes ``pos_in_seg`` position-dependent.
    """

    import numpy as np

    ids = np.asarray(candidates, dtype=np.int64).reshape(-1)
    width = len(prefix) + 1
    block = np.empty((ids.shape[0], width), dtype=np.int64)
    if prefix:
        block[:, : len(prefix)] = np.asarray(prefix, dtype=np.int64)
    block[:, -1] = ids
    prev = np.broadcast_to(
        np.asarray(previous, dtype=np.int64).reshape(1, -1),
        (ids.shape[0], len(previous)),
    )
    resolved, _history = rows(block, prev)
    return np.ascontiguousarray(resolved[:, -1, :])


class CandidateRowPrefetch:
    """Per-request buffer of candidate PLE rows, filled off the critical path.

    One instance lives on the fixed-M4 sidecar aux for the life of a request.
    ``submit`` is called from the generation thread the instant a depth's
    candidates are known; ``resolve`` is called by the aux when the window's
    real row ids exist.
    """

    __slots__ = (
        "_buckets",
        "_buffers",
        "_closed",
        "_index",
        "_maps",
        "_names",
        "_pending",
        "_pool",
        "_prompt_tail",
        "_row_bytes",
        "_rows",
        "_sidecar",
        "_slots_per_bucket",
        "_specs",
        "_windows",
    )

    def __init__(
        self,
        *,
        rows,
        sidecar,
        prompt_tail,
        window: int = DEFAULT_WINDOW,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        import numpy as np

        pool = sidecar._pool
        if pool is None:
            raise ValueError(
                "PLE candidate prefetch requires the sidecar worker pool "
                "(MTPLX_NGRAM_PREFETCH=0 disables it)"
            )
        self._rows = rows
        self._sidecar = sidecar
        self._prompt_tail = tuple(int(token) for token in prompt_tail)
        self._pool = pool
        self._names = ("weight", "scales", "biases")
        self._maps = {name: sidecar._maps[name][0] for name in self._names}
        # Heads per token (16 in production) is the model's ``ngram_heads``.
        # Deriving it from the aux's OWN row arithmetic rather than a constant
        # keeps the capacity honest if the geometry ever changes.
        probe = candidate_rows(
            rows, self._prompt_tail, (), (int(self._prompt_tail[-1]),)
        )
        heads_per_token = int(probe.shape[1])
        self._windows = int(window)
        # One fixed slice per bucket, sized to that bucket's worst case, so
        # the owner can allocate without waiting to hear how many unique rows
        # the worker found.
        self._slots_per_bucket = int(top_k) * heads_per_token
        capacity = self._windows * self._slots_per_bucket
        self._buffers = {
            name: np.empty(
                (capacity, *self._maps[name].shape[1:]),
                dtype=self._maps[name].dtype,
            )
            for name in self._names
        }
        self._specs = tuple(
            (
                name,
                int(self._maps[name].offset),
                int(np.prod(self._maps[name].shape[1:]))
                * int(self._maps[name].dtype.itemsize),
                self._maps[name].dtype,
                int(np.prod(self._maps[name].shape[1:])),
                tuple(self._maps[name].shape[1:]),
            )
            for name in self._names
        )
        self._row_bytes = sum(spec[2] for spec in self._specs)
        self._index: dict[int, int] = {}
        self._pending: list = []
        self._buckets = 0
        self._closed = False

    # -- owner thread ----------------------------------------------------

    def begin_cycle(self) -> None:
        """Start a fresh cycle: join what is in flight, drop the buffer.

        An in-flight bucket writes into a slice this cycle is about to reuse,
        so it MUST be joined before the reset -- that is what makes the reuse
        safe rather than merely probable.  ``resolve`` normally joined them
        already.
        """

        self._join()
        self._index = {}
        self._buckets = 0
        self._closed = False

    def previous_tokens(self, completion_tokens, committed_count: int):
        """The two cache-owned tokens ahead of the window.

        Mirrors ``qwen4_fixed_verify._fixed_m4_previous_tokens`` on the aux's
        own prompt tail; kept here so a caller that is not the aux (the
        generation loop's draft hook) does not have to reach into it.
        """

        committed = int(committed_count)
        if committed >= 2:
            return (
                int(completion_tokens[committed - 2]),
                int(completion_tokens[committed - 1]),
            )
        if committed == 1:
            return int(self._prompt_tail[1]), int(completion_tokens[0])
        return self._prompt_tail

    def submit(
        self,
        *,
        prefix_tokens,
        candidate_ids,
        completion_tokens,
        committed_count: int,
    ) -> int:
        """Queue one window position's candidate rows.  Returns slots reserved.

        **Everything this does is O(depth) on the owner thread.**  The row
        arithmetic, the dedup and the read all run on the worker: the whole
        point of the lane is to take host work OFF the decode cycle, and
        ``_ngram_rows_np`` over a [20, d+1] block plus a 320-way ``np.unique``
        plus 320 dict probes is ~60-80 us -- the same order as the ~50 us
        gather the lane is trying to hide.  Doing it here would hand back the
        win at the door.

        Each bucket gets a FIXED slice of the buffer (its own worst case), so
        the owner needs no count from the worker to allocate.  Cross-bucket
        duplicate rows are therefore read twice; that is a handful of rows a
        cycle against 60-80 us of owner time, and both copies hold the same
        bytes, so a duplicate can never be wrong -- only redundant.
        """

        if self._closed:
            # A resolve ended the last cycle.  Self-synchronising so a lane
            # whose owner never calls begin_cycle (a fixed-M4 route with no
            # primary prefetch) still gets one buffer per cycle instead of
            # filling up once and declining forever.
            self.begin_cycle()
        bucket = self._buckets
        if bucket >= self._windows:
            # More positions than a window holds: a wiring bug, not a state we
            # can serve.  Decline -- a declined bucket is a miss at resolve,
            # never a wrong row.
            return 0
        self._buckets = bucket + 1
        _bump("depths")
        self._pending.append(
            self._pool.submit(
                self._fill,
                self.previous_tokens(completion_tokens, committed_count),
                tuple(int(token) for token in prefix_tokens),
                candidate_ids,
                bucket * self._slots_per_bucket,
            )
        )
        return self._slots_per_bucket

    def resolve(self, flat):
        """Row matrices for the window's exact ids, or ``None`` on a miss.

        The return is the same ``{name: [N, ...]}`` shape
        ``_SidecarGather._rows_matrices`` produces, so the caller hands it
        straight to ``gather_np(flat, prepared=...)``.  All-or-nothing: a
        window with one uncovered row takes the shipped gather whole, which
        is the path that is already correct.
        """

        import numpy as np

        self._join()
        self._closed = True
        index = self._index
        wanted = np.asarray(flat, dtype=np.int64).reshape(-1)
        slots = [index.get(int(row), -1) for row in wanted.tolist()]
        served = sum(1 for slot in slots if slot >= 0)
        if served != len(slots):
            _bump("misses")
            _bump("rows_served", served)
            _bump("rows_missing", len(slots) - served)
            return None
        _bump("hits")
        _bump("rows_served", served)
        picks = np.asarray(slots, dtype=np.int64)
        return {
            name: np.ascontiguousarray(self._buffers[name][picks])
            for name in self._names
        }

    def _join(self) -> None:
        """Wait for every in-flight bucket, then fold its outcome.

        The counters are folded HERE, on the owner thread, not by the workers:
        ``_RECEIPT[name] = _RECEIPT[name] + value`` is a read-modify-write and
        four concurrent buckets would silently lose updates -- in a receipt
        that decides an A/B verdict.
        """

        pending = self._pending
        if not pending:
            return
        self._pending = []
        started = time.perf_counter()
        outcomes = [future.result() for future in pending]
        _bump("worker_wait_ms", (time.perf_counter() - started) * 1000.0)
        for path, rows_read in outcomes:
            _bump(_BUCKET_COUNTER[path])
            if rows_read:
                _bump("candidate_rows", rows_read)
                _bump("bytes", rows_read * self._row_bytes)

    # -- worker thread ---------------------------------------------------

    def _fill(self, previous, prefix, candidate_ids, start: int):
        """Resolve, dedup and read one bucket's rows.  ALL of it on the pool.

        Returns ``(path, rows_read)`` for the owner to fold into the receipt.
        The buffer slice ``[start, start + slots_per_bucket)`` and the index
        entries this writes belong to this bucket alone; the owner reads them
        only after ``future.result()``, so the handoff needs no lock.  (Two
        buckets can name the same table row; last writer wins, and both slots
        hold the same bytes, so the race is redundant rather than wrong.)

        **Formulation is not a free choice here.**  A candidate bucket is 320
        rows x 3 maps = 960 ``os.pread`` calls, and the syscall count IS the
        cost: ~5 us of GIL-CONTENDED Python each (W46), so a preaded bucket
        would burn ~4.8 ms of GIL against a 37 ms cycle -- two orders above
        the ~50 us/cycle gather this lane is trying to move.  Warm, the same
        rows cost ~13 ns each through the memmap fancy index (~12 us for the
        whole bucket).  So:

        * ``mincore`` says warm -> the fancy index.  This is THE lane.
        * cold, and the bucket is primary-sized -> ``pread``, the same
          formulation and the same ~48 syscalls the shipped fixed-M4 primary
          prefetch already pays every cycle.
        * cold and large -> DECLINE, writing no index entries, so the window
          resolves as a miss and the shipped gather runs unchanged.  Faulting
          320 scattered pages serially with the GIL held (~60 us each) is the
          stall this lane exists to remove, not a fallback for it; under a
          cold page cache the right lane is the pre-touch (W46), which this
          one composes with rather than duplicates.

        The probe is sampled at :data:`PROBE_ROWS`, not ``ple_row_gather``'s
        prefill default of 256: at ~1.9 us a probe, 256 draws x 3 maps would
        cost 1.5 ms to decide a 12 us read.  Residency here has no realistic
        middle (a prewarmed table probes 1.00, a cold one 0.00), so a handful
        of draws separates the two regimes.
        """

        import numpy as np

        from mtplx.ple_row_gather import warm_decision

        ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return "empty", 0
        rows = np.unique(
            candidate_rows(self._rows, previous, prefix, ids).reshape(-1)
        )
        count = int(rows.shape[0])
        if count > self._slots_per_bucket:
            # Cannot happen at the declared geometry (unique <= 20 x heads);
            # refuse rather than write past this bucket's slice.
            return "overflow", 0
        path, _fraction = warm_decision(
            list(self._maps.values()), rows, sample=PROBE_ROWS
        )
        if path == "vectorized":
            stop = start + count
            for name in self._names:
                self._buffers[name][start:stop] = self._maps[name][rows]
            self._publish(rows, start)
            return "vectorized", count
        if count > PREAD_ROW_BUDGET:
            return "cold_decline", 0
        fd = self._sidecar._fd
        for offset, row in enumerate(rows.tolist()):
            for name, base, row_bytes, dtype, item_count, row_shape in self._specs:
                self._buffers[name][start + offset] = np.frombuffer(
                    os.pread(fd, row_bytes, base + int(row) * row_bytes),
                    dtype=dtype,
                    count=item_count,
                ).reshape(row_shape)
        self._publish(rows, start)
        return "pread", count

    def _publish(self, rows, start: int) -> None:
        """Make this bucket's rows findable, AFTER their bytes are in place.

        Publishing only on success is what keeps a declined or failed bucket
        from handing ``resolve`` an uninitialised slice: no entry, no hit.
        """

        index = self._index
        for offset, row in enumerate(rows.tolist()):
            index[row] = start + offset
