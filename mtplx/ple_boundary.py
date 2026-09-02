"""Host levers for the decode-cycle PLE boundary (W62).  Default off.

What the boundary is, measured
------------------------------
Reduced from ``current-exact-early-d3-census-2410.jsonl`` (383 cycles, the same
capture ``scripts/fable/census_retained_stack.py`` reduces; the pair-classifier
is that module's ``is_ple_boundary`` verbatim).  Per M4 verify window the GPU
idles 6.380 ms, and exactly TWO of those gaps -- one per cycle, in this order,
in 374 of 374 cycles -- are the boundary:

======================================================  =========  =========
gap (previous kernel -> next kernel)                    ms/cycle   host-late
======================================================  =========  =========
``gg1_copyuint32uint32`` -> ``affine_dequantize_gs32``     2.222      84.1 %
``affine_dequantize_gs32`` -> ``gather_front...``          1.780      90.9 %
**total**                                                **4.002**  **87.1 %**
======================================================  =========  =========

Split further, per gap, into (previous GPU end -> host starts encoding) +
(encode) + (encode end -> GPU start)::

    gap A   1890.7 us host   +  22.0 us encode  +  362.8 us submit->start
    gap B   1640.8 us host   +  15.7 us encode  +  166.5 us submit->start

So the boundary is **3.53 ms/cycle of serial HOST time** plus 0.53 ms/cycle of
Metal commit->start latency; encoding itself is 0.04 ms.  It is not a driver
problem and it is not a GPU problem.

Gap A is this module's target: the host is between the draft chain's last
dispatch (the D3 token-id copy) and the ``mx.async_eval`` of the PLE auxiliary,
running the n-gram row arithmetic, the sidecar row read, and the MLX array
construction.  **Gap B is not**: nothing PLE-shaped runs in it -- it is the
compiled verify graph's own host-side construction, and no host lever in this
module touches it (see ``docs/perf/ple-boundary.md``).

The levers
----------
``warm_skip``
    ``_SidecarGather._rows_matrices``'s hot-row branch opens EVERY decode
    gather with ``self._warm(miss_np)``: a blocking, threaded ``os.pread`` pass
    over the missing rows.  At M=4 that is ~48 missing rows x 3 maps = ~144
    ``os.pread`` calls at ~5.03 us of GIL-contended Python each (W46's
    measurement: 32,768 rows x 1 map = 164.8 ms), joined before the read, plus
    ~24 ``pool.submit``/``future.result`` round trips.  On a page-cache-warm
    table the fancy index behind it costs ~13 ns a row, so the warm pass IS the
    gather -- the same inversion ``ple_row_gather`` documents for prefill,
    which the decode branch never got.  This gates it on ``mincore``.

``hot_block`` (built, NOT in the default arm)
    The same branch then assembles its output one row at a time:
    ``_stack_hot_rows`` copies 64 rows x 3 maps = 192 small NumPy assignments.
    ~48 of the 64 rows just came back from the fancy index as one contiguous
    block, so those 144 copies can be three scatters.  Predicted 0.10-0.20
    ms/cycle; ``scripts/fable/micro_ple_boundary.py --self-test`` measured
    -0.003 / -0.017 / +0.032 ms across probe widths 4 / 8 / 32, i.e. noise.
    Kept selectable, kept out of :data:`DEFAULT_ITEMS`.

``primary_vectorized``
    ``qwen4_fixed_verify._bind_fixed_m4_owned_row_prefetch`` preads the 16
    primary rows one row per pool task: 16 ``pool.submit`` + 16
    ``future.result`` on the OWNER thread and 48 ``os.pread`` holding the GIL
    against the draft chain, which Report M A'.1 measures as 77 % latency.
    When the pages are resident there is nothing to prefetch: the read is a
    fancy index, and doing it inline costs less than submitting the tasks.

``timing``
    Not a lever.  Host ``perf_counter`` marks inside the replaced gather (and,
    via :data:`GRAPH_TIMING`, around the compiled verify call) so the ms above
    can be attributed to page faults / Python / sync on the real box.  Excluded
    from the default item set so an A/B measures levers only.

Exactness
---------
Every lever changes WHEN bytes are read, never WHICH bytes.

* ``warm_skip`` removes reads whose results are *discarded*: ``_warm`` preads
  into a throwaway buffer and returns None; the values ``_rows_matrices``
  returns come only from the memmap fancy index in either case.  Skipping it
  cannot change a byte -- this is stronger than a content-addressed cache
  argument, there is no cache.
* ``hot_block`` writes the same values into the same output positions by a
  different assignment order, and leaves the hot LRU's contents, insertion
  order, ``move_to_end`` order and eviction point exactly as the shipped
  branch leaves them.
* ``primary_vectorized`` puts the same row payloads into the hot LRU in the
  same order, as owned copies (``np.array``, not a memmap view -- a view would
  defer the read to use time and could refault).

The row ids consulted (``_ngram_rows_np``) and the candidates applied are
untouched by every item here.  ``tests/test_fable_ple_boundary.py`` asserts the
equivalence rather than asserting the argument.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from . import ple_row_gather as _row_gather

__all__ = [
    "ENV_FLAG",
    "ITEMS_ENV",
    "PROBE_ENV",
    "ITEMS",
    "DEFAULT_ITEMS",
    "GRAPH_TIMING",
    "bind_owned_row_prefetch",
    "bind_sidecar",
    "enabled",
    "graph_timing_enabled",
    "item",
    "items",
    "last_receipt",
    "note_graph_build",
    "probe_rows",
    "reset_receipt",
]

ENV_FLAG = "MTPLX_FABLE_PLE_BOUNDARY"
ITEMS_ENV = "MTPLX_FABLE_PLE_BOUNDARY_ITEMS"
PROBE_ENV = "MTPLX_FABLE_PLE_BOUNDARY_PROBE_ROWS"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Every selectable item.  A name not in here raises at resolution time: an
#: A/B whose candidate silently ran nothing is worse than one that fails.
ITEMS: tuple[str, ...] = (
    "warm_skip",
    "hot_block",
    "primary_vectorized",
    "timing",
)

#: What ``MTPLX_FABLE_PLE_BOUNDARY=1`` alone means.  ``timing`` is an
#: instrument, not a lever, so it is opt-in by name.  ``hot_block`` is a lever
#: that MEASURED AS NOISE (-0.003 to +0.032 ms/cycle over three synthetic
#: micro sweeps, against a 0.3-0.7 % within-seed ABBA floor = 0.11-0.26 ms),
#: so it is out of the default arm too: a candidate should carry the code
#: whose win it is claiming and nothing else.  It stays selectable because the
#: micro's synthetic table is not the production one and the question is
#: cheap to re-ask.
DEFAULT_ITEMS: tuple[str, ...] = ("warm_skip", "primary_vectorized")

#: ``mincore`` draws per map for one decode residency decision.  The prefill
#: default (``ple_row_gather.SAMPLE_ROWS`` = 256) is calibrated for a 32,768-row
#: gather; at ~1.9 us a probe it would cost 1.5 ms to decide a ~0.7 ms read.
#: Residency has no realistic middle -- a prewarmed table probes 1.00 and a
#: cold one 0.00 -- so 8 draws x 3 maps (~46 us) separates the two regimes.
DEFAULT_PROBE_ROWS = 8


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


@lru_cache(maxsize=1)
def items() -> frozenset[str]:
    """The armed item set, resolved once.  Empty when the flag is off."""

    if not enabled():
        return frozenset()
    raw = (os.environ.get(ITEMS_ENV) or "").strip()
    if not raw:
        return frozenset(DEFAULT_ITEMS)
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = tuple(name for name in names if name not in ITEMS)
    if unknown:
        raise ValueError(
            f"{ITEMS_ENV} names unknown item(s) {unknown!r}; "
            f"known items are {ITEMS}"
        )
    return frozenset(names)


def item(name: str) -> bool:
    """Whether one item is armed.  Unknown names raise."""

    if name not in ITEMS:
        raise ValueError(f"unknown {ENV_FLAG} item {name!r}; known items {ITEMS}")
    return name in items()


@lru_cache(maxsize=1)
def probe_rows() -> int:
    """``mincore`` draws per map, from :data:`PROBE_ENV`.  Must be >= 1."""

    raw = (os.environ.get(PROBE_ENV) or "").strip()
    if not raw:
        return DEFAULT_PROBE_ROWS
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{PROBE_ENV} must be an integer, got {raw!r}") from error
    if value < 1:
        raise ValueError(f"{PROBE_ENV} must be >= 1, got {value}")
    return value


@lru_cache(maxsize=1)
def graph_timing_enabled() -> bool:
    """Whether the compiled-verify host timer in ``graphbank`` is armed."""

    return item("timing")


#: Module constant read ONCE at import by ``mtplx.graphbank``: with the flag
#: off this is False and the one site there is a constant-False branch, so the
#: control arm runs the code it ran before this module existed.
try:
    GRAPH_TIMING = graph_timing_enabled()
except ValueError:  # a malformed flag must fail at the lane, not at import
    GRAPH_TIMING = False


_RECEIPT_ZERO: dict[str, float] = {
    # Decode gathers that took the replaced hot-row branch.
    "gathers": 0,
    # Rows asked for / rows that missed the hot LRU, summed over those gathers.
    "rows": 0,
    "misses": 0,
    # warm_skip: residency decisions, and what they decided.
    "probes": 0,
    "warm_skipped": 0,
    "warm_taken": 0,
    # primary_vectorized: 16-row primary prefetches served inline vs pooled.
    "primary_inline": 0,
    "primary_pooled": 0,
    "primary_rows": 0,
    # timing (ms, cumulative over the request).
    "probe_ms": 0.0,
    "warm_ms": 0.0,
    "read_ms": 0.0,
    "assemble_ms": 0.0,
    "gather_ms": 0.0,
    "primary_ms": 0.0,
    "graph_build_ms": 0.0,
    "graph_build_calls": 0,
}

_RECEIPT: dict[str, float] = dict(_RECEIPT_ZERO)


def last_receipt() -> dict[str, float]:
    """The lane's cumulative engagement receipt for this process."""

    receipt = dict(_RECEIPT)
    receipt["items"] = ",".join(sorted(items()))
    receipt["probe_rows"] = probe_rows()
    return receipt


def reset_receipt() -> None:
    _RECEIPT.update(_RECEIPT_ZERO)


def _bump(name: str, value: float = 1) -> None:
    _RECEIPT[name] = _RECEIPT[name] + value


def note_graph_build(seconds: float) -> None:
    """Fold one compiled-verify host construction into the receipt.

    Called from ``graphbank`` only when :data:`GRAPH_TIMING`; this is the
    census's gap-B host term measured from inside the process.
    """

    _bump("graph_build_ms", seconds * 1000.0)
    _bump("graph_build_calls")


# --------------------------------------------------------------------------
# warm_skip + hot_block: the replaced decode gather
# --------------------------------------------------------------------------
def bind_sidecar(sidecar, *, stack_hot_rows=None) -> str | None:
    """Install the replaced hot-row gather on ONE sidecar.  Idempotent.

    Returns the engagement line to log, or ``None`` when nothing was armed.
    The shipped ``_SidecarGather._rows_matrices`` is not edited: the control
    arm runs it unchanged, and this binds an instance attribute that shadows
    it only when an item asks for it.  Big gathers (prefill) always delegate
    back to the shipped method, which owns the vectorised/pread decision there
    already (``MTPLX_FABLE_PLE_FIRST_GATHER_EARLY``).
    """

    armed = tuple(name for name in ("warm_skip", "hot_block", "timing") if item(name))
    if not armed:
        return None
    if getattr(sidecar, "_fable_ple_boundary_bound", False):
        return None
    original = type(sidecar)._rows_matrices
    skip_warm = "warm_skip" in armed
    block = "hot_block" in armed
    timing = "timing" in armed
    # Resolved at BIND time: with `hot_block` disarmed the replacement still
    # has to assemble exactly the way the shipped branch does, and a per-cycle
    # `from ... import` for that would be host time this lane exists to remove.
    if not block and stack_hot_rows is None:
        from mtplx.models.qwen4_exp import _stack_hot_rows

        stack_hot_rows = _stack_hot_rows

    def _rows_matrices(flat, names):
        return _boundary_rows_matrices(
            sidecar,
            original,
            flat,
            names,
            skip_warm=skip_warm,
            block=block,
            timing=timing,
            stack_hot_rows=stack_hot_rows,
        )

    sidecar._rows_matrices = _rows_matrices
    sidecar._fable_ple_boundary_bound = True
    return (
        f"[fable] {ENV_FLAG} decode gather armed: "
        f"items={','.join(armed)} probe_rows={probe_rows()}"
    )


def _boundary_rows_matrices(
    self,
    original,
    flat,
    names,
    *,
    skip_warm: bool,
    block: bool,
    timing: bool,
    stack_hot_rows=None,
):
    """``_SidecarGather._rows_matrices``'s hot-row branch, item by item.

    Structurally the shipped branch with two substitutions.  Everything that
    is not an armed item is the shipped expression, in the shipped order, so
    a partially-armed candidate is still a valid measurement of the item it
    armed.
    """

    import numpy as np

    entered = time.perf_counter() if timing else 0.0
    uniq, inverse = np.unique(flat, return_inverse=True)
    if not (0 < len(uniq) <= self._HOT_PATH_MAX_ROWS and self._hot_cap_rows):
        # Prefill-sized: the shipped method owns this branch whole.
        return original(self, flat, names)

    _bump("gathers")
    _bump("rows", int(len(uniq)))
    hot = self._hot
    uniq_ids = [int(r) for r in uniq.tolist()]
    if block:
        miss = []
        miss_pos = []
        hit_pos = []
        for position, row in enumerate(uniq_ids):
            if row in hot:
                hit_pos.append(position)
            else:
                miss.append(row)
                miss_pos.append(position)
    else:
        miss = [r for r in uniq_ids if r not in hot]
        miss_pos = hit_pos = None
    _bump("misses", len(miss))

    fetched = None
    if miss:
        miss_np = np.asarray(miss, dtype=np.int64)
        if self._pool is not None:
            take_warm = True
            if skip_warm:
                probe_started = time.perf_counter() if timing else 0.0
                path, _fraction = _row_gather.warm_decision(
                    [self._maps[name][0] for name in names],
                    miss_np,
                    sample=probe_rows(),
                )
                if timing:
                    _bump("probe_ms", (time.perf_counter() - probe_started) * 1000.0)
                _bump("probes")
                take_warm = path != "vectorized"
                _bump("warm_taken" if take_warm else "warm_skipped")
            if take_warm:
                warm_started = time.perf_counter() if timing else 0.0
                self._warm(miss_np)
                if timing:
                    _bump("warm_ms", (time.perf_counter() - warm_started) * 1000.0)
        read_started = time.perf_counter() if timing else 0.0
        fetched = {
            name: np.ascontiguousarray(self._maps[name][0][miss_np])
            for name in names
        }
        if timing:
            _bump("read_ms", (time.perf_counter() - read_started) * 1000.0)
        for i, r in enumerate(miss):
            hot[r] = tuple(fetched[name][i] for name in names)
    self.hot_hits += len(uniq) - len(miss)
    self.hot_misses += len(miss)

    assemble_started = time.perf_counter() if timing else 0.0
    move_to_end = hot.move_to_end
    if block:
        out = {}
        miss_index = np.asarray(miss_pos, dtype=np.int64) if miss else None
        for j, name in enumerate(names):
            source = self._maps[name][0]
            buffer = np.empty(
                (len(uniq_ids), *source.shape[1:]), dtype=source.dtype
            )
            if miss:
                buffer[miss_index] = fetched[name]
            for position in hit_pos:
                buffer[position] = hot[uniq_ids[position]][j]
            out[name] = buffer[inverse]
        for key in uniq_ids:
            move_to_end(key)
    else:
        rows = []
        for key in uniq_ids:
            rows.append(hot[key])
            move_to_end(key)
        out = {
            name: stack_hot_rows(rows, j)[inverse]
            for j, name in enumerate(names)
        }
    while len(hot) > self._hot_cap_rows:
        hot.popitem(last=False)
    if timing:
        now = time.perf_counter()
        _bump("assemble_ms", (now - assemble_started) * 1000.0)
        _bump("gather_ms", (now - entered) * 1000.0)
    return out


# --------------------------------------------------------------------------
# primary_vectorized: the fixed-M4 16-row primary prefetch
# --------------------------------------------------------------------------
def bind_owned_row_prefetch(sidecar, *, submit_primary, install, names):
    """Wrap the fixed-M4 primary prefetch pair, or hand back the shipped one.

    ``names`` is the map order the shipped ``fetch`` builds its payload tuple
    in -- ``("weight", "scales", "biases")``.  Passed rather than assumed so
    a geometry change has to be made here too instead of silently reordering
    the payload.

    Returns ``(submit_primary, install, engagement_line_or_None)``.
    """

    if not item("primary_vectorized"):
        return submit_primary, install, None

    import numpy as np

    names = tuple(names)
    maps = [sidecar._maps[name][0] for name in names]
    hot = sidecar._hot
    timing = item("timing")

    def wrapped_submit_primary(rows):
        """Read the 16 primary rows inline when their pages are resident.

        The shipped path is 16 ``pool.submit`` on this thread, 48
        ``os.pread`` holding the GIL against the draft chain, and 16
        ``future.result`` at the aux.  Resident, the same bytes are three
        fancy indexes.  ``np.array`` and not the memmap row itself: the hot
        LRU holds OWNED bytes, and a view would defer the read to use time.
        """

        started = time.perf_counter() if timing else 0.0
        ids = np.asarray(rows, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return submit_primary(rows)
        path, _fraction = _row_gather.warm_decision(maps, ids, sample=probe_rows())
        _bump("probes")
        if path != "vectorized":
            _bump("primary_pooled")
            _bump("warm_taken")
            return submit_primary(rows)
        _bump("warm_skipped")
        unique, inverse = np.unique(ids, return_inverse=True)
        blocks = [np.array(memmap[unique]) for memmap in maps]
        # Insert in the SAME order the shipped install would: one entry per
        # element of `rows`, duplicates included, each overwriting then moved
        # to the end.  Eviction stays with `install`, exactly as shipped.
        for position in inverse.tolist():
            row = int(unique[position])
            hot[row] = tuple(block[position] for block in blocks)
            hot.move_to_end(row)
        _bump("primary_inline")
        _bump("primary_rows", int(ids.size))
        if timing:
            _bump("primary_ms", (time.perf_counter() - started) * 1000.0)
        return ()

    return (
        wrapped_submit_primary,
        install,
        f"[fable] {ENV_FLAG} primary prefetch armed: probe_rows={probe_rows()}",
    )
