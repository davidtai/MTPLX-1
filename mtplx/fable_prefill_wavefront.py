"""Wavefront prefill: the schedule arithmetic, the memory model, the refusals.

STATUS: seam only.  Nothing in the serving path imports this module, so the
stack is byte-identical with or without it.  ``MTPLX_FABLE_PREFILL_WAVEFRONT``
is defined here and read here; no forward consults it yet.  The go/no-go is
``scripts/fable/micro_two_stream_prefill.py`` and the write-up is
``docs/perf/pr391-prefill-wavefront-two-stream.md``.

THE SHAPE
---------
Chunked prefill walks a (chunk, layer) grid one full chunk at a time.  The
grid has exactly two dependency edges::

    (k, L)  ->  (k, L+1)     hidden state
    (k, L)  ->  (k+1, L)     KV + GDN recurrent state (cache entry L)

Both point strictly backwards along the anti-diagonals, so any schedule that
issues anti-diagonals in order is legal.  ``(k, L+1)`` and ``(k+1, L)`` sit on
the same anti-diagonal and share no cache entry: they are the overlappable
pair.

WHY THE WAVEFRONT IS GROUPED, NOT CONTINUOUS
--------------------------------------------
A continuous wavefront over an 8-chunk prompt reaches 8 lanes live at its
widest (``lanes_live(8, 48, lanes=0) == 8``), which multiplies the QSA dense
prefill transient by 8 -- ``mtplx.memory_plan`` prices that at 12.75 B per
(chunk row x context token) per live layer, and the production geometry
already peaks at 87.4 GB against a 90 GiB wired limit.  Eight lanes is not a
memory question, it is an OOM.

So the schedule here is **grouped**: chunks are cut into groups of ``lanes``
(default 2), each group runs its own diagonal, and the pipeline DRAINS
between groups.  Two consequences, both good:

* Peak lanes live is exactly ``lanes``, so the memory model is a clean
  multiplier on the existing ``fable_prefill_chunk`` plan.
* The drain point is a moment when the cache is in a clean end-of-chunk
  state, which is the only place the per-chunk bookkeeping can safely run --
  MTP history append, chunk receipts, and above all the GDN boundary
  snapshot.  See ``assert_boundary_capture_compatible``.

The cost of draining is small: a group of ``lanes`` chunks over ``layers``
layers takes ``layers + lanes - 1`` steps instead of ``layers``, so at
lanes=2, layers=48 the schedule is 2-wide for 47 of its 49 steps.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
------------------------------------------
It does not touch ``mtplx/generation.py`` and it builds no graph.  Wiring the
schedule into the chunk loop needs a new model-level entry point that
interleaves two hidden streams through one layer loop, plus per-lane PLE
staging and per-lane ``_last_widened``; the design document lists those as
open, and none of it should be written before the falsifier reports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .fable_prefill_chunk import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GUARD_MARGIN_BYTES,
    PrefillChunkMemoryError,
    PrefillChunkPlan,
    plan_prefill_chunk_memory,
)

#: Read per call (never memoized) so tests and the harness can flip it.
ENV_FLAG = "MTPLX_FABLE_PREFILL_WAVEFRONT"
#: Chunks in flight per group.  2 is the only width the memory model admits
#: at the production geometry; the knob exists so a smaller prompt or a
#: cheaper attention lane can be priced without editing code.
LANES_ENV = "MTPLX_FABLE_PREFILL_WAVEFRONT_LANES"
DEFAULT_LANES = 2

#: A schedule that leaves the last chunk alone.  The GDN boundary tail grid
#: (``MTPLX_GDN_BOUNDARY_TAIL_INTERVAL``, default 256) refines the final
#: chunk into fine spans specifically so a warm restore can land close to the
#: prompt tail; pairing those spans would halve exactly the resolution that
#: refinement exists to buy.
TAIL_SOLO_ENV = "MTPLX_FABLE_PREFILL_WAVEFRONT_TAIL_SOLO"


class PrefillWavefrontError(RuntimeError):
    """A wavefront geometry the rest of the stack would mis-serve."""


def _env(name: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get(name) or "").strip()


def _env_truthy(
    name: str, default: bool, environ: Mapping[str, str] | None = None
) -> bool:
    raw = _env(name, environ).lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``MTPLX_FABLE_PREFILL_WAVEFRONT``; default OFF."""

    return _env_truthy(ENV_FLAG, False, environ)


def resolve_lanes(environ: Mapping[str, str] | None = None) -> int:
    """Chunks in flight per group.  1 == the shipped serial schedule."""

    raw = _env(LANES_ENV, environ)
    if not raw:
        return DEFAULT_LANES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_LANES


def tail_solo(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the final chunk runs alone.  Default True; see TAIL_SOLO_ENV."""

    return _env_truthy(TAIL_SOLO_ENV, True, environ)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------
def chunk_groups(
    chunks: int, *, lanes: int = DEFAULT_LANES, tail_solo_chunk: bool = False
) -> list[tuple[int, ...]]:
    """Cut ``chunks`` into consecutive groups of at most ``lanes``.

    ``tail_solo_chunk`` peels the last chunk into a group of its own, so the
    GDN boundary tail grid keeps its per-chunk resolution.
    """

    total = max(0, int(chunks))
    width = max(1, int(lanes))
    if not total:
        return []
    body = total - 1 if (tail_solo_chunk and total > 1) else total
    groups = [
        tuple(range(start, min(body, start + width)))
        for start in range(0, body, width)
    ]
    if body < total:
        groups.append((total - 1,))
    return [g for g in groups if g]


def wavefront_steps(
    chunks: int,
    layers: int,
    *,
    lanes: int = DEFAULT_LANES,
    tail_solo_chunk: bool = False,
) -> list[list[tuple[int, int]]]:
    """Issue order for the whole prefill as a list of ``(chunk, layer)`` steps.

    Every step is a set of mutually independent nodes.  Groups are drained
    between each other -- the last step of group ``g`` completes before the
    first step of group ``g+1`` is issued -- which is what bounds lanes live
    and what gives the per-chunk bookkeeping a clean place to run.

    ``lanes=1`` reproduces the shipped serial schedule exactly: one node per
    step, chunk-major then layer-major.
    """

    n_layers = max(0, int(layers))
    if not n_layers:
        return []
    steps: list[list[tuple[int, int]]] = []
    for group in chunk_groups(
        chunks, lanes=lanes, tail_solo_chunk=tail_solo_chunk
    ):
        base = group[0]
        width = len(group)
        for t in range(n_layers + width - 1):
            step = [
                (base + i, t - i)
                for i in range(width)
                if 0 <= t - i < n_layers
            ]
            if step:
                steps.append(step)
    return steps


def lanes_live(
    chunks: int,
    layers: int,
    *,
    lanes: int = DEFAULT_LANES,
    tail_solo_chunk: bool = False,
) -> int:
    """Widest step -- how many layer bodies are simultaneously live.

    This is the number the memory model multiplies by.  ``lanes=0`` means the
    unbounded continuous wavefront, whose width is ``min(chunks, layers)``;
    it is admitted here only so the docs can quote what it would cost.
    """

    if int(lanes) <= 0:
        return min(max(0, int(chunks)), max(0, int(layers)))
    steps = wavefront_steps(
        chunks, layers, lanes=lanes, tail_solo_chunk=tail_solo_chunk
    )
    return max((len(step) for step in steps), default=0)


def overlappable_step_fraction(
    chunks: int,
    layers: int,
    *,
    lanes: int = DEFAULT_LANES,
    tail_solo_chunk: bool = False,
) -> float:
    """Share of steps that carry more than one node.

    The drain steps at each group's head and tail carry one node and cannot
    overlap; this is the ceiling on how much of the schedule the wavefront
    can touch at all, before any question of whether Metal actually overlaps.
    """

    steps = wavefront_steps(
        chunks, layers, lanes=lanes, tail_solo_chunk=tail_solo_chunk
    )
    if not steps:
        return 0.0
    wide = sum(1 for step in steps if len(step) > 1)
    return wide / len(steps)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WavefrontPlan:
    """A chunk plan plus the lanes the wavefront holds live."""

    lanes: int
    serial: PrefillChunkPlan
    wavefront: PrefillChunkPlan

    @property
    def extra_bytes(self) -> int:
        return (
            self.wavefront.projected_peak_bytes
            - self.serial.projected_peak_bytes
        )

    @property
    def fits(self) -> bool:
        return self.wavefront.fits

    def as_receipt(self) -> dict:
        return {
            "lanes": self.lanes,
            "serial": self.serial.as_receipt(),
            "wavefront": self.wavefront.as_receipt(),
            "extra_bytes": self.extra_bytes,
            "fits": self.fits,
        }


def plan_wavefront_memory(
    *,
    chunk_size: int,
    total_tokens: int,
    transient_bytes_per_token: int,
    lanes: int = DEFAULT_LANES,
    resident_bytes: int = 0,
    budget_bytes: int | None = None,
    margin_bytes: int = DEFAULT_GUARD_MARGIN_BYTES,
    query_tile: int = 0,
) -> WavefrontPlan:
    """Price the wavefront against the shipped serial schedule.

    The wavefront's transient is the serial one times ``lanes``: each live
    lane materializes its own attention/indexer chain, and the lanes are live
    at the same moment by construction.  ``plan_prefill_chunk_memory``
    already models the transient as linear in the live query rows, so the
    lane count enters as ``live_lanes``.

    The headline number this produces: at 16,384 prompt tokens, a 2,048-row
    chunk and 2 lanes, the wavefront's transient equals a single 4,096-row
    chunk's -- but WITHOUT the +11.1% attention work term that widening to
    4,096 costs, because each 2,048-row chunk still attends only over its own
    context.  The two levers are therefore not substitutes; they buy
    different things at the same memory price.
    """

    common = dict(
        chunk_size=chunk_size,
        total_tokens=total_tokens,
        transient_bytes_per_token=transient_bytes_per_token,
        resident_bytes=resident_bytes,
        budget_bytes=budget_bytes,
        margin_bytes=margin_bytes,
        query_tile=query_tile,
    )
    return WavefrontPlan(
        lanes=max(1, int(lanes)),
        serial=plan_prefill_chunk_memory(live_lanes=1, **common),
        wavefront=plan_prefill_chunk_memory(
            live_lanes=max(1, int(lanes)), **common
        ),
    )


def guard_wavefront_geometry(
    *,
    chunk_size: int,
    total_tokens: int,
    transient_bytes_per_token: int,
    lanes: int,
    resident_bytes: int = 0,
    budget_bytes: int | None = None,
    margin_bytes: int = DEFAULT_GUARD_MARGIN_BYTES,
    query_tile: int = 0,
) -> WavefrontPlan:
    """Refuse a wavefront whose projected peak overruns the budget.

    Inert (returns the plan, refuses nothing) when no budget is resolvable or
    the family has no transient model, exactly like
    ``fable_prefill_chunk.guard_prefill_chunk_geometry``.
    """

    plan = plan_wavefront_memory(
        chunk_size=chunk_size,
        total_tokens=total_tokens,
        transient_bytes_per_token=transient_bytes_per_token,
        lanes=lanes,
        resident_bytes=resident_bytes,
        budget_bytes=budget_bytes,
        margin_bytes=margin_bytes,
        query_tile=query_tile,
    )
    if budget_bytes is None or transient_bytes_per_token <= 0:
        return plan
    if plan.fits:
        return plan
    headroom = plan.wavefront.headroom_bytes or 0
    raise PrefillChunkMemoryError(
        f"{ENV_FLAG} with {plan.lanes} lanes projects "
        f"{plan.wavefront.projected_peak_bytes / 1024 ** 3:.2f} GiB at "
        f"chunk_size={chunk_size}, total_tokens={total_tokens} -- "
        f"{abs(headroom) / 1024 ** 3:.2f} GiB over the budget. The serial "
        f"schedule projects "
        f"{plan.serial.projected_peak_bytes / 1024 ** 3:.2f} GiB. Lower "
        f"{LANES_ENV}, narrow MTPLX_PREFILL_CHUNK_SIZE, or arm "
        "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE."
    )


# ---------------------------------------------------------------------------
# Correctness refusals
# ---------------------------------------------------------------------------
def assert_boundary_capture_compatible(
    *, capture_boundaries: bool, lanes: int, drains_per_group: bool = True
) -> None:
    """Refuse the combination that would snapshot a TORN cache.

    ``mtplx.generation._capture_gdn_boundary`` snapshots the WHOLE cache at a
    chunk end, and ``mtplx.cache_state.snapshot_untrimmable_cache`` clones
    every recurrent entry.  Inside a group the wavefront has chunk k+1
    already past layers 0..L-1 while chunk k is still finishing layer L, so a
    snapshot taken at "the end of chunk k" would mix chunk k's state for the
    late layers with chunk k+1's for the early ones.  A warm restore from
    that record resumes from a state no forward ever produced.

    The schedule this module emits is safe only because it DRAINS between
    groups: at a drain every cache entry is at the same token count, so a
    snapshot there is a real end-of-chunk state.  Capturing anywhere else is
    the bug this refusal exists to prevent.
    """

    if not capture_boundaries or int(lanes) <= 1:
        return
    if not drains_per_group:
        raise PrefillWavefrontError(
            "GDN boundary capture cannot run under a wavefront that does not "
            "drain: inside a group the cache holds chunk k's state for the "
            "late layers and chunk k+1's for the early ones, and "
            "snapshot_untrimmable_cache would clone that torn mix. Capture "
            "only at group drains, or set "
            f"{LANES_ENV}=1 for this request."
        )


def boundary_records_per_prompt(
    chunks: int, *, lanes: int, tail_solo_chunk: bool = True
) -> int:
    """How many GDN boundary records a wavefront prefill can emit.

    One per group drain instead of one per chunk.  This is a real behaviour
    change to warm-restore resolution and must be stated, not discovered:
    at 8 chunks and 2 lanes with a solo tail the count goes 8 -> 5.
    """

    return len(
        chunk_groups(chunks, lanes=lanes, tail_solo_chunk=tail_solo_chunk)
    )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_LANES",
    "ENV_FLAG",
    "LANES_ENV",
    "TAIL_SOLO_ENV",
    "PrefillWavefrontError",
    "WavefrontPlan",
    "assert_boundary_capture_compatible",
    "boundary_records_per_prompt",
    "chunk_groups",
    "enabled",
    "guard_wavefront_geometry",
    "lanes_live",
    "overlappable_step_fraction",
    "plan_wavefront_memory",
    "resolve_lanes",
    "tail_solo",
    "wavefront_steps",
]
