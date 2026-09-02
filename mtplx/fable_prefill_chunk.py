"""Prefill chunk geometry: a harness-selectable width, with a memory refusal.

Production prefills a 16,384-token prompt in 8 chunks of 2,048
(``MTPLX_PREFILL_CHUNK_SIZE=auto`` -> ``MTPLX_PREFILL_CHUNK_SIZE_DENSE=2048``,
``mtplx/profiles.py`` ``SUSTAINED_PREFILL_ENV``; resolved by
``mtplx.generation._prefill_chunk_size``).  Two facts make the width a real
lever and a real hazard:

* **Lever.** The routed MoE grouped GEMM is the largest prefill family and
  runs at 45 % of the dense-q4 rate.  20,480 (row, expert) assignments over
  512 experts is 40 rows per expert per chunk -- two ``bm=32`` tiles holding
  40 useful rows (62.5 % occupancy).  At 4,096 rows per chunk it is 80 rows
  in three tiles (83.3 %).  Same total FLOPs, better tiles.

* **Hazard.** The dense QSA prefill lane materializes an ``[H, S, T]`` score
  tensor plus its mask/softmax chain.  ``mtplx.memory_plan`` already prices
  that at ``QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM`` (12.75 B) per
  (chunk row x context token) with ``QSA_TRANSIENT_LIVE_LAYERS`` (4) live --
  i.e. **linear in the chunk width**.  Doubling the chunk doubles the peak
  transient.  Nothing in the tree checked that against the wired limit
  before this module: ``mtplx/server/openai.py`` priced the plan with the
  function's *default* 2048 regardless of the width actually resolved.

This module owns three pure, testable pieces:

1. ``plan_prefill_chunk_memory`` -- the projected peak for a geometry.
2. ``guard_prefill_chunk_geometry`` -- the construction-time refusal.
3. ``query_tile_spans`` -- the "middle path" span arithmetic that lets a
   4,096-row chunk keep a 2,048-row attention peak (and, as a bonus, the
   2,048-chunk attention *cost*; see the docstring there).

Every knob is off / inert unless explicitly armed, and the guard is inert
unless a budget is resolvable, so flag-off behaviour is byte-identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

#: Shipped chunk width; the value both ``MTPLX_PREFILL_CHUNK_SIZE_DENSE`` and
#: ``MTPLX_QSA_PREFILL_COMPILE_ROWS`` default to today.
DEFAULT_CHUNK_SIZE = 2048

GUARD_ENV = "MTPLX_FABLE_PREFILL_CHUNK_GUARD"
BUDGET_ENV = "MTPLX_FABLE_PREFILL_CHUNK_BUDGET_BYTES"
WIRED_LIMIT_ENV = "MTPLX_WIRED_LIMIT_BYTES"
RESIDENT_ENV = "MTPLX_FABLE_PREFILL_CHUNK_RESIDENT_BYTES"
MARGIN_ENV = "MTPLX_FABLE_PREFILL_CHUNK_GUARD_MARGIN_BYTES"
COMPILE_ROWS_ENV = "MTPLX_QSA_PREFILL_COMPILE_ROWS"
ALLOW_COMPILE_ROWS_MISMATCH_ENV = (
    "MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH"
)
QUERY_TILE_ENV = "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE"
CHUNK_SIZE_ENV = "MTPLX_PREFILL_CHUNK_SIZE"
CHUNK_SIZE_DENSE_ENV = "MTPLX_PREFILL_CHUNK_SIZE_DENSE"
CHUNK_SIZE_REPAGE_ENV = "MTPLX_PREFILL_CHUNK_SIZE_REPAGE"

#: ``assert_prefill_chunk_coherent`` verdicts.  ``COHERENCE_COMPILED`` means
#: the graph bank serves this width; ``COHERENCE_NARROW_EAGER`` means a
#: partial/tail/warm-up chunk was admitted and takes the eager selector.
COHERENCE_COMPILED = "compiled"
COHERENCE_NARROW_EAGER = "narrow_eager"

#: Receipt counter the caller bumps on a ``COHERENCE_NARROW_EAGER`` verdict.
NARROW_EAGER_COUNTER = "prefill_chunk_narrow_eager_fallbacks"

#: Headroom left unclaimed by the projected peak.  2 GiB is the smallest
#: slack the box has survived: the census peak (87.39 GB) sits 9.2 GB under
#: the driver's 90 GiB wired limit, and a refusal that only fires at the
#: exact limit would admit a geometry that swaps.
DEFAULT_GUARD_MARGIN_BYTES = 2 * 1024**3


class PrefillChunkMemoryError(RuntimeError):
    """A prefill chunk geometry whose projected peak exceeds the budget."""


class PrefillChunkGeometryError(RuntimeError):
    """A prefill chunk geometry the rest of the stack silently mis-serves."""


def _env(name: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get(name) or "").strip()


def _env_int(
    name: str, default: int | None, environ: Mapping[str, str] | None = None
) -> int | None:
    raw = _env(name, environ)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_truthy(
    name: str, default: bool, environ: Mapping[str, str] | None = None
) -> bool:
    raw = _env(name, environ).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Memory projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefillChunkPlan:
    """Projected peak for one (chunk width, prompt length) geometry."""

    chunk_size: int
    total_tokens: int
    chunks: int
    #: Query rows whose attention chain is simultaneously live.  Equal to
    #: ``chunk_size`` unless the QSA query tile is armed.
    live_query_rows: int
    transient_bytes: int
    resident_bytes: int
    projected_peak_bytes: int
    budget_bytes: int | None
    margin_bytes: int
    #: sum over chunks of (rows x context) -- the attention work term, which
    #: is NOT constant in the chunk width unless the query tile is armed.
    attention_row_context_products: int

    @property
    def headroom_bytes(self) -> int | None:
        if self.budget_bytes is None:
            return None
        return self.budget_bytes - self.margin_bytes - self.projected_peak_bytes

    @property
    def fits(self) -> bool:
        headroom = self.headroom_bytes
        return headroom is None or headroom >= 0

    def as_receipt(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "total_tokens": self.total_tokens,
            "chunks": self.chunks,
            "live_query_rows": self.live_query_rows,
            "transient_bytes": self.transient_bytes,
            "resident_bytes": self.resident_bytes,
            "projected_peak_bytes": self.projected_peak_bytes,
            "budget_bytes": self.budget_bytes,
            "margin_bytes": self.margin_bytes,
            "headroom_bytes": self.headroom_bytes,
            "attention_row_context_products": (
                self.attention_row_context_products
            ),
            "fits": self.fits,
        }


def attention_row_context_products(
    total_tokens: int, chunk_size: int, *, query_tile: int = 0
) -> int:
    """``sum over tiles of (rows x keys visible to those rows)``.

    This is the QSA dense-attention work term.  Widening the chunk *raises*
    it, because every row of a chunk attends over the whole chunk's context:
    8 x 2,048 gives 150,994,944 at a 16,384-token prompt, 4 x 4,096 gives
    167,772,160 (+11.1 %).  Splitting a 4,096-row chunk into two 2,048-row
    query tiles brings it back to exactly the 8 x 2,048 value, because tile A
    never reads tile B's keys.
    """

    total = max(0, int(total_tokens))
    chunk = max(1, int(chunk_size))
    tile = max(0, int(query_tile)) or chunk
    products = 0
    start = 0
    while start < total:
        end = min(total, start + chunk)
        row = start
        while row < end:
            row_end = min(end, row + tile)
            products += (row_end - row) * row_end
            row = row_end
        start = end
    return products


def plan_prefill_chunk_memory(
    *,
    chunk_size: int,
    total_tokens: int,
    transient_bytes_per_token: int,
    resident_bytes: int = 0,
    budget_bytes: int | None = None,
    margin_bytes: int = DEFAULT_GUARD_MARGIN_BYTES,
    query_tile: int = 0,
) -> PrefillChunkPlan:
    """Project the peak for one geometry.

    ``transient_bytes_per_token`` is the *per-context-token* dense-lane
    transient at the SHIPPED width, i.e. exactly what
    ``mtplx.memory_plan.qsa_prefill_transient_bytes_per_token_from_config``
    returns for ``chunk_size=DEFAULT_CHUNK_SIZE``.  It is linear in the
    number of query rows simultaneously live, so a wider chunk scales it and
    an armed query tile caps it.
    """

    chunk = max(1, int(chunk_size))
    total = max(0, int(total_tokens))
    tile = max(0, int(query_tile))
    live_rows = min(chunk, tile) if tile else chunk
    per_token = max(0, int(transient_bytes_per_token))
    scaled_per_token = int(
        round(per_token * (live_rows / float(DEFAULT_CHUNK_SIZE)))
    )
    transient = scaled_per_token * total
    resident = max(0, int(resident_bytes))
    chunks = (total + chunk - 1) // chunk if total else 0
    return PrefillChunkPlan(
        chunk_size=chunk,
        total_tokens=total,
        chunks=chunks,
        live_query_rows=live_rows,
        transient_bytes=transient,
        resident_bytes=resident,
        projected_peak_bytes=resident + transient,
        budget_bytes=None if budget_bytes is None else int(budget_bytes),
        margin_bytes=max(0, int(margin_bytes)),
        attention_row_context_products=attention_row_context_products(
            total, chunk, query_tile=tile
        ),
    )


# ---------------------------------------------------------------------------
# Construction-time guard
# ---------------------------------------------------------------------------


def resolve_budget_bytes(environ: Mapping[str, str] | None = None) -> int | None:
    """Explicit budget, else the process's wired limit, else None (inert)."""

    explicit = _env_int(BUDGET_ENV, None, environ)
    if explicit is not None and explicit > 0:
        return explicit
    wired = _env_int(WIRED_LIMIT_ENV, None, environ)
    if wired is not None and wired > 0:
        return wired
    return None


def _default_resident_bytes() -> int | None:
    """MLX's live allocation, if the allocator can be asked for it.

    Reads an allocator counter; it launches no kernel and evaluates no
    array.  Any failure (no MLX, older API) leaves the guard resident-blind,
    in which case it bounds the transient alone.
    """

    try:  # pragma: no cover - exercised only in a process with MLX
        import mlx.core as mx
    except Exception:
        return None
    for name in ("get_active_memory", "get_peak_memory"):
        getter = getattr(mx, name, None)
        if getter is None:
            getter = getattr(getattr(mx, "metal", None), name, None)
        if getter is None:
            continue
        try:
            value = int(getter())
        except Exception:
            continue
        if value > 0:
            return value
    return None


def guard_prefill_chunk_geometry(
    *,
    chunk_size: int,
    total_tokens: int,
    transient_bytes_per_token: int,
    environ: Mapping[str, str] | None = None,
    resident_bytes: int | None = None,
    resident_probe: Callable[[], int | None] | None = None,
) -> PrefillChunkPlan | None:
    """Refuse a geometry whose projected peak overruns the wired limit.

    Returns the plan (for the receipt) or ``None`` when the guard is inert:
    disabled, no budget resolvable, or no transient model for this family.
    Raises :class:`PrefillChunkMemoryError` when it does not fit.
    """

    if not _env_truthy(GUARD_ENV, True, environ):
        return None
    if transient_bytes_per_token <= 0 or total_tokens <= 0:
        return None
    budget = resolve_budget_bytes(environ)
    if budget is None:
        return None
    if resident_bytes is None:
        override = _env_int(RESIDENT_ENV, None, environ)
        if override is not None and override >= 0:
            resident_bytes = override
        else:
            probe = resident_probe or _default_resident_bytes
            resident_bytes = probe() or 0
    margin = _env_int(MARGIN_ENV, DEFAULT_GUARD_MARGIN_BYTES, environ)
    plan = plan_prefill_chunk_memory(
        chunk_size=chunk_size,
        total_tokens=total_tokens,
        transient_bytes_per_token=transient_bytes_per_token,
        resident_bytes=resident_bytes,
        budget_bytes=budget,
        margin_bytes=(
            DEFAULT_GUARD_MARGIN_BYTES if margin is None else max(0, margin)
        ),
        query_tile=resolve_query_tile_rows(environ),
    )
    if plan.fits:
        return plan
    raise PrefillChunkMemoryError(
        "prefill chunk geometry refused: "
        f"chunk_size={plan.chunk_size} over {plan.total_tokens} prompt tokens "
        f"projects {plan.projected_peak_bytes / 1024**3:.2f} GiB "
        f"(resident {plan.resident_bytes / 1024**3:.2f} + transient "
        f"{plan.transient_bytes / 1024**3:.2f}) against a "
        f"{plan.budget_bytes / 1024**3:.2f} GiB budget minus a "
        f"{plan.margin_bytes / 1024**3:.2f} GiB margin. "
        f"Lower {chunk_size_env_name()}, arm {QUERY_TILE_ENV}="
        f"{DEFAULT_CHUNK_SIZE} to cap the live attention rows, raise "
        f"{BUDGET_ENV}, or set {GUARD_ENV}=0 to run it anyway."
    )


def chunk_size_env_name() -> str:
    """The knob an operator sets to change the width.

    ``MTPLX_PREFILL_CHUNK_SIZE`` overrides both layouts when numeric; the
    per-layout ``_DENSE`` / ``_REPAGE`` keys only apply under ``auto``.
    """

    return CHUNK_SIZE_ENV


def configured_full_chunk_widths(
    environ: Mapping[str, str] | None = None,
) -> frozenset[int]:
    """Widths a FULL serving chunk can have, from the environment alone.

    Mirrors ``mtplx.generation._prefill_chunk_size`` minus its request-local
    ContextVar override -- deliberately.  The override is how a caller asks
    for a NARROWER chunk than the serve is configured for (the background
    warm-up ladder passes 256 so a warming prefill releases the model lock
    every ~0.4 s), and honouring it here would make every such chunk look
    like the serve's own full width and be refused.

    ``auto`` resolves per KV layout at request time, so both per-layout keys
    count as full widths; a numeric knob pins exactly one.
    """

    raw = _env(CHUNK_SIZE_ENV, environ).lower() or str(DEFAULT_CHUNK_SIZE)
    if raw == "auto":
        dense = _env_int(CHUNK_SIZE_DENSE_ENV, DEFAULT_CHUNK_SIZE, environ)
        repage = _env_int(CHUNK_SIZE_REPAGE_ENV, DEFAULT_CHUNK_SIZE, environ)
        return frozenset(
            max(1, int(value))
            for value in (dense, repage)
            if value is not None
        )
    try:
        return frozenset({max(1, int(raw))})
    except ValueError:
        return frozenset({DEFAULT_CHUNK_SIZE})


def assert_prefill_chunk_coherent(
    chunk_size: int, environ: Mapping[str, str] | None = None
) -> str:
    """Refuse a FULL width the QSA prefill graph bank would stop serving.

    ``mtplx/models/qwen4_exp.py`` gates the fused/compiled QSA prefill
    indexer on ``rows == _qsa_prefill_compile_rows()`` so that arbitrary
    restored-suffix widths cannot grow the ``mx.compile`` bank without
    bound.  Both that gate and the chunk width default to 2,048, so moving
    one and not the other demotes every full chunk to the eager selector --
    a quiet regression that would be scored as "4,096 lost".

    The gate is ``==``, so a chunk NARROWER than the compiled width already
    falls back by design: partial tails, GDN-boundary sub-spans, and the
    server's background warm-up ladder (256-token chunks, ``openai.py``
    ``_BackgroundWarmup.WARMUP_PREFILL_CHUNK_TOKENS``) are all legitimate
    eager-selector work, not a mis-paired serve.  Refusing them is what
    made both ladder rungs fail instantly under
    ``MTPLX_QSA_PREFILL_COMPILE_ROWS=4096``.

    The rule, given a width ``W`` and ``MTPLX_QSA_PREFILL_COMPILE_ROWS=C``:

    * ``W == C`` -> accept, ``COHERENCE_COMPILED``: the bank serves it.
    * ``W > C`` -> refuse: only a full chunk can exceed the compiled width,
      and the bank cannot serve it.
    * ``W < C`` and ``W`` is a configured FULL chunk width
      (``configured_full_chunk_widths``) -> refuse: this is the W27 case,
      an operator who moved ``C`` without moving the serving width.
    * ``W < C`` otherwise -> accept, ``COHERENCE_NARROW_EAGER``: a
      partial/tail/warm-up chunk on the eager selector.  Callers bump
      ``NARROW_EAGER_COUNTER`` so the fallback is on the receipt.

    ``MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH=1`` waives every
    refusal.
    """

    width = int(chunk_size)
    raw_rows = _env_int(COMPILE_ROWS_ENV, DEFAULT_CHUNK_SIZE, environ)
    compile_rows = (
        DEFAULT_CHUNK_SIZE if raw_rows is None else max(2, int(raw_rows))
    )
    if compile_rows == width:
        return COHERENCE_COMPILED
    narrow = width < compile_rows
    if narrow and width not in configured_full_chunk_widths(environ):
        return COHERENCE_NARROW_EAGER
    if _env_truthy(ALLOW_COMPILE_ROWS_MISMATCH_ENV, False, environ):
        return COHERENCE_COMPILED if not narrow else COHERENCE_NARROW_EAGER
    raise PrefillChunkGeometryError(
        f"prefill chunk width {width} does not match "
        f"{COMPILE_ROWS_ENV}={compile_rows}: the QSA prefill graph bank only "
        "captures its own row width, so every full chunk would fall back to "
        f"the eager selector. Set {COMPILE_ROWS_ENV}={width} "
        f"alongside the width, or {ALLOW_COMPILE_ROWS_MISMATCH_ENV}=1. "
        "(A chunk narrower than the compiled width is a partial/warm-up "
        "chunk and is admitted; this one is a full serving width.)"
    )


# ---------------------------------------------------------------------------
# The middle path: wide chunks, narrow attention
# ---------------------------------------------------------------------------


def resolve_query_tile_rows(environ: Mapping[str, str] | None = None) -> int:
    """Rows per QSA attention query tile; 0 (default) = whole chunk."""

    rows = _env_int(QUERY_TILE_ENV, 0, environ)
    if rows is None or rows <= 0:
        return 0
    return int(rows)


def query_tile_spans(
    rows: int, *, context_before: int, tile: int
) -> list[tuple[int, int, int]]:
    """``(row_start, row_end, keys_visible)`` for one chunk's query tiles.

    Attention rows are independent -- each row's softmax runs over its own
    causal/selected key set -- so grouping rows differently cannot change
    which keys a row sees.  ``keys_visible`` is the exclusive key bound for
    the tile's LAST row, and every earlier row in the tile is masked down to
    its own bound exactly as before.  Dropping the keys past that bound is
    mathematically a no-op: the dense path fills them with the score dtype's
    ``finfo.min``, whose ``exp`` underflows to a hard zero.

    The reduction *order* changes (shorter softmax rows, shorter P@V K), so
    the result is exact-visible-set, not bit-identical -- the same class as
    the portable gather tier.

    Empty list when tiling does not apply, so callers keep one code path.
    """

    span_rows = int(rows)
    tile_rows = int(tile)
    if span_rows <= 0 or tile_rows <= 0 or tile_rows >= span_rows:
        return []
    base = max(0, int(context_before))
    spans: list[tuple[int, int, int]] = []
    row = 0
    while row < span_rows:
        row_end = min(span_rows, row + tile_rows)
        spans.append((row, row_end, base + row_end))
        row = row_end
    return spans


def summarize_spans(spans: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """``(chunk_count, widest_span)`` for an already-cut span list."""

    if not spans:
        return 0, 0
    return len(spans), max(int(end) - int(start) for start, end in spans)
