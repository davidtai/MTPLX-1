"""Expert-major (super-chunk) prefill: the MoE schedule, off by default.

Pure Python: no MLX import, no array, nothing on the GPU.  This module owns
the *decision* -- whether the lane is armed, how many chunks a group holds,
what that costs in bytes, and why a geometry is refused -- so all of it is
unit-testable without the model.  The schedule itself lives in
``Qwen4ExpTextModel.forward_prefill_group``; the loop that calls it lives in
``mtplx.generation``.

===========================================================================
The problem
===========================================================================
MTPLX prefills **chunk-major**: each 2,048-token chunk goes through all 48
layers before the next chunk starts.  The prefill census
(``scratchpad/J-prefill-attribution.md`` §2.2) finds the routed MoE grouped
GEMM is the single largest family -- **3,391.6 ms, 31.6 % of GPU busy,
78.72 TFLOP at 23.2 TFLOP/s, i.e. 45 % of the rate the same q4/g32 kernel
reaches dense** (51.4 TF/s).

That gap is a schedule artifact.  2,048 tokens x top-10 = 20,480 (row,
expert) assignments over 512 experts is **40 rows per expert**, and MLX's
sorted grouped kernel tiles rows at ``BM = 32`` and walks *runs of equal
expert id inside each tile*
(``mlx/backend/metal/kernels/quantized_nax.h:1475``).  Every tile that
straddles an expert boundary streams a second weight tile and runs the whole
K loop again.  Measured exactly (``scripts/fable/micro_moe_prefill_rows.py``
tile model, uniform-router routing):

======  ============  =======  =======  ==================
R       chunk rows    tiles    runs     runs/tile
======  ============  =======  =======  ==================
40      2,048         640      1,129    **1.76**
80      4,096         1,280    1,773    1.39
160     8,192         2,560    3,051    1.19
320     16,384        5,120    5,615    **1.10**
======  ============  =======  =======  ==================

**1.76 -> 1.10 is a 1.61x cut in issued weight streams and K loops** for the
same useful FLOPs, and it also takes the MoE bank from being re-streamed once
per chunk to once per group (census: 75.5 GB/chunk; 302 GB -> 75 GB over a
16K prompt at chunk 4096).

===========================================================================
Why a schedule change is needed at all, and what it looks like
===========================================================================
Rows/expert is ``chunk_tokens x top_k / num_experts``, so the only way to
raise it is more rows in one GEMM.  Simply widening the chunk does that, but
it also widens *attention*: the dense QSA lane's work term
``sum over chunks of (rows x context)`` **rises 11 %** from 8x2,048 to
4x4,096 (``mtplx/fable_prefill_chunk.attention_row_context_products``), and
its transient is linear in the live query rows.  The MoE, unlike attention
and GDN, has **no token-token dependency** -- every row is an independent
(router -> 10 experts -> weighted sum).  So the rows it sees can be widened
without widening anything else, if the *schedule* changes rather than the
chunk.

The schedule, for a group of ``G`` consecutive chunks::

    for L in layers:                      # LAYER-major over the group
        for k in 0..G-1:                  # chunk order preserved inside L
            h[k] = layer.prefill_attn_half(h[k], ids[k], cache[L])
        for k in 0..G-1:
            mixed[k], hyper[k], inject[k] = layer.mlp_hyper_connection(h[k])
            inds[k], scores[k] = layer.mlp.prefill_route(mixed[k])
        y = layer.mlp.switch_mlp(concat(mixed), concat(inds))   # ONE GEMM
        for k in 0..G-1:                  # split back, per-chunk tail
            h[k] = write(hyper[k], mlp.prefill_combine(y[k], scores[k],
                                                       mixed[k]), inject[k])

**The dependency the schedule has to respect.** Layer L's MoE output feeds
layer L+1's attention *for the same chunk*, so nothing may cross a layer
until every chunk of the group has finished that layer -- which is exactly
what layer-major does.  The other direction is the sequence dependency:
chunk k's KV/GDN state at layer L+1 must see chunks < k at layer L+1 and
nothing else.  Processing the group **layer by layer, in chunk order inside
each layer** satisfies both:

* at layer L, chunk k is appended to ``cache[L]`` after chunks 0..k-1 and
  before chunks k+1.., so ``cache[L]``'s offset when chunk k runs is
  ``group_start + k*chunk`` -- identical to chunk-major;
* the GDN recurrent state at layer L is likewise advanced in chunk order;
* the PLE n-gram history (``cache[i][NGRAM_IDX]``) is staged and consumed at
  the PLE layer in chunk order, so ``_stage_body``'s ``prev`` is the same
  array chunk-major would have handed it.

===========================================================================
Exactness
===========================================================================
The claim is **bit-exact, not merely close**.  It rests on one invariant:

    **Only ``mx.gather_qmm`` sees a different M.**

Everything else in the schedule -- the router ``gate`` matmul, the softmax,
the argpartition, the top-k gather, the weighted sum, the shared expert's
``quantized_matmul``, the hyper-connection reads and residual writes, all of
attention and GDN -- runs on the same per-chunk tensors it runs on today,
with the same shapes, in the same order.  Concatenation before the routed
GEMM and the split after it are copies.

For the routed GEMM itself, more rows changes only tiling:

* the kernel accumulates each output element over K in a fixed ``BK``-sized
  loop into a per-simdgroup ``NAXTile`` cleared once per (tile, expert run);
  the row count enters only through the y grid and ``tgp_bm``;
* the shipped ``mlx.metallib`` (mlx 0.32.2) offers exactly two variants of
  ``affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_32_b_4``:
  ``bm_32_bn_64_bk_64_wm_2_wn_2`` and ``bm_64_bn_64_bk_64_wm_2_wn_2``.
  **Both are bk_64**, so even if the dispatcher picks a different BM at a
  larger M the K reduction order does not move;
* ``_gather_sort``'s permutation differs (it sorts a longer index array), but
  each output row is an independent dot product and ``_scatter_unsort``
  restores the original order.

That is an argument, so the tree measures it in two places instead of
trusting it: ``scripts/fable/micro_moe_prefill_rows.py --exactness`` (same
rows alone vs as the head of a 8x larger batch, bitwise) and
``tests/test_fable_prefill_expert_major.py`` (toy model, CPU stream,
group forward vs chunk-major forward, every layer, plus the caches).

===========================================================================
Memory
===========================================================================
Two terms grow with the group, both per-layer and both freed at the layer
boundary; nothing accumulates across layers.

``live hidden``
    ``G`` chunks' widened streams are alive between layers instead of one:
    ``group_rows x hc_count x hidden x 2`` bytes
    (= ``group_rows x 20,480`` at the served geometry; 335 MB at 16K rows).

``routed transient``
    the fused chain at ``top_k x group_rows`` expert-rows::

        sorted x     [10R, 2560] bf16   R x  51,200 B
        gate+up out  [10R, 1280] bf16   R x  25,600 B
        silu*up      [10R,  640] bf16   R x  12,800 B
        down out     [10R, 2560] bf16   R x  51,200 B
        unsorted     [10R, 2560] bf16   R x  51,200 B
                                        -------------
                                        R x 192,000 B   (3.15 GB at R=16K)

    ``ROUTED_TRANSIENT_BYTES_PER_ROW`` below is that 192,000, and it is an
    upper bound: it charges every stage as if all five were live at once.

The *delta* against chunk-major is what the guard must clear -- chunk-major
already pays one chunk's worth of both::

    delta = (G - 1) x chunk_rows x (192,000 + hc_count x hidden x 2)

At the served geometry that is **(G-1) x chunk_rows x 212,480 B**:

======  ============  ==================  ============================
chunk   G             group rows          delta over chunk-major
======  ============  ==================  ============================
4,096   2             8,192               **+0.87 GB**
4,096   4             16,384              **+2.61 GB**
4,096   8             32,768              +6.09 GB
2,048   8             16,384              +3.05 GB
======  ============  ==================  ============================

against a measured chunk-4096 peak of **92.22 GB** (W32).  ``plan_group``
takes a budget, subtracts a margin, and returns the largest G that fits --
construction-time, before a single kernel is submitted, the same contract as
``mtplx.fable_prefill_chunk``.

===========================================================================
What does NOT compose, and is refused rather than degraded
===========================================================================
``gdn_boundary_sink``
    ``mtplx.generation._capture_gdn_boundary`` snapshots the recurrent state
    at every chunk end for the session bank.  Layer-major never has all 48
    layers' state at an interior chunk end -- at layer L, chunk k's state is
    overwritten by chunk k+1 before layer L+1 starts.  The group end is
    therefore the boundary granularity.  ``BOUNDARY_POLICY_ENV`` chooses:
    ``refuse`` (default -- a boundary-capturing prefill runs chunk-major and
    the lane counts ``refused_boundary_capture``) or ``group`` (capture at
    group ends only, counting ``boundaries_coarsened``).  There is no silent
    third option.
``vision splice``
    per-chunk spliced embeddings are supported by the seam but untested;
    refused.
``compiled-verify PLE`` (``_COMPILED_VERIFY_PLE``)
    a decode/verify scope, never live during prefill; refused if seen.

Everything else is preserved: per-chunk prefill receipts
(``_record_prefill_chunk``) are still one record per chunk, with the group's
wall attributed to the chunk that started it and the rest recorded at 0 plus
a ``group_span`` field; the committed-MTP history append still runs per
chunk, in chunk order, after the group's trunk pass (it consumes only the
returned hidden and the token ids, so its result is unchanged); and the PLE
prefill lookahead still prepares one chunk span ahead, because the group
stages its chunks one at a time at the PLE layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "COUNTERS",
    "ENV_FLAG",
    "GROUP_ENV",
    "BUDGET_ENV",
    "MARGIN_ENV",
    "BOUNDARY_POLICY_ENV",
    "ExpertMajorPlan",
    "ExpertMajorRefusal",
    "boundary_policy",
    "count",
    "enabled",
    "group_bytes",
    "group_spans",
    "max_group_for_budget",
    "plan_group",
    "requested_group",
    "reset_counters",
    "rows_per_expert",
    "snapshot_counters",
]

ENV_FLAG = "MTPLX_FABLE_PREFILL_EXPERT_MAJOR"
GROUP_ENV = "MTPLX_FABLE_PREFILL_EXPERT_MAJOR_GROUP"
BUDGET_ENV = "MTPLX_FABLE_PREFILL_EXPERT_MAJOR_BUDGET_BYTES"
MARGIN_ENV = "MTPLX_FABLE_PREFILL_EXPERT_MAJOR_MARGIN_BYTES"
BOUNDARY_POLICY_ENV = "MTPLX_FABLE_PREFILL_EXPERT_MAJOR_BOUNDARIES"

_TRUE = frozenset({"1", "true", "yes", "on"})

#: Upper-bound bytes of routed-MoE transient per CHUNK ROW at the served
#: geometry (top_k 10, hidden 2560, moe_intermediate 640, bf16).  See the
#: module docstring's table; ``routed_transient_bytes_per_row`` recomputes it
#: for any geometry.
ROUTED_TRANSIENT_BYTES_PER_ROW = 192_000

#: Bytes of widened hidden per chunk row at the served geometry
#: (hc_count 4 x hidden 2560 x bf16).
HIDDEN_BYTES_PER_ROW = 20_480

#: Headroom left unclaimed, mirroring ``fable_prefill_chunk``: the smallest
#: slack the box has survived.
DEFAULT_MARGIN_BYTES = 2 * 1024**3

#: Default chunks per group when the flag is armed without a size.  4 chunks
#: at the 4,096 width is B1's 16K-row super-chunk (320 rows/expert).
DEFAULT_GROUP = 4

BOUNDARY_POLICIES = ("refuse", "group")

#: Engagement receipts.  A lane with no counter is a lane whose benchmark
#: cannot be read; every refusal and every group bumps exactly one.
COUNTERS: dict[str, int] = {}


class ExpertMajorRefusal(RuntimeError):
    """An armed expert-major geometry that this request cannot serve."""


def count(name: str, delta: int = 1) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + int(delta)


def reset_counters() -> None:
    COUNTERS.clear()


def snapshot_counters() -> dict[str, int]:
    return dict(COUNTERS)


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


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Off unless explicitly armed.  Read once, at construction."""

    return _env(ENV_FLAG, environ).lower() in _TRUE


def requested_group(environ: Mapping[str, str] | None = None) -> int:
    value = _env_int(GROUP_ENV, DEFAULT_GROUP, environ) or DEFAULT_GROUP
    return max(1, int(value))


def boundary_policy(environ: Mapping[str, str] | None = None) -> str:
    raw = _env(BOUNDARY_POLICY_ENV, environ).lower() or BOUNDARY_POLICIES[0]
    if raw not in BOUNDARY_POLICIES:
        raise ExpertMajorRefusal(
            f"{BOUNDARY_POLICY_ENV}={raw!r} is not one of {BOUNDARY_POLICIES}"
        )
    return raw


# ---------------------------------------------------------------------------
# Geometry and memory
# ---------------------------------------------------------------------------


def rows_per_expert(chunk_rows: int, group: int, top_k: int, num_experts: int) -> float:
    """Mean rows one expert sees in one grouped GEMM.

    ``chunk x G x top_k / num_experts``: 40 at the shipped 2,048 chunk, 80 at
    4,096, 320 at B1's 4x4,096 super-chunk.
    """

    if num_experts <= 0:
        return 0.0
    return max(0, int(chunk_rows)) * max(1, int(group)) * int(top_k) / int(num_experts)


def routed_transient_bytes_per_row(
    *, hidden: int, moe_intermediate: int, top_k: int, bytes_per_elem: int = 2
) -> int:
    """Upper-bound routed-MoE chain bytes per CHUNK ROW.

    Five stages of the ``_FusedGateUpSwitchGLU`` chain, each ``top_k`` rows
    per token: sorted x [hidden], fused gate+up [2*mi], silu*up [mi], down
    [hidden], unsorted [hidden].  Charged as if all five are live at once,
    which is what an unsynchronised lazy chunk graph can do.
    """

    per_expert_row = bytes_per_elem * (3 * int(hidden) + 3 * int(moe_intermediate))
    return int(top_k) * per_expert_row


def hidden_bytes_per_row(*, hidden: int, hc_count: int, bytes_per_elem: int = 2) -> int:
    """Widened residual stream bytes for one chunk row."""

    return int(hidden) * int(hc_count) * int(bytes_per_elem)


def group_bytes(
    *,
    chunk_rows: int,
    group: int,
    routed_per_row: int = ROUTED_TRANSIENT_BYTES_PER_ROW,
    hidden_per_row: int = HIDDEN_BYTES_PER_ROW,
) -> dict[str, int]:
    """Bytes the group holds, and the delta over chunk-major.

    ``delta_bytes`` is the number a budget has to clear: chunk-major already
    pays one chunk's routed transient and one chunk's hidden.
    """

    chunk = max(0, int(chunk_rows))
    g = max(1, int(group))
    rows = chunk * g
    routed = rows * int(routed_per_row)
    hidden = rows * int(hidden_per_row)
    base = chunk * (int(routed_per_row) + int(hidden_per_row))
    return {
        "group_rows": rows,
        "routed_transient_bytes": routed,
        "live_hidden_bytes": hidden,
        "total_bytes": routed + hidden,
        "chunk_major_bytes": base,
        "delta_bytes": routed + hidden - base,
    }


def max_group_for_budget(
    *,
    chunk_rows: int,
    headroom_bytes: int,
    routed_per_row: int = ROUTED_TRANSIENT_BYTES_PER_ROW,
    hidden_per_row: int = HIDDEN_BYTES_PER_ROW,
    cap: int = 64,
) -> int:
    """Largest G whose delta fits ``headroom_bytes`` (>= 1 always)."""

    chunk = max(1, int(chunk_rows))
    per_extra_chunk = chunk * (int(routed_per_row) + int(hidden_per_row))
    if per_extra_chunk <= 0:
        return max(1, int(cap))
    extra = max(0, int(headroom_bytes)) // per_extra_chunk
    return max(1, min(int(cap), 1 + int(extra)))


@dataclass(frozen=True)
class ExpertMajorPlan:
    """One request's expert-major geometry, decided before any kernel runs."""

    group: int
    requested_group: int
    chunk_rows: int
    group_rows: int
    rows_per_expert: float
    chunk_major_rows_per_expert: float
    delta_bytes: int
    routed_transient_bytes: int
    live_hidden_bytes: int
    budget_bytes: int | None
    margin_bytes: int
    resident_bytes: int
    boundary_policy: str

    @property
    def headroom_bytes(self) -> int | None:
        if self.budget_bytes is None:
            return None
        return (
            self.budget_bytes
            - self.margin_bytes
            - self.resident_bytes
            - self.delta_bytes
        )

    @property
    def engaged(self) -> bool:
        return self.group > 1

    def as_receipt(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "requested_group": self.requested_group,
            "chunk_rows": self.chunk_rows,
            "group_rows": self.group_rows,
            "rows_per_expert": self.rows_per_expert,
            "chunk_major_rows_per_expert": self.chunk_major_rows_per_expert,
            "delta_bytes": self.delta_bytes,
            "routed_transient_bytes": self.routed_transient_bytes,
            "live_hidden_bytes": self.live_hidden_bytes,
            "budget_bytes": self.budget_bytes,
            "margin_bytes": self.margin_bytes,
            "resident_bytes": self.resident_bytes,
            "headroom_bytes": self.headroom_bytes,
            "boundary_policy": self.boundary_policy,
            "engaged": self.engaged,
        }


def plan_group(
    *,
    chunk_rows: int,
    top_k: int,
    num_experts: int,
    hidden: int,
    moe_intermediate: int,
    hc_count: int,
    group: int | None = None,
    budget_bytes: int | None = None,
    resident_bytes: int = 0,
    margin_bytes: int | None = None,
    policy: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ExpertMajorPlan:
    """Resolve the group size for one request.

    The requested group is capped by the budget rather than refused: a
    request that cannot afford 4 chunks can usually afford 2, and 2 already
    doubles rows/expert.  ``group == 1`` is the chunk-major path -- the
    caller then runs the shipped loop, so an over-budget request degrades to
    today's behaviour instead of failing.
    """

    wanted = requested_group(environ) if group is None else max(1, int(group))
    margin = (
        DEFAULT_MARGIN_BYTES
        if margin_bytes is None
        else max(0, int(margin_bytes))
    )
    if margin_bytes is None:
        override = _env_int(MARGIN_ENV, None, environ)
        if override is not None and override >= 0:
            margin = int(override)
    if budget_bytes is None:
        budget_bytes = _env_int(BUDGET_ENV, None, environ)
        if budget_bytes is None:
            budget_bytes = _env_int("MTPLX_WIRED_LIMIT_BYTES", None, environ)

    routed_per_row = routed_transient_bytes_per_row(
        hidden=hidden, moe_intermediate=moe_intermediate, top_k=top_k
    )
    hidden_per_row = hidden_bytes_per_row(hidden=hidden, hc_count=hc_count)

    resolved = wanted
    if budget_bytes is not None:
        headroom = int(budget_bytes) - margin - max(0, int(resident_bytes))
        resolved = min(
            wanted,
            max_group_for_budget(
                chunk_rows=chunk_rows,
                headroom_bytes=headroom,
                routed_per_row=routed_per_row,
                hidden_per_row=hidden_per_row,
            ),
        )
    sizes = group_bytes(
        chunk_rows=chunk_rows,
        group=resolved,
        routed_per_row=routed_per_row,
        hidden_per_row=hidden_per_row,
    )
    return ExpertMajorPlan(
        group=resolved,
        requested_group=wanted,
        chunk_rows=int(chunk_rows),
        group_rows=sizes["group_rows"],
        rows_per_expert=rows_per_expert(chunk_rows, resolved, top_k, num_experts),
        chunk_major_rows_per_expert=rows_per_expert(
            chunk_rows, 1, top_k, num_experts
        ),
        delta_bytes=sizes["delta_bytes"],
        routed_transient_bytes=sizes["routed_transient_bytes"],
        live_hidden_bytes=sizes["live_hidden_bytes"],
        budget_bytes=None if budget_bytes is None else int(budget_bytes),
        margin_bytes=margin,
        resident_bytes=max(0, int(resident_bytes)),
        boundary_policy=boundary_policy(environ) if policy is None else str(policy),
    )


# ---------------------------------------------------------------------------
# Span grouping
# ---------------------------------------------------------------------------


def group_spans(
    spans: Sequence[tuple[int, int]], group: int, *, chunk_rows: int | None = None
) -> list[list[tuple[int, int]]]:
    """Partition chunk spans into contiguous groups of at most ``group``.

    Two rules, both about not changing anything except the MoE's row count:

    * a span **narrower than the configured chunk** ends its group and forms
      its own.  Those are the GDN boundary tail grid (256 tokens) and the
      final short chunk; grouping them would batch a handful of rows with a
      full chunk for no gain and would silently move where the tail grid's
      boundaries land.
    * groups never span a gap: the spans must be contiguous.  A discontiguity
      (restored-prefix suffix prefills cut spans that way) ends the group.
    """

    ordered = [(int(a), int(b)) for a, b in spans]
    g = max(1, int(group))
    if g == 1:
        return [[span] for span in ordered]
    width = max((b - a) for a, b in ordered) if ordered else 0
    if chunk_rows:
        width = int(chunk_rows)
    out: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for start, end in ordered:
        narrow = (end - start) < width
        contiguous = bool(current) and current[-1][1] == start
        if current and (narrow or not contiguous or len(current) >= g):
            out.append(current)
            current = []
        current.append((start, end))
        if narrow or len(current) >= g:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out
