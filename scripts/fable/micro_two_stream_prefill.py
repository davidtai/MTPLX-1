#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Falsify (or bank) wavefront prefill: two Metal queues, one layer apart.

THE IDEA UNDER TEST
-------------------
Chunked prefill runs 8 x 2,048 rows through 48 layers, one chunk at a time,
with a host-blocking ``mx.eval`` at every chunk boundary
(``mtplx.generation._prefill_committed_mtp_history_streaming``).  The chunk
loop is strictly serial, but the (chunk, layer) grid is not:

    chunk k+1 @ layer L   depends on chunk k @ layer L   (KV + GDN state)
    chunk k   @ layer L+1 depends on chunk k @ layer L   (hidden state)
    chunk k+1 @ layer L   and chunk k @ layer L+1  are INDEPENDENT

So a wavefront -- process the anti-diagonals of the grid -- can keep two
layer bodies in flight at once.  The pitch is that the bandwidth-bound
families (elementwise, copies, norms, routing gathers, softmax/mask; ~1.9 s
of the 16K prefill census) would overlap the compute-bound GEMMs instead of
queueing behind them.

MLX exposes the machinery: ``mx.new_stream(mx.gpu)`` makes a second stream,
``mx.stream(s)`` scopes op placement, and MLX inserts the cross-stream
dependency automatically when an array produced on one stream is consumed on
another.  Correctness is therefore free; *concurrency* is the open question.

WHY THIS IS A FALSIFIER AND NOT A BENCHMARK
-------------------------------------------
Three priors say this probably does not work, and each has a receipt:

1. ``mtplx/cache_state.py`` already ships a two-plus-stream route
   (``MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE=async_per_head``: one
   ``mx.new_stream`` per KV head, one ``mx.async_eval`` each).  Issue #228
   measured it **4-7x SLOWER** than the single-stream route below 64K
   context.  Extra Metal queues on this box have already been observed to
   cost more than they buy.

2. ``mtplx/prefill_rungs.py`` says the prefill GPU idle it was written for is
   *host-build lag*, not queue starvation: "MLX builds a whole prefill-chunk
   forward lazily and dispatches only at the end-of-chunk eval, so the GPU
   idles while the host walks 64 layers of graph construction."  Its fix is
   one stream and periodic ``mx.async_eval`` rungs.  If the idle is host lag,
   a second stream makes it WORSE -- MLX streams are thread-local
   (``mtplx/backends/gemma4_assistant.py:2340``), so both lanes' graphs are
   built by the same Python thread and the wavefront doubles the host work
   per wavefront step while the GPU still waits on it.

3. A 2,048-row layer body is not a small kernel.  If the routed MoE grouped
   GEMM already saturates the GPU, there is no idle width for a second
   stream to fill and arm (c) below will come back flat.

A NO-GO here is cheap and closes the row.  A GO is the expensive answer and
is deliberately made hard to fake: see "WHERE THIS BENCH IS OPTIMISTIC".

ARMS
----
All arms run the SAME four layer bodies over the SAME synthetic weights, at
the same 2,048 rows and the same context offset.  Only stream placement
changes, so a difference is scheduling and nothing else.

    A = a GDN DecoderLayer   (layer_types[0]  == "linear_attention")
    B = a QSA DecoderLayer   (layer_types[3]  == "full_attention")

The four nodes of one 2-chunk x 2-layer wavefront tile:

    n0 = A(h_c0)   chunk 0 @ layer L      -- writes cache_a
    n1 = B(n0)     chunk 0 @ layer L+1    -- writes cache_b
    n2 = A(h_c1)   chunk 1 @ layer L      -- reads cache_a after n0
    n3 = B(n2)     chunk 1 @ layer L+1    -- reads cache_b after n1

  (a) ``serial``      all four on the default stream, one eval.  This is
                      today's shape: n0 -> n1 -> n2 -> n3.
  (b) ``wavefront``   n0 on stream A; then n1 on stream A and n2 on stream B
                      CONCURRENTLY (the anti-diagonal); then n3 on stream B.
                      Identical ops, identical order, two queues.
  (c) ``independent`` A and B with private caches and pre-evaluated inputs,
                      one per stream -- no dependency at all.  This is the
                      upper bound and the concurrency probe: if (c) is not
                      meaningfully faster than running A and B back to back
                      on one stream, MLX/Metal is time-slicing the two queues
                      (or the GPU is already saturated) and (b) cannot win.

Solo timings ``t_gdn`` and ``t_qsa`` (one body, one stream, alone) anchor the
overlap arithmetic:

    overlap_fraction = (t_gdn + t_qsa - t_pair) / min(t_gdn, t_qsa)

    1.0  the shorter body was fully hidden inside the longer one
    0.0  no overlap at all -- two queues behaved as one
   <0.0  contention: the pair cost MORE than running them back to back

GO GATE
-------
GO only if arm (b) is at least 15% faster than arm (a) -- i.e.
``total_ms(wavefront) <= 0.85 * total_ms(serial)``, where total is host build
PLUS encode+GPU -- AND arm (c) shows real
concurrency (``overlap_fraction(independent) > 0``) -- AND the tile's own
ceiling clears the gate.  A (b) win with a flat (c) is a measurement
artifact, not a mechanism, and the script says so.

Only ONE of the four nodes' pairs can overlap in a 2x2 tile (step 1 =
{n1, n2}), so the largest saving the tile can possibly show is
``min(t_gdn, t_qsa) / (2*t_gdn + 2*t_qsa)`` -- 25% when the two bodies cost
the same, 12.5% when one is 3x the other, and it falls from there.  The
script prints that ceiling next to the measured saving.  **If the ceiling is
below 15%, the gate is unreachable at this tile depth and the measurement
says nothing about the row**: the honest read is then that the two bodies are
too lopsided for a 2-lane wavefront to matter, and a deeper wavefront is the
only shape left -- which the memory model in
``docs/perf/pr391-prefill-wavefront-two-stream.md`` refuses at this geometry.

EXACTNESS
---------
Arms (a) and (b) build the same graph in the same order; only the stream
annotation differs, and ``mx.async_eval``/stream placement change scheduling,
never values.  The script therefore asserts arm (b) is **bit-identical** to
arm (a) -- outputs AND the resulting cache state -- and prints the differing
element count.  A nonzero count is a finding, not a rounding allowance: it
would mean MLX reassociated something across the stream boundary, and that
alone would kill the row.

WHERE THIS BENCH IS OPTIMISTIC (read before believing a GO)
-----------------------------------------------------------
* ``--share-moe`` (the default) gives both DecoderLayers the SAME 512-expert
  4-bit bank, because two distinct banks are ~2.8 GB and this script holds
  itself to a 2 GiB weight budget.  Two streams reading the same 1.4 GB is
  slightly friendlier to the memory system than two streams reading 2.8 GB,
  so sharing can only FLATTER the concurrent arms.  A NO-GO under sharing is
  therefore trustworthy; a GO must be re-run with ``--no-share-moe
  --allow-large`` before it is banked.
* The tile is two layers deep, not 48.  Real prefill would pay the wavefront
  bookkeeping 48 times per chunk pair; this bench pays it once.
* Synthetic weights are random, so expert routing is near-uniform.  Real
  routing is skewed, which changes grouped-GEMM tile occupancy but not the
  overlap question.

The bench is NOT optimistic about the thing being tested: arms (a) and (b)
issue identical work from the same thread.

MEMORY
------
The script refuses to run a geometry whose projected peak exceeds
``--max-peak-bytes`` (default 24 GiB) and prints the projection either way.
The projection is deliberately conservative -- it assumes the dense QSA
attention score tensor is materialized (``n_heads x rows x context x 4`` fp32
plus a softmax twin) and adds ``mtplx.memory_plan``'s measured indexer
transient (12.75 B per row-context element) for EACH lane, because the
wavefront's whole point is holding two lanes live.  ``--context-before``
defaults to 6,144 (chunk 3 of the 16,384-token production cell) rather than
14,336 (the last chunk) precisely so the default geometry stays a
microbench.  Pass ``--context-before 14336`` for the production-tail cell and
read the printed projection before arming it.

GUARDED COMMAND
---------------
Read the live served model id first and pass the plist the guard captured;
never hand-edit the lock.

    /opt/homebrew/bin/python3 \\
      /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \\
      --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \\
      --lock-timeout-seconds 120 \\
      --timeout-seconds 1800 \\
      --child-timeout-seconds 1500 \\
      -- env PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w54-wavefront-prefill \\
         /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \\
         /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w54-wavefront-prefill/scripts/fable/micro_two_stream_prefill.py \\
         --reps 12 --warmup 3 \\
         --out /Users/davidtai/projects/OpenSourceWTF/bench/results/two-stream-prefill.json

``--self-test`` and ``--shapes`` import no MLX and issue no Metal work; they
are safe to run outside the window and are what the unit tests exercise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The schedule is IMPORTED, never copied: the seam module owns it, so a bench
# result and any future wiring can never disagree about what a wavefront is.
# ``mtplx.fable_prefill_wavefront`` is pure arithmetic and pulls in no MLX,
# which is what keeps ``--self-test`` / ``--shapes`` runnable off-window.
from mtplx.fable_prefill_wavefront import (  # noqa: E402
    lanes_live,
    overlappable_step_fraction,
    wavefront_steps,
)

# MLX and mtplx.models are imported lazily so ``--self-test`` / ``--shapes``
# stay runnable on a box that is not holding the GPU lock.
mx = None
nn = None

LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
BANNER = (
    "[micro_two_stream_prefill] GPU WINDOW REQUIRED -- run under "
    f"{LOCK_PATH} via bench/laguna/run_guarded.py"
)

# --- Qwen3.8-Flash-Next geometry (mtplx.models.qwen4_exp.TextArgs defaults) --
CHUNK_ROWS = 2048           # MTPLX_PREFILL_CHUNK_SIZE_DENSE
DEFAULT_CONTEXT_BEFORE = 6144   # chunk 3 of the 8 x 2048 production cell
PRODUCTION_TAIL_CONTEXT = 14336  # chunk 7 -- the widest, priciest chunk
GDN_LAYER_IDX = 0           # layer_types[0]  == "linear_attention"
QSA_LAYER_IDX = 3           # layer_types[3]  == "full_attention"
MOE_BITS = 4                # the 2026-08-27 reforge: 4-bit affine
MOE_GROUP_SIZE = 64         # ... at group 64
DEFAULT_MAX_WEIGHT_BYTES = 2 * 1024**3
#: Pessimistic-projection cap, not a machine limit.  ``project_peak_bytes``
#: assumes the dense QSA score tensor is fully materialized, so a geometry
#: that clears this cap is safe even if MLX does materialize it; the
#: refusal exists to catch a context width nobody priced (65K+), not to
#: shave GiB inside an exclusive window.
DEFAULT_MAX_PEAK_BYTES = 24 * 1024**3

#: mtplx.memory_plan.QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM -- the measured
#: dense-lane indexer transient per (chunk row x context token), for ONE
#: layer. Duplicated as a float here so ``--shapes`` needs no mtplx import;
#: ``test_micro_two_stream_prefill.py`` asserts the two stay equal.
QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM = 12.75

GO_THRESHOLD = 0.15         # arm (b) must be >=15% faster than arm (a)

ARMS = ("serial", "wavefront", "independent")


# ---------------------------------------------------------------------------
# Pure arithmetic (no MLX) -- the part the unit tests can run
# ---------------------------------------------------------------------------
def overlap_fraction(t_solo_a: float, t_solo_b: float, t_pair: float) -> float:
    """How much of the shorter body disappeared inside the longer one.

    ``1.0`` = perfect overlap, ``0.0`` = the two queues behaved as one,
    negative = the pair cost more than running them back to back (contention).
    """

    shorter = min(float(t_solo_a), float(t_solo_b))
    if shorter <= 0.0:
        return 0.0
    return (float(t_solo_a) + float(t_solo_b) - float(t_pair)) / shorter


def tile_speedup_ceiling(t_gdn: float, t_qsa: float) -> float:
    """Best fractional saving a 2-chunk x 2-layer wavefront tile can reach.

    The tile has four nodes and exactly one overlappable pair (step 1 =
    ``{n1, n2}``), so at perfect overlap the tile costs
    ``2*t_gdn + 2*t_qsa - min(t_gdn, t_qsa)``.
    """

    serial = 2.0 * float(t_gdn) + 2.0 * float(t_qsa)
    if serial <= 0.0:
        return 0.0
    return min(float(t_gdn), float(t_qsa)) / serial


def go_verdict(
    serial_ms: float,
    wavefront_ms: float,
    independent_overlap: float,
    ceiling: float | None = None,
) -> tuple[bool, float, str]:
    """``(go, saving_fraction, reason)`` for the 15% gate.

    Three ways to fail, in the order they are checked:

    * ``ceiling`` below the gate -- the tile's own arithmetic cannot reach
      15% however perfectly the GPU overlaps, so the run is INCONCLUSIVE for
      the row rather than a NO-GO for it;
    * no concurrency in arm (c) -- the two arms differ only in stream
      placement, so a (b) win with a flat (c) has no mechanism behind it;
    * the saving simply misses the gate.
    """

    if serial_ms <= 0.0:
        return False, 0.0, "serial arm did not time"
    saving = (float(serial_ms) - float(wavefront_ms)) / float(serial_ms)
    if ceiling is not None and float(ceiling) < GO_THRESHOLD:
        return (
            False,
            saving,
            f"INCONCLUSIVE: the tile's ceiling is {float(ceiling) * 100:.1f}%, "
            f"below the {GO_THRESHOLD * 100:.0f}% gate -- the two bodies are "
            "too lopsided for a 2-lane tile to reach it even at perfect "
            "overlap. Report the ceiling, not a NO-GO for the row.",
        )
    if independent_overlap <= 0.0:
        return (
            False,
            saving,
            "NO-GO: arm (c) shows no concurrency (overlap "
            f"{independent_overlap:+.3f}); MLX is time-slicing the second "
            "queue or the GPU is already saturated",
        )
    if saving < GO_THRESHOLD:
        return (
            False,
            saving,
            f"NO-GO: wavefront saved {saving * 100:.1f}%, gate is "
            f"{GO_THRESHOLD * 100:.0f}%",
        )
    return (
        True,
        saving,
        f"GO: wavefront saved {saving * 100:.1f}% with arm (c) overlap "
        f"{independent_overlap:+.3f}; re-run --no-share-moe --allow-large "
        "before banking",
    )


def project_peak_bytes(
    *,
    weight_bytes: int,
    rows: int,
    context_before: int,
    lanes_live: int,
    n_heads: int = 24,
    hidden_widened: int = 4 * 2560,
) -> dict[str, int]:
    """Conservative peak projection for ``lanes_live`` simultaneous lanes.

    Conservative on purpose: it assumes the dense QSA lane materializes the
    full fp32 score tensor plus a softmax twin (the #393 shape), which is the
    bound the guard should refuse against even if MLX's flash kernel avoids
    it on the day.  ``mtplx.memory_plan``'s measured indexer transient is
    added per lane on top.
    """

    rows = max(0, int(rows))
    total = max(0, int(context_before)) + rows
    lanes = max(1, int(lanes_live))
    scores = int(n_heads) * rows * total * 4 * 2
    indexer = int(QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM * rows * total)
    activations = rows * int(hidden_widened) * 2 * 4  # a few widened leaves
    per_lane = scores + indexer + activations
    return {
        "weight_bytes": int(weight_bytes),
        "lanes_live": lanes,
        "per_lane_transient_bytes": per_lane,
        "dense_scores_bytes": scores,
        "indexer_transient_bytes": indexer,
        "projected_peak_bytes": int(weight_bytes) + lanes * per_lane,
    }


# ---------------------------------------------------------------------------
# MLX construction
# ---------------------------------------------------------------------------
def _require_mlx() -> None:
    global mx, nn
    import mlx.core
    import mlx.nn

    mx = mlx.core
    nn = mlx.nn


def build_args():
    """The production TextArgs, imported (never copied) so it cannot drift."""

    from mtplx.models.qwen4_exp import TextArgs

    args = TextArgs()
    assert args.layer_types[GDN_LAYER_IDX] == "linear_attention", (
        f"layer {GDN_LAYER_IDX} is {args.layer_types[GDN_LAYER_IDX]}"
    )
    assert args.layer_types[QSA_LAYER_IDX] == "full_attention", (
        f"layer {QSA_LAYER_IDX} is {args.layer_types[QSA_LAYER_IDX]}"
    )
    assert not args.ple_layer_ids, (
        "default TextArgs must carry no PLE layer; this bench does not stage "
        "the n-gram sidecar"
    )
    return args


def parameter_bytes(module) -> int:
    from mlx.utils import tree_flatten

    return sum(int(v.nbytes) for _, v in tree_flatten(module.parameters()))


def build_bodies(args, *, share_moe: bool, seed: int, quantize: bool = True):
    """One GDN DecoderLayer and one QSA DecoderLayer, quantized like the pack.

    Returns ``(gdn_layer, qsa_layer, weight_bytes)``.  ``share_moe`` points
    both layers' ``mlp`` at ONE SparseMoeBlock: read the "WHERE THIS BENCH IS
    OPTIMISTIC" note before trusting a GO measured under it.

    ``quantize=False`` exists for the CPU wiring test, whose tiny geometry has
    dimensions below the 64-wide quantization group.
    """

    from mtplx.models.qwen4_exp import DecoderLayer

    mx.random.seed(int(seed))
    gdn = DecoderLayer(args, GDN_LAYER_IDX)
    qsa = DecoderLayer(args, QSA_LAYER_IDX)
    for layer in (gdn, qsa):
        layer.eval()
        if quantize:
            nn.quantize(layer, group_size=MOE_GROUP_SIZE, bits=MOE_BITS)
        mx.eval(layer.parameters())
    # Distinct bytes on the device, which is what the budget is about: a
    # shared bank is counted ONCE.
    weight_bytes = parameter_bytes(gdn) + parameter_bytes(qsa)
    if share_moe:
        # Assigning the same Module to both parents is legal in MLX; the
        # expert weights are read-only during a forward, so sharing changes
        # no dependency -- only how many distinct bytes the two lanes read.
        # Quantize first, then share, so both banks were built the same way
        # and the only difference between the two modes is aliasing.
        weight_bytes -= parameter_bytes(qsa.mlp)
        qsa.mlp = gdn.mlp
    return gdn, qsa, weight_bytes


def prime_qsa_cache(qsa_layer, args, *, context_before: int, rows: int, dtype):
    """Advance a fresh QSACache to ``context_before`` using the production path.

    ``DecoderLayer.__call__(kv_only=True)`` writes exactly the indexer's raw
    and pooled rows plus the attention K/V for the tokens it is handed, and
    computes nothing else (``QSAIndexer(..., write_only=True)`` never scores).
    That is the cheapest priming that leaves a cache the real forward would
    have produced.
    """

    from mtplx.models.qwen4_exp import QSACache

    cache = QSACache(args.indexer_compress_ratio or 4)
    remaining = int(context_before)
    while remaining > 0:
        span = min(rows, remaining)
        prime = (mx.random.normal((1, span, args.hc_count * args.hidden_size))
                 * 0.3).astype(dtype)
        qsa_layer(prime, input_ids=None, ssm_mask=None, cache=cache,
                  kv_only=True)
        mx.eval([a for a in cache.state if a is not None])
        remaining -= span
    assert cache.offset == int(context_before), (
        f"primed to {cache.offset}, wanted {context_before}"
    )
    return tuple(cache.state)


def gdn_state_template(args, *, dtype):
    """Fixed-shape GDN recurrent state: the conv tape and the delta matrix.

    Neither grows with context -- only their VALUES depend on history -- so a
    synthetic state of the right shapes is the honest cheap priming.  Running
    14,336 rows of real GDN scan to get them would change nothing this bench
    measures and would cost more memory than the measurement.
    """

    conv_dim = (
        args.linear_key_head_dim * args.linear_num_key_heads * 2
        + args.linear_value_head_dim * args.linear_num_value_heads
    )
    conv_tape = (mx.random.normal(
        (1, args.linear_conv_kernel_dim - 1, conv_dim)) * 0.1).astype(dtype)
    delta = (mx.random.normal(
        (1, args.linear_num_value_heads,
         args.linear_value_head_dim, args.linear_key_head_dim)
    ) * 0.1).astype(mx.float32)
    mx.eval(conv_tape, delta)
    return conv_tape, delta


def fresh_gdn_cache(state, *, context_before: int):
    """A new ArraysCache bound to the shared state template.

    Fresh per rep because the forward rebinds ``cache[0]``/``cache[1]``; the
    arrays themselves are never mutated, so every rep starts from exactly the
    same recurrent state and the arms stay comparable.
    """

    from mlx_lm.models.cache import ArraysCache

    cache = ArraysCache(size=2)
    cache[0], cache[1] = state
    advance = getattr(cache, "advance", None)
    if callable(advance):
        advance(int(context_before))
    return cache


def fresh_qsa_cache(args, state):
    """A new QSACache bound to the primed state tuple (keys, values, raw, pooled)."""

    from mtplx.models.qwen4_exp import QSACache

    cache = QSACache(args.indexer_compress_ratio or 4)
    cache.state = list(state)
    return cache


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _kick(*arrays) -> None:
    """Hand the queue what has been built so far without blocking the host."""

    live = [a for a in arrays if a is not None]
    if live:
        mx.async_eval(*live)


def run_tile(gdn, qsa, h_c0, h_c1, caches, *, streams):
    """The four-node wavefront tile.  ``streams=None`` puts it all on one.

    Op-for-op identical in both modes: the ONLY difference is which stream
    each body is annotated with.  Returns ``(n1, n3, state)`` where ``state``
    is the post-tile cache contents, so the caller can compare values as well
    as timings.
    """

    cache_a, cache_b = caches
    if streams is None:
        n0 = gdn(h_c0, input_ids=None, ssm_mask=None, cache=cache_a)
        n1 = qsa(n0, input_ids=None, ssm_mask=None, cache=cache_b)
        n2 = gdn(h_c1, input_ids=None, ssm_mask=None, cache=cache_a)
        n3 = qsa(n2, input_ids=None, ssm_mask=None, cache=cache_b)
        return n1, n3, (cache_a[0], cache_a[1], cache_b.kv.keys,
                        cache_b.kv.values)

    stream_a, stream_b = streams
    # step 0: {(chunk 0, layer L)}
    with mx.stream(stream_a):
        n0 = gdn(h_c0, input_ids=None, ssm_mask=None, cache=cache_a)
    state_after_c0 = (cache_a[0], cache_a[1])
    _kick(n0, *state_after_c0)
    # step 1: {(chunk 0, layer L+1), (chunk 1, layer L)} -- the overlap
    with mx.stream(stream_a):
        n1 = qsa(n0, input_ids=None, ssm_mask=None, cache=cache_b)
    kv_after_c0 = (cache_b.kv.keys, cache_b.kv.values)
    _kick(n1, *kv_after_c0)
    with mx.stream(stream_b):
        n2 = gdn(h_c1, input_ids=None, ssm_mask=None, cache=cache_a)
    _kick(n2, cache_a[0], cache_a[1])
    # step 2: {(chunk 1, layer L+1)}
    with mx.stream(stream_b):
        n3 = qsa(n2, input_ids=None, ssm_mask=None, cache=cache_b)
    return n1, n3, (cache_a[0], cache_a[1], cache_b.kv.keys, cache_b.kv.values)


def run_independent(gdn, qsa, h_gdn, h_qsa, caches, *, streams):
    """One GDN body and one QSA body with NO dependency between them.

    The concurrency probe.  ``streams=None`` runs them back to back on one
    stream; two streams is the ceiling arm (c).
    """

    cache_a, cache_b = caches
    if streams is None:
        out_a = gdn(h_gdn, input_ids=None, ssm_mask=None, cache=cache_a)
        out_b = qsa(h_qsa, input_ids=None, ssm_mask=None, cache=cache_b)
        return out_a, out_b
    stream_a, stream_b = streams
    with mx.stream(stream_a):
        out_a = gdn(h_gdn, input_ids=None, ssm_mask=None, cache=cache_a)
    _kick(out_a)
    with mx.stream(stream_b):
        out_b = qsa(h_qsa, input_ids=None, ssm_mask=None, cache=cache_b)
    return out_a, out_b


def time_arm(build, *, reps: int, warmup: int):
    """Wall time per rep, with the host-build / encode+GPU split kept apart.

    ``median_ms`` is the TOTAL (build + eval) and it is what every verdict
    reads.  Using eval alone would flatter the wavefront arm for free: its
    ``mx.async_eval`` calls deliberately move GPU work out of the eval window
    and into the build window, so an eval-only comparison would credit the
    wavefront with work it merely relocated.  ``build_ms`` and ``eval_ms``
    stay in the receipt because the split is the diagnosis -- a wavefront that
    loses on total while its build doubles is host-bound, which is prior #2.

    Queued lane: build the whole arm, ONE host sync at the end (see
    ``docs/laguna-mlxfast-port/bench/scratchpad_dense_mlp_check.py``).
    """

    for _ in range(warmup):
        mx.eval(build())
    mx.synchronize()
    evals, builds, totals = [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = build()
        t1 = time.perf_counter()
        mx.eval(out)
        mx.synchronize()
        t2 = time.perf_counter()
        builds.append((t1 - t0) * 1e3)
        evals.append((t2 - t1) * 1e3)
        totals.append((t2 - t0) * 1e3)
    ordered = sorted(totals)
    return {
        "median_ms": statistics.median(totals),
        "p10_ms": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "build_ms": statistics.median(builds),
        "eval_ms": statistics.median(evals),
        "reps": reps,
    }


def numerics(got, ref) -> tuple[float, int]:
    """(max abs diff, differing element count) over every paired array."""

    worst, differing = 0.0, 0
    for a, b in zip(got, ref):
        if a is None or b is None:
            continue
        if a.shape != b.shape:
            return float("inf"), -1
        af = a.astype(mx.float32)
        bf = b.astype(mx.float32)
        diff = mx.abs(af - bf)
        worst = max(worst, float(mx.max(diff).item()))
        differing += int(mx.sum(diff != 0).item())
    return worst, differing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=CHUNK_ROWS,
                   help="prefill chunk rows (default: the shipped 2048)")
    p.add_argument("--context-before", type=int, default=DEFAULT_CONTEXT_BEFORE,
                   help=(f"tokens already in cache (default {DEFAULT_CONTEXT_BEFORE}"
                         f"; the production tail chunk is {PRODUCTION_TAIL_CONTEXT})"))
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--share-moe", dest="share_moe", action="store_true",
                   default=True,
                   help="both layers use ONE expert bank (default; flatters "
                        "the concurrent arms -- see the docstring)")
    p.add_argument("--no-share-moe", dest="share_moe", action="store_false",
                   help="distinct expert banks; needs --allow-large")
    p.add_argument("--max-weight-bytes", type=int,
                   default=DEFAULT_MAX_WEIGHT_BYTES)
    p.add_argument("--max-peak-bytes", type=int, default=DEFAULT_MAX_PEAK_BYTES,
                   help="refuse a geometry whose PESSIMISTIC projection "
                        "exceeds this (default 24 GiB)")
    p.add_argument("--allow-large", action="store_true",
                   help="waive the weight/peak refusals (say why in the receipt)")
    p.add_argument("--shapes", action="store_true",
                   help="print the geometry and the memory projection, run nothing")
    p.add_argument("--self-test", action="store_true",
                   help="exercise the pure schedule/verdict arithmetic, no MLX")
    p.add_argument("--out", type=str, default="")
    return p


def self_test() -> int:
    steps = wavefront_steps(2, 2, lanes=2)
    assert steps == [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]], steps
    assert lanes_live(2, 2, lanes=2) == 2
    assert lanes_live(8, 48, lanes=2) == 2
    assert lanes_live(8, 48, lanes=0) == 8, "unbounded wavefront width"
    assert lanes_live(1, 48, lanes=2) == 1
    for k, layer in ((k, layer) for step in wavefront_steps(4, 6, lanes=2)
                     for k, layer in step):
        assert 0 <= k < 4 and 0 <= layer < 6
    assert abs(overlap_fraction(10.0, 4.0, 10.0) - 1.0) < 1e-9
    assert abs(overlap_fraction(10.0, 4.0, 14.0) - 0.0) < 1e-9
    assert overlap_fraction(10.0, 4.0, 16.0) < 0.0
    go, saving, reason = go_verdict(100.0, 80.0, 0.5)
    assert go and abs(saving - 0.20) < 1e-9, (go, saving, reason)
    go, _, reason = go_verdict(100.0, 80.0, 0.0)
    assert not go and "no concurrency" in reason, reason
    go, _, reason = go_verdict(100.0, 90.0, 0.5)
    assert not go and "gate is" in reason, reason
    go, _, reason = go_verdict(100.0, 80.0, 0.5, 0.125)
    assert not go and "INCONCLUSIVE" in reason, reason
    go, _, reason = go_verdict(100.0, 80.0, 0.5, 0.25)
    assert go, reason
    ceiling = tile_speedup_ceiling(10.0, 30.0)
    assert abs(ceiling - 10.0 / 80.0) < 1e-9, ceiling
    proj = project_peak_bytes(weight_bytes=0, rows=2048,
                              context_before=14336, lanes_live=2)
    assert proj["projected_peak_bytes"] == 2 * proj["per_lane_transient_bytes"]
    print("[self-test] ok: schedule, overlap, verdict, ceiling, projection")
    return 0


def print_shapes(args_ns) -> int:
    rows = args_ns.rows
    total = rows + args_ns.context_before
    steps = wavefront_steps(2, 2, lanes=2)
    print(BANNER)
    print(f"\nchunk rows          {rows}")
    print(f"context before      {args_ns.context_before}")
    print(f"total tokens        {total}")
    print(f"GDN layer index     {GDN_LAYER_IDX} (linear_attention)")
    print(f"QSA layer index     {QSA_LAYER_IDX} (full_attention)")
    print(f"MoE quant           {MOE_BITS}-bit group {MOE_GROUP_SIZE}"
          f"{'  SHARED bank' if args_ns.share_moe else '  distinct banks'}")
    print(f"wavefront tile      {steps}")
    print(f"lanes live          {lanes_live(2, 2, lanes=2)}")
    print(f"full-prefill lanes  {lanes_live(8, 48, lanes=2)} bounded / "
          f"{lanes_live(8, 48, lanes=0)} unbounded (8 chunks x 48 layers)")
    print(f"overlappable steps  "
          f"{overlappable_step_fraction(8, 48, lanes=2) * 100:.1f}% "
          "(the rest are group drains)")
    proj = project_peak_bytes(weight_bytes=0, rows=rows,
                              context_before=args_ns.context_before,
                              lanes_live=2)
    gib = 1024**3
    print(f"\nper-lane transient  {proj['per_lane_transient_bytes'] / gib:.2f} GiB"
          f"  (dense scores {proj['dense_scores_bytes'] / gib:.2f} GiB"
          f" + indexer {proj['indexer_transient_bytes'] / gib:.2f} GiB)")
    print(f"2-lane transient    "
          f"{2 * proj['per_lane_transient_bytes'] / gib:.2f} GiB")
    print(f"peak cap            {args_ns.max_peak_bytes / gib:.2f} GiB")
    return 0


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.self_test:
        return self_test()
    if ns.shapes:
        return print_shapes(ns)

    print(BANNER, flush=True)
    _require_mlx()
    args = build_args()
    dtype = mx.bfloat16
    rows = int(ns.rows)
    widened = args.hc_count * args.hidden_size

    gdn, qsa, weight_bytes = build_bodies(
        args, share_moe=ns.share_moe, seed=ns.seed
    )
    proj = project_peak_bytes(
        weight_bytes=weight_bytes, rows=rows,
        context_before=ns.context_before, lanes_live=2,
        n_heads=args.num_attention_heads, hidden_widened=widened,
    )
    gib = 1024**3
    print(f"[mem] weights {weight_bytes / gib:.2f} GiB  "
          f"per-lane transient {proj['per_lane_transient_bytes'] / gib:.2f} GiB  "
          f"projected peak {proj['projected_peak_bytes'] / gib:.2f} GiB",
          flush=True)
    # The weight refusal fires AFTER construction because the budget is about
    # measured device bytes, not an estimate; a 2.8 GB over-budget build is
    # safe to allocate and then abandon inside an exclusive window. The PEAK
    # refusal is the one that matters and it fires before any forward runs.
    if not ns.allow_large:
        if weight_bytes > ns.max_weight_bytes:
            raise SystemExit(
                f"weights {weight_bytes / gib:.2f} GiB exceed the "
                f"{ns.max_weight_bytes / gib:.2f} GiB budget. Keep --share-moe, "
                "or pass --allow-large and say why in the receipt."
            )
        if proj["projected_peak_bytes"] > ns.max_peak_bytes:
            raise SystemExit(
                f"projected peak {proj['projected_peak_bytes'] / gib:.2f} GiB "
                f"exceeds the {ns.max_peak_bytes / gib:.2f} GiB cap at "
                f"context_before={ns.context_before}. Lower --context-before "
                "or pass --allow-large."
            )

    def fresh_inputs(n: int):
        return [
            (mx.random.normal((1, rows, widened)) * 0.3).astype(dtype)
            for _ in range(n)
        ]

    h_c0, h_c1, h_ind_gdn, h_ind_qsa = fresh_inputs(4)
    mx.eval(h_c0, h_c1, h_ind_gdn, h_ind_qsa)

    stream_a = mx.new_stream(mx.gpu)
    stream_b = mx.new_stream(mx.gpu)
    results: dict[str, dict] = {}

    from mtplx.attention_context import attention_phase

    with attention_phase("prefill"):
        gdn_state = gdn_state_template(args, dtype=dtype)
        qsa_state = prime_qsa_cache(
            qsa, args, context_before=ns.context_before, rows=rows, dtype=dtype
        )

        def caches():
            return (
                fresh_gdn_cache(gdn_state, context_before=ns.context_before),
                fresh_qsa_cache(args, qsa_state),
            )

        # --- solo anchors -------------------------------------------------
        def solo_gdn():
            return gdn(
                h_ind_gdn, input_ids=None, ssm_mask=None,
                cache=fresh_gdn_cache(
                    gdn_state, context_before=ns.context_before),
            )

        def solo_qsa():
            return qsa(h_ind_qsa, input_ids=None, ssm_mask=None,
                       cache=fresh_qsa_cache(args, qsa_state))

        results["solo_gdn"] = time_arm(solo_gdn, reps=ns.reps,
                                       warmup=ns.warmup)
        results["solo_qsa"] = time_arm(solo_qsa, reps=ns.reps,
                                       warmup=ns.warmup)
        t_gdn = results["solo_gdn"]["median_ms"]
        t_qsa = results["solo_qsa"]["median_ms"]

        # --- arm (c): independent pair, one stream vs two -----------------
        def indep(streams):
            def build():
                return run_independent(gdn, qsa, h_ind_gdn, h_ind_qsa,
                                       caches(), streams=streams)
            return build

        results["independent_1stream"] = time_arm(
            indep(None), reps=ns.reps, warmup=ns.warmup)
        results["independent_2stream"] = time_arm(
            indep((stream_a, stream_b)), reps=ns.reps, warmup=ns.warmup)

        # --- arms (a) and (b): the wavefront tile -------------------------
        def tile(streams):
            def build():
                return run_tile(gdn, qsa, h_c0, h_c1, caches(),
                                streams=streams)
            return build

        # Exactness receipt: same graph, same order, different streams.
        serial_out = tile(None)()
        mx.eval(serial_out)
        wave_out = tile((stream_a, stream_b))()
        mx.eval(wave_out)
        flat_serial = [serial_out[0], serial_out[1], *serial_out[2]]
        flat_wave = [wave_out[0], wave_out[1], *wave_out[2]]
        worst, differing = numerics(flat_wave, flat_serial)
        del serial_out, wave_out, flat_serial, flat_wave
        mx.clear_cache()

        results["serial"] = time_arm(tile(None), reps=ns.reps,
                                     warmup=ns.warmup)
        results["wavefront"] = time_arm(tile((stream_a, stream_b)),
                                        reps=ns.reps, warmup=ns.warmup)

    ind_overlap = overlap_fraction(
        t_gdn, t_qsa, results["independent_2stream"]["median_ms"])
    ind_overlap_1s = overlap_fraction(
        t_gdn, t_qsa, results["independent_1stream"]["median_ms"])
    ceiling = tile_speedup_ceiling(t_gdn, t_qsa)
    go, saving, reason = go_verdict(
        results["serial"]["median_ms"],
        results["wavefront"]["median_ms"],
        ind_overlap,
        ceiling,
    )

    hdr = (f"{'arm':<24}{'total ms':>10}{'p10':>9}{'p90':>9}"
           f"{'build ms':>10}{'eval ms':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in ("solo_gdn", "solo_qsa", "independent_1stream",
                 "independent_2stream", "serial", "wavefront"):
        r = results[name]
        print(f"{name:<24}{r['median_ms']:>10.3f}{r['p10_ms']:>9.3f}"
              f"{r['p90_ms']:>9.3f}{r['build_ms']:>10.3f}"
              f"{r['eval_ms']:>10.3f}")

    print(f"\nconcurrency  (arm c) 2-stream overlap {ind_overlap:+.3f}   "
          f"1-stream control {ind_overlap_1s:+.3f}")
    print(f"tile ceiling         {ceiling * 100:.1f}%  "
          "(one overlappable pair in a 2x2 tile; below the gate makes the "
          "run inconclusive, not a NO-GO)")
    print(f"wavefront saving     {saving * 100:+.1f}%  "
          f"(gate {GO_THRESHOLD * 100:.0f}%)")
    print(f"exactness            max|diff| {worst:.6g}, differing {differing}"
          f"{'' if differing == 0 else '   <-- NOT EXACT, this kills the row'}")
    print(f"\nVERDICT  {reason}")

    summary = {
        "banner": BANNER,
        "geometry": {
            "rows": rows,
            "context_before": ns.context_before,
            "gdn_layer_idx": GDN_LAYER_IDX,
            "qsa_layer_idx": QSA_LAYER_IDX,
            "share_moe": ns.share_moe,
            "moe_bits": MOE_BITS,
            "moe_group_size": MOE_GROUP_SIZE,
            "wavefront_tile": [list(map(list, step))
                               for step in wavefront_steps(2, 2, lanes=2)],
        },
        "memory": proj,
        "arms": results,
        "derived": {
            "solo_gdn_ms": t_gdn,
            "solo_qsa_ms": t_qsa,
            "independent_2stream_overlap": ind_overlap,
            "independent_1stream_overlap": ind_overlap_1s,
            "tile_speedup_ceiling": ceiling,
            "wavefront_saving": saving,
            "go": go,
            "reason": reason,
        },
        "numerics": {"max_abs_diff": worst, "differing": differing},
    }
    if ns.out:
        Path(ns.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {ns.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
