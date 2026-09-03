#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""B1 falsifier: the routed grouped GEMM's rows-per-expert curve.

The prefill census (``scratchpad/J-prefill-attribution.md`` §2.2) puts the
routed MoE grouped GEMM at **3,391.6 ms, 31.6 % of prefill busy, 78.72 TFLOP
at 23.2 TFLOP/s** -- **45 % of the rate the same q4/g32 kernel reaches dense**
(51.4 TF/s).  B1 (``scratchpad/M-holistic-tps-program.md`` §B) claims that gap
is a *schedule* problem, not a kernel problem, and that the fix is more rows
per expert per GEMM.

Why rows/expert is the variable
-------------------------------
MTPLX prefills chunk-major: 2,048 tokens x top-10 = 20,480 (row, expert)
assignments over 512 experts = **40 rows per expert**.  The kernel MLX
selects (census: ``affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_32_b_4_bm_32_
bn_64_bk_64_wm_2_wn_2``, grid ``[20, 640, 1]``) tiles the sorted rows at
**BM = 32**, and its inner loop walks *runs of equal expert id inside each
32-row tile*::

    uint32_t index_next = indices[y_row];
    while (n < tgp_bm) {                  // one pass per DISTINCT expert
      ...                                 // in this 32-row tile
      thread loader_w_t loader_w(wl + index * stride_w, ...);
      for (int k = 0; k < K_it; k++) { loader_w.load(); ... }
    }

-- ``mlx/backend/metal/kernels/quantized_nax.h:1475``.  A tile that straddles
an expert boundary pays the **whole K loop twice**: two weight-tile streams,
two matmul passes, with the inactive simdgroups masked off by ``sg_active``.
At 40 rows/expert every expert boundary lands inside a tile
(``512 boundaries / 640 tiles``), so the average tile does **1.8 passes** for
32 rows of useful work.  At 320 rows/expert (a 16K-row super-chunk) it is
1.1 passes.  That single number -- ``runs / tiles`` -- is the whole B1 thesis,
and this bench measures it against the clock.

What it runs
------------
The production chain, at the served shapes (hidden 2560, ``moe_intermediate``
640, fused gate+up N = 1280, 512 experts, top-10, q4/g32), for
``--rows-per-expert 40,80,160,320``:

===== ============ ============ =============================
R     tokens       expert-rows  chunk this corresponds to
===== ============ ============ =============================
40    2,048        20,480       today's shipped chunk
80    4,096        40,960       MTPLX_PREFILL_CHUNK_SIZE=4096
160   8,192        81,920       2-chunk super-chunk at 4096
320   16,384       163,840      B1's 4-chunk super-chunk
===== ============ ============ =============================

Variants:

``chain``
    ``_gather_sort`` -> fused gate+up ``gather_qmm`` -> ``silu*up`` -> down
    ``gather_qmm`` -> ``_scatter_unsort``.  Exactly ``_FusedGateUpSwitchGLU.
    __call__`` (``mtplx/models/qwen4_exp.py:1099``), which is what the model
    runs at prefill widths.
``gemm``
    the two ``gather_qmm`` calls alone, on already-sorted inputs.  **This is
    the variant the verdict reads**, because it is the census family: J §2.2's
    "MoE routed grouped GEMM (gate/up + down, ``affine_gather_qmm_rhs``)",
    789 dispatches, 3,391.6 ms, 23.2 TF/s.  The sort, the routing gather and
    the scatter are three *separate* census families (221.5 + 85.4 ms), and in
    isolation here they cost far more than they do inside the model's fused
    48-layer chunk graph -- charging them to the GEMM would understate every
    setting by the same large constant and flatten the curve.

Every setting reports TF/s against **useful** FLOPs and against the FLOPs the
tile model says are actually *issued*, plus GB/s against the ideal weight
stream (each expert once) and against the tile model's re-reads.  ``runs`` and
``tile_reload_factor`` are computed exactly, in NumPy, from the sorted index
array -- no fitting.

Decision rule (M §B1)
---------------------
``gemm`` TF/s at R=320 vs R=40 (``chain`` is printed for context):

* **>= 1.5x** -- the schedule is the binding term; B1's -0.9..-1.1 s of MoE
  at 16K is real and the super-chunk earns its complexity.
* **1.15-1.5x** -- partial; the ~-0.4 s floor M quotes for "plateau by 160".
* **<= 1.15x** -- the grouped GEMM's 45 % is addressing/occupancy inside the
  kernel, not tiling.  B1 dies here and the lever moves to a kernel (a larger
  BM, or the expert-major grid ``micro_expert_major.py`` prices for decode).

Exactness (``--exactness``, on by default)
------------------------------------------
B1 also claims a super-chunk is **bit-exact**: more rows per expert changes
tiling, not per-row arithmetic.  The kernel backs that up -- ``BK`` is 64 in
every ``affine_gather_qmm_rhs_nax_nt_..._gs_32_b_4`` variant shipped in
``mlx.metallib`` (only ``bm_32``/``bm_64`` differ), so each output element's
K reduction runs in the same order at any M.  The probe measures it instead of
arguing it: one fixed block of 20,480 (row, expert) pairs is run alone, then
re-run as the *head* of a 163,840-row batch, and the probe rows are compared
**bitwise** after the unsort.  ``differing`` must be 0.

Standalone: builds its own q4 banks, imports nothing from ``mtplx``.
``--self-test`` runs the pure-NumPy accounting with no MLX and no lock.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Imported lazily (see micro_moe_dedup.py) so the CLI and the pure-python
# accounting stay testable off the GPU lock.
mx = None

LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")

#: Evidence that ``bench/laguna/run_guarded.py`` launched this process.  The
#: lock FILE always exists and is normally held by whoever owns the box, so
#: ``LOCK.exists()`` proves nothing and a flock probe only says "somebody" --
#: it cannot say "this run's guard".  ``abba_driver.acquire_guard(mode="auto")``
#: keys on exactly these two env vars, and so does this.
GUARD_ENV_VARS = ("MTPLX_DSV4_GUARD_WINDOW_PATH", "MTPLX_GUARD_ATTEST_FD")


def require_guard_window(allow_unlocked: bool) -> None:
    """Refuse to issue Metal work outside a guarded window."""

    import os

    if allow_unlocked:
        return
    if any(os.environ.get(name) for name in GUARD_ENV_VARS):
        return
    raise SystemExit(
        "no GPU guard evidence ("
        + " / ".join(GUARD_ENV_VARS)
        + " unset): run this under bench/laguna/run_guarded.py.  The lock file "
        "existing is NOT evidence -- it is normally held by another job, and "
        "starting Metal work here interrupts it.  --allow-unlocked overrides."
    )


# ---- served geometry -------------------------------------------------------
HIDDEN = 2560
MOE_INTERMEDIATE = 640
GU_OUT = 2 * MOE_INTERMEDIATE  # fused gate+up
NUM_EXPERTS = 512
TOP_K = 10
GROUP_SIZE = 32
BITS = 4

#: The only BM the shipped metallib offers for this kernel family alongside
#: bm_64; the census run took bm_32 at M = 20,480.  Both are bk_64, which is
#: why the row count cannot move the arithmetic.
KERNEL_BM = 32
KERNEL_BN = 64

DEFAULT_ROWS = (40, 80, 160, 320)
VARIANTS = ("chain", "gemm")


def _require_mlx():
    global mx
    if mx is None:
        import mlx.core as _mx

        mx = _mx
    return mx


# ---------------------------------------------------------------------------
# Pure-python accounting (unit-testable without MLX)
# ---------------------------------------------------------------------------


def flops_per_call(expert_rows: int) -> int:
    """Useful FLOPs for one (gate+up, down) pair over ``expert_rows`` rows."""

    gu = 2 * expert_rows * GU_OUT * HIDDEN
    dn = 2 * expert_rows * HIDDEN * MOE_INTERMEDIATE
    return gu + dn


def quantized_bytes(out_dims: int, in_dims: int, experts: int = NUM_EXPERTS) -> int:
    """q4/g32 packed weight + bf16 scales + bf16 biases for one bank."""

    elems = experts * out_dims * in_dims
    packed = elems * BITS // 8
    groups = elems // GROUP_SIZE
    return packed + 2 * groups * 2  # scales + biases, bf16


def ideal_weight_bytes() -> int:
    """Every expert's gate+up and down streamed exactly once."""

    return quantized_bytes(GU_OUT, HIDDEN) + quantized_bytes(HIDDEN, MOE_INTERMEDIATE)


def tile_runs(sorted_ids: np.ndarray, bm: int = KERNEL_BM) -> int:
    """Distinct-expert passes the kernel makes over ``sorted_ids``.

    One per (tile, run of equal ids inside that tile) -- i.e. exactly the
    ``while (n < tgp_bm)`` iterations summed over the y grid.  This is the
    number of *weight-tile streams and K loops*; ``runs / tiles`` is the
    multiplier B1 is trying to drive to 1.
    """

    ids = np.asarray(sorted_ids).reshape(-1)
    total = ids.size
    runs = 0
    for start in range(0, total, bm):
        tile = ids[start : start + bm]
        if tile.size == 0:
            continue
        runs += 1 + int(np.count_nonzero(tile[1:] != tile[:-1]))
    return runs


def tile_model(sorted_ids: np.ndarray) -> dict:
    """Issued-FLOP and weight-byte inflation implied by the tile walk."""

    ids = np.asarray(sorted_ids).reshape(-1)
    rows = int(ids.size)
    tiles = (rows + KERNEL_BM - 1) // KERNEL_BM
    runs = tile_runs(ids)
    factor = runs / tiles if tiles else 0.0
    return {
        "expert_rows": rows,
        "row_tiles": tiles,
        "runs": runs,
        "tile_reload_factor": factor,
        "issued_flops": int(round(flops_per_call(tiles * KERNEL_BM) * factor)),
        "weight_bytes_ideal": ideal_weight_bytes(),
        "weight_bytes_tiled": int(round(ideal_weight_bytes() * factor)),
    }


def _tokens_for(rows_per_expert: int) -> int:
    if rows_per_expert <= 0:
        raise ValueError("rows_per_expert must be positive")
    total = NUM_EXPERTS * rows_per_expert
    if total % TOP_K:
        raise ValueError(
            f"rows_per_expert={rows_per_expert} gives {total} expert-rows, "
            f"not a multiple of top_k={TOP_K}"
        )
    return total // TOP_K


def uniform_routing(rows_per_expert: int, seed: int) -> np.ndarray:
    """The default synthetic: a uniform router's top-10, ``[tokens, TOP_K]``.

    Ten *distinct* experts per token (top-k of i.i.d. scores), which is what
    the model's ``argpartition`` produces, and per-expert counts that are
    multinomial around ``rows_per_expert`` rather than exactly equal.  The
    census (J §2.4) says every expert is hit in every layer of every chunk,
    which this reproduces, and the jitter is what makes the tile-straddle rate
    the honest ``~1 + BM/R``.

    Exact balance (``--routing balanced``) is *not* the right default: it puts
    every expert boundary at a multiple of R, so at R = 160 and R = 320 (both
    multiples of BM = 32) every boundary lands on a tile edge and the reload
    factor collapses to exactly 1.0.  That is a real bound -- it is what a
    sorted, perfectly balanced router would buy -- but reading the curve off
    it would credit B1 with an alignment the production routing never has.
    """

    tokens = _tokens_for(rows_per_expert)
    rng = np.random.default_rng(seed)
    scores = rng.random((tokens, NUM_EXPERTS))
    top = np.argpartition(scores, NUM_EXPERTS - TOP_K, axis=-1)[:, -TOP_K:]
    return top.astype(np.int64)


def balanced_routing(rows_per_expert: int, seed: int) -> np.ndarray:
    """Exactly ``rows_per_expert`` rows per expert -- the aligned best case."""

    tokens = _tokens_for(rows_per_expert)
    ids = np.repeat(np.arange(NUM_EXPERTS, dtype=np.int64), rows_per_expert)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return ids.reshape(tokens, TOP_K)


def dirichlet_routing(
    rows_per_expert: int, seed: int, concentration: float
) -> np.ndarray:
    """Skewed routing at the same total row count (sensitivity check)."""

    tokens = _tokens_for(rows_per_expert)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(NUM_EXPERTS, concentration))
    ids = rng.choice(NUM_EXPERTS, size=tokens * TOP_K, p=weights).astype(np.int64)
    return ids.reshape(tokens, TOP_K)


def make_routing(
    rows_per_expert: int, seed: int, mode: str, concentration: float
) -> np.ndarray:
    if mode == "uniform":
        return uniform_routing(rows_per_expert, seed)
    if mode == "balanced":
        return balanced_routing(rows_per_expert, seed)
    if mode == "dirichlet":
        return dirichlet_routing(rows_per_expert, seed, concentration)
    raise ValueError(f"unknown routing mode {mode!r}")


def summarize_routing(ids: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(ids).reshape(-1), minlength=NUM_EXPERTS)
    touched = int(np.count_nonzero(counts))
    return {
        "experts_touched": touched,
        "rows_per_expert_mean": float(counts.mean()),
        "rows_per_expert_min": int(counts.min()),
        "rows_per_expert_max": int(counts.max()),
    }


# ---------------------------------------------------------------------------
# Device-side chain
# ---------------------------------------------------------------------------


def _quantized_bank(experts: int, out_dims: int, in_dims: int):
    scale = (1.0 / in_dims) ** 0.5
    w = mx.random.uniform(
        low=-scale, high=scale, shape=(experts, out_dims, in_dims)
    ).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(packed, scales, biases)
    del w
    return packed, scales, biases


def _sort_plan(ids):
    """``mlx_lm.models.switch_layers._gather_sort``'s index arithmetic."""

    flat = ids.flatten()
    order = mx.argsort(flat)
    inv_order = mx.argsort(order)
    return flat, order, inv_order


def run_chain(x, ids, gu, dn):
    """``_FusedGateUpSwitchGLU.__call__`` at prefill width, inlined.

    ``x`` is ``[tokens, HIDDEN]``; ``ids`` is ``[tokens, TOP_K]``.
    """

    import mlx.nn as nn

    tokens = x.shape[0]
    xr = mx.expand_dims(x.reshape(1, tokens, HIDDEN), (-2, -3))
    flat, order, inv_order = _sort_plan(ids)
    xs = xr.flatten(0, -3)[order // TOP_K]
    idx = flat[order]
    g = mx.gather_qmm(
        xs, gu[0], gu[1], gu[2], rhs_indices=idx, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    out = mx.gather_qmm(
        h, dn[0], dn[1], dn[2], rhs_indices=idx, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )
    return mx.unflatten(out[inv_order], 0, (tokens, TOP_K)).squeeze(-2)


def run_gemm(xs, idx, gu, dn):
    """The two grouped GEMMs on pre-sorted rows -- no sort, no unsort."""

    import mlx.nn as nn

    g = mx.gather_qmm(
        xs, gu[0], gu[1], gu[2], rhs_indices=idx, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )
    gate, up = mx.split(g, 2, axis=-1)
    h = nn.silu(gate) * up
    return mx.gather_qmm(
        h, dn[0], dn[1], dn[2], rhs_indices=idx, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
    )


def time_run(run, reps: int, warmup: int, lane: str, clear_cache: bool) -> dict:
    """Queued lane by default: ``reps`` graphs issued, one terminal eval.

    Per ``queued-vs-eager-metal-microbench`` the eager lane charges every call
    a host sync, which for a >100 us kernel is noise but for the sort chain's
    small ops is not.  The production site is a lazily-built 48-layer chunk
    graph evaluated once, i.e. the queued lane -- promote on that number.
    """

    for _ in range(warmup):
        mx.eval(run())
    samples, builds = [], []
    if lane == "queued":
        if clear_cache:
            mx.clear_cache()
        t0 = time.perf_counter()
        outs = [run() for _ in range(reps)]
        t1 = time.perf_counter()
        mx.eval(outs)
        total_ms = (time.perf_counter() - t1) * 1e3
        return {
            "lane": lane,
            "median_ms": total_ms / reps,
            "p10_ms": total_ms / reps,
            "p90_ms": total_ms / reps,
            "build_ms": (t1 - t0) * 1e3 / reps,
            "reps": reps,
        }
    for _ in range(reps):
        if clear_cache:
            mx.clear_cache()
        t0 = time.perf_counter()
        out = run()
        t1 = time.perf_counter()
        mx.eval(out)
        samples.append((time.perf_counter() - t1) * 1e3)
        builds.append((t1 - t0) * 1e3)
    samples.sort()
    return {
        "lane": lane,
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, int(0.10 * (len(samples) - 1)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.90 * (len(samples) - 1)))],
        "build_ms": statistics.median(builds),
        "reps": reps,
    }


# ---------------------------------------------------------------------------
# Exactness probe
# ---------------------------------------------------------------------------


def exactness_probe(gu, dn, probe_rows: int, filler_rows: int, seed: int) -> dict:
    """Same rows, two batch sizes, bitwise comparison after the unsort.

    Builds ``probe_rows`` tokens, runs them alone, then runs them as the head
    of a ``probe_rows + filler_rows`` batch and slices the probe back out.
    Anything but ``differing == 0`` falsifies B1's bit-exactness claim and the
    super-chunk becomes a rounding-class change needing an agreement screen.
    """

    mx.random.seed(seed)
    probe_x = mx.random.normal((probe_rows, HIDDEN)).astype(mx.bfloat16)
    fill_x = mx.random.normal((filler_rows, HIDDEN)).astype(mx.bfloat16)
    rng = np.random.default_rng(seed)
    probe_ids = rng.integers(
        0, NUM_EXPERTS, size=(probe_rows, TOP_K), dtype=np.int64
    )
    fill_ids = rng.integers(
        0, NUM_EXPERTS, size=(filler_rows, TOP_K), dtype=np.int64
    )
    probe_i = mx.array(probe_ids.astype(np.uint32))
    fill_i = mx.array(fill_ids.astype(np.uint32))
    mx.eval(probe_x, fill_x, probe_i, fill_i)

    small = run_chain(probe_x, probe_i, gu, dn)
    big = run_chain(
        mx.concatenate([probe_x, fill_x], axis=0),
        mx.concatenate([probe_i, fill_i], axis=0),
        gu,
        dn,
    )
    head = big[:probe_rows]
    mx.eval(small, head)
    diff = mx.abs(small.astype(mx.float32) - head.astype(mx.float32))
    return {
        "probe_rows": probe_rows,
        "batch_rows": probe_rows + filler_rows,
        "probe_expert_rows": probe_rows * TOP_K,
        "batch_expert_rows": (probe_rows + filler_rows) * TOP_K,
        "differing": int(mx.sum(small != head).item()),
        "total_elements": int(small.size),
        "max_abs_diff": float(mx.max(diff).item()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rows-per-expert",
        type=str,
        default=",".join(str(r) for r in DEFAULT_ROWS),
        help="comma-separated rows/expert settings (default 40,80,160,320)",
    )
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    p.add_argument("--lane", choices=("queued", "eager"), default="queued")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--routing",
        choices=("uniform", "balanced", "dirichlet"),
        default="uniform",
        help="synthetic routing law (default: a uniform router's top-10)",
    )
    p.add_argument(
        "--concentration",
        type=float,
        default=0.5,
        help="Dirichlet concentration when --routing dirichlet",
    )
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--no-exactness", dest="exactness", action="store_false")
    p.add_argument("--exactness-probe-rows", type=int, default=2048)
    p.add_argument("--exactness-filler-rows", type=int, default=14336)
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--allow-unlocked", action="store_true")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run the NumPy accounting checks; no MLX, no GPU, no lock",
    )
    return p


def self_test() -> int:
    """Pure-NumPy checks of the run counter and the byte accounting."""

    ids = np.repeat(np.arange(4), 32)  # 4 experts x 32 rows == 4 clean tiles
    assert tile_runs(ids) == 4, tile_runs(ids)
    ids = np.repeat(np.arange(8), 16)  # every 32-row tile holds 2 experts
    assert tile_runs(ids) == 8, tile_runs(ids)
    ids = np.repeat(np.arange(2), 40)  # 80 rows -> tiles [32,32,16], 1 split
    assert tile_runs(ids) == 4, tile_runs(ids)

    factors = []
    for rows in DEFAULT_ROWS:
        table = balanced_routing(rows, seed=1)
        counts = np.bincount(table.reshape(-1), minlength=NUM_EXPERTS)
        assert counts.min() == counts.max() == rows, (rows, counts.min())

        table = uniform_routing(rows, seed=1)
        assert table.shape == (NUM_EXPERTS * rows // TOP_K, TOP_K), table.shape
        # top-k of i.i.d. scores: TOP_K distinct experts per token, and with
        # >= 20,480 draws over 512 experts nothing is missed.
        assert all(len(set(r.tolist())) == TOP_K for r in table[:64])
        counts = np.bincount(table.reshape(-1), minlength=NUM_EXPERTS)
        assert counts.min() > 0, counts.min()

        model = tile_model(np.sort(table.reshape(-1)))
        assert model["expert_rows"] == NUM_EXPERTS * rows
        # The straddle rate is ~BM/R once boundaries are not tile-aligned:
        # ~1.8 passes/tile at R=40, ~1.1 at R=320.
        expected = 1.0 + KERNEL_BM / rows
        assert abs(model["tile_reload_factor"] - expected) < 0.06, (
            rows, model["tile_reload_factor"], expected
        )
        factors.append(model["tile_reload_factor"])
    assert factors == sorted(factors, reverse=True), factors

    # Exact balance IS the aligned bound the docstring claims: R a multiple of
    # BM puts every boundary on a tile edge.
    aligned = tile_model(np.sort(balanced_routing(320, seed=1).reshape(-1)))
    assert aligned["tile_reload_factor"] == 1.0, aligned["tile_reload_factor"]

    assert quantized_bytes(GU_OUT, HIDDEN) == (
        NUM_EXPERTS * GU_OUT * HIDDEN * BITS // 8
        + 2 * (NUM_EXPERTS * GU_OUT * HIDDEN // GROUP_SIZE) * 2
    )
    assert flops_per_call(10) == 2 * 10 * (GU_OUT * HIDDEN + HIDDEN * MOE_INTERMEDIATE)

    summary = summarize_routing(balanced_routing(40, seed=3))
    assert summary["experts_touched"] == NUM_EXPERTS
    print("[self-test] ok: tile runs, routing balance, byte and FLOP accounting")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    require_guard_window(args.allow_unlocked)
    print(
        "[micro-moe-prefill-rows] must run under /tmp/mtplx-gpu-exclusive.lock",
        flush=True,
    )
    _require_mlx()
    mx.random.seed(args.seed)

    rows_settings = [int(r) for r in args.rows_per_expert.split(",") if r.strip()]
    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; choose from {VARIANTS}")

    print(
        f"[build] {NUM_EXPERTS} experts, gate+up [{GU_OUT}, {HIDDEN}], "
        f"down [{HIDDEN}, {MOE_INTERMEDIATE}], q{BITS}/g{GROUP_SIZE}, "
        f"top_k={TOP_K}",
        flush=True,
    )
    gu = _quantized_bank(NUM_EXPERTS, GU_OUT, HIDDEN)
    dn = _quantized_bank(NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE)
    print(
        f"[build] bank {ideal_weight_bytes() / 1e9:.3f} GB "
        f"(gate+up {quantized_bytes(GU_OUT, HIDDEN) / 1e9:.3f} + "
        f"down {quantized_bytes(HIDDEN, MOE_INTERMEDIATE) / 1e9:.3f})",
        flush=True,
    )

    results = {}
    for rows in rows_settings:
        table = make_routing(rows, args.seed, args.routing, args.concentration)
        tokens = table.shape[0]
        model = tile_model(np.sort(table.reshape(-1)))
        model.update(summarize_routing(table))
        model["tokens"] = tokens

        x = mx.random.normal((tokens, HIDDEN)).astype(mx.bfloat16)
        ids = mx.array(table.astype(np.uint32))
        mx.eval(x, ids)

        # pre-sorted inputs for the `gemm` variant
        flat, order, _ = _sort_plan(ids)
        xs = mx.expand_dims(x.reshape(1, tokens, HIDDEN), (-2, -3)).flatten(0, -3)[
            order // TOP_K
        ]
        idx = flat[order]
        mx.eval(xs, idx)

        useful = flops_per_call(model["expert_rows"])
        per_rows = {"model": model}
        for name in names:
            run = (
                (lambda: run_chain(x, ids, gu, dn))
                if name == "chain"
                else (lambda: run_gemm(xs, idx, gu, dn))
            )
            stats = time_run(run, args.reps, args.warmup, args.lane, args.clear_cache)
            seconds = stats["median_ms"] / 1e3
            stats.update(
                {
                    "useful_tflops": useful / 1e12,
                    "useful_tf_s": useful / 1e12 / seconds,
                    "issued_tf_s": model["issued_flops"] / 1e12 / seconds,
                    "weight_gb_ideal": model["weight_bytes_ideal"] / 1e9,
                    "weight_gb_tiled": model["weight_bytes_tiled"] / 1e9,
                    "weight_gb_s_ideal": model["weight_bytes_ideal"] / 1e9 / seconds,
                    "weight_gb_s_tiled": model["weight_bytes_tiled"] / 1e9 / seconds,
                    "us_per_expert_row": stats["median_ms"] * 1e3
                    / model["expert_rows"],
                }
            )
            per_rows[name] = stats
        results[str(rows)] = per_rows
        del x, ids, xs, idx
        mx.clear_cache()

    exact = None
    if args.exactness:
        exact = exactness_probe(
            gu, dn, args.exactness_probe_rows, args.exactness_filler_rows, args.seed
        )

    # ---- report ------------------------------------------------------------
    print(
        f"\nlane={args.lane} reps={args.reps} seed={args.seed} "
        f"routing={args.routing}"
    )
    hdr = (
        f"{'R':>5}{'tokens':>8}{'exp-rows':>10}{'runs/tile':>11}"
        f"{'variant':>9}{'ms':>10}{'TF/s':>9}{'issTF/s':>9}"
        f"{'GB/s(w)':>10}{'us/row':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for rows in rows_settings:
        block = results[str(rows)]
        model = block["model"]
        for name in names:
            s = block[name]
            print(
                f"{rows:>5}{model['tokens']:>8}{model['expert_rows']:>10}"
                f"{model['tile_reload_factor']:>11.3f}{name:>9}"
                f"{s['median_ms']:>10.3f}{s['useful_tf_s']:>9.2f}"
                f"{s['issued_tf_s']:>9.2f}{s['weight_gb_s_ideal']:>10.1f}"
                f"{s['us_per_expert_row']:>9.4f}"
            )

    verdict = {}
    # The census family is the pair of grouped GEMMs, so ``gemm`` carries the
    # call; ``chain`` is reported the same way but is not the decision.
    decisive = "gemm" if "gemm" in names else names[0]
    for name in names:
        lo = results[str(rows_settings[0])][name]["useful_tf_s"]
        hi = results[str(rows_settings[-1])][name]["useful_tf_s"]
        ratio = hi / lo if lo else 0.0
        call = (
            "schedule-bound (B1 GO)"
            if ratio >= 1.5
            else "partial" if ratio >= 1.15 else "kernel-bound (B1 DEAD)"
        )
        verdict[name] = {
            "decisive": name == decisive,
            "low_rows": rows_settings[0],
            "high_rows": rows_settings[-1],
            "tf_s_low": lo,
            "tf_s_high": hi,
            "ratio": ratio,
            "call": call,
        }
        mark = "" if name == decisive else "  (context only)"
        print(
            f"\n[verdict:{name}] R={rows_settings[0]} {lo:.2f} TF/s -> "
            f"R={rows_settings[-1]} {hi:.2f} TF/s = {ratio:.2f}x  -> {call}"
            f"{mark}"
        )
        print(
            "  census reference: routed grouped GEMM 23.2 TF/s vs 51.4 dense q4"
        )

    if exact is not None:
        print(
            f"\n[exactness] {exact['probe_rows']} probe rows alone vs as the head "
            f"of {exact['batch_rows']}: differing={exact['differing']}"
            f"/{exact['total_elements']} max_abs_diff={exact['max_abs_diff']:.6g}"
        )
        if exact["differing"]:
            print("  ** NOT bit-exact: a super-chunk is a rounding-class change **")

    summary = {
        "geometry": {
            "hidden": HIDDEN,
            "moe_intermediate": MOE_INTERMEDIATE,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "kernel_bm": KERNEL_BM,
            "kernel_bn": KERNEL_BN,
        },
        "lane": args.lane,
        "reps": args.reps,
        "seed": args.seed,
        "routing": args.routing,
        "concentration": args.concentration,
        "rows_per_expert": rows_settings,
        "results": results,
        "verdict": verdict,
        "exactness": exact,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"\n[out] {args.json}")
    else:
        print("\n" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
