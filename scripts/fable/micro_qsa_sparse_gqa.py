#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Parity + timing for the native direct-index sparse-GQA QSA attention (B3).

The kernel under test is ``mtplx.native.qsa_sparse_gqa`` -- MTPLX's port of
oMLX's Steel-MMA ``qwen4_qsa_sparse_gqa`` (Jonathan Spangler, jundot/omlx
7467dce8, Apache-2.0).  One threadgroup owns one (query row, KV head); the
twelve query heads sharing a KV head are a padded 16-row MMA tile; the
chronological block ids are walked directly out of the full KV cache backing;
the zero-to-three-token causal tail is generated in-kernel.  No score tensor,
no gathered K/V, no ``[S, T]`` bool mask.

Nothing in the model calls it yet.  This bench is the phase-1 falsifier the
program note asks for (M §B3: *"standalone parity + timing at 4096 rows x
T in {16K, 32K, 64K}, queued lane"*), and the same kernel at M=4 / M=1 is
K-Q2 / K-D6, so those cells are here too.

WHAT IS COMPARED
----------------
Selection is NOT re-implemented.  Every cell runs the production selector,
``QSAIndexer._select_eager``, imported from ``mtplx.models.qwen4_exp``, over a
synthetic pooled bank, and hands its ``("flash_prefill", block_ids,
block_valid)`` output to every arm.  So all arms see one visible set and the
only thing that can differ is arithmetic.

Arms (all fed the same block ids):
  native_bk{BK}_dc{DC}   the ported kernel, one per instantiated tile pair
  flash                  mtplx.kernels.qsa_prefill_flash -- the shipped Metal 4
                         NAX consumer of the same contract (skipped where
                         unsupported: it needs TensorOps, S > 1, and the exact
                         production 1/sqrt(256) scale)
  gather                 _qsa_prefill_gather_attention -- the portable bounded
                         tier, the honest baseline on a non-NAX machine
  dense                  _qsa_blocks_to_dense_mask + _qsa_dense_attention --
                         the production dense lane and the numerics reference

VISIBLE-SET IDENTITY (asserted, not assumed)
--------------------------------------------
The kernel does not read ``block_valid``.  It derives validity in-kernel as
"the first ``min(512, (pos + 1) // 4)`` slots of the row are valid", which is
identical to the selector's output BY CONSTRUCTION: ``_select_eager`` sorts the
raw top-k ascending, and validity there is the threshold predicate
``id < complete_blocks``, so the valid entries are exactly a leading prefix.
That is an invariant of the selector, not of the kernel, so this bench
*asserts* it on every row of every cell (``--strict-identity``, default on)
before it trusts a single parity number:

  A1 popcount(block_valid[s]) == min(512, (pos_start + s + 1) // 4)
  A2 block_valid[s] is a prefix mask (no valid slot after an invalid one)
  A3 block_ids[s, :n] is strictly ascending
  A4 0 <= block_ids[s, :n] < (pos_start + s + 1) // 4

and then, on a sample of rows, materialises both token lists in full -- the
lane's (blocks + tail) expansion and the kernel's -- and compares them
element-for-element.  A1-A4 holding makes the two expansions the same set;
the sample makes that concrete rather than merely argued.

TOLERANCE
---------
This is a ROUNDING-CLASS change, not a bit-exact one, and no arm here is the
"true" answer:

  dense   bf16 Q@K^T accumulated by MLX's SDPA fallback into an fp32 [H, S, T]
          score tensor, precise (max-subtracting) softmax over the whole row,
          then a bf16-operand P@V over T keys.
  native  fp32 online softmax with base-2 exponentials (``fast::exp2``, and
          the scale folded as ``scale * log2(e)``), rescaled once per BK-wide
          tile, fp32 P@V accumulated over ~2,051 keys in tile order.

So the expected delta is the accumulated difference of two fp32 reduction
orders over the same visible set, rendered back to bf16.  The bar this bench
holds the kernel to, per cell:

  max |native - dense| <= 2 bf16 ULP at the observed output magnitude
  (bf16 has 8 significand bits, so 1 ULP ~ 2^-8 of the value's binade)
  AND the count of elements differing by more than 1 ULP is < 0.5% of them
  AND max |native - flash| <= max |dense - flash|  (the ported kernel is no
      further from dense than the kernel already shipping is)

Anything worse is a bug, not a rounding class, and must be root-caused before
the lane is wired.  ``--fail-on-tolerance`` (default on) makes that a non-zero
exit rather than a line of output.

RUN IT (one guarded window; the default sweep is minutes of GPU)
----------------------------------------------------------------
    PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w50-qsa-sparse-gqa \\
    /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \\
      /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \\
      --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \\
      --lock-timeout-seconds 3600 \\
      --child-timeout-seconds 3600 \\
      -- \\
      /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \\
      /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w50-qsa-sparse-gqa/scripts/fable/micro_qsa_sparse_gqa.py \\
        --out /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/micro-qsa-sparse-gqa.json

Peak is bounded by the 64K prefill cell: ~134 MB K/V + ~50 MB Q + ~50 MB per
live output, plus the dense arm's tiled score transient (``--dense-query-tile``
rows x T x 24 x 4 B; 512 x 64K = 3.2 GB).  ``--cells`` trims it; the script
prints its own estimate and refuses a cell over ``--max-transient-gb``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# The selector's large-prefill branch is what emits the block-id contract, so
# it has to be armed before mtplx.models.qwen4_exp is imported (several of its
# gates are lru_cached on first read).  MIN_ROWS=2 lets the M=4 verify cell
# take the same branch as a 4,096-row prefill chunk; the crossover stays at the
# real 2,048-token boundary.
os.environ.setdefault("MTPLX_QSA_PREFILL", "1")
os.environ.setdefault("MTPLX_QSA_PREFILL_MIN_ROWS", "2")
os.environ.setdefault("MTPLX_QSA_PREFILL_MIN_CONTEXT", "2049")
os.environ.setdefault("MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT", "2049")

# --- Qwen3.8 Flash-Next QSA geometry (TextArgs defaults) -------------------
Q_HEADS = 24
KV_HEADS = 2
GQA = 12
HEAD_DIM = 256
IDX_HEADS = 4
IDX_HEAD_DIM = 128
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
SELECTED_TOKENS = BLOCK_TOPK * COMPRESS_RATIO + (COMPRESS_RATIO - 1)  # 2051
SCALE = HEAD_DIM ** -0.5  # 0.0625, exactly representable

TILES = ((128, 32), (256, 32), (64, 64), (128, 64))

#: rows x context per cell.  Prefill cells are the program note's falsifier
#: shapes; the decode cells are K-Q2 (M=4 verify) and K-D6 (M=1 draft).
CELLS = {
    "prefill-16k": (4096, 16_384),
    "prefill-32k": (4096, 32_768),
    "prefill-64k": (4096, 65_536),
    "decode-m4-16k": (4, 16_384),
    "decode-m1-16k": (1, 16_384),
}

mx = None
np = None
_mods: dict = {}


def _require_mlx() -> None:
    """Import MLX and mtplx lazily: everything below runs on the GPU."""

    global mx, np
    import mlx.core
    import numpy

    mx = mlx.core
    np = numpy

    import mtplx.models.qwen4_exp as qwen4_exp
    import mtplx.native as native
    from mtplx.attention_context import attention_phase
    from mtplx.fable_prefill_chunk import QUERY_TILE_ENV
    from mtplx.kernels import qsa_prefill_flash

    _mods.update(
        qwen4_exp=qwen4_exp,
        native=native,
        flash=qsa_prefill_flash,
        attention_phase=attention_phase,
        query_tile_env=QUERY_TILE_ENV,
    )


# --------------------------------------------------------------------------
# cell construction: synthetic tensors, PRODUCTION selection
# --------------------------------------------------------------------------
def build_indexer(rope_theta: float):
    qwen4_exp = _mods["qwen4_exp"]
    text_args = qwen4_exp.TextArgs(rope_theta=rope_theta)
    indexer = qwen4_exp.QSAIndexer(text_args)
    indexer.update(
        {
            "q_layernorm": {
                "weight": mx.random.uniform(
                    low=0.5, high=1.5, shape=(IDX_HEAD_DIM,)
                ).astype(mx.bfloat16)
            },
            "k_layernorm": {
                "weight": mx.random.uniform(
                    low=0.5, high=1.5, shape=(IDX_HEAD_DIM,)
                ).astype(mx.bfloat16)
            },
        }
    )
    return indexer


def select_blocks(indexer, cache, pooled, rows: int, pos_start: int, total: int):
    """Run the production selector and return (block_ids, block_valid).

    Selection is strictly per-row -- scores depend only on that row's indexer
    query and the pooled table, and the validity mask only on that row's
    absolute position -- so the M=1 cell is served by scoring TWO rows at
    ``pos_start - 1`` and keeping the last.  That keeps the expression the
    production one instead of copying the two lines that build ``block_ids``
    out of ``top_idx`` (``_select_eager`` reaches its ``flash_prefill`` branch
    only for ``S > 1``).
    """

    score_rows = max(2, rows)
    score_start = pos_start - (score_rows - rows)
    if score_start < 0:
        raise SystemExit("cell needs at least two scoreable query positions")
    q_idx = mx.random.normal((1, score_rows, IDX_HEADS, IDX_HEAD_DIM)).astype(
        mx.bfloat16
    )
    with _mods["attention_phase"]("prefill"):
        selection = indexer._select_eager(q_idx, score_start, cache, pooled, total)
    if not (isinstance(selection, tuple) and selection[0] == "flash_prefill"):
        raise SystemExit(
            "the selector did not take its flash_prefill branch; "
            f"got {selection[0] if isinstance(selection, tuple) else type(selection)}"
        )
    _, block_ids, block_valid = selection
    return block_ids[-rows:], block_valid[-rows:]


def build_cell(name: str, rows: int, total: int, seed: int, rope_theta: float):
    """Synthetic KV backing + production selection for one cell."""

    mx.random.seed(seed)
    nb_total = total // COMPRESS_RATIO
    pos_start = total - rows

    # Q in the layout Attention actually produces: [1, S, H, D] projected,
    # then transposed to [1, H, S, D].  Keeping the transpose (rather than
    # allocating [1, H, S, D] directly) is the point -- the kernel has to read
    # a strided view without a hidden contiguous copy.
    q_rows = mx.random.normal((1, rows, Q_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    queries = q_rows.transpose(0, 2, 1, 3)
    # K/V are the FULL cache backing, never a :total slice.
    keys = mx.random.normal((1, KV_HEADS, total, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, KV_HEADS, total, HEAD_DIM)).astype(mx.bfloat16)

    indexer = build_indexer(rope_theta)
    cache = _mods["qwen4_exp"].QSACache(COMPRESS_RATIO)
    pooled = mx.random.normal((1, nb_total, IDX_HEAD_DIM)).astype(mx.bfloat16)
    cache.write_pooled(pooled, 0, nb_total)
    mx.eval(queries, keys, values, cache.pooled, cache.pooled_f32_t)

    block_ids, block_valid = select_blocks(
        indexer, cache, pooled, rows, pos_start, total
    )
    block_ids = mx.contiguous(block_ids.astype(mx.int32))
    block_valid = mx.contiguous(block_valid)
    mx.eval(block_ids, block_valid)
    return {
        "name": name,
        "rows": rows,
        "total": total,
        "pos_start": pos_start,
        "nb_total": nb_total,
        "queries": queries,
        "keys": keys,
        "values": values,
        "block_ids": block_ids,
        "block_valid": block_valid,
    }


# --------------------------------------------------------------------------
# visible-set identity: the kernel's in-kernel validity rule vs the selector's
# --------------------------------------------------------------------------
def _kernel_tokens(ids_row, pos: int) -> "np.ndarray":
    """Exactly what the kernel's staging loop admits for one query row."""

    complete = (pos + 1) // COMPRESS_RATIO
    valid_blocks = min(BLOCK_TOPK, complete)
    tokens = []
    for slot in range(SELECTED_TOKENS):
        if slot < BLOCK_TOPK * COMPRESS_RATIO:
            block_slot = slot // COMPRESS_RATIO
            if block_slot >= valid_blocks:
                continue
            candidate = int(ids_row[block_slot]) * COMPRESS_RATIO + (
                slot % COMPRESS_RATIO
            )
        else:
            candidate = complete * COMPRESS_RATIO + (
                slot - BLOCK_TOPK * COMPRESS_RATIO
            )
        if 0 <= candidate <= pos:
            tokens.append(candidate)
    return np.asarray(tokens, dtype=np.int64)


def _lane_tokens(ids_row, ok_row, pos: int) -> "np.ndarray":
    """What the lane's own expansion admits, from block_ids + block_valid."""

    complete = (pos + 1) // COMPRESS_RATIO
    tokens = []
    for slot in range(BLOCK_TOPK):
        if not bool(ok_row[slot]):
            continue
        base = int(ids_row[slot]) * COMPRESS_RATIO
        for r in range(COMPRESS_RATIO):
            if base + r <= pos:
                tokens.append(base + r)
    for tok in range(complete * COMPRESS_RATIO, pos + 1):
        tokens.append(tok)
    return np.asarray(tokens, dtype=np.int64)


def check_identity(cell: dict, sample_rows: int, strict: bool) -> dict:
    """Assert the selector invariant the kernel's validity rule depends on."""

    # Convert without a dtype argument first: MLX arrays implement
    # __array__, but the numpy conversion is cheaper and better defined
    # when the cast happens on the numpy side.
    ids = np.asarray(cell["block_ids"]).astype(np.int64)
    ok = np.asarray(cell["block_valid"]).astype(bool)
    rows, pos_start = cell["rows"], cell["pos_start"]
    pos = pos_start + np.arange(rows, dtype=np.int64)
    complete = (pos + 1) // COMPRESS_RATIO
    expected_n = np.minimum(BLOCK_TOPK, complete)

    failures = []
    counts = ok.sum(axis=1)
    if not np.array_equal(counts, expected_n):
        bad = int(np.argmax(counts != expected_n))
        failures.append(
            f"A1 valid count: row {bad} has {int(counts[bad])}, "
            f"expected {int(expected_n[bad])}"
        )
    # A2: prefix mask -- no True may follow a False.
    prefix = np.arange(BLOCK_TOPK)[None, :] < counts[:, None]
    if not np.array_equal(ok, prefix):
        bad = int(np.argmax((ok != prefix).any(axis=1)))
        failures.append(f"A2 validity is not a prefix mask: row {bad}")
    for r in range(rows):
        n = int(counts[r])
        if n == 0:
            continue
        head = ids[r, :n]
        if n > 1 and not bool((np.diff(head) > 0).all()):
            failures.append(f"A3 block ids not strictly ascending: row {r}")
            break
        if head[0] < 0 or head[-1] >= complete[r]:
            failures.append(
                f"A4 block id out of the row's complete range: row {r}"
            )
            break

    # Concrete token-list identity on a deterministic row sample.
    if sample_rows > 0 and rows > 0:
        step = max(1, rows // sample_rows)
        sample = sorted(
            set(list(range(0, rows, step))[:sample_rows]) | {0, rows - 1}
        )
        for r in sample:
            lane = np.unique(_lane_tokens(ids[r], ok[r], int(pos[r])))
            kern = np.unique(_kernel_tokens(ids[r], int(pos[r])))
            if not np.array_equal(lane, kern):
                failures.append(
                    f"token sets differ at row {r}: "
                    f"|lane|={lane.size} |kernel|={kern.size} "
                    f"|sym-diff|={np.setxor1d(lane, kern).size}"
                )
                break
        sampled = len(sample)
    else:
        sampled = 0

    result = {
        "rows_checked": rows,
        "rows_token_sampled": sampled,
        "visible_tokens_first_row": int(expected_n[0]) * COMPRESS_RATIO,
        "failures": failures,
    }
    if failures and strict:
        for line in failures:
            print(f"  [identity] FAIL {line}", flush=True)
        raise SystemExit(
            "visible-set identity failed: the kernel's in-kernel validity rule "
            "does not match the selector's output, so no parity number below "
            "would mean anything"
        )
    return result


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def native_arm(cell: dict, key_tile: int, dim_tile: int):
    native = _mods["native"]

    def call():
        return native.qsa_sparse_gqa(
            cell["queries"],
            cell["keys"],
            cell["values"],
            cell["block_ids"],
            pos_start=cell["pos_start"],
            total_tokens=cell["total"],
            scale=SCALE,
            key_tile=key_tile,
            dimension_tile=dim_tile,
        )

    return call


def flash_arm(cell: dict):
    flash = _mods["flash"]
    if not flash.qsa_prefill_flash_supported(
        cell["queries"],
        cell["keys"],
        cell["values"],
        cell["block_ids"],
        cell["block_valid"],
        pos_start=cell["pos_start"],
        total_tokens=cell["total"],
        scale=SCALE,
    ):
        return None

    def call():
        return flash.qsa_prefill_flash(
            cell["queries"],
            cell["keys"],
            cell["values"],
            cell["block_ids"],
            cell["block_valid"],
            pos_start=cell["pos_start"],
            total_tokens=cell["total"],
            scale=SCALE,
        )

    return call


def gather_arm(cell: dict, tile_rows: int):
    qwen4_exp = _mods["qwen4_exp"]

    def call():
        return qwen4_exp._qsa_prefill_gather_attention(
            cell["queries"],
            cell["keys"],
            cell["values"],
            cell["block_ids"],
            cell["block_valid"],
            pos_start=cell["pos_start"],
            total_tokens=cell["total"],
            compress_ratio=COMPRESS_RATIO,
            scale=SCALE,
            tile_rows=tile_rows,
        )

    return call


def dense_arm(cell: dict, row_slice: tuple[int, int] | None = None):
    """The production dense lane: reconstruct the mask, run masked SDPA.

    ``row_slice`` restricts it to a contiguous window of query rows, which is
    how parity is taken at 4,096 x 64K without materialising a 25 GB score
    tensor.  Rows are independent under attention, so a window's output is the
    full lane's output for those rows.
    """

    qwen4_exp = _mods["qwen4_exp"]
    if row_slice is None:
        r0, r1 = 0, cell["rows"]
    else:
        r0, r1 = row_slice

    def call():
        mask = qwen4_exp._qsa_blocks_to_dense_mask(
            cell["block_ids"][r0:r1],
            cell["block_valid"][r0:r1],
            pos_start=cell["pos_start"] + r0,
            total_tokens=cell["total"],
            compress_ratio=COMPRESS_RATIO,
        )
        return qwen4_exp._qsa_dense_attention(
            cell["queries"][:, :, r0:r1],
            cell["keys"][:, :, : cell["total"]],
            cell["values"][:, :, : cell["total"]],
            mask=mask,
            scale=SCALE,
        )

    return call


# --------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------
def bf16_ulp(reference):
    """One bf16 ULP at each element's own magnitude (8 significand bits)."""

    mag = np.abs(reference.astype(np.float32))
    exponent = np.where(mag > 0, np.floor(np.log2(np.maximum(mag, 1e-30))), -126.0)
    return np.exp2(exponent - 8.0).astype(np.float32)


def compare(candidate, reference) -> dict:
    a = np.asarray(candidate.astype(mx.float32))
    b = np.asarray(reference.astype(mx.float32))
    diff = np.abs(a - b)
    ulp = bf16_ulp(b)
    ratio = diff / np.maximum(ulp, 1e-30)
    return {
        "max_abs": float(diff.max()),
        "max_ulp": float(ratio.max()),
        "mean_ulp": float(ratio.mean()),
        "frac_over_1ulp": float((ratio > 1.0).mean()),
        "elements": int(diff.size),
    }


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def time_arm(call, reps: int, warmup: int, queue_depth: int) -> dict:
    """Both lanes; the QUEUED number is the verdict.

    ``synced_ms`` evals every rep on its own, so each rep pays a full host
    round trip -- that lane can invert a verdict for short kernels (see the
    queued-vs-eager microbench note).  ``queued_ms`` builds ``queue_depth``
    independent outputs, submits them, and takes one synchronize, so the host
    cost is amortised the way it is inside a real prefill chunk.
    """

    for _ in range(warmup):
        mx.eval(call())
    mx.synchronize()

    synced = []
    for _ in range(reps):
        out = call()
        t0 = time.perf_counter()
        mx.eval(out)
        mx.synchronize()
        synced.append((time.perf_counter() - t0) * 1e3)

    queued = []
    batches = max(1, reps // queue_depth)
    for _ in range(batches):
        outs = [call() for _ in range(queue_depth)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        mx.synchronize()
        queued.append((time.perf_counter() - t0) * 1e3 / queue_depth)
        del outs

    synced.sort()
    return {
        "queued_ms": statistics.median(queued),
        "queued_min_ms": min(queued),
        "synced_ms": statistics.median(synced),
        "synced_p10_ms": synced[max(0, int(0.10 * (len(synced) - 1)))],
        "synced_p90_ms": synced[min(len(synced) - 1, int(0.90 * (len(synced) - 1)))],
        "reps": reps,
        "queue_depth": queue_depth,
    }


# --------------------------------------------------------------------------
# transient estimate (the box has been panicked by unbudgeted benches before)
# --------------------------------------------------------------------------
def transient_gb(cell: dict, dense_tile: int) -> float:
    rows, total = cell["rows"], cell["total"]
    kv = 2 * KV_HEADS * total * HEAD_DIM * 2
    q = Q_HEADS * rows * HEAD_DIM * 2
    ids = rows * BLOCK_TOPK * 5
    mask = rows * total  # bool
    scores = Q_HEADS * min(rows, dense_tile) * total * 4
    gathered = min(rows, 64) * SELECTED_TOKENS * KV_HEADS * HEAD_DIM * 2 * 2
    return (kv + 4 * q + ids + 2 * mask + 2 * scores + gathered) / 1e9


# --------------------------------------------------------------------------
def parity_windows(rows: int, width: int, count: int) -> list[tuple[int, int]]:
    """Contiguous row windows: the first, the last, and evenly spaced ones.

    The first window matters most -- those are the rows whose complete-block
    prefix is shortest, where the kernel's leading all-invalid tiles and the
    true -INFINITY masking are exercised.
    """

    if rows <= width:
        return [(0, rows)]
    starts = {0, rows - width}
    for i in range(1, max(1, count - 1)):
        starts.add(min(rows - width, (rows * i) // max(1, count - 1)))
    return [(s, min(rows, s + width)) for s in sorted(starts)]


def run_cell(cell: dict, args) -> dict:
    print(
        f"\n[cell] {cell['name']}: rows={cell['rows']} T={cell['total']} "
        f"pos_start={cell['pos_start']} blocks={cell['nb_total']} "
        f"visible<={SELECTED_TOKENS} tokens/row  "
        f"est_transient={transient_gb(cell, args.dense_query_tile):.2f} GB",
        flush=True,
    )
    report: dict = {
        "rows": cell["rows"],
        "total": cell["total"],
        "pos_start": cell["pos_start"],
    }

    identity = check_identity(
        cell, args.identity_sample_rows, not args.no_strict_identity
    )
    report["identity"] = identity
    # Never print OK over a failure list: with --no-strict-identity the check
    # is downgraded to a warning, and a receipt that still said OK would be
    # the most misleading line in the file.
    if identity["failures"]:
        print(
            f"  [identity] FAILED on {identity['rows_checked']} rows "
            f"(--no-strict-identity): {identity['failures'][0]}",
            flush=True,
        )
    else:
        print(
            f"  [identity] OK on {identity['rows_checked']} rows "
            f"({identity['rows_token_sampled']} token-sampled)",
            flush=True,
        )

    # ---- parity, on row windows ----
    windows = parity_windows(cell["rows"], args.parity_window, args.parity_windows)
    flash_call = flash_arm(cell)
    arms = [(k, d) for (k, d) in TILES if (k, d) in args.tiles]
    agg = {
        f"bk{k}_dc{d}": {"max_abs": 0.0, "max_ulp": 0.0, "frac_over_1ulp": 0.0}
        for k, d in arms
    }
    agg_flash = {f"bk{k}_dc{d}": {"max_abs": 0.0, "max_ulp": 0.0} for k, d in arms}
    dense_flash = {"max_abs": 0.0, "max_ulp": 0.0}
    rows_seen = 0
    for r0, r1 in windows:
        # One reference per window, shared by every tile arm: the dense lane
        # is by far the most expensive thing in this loop.
        ref = dense_arm(cell, (r0, r1))()
        mx.eval(ref)
        flash_slice = None
        if flash_call is not None:
            flash_slice = flash_call()[:, :, r0:r1]
            mx.eval(flash_slice)
            stats = compare(ref, flash_slice)
            dense_flash["max_abs"] = max(dense_flash["max_abs"], stats["max_abs"])
            dense_flash["max_ulp"] = max(dense_flash["max_ulp"], stats["max_ulp"])
        for key_tile, dim_tile in arms:
            arm = f"bk{key_tile}_dc{dim_tile}"
            got = native_arm(cell, key_tile, dim_tile)()[:, :, r0:r1]
            mx.eval(got)
            stats = compare(got, ref)
            agg[arm]["max_abs"] = max(agg[arm]["max_abs"], stats["max_abs"])
            agg[arm]["max_ulp"] = max(agg[arm]["max_ulp"], stats["max_ulp"])
            agg[arm]["frac_over_1ulp"] = max(
                agg[arm]["frac_over_1ulp"], stats["frac_over_1ulp"]
            )
            if flash_slice is not None:
                nf = compare(got, flash_slice)
                agg_flash[arm]["max_abs"] = max(
                    agg_flash[arm]["max_abs"], nf["max_abs"]
                )
                agg_flash[arm]["max_ulp"] = max(
                    agg_flash[arm]["max_ulp"], nf["max_ulp"]
                )
            del got
        rows_seen += r1 - r0
        del ref, flash_slice
        mx.clear_cache()

    parity: dict = {}
    for key_tile, dim_tile in arms:
        arm = f"bk{key_tile}_dc{dim_tile}"
        entry = {
            "rows_compared": rows_seen,
            "windows": [list(w) for w in windows],
            "vs_dense": agg[arm],
        }
        if flash_call is not None:
            entry["vs_flash"] = agg_flash[arm]
            entry["dense_vs_flash"] = dense_flash
        mine, mine_flash = agg[arm], agg_flash[arm]
        verdict = []
        if mine["max_ulp"] > args.max_ulp:
            verdict.append(f"max_ulp {mine['max_ulp']:.2f} > {args.max_ulp}")
        if mine["frac_over_1ulp"] > args.max_frac_over_1ulp:
            verdict.append(
                f"frac_over_1ulp {mine['frac_over_1ulp']:.4f} > "
                f"{args.max_frac_over_1ulp}"
            )
        if flash_call is not None and mine_flash["max_abs"] > dense_flash["max_abs"]:
            verdict.append(
                f"further from dense than the shipped flash kernel is "
                f"({mine_flash['max_abs']:.6f} > {dense_flash['max_abs']:.6f})"
            )
        entry["tolerance_failures"] = verdict
        parity[arm] = entry
        flag = "FAIL" if verdict else "ok"
        line = (
            f"  [parity] {arm}: vs dense max_abs={mine['max_abs']:.6f} "
            f"max_ulp={mine['max_ulp']:.2f} "
            f"over1ulp={mine['frac_over_1ulp'] * 100:.3f}%"
        )
        if flash_call is not None:
            line += (
                f" | vs flash max_abs={mine_flash['max_abs']:.6f}"
                f" (dense vs flash {dense_flash['max_abs']:.6f})"
            )
        print(f"{line}  [{flag}]", flush=True)
    report["parity"] = parity

    # ---- timing, full shape ----
    timings: dict = {}
    for key_tile, dim_tile in TILES:
        if (key_tile, dim_tile) not in args.tiles:
            continue
        timings[f"native_bk{key_tile}_dc{dim_tile}"] = time_arm(
            native_arm(cell, key_tile, dim_tile),
            args.reps,
            args.warmup,
            args.queue_depth,
        )
        mx.clear_cache()
    if flash_call is not None:
        timings["flash"] = time_arm(
            flash_call, args.reps, args.warmup, args.queue_depth
        )
        mx.clear_cache()
    else:
        timings["flash"] = {"skipped": "qsa_prefill_flash does not support this cell"}
    if not args.no_gather:
        timings["gather"] = time_arm(
            gather_arm(cell, args.gather_tile),
            args.reps,
            args.warmup,
            args.queue_depth,
        )
        mx.clear_cache()
    if not args.no_dense_timing:
        # The production dense lane needs its query tile armed, or a
        # 4,096-row x 64K cell materialises a 25 GB score tensor.  Take
        # the env NAME from the module that owns it, never a literal.
        query_tile_env = _mods["query_tile_env"]
        os.environ[query_tile_env] = str(args.dense_query_tile)
        try:
            timings["dense"] = time_arm(
                dense_arm(cell), args.reps, args.warmup, max(1, args.queue_depth // 4)
            )
            timings["dense"]["query_tile_rows"] = args.dense_query_tile
        finally:
            os.environ.pop(query_tile_env, None)
        mx.clear_cache()
    report["timing"] = timings

    flash_ms = timings.get("flash", {}).get("queued_ms")
    for name, entry in timings.items():
        if "queued_ms" not in entry:
            print(f"  [time] {name}: skipped ({entry.get('skipped')})", flush=True)
            continue
        speedup = ""
        if flash_ms is not None and name != "flash":
            speedup = f"  ({flash_ms / entry['queued_ms']:.2f}x flash)"
        print(
            f"  [time] {name}: queued={entry['queued_ms']:.3f} ms "
            f"synced={entry['synced_ms']:.3f} ms{speedup}",
            flush=True,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=str, default=",".join(CELLS))
    p.add_argument(
        "--tiles",
        type=str,
        default="128:32,64:64",
        help="comma list of key_tile:dimension_tile from "
        + ", ".join(f"{a}:{b}" for a, b in TILES),
    )
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--queue-depth", type=int, default=4)
    p.add_argument("--parity-window", type=int, default=16)
    p.add_argument("--parity-windows", type=int, default=4)
    p.add_argument("--identity-sample-rows", type=int, default=16)
    p.add_argument("--no-strict-identity", action="store_true")
    p.add_argument("--max-ulp", type=float, default=2.0)
    p.add_argument("--max-frac-over-1ulp", type=float, default=0.005)
    p.add_argument("--no-fail-on-tolerance", action="store_true")
    p.add_argument("--dense-query-tile", type=int, default=512)
    p.add_argument("--gather-tile", type=int, default=64)
    p.add_argument("--no-gather", action="store_true")
    p.add_argument("--no-dense-timing", action="store_true")
    p.add_argument("--max-transient-gb", type=float, default=12.0)
    p.add_argument("--rope-theta", type=float, default=10_000_000.0)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    names = [c.strip() for c in args.cells.split(",") if c.strip()]
    for name in names:
        if name not in CELLS:
            raise SystemExit(f"unknown cell {name!r}; expected {list(CELLS)}")
    tiles = []
    for item in args.tiles.split(","):
        item = item.strip()
        if not item:
            continue
        a, _, b = item.partition(":")
        pair = (int(a), int(b))
        if pair not in TILES:
            raise SystemExit(f"tile {item!r} is not instantiated; have {TILES}")
        tiles.append(pair)
    args.tiles = tuple(tiles)

    print("[micro-qsa-sparse-gqa] must run under /tmp/mtplx-gpu-exclusive.lock",
          flush=True)
    _require_mlx()
    native = _mods["native"]
    if not native.native_qsa_available():
        raise SystemExit(
            "the native QSA extension is not built; see mtplx/native/__init__.py "
            "for the CPU-only cmake build"
        )

    results: dict = {
        "geometry": {
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "gqa": GQA,
            "head_dim": HEAD_DIM,
            "compress_ratio": COMPRESS_RATIO,
            "block_topk": BLOCK_TOPK,
            "selected_tokens": SELECTED_TOKENS,
            "scale": SCALE,
        },
        "args": {k: v for k, v in vars(args).items() if k != "out"},
        "cells": {},
    }
    failures = []
    for name in names:
        rows, total = CELLS[name]
        cell = build_cell(name, rows, total, args.seed, args.rope_theta)
        estimate = transient_gb(cell, args.dense_query_tile)
        if estimate > args.max_transient_gb:
            raise SystemExit(
                f"cell {name} estimates {estimate:.2f} GB of transient, over the "
                f"--max-transient-gb {args.max_transient_gb} budget; raise it "
                "deliberately or drop the cell"
            )
        report = run_cell(cell, args)
        results["cells"][name] = report
        for line in report["identity"]["failures"]:
            failures.append(f"{name}/identity: {line}")
        for arm, entry in report["parity"].items():
            for line in entry["tolerance_failures"]:
                failures.append(f"{name}/{arm}: {line}")
        del cell
        mx.clear_cache()

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\n[out] {path}", flush=True)

    if failures:
        print("\n[FAILURES]", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        if not args.no_fail_on_tolerance:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
