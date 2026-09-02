#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Parity + timing for the SPLIT-K QSA sparse-GQA DECODE kernel (K-Q2, K-D6).

The kernel under test is ``mtplx.native.qsa_sparse_gqa_decode`` -- the
KV-split variant of MTPLX's port of oMLX's Steel-MMA sparse GQA.  One
threadgroup owns one (query row, KV head, KV SPLIT); the twelve query heads
sharing a KV head are a padded 16-row MMA tile; the selected block ids are
walked directly out of the full KV cache backing; each split keeps its own
fp32 online-softmax state and a merge pass combines them.  No gathered K/V
tensor, no transposed copy, no score tensor, no bool mask.

WHY THIS BENCH EXISTS AND WHAT IT MUST NOT BE READ AS
------------------------------------------------------
The phase-1 kernel's grid is ``(qL, kv_heads, 1)``: EIGHT threadgroups at
M=4, TWO at M=1, on a 40-core M5 Max.  Phase 1's own note said those cells
were there to measure a loss, and priced the split-K variant as its own
item.  This is that variant, so the question this bench answers is "does the
split fix the occupancy without the partial-state traffic eating the win",
not "is the kernel faster than nothing".

It is ALSO not the ship gate.  Prior QSA work on this runtime has had an
isolated -1.9 ms turn into 0 end-to-end (W16), and the lightning lane lost to
dense at 16K after looking good in isolation (W19).  The verdict is the 16K
ABBA plus a HumanEval run; the command for both is at the bottom of this
docstring.

WHAT IS COMPARED
----------------
Selection is NOT re-implemented: a synthetic ``top_idx`` stands in for
``mx.argpartition``'s output and is fed UNSORTED to every arm, because that
is what ``QSAIndexer._select_m4`` produces and because an ordering assumption
is exactly the bug this kernel must not have.

Arms, all over one visible set:
  native_bk{BK}_dc{DC}_s{N}   the split-K kernel, per tile and split target
  production_gather_kernel    THE BASELINE.  The shipped fused K/V gather
                              (MTPLX_QSA_M4_FUSED_KV_GATHER, one dispatch)
                              plus the score GEMM / mask / fp32 softmax /
                              bf16 cast / P@V that follow it.  This is what
                              the ABBA replaces and the only arm a speedup
                              may be quoted against.
  portable_take_reference     ``qsa_sparse_decode.stock_reference``: the SAME
                              math and the same bytes, but gathering through
                              two generic ``mx.take`` calls instead of the
                              fused kernel.  It is the numerics reference for
                              parity and NOT a baseline -- measured 2.35x
                              slower than production (0.542 vs 0.230 ms/layer,
                              2026-09-02), which is the generic gather's
                              overhead, not the production path's.

MEASURED 2026-09-02, second guarded run (M=4, 16K, queued lane, 12 layers):

    portable_take_reference   0.531 ms/layer   6.38 ms/cycle   134 GB/s
    production_gather_kernel  0.226 ms/layer   2.71 ms/cycle   315 GB/s   BASELINE

    native, by THREADGROUP COUNT (the whole story is occupancy):

      tgs   BK:DC   splits   ms/layer   ms/cycle   x baseline
       32   128:32     4       0.325      3.90        0.70
       48   128:32     8       0.210      2.51        1.08
       72   128:32    16       0.149      1.79        1.52
      136   128:32    17       0.099      1.19        2.28
      136   128:32    32       0.094      1.13        2.41
      136    64:64    17       0.098      1.18        2.30
      136    64:64    32       0.090      1.08        2.51

    Time x threadgroups is 10.4 / 10.1 / 10.7 / 12.8 -- near-perfect inverse
    scaling to 72 threadgroups and a mild falloff at 136.  This kernel is
    OCCUPANCY-bound, not bandwidth-bound, which is exactly the risk the split
    was built to remove: at 136 tgs it achieves 216 GB/s of 544 and still
    wins 2.4x, because it moves 0.28x the bytes.

    NOISE FLOOR, from configurations that are provably identical: at BK=128
    there are 17 tiles, so splits 17 and 32 both clamp to a 17-split,
    136-threadgroup grid -- the same work measured twice.  They differ by
    5.3% (0.0992 vs 0.0939).  BK=64 s17/s32 likewise (8.2%), BK=256
    s16/s17/s32 likewise (3.4%).  So the noise floor is 3-8%, the 2.6% gap
    between 128:32 and 64:64 at 136 tgs is NOT a difference, and no arm may
    be declared a winner on a margin under ~8%.

    Context: micro_qsa_m4.py (2026-09-01) attributes 1.501 ms/cycle of the
    baseline's 2.71 to the fused gather plus the transposed-K copy alone --
    55% of the production attention chain, and the part this kernel deletes.

VISIBLE-SET IDENTITY (asserted, not assumed)
--------------------------------------------
The kernel reads no ``block_valid``.  It applies the shipped lane's own
per-slot predicate ``block < (pos + 1) // 4`` to every slot, which is what
``qsa_m4_row_tokens`` does -- see ``mtplx/kernels/qsa_sparse_decode.py``.
Every cell asserts, on the host and with integers only, that the kernel's
model of the selected key set equals the shipped closed form's, for every
row.  If that assertion ever fails the timing numbers are meaningless and the
run aborts.

PARITY IS ROUNDING CLASS, MEASURED AGAINST A LADDER
----------------------------------------------------
The first run reported the SAME parity to four significant figures for all
twenty configurations -- BK 64/128/256, DC 32/64, splits 4..32 -- at max_abs
1.953e-3 (= 2**-9 exactly), rel_l2 4.78e-3, top-1 1.0000.  DC changes the fp32
score contraction order and the split count changes the online-softmax merge
tree, so a delta that does not move across either is not the kernel's.

Three references now isolate it, differing only in where bf16 rounding is
applied:

  shipped             bf16 scores (mx.matmul on bf16 operands RETURNS bf16,
                      so the shipped path rounds the scores before the
                      softmax) and bf16 probabilities -- the production path
  shipped_fp32_probs  bf16 scores, fp32 probabilities
  fp32                fp32 scores and fp32 probabilities -- what the kernel
                      computes, and the DECIDING reference

Prediction to be confirmed or falsified by the ladder: the score cast should
dominate.  A relative score error u = 2**-9 shifts a softmax logit by u*|x|,
and with scaled logits of order 5 that is ~2e-2 relative on the probabilities,
against 2e-3 for the probability cast alone.

Gates (mtplx/kernels/qsa_sparse_decode.py states the derivations): TIGHT vs
fp32 -- 2 bf16 ulp and rel_l2 5e-4, because the only differences left are fp32
reassociation (~1e-6) and one bf16 store; LOOSE vs shipped -- rel_l2 5e-2,
which bounds the SHIPPED path's own quantisation and cannot be tightened by
improving this kernel.  Both are SANITY gates.  The quality gate is
model-level greedy-token agreement plus a full HumanEval run, exactly as for
MTPLX_FABLE_HC_M4.

COMMANDS
--------
Build (CPU only, no Metal execution -- but still take the lock so the build's
CPU/disk load does not land in the middle of someone's arm)::

    W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w68-qsa-sparse-decode
    PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
    # nanobind MUST match the one mlx.core was built with (internals v21 for
    # mlx 0.32.2); the venv ships 2.12.0 = v19, which builds and imports fine
    # and then makes every call raise TypeError.  CMake now refuses that, and
    # scripts/fable/check_native_qsa_abi.py diagnoses it without building.
    NB=/Users/davidtai/.local/share/uv/tools/mtplx/lib/python3.13/site-packages/nanobind
    $PY $W/scripts/fable/check_native_qsa_abi.py --python $PY --nanobind $NB
    rm -rf $W/native_extensions/qsa_sparse_gqa/build   # stale v19 cache
    cd $W/native_extensions/qsa_sparse_gqa && \
      cmake -S . -B build \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/mtplx_native_qsa/ \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DPython_EXECUTABLE=$PY -DMTPLX_NANOBIND_DIR=$NB && \
      cmake --build build -j4

This bench (one guarded window)::

    PYTHONPATH=$W $PY /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
      --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
      --lock-timeout-seconds 3600 --child-timeout-seconds 3600 -- \
      $PY $W/scripts/fable/micro_qsa_sparse_decode.py \
        --out $W/.benchmark-artifacts/fable/micro-qsa-sparse-decode.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from mtplx.kernels import qsa_sparse_decode as lane  # noqa: E402
from mtplx.native import (  # noqa: E402
    native_qsa_available,
    qsa_sparse_gqa_decode,
    qsa_sparse_gqa_decode_partial_shape,
    qsa_sparse_gqa_decode_split_geometry,
    qsa_sparse_gqa_decode_unsupported_reason,
)

Q_HEADS = lane.Q_HEADS
KV_HEADS = lane.KV_HEADS
HEAD_DIM = lane.HEAD_DIM
TOP_K = lane.TOP_K
RATIO = lane.COMPRESS_RATIO
SCALE = float(HEAD_DIM) ** -0.5
#: The retained stack's own numbers: 12 QSA layers, capacity 17,408 (the
#: census kernel name is ``..._fused_kv_gather_c17408``).
QSA_LAYERS = 12
CAPACITY = 17_408
TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
#: 17 is the tile count at BK=128, so 17/32/64 all clamp to the same 17-split
#: grid there -- which is why the 2026-09-02 run measured s17 and s32 as
#: separate rows with identical geometry, and why their 5% spread IS the
#: measurement's noise floor rather than a difference.  33 and 64 are here
#: because at BK=64 there are 33 tiles, so 33 is the first value that reaches
#: 33 splits = 264 threadgroups, and it was NOT covered by the first sweep.
SPLIT_TARGETS = (4, 8, 16, 17, 32, 33, 64)
#: M5 Max measured GPU read bandwidth (TEST_MACHINES.md).
MACHINE_GBPS = 544.0


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def make_cell(rows: int, q_offset: int, total_tokens: int, seed: int):
    """One synthetic QSA layer at the production geometry.

    ``top_idx`` is a shuffled prefix of the logical block range, NOT a sorted
    one: ``_select_m4`` hands ``mx.argpartition``'s raw output through, and a
    bench that sorts it would hide the very assumption the kernel must not
    make.
    """

    mx.random.seed(seed)
    nb_total = total_tokens // RATIO
    queries = mx.random.normal((1, Q_HEADS, rows, HEAD_DIM)).astype(mx.bfloat16)
    keys = mx.random.normal((1, KV_HEADS, CAPACITY, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, KV_HEADS, CAPACITY, HEAD_DIM)).astype(mx.bfloat16)
    order = mx.argsort(mx.random.uniform(shape=(rows, nb_total)), axis=-1)
    top_idx = order[:, :TOP_K].astype(mx.int32)
    mx.eval(queries, keys, values, top_idx)
    return {
        "queries": queries,
        "keys": keys,
        "values": values,
        "top_idx": top_idx,
        "rows": rows,
        "q_offset": q_offset,
        "total_tokens": total_tokens,
    }


def assert_visible_sets_agree(cell) -> dict:
    """Integer-exact identity of the attended key set, per row.  Aborts on a miss."""

    ids = cell["top_idx"].tolist()
    q0 = cell["q_offset"]
    checked = 0
    for row, row_ids in enumerate(ids):
        q_abs = q0 + row
        if not lane.visible_sets_agree(
            row_ids, q_abs, key_length=cell["total_tokens"]
        ):
            raise SystemExit(
                f"visible-set identity FAILED at row {row} (q_abs={q_abs}): the "
                "kernel and the shipped rows-gather lane would attend different "
                "keys; every timing number below would be meaningless"
            )
        checked += 1
    counts = [
        sum(
            1
            for p in lane.kernel_row_tokens(
                row_ids, q0 + row, key_length=cell["total_tokens"]
            )
            if p >= 0
        )
        for row, row_ids in enumerate(ids)
    ]
    return {"rows_checked": checked, "visible_keys_per_row": counts}


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def native_arm(cell, key_tile: int, dim_tile: int, key_splits: int):
    reason = qsa_sparse_gqa_decode_unsupported_reason(
        cell["queries"],
        cell["keys"],
        cell["values"],
        cell["top_idx"],
        query_offset=cell["q_offset"],
        total_tokens=cell["total_tokens"],
        scale=SCALE,
        key_tile=key_tile,
        dimension_tile=dim_tile,
        key_splits=key_splits,
    )
    if reason is not None:
        return None, reason

    def call():
        return qsa_sparse_gqa_decode(
            cell["queries"],
            cell["keys"],
            cell["values"],
            cell["top_idx"],
            query_offset=cell["q_offset"],
            total_tokens=cell["total_tokens"],
            scale=SCALE,
            key_tile=key_tile,
            dimension_tile=dim_tile,
            key_splits=key_splits,
        )

    return call, None


def reference_arm(cell, *, fp32_scores: bool, fp32_probs: bool):
    def call():
        return lane.reference_attention(
            cell["queries"],
            cell["keys"],
            cell["values"],
            cell["top_idx"],
            query_offset=cell["q_offset"],
            scale=SCALE,
            fp32_scores=fp32_scores,
            fp32_probs=fp32_probs,
        )

    return call


def stock_arm(cell):
    return reference_arm(cell, fp32_scores=False, fp32_probs=False)


def gather_kernel_arm(cell):
    """The shipped production path: fused K/V gather + rows-gather attention.

    This is the arm the ABBA will actually replace, so it is the one whose
    time matters.  Skipped when the fused gather cannot bind.
    """

    rows = cell["rows"]
    if rows != lane.VERIFY_ROWS:
        # kernels/qwen4_qsa_m4_fused_kv_gather.py compiles ``_ROWS = 4`` into
        # the kernel, so it emits a 4-row [1,2,4,2052,256] pair whatever it is
        # handed.  On 2026-09-02 that reached the reshape below and raised
        # "Cannot reshape array of size 4202496 into shape (1,2,1,1,256,2052)"
        # AFTER the M=4 cell had already produced its numbers.  There is no
        # M=1 production path to baseline against anyway -- the retained-stack
        # census shows the draft chain running the MTP block, not the twelve
        # QSA layers -- so refuse here rather than crash three arms later.
        return None, (
            f"the shipped fused K/V gather is compiled for {lane.VERIFY_ROWS} "
            f"rows (_ROWS = 4) and cannot serve M={rows}; there is no M={rows} "
            "production attention path to baseline against"
        )
    try:
        from mtplx.kernels.qwen4_qsa_m4_fused_kv_gather import (
            bind_qwen4_qsa_m4_fused_kv_gather,
        )
        from mtplx.kernels.qwen4_qsa_m4_indexer import qsa_m4_row_tokens
    except Exception as exc:  # pragma: no cover - import shape only
        return None, f"fused gather unavailable ({exc})"

    gather = bind_qwen4_qsa_m4_fused_kv_gather(
        capacity=CAPACITY, transposed_keys=False
    )

    def call():
        token_idx, token_ok = qsa_m4_row_tokens(
            cell["top_idx"], cell["q_offset"], compress_ratio=RATIO
        )
        k_sel, v_sel = gather(cell["keys"], cell["values"], token_idx)
        width = int(token_idx.shape[1])
        q_view = cell["queries"].reshape(1, KV_HEADS, lane.GQA, rows, 1, HEAD_DIM)
        k_view = k_sel.swapaxes(-1, -2).reshape(
            1, KV_HEADS, 1, rows, HEAD_DIM, width
        )
        scores = mx.matmul(q_view, k_view).squeeze(-2).astype(mx.float32) * SCALE
        scores = mx.where(
            token_ok[None, None, None], scores, mx.array(-mx.inf, dtype=mx.float32)
        )
        probs = mx.softmax(scores, axis=-1).astype(mx.bfloat16)
        v_view = v_sel.reshape(1, KV_HEADS, 1, rows, width, HEAD_DIM)
        out = mx.matmul(probs[..., None, :], v_view).squeeze(-2)
        return out.reshape(1, Q_HEADS, rows, HEAD_DIM)

    return call, None


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------
_BF16_REL_ULP = 2.0**-8


def compare(reference: mx.array, candidate: mx.array) -> dict:
    ref = reference.astype(mx.float32)
    got = candidate.astype(mx.float32)
    diff = mx.abs(ref - got)
    stats = mx.stack(
        [
            mx.max(diff),
            mx.mean(diff),
            mx.max(mx.abs(ref)),
            mx.sqrt(mx.sum(diff * diff)),
            mx.sqrt(mx.sum(ref * ref)),
            mx.mean(
                (mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1)).astype(
                    mx.float32
                )
            ),
            mx.sum((diff > 0).astype(mx.float32)),
        ]
    )
    mx.eval(stats)
    max_abs, mean_abs, absmax, l2d, l2r, top1, differing = (
        float(x) for x in stats.tolist()
    )
    scale = max(absmax, 1e-3)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "ref_absmax": absmax,
        "max_abs_bf16_ulps": max_abs / (_BF16_REL_ULP * scale),
        "rel_l2": l2d / max(l2r, 1e-12),
        "top1": top1,
        "differing_elements": int(differing),
        "elements": int(reference.size),
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
    cost is amortised the way it is inside a real verify cycle.
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
# byte models -- stated, so the GB/s column means something
# --------------------------------------------------------------------------
def native_bytes(rows: int, key_tile: int, key_splits: int) -> dict:
    """What the split-K lane moves, per layer.

    Reads: the selected K and V rows once per (row, KV head) -- the kernel
    never writes a gathered copy.  Writes+reads: the fp32 partial states.
    """

    selected = lane.SELECTED_TOKENS
    kv_read = 2 * KV_HEADS * rows * selected * HEAD_DIM * 2
    q_read = Q_HEADS * rows * HEAD_DIM * 2
    ids_read = rows * TOP_K * 4
    shape = qsa_sparse_gqa_decode_partial_shape(rows, key_tile, key_splits)
    partial = shape[0] * shape[1] * shape[2] * shape[3] * 4
    out_write = Q_HEADS * rows * HEAD_DIM * 2
    return {
        "kv_read": kv_read,
        "q_read": q_read,
        "ids_read": ids_read,
        "partial_write": partial,
        "partial_read": partial,
        "out_write": out_write,
        "total": kv_read + q_read + ids_read + 2 * partial + out_write,
    }


def stock_bytes(rows: int) -> dict:
    """What the shipped lane moves, per layer.

    The dispatch census does NOT price this: its QSA row charges the gather a
    flat 4.19 MB and gives the score/softmax/PV dispatches zero bytes, and the
    transposed copy lands in the Copy family.  Counted honestly the shipped
    lane is near this machine's bandwidth ceiling and the lever is BYTES.
    """

    width = TOP_K * RATIO + RATIO  # the shipped lane builds 2,052 slots
    gathered = KV_HEADS * rows * width * HEAD_DIM * 2
    scores = Q_HEADS * rows * width * 4
    probs_bf16 = Q_HEADS * rows * width * 2
    return {
        "gather_read": 2 * gathered,
        "gather_write": 2 * gathered,
        "transpose_copy": 2 * gathered // 2,  # K only, read + write
        "score_read_k": gathered,
        "score_write": scores,
        "softmax_rw": 2 * scores,
        "cast_rw": scores + probs_bf16,
        "pv_read": gathered + probs_bf16,
        "out_write": Q_HEADS * rows * HEAD_DIM * 2,
        "total": (
            4 * gathered  # gather read + write of K and V
            + 2 * gathered  # MLX's contiguous copy of the transposed K view
            + gathered  # the score GEMM re-reads K_sel
            + gathered  # the PV GEMM re-reads V_sel
            + 4 * scores
            + 2 * probs_bf16
            + Q_HEADS * rows * HEAD_DIM * 2
        ),
    }


def gbps(total_bytes: int, ms: float) -> float:
    return (total_bytes / 1e9) / (ms / 1e3) if ms > 0 else 0.0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run_cell(name: str, rows: int, q_offset: int, total_tokens: int, args) -> dict:
    cell = make_cell(rows, q_offset, total_tokens, seed=hash(name) & 0xFFFF)
    identity = assert_visible_sets_agree(cell)
    print(
        f"[{name}] rows={rows} q_offset={q_offset} total={total_tokens} "
        f"visible/row={identity['visible_keys_per_row']}"
    )

    # The reference LADDER.  ``shipped`` is the production transcription;
    # ``fp32`` is what the kernel computes; the middle rung removes ONLY the
    # bf16 probability cast, so the two gaps attribute the delta between them.
    references = {
        "shipped": stock_arm(cell)(),
        "shipped_fp32_probs": reference_arm(
            cell, fp32_scores=False, fp32_probs=True
        )(),
        "fp32": reference_arm(cell, fp32_scores=True, fp32_probs=True)(),
    }
    mx.eval(*references.values())
    reference = references["shipped"]
    out_ladder = {
        "fp32_vs_shipped": compare(references["shipped"], references["fp32"]),
        "fp32_probs_vs_shipped": compare(
            references["shipped"], references["shipped_fp32_probs"]
        ),
    }
    print(
        "  [ladder] removing ONLY the bf16 probability cast moves the "
        f"reference by rel_l2={out_ladder['fp32_probs_vs_shipped']['rel_l2']:.3e}; "
        "removing the bf16 SCORE cast as well moves it by "
        f"rel_l2={out_ladder['fp32_vs_shipped']['rel_l2']:.3e}"
    )

    out: dict = {
        "rows": rows,
        "q_offset": q_offset,
        "total_tokens": total_tokens,
        "identity": identity,
        "reference_ladder": out_ladder,
        "arms": {},
        "stock_bytes": stock_bytes(rows),
    }

    stock_time = time_arm(stock_arm(cell), args.reps, args.warmup, args.queue_depth)
    stock_time["gbps_per_layer"] = gbps(
        out["stock_bytes"]["total"], stock_time["queued_ms"]
    )
    stock_time["ms_per_cycle_12_layers"] = stock_time["queued_ms"] * QSA_LAYERS
    out["arms"]["portable_take_reference"] = {"time": stock_time}
    print(
        f"  [time] portable_take_reference (NOT the baseline): "
        f"queued={stock_time['queued_ms']:.4f} ms/layer  "
        f"({stock_time['ms_per_cycle_12_layers']:.3f} ms/cycle over "
        f"{QSA_LAYERS} layers, {stock_time['gbps_per_layer']:.0f} GB/s of "
        f"{MACHINE_GBPS:.0f})"
    )

    gather_call, gather_reason = gather_kernel_arm(cell)
    if gather_call is None:
        out["arms"]["production_gather_kernel"] = {"skipped": gather_reason}
        print(
            f"  [SKIP] production_gather_kernel: {gather_reason} -- without it "
            "there is NO valid baseline and no speedup may be quoted"
        )
    else:
        gtime = time_arm(gather_call, args.reps, args.warmup, args.queue_depth)
        gtime["gbps_per_layer"] = gbps(
            out["stock_bytes"]["total"], gtime["queued_ms"]
        )
        gtime["ms_per_cycle_12_layers"] = gtime["queued_ms"] * QSA_LAYERS
        out["arms"]["production_gather_kernel"] = {
            "time": gtime,
            "parity_vs_portable_reference": compare(reference, gather_call()),
        }
        print(
            f"  [time] production_gather_kernel (BASELINE): "
            f"queued={gtime['queued_ms']:.4f} ms/layer  "
            f"({gtime['ms_per_cycle_12_layers']:.3f} ms/cycle, "
            f"{gtime['gbps_per_layer']:.0f} GB/s)"
        )

    baseline = out["arms"].get("production_gather_kernel", {}).get("time")
    out["baseline_arm"] = (
        "production_gather_kernel" if baseline else "portable_take_reference"
    )
    baseline_ms = (baseline or stock_time)["queued_ms"]

    for key_tile, dim_tile in args.tiles:
        for key_splits in args.splits:
            arm = f"native_bk{key_tile}_dc{dim_tile}_s{key_splits}"
            call, reason = native_arm(cell, key_tile, dim_tile, key_splits)
            if call is None:
                out["arms"][arm] = {"skipped": reason}
                print(f"  [skip] {arm}: {reason}")
                continue
            n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
                lane.SELECTED_TOKENS, key_tile, key_splits
            )
            candidate = call()
            parity = {
                "vs_fp32": compare(references["fp32"], candidate),
                "vs_shipped": compare(references["shipped"], candidate),
            }
            parity["gate"] = (
                parity["vs_fp32"]["max_abs_bf16_ulps"]
                <= lane.PARITY_FP32_MAX_ABS_ULPS
                and parity["vs_fp32"]["rel_l2"] <= lane.PARITY_FP32_MAX_REL_L2
                and parity["vs_fp32"]["top1"] >= lane.PARITY_MIN_TOP1
                and parity["vs_shipped"]["rel_l2"]
                <= lane.PARITY_SHIPPED_MAX_REL_L2
                and parity["vs_shipped"]["top1"] >= lane.PARITY_MIN_TOP1
            )
            del candidate
            timing = time_arm(call, args.reps, args.warmup, args.queue_depth)
            model = native_bytes(rows, key_tile, key_splits)
            timing["gbps_per_layer"] = gbps(model["total"], timing["queued_ms"])
            timing["ms_per_cycle_12_layers"] = timing["queued_ms"] * QSA_LAYERS
            out["arms"][arm] = {
                "time": timing,
                "parity": parity,
                "bytes": model,
                "grid": {
                    "n_tiles": n_tiles,
                    "tiles_per_split": per_split,
                    "n_splits": n_splits,
                    "threadgroups": rows * KV_HEADS * n_splits,
                    "merge_threadgroups": Q_HEADS * rows,
                },
                "baseline_arm": out["baseline_arm"],
                "speedup_vs_baseline": (
                    baseline_ms / timing["queued_ms"]
                    if timing["queued_ms"] > 0
                    else 0.0
                ),
                "bytes_ratio_vs_stock": (
                    model["total"] / out["stock_bytes"]["total"]
                ),
            }
            print(
                f"  [time] {arm}: queued={timing['queued_ms']:.4f} ms/layer  "
                f"({timing['ms_per_cycle_12_layers']:.3f} ms/cycle, "
                f"{timing['gbps_per_layer']:.0f} GB/s, "
                f"{out['arms'][arm]['speedup_vs_baseline']:.2f}x baseline, "
                f"{out['arms'][arm]['bytes_ratio_vs_stock']:.2f}x bytes, "
                f"{rows * KV_HEADS * n_splits} tgs)"
            )
            f32, shp = parity["vs_fp32"], parity["vs_shipped"]
            print(
                f"         vs fp32   : max_abs={f32['max_abs']:.3e} "
                f"({f32['max_abs_bf16_ulps']:.2f} ulp), "
                f"rel_l2={f32['rel_l2']:.3e}, top1={f32['top1']:.4f}"
            )
            print(
                f"         vs shipped: max_abs={shp['max_abs']:.3e} "
                f"({shp['max_abs_bf16_ulps']:.2f} ulp), "
                f"rel_l2={shp['rel_l2']:.3e}, top1={shp['top1']:.4f}   "
                f"gate={'PASS' if parity['gate'] else 'FAIL'}"
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument(
        "--tiles",
        default=",".join(f"{a}:{b}" for a, b in TILES),
        help="comma list of BK:DC pairs",
    )
    parser.add_argument(
        "--splits",
        default=",".join(str(s) for s in SPLIT_TARGETS),
        help="comma list of KV-split targets",
    )
    parser.add_argument(
        "--include-m1",
        action="store_true",
        help=(
            "also run the M=1 draft cell; off by default because the retained "
            "stack has no M=1 QSA attention and the shipped fused gather is "
            "compiled for 4 rows, so the cell has no baseline"
        ),
    )
    args = parser.parse_args()
    args.tiles = [
        tuple(int(x) for x in token.split(":"))
        for token in str(args.tiles).split(",")
        if token
    ]
    args.splits = [int(x) for x in str(args.splits).split(",") if x]

    if not native_qsa_available():
        print(
            "native QSA extension is not built; run the cmake command in this "
            "script's docstring first",
            file=sys.stderr,
        )
        return 2
    if not mx.metal.is_available():
        print("no Metal device", file=sys.stderr)
        return 2

    report = {
        "machine_gbps": MACHINE_GBPS,
        "qsa_layers": QSA_LAYERS,
        "capacity": CAPACITY,
        "selected_tokens": lane.SELECTED_TOKENS,
        "parity_gates": {
            "max_abs_bf16_ulps": lane.PARITY_MAX_ABS_ULPS,
            "rel_l2": lane.PARITY_MAX_REL_L2,
            "top1": lane.PARITY_MIN_TOP1,
        },
        "cells": {},
    }

    # M=4 verify is the whole prize.  The M=1 cell is OPT-IN: the retained
    # stack has no M=1 QSA attention at all (the census shows exactly 36 GDN
    # and 48 MoE layers per cycle, so the full stack runs once, at M=4), and
    # the shipped fused gather is compiled for 4 rows so there is nothing to
    # baseline it against either.
    cells = [("verify-m4-16k", 4, 16_380, CAPACITY)]
    if args.include_m1:
        cells.append(("draft-m1-16k", 1, 16_383, CAPACITY))
    for name, rows, q_offset, total in cells:
        report["cells"][name] = run_cell(name, rows, q_offset, total, args)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
