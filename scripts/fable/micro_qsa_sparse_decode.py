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
  stock                       ``qsa_sparse_decode.stock_reference`` -- the
                              shipped rows-gather lane (gathered K/V, score
                              GEMM, fp32 softmax, bf16 P@V), transcribed
  gather_kernel               the same, but with the SHIPPED fused K/V gather
                              (MTPLX_QSA_M4_FUSED_KV_GATHER) in front of it,
                              which is what actually runs in production

VISIBLE-SET IDENTITY (asserted, not assumed)
--------------------------------------------
The kernel reads no ``block_valid``.  It applies the shipped lane's own
per-slot predicate ``block < (pos + 1) // 4`` to every slot, which is what
``qsa_m4_row_tokens`` does -- see ``mtplx/kernels/qsa_sparse_decode.py``.
Every cell asserts, on the host and with integers only, that the kernel's
model of the selected key set equals the shipped closed form's, for every
row.  If that assertion ever fails the timing numbers are meaningless and the
run aborts.

PARITY IS ROUNDING CLASS, AND THAT IS THE POINT
------------------------------------------------
fp32 online softmax in exp2, fp32 probabilities into an fp32 P@V, Steel-MMA
reassociation of the 256-term score contraction, and one split-K rescale.
Reported per cell: max abs diff, the same in bf16 ulp at the reference's own
magnitude, relative L2, and head-dim top-1 agreement.  The gates are the
ones ``qsa_sparse_decode`` installs with (8 bf16 ulp / 2e-3 / 0.98).  They
are a SANITY gate.  The quality gate is model-level greedy-token agreement
plus a full HumanEval run, exactly as for MTPLX_FABLE_HC_M4.

COMMANDS
--------
Build (CPU only, no Metal execution -- but still take the lock so the build's
CPU/disk load does not land in the middle of someone's arm)::

    W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w68-qsa-sparse-decode
    PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
    cd $W/native_extensions/qsa_sparse_gqa && \
      cmake -S . -B build \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/mtplx_native_qsa/ \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DPython_EXECUTABLE=$PY && \
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
SPLIT_TARGETS = (4, 8, 16, 17, 32)
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


def stock_arm(cell):
    def call():
        return lane.stock_reference(
            cell["queries"],
            cell["keys"],
            cell["values"],
            cell["top_idx"],
            query_offset=cell["q_offset"],
            scale=SCALE,
        )

    return call


def gather_kernel_arm(cell):
    """The shipped production path: fused K/V gather + rows-gather attention.

    This is the arm the ABBA will actually replace, so it is the one whose
    time matters.  Skipped when the fused gather cannot bind.
    """

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
    rows = cell["rows"]

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
        "passes_install_gate": (
            max_abs / (_BF16_REL_ULP * scale) <= lane.PARITY_MAX_ABS_ULPS
            and l2d / max(l2r, 1e-12) <= lane.PARITY_MAX_REL_L2
            and top1 >= lane.PARITY_MIN_TOP1
        ),
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

    reference = stock_arm(cell)()
    mx.eval(reference)

    out: dict = {
        "rows": rows,
        "q_offset": q_offset,
        "total_tokens": total_tokens,
        "identity": identity,
        "arms": {},
        "stock_bytes": stock_bytes(rows),
    }

    stock_time = time_arm(stock_arm(cell), args.reps, args.warmup, args.queue_depth)
    stock_time["gbps_per_layer"] = gbps(
        out["stock_bytes"]["total"], stock_time["queued_ms"]
    )
    stock_time["ms_per_cycle_12_layers"] = stock_time["queued_ms"] * QSA_LAYERS
    out["arms"]["stock"] = {"time": stock_time}
    print(
        f"  [time] stock: queued={stock_time['queued_ms']:.4f} ms/layer  "
        f"({stock_time['ms_per_cycle_12_layers']:.3f} ms/cycle over "
        f"{QSA_LAYERS} layers, {stock_time['gbps_per_layer']:.0f} GB/s of "
        f"{MACHINE_GBPS:.0f})"
    )

    gather_call, gather_reason = gather_kernel_arm(cell)
    if gather_call is None:
        out["arms"]["gather_kernel"] = {"skipped": gather_reason}
        print(f"  [skip] gather_kernel: {gather_reason}")
    else:
        gtime = time_arm(gather_call, args.reps, args.warmup, args.queue_depth)
        gtime["gbps_per_layer"] = gbps(
            out["stock_bytes"]["total"], gtime["queued_ms"]
        )
        gtime["ms_per_cycle_12_layers"] = gtime["queued_ms"] * QSA_LAYERS
        out["arms"]["gather_kernel"] = {
            "time": gtime,
            "parity_vs_stock": compare(reference, gather_call()),
        }
        print(
            f"  [time] gather_kernel: queued={gtime['queued_ms']:.4f} ms/layer  "
            f"({gtime['ms_per_cycle_12_layers']:.3f} ms/cycle, "
            f"{gtime['gbps_per_layer']:.0f} GB/s)"
        )

    baseline_ms = out["arms"].get("gather_kernel", {}).get(
        "time", stock_time
    )["queued_ms"]

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
            parity = compare(reference, call())
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
            print(
                f"         parity: max_abs={parity['max_abs']:.3e} "
                f"({parity['max_abs_bf16_ulps']:.2f} bf16 ulp), "
                f"rel_l2={parity['rel_l2']:.3e}, top1={parity['top1']:.4f}, "
                f"gate={'PASS' if parity['passes_install_gate'] else 'FAIL'}"
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

    # M=4 verify first (the whole prize), M=1 draft second.
    cells = [
        ("verify-m4-16k", 4, 16_380, CAPACITY),
        ("draft-m1-16k", 1, 16_383, CAPACITY),
    ]
    for name, rows, q_offset, total in cells:
        report["cells"][name] = run_cell(name, rows, q_offset, total, args)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
