#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price MoE expert dedup at the Flash-Next verifier's physical M=4 shape.

The 125B-A6B verifier routes 4 rows x top-10 over 512 experts.  Today the
routed MoE is one ``gather_qmm`` with ``rhs_indices=[4, 10]``: 40 independent
M=1 mat-vecs, so an expert chosen by two rows is streamed from memory twice.
Profiler attribution puts that path at ~31% of GPU-busy decode and ~568 GB/s,
i.e. bandwidth-bound -- which makes the duplicate streams pure waste.

Variants timed over a full 48-layer cycle's worth of routing decisions:

  a  stock          gather_qmm, unsorted, 40 rows of M=1 (today)
  b1 sorted         ids argsorted so duplicates are adjacent, sorted_indices=False
  b2 sorted-flag    same, sorted_indices=True (MLX only takes the fast rhs
                    kernel at M==1 && B>=16 && B/E>=4; with E=512 that is
                    false, so this should fall back to gather_qmv)
  c  unique-M4      distinct experts computed ON DEVICE, padded to 40, each run
                    once at M=4, results scattered back to (row, slot)
  c2 unique-M4 exact host-known U, no padding -- the ideal this restructuring
                    can reach

Standalone by construction: imports no mtplx.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# MLX is imported lazily by ``main`` so the CLI surface and the id patterns can
# be unit tested on a box that is not holding the GPU lock -- importing MLX is
# the first step toward touching the device, and every code path below that
# uses ``mx`` runs only inside the guarded window.
mx = None


def _require_mlx() -> None:
    global mx
    import mlx.core

    mx = mlx.core

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_id_patterns import (  # noqa: E402
    NUM_EXPERTS,
    ROWS,
    SLOTS,
    TOP_K,
    UNIQUE_CHOICES,
    load_census_id_sets,
    make_layer_id_sets,
    unique_count,
)

HIDDEN = 2560
MOE_INTERMEDIATE = 640
GU_OUT = 2 * MOE_INTERMEDIATE
GROUP_SIZE = 32
BITS = 4
LAYERS = 48

# Static per-layer dispatch counts, hand-counted off the graph each variant
# builds (reshape/expand_dims/squeeze are metadata and cost nothing; MLX may
# additionally elide a contiguous slice, so treat these as an upper bound).
# The spread is the point: (c) buys its bandwidth saving by adding ~18 tiny
# dispatches per layer for the on-device dedup.  At the ~2.0 us/dispatch this
# box fits, that is ~36 us/layer, ~1.7 ms/cycle -- which is why
# micro_dispatch_overhead.py has to be read alongside this table, and why (c2)
# exists to show the restructuring's ceiling without the dedup tax.
DISPATCHES = {
    "a": 7,    # gather_qmm, split x2, sigmoid, mul x2, gather_qmm
    "b1": 14,  # + argsort x2, floor_divide, take x3 (sort/unsort)
    "b2": 14,
    "c": 25,   # c2 + sort, slice x2, ne, concat, astype, cumsum, sub, zeros,
               #      take, add, put_along_axis
    "c2": 13,  # zeros, gather_qmm, split x2, sigmoid, mul x2, gather_qmm,
               #      eq, astype, argmax, arange, take
}


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _quantized_bank(experts: int, out_dim: int, in_dim: int, chunk: int = 64):
    """Stack ``experts`` independently-quantized [out_dim, in_dim] matrices.

    Chunked so the bf16 source never materializes for all 512 at once.  Each
    expert holds different values, which is what makes the numerics check able
    to catch an index bug instead of hiding it behind identical weights.
    """

    ws, ss, bs = [], [], []
    for start in range(0, experts, chunk):
        n = min(chunk, experts - start)
        src = mx.random.normal((n, out_dim, in_dim)).astype(mx.bfloat16)
        w, s, b = mx.quantize(src, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(w, s, b)
        ws.append(w)
        ss.append(s)
        bs.append(b)
        del src
    bank = (
        mx.concatenate(ws, axis=0),
        mx.concatenate(ss, axis=0),
        mx.concatenate(bs, axis=0),
    )
    mx.eval(*bank)
    return bank


def bytes_per_expert(gu, dn) -> int:
    total = 0
    for w, s, b in (gu, dn):
        for arr in (w, s, b):
            total += int(np.prod(arr.shape[1:])) * arr.dtype.size
    return total


# --------------------------------------------------------------------------
# variants -- each returns [ROWS, TOP_K, HIDDEN] bf16 for one layer
# --------------------------------------------------------------------------


def run_stock(x, ids, gu, dn, sorted_indices=False):
    xr = mx.expand_dims(x, (-2, -3))  # [4, 1, 1, 2560]
    g = mx.gather_qmm(xr, gu[0], gu[1], gu[2], rhs_indices=ids, transpose=True,
                      group_size=GROUP_SIZE, bits=BITS,
                      sorted_indices=sorted_indices)  # [4, 10, 1, 1280]
    gate, up = mx.split(g, 2, axis=-1)
    h = silu(gate) * up
    out = mx.gather_qmm(h, dn[0], dn[1], dn[2], rhs_indices=ids, transpose=True,
                        group_size=GROUP_SIZE, bits=BITS,
                        sorted_indices=sorted_indices)
    return out.squeeze(-2)


def run_sorted(x, ids, gu, dn, sorted_indices):
    """mlx_lm switch_layers._gather_sort/_scatter_unsort, inlined."""

    xr = mx.expand_dims(x, (-2, -3))
    flat = ids.flatten()
    order = mx.argsort(flat)
    inv_order = mx.argsort(order)
    xs = xr.flatten(0, -3)[order // TOP_K]  # [40, 1, 2560]
    idx = flat[order]
    g = mx.gather_qmm(xs, gu[0], gu[1], gu[2], rhs_indices=idx, transpose=True,
                      group_size=GROUP_SIZE, bits=BITS,
                      sorted_indices=sorted_indices)
    gate, up = mx.split(g, 2, axis=-1)
    h = silu(gate) * up
    out = mx.gather_qmm(h, dn[0], dn[1], dn[2], rhs_indices=idx, transpose=True,
                        group_size=GROUP_SIZE, bits=BITS,
                        sorted_indices=sorted_indices)
    out = mx.unflatten(out[inv_order], 0, (ROWS, TOP_K))
    return out.squeeze(-2)


def _select(out_u, ids, uniq):
    """Map each (row, slot) back to its unique-expert result, on device."""

    eq = ids[..., None] == uniq[None, None, :]  # [4, 10, U]
    slot = mx.argmax(eq.astype(mx.uint8), axis=-1)
    rowb = mx.broadcast_to(mx.arange(ROWS).reshape(ROWS, 1), (ROWS, TOP_K))
    return out_u[slot, rowb]  # [4, 10, 2560]


def run_unique_m4(x, ids, gu, dn):
    """Device-side dedup, fixed output width SLOTS (no host sync)."""

    flat = ids.flatten()
    s = mx.sort(flat)
    first = mx.concatenate([mx.array([True]), s[1:] != s[:-1]])
    pos = mx.cumsum(first.astype(mx.int32)) - 1
    # Pad with s[0] (a real id, so the padded lanes are cache-hot repeats) and
    # scatter each run's value into its compacted slot.  Non-first members of a
    # run scatter the same value to the same slot, so write order is immaterial.
    uniq = mx.zeros((SLOTS,), dtype=flat.dtype) + s[0]
    uniq = mx.put_along_axis(uniq, pos, s, axis=0)
    return _run_m4_core(x, ids, uniq, gu, dn)


def _run_m4_core(x, ids, uniq, gu, dn):
    x4 = x.reshape(1, ROWS, HIDDEN)
    lhs = mx.zeros(uniq.shape, dtype=mx.uint32)
    g = mx.gather_qmm(x4, gu[0], gu[1], gu[2], lhs_indices=lhs, rhs_indices=uniq,
                      transpose=True, group_size=GROUP_SIZE, bits=BITS)
    gate, up = mx.split(g, 2, axis=-1)  # [U, 4, 640]
    h = silu(gate) * up
    out_u = mx.gather_qmm(h, dn[0], dn[1], dn[2], rhs_indices=uniq,
                          transpose=True, group_size=GROUP_SIZE, bits=BITS)
    return _select(out_u, ids, uniq)  # [U, 4, 2560] -> [4, 10, 2560]


def build_variant(name, layer_inputs, gu, dn):
    def run():
        outs = []
        for x, ids, uniq_exact in layer_inputs:
            if name == "a":
                outs.append(run_stock(x, ids, gu, dn, sorted_indices=False))
            elif name == "b1":
                outs.append(run_sorted(x, ids, gu, dn, sorted_indices=False))
            elif name == "b2":
                outs.append(run_sorted(x, ids, gu, dn, sorted_indices=True))
            elif name == "c":
                outs.append(run_unique_m4(x, ids, gu, dn))
            elif name == "c2":
                outs.append(_run_m4_core(x, ids, uniq_exact, gu, dn))
            else:  # pragma: no cover
                raise ValueError(name)
        return outs

    return run


def time_variant(run, reps, warmup, clear_cache):
    """MLX is lazy: ``run()`` only builds the graph, ``mx.eval`` executes it.

    ``median_ms`` is the encode+GPU time -- the number the bandwidth claim rests
    on.  ``build_ms`` is the Python graph-construction tax, reported separately
    because variant (c) builds more ops per layer and should not be charged for
    that on the GPU line.
    """

    for _ in range(warmup):
        mx.eval(run())
    evals, builds = [], []
    for _ in range(reps):
        if clear_cache:
            mx.clear_cache()
        t0 = time.perf_counter()
        out = run()
        t1 = time.perf_counter()
        mx.eval(out)
        evals.append((time.perf_counter() - t1) * 1e3)
        builds.append((t1 - t0) * 1e3)
    evals.sort()
    return {
        "median_ms": statistics.median(evals),
        "p10_ms": evals[max(0, int(0.10 * (len(evals) - 1)))],
        "p90_ms": evals[min(len(evals) - 1, int(0.90 * (len(evals) - 1)))],
        "build_ms": statistics.median(builds),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unique", type=int, default=40, choices=list(UNIQUE_CHOICES))
    p.add_argument("--from-census", type=str, default=None,
                   help="JSON list of real [4, 10] expert-id arrays to replay")
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clear-cache", action="store_true",
                   help="mx.clear_cache() before each rep (colder buffer pool)")
    p.add_argument("--variants", type=str, default="a,b1,b2,c,c2")
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print("[micro-moe-dedup] must run under /tmp/mtplx-gpu-exclusive.lock", flush=True)
    _require_mlx()
    mx.random.seed(args.seed)

    if args.from_census:
        id_sets = load_census_id_sets(args.from_census, args.layers, seed=args.seed)
        source = f"census:{args.from_census}"
    else:
        id_sets = make_layer_id_sets(args.unique, args.layers, seed=args.seed)
        source = f"synthetic:U={args.unique}"
    uniques = [unique_count(i) for i in id_sets]

    print(f"[build] {NUM_EXPERTS} experts, gu [{GU_OUT}, {HIDDEN}], "
          f"down [{HIDDEN}, {MOE_INTERMEDIATE}], q{BITS} g{GROUP_SIZE}", flush=True)
    gu = _quantized_bank(NUM_EXPERTS, GU_OUT, HIDDEN)
    dn = _quantized_bank(NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE)
    per_expert = bytes_per_expert(gu, dn)
    print(f"[build] gu weight {gu[0].shape} {gu[0].dtype}, scales {gu[1].shape} "
          f"{gu[1].dtype}; down weight {dn[0].shape}; {per_expert / 1e6:.3f} MB/expert",
          flush=True)

    layer_inputs = []
    for ids_np in id_sets:
        x = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
        ids = mx.array(ids_np.astype(np.int32))
        uniq_exact = mx.array(np.unique(ids_np).astype(np.int32))
        mx.eval(x, ids, uniq_exact)
        layer_inputs.append((x, ids, uniq_exact))

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    results = {}
    for name in names:
        run = build_variant(name, layer_inputs, gu, dn)
        stats = time_variant(run, args.reps, args.warmup, args.clear_cache)
        streamed = float(np.mean(uniques)) if name in ("c", "c2") else float(SLOTS)
        gb = streamed * per_expert * args.layers / 1e9
        stats.update({
            "distinct_experts_streamed": streamed,
            "rows_issued_per_layer": SLOTS if name != "c2" else float(np.mean(uniques)),
            "gb_moved": gb,
            "gbps": gb / (stats["median_ms"] / 1e3),
            "ms_per_layer": stats["median_ms"] / args.layers,
            "dispatches_per_layer": DISPATCHES[name],
        })
        results[name] = stats

    # numerics: every variant against stock on the same layer-0 inputs
    x0, ids0, uniq0 = layer_inputs[0]
    ref = run_stock(x0, ids0, gu, dn, sorted_indices=False)
    mx.eval(ref)
    numerics = {}
    cands = {"b1": lambda: run_sorted(x0, ids0, gu, dn, False),
             "b2": lambda: run_sorted(x0, ids0, gu, dn, True),
             "c": lambda: run_unique_m4(x0, ids0, gu, dn),
             "c2": lambda: _run_m4_core(x0, ids0, uniq0, gu, dn)}
    for name, fn in cands.items():
        if name not in names:
            continue
        got = fn()
        mx.eval(got)
        diff = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        numerics[name] = {
            "max_abs_diff": float(mx.max(diff).item()),
            "differing_elements": int(mx.sum(got != ref).item()),
            "total_elements": int(ref.size),
        }

    print(f"\nsource={source}  layers={args.layers}  reps={args.reps}  "
          f"mean_unique={np.mean(uniques):.2f}  min/max={min(uniques)}/{max(uniques)}")
    hdr = (f"{'var':<4}{'U':>7}{'eval/48L':>10}{'p10':>9}{'p90':>9}"
           f"{'ms/layer':>10}{'GB':>8}{'GB/s':>9}{'disp/L':>8}{'build':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        r = results[name]
        print(f"{name:<4}{r['distinct_experts_streamed']:>7.1f}{r['median_ms']:>10.3f}"
              f"{r['p10_ms']:>9.3f}{r['p90_ms']:>9.3f}{r['ms_per_layer']:>10.4f}"
              f"{r['gb_moved']:>8.2f}{r['gbps']:>9.1f}"
              f"{r['dispatches_per_layer']:>8d}{r['build_ms']:>9.3f}")
    print("\nnumerics vs stock (a):")
    for name, n in numerics.items():
        print(f"  {name:<3} max_abs_diff={n['max_abs_diff']:.6g}  "
              f"differing={n['differing_elements']}/{n['total_elements']}")

    summary = {
        "source": source, "layers": args.layers, "reps": args.reps,
        "unique_counts": uniques, "mean_unique": float(np.mean(uniques)),
        "bytes_per_expert": per_expert, "variants": results, "numerics": numerics,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
