#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price the expert-major routed gate/up kernel against the retained one.

``micro_moe_dedup.py`` established the shape of the problem: at the verifier's
physical M=4, the routed MoE is 40 independent (row, expert) M=1 products while
the census says only ~28 of those experts are distinct, and MLX's M=4
``gather_qmm`` path -- the obvious way to exploit that -- is 2.2x SLOWER.  The
remaining lever is a kernel that stays at M=1 arithmetic but re-indexes its grid
from lane to expert, so a duplicated expert's q4 tile is streamed once.

This script times the gate/up half of that in isolation:

  a  retained     mtplx/kernels/qwen4_m4_routed_glu.py, lane-major, 3,200
                  threadgroups, one weight tile per LANE                (baseline)
  b4 expert-major mtplx/kernels/qwen4_m4_expert_major_glu.py at
                  outputs_per_simd=4 -- same 3,200-threadgroup grid, one weight
                  tile per DISTINCT expert, ~106 registers/thread
  b2 expert-major the same kernel at outputs_per_simd=2 -- 6,400 threadgroups,
                  ~74 registers/thread.  Bit-identical to b4 by construction;
                  it exists because the register pressure b4 pays for holding
                  four rows of accumulators may cost more occupancy than the
                  re-read it saves, and that cannot be reasoned about off-box.
  c  stock        mx.gather_qmm + split + silu + mul, i.e. what the model ran
                  before the retained kernel landed

Correctness bar is bit-exactness with (a): same per-output accumulation order,
so ``differing`` must be 0, not merely small.

Not standalone: unlike ``micro_moe_dedup.py`` this has to import the two kernel
modules under test.  Both are leaf modules whose only dependency is
``mlx.core``, and the import happens inside ``main`` so the CLI surface and the
byte accounting stay unit-testable without MLX.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# See micro_moe_dedup.py: MLX is imported lazily so this file's CLI and its
# pure-python helpers can be tested off the GPU lock.
mx = None
_retained_glu = None
_expert_major_glu = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_id_patterns import (  # noqa: E402
    NUM_EXPERTS,
    ROWS,
    SLOTS,
    TOP_K,
    UNIQUE_CHOICES,
    expert_major_active_entries,
    expert_major_plan,
    load_census_id_sets,
    make_layer_id_sets,
    unique_count,
    validate_expert_major_plan,
)

HIDDEN = 2560
MOE_INTERMEDIATE = 640
GU_OUT = 2 * MOE_INTERMEDIATE
GROUP_SIZE = 32
BITS = 4
LAYERS = 48

VARIANTS = ("a", "b4", "b2", "c")

#: Static per-layer dispatch counts, hand-counted off the graph each variant
#: builds.  The expert-major kernel deliberately recomputes its plan inside the
#: threadgroup instead of building one with stock ops, so it costs the SAME one
#: dispatch as the retained kernel -- the ~17-dispatch stock plan builder would
#: have spent ~1.6 ms/cycle at this box's ~2.0 us/dispatch against a ~2.9
#: ms/cycle prize.  Read alongside micro_dispatch_overhead.py.
DISPATCHES = {"a": 1, "b4": 1, "b2": 1, "c": 5}


def _require_mlx() -> None:
    global mx, _retained_glu, _expert_major_glu
    import mlx.core

    mx = mlx.core
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mtplx.kernels import qwen4_m4_expert_major_glu, qwen4_m4_routed_glu

    _retained_glu = qwen4_m4_routed_glu
    _expert_major_glu = qwen4_m4_expert_major_glu


# --------------------------------------------------------------------------
# byte accounting (pure python; unit testable without MLX)
# --------------------------------------------------------------------------


def gu_bytes_per_expert() -> int:
    """q4/g32 fused gate+up for one expert: packed words + bf16 scales/biases."""

    packed = GU_OUT * (HIDDEN // 8) * 4
    metadata = 2 * GU_OUT * (HIDDEN // GROUP_SIZE) * 2
    return packed + metadata


def tiles_streamed(name: str, uniques: list[int]) -> float:
    """Mean expert weight tiles a variant reads per layer."""

    if name in ("b4", "b2"):
        return float(np.mean(uniques))
    return float(SLOTS)


# --------------------------------------------------------------------------
# adversarial routing decisions
# --------------------------------------------------------------------------


def adversarial_id_sets() -> dict[str, np.ndarray]:
    """Routing decisions that stress the plan's edges, not its average case."""

    all_distinct = np.arange(SLOTS, dtype=np.int32).reshape(ROWS, TOP_K)
    row = np.arange(TOP_K, dtype=np.int32)
    all_same = np.stack([row] * ROWS)
    shared_one = (np.arange(SLOTS, dtype=np.int32) + 1).reshape(ROWS, TOP_K)
    shared_one[:, 0] = 0
    return {
        "all-distinct": all_distinct,
        "all-same": all_same,
        "one-expert-in-four-rows": shared_one,
    }


# --------------------------------------------------------------------------
# variants -- each returns [ROWS, TOP_K, MOE_INTERMEDIATE] bf16 for one layer
# --------------------------------------------------------------------------


def silu(x):
    return x * mx.sigmoid(x)


def run_stock(x, ids, gu):
    xr = mx.expand_dims(x, (-2, -3))  # [4, 1, 1, 2560]
    g = mx.gather_qmm(
        xr,
        gu[0],
        gu[1],
        gu[2],
        rhs_indices=ids,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
        sorted_indices=False,
    )  # [4, 10, 1, 1280]
    gate, up = mx.split(g, 2, axis=-1)
    return (silu(gate) * up).squeeze(-2)


def _quantized_bank(experts: int, out_dim: int, in_dim: int, chunk: int = 64):
    """Stack ``experts`` independently-quantized [out_dim, in_dim] matrices.

    Chunked so the bf16 source never materializes for all 512 at once.  Distinct
    values per expert are what lets the numerics check catch an index bug rather
    than hide it behind identical weights.
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


def build_runner(name):
    if name == "a":
        return _retained_glu.bind()
    if name == "b4":
        return _expert_major_glu.bind(outputs_per_simd=4)
    if name == "b2":
        return _expert_major_glu.bind(outputs_per_simd=2)
    raise ValueError(name)


def build_variant(name, layer_inputs, gu, runners):
    def run():
        outs = []
        for x, ids in layer_inputs:
            if name == "c":
                outs.append(run_stock(x, ids, gu))
            else:
                outs.append(runners[name](x, gu[0], gu[1], gu[2], ids))
        return outs

    return run


def time_variant(run, reps, warmup, clear_cache):
    """``median_ms`` is encode+GPU; ``build_ms`` is the python graph tax."""

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


def _compare(reference, candidate):
    diff = mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
    return {
        "max_abs_diff": float(mx.max(diff).item()),
        "differing_elements": int(mx.sum(candidate != reference).item()),
        "total_elements": int(reference.size),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unique", type=int, default=28, choices=list(UNIQUE_CHOICES))
    p.add_argument(
        "--from-census",
        type=str,
        default=None,
        help="JSON list of real [4, 10] expert-id arrays "
        "(scripts/fable/expert_census_report.py --sample-out)",
    )
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="mx.clear_cache() before each rep (colder buffer pool)",
    )
    p.add_argument("--variants", type=str, default="a,b4,b2,c")
    p.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="skip the all-distinct / all-same / shared-expert id checks",
    )
    p.add_argument("--out", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print(
        "[micro-expert-major] must run under /tmp/mtplx-gpu-exclusive.lock",
        flush=True,
    )
    _require_mlx()
    mx.random.seed(args.seed)

    if args.from_census:
        id_sets = load_census_id_sets(args.from_census, args.layers, seed=args.seed)
        source = f"census:{args.from_census}"
    else:
        id_sets = make_layer_id_sets(args.unique, args.layers, seed=args.seed)
        source = f"synthetic:U={args.unique}"
    uniques = [unique_count(i) for i in id_sets]

    # The kernel's in-threadgroup plan has to agree with the MLX-free reference
    # on every decision this run will replay, or the timing below is timing the
    # wrong amount of work.
    for ids_np in id_sets:
        expert, member = expert_major_plan(ids_np)
        validate_expert_major_plan(ids_np, expert, member)
    leaders = [expert_major_active_entries(i) for i in id_sets]
    assert leaders == uniques, "plan leader count must equal the distinct count"

    per_expert = gu_bytes_per_expert()
    print(
        f"[build] {NUM_EXPERTS} experts, gu [{GU_OUT}, {HIDDEN}] q{BITS} "
        f"g{GROUP_SIZE}, {per_expert / 1e6:.3f} MB/expert",
        flush=True,
    )
    gu = _quantized_bank(NUM_EXPERTS, GU_OUT, HIDDEN)
    assert per_expert == sum(
        int(np.prod(a.shape[1:])) * a.dtype.size for a in gu
    ), "byte model disagrees with the built bank"

    layer_inputs = []
    for ids_np in id_sets:
        x = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
        # uint32 is what mx.argpartition hands the production call site, and
        # mx.fast.metal_kernel specializes on input dtype, so bind the same one.
        ids = mx.array(ids_np.astype(np.uint32))
        mx.eval(x, ids)
        layer_inputs.append((x, ids))

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; choose from {VARIANTS}")
    runners = {n: build_runner(n) for n in names if n != "c"}

    results = {}
    for name in names:
        run = build_variant(name, layer_inputs, gu, runners)
        stats = time_variant(run, args.reps, args.warmup, args.clear_cache)
        streamed = tiles_streamed(name, uniques)
        gb = streamed * per_expert * args.layers / 1e9
        stats.update(
            {
                "expert_tiles_streamed": streamed,
                "gb_moved": gb,
                "gbps": gb / (stats["median_ms"] / 1e3),
                "ms_per_layer": stats["median_ms"] / args.layers,
                "dispatches_per_layer": DISPATCHES[name],
            }
        )
        results[name] = stats

    # numerics: every variant against the RETAINED kernel on layer 0.  The bar
    # for b4/b2 is bit-exactness (differing == 0); c is stock and only has to be
    # close, since it is the thing the retained kernel already replaced.
    if "a" not in names:
        raise SystemExit("variant 'a' is the correctness reference; keep it in")
    x0, ids0 = layer_inputs[0]
    ref = runners["a"](x0, gu[0], gu[1], gu[2], ids0)
    mx.eval(ref)
    numerics = {}
    for name in names:
        if name == "a":
            continue
        got = run_stock(x0, ids0, gu) if name == "c" else runners[name](
            x0, gu[0], gu[1], gu[2], ids0
        )
        mx.eval(got)
        numerics[name] = _compare(ref, got)

    adversarial = {}
    if not args.skip_adversarial:
        xa = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
        mx.eval(xa)
        for label, ids_np in adversarial_id_sets().items():
            expert, member = expert_major_plan(ids_np)
            validate_expert_major_plan(ids_np, expert, member)
            ids = mx.array(ids_np.astype(np.uint32))
            mx.eval(ids)
            ref_a = runners["a"](xa, gu[0], gu[1], gu[2], ids)
            mx.eval(ref_a)
            entry = {"unique": unique_count(ids_np)}
            for name in names:
                if name in ("a", "c"):
                    continue
                got = runners[name](xa, gu[0], gu[1], gu[2], ids)
                mx.eval(got)
                entry[name] = _compare(ref_a, got)
            adversarial[label] = entry

    print(
        f"\nsource={source}  layers={args.layers}  reps={args.reps}  "
        f"mean_unique={np.mean(uniques):.2f}  min/max={min(uniques)}/{max(uniques)}"
    )
    hdr = (
        f"{'var':<4}{'tiles':>7}{'eval/48L':>10}{'p10':>9}{'p90':>9}"
        f"{'ms/layer':>10}{'GB':>8}{'GB/s':>9}{'disp/L':>8}{'build':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        r = results[name]
        print(
            f"{name:<4}{r['expert_tiles_streamed']:>7.1f}{r['median_ms']:>10.3f}"
            f"{r['p10_ms']:>9.3f}{r['p90_ms']:>9.3f}{r['ms_per_layer']:>10.4f}"
            f"{r['gb_moved']:>8.2f}{r['gbps']:>9.1f}"
            f"{r['dispatches_per_layer']:>8d}{r['build_ms']:>9.3f}"
        )
    base = results["a"]["ms_per_layer"]
    print("\nms/layer vs retained (a):")
    for name in names:
        if name == "a":
            continue
        delta = results[name]["ms_per_layer"] / base - 1.0
        print(f"  {name:<3} {100 * delta:+7.2f}%")

    print("\nnumerics vs retained (a)  [b4/b2 bar: differing == 0]:")
    for name, n in numerics.items():
        print(
            f"  {name:<3} max_abs_diff={n['max_abs_diff']:.6g}  "
            f"differing={n['differing_elements']}/{n['total_elements']}"
        )
    if adversarial:
        print("\nadversarial id sets vs retained (a):")
        for label, entry in adversarial.items():
            parts = [f"U={entry['unique']:2d}"]
            for name in names:
                if name in ("a", "c"):
                    continue
                n = entry[name]
                parts.append(
                    f"{name} differing={n['differing_elements']} "
                    f"max={n['max_abs_diff']:.6g}"
                )
            print(f"  {label:<26} " + "  ".join(parts))

    summary = {
        "source": source,
        "layers": args.layers,
        "reps": args.reps,
        "unique_counts": uniques,
        "mean_unique": float(np.mean(uniques)),
        "gu_bytes_per_expert": per_expert,
        "variants": results,
        "numerics": numerics,
        "adversarial": adversarial,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
