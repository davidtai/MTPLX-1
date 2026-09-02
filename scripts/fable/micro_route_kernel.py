#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price the two-kernel MoE routing head against the ten-dispatch scaffold.

WHAT IS BEING PRICED
--------------------
Census item 4 (``G-opdiet-census.md`` §3, and §2.1 rows 35-50): every one of
the 48 MoE blocks spends TEN dispatches producing 40 expert ids, 40 weights and
4 shared-gate scalars.  Eight of them move no weight bytes at all and process
at most 512 numbers; the other two are the q8 router GEMV (1.393 MB/layer,
measured at 294 GB/s) and the one-output shared-gate GEMV.
``mtplx/kernels/qwen4_m4_route.py`` emits the identical tuple in two.

ARMS
----
  stock  the shipped head: block.gate q8 GEMV, precise softmax, argpartition
         top-10, take_along_axis, bf16 renormalise, shared-gate GEMV, sigmoid.
         10 dispatches/layer.
  k1     the route kernel at VEC_LANES=1 -- MLX's own ``qmv_wide`` thread
         layout (8 rows/threadgroup, 64 threads, 4,160 threads total), with
         all four verifier vectors in one lane's registers.
         2 dispatches/layer.
  k4     the route kernel at VEC_LANES=4 -- each verifier vector gets its own
         lane octet (256 threads/threadgroup, 16,640 threads total).  The
         per-vector accumulation sequence is unchanged, so it is bit-identical
         to k1 by construction; it exists because the GEMV's 294 GB/s at
         4,160 threads looks occupancy-bound and the four vec-lanes read the
         SAME weight octet in the same threadgroup on the same cycle, so DRAM
         traffic does not move.  Whether that converts to bandwidth is the
         question this script answers.

CORRECTNESS BAR
---------------
Bit-exactness against ``stock``, on all three emitted tensors, for every layer
-- not a max-abs-diff.  ``expert_ids`` decides WHICH experts run: one flipped
near-tie changes the visible expert set, and the retained routed-down kernel's
fixed SLOT_ORDER reduction tree is validated against the exact slot order, so
even a permutation is a failure.  The parity block therefore reports three
separate counters: layers whose expert SET differs, layers whose set matches
but whose slot ORDER differs, and differing elements per tensor.  Anything
non-zero invalidates the timings above it, and the run says so.

NOT A GO/NO-GO ON ITS OWN
-------------------------
The dispatch saving (-8/layer, -384/cycle) is real regardless of what the GEMV
does; the bandwidth question is only whether k4 beats k1 beats stock's GEMV.
Adoption still gates on an ABBA window with MTPLX_FABLE_ROUTE_KERNEL=1 as the
candidate flag -- this only decides which VEC_LANES that window should arm.

RUN IT (guarded; this file must not be run outside the window)
--------------------------------------------------------------
  PYTHONPATH=<worktree> <worktree>/.venv/bin/python \\
    /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \\
    --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \\
    --lock-timeout-seconds 1800 --child-timeout-seconds 3600 \\
    -- <worktree>/.venv/bin/python \\
       <worktree>/scripts/fable/micro_route_kernel.py \\
       --layers 48 --reps 200 --out <artifacts>/micro-route-kernel.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# MLX is imported lazily (same reason as micro_expert_major.py): the CLI
# surface and the byte model stay checkable without touching Metal.
mx = None
_route = None

REPO_ROOT = Path(__file__).resolve().parents[2]

ROWS = 4
TOP_K = 10
HIDDEN = 2560
NUM_EXPERTS = 512
GROUP_SIZE = 64
BITS = 8
LAYERS = 48

VARIANTS = ("stock", "k1", "k4")
VEC_LANES = {"k1": 1, "k4": 4}

#: Hand-counted off the graph each arm builds; matches census §2.1 rows 35-50.
DISPATCHES = {"stock": 10, "k1": 2, "k4": 2}


def _require_mlx() -> None:
    global mx, _route
    import mlx.core

    mx = mlx.core
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mtplx.kernels import qwen4_m4_route

    _route = qwen4_m4_route


# --------------------------------------------------------------------------
# byte accounting (pure python; checkable without MLX)
# --------------------------------------------------------------------------


def route_bytes_per_layer() -> int:
    """q8/g64 router pack + the folded shared-expert gate, in bytes.

    Every arm streams exactly this once per layer: the stock head as two
    ``affine_qmv_wide`` dispatches, the kernel arms as one.  1,392,640 is the
    census's ``router 1.393 MB``.
    """

    router = NUM_EXPERTS * HIDDEN + 2 * NUM_EXPERTS * (HIDDEN // GROUP_SIZE) * 2
    shared = HIDDEN + 2 * (HIDDEN // GROUP_SIZE) * 2
    return router + shared


# --------------------------------------------------------------------------
# packs
# --------------------------------------------------------------------------


def _q8_pack(out_dim: int, in_dim: int):
    """One independently-quantized q8/g64 projection, bf16 metadata.

    Distinct values per layer are what lets the parity block catch an indexing
    bug instead of hiding it behind repeated weights.
    """

    src = mx.random.normal((out_dim, in_dim)).astype(mx.bfloat16)
    w, s, b = mx.quantize(src, group_size=GROUP_SIZE, bits=BITS)
    pack = (w, s.astype(mx.bfloat16), b.astype(mx.bfloat16))
    mx.eval(*pack)
    del src
    return pack


class _Block:
    """The four attributes ``qwen4_m4_route.stock_route`` reads, no model."""

    class _Proj:
        def __init__(self, pack):
            self.weight, self.scales, self.biases = pack
            self.bits = BITS
            self.group_size = GROUP_SIZE
            self.mode = "affine"

        def __call__(self, x):
            return mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )

    def __init__(self):
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.norm_topk_prob = True
        self.gate = self._Proj(_q8_pack(NUM_EXPERTS, HIDDEN))
        self.shared_expert_gate = self._Proj(_q8_pack(1, HIDDEN))


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def _run_kernel(route, block, x):
    return route(
        x,
        block.gate.weight,
        block.gate.scales,
        block.gate.biases,
        block.shared_expert_gate.weight,
        block.shared_expert_gate.scales,
        block.shared_expert_gate.biases,
    )


def build_variant(name, layers, routes):
    def run():
        outs = []
        for block, x in layers:
            if name == "stock":
                outs.append(_route.stock_route(block, x))
            else:
                outs.append(_run_kernel(routes[name], block, x))
        return outs

    return run


def time_variant(run, reps, warmup, clear_cache):
    """Queued lane: build the whole 48-layer graph, then ONE ``mx.eval``.

    Per-op eager evaluation would host-sync between every dispatch and can
    invert a verdict on kernels this small (see the queued-vs-eager note in the
    bench README).  ``build_ms`` is the python graph tax, reported separately
    because a 10-op arm pays it eight more times per layer than a 2-op arm and
    that cost is real on the verify path.
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


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------


def _parity(reference, candidate):
    """Three counters, because the failure modes are not interchangeable."""

    ref_ids, ref_scores, ref_shared = reference
    got_ids, got_scores, got_shared = candidate
    mx.eval(ref_ids, ref_scores, ref_shared, got_ids, got_scores, got_shared)
    ref_np = np.asarray(ref_ids)
    got_np = np.asarray(got_ids)
    set_differs = not np.array_equal(
        np.sort(ref_np, axis=-1), np.sort(got_np, axis=-1)
    )
    order_differs = (not set_differs) and (not np.array_equal(ref_np, got_np))
    return {
        "expert_set_differs": bool(set_differs),
        "slot_order_differs": bool(order_differs),
        "ids_differing": int(np.sum(ref_np != got_np)),
        "scores_differing": int(mx.sum(got_scores != ref_scores).item()),
        "shared_differing": int(mx.sum(got_shared != ref_shared).item()),
        "scores_max_abs_diff": float(
            mx.max(
                mx.abs(
                    got_scores.astype(mx.float32) - ref_scores.astype(mx.float32)
                )
            ).item()
        ),
    }


def _fold_parity(per_layer):
    return {
        "layers": len(per_layer),
        "layers_expert_set_differs": sum(
            1 for p in per_layer if p["expert_set_differs"]
        ),
        "layers_slot_order_differs": sum(
            1 for p in per_layer if p["slot_order_differs"]
        ),
        "ids_differing": sum(p["ids_differing"] for p in per_layer),
        "scores_differing": sum(p["scores_differing"] for p in per_layer),
        "shared_differing": sum(p["shared_differing"] for p in per_layer),
        "scores_max_abs_diff": max(
            (p["scores_max_abs_diff"] for p in per_layer), default=0.0
        ),
    }


def parity_is_clean(folded) -> bool:
    return (
        folded["layers_expert_set_differs"] == 0
        and folded["layers_slot_order_differs"] == 0
        and folded["ids_differing"] == 0
        and folded["scores_differing"] == 0
        and folded["shared_differing"] == 0
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="mx.clear_cache() before each rep (colder buffer pool)",
    )
    p.add_argument("--variants", type=str, default="stock,k1,k4")
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the byte model; touch no GPU",
    )
    return p


def _plan(args, names):
    per_layer = route_bytes_per_layer()
    return {
        "layers": args.layers,
        "reps": args.reps,
        "variants": names,
        "bytes_per_layer": per_layer,
        "mb_per_cycle": per_layer * args.layers / 1e6,
        "dispatches_per_cycle": {
            n: DISPATCHES[n] * args.layers for n in names
        },
        "dispatches_removed_per_cycle": (
            DISPATCHES["stock"] - DISPATCHES["k1"]
        )
        * args.layers,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; choose from {VARIANTS}")
    if "stock" not in names:
        raise SystemExit("variant 'stock' is the correctness reference; keep it in")

    plan = _plan(args, names)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    print(
        "[micro-route-kernel] must run under /tmp/mtplx-gpu-exclusive.lock",
        flush=True,
    )
    _require_mlx()
    mx.random.seed(args.seed)

    assert route_bytes_per_layer() == _route.router_bytes_per_layer(), (
        "byte model disagrees with the kernel module's own accounting"
    )

    print(
        f"[build] {args.layers} layers, router [{NUM_EXPERTS}, {HIDDEN}] "
        f"q{BITS} g{GROUP_SIZE}, "
        f"{plan['bytes_per_layer'] / 1e6:.3f} MB/layer",
        flush=True,
    )
    layers = []
    for index in range(args.layers):
        block = _Block()
        _route.check_contract(block, index=index)
        x = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)
        _route.check_input(x)
        layers.append((block, x))

    routes = {n: _route.bind(vec_lanes=VEC_LANES[n]) for n in names if n != "stock"}

    results = {}
    for name in names:
        run = build_variant(name, layers, routes)
        stats = time_variant(run, args.reps, args.warmup, args.clear_cache)
        gb = plan["bytes_per_layer"] * args.layers / 1e9
        stats.update(
            {
                "gb_moved": gb,
                "gbps": gb / (stats["median_ms"] / 1e3),
                "us_per_layer": stats["median_ms"] * 1e3 / args.layers,
                "dispatches_per_layer": DISPATCHES[name],
                "dispatches_per_cycle": DISPATCHES[name] * args.layers,
            }
        )
        results[name] = stats

    parity = {}
    for name in names:
        if name == "stock":
            continue
        per_layer = []
        for block, x in layers:
            per_layer.append(
                _parity(
                    _route.stock_route(block, x),
                    _run_kernel(routes[name], block, x),
                )
            )
        parity[name] = _fold_parity(per_layer)

    print(
        f"\nlayers={args.layers}  reps={args.reps}  "
        f"{plan['mb_per_cycle']:.1f} MB/cycle"
    )
    hdr = (
        f"{'arm':<7}{'eval/48L':>11}{'p10':>9}{'p90':>9}{'us/layer':>10}"
        f"{'GB/s':>9}{'disp/L':>8}{'disp/cyc':>10}{'build':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        r = results[name]
        print(
            f"{name:<7}{r['median_ms']:>11.3f}{r['p10_ms']:>9.3f}"
            f"{r['p90_ms']:>9.3f}{r['us_per_layer']:>10.2f}{r['gbps']:>9.1f}"
            f"{r['dispatches_per_layer']:>8}{r['dispatches_per_cycle']:>10}"
            f"{r['build_ms']:>9.3f}"
        )

    print("\nparity vs stock (bar is bit-exact: every counter 0)")
    for name, folded in parity.items():
        verdict = "EXACT" if parity_is_clean(folded) else "*** NOT EXACT ***"
        print(
            f"  {name:<4} {verdict}  set_diff={folded['layers_expert_set_differs']}"
            f"/{folded['layers']}  order_diff={folded['layers_slot_order_differs']}"
            f"/{folded['layers']}  ids={folded['ids_differing']}"
            f"  scores={folded['scores_differing']}"
            f"  shared={folded['shared_differing']}"
            f"  score_dmax={folded['scores_max_abs_diff']:.3e}"
        )

    dirty = [n for n, f in parity.items() if not parity_is_clean(f)]
    if dirty:
        print(
            "\n!! the timings above are NOT a verdict: "
            f"{', '.join(dirty)} changed the routing decision.",
        )

    payload = {"plan": plan, "results": results, "parity": parity, "seed": args.seed}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 1 if dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
