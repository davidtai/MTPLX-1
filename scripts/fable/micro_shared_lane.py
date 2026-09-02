#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price the physical-M4 shared expert, and the second-stream lane that hides it.

WHY THIS EXISTS
---------------
Program row 9 ("shared-expert lane") is costed off the retained-stack census's
``MoE shared`` family: 251.7 MB/cycle at 135-203 GB/s, 1.24-1.86 ms of a 37.4 ms
window.  That rate is **not a measurement**.  The instrumented MLX build records
per-command-buffer GPU intervals only; ``scripts/fable/census_retained_stack.py``
fits four global coefficients (cb floor, per-dispatch ns, weight ns/B,
activation ns/B) by NNLS and then splits each buffer's measured duration across
its ops in proportion to their *modelled* cost.  The resulting GB/s column is
therefore monotone in bytes-per-dispatch, not in memory efficiency -- and the
control and composed censuses report this family with identical dispatch counts
(144.0/cyc) and identical bytes (251.7 MB/cyc) but 203 vs 135 GB/s, a 50 % swing
that can only come from the fit.

The census also cannot bracket the shared branch by hand.  Exactly one command
buffer in the whole capture isolates any of it -- 381 occurrences (1.0/cycle) of

    Compiled Divide(10,4,1) | v_Sigmoid(4,1,1) | affine_qmv_wide gs64_b8
    (1,320,1) | g2_copy uint32(10,4,1) | CK paired_routed_glu(5120,40,1)

at a 139.62 us median (p10 115.88, p90 158.12).  That is the shared *down*
matvec plus the paired routed GLU, and the GLU's own byte-driven spread is
wider than the shared matvec being solved for.  Depending on what the GLU's
81.92 MB of q4 lanes achieve (600-800 GB/s is the plausible band), the shared
down projection lands anywhere in 0-35 us per layer.  Nothing in the capture
narrows it further.

So the first job of this script is the number nobody has: what the two shared
q8/group-64 matvecs actually cost, isolated, on the queued lane, at the real
shapes.  The second job is whether ``MTPLX_FABLE_SHARED_LANE`` -- which runs the
branch on a second ``mx.gpu`` stream so it overlaps the routed kernels -- beats
the two cross-stream fences MLX charges for the privilege.

WHAT THE SHARED BRANCH IS (measured, w58 retained-control census, seq
400045-400062, one MoE layer)
--------------------------------------------------------------------
    affine_qmv_wide gs_64_b_8 (1, 160, 1)   gate/up  N=1280 K=2560   3.482 MB
    Compiled Sigmoid/Mul/Mul  (640, 4, 1)   split + SiLU(gate) * up  ~0 MB
    affine_qmv_wide gs_64_b_8 (1, 320, 1)   down     N=2560 K=640    1.741 MB

Three dependent dispatches -> at least three barrier waves in MLX's concurrent
encoder, which fences the whole encoder on every dependency
(``mlx/backend/metal/device.cpp::maybeInsertBarrier``).  Both grids are small
(160 and 320 threadgroups of 64 threads) and the down projection's K is only
640, i.e. ~85 bytes of weights per thread -- launch-ramp territory, not
bandwidth territory.  That is the physical reason to suspect the branch costs
more than its 8.7 us/layer byte floor, and the reason the fix is scheduling
rather than fusion.

WHY NOT FUSION (both directions are closed, and this script does not re-test
them)
----------------------------------------------------------------------------
* Into the paired routed GLU: that kernel is hard-specialized to affine
  q4/group-32 (``GROUP_SIZE = 32``, ``load_q4_vector``, ``qdot_q4``,
  ``WEIGHT_BYTES_PER_ROW = HIDDEN/2``).  The shared expert is q8/group-64.  An
  "eleventh lane" is a second dequant path, not an extra lane.
* Into one gate/up -> SiLU -> down kernel: the down projection needs the whole
  ``[4, 640]`` activation before any output row starts, so one dispatch means
  either a grid-wide barrier or one threadgroup per row -- four threadgroups on
  a 40-core GPU, re-reading the 3.482 MB pack four times.  That is the
  ``fused_hyper_read`` (1024, S, 1) failure, 13 tok/s.
* Into the routed-down/residual tail: measured, +34 % on the component
  (PR391 ledger); and packing the scalar gate into the gate/up projection:
  measured, +36 ms decode (``docs/perf/pr391-m4-shared-gate-pack-result.md``).

ARMS
----
  shared_gu      48 x the gate/up matvec alone
  shared_down    48 x the down matvec alone (from a resident [4,640] input)
  shared_branch  48 x the full three-dispatch chain
  routed         48 x {paired routed GLU -> routed-down reduce -> residual
                 tail} with a resident constant shared_down: the routed lane
                 with no shared branch in the graph at all
  stock          48 x the shipped MoE tail: routed + the shared branch on the
                 default stream
  lane           48 x the same with the shared branch on a second gpu stream

READ IT LIKE THIS
-----------------
  stock - routed   = what the shared branch costs the window TODAY.  This is
                     the ceiling on anything row 9 can win.  If it is near the
                     8.7 us/layer byte floor, close row 9.
  lane  - routed   = what it costs with the lane.
  lane  - stock    = the number an ABBA window should reproduce, once per
                     verify cycle.  Positive means the fences beat the overlap.
  shared_branch    = the isolated serial cost, for comparison with the census's
                     apportioned 1.24-1.86 ms/cycle claim.

KNOWN BIAS (state it in any writeup)
------------------------------------
All 48 harness layers share ONE 512-expert q4 routed bank (~1.57 GB) instead of
48 distinct banks (~67 GB, which will not fit).  The routed kernels therefore
run warmer than production, so the window the shared branch can hide inside is
SMALLER here than in the model.  This biases the lane's measured benefit DOWN;
a win here is real, a loss here is not conclusive on its own.  The shared packs
are distinct per layer, so the shared arms carry no such bias.

CORRECTNESS BAR
---------------
Bit-exactness, not max-abs-diff, on every layer's tail output and on every
layer's ``shared_down``.  The lane emits the identical ops with the identical
arguments and only changes the stream they are recorded on, so any difference
at all is a defect and invalidates every timing above it.

RUN IT (guarded; this file must not be run outside the window)
--------------------------------------------------------------
  PYTHONPATH=<worktree> <worktree>/.venv/bin/python \\
    /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \\
    --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \\
    --lock-timeout-seconds 1800 --child-timeout-seconds 3600 \\
    -- <worktree>/.venv/bin/python \\
       <worktree>/scripts/fable/micro_shared_lane.py \\
       --layers 48 --reps 200 \\
       --out <artifacts>/micro-shared-lane.json

``--plan`` prints the byte/memory model and exits without importing MLX.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# MLX is imported lazily (same reason as micro_route_kernel.py): the CLI
# surface and the byte model stay checkable without touching Metal.
mx = None
_shared_lane = None
_routed_glu = None
_routed_down = None

REPO_ROOT = Path(__file__).resolve().parents[2]

ROWS = 4
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640
NUM_EXPERTS = 512
LAYERS = 48

SHARED_BITS = 8
SHARED_GROUP = 64
ROUTED_BITS = 4
ROUTED_GROUP = 32

ARMS = (
    "shared_gu",
    "shared_down",
    "shared_branch",
    "routed",
    "stock",
    "lane",
)

#: Dispatches each arm issues per layer, hand-counted off the graph it builds
#: and cross-checked against the census stream quoted in the docstring.
DISPATCHES_PER_LAYER = {
    "shared_gu": 1,
    "shared_down": 1,
    "shared_branch": 3,
    "routed": 3,
    "stock": 6,
    # Same six, plus MLX's fence_update/fence_wait pair in each direction.
    "lane": 10,
}


def _require_mlx() -> None:
    global mx, _shared_lane, _routed_glu, _routed_down
    import mlx.core

    mx = mlx.core
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mtplx.kernels import qwen4_m4_routed_down, qwen4_m4_routed_glu
    from mtplx.kernels import qwen4_m4_shared_lane

    _shared_lane = qwen4_m4_shared_lane
    _routed_glu = qwen4_m4_routed_glu
    _routed_down = qwen4_m4_routed_down


# --------------------------------------------------------------------------
# byte / memory model (pure python; checkable without MLX)
# --------------------------------------------------------------------------


def _affine_bytes(out_dim: int, in_dim: int, bits: int, group: int) -> int:
    """Packed weights plus one bf16 scale and one bf16 bias per group."""

    return out_dim * in_dim * bits // 8 + 2 * 2 * out_dim * (in_dim // group)


def shared_bytes_per_layer() -> dict[str, int]:
    gu = _affine_bytes(2 * INTERMEDIATE, HIDDEN, SHARED_BITS, SHARED_GROUP)
    down = _affine_bytes(HIDDEN, INTERMEDIATE, SHARED_BITS, SHARED_GROUP)
    return {"shared_gu": gu, "shared_down": down, "shared_branch": gu + down}


def routed_bytes_per_layer() -> int:
    """The 40 selected (row, expert) lanes, counted as issued (not unique).

    Duplicate experts are absorbed by the SLC, so this is an upper bound on
    DRAM traffic and a lower bound on achieved GB/s.
    """

    lanes = ROWS * TOP_K
    gu = _affine_bytes(2 * INTERMEDIATE, HIDDEN, ROUTED_BITS, ROUTED_GROUP)
    down = _affine_bytes(HIDDEN, INTERMEDIATE, ROUTED_BITS, ROUTED_GROUP)
    return lanes * (gu + down)


def bank_bytes() -> dict[str, int]:
    """Resident weight footprint of the harness."""

    routed = NUM_EXPERTS * (
        _affine_bytes(2 * INTERMEDIATE, HIDDEN, ROUTED_BITS, ROUTED_GROUP)
        + _affine_bytes(HIDDEN, INTERMEDIATE, ROUTED_BITS, ROUTED_GROUP)
    )
    shared = LAYERS * shared_bytes_per_layer()["shared_branch"]
    return {"routed_bank": routed, "shared_packs": shared, "total": routed + shared}


def plan(layers: int) -> dict:
    shared = shared_bytes_per_layer()
    return {
        "layers": layers,
        "rows": ROWS,
        "top_k": TOP_K,
        "shared_bytes_per_layer": shared,
        "shared_bytes_per_cycle": {
            name: value * layers for name, value in shared.items()
        },
        "routed_issued_bytes_per_layer": routed_bytes_per_layer(),
        "dispatches_per_layer": DISPATCHES_PER_LAYER,
        "resident_bank_bytes": bank_bytes(),
        "byte_floor_us_per_layer_at_600GBs": round(
            shared["shared_branch"] / 600e9 * 1e6, 3
        ),
        "byte_floor_ms_per_cycle_at_600GBs": round(
            shared["shared_branch"] * layers / 600e9 * 1e3, 4
        ),
        "census_apportioned_ms_per_cycle": {"control": 1.238, "composed": 1.859},
        "census_apportioned_GBs": {"control": 203, "composed": 135},
        "note": (
            "the census GB/s is a fitted apportionment, not a measurement; "
            "this run replaces it"
        ),
    }


# --------------------------------------------------------------------------
# packs
# --------------------------------------------------------------------------


def _quantize(src, bits: int, group: int):
    w, s, b = mx.quantize(src, group_size=group, bits=bits)
    pack = (w, s.astype(mx.bfloat16), b.astype(mx.bfloat16))
    mx.eval(*pack)
    return pack


class _Proj:
    """The attribute surface ``mx.quantized_matmul`` callers read, no model."""

    def __init__(self, pack, bits: int, group: int):
        self.weight, self.scales, self.biases = pack
        self.bits = bits
        self.group_size = group
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


class _SharedExpert:
    """The five attributes ``qwen4_m4_shared_lane._emit_branch`` reads."""

    def __init__(self):
        gu = _quantize(
            mx.random.normal((2 * INTERMEDIATE, HIDDEN)).astype(mx.bfloat16),
            SHARED_BITS,
            SHARED_GROUP,
        )
        self.gu_weight, self.gu_scales, self.gu_biases = gu
        self.bits = SHARED_BITS
        self.group_size = SHARED_GROUP
        self.mode = "affine"
        self.down_proj = _Proj(
            _quantize(
                mx.random.normal((HIDDEN, INTERMEDIATE)).astype(mx.bfloat16),
                SHARED_BITS,
                SHARED_GROUP,
            ),
            SHARED_BITS,
            SHARED_GROUP,
        )


class _Block:
    def __init__(self, shared):
        self.shared_expert = shared


#: Experts quantized per pass when building the routed bank.  A whole
#: [512, 1280, 2560] fp32 normal is 6.7 GB of transient before the cast; at 32
#: experts the transient is ~0.42 GB and the peak is dominated by the packs
#: themselves (~1.57 GB).  The guard cares about peak, so this is not cosmetic.
BANK_CHUNK = 32


def _quantize_bank(experts: int, out_dim: int, in_dim: int, bits, group):
    weights, scales, biases = [], [], []
    for start in range(0, experts, BANK_CHUNK):
        count = min(BANK_CHUNK, experts - start)
        src = mx.random.normal((count, out_dim, in_dim)).astype(mx.bfloat16)
        mx.eval(src)
        w, s, b = _quantize(src, bits, group)
        del src
        weights.append(w)
        scales.append(s)
        biases.append(b)
    pack = (
        mx.concatenate(weights, axis=0),
        mx.concatenate(scales, axis=0),
        mx.concatenate(biases, axis=0),
    )
    mx.eval(*pack)
    del weights, scales, biases
    return pack


class _RoutedBank:
    """One 512-expert q4/g32 bank, shared by every harness layer.

    See KNOWN BIAS in the module docstring: this makes the routed kernels
    warmer than production and therefore biases the lane's benefit DOWN.
    """

    def __init__(self):
        self.gu_weight, self.gu_scales, self.gu_biases = _quantize_bank(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, ROUTED_BITS, ROUTED_GROUP
        )
        self.down_weight, self.down_scales, self.down_biases = _quantize_bank(
            NUM_EXPERTS, HIDDEN, INTERMEDIATE, ROUTED_BITS, ROUTED_GROUP
        )


def build_layers(layer_count: int, seed: int):
    """One resident routed bank, ``layer_count`` distinct shared packs."""

    mx.random.seed(seed)
    bank = _RoutedBank()
    layers = []
    for index in range(layer_count):
        mx.random.seed(seed + 1 + index)
        block = _Block(_SharedExpert())
        x = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
        # Distinct, deterministic, in-range expert ids per layer; duplicates
        # across the 40 lanes are exactly what production sees (the census puts
        # the real overlap near 28 distinct experts per 40-lane layer).
        ids = (
            mx.random.randint(0, NUM_EXPERTS, (ROWS, TOP_K))
            .astype(mx.uint32)
        )
        scores = mx.random.uniform(shape=(ROWS, TOP_K)).astype(mx.bfloat16)
        factor = mx.random.uniform(shape=(ROWS,)).astype(mx.bfloat16)
        hyper = mx.random.normal((ROWS, 4 * HIDDEN)).astype(mx.bfloat16)
        inject = mx.random.normal((ROWS, 4)).astype(mx.bfloat16)
        # A resident constant for the "routed" arm, so that arm's graph
        # contains no shared branch at all.
        const_shared_down = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
        # A resident [4,640] activation for the "shared_down" arm, so that arm
        # measures the matvec and not the gate/up matvec feeding it.
        const_shared_h = mx.random.normal((ROWS, INTERMEDIATE)).astype(
            mx.bfloat16
        )
        mx.eval(
            x, ids, scores, factor, hyper, inject, const_shared_down,
            const_shared_h,
        )
        layers.append(
            {
                "block": block,
                "x": x,
                "ids": ids,
                "scores": scores,
                "factor": factor,
                "hyper": hyper,
                "inject": inject,
                "const_shared_down": const_shared_down,
                "const_shared_h": const_shared_h,
            }
        )
    return bank, layers


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def _shared_gu(block, x):
    shared = block.shared_expert
    return mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )


def _routed_tail(glu, tail, bank, layer, shared_down):
    routed_h = glu(
        layer["x"],
        bank.gu_weight,
        bank.gu_scales,
        bank.gu_biases,
        layer["ids"],
    )
    return tail(
        routed_h,
        bank.down_weight,
        bank.down_scales,
        bank.down_biases,
        layer["ids"],
        layer["scores"],
        shared_down,
        layer["factor"],
        layer["hyper"],
        layer["inject"],
    )


def build_arm(name, glu, tail, bank, layers):
    def run():
        outs = []
        for layer in layers:
            block, x = layer["block"], layer["x"]
            if name == "shared_gu":
                outs.append(_shared_gu(block, x))
            elif name == "shared_down":
                outs.append(
                    block.shared_expert.down_proj(layer["const_shared_h"])
                )
            elif name == "shared_branch":
                outs.append(_shared_lane.stock_shared_branch(block, x))
            elif name == "routed":
                outs.append(
                    _routed_tail(
                        glu, tail, bank, layer, layer["const_shared_down"]
                    )
                )
            elif name == "stock":
                shared_down = _shared_lane.stock_shared_branch(block, x)
                outs.append(_routed_tail(glu, tail, bank, layer, shared_down))
            elif name == "lane":
                shared_down = _shared_lane.shared_branch(block, x)
                outs.append(_routed_tail(glu, tail, bank, layer, shared_down))
            else:  # pragma: no cover - guarded by the CLI choices
                raise ValueError(f"unknown arm {name!r}")
        return outs

    return run


def time_arm(run, reps, warmup, clear_cache):
    """Queued lane: build the whole N-layer graph, then ONE ``mx.eval``.

    Per-op eager evaluation host-syncs between dispatches and can invert a
    verdict on kernels this small (queued-vs-eager note in the bench README).
    ``build_ms`` is reported separately because it is real host cost on the
    verify path and the lane's arm pays a ``mx.stream`` context per layer.
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


def parity(glu, tail, bank, layers):
    """Bit-exactness of the lane against the stock branch, per layer.

    Two independent checks: the branch's own output, and the tail output that
    consumes it.  Both must be array-equal on every layer; the lane changes no
    operand, so anything else is a defect, not a tolerance question.
    """

    branch_mismatches = []
    tail_mismatches = []
    for index, layer in enumerate(layers):
        block, x = layer["block"], layer["x"]
        want_branch = _shared_lane.stock_shared_branch(block, x)
        got_branch = _shared_lane.shared_branch(block, x)
        want_tail = _routed_tail(glu, tail, bank, layer, want_branch)
        got_tail = _routed_tail(glu, tail, bank, layer, got_branch)
        same_branch = mx.array_equal(want_branch, got_branch)
        same_tail = mx.array_equal(want_tail, got_tail)
        mx.eval(same_branch, same_tail)
        if not bool(same_branch.item()):
            branch_mismatches.append(index)
        if not bool(same_tail.item()):
            tail_mismatches.append(index)
    return {
        "layers": len(layers),
        "shared_down_mismatched_layers": branch_mismatches,
        "tail_mismatched_layers": tail_mismatches,
        "bit_exact": not branch_mismatches and not tail_mismatches,
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def derive(timings: dict, layer_count: int) -> dict:
    """The four differences that decide row 9."""

    def us_per_layer(ms):
        return ms * 1e3 / layer_count

    stock = timings["stock"]["median_ms"]
    lane = timings["lane"]["median_ms"]
    routed = timings["routed"]["median_ms"]
    branch = timings["shared_branch"]["median_ms"]
    shared = shared_bytes_per_layer()

    def gbs(ms, bytes_per_layer):
        seconds = ms / 1e3
        return (bytes_per_layer * layer_count) / seconds / 1e9 if seconds else 0.0

    return {
        "exposed_shared_cost_today_ms": stock - routed,
        "exposed_shared_cost_today_us_per_layer": us_per_layer(stock - routed),
        "exposed_shared_cost_with_lane_ms": lane - routed,
        "exposed_shared_cost_with_lane_us_per_layer": us_per_layer(lane - routed),
        "lane_delta_ms_per_cycle": lane - stock,
        "isolated_shared_branch_ms": branch,
        "isolated_shared_branch_us_per_layer": us_per_layer(branch),
        "achieved_GBs": {
            "shared_gu": gbs(
                timings["shared_gu"]["median_ms"], shared["shared_gu"]
            ),
            "shared_down": gbs(
                timings["shared_down"]["median_ms"], shared["shared_down"]
            ),
            "shared_branch": gbs(branch, shared["shared_branch"]),
            "routed_issued": gbs(routed, routed_bytes_per_layer()),
        },
        "byte_floor_ms_at_600GBs": shared["shared_branch"] * layer_count / 600e9 * 1e3,
        "verdict_rule": (
            "close row 9 if exposed_shared_cost_today is within ~2x of "
            "byte_floor_ms_at_600GBs; promote the lane only if "
            "lane_delta_ms_per_cycle is negative by more than the arm's p10-p90 "
            "spread AND parity.bit_exact is true"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="mx.clear_cache() before every rep (colder, higher variance)",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"comma-separated subset of {','.join(ARMS)}",
    )
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the byte/memory model and exit without importing MLX",
    )
    args = parser.parse_args(argv)

    if args.layers < 1:
        parser.error("--layers must be >= 1")
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {unknown}; choose from {list(ARMS)}")

    if args.plan:
        print(json.dumps(plan(args.layers), indent=2, sort_keys=True))
        return 0

    _require_mlx()
    bank, layers = build_layers(args.layers, args.seed)
    glu = _routed_glu.bind()
    tail = _routed_down.bind_residual_tail()

    report = {
        "script": "micro_shared_lane",
        "layers": args.layers,
        "reps": args.reps,
        "seed": args.seed,
        "clear_cache": bool(args.clear_cache),
        "plan": plan(args.layers),
    }

    if not args.skip_parity:
        report["parity"] = parity(glu, tail, bank, layers)

    timings = {}
    for name in arms:
        run = build_arm(name, glu, tail, bank, layers)
        timings[name] = time_arm(run, args.reps, args.warmup, args.clear_cache)
        timings[name]["dispatches_per_layer"] = DISPATCHES_PER_LAYER[name]
    report["timings"] = timings
    if set(("shared_gu", "shared_down", "shared_branch", "routed", "stock", "lane")) <= set(
        timings
    ):
        report["derived"] = derive(timings, args.layers)
    report["counters"] = dict(_shared_lane.COUNTERS)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    parity_ok = report.get("parity", {}).get("bit_exact", True)
    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
