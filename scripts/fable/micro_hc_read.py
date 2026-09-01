#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price the Qwen3.8 Flash-Next gated-residual READ at the fixed-M4 verify width.

``GatedResidual.__call__`` is read 97 times per verify forward (2 per layer x 48
+ the trunk mixer).  Eagerly that is 11 dispatches and 13.21 MB of bf16 mix
weights per read: 1,067 dispatches and ~1.28 GB per cycle -- simultaneously the
largest zero-byte dispatch family in the compiled verify graph AND its largest
non-MoE byte consumer.  The census puts the ``[320, 10240]`` down GEMV at
~385 GB/s, well under the box's ceiling.

Variants, all over one full 97-read cycle:

  a   eager      the real ``GatedResidual`` module, fused paths forced off --
                 this is what the compiled verifier runs today
  b   m4         ``kernels.qwen4_m4_hyper_read.fused_hc_read_m4``: 3 dispatches,
                 threadgroups over output columns, every weight element read
                 once per call regardless of R
  bn  m4-norm    stage K0 only (grouped rms + gamma).  Timed so the down/up
                 stage times below are differences, not guesses.
  bd  m4-down    stages K0+K1 (norm + folded down/inject GEMV)
  c   v1-fused   ``kernels.hyper_connection.fused_hyper_read``, grid (1024,S,1):
                 one threadgroup per row, so at S=4 it runs 4 threadgroups and
                 re-reads all 13.21 MB once PER ROW.  Reference, not a
                 candidate -- measured 13.2 tok/s vs 67.8 control on the M4
                 verifier 2026-09-01.

Derived, printed in the stage table:

  down GB/s = calls * (wd + wi bytes) / (t_bd - t_bn)
  up   GB/s = calls * wu bytes        / (t_b  - t_bd)

Weights are synthetic but at the real shapes and, by default, DISTINCT per
call: the 97 reads in a real cycle own private weights and reuse nothing, so a
shared-weight bench would measure an L2 hit that does not exist (1.28 GB per
cycle is the point).

Standalone-ish: MLX and mtplx are imported lazily by ``main`` so the CLI
surface can be unit-tested off-GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Family geometry (qwen4_exp TextArgs defaults).
HC_COUNT = 4
HIDDEN = 2560
HCD = HC_COUNT * HIDDEN  # 10240
LOWRANK = 320
RMS_EPS = 1e-6

#: 2 reads per layer x 48 layers + the trunk mixer.  The mixer is the one
#: read built with ``use_combine=False`` (no block_inject_weight).
CALLS_PER_CYCLE = 97
NONCOMBINE_CALLS = 1

#: Physical verify width of the fixed-M4 lane.
ROWS = 4

#: Dispatches per read, hand-counted off the ops each path builds.
#:
#: eager (qwen4_exp.py GatedResidual.__call__ + GroupedRMSNorm.__call__):
#:   rms_norm, normed*gamma, down gemv, silu(x/4) [fused divide+sigmoid+mul],
#:   up gemv, sigmoid, mix*grouped, hc-mean sum, mean scale, inject gemv,
#:   2*sigmoid(x/4)  = 11.  The census measures exactly 11 x 97 = 1,067/cycle.
DISPATCHES = {"a": 11, "b": 3, "bn": 1, "bd": 2, "c": 1}

mx = None
_qwen4_exp = None
_hcm4 = None
_hc_v1 = None


def _require_mlx() -> None:
    """Import MLX and mtplx, with every fused hyper path forced OFF.

    Variant (a) must be the eager chain, so the two env-read gates are pinned
    to 0 BEFORE ``qwen4_exp`` is imported and the import-time
    ``MTPLX_FABLE_HC_M4`` constant is pinned False after.  Otherwise a shell
    that happens to have one armed silently benchmarks a kernel against
    itself.
    """

    global mx, _qwen4_exp, _hcm4, _hc_v1
    os.environ["MTPLX_FUSED_HC"] = "0"
    os.environ["MTPLX_FUSED_HC_V3"] = "0"
    os.environ.setdefault("MTPLX_FABLE_HC_M4", "0")
    import mlx.core

    mx = mlx.core
    import mtplx.runtime_options as runtime_options

    runtime_options._FABLE_HC_M4 = False
    import mtplx.models.qwen4_exp as qwen4_exp
    from mtplx.kernels import hyper_connection, qwen4_m4_hyper_read

    _qwen4_exp = qwen4_exp
    _hcm4 = qwen4_m4_hyper_read
    _hc_v1 = hyper_connection


def weight_bytes(combine: bool, itemsize: int = 2) -> dict:
    """Per-read weight bytes, broken out by stage."""

    gamma = HCD * itemsize
    down = LOWRANK * HCD * itemsize
    up = HCD * LOWRANK * itemsize
    inject = (HC_COUNT * HCD * itemsize) if combine else 0
    return {
        "gamma": gamma,
        "down": down,
        "up": up,
        "inject": inject,
        "total": gamma + down + up + inject,
    }


def cycle_bytes(calls: int, noncombine: int, itemsize: int = 2) -> dict:
    """Weight bytes for a whole cycle, by stage."""

    nc = min(noncombine, calls)
    a = weight_bytes(True, itemsize)
    b = weight_bytes(False, itemsize)
    return {
        k: (calls - nc) * a[k] + nc * b[k]
        for k in ("gamma", "down", "up", "inject", "total")
    }


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def build_reads(calls: int, noncombine: int, rows: int, distinct: bool, seed: int):
    """One ``GatedResidual`` per read, weights assigned from the same arrays
    the kernel variants are handed, so all three see identical bits."""

    args = _qwen4_exp.TextArgs()
    assert args.hc_count == HC_COUNT and args.hidden_size == HIDDEN
    assert args.hc_lowrank == LOWRANK

    mx.random.seed(seed)
    reads = []
    shared = None
    for i in range(calls):
        combine = i < (calls - min(noncombine, calls))
        if distinct or shared is None:
            gamma = mx.random.normal((HCD,)).astype(mx.bfloat16)
            wd = (mx.random.normal((LOWRANK, HCD)) * 0.02).astype(mx.bfloat16)
            wu = (mx.random.normal((HCD, LOWRANK)) * 0.02).astype(mx.bfloat16)
            wi = (mx.random.normal((HC_COUNT, HCD)) * 0.02).astype(mx.bfloat16)
            mx.eval(gamma, wd, wu, wi)
            if not distinct:
                shared = (gamma, wd, wu, wi)
        else:
            gamma, wd, wu, wi = shared
        x = mx.random.normal((rows, HCD)).astype(mx.bfloat16)
        mx.eval(x)

        mod = _qwen4_exp.GatedResidual(args, use_combine=combine)
        mod.hc_norm.weight = gamma
        mod.input_mix_weight_down.weight = wd
        mod.input_mix_weight_up.weight = wu
        if combine:
            mod.block_inject_weight.weight = wi
        reads.append(
            {
                "combine": combine,
                "module": mod,
                "x": x,
                "gamma": gamma,
                "wd": wd,
                "wu": wu,
                "wi": wi if combine else None,
            }
        )
    return reads


# --------------------------------------------------------------------------
# variants -- each returns the list of arrays a cycle produces
# --------------------------------------------------------------------------


def run_eager(reads):
    outs = []
    for rd in reads:
        got = rd["module"](rd["x"])
        outs.append(got[0] if rd["combine"] else got)
        if rd["combine"]:
            outs.append(got[2])
    return outs


def run_m4(reads, tune, stage="b"):
    """stage "b" goes through the shipping entry point; "bn"/"bd" reach past
    it into the individual kernels so the stage table can be a difference of
    two measured cycles instead of a guess."""

    if stage == "b":
        outs = []
        for rd in reads:
            mixed, inject = _hcm4.fused_hc_read_m4(
                rd["x"], rd["gamma"], rd["wd"], rd["wu"], rd["wi"],
                eps=RMS_EPS, **tune,
            )
            outs.append(mixed)
            if inject is not None:
                outs.append(inject)
        return outs
    outs = []
    for rd in reads:
        dt = rd["x"].dtype
        has_inject = rd["wi"] is not None
        normed = _hcm4._kernel_norm(_hcm4._eps_bits(RMS_EPS))(
            inputs=[rd["x"], rd["gamma"]],
            template=[("T", dt), ("NTHREADS", tune["norm_threads"])],
            grid=(tune["norm_threads"] * rd["x"].shape[0] * HC_COUNT, 1, 1),
            threadgroup=(tune["norm_threads"], 1, 1),
            output_shapes=[(rd["x"].shape[0], HCD)],
            output_dtypes=[dt],
        )[0]
        if stage == "bn":
            outs.append(normed)
            continue
        n_tot = (LOWRANK + HC_COUNT) if has_inject else LOWRANK
        n_blk = (n_tot + tune["out_per_tg"] - 1) // tune["out_per_tg"]
        mixv, inject = _hcm4._kernel_down()(
            inputs=[normed, rd["wd"], rd["wi"] if has_inject else rd["wd"]],
            template=[
                ("T", dt),
                ("ROWS", int(rd["x"].shape[0])),
                ("NTHREADS", tune["down_threads"]),
                ("OUT_PER_TG", tune["out_per_tg"]),
                ("HAS_INJECT", 1 if has_inject else 0),
            ],
            grid=(tune["down_threads"] * n_blk, 1, 1),
            threadgroup=(tune["down_threads"], 1, 1),
            output_shapes=[(rd["x"].shape[0], LOWRANK), (rd["x"].shape[0], HC_COUNT)],
            output_dtypes=[dt, dt],
        )
        outs.append(mixv)          # stage "bd": norm + folded down/inject
        if has_inject:
            outs.append(inject)
    return outs


def run_v1(reads):
    outs = []
    for rd in reads:
        mixed, inject = _hc_v1.fused_hyper_read(
            rd["x"], rd["gamma"], rd["wd"], rd["wu"], rd["wi"]
        )
        outs.append(mixed)
        if rd["wi"] is not None:
            outs.append(inject)
    return outs


def build_variant(name, reads, tune):
    if name == "a":
        return lambda: run_eager(reads)
    if name in ("b", "bn", "bd"):
        return lambda: run_m4(reads, tune, stage=name)
    if name == "c":
        return lambda: run_v1(reads)
    raise ValueError(name)


def time_variant(run, reps, warmup, clear_cache):
    """``run()`` only builds the graph; ``mx.eval`` executes it.  ``median_ms``
    is the encode+GPU number every bandwidth claim below rests on; ``build_ms``
    is the Python graph-construction tax, reported separately."""

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
# numerics
# --------------------------------------------------------------------------


def _diff(got, ref):
    g = got.astype(mx.float32)
    r = ref.astype(mx.float32)
    d = mx.abs(g - r)
    denom = mx.maximum(mx.abs(r), mx.array(1e-30, mx.float32))
    return {
        "max_abs_diff": float(mx.max(d).item()),
        "max_rel_diff": float(mx.max(d / denom).item()),
        "differing_elements": int(mx.sum(got != ref).item()),
        "total_elements": int(ref.size),
    }


def numerics(reads, tune):
    """Every variant against the eager module, on the first combine read and
    on the no-combine trunk mixer."""

    out = {}
    picks = [("combine", 0)]
    for i, rd in enumerate(reads):
        if not rd["combine"]:
            picks.append(("mixer", i))
            break
    for label, i in picks:
        rd = reads[i]
        ref = rd["module"](rd["x"])
        ref_mixed = ref[0] if rd["combine"] else ref
        ref_inject = ref[2] if rd["combine"] else None
        if ref_inject is None:
            mx.eval(ref_mixed)
        else:
            mx.eval(ref_mixed, ref_inject)

        got_mixed, got_inject = _hcm4.fused_hc_read_m4(
            rd["x"],
            rd["gamma"],
            rd["wd"],
            rd["wu"],
            rd["wi"],
            eps=RMS_EPS,
            norm_threads=tune["norm_threads"],
            down_threads=tune["down_threads"],
            out_per_tg=tune["out_per_tg"],
            d_per_block=tune["d_per_block"],
        )
        mx.eval(got_mixed)
        out[f"b:{label}:mixed"] = _diff(got_mixed, ref_mixed)
        if ref_inject is not None:
            mx.eval(got_inject)
            out[f"b:{label}:inject"] = _diff(got_inject, ref_inject)

        v1_mixed, v1_inject = _hc_v1.fused_hyper_read(
            rd["x"], rd["gamma"], rd["wd"], rd["wu"], rd["wi"]
        )
        mx.eval(v1_mixed)
        out[f"c:{label}:mixed"] = _diff(v1_mixed, ref_mixed)
        if ref_inject is not None:
            mx.eval(v1_inject)
            out[f"c:{label}:inject"] = _diff(v1_inject, ref_inject)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=ROWS, help="verify width R (2..8)")
    p.add_argument("--calls", type=int, default=CALLS_PER_CYCLE)
    p.add_argument("--noncombine", type=int, default=NONCOMBINE_CALLS,
                   help="reads built with use_combine=False (the trunk mixer)")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shared-weights", action="store_true",
                   help="reuse one weight set for all reads (L2-hot; NOT the "
                        "real cycle, which streams 1.28 GB of private weights)")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--variants", type=str, default="a,bn,bd,b,c")
    p.add_argument("--norm-threads", type=int, default=None)
    p.add_argument("--down-threads", type=int, default=None)
    p.add_argument("--out-per-tg", type=int, default=None)
    p.add_argument("--d-per-block", type=int, default=None)
    p.add_argument("--sweep", type=str, default=None,
                   help="comma list of out_per_tg:down_threads:d_per_block "
                        "triples to time variant b at, e.g. 4:256:8,6:256:8")
    p.add_argument("--out", type=str, default=None)
    return p


def default_tune():
    return {
        "norm_threads": _hcm4.DEFAULT_NORM_THREADS,
        "down_threads": _hcm4.DEFAULT_NORM_THREADS,
        "out_per_tg": _hcm4.DEFAULT_OUT_PER_TG,
        "d_per_block": _hcm4.DEFAULT_D_PER_BLOCK,
    }


def parse_sweep(spec: str):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"sweep item {item!r}: want out_per_tg:threads:dpb")
        out.append(tuple(int(v) for v in parts))
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print("[micro-hc-read] must run under /tmp/mtplx-gpu-exclusive.lock", flush=True)
    _require_mlx()

    tune = default_tune()
    for key, val in (
        ("norm_threads", args.norm_threads),
        ("down_threads", args.down_threads),
        ("out_per_tg", args.out_per_tg),
        ("d_per_block", args.d_per_block),
    ):
        if val is not None:
            tune[key] = val

    cb = cycle_bytes(args.calls, args.noncombine)
    print(
        f"[build] {args.calls} reads (R={args.rows}, {args.noncombine} no-combine), "
        f"gamma [{HCD}] down [{LOWRANK}, {HCD}] up [{HCD}, {LOWRANK}] "
        f"inject [{HC_COUNT}, {HCD}] bf16 -> {cb['total'] / 1e9:.3f} GB/cycle "
        f"({'shared' if args.shared_weights else 'distinct'} weights)",
        flush=True,
    )
    reads = build_reads(
        args.calls, args.noncombine, args.rows, not args.shared_weights, args.seed
    )
    print(f"[build] tune {tune}", flush=True)

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    results = {}
    for name in names:
        stats = time_variant(
            build_variant(name, reads, tune), args.reps, args.warmup, args.clear_cache
        )
        if name == "a":
            moved = cb["total"]
        elif name == "bn":
            moved = cb["gamma"]
        elif name == "bd":
            moved = cb["gamma"] + cb["down"] + cb["inject"]
        elif name == "b":
            moved = cb["total"]
        else:  # c re-reads every weight once per row
            moved = cb["total"] * args.rows
        stats.update(
            {
                "gb_moved": moved / 1e9,
                "gbps": (moved / 1e9) / (stats["median_ms"] / 1e3),
                "ms_per_call": stats["median_ms"] / args.calls,
                "us_per_call": stats["median_ms"] * 1e3 / args.calls,
                "dispatches_per_call": DISPATCHES[name],
                "dispatches_per_cycle": DISPATCHES[name] * args.calls,
            }
        )
        results[name] = stats

    sweep = {}
    if args.sweep:
        for opt, thr, dpb in parse_sweep(args.sweep):
            t = dict(tune, out_per_tg=opt, down_threads=thr, d_per_block=dpb)
            st = time_variant(
                build_variant("b", reads, t), args.reps, args.warmup, args.clear_cache
            )
            st["gbps"] = (cb["total"] / 1e9) / (st["median_ms"] / 1e3)
            st["us_per_call"] = st["median_ms"] * 1e3 / args.calls
            sweep[f"{opt}:{thr}:{dpb}"] = st

    num = numerics(reads, tune)

    # ---- report
    hdr = (f"{'var':<4}{'ms/cycle':>10}{'p10':>9}{'p90':>9}{'us/call':>10}"
           f"{'GB':>8}{'GB/s':>9}{'disp/call':>11}{'disp/cyc':>10}{'build':>9}")
    print(f"\nR={args.rows}  calls={args.calls}  reps={args.reps}")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        r = results[name]
        print(f"{name:<4}{r['median_ms']:>10.3f}{r['p10_ms']:>9.3f}{r['p90_ms']:>9.3f}"
              f"{r['us_per_call']:>10.2f}{r['gb_moved']:>8.3f}{r['gbps']:>9.1f}"
              f"{r['dispatches_per_call']:>11d}{r['dispatches_per_cycle']:>10d}"
              f"{r['build_ms']:>9.3f}")

    stages = {}
    if {"a", "b"} <= set(names):
        stages["speedup_vs_eager"] = results["a"]["median_ms"] / results["b"]["median_ms"]
        stages["pct_faster_than_eager"] = 100.0 * (
            1.0 - results["b"]["median_ms"] / results["a"]["median_ms"]
        )
    if {"bn", "bd"} <= set(names):
        dt = (results["bd"]["median_ms"] - results["bn"]["median_ms"]) / 1e3
        db = (cb["down"] + cb["inject"]) / 1e9
        stages["down_stage_ms"] = dt * 1e3
        stages["down_gbps"] = (db / dt) if dt > 0 else float("inf")
    if {"bd", "b"} <= set(names):
        ut = (results["b"]["median_ms"] - results["bd"]["median_ms"]) / 1e3
        ub = cb["up"] / 1e9
        stages["up_stage_ms"] = ut * 1e3
        stages["up_gbps"] = (ub / ut) if ut > 0 else float("inf")
    if stages:
        print("\nstages (differences of the timed variants):")
        for k, v in stages.items():
            print(f"  {k:<22} {v:.3f}")

    if sweep:
        print("\nsweep (variant b, out_per_tg:down_threads:d_per_block):")
        print(f"  {'cfg':<14}{'ms/cycle':>10}{'us/call':>10}{'GB/s':>9}")
        for k, v in sorted(sweep.items(), key=lambda kv: kv[1]["median_ms"]):
            print(f"  {k:<14}{v['median_ms']:>10.3f}{v['us_per_call']:>10.2f}"
                  f"{v['gbps']:>9.1f}")

    print("\nnumerics vs the eager module (a):")
    for k, n in num.items():
        print(f"  {k:<20} max_abs={n['max_abs_diff']:.6g}  "
              f"max_rel={n['max_rel_diff']:.6g}  "
              f"differing={n['differing_elements']}/{n['total_elements']}")

    summary = {
        "rows": args.rows,
        "calls": args.calls,
        "noncombine": args.noncombine,
        "reps": args.reps,
        "shared_weights": bool(args.shared_weights),
        "tune": tune,
        "cycle_bytes": cb,
        "variants": results,
        "stages": stages,
        "sweep": sweep,
        "numerics": num,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
