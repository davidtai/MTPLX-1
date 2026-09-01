#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price per-dispatch and per-command-buffer overhead on this box.

The Flash-Next verify cycle issues ~5,400 dispatches inside ~171 command
buffers; a fit over the measured cycle puts ~2.0 us on each dispatch and
~10.4 us on each command buffer.  If that fit is right, then batching more ops
per command buffer is worth ~1.5 ms/cycle and is free -- but MLX decides the
buffer boundary from MLX_MAX_OPS_PER_BUFFER (default 50 in 0.32.2), which is
read once at device init.  So the sweep has to re-exec itself per value.

Two chains are timed at each setting:

  elementwise  N dependent [4, 2560] bf16 adds -- pure launch cost, no memory
  mixed        48 "layers" of rms_norm -> q8 quantized_matmul (M=4) -> silu*mul
               -> cast -> add -> cast: a chain that actually carries compute,
               so the overhead share is the part a real decode would recover

Timings split Python graph construction (build_ms) from encode+GPU (eval_ms);
MLX is lazy, so only the latter contains the dispatch cost being priced.

Note MLX also caps a buffer by MLX_MAX_MB_PER_BUFFER; a large ops setting can
be bounded by that instead, which shows up as the timing flattening out.

Standalone by construction: imports no mtplx.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROWS = 4
HIDDEN = 2560
LAYERS = 48
CHAIN_LENGTHS = (100, 1000, 5000)
OPS_PER_BUFFER_SWEEP = (50, 100, 200, 500, 1000)
MIXED_OPS_PER_LAYER = 9  # rms_norm, qmm, sigmoid, mul, mul, 2x cast, add, cast


def mixed_delta_vs_baseline(base_ms: float, value_ms: float) -> tuple[float, float]:
    """Signed delta of one mixed-chain eval time against the baseline setting.

    Sign convention: POSITIVE means this setting is SLOWER than the baseline.
    It used to be computed as ``base - value``, which printed a *negative*
    percentage for a setting that got slower -- 1.240 -> 1.661 ms was reported
    as -33.97% instead of +33.95%.
    """

    delta = float(value_ms) - float(base_ms)
    pct = 100.0 * delta / float(base_ms) if base_ms else 0.0
    return delta, pct


def _percentiles(samples):
    samples = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, int(0.10 * (len(samples) - 1)))],
        "p90_ms": samples[min(len(samples) - 1, int(0.90 * (len(samples) - 1)))],
    }


def _time(build, reps, warmup):
    """Separate Python graph construction from GPU execution.

    MLX is lazy: ``build()`` only assembles the graph, and every dispatch is
    encoded and run inside ``mx.eval``.  At ~1-3 us of Python per op that
    construction cost is the same order as the ~2.0 us/dispatch this script is
    trying to measure, so folding the two together would measure the wrong
    thing.  ``eval_ms`` is the encode+GPU number; ``build_ms`` is the Python
    tax, reported so it can be seen rather than silently absorbed.
    """

    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(build())
    evals, builds, totals = [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = build()
        t1 = time.perf_counter()
        mx.eval(out)
        t2 = time.perf_counter()
        builds.append((t1 - t0) * 1e3)
        evals.append((t2 - t1) * 1e3)
        totals.append((t2 - t0) * 1e3)
    stats = _percentiles(evals)
    stats["build_ms"] = statistics.median(builds)
    stats["total_ms"] = statistics.median(totals)
    return stats


def run_child(args) -> dict:
    import mlx.core as mx

    mx.random.seed(args.seed)
    x = mx.random.normal((ROWS, HIDDEN)).astype(mx.bfloat16)
    delta = mx.array(1e-4).astype(mx.bfloat16)
    mx.eval(x, delta)

    elementwise = {}
    for n in args.chain_lengths:
        def build(n=n):
            a = x
            for _ in range(n):
                a = a + delta
            return a

        stats = _time(build, args.reps, args.warmup)
        stats["ops"] = n
        stats["us_per_op"] = stats["median_ms"] * 1e3 / n
        stats["us_per_op_build"] = stats["build_ms"] * 1e3 / n
        stats["est_command_buffers"] = -(-n // args.ops_per_buffer)
        elementwise[str(n)] = stats

    # mixed, compute-bearing chain
    src = mx.random.normal((HIDDEN, HIDDEN)).astype(mx.bfloat16)
    w8, s8, b8 = mx.quantize(src, group_size=64, bits=8)
    norm_w = mx.ones((HIDDEN,), dtype=mx.bfloat16)
    mx.eval(w8, s8, b8, norm_w)
    del src

    def build_mixed():
        h = x
        for _ in range(args.layers):
            n = mx.fast.rms_norm(h, norm_w, 1e-6)
            y = mx.quantized_matmul(n, w8, s8, b8, transpose=True,
                                    group_size=64, bits=8)
            y = y * mx.sigmoid(y)
            y = y * n
            h = (h.astype(mx.float32) + y.astype(mx.float32)).astype(mx.bfloat16)
        return h

    total_ops = args.layers * MIXED_OPS_PER_LAYER
    mixed = _time(build_mixed, args.reps, args.warmup)
    mixed["ops"] = total_ops
    mixed["us_per_op"] = mixed["median_ms"] * 1e3 / total_ops
    mixed["us_per_op_build"] = mixed["build_ms"] * 1e3 / total_ops
    mixed["est_command_buffers"] = -(-total_ops // args.ops_per_buffer)

    return {
        "ops_per_buffer": args.ops_per_buffer,
        "max_mb_per_buffer": os.environ.get("MLX_MAX_MB_PER_BUFFER"),
        "elementwise": elementwise,
        "mixed": mixed,
    }


def run_parent(args) -> int:
    print("[micro-dispatch] must run under /tmp/mtplx-gpu-exclusive.lock")
    print("[micro-dispatch] timings assume the GPU is otherwise IDLE; this "
          "script does not verify that -- the lock does.")
    results = []
    for value in args.ops_sweep:
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="dispatch_")
        os.close(fd)
        env = dict(os.environ)
        env["MLX_MAX_OPS_PER_BUFFER"] = str(value)
        if args.max_mb_per_buffer is not None:
            env["MLX_MAX_MB_PER_BUFFER"] = str(args.max_mb_per_buffer)
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--child",
            "--ops-per-buffer", str(value), "--child-out", tmp,
            "--reps", str(args.reps), "--warmup", str(args.warmup),
            "--layers", str(args.layers), "--seed", str(args.seed),
            "--chain-lengths", ",".join(str(c) for c in args.chain_lengths),
        ]
        print(f"[child] MLX_MAX_OPS_PER_BUFFER={value}", flush=True)
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"[child] FAILED at ops_per_buffer={value}", file=sys.stderr)
            return proc.returncode
        results.append(json.loads(Path(tmp).read_text()))
        os.unlink(tmp)

    print(f"\nelementwise chain: dependent [{ROWS}, {HIDDEN}] bf16 adds")
    hdr = (f"{'ops/buf':>8}{'N':>7}{'eval_ms':>10}{'p10':>9}{'p90':>9}"
           f"{'us/op':>9}{'build_ms':>10}{'cmdbufs':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        for n in args.chain_lengths:
            e = r["elementwise"][str(n)]
            print(f"{r['ops_per_buffer']:>8}{n:>7}{e['median_ms']:>10.3f}"
                  f"{e['p10_ms']:>9.3f}{e['p90_ms']:>9.3f}{e['us_per_op']:>9.3f}"
                  f"{e['build_ms']:>10.3f}{e['est_command_buffers']:>9}")

    print(f"\nmixed chain: {args.layers} layers x {MIXED_OPS_PER_LAYER} ops "
          f"(rms_norm, q8 qmm M={ROWS}, silu*mul, add, cast)")
    hdr = (f"{'ops/buf':>8}{'ops':>7}{'eval_ms':>10}{'p10':>9}{'p90':>9}"
           f"{'us/op':>9}{'build_ms':>10}{'cmdbufs':>9}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for r in results:
        m = r["mixed"]
        if base is None:
            base = m["median_ms"]
        print(f"{r['ops_per_buffer']:>8}{m['ops']:>7}{m['median_ms']:>10.3f}"
              f"{m['p10_ms']:>9.3f}{m['p90_ms']:>9.3f}{m['us_per_op']:>9.3f}"
              f"{m['build_ms']:>10.3f}{m['est_command_buffers']:>9}")
    baseline = results[0]["ops_per_buffer"]
    for r in results:
        delta, pct = mixed_delta_vs_baseline(base, r["mixed"]["median_ms"])
        print(f"  ops/buf={r['ops_per_buffer']:<5} delta vs {baseline}: "
              f"{delta:+.3f} ms ({pct:+.2f}%)")

    summary = {
        "rows": ROWS, "hidden": HIDDEN, "layers": args.layers,
        "reps": args.reps, "chain_lengths": list(args.chain_lengths),
        "mixed_ops_per_layer": MIXED_OPS_PER_LAYER, "settings": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out}")
    else:
        print("\n" + json.dumps(summary))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chain-lengths", type=str,
                   default=",".join(str(c) for c in CHAIN_LENGTHS))
    p.add_argument("--ops-sweep", type=str,
                   default=",".join(str(v) for v in OPS_PER_BUFFER_SWEEP),
                   help="MLX_MAX_OPS_PER_BUFFER values; one subprocess each")
    p.add_argument("--max-mb-per-buffer", type=int, default=None,
                   help="also pin MLX_MAX_MB_PER_BUFFER in every child")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--ops-per-buffer", type=int, default=OPS_PER_BUFFER_SWEEP[0],
                   help=argparse.SUPPRESS)
    p.add_argument("--child-out", type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.chain_lengths = tuple(int(c) for c in args.chain_lengths.split(",") if c)
    args.ops_sweep = tuple(int(v) for v in str(args.ops_sweep).split(",") if v)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.child:
        payload = run_child(args)
        if args.child_out:
            Path(args.child_out).write_text(json.dumps(payload))
        else:
            print(json.dumps(payload))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
