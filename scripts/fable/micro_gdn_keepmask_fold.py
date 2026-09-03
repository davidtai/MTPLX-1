#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GPU WINDOW REQUIRED.  Run only under /tmp/mtplx-gpu-exclusive.lock via
# bench/laguna/run_guarded.py.  This script issues Metal work; running it
# outside the serialized window interrupts whatever else holds the box.
# ---------------------------------------------------------------------------
"""Price -- and falsify -- the GDN keep-mask fold before anyone builds it in.

WHAT THIS DECIDES
-----------------
The retained control lane makes TWO passes over every GDN layer's f32
recurrent state per decode cycle:

  verify   36 x gated_delta_step(T=4) from the pre-window state   226.5 MB
  replay   36 x gated_delta_step(T=keep) from the SAME pre-window
           state, in commit_verified_window, on the ~69 % of cycles
           that reject something                                  226.5 MB

The W58 retained census measures the second one directly.  An all-accept cycle
returns before the commit, so it runs 36 step dispatches; a partial-accept
cycle runs 36 + ~36.  Over the 381-cycle window:

    all-accept     n=112   4,317.7 dispatches   30.008 ms GPU busy   36.13 ms wall
    partial        n=262   4,548.8 dispatches   30.706 ms GPU busy   37.70 ms wall
    delta                  +231.1               +0.699 ms            +1.57 ms

The fold removes the second pass by handing the kept rows to the NEXT window's
step kernel.  That is bit-exact -- the kernel loads the fp32 state once,
iterates t, stores it once, so splitting or merging the T loop is the identity
-- but it makes every verify step kernel run ``4 * ring_windows`` extra ``t``
iterations, most of them masked.

    IF gated_delta_step is state-bound, its time is flat in T and the fold is
    free.  IF it is T-bound, the extra iterations cancel the saving and the
    fold is dead.

The census records dispatch counts and command-buffer times, not per-kernel
times, so it cannot answer that.  This script does.  **Arm A (the T sweep) is
the falsifier: run it first and read it before building anything.**

ARMS
----
  tsweep       stock gated_delta_step over ``--layers`` layers at
               T in {1,2,3,4,8,12,16,20}.  Flat curve => fold is free.
  today        verify(T=4) + replay(T=keep) -- the shipped two-pass commit.
  fold_mlx     one stock masked step at T = 4*W + 4 over concatenated rows.
               No new Metal, but ``mx.concatenate`` copies every input: five
               tensors x (W+1) pieces x 35 layers = 525 copy dispatches/cycle.
  fold_kernel  mtplx_gated_delta_step_prefix: the same recurrence with the
               prefix in its own buffers.  One state pass, zero concats.

PARITY (the bar, not a tolerance)
---------------------------------
Bit-equality of the committed f32 state, and of the window rows' bf16 ``y``,
against ``today``, for every kept width k in 1..3 and every ring depth in
1..W.  The state is the model's memory: a single differing element is a
different model, so this reports ``differing elements``, never a max-abs-diff.

RUN IT (guarded; this file must not be run outside the window)
--------------------------------------------------------------
  PYTHONPATH=<worktree> <worktree>/.venv/bin/python \\
    /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \\
    --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \\
    --lock-timeout-seconds 1800 --child-timeout-seconds 3600 \\
    -- <worktree>/.venv/bin/python \\
       <worktree>/scripts/fable/micro_gdn_keepmask_fold.py \\
       --layers 35 --reps 200 \\
       --out <artifacts>/micro-gdn-keepmask-fold.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# MLX is imported lazily so the CLI surface, the byte model and the pass
# accounting stay checkable without touching Metal (--dry-run).
mx = None
_gated = None
_fold = None
_foldk = None

REPO_ROOT = Path(__file__).resolve().parents[2]

# The one production geometry (Qwen3.8 Flash-Next 125B-A6B, 36 GDN layers).
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_DIM = 128
VERIFY_WIDTH = 4
#: Foldable layers: 36 GDN minus the single PLE-carrying layer, whose commit
#: is a whole-layer exact-width replay and stays as it is.
DEFAULT_LAYERS = 35
STATE_BYTES = NUM_V_HEADS * HEAD_DIM * HEAD_DIM * 4  # 3,145,728

T_SWEEP = (1, 2, 3, 4, 8, 12, 16, 20)

#: Measured on the retained control census (w58-retained-control, 381 cycles):
#: 112 all-accept / 262 partial.
P_ALL_ACCEPT = 112 / 374


def _require_mlx() -> None:
    global mx, _gated, _fold, _foldk
    import mlx.core

    mx = mlx.core
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mlx_lm.models import gated_delta

    from mtplx import fable_gdn_keepmask_fold
    from mtplx.kernels import gdn_keepmask_fold

    _gated = gated_delta
    _fold = fable_gdn_keepmask_fold
    _foldk = gdn_keepmask_fold


# --------------------------------------------------------------------------
# pass / byte accounting (pure python, checkable without MLX)
# --------------------------------------------------------------------------


def pass_model(layers: int, max_windows: int, p_all_accept: float) -> dict:
    """State passes and bytes per decode cycle, today vs folded."""

    from mtplx.fable_gdn_keepmask_fold import expected_state_passes_per_cycle

    eager = 1.0 - p_all_accept
    folded = expected_state_passes_per_cycle(p_all_accept, max_windows=max_windows)
    per_pass_bytes = 2 * STATE_BYTES * layers
    return {
        "layers": layers,
        "max_windows": max_windows,
        "p_all_accept": p_all_accept,
        "verify_passes_per_cycle": 1.0,
        "commit_passes_today": eager,
        "commit_passes_folded": folded,
        "passes_removed_per_cycle": eager - folded,
        "bytes_per_pass": per_pass_bytes,
        "mb_removed_per_cycle": (eager - folded) * per_pass_bytes / 1e6,
        "step_T_today": VERIFY_WIDTH,
        "step_T_folded": VERIFY_WIDTH * (max_windows + 1),
        "concat_dispatches_fold_mlx_per_cycle": 5 * (max_windows + 1) * layers,
    }


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _rows(width: int):
    """One window's captured (q, k, v, a, b) at the production shapes."""

    q = mx.random.normal((1, width, NUM_K_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    k = mx.random.normal((1, width, NUM_K_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal((1, width, NUM_V_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    a = mx.random.normal((1, width, NUM_V_HEADS)).astype(mx.bfloat16)
    b = mx.random.normal((1, width, NUM_V_HEADS)).astype(mx.bfloat16)
    return q, k, v, a, b


def build_layers(count: int, ring: int):
    """Per-layer fixtures: A_log, dt_bias, the base state, and ring+1 windows.

    Distinct values per layer -- repeated weights hide indexing bugs behind an
    accidentally-correct answer.
    """

    layers = []
    for _ in range(count):
        A_log = mx.random.normal((NUM_V_HEADS,)).astype(mx.float32)
        dt_bias = mx.random.normal((NUM_V_HEADS,)).astype(mx.bfloat16)
        state = mx.random.normal(
            (1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM)
        ).astype(mx.float32)
        windows = [_rows(VERIFY_WIDTH) for _ in range(ring + 1)]
        layers.append((A_log, dt_bias, state, windows))
    mx.eval(
        *[leaf for A, d, s, w in layers for leaf in (A, d, s)],
        *[t for _A, _d, _s, w in layers for row in w for t in row],
    )
    return layers


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def _sliced(rows, keep: int):
    return tuple(t[:, :keep] for t in rows)


def arm_today(layers, keeps):
    """The shipped two-pass commit, exactly as commit_verified_window builds it.

    Replay the kept prefix of each deferred window from the pre-window state
    (unmasked kernel at T=keep, one dispatch per window), then run the new
    window at T=4 from the committed state.
    """

    outs = []
    for A_log, dt_bias, state, windows in layers:
        current = state
        for window, keep in zip(windows[: len(keeps)], keeps):
            q, k, v, a, b = _sliced(window, keep)
            _y, current = _gated.gated_delta_update(
                q, k, v, a, b, A_log, dt_bias, current, None, use_kernel=True
            )
        q, k, v, a, b = windows[-1]
        y, current = _gated.gated_delta_update(
            q, k, v, a, b, A_log, dt_bias, current, None, use_kernel=True
        )
        outs.append((y, current))
    return outs


def arm_fold_mlx(layers, keeps, max_windows):
    outs = []
    for A_log, dt_bias, state, windows in layers:
        y, out_state = _foldk.folded_gated_delta_update(
            windows[: len(keeps)],
            keeps,
            *windows[-1],
            A_log,
            dt_bias,
            state,
            max_windows=max_windows,
        )
        outs.append((y, out_state))
    return outs


def _padded_prefix(windows, keeps, max_windows):
    """Prefix buffers at the fixed [1, 4*W, ...] shape, pad slots masked off."""

    width = VERIFY_WIDTH * max_windows
    pad = width - VERIFY_WIDTH * len(keeps)
    pieces = [[] for _ in range(5)]
    shapes = (
        (1, pad, NUM_K_HEADS, HEAD_DIM),
        (1, pad, NUM_K_HEADS, HEAD_DIM),
        (1, pad, NUM_V_HEADS, HEAD_DIM),
        (1, pad, NUM_V_HEADS),
        (1, pad, NUM_V_HEADS),
    )
    if pad:
        for index, shape in enumerate(shapes):
            pieces[index].append(mx.zeros(shape, dtype=mx.bfloat16))
    for window in windows:
        for index, tensor in enumerate(window):
            pieces[index].append(tensor)
    prefix = tuple(mx.concatenate(p, axis=1) for p in pieces)
    mask = mx.array(
        [_fold.prefix_mask_rows(keeps, max_windows=max_windows)], dtype=mx.bool_
    )
    return prefix, mask


def arm_fold_kernel(layers, keeps, max_windows, prepared):
    outs = []
    for (A_log, dt_bias, state, windows), (prefix, mask) in zip(layers, prepared):
        y, out_state = _foldk.prefix_gated_delta_update(
            *prefix,
            mask,
            *windows[-1],
            A_log,
            dt_bias,
            state,
        )
        outs.append((y, out_state))
    return outs


def arm_tsweep(layers, width):
    outs = []
    for A_log, dt_bias, state, _windows in layers:
        q, k, v, a, b = _rows(width)
        outs.append(
            _gated.gated_delta_update(
                q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=True
            )
        )
    return outs


# --------------------------------------------------------------------------
# timing (queued lane; see the queued-vs-eager note in the bench README)
# --------------------------------------------------------------------------


def time_arm(run, reps: int, warmup: int, clear_cache: bool) -> dict:
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


def parity(reference, candidate) -> dict:
    """Bit-equality, element counts.  No tolerance: the state is the model."""

    state_diff = 0
    y_diff = 0
    for (ref_y, ref_state), (got_y, got_state) in zip(reference, candidate):
        state_diff += int(mx.sum(ref_state != got_state).item())
        y_diff += int(mx.sum(ref_y != got_y).item())
    return {
        "state_differing_elements": state_diff,
        "y_differing_elements": y_diff,
        "bit_exact": state_diff == 0 and y_diff == 0,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-windows",
        type=int,
        default=2,
        help="ring depth in whole verify windows (1..4)",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="mx.clear_cache() before each rep (colder buffer pool)",
    )
    p.add_argument(
        "--skip-kernel",
        action="store_true",
        help="skip the prefix-kernel arm (parity + timing of the MLX fold only)",
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the pass model; touch no GPU",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_windows not in (1, 2, 3, 4):
        raise SystemExit("--max-windows must be 1..4")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    plan = {
        "layers": args.layers,
        "reps": args.reps,
        "max_windows": args.max_windows,
        "t_sweep": list(T_SWEEP),
        "pass_model": pass_model(args.layers, args.max_windows, P_ALL_ACCEPT),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    print(
        "[micro-gdn-keepmask-fold] must run under "
        "/tmp/mtplx-gpu-exclusive.lock",
        flush=True,
    )
    _require_mlx()
    mx.random.seed(args.seed)

    model = plan["pass_model"]
    print(
        f"[plan] {args.layers} foldable layers, ring {args.max_windows} "
        f"window(s) -> step T {model['step_T_today']} -> "
        f"{model['step_T_folded']}; commit passes/cycle "
        f"{model['commit_passes_today']:.3f} -> "
        f"{model['commit_passes_folded']:.3f} "
        f"({model['mb_removed_per_cycle']:.1f} MB/cycle removed)",
        flush=True,
    )

    results: dict = {"plan": plan, "tsweep": {}, "arms": {}, "parity": {}}

    # ---- A: the falsifier ------------------------------------------------
    layers = build_layers(args.layers, args.max_windows)
    print("\n[A] stock gated_delta_step vs T (the falsifier)", flush=True)
    print(f"{'T':>4}  {'ms/cycle':>10}  {'us/layer':>10}  {'vs T=4':>8}", flush=True)
    base_ms = None
    for width in T_SWEEP:
        stats = time_arm(
            lambda w=width: arm_tsweep(layers, w),
            args.reps,
            args.warmup,
            args.clear_cache,
        )
        if width == VERIFY_WIDTH:
            base_ms = stats["median_ms"]
        results["tsweep"][str(width)] = stats
    for width in T_SWEEP:
        stats = results["tsweep"][str(width)]
        ratio = stats["median_ms"] / base_ms if base_ms else float("nan")
        stats["ratio_vs_T4"] = ratio
        print(
            f"{width:>4}  {stats['median_ms']:>10.4f}  "
            f"{stats['median_ms'] * 1e3 / args.layers:>10.2f}  {ratio:>8.3f}",
            flush=True,
        )
    folded_T = model["step_T_folded"]
    folded_ratio = results["tsweep"].get(str(folded_T), {}).get("ratio_vs_T4")
    if folded_ratio is not None:
        extra_ms = (folded_ratio - 1.0) * base_ms
        verdict = (
            "STATE-BOUND: the fold's extra rows are ~free"
            if extra_ms < 0.5 * model["mb_removed_per_cycle"] / 400.0
            else "T-BOUND: the extra rows eat the saving -- do not build the fold"
        )
        results["falsifier"] = {
            "folded_T": folded_T,
            "ratio_vs_T4": folded_ratio,
            "extra_ms_per_cycle": extra_ms,
            "mb_removed_per_cycle": model["mb_removed_per_cycle"],
            "verdict": verdict,
        }
        print(f"\n[A] T={folded_T} costs {extra_ms:+.4f} ms/cycle extra -> {verdict}")

    # ---- B: the three commit arms ---------------------------------------
    for keeps in _keep_cases(args.max_windows):
        label = "+".join(str(k) for k in keeps)
        prepared = [
            _padded_prefix(windows[: len(keeps)], keeps, args.max_windows)
            for _A, _d, _s, windows in layers
        ]
        mx.eval(*[t for prefix, mask in prepared for t in (*prefix, mask)])

        reference = arm_today(layers, keeps)
        mx.eval(*[t for pair in reference for t in pair])

        arms = {
            "today": lambda: arm_today(layers, keeps),
            "fold_mlx": lambda: arm_fold_mlx(layers, keeps, args.max_windows),
        }
        if not args.skip_kernel:
            arms["fold_kernel"] = lambda: arm_fold_kernel(
                layers, keeps, args.max_windows, prepared
            )
        row: dict = {}
        for name, run in arms.items():
            stats = time_arm(run, args.reps, args.warmup, args.clear_cache)
            row[name] = stats
            if name != "today":
                row[name]["parity"] = parity(reference, run())
        results["arms"][label] = row
        print(f"\n[B] ring keeps = {label}", flush=True)
        print(
            f"{'arm':>12}  {'ms':>9}  {'us/layer':>9}  {'vs today':>9}  parity",
            flush=True,
        )
        today_ms = row["today"]["median_ms"]
        for name, stats in row.items():
            par = stats.get("parity")
            par_text = (
                "-"
                if par is None
                else (
                    "bit-exact"
                    if par["bit_exact"]
                    else f"DIFF state={par['state_differing_elements']} "
                    f"y={par['y_differing_elements']}"
                )
            )
            print(
                f"{name:>12}  {stats['median_ms']:>9.4f}  "
                f"{stats['median_ms'] * 1e3 / args.layers:>9.2f}  "
                f"{stats['median_ms'] / today_ms:>9.3f}  {par_text}",
                flush=True,
            )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[out] {args.out}", flush=True)
    bad = [
        f"{label}/{name}"
        for label, row in results["arms"].items()
        for name, stats in row.items()
        if stats.get("parity") and not stats["parity"]["bit_exact"]
    ]
    if bad:
        print(f"\nPARITY FAILED: {', '.join(bad)}", flush=True)
        return 1
    return 0


def _keep_cases(max_windows: int) -> list[tuple[int, ...]]:
    """Ring contents worth measuring: every depth, every kept width.

    ``keep`` is ``accepted_count + 1`` in 1..3 -- a whole-window accept never
    reaches the commit at all.
    """

    cases: list[tuple[int, ...]] = [(1,), (2,), (3,)]
    if max_windows >= 2:
        cases.extend([(1, 1), (3, 3), (1, 3)])
    if max_windows >= 3:
        cases.append((2, 2, 2))
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
