#!/usr/bin/env python3
"""GDN recurrent-state pass anatomy from a dispatch census.  CPU only.

No MLX, no GPU, no flock: this reduces a JSONL captured by
``scripts/fable/census_retained_stack.py run`` (the instrumented-MLX
``MLX_DISPATCH_CENSUS`` build) and answers two questions the family table
cannot:

1.  **How many passes over the GDN state does a cycle actually make, and
    where do they come from?**  The census's GDN family reports 1,952 MB/cycle,
    but 79 % of that is the in_proj/out_proj weights.  The recurrent state is
    only the ``gated_delta_step`` dispatches, at ``2 * 48*128*128*4`` =
    6.29 MB each.

2.  **What does the eager kept-prefix replay cost?**  There is a natural A/B
    inside every census: an all-accept cycle returns in ``generation.py``
    before ``commit_verified_window``, so it runs 36 step dispatches and no
    replay; a partial-accept cycle runs 36 + ~36.  Everything else in the two
    cycle kinds -- the 4-row verify, the 3 draft depths, the MoE and QSA
    families -- is identical, so the difference IS the replay.

Usage::

    python3 scripts/fable/census_gdn_state_passes.py \\
        .benchmark-artifacts/pr391/w58-retained-control-census-1788370322.jsonl \\
        --lo 63477 --hi 1853245
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from statistics import mean, median

#: The one dispatch that happens exactly once per verify cycle (the target
#: lm_head matvec), used as the cycle marker.  Same anchor
#: ``census_retained_stack.py`` uses.
LM_HEAD_GRID = (1, 31040, 1)
#: f32 recurrent state per GDN layer: 48 value heads x 128 x 128 x 4 bytes.
STATE_BYTES = 48 * 128 * 128 * 4
#: The library step kernel.  ``mtplx_gdn_step_fused`` is MTPLX's own S=1 kernel
#: and is excluded -- it is a different pass with a different byte profile.
STEP_KERNEL = "gated_delta_step"
#: GDN layers in the trunk; the verify pass dispatches exactly one each.
GDN_LAYERS = 36


def _is_step(name: str) -> bool:
    return STEP_KERNEL in name and "mtplx" not in name


def reduce_census(path: str, lo: int, hi: int) -> dict:
    """One streaming pass over the ops, one over the command buffers."""

    cyc_ops: dict[int, int] = defaultdict(int)
    cyc_steps: dict[int, int] = defaultdict(int)
    cyc_kernels: dict[int, Counter] = defaultdict(Counter)
    cyc_lo: dict[int, int] = {}
    current = -1

    with open(path) as handle:
        for line in handle:
            if '"record":"op"' not in line:
                continue
            record = json.loads(line)
            seq = record["seq"]
            if seq < lo or seq > hi:
                continue
            name = record["kernel_name"]
            grid = tuple(record["grid"])
            if grid == LM_HEAD_GRID and "qmv" in name:
                current += 1
                cyc_lo[current] = seq
            if current < 0:
                continue
            cyc_ops[current] += 1
            if _is_step(name):
                cyc_steps[current] += 1
            cyc_kernels[current][name.split("_bfloat16")[0][:52]] += 1

    cycles = sorted(cyc_ops)
    starts = [cyc_lo[c] for c in cycles]
    cyc_busy: dict[int, float] = defaultdict(float)
    cyc_span: dict[int, list] = defaultdict(lambda: [None, None])
    with open(path) as handle:
        for line in handle:
            if '"record":"cb"' not in line:
                continue
            record = json.loads(line)
            first = record.get("first_op_seq")
            if first is None or first < lo or first > hi:
                continue
            index = bisect.bisect_right(starts, first) - 1
            if index < 0:
                continue
            cycle = cycles[index]
            start, end = record.get("gpu_start_ns"), record.get("gpu_end_ns")
            if start and end and end > start:
                cyc_busy[cycle] += (end - start) / 1e6
                span = cyc_span[cycle]
                span[0] = start if span[0] is None else min(span[0], start)
                span[1] = end if span[1] is None else max(span[1], end)

    # Drop the first and last cycles: both are clipped by the window bounds.
    inner = [c for c in cycles if c not in (cycles[0], cycles[-1])] if cycles else []
    return {
        "cycles": inner,
        "ops": cyc_ops,
        "steps": cyc_steps,
        "kernels": cyc_kernels,
        "busy_ms": cyc_busy,
        "span_ns": cyc_span,
    }


def split_by_replay(
    data: dict, *, verify_steps: int = GDN_LAYERS
) -> tuple[list, list, list]:
    """All-accept cycles, one-replay cycles, and everything else.

    ``(without, with_replay, excluded)``.  A cycle carrying more than two
    passes is a context-copy BLOCK ROUND (``mtplx/context_copy.py``): a wider
    forward with its own commit, ~9 % of emitted tokens and a different shape
    entirely.  Leaving those in the "partial" bucket would charge the replay
    with a whole extra forward -- it moves the measured delta from +231
    dispatches / +0.70 ms to +521 / +3.39.  They are excluded and counted.
    """

    inner, steps = data["cycles"], data["steps"]
    half = verify_steps // 2
    without, with_replay, excluded = [], [], []
    for cycle in inner:
        count = steps[cycle]
        if count <= verify_steps + half:
            without.append(cycle)
        elif count <= 2 * verify_steps + half:
            with_replay.append(cycle)
        else:
            excluded.append(cycle)
    return without, with_replay, excluded


def summarise(data: dict, group: list[int]) -> dict:
    ops = [data["ops"][c] for c in group]
    busy = [data["busy_ms"][c] for c in group]
    walls = [
        (data["span_ns"][c][1] - data["span_ns"][c][0]) / 1e6
        for c in group
        if data["span_ns"][c][0]
    ]
    return {
        "cycles": len(group),
        "dispatches_per_cycle": mean(ops) if ops else 0.0,
        "gpu_busy_ms_mean": mean(busy) if busy else 0.0,
        "gpu_busy_ms_median": median(busy) if busy else 0.0,
        "wall_ms_median": median(walls) if walls else 0.0,
        "step_dispatches_per_cycle": mean(data["steps"][c] for c in group)
        if group
        else 0.0,
    }


def kernel_delta(data: dict, without: list[int], with_replay: list[int]) -> list:
    left: dict[str, float] = defaultdict(float)
    right: dict[str, float] = defaultdict(float)
    for cycle in without:
        for name, count in data["kernels"][cycle].items():
            left[name] += count / len(without)
    for cycle in with_replay:
        for name, count in data["kernels"][cycle].items():
            right[name] += count / len(with_replay)
    rows = [
        (name, right[name] - left[name])
        for name in set(left) | set(right)
        if abs(right[name] - left[name]) >= 1.0
    ]
    return sorted(rows, key=lambda row: -row[1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", help="dispatch-census JSONL")
    parser.add_argument("--lo", type=int, required=True, help="window ops_lo")
    parser.add_argument("--hi", type=int, required=True, help="window ops_hi")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    data = reduce_census(args.census, args.lo, args.hi)
    without, with_replay, excluded = split_by_replay(data)
    a, b = summarise(data, without), summarise(data, with_replay)
    total = len(without) + len(with_replay)
    steps = mean(data["steps"][c] for c in data["cycles"])

    report = {
        "window": {
            "lo": args.lo,
            "hi": args.hi,
            "cycles": total,
            "excluded_copy_round_cycles": len(excluded),
        },
        "state_bytes_per_layer_pass": 2 * STATE_BYTES,
        "step_dispatches_per_cycle": steps,
        "state_mb_per_cycle": steps * 2 * STATE_BYTES / 1e6,
        "verify_pass_mb_per_cycle": GDN_LAYERS * 2 * STATE_BYTES / 1e6,
        "replay_pass_mb_per_cycle": (steps - GDN_LAYERS) * 2 * STATE_BYTES / 1e6,
        "all_accept": a,
        "partial_accept": b,
        "replay_cost": {
            "p_partial": len(with_replay) / total if total else 0.0,
            "dispatches": b["dispatches_per_cycle"] - a["dispatches_per_cycle"],
            "gpu_busy_ms": b["gpu_busy_ms_mean"] - a["gpu_busy_ms_mean"],
            "gpu_busy_ms_median": b["gpu_busy_ms_median"] - a["gpu_busy_ms_median"],
            "wall_ms_median": b["wall_ms_median"] - a["wall_ms_median"],
        },
        "kernel_delta_per_cycle": kernel_delta(data, without, with_replay),
    }
    report["replay_cost"]["gpu_busy_ms_amortised"] = (
        report["replay_cost"]["p_partial"] * report["replay_cost"]["gpu_busy_ms"]
    )

    print(
        f"cycles {total} (+{len(excluded)} copy-round cycles excluded)  |  "
        f"gated_delta_step {steps:.2f}/cycle = "
        f"{report['state_mb_per_cycle']:.1f} MB/cycle of f32 state "
        f"({report['verify_pass_mb_per_cycle']:.1f} verify + "
        f"{report['replay_pass_mb_per_cycle']:.1f} replay)"
    )
    header = f"{'group':>22} {'n':>5} {'disp/cyc':>10} {'busy ms':>9} {'wall ms':>9}"
    print(header)
    for label, row in (("all-accept (no replay)", a), ("partial (replay)", b)):
        print(
            f"{label:>22} {row['cycles']:>5} {row['dispatches_per_cycle']:>10.1f} "
            f"{row['gpu_busy_ms_mean']:>9.3f} {row['wall_ms_median']:>9.3f}"
        )
    cost = report["replay_cost"]
    print(
        f"{'replay cost (B-A)':>22} {'':>5} {cost['dispatches']:>10.1f} "
        f"{cost['gpu_busy_ms']:>9.3f} {cost['wall_ms_median']:>9.3f}"
    )
    print(
        f"  amortised over every cycle "
        f"({cost['p_partial'] * 100:.1f} % are partial): "
        f"{cost['gpu_busy_ms_amortised']:.3f} ms GPU busy"
    )
    print("\nper-cycle dispatch delta by kernel (partial - all-accept), |d| >= 1:")
    for name, delta in report["kernel_delta_per_cycle"]:
        print(f"  {delta:+9.2f}  {name}")

    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
