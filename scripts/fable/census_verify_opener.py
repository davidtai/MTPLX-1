#!/usr/bin/env python3
"""W63: the compiled verify body's opening GPU idle, per cycle, with NO 10 us floor.

Why this exists.  ``census_retained_stack``'s gap map classifies idle by the
KERNEL PAIR that brackets it, and the retained stack does not always give the
PLE q4 dequant its own command buffer -- so the one structural gap that is
present in every cycle (the host replaying the ~5,200-node compiled fixed-M4
verify graph) is split across two families and its largest half reads as an
unrelated ``g1_copy -> gather_front``.  Anchoring on the verify body's fixed
dispatch offset from the cycle's ``lm_head`` instead recovers it whole:

    w58 retained control  1.934 ms/cycle, 86.9 % host-late, 382/382 cycles

Usage::

    python scripts/fable/census_verify_opener.py <census.jsonl> [body_len]

``body_len`` is the verify body's dispatch length + 1 (3669 on the retained
stack; print ``offsets seen`` to confirm it landed on the
``gather_frontbfloat16_int32_int_2`` buffer).  Everything else -- the
union-busy timeline, the ``host_late`` split, the lm_head cycle marker -- is
``census_retained_stack``'s own, imported rather than restated.
"""
from __future__ import annotations

import bisect
import importlib.util
import statistics
import sys
from collections import Counter
from pathlib import Path

MODULE = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/"
    "scripts/fable/census_retained_stack.py"
)
spec = importlib.util.spec_from_file_location("census_retained_stack", MODULE)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)

path = Path(sys.argv[1])
body_len = int(sys.argv[2]) if len(sys.argv) > 2 else 3668
marks = census.find_cycle_marks(path)
lo, hi, cycles = census.auto_window(marks, 0)
buffers, waits, summary, kernels = census.read_census(path, lo, hi)
buffers.sort(key=lambda cb: cb.gpu_start_ns)

# union-merged "covered end" before each buffer
covered_before = []
covered = 0
prev_cb = None
for cb in buffers:
    covered_before.append((covered, prev_cb))
    if cb.gpu_end_ns > covered:
        covered = cb.gpu_end_ns
        prev_cb = cb

first_seqs = [cb.first_op_seq for cb in buffers]
order = sorted(range(len(buffers)), key=lambda i: buffers[i].first_op_seq)
ordered_first = [buffers[i].first_op_seq for i in order]

def buffer_opening(target_seq: int):
    """The buffer whose first_op_seq is the smallest >= target_seq."""
    j = bisect.bisect_left(ordered_first, target_seq)
    if j >= len(order):
        return None
    return order[j]

rows = []
for k in range(cycles):
    mark = marks[k + 1]
    target = mark - body_len + 1
    i = buffer_opening(target)
    if i is None:
        continue
    cb = buffers[i]
    prior_end, prior_cb = covered_before[i]
    idle = cb.gpu_start_ns - prior_end
    host_late = max(0, min(cb.gpu_start_ns, cb.encode_end_ns) - prior_end)
    rows.append(
        {
            "cycle": k,
            "idle_ns": max(0, idle),
            "host_late_ns": host_late if idle > 0 else 0,
            "first_op_seq": cb.first_op_seq,
            "offset": mark - cb.first_op_seq,
            "next_kernel": cb.ops[0][0] if cb.ops else "?",
            "prev_kernel": (prior_cb.ops[-1][0] if prior_cb and prior_cb.ops else "?"),
            "op_count": len(cb.ops),
        }
    )

idles = [r["idle_ns"] for r in rows]
late = [r["host_late_ns"] for r in rows]
print(f"census {path.name}  body_len={body_len}  cycles={len(rows)}")
print(f"offsets seen: {Counter(r['offset'] for r in rows).most_common(5)}")
print(f"opening kernel: {Counter(r['next_kernel'][:44] for r in rows).most_common(4)}")
print(f"preceding kernel: {Counter(r['prev_kernel'][:44] for r in rows).most_common(6)}")
print(
    f"idle before the verify body: total {sum(idles)/1e6:.1f} ms = "
    f"{sum(idles)/1e6/len(rows):.3f} ms/cycle; host-late {100*sum(late)/max(1,sum(idles)):.1f}%"
)
print(
    f"  mean {statistics.mean(idles)/1e3:.0f} us, median {statistics.median(idles)/1e3:.0f} us, "
    f"min {min(idles)/1e3:.0f}, max {max(idles)/1e3:.0f}"
)
buckets = Counter()
for value in idles:
    us = value / 1e3
    if us < 10:
        buckets["<10us"] += 1
    elif us < 100:
        buckets["10-100us"] += 1
    elif us < 500:
        buckets["100-500us"] += 1
    elif us < 1500:
        buckets["0.5-1.5ms"] += 1
    else:
        buckets[">1.5ms"] += 1
print(f"  distribution: {dict(buckets)}")
