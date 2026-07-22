#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 K-SELECTION window
(`hy3-oq2e-rq4-88e` @ 16384 real-prefill context / 16640 KV / kv_quant=q4,
2026-07-22). This is the cell that decides which K locks matrix-wide.

Same gate, same guarantee as research/t3-pilot-realprefill/preflight.py:
no MLX buffer is ever allocated, no weight byte is ever read
(ExpertStreamingRuntime.open / apis.load() are never reached; gate stops at
`config.memory_plan()` / `plan.fits_fixed`). Uses research/t3-cachelru/
sweep.py's `evaluate_admission_kv()` (the kv_quant-aware variant) directly,
never re-implemented.

Per HANDOFF-hy3-matrix-restart.md work-queue item 1: hy3-oq2e-rq4-88e
ADMITs at its own declared 96 GiB limit with NO --memory-limit override at
kv=16640/kv_quant=q4 (fixed=95.7857 GiB, margin +0.2143 GiB) -- this script
re-derives that number live rather than trusting the handoff's prose.

Usage (single-cell gate mode, exit 0 iff ADMIT):
    PYTHONPATH=<worktree> python3 research/t3-kselect-88e16k/preflight.py \\
        --kv 16640 --kv-quant q4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
for _path in (REPO_ROOT, RESEARCH_DIR, RESEARCH_DIR / "t3-cachelru"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sweep import evaluate_admission_kv  # noqa: E402

GIB = 1024**3
PRESET = "hy3-oq2e-rq4-88e"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv", type=int, required=True, help="max-live-kv-tokens")
    parser.add_argument("--kv-quant", default="q4", choices=["q8", "q4"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result: dict[str, Any] = evaluate_admission_kv(
        PRESET, args.kv, kv_quant=args.kv_quant
    )
    status = result["status"]
    print(
        f"{PRESET} kv={args.kv} kv_quant={args.kv_quant}: {status} "
        f"(fixed={result['fixed_bytes'] / GIB:.4f} GiB, "
        f"limit={result['memory_limit_bytes'] / GIB:.4f} GiB, "
        f"margin={result['unallocated_bytes'] / GIB:+.4f} GiB)",
        file=sys.stderr,
    )
    payload = {
        "schema": "mtplx-t3-kselect-88e16k-admission-preflight-v1",
        "generated": "2026-07-22",
        "note": (
            "CPU-only admission preflight for the K-SELECTION window "
            "(88-efficient x 16k real-prefill kv4, AR+K1/K2/K3 x 3 reps). "
            "Gate replica of mtplx/runtime.py's _load_impl, via research/"
            "t3-cachelru/sweep.py's evaluate_admission_kv()."
        ),
        "preset": PRESET,
        "cell": result,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "ADMIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
