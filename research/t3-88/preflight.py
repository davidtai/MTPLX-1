#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the two `hy3-oq2e-rq4-88` fix
candidates staged in benchmarks/presets.toml (2026-07-22).

Context: `hy3-oq2e-rq4-88` (islands 79) REJECTs the config-time admission
gate at every swept KV level -- a constant 0.629 GiB over-book of its own
declared 95 GiB memory-limit, independent of KV (research/envelope-
admission-sweep-2026-07-22.json, NOTES.md T4 entry). Two fix candidates:

  hy3-oq2e-rq4-88r ("r" = re-derived limit): islands 79 KEPT (true full
    residency preserved). memory-limit re-derived from the harness's own
    fixed_bytes at this exact config (kv=4096 control) + a family-healthy
    margin, rounded up to a clean GiB -> 96 GiB.

  hy3-oq2e-rq4-88f ("f" = family-formula fallback): islands 78 (one fewer
    than -88's 79), memory-limit kept at 95 GiB, per the family's own
    sizing formula (20.64 GiB fixed w/ requant credit + 0.949 GiB/island).

This script is THE SAME admission logic as research/envelope_admission_
sweep.py -- it imports and calls that module's own `evaluate_admission()`
and constants directly (no re-implementation), restricted to these two new
presets across the same KV_TOKEN_SWEEP = (4096, 16384, 32768, 65536). Same
CPU-only guarantee as that script's own docstring: no MLX buffer is ever
allocated, no weight byte is ever read (ExpertStreamingRuntime.open /
apis.load() are never reached; the MTP artifact is opened only far enough
to read its safetensors HEADER via os.pread).

Used two ways:
  1. Standalone, to generate the byte-exact receipt for both new presets
     across the full KV sweep (this invocation):
         PYTHONPATH=<worktree> python3 research/t3-88/preflight.py \\
             --output research/envelope-admission-sweep-2026-07-22-88rf.json
  2. As the CPU-only gate research/t3-88/run_88x16k.sh and
     run_88_ladder.sh run BEFORE opening any guarded window (--preset/--kv
     args select a single cell; exit 0 iff it ADMITs, exit 1 otherwise --
     the same STOP-before-GPU discipline as research/t3-64-ab/preflight.py).

Usage:
    PYTHONPATH=<worktree> python3 research/t3-88/preflight.py [--output PATH]
    PYTHONPATH=<worktree> python3 research/t3-88/preflight.py \\
        --preset hy3-oq2e-rq4-88r --kv 16384   # single-cell gate mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
for _path in (REPO_ROOT, RESEARCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from envelope_admission_sweep import (  # noqa: E402
    CONTROL_KV_TOKENS,
    KV_BYTES_PER_TOKEN,
    KV_TOKEN_SWEEP,
    evaluate_admission,
)

GIB = 1024**3
PRESETS = ("hy3-oq2e-rq4-88r", "hy3-oq2e-rq4-88f")


def sweep_cell(preset: str, kv_tokens: int) -> dict[str, Any]:
    """Same per-cell logic as envelope_admission_sweep.py's sweep(): evaluate
    at the preset's declared memory-limit first; if (and only if) that
    REJECTs, re-evaluate with the KV-delta override applied."""

    base = evaluate_admission(preset, kv_tokens)
    cell: dict[str, Any] = {"preset": preset, "max_live_kv_tokens": kv_tokens, "base": base}
    if base["status"] == "REJECT":
        kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
        override_bytes = kv_delta_tokens * KV_BYTES_PER_TOKEN
        override_memory_limit_bytes = base["memory_limit_bytes"] + override_bytes
        overridden = evaluate_admission(
            preset, kv_tokens, memory_limit_override_bytes=override_memory_limit_bytes
        )
        cell["kv_delta_tokens"] = kv_delta_tokens
        cell["override_bytes"] = override_bytes
        cell["override_memory_limit_bytes"] = override_memory_limit_bytes
        cell["overridden"] = overridden
        cell["still_rejects_with_override"] = overridden["status"] == "REJECT"
    return cell


def sweep() -> list[dict[str, Any]]:
    return [sweep_cell(preset, kv) for preset in PRESETS for kv in KV_TOKEN_SWEEP]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default=None,
        help="Single-cell gate mode: check just this preset (requires --kv).",
    )
    parser.add_argument(
        "--kv", type=int, default=None, help="Single-cell gate mode: max-live-kv-tokens."
    )
    args = parser.parse_args()

    if args.preset is not None or args.kv is not None:
        if args.preset is None or args.kv is None:
            parser.error("--preset and --kv must be given together (single-cell gate mode)")
        cell = sweep_cell(args.preset, args.kv)
        final = cell.get("overridden", cell["base"])
        status = final["status"]
        print(
            f"{args.preset} kv={args.kv}: {status} "
            f"(fixed={final['fixed_bytes'] / GIB:.4f} GiB, "
            f"limit={final['memory_limit_bytes'] / GIB:.4f} GiB, "
            # unallocated_bytes is already signed: positive = ADMIT headroom,
            # negative = REJECT shortfall. Do not negate it here.
            f"margin={final['unallocated_bytes'] / GIB:+.4f} GiB)",
            file=sys.stderr,
        )
        print(json.dumps(cell, indent=2, sort_keys=True))
        return 0 if status == "ADMIT" else 1

    cells = sweep()
    for cell in cells:
        final = cell.get("overridden", cell["base"])
        print(
            f"{cell['preset']:20} kv={cell['max_live_kv_tokens']:6} "
            f"{final['status']:6} "
            f"fixed={final['fixed_bytes'] / GIB:8.4f} GiB "
            f"limit={final['memory_limit_bytes'] / GIB:8.4f} GiB "
            f"margin={final['unallocated_bytes'] / GIB:+8.4f} GiB",
            file=sys.stderr,
        )
    payload = {
        "schema": "mtplx-envelope-admission-sweep-v1",
        "generated": "2026-07-22",
        "note": (
            "Second dated file per the 2026-07-22 T4 follow-up: validates the "
            "two -88 fix candidates (hy3-oq2e-rq4-88r, hy3-oq2e-rq4-88f) staged "
            "in benchmarks/presets.toml against the SAME gate as "
            "envelope-admission-sweep-2026-07-22.json (research/"
            "envelope_admission_sweep.py's evaluate_admission()), which "
            "established the hy3-oq2e-rq4-88 anomaly this sweep fixes. Does "
            "NOT overwrite that file."
        ),
        "gate_exercised": {
            "file": "mtplx/runtime.py",
            "function": "_load_impl",
            "description": (
                "config-time admission preflight inside the streaming_requested "
                "branch; identical gate replica as envelope_admission_sweep.py "
                "(imported directly from that module, not re-implemented)."
            ),
        },
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "presets": list(PRESETS),
        "kv_token_sweep": list(KV_TOKEN_SWEEP),
        "cells": cells,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0 if all(
        cell.get("overridden", cell["base"])["status"] == "ADMIT" for cell in cells
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
