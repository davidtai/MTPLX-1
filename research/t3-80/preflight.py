#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 80 GiB envelope cell
(`hy3-oq2e-rq4-80`, islands 69, David's no-islands-change directive applies
only to <=64 GiB so this preset is used AS-IS -- 2026-07-22).

Same gate, same guarantee, same pattern as research/t3-88/preflight.py and
research/envelope_admission_sweep.py's own `evaluate_admission()`, which this
script imports and calls directly (no re-implementation): no MLX buffer is
ever allocated, no weight byte is ever read (ExpertStreamingRuntime.open /
apis.load() are never reached; the MTP artifact is opened only far enough to
read its safetensors HEADER via os.pread).

Unlike the -88 cell (which needed two fix-candidate variants, 88r/88f, to
clear a REJECT), `hy3-oq2e-rq4-80` is used AS-IS -- this script targets that
one preset only, in the two modes research/t3-88/preflight.py supports:

  1. Standalone sweep across the full KV_TOKEN_SWEEP, for the byte-exact
     receipt:
         PYTHONPATH=<worktree> python3 research/t3-80/preflight.py \\
             --output research/envelope-admission-sweep-2026-07-22-80.json

  2. Single-cell gate mode, used by run_80x16k.sh / run_80x32k.sh BEFORE
     opening any guarded window (exit 0 iff ADMIT, exit 1 otherwise -- the
     same STOP-before-GPU discipline as research/t3-64-ab/preflight.py):
         PYTHONPATH=<worktree> python3 research/t3-80/preflight.py \\
             --kv 16384
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
PRESET = "hy3-oq2e-rq4-80"


def sweep_cell(kv_tokens: int) -> dict[str, Any]:
    """Same per-cell logic as envelope_admission_sweep.py's sweep(): evaluate
    at the preset's declared memory-limit first; if (and only if) that
    REJECTs, re-evaluate with the KV-delta override applied."""

    base = evaluate_admission(PRESET, kv_tokens)
    cell: dict[str, Any] = {"preset": PRESET, "max_live_kv_tokens": kv_tokens, "base": base}
    if base["status"] == "REJECT":
        kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
        override_bytes = kv_delta_tokens * KV_BYTES_PER_TOKEN
        override_memory_limit_bytes = base["memory_limit_bytes"] + override_bytes
        overridden = evaluate_admission(
            PRESET, kv_tokens, memory_limit_override_bytes=override_memory_limit_bytes
        )
        cell["kv_delta_tokens"] = kv_delta_tokens
        cell["override_bytes"] = override_bytes
        cell["override_memory_limit_bytes"] = override_memory_limit_bytes
        cell["overridden"] = overridden
        cell["still_rejects_with_override"] = overridden["status"] == "REJECT"
    return cell


def sweep() -> list[dict[str, Any]]:
    return [sweep_cell(kv) for kv in KV_TOKEN_SWEEP]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--kv", type=int, default=None, help="Single-cell gate mode: max-live-kv-tokens."
    )
    args = parser.parse_args()

    if args.kv is not None:
        cell = sweep_cell(args.kv)
        final = cell.get("overridden", cell["base"])
        status = final["status"]
        print(
            f"{PRESET} kv={args.kv}: {status} "
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
            "T3 80 GiB envelope cell preflight: validates hy3-oq2e-rq4-80 "
            "(islands 69, used AS-IS -- David's no-islands-change directive "
            "applies only to <=64 GiB presets) against the SAME gate as "
            "envelope-admission-sweep-2026-07-22.json (research/"
            "envelope_admission_sweep.py's evaluate_admission()), restricted "
            "to this one preset across the full KV_TOKEN_SWEEP. Does NOT "
            "overwrite that file."
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
        "preset": PRESET,
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
