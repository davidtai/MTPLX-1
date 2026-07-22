#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 88-column completion window
(2026-07-22): two small real-prefill cells at the `hy3-oq2e-rq4-88e` preset,
bf16 KV, K1 (+ free AR row), 3 reps each --

  ARM 1: fresh matrix-shape cell, contexts=1024, output-tokens=256.
  ARM 2: naturalistic short-prompt arm, contexts=265 (custom non-code tail,
         264 chat-templated tokens + 1 structurally-required filler token --
         see research/t3-88col-1024-nat/run_naturalistic_rep.py header),
         output-tokens=256.

Both arms leave `--max-live-kv-tokens` at the preset's own argparse default
(4096) -- 1024+256=1280 and 265+256=521 are both comfortably under that
control point, so this is the IDENTICAL byte-budget already established in
the `hy3-oq2e-rq4-88e` preset description (fixed=95.6294 GiB vs declared
96 GiB limit, margin +0.3706 GiB, ADMIT) -- no KV-delta override needed at
either context. This script re-derives that ADMIT via the real harness gate
(research/t3-cachelru/sweep.py's `evaluate_admission_kv`, the same CPU-only
`_load_impl` prefix used by every sibling preflight in this family), never
hand-derived, before either arm's guarded window opens.

Usage:
    PYTHONPATH=<worktree> python3 research/t3-88col-1024-nat/preflight.py
"""

from __future__ import annotations

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
CONTROL_KV_TOKENS = 4096  # preset's own argparse default; unchanged by either arm


def main() -> int:
    result: dict[str, Any] = evaluate_admission_kv(PRESET, CONTROL_KV_TOKENS, kv_quant=None)
    status = result["status"]
    print(
        f"[preflight] {PRESET} kv={CONTROL_KV_TOKENS} kv_quant=off(bf16): {status} "
        f"(fixed={result['fixed_bytes'] / GIB:.4f} GiB, "
        f"declared_limit={result['memory_limit_bytes'] / GIB:.4f} GiB, "
        f"margin={result['unallocated_bytes'] / GIB:+.4f} GiB)",
        file=sys.stderr,
    )
    payload = {
        "schema": "mtplx-t3-88col-1024-nat-admission-preflight-v1",
        "generated": "2026-07-22",
        "note": (
            "CPU-only admission preflight covering BOTH 88-column completion "
            "arms (fresh 1024 cell + naturalistic ~265 cell) -- both leave "
            "max-live-kv-tokens at the preset default (4096), so this single "
            "check covers both arms' byte budget identically."
        ),
        "preset": PRESET,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "arm1_contexts": 1024,
        "arm2_contexts": 265,
        "output_tokens": 256,
        "result": result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "ADMIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
