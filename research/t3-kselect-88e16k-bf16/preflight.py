#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 **bf16-KV** K-SELECTION
reference window (`hy3-oq2e-rq4-88e` @ 16384 real-prefill context / 16640
KV / kv_quant=off i.e. bf16, 2026-07-22). This is David's follow-on grant
after the kv4 K-selection cell (`db46f26`, #130 comment 5047822076): same
shape, bf16 KV instead of kv4, to isolate kv4's acceptance cost at real
prefill.

Same gate, same guarantee as research/t3-kselect-88e16k/preflight.py (the
kv4 sibling): no MLX buffer is ever allocated, no weight byte is ever read
(ExpertStreamingRuntime.open / apis.load() are never reached; gate stops at
`config.memory_plan()` / `plan.fits_fixed`). Uses research/t3-cachelru/
sweep.py's `evaluate_admission_kv()` directly, never re-implemented.

UNLIKE the kv4 cell, bf16 KV at 16640 live-KV tokens REJECTs at the preset's
own declared 96 GiB limit (bf16 is 327,680 B/token vs kv4's 92,160 B/token
-- 3.556x the bytes). This script derives the EXACT override the same way
every other KV-delta override in this family was derived (research/t3-80/,
research/t3-88-efficient/derive.py, the -88r/-88e preset descriptions in
benchmarks/presets.toml): run the gate with NO override first to read the
preset's own declared memory_limit_bytes and this shape's fixed_bytes under
bf16 pricing, then

    override_bytes = declared_limit_bytes
                      + (kv_tokens - CONTROL_KV_TOKENS) * KV_BYTES_PER_TOKEN_BF16

(CONTROL_KV_TOKENS=4096, KV_BYTES_PER_TOKEN_BF16=327_680, both
envelope_admission_sweep.py's own constants). This preserves the family's
constant ~0.3706 GiB margin at any KV level -- proven in the -88e preset
description: fixed_bytes at kv=4096 is 95.6294 GiB against the declared
96 GiB limit == +0.3706 GiB margin, and fixed_bytes scales by the identical
per-token bf16 price, so the override formula reproduces that same margin
here. The gate is then re-run WITH that override and must ADMIT before this
script exits 0 -- never hand-derived, always harness-verified twice
(before/after).

Usage (two-phase gate, exit 0 iff the OVERRIDDEN pass ADMITs):
    PYTHONPATH=<worktree> python3 research/t3-kselect-88e16k-bf16/preflight.py \\
        --kv 16640
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
from envelope_admission_sweep import (  # noqa: E402
    CONTROL_KV_TOKENS,
    KV_BYTES_PER_TOKEN as KV_BYTES_PER_TOKEN_BF16,
)

GIB = 1024**3
PRESET = "hy3-oq2e-rq4-88e"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv", type=int, required=True, help="max-live-kv-tokens")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Phase 1: no override -- read the preset's own declared limit and this
    # shape's fixed_bytes under bf16 (kv_quant=None) pricing. Expected to
    # REJECT (bf16 is 3.556x the bytes/token of kv4).
    base: dict[str, Any] = evaluate_admission_kv(PRESET, args.kv, kv_quant=None)
    declared_limit_bytes = base["memory_limit_bytes"]
    print(
        f"[phase1 no-override] {PRESET} kv={args.kv} kv_quant=off(bf16): "
        f"{base['status']} (fixed={base['fixed_bytes'] / GIB:.4f} GiB, "
        f"declared_limit={declared_limit_bytes / GIB:.4f} GiB, "
        f"margin={base['unallocated_bytes'] / GIB:+.4f} GiB)",
        file=sys.stderr,
    )

    # Phase 2 derivation: established KV-delta formula (research/t3-80,
    # research/t3-88-efficient/derive.py, benchmarks/presets.toml -88r/-88e
    # descriptions): declared base limit (at CONTROL_KV_TOKENS=4096) +
    # (kv_tokens - 4096) * bf16 bytes/token.
    kv_delta_tokens = args.kv - CONTROL_KV_TOKENS
    override_bytes = declared_limit_bytes + kv_delta_tokens * KV_BYTES_PER_TOKEN_BF16
    print(
        f"[derive] override_bytes = declared_limit({declared_limit_bytes}) + "
        f"(kv={args.kv} - control={CONTROL_KV_TOKENS}) * "
        f"{KV_BYTES_PER_TOKEN_BF16} B/token = {override_bytes} B "
        f"({override_bytes / GIB:.4f} GiB)",
        file=sys.stderr,
    )

    # Phase 3: re-run the SAME real gate WITH the derived override. This is
    # the number that gates the GPU window.
    overridden: dict[str, Any] = evaluate_admission_kv(
        PRESET, args.kv, kv_quant=None, memory_limit_override_bytes=override_bytes
    )
    status = overridden["status"]
    print(
        f"[phase2 overridden] {PRESET} kv={args.kv} kv_quant=off(bf16) "
        f"memory-limit={override_bytes}: {status} "
        f"(fixed={overridden['fixed_bytes'] / GIB:.4f} GiB, "
        f"limit={overridden['memory_limit_bytes'] / GIB:.4f} GiB, "
        f"margin={overridden['unallocated_bytes'] / GIB:+.4f} GiB)",
        file=sys.stderr,
    )

    payload = {
        "schema": "mtplx-t3-kselect-88e16k-bf16-admission-preflight-v1",
        "generated": "2026-07-22",
        "note": (
            "CPU-only admission preflight for the bf16-KV K-SELECTION "
            "reference window (88-efficient x 16k real-prefill bf16 KV, "
            "AR+K1/K2/K3 x 3 reps). Two-phase: (1) no-override gate read to "
            "learn declared_limit_bytes + fixed_bytes, (2) derived-override "
            "gate that must ADMIT before any GPU window opens."
        ),
        "preset": PRESET,
        "max_live_kv_tokens": args.kv,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "kv_bytes_per_token_bf16": KV_BYTES_PER_TOKEN_BF16,
        "declared_limit_bytes": declared_limit_bytes,
        "derived_override_bytes": override_bytes,
        "derived_override_gib": override_bytes / GIB,
        "phase1_no_override": base,
        "phase2_overridden": overridden,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "ADMIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
