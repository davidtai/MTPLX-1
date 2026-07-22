#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 **80-envelope column**
real-prefill matrix (2026-07-22): `hy3-oq2e-rq4-80` (islands 69, 10 streamed
layers), real long-code prefill at contexts 1024/16384/65536, decode 256,
K1 (+free AR), bf16 KV throughout, plus one naturalistic ~265-token arm.
Sibling of research/t3-88col-1024-nat/preflight.py (1024/naturalistic) and
research/t3-kselect-88e16k-bf16/preflight.py (the KV-delta override
derivation for a large context), generalized to all four 80-column arms in
one script.

Same gate, same guarantee as every preflight in this family: no MLX buffer
is ever allocated, no weight byte is ever read (ExpertStreamingRuntime.open
/ apis.load() are never reached; the gate stops at `config.memory_plan()` /
`plan.fits_fixed`, via `research/envelope_admission_sweep.py`'s
`evaluate_admission()` -- imported directly, never re-implemented).

Per-arm KV handling:
  - 1024 arm and naturalistic (265) arm: both leave --max-live-kv-tokens at
    the preset's own declared default (4096) -- 1024+256=1280 and
    265+256=521 tokens are both comfortably inside it. No override.
  - 16384 arm: --max-live-kv-tokens 16640 (contexts+256). The preset's own
    87 GiB declared limit REJECTs at this KV level (bf16 pricing); override
    derived by the SAME family formula as every sibling script:
        override_bytes = declared_limit_bytes
                          + (kv_tokens - CONTROL_KV_TOKENS) * KV_BYTES_PER_TOKEN
  - 65536 arm: --max-live-kv-tokens 65792 (contexts+256). Same formula.

WIRED-KNOB CAP (mandatory, this window's launch instructions): any arm
whose derived override exceeds 112 GiB (114688 MiB, the sanctioned wired
knob) is marked SKIP -- never exceed the knob, regardless of what the
harness's own ADMIT/REJECT gate says (the harness only knows about the
preset's own declared limit chain, not the box-wide wired-knob ceiling).

Usage:
    PYTHONPATH=<worktree> python3 research/t3-80col/preflight.py \\
        --output evals/tier2/t3_80col_admission_preflight.json
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
    evaluate_admission,
)

GIB = 1024**3
PRESET = "hy3-oq2e-rq4-80"
WIRED_KNOB_LIMIT_GIB = 112.0  # 114688 MiB, sanctioned ceiling; never exceeded

# (arm_name, contexts, output_tokens, max_live_kv_tokens or None-for-default)
ARMS: tuple[tuple[str, int, int, int | None], ...] = (
    ("1024", 1024, 256, None),
    ("16384", 16384, 256, 16640),
    ("65536", 65536, 256, 65792),
    ("naturalistic", 265, 256, None),
)


def evaluate_arm(name: str, contexts: int, output_tokens: int, kv_override_tokens: int | None) -> dict[str, Any]:
    kv_tokens = kv_override_tokens if kv_override_tokens is not None else CONTROL_KV_TOKENS
    base = evaluate_admission(PRESET, kv_tokens)
    cell: dict[str, Any] = {
        "arm": name,
        "contexts": contexts,
        "output_tokens": output_tokens,
        "max_live_kv_tokens": kv_tokens,
        "max_live_kv_tokens_is_override": kv_override_tokens is not None,
        "base": base,
    }

    if base["status"] == "ADMIT":
        # No override needed -- 1024/naturalistic arms land here (preset's
        # own 87 GiB declared limit already ADMITs at kv=4096).
        cell["final_memory_limit_bytes"] = base["memory_limit_bytes"]
        cell["final_status"] = "ADMIT"
        cell["knob_exceeded"] = False
        return cell

    # REJECT at the preset's own declared limit: derive the KV-delta
    # override, same formula as every sibling script in this family.
    declared_limit_bytes = base["memory_limit_bytes"]
    kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
    override_bytes = declared_limit_bytes + kv_delta_tokens * KV_BYTES_PER_TOKEN
    override_gib = override_bytes / GIB
    cell["kv_delta_tokens"] = kv_delta_tokens
    cell["derived_override_bytes"] = override_bytes
    cell["derived_override_gib"] = override_gib

    if override_gib > WIRED_KNOB_LIMIT_GIB:
        cell["final_status"] = "SKIP"
        cell["knob_exceeded"] = True
        cell["skip_reason"] = (
            f"derived override {override_gib:.4f} GiB exceeds the sanctioned "
            f"{WIRED_KNOB_LIMIT_GIB:.1f} GiB wired knob -- never exceed the knob"
        )
        return cell

    overridden = evaluate_admission(PRESET, kv_tokens, memory_limit_override_bytes=override_bytes)
    cell["overridden"] = overridden
    cell["final_memory_limit_bytes"] = override_bytes
    cell["final_status"] = overridden["status"]
    cell["knob_exceeded"] = False
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cells = [evaluate_arm(*arm) for arm in ARMS]

    for cell in cells:
        status = cell["final_status"]
        limit_gib = cell.get("final_memory_limit_bytes", cell["base"]["memory_limit_bytes"]) / GIB
        fixed_gib = cell["base"]["fixed_bytes"] / GIB
        print(
            f"[preflight] arm={cell['arm']:12} contexts={cell['contexts']:6} "
            f"kv={cell['max_live_kv_tokens']:6} override={cell['max_live_kv_tokens_is_override']!s:5} "
            f"-> {status:6} fixed={fixed_gib:8.4f} GiB limit={limit_gib:8.4f} GiB "
            f"knob_exceeded={cell['knob_exceeded']}",
            file=sys.stderr,
        )

    runnable = [c for c in cells if c["final_status"] != "SKIP"]
    all_admit = all(c["final_status"] == "ADMIT" for c in runnable)
    any_skip = any(c["final_status"] == "SKIP" for c in cells)

    payload = {
        "schema": "mtplx-t3-80col-admission-preflight-v1",
        "generated": "2026-07-22",
        "note": (
            "CPU-only admission preflight for all four 80-envelope-column "
            "real-prefill arms (1024/16384/65536/naturalistic-265), bf16 KV "
            "throughout, hy3-oq2e-rq4-80 (islands 69). Arms whose derived "
            "KV-delta override would exceed the 112 GiB wired knob are "
            "marked SKIP and MUST NOT be run."
        ),
        "preset": PRESET,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "wired_knob_limit_gib": WIRED_KNOB_LIMIT_GIB,
        "cells": cells,
        "all_runnable_admit": all_admit,
        "any_skip": any_skip,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)

    return 0 if all_admit else 1


if __name__ == "__main__":
    raise SystemExit(main())
