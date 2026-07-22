#!/usr/bin/env python3
"""CPU-only admission-gate preflight for the T3 **64-envelope column**
real-prefill matrix (2026-07-22): `hy3-oq2e-rq4-64-cachelru` (ZERO islands,
all 79 routed layers streamed through one shared LRU cache -- David's
directive, envelopes <=64 GiB never use islands), real long-code prefill at
contexts 1024/16384/65536, decode 256, K1 (+free AR), bf16 KV throughout,
plus one naturalistic ~265-token arm. Sibling of research/t3-80col/preflight.py
(islands 69), retargeted at the cache-heavy LRU preset; same gate machinery,
same guarantee: no MLX buffer is ever allocated, no weight byte is ever read
(ExpertStreamingRuntime.open / apis.load() are never reached; the gate stops
at `config.memory_plan()` / `plan.fits_fixed`, via
`research/envelope_admission_sweep.py`'s `evaluate_admission()` -- imported
directly, never re-implemented).

Per-arm KV handling (same family formula as every sibling script):
  - 1024 arm and naturalistic (265) arm: both leave --max-live-kv-tokens at
    the preset's own declared default (4096) -- 1024+256=1280 and
    265+256=521 tokens are both comfortably inside it. No override.
  - 16384 arm: --max-live-kv-tokens 16640 (contexts+256).
  - 65536 arm: --max-live-kv-tokens 65792 (contexts+256).
  Override, if the base check REJECTs at the preset's own declared limit:
        override_bytes = declared_limit_bytes
                          + (kv_tokens - CONTROL_KV_TOKENS) * KV_BYTES_PER_TOKEN

NOTE (measured by THIS script, 2026-07-22): unlike the islands-69 80-column
preset (whose tiny ~2 GiB cache leaves no room to absorb KV growth, forcing
an override past its own declared limit for every context arm above 1024),
this cache-heavy LRU preset's ~50 GiB elastic cache pool SHRINKS to absorb
KV growth while staying inside its OWN declared 71 GiB limit -- fixed_bytes
grows with kv_tokens (20.64 -> 24.47 -> 39.47 GiB) while persistent_cache_bytes
falls in step (49.99 -> 46.48 -> 31.25 GiB), and every arm here ADMITs at the
preset's own declared limit with NO override needed at all (confirmed by
running this exact script; see evals/tier2/t3_64col_admission_preflight.json).
Do not assume this ahead of running the harness for other columns -- the
32-envelope column's 65536 arm needs a real override (see
research/t3-32col/preflight.py) once the shrinking cache bottoms out at zero
and KV growth still exceeds the declared limit.

WIRED-KNOB CAP (mandatory, this window's launch instructions): any arm
whose derived override exceeds 112 GiB (114688 MiB, the sanctioned wired
knob) is marked SKIP -- never exceed the knob, regardless of what the
harness's own ADMIT/REJECT gate says (the harness only knows about the
preset's own declared limit chain, not the box-wide wired-knob ceiling).

Usage:
    PYTHONPATH=<worktree> python3 research/t3-64col/preflight.py \\
        --output evals/tier2/t3_64col_admission_preflight.json
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
PRESET = "hy3-oq2e-rq4-64-cachelru"
WIRED_KNOB_LIMIT_GIB = 112.0  # 114688 MiB, sanctioned ceiling; never exceeded
SIGNOFF_CEILING_GIB = 90.0  # David's rule: >~90 GiB projected peak here needs a fresh ask

# (arm_name, contexts, output_tokens, max_live_kv_tokens or None-for-default)
ARMS: tuple[tuple[str, int, int, int | None], ...] = (
    ("1024", 1024, 256, None),
    ("16384", 16384, 256, 16640),
    ("65536", 65536, 256, 65792),
    ("naturalistic", 265, 256, None),
)

# Rough wall-time model (ANALYSIS section below) -- see the "analysis" key in
# this script's JSON output for the full derivation and caveats.
PREFILL_TOK_S_ASSUMED = 75.0
LOAD_MINUTES_ASSUMED = 7.5
DECODE_TOK_S_BY_ARM = {
    "1024": 10.0,
    "16384": 4.0,
    "65536": 3.0,
    "naturalistic": 10.0,
}
OUTPUT_TOKENS_ASSUMED = 256


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
        cell["final_memory_limit_bytes"] = base["memory_limit_bytes"]
        cell["final_status"] = "ADMIT"
        cell["knob_exceeded"] = False
        return cell

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
    if overridden["status"] != "ADMIT":
        cell["final_status"] = "SKIP"
        cell["skip_reason"] = (
            f"still REJECTs at the derived override ({override_gib:.4f} GiB): "
            f"excess {overridden.get('excess_bytes')} bytes -- preset's own "
            "fixed footprint over-books its declared limit chain independent "
            "of the wired knob"
        )
    return cell


def build_analysis(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Rough (non-authoritative) wall-time projection per arm, derived from
    the only real-prefill receipt this preset family has: the zero-island
    32-cachelru pilot (evals/tier2/pilot_realprefill_32cachelru_16k_kv4.json)
    -- real 16384-token prefill measured 76.07/75.19 tok/s ingestion (kv4),
    TTFT ~215-218s, one model load (~7.5 min net of that window's own
    prefill+decode, per the pilot's 17m13s total wall time note in
    HANDOFF-hy3-matrix-restart.md). Assumptions, stated plainly:
      - PREFILL_TOK_S_ASSUMED=75: portable from kv4 to bf16 because KV
        quantization changes the KV cache's own bytes/bandwidth (a decode-
        time cost), not the weight-streaming cost that dominates prefill
        ingestion through a zero-island cache. This column's cache is
        MUCH larger (49.99 GiB vs the pilot's 17.97 GiB at 32-cachelru), so
        75 tok/s is likely a conservative (slow) floor here -- fewer
        forced re-reads once warm, if anything.
      - LOAD_MINUTES_ASSUMED=7.5: the pilot window's total (17m13s) minus
        its own two prefill+decode passes (AR depth-0 + K3 depth-3, both at
        the SAME 16384 context) leaves ~7.5 min as the one-time model-load
        component. Each rep in this column's inner script is a SEPARATE
        process invocation (fresh model load every time), so this cost is
        paid once PER REP, not once per arm or per column.
      - DECODE_TOK_S_BY_ARM: bf16 should decode faster than the pilot's kv4
        numbers (kv acceptance collapses under kv4 per the kv A/B receipt,
        36add87e) but there is no direct bf16 real-prefill decode
        measurement at this preset/context yet -- these are informed
        guesses, not receipts. 1024/naturalistic reuse the bench-shape
        64-cachelru K1 rate (~10.8 tok/s, evals/tier2/t3_64x16k_armB_lru_repN.json)
        since those contexts are short enough that KV growth barely bites.
        16384/65536 assume degradation with context (following the SAME
        AR-degrades-with-context pattern already measured on the 80-column,
        islands preset: AR 15.9 at 16k -> 9.2 at 64k, a ~42% drop) --
        picked round numbers (4 / 3 tok/s) deliberately on the slow side.
      - Each process invocation runs BOTH the AR (depth 0) and K1 (depth 1)
        lane at the SAME context (--hy3-depths 1 auto-runs AR alongside),
        i.e. two full prefill+decode passes per rep.
    Do not treat these numbers as anything but a rough planning budget.
    """

    per_arm: dict[str, Any] = {}
    total_minutes = 0.0
    for cell in cells:
        name = cell["arm"]
        if cell["final_status"] == "SKIP":
            per_arm[name] = {"skipped": True}
            continue
        contexts = cell["contexts"]
        decode_tok_s = DECODE_TOK_S_BY_ARM[name]
        prefill_s = contexts / PREFILL_TOK_S_ASSUMED
        decode_s = OUTPUT_TOKENS_ASSUMED / decode_tok_s
        per_pass_s = prefill_s + decode_s
        two_passes_s = 2 * per_pass_s  # AR + K1
        per_rep_minutes = LOAD_MINUTES_ASSUMED + two_passes_s / 60.0
        arm_total_minutes = 3 * per_rep_minutes  # 3 reps
        per_arm[name] = {
            "assumed_prefill_tok_s": PREFILL_TOK_S_ASSUMED,
            "assumed_decode_tok_s": decode_tok_s,
            "assumed_prefill_seconds_per_pass": round(prefill_s, 1),
            "assumed_decode_seconds_per_pass": round(decode_s, 1),
            "assumed_load_minutes_per_rep": LOAD_MINUTES_ASSUMED,
            "assumed_minutes_per_rep": round(per_rep_minutes, 1),
            "assumed_minutes_for_3_reps": round(arm_total_minutes, 1),
        }
        total_minutes += arm_total_minutes

    return {
        "method": (
            "load(7.5min, once per rep) + 2x(prefill=contexts/75tok_s + "
            "decode=256/assumed_decode_tok_s), x3 reps per arm; summed "
            "across the 4 arms for the column total. See this function's "
            "docstring for the full derivation and honest caveats."
        ),
        "per_arm": per_arm,
        "assumed_column_total_minutes": round(total_minutes, 1),
        "assumed_column_total_hours": round(total_minutes / 60.0, 2),
    }


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
        if status != "SKIP" and limit_gib > SIGNOFF_CEILING_GIB:
            print(
                f"[preflight] WARNING: arm={cell['arm']} limit {limit_gib:.4f} GiB exceeds the "
                f"{SIGNOFF_CEILING_GIB:.1f} GiB sub-85/90 sign-off band -- confirm with David "
                "before launching this arm.",
                file=sys.stderr,
            )

    runnable = [c for c in cells if c["final_status"] != "SKIP"]
    all_admit = all(c["final_status"] == "ADMIT" for c in runnable)
    any_skip = any(c["final_status"] == "SKIP" for c in cells)

    payload = {
        "schema": "mtplx-t3-64col-admission-preflight-v1",
        "generated": "2026-07-22",
        "note": (
            "CPU-only admission preflight for all four 64-envelope-column "
            "real-prefill arms (1024/16384/65536/naturalistic-265), bf16 KV "
            "throughout, hy3-oq2e-rq4-64-cachelru (ZERO islands, cache-heavy "
            "LRU -- David's <=64 GiB no-islands directive). Arms whose "
            "derived KV-delta override would exceed the 112 GiB wired knob, "
            "or that still REJECT even at that override, are marked SKIP "
            "and MUST NOT be run."
        ),
        "preset": PRESET,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "wired_knob_limit_gib": WIRED_KNOB_LIMIT_GIB,
        "signoff_ceiling_gib": SIGNOFF_CEILING_GIB,
        "cells": cells,
        "all_runnable_admit": all_admit,
        "any_skip": any_skip,
        "analysis": build_analysis(cells),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)

    return 0 if all_admit else 1


if __name__ == "__main__":
    raise SystemExit(main())
