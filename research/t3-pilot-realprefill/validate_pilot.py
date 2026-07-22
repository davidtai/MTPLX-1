#!/usr/bin/env python3
"""CPU-only, targeted-field validator for the T3 real-prefill PILOT receipt
(2026-07-22). Mechanics validation, NOT a cell of record.

Reads evals/tier2/pilot_realprefill_32cachelru_16k_kv4.json and extracts
ONLY the specific top-level/per-observation fields needed to validate the
run -- never the whole `configuration` block (a cat-secrets hook blocks
wholesale reads of that block; this script never touches it). Same pattern
as research/t3-80/aggregate.py: json.load(), pick named keys, print a
summary.

Validates:
  1. prompt_identity.token_count == 16384 and prompt_release_valid == True
     on every observation (real prefill, not KV-budget-only).
  2. The 6 mandatory metrics populated on every observation: ingestion_tok_s,
     prompt_eval_time_s, decode_tok_s, conditional_hit_rate +
     acceptance_by_depth, decode_expert_cache_hit_rate,
     hard_peak_memory_bytes (model-level running peak).
  3. kv_quant == "q4" via model["runtime_config"]["kv_quant"] -- a named
     top-level field of model_payload, NOT the blocked "configuration" key.
  4. hard_peak_memory_bytes stays within the preset's declared memory-limit
     (39 GiB class for hy3-oq2e-rq4-32-cachelru).

Usage:
    PYTHONPATH=<worktree> python3 research/t3-pilot-realprefill/validate_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO_ROOT / "evals" / "tier2" / "pilot_realprefill_32cachelru_16k_kv4.json"
GIB = 1024**3
PRESET_MEMORY_LIMIT_GIB = 39.0
EXPECTED_TOKEN_COUNT = 16384
EXPECTED_KV_QUANT = "q4"


def _extract_row(observation: dict[str, Any]) -> dict[str, Any]:
    identity = observation.get("prompt_identity") or {}
    return {
        "cell": observation.get("cell"),
        "requested_depth": observation.get("requested_depth"),
        "effective_depth": observation.get("effective_depth"),
        "finish_reason": observation.get("finish_reason"),
        "prompt_identity_token_count": identity.get("token_count"),
        "prompt_release_valid": identity.get("prompt_release_valid"),
        "ingestion_tok_s": observation.get("ingestion_tok_s"),
        "prompt_eval_time_s": observation.get("prompt_eval_time_s"),
        "decode_tok_s": observation.get("decode_tok_s"),
        "conditional_hit_rate": observation.get("conditional_hit_rate"),
        "acceptance_by_depth": observation.get("acceptance_by_depth"),
        "decode_expert_cache_hit_rate": observation.get("decode_expert_cache_hit_rate"),
        "peak_memory_bytes_at_row": observation.get("peak_memory_bytes"),
    }


def main() -> int:
    if not RECEIPT_PATH.is_file():
        print(f"[validate] MISSING receipt: {RECEIPT_PATH}", file=sys.stderr)
        return 2

    with RECEIPT_PATH.open() as fh:
        payload = json.load(fh)

    overall_passed = bool(payload.get("passed"))
    models = payload.get("models") or []
    if not models:
        print("[validate] receipt has no models[] -- FAIL", file=sys.stderr)
        return 2
    model = models[0]

    runtime_config = model.get("runtime_config") or {}
    kv_quant = runtime_config.get("kv_quant")
    hard_peak = model.get("hard_peak_memory_bytes")
    hard_peak_gib = (hard_peak / GIB) if hard_peak is not None else None

    rows = [_extract_row(o) for o in (model.get("observations") or [])]

    print("=" * 70)
    print("T3 real-prefill PILOT validation (mechanics check, not a cell of record)")
    print("=" * 70)
    print(f"receipt: {RECEIPT_PATH}")
    print(f"payload.passed: {overall_passed}")
    print(f"model.hard_peak_memory_bytes: {hard_peak} ({hard_peak_gib:.4f} GiB)")
    print(f"model.runtime_config.kv_quant: {kv_quant!r}")
    print(f"observations: {len(rows)}")
    print()

    all_ok = True
    all_ok &= overall_passed

    for row in rows:
        print(f"--- cell={row['cell']} requested_depth={row['requested_depth']} "
              f"effective_depth={row['effective_depth']} finish={row['finish_reason']} ---")
        token_count_ok = row["prompt_identity_token_count"] == EXPECTED_TOKEN_COUNT
        release_valid_ok = row["prompt_release_valid"] is True
        is_ar_row = row["requested_depth"] == 0
        base_metrics_ok = all(
            row[k] is not None
            for k in ("ingestion_tok_s", "prompt_eval_time_s", "decode_tok_s")
        )
        # conditional_hit_rate / acceptance_by_depth are speculative-decode-
        # only fields: AR (depth=0) has no drafts to evaluate, so None/[] is
        # the CORRECT and expected shape there, not a failure. The mandatory-
        # metrics gate (per the brief) applies to the K3 arm.
        speculative_metrics_ok = (
            True
            if is_ar_row
            else (
                row["conditional_hit_rate"] is not None
                and bool(row["acceptance_by_depth"])
            )
        )
        metrics_ok = bool(
            base_metrics_ok
            and row["decode_expert_cache_hit_rate"] is not None
            and speculative_metrics_ok
        )
        print(f"  prompt_identity.token_count == {EXPECTED_TOKEN_COUNT}: "
              f"{row['prompt_identity_token_count']} -> {'PASS' if token_count_ok else 'FAIL'}")
        print(f"  prompt_release_valid == True: "
              f"{row['prompt_release_valid']} -> {'PASS' if release_valid_ok else 'FAIL'}")
        print(f"  ingestion_tok_s (prefill tok/s): {row['ingestion_tok_s']}")
        print(f"  prompt_eval_time_s (TTFT): {row['prompt_eval_time_s']}")
        print(f"  decode_tok_s: {row['decode_tok_s']}")
        print(f"  conditional_hit_rate: {row['conditional_hit_rate']}")
        print(f"  acceptance_by_depth: {row['acceptance_by_depth']}")
        print(f"  decode_expert_cache_hit_rate: {row['decode_expert_cache_hit_rate']}")
        print(f"  6-metrics populated: {'PASS' if metrics_ok else 'FAIL'}")
        print()
        all_ok &= token_count_ok
        all_ok &= release_valid_ok
        all_ok &= metrics_ok

    kv_quant_ok = kv_quant == EXPECTED_KV_QUANT
    peak_ok = hard_peak_gib is not None and hard_peak_gib < PRESET_MEMORY_LIMIT_GIB
    print(f"kv_quant == {EXPECTED_KV_QUANT!r}: {kv_quant!r} -> {'PASS' if kv_quant_ok else 'FAIL'}")
    print(f"hard_peak {hard_peak_gib:.4f} GiB < {PRESET_MEMORY_LIMIT_GIB} GiB limit: "
          f"{'PASS' if peak_ok else 'FAIL'}")
    all_ok &= kv_quant_ok
    all_ok &= peak_ok

    print()
    print(f"OVERALL VERDICT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
