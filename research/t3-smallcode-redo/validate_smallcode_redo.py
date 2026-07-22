#!/usr/bin/env python3
"""CPU-only, targeted-field validator + aggregator for the T3 small-code
arm-of-record receipts (2026-07-22): contexts=256, output-tokens=256, K1
(+free AR), bf16 KV, 3 reps each, against the three already-done envelopes --

  ARM 88e: hy3-oq2e-rq4-88e (islands 79, full residency).
           evals/tier2/t3_smallcode_88e_rep{1,2,3}.json.
  ARM 80:  hy3-oq2e-rq4-80 (islands 69, 10 streamed).
           evals/tier2/t3_smallcode_80_rep{1,2,3}.json.
  ARM 64:  hy3-oq2e-rq4-64-cachelru (ZERO islands, cache-heavy LRU).
           evals/tier2/t3_smallcode_64_rep{1,2,3}.json.

Reads ONLY the specific top-level/per-observation fields needed to validate
and score the run -- never the whole `configuration` block (a cat-secrets
hook blocks wholesale reads of that block). Same pattern as
research/t3-80col/validate_80col.py, research/t3-64col/validate_64col.py,
and research/t3-88col-1024-nat/validate_88col.py.

Validates per row:
  1. prompt_identity.token_count == 256.
  2. prompt_release_valid is True.
  3. The mandatory metrics emitted: ingestion_tok_s, prompt_eval_time_s,
     decode_tok_s, conditional_hit_rate + acceptance_by_depth (null on AR
     is CORRECT), hard_peak_memory_bytes (model-level). decode_expert_cache_hit_rate
     is gated PER-ARM: EXPECTED non-null at 80 (islands 69) and 64-cachelru
     (ZERO islands, all layers stream) -- gated "must be non-null" there;
     structurally None/absent at 88e (full residency, 0 streamed) -- NOT
     gated there, same as every other 88e window.
  4. kv_quant is None/off (bf16) via model["runtime_config"]["kv_quant"].
  5. hard_peak_memory_bytes <= this arm's own preset declared admission
     limit (88e 96 GiB, 80 87 GiB, 64-cachelru 71 GiB; no override used or
     needed for any arm here).

Then prints a per-arm x per-lane (AR/K1) x per-rep table of the metrics and
per-arm/per-lane means, including decode_expert_cache_hit_rate where
applicable.

Usage:
    PYTHONPATH=<worktree> python3 research/t3-smallcode-redo/validate_smallcode_redo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "evals" / "tier2"
GIB = 1024**3
EXPECTED_KV_QUANT = None  # bf16 / off
EXPECTED_TOKEN_COUNT = 256
REPS = (1, 2, 3)
LANE_NAMES = {0: "AR", 1: "K1"}

ARMS = {
    "88e": {
        "prefix": "t3_smallcode_88e_rep",
        "limit_gib": 96.0000,
        "label": "ARM smallcode-88e (hy3-oq2e-rq4-88e, full residency)",
        "expect_cache_hit": False,
    },
    "80": {
        "prefix": "t3_smallcode_80_rep",
        "limit_gib": 87.0000,
        "label": "ARM smallcode-80 (hy3-oq2e-rq4-80, islands 69)",
        "expect_cache_hit": True,
    },
    "64": {
        "prefix": "t3_smallcode_64_rep",
        "limit_gib": 71.0000,
        "label": "ARM smallcode-64 (hy3-oq2e-rq4-64-cachelru, zero islands)",
        "expect_cache_hit": True,
    },
}


def _extract_row(observation: dict[str, Any]) -> dict[str, Any]:
    identity = observation.get("prompt_identity") or {}
    return {
        "cell": observation.get("cell"),
        "requested_depth": observation.get("requested_depth"),
        "finish_reason": observation.get("finish_reason"),
        "token_count": identity.get("token_count"),
        "prompt_policy": identity.get("prompt_policy"),
        "prompt_release_valid": identity.get("prompt_release_valid"),
        "ingestion_tok_s": observation.get("ingestion_tok_s"),
        "prompt_eval_time_s": observation.get("prompt_eval_time_s"),
        "decode_tok_s": observation.get("decode_tok_s"),
        "end_to_end_tok_s": observation.get("end_to_end_tok_s"),
        "conditional_hit_rate": observation.get("conditional_hit_rate"),
        "acceptance_by_depth": observation.get("acceptance_by_depth"),
        "decode_expert_cache_hit_rate": observation.get("decode_expert_cache_hit_rate"),
    }


def _validate_arm(arm_key: str, spec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    all_ok = True
    per_rep: dict[int, dict[str, Any]] = {}
    for rep in REPS:
        path = OUTDIR / f"{spec['prefix']}{rep}.json"
        if not path.is_file():
            print(f"[validate:{arm_key}] MISSING receipt: {path}", file=sys.stderr)
            all_ok = False
            continue
        with path.open() as fh:
            payload = json.load(fh)
        models = payload.get("models") or []
        if not models:
            print(f"[validate:{arm_key}] rep{rep}: no models[] -- FAIL", file=sys.stderr)
            all_ok = False
            continue
        model = models[0]
        runtime_config = model.get("runtime_config") or {}
        kv_quant = runtime_config.get("kv_quant")
        hard_peak = model.get("hard_peak_memory_bytes")
        hard_peak_gib = (hard_peak / GIB) if hard_peak is not None else None
        rows = [_extract_row(o) for o in (model.get("observations") or [])]
        per_rep[rep] = {
            "payload_passed": bool(payload.get("passed")),
            "kv_quant": kv_quant,
            "hard_peak": hard_peak,
            "hard_peak_gib": hard_peak_gib,
            "rows": rows,
        }

    print("=" * 78)
    print(f"{spec['label']} ({spec['prefix']}N.json)")
    print("=" * 78)

    lane_metrics: dict[str, dict[str, list[float]]] = {}
    lane_acceptance: dict[str, list[dict[str, Any]]] = {}
    peaks_gib: list[float] = []

    for rep in REPS:
        info = per_rep.get(rep)
        if info is None:
            continue
        print(f"\n--- rep {rep} ---")
        print(f"payload.passed: {info['payload_passed']}")
        print(f"model.runtime_config.kv_quant: {info['kv_quant']!r}")
        print(f"model.hard_peak_memory_bytes: {info['hard_peak']} ({info['hard_peak_gib']:.4f} GiB)")
        all_ok &= info["payload_passed"]
        kv_quant_ok = info["kv_quant"] == EXPECTED_KV_QUANT
        peak_ok = info["hard_peak_gib"] is not None and info["hard_peak_gib"] <= spec["limit_gib"]
        all_ok &= kv_quant_ok
        all_ok &= peak_ok
        if info["hard_peak_gib"] is not None:
            peaks_gib.append(info["hard_peak_gib"])
        print(f"kv_quant == {EXPECTED_KV_QUANT!r} (bf16/off): {'PASS' if kv_quant_ok else 'FAIL'}")
        print(f"hard_peak <= {spec['limit_gib']:.4f} GiB: {'PASS' if peak_ok else 'FAIL'}")

        for row in info["rows"]:
            depth = row["requested_depth"]
            lane = LANE_NAMES.get(depth, f"depth{depth}")
            is_ar_row = depth == 0
            token_count_ok = row["token_count"] == EXPECTED_TOKEN_COUNT
            release_valid_ok = row["prompt_release_valid"] is True
            base_metrics_ok = all(
                row[k] is not None
                for k in ("ingestion_tok_s", "prompt_eval_time_s", "decode_tok_s")
            )
            speculative_metrics_ok = (
                True
                if is_ar_row
                else (
                    row["conditional_hit_rate"] is not None
                    and bool(row["acceptance_by_depth"])
                )
            )
            cache_hit_present_ok = (
                row["decode_expert_cache_hit_rate"] is not None
                if spec["expect_cache_hit"]
                else True
            )
            metrics_ok = bool(base_metrics_ok and speculative_metrics_ok and cache_hit_present_ok)
            row_ok = token_count_ok and release_valid_ok and metrics_ok
            all_ok &= row_ok
            print(
                f"  lane={lane:3s} finish={row['finish_reason']!s:12s} "
                f"token_count={row['token_count']} "
                f"policy={row['prompt_policy']} "
                f"release_valid={row['prompt_release_valid']} "
                f"ingestion_tok_s={row['ingestion_tok_s']} "
                f"prompt_eval_time_s={row['prompt_eval_time_s']} "
                f"decode_tok_s={row['decode_tok_s']} "
                f"cond_hit={row['conditional_hit_rate']} "
                f"cache_hit={row['decode_expert_cache_hit_rate']} "
                f"acceptance_by_depth={row['acceptance_by_depth']} "
                f"-> {'PASS' if row_ok else 'FAIL'}"
            )

            lm = lane_metrics.setdefault(lane, {})
            lm.setdefault("decode_tok_s", []).append(row["decode_tok_s"])
            if row["end_to_end_tok_s"] is not None:
                lm.setdefault("end_to_end_tok_s", []).append(row["end_to_end_tok_s"])
            if row["ingestion_tok_s"] is not None:
                lm.setdefault("ingestion_tok_s", []).append(row["ingestion_tok_s"])
            if row["prompt_eval_time_s"] is not None:
                lm.setdefault("prompt_eval_time_s", []).append(row["prompt_eval_time_s"])
            if row["decode_expert_cache_hit_rate"] is not None:
                lm.setdefault("decode_expert_cache_hit_rate", []).append(row["decode_expert_cache_hit_rate"])
            if row["conditional_hit_rate"] is not None:
                lm.setdefault("conditional_hit_rate", []).append(row["conditional_hit_rate"])
            if row["acceptance_by_depth"]:
                lane_acceptance.setdefault(lane, []).append(row["acceptance_by_depth"])

    print("\nPER-LANE MEANS")
    means: dict[str, dict[str, float | None]] = {}
    for lane in ("AR", "K1"):
        m = lane_metrics.get(lane)
        if not m:
            print(f"{lane}: NO DATA")
            continue

        def mean(key: str) -> float | None:
            vals = m.get(key)
            return (sum(vals) / len(vals)) if vals else None

        decode_mean = mean("decode_tok_s")
        ing_mean = mean("ingestion_tok_s")
        ttft_mean = mean("prompt_eval_time_s")
        cache_hit_mean = mean("decode_expert_cache_hit_rate")
        cond_hit_mean = mean("conditional_hit_rate")
        n = len(m.get("decode_tok_s", []))
        means[lane] = {
            "n": n,
            "decode_tok_s_mean": decode_mean,
            "ingestion_tok_s_mean": ing_mean,
            "prompt_eval_time_s_mean": ttft_mean,
            "decode_expert_cache_hit_rate_mean": cache_hit_mean,
            "conditional_hit_rate_mean": cond_hit_mean,
        }
        print(
            f"{lane}: n={n} decode_tok_s_mean={decode_mean} "
            f"ingestion_tok_s_mean={ing_mean} prompt_eval_time_s_mean={ttft_mean} "
            f"decode_expert_cache_hit_rate_mean={cache_hit_mean} "
            f"conditional_hit_rate_mean={cond_hit_mean} "
            f"acceptance_by_depth_samples={lane_acceptance.get(lane)}"
        )

    print(f"\npeak_gib_all_reps: {peaks_gib}")
    print(f"ARM VERDICT: {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok, means


def main() -> int:
    overall_ok = True
    all_means: dict[str, Any] = {}
    for arm_key, spec in ARMS.items():
        ok, means = _validate_arm(arm_key, spec)
        overall_ok &= ok
        all_means[arm_key] = means

    print("=" * 78)
    print(f"OVERALL VERDICT: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
