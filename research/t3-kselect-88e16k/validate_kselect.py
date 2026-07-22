#!/usr/bin/env python3
"""CPU-only, targeted-field validator + aggregator for the T3 K-SELECTION
receipts (2026-07-22): hy3-oq2e-rq4-88e x 16384 real-prefill x kv4, AR +
K1/K2/K3, 3 reps.

Reads evals/tier2/t3_kselect_88e16k_kv4_rep{1,2,3}.json and extracts ONLY
the specific top-level/per-observation fields needed to validate and score
the run -- never the whole `configuration` block (a cat-secrets hook blocks
wholesale reads of that block). Same pattern as research/t3-pilot-
realprefill/validate_pilot.py / research/t3-80/aggregate.py.

Validates per row:
  1. prompt_identity.token_count == 16384 and prompt_release_valid == True
     (real prefill, not KV-budget-only).
  2. The 6 mandatory metrics populated: ingestion_tok_s, prompt_eval_time_s,
     decode_tok_s, conditional_hit_rate + acceptance_by_depth (null on AR
     is CORRECT, not a failure), decode_expert_cache_hit_rate,
     hard_peak_memory_bytes (model-level).
  3. kv_quant == "q4" via model["runtime_config"]["kv_quant"].
  4. hard_peak_memory_bytes <= 96 GiB limit.

Then prints a per-lane (AR/K1/K2/K3) x per-rep table of the 6 metrics, the
per-lane means (decode_tok_s and end_to_end_tok_s, acceptance by depth), and
the K verdict (lane with the highest mean decode_tok_s), with margin over
the runner-up.

Usage:
    PYTHONPATH=<worktree> python3 research/t3-kselect-88e16k/validate_kselect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "evals" / "tier2"
GIB = 1024**3
LIMIT_GIB = 96.0
EXPECTED_TOKEN_COUNT = 16384
EXPECTED_KV_QUANT = "q4"
REPS = (1, 2, 3)

LANE_NAMES = {0: "AR", 1: "K1", 2: "K2", 3: "K3"}


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
        "end_to_end_tok_s": observation.get("end_to_end_tok_s"),
        "conditional_hit_rate": observation.get("conditional_hit_rate"),
        "acceptance_by_depth": observation.get("acceptance_by_depth"),
        "decode_expert_cache_hit_rate": observation.get("decode_expert_cache_hit_rate"),
        "peak_memory_bytes_at_row": observation.get("peak_memory_bytes"),
    }


def main() -> int:
    all_ok = True
    per_rep: dict[int, dict[str, Any]] = {}

    for rep in REPS:
        path = OUTDIR / f"t3_kselect_88e16k_kv4_rep{rep}.json"
        if not path.is_file():
            print(f"[validate] MISSING receipt: {path}", file=sys.stderr)
            all_ok = False
            continue
        with path.open() as fh:
            payload = json.load(fh)
        models = payload.get("models") or []
        if not models:
            print(f"[validate] rep{rep}: receipt has no models[] -- FAIL", file=sys.stderr)
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

    if not per_rep:
        print("[validate] NO receipts found at all -- FAIL", file=sys.stderr)
        return 2

    print("=" * 78)
    print("T3 K-SELECTION validation: hy3-oq2e-rq4-88e x 16384 real-prefill x kv4")
    print("=" * 78)

    lane_metrics: dict[str, dict[str, list[float]]] = {}
    lane_acceptance: dict[str, list[dict[str, Any]]] = {}

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
        peak_ok = info["hard_peak_gib"] is not None and info["hard_peak_gib"] <= LIMIT_GIB
        all_ok &= kv_quant_ok
        all_ok &= peak_ok
        print(f"kv_quant == {EXPECTED_KV_QUANT!r}: {'PASS' if kv_quant_ok else 'FAIL'}")
        print(f"hard_peak <= {LIMIT_GIB} GiB: {'PASS' if peak_ok else 'FAIL'}")

        for row in info["rows"]:
            depth = row["requested_depth"]
            lane = LANE_NAMES.get(depth, f"depth{depth}")
            is_ar_row = depth == 0
            token_count_ok = row["prompt_identity_token_count"] == EXPECTED_TOKEN_COUNT
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
            metrics_ok = bool(
                base_metrics_ok
                and row["decode_expert_cache_hit_rate"] is not None
                and speculative_metrics_ok
            )
            row_ok = token_count_ok and release_valid_ok and metrics_ok
            all_ok &= row_ok
            print(
                f"  lane={lane:3s} finish={row['finish_reason']!s:12s} "
                f"token_count={row['prompt_identity_token_count']} "
                f"release_valid={row['prompt_release_valid']} "
                f"ingestion_tok_s={row['ingestion_tok_s']} "
                f"prompt_eval_time_s={row['prompt_eval_time_s']} "
                f"decode_tok_s={row['decode_tok_s']} "
                f"end_to_end_tok_s={row['end_to_end_tok_s']} "
                f"cond_hit={row['conditional_hit_rate']} "
                f"cache_hit={row['decode_expert_cache_hit_rate']} "
                f"acceptance_by_depth={row['acceptance_by_depth']} "
                f"-> {'PASS' if row_ok else 'FAIL'}"
            )

            lane_metrics.setdefault(lane, {}).setdefault("decode_tok_s", []).append(row["decode_tok_s"])
            if row["end_to_end_tok_s"] is not None:
                lane_metrics[lane].setdefault("end_to_end_tok_s", []).append(row["end_to_end_tok_s"])
            if row["ingestion_tok_s"] is not None:
                lane_metrics[lane].setdefault("ingestion_tok_s", []).append(row["ingestion_tok_s"])
            if row["prompt_eval_time_s"] is not None:
                lane_metrics[lane].setdefault("prompt_eval_time_s", []).append(row["prompt_eval_time_s"])
            if row["decode_expert_cache_hit_rate"] is not None:
                lane_metrics[lane].setdefault("decode_expert_cache_hit_rate", []).append(row["decode_expert_cache_hit_rate"])
            if row["conditional_hit_rate"] is not None:
                lane_metrics[lane].setdefault("conditional_hit_rate", []).append(row["conditional_hit_rate"])
            if row["acceptance_by_depth"]:
                lane_acceptance.setdefault(lane, []).append(row["acceptance_by_depth"])

    print("\n" + "=" * 78)
    print("PER-LANE MEANS (across available reps)")
    print("=" * 78)
    lane_order = ["AR", "K1", "K2", "K3"]
    means: dict[str, float] = {}
    for lane in lane_order:
        m = lane_metrics.get(lane)
        if not m:
            print(f"{lane}: NO DATA")
            continue
        def mean(key: str) -> float | None:
            vals = m.get(key)
            return (sum(vals) / len(vals)) if vals else None
        decode_mean = mean("decode_tok_s")
        e2e_mean = mean("end_to_end_tok_s")
        ing_mean = mean("ingestion_tok_s")
        ttft_mean = mean("prompt_eval_time_s")
        cache_hit_mean = mean("decode_expert_cache_hit_rate")
        cond_hit_mean = mean("conditional_hit_rate")
        n = len(m.get("decode_tok_s", []))
        if decode_mean is not None:
            means[lane] = decode_mean
        print(
            f"{lane}: n={n} decode_tok_s_mean={decode_mean} end_to_end_tok_s_mean={e2e_mean} "
            f"ingestion_tok_s_mean={ing_mean} prompt_eval_time_s_mean={ttft_mean} "
            f"decode_expert_cache_hit_rate_mean={cache_hit_mean} "
            f"conditional_hit_rate_mean={cond_hit_mean} "
            f"acceptance_by_depth_samples={lane_acceptance.get(lane)}"
        )

    print("\n" + "=" * 78)
    print("K VERDICT (by mean decode_tok_s)")
    print("=" * 78)
    if means:
        ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        winner, winner_val = ranked[0]
        print(f"winner: {winner} ({winner_val:.4f} decode_tok_s)")
        for lane, val in ranked[1:]:
            pct = (winner_val - val) / val * 100 if val else float("nan")
            print(f"  vs {lane}: {val:.4f} decode_tok_s (winner +{pct:.2f}%)")
    else:
        print("no lane means available -- cannot compute verdict")
        all_ok = False

    print("\n" + "=" * 78)
    print(f"OVERALL VERDICT: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
