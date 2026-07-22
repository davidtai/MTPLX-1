#!/usr/bin/env python3
"""Aggregate the 3-rep receipts from research/t3-64-ab/run_window_inner.sh
into the canonical per-lane receipts the T3 64 GiB island-vs-cache A/B
mission asks for. CPU-only (reads JSON/log files already on disk; no MLX,
no GPU touch -- safe to run any time, including outside a guarded window).

Per lane (armA, armB_frequency, armB_lru, patch_k3exact), reads
evals/tier2/t3_64x16k_{label}_rep{1,2,3}.json (patch_k3exact instead reads
t3_64x32k_k3exact_rep{1,2,3}.json), extracts the per-cell (AR/K1/K2/K3, or
AR/K3 for the patch lane) headline metrics from each rep's single
`models[0].observations[*]` list, and writes:

  - evals/tier2/t3_64x16k_armA.json / t3_64x16k_armB_{frequency,lru}.json /
    t3_64x32k_k3exact.json -- one canonical JSON per lane with a per-cell
    table (mean/min/max across the 3 retained reps) plus each rep's raw
    per-cell row for full provenance.
  - the matching .log -- the 3 rep logs concatenated with rep-boundary
    markers, so the canonical log path still carries every rep's full
    console output.

Also prints a combined summary table (all lanes x all cells) to stdout for
the driving session's own report.

Usage:
    python3 research/t3-64-ab/aggregate.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "evals" / "tier2"

LANES: dict[str, dict[str, Any]] = {
    "armA": {
        "rep_prefix": "t3_64x16k_armA",
        "canonical": "t3_64x16k_armA",
        "cells": ["ar", "d1", "d2", "d3"],
    },
    "armB_frequency": {
        "rep_prefix": "t3_64x16k_armB_frequency",
        "canonical": "t3_64x16k_armB_frequency",
        "cells": ["ar", "d1", "d2", "d3"],
    },
    "armB_lru": {
        "rep_prefix": "t3_64x16k_armB_lru",
        "canonical": "t3_64x16k_armB_lru",
        "cells": ["ar", "d1", "d2", "d3"],
    },
    "patch_k3exact": {
        "rep_prefix": "t3_64x32k_k3exact",
        "canonical": "t3_64x32k_k3exact",
        "cells": ["ar", "d3"],
    },
}

REP_COUNT = 3


def _cell_key(observation: dict[str, Any]) -> str:
    return observation["cell"]


def _extract_row(observation: dict[str, Any]) -> dict[str, Any]:
    counters = observation.get("expert_streaming_counters") or {}
    loads = int(counters.get("persistent_loads", 0)) + int(counters.get("transient_loads", 0))
    decode_elapsed_s = observation.get("decode_elapsed_s")
    svc_ms_per_load = (
        (decode_elapsed_s / loads * 1000.0) if loads and decode_elapsed_s else None
    )
    ar_comparison = observation.get("ar_comparison") or {}
    generated_tokens = observation.get("generated_tokens")
    loads_per_token = (loads / generated_tokens) if generated_tokens else None
    row = {
        "cell": observation.get("cell"),
        "requested_depth": observation.get("requested_depth"),
        "effective_depth": observation.get("effective_depth"),
        "decode_tok_s": observation.get("decode_tok_s"),
        "end_to_end_tok_s": observation.get("end_to_end_tok_s"),
        "accepted_per_verify": observation.get("accepted_per_verify"),
        "decode_expert_cache_hit_rate": observation.get("decode_expert_cache_hit_rate"),
        "loads": loads,
        "loads_per_token": loads_per_token,
        "persistent_loads": counters.get("persistent_loads"),
        "transient_loads": counters.get("transient_loads"),
        "bytes_read": counters.get("bytes_read"),
        "svc_ms_per_load": svc_ms_per_load,
        "peak_memory_bytes": observation.get("peak_memory_bytes"),
        "decode_elapsed_s": decode_elapsed_s,
        "generated_tokens": observation.get("generated_tokens"),
        "finish_reason": observation.get("finish_reason"),
        "token_parity": ar_comparison.get("token_parity"),
        "first_divergence": ar_comparison.get("first_divergence"),
        "differing_token_count": ar_comparison.get("differing_token_count"),
        "divergence_attribution": ar_comparison.get("divergence_attribution"),
        "observed_token_sha256": ar_comparison.get("observed_token_sha256"),
        "reference_token_sha256": ar_comparison.get("reference_token_sha256"),
        "ar_comparison_status": ar_comparison.get("status"),
    }
    return row


def _load_rep(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as fh:
        return json.load(fh)


def _mean(values: list[float | int | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def aggregate_lane(lane_name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    rep_prefix = spec["rep_prefix"]
    rep_payloads: list[dict[str, Any]] = []
    rep_paths: list[Path] = []
    for rep in range(1, REP_COUNT + 1):
        path = OUTDIR / f"{rep_prefix}_rep{rep}.json"
        payload = _load_rep(path)
        if payload is None:
            print(f"[aggregate] MISSING {path}", file=sys.stderr)
            continue
        rep_payloads.append(payload)
        rep_paths.append(path)

    if not rep_payloads:
        print(f"[aggregate] lane {lane_name}: NO REPS FOUND, skipping", file=sys.stderr)
        return None

    passed_all = all(bool(p.get("passed")) for p in rep_payloads)

    # cell -> list of (rep_index, row)
    by_cell: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    model_meta_by_rep: list[dict[str, Any]] = []
    for rep_idx, payload in enumerate(rep_payloads, start=1):
        models = payload.get("models") or []
        if not models:
            continue
        model = models[0]
        model_meta_by_rep.append(
            {
                "rep": rep_idx,
                "load_count": model.get("load_count"),
                "load_peak_memory_bytes": model.get("load_peak_memory_bytes"),
                "hard_peak_memory_bytes": model.get("hard_peak_memory_bytes"),
                "model_root": model.get("model_root"),
                "manifest": model.get("manifest"),
                "runtime_config": model.get("runtime_config"),
            }
        )
        for observation in model.get("observations") or []:
            row = _extract_row(observation)
            by_cell.setdefault(row["cell"], []).append((rep_idx, row))

    cell_table: dict[str, Any] = {}
    for cell, rows in sorted(by_cell.items()):
        per_rep = [{"rep": rep_idx, **row} for rep_idx, row in rows]
        cell_table[cell] = {
            "per_rep": per_rep,
            "mean": {
                "decode_tok_s": _mean([r["decode_tok_s"] for _, r in rows]),
                "end_to_end_tok_s": _mean([r["end_to_end_tok_s"] for _, r in rows]),
                "accepted_per_verify": _mean([r["accepted_per_verify"] for _, r in rows]),
                "decode_expert_cache_hit_rate": _mean(
                    [r["decode_expert_cache_hit_rate"] for _, r in rows]
                ),
                "loads": _mean([r["loads"] for _, r in rows]),
                "loads_per_token": _mean([r["loads_per_token"] for _, r in rows]),
                "svc_ms_per_load": _mean([r["svc_ms_per_load"] for _, r in rows]),
                "peak_memory_bytes": _mean([r["peak_memory_bytes"] for _, r in rows]),
            },
            "min_decode_tok_s": min(
                (r["decode_tok_s"] for _, r in rows if r["decode_tok_s"] is not None),
                default=None,
            ),
            "max_decode_tok_s": max(
                (r["decode_tok_s"] for _, r in rows if r["decode_tok_s"] is not None),
                default=None,
            ),
            # token-hash / parity is a HARD property, not averaged: report
            # whether every rep agrees, and flag if any rep disagrees.
            "token_parity_all_reps": [r["token_parity"] for _, r in rows],
            "token_parity_consistent": len({r["token_parity"] for _, r in rows}) <= 1,
            "observed_token_sha256_all_reps": [r["observed_token_sha256"] for _, r in rows],
            "observed_token_sha256_consistent": len(
                {r["observed_token_sha256"] for _, r in rows}
            ) <= 1,
        }

    hard_peaks = [
        m["hard_peak_memory_bytes"]
        for m in model_meta_by_rep
        if m.get("hard_peak_memory_bytes") is not None
    ]

    return {
        "schema": "mtplx-t3-64-ab-lane-receipt-v1",
        "lane": lane_name,
        "rep_count": len(rep_payloads),
        "rep_files": [str(p) for p in rep_paths],
        "all_reps_passed": passed_all,
        "model_meta_by_rep": model_meta_by_rep,
        "peak_hard_watermark_bytes": max(hard_peaks) if hard_peaks else None,
        "peak_hard_watermark_gib": (max(hard_peaks) / (1024**3)) if hard_peaks else None,
        "cells": cell_table,
    }


def _concat_logs(spec: dict[str, Any], canonical_log_path: Path) -> None:
    rep_prefix = spec["rep_prefix"]
    chunks = []
    for rep in range(1, REP_COUNT + 1):
        log_path = OUTDIR / f"{rep_prefix}_rep{rep}.log"
        if not log_path.is_file():
            continue
        chunks.append(f"===== rep {rep}: {log_path.name} =====\n")
        chunks.append(log_path.read_text(errors="replace"))
        chunks.append("\n")
    canonical_log_path.write_text("".join(chunks))


def main() -> int:
    summary_lines: list[str] = []
    any_missing = False
    for lane_name, spec in LANES.items():
        result = aggregate_lane(lane_name, spec)
        if result is None:
            any_missing = True
            continue
        canonical_json = OUTDIR / f"{spec['canonical']}.json"
        canonical_log = OUTDIR / f"{spec['canonical']}.log"
        canonical_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        _concat_logs(spec, canonical_log)
        print(f"[aggregate] wrote {canonical_json} ({result['rep_count']} reps)", file=sys.stderr)

        for cell in spec["cells"]:
            table = result["cells"].get(cell)
            if table is None:
                summary_lines.append(f"{lane_name:16} {cell:4} NO DATA")
                continue
            mean = table["mean"]
            summary_lines.append(
                f"{lane_name:16} {cell:4} "
                f"tok/s={mean['decode_tok_s']:.3f} "
                f"(min={table['min_decode_tok_s']:.3f} max={table['max_decode_tok_s']:.3f}) "
                f"accept={mean['accepted_per_verify']:.4f} "
                f"hit={mean['decode_expert_cache_hit_rate']:.4f} "
                f"loads/tok={mean['loads_per_token']:.2f} "
                f"svc_ms/load={mean['svc_ms_per_load']:.3f} "
                f"parity_consistent={table['token_parity_consistent']} "
                f"parity={table['token_parity_all_reps']}"
            )

    print("\n".join(summary_lines))
    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
