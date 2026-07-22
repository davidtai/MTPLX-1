#!/usr/bin/env python3
"""CPU-only admission-gate sweep for the three `-cachelru` presets staged in
benchmarks/presets.toml (2026-07-22, David directive): `hy3-oq2e-rq4-
{64,48,32}-cachelru`. Companion to research/t3-cachelru/derive.py, which
derived each preset's `expert-cache-limit`; this script validates the
FINISHED presets (via the real --preset resolution machinery, exactly like
research/envelope_admission_sweep.py's `evaluate_admission()`, which this
module imports and reuses directly) across the full
KV_TOKEN_SWEEP = (4096, 16384, 32768, 65536), in two pricing modes:

  bf16 (default): the family's existing KV-delta override rule,
    override_bytes = (kv_tokens - 4096) * 327_680
    (HY3_EXPERT_OQ2E.kv_bytes_per_token).

  kv4: David's directive -- "check whether the admission path prices
    kv_quant via the spec's kv_bytes_per_token_for and if so sweep kv4
    admissions too". Confirmed yes: `ExpertStreamingConfig.memory_plan()`
    passes `kv_quant=self.kv_quant` straight into
    `plan_expert_memory(...)`, which prices
    `kv_bytes = context_tokens * spec.kv_bytes_per_token_for(kv_quant)`
    (mtplx/expert_streaming_models.py). For hy3-oq2e,
    kv_bytes_per_token_for("q4") == affine_quant_kept_bytes(327_680, "q4")
    == 327_680 * 9 // 32 == 92_160 B/token exactly (no rounding -- 327_680
    is a multiple of 32), matching David's stated figure. This script
    threads `--kv-quant q4` through the SAME real CLI/preset machinery (the
    benchmark script's own `--kv-quant` flag, confirmed plumbed at
    scripts/benchmark_q2_mtp_depth_matrix.py: `"kv_quant": args.kv_quant`)
    and re-derives the override at the q4 rate when the un-overridden base
    REJECTs.

No MLX allocation, no weight load, no GPU touch -- same CPU-only guarantee
as every other script in this admission-sweep family (stops at
`config.memory_plan()` / `plan.fits_fixed`, never reaches
`ExpertStreamingRuntime.open`).

Usage:
    PYTHONPATH=<worktree> python3 research/t3-cachelru/sweep.py \\
        [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
for _path in (REPO_ROOT, RESEARCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from envelope_admission_sweep import (  # noqa: E402
    CONTROL_KV_TOKENS,
    KV_BYTES_PER_TOKEN as KV_BYTES_PER_TOKEN_BF16,
    KV_TOKEN_SWEEP,
    build_config_and_request,
)
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfigurationError,
    proj_quant_plan_discount,
    proj_requant_plan_discount,
    resolve_island_placement,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.hy3_mtp_patch import open_verified_hy3_mtp_artifacts  # noqa: E402
from mtplx.artifacts import load_config  # noqa: E402
from mtplx.models.hy3_mlx import (  # noqa: E402
    estimate_hy3_router_kernel_incremental_bytes,
)

GIB = 1024**3
# Confirmed 2026-07-22: 327_680 * 9 // 32 == 92_160 exactly (David's figure).
KV_BYTES_PER_TOKEN_Q4 = 92_160

PRESETS = (
    "hy3-oq2e-rq4-64-cachelru",
    "hy3-oq2e-rq4-48-cachelru",
    "hy3-oq2e-rq4-32-cachelru",
)


def evaluate_admission_kv(
    preset: str,
    max_live_kv_tokens: int,
    *,
    kv_quant: str | None = None,
    memory_limit_override_bytes: int | None = None,
) -> dict[str, Any]:
    """Same CPU-only `_load_impl` admission prefix as
    envelope_admission_sweep.py's `evaluate_admission()`, extended with an
    optional `kv_quant` override (that function predates kv_quant sweeping
    and does not accept it)."""

    config, request = build_config_and_request(preset, max_live_kv_tokens)
    replace_kwargs: dict[str, Any] = {}
    if memory_limit_override_bytes is not None:
        replace_kwargs["memory_limit_bytes"] = memory_limit_override_bytes
    if kv_quant is not None:
        replace_kwargs["kv_quant"] = kv_quant
    if replace_kwargs:
        if config.island_layer_count is not None and config.island_layers:
            # See envelope_admission_sweep.py's evaluate_admission() for the
            # full explanation: __post_init__ eagerly resolves island_layers,
            # leaving both fields populated; replace() re-runs __post_init__'s
            # mutual-exclusivity guard against that state, so drop the count.
            # Not expected to trigger for these zero-island cachelru presets
            # (island_layer_count is always None already), kept for parity.
            replace_kwargs["island_layer_count"] = None
        config = dataclass_replace(config, **replace_kwargs)

    streaming_spec = get_model_spec(config.model_key)
    model_root = request["model_root"]
    model_config = load_config(model_root)

    mtp_precision = request.get("mtp_precision", "bf16")
    mtp_artifacts = request["mtp_artifacts"]
    with open_verified_hy3_mtp_artifacts(
        mtp_artifacts,
        precision=mtp_precision,
        expected_revision=streaming_spec.source_revision,
    ) as verified:
        streamed_mtp_resident_bytes = verified.payload_bytes

    hy3_router_incremental_bytes = 0
    if (
        str(model_config.get("model_type") or "") == "hy_v3"
        and config.hy3_router_kernel != "stock"
    ):
        hy3_router_incremental_bytes = estimate_hy3_router_kernel_incremental_bytes(
            model_config, config.hy3_router_kernel, include_mtp=True
        )
    additional_resident_bytes = streamed_mtp_resident_bytes + hy3_router_incremental_bytes

    resolved_config = config
    if resolved_config.island_layer_count is not None:
        resolved_config = resolve_island_placement(resolved_config, model_root)

    resident_discount_bytes = 0
    if resolved_config.proj_quant or resolved_config.proj_requant:
        manifest = load_expert_manifest(request["manifest"])
        resident_discount_bytes = proj_quant_plan_discount(
            manifest, resolved_config.proj_quant
        ) + proj_requant_plan_discount(manifest, resolved_config.proj_requant)

    plan = resolved_config.memory_plan(
        streaming_spec,
        additional_resident_bytes=additional_resident_bytes,
        resident_discount_bytes=resident_discount_bytes,
    )

    result: dict[str, Any] = {
        "preset": preset,
        "max_live_kv_tokens": max_live_kv_tokens,
        "kv_quant": kv_quant,
        "memory_limit_bytes": resolved_config.memory_limit_bytes,
        "memory_limit_override_applied": memory_limit_override_bytes is not None,
        "island_layer_count": len(resolved_config.island_layers),
        "streamed_mtp_resident_bytes": streamed_mtp_resident_bytes,
        "hy3_router_incremental_bytes": hy3_router_incremental_bytes,
        "additional_resident_bytes": additional_resident_bytes,
        "resident_discount_bytes": resident_discount_bytes,
        "fixed_bytes": plan.fixed_bytes,
        "persistent_cache_bytes": plan.persistent_cache_bytes,
        "unallocated_bytes": plan.unallocated_bytes,
        "fits_fixed": plan.fits_fixed,
    }
    if plan.fits_fixed:
        result["status"] = "ADMIT"
        result["excess_bytes"] = 0
        result["implied_admission_total_bytes"] = plan.fixed_bytes + plan.persistent_cache_bytes
        result["implied_admission_total_gib"] = (
            plan.fixed_bytes + plan.persistent_cache_bytes
        ) / GIB
    else:
        result["status"] = "REJECT"
        result["excess_bytes"] = -plan.unallocated_bytes
        result["error_type"] = ExpertStreamingConfigurationError.__name__
        result["error_message"] = (
            "fixed expert-streaming footprint exceeds limit by "
            f"{-plan.unallocated_bytes} bytes"
        )
    return result


def sweep_cell(preset: str, kv_tokens: int, kv_quant: str | None) -> dict[str, Any]:
    kv_bytes_per_token = KV_BYTES_PER_TOKEN_Q4 if kv_quant == "q4" else KV_BYTES_PER_TOKEN_BF16
    base = evaluate_admission_kv(preset, kv_tokens, kv_quant=kv_quant)
    cell: dict[str, Any] = {
        "preset": preset,
        "max_live_kv_tokens": kv_tokens,
        "kv_quant": kv_quant,
        "kv_bytes_per_token_used": kv_bytes_per_token,
        "base": base,
    }
    if base["status"] == "REJECT":
        kv_delta_tokens = kv_tokens - CONTROL_KV_TOKENS
        override_bytes = kv_delta_tokens * kv_bytes_per_token
        override_memory_limit_bytes = base["memory_limit_bytes"] + override_bytes
        overridden = evaluate_admission_kv(
            preset,
            kv_tokens,
            kv_quant=kv_quant,
            memory_limit_override_bytes=override_memory_limit_bytes,
        )
        cell["kv_delta_tokens"] = kv_delta_tokens
        cell["override_bytes"] = override_bytes
        cell["override_memory_limit_bytes"] = override_memory_limit_bytes
        cell["overridden"] = overridden
        cell["still_rejects_with_override"] = overridden["status"] == "REJECT"
    return cell


def sweep() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for preset in PRESETS:
        for kv in KV_TOKEN_SWEEP:
            cells.append(sweep_cell(preset, kv, None))
            cells.append(sweep_cell(preset, kv, "q4"))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cells = sweep()
    for cell in cells:
        final = cell.get("overridden", cell["base"])
        mode = cell["kv_quant"] or "bf16"
        print(
            f"{cell['preset']:26} kv={cell['max_live_kv_tokens']:6} mode={mode:5} "
            f"{final['status']:6} "
            f"fixed={final['fixed_bytes'] / GIB:8.4f} GiB "
            f"cache={final['persistent_cache_bytes'] / GIB:8.4f} GiB "
            f"limit={final['memory_limit_bytes'] / GIB:8.4f} GiB "
            f"margin={final['unallocated_bytes'] / GIB:+8.4f} GiB "
            f"override={'yes' if cell.get('overridden') else 'no '}",
            file=sys.stderr,
        )
    payload = {
        "schema": "mtplx-envelope-admission-sweep-v1",
        "generated": "2026-07-22",
        "note": (
            "Second/third dated file per the 2026-07-22 David directive "
            "(envelopes <=64 GiB never use islands; cache-heavy LRU): "
            "validates hy3-oq2e-rq4-{64,48,32}-cachelru (staged in "
            "benchmarks/presets.toml, cache-limit derived by "
            "research/t3-cachelru/derive.py) across the full KV sweep in "
            "both bf16 and kv4 (--kv-quant q4) pricing. Does NOT overwrite "
            "envelope-admission-sweep-2026-07-22.json or "
            "envelope-admission-sweep-2026-07-22-88rf.json."
        ),
        "gate_exercised": {
            "file": "mtplx/runtime.py",
            "function": "_load_impl",
            "description": (
                "config-time admission preflight; identical gate replica as "
                "envelope_admission_sweep.py, extended with an optional "
                "kv_quant override (dataclass_replace(config, "
                "kv_quant=...)), which mtplx/expert_runtime.py's "
                "ExpertStreamingConfig.memory_plan() passes straight into "
                "plan_expert_memory(kv_quant=...), which prices "
                "kv_bytes = context_tokens * spec.kv_bytes_per_token_for"
                "(kv_quant) (mtplx/expert_streaming_models.py)."
            ),
        },
        "kv_bytes_per_token_bf16": KV_BYTES_PER_TOKEN_BF16,
        "kv_bytes_per_token_q4": KV_BYTES_PER_TOKEN_Q4,
        "control_kv_tokens": CONTROL_KV_TOKENS,
        "presets": list(PRESETS),
        "kv_token_sweep": list(KV_TOKEN_SWEEP),
        "kv_quant_modes": [None, "q4"],
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
