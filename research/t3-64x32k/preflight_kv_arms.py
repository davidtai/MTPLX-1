#!/usr/bin/env python3
"""CPU-only admission preflight for the T3-pre 64x32k paired kv-mode arms.

Extends research/envelope_admission_sweep.py's exact gate-replica machinery
(same functions, same CPU-only prefix of mtplx/runtime.py's ``_load_impl``,
same non-allocation guarantee: only file headers/JSON are ever read, no MLX
buffer is ever allocated, no weight byte is ever read) to ALSO vary
``kv_quant`` (None/bf16, "q8", "q4") at a FIXED override memory-limit -- the
mission's comparability rule: the override is the SAME ceiling for all three
arms, applied as a bookkeeping bound, not sized per-arm.

Cell: ``hy3-oq2e-rq4-64`` @ ``--max-live-kv-tokens 32768``. Per
``research/envelope_admission_sweep.py``'s T4 table, this preset's own
4096-token control admits at 71 GiB (memory-limit) with implied 70.93 GiB;
the KV-token override at 32768 is +8.75 GiB (28672 delta tokens *
327_680 B/token, the bf16 ``kv_bytes_per_token`` constant -- the override
formula never itself depends on kv_quant, per the envelope-accounting rule
that the override is a ceiling, not a per-mode allocation), giving
override_limit = 71 + 8.75 = 79.75 GiB, matching the T3-pre mission brief
exactly.

Run:
    PYTHONPATH=<worktree> python3 research/t3-64x32k/preflight_kv_arms.py

Exit 0 iff all three kv_quant arms ADMIT under the shared 79.75 GiB ceiling;
exit 1 otherwise (a REJECT here means STOP, no GPU, per the box laws).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research"))

import envelope_admission_sweep as eas  # noqa: E402

GIB = 1024**3
PRESET = "hy3-oq2e-rq4-64"
KV_TOKENS = 32768
# benchmarks/presets.toml [preset.hy3-oq2e-rq4-64] memory-limit = "71GiB"
PRESET_MEMORY_LIMIT_BYTES = 71 * GIB
KV_DELTA_TOKENS = KV_TOKENS - eas.CONTROL_KV_TOKENS
OVERRIDE_BYTES = KV_DELTA_TOKENS * eas.KV_BYTES_PER_TOKEN
OVERRIDE_LIMIT_BYTES = PRESET_MEMORY_LIMIT_BYTES + OVERRIDE_BYTES


def evaluate_admission_kv(
    preset: str,
    max_live_kv_tokens: int,
    *,
    memory_limit_override_bytes: int,
    kv_quant: str | None,
) -> dict:
    """Same gate as eas.evaluate_admission, plus a kv_quant override."""

    config, request = eas.build_config_and_request(preset, max_live_kv_tokens)
    replace_kwargs: dict = {
        "memory_limit_bytes": memory_limit_override_bytes,
        "kv_quant": kv_quant,
    }
    if config.island_layer_count is not None and config.island_layers:
        # See eas.evaluate_admission's identical comment: __post_init__
        # eagerly resolves island_layers from island_layer_count for specs
        # with a static island_pin_order, leaving both populated; replace()
        # re-runs __post_init__'s mutual-exclusivity guard, so drop the count.
        replace_kwargs["island_layer_count"] = None
    config = dataclass_replace(config, **replace_kwargs)

    from mtplx.artifacts import load_config
    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.expert_runtime import (
        proj_quant_plan_discount,
        proj_requant_plan_discount,
        resolve_island_placement,
    )
    from mtplx.expert_streaming_models import get_model_spec
    from mtplx.hy3_mtp_patch import open_verified_hy3_mtp_artifacts
    from mtplx.models.hy3_mlx import estimate_hy3_router_kernel_incremental_bytes

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

    result = {
        "preset": preset,
        "max_live_kv_tokens": max_live_kv_tokens,
        "kv_quant": kv_quant,
        "memory_limit_override_bytes": memory_limit_override_bytes,
        "memory_limit_override_gib": memory_limit_override_bytes / GIB,
        "island_layer_count": len(resolved_config.island_layers),
        "streamed_mtp_resident_bytes": streamed_mtp_resident_bytes,
        "hy3_router_incremental_bytes": hy3_router_incremental_bytes,
        "additional_resident_bytes": additional_resident_bytes,
        "resident_discount_bytes": resident_discount_bytes,
        "fixed_bytes": plan.fixed_bytes,
        "persistent_cache_bytes": plan.persistent_cache_bytes,
        "unallocated_bytes": plan.unallocated_bytes,
        "kv_bytes": plan.kv_bytes,
        "fits_fixed": plan.fits_fixed,
    }
    if plan.fits_fixed:
        result["status"] = "ADMIT"
        result["implied_admission_total_bytes"] = plan.fixed_bytes + plan.persistent_cache_bytes
        result["implied_admission_total_gib"] = (
            plan.fixed_bytes + plan.persistent_cache_bytes
        ) / GIB
    else:
        result["status"] = "REJECT"
        result["excess_bytes"] = -plan.unallocated_bytes
    return result


def main() -> int:
    print(f"preset={PRESET} kv_tokens={KV_TOKENS}", file=sys.stderr)
    print(
        f"preset_memory_limit_bytes={PRESET_MEMORY_LIMIT_BYTES} "
        f"({PRESET_MEMORY_LIMIT_BYTES / GIB} GiB)",
        file=sys.stderr,
    )
    print(
        f"kv_delta_tokens={KV_DELTA_TOKENS} override_bytes={OVERRIDE_BYTES} "
        f"({OVERRIDE_BYTES / GIB} GiB)",
        file=sys.stderr,
    )
    print(
        f"OVERRIDE_LIMIT_BYTES={OVERRIDE_LIMIT_BYTES} ({OVERRIDE_LIMIT_BYTES / GIB} GiB)",
        file=sys.stderr,
    )

    results = {}
    for kv_quant in (None, "q8", "q4"):
        r = evaluate_admission_kv(
            PRESET,
            KV_TOKENS,
            memory_limit_override_bytes=OVERRIDE_LIMIT_BYTES,
            kv_quant=kv_quant,
        )
        results[str(kv_quant)] = r
        print(
            f"kv_quant={kv_quant!r:6} status={r['status']:6} "
            f"kv_bytes={r['kv_bytes']:>14,} fixed_bytes={r['fixed_bytes']:>14,} "
            f"implied_total_gib={r.get('implied_admission_total_gib', float('nan')):.4f} "
            f"islands={r['island_layer_count']}",
            file=sys.stderr,
        )

    payload = {
        "schema": "mtplx-t3-64x32k-kv-arms-admission-preflight-v1",
        "preset": PRESET,
        "max_live_kv_tokens": KV_TOKENS,
        "preset_memory_limit_bytes": PRESET_MEMORY_LIMIT_BYTES,
        "kv_delta_tokens": KV_DELTA_TOKENS,
        "override_bytes": OVERRIDE_BYTES,
        "override_limit_bytes": OVERRIDE_LIMIT_BYTES,
        "override_limit_gib": OVERRIDE_LIMIT_BYTES / GIB,
        "arms": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    all_admit = all(r["status"] == "ADMIT" for r in results.values())
    print(f"\nALL ARMS ADMIT: {all_admit}", file=sys.stderr)
    return 0 if all_admit else 1


if __name__ == "__main__":
    raise SystemExit(main())
