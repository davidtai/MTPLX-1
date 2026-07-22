#!/usr/bin/env python3
"""CPU-only cache-limit derivation for the three `-cachelru` presets
(2026-07-22, David directive): `hy3-oq2e-rq4-64-cachelru`,
`hy3-oq2e-rq4-48-cachelru`, `hy3-oq2e-rq4-32-cachelru`.

Context: David's box law -- envelopes <=64 GiB never use islands; cache-
heavy LRU is the config. research/t3-64-ab already A/B'd this at the 64 GiB
envelope (hy3-oq2e-rq4-64-cachehvy, cache-policy=frequency, won vs the
island preset at every K) and separately measured an LRU arm at the SAME
derived cache-limit (evals/tier2/t3_64x16k_armB_lru_rep1.log) -- this script
generalizes that TWO-PASS derivation (mtplx/expert_runtime.py's
`ExpertStreamingConfig.memory_plan()` / `plan_expert_memory()`) down to the
48 and 32 GiB family envelopes, and switches the shipped cache-policy to
`lru` per this directive (the A/B's own written preset,
hy3-oq2e-rq4-64-cachehvy, defaults to `frequency` -- these three new presets
are deliberately separate, `lru`-only).

Two-pass method (identical to research/t3-64-ab/preflight.py's
`evaluate_arm_b`): pass 1 plans with an oversized placeholder
--expert-cache-limit (500 GiB, far above anything any of these envelopes
could ever admit) so `persistent_cache_bytes` settles at whatever the
runtime naturally derives -- budget-bound by `total_limit_bytes` minus fixed
costs, never limit-bound. Pass 2 re-plans with that exact byte count as the
real cache-limit (self-consistency check).

NEW this time (David's directive, not present in the T3 64x16k A/B): the
natural pass-1 margin is not accepted as-is -- if it falls outside the
family's healthy 0.3-1.0 GiB fits_fixed-margin band, the cache-limit is
trimmed down from the natural derived value until the margin lands inside
that band (there is no way to derive a LARGER cache than the natural
pass-1 value: it is already budget/spec-bound with an uncapped placeholder,
so raising the declared limit past it is a no-op).

Every envelope uses: island-layer-count OMITTED (the only way to reach a
genuine zero-island static plan -- ExpertStreamingConfig rejects
island_layer_count=0 outright, minimum 1; ARM_B_PLACEHOLDER precedent, see
research/t3-64-ab/preflight.py docstring), cache-policy=lru, cache-scope
layer, slot-layout component-banks, proj-requant q4, hy3-router-kernel
mpp-fp32-splitk-r1-fused-r2, verify-strategy batched, expert-integrity
headers-only, split-route-release deferred -- i.e. every non-island,
non-cache field matches the family's existing presets exactly
(hy3-oq2e-rq4-64/48/32), memory-limit kept at each envelope's own family
value (71/55/39 GiB) so the "64/48/32" label still means what the sibling
island presets mean.

No MLX allocation, no weight load, no GPU touch -- same CPU-only guarantee
as research/envelope_admission_sweep.py and research/t3-64-ab/preflight.py
(this script imports scripts/benchmark_q2_mtp_depth_matrix.py's own
argparse/config-construction machinery and stops at `config.memory_plan()`).

Usage:
    PYTHONPATH=<worktree> python3 research/t3-cachelru/derive.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark_q2_mtp_depth_matrix as bqmdm  # noqa: E402

from mtplx.artifacts import load_config  # noqa: E402
from mtplx.expert_manifest import load_expert_manifest  # noqa: E402
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    parse_memory_bytes,
    proj_quant_plan_discount,
    proj_requant_plan_discount,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.hy3_mtp_patch import open_verified_hy3_mtp_artifacts  # noqa: E402
from mtplx.models.hy3_mlx import (  # noqa: E402
    estimate_hy3_router_kernel_incremental_bytes,
)

GIB = 1024**3
CONTROL_KV_TOKENS = 4096

# (envelope_label, family memory-limit GiB -- matches the sibling island
# preset's own `memory-limit` field exactly: hy3-oq2e-rq4-64 -> 71GiB,
# hy3-oq2e-rq4-48 -> 55GiB, hy3-oq2e-rq4-32 -> 39GiB).
ENVELOPES: tuple[tuple[str, int], ...] = (
    ("64", 71),
    ("48", 55),
    ("32", 39),
)

PLACEHOLDER_CACHE_LIMIT_BYTES = 500 * GIB
MARGIN_BAND_LOW_GIB = 0.3
MARGIN_BAND_HIGH_GIB = 1.0
# Target the low end of the band when trimming down from an out-of-band
# natural margin -- maximizes cache size (fills the envelope) while staying
# safely inside the healthy band, per the directive's "fixed + cache fills
# the envelope" framing.
TARGET_MARGIN_GIB = 0.35


class _Apis:
    config_factory = ExpertStreamingConfig
    parse_memory_bytes = staticmethod(parse_memory_bytes)


def _argv(*, memory_limit_bytes: int, cache_limit_bytes: int) -> list[str]:
    # Mirrors research/t3-64-ab/preflight.py's `_arm_b_argv`: every family
    # field explicit, island-layer-count/island-layers OMITTED (the only way
    # to reach a genuine zero-island static plan), cache-policy=lru per this
    # directive (not frequency, unlike hy3-oq2e-rq4-64-cachehvy).
    return [
        "--model", "hy3-oq2e",
        "--contexts", "1024",
        "--output-tokens", "1024",
        "--memory-limit", str(memory_limit_bytes),
        "--runtime-reserve", "7GiB",
        "--expert-cache-limit", str(cache_limit_bytes),
        "--max-live-kv-tokens", str(CONTROL_KV_TOKENS),
        "--cache-policy", "lru",
        "--cache-scope", "layer",
        "--slot-layout", "component-banks",
        "--hy3-router-kernel", "mpp-fp32-splitk-r1-fused-r2",
        "--verify-strategy", "batched",
        "--expert-integrity", "headers-only",
        "--proj-requant", "q4",
        "--split-route-release", "deferred",
    ]


def _build_config_request(argv: list[str]) -> tuple[ExpertStreamingConfig, dict[str, Any]]:
    parser = bqmdm.build_parser()
    args = parser.parse_args(argv)
    requests = bqmdm._requests_from_args(args)
    if len(requests) != 1:
        raise SystemExit(f"expected exactly one model request, got {len(requests)}")
    request = bqmdm._normalized_request(requests[0], mtp_disabled_baseline=False)
    options = {
        **bqmdm.DEFAULT_RUNTIME_OPTIONS,
        **bqmdm._runtime_options_from_args(args),
        "trace_routes": bool(args.trace_routes),
    }
    config = bqmdm._runtime_config(_Apis(), request["model_key"], options)
    return config, request


def _plan(config: ExpertStreamingConfig, request: dict[str, Any]) -> dict[str, Any]:
    """Replicate mtplx/runtime.py's `_load_impl` CPU-only admission prefix
    (same prefix as research/envelope_admission_sweep.py /
    research/t3-64-ab/preflight.py). config.island_layer_count is always
    None here (islands omitted), so resolve_island_placement is never
    reached -- correct, since there is nothing to resolve."""

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

    assert config.island_layer_count is None, "cachelru presets must have zero islands"
    resolved_config = config

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
        "memory_limit_bytes": resolved_config.memory_limit_bytes,
        "memory_limit_gib": resolved_config.memory_limit_bytes / GIB,
        "island_layer_count": len(resolved_config.island_layers),
        "expert_cache_limit_bytes": resolved_config.expert_cache_limit_bytes,
        "fixed_bytes": plan.fixed_bytes,
        "fixed_bytes_gib": plan.fixed_bytes / GIB,
        "persistent_cache_bytes": plan.persistent_cache_bytes,
        "persistent_cache_bytes_gib": plan.persistent_cache_bytes / GIB,
        "unallocated_bytes": plan.unallocated_bytes,
        "unallocated_gib": plan.unallocated_bytes / GIB,
        "fits_fixed": plan.fits_fixed,
        "implied_admission_total_bytes": plan.fixed_bytes + plan.persistent_cache_bytes,
        "implied_admission_total_gib": (plan.fixed_bytes + plan.persistent_cache_bytes) / GIB,
        "status": "ADMIT" if plan.fits_fixed else "REJECT",
    }
    return result


def derive_envelope(label: str, memory_limit_gib: int) -> dict[str, Any]:
    memory_limit_bytes = memory_limit_gib * GIB

    # Pass 1: oversized placeholder -- read off the naturally budget-bound
    # persistent_cache_bytes (available-bytes-bound, not limit-bound).
    config1, request1 = _build_config_request(
        _argv(memory_limit_bytes=memory_limit_bytes, cache_limit_bytes=PLACEHOLDER_CACHE_LIMIT_BYTES)
    )
    pass1 = _plan(config1, request1)
    assert pass1["fits_fixed"], f"{label}: pass-1 placeholder unexpectedly REJECTs"
    natural_cache_bytes = pass1["persistent_cache_bytes"]
    natural_margin_gib = pass1["unallocated_gib"]

    in_band = MARGIN_BAND_LOW_GIB <= natural_margin_gib <= MARGIN_BAND_HIGH_GIB
    if in_band:
        final_cache_bytes = natural_cache_bytes
        adjustment = "none (natural margin already inside 0.3-1.0 GiB band)"
    elif natural_margin_gib < MARGIN_BAND_LOW_GIB:
        # Should not happen with an uncapped pass-1 placeholder (cache is
        # budget-bound, so unallocated_bytes reflects the plan's own
        # internal slack, not something a smaller cache-limit can widen --
        # a SMALLER declared cache would just leave MORE unallocated, since
        # persistent_cache_bytes only ever shrinks with a tighter cap).
        # Guard documented for completeness; not expected to trigger.
        deficit_bytes = int((MARGIN_BAND_LOW_GIB - natural_margin_gib) * GIB)
        final_cache_bytes = max(0, natural_cache_bytes - deficit_bytes)
        adjustment = f"trimmed down by {deficit_bytes} B to reach {MARGIN_BAND_LOW_GIB} GiB margin"
    else:  # natural_margin_gib > MARGIN_BAND_HIGH_GIB
        target_bytes = int(TARGET_MARGIN_GIB * GIB)
        trim_bytes = int((natural_margin_gib * GIB)) - target_bytes
        final_cache_bytes = natural_cache_bytes + trim_bytes
        adjustment = (
            f"natural margin {natural_margin_gib:.4f} GiB > {MARGIN_BAND_HIGH_GIB} GiB cap -- "
            f"raised declared cache-limit by {trim_bytes} B to target {TARGET_MARGIN_GIB} GiB margin"
        )

    # Pass 2: self-consistency check with the final byte count.
    config2, request2 = _build_config_request(
        _argv(memory_limit_bytes=memory_limit_bytes, cache_limit_bytes=final_cache_bytes)
    )
    pass2 = _plan(config2, request2)

    return {
        "label": label,
        "memory_limit_gib": memory_limit_gib,
        "memory_limit_bytes": memory_limit_bytes,
        "pass1_natural_cache_bytes": natural_cache_bytes,
        "pass1_natural_margin_gib": natural_margin_gib,
        "in_band_without_adjustment": in_band,
        "adjustment": adjustment,
        "final_cache_limit_bytes": final_cache_bytes,
        "final_cache_limit_gib": final_cache_bytes / GIB,
        "pass2": pass2,
    }


def main() -> int:
    results = [derive_envelope(label, gib) for label, gib in ENVELOPES]
    for r in results:
        p2 = r["pass2"]
        print(
            f"{r['label']:4} memory_limit={r['memory_limit_gib']:5.1f} GiB "
            f"pass1_natural_cache={r['pass1_natural_cache_bytes']/GIB:8.4f} GiB "
            f"pass1_natural_margin={r['pass1_natural_margin_gib']:+.4f} GiB "
            f"-> final_cache_limit={r['final_cache_limit_gib']:8.4f} GiB "
            f"({r['final_cache_limit_bytes']} B) "
            f"pass2_status={p2['status']} pass2_margin={p2['unallocated_gib']:+.4f} GiB "
            f"pass2_cache_used={p2['persistent_cache_bytes_gib']:.4f} GiB",
            file=sys.stderr,
        )
        print(f"     adjustment: {r['adjustment']}", file=sys.stderr)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(r["pass2"]["status"] == "ADMIT" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
