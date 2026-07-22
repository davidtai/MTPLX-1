#!/usr/bin/env python3
"""CPU-only admission preflight + Arm B cache-limit derivation for the T3
64 GiB envelope island-vs-cache A/B mission (2026-07-22).

Three lane configs, ALL evaluated through the exact `mtplx/runtime.py`
`_load_impl` admission-gate replica this campaign has used since T4
(research/envelope_admission_sweep.py) / T3-pre
(research/t3-64x32k/preflight_kv_arms.py) -- same CPU-only prefix (build the
real ExpertStreamingConfig via the real preset/argparse machinery, open the
MTP artifact HEADER only, call config.memory_plan(), stop at fits_fixed).
No MLX buffer is ever allocated, no weight byte is ever read.

1. Arm A: preset hy3-oq2e-rq4-64 (islands 52), --max-live-kv-tokens 16384,
   --memory-limit 80262201344 (71 GiB preset + (16384-4096)*327680 exactly).
   This is also the T3 matrix-cell 64x16k bf16 receipt.

2. Arm B (cache-heavy, per 0157fae's motivation): SAME 74.75 GiB ceiling,
   SAME 16k KV, but ZERO islands (island_layer_count must be None +
   island_layers=() -- ExpertStreamingConfig rejects island_layer_count=0
   outright, minimum=1 -- so this is built WITHOUT --preset, passing every
   hy3-oq2e-rq4-64 flag explicitly and simply omitting --island-layer-count/
   --island-layers so they fall back to the hardcoded argparse defaults
   (None / ""), which is the only way to reach a genuine zero-island static
   plan through this CLI). The freed ~49.36 GiB (52 islands * 0.949 GiB/isl,
   matching the preset's own documented island cost) is priced into an
   explicit --expert-cache-limit via a two-pass derivation: pass 1 plans
   with an oversized placeholder cache-limit to read off the
   budget-capped persistent_cache_bytes the runtime would naturally settle
   on; pass 2 re-plans with that EXACT byte count as the real
   --expert-cache-limit (a static plan, matching Arm A's static-plan class,
   rather than leaving expert_cache_limit_bytes=None, which would flip
   derived_expert_cache_policy to a differently-behaved dynamic replan).

3. Patch lane: Arm A's preset unchanged, --max-live-kv-tokens 32768,
   --memory-limit 85630910464 (= 71 GiB + (32768-4096)*327680 exactly --
   the SAME override research/t3-64x32k already GPU-verified as ADMIT/79.68
   GiB implied), K3 only (--hy3-depths 3; AR auto-runs as d0 regardless of
   what --hy3-depths requests, so this still yields an AR reference row for
   the K3 token-hash comparison).

Also runs mtplx.expert_manifest.verify_expert_manifest on the hy3-oq2e-mlx
serving root (box law #4), PLUS an independent, non-truncated root-vs-
manifest *.safetensors diff (the validator's own extra=[...][:4] message is
not authoritative for counting) -- expects exactly 2 extras
(layer80-bf16.safetensors, layer80-residents-q.safetensors, both accounted
for by the checkpoint's resident/MTP-head layout per the T3-pre validation
window's receipt) and 0 missing. A different outcome is a STOP condition.

Usage:
    PYTHONPATH=<worktree> python3 research/t3-64-ab/preflight.py

Exit 0 iff every lane ADMITs and the manifest diff matches the expected
(2 extra, 0 missing) state; exit 1 otherwise (STOP, no GPU, per box law #4).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark_q2_mtp_depth_matrix as bqmdm  # noqa: E402

from mtplx.artifacts import load_config  # noqa: E402
from mtplx.expert_manifest import (  # noqa: E402
    load_expert_manifest,
    verify_expert_manifest,
)
from mtplx.expert_runtime import (  # noqa: E402
    ExpertStreamingConfig,
    parse_memory_bytes,
    proj_quant_plan_discount,
    proj_requant_plan_discount,
    resolve_island_placement,
)
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.hy3_mtp_patch import open_verified_hy3_mtp_artifacts  # noqa: E402
from mtplx.models.hy3_mlx import (  # noqa: E402
    estimate_hy3_router_kernel_incremental_bytes,
)

GIB = 1024**3
PRESET = "hy3-oq2e-rq4-64"
KV_BYTES_PER_TOKEN = 327_680  # HY3_EXPERT_OQ2E.kv_bytes_per_token
CONTROL_KV_TOKENS = 4096
PRESET_MEMORY_LIMIT_BYTES = 71 * GIB  # benchmarks/presets.toml [preset.hy3-oq2e-rq4-64]

ARM_A_KV_TOKENS = 16384
ARM_A_OVERRIDE_BYTES = PRESET_MEMORY_LIMIT_BYTES + (
    (ARM_A_KV_TOKENS - CONTROL_KV_TOKENS) * KV_BYTES_PER_TOKEN
)  # 80,262,201,344 -- 74.75 GiB, matches the mission brief exactly

PATCH_KV_TOKENS = 32768
PATCH_OVERRIDE_BYTES = PRESET_MEMORY_LIMIT_BYTES + (
    (PATCH_KV_TOKENS - CONTROL_KV_TOKENS) * KV_BYTES_PER_TOKEN
)  # 85,630,910,464 -- 79.75 GiB, the SAME override T3-pre already GPU-verified

# Oversized so pass-1 planning never gets cache-limit-capped -- only
# available_bytes / streamed_expert_bytes can bind. Arbitrary but far above
# anything this envelope could ever admit.
ARM_B_PLACEHOLDER_CACHE_LIMIT_BYTES = 500 * GIB

EXPECTED_MANIFEST_ROOT_NAME = "hy3-oq2e-mlx"
EXPECTED_EXTRA_SAFETENSORS = {
    "layer80-bf16.safetensors",
    "layer80-residents-q.safetensors",
}


class _Apis:
    config_factory = ExpertStreamingConfig
    parse_memory_bytes = staticmethod(parse_memory_bytes)


def _build_config_request(argv: list[str], *, preset: str | None) -> tuple[ExpertStreamingConfig, dict[str, Any]]:
    parser = bqmdm.build_parser()
    full_argv = (["--preset", preset] if preset else []) + argv
    if preset:
        try:
            bqmdm.preselect_preset(parser, full_argv, environ={})
        except bqmdm.PresetError as exc:
            raise SystemExit(f"preset error resolving {preset!r}: {exc}") from exc
    args = parser.parse_args(full_argv)
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


def _override_memory_limit(config: ExpertStreamingConfig, override_bytes: int) -> ExpertStreamingConfig:
    replace_kwargs: dict[str, Any] = {"memory_limit_bytes": override_bytes}
    if config.island_layer_count is not None and config.island_layers:
        # __post_init__ eagerly resolves island_layers from island_layer_count
        # for specs with a static island_pin_order (true here), leaving both
        # populated -- a valid post-resolution state, but replace() re-runs
        # __post_init__'s entry-time mutual-exclusivity guard against it.
        replace_kwargs["island_layer_count"] = None
    return dataclass_replace(config, **replace_kwargs)


def _plan(config: ExpertStreamingConfig, request: dict[str, Any]) -> dict[str, Any]:
    """Replicate mtplx/runtime.py's `_load_impl` CPU-only admission prefix."""

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
        "model_root": str(model_root),
        "manifest": request["manifest"],
        "memory_limit_bytes": resolved_config.memory_limit_bytes,
        "memory_limit_gib": resolved_config.memory_limit_bytes / GIB,
        "island_layer_count": len(resolved_config.island_layers),
        "streamed_mtp_resident_bytes": streamed_mtp_resident_bytes,
        "hy3_router_incremental_bytes": hy3_router_incremental_bytes,
        "additional_resident_bytes": additional_resident_bytes,
        "resident_discount_bytes": resident_discount_bytes,
        "fixed_bytes": plan.fixed_bytes,
        "fixed_bytes_gib": plan.fixed_bytes / GIB,
        "island_bytes": plan.island_bytes,
        "island_bytes_gib": plan.island_bytes / GIB,
        "persistent_cache_bytes": plan.persistent_cache_bytes,
        "persistent_cache_bytes_gib": plan.persistent_cache_bytes / GIB,
        "expert_cache_limit_bytes": plan.expert_cache_limit_bytes,
        "slots_per_layer": plan.slots_per_layer,
        "persistent_slots": plan.persistent_slots,
        "unallocated_bytes": plan.unallocated_bytes,
        "unallocated_gib": plan.unallocated_bytes / GIB,
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


def evaluate_arm_a() -> dict[str, Any]:
    config, request = _build_config_request(
        ["--max-live-kv-tokens", str(ARM_A_KV_TOKENS)], preset=PRESET
    )
    config = _override_memory_limit(config, ARM_A_OVERRIDE_BYTES)
    result = _plan(config, request)
    result["lane"] = "armA"
    result["preset"] = PRESET
    result["max_live_kv_tokens"] = ARM_A_KV_TOKENS
    result["cli"] = (
        f"--preset {PRESET} --max-live-kv-tokens {ARM_A_KV_TOKENS} "
        f"--memory-limit {ARM_A_OVERRIDE_BYTES} --hy3-depths 1,2,3"
    )
    return result, request


def _arm_b_argv(*, cache_limit_bytes: int, cache_policy: str) -> list[str]:
    # Mirrors --show-preset hy3-oq2e-rq4-64's expanded flags exactly, MINUS
    # --island-layer-count/--island-layers (there is no CLI value that
    # resolves to a genuine zero-island static plan other than omitting both
    # -- ExpertStreamingConfig.__post_init__ rejects island_layer_count=0
    # with "must be at least 1"), so they fall back to the hardcoded
    # argparse defaults (None / ""). --deferred-pin-release also omitted:
    # its argparse default is already True, matching the preset.
    return [
        "--model", "hy3-oq2e",
        "--contexts", "1024",
        "--output-tokens", "1024",
        "--memory-limit", str(ARM_A_OVERRIDE_BYTES),
        "--runtime-reserve", "7GiB",
        "--expert-cache-limit", str(cache_limit_bytes),
        "--max-live-kv-tokens", str(ARM_A_KV_TOKENS),
        "--cache-policy", cache_policy,
        "--cache-scope", "layer",
        "--slot-layout", "component-banks",
        "--hy3-router-kernel", "mpp-fp32-splitk-r1-fused-r2",
        "--verify-strategy", "batched",
        "--expert-integrity", "headers-only",
        "--proj-requant", "q4",
        "--split-route-release", "deferred",
    ]


def evaluate_arm_b(cache_policy: str) -> dict[str, Any]:
    # Pass 1: oversized placeholder cache-limit -- read off the naturally
    # budget-capped persistent_cache_bytes (available_bytes-bound, not
    # cache-limit-bound, since 500 GiB >> any headroom this envelope has).
    config1, request1 = _build_config_request(
        _arm_b_argv(
            cache_limit_bytes=ARM_B_PLACEHOLDER_CACHE_LIMIT_BYTES,
            cache_policy=cache_policy,
        ),
        preset=None,
    )
    pass1 = _plan(config1, request1)
    assert pass1["fits_fixed"], "arm B pass-1 (placeholder cache-limit) unexpectedly rejects"
    derived_cache_limit_bytes = pass1["persistent_cache_bytes"]

    # Pass 2: re-plan with the EXACT derived byte count as the real static
    # --expert-cache-limit (self-consistency check: persistent_cache_bytes
    # must reproduce pass 1 exactly, proving the cap is budget-derived, not
    # an artifact of the placeholder).
    config2, request2 = _build_config_request(
        _arm_b_argv(
            cache_limit_bytes=derived_cache_limit_bytes,
            cache_policy=cache_policy,
        ),
        preset=None,
    )
    pass2 = _plan(config2, request2)
    assert pass2["persistent_cache_bytes"] == derived_cache_limit_bytes, (
        "arm B pass-2 cache-limit self-consistency check failed: "
        f"{pass2['persistent_cache_bytes']} != {derived_cache_limit_bytes}"
    )

    pass2["lane"] = f"armB_{cache_policy}"
    pass2["cache_policy"] = cache_policy
    pass2["derived_cache_limit_bytes"] = derived_cache_limit_bytes
    pass2["derived_cache_limit_gib"] = derived_cache_limit_bytes / GIB
    pass2["max_live_kv_tokens"] = ARM_A_KV_TOKENS
    pass2["cli"] = " ".join(
        _arm_b_argv(cache_limit_bytes=derived_cache_limit_bytes, cache_policy=cache_policy)
        + ["--hy3-depths", "1,2,3"]
    )
    return pass2, request2


def evaluate_patch_lane() -> dict[str, Any]:
    config, request = _build_config_request(
        ["--max-live-kv-tokens", str(PATCH_KV_TOKENS)], preset=PRESET
    )
    config = _override_memory_limit(config, PATCH_OVERRIDE_BYTES)
    result = _plan(config, request)
    result["lane"] = "patch_k3exact"
    result["preset"] = PRESET
    result["max_live_kv_tokens"] = PATCH_KV_TOKENS
    result["cli"] = (
        f"--preset {PRESET} --max-live-kv-tokens {PATCH_KV_TOKENS} "
        f"--memory-limit {PATCH_OVERRIDE_BYTES} --hy3-depths 3"
    )
    return result, request


def _manifest_check(request: dict[str, Any]) -> dict[str, Any]:
    model_root = Path(request["model_root"])
    manifest_path = Path(request["manifest"])
    manifest = load_expert_manifest(manifest_path)
    validator_result: dict[str, Any] = {}
    validator_error: str | None = None
    try:
        validator_result = verify_expert_manifest(manifest, model_root)
    except Exception as exc:  # noqa: BLE001 -- record and continue to the independent diff
        validator_error = f"{type(exc).__name__}: {exc}"

    expected_safetensors = {shard.name for shard in manifest.shards if shard.kind == "safetensors"}
    actual_safetensors = {path.name for path in model_root.glob("*.safetensors")}
    extra = sorted(actual_safetensors - expected_safetensors)
    missing = sorted(expected_safetensors - actual_safetensors)

    matches_expected = set(extra) == EXPECTED_EXTRA_SAFETENSORS and not missing
    return {
        "model_root": str(model_root),
        "manifest_path": str(manifest_path),
        "validator_error": validator_error,
        "validator_result": validator_result,
        "expected_safetensors_count": len(expected_safetensors),
        "actual_safetensors_count": len(actual_safetensors),
        "extra_safetensors": extra,
        "missing_safetensors": missing,
        "matches_expected_2_extra_0_missing": matches_expected,
    }


def main() -> int:
    arm_a, arm_a_request = evaluate_arm_a()
    arm_b_frequency, arm_b_freq_request = evaluate_arm_b("frequency")
    arm_b_lru, _arm_b_lru_request = evaluate_arm_b("lru")
    patch, _patch_request = evaluate_patch_lane()

    manifest_check = _manifest_check(arm_a_request)

    lanes = {
        "armA": arm_a,
        "armB_frequency": arm_b_frequency,
        "armB_lru": arm_b_lru,
        "patch_k3exact": patch,
    }

    for name, lane in lanes.items():
        print(
            f"{name:16} status={lane['status']:6} "
            f"islands={lane['island_layer_count']:>3} "
            f"fixed_gib={lane['fixed_bytes_gib']:.4f} "
            f"cache_gib={lane['persistent_cache_bytes_gib']:.4f} "
            f"implied_total_gib={lane.get('implied_admission_total_gib', float('nan')):.4f} "
            f"limit_gib={lane['memory_limit_gib']:.4f}",
            file=sys.stderr,
        )
    print(
        f"manifest: extra={manifest_check['extra_safetensors']} "
        f"missing={manifest_check['missing_safetensors']} "
        f"matches_expected={manifest_check['matches_expected_2_extra_0_missing']} "
        f"validator_error={manifest_check['validator_error']}",
        file=sys.stderr,
    )

    all_admit = all(lane["status"] == "ADMIT" for lane in lanes.values())
    manifest_ok = manifest_check["matches_expected_2_extra_0_missing"]

    payload = {
        "schema": "mtplx-t3-64-ab-admission-preflight-v1",
        "arm_a_override_bytes": ARM_A_OVERRIDE_BYTES,
        "arm_a_override_gib": ARM_A_OVERRIDE_BYTES / GIB,
        "patch_override_bytes": PATCH_OVERRIDE_BYTES,
        "patch_override_gib": PATCH_OVERRIDE_BYTES / GIB,
        "lanes": lanes,
        "manifest_check": manifest_check,
        "all_admit": all_admit,
        "manifest_ok": manifest_ok,
        "go": all_admit and manifest_ok,
    }
    print(json.dumps(bqmdm._jsonable(payload), indent=2, sort_keys=True))
    print(f"\nGO: {payload['go']}", file=sys.stderr)
    return 0 if payload["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
