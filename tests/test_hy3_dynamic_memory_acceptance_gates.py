from __future__ import annotations

from dataclasses import dataclass

import pytest

from mtplx.runtime_options import HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV
from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    BenchmarkGateError,
    CONTEXT_MATRIX_TOKENS,
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BLOCK_SIZE_TOKENS,
    HY3_Q4_KV_LAYERS,
    HY3_Q4_MAX_BLOCKS,
    canonical_sha256,
    normalize_arm_config,
    run_balanced_campaign,
    validate_allocator_probe,
    validate_campaign_observation,
)


GIB = 1024**3
OPERATING_TARGET_BYTES = 110 * GIB
HARD_CEILING_BYTES = 112 * GIB
STATIC_EXPERT_BYTES = 79 * GIB
EXPERT_SLAB_BYTES = 1 * GIB
RESIDENT_MODEL_BYTES = 12 * GIB
INFLIGHT_STAGING_BYTES = 2 * GIB
RUNTIME_WORKSPACE_BYTES = 5 * GIB
BASE_COMPRESSED_BYTES = GIB // 2
_ARTIFACT_FINGERPRINT = {
    "device": 1,
    "inode": 2,
    "size": 3,
    "mtime_ns": 4,
    "ctime_ns": 5,
}
_RESIDENT_SHARD_FINGERPRINTS = [
    {
        "name": "model-00001-of-00001.safetensors",
        "device": 6,
        "inode": 7,
        "size": 8,
        "mtime_ns": 9,
        "ctime_ns": 10,
    }
]
_ARTIFACT_STAT_SHA256 = canonical_sha256(
    {
        "sidecar": _ARTIFACT_FINGERPRINT,
        "resident_shards": _RESIDENT_SHARD_FINGERPRINTS,
    }
)


def _expert_route_binding(
    *,
    route_trace: object,
    expert_hashes: object,
) -> dict[str, object]:
    return {
        "schema": "mtplx-hy3-expert-route-binding-v1",
        "producer_verification_scope": "route-map-from-loaded-manifest",
        "offline_verification_scope": "structural-binding-only",
        "expert_manifest_sha256": "b" * 64,
        "route_trace_sha256": canonical_sha256(route_trace),
        "expert_hashes_sha256": canonical_sha256(expert_hashes),
    }


@dataclass(frozen=True)
class CampaignScenario:
    normal_extra_bytes: int = 0
    stress_peak_charged_bytes: int = 111 * GIB
    system_swap_delta_bytes: int = 0
    process_compressed_growth_bytes: int = 0
    short_dynamic_expert_bonus_bytes: int = EXPERT_SLAB_BYTES
    short_dynamic_hit_rate_delta: float = 0.10
    short_dynamic_ssd_ratio: float = 0.80
    short_dynamic_tps_ratio: float = 1.10
    long_dynamic_expert_delta_bytes: int = 0
    long_dynamic_tps_ratio: float = 0.98
    long_dynamic_hit_rate_delta: float = 0.0
    long_dynamic_ssd_ratio: float = 1.0


def _identity(arm: str = "static") -> dict[str, object]:
    arm_config = {
        "dynamic_memory": arm == "dynamic",
        "attention_runtime_env": dict(HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV),
        "context_window": 131_072,
        "kv_quantization": "q4",
        "expert_streaming_config": {
            "expert_slab_slots": 32,
            "allocator_headroom_bytes": GIB,
            "kv_bytes_per_token_override": 84_480,
            "memory_limit_bytes": OPERATING_TARGET_BYTES,
            "max_live_kv_tokens": 131_072,
            "runtime_reserve_bytes": 8 * GIB,
            "transient_slots": 32,
            "cache_scope": "global",
            "slot_layout": "component-banks",
            "dynamic_expert_slabs": arm == "dynamic",
        },
        "planned_persistent_slots": 9_696 if arm == "dynamic" else 8_673,
    }
    normalized = normalize_arm_config(arm_config)
    return {
        "model_key": "hy3-q4",
        "model_artifact_id": "pipenetwork/Hy3-4bit@160619d3",
        "model_artifact_sha256": "a" * 64,
        "expert_manifest_id": "hy3-q4/component-banks/manifest.json",
        "expert_manifest_sha256": "b" * 64,
        "artifact_pins_sha256": "d" * 64,
        "artifact_stat_sha256": _ARTIFACT_STAT_SHA256,
        "resident_payload_bytes": 3,
        "resident_payload_sha256": "e" * 64,
        "source_git_commit": "c" * 40,
        "arm_config": arm_config,
        "arm_config_sha256": canonical_sha256(arm_config),
        "normalized_config": normalized,
        "normalized_config_sha256": canonical_sha256(normalized),
        "kv_quantization": "q4",
        "kv_block_size_tokens": 16,
        "total_context_tokens": 131_072,
    }


def _slot_health() -> dict[str, int]:
    return {
        "active_routes": 0,
        "pins": 0,
        "loading": 0,
        "failed": 0,
        "integrity_errors": 0,
        "completion_fence_failures": 0,
        "global_device_synchronizations": 0,
    }


def _point(
    phase: str,
    timestamp_ns: int,
    *,
    expert_bytes: int,
    kv_blocks: int,
    normal_extra_bytes: int,
    system_swap_delta_bytes: int,
    kv_logical_tokens: int | None = None,
    process_compressed_bytes: int = BASE_COMPRESSED_BYTES,
) -> dict[str, object]:
    kv_bytes = kv_blocks * HY3_Q4_KV_BLOCK_BYTES
    active_bytes = (
        RESIDENT_MODEL_BYTES
        + INFLIGHT_STAGING_BYTES
        + RUNTIME_WORKSPACE_BYTES
        + expert_bytes
        + kv_bytes
        + normal_extra_bytes
    )
    slab_count = expert_bytes // EXPERT_SLAB_BYTES
    health = _slot_health()
    return {
        "phase": phase,
        "monotonic_ns": timestamp_ns,
        "allocator_active_bytes": active_bytes,
        "allocator_cache_bytes": 0,
        "allocator_peak_bytes": active_bytes,
        "expert_slab_physical_bytes": expert_bytes,
        "kv_physical_bytes": kv_bytes,
        "kv_allocated_blocks": kv_blocks,
        "slot_health": health,
        "slot_health_sha256": canonical_sha256(health),
        # Complete issue #46 resource evidence is sampled at every physical point.
        "operating_target_bytes": OPERATING_TARGET_BYTES,
        "hard_ceiling_bytes": HARD_CEILING_BYTES,
        "allocator_headroom_bytes": 1024**3,
        "classified_target_bytes": OPERATING_TARGET_BYTES - 1024**3,
        "classified_bytes": active_bytes - normal_extra_bytes,
        "charged_bytes": active_bytes,
        "charged_residual_bytes": OPERATING_TARGET_BYTES - active_bytes,
        "resident_model_bytes": RESIDENT_MODEL_BYTES,
        "kv_representation": "q4",
        "kv_logical_tokens": (
            kv_blocks * 16 if kv_logical_tokens is None else kv_logical_tokens
        ),
        "expert_logical_records": slab_count * 32,
        "expert_active_records": slab_count * 32,
        "expert_resident_records": slab_count * 24,
        "expert_logical_slabs": slab_count,
        "expert_active_slabs": slab_count,
        "expert_draining_slabs": 0,
        "expert_released_slabs": 0,
        "pinned_expert_bytes": 0,
        "inflight_expert_bytes": 0,
        "speculative_expert_bytes": 0,
        "runtime_workspace_bytes": RUNTIME_WORKSPACE_BYTES,
        "inflight_expert_staging_bytes": INFLIGHT_STAGING_BYTES,
        "requested_reclaim_bytes": 0,
        "reclaimed_bytes": 0,
        "regrown_bytes": 0,
        "evicted_expert_records": 0,
        "evicted_expert_slabs": 0,
        "resize_duration_ns": 0,
        "total_resize_duration_ns": 0,
        "max_resize_duration_ns": 0,
        "blocked_by_pin_bytes": 0,
        "admission_failures": 0,
        "resize_failures": 0,
        "allocator_cache_charged_bytes": normal_extra_bytes,
        "process_rss_bytes": active_bytes,
        "process_compressed_bytes": process_compressed_bytes,
        "system_swap_delta_bytes": system_swap_delta_bytes,
        "failed_closed": False,
        "failure_reason": None,
    }


def _performance_samples(
    *,
    tokens_per_second: float,
    expert_hit_rate: float,
    ssd_bytes_per_token: float,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, multiplier in enumerate((0.99, 1.0, 1.01)):
        route_trace = [
            {
                "phase": "ar_decode",
                "layer": index,
                "expert_ids": [3, 7],
            }
        ]
        expert_hashes = {
            f"{index}:3": "2" * 64,
            f"{index}:7": "3" * 64,
        }
        result.append(
            {
                "tokens_per_second": tokens_per_second * multiplier,
                "expert_hit_rate": expert_hit_rate,
                "ssd_bytes_per_token": ssd_bytes_per_token,
                "p50_token_latency_ms": 1000.0 / tokens_per_second,
                "p95_token_latency_ms": 1100.0 / tokens_per_second,
                "generated_token_ids": [
                    500 + index * 10 + offset for offset in range(8)
                ],
                "generated_token_sha256": canonical_sha256(
                    [500 + index * 10 + offset for offset in range(8)]
                ),
                "route_trace": route_trace,
                "route_trace_sha256": canonical_sha256(route_trace),
                "expert_hashes": expert_hashes,
                "expert_route_binding": _expert_route_binding(
                    route_trace=route_trace,
                    expert_hashes=expert_hashes,
                ),
            }
        )
    return result


def _hold_warmup_sample(
    *,
    tokens_per_second: float,
    expert_hit_rate: float,
    ssd_bytes_per_token: float,
) -> dict[str, object]:
    sample = _performance_samples(
        tokens_per_second=tokens_per_second,
        expert_hit_rate=expert_hit_rate,
        ssd_bytes_per_token=ssd_bytes_per_token,
    )[0]
    generated_tokens = list(range(400, 408))
    route_trace = [{"phase": "ar_decode", "layer": 70, "expert_ids": [3, 7]}]
    expert_hashes = {"70:3": "2" * 64, "70:7": "3" * 64}
    sample.update(
        generated_token_ids=generated_tokens,
        generated_token_sha256=canonical_sha256(generated_tokens),
        route_trace=route_trace,
        route_trace_sha256=canonical_sha256(route_trace),
        expert_hashes=expert_hashes,
        expert_route_binding=_expert_route_binding(
            route_trace=route_trace,
            expert_hashes=expert_hashes,
        ),
    )
    return sample


def _observation(
    arm: str,
    context_tokens: int,
    repetition: int,
    scenario: CampaignScenario,
) -> dict[str, object]:
    static_tps = 10.0
    if context_tokens < 131_072:
        expert_bytes = (
            STATIC_EXPERT_BYTES
            if arm == "static"
            else STATIC_EXPERT_BYTES + scenario.short_dynamic_expert_bonus_bytes
        )
        tps = (
            static_tps
            if arm == "static"
            else static_tps * scenario.short_dynamic_tps_ratio
        )
        hit_rate = (
            0.60 if arm == "static" else 0.60 + scenario.short_dynamic_hit_rate_delta
        )
        ssd_bytes_per_token = (
            1_000.0 if arm == "static" else 1_000.0 * scenario.short_dynamic_ssd_ratio
        )
    else:
        expert_bytes = (
            STATIC_EXPERT_BYTES
            if arm == "static"
            else STATIC_EXPERT_BYTES + scenario.long_dynamic_expert_delta_bytes
        )
        tps = (
            static_tps
            if arm == "static"
            else static_tps * scenario.long_dynamic_tps_ratio
        )
        hit_rate = (
            0.60 if arm == "static" else 0.60 + scenario.long_dynamic_hit_rate_delta
        )
        ssd_bytes_per_token = (
            1_000.0 if arm == "static" else 1_000.0 * scenario.long_dynamic_ssd_ratio
        )

    point_kwargs = {
        "normal_extra_bytes": scenario.normal_extra_bytes,
        "system_swap_delta_bytes": scenario.system_swap_delta_bytes,
    }
    if arm == "static":
        kv_growth_steps: list[dict[str, object]] = []
        timeline = [
            _point(
                "pre_growth",
                1,
                expert_bytes=expert_bytes,
                kv_blocks=HY3_Q4_MAX_BLOCKS,
                **point_kwargs,
            ),
            _point(
                "post_kv_growth",
                2,
                expert_bytes=expert_bytes,
                kv_blocks=HY3_Q4_MAX_BLOCKS,
                **point_kwargs,
            ),
            _point(
                "hold_warmup",
                2_500_000_000,
                expert_bytes=expert_bytes,
                kv_blocks=HY3_Q4_MAX_BLOCKS,
                **point_kwargs,
            ),
            *[
                _point(
                    "hold",
                    timestamp,
                    expert_bytes=expert_bytes,
                    kv_blocks=HY3_Q4_MAX_BLOCKS,
                    process_compressed_bytes=(
                        BASE_COMPRESSED_BYTES
                        + (
                            scenario.process_compressed_growth_bytes
                            if index == 2
                            else 0
                        )
                    ),
                    **point_kwargs,
                )
                for index, timestamp in enumerate(
                    (3_000_000_000, 3_500_000_000, 4_000_000_000)
                )
            ],
            _point(
                "post_reset",
                5_000_000_000,
                expert_bytes=expert_bytes,
                kv_blocks=0,
                process_compressed_bytes=(
                    BASE_COMPRESSED_BYTES + scenario.process_compressed_growth_bytes
                ),
                **point_kwargs,
            ),
        ]
        cache_start = {
            "kind": "static-128k-reserved-q4",
            "kv_physical_bytes": HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES,
            "kv_blocks": HY3_Q4_MAX_BLOCKS,
        }
    else:
        final_blocks = (
            context_tokens + HY3_Q4_KV_BLOCK_SIZE_TOKENS - 1
        ) // HY3_Q4_KV_BLOCK_SIZE_TOKENS
        pre_reclaim_expert_bytes = expert_bytes + EXPERT_SLAB_BYTES
        timeline = [
            _point(
                "pre_growth",
                1,
                expert_bytes=pre_reclaim_expert_bytes,
                kv_blocks=1,
                kv_logical_tokens=1,
                **point_kwargs,
            ),
            _point(
                "post_expert_reclaim",
                20,
                expert_bytes=expert_bytes,
                kv_blocks=1,
                kv_logical_tokens=1,
                **point_kwargs,
            ),
            _point(
                "post_kv_growth",
                70,
                expert_bytes=expert_bytes,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
                **point_kwargs,
            ),
            _point(
                "hold_warmup",
                3_000_000_000,
                expert_bytes=expert_bytes,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
                **point_kwargs,
            ),
            *[
                _point(
                    "hold",
                    timestamp,
                    expert_bytes=expert_bytes,
                    kv_blocks=final_blocks,
                    kv_logical_tokens=context_tokens,
                    process_compressed_bytes=(
                        BASE_COMPRESSED_BYTES
                        + (
                            scenario.process_compressed_growth_bytes
                            if index == 2
                            else 0
                        )
                    ),
                    **point_kwargs,
                )
                for index, timestamp in enumerate(
                    (4_000_000_000, 4_500_000_000, 5_000_000_000)
                )
            ],
            _point(
                "post_reset",
                6_000_000_000,
                expert_bytes=expert_bytes,
                kv_blocks=0,
                process_compressed_bytes=(
                    BASE_COMPRESSED_BYTES + scenario.process_compressed_growth_bytes
                ),
                **point_kwargs,
            ),
            _point(
                "post_regrow",
                7_000_000_000,
                expert_bytes=pre_reclaim_expert_bytes,
                kv_blocks=0,
                process_compressed_bytes=(
                    BASE_COMPRESSED_BYTES + scenario.process_compressed_growth_bytes
                ),
                **point_kwargs,
            ),
        ]

        def growth_ledger(point: dict[str, object]) -> dict[str, object]:
            return {
                key: value
                for key, value in point.items()
                if key not in {"phase", "monotonic_ns", "slot_health_sha256"}
            }

        first_before = growth_ledger(timeline[0])
        first_gap = growth_ledger(timeline[1])
        first_gap["captured_monotonic_ns"] = 20
        first_after = growth_ledger(
            _point(
                "growth_step_0_after",
                30,
                expert_bytes=expert_bytes,
                kv_blocks=final_blocks - 1,
                kv_logical_tokens=1,
                **point_kwargs,
            )
        )
        second_gap = dict(first_after)
        second_gap["captured_monotonic_ns"] = 50
        second_after = growth_ledger(
            _point(
                "growth_step_1_after",
                60,
                expert_bytes=expert_bytes,
                kv_blocks=final_blocks,
                kv_logical_tokens=1,
                **point_kwargs,
            )
        )
        kv_growth_steps = [
            {
                "sequence_index": 0,
                "requested_tokens": (final_blocks - 1) * HY3_Q4_KV_BLOCK_SIZE_TOKENS,
                "target_blocks": final_blocks - 1,
                "before_monotonic_ns": 10,
                "reclaim_monotonic_ns": 20,
                "after_monotonic_ns": 30,
                "before": dict(first_before),
                "reclaim_gap": dict(first_gap),
                "after": dict(first_after),
                "steady_delta_bytes": (final_blocks - 2) * HY3_Q4_KV_BLOCK_BYTES,
                "max_transient_delta_bytes": (final_blocks - 1)
                * (HY3_Q4_KV_BLOCK_BYTES // HY3_Q4_KV_LAYERS),
                "reclaimed_expert_bytes": EXPERT_SLAB_BYTES,
                "kv_growth_bytes": (final_blocks - 2) * HY3_Q4_KV_BLOCK_BYTES,
            },
            {
                "sequence_index": 1,
                "requested_tokens": context_tokens,
                "target_blocks": final_blocks,
                "before_monotonic_ns": 40,
                "reclaim_monotonic_ns": 50,
                "after_monotonic_ns": 60,
                "before": dict(first_after),
                "reclaim_gap": second_gap,
                "after": dict(second_after),
                "steady_delta_bytes": HY3_Q4_KV_BLOCK_BYTES,
                "max_transient_delta_bytes": final_blocks
                * (HY3_Q4_KV_BLOCK_BYTES // HY3_Q4_KV_LAYERS),
                "reclaimed_expert_bytes": 0,
                "kv_growth_bytes": HY3_Q4_KV_BLOCK_BYTES,
            },
        ]
        cache_start = {
            "kind": "dynamic-declared-q4",
            "kv_physical_bytes": HY3_Q4_KV_BLOCK_BYTES,
            "kv_blocks": 1,
        }

    generated_token_ids = [101, context_tokens, 202]
    route_trace = [
        {"phase": "ar_decode", "layer": 0, "expert_ids": [3, 7]},
        {"phase": "ar_decode", "layer": 1, "expert_ids": [2, 9]},
    ]
    detailed_samples = _performance_samples(
        tokens_per_second=tps,
        expert_hit_rate=hit_rate,
        ssd_bytes_per_token=ssd_bytes_per_token,
    )
    timeline[-1]["allocator_peak_bytes"] = max(
        int(timeline[-1]["allocator_peak_bytes"]),
        scenario.stress_peak_charged_bytes,
    )
    identity = _identity(arm)
    expert_hashes = {
        "0:3": "2" * 64,
        "0:7": "3" * 64,
        "1:2": "4" * 64,
        "1:9": "5" * 64,
    }
    return {
        "schema": "mtplx-hy3-dynamic-memory-observation-v1",
        "arm": arm,
        "context_tokens": context_tokens,
        "repetition": repetition,
        "cache_start_state": cache_start,
        "identity": identity,
        "prompt_sha256": "1" * 64,
        "generated_token_ids": generated_token_ids,
        "generated_token_sha256": canonical_sha256(generated_token_ids),
        "route_trace": route_trace,
        "route_trace_sha256": canonical_sha256(route_trace),
        "expert_hashes": expert_hashes,
        "expert_route_binding": _expert_route_binding(
            route_trace=route_trace,
            expert_hashes=expert_hashes,
        ),
        "kv_growth_steps": kv_growth_steps,
        "timeline": timeline,
        "metrics": {
            "generated_tokens": len(generated_token_ids),
            "elapsed_seconds": len(generated_token_ids) / tps,
            "tokens_per_second": tps,
            "peak_charged_bytes": max(
                int(point["charged_bytes"]) for point in timeline
            ),
            "stress_peak_charged_bytes": scenario.stress_peak_charged_bytes,
            "hold_performance_samples": [
                sample["tokens_per_second"] for sample in detailed_samples
            ],
            "hold_warmup_sample": _hold_warmup_sample(
                tokens_per_second=tps,
                expert_hit_rate=hit_rate,
                ssd_bytes_per_token=ssd_bytes_per_token,
            ),
            "performance_samples": detailed_samples,
        },
    }


def _probe_result() -> dict[str, object]:
    return {
        "schema": "mtplx-hy3-allocator-release-probe-v1",
        "manifest": {
            "identity": _identity(),
            "selected_slab_id": "slab-a",
            "selected_registered_physical_bytes": 128,
            "untouched_slab_id": "slab-b",
            "allocated_slab_count": 2,
            "slabs": [
                {"slab_id": "slab-a", "registered_physical_bytes": 128},
                {"slab_id": "slab-b", "registered_physical_bytes": 128},
            ],
        },
        "before_allocation": {
            "active_bytes": 700,
            "cache_bytes": 200,
            "peak_bytes": 900,
        },
        "before_release": {
            "active_bytes": 1_000,
            "cache_bytes": 200,
            "peak_bytes": 1_200,
        },
        "after_release": {
            "active_bytes": 872,
            "cache_bytes": 200,
            "peak_bytes": 1_200,
        },
        "untouched_slab_executable": True,
        "error": None,
    }


def _run_campaign(scenario: CampaignScenario) -> dict[str, object]:
    probe = validate_allocator_probe(_probe_result())
    return run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=lambda arm, context_tokens, repetition: _observation(
            arm,
            context_tokens,
            repetition,
            scenario,
        ),
        repetitions=2,
        contexts=CONTEXT_MATRIX_TOKENS,
        bootstrap_resamples=200,
        bootstrap_seed=46,
    )


def test_acceptance_passes_safe_beneficial_converged_campaign() -> None:
    result = _run_campaign(CampaignScenario())

    assert result["status"] == "passed"


def test_acceptance_rejects_dynamic_kv_growth_that_breaks_physical_chain() -> None:
    context_tokens = CONTEXT_MATRIX_TOKENS[0]
    row = _observation("dynamic", context_tokens, 0, CampaignScenario())
    growth_steps = row["kv_growth_steps"]
    assert isinstance(growth_steps, list)
    final_blocks = (
        context_tokens + HY3_Q4_KV_BLOCK_SIZE_TOKENS - 1
    ) // HY3_Q4_KV_BLOCK_SIZE_TOKENS
    assert [step["target_blocks"] for step in growth_steps] == [
        final_blocks - 1,
        final_blocks,
    ]

    second_step = growth_steps[1]
    assert isinstance(second_step, dict)
    second_before = second_step["before"]
    assert isinstance(second_before, dict)
    released_kv_bytes = int(second_before["kv_physical_bytes"]) - HY3_Q4_KV_BLOCK_BYTES
    second_before["kv_allocated_blocks"] = 1
    second_before["kv_physical_bytes"] = HY3_Q4_KV_BLOCK_BYTES
    second_before["kv_logical_tokens"] = 1
    for field in (
        "allocator_active_bytes",
        "allocator_peak_bytes",
        "classified_bytes",
        "charged_bytes",
        "process_rss_bytes",
    ):
        second_before[field] -= released_kv_bytes
    second_before["charged_residual_bytes"] += released_kv_bytes

    with pytest.raises(BenchmarkGateError, match="one physical chain"):
        validate_campaign_observation(row)


def test_acceptance_allows_a_partial_static_final_slab_at_128k() -> None:
    probe = validate_allocator_probe(_probe_result())
    partial_records = 6
    record_bytes = EXPERT_SLAB_BYTES // 32
    partial_bytes = partial_records * record_bytes

    def execute(arm: str, context_tokens: int, repetition: int) -> dict[str, object]:
        row = _observation(arm, context_tokens, repetition, CampaignScenario())
        if arm != "static" or context_tokens != 131_072:
            return row
        timeline = row["timeline"]
        assert isinstance(timeline, list)
        for point in timeline:
            assert isinstance(point, dict)
            point["expert_slab_physical_bytes"] += partial_bytes
            point["allocator_active_bytes"] += partial_bytes
            point["allocator_peak_bytes"] += partial_bytes
            point["classified_bytes"] += partial_bytes
            point["charged_bytes"] += partial_bytes
            point["charged_residual_bytes"] -= partial_bytes
            point["process_rss_bytes"] += partial_bytes
            point["expert_logical_records"] += partial_records
            point["expert_active_records"] += partial_records
            point["expert_resident_records"] += partial_records
            point["expert_logical_slabs"] += 1
            point["expert_active_slabs"] += 1
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        metrics["peak_charged_bytes"] += partial_bytes
        metrics["stress_peak_charged_bytes"] += partial_bytes
        return row

    result = run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=2,
        contexts=CONTEXT_MATRIX_TOKENS,
        bootstrap_resamples=200,
        bootstrap_seed=46,
    )

    assert result["status"] == "passed"


def test_acceptance_rejects_normal_peak_above_110_gib() -> None:
    result = _run_campaign(CampaignScenario(normal_extra_bytes=2 * GIB))

    assert result["status"] == "rejected"


def test_acceptance_rejects_paired_fixed_pool_confounding() -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int) -> dict[str, object]:
        row = _observation(arm, context_tokens, repetition, CampaignScenario())
        if arm != "dynamic":
            return row
        timeline = row["timeline"]
        assert isinstance(timeline, list)
        for point in timeline:
            assert isinstance(point, dict)
            point["resident_model_bytes"] -= GIB
            point["allocator_active_bytes"] -= GIB
            point["allocator_peak_bytes"] -= GIB
            point["classified_bytes"] -= GIB
            point["charged_bytes"] -= GIB
            point["charged_residual_bytes"] += GIB
            point["process_rss_bytes"] -= GIB
        growth_steps = row["kv_growth_steps"]
        assert isinstance(growth_steps, list)
        for step in growth_steps:
            assert isinstance(step, dict)
            for checkpoint in ("before", "reclaim_gap", "after"):
                ledger = step[checkpoint]
                assert isinstance(ledger, dict)
                ledger["resident_model_bytes"] -= GIB
                ledger["allocator_active_bytes"] -= GIB
                ledger["allocator_peak_bytes"] -= GIB
                ledger["classified_bytes"] -= GIB
                ledger["charged_bytes"] -= GIB
                ledger["charged_residual_bytes"] += GIB
                ledger["process_rss_bytes"] -= GIB
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        metrics["peak_charged_bytes"] = max(
            int(point["charged_bytes"]) for point in timeline
        )
        metrics["stress_peak_charged_bytes"] = max(
            max(
                int(point["allocator_peak_bytes"])
                + int(point["allocator_cache_bytes"]),
                int(point["charged_bytes"]),
            )
            for point in timeline
        )
        return row

    result = run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=2,
        contexts=CONTEXT_MATRIX_TOKENS,
        bootstrap_resamples=200,
        bootstrap_seed=46,
    )

    assert result["status"] == "rejected"
    assert any(
        "paired fixed physical pools differ" in reason
        for reason in result["acceptance"]["rejection_reasons"]
    )


def test_acceptance_rejects_stress_peak_at_112_gib() -> None:
    result = _run_campaign(
        CampaignScenario(stress_peak_charged_bytes=HARD_CEILING_BYTES)
    )

    assert result["status"] == "rejected"


def test_acceptance_rejects_positive_swap_growth() -> None:
    result = _run_campaign(CampaignScenario(system_swap_delta_bytes=1))

    assert result["status"] == "rejected"


def test_acceptance_rejects_compressor_runaway() -> None:
    result = _run_campaign(CampaignScenario(process_compressed_growth_bytes=GIB))

    assert result["status"] == "rejected"


def test_acceptance_rejects_no_short_context_expert_capacity_gain() -> None:
    result = _run_campaign(CampaignScenario(short_dynamic_expert_bonus_bytes=0))

    assert result["status"] == "rejected"


def test_acceptance_rejects_no_measurable_short_context_improvement() -> None:
    result = _run_campaign(
        CampaignScenario(
            short_dynamic_hit_rate_delta=0.0,
            short_dynamic_ssd_ratio=1.0,
            short_dynamic_tps_ratio=1.0,
        )
    )

    assert result["status"] == "rejected"


def test_acceptance_rejects_128k_expert_capacity_divergence() -> None:
    result = _run_campaign(
        CampaignScenario(long_dynamic_expert_delta_bytes=-EXPERT_SLAB_BYTES)
    )

    assert result["status"] == "rejected"


def test_acceptance_rejects_unexplained_128k_tps_regression() -> None:
    result = _run_campaign(CampaignScenario(long_dynamic_tps_ratio=0.80))

    assert result["status"] == "rejected"


@pytest.mark.parametrize(
    "scenario",
    (
        CampaignScenario(long_dynamic_hit_rate_delta=-0.10),
        CampaignScenario(long_dynamic_ssd_ratio=1.10),
    ),
)
def test_acceptance_rejects_128k_cache_performance_divergence(
    scenario: CampaignScenario,
) -> None:
    result = _run_campaign(scenario)

    assert result["status"] == "rejected"
