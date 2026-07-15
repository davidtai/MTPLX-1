from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

import mtplx.benchmarks.runners.hy3_dynamic_memory as runner_module
from mtplx.benchmarks.hy3_dynamic_memory_observation import (
    ArmRequest,
    _build_identity,
)
from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    CONTEXT_MATRIX_TOKENS,
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BYTES_PER_TOKEN,
    HY3_Q4_MAX_BLOCKS,
    AllocatorSample,
    BenchmarkGateError,
    CacheStartState,
    QwenIsolationHooks,
    balanced_campaign_schedule,
    canonical_sha256,
    run_direct_cache_probe,
    run_balanced_campaign,
    run_exclusive_hardware_window,
    validate_allocator_probe,
    validate_campaign_observation,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expert_route_binding(
    *,
    manifest_sha256: str,
    route_trace: object,
    expert_hashes: object,
) -> dict[str, object]:
    return {
        "schema": "mtplx-hy3-expert-route-binding-v1",
        "producer_verification_scope": "route-map-from-loaded-manifest",
        "offline_verification_scope": "structural-binding-only",
        "expert_manifest_sha256": manifest_sha256,
        "route_trace_sha256": _sha(route_trace),
        "expert_hashes_sha256": _sha(expert_hashes),
    }


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
_ARTIFACT_STAT_SHA256 = _sha(
    {
        "sidecar": _ARTIFACT_FINGERPRINT,
        "resident_shards": _RESIDENT_SHARD_FINGERPRINTS,
    }
)


def _identity(arm: str = "dynamic") -> dict[str, object]:
    arm_config = {
        "dynamic_memory": arm == "dynamic",
        "attention_runtime_env": dict(
            runner_module.HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV
        ),
        "context_window": 131_072,
        "kv_quantization": "q4",
        "max_active_sequences": 1,
        "expert_streaming_config": {
            "allocator_headroom_bytes": 1024**3,
            "kv_bytes_per_token_override": 84_480,
            "memory_limit_bytes": 100 * 1024**3,
            "max_live_kv_tokens": 131_072,
            "runtime_reserve_bytes": 8 * 1024**3,
            "transient_slots": 32,
            "cache_policy": "lru",
            "cache_scope": "global",
            "slot_layout": "direct-slots",
            "dynamic_expert_cache": arm == "dynamic",
            "resource_telemetry": True,
        },
        "planned_persistent_slots": 8_673,
    }
    normalized_config = runner_module.normalize_arm_config(arm_config)
    return {
        "model_key": "hy3-q4",
        "model_artifact_id": "pipenetwork/Hy3-4bit@160619d3",
        "model_artifact_sha256": "a" * 64,
        "expert_manifest_id": "hy3-q4/direct-slots/manifest.json",
        "expert_manifest_sha256": "b" * 64,
        "artifact_pins_sha256": "d" * 64,
        "artifact_stat_sha256": _ARTIFACT_STAT_SHA256,
        "resident_payload_bytes": 3,
        "resident_payload_sha256": "e" * 64,
        "source_git_commit": "c" * 40,
        "arm_config": arm_config,
        "arm_config_sha256": _sha(arm_config),
        "normalized_config": normalized_config,
        "normalized_config_sha256": _sha(normalized_config),
        "kv_quantization": "q4",
        "kv_block_size_tokens": 16,
        "total_context_tokens": 131_072,
    }


def _quality_identity() -> dict[str, object]:
    identity = _identity()
    return {
        field: identity[field]
        for field in (
            "model_artifact_id",
            "model_artifact_sha256",
            "expert_manifest_sha256",
            "artifact_pins_sha256",
            "artifact_stat_sha256",
            "resident_payload_bytes",
            "resident_payload_sha256",
            "source_git_commit",
        )
    }


def _production_lane_identity(arm: str) -> dict[str, object]:
    identity = _identity(arm)
    identity["arm_config"] = {
        "dynamic_memory": arm == "dynamic",
        "attention_runtime_env": dict(
            runner_module.HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV
        ),
        "expert_streaming_config": {
            "model_key": "hy3-q4",
            "memory_limit_bytes": 100 * 1024**3,
            "max_live_kv_tokens": 131_072,
            "kv_bytes_per_token_override": 84_480,
            "runtime_reserve_bytes": 8 * 1024**3,
            "allocator_headroom_bytes": 1024**3,
            "transient_slots": 32,
            "cache_policy": "lru",
            "cache_scope": "global",
            "slot_layout": "direct-slots",
            "dynamic_expert_cache": arm == "dynamic",
            "resource_telemetry": True,
        },
        "planned_persistent_slots": 8_673,
    }
    return identity


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


def _performance_samples(tok_s: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, (multiplier, hit_rate, ssd_bytes, p50_ms, p95_ms) in enumerate(
        (
            (0.99, 0.70, 1_024.0, 80.0, 100.0),
            (1.00, 0.75, 768.0, 79.0, 98.0),
            (1.01, 0.80, 512.0, 78.0, 96.0),
        )
    ):
        generated_tokens = [500 + index * 10 + offset for offset in range(8)]
        route_trace = [
            {
                "phase": "ar_decode",
                "layer": index + 1,
                "expert_ids": [3, 7, 11],
            }
        ]
        expert_hashes = {
            f"{index + 1}:3": "2" * 64,
            f"{index + 1}:7": "3" * 64,
            f"{index + 1}:11": "4" * 64,
        }
        result.append(
            {
                "tokens_per_second": tok_s * multiplier,
                "expert_hit_rate": hit_rate,
                "ssd_bytes_per_token": ssd_bytes,
                "p50_token_latency_ms": p50_ms,
                "p95_token_latency_ms": p95_ms,
                "generated_token_ids": generated_tokens,
                "generated_token_sha256": _sha(generated_tokens),
                "route_trace": route_trace,
                "route_trace_sha256": _sha(route_trace),
                "expert_hashes": expert_hashes,
                "expert_route_binding": _expert_route_binding(
                    manifest_sha256="b" * 64,
                    route_trace=route_trace,
                    expert_hashes=expert_hashes,
                ),
            }
        )
    return result


def _hold_warmup_sample(tok_s: float) -> dict[str, object]:
    sample = _performance_samples(tok_s)[0]
    generated_tokens = list(range(400, 408))
    route_trace = [{"phase": "ar_decode", "layer": 0, "expert_ids": [3, 7, 11]}]
    expert_hashes = {
        "0:3": "2" * 64,
        "0:7": "3" * 64,
        "0:11": "4" * 64,
    }
    sample.update(
        generated_token_ids=generated_tokens,
        generated_token_sha256=_sha(generated_tokens),
        route_trace=route_trace,
        route_trace_sha256=_sha(route_trace),
        expert_hashes=expert_hashes,
        expert_route_binding=_expert_route_binding(
            manifest_sha256="b" * 64,
            route_trace=route_trace,
            expert_hashes=expert_hashes,
        ),
    )
    return sample


def _point(
    phase: str,
    timestamp_ns: int,
    *,
    expert: int,
    kv_blocks: int,
    kv_logical_tokens: int | None = None,
    active: int | None = None,
    cache: int = 0,
) -> dict[str, object]:
    health = _slot_health()
    kv = kv_blocks * HY3_Q4_KV_BLOCK_BYTES
    active_bytes = expert + kv if active is None else active
    allocator_cache_charged_bytes = max(
        cache,
        active_bytes + cache - expert - kv,
    )
    python_control_cpu = {
        category: {
            f"{phase}_{metric}": (
                1 if phase == "decode" and metric == "calls" else 0
            )
            for phase in ("decode", "prefill", "unscoped")
            for metric in ("calls", "cpu_ns")
        }
        for category in (
            "route_control",
            "cache_policy",
            "cache_budget",
            "kv_broker",
            "reader_thread",
        )
    }
    python_control_cpu["inclusive_relationships"] = {
        "cache_policy": "subset_of_route_control",
        "reader_thread": "separate_worker",
    }
    python_control_cpu["clock"] = "thread_time_ns"
    return {
        "phase": phase,
        "monotonic_ns": timestamp_ns,
        "allocator_active_bytes": active_bytes,
        "allocator_cache_bytes": cache,
        "allocator_peak_bytes": expert + kv + cache,
        "expert_cache_physical_bytes": expert,
        "kv_physical_bytes": kv,
        "kv_allocated_blocks": kv_blocks,
        "slot_health": health,
        "slot_health_sha256": _sha(health),
        "memory_limit_bytes": 100 * 1024**3,
        "allocator_headroom_bytes": 1024**3,
        "classified_limit_bytes": 99 * 1024**3,
        "classified_bytes": expert + kv,
        "charged_bytes": expert + kv + allocator_cache_charged_bytes,
        "charged_residual_bytes": (
            100 * 1024**3 - expert - kv - allocator_cache_charged_bytes
        ),
        "resident_model_bytes": 0,
        "kv_representation": "q4",
        "kv_logical_tokens": (
            kv_blocks * 16 if kv_logical_tokens is None else kv_logical_tokens
        ),
        "expert_logical_records": expert,
        "expert_allocated_records": expert,
        "expert_active_records": expert,
        "expert_resident_records": expert,
        "record_allocations": 0,
        "record_reuses": 0,
        "record_evictions": 0,
        "record_releases": 0,
        "pinned_expert_bytes": 0,
        "inflight_expert_bytes": 0,
        "speculative_expert_bytes": 0,
        "runtime_workspace_bytes": 0,
        "inflight_expert_staging_bytes": 0,
        "admission_failures": 0,
        "allocator_cache_charged_bytes": allocator_cache_charged_bytes,
        "process_rss_bytes": active_bytes,
        "process_compressed_bytes": 0,
        "system_swap_delta_bytes": 0,
        "gpu_utilization_percent": 20.0,
        "python_control_cpu": python_control_cpu,
        "failed_closed": False,
        "failure_reason": None,
    }


def _observation(
    arm: str,
    context_tokens: int,
    repetition: int,
    *,
    tok_s: float,
) -> dict[str, object]:
    tokens = [101, context_tokens, 202]
    routes = [
        {"phase": "ar_decode", "layer": 0, "expert_ids": [3, 7]},
        {"phase": "ar_decode", "layer": 1, "expert_ids": [2, 9]},
    ]
    if arm == "dynamic":
        final_blocks = context_tokens // 16
        timeline = [
            _point("pre_growth", 1, expert=800, kv_blocks=1, kv_logical_tokens=1),
            _point(
                "post_expert_reclaim",
                20,
                expert=600,
                kv_blocks=1,
                kv_logical_tokens=1,
            ),
            _point(
                "post_kv_growth",
                70,
                expert=600,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
            ),
            _point(
                "hold_warmup",
                3_000_000_000,
                expert=600,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
            ),
            _point(
                "hold",
                4_000_000_000,
                expert=600,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
            ),
            _point(
                "hold",
                4_500_000_000,
                expert=600,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
            ),
            _point(
                "hold",
                5_000_000_000,
                expert=600,
                kv_blocks=final_blocks,
                kv_logical_tokens=context_tokens,
            ),
            _point("post_reset", 6_000_000_000, expert=600, kv_blocks=0),
            _point("post_record_rewarm", 7_000_000_000, expert=800, kv_blocks=0),
        ]

        def growth_ledger(point: Mapping[str, object]) -> dict[str, object]:
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
                expert=600,
                kv_blocks=final_blocks - 1,
                kv_logical_tokens=1,
            )
        )
        second_gap = dict(first_after)
        second_gap["captured_monotonic_ns"] = 50
        second_after = growth_ledger(
            _point(
                "growth_step_1_after",
                60,
                expert=600,
                kv_blocks=final_blocks,
                kv_logical_tokens=1,
            )
        )
        kv_growth_steps = [
            {
                "sequence_index": 0,
                "requested_tokens": (final_blocks - 1) * 16,
                "target_blocks": final_blocks - 1,
                "before_monotonic_ns": 10,
                "reclaim_monotonic_ns": 20,
                "after_monotonic_ns": 30,
                "before": first_before,
                "reclaim_gap": first_gap,
                "after": first_after,
                "steady_delta_bytes": (final_blocks - 2) * HY3_Q4_KV_BLOCK_BYTES,
                "max_transient_delta_bytes": (final_blocks - 1)
                * (HY3_Q4_KV_BLOCK_BYTES // 80),
                "reclaimed_expert_bytes": 200,
                "kv_growth_bytes": (final_blocks - 2) * HY3_Q4_KV_BLOCK_BYTES,
            },
            {
                "sequence_index": 1,
                "requested_tokens": context_tokens,
                "target_blocks": final_blocks,
                "before_monotonic_ns": 40,
                "reclaim_monotonic_ns": 50,
                "after_monotonic_ns": 60,
                "before": first_after,
                "reclaim_gap": second_gap,
                "after": second_after,
                "steady_delta_bytes": HY3_Q4_KV_BLOCK_BYTES,
                "max_transient_delta_bytes": final_blocks
                * (HY3_Q4_KV_BLOCK_BYTES // 80),
                "reclaimed_expert_bytes": 0,
                "kv_growth_bytes": HY3_Q4_KV_BLOCK_BYTES,
            },
        ]
        start = {
            "kind": "dynamic-declared-q4",
            "kv_physical_bytes": HY3_Q4_KV_BLOCK_BYTES,
            "kv_blocks": 1,
        }
    else:
        kv_growth_steps = []
        timeline = [
            _point("pre_growth", 1, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
            _point("post_kv_growth", 2, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
            _point(
                "hold_warmup",
                2_500_000_000,
                expert=600,
                kv_blocks=HY3_Q4_MAX_BLOCKS,
            ),
            _point("hold", 3_000_000_000, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
            _point("hold", 3_500_000_000, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
            _point("hold", 4_000_000_000, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
            _point("post_reset", 5_000_000_000, expert=600, kv_blocks=0),
        ]
        start = {
            "kind": "static-128k-reserved-q4",
            "kv_physical_bytes": HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES,
            "kv_blocks": HY3_Q4_MAX_BLOCKS,
        }
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
        "cache_start_state": start,
        "identity": identity,
        "prompt_sha256": "1" * 64,
        "generated_token_ids": tokens,
        "generated_token_sha256": _sha(tokens),
        "route_trace": routes,
        "route_trace_sha256": _sha(routes),
        "expert_hashes": expert_hashes,
        "expert_route_binding": _expert_route_binding(
            manifest_sha256=str(identity["expert_manifest_sha256"]),
            route_trace=routes,
            expert_hashes=expert_hashes,
        ),
        "kv_growth_steps": kv_growth_steps,
        "timeline": timeline,
        "metrics": {
            "generated_tokens": len(tokens),
            "elapsed_seconds": len(tokens) / tok_s,
            "tokens_per_second": tok_s,
            "peak_charged_bytes": max(point["charged_bytes"] for point in timeline),
            "stress_peak_charged_bytes": max(
                max(
                    point["allocator_peak_bytes"] + point["allocator_cache_bytes"],
                    point["charged_bytes"],
                )
                for point in timeline
            ),
            "hold_performance_samples": [tok_s * 0.99, tok_s, tok_s * 1.01],
            "hold_warmup_sample": _hold_warmup_sample(tok_s),
            "performance_samples": _performance_samples(tok_s),
        },
    }


def _probe_result(*, released: int = 128, replacement_executable: bool = True):
    return {
        "schema": "mtplx-hy3-direct-cache-probe-v1",
        "identity": _identity(),
        "backend": "mlx-metal-direct-slots",
        "startup_persistent_bytes": 0,
        "first_record_allocated_bytes": 128,
        "first_record_executable": True,
        "replacement_buffer_identity_preserved": True,
        "replacement_record_executable": replacement_executable,
        "released_record_bytes": 128,
        "before_allocation": {
            "active_bytes": 700,
            "cache_bytes": 200,
            "peak_bytes": 700,
        },
        "before_release": {
            "active_bytes": 1_000,
            "cache_bytes": 200,
            "peak_bytes": 1_200,
        },
        "after_release": {
            "active_bytes": 1_000 - released,
            "cache_bytes": 200,
            "peak_bytes": 1_200,
        },
        "allocator_charged_drop_bytes": released,
        "error": None,
    }


def _artifact_attestation(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "mtplx-hy3-artifact-attestation-v1",
        "model_artifact_sha256": "a" * 64,
        "expert_manifest_sha256": "b" * 64,
        "artifact_pins_sha256": "d" * 64,
        "artifact_stat_sha256": _ARTIFACT_STAT_SHA256,
        "sidecar_fingerprint": dict(_ARTIFACT_FINGERPRINT),
        "resident_payload_bytes": 3,
        "resident_payload_sha256": "e" * 64,
        "resident_shard_fingerprints": [
            dict(value) for value in _RESIDENT_SHARD_FINGERPRINTS
        ],
        "payload_hash_verified": True,
        "payload_hash_io_mode": "f-nocache",
    }
    result.update(overrides)
    return result


def test_context_matrix_and_schedule_balance_both_physical_orders() -> None:
    assert CONTEXT_MATRIX_TOKENS == (4096, 32768, 65536, 131072)

    schedule = balanced_campaign_schedule(repetitions=4)

    assert len(schedule) == 4 * 4 * 2
    for context in CONTEXT_MATRIX_TOKENS:
        rows = [row for row in schedule if row.context_tokens == context]
        assert [row.arms for row in rows[::2]] == [
            ("static", "dynamic"),
            ("dynamic", "static"),
            ("static", "dynamic"),
            ("dynamic", "static"),
        ]
        assert sum(row.arm == "static" and row.order_index == 0 for row in rows) == 2
        assert sum(row.arm == "dynamic" and row.order_index == 0 for row in rows) == 2


def test_schedule_rejects_an_unbalanced_repetition_count() -> None:
    with pytest.raises(ValueError, match="positive even"):
        balanced_campaign_schedule(repetitions=3)


def test_schedule_allows_an_ordered_context_subset_for_staged_foreground_runs() -> None:
    schedule = balanced_campaign_schedule(repetitions=2, contexts=(4_096,))

    assert [(row.context_tokens, row.arm) for row in schedule] == [
        (4_096, "static"),
        (4_096, "dynamic"),
        (4_096, "dynamic"),
        (4_096, "static"),
    ]


@pytest.mark.parametrize("contexts", ((), (4_096, 4_096), (32_768, 4_096), (8_192,)))
def test_schedule_rejects_invalid_context_subsets(contexts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="ordered subset"):
        balanced_campaign_schedule(repetitions=2, contexts=contexts)


def test_observation_requires_declared_start_and_physical_ordering() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)

    validated = validate_campaign_observation(row)

    assert isinstance(validated.cache_start_state, CacheStartState)
    assert validated.context_tokens == 4096

    stale_kind = _observation("dynamic", 4096, 0, tok_s=12.0)
    stale_kind["cache_start_state"]["kind"] = "empty-q4"
    with pytest.raises(BenchmarkGateError, match="dynamic-declared-q4"):
        validate_campaign_observation(stale_kind)

    reordered = _observation("dynamic", 4096, 0, tok_s=12.0)
    reordered["timeline"][1]["kv_allocated_blocks"] = 2
    reordered["timeline"][1]["kv_physical_bytes"] = 2 * HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["allocator_active_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["allocator_peak_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["classified_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["charged_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["charged_residual_bytes"] -= HY3_Q4_KV_BLOCK_BYTES
    with pytest.raises(BenchmarkGateError, match="before KV"):
        validate_campaign_observation(reordered)


def test_observation_requires_exact_issue46_q4_physical_geometry() -> None:
    assert HY3_Q4_KV_BYTES_PER_TOKEN == 84_480
    assert HY3_Q4_KV_BLOCK_BYTES == 16 * 84_480
    assert HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES == int(10.3125 * 1024**3)

    static = validate_campaign_observation(_observation("static", 4096, 0, tok_s=12.0))
    assert static.cache_start_state.kv_physical_bytes == int(10.3125 * 1024**3)


def test_dynamic_observation_rejects_overallocated_q4_blocks() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    for point in row["timeline"]:
        if point["phase"] not in {"post_kv_growth", "hold"}:
            continue
        point["kv_allocated_blocks"] += 1
        point["kv_physical_bytes"] += HY3_Q4_KV_BLOCK_BYTES
        point["allocator_active_bytes"] += HY3_Q4_KV_BLOCK_BYTES
        point["allocator_peak_bytes"] += HY3_Q4_KV_BLOCK_BYTES
        point["classified_bytes"] += HY3_Q4_KV_BLOCK_BYTES
        point["charged_bytes"] += HY3_Q4_KV_BLOCK_BYTES
        point["charged_residual_bytes"] -= HY3_Q4_KV_BLOCK_BYTES
        point["process_rss_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    row["metrics"]["peak_charged_bytes"] = max(
        point["charged_bytes"] for point in row["timeline"]
    )
    row["metrics"]["stress_peak_charged_bytes"] = max(
        max(
            point["allocator_peak_bytes"] + point["allocator_cache_bytes"],
            point["charged_bytes"],
        )
        for point in row["timeline"]
    )

    with pytest.raises(BenchmarkGateError, match="exactly cover"):
        validate_campaign_observation(row)


def test_production_identity_pins_every_q4_attention_route_control() -> None:
    expected = {
        "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW": "0",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "0",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE": "off",
        "MTPLX_PAGED_GQA_SDPA_ROUTE": "off",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA": "0",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT": "65536",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q": "4",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q": "5",
        "MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q": "16",
        "MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE": "2048",
        "MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE": "1024",
    }

    for arm in ("static", "dynamic"):
        identity = _production_lane_identity(arm)
        assert identity["arm_config"]["attention_runtime_env"] == expected

    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    arm_config = row["identity"]["arm_config"]
    arm_config["attention_runtime_env"]["MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q"] = "4096"
    row["identity"]["arm_config_sha256"] = _sha(arm_config)
    normalized = runner_module.normalize_arm_config(arm_config)
    row["identity"]["normalized_config"] = normalized
    row["identity"]["normalized_config_sha256"] = _sha(normalized)

    with pytest.raises(
        BenchmarkGateError, match="exact full-history Q4 attention route"
    ):
        validate_campaign_observation(row)


def test_observation_requires_exact_headroom_and_classified_limit() -> None:
    wrong_headroom = _observation("dynamic", 4096, 0, tok_s=12.0)
    point = wrong_headroom["timeline"][0]
    point["allocator_headroom_bytes"] = 0
    point["classified_limit_bytes"] = 100 * 1024**3
    with pytest.raises(BenchmarkGateError, match="exactly 1 GiB"):
        validate_campaign_observation(wrong_headroom)

    overclassified = _observation("dynamic", 4096, 0, tok_s=12.0)
    point = overclassified["timeline"][0]
    delta = int(point["classified_limit_bytes"]) + 1 - int(point["classified_bytes"])
    point["expert_cache_physical_bytes"] += delta
    point["allocator_active_bytes"] += delta
    point["allocator_peak_bytes"] += delta
    point["classified_bytes"] += delta
    point["charged_bytes"] += delta
    point["charged_residual_bytes"] -= delta
    point["process_rss_bytes"] += delta
    with pytest.raises(BenchmarkGateError, match="exceeds the classified limit"):
        validate_campaign_observation(overclassified)

    bad_start = _observation("dynamic", 4096, 0, tok_s=12.0)
    bad_start["cache_start_state"]["kv_physical_bytes"] += 1
    with pytest.raises(BenchmarkGateError, match="exact Q4 geometry"):
        validate_campaign_observation(bad_start)

    bad_timeline = _observation("dynamic", 4096, 0, tok_s=12.0)
    bad_timeline["timeline"][3]["kv_physical_bytes"] += 1
    with pytest.raises(BenchmarkGateError, match="exact Q4 geometry"):
        validate_campaign_observation(bad_timeline)

    with pytest.raises(BenchmarkGateError, match="exact Q4 geometry"):
        CacheStartState.from_mapping(
            {"kind": "impossible-q4", "kv_physical_bytes": 1, "kv_blocks": 0}
        )


def test_observation_requires_stable_hold_reset_rewarm_and_block_crossing() -> None:
    unstable_resource = _observation("dynamic", 4096, 0, tok_s=12.0)
    unstable_resource["timeline"][4]["record_allocations"] += 1
    with pytest.raises(BenchmarkGateError, match="record_allocations"):
        validate_campaign_observation(unstable_resource)

    no_reset = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_reset["timeline"] = no_reset["timeline"][:-2]
    with pytest.raises(BenchmarkGateError, match="post_reset"):
        validate_campaign_observation(no_reset)

    no_rewarm = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_rewarm["timeline"] = no_rewarm["timeline"][:-1]
    with pytest.raises(BenchmarkGateError, match="post_record_rewarm"):
        validate_campaign_observation(no_rewarm)

    no_boundary = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_boundary["cache_start_state"]["kv_blocks"] = 255
    no_boundary["cache_start_state"]["kv_physical_bytes"] = 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][0]["kv_allocated_blocks"] = 255
    no_boundary["timeline"][0]["kv_physical_bytes"] = 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][0]["allocator_active_bytes"] = (
        800 + 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][0]["allocator_peak_bytes"] = (
        800 + 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][0]["charged_bytes"] = 800 + 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][0]["classified_bytes"] = 800 + 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][0]["charged_residual_bytes"] = (
            100 * 1024**3 - 800 - 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][1]["kv_allocated_blocks"] = 255
    no_boundary["timeline"][1]["kv_physical_bytes"] = 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][1]["allocator_active_bytes"] = (
        600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][1]["allocator_peak_bytes"] = (
        600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][1]["charged_bytes"] = 600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][1]["classified_bytes"] = 600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][1]["charged_residual_bytes"] = (
            100 * 1024**3 - 600 - 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][2]["kv_allocated_blocks"] = 256
    with pytest.raises(
        BenchmarkGateError, match="one physical Q4 block|first KV growth"
    ):
        validate_campaign_observation(no_boundary)

    too_short = _observation("dynamic", 4096, 0, tok_s=12.0)
    too_short["timeline"][4]["monotonic_ns"] = 4_100_000_000
    too_short["timeline"][5]["monotonic_ns"] = 4_200_000_000
    with pytest.raises(BenchmarkGateError, match="hold duration"):
        validate_campaign_observation(too_short)

    no_perf_samples = _observation("dynamic", 4096, 0, tok_s=12.0)
    del no_perf_samples["metrics"]["hold_performance_samples"]
    with pytest.raises(BenchmarkGateError, match="hold_performance_samples"):
        validate_campaign_observation(no_perf_samples)

    unstable_perf = _observation("dynamic", 4096, 0, tok_s=12.0)
    unstable_values = [6.0, 10.0, 12.0]
    unstable_perf["metrics"]["hold_performance_samples"] = unstable_values
    for sample, value in zip(
        unstable_perf["metrics"]["performance_samples"],
        unstable_values,
        strict=True,
    ):
        sample["tokens_per_second"] = value
    with pytest.raises(BenchmarkGateError, match="not stable within 10%") as exc_info:
        validate_campaign_observation(unstable_perf)
    diagnostic = str(exc_info.value)
    assert '"hold_warmup"' in diagnostic
    assert '"hold_samples"' in diagnostic
    assert '"tokens_per_second": 6.0' in diagnostic
    assert '"expert_hit_rate": 0.7' in diagnostic
    assert '"ssd_bytes_per_token": 1024.0' in diagnostic


def test_observation_requires_exact_phase_order_and_logical_kv_bounds() -> None:
    static_reordered = _observation("static", 4096, 0, tok_s=12.0)
    static_reordered["timeline"][0]["phase"] = "post_kv_growth"
    static_reordered["timeline"][1]["phase"] = "pre_growth"
    with pytest.raises(BenchmarkGateError, match="order pre-growth"):
        validate_campaign_observation(static_reordered)

    reset_between_holds = _observation("dynamic", 4096, 0, tok_s=12.0)
    reset = reset_between_holds["timeline"].pop(7)
    reset["monotonic_ns"] = 4_250_000_000
    reset_between_holds["timeline"].insert(5, reset)
    with pytest.raises(BenchmarkGateError, match="every hold"):
        validate_campaign_observation(reset_between_holds)

    no_logical_context = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_logical_context["timeline"][2]["kv_logical_tokens"] = 0
    with pytest.raises(BenchmarkGateError, match="requested context"):
        validate_campaign_observation(no_logical_context)

    impossible_logical_capacity = _observation("dynamic", 4096, 0, tok_s=12.0)
    impossible_logical_capacity["timeline"][0]["kv_logical_tokens"] = 17
    with pytest.raises(BenchmarkGateError, match="physical Q4 capacity"):
        validate_campaign_observation(impossible_logical_capacity)


def test_observation_requires_exactly_one_ordered_hold_warmup() -> None:
    missing = _observation("dynamic", 4096, 0, tok_s=12.0)
    missing["timeline"] = [
        point for point in missing["timeline"] if point["phase"] != "hold_warmup"
    ]
    with pytest.raises(BenchmarkGateError, match="hold_warmup"):
        validate_campaign_observation(missing)

    duplicate = _observation("dynamic", 4096, 0, tok_s=12.0)
    extra = dict(duplicate["timeline"][3])
    extra["monotonic_ns"] = 3_500_000_000
    duplicate["timeline"].insert(4, extra)
    with pytest.raises(BenchmarkGateError, match="hold_warmup"):
        validate_campaign_observation(duplicate)

    misplaced = _observation("dynamic", 4096, 0, tok_s=12.0)
    warmup = misplaced["timeline"].pop(3)
    warmup["monotonic_ns"] = 4_250_000_000
    misplaced["timeline"].insert(4, warmup)
    with pytest.raises(BenchmarkGateError, match="hold_warmup"):
        validate_campaign_observation(misplaced)


@pytest.mark.parametrize("phase", ("pre_growth", "post_expert_reclaim"))
def test_dynamic_observation_rejects_pre_growth_logical_page_capacity(
    phase: str,
) -> None:
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    target = next(point for point in observation["timeline"] if point["phase"] == phase)
    target["kv_logical_tokens"] = 16

    with pytest.raises(BenchmarkGateError, match="logical KV ownership.*exactly 1"):
        validate_campaign_observation(observation)


@pytest.mark.parametrize(
    ("step_index", "ledger_name"),
    tuple(
        (step_index, ledger_name)
        for step_index in (0, 1)
        for ledger_name in ("before", "reclaim_gap", "after")
    ),
)
def test_dynamic_observation_rejects_growth_ledger_logical_page_capacity(
    step_index: int,
    ledger_name: str,
) -> None:
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    observation["kv_growth_steps"][step_index][ledger_name]["kv_logical_tokens"] = 16

    with pytest.raises(BenchmarkGateError, match="logical KV ownership.*exactly 1"):
        validate_campaign_observation(observation)


def test_observation_rejects_hold_physical_record_churn_and_fixed_pool_drift() -> None:
    churn = _observation("dynamic", 4096, 0, tok_s=12.0)
    cumulative_fields = (
        "record_allocations",
        "record_releases",
    )
    for increment, point in enumerate(churn["timeline"][4:6], start=1):
        for field in cumulative_fields:
            point[field] += increment
    with pytest.raises(BenchmarkGateError, match="classified or charged memory ledger"):
        validate_campaign_observation(churn)

    fixed_drift = _observation("dynamic", 4096, 0, tok_s=12.0)
    point = fixed_drift["timeline"][1]
    point["resident_model_bytes"] += 1
    point["classified_bytes"] += 1
    point["charged_bytes"] += 1
    point["charged_residual_bytes"] -= 1
    with pytest.raises(BenchmarkGateError, match="fixed physical memory pools"):
        validate_campaign_observation(fixed_drift)


def test_observation_allows_ordinary_hold_record_eviction_churn() -> None:
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    holds = [point for point in observation["timeline"] if point["phase"] == "hold"]
    for point, evictions in zip(holds, (3242, 4027, 4674), strict=True):
        point["record_evictions"] = evictions

    validated = validate_campaign_observation(observation)

    assert validated.arm == "dynamic"


def test_observation_allows_one_raw_allocator_bookkeeping_buffer_of_hold_jitter() -> (
    None
):
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    holds = [point for point in observation["timeline"] if point["phase"] == "hold"]
    for point in holds:
        point["allocator_cache_charged_bytes"] = 8 * 1024
        point["charged_bytes"] += 8 * 1024
        point["charged_residual_bytes"] -= 8 * 1024
    holds[1]["allocator_cache_bytes"] = 8 * 1024
    observation["metrics"]["peak_charged_bytes"] = max(
        point["charged_bytes"] for point in observation["timeline"]
    )
    observation["metrics"]["stress_peak_charged_bytes"] = max(
        max(
            point["allocator_peak_bytes"] + point["allocator_cache_bytes"],
            point["charged_bytes"],
        )
        for point in observation["timeline"]
    )

    validated = validate_campaign_observation(observation)

    assert validated.arm == "dynamic"


def test_observation_allows_measured_raw_allocator_cache_churn_when_fully_charged() -> (
    None
):
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    holds = [point for point in observation["timeline"] if point["phase"] == "hold"]
    measured_cache_bytes = (1_079_101_220, 1_076_971_300, 1_080_870_692)
    for point, cache_bytes in zip(holds, measured_cache_bytes, strict=True):
        point["allocator_cache_bytes"] = cache_bytes
        point["allocator_cache_charged_bytes"] = cache_bytes
        point["charged_bytes"] = point["classified_bytes"] + cache_bytes
        point["charged_residual_bytes"] = (
            point["memory_limit_bytes"] - point["charged_bytes"]
        )
    observation["metrics"]["peak_charged_bytes"] = max(
        point["charged_bytes"] for point in observation["timeline"]
    )
    observation["metrics"]["stress_peak_charged_bytes"] = max(
        max(
            point["allocator_peak_bytes"] + point["allocator_cache_bytes"],
            point["charged_bytes"],
        )
        for point in observation["timeline"]
    )

    validated = validate_campaign_observation(observation)

    assert validated.arm == "dynamic"


def test_observation_rejects_raw_allocator_cache_churn_that_is_not_charged() -> None:
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    holds = [point for point in observation["timeline"] if point["phase"] == "hold"]
    holds[1]["allocator_cache_bytes"] = 8 * 1024 + 1

    with pytest.raises(BenchmarkGateError, match="charged.*below raw MLX"):
        validate_campaign_observation(observation)


def test_observation_rejects_hold_record_eviction_counter_regression() -> None:
    observation = _observation("dynamic", 4096, 0, tok_s=12.0)
    hold_window = [
        point
        for point in observation["timeline"]
        if point["phase"] in {"hold_warmup", "hold"}
    ]
    for point, evictions in zip(hold_window, (3200, 3242, 3100, 4674), strict=True):
        point["record_evictions"] = evictions

    with pytest.raises(BenchmarkGateError, match="record_evictions.*decreased"):
        validate_campaign_observation(observation)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("memory_limit_bytes", 1, "memory_limit_bytes"),
        ("dynamic_expert_cache", False, "contradicts"),
    ),
)
def test_observation_identity_binds_the_resolved_runtime_lane(
    field: str,
    value: object,
    match: str,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    identity = row["identity"]
    arm_config = identity["arm_config"]
    arm_config["expert_streaming_config"][field] = value
    identity["arm_config_sha256"] = _sha(arm_config)
    normalized = runner_module.normalize_arm_config(arm_config)
    identity["normalized_config"] = normalized
    identity["normalized_config_sha256"] = _sha(normalized)

    with pytest.raises(BenchmarkGateError, match=match):
        validate_campaign_observation(row)


def test_production_arm_interventions_normalize_to_one_paired_identity() -> None:
    validated_identities: dict[str, Mapping[str, object]] = {}
    for arm in ("static", "dynamic"):
        identity = _build_identity(
            _production_lane_identity(arm),
            request=ArmRequest(arm=arm, context_tokens=4096, repetition=0),
            source_git_commit="c" * 40,
        )
        row = _observation(arm, 4096, 0, tok_s=12.0)
        row["identity"] = identity
        validated_identities[arm] = validate_campaign_observation(row).identity

    static = validated_identities["static"]
    dynamic = validated_identities["dynamic"]
    assert static["arm_config"] != dynamic["arm_config"]
    assert static["arm_config"]["planned_persistent_slots"] == 8_673
    assert dynamic["arm_config"]["planned_persistent_slots"] == 8_673
    assert static["normalized_config"] == dynamic["normalized_config"]
    assert static["normalized_config_sha256"] == dynamic["normalized_config_sha256"]


def test_observation_requires_and_preserves_detailed_performance_samples() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)

    validated = validate_campaign_observation(row)

    assert (
        validated.metrics["performance_samples"]
        == row["metrics"]["performance_samples"]
    )

    missing = _observation("dynamic", 4096, 0, tok_s=12.0)
    del missing["metrics"]["performance_samples"]
    with pytest.raises(BenchmarkGateError, match="performance_samples"):
        validate_campaign_observation(missing)


@pytest.mark.parametrize(
    "field",
    (
        "tokens_per_second",
        "expert_hit_rate",
        "ssd_bytes_per_token",
        "p50_token_latency_ms",
        "p95_token_latency_ms",
        "generated_token_ids",
        "generated_token_sha256",
        "route_trace",
        "route_trace_sha256",
        "expert_hashes",
        "expert_route_binding",
    ),
)
def test_observation_requires_every_detailed_performance_field(field: str) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    del row["metrics"]["performance_samples"][0][field]

    with pytest.raises(BenchmarkGateError, match=field):
        validate_campaign_observation(row)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("tokens_per_second", float("nan")),
        ("expert_hit_rate", -0.01),
        ("expert_hit_rate", 1.01),
        ("ssd_bytes_per_token", -1.0),
        ("p50_token_latency_ms", 0.0),
        ("p95_token_latency_ms", float("inf")),
    ),
)
def test_observation_rejects_invalid_detailed_performance_values(
    field: str,
    invalid_value: float,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["metrics"]["performance_samples"][0][field] = invalid_value

    with pytest.raises(BenchmarkGateError, match=field):
        validate_campaign_observation(row)


def test_observation_rejects_contradictory_detailed_performance_samples() -> None:
    wrong_count = _observation("dynamic", 4096, 0, tok_s=12.0)
    wrong_count["metrics"]["performance_samples"].pop()
    with pytest.raises(BenchmarkGateError, match="performance_samples"):
        validate_campaign_observation(wrong_count)

    wrong_tps = _observation("dynamic", 4096, 0, tok_s=12.0)
    wrong_tps["metrics"]["performance_samples"][0]["tokens_per_second"] = 99.0
    with pytest.raises(BenchmarkGateError, match="tokens_per_second"):
        validate_campaign_observation(wrong_tps)

    reversed_latency = _observation("dynamic", 4096, 0, tok_s=12.0)
    reversed_latency["metrics"]["performance_samples"][0].update(
        p50_token_latency_ms=100.0,
        p95_token_latency_ms=99.0,
    )
    with pytest.raises(BenchmarkGateError, match="p95_token_latency_ms"):
        validate_campaign_observation(reversed_latency)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("generated_token_ids", []),
        ("generated_token_ids", [True]),
        ("generated_token_sha256", "0" * 64),
        ("route_trace", []),
        ("route_trace", "not-a-route-array"),
        ("route_trace_sha256", "0" * 64),
    ),
)
def test_observation_rejects_invalid_hold_workload_evidence(
    field: str,
    invalid_value: object,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["metrics"]["performance_samples"][0][field] = invalid_value

    with pytest.raises(BenchmarkGateError, match=field):
        validate_campaign_observation(row)


def test_observation_recomputes_token_route_and_slot_health_hashes() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["generated_token_ids"].append(303)
    with pytest.raises(BenchmarkGateError, match="generated_token_sha256"):
        validate_campaign_observation(row)

    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["route_trace"][0]["expert_ids"].append(11)
    with pytest.raises(BenchmarkGateError, match="route_trace_sha256"):
        validate_campaign_observation(row)

    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["timeline"][-1]["slot_health"]["failed"] = 1
    with pytest.raises(BenchmarkGateError, match="slot health"):
        validate_campaign_observation(row)

    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["identity"]["arm_config"]["context_window"] = 65_536
    with pytest.raises(BenchmarkGateError, match="arm_config_sha256"):
        validate_campaign_observation(row)


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong_route"))
def test_observation_expert_hashes_cover_exact_routed_pairs(mutation: str) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    if mutation == "missing":
        del row["expert_hashes"]["0:7"]
    elif mutation == "extra":
        row["expert_hashes"]["2:11"] = "6" * 64
    else:
        row["route_trace"][0]["expert_ids"] = [3, 11]
        row["route_trace_sha256"] = canonical_sha256(row["route_trace"])

    with pytest.raises(BenchmarkGateError, match="expert hashes.*routed"):
        validate_campaign_observation(row)


def test_observation_rejects_zeroed_expert_hash_with_unchanged_binding() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["expert_hashes"]["0:3"] = "0" * 64

    with pytest.raises(BenchmarkGateError, match=r"expert.*SHA-256|zero"):
        validate_campaign_observation(row)


@pytest.mark.parametrize("mutation", ("malformed_route", "missing_hash", "extra_hash"))
def test_observation_rejects_unbound_hold_route_payload(
    mutation: str,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    sample = row["metrics"]["performance_samples"][0]
    if mutation == "malformed_route":
        sample["route_trace"] = [{"phase": "ar_decode", "layer": 1}]
        sample["route_trace_sha256"] = canonical_sha256(sample["route_trace"])
    elif mutation == "missing_hash":
        del sample["expert_hashes"]["1:7"]
    else:
        sample["expert_hashes"]["9:9"] = "9" * 64

    with pytest.raises(BenchmarkGateError, match=r"route|expert hashes|binding"):
        validate_campaign_observation(row)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expert_manifest_sha256", "f" * 64),
        ("route_trace_sha256", "f" * 64),
        ("expert_hashes_sha256", "f" * 64),
        ("producer_verification_scope", "unverified"),
        ("offline_verification_scope", "content-verified"),
    ),
)
def test_observation_rejects_false_expert_route_binding(
    field: str,
    value: str,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["expert_route_binding"][field] = value

    with pytest.raises(BenchmarkGateError, match=r"expert_route_binding|binding"):
        validate_campaign_observation(row)


def test_observation_rejects_hold_zeroed_hash_even_if_binding_is_recomputed() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    sample = row["metrics"]["performance_samples"][0]
    sample["expert_hashes"]["1:3"] = "0" * 64
    sample["expert_route_binding"]["expert_hashes_sha256"] = canonical_sha256(
        sample["expert_hashes"]
    )

    with pytest.raises(BenchmarkGateError, match=r"expert.*SHA-256|zero"):
        validate_campaign_observation(row)


def test_observation_recomputes_exact_per_entry_growth_transient() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    row["kv_growth_steps"][0]["max_transient_delta_bytes"] += 1

    with pytest.raises(BenchmarkGateError, match="transient.*geometry"):
        validate_campaign_observation(row)


def test_observation_requires_complete_growth_resource_ledgers() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    del row["kv_growth_steps"][0]["reclaim_gap"]["charged_bytes"]

    with pytest.raises(BenchmarkGateError, match=r"reclaim_gap.*charged_bytes"):
        validate_campaign_observation(row)


def test_observation_rejects_unaccounted_two_tib_growth_allocator_footprint() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    gap = row["kv_growth_steps"][0]["reclaim_gap"]
    gap["allocator_active_bytes"] = 1024**4
    gap["allocator_cache_bytes"] = 1024**4

    with pytest.raises(BenchmarkGateError, match=r"charged|allocator"):
        validate_campaign_observation(row)


@pytest.mark.parametrize("ledger_kind", ("charged", "classified", "transient"))
def test_observation_growth_ledgers_stay_within_the_configured_limit(
    ledger_kind: str,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    gap = row["kv_growth_steps"][0]["reclaim_gap"]
    memory_limit = 100 * 1024**3
    if ledger_kind == "charged":
        gap["charged_bytes"] = memory_limit + 1
        gap["charged_residual_bytes"] = -1
        gap["allocator_cache_charged_bytes"] = (
            memory_limit + 1 - gap["classified_bytes"]
        )
    elif ledger_kind == "classified":
        increase = gap["classified_limit_bytes"] + 1 - gap["classified_bytes"]
        gap["runtime_workspace_bytes"] += increase
        gap["classified_bytes"] += increase
        gap["charged_bytes"] += increase
        gap["charged_residual_bytes"] -= increase
        gap["allocator_active_bytes"] += increase
        gap["allocator_peak_bytes"] += increase
    else:
        gap["allocator_peak_bytes"] = memory_limit + 1

    with pytest.raises(
        BenchmarkGateError,
        match=r"100 GiB|configured memory limit|classified limit",
    ):
        validate_campaign_observation(row)


def _adjust_growth_expert_ledger(ledger: dict[str, object], delta: int) -> None:
    ledger["expert_cache_physical_bytes"] += delta
    ledger["allocator_active_bytes"] += delta
    ledger["allocator_peak_bytes"] += delta
    ledger["classified_bytes"] += delta
    ledger["charged_bytes"] += delta
    ledger["charged_residual_bytes"] -= delta
    ledger["process_rss_bytes"] += delta


@pytest.mark.parametrize("checkpoint", ("first_before", "reclaim", "final_after"))
def test_observation_cross_links_growth_ledgers_to_timeline(checkpoint: str) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    steps = row["kv_growth_steps"]
    if checkpoint == "first_before":
        _adjust_growth_expert_ledger(steps[0]["before"], 1)
        steps[0]["reclaimed_expert_bytes"] += 1
    elif checkpoint == "reclaim":
        gap = steps[0]["reclaim_gap"]
        gap["allocator_cache_bytes"] += 1
        gap["allocator_cache_charged_bytes"] += 1
        gap["charged_bytes"] += 1
        gap["charged_residual_bytes"] -= 1
    else:
        _adjust_growth_expert_ledger(steps[-1]["after"], -1)

    with pytest.raises(BenchmarkGateError, match=r"timeline.*growth|growth.*timeline"):
        validate_campaign_observation(row)


@pytest.mark.parametrize("checkpoint", ("pre", "reclaim", "growth"))
def test_observation_cross_links_growth_timestamps_to_timeline(
    checkpoint: str,
) -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)
    if checkpoint == "pre":
        row["timeline"][0]["monotonic_ns"] = 11
    elif checkpoint == "reclaim":
        row["timeline"][1]["monotonic_ns"] = 21
    else:
        row["timeline"][2]["monotonic_ns"] = 59

    with pytest.raises(BenchmarkGateError, match=r"timeline.*growth|growth.*timeline"):
        validate_campaign_observation(row)


def test_observation_rejects_a_global_device_synchronization() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=10.0)
    for point in row["timeline"]:
        point["slot_health"]["global_device_synchronizations"] = 1
        point["slot_health_sha256"] = _sha(point["slot_health"])

    with pytest.raises(
        BenchmarkGateError,
        match="global_device_synchronizations",
    ):
        validate_campaign_observation(row)


def test_allocator_probe_requires_direct_record_release_and_replacement_execution() -> (
    None
):
    passed = validate_allocator_probe(_probe_result())
    assert passed["gate_passed"] is True
    assert passed["charged_release_bytes"] == 128

    with pytest.raises(BenchmarkGateError, match="charged allocator"):
        validate_allocator_probe(_probe_result(released=127))
    with pytest.raises(BenchmarkGateError, match="replacement_record_executable"):
        validate_allocator_probe(_probe_result(replacement_executable=False))


def test_allocator_probe_allocates_replaces_and_releases_one_direct_record() -> None:
    calls: list[object] = []
    samples = iter(
        (
            AllocatorSample(active_bytes=100, cache_bytes=20, peak_bytes=120),
            AllocatorSample(active_bytes=300, cache_bytes=20, peak_bytes=320),
            AllocatorSample(active_bytes=200, cache_bytes=20, peak_bytes=320),
        )
    )
    result = run_direct_cache_probe(
        identity=_identity(),
        backend="mlx-metal-direct-slots",
        startup_persistent_bytes=0,
        sample_allocator=lambda: next(samples),
        allocate_first_record=lambda: calls.append("allocate") or 100,
        execute_first_record=lambda: calls.append("execute-first") or True,
        replace_record=lambda: calls.append("replace") or True,
        execute_replacement_record=(
            lambda: calls.append("execute-replacement") or True
        ),
        release_record=lambda: calls.append("release") or 100,
    )

    assert result["gate_passed"] is True
    assert calls == [
        "allocate",
        "execute-first",
        "replace",
        "execute-replacement",
        "release",
    ]


def test_allocator_probe_fails_closed_on_injected_errors() -> None:
    result = run_direct_cache_probe(
        identity=_identity(),
        backend="mlx-metal-direct-slots",
        startup_persistent_bytes=0,
        sample_allocator=lambda: AllocatorSample(0, 0, 0),
        allocate_first_record=lambda: (_ for _ in ()).throw(
            RuntimeError("injected")
        ),
        execute_first_record=lambda: True,
        replace_record=lambda: True,
        execute_replacement_record=lambda: True,
        release_record=lambda: 1,
    )

    assert result["gate_passed"] is False
    assert "injected" in result["error"]


def test_balanced_campaign_retains_raw_pairs_and_confidence_intervals() -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int) -> Mapping[str, object]:
        base = 10.0 + repetition
        return _observation(
            arm,
            context_tokens,
            repetition,
            tok_s=base * (1.1 if arm == "dynamic" else 1.0),
        )

    result = run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=4,
        bootstrap_resamples=500,
        bootstrap_seed=46,
    )

    assert result["schema"] == "mtplx-hy3-dynamic-memory-campaign-v1"
    assert len(result["raw_observations"]) == 32
    assert len(result["paired_samples"]) == 16
    assert set(result["contexts"]) == set(CONTEXT_MATRIX_TOKENS)
    for summary in result["paired_metrics_by_context"].values():
        assert summary["sample_count"] == 4
        assert summary["dynamic_vs_static_tps_ratio"]["mean"] == pytest.approx(1.1)
        low, high = summary["dynamic_vs_static_tps_ratio"]["confidence_interval_95"]
        assert low <= 1.1 <= high


def test_balanced_campaign_runs_only_the_selected_context_subset() -> None:
    probe = validate_allocator_probe(_probe_result())
    calls: list[tuple[str, int, int]] = []

    def execute(arm: str, context_tokens: int, repetition: int) -> Mapping[str, object]:
        calls.append((arm, context_tokens, repetition))
        return _observation(
            arm,
            context_tokens,
            repetition,
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    result = run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=2,
        contexts=(4_096,),
        bootstrap_resamples=100,
    )

    assert result["contexts"] == [4_096]
    assert set(result["paired_metrics_by_context"]) == {"4096"}
    assert len(calls) == 4


def test_subprocess_campaign_attests_artifact_after_every_arm() -> None:
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        calls.append(command)
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            return _artifact_attestation()
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    result = runner_module.run_subprocess_campaign(
        probe_command=("probe",),
        artifact_verify_command=("verify-artifact",),
        arm_command_template=("arm", "{arm}", "{context_tokens}", "{repetition}"),
        repetitions=2,
        command_runner=run,
        bootstrap_resamples=100,
    )

    assert calls[0] == ("probe",)
    assert calls[-1] == ("verify-artifact",)
    assert len(calls[1:-1]) == len(CONTEXT_MATRIX_TOKENS) * 2 * 2
    assert result["post_campaign_artifact_attestation"] == _artifact_attestation()


def test_subprocess_campaign_runs_kv_quality_before_final_artifact_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    calls: list[tuple[str, ...]] = []
    quality = {
        "schema": "mtplx-hy3-kv-quality-v1",
        "identity": _quality_identity(),
        "acceptance": {"passed": True, "gates": {"retrieval": True}},
    }

    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        calls.append(command)
        if command == ("probe",):
            return _probe_result()
        if command == ("quality",):
            return quality
        if command == ("verify-artifact",):
            return _artifact_attestation()
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    monkeypatch.setattr(
        hy3_kv_quality,
        "validate_quality_result",
        lambda value: value["acceptance"],
    )
    result = runner_module.run_subprocess_campaign(
        probe_command=("probe",),
        quality_command=("quality",),
        artifact_verify_command=("verify-artifact",),
        arm_command_template=("arm", "{arm}", "{context_tokens}", "{repetition}"),
        repetitions=2,
        command_runner=run,
        bootstrap_resamples=100,
    )

    assert calls[-2:] == [("quality",), ("verify-artifact",)]
    assert result["kv_quality"] == quality
    assert result["acceptance"]["quality_gate_status"] == "passed"


def test_subprocess_campaign_wires_outer_deadline_to_every_actual_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    calls: list[tuple[tuple[str, ...], float, float]] = []
    quality = {
        "schema": "mtplx-hy3-kv-quality-v1",
        "identity": _quality_identity(),
        "acceptance": {"passed": True, "gates": {"retrieval": True}},
    }

    def run_json(command, *, timeout_seconds, termination_grace_seconds, **_kwargs):
        actual = tuple(command)
        calls.append((actual, timeout_seconds, termination_grace_seconds))
        if actual == ("probe",):
            return _probe_result()
        if actual == ("verify-artifact",):
            return _artifact_attestation()
        _, arm, context_tokens, repetition = actual
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    def run_quality(
        command,
        *,
        timeout_seconds,
        termination_grace_seconds,
        **_kwargs,
    ):
        calls.append((tuple(command), timeout_seconds, termination_grace_seconds))
        return quality, 0

    monkeypatch.setattr(runner_module, "run_json_subprocess", run_json)
    monkeypatch.setattr(runner_module, "_run_json_subprocess", run_quality)
    monkeypatch.setattr(
        hy3_kv_quality,
        "validate_quality_result",
        lambda value: value["acceptance"],
    )

    runner_module.run_subprocess_campaign(
        probe_command=("probe",),
        quality_command=("quality",),
        artifact_verify_command=("verify-artifact",),
        arm_command_template=("arm", "{arm}", "{context_tokens}", "{repetition}"),
        repetitions=2,
        bootstrap_resamples=100,
        subprocess_timeout_seconds=123.0,
        subprocess_termination_grace_seconds=4.0,
    )

    assert calls[0][0] == ("probe",)
    assert calls[-2][0] == ("quality",)
    assert calls[-1][0] == ("verify-artifact",)
    assert len(calls) == 1 + len(CONTEXT_MATRIX_TOKENS) * 2 * 2 + 2
    assert all(timeout == 123.0 for _command, timeout, _grace in calls)
    assert all(grace == 4.0 for _command, _timeout, grace in calls)


def test_subprocess_campaign_retains_quality_rejection_and_still_hashes_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    calls: list[tuple[str, ...]] = []
    quality = {
        "schema": "mtplx-hy3-kv-quality-v1",
        "identity": _quality_identity(),
        "acceptance": {
            "passed": False,
            "rejection_reasons": ["128K earliest marker retrieval failed"],
        },
    }

    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        calls.append(command)
        if command == ("probe",):
            return _probe_result()
        if command == ("quality",):
            return quality
        if command == ("verify-artifact",):
            return _artifact_attestation()
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    monkeypatch.setattr(
        hy3_kv_quality,
        "validate_quality_result",
        lambda value: value["acceptance"],
    )
    result = runner_module.run_subprocess_campaign(
        probe_command=("probe",),
        quality_command=("quality",),
        artifact_verify_command=("verify-artifact",),
        arm_command_template=("arm", "{arm}", "{context_tokens}", "{repetition}"),
        repetitions=2,
        command_runner=run,
        bootstrap_resamples=100,
    )

    assert calls[-2:] == [("quality",), ("verify-artifact",)]
    assert result["status"] == "rejected"
    assert result["acceptance"]["passed"] is False
    assert result["acceptance"]["quality_gate_status"] == "rejected"
    assert (
        "128K earliest marker retrieval failed"
        in result["acceptance"]["rejection_reasons"]
    )


def test_subprocess_campaign_rejects_quality_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    quality_identity = _quality_identity()
    quality_identity["source_git_commit"] = "f" * 40
    quality = {
        "schema": "mtplx-hy3-kv-quality-v1",
        "identity": quality_identity,
        "acceptance": {"passed": True, "gates": {"retrieval": True}},
    }

    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("quality",):
            return quality
        if command == ("verify-artifact",):
            return _artifact_attestation()
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    monkeypatch.setattr(
        hy3_kv_quality,
        "validate_quality_result",
        lambda value: value["acceptance"],
    )
    with pytest.raises(BenchmarkGateError, match="quality identity.*source_git_commit"):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            quality_command=("quality",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_json_subprocess_retains_a_structured_rejection_return_code() -> None:
    command = (
        sys.executable,
        "-c",
        (
            "import json,sys; "
            "print(json.dumps({'acceptance': {'passed': False}})); "
            "sys.exit(2)"
        ),
    )

    value, returncode = runner_module._run_json_subprocess(
        command,
        allowed_returncodes=(0, 2),
    )

    assert returncode == 2
    assert value == {"acceptance": {"passed": False}}
    with pytest.raises(BenchmarkGateError, match=r"failed \(2\)"):
        runner_module.run_json_subprocess(command)


def test_json_subprocess_delivers_the_input_payload_on_stdin() -> None:
    command = (
        sys.executable,
        "-c",
        (
            "import json,sys; "
            "value=json.load(sys.stdin); "
            "print(json.dumps({'received': value}))"
        ),
    )
    payload = {"loaded": False, "models": []}

    result = runner_module.run_json_subprocess(command, input_payload=payload)

    assert result == {"received": payload}


def test_json_subprocess_timeout_terminates_and_reaps_its_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("hung-arm", "--context-tokens", "131072")
    communicate_calls: list[tuple[str | None, float | None]] = []
    termination_signals: list[tuple[int, int]] = []
    popen_kwargs: dict[str, object] = {}

    class HungProcess:
        pid = 2468
        returncode = None

        def communicate(self, input=None, timeout=None):
            communicate_calls.append((input, timeout))
            if len(communicate_calls) == 1:
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = -signal.SIGTERM
            return ("", "terminated during cleanup")

    def fake_popen(argv, **kwargs):
        assert argv == command
        popen_kwargs.update(kwargs)
        return HungProcess()

    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("timeout-safe path must use Popen"),
    )
    monkeypatch.setattr(runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda pgid, signum: termination_signals.append((pgid, signum)),
    )
    monkeypatch.setattr(
        runner_module,
        "_process_group_exists",
        lambda _pgid: False,
    )

    with pytest.raises(BenchmarkGateError, match=r"timed out after 0\.25 seconds"):
        runner_module._run_json_subprocess(
            command,
            timeout_seconds=0.25,
            termination_grace_seconds=0.1,
        )

    assert popen_kwargs["start_new_session"] is True
    assert termination_signals == [(2468, signal.SIGTERM)]
    assert communicate_calls == [(None, 0.25), (None, 0.1)]


def test_json_subprocess_timeout_escalates_stubborn_child_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("stubborn-probe",)
    communicate_timeouts: list[float | None] = []
    termination_signals: list[tuple[int, int]] = []

    class StubbornProcess:
        pid = 8642
        returncode = None

        def communicate(self, input=None, timeout=None):
            del input
            communicate_timeouts.append(timeout)
            if len(communicate_timeouts) < 3:
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = -signal.SIGKILL
            return ("", "killed after grace period")

    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("timeout-safe path must use Popen"),
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: StubbornProcess(),
    )
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda pgid, signum: termination_signals.append((pgid, signum)),
    )
    monkeypatch.setattr(
        runner_module,
        "_process_group_exists",
        lambda _pgid: False,
    )
    monkeypatch.setattr(
        runner_module,
        "_wait_for_process_group_exit",
        lambda _pgid, *, timeout_seconds: timeout_seconds == 0.05,
    )

    with pytest.raises(BenchmarkGateError, match="stubborn-probe"):
        runner_module._run_json_subprocess(
            command,
            timeout_seconds=0.2,
            termination_grace_seconds=0.05,
        )

    assert termination_signals == [
        (8642, signal.SIGTERM),
        (8642, signal.SIGKILL),
    ]
    assert communicate_timeouts == [0.2, 0.05, 0.05]


def test_json_subprocess_interrupt_cleans_process_group_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("interrupted-quality",)
    termination_signals: list[tuple[int, int]] = []

    class InterruptedProcess:
        pid = 9753
        returncode = None

        def communicate(self, input=None, timeout=None):
            del input, timeout
            if self.returncode is None:
                self.returncode = -signal.SIGTERM
                raise KeyboardInterrupt
            return ("", "interrupted")

    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: InterruptedProcess(),
    )
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda pgid, signum: termination_signals.append((pgid, signum)),
    )
    monkeypatch.setattr(
        runner_module,
        "_process_group_exists",
        lambda _pgid: False,
        raising=False,
    )

    with pytest.raises(KeyboardInterrupt):
        runner_module._run_json_subprocess(
            command,
            timeout_seconds=1.0,
            termination_grace_seconds=0.1,
        )

    assert termination_signals == [(9753, signal.SIGTERM)]


def test_json_subprocess_masks_repeated_termination_while_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("interrupted-arm",)
    cleanup_signal_masked = False
    mask_events: list[str] = []

    class InterruptedProcess:
        pid = 9864
        returncode = None

        @staticmethod
        def communicate(input=None, timeout=None):
            del input, timeout
            raise SystemExit(128 + signal.SIGTERM)

    @contextmanager
    def blocked_termination_signals():
        nonlocal cleanup_signal_masked
        mask_events.append("block")
        cleanup_signal_masked = True
        try:
            yield
        finally:
            cleanup_signal_masked = False
            mask_events.append("unblock")

    def terminate_group(process, *, grace_seconds):
        del process, grace_seconds
        assert cleanup_signal_masked is True
        mask_events.append("cleanup")
        return "", ""

    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: InterruptedProcess(),
    )
    monkeypatch.setattr(
        runner_module,
        "_blocked_termination_signals",
        blocked_termination_signals,
    )
    monkeypatch.setattr(
        runner_module,
        "_terminate_json_subprocess_group",
        terminate_group,
    )

    with pytest.raises(SystemExit) as raised:
        runner_module._run_json_subprocess(command)

    assert raised.value.code == 128 + signal.SIGTERM
    assert mask_events == ["block", "cleanup", "unblock"]


def test_qwen_isolation_retains_lane_when_subprocess_group_cleanup_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("unkillable-arm",)
    calls: list[object] = []
    captured = {"loaded": True, "models": ["qwen"]}

    class UnkillableProcess:
        pid = 9975
        returncode = None

        @staticmethod
        def communicate(input=None, timeout=None):
            del input
            raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: UnkillableProcess(),
    )
    monkeypatch.setattr(runner_module, "_signal_process_group", lambda *_args: None)
    monkeypatch.setattr(runner_module, "_process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        runner_module,
        "_wait_for_process_group_exit",
        lambda _pgid, *, timeout_seconds: False,
    )
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or captured,
        unload=lambda state: calls.append(("unload", state)),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    with pytest.raises(BenchmarkGateError, match="survived SIGKILL"):
        run_exclusive_hardware_window(
            lambda: runner_module._run_json_subprocess(
                command,
                timeout_seconds=0.1,
                termination_grace_seconds=0.01,
            ),
            hooks=hooks,
        )

    assert calls == ["acquire", "capture", ("unload", captured)]


def test_json_subprocess_rejects_descendants_after_clean_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("leader-exits-before-child",)
    termination_signals: list[tuple[int, int]] = []
    group_waits: list[float] = []
    group_exit_results = iter((False, True))

    class ExitedLeader:
        pid = 7531
        returncode = 0

        @staticmethod
        def communicate(input=None, timeout=None):
            del input, timeout
            return ('{"ok": true}', "")

    monkeypatch.setattr(
        runner_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: ExitedLeader(),
    )
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda pgid, signum: termination_signals.append((pgid, signum)),
    )
    monkeypatch.setattr(
        runner_module,
        "_process_group_exists",
        lambda _pgid: True,
        raising=False,
    )

    def wait_for_group(_pgid, *, timeout_seconds):
        group_waits.append(timeout_seconds)
        return next(group_exit_results)

    monkeypatch.setattr(
        runner_module,
        "_wait_for_process_group_exit",
        wait_for_group,
        raising=False,
    )

    with pytest.raises(BenchmarkGateError, match="descendants remained"):
        runner_module._run_json_subprocess(
            command,
            timeout_seconds=1.0,
            termination_grace_seconds=0.1,
        )

    assert termination_signals == [
        (7531, signal.SIGTERM),
        (7531, signal.SIGKILL),
    ]
    assert group_waits == [0.1, 0.1]


@pytest.mark.parametrize(("returncode", "passed"), ((0, False), (2, True)))
def test_quality_subprocess_return_code_must_match_recomputed_acceptance(
    returncode: int,
    passed: bool,
) -> None:
    with pytest.raises(BenchmarkGateError, match="return code.*acceptance"):
        runner_module._require_quality_returncode(returncode, passed=passed)


@pytest.mark.parametrize(
    "field",
    (
        "model_artifact_sha256",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "resident_payload_sha256",
    ),
)
def test_subprocess_campaign_rejects_post_campaign_artifact_drift(field: str) -> None:
    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            return _artifact_attestation(**{field: "f" * 64})
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    with pytest.raises(BenchmarkGateError, match=field):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_subprocess_campaign_requires_a_full_post_campaign_payload_hash() -> None:
    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            return _artifact_attestation(payload_hash_verified=False)
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    with pytest.raises(BenchmarkGateError, match="payload_hash_verified"):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_subprocess_campaign_requires_f_nocache_payload_verification() -> None:
    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            return _artifact_attestation(payload_hash_io_mode="buffered")
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    with pytest.raises(BenchmarkGateError, match="f-nocache"):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_subprocess_campaign_rejects_post_campaign_resident_byte_drift() -> None:
    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            return _artifact_attestation(resident_payload_bytes=4)
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    with pytest.raises(BenchmarkGateError, match="resident_payload_bytes"):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_subprocess_campaign_recomputes_post_campaign_artifact_stat_hash() -> None:
    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            attestation = _artifact_attestation()
            attestation["sidecar_fingerprint"] = {
                **_ARTIFACT_FINGERPRINT,
                "size": 99,
            }
            return attestation
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    with pytest.raises(BenchmarkGateError, match="artifact_stat_sha256"):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_subprocess_campaign_recomputes_resident_fingerprint_stat_hash() -> None:
    def run(command: tuple[str, ...]) -> Mapping[str, object]:
        if command == ("probe",):
            return _probe_result()
        if command == ("verify-artifact",):
            attestation = _artifact_attestation()
            fingerprints = attestation["resident_shard_fingerprints"]
            assert isinstance(fingerprints, list)
            fingerprints[0] = {**fingerprints[0], "size": 99}
            return attestation
        _, arm, context_tokens, repetition = command
        return _observation(
            arm,
            int(context_tokens),
            int(repetition),
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )

    with pytest.raises(BenchmarkGateError, match="artifact_stat_sha256"):
        runner_module.run_subprocess_campaign(
            probe_command=("probe",),
            artifact_verify_command=("verify-artifact",),
            arm_command_template=(
                "arm",
                "{arm}",
                "{context_tokens}",
                "{repetition}",
            ),
            repetitions=2,
            command_runner=run,
            bootstrap_resamples=100,
        )


def test_campaign_fails_before_arm_execution_when_probe_gate_did_not_pass() -> None:
    calls: list[object] = []
    with pytest.raises(BenchmarkGateError, match="allocator-release probe"):
        run_balanced_campaign(
            allocator_probe={"gate_passed": False},
            execute_arm=lambda *args: calls.append(args),
            repetitions=2,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("model_artifact_id", "pipenetwork/Hy3-4bit@different"),
        ("model_artifact_sha256", "d" * 64),
        ("expert_manifest_id", "hy3-q4/component-banks/other-manifest.json"),
        ("expert_manifest_sha256", "e" * 64),
        ("source_git_commit", "d" * 40),
    ],
)
def test_campaign_rejects_every_arm_whose_identity_differs_from_probe(
    field: str,
    different_value: str,
) -> None:
    probe = validate_allocator_probe(_probe_result())
    calls: list[tuple[str, int, int]] = []

    def execute(arm: str, context_tokens: int, repetition: int):
        calls.append((arm, context_tokens, repetition))
        row = _observation(arm, context_tokens, repetition, tok_s=10.0)
        row["identity"][field] = different_value
        if field == "expert_manifest_sha256":
            row["expert_route_binding"]["expert_manifest_sha256"] = different_value
            row["metrics"]["hold_warmup_sample"]["expert_route_binding"][
                "expert_manifest_sha256"
            ] = different_value
            for sample in row["metrics"]["performance_samples"]:
                sample["expert_route_binding"]["expert_manifest_sha256"] = (
                    different_value
                )
        return row

    with pytest.raises(
        BenchmarkGateError,
        match=f"allocator probe identity drifted at {field}",
    ):
        run_balanced_campaign(
            allocator_probe=probe,
            execute_arm=execute,
            repetitions=2,
        )

    assert calls == [("static", 4096, 0)]


def test_campaign_rejects_pair_identity_or_output_drift() -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int):
        row = _observation(arm, context_tokens, repetition, tok_s=10.0)
        if arm == "dynamic" and context_tokens == 4096 and repetition == 0:
            row["generated_token_ids"] = [999]
            row["generated_token_sha256"] = canonical_sha256([999])
            row["metrics"]["generated_tokens"] = 1
            row["metrics"]["elapsed_seconds"] = 0.1
        return row

    with pytest.raises(BenchmarkGateError, match="paired generated tokens"):
        run_balanced_campaign(
            allocator_probe=probe,
            execute_arm=execute,
            repetitions=2,
        )


@pytest.mark.parametrize(
    ("field", "error_pattern"),
    (
        ("generated_token_ids", r"paired.*tokens"),
        ("route_trace", r"paired.*routes"),
        ("expert_hashes", r"paired hold expert hashes"),
    ),
)
def test_campaign_rejects_paired_hold_workload_drift(
    field: str,
    error_pattern: str,
) -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int):
        row = _observation(
            arm,
            context_tokens,
            repetition,
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )
        if arm == "dynamic" and context_tokens == 4096 and repetition == 0:
            sample = row["metrics"]["performance_samples"][1]
            if field == "generated_token_ids":
                sample[field] = [900, 901, 902, 903, 904, 905, 906, 907]
                sample["generated_token_sha256"] = canonical_sha256(sample[field])
            elif field == "route_trace":
                sample[field] = [
                    {
                        "phase": "ar_decode",
                        "layer": 70,
                        "expert_ids": [2, 5, 13],
                    }
                ]
                sample["route_trace_sha256"] = canonical_sha256(sample[field])
                sample["expert_hashes"] = {
                    "70:2": "2" * 64,
                    "70:5": "3" * 64,
                    "70:13": "4" * 64,
                }
                sample["expert_route_binding"] = _expert_route_binding(
                    manifest_sha256="b" * 64,
                    route_trace=sample[field],
                    expert_hashes=sample["expert_hashes"],
                )
            else:
                sample[field]["2:3"] = "9" * 64
                sample["expert_route_binding"] = _expert_route_binding(
                    manifest_sha256="b" * 64,
                    route_trace=sample["route_trace"],
                    expert_hashes=sample[field],
                )
        return row

    with pytest.raises(BenchmarkGateError, match=error_pattern):
        run_balanced_campaign(
            allocator_probe=probe,
            execute_arm=execute,
            repetitions=2,
            bootstrap_resamples=100,
        )


@pytest.mark.parametrize(
    ("field", "error_pattern"),
    (
        ("generated_token_ids", r"paired warm-up tokens"),
        ("route_trace", r"paired warm-up routes"),
        ("expert_hashes", r"paired warm-up expert hashes"),
    ),
)
def test_campaign_rejects_paired_hold_warmup_workload_drift(
    field: str,
    error_pattern: str,
) -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int):
        row = _observation(
            arm,
            context_tokens,
            repetition,
            tok_s=11.0 if arm == "dynamic" else 10.0,
        )
        if arm == "dynamic" and context_tokens == 4096 and repetition == 0:
            sample = row["metrics"]["hold_warmup_sample"]
            if field == "generated_token_ids":
                sample[field] = list(range(900, 908))
                sample["generated_token_sha256"] = canonical_sha256(sample[field])
            elif field == "route_trace":
                sample[field] = [
                    {
                        "phase": "ar_decode",
                        "layer": 70,
                        "expert_ids": [2, 5, 13],
                    }
                ]
                sample["route_trace_sha256"] = canonical_sha256(sample[field])
                sample["expert_hashes"] = {
                    "70:2": "2" * 64,
                    "70:5": "3" * 64,
                    "70:13": "4" * 64,
                }
            else:
                sample[field]["0:3"] = "9" * 64
            sample["expert_route_binding"] = _expert_route_binding(
                manifest_sha256="b" * 64,
                route_trace=sample["route_trace"],
                expert_hashes=sample["expert_hashes"],
            )
        return row

    with pytest.raises(BenchmarkGateError, match=error_pattern):
        run_balanced_campaign(
            allocator_probe=probe,
            execute_arm=execute,
            repetitions=2,
            bootstrap_resamples=100,
        )


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("model_artifact_id", "pipenetwork/Hy3-4bit@different"),
        ("expert_manifest_id", "hy3-q4/component-banks/other-manifest.json"),
    ],
)
def test_campaign_rejects_arm_artifact_ids_even_when_hashes_match(
    field: str,
    different_value: str,
) -> None:
    probe = validate_allocator_probe(_probe_result())

    def execute(arm: str, context_tokens: int, repetition: int):
        row = _observation(arm, context_tokens, repetition, tok_s=10.0)
        if arm == "dynamic" and context_tokens == 4096 and repetition == 0:
            row["identity"][field] = different_value
        return row

    with pytest.raises(
        BenchmarkGateError,
        match=f"allocator probe identity drifted at {field}",
    ):
        run_balanced_campaign(
            allocator_probe=probe,
            execute_arm=execute,
            repetitions=2,
        )


def test_qwen_isolation_restores_and_releases_lane_on_workload_exception() -> None:
    calls: list[object] = []
    captured = {"loaded": True, "models": ["qwen"]}
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or captured,
        unload=lambda state: calls.append(("unload", state)),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    with pytest.raises(RuntimeError, match="workload failed"):
        run_exclusive_hardware_window(
            lambda: (_ for _ in ()).throw(RuntimeError("workload failed")),
            hooks=hooks,
        )

    assert calls == [
        "acquire",
        "capture",
        ("unload", captured),
        ("restore", captured),
        ("verify", captured),
        "release",
    ]


def test_qwen_isolation_restores_even_when_unload_raises() -> None:
    calls: list[object] = []
    captured = {"loaded": True, "models": ["qwen"]}
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or captured,
        unload=lambda state: (
            calls.append(("unload", state))
            or (_ for _ in ()).throw(RuntimeError("unload failed"))
        ),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    with pytest.raises(RuntimeError, match="unload failed"):
        run_exclusive_hardware_window(lambda: calls.append("workload"), hooks=hooks)

    assert "workload" not in calls
    assert calls[-3:] == [("restore", captured), ("verify", captured), "release"]


@pytest.mark.parametrize("failure", ("restore", "verify"))
def test_qwen_isolation_retains_lane_when_restoration_is_not_proven(
    failure: str,
) -> None:
    calls: list[object] = []
    captured = {"loaded": True, "models": ["qwen"]}

    def restore(state: object) -> None:
        calls.append(("restore", state))
        if failure == "restore":
            raise RuntimeError("restore failed")

    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or captured,
        unload=lambda state: calls.append(("unload", state)),
        restore=restore,
        verify_restored=lambda state: (
            calls.append(("verify", state)) or failure != "verify"
        ),
    )

    with pytest.raises(
        (RuntimeError, BenchmarkGateError), match="restore|captured state"
    ):
        run_exclusive_hardware_window(lambda: calls.append("workload"), hooks=hooks)

    assert "release" not in calls


def test_qwen_isolation_masks_termination_during_lane_handoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    previous = {runner_module.signal.SIGUSR1}

    def mask(how: int, signals: object) -> set[object]:
        calls.append(("mask", how, signals))
        return previous

    monkeypatch.setattr(runner_module.signal, "pthread_sigmask", mask)
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or {"loaded": False, "models": []},
        unload=lambda state: calls.append(("unload", state)),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    run_exclusive_hardware_window(lambda: calls.append("workload"), hooks=hooks)

    blocked = {
        runner_module.signal.SIGINT,
        runner_module.signal.SIGTERM,
        runner_module.signal.SIGHUP,
    }
    assert calls == [
        ("mask", runner_module.signal.SIG_BLOCK, blocked),
        "acquire",
        ("mask", runner_module.signal.SIG_SETMASK, previous),
        "capture",
        ("unload", {"loaded": False, "models": []}),
        "workload",
        ("mask", runner_module.signal.SIG_BLOCK, blocked),
        ("restore", {"loaded": False, "models": []}),
        ("verify", {"loaded": False, "models": []}),
        "release",
        ("mask", runner_module.signal.SIG_SETMASK, previous),
    ]


def test_qwen_isolation_turns_sigterm_into_cleanup_before_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    handlers: dict[int, object] = {}

    def install(signum: int, handler: object) -> object:
        previous = handlers.get(signum, runner_module.signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(runner_module.signal, "signal", install)
    hooks = QwenIsolationHooks(
        acquire_lane=lambda: calls.append("acquire"),
        release_lane=lambda: calls.append("release"),
        capture=lambda: calls.append("capture") or {"loaded": True, "models": ["qwen"]},
        unload=lambda state: calls.append(("unload", state)),
        restore=lambda state: calls.append(("restore", state)),
        verify_restored=lambda state: calls.append(("verify", state)) or True,
    )

    def terminate() -> None:
        handler = handlers[runner_module.signal.SIGTERM]
        assert callable(handler)
        handler(runner_module.signal.SIGTERM, None)

    with pytest.raises(SystemExit) as raised:
        run_exclusive_hardware_window(terminate, hooks=hooks)

    assert raised.value.code == 128 + runner_module.signal.SIGTERM
    assert calls[-3:] == [
        ("restore", {"loaded": True, "models": ["qwen"]}),
        ("verify", {"loaded": True, "models": ["qwen"]}),
        "release",
    ]


def _tracked_cli_plan_repo(
    tmp_path: Path,
    *,
    decoy_probe: bool = False,
) -> tuple[Path, Path]:
    repo = tmp_path / "plan-repo"
    repo.mkdir()
    sources = (
        "benchmarks/probe.py",
        "benchmarks/quality.py",
        "benchmarks/arm.py",
        "benchmarks/qwen.py",
    )
    for relative in sources:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen campaign source\n", encoding="utf-8")
    hooks_path = repo / "benchmarks/hooks.json"
    hooks_path.write_text('{"frozen":true}\n', encoding="utf-8")
    probe_command = [sys.executable, "benchmarks/probe.py"]
    if decoy_probe:
        probe_command.append("decoy.py")
    probe_command.extend(("--hooks-config", "benchmarks/hooks.json", "--json"))
    common_hooks = ["--hooks-config", "benchmarks/hooks.json"]
    spec = {
        "legacy_exclusive_lane_lock": "/tmp/mtplx-gpu-exclusive.lock",
        "legacy_exclusive_lane_wait_seconds": 0,
        "artifact_verify_command": [
            sys.executable,
            "benchmarks/probe.py",
            *common_hooks,
            "--verify-artifact-only",
        ],
        "probe_command": probe_command,
        "quality_command": [
            sys.executable,
            "benchmarks/quality.py",
            *common_hooks,
        ],
        "arm_command_template": [
            sys.executable,
            "benchmarks/arm.py",
            *common_hooks,
            "--arm",
            "{arm}",
            "--context",
            "{context_tokens}",
            "--repetition",
            "{repetition}",
        ],
        "repetitions": 2,
        "qwen": {
            field: [sys.executable, "benchmarks/qwen.py", field]
            for field in (
                "acquire_lane_command",
                "release_lane_command",
                "capture_command",
                "unload_command",
                "restore_command",
                "verify_command",
            )
        },
    }
    spec_path = repo / "benchmarks/spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.name", "Issue 46 Test"), cwd=repo, check=True
    )
    subprocess.run(
        ("git", "config", "user.email", "issue46@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(("git", "add", "."), cwd=repo, check=True)
    subprocess.run(
        ("git", "-c", "commit.gpgsign=false", "commit", "-qm", "frozen plan"),
        cwd=repo,
        check=True,
    )
    return repo, spec_path


def test_cli_plan_declares_exact_matrix_balanced_order_and_qwen_hooks(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "benchmark_hy3_dynamic_memory.py"
    )
    repo, spec_path = _tracked_cli_plan_repo(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--spec",
            str(spec_path),
            "--cwd",
            str(repo),
            "--plan-only",
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["context_matrix_tokens"] == list(CONTEXT_MATRIX_TOKENS)
    assert plan["qwen_isolation_configured"] is True
    assert plan["legacy_exclusive_lane_lock"] == "/tmp/mtplx-gpu-exclusive.lock"
    assert plan["legacy_exclusive_lane_wait_seconds"] == 0
    assert plan["quality_command"][-2:] == [
        "--hooks-config",
        "benchmarks/hooks.json",
    ]
    assert [row["arm"] for row in plan["schedule"][:4]] == [
        "static",
        "dynamic",
        "dynamic",
        "static",
    ]
    assert plan["execution_mode"] == "foreground-synchronous"
    assert plan["background_workloads"] is False


def test_cli_plan_selects_4k_before_any_hardware_or_lock_work(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "benchmark_hy3_dynamic_memory.py"
    )
    repo, spec_path = _tracked_cli_plan_repo(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--spec",
            str(spec_path),
            "--cwd",
            str(repo),
            "--contexts",
            "4096",
            "--plan-only",
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["context_matrix_tokens"] == [4_096]
    assert len(plan["schedule"]) == 4


def test_cli_plan_rejects_a_decoy_command_shape(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "benchmark_hy3_dynamic_memory.py"
    )
    repo, spec_path = _tracked_cli_plan_repo(tmp_path, decoy_probe=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--spec",
            str(spec_path),
            "--cwd",
            str(repo),
            "--plan-only",
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "exactly one tracked Python script" in completed.stderr
