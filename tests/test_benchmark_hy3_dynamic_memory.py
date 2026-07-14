from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
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
    ProbeSlab,
    QwenIsolationHooks,
    balanced_campaign_schedule,
    canonical_sha256,
    run_allocator_release_probe,
    run_balanced_campaign,
    run_exclusive_hardware_window,
    validate_allocator_probe,
    validate_campaign_observation,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity(arm: str = "dynamic") -> dict[str, object]:
    arm_config = {
        "dynamic_memory": arm == "dynamic",
        "context_window": 131_072,
        "kv_quantization": "q4",
        "max_active_sequences": 1,
        "expert_streaming_config": {"expert_slab_slots": 32},
    }
    normalized_config = {
        key: value for key, value in arm_config.items() if key != "dynamic_memory"
    }
    return {
        "model_key": "hy3-q4",
        "model_artifact_id": "pipenetwork/Hy3-4bit@160619d3",
        "model_artifact_sha256": "a" * 64,
        "expert_manifest_id": "hy3-q4/component-banks/manifest.json",
        "expert_manifest_sha256": "b" * 64,
        "source_git_commit": "c" * 40,
        "arm_config": arm_config,
        "arm_config_sha256": _sha(arm_config),
        "normalized_config": normalized_config,
        "normalized_config_sha256": _sha(normalized_config),
        "kv_quantization": "q4",
        "kv_block_size_tokens": 16,
        "total_context_tokens": 131_072,
    }


def _production_lane_identity(arm: str) -> dict[str, object]:
    identity = _identity(arm)
    identity["arm_config"] = {
        "dynamic_memory": arm == "dynamic",
        "expert_streaming_config": {
            "model_key": "hy3-q4",
            "memory_limit_bytes": 110 * 1024**3,
            "max_live_kv_tokens": 131_072,
            "runtime_reserve_bytes": 8 * 1024**3,
            "cache_policy": "lru",
            "cache_scope": "global",
            "slot_layout": "component-banks",
            "dynamic_expert_slabs": arm == "dynamic",
            "expert_slab_slots": 32,
        },
        "planned_persistent_slots": 9_792 if arm == "dynamic" else 5_771,
        "probe_slab_ids": [0, 1],
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
            }
        )
    return result


def _point(
    phase: str,
    timestamp_ns: int,
    *,
    expert: int,
    kv_blocks: int,
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
    return {
        "phase": phase,
        "monotonic_ns": timestamp_ns,
        "allocator_active_bytes": active_bytes,
        "allocator_cache_bytes": cache,
        "allocator_peak_bytes": expert + kv + cache,
        "expert_slab_physical_bytes": expert,
        "kv_physical_bytes": kv,
        "kv_allocated_blocks": kv_blocks,
        "slot_health": health,
        "slot_health_sha256": _sha(health),
        "operating_target_bytes": 110 * 1024**3,
        "hard_ceiling_bytes": 112 * 1024**3,
        "charged_bytes": expert + kv + allocator_cache_charged_bytes,
        "resident_model_bytes": 0,
        "kv_representation": "q4",
        "kv_logical_tokens": kv_blocks * 16,
        "expert_logical_records": expert,
        "expert_active_records": expert,
        "expert_resident_records": expert,
        "expert_logical_slabs": expert,
        "expert_active_slabs": expert,
        "expert_draining_slabs": 0,
        "expert_released_slabs": 0,
        "pinned_expert_bytes": 0,
        "inflight_expert_bytes": 0,
        "speculative_expert_bytes": 0,
        "runtime_workspace_bytes": 0,
        "inflight_expert_staging_bytes": 0,
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
        "allocator_cache_charged_bytes": allocator_cache_charged_bytes,
        "process_rss_bytes": active_bytes,
        "process_compressed_bytes": 0,
        "system_swap_delta_bytes": 0,
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
    routes = [[0, 3, 7], [1, 2, 9]]
    if arm == "dynamic":
        final_blocks = context_tokens // 16
        timeline = [
            _point("pre_growth", 1, expert=800, kv_blocks=1),
            _point("post_expert_reclaim", 2, expert=600, kv_blocks=1),
            _point("post_kv_growth", 3, expert=600, kv_blocks=final_blocks),
            _point("hold", 4_000_000_000, expert=600, kv_blocks=final_blocks),
            _point("hold", 4_500_000_000, expert=600, kv_blocks=final_blocks),
            _point("hold", 5_000_000_000, expert=600, kv_blocks=final_blocks),
            _point("post_reset", 6_000_000_000, expert=600, kv_blocks=0),
            _point("post_regrow", 7_000_000_000, expert=800, kv_blocks=0),
        ]
        start = {
            "kind": "empty-q4",
            "kv_physical_bytes": HY3_Q4_KV_BLOCK_BYTES,
            "kv_blocks": 1,
        }
    else:
        timeline = [
            _point("pre_growth", 1, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
            _point("post_kv_growth", 2, expert=600, kv_blocks=HY3_Q4_MAX_BLOCKS),
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
        "expert_hashes": {"0:3": "2" * 64, "1:2": "3" * 64},
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
            "performance_samples": _performance_samples(tok_s),
        },
    }


def _probe_result(*, released: int = 128, untouched_executable: bool = True):
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
        "untouched_slab_executable": untouched_executable,
        "error": None,
    }


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


def test_observation_requires_declared_start_and_physical_ordering() -> None:
    row = _observation("dynamic", 4096, 0, tok_s=12.0)

    validated = validate_campaign_observation(row)

    assert isinstance(validated.cache_start_state, CacheStartState)
    assert validated.context_tokens == 4096

    reordered = _observation("dynamic", 4096, 0, tok_s=12.0)
    reordered["timeline"][1]["kv_allocated_blocks"] = 2
    reordered["timeline"][1]["kv_physical_bytes"] = 2 * HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["allocator_active_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["allocator_peak_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    reordered["timeline"][1]["charged_bytes"] += HY3_Q4_KV_BLOCK_BYTES
    with pytest.raises(BenchmarkGateError, match="before KV"):
        validate_campaign_observation(reordered)


def test_observation_requires_exact_issue46_q4_physical_geometry() -> None:
    assert HY3_Q4_KV_BYTES_PER_TOKEN == 84_480
    assert HY3_Q4_KV_BLOCK_BYTES == 16 * 84_480
    assert HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES == int(10.3125 * 1024**3)

    static = validate_campaign_observation(_observation("static", 4096, 0, tok_s=12.0))
    assert static.cache_start_state.kv_physical_bytes == int(10.3125 * 1024**3)

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


def test_observation_requires_stable_hold_reset_regrow_and_block_crossing() -> None:
    unstable = _observation("dynamic", 4096, 0, tok_s=12.0)
    unstable["timeline"][4]["allocator_cache_bytes"] = 1
    unstable["timeline"][4]["allocator_cache_charged_bytes"] = 1
    unstable["timeline"][4]["charged_bytes"] += 1
    with pytest.raises(BenchmarkGateError, match="stable hold"):
        validate_campaign_observation(unstable)

    no_reset = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_reset["timeline"] = no_reset["timeline"][:-2]
    with pytest.raises(BenchmarkGateError, match="post_reset"):
        validate_campaign_observation(no_reset)

    no_regrow = _observation("dynamic", 4096, 0, tok_s=12.0)
    no_regrow["timeline"] = no_regrow["timeline"][:-1]
    with pytest.raises(BenchmarkGateError, match="post_regrow"):
        validate_campaign_observation(no_regrow)

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
    no_boundary["timeline"][1]["kv_allocated_blocks"] = 255
    no_boundary["timeline"][1]["kv_physical_bytes"] = 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][1]["allocator_active_bytes"] = (
        600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][1]["allocator_peak_bytes"] = (
        600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    )
    no_boundary["timeline"][1]["charged_bytes"] = 600 + 255 * HY3_Q4_KV_BLOCK_BYTES
    no_boundary["timeline"][2]["kv_allocated_blocks"] = 256
    with pytest.raises(BenchmarkGateError, match="block boundaries"):
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
    assert static["arm_config"]["planned_persistent_slots"] == 5_771
    assert dynamic["arm_config"]["planned_persistent_slots"] == 9_792
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
    row["route_trace"][0].append(11)
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


def test_allocator_probe_requires_real_charged_release_and_untouched_execution() -> (
    None
):
    passed = validate_allocator_probe(_probe_result())
    assert passed["gate_passed"] is True
    assert passed["charged_release_bytes"] == 128

    with pytest.raises(BenchmarkGateError, match="charged allocator"):
        validate_allocator_probe(_probe_result(released=127))
    with pytest.raises(BenchmarkGateError, match="untouched slab"):
        validate_allocator_probe(_probe_result(untouched_executable=False))


def test_allocator_probe_samples_two_evaluated_slabs_and_releases_one() -> None:
    calls: list[object] = []
    samples = iter(
        (
            AllocatorSample(active_bytes=100, cache_bytes=20, peak_bytes=120),
            AllocatorSample(active_bytes=300, cache_bytes=20, peak_bytes=320),
            AllocatorSample(active_bytes=200, cache_bytes=20, peak_bytes=320),
        )
    )
    slabs = (
        ProbeSlab("slab-a", registered_physical_bytes=100),
        ProbeSlab("slab-b", registered_physical_bytes=100),
    )

    result = run_allocator_release_probe(
        identity=_identity(),
        allocate_slabs=lambda: calls.append("allocate") or slabs,
        evaluate_slabs=lambda actual: calls.append(("evaluate", actual)),
        sample_allocator=lambda: next(samples),
        release_slab=lambda slab: calls.append(("release", slab.slab_id)),
        execute_slab=lambda slab: calls.append(("execute", slab.slab_id)) or True,
    )

    assert result["gate_passed"] is True
    assert calls == [
        "allocate",
        ("evaluate", slabs),
        ("release", "slab-a"),
        ("execute", "slab-b"),
    ]


def test_allocator_probe_fails_closed_on_injected_errors() -> None:
    result = run_allocator_release_probe(
        identity=_identity(),
        allocate_slabs=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        evaluate_slabs=lambda _slabs: None,
        sample_allocator=lambda: AllocatorSample(0, 0, 0),
        release_slab=lambda _slab: None,
        execute_slab=lambda _slab: True,
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
            else:
                sample[field] = [
                    {
                        "phase": "ar_decode",
                        "layer": 70,
                        "expert_ids": [2, 5, 13],
                    }
                ]
                sample["route_trace_sha256"] = canonical_sha256(sample[field])
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


def test_cli_plan_declares_exact_matrix_balanced_order_and_qwen_hooks(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "benchmark_hy3_dynamic_memory.py"
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "probe_command": [sys.executable, "benchmarks/probe.py", "--json"],
                "arm_command_template": [
                    sys.executable,
                    "benchmarks/arm.py",
                    "--arm",
                    "{arm}",
                    "--context",
                    "{context_tokens}",
                    "--repetition",
                    "{repetition}",
                ],
                "repetitions": 2,
                "qwen": {
                    "acquire_lane_command": [sys.executable, "qwen.py", "acquire"],
                    "release_lane_command": [sys.executable, "qwen.py", "release"],
                    "capture_command": [sys.executable, "qwen.py", "capture"],
                    "unload_command": [sys.executable, "qwen.py", "unload"],
                    "restore_command": [sys.executable, "qwen.py", "restore"],
                    "verify_command": [sys.executable, "qwen.py", "verify"],
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--spec", str(spec_path), "--plan-only"],
        cwd=script.parent.parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["context_matrix_tokens"] == list(CONTEXT_MATRIX_TOKENS)
    assert plan["qwen_isolation_configured"] is True
    assert [row["arm"] for row in plan["schedule"][:4]] == [
        "static",
        "dynamic",
        "dynamic",
        "static",
    ]


def test_cli_plan_rejects_a_decoy_command_shape(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "benchmark_hy3_dynamic_memory.py"
    )
    spec = {
        "probe_command": [sys.executable, "probe.py", "decoy.py"],
        "arm_command_template": [sys.executable, "arm.py"],
        "repetitions": 2,
        "qwen": {
            field: [sys.executable, "qwen.py", field]
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
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script), "--spec", str(spec_path), "--plan-only"],
        cwd=script.parent.parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "exactly one tracked Python script" in completed.stderr
