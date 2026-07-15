from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import mtplx.benchmarks.hy3_dynamic_memory_observation as observation_module
from mtplx.runtime_options import HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV
from mtplx.benchmarks.hy3_dynamic_memory_observation import (
    ArmObservationError,
    ArmRequest,
    load_tracked_hooks,
    main,
    produce_arm_observation,
    require_clean_source,
)
from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_MAX_BLOCKS,
    canonical_sha256,
    format_arm_command,
    validate_campaign_observation,
)


SOURCE_COMMIT = "a" * 40


class FakeLane:
    def __init__(self, arm: str, calls: list[str]) -> None:
        self.arm = arm
        self.calls = calls
        self.expert_bytes = 800
        self.kv_blocks = HY3_Q4_MAX_BLOCKS if arm == "static" else 1
        self.logical_tokens = HY3_Q4_MAX_BLOCKS * 16 if arm == "static" else 1
        self.hold_tps: Iterator[float] = iter((15.9, 16.0, 16.1))
        self.hold_sample_index = 0
        self.prepared_context_tokens: int | None = None

    def identity(self) -> dict[str, object]:
        self.calls.append("identity")
        return {
            "model_key": "hy3-q4",
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
            "model_artifact_sha256": "1" * 64,
            "expert_manifest_id": "expert-manifest.json",
            "expert_manifest_sha256": "2" * 64,
            "artifact_pins_sha256": "5" * 64,
            "artifact_stat_sha256": "6" * 64,
            "resident_payload_bytes": 3,
            "resident_payload_sha256": "7" * 64,
            "source_git_commit": SOURCE_COMMIT,
            "arm_config": {
                "dynamic_memory": self.arm == "dynamic",
                "attention_runtime_env": dict(HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV),
                "context_window": 131_072,
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
                    "dynamic_expert_cache": self.arm == "dynamic",
                    "resource_telemetry": True,
                },
                "planned_persistent_slots": 8_673,
            },
            "kv_quantization": "q4",
            "kv_block_size_tokens": 16,
            "total_context_tokens": 131_072,
        }

    def _ledger(
        self,
        *,
        blocks: int,
        experts: int,
        logical_tokens: int | None = None,
    ) -> dict[str, object]:
        kv_bytes = blocks * HY3_Q4_KV_BLOCK_BYTES
        active_bytes = experts + kv_bytes
        allocator_cache_bytes = 7
        runtime_workspace_bytes = 64
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
            "allocator_active_bytes": active_bytes,
            "allocator_cache_bytes": allocator_cache_bytes,
            "allocator_peak_bytes": active_bytes,
            "expert_cache_physical_bytes": experts,
            "kv_physical_bytes": kv_bytes,
            "kv_allocated_blocks": blocks,
            "slot_health": {
                "active_routes": 0,
                "pins": 0,
                "loading": 0,
                "failed": 0,
                "integrity_errors": 0,
                "completion_fence_failures": 0,
                "global_device_synchronizations": 0,
            },
            "memory_limit_bytes": 100 * 1024**3,
            "allocator_headroom_bytes": 1024**3,
            "classified_limit_bytes": 99 * 1024**3,
            "classified_bytes": active_bytes + runtime_workspace_bytes,
            "charged_bytes": (
                active_bytes + runtime_workspace_bytes + allocator_cache_bytes
            ),
            "charged_residual_bytes": (
                100 * 1024**3
                - active_bytes
                - runtime_workspace_bytes
                - allocator_cache_bytes
            ),
            "resident_model_bytes": 0,
            "kv_representation": "q4",
            "kv_logical_tokens": (
                self.logical_tokens if logical_tokens is None else logical_tokens
            ),
            "expert_logical_records": experts,
            "expert_allocated_records": experts,
            "expert_active_records": experts,
            "expert_resident_records": experts,
            "record_allocations": 0,
            "record_reuses": 0,
            "record_evictions": 0,
            "record_releases": 0,
            "pinned_expert_bytes": 0,
            "inflight_expert_bytes": 0,
            "speculative_expert_bytes": 0,
            "runtime_workspace_bytes": runtime_workspace_bytes,
            "inflight_expert_staging_bytes": 0,
            "admission_failures": 0,
            "allocator_cache_charged_bytes": allocator_cache_bytes,
            "process_rss_bytes": active_bytes,
            "process_compressed_bytes": 0,
            "system_swap_delta_bytes": 0,
            "gpu_utilization_percent": 20.0,
            "python_control_cpu": python_control_cpu,
            "failed_closed": False,
            "failure_reason": None,
        }

    def physical_ledger(self) -> dict[str, object]:
        self.calls.append("ledger")
        return self._ledger(blocks=self.kv_blocks, experts=self.expert_bytes)

    def reclaim_experts_for_q4(self, context_tokens: int) -> None:
        self.calls.append(f"reclaim:{context_tokens}")
        self.expert_bytes = 600

    def prepare_q4_context(self, context_tokens: int) -> None:
        self.calls.append(f"prepare:{context_tokens}")
        self.prepared_context_tokens = context_tokens
        if self.arm == "dynamic":
            self.kv_blocks = context_tokens // 16
            self.logical_tokens = context_tokens

    def kv_growth_steps(self) -> list[dict[str, object]]:
        self.calls.append("kv_growth_steps")
        if self.arm == "static":
            return []
        assert self.prepared_context_tokens is not None
        target_blocks = self.prepared_context_tokens // 16

        first_before = self._ledger(blocks=1, experts=800, logical_tokens=1)
        first_gap = self._ledger(blocks=1, experts=600, logical_tokens=1)
        first_gap["captured_monotonic_ns"] = 20
        first_after = self._ledger(
            blocks=target_blocks - 1,
            experts=600,
            logical_tokens=1,
        )
        second_gap = dict(first_after)
        second_gap["captured_monotonic_ns"] = 50
        second_after = self._ledger(
            blocks=target_blocks,
            experts=600,
            logical_tokens=1,
        )
        return [
            {
                "sequence_index": 0,
                "requested_tokens": (target_blocks - 1) * 16,
                "target_blocks": target_blocks - 1,
                "before_monotonic_ns": 10,
                "reclaim_monotonic_ns": 20,
                "after_monotonic_ns": 30,
                "before": first_before,
                "reclaim_gap": first_gap,
                "after": first_after,
                "steady_delta_bytes": (target_blocks - 2) * HY3_Q4_KV_BLOCK_BYTES,
                "max_transient_delta_bytes": (target_blocks - 1)
                * (HY3_Q4_KV_BLOCK_BYTES // 80),
                "reclaimed_expert_bytes": 200,
                "kv_growth_bytes": (target_blocks - 2) * HY3_Q4_KV_BLOCK_BYTES,
            },
            {
                "sequence_index": 1,
                "requested_tokens": self.prepared_context_tokens,
                "target_blocks": target_blocks,
                "before_monotonic_ns": 40,
                "reclaim_monotonic_ns": 50,
                "after_monotonic_ns": 60,
                "before": first_after,
                "reclaim_gap": second_gap,
                "after": second_after,
                "steady_delta_bytes": HY3_Q4_KV_BLOCK_BYTES,
                "max_transient_delta_bytes": target_blocks
                * (HY3_Q4_KV_BLOCK_BYTES // 80),
                "reclaimed_expert_bytes": 0,
                "kv_growth_bytes": HY3_Q4_KV_BLOCK_BYTES,
            },
        ]

    def invoke_context(self, context_tokens: int) -> dict[str, object]:
        self.calls.append(f"invoke:{context_tokens}")
        return {
            "prompt_token_ids": [11, context_tokens, 12],
            "generated_token_ids": [101, 202, 303, 404],
            "route_trace": [
                {"phase": "ar_decode", "layer": 0, "expert_ids": [3, 7]},
                {"phase": "ar_decode", "layer": 1, "expert_ids": [2, 9]},
            ],
            "expert_hashes": {
                "0:3": "3" * 64,
                "0:7": "4" * 64,
                "1:2": "5" * 64,
                "1:9": "6" * 64,
            },
            "elapsed_seconds": 0.25,
        }

    def sample_hold_performance(self) -> dict[str, object]:
        self.calls.append("sample_hold")
        sample_index = self.hold_sample_index
        self.hold_sample_index += 1
        route_trace = [
            {
                "phase": "ar_decode",
                "layer": sample_index + 1,
                "expert_ids": [3, 7, 11],
            }
        ]
        return {
            "tokens_per_second": next(self.hold_tps),
            "expert_hit_rate": 0.75,
            "ssd_bytes_per_token": 1024.0,
            "p50_token_latency_ms": 62.0,
            "p95_token_latency_ms": 70.0,
            "generated_token_ids": [
                500 + sample_index * 10 + offset for offset in range(8)
            ],
            "route_trace": route_trace,
            "expert_hashes": {
                f"{sample_index + 1}:3": "3" * 64,
                f"{sample_index + 1}:7": "4" * 64,
                f"{sample_index + 1}:11": "5" * 64,
            },
        }

    def stabilize_hold(self) -> dict[str, object]:
        self.calls.append("stabilize_hold")
        route_trace = [
            {
                "phase": "ar_decode",
                "layer": 0,
                "expert_ids": [3, 7, 11],
            }
        ]
        return {
            "tokens_per_second": 15.8,
            "expert_hit_rate": 0.70,
            "ssd_bytes_per_token": 1280.0,
            "p50_token_latency_ms": 63.0,
            "p95_token_latency_ms": 71.0,
            "generated_token_ids": list(range(400, 408)),
            "route_trace": route_trace,
            "expert_hashes": {
                "0:3": "3" * 64,
                "0:7": "4" * 64,
                "0:11": "5" * 64,
            },
        }

    def reset_q4_context(self) -> None:
        self.calls.append("reset")
        self.kv_blocks = 0
        self.logical_tokens = 0

    def trigger_future_expert_demand(self) -> None:
        self.calls.append("future_demand")
        self.expert_bytes = 800

    def close(self) -> None:
        self.calls.append("close")


class FakeHooks:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def load_static_lane(self, request: ArmRequest) -> FakeLane:
        self.calls.append(f"load_static:{request.context_tokens}")
        return FakeLane("static", self.calls)

    def load_dynamic_lane(self, request: ArmRequest) -> FakeLane:
        self.calls.append(f"load_dynamic:{request.context_tokens}")
        return FakeLane("dynamic", self.calls)


class FixedHoldHooks(FakeHooks):
    hold_samples = 4


class BrokenQ4LedgerLane(FakeLane):
    def physical_ledger(self) -> dict[str, object]:
        result = super().physical_ledger()
        result["kv_physical_bytes"] = int(result["kv_physical_bytes"]) + 1
        return result


class BrokenQ4LedgerHooks(FakeHooks):
    def load_dynamic_lane(self, request: ArmRequest) -> FakeLane:
        self.calls.append(f"load_dynamic:{request.context_tokens}")
        return BrokenQ4LedgerLane("dynamic", self.calls)


class BrokenInvocationRouteLane(FakeLane):
    def invoke_context(self, context_tokens: int) -> dict[str, object]:
        result = super().invoke_context(context_tokens)
        result["route_trace"] = [[0, 3, 7]]
        return result


class BrokenHoldRouteLane(FakeLane):
    def sample_hold_performance(self) -> dict[str, object]:
        result = super().sample_hold_performance()
        result["route_trace"] = [{"phase": "ar_decode", "layer": 1}]
        return result


class ZeroExpertHashLane(FakeLane):
    def invoke_context(self, context_tokens: int) -> dict[str, object]:
        result = super().invoke_context(context_tokens)
        hashes = result["expert_hashes"]
        assert isinstance(hashes, dict)
        hashes["0:3"] = "0" * 64
        return result


class SingleLaneHooks(FakeHooks):
    def __init__(self, calls: list[str], lane_type: type[FakeLane]) -> None:
        super().__init__(calls)
        self.lane_type = lane_type

    def load_dynamic_lane(self, request: ArmRequest) -> FakeLane:
        self.calls.append(f"load_dynamic:{request.context_tokens}")
        return self.lane_type("dynamic", self.calls)


class CapturedReclaimLane(FakeLane):
    captured_reclaim_pending = False

    def reclaim_experts_for_q4(self, context_tokens: int) -> None:
        super().reclaim_experts_for_q4(context_tokens)
        self.captured_reclaim_pending = True

    def physical_ledger(self) -> dict[str, object]:
        result = super().physical_ledger()
        if self.captured_reclaim_pending:
            result["captured_monotonic_ns"] = 200
            self.captured_reclaim_pending = False
        return result

    def kv_growth_steps(self) -> list[dict[str, object]]:
        steps = super().kv_growth_steps()
        steps[0].update(
            before_monotonic_ns=150,
            reclaim_monotonic_ns=200,
            after_monotonic_ns=250,
        )
        steps[0]["reclaim_gap"]["captured_monotonic_ns"] = 200
        steps[1].update(
            before_monotonic_ns=260,
            reclaim_monotonic_ns=270,
            after_monotonic_ns=280,
        )
        steps[1]["reclaim_gap"]["captured_monotonic_ns"] = 270
        return steps


class CapturedReclaimHooks(FakeHooks):
    def load_dynamic_lane(self, request: ArmRequest) -> FakeLane:
        self.calls.append(f"load_dynamic:{request.context_tokens}")
        return CapturedReclaimLane("dynamic", self.calls)


def test_dynamic_producer_tracks_full_lifecycle_and_emits_schema_v1() -> None:
    calls: list[str] = []
    timestamps = iter(
        (
            1,
            20,
            70,
            4_000_000_000,
            4_500_000_000,
            5_000_000_000,
            6_000_000_000,
            7_000_000_000,
            8_000_000_000,
        )
    )

    result = produce_arm_observation(
        hooks=FakeHooks(calls),
        request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=2),
        source_git_commit=SOURCE_COMMIT,
        monotonic_ns=lambda: next(timestamps),
        sleep=lambda _seconds: None,
    )

    validated = validate_campaign_observation(result)
    assert validated.arm == "dynamic"
    assert result["schema"] == "mtplx-hy3-dynamic-memory-observation-v1"
    assert result["prompt_sha256"] == canonical_sha256([11, 4096, 12])
    assert result["generated_token_sha256"] == canonical_sha256([101, 202, 303, 404])
    expected_routes = [
        {"phase": "ar_decode", "layer": 0, "expert_ids": [3, 7]},
        {"phase": "ar_decode", "layer": 1, "expert_ids": [2, 9]},
    ]
    assert result["route_trace_sha256"] == canonical_sha256(expected_routes)
    assert result["expert_route_binding"] == {
        "schema": "mtplx-hy3-expert-route-binding-v1",
        "producer_verification_scope": "route-map-from-loaded-manifest",
        "offline_verification_scope": "structural-binding-only",
        "expert_manifest_sha256": "2" * 64,
        "route_trace_sha256": canonical_sha256(expected_routes),
        "expert_hashes_sha256": canonical_sha256(result["expert_hashes"]),
    }
    assert [point["phase"] for point in result["timeline"]] == [
        "pre_growth",
        "post_expert_reclaim",
        "post_kv_growth",
        "hold_warmup",
        "hold",
        "hold",
        "hold",
        "post_reset",
        "post_record_rewarm",
    ]
    assert [point["kv_logical_tokens"] for point in result["timeline"]] == [
        1,
        1,
        4096,
        4096,
        4096,
        4096,
        4096,
        0,
        0,
    ]
    assert {
        step[ledger_name]["kv_logical_tokens"]
        for step in result["kv_growth_steps"]
        for ledger_name in ("before", "reclaim_gap", "after")
    } == {1}
    assert result["lifecycle"] == {
        "reset_observed": True,
        "future_demand_invoked": True,
        "post_record_rewarm_observed": True,
    }
    assert result["metrics"]["hold_performance_samples"] == [15.9, 16.0, 16.1]
    assert result["metrics"]["hold_warmup_sample"]["generated_token_ids"] == list(
        range(400, 408)
    )
    assert calls.index("stabilize_hold") < calls.index("sample_hold")
    assert [step["target_blocks"] for step in result["kv_growth_steps"]] == [
        255,
        256,
    ]
    assert result["metrics"]["peak_charged_bytes"] == max(
        point["charged_bytes"] for point in result["timeline"]
    )
    first_sample = result["metrics"]["performance_samples"][0]
    assert first_sample["expert_hit_rate"] == 0.75
    assert first_sample["generated_token_ids"] == list(range(500, 508))
    assert first_sample["generated_token_sha256"] == canonical_sha256(
        list(range(500, 508))
    )
    assert first_sample["route_trace"] == [
        {"phase": "ar_decode", "layer": 1, "expert_ids": [3, 7, 11]}
    ]
    assert first_sample["route_trace_sha256"] == canonical_sha256(
        [{"phase": "ar_decode", "layer": 1, "expert_ids": [3, 7, 11]}]
    )
    assert first_sample["expert_route_binding"]["expert_manifest_sha256"] == "2" * 64
    assert (
        first_sample["expert_route_binding"]["offline_verification_scope"]
        == "structural-binding-only"
    )
    assert "load_static:4096" not in calls
    assert calls[-3:] == ["future_demand", "ledger", "close"]


def test_producer_rejects_hold_count_that_drifts_from_hardware_prompt_reserve() -> None:
    calls: list[str] = []

    with pytest.raises(ArmObservationError, match="hold sample count"):
        produce_arm_observation(
            hooks=FixedHoldHooks(calls),
            request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=0),
            source_git_commit=SOURCE_COMMIT,
            hold_samples=3,
        )

    assert calls == []


def test_static_producer_uses_reserved_control_without_reclaim_or_rewarm() -> None:
    calls: list[str] = []
    timestamps = iter(
        (
            1,
            2,
            3_000_000_000,
            3_500_000_000,
            4_000_000_000,
            5_000_000_000,
            6_000_000_000,
        )
    )

    result = produce_arm_observation(
        hooks=FakeHooks(calls),
        request=ArmRequest(arm="static", context_tokens=32_768, repetition=1),
        source_git_commit=SOURCE_COMMIT,
        monotonic_ns=lambda: next(timestamps),
        sleep=lambda _seconds: None,
    )

    validated = validate_campaign_observation(result)
    assert validated.arm == "static"
    assert result["cache_start_state"] == {
        "kind": "static-128k-reserved-q4",
        "kv_physical_bytes": HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES,
        "kv_blocks": HY3_Q4_MAX_BLOCKS,
    }
    assert [point["phase"] for point in result["timeline"]] == [
        "pre_growth",
        "post_kv_growth",
        "hold_warmup",
        "hold",
        "hold",
        "hold",
        "post_reset",
    ]
    assert result["lifecycle"] == {
        "reset_observed": True,
        "future_demand_invoked": False,
        "post_record_rewarm_observed": False,
    }
    assert result["kv_growth_steps"] == []
    assert "load_dynamic:32768" not in calls
    assert not any(call.startswith("reclaim:") for call in calls)
    assert "future_demand" not in calls
    assert calls[-2:] == ["ledger", "close"]


def test_contradictory_q4_ledger_fails_before_model_invocation_and_closes() -> None:
    calls: list[str] = []

    with pytest.raises(ArmObservationError, match="Q4 block geometry"):
        produce_arm_observation(
            hooks=BrokenQ4LedgerHooks(calls),
            request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=0),
            source_git_commit=SOURCE_COMMIT,
            monotonic_ns=lambda: 1,
            sleep=lambda _seconds: None,
        )

    assert "invoke:4096" not in calls
    assert calls[-1] == "close"


@pytest.mark.parametrize(
    ("lane_type", "match"),
    (
        (BrokenInvocationRouteLane, "route_trace.*object"),
        (BrokenHoldRouteLane, "route_trace.*incomplete"),
        (ZeroExpertHashLane, "expert.*SHA-256|zero"),
    ),
)
def test_producer_rejects_unverified_route_or_expert_map_evidence(
    lane_type: type[FakeLane],
    match: str,
) -> None:
    calls: list[str] = []
    timestamps = iter(
        (
            1,
            20,
            70,
            4_000_000_000,
            4_500_000_000,
            5_000_000_000,
            6_000_000_000,
            7_000_000_000,
        )
    )

    with pytest.raises(ArmObservationError, match=match):
        produce_arm_observation(
            hooks=SingleLaneHooks(calls, lane_type),
            request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=0),
            source_git_commit=SOURCE_COMMIT,
            monotonic_ns=lambda: next(timestamps),
            sleep=lambda _seconds: None,
        )

    assert calls[-1] == "close"


def test_producer_preserves_timestamp_captured_inside_reclaim_allocation_gap() -> None:
    timestamps = iter(
        (
            100,
            300,
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
            4_000_000_000,
            5_000_000_000,
            6_000_000_000,
        )
    )

    result = produce_arm_observation(
        hooks=CapturedReclaimHooks([]),
        request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=0),
        source_git_commit=SOURCE_COMMIT,
        monotonic_ns=lambda: next(timestamps),
        sleep=lambda _seconds: None,
    )

    assert result["timeline"][1]["phase"] == "post_expert_reclaim"
    assert result["timeline"][1]["monotonic_ns"] == 200
    assert result["timeline"][2]["monotonic_ns"] == 300


def test_clean_source_requires_exact_repo_root_full_commit_and_no_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    calls: list[tuple[str, ...]] = []

    def fake_git_output(repo_root: Path, *args: str) -> str:
        assert repo_root == root
        calls.append(args)
        if args == ("rev-parse", "--show-toplevel"):
            return str(root)
        if args == ("rev-parse", "HEAD"):
            return SOURCE_COMMIT
        if args == ("ls-files", "-v"):
            return "H tracked.py"
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(observation_module, "_git_output", fake_git_output)

    assert require_clean_source(root) == SOURCE_COMMIT
    assert calls == [
        ("rev-parse", "--show-toplevel"),
        ("rev-parse", "HEAD"),
        ("ls-files", "-v"),
        ("status", "--porcelain=v1", "--untracked-files=all"),
    ]

    def dirty_git_output(repo_root: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo_root)
        if args == ("rev-parse", "HEAD"):
            return SOURCE_COMMIT
        if args == ("ls-files", "-v"):
            return "H tracked.py"
        return "?? local_issue46_hooks.py"

    monkeypatch.setattr(observation_module, "_git_output", dirty_git_output)
    with pytest.raises(ArmObservationError, match="clean source worktree"):
        require_clean_source(root)


def test_hook_loader_checks_factory_source_is_tracked_and_passes_exact_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "issue46_test_hooks.py"
    module_path.write_text(
        """
class Hooks:
    def __init__(self, config):
        self.config = config
    def load_static_lane(self, request):
        raise AssertionError(request)
    def load_dynamic_lane(self, request):
        raise AssertionError(request)

def create_hooks(config):
    return Hooks(config)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    tracked: list[Path] = []
    monkeypatch.setattr(
        observation_module,
        "_require_tracked_file",
        lambda repo_root, path: tracked.append(path.resolve()),
    )

    hooks = load_tracked_hooks(
        "issue46_test_hooks:create_hooks",
        {"model": "/models/hy3", "slab_slots": 32},
        repo_root=tmp_path,
    )

    assert hooks.config == {"model": "/models/hy3", "slab_slots": 32}
    assert tracked == [module_path.resolve()]


def test_hook_loader_fails_closed_when_factory_source_is_untracked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "issue46_untracked_hooks.py"
    module_path.write_text(
        """
class Hooks:
    load_static_lane = lambda self, request: None
    load_dynamic_lane = lambda self, request: None

def create_hooks(config):
    return Hooks()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    def reject_untracked(_repo_root: Path, path: Path) -> None:
        raise ArmObservationError(f"hook factory source is not tracked: {path.name}")

    monkeypatch.setattr(
        observation_module,
        "_require_tracked_file",
        reject_untracked,
    )

    with pytest.raises(ArmObservationError, match="not tracked"):
        load_tracked_hooks(
            "issue46_untracked_hooks:create_hooks",
            {},
            repo_root=tmp_path,
        )


def test_json_only_cli_is_direct_arm_command_template_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "arm-hooks.json"
    config_path.write_text(json.dumps({"model_root": "/models/hy3"}), encoding="utf-8")
    hooks = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        observation_module,
        "require_clean_source",
        lambda _repo_root: SOURCE_COMMIT,
    )
    monkeypatch.setattr(
        observation_module,
        "_require_tracked_file",
        lambda _repo_root, _path: None,
    )

    def fake_load_hooks(spec: str, config: object, *, repo_root: Path) -> object:
        print("hook factory diagnostic")
        captured.update(spec=spec, config=config, repo_root=repo_root)
        return hooks

    def fake_produce(**kwargs: object) -> dict[str, object]:
        print("lane diagnostic")
        captured.update(kwargs)
        request = kwargs["request"]
        assert isinstance(request, ArmRequest)
        return {
            "schema": "mtplx-hy3-dynamic-memory-observation-v1",
            "arm": request.arm,
            "context_tokens": request.context_tokens,
            "repetition": request.repetition,
        }

    monkeypatch.setattr(observation_module, "load_tracked_hooks", fake_load_hooks)
    monkeypatch.setattr(observation_module, "produce_arm_observation", fake_produce)
    command = format_arm_command(
        (
            "observe-hy3-arm",
            "--repo-root",
            str(tmp_path),
            "--hooks",
            "tracked_hooks:create_hooks",
            "--hooks-config",
            str(config_path),
            "--arm",
            "{arm}",
            "--context-tokens",
            "{context_tokens}",
            "--repetition",
            "{repetition}",
        ),
        arm="dynamic",
        context_tokens=65_536,
        repetition=3,
    )

    assert main(command[1:]) == 0

    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "schema": "mtplx-hy3-dynamic-memory-observation-v1",
        "arm": "dynamic",
        "context_tokens": 65_536,
        "repetition": 3,
    }
    assert "hook factory diagnostic" in output.err
    assert "lane diagnostic" in output.err
    assert captured["spec"] == "tracked_hooks:create_hooks"
    assert captured["config"] == {"model_root": "/models/hy3"}
    request = captured["request"]
    assert isinstance(request, ArmRequest)
    assert request.context_tokens == 65_536


def test_json_only_cli_emits_no_partial_json_when_source_is_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "arm-hooks.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        observation_module,
        "require_clean_source",
        lambda _repo_root: (_ for _ in ()).throw(
            ArmObservationError("clean source worktree required")
        ),
    )

    exit_code = main(
        (
            "--repo-root",
            str(tmp_path),
            "--hooks",
            "tracked_hooks:create_hooks",
            "--hooks-config",
            str(config_path),
            "--arm",
            "static",
            "--context-tokens",
            "4096",
            "--repetition",
            "0",
        )
    )

    output = capsys.readouterr()
    assert exit_code == 2
    assert output.out == ""
    assert "clean source worktree required" in output.err
