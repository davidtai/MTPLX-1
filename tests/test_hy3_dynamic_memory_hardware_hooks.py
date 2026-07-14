from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.benchmarks.hy3_dynamic_memory_hardware as hardware_module
from mtplx.benchmarks.hy3_dynamic_memory_hardware import (
    Hy3HardwareConfig,
    MlxHy3HardwareLane,
    ProductionHy3HardwareHooks,
    create_hooks,
    preflight_and_grow_dynamic_q4,
    trigger_future_demand_regrow,
)
from mtplx.benchmarks.hy3_dynamic_memory_observation import (
    ArmObservationError,
    ArmRequest,
)
from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    CONTEXT_MATRIX_TOKENS,
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_MAX_BLOCKS,
)
from mtplx.expert_streaming_models import HY3_Q4


@dataclass
class FakeBrokerSnapshot:
    kv_physical_bytes: int
    expert_slab_physical_bytes: int


@dataclass
class FakeTicket:
    ticket_id: int = 7


class FakeExpertRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.snapshot = FakeBrokerSnapshot(
            kv_physical_bytes=HY3_Q4_KV_BLOCK_BYTES,
            expert_slab_physical_bytes=800,
        )
        self.memory_broker = self

    def reserve_growth(self, **kwargs: int | str) -> FakeTicket:
        self.events.append(
            f"reserve:{kwargs['steady_delta_bytes']}:{kwargs['transient_delta_bytes']}"
        )
        self.snapshot.expert_slab_physical_bytes = 600
        return FakeTicket()

    def abort_growth(
        self,
        ticket: FakeTicket,
        *,
        observed_physical_bytes: int,
    ) -> None:
        assert ticket.ticket_id == 7
        assert observed_physical_bytes == 0
        self.events.append("abort")

    def commit_growth(
        self,
        ticket: FakeTicket,
        *,
        measured_physical_bytes: int,
        allocator_before: object,
        allocator_after: object,
    ) -> object:
        assert ticket.ticket_id == 7
        assert allocator_before is None
        assert allocator_after is None
        self.events.append(f"commit:{measured_physical_bytes}")
        return object()


class FakeRuntime:
    def __init__(self, events: list[str]) -> None:
        self.expert_streaming = FakeExpertRuntime(events)


class FakeQ4Entry:
    def __init__(
        self,
        events: list[str],
        label: str,
        allocation_observer: object | None = None,
    ) -> None:
        self.events = events
        self.label = label
        self.block_size = 16
        self.num_blocks = 1
        self.nbytes = HY3_Q4_KV_BLOCK_BYTES // 2
        self.kv_quant = True
        self.kv_quant_config = type("Q4", (), {"normalized_mode": "q4"})()
        self._shape = (1, 1, 1)
        self._dtypes = (object(), object())
        self.allocation_observer = allocation_observer
        self.cache_id = f"fake-{label}"

    def _planned_capacity_bytes(
        self,
        *,
        num_blocks: int,
        shape: object,
        dtypes: object,
    ) -> int:
        assert shape is self._shape
        assert dtypes is self._dtypes
        return num_blocks * (HY3_Q4_KV_BLOCK_BYTES // 2)

    def _grow_to_capacity(self, required_tokens: int) -> bool:
        target_blocks = required_tokens // 16
        target_bytes = target_blocks * (HY3_Q4_KV_BLOCK_BYTES // 2)
        old_bytes = self.nbytes
        observer = self.allocation_observer
        assert observer is not None
        ticket = observer.reserve_growth(
            cache_id=self.cache_id,
            steady_delta_bytes=target_bytes - old_bytes,
            transient_delta_bytes=target_bytes,
        )
        self.events.append(f"grow:{self.label}:{required_tokens}")
        self.num_blocks = target_blocks
        self.nbytes = target_bytes
        observer.commit_growth(
            ticket,
            measured_physical_bytes=target_bytes - old_bytes,
            allocator_before=None,
            allocator_after=None,
        )
        return True


@pytest.mark.parametrize("context_tokens", CONTEXT_MATRIX_TOKENS)
def test_dynamic_preflight_captures_real_reclaim_gap_before_any_q4_growth(
    context_tokens: int,
) -> None:
    events: list[str] = []
    runtime = FakeRuntime(events)
    cache = [
        FakeQ4Entry(events, "a", runtime.expert_streaming),
        FakeQ4Entry(events, "b", runtime.expert_streaming),
    ]

    def ledger() -> dict[str, object]:
        events.append("ledger")
        broker = runtime.expert_streaming.snapshot
        blocks = cache[0].num_blocks
        kv_physical_bytes = sum(entry.nbytes for entry in cache)
        return {
            "allocator_active_bytes": (
                broker.expert_slab_physical_bytes + kv_physical_bytes
            ),
            "allocator_cache_bytes": 0,
            "allocator_peak_bytes": (
                broker.expert_slab_physical_bytes + kv_physical_bytes
            ),
            "expert_slab_physical_bytes": broker.expert_slab_physical_bytes,
            "kv_physical_bytes": kv_physical_bytes,
            "kv_allocated_blocks": blocks,
            "slot_health": {
                "active_routes": 0,
                "pins": 0,
                "loading": 0,
                "failed": 0,
                "integrity_errors": 0,
                "completion_fence_failures": 0,
            },
        }

    def clock() -> int:
        events.append("clock")
        return 77

    captured = preflight_and_grow_dynamic_q4(
        runtime=runtime,
        cache=cache,
        context_tokens=context_tokens,
        physical_ledger=ledger,
        monotonic_ns=clock,
    )

    target_blocks = context_tokens // 16
    per_entry_delta = (target_blocks - 1) * (HY3_Q4_KV_BLOCK_BYTES // 2)
    largest_replacement = target_blocks * (HY3_Q4_KV_BLOCK_BYTES // 2)
    assert captured["expert_slab_physical_bytes"] == 600
    assert captured["kv_physical_bytes"] == HY3_Q4_KV_BLOCK_BYTES
    assert captured["captured_monotonic_ns"] == 77
    assert events == [
        "ledger",
        f"reserve:{per_entry_delta}:{largest_replacement}",
        "ledger",
        "clock",
        f"grow:a:{context_tokens}",
        f"commit:{per_entry_delta}",
        f"reserve:{per_entry_delta}:{largest_replacement}",
        f"grow:b:{context_tokens}",
        f"commit:{per_entry_delta}",
        "ledger",
    ]


def test_host_memory_health_uses_exact_mach_ledgers_without_subprocess_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hardware_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "process memory telemetry must not invoke ps, vmmap, or sysctl"
        ),
    )

    baseline = hardware_module._system_swap_used_bytes()
    snapshot = hardware_module._host_memory_health_snapshot(initial_swap_bytes=baseline)

    assert snapshot["process_rss_bytes"] > 0
    assert snapshot["process_compressed_bytes"] >= 0
    assert isinstance(snapshot["system_swap_delta_bytes"], int)


class FakeReadyRoute:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def release(self) -> None:
        self.events.append("release_route")


class FakeDemandRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def ensure_route(
        self,
        layer: int,
        expert_ids: tuple[int, ...],
        *,
        phase: str,
    ) -> FakeReadyRoute:
        self.events.append(f"ensure_route:{layer}:{expert_ids}:{phase}")
        return FakeReadyRoute(self.events)

    def maybe_regrow_expert_slabs(self, **_kwargs: object) -> None:
        raise AssertionError("regrow must be caused by route demand")


def test_regrow_is_triggered_only_by_a_real_future_route_demand() -> None:
    events: list[str] = []
    runtime = FakeDemandRuntime(events)
    physical = iter((600, 800))

    trigger_future_demand_regrow(
        expert_runtime=runtime,
        route_trace=[
            {
                "phase": "decode",
                "layer": 7,
                "expert_ids": [1, 4, 9],
            }
        ],
        expert_physical_bytes=lambda: next(physical),
    )

    assert events == ["ensure_route:7:(1, 4, 9):decode", "release_route"]


def test_production_factory_builds_real_static_and_dynamic_hook_loader(
    tmp_path: Path,
) -> None:
    config = {
        "repo_root": str(tmp_path),
        "model_root": str(tmp_path / "Hy3-4bit"),
        "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
        "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        "generated_tokens": 16,
        "hold_sample_count": 3,
        "hold_tokens": 8,
    }

    hooks = create_hooks(config)

    assert isinstance(hooks, ProductionHy3HardwareHooks)
    assert isinstance(hooks.config, Hy3HardwareConfig)
    assert hooks.config.model_root == tmp_path / "Hy3-4bit"
    assert hooks.hold_samples == 3
    assert callable(hooks.load_static_lane)
    assert callable(hooks.load_dynamic_lane)


def test_hardware_arm_environment_forces_full_history_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        }
    )
    sliding_key = "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"
    monkeypatch.setenv(sliding_key, "2048")
    observed: dict[str, str | None] = {}

    for arm in ("static", "dynamic"):
        lease = hardware_module._EnvironmentLease(
            hardware_module._arm_environment(arm, config)
        )
        lease.acquire()
        try:
            observed[arm] = os.environ.get(sliding_key)
        finally:
            lease.release()

    assert observed == {"static": "0", "dynamic": "0"}
    assert os.environ[sliding_key] == "2048"


def test_static_control_budget_charges_exact_physical_q4_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.expert_manifest as expert_manifest_module
    import mtplx.runtime as runtime_module

    class ModelLoadBoundary(RuntimeError):
        pass

    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        }
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        expert_manifest_module,
        "load_expert_manifest",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        hardware_module,
        "_artifact_identity",
        lambda *_args, **_kwargs: ("1" * 64, "2" * 64),
    )

    def stop_at_model_load(*_args: object, **kwargs: object) -> object:
        observed["config"] = kwargs["expert_streaming_config"]
        raise ModelLoadBoundary

    monkeypatch.setattr(runtime_module, "load", stop_at_model_load)
    hooks = ProductionHy3HardwareHooks(config)

    with pytest.raises(ModelLoadBoundary):
        hooks.load_static_lane(
            ArmRequest(arm="static", context_tokens=4096, repetition=0)
        )

    runtime_config = observed["config"]
    plan = runtime_config.memory_plan(HY3_Q4)
    assert {
        "dynamic_expert_slabs": runtime_config.dynamic_expert_slabs,
        "planned_context_tokens": plan.context_tokens,
        "planned_kv_bytes": plan.kv_bytes,
    } == {
        "dynamic_expert_slabs": False,
        "planned_context_tokens": hardware_module.TOTAL_CONTEXT_TOKENS,
        "planned_kv_bytes": HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES,
    }
    assert plan.kv_bytes != (
        hardware_module.TOTAL_CONTEXT_TOKENS * HY3_Q4.kv_bytes_per_token
    )


def test_static_lane_keeps_full_q4_reservation_for_short_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        }
    )
    cache = [FakeQ4Entry([], "a"), FakeQ4Entry([], "b")]
    for entry in cache:
        entry.num_blocks = HY3_Q4_MAX_BLOCKS
        entry.nbytes = HY3_Q4_MAX_BLOCKS * (HY3_Q4_KV_BLOCK_BYTES // 2)
    observed: dict[str, object] = {}
    sentinel_logits = object()

    def fake_forward_prefill(
        runtime: object,
        forwarded_cache: object,
        token_ids: object,
        *,
        chunk_size: int,
    ) -> object:
        observed.update(
            runtime=runtime,
            cache=forwarded_cache,
            token_ids=token_ids,
            chunk_size=chunk_size,
        )
        return sentinel_logits

    monkeypatch.setattr(hardware_module, "_forward_prefill", fake_forward_prefill)
    runtime = object()
    lane = MlxHy3HardwareLane(
        config=config,
        request=ArmRequest(arm="static", context_tokens=4096, repetition=0),
        runtime=runtime,
        runtime_config=object(),
        manifest=object(),
        prompt_ids=[11, 22, 33],
        cache=cache,
        logits=object(),
        admission=object(),
        environment=object(),
        model_artifact_sha256="1" * 64,
        manifest_file_sha256="2" * 64,
        source_git_commit="a" * 40,
    )

    lane.prepare_q4_context(4096)

    assert all(entry.num_blocks == HY3_Q4_MAX_BLOCKS for entry in cache)
    assert lane.logits is sentinel_logits
    assert observed == {
        "runtime": runtime,
        "cache": cache,
        "token_ids": [22, 33],
        "chunk_size": config.prefill_chunk_size,
    }


def test_static_physical_ledger_does_not_require_dynamic_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.expert_runtime as expert_runtime_module

    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        }
    )
    cache = [FakeQ4Entry([], "a"), FakeQ4Entry([], "b")]
    for entry in cache:
        entry.num_blocks = HY3_Q4_MAX_BLOCKS
        entry.nbytes = HY3_Q4_MAX_BLOCKS * (HY3_Q4_KV_BLOCK_BYTES // 2)
    slot_snapshot = {
        "slabs": {"physical_bytes": 800},
        "metrics": {"active_routes": 0, "completion_fence_failures": 0},
        "states": {"loading": 0, "failed": 0},
        "pins": 0,
    }
    expert_runtime = SimpleNamespace(
        _raise_if_unhealthy=lambda: None,
        slots=SimpleNamespace(
            snapshot=lambda: pytest.fail("physical ledger must not globally drain"),
            expert_slab_telemetry_snapshot=lambda: {
                "physical_bytes": 800,
                "logical_slab_count": 1,
                "active_slab_count": 1,
                "draining_slab_count": 0,
                "released_slab_count": 0,
                "logical_slot_count": 1,
                "active_slot_count": 1,
                "resident_record_count": 0,
                "in_flight_bytes": 0,
                "pinned_bytes": 0,
            },
            health_telemetry_snapshot=lambda: {
                "metrics": slot_snapshot["metrics"],
                "states": slot_snapshot["states"],
                "pins": 0,
            },
        ),
        memory_broker=None,
    )
    runtime = SimpleNamespace(expert_streaming=expert_runtime)
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 900,
            "cache_memory_bytes": 100,
            "peak_memory_bytes": 1_000,
        },
    )
    lane = MlxHy3HardwareLane(
        config=config,
        request=ArmRequest(arm="static", context_tokens=4096, repetition=0),
        runtime=runtime,
        runtime_config=object(),
        manifest=object(),
        prompt_ids=[11],
        cache=cache,
        logits=object(),
        admission=object(),
        environment=object(),
        model_artifact_sha256="1" * 64,
        manifest_file_sha256="2" * 64,
        source_git_commit="a" * 40,
    )

    ledger = lane._live_physical_ledger()

    assert ledger["kv_allocated_blocks"] == HY3_Q4_MAX_BLOCKS
    assert ledger["kv_physical_bytes"] == (HY3_Q4_MAX_BLOCKS * HY3_Q4_KV_BLOCK_BYTES)
    assert ledger["expert_slab_physical_bytes"] == 800
    assert ledger["allocator_cache_bytes"] == 100
    assert ledger["allocator_cache_charged_bytes"] == 100
    assert ledger["charged_bytes"] == (ledger["kv_physical_bytes"] + 800 + 100)

    lane.request = ArmRequest(arm="dynamic", context_tokens=4096, repetition=0)
    with pytest.raises(ArmObservationError, match="dynamic arm lost"):
        lane._live_physical_ledger()


def test_dynamic_physical_ledger_emits_complete_issue46_resource_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.expert_runtime as expert_runtime_module

    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        }
    )
    cache = [FakeQ4Entry([], "a"), FakeQ4Entry([], "b")]
    slot_snapshot = {
        "slabs": {"physical_bytes": 800},
        "metrics": {"active_routes": 0, "completion_fence_failures": 0},
        "states": {"loading": 0, "failed": 0},
        "pins": 0,
    }
    broker_snapshot = SimpleNamespace(
        resident_model_bytes=100,
        kv_physical_bytes=HY3_Q4_KV_BLOCK_BYTES,
        expert_slab_physical_bytes=800,
        in_flight_expert_staging_bytes=20,
        runtime_workspace_bytes=30,
        allocator_cache_bytes=10,
        pinned_expert_bytes=40,
        speculative_expert_bytes=50,
        charged_bytes=960 + HY3_Q4_KV_BLOCK_BYTES,
        admission_failure_count=2,
        failed_closed=False,
        failure_reason=None,
    )
    broker = SimpleNamespace(
        budget=SimpleNamespace(
            operating_target_bytes=110 * hardware_module.GIB,
            hard_ceiling_bytes=112 * hardware_module.GIB,
        ),
        snapshot=lambda: broker_snapshot,
    )
    expert_runtime = SimpleNamespace(
        _raise_if_unhealthy=lambda: None,
        slots=SimpleNamespace(
            snapshot=lambda: pytest.fail("physical ledger must not globally drain"),
            expert_slab_telemetry_snapshot=lambda: {
                "physical_bytes": 800,
                "logical_slab_count": 4,
                "active_slab_count": 3,
                "draining_slab_count": 0,
                "released_slab_count": 1,
                "logical_slot_count": 128,
                "active_slot_count": 96,
                "resident_record_count": 80,
                "in_flight_bytes": 60,
                "pinned_bytes": 40,
            },
            health_telemetry_snapshot=lambda: {
                "metrics": slot_snapshot["metrics"],
                "states": slot_snapshot["states"],
                "pins": 0,
            },
        ),
        memory_broker=broker,
    )
    resource_snapshot = {
        "dynamic_memory": {
            "logical_expert_records": 128,
            "active_expert_records": 96,
            "resident_expert_records": 80,
            "logical_slab_count": 4,
            "active_slab_count": 3,
            "draining_slab_count": 0,
            "released_slab_count": 1,
            "in_flight_expert_bytes": 60,
            "requested_reclaim_bytes": 70,
            "reclaimed_bytes": 64,
            "regrown_bytes": 32,
            "resize_duration_ns": 900,
            "total_resize_duration_ns": 1_800,
            "max_resize_duration_ns": 1_000,
            "blocked_by_pin_bytes": 16,
            "resize_failures": 1,
        },
        "cache": {"evictions": 7},
        "kv": {
            "representation": "q4",
            "logical_tokens": 16,
            "physical_blocks": 1,
            "physical_bytes": HY3_Q4_KV_BLOCK_BYTES,
        },
        "expert_evicted_slabs": 2,
    }
    runtime = SimpleNamespace(
        expert_streaming=expert_runtime,
        expert_resource_telemetry_snapshot=lambda: resource_snapshot,
    )
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 900,
            "cache_memory_bytes": 10,
            "peak_memory_bytes": 1_000,
        },
    )
    monkeypatch.setattr(
        hardware_module,
        "_host_memory_health_snapshot",
        lambda *_args, **_kwargs: {
            "process_rss_bytes": 1_200,
            "process_compressed_bytes": 300,
            "system_swap_delta_bytes": 10,
        },
        raising=False,
    )
    lane = MlxHy3HardwareLane(
        config=config,
        request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=0),
        runtime=runtime,
        runtime_config=object(),
        manifest=object(),
        prompt_ids=[11],
        cache=cache,
        logits=object(),
        admission=object(),
        environment=object(),
        model_artifact_sha256="1" * 64,
        manifest_file_sha256="2" * 64,
        source_git_commit="a" * 40,
    )

    ledger = lane._live_physical_ledger()

    expected = {
        "operating_target_bytes": 110 * hardware_module.GIB,
        "hard_ceiling_bytes": 112 * hardware_module.GIB,
        "charged_bytes": 960 + HY3_Q4_KV_BLOCK_BYTES,
        "resident_model_bytes": 100,
        "kv_representation": "q4",
        "kv_logical_tokens": 16,
        "expert_logical_records": 128,
        "expert_active_records": 96,
        "expert_resident_records": 80,
        "expert_logical_slabs": 4,
        "expert_active_slabs": 3,
        "expert_draining_slabs": 0,
        "expert_released_slabs": 1,
        "pinned_expert_bytes": 40,
        "inflight_expert_bytes": 60,
        "speculative_expert_bytes": 50,
        "runtime_workspace_bytes": 30,
        "inflight_expert_staging_bytes": 20,
        "requested_reclaim_bytes": 70,
        "reclaimed_bytes": 64,
        "regrown_bytes": 32,
        "evicted_expert_records": 7,
        "evicted_expert_slabs": 2,
        "resize_duration_ns": 900,
        "total_resize_duration_ns": 1_800,
        "max_resize_duration_ns": 1_000,
        "blocked_by_pin_bytes": 16,
        "admission_failures": 2,
        "resize_failures": 1,
        "allocator_cache_charged_bytes": 10,
        "process_rss_bytes": 1_200,
        "process_compressed_bytes": 300,
        "system_swap_delta_bytes": 10,
        "failed_closed": False,
        "failure_reason": None,
    }
    missing = set(expected) - set(ledger)
    assert not missing, f"physical ledger omitted issue #46 resource fields: {missing}"
    assert {field: ledger[field] for field in expected} == expected
