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
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_MAX_BLOCKS,
)
from mtplx.expert_streaming_models import HY3_Q4


def _artifact_pins() -> dict[str, object]:
    return {
        "model_config_sha256": "a" * 64,
        "manifest_file_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "sidecar_file": "experts.bin",
        "sidecar_bytes": 1,
        "sidecar_sha256": "d" * 64,
        "resident_payload_bytes": 1,
        "resident_payload_sha256": "e" * 64,
        "source_revision": "revision",
    }


@dataclass
class FakeBrokerSnapshot:
    kv_physical_bytes: int
    expert_slab_physical_bytes: int


class FakeGrowthGroup:
    def __init__(
        self,
        events: list[str],
        members: tuple[tuple[str, int, int], ...],
    ) -> None:
        self.events = events
        self.members = members
        self.commits: list[tuple[str, int]] = []
        self.completed = False
        self.aborted = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self.completed:
            self.aborted = True

    def commit_member(
        self,
        *,
        cache_id: str,
        measured_physical_bytes: int,
        allocator_before: object,
        allocator_after: object,
    ) -> object:
        assert allocator_before is None
        assert allocator_after is None
        self.commits.append((cache_id, measured_physical_bytes))
        self.events.append(f"commit:{cache_id}")
        self.completed = len(self.commits) == len(self.members)
        return object()

    def abort(self, **_kwargs: object) -> None:
        self.aborted = True
        self.events.append("group-abort")


class FakeExpertRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.groups: list[FakeGrowthGroup] = []
        self.single_reservations = 0
        self.snapshot = FakeBrokerSnapshot(
            kv_physical_bytes=HY3_Q4_KV_BLOCK_BYTES,
            expert_slab_physical_bytes=800,
        )
        self.memory_broker = self

    def reserve_growth(self, **_kwargs: int | str) -> object:
        self.single_reservations += 1
        raise AssertionError("hardware preflight used a single-entry reservation")

    def reserve_growth_group(
        self,
        *,
        members: tuple[tuple[str, int, int], ...],
    ) -> FakeGrowthGroup:
        self.events.append(f"group-reserve:{len(members)}")
        self.snapshot.expert_slab_physical_bytes = 600
        group = FakeGrowthGroup(self.events, members)
        self.groups.append(group)
        return group


class FakeRuntime:
    def __init__(self, events: list[str]) -> None:
        self.expert_streaming = FakeExpertRuntime(events)


class FakeQ4Entry:
    def __init__(
        self,
        events: list[str],
        index: int,
        allocation_observer: object | None = None,
    ) -> None:
        self.events = events
        self.index = index
        self.block_size = 16
        self.num_blocks = 1
        self.nbytes = HY3_Q4_KV_BLOCK_BYTES // 80
        self.offset = 0
        self.kv_quant = True
        self.kv_quant_config = type("Q4", (), {"normalized_mode": "q4"})()
        self._shape = (8, 128, 128)
        self._dtypes = (object(), object())
        self.allocation_observer = allocation_observer
        self.cache_id = f"target:fake:{index}"
        self.key_cache = object()
        self.value_cache = object()
        self._active_growth_group = None
        self._active_growth_spec = None

    def _planned_capacity_bytes(
        self,
        *,
        num_blocks: int,
        shape: object,
        dtypes: object,
    ) -> int:
        del shape, dtypes
        return num_blocks * (HY3_Q4_KV_BLOCK_BYTES // 80)

    def _grow_to_capacity(self, required_tokens: int) -> bool:
        target_blocks = (required_tokens + self.block_size - 1) // self.block_size
        target_bytes = target_blocks * (HY3_Q4_KV_BLOCK_BYTES // 80)
        old_bytes = self.nbytes
        expected = (target_bytes - old_bytes, target_bytes)
        if self._active_growth_spec != expected or self._active_growth_group is None:
            observer = self.allocation_observer
            assert observer is not None
            observer.reserve_growth(
                cache_id=self.cache_id,
                steady_delta_bytes=expected[0],
                transient_delta_bytes=expected[1],
            )
            raise AssertionError("single-entry growth unexpectedly returned")
        self.events.append(f"grow:{self.index}:{required_tokens}")
        self.num_blocks = target_blocks
        self.nbytes = target_bytes
        self._active_growth_group.commit_member(
            cache_id=self.cache_id,
            measured_physical_bytes=target_bytes - old_bytes,
            allocator_before=None,
            allocator_after=None,
        )
        return True


@pytest.mark.parametrize(
    ("context_tokens", "target_blocks"),
    (
        (4_095, 256),
        (4_096, 256),
        (4_097, 257),
        (32_768, 2_048),
        (65_536, 4_096),
        (131_072, 8_192),
    ),
)
def test_dynamic_preflight_captures_real_reclaim_gap_before_any_q4_growth(
    monkeypatch: pytest.MonkeyPatch,
    context_tokens: int,
    target_blocks: int,
) -> None:
    import mtplx.cache_state as cache_state_module

    monkeypatch.setattr(cache_state_module, "VllmMetalPagedKVCache", FakeQ4Entry)
    events: list[str] = []
    runtime = FakeRuntime(events)
    cache = [
        FakeQ4Entry(events, index, runtime.expert_streaming) for index in range(80)
    ]

    def ledger() -> dict[str, object]:
        block_counts = {entry.num_blocks for entry in cache}
        assert len(block_counts) == 1, "ledger observed a partially committed group"
        blocks = block_counts.pop()
        events.append(f"ledger:{blocks}")
        broker = runtime.expert_streaming.snapshot
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
                "global_device_synchronizations": 0,
            },
        }

    ticks = iter(range(77, 100))

    def clock() -> int:
        events.append("clock")
        return next(ticks)

    captured = preflight_and_grow_dynamic_q4(
        runtime=runtime,
        cache=cache,
        context_tokens=context_tokens,
        physical_ledger=ledger,
        monotonic_ns=clock,
    )

    assert captured["expert_slab_physical_bytes"] == 600
    assert captured["kv_physical_bytes"] == HY3_Q4_KV_BLOCK_BYTES
    growth_steps = captured["kv_growth_steps"]
    assert isinstance(growth_steps, list)
    assert len(growth_steps) == 2
    checkpoint_tokens = (target_blocks - 1) * 16
    assert [step["requested_tokens"] for step in growth_steps] == [
        checkpoint_tokens,
        context_tokens,
    ]
    assert [step["target_blocks"] for step in growth_steps] == [
        target_blocks - 1,
        target_blocks,
    ]
    assert all(
        step["before_monotonic_ns"]
        < step["reclaim_monotonic_ns"]
        < step["after_monotonic_ns"]
        for step in growth_steps
    )
    assert growth_steps[0]["reclaim_gap"]["kv_allocated_blocks"] == 1
    assert growth_steps[0]["after"]["kv_allocated_blocks"] == target_blocks - 1
    assert growth_steps[1]["before"]["kv_allocated_blocks"] == target_blocks - 1
    assert growth_steps[1]["after"]["kv_allocated_blocks"] == target_blocks
    assert growth_steps[0]["reclaimed_expert_bytes"] > 0
    assert growth_steps[1]["kv_growth_bytes"] == HY3_Q4_KV_BLOCK_BYTES
    assert all(entry.num_blocks == target_blocks for entry in cache)
    assert all(entry.num_blocks * entry.block_size >= context_tokens for entry in cache)
    assert sum(entry.nbytes for entry in cache) == (
        target_blocks * HY3_Q4_KV_BLOCK_BYTES
    )
    assert runtime.expert_streaming.single_reservations == 0
    assert len(runtime.expert_streaming.groups) == 2
    first_group, second_group = runtime.expert_streaming.groups
    assert len(first_group.members) == len(first_group.commits) == 80
    assert len(second_group.members) == len(second_group.commits) == 80
    assert first_group.completed is second_group.completed is True
    assert first_group.aborted is second_group.aborted is False
    assert [owner for owner, _measured in first_group.commits] == [
        f"target:fake:{index}" for index in range(80)
    ]
    assert [owner for owner, _measured in second_group.commits] == [
        f"target:fake:{index}" for index in range(80)
    ]
    assert (
        sum(member[1] for member in first_group.members)
        == growth_steps[0]["steady_delta_bytes"]
    )
    assert (
        max(member[2] for member in first_group.members)
        == growth_steps[0]["max_transient_delta_bytes"]
    )
    assert (
        sum(member[1] for member in second_group.members)
        == growth_steps[1]["steady_delta_bytes"]
    )
    assert (
        max(member[2] for member in second_group.members)
        == growth_steps[1]["max_transient_delta_bytes"]
    )
    first_grow = events.index(f"grow:0:{checkpoint_tokens}")
    first_group_reserve = events.index("group-reserve:80")
    assert (
        first_group_reserve < events.index("ledger:1", first_group_reserve) < first_grow
    )
    second_group_reserve = events.index("group-reserve:80", first_group_reserve + 1)
    second_grow = events.index(f"grow:0:{target_blocks * 16}")
    assert (
        second_group_reserve
        < events.index(f"ledger:{target_blocks - 1}", second_group_reserve)
        < second_grow
    )


def test_hardware_campaign_front_door_remains_the_exact_context_matrix() -> None:
    assert ArmRequest(arm="dynamic", context_tokens=4_096, repetition=0)
    for context_tokens in (4_095, 4_097):
        with pytest.raises(ArmObservationError, match="context_tokens must be in"):
            ArmRequest(
                arm="dynamic",
                context_tokens=context_tokens,
                repetition=0,
            )


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

    def release(self, *, synchronize: bool = True) -> None:
        self.events.append(f"release_route:synchronize={synchronize}")


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

    assert events == [
        "ensure_route:7:(1, 4, 9):decode",
        "release_route:synchronize=False",
    ]


def test_production_factory_builds_real_static_and_dynamic_hook_loader(
    tmp_path: Path,
) -> None:
    config = {
        "repo_root": str(tmp_path),
        "model_root": str(tmp_path / "Hy3-4bit"),
        "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
        "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
        "artifact_pins": _artifact_pins(),
        "generated_tokens": 16,
        "hold_sample_count": 3,
        "hold_tokens": 8,
    }

    hooks = create_hooks(config)

    assert isinstance(hooks, ProductionHy3HardwareHooks)
    assert isinstance(hooks.config, Hy3HardwareConfig)
    assert hooks.config.allocator_headroom_bytes == 1024**3
    assert hooks.config.model_root == tmp_path / "Hy3-4bit"
    assert hooks.hold_samples == 3
    assert callable(hooks.load_static_lane)
    assert callable(hooks.load_dynamic_lane)


def test_hardware_config_requires_exact_external_artifact_pins(
    tmp_path: Path,
) -> None:
    base = {
        "repo_root": str(tmp_path),
        "model_root": str(tmp_path / "Hy3-4bit"),
        "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
        "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
    }

    with pytest.raises(ArmObservationError, match="artifact_pins"):
        Hy3HardwareConfig.from_mapping(base)

    parsed = Hy3HardwareConfig.from_mapping({**base, "artifact_pins": _artifact_pins()})
    assert parsed.artifact_pins.sidecar_file == "experts.bin"
    for invalid_headroom in (-1, 0, 2 * 1024**3):
        with pytest.raises(ArmObservationError, match="allocator_headroom_bytes"):
            Hy3HardwareConfig.from_mapping(
                {
                    **base,
                    "artifact_pins": _artifact_pins(),
                    "allocator_headroom_bytes": invalid_headroom,
                }
            )


def test_hardware_arms_pin_one_gib_headroom_and_dynamic_slot_counts(
    tmp_path: Path,
) -> None:
    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
            "artifact_pins": _artifact_pins(),
            "allocator_headroom_bytes": 1024**3,
        }
    )

    static = hardware_module._build_runtime_config(config, arm="static").memory_plan(
        HY3_Q4
    )
    dynamic = hardware_module._build_runtime_config(config, arm="dynamic").memory_plan(
        HY3_Q4
    )

    assert (
        static.allocator_headroom_bytes == dynamic.allocator_headroom_bytes == 1024**3
    )
    assert static.persistent_slots == 8_673
    assert static.unallocated_bytes == 1_076_826_880
    assert dynamic.persistent_slots == 9_696
    assert dynamic.persistent_slots // config.expert_slab_slots == 303
    assert dynamic.unallocated_bytes == 1_288_770_304


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
            "artifact_pins": _artifact_pins(),
        }
    )
    expected = {
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW": "0",
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
    hostile = {
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW": "2048",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "fast_sdpa_gather",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "0",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "1",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "64",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE": "async_per_head",
        "MTPLX_PAGED_GQA_SDPA_ROUTE": "grouped",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA": "1",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT": "1",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q": "1",
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q": "4096",
        "MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q": "4096",
        "MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE": "64",
        "MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE": "32",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    observed: dict[str, dict[str, str | None]] = {}

    for arm in ("static", "dynamic"):
        lease = hardware_module._EnvironmentLease(
            hardware_module._arm_environment(arm, config)
        )
        lease.acquire()
        try:
            observed[arm] = {name: os.environ.get(name) for name in expected}
        finally:
            lease.release()

    assert observed == {"static": expected, "dynamic": expected}
    assert {name: os.environ[name] for name in hostile} == hostile


def test_hardware_lane_stages_dynamic_logical_admission_until_q4_growth() -> None:
    assert hardware_module._initial_kv_admission_tokens("static", 4096) == 4096
    assert hardware_module._initial_kv_admission_tokens("dynamic", 4096) == 1


def test_static_control_budget_charges_exact_physical_q4_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.runtime as runtime_module

    class ModelLoadBoundary(RuntimeError):
        pass

    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
            "artifact_pins": _artifact_pins(),
        }
    )
    observed: dict[str, object] = {}
    frozen_manifest = object()
    frozen_model_config = {"model_type": "hy_v3"}
    monkeypatch.setattr(
        hardware_module,
        "_attest_config_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest=frozen_manifest,
            model_config=frozen_model_config,
        ),
    )

    def stop_at_model_load(*_args: object, **kwargs: object) -> object:
        observed["config"] = kwargs["expert_streaming_config"]
        observed["expert_manifest"] = kwargs["expert_manifest"]
        observed["model_config"] = kwargs["model_config"]
        raise ModelLoadBoundary

    monkeypatch.setattr(runtime_module, "load", stop_at_model_load)
    hooks = ProductionHy3HardwareHooks(config)

    with pytest.raises(ModelLoadBoundary):
        hooks.load_static_lane(
            ArmRequest(arm="static", context_tokens=4096, repetition=0)
        )

    runtime_config = observed["config"]
    assert observed["expert_manifest"] is frozen_manifest
    assert observed["model_config"] is frozen_model_config
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
            "artifact_pins": _artifact_pins(),
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
        artifact_pins_sha256="3" * 64,
        artifact_stat_sha256="4" * 64,
        initial_artifact_evidence={},
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


def test_dynamic_lane_extends_logical_admission_only_after_exact_q4_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Hy3HardwareConfig.from_mapping(
        {
            "repo_root": str(tmp_path),
            "model_root": str(tmp_path / "Hy3-4bit"),
            "manifest": str(tmp_path / "Hy3-4bit" / "expert-manifest.json"),
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision",
            "artifact_pins": _artifact_pins(),
        }
    )
    cache = [FakeQ4Entry([], "a"), FakeQ4Entry([], "b")]
    for entry in cache:
        entry.num_blocks = 256
        entry.nbytes = 256 * (HY3_Q4_KV_BLOCK_BYTES // 2)

    class Admission:
        def __init__(self, tokens: int) -> None:
            self.tokens = tokens
            self.growth_targets: list[int] = []
            self.release_calls = 0

        def grow_to(self, tokens: int) -> None:
            assert all(entry.num_blocks == 256 for entry in cache)
            self.tokens = tokens
            self.growth_targets.append(tokens)

        def release(self) -> None:
            self.release_calls += 1

    def admit(_tokens: int) -> Admission:
        raise AssertionError("hardware lane must grow its existing KV admission")

    runtime = SimpleNamespace(admit_kv_tokens=admit)
    sentinel_logits = object()
    monkeypatch.setattr(
        hardware_module,
        "_forward_prefill",
        lambda *_args, **_kwargs: sentinel_logits,
    )
    admission = Admission(1)
    lane = MlxHy3HardwareLane(
        config=config,
        request=ArmRequest(arm="dynamic", context_tokens=4096, repetition=0),
        runtime=runtime,
        runtime_config=object(),
        manifest=object(),
        prompt_ids=[11, 22],
        cache=cache,
        logits=object(),
        admission=admission,
        environment=object(),
        model_artifact_sha256="1" * 64,
        manifest_file_sha256="2" * 64,
        source_git_commit="a" * 40,
        artifact_pins_sha256="3" * 64,
        artifact_stat_sha256="4" * 64,
        initial_artifact_evidence={},
    )

    lane.prepare_q4_context(4096)

    assert admission.tokens == 4096
    assert admission.growth_targets == [4096]
    assert lane.logits is sentinel_logits

    lane._release_admission()
    lane._release_admission()
    assert admission.release_calls == 1


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
            "artifact_pins": _artifact_pins(),
        }
    )
    cache = [FakeQ4Entry([], "a"), FakeQ4Entry([], "b")]
    for entry in cache:
        entry.num_blocks = HY3_Q4_MAX_BLOCKS
        entry.nbytes = HY3_Q4_MAX_BLOCKS * (HY3_Q4_KV_BLOCK_BYTES // 2)
    slot_snapshot = {
        "slabs": {"physical_bytes": 800},
        "metrics": {
            "active_routes": 0,
            "completion_fence_failures": 0,
            "global_device_synchronizations": 0,
        },
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
                "io": {"integrity_errors": 0},
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
        artifact_pins_sha256="3" * 64,
        artifact_stat_sha256="4" * 64,
        initial_artifact_evidence={},
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
            "artifact_pins": _artifact_pins(),
        }
    )
    cache = [FakeQ4Entry([], index) for index in range(80)]
    slot_snapshot = {
        "slabs": {"physical_bytes": 800},
        "metrics": {
            "active_routes": 0,
            "completion_fence_failures": 0,
            "global_device_synchronizations": 0,
        },
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
                "io": {"integrity_errors": 7},
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
        artifact_pins_sha256="3" * 64,
        artifact_stat_sha256="4" * 64,
        initial_artifact_evidence={},
    )

    ledger = lane._live_physical_ledger()

    expected = {
        "operating_target_bytes": 110 * hardware_module.GIB,
        "hard_ceiling_bytes": 112 * hardware_module.GIB,
        "allocator_headroom_bytes": hardware_module.GIB,
        "classified_target_bytes": 109 * hardware_module.GIB,
        "classified_bytes": 950 + HY3_Q4_KV_BLOCK_BYTES,
        "charged_bytes": 960 + HY3_Q4_KV_BLOCK_BYTES,
        "charged_residual_bytes": (
            110 * hardware_module.GIB - 960 - HY3_Q4_KV_BLOCK_BYTES
        ),
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
    assert ledger["slot_health"]["integrity_errors"] == 7
