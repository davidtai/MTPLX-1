from __future__ import annotations

import inspect
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from mtplx import expert_runtime as expert_runtime_module
from mtplx.expert_runtime import (
    ExpertStreamingConfig,
    ExpertStreamingConfigurationError,
    ExpertStreamingRuntime,
)
from mtplx.expert_slots import (
    ExpertSlabReclaimError,
    ExpertSlabReclaimResult,
)
from mtplx.expert_streaming import GlobalExpertSlotBank
from mtplx.expert_streaming_models import HY3_Q4
from mtplx.memory_broker import (
    BINARY_GIB,
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BLOCK_TOKENS,
    AllocatorMemorySample,
    BrokerSnapshot,
    MemoryAdmissionError,
    MemoryBudget,
    MemoryTelemetryError,
    UnifiedMemoryBroker,
)
from mtplx.mtp_patch import MTPContract
from mtplx import runtime as runtime_module
from mtplx.runtime import MTPLXRuntime


def _snapshot(*, resident: int, experts: int) -> BrokerSnapshot:
    return BrokerSnapshot(
        resident_model_bytes=resident,
        kv_physical_bytes=0,
        expert_slab_physical_bytes=experts,
        in_flight_expert_staging_bytes=0,
        runtime_workspace_bytes=0,
        allocator_cache_bytes=0,
    )


class _FakeBank:
    def __init__(self) -> None:
        self.ranked_with: tuple[dict[int, tuple[int, ...]], tuple[int, ...]] | None = (
            None
        )
        self.deactivated: list[tuple[int, ...]] = []
        self.activated: list[tuple[int, ...]] = []

    def rank_reclaim_slabs(self, slabs, protected_slots=()):
        normalized = {int(key): tuple(value) for key, value in slabs.items()}
        protected = tuple(protected_slots)
        self.ranked_with = (normalized, protected)
        return tuple(
            slab_id
            for slab_id, slot_ids in normalized.items()
            if not set(slot_ids).intersection(protected)
        )

    def deactivate_slots(self, slot_ids):
        normalized = tuple(slot_ids)
        self.deactivated.append(normalized)
        return ()

    def activate_slots(self, slot_ids):
        self.activated.append(tuple(slot_ids))


class _FakeSlots:
    def __init__(self, *, physical_bytes: int = 64) -> None:
        self._physical_bytes = physical_bytes
        self.prepared: list[tuple[tuple[int, ...], int]] = []
        self.regrown: list[int] = []
        self.poison_full_snapshot = False

    def slab_layout(self) -> dict[int, tuple[int, ...]]:
        return {0: (0, 1), 1: (2, 3)}

    def protected_slot_ids(self) -> tuple[int, ...]:
        return (2,)

    def prepare_slab_reclaim(
        self,
        slab_ids,
        *,
        requested_bytes: int,
        deadline_ns: int | None = None,
    ):
        del deadline_ns
        normalized = tuple(slab_ids)
        self.prepared.append((normalized, requested_bytes))
        return SimpleNamespace(slab_ids=(0,))

    def commit_slab_reclaim(self, ticket) -> ExpertSlabReclaimResult:
        assert ticket.slab_ids == (0,)
        self._physical_bytes -= 32
        return ExpertSlabReclaimResult(
            slab_ids=(0,),
            released_slot_ids=(0, 1),
            physical_bytes=32,
        )

    def abort_slab_reclaim(self, _ticket) -> None:
        raise AssertionError("a successful reclaim must not abort")

    def released_slab_ids(self) -> tuple[int, ...]:
        if self._physical_bytes == 0:
            return (0, 1)
        return (0,) if self._physical_bytes == 32 else ()

    def slot_ids_for_slab(self, slab_id: int) -> tuple[int, ...]:
        return self.slab_layout()[slab_id]

    def regrow_slab(self, slab_id: int) -> None:
        self.regrown.append(slab_id)
        self._physical_bytes += 32

    def snapshot(self) -> dict[str, object]:
        if self.poison_full_snapshot:
            raise AssertionError("unrelated completion fence was drained")
        return {"slabs": {"physical_bytes": self._physical_bytes}}

    def expert_slab_telemetry_snapshot(self) -> dict[str, int]:
        physical_slabs = self._physical_bytes // 32
        return {
            "logical_slab_count": 2,
            "active_slab_count": physical_slabs,
            "draining_slab_count": 0,
            "released_slab_count": 2 - physical_slabs,
            "logical_slot_count": 4,
            "active_slot_count": physical_slabs * 2,
            "resident_record_count": 0,
            "in_flight_record_count": 0,
            "pinned_record_count": 0,
            "pin_count": 0,
            "logical_bytes": 64,
            "physical_bytes": self._physical_bytes,
            "released_bytes": 64 - self._physical_bytes,
            "in_flight_bytes": 0,
            "pinned_bytes": 0,
        }


def _runtime(
    broker: UnifiedMemoryBroker,
    *,
    slots: _FakeSlots,
    bank: _FakeBank,
    samples: list[AllocatorMemorySample],
    initial_sample: AllocatorMemorySample | None = None,
) -> ExpertStreamingRuntime:
    runtime = object.__new__(ExpertStreamingRuntime)
    runtime.memory_broker = broker
    runtime.slots = slots
    runtime._global_bank = bank
    runtime.spec = SimpleNamespace(expert_record_bytes=16)
    runtime.plan = SimpleNamespace(persistent_slots=4)
    runtime.config = SimpleNamespace(
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ms=0,
    )
    runtime._dynamic_resize_lock = threading.RLock()
    runtime._memory_transaction_lock = threading.RLock()
    runtime._dynamic_resize_metrics = {}
    runtime._cleanup_error_lock = threading.Lock()
    runtime._cleanup_error = None
    runtime._pending_physical_kv_lock = threading.Lock()
    runtime._pending_physical_kv_caches = {}
    snapshot = broker.snapshot()
    runtime._last_allocator_sample = initial_sample or AllocatorMemorySample(
        active_bytes=(
            snapshot.resident_model_bytes
            + snapshot.expert_slab_physical_bytes
            + snapshot.in_flight_expert_staging_bytes
        ),
        cache_bytes=snapshot.allocator_cache_bytes,
        peak_bytes=snapshot.charged_bytes,
    )

    def sample_allocator_memory() -> AllocatorMemorySample:
        sample = samples.pop(0)
        runtime._last_allocator_sample = sample
        return sample

    runtime._sample_allocator_memory = sample_allocator_memory
    return runtime


class _StrictNonReentrantLock:
    def __init__(self) -> None:
        self._owner: int | None = None
        self.acquisitions = 0

    def acquire(self) -> bool:
        owner = threading.get_ident()
        if self._owner == owner:
            raise AssertionError("global route lock was reacquired")
        assert self._owner is None
        self._owner = owner
        self.acquisitions += 1
        return True

    def release(self) -> None:
        assert self._owner == threading.get_ident()
        self._owner = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def test_real_route_demand_regrows_one_clean_slab_before_policy_planning() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=0))
    slots = _FakeSlots(physical_bytes=0)
    bank = GlobalExpertSlotBank(
        layer_indices=(1,),
        expert_count=4,
        persistent_slots=4,
        transient_slots=1,
        prefill_slots_per_layer=4,
    )
    bank.deactivate_slots((0, 1, 2, 3))
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(0, 0, 0),
            AllocatorMemorySample(32, 0, 32),
        ],
    )
    route_lock = _StrictNonReentrantLock()
    runtime._layer_locks = {1: route_lock}
    runtime._closed = False
    runtime._closing = False
    runtime._raise_if_unhealthy = lambda: None
    runtime._observe_plan = lambda *_args: None
    slots.ensure_route = lambda _layer, plan, **_kwargs: SimpleNamespace(
        plan=plan,
        release=lambda **_release_kwargs: None,
    )

    ready = runtime.ensure_route(1, (0,), phase="decode")

    assert ready.plan.experts == (0,)
    assert slots.regrown == [0]
    assert bank.active_capacity == 2
    assert route_lock.acquisitions == 2


def test_route_regrow_uses_actual_unique_demand_beyond_model_top_k() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = _FakeSlots(physical_bytes=32)
    bank = GlobalExpertSlotBank(
        layer_indices=(1,),
        expert_count=4,
        persistent_slots=4,
        transient_slots=1,
        prefill_slots_per_layer=4,
    )
    bank.deactivate_slots((0, 1, 2))
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    assert tuple(runtime._regrow_for_route_demand((0, 1, 2))) == (0, 1, 2)

    assert slots.regrown == [0]
    assert bank.active_capacity == 3


def test_route_pressure_lazily_regrows_after_large_kv_reclaim() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = _FakeSlots(physical_bytes=32)
    bank = GlobalExpertSlotBank(
        layer_indices=(1,),
        expert_count=4,
        persistent_slots=4,
        transient_slots=1,
        prefill_slots_per_layer=4,
    )
    for expert in range(4):
        plan, transaction = bank.plan_transaction(1, (expert,), phase="decode")
        transaction.commit()
    bank.deactivate_slots((0, 1))
    assert bank.occupancy == bank.active_capacity == 2
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    assert tuple(runtime._regrow_for_route_demand((2,))) == (2,)

    assert slots.regrown == [0]
    assert bank.active_capacity == 4


def test_split_route_demand_regrows_before_acquiring_non_reentrant_policy_lock() -> (
    None
):
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=0))
    slots = _FakeSlots(physical_bytes=0)
    bank = GlobalExpertSlotBank(
        layer_indices=(1,),
        expert_count=4,
        persistent_slots=4,
        transient_slots=1,
        prefill_slots_per_layer=4,
    )
    bank.deactivate_slots((0, 1, 2, 3))
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(0, 0, 0),
            AllocatorMemorySample(32, 0, 32),
        ],
    )
    route_lock = _StrictNonReentrantLock()
    runtime._layer_locks = {1: route_lock}
    runtime._closed = False
    runtime._closing = False
    runtime._raise_if_unhealthy = lambda: None

    def observe_plan(*_args, **_kwargs):
        assert bank.active_capacity == 2
        raise RuntimeError("stop after demand-regrow observation")

    runtime._plan_route_transaction = observe_plan

    with pytest.raises(RuntimeError, match="stop after demand-regrow"):
        runtime.begin_split_route(1, (0,), phase="decode")

    assert slots.regrown == [0]
    assert route_lock.acquisitions == 2


def test_registered_slab_telemetry_does_not_drain_unrelated_completion_fence() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
    )
    slots = _FakeSlots()
    slots.poison_full_snapshot = True
    runtime = _runtime(broker, slots=slots, bank=_FakeBank(), samples=[])

    assert runtime._registered_expert_slab_bytes() == 64


def test_dynamic_broker_initialization_charges_all_pools_additively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        expert_slab_slots=2,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    spec = SimpleNamespace(expert_record_bytes=16)
    plan = SimpleNamespace(
        persistent_slots=4,
        resident_bytes=10,
        transient_bytes=20,
        io_staging_bytes=30,
        runtime_reserve_bytes=40,
        execution_workspace_bytes=50,
    )
    slots = _FakeSlots()
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 64,
            "cache_memory_bytes": 7,
            "peak_memory_bytes": 71,
        },
    )

    broker = ExpertStreamingRuntime._initialize_dynamic_memory_broker(
        config=config,
        spec=spec,
        plan=plan,
        slots=slots,
        mx_module=object(),
    )

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == 10
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.in_flight_expert_staging_bytes == 50
    assert snapshot.runtime_workspace_bytes == 90
    assert snapshot.allocator_cache_bytes == 7
    assert snapshot.charged_bytes == 221
    assert broker.initial_allocator_sample == AllocatorMemorySample(64, 7, 71)


def test_dynamic_broker_initialization_charges_unclassified_active_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        expert_slab_slots=2,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    spec = SimpleNamespace(expert_record_bytes=16)
    plan = SimpleNamespace(
        persistent_slots=4,
        resident_bytes=10,
        transient_bytes=20,
        io_staging_bytes=30,
        runtime_reserve_bytes=40,
        execution_workspace_bytes=50,
    )
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 300,
            "cache_memory_bytes": 7,
            "peak_memory_bytes": 307,
        },
    )

    broker = ExpertStreamingRuntime._initialize_dynamic_memory_broker(
        config=config,
        spec=spec,
        plan=plan,
        slots=_FakeSlots(),
        mx_module=object(),
    )

    snapshot = broker.snapshot()
    assert snapshot.allocator_cache_bytes == 93
    assert snapshot.charged_bytes == 307


def test_dynamic_broker_rejects_initial_allocator_footprint_above_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        expert_slab_slots=2,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    spec = SimpleNamespace(expert_record_bytes=16)
    plan = SimpleNamespace(
        persistent_slots=4,
        resident_bytes=10,
        transient_bytes=0,
        io_staging_bytes=0,
        runtime_reserve_bytes=0,
        execution_workspace_bytes=0,
    )
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 112 * BINARY_GIB,
            "cache_memory_bytes": 0,
            "peak_memory_bytes": 112 * BINARY_GIB,
        },
    )

    with pytest.raises(ExpertStreamingConfigurationError, match="110 GiB"):
        ExpertStreamingRuntime._initialize_dynamic_memory_broker(
            config=config,
            spec=spec,
            plan=plan,
            slots=_FakeSlots(),
            mx_module=object(),
        )


def test_dynamic_broker_rejects_missing_or_mismatched_physical_slab_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        expert_slab_slots=2,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    spec = SimpleNamespace(expert_record_bytes=16)
    plan = SimpleNamespace(
        persistent_slots=4,
        resident_bytes=10,
        transient_bytes=0,
        io_staging_bytes=0,
        runtime_reserve_bytes=0,
        execution_workspace_bytes=0,
    )
    slots = _FakeSlots(physical_bytes=0)
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 0,
            "cache_memory_bytes": 0,
            "peak_memory_bytes": 0,
        },
    )

    with pytest.raises(ExpertStreamingConfigurationError, match="physical slab"):
        ExpertStreamingRuntime._initialize_dynamic_memory_broker(
            config=config,
            spec=spec,
            plan=plan,
            slots=slots,
            mx_module=object(),
        )


def test_dynamic_broker_rejects_aggregate_bytes_from_malformed_slab_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedSlots(_FakeSlots):
        def slab_layout(self) -> dict[int, tuple[int, ...]]:
            return {0: (0, 1, 2, 3)}

    config = SimpleNamespace(
        expert_slab_slots=2,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    spec = SimpleNamespace(expert_record_bytes=16)
    plan = SimpleNamespace(
        persistent_slots=4,
        resident_bytes=10,
        transient_bytes=0,
        io_staging_bytes=0,
        runtime_reserve_bytes=0,
        execution_workspace_bytes=0,
    )
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 64,
            "cache_memory_bytes": 0,
            "peak_memory_bytes": 64,
        },
    )

    with pytest.raises(ExpertStreamingConfigurationError, match="physical slab"):
        ExpertStreamingRuntime._initialize_dynamic_memory_broker(
            config=config,
            spec=spec,
            plan=plan,
            slots=MalformedSlots(),
            mx_module=object(),
        )


def test_dynamic_broker_rejects_additive_initial_pools_above_110_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        expert_slab_slots=2,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    spec = SimpleNamespace(expert_record_bytes=16)
    plan = SimpleNamespace(
        persistent_slots=4,
        resident_bytes=110 * BINARY_GIB,
        transient_bytes=0,
        io_staging_bytes=0,
        runtime_reserve_bytes=1,
        execution_workspace_bytes=0,
    )
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 64,
            "cache_memory_bytes": 1,
            "peak_memory_bytes": 65,
        },
    )

    with pytest.raises(ExpertStreamingConfigurationError, match="110 GiB"):
        ExpertStreamingRuntime._initialize_dynamic_memory_broker(
            config=config,
            spec=spec,
            plan=plan,
            slots=_FakeSlots(),
            mx_module=object(),
        )


def test_post_load_reconciliation_charges_measured_resident_and_mtp_memory() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=10,
            kv_physical_bytes=0,
            expert_slab_physical_bytes=64,
            in_flight_expert_staging_bytes=20,
            runtime_workspace_bytes=30,
            allocator_cache_bytes=5,
        ),
        expert_slab_bytes=32,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(120, 7, 127)],
    )
    runtime.plan = SimpleNamespace(
        persistent_slots=4,
        transient_bytes=20,
        execution_workspace_bytes=4,
    )

    runtime.reconcile_post_load_memory()

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == 36
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.in_flight_expert_staging_bytes == 20
    assert snapshot.runtime_workspace_bytes == 4
    assert snapshot.allocator_cache_bytes == 7


def test_post_load_reconciliation_retains_unused_reserve_and_reclassifies_cache() -> (
    None
):
    broker = UnifiedMemoryBroker(
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=10,
            kv_physical_bytes=0,
            expert_slab_physical_bytes=64,
            in_flight_expert_staging_bytes=50,
            runtime_workspace_bytes=30,
            allocator_cache_bytes=5,
        ),
        expert_slab_bytes=32,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(99, 3, 102)],
        initial_sample=AllocatorMemorySample(94, 5, 99),
    )
    runtime.plan = SimpleNamespace(
        persistent_slots=4,
        transient_bytes=20,
        io_staging_bytes=30,
        execution_workspace_bytes=4,
    )

    runtime.reconcile_post_load_memory()

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == 15
    assert snapshot.in_flight_expert_staging_bytes == 50
    assert snapshot.runtime_workspace_bytes == 25
    assert snapshot.allocator_cache_bytes == 3


def test_post_load_reconciliation_never_drops_planned_resident_bytes() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=10,
            kv_physical_bytes=0,
            expert_slab_physical_bytes=64,
            in_flight_expert_staging_bytes=20,
            runtime_workspace_bytes=30,
            allocator_cache_bytes=5,
        ),
        expert_slab_bytes=32,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(50, 5, 55)],
    )
    runtime.plan = SimpleNamespace(
        persistent_slots=4,
        transient_bytes=20,
        execution_workspace_bytes=4,
    )

    runtime.reconcile_post_load_memory()

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == 10
    assert snapshot.runtime_workspace_bytes == 30


def test_post_load_reconciliation_rejects_resident_growth_beyond_reserve() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=10,
            kv_physical_bytes=0,
            expert_slab_physical_bytes=64,
            in_flight_expert_staging_bytes=20,
            runtime_workspace_bytes=30,
            allocator_cache_bytes=5,
        ),
        expert_slab_bytes=32,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(130, 7, 137)],
    )
    runtime.plan = SimpleNamespace(
        persistent_slots=4,
        transient_bytes=20,
        execution_workspace_bytes=4,
    )

    with pytest.raises(ExpertStreamingConfigurationError, match="runtime reserve"):
        runtime.reconcile_post_load_memory()

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == 46
    assert snapshot.runtime_workspace_bytes == 4
    assert snapshot.allocator_cache_bytes == 7


def test_runtime_load_calls_post_load_reconciliation_after_mtp_injection() -> None:
    source = inspect.getsource(runtime_module.load)

    injection = source.index("mtp_enabled = inject_hy3_streamed_mtp_support")
    attention_setup = source.index("configure_split_full_attention(model)")
    native_setup = source.index("configure_native_mlp(model)")
    selfcheck = source.index("maybe_run_model_selfcheck(model)")
    adapter_setup = source.index("adapter_merge_report = merge_installed_mtp_lora")
    reconciliation = source.index("expert_runtime.reconcile_post_load_memory()")
    returned = source.index("return MTPLXRuntime")

    assert (
        max(
            injection,
            attention_setup,
            native_setup,
            selfcheck,
            adapter_setup,
        )
        < reconciliation
        < returned
    )


def test_dynamic_memory_telemetry_reports_complete_bounded_resize_state() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ns=1_000_000_000,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    bank = _FakeBank()
    bank.active_capacity = 2
    bank.occupancy = 1
    runtime = _runtime(
        broker,
        slots=_FakeSlots(physical_bytes=32),
        bank=bank,
        samples=[],
    )
    runtime.config = SimpleNamespace(
        resource_telemetry=False,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ms=1000,
    )
    runtime._last_allocator_sample = AllocatorMemorySample(40, 3, 45)
    runtime._dynamic_resize_metrics = {
        "reclaim_requests": 2,
        "regrow_requests": 1,
        "requested_reclaim_bytes": 64,
        "reclaimed_bytes": 32,
        "regrown_bytes": 32,
        "resize_operations": 2,
        "resize_failures": 1,
        "blocked_by_pin_bytes": 16,
        "last_resize_duration_ns": 7,
        "total_resize_duration_ns": 11,
        "max_resize_duration_ns": 7,
    }

    telemetry = runtime.dynamic_memory_telemetry_snapshot(now_ns=5)

    assert telemetry["operating_target_bytes"] == 110 * BINARY_GIB
    assert telemetry["hard_ceiling_bytes"] == 112 * BINARY_GIB
    assert telemetry["logical_expert_records"] == 4
    assert telemetry["active_expert_records"] == 2
    assert telemetry["resident_expert_records"] == 1
    assert telemetry["logical_slab_count"] == 2
    assert telemetry["active_slab_count"] == 1
    assert telemetry["released_slab_count"] == 1
    assert telemetry["expert_slab_physical_bytes"] == 32
    assert telemetry["pinned_expert_bytes"] == 0
    assert telemetry["in_flight_expert_bytes"] == 0
    assert telemetry["speculative_expert_bytes"] == 0
    assert telemetry["requested_reclaim_bytes"] == 64
    assert telemetry["reclaimed_bytes"] == 32
    assert telemetry["resize_duration_ns"] == 7
    assert telemetry["blocked_by_pin_bytes"] == 16
    assert telemetry["resize_failures"] == 1
    assert telemetry["hysteresis_slabs"] == 1
    assert telemetry["minimum_resize_interval_ns"] == 1_000_000_000
    assert telemetry["allocator_active_bytes"] == 40
    assert telemetry["allocator_cache_bytes"] == 3
    assert telemetry["allocator_peak_bytes"] == 45
    assert set(runtime._dynamic_resize_metrics) == {
        "reclaim_requests",
        "regrow_requests",
        "requested_reclaim_bytes",
        "reclaimed_bytes",
        "regrown_bytes",
        "resize_operations",
        "resize_failures",
        "blocked_by_pin_bytes",
        "last_resize_duration_ns",
        "total_resize_duration_ns",
        "max_resize_duration_ns",
    }


def test_dynamic_memory_telemetry_uses_live_pin_and_speculation_truth() -> None:
    class PinnedSlots(_FakeSlots):
        def expert_slab_telemetry_snapshot(self) -> dict[str, int]:
            telemetry = super().expert_slab_telemetry_snapshot()
            telemetry["pinned_record_count"] = 1
            telemetry["pin_count"] = 2
            telemetry["pinned_bytes"] = 16
            return telemetry

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
    )
    bank = _FakeBank()
    bank.active_capacity = 4
    bank.occupancy = 2
    bank.speculative_record_count = 1
    runtime = _runtime(
        broker,
        slots=PinnedSlots(),
        bank=bank,
        samples=[],
    )
    runtime._dynamic_resize_metrics = {
        "reclaim_requests": 0,
        "regrow_requests": 0,
        "requested_reclaim_bytes": 0,
        "reclaimed_bytes": 0,
        "regrown_bytes": 0,
        "resize_operations": 0,
        "resize_failures": 0,
        "blocked_by_pin_bytes": 0,
        "last_resize_duration_ns": 0,
        "total_resize_duration_ns": 0,
        "max_resize_duration_ns": 0,
    }

    telemetry = runtime.dynamic_memory_telemetry_snapshot()

    assert broker.snapshot().pinned_expert_bytes == 0
    assert broker.snapshot().speculative_expert_bytes == 0
    assert telemetry["pinned_expert_bytes"] == 16
    assert telemetry["speculative_expert_bytes"] == 16


def test_resource_telemetry_publishes_q4_kv_geometry_for_health() -> None:
    class ResourceSlots(_FakeSlots):
        def resource_telemetry_snapshot(self) -> dict[str, object]:
            return {}

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
    )
    ticket = broker.plan_kv_growth(
        cache_id="target:telemetry",
        steady_delta_bytes=HY3_Q4_KV_BLOCK_BYTES,
        transient_delta_bytes=0,
    )
    broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=HY3_Q4_KV_BLOCK_BYTES,
    )
    runtime = _runtime(
        broker,
        slots=ResourceSlots(),
        bank=_FakeBank(),
        samples=[],
    )
    runtime.spec = HY3_Q4
    runtime._counter_lock = threading.Lock()
    runtime.counters = SimpleNamespace(as_dict=lambda: {})
    runtime._layer_counters = {}
    runtime._phase_counters = {}
    runtime._incremental_miss_routes = 0
    runtime._incremental_miss_parts = 0
    runtime._pipeline_ledger = None
    runtime._kv_lock = threading.Lock()
    runtime._live_kv_tokens = HY3_Q4_KV_BLOCK_TOKENS
    runtime._live_kv_peak = HY3_Q4_KV_BLOCK_TOKENS
    runtime._dynamic_resize_metrics = {
        "reclaim_requests": 0,
        "regrow_requests": 0,
        "requested_reclaim_bytes": 0,
        "reclaimed_bytes": 0,
        "regrown_bytes": 0,
        "resize_operations": 0,
        "resize_failures": 0,
        "blocked_by_pin_bytes": 0,
        "last_resize_duration_ns": 0,
        "total_resize_duration_ns": 0,
        "max_resize_duration_ns": 0,
    }

    telemetry = runtime.resource_telemetry_snapshot(mx_module=object())

    assert telemetry["kv"] == {
        "representation": "q4",
        "logical_tokens": HY3_Q4_KV_BLOCK_TOKENS,
        "physical_blocks": 1,
        "physical_bytes": HY3_Q4_KV_BLOCK_BYTES,
    }


def test_kv_admission_reconciles_live_pinned_expert_bytes() -> None:
    class PinnedSlots(_FakeSlots):
        def expert_slab_telemetry_snapshot(self) -> dict[str, int]:
            telemetry = super().expert_slab_telemetry_snapshot()
            telemetry["pinned_record_count"] = 2
            telemetry["pin_count"] = 2
            telemetry["pinned_bytes"] = 32
            return telemetry

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    runtime = _runtime(
        broker,
        slots=PinnedSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(target, 0, target)],
    )

    with pytest.raises(MemoryAdmissionError, match="pinned expert bytes"):
        runtime.reserve_growth(
            cache_id="target:pinned",
            steady_delta_bytes=64,
            transient_delta_bytes=0,
        )

    snapshot = broker.snapshot()
    assert snapshot.pinned_expert_bytes == 32
    assert snapshot.pending_kv_ticket_id is None


def test_dynamic_plan_is_zero_kv_slab_aligned_and_bounded_by_110_gib() -> None:
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=112 * BINARY_GIB,
        max_live_kv_tokens=131_072,
        runtime_reserve_bytes=8 * BINARY_GIB,
        transient_slots=32,
        slot_layout="component-banks",
        cache_scope="global",
        dynamic_expert_slabs=True,
        expert_slab_slots=32,
    )

    plan = config.memory_plan(HY3_Q4)

    assert plan.total_limit_bytes == 110 * BINARY_GIB
    assert plan.context_tokens == 0
    assert plan.kv_bytes == 0
    assert plan.persistent_slots == 9_792
    assert plan.persistent_slots % config.expert_slab_slots == 0
    assert plan.allocated_bytes <= 110 * BINARY_GIB


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_key": "glm52-q4"},
        {"cache_scope": "layer"},
        {"slot_layout": "direct-slots"},
        {"expert_slab_slots": 0},
    ],
)
def test_dynamic_plan_rejects_unsupported_lanes(overrides) -> None:
    values = {
        "model_key": "hy3-q4",
        "memory_limit_bytes": 112 * BINARY_GIB,
        "max_live_kv_tokens": 131_072,
        "runtime_reserve_bytes": 8 * BINARY_GIB,
        "transient_slots": 32,
        "slot_layout": "component-banks",
        "cache_scope": "global",
        "dynamic_expert_slabs": True,
        "expert_slab_slots": 32,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        ExpertStreamingConfig(**values)


def test_reclaim_uses_ranked_unprotected_slab_and_confirms_broker_ticket() -> None:
    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
        cache_id="target",
    )
    slots = _FakeSlots()
    bank = _FakeBank()
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(64, 0, 64),
            AllocatorMemorySample(32, 0, 64),
        ],
    )

    result = runtime.reclaim_expert_bytes(ticket, now_ns=1)

    assert result.physical_bytes == 32
    assert slots.prepared == [((0,), 32)]
    assert bank.ranked_with == ({0: (0, 1), 1: (2, 3)}, (2,))
    assert bank.deactivated == [(0, 1)]
    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.pending_kv_ticket_id == ticket.ticket_id


def test_kv_observer_reserves_reclaims_and_commits_exact_growth() -> None:
    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    slots = _FakeSlots()
    bank = _FakeBank()
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(target, 0, target),
            AllocatorMemorySample(64, 0, 64),
            AllocatorMemorySample(32, 0, 64),
        ],
    )

    ticket = runtime.reserve_growth(
        cache_id="target:0",
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    allocation = runtime.commit_growth(ticket, measured_physical_bytes=32)

    assert allocation.cache_id == "target:0"
    assert allocation.physical_bytes == 32
    assert bank.deactivated == [(0, 1)]
    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.kv_physical_bytes == 32
    assert snapshot.pending_kv_ticket_id is None


def test_kv_observer_reconciles_allocator_truth_before_planning_growth() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(
            operating_target_bytes=100,
            hard_ceiling_bytes=112,
        ),
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=90,
            kv_physical_bytes=0,
            expert_slab_physical_bytes=0,
            in_flight_expert_staging_bytes=0,
            runtime_workspace_bytes=0,
            allocator_cache_bytes=5,
        ),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(90, 11, 101)],
    )

    with pytest.raises(MemoryAdmissionError, match="operating target"):
        runtime.reserve_growth(
            cache_id="target:stale-before-plan",
            steady_delta_bytes=1,
            transient_delta_bytes=0,
        )

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.allocator_cache_bytes == 11
    assert snapshot.charged_bytes == 101


def test_kv_observer_aborts_ticket_when_reclaim_cannot_start() -> None:
    class ProtectedSlots(_FakeSlots):
        def protected_slot_ids(self) -> tuple[int, ...]:
            return (0, 1, 2, 3)

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    runtime = _runtime(
        broker,
        slots=ProtectedSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(target, 0, target)],
    )

    with pytest.raises(MemoryAdmissionError, match="no unprotected"):
        runtime.reserve_growth(
            cache_id="target:0",
            steady_delta_bytes=32,
            transient_delta_bytes=0,
        )

    assert broker.snapshot().pending_kv_ticket_id is None


def test_kv_observer_release_reconciles_exact_owner_batch() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(100, 0, 100)],
    )
    ticket = runtime.reserve_growth(
        cache_id="target:0",
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    allocation = runtime.commit_growth(ticket, measured_physical_bytes=32)
    runtime.maybe_regrow_expert_slabs = lambda **_kwargs: pytest.fail(
        "KV release must not regrow expert slabs"
    )

    runtime.release_cache(
        cache_id="target:0",
        allocations=(allocation,),
        released_physical_bytes=32,
        allocator_before=AllocatorMemorySample(32, 0, 32),
        allocator_after=AllocatorMemorySample(0, 0, 32),
    )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0


def test_runtime_passes_dynamic_kv_observer_to_target_and_mtp_cache(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object, str]] = []

    class Model:
        def make_cache(self):
            return []

        def make_mtp_cache(self):
            return []

    observer = SimpleNamespace(memory_broker=object())
    runtime = MTPLXRuntime(
        model=Model(),
        tokenizer=object(),
        model_path=Path("tiny"),
        mtp_enabled=True,
        contract=MTPContract(),
        expert_streaming=observer,
    )
    monkeypatch.setattr(
        "mtplx.cache_state.configure_owned_recurrent_state_cache",
        lambda _cache: None,
    )
    monkeypatch.setattr(
        "mtplx.cache_state.configure_tail_owned_attention_kv_cache",
        lambda _cache, *, allocation_observer, cache_id_prefix: calls.append(
            ("target", allocation_observer, cache_id_prefix)
        ),
    )
    monkeypatch.setattr(
        "mtplx.cache_state.configure_mtp_attention_kv_cache",
        lambda _cache, *, allocation_observer, cache_id_prefix: calls.append(
            ("mtp", allocation_observer, cache_id_prefix)
        ),
    )

    runtime.make_cache()
    runtime.make_cache()
    runtime.make_mtp_cache()
    runtime.make_mtp_cache()

    assert calls == [
        ("target", observer, "target:0"),
        ("target", observer, "target:1"),
        ("mtp", observer, "mtp:2"),
        ("mtp", observer, "mtp:3"),
    ]


def test_lazy_regrow_confirms_allocator_growth_before_reactivating_slots() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = _FakeSlots(physical_bytes=32)
    bank = _FakeBank()
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    count = runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    assert count == 1
    assert slots.regrown == [0]
    assert bank.activated == [(0, 1)]
    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.last_expert_resize_ns == 5


def test_dynamic_resize_metrics_are_mutated_only_under_resize_lock() -> None:
    class GuardedLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._owner: int | None = None
            self._depth = 0

        @property
        def owned(self) -> bool:
            return self._owner == threading.get_ident()

        def __enter__(self):
            self._lock.acquire()
            self._owner = threading.get_ident()
            self._depth += 1
            return self

        def __exit__(self, *_exc: object) -> None:
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
            self._lock.release()

    class GuardedMetrics(dict[str, int]):
        def __init__(self, guard: GuardedLock) -> None:
            super().__init__()
            self.guard = guard

        def get(self, key: str, default=None):
            assert self.guard.owned, "resize metrics escaped the resize lock"
            return super().get(key, default)

        def __setitem__(self, key: str, value: int) -> None:
            assert self.guard.owned, "resize metrics escaped the resize lock"
            super().__setitem__(key, value)

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[],
    )
    guard = GuardedLock()
    runtime._dynamic_resize_lock = guard
    runtime._dynamic_resize_metrics = GuardedMetrics(guard)

    assert runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5) == 0


def test_reclaim_fails_without_touching_policy_when_every_slab_is_protected() -> None:
    class ProtectedSlots(_FakeSlots):
        def protected_slot_ids(self) -> tuple[int, ...]:
            return (0, 1, 2, 3)

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    slots = ProtectedSlots()
    bank = _FakeBank()
    runtime = _runtime(broker, slots=slots, bank=bank, samples=[])

    with pytest.raises(MemoryAdmissionError, match="no unprotected"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    assert slots.prepared == []
    assert bank.deactivated == []
    assert broker.snapshot().pending_kv_ticket_id == ticket.ticket_id


def test_allocator_retention_keeps_released_expert_bytes_charged_fail_closed() -> None:
    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    slots = _FakeSlots()
    bank = _FakeBank()
    runtime = _runtime(
        broker,
        slots=slots,
        bank=bank,
        samples=[
            AllocatorMemorySample(64, 0, 64),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.allocator_cache_bytes == 32
    assert snapshot.failed_closed is True
    assert bank.deactivated == [(0, 1)]


def test_reclaim_allocator_telemetry_failure_consumes_pending_with_physical_truth() -> (
    None
):
    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    slots = _FakeSlots()
    runtime = _runtime(broker, slots=slots, bank=_FakeBank(), samples=[])
    samples = iter((AllocatorMemorySample(64, 0, 64),))

    def sample_allocator() -> AllocatorMemorySample:
        try:
            return next(samples)
        except StopIteration as exc:
            raise MemoryTelemetryError("injected allocator telemetry failure") from exc

    runtime._sample_allocator_memory = sample_allocator

    with pytest.raises(MemoryTelemetryError, match="allocator telemetry failure"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.allocator_cache_bytes == 32
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_reclaim_registry_telemetry_failure_consumes_pending_with_derived_truth() -> (
    None
):
    class RegistryFailureSlots(_FakeSlots):
        def expert_slab_telemetry_snapshot(self) -> dict[str, int]:
            if self._physical_bytes == 32:
                raise MemoryTelemetryError("injected slab registry telemetry failure")
            return super().expert_slab_telemetry_snapshot()

        def snapshot(self) -> dict[str, object]:
            if self._physical_bytes == 32:
                raise MemoryTelemetryError("injected slab registry telemetry failure")
            return super().snapshot()

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    runtime = _runtime(
        broker,
        slots=RegistryFailureSlots(),
        bank=_FakeBank(),
        samples=[
            AllocatorMemorySample(64, 0, 64),
            AllocatorMemorySample(32, 0, 64),
        ],
    )

    with pytest.raises(MemoryTelemetryError, match="registry telemetry failure"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_partial_reclaim_failure_consumes_pending_and_publishes_confirmed_drop() -> (
    None
):
    class PartialFailureSlots(_FakeSlots):
        def commit_slab_reclaim(self, _ticket) -> ExpertSlabReclaimResult:
            self._physical_bytes -= 32
            result = ExpertSlabReclaimResult(
                slab_ids=(0,),
                released_slot_ids=(0, 1),
                physical_bytes=32,
            )
            raise ExpertSlabReclaimError(
                "injected partial reclaim failure",
                result=result,
                failed_slab_id=1,
                destructive_boundary_crossed=False,
            )

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    runtime = _runtime(
        broker,
        slots=PartialFailureSlots(),
        bank=_FakeBank(),
        samples=[
            AllocatorMemorySample(64, 0, 64),
            AllocatorMemorySample(32, 0, 64),
        ],
    )

    with pytest.raises(ExpertSlabReclaimError, match="partial reclaim"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.allocator_cache_bytes == 0
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_ambiguous_reclaim_consumes_pending_without_claiming_freed_bytes() -> None:
    class AmbiguousFailureSlots(_FakeSlots):
        def commit_slab_reclaim(self, _ticket) -> ExpertSlabReclaimResult:
            raise ExpertSlabReclaimError(
                "injected ambiguous reclaim failure",
                result=ExpertSlabReclaimResult((), (), 0),
                failed_slab_id=0,
                destructive_boundary_crossed=None,
            )

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    runtime = _runtime(
        broker,
        slots=AmbiguousFailureSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(64, 0, 64)],
    )

    with pytest.raises(ExpertSlabReclaimError, match="ambiguous reclaim"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.allocator_cache_bytes == 0
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_empty_message_ambiguous_reclaim_still_terminalizes_ticket() -> None:
    class EmptyMessageAmbiguousFailureSlots(_FakeSlots):
        def commit_slab_reclaim(self, _ticket) -> ExpertSlabReclaimResult:
            raise ExpertSlabReclaimError(
                "",
                result=ExpertSlabReclaimResult((), (), 0),
                failed_slab_id=0,
                destructive_boundary_crossed=None,
            )

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    runtime = _runtime(
        broker,
        slots=EmptyMessageAmbiguousFailureSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(64, 0, 64)],
    )

    with pytest.raises(ExpertSlabReclaimError):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.allocator_cache_bytes == 0
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_resize_terminalization_preserves_existing_kv_owner_handles() -> None:
    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 96, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    first_ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
        cache_id="target:existing",
    )
    first = broker.commit_kv_growth(
        first_ticket,
        allocated_physical_bytes=32,
    )
    resize_ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
        cache_id="target:next",
    )

    broker.terminalize_expert_resize(
        resize_ticket,
        registered_slab_bytes_after=32,
        allocator_before=AllocatorMemorySample(64, 0, 64),
        allocator_after=None,
        reason="injected expert telemetry failure",
    )

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.allocator_cache_bytes == 32
    assert snapshot.owned_kv_physical_bytes == 32
    assert snapshot.failed_closed is True
    broker.release_kv_batch(
        cache_id="target:existing",
        allocations=(first,),
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(32, 32, 64),
        allocator_after=AllocatorMemorySample(0, 32, 64),
    )
    assert broker.snapshot().owned_kv_physical_bytes == 0


def test_runtime_resize_terminalization_preserves_existing_kv_owner_handles() -> None:
    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 96, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    first_ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
        cache_id="target:existing",
    )
    first = broker.commit_kv_growth(
        first_ticket,
        allocated_physical_bytes=32,
    )
    resize_ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
        cache_id="target:next",
    )
    runtime = _runtime(broker, slots=_FakeSlots(), bank=_FakeBank(), samples=[])
    samples = iter((AllocatorMemorySample(96, 0, 96),))

    def sample_allocator() -> AllocatorMemorySample:
        try:
            return next(samples)
        except StopIteration as exc:
            raise MemoryTelemetryError("injected allocator telemetry failure") from exc

    runtime._sample_allocator_memory = sample_allocator

    with pytest.raises(MemoryTelemetryError, match="allocator telemetry failure"):
        runtime.reclaim_expert_bytes(resize_ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.owned_kv_physical_bytes == 32
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is True
    broker.release_kv_batch(
        cache_id="target:existing",
        allocations=(first,),
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(64, 32, 96),
        allocator_after=AllocatorMemorySample(32, 32, 96),
    )
    assert broker.snapshot().owned_kv_physical_bytes == 0


def test_regrow_failure_before_allocation_aborts_the_broker_ticket() -> None:
    class FailingSlots(_FakeSlots):
        def regrow_slab(self, slab_id: int) -> None:
            del slab_id
            raise RuntimeError("injected allocation failure")

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = FailingSlots(physical_bytes=32)
    runtime = _runtime(
        broker,
        slots=slots,
        bank=_FakeBank(),
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(32, 0, 32),
        ],
    )

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.pending_expert_regrow_ticket_id is None


def test_regrow_registry_failure_before_allocation_aborts_the_broker_ticket() -> None:
    class FailingRegistrySlots(_FakeSlots):
        def expert_slab_telemetry_snapshot(self) -> dict[str, int]:
            raise MemoryTelemetryError("injected expert registry failure")

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = FailingRegistrySlots(physical_bytes=32)
    runtime = _runtime(
        broker,
        slots=slots,
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(32, 0, 32)],
    )

    with pytest.raises(MemoryTelemetryError, match="expert registry failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    assert slots.regrown == []
    assert broker.snapshot().pending_expert_regrow_ticket_id is None


def test_regrow_allocator_baseline_failure_aborts_the_broker_ticket() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = _FakeSlots(physical_bytes=32)
    runtime = _runtime(
        broker,
        slots=slots,
        bank=_FakeBank(),
        samples=[],
    )

    with pytest.raises(IndexError):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    assert slots.regrown == []
    assert broker.snapshot().pending_expert_regrow_ticket_id is None


def test_regrow_failure_after_ambiguous_allocation_publishes_physical_truth() -> None:
    class AmbiguousSlots(_FakeSlots):
        def regrow_slab(self, slab_id: int) -> None:
            self.regrown.append(slab_id)
            self._physical_bytes += 32
            raise RuntimeError("injected post-allocation failure")

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = AmbiguousSlots(physical_bytes=32)
    runtime = _runtime(
        broker,
        slots=slots,
        bank=_FakeBank(),
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    with pytest.raises(RuntimeError, match="post-allocation failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.failed_closed is True


def test_empty_message_ambiguous_regrow_still_terminalizes_ticket() -> None:
    class EmptyMessageAmbiguousSlots(_FakeSlots):
        def regrow_slab(self, slab_id: int) -> None:
            self.regrown.append(slab_id)
            self._physical_bytes += 32
            raise MemoryError()

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    runtime = _runtime(
        broker,
        slots=EmptyMessageAmbiguousSlots(physical_bytes=32),
        bank=_FakeBank(),
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    with pytest.raises(MemoryError):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.failed_closed is True


def test_regrow_policy_activation_failure_terminalizes_physical_truth() -> None:
    class FailingActivationBank(_FakeBank):
        def activate_slots(self, slot_ids):
            del slot_ids
            raise RuntimeError("injected policy activation failure")

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    runtime = _runtime(
        broker,
        slots=_FakeSlots(physical_bytes=32),
        bank=FailingActivationBank(),
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    with pytest.raises(RuntimeError, match="policy activation failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.failed_closed is True


def test_reclaim_policy_failure_after_destruction_publishes_physical_truth() -> None:
    class FailingPolicyBank(_FakeBank):
        def deactivate_slots(self, slot_ids):
            del slot_ids
            raise RuntimeError("injected policy deactivation failure")

    target = 110 * BINARY_GIB
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=target - 64, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=FailingPolicyBank(),
        samples=[
            AllocatorMemorySample(target, 0, target),
            AllocatorMemorySample(target - 32, 0, target),
        ],
    )

    with pytest.raises(RuntimeError, match="policy deactivation failure"):
        runtime.reclaim_expert_bytes(ticket, now_ns=1)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_q4_release_waits_for_growth_transaction_and_keeps_exact_owners() -> None:
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    existing_ticket = broker.plan_kv_growth(
        cache_id="target:existing",
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    existing = broker.commit_kv_growth(
        existing_ticket,
        allocated_physical_bytes=32,
    )
    runtime = _runtime(
        broker,
        slots=_FakeSlots(),
        bank=_FakeBank(),
        samples=[AllocatorMemorySample(132, 0, 132)],
    )
    growth = runtime.reserve_growth(
        cache_id="target:growing",
        steady_delta_bytes=32,
        transient_delta_bytes=0,
    )
    started = threading.Event()
    release_errors: list[BaseException] = []

    def release_existing() -> None:
        started.set()
        try:
            runtime.release_cache(
                cache_id="target:existing",
                allocations=(existing,),
                released_physical_bytes=32,
                allocator_before=AllocatorMemorySample(32, 0, 32),
                allocator_after=AllocatorMemorySample(0, 0, 32),
            )
        except BaseException as exc:
            release_errors.append(exc)

    thread = threading.Thread(target=release_existing)
    thread.start()
    assert started.wait(timeout=1)
    time.sleep(0.01)
    assert thread.is_alive()

    growing = runtime.commit_growth(growth, measured_physical_bytes=32)
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert release_errors == []
    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 32
    assert snapshot.owned_kv_physical_bytes == 32
    assert snapshot.unreconciled_kv_physical_bytes == 0
    broker.release_kv(
        growing,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(32, 0, 32),
        allocator_after=AllocatorMemorySample(0, 0, 32),
    )


def test_regrow_allocator_telemetry_failure_consumes_ticket_with_physical_truth() -> (
    None
):
    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=64),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    runtime = _runtime(
        broker,
        slots=_FakeSlots(physical_bytes=32),
        bank=_FakeBank(),
        samples=[],
    )
    samples = iter((AllocatorMemorySample(32, 0, 32),))

    def sample_allocator() -> AllocatorMemorySample:
        try:
            return next(samples)
        except StopIteration as exc:
            raise MemoryTelemetryError("injected allocator telemetry failure") from exc

    runtime._sample_allocator_memory = sample_allocator

    with pytest.raises(MemoryTelemetryError, match="allocator telemetry failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.failed_closed is True


def test_partial_regrow_is_published_and_fails_closed() -> None:
    class PartialSlots(_FakeSlots):
        def slab_layout(self) -> dict[int, tuple[int, ...]]:
            return {0: (0, 1), 1: (2, 3), 2: (4, 5)}

        def released_slab_ids(self) -> tuple[int, ...]:
            return (0, 1)

        def slot_ids_for_slab(self, slab_id: int) -> tuple[int, ...]:
            return self.slab_layout()[slab_id]

        def regrow_slab(self, slab_id: int) -> None:
            if slab_id == 1:
                raise RuntimeError("injected second-slab failure")
            self.regrown.append(slab_id)
            self._physical_bytes += 32

    broker = UnifiedMemoryBroker(
        initial_snapshot=_snapshot(resident=36, experts=96),
        expert_slab_bytes=32,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    broker.replace_snapshot(_snapshot(resident=36, experts=32))
    slots = PartialSlots(physical_bytes=32)
    runtime = _runtime(
        broker,
        slots=slots,
        bank=_FakeBank(),
        samples=[
            AllocatorMemorySample(32, 0, 32),
            AllocatorMemorySample(64, 0, 64),
        ],
    )

    with pytest.raises(RuntimeError, match="second-slab failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=64, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.failed_closed is True
