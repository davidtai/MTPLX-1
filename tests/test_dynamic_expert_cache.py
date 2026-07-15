from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import expert_runtime as expert_runtime_module
from mtplx.expert_runtime import ExpertStreamingRuntime
from mtplx.expert_slots import ExpertRecordReleaseResult
from mtplx.memory_broker import (
    AllocatorMemorySample,
    BrokerSnapshot,
    MemoryAdmissionError,
    MemoryBudget,
    MemoryTelemetryError,
    UnifiedMemoryBroker,
)


class _Ready:
    def __init__(self, plan, bindings, pool) -> None:
        self.plan = plan
        self.bindings = tuple(bindings)
        self.pool = pool

    def validate(self) -> None:
        return

    def release(self, *, synchronize: bool) -> None:
        del synchronize


class _DirectSlots:
    def __init__(self, *, record_bytes: int, persistent_slots: int) -> None:
        self.record_bytes = record_bytes
        self.persistent = [None] * persistent_slots
        self.transient = [bytearray(record_bytes), bytearray(record_bytes)]
        self.protected: set[int] = set()
        self.allocations = 0
        self.releases = 0

    def raise_if_unhealthy(self) -> None:
        return

    def commit_if_healthy(self, callback) -> None:
        callback()

    def reset(self) -> None:
        return

    def retain_admitted_split_lifecycle(self):
        return SimpleNamespace(release=lambda: None)

    def allocate_persistent_slot(self, slot_id: int) -> int:
        assert self.persistent[slot_id] is None
        self.persistent[slot_id] = bytearray(self.record_bytes)
        self.allocations += 1
        return self.record_bytes

    def release_persistent_slots(self, slot_ids) -> ExpertRecordReleaseResult:
        normalized = tuple(slot_ids)
        for slot_id in normalized:
            assert self.persistent[slot_id] is not None
            self.persistent[slot_id] = None
        self.releases += len(normalized)
        return ExpertRecordReleaseResult(
            normalized,
            len(normalized) * self.record_bytes,
        )

    def persistent_cache_telemetry_snapshot(self) -> dict[str, int]:
        allocated = sum(buffer is not None for buffer in self.persistent)
        return {
            "logical_record_capacity": len(self.persistent),
            "allocated_record_count": allocated,
            "resident_record_count": allocated,
            "in_flight_record_count": 0,
            "pinned_record_count": len(self.protected),
            "physical_bytes": allocated * self.record_bytes,
        }

    def protected_slot_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.protected))

    def ensure_route(self, _layer, plan, **_kwargs):
        io_admission = _kwargs.get("io_admission")
        route_admitted = _kwargs.get("route_admitted")
        if plan.loads and io_admission is not None:
            io_admission.mark_accepted()
        if plan.loads and route_admitted is not None:
            route_admitted()
        bindings = []
        persistent_count = len(self.persistent)
        for expert, slot_id in zip(plan.experts, plan.slots, strict=True):
            if slot_id < persistent_count:
                buffer = self.persistent[slot_id]
                assert buffer is not None
            else:
                buffer = self.transient[slot_id - persistent_count]
            bindings.append(
                SimpleNamespace(
                    expert=expert,
                    slot=slot_id,
                    logical_slot=slot_id,
                    buffer=buffer,
                    generation=None,
                )
            )
        return _Ready(plan, bindings, self)

    ensure_route_part = ensure_route


def _runtime(
    *,
    expert_cache_limit_bytes: int = 10,
    memory_limit_bytes: int = 60,
) -> ExpertStreamingRuntime:
    record_bytes = 10
    slots = _DirectSlots(record_bytes=record_bytes, persistent_slots=2)
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=memory_limit_bytes),
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=50,
            kv_physical_bytes=0,
            expert_cache_physical_bytes=0,
            in_flight_expert_staging_bytes=0,
            runtime_workspace_bytes=0,
            allocator_cache_bytes=0,
        ),
        expert_record_bytes=record_bytes,
        expert_cache_limit_bytes=expert_cache_limit_bytes,
    )
    runtime = ExpertStreamingRuntime(
        root=SimpleNamespace(),
        spec=SimpleNamespace(
            key="tiny-dynamic",
            routed_layer_indices=(1,),
            expert_count=3,
            expert_record_bytes=record_bytes,
        ),
        config=SimpleNamespace(
            cache_scope="global",
            frequency_decay=0.995,
            cache_policy="lru",
            dynamic_expert_cache=True,
            expert_cache_limit_bytes=expert_cache_limit_bytes,
            trace_routes=False,
        ),
        manifest=SimpleNamespace(),
        plan=SimpleNamespace(
            persistent_slots=2,
            transient_slots=2,
            slots_per_layer=2,
        ),
        reader=SimpleNamespace(),
        slots=slots,
        memory_broker=broker,
    )
    peak = 50

    def sample_allocator_memory() -> AllocatorMemorySample:
        nonlocal peak
        active = 50 + slots.persistent_cache_telemetry_snapshot()["physical_bytes"]
        peak = max(peak, active)
        sample = AllocatorMemorySample(active, 0, peak)
        runtime._last_allocator_sample = sample
        return sample

    runtime._sample_allocator_memory = sample_allocator_memory
    return runtime


def test_dynamic_runtime_starts_empty_warms_reuses_and_gives_record_to_kv(
    monkeypatch,
) -> None:
    runtime = _runtime()
    broker_calls = 0
    original_plan = runtime.memory_broker.plan_expert_cache_growth

    def counted_plan():
        nonlocal broker_calls
        broker_calls += 1
        return original_plan()

    monkeypatch.setattr(runtime.memory_broker, "plan_expert_cache_growth", counted_plan)
    try:
        record = runtime.spec.expert_record_bytes
        assert runtime.slots.persistent_cache_telemetry_snapshot()["physical_bytes"] == 0

        first = runtime.ensure_route(1, [0], phase="decode")
        first_buffer = first.bindings[0].buffer
        first.release(synchronize=False)
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == record
        assert broker_calls == 1

        hit = runtime.ensure_route(1, [0], phase="decode")
        assert hit.bindings[0].buffer is first_buffer
        hit.release(synchronize=False)
        assert broker_calls == 1

        second = runtime.ensure_route(1, [1], phase="decode")
        assert second.bindings[0].buffer is first_buffer
        second.release(synchronize=False)
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == record

        ticket = runtime.memory_broker.plan_kv_growth(
            steady_delta_bytes=record,
            transient_delta_bytes=0,
            cache_id="target",
        )
        released = runtime.reclaim_expert_records(ticket)
        assert released.physical_bytes == record
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == 0
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_split_route_uses_same_miss_warming_and_hit_fast_path(monkeypatch) -> None:
    runtime = _runtime()
    broker_calls = 0
    original_plan = runtime.memory_broker.plan_expert_cache_growth

    def counted_plan():
        nonlocal broker_calls
        broker_calls += 1
        return original_plan()

    monkeypatch.setattr(runtime.memory_broker, "plan_expert_cache_growth", counted_plan)
    try:
        first = runtime.begin_split_route(1, [0], phase="decode")
        first_misses = first.finish_misses()
        assert first_misses is not None
        first_buffer = first_misses.bindings[0].buffer
        first.release_misses(first_misses)
        first.close()
        assert broker_calls == 1

        hit = runtime.begin_split_route(1, [0], phase="decode")
        assert hit.hit_ready is not None
        assert hit.hit_ready.bindings[0].buffer is first_buffer
        hit.release_hits()
        hit.close()
        assert broker_calls == 1
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_prefill_warms_only_the_actual_seed_demand() -> None:
    runtime = _runtime(expert_cache_limit_bytes=20, memory_limit_bytes=70)
    try:
        assert runtime.prepare_prefill_seed(1, [0]) == (0,)

        ready = runtime.ensure_route(1, [0, 1], phase="prefill")

        assert runtime.slots.allocations == 1
        assert ready.plan.slots == (0, 2)
        ready.release(synchronize=False)
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_decode_miss_grows_only_the_requested_unique_records() -> None:
    runtime = _runtime(expert_cache_limit_bytes=20, memory_limit_bytes=70)
    try:
        ready = runtime.ensure_route(1, [0, 1, 0], phase="decode")

        assert runtime.slots.allocations == 2
        assert ready.plan.slots == (0, 1, 0)
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == 20
        ready.release(synchronize=False)
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_limit_refusal_serves_a_miss_from_transient_storage() -> None:
    runtime = _runtime(expert_cache_limit_bytes=0)
    try:
        ready = runtime.ensure_route(1, [0], phase="decode")

        assert ready.plan.slots == (2,)
        assert runtime.slots.allocations == 0
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == 0
        ready.release(synchronize=False)
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_kv_reclaim_rounds_up_to_one_record_and_preserves_a_selected_pin() -> None:
    runtime = _runtime(expert_cache_limit_bytes=20, memory_limit_bytes=70)
    try:
        for expert in (0, 1):
            ready = runtime.ensure_route(1, [expert], phase="decode")
            ready.release(synchronize=False)
        runtime.slots.protected.add(0)

        ticket = runtime.memory_broker.plan_kv_growth(
            steady_delta_bytes=1,
            transient_delta_bytes=0,
            cache_id="target",
        )
        assert ticket.required_expert_reclaim_bytes == 1

        released = runtime.reclaim_expert_records(ticket)

        assert released.slot_ids == (1,)
        assert released.physical_bytes == runtime.spec.expert_record_bytes
        assert runtime.slots.persistent[0] is not None
        assert runtime.slots.persistent[1] is None
        runtime.memory_broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_allocator_retention_after_record_release_fails_kv_admission_closed() -> None:
    runtime = _runtime()
    try:
        ready = runtime.ensure_route(1, [0], phase="decode")
        ready.release(synchronize=False)
        ticket = runtime.memory_broker.plan_kv_growth(
            steady_delta_bytes=10,
            transient_delta_bytes=0,
            cache_id="target",
        )
        samples = iter(
            (
                AllocatorMemorySample(60, 0, 60),
                AllocatorMemorySample(60, 0, 60),
            )
        )
        runtime._sample_allocator_memory = lambda: next(samples)

        with pytest.raises(MemoryTelemetryError, match="allocator footprint"):
            runtime.reclaim_expert_records(ticket)

        snapshot = runtime.memory_broker.snapshot()
        assert snapshot.expert_cache_physical_bytes == 0
        assert snapshot.allocator_cache_bytes == 10
        assert snapshot.failed_closed is True
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_all_pinned_records_block_reclaim_without_draining_the_cache() -> None:
    runtime = _runtime()
    try:
        ready = runtime.ensure_route(1, [0], phase="decode")
        ready.release(synchronize=False)
        runtime.slots.protected.add(0)
        ticket = runtime.memory_broker.plan_kv_growth(
            steady_delta_bytes=10,
            transient_delta_bytes=0,
            cache_id="target",
        )

        with pytest.raises(MemoryAdmissionError, match="unpinned"):
            runtime.reclaim_expert_records(ticket)

        assert runtime.slots.releases == 0
        assert runtime.slots.persistent[0] is not None
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == 10
        runtime.memory_broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_kv_release_and_runtime_reset_do_not_warm_records(monkeypatch) -> None:
    runtime = _runtime(memory_limit_bytes=70)
    try:
        ready = runtime.ensure_route(1, [0], phase="decode")
        ready.release(synchronize=False)
        allocation_count = runtime.slots.allocations
        monkeypatch.setattr(
            runtime.memory_broker,
            "plan_expert_cache_growth",
            lambda: pytest.fail("release/reset must not warm expert records"),
        )
        ticket = runtime.memory_broker.plan_kv_growth(
            steady_delta_bytes=1,
            transient_delta_bytes=0,
            cache_id="target",
        )
        allocation = runtime.memory_broker.commit_kv_growth(
            ticket,
            allocated_physical_bytes=1,
        )

        runtime.release_cache(
            cache_id="target",
            allocations=(allocation,),
            released_physical_bytes=1,
            allocator_before=AllocatorMemorySample(61, 0, 61),
            allocator_after=AllocatorMemorySample(60, 0, 61),
        )
        runtime.reset()

        assert runtime.slots.allocations == allocation_count
        assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == 10
    finally:
        runtime._split_executor.shutdown(wait=True)


def test_dynamic_broker_initializes_empty_with_one_configured_limit(
    monkeypatch,
) -> None:
    slots = _DirectSlots(record_bytes=10, persistent_slots=2)
    slots.lazy_persistent_buffers = True
    monkeypatch.setattr(
        expert_runtime_module,
        "mlx_memory_telemetry",
        lambda _mx: {
            "active_memory_bytes": 80,
            "cache_memory_bytes": 5,
            "peak_memory_bytes": 80,
        },
    )

    broker = ExpertStreamingRuntime._initialize_dynamic_memory_broker(
        config=SimpleNamespace(
            memory_limit_bytes=100,
            allocator_headroom_bytes=0,
            expert_cache_limit_bytes=20,
        ),
        spec=SimpleNamespace(expert_record_bytes=10),
        plan=SimpleNamespace(
            persistent_slots=2,
            resident_bytes=50,
            transient_bytes=10,
            io_staging_bytes=5,
            runtime_reserve_bytes=10,
            execution_workspace_bytes=5,
        ),
        slots=slots,
        mx_module=object(),
    )

    snapshot = broker.snapshot()
    assert broker.budget.memory_limit_bytes == 100
    assert snapshot.expert_cache_physical_bytes == 0
    assert snapshot.in_flight_expert_staging_bytes == 15
    assert snapshot.runtime_workspace_bytes == 15
    assert snapshot.allocator_cache_bytes == 5


def test_dynamic_telemetry_contains_records_and_no_grouped_fields() -> None:
    runtime = _runtime()
    try:
        ready = runtime.ensure_route(1, [0], phase="decode")
        ready.release(synchronize=False)

        telemetry = runtime.dynamic_memory_telemetry_snapshot()

        assert telemetry["memory_limit_bytes"] == 60
        assert telemetry["expert_cache_physical_bytes"] == 10
        assert telemetry["record_allocations"] == 1
        assert "logical_slab_count" not in telemetry
        assert "hysteresis_slabs" not in telemetry
        assert "regrow_requests" not in telemetry
    finally:
        runtime._split_executor.shutdown(wait=True)
