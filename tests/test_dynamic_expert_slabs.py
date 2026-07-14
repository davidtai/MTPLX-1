from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.expert_runtime import ExpertStreamingConfig, ExpertStreamingRuntime
from mtplx.expert_slots import ExpertSlabReclaimResult
from mtplx.expert_streaming_models import HY3_Q4
from mtplx.memory_broker import (
    BINARY_GIB,
    AllocatorMemorySample,
    BrokerSnapshot,
    MemoryAdmissionError,
    MemoryTelemetryError,
    UnifiedMemoryBroker,
)


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
        return (0,) if self._physical_bytes == 32 else ()

    def slot_ids_for_slab(self, slab_id: int) -> tuple[int, ...]:
        assert slab_id == 0
        return (0, 1)

    def regrow_slab(self, slab_id: int) -> None:
        assert slab_id == 0
        self.regrown.append(slab_id)
        self._physical_bytes += 32

    def snapshot(self) -> dict[str, object]:
        return {"slabs": {"physical_bytes": self._physical_bytes}}


def _runtime(
    broker: UnifiedMemoryBroker,
    *,
    slots: _FakeSlots,
    bank: _FakeBank,
    samples: list[AllocatorMemorySample],
) -> ExpertStreamingRuntime:
    runtime = object.__new__(ExpertStreamingRuntime)
    runtime.memory_broker = broker
    runtime.slots = slots
    runtime._global_bank = bank
    runtime._dynamic_resize_lock = __import__("threading").RLock()
    runtime._sample_allocator_memory = lambda: samples.pop(0)
    return runtime


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
        samples=[AllocatorMemorySample(32, 0, 32)],
    )

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        runtime.maybe_regrow_expert_slabs(target_bytes=32, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 32
    assert snapshot.pending_expert_regrow_ticket_id is None


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

    with pytest.raises(MemoryTelemetryError, match="did not match"):
        runtime.maybe_regrow_expert_slabs(target_bytes=64, now_ns=5)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 64
    assert snapshot.pending_expert_regrow_ticket_id is None
    assert snapshot.failed_closed is True
