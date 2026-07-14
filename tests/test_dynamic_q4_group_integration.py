from __future__ import annotations

import gc
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_state import (
    VllmMetalPagedKVCache,
    prepare_brokered_q4_cache_group,
)
from mtplx.expert_runtime import ExpertStreamingRuntime, KVGroupGrowthContext
from mtplx.expert_slots import ExpertSlabReclaimResult
from mtplx.kv_quant import PagedKVQuantConfig
from mtplx.memory_broker import (
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_LAYERS,
    AllocatorMemorySample,
    BrokerSnapshot,
    MemoryBudget,
    UnifiedMemoryBroker,
)


_EXPERT_SLAB_BYTES = 2 * 1024 * 1024
_INITIAL_SLACK_BYTES = 700_000


def _allocator_sample() -> AllocatorMemorySample:
    return AllocatorMemorySample(
        active_bytes=int(mx.get_active_memory()),
        cache_bytes=int(mx.get_cache_memory()),
        peak_bytes=int(mx.get_peak_memory()),
    )


class _TrackingRLock:
    """Expose transaction ownership without changing RLock semantics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.owner: int | None = None
        self.depth = 0
        self.max_depth = 0
        self.acquisitions = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if timeout == -1:
            acquired = self._lock.acquire(blocking)
        else:
            acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            owner = threading.get_ident()
            if self.owner is None:
                self.owner = owner
            else:
                assert self.owner == owner
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            self.acquisitions += 1
        return acquired

    def release(self) -> None:
        assert self.owner == threading.get_ident()
        assert self.depth > 0
        self.depth -= 1
        if self.depth == 0:
            self.owner = None
        self._lock.release()

    def __enter__(self) -> _TrackingRLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _RealExpertSlab:
    """One registry slab backed by a real evaluated MLX allocation."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.prepared: list[tuple[tuple[int, ...], int]] = []
        self.commit_calls = 0
        self.abort_calls = 0
        self._physical_bytes = _EXPERT_SLAB_BYTES
        self._array = mx.zeros((_EXPERT_SLAB_BYTES,), dtype=mx.uint8)
        mx.eval(self._array)

    def slab_layout(self) -> dict[int, tuple[int, ...]]:
        return {0: (0,)}

    def protected_slot_ids(self) -> tuple[int, ...]:
        return ()

    def prepare_slab_reclaim(
        self,
        slab_ids,
        *,
        requested_bytes: int,
        deadline_ns: int | None = None,
    ):
        del deadline_ns
        selected = tuple(int(slab_id) for slab_id in slab_ids)
        assert selected == (0,)
        assert 0 < requested_bytes <= self._physical_bytes
        self.prepared.append((selected, int(requested_bytes)))
        return SimpleNamespace(slab_ids=(0,))

    def commit_slab_reclaim(self, ticket) -> ExpertSlabReclaimResult:
        assert ticket.slab_ids == (0,)
        assert self._array is not None
        released = self._physical_bytes
        self._array = None
        self._physical_bytes = 0
        gc.collect()
        mx.clear_cache()
        self.commit_calls += 1
        self.events.append("expert-reclaim-complete")
        return ExpertSlabReclaimResult(
            slab_ids=(0,),
            released_slot_ids=(0,),
            physical_bytes=released,
        )

    def abort_slab_reclaim(self, _ticket) -> None:
        self.abort_calls += 1

    def slot_ids_for_slab(self, slab_id: int) -> tuple[int, ...]:
        assert slab_id == 0
        return (0,)

    def released_slab_ids(self) -> tuple[int, ...]:
        return (0,) if self._physical_bytes == 0 else ()

    def snapshot(self) -> dict[str, object]:
        return {"slabs": {"physical_bytes": self._physical_bytes}}

    def expert_slab_telemetry_snapshot(self) -> dict[str, int]:
        active = int(self._physical_bytes > 0)
        return {
            "logical_slab_count": 1,
            "active_slab_count": active,
            "draining_slab_count": 0,
            "released_slab_count": 1 - active,
            "logical_slot_count": 1,
            "active_slot_count": active,
            "resident_record_count": 0,
            "in_flight_record_count": 0,
            "pinned_record_count": 0,
            "pin_count": 0,
            "logical_bytes": _EXPERT_SLAB_BYTES,
            "physical_bytes": self._physical_bytes,
            "released_bytes": _EXPERT_SLAB_BYTES - self._physical_bytes,
            "in_flight_bytes": 0,
            "pinned_bytes": 0,
        }


class _ExpertBank:
    speculative_record_count = 0

    def __init__(self) -> None:
        self.deactivated: list[tuple[int, ...]] = []

    def rank_reclaim_slabs(self, slabs, protected_slots=()):
        assert tuple(protected_slots) == ()
        return tuple(int(slab_id) for slab_id in slabs)

    def deactivate_slots(self, slot_ids):
        selected = tuple(int(slot_id) for slot_id in slot_ids)
        self.deactivated.append(selected)
        return ()


def _runtime_harness(events: list[str]):
    gc.collect()
    mx.clear_cache()
    baseline_allocator = _allocator_sample()
    slots = _RealExpertSlab(events)
    initial_allocator = _allocator_sample()
    resident_bytes = baseline_allocator.active_bytes
    classified_bytes = resident_bytes + _EXPERT_SLAB_BYTES
    allocator_cache_bytes = max(
        initial_allocator.cache_bytes,
        initial_allocator.charged_footprint_bytes - classified_bytes,
    )
    initial_snapshot = BrokerSnapshot(
        resident_model_bytes=resident_bytes,
        kv_physical_bytes=0,
        expert_slab_physical_bytes=_EXPERT_SLAB_BYTES,
        in_flight_expert_staging_bytes=0,
        runtime_workspace_bytes=0,
        allocator_cache_bytes=allocator_cache_bytes,
    )
    operating_target = initial_snapshot.charged_bytes + _INITIAL_SLACK_BYTES
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(
            operating_target_bytes=operating_target,
            hard_ceiling_bytes=operating_target + 4 * 1024 * 1024,
        ),
        initial_snapshot=initial_snapshot,
        initial_allocator_sample=initial_allocator,
        expert_slab_bytes=_EXPERT_SLAB_BYTES,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )

    runtime = object.__new__(ExpertStreamingRuntime)
    runtime.memory_broker = broker
    runtime.slots = slots
    runtime._global_bank = _ExpertBank()
    runtime.spec = SimpleNamespace(expert_record_bytes=_EXPERT_SLAB_BYTES)
    runtime.plan = SimpleNamespace(persistent_slots=1)
    runtime.config = SimpleNamespace(
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ms=0,
    )
    runtime._closed = False
    runtime._closing = False
    runtime._active_kv_growth_group = None
    runtime._layer_locks = {}
    runtime._dynamic_resize_lock = threading.RLock()
    runtime._memory_transaction_lock = _TrackingRLock()
    runtime._dynamic_resize_metrics = {}
    runtime._cleanup_error_lock = threading.Lock()
    runtime._cleanup_error = None
    runtime._pending_physical_kv_lock = threading.Lock()
    runtime._pending_physical_kv_caches = {}
    runtime._mx_module = mx
    runtime._last_allocator_sample = initial_allocator

    groups: list[KVGroupGrowthContext] = []
    reserve_group = ExpertStreamingRuntime.reserve_growth_group.__get__(
        runtime,
        ExpertStreamingRuntime,
    )

    def record_group(*, members):
        group = reserve_group(members=members)
        groups.append(group)
        return group

    runtime.reserve_growth_group = record_group
    return SimpleNamespace(
        broker=broker,
        runtime=runtime,
        slots=slots,
        groups=groups,
    )


def _target_q4_group(runtime: ExpertStreamingRuntime):
    return [
        VllmMetalPagedKVCache(
            block_size=16,
            num_blocks=1,
            kv_quant_config=PagedKVQuantConfig("q4"),
            allocation_observer=runtime,
            cache_id=f"target:integration:{index}",
        )
        for index in range(HY3_Q4_KV_LAYERS)
    ]


def _assert_exact_owner_handles(cache: list[VllmMetalPagedKVCache]) -> None:
    for entry in cache:
        assert entry._kv_allocations
        assert all(
            allocation.cache_id == entry.cache_id
            for allocation in entry._kv_allocations
        )
        assert (
            sum(allocation.physical_bytes for allocation in entry._kv_allocations)
            == entry.nbytes
        )


def _assert_transaction_finished(harness) -> None:
    snapshot = harness.broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert harness.runtime._active_kv_growth_group is None
    assert harness.runtime._memory_transaction_lock.depth == 0
    assert harness.runtime._memory_transaction_lock.owner is None


def test_real_q4_group_reclaims_once_grows_by_exact_page_and_closes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    events: list[str] = []
    harness = _runtime_harness(events)
    cache = _target_q4_group(harness.runtime)
    materialize_member_zero = cache[0]._materialize_brokered_q4_arrays

    def record_member_zero_materialization(*, shape):
        events.append("member-0-allocation-start")
        return materialize_member_zero(shape=shape)

    monkeypatch.setattr(
        cache[0],
        "_materialize_brokered_q4_arrays",
        record_member_zero_materialization,
    )

    initialized = prepare_brokered_q4_cache_group(cache)

    assert initialized == {
        "entries": HY3_Q4_KV_LAYERS,
        "allocated_entries": HY3_Q4_KV_LAYERS,
        "grown_entries": 0,
        "target_blocks": 1,
    }
    assert len(harness.groups) == 1
    initial_group = harness.groups[0]
    assert initial_group.completed is True
    assert 0 < initial_group.ticket.required_expert_reclaim_bytes
    assert harness.slots.prepared == [
        ((0,), initial_group.ticket.required_expert_reclaim_bytes)
    ]
    assert harness.slots.commit_calls == 1
    assert harness.runtime._global_bank.deactivated == [(0,)]
    assert events.index("expert-reclaim-complete") < events.index(
        "member-0-allocation-start"
    )
    _assert_exact_owner_handles(cache)
    initial_physical = sum(entry.nbytes for entry in cache)
    assert initial_physical == HY3_Q4_KV_BLOCK_BYTES
    initial_snapshot = harness.broker.snapshot()
    assert initial_snapshot.kv_physical_bytes == initial_physical
    assert initial_snapshot.owned_kv_physical_bytes == initial_physical
    assert initial_snapshot.unreconciled_kv_physical_bytes == 0
    _assert_transaction_finished(harness)
    assert harness.runtime._memory_transaction_lock.acquisitions == 1
    assert harness.runtime._memory_transaction_lock.max_depth == 1

    within_page = prepare_brokered_q4_cache_group(cache, required_tokens=16)

    assert within_page["target_blocks"] == 1
    assert within_page["grown_entries"] == 0
    assert len(harness.groups) == 1

    crossed_page = prepare_brokered_q4_cache_group(cache, required_tokens=17)

    assert crossed_page == {
        "entries": HY3_Q4_KV_LAYERS,
        "allocated_entries": 0,
        "grown_entries": HY3_Q4_KV_LAYERS,
        "target_blocks": 2,
    }
    assert len(harness.groups) == 2
    growth_group = harness.groups[1]
    assert growth_group.completed is True
    assert growth_group.ticket.steady_delta_bytes == HY3_Q4_KV_BLOCK_BYTES
    assert growth_group.ticket.transient_delta_bytes == (
        2 * HY3_Q4_KV_BLOCK_BYTES // HY3_Q4_KV_LAYERS
    )
    assert growth_group.ticket.required_expert_reclaim_bytes == 0
    assert harness.slots.commit_calls == 1
    assert all(entry.num_blocks == 2 for entry in cache)
    assert all(len(entry._kv_allocations) == 2 for entry in cache)
    _assert_exact_owner_handles(cache)
    grown_physical = sum(entry.nbytes for entry in cache)
    assert grown_physical == 2 * HY3_Q4_KV_BLOCK_BYTES
    grown_snapshot = harness.broker.snapshot()
    assert grown_snapshot.kv_physical_bytes == grown_physical
    assert grown_snapshot.owned_kv_physical_bytes == grown_physical
    assert grown_snapshot.unreconciled_kv_physical_bytes == 0
    _assert_transaction_finished(harness)
    assert harness.runtime._memory_transaction_lock.acquisitions == 2

    for entry in cache:
        entry.close()

    released = harness.broker.snapshot()
    assert released.kv_physical_bytes == 0
    assert released.owned_kv_physical_bytes == 0
    assert released.unreconciled_kv_physical_bytes == 0
    assert released.failed_closed is False
    assert all(entry._closed for entry in cache)
    assert all(entry._kv_allocations == [] for entry in cache)
    _assert_transaction_finished(harness)


def test_real_q4_group_member_failure_preserves_releasable_partial_handles(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "1")
    events: list[str] = []
    harness = _runtime_harness(events)
    cache = _target_q4_group(harness.runtime)
    failed_index = 3

    def fail_member_materialization(*, shape):
        del shape
        raise RuntimeError("injected member materialization failure")

    monkeypatch.setattr(
        cache[failed_index],
        "_materialize_brokered_q4_arrays",
        fail_member_materialization,
    )

    with pytest.raises(RuntimeError, match="injected member materialization failure"):
        prepare_brokered_q4_cache_group(cache)

    assert len(harness.groups) == 1
    group = harness.groups[0]
    assert group.aborted is True
    assert group.completed is False
    assert harness.slots.commit_calls == 1
    committed = cache[:failed_index]
    uncommitted = cache[failed_index:]
    assert all(len(entry._kv_allocations) == 1 for entry in committed)
    assert all(entry.nbytes > 0 for entry in committed)
    assert all(entry._kv_allocations == [] for entry in uncommitted)
    assert all(entry.nbytes == 0 for entry in uncommitted)
    _assert_exact_owner_handles(committed)
    partial_physical = sum(entry.nbytes for entry in committed)
    partial_snapshot = harness.broker.snapshot()
    assert partial_snapshot.kv_physical_bytes == partial_physical
    assert partial_snapshot.owned_kv_physical_bytes == partial_physical
    assert partial_snapshot.unreconciled_kv_physical_bytes == 0
    assert partial_snapshot.failed_closed is False
    _assert_transaction_finished(harness)

    for entry in cache:
        entry.close()

    released = harness.broker.snapshot()
    assert released.kv_physical_bytes == 0
    assert released.owned_kv_physical_bytes == 0
    assert released.unreconciled_kv_physical_bytes == 0
    assert released.failed_closed is False
    _assert_transaction_finished(harness)
