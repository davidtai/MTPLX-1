from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.attention_context import attention_phase
from mtplx.expert_runtime import ExpertStreamingRuntime
from mtplx.expert_streaming import (
    CacheCounters,
    EvictedResident,
    ExpertResidencyClass,
    GlobalExpertSlotBank,
    RoutePlan,
    RoutingPhase,
)
from mtplx.models.expert_mlx import current_expert_routing_phase
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime


class _AllHitReady:
    def __init__(self, plan: RoutePlan) -> None:
        self.plan = plan
        self.released = False

    def release(self, *, synchronize: bool = True) -> None:
        assert synchronize is False
        self.released = True


class _AllHitSlots:
    def raise_if_unhealthy(self) -> None:
        pass

    def ensure_route(
        self,
        _layer: int,
        plan: RoutePlan,
        **_kwargs: object,
    ) -> _AllHitReady:
        return _AllHitReady(plan)


def _bank(
    *, persistent_slots: int = 6, cache_policy: str = "lru"
) -> GlobalExpertSlotBank:
    return GlobalExpertSlotBank(
        layer_indices=(1,),
        expert_count=16,
        persistent_slots=persistent_slots,
        transient_slots=1,
        prefill_slots_per_layer=0,
        cache_policy=cache_policy,
    )


def _admit(
    bank: GlobalExpertSlotBank,
    expert: int,
    *,
    residency_class: ExpertResidencyClass = ExpertResidencyClass.ORDINARY,
):
    plan = bank.plan(
        1,
        [expert],
        phase=RoutingPhase.DECODE,
        residency_class=residency_class,
    )
    bank.publish_ready(1, plan)
    return plan.loads[0]


def _actual_streaming_runtime(
    bank: GlobalExpertSlotBank,
) -> ExpertStreamingRuntime:
    runtime = object.__new__(ExpertStreamingRuntime)
    runtime._closed = False
    runtime._closing = False
    runtime._global_bank = bank
    runtime._banks = {}
    runtime._layer_locks = {1: threading.Lock()}
    runtime._pipeline_ledger = None
    runtime._cleanup_error_lock = threading.Lock()
    runtime._cleanup_error = None
    runtime.slots = _AllHitSlots()
    runtime.spec = SimpleNamespace(key="test", expert_record_bytes=1)
    runtime.counters = CacheCounters()
    runtime._layer_counters = {1: CacheCounters()}
    runtime._phase_counters = {phase: CacheCounters() for phase in RoutingPhase}
    runtime._counter_lock = threading.Lock()
    runtime._incremental_miss_routes = 0
    runtime._incremental_miss_parts = 0
    return runtime


def test_deactivate_preserves_logical_arrays_and_generation_watermarks() -> None:
    bank = _bank(persistent_slots=3)
    first = _admit(bank, 0, residency_class=ExpertResidencyClass.SPECULATIVE)
    second = _admit(bank, 1)
    slot_array = bank._slot_to_key
    generation_array = bank._slot_generations
    generations_before = tuple(generation_array)

    evicted = bank.deactivate_slots((first.slot, second.slot))

    assert len(evicted) == 2
    assert isinstance(evicted[0], EvictedResident)
    assert evicted[0].slot == first.slot
    assert evicted[0].layer == 1
    assert evicted[0].expert == 0
    assert evicted[0].generation == first.generation
    assert evicted[0].residency_class is ExpertResidencyClass.SPECULATIVE
    assert isinstance(evicted[1], EvictedResident)
    assert evicted[1].slot == second.slot
    assert evicted[1].layer == 1
    assert evicted[1].expert == 1
    assert evicted[1].generation == second.generation
    assert evicted[1].residency_class is ExpertResidencyClass.ORDINARY
    assert bank._slot_to_key is slot_array
    assert bank._slot_generations is generation_array
    assert tuple(bank._slot_generations) == generations_before
    assert bank.active_slot_mask == (False, False, True)
    assert bank.occupancy == 0


def test_deactivated_slots_are_neither_empty_candidates_nor_victims() -> None:
    bank = _bank(persistent_slots=3)
    first = _admit(bank, 0)
    _admit(bank, 1)
    _admit(bank, 2)
    generation_before = first.generation

    bank.deactivate_slots((first.slot,))
    replacement = _admit(bank, 3)

    assert replacement.slot != first.slot
    assert bank._slot_to_key[first.slot] is None
    assert bank._slot_generations[first.slot] == generation_before

    bank.activate_slots((first.slot,))
    recreated = _admit(bank, 4)
    assert recreated.slot == first.slot
    assert recreated.generation == generation_before + 1
    assert bank.active_slot_mask == (True, True, True)


def test_deactivate_rejects_loading_resident_before_rollback_and_reuse() -> None:
    bank = _bank(persistent_slots=1)
    loading_plan, loading_txn = bank.plan_transaction(
        1,
        [0],
        phase=RoutingPhase.DECODE,
    )
    loading = loading_plan.loads[0]

    assert bank._directory[(1, 0)].state == "loading"
    with pytest.raises(RuntimeError, match="non-ready resident"):
        bank.deactivate_slots((loading.slot,))

    assert bank.active_slot_mask == (True,)
    assert bank._slot_to_key[loading.slot] == (1, 0)
    assert bank._slot_generations[loading.slot] == loading.generation

    loading_txn.rollback_completion()
    assert bank.active_slot_mask == (True,)
    assert bank._slot_to_key[loading.slot] is None

    ready_plan, ready_txn = bank.plan_transaction(
        1,
        [0],
        phase=RoutingPhase.DECODE,
    )
    ready_txn.commit()
    ready = ready_plan.loads[0]
    bank.deactivate_slots((ready.slot,))
    bank.activate_slots((ready.slot,))
    replacement = bank.plan(1, [1], phase=RoutingPhase.DECODE).loads[0]

    assert replacement.slot == ready.slot
    assert replacement.generation == ready.generation + 1


def test_deactivate_preflight_failure_does_not_mutate_earlier_slots() -> None:
    bank = _bank(persistent_slots=2)
    first = _admit(bank, 0)
    second = _admit(bank, 1)
    bank._directory.pop((1, 1))
    state_before = (
        bank.active_slot_mask,
        tuple(bank._slot_to_key),
        dict(bank._key_to_slot),
        tuple(bank._free_slots),
        set(bank._free_slot_set),
    )

    with pytest.raises(RuntimeError, match="missing its directory entry"):
        bank.deactivate_slots((first.slot, second.slot))

    assert (
        bank.active_slot_mask,
        tuple(bank._slot_to_key),
        dict(bank._key_to_slot),
        tuple(bank._free_slots),
        set(bank._free_slot_set),
    ) == state_before


def test_activate_preflight_failure_does_not_mutate_earlier_slots() -> None:
    bank = _bank(persistent_slots=2)
    bank.deactivate_slots((0, 1))
    bank._slot_to_key[1] = (1, 0)
    state_before = (
        bank.active_slot_mask,
        tuple(bank._slot_to_key),
        tuple(bank._free_slots),
        set(bank._free_slot_set),
    )

    with pytest.raises(RuntimeError, match="inactive global expert slot"):
        bank.activate_slots((0, 1))

    assert (
        bank.active_slot_mask,
        tuple(bank._slot_to_key),
        tuple(bank._free_slots),
        set(bank._free_slot_set),
    ) == state_before


def test_speculative_only_slabs_rank_before_ordinary_slabs() -> None:
    bank = _bank(persistent_slots=4)
    speculative = (
        _admit(bank, 0, residency_class=ExpertResidencyClass.SPECULATIVE),
        _admit(bank, 1, residency_class=ExpertResidencyClass.SPECULATIVE),
    )
    ordinary = (_admit(bank, 2), _admit(bank, 3))

    ranked = bank.rank_reclaim_slabs(
        {
            10: tuple(load.slot for load in ordinary),
            20: tuple(load.slot for load in speculative),
        }
    )

    assert ranked == (20, 10)


@pytest.mark.parametrize("cache_policy", ["lru", "frequency"])
def test_ordinary_slabs_rank_by_their_hottest_members_coldness(
    cache_policy: str,
) -> None:
    bank = _bank(persistent_slots=4, cache_policy=cache_policy)
    loads = tuple(_admit(bank, expert) for expert in range(4))
    # Refresh expert 0 so slab 10 contains the globally hottest ordinary record.
    hit = bank.plan(1, [0], phase=RoutingPhase.DECODE)
    assert hit.loads == ()

    ranked = bank.rank_reclaim_slabs(
        {
            10: (loads[0].slot, loads[1].slot),
            20: (loads[2].slot, loads[3].slot),
        }
    )

    assert ranked == (20, 10)


def test_protected_slot_excludes_the_entire_slab() -> None:
    bank = _bank(persistent_slots=4)
    loads = tuple(_admit(bank, expert) for expert in range(4))

    ranked = bank.rank_reclaim_slabs(
        {
            10: (loads[0].slot, loads[1].slot),
            20: (loads[2].slot, loads[3].slot),
        },
        (loads[1].slot,),
    )

    assert ranked == (20,)


def test_decode_demand_is_ordinary_unless_explicitly_speculative() -> None:
    bank = _bank(persistent_slots=2)

    ordinary = _admit(bank, 0)
    speculative = _admit(
        bank,
        1,
        residency_class=ExpertResidencyClass.SPECULATIVE,
    )

    assert ordinary.residency_class is ExpertResidencyClass.ORDINARY
    assert speculative.residency_class is ExpertResidencyClass.SPECULATIVE
    assert bank._directory[(1, 0)].residency_class is ExpertResidencyClass.ORDINARY
    assert bank._directory[(1, 1)].residency_class is ExpertResidencyClass.SPECULATIVE

    # A later authoritative decode demand promotes a speculative resident.
    hit = bank.plan(1, [1], phase=RoutingPhase.DECODE)
    assert hit.loads == ()
    assert bank._directory[(1, 1)].residency_class is ExpertResidencyClass.ORDINARY


def test_mtp_decode_verify_real_route_defaults_to_ordinary_residency() -> None:
    bank = _bank(persistent_slots=1)
    _admit(bank, 0, residency_class=ExpertResidencyClass.SPECULATIVE)
    streaming = _actual_streaming_runtime(bank)
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_verify_width=8),
        tokenizer=None,
        model_path=Path("."),
        mtp_enabled=True,
        contract=MTPContract(),
        expert_streaming=streaming,
    )

    # Production path: attention_phase("decode_verify") ->
    # MTPLXRuntime._expert_routing_context -> current_expert_routing_phase ->
    # ExpertStreamingRuntime.begin_split_route -> _plan_route_transaction ->
    # GlobalExpertSlotBank.plan_transaction, with no residency-class override.
    with attention_phase("decode_verify"):
        with runtime._expert_routing_context(SimpleNamespace(shape=(1, 8))):
            phase = current_expert_routing_phase(token_count=8)
            pending = streaming.begin_split_route(1, [0], phase=phase)

    try:
        assert pending.plan.phase is RoutingPhase.DECODE
        assert bank._directory[(1, 0)].residency_class is ExpertResidencyClass.ORDINARY
    finally:
        pending.close()
