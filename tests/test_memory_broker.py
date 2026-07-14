from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from mtplx.memory_broker import (
    BINARY_GIB,
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BLOCK_TOKENS,
    HY3_Q4_KV_BYTES_PER_TOKEN,
    HY3_Q4_KV_HEAD_DIM,
    HY3_Q4_KV_HEADS,
    HY3_Q4_KV_LAYERS,
    HY3_Q4_KV_SCALE_BYTES,
    AllocatorMemorySample,
    BrokerSnapshot,
    DuplicateReleaseError,
    MemoryAdmissionError,
    MemoryBudget,
    MemoryTelemetryError,
    MemoryTransactionError,
    TerminalizedKVReleaseError,
    UnifiedMemoryBroker,
    hy3_q4_kv_physical_geometry,
)


GIB = 1024**3


def _snapshot(
    *,
    resident: int = 0,
    kv: int = 0,
    experts: int = 0,
    staging: int = 0,
    workspace: int = 0,
    cache: int = 0,
    pinned: int = 0,
    speculative: int = 0,
) -> BrokerSnapshot:
    return BrokerSnapshot(
        resident_model_bytes=resident,
        kv_physical_bytes=kv,
        expert_slab_physical_bytes=experts,
        in_flight_expert_staging_bytes=staging,
        runtime_workspace_bytes=workspace,
        allocator_cache_bytes=cache,
        pinned_expert_bytes=pinned,
        speculative_expert_bytes=speculative,
    )


def _install(broker: UnifiedMemoryBroker, snapshot: BrokerSnapshot) -> None:
    broker.replace_snapshot(snapshot)


def test_binary_gib_budget_and_immutable_pool_accounting() -> None:
    assert BINARY_GIB == GIB
    assert MemoryBudget().operating_target_bytes == 110 * GIB
    assert MemoryBudget().hard_ceiling_bytes == 112 * GIB

    snapshot = _snapshot(
        resident=1,
        kv=2,
        experts=4,
        staging=8,
        workspace=16,
        cache=32,
        pinned=4,
    )
    assert snapshot.charged_bytes == 63
    assert snapshot.pinned_expert_bytes == 4
    with pytest.raises(FrozenInstanceError):
        snapshot.kv_physical_bytes = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tokens", "blocks"),
    [
        (0, 0),
        (HY3_Q4_KV_BLOCK_TOKENS - 1, 1),
        (HY3_Q4_KV_BLOCK_TOKENS, 1),
        (HY3_Q4_KV_BLOCK_TOKENS + 1, 2),
    ],
)
def test_hy3_q4_physical_block_rounding(tokens: int, blocks: int) -> None:
    geometry = hy3_q4_kv_physical_geometry(tokens)

    packed_head_bytes = HY3_Q4_KV_HEAD_DIM // 2
    per_layer_token_bytes = (
        2 * HY3_Q4_KV_HEADS * (packed_head_bytes + HY3_Q4_KV_SCALE_BYTES)
    )
    assert HY3_Q4_KV_LAYERS == 80
    assert HY3_Q4_KV_HEADS == 8
    assert HY3_Q4_KV_HEAD_DIM == 128
    assert HY3_Q4_KV_SCALE_BYTES == 2
    assert per_layer_token_bytes == 1_056
    assert HY3_Q4_KV_BYTES_PER_TOKEN == 80 * 1_056 == 84_480
    assert HY3_Q4_KV_BLOCK_TOKENS == 16
    assert HY3_Q4_KV_BLOCK_BYTES == 16 * 84_480 == 1_351_680
    assert geometry.requested_tokens == tokens
    assert geometry.physical_blocks == blocks
    assert geometry.physical_bytes == blocks * HY3_Q4_KV_BLOCK_BYTES


def test_hy3_q4_full_context_physical_geometry() -> None:
    geometry = hy3_q4_kv_physical_geometry(131_072)

    assert geometry.physical_blocks == 8_192
    assert geometry.physical_bytes == 11_072_962_560
    assert geometry.physical_bytes / GIB == 10.3125


@pytest.mark.parametrize(
    ("charged", "allowed"),
    [
        (110 * GIB - 1, True),
        (110 * GIB, True),
        (110 * GIB + 1, False),
        (112 * GIB - 1, False),
        (112 * GIB, False),
        (112 * GIB + 1, False),
    ],
)
def test_operating_and_hard_boundaries(charged: int, allowed: bool) -> None:
    broker = UnifiedMemoryBroker.standard_hy3()

    if allowed:
        broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=charged))
        assert broker.snapshot().charged_bytes == charged
    else:
        with pytest.raises(MemoryAdmissionError):
            broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=charged))
        assert broker.snapshot().charged_bytes == charged

    expected_hard_failures = int(charged >= 112 * GIB)
    assert broker.snapshot().hard_failure_count == expected_hard_failures


def test_hard_snapshot_rejects_all_later_allocation() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    with pytest.raises(MemoryAdmissionError):
        broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=112 * GIB))

    with pytest.raises(MemoryAdmissionError, match="hard ceiling"):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)


def test_allocator_cache_reconciliation_updates_only_the_cache_pool() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(
            resident=40,
            kv=10,
            experts=20,
            staging=5,
            workspace=10,
            cache=5,
        ),
        expert_slab_bytes=10,
    )

    snapshot = broker.reconcile_allocator_cache(
        AllocatorMemorySample(active_bytes=75, cache_bytes=9, peak_bytes=90)
    )

    assert snapshot.resident_model_bytes == 40
    assert snapshot.kv_physical_bytes == 10
    assert snapshot.expert_slab_physical_bytes == 20
    assert snapshot.in_flight_expert_staging_bytes == 5
    assert snapshot.runtime_workspace_bytes == 10
    assert snapshot.allocator_cache_bytes == 9
    assert snapshot.charged_bytes == 94


def test_allocator_cache_reconciliation_grants_no_credit_for_cache_to_active_move() -> (
    None
):
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(
            resident=40,
            kv=10,
            experts=20,
            staging=5,
            workspace=10,
            cache=9,
        ),
        expert_slab_bytes=10,
    )

    snapshot = broker.reconcile_allocator_cache(
        AllocatorMemorySample(active_bytes=79, cache_bytes=5, peak_bytes=90)
    )

    assert snapshot.allocator_cache_bytes == 9
    assert snapshot.charged_bytes == 94


def test_allocator_cache_reconciliation_charges_unclassified_active_drift() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(
            resident=40,
            kv=10,
            experts=20,
            staging=5,
            workspace=10,
            cache=5,
        ),
        expert_slab_bytes=10,
    )

    snapshot = broker.reconcile_allocator_cache(
        AllocatorMemorySample(active_bytes=90, cache_bytes=5, peak_bytes=95)
    )

    assert snapshot.allocator_cache_bytes == 10
    assert snapshot.charged_bytes == 95


def test_allocator_cache_reconciliation_retains_over_budget_truth_and_fails_closed() -> (
    None
):
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(resident=90, cache=5),
        expert_slab_bytes=10,
    )

    with pytest.raises(MemoryAdmissionError, match="operating target"):
        broker.reconcile_allocator_cache(
            AllocatorMemorySample(active_bytes=90, cache_bytes=11, peak_bytes=101)
        )

    snapshot = broker.snapshot()
    assert snapshot.allocator_cache_bytes == 11
    assert snapshot.charged_bytes == 101
    with pytest.raises(MemoryAdmissionError, match="failed closed"):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)


def test_allocator_cache_reconciliation_cannot_interrupt_a_transaction() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(resident=50, cache=5),
        expert_slab_bytes=10,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=1,
        transient_delta_bytes=0,
        cache_id="target:1:0",
    )

    with pytest.raises(MemoryTransactionError, match="active memory transaction"):
        broker.reconcile_allocator_cache(
            AllocatorMemorySample(active_bytes=50, cache_bytes=7, peak_bytes=57)
        )

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id == ticket.ticket_id
    assert snapshot.allocator_cache_bytes == 5


def test_post_load_classification_reconciliation_preserves_kv_owner_handles() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(resident=40, experts=20, workspace=10),
        expert_slab_bytes=10,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=4,
        transient_delta_bytes=0,
        cache_id="target:existing",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=4)

    snapshot = broker.reconcile_post_load_classification(
        resident_model_bytes=45,
        expert_slab_physical_bytes=20,
        in_flight_expert_staging_bytes=5,
        runtime_workspace_bytes=5,
        allocator_before=AllocatorMemorySample(74, 0, 74),
        allocator_after=AllocatorMemorySample(79, 0, 79),
    )

    assert snapshot.owned_kv_physical_bytes == 4
    broker.release_kv_batch(
        cache_id="target:existing",
        allocations=(allocation,),
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(4, 0, 4),
        allocator_after=AllocatorMemorySample(0, 0, 4),
    )
    assert broker.snapshot().owned_kv_physical_bytes == 0


def test_post_load_classification_atomically_reclassifies_consumed_cache() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(
            resident=40,
            experts=20,
            staging=5,
            workspace=10,
            cache=10,
        ),
        expert_slab_bytes=10,
    )

    snapshot = broker.reconcile_post_load_classification(
        resident_model_bytes=45,
        expert_slab_physical_bytes=20,
        in_flight_expert_staging_bytes=5,
        runtime_workspace_bytes=10,
        allocator_before=AllocatorMemorySample(65, 10, 75),
        allocator_after=AllocatorMemorySample(70, 5, 75),
    )

    assert snapshot.resident_model_bytes == 45
    assert snapshot.allocator_cache_bytes == 5
    assert snapshot.charged_bytes == 85


def test_post_load_classification_does_not_double_charge_planned_resident() -> None:
    initial_sample = AllocatorMemorySample(25, 0, 25)
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(
            resident=40,
            experts=20,
            staging=5,
            workspace=10,
        ),
        initial_allocator_sample=initial_sample,
        expert_slab_bytes=10,
    )

    snapshot = broker.reconcile_post_load_classification(
        resident_model_bytes=45,
        expert_slab_physical_bytes=20,
        in_flight_expert_staging_bytes=5,
        runtime_workspace_bytes=5,
        allocator_before=initial_sample,
        allocator_after=AllocatorMemorySample(70, 0, 70),
    )

    assert snapshot.allocator_cache_bytes == 0
    assert snapshot.charged_bytes == 75


def test_post_load_classification_rejects_active_ticket_without_mutation() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(resident=40, experts=20, workspace=10),
        expert_slab_bytes=10,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=4,
        transient_delta_bytes=0,
        cache_id="target:pending",
    )

    before = broker.snapshot()
    with pytest.raises(MemoryTransactionError, match="active memory transaction"):
        broker.reconcile_post_load_classification(
            resident_model_bytes=41,
            expert_slab_physical_bytes=20,
            in_flight_expert_staging_bytes=0,
            runtime_workspace_bytes=10,
            allocator_before=AllocatorMemorySample(60, 0, 60),
            allocator_after=AllocatorMemorySample(61, 0, 61),
        )

    after = broker.snapshot()
    assert after.pending_kv_ticket_id == ticket.ticket_id
    assert after.revision == before.revision


def test_post_load_classification_consumes_startup_baseline_once() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(operating_target_bytes=100, hard_ceiling_bytes=112),
        initial_snapshot=_snapshot(resident=40, experts=20, workspace=10),
        expert_slab_bytes=10,
    )
    kwargs = {
        "resident_model_bytes": 41,
        "expert_slab_physical_bytes": 20,
        "in_flight_expert_staging_bytes": 0,
        "runtime_workspace_bytes": 9,
        "allocator_before": AllocatorMemorySample(60, 0, 60),
        "allocator_after": AllocatorMemorySample(61, 0, 61),
    }
    broker.reconcile_post_load_classification(**kwargs)
    before = broker.snapshot()

    with pytest.raises(MemoryTransactionError, match="already reconciled"):
        broker.reconcile_post_load_classification(**kwargs)

    assert broker.snapshot().revision == before.revision


def test_two_phase_ticket_accounts_steady_and_transient_peak() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))

    ticket = broker.plan_kv_growth(
        steady_delta_bytes=1 * GIB,
        transient_delta_bytes=1 * GIB,
    )
    assert ticket.required_expert_reclaim_bytes == 2 * GIB
    assert ticket.planned_steady_bytes == 109 * GIB
    assert ticket.planned_peak_bytes == 110 * GIB
    assert broker.snapshot().kv_physical_bytes == 0

    broker.confirm_expert_reclaim(
        ticket,
        registered_slab_bytes_after=8 * GIB,
        allocator_before=AllocatorMemorySample(
            active_bytes=10 * GIB,
            cache_bytes=0,
            peak_bytes=10 * GIB,
        ),
        allocator_after=AllocatorMemorySample(
            active_bytes=8 * GIB,
            cache_bytes=0,
            peak_bytes=10 * GIB,
        ),
        now_ns=10,
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=1 * GIB,
    )

    assert allocation.physical_bytes == 1 * GIB
    assert broker.snapshot().charged_bytes == 109 * GIB
    assert broker.snapshot().kv_physical_bytes == 1 * GIB


def test_transient_peak_requires_reclaim_even_when_steady_state_fits() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=108 * GIB, experts=2 * GIB))

    ticket = broker.plan_kv_growth(
        steady_delta_bytes=1,
        transient_delta_bytes=GIB - 1,
    )

    assert ticket.required_expert_reclaim_bytes == GIB
    assert ticket.planned_steady_bytes == 109 * GIB + 1
    assert ticket.planned_peak_bytes == 110 * GIB


def test_pinned_expert_shortfall_rejects_without_reserving() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(
        broker,
        _snapshot(
            resident=107 * GIB,
            experts=3 * GIB,
            pinned=2 * GIB,
        ),
    )

    with pytest.raises(MemoryAdmissionError, match="pinned"):
        broker.plan_kv_growth(
            steady_delta_bytes=1 * GIB,
            transient_delta_bytes=1 * GIB,
        )

    snapshot = broker.snapshot()
    assert snapshot.charged_bytes == 110 * GIB
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.admission_failure_count == 1


def test_speculative_experts_remain_in_pool_and_reclaimable() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(
        broker,
        _snapshot(
            resident=100 * GIB,
            experts=10 * GIB,
            pinned=7 * GIB,
            speculative=5 * GIB,
        ),
    )

    snapshot = broker.snapshot()
    assert snapshot.charged_bytes == 110 * GIB
    assert snapshot.reclaimable_expert_bytes == 3 * GIB
    assert snapshot.speculative_reclaimable_bytes == 3 * GIB
    assert snapshot.speculative_expert_bytes == 5 * GIB

    ticket = broker.plan_kv_growth(
        steady_delta_bytes=3 * GIB,
        transient_delta_bytes=0,
    )
    assert ticket.required_expert_reclaim_bytes == 3 * GIB
    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    with pytest.raises(MemoryAdmissionError, match="pinned"):
        broker.plan_kv_growth(
            steady_delta_bytes=3 * GIB + 1,
            transient_delta_bytes=0,
        )


def test_expert_reclaim_requires_registered_and_allocator_reduction() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=GIB,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint") as caught:
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(
                active_bytes=10 * GIB,
                cache_bytes=0,
                peak_bytes=10 * GIB,
            ),
            allocator_after=AllocatorMemorySample(
                active_bytes=8 * GIB,
                cache_bytes=2 * GIB,
                peak_bytes=10 * GIB,
            ),
            now_ns=1,
        )
    assert not isinstance(caught.value, TerminalizedKVReleaseError)

    # Active-to-cache movement is reclassification, not physical reclaim.
    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 8 * GIB
    assert snapshot.allocator_cache_bytes == 2 * GIB
    assert snapshot.charged_bytes == 110 * GIB
    assert snapshot.failed_closed is True


def test_reclaim_below_pinned_bytes_preserves_physical_truth_and_invariant() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(
        broker,
        _snapshot(
            resident=100 * GIB,
            experts=10 * GIB,
            pinned=9 * GIB,
        ),
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="pinned"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(8 * GIB, 0, 10 * GIB),
            now_ns=1,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 8 * GIB
    assert snapshot.pinned_expert_bytes == 8 * GIB
    assert snapshot.expert_slab_physical_bytes >= snapshot.pinned_expert_bytes
    assert snapshot.failed_closed is True


def test_allocator_cache_retention_must_leave_the_ticket_peak_within_target() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=GIB,
    )

    # The total allocator footprint fell by the two registered slab GiB, but
    # one additional GiB is now allocator cache and remains charged.
    with pytest.raises(MemoryTelemetryError, match="cache retention"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(7 * GIB, GIB, 10 * GIB),
            now_ns=1,
        )

    snapshot = broker.snapshot()
    assert snapshot.charged_bytes == 109 * GIB
    assert snapshot.failed_closed is True


def test_logical_only_expert_eviction_gets_zero_credit() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="registered slab"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=10 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        )

    assert broker.snapshot().expert_slab_physical_bytes == 10 * GIB


def test_missing_allocator_telemetry_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="unavailable"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=9 * GIB,
            allocator_before=None,
            allocator_after=None,
        )

    assert broker.snapshot().failed_closed is True
    with pytest.raises(MemoryAdmissionError, match="failed closed"):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)


def test_physical_reclaim_without_timestamp_cannot_bypass_regrow_interval() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_resize_min_interval_ns=1_000,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="now_ns"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=9 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(9 * GIB, 0, 10 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 9 * GIB
    assert snapshot.last_expert_resize_ns is None
    assert snapshot.failed_closed is True
    with pytest.raises(MemoryAdmissionError, match="failed closed"):
        broker.plan_expert_regrow(target_bytes=GIB, now_ns=0)


def test_reclaim_timestamp_rollback_rejects_before_physical_mutation() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_resize_min_interval_ns=1_000,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    first = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    broker.confirm_expert_reclaim(
        first,
        registered_slab_bytes_after=9 * GIB,
        allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        allocator_after=AllocatorMemorySample(9 * GIB, 0, 10 * GIB),
        now_ns=100,
    )
    broker.abort_kv_growth(first, observed_kv_delta_bytes=0)
    second = broker.plan_kv_growth(
        steady_delta_bytes=2 * GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTransactionError, match="before last resize"):
        broker.confirm_expert_reclaim(
            second,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(9 * GIB, 0, 9 * GIB),
            allocator_after=AllocatorMemorySample(8 * GIB, 0, 9 * GIB),
            now_ns=99,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 9 * GIB
    assert snapshot.last_expert_resize_ns == 100
    assert snapshot.failed_closed is True
    with pytest.raises(MemoryAdmissionError, match="failed closed"):
        broker.plan_expert_regrow(target_bytes=GIB, now_ns=1_100)


def test_negative_allocator_delta_preserves_a_conservative_charge() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=9 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(11 * GIB, 0, 11 * GIB),
            now_ns=1,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 9 * GIB
    assert snapshot.charged_bytes >= 110 * GIB
    assert snapshot.failed_closed is True


def test_expert_release_allocator_growth_is_charged_and_records_hard_failure() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=9 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(12 * GIB, 0, 12 * GIB),
            now_ns=1,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 9 * GIB
    assert snapshot.allocator_cache_bytes == 3 * GIB
    assert snapshot.charged_bytes == 112 * GIB
    assert snapshot.hard_failure_count == 1
    assert snapshot.failed_closed is True


def test_expert_registry_growth_is_published_before_hard_failure() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="increased"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_slab_bytes_after=11 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(12 * GIB, 0, 12 * GIB),
            now_ns=1,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 11 * GIB
    assert snapshot.allocator_cache_bytes == GIB
    assert snapshot.charged_bytes == 112 * GIB
    assert snapshot.hard_failure_count == 1
    assert snapshot.failed_closed is True


def test_ticket_is_revision_checked_and_single_use() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTransactionError, match="interrupted"):
        broker.replace_snapshot(_snapshot(resident=50 * GIB))
    with pytest.raises(MemoryTransactionError, match="consumed"):
        broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)
    assert broker.snapshot().failed_closed is True


def test_ticket_revision_mismatch_branch_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    with pytest.raises(MemoryAdmissionError, match="already active"):
        broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
        )

    with pytest.raises(MemoryTransactionError, match="revision changed"):
        broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)
    assert broker.snapshot().failed_closed is True


def test_safe_pre_destructive_abort_restores_exact_snapshot() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    original = _snapshot(resident=100 * GIB, experts=10 * GIB)
    _install(broker, original)
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == original.resident_model_bytes
    assert snapshot.expert_slab_physical_bytes == original.expert_slab_physical_bytes
    assert snapshot.kv_physical_bytes == original.kv_physical_bytes
    assert snapshot.allocator_cache_bytes == original.allocator_cache_bytes
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is False


def test_known_zero_allocation_failure_preserves_confirmed_reclaim() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    broker.confirm_expert_reclaim(
        ticket,
        registered_slab_bytes_after=9 * GIB,
        allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        allocator_after=AllocatorMemorySample(9 * GIB, 0, 10 * GIB),
        now_ns=1,
    )

    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 9 * GIB
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.charged_bytes == 109 * GIB
    assert snapshot.transaction_failure_count == 1
    assert snapshot.failed_closed is False
    with pytest.raises(MemoryTransactionError, match="consumed"):
        broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)


def test_unknown_interrupted_allocation_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTransactionError, match="unknown"):
        broker.abort_kv_growth(ticket, observed_kv_delta_bytes=None)

    assert broker.snapshot().failed_closed is True
    assert broker.snapshot().pending_kv_ticket_id is None


def test_allocation_size_mismatch_is_charged_and_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTransactionError, match="planned"):
        broker.commit_kv_growth(
            ticket,
            allocated_physical_bytes=GIB + 1,
        )

    assert broker.snapshot().kv_physical_bytes == GIB + 1
    assert broker.snapshot().failed_closed is True


def test_release_is_exact_and_duplicate_release_cannot_credit_twice() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
    )

    broker.release_kv(
        allocation,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(GIB, 0, GIB),
        allocator_after=AllocatorMemorySample(0, 0, GIB),
    )
    after_first = broker.snapshot()
    assert after_first.kv_physical_bytes == 0
    assert after_first.allocator_cache_bytes == 0

    with pytest.raises(DuplicateReleaseError):
        broker.release_kv(
            allocation,
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(0, 0, GIB),
            allocator_after=AllocatorMemorySample(0, 0, GIB),
        )
    assert broker.snapshot().kv_physical_bytes == 0


def test_kv_release_active_to_cache_gets_zero_credit() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
    )
    before_charged = broker.snapshot().charged_bytes

    broker.release_kv(
        allocation,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(GIB, 0, GIB),
        allocator_after=AllocatorMemorySample(0, GIB, GIB),
    )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.allocator_cache_bytes == GIB
    assert snapshot.charged_bytes == before_charged


def test_kv_release_without_allocator_reduction_retains_charge_and_fails() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
    )
    before_charged = broker.snapshot().charged_bytes

    with pytest.raises(MemoryTelemetryError, match="did not prove"):
        broker.release_kv(
            allocation,
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(GIB, 0, GIB),
            allocator_after=AllocatorMemorySample(GIB, 0, GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.allocator_cache_bytes == GIB
    assert snapshot.charged_bytes == before_charged
    assert snapshot.failed_closed is True


def test_kv_release_allocator_growth_is_charged_and_records_hard_failure() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=109 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
    )

    with pytest.raises(MemoryTelemetryError, match="increased"):
        broker.release_kv(
            allocation,
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(GIB, 0, GIB),
            allocator_after=AllocatorMemorySample(3 * GIB, 0, 3 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.allocator_cache_bytes == 3 * GIB
    assert snapshot.charged_bytes == 112 * GIB
    assert snapshot.hard_failure_count == 1
    assert snapshot.failed_closed is True


def test_kv_release_size_mismatch_preserves_observed_truth_and_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
    )

    with pytest.raises(TerminalizedKVReleaseError, match="registered KV"):
        broker.release_kv(
            allocation,
            registered_kv_bytes_after=GIB // 2,
            allocator_before=AllocatorMemorySample(GIB, 0, GIB),
            allocator_after=AllocatorMemorySample(GIB // 2, 0, GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == GIB // 2
    assert snapshot.failed_closed is True


def test_kv_release_missing_telemetry_is_retryable_not_terminalized() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)

    with pytest.raises(MemoryTelemetryError, match="unavailable") as caught:
        broker.release_kv(
            allocation,
            registered_kv_bytes_after=0,
            allocator_before=None,
            allocator_after=None,
        )

    assert not isinstance(caught.value, TerminalizedKVReleaseError)
    assert broker.snapshot().owned_kv_physical_bytes == GIB
    broker.release_kv(
        allocation,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(GIB, 0, GIB),
        allocator_after=AllocatorMemorySample(0, 0, GIB),
    )
    assert broker.snapshot().owned_kv_physical_bytes == 0


def test_multi_growth_same_cache_releases_in_one_physical_close() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    first_ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    first = broker.commit_kv_growth(
        first_ticket,
        allocated_physical_bytes=GIB,
    )
    second_ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    second = broker.commit_kv_growth(
        second_ticket,
        allocated_physical_bytes=GIB,
    )
    assert first.cache_id == second.cache_id == "cache-a"
    assert first.allocation_id != second.allocation_id

    broker.release_kv_batch(
        cache_id="cache-a",
        allocations=(second, first),
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
        allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
    )

    assert broker.snapshot().kv_physical_bytes == 0


def test_partial_same_cache_close_preserves_observed_truth_and_fails() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    allocations = []
    for _ in range(2):
        ticket = broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
            cache_id="cache-a",
        )
        allocations.append(
            broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)
        )

    with pytest.raises(MemoryTelemetryError, match="all active allocations"):
        broker.release_kv_batch(
            cache_id="cache-a",
            allocations=(allocations[0],),
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is True


def test_wrong_aggregate_cache_close_preserves_partial_physical_truth() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    allocations = []
    for _ in range(2):
        ticket = broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
            cache_id="cache-a",
        )
        allocations.append(
            broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)
        )

    with pytest.raises(MemoryTelemetryError, match="aggregate"):
        broker.release_kv_batch(
            cache_id="cache-a",
            allocations=tuple(allocations),
            registered_kv_bytes_after=GIB,
            allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(GIB, 0, 2 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == GIB
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == GIB
    assert snapshot.failed_closed is True


def test_cross_cache_batch_close_fails_without_combining_ownership() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    handles = []
    for cache_id in ("cache-a", "cache-b"):
        ticket = broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
            cache_id=cache_id,
        )
        handles.append(broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB))

    with pytest.raises(MemoryTelemetryError, match="cross-cache"):
        broker.release_kv_batch(
            cache_id="cache-a",
            allocations=tuple(handles),
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is True


def test_batch_release_handles_cannot_be_replayed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)
    broker.release_kv_batch(
        cache_id="cache-a",
        allocations=(allocation,),
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(GIB, 0, GIB),
        allocator_after=AllocatorMemorySample(0, 0, GIB),
    )

    with pytest.raises(DuplicateReleaseError):
        broker.release_kv_batch(
            cache_id="cache-a",
            allocations=(allocation,),
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(0, 0, GIB),
            allocator_after=AllocatorMemorySample(0, 0, GIB),
        )
    assert broker.snapshot().kv_physical_bytes == 0


def test_larger_than_selected_drop_terminalizes_affected_ownership() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    handles = {}
    for cache_id in ("cache-a", "cache-b"):
        ticket = broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
            cache_id=cache_id,
        )
        handles[cache_id] = broker.commit_kv_growth(
            ticket,
            allocated_physical_bytes=GIB,
        )

    with pytest.raises(MemoryTelemetryError, match="aggregate"):
        broker.release_kv_batch(
            cache_id="cache-a",
            allocations=(handles["cache-a"],),
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 0
    with pytest.raises(DuplicateReleaseError):
        broker.release_kv(
            handles["cache-b"],
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(0, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
        )


def test_cross_cache_partial_drop_becomes_terminal_unreconciled_residual() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    handles = []
    for cache_id in ("cache-a", "cache-b"):
        ticket = broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
            cache_id=cache_id,
        )
        handles.append(broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB))

    with pytest.raises(MemoryTelemetryError, match="cross-cache"):
        broker.release_kv_batch(
            cache_id="cache-a",
            allocations=tuple(handles),
            registered_kv_bytes_after=GIB,
            allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(GIB, 0, 2 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == GIB
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == GIB
    with pytest.raises(DuplicateReleaseError):
        broker.release_kv(
            handles[0],
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(GIB, 0, 2 * GIB),
            allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
        )


def test_value_equal_forged_handle_is_rejected_before_physical_mutation() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=50 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=GIB)
    forged = replace(allocation)
    before = broker.snapshot()

    with pytest.raises(MemoryTransactionError, match="unknown"):
        broker.release_kv(
            forged,
            registered_kv_bytes_after=0,
            allocator_before=AllocatorMemorySample(GIB, 0, GIB),
            allocator_after=AllocatorMemorySample(0, 0, GIB),
        )

    after = broker.snapshot()
    assert after.kv_physical_bytes == before.kv_physical_bytes == GIB
    assert after.owned_kv_physical_bytes == GIB
    broker.release_kv(
        allocation,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(GIB, 0, GIB),
        allocator_after=AllocatorMemorySample(0, 0, GIB),
    )


def test_expert_regrow_hysteresis_and_minimum_interval() -> None:
    interval_ns = 1_000
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ns=interval_ns,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    broker.confirm_expert_reclaim(
        ticket,
        registered_slab_bytes_after=7 * GIB,
        allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        allocator_after=AllocatorMemorySample(7 * GIB, 0, 10 * GIB),
        now_ns=100,
    )
    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    assert (
        broker.plan_expert_regrow(
            target_bytes=GIB,
            now_ns=100 + interval_ns - 1,
        )
        is None
    )
    regrow = broker.plan_expert_regrow(
        target_bytes=GIB,
        now_ns=100 + interval_ns,
    )
    assert regrow is not None
    assert regrow.planned_physical_bytes == GIB

    # Planning reserves a transaction but does not publish physical bytes or
    # advance the resize clock.
    assert broker.snapshot().expert_slab_physical_bytes == 7 * GIB
    assert broker.snapshot().last_expert_resize_ns == 100
    assert (
        broker.plan_expert_regrow(
            target_bytes=GIB,
            now_ns=100 + interval_ns,
        )
        is None
    )

    broker.confirm_expert_regrow(
        regrow,
        registered_slab_bytes_after=8 * GIB,
        allocator_before=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
        allocator_after=AllocatorMemorySample(8 * GIB, 0, 8 * GIB),
        now_ns=100 + interval_ns,
    )
    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 8 * GIB
    assert snapshot.last_expert_resize_ns == 100 + interval_ns

    # A physically confirmed resize, not the earlier plan, gates another
    # otherwise-admissible same-time regrow.
    assert (
        broker.plan_expert_regrow(
            target_bytes=GIB,
            now_ns=100 + interval_ns,
        )
        is None
    )


def test_non_slab_aligned_regrow_rounds_up_and_preserves_hysteresis() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=1,
        expert_resize_min_interval_ns=0,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    broker.replace_snapshot(_snapshot(resident=100 * GIB, experts=7 * GIB))

    regrow = broker.plan_expert_regrow(
        target_bytes=GIB + 1,
        now_ns=1,
    )
    assert regrow is not None
    assert regrow.planned_slabs == 2
    assert regrow.planned_physical_bytes == 2 * GIB
    assert regrow.planned_peak_bytes + GIB == 110 * GIB
    broker.abort_expert_regrow(regrow)

    broker.replace_snapshot(_snapshot(resident=101 * GIB, experts=7 * GIB))
    assert broker.plan_expert_regrow(target_bytes=2 * GIB, now_ns=2) is None


def test_expert_regrow_ticket_is_single_use() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    broker.replace_snapshot(_snapshot(resident=100 * GIB, experts=7 * GIB))
    ticket = broker.plan_expert_regrow(target_bytes=GIB, now_ns=10)
    assert ticket is not None

    broker.confirm_expert_regrow(
        ticket,
        registered_slab_bytes_after=8 * GIB,
        allocator_before=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
        allocator_after=AllocatorMemorySample(8 * GIB, 0, 8 * GIB),
        now_ns=10,
    )

    with pytest.raises(MemoryTransactionError, match="consumed"):
        broker.confirm_expert_regrow(
            ticket,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(8 * GIB, 0, 8 * GIB),
            allocator_after=AllocatorMemorySample(8 * GIB, 0, 8 * GIB),
            now_ns=10,
        )


def test_expert_regrow_ticket_revision_mismatch_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    broker.replace_snapshot(_snapshot(resident=100 * GIB, experts=7 * GIB))
    ticket = broker.plan_expert_regrow(target_bytes=GIB, now_ns=10)
    assert ticket is not None
    with pytest.raises(MemoryAdmissionError, match="already active"):
        broker.plan_kv_growth(
            steady_delta_bytes=GIB,
            transient_delta_bytes=0,
        )

    with pytest.raises(MemoryTransactionError, match="revision changed"):
        broker.confirm_expert_regrow(
            ticket,
            registered_slab_bytes_after=7 * GIB,
            allocator_before=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
            allocator_after=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
            now_ns=10,
        )
    assert broker.snapshot().failed_closed is True


def test_expert_regrow_timestamp_before_plan_rejects_before_mutation() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    broker.replace_snapshot(_snapshot(resident=100 * GIB, experts=7 * GIB))
    ticket = broker.plan_expert_regrow(target_bytes=GIB, now_ns=1_000)
    assert ticket is not None

    with pytest.raises(MemoryTransactionError, match="before ticket plan"):
        broker.confirm_expert_regrow(
            ticket,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
            allocator_after=AllocatorMemorySample(8 * GIB, 0, 8 * GIB),
            now_ns=999,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 7 * GIB
    assert snapshot.last_expert_resize_ns is None
    assert snapshot.failed_closed is True


def test_expert_regrow_timestamp_omission_preserves_physical_without_clock() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    broker.replace_snapshot(_snapshot(resident=100 * GIB, experts=7 * GIB))
    ticket = broker.plan_expert_regrow(target_bytes=GIB, now_ns=10)
    assert ticket is not None

    with pytest.raises(MemoryTelemetryError, match="now_ns"):
        broker.confirm_expert_regrow(
            ticket,
            registered_slab_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
            allocator_after=AllocatorMemorySample(8 * GIB, 0, 8 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 8 * GIB
    assert snapshot.last_expert_resize_ns is None
    assert snapshot.failed_closed is True


def test_failed_regrow_below_pins_preserves_physical_and_classification_invariants() -> (
    None
):
    broker = UnifiedMemoryBroker.standard_hy3(
        expert_slab_bytes=GIB,
        expert_regrow_hysteresis_slabs=0,
        expert_resize_min_interval_ns=0,
    )
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    broker.replace_snapshot(
        _snapshot(
            resident=100 * GIB,
            experts=7 * GIB,
            pinned=7 * GIB,
            speculative=5 * GIB,
        )
    )
    ticket = broker.plan_expert_regrow(target_bytes=GIB, now_ns=10)
    assert ticket is not None

    with pytest.raises(MemoryTelemetryError, match="did not match"):
        broker.confirm_expert_regrow(
            ticket,
            registered_slab_bytes_after=6 * GIB,
            allocator_before=AllocatorMemorySample(7 * GIB, 0, 7 * GIB),
            allocator_after=AllocatorMemorySample(6 * GIB, 0, 7 * GIB),
            now_ns=10,
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_slab_physical_bytes == 6 * GIB
    assert snapshot.pinned_expert_bytes == 6 * GIB
    assert snapshot.speculative_expert_bytes == 5 * GIB
    assert snapshot.allocator_cache_bytes == GIB
    assert snapshot.expert_slab_physical_bytes >= snapshot.pinned_expert_bytes
    assert snapshot.speculative_expert_bytes <= snapshot.expert_slab_physical_bytes
    assert snapshot.failed_closed is True


def test_commit_kv_growth_atomically_reclassifies_consumed_allocator_cache() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, cache=2 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="target:atomic",
    )

    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
        allocator_before=AllocatorMemorySample(10 * GIB, 2 * GIB, 12 * GIB),
        allocator_after=AllocatorMemorySample(11 * GIB, GIB, 12 * GIB),
    )

    snapshot = broker.snapshot()
    assert allocation.physical_bytes == GIB
    assert snapshot.kv_physical_bytes == GIB
    assert snapshot.allocator_cache_bytes == GIB
    assert snapshot.owned_kv_physical_bytes == GIB


def test_commit_kv_growth_reconciles_allocator_cache_from_absolute_pools() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, cache=2 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="target:absolute",
    )

    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=GIB,
        allocator_before=AllocatorMemorySample(10 * GIB, GIB, 11 * GIB),
        allocator_after=AllocatorMemorySample(11 * GIB, 0, 11 * GIB),
    )

    snapshot = broker.snapshot()
    assert allocation.physical_bytes == GIB
    assert snapshot.allocator_cache_bytes == 0
    assert snapshot.charged_bytes == 101 * GIB


def test_commit_kv_growth_over_target_terminalizes_without_stranded_handle() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=108 * GIB, cache=GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="target:over-target",
    )

    with pytest.raises(MemoryTransactionError, match="operating target"):
        broker.commit_kv_growth(
            ticket,
            allocated_physical_bytes=GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, GIB, 11 * GIB),
            allocator_after=AllocatorMemorySample(
                11 * GIB,
                GIB + 1,
                12 * GIB + 1,
            ),
        )

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == GIB
    assert snapshot.allocator_cache_bytes == GIB + 1
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == GIB
    assert snapshot.failed_closed is True


def test_commit_kv_growth_missing_atomic_sample_terminalizes_unowned_bytes() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB, cache=2 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="target:ambiguous",
    )

    with pytest.raises(MemoryTelemetryError, match="allocator telemetry"):
        broker.commit_kv_growth(
            ticket,
            allocated_physical_bytes=GIB,
            allocator_before=AllocatorMemorySample(
                10 * GIB,
                2 * GIB,
                12 * GIB,
            ),
            allocator_after=None,
        )

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == GIB
    assert snapshot.allocator_cache_bytes == 2 * GIB
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == GIB
    assert snapshot.failed_closed is True


def test_abort_kv_growth_atomically_charges_allocator_cache_retention() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    snapshot = broker.abort_kv_growth(
        ticket,
        observed_kv_delta_bytes=0,
        allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        allocator_after=AllocatorMemorySample(10 * GIB, GIB, 11 * GIB),
    )

    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.allocator_cache_bytes == GIB


def test_abort_kv_growth_invalid_atomic_sample_consumes_ticket_fail_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3()
    _install(broker, _snapshot(resident=100 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator telemetry"):
        broker.abort_kv_growth(
            ticket,
            observed_kv_delta_bytes=0,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=None,
        )

    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True
