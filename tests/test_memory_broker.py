from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import threading

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
    KVAllocationGroupTicket,
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
        expert_cache_physical_bytes=experts,
        in_flight_expert_staging_bytes=staging,
        runtime_workspace_bytes=workspace,
        allocator_cache_bytes=cache,
        pinned_expert_bytes=pinned,
        speculative_expert_bytes=speculative,
    )


def _install(broker: UnifiedMemoryBroker, snapshot: BrokerSnapshot) -> None:
    broker.replace_snapshot(snapshot)


def test_budget_has_one_machine_configurable_limit() -> None:
    budget = MemoryBudget(memory_limit_bytes=100, allocator_headroom_bytes=10)

    assert budget.memory_limit_bytes == 100
    assert budget.classified_limit_bytes == 90
    assert not hasattr(budget, "hard_ceiling_bytes")
    assert not hasattr(budget, "operating_target_bytes")


def test_cache_record_growth_is_single_use_and_cap_bounded() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=BrokerSnapshot(
            resident_model_bytes=50,
            kv_physical_bytes=0,
            expert_cache_physical_bytes=20,
            in_flight_expert_staging_bytes=0,
            runtime_workspace_bytes=0,
            allocator_cache_bytes=0,
        ),
        expert_record_bytes=10,
        expert_cache_limit_bytes=30,
    )

    ticket = broker.plan_expert_cache_growth()

    assert ticket is not None
    after = broker.commit_expert_cache_growth(
        ticket,
        registered_cache_bytes_after=30,
        allocator_before=AllocatorMemorySample(70, 0, 70),
        allocator_after=AllocatorMemorySample(80, 0, 80),
    )
    assert after.expert_cache_physical_bytes == 30
    with pytest.raises(MemoryTransactionError, match="consumed"):
        broker.commit_expert_cache_growth(
            ticket,
            registered_cache_bytes_after=30,
            allocator_before=AllocatorMemorySample(70, 0, 70),
            allocator_after=AllocatorMemorySample(80, 0, 80),
        )
    assert broker.plan_expert_cache_growth() is None


def test_cache_record_growth_returns_none_at_total_limit() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=91),
        expert_record_bytes=10,
    )

    assert broker.plan_expert_cache_growth() is None


def test_cache_record_growth_excludes_active_kv_transaction() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50),
        expert_record_bytes=10,
    )
    kv_ticket = broker.plan_kv_growth(
        steady_delta_bytes=10,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTransactionError, match="already active"):
        broker.plan_expert_cache_growth()

    broker.abort_kv_growth(kv_ticket, observed_kv_delta_bytes=0)


def test_failed_cache_record_allocation_can_be_aborted_once() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50),
        expert_record_bytes=10,
    )
    ticket = broker.plan_expert_cache_growth()
    assert ticket is not None

    with pytest.raises(MemoryTelemetryError, match="exactly one"):
        broker.commit_expert_cache_growth(
            ticket,
            registered_cache_bytes_after=9,
            allocator_before=AllocatorMemorySample(50, 0, 50),
            allocator_after=AllocatorMemorySample(59, 0, 59),
        )

    snapshot = broker.abort_expert_cache_growth(ticket)
    assert snapshot.expert_cache_physical_bytes == 0
    assert snapshot.transaction_failure_count == 1
    with pytest.raises(MemoryTransactionError, match="consumed"):
        broker.abort_expert_cache_growth(ticket)


def test_cache_record_growth_retains_allocator_residual_charge() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50, cache=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_expert_cache_growth()
    assert ticket is not None

    snapshot = broker.commit_expert_cache_growth(
        ticket,
        registered_cache_bytes_after=10,
        allocator_before=AllocatorMemorySample(50, 10, 60),
        allocator_after=AllocatorMemorySample(75, 5, 80),
    )

    assert snapshot.expert_cache_physical_bytes == 10
    assert snapshot.allocator_cache_bytes == 20
    assert snapshot.charged_bytes == 80


def test_grouped_expert_growth_state_and_apis_are_absent() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        expert_record_bytes=10,
    )
    snapshot = broker.snapshot()

    assert not hasattr(snapshot, "pending_expert_regrow_ticket_id")
    assert not hasattr(snapshot, "last_expert_resize_ns")
    assert not hasattr(broker, "plan_expert_regrow")
    assert not hasattr(broker, "terminalize_expert_resize")


def test_binary_gib_budget_and_immutable_pool_accounting() -> None:
    assert BINARY_GIB == GIB
    budget = MemoryBudget(memory_limit_bytes=110 * GIB)
    assert budget.memory_limit_bytes == 110 * GIB
    assert budget.allocator_headroom_bytes == 0
    assert budget.classified_limit_bytes == 110 * GIB

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
    assert snapshot.classified_bytes == 31
    assert snapshot.pinned_expert_bytes == 4
    with pytest.raises(FrozenInstanceError):
        snapshot.kv_physical_bytes = 0  # type: ignore[misc]


def test_allocator_headroom_rejects_steady_classified_pool_overcommit() -> None:
    budget = MemoryBudget(
        memory_limit_bytes=100,
        allocator_headroom_bytes=10,
    )
    assert budget.classified_limit_bytes == 90
    broker = UnifiedMemoryBroker(budget=budget, expert_record_bytes=10)

    with pytest.raises(MemoryAdmissionError, match="classified"):
        broker.replace_snapshot(_snapshot(resident=91))


def test_kv_growth_enforces_classified_headroom_and_total_transition_peak() -> None:
    budget = MemoryBudget(
        memory_limit_bytes=100,
        allocator_headroom_bytes=10,
    )
    classified_only = UnifiedMemoryBroker(
        budget=budget,
        initial_snapshot=_snapshot(resident=50, experts=40),
        expert_record_bytes=10,
    )
    ticket = classified_only.plan_kv_growth(
        steady_delta_bytes=10,
        transient_delta_bytes=0,
    )
    assert ticket.required_expert_reclaim_bytes == 10
    classified_only.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    total_peak = UnifiedMemoryBroker(
        budget=budget,
        initial_snapshot=_snapshot(resident=40, experts=50, cache=10),
        expert_record_bytes=10,
    )
    ticket = total_peak.plan_kv_growth(
        steady_delta_bytes=10,
        transient_delta_bytes=10,
    )
    assert ticket.required_expert_reclaim_bytes == 20
    total_peak.abort_kv_growth(ticket, observed_kv_delta_bytes=0)



def test_group_kv_growth_admits_aggregate_and_reclaims_before_member_one() -> None:
    record_batch_bytes = 10_616_832 * 32
    member_bytes = 4_500_000
    member_count = 80
    slack_bytes = 215 * 1024**2
    target_bytes = 2 * GIB
    initial_charged = target_bytes - slack_bytes
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(
            memory_limit_bytes=target_bytes,
        ),
        initial_snapshot=_snapshot(
            resident=initial_charged - record_batch_bytes,
            experts=record_batch_bytes,
        ),
        expert_record_bytes=record_batch_bytes,
    )

    ticket = broker.plan_kv_growth_group(
        members=tuple(
            (f"layer-{index}", member_bytes, index % 3 * 1024**2)
            for index in range(member_count)
        )
    )

    assert isinstance(ticket, KVAllocationGroupTicket)
    assert ticket.steady_delta_bytes == member_count * member_bytes
    assert ticket.transient_delta_bytes == 2 * 1024**2
    assert ticket.required_expert_reclaim_bytes == (
        member_count * member_bytes + 2 * 1024**2 - slack_bytes
    )
    assert member_bytes < slack_bytes
    assert ticket.steady_delta_bytes > slack_bytes

    reclaimed = broker.confirm_expert_reclaim(
        ticket,
        registered_cache_bytes_after=0,
        allocator_before=AllocatorMemorySample(
            active_bytes=initial_charged,
            cache_bytes=0,
            peak_bytes=initial_charged,
        ),
        allocator_after=AllocatorMemorySample(
            active_bytes=initial_charged - record_batch_bytes,
            cache_bytes=0,
            peak_bytes=initial_charged,
        ),
    )
    assert reclaimed.expert_cache_physical_bytes == 0
    assert initial_charged - reclaimed.charged_bytes == record_batch_bytes

    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="layer-0",
        allocated_physical_bytes=member_bytes,
        allocator_before=AllocatorMemorySample(
            active_bytes=initial_charged - record_batch_bytes,
            cache_bytes=0,
            peak_bytes=initial_charged,
        ),
        allocator_after=AllocatorMemorySample(
            active_bytes=initial_charged - record_batch_bytes + member_bytes,
            cache_bytes=0,
            peak_bytes=initial_charged,
        ),
    )
    assert allocation.cache_id == "layer-0"
    assert allocation.physical_bytes == member_bytes
    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == member_bytes
    assert snapshot.owned_kv_physical_bytes == member_bytes
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.pending_kv_ticket_id == ticket.ticket_id


def test_group_kv_zero_abort_preserves_commits_and_confirmed_reclaim() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=60, experts=30),
        expert_record_bytes=30,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 20, 0))
    )
    assert ticket.required_expert_reclaim_bytes == 20
    broker.confirm_expert_reclaim(
        ticket,
        registered_cache_bytes_after=0,
        allocator_before=AllocatorMemorySample(90, 0, 90),
        allocator_after=AllocatorMemorySample(60, 0, 90),
    )
    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=7,
        allocator_before=AllocatorMemorySample(60, 0, 90),
        allocator_after=AllocatorMemorySample(67, 0, 90),
    )

    snapshot = broker.abort_kv_growth_group(
        ticket,
        observed_uncommitted_kv_delta_bytes=0,
    )

    assert allocation.physical_bytes == 7
    assert snapshot.expert_cache_physical_bytes == 0
    assert snapshot.kv_physical_bytes == 7
    assert snapshot.owned_kv_physical_bytes == 7
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is False
    assert snapshot.pending_kv_ticket_id is None


def test_group_kv_finalizes_only_after_every_member_commits_measured_bytes() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 2), ("cache-b", 20, 5))
    )

    first = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=7,
        allocator_before=AllocatorMemorySample(50, 0, 50),
        allocator_after=AllocatorMemorySample(57, 0, 57),
    )
    assert broker.snapshot().pending_kv_ticket_id == ticket.ticket_id
    second = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-b",
        allocated_physical_bytes=20,
        allocator_before=AllocatorMemorySample(57, 0, 57),
        allocator_after=AllocatorMemorySample(77, 0, 77),
    )

    assert first.cache_id == "cache-a"
    assert first.physical_bytes == 7
    assert second.cache_id == "cache-b"
    assert second.physical_bytes == 20
    assert first.allocation_id != second.allocation_id
    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == 27
    assert snapshot.owned_kv_physical_bytes == 27
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is False
    with pytest.raises(MemoryTransactionError, match="already consumed"):
        broker.commit_kv_growth_group_member(
            ticket,
            cache_id="cache-b",
            allocated_physical_bytes=20,
            allocator_before=AllocatorMemorySample(57, 0, 57),
            allocator_after=AllocatorMemorySample(77, 0, 77),
        )


@pytest.mark.parametrize("retained", [3, 300])
def test_group_kv_known_abort_residual_is_charged_and_fails_closed(
    retained: int,
) -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 20, 0))
    )
    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=7,
        allocator_before=AllocatorMemorySample(10, 0, 10),
        allocator_after=AllocatorMemorySample(17, 0, 17),
    )

    snapshot = broker.abort_kv_growth_group(
        ticket,
        observed_uncommitted_kv_delta_bytes=retained,
    )

    assert allocation.cache_id == "cache-a"
    assert snapshot.kv_physical_bytes == 7 + retained
    assert snapshot.owned_kv_physical_bytes == 7
    assert snapshot.unreconciled_kv_physical_bytes == retained
    assert snapshot.failed_closed is True
    assert snapshot.pending_kv_ticket_id is None
    with pytest.raises(MemoryAdmissionError):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)


def test_group_kv_unknown_abort_fails_closed_without_inventing_bytes() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 20, 0))
    )

    with pytest.raises(MemoryTransactionError, match="unknown physical delta"):
        broker.abort_kv_growth_group(
            ticket,
            observed_uncommitted_kv_delta_bytes=None,
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is True
    assert snapshot.pending_kv_ticket_id is None


def test_group_kv_rejects_stale_first_member_allocator_baseline() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )

    with pytest.raises(MemoryTelemetryError, match="stale"):
        broker.commit_kv_growth_group_member(
            ticket,
            cache_id="cache-a",
            allocated_physical_bytes=10,
            allocator_before=AllocatorMemorySample(60, 0, 60),
            allocator_after=AllocatorMemorySample(70, 0, 70),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 10
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 10
    assert snapshot.failed_closed is True
    assert snapshot.pending_kv_ticket_id is None


def test_group_kv_rejects_overlapping_member_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    first_inside_commit = threading.Event()
    release_first = threading.Event()
    original = broker._allocator_cache_after_growth

    def blocking_cache_reconciliation(**kwargs: object) -> int:
        if kwargs["context"] == "KV group commit for cache-a":
            first_inside_commit.set()
            assert release_first.wait(timeout=2)
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        broker,
        "_allocator_cache_after_growth",
        blocking_cache_reconciliation,
    )
    outcomes: list[object] = []

    def commit(
        cache_id: str,
        before: AllocatorMemorySample,
        after: AllocatorMemorySample,
    ) -> None:
        try:
            outcomes.append(
                broker.commit_kv_growth_group_member(
                    ticket,
                    cache_id=cache_id,
                    allocated_physical_bytes=10,
                    allocator_before=before,
                    allocator_after=after,
                )
            )
        except Exception as exc:  # noqa: BLE001 - thread result is asserted below
            outcomes.append(exc)

    first = threading.Thread(
        target=commit,
        args=(
            "cache-a",
            AllocatorMemorySample(50, 0, 50),
            AllocatorMemorySample(60, 0, 60),
        ),
    )
    second = threading.Thread(
        target=commit,
        args=(
            "cache-b",
            AllocatorMemorySample(60, 0, 60),
            AllocatorMemorySample(70, 0, 70),
        ),
    )
    first.start()
    assert first_inside_commit.wait(timeout=2)
    second.start()
    assert second.is_alive()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive()
    assert not second.is_alive()

    allocations = [item for item in outcomes if not isinstance(item, Exception)]
    failures = [item for item in outcomes if isinstance(item, Exception)]
    assert len(allocations) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], MemoryTransactionError)
    assert "concurrent" in str(failures[0])
    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 20
    assert snapshot.owned_kv_physical_bytes == 10
    assert snapshot.unreconciled_kv_physical_bytes == 10
    assert snapshot.failed_closed is True



def test_group_kv_rejects_duplicate_owners_and_aggregate_pinned_overcommit() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50, experts=30, pinned=30),
        expert_record_bytes=30,
    )

    with pytest.raises(ValueError, match="duplicate.*cache-a"):
        broker.plan_kv_growth_group(members=(("cache-a", 15, 0), (" cache-a ", 15, 0)))
    with pytest.raises(MemoryAdmissionError, match="pinned"):
        broker.plan_kv_growth_group(members=(("cache-a", 15, 0), ("cache-b", 15, 0)))


def test_group_kv_uses_serialized_peak_instead_of_summing_member_transients() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=70),
        expert_record_bytes=10,
    )

    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 5, 15), ("cache-b", 5, 15))
    )

    assert ticket.steady_delta_bytes == 10
    assert ticket.transient_delta_bytes == 15
    assert ticket.planned_steady_bytes == 80
    assert ticket.planned_peak_bytes == 95
    broker.abort_kv_growth_group(
        ticket,
        observed_uncommitted_kv_delta_bytes=0,
    )


@pytest.mark.parametrize(
    ("cache_id", "allocated", "error_pattern"),
    [
        ("unknown-cache", 4, "unknown KV group member"),
        ("cache-a", 11, "exceeds group member plan"),
    ],
)
def test_group_kv_unknown_or_oversize_member_is_unreconciled_fail_closed(
    cache_id: str,
    allocated: int,
    error_pattern: str,
) -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )

    with pytest.raises(MemoryTransactionError, match=error_pattern):
        broker.commit_kv_growth_group_member(
            ticket,
            cache_id=cache_id,
            allocated_physical_bytes=allocated,
            allocator_before=AllocatorMemorySample(10, 0, 10),
            allocator_after=AllocatorMemorySample(10 + allocated, 0, 10 + allocated),
        )

    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == allocated
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == allocated
    assert snapshot.failed_closed is True
    assert snapshot.pending_kv_ticket_id is None


def test_group_kv_duplicate_member_commit_is_rejected_without_double_charge() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=7,
        allocator_before=AllocatorMemorySample(10, 0, 10),
        allocator_after=AllocatorMemorySample(17, 0, 17),
    )

    with pytest.raises(MemoryTransactionError, match="already committed"):
        broker.commit_kv_growth_group_member(
            ticket,
            cache_id="cache-a",
            allocated_physical_bytes=7,
            allocator_before=AllocatorMemorySample(10, 0, 10),
            allocator_after=AllocatorMemorySample(17, 0, 17),
        )

    snapshot = broker.snapshot()
    assert allocation.physical_bytes == 7
    assert snapshot.kv_physical_bytes == 7
    assert snapshot.owned_kv_physical_bytes == 7
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is True


def test_group_kv_accepts_active_cache_reclassification_at_equal_footprint() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    first = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=10,
        allocator_before=AllocatorMemorySample(0, 0, 100),
        allocator_after=AllocatorMemorySample(16, 0, 100),
    )

    second = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-b",
        allocated_physical_bytes=10,
        allocator_before=AllocatorMemorySample(10, 6, 100),
        allocator_after=AllocatorMemorySample(20, 6, 100),
    )

    snapshot = broker.snapshot()
    assert first.physical_bytes == second.physical_bytes == 10
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == 20
    assert snapshot.owned_kv_physical_bytes == 20
    assert snapshot.allocator_cache_bytes == 6
    assert snapshot.charged_bytes == 26
    assert snapshot.failed_closed is False


def test_group_kv_downward_footprint_drift_never_credits_allocator_cache() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=10,
        allocator_before=AllocatorMemorySample(0, 0, 100),
        allocator_after=AllocatorMemorySample(16, 0, 100),
    )

    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-b",
        allocated_physical_bytes=10,
        allocator_before=AllocatorMemorySample(9, 6, 100),
        allocator_after=AllocatorMemorySample(25, 0, 100),
    )

    snapshot = broker.snapshot()
    assert allocation.physical_bytes == 10
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == 20
    assert snapshot.owned_kv_physical_bytes == 20
    assert snapshot.allocator_cache_bytes == 6
    assert snapshot.charged_bytes == 26
    assert snapshot.failed_closed is False


@pytest.mark.parametrize(
    "allocator_before",
    [
        pytest.param(
            AllocatorMemorySample(10, 7, 100),
            id="one-byte-footprint-drift",
        ),
        pytest.param(
            AllocatorMemorySample(10, 6, 99),
            id="regressing-peak",
        ),
    ],
)
def test_group_kv_reclassification_rejects_upward_drift_or_peak_contradiction(
    allocator_before: AllocatorMemorySample,
) -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=10,
        allocator_before=AllocatorMemorySample(0, 0, 100),
        allocator_after=AllocatorMemorySample(16, 0, 100),
    )

    with pytest.raises(MemoryTelemetryError, match="not serialized") as exc_info:
        broker.commit_kv_growth_group_member(
            ticket,
            cache_id="cache-b",
            allocated_physical_bytes=10,
            allocator_before=allocator_before,
            allocator_after=AllocatorMemorySample(20, 6, 100),
        )

    assert getattr(exc_info.value, "transaction_terminalized", False) is True
    snapshot = broker.snapshot()
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is True


def test_group_kv_abort_accepts_active_cache_reclassification_at_equal_footprint() -> (
    None
):
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=10,
        allocator_before=AllocatorMemorySample(0, 0, 100),
        allocator_after=AllocatorMemorySample(16, 0, 100),
    )

    snapshot = broker.abort_kv_growth_group(
        ticket,
        observed_uncommitted_kv_delta_bytes=0,
        allocator_before=AllocatorMemorySample(10, 6, 100),
        allocator_after=AllocatorMemorySample(10, 6, 100),
    )

    assert allocation.physical_bytes == 10
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.kv_physical_bytes == 10
    assert snapshot.owned_kv_physical_bytes == 10
    assert snapshot.allocator_cache_bytes == 6
    assert snapshot.charged_bytes == 16
    assert snapshot.failed_closed is False


def test_group_kv_abort_rejects_ambiguous_allocator_chain_after_partial_commit() -> (
    None
):
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth_group(
        members=(("cache-a", 10, 0), ("cache-b", 10, 0))
    )
    allocation = broker.commit_kv_growth_group_member(
        ticket,
        cache_id="cache-a",
        allocated_physical_bytes=7,
        allocator_before=AllocatorMemorySample(10, 0, 10),
        allocator_after=AllocatorMemorySample(17, 0, 17),
    )

    with pytest.raises(MemoryTelemetryError, match="not serialized"):
        broker.abort_kv_growth_group(
            ticket,
            observed_uncommitted_kv_delta_bytes=0,
            allocator_before=AllocatorMemorySample(10, 0, 10),
            allocator_after=AllocatorMemorySample(10, 0, 10),
        )

    snapshot = broker.snapshot()
    assert allocation.physical_bytes == 7
    assert snapshot.kv_physical_bytes == 7
    assert snapshot.owned_kv_physical_bytes == 7
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is True
    assert snapshot.pending_kv_ticket_id is None


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
    ],
)
def test_configured_memory_limit_boundary(charged: int, allowed: bool) -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)

    if allowed:
        broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=charged))
        assert broker.snapshot().charged_bytes == charged
    else:
        with pytest.raises(MemoryAdmissionError):
            broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=charged))
        assert broker.snapshot().charged_bytes == charged

    assert broker.snapshot().hard_failure_count == int(not allowed)


def test_over_limit_snapshot_rejects_all_later_allocation() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    with pytest.raises(MemoryAdmissionError):
        broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=112 * GIB))

    with pytest.raises(MemoryAdmissionError, match="configured limit"):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)


def test_allocator_cache_reconciliation_updates_only_the_cache_pool() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(
            resident=40,
            kv=10,
            experts=20,
            staging=5,
            workspace=10,
            cache=5,
        ),
        expert_record_bytes=10,
    )

    snapshot = broker.reconcile_allocator_cache(
        AllocatorMemorySample(active_bytes=75, cache_bytes=9, peak_bytes=90)
    )

    assert snapshot.resident_model_bytes == 40
    assert snapshot.kv_physical_bytes == 10
    assert snapshot.expert_cache_physical_bytes == 20
    assert snapshot.in_flight_expert_staging_bytes == 5
    assert snapshot.runtime_workspace_bytes == 10
    assert snapshot.allocator_cache_bytes == 9
    assert snapshot.charged_bytes == 94


def test_allocator_cache_reconciliation_grants_no_credit_for_cache_to_active_move() -> (
    None
):
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(
            resident=40,
            kv=10,
            experts=20,
            staging=5,
            workspace=10,
            cache=9,
        ),
        expert_record_bytes=10,
    )

    snapshot = broker.reconcile_allocator_cache(
        AllocatorMemorySample(active_bytes=79, cache_bytes=5, peak_bytes=90)
    )

    assert snapshot.allocator_cache_bytes == 9
    assert snapshot.charged_bytes == 94


def test_reclaim_accepts_conservative_allocator_cache_overcharge() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50, experts=40, cache=10),
        expert_record_bytes=10,
    )

    reconciled = broker.reconcile_allocator_cache(
        AllocatorMemorySample(active_bytes=95, cache_bytes=0, peak_bytes=95)
    )
    assert reconciled.allocator_cache_bytes == 10

    ticket = broker.plan_kv_growth(
        steady_delta_bytes=10,
        transient_delta_bytes=0,
    )
    snapshot = broker.confirm_expert_reclaim(
        ticket,
        registered_cache_bytes_after=30,
        allocator_before=AllocatorMemorySample(
            active_bytes=95,
            cache_bytes=0,
            peak_bytes=95,
        ),
        allocator_after=AllocatorMemorySample(
            active_bytes=85,
            cache_bytes=0,
            peak_bytes=95,
        ),
    )

    assert snapshot.expert_cache_physical_bytes == 30
    assert snapshot.allocator_cache_bytes == 10
    assert snapshot.failed_closed is False


def test_allocator_cache_reconciliation_charges_unclassified_active_drift() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(
            resident=40,
            kv=10,
            experts=20,
            staging=5,
            workspace=10,
            cache=5,
        ),
        expert_record_bytes=10,
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
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=90, cache=5),
        expert_record_bytes=10,
    )

    with pytest.raises(MemoryAdmissionError, match="memory limit"):
        broker.reconcile_allocator_cache(
            AllocatorMemorySample(active_bytes=90, cache_bytes=11, peak_bytes=101)
        )

    snapshot = broker.snapshot()
    assert snapshot.allocator_cache_bytes == 11
    assert snapshot.charged_bytes == 101
    with pytest.raises(MemoryAdmissionError, match="configured limit"):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)


def test_allocator_cache_reconciliation_cannot_interrupt_a_transaction() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50, cache=5),
        expert_record_bytes=10,
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
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=40, experts=20, workspace=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=4,
        transient_delta_bytes=0,
        cache_id="target:existing",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=4)

    snapshot = broker.reconcile_post_load_classification(
        resident_model_bytes=45,
        expert_cache_physical_bytes=20,
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
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(
            resident=40,
            experts=20,
            staging=5,
            workspace=10,
            cache=10,
        ),
        expert_record_bytes=10,
    )

    snapshot = broker.reconcile_post_load_classification(
        resident_model_bytes=45,
        expert_cache_physical_bytes=20,
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
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(
            resident=40,
            experts=20,
            staging=5,
            workspace=10,
        ),
        initial_allocator_sample=initial_sample,
        expert_record_bytes=10,
    )

    snapshot = broker.reconcile_post_load_classification(
        resident_model_bytes=45,
        expert_cache_physical_bytes=20,
        in_flight_expert_staging_bytes=5,
        runtime_workspace_bytes=5,
        allocator_before=initial_sample,
        allocator_after=AllocatorMemorySample(70, 0, 70),
    )

    assert snapshot.allocator_cache_bytes == 0
    assert snapshot.charged_bytes == 75


def test_post_load_classification_rejects_active_ticket_without_mutation() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=40, experts=20, workspace=10),
        expert_record_bytes=10,
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
            expert_cache_physical_bytes=20,
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
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=40, experts=20, workspace=10),
        expert_record_bytes=10,
    )
    kwargs = {
        "resident_model_bytes": 41,
        "expert_cache_physical_bytes": 20,
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
        registered_cache_bytes_after=8 * GIB,
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
    )
    allocation = broker.commit_kv_growth(
        ticket,
        allocated_physical_bytes=1 * GIB,
    )

    assert allocation.physical_bytes == 1 * GIB
    assert broker.snapshot().charged_bytes == 109 * GIB
    assert broker.snapshot().kv_physical_bytes == 1 * GIB


def test_transient_peak_requires_reclaim_even_when_steady_state_fits() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=108 * GIB, experts=2 * GIB))

    ticket = broker.plan_kv_growth(
        steady_delta_bytes=1,
        transient_delta_bytes=GIB - 1,
    )

    assert ticket.required_expert_reclaim_bytes == GIB
    assert ticket.planned_steady_bytes == 109 * GIB + 1
    assert ticket.planned_peak_bytes == 110 * GIB


def test_pinned_expert_shortfall_rejects_without_reserving() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=GIB,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint") as caught:
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=8 * GIB,
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
        )
    assert not isinstance(caught.value, TerminalizedKVReleaseError)

    # Active-to-cache movement is reclassification, not physical reclaim.
    snapshot = broker.snapshot()
    assert snapshot.expert_cache_physical_bytes == 8 * GIB
    assert snapshot.allocator_cache_bytes == 2 * GIB
    assert snapshot.charged_bytes == 110 * GIB
    assert snapshot.failed_closed is True


def test_reclaim_below_pinned_bytes_preserves_physical_truth_and_invariant() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
            registered_cache_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(8 * GIB, 0, 10 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_cache_physical_bytes == 8 * GIB
    assert snapshot.pinned_expert_bytes == 8 * GIB
    assert snapshot.expert_cache_physical_bytes >= snapshot.pinned_expert_bytes
    assert snapshot.failed_closed is True


def test_allocator_cache_retention_must_leave_the_ticket_peak_within_target() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=GIB,
    )

    # The total allocator footprint fell by the two registered cache GiB, but
    # one additional GiB is now allocator cache and remains charged.
    with pytest.raises(MemoryTelemetryError, match="cache retention"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=8 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(7 * GIB, GIB, 10 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.charged_bytes == 109 * GIB
    assert snapshot.failed_closed is True


def test_logical_only_expert_eviction_gets_zero_credit() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="registered cache"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=10 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        )

    assert broker.snapshot().expert_cache_physical_bytes == 10 * GIB


def test_missing_allocator_telemetry_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="unavailable"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=9 * GIB,
            allocator_before=None,
            allocator_after=None,
        )

    assert broker.snapshot().failed_closed is True
    with pytest.raises(MemoryAdmissionError, match="failed closed"):
        broker.plan_kv_growth(steady_delta_bytes=1, transient_delta_bytes=0)




def test_negative_allocator_delta_preserves_a_conservative_charge() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=9 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(11 * GIB, 0, 11 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_cache_physical_bytes == 9 * GIB
    assert snapshot.charged_bytes >= 110 * GIB
    assert snapshot.failed_closed is True


def test_expert_release_allocator_growth_is_charged_and_records_hard_failure() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="allocator footprint"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=9 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(12 * GIB, 0, 12 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_cache_physical_bytes == 9 * GIB
    assert snapshot.allocator_cache_bytes == 3 * GIB
    assert snapshot.charged_bytes == 112 * GIB
    assert snapshot.hard_failure_count == 1
    assert snapshot.failed_closed is True


def test_expert_registry_growth_is_published_before_hard_failure() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    with pytest.raises(MemoryTelemetryError, match="increased"):
        broker.confirm_expert_reclaim(
            ticket,
            registered_cache_bytes_after=11 * GIB,
            allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
            allocator_after=AllocatorMemorySample(12 * GIB, 0, 12 * GIB),
        )

    snapshot = broker.snapshot()
    assert snapshot.expert_cache_physical_bytes == 11 * GIB
    assert snapshot.allocator_cache_bytes == GIB
    assert snapshot.charged_bytes == 112 * GIB
    assert snapshot.hard_failure_count == 1
    assert snapshot.failed_closed is True


def test_ticket_is_revision_checked_and_single_use() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    original = _snapshot(resident=100 * GIB, experts=10 * GIB)
    _install(broker, original)
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )

    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    snapshot = broker.snapshot()
    assert snapshot.resident_model_bytes == original.resident_model_bytes
    assert snapshot.expert_cache_physical_bytes == original.expert_cache_physical_bytes
    assert snapshot.kv_physical_bytes == original.kv_physical_bytes
    assert snapshot.allocator_cache_bytes == original.allocator_cache_bytes
    assert snapshot.pending_kv_ticket_id is None
    assert snapshot.failed_closed is False


def test_known_zero_allocation_failure_preserves_confirmed_reclaim() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=100 * GIB, experts=10 * GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
    )
    broker.confirm_expert_reclaim(
        ticket,
        registered_cache_bytes_after=9 * GIB,
        allocator_before=AllocatorMemorySample(10 * GIB, 0, 10 * GIB),
        allocator_after=AllocatorMemorySample(9 * GIB, 0, 10 * GIB),
    )

    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)

    snapshot = broker.snapshot()
    assert snapshot.expert_cache_physical_bytes == 9 * GIB
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.charged_bytes == 109 * GIB
    assert snapshot.transaction_failure_count == 1
    assert snapshot.failed_closed is False
    with pytest.raises(MemoryTransactionError, match="consumed"):
        broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)


def test_unknown_interrupted_allocation_fails_closed() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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


def test_kv_release_absorbs_new_allocator_residual_before_crediting_release() -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=40, cache=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=10,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=10)

    snapshot = broker.release_kv(
        allocation,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(65, 0, 65),
        allocator_after=AllocatorMemorySample(55, 0, 65),
    )

    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.allocator_cache_bytes == 15
    assert snapshot.charged_bytes == 55
    assert snapshot.failed_closed is False


@pytest.mark.parametrize(
    "allocator_before_bytes",
    [
        105,
        115,
    ],
)
def test_kv_release_latches_pre_release_allocator_limit_violation(
    allocator_before_bytes: int,
) -> None:
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=40, cache=10),
        expert_record_bytes=10,
    )
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=10,
        transient_delta_bytes=0,
        cache_id="cache-a",
    )
    allocation = broker.commit_kv_growth(ticket, allocated_physical_bytes=10)

    snapshot = broker.release_kv(
        allocation,
        registered_kv_bytes_after=0,
        allocator_before=AllocatorMemorySample(
            allocator_before_bytes,
            0,
            allocator_before_bytes,
        ),
        allocator_after=AllocatorMemorySample(90, 0, allocator_before_bytes),
    )

    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.charged_bytes == 90
    assert snapshot.admission_failure_count == 1
    assert snapshot.transaction_failure_count == 1
    assert snapshot.hard_failure_count == 0
    assert snapshot.failed_closed is True
    assert "memory limit" in (snapshot.failure_reason or "")


def test_kv_release_without_allocator_reduction_retains_charge_and_fails() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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


def test_concurrent_cache_closes_derive_each_release_from_locked_owner_truth() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def release(cache_id: str) -> None:
        try:
            barrier.wait(timeout=1)
            broker.release_kv_batch(
                cache_id=cache_id,
                allocations=(handles[cache_id],),
                registered_kv_bytes_after=None,
                allocator_before=AllocatorMemorySample(2 * GIB, 0, 2 * GIB),
                allocator_after=AllocatorMemorySample(0, 0, 2 * GIB),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=release, args=(cache_id,)) for cache_id in handles
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    snapshot = broker.snapshot()
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.failed_closed is False


def test_partial_same_cache_close_preserves_observed_truth_and_fails() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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









def test_commit_kv_growth_atomically_reclassifies_consumed_allocator_cache() -> None:
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
    _install(broker, _snapshot(resident=108 * GIB, cache=GIB))
    ticket = broker.plan_kv_growth(
        steady_delta_bytes=GIB,
        transient_delta_bytes=0,
        cache_id="target:over-target",
    )

    with pytest.raises(MemoryTransactionError, match="memory limit"):
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
    broker = UnifiedMemoryBroker.standard_hy3(expert_record_bytes=GIB)
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
