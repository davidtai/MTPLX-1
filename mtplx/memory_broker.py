"""Fail-closed unified-memory accounting for the Hy3 dynamic-memory lane.

The broker is deliberately allocator-agnostic.  It serializes byte-accounting
transactions, but callers remain responsible for choosing expert slabs and for
performing MLX allocations on the correct owner thread.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from threading import Lock, RLock


BINARY_GIB = 1024**3
HY3_Q4_ALLOCATOR_HEADROOM_BYTES = BINARY_GIB
HY3_Q4_KV_BLOCK_TOKENS = 16
HY3_Q4_KV_LAYERS = 80
HY3_Q4_KV_HEADS = 8
HY3_Q4_KV_HEAD_DIM = 128
HY3_Q4_KV_SCALE_BYTES = 2
_HY3_Q4_PACKED_HEAD_BYTES = HY3_Q4_KV_HEAD_DIM // 2
HY3_Q4_KV_BYTES_PER_TOKEN = (
    HY3_Q4_KV_LAYERS
    * 2
    * HY3_Q4_KV_HEADS
    * (_HY3_Q4_PACKED_HEAD_BYTES + HY3_Q4_KV_SCALE_BYTES)
)
HY3_Q4_KV_BLOCK_BYTES = HY3_Q4_KV_BLOCK_TOKENS * HY3_Q4_KV_BYTES_PER_TOKEN


class MemoryBrokerError(RuntimeError):
    """Base error for fail-closed memory-broker operations."""


class MemoryAdmissionError(MemoryBrokerError):
    """A proposed allocation cannot be admitted safely."""


class MemoryTelemetryError(MemoryBrokerError):
    """Physical allocator telemetry did not prove a requested release."""


class TerminalizedKVReleaseError(MemoryTelemetryError):
    """A KV release failed after its selected handles became terminal."""

    terminalized = True


class MemoryTransactionError(MemoryBrokerError):
    """A two-phase memory transaction was invalid or interrupted."""


class DuplicateReleaseError(MemoryTransactionError):
    """A physical allocation was reported released more than once."""


def _exact_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _exact_positive_int(name: str, value: object) -> int:
    parsed = _exact_nonnegative_int(name, value)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class MemoryBudget:
    """Binary-byte limits for normal operation and the stress ceiling."""

    operating_target_bytes: int = 110 * BINARY_GIB
    hard_ceiling_bytes: int = 112 * BINARY_GIB
    allocator_headroom_bytes: int = 0

    def __post_init__(self) -> None:
        operating = _exact_positive_int(
            "operating_target_bytes", self.operating_target_bytes
        )
        hard = _exact_positive_int("hard_ceiling_bytes", self.hard_ceiling_bytes)
        headroom = _exact_nonnegative_int(
            "allocator_headroom_bytes", self.allocator_headroom_bytes
        )
        if operating >= hard:
            raise ValueError("operating_target_bytes must be below hard_ceiling_bytes")
        if headroom >= operating:
            raise ValueError(
                "allocator_headroom_bytes must be below operating_target_bytes"
            )

    @property
    def classified_target_bytes(self) -> int:
        return self.operating_target_bytes - self.allocator_headroom_bytes


@dataclass(frozen=True)
class Hy3Q4KVPhysicalGeometry:
    """Exact block-rounded Hy3 Q4 KV geometry, independent of an allocator."""

    requested_tokens: int
    physical_blocks: int
    physical_capacity_tokens: int
    physical_bytes: int


def hy3_q4_kv_physical_geometry(tokens: int) -> Hy3Q4KVPhysicalGeometry:
    """Return packed K/V plus fp16 K/V-scale storage, rounded to blocks.

    Slot mappings and block tables are transient runtime workspace and are not
    persistent KV bytes in this geometry.
    """

    requested = _exact_nonnegative_int("tokens", tokens)
    blocks = (requested + HY3_Q4_KV_BLOCK_TOKENS - 1) // HY3_Q4_KV_BLOCK_TOKENS
    return Hy3Q4KVPhysicalGeometry(
        requested_tokens=requested,
        physical_blocks=blocks,
        physical_capacity_tokens=blocks * HY3_Q4_KV_BLOCK_TOKENS,
        physical_bytes=blocks * HY3_Q4_KV_BLOCK_BYTES,
    )


@dataclass(frozen=True)
class AllocatorMemorySample:
    """One physical allocator observation.

    Active plus cached bytes is the charged allocator footprint used to prove
    release.  Peak is retained for diagnostics and is not added to the steady
    memory total.
    """

    active_bytes: int
    cache_bytes: int
    peak_bytes: int

    @property
    def charged_footprint_bytes(self) -> int:
        return self.active_bytes + self.cache_bytes


@dataclass(frozen=True)
class BrokerSnapshot:
    """Immutable classified physical-memory snapshot.

    Pinned and speculative expert bytes describe the expert pool and are not
    charged a second time.
    """

    resident_model_bytes: int
    kv_physical_bytes: int
    expert_slab_physical_bytes: int
    in_flight_expert_staging_bytes: int
    runtime_workspace_bytes: int
    allocator_cache_bytes: int
    pinned_expert_bytes: int = 0
    speculative_expert_bytes: int = 0
    revision: int = 0
    hard_failure_count: int = 0
    admission_failure_count: int = 0
    transaction_failure_count: int = 0
    failed_closed: bool = False
    failure_reason: str | None = None
    pending_kv_ticket_id: int | None = None
    pending_expert_regrow_ticket_id: int | None = None
    last_expert_resize_ns: int | None = None
    owned_kv_physical_bytes: int = 0
    unreconciled_kv_physical_bytes: int = 0

    @property
    def charged_bytes(self) -> int:
        return (
            self.resident_model_bytes
            + self.kv_physical_bytes
            + self.expert_slab_physical_bytes
            + self.in_flight_expert_staging_bytes
            + self.runtime_workspace_bytes
            + self.allocator_cache_bytes
        )

    @property
    def classified_bytes(self) -> int:
        return (
            self.resident_model_bytes
            + self.kv_physical_bytes
            + self.expert_slab_physical_bytes
            + self.in_flight_expert_staging_bytes
            + self.runtime_workspace_bytes
        )

    @property
    def reclaimable_expert_bytes(self) -> int:
        """Expert bytes not protected by a physical pin."""

        return max(
            0,
            self.expert_slab_physical_bytes - self.pinned_expert_bytes,
        )

    @property
    def speculative_reclaimable_bytes(self) -> int:
        """Lower-priority speculative bytes within the reclaimable pool."""

        return min(
            self.speculative_expert_bytes,
            self.reclaimable_expert_bytes,
        )

    @classmethod
    def synthetic(cls, *, charged_bytes: int) -> BrokerSnapshot:
        """Build a boundary-test snapshot with all bytes classified resident."""

        return cls(
            resident_model_bytes=charged_bytes,
            kv_physical_bytes=0,
            expert_slab_physical_bytes=0,
            in_flight_expert_staging_bytes=0,
            runtime_workspace_bytes=0,
            allocator_cache_bytes=0,
        )


@dataclass(frozen=True)
class KVAllocationTicket:
    """Single-use reservation for one physical KV growth operation.

    ``transient_delta_bytes`` is additional peak memory above the final steady
    allocation, not the full peak allocation size.
    """

    ticket_id: int
    cache_id: str
    steady_delta_bytes: int
    transient_delta_bytes: int
    required_expert_reclaim_bytes: int
    snapshot_revision: int
    planned_steady_bytes: int
    planned_peak_bytes: int


@dataclass(frozen=True)
class KVPhysicalAllocation:
    """Exact-once handle returned after committed physical KV growth."""

    allocation_id: int
    cache_id: str
    physical_bytes: int
    committed_revision: int


@dataclass(frozen=True)
class KVAllocationGroupMember:
    """One declared owner within an aggregate KV growth reservation."""

    allocation_id: int
    cache_id: str
    steady_delta_bytes: int
    transient_delta_bytes: int


@dataclass(frozen=True)
class KVAllocationGroupTicket:
    """Single reclaim permit covering serialized growth by many KV owners."""

    ticket_id: int
    members: tuple[KVAllocationGroupMember, ...]
    steady_delta_bytes: int
    transient_delta_bytes: int
    required_expert_reclaim_bytes: int
    snapshot_revision: int
    planned_steady_bytes: int
    planned_peak_bytes: int


@dataclass(frozen=True)
class ExpertRegrowTicket:
    """Single-use reservation for a slab-aligned physical expert regrow."""

    ticket_id: int
    requested_target_bytes: int
    planned_slabs: int
    planned_physical_bytes: int
    snapshot_revision: int
    planned_peak_bytes: int
    planned_at_ns: int


@dataclass
class _PendingKVTransaction:
    ticket: KVAllocationTicket
    expected_revision: int
    reclaim_confirmed: bool = False


@dataclass
class _PendingKVGroupTransaction:
    ticket: KVAllocationGroupTicket
    expected_revision: int
    reclaim_confirmed: bool = False
    committed_cache_ids: set[str] = field(default_factory=set)
    last_allocator_sample: AllocatorMemorySample | None = None


@dataclass
class _PendingExpertRegrow:
    ticket: ExpertRegrowTicket
    expected_revision: int


class UnifiedMemoryBroker:
    """Lock-protected authoritative accounting and two-phase reservations."""

    _DEFAULT_EXPERT_RECORD_BYTES = 10_616_832
    _DEFAULT_EXPERT_SLAB_SLOTS = 32

    def __init__(
        self,
        *,
        budget: MemoryBudget | None = None,
        initial_snapshot: BrokerSnapshot | None = None,
        initial_allocator_sample: AllocatorMemorySample | None = None,
        expert_slab_bytes: int = (
            _DEFAULT_EXPERT_RECORD_BYTES * _DEFAULT_EXPERT_SLAB_SLOTS
        ),
        expert_regrow_hysteresis_slabs: int = 1,
        expert_resize_min_interval_ns: int = 1_000_000_000,
    ) -> None:
        self._budget = budget or MemoryBudget()
        self._expert_slab_bytes = _exact_positive_int(
            "expert_slab_bytes", expert_slab_bytes
        )
        self._expert_regrow_hysteresis_slabs = _exact_nonnegative_int(
            "expert_regrow_hysteresis_slabs",
            expert_regrow_hysteresis_slabs,
        )
        self._expert_resize_min_interval_ns = _exact_nonnegative_int(
            "expert_resize_min_interval_ns",
            expert_resize_min_interval_ns,
        )
        self._lock = RLock()
        self._group_commit_lock = Lock()
        self._pools = BrokerSnapshot.synthetic(charged_bytes=0)
        self._revision = 0
        self._hard_failure_count = 0
        self._admission_failure_count = 0
        self._transaction_failure_count = 0
        self._failed_reason: str | None = None
        self._last_expert_resize_ns: int | None = None
        self._post_load_reconciled = False
        self._max_expert_slab_bytes = 0
        self._next_ticket_id = 1
        self._pending: _PendingKVTransaction | _PendingKVGroupTransaction | None = None
        self._pending_regrow: _PendingExpertRegrow | None = None
        self._consumed_ticket_ids: set[int] = set()
        self._consumed_regrow_ticket_ids: set[int] = set()
        self._allocations: dict[int, KVPhysicalAllocation] = {}
        self._released_allocation_ids: set[int] = set()
        self._ambiguous_allocation_ids: set[int] = set()
        self._unreconciled_kv_by_owner: dict[str, int] = {}
        if initial_allocator_sample is not None:
            self._validate_allocator_sample(
                "initial_allocator_sample",
                initial_allocator_sample,
            )
            if (
                initial_snapshot is not None
                and max(
                    initial_allocator_sample.cache_bytes,
                    initial_allocator_sample.charged_footprint_bytes
                    - (
                        initial_snapshot.resident_model_bytes
                        + initial_snapshot.kv_physical_bytes
                        + initial_snapshot.expert_slab_physical_bytes
                        + initial_snapshot.in_flight_expert_staging_bytes
                        + initial_snapshot.runtime_workspace_bytes
                    ),
                )
                != initial_snapshot.allocator_cache_bytes
            ):
                raise ValueError(
                    "initial allocator residual must match the initial snapshot"
                )
        self._initial_allocator_sample = initial_allocator_sample
        if initial_snapshot is not None:
            self.replace_snapshot(initial_snapshot)

    @classmethod
    def standard_hy3(
        cls,
        *,
        initial_snapshot: BrokerSnapshot | None = None,
        initial_allocator_sample: AllocatorMemorySample | None = None,
        expert_slab_bytes: int = (
            _DEFAULT_EXPERT_RECORD_BYTES * _DEFAULT_EXPERT_SLAB_SLOTS
        ),
        expert_regrow_hysteresis_slabs: int = 1,
        expert_resize_min_interval_ns: int = 1_000_000_000,
        allocator_headroom_bytes: int = 0,
    ) -> UnifiedMemoryBroker:
        return cls(
            budget=MemoryBudget(allocator_headroom_bytes=allocator_headroom_bytes),
            initial_snapshot=initial_snapshot,
            initial_allocator_sample=initial_allocator_sample,
            expert_slab_bytes=expert_slab_bytes,
            expert_regrow_hysteresis_slabs=(expert_regrow_hysteresis_slabs),
            expert_resize_min_interval_ns=expert_resize_min_interval_ns,
        )

    @property
    def budget(self) -> MemoryBudget:
        return self._budget

    @property
    def initial_allocator_sample(self) -> AllocatorMemorySample | None:
        return self._initial_allocator_sample

    def snapshot(self) -> BrokerSnapshot:
        with self._lock:
            self._assert_kv_ledger_invariant()
            pending_id = (
                None if self._pending is None else self._pending.ticket.ticket_id
            )
            pending_regrow_id = (
                None
                if self._pending_regrow is None
                else self._pending_regrow.ticket.ticket_id
            )
            return replace(
                self._pools,
                revision=self._revision,
                hard_failure_count=self._hard_failure_count,
                admission_failure_count=self._admission_failure_count,
                transaction_failure_count=self._transaction_failure_count,
                failed_closed=self._failed_reason is not None,
                failure_reason=self._failed_reason,
                pending_kv_ticket_id=pending_id,
                pending_expert_regrow_ticket_id=pending_regrow_id,
                last_expert_resize_ns=self._last_expert_resize_ns,
                owned_kv_physical_bytes=sum(
                    allocation.physical_bytes
                    for allocation in self._allocations.values()
                ),
                unreconciled_kv_physical_bytes=sum(
                    self._unreconciled_kv_by_owner.values()
                ),
            )

    def replace_snapshot(self, snapshot: BrokerSnapshot) -> None:
        """Install an observed classified snapshot and enforce both limits.

        An observation is retained even when it violates a limit: discarding an
        over-budget physical measurement would undercount reality.
        """

        with self._lock:
            self._validate_snapshot(snapshot)
            interrupted = self._pending is not None or self._pending_regrow is not None
            if interrupted:
                self._consume_pending()
                self._consume_pending_regrow()
                self._transaction_failure_count += 1
                self._failed_reason = (
                    "snapshot replacement interrupted an active KV transaction"
                )
            self._ambiguous_allocation_ids.update(self._allocations)
            self._allocations.clear()
            self._unreconciled_kv_by_owner.clear()
            if snapshot.kv_physical_bytes:
                self._unreconciled_kv_by_owner[f"snapshot:{self._revision + 1}"] = (
                    snapshot.kv_physical_bytes
                )
            self._copy_pool_fields(snapshot)
            self._max_expert_slab_bytes = max(
                self._max_expert_slab_bytes,
                snapshot.expert_slab_physical_bytes,
            )
            self._revision += 1
            self._assert_kv_ledger_invariant()
            self._record_hard_failure_if_needed()
            if interrupted:
                raise MemoryTransactionError(
                    "snapshot replacement interrupted an active KV transaction"
                )
            if snapshot.charged_bytes >= self._budget.hard_ceiling_bytes:
                self._admission_failure_count += 1
                raise MemoryAdmissionError(
                    "observed memory is at or above the hard ceiling"
                )
            if snapshot.classified_bytes > self._budget.classified_target_bytes:
                self._admission_failure_count += 1
                raise MemoryAdmissionError(
                    "observed classified memory is above the allocator-headroom target"
                )
            if snapshot.charged_bytes > self._budget.operating_target_bytes:
                self._admission_failure_count += 1
                raise MemoryAdmissionError(
                    "observed memory is above the operating target"
                )

    def reconcile_allocator_cache(
        self,
        sample: AllocatorMemorySample,
    ) -> BrokerSnapshot:
        """Refresh unclassified allocator truth without disturbing ownership.

        Physical Q4 growth can consume or retain MLX allocator-cache bytes even
        when its owned steady allocation is measured exactly.  Callers use
        this boundary between transactions so the next admission never plans
        from stale allocator truth.  Known resident, KV, expert, staging, and
        workspace pools remain authoritative; any charged footprint above
        those pools is conservatively retained in the allocator-cache pool.
        """

        self._validate_allocator_sample("sample", sample)
        with self._lock:
            if self._pending is not None or self._pending_regrow is not None:
                raise MemoryTransactionError(
                    "cannot reconcile allocator cache during an active memory "
                    "transaction"
                )
            classified_bytes = (
                self._pools.resident_model_bytes
                + self._pools.kv_physical_bytes
                + self._pools.expert_slab_physical_bytes
                + self._pools.in_flight_expert_staging_bytes
                + self._pools.runtime_workspace_bytes
            )
            unclassified_footprint = max(
                0,
                sample.charged_footprint_bytes - classified_bytes,
            )
            self._pools = replace(
                self._pools,
                allocator_cache_bytes=max(
                    self._pools.allocator_cache_bytes,
                    sample.cache_bytes,
                    unclassified_footprint,
                ),
            )
            self._revision += 1
            self._record_hard_failure_if_needed()
            charged = self._pools.charged_bytes
            if charged >= self._budget.hard_ceiling_bytes:
                self._admission_failure_count += 1
                raise MemoryAdmissionError(
                    "allocator cache reconciliation reached the hard ceiling"
                )
            if charged > self._budget.operating_target_bytes:
                self._admission_failure_count += 1
                self._transaction_failure_count += 1
                self._failed_reason = (
                    "allocator cache reconciliation exceeded the operating target"
                )
                raise MemoryAdmissionError(self._failed_reason)
            return self.snapshot()

    def reconcile_expert_protection(
        self,
        *,
        pinned_expert_bytes: int,
        speculative_expert_bytes: int,
    ) -> BrokerSnapshot:
        """Publish live expert protection classes before KV admission."""

        pinned = _exact_nonnegative_int("pinned_expert_bytes", pinned_expert_bytes)
        speculative = _exact_nonnegative_int(
            "speculative_expert_bytes",
            speculative_expert_bytes,
        )
        with self._lock:
            if self._pending is not None or self._pending_regrow is not None:
                raise MemoryTransactionError(
                    "cannot reconcile expert protection during an active memory "
                    "transaction"
                )
            experts = self._pools.expert_slab_physical_bytes
            if pinned > experts:
                raise MemoryTelemetryError(
                    "live pinned expert bytes exceed physical expert slabs"
                )
            if speculative > experts:
                raise MemoryTelemetryError(
                    "live speculative expert bytes exceed physical expert slabs"
                )
            self._pools = replace(
                self._pools,
                pinned_expert_bytes=pinned,
                speculative_expert_bytes=speculative,
            )
            self._revision += 1
            return self.snapshot()

    def reconcile_post_load_classification(
        self,
        *,
        resident_model_bytes: int,
        expert_slab_physical_bytes: int,
        in_flight_expert_staging_bytes: int,
        runtime_workspace_bytes: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
    ) -> BrokerSnapshot:
        """Atomically publish post-load truth without invalidating KV ownership."""

        resident = _exact_nonnegative_int(
            "resident_model_bytes",
            resident_model_bytes,
        )
        experts = _exact_nonnegative_int(
            "expert_slab_physical_bytes",
            expert_slab_physical_bytes,
        )
        staging = _exact_nonnegative_int(
            "in_flight_expert_staging_bytes",
            in_flight_expert_staging_bytes,
        )
        workspace = _exact_nonnegative_int(
            "runtime_workspace_bytes",
            runtime_workspace_bytes,
        )
        if allocator_before is None or allocator_after is None:
            raise MemoryTelemetryError(
                "allocator telemetry unavailable during post-load reconciliation"
            )
        try:
            self._validate_allocator_sample("allocator_before", allocator_before)
            self._validate_allocator_sample("allocator_after", allocator_after)
        except (TypeError, ValueError) as exc:
            raise MemoryTelemetryError(
                f"invalid allocator telemetry during post-load reconciliation: {exc}"
            ) from exc
        with self._lock:
            if self._post_load_reconciled:
                raise MemoryTransactionError("post-load memory was already reconciled")
            if self._pending is not None or self._pending_regrow is not None:
                raise MemoryTransactionError(
                    "cannot reconcile post-load memory during an active memory "
                    "transaction"
                )
            if (
                self._allocator_residual_bytes(allocator_before)
                != self._pools.allocator_cache_bytes
            ):
                raise MemoryTelemetryError(
                    "allocator telemetry is stale during post-load reconciliation"
                )
            if (
                self._initial_allocator_sample is not None
                and allocator_before != self._initial_allocator_sample
            ):
                raise MemoryTelemetryError(
                    "allocator baseline drifted before post-load reconciliation"
                )
            classified_after = (
                resident + experts + staging + workspace + self._pools.kv_physical_bytes
            )
            residual_cache = max(
                0,
                allocator_after.charged_footprint_bytes - classified_after,
            )
            cache_after = max(0, allocator_after.cache_bytes, residual_cache)
            self._pools = replace(
                self._pools,
                resident_model_bytes=resident,
                expert_slab_physical_bytes=experts,
                in_flight_expert_staging_bytes=staging,
                runtime_workspace_bytes=workspace,
                allocator_cache_bytes=cache_after,
                pinned_expert_bytes=min(self._pools.pinned_expert_bytes, experts),
                speculative_expert_bytes=min(
                    self._pools.speculative_expert_bytes,
                    experts,
                ),
            )
            self._max_expert_slab_bytes = max(
                self._max_expert_slab_bytes,
                experts,
            )
            self._revision += 1
            self._post_load_reconciled = True
            self._assert_kv_ledger_invariant()
            self._record_hard_failure_if_needed()
            charged = self._pools.charged_bytes
            if charged >= self._budget.hard_ceiling_bytes:
                self._admission_failure_count += 1
                raise MemoryAdmissionError(
                    "post-load memory reconciliation reached the hard ceiling"
                )
            if self._pools.classified_bytes > self._budget.classified_target_bytes:
                self._admission_failure_count += 1
                self._transaction_failure_count += 1
                self._failed_reason = (
                    "post-load classified memory exceeded the allocator-headroom target"
                )
                raise MemoryAdmissionError(self._failed_reason)
            if charged > self._budget.operating_target_bytes:
                self._admission_failure_count += 1
                self._transaction_failure_count += 1
                self._failed_reason = (
                    "post-load memory reconciliation exceeded the operating target"
                )
                raise MemoryAdmissionError(self._failed_reason)
            return self.snapshot()

    def plan_kv_growth(
        self,
        *,
        steady_delta_bytes: int,
        transient_delta_bytes: int,
        cache_id: str | None = None,
    ) -> KVAllocationTicket:
        """Reserve one KV growth, including its additional transition peak."""

        steady = _exact_positive_int("steady_delta_bytes", steady_delta_bytes)
        transient = _exact_nonnegative_int(
            "transient_delta_bytes", transient_delta_bytes
        )
        if cache_id is not None:
            cache_id = self._validate_cache_id(cache_id)
        with self._lock:
            try:
                self._ensure_allocation_open()
            except MemoryAdmissionError as exc:
                owner = cache_id or "anonymous KV owner"
                raise MemoryAdmissionError(
                    f"{exc} while planning KV growth for {owner}"
                ) from exc
            if self._pending is not None or self._pending_regrow is not None:
                self._reject_admission(
                    "another memory allocation transaction is already active"
                )

            charged = self._pools.charged_bytes
            classified = self._pools.classified_bytes
            target = self._budget.operating_target_bytes
            required_reclaim = max(
                0,
                classified + steady - self._budget.classified_target_bytes,
                charged + steady + transient - target,
            )
            reclaimable = (
                self._pools.expert_slab_physical_bytes - self._pools.pinned_expert_bytes
            )
            if required_reclaim > reclaimable:
                self._reject_admission(
                    "pinned expert bytes leave insufficient physical reclaim "
                    f"({reclaimable} available, {required_reclaim} required)"
                )

            planned_steady = charged - required_reclaim + steady
            planned_peak = planned_steady + transient
            planned_classified = classified - required_reclaim + steady
            if planned_classified > self._budget.classified_target_bytes:
                self._reject_admission(
                    "KV steady allocation exceeds the allocator-headroom target"
                )
            if planned_peak > target:
                self._reject_admission(
                    "KV allocation peak exceeds the operating target"
                )
            if planned_peak >= self._budget.hard_ceiling_bytes:
                self._reject_admission("KV allocation peak reaches the hard ceiling")

            ticket_id = self._next_ticket_id
            self._next_ticket_id += 1
            owner_id = cache_id or f"kv-ticket:{ticket_id}"
            self._revision += 1
            ticket = KVAllocationTicket(
                ticket_id=ticket_id,
                cache_id=owner_id,
                steady_delta_bytes=steady,
                transient_delta_bytes=transient,
                required_expert_reclaim_bytes=required_reclaim,
                snapshot_revision=self._revision,
                planned_steady_bytes=planned_steady,
                planned_peak_bytes=planned_peak,
            )
            self._pending = _PendingKVTransaction(
                ticket=ticket,
                expected_revision=self._revision,
                reclaim_confirmed=required_reclaim == 0,
            )
            return ticket

    def plan_kv_growth_group(
        self,
        *,
        members: Sequence[tuple[str, int, int]],
    ) -> KVAllocationGroupTicket:
        """Reserve aggregate steady growth with serialized member transients."""

        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            raise TypeError("members must be a sequence of three-item tuples")
        if not members:
            raise ValueError("members must contain at least one KV owner")

        parsed: list[tuple[str, int, int]] = []
        seen_cache_ids: set[str] = set()
        for index, member in enumerate(members):
            if not isinstance(member, tuple) or len(member) != 3:
                raise TypeError(f"members[{index}] must be a three-item tuple")
            cache_id = self._validate_cache_id(member[0])
            steady = _exact_positive_int(
                f"members[{index}].steady_delta_bytes",
                member[1],
            )
            transient = _exact_nonnegative_int(
                f"members[{index}].transient_delta_bytes",
                member[2],
            )
            if cache_id in seen_cache_ids:
                raise ValueError(f"duplicate KV group cache_id: {cache_id}")
            seen_cache_ids.add(cache_id)
            parsed.append((cache_id, steady, transient))

        total_steady = sum(member[1] for member in parsed)
        serialized_transient = max(member[2] for member in parsed)
        with self._lock:
            self._ensure_allocation_open()
            if self._pending is not None or self._pending_regrow is not None:
                self._reject_admission(
                    "another memory allocation transaction is already active"
                )

            charged = self._pools.charged_bytes
            classified = self._pools.classified_bytes
            required_reclaim = max(
                0,
                classified + total_steady - self._budget.classified_target_bytes,
                charged
                + total_steady
                + serialized_transient
                - self._budget.operating_target_bytes,
            )
            reclaimable = (
                self._pools.expert_slab_physical_bytes - self._pools.pinned_expert_bytes
            )
            if required_reclaim > reclaimable:
                self._reject_admission(
                    "pinned expert bytes leave insufficient physical reclaim "
                    f"({reclaimable} available, {required_reclaim} required)"
                )

            planned_steady = charged - required_reclaim + total_steady
            planned_peak = planned_steady + serialized_transient
            planned_classified = classified - required_reclaim + total_steady
            if planned_classified > self._budget.classified_target_bytes:
                self._reject_admission(
                    "KV group steady allocation exceeds the allocator-headroom target"
                )
            if planned_peak > self._budget.operating_target_bytes:
                self._reject_admission(
                    "KV group allocation peak exceeds the operating target"
                )
            if planned_peak >= self._budget.hard_ceiling_bytes:
                self._reject_admission(
                    "KV group allocation peak reaches the hard ceiling"
                )

            ticket_id = self._next_ticket_id
            self._next_ticket_id += 1
            planned_members: list[KVAllocationGroupMember] = []
            for cache_id, steady, transient in parsed:
                allocation_id = self._next_ticket_id
                self._next_ticket_id += 1
                planned_members.append(
                    KVAllocationGroupMember(
                        allocation_id=allocation_id,
                        cache_id=cache_id,
                        steady_delta_bytes=steady,
                        transient_delta_bytes=transient,
                    )
                )

            self._revision += 1
            ticket = KVAllocationGroupTicket(
                ticket_id=ticket_id,
                members=tuple(planned_members),
                steady_delta_bytes=total_steady,
                transient_delta_bytes=serialized_transient,
                required_expert_reclaim_bytes=required_reclaim,
                snapshot_revision=self._revision,
                planned_steady_bytes=planned_steady,
                planned_peak_bytes=planned_peak,
            )
            self._pending = _PendingKVGroupTransaction(
                ticket=ticket,
                expected_revision=self._revision,
                reclaim_confirmed=required_reclaim == 0,
            )
            return ticket

    def confirm_expert_reclaim(
        self,
        ticket: KVAllocationTicket | KVAllocationGroupTicket,
        *,
        registered_slab_bytes_after: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
        now_ns: int | None = None,
    ) -> BrokerSnapshot:
        """Credit reclaim only after registry and allocator footprint agree."""

        registered_after = _exact_nonnegative_int(
            "registered_slab_bytes_after", registered_slab_bytes_after
        )
        if now_ns is not None:
            now_ns = _exact_nonnegative_int("now_ns", now_ns)
        with self._lock:
            if isinstance(ticket, KVAllocationGroupTicket):
                transaction = self._require_group_ticket(ticket)
            else:
                transaction = self._require_ticket(ticket)
            required = ticket.required_expert_reclaim_bytes
            if required == 0:
                raise MemoryTransactionError(
                    "ticket does not require expert reclaim confirmation"
                )
            if transaction.reclaim_confirmed:
                raise MemoryTransactionError(
                    "expert reclaim was already confirmed for this ticket"
                )
            if (
                now_ns is not None
                and self._last_expert_resize_ns is not None
                and now_ns < self._last_expert_resize_ns
            ):
                reason = (
                    f"expert reclaim timestamp {now_ns} is before last resize "
                    f"{self._last_expert_resize_ns}"
                )
                self._fail_pending(reason)
                raise MemoryTransactionError(reason)
            if allocator_before is None or allocator_after is None:
                self._fail_pending(
                    "allocator telemetry unavailable during expert reclaim"
                )
                raise MemoryTelemetryError(
                    "allocator telemetry unavailable during expert reclaim"
                )
            try:
                self._validate_allocator_sample("allocator_before", allocator_before)
                self._validate_allocator_sample("allocator_after", allocator_after)
            except (TypeError, ValueError) as exc:
                self._fail_pending(f"invalid allocator telemetry: {exc}")
                raise MemoryTelemetryError(
                    f"invalid allocator telemetry: {exc}"
                ) from exc
            if (
                self._allocator_residual_bytes(allocator_before)
                > self._pools.allocator_cache_bytes
            ):
                self._fail_pending(
                    "allocator telemetry is stale relative to broker cache "
                    "during expert reclaim"
                )
                raise MemoryTelemetryError(
                    "allocator telemetry is stale relative to broker cache "
                    "during expert reclaim"
                )

            registered_before = self._pools.expert_slab_physical_bytes
            registered_drop = registered_before - registered_after
            allocator_drop = (
                allocator_before.charged_footprint_bytes
                - allocator_after.charged_footprint_bytes
            )
            failure: str | None = None
            pinned_after = min(
                self._pools.pinned_expert_bytes,
                registered_after,
            )
            speculative_after = min(
                self._pools.speculative_expert_bytes,
                registered_after,
            )
            if registered_drop < 0:
                failure = "registered slab bytes increased during expert reclaim"
            elif registered_drop > 0 and now_ns is None:
                failure = "now_ns is required when physical expert bytes decrease"
            elif registered_after < self._pools.pinned_expert_bytes:
                failure = "physical reclaim destroyed registered pinned expert bytes"
            elif registered_drop < required:
                failure = (
                    "registered slab reduction is smaller than requested "
                    f"({registered_drop} observed, {required} required)"
                )
            elif allocator_drop < required:
                failure = (
                    "allocator footprint did not fall with registered slabs "
                    f"({allocator_drop} observed, {required} required)"
                )

            # With complete telemetry, retain the observed physical
            # classification even when it proves zero reclaim.  Any footprint
            # not proven released remains conservatively charged as allocator
            # retention.  This prevents stale active references from creating
            # apparent headroom merely because the slab registry changed.
            conservative_cache = self._conservative_cache_after_release(
                classified_bytes_before=registered_before,
                classified_bytes_after=registered_after,
                cache_bytes_before=self._pools.allocator_cache_bytes,
                observed_cache_bytes_after=allocator_after.cache_bytes,
                allocator_footprint_drop=allocator_drop,
            )
            self._pools = replace(
                self._pools,
                expert_slab_physical_bytes=registered_after,
                allocator_cache_bytes=conservative_cache,
                pinned_expert_bytes=pinned_after,
                speculative_expert_bytes=speculative_after,
            )
            confirmed_peak = (
                self._pools.charged_bytes
                + ticket.steady_delta_bytes
                + ticket.transient_delta_bytes
            )
            confirmed_classified = (
                self._pools.classified_bytes + ticket.steady_delta_bytes
            )
            if (
                failure is None
                and confirmed_classified > self._budget.classified_target_bytes
            ):
                failure = (
                    "confirmed KV steady allocation exceeded the "
                    "allocator-headroom target"
                )
            elif (
                failure is None and confirmed_peak > self._budget.operating_target_bytes
            ):
                failure = (
                    "allocator cache retention leaves the confirmed allocation "
                    f"peak above the operating target ({confirmed_peak} bytes)"
                )
            if failure is not None:
                self._fail_pending(failure, pools_already_updated=True)
                raise MemoryTelemetryError(failure)

            if registered_drop > 0:
                assert now_ns is not None
                self._last_expert_resize_ns = now_ns
            self._revision += 1
            transaction.reclaim_confirmed = True
            transaction.expected_revision = self._revision
            if isinstance(transaction, _PendingKVGroupTransaction):
                transaction.last_allocator_sample = allocator_after
            self._record_hard_failure_if_needed()
            return self.snapshot()

    def terminalize_expert_resize(
        self,
        ticket: KVAllocationTicket | KVAllocationGroupTicket | ExpertRegrowTicket,
        *,
        registered_slab_bytes_after: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
        reason: str,
    ) -> BrokerSnapshot:
        """Fail one resize closed while preserving unrelated KV ownership.

        Runtime slab destruction and allocation can cross their physical
        boundary before registry or allocator telemetry fails.  This method
        consumes only that active resize ticket and publishes conservative
        expert/cache truth without replacing the snapshot or invalidating
        already-authenticated KV allocation handles.
        """

        registered_after = _exact_nonnegative_int(
            "registered_slab_bytes_after",
            registered_slab_bytes_after,
        )
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        failure = reason.strip()
        if not failure:
            raise ValueError("reason must not be empty")
        with self._lock:
            if isinstance(ticket, KVAllocationTicket):
                self._require_ticket(ticket)
                consume = self._consume_pending
            elif isinstance(ticket, KVAllocationGroupTicket):
                self._require_group_ticket(ticket)
                consume = self._consume_pending
            elif isinstance(ticket, ExpertRegrowTicket):
                self._require_regrow_ticket(ticket)
                consume = self._consume_pending_regrow
            else:
                raise TypeError(
                    "ticket must be a KVAllocationTicket, KVAllocationGroupTicket, "
                    "or ExpertRegrowTicket"
                )

            registered_before = self._pools.expert_slab_physical_bytes
            samples_valid = allocator_before is not None and allocator_after is not None
            if samples_valid:
                assert allocator_before is not None
                assert allocator_after is not None
                try:
                    self._validate_allocator_sample(
                        "allocator_before",
                        allocator_before,
                    )
                    self._validate_allocator_sample(
                        "allocator_after",
                        allocator_after,
                    )
                except (TypeError, ValueError):
                    samples_valid = False
                else:
                    samples_valid = (
                        self._allocator_residual_bytes(allocator_before)
                        == self._pools.allocator_cache_bytes
                    )

            if samples_valid:
                assert allocator_before is not None
                assert allocator_after is not None
                classified_delta = registered_after - registered_before
                footprint_delta = (
                    allocator_after.charged_footprint_bytes
                    - allocator_before.charged_footprint_bytes
                )
                residual = (
                    self._pools.allocator_cache_bytes
                    + footprint_delta
                    - classified_delta
                )
                conservative_cache = max(
                    0,
                    allocator_after.cache_bytes,
                    residual,
                )
            else:
                conservative_cache = self._pools.allocator_cache_bytes + max(
                    0, registered_before - registered_after
                )

            self._pools = replace(
                self._pools,
                expert_slab_physical_bytes=registered_after,
                allocator_cache_bytes=conservative_cache,
                pinned_expert_bytes=min(
                    self._pools.pinned_expert_bytes,
                    registered_after,
                ),
                speculative_expert_bytes=min(
                    self._pools.speculative_expert_bytes,
                    registered_after,
                ),
            )
            consume()
            self._failed_reason = failure
            self._transaction_failure_count += 1
            self._revision += 1
            self._assert_kv_ledger_invariant()
            self._record_hard_failure_if_needed()
            return self.snapshot()

    def commit_kv_growth(
        self,
        ticket: KVAllocationTicket,
        *,
        allocated_physical_bytes: int,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> KVPhysicalAllocation:
        """Commit an exactly measured steady KV allocation atomically.

        When allocator samples are supplied, cache consumption is reclassified
        in the same broker transaction as the new owned KV bytes.  Supplying
        only one sample is ambiguous after the physical allocation boundary and
        therefore terminalizes the bytes as unowned and fails closed.
        """

        allocated = _exact_nonnegative_int(
            "allocated_physical_bytes", allocated_physical_bytes
        )
        with self._lock:
            transaction = self._require_ticket(ticket)
            if not transaction.reclaim_confirmed:
                raise MemoryTransactionError(
                    "required expert reclaim has not been confirmed"
                )
            cache_after = self._pools.allocator_cache_bytes
            samples_requested = (
                allocator_before is not None or allocator_after is not None
            )
            if samples_requested:
                try:
                    cache_after = self._allocator_cache_after_growth(
                        allocator_before=allocator_before,
                        allocator_after=allocator_after,
                        classified_delta_bytes=allocated,
                        context="KV commit",
                    )
                except MemoryTelemetryError as exc:
                    self._terminalize_unowned_kv_growth(
                        ticket,
                        allocated,
                        reason=str(exc),
                    )
                    setattr(exc, "transaction_terminalized", True)
                    raise
            if allocated != ticket.steady_delta_bytes:
                self._pools = replace(
                    self._pools,
                    kv_physical_bytes=(self._pools.kv_physical_bytes + allocated),
                    allocator_cache_bytes=cache_after,
                )
                if allocated:
                    self._unreconciled_kv_by_owner[
                        f"failed-growth:{ticket.ticket_id}"
                    ] = allocated
                reason = (
                    f"physical KV allocation {allocated} did not match planned "
                    f"{ticket.steady_delta_bytes} bytes"
                )
                self._fail_pending(reason, pools_already_updated=True)
                error = MemoryTransactionError(reason)
                setattr(error, "transaction_terminalized", True)
                raise error

            self._pools = replace(
                self._pools,
                kv_physical_bytes=(self._pools.kv_physical_bytes + allocated),
                allocator_cache_bytes=cache_after,
            )
            if self._pools.classified_bytes > self._budget.classified_target_bytes:
                if allocated:
                    self._unreconciled_kv_by_owner[
                        f"failed-growth:{ticket.ticket_id}"
                    ] = allocated
                reason = (
                    "committed KV allocation exceeded the allocator-headroom target"
                )
                self._fail_pending(reason, pools_already_updated=True)
                self._assert_kv_ledger_invariant()
                error = MemoryTransactionError(reason)
                setattr(error, "transaction_terminalized", True)
                raise error
            if self._pools.charged_bytes > self._budget.operating_target_bytes:
                if allocated:
                    self._unreconciled_kv_by_owner[
                        f"failed-growth:{ticket.ticket_id}"
                    ] = allocated
                reason = "committed KV allocation exceeded the operating target"
                self._fail_pending(reason, pools_already_updated=True)
                self._assert_kv_ledger_invariant()
                error = MemoryTransactionError(reason)
                setattr(error, "transaction_terminalized", True)
                raise error
            allocation = KVPhysicalAllocation(
                allocation_id=ticket.ticket_id,
                cache_id=ticket.cache_id,
                physical_bytes=allocated,
                committed_revision=self._revision + 1,
            )
            self._allocations[allocation.allocation_id] = allocation
            self._consume_pending()
            self._revision += 1
            self._record_hard_failure_if_needed()
            self._assert_kv_ledger_invariant()
            return allocation

    def commit_kv_growth_group_member(
        self,
        ticket: KVAllocationGroupTicket,
        *,
        cache_id: str,
        allocated_physical_bytes: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
    ) -> KVPhysicalAllocation:
        """Commit one member while rejecting overlapping commit windows."""

        owner_id = self._validate_cache_id(cache_id)
        allocated = _exact_nonnegative_int(
            "allocated_physical_bytes",
            allocated_physical_bytes,
        )
        if not self._group_commit_lock.acquire(blocking=False):
            reason = "concurrent KV group member commits are not permitted"
            with self._lock:
                try:
                    self._require_group_ticket(ticket)
                except MemoryTransactionError:
                    pass
                else:
                    self._terminalize_unowned_group_growth(
                        ticket,
                        allocated,
                        reason=reason,
                        owner=owner_id,
                    )
            error = MemoryTransactionError(reason)
            setattr(error, "transaction_terminalized", True)
            raise error
        try:
            return self._commit_kv_growth_group_member_serialized(
                ticket,
                cache_id=owner_id,
                allocated_physical_bytes=allocated,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        finally:
            self._group_commit_lock.release()

    def _commit_kv_growth_group_member_serialized(
        self,
        ticket: KVAllocationGroupTicket,
        *,
        cache_id: str,
        allocated_physical_bytes: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
    ) -> KVPhysicalAllocation:
        """Commit one declared group member against a serialized telemetry chain."""

        owner_id = self._validate_cache_id(cache_id)
        allocated = _exact_nonnegative_int(
            "allocated_physical_bytes",
            allocated_physical_bytes,
        )
        with self._lock:
            transaction = self._require_group_ticket(ticket)
            if not transaction.reclaim_confirmed:
                raise MemoryTransactionError(
                    "required expert reclaim has not been confirmed"
                )
            member = next(
                (item for item in ticket.members if item.cache_id == owner_id),
                None,
            )
            if member is None:
                reason = f"unknown KV group member {owner_id}"
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=reason,
                    owner=owner_id,
                )
                raise MemoryTransactionError(reason)
            if owner_id in transaction.committed_cache_ids:
                reason = f"KV group member {owner_id} was already committed"
                self._fail_pending(reason)
                raise MemoryTransactionError(reason)
            if allocated > member.steady_delta_bytes:
                reason = (
                    f"physical KV allocation {allocated} exceeds group member plan "
                    f"of {member.steady_delta_bytes} bytes for {owner_id}"
                )
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=reason,
                    owner=owner_id,
                )
                raise MemoryTransactionError(reason)
            if allocator_before is None or allocator_after is None:
                reason = (
                    f"allocator telemetry unavailable during KV group commit for "
                    f"{owner_id}"
                )
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=reason,
                    owner=owner_id,
                )
                error = MemoryTelemetryError(reason)
                setattr(error, "transaction_terminalized", True)
                raise error
            try:
                self._validate_allocator_sample("allocator_before", allocator_before)
                self._validate_allocator_sample("allocator_after", allocator_after)
            except (TypeError, ValueError) as exc:
                reason = f"invalid allocator telemetry during KV group commit: {exc}"
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=reason,
                    owner=owner_id,
                )
                raise MemoryTelemetryError(reason) from exc
            if (
                transaction.last_allocator_sample is not None
                and not self._allocator_group_chain_has_no_unseen_growth(
                    transaction.last_allocator_sample,
                    allocator_before,
                )
            ):
                reason = (
                    "KV group member allocator telemetry is not serialized with "
                    "the preceding reclaim or member commit"
                )
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=reason,
                    owner=owner_id,
                )
                error = MemoryTelemetryError(reason)
                setattr(error, "transaction_terminalized", True)
                raise error
            if (
                transaction.last_allocator_sample is None
                and self._allocator_residual_bytes(allocator_before)
                > self._pools.allocator_cache_bytes
            ):
                reason = (
                    "allocator telemetry is stale relative to broker cache during "
                    "the first KV group member commit"
                )
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=reason,
                    owner=owner_id,
                )
                raise MemoryTelemetryError(reason)
            try:
                cache_after = max(
                    self._pools.allocator_cache_bytes,
                    self._allocator_cache_after_growth(
                        allocator_before=allocator_before,
                        allocator_after=allocator_after,
                        classified_delta_bytes=allocated,
                        context=f"KV group commit for {owner_id}",
                    ),
                )
            except MemoryTelemetryError as exc:
                self._terminalize_unowned_group_growth(
                    ticket,
                    allocated,
                    reason=str(exc),
                    owner=owner_id,
                )
                setattr(exc, "transaction_terminalized", True)
                raise

            self._pools = replace(
                self._pools,
                kv_physical_bytes=self._pools.kv_physical_bytes + allocated,
                allocator_cache_bytes=cache_after,
            )
            allocation = KVPhysicalAllocation(
                allocation_id=member.allocation_id,
                cache_id=member.cache_id,
                physical_bytes=allocated,
                committed_revision=self._revision + 1,
            )
            self._allocations[allocation.allocation_id] = allocation
            transaction.committed_cache_ids.add(owner_id)
            transaction.last_allocator_sample = allocator_after
            self._revision += 1

            failure: str | None = None
            if self._pools.classified_bytes > self._budget.classified_target_bytes:
                failure = (
                    "committed KV group allocation exceeded the "
                    "allocator-headroom target"
                )
            elif self._pools.charged_bytes > self._budget.operating_target_bytes:
                failure = "committed KV group allocation exceeded the operating target"
            if failure is not None:
                del self._allocations[allocation.allocation_id]
                self._unreconciled_kv_by_owner[
                    f"failed-group-growth:{ticket.ticket_id}:{owner_id}"
                ] = allocated
                self._fail_pending(failure, pools_already_updated=True)
                self._assert_kv_ledger_invariant()
                error = MemoryTransactionError(failure)
                setattr(error, "transaction_terminalized", True)
                raise error

            if len(transaction.committed_cache_ids) == len(ticket.members):
                self._consume_pending()
            else:
                transaction.expected_revision = self._revision
            self._record_hard_failure_if_needed()
            self._assert_kv_ledger_invariant()
            return allocation

    def abort_kv_growth(
        self,
        ticket: KVAllocationTicket,
        *,
        observed_kv_delta_bytes: int | None,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> BrokerSnapshot:
        """Abort a ticket without restoring already destroyed expert slabs.

        ``None`` means an interruption left the physical KV delta unknown; that
        poisons further allocation.  A measured zero is a safe allocation
        failure and retains any already-confirmed expert reduction.
        """

        if observed_kv_delta_bytes is not None:
            observed_kv_delta_bytes = _exact_nonnegative_int(
                "observed_kv_delta_bytes", observed_kv_delta_bytes
            )
        with self._lock:
            self._require_ticket(ticket)
            if observed_kv_delta_bytes is None:
                reason = "interrupted KV allocation has an unknown physical delta"
                self._fail_pending(reason)
                raise MemoryTransactionError(reason)
            cache_after = self._pools.allocator_cache_bytes
            samples_requested = (
                allocator_before is not None or allocator_after is not None
            )
            if samples_requested:
                try:
                    cache_after = self._allocator_cache_after_growth(
                        allocator_before=allocator_before,
                        allocator_after=allocator_after,
                        classified_delta_bytes=observed_kv_delta_bytes,
                        context="KV abort",
                    )
                except MemoryTelemetryError as exc:
                    self._terminalize_unowned_kv_growth(
                        ticket,
                        observed_kv_delta_bytes,
                        reason=str(exc),
                    )
                    raise
            self._pools = replace(
                self._pools,
                allocator_cache_bytes=cache_after,
            )
            if observed_kv_delta_bytes:
                self._pools = replace(
                    self._pools,
                    kv_physical_bytes=(
                        self._pools.kv_physical_bytes + observed_kv_delta_bytes
                    ),
                )
                self._failed_reason = (
                    "aborted KV allocation retained observed physical bytes"
                )
                self._unreconciled_kv_by_owner[f"aborted-growth:{ticket.ticket_id}"] = (
                    observed_kv_delta_bytes
                )
            self._transaction_failure_count += 1
            self._consume_pending()
            self._revision += 1
            self._record_hard_failure_if_needed()
            return self.snapshot()

    def abort_kv_growth_group(
        self,
        ticket: KVAllocationGroupTicket,
        *,
        observed_uncommitted_kv_delta_bytes: int | None,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> BrokerSnapshot:
        """Consume a group permit without discarding committed member handles."""

        observed = observed_uncommitted_kv_delta_bytes
        if observed is not None:
            observed = _exact_nonnegative_int(
                "observed_uncommitted_kv_delta_bytes",
                observed,
            )
        with self._lock:
            transaction = self._require_group_ticket(ticket)
            if observed is None:
                reason = "interrupted KV group allocation has an unknown physical delta"
                self._fail_pending(reason)
                raise MemoryTransactionError(reason)

            samples_requested = (
                allocator_before is not None or allocator_after is not None
            )
            cache_after = self._pools.allocator_cache_bytes
            telemetry_failure: MemoryTelemetryError | None = None
            if samples_requested:
                if (
                    transaction.last_allocator_sample is not None
                    and not self._allocator_group_chain_has_no_unseen_growth(
                        transaction.last_allocator_sample,
                        allocator_before,
                    )
                ):
                    telemetry_failure = MemoryTelemetryError(
                        "KV group abort allocator telemetry is not serialized with "
                        "the preceding reclaim or member commit"
                    )
                else:
                    try:
                        cache_after = max(
                            self._pools.allocator_cache_bytes,
                            self._allocator_cache_after_growth(
                                allocator_before=allocator_before,
                                allocator_after=allocator_after,
                                classified_delta_bytes=observed,
                                context="KV group abort",
                            ),
                        )
                    except MemoryTelemetryError as exc:
                        telemetry_failure = exc
                    else:
                        if (
                            transaction.last_allocator_sample is None
                            and allocator_before is not None
                            and self._allocator_residual_bytes(allocator_before)
                            > self._pools.allocator_cache_bytes
                        ):
                            telemetry_failure = MemoryTelemetryError(
                                "allocator telemetry is stale relative to broker "
                                "cache during KV group abort"
                            )
                            cache_after = self._pools.allocator_cache_bytes

            self._pools = replace(
                self._pools,
                allocator_cache_bytes=cache_after,
                kv_physical_bytes=self._pools.kv_physical_bytes + observed,
            )
            if observed:
                self._unreconciled_kv_by_owner[
                    f"aborted-group-growth:{ticket.ticket_id}"
                ] = observed
                self._failed_reason = (
                    "aborted KV group allocation retained observed physical bytes"
                )
            if telemetry_failure is not None:
                self._failed_reason = str(telemetry_failure)
            self._transaction_failure_count += 1
            self._consume_pending()
            self._revision += 1
            self._record_hard_failure_if_needed()
            self._assert_kv_ledger_invariant()
            if telemetry_failure is not None:
                raise telemetry_failure
            return self.snapshot()

    def _allocator_cache_after_growth(
        self,
        *,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
        classified_delta_bytes: int,
        context: str,
    ) -> int:
        """Conservatively reclassify one allocator transition under the lock."""

        if allocator_before is None or allocator_after is None:
            raise MemoryTelemetryError(
                f"allocator telemetry unavailable during {context}"
            )
        try:
            self._validate_allocator_sample("allocator_before", allocator_before)
            self._validate_allocator_sample("allocator_after", allocator_after)
        except (TypeError, ValueError) as exc:
            raise MemoryTelemetryError(
                f"invalid allocator telemetry during {context}: {exc}"
            ) from exc
        classified_after = (
            self._pools.resident_model_bytes
            + self._pools.kv_physical_bytes
            + classified_delta_bytes
            + self._pools.expert_slab_physical_bytes
            + self._pools.in_flight_expert_staging_bytes
            + self._pools.runtime_workspace_bytes
        )
        residual = max(
            0,
            allocator_after.charged_footprint_bytes - classified_after,
        )
        return max(0, allocator_after.cache_bytes, residual)

    @staticmethod
    def _allocator_group_chain_has_no_unseen_growth(
        previous: AllocatorMemorySample,
        current: AllocatorMemorySample | None,
    ) -> bool:
        """Reject unseen growth while allowing conservatively charged retirement."""

        return bool(
            isinstance(current, AllocatorMemorySample)
            and current.charged_footprint_bytes <= previous.charged_footprint_bytes
            and current.peak_bytes == previous.peak_bytes
        )

    def _terminalize_unowned_kv_growth(
        self,
        ticket: KVAllocationTicket,
        physical_bytes: int,
        *,
        reason: str,
    ) -> None:
        """Consume a crossed allocation boundary without inventing ownership."""

        if physical_bytes:
            self._pools = replace(
                self._pools,
                kv_physical_bytes=self._pools.kv_physical_bytes + physical_bytes,
            )
            self._unreconciled_kv_by_owner[f"ambiguous-growth:{ticket.ticket_id}"] = (
                physical_bytes
            )
        self._fail_pending(reason, pools_already_updated=True)
        self._assert_kv_ledger_invariant()

    def _terminalize_unowned_group_growth(
        self,
        ticket: KVAllocationGroupTicket,
        physical_bytes: int,
        *,
        reason: str,
        owner: str,
    ) -> None:
        """Consume a group permit while preserving prior owned member handles."""

        if physical_bytes:
            self._pools = replace(
                self._pools,
                kv_physical_bytes=self._pools.kv_physical_bytes + physical_bytes,
            )
            self._unreconciled_kv_by_owner[
                f"ambiguous-group-growth:{ticket.ticket_id}:{owner}"
            ] = physical_bytes
        self._fail_pending(reason, pools_already_updated=True)
        self._assert_kv_ledger_invariant()

    def release_kv(
        self,
        allocation: KVPhysicalAllocation,
        *,
        registered_kv_bytes_after: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
    ) -> BrokerSnapshot:
        """Backward-compatible exact release for a single-owner allocation."""

        if not isinstance(allocation, KVPhysicalAllocation):
            raise TypeError("allocation must be a KVPhysicalAllocation")
        return self.release_kv_batch(
            cache_id=allocation.cache_id,
            allocations=(allocation,),
            registered_kv_bytes_after=registered_kv_bytes_after,
            allocator_before=allocator_before,
            allocator_after=allocator_after,
        )

    def release_kv_batch(
        self,
        *,
        cache_id: str,
        allocations: Sequence[KVPhysicalAllocation],
        registered_kv_bytes_after: int | None = None,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> BrokerSnapshot:
        """Reconcile one cache's complete ownership set with one physical close.

        All selected handles must be live, unique, and owned by ``cache_id``.
        The observed global KV reduction must exactly equal their aggregate.
        Active-to-cache movement is reclassified and receives zero headroom.
        """

        owner_id = self._validate_cache_id(cache_id)
        selected = tuple(allocations)
        if not selected:
            raise ValueError("allocations must contain at least one handle")
        with self._lock:
            if any(
                not isinstance(allocation, KVPhysicalAllocation)
                for allocation in selected
            ):
                raise TypeError(
                    "allocations must contain only KVPhysicalAllocation handles"
                )
            selected_ids = [item.allocation_id for item in selected]
            if len(set(selected_ids)) != len(selected_ids):
                self._transaction_failure_count += 1
                self._revision += 1
                raise MemoryTransactionError(
                    "KV batch contains duplicate allocation handles"
                )
            replayed = [
                allocation_id
                for allocation_id in selected_ids
                if allocation_id in self._released_allocation_ids
                or allocation_id in self._ambiguous_allocation_ids
            ]
            if replayed:
                self._transaction_failure_count += 1
                self._revision += 1
                raise DuplicateReleaseError(
                    f"KV allocations were already released: {replayed}"
                )
            for allocation in selected:
                registered = self._allocations.get(allocation.allocation_id)
                if registered is None or registered is not allocation:
                    self._transaction_failure_count += 1
                    self._revision += 1
                    raise MemoryTransactionError(
                        f"unknown KV allocation {allocation.allocation_id}"
                    )
            if self._pending is not None or self._pending_regrow is not None:
                raise MemoryTransactionError(
                    "cannot release KV during an active memory transaction"
                )
            if allocator_before is None or allocator_after is None:
                reason = "allocator telemetry unavailable during KV release"
                self._fail_without_pending(reason)
                raise MemoryTelemetryError(reason)
            try:
                self._validate_allocator_sample("allocator_before", allocator_before)
                self._validate_allocator_sample("allocator_after", allocator_after)
            except (TypeError, ValueError) as exc:
                reason = f"invalid allocator telemetry during KV release: {exc}"
                self._fail_without_pending(reason)
                raise MemoryTelemetryError(reason) from exc
            observed_residual_before = self._allocator_residual_bytes(allocator_before)
            cache_bytes_before = max(
                self._pools.allocator_cache_bytes,
                observed_residual_before,
            )
            observed_charged_before = self._pools.classified_bytes + cache_bytes_before
            pre_release_limit_failure: str | None = None
            if observed_charged_before >= self._budget.hard_ceiling_bytes:
                self._hard_failure_count += 1
                self._admission_failure_count += 1
                pre_release_limit_failure = (
                    "allocator telemetry reached the hard ceiling during KV "
                    f"release for {owner_id}"
                )
            elif observed_charged_before > self._budget.operating_target_bytes:
                self._admission_failure_count += 1
                pre_release_limit_failure = (
                    "allocator telemetry exceeded the operating target during "
                    f"KV release for {owner_id}"
                )

            registered_before = self._pools.kv_physical_bytes
            selected_physical_bytes = sum(
                allocation.physical_bytes for allocation in selected
            )
            registered_after = (
                registered_before - selected_physical_bytes
                if registered_kv_bytes_after is None
                else _exact_nonnegative_int(
                    "registered_kv_bytes_after",
                    registered_kv_bytes_after,
                )
            )
            registered_drop = registered_before - registered_after
            allocator_drop = (
                allocator_before.charged_footprint_bytes
                - allocator_after.charged_footprint_bytes
            )
            conservative_cache = self._conservative_cache_after_release(
                classified_bytes_before=registered_before,
                classified_bytes_after=registered_after,
                cache_bytes_before=cache_bytes_before,
                observed_cache_bytes_after=allocator_after.cache_bytes,
                allocator_footprint_drop=allocator_drop,
            )
            self._pools = replace(
                self._pools,
                kv_physical_bytes=registered_after,
                allocator_cache_bytes=conservative_cache,
            )

            failure: str | None = None
            if any(allocation.cache_id != owner_id for allocation in selected):
                failure = "cross-cache KV release batch is forbidden"
            else:
                active_owner_ids = {
                    allocation_id
                    for allocation_id, allocation in self._allocations.items()
                    if allocation.cache_id == owner_id
                }
                if set(selected_ids) != active_owner_ids:
                    failure = (
                        "KV cache close must include all active allocations "
                        "for its ownership unit"
                    )
            if failure is None and registered_drop != selected_physical_bytes:
                failure = (
                    "registered KV aggregate physical reduction did not match "
                    "selected allocations "
                    f"({registered_drop} observed, "
                    f"{selected_physical_bytes} required)"
                )
            elif failure is None and allocator_drop < 0:
                failure = (
                    "allocator footprint increased during KV release "
                    f"({allocator_drop} byte reduction)"
                )
            elif failure is None:
                cache_transfer = max(
                    0,
                    allocator_after.cache_bytes - allocator_before.cache_bytes,
                )
                if allocator_drop + cache_transfer < registered_drop:
                    failure = (
                        "allocator footprint did not prove the registered KV "
                        "physical reduction"
                    )

            if failure is not None:
                self._terminalize_ambiguous_kv_release(
                    selected,
                    registered_kv_bytes_after=registered_after,
                )
                self._assert_kv_ledger_invariant()
                self._fail_without_pending(failure, pools_already_updated=True)
                raise TerminalizedKVReleaseError(failure)

            for allocation in selected:
                del self._allocations[allocation.allocation_id]
                self._released_allocation_ids.add(allocation.allocation_id)
            if pre_release_limit_failure is not None:
                self._failed_reason = pre_release_limit_failure
                self._transaction_failure_count += 1
            self._assert_kv_ledger_invariant()
            self._revision += 1
            self._record_hard_failure_if_needed()
            return self.snapshot()

    def plan_expert_regrow(
        self, *, target_bytes: int, now_ns: int
    ) -> ExpertRegrowTicket | None:
        """Reserve a slab-aligned regrow without publishing physical bytes."""

        target = _exact_nonnegative_int("target_bytes", target_bytes)
        now = _exact_nonnegative_int("now_ns", now_ns)
        if target == 0:
            return None
        with self._lock:
            self._ensure_allocation_open()
            if self._pending is not None or self._pending_regrow is not None:
                return None
            if (
                self._last_expert_resize_ns is not None
                and now - self._last_expert_resize_ns
                < self._expert_resize_min_interval_ns
            ):
                return None

            missing = max(
                0,
                self._max_expert_slab_bytes - self._pools.expert_slab_physical_bytes,
            )
            if missing < self._expert_slab_bytes:
                return None
            requested_slabs = (
                target + self._expert_slab_bytes - 1
            ) // self._expert_slab_bytes
            missing_slabs = missing // self._expert_slab_bytes
            planned_slabs = min(requested_slabs, missing_slabs)
            planned_bytes = planned_slabs * self._expert_slab_bytes
            hysteresis_bytes = (
                self._expert_regrow_hysteresis_slabs * self._expert_slab_bytes
            )
            headroom = min(
                self._budget.classified_target_bytes - self._pools.classified_bytes,
                self._budget.operating_target_bytes - self._pools.charged_bytes,
            )
            if planned_bytes + hysteresis_bytes > headroom:
                return None

            ticket_id = self._next_ticket_id
            self._next_ticket_id += 1
            self._revision += 1
            ticket = ExpertRegrowTicket(
                ticket_id=ticket_id,
                requested_target_bytes=target,
                planned_slabs=planned_slabs,
                planned_physical_bytes=planned_bytes,
                snapshot_revision=self._revision,
                planned_peak_bytes=self._pools.charged_bytes + planned_bytes,
                planned_at_ns=now,
            )
            self._pending_regrow = _PendingExpertRegrow(
                ticket=ticket,
                expected_revision=self._revision,
            )
            return ticket

    def confirm_expert_regrow(
        self,
        ticket: ExpertRegrowTicket,
        *,
        registered_slab_bytes_after: int,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
        now_ns: int | None = None,
    ) -> BrokerSnapshot:
        """Publish a measured expert regrow and then advance its resize clock."""

        registered_after = _exact_nonnegative_int(
            "registered_slab_bytes_after", registered_slab_bytes_after
        )
        now = None if now_ns is None else _exact_nonnegative_int("now_ns", now_ns)
        with self._lock:
            self._require_regrow_ticket(ticket)
            if now is not None and now < ticket.planned_at_ns:
                reason = (
                    f"expert regrow timestamp {now} is before ticket plan "
                    f"{ticket.planned_at_ns}"
                )
                self._fail_pending_regrow(reason)
                raise MemoryTransactionError(reason)
            if (
                now is not None
                and self._last_expert_resize_ns is not None
                and now < self._last_expert_resize_ns
            ):
                reason = (
                    f"expert regrow timestamp {now} is before last resize "
                    f"{self._last_expert_resize_ns}"
                )
                self._fail_pending_regrow(reason)
                raise MemoryTransactionError(reason)
            if allocator_before is None or allocator_after is None:
                reason = "allocator telemetry unavailable during expert regrow"
                self._fail_pending_regrow(reason)
                raise MemoryTelemetryError(reason)
            try:
                self._validate_allocator_sample("allocator_before", allocator_before)
                self._validate_allocator_sample("allocator_after", allocator_after)
            except (TypeError, ValueError) as exc:
                reason = f"invalid allocator telemetry during expert regrow: {exc}"
                self._fail_pending_regrow(reason)
                raise MemoryTelemetryError(reason) from exc
            if (
                self._allocator_residual_bytes(allocator_before)
                > self._pools.allocator_cache_bytes
            ):
                reason = (
                    "allocator telemetry is stale relative to broker cache "
                    "during expert regrow"
                )
                self._fail_pending_regrow(reason)
                raise MemoryTelemetryError(reason)

            registered_before = self._pools.expert_slab_physical_bytes
            registered_delta = registered_after - registered_before
            allocator_delta = (
                allocator_after.charged_footprint_bytes
                - allocator_before.charged_footprint_bytes
            )
            observed_pair_delta = (
                registered_after
                + allocator_after.cache_bytes
                - registered_before
                - allocator_before.cache_bytes
            )
            minimum_pair_after = (
                registered_before
                + self._pools.allocator_cache_bytes
                + max(0, allocator_delta)
            )
            conservative_cache = max(
                allocator_after.cache_bytes,
                minimum_pair_after - registered_after,
            )
            pinned_after = min(
                self._pools.pinned_expert_bytes,
                registered_after,
            )
            speculative_after = min(
                self._pools.speculative_expert_bytes,
                registered_after,
            )
            self._pools = replace(
                self._pools,
                expert_slab_physical_bytes=registered_after,
                allocator_cache_bytes=conservative_cache,
                pinned_expert_bytes=pinned_after,
                speculative_expert_bytes=speculative_after,
            )

            failure: str | None = None
            if registered_delta != ticket.planned_physical_bytes:
                failure = (
                    "registered expert regrow did not match the reservation "
                    f"({registered_delta} observed, "
                    f"{ticket.planned_physical_bytes} planned)"
                )
            elif registered_after > self._max_expert_slab_bytes:
                failure = "registered expert regrow exceeded original capacity"
            elif now is None:
                failure = "now_ns is required to confirm physical expert regrow"
            elif allocator_delta < 0:
                failure = "allocator footprint fell during expert regrow"
            elif allocator_delta != observed_pair_delta:
                failure = (
                    "allocator footprint did not match the registered expert "
                    "regrow classification"
                )
            else:
                hysteresis_bytes = (
                    self._expert_regrow_hysteresis_slabs * self._expert_slab_bytes
                )
                if (
                    self._pools.classified_bytes + hysteresis_bytes
                    > self._budget.classified_target_bytes
                ):
                    failure = (
                        "confirmed expert regrow violated allocator-headroom hysteresis"
                    )
                elif (
                    self._pools.charged_bytes + hysteresis_bytes
                    > self._budget.operating_target_bytes
                ):
                    failure = "confirmed expert regrow violated operating hysteresis"
            if failure is not None:
                self._fail_pending_regrow(
                    failure,
                    pools_already_updated=True,
                )
                raise MemoryTelemetryError(failure)

            self._consume_pending_regrow()
            assert now is not None
            self._last_expert_resize_ns = now
            self._revision += 1
            self._record_hard_failure_if_needed()
            return self.snapshot()

    def abort_expert_regrow(self, ticket: ExpertRegrowTicket) -> BrokerSnapshot:
        """Cancel a regrow reservation before any physical allocation."""

        with self._lock:
            self._require_regrow_ticket(ticket)
            self._consume_pending_regrow()
            self._transaction_failure_count += 1
            self._revision += 1
            return self.snapshot()

    def _copy_pool_fields(self, snapshot: BrokerSnapshot) -> None:
        self._pools = BrokerSnapshot(
            resident_model_bytes=snapshot.resident_model_bytes,
            kv_physical_bytes=snapshot.kv_physical_bytes,
            expert_slab_physical_bytes=(snapshot.expert_slab_physical_bytes),
            in_flight_expert_staging_bytes=(snapshot.in_flight_expert_staging_bytes),
            runtime_workspace_bytes=snapshot.runtime_workspace_bytes,
            allocator_cache_bytes=snapshot.allocator_cache_bytes,
            pinned_expert_bytes=snapshot.pinned_expert_bytes,
            speculative_expert_bytes=snapshot.speculative_expert_bytes,
        )

    def _validate_snapshot(self, snapshot: BrokerSnapshot) -> None:
        if not isinstance(snapshot, BrokerSnapshot):
            raise TypeError("snapshot must be a BrokerSnapshot")
        for name in (
            "resident_model_bytes",
            "kv_physical_bytes",
            "expert_slab_physical_bytes",
            "in_flight_expert_staging_bytes",
            "runtime_workspace_bytes",
            "allocator_cache_bytes",
            "pinned_expert_bytes",
            "speculative_expert_bytes",
        ):
            _exact_nonnegative_int(name, getattr(snapshot, name))
        if snapshot.pinned_expert_bytes > snapshot.expert_slab_physical_bytes:
            raise ValueError("pinned_expert_bytes cannot exceed physical expert slabs")
        if snapshot.speculative_expert_bytes > snapshot.expert_slab_physical_bytes:
            raise ValueError(
                "speculative_expert_bytes cannot exceed physical expert slabs"
            )

    @staticmethod
    def _validate_allocator_sample(name: str, sample: AllocatorMemorySample) -> None:
        if not isinstance(sample, AllocatorMemorySample):
            raise TypeError(f"{name} must be an AllocatorMemorySample")
        _exact_nonnegative_int(f"{name}.active_bytes", sample.active_bytes)
        _exact_nonnegative_int(f"{name}.cache_bytes", sample.cache_bytes)
        _exact_nonnegative_int(f"{name}.peak_bytes", sample.peak_bytes)

    def _allocator_residual_bytes(self, sample: AllocatorMemorySample) -> int:
        classified_bytes = (
            self._pools.resident_model_bytes
            + self._pools.kv_physical_bytes
            + self._pools.expert_slab_physical_bytes
            + self._pools.in_flight_expert_staging_bytes
            + self._pools.runtime_workspace_bytes
        )
        return max(
            sample.cache_bytes,
            sample.charged_footprint_bytes - classified_bytes,
        )

    @staticmethod
    def _validate_cache_id(cache_id: object) -> str:
        if not isinstance(cache_id, str):
            raise TypeError("cache_id must be a string")
        normalized = cache_id.strip()
        if not normalized:
            raise ValueError("cache_id must not be empty")
        return normalized

    @staticmethod
    def _conservative_cache_after_release(
        *,
        classified_bytes_before: int,
        classified_bytes_after: int,
        cache_bytes_before: int,
        observed_cache_bytes_after: int,
        allocator_footprint_drop: int,
    ) -> int:
        """Keep every byte not proven released in a charged residual pool."""

        residual = (
            classified_bytes_before
            + cache_bytes_before
            - allocator_footprint_drop
            - classified_bytes_after
        )
        return max(0, observed_cache_bytes_after, residual)

    def _terminalize_ambiguous_kv_release(
        self,
        selected: Sequence[KVPhysicalAllocation],
        *,
        registered_kv_bytes_after: int,
    ) -> None:
        """Replace uncertain handles with an explicit terminal residual."""

        selected_ids = {allocation.allocation_id for allocation in selected}
        for allocation_id in selected_ids:
            self._allocations.pop(allocation_id, None)
            self._ambiguous_allocation_ids.add(allocation_id)

        remaining_ledger_bytes = sum(
            allocation.physical_bytes for allocation in self._allocations.values()
        ) + sum(self._unreconciled_kv_by_owner.values())
        if registered_kv_bytes_after >= remaining_ledger_bytes:
            residual = registered_kv_bytes_after - remaining_ledger_bytes
            if residual:
                key = (
                    f"ambiguous:{self._revision}:"
                    f"{','.join(str(value) for value in sorted(selected_ids))}"
                )
                self._unreconciled_kv_by_owner[key] = residual
            return

        # The observation says bytes outside the selected set also vanished.
        # Their ownership can no longer be authenticated, so terminalize the
        # complete ledger rather than falsely claiming any individual release.
        self._ambiguous_allocation_ids.update(self._allocations)
        self._allocations.clear()
        self._unreconciled_kv_by_owner.clear()
        if registered_kv_bytes_after:
            self._unreconciled_kv_by_owner[f"ambiguous-global:{self._revision}"] = (
                registered_kv_bytes_after
            )

    def _assert_kv_ledger_invariant(self) -> None:
        owned = sum(
            allocation.physical_bytes for allocation in self._allocations.values()
        )
        unreconciled = sum(self._unreconciled_kv_by_owner.values())
        if owned < 0 or unreconciled < 0:
            raise AssertionError("KV ownership ledger contains negative bytes")
        if self._pools.kv_physical_bytes != owned + unreconciled:
            raise AssertionError(
                "KV physical accounting invariant violated: "
                f"snapshot={self._pools.kv_physical_bytes}, owned={owned}, "
                f"unreconciled={unreconciled}"
            )

    def _ensure_allocation_open(self) -> None:
        if self._pools.charged_bytes >= self._budget.hard_ceiling_bytes:
            self._reject_admission("memory is at or above the hard ceiling")
        if self._pools.classified_bytes > self._budget.classified_target_bytes:
            self._reject_admission(
                "classified memory is above the allocator-headroom target"
            )
        if self._failed_reason is not None:
            self._reject_admission(
                f"memory broker failed closed: {self._failed_reason}"
            )

    def _reject_admission(self, reason: str) -> None:
        self._admission_failure_count += 1
        self._revision += 1
        raise MemoryAdmissionError(reason)

    def _require_ticket(self, ticket: KVAllocationTicket) -> _PendingKVTransaction:
        if not isinstance(ticket, KVAllocationTicket):
            raise TypeError("ticket must be a KVAllocationTicket")
        if ticket.ticket_id in self._consumed_ticket_ids:
            raise MemoryTransactionError(
                f"KV ticket {ticket.ticket_id} was already consumed"
            )
        transaction = self._pending
        if transaction is None or transaction.ticket is not ticket:
            raise MemoryTransactionError(
                f"KV ticket {ticket.ticket_id} is not the active ticket"
            )
        if transaction.expected_revision != self._revision:
            self._fail_pending(
                f"KV ticket {ticket.ticket_id} revision changed during transaction"
            )
            raise MemoryTransactionError(
                f"KV ticket {ticket.ticket_id} revision changed during transaction"
            )
        return transaction

    def _require_group_ticket(
        self,
        ticket: KVAllocationGroupTicket,
    ) -> _PendingKVGroupTransaction:
        if not isinstance(ticket, KVAllocationGroupTicket):
            raise TypeError("ticket must be a KVAllocationGroupTicket")
        if ticket.ticket_id in self._consumed_ticket_ids:
            raise MemoryTransactionError(
                f"KV group ticket {ticket.ticket_id} was already consumed"
            )
        transaction = self._pending
        if (
            not isinstance(transaction, _PendingKVGroupTransaction)
            or transaction.ticket is not ticket
        ):
            raise MemoryTransactionError(
                f"KV group ticket {ticket.ticket_id} is not the active ticket"
            )
        if transaction.expected_revision != self._revision:
            reason = (
                f"KV group ticket {ticket.ticket_id} revision changed during "
                "transaction"
            )
            self._fail_pending(reason)
            raise MemoryTransactionError(reason)
        return transaction

    def _consume_pending(self) -> None:
        if self._pending is None:
            return
        self._consumed_ticket_ids.add(self._pending.ticket.ticket_id)
        self._pending = None

    def _require_regrow_ticket(
        self, ticket: ExpertRegrowTicket
    ) -> _PendingExpertRegrow:
        if not isinstance(ticket, ExpertRegrowTicket):
            raise TypeError("ticket must be an ExpertRegrowTicket")
        if ticket.ticket_id in self._consumed_regrow_ticket_ids:
            raise MemoryTransactionError(
                f"expert regrow ticket {ticket.ticket_id} was already consumed"
            )
        transaction = self._pending_regrow
        if transaction is None or transaction.ticket is not ticket:
            raise MemoryTransactionError(
                f"expert regrow ticket {ticket.ticket_id} is not active"
            )
        if transaction.expected_revision != self._revision:
            reason = (
                f"expert regrow ticket {ticket.ticket_id} revision changed "
                "during transaction"
            )
            self._fail_pending_regrow(reason)
            raise MemoryTransactionError(reason)
        return transaction

    def _consume_pending_regrow(self) -> None:
        if self._pending_regrow is None:
            return
        self._consumed_regrow_ticket_ids.add(self._pending_regrow.ticket.ticket_id)
        self._pending_regrow = None

    def _fail_pending(
        self,
        reason: str,
        *,
        pools_already_updated: bool = False,
    ) -> None:
        del pools_already_updated  # Documents that observed pools were retained.
        self._failed_reason = reason
        self._transaction_failure_count += 1
        self._consume_pending()
        self._revision += 1
        self._record_hard_failure_if_needed()

    def _fail_pending_regrow(
        self,
        reason: str,
        *,
        pools_already_updated: bool = False,
    ) -> None:
        del pools_already_updated
        self._failed_reason = reason
        self._transaction_failure_count += 1
        self._consume_pending_regrow()
        self._revision += 1
        self._record_hard_failure_if_needed()

    def _fail_without_pending(
        self,
        reason: str,
        *,
        pools_already_updated: bool = False,
    ) -> None:
        del pools_already_updated
        self._failed_reason = reason
        self._transaction_failure_count += 1
        self._revision += 1
        self._record_hard_failure_if_needed()

    def _record_hard_failure_if_needed(self) -> None:
        if self._pools.charged_bytes < self._budget.hard_ceiling_bytes:
            return
        self._hard_failure_count += 1
        if self._failed_reason is None:
            self._failed_reason = "observed memory reached the hard ceiling"


__all__ = [
    "BINARY_GIB",
    "HY3_Q4_KV_BLOCK_BYTES",
    "HY3_Q4_ALLOCATOR_HEADROOM_BYTES",
    "HY3_Q4_KV_BLOCK_TOKENS",
    "HY3_Q4_KV_BYTES_PER_TOKEN",
    "HY3_Q4_KV_HEAD_DIM",
    "HY3_Q4_KV_HEADS",
    "HY3_Q4_KV_LAYERS",
    "HY3_Q4_KV_SCALE_BYTES",
    "AllocatorMemorySample",
    "BrokerSnapshot",
    "DuplicateReleaseError",
    "ExpertRegrowTicket",
    "KVAllocationGroupMember",
    "KVAllocationGroupTicket",
    "KVAllocationTicket",
    "KVPhysicalAllocation",
    "Hy3Q4KVPhysicalGeometry",
    "MemoryAdmissionError",
    "MemoryBrokerError",
    "MemoryBudget",
    "MemoryTelemetryError",
    "MemoryTransactionError",
    "TerminalizedKVReleaseError",
    "UnifiedMemoryBroker",
    "hy3_q4_kv_physical_geometry",
]
