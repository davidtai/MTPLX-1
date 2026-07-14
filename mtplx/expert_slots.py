"""Fixed expert slot buffers with generation-safe load and pin lifetimes."""

from __future__ import annotations

import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from .expert_io import ExpertIOError, PositionalExpertReader
from .expert_manifest import ExpertManifest, ExpertManifestError, ExpertRecord
from .resource_metrics import (
    ExpertPipelineLedger,
    ExpertPipelineRoute,
    PoolOccupancy,
)
from .expert_streaming import RoutePlan, SlotLoad
from .expert_streaming_models import (
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
)


class ExpertSlotError(RuntimeError):
    pass


class ExpertCompletionFenceError(ExpertSlotError):
    """Sticky completion failure annotated with policy rollback safety."""

    def __init__(self, message: str, *, policy_rollback_safe: bool) -> None:
        super().__init__(message)
        self.policy_rollback_safe = bool(policy_rollback_safe)


def _closed_allocator(_size: int, _label: str) -> Any:
    raise ExpertSlotError("expert slot pool is closed; its allocator was released")


@dataclass
class RouteIOAdmission:
    """Per-route record of executor work accepted past the rollback boundary."""

    accepted_submissions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _parent: RouteIOAdmission | None = field(default=None, repr=False)

    @property
    def any_accepted(self) -> bool:
        with self._lock:
            return self.accepted_submissions > 0

    def mark_accepted(self) -> None:
        with self._lock:
            self.accepted_submissions += 1
        if self._parent is not None:
            self._parent.mark_accepted()

    def child(self) -> RouteIOAdmission:
        """Create a part-local admission that also updates this route."""

        return RouteIOAdmission(_parent=self)


class ExpertSlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class ExpertSlabState(str, Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    RELEASED = "released"


@dataclass
class ExpertSlab:
    slab_id: int
    slot_ids: tuple[int, ...]
    state: ExpertSlabState
    incarnation: int
    physical_bytes: int


@dataclass(frozen=True)
class _SlabSlotSnapshot:
    slot_id: int
    state: ExpertSlotState
    layer: int | None
    expert: int | None
    generation: int
    digest: str | None
    error: BaseException | None


@dataclass(frozen=True)
class ExpertSlabReclaimTicket:
    ticket_id: int
    slab_ids: tuple[int, ...]
    slot_ids: tuple[int, ...]
    physical_bytes: int
    incarnations: tuple[int, ...]
    slots: tuple[_SlabSlotSnapshot, ...]


@dataclass(frozen=True)
class ExpertSlabReclaimResult:
    slab_ids: tuple[int, ...]
    released_slot_ids: tuple[int, ...]
    physical_bytes: int


@dataclass(frozen=True)
class ExpertSlabReleaseResult:
    """Allocator proof that one slab crossed the destructive boundary."""

    slab_id: int
    physical_bytes: int
    destroyed: bool


class ExpertSlabAllocatorError(RuntimeError):
    """Allocator failure annotated with destructive-boundary truth."""

    def __init__(
        self,
        message: str,
        *,
        destroyed: bool,
        physical_bytes: int,
    ) -> None:
        super().__init__(message)
        self.destroyed = bool(destroyed)
        self.physical_bytes = int(physical_bytes)


class ExpertSlabReclaimError(ExpertSlotError):
    """Partial reclaim result plus the failed slab's boundary state."""

    def __init__(
        self,
        message: str,
        *,
        result: ExpertSlabReclaimResult,
        failed_slab_id: int,
        destructive_boundary_crossed: bool | None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.failed_slab_id = int(failed_slab_id)
        self.destructive_boundary_crossed = destructive_boundary_crossed


@dataclass(eq=False)
class _RouteReleaseClaim:
    released: bool = False


@dataclass(eq=False)
class _SlotPinClaim:
    active: bool = True
    completion_owned: bool = False


@dataclass
class ExpertSlotMetrics:
    ensure_calls: int = 0
    load_requests: int = 0
    owned_loads: int = 0
    deduplicated_loads: int = 0
    ready_hits: int = 0
    load_failures: int = 0
    generation_replacements: int = 0
    pin_waits: int = 0
    load_waits: int = 0
    active_routes: int = 0
    active_routes_peak: int = 0
    completion_fences: int = 0
    completion_fence_slots: int = 0
    completion_fence_fallbacks: int = 0
    completion_fence_failures: int = 0
    _physical_reads_by_layer: dict[int, dict[str, int]] = field(
        default_factory=dict, repr=False
    )
    synchronous_fences: int = 0
    synchronous_fence_slots: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _route_claims: list[_RouteReleaseClaim] = field(default_factory=list, repr=False)
    _admission_test_hook: Callable[[str], None] | None = field(default=None, repr=False)

    def update(self, **values: int) -> None:
        with self._lock:
            for name, value in values.items():
                setattr(self, name, int(getattr(self, name)) + int(value))
            self.active_routes_peak = max(self.active_routes_peak, self.active_routes)

    def admit_route(self, claim: _RouteReleaseClaim) -> None:
        with self._lock:
            previous_claims = self._route_claims
            previous_active = self.active_routes
            previous_peak = self.active_routes_peak
            hook = self._admission_test_hook
            try:
                if hook is not None:
                    hook("before_filter")
                filtered = [item for item in previous_claims if not item.released]
                if hook is not None:
                    hook("after_filter")
                next_claims = [*filtered, claim]
                if hook is not None:
                    hook("after_append")
                next_active = sum(not item.released for item in next_claims)
                if hook is not None:
                    hook("after_active_count")
                next_peak = max(previous_peak, next_active)

                self._route_claims = next_claims
                self.active_routes = next_active
                if hook is not None:
                    hook("after_publish")
                self.active_routes_peak = next_peak
                if hook is not None:
                    hook("after_peak")
            except BaseException:
                self._route_claims = previous_claims
                self.active_routes = previous_active
                self.active_routes_peak = previous_peak
                raise

    def release_route(self, claim: _RouteReleaseClaim) -> None:
        """Consume one token and reconcile aggregate accounting idempotently."""

        with self._lock:
            claim.released = True
            self.active_routes = sum(not item.released for item in self._route_claims)

    def observe_physical_read(
        self,
        layer: int,
        *,
        records: int,
        elapsed_ns: int,
    ) -> None:
        with self._lock:
            row = self._physical_reads_by_layer.setdefault(
                int(layer), {"operations": 0, "records": 0, "elapsed_ns": 0}
            )
            row["operations"] += 1
            row["records"] += int(records)
            row["elapsed_ns"] += int(elapsed_ns)

    def physical_read_latency_by_layer(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            rows = {
                layer: dict(values)
                for layer, values in self._physical_reads_by_layer.items()
            }
        return {
            str(layer): {
                **values,
                "mean_operation_ms": (
                    values["elapsed_ns"] / values["operations"] / 1e6
                    if values["operations"]
                    else 0.0
                ),
                "mean_record_ms": (
                    values["elapsed_ns"] / values["records"] / 1e6
                    if values["records"]
                    else 0.0
                ),
            }
            for layer, values in sorted(rows.items())
        }

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                name: int(getattr(self, name))
                for name in (
                    "ensure_calls",
                    "load_requests",
                    "owned_loads",
                    "deduplicated_loads",
                    "ready_hits",
                    "load_failures",
                    "generation_replacements",
                    "pin_waits",
                    "load_waits",
                    "active_routes",
                    "active_routes_peak",
                    "completion_fences",
                    "completion_fence_slots",
                    "completion_fence_fallbacks",
                    "completion_fence_failures",
                    "synchronous_fences",
                    "synchronous_fence_slots",
                )
            }


@dataclass
class _PhysicalSlot:
    label: str
    buffer: Any
    slab_id: int | None = None
    state: ExpertSlotState = ExpertSlotState.EMPTY
    layer: int | None = None
    expert: int | None = None
    generation: int = 0
    pins: int = 0
    pin_claims: list[_SlotPinClaim] = field(default_factory=list, repr=False)
    digest: str | None = None
    error: BaseException | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)


@dataclass(frozen=True)
class _PreparedSlotState:
    slot: _PhysicalSlot
    owned_generation: int
    state: ExpertSlotState
    layer: int | None
    expert: int | None
    generation: int
    digest: str | None
    error: BaseException | None


@dataclass(frozen=True)
class ExpertSlotBinding:
    layer: int
    expert: int
    logical_slot: int
    generation: int
    record: ExpertRecord
    buffer: Any

    def component_view(self, component: str) -> memoryview:
        direct = getattr(self.buffer, "component_view", None)
        if callable(direct):
            return direct(component)
        view = memoryview(self.buffer)
        if view.readonly or not view.c_contiguous:
            raise ExpertSlotError("slot buffer is not a writable contiguous buffer")
        raw = view.cast("B")
        cursor = 0
        for segment in self.record.segments:
            end = cursor + segment.length
            if segment.component == component:
                return raw[cursor:end]
            cursor = end
        raise KeyError(component)


class ReadyRoute:
    """Pinned slot bindings that remain valid until explicit release."""

    def __init__(
        self,
        pool: ExpertSlotPool,
        plan: RoutePlan,
        bindings: tuple[ExpertSlotBinding, ...],
        pinned: tuple[tuple[_PhysicalSlot, _SlotPinClaim], ...],
        lifecycle_claim: _RouteReleaseClaim,
    ) -> None:
        self.pool = pool
        self.plan = plan
        self.bindings = bindings
        self._pinned = tuple(slot for slot, _claim in pinned)
        self._released = False
        self._route_finished = False
        self._route_finishing = False
        self._lifecycle_claim = lifecycle_claim
        self._pin_claims = {id(slot): claim for slot, claim in pinned}
        self._release_in_progress = False
        self._release_lock = threading.Lock()
        self._release_condition = threading.Condition(self._release_lock)
        self._pending_slots = {id(slot): slot for slot, _claim in pinned}
        self._scheduled_slots: set[int] = set()
        self._completion_futures: list[Future[None]] = []
        self._registrations_in_progress = 0

    @property
    def slots(self) -> tuple[int, ...]:
        return self.plan.slots

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    def validate(self) -> None:
        if self._released:
            raise ExpertSlotError("ready route has already been released")
        for binding, slot in zip(self.bindings, self._binding_slots(), strict=True):
            with slot.condition:
                if (
                    slot.state is not ExpertSlotState.READY
                    or slot.layer != binding.layer
                    or slot.expert != binding.expert
                    or slot.generation != binding.generation
                ):
                    raise ExpertSlotError(
                        "ready route references a stale slot generation"
                    )

    def defer_bindings_until(
        self,
        bindings: Iterable[ExpertSlotBinding],
        completion_waiter: Callable[[], None],
    ) -> bool:
        """Hold selected slot generations until their Metal work completes.

        The waiter runs on the pool's bounded completion lane. This lets the
        generation thread proceed to miss I/O without making a slot reusable
        before the asynchronous kernels that consume it have completed.
        """

        if not callable(completion_waiter):
            raise TypeError("completion_waiter must be callable")
        selected: dict[int, _PhysicalSlot] = {}
        for binding in bindings:
            slot = self.pool._physical(binding.layer, binding.logical_slot)
            selected[id(slot)] = slot
        if not selected:
            return False
        with self._release_condition:
            if self._released:
                raise ExpertSlotError("cannot fence a released ready route")
            unknown = set(selected) - set(self._pending_slots)
            if unknown:
                raise ExpertSlotError("completion fence references an unpinned slot")
            duplicate = set(selected) & self._scheduled_slots
            if duplicate:
                raise ExpertSlotError("slot already has a completion fence")
            self._scheduled_slots.update(selected)
            for slot_id in selected:
                self._pin_claims[slot_id].completion_owned = True
            self._registrations_in_progress += 1
        try:
            future = self.pool._submit_completion_fence(
                completion_waiter,
                lambda: self._run_claimed_slot_cleanup(tuple(selected.values())),
                slot_count=len(selected),
            )
        except BaseException:
            with self._release_condition:
                self._scheduled_slots.difference_update(selected)
                for slot_id in selected:
                    claim = self._pin_claims.get(slot_id)
                    if claim is not None:
                        claim.completion_owned = False
                self._registrations_in_progress -= 1
                self._release_condition.notify_all()
            raise
        with self._release_condition:
            if future is not None:
                self._completion_futures.append(future)
            self._registrations_in_progress -= 1
            self._release_condition.notify_all()
        return True

    def _binding_slots(self) -> tuple[_PhysicalSlot, ...]:
        return tuple(
            self.pool._physical(binding.layer, binding.logical_slot)
            for binding in self.bindings
        )

    def _finish_slots(self, slots: tuple[_PhysicalSlot, ...]) -> None:
        for slot in slots:
            self._finish_slot_claim(slot)
        finish_route = False
        with self._release_condition:
            if (
                self._released
                and not self._pending_slots
                and not self._route_finished
                and not self._route_finishing
            ):
                self._route_finishing = True
                finish_route = True
        if finish_route:
            self._finish_route_lifecycle()

    def _finish_slot_claim(self, slot: _PhysicalSlot) -> None:
        slot_id = id(slot)
        with self._release_condition:
            if slot_id not in self._pending_slots:
                self._scheduled_slots.discard(slot_id)
                return
            claim = self._pin_claims[slot_id]
            self._release_physical_claim(slot, claim)
            self._complete_slot_claim(slot_id)
            self._release_condition.notify_all()

    @staticmethod
    def _release_physical_claim(slot: _PhysicalSlot, claim: _SlotPinClaim) -> None:
        with slot.condition:
            claim.active = False
            slot.pins = sum(item.active for item in slot.pin_claims)
            slot.condition.notify_all()

    def _complete_slot_claim(self, slot_id: int) -> None:
        slot = self._pending_slots.get(slot_id)
        claim = self._pin_claims.get(slot_id)
        if slot is not None and claim is not None and claim in slot.pin_claims:
            slot.pin_claims.remove(claim)
        self._pending_slots.pop(slot_id, None)
        self._pin_claims.pop(slot_id, None)
        self._scheduled_slots.discard(slot_id)

    def _mark_slot_cleanup_failure(
        self,
        slots: tuple[_PhysicalSlot, ...],
        error: BaseException,
    ) -> None:
        with self._release_condition:
            for slot in slots:
                slot_id = id(slot)
                if slot_id in self._pending_slots:
                    self._scheduled_slots.discard(slot_id)
            self._release_condition.notify_all()
        self.pool._record_cleanup_error(error)

    def _run_claimed_slot_cleanup(self, slots: tuple[_PhysicalSlot, ...]) -> None:
        try:
            self._finish_slots(slots)
        except BaseException as exc:
            self._mark_slot_cleanup_failure(slots, exc)
            raise

    def _finish_route_lifecycle(self) -> None:
        try:
            self.pool._route_released(self._lifecycle_claim)
        except BaseException as exc:
            with self._release_condition:
                self._route_finishing = False
                self._release_condition.notify_all()
            self.pool._record_cleanup_error(exc)
            raise
        with self._release_condition:
            self._route_finishing = False
            self._route_finished = True
            self._release_condition.notify_all()

    def release(self, *, synchronize: bool = True) -> None:
        with self._release_condition:
            while self._release_in_progress:
                if not synchronize:
                    return
                self._release_condition.wait()
            self._release_in_progress = True
        try:
            self._release_owned(synchronize=synchronize)
        finally:
            with self._release_condition:
                self._release_in_progress = False
                self._release_condition.notify_all()

    def _release_owned(self, *, synchronize: bool) -> None:
        with self._release_condition:
            first_release = not self._released
            self._released = True
            while self._registrations_in_progress:
                self._release_condition.wait()
            immediate = tuple(
                slot
                for slot_id, slot in self._pending_slots.items()
                if slot_id not in self._scheduled_slots
            )
            self._scheduled_slots.update(id(slot) for slot in immediate)
            futures = tuple(self._completion_futures)
            finish_empty = (
                not self._pending_slots
                and not self._route_finished
                and not self._route_finishing
            )
            if finish_empty:
                self._route_finishing = True
        synchronize_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        future_error: BaseException | None = None
        if first_release and synchronize and self.pool.device_synchronize is not None:
            try:
                self.pool.device_synchronize()
            except BaseException as exc:
                self.pool.metrics.update(completion_fence_failures=1)
                self.pool._record_completion_error(exc)
                synchronize_error = exc
        try:
            if immediate:
                self._run_claimed_slot_cleanup(immediate)
            elif finish_empty:
                self._finish_route_lifecycle()
        except BaseException as exc:
            cleanup_error = exc
        if synchronize:
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    if future_error is None:
                        future_error = exc
        if synchronize_error is not None:
            raise synchronize_error
        if future_error is not None:
            raise ExpertSlotError("expert completion fence failed") from future_error
        if cleanup_error is not None:
            raise cleanup_error
        if synchronize:
            self.pool._raise_completion_error()

    def __enter__(self) -> ReadyRoute:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _RouteLifecycle:
    """One extra active-route hold spanning a multi-part transaction."""

    def __init__(self, pool: ExpertSlotPool, claim: _RouteReleaseClaim) -> None:
        self.pool = pool
        self._claim = claim

    def release(self) -> None:
        try:
            self.pool._route_released(self._claim)
        except BaseException as exc:
            self.pool._record_cleanup_error(exc)
            raise


class _FailedRouteSetupOwner:
    """Retryable owner for pins and lifecycle from failed route construction."""

    def __init__(
        self,
        pool: ExpertSlotPool,
        lifecycle: _RouteLifecycle,
        pins: dict[int, tuple[_PhysicalSlot, _SlotPinClaim]],
    ) -> None:
        self.pool = pool
        self.lifecycle = lifecycle
        self.pins = pins

    @property
    def terminal(self) -> bool:
        return not self.pins and self.lifecycle._claim.released

    def release(self) -> None:
        hook = self.pool._failed_setup_cleanup_test_hook
        for slot_id, (slot, claim) in tuple(self.pins.items()):
            if hook is not None:
                hook("before_pin", slot_id)
            with slot.condition:
                claim.active = False
                slot.pins = sum(item.active for item in slot.pin_claims)
                slot.condition.notify_all()
            if hook is not None:
                hook("after_pin_reconcile", slot_id)
            with slot.condition:
                if claim in slot.pin_claims:
                    slot.pin_claims.remove(claim)
            if hook is not None:
                hook("after_pin_list", slot_id)
            self.pins.pop(slot_id, None)
        if hook is not None:
            hook("before_lifecycle", -1)
        self.lifecycle.release()
        if hook is not None:
            hook("after_lifecycle", -1)


class _CombinedCancel:
    def __init__(
        self, external: threading.Event | None, internal: threading.Event
    ) -> None:
        self.external = external
        self.internal = internal

    def is_set(self) -> bool:
        return self.internal.is_set() or (
            self.external is not None and self.external.is_set()
        )


class ExpertSlotPool:
    """Own all persistent and transient record buffers for one model."""

    def __init__(
        self,
        spec: ExpertStreamingModelSpec,
        plan: ExpertMemoryPlan,
        manifest: ExpertManifest,
        reader: PositionalExpertReader,
        *,
        buffer_allocator: Callable[[int, str], Any] | None = None,
        max_inflight_io_bytes: int | None = None,
        prefer_sidecar: bool = True,
        verify_hashes: bool = True,
        device_synchronize: Callable[[], None] | None = None,
        cache_scope: str = "layer",
        resource_telemetry: bool = False,
        pipeline_ledger: ExpertPipelineLedger | None = None,
    ) -> None:
        if plan.model_key != spec.key or manifest.model_key != spec.key:
            raise ValueError("spec, memory plan, and manifest model keys must match")
        if (
            manifest.quant_bits != spec.quant_bits
            or manifest.quant_group_size != spec.quant_group_size
        ):
            raise ValueError("manifest quantization does not match model descriptor")
        if manifest.routed_expert_bytes != spec.routed_expert_bytes:
            raise ValueError("manifest routed bytes do not match model descriptor")
        if plan.transient_slots < spec.top_k:
            raise ValueError("memory plan transient slots do not cover model top-k")
        if max_inflight_io_bytes is None:
            max_inflight_io_bytes = max(plan.transient_bytes, spec.expert_record_bytes)
        if (
            isinstance(max_inflight_io_bytes, bool)
            or not isinstance(max_inflight_io_bytes, int)
            or max_inflight_io_bytes < spec.expert_record_bytes
        ):
            raise ValueError(
                "max_inflight_io_bytes must cover at least one expert record"
            )
        self.spec = spec
        self.plan = plan
        self.manifest = manifest
        self.reader = reader
        self.prefer_sidecar = prefer_sidecar
        self.verify_hashes = verify_hashes
        self.device_synchronize = device_synchronize
        if not isinstance(resource_telemetry, bool):
            raise TypeError("resource_telemetry must be bool")
        self.resource_telemetry_enabled = resource_telemetry
        self._pipeline_ledger = pipeline_ledger
        if cache_scope not in {"layer", "global"}:
            raise ValueError("cache_scope must be 'layer' or 'global'")
        self.cache_scope = cache_scope
        self.global_persistent_slots = (
            plan.persistent_slots if cache_scope == "global" else 0
        )
        self._persistent_route_capacity = (
            self.global_persistent_slots
            if cache_scope == "global"
            else plan.slots_per_layer
        )
        self.metrics = ExpertSlotMetrics()
        self._allocator = buffer_allocator or (lambda size, _label: bytearray(size))
        self._owner_thread_id = threading.get_ident()
        self.buffer_backend = str(
            getattr(self._allocator, "backend", "python-bytearray")
        )
        self._slab_lock = threading.RLock()
        self._slab_ticket_clock = 0
        self._slab_tickets: dict[int, ExpertSlabReclaimTicket] = {}
        self._slab_release_failures: dict[int, BaseException] = {}
        self._slab_ambiguous_allocations: set[int] = set()
        self._slab_reclaim_test_hook: Callable[[str, int], None] | None = None
        slab_layout_provider = getattr(self._allocator, "slab_layout", None)
        raw_slab_layout = (
            slab_layout_provider()
            if self.cache_scope == "global" and callable(slab_layout_provider)
            else {}
        )
        slab_layout: dict[int, tuple[int, ...]] = {}
        seen_slab_slots: set[int] = set()
        for raw_slab_id, raw_slot_ids in raw_slab_layout.items():
            if (
                isinstance(raw_slab_id, bool)
                or not isinstance(raw_slab_id, int)
                or raw_slab_id < 0
            ):
                raise ValueError("expert slab ids must be nonnegative integers")
            slot_ids = tuple(raw_slot_ids)
            if not slot_ids:
                raise ValueError("expert slabs must contain at least one slot")
            if any(
                isinstance(slot_id, bool)
                or not isinstance(slot_id, int)
                or not 0 <= slot_id < self.global_persistent_slots
                for slot_id in slot_ids
            ):
                raise ValueError("expert slab slot is outside persistent capacity")
            if len(set(slot_ids)) != len(slot_ids):
                raise ValueError("expert slab contains duplicate slots")
            if seen_slab_slots.intersection(slot_ids):
                raise ValueError("persistent slots belong to multiple expert slabs")
            seen_slab_slots.update(slot_ids)
            slab_layout[raw_slab_id] = slot_ids
        if slab_layout and seen_slab_slots != set(range(self.global_persistent_slots)):
            raise ValueError("expert slab layout does not cover persistent capacity")
        slab_bytes_provider = getattr(self._allocator, "slab_physical_bytes", None)
        if slab_layout and (
            not callable(getattr(self._allocator, "allocate_slab", None))
            or not callable(getattr(self._allocator, "release_slab", None))
            or not callable(getattr(self._allocator, "slab_is_allocated", None))
            or not callable(slab_bytes_provider)
        ):
            raise ValueError("expert slab allocator lifecycle methods are incomplete")
        self._slab_for_slot = {
            slot_id: slab_id
            for slab_id, slot_ids in slab_layout.items()
            for slot_id in slot_ids
        }
        self._slabs = {
            slab_id: ExpertSlab(
                slab_id=slab_id,
                slot_ids=slot_ids,
                state=ExpertSlabState.ACTIVE,
                incarnation=0,
                physical_bytes=int(slab_bytes_provider(slab_id)),
            )
            for slab_id, slot_ids in slab_layout.items()
        }
        for slab in self._slabs.values():
            expected_slab_bytes = len(slab.slot_ids) * spec.expert_record_bytes
            if slab.physical_bytes != expected_slab_bytes:
                raise ValueError(
                    "expert slab physical bytes differ from its slot geometry"
                )
        self._persistent: dict[tuple[int, int], _PhysicalSlot] = {}
        self._transient: tuple[_PhysicalSlot, ...]
        self._record_map = {
            (record.layer, record.expert): record for record in manifest.records
        }
        self._ensure_locks = {
            layer: threading.Lock() for layer in spec.routed_layer_indices
        }
        self._lifecycle = threading.Condition()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._route_setup_test_hook: Callable[[str], None] | None = None
        self._failed_setup_cleanup_test_hook: Callable[[str, int], None] | None = None
        self._cleanup_owner_lock = threading.Lock()
        self._cleanup_owners: list[_FailedRouteSetupOwner] = []

        allocated = 0
        try:
            persistent_layout = (
                (
                    (None, slot_index)
                    for slot_index in range(self.global_persistent_slots)
                )
                if self.cache_scope == "global"
                else (
                    (layer, slot_index)
                    for layer in spec.routed_layer_indices
                    for slot_index in range(plan.slots_per_layer)
                )
            )
            for layer, slot_index in persistent_layout:
                label = (
                    f"global-persistent-{slot_index}"
                    if layer is None
                    else f"layer-{layer}-persistent-{slot_index}"
                )
                buffer = self._allocate_buffer(label)
                key_layer = -1 if layer is None else layer
                slab_id = self._slab_for_slot.get(slot_index) if layer is None else None
                self._persistent[(key_layer, slot_index)] = _PhysicalSlot(
                    label,
                    buffer,
                    slab_id=slab_id,
                )
                allocated += spec.expert_record_bytes
            transient: list[_PhysicalSlot] = []
            for slot_index in range(plan.transient_slots):
                label = f"global-transient-{slot_index}"
                buffer = self._allocate_buffer(label)
                transient.append(_PhysicalSlot(label, buffer))
                allocated += spec.expert_record_bytes
            self._transient = tuple(transient)
        except Exception:
            self._persistent.clear()
            self._transient = ()
            raise
        expected = plan.persistent_cache_bytes + plan.transient_bytes
        if allocated != expected:
            raise ExpertSlotError(
                f"allocated slot bytes {allocated} do not match memory plan {expected}"
            )
        self.allocated_bytes = allocated
        workers = max(1, max_inflight_io_bytes // spec.expert_record_bytes)
        workers = min(workers, plan.transient_slots)
        self.max_inflight_io_bytes = workers * spec.expert_record_bytes
        self._reader_pool_telemetry = (
            PoolOccupancy(worker_capacity=workers) if resource_telemetry else None
        )
        self._completion_fence_telemetry = (
            PoolOccupancy(worker_capacity=1) if resource_telemetry else None
        )
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mtplx-expert-io",
        )
        self._completion_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mtplx-slot-fence",
        )
        self._completion_error_lock = threading.Lock()
        self._completion_error: BaseException | None = None
        self._cleanup_error: BaseException | None = None

    @staticmethod
    def _run_tracked(
        telemetry: PoolOccupancy,
        units: int,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        telemetry.started(units)
        try:
            return callback(*args, **kwargs)
        finally:
            telemetry.completed(units)

    def _submit_tracked(
        self,
        executor: ThreadPoolExecutor,
        telemetry: PoolOccupancy,
        units: int,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        telemetry.submitted(units)
        try:
            return executor.submit(
                self._run_tracked,
                telemetry,
                units,
                callback,
                *args,
                **kwargs,
            )
        except BaseException:
            telemetry.rejected(units)
            raise

    def _pipeline_call(
        self,
        route: ExpertPipelineRoute,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Publish optional diagnostics without changing slot outcomes."""

        try:
            getattr(route, method)(*args, **kwargs)
        except Exception:
            self._mark_pipeline_incomplete(route)

    def _mark_pipeline_incomplete(self, route: ExpertPipelineRoute) -> None:
        ledger = self._pipeline_ledger
        if ledger is not None:
            try:
                ledger.mark_incomplete(phase=route.phase)
            except Exception:
                pass

    @staticmethod
    def _diagnostic_monotonic_ns() -> int | None:
        try:
            value = time.monotonic_ns()
        except Exception:
            return None
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    @staticmethod
    def _diagnostic_thread_time_ns() -> int | None:
        try:
            value = time.thread_time_ns()
        except Exception:
            return None
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    def _pipeline_thread_cpu_elapsed(
        self,
        route: ExpertPipelineRoute,
        started_ns: int | None,
    ) -> int:
        if started_ns is None:
            return 0
        completed_ns = self._diagnostic_thread_time_ns()
        if completed_ns is None or completed_ns < started_ns:
            self._mark_pipeline_incomplete(route)
            return 0
        return completed_ns - started_ns

    def _run_pipeline_reader(
        self,
        telemetry: PoolOccupancy | None,
        units: int,
        route: ExpertPipelineRoute,
        experts: tuple[int, ...],
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if telemetry is not None:
            telemetry.started(units)
        self._pipeline_call(route, "reader_started", experts)
        cpu_started_ns = self._diagnostic_thread_time_ns()
        if cpu_started_ns is None:
            self._mark_pipeline_incomplete(route)
        try:
            result = callback(*args, **kwargs)
        except BaseException:
            self._pipeline_call(
                route,
                "reader_failed",
                experts,
                thread_cpu_ns=self._pipeline_thread_cpu_elapsed(
                    route,
                    cpu_started_ns,
                ),
            )
            raise
        else:
            self._pipeline_call(
                route,
                "reader_completed",
                experts,
                thread_cpu_ns=self._pipeline_thread_cpu_elapsed(
                    route,
                    cpu_started_ns,
                ),
            )
            return result
        finally:
            if telemetry is not None:
                telemetry.completed(units)

    def _submit_pipeline_reader(
        self,
        route: ExpertPipelineRoute,
        experts: tuple[int, ...],
        units: int,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        telemetry = self._reader_pool_telemetry
        if telemetry is not None:
            telemetry.submitted(units)
        self._pipeline_call(route, "submission_attempted", experts)
        try:
            future = self._executor.submit(
                self._run_pipeline_reader,
                telemetry,
                units,
                route,
                experts,
                callback,
                *args,
                **kwargs,
            )
        except BaseException:
            if telemetry is not None:
                telemetry.rejected(units)
            self._pipeline_call(route, "submission_rejected", experts)
            raise
        self._pipeline_call(route, "submission_accepted", experts)
        return future

    def _submit_completion_fence(
        self,
        completion_waiter: Callable[[], None],
        on_complete: Callable[[], None],
        *,
        slot_count: int,
    ) -> Future[None] | None:
        self.metrics.update(
            completion_fences=1,
            completion_fence_slots=slot_count,
        )

        def wait_and_release() -> None:
            waiter_error: BaseException | None = None
            cleanup_error: BaseException | None = None
            try:
                completion_waiter()
            except BaseException as exc:
                self.metrics.update(completion_fence_failures=1)
                self._record_completion_error(exc)
                waiter_error = exc
            try:
                on_complete()
            except BaseException as exc:
                self.metrics.update(completion_fence_failures=1)
                self._record_cleanup_error(exc)
                cleanup_error = exc
            if waiter_error is not None:
                raise waiter_error
            if cleanup_error is not None:
                raise cleanup_error

        try:
            if self._completion_fence_telemetry is None:
                return self._completion_executor.submit(wait_and_release)
            return self._submit_tracked(
                self._completion_executor,
                self._completion_fence_telemetry,
                slot_count,
                wait_and_release,
            )
        except RuntimeError:
            # Shutdown races or stripped-down runtimes fail closed: wait on the
            # caller before allowing any generation to be overwritten.
            self.metrics.update(completion_fence_fallbacks=1)
            if self._completion_fence_telemetry is None:
                wait_and_release()
            else:
                self._completion_fence_telemetry.submitted(slot_count)
                self._run_tracked(
                    self._completion_fence_telemetry,
                    slot_count,
                    wait_and_release,
                )
            return None

    def _record_completion_error(self, error: BaseException) -> None:
        with self._completion_error_lock:
            if self._completion_error is None:
                self._completion_error = error

    def _record_cleanup_error(self, error: BaseException) -> None:
        with self._completion_error_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error

    def _raise_completion_error_locked(
        self,
        *,
        policy_rollback_safe: bool,
    ) -> None:
        error = self._completion_error
        if error is not None:
            raise ExpertCompletionFenceError(
                "an asynchronous expert completion fence failed",
                policy_rollback_safe=policy_rollback_safe,
            ) from error
        if self._cleanup_error is not None:
            raise ExpertSlotError("expert slot cleanup failed") from self._cleanup_error

    def _raise_completion_error(
        self,
        *,
        policy_rollback_safe: bool = True,
    ) -> None:
        with self._completion_error_lock:
            self._raise_completion_error_locked(
                policy_rollback_safe=policy_rollback_safe
            )

    def raise_if_unhealthy(self) -> None:
        """Reject new policy or slot work after a terminal fence failure."""

        self._retry_cleanup_owners()
        self._raise_completion_error()

    def commit_if_healthy(self, callback: Callable[[], None]) -> None:
        """Run one host-only policy commit atomically with the health check."""

        with self._completion_error_lock:
            self._raise_completion_error_locked(policy_rollback_safe=False)
            callback()

    def _drain_completion_fences(self) -> None:
        """Wait for completion tasks queued before this diagnostic snapshot."""

        try:
            barrier = self._completion_executor.submit(lambda: None)
        except RuntimeError:
            return
        barrier.result()

    def _allocate_buffer(self, label: str) -> Any:
        buffer = self._allocator(self.spec.expert_record_bytes, label)
        return self._validate_allocated_buffer(buffer, label)

    def _validate_allocated_buffer(self, buffer: Any, label: str) -> Any:
        direct_nbytes = getattr(buffer, "nbytes", None)
        record_views = getattr(buffer, "record_views", None)
        if callable(record_views):
            if int(direct_nbytes or -1) != self.spec.expert_record_bytes:
                raise ExpertSlotError(
                    f"allocator returned invalid composite buffer for {label}: "
                    f"bytes={direct_nbytes}"
                )
            return buffer
        try:
            view = memoryview(buffer)
        except TypeError as exc:
            raise ExpertSlotError(
                f"allocator returned a non-buffer for {label}"
            ) from exc
        if (
            view.readonly
            or not view.c_contiguous
            or view.nbytes != self.spec.expert_record_bytes
        ):
            raise ExpertSlotError(
                f"allocator returned invalid buffer for {label}: "
                f"readonly={view.readonly}, contiguous={view.c_contiguous}, bytes={view.nbytes}"
            )
        return buffer

    def _physical(self, layer: int, logical_slot: int) -> _PhysicalSlot:
        if layer not in self.spec.routed_layer_indices:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer")
        if logical_slot < self._persistent_route_capacity:
            try:
                key_layer = -1 if self.cache_scope == "global" else layer
                slot = self._persistent[(key_layer, logical_slot)]
            except KeyError as exc:
                raise ExpertSlotError(
                    "persistent slot is outside the memory plan"
                ) from exc
            if slot.slab_id is not None:
                with self._slab_lock:
                    slab = self._slabs[slot.slab_id]
                    if slab.state is not ExpertSlabState.ACTIVE:
                        raise ExpertSlotError(
                            f"expert slab {slab.slab_id} is {slab.state.value}"
                        )
            return slot
        transient_index = logical_slot - self._persistent_route_capacity
        if not 0 <= transient_index < len(self._transient):
            raise ExpertSlotError("transient slot is outside the memory plan")
        return self._transient[transient_index]

    @staticmethod
    def _remaining(deadline_ns: int | None) -> float | None:
        if deadline_ns is None:
            return None
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            raise TimeoutError("expert slot deadline exceeded")
        return remaining_ns / 1e9

    def _prepare_load(
        self,
        layer: int,
        load: SlotLoad,
        *,
        deadline_ns: int | None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> tuple[_PhysicalSlot, int, bool, _PreparedSlotState | None]:
        slot = self._physical(layer, load.slot)
        block_observations: list[tuple[str, int]] | None = (
            [] if pipeline_route is not None else None
        )
        block_clock_failed = False
        try:
            with slot.condition:
                while True:
                    if self._closed or slot.state is ExpertSlotState.CLOSED:
                        raise ExpertSlotError("expert slot pool is closed")
                    if self._closing:
                        raise ExpertSlotError("expert slot pool is closing")
                    if slot.buffer is None:
                        raise ExpertSlotError("expert slot has no physical buffer")
                    self._raise_completion_error()
                    if (
                        slot.layer == layer
                        and slot.expert == load.expert
                        and slot.state
                        in {ExpertSlotState.LOADING, ExpertSlotState.READY}
                        and (
                            load.generation is None
                            or slot.generation == load.generation
                        )
                    ):
                        if slot.state is ExpertSlotState.LOADING:
                            self.metrics.update(deduplicated_loads=1)
                        else:
                            self.metrics.update(ready_hits=1)
                        return slot, slot.generation, False, None
                    if slot.state is ExpertSlotState.LOADING:
                        self.metrics.update(load_waits=1)
                        wait_started_ns = (
                            self._diagnostic_monotonic_ns()
                            if block_observations is not None
                            else None
                        )
                        if block_observations is not None and wait_started_ns is None:
                            block_clock_failed = True
                        slot.condition.wait(self._remaining(deadline_ns))
                        if block_observations is not None:
                            wait_completed_ns = self._diagnostic_monotonic_ns()
                            if (
                                wait_started_ns is None
                                or wait_completed_ns is None
                                or wait_completed_ns < wait_started_ns
                            ):
                                block_clock_failed = True
                            else:
                                block_observations.append(
                                    (
                                        "slot_loading",
                                        wait_completed_ns - wait_started_ns,
                                    )
                                )
                        continue
                    if slot.pins:
                        self.metrics.update(pin_waits=1)
                        wait_started_ns = (
                            self._diagnostic_monotonic_ns()
                            if block_observations is not None
                            else None
                        )
                        if block_observations is not None and wait_started_ns is None:
                            block_clock_failed = True
                        slot.condition.wait(self._remaining(deadline_ns))
                        if block_observations is not None:
                            wait_completed_ns = self._diagnostic_monotonic_ns()
                            if (
                                wait_started_ns is None
                                or wait_completed_ns is None
                                or wait_completed_ns < wait_started_ns
                            ):
                                block_clock_failed = True
                            else:
                                block_observations.append(
                                    ("pin_held", wait_completed_ns - wait_started_ns)
                                )
                        self._raise_completion_error()
                        continue
                    self._raise_completion_error()
                    previous_state = slot.state
                    previous_layer = slot.layer
                    previous_expert = slot.expert
                    previous_generation = slot.generation
                    previous_digest = slot.digest
                    previous_error = slot.error
                    replacing = slot.state is ExpertSlotState.READY
                    if load.generation is None:
                        slot.generation += 1
                    else:
                        if load.generation <= slot.generation:
                            raise ExpertSlotError(
                                "stale global cache generation requested for slot"
                            )
                        slot.generation = load.generation
                    previous = _PreparedSlotState(
                        slot=slot,
                        owned_generation=slot.generation,
                        state=previous_state,
                        layer=previous_layer,
                        expert=previous_expert,
                        generation=previous_generation,
                        digest=previous_digest,
                        error=previous_error,
                    )
                    slot.layer = layer
                    slot.expert = load.expert
                    slot.state = ExpertSlotState.LOADING
                    slot.digest = None
                    slot.error = None
                    if replacing:
                        self.metrics.update(generation_replacements=1)
                    self.metrics.update(owned_loads=1)
                    return slot, slot.generation, True, previous
        finally:
            if pipeline_route is not None:
                if block_clock_failed:
                    self._mark_pipeline_incomplete(pipeline_route)
                if block_observations:
                    for reason, elapsed_ns in block_observations:
                        self._pipeline_call(
                            pipeline_route,
                            "observe_block",
                            load.expert,
                            reason,
                            elapsed_ns=elapsed_ns,
                        )

    @staticmethod
    def _restore_prepared_slots(prepared: Iterable[_PreparedSlotState]) -> None:
        for previous in reversed(tuple(prepared)):
            slot = previous.slot
            with slot.condition:
                if (
                    slot.state is not ExpertSlotState.LOADING
                    or slot.generation != previous.owned_generation
                ):
                    continue
                slot.state = previous.state
                slot.layer = previous.layer
                slot.expert = previous.expert
                slot.generation = previous.generation
                slot.digest = previous.digest
                slot.error = previous.error
                slot.condition.notify_all()

    def _fill(
        self,
        slot: _PhysicalSlot,
        generation: int,
        record: ExpertRecord,
        *,
        cancel_event: Any,
        deadline_ns: int | None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> None:
        try:
            read_started = time.monotonic_ns()
            if pipeline_route is None:
                digest = self.reader.read_record_into(
                    self.manifest,
                    record,
                    slot.buffer,
                    prefer_sidecar=self.prefer_sidecar,
                    verify_hash=self.verify_hashes,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
            else:
                digest = self.reader.read_record_into(
                    self.manifest,
                    record,
                    slot.buffer,
                    prefer_sidecar=self.prefer_sidecar,
                    verify_hash=self.verify_hashes,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    pipeline_phase=pipeline_route.phase,
                )
            self.metrics.observe_physical_read(
                record.layer,
                records=1,
                elapsed_ns=time.monotonic_ns() - read_started,
            )
        except BaseException as exc:
            with slot.condition:
                if (
                    slot.generation == generation
                    and slot.state is ExpertSlotState.LOADING
                ):
                    slot.state = ExpertSlotState.FAILED
                    slot.error = exc
                    slot.condition.notify_all()
            self.metrics.update(load_failures=1)
            raise
        with slot.condition:
            if (
                slot.generation != generation
                or slot.state is not ExpertSlotState.LOADING
            ):
                raise ExpertSlotError("slot generation changed during an expert read")
            slot.digest = digest
            slot.state = ExpertSlotState.READY
            slot.error = None
            slot.condition.notify_all()
        if pipeline_route is not None:
            self._pipeline_call(pipeline_route, "record_ready", record.expert)

    def _fill_batch(
        self,
        owned: tuple[tuple[_PhysicalSlot, int, ExpertRecord], ...],
        *,
        cancel_event: Any,
        deadline_ns: int | None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> None:
        try:
            read_started = time.monotonic_ns()
            items = tuple((record, slot.buffer) for slot, _generation, record in owned)
            if pipeline_route is None:
                digests = self.reader.read_component_records_into(
                    self.manifest,
                    items,
                    verify_hash=self.verify_hashes,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
            else:
                digests = self.reader.read_component_records_into(
                    self.manifest,
                    items,
                    verify_hash=self.verify_hashes,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    pipeline_phase=pipeline_route.phase,
                )
            elapsed_ns = time.monotonic_ns() - read_started
            records_by_layer = Counter(
                record.layer for _slot, _generation, record in owned
            )
            total_records = len(owned)
            distributed_ns = 0
            for index, (layer, records) in enumerate(sorted(records_by_layer.items())):
                layer_ns = (
                    elapsed_ns - distributed_ns
                    if index == len(records_by_layer) - 1
                    else elapsed_ns * records // total_records
                )
                distributed_ns += layer_ns
                self.metrics.observe_physical_read(
                    layer,
                    records=records,
                    elapsed_ns=layer_ns,
                )
        except BaseException as exc:
            for slot, generation, _record in owned:
                with slot.condition:
                    if (
                        slot.generation == generation
                        and slot.state is ExpertSlotState.LOADING
                    ):
                        slot.state = ExpertSlotState.FAILED
                        slot.error = exc
                        slot.condition.notify_all()
            self.metrics.update(load_failures=len(owned))
            raise
        for (slot, generation, record), digest in zip(owned, digests, strict=True):
            with slot.condition:
                if (
                    slot.generation != generation
                    or slot.state is not ExpertSlotState.LOADING
                ):
                    raise ExpertSlotError(
                        "slot generation changed during a batched expert read"
                    )
                slot.digest = digest
                slot.state = ExpertSlotState.READY
                slot.error = None
                slot.condition.notify_all()
            if pipeline_route is not None:
                self._pipeline_call(
                    pipeline_route,
                    "record_ready",
                    record.expert,
                )

    def _can_batch_component_sidecar(
        self,
        plan: RoutePlan,
        owned: list[tuple[_PhysicalSlot, int, ExpertRecord]],
    ) -> bool:
        if (
            plan.phase.value != "prefill"
            or not self.prefer_sidecar
            or self.manifest.sidecar is None
            or len(owned) < 2
            or not all(
                callable(getattr(slot.buffer, "record_views", None))
                for slot, _generation, _record in owned
            )
        ):
            return False
        ordered = sorted(
            (record for _slot, _generation, record in owned),
            key=lambda record: int(record.sidecar_offset or 0),
        )
        return all(
            int(left.sidecar_offset or 0) + int(left.sidecar_length or 0)
            == int(right.sidecar_offset or 0)
            for left, right in zip(ordered, ordered[1:], strict=False)
        )

    def _wait_ready(
        self,
        slot: _PhysicalSlot,
        *,
        layer: int,
        expert: int,
        generation: int,
        deadline_ns: int | None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> None:
        block_observations: list[int] | None = (
            [] if pipeline_route is not None else None
        )
        block_clock_failed = False
        try:
            with slot.condition:
                while slot.state is ExpertSlotState.LOADING:
                    self.metrics.update(load_waits=1)
                    wait_started_ns = (
                        self._diagnostic_monotonic_ns()
                        if block_observations is not None
                        else None
                    )
                    if block_observations is not None and wait_started_ns is None:
                        block_clock_failed = True
                    slot.condition.wait(self._remaining(deadline_ns))
                    if block_observations is not None:
                        wait_completed_ns = self._diagnostic_monotonic_ns()
                        if (
                            wait_started_ns is None
                            or wait_completed_ns is None
                            or wait_completed_ns < wait_started_ns
                        ):
                            block_clock_failed = True
                        else:
                            block_observations.append(
                                wait_completed_ns - wait_started_ns
                            )
                if (
                    slot.state is not ExpertSlotState.READY
                    or slot.layer != layer
                    or slot.expert != expert
                    or slot.generation != generation
                ):
                    if slot.error is not None:
                        raise ExpertSlotError(
                            f"expert load failed for ({layer}, {expert}): {slot.error}"
                        ) from slot.error
                    raise ExpertSlotError(
                        "expert slot did not reach the requested generation"
                    )
        finally:
            if pipeline_route is not None:
                if block_clock_failed:
                    self._mark_pipeline_incomplete(pipeline_route)
                if block_observations:
                    for elapsed_ns in block_observations:
                        self._pipeline_call(
                            pipeline_route,
                            "observe_block",
                            expert,
                            "slot_loading",
                            elapsed_ns=elapsed_ns,
                        )

    def ensure_route(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
        io_admission: RouteIOAdmission | None = None,
        route_admitted: Callable[[], None] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> ReadyRoute:
        """Load all misses, validate mappings, and pin the route's slots."""

        self._raise_completion_error()
        try:
            layer_lock = self._ensure_locks[layer]
        except KeyError as exc:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer") from exc
        with layer_lock:
            self._raise_completion_error()
            if pipeline_route is None:
                if io_admission is None:
                    if route_admitted is None:
                        return self._ensure_route_locked(
                            layer,
                            plan,
                            cancel_event=cancel_event,
                            deadline_ns=deadline_ns,
                        )
                    return self._ensure_route_locked(
                        layer,
                        plan,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        route_admitted=route_admitted,
                    )
                if route_admitted is None:
                    return self._ensure_route_locked(
                        layer,
                        plan,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        io_admission=io_admission,
                    )
                return self._ensure_route_locked(
                    layer,
                    plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    io_admission=io_admission,
                    route_admitted=route_admitted,
                )
            if io_admission is None:
                if route_admitted is None:
                    return self._ensure_route_locked(
                        layer,
                        plan,
                        cancel_event=cancel_event,
                        deadline_ns=deadline_ns,
                        pipeline_route=pipeline_route,
                    )
                return self._ensure_route_locked(
                    layer,
                    plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    route_admitted=route_admitted,
                    pipeline_route=pipeline_route,
                )
            if route_admitted is None:
                return self._ensure_route_locked(
                    layer,
                    plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    io_admission=io_admission,
                    pipeline_route=pipeline_route,
                )
            return self._ensure_route_locked(
                layer,
                plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                io_admission=io_admission,
                route_admitted=route_admitted,
                pipeline_route=pipeline_route,
            )

    def retain_split_lifecycle(self) -> _RouteLifecycle:
        """Keep close from finalizing slots during multi-part commit/rollback."""

        self._raise_completion_error()
        claim = _RouteReleaseClaim()
        lifecycle = _RouteLifecycle(self, claim)
        with self._lifecycle:
            if self._closed:
                raise ExpertSlotError("expert slot pool is closed")
            if self._closing:
                raise ExpertSlotError("expert slot pool is closing")
            self.metrics.admit_route(claim)
        return lifecycle

    def retain_admitted_split_lifecycle(self) -> _RouteLifecycle:
        """Retain rollback lifetime after this worker owns an active route."""

        claim = _RouteReleaseClaim()
        lifecycle = _RouteLifecycle(self, claim)
        with self._lifecycle:
            if self._closed:
                raise ExpertSlotError("expert slot pool is closed")
            self.metrics.admit_route(claim)
        return lifecycle

    def ensure_route_part(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
        io_admission: RouteIOAdmission | None = None,
        route_admitted: Callable[[], None] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> ReadyRoute:
        """Load one disjoint part of a runtime-locked route transaction.

        ``ExpertStreamingRuntime`` holds its layer transaction lock while all
        parts are active. Each part owns distinct logical slots, so individual
        slot conditions provide the remaining synchronization without forcing
        completed records to wait behind the slowest read in the layer.
        """

        self._raise_completion_error()
        if layer not in self._ensure_locks:
            raise ExpertSlotError(f"layer {layer} is not a routed model layer")
        if pipeline_route is None:
            if route_admitted is None:
                return self._ensure_route_locked(
                    layer,
                    plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    io_admission=io_admission,
                )
            return self._ensure_route_locked(
                layer,
                plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                io_admission=io_admission,
                route_admitted=route_admitted,
            )
        if route_admitted is None:
            return self._ensure_route_locked(
                layer,
                plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                io_admission=io_admission,
                pipeline_route=pipeline_route,
            )
        return self._ensure_route_locked(
            layer,
            plan,
            cancel_event=cancel_event,
            deadline_ns=deadline_ns,
            io_admission=io_admission,
            route_admitted=route_admitted,
            pipeline_route=pipeline_route,
        )

    def _ensure_route_locked(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
        io_admission: RouteIOAdmission | None = None,
        route_admitted: Callable[[], None] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> ReadyRoute:
        self._raise_completion_error()
        if len(plan.experts) != len(plan.slots):
            raise ExpertSlotError("route experts and slots differ in length")
        lifecycle_claim = _RouteReleaseClaim()
        lifecycle_owner = _RouteLifecycle(self, lifecycle_claim)
        setup_pins: dict[int, tuple[_PhysicalSlot, _SlotPinClaim]] = {}
        setup_owner = _FailedRouteSetupOwner(self, lifecycle_owner, setup_pins)
        with self._lifecycle:
            if self._closed:
                raise ExpertSlotError("expert slot pool is closed")
            if self._closing:
                raise ExpertSlotError("expert slot pool is closing")
            self.metrics.admit_route(lifecycle_claim)
        try:
            if pipeline_route is None:
                return self._ensure_route_owned(
                    layer,
                    plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                    io_admission=io_admission,
                    route_admitted=route_admitted,
                    lifecycle_claim=lifecycle_claim,
                    setup_pins=setup_pins,
                )
            return self._ensure_route_owned(
                layer,
                plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                io_admission=io_admission,
                route_admitted=route_admitted,
                pipeline_route=pipeline_route,
                lifecycle_claim=lifecycle_claim,
                setup_pins=setup_pins,
            )
        except BaseException:
            self._cleanup_failed_route_setup(setup_owner)
            raise

    def _cleanup_failed_route_setup(
        self,
        setup_owner: _FailedRouteSetupOwner,
    ) -> None:
        for _attempt in range(2):
            try:
                setup_owner.release()
            except BaseException as exc:
                self._record_cleanup_error(exc)
                if setup_owner.terminal:
                    return
            else:
                return
        with self._lifecycle:
            with self._cleanup_owner_lock:
                if setup_owner not in self._cleanup_owners:
                    self._cleanup_owners.append(setup_owner)
            self._lifecycle.notify_all()

    def _retry_cleanup_owners(self) -> None:
        with self._cleanup_owner_lock:
            owners = tuple(self._cleanup_owners)
        for owner in owners:
            remove = False
            try:
                owner.release()
            except BaseException as exc:
                self._record_cleanup_error(exc)
                remove = owner.terminal
            else:
                remove = True
            if remove:
                with self._cleanup_owner_lock:
                    if owner in self._cleanup_owners:
                        self._cleanup_owners.remove(owner)

    def _has_cleanup_owners(self) -> bool:
        with self._cleanup_owner_lock:
            return bool(self._cleanup_owners)

    def _ensure_route_owned(
        self,
        layer: int,
        plan: RoutePlan,
        *,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
        io_admission: RouteIOAdmission | None = None,
        route_admitted: Callable[[], None] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
        lifecycle_claim: _RouteReleaseClaim,
        setup_pins: dict[int, tuple[_PhysicalSlot, _SlotPinClaim]],
    ) -> ReadyRoute:
        setup_hook = self._route_setup_test_hook
        if route_admitted is not None:
            route_admitted()
        if setup_hook is not None:
            setup_hook("after_callback")
        self.metrics.update(ensure_calls=1, load_requests=len(plan.loads))
        if setup_hook is not None:
            setup_hook("after_metrics")
        internal_cancel = threading.Event()
        if setup_hook is not None:
            setup_hook("after_cancel_event")
        combined_cancel = _CombinedCancel(cancel_event, internal_cancel)
        if setup_hook is not None:
            setup_hook("after_combined_cancel")
        prepared: dict[tuple[int, int], tuple[_PhysicalSlot, int]] = {}
        prepared_states: list[_PreparedSlotState] = []
        futures: list[Future[None]] = []
        owned_loads: list[tuple[_PhysicalSlot, int, ExpertRecord]] = []
        submitted_loads: set[tuple[int, int]] = set()
        nonowned_loads: set[tuple[int, int]] | None = (
            set() if pipeline_route is not None else None
        )
        if setup_hook is not None:
            setup_hook("after_containers")
        admission = io_admission if io_admission is not None else RouteIOAdmission()
        if setup_hook is not None:
            setup_hook("after_io_admission")
        try:
            try:
                for load in plan.loads:
                    try:
                        record = self._record_map[(layer, load.expert)]
                    except KeyError as exc:
                        raise ExpertSlotError(
                            f"manifest has no expert record ({layer}, {load.expert})"
                        ) from exc
                    if pipeline_route is None:
                        slot, generation, owner, previous = self._prepare_load(
                            layer,
                            load,
                            deadline_ns=deadline_ns,
                        )
                    else:
                        slot, generation, owner, previous = self._prepare_load(
                            layer,
                            load,
                            deadline_ns=deadline_ns,
                            pipeline_route=pipeline_route,
                        )
                    prepared[(load.expert, load.slot)] = (slot, generation)
                    if owner:
                        owned_loads.append((slot, generation, record))
                        assert previous is not None
                        prepared_states.append(previous)
                    elif nonowned_loads is not None:
                        nonowned_loads.add((load.expert, load.slot))
                use_batch = self._can_batch_component_sidecar(plan, owned_loads)
                # Completion failure recording uses this same lock.  Keeping
                # it through every submission makes the rollback boundary
                # exact: either no read was accepted, or policy restoration is
                # conservatively unsafe because at least one read may replace
                # a victim generation.
                with self._completion_error_lock:
                    self._raise_completion_error_locked(policy_rollback_safe=True)
                    if use_batch:
                        if pipeline_route is not None:
                            experts = tuple(
                                record.expert
                                for _slot, _generation, record in owned_loads
                            )
                            future = self._submit_pipeline_reader(
                                pipeline_route,
                                experts,
                                sum(
                                    record.logical_bytes for _, _, record in owned_loads
                                ),
                                self._fill_batch,
                                tuple(owned_loads),
                                cancel_event=combined_cancel,
                                deadline_ns=deadline_ns,
                                pipeline_route=pipeline_route,
                            )
                        elif self._reader_pool_telemetry is None:
                            future = self._executor.submit(
                                self._fill_batch,
                                tuple(owned_loads),
                                cancel_event=combined_cancel,
                                deadline_ns=deadline_ns,
                            )
                        else:
                            future = self._submit_tracked(
                                self._executor,
                                self._reader_pool_telemetry,
                                sum(
                                    record.logical_bytes for _, _, record in owned_loads
                                ),
                                self._fill_batch,
                                tuple(owned_loads),
                                cancel_event=combined_cancel,
                                deadline_ns=deadline_ns,
                            )
                        admission.mark_accepted()
                        futures.append(future)
                        submitted_loads.update(
                            (id(slot), generation)
                            for slot, generation, _record in owned_loads
                        )
                    else:
                        for slot, generation, record in owned_loads:
                            if pipeline_route is not None:
                                future = self._submit_pipeline_reader(
                                    pipeline_route,
                                    (record.expert,),
                                    record.logical_bytes,
                                    self._fill,
                                    slot,
                                    generation,
                                    record,
                                    cancel_event=combined_cancel,
                                    deadline_ns=deadline_ns,
                                    pipeline_route=pipeline_route,
                                )
                            elif self._reader_pool_telemetry is None:
                                future = self._executor.submit(
                                    self._fill,
                                    slot,
                                    generation,
                                    record,
                                    cancel_event=combined_cancel,
                                    deadline_ns=deadline_ns,
                                )
                            else:
                                future = self._submit_tracked(
                                    self._executor,
                                    self._reader_pool_telemetry,
                                    record.logical_bytes,
                                    self._fill,
                                    slot,
                                    generation,
                                    record,
                                    cancel_event=combined_cancel,
                                    deadline_ns=deadline_ns,
                                )
                            admission.mark_accepted()
                            futures.append(future)
                            submitted_loads.add((id(slot), generation))
            except BaseException:
                self._restore_prepared_slots(
                    previous
                    for previous in prepared_states
                    if (id(previous.slot), previous.owned_generation)
                    not in submitted_loads
                )
                if futures:
                    internal_cancel.set()
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
                raise
            future_error: BaseException | None = None
            for future in futures:
                try:
                    future.result(timeout=self._remaining(deadline_ns))
                except BaseException as exc:
                    internal_cancel.set()
                    if future_error is None:
                        future_error = exc
            if future_error is not None:
                for future in futures:
                    if not future.done():
                        try:
                            future.result()
                        except BaseException:
                            pass
                raise future_error

            bindings: list[ExpertSlotBinding] = []
            unique_pins = setup_pins
            satisfied_experts: set[int] | None = (
                set() if pipeline_route is not None else None
            )
            try:
                plan_generations = (
                    plan.generations
                    if plan.generations
                    else (None,) * len(plan.experts)
                )
                if len(plan_generations) != len(plan.experts):
                    raise ExpertSlotError(
                        "route experts and generations differ in length"
                    )
                for expert, logical_slot, expected_generation in zip(
                    plan.experts,
                    plan.slots,
                    plan_generations,
                    strict=True,
                ):
                    slot = self._physical(layer, logical_slot)
                    with slot.condition:
                        current_generation = slot.generation
                    generation = prepared.get(
                        (expert, logical_slot), (slot, current_generation)
                    )[1]
                    if expected_generation is not None:
                        generation = expected_generation
                    if pipeline_route is None:
                        self._wait_ready(
                            slot,
                            layer=layer,
                            expert=expert,
                            generation=generation,
                            deadline_ns=deadline_ns,
                        )
                    else:
                        self._wait_ready(
                            slot,
                            layer=layer,
                            expert=expert,
                            generation=generation,
                            deadline_ns=deadline_ns,
                            pipeline_route=pipeline_route,
                        )
                    if (
                        pipeline_route is not None
                        and nonowned_loads is not None
                        and satisfied_experts is not None
                        and (expert, logical_slot) in nonowned_loads
                        and expert not in satisfied_experts
                    ):
                        self._pipeline_call(
                            pipeline_route,
                            "satisfied_without_submit",
                            (expert,),
                        )
                        satisfied_experts.add(expert)
                    try:
                        record = self._record_map[(layer, expert)]
                    except KeyError as exc:
                        raise ExpertSlotError(
                            f"manifest has no expert record ({layer}, {expert})"
                        ) from exc
                    with slot.condition:
                        if admission.any_accepted:
                            self._raise_completion_error(policy_rollback_safe=False)
                        else:
                            self._raise_completion_error()
                        if id(slot) not in unique_pins:
                            claim = _SlotPinClaim()
                            unique_pins[id(slot)] = (slot, claim)
                            slot.pin_claims.append(claim)
                            slot.pins = sum(item.active for item in slot.pin_claims)
                        bindings.append(
                            ExpertSlotBinding(
                                layer=layer,
                                expert=expert,
                                logical_slot=logical_slot,
                                generation=generation,
                                record=record,
                                buffer=slot.buffer,
                            )
                        )
                if admission.any_accepted:
                    self._raise_completion_error(policy_rollback_safe=False)
                else:
                    self._raise_completion_error()
                if setup_hook is not None:
                    setup_hook("after_pin_materialization")
            except BaseException:
                for slot, claim in unique_pins.values():
                    with slot.condition:
                        claim.active = False
                        slot.pins = sum(item.active for item in slot.pin_claims)
                        if claim in slot.pin_claims:
                            slot.pin_claims.remove(claim)
                        slot.condition.notify_all()
                raise
        except (ExpertManifestError, ExpertIOError) as exc:
            internal_cancel.set()
            raise ExpertSlotError(str(exc)) from exc
        except BaseException:
            internal_cancel.set()
            raise

        binding_tuple = tuple(bindings)
        if setup_hook is not None:
            setup_hook("after_bindings_tuple")
        pin_tuple = tuple(unique_pins.values())
        if setup_hook is not None:
            setup_hook("after_pins_tuple")
        ready = ReadyRoute(
            self,
            plan,
            binding_tuple,
            pin_tuple,
            lifecycle_claim,
        )
        if pipeline_route is not None:
            for expert in dict.fromkeys(load.expert for load in plan.loads):
                self._pipeline_call(pipeline_route, "record_runnable", expert)
        if setup_hook is not None:
            setup_hook("before_ownership_transfer")
        return ready

    def _route_released(self, claim: _RouteReleaseClaim) -> None:
        with self._lifecycle:
            self.metrics.release_route(claim)
            self._lifecycle.notify_all()

    def _require_slab_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise ExpertSlotError(
                "expert slab allocation and destruction require the owner thread"
            )

    def _normalize_slab_ids(self, slab_ids: Iterable[int]) -> tuple[int, ...]:
        normalized = tuple(slab_ids)
        if not normalized:
            raise ValueError("at least one expert slab is required")
        if any(
            isinstance(slab_id, bool)
            or not isinstance(slab_id, int)
            or slab_id not in self._slabs
            for slab_id in normalized
        ):
            raise ValueError("unknown expert slab id")
        if len(set(normalized)) != len(normalized):
            raise ValueError("expert slab ids must be unique")
        return normalized

    def slab_layout(self) -> dict[int, tuple[int, ...]]:
        with self._slab_lock:
            return {slab_id: slab.slot_ids for slab_id, slab in self._slabs.items()}

    def slot_ids_for_slab(self, slab_id: int) -> tuple[int, ...]:
        with self._slab_lock:
            try:
                return self._slabs[slab_id].slot_ids
            except KeyError as exc:
                raise ValueError("unknown expert slab id") from exc

    def prepare_slab_reclaim(
        self,
        slab_ids: Iterable[int],
        *,
        requested_bytes: int | None = None,
        deadline_ns: int | None = None,
    ) -> ExpertSlabReclaimTicket:
        with self._close_lock:
            return self._prepare_slab_reclaim_locked(
                slab_ids,
                requested_bytes=requested_bytes,
                deadline_ns=deadline_ns,
            )

    def _prepare_slab_reclaim_locked(
        self,
        slab_ids: Iterable[int],
        *,
        requested_bytes: int | None,
        deadline_ns: int | None,
    ) -> ExpertSlabReclaimTicket:
        """Drain selected completion owners without touching unrelated slabs."""

        self._raise_completion_error()
        if self._closed or self._closing:
            raise ExpertSlotError("expert slot pool is closing")
        candidates = self._normalize_slab_ids(slab_ids)
        if requested_bytes is not None and (
            isinstance(requested_bytes, bool)
            or not isinstance(requested_bytes, int)
            or requested_bytes <= 0
        ):
            raise ValueError("requested_bytes must be a positive integer")
        selected: list[int] = []
        selected_bytes = 0
        with self._slab_lock:
            for slab_id in candidates:
                slab = self._slabs[slab_id]
                if slab.state is not ExpertSlabState.ACTIVE:
                    raise ExpertSlotError(
                        f"expert slab {slab_id} is {slab.state.value}"
                    )
                if slab_id in self._slab_release_failures:
                    raise ExpertSlotError("expert slab has a prior release failure")
                selected.append(slab_id)
                selected_bytes += slab.physical_bytes
                if requested_bytes is not None and selected_bytes >= requested_bytes:
                    break
            if requested_bytes is not None and selected_bytes < requested_bytes:
                raise ExpertSlotError("candidate slabs cannot satisfy requested bytes")

            selected_slots = tuple(
                (slot_id, self._persistent[(-1, slot_id)])
                for slab_id in selected
                for slot_id in self._slabs[slab_id].slot_ids
            )
            for _slot_id, slot in selected_slots:
                with slot.condition:
                    if slot.buffer is None:
                        raise ExpertSlotError(
                            "active expert slab has no physical buffer"
                        )
                    if slot.state is ExpertSlotState.LOADING:
                        raise ExpertSlotError(
                            "cannot reclaim an expert slab with a loading slot"
                        )
                    active_claims = tuple(
                        claim for claim in slot.pin_claims if claim.active
                    )
                    if slot.pins != len(active_claims):
                        raise ExpertSlotError("expert slot pin state is inconsistent")
                    if active_claims and not all(
                        claim.completion_owned for claim in active_claims
                    ):
                        raise ExpertSlotError(
                            "cannot reclaim an expert slab with a pinned slot"
                        )
            for slab_id in selected:
                self._slabs[slab_id].state = ExpertSlabState.DRAINING

        try:
            for _slot_id, slot in selected_slots:
                with slot.condition:
                    while slot.pins:
                        active_claims = tuple(
                            claim for claim in slot.pin_claims if claim.active
                        )
                        if not active_claims or not all(
                            claim.completion_owned for claim in active_claims
                        ):
                            raise ExpertSlotError(
                                "cannot reclaim an expert slab with a pinned slot"
                            )
                        slot.condition.wait(self._remaining(deadline_ns))
                    if slot.state is ExpertSlotState.LOADING:
                        raise ExpertSlotError(
                            "cannot reclaim an expert slab with a loading slot"
                        )
            self._raise_completion_error()
            snapshots: list[_SlabSlotSnapshot] = []
            for slot_id, slot in selected_slots:
                with slot.condition:
                    snapshots.append(
                        _SlabSlotSnapshot(
                            slot_id=slot_id,
                            state=slot.state,
                            layer=slot.layer,
                            expert=slot.expert,
                            generation=slot.generation,
                            digest=slot.digest,
                            error=slot.error,
                        )
                    )
        except BaseException:
            with self._slab_lock:
                for slab_id in selected:
                    slab = self._slabs[slab_id]
                    if slab.state is ExpertSlabState.DRAINING:
                        slab.state = ExpertSlabState.ACTIVE
            raise

        with self._slab_lock:
            if any(
                self._slabs[slab_id].state is not ExpertSlabState.DRAINING
                for slab_id in selected
            ):
                raise ExpertSlotError("expert slab state changed during preparation")
            self._slab_ticket_clock += 1
            ticket = ExpertSlabReclaimTicket(
                ticket_id=self._slab_ticket_clock,
                slab_ids=tuple(selected),
                slot_ids=tuple(slot_id for slot_id, _slot in selected_slots),
                physical_bytes=selected_bytes,
                incarnations=tuple(
                    self._slabs[slab_id].incarnation for slab_id in selected
                ),
                slots=tuple(snapshots),
            )
            self._slab_tickets[ticket.ticket_id] = ticket
            return ticket

    def _restore_undestroyed_ticket_slabs(
        self,
        ticket: ExpertSlabReclaimTicket,
    ) -> None:
        for slab_id in ticket.slab_ids:
            slab = self._slabs[slab_id]
            if slab.state is ExpertSlabState.DRAINING:
                slab.state = ExpertSlabState.ACTIVE
        self._slab_tickets.pop(ticket.ticket_id, None)

    @staticmethod
    def _reclaim_result(
        slab_ids: list[int],
        slot_ids: list[int],
        physical_bytes: int,
    ) -> ExpertSlabReclaimResult:
        return ExpertSlabReclaimResult(
            slab_ids=tuple(slab_ids),
            released_slot_ids=tuple(slot_ids),
            physical_bytes=physical_bytes,
        )

    def _allocator_destroyed_state(
        self,
        slab_id: int,
        error: BaseException,
    ) -> bool | None:
        if isinstance(error, ExpertSlabAllocatorError):
            return error.destroyed
        try:
            return not bool(getattr(self._allocator, "slab_is_allocated")(slab_id))
        except BaseException:
            return None

    def _finalize_destroyed_slab(self, slab: ExpertSlab) -> None:
        for slot_id in slab.slot_ids:
            slot = self._persistent[(-1, slot_id)]
            with slot.condition:
                slot.state = ExpertSlotState.EMPTY
                slot.layer = None
                slot.expert = None
                slot.digest = None
                slot.error = None
                slot.buffer = None
                slot.condition.notify_all()
        slab.state = ExpertSlabState.RELEASED
        self.allocated_bytes -= slab.physical_bytes

    def commit_slab_reclaim(
        self,
        ticket: ExpertSlabReclaimTicket,
    ) -> ExpertSlabReclaimResult:
        """Destroy prepared slabs, preserving released state after the boundary."""

        self._require_slab_owner_thread()
        self._raise_completion_error()
        with self._slab_lock:
            if self._slab_tickets.get(ticket.ticket_id) is not ticket:
                raise ExpertSlotError("expert slab reclaim ticket is not active")
            snapshots = {snapshot.slot_id: snapshot for snapshot in ticket.slots}
            try:
                for slab_id, incarnation in zip(
                    ticket.slab_ids,
                    ticket.incarnations,
                    strict=True,
                ):
                    slab = self._slabs[slab_id]
                    if (
                        slab.state is not ExpertSlabState.DRAINING
                        or slab.incarnation != incarnation
                    ):
                        raise ExpertSlotError(
                            "expert slab changed during reclaim transaction"
                        )
                    for slot_id in slab.slot_ids:
                        slot = self._persistent[(-1, slot_id)]
                        snapshot = snapshots[slot_id]
                        with slot.condition:
                            if slot.state is ExpertSlotState.LOADING:
                                raise ExpertSlotError(
                                    "cannot destroy a slab with a loading slot"
                                )
                            if slot.pins:
                                raise ExpertSlotError(
                                    "cannot destroy a slab with a pinned slot"
                                )
                            if (
                                slot.state is not snapshot.state
                                or slot.layer != snapshot.layer
                                or slot.expert != snapshot.expert
                                or slot.generation != snapshot.generation
                                or slot.digest != snapshot.digest
                                or slot.error is not snapshot.error
                            ):
                                raise ExpertSlotError(
                                    "expert slot changed during slab reclaim"
                                )
                hook = self._slab_reclaim_test_hook
                if hook is not None:
                    for slab_id in ticket.slab_ids:
                        hook("before_destroy", slab_id)
            except BaseException:
                self._restore_undestroyed_ticket_slabs(ticket)
                raise

            release_slab = getattr(self._allocator, "release_slab")
            released_slabs: list[int] = []
            released_slots: list[int] = []
            released_bytes = 0
            for slab_id in ticket.slab_ids:
                slab = self._slabs[slab_id]
                try:
                    outcome = release_slab(slab_id)
                    if (
                        not isinstance(outcome, ExpertSlabReleaseResult)
                        or not outcome.destroyed
                        or outcome.slab_id != slab_id
                        or outcome.physical_bytes != slab.physical_bytes
                    ):
                        raise RuntimeError(
                            "allocator did not confirm exact slab destruction"
                        )
                    self._finalize_destroyed_slab(slab)
                    released_bytes += slab.physical_bytes
                    released_slabs.append(slab_id)
                    released_slots.extend(slab.slot_ids)
                    if hook is not None:
                        hook("after_destroy", slab_id)
                except BaseException as exc:
                    destroyed = self._allocator_destroyed_state(slab_id, exc)
                    if destroyed is True and slab.state is not ExpertSlabState.RELEASED:
                        self._finalize_destroyed_slab(slab)
                        released_bytes += slab.physical_bytes
                        released_slabs.append(slab_id)
                        released_slots.extend(slab.slot_ids)
                    elif destroyed is None:
                        slab.state = ExpertSlabState.RELEASED
                        self._slab_ambiguous_allocations.add(slab_id)
                    if destroyed is not False:
                        self._slab_release_failures[slab_id] = exc
                    self._restore_undestroyed_ticket_slabs(ticket)
                    result = self._reclaim_result(
                        released_slabs,
                        released_slots,
                        released_bytes,
                    )
                    raise ExpertSlabReclaimError(
                        str(exc),
                        result=result,
                        failed_slab_id=slab_id,
                        destructive_boundary_crossed=destroyed,
                    ) from exc
            self._slab_tickets.pop(ticket.ticket_id, None)
            return self._reclaim_result(
                released_slabs,
                released_slots,
                released_bytes,
            )

    def abort_slab_reclaim(self, ticket: ExpertSlabReclaimTicket) -> None:
        with self._slab_lock:
            if self._slab_tickets.get(ticket.ticket_id) is not ticket:
                raise ExpertSlotError("expert slab reclaim ticket is not active")
            if any(
                self._slabs[slab_id].state is ExpertSlabState.RELEASED
                for slab_id in ticket.slab_ids
            ):
                raise ExpertSlotError(
                    "cannot abort an expert slab reclaim after destruction"
                )
            self._restore_undestroyed_ticket_slabs(ticket)

    def regrow_slab(self, slab_id: int) -> ExpertSlab:
        """Recreate one slab without replacing its logical slot objects."""

        self._require_slab_owner_thread()
        with self._close_lock:
            if self._closed or self._closing:
                raise ExpertSlotError("expert slot pool is closing")
            return self._regrow_slab_locked(slab_id)

    def _regrow_slab_locked(self, slab_id: int) -> ExpertSlab:
        self._raise_completion_error()
        with self._slab_lock:
            try:
                slab = self._slabs[slab_id]
            except KeyError as exc:
                raise ValueError("unknown expert slab id") from exc
            if slab.state is not ExpertSlabState.RELEASED:
                raise ExpertSlotError("expert slab is not released")
            if slab_id in self._slab_release_failures:
                raise ExpertSlotError(
                    "expert slab release failed; regrowth is disabled"
                ) from self._slab_release_failures[slab_id]
            allocate_slab = getattr(self._allocator, "allocate_slab")
            try:
                buffers = allocate_slab(slab_id)
                if set(buffers) != set(slab.slot_ids):
                    raise ExpertSlotError(
                        "allocator returned an incomplete expert slab"
                    )
                validated = {
                    slot_id: self._validate_allocated_buffer(
                        buffers[slot_id],
                        f"global-persistent-{slot_id}",
                    )
                    for slot_id in slab.slot_ids
                }
            except BaseException as allocation_error:
                try:
                    allocated = bool(
                        getattr(self._allocator, "slab_is_allocated")(slab_id)
                    )
                except BaseException:
                    allocated = True
                if not allocated:
                    raise
                try:
                    cleanup = getattr(self._allocator, "release_slab")(slab_id)
                    if (
                        not isinstance(cleanup, ExpertSlabReleaseResult)
                        or not cleanup.destroyed
                        or cleanup.slab_id != slab_id
                        or cleanup.physical_bytes != slab.physical_bytes
                    ):
                        raise RuntimeError(
                            "allocator did not confirm failed-regrow cleanup"
                        )
                except BaseException as cleanup_error:
                    destroyed = self._allocator_destroyed_state(
                        slab_id,
                        cleanup_error,
                    )
                    if destroyed is not True:
                        if slab_id not in self._slab_ambiguous_allocations:
                            self._slab_ambiguous_allocations.add(slab_id)
                            self.allocated_bytes += slab.physical_bytes
                    self._slab_release_failures[slab_id] = cleanup_error
                    raise ExpertSlotError(
                        "expert slab allocation cleanup failed; regrowth is disabled"
                    ) from allocation_error
                raise
            for slot_id in slab.slot_ids:
                slot = self._persistent[(-1, slot_id)]
                with slot.condition:
                    slot.buffer = validated[slot_id]
                    slot.state = ExpertSlotState.EMPTY
                    slot.layer = None
                    slot.expert = None
                    slot.digest = None
                    slot.error = None
                    slot.condition.notify_all()
            slab.incarnation += 1
            slab.state = ExpertSlabState.ACTIVE
            self.allocated_bytes += slab.physical_bytes
            return slab

    def invalidate(
        self,
        layer: int,
        logical_slot: int,
        *,
        expert: int | None = None,
        generation: int | None = None,
    ) -> bool:
        slot = self._physical(layer, logical_slot)
        with slot.condition:
            if slot.layer != layer:
                return False
            if expert is not None and slot.expert != expert:
                return False
            if generation is not None and slot.generation != generation:
                return False
            if slot.state is ExpertSlotState.LOADING or slot.pins:
                raise ExpertSlotError("cannot invalidate an active expert slot")
            slot.state = ExpertSlotState.EMPTY
            slot.layer = None
            slot.expert = None
            slot.digest = None
            slot.error = None
            slot.condition.notify_all()
            return True

    def snapshot(self) -> dict[str, Any]:
        self._drain_completion_fences()
        self._raise_completion_error()
        states: dict[str, int] = {state.value: 0 for state in ExpertSlotState}
        pins = 0
        for slot in (*self._persistent.values(), *self._transient):
            with slot.condition:
                states[slot.state.value] += 1
                pins += slot.pins
        snapshot = {
            "buffer_backend": self.buffer_backend,
            "io_cache_mode": self.reader.cache_mode,
            "allocated_bytes": self.allocated_bytes,
            "max_inflight_io_bytes": self.max_inflight_io_bytes,
            "persistent_slot_count": len(self._persistent),
            "cache_scope": self.cache_scope,
            "persistent_route_capacity": self._persistent_route_capacity,
            "transient_slot_count": len(self._transient),
            "states": states,
            "pins": pins,
            "slabs": {
                "active": sum(
                    slab.state is ExpertSlabState.ACTIVE
                    for slab in self._slabs.values()
                ),
                "draining": sum(
                    slab.state is ExpertSlabState.DRAINING
                    for slab in self._slabs.values()
                ),
                "released": sum(
                    slab.state is ExpertSlabState.RELEASED
                    for slab in self._slabs.values()
                ),
                "physical_bytes": sum(
                    slab.physical_bytes
                    for slab_id, slab in self._slabs.items()
                    if slab.state is ExpertSlabState.ACTIVE
                    or slab_id in self._slab_ambiguous_allocations
                ),
            },
            "metrics": self.metrics.as_dict(),
            "physical_read_latency_by_layer": (
                self.metrics.physical_read_latency_by_layer()
            ),
            "io": self.reader.metrics.as_dict(),
        }
        if self._reader_pool_telemetry is not None:
            assert self._completion_fence_telemetry is not None
            snapshot["reader_pool"] = self._reader_pool_telemetry.snapshot()
            snapshot["completion_fences"] = self._completion_fence_telemetry.snapshot()
        return snapshot

    def resource_telemetry_snapshot(self) -> dict[str, Any]:
        """Return cumulative resource counters without draining or walking slots."""

        if self._reader_pool_telemetry is None:
            raise ExpertSlotError("resource telemetry is disabled")
        assert self._completion_fence_telemetry is not None
        return {
            "io_cache_mode": self.reader.cache_mode,
            "metrics": self.metrics.as_dict(),
            "io": self.reader.metrics.as_dict(),
            "reader_pool": self._reader_pool_telemetry.snapshot(),
            "completion_fences": self._completion_fence_telemetry.snapshot(),
        }

    def reset(self) -> None:
        self._retry_cleanup_owners()
        self._raise_completion_error()
        with self._slab_lock:
            if any(
                slab.state is ExpertSlabState.DRAINING for slab in self._slabs.values()
            ):
                raise ExpertSlotError("cannot reset while an expert slab is draining")
        with self._lifecycle:
            if self.metrics.as_dict()["active_routes"]:
                raise ExpertSlotError("cannot reset while expert routes are active")
        for slot in (*self._persistent.values(), *self._transient):
            with slot.condition:
                if slot.state is ExpertSlotState.LOADING or slot.pins:
                    raise ExpertSlotError("cannot reset an active expert slot")
                slot.state = ExpertSlotState.EMPTY
                slot.layer = None
                slot.expert = None
                slot.digest = None
                slot.error = None
                slot.condition.notify_all()

    def close(self, *, timeout: float | None = None) -> None:
        if self._slabs:
            self._require_slab_owner_thread()
        deadline = None if timeout is None else time.monotonic() + timeout
        if deadline is None:
            self._close_lock.acquire()
        else:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._close_lock.acquire(timeout=remaining):
                raise TimeoutError("expert slot close already in progress at deadline")
        try:
            with self._lifecycle:
                if not self._closed:
                    self._closing = True
            with self._slab_lock:
                for ticket in tuple(self._slab_tickets.values()):
                    self._restore_undestroyed_ticket_slabs(ticket)
                for slab in self._slabs.values():
                    if slab.state is ExpertSlabState.DRAINING:
                        slab.state = ExpertSlabState.ACTIVE
            while True:
                self._retry_cleanup_owners()
                if self._has_cleanup_owners():
                    self._raise_completion_error()
                with self._lifecycle:
                    if self._closed:
                        finalized = True
                        break
                    self._closing = True
                    if not self.metrics.as_dict()["active_routes"]:
                        finalized = False
                        break
                    # Registration holds this condition while publishing an
                    # owner, so checking here closes the scan-to-wait race.
                    if self._has_cleanup_owners():
                        continue
                    if deadline is None:
                        self._lifecycle.wait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                "active expert routes did not release before close"
                            )
                        self._lifecycle.wait(remaining)
            if not finalized:
                self._completion_executor.shutdown(wait=True, cancel_futures=False)
                self._executor.shutdown(wait=True, cancel_futures=True)
                for slot in (*self._persistent.values(), *self._transient):
                    with slot.condition:
                        slot.state = ExpertSlotState.CLOSED
                        slot.condition.notify_all()
                self.reader.close()
                allocator_close = getattr(self._allocator, "close", None)
                if callable(allocator_close):
                    allocator_close()
                for slot in (*self._persistent.values(), *self._transient):
                    slot.buffer = None
                self._allocator = _closed_allocator
                with self._lifecycle:
                    self._closed = True
                    self._closing = False
                    self._lifecycle.notify_all()
            self._raise_completion_error()
        finally:
            self._close_lock.release()

    def __enter__(self) -> ExpertSlotPool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def release_routes(routes: Iterable[ReadyRoute]) -> None:
    for route in routes:
        route.release()
