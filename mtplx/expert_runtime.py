"""Model-independent orchestration for bounded SSD expert streaming."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .expert_io import PositionalExpertReader
from .expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    load_expert_manifest,
    verify_expert_manifest,
)
from .expert_slots import (
    ExpertCompletionFenceError,
    ExpertRecordReleaseResult,
    ExpertSlotError,
    ExpertSlotPool,
    ReadyRoute,
    RouteIOAdmission,
)
from .expert_streaming import (
    CacheCounters,
    GlobalExpertSlotBank,
    LayerExpertSlotBank,
    RoutePlan,
    RoutePolicyTxn,
    RoutingPhase,
)
from .expert_streaming_models import (
    ExpertMemoryPlan,
    ExpertStreamingModelSpec,
    get_model_spec,
    plan_expert_memory,
)
from .memory_broker import (
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BYTES_PER_TOKEN,
    AllocatorMemorySample,
    BrokerSnapshot,
    KVAllocationGroupTicket,
    KVAllocationTicket,
    KVPhysicalAllocation,
    MemoryAdmissionError,
    MemoryTelemetryError,
    UnifiedMemoryBroker,
    hy3_q4_kv_physical_geometry,
)
from .resource_metrics import ExpertPipelineLedger, ExpertPipelineRoute


_MEMORY_RE = re.compile(r"^([0-9]+)([kmgt]i?b?|b)?$", re.IGNORECASE)
_ALLOCATOR_STABLE_SAMPLE_READS = 8


class ExpertStreamingConfigurationError(ValueError):
    pass


def _pipeline_call(
    ledger: ExpertPipelineLedger | None,
    route: ExpertPipelineRoute,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Publish optional diagnostics without changing runtime outcomes."""

    try:
        getattr(route, method)(*args, **kwargs)
    except Exception:
        if ledger is not None:
            try:
                ledger.mark_incomplete(phase=route.phase)
            except Exception:
                pass


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def parse_memory_bytes(value: str | int) -> int:
    if isinstance(value, bool):
        raise ExpertStreamingConfigurationError("memory size must not be bool")
    if isinstance(value, int):
        if value <= 0:
            raise ExpertStreamingConfigurationError("memory size must be positive")
        return value
    if not isinstance(value, str):
        raise ExpertStreamingConfigurationError(
            "memory size must be bytes or a suffixed string"
        )
    normalized = value.strip().lower()
    match = _MEMORY_RE.fullmatch(normalized)
    if match is None:
        raise ExpertStreamingConfigurationError(f"invalid memory size {value!r}")
    number = int(match.group(1))
    suffix = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    result = number * multipliers[suffix]
    if result <= 0:
        raise ExpertStreamingConfigurationError("memory size must be positive")
    return result


@dataclass(frozen=True)
class ExpertStreamingConfig:
    model_key: str
    memory_limit_bytes: int
    max_live_kv_tokens: int
    kv_bytes_per_token_override: int | None = None
    runtime_reserve_bytes: int = 16 * 1024**3
    allocator_headroom_bytes: int = 0
    expert_cache_limit_bytes: int | None = None
    transient_slots: int | None = None
    io_staging_bytes: int = 0
    execution_workspace_bytes: int = 0
    max_inflight_io_bytes: int | None = None
    max_open_files: int = 16
    max_read_chunk_bytes: int = 8 * 1024 * 1024
    frequency_decay: float = 0.995
    prefer_sidecar: bool = True
    verify_record_hashes: bool = True
    verify_artifact_headers: bool = True
    verify_sidecar_hash_at_open: bool = False
    prefill_admission: bool = False
    slot_layout: str = "direct-slots"
    trace_routes: bool = False
    cache_policy: str = "frequency"
    cache_scope: str = "layer"
    bypass_page_cache: bool = False
    resource_telemetry: bool = False
    dynamic_expert_cache: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise TypeError("model_key must be a non-empty string")
        for name, minimum in (
            ("memory_limit_bytes", 1),
            ("max_live_kv_tokens", 0),
            ("runtime_reserve_bytes", 0),
            ("allocator_headroom_bytes", 0),
            ("io_staging_bytes", 0),
            ("execution_workspace_bytes", 0),
            ("max_open_files", 1),
            ("max_read_chunk_bytes", 1),
        ):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), minimum=minimum)
            )
        for name in (
            "expert_cache_limit_bytes",
            "kv_bytes_per_token_override",
            "transient_slots",
            "max_inflight_io_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(name, value, minimum=0))
        if self.max_inflight_io_bytes == 0:
            raise ValueError("max_inflight_io_bytes must be positive when supplied")
        if self.kv_bytes_per_token_override == 0:
            raise ValueError(
                "kv_bytes_per_token_override must be positive when supplied"
            )
        if isinstance(self.frequency_decay, bool):
            raise TypeError("frequency_decay must be numeric")
        decay = float(self.frequency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("frequency_decay must be in (0, 1]")
        object.__setattr__(self, "frequency_decay", decay)
        if self.cache_policy not in {"frequency", "lru"}:
            raise ValueError("cache_policy must be 'frequency' or 'lru'")
        if self.cache_scope not in {"layer", "global"}:
            raise ValueError("cache_scope must be 'layer' or 'global'")
        if self.slot_layout not in {
            "direct-slots",
            "component-banks",
            "metal-mmap",
        }:
            raise ValueError(
                "slot_layout must be 'direct-slots', 'component-banks', or 'metal-mmap'"
            )
        for name in (
            "prefer_sidecar",
            "verify_record_hashes",
            "verify_artifact_headers",
            "verify_sidecar_hash_at_open",
            "prefill_admission",
            "trace_routes",
            "bypass_page_cache",
            "resource_telemetry",
            "dynamic_expert_cache",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.verify_sidecar_hash_at_open and not self.prefer_sidecar:
            raise ValueError(
                "verify_sidecar_hash_at_open requires prefer_sidecar: source-shard "
                "reads are not covered by the sidecar digest"
            )
        if self.slot_layout == "metal-mmap" and not self.verify_sidecar_hash_at_open:
            raise ValueError(
                "metal-mmap executes mapped weights without per-record hashing; "
                "it requires verify_sidecar_hash_at_open"
            )
        if self.cache_scope == "global" and self.slot_layout == "metal-mmap":
            raise ValueError(
                "global expert caching requires direct-slots or component-banks"
            )
        if self.prefill_admission:
            raise ValueError(
                "prefill admission is not implemented; prefill must use transient slots"
            )
        if self.dynamic_expert_cache:
            if self.model_key != "hy3-q4":
                raise ValueError("dynamic expert cache requires model_key='hy3-q4'")
            if self.cache_scope != "global":
                raise ValueError("dynamic expert cache requires global caching")
            if self.slot_layout != "direct-slots":
                raise ValueError("dynamic expert cache requires direct-slots")
            if self.cache_policy != "lru":
                raise ValueError("dynamic expert cache requires cache_policy='lru'")
            if self.max_live_kv_tokens != 131_072:
                raise ValueError(
                    "dynamic expert cache requires max_live_kv_tokens=131072"
                )

    def memory_plan(self, spec: ExpertStreamingModelSpec) -> ExpertMemoryPlan:
        # File-backed Metal records use the OS page cache as their physical
        # tier and never consume fixed MLX expert slots. Retain only a tiny
        # unreachable transient pool so the generic runtime invariants and
        # diagnostics remain valid while the mapped switch owns execution.
        expert_cache_limit_bytes = self.expert_cache_limit_bytes
        transient_slots = self.transient_slots
        if self.slot_layout == "metal-mmap":
            expert_cache_limit_bytes = 0
            transient_slots = spec.top_k
        total_limit_bytes = self.memory_limit_bytes
        context_tokens = self.max_live_kv_tokens
        if self.dynamic_expert_cache:
            context_tokens = 0
        requested_context_tokens = context_tokens
        if (
            self.model_key == "hy3-q4"
            and self.kv_bytes_per_token_override == HY3_Q4_KV_BYTES_PER_TOKEN
        ):
            context_tokens = hy3_q4_kv_physical_geometry(
                context_tokens
            ).physical_capacity_tokens
        planning_spec = (
            spec
            if self.kv_bytes_per_token_override is None
            else replace(
                spec,
                kv_bytes_per_token=self.kv_bytes_per_token_override,
            )
        )
        plan = plan_expert_memory(
            planning_spec,
            total_limit_bytes=total_limit_bytes,
            context_tokens=context_tokens,
            runtime_reserve_bytes=self.runtime_reserve_bytes,
            allocator_headroom_bytes=self.allocator_headroom_bytes,
            expert_cache_limit_bytes=expert_cache_limit_bytes,
            transient_slots=transient_slots,
            io_staging_bytes=self.io_staging_bytes,
            execution_workspace_bytes=self.execution_workspace_bytes,
            cache_scope=self.cache_scope,
        )
        if plan.context_tokens != requested_context_tokens:
            plan = replace(plan, context_tokens=requested_context_tokens)
        return plan

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _pipeline_ledger_for_config(
    config: ExpertStreamingConfig,
) -> ExpertPipelineLedger | None:
    """Enable pipeline attribution only on the instrumented slot-backed path."""

    if not config.resource_telemetry or config.slot_layout == "metal-mmap":
        return None
    return ExpertPipelineLedger(strict=False)


@dataclass(frozen=True)
class RouteWave:
    positions: tuple[int, ...]
    experts: tuple[int, ...]


def partition_route_waves(
    expert_ids: Iterable[int],
    *,
    max_unique_experts: int,
    sort_unique: bool = False,
) -> tuple[RouteWave, ...]:
    """Greedily partition flattened assignments into bounded expert unions."""

    capacity = _integer("max_unique_experts", max_unique_experts, minimum=1)
    experts = tuple(expert_ids)
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for expert in experts:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError("expert ids must be exact integers")
        if expert not in seen:
            seen.add(expert)
            ordered_unique.append(expert)
    if sort_unique:
        ordered_unique.sort()
    waves: list[RouteWave] = []
    for start in range(0, len(ordered_unique), capacity):
        selected = set(ordered_unique[start : start + capacity])
        positions = tuple(
            position for position, expert in enumerate(experts) if expert in selected
        )
        waves.append(
            RouteWave(
                positions=positions,
                experts=tuple(experts[position] for position in positions),
            )
        )
    return tuple(waves)


@dataclass
class KVAdmission:
    runtime: ExpertStreamingRuntime
    tokens: int
    released: bool = False

    def grow_to(self, tokens: int) -> None:
        """Monotonically extend this lease without changing its identity."""

        self.runtime._grow_kv_admission(self, tokens)

    def grow_to_page_boundary(self, tokens: int, *, page_size_tokens: int) -> bool:
        """Track exact logical progress and report physical page crossings."""

        target = _integer("tokens", tokens, minimum=1)
        page_size = _integer("page_size_tokens", page_size_tokens, minimum=1)
        current = int(self.tokens)
        if target <= current:
            return False
        current_pages = (current + page_size - 1) // page_size
        target_pages = (target + page_size - 1) // page_size
        self.grow_to(target)
        return target_pages > current_pages

    def release(self) -> None:
        self.runtime._release_kv_admission(self)

    def __enter__(self) -> KVAdmission:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass
class KVGroupGrowthContext:
    """One aggregate physical-KV transaction held across serial member growth."""

    runtime: ExpertStreamingRuntime
    ticket: KVAllocationGroupTicket
    completed: bool = False
    aborted: bool = False
    _owner_thread_id: int = field(default_factory=threading.get_ident, repr=False)
    _allocations: dict[str, KVPhysicalAllocation] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _active: bool = field(default=True, init=False, repr=False)

    @property
    def remaining_cache_ids(self) -> tuple[str, ...]:
        return tuple(
            member.cache_id
            for member in self.ticket.members
            if member.cache_id not in self._allocations
        )

    @property
    def allocations(self) -> tuple[KVPhysicalAllocation, ...]:
        return tuple(
            self._allocations[member.cache_id]
            for member in self.ticket.members
            if member.cache_id in self._allocations
        )

    def commit_member(
        self,
        *,
        cache_id: str,
        measured_physical_bytes: int,
        allocator_before: AllocatorMemorySample,
        allocator_after: AllocatorMemorySample,
    ) -> KVPhysicalAllocation:
        return self.runtime._commit_growth_group_member(
            self,
            cache_id=cache_id,
            measured_physical_bytes=measured_physical_bytes,
            allocator_before=allocator_before,
            allocator_after=allocator_after,
        )

    def abort(
        self,
        *,
        observed_uncommitted_physical_bytes: int | None,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> BrokerSnapshot:
        return self.runtime._abort_growth_group(
            self,
            observed_uncommitted_physical_bytes=(observed_uncommitted_physical_bytes),
            allocator_before=allocator_before,
            allocator_after=allocator_after,
        )

    def __enter__(self) -> KVGroupGrowthContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object,
    ) -> None:
        if not self._active:
            return
        try:
            self.abort(observed_uncommitted_physical_bytes=None)
        except BaseException:
            if exc_type is None:
                raise


class _ReadyRouteGroup:
    """Several independently completed route parts in original route order."""

    def __init__(self, plan: RoutePlan, parts: tuple[ReadyRoute, ...]) -> None:
        if not parts:
            raise ExpertSlotError("incremental miss group must contain a route part")
        pool = parts[0].pool
        if any(part.pool is not pool for part in parts[1:]):
            raise ExpertSlotError("incremental miss parts belong to different pools")
        self.plan = plan
        self.parts = parts
        self.pool = pool
        bindings_by_slot: dict[tuple[int, int], list[Any]] = {}
        for part in parts:
            for binding in part.bindings:
                bindings_by_slot.setdefault(
                    (binding.expert, binding.logical_slot),
                    [],
                ).append(binding)
        try:
            self.bindings = tuple(
                bindings_by_slot[(expert, slot)].pop(0)
                for expert, slot in zip(plan.experts, plan.slots, strict=True)
            )
        except (KeyError, IndexError) as exc:
            raise ExpertSlotError(
                "incremental miss parts do not cover the original route"
            ) from exc
        if any(bindings_by_slot.values()):
            raise ExpertSlotError(
                "incremental miss parts exceed the original route coverage"
            )

    @property
    def slots(self) -> tuple[int, ...]:
        return self.plan.slots

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    def validate(self) -> None:
        for part in self.parts:
            part.validate()


class _RouteCancel:
    """One route-local cancellation source composed with its caller."""

    def __init__(
        self,
        caller: threading.Event | None,
        internal: threading.Event,
    ) -> None:
        self.caller = caller
        self.internal = internal

    def is_set(self) -> bool:
        return self.internal.is_set() or (
            self.caller is not None and self.caller.is_set()
        )


class PendingSplitRoute:
    """One layer transaction with pinned hits and asynchronously loading misses."""

    def __init__(
        self,
        runtime: "ExpertStreamingRuntime",
        layer: int,
        plan: RoutePlan,
        layer_lock: threading.Lock,
        hit_ready: ReadyRoute | None,
        miss_futures: dict[Future[ReadyRoute], RoutePlan],
        policy_txn: RoutePolicyTxn | None = None,
        io_admission: RouteIOAdmission | None = None,
        miss_cancel_event: threading.Event | None = None,
        lifecycle_release: Callable[[], None] | None = None,
        miss_parts: tuple[RoutePlan, ...] | None = None,
        pipeline_route: ExpertPipelineRoute | None = None,
    ) -> None:
        self.runtime = runtime
        self.layer = layer
        self.plan = plan
        self._policy_txn = policy_txn or RoutePolicyTxn(rollback=lambda: None)
        self._io_admission = io_admission
        self._policy_observed = False
        self.hit_ready = hit_ready
        self._miss_futures = dict(miss_futures)
        self._miss_ordinals = {
            future: ordinal for ordinal, future in enumerate(self._miss_futures)
        }
        self._all_miss_ordinals = set(self._miss_ordinals.values())
        initial_parts = tuple(self._miss_futures.values())
        self._all_miss_parts = {
            ordinal: part
            for ordinal, part in enumerate(
                initial_parts if miss_parts is None else miss_parts
            )
        }
        self._submitted_miss_ordinals = set(self._miss_ordinals.values())
        self._miss_admissions: dict[int, RouteIOAdmission] = {}
        self._completed_miss_ordinals: set[int] = set()
        self._miss_ready_parts: dict[int, ReadyRoute] = {}
        self._claimed_miss_futures: dict[Future[ReadyRoute], int] = {}
        self._consumer_leases: set[int] = set()
        self._releasing_consumer_leases: set[int] = set()
        self._miss_ready: _ReadyRouteGroup | None = None
        self._aggregate_lease: _ReadyRouteGroup | None = None
        self._aggregate_lease_ordinals: set[int] = set()
        self._miss_cancel_event = miss_cancel_event or threading.Event()
        self._lifecycle_release = lifecycle_release
        self._pipeline_route = pipeline_route
        self._layer_lock = layer_lock
        self._state_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._failure_callbacks = 0
        self._failure_finalizing = False
        self._failure_finalized = False
        self._cleanup_error: BaseException | None = None
        self._ready_cleanup_complete = False
        self._ready_cleanup_finalizing = False
        self._hits_released = hit_ready is None
        self._close_requested = False
        self._finalized = False
        self._closed = False

    def release_hits(self) -> None:
        ready = self.hit_ready
        if ready is None:
            return
        self.hit_ready = None
        self._hits_released = True
        try:
            ready.release(synchronize=False)
        except BaseException as exc:
            self._record_cleanup_error(exc)
            raise

    @property
    def misses_pending(self) -> bool:
        """Whether miss I/O still offers useful work-overlap headroom."""

        with self._state_lock:
            return any(not future.done() for future in self._miss_futures)

    def claim_misses(self, ready: ReadyRoute | _ReadyRouteGroup) -> None:
        """Record the point at which runnable miss bindings are consumed."""

        route = self._pipeline_route
        if route is None:
            return
        experts = tuple(dict.fromkeys(load.expert for load in ready.plan.loads))
        if experts:
            _pipeline_call(
                self.runtime._pipeline_ledger,
                route,
                "claim_misses",
                experts,
            )

    def _attach_miss_future(
        self,
        future: Future[ReadyRoute],
        plan: RoutePlan,
        *,
        ordinal: int,
        io_admission: RouteIOAdmission | None = None,
    ) -> None:
        with self._state_lock:
            self._miss_futures[future] = plan
            self._miss_ordinals[future] = ordinal
            self._all_miss_ordinals.add(ordinal)
            self._all_miss_parts[ordinal] = plan
            self._submitted_miss_ordinals.add(ordinal)
            if io_admission is not None:
                self._miss_admissions[ordinal] = io_admission

    def _retain_lifecycle_after_admission(self) -> None:
        with self._state_lock:
            if (
                self._lifecycle_release is not None
                or self._failure_finalized
                or self._policy_observed
            ):
                return
            lifecycle = self.runtime.slots.retain_admitted_split_lifecycle()
            self._lifecycle_release = lifecycle.release

    def _record_cleanup_error(self, error: BaseException) -> None:
        promote = False
        with self._state_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error
                promote = True
        if promote:
            # Never nest the runtime health lock under Pending state.
            recorder = getattr(self.runtime, "_record_cleanup_error", None)
            if callable(recorder):
                recorder(error)

    def _release_lifecycle(self) -> None:
        with self._state_lock:
            release = self._lifecycle_release
            self._lifecycle_release = None
        if release is None:
            return
        try:
            release()
        except BaseException as exc:
            with self._state_lock:
                if self._lifecycle_release is None:
                    self._lifecycle_release = release
            self._record_cleanup_error(exc)

    @staticmethod
    def _release_routes(routes: Iterable[ReadyRoute]) -> BaseException | None:
        first_error: BaseException | None = None
        for ready in routes:
            try:
                ready.release(synchronize=False)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    def _store_completed_future(
        self,
        future: Future[ReadyRoute],
        ordinal: int,
    ) -> None:
        try:
            ready = future.result()
        except BaseException:
            return
        with self._state_lock:
            self._miss_ready_parts[ordinal] = ready

    def _failure_callback(
        self,
        future: Future[ReadyRoute],
        ordinal: int,
    ) -> None:
        self._store_completed_future(future, ordinal)
        with self._state_lock:
            self._failure_callbacks -= 1
        self._finish_failure_if_ready()

    def abort(self, error: BaseException) -> None:
        """Cancel this transaction without waiting for running miss workers."""

        # Cancellation is the first observable failure action. Every miss part
        # sees this same route-local event, composed with the caller event.
        self._miss_cancel_event.set()
        with self._state_lock:
            if self._failure is not None:
                return
            self._failure = error
            pending = tuple(
                sorted(
                    self._miss_futures,
                    key=lambda future: self._miss_ordinals[future],
                )
            )
            ordinals = {future: self._miss_ordinals[future] for future in pending}
            self._miss_futures.clear()
            self._miss_ordinals.clear()
            self._miss_ready = None
            # Claim every future before failure becomes observable without the
            # state lock. Completed paths and callbacks each consume one claim.
            self._failure_callbacks = len(pending)

        for future in pending:
            future.cancel()
        completed: list[Future[ReadyRoute]] = []
        running: list[Future[ReadyRoute]] = []
        for future in pending:
            (completed if future.done() else running).append(future)
        for future in completed:
            self._failure_callback(future, ordinals[future])
        for future in running:
            future.add_done_callback(
                lambda done, ordinal=ordinals[future]: self._failure_callback(
                    done,
                    ordinal,
                )
            )
        self._finish_failure_if_ready()

    def _finish_failure_if_ready(self) -> None:
        with self._state_lock:
            if (
                self._failure is None
                or self._failure_callbacks
                or self._claimed_miss_futures
                or self._consumer_leases
                or self._releasing_consumer_leases
                or self._aggregate_lease is not None
                or self._ready_cleanup_finalizing
                or self._failure_finalizing
                or self._failure_finalized
            ):
                return
            self._failure_finalizing = True
            failure = self._failure
            policy_observed = self._policy_observed

        try:
            with self._state_lock:
                routes = tuple(
                    self._miss_ready_parts[ordinal]
                    for ordinal in sorted(self._miss_ready_parts)
                )
                self._miss_ready_parts.clear()
                self._miss_ready = None
            release_error = self._release_routes(routes)
            if release_error is not None:
                self._record_cleanup_error(release_error)
            if not policy_observed:
                try:
                    accepted_ordinals = {
                        ordinal
                        for ordinal, admission in self._miss_admissions.items()
                        if admission.any_accepted
                    }
                    if not self._miss_admissions and (
                        self._io_admission is not None
                        and self._io_admission.any_accepted
                    ):
                        accepted_ordinals = set(self._submitted_miss_ordinals)
                    accepted_parts = tuple(
                        self._all_miss_parts[ordinal]
                        for ordinal in sorted(accepted_ordinals)
                    )
                    self.runtime._handle_split_route_failure(
                        self.layer,
                        self.plan,
                        self._policy_txn,
                        failure,
                        accepted_parts=accepted_parts,
                        io_admission=self._io_admission,
                    )
                except BaseException as exc:
                    self._record_cleanup_error(exc)
        finally:
            self._release_lifecycle()
            with self._state_lock:
                self._ready_cleanup_complete = True
                self._failure_finalized = True
                self._failure_finalizing = False
            self._finalize_if_ready()

    def _iter_pipeline_miss_completions(
        self,
        snapshot: tuple[Future[ReadyRoute], ...],
    ) -> Iterable[Future[ReadyRoute]]:
        """Bound a completion step that may block on the existing iterator.

        The readiness scan and ``next(as_completed(...))`` cannot be atomic through
        the public Future API. The measured span is therefore an upper bound that
        can include completion races, iterator work, and telemetry-lock delay.
        """

        route = self._pipeline_route
        assert route is not None
        completions = iter(as_completed(snapshot))
        remaining = set(snapshot)
        while remaining:
            try:
                may_block_for_next = not any(future.done() for future in remaining)
            except Exception:
                may_block_for_next = False
                ledger = self.runtime._pipeline_ledger
                if ledger is not None:
                    try:
                        ledger.mark_incomplete(phase=route.phase)
                    except Exception:
                        pass
            if may_block_for_next:
                _pipeline_call(
                    self.runtime._pipeline_ledger,
                    route,
                    "begin_potentially_blocking_next_miss_step",
                )
            try:
                future = next(completions)
            finally:
                if may_block_for_next:
                    _pipeline_call(
                        self.runtime._pipeline_ledger,
                        route,
                        "end_potentially_blocking_next_miss_step",
                    )
            remaining.discard(future)
            yield future

    def iter_ready_misses(self) -> Iterable[ReadyRoute]:
        """Yield authoritative miss bindings in physical completion order."""

        with self._state_lock:
            snapshot = tuple(self._miss_futures)
        completion_order = (
            as_completed(snapshot)
            if self._pipeline_route is None
            else self._iter_pipeline_miss_completions(snapshot)
        )
        for future in completion_order:
            with self._state_lock:
                if future not in self._miss_futures:
                    continue
                self._miss_futures.pop(future)
                ordinal = self._miss_ordinals.pop(future)
                self._claimed_miss_futures[future] = ordinal
            try:
                ready = future.result()
            except BaseException as exc:
                with self._state_lock:
                    self._claimed_miss_futures.pop(future, None)
                    failure = self._failure
                if failure is not None:
                    self._finish_failure_if_ready()
                    raise failure
                self.abort(exc)
                raise
            with self._state_lock:
                self._claimed_miss_futures.pop(future, None)
                self._miss_ready_parts[ordinal] = ready
                self._completed_miss_ordinals.add(ordinal)
                failure = self._failure
            if failure is not None:
                self._finish_failure_if_ready()
                raise failure
            try:
                self.runtime.slots.raise_if_unhealthy()
            except BaseException as exc:
                self.abort(exc)
                raise
            with self._state_lock:
                is_final_part = (
                    not self._miss_futures and not self._claimed_miss_futures
                )
            if is_final_part:
                try:
                    self.runtime.slots.commit_if_healthy(
                        lambda ordinal=ordinal: self._validate_and_commit_policy(
                            lease_ordinal=ordinal
                        )
                    )
                except BaseException as exc:
                    self.abort(exc)
                    raise
            else:
                with self._state_lock:
                    if self._failure is None and not self._close_requested:
                        self._consumer_leases.add(ordinal)
            with self._state_lock:
                leased = ordinal in self._consumer_leases
                failure = self._failure
                if failure is not None and leased:
                    self._consumer_leases.remove(ordinal)
                    leased = False
            if failure is not None:
                self._finish_failure_if_ready()
                raise failure
            if not leased:
                failure = ExpertSlotError("split route closed before miss yield")
                self.abort(failure)
                self._finish_failure_if_ready()
                raise failure
            yield ready
        if not self._policy_observed:
            try:
                self.runtime.slots.raise_if_unhealthy()
                self.runtime.slots.commit_if_healthy(self._validate_and_commit_policy)
            except BaseException as exc:
                self.abort(exc)
                raise

    def _validate_completed_misses(self) -> None:
        with self._state_lock:
            if self._completed_miss_ordinals != self._all_miss_ordinals:
                raise ExpertSlotError(
                    "incremental miss completion does not cover every route part"
                )

    def _validate_and_commit_policy(self, *, lease_ordinal: int | None = None) -> None:
        """Validate and publish only host policy state and counters."""

        self._validate_completed_misses()
        self._commit_policy(lease_ordinal=lease_ordinal)

    def _prepare_ready_group(self) -> _ReadyRouteGroup | None:
        miss_plan = self.runtime._subset_route_plan(self.plan, hits=False)
        if miss_plan is None:
            return None
        with self._state_lock:
            if self._miss_ready is not None:
                return self._miss_ready
            if self._failure is not None:
                raise self._failure
            if self._close_requested:
                raise ExpertSlotError("split route closed before miss aggregation")
            if self._completed_miss_ordinals != self._all_miss_ordinals:
                raise ExpertSlotError(
                    "incremental miss completion does not cover every route part"
                )
            expected_ordinals = set(self._all_miss_ordinals)
            if set(self._miss_ready_parts) != expected_ordinals:
                raise ExpertSlotError(
                    "incremental miss parts do not cover the original route"
                )
            if self._consumer_leases != expected_ordinals:
                raise ExpertSlotError(
                    "incremental miss aggregation does not own every route part"
                )
            parts = tuple(
                self._miss_ready_parts[ordinal] for ordinal in sorted(expected_ordinals)
            )
            group = _ReadyRouteGroup(miss_plan, parts)
            self._consumer_leases.clear()
            self._aggregate_lease = group
            self._aggregate_lease_ordinals = expected_ordinals
            self._miss_ready = group
            return group

    def finish_misses(self) -> _ReadyRouteGroup | None:
        with self._state_lock:
            if self._miss_ready is not None:
                return self._miss_ready
            empty = not self._miss_futures and not self._miss_ready_parts
        if empty:
            return None
        try:
            for _ready in self.iter_ready_misses():
                pass
            return self._prepare_ready_group()
        except BaseException as exc:
            self.abort(exc)
            with self._state_lock:
                leased_readies = tuple(
                    self._miss_ready_parts[ordinal]
                    for ordinal in sorted(self._consumer_leases)
                )
            for ready in leased_readies:
                try:
                    self.release_miss(ready)
                except BaseException:
                    pass
            self._finish_failure_if_ready()
            raise

    def _commit_policy(self, *, lease_ordinal: int | None = None) -> None:
        committed = False
        with self._state_lock:
            if self._failure is not None:
                return
            if not self._policy_observed:
                incremental_parts = len(self._completed_miss_ordinals)
                self.runtime._publish_route_transaction(
                    self.layer,
                    self.plan,
                    self._policy_txn,
                    incremental_parts=incremental_parts,
                )
                self._policy_observed = True
                committed = True
            if lease_ordinal is not None:
                self._consumer_leases.add(lease_ordinal)
        if committed:
            self._release_lifecycle()

    def release_miss(self, ready: ReadyRoute) -> None:
        """Release one streamed part while retaining sole route ownership."""

        with self._state_lock:
            matches = tuple(
                ordinal
                for ordinal, candidate in self._miss_ready_parts.items()
                if candidate is ready
            )
            if len(matches) != 1:
                raise ExpertSlotError("miss part is not owned by this split route")
            ordinal = matches[0]
            if ordinal not in self._consumer_leases:
                raise ExpertSlotError("miss part has no active consumer lease")
            self._consumer_leases.remove(ordinal)
            self._releasing_consumer_leases.add(ordinal)
            self._miss_ready_parts.pop(ordinal)
            self._miss_ready = None
        release_error: BaseException | None = None
        try:
            ready.release(synchronize=False)
        except BaseException as exc:
            release_error = exc
            self._record_cleanup_error(exc)
        with self._state_lock:
            self._releasing_consumer_leases.remove(ordinal)
        success_error = self._finish_success_close_if_ready()
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if release_error is not None:
            raise release_error
        if success_error is not None:
            raise success_error

    def release_misses(self, ready: _ReadyRouteGroup) -> None:
        """Return one aggregate consumer lease to Pending ownership."""

        with self._state_lock:
            if self._aggregate_lease is not ready:
                raise ExpertSlotError("miss aggregate is not owned by this split route")
            ordinals = tuple(sorted(self._aggregate_lease_ordinals))
            routes = tuple(self._miss_ready_parts[ordinal] for ordinal in ordinals)
            self._aggregate_lease = None
            self._aggregate_lease_ordinals.clear()
            self._miss_ready = None
            self._releasing_consumer_leases.update(ordinals)
            for ordinal in ordinals:
                self._miss_ready_parts.pop(ordinal)
        release_error = self._release_routes(routes)
        if release_error is not None:
            self._record_cleanup_error(release_error)
        with self._state_lock:
            self._releasing_consumer_leases.difference_update(ordinals)
        success_error = self._finish_success_close_if_ready()
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if release_error is not None:
            raise release_error
        if success_error is not None:
            raise success_error

    def _finish_success_close_if_ready(self) -> BaseException | None:
        with self._state_lock:
            if (
                self._failure is not None
                or not self._close_requested
                or self._claimed_miss_futures
                or self._consumer_leases
                or self._releasing_consumer_leases
                or self._aggregate_lease is not None
                or self._ready_cleanup_finalizing
                or self._ready_cleanup_complete
            ):
                return None
            self._ready_cleanup_finalizing = True
            routes = tuple(
                self._miss_ready_parts[ordinal]
                for ordinal in sorted(self._miss_ready_parts)
            )
            self._miss_ready_parts.clear()
            self._miss_ready = None
        release_error = self._release_routes(routes)
        if release_error is not None:
            self._record_cleanup_error(release_error)
        with self._state_lock:
            self._ready_cleanup_complete = True
            self._ready_cleanup_finalizing = False
        self._finalize_if_ready()
        return release_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._state_lock:
            needs_abort = self._failure is None and not self._policy_observed
        if needs_abort:
            self.abort(ExpertSlotError("split route closed before commit"))
        first_error: BaseException | None = None
        try:
            self.release_hits()
        except BaseException as exc:
            first_error = exc
        with self._state_lock:
            failure = self._failure
            self._close_requested = True
        release_error = self._finish_success_close_if_ready()
        if first_error is None:
            first_error = release_error
        self._finish_failure_if_ready()
        self._finalize_if_ready()
        if failure is None and first_error is not None:
            raise first_error

    def _finalize_if_ready(self) -> None:
        with self._state_lock:
            if (
                self._finalized
                or not self._close_requested
                or not self._hits_released
                or not self._ready_cleanup_complete
            ):
                return
            self._finalized = True
            pipeline_route = self._pipeline_route
            self._pipeline_route = None
        try:
            if pipeline_route is not None:
                _pipeline_call(
                    self.runtime._pipeline_ledger,
                    pipeline_route,
                    "close",
                )
        finally:
            self._layer_lock.release()

    def __enter__(self) -> "PendingSplitRoute":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        if exc is not None:
            self.abort(exc)
        self.close()


def reconcile_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Resolve the MLX-owned portion and reject a conflicting env cap."""

    mlx_limit = (
        plan.total_limit_bytes - plan.runtime_reserve_bytes - plan.io_staging_bytes
    )
    if mlx_limit <= 0:
        raise ExpertStreamingConfigurationError(
            "memory plan leaves no MLX allocation budget"
        )
    source = os.environ if env is None else env
    existing = source.get("MTPLX_MEMORY_LIMIT_BYTES")
    if existing:
        parsed = parse_memory_bytes(existing)
        if parsed != mlx_limit:
            raise ExpertStreamingConfigurationError(
                "MTPLX_MEMORY_LIMIT_BYTES conflicts with expert streaming plan: "
                f"env={parsed}, planned={mlx_limit}"
            )
    return mlx_limit


def apply_mlx_memory_cap(
    plan: ExpertMemoryPlan,
    *,
    mx_module: Any | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the reconciled cap before resident or expert-slot allocation."""

    target_env = os.environ if env is None else env
    limit = reconcile_mlx_memory_cap(plan, env=target_env)
    target_env["MTPLX_MEMORY_LIMIT_BYTES"] = str(limit)
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {
                "applied": False,
                "reason": "mlx_unavailable",
                "error": repr(exc),
                "limit": limit,
            }
    else:
        mx = mx_module
    setter = getattr(mx, "set_memory_limit", None)
    if not callable(setter):
        metal = getattr(mx, "metal", None)
        setter = getattr(metal, "set_memory_limit", None)
    if not callable(setter):
        raise ExpertStreamingConfigurationError("MLX memory limit API is unavailable")
    setter(limit)
    return {"applied": True, "limit": limit}


def mlx_memory_telemetry(mx_module: Any | None = None) -> dict[str, int | str]:
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {"error": repr(exc)}
    else:
        mx = mx_module
    report: dict[str, int | str] = {}
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        getter = getattr(mx, name, None)
        if not callable(getter):
            getter = getattr(getattr(mx, "metal", None), name, None)
        if callable(getter):
            try:
                report[name.removeprefix("get_") + "_bytes"] = int(getter())
            except Exception as exc:
                report[name + "_error"] = repr(exc)
    return report


class ExpertStreamingRuntime:
    """Connect cache policy, checked I/O, fixed slots, and KV admission."""

    def __init__(
        self,
        root: Path,
        spec: ExpertStreamingModelSpec,
        config: ExpertStreamingConfig,
        manifest: ExpertManifest,
        plan: ExpertMemoryPlan,
        reader: PositionalExpertReader,
        slots: ExpertSlotPool,
        *,
        memory_cap_report: dict[str, Any] | None = None,
        integrity_report: dict[str, Any] | None = None,
        pipeline_ledger: ExpertPipelineLedger | None = None,
        memory_broker: UnifiedMemoryBroker | None = None,
        mx_module: Any | None = None,
    ) -> None:
        self.root = root
        self.spec = spec
        self.config = config
        self.manifest = manifest
        self.plan = plan
        self.reader = reader
        self.slots = slots
        self.memory_cap_report = memory_cap_report
        self.integrity_report = integrity_report
        self._pipeline_ledger = pipeline_ledger
        self.memory_broker = memory_broker
        self._mx_module = mx_module
        self._memory_transaction_lock = threading.RLock()
        self._active_kv_growth_group: KVGroupGrowthContext | None = None
        self._dynamic_cache_lock = threading.Lock()
        self._last_allocator_sample = (
            None if memory_broker is None else memory_broker.initial_allocator_sample
        )
        self._dynamic_cache_metrics = {
            "record_allocations": 0,
            "record_reuses": 0,
            "record_evictions": 0,
            "record_releases": 0,
        }
        self.counters = CacheCounters()
        self._layer_counters = {
            layer: CacheCounters() for layer in spec.routed_layer_indices
        }
        self._phase_counters = {phase: CacheCounters() for phase in RoutingPhase}
        self._counter_lock = threading.Lock()
        self._global_bank = (
            GlobalExpertSlotBank(
                layer_indices=spec.routed_layer_indices,
                expert_count=spec.expert_count,
                persistent_slots=plan.persistent_slots,
                transient_slots=plan.transient_slots,
                prefill_slots_per_layer=plan.slots_per_layer,
                frequency_decay=config.frequency_decay,
                cache_policy=config.cache_policy,
                initial_active_slots=(0 if config.dynamic_expert_cache else None),
            )
            if config.cache_scope == "global"
            else None
        )
        self._banks = (
            {}
            if self._global_bank is not None
            else {
                layer: LayerExpertSlotBank(
                    expert_count=spec.expert_count,
                    persistent_slots=plan.slots_per_layer,
                    transient_slots=plan.transient_slots,
                    frequency_decay=config.frequency_decay,
                    cache_policy=config.cache_policy,
                )
                for layer in spec.routed_layer_indices
            }
        )
        if self._global_bank is not None:
            # A route holds this lock through hit execution and miss loading.
            # Physical pinning remains the final overwrite fence, while this
            # lock prevents another layer from selecting the same global
            # victim before the current transaction publishes its mapping.
            global_lock = threading.Lock()
            self._layer_locks = {
                layer: global_lock for layer in spec.routed_layer_indices
            }
        else:
            self._layer_locks = {
                layer: threading.Lock() for layer in spec.routed_layer_indices
            }
        self._kv_lock = threading.Lock()
        self._live_kv_tokens = 0
        self._live_kv_peak = 0
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._cleanup_error_lock = threading.Lock()
        self._cleanup_error: BaseException | None = None
        self._pending_physical_kv_lock = threading.Lock()
        self._pending_physical_kv_condition = threading.Condition(
            self._pending_physical_kv_lock
        )
        self._pending_physical_kv_drainers = 0
        self._pending_physical_kv_caches: dict[int, Any] = {}
        self._mapped_expert_store: Any | None = None
        self._route_trace_lock = threading.Lock()
        self._route_trace: list[dict[str, Any]] = []
        self._route_trace_epoch = 0
        self._route_trace_decode_step = 0
        self._route_trace_decode_layers_seen: set[int] = set()
        self._incremental_miss_routes = 0
        self._incremental_miss_parts = 0
        self._split_executor = ThreadPoolExecutor(
            max_workers=max(1, plan.transient_slots),
            thread_name_prefix="mtplx-route-miss",
        )

    @staticmethod
    def _initialize_dynamic_memory_broker(
        *,
        config: Any,
        spec: Any,
        plan: Any,
        slots: Any,
        mx_module: Any | None,
    ) -> UnifiedMemoryBroker:
        """Build the additive physical-memory ledger for the dynamic Hy3 lane."""

        telemetry = mlx_memory_telemetry(mx_module)
        try:
            initial_allocator_sample = AllocatorMemorySample(
                active_bytes=int(telemetry["active_memory_bytes"]),
                cache_bytes=int(telemetry["cache_memory_bytes"]),
                peak_bytes=int(telemetry["peak_memory_bytes"]),
            )
            cache_telemetry = slots.persistent_cache_telemetry_snapshot()
            persistent_slots = int(plan.persistent_slots)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ExpertStreamingConfigurationError(
                "dynamic expert cache requires allocator and record telemetry"
            ) from exc
        if (
            not bool(getattr(slots, "lazy_persistent_buffers", False))
            or int(cache_telemetry["logical_record_capacity"]) != persistent_slots
            or int(cache_telemetry["allocated_record_count"]) != 0
            or int(cache_telemetry["physical_bytes"]) != 0
        ):
            raise ExpertStreamingConfigurationError(
                "dynamic expert cache must start with empty lazy direct records"
            )
        if not all(
            callable(getattr(slots, method, None))
            for method in (
                "allocate_persistent_slot",
                "release_persistent_slots",
            )
        ):
            raise ExpertStreamingConfigurationError(
                "dynamic expert cache requires direct record lifecycle support"
            )
        classified_bytes = (
            int(plan.resident_bytes)
            + int(plan.transient_bytes)
            + int(plan.io_staging_bytes)
            + int(plan.runtime_reserve_bytes)
            + int(plan.execution_workspace_bytes)
        )
        allocator_cache_bytes = max(
            initial_allocator_sample.cache_bytes,
            initial_allocator_sample.charged_footprint_bytes - classified_bytes,
        )
        initial_snapshot = BrokerSnapshot(
            resident_model_bytes=int(plan.resident_bytes),
            kv_physical_bytes=0,
            expert_cache_physical_bytes=0,
            in_flight_expert_staging_bytes=(
                int(plan.transient_bytes) + int(plan.io_staging_bytes)
            ),
            runtime_workspace_bytes=(
                int(plan.runtime_reserve_bytes) + int(plan.execution_workspace_bytes)
            ),
            allocator_cache_bytes=allocator_cache_bytes,
        )
        if initial_snapshot.charged_bytes > int(config.memory_limit_bytes):
            raise ExpertStreamingConfigurationError(
                "dynamic expert physical pools exceed the configured memory limit"
            )
        if initial_snapshot.classified_bytes > (
            int(config.memory_limit_bytes) - int(config.allocator_headroom_bytes)
        ):
            raise ExpertStreamingConfigurationError(
                "dynamic expert physical pools exceed the allocator-headroom target"
            )
        return UnifiedMemoryBroker.standard_hy3(
            initial_snapshot=initial_snapshot,
            initial_allocator_sample=initial_allocator_sample,
            memory_limit_bytes=int(config.memory_limit_bytes),
            expert_record_bytes=int(spec.expert_record_bytes),
            expert_cache_limit_bytes=config.expert_cache_limit_bytes,
            allocator_headroom_bytes=int(config.allocator_headroom_bytes),
        )

    @classmethod
    def open(
        cls,
        root: Path | str,
        manifest_source: ExpertManifest | Path | str,
        config: ExpertStreamingConfig,
        *,
        spec: ExpertStreamingModelSpec | None = None,
        buffer_allocator: Callable[[int, str], Any] | None = None,
        device_synchronize: Callable[[], None] | None = None,
        apply_memory_cap: bool = True,
        mx_module: Any | None = None,
        env: dict[str, str] | None = None,
    ) -> ExpertStreamingRuntime:
        artifact_root = Path(root).resolve()
        model_spec = get_model_spec(config.model_key) if spec is None else spec
        if model_spec.key != config.model_key:
            raise ExpertStreamingConfigurationError("config and spec model keys differ")
        manifest = (
            manifest_source
            if isinstance(manifest_source, ExpertManifest)
            else load_expert_manifest(manifest_source)
        )
        cls._validate_manifest_identity(manifest, model_spec)
        integrity_report = None
        if config.verify_artifact_headers or config.verify_sidecar_hash_at_open:
            if config.verify_sidecar_hash_at_open and manifest.sidecar is None:
                raise ExpertStreamingConfigurationError(
                    "verified-sidecar mode requires a sidecar manifest"
                )
            integrity_report = verify_expert_manifest(
                manifest,
                artifact_root,
                verify_sidecar_hash=config.verify_sidecar_hash_at_open,
            )
        plan = config.memory_plan(model_spec)
        if not plan.fits_fixed:
            raise ExpertStreamingConfigurationError(
                f"fixed expert-streaming footprint exceeds limit by {-plan.unallocated_bytes} bytes"
            )
        if not plan.fits_headroom:
            shortfall = plan.allocator_headroom_bytes - plan.unallocated_bytes
            raise ExpertStreamingConfigurationError(
                "expert-streaming allocator headroom exceeds the remaining "
                f"memory limit by {shortfall} bytes"
            )
        cap_report = (
            apply_mlx_memory_cap(plan, mx_module=mx_module, env=env)
            if apply_memory_cap
            else None
        )
        pipeline_ledger = _pipeline_ledger_for_config(config)
        pipeline_kwargs = (
            {} if pipeline_ledger is None else {"pipeline_ledger": pipeline_ledger}
        )
        reader = PositionalExpertReader(
            artifact_root,
            max_open_files=config.max_open_files,
            max_read_chunk_bytes=config.max_read_chunk_bytes,
            bypass_page_cache=config.bypass_page_cache,
            **pipeline_kwargs,
        )
        try:
            slots = ExpertSlotPool(
                model_spec,
                plan,
                manifest,
                reader,
                buffer_allocator=buffer_allocator,
                max_inflight_io_bytes=config.max_inflight_io_bytes,
                prefer_sidecar=config.prefer_sidecar,
                verify_hashes=(
                    config.verify_record_hashes
                    and not config.verify_sidecar_hash_at_open
                ),
                device_synchronize=device_synchronize,
                cache_scope=config.cache_scope,
                resource_telemetry=config.resource_telemetry,
                lazy_persistent_buffers=config.dynamic_expert_cache,
                **pipeline_kwargs,
            )
        except Exception:
            reader.close()
            raise
        memory_broker = None
        if config.dynamic_expert_cache:
            try:
                memory_broker = cls._initialize_dynamic_memory_broker(
                    config=config,
                    spec=model_spec,
                    plan=plan,
                    slots=slots,
                    mx_module=mx_module,
                )
            except BaseException:
                slots.close()
                reader.close()
                raise
        return cls(
            artifact_root,
            model_spec,
            config,
            manifest,
            plan,
            reader,
            slots,
            memory_cap_report=cap_report,
            integrity_report=integrity_report,
            memory_broker=memory_broker,
            mx_module=mx_module,
            **pipeline_kwargs,
        )

    @staticmethod
    def _validate_manifest_identity(
        manifest: ExpertManifest,
        spec: ExpertStreamingModelSpec,
    ) -> None:
        errors: list[str] = []
        if manifest.model_key != spec.key:
            errors.append("model key")
        if manifest.source_repo != spec.quant_model:
            errors.append("source repository")
        if manifest.source_revision != spec.quant_revision:
            errors.append("source revision")
        if manifest.quant_bits != spec.quant_bits:
            errors.append("quantization bits")
        if manifest.quant_group_size != spec.quant_group_size:
            errors.append("quantization group size")
        if manifest.artifact_tensor_bytes != spec.total_tensor_bytes:
            errors.append("artifact tensor bytes")
        if errors:
            raise ExpertStreamingConfigurationError(
                "manifest does not match pinned model descriptor: " + ", ".join(errors)
            )

    def _record_cleanup_error(self, error: BaseException) -> None:
        with self._cleanup_error_lock:
            if self._cleanup_error is None:
                self._cleanup_error = error

    def _raise_cleanup_error(self) -> None:
        with self._cleanup_error_lock:
            error = self._cleanup_error
        if error is not None:
            raise ExpertSlotError("expert streaming runtime cleanup failed") from error

    def _raise_if_unhealthy(self) -> None:
        self.slots.raise_if_unhealthy()
        self._raise_cleanup_error()

    def retain_pending_physical_kv_cache(self, cache: Any) -> None:
        """Own a retryable Q4 close after a generation scope unwinds."""

        if getattr(cache, "_closed", False):
            return
        condition = self._physical_kv_drain_condition()
        with condition:
            if not self._closed:
                self._pending_physical_kv_caches[id(cache)] = cache
                return

        # The terminal handoff gate is already sealed.  Retry synchronously so
        # a generation unwinding after shutdown cannot silently attach a new
        # owner to a closed runtime.  A second failure remains retained for an
        # explicit close retry and makes runtime health fail closed.
        try:
            cache.close()
        except BaseException as exc:
            with condition:
                self._pending_physical_kv_caches.setdefault(id(cache), cache)
            self._record_cleanup_error(exc)

    def _physical_kv_drain_condition(self) -> threading.Condition:
        """Return the drain condition, including for lightweight test runtimes."""

        condition = getattr(self, "_pending_physical_kv_condition", None)
        if condition is not None:
            return condition
        lock = self._pending_physical_kv_lock
        with lock:
            condition = getattr(self, "_pending_physical_kv_condition", None)
            if condition is None:
                condition = threading.Condition(lock)
                self._pending_physical_kv_condition = condition
                self._pending_physical_kv_drainers = 0
        return condition

    def _drain_pending_physical_kv_caches(
        self,
        *,
        wait_for_inflight: bool = False,
        deadline: float | None = None,
    ) -> None:
        lock = getattr(self, "_pending_physical_kv_lock", None)
        if lock is None:
            return
        condition = self._physical_kv_drain_condition()
        with condition:
            while wait_for_inflight and self._pending_physical_kv_drainers:
                if deadline is None:
                    condition.wait()
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    raise TimeoutError(
                        "pending physical KV cache drain exceeded close deadline"
                    )
                condition.wait(timeout=remaining)
            pending = tuple(self._pending_physical_kv_caches.items())
            # Transfer retry ownership to this drainer before touching a
            # cache.  Close, admission, and growth may all arrive here at the
            # same time; letting two callers snapshot the same cache can
            # invert cache-close and memory-transaction locks.
            self._pending_physical_kv_caches.clear()
            if pending:
                self._pending_physical_kv_drainers += 1
        if not pending:
            return
        first_error: BaseException | None = None
        try:
            for identity, cache in pending:
                try:
                    if deadline is None:
                        cache.close()
                    else:
                        cache.close(deadline=deadline)
                except BaseException as exc:
                    if not getattr(cache, "_closed", False):
                        with condition:
                            self._pending_physical_kv_caches.setdefault(identity, cache)
                    if first_error is None:
                        first_error = exc
                    continue
                with condition:
                    if self._pending_physical_kv_caches.get(identity) is cache:
                        self._pending_physical_kv_caches.pop(identity, None)
        finally:
            with condition:
                self._pending_physical_kv_drainers -= 1
                condition.notify_all()
        if first_error is not None:
            raise first_error

    def _seal_physical_kv_handoffs(self, *, deadline: float | None) -> None:
        """Drain and atomically close the retained-cache ownership gate."""

        condition = self._physical_kv_drain_condition()
        while True:
            self._drain_pending_physical_kv_caches(
                wait_for_inflight=True,
                deadline=deadline,
            )
            with condition:
                if (
                    self._pending_physical_kv_drainers
                    or self._pending_physical_kv_caches
                ):
                    continue
                self._closed = True
                return

    def admit_kv_tokens(self, tokens: int) -> KVAdmission:
        # Lifecycle order is close -> slot/runtime health -> KV accounting.
        # Release takes only the KV lock so an existing lease can always drain.
        with self._close_lock:
            if self._closed:
                raise ExpertSlotError("expert streaming runtime is closed")
            if self._closing:
                raise ExpertSlotError("expert streaming runtime is closing")
            self._raise_if_unhealthy()
            self._drain_pending_physical_kv_caches()
            count = _integer("tokens", tokens, minimum=1)
            with self._kv_lock:
                requested = self._live_kv_tokens + count
                if requested > self.config.max_live_kv_tokens:
                    raise ExpertStreamingConfigurationError(
                        f"live KV admission {requested} exceeds planned "
                        f"{self.config.max_live_kv_tokens} tokens"
                    )
                self._live_kv_tokens = requested
                self._live_kv_peak = max(self._live_kv_peak, requested)
            return KVAdmission(self, count)

    def _grow_kv_admission(self, admission: KVAdmission, tokens: int) -> None:
        """Atomically extend one live logical-KV lease up to its planned limit."""

        if not isinstance(admission, KVAdmission) or admission.runtime is not self:
            raise TypeError("KV admission does not belong to this runtime")
        target = _integer("tokens", tokens, minimum=1)
        with self._close_lock:
            if self._closed:
                raise ExpertSlotError("expert streaming runtime is closed")
            if self._closing:
                raise ExpertSlotError("expert streaming runtime is closing")
            self._raise_if_unhealthy()
            self._drain_pending_physical_kv_caches()
            with self._kv_lock:
                if admission.released:
                    raise RuntimeError("released KV admission cannot grow")
                current = int(admission.tokens)
                if target < current:
                    raise ValueError("KV admission growth must be monotonic")
                delta = target - current
                if delta == 0:
                    return
                requested = self._live_kv_tokens + delta
                if requested > self.config.max_live_kv_tokens:
                    raise ExpertStreamingConfigurationError(
                        f"live KV admission {requested} exceeds planned "
                        f"{self.config.max_live_kv_tokens} tokens"
                    )
                self._live_kv_tokens = requested
                self._live_kv_peak = max(self._live_kv_peak, requested)
                admission.tokens = target

    def _release_kv_admission(self, admission: KVAdmission) -> None:
        """Release a possibly grown lease exactly once under the KV lock."""

        if not isinstance(admission, KVAdmission) or admission.runtime is not self:
            raise TypeError("KV admission does not belong to this runtime")
        with self._kv_lock:
            if admission.released:
                return
            count = int(admission.tokens)
            if count > self._live_kv_tokens:
                raise RuntimeError("KV admission accounting underflow")
            self._live_kv_tokens -= count
            admission.released = True

    def release_kv_tokens(self, tokens: int) -> None:
        count = _integer("tokens", tokens, minimum=1)
        with self._kv_lock:
            if count > self._live_kv_tokens:
                raise RuntimeError("KV admission accounting underflow")
            self._live_kv_tokens -= count

    def ensure_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute:
        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        normalized_experts = tuple(expert_ids)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc

        if self.memory_broker is not None and self._global_bank is not None:
            ready = self.try_all_hit_route(
                layer,
                normalized_experts,
                phase=phase,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
            )
            if ready is not None:
                return ready
            with self._memory_transaction_lock, lock:
                self._raise_if_unhealthy()
                self._warm_direct_cache_records(
                    layer=layer,
                    expert_ids=normalized_experts,
                    phase=phase,
                )
                return self._ensure_route_locked(
                    layer,
                    normalized_experts,
                    phase=phase,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )

        with lock:
            self._raise_if_unhealthy()
            return self._ensure_route_locked(
                layer,
                normalized_experts,
                phase=phase,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
            )

    def _ensure_route_locked(
        self,
        layer: int,
        expert_ids: tuple[int, ...],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None,
        deadline_ns: int | None,
    ) -> ReadyRoute:
        """Execute a route while its policy lock is already held."""

        route_plan, policy_txn = self._plan_route_transaction(
            layer,
            expert_ids,
            phase=phase,
        )
        ready: ReadyRoute | None = None
        io_admission = RouteIOAdmission()
        try:
            ready = self.slots.ensure_route(
                layer,
                route_plan,
                cancel_event=cancel_event,
                deadline_ns=deadline_ns,
                io_admission=io_admission,
            )
            policy_txn.commit()
        except BaseException as exc:
            if ready is not None:
                ready.release(synchronize=False)
            self._handle_route_failure(
                layer,
                route_plan,
                policy_txn,
                exc,
                io_admission=io_admission,
            )
            raise
        self._observe_plan(layer, route_plan)
        self._observe_dynamic_cache_plan(route_plan)
        assert ready is not None
        return ready

    def _warm_direct_cache_records(
        self,
        *,
        layer: int,
        expert_ids: tuple[int, ...],
        phase: RoutingPhase | str,
    ) -> int:
        """Allocate only the inactive direct records demanded by one miss."""

        bank = self._global_bank
        broker = self.memory_broker
        if bank is None or broker is None:
            return 0
        demand = bank.growth_demand(layer, expert_ids, phase=phase)
        grown = 0
        for slot_id in bank.inactive_slot_ids()[:demand]:
            ticket = broker.plan_expert_cache_growth()
            if ticket is None:
                break
            bank.preflight_activate_slots((slot_id,))
            allocator_before = self._sample_allocator_memory()
            allocated = False
            committed = False
            try:
                self.slots.allocate_persistent_slot(slot_id)
                allocated = True
                allocator_after = self._sample_allocator_memory()
                registered_after = int(
                    self.slots.persistent_cache_telemetry_snapshot()[
                        "physical_bytes"
                    ]
                )
                broker.commit_expert_cache_growth(
                    ticket,
                    registered_cache_bytes_after=registered_after,
                    allocator_before=allocator_before,
                    allocator_after=allocator_after,
                )
                committed = True
                bank.activate_slots((slot_id,))
            except BaseException as growth_error:
                if allocated and not committed:
                    try:
                        self.slots.release_persistent_slots((slot_id,))
                    except BaseException as release_error:
                        raise release_error from growth_error
                if not committed:
                    try:
                        broker.abort_expert_cache_growth(ticket)
                    except BaseException as abort_error:
                        raise abort_error from growth_error
                raise
            with self._dynamic_cache_lock:
                self._dynamic_cache_metrics["record_allocations"] += 1
            grown += 1
        return grown

    def _observe_dynamic_cache_plan(self, plan: RoutePlan) -> None:
        evictions = len(plan.evictions)
        if not evictions or self.memory_broker is None:
            return
        with self._dynamic_cache_lock:
            self._dynamic_cache_metrics["record_reuses"] += evictions
            self._dynamic_cache_metrics["record_evictions"] += evictions

    def try_all_hit_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> ReadyRoute | None:
        """Pin one fully resident route without wave or split execution.

        The policy probe is side-effect free when any assignment misses,
        allowing the caller to use the regular split route unchanged.
        """

        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            planned = (
                self._global_bank.try_plan_all_hits_transaction(
                    layer,
                    expert_ids,
                    phase=phase,
                )
                if self._global_bank is not None
                else self._banks[layer].try_plan_all_hits_transaction(
                    expert_ids,
                    phase=phase,
                )
            )
            if planned is None:
                return None
            route_plan, policy_txn = planned
            ready: ReadyRoute | None = None

            def publish_route() -> None:
                self._publish_route_transaction(layer, route_plan, policy_txn)

            try:
                ready = self.slots.ensure_route(
                    layer,
                    route_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                self.slots.commit_if_healthy(publish_route)
            except BaseException:
                # A successful all-hit probe has no loads and therefore can
                # never cross the destructive I/O boundary.  Any pin-path
                # failure must restore its decode history and epoch exactly.
                if ready is not None:
                    try:
                        ready.release(synchronize=False)
                    except BaseException as cleanup_error:
                        self._record_cleanup_error(cleanup_error)
                        try:
                            ready.release(synchronize=False)
                        except BaseException as retry_error:
                            self._record_cleanup_error(retry_error)
                try:
                    policy_txn.rollback_publication()
                except BaseException as rollback_error:
                    self._record_cleanup_error(rollback_error)
                raise
            assert ready is not None
            return ready

    def _observe_plan(self, layer: int, plan: RoutePlan) -> None:
        with self._counter_lock:
            self._observe_plan_unlocked(layer, plan)

    def _observe_plan_unlocked(self, layer: int, plan: RoutePlan) -> None:
        self.counters.observe(
            plan,
            expert_record_bytes=self.spec.expert_record_bytes,
        )
        self._layer_counters[layer].observe(
            plan,
            expert_record_bytes=self.spec.expert_record_bytes,
        )
        self._phase_counters[plan.phase].observe(
            plan,
            expert_record_bytes=self.spec.expert_record_bytes,
        )

    def _observe_incremental_unlocked(self, *, routes: int, parts: int) -> None:
        self._incremental_miss_routes += routes
        self._incremental_miss_parts += parts

    def _publish_route_transaction(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        *,
        incremental_parts: int = 0,
    ) -> None:
        counters = (
            self.counters,
            self._layer_counters[layer],
            self._phase_counters[plan.phase],
        )
        with self._counter_lock:
            counter_snapshots = tuple(counter.__dict__.copy() for counter in counters)
            incremental_snapshot = (
                self._incremental_miss_routes,
                self._incremental_miss_parts,
            )
            try:
                self._observe_plan_unlocked(layer, plan)
                if plan.phase is RoutingPhase.DECODE and plan.misses:
                    self._observe_incremental_unlocked(
                        routes=1,
                        parts=incremental_parts,
                    )
                # Publish policy last.  Counter observation is reversible and
                # cannot expose a partial snapshot while this lock is held.
                # Deferring the commit also means an all-hit global route does
                # not need to copy the entire LRU merely to undo a later
                # counter failure.
                policy_txn.commit()
            except BaseException:
                for counter, snapshot in zip(counters, counter_snapshots, strict=True):
                    counter.__dict__.clear()
                    counter.__dict__.update(snapshot)
                (
                    self._incremental_miss_routes,
                    self._incremental_miss_parts,
                ) = incremental_snapshot
                try:
                    policy_txn.rollback_publication()
                except BaseException as rollback_error:
                    self._record_cleanup_error(rollback_error)
                raise
        self._observe_dynamic_cache_plan(plan)

    def _plan_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> RoutePlan:
        if self._global_bank is not None:
            return self._global_bank.plan(layer, expert_ids, phase=phase)
        return self._banks[layer].plan(expert_ids, phase=phase)

    def _plan_route_transaction(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
    ) -> tuple[RoutePlan, RoutePolicyTxn]:
        if self._global_bank is not None:
            return self._global_bank.plan_transaction(
                layer,
                expert_ids,
                phase=phase,
            )
        return self._banks[layer].plan_transaction(expert_ids, phase=phase)

    def _invalidate_policy_expert(self, layer: int, expert: int) -> int | None:
        if self._global_bank is not None:
            return self._global_bank.invalidate_expert(layer, expert)
        return self._banks[layer].invalidate_expert(expert)

    @staticmethod
    def _subset_route_plan(
        plan: RoutePlan,
        *,
        hits: bool,
    ) -> RoutePlan | None:
        hit_set = set(plan.hits)
        selected_indices = [
            index
            for index, expert in enumerate(plan.experts)
            if (expert in hit_set) is hits
        ]
        selected = tuple(
            (plan.experts[index], plan.slots[index]) for index in selected_indices
        )
        if not selected:
            return None
        return RoutePlan(
            phase=plan.phase,
            experts=tuple(expert for expert, _slot in selected),
            slots=tuple(slot for _expert, slot in selected),
            hits=plan.hits if hits else (),
            misses=() if hits else plan.misses,
            loads=() if hits else plan.loads,
            evictions=() if hits else plan.evictions,
            generations=(
                tuple(plan.generations[index] for index in selected_indices)
                if plan.generations
                else ()
            ),
        )

    @staticmethod
    def _miss_route_parts(plan: RoutePlan) -> tuple[RoutePlan, ...]:
        """Split a miss plan by expert while preserving assignment duplicates."""

        unique_experts = tuple(dict.fromkeys(plan.experts))
        load_experts = tuple(load.expert for load in plan.loads)
        if len(set(load_experts)) != len(load_experts) or set(load_experts) != set(
            unique_experts
        ):
            raise ExpertSlotError(
                "incremental miss experts and slot loads must match one-to-one"
            )
        if len({load.slot for load in plan.loads}) != len(plan.loads):
            raise ExpertSlotError("incremental miss parts must own disjoint slots")
        parts: list[RoutePlan] = []
        for expert in unique_experts:
            positions = tuple(
                index
                for index, candidate in enumerate(plan.experts)
                if candidate == expert
            )
            loads = tuple(load for load in plan.loads if load.expert == expert)
            if len(loads) != 1:
                raise ExpertSlotError(
                    "each incremental miss expert must own exactly one slot load"
                )
            parts.append(
                RoutePlan(
                    phase=plan.phase,
                    experts=tuple(plan.experts[index] for index in positions),
                    slots=tuple(plan.slots[index] for index in positions),
                    hits=(),
                    misses=(expert,),
                    loads=loads,
                    evictions=tuple(
                        eviction
                        for eviction in plan.evictions
                        if eviction.next_expert == expert
                    ),
                    generations=(
                        tuple(plan.generations[index] for index in positions)
                        if plan.generations
                        else ()
                    ),
                )
            )
        return tuple(parts)

    def _rollback_route_loads(self, layer: int, plan: RoutePlan) -> None:
        for load in plan.loads:
            if load.persistent:
                self._invalidate_policy_expert(layer, load.expert)
            try:
                self.slots.invalidate(
                    layer,
                    load.slot,
                    expert=load.expert,
                    generation=load.generation,
                )
            except ExpertSlotError:
                pass

    def _handle_route_failure(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        error: BaseException,
        *,
        io_admission: RouteIOAdmission | None = None,
    ) -> None:
        rollback_safe = (
            not io_admission.any_accepted
            if io_admission is not None
            else (
                isinstance(error, ExpertCompletionFenceError)
                and error.policy_rollback_safe
            )
        )
        if rollback_safe:
            policy_txn.rollback_completion()
            return
        self._rollback_route_loads(layer, plan)

    def _handle_split_route_failure(
        self,
        layer: int,
        plan: RoutePlan,
        policy_txn: RoutePolicyTxn,
        error: BaseException,
        *,
        accepted_parts: tuple[RoutePlan, ...],
        io_admission: RouteIOAdmission | None,
    ) -> None:
        """Restore untouched victims while quarantining accepted split loads."""

        if io_admission is None or not io_admission.any_accepted:
            policy_txn.rollback_completion()
            return
        if not accepted_parts:
            self._handle_route_failure(
                layer,
                plan,
                policy_txn,
                error,
                io_admission=io_admission,
            )
            return

        # Every submitted future has settled before this runs. Remove only the
        # physical records that crossed their part-local admission boundary,
        # then restore the full policy snapshot. Accepted evictions cannot be
        # restored physically, so quarantine those victims again afterward.
        for part in accepted_parts:
            self._rollback_route_loads(layer, part)
        policy_txn.rollback_completion()
        for part in accepted_parts:
            for eviction in part.evictions:
                previous_layer = (
                    layer
                    if eviction.previous_layer is None
                    else eviction.previous_layer
                )
                self._invalidate_policy_expert(
                    previous_layer,
                    eviction.previous_expert,
                )
        if self._global_bank is not None:
            for part in accepted_parts:
                for load in part.loads:
                    if load.persistent and load.generation is not None:
                        self._global_bank.reconcile_slot_generation(
                            load.slot,
                            load.generation,
                        )

    def begin_split_route(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        phase: RoutingPhase | str,
        cancel_event: threading.Event | None = None,
        deadline_ns: int | None = None,
    ) -> PendingSplitRoute:
        """Pin hits now and load misses while the caller evaluates hit work."""

        if self._closed:
            raise ExpertSlotError("expert streaming runtime is closed")
        if self._closing:
            raise ExpertSlotError("expert streaming runtime is closing")
        normalized_experts = tuple(expert_ids)
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        planned: tuple[RoutePlan, RoutePolicyTxn] | None = None
        memory_lock_held = False
        if self.memory_broker is not None and self._global_bank is not None:
            lock.acquire()
            try:
                self._raise_if_unhealthy()
                planned = self._global_bank.try_plan_all_hits_transaction(
                    layer,
                    normalized_experts,
                    phase=phase,
                )
            except BaseException:
                lock.release()
                raise
            if planned is None:
                lock.release()
                self._memory_transaction_lock.acquire()
                memory_lock_held = True
                try:
                    lock.acquire()
                    self._raise_if_unhealthy()
                    self._warm_direct_cache_records(
                        layer=layer,
                        expert_ids=normalized_experts,
                        phase=phase,
                    )
                except BaseException:
                    lock.release()
                    self._memory_transaction_lock.release()
                    raise
        else:
            lock.acquire()
        plan: RoutePlan | None = None
        policy_txn: RoutePolicyTxn | None = None
        hit_ready: ReadyRoute | None = None
        pending: PendingSplitRoute | None = None
        pipeline_route: ExpertPipelineRoute | None = None
        io_admission = RouteIOAdmission()
        miss_cancel_event = threading.Event()
        combined_cancel = _RouteCancel(cancel_event, miss_cancel_event)
        try:
            self._raise_if_unhealthy()
            plan, policy_txn = (
                planned
                if planned is not None
                else self._plan_route_transaction(
                    layer,
                    normalized_experts,
                    phase=phase,
                )
            )
            hit_plan = self._subset_route_plan(plan, hits=True)
            miss_plan = self._subset_route_plan(plan, hits=False)
            miss_parts = (
                self._miss_route_parts(miss_plan)
                if miss_plan is not None and plan.phase is RoutingPhase.DECODE
                else ((miss_plan,) if miss_plan is not None else ())
            )
            pipeline_ledger = self._pipeline_ledger
            if pipeline_ledger is not None:
                try:
                    load_experts = tuple(
                        dict.fromkeys(
                            load.expert
                            for load in (() if miss_plan is None else miss_plan.loads)
                        )
                    )
                    pipeline_route = pipeline_ledger.begin_route(
                        layer=layer,
                        phase=plan.phase,
                        load_experts=load_experts,
                        load_logical_bytes=tuple(
                            self.manifest.record(layer, expert).logical_bytes
                            for expert in load_experts
                        ),
                    )
                except Exception:
                    pipeline_route = None
                    try:
                        pipeline_ledger.mark_incomplete(phase=plan.phase)
                    except Exception:
                        pass
            hit_ready = (
                self.slots.ensure_route(
                    layer,
                    hit_plan,
                    cancel_event=cancel_event,
                    deadline_ns=deadline_ns,
                )
                if hit_plan is not None
                else None
            )
            pending = PendingSplitRoute(
                runtime=self,
                layer=layer,
                plan=plan,
                layer_lock=lock,
                hit_ready=hit_ready,
                miss_futures={},
                policy_txn=policy_txn,
                io_admission=io_admission,
                miss_cancel_event=miss_cancel_event,
                miss_parts=miss_parts,
                pipeline_route=pipeline_route,
            )
            if miss_plan is not None:
                ensure = (
                    self.slots.ensure_route_part
                    if plan.phase is RoutingPhase.DECODE
                    else self.slots.ensure_route
                )
                for ordinal, miss_part in enumerate(miss_parts):
                    part_admission = io_admission.child()
                    if pipeline_route is None:
                        future = self._split_executor.submit(
                            ensure,
                            layer,
                            miss_part,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                            io_admission=part_admission,
                            route_admitted=pending._retain_lifecycle_after_admission,
                        )
                    else:
                        future = self._split_executor.submit(
                            ensure,
                            layer,
                            miss_part,
                            cancel_event=combined_cancel,
                            deadline_ns=deadline_ns,
                            io_admission=part_admission,
                            route_admitted=pending._retain_lifecycle_after_admission,
                            pipeline_route=pipeline_route,
                        )
                    pending._attach_miss_future(
                        future,
                        miss_part,
                        ordinal=ordinal,
                        io_admission=part_admission,
                    )
            else:
                pending._commit_policy()
            return pending
        except BaseException as setup_error:
            # Mirror the sync-path rollback: without it, a failed hit pin or
            # submit leaves the bank mapping experts to never-loaded slots,
            # wedging every later route on this layer until reset().
            miss_cancel_event.set()
            if pending is not None:
                pending.abort(setup_error)
                pending.close()
            else:
                if pipeline_route is not None:
                    _pipeline_call(
                        self._pipeline_ledger,
                        pipeline_route,
                        "close",
                    )
                if hit_ready is not None:
                    try:
                        hit_ready.release(synchronize=False)
                    except BaseException:
                        pass
                if policy_txn is not None:
                    try:
                        self._handle_route_failure(
                            layer,
                            plan,
                            policy_txn,
                            setup_error,
                            io_admission=io_admission,
                        )
                    except BaseException:
                        pass
                lock.release()
            raise
        finally:
            if memory_lock_held:
                self._memory_transaction_lock.release()

    def _sample_allocator_memory(self) -> AllocatorMemorySample:
        previous: AllocatorMemorySample | None = None
        for _ in range(_ALLOCATOR_STABLE_SAMPLE_READS):
            telemetry = mlx_memory_telemetry(self._mx_module)
            try:
                sample = AllocatorMemorySample(
                    active_bytes=int(telemetry["active_memory_bytes"]),
                    cache_bytes=int(telemetry["cache_memory_bytes"]),
                    peak_bytes=int(telemetry["peak_memory_bytes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MemoryTelemetryError(
                    "MLX allocator active/cache/peak telemetry is unavailable"
                ) from exc
            if min(sample.active_bytes, sample.cache_bytes, sample.peak_bytes) < 0:
                raise MemoryTelemetryError(
                    "MLX allocator active/cache/peak telemetry is invalid"
                )
            if sample.peak_bytes < sample.active_bytes:
                raise MemoryTelemetryError("MLX allocator peak is below active memory")
            if previous is not None:
                if sample.peak_bytes < previous.peak_bytes:
                    raise MemoryTelemetryError(
                        "MLX allocator peak regressed during telemetry sampling"
                    )
                if (
                    sample.charged_footprint_bytes == previous.charged_footprint_bytes
                    and sample.peak_bytes == previous.peak_bytes
                ):
                    self._last_allocator_sample = sample
                    return sample
            previous = sample
        raise MemoryTelemetryError(
            "MLX allocator telemetry did not stabilize after "
            f"{_ALLOCATOR_STABLE_SAMPLE_READS} reads"
        )

    def reconcile_post_load_memory(self) -> None:
        """Reclassify startup reserve into measured resident/MTP truth."""

        broker = self.memory_broker
        if broker is None:
            return
        before = broker.snapshot()
        allocator_before = self._last_allocator_sample
        sample = self._sample_allocator_memory()
        expert_bytes = self._registered_expert_cache_bytes()
        transient_bytes = int(self.plan.transient_bytes)
        staging_bytes = transient_bytes + int(getattr(self.plan, "io_staging_bytes", 0))
        execution_workspace = int(self.plan.execution_workspace_bytes)
        resident_bytes = max(
            before.resident_model_bytes,
            int(sample.active_bytes) - expert_bytes - transient_bytes,
        )
        resident_growth = resident_bytes - before.resident_model_bytes
        generic_runtime_reserve = max(
            0,
            before.runtime_workspace_bytes - execution_workspace,
        )
        remaining_runtime_reserve = max(
            0,
            generic_runtime_reserve - resident_growth,
        )
        admission_error: MemoryAdmissionError | None = None
        try:
            broker.reconcile_post_load_classification(
                resident_model_bytes=resident_bytes,
                expert_cache_physical_bytes=expert_bytes,
                in_flight_expert_staging_bytes=staging_bytes,
                runtime_workspace_bytes=(
                    execution_workspace + remaining_runtime_reserve
                ),
                allocator_before=allocator_before,
                allocator_after=sample,
            )
        except MemoryAdmissionError as exc:
            admission_error = exc
        if resident_growth > generic_runtime_reserve:
            raise ExpertStreamingConfigurationError(
                "post-load resident growth exceeded the planned runtime reserve"
            )
        if admission_error is not None:
            raise ExpertStreamingConfigurationError(
                "post-load physical pools exceeded the dynamic memory budget"
            ) from admission_error

    def dynamic_memory_telemetry_snapshot(
        self,
        *,
        now_ns: int | None = None,
    ) -> dict[str, int | bool | str | None]:
        """Return bounded broker, direct-cache, and allocator telemetry."""

        del now_ns
        broker = self.memory_broker
        if broker is None:
            return {}
        snapshot = broker.snapshot()
        cache = self.slots.persistent_cache_telemetry_snapshot()
        bank = self._global_bank
        allocator = self._last_allocator_sample
        with self._dynamic_cache_lock:
            metrics = dict(self._dynamic_cache_metrics)
        record_bytes = int(self.spec.expert_record_bytes)
        pinned_expert_bytes = int(cache["pinned_record_count"]) * record_bytes
        speculative_expert_bytes = int(
            getattr(bank, "speculative_record_count", 0)
        ) * record_bytes
        return {
            "memory_limit_bytes": broker.budget.memory_limit_bytes,
            "allocator_headroom_bytes": broker.budget.allocator_headroom_bytes,
            "classified_limit_bytes": broker.budget.classified_limit_bytes,
            "classified_bytes": snapshot.classified_bytes,
            "charged_bytes": snapshot.charged_bytes,
            "charged_residual_bytes": (
                broker.budget.memory_limit_bytes - snapshot.charged_bytes
            ),
            "expert_cache_limit_bytes": self.config.expert_cache_limit_bytes,
            "logical_expert_records": int(cache["logical_record_capacity"]),
            "active_expert_records": int(
                getattr(bank, "active_capacity", cache["allocated_record_count"])
            ),
            "resident_expert_records": int(getattr(bank, "occupancy", 0)),
            "expert_cache_physical_bytes": snapshot.expert_cache_physical_bytes,
            "pinned_expert_bytes": pinned_expert_bytes,
            "in_flight_expert_bytes": (
                int(cache["in_flight_record_count"]) * record_bytes
            ),
            "speculative_expert_bytes": speculative_expert_bytes,
            **cache,
            "record_allocations": int(metrics["record_allocations"]),
            "record_reuses": int(metrics["record_reuses"]),
            "record_evictions": int(metrics["record_evictions"]),
            "record_releases": int(metrics["record_releases"]),
            "allocator_active_bytes": (
                None if allocator is None else allocator.active_bytes
            ),
            "allocator_cache_bytes": (
                None if allocator is None else allocator.cache_bytes
            ),
            "allocator_peak_bytes": (
                None if allocator is None else allocator.peak_bytes
            ),
            "failed_closed": snapshot.failed_closed,
            "failure_reason": snapshot.failure_reason,
        }

    def _registered_expert_cache_bytes(self) -> int:
        snapshot = self.slots.persistent_cache_telemetry_snapshot()
        try:
            return int(snapshot["physical_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryTelemetryError(
                "expert cache physical-byte telemetry is unavailable"
            ) from exc

    def _route_resize_context(self):
        locks = getattr(self, "_layer_locks", {})
        route_lock = next(iter(locks.values()), None)
        return nullcontext() if route_lock is None else route_lock

    def reserve_growth(
        self,
        *,
        cache_id: str,
        steady_delta_bytes: int,
        transient_delta_bytes: int,
    ) -> KVAllocationTicket:
        """Reserve physical Q4 KV growth and release expert records if required."""

        broker = self.memory_broker
        if broker is None:
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        transaction_lock = self._memory_transaction_lock
        transaction_lock.acquire()
        ticket: KVAllocationTicket | None = None
        handed_off = False
        try:
            self._drain_pending_physical_kv_caches()
            # Reconcile upward allocator drift before planning.  Sampling only
            # after MLX allocation would discover an unsafe transition only
            # after the configured memory limit had already been crossed.
            broker.reconcile_allocator_cache(self._sample_allocator_memory())
            cache_telemetry = self.slots.persistent_cache_telemetry_snapshot()
            bank = self._global_bank
            broker.reconcile_expert_protection(
                pinned_expert_bytes=(
                    int(cache_telemetry["pinned_record_count"])
                    * int(self.spec.expert_record_bytes)
                ),
                speculative_expert_bytes=(
                    int(getattr(bank, "speculative_record_count", 0))
                    * int(self.spec.expert_record_bytes)
                ),
            )
            ticket = broker.plan_kv_growth(
                cache_id=cache_id,
                steady_delta_bytes=steady_delta_bytes,
                transient_delta_bytes=transient_delta_bytes,
            )
            if ticket.required_expert_reclaim_bytes:
                self.reclaim_expert_records(ticket, _transaction_lock_held=True)
            handed_off = True
            return ticket
        except BaseException as reservation_error:
            if (
                ticket is not None
                and broker.snapshot().pending_kv_ticket_id is not None
            ):
                try:
                    broker.abort_kv_growth(ticket, observed_kv_delta_bytes=0)
                except BaseException as abort_error:
                    raise abort_error from reservation_error
            raise
        finally:
            if not handed_off:
                transaction_lock.release()

    def reserve_growth_group(
        self,
        *,
        members: Iterable[tuple[str, int, int]],
    ) -> KVGroupGrowthContext:
        """Reserve one aggregate KV permit across a known set of cache owners."""

        broker = self.memory_broker
        if broker is None:
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        active_group = getattr(self, "_active_kv_growth_group", None)
        if active_group is not None and active_group._active:
            raise RuntimeError("another KV growth group is active")
        member_specs = tuple(members)
        transaction_lock = self._memory_transaction_lock
        transaction_lock.acquire()
        ticket: KVAllocationGroupTicket | None = None
        handed_off = False
        try:
            if getattr(self, "_closed", False) or getattr(self, "_closing", False):
                raise ExpertSlotError("expert streaming runtime is closing")
            self._drain_pending_physical_kv_caches()
            broker.reconcile_allocator_cache(self._sample_allocator_memory())
            cache_telemetry = self.slots.persistent_cache_telemetry_snapshot()
            bank = self._global_bank
            broker.reconcile_expert_protection(
                pinned_expert_bytes=(
                    int(cache_telemetry["pinned_record_count"])
                    * int(self.spec.expert_record_bytes)
                ),
                speculative_expert_bytes=(
                    int(getattr(bank, "speculative_record_count", 0))
                    * int(self.spec.expert_record_bytes)
                ),
            )
            ticket = broker.plan_kv_growth_group(members=member_specs)
            if ticket.required_expert_reclaim_bytes:
                self.reclaim_expert_records(ticket, _transaction_lock_held=True)
            context = KVGroupGrowthContext(runtime=self, ticket=ticket)
            self._active_kv_growth_group = context
            handed_off = True
            return context
        except BaseException as reservation_error:
            if (
                ticket is not None
                and broker.snapshot().pending_kv_ticket_id == ticket.ticket_id
            ):
                try:
                    broker.abort_kv_growth_group(
                        ticket,
                        observed_uncommitted_kv_delta_bytes=0,
                    )
                except BaseException as abort_error:
                    raise abort_error from reservation_error
            raise
        finally:
            if not handed_off:
                transaction_lock.release()

    def _require_growth_group_context(
        self,
        context: KVGroupGrowthContext,
    ) -> None:
        if not isinstance(context, KVGroupGrowthContext) or context.runtime is not self:
            raise TypeError("KV growth group does not belong to this runtime")
        if context._owner_thread_id != threading.get_ident():
            raise RuntimeError("KV growth group must remain on its reserving thread")
        if not context._active:
            raise RuntimeError("KV growth group is already finalized")

    def _finish_growth_group(
        self,
        context: KVGroupGrowthContext,
        *,
        completed: bool,
    ) -> None:
        if not context._active:
            return
        context._active = False
        context.completed = completed
        context.aborted = not completed
        if getattr(self, "_active_kv_growth_group", None) is context:
            self._active_kv_growth_group = None
        self._memory_transaction_lock.release()

    def _ensure_no_active_growth_group(self) -> None:
        active_group = getattr(self, "_active_kv_growth_group", None)
        if active_group is not None and active_group._active:
            raise RuntimeError(
                "physical KV release or expert regrow is unavailable while a "
                "KV growth group is active"
            )

    def _commit_growth_group_member(
        self,
        context: KVGroupGrowthContext,
        *,
        cache_id: str,
        measured_physical_bytes: int,
        allocator_before: AllocatorMemorySample,
        allocator_after: AllocatorMemorySample,
    ) -> KVPhysicalAllocation:
        """Commit one measured member and release the permit after the last."""

        self._require_growth_group_context(context)
        measured = _integer(
            "measured_physical_bytes",
            measured_physical_bytes,
            minimum=0,
        )
        broker = self.memory_broker
        if broker is None:
            self._finish_growth_group(context, completed=False)
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        try:
            allocation = broker.commit_kv_growth_group_member(
                context.ticket,
                cache_id=cache_id,
                allocated_physical_bytes=measured,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        except BaseException as commit_error:
            if broker.snapshot().pending_kv_ticket_id == context.ticket.ticket_id:
                try:
                    broker.abort_kv_growth_group(
                        context.ticket,
                        observed_uncommitted_kv_delta_bytes=measured,
                        allocator_before=allocator_before,
                        allocator_after=allocator_after,
                    )
                except BaseException as abort_error:
                    self._finish_growth_group(context, completed=False)
                    raise abort_error from commit_error
            self._finish_growth_group(context, completed=False)
            raise
        context._allocations[allocation.cache_id] = allocation
        if not context.remaining_cache_ids:
            self._finish_growth_group(context, completed=True)
        return allocation

    def _abort_growth_group(
        self,
        context: KVGroupGrowthContext,
        *,
        observed_uncommitted_physical_bytes: int | None,
        allocator_before: AllocatorMemorySample | None,
        allocator_after: AllocatorMemorySample | None,
    ) -> BrokerSnapshot:
        """Abort remaining members while preserving prior owned handles."""

        self._require_growth_group_context(context)
        observed = (
            None
            if observed_uncommitted_physical_bytes is None
            else _integer(
                "observed_uncommitted_physical_bytes",
                observed_uncommitted_physical_bytes,
                minimum=0,
            )
        )
        broker = self.memory_broker
        if broker is None:
            self._finish_growth_group(context, completed=False)
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        try:
            return broker.abort_kv_growth_group(
                context.ticket,
                observed_uncommitted_kv_delta_bytes=observed,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        finally:
            self._finish_growth_group(context, completed=False)

    def commit_growth(
        self,
        ticket: KVAllocationTicket,
        *,
        measured_physical_bytes: int,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> KVPhysicalAllocation:
        broker = self.memory_broker
        if broker is None:
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        try:
            return broker.commit_kv_growth(
                ticket,
                allocated_physical_bytes=measured_physical_bytes,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        finally:
            self._memory_transaction_lock.release()

    def abort_growth(
        self,
        ticket: KVAllocationTicket,
        *,
        observed_physical_bytes: int | None,
        allocator_before: AllocatorMemorySample | None = None,
        allocator_after: AllocatorMemorySample | None = None,
    ) -> None:
        broker = self.memory_broker
        if broker is None:
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        try:
            broker.abort_kv_growth(
                ticket,
                observed_kv_delta_bytes=observed_physical_bytes,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        finally:
            self._memory_transaction_lock.release()

    def sample_allocator_memory(self) -> AllocatorMemorySample:
        return self._sample_allocator_memory()

    @contextmanager
    def physical_kv_release_context(self):
        """Serialize the complete Q4 release boundary with growth transactions."""

        with self._memory_transaction_lock:
            self._ensure_no_active_growth_group()
            yield

    def release_cache(
        self,
        *,
        cache_id: str,
        allocations: tuple[KVPhysicalAllocation, ...],
        released_physical_bytes: int,
        allocator_before: AllocatorMemorySample,
        allocator_after: AllocatorMemorySample,
    ) -> None:
        """Reconcile an exact cache-owner release."""

        broker = self.memory_broker
        if broker is None:
            raise MemoryAdmissionError("dynamic KV allocation is not enabled")
        del released_physical_bytes
        with self.physical_kv_release_context():
            broker.release_kv_batch(
                cache_id=cache_id,
                allocations=allocations,
                registered_kv_bytes_after=None,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )

    def reclaim_expert_records(
        self,
        ticket: KVAllocationTicket | KVAllocationGroupTicket,
        *,
        deadline_ns: int | None = None,
        _transaction_lock_held: bool = False,
    ) -> ExpertRecordReleaseResult:
        """Release the coldest exact records needed by one KV reservation."""

        del deadline_ns
        broker = self.memory_broker
        bank = self._global_bank
        if broker is None or bank is None:
            raise MemoryAdmissionError("dynamic expert cache is not enabled")
        required = int(ticket.required_expert_reclaim_bytes)
        if required <= 0:
            return ExpertRecordReleaseResult((), 0)
        record_bytes = int(self.spec.expert_record_bytes)
        record_count = (required + record_bytes - 1) // record_bytes

        transaction_context = (
            nullcontext()
            if _transaction_lock_held
            else self._memory_transaction_lock
        )
        with transaction_context, self._route_resize_context():
            candidates = bank.rank_reclaim_slots(self.slots.protected_slot_ids())
            selected = candidates[:record_count]
            if len(selected) != record_count:
                raise MemoryAdmissionError(
                    "insufficient unpinned expert records for KV growth"
                )
            bank.preflight_deactivate_slots(selected)
            allocator_before = self._sample_allocator_memory()
            bank.deactivate_slots(selected)
            release_error: BaseException | None = None
            result: ExpertRecordReleaseResult | None = None
            try:
                result = self.slots.release_persistent_slots(selected)
            except BaseException as exc:
                release_error = exc

            allocator_after: AllocatorMemorySample | None
            try:
                allocator_after = self._sample_allocator_memory()
            except BaseException:
                allocator_after = None
            try:
                registered_after = int(
                    self.slots.persistent_cache_telemetry_snapshot()[
                        "physical_bytes"
                    ]
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise MemoryTelemetryError(
                    "expert cache physical-byte telemetry is unavailable"
                ) from exc

            try:
                broker.confirm_expert_reclaim(
                    ticket,
                    registered_cache_bytes_after=registered_after,
                    allocator_before=allocator_before,
                    allocator_after=allocator_after,
                )
            except BaseException as accounting_error:
                if release_error is not None:
                    raise accounting_error from release_error
                raise
            if release_error is not None:
                raise release_error
            assert result is not None
            with self._dynamic_cache_lock:
                self._dynamic_cache_metrics["record_releases"] += len(
                    result.slot_ids
                )
            return result

    def route_waves(
        self,
        expert_ids: Iterable[int],
        *,
        sort_unique: bool = False,
    ) -> tuple[RouteWave, ...]:
        return partition_route_waves(
            expert_ids,
            max_unique_experts=self.plan.transient_slots,
            sort_unique=sort_unique,
        )

    def observe_route(
        self,
        layer: int,
        phase: RoutingPhase | str,
        expert_ids: Iterable[int],
        *,
        token_count: int,
    ) -> None:
        if not self.config.trace_routes:
            return
        normalized_phase = RoutingPhase(phase)
        routed_experts = [int(expert) for expert in expert_ids]
        with self._route_trace_lock:
            entry = {
                "layer": int(layer),
                "phase": normalized_phase.value,
                "trace_epoch": self._route_trace_epoch,
                "token_count": int(token_count),
                "expert_ids": routed_experts,
            }
            if normalized_phase is RoutingPhase.DECODE:
                routed_layers = set(self.spec.routed_layer_indices)
                if layer in self._route_trace_decode_layers_seen:
                    # Preserve evidence rather than silently assigning a
                    # duplicate layer to the current step. The analyzer will
                    # reject the resulting incomplete step sequences.
                    self._route_trace_decode_step += 1
                    self._route_trace_decode_layers_seen.clear()
                entry["decode_step"] = self._route_trace_decode_step
                self._route_trace_decode_layers_seen.add(int(layer))
                if self._route_trace_decode_layers_seen == routed_layers:
                    self._route_trace_decode_step += 1
                    self._route_trace_decode_layers_seen.clear()
            self._route_trace.append(entry)

    def route_trace(self) -> list[dict[str, Any]]:
        with self._route_trace_lock:
            return [dict(entry) for entry in self._route_trace]

    def prepare_prefill_seed(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[int, ...]:
        try:
            lock = self._layer_locks[layer]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer} is not routed for {self.spec.key}"
            ) from exc
        with lock:
            self._raise_if_unhealthy()
            if self._global_bank is not None:
                return self._global_bank.prepare_prefill_seed(layer, expert_ids)
            return self._banks[layer].prepare_prefill_seed(expert_ids)

    def reset(self) -> None:
        locks = tuple(dict.fromkeys(self._layer_locks.values()))
        for lock in locks:
            lock.acquire()
        try:
            self._raise_if_unhealthy()
            self.slots.reset()
            if self._global_bank is not None:
                self._global_bank.reset()
            else:
                for bank in self._banks.values():
                    bank.reset()
            with self._counter_lock:
                self.counters = CacheCounters()
                self._layer_counters = {
                    layer: CacheCounters() for layer in self.spec.routed_layer_indices
                }
                self._phase_counters = {
                    phase: CacheCounters() for phase in RoutingPhase
                }
                self._incremental_miss_routes = 0
                self._incremental_miss_parts = 0
            if self.config.trace_routes:
                with self._route_trace_lock:
                    previous_epoch = self._route_trace_epoch
                    self._route_trace_epoch += 1
                    self._route_trace_decode_step = 0
                    self._route_trace_decode_layers_seen.clear()
                    self._route_trace.append(
                        {
                            "phase": "reset",
                            "previous_trace_epoch": previous_epoch,
                            "trace_epoch": self._route_trace_epoch,
                        }
                    )
        finally:
            for lock in reversed(locks):
                lock.release()

    def snapshot(self, *, mx_module: Any | None = None) -> dict[str, Any]:
        with self._kv_lock:
            live_kv = self._live_kv_tokens
            peak_kv = self._live_kv_peak
        with self._counter_lock:
            cache = self.counters.as_dict()
            cache_by_layer = {
                str(layer): counters.as_dict()
                for layer, counters in self._layer_counters.items()
            }
            cache_by_phase = {
                phase.value: counters.as_dict()
                for phase, counters in self._phase_counters.items()
            }
            incremental_misses = {
                "routes": self._incremental_miss_routes,
                "parts": self._incremental_miss_parts,
            }
        # Never hold the counter lock across slot health/fence inspection.
        slots = self.slots.snapshot()
        snapshot = {
            "model_key": self.spec.key,
            "manifest_sha256": self.manifest.manifest_sha256,
            "memory_plan": {
                "total_limit_bytes": self.plan.total_limit_bytes,
                "fixed_bytes": self.plan.fixed_bytes,
                "persistent_cache_bytes": self.plan.persistent_cache_bytes,
                "slots_per_layer": self.plan.slots_per_layer,
                "cache_scope": self.config.cache_scope,
                "global_persistent_slots": (
                    self.plan.persistent_slots
                    if self.config.cache_scope == "global"
                    else None
                ),
                "transient_slots": self.plan.transient_slots,
                "allocated_bytes": self.plan.allocated_bytes,
                "unallocated_bytes": self.plan.unallocated_bytes,
                "allocator_headroom_bytes": self.plan.allocator_headroom_bytes,
                "rounding_residual_bytes": self.plan.rounding_residual_bytes,
                "fits_headroom": self.plan.fits_headroom,
            },
            "memory_cap": self.memory_cap_report,
            "integrity": self.integrity_report,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "live_kv_tokens": live_kv,
            "live_kv_tokens_peak": peak_kv,
            "cache": cache,
            "cache_by_layer": cache_by_layer,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            "slots": slots,
        }
        if self._global_bank is not None:
            snapshot["global_cache"] = {
                **self._global_bank.snapshot(),
                "resident_experts_by_layer": {
                    str(layer): list(experts)
                    for layer, experts in self._global_bank.resident_experts_by_layer.items()
                },
            }
        if self._mapped_expert_store is not None:
            snapshot["mapped_experts"] = self._mapped_expert_store.snapshot()
        if self._pipeline_ledger is not None:
            snapshot["expert_pipeline"] = self._pipeline_ledger.snapshot()
        if self.memory_broker is not None:
            snapshot["memory_broker"] = asdict(self.memory_broker.snapshot())
            snapshot["dynamic_memory"] = self.dynamic_memory_telemetry_snapshot()
        self._raise_if_unhealthy()
        return snapshot

    def resource_telemetry_snapshot(
        self,
        *,
        mx_module: Any | None = None,
    ) -> dict[str, Any]:
        """Return cheap cumulative counters for the benchmark sampler."""

        with self._counter_lock:
            cache = self.counters.as_dict()
            cache_by_layer = {
                str(layer): counters.as_dict()
                for layer, counters in self._layer_counters.items()
            }
            cache_by_phase = {
                phase.value: counters.as_dict()
                for phase, counters in self._phase_counters.items()
            }
            incremental_misses = {
                "routes": self._incremental_miss_routes,
                "parts": self._incremental_miss_parts,
            }
        # Pool occupancy has independent locks; do not hold the counter lock
        # across that snapshot.
        slots = self.slots.resource_telemetry_snapshot()
        snapshot = {
            "model_key": self.spec.key,
            "quant_bits": self.spec.quant_bits,
            "expert_record_bytes": self.spec.expert_record_bytes,
            "mlx_memory": mlx_memory_telemetry(mx_module),
            "cache": cache,
            "cache_by_layer": cache_by_layer,
            "cache_by_phase": cache_by_phase,
            "incremental_misses": incremental_misses,
            **slots,
        }
        if self._pipeline_ledger is not None:
            snapshot["expert_pipeline"] = self._pipeline_ledger.snapshot()
        if self.memory_broker is not None:
            broker_snapshot = self.memory_broker.snapshot()
            physical_bytes = int(broker_snapshot.kv_physical_bytes)
            physical_blocks, partial_block_bytes = divmod(
                physical_bytes,
                HY3_Q4_KV_BLOCK_BYTES,
            )
            with self._kv_lock:
                logical_tokens = int(self._live_kv_tokens)
            snapshot["kv"] = {
                "representation": "q4",
                "logical_tokens": logical_tokens,
                "physical_blocks": (
                    physical_blocks if partial_block_bytes == 0 else None
                ),
                "physical_bytes": physical_bytes,
            }
            snapshot["memory_broker"] = asdict(broker_snapshot)
            snapshot["dynamic_memory"] = self.dynamic_memory_telemetry_snapshot()
        return snapshot

    def close(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        if deadline is None:
            self._close_lock.acquire()
        else:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._close_lock.acquire(timeout=remaining):
                raise TimeoutError(
                    "expert streaming runtime close already in progress at deadline"
                )
        memory_transaction_acquired = False
        try:
            if deadline is None:
                self._memory_transaction_lock.acquire()
                memory_transaction_acquired = True
            else:
                remaining = max(0.0, deadline - time.monotonic())
                if not self._memory_transaction_lock.acquire(timeout=remaining):
                    raise TimeoutError(
                        "expert streaming runtime memory transaction did not "
                        "finish before close deadline"
                    )
                memory_transaction_acquired = True
            self._ensure_no_active_growth_group()
            if not self._closed:
                self._closing = True
            self._memory_transaction_lock.release()
            memory_transaction_acquired = False
            self._drain_pending_physical_kv_caches(
                wait_for_inflight=True,
                deadline=deadline,
            )
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if self._closed:
                slots_error: BaseException | None = None
                try:
                    self.slots.close(timeout=remaining)
                except BaseException as exc:
                    slots_error = exc
                self._seal_physical_kv_handoffs(deadline=deadline)
                if slots_error is not None:
                    raise slots_error
                self._raise_cleanup_error()
                return
            slots_error: BaseException | None = None
            try:
                self.slots.close(timeout=remaining)
            except BaseException as exc:
                if not self.slots._closed:
                    raise
                slots_error = exc
            self._drain_pending_physical_kv_caches(
                wait_for_inflight=True,
                deadline=deadline,
            )
            self._split_executor.shutdown(
                wait=deadline is None,
                cancel_futures=True,
            )
            if self._mapped_expert_store is not None:
                self._mapped_expert_store.close()
                self._mapped_expert_store = None
            self._seal_physical_kv_handoffs(deadline=deadline)
            self._closing = False
            if slots_error is not None:
                raise slots_error
            self._raise_cleanup_error()
        finally:
            if memory_transaction_acquired:
                self._memory_transaction_lock.release()
            self._close_lock.release()

    def __enter__(self) -> ExpertStreamingRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_configured_expert_runtime(
    root: Path | str,
    manifest_source: ExpertManifest | Path | str,
    config: ExpertStreamingConfig,
    **kwargs: Any,
) -> ExpertStreamingRuntime:
    try:
        return ExpertStreamingRuntime.open(root, manifest_source, config, **kwargs)
    except ExpertManifestError as exc:
        raise ExpertStreamingConfigurationError(str(exc)) from exc
