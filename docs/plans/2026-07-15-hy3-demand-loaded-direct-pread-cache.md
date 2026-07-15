# Hy3 Demand-Loaded Direct-Pread Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (- [x]) syntax for tracking.

**Goal:** Replace issue 46's eager grouped expert storage with a machine-configurable, demand-loaded direct-pread cache that evicts individual experts to keep Hy3 Q4 under its unified-memory limit, while measuring Python control-plane CPU cost.

**Architecture:** The dynamic lane keeps the existing global O(1) LRU policy, direct writable MLX expert-record buffers, positional reader, generation checks, and Metal completion fences. It starts with zero persistent buffers, allocates one direct record only for an admissible miss, reuses one safe LRU buffer at full capacity, and releases individual LRU buffers before physical KV growth. The byte broker accounts current direct-cache bytes rather than grouped storage; cache hits bypass the broker entirely.

**Tech Stack:** Python 3.12, MLX, os.pread/os.preadv, pytest, Ruff, the existing Hy3 Q4 paged-KV runtime, and the existing exclusive GPU/Qwen campaign runner.

**Assumptions:**

- Assumes the dynamic lane remains Hy3 Q4, one active sequence, 131,072-token full-history AR, global LRU, and direct-slot execution — it will NOT enable MTP, AR batching, component-bank execution, or mapped expert execution.
- Assumes the manifest's exact expert_record_bytes is the unit of cache allocation and release — it will NOT infer record size from the checkpoint or round to a multi-record group.
- Assumes direct MLX buffers remain writable by the existing positional reader and executable by the existing direct Q4 path — it will NOT introduce a new tensor layout or kernel.
- Assumes allocator active/cache telemetry is available for the hardware lane — KV growth will NOT be admitted when physical expert release cannot be confirmed.
- Assumes static streaming configurations are compatibility scope — they will NOT become lazy or change cache policy as part of issue 46.
- Assumes /tmp/mtplx-gpu-exclusive.lock may remain owned by another issue — hardware work will wait and will NOT terminate, signal, or overlap the owner.

---

## File Structure

**Create:**

- tests/test_dynamic_expert_cache.py — dynamic direct-cache configuration, lazy warming, individual eviction, KV interaction, and runtime telemetry.
- benchmarks/probe_hy3_direct_cache.py — foreground real-MLX proof of one-record allocation, in-place replacement, individual release, and allocator accounting.

**Delete after replacement coverage is green:**

- tests/test_dynamic_expert_slabs.py — grouped-storage runtime tests superseded by individual-record tests.
- tests/test_expert_slab_policy.py — grouped victim-ranking tests superseded by LRU record ranking.
- benchmarks/probe_hy3_component_slabs.py — grouped allocator probe superseded by the direct-cache probe.

**Modify:**

- mtplx/expert_runtime.py — dynamic direct-cache configuration, lazy miss admission, individual KV reclaim, snapshots, and Python CPU scopes.
- mtplx/expert_streaming.py — initially inactive global capacity, demand calculation, and individual LRU reclaim ranking.
- mtplx/expert_slots.py — optional direct buffers, one-record allocation/release, ownership checks, and cache telemetry; remove grouped lifecycle machinery.
- mtplx/models/expert_mlx.py — add individual release and batched allocator-cache flush to the direct allocator; retain static component banks unchanged.
- mtplx/memory_broker.py — machine-configurable single limit, direct expert-cache accounting, one-record miss reservations, and individual reclaim confirmation.
- mtplx/resource_metrics.py — opt-in phase-scoped Python control CPU counters.
- mtplx/runtime.py — select the direct allocator and lazy pool for the dynamic lane.
- mtplx/runtime_options.py, mtplx/expert_cli.py, mtplx/profiles.py, mtplx/server/openai.py — rename the dynamic capability and remove grouped allocator knobs and health fields.
- mtplx/benchmarks/*hy3_dynamic_memory*.py and benchmark issue-46 sources/specs — direct-cache evidence schema and gates.
- docs/HY3_Q4_DYNAMIC_MEMORY.md and docs/HY3_Q4_DYNAMIC_MEMORY_HARDWARE_CAMPAIGN.md — operator and evidence contracts.

## Task 1: Switch the Dynamic Contract to Direct Records

**Files:**

- Modify: mtplx/expert_runtime.py
- Modify: mtplx/runtime_options.py
- Modify: mtplx/expert_cli.py
- Modify: mtplx/profiles.py
- Modify: mtplx/runtime.py
- Test: tests/test_expert_cli_runtime.py
- Test: tests/test_expert_streaming_models.py
- Test: tests/test_profiles.py
- Test: tests/test_server_openai.py

**Security flag:** none

**Does NOT cover:** This task changes configuration and analytical planning only. Dynamic runtime opening remains blocked until Tasks 2-4 provide physical record lifetimes and broker integration.

- [x] **Step 1: Write failing configuration tests**

~~~python
def test_dynamic_cache_is_direct_machine_configurable_and_record_granular():
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=100 * BINARY_GIB,
        max_live_kv_tokens=131_072,
        runtime_reserve_bytes=8 * BINARY_GIB,
        transient_slots=32,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="direct-slots",
        dynamic_expert_cache=True,
    )
    plan = config.memory_plan(HY3_Q4)
    assert config.memory_limit_bytes == 100 * BINARY_GIB
    assert plan.persistent_slots > 0
    assert not hasattr(config, "dynamic_expert_slabs")
    assert not hasattr(config, "expert_slab_slots")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("slot_layout", "component-banks", "direct-slots"),
        ("cache_scope", "layer", "global"),
        ("cache_policy", "frequency", "lru"),
        ("max_live_kv_tokens", 4096, "131072"),
    ],
)
def test_dynamic_cache_rejects_nonqualified_contract(field, value, message):
    kwargs = _hy3_dynamic_config_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        ExpertStreamingConfig(**kwargs)
~~~

Add CLI tests asserting --hy3-q4-dynamic-memory sets dynamic_expert_cache=True, accepts --expert-memory-limit 100GiB, and rejects the removed --expert-slab-slots, --expert-regrow-hysteresis-slabs, and --expert-resize-min-interval-ms flags.

- [x] **Step 2: Run the tests and confirm RED**

~~~bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_cli_runtime.py \
  tests/test_expert_streaming_models.py \
  tests/test_profiles.py \
  tests/test_server_openai.py \
  -k 'dynamic or slab or memory_limit'
~~~

Expected: FAIL because dynamic_expert_cache does not exist, component banks remain mandatory, and grouped allocator flags remain registered.

- [x] **Step 3: Replace configuration fields and validation**

Use this dynamic surface:

~~~python
resource_telemetry: bool = False
dynamic_expert_cache: bool = False

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
        raise ValueError("dynamic expert cache requires max_live_kv_tokens=131072")
~~~

Delete dynamic_expert_slabs, expert_slab_slots, expert_regrow_hysteresis_slabs, and expert_resize_min_interval_ms. In memory_plan, use context_tokens=0 only to calculate maximum logical cache capacity for the dynamic lane. Do not clamp memory_limit_bytes to 110 GiB and do not align persistent_slots to a multi-record unit. Preserve expert_cache_limit_bytes as an optional stricter ceiling.

- [x] **Step 4: Route dynamic mode to the direct allocator**

~~~python
if expert_streaming_config.dynamic_expert_cache:
    slot_allocator = make_mlx_slot_buffer_allocator(streaming_plan, streaming_spec)
elif expert_streaming_config.slot_layout == "component-banks":
    slot_allocator = make_mlx_component_bank_allocator(
        streaming_plan,
        streaming_spec,
        streaming_manifest,
    )
else:
    slot_allocator = make_mlx_slot_buffer_allocator(streaming_plan, streaming_spec)
~~~

Remove grouped flags from CLI registration, runtime-option attestation, profiles, and server serialization. Do not add compatibility aliases; stale grouped options must fail rather than be ignored.

- [x] **Step 5: Verify and commit**

~~~bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_cli_runtime.py \
  tests/test_expert_streaming_models.py \
  tests/test_profiles.py \
  tests/test_server_openai.py \
  -k 'dynamic or memory_limit or expert_cache'
uv run --frozen --extra dev --extra server ruff check \
  mtplx/expert_runtime.py mtplx/runtime_options.py mtplx/expert_cli.py \
  mtplx/profiles.py mtplx/runtime.py
git add mtplx/expert_runtime.py mtplx/runtime_options.py mtplx/expert_cli.py \
  mtplx/profiles.py mtplx/runtime.py tests/test_expert_cli_runtime.py \
  tests/test_expert_streaming_models.py tests/test_profiles.py \
  tests/test_server_openai.py
git commit -m "refactor(hy3): select direct records for dynamic cache"
~~~

Expected: tests and lint PASS; commit succeeds.

## Task 2: Add Lazy Individual Direct-Buffer Lifetimes

**Files:**

- Modify: mtplx/models/expert_mlx.py
- Modify: mtplx/expert_slots.py
- Modify: mtplx/expert_streaming.py
- Test: tests/test_expert_slots_runtime.py
- Test: tests/test_expert_streaming.py
- Test: tests/test_streamed_models.py

**Security flag:** none

**Does NOT cover:** This task provides record primitives but does not connect them to the broker or KV transactions. Static pools continue allocating all planned buffers.

- [x] **Step 1: Write failing policy and pool tests**

~~~python
def test_global_lazy_cache_activates_and_ranks_individual_lru_records():
    bank = GlobalExpertSlotBank(
        layer_indices=(1, 2),
        expert_count=4,
        persistent_slots=4,
        transient_slots=2,
        prefill_slots_per_layer=2,
        cache_policy="lru",
        initial_active_slots=0,
    )
    assert bank.active_capacity == 0
    assert bank.growth_demand(1, [0, 1], phase="decode") == 2
    bank.activate_slots((0, 1))
    plan, transaction = bank.plan_transaction(1, [0, 1], phase="decode")
    transaction.commit()
    assert tuple(load.slot for load in plan.loads) == (0, 1)
    assert bank.rank_reclaim_slots(protected_slots=(1,)) == (0,)


def test_lazy_pool_allocates_no_persistent_buffers_at_open(tmp_path):
    pool, allocator, spec = _lazy_global_direct_pool(
        tmp_path,
        persistent_slots=2,
    )
    assert allocator.allocated_labels == {"global-transient-0"}
    assert pool.persistent_cache_telemetry_snapshot()["physical_bytes"] == 0
    assert pool.allocate_persistent_slot(0) == spec.expert_record_bytes
    assert allocator.allocated_labels == {
        "global-persistent-0",
        "global-transient-0",
    }
~~~

Add a replacement test that loads expert 0, replaces it with expert 1 in the same slot, and asserts identical id(binding.buffer), incremented generation, exact payload, and zero allocator calls during replacement. Add release tests proving loading, pinned, and Metal-in-flight records are rejected, while an unpinned record releases exactly expert_record_bytes.

- [x] **Step 2: Run tests and confirm RED**

~~~bash
uv run --frozen --extra dev pytest -q \
  tests/test_expert_streaming.py \
  tests/test_expert_slots_runtime.py \
  tests/test_streamed_models.py \
  -k 'lazy or individual_lru or direct_buffer'
~~~

Expected: FAIL because global capacity always starts active, persistent buffers allocate eagerly, and individual release APIs do not exist.

- [x] **Step 3: Add individual direct allocator release**

Keep make_mlx_slot_buffer_allocator callable-compatible and attach:

~~~python
def release_record(label: str) -> int:
    try:
        value = slots.pop(label)
    except KeyError as exc:
        raise RuntimeError(f"direct expert record is not allocated: {label}") from exc
    del value
    return spec.expert_record_bytes


def flush_released_records() -> None:
    _release_mlx_cache()


def close() -> None:
    slots.clear()
    _release_mlx_cache()


setattr(allocate, "release_record", release_record)
setattr(allocate, "flush_released_records", flush_released_records)
setattr(allocate, "close", close)
~~~

Do not change component-bank construction or execution for static configurations.

- [x] **Step 4: Make physical slots lazy and individually releasable**

Change _PhysicalSlot.buffer to Any | None and add lazy_persistent_buffers: bool = False to ExpertSlotPool.__init__. For global lazy pools, construct logical persistent slots with buffer=None, allocate only transient buffers, and set allocated_bytes to actual physical bytes.

Add:

~~~python
@dataclass(frozen=True)
class ExpertRecordReleaseResult:
    slot_ids: tuple[int, ...]
    physical_bytes: int


def _global_persistent_slot(self, slot_id: int) -> _PhysicalSlot:
    normalized = _integer("slot_id", slot_id, minimum=0)
    try:
        return self._persistent[(-1, normalized)]
    except KeyError as exc:
        raise ExpertSlotError("persistent slot is outside the memory plan") from exc


def allocate_persistent_slot(self, slot_id: int) -> int:
    if threading.get_ident() != self._owner_thread_id:
        raise ExpertSlotError("direct record allocation requires its owner thread")
    slot = self._global_persistent_slot(slot_id)
    with slot.condition:
        if slot.buffer is not None or slot.state is not ExpertSlotState.EMPTY:
            raise ExpertSlotError("persistent slot is already allocated or active")
        buffer = self._allocate_buffer(slot.label)
        slot.buffer = buffer
        self.allocated_bytes += self.spec.expert_record_bytes
        return self.spec.expert_record_bytes


def release_persistent_slots(
    self,
    slot_ids: Iterable[int],
) -> ExpertRecordReleaseResult:
    if threading.get_ident() != self._owner_thread_id:
        raise ExpertSlotError("direct record release requires its owner thread")
    normalized = tuple(dict.fromkeys(_integer("slot_id", value, minimum=0) for value in slot_ids))
    slots = tuple(self._global_persistent_slot(slot_id) for slot_id in normalized)
    for slot in slots:
        with slot.condition:
            if slot.buffer is None:
                raise ExpertSlotError("persistent slot is not allocated")
            if slot.state is ExpertSlotState.LOADING or slot.pins:
                raise ExpertSlotError("cannot release an active expert record")
    release_record = getattr(self._allocator, "release_record")
    for slot in slots:
        with slot.condition:
            slot.state = ExpertSlotState.EMPTY
            slot.layer = slot.expert = None
            slot.digest = None
            slot.error = None
            slot.buffer = None
        release_record(slot.label)
    getattr(self._allocator, "flush_released_records")()
    physical_bytes = len(slots) * self.spec.expert_record_bytes
    self.allocated_bytes -= physical_bytes
    return ExpertRecordReleaseResult(normalized, physical_bytes)


def persistent_cache_telemetry_snapshot(self) -> dict[str, int]:
    allocated = resident = loading = pinned = 0
    for slot in self._persistent.values():
        with slot.condition:
            allocated += slot.buffer is not None
            resident += slot.buffer is not None and slot.state is ExpertSlotState.READY
            loading += slot.state is ExpertSlotState.LOADING
            pinned += slot.pins > 0
    return {
        "logical_record_capacity": len(self._persistent),
        "allocated_record_count": allocated,
        "resident_record_count": resident,
        "in_flight_record_count": loading,
        "pinned_record_count": pinned,
        "physical_bytes": allocated * self.spec.expert_record_bytes,
    }
~~~

allocate_persistent_slot runs on the allocator owner thread, accepts only an inactive global slot, allocates global-persistent-N, validates it, publishes under the slot condition, and increments bytes once. release_persistent_slots normalizes unique IDs, rejects LOADING, pins, completion ownership, or buffer=None, clears record state and buffer references, invokes release_record for every label, then calls flush_released_records once for the batch. _physical rejects an absent persistent buffer before any read or binding.

Delete ExpertSlabState, ExpertSlab, grouped tickets/results/errors, grouped registries, prepare/commit/abort/regrow methods, and grouped snapshot sections. Keep logical slot generation watermarks across release and reallocation.

- [x] **Step 5: Add initially inactive policy capacity and record ranking**

Add initial_active_slots: int | None = None to GlobalExpertSlotBank. None preserves static behavior; zero creates an all-false mask and empty free deque.

~~~python
def growth_demand(
    self,
    layer: int,
    expert_ids: Iterable[int],
    *,
    phase: RoutingPhase | str,
) -> int:
    layer, experts = self._validate_experts_without_capacity(layer, expert_ids)
    missing = sum(
        (layer, expert) not in self._key_to_slot
        for expert in dict.fromkeys(experts)
    )
    inactive = self.persistent_slots - self.active_capacity
    if RoutingPhase(phase) is RoutingPhase.PREFILL:
        missing = min(
            missing,
            max(0, self.prefill_slots_per_layer - self._layer_occupancy[layer]),
        )
    return min(missing, inactive)


def rank_reclaim_slots(
    self,
    protected_slots: Iterable[int] = (),
) -> tuple[int, ...]:
    protected = {
        _integer("protected slot", slot, minimum=0)
        for slot in protected_slots
    }
    empty = tuple(
        slot
        for slot, key in enumerate(self._slot_to_key)
        if self._active_slot_mask[slot] and key is None and slot not in protected
    )
    residents = tuple(
        slot
        for key, slot in self._lru.items()
        if slot not in protected
        and self._active_slot_mask[slot]
        and self._directory[key].state == "ready"
    )
    return empty + residents


def inactive_slot_ids(self) -> tuple[int, ...]:
    return tuple(
        slot for slot, active in enumerate(self._active_slot_mask) if not active
    )
~~~

rank_reclaim_slots runs only at KV boundaries. It returns active empty slots first, then ready residents in oldest-to-newest LRU order, excluding protected IDs and non-ready entries. Remove rank_reclaim_slabs; retain existing activate/deactivate generation and rollback invariants.

- [x] **Step 6: Verify and commit**

~~~bash
uv run --frozen --extra dev pytest -q \
  tests/test_expert_streaming.py \
  tests/test_expert_slots_runtime.py \
  tests/test_streamed_models.py
uv run --frozen --extra dev ruff check \
  mtplx/models/expert_mlx.py mtplx/expert_slots.py mtplx/expert_streaming.py \
  tests/test_expert_streaming.py tests/test_expert_slots_runtime.py \
  tests/test_streamed_models.py
git add mtplx/models/expert_mlx.py mtplx/expert_slots.py \
  mtplx/expert_streaming.py tests/test_expert_streaming.py \
  tests/test_expert_slots_runtime.py tests/test_streamed_models.py
git commit -m "feat(hy3): manage direct cache records individually"
~~~

Expected: PASS, including existing static direct-slot and component-bank tests.

## Task 3: Make the Broker Account Direct Cache Records

**Files:**

- Modify: mtplx/memory_broker.py
- Modify: tests/test_memory_broker.py

**Security flag:** none

**Does NOT cover:** The broker remains allocator-agnostic and does not choose LRU victims or mutate MLX buffers. Runtime integration follows in Task 4.

- [x] **Step 1: Write failing one-limit and record-transaction tests**

~~~python
def _snapshot(*, resident=0, kv=0, experts=0, cache=0, pinned=0):
    return BrokerSnapshot(
        resident_model_bytes=resident,
        kv_physical_bytes=kv,
        expert_cache_physical_bytes=experts,
        in_flight_expert_staging_bytes=0,
        runtime_workspace_bytes=0,
        allocator_cache_bytes=cache,
        pinned_expert_bytes=pinned,
    )


def test_budget_has_one_machine_configurable_limit():
    budget = MemoryBudget(memory_limit_bytes=100, allocator_headroom_bytes=10)
    assert budget.memory_limit_bytes == 100
    assert budget.classified_limit_bytes == 90
    assert not hasattr(budget, "hard_ceiling_bytes")


def test_cache_record_growth_is_single_use_and_cap_bounded():
    broker = UnifiedMemoryBroker(
        budget=MemoryBudget(memory_limit_bytes=100),
        initial_snapshot=_snapshot(resident=50, experts=20),
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
    assert broker.plan_expert_cache_growth() is None
~~~

Also cover total-limit refusal returning None, active KV transaction exclusion, failed allocation abort, duplicate ticket rejection, exact record reclaim, pinned refusal, allocator retention remaining charged, grouped-growth APIs absent, and snapshots containing no grouped fields.

- [x] **Step 2: Run tests and confirm RED**

Run: uv run --frozen --extra dev pytest -q tests/test_memory_broker.py

Expected: FAIL because the broker exposes fixed 110/112 GiB limits, grouped fields, and no one-record growth reservation.

- [x] **Step 3: Replace budget and snapshot contracts**

~~~python
@dataclass(frozen=True)
class MemoryBudget:
    memory_limit_bytes: int
    allocator_headroom_bytes: int = 0

    @property
    def classified_limit_bytes(self) -> int:
        return self.memory_limit_bytes - self.allocator_headroom_bytes


@dataclass(frozen=True)
class ExpertCacheAllocationTicket:
    ticket_id: int
    physical_bytes: int
    snapshot_revision: int
    planned_charged_bytes: int


@dataclass
class _PendingExpertCacheGrowth:
    ticket: ExpertCacheAllocationTicket
    registered_cache_bytes_before: int


# UnifiedMemoryBroker.__init__
self._pending_cache_growth: _PendingExpertCacheGrowth | None = None


def _require_cache_growth_ticket(
    self,
    ticket: ExpertCacheAllocationTicket,
) -> _PendingExpertCacheGrowth:
    pending = self._pending_cache_growth
    if pending is None or pending.ticket is not ticket:
        raise MemoryTransactionError("expert cache ticket is not active")
    if ticket.ticket_id in self._consumed_ticket_ids:
        raise MemoryTransactionError("expert cache ticket was already consumed")
    return pending
~~~

Rename BrokerSnapshot.expert_slab_physical_bytes to expert_cache_physical_bytes. Remove pending grouped-growth IDs, last-resize timestamps, grouped hysteresis, and grouped maximum-capacity state. Replace hard/operating threshold branches with charged_bytes <= memory_limit_bytes; allocator headroom continues to reserve space from classified pools.

- [x] **Step 4: Add one-record miss reservations**

~~~python
def plan_expert_cache_growth(self) -> ExpertCacheAllocationTicket | None:
    with self._lock:
        self._ensure_allocation_open()
        if self._pending is not None or self._pending_cache_growth is not None:
            raise MemoryTransactionError("another memory transaction is active")
        registered = self._pools.expert_cache_physical_bytes
        planned_cache = registered + self._expert_record_bytes
        if (
            self._expert_cache_limit_bytes is not None
            and planned_cache > self._expert_cache_limit_bytes
        ):
            return None
        planned_charged = self._pools.charged_bytes + self._expert_record_bytes
        planned_classified = (
            self._pools.classified_bytes + self._expert_record_bytes
        )
        if (
            planned_charged > self._budget.memory_limit_bytes
            or planned_classified > self._budget.classified_limit_bytes
        ):
            return None
        self._revision += 1
        ticket = ExpertCacheAllocationTicket(
            ticket_id=self._next_ticket_id,
            physical_bytes=self._expert_record_bytes,
            snapshot_revision=self._revision,
            planned_charged_bytes=planned_charged,
        )
        self._next_ticket_id += 1
        self._pending_cache_growth = _PendingExpertCacheGrowth(ticket, registered)
        return ticket


def commit_expert_cache_growth(
    self,
    ticket: ExpertCacheAllocationTicket,
    *,
    registered_cache_bytes_after: int,
    allocator_before: AllocatorMemorySample,
    allocator_after: AllocatorMemorySample,
) -> BrokerSnapshot:
    with self._lock:
        pending = self._require_cache_growth_ticket(ticket)
        expected = pending.registered_cache_bytes_before + ticket.physical_bytes
        if registered_cache_bytes_after != expected:
            raise MemoryTelemetryError("registered cache growth is not one record")
        self._validate_allocator_sample("allocator_before", allocator_before)
        self._validate_allocator_sample("allocator_after", allocator_after)
        classified_after = (
            self._pools.classified_bytes
            - self._pools.expert_cache_physical_bytes
            + registered_cache_bytes_after
        )
        allocator_cache_after = max(
            allocator_after.cache_bytes,
            allocator_after.charged_footprint_bytes - classified_after,
        )
        pools = replace(
            self._pools,
            expert_cache_physical_bytes=registered_cache_bytes_after,
            allocator_cache_bytes=allocator_cache_after,
        )
        if (
            pools.charged_bytes > self._budget.memory_limit_bytes
            or pools.classified_bytes > self._budget.classified_limit_bytes
        ):
            raise MemoryAdmissionError("expert record allocation exceeds memory limit")
        self._pools = pools
        self._pending_cache_growth = None
        self._consumed_ticket_ids.add(ticket.ticket_id)
        self._revision += 1
        return self.snapshot()


def abort_expert_cache_growth(
    self,
    ticket: ExpertCacheAllocationTicket,
) -> BrokerSnapshot:
    with self._lock:
        self._require_cache_growth_ticket(ticket)
        self._pending_cache_growth = None
        self._consumed_ticket_ids.add(ticket.ticket_id)
        self._transaction_failure_count += 1
        self._revision += 1
        return self.snapshot()
~~~

Planning returns None when one more exact record crosses the optional expert cap or total limit. It raises only for a failed-closed broker or conflicting transaction. Commit requires registered_after == registered_before + expert_record_bytes, reconciles allocator residual conservatively, and rejects a charged result above the configured limit without consuming the ticket. Abort is valid only before commit.

Keep existing KV single/group ownership, but compute required reclaim from expert_cache_physical_bytes. Rename confirm_expert_reclaim's parameter to registered_cache_bytes_after; accept any record-multiple reduction at least as large as the ticket requirement, retaining unproven allocator bytes as charged cache. Delete grouped expert-growth plan/confirm/abort methods.

- [x] **Step 5: Verify and commit**

~~~bash
uv run --frozen --extra dev pytest -q tests/test_memory_broker.py
uv run --frozen --extra dev ruff check mtplx/memory_broker.py tests/test_memory_broker.py
git add mtplx/memory_broker.py tests/test_memory_broker.py
git commit -m "refactor(hy3): account dynamic cache by expert record"
~~~

Expected: PASS.

## Task 4: Wire Miss Warming and KV-Driven Individual Eviction

**Files:**

- Create: tests/test_dynamic_expert_cache.py
- Modify: mtplx/expert_runtime.py
- Modify: mtplx/runtime.py
- Modify: mtplx/cache_state.py
- Modify: mtplx/generation.py
- Modify: tests/test_cache_state.py
- Modify: tests/test_generation_dynamic_q4_broker.py
- Modify: tests/test_dynamic_q4_group_integration.py
- Delete: tests/test_dynamic_expert_slabs.py
- Delete: tests/test_expert_slab_policy.py

**Security flag:** none

**Does NOT cover:** Hardware performance and Python CPU attribution are separate gates. This task preserves Q4 block geometry, aggregate target-cache ownership, attention semantics, and exact-once KV ownership.

- [x] **Step 1: Write failing runtime lifecycle tests**

~~~python
def test_dynamic_runtime_starts_empty_warms_reuses_and_gives_records_to_kv(runtime):
    record = runtime.spec.expert_record_bytes
    assert runtime.slots.persistent_cache_telemetry_snapshot()["physical_bytes"] == 0

    first = runtime.ensure_route(1, [0], phase="decode")
    first_buffer = first.bindings[0].buffer
    first.release(synchronize=False)
    assert runtime.memory_broker.snapshot().expert_cache_physical_bytes == record

    runtime.set_test_expert_cache_limit(record)
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
~~~

Use a constructor parameter in the fixture instead of mutating production internals; set_test_expert_cache_limit is a test-fixture helper, not production API.

Also test: hits never call the broker; prefill growth is limited to actual seed demand; a miss grows only policy-requested records; no safe victim uses transient service; KV reclaim rounds only to one record; selected pins block without global drain; KV release/reset/cancel do not warm; allocator-retained bytes fail KV admission; split routes share the path; and dynamic snapshots contain no grouped fields.

- [x] **Step 2: Run tests and confirm RED**

~~~bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_dynamic_expert_cache.py \
  tests/test_cache_state.py \
  tests/test_generation_dynamic_q4_broker.py \
  tests/test_dynamic_q4_group_integration.py
~~~

Expected: FAIL because dynamic open and route/KV paths still call grouped lifecycle methods.

- [x] **Step 3: Initialize an empty cache and generic broker**

Pass lazy_persistent_buffers=config.dynamic_expert_cache to ExpertSlotPool and initial_active_slots=0 to GlobalExpertSlotBank. Build the initial snapshot with expert_cache_physical_bytes=0 and charge resident model, transient service, staging, workspace, and allocator residual additively. Construct MemoryBudget from config.memory_limit_bytes and allocator_headroom_bytes; pass the exact record size and optional expert cap to UnifiedMemoryBroker.

Remove grouped-layout attestation. Assert the pool has planned logical record capacity, zero allocated persistent records, expected transient bytes, and the direct allocator backend.

- [x] **Step 4: Add miss-only cache warming**

~~~python
def _warm_direct_cache_records(
    self,
    *,
    layer: int,
    expert_ids: tuple[int, ...],
    phase: RoutingPhase,
) -> int:
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
            cache_bytes = self.slots.persistent_cache_telemetry_snapshot()[
                "physical_bytes"
            ]
            broker.commit_expert_cache_growth(
                ticket,
                registered_cache_bytes_after=cache_bytes,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
            committed = True
            bank.activate_slots((slot_id,))
            grown += 1
        except BaseException:
            if allocated and not committed:
                self.slots.release_persistent_slots((slot_id,))
            if not committed:
                broker.abort_expert_cache_growth(ticket)
            raise
    return grown
~~~

Run it only after a route miss is established. Ask GlobalExpertSlotBank.growth_demand and, for each requested record: obtain plan_expert_cache_growth; sample allocator memory; allocate the next inactive pool slot; commit the broker record; activate the same policy slot; stop without error when the cap/limit returns None. On precommit failure, release the unpublished buffer, flush once, abort the ticket, and leave the slot inactive.

Preserve lock ordering as _memory_transaction_lock then shared global route lock whenever both are held. Keep the all-hit probe and hit path free of the memory lock, allocator samples, and growth calls. For a miss, release the route lock after the side-effect-free miss check, acquire memory transaction, reacquire route lock, recheck demand, warm, then plan. Use the same helper in ensure_route and begin_split_route.

Delete _regrow_for_route_demand and all grouped-growth calls. Full-cache misses use existing LRU replacement and overwrite one direct buffer without broker activity.

- [x] **Step 5: Replace grouped reclaim with individual eviction**

~~~python
def reclaim_expert_records(
    self,
    ticket: KVAllocationTicket | KVAllocationGroupTicket,
    *,
    deadline_ns: int | None = None,
) -> ExpertRecordReleaseResult:
    del deadline_ns
    broker = self.memory_broker
    bank = self._global_bank
    if broker is None or bank is None:
        raise MemoryAdmissionError("dynamic expert cache is not enabled")
    required = ticket.required_expert_reclaim_bytes
    if required == 0:
        return ExpertRecordReleaseResult((), 0)
    record_bytes = self.spec.expert_record_bytes
    record_count = (required + record_bytes - 1) // record_bytes
    candidates = bank.rank_reclaim_slots(self.slots.protected_slot_ids())
    selected = candidates[:record_count]
    if len(selected) != record_count:
        raise MemoryAdmissionError("insufficient unpinned expert records")
    bank.preflight_deactivate_slots(selected)
    allocator_before = self._sample_allocator_memory()
    bank.deactivate_slots(selected)
    released = self.slots.release_persistent_slots(selected)
    allocator_after = self._sample_allocator_memory()
    registered_after = self.slots.persistent_cache_telemetry_snapshot()[
        "physical_bytes"
    ]
    broker.confirm_expert_reclaim(
        ticket,
        registered_cache_bytes_after=registered_after,
        allocator_before=allocator_before,
        allocator_after=allocator_after,
    )
    return released
~~~

Under the existing memory transaction and route-resize exclusion, compute ceil(required_expert_reclaim_bytes / expert_record_bytes), rank individual slots, reject if too few safe records exist, deactivate that exact prefix from policy, release those pool buffers in one batch, sample allocator memory, and call confirm_expert_reclaim. If physical release is incomplete, keep the logically removed records absent, charge retained allocator bytes, and fail KV admission closed. Do not restore destroyed entries and do not allocate after KV release.

Update Q4 single/group growth callbacks to reclaim once before any target cache member allocates. Preserve aggregate declarations, per-member commits, abort, exact block bytes, reset, cancellation, and close behavior.

- [x] **Step 6: Replace telemetry and remove superseded tests**

Dynamic telemetry must report:

~~~python
broker_snapshot = broker.snapshot()
cache_snapshot = self.slots.persistent_cache_telemetry_snapshot()
cache_metrics = self._dynamic_cache_metrics.copy()
result = {
    "memory_limit_bytes": broker.budget.memory_limit_bytes,
    "charged_bytes": broker_snapshot.charged_bytes,
    "charged_residual_bytes": (
        broker.budget.memory_limit_bytes - broker_snapshot.charged_bytes
    ),
    "expert_cache_physical_bytes": (
        broker_snapshot.expert_cache_physical_bytes
    ),
    "expert_cache_limit_bytes": self.config.expert_cache_limit_bytes,
    **cache_snapshot,
    "record_allocations": cache_metrics["record_allocations"],
    "record_reuses": cache_metrics["record_reuses"],
    "record_evictions": cache_metrics["record_evictions"],
    "record_releases": cache_metrics["record_releases"],
}
~~~

Remove grouped counts, group size, hysteresis, growth, and interval fields. Delete grouped test files only after every ownership, failure, KV, and route case has replacement coverage.

- [x] **Step 7: Verify and commit**

~~~bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_dynamic_expert_cache.py \
  tests/test_memory_broker.py \
  tests/test_cache_state.py \
  tests/test_generation_dynamic_q4_broker.py \
  tests/test_dynamic_q4_group_integration.py \
  tests/test_expert_slots_runtime.py \
  tests/test_expert_streaming.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/expert_runtime.py mtplx/runtime.py mtplx/cache_state.py \
  mtplx/generation.py tests/test_dynamic_expert_cache.py
git add mtplx/expert_runtime.py mtplx/runtime.py mtplx/cache_state.py \
  mtplx/generation.py tests/test_dynamic_expert_cache.py \
  tests/test_cache_state.py tests/test_generation_dynamic_q4_broker.py \
  tests/test_dynamic_q4_group_integration.py
git add -u tests/test_dynamic_expert_slabs.py tests/test_expert_slab_policy.py
git commit -m "feat(hy3): evict direct experts within memory budget"
~~~

Expected: tests/lint PASS; commit succeeds.

## Task 5: Attribute Python Control-Plane CPU Cost

**Files:**

- Modify: mtplx/resource_metrics.py
- Modify: mtplx/expert_runtime.py
- Modify: mtplx/expert_slots.py
- Modify: tests/test_resource_metrics.py
- Modify: tests/test_dynamic_expert_cache.py
- Modify: tests/test_expert_slots_runtime.py

**Security flag:** none

**Does NOT cover:** Counters attribute CPU time in measured Python threads. They do not label SSD wait, Metal wait, or every same-thread native call as rewrite-removable Python work.

- [x] **Step 1: Write failing disabled/enabled tests**

~~~python
def test_python_control_cpu_is_absent_when_resource_telemetry_is_disabled(runtime):
    runtime.ensure_route(1, [0], phase="decode").release(synchronize=False)
    assert "python_control_cpu" not in runtime.snapshot()


def test_python_control_cpu_uses_injected_thread_clock(runtime_with_clock):
    runtime = runtime_with_clock([10, 20, 25, 40, 50, 80])
    runtime.ensure_route(1, [0], phase="decode").release(synchronize=False)
    cpu = runtime.resource_telemetry_snapshot()["python_control_cpu"]
    assert cpu["route_control"]["decode_calls"] == 1
    assert cpu["route_control"]["decode_cpu_ns"] == 70
    assert cpu["cache_policy"]["decode_cpu_ns"] <= 70
    assert cpu["reader_thread"]["decode_cpu_ns"] >= 0
~~~

Also cover prefill/decode separation, cache-budget and KV-broker calls, inclusive-subset metadata, nonnegative serialization, and zero clock reads when disabled.

- [x] **Step 2: Run tests and confirm RED**

~~~bash
uv run --frozen --extra dev pytest -q \
  tests/test_resource_metrics.py \
  tests/test_dynamic_expert_cache.py \
  tests/test_expert_slots_runtime.py \
  -k 'python_control or thread_cpu or reader_thread'
~~~

Expected: FAIL because only reader-worker CPU exists.

- [x] **Step 3: Add an opt-in ledger**

~~~python
PYTHON_CONTROL_CATEGORIES = (
    "route_control",
    "cache_policy",
    "cache_budget",
    "kv_broker",
)


@contextmanager
def measure(self, category: str, phase: RoutingPhase | str):
    started = self._thread_cpu_clock()
    try:
        yield
    finally:
        self.record(category, phase, self._thread_cpu_clock() - started)
~~~

Construct ExpertPythonControlLedger only when resource_telemetry=True. The disabled path stores None, branches once before scopes, and never calls time.thread_time_ns. Accept an injected thread_cpu_clock for deterministic tests; default to time.thread_time_ns when enabled.

- [x] **Step 4: Instrument exact boundaries**

Measure route setup/commit/cleanup as route_control; only LRU planning/mutation as cache_policy (a subset); miss growth, record reservation, and eviction as cache_budget; and KV plan/commit/abort/release bookkeeping as kv_broker. Reuse existing reader-worker CPU and expose it in the same phase schema without summing it into route control.

Publish:

~~~python
{
    "inclusive_relationships": {
        "cache_policy": "subset_of_route_control",
        "reader_thread": "separate_worker",
    },
    "clock": "thread_time_ns",
}
~~~

- [x] **Step 5: Verify and commit**

~~~bash
uv run --frozen --extra dev pytest -q \
  tests/test_resource_metrics.py \
  tests/test_dynamic_expert_cache.py \
  tests/test_expert_slots_runtime.py
uv run --frozen --extra dev ruff check \
  mtplx/resource_metrics.py mtplx/expert_runtime.py mtplx/expert_slots.py
git add mtplx/resource_metrics.py mtplx/expert_runtime.py mtplx/expert_slots.py \
  tests/test_resource_metrics.py tests/test_dynamic_expert_cache.py \
  tests/test_expert_slots_runtime.py
git commit -m "feat(bench): attribute dynamic cache Python CPU"
~~~

Expected: PASS.

## Task 6: Replace Grouped Evidence with Direct-Cache Evidence

**Files:**

- Create: benchmarks/probe_hy3_direct_cache.py
- Delete: benchmarks/probe_hy3_component_slabs.py
- Modify: benchmarks/benchmark_hy3_dynamic_memory.py
- Modify: benchmarks/observe_hy3_dynamic_memory_arm.py
- Modify: benchmarks/specs/issue46-hy3-dynamic-memory.json
- Modify: benchmarks/specs/issue46-hy3-hardware-hooks.json
- Modify: mtplx/benchmarks/hy3_dynamic_memory_artifacts.py
- Modify: mtplx/benchmarks/hy3_dynamic_memory_hardware.py
- Modify: mtplx/benchmarks/hy3_dynamic_memory_observation.py
- Modify: mtplx/benchmarks/runners/hy3_dynamic_memory.py
- Modify: docs/HY3_Q4_DYNAMIC_MEMORY.md
- Modify: docs/HY3_Q4_DYNAMIC_MEMORY_HARDWARE_CAMPAIGN.md
- Rename test: tests/test_probe_hy3_component_slabs.py to tests/test_probe_hy3_direct_cache.py
- Test: tests/test_benchmark_hy3_dynamic_memory.py
- Test: tests/test_hy3_dynamic_memory_artifacts.py
- Test: tests/test_hy3_dynamic_memory_hardware_hooks.py
- Test: tests/test_hy3_dynamic_memory_observation.py
- Test: tests/test_hy3_dynamic_memory_acceptance_gates.py
- Test: tests/test_hy3_dynamic_memory_campaign_provenance.py

**Security flag:** none

**Does NOT cover:** This task updates evidence producers and validators but performs no hardware run. The campaign remains a foreground parent that awaits every child and restores Qwen in finally.

- [x] **Step 1: Write failing direct-cache schema/probe tests**

~~~python
assert result["backend"] == "mlx-metal-direct-slots"
assert result["startup_persistent_bytes"] == 0
assert result["first_record_allocated_bytes"] == HY3_Q4.expert_record_bytes
assert result["replacement_buffer_identity_preserved"] is True
assert result["released_record_bytes"] == HY3_Q4.expert_record_bytes
assert result["allocator_charged_drop_bytes"] >= HY3_Q4.expert_record_bytes
assert "slabs" not in result
~~~

Observation tests must require record counts/bytes, allocations/reuses/evictions/releases, Python CPU, GPU utilization, process compression, and configured limit. Reject grouped fields in candidate observations.

- [x] **Step 2: Run tests and confirm RED**

~~~bash
uv run --frozen --extra dev pytest -q \
  tests/test_benchmark_hy3_dynamic_memory.py \
  tests/test_probe_hy3_direct_cache.py \
  tests/test_hy3_dynamic_memory_artifacts.py \
  tests/test_hy3_dynamic_memory_hardware_hooks.py \
  tests/test_hy3_dynamic_memory_observation.py \
  tests/test_hy3_dynamic_memory_acceptance_gates.py \
  tests/test_hy3_dynamic_memory_campaign_provenance.py
~~~

Expected: FAIL because the probe and schemas require grouped storage.

- [x] **Step 3: Build the real direct-cache probe**

Reuse artifact attestation and exclusive-safe JSON I/O. Instantiate the direct allocator and lazy pool; sample allocator; assert zero persistent bytes; allocate and pread one verified record; execute deterministic Q4; overwrite the same buffer with another verified record and re-execute; release the record; call one allocator-cache flush; resample; emit Step 1 fields. Never construct component banks or invoke grouped lifecycle methods.

- [x] **Step 4: Update campaign identity and gates**

Use this candidate identity:

~~~json
{
  "memory_limit_bytes": 107374182400,
  "max_live_kv_tokens": 131072,
  "transient_slots": 32,
  "cache_policy": "lru",
  "cache_scope": "global",
  "slot_layout": "direct-slots",
  "dynamic_expert_cache": true,
  "resource_telemetry": true
}
~~~

Remove group size/hysteresis/interval. The 4K control uses the proven static direct-slot path with identical artifact, prompt, tokens, reader settings, and effective expert allowance. Candidate starts empty and uses identical direct Q4 execution. Keep 32K/64K/128K behind 4K acceptance.

Add a foreground CLI selector used by Task 7:

~~~python
parser.add_argument(
    "--contexts",
    default=",".join(str(value) for value in CONTEXT_MATRIX_TOKENS),
)


def _parse_contexts(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise BenchmarkGateError("--contexts must contain unique context sizes")
    unsupported = set(values) - set(CONTEXT_MATRIX_TOKENS)
    if unsupported:
        raise BenchmarkGateError(f"unsupported context sizes: {sorted(unsupported)}")
    return values
~~~

The selected contexts filter the balanced schedule before any lock acquisition. A multi-context request is rejected unless it includes 4096 first or the supplied 4K evidence path already contains an accepted result for the same source/spec hashes.

Require token/route/hash parity, healthy ownership, no material compression or swap growth, non-stalled GPU activity, and candidate 4K decode no more than 20% below control. Report Python CPU per token, route, miss, eviction, and streamed byte, plus telemetry-on/off perturbation.

- [x] **Step 5: Update docs and foreground guarantees**

Document the exact 100 GiB candidate command using --expert-slot-layout direct-slots and no grouped flags. State that the benchmark stays foreground: its parent holds the lock, synchronously awaits every child, restores/verifies Qwen, then exits. Retain recovery-journal instructions and never recommend stealing a live lock.

- [x] **Step 6: Verify and commit**

~~~bash
uv run --frozen --extra dev pytest -q \
  tests/test_benchmark_hy3_dynamic_memory.py \
  tests/test_probe_hy3_direct_cache.py \
  tests/test_hy3_dynamic_memory_artifacts.py \
  tests/test_hy3_dynamic_memory_hardware_hooks.py \
  tests/test_hy3_dynamic_memory_observation.py \
  tests/test_hy3_dynamic_memory_acceptance_gates.py \
  tests/test_hy3_dynamic_memory_campaign_provenance.py
uv run --frozen --extra dev ruff check \
  benchmarks/probe_hy3_direct_cache.py \
  benchmarks/benchmark_hy3_dynamic_memory.py \
  mtplx/benchmarks/hy3_dynamic_memory_artifacts.py \
  mtplx/benchmarks/hy3_dynamic_memory_hardware.py \
  mtplx/benchmarks/hy3_dynamic_memory_observation.py \
  mtplx/benchmarks/runners/hy3_dynamic_memory.py
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --plan-only
git add benchmarks mtplx/benchmarks docs/HY3_Q4_DYNAMIC_MEMORY.md \
  docs/HY3_Q4_DYNAMIC_MEMORY_HARDWARE_CAMPAIGN.md tests
git commit -m "bench(hy3): qualify demand-loaded direct cache"
~~~

Expected: tests, lint, and plan-only validation PASS without touching Qwen or the GPU lock; commit succeeds.

## Task 7: Verify, Benchmark Under the Lock, and Publish

**Files:**

- Modify only if a gate requires a tested correction: the file named by that failure
- Create outside repository: /tmp/mtplx-issue46/direct-cache-4k.json
- Create outside repository after 4K acceptance: /tmp/mtplx-issue46/direct-cache-campaign.json

**Security flag:** none

**Does NOT cover:** A failed hardware gate does not authorize lowering correctness thresholds, terminating the lock owner, changing another issue's process, or publishing a performance claim.

- [x] **Step 1: Run complete CPU-safe verification**

~~~bash
uv run --frozen --extra dev --extra server pytest -q
uv run --frozen --extra dev --extra server ruff check .
uv run --frozen --extra dev --extra server ruff format --check .
git diff --check
rg -n "dynamic_expert_slabs|expert_slab_slots|expert_regrow|slab_physical|active_slab|released_slab" \
  mtplx benchmarks tests docs/HY3_Q4_DYNAMIC_MEMORY.md \
  docs/HY3_Q4_DYNAMIC_MEMORY_HARDWARE_CAMPAIGN.md
~~~

Expected: tests/lint/format/diff PASS; the search has no live dynamic-lane code, test, schema, or operator-doc match. Historical plans/specs may retain contextual references.

- [ ] **Step 2: Require a committed clean worktree**

~~~bash
git status --short
test -z "$(git status --porcelain=v1 --untracked-files=all)"
~~~

Expected: clean. If Step 1 required corrections, rerun all of Step 1 before committing them.

- [ ] **Step 3: Wait for the shared GPU lock without disturbing its owner**

Use the campaign's bounded foreground acquisition. Do not run kill, pkill, launchctl stop, or remove /tmp/mtplx-gpu-exclusive* outside the campaign's captured Qwen lifecycle. While waiting, report lock-owner state at least once per minute; an unchanged live owner is expected.

- [ ] **Step 4: Run the foreground direct-cache probe and 4K gate**

~~~bash
mkdir -p /tmp/mtplx-issue46
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --contexts 4096 \
  --output-json /tmp/mtplx-issue46/direct-cache-4k.json
~~~

Expected: parent remains foreground; probe passes; Qwen restores/verifies; tokens/routes match; compression does not materially grow; GPU is active; decode is within 20%; Python telemetry is complete.

- [ ] **Step 5: Run longer contexts only after 4K acceptance**

~~~bash
uv run python benchmarks/benchmark_hy3_dynamic_memory.py \
  --spec benchmarks/specs/issue46-hy3-dynamic-memory.json \
  --cwd . \
  --contexts 4096,32768,65536,131072 \
  --output-json /tmp/mtplx-issue46/direct-cache-campaign.json
~~~

Expected: correctness holds; KV plus direct-cache bytes stays within 100 GiB; individual records release as KV grows; no runaway compression/swap; Qwen restores exactly.

- [ ] **Step 6: Record accepted evidence and reverify**

Copy only the compact accepted summary into the checked-in result artifact; retain raw runs in /tmp. Record prefill/decode TPS, GPU utilization, compression, peak charged bytes, allocations/reuses/releases, and Python CPU ratios.

~~~bash
uv run --frozen --extra dev --extra server pytest -q
uv run --frozen --extra dev --extra server ruff check .
uv run --frozen --extra dev --extra server ruff format --check .
git diff --check
git add benchmarks/results docs
git commit -m "bench(hy3): record direct cache memory gate"
~~~

Expected: fresh verification PASS and evidence commit succeeds.

- [ ] **Step 7: Push normally and report exact state**

~~~bash
git status --short
git log -1 --oneline
git push origin feat/issue46-dynamic-memory-broker
~~~

Expected: clean tree, verified evidence at HEAD, normal non-force push. Report commit, remote branch, 4K/long-context results, remaining failures, and whether GitHub still needs an issue/PR update.

## Self-Review Results

- **Spec coverage:** Tasks 1-4 cover direct-only configuration, empty startup, exact record allocation, O(1) hits, in-place replacement, transient fallback, individual KV eviction, configurable limits, and removal of grouped storage. Task 5 covers Python CPU. Tasks 6-7 cover foreground lock discipline, Qwen restoration, compression/GPU/performance gates, and publication.
- **Placeholder scan:** No deferred implementation markers, omitted method bodies, or unspecified error-handling steps remain. The remaining ellipses are Python variadic-tuple type syntax.
- **Type consistency:** dynamic_expert_cache, expert_cache_physical_bytes, ExpertRecordReleaseResult, ExpertCacheAllocationTicket, plan_expert_cache_growth, commit_expert_cache_growth, reclaim_expert_records, and python_control_cpu retain one spelling and responsibility.
- **Scope-reduction scan:** The plan preserves 128K full-history Q4, aggregate target-cache ownership, static compatibility, foreground hardware qualification, and Python-overhead evidence. The 100 GiB value is a benchmark input, not a hard-coded runtime default.
