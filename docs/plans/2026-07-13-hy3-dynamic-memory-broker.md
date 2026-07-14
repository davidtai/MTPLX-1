# Hy3 Q4 Dynamic Unified-Memory Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the complete single-sequence Hy3 Q4 inference working set at or below a 110 GiB operating target and strictly below a 112 GiB hard unified-memory ceiling while preserving a fully attendable 131,072-token Q4 KV history and dynamically lending unused KV budget to physically releasable expert-cache slabs.

**Architecture:** The first stack layer establishes an explicit Hy3 Q4, one-active-sequence, 131,072-token admission contract. The second layer adds a pure fail-closed byte broker, stable logical expert slots backed by independently releasable component-bank slabs, and physical Q4 KV allocation observers. Every KV allocation uses a two-phase reservation that reclaims and confirms expert bytes before MLX allocation; every release credits bytes only after allocator telemetry confirms the physical reduction. The entire dynamic lane remains off by default until software, allocator-release, quality, memory, and paired-performance gates pass.

**Tech Stack:** Python 3.12, pytest, MLX, FastAPI server runtime, existing Hy3 paged-Q4 cache and global component-bank expert runtime, `uv`, Ruff, authenticated `gh`.

**Assumptions:**

- Assumes one active Hy3 sequence and a fully attendable 131,072-token history — it will NOT enable AR batching, multiple live KV stores, sliding-window attention, or SSD-backed active KV.
- Assumes `model_key="hy3-q4"`, `cache_scope="global"`, `slot_layout="component-banks"`, and paged KV quantization `q4` — it will NOT silently enable the broker for other models, Q8/BF16 KV, layer-local caches, direct slots, or Metal mmap.
- Assumes binary GiB, a 110 GiB normal target, and a strict `<112 GiB` stress bound — decimal GB and the machine wired limit are NOT substitutes.
- Assumes request completion, reset, and cancellation release live Q4 arrays rather than transferring them into SessionBank live references — the dynamic lane will NOT retain unmetered live cache objects between requests.
- Assumes MLX exposes active/cache/peak memory telemetry on the target host — if released arrays remain charged or telemetry is unavailable, reclamation fails closed and the dynamic lane is not promoted.
- Assumes exclusive access to the MLX/Metal hardware lane for measurement — no benchmark result is valid unless Qwen state is captured, Qwen is unloaded for the window, the exclusive lock is held, and the exact prior state is restored and verified.

---

## File structure

**Create:**

- `mtplx/hy3_q4_context.py` — exact 131,072-token request contract and Q4 lane validation.
- `mtplx/memory_broker.py` — pure byte accounting, allocation tickets, physical confirmation, hysteresis, and telemetry.
- `tests/test_hy3_q4_context.py` — logical boundary and one-sequence lane tests.
- `tests/test_memory_broker.py` — 110/112 GiB boundaries, reservations, releases, and injected failures.
- `tests/test_expert_slab_policy.py` — active-slot and reclaim-priority policy tests.
- `tests/test_dynamic_expert_slabs.py` — runtime-level reclaim/regrow, pin pressure, telemetry, and rollback tests.
- `tests/test_hy3_dynamic_memory_server.py` — server wiring, cancellation/reset, and disabled-default tests.
- `mtplx/benchmarks/runners/hy3_dynamic_memory.py` — allocator probe and 4K/32K/64K/128K paired experiment runner.
- `benchmarks/benchmark_hy3_dynamic_memory.py` — CLI entrypoint for the issue #46 experiment.
- `docs/HY3_Q4_DYNAMIC_MEMORY.md` — operator contract, flags, telemetry, exact commands, and failure semantics.

**Modify:**

- `mtplx/runtime_options.py` — normalize and attest the opt-in Q4 dynamic lane.
- `mtplx/server/openai.py` — exact context admission, one-sequence enforcement, lifecycle cleanup, and health telemetry.
- `mtplx/cache_state.py` — physical-byte geometry, two-phase Q4 allocation callbacks, close/release, and stats.
- `mtplx/runtime.py` — pass the broker into target and MTP cache construction.
- `mtplx/generation.py` — close or transfer Q4 cache ownership explicitly on all terminal paths.
- `mtplx/expert_streaming.py` — stable active-slot masks, residency class, slab candidate ranking, and generation preservation.
- `mtplx/expert_slots.py` — stable physical slot metadata plus slab reclaim/regrow transactions.
- `mtplx/models/expert_mlx.py` — one component bank per persistent slab and owner-thread allocation/release.
- `mtplx/expert_runtime.py` — dynamic configuration, broker ownership, expert reclaim/regrow, admission, reset, and telemetry.
- `mtplx/expert_streaming_models.py` — stop treating analytical BF16 KV bytes/token as physical Q4 allocation truth in the dynamic lane.
- `mtplx/benchmarks/runners/__init__.py` — expose the issue #46 runner.
- `tests/test_cache_state.py`, `tests/test_expert_slots_runtime.py`, `tests/test_streamed_models.py`, `tests/test_server_openai.py` — integration and regression coverage.

## Execution waves

- **Wave 1:** Tasks 1, 2, and 3 are independent and touch disjoint production modules.
- **Wave 2:** Tasks 4 and 5 follow Task 3; Task 6 follows Tasks 1 and 2. Run them sequentially because they converge on slot/runtime ownership.
- **Wave 3:** Tasks 7 and 8 integrate the completed runtime; Task 9 performs whole-branch verification and exclusive hardware gates.

---

### Task 1: Establish the single-sequence 131,072-token Q4 contract

**Files:**

- Create: `mtplx/hy3_q4_context.py`
- Modify: `mtplx/runtime_options.py`
- Modify: `mtplx/server/openai.py`
- Create: `tests/test_hy3_q4_context.py`
- Modify: `tests/test_server_openai.py`

**Security flag:** none

**Does NOT cover:** This gate applies only when the explicit Hy3 Q4 dynamic-memory lane is enabled. Other models and ordinary server modes retain their existing context and batching behavior.

- [ ] **Step 1: Write failing pure boundary tests**

```python
import pytest

from mtplx.hy3_q4_context import (
    HY3_Q4_TOTAL_CONTEXT_TOKENS,
    Hy3Q4ContextError,
    admit_hy3_q4_context,
)


@pytest.mark.parametrize(
    ("rendered", "requested", "accepted"),
    [
        (131_071, 0, True),
        (131_071, 1, True),
        (131_072, 0, True),
        (131_072, 1, False),
        (131_073, 0, False),
    ],
)
def test_hy3_q4_total_context_boundary(rendered, requested, accepted):
    if accepted:
        admission = admit_hy3_q4_context(rendered, requested)
        assert admission.total_tokens == rendered + requested
        assert admission.limit_tokens == HY3_Q4_TOTAL_CONTEXT_TOKENS
    else:
        with pytest.raises(Hy3Q4ContextError):
            admit_hy3_q4_context(rendered, requested)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `uv run --frozen --extra dev --extra server pytest -q tests/test_hy3_q4_context.py`

Expected: FAIL because `mtplx.hy3_q4_context` does not exist.

- [ ] **Step 3: Implement the exact contract and lane validation**

```python
HY3_Q4_TOTAL_CONTEXT_TOKENS = 131_072


@dataclass(frozen=True)
class Hy3Q4ContextAdmission:
    rendered_tokens: int
    requested_output_tokens: int
    total_tokens: int
    limit_tokens: int = HY3_Q4_TOTAL_CONTEXT_TOKENS


def admit_hy3_q4_context(
    rendered_tokens: int,
    requested_output_tokens: int,
) -> Hy3Q4ContextAdmission:
    rendered = _exact_nonnegative_int("rendered_tokens", rendered_tokens)
    requested = _exact_nonnegative_int(
        "requested_output_tokens", requested_output_tokens
    )
    total = rendered + requested
    if total > HY3_Q4_TOTAL_CONTEXT_TOKENS:
        raise Hy3Q4ContextError(
            f"Hy3 Q4 total context {total} exceeds "
            f"{HY3_Q4_TOTAL_CONTEXT_TOKENS} tokens"
        )
    return Hy3Q4ContextAdmission(rendered, requested, total)
```

Add `Hy3Q4DynamicLaneConfig.validate()` requiring Hy3, q4 paged KV, `context_window == 131_072`, one active sequence, global component banks, and disabled SessionBank live references. Reject invalid combinations at startup instead of coercing them.

- [ ] **Step 4: Add server RED tests for rendered template overhead and batching exclusion**

Test that `_generation_params` raises before generation for totals above 131,072, accepts exactly 131,072, does not force one output token when zero remain, and makes `_use_live_ar_batch` return false for the lane. Assert the lane is disabled by default and cannot be enabled with `q8`, `off`, concurrency greater than one, or live SessionBank cache references.

- [ ] **Step 5: Integrate the admission into the post-template request path**

Call `admit_hy3_q4_context(prompt_token_count, requested_output_tokens)` after the request is fully rendered/tokenized and before response caps or generation. Preserve the caller's requested output reservation in admission accounting. Publish `model_context_limit_tokens`, `rendered_input_tokens`, `requested_output_tokens`, and `admitted_total_tokens` in generation limits and health telemetry.

- [ ] **Step 6: Verify GREEN and commit the prerequisite layer**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_hy3_q4_context.py tests/test_server_openai.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/hy3_q4_context.py mtplx/runtime_options.py mtplx/server/openai.py \
  tests/test_hy3_q4_context.py tests/test_server_openai.py
git add mtplx/hy3_q4_context.py mtplx/runtime_options.py mtplx/server/openai.py \
  tests/test_hy3_q4_context.py tests/test_server_openai.py
git commit -m "feat(hy3): enforce the single-sequence 128K Q4 contract"
```

---

### Task 2: Build the fail-closed shared byte broker

**Files:**

- Create: `mtplx/memory_broker.py`
- Create: `tests/test_memory_broker.py`

**Security flag:** none

**Does NOT cover:** The broker performs authoritative byte accounting and reservations; it does not itself select expert victims or allocate MLX arrays.

- [ ] **Step 1: Write failing 110/112 GiB boundary and reservation tests**

```python
GIB = 1024**3


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
def test_operating_and_hard_boundaries(charged, allowed):
    broker = UnifiedMemoryBroker.standard_hy3()
    if allowed:
        broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=charged))
    else:
        with pytest.raises(MemoryAdmissionError):
            broker.replace_snapshot(BrokerSnapshot.synthetic(charged_bytes=charged))
```

Add RED tests for exact physical Q4 block rounding, steady plus transient allocation peak, logical eviction with zero physical release, pinned-byte shortfall, allocator-retained memory, duplicate release, interrupted allocation, and rollback before/after the destructive boundary.

- [ ] **Step 2: Run the broker tests and confirm RED**

Run: `uv run --frozen --extra dev pytest -q tests/test_memory_broker.py`

Expected: FAIL because `mtplx.memory_broker` does not exist.

- [ ] **Step 3: Implement immutable snapshots and two-phase tickets**

```python
@dataclass(frozen=True)
class MemoryBudget:
    operating_target_bytes: int = 110 * 1024**3
    hard_ceiling_bytes: int = 112 * 1024**3


@dataclass(frozen=True)
class BrokerSnapshot:
    resident_model_bytes: int
    kv_physical_bytes: int
    expert_slab_physical_bytes: int
    in_flight_expert_staging_bytes: int
    runtime_workspace_bytes: int
    allocator_cache_bytes: int
    pinned_expert_bytes: int = 0
    speculative_expert_bytes: int = 0

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


@dataclass(frozen=True)
class KVAllocationTicket:
    ticket_id: int
    steady_delta_bytes: int
    transient_delta_bytes: int
    required_expert_reclaim_bytes: int
    snapshot_revision: int
```

Implement a lock-protected `UnifiedMemoryBroker` with `plan_kv_growth`, `confirm_expert_reclaim`, `commit_kv_growth`, `abort_kv_growth`, `release_kv`, `plan_expert_regrow`, and `snapshot`. Tickets are single-use and revision checked. Normal reservations must end at or below 110 GiB; every observed snapshot at or above 112 GiB records a hard failure and rejects further allocation.

- [ ] **Step 4: Implement physical confirmation semantics**

Define allocator samples as active/cache/peak bytes. Credit expert reclaim only when the charged allocator footprint and registered slab bytes decrease by the requested amount. Moving bytes from MLX active memory into MLX cache is not reclaim. Missing telemetry, negative deltas, or allocator-cache retention fails closed.

- [ ] **Step 5: Verify GREEN and commit**

```bash
uv run --frozen --extra dev pytest -q tests/test_memory_broker.py
uv run --frozen --extra dev ruff check mtplx/memory_broker.py tests/test_memory_broker.py
uv run --frozen --extra dev ruff format --check \
  mtplx/memory_broker.py tests/test_memory_broker.py
git add mtplx/memory_broker.py tests/test_memory_broker.py
git commit -m "feat(hy3): add fail-closed unified-memory accounting"
```

---

### Task 3: Make global expert policy slab-aware without renumbering slots

**Files:**

- Modify: `mtplx/expert_streaming.py`
- Create: `tests/test_expert_slab_policy.py`

**Security flag:** none

**Does NOT cover:** Policy deactivation changes eligibility only; it does not claim physical bytes were released.

- [ ] **Step 1: Write failing active-slot, generation, and priority tests**

Create tests proving deactivated slots are never selected as empty slots or victims; destroy/recreate retains generation watermarks; explicit speculative residents rank before ordinary residents; cold ordinary slabs rank by their hottest member; pinned/current-demand slots exclude their slab; and MTP verify demand is ordinary unless explicitly marked speculative.

- [ ] **Step 2: Confirm RED**

Run: `uv run --frozen --extra dev pytest -q tests/test_expert_slab_policy.py`

Expected: FAIL because active-slot and slab-ranking APIs are absent.

- [ ] **Step 3: Add stable policy metadata and APIs**

```python
class ExpertResidencyClass(str, Enum):
    ORDINARY = "ordinary"
    SPECULATIVE = "speculative"


@dataclass(frozen=True)
class EvictedResident:
    slot: int
    layer: int
    expert: int
    generation: int
    residency_class: ExpertResidencyClass
```

Add `_active_slots` and per-slot residency class to `GlobalExpertSlotBank`. Implement `deactivate_slots`, `activate_slots`, and `rank_reclaim_slabs`. Keep `_slot_to_key`, `_slot_generations`, and history arrays at maximum logical capacity; never resize or renumber them. Deactivation returns exact `(slot, key, generation)` eviction records for the physical transaction.

- [ ] **Step 4: Verify policy regression coverage and commit**

```bash
uv run --frozen --extra dev pytest -q \
  tests/test_expert_slab_policy.py \
  tests/test_expert_slots_runtime.py -k 'global or generation'
uv run --frozen --extra dev ruff check \
  mtplx/expert_streaming.py tests/test_expert_slab_policy.py
git add mtplx/expert_streaming.py tests/test_expert_slab_policy.py
git commit -m "feat(hy3): make global expert policy slab-aware"
```

---

### Task 4: Replace the monolithic persistent component bank with owned slabs

**Files:**

- Modify: `mtplx/models/expert_mlx.py`
- Modify: `mtplx/expert_slots.py`
- Modify: `tests/test_expert_slots_runtime.py`
- Modify: `tests/test_streamed_models.py`

**Security flag:** none

**Does NOT cover:** Transient top-k service slots remain permanently allocated and charged; only persistent global component-bank slots are releasable.

- [ ] **Step 1: Write failing slab lifetime and stale-generation tests**

Cover warm generation `g` followed by slab destroy/recreate and stale `g` rejection; pin/LOADING refusal without mutation; selected-fence waiting without a global drain; unrelated slab release while one is fenced; owner-thread enforcement; interrupted pre-destruction rollback; valid post-destruction released state; and multi-slab Q4 output/router-order parity.

- [ ] **Step 2: Confirm RED against focused suites**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_slots_runtime.py -k 'slab or generation or completion_fence' \
  tests/test_streamed_models.py -k 'component_bank and router_order'
```

Expected: FAIL because slab allocation/release APIs do not exist.

- [ ] **Step 3: Implement stable logical slots and physical slab ownership**

```python
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
```

Use one `MlxComponentBank` per persistent slab. Keep `_PhysicalSlot` objects stable and set `buffer=None` only after the destructive boundary. Add allocator methods `allocate_slab` and `release_slab`; add pool methods `prepare_slab_reclaim`, `commit_slab_reclaim`, `abort_slab_reclaim`, and `regrow_slab`. Capture the MLX owner thread at construction and reject slab allocate/destroy elsewhere.

- [ ] **Step 4: Preserve selected ownership/fence lifecycle without a global drain**

Mark only selected slabs `DRAINING`, recheck each selected slot for `LOADING` and pins, wait only for completion owners attached to those slots, invalidate their exact generations, then release their component banks. Add poison assertions that `_drain_completion_fences` and device-wide synchronize are not called by reclaim.

- [ ] **Step 5: Verify GREEN and commit**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_expert_slots_runtime.py tests/test_streamed_models.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/models/expert_mlx.py mtplx/expert_slots.py \
  tests/test_expert_slots_runtime.py tests/test_streamed_models.py
git add mtplx/models/expert_mlx.py mtplx/expert_slots.py \
  tests/test_expert_slots_runtime.py tests/test_streamed_models.py
git commit -m "feat(hy3): allocate releasable expert component slabs"
```

---

### Task 5: Integrate brokered expert reclaim and lazy regrowth

**Files:**

- Modify: `mtplx/expert_runtime.py`
- Modify: `mtplx/expert_streaming_models.py`
- Modify: `mtplx/runtime.py`
- Create: `tests/test_dynamic_expert_slabs.py`

**Security flag:** none

**Does NOT cover:** Dynamic behavior is rejected unless every attestation for the Hy3 Q4 global component-bank lane is true; existing static plans remain unchanged.

- [ ] **Step 1: Write failing configuration, reclaim, and hysteresis tests**

Use these opt-in defaults:

```python
dynamic_expert_slabs: bool = False
expert_slab_slots: int = 32
expert_regrow_hysteresis_slabs: int = 1
expert_resize_min_interval_ms: int = 1000
```

Test speculative-first and cold-ordinary reclaim, pinned/current-demand exclusion, zero credit for logical-only eviction, allocator-retention failure, no unrelated fence wait, lazy on-demand regrowth, hysteresis, minimum interval, allocation failure, interrupted resize, and exact restoration to a valid capacity.

- [ ] **Step 2: Confirm RED**

Run: `uv run --frozen --extra dev --extra server pytest -q tests/test_dynamic_expert_slabs.py`

Expected: FAIL because runtime reclaim/regrow and telemetry do not exist.

- [ ] **Step 3: Implement runtime transactions**

```python
def reclaim_expert_bytes(
    self,
    requested_bytes: int,
    *,
    deadline_ns: int | None = None,
) -> ExpertReclaimResult:
    requested = _integer("requested_bytes", requested_bytes, minimum=1)
    with self._dynamic_resize_lock:
        candidates = self._global_bank.rank_reclaim_slabs(
            self.slots.slab_layout(),
            self.slots.protected_slot_ids(),
        )
        ticket = self.slots.prepare_slab_reclaim(
            candidates,
            requested_bytes=requested,
            deadline_ns=deadline_ns,
        )
        result = self.slots.commit_slab_reclaim(ticket)
        self.memory_broker.confirm_expert_reclaim(result.physical_sample)
        self._global_bank.deactivate_slots(result.released_slot_ids)
        return result


def maybe_regrow_expert_slabs(
    self,
    target_bytes: int,
    *,
    now_ns: int,
) -> int:
    target = _integer("target_bytes", target_bytes, minimum=0)
    with self._dynamic_resize_lock:
        slab_ids = self.memory_broker.plan_expert_regrow(
            target_bytes=target,
            now_ns=now_ns,
        )
        for slab_id in slab_ids:
            self.slots.regrow_slab(slab_id)
            self._global_bank.activate_slots(
                self.slots.slot_ids_for_slab(slab_id)
            )
        return len(slab_ids)
```

Run candidate ranking under the existing global policy lock, prepare physical reclaim, invalidate exact policy generations, cross the physical destruction boundary, sample allocator memory, and confirm with the broker. Before destruction, abort restores policy and physical state. After destruction, failures leave a valid released capacity and never republish old mappings. On expert demand, regrow only enough slabs to satisfy the route after hysteresis and interval gates.

- [ ] **Step 4: Replace analytical KV planning in the dynamic lane**

Retain analytical `kv_bytes_per_token` only for static diagnostics. When dynamic Q4 is enabled, initialize the broker from resident model, transient service bank, staging, workspace, measured allocator cache, and zero physical KV; never pre-advertise expert capacity that exceeds allocated active slabs.

- [ ] **Step 5: Add telemetry and verify GREEN**

Snapshot must report target/ceiling; expert logical capacity, resident records, active/released slab count, physical bytes; pinned/in-flight/speculative bytes; requested/reclaimed bytes; resize duration; blocked-by-pin bytes; admission failures; hysteresis state; and allocator active/cache/peak.

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_dynamic_expert_slabs.py tests/test_expert_slots_runtime.py \
  tests/test_streamed_models.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/expert_runtime.py mtplx/expert_streaming_models.py mtplx/runtime.py \
  tests/test_dynamic_expert_slabs.py
git add mtplx/expert_runtime.py mtplx/expert_streaming_models.py mtplx/runtime.py \
  tests/test_dynamic_expert_slabs.py
git commit -m "feat(hy3): reclaim and regrow expert slabs under the broker"
```

---

### Task 6: Broker every physical Q4 KV allocation and release

**Files:**

- Modify: `mtplx/cache_state.py`
- Modify: `mtplx/runtime.py`
- Modify: `mtplx/generation.py`
- Modify: `tests/test_cache_state.py`
- Modify: `tests/test_generation_sustained.py`

**Security flag:** none

**Does NOT cover:** Lowering a logical offset with `trim()` releases no physical bytes and must never produce broker credit.

- [ ] **Step 1: Write failing geometry and lifecycle tests**

Test actual q4 key/value/scales byte counts from block shapes and dtypes; block rounding; first allocation; 1.5x growth; the concatenate peak (`old + extra + new`) reserved before any `mx.zeros`; no allocation when expert reclaim is short; commit only after `mx.eval`; abort on allocation failure; no credit from `trim`; exact-once physical release on success/reset/cancel; target and MTP cache inclusion; and stale-cache absence on a second sequential request.

- [ ] **Step 2: Confirm RED**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_cache_state.py -k 'q4 and (physical or broker or growth or release)' \
  tests/test_generation_sustained.py -k 'q4 and (cancel or reset or second)'
```

Expected: FAIL because physical allocation observers and explicit release are absent.

- [ ] **Step 3: Add the observer contract and actual byte geometry**

```python
class KVPhysicalAllocationObserver(Protocol):
    def reserve_growth(
        self,
        *,
        cache_id: str,
        steady_delta_bytes: int,
        transient_delta_bytes: int,
        new_physical_blocks: int,
    ) -> KVAllocationTicket:
        raise NotImplementedError

    def commit_growth(
        self,
        ticket: KVAllocationTicket,
        *,
        measured_physical_bytes: int,
    ) -> None:
        raise NotImplementedError

    def release_cache(
        self,
        *,
        cache_id: str,
        measured_physical_bytes: int,
    ) -> None:
        raise NotImplementedError
```

Pass the runtime broker to every target and MTP `VllmMetalPagedKVCache`. Calculate bytes from concrete shapes and dtype widths before allocation. In `_grow_to_capacity`, reserve the steady-state extra blocks plus the full transient concatenate allocation; reclaim and confirm expert slabs before creating any MLX array.

- [ ] **Step 4: Add deterministic close/ownership transfer**

Implement `VllmMetalPagedKVCache.close()` as idempotent. Generation owns the cache until it either transfers ownership to an allowed holder or closes it on every success/error/cancellation path. The dynamic lane disallows SessionBank live-reference transfer, so request completion closes target and MTP caches before releasing the admission. `trim()` changes logical tokens only.

- [ ] **Step 5: Verify GREEN and commit**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_cache_state.py tests/test_generation_sustained.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/cache_state.py mtplx/runtime.py mtplx/generation.py \
  tests/test_cache_state.py tests/test_generation_sustained.py
git add mtplx/cache_state.py mtplx/runtime.py mtplx/generation.py \
  tests/test_cache_state.py tests/test_generation_sustained.py
git commit -m "feat(hy3): broker physical Q4 KV block lifetimes"
```

---

### Task 7: Wire the dynamic lane into serving and full telemetry

**Files:**

- Modify: `mtplx/server/openai.py`
- Create: `tests/test_hy3_dynamic_memory_server.py`
- Modify: `tests/test_server_openai.py`
- Create: `docs/HY3_Q4_DYNAMIC_MEMORY.md`

**Security flag:** none

**Does NOT cover:** The feature remains off by default and never treats OS pressure, swap, compressor activity, allocation failure, or process termination as a resizing signal.

- [ ] **Step 1: Write failing server lifecycle and telemetry tests**

Cover startup attestation, disabled default, speculative admission stop before reclaim, request rejection before oversubscription, exact-once cancellation/reset release, no SessionBank live reference, normal charged memory at/below 110 GiB, stress snapshots strictly below 112 GiB, and every telemetry field named in issue #46.

- [ ] **Step 2: Confirm RED**

Run: `uv run --frozen --extra dev --extra server pytest -q tests/test_hy3_dynamic_memory_server.py tests/test_server_openai.py`

Expected: FAIL because dynamic lane wiring and telemetry are absent.

- [ ] **Step 3: Wire explicit flags and fail-closed flow**

Add `--hy3-q4-dynamic-memory`, slab-size, hysteresis, and minimum-resize-interval options. Validate the lane before model load. Before KV growth, stop speculative admission, request selected expert reclaim, confirm allocator reduction, then authorize Q4 allocation. On reset/cancel/completion, close KV, confirm release, and permit lazy expert regrowth only on future demand.

- [ ] **Step 4: Publish complete resource telemetry**

Include operating target/hard ceiling; resident model; KV representation/logical tokens/physical blocks/bytes; expert capacity/residents/slabs/bytes; pinned/in-flight/speculative bytes; workspace/staging reserves; allocator active/cache/peak; process resident/compressed memory and swap delta where available; requested/reclaimed bytes; evicted records/slabs; resize duration; blocked-by-pin bytes; admission failures; hit rate; SSD bytes/token; decode TPS; and p50/p95 latency.

- [ ] **Step 5: Document exact operator contract and verify**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_hy3_dynamic_memory_server.py tests/test_server_openai.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/server/openai.py tests/test_hy3_dynamic_memory_server.py \
  tests/test_server_openai.py
git add mtplx/server/openai.py tests/test_hy3_dynamic_memory_server.py \
  tests/test_server_openai.py docs/HY3_Q4_DYNAMIC_MEMORY.md
git commit -m "feat(server): expose the opt-in Hy3 Q4 memory broker"
```

---

### Task 8: Add allocator probe and balanced hardware experiment runner

**Files:**

- Create: `mtplx/benchmarks/runners/hy3_dynamic_memory.py`
- Create: `benchmarks/benchmark_hy3_dynamic_memory.py`
- Modify: `mtplx/benchmarks/runners/__init__.py`
- Create: `tests/test_benchmark_hy3_dynamic_memory.py`

**Security flag:** none

**Does NOT cover:** The runner records evidence; it does not automatically promote the broker or change default configuration.

- [ ] **Step 1: Write failing manifest, pairing, and restoration tests**

Test declared cache starting state, context matrix `(4096, 32768, 65536, 131072)`, static/dynamic balanced ordering, multiple block boundaries, physical expert decrease before KV increase, stable hold samples, reset/regrow samples, intervals, exact model/artifact/config identity, exact tokens/routes/expert hashes/slot health, and Qwen capture/unload/restore hooks that restore on exceptions.

- [ ] **Step 2: Confirm RED**

Run: `uv run --frozen --extra dev --extra server pytest -q tests/test_benchmark_hy3_dynamic_memory.py`

Expected: FAIL because the runner and CLI do not exist.

- [ ] **Step 3: Implement the focused allocator-release probe**

Allocate at least two real Hy3 component slabs, evaluate them, release one selected slab, sample MLX active/cache memory before and after, and emit a fail-closed JSON result. Do not run the full matrix unless charged allocator bytes fall by at least the slab's registered physical bytes and the untouched slab remains executable.

- [ ] **Step 4: Implement the paired 4K/32K/64K/128K campaign**

Compare the dynamic lane against a correctly budgeted static 128K-reserved Q4 control. At every point grow across block boundaries, prove expert bytes fall before KV bytes rise, hold long enough for stable cache/performance samples, reset and prove KV release plus lazy expert regrow, and retain raw paired samples plus confidence intervals.

- [ ] **Step 5: Verify runner tests and commit**

```bash
uv run --frozen --extra dev --extra server pytest -q \
  tests/test_benchmark_hy3_dynamic_memory.py
uv run --frozen --extra dev --extra server ruff check \
  mtplx/benchmarks/runners/hy3_dynamic_memory.py \
  benchmarks/benchmark_hy3_dynamic_memory.py \
  tests/test_benchmark_hy3_dynamic_memory.py
git add mtplx/benchmarks/runners/hy3_dynamic_memory.py \
  mtplx/benchmarks/runners/__init__.py \
  benchmarks/benchmark_hy3_dynamic_memory.py \
  tests/test_benchmark_hy3_dynamic_memory.py
git commit -m "bench(hy3): add issue 46 dynamic-memory gates"
```

---

### Task 9: Prove software, quality, memory, and paired-performance acceptance

**Files:**

- Create after measurement: `benchmarks/results/hy3-dynamic-memory-issue46-20260713.md`
- Modify after measurement: `docs/HY3_Q4_DYNAMIC_MEMORY.md`

**Security flag:** none

**Does NOT cover:** A failed allocator, quality, memory, or benefit gate leaves the feature off and records rejection; it must not be reframed as successful implementation.

- [ ] **Step 1: Run changed-file and full repository verification**

```bash
files="mtplx/hy3_q4_context.py mtplx/memory_broker.py mtplx/runtime_options.py mtplx/cache_state.py mtplx/runtime.py mtplx/generation.py mtplx/expert_streaming.py mtplx/expert_slots.py mtplx/models/expert_mlx.py mtplx/expert_runtime.py mtplx/expert_streaming_models.py mtplx/server/openai.py mtplx/benchmarks/runners/hy3_dynamic_memory.py benchmarks/benchmark_hy3_dynamic_memory.py tests/test_hy3_q4_context.py tests/test_memory_broker.py tests/test_expert_slab_policy.py tests/test_dynamic_expert_slabs.py tests/test_hy3_dynamic_memory_server.py tests/test_benchmark_hy3_dynamic_memory.py"
uv run --frozen --extra dev --extra server ruff check $files
uv run --frozen --extra dev --extra server ruff format --check $files
git diff --check origin/experiment/moe-pr13-pr14-stack...HEAD
uv run --frozen --extra dev --extra server python -m pytest -q
```

Expected: Ruff, format, diff hygiene, and the full suite exit 0.

- [ ] **Step 2: Run the Q4 correctness and long-context retrieval lane**

Use the same rendered prompts, sampler, artifact revision, and full-attention semantics for the BF16 numerical control and Q4 candidate. Cover short-context regression plus 4K/32K/64K/128K retrieval, including earliest-context retrieval at 128K. Save exact commands, tokens, text, route traces, expert hashes, and cache/slot health.

- [ ] **Step 3: Run the exclusive allocator probe**

Capture the exact Qwen service state, acquire `/tmp/mtplx-gpu-exclusive`, unload Qwen only for the measurement window, run the focused real-MLX slab release probe, restore the exact prior Qwen service, verify its model endpoint, and release the lock. Stop if allocator-active plus allocator-cache bytes do not fall by the required slab bytes.

- [ ] **Step 4: Run balanced static/dynamic pairs at all four contexts**

For every context, use a declared cache state and balanced arm ordering. Reject any run with swap growth, compressor runaway, allocation failure, stale publication, pin violation, global barrier, token/route/hash mismatch, or unexplained memory identity drift. Report intervals for expert hit rate, SSD bytes/token, decode TPS, and p50/p95 token latency.

- [ ] **Step 5: Apply the issue #46 acceptance decision**

Promote only if normal charged peak is at most 110 GiB, stress peak is strictly below 112 GiB, 128K history passes retrieval, short-context physical expert capacity exceeds static control, that extra capacity measurably improves hit rate/SSD bytes/TPS, and 128K converges to static control without unexplained regression. Otherwise retain static partitioning and keep the dynamic feature off.

- [ ] **Step 6: Curate and publish evidence**

Commit the curated report and documentation, leaving bulky raw artifacts in ignored storage. Post exact token-generation TPS, memory peaks, intervals, commands, revision, Qwen restoration evidence, and the promotion/rejection decision to GitHub issue #46 using authenticated `gh`.

```bash
git add benchmarks/results/hy3-dynamic-memory-issue46-20260713.md \
  docs/HY3_Q4_DYNAMIC_MEMORY.md
git commit -m "bench(hy3): record issue 46 dynamic-memory evidence"
```

---

## Self-review

- **Spec coverage:** Tasks 1 and 6 cover Q4 131,072-token full-history semantics and physical block rounding. Tasks 2, 5, and 7 cover every charged pool, 110/112 GiB boundaries, fail-closed admission, allocator cache, lifecycle, hysteresis, and telemetry. Tasks 3 and 4 cover speculative-first/cold-ordinary reclamation, pins/fences, generation safety, physical slabs, owner-thread rules, and no global drain. Tasks 8 and 9 cover allocator proof, exact correctness, all four context points, paired intervals, Qwen preservation, and the promotion gate.
- **Completeness scan:** Every task names concrete files, behavior tests, implementation APIs, verification commands, and a commit boundary.
- **Type consistency:** `Hy3Q4ContextAdmission`, `MemoryBudget`, `BrokerSnapshot`, `KVAllocationTicket`, `ExpertResidencyClass`, `ExpertSlab`, and `KVPhysicalAllocationObserver` retain the same names and responsibilities across tasks.
- **Scope-reduction scan:** The opt-in lane restrictions are explicit safety boundaries from issue #46 and do not replace any requested outcome. Hardware or allocator failure records a real rejection instead of being relabeled as completion.
