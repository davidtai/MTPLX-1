# Hy3 Bounded Streaming Expert Cache Design

**Status:** Pending user review

## Problem

Hy3 is a streaming MoE model. Routed expert tensors are read from SSD into a
bounded set of reusable MLX component-bank buffers. A full expert cache is a
normal replacement condition, not evidence that more physical memory is needed.

The issue-46 implementation violates that invariant. It plans expert capacity
against unused KV budget, physically materializes the planned slabs, and later
regrows released slabs from route pressure. A long prefill therefore turns a
streaming cache into an accumulating anonymous-memory cache. macOS compresses
the oversized dirty working set, and later expert hits pay decompression and
residency stalls before Metal can execute.

## Premise Decision

The work is necessary and proportional:

- The same 4K workload fell from 1.239 tok/s with 75 active slabs and no
  compression to 0.303 tok/s with 243 active slabs and 76.9 GB of process
  compression.
- Existing LRU/frequency replacement and slab-safe release machinery already
  provide most of the required behavior.
- Leaving route-demand regrowth in place makes longer prompts converge toward
  the same compression cliff regardless of the startup seed.

## Scope

This change will:

1. Make `expert_cache_limit_bytes` the explicit, user-overridable physical
   expert-cache ceiling for the dynamic Hy3 lane. It remains a cap, never an
   allocation request inferred from the 110 GiB broker budget.
2. Require an explicit expert-cache limit when dynamic expert slabs are enabled,
   so different machines choose their own qualified value and omission fails
   closed.
3. Allocate only the slab-aligned pool covered by that cap at model open.
4. Remove route-demand slab regrowth from prefill, decode, and split-route hot
   paths.
5. Reuse or evict an unpinned expert slot whenever the active cache is full.
6. Continue releasing cold, unpinned whole slabs when KV growth lowers the
   currently admissible expert capacity.
7. Permit restoration toward the configured cap only after KV release at an
   explicit reset/request lifecycle boundary, never because an expert route
   missed.
8. Add opt-in Python control-plane CPU attribution suitable for deciding what a
   Rust rewrite should replace.

## Non-goals

- No automatic compressor-feedback controller in this change. Each machine's
  explicit cap is qualified by the hardware benchmark; compression remains a
  failed qualification signal.
- No per-expert MLX allocation or destruction. Expert replacement reuses an
  existing slot; physical shrink remains whole-slab.
- No change to expert tensor format, `pread`/`preadv`, QMM kernels, cache policy,
  or output semantics.
- No route-time background allocator, detached benchmark process, or GPU-lane
  behavior change.
- No Rust implementation. The added telemetry defines the measured boundary for
  a later rewrite.

## Alternatives Considered

### A. Bounded pool with lifecycle-only resize — selected

The configured expert-cache limit determines the maximum physical slab pool.
Misses reuse the pool. KV growth may shrink it; reset after confirmed KV release
may restore it. This keeps the hot route simple and makes memory use predictable.

### B. Lazy route-driven growth up to a cap — rejected

This avoids startup allocation but preserves allocation in the route path,
creates cold-start decode variance, and risks rebuilding the same demand-growth
controller under a different ceiling.

### C. Per-expert allocation — rejected

This gives finer memory granularity but loses the component-major slab geometry,
adds allocator traffic to the hot path, and requires a broad QMM/storage rewrite.

## Architecture and Invariants

### Physical cache contract

Let:

```text
slab_bytes = expert_slab_slots * expert_record_bytes
configured_expert_cap = floor(expert_cache_limit_bytes / slab_bytes) * slab_bytes
admissible_expert_bytes = min(configured_expert_cap, broker_available_expert_bytes)
```

The runtime continuously preserves:

```text
expert_slab_physical_bytes <= configured_expert_cap
expert_slab_physical_bytes <= admissible_expert_bytes after a completed resize
```

The cap may be zero; transient slots still cover model top-k and preserve
correct streaming execution. A non-aligned cap rounds down and is reported in
telemetry.

### Model open

The memory plan uses the explicit expert cap rather than lending all initially
unused KV budget to expert slabs. `ExpertSlotPool` therefore constructs only the
bounded persistent pool plus the existing transient service bank. The broker's
registered physical expert bytes must exactly match that pool.

There is no allocate-everything-then-release startup sequence.

### Route miss

`ensure_route` and `begin_split_route` never call a slab-regrowth function.
Within the active pool:

- a hit pins the existing slot;
- an empty slot accepts the streamed expert;
- a full pool selects an eligible LRU/frequency victim and overwrites that slot;
- pinned, loading, current-demand, or Metal-in-flight slots remain ineligible;
- if no persistent victim is safe, the existing transient streaming path serves
  the route.

Prefill may seed empty persistent slots but cannot change physical capacity.

### KV growth and shrink

Before KV growth, the existing broker transaction computes required expert
reclaim. Candidate ranking selects cold whole slabs, excludes protected slots,
waits only for their completion owners, invalidates their exact generations,
and confirms allocator reduction before KV allocation proceeds.

If the target is temporarily below pinned capacity, admission fails closed and
reports blocked bytes; it never evicts an unsafe slot.

### KV release and cache restoration

Confirmed KV release raises the admissible expert capacity. Restoration, when
requested, occurs once at reset/request-boundary cleanup under the existing
memory transaction and owner-thread locks. It restores only enough whole slabs
to reach the lower of the configured cap and current broker headroom.

No restoration occurs in prefill, decode, route planning, or reader workers.

## Python Overhead Attribution

Python overhead is measured as thread CPU time, not wall time. This excludes
time sleeping on SSD, completion fences, and Metal, while retaining Python and
same-thread native bookkeeping cost.

The existing `resource_telemetry` opt-in controls collection. When disabled,
the hot path performs no clock reads and publishes no Python attribution block.
When enabled, `time.thread_time_ns()` scopes accumulate:

- `route_control_cpu_ns`: generation-thread CPU inside route setup, commit, and
  cleanup;
- `route_policy_cpu_ns`: the subset spent in LRU/frequency policy planning;
- `slot_control_cpu_ns`: slot selection, pin/generation bookkeeping, and task
  orchestration outside policy planning;
- `resize_control_cpu_ns`: broker and slab resize orchestration CPU;
- `reader_thread_cpu_ns`: the existing reader-worker CPU counter.

Each category includes call counts and prefill/decode phase totals. The schema
labels policy and slot counters as subsets of route control so consumers do not
sum inclusive values incorrectly.

The hardware artifact also reports:

```text
python_route_cpu_ns_per_generated_token
python_policy_cpu_ns_per_route
python_slot_cpu_ns_per_route
python_reader_cpu_ns_per_streamed_byte
python_control_cpu_fraction_of_decode_wall
```

A matched telemetry-off/telemetry-on run measures instrumentation cost. These
numbers are control-plane CPU attribution, not a claim that every native call
inside the measured thread would disappear in Rust.

## Error Handling

- Dynamic mode without `expert_cache_limit_bytes` is rejected before model load.
- A cap smaller than one slab creates zero persistent slabs and uses transient
  streaming only.
- Reclaim that cannot cross a pin/fence boundary fails closed without changing
  published ownership.
- Allocator release/regrow telemetry remains conservative; unexplained negative
  ownership fails closed. Positive allocator residual is retained as charged
  overhead rather than requiring byte-perfect equality.
- A failed lifecycle-boundary restore leaves the smaller released capacity valid
  and does not poison ordinary transient streaming unless ownership is ambiguous.

## Testing Strategy

### Deterministic tests

- Dynamic configuration requires an explicit expert cap and rounds it down.
- The opened pool never exceeds the configured cap.
- Prefill on a full cache does not allocate a slab.
- Decode on a full cache reuses/evicts or falls back to transient slots without
  allocating a slab.
- KV growth releases enough whole slabs and confirms physical reduction.
- KV release alone does not regrow; reset-boundary restoration does and stops at
  the cap.
- Pinned/in-flight slabs block shrink without corrupting policy generations.
- Python counters are absent with telemetry disabled, phase-correct when
  enabled, and use deterministic injected clocks in tests.
- Benchmark serialization rejects missing, negative, overlapping, or
  semantically inconsistent Python counters.

### Hardware qualification

Using the existing foreground exclusive-lane runner:

1. Wait for the shared GPU lock; never terminate or overlap its owner.
2. Capture and unload Qwen only for the measurement window, then restore and
   verify it before releasing the lane.
3. Run matched 4K control/candidate arms with identical prompt, cache geometry,
   generated tokens, and correctness checks.
4. Require identical tokens/routes/hashes and healthy final slot ownership.
5. Record prefill, decode TPS, active slab bytes, process compression growth,
   allocator footprint, and Python CPU attribution.
6. Reject a cap that causes sustained compression growth or more than the
   previously accepted 10-20% 4K slowdown.

## Failure-mode Check

1. **Configured cap is still above the machine's residency knee.** Critical for
   that machine. Qualification rejects the cap; the operator lowers the explicit
   value. Automatic pressure feedback is intentionally deferred.
2. **Target falls below protected slab bytes during KV growth.** Critical to
   correctness but handled fail-closed: KV admission waits/fails and reports
   blocked bytes rather than evicting active Metal ownership.
3. **Instrumentation perturbs the Python hot path.** Minor if bounded. Collection
   is opt-in and a paired telemetry-off/on run quantifies the perturbation before
   using the figures for a Rust rewrite decision.

## Rollout

The behavior remains confined to the opt-in Hy3 Q4 dynamic-memory lane. Static
streaming configurations retain their existing pool construction and policy.
The issue-46 hardware gate must pass before publication or merge; unit-test
success alone is insufficient.
