# Hy3 Demand-Loaded Direct-Pread Expert Cache Design

**Status:** Pending user approval

## Decision

The issue-46 dynamic Hy3 lane will not use expert slabs or component banks.
It will use the existing direct-record execution path: load one routed expert
with positional `pread` into one writable MLX record buffer, cache that record
under `(layer, expert)`, and evict individual records under a byte budget.

The cache starts empty. It does not allocate its configured capacity at model
open. It grows one expert record at a time on real misses, reuses an evicted
record buffer when capacity is unchanged, and releases individual record
buffers only when KV growth reduces the cache's allowed bytes.

## Problem

Hy3 is a streaming MoE model. The issue-46 implementation replaced the simple
record-streaming cache with independently allocated groups of component-bank
storage. It materialized nearly the entire planned expert allowance and then
allowed route pressure to add back storage that KV accounting had removed.

On the measured 4K workload, that turned a bounded streaming cache into a large
dirty anonymous-memory allocation. The 77 GiB eager arm reached 76.9 GB of
process compression and 0.303 decode tok/s. A demand-bounded arm using about
23.7 GiB of expert storage had no compression and reached 1.239 decode tok/s.
The regression is therefore the allocation architecture, not 4K attention and
not `pread` throughput.

## Scope

This change will:

1. Route the issue-46 dynamic lane through `direct-slots`, not
   `component-banks`.
2. Remove the issue-46 grouped expert allocator, grouped lifecycle methods,
   route-demand capacity growth, and their configuration and telemetry fields.
3. Start the persistent expert cache with zero allocated record buffers.
4. Load cache misses directly from the sidecar with the existing checked
   `pread`/`preadv` reader.
5. Account cached expert memory as an exact count of individual record buffers.
6. Enforce a user-configurable total-memory limit and an optional explicit
   expert-cache ceiling; neither value is hard-coded for one Mac.
7. Evict the least-recently-used unpinned expert records until the cache fits
   the current byte allowance.
8. Reuse an evicted direct buffer for ordinary full-cache replacement so the
   hot path does not allocate and free memory repeatedly.
9. Add opt-in Python control-plane CPU attribution for a possible Rust rewrite.

The existing small transient service bank remains fixed and separately
charged. It bounds in-flight top-k misses when every cached record is pinned;
it is not persistent cache capacity.

## Non-goals

- No eager allocation of the expert-cache limit.
- No grouped expert storage, grouped release, grouped growth, or group-size
  rounding in the dynamic lane.
- No `mmap` expert execution or reliance on macOS page faults as the loader.
- No automatic compressor-feedback controller in this change. The total-memory
  limit is explicitly set per machine and qualified by compression telemetry.
- No new cache-policy experiment. The dynamic lane uses the existing global
  O(1) LRU directory.
- No background allocator, detached benchmark, or GPU-lane behavior change.
- No Rust implementation yet; the telemetry identifies what is worth moving.

## Alternatives Considered

### A. Demand-loaded direct record cache — selected

Each cached expert owns one direct MLX record buffer filled by `pread`. The
cache is empty at startup, is byte-accounted exactly, replaces individual LRU
records, and gives bytes back to KV at individual-record granularity. This is
the smallest change that preserves the proven direct-slot data path and the
dynamic-memory goal.

### B. Preallocate a fixed direct-slot cache — rejected

This is simpler than grouped storage and is a useful control benchmark, but it
cannot lend unused expert memory to a growing KV cache.

### C. Grouped component storage — rejected

It creates large dirty allocations, coarse reclamation, extra ownership state,
and the measured compression regression. Its small execution benefit does not
justify using it in the dynamic-memory lane.

### D. Fully adaptive OS-pressure controller — deferred

Reacting continuously to compressor and system-wide pressure may eventually
improve portability, but it adds feedback-loop behavior before the simple
byte-bounded design is qualified. This version exposes the measurements needed
to evaluate that follow-up.

## Architecture and Invariants

### Budget

Let:

```text
record_bytes = exact manifest bytes for one routed expert
configured_cache_cap = optional expert_cache_limit_bytes, or infinity
noncache_bytes = resident model + current physical KV + transient service
                 + in-flight I/O + workspace + charged allocator overhead
available_cache_bytes = max(0, memory_limit_bytes - noncache_bytes)
allowed_cache_bytes = floor(
    min(configured_cache_cap, available_cache_bytes) / record_bytes
) * record_bytes
allocated_cache_bytes = allocated_record_count * record_bytes
```

At every completed memory transaction:

```text
allocated_cache_bytes <= allowed_cache_bytes
total_charged_bytes <= memory_limit_bytes
```

`memory_limit_bytes` is a required, user-overridable value for the dynamic
lane. The first hardware experiment will use 100 GiB on this 128 GB Mac; that
number is a qualification input, not a source-code constant. The existing
expert-cache option remains an optional stricter ceiling.

### Model open

The runtime allocates only resident model state, the small transient service
bank, and bounded I/O workspace. It creates an empty LRU directory and zero
persistent expert record buffers. Maximum KV is not preallocated and expert
capacity is not materialized from unused KV budget.

### Cache hit

A hit validates the record generation, pins the direct buffer, moves the key to
the LRU tail, executes Q4 from that buffer, and releases the pin only after its
Metal completion fence. Hits perform no memory allocation, allocator sampling,
or budget recomputation. With telemetry disabled, the hit path is only the
existing O(1) directory/LRU operation and ownership bookkeeping.

### Cache miss while below the current allowance

The broker reserves exactly one `record_bytes` increment. The owner thread
allocates one direct writable MLX buffer, the reader fills it with checked
positional I/O, and the directory publishes it only after the load and hash
checks succeed. A failed load destroys the unpublished buffer and rolls back
the reservation.

### Cache miss at the current allowance

The LRU directory selects the oldest record that is not loading, pinned,
current route demand, or Metal-in-flight. After the previous generation is no
longer observable, the reader overwrites that same direct buffer and publishes
a new generation. This is an individual expert replacement with zero net
allocation.

If no cached record is safe to replace, the existing transient service bank
serves the miss. The runtime never exceeds the memory limit to create a
temporary persistent entry.

### KV growth

Before allocating a physical Q4 KV block, the broker computes the post-growth
`allowed_cache_bytes`. If the cache is too large, it removes individual LRU
records until the exact byte requirement is met. Pinned, loading, demanded, and
Metal-in-flight records are ineligible.

Released buffers are batched, MLX allocator cache clearing occurs once for the
batch, and allocator telemetry confirms the charged-byte reduction before KV
allocation commits. If safe records or confirmed physical bytes are
insufficient, KV admission waits or fails closed; it never crosses the limit.

### KV release and later cache warming

KV release increases the logical cache allowance but allocates nothing.
Subsequent real misses may add one record buffer at a time. Reset, cancellation,
and request completion never prefill or preallocate the newly available space.

Budget recomputation therefore occurs only at model open, a persistent cache
miss, and a physical KV growth or release transaction. There is no polling loop,
background memory controller, or per-hit memory query.

## Python Overhead Attribution

Python overhead is measured with `time.thread_time_ns()`, not wall time. This
excludes time sleeping on SSD, Metal, and completion fences while retaining
Python and same-thread native bookkeeping cost.

The existing `resource_telemetry` opt-in controls collection. Disabled means
no clock reads and no Python-attribution output. Enabled collection reports
prefill/decode totals and call counts for:

- `route_control_cpu_ns`: route setup, commit, and cleanup;
- `cache_policy_cpu_ns`: LRU lookup, touch, victim selection, and directory
  mutation; this is a subset of route control;
- `cache_budget_cpu_ns`: byte-budget calculation, individual eviction, and
  allocation reservation;
- `kv_broker_cpu_ns`: KV growth/release transaction bookkeeping;
- `reader_thread_cpu_ns`: existing reader-worker CPU attribution.

The benchmark derives CPU time per generated token, per route, per cache miss,
per evicted record, and per streamed byte. A paired telemetry-off/on run reports
instrumentation cost. Inclusive subsets are labeled so they are not summed.

## Error Handling

- Missing or invalid `memory_limit_bytes` rejects the dynamic lane before model
  load.
- An expert-cache ceiling below one record means persistent caching is disabled;
  transient `pread` service remains correct.
- Failed or short reads never publish a cache entry.
- A record is never evicted while loading, pinned, demanded, or Metal-in-flight.
- Insufficient safe cache bytes blocks or rejects KV growth without corrupting
  cache ownership.
- Allocator retention is charged as memory; logical eviction is not reported as
  physical reclamation until telemetry confirms it.

## Testing Strategy

### Deterministic tests

- Dynamic model open allocates zero persistent expert records.
- The first miss allocates exactly one direct record and reads it with the
  positional reader.
- Repeated hits allocate nothing and update LRU order.
- A miss below allowance grows by exactly one record.
- A miss at allowance reuses exactly one unpinned LRU buffer with a new
  generation and zero net bytes.
- A fully pinned cache uses bounded transient service without exceeding budget.
- KV growth evicts the exact number of individual records needed, with no
  group-size rounding.
- KV release, reset, and cancellation do not eagerly warm the cache.
- Dynamic-lane configuration and telemetry contain no grouped allocator fields
  or lifecycle operations.
- Python counters are absent when disabled and phase-correct with deterministic
  injected clocks when enabled.

### Hardware qualification

Using the existing foreground exclusive-lane runner:

1. Wait for the shared GPU lock and never terminate or overlap its owner.
2. Capture and unload Qwen only for the measurement window, then restore and
   verify its exact prior state before releasing the lane.
3. Run a same-geometry 4K control using the proven direct-slot path and a
   candidate using the demand-loaded direct-pread cache at a 100 GiB total
   limit.
4. Require identical generated tokens, routes, hashes, and healthy final
   ownership.
5. Record prefill/decode TPS, GPU utilization, cache records/bytes, allocations,
   buffer reuses, individual evictions, SSD bytes/token, MLX active/cache bytes,
   process resident/compressed bytes, swap, and Python CPU attribution.
6. Reject the candidate if compression grows materially, GPU activity stalls,
   or the 4K slowdown exceeds the previously observed 10-20% range.
7. Only after the 4K gate passes, repeat at longer contexts to demonstrate that
   KV growth reduces expert bytes without crossing the configured total limit.

## Failure-mode Check

1. **The configured total is above this Mac's no-compression knee.** The hardware
   gate rejects it and the operator lowers the per-machine value. The first
   candidate is 100 GiB.
2. **Pinned records prevent enough eviction for KV growth.** KV admission waits
   or fails closed. It does not allocate through the limit or globally drain
   unrelated Metal work.
3. **MLX retains released buffers in its allocator cache.** The retained bytes
   remain charged; KV growth fails closed if the single batched cache clear does
   not make enough physical room.
4. **Lazy allocation adds miss-path variance.** Full-cache replacement reuses
   direct buffers, so allocation occurs only while warming into newly available
   budget. The matched hardware gate measures the remaining cost.
5. **Instrumentation perturbs the control plane.** Collection is opt-in and the
   paired telemetry-off/on arm quantifies the perturbation.

## Rollout

The change remains confined to the opt-in Hy3 Q4 dynamic-memory lane. Existing
static direct-slot and component-bank configurations remain unchanged. The
issue-46 dynamic grouped-storage code and flags are removed rather than kept as
an alternate runtime path. Unit tests are necessary but the foreground 4K
hardware gate must pass before publication or merge.
