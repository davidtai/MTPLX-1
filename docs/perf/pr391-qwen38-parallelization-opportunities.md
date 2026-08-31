# PR391 Qwen3.8 Flash-Next Parallelization and Fusion Opportunities

## Status and purpose

This document is the optimization inventory for the corrected variable-length
M=4 verifier in PR #391. It distinguishes:

- work that is already parallel inside MLX/Metal kernels;
- independent work that can be fused or overlapped safely;
- serial model dependencies that cannot be parallelized without speculative
  fan-out;
- measured synchronization gaps versus unmeasured optimization ideas.

The performance target remains more than 80 decode tokens/second on the exact
16,384-token prefill, 1,024-token output, temperature-1, top-k-20, top-p-0.95
workload. No item is a performance claim until it passes an isolated guarded
A/B on that workload.

The production candidate must retain exact NumPy PCG64 behavior, exact
softfloat64 sampling arithmetic, the corrected variable-length verifier, and
the existing QSA, GDN, FRSpec, MoE, cache, and fixed-M4 routes. Only approved
token-ID tie-breaking differences may change semantics.

## Current measured context

The retained exact scheduling/routing change is effectively throughput-neutral
and remains useful as the foundation for additional device chaining:

| Measurement | Decode time | Decode throughput |
| --- | ---: | ---: |
| Exact three-seed baseline | 16.056480 s mean | 63.851758 TPS mean |
| Early exact selected-width replay and D3 enqueue | 16.045084 s mean | 63.892370 TPS mean |

The latter preserved all three reference token digests, PCG64 states, verifier
counts, and variable-length M=4 behavior. It does not, by itself, prove that a
device dependency was removed.

A fresh one-seed run under the instrumented MLX profiler measured the current
exact route as follows. Profiler throughput is diagnostic and must not be
compared directly with production throughput:

| Profiler quantity | Value |
| --- | ---: |
| Decode time | 16.443932 s |
| Decode throughput | 62.272210 TPS |
| GPU timeline | 16.532194 s |
| GPU busy | 14.089182 s |
| GPU idle | 2.443012 s |
| GPU utilization | 85.22% |

The 80-TPS deadline is 12.8 seconds for 1,024 output tokens. The profiler
therefore indicates two distinct requirements:

1. remove as much as possible of the 2.443-second idle interval; and
2. remove roughly another 1.3 seconds of GPU work, because eliminating every
   measured idle interval would still leave approximately 14.09 seconds of GPU
   execution, or about 72.7 TPS.

These are diagnostic ceilings, not predicted production savings.

## Model dependency structure

The production model has 48 sequential transformer layers: 36 GDN layers and
12 QSA layers. A layer consumes the hidden state produced by the preceding
layer, so layers cannot execute concurrently for one request:

```text
layer 1 -> layer 2 -> ... -> layer 48
```

The exact sampled D3 chain is also serial:

```text
primary -> D1 sample -> D2 sample -> D3 sample
```

D2 requires D1's selected token, and D3 requires D2's selected token. The
fixed-M4 target verifier then determines the accepted prefix, correction or
bonus token, authoritative target state, and the primary for the following
cycle:

```text
D3 -> target M=4 -> acceptance/state selection -> next D3
```

These dependencies cannot be removed. The optimization is to keep them in one
device dependency chain so the CPU does not leave the GPU queue empty between
stages.

## Parallelization inventory

### 1. PLE n-gram row preparation and dequantization

For one physical M=4 verifier call, PLE computes 64 hashed row IDs: four token
positions times sixteen n-gram heads. The streamed sidecar gathers the exact
packed q4 payload:

| Payload | Shape | Type |
| --- | --- | --- |
| packed weight | `[64, 20]` | `uint32` |
| scales | `[64, 5]` | BF16 payload |
| biases | `[64, 5]` | BF16 payload |

The three inputs total 6,400 bytes. `mx.dequantize` with `bits=4` and
`group_size=32` produces `[64, 160]` BF16, then reshapes it to the verifier's
`[1, 4, 2560]` PLE embedding, approximately 20 KiB.

Already parallel:

- cold row warming uses a 16-worker `pread` pool;
- q4 dequantization is a GPU-parallel kernel over rows and values;
- the 64 row payloads are gathered in a batch rather than one at a time.

Fresh cache receipt:

| PLE hot-row quantity | Value |
| --- | ---: |
| Hot-cache allocation | 1 GiB |
| Capacity | 10,737,418 rows |
| Hits | 7,304 |
| Misses | 20,440 |
| Hit rate | 26.326% |
| Prefetch batches | 386 |

The cache is not capacity-bound: this request touched only about 20K new rows
against capacity for 10.7M rows. Enlarging the cache would consume memory
without improving this workload's hit rate.

Measured PLE handoff gaps across 374 ordinary verifier cycles:

| Boundary | Total | Host-late | Driver |
| --- | ---: | ---: | ---: |
| prior device work -> q4 PLE dequant | 0.851007 s | 0.715320 s | 0.135687 s |
| q4 PLE dequant -> target verifier | 0.681809 s | 0.619545 s | 0.062263 s |
| Combined | 1.532816 s | 1.334866 s | 0.197950 s |

Ranked PLE changes:

1. **Move q4 dequantization inside the compiled fixed-M4 verifier.** Pass the
   three exact raw arrays as explicit graph inputs, run the unchanged
   `mx.dequantize` at the top of the verifier graph, and return the BF16
   embedding as an explicit output for exact device-state commit. This targets
   the second measured fence.
2. **Fuse dequantization with the first PLE consumers only after exact parity.**
   The PLE key and value projections both consume the same embedding. A custom
   fused kernel could avoid writing and rereading the expanded tensor, but it
   must preserve the explicit BF16 rounding boundary and exact state evolution.
3. **Reduce host row-preparation overhead.** Preserve `_ngram_rows_np` ordering
   and raw byte identity while measuring NumPy uniqueness, LRU bookkeeping,
   thread-pool scheduling, and small-batch copying individually. More worker
   threads are not automatically better for approximately 64 rows.
4. **Overlap host row preparation with independent device work.** Row IDs are
   unavailable until the exact D3 tokens are known, but row gathering can
   overlap cache/state preparation that does not consume the PLE embedding.

Unsafe or unsupported PLE alternatives:

- making the approximately 32-GB n-gram table resident on the 128-GB machine;
- increasing the already-oversized 1-GiB hot cache;
- dequantizing on the CPU and transferring the expanded BF16 tensor;
- evaluating a tree of possible D1/D2/D3 rows before exact tokens are selected.

### 2. PLE projection and gating branches

After the embedding and incoming hidden state exist, these calculations are
independent until their first join:

- `key_proj(embedding)`;
- `value_proj(embedding)`;
- `norm_query(hidden)`.

They can be represented in one compiled graph or a shape-specific fused kernel.
The following gate, sigmoid, value multiplication, grouped normalization, and
short convolution contain additional elementwise fusion opportunities.

The preferred implementation is one wider fused dispatch, not several small
command buffers competing for the same GPU. Measure this only after the PLE
dequantization fence is removed, so attribution remains individual.

### 3. Mixture-of-experts execution

Each MoE layer selects ten routed experts from a 512-expert model. The selected
experts are mathematically independent until their weighted outputs are
reduced. Parallelism is available across:

- selected experts;
- expert rows within a grouped matrix operation;
- gate and up projections that consume the same activation;
- routed and shared branches, if the installed layer exposes them separately.

Most of this parallelism should be expressed through MLX's grouped/sorted qmm
path, which amortizes weight reads and fills the GPU. Ten separate command
buffers or a scalar hand-written QMV can underfill the GPU and reread weights.
Previous affine-model work found hand-written per-expert kernels substantially
slower than stock grouped execution, so a new MoE candidate requires a current
kernel census and shape-specific microbenchmark before model integration.

Potential measured candidates:

1. combine gate/up work without changing affine dequantization order;
2. reduce dispatches around expert weighting and output reduction;
3. improve row ownership or grouping only where the current census shows
   fragmented expert work;
4. fuse shared/routed output addition when both branches are independently
   complete.

### 4. QSA attention layers

Already parallel:

- attention heads;
- Q/K/V projection rows;
- indexer scoring rows;
- selected-key/value gathering;
- value aggregation within the installed QSA kernels.

Possible fusion or overlap:

- derive independent Q/K/V projections from the same hidden input in one
  packed operation;
- overlap indexer work with projection work until the selected-row dependency;
- retain the existing fused K/V gather and remove surrounding shape/copy
  dispatches;
- fuse small normalization, reshape, and cache-offset operations around the
  attention kernel.

Attention layers themselves remain ordered with the rest of the 48-layer
stack. The existing QSA gather and fused-K/V routes must not be bypassed by an
experiment that merely has a similar name or topology.

### 5. GDN recurrent layers

The recurrence is serial across token positions, but its channels, heads, and
state elements are parallel within a position. For the physical M=4 verifier,
the four-row scan can use a fixed-width parallel scan while preserving its
specified recurrence order.

Possible candidates:

- fuse the independent q/k/v/a/b input projections;
- retain the existing fused convolution/normalization and gated-delta kernels;
- fuse small masks, casts, copies, and state-window selections around the main
  recurrence;
- reduce command-buffer boundaries between recurrence output and the following
  hyper-connection work.

Changing scan association can alter floating-point state. Any new scan or
fusion must compare every committed state, not only the final sampled token.

### 6. Hyper-connections, normalization, and residual work

The model's hyper-connection branches operate on the same layer input and have
parallel work across branches and hidden channels. Small operations such as
normalization, gates, residual mixing, masks, casts, and copies are candidates
for graph compilation or fusion.

The goal is dispatch collapse. Running each tiny operation on a separate Metal
stream is likely to increase driver and synchronization overhead. A fused
shape-specific kernel is appropriate only when a dispatch census shows that
the small operations form an exposed command-buffer train.

### 7. Draft MTP and FRSpec sampling

D1, D2, and D3 cannot run concurrently because each selected token feeds the
next depth. They can, however, execute as one device-resident dependency graph:

```text
MTP D1 -> support/softfloat sample ->
MTP D2 -> support/softfloat sample ->
MTP D3 -> support/softfloat sample
```

Within each depth, the following work is parallel:

- MTP QSA and projection kernels;
- compact FRSpec logits;
- K20 support extraction and ordering;
- probability shaping and the softfloat selector.

The exact NumPy uniform tape is generated once on the CPU and installed as a
device input. Batch RNG generation is not itself a meaningful throughput
lever; its value is avoiding a host decision between draft depths.

Computing a full candidate tree would make the depths superficially parallel,
but most branches would be discarded. The extra MTP forwards, memory, and
sampling surface make that a poor exact single-request optimization.

### 8. Fixed-M4 target verification

The target already processes `[primary, d1, d2, d3]` as a four-row batch. Dense
projections, MoE rows, attention heads, and many elementwise operations are
parallel across those rows. Causal attention and GDN recurrence still impose
internal ordering where specified by the model.

The verifier decision can compute in parallel:

- target K20 support for all relevant rows;
- three draft acceptance probabilities;
- correction distributions;
- all-accepted bonus eligibility.

The logically serial result, "accept until the first rejection," can be
implemented with a small prefix/first-rejection operation after all three
probabilities exist. The exact softfloat64 device verifier already follows this
shape.

### 9. Target state selection and next-cycle launch

Accepted-count values 0, 1, 2, or 3 select different authoritative cache and
hidden-state frontiers. Computing all complete state-update branches with
`mx.where` is expensive because MLX evaluates every arm. The preferred route is
to construct the exact candidate leaves once, select compactly on-device, and
continue from only the selected frontier.

The ideal device chain is:

```text
target logits/state
    -> K20 target shaping
    -> exact accept/reject/correction decision
    -> first-rejection and state-frontier selection
    -> exact selected-width target/MTP replay
    -> next D3 launch
```

This chain is dependency-ordered rather than concurrent. Its benefit is that
the GPU queue remains populated without waiting for Python to decode an
accepted count or token.

### 10. CPU and GPU overlap

The following CPU work can be moved outside the measured path or overlapped
with already-queued GPU work:

- request-level PCG64 tape generation;
- construction-time invariant checks and route installation;
- PLE raw-row lookup after D3 tokens become available;
- token decoding and streaming after the committed token buffer is available;
- receipts, digest calculation, statistics, and hit/miss reporting after the
  measured interval;
- safe cache bookkeeping that does not determine the next device input.

No per-token environment reads, fallback checks, route eligibility tests, or
proof counters should be added to the optimized hot path.

### 11. Multi-request batching

Multiple independent requests can increase aggregate GPU utilization through
continuous batching because their layer work is independent. This is a serving
throughput technique, not a solution for the canonical single-request PR #391
benchmark. It changes latency, memory use, cache scheduling, and the metric
being optimized, so it must be evaluated separately.

## What should not be called parallelization

Several changes may move work without reducing the dependency chain:

- calling `mx.async_eval` after a required host decision;
- moving a timer boundary around graph construction;
- launching additional command buffers that serialize on the same GPU;
- materializing a device scalar earlier but still waiting for it on the host;
- computing all state-update arms and selecting with `mx.where`;
- replacing an optimized MLX kernel with a whole-model custom Metal kernel that
  cannot reuse existing QSA, MoE, or qmm implementations.

Evidence of success must be a reduced end-to-end decode time and a fresh
dispatch census showing that the attributed idle interval actually shrank.

## Ranked implementation order

1. **PLE raw-q4 verifier input:** move the unchanged dequantization inside the
   fixed-M4 compiled graph and return its BF16 result for exact state commit.
2. **Re-profile the exact route:** verify that the 0.682-second PLE
   dequant-to-target boundary shrank and identify any shifted gap.
3. **PLE first-consumer fusion:** evaluate dequant plus key/value projection
   fusion while anchoring the BF16 arithmetic boundary.
4. **Complete device controller chain:** keep target support, exact verifier
   decision, accepted-state selection, selected-width replay, and next D3
   dependent on-device.
5. **Profile GPU-busy work:** after synchronization improvements, rank MoE,
   QSA, GDN, hyper-connection, and copy/reduction kernels by actual GPU time.
6. **Test one compute candidate at a time:** prefer dispatch collapse and
   stock-kernel composition before custom replacements.
7. **Run the corrected three-seed gate:** only retained candidates proceed to
   the exact 16K/1K benchmark and memory review.

Do not combine these stages before each individual A/B establishes attribution.

## Correctness gates

Every retained candidate must preserve:

- NumPy PCG64 request-level cursor progression;
- one uniform consumed for every weighted choice and no speculative cursor
  advancement;
- candidate support and ordering;
- descending-logit ordering with ascending-token-ID tie-breaking;
- top-p cumulative-before behavior;
- positive-probability filtering;
- normalization and `searchsorted(..., side="right")` behavior;
- first-rejection selection and correction-token selection;
- variable-length verifier output at M=4;
- target and MTP committed cache state;
- maximum output length of 16,384 tokens;
- all existing QSA, FRSpec, fixed-M4, context-copy, and cache routes.

PLE fusion additionally requires bit-for-bit comparison of the returned
`[1, 4, 2560]` BF16 embedding against the current pre-evaluated reference before
comparing logits, accepted counts, committed states, and output digests.

## Benchmark and operational gates

Every MLX/Metal execution must follow
`docs/GPU_LOCK_AND_SERVICE_RUNBOOK.md`:

1. acquire `/tmp/mtplx-gpu-exclusive.lock` before model loading, compilation,
   profiling, self-checks, or service unloading;
2. account for wired memory, the resident service, caches, and candidate graph
   peak before loading the full model;
3. retain the lock-owning guard for the entire GPU child;
4. restore the exact production service after the test;
5. verify service health and normal fan mode;
6. report individual and mean decode time/TPS, exact token and RNG digests,
   verifier statistics, route receipt, PLE hit/miss rates, and peak memory.

The PR body must not be updated until the corrected three-seed result exceeds
80 TPS with complete parity evidence.

