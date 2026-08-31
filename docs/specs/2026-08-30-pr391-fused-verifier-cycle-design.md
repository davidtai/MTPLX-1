# PR391 Fused Verifier Cycle Design

## Status

Proposed follow-on to the approved exact-D3 design. This design incorporates the
corrected profiler result: the compiled D3 chain removed 1.202554 seconds of
post-selector gaps, but most of the gain moved to the terminal D3 host
materialization and CPU-built target handoff.

The float32 implementation is a topology and performance experiment only. It is
not eligible for retention. Exact binary64-equivalent softfloat replaces the
probability arithmetic only after the topology produces a measured improvement.

## Scope

Build a construction-installed fixed D3/M4 sampled route for the exact PR391
workload:

- Qwen4 Flash-Next production artifact;
- 16,384 prompt tokens, 1,024 generated tokens;
- temperature 1, top-k 20, top-p 0.95;
- speculative depth 3 and physical verifier width M=4;
- request-owned NumPy PCG64 tape;
- variable-length commit with a maximum output length of 16,384.

The route must preserve every already-selected model optimization. In
particular, it must not replace or bypass the existing MTP QSA implementation,
FRSpec head, target QSA gather/fused-KV route, fixed-M4 verifier, M4 stage-3
kernel, context-copy route, or cache capture/commit machinery.

## Non-goals

- Do not fuse the MTP or 48-layer target model into one custom Metal kernel.
  Their optimized model kernels remain intact.
- Do not enable currently disabled QSA/MTP-precompute, direct-attention, or
  unrelated compiled-MTP experiments while measuring this change.
- Do not make float32 output a production or parity candidate.
- Do not add eligibility checks, environment reads, fallback branches, or
  instrumentation to the measured loop.
- Do not update the PR body until exact parity exceeds 80 TPS.

## Alternatives

### A. Literal whole-model megakernel

Reimplement D1-D3, QSA, the 48-layer target verifier, MoE, attention, sidecar,
sampling, and cache commit in one custom Metal dispatch.

Rejected. A Metal kernel cannot call the existing optimized kernels. This would
discard the proven routes, greatly expand the correctness surface, and prevent
isolated attribution.

### B. Packed D3 only

Pack D3 tokens and distributions, then retain host target sampling and
acceptance.

Useful as an isolated A/B stage, but insufficient as the final topology. It
removes scalar gathers and sparse-distribution construction before target
enqueue while retaining a target-to-host-to-device decision boundary.

### C. Optimized model graphs plus a fused decision controller

Recommended. Keep the existing optimized MTP and target graphs. Chain device
proposal tensors through target distribution shaping and one fused
accept/reject/correction decision kernel. Materialize one compact cycle result
for Python streaming and exact PCG64 advancement.

This is a single device dependency chain, not a single whole-model dispatch.

## Route Partition

Routing is fixed at request construction. The measured loop executes installed
callables directly.

| Cycle class | Draft route | Verify route | Commit route |
| --- | --- | --- | --- |
| Context-copy block | Existing block producer | Existing block verifier | Existing block commit/correction |
| Context-copy depth-1 substitution | Existing copy token | Existing variable verifier | Existing capture/trim commit |
| Fixed D3 sampled cycle | Compiled D1-selector-D2-selector-D3-selector graph using `rt.draft_mtp` | Installed physical-M4 compiled verifier | Fused decision result selects the existing capture/trim outcome |
| Terminal or shortened cycle | Existing stock terminal route | Existing variable verifier | Existing commit route |

The fused controller owns only fixed D3 sampled cycles. It must not claim
context-copy, shortened, constraint, penalty, steering, adaptive-width, or
terminal cycles.

## Route Proof Gate

Before performance work is accepted, the benchmark postprocessor must build a
route receipt from existing outputs after the timer:

1. Count actual D3 cycles and D3 tokens from existing per-cycle `draft_core`
   event tags. Do not infer D3 engagement from aggregate `drafted_tokens`.
2. Partition context-copy block rounds and depth-1 copy substitutions from the
   D3 counts.
3. Require the installed fixed-M4 verifier to report one compiled entry,
   `m4:post_norm:b1`, zero demotions, zero fallbacks, and compiled calls matching
   the eligible physical-M4 cycles.
4. Require the construction receipt to retain the reviewed production flags:
   full FRSpec, QSA gather, fused QSA KV gather, compiled MTP prepare, M4 stage 3,
   and fixed-M4 compiled verification. Disabled comparison arms remain disabled.
5. Partition commit outcomes from existing `capture_repair` event fields for
   acceptance counts 0, 1, 2, and 3, including rejection correction and
   all-accepted bonus paths.
6. Require the request PCG64 start/final state hashes and exact cursor usage to
   be present.

No new per-token or per-dispatch proof counters are added to generation.

## Device Data Flow

### 1. Packed D3 graph

Inputs:

- primary hidden `[1, 1, H]`;
- primary token `[1, 1]`;
- three draft descriptor rows;
- installed MTP state tree.

Outputs:

- draft IDs `[1, 3] uint32`;
- proposal IDs `[3, 20] uint32`;
- proposal values `[3, 20] float32` for the experiment;
- proposal probabilities `[3, 20] float32` for the experiment;
- updated installed MTP state.

The graph continues to call `rt.draft_mtp` at each depth, so its QSA and FRSpec
routes are preserved. D1, D2, and D3 are serial dependencies but require no host
decision between depths.

### 2. Fixed-M4 sidecar bridge

The current SSD-backed PLE sidecar computes n-gram rows and gathers payloads on
the host. Therefore the initial route uses one packed `[primary, d1, d2, d3]`
token-vector materialization for sidecar preparation. Proposal IDs and
probabilities remain device-resident.

This is the only allowed pre-verifier host boundary. Replacing the sidecar with
a resident 20-million-row device table is out of scope because its memory cost
is not safely bounded for the production model. A later measured sidecar
pipeline may overlap host row preparation without changing its storage owner.

### 3. Target and fused decision controller

The installed fixed-M4 verifier consumes the prepared auxiliary tensor and
returns four target rows plus hidden/capture state. Target K20/top-p shaping and
the controller remain device-dependent on those rows.

The float32 experiment controller consumes:

- draft IDs `[3]` and q IDs/probabilities `[3, 20]`;
- target IDs/probabilities `[4, 20]`;
- up to four post-draft uniform descriptors;
- request flags fixed at construction, including whether a bonus row is legal.

It produces:

- accepted count `uint32` in `[0, 3]`;
- first-rejection index `int32`, or `-1` when all accepted;
- correction-or-bonus token and presence flag;
- exact number of post-draft draws consumed;
- fixed-width committed token buffer and committed count;
- selected next verifier row index;
- the cache/capture route index needed by the unchanged commit machinery.

The first experiment materializes this compact decision once and lets the
existing cache commit/rebase code execute the selected path. A subsequent stage
may select verifier hidden/capture leaves on device if the first experiment
proves that the remaining post-decision boundary is material.

### 4. PCG64 ownership

The immutable request tape may be read ahead on device. The authoritative NumPy
generator is advanced only by the returned draw count:

- D3 always consumes three draws;
- a rejection after depth `r` consumes `r + 1` acceptance draws and one
  correction draw;
- all accepted consumes three acceptance draws and one bonus draw only when a
  bonus is sampled.

No fixed post-D3 reservation may advance the cursor speculatively. The exact
production version must reproduce NumPy 2.4.4 normalization, ordering,
comparison, residual construction, and `searchsorted(..., side="right")`.

## Implementation Stages and Gates

### R0: Honest routing receipt

Correct the benchmark receipt so it partitions D3, context-copy, fixed-M4, and
commit routes using existing events/statistics. This changes no model behavior.

Gate: a synthetic mixed-route receipt test fails before and passes after the
change; the existing candidate artifact is identified as aggregate-attribution,
not direct D3 proof.

### R1: Packed D3 handoff

Pack D3 outputs, materialize its token vector once for the sidecar, keep q
arrays on device until the target boundary, and batch any remaining host
conversion.

Gate: corrected route receipt, unchanged exact-control token/RNG/verifier
evidence, and an individual exact-workload A/B.

### R2: Float32 fused decision experiment

Add the prebound float32 target shaping and decision controller. Keep existing
host cache commit/rebase and context-copy routes.

Gate: report TPS, every route partition, hit/miss rates, output digest drift,
first differing token when available, verifier statistics, PCG64 state, and
peak memory. This stage may drift and is never retainable.

### R3: Device state selection

Only if R2 attributes a remaining post-decision gap, select the proper target
hidden/capture state and MTP committed-history prefix on device.

Gate: one focused acceptance-count matrix for 0/1/2/3 plus correction and bonus,
then an individual A/B. Do not combine this stage with R2's first measurement.

### R4: Exact arithmetic

Only if the float32 topology materially improves throughput, replace the
controller's arithmetic with the exact binary64-equivalent softfloat design.

Gate: exact token digest, RNG state/cursor, verifier/acceptance/correction/bonus
counts, variable-length behavior, and the guarded three-seed benchmark. PR-body
promotion requires mean throughput above 80 TPS.

## Testing and Benchmark Safety

Use one focused red/green test for each stage; do not expand broad test coverage.
All real MLX/Metal execution follows `docs/GPU_LOCK_AND_SERVICE_RUNBOOK.md`:

- acquire `/tmp/mtplx-gpu-exclusive.lock` before model loading or unloading;
- account for wired memory and graph peak;
- keep the guard alive for the entire GPU child;
- restore the exact production service and verify health;
- report per-seed decode seconds/TPS, aggregate TPS, route receipt, digests,
  verifier statistics, hit/miss rates, and peak memory.

## Failure-mode Review

1. **Critical: aggregate statistics falsely prove D3 routing.** Corrected by R0:
   event-tag partitioning is mandatory and context-copy is separate.
2. **Critical: a device acceptance result commits the wrong recurrent/cache
   state.** R2 retains existing commit code; R3 requires all four acceptance
   counts plus correction/bonus tests before benchmarking.
3. **Critical: read-ahead advances PCG64 past an unconsumed conditional draw.**
   The device returns `draws_used`; the host advances exactly that count.
4. **Minor: the SSD sidecar retains one pre-verifier host boundary.** This is an
   explicit initial limitation. It is revisited only with direct attribution
   and safe memory accounting.
5. **Minor: float32 changes trajectory and therefore route counts.** Drift is
   reported per seed and never used as parity evidence.
