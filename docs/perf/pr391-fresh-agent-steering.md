# PR #391 Fresh-Agent Steering: Exact M=4 GPU-Starvation Removal

Use this document as the complete steering prompt for the next agent. Continue
from the existing worktree; do not restart the implementation, discard dirty
changes, or reinterpret the workload.

## Mission

Recover the corrected variable-length M=4 verifier toward and ultimately above
80 decode tokens/second without changing model semantics. The immediate task is
to overlap the CPU PLE n-gram lookup with independent target GPU work after D3,
then continue removing measured GPU queue starvation one individually
attributed stage at a time.

The next implementation is specifically this fork and join:

```text
                                +-> CPU: hash the exact D3 window and fetch PLE rows --+
final D3 token is available ----+                                                       +-> join at layer 1 PLE
                                +-> GPU: target embedding and complete decoder layer 0 -+
```

After that stage has its own correctness and performance result, separately
test extending the GPU prefix through the PLE-independent layer-1 query work,
`norm_query(layer0_hidden)`. Do not combine these stages in the first A/B.

## Worktree and current state

- Repository worktree:
  `/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-flash-next-210-restack`
- Branch: `port/qwen38-flash-next-210-restack`
- Recorded starting HEAD: `4978b013c87bc55a0ffc402666862ca6250f7afb`
- Pull request: <https://github.com/youssofal/MTPLX/pull/391>
- The worktree is intentionally dirty and contains substantial uncommitted
  PR391 work. Preserve it. Do not reset, clean, overwrite, or revert unrelated
  changes.
- Do not update the PR body until the corrected three-seed result exceeds
  80 TPS with complete parity evidence.
- Do not commit or push unless the user explicitly asks.

The active PR391 route is currently a request-installed benchmark experiment,
not normal-serving production wiring. It is installed by
`scripts/pr391_metal_choice_benchmark_launcher.py`; source comments and route
receipts still identify the D3 lane as isolated/test-only. Exact benchmark
results prove the experimental route, not production usability. After the
route exceeds the performance/correctness gates, normal-serving
construction-time installation and a non-test route receipt are a separate
promotion gate. Do not relabel the receipt or claim production usability before
that wiring is implemented and verified.

Read before editing:

1. `/Users/davidtai/projects/OpenSourceWTF/AGENTS.md`
2. `/Users/davidtai/projects/OpenSourceWTF/docs/GPU_LOCK_AND_SERVICE_RUNBOOK.md`
3. `docs/perf/pr391-qwen38-parallelization-opportunities.md`
4. `docs/specs/2026-08-30-pr391-fused-verifier-cycle-design.md`
5. this file

Begin with `git status --short`, `git diff --check`, and a scoped diff of the
files involved. Existing edits belong to the user or prior agents.

### Known red/contradictory test debt

Do not assume the whole focused PR391 suite is a green baseline. The current
source-level contracts in `tests/test_pr391_float32_d3_core.py` contradict one
another:

- `test_exact_width_verifier_mtp_replay_uses_only_the_host_selected_width`,
  `test_canonical_d3_is_queued_immediately_after_selected_replay`, and
  `test_exact_width_replay_follows_host_decision_and_precedes_live_d3` require
  the obsolete host-decoded replay schedule and forbid device replay;
- `test_device_mtp_replay_is_queued_before_host_decision_decode` and
  `test_device_next_d3_is_queued_before_host_decision_decode` require the
  currently implemented device-before-decode schedule;
- an earlier construction test expects one `mx.compile` call, while
  `_pr391_make_float32_d3_core` currently installs a bare `chain_fn`.

Before using this file as a gate, make one source-level scheduling test express
the currently approved device-before-host direction and remove or rewrite only
the mutually stale assertions. Do not change production scheduling merely to
make an obsolete test pass. Treat the missing outer `mx.compile` separately:
first determine whether it is intentional/currently measured, then either fix
the construction or correct the stale expectation with evidence.

## Non-negotiable workload

All performance comparisons use the corrected verifier and exactly:

- model: the production Qwen3.8 Flash-Next MTPLX artifact and installed routes;
- prompt/prefill length: 16,384 tokens;
- requested generated-token budget: 1,024 tokens;
- temperature: 1;
- top-k: 20;
- top-p: 0.95;
- speculative draft depth: D3;
- physical target width: M=4;
- seeds: 20260829, 20260830, and 20260831 for the final gate;
- verifier output: variable length per cycle;
- maximum supported request output length: 16,384 tokens.

Do not substitute a fixed verifier output length, temperature 0, a shorter
prefill, a different output budget, a different artifact, or aggregate serving
throughput. Short runs are allowed only as correctness or compile smokes and
must never be reported as performance evidence.

The 80-TPS deadline is 12.8 seconds for 1,024 generated tokens.

## Correctness contract

Every retained candidate must preserve:

- NumPy PCG64 request-level state and cursor progression;
- one uniform consumed for each weighted choice and no speculative cursor
  advancement;
- NumPy-generated request uniform tape installed as device input;
- candidate support and candidate ordering;
- descending-logit ordering with ascending-token-ID tie-breaking;
- top-p cumulative-before behavior;
- positive-probability filtering;
- normalization and `searchsorted(..., side="right")` behavior;
- first-rejection selection;
- correction-token and all-accepted bonus selection;
- exact softfloat64 production sampling arithmetic;
- variable-length M=4 verifier output;
- authoritative target and MTP cache/state selection;
- output token digest;
- existing QSA, fused QSA K/V gather, GDN, FRSpec, MoE, context-copy,
  fixed-M4, M4 stage-3, and cache capture/commit routes;
- maximum request output length of 16,384 tokens.

Only an already-approved ascending-token-ID tie-break difference may change
semantics. Float32 results are diagnostic only and can never establish
production parity.

Validate invariant model metadata, topology, dtype, shapes, route eligibility,
and self-checks once during construction/installation. The enabled hot path
must contain no metadata revalidation, environment reads, proof counters,
`eligible-or-stock` branches, or silent fallbacks. Install the control or
candidate route at construction time and call it directly.

## Current measured control and historical evidence

The last recorded retained exact three-seed control is the owned-row PLE
handoff route:

| Seed | Decode seconds | Decode TPS |
| --- | ---: | ---: |
| 20260829 | 16.062098167 | 63.752567650 |
| 20260830 | 16.285677750 | 62.877334043 |
| 20260831 | 15.055215250 | 68.016297542 |
| Mean | 15.800997056 | 64.882066412 |

Recorded maximum peak allocation: 89,145,498,216 bytes. The prior artifact was
named
`.benchmark-artifacts/pr391/rebench3-2650-owned-row-ple-handoff-seeds-16k-1k.json`.
The ignored artifact was not visible in this worktree when this steering file
was written. Treat these as prior-run evidence, verify any artifact before
using it, and generate a fresh matched control when a candidate is ready.

The preceding copy-free replay/device-next-D3 control recorded a
15.844192555-second mean and 64.707663635 TPS. An older corrected baseline
recorded 15.414148 seconds and 66.5219 TPS; it is historical context, not a
matched current control.

The retained D3/PLE scheduling changes have so far been effectively neutral.
That is acceptable as a composable foundation, but it is not proof that queue
starvation was removed.

## Exact trace diagnosis

Across the retained three-seed trace, the D3-to-target handoff contains two
separate 1,088-event gap families:

- selector/copy to PLE q4 dequantization: 2.223364 seconds;
- PLE dequantization to the first target gather: 2.124750 seconds.

Their combined diagnosis is:

- 1,088 ordinary events;
- 4.348114 seconds total;
- 3.754339 seconds host-late;
- 0.593775 seconds driver time;
- 3.9964 ms total per event;
- 3.450679 ms host-late per cycle.

The target prefix available before PLE is target embedding plus decoder layer
0. The production configuration has `ple_layer_ids=[2]`, which is one-based;
PLE therefore begins at Python layer index 1, the second decoder layer.

Measured strictly PLE-independent target GPU work:

- embedding plus layer 0 mean: 0.575398 ms/cycle;
- median: 0.538834 ms/cycle;
- p05/p95: 0.524190/0.682242 ms/cycle;
- total across the three seeds: 0.639843 seconds;
- including the six layer-0 tail operations gives a realistic
  0.62-0.66 ms/cycle envelope.

Therefore 0.20-0.24 seconds per request, approximately 0.7-1.0 TPS, is an
optimistic available-GPU-work ceiling for the first fork, not an expected
saving. Primary-token PLE rows are already prefetched before initial D3 and
after the carried-D3 queue; the new fork can overlap only the remaining
token-dependent row work and join. It cannot by itself recover 80 TPS. Its
purpose is to remove a real portion of the measured boundary and establish the
correct overlap architecture.

## Immediate implementation: post-D3 target-prefix fork

The D3 selector must remain semantically unchanged. The desired topology is to
enqueue the target prefix immediately after the final device D3 token tensor
exists and before Python decodes or performs PLE bookkeeping. This topology
does not exist yet. In the current source, the relevant points are:

- `_pr391_run_float32_d3_core` in `mtplx/generation.py` creates one packed
  `(1, 3)` D3 token root and explicitly runs `mx.eval(result[0])` before it
  returns on the non-carried path;
- its call site begins near the prior line 10751;
- the installed fixed-M4 verifier is invoked much later near the prior line
  11641;
- `Qwen4ExpTextModel._decode_layers_compiled` in
  `mtplx/models/qwen4_exp.py` groups a first run containing layer 0, then
  performs the eager layer-1 PLE injection. This is only a partitioning oracle:
  the installed fixed-M4 path currently invokes one outer compiled verifier
  graph, so there is no independently schedulable prefix ABI yet.

Add two construction-bound compiled entrypoints with explicit state/capture
plans: a prefix and suffix. Restructure the D3 handoff so the packed device token
root remains available, build the four-token target input from a device primary
token plus that root, schedule all prefix outputs, and only then perform the
host materialization needed for exact PLE row lookup. Simply calling the new
prefix after `_pr391_run_float32_d3_core` returns preserves the current blocking
boundary and is a failed implementation.

Audit the initial/non-carried and carried-D3 paths separately. The initial path
currently blocks inside `_pr391_run_float32_d3_core`; the carried path reuses an
already-queued result with different cache ownership. Neither current ABI
retains a device primary-token root: the initial route receives a host `int`,
and the carried record stores only result/future/descriptor/cycle offsets.
Extend the installed ABI so the target input is created/rooted before host
decode; silently rebuilding it from a host primary preserves the boundary.
Both paths must root every device result and cache leaf through the suffix join
without duplicating D3 or rebinding the live cache prematurely.

Install the split only for the physical fixed-M4 case where
`verified_token_count == 4`. Shortened terminal D2/D1 cycles and context-copy
phases must retain their existing generic routes.

### Safe first-stage split ABI

Conceptual prefix:

```text
prefix(input_ids, layer0_conv_state, layer0_delta_state)
  -> layer0_hidden [1, 4, 10240]
  -> layer0 capture leaves
  -> layer0_conv_state_out
  -> layer0_delta_state_out
```

The prior production census found two layer-0 state inputs and six layer-0
capture leaves. Schedule/evaluate ownership for every prefix output, not only
the hidden tensor.

Conceptual suffix:

```text
suffix(layer0_hidden, input_ids, ple_aux, state/capture leaves for layers 1..47)
  -> logits [1, 4, 248320]
  -> verifier hidden [1, 4, 10240]
  -> returned PLE embedding when the selected construction route requires it
  -> captures and state outputs for layers 1..47
```

The prior production census found 132 suffix state leaves and 213 suffix
capture leaves. The exact counts must be revalidated at construction against
the installed model; do not recheck them per cycle.

### Ownership requirements

- Partition the verifier shadow cache: prefix owns layer 0, suffix owns layers
  1 through 47.
- Do not commit layer-0 state to the authoritative cache before suffix
  construction/execution and the existing accepted-frontier commit succeed.
- Keep all prefix tensors and cache/capture references alive until the suffix
  has been scheduled and the existing commit path has selected the
  authoritative frontier.
- Disable prefix donation initially. Enable it only in a separate measured
  stage after ownership is proven.
- Return verifier hidden explicitly. Do not communicate it through mutable
  `_last_widened` or similar side state.
- Preserve existing capture scopes and PLE scopes.
- Do not duplicate target embedding or layer 0 in the suffix.
- Do not create a new stream merely to appear parallel. The desired overlap is
  CPU work against already-enqueued GPU work on the established stream.

### Separately gated extension

After the basic prefix has an individual A/B, the only clear additional model
work available before PLE rows arrive is the query side of the layer-1 PLE:

```text
norm_query(layer0_hidden)
```

`key_proj(ple_embedding)` and `value_proj(ple_embedding)` require the PLE
embedding and cannot start early. Layer-1 attention/MLP and all later layers
depend on the PLE injection and cannot start early. Add `norm_query` only as a
second candidate and retain it only if its saved work exceeds added output,
materialization, and command-buffer cost.

Fixed masks, positions, and required cache/state copies may also be moved into
the prefix only when they are genuinely independent, already required, and a
trace attributes time to them. Do not add speculative preparation work.

## Optimization queue A: synchronization and GPU-starvation removal

Complete and measure these in order. Do not merge candidates before each has an
individual result.

1. **Post-D3 CPU/GPU fork:** CPU PLE row hashing/gather in parallel with target
   embedding and complete layer 0. This is the active task.
2. **PLE-independent query extension:** add layer-1 `norm_query` to the prefix
   only after stage 1 is measured.
3. **Raw-q4/dequant handoff revisitation:** determine whether the raw packed PLE
   arrays can enter the compiled suffix without the copy, output, and command
   buffer costs that defeated the earlier route. Preserve the explicit BF16
   dequantization boundary and exact returned embedding used for commit.
4. **PLE first-consumer fusion:** express q4 dequantization, key projection,
   value projection, and the already-computed query as one wider dependent
   graph or shape-specific fusion. Avoid a train of tiny dispatches.
5. **Compact the existing exact target decision chain:** target K20 shaping,
   softfloat64 decision, device target-state commit, device MTP replay, and
   device next-D3 construction are already wired before host decision decode.
   The current replay constructs S0/S1/S2/S3 histories and selects among all
   four. Trace whether that all-width work and its command scheduling merely
   replaced the old idle gap; then design a genuinely compact selected
   transition without reintroducing a host read.
6. **Additional handoff removal:** investigate the previously observed
   112-event boundary only after direct trace attribution; it has not been
   proven to be a bonus-token boundary.
7. **Copy and command-buffer cleanup:** remove only copies and empty/sentinel
   dispatches shown by the current trace to be unnecessary. Verify that removal
   does not introduce queue backpressure or force later materialization.
8. **Compact FRSpec sampling:** consider only after synchronization work. Total
   measured draft-sampler GPU work was 0.156428 seconds, so its absolute saving
   cannot be 0.3 seconds.

Original diagnostic ceilings for the major device-chain candidates were:

| Candidate | Associated-time ceiling | Confidence |
| --- | ---: | --- |
| Exact device D3 chain with NumPy uniform tape | 1.268 s total; 1.022 s host-late | high attribution |
| Device target accept/correct pipeline | 1.268 s total; 1.081 s host-late | medium-high |
| All-accepted/bonus-like device handoff | 0.233 s total; 0.208 s host-late | low-medium |
| Compact FRSpec draft sampling | at most 0.156 s; ideal estimate at most 0.115 s | medium bound |
| Batch RNG generation alone | approximately 0.0035 s | high |

These are ceilings, not predicted production savings. A host-late interval
identifies an empty-queue period but does not prove that the full interval is
removable.

## Optimization queue B: compute and kernel work after starvation

Queue these after the synchronization work is nailed down. The full inventory
and its arithmetic constraints live in
`docs/perf/pr391-qwen38-parallelization-opportunities.md`.

1. **PLE projection/gating fusion:** combine independent key/value/query work,
   then collapse small gate, sigmoid, multiply, normalization, and short-conv
   dispatches where the census supports it.
2. **MoE grouped execution:** profile selected-expert work; consider packed
   gate/up, fewer weighting/reduction dispatches, row ownership/grouping, and
   shared/routed addition. Preserve the installed grouped/sorted qmm route.
3. **QSA surrounding work:** retain the existing QSA gather and fused K/V
   kernel; measure packed Q/K/V projection, indexer/projection overlap, and
   removable normalization/reshape/cache-offset dispatches.
4. **GDN fixed-M4 work:** profile q/k/v/a/b projection packing, the existing
   fused convolution/normalization and gated-delta kernels, state-window
   copies, and recurrence-to-hyper-connection boundaries. Preserve recurrence
   association and every committed state.
5. **Hyper-connection/residual dispatch collapse:** fuse exposed trains of
   normalization, gates, masks, casts, copies, and residual mixing only after a
   command-buffer census identifies them.
6. **Draft MTP/FRSpec GPU-busy reduction:** after host boundaries are removed,
   profile compact support extraction and softfloat selection without
   replacing the installed MTP QSA and FRSpec heads.
7. **Fixed-M4 target GPU-busy reduction:** profile dense, attention, MoE, GDN,
   and decision kernels across the four verifier rows while preserving causal
   and recurrent order.
8. **Multi-request batching:** evaluate separately as serving throughput work;
   it is not evidence for the canonical single-request PR391 target.

For compute candidates, profile real production shapes before transplanting an
optimization. Preserve arithmetic, ownership, tiling, data layout, compilation
behavior, and installed routes. A topology with the same name is not proof that
a kernel fits this model.

## Previously tested failures: do not repeat blindly

### Raw q4 inside the compiled verifier

- Exact three-seed mean: 15.863136348 seconds, 64.630029047 TPS.
- Delta versus the retained 64.882066412-TPS control: -0.252037365 TPS.
- Correctness: exact.
- Cause: it moved work but did not create useful CPU/GPU overlap; extra graph
  inputs/outputs and command-buffer scheduling increased host-late by roughly
  0.4651 ms/cycle.
- Status: retain the dormant construction route for later composition, but
  leave it off.

### Synchronous all-miss owned-row path

- Exact three-seed mean: 15.940566778 seconds, 64.345262397 TPS.
- Correctness: exact.
- Cause: synchronous PLE row resolution exposed the lookup on the critical path
  and removed useful scheduling overlap.
- Status: off.

### Early full-window PLE prefetch

- Exact three-seed mean: 15.864444069 seconds, 64.622187397 TPS.
- Correctness: exact.
- It recovered approximately 76 ms versus synchronous all-miss, but remained
  approximately 63 ms slower than the retained control.
- Cause: it was not true parallel computation; its timing and bookkeeping did
  not cover enough independent GPU work and added overhead.
- Status: hooks may remain dormant for future composition, but the production
  construction route and hot-path calls must remain off.

### Incremental D1/D2-root PLE fan-out

- Result: exact experiments regressed by approximately 324-554 ms.
- Cause: computing/fetching multiple speculative PLE row families increased
  CPU work and scheduling without enough certain overlap; most work did not
  belong to the final exact D3 window.
- Status: rejected. Do not rediscover speculative D1/D2 row fan-out while
  implementing the post-D3 exact-window fork.

### Empty command-buffer suppression

- Result: approximately +3.16 seconds of host-late time and parity failure.
- Cause: the apparently empty/sentinel command buffers carried scheduling and
  backpressure consequences; suppressing them changed when dependencies were
  committed/materialized instead of simply deleting free overhead.
- Status: off. Never remove an apparently empty dispatch without tracing both
  dependency ownership and queue timing.

### All-width device replay / early next-D3 scheduling

- Result: effectively throughput-neutral while exact.
- Interpretation: it improved composition/routing but did not remove the
  dominant host decision boundary. The current implementation builds all
  S0/S1/S2/S3 MTP history candidates with `mx.where`; it is not a compact
  selected-width replay.
- Status: keep as the retained foundation unless a matched A/B proves it now
  harms the new split.

## Minimal testing discipline

Do not spam broad test suites before confirming a candidate's performance
characteristics. Use the smallest test that rules out the concrete failure
mode, followed by the exact benchmark.

For each stage:

1. Write one focused failing test for the new boundary or ownership contract.
2. Implement only that stage.
3. Run the focused static/CPU test set.
4. If MLX/Metal executes at all, use the guarded workflow even for a self-check
   or focused test.
5. Run one guarded correctness/compile smoke with safely bounded memory.
6. Run an exact 16K-prefill/1K-output, temperature-1 matched A/B. A short
   workload is not a performance gate.
7. Profile only if the A/B is positive or a precise failure cause is needed.
8. Run all three exact seeds only for a candidate that survives the individual
   gate, or when the user explicitly requests the full negative characterization.
9. Run broader regression tests only after the performance route is selected.

### Split-specific correctness gate

Before performance claims, compare split and monolithic fixed-M4 execution over
at least two consecutive verifier windows. Start the monolithic and split arms
from identical but independent cloned cache/capture/state trees. Each arm
mutates its own shadow state; advance the two branches independently into the
second window. Never execute both arms sequentially against the same live or
shadow objects. Require exact equality for:

- layer-0 hidden output;
- all prefix captures;
- layer-0 convolution and delta states;
- suffix logits and verifier hidden;
- every suffix capture/state leaf;
- returned PLE embedding when required;
- accepted count, correction/bonus result, and committed frontier.

Then require the production three-seed gate to preserve token digests, PCG64
start/final state and cursor, verifier counts, acceptance/correction/bonus
statistics, variable output length, and route receipt.

Do not add hot-path counters to prove engagement. Derive engagement after the
timer from existing events, compiled-entry statistics, route receipts, and
trace structure.

The mandatory report schema below is exhaustive, not permission to run every
diagnostic for an obvious early loser. Mark trace-only fields `N/A: early
stopped after matched A/B` when the result is already decisively negative and
the causal boundary is known. Run an additional profiler only when it can
distinguish competing causes or inform the next candidate.

### Split-specific performance gate

The trace must show real overlap, not merely moved work:

- target embedding/layer 0 begins before PLE raw-row preparation/dequantization
  completes;
- D3-to-prefix plus prefix-to-PLE host-late decreases by at least
  0.45 ms/cycle, or at least 0.163 seconds/request at the observed cycle count;
- GPU busy increases by no more than 0.05 ms/cycle;
- driver time increases by no more than 0.05 ms/cycle;
- no material increase in empty/sentinel command buffers;
- end-to-end decode time improves against a fresh matched control.

The 0.45-ms trace threshold is an attribution gate, not a guarantee that normal
run-to-run noise will produce a large one-seed TPS delta.

## GPU, memory, service, and fan safety

Every MLX/Metal execution—including model loading, compilation, profiling,
self-checks, and tests—must be launched only while the parent guard holds:

```text
/tmp/mtplx-gpu-exclusive.lock
```

Never steal or delete the lock. A free lock is not proof of safe memory. Before
a full-model run, account for current wired memory, the resident service,
caches, and the candidate compile/graph peak. If the candidate peak is unknown
or not hard-bounded with safe headroom, establish a bound using static
accounting or smaller shapes before the full model.

Canonical invocation:

```sh
/opt/homebrew/bin/python3 \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 120 \
  --timeout-seconds 900 \
  --child-timeout-seconds 600 \
  -- your-command arguments...
```

The guard must remain alive throughout the GPU child. It must unload only the
captured production service, restore that exact service configuration, and
verify a matching restored process and model identity before the workflow is
released. The last observed
served model identity was `mtplx-flash-next-optimized-speed`; always capture and
use the live pre-run identity rather than assuming this note is current.

After every success, failure, timeout, interruption, or crash, verify:

- the guarded child exit status and absence of owned descendants;
- `/health` is successful and has no active request;
- `/v1/models` exactly matches the pre-window tuple;
- no process owns `/tmp/mtplx-gpu-exclusive.lock`;
- memory pressure and swap are not unexpectedly worse;
- fan control is restored to the normal/default Thermal Forge mode, never left
  forced at maximum. The canonical GPU guard does not manage Thermal Forge: do
  not guess a command or use `sudo`. Capture its mode read-only and perform a
  restore only with the exact previously verified, explicitly authorized
  Thermal Forge action.

If the API is down while the lock is free, treat it as a service incident, not
permission to load the model.

## Mandatory result and failure reporting

Every candidate, including a negative result, gets one compact report with all
of the following fields. Do not say only "failed," "slower," "drifted," or
"fragmented."

```text
Candidate:
Hypothesis and exact boundary targeted:
Construction route/control route:
Files and functions changed:
Source revision and dirty-diff identifier:
Exact command and artifact path:
Workload, model/artifact identity, seeds, and order:

Correctness:
  token digest per seed:
  first differing cycle/token, if any:
  PCG64 start/final state, cursor, and draws used:
  verifier calls/counts and accepted-count histogram:
  correction/bonus counts:
  cache/capture/state parity:
  route receipt and fallback/demotion counts:

Performance:
  control decode seconds/TPS per seed and mean:
  candidate decode seconds/TPS per seed and mean:
  absolute and percentage deltas:
  peak allocation and delta:
  PLE hits, misses, hit rate, and prefetch batches:

Trace attribution:
  target boundary event count:
  host-late total and per cycle, control vs candidate:
  driver total and per cycle, control vs candidate:
  GPU busy/idle and utilization, control vs candidate:
  command-buffer count and empty/sentinel count:
  proof that work overlapped rather than merely moved:

Failure cause:
  first violated dependency/invariant or measured cost center:
  evidence distinguishing arithmetic, state routing, host wait, driver
  scheduling, added GPU work, copy/materialization, compile, or memory cause:
  why the hypothesis did or did not hold:

Disposition:
  retain enabled, retain construction-disabled for later composition, or
  remove:
  next individually attributable candidate:

Operational receipt:
  child exit/signal:
  service identity restored and health verified:
  lock released:
  fan mode normal:
```

For correctness drift, report the first differing weighted choice and determine
whether the cause is candidate ordering, arithmetic/reordering, RNG state,
acceptance, correction, or cache/state routing. Compare against the unchanged
CPU NumPy reference and exact softfloat64 production path. Do not patch later
state to hide an earlier divergence.

For a performance regression, report which term grew: GPU busy work, GPU idle
host-late, driver delay, command-buffer count, copy/materialization, or CPU PLE
time. A scheduling change that only moves a gap is not a successful starvation
optimization.

For a crash, preserve the exact command, exit code or signal, last relevant log
lines, memory state, owned-child cleanup, service restoration, lock release,
and the narrowest reproducing stage. Do not immediately rerun a full benchmark.

## Stop and promotion rules

- Never claim an optimization from profiler ceilings alone.
- Never compare profiler TPS directly with non-profiled production TPS.
- Never combine several candidates to hide an individually negative result.
- Keep a useful but neutral compositional route construction-disabled when it
  may support a later measured fusion; do not leave its runtime branch in the
  enabled hot path.
- Do not update the PR body before exact parity and corrected three-seed mean
  throughput exceed 80 TPS.
- Reaching 80 TPS still requires reporting every seed, mean decode seconds/TPS,
  peak memory, digests, PCG64 receipt, verifier statistics, PLE hit/miss data,
  route receipt, trace attribution, service restoration, lock release, and
  normal fan mode.

The immediate deliverable is not a speculative megakernel. It is a correct,
construction-installed post-D3 target-prefix fork whose trace proves that CPU
PLE work and GPU target work actually overlap.
