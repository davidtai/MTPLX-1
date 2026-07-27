# Laguna Dual-Lane Fixed-M2 Design

**Date:** 2026-07-27
**Status:** Approved for planning
**Target:** `Laguna-S-2.1-oQ4e` in `perf/laguna-batch-kernels`

## Goal

Serve one OpenSource Leaderboard request and one Cline request concurrently
through a true two-row Laguna decode step, while guaranteeing that the
fixed-M2 router kernel reads each router weight once for both rows. Preserve
Laguna's existing arithmetic contract and prevent leaderboard saturation from
occupying Cline's reserved capacity.

## Current evidence

The live service is running Laguna in serial mode even though its request
limit is three:

```text
--scheduler-mode serial
--batching-preset latency
--max-active-requests 3
```

The live queue contained three running leaderboard scans and 55 pending scans.
Recent identifiable Cline requests decoded at about 49.6 tok/s once running,
but queueing and prefill reduced observed end-to-end throughput to about
8.5 tok/s. The slowdown is therefore primarily head-of-line blocking, not a
collapse of Laguna's single-row decode kernel.

The existing Laguna AR batch lane has already established the unchanged
performance control at the real checkpoint:

- B1: approximately 56 aggregate decode tok/s.
- B2: approximately 80 aggregate decode tok/s, approximately 40 tok/s per row.
- The currently installed router GEMV launches one threadgroup per
  `(row, expert)`, so B2 launches 512 threadgroups and traverses the
  `[256, 3072]` BF16 router matrix once for each row.

## Scope

This design delivers:

1. A strict two-slot admission policy: one Cline slot and one
   background/leaderboard slot.
2. Explicit request classification at request construction, outside the
   measured decode path.
3. Real B2 execution through the existing `BatchGenerator` AR lane.
4. A Laguna-specific fixed-M2 router projection whose expert-axis owner loads
   each BF16 router weight once and applies it to both rows.
5. Offline route-overlap measurement for representative Cline/leaderboard
   pairs.
6. A separately gated routed-expert coalescing experiment if the measured
   overlap supports it.
7. A portable, repository-owned Laguna launcher in the upstream MTPLX Laguna
   line and `mtplx-moe`, including the dual-lane configuration and operational
   startup hardening.
8. Focused correctness, dispatch, microbenchmark, full-model, and live-serving
   gates before deployment.

## Non-goals

- No second Laguna process and no duplicate model residency.
- No top-k reduction, expert requantization, architecture change, or quality
  trade.
- No MTP; Laguna remains autoregressive.
- No production per-token counters or router instrumentation.
- No slot borrowing. Two background requests may not consume both slots when
  Cline is idle.
- No direct transplant of A3B source, tile sizes, quantization arithmetic, or
  whole-MoE accumulation order.
- No promotion of a routed-expert kernel that merely groups rows without
  proving a shared fixed-weight read.

## Model contract

The installed kernel is specific to the real checkpoint contract:

| Property | Laguna value |
|---|---:|
| MoE layers | 47 |
| Decode rows | 1 or 2 |
| Hidden size | 3072 |
| Experts | 256 |
| Top-k | 10 |
| Routed intermediate size | 1024 |
| Shared intermediate size | 1024 |
| Router | BF16 `[256, 3072]` |
| Routed gate/up | Q4 affine group-128 |
| Routed down | Q4 affine group-128 |
| Shared gate/up/down | Q8 affine group-128 |

All model, dtype, layout, quantization, and self-check validation occurs once
while the optimized route is installed. If the contract is not exact, the
fixed-M2 route is not installed. A requested fixed-M2 launch must then fail
server startup clearly rather than silently enter a stock fallback.

## Architecture

### 1. Logical traffic slots

`_BatchedARJob` receives an immutable `traffic_class` at construction:

```text
cline       explicit request_client_hint == "cline"
background  every other request, including opensource-leaderboard and unknown
```

The service maintains FIFO pending queues and at most one active request for
each class:

```text
background slot: OpenSource Leaderboard and unlabelled/local traffic
cline slot:      Cline only
```

Admission selects the oldest non-cancelled request for each free class. A
request finishing or cancelling releases only its class slot. The next request
from that class may be admitted on the following pump iteration.

This is a logical reservation, not a permanent physical row number.
`BatchGenerator` may compact row order as requests finish. The M2 kernel treats
both rows symmetrically and does not depend on Cline being row zero.

Both clients identify themselves explicitly:

```http
X-MTPLX-Client: opensource-leaderboard
X-MTPLX-Client: cline
```

The leaderboard client adds its header in source. Cline uses its existing
custom OpenAI headers setting. Unknown traffic deliberately joins the
background class so it cannot take Cline's reservation.

### 2. Serving configuration

The Laguna service runs with:

```text
--scheduler-mode ar_batch
--batching-preset latency
--max-active-requests 2
--decode-batch-max 2
--prefill-chunk-tokens 1024
--batch-wait-ms 0
```

`max-active-requests` controls admission and `decode-batch-max` controls the
actual model batch; both must be two. A zero batch wait preserves solo Cline
latency and continuous admission allows the second class to join an active
cohort.

At one active request, decode uses the installed B1 route. At two active
requests, one model invocation consumes both rows and uses the installed M2
route. Prefill remains chunked and may run at up to two rows through
`BatchGenerator`.

### 2a. Repository-owned launcher and provenance

The working Laguna launch configuration is a product artifact, not
machine-local state. A portable `scripts/start-laguna-s21.sh` is committed on
both:

- the upstream MTPLX Laguna development line descended from
  `agent/laguna-support` / `upstream/pr-195`; and
- the `OpenSourceWTF/mtplx-moe` line after the accepted scheduler and kernel
  stack is ported there.

The launcher derives the repository root from its own location, accepts model,
Python, host, port, and memory-policy overrides through documented environment
variables, and contains no `/Users/davidtai`, qwen36-server, or private
worktree assumptions.

The launcher and its documentation explicitly note that Blackwellboy's
operational changes were applied to make the real Laguna serving path work
correctly. The note preserves attribution without treating those operational
changes as evidence for the new fixed-M2 kernel.

### 3. Fixed-M2 expert-axis router projection

The existing Laguna router GEMV uses:

```text
owner:       one threadgroup per (row, expert)
threadgroup: 8 simdgroups x 32 lanes = 256 threads
work:        one 3072-wide BF16 dot
grid at B2:  2 rows x 256 experts = 512 threadgroups
```

The fixed-M2 projection changes the owner to one expert:

```text
owner:       one threadgroup per expert, owning both rows
threadgroup: 8 simdgroups x 32 lanes = 256 threads
work:        two 3072-wide BF16 dots sharing every weight load
grid at B2:  256 experts = 256 threadgroups
```

Each lane keeps `acc_row0` and `acc_row1`. For every `k` block it loads one
BF16 router weight and multiplies it by the corresponding value from both input
rows. Each simdgroup reduces both accumulators. Thread zero combines the same
eight partials in the same ascending order as the current kernel and rounds
each result to BF16 before widening to FP32.

The arithmetic order for each row therefore matches the currently deployed
Laguna router GEMV. The only structural difference is that the weight load is
outside the two-row work.

The projection writes FP32 logits `[2, 256]`. The existing fused sigmoid,
correction-bias, top-10, normalization, and scale kernel remains unchanged.

The router matrix is exactly 1.5 MiB per MoE layer. The explicit traversal
therefore changes from 3 MiB at B2 to 1.5 MiB, avoiding 70.5 MiB across the 47
MoE layers of one B2 decode step.

### 4. Installed route

The model installer constructs and validates fixed callables for:

```text
decode M1 -> existing characterized Laguna router GEMV
decode M2 -> fixed-M2 expert-axis router GEMV
prefill   -> stock phase route
```

The execution path may branch on logical M and phase because those values
genuinely vary at runtime. It may not re-check model metadata, dtype,
quantization, environment variables, or self-check state inside a layer or
token step. The enabled M2 entrypoint calls the M2 kernel directly and has no
`eligible-or-stock` or exception fallback.

### 5. Offline routed-expert overlap probe

A benchmark-only probe records the two top-10 expert sets for each of 47
layers over representative paired decode trajectories:

- Cline-sized prompt paired with a normal leaderboard scan.
- Cline-sized prompt paired with a large leaderboard scan.
- At least three deterministic prompt pairs per workload class.

The artifact stores token counts, prompt hashes, layer number, overlap count,
unique-expert count, and aggregate histograms. It stores no prompt text and
adds no production instrumentation.

### 6. Routed-expert coalescing experiment

This experiment is conditional on the offline overlap result and is not part
of the serving hot path unless it passes every gate.

For fixed B2/top-10, a candidate expert owner uses a 20-slot static ownership
scheme:

1. Owners 0-9 always own row zero's ten expert IDs.
2. An owner also owns a matching row-one slot when the same expert appears in
   row one's top-10.
3. Owners 10-19 execute only for row-one experts absent from row zero.
4. A shared owner loads a Q4 weight tile once and keeps separate row
   accumulators for its one- or two-row mask.

Top-k IDs are unique within each row, so a fixed owner never has multiple
matches in the same row. Runtime comparisons of the two top-10 lists are
legitimate routing work; model and layout validation are not repeated.

Gate/up activations and down-projection results remain row- and slot-addressed.
The existing combine stage consumes results in each row's original top-10 slot
order, preserving route-weight and accumulation order. One overlapping routed
expert represents approximately 4.78 MiB of Q4 weights and metadata per layer.

The routed-expert candidate is rejected if:

- measured overlap is too low to repay union construction and conditional
  ownership;
- a physical one-read source contract cannot be demonstrated;
- it changes full-MoE output beyond the current lossless contract;
- it does not beat the unchanged B2 control reproducibly.

The scheduler and fixed-M2 router remain independently useful if this
experiment is rejected.

## Interfaces

### Traffic classification

```python
TrafficClass = Literal["cline", "background"]

def _ar_batch_traffic_class(
    request_observability: Mapping[str, Any] | None,
) -> TrafficClass:
    ...
```

Classification runs once when `_BatchedARJob` is constructed. The result is
stored on the job and used only by admission and observability.

### Pending and active ownership

The batched service exposes, in its snapshot:

```json
{
  "pending_by_class": {"cline": 0, "background": 0},
  "active_by_class": {"cline": 0, "background": 0}
}
```

Existing aggregate `pending` and `active` fields remain for compatibility.

### Kernel entrypoints

```python
def router_gemv_logits_m1(x, gate_weight): ...
def router_gemv_logits_m2(x, gate_weight): ...
```

Both return FP32 logits at BF16 output precision. The M2 entrypoint accepts
only `[2, 3072]` input and `[256, 3072]` BF16 router weight because the
installer already proved the contract.

## Error handling

- Missing or unrecognized client labels enter the background class.
- A second request from an occupied class remains pending; it cannot borrow
  the other slot.
- Cancellation removes a pending job or releases its active class slot.
- A fixed-M2 install request with a mismatched model contract fails once at
  startup with the mismatched property in the error.
- Kernel self-check failure aborts installation and startup. There is no
  production fallback from an enabled fixed-M2 route.
- Client header values affect scheduling only; they grant no authorization or
  data access.

## Failure-mode check

### Critical: the service batches two leaderboard requests before Cline arrives

Resolved by class-owned active slots and no borrowing. FIFO over a single
pending list is not acceptable.

### Critical: the M2 diagram halves threadgroups but still rereads router weights

Resolved by making the expert the owner and placing the weight load outside
the row arithmetic. Source-contract tests and Metal inspection must prove that
the one loaded value feeds both row accumulators.

### Critical: a whole-MoE port changes Laguna's Q4 arithmetic

Resolved by keeping the fixed-M2 router as the required promoted kernel and
making routed-expert coalescing a separate lossless promotion gate. A
tolerance-only A3B result is insufficient for Laguna.

### Minor: a very large background prefill temporarily increases Cline TTFT

Accepted as a bounded limitation of one shared GPU and chunked prefill. The
realistic paired benchmark must measure it, and the 1024-token prefill chunk
prevents a full large prompt from becoming one uninterruptible admission unit.

### Minor: strict reservation reduces leaderboard-only throughput

Intentional. When Cline is absent the leaderboard uses one slot, preserving
the second slot for interactive work.

## Testing strategy

### Scheduler tests

- Cline plus background are admitted together at capacity two.
- Two background requests admit only one.
- Two Cline requests admit only one.
- Unknown traffic is background.
- Cancellation releases the correct class.
- Completion admits the next request from the same class.
- Snapshot retains aggregate fields and reports per-class fields.

### Kernel correctness

- M2 source-contract test proves one expert owner and one weight load feeding
  both row accumulators.
- Random and adversarial `[2, 3072]` inputs compare M2 logits bitwise with two
  invocations of the current per-row kernel.
- Expert IDs and route weights match the current B2 route.
- The construction self-check cycles all 47 distinct router weights.
- B1 token digest is unchanged.
- B2 token digests match the unchanged B2 control for deterministic prompts.

### Performance

Measurements run under the existing exclusive Metal lane and thermal protocol.
Each comparison uses unchanged control and candidate in paired order.

1. Chained 47-layer router microbenchmark, current B2 versus fixed-M2.
2. Full Laguna B1 control versus installed stack to prove no B1 regression.
3. Full Laguna B2 control versus installed stack at the existing benchmark
   shapes.
4. Paired Cline/normal-leaderboard and Cline/large-leaderboard serving runs.
5. Dispatch census confirms 256 M2 router threadgroups and the intended kernel.

The router kernel promotes only if:

- correctness gates are green;
- B2 median decode-cycle improvement is at least 1%;
- at least three of four paired repetitions improve;
- B1 median decode throughput does not regress by more than 1%.

The serving feature promotes only if:

- one request from each class is simultaneously active;
- leaderboard saturation cannot occupy the Cline slot;
- aggregate B2 decode throughput remains above the unchanged B1 aggregate;
- Cline no longer waits for completion of an already-running leaderboard
  decode request before joining the batch.

## Rollout

1. Implement and verify strict class admission without changing the live
   service.
2. Implement the fixed-M2 router under a dedicated load-time flag and run
   correctness plus performance gates.
3. Add the leaderboard client header.
4. Configure Cline with `X-MTPLX-Client: cline`.
5. Restart Laguna with the B2 scheduler flags and promoted M2 kernel.
6. Confirm startup installation report, per-class snapshot, two active rows,
   request labels, and live decode throughput.
7. Retain the previous serial launch command as the rollback path.
8. Run the routed-expert overlap probe and decide independently whether a
   coalescing kernel merits implementation and promotion.
9. Commit the portable launcher and its tests on the upstream MTPLX Laguna
   development branch.
10. Port the accepted dual-lane commits and the same launcher to a clean
    branch based on `mtplx-moe/main`, verify it there, and preserve both
    repository copies as supported entrypoints.

## Acceptance criteria

- The live service processes one Cline prompt and one leaderboard prompt in
  the same B2 decode cohort.
- Leaderboard backlog cannot consume the Cline reservation.
- The installed M2 router reads each router weight once for both rows by
  construction.
- The M2 kernel preserves the current Laguna router and token-output contract.
- The unchanged B2 benchmark is beaten by the defined promotion gate.
- No production hot-path validation, fallback, or diagnostic counters are
  added.
- The portable launcher, Blackwellboy provenance note, and launcher tests are
  committed in both the upstream MTPLX Laguna line and `mtplx-moe`.
