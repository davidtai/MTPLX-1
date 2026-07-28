# Laguna Prefill Admission Latency

## Status

Approved for implementation on 2026-07-27.

## Problem

Laguna's dual-lane scheduler admits pending work only between synchronous
`BatchGenerator.next()` calls. The production launcher configures a 1,024-token
prefill chunk and zero initial cohort wait. A request arriving for the empty
Cline slot while the background lane is inside a prefill chunk therefore waits
until that chunk returns.

A live background-first reproduction recorded:

- Cline server queue wait: 0.740 seconds;
- Cline time to active: 0.775 seconds;
- Cline client TTFT: 1.304 seconds;
- simultaneous class state: one Cline and one background request;
- both request receipts: `ar_batch_max_observed=2`.

An earlier run following a cache-tier smoke workload amplified the same
boundary to a 118-second wait. The Cline provider state was inspected
read-only and already sends `X-MTPLX-Client: cline`; this is not a traffic-class
misclassification.

## Scope

Measure the unchanged 1,024-token production prefill chunk against 512, 256,
and 128 tokens on the real Laguna oQ4e serving path. Promote the largest
candidate that meets every acceptance gate, then make that value the canonical
launcher default.

The change is construction-time scheduler configuration. It must not add
per-token validation, eligibility checks, fallback routing, or engagement
counters to the measured path.

## Non-goals

- Do not change the fixed-M2 router kernel, MoE arithmetic, row ownership,
  tiling, data layout, or decode geometry.
- Do not run two independent model execution loops.
- Do not change the one-slot-per-class non-borrowing contract.
- Do not add mid-kernel GPU preemption.
- Do not treat a small initial cohort delay as a substitute for interruptible
  prefill admission.

## Benchmark design

Run under `/tmp/mtplx-gpu-exclusive.lock` with production Laguna unloaded by
the existing guarded runner. Use the same checkout, model, scheduler, fixed-M2
environment, prompts, seeds, and sampling controls for every candidate. Vary
only `prefill_chunk_tokens`.

For each value:

1. Start from an idle scheduler.
2. Submit a deterministic background request with a realistic
   leaderboard-sized cold prompt.
3. As soon as the background request is active, submit a deterministic
   Cline-labelled request.
4. Capture Cline server queue wait, time to active, TTFT, background prefill
   tokens per second, simultaneous per-class state, per-request decode
   throughput, M=2 engagement, output digests, errors, and service health.
5. Repeat enough times to distinguish stable latency and throughput from a
   cold compile or thermal outlier.
6. Re-run the unchanged 1,024-token control after candidates to detect drift.

Do not promote results contaminated by unrelated active or pending requests,
model reload overlap, a construction error, or missing request receipts.

## Acceptance gates

Promote the largest candidate satisfying all of the following:

- Cline queue latency p95 is below 250 milliseconds while background prefill
  is active.
- Median background prefill throughput regresses by no more than 5% from the
  bracketed unchanged 1,024-token control.
- Both traffic classes are simultaneously active and both receipts report
  `scheduler_lane=ar_batch` and `ar_batch_max_observed=2`.
- Deterministic candidate output digests match the corresponding unchanged
  control digests.
- M=2 aggregate decode throughput does not regress beyond run-to-run noise.
- The existing Laguna launcher and focused server tests pass.
- Startup still validates and self-checks all 47 fixed-M2 router layers before
  serving.

If no candidate clears every gate, leave 1,024 as the production default and
report that smaller prefill chunks are not promotion-ready.

## Implementation and rollout

After measurement, change only the canonical Laguna launcher, its focused
contract tests, and the Laguna documentation/benchmark receipt needed to
record the promoted value. The launcher must report the selected value in
`--print-config` and startup logging.

Restart production only through the existing guarded service workflow. Verify
the restored service reports the promoted chunk, reproduce the
background-first Cline join, and preserve the previous 1,024-token launch
configuration as the rollback.

## Failure modes

- **Smaller chunks improve queue latency but lose prefill throughput.**
  Reject values beyond the 5% regression gate.
- **Cold compile, memory teardown, or thermal drift masquerades as scheduler
  latency.** Use repeated runs and a bracketed unchanged control; reject
  contaminated trials.
- **Toy prompt geometry hides the real leaderboard cost.** Use realistic cold
  prompt sizes and report token counts from server receipts.
- **A candidate changes decode behavior despite being a prefill setting.**
  Require M=2 receipts, deterministic digest parity, and decode-throughput
  comparison before promotion.

## Execution record

Executed on 2026-07-27 under `/tmp/mtplx-gpu-exclusive.lock` with the
production service unloaded by the guarded runner. The exact receipt is
`benchmarks/results/laguna-prefill-admission-m5max-2026-07-27.json`.

| Chunk | Cline queue p95 | Background prefill median | Aggregate decode median | Result |
|---:|---:|---:|---:|---|
| 1,024 control | 0.713 s | 891.21 tok/s | 37.69 tok/s | unchanged control |
| 512 | 0.485 s | 822.35 tok/s | 32.17 tok/s | rejected |
| 256 | 0.357 s | 749.05 tok/s | 33.88 tok/s | rejected |
| 128 | 0.302 s | 549.40 tok/s | 32.11 tok/s | rejected |

Every trial completed without an HTTP or scheduler error and both request
receipts observed M=2. No candidate reached the 250-millisecond queue target.
Every candidate also exceeded the five-percent throughput regression gate;
512 and 128 additionally changed deterministic output digests.

No launcher value was promoted. Production was restored healthy with the
unchanged 1,024-token chunk, and the GPU lock was released. Avoiding this
latency without the measured global-chunk regression requires a separately
approved class-aware prefill-yield design rather than further shrinking the
global chunk.
