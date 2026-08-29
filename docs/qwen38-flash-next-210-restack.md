# Qwen3.8 Flash-Next optimization on MTPLX 2.10

This branch builds a clean optimization stack on MTPLX 2.10. It uses the
official `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` model. Each
retained runtime change has a matched result on the production workload.

The closed [PR #368](https://github.com/youssofal/MTPLX/pull/368) contains the
old implementation history, rejected experiments, and detailed benchmark
notes. This document records the new production baseline and each optimization
that remains in the new stack.

## Production configuration

- Runtime base: MTPLX 2.10.0, commit `4ce96908`
- Model revision: `29ba90f82124961d0d902a9ea9bbb1034972af2f`
- MLX: 0.32.2
- Decode mode: native MTP, depth 3
- Workload: 16,384 prompt tokens and exactly 1,024 generated tokens
- Prompt: natural Python programming task
- Reasoning effort: `xhigh`
- Sampler: temperature 1.0, top-p 0.95, top-k 20, min-p 0
- Penalties: presence 0 and repetition 1
- Seed: `20260829`
- N-gram hot-row cache: bounded at 1 GiB

The 29.8 GiB n-gram table stays in a file-backed memory map. Production uses
the required bounded 1 GiB hot-row cache above the memory map. All optimization
decisions use this configuration.

## Step 1 production boundary

Decode throughput excludes prefill. All runs below generated 1,024 tokens with
the same output digest, the same 578 / 1,282 MTP acceptance path, 432 verifier
calls, and zero repair time.

| Result | Decode throughput | End-to-end time | Notes |
|---|---:|---:|---|
| Lazy control 1 | 52.89 tok/s | 33.81 s | Unchanged MTPLX 2.10 target sampling |
| Batched candidate 1 | 53.58 tok/s | 33.59 s | Fixed-M4 target arrays |
| Batched candidate 2 | 54.20 tok/s | 33.31 s | Fixed-M4 target arrays |
| Lazy control 2 | 54.02 tok/s | 33.28 s | Unchanged MTPLX 2.10 target sampling |
| Lazy control mean | **53.46 tok/s** | **33.55 s** | Matched step-0 control |
| Batched candidate mean | **53.89 tok/s** | **33.45 s** | Retained step 1 |

The batched lane improved decode throughput by 0.81% and reduced end-to-end
time by 0.28%. Both paired comparisons favored the candidate. The output
digest, acceptance path, and work counts were identical in all four runs.

Earlier unchanged-control brackets measured 51.29 and 53.29 tok/s mean. The
53.29 result explicitly set compiled GDN, but MTPLX 2.10 already enables that
path. It is a baseline observation, not an optimization step.

## Step 2 production boundary

Step 2 installs one construction-bound compiled replay for physical M4 target
verification. Shorter adaptive windows use the normal Qwen4 family capture
route. The enabled M4 path does not repeat model checks or fall back silently.

| Run | Decode throughput | Decode time | Verify cost | Accepted / drafted | Verify calls | Repair |
|---|---:|---:|---:|---:|---:|---:|
| Fresh step-1 control | 52.38 tok/s | 19.55 s | 37.25 ms/call | 578 / 1,282 | 432 | 0 s |
| Fixed-M4 candidate 1 | 61.69 tok/s | 16.60 s | 36.82 ms/call | 605 / 1,087 | 371 | 0 s |
| Fixed-M4 candidate 2 | 73.88 tok/s | 13.86 s | 36.78 ms/call | 659 / 888 | 307 | 0 s |
| Fixed-M4 candidate 3 | 62.62 tok/s | 16.35 s | 36.88 ms/call | 611 / 1,071 | 365 | 0 s |
| Candidate mean | **66.06 tok/s** | **15.60 s** | **36.83 ms/call** | 625 / 1,015 mean | 347.7 mean | **0 s** |
| Candidate median | **62.62 tok/s** | - | - | - | - | **0 s** |

All candidate runs generated exactly 1,024 tokens. Each run traced once and
had zero fallback and zero repair. Weighted verifier cost fell by 1.13%.
Mean decode throughput improved by 26.12% against the fresh control and by
22.58% against the retained step-1 mean.

The compiled PLE block changes bf16 operation grouping. This can change close
sampling decisions, so the three candidate runs do not have the same output
digest or MTP trajectory. The benchmark records every trajectory instead of
claiming that the full TPS change comes from kernel speed. The stable direct
execution result is 36.78 to 36.88 ms per verifier call.

## Accepted optimization hill climb

This table records only changes that remain in the stack. Each new row must use
the production configuration above and must show a matched, repeatable win
against the unchanged prior step.

| Step | Commit | Retained stack | Production evidence | Result |
|---:|---|---|---:|---|
| 0 | `4ce96908` | Unchanged MTPLX 2.10 | **53.46 tok/s mean** | Matched production control |
| 1 | `ffdb8684` | Batch fixed-M4 target distributions | **53.89 tok/s mean** | +0.81% decode; +0.28% end-to-end |
| 2 | `c5034156` | Construction-bound fixed-M4 compiled verifier | **66.06 tok/s mean; 62.62 median** | +22.58% mean vs step 1; verifier cost -1.13% |

Invalid cache settings, duplicate defaults, profiler-only runs, rejected
experiments, and unconfirmed samples do not become hill-climb rows. A confirmed
optimization is added here after it is committed.

## Remaining GPU idle time

The MLX profiler measured the same production path. The profiler run is a
diagnostic and is not a promotion result.

| Decode measurement | Result |
|---|---:|
| Decode wall time | 19.20 s |
| Metal GPU work | 15.90 s |
| Metal GPU idle | 3.30 s |
| GPU use | 82.81% |
| Idle gaps of at least 10 us | 4,320 gaps / 3.22 s |
| Largest gap | 8.31 ms |
| Explicit GPU-drain wait | 0.02 ms |

The 3.30 seconds consist of many small submission gaps. They are not one long
wait. If all measured idle time disappeared and GPU work stayed unchanged, the
ceiling would be 64.42 tok/s. A 90 tok/s result also needs less GPU work or more
accepted work per verifier cycle.

## What MTPLX 2.10 already contains

The audit found these features in the MTPLX 2.10 Qwen4 path:

- Selected-row QSA gathering and accepted-prefix state commit
- Fused attention projections
- AR pipelining
- Construction-bound compiled GDN
- Fused GDN input, decode, and verify operations
- Fused MoE gate and up projections
- Captured-prefix state commit

The new stack does not copy these features again. New work must preserve the
target path's arithmetic, ownership, shapes, data layout, and compile behavior.

## Use it

Pull the exact model revision:

```bash
mtplx pull Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --revision 29ba90f82124961d0d902a9ea9bbb1034972af2f
```

Start the server:

```bash
mtplx serve \
  --model ~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --model-id mtplx-flash-next-optimized-speed
```

The model defaults select Turbo, native MTP depth 3, batched verification,
`xhigh` reasoning, the temperature-1 sampler, and the bounded 1 GiB hot-row
cache.

## Review rule

Restack one optimization at a time. Check construction-time invariants and
focused parity before the production benchmark. Keep and commit only matched
wins. Record each retained win in the hill-climb table and in the pull request.
