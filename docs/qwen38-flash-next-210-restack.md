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
- Seeds: `20260829`, `20260830`, and `20260831`
- N-gram hot-row cache: bounded at 1 GiB

The 29.8 GiB n-gram table stays in a file-backed memory map. Production uses
the required bounded 1 GiB hot-row cache above the memory map. All optimization
decisions use this configuration.

## Step 1 production boundary

Decode throughput excludes prefill. Step 0 and Step 1 used the same three
seeds. Each pair had the same output digest, MTP trajectory, and work counts.

| Seed | Step 0 lazy | Step 1 batched | Paired change | Accepted / drafted |
|---:|---:|---:|---:|---:|
| `20260829` | 52.27 tok/s | 52.67 tok/s | +0.76% | 578 / 1,282 |
| `20260830` | 53.43 tok/s | 53.78 tok/s | +0.65% | 555 / 1,220 |
| `20260831` | 60.57 tok/s | 60.96 tok/s | +0.65% | 535 / 1,057 |
| Mean | **55.42 tok/s** | **55.80 tok/s** | **+0.68%** | 1,668 / 3,559 total |
| Median | **53.43 tok/s** | **53.78 tok/s** | **+0.65%** | - |

Weighted verifier cost fell from 38.04 to 37.64 ms per call. All six runs
generated exactly 1,024 tokens and had zero repair. Step 1 remains a small but
repeatable win.

## Step 2 production boundary

Step 2 installs one construction-bound compiled replay for physical M4 target
verification. Shorter adaptive windows use the normal Qwen4 family capture
route. The enabled M4 path does not repeat model checks or fall back silently.

| Seed | Decode throughput | Accepted / drafted | Verify calls | Repair |
|---:|---:|---:|---:|---:|
| `20260829` | 63.03 tok/s | 613 / 1,065 | 364 | 0 s |
| `20260830` | 67.63 tok/s | 666 / 1,014 | 344 | 0 s |
| `20260831` | 72.22 tok/s | 603 / 912 | 313 | 0 s |
| Mean | **67.63 tok/s** | 1,882 / 2,991 total | 340.3 | **0 s** |
| Median | **67.63 tok/s** | - | - | **0 s** |
| Range | **63.03-72.22 tok/s** | - | - | **0 s** |

All three runs generated exactly 1,024 tokens. Each run traced once and had
zero fallback and zero repair. Weighted verifier cost was 36.38 ms per call.
The three-seed Step 2 mean is 21.19% above the Step 1 mean.

The compiled PLE block changes bf16 operation grouping. Close sampling
decisions can change, so Step 2 uses aggregate and per-window measurements.
The earlier retained controls measured 70.87 and 74.93 tok/s. Their 72.90
tok/s mean remains valid historical evidence; 74.93 is the highest single
retained observation, not the three-seed mean.

## Rejected scheduling candidate

The lazy stochastic D3 candidate built all three draft depths before one
terminal device read. It did not improve the three-seed production result.

| Seed | Lazy D3 throughput | Accepted / drafted |
|---:|---:|---:|
| `20260829` | 68.39 tok/s | 663 / 976 |
| `20260830` | 63.76 tok/s | 658 / 1,068 |
| `20260831` | 62.82 tok/s | 523 / 1,023 |
| Mean | **64.99 tok/s** | 1,844 / 3,067 total |
| Median | **63.76 tok/s** | - |

The candidate was 3.90% below the Step 2 mean. Its weighted draft cost was
2.71 ms per drafted token versus 2.54 ms for Step 2. Its non-verifier cost was
9.27 ms per window versus 8.25 ms. The larger lazy graph increased host work,
so this candidate is not retained.

## Accepted optimization hill climb

This table records only changes that remain in the stack. Each new row must use
the production configuration above and must show a matched, repeatable win
against the unchanged prior step.

| Step | Commit | Retained stack | Production evidence | Result |
|---:|---|---|---:|---|
| 0 | `4ce96908` | Unchanged MTPLX 2.10 | **55.42 tok/s mean; 53.43 median** | Three-seed production control |
| 1 | `ffdb8684` | Batch fixed-M4 target distributions | **55.80 tok/s mean; 53.78 median** | +0.68% mean with identical paired work |
| 2 | `c5034156` | Construction-bound fixed-M4 compiled verifier | **67.63 tok/s mean and median** | +21.19% mean vs Step 1; 36.38 ms weighted verify cost |

Invalid cache settings, duplicate defaults, profiler-only runs, rejected
experiments, and unconfirmed samples do not become hill-climb rows. A confirmed
optimization is added here after it is committed.

## Remaining GPU idle time

The MLX profiler measured the same production path. The profiler run is a
diagnostic and is not a promotion result.

| Decode measurement | Result |
|---|---:|
| GPU timeline | 17.195 s |
| Metal GPU work | 13.203 s |
| Metal GPU idle | 3.992 s |
| GPU use | 76.78% |
| One-time M4 compilation gap | 0.753 s |
| Steady idle after compilation | 3.240 s |
| Steady GPU use | 80.30% |
| Explicit GPU-drain wait | 0.025 ms |

The steady idle time contains 2.563 seconds of host-late submission gaps and
0.677 seconds of encoded or driver-side gaps. These are many small gaps, not
one long wait. Removing all idle time still requires a separate GPU-compute
win to reach 90 tok/s.

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
