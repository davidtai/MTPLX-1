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

## Step 3 production boundary

Step 3 prepares the exact fixed-M4 n-gram row IDs on the host. It starts the
file-backed row gather before the compiled verifier needs the embedding. This
removes one forced auxiliary-graph wait. The 29.8 GiB table remains mapped and
the hot-row cache remains bounded at 1 GiB.

| Seed | Control | Staged n-gram | Paired change |
|---:|---:|---:|---:|
| `20260829` | 59.84 tok/s | 61.46 tok/s | +2.70% |
| `20260830` | 66.06 tok/s | 68.66 tok/s | +3.95% |
| `20260831` | 71.69 tok/s | 73.89 tok/s | +3.08% |
| Mean | **65.86 tok/s** | **68.01 tok/s** | **+3.25%** |

The candidate won all three pairs. Every run generated 1,024 tokens with zero
repair and zero compiled-verifier fallback.

## Step 4 production boundary

Step 4 selects fixed-M4 n-gram rows from the authoritative host token ledger.
It no longer materializes pending device history back to the CPU before each
verification window. The device history remains the source for capture,
commit, and rollback.

| Seed | Control | Host-owned selection | Paired change |
|---:|---:|---:|---:|
| `20260829` | 63.69 tok/s | 72.28 tok/s | +13.49% |
| `20260830` | 68.55 tok/s | 68.77 tok/s | +0.32% |
| `20260831` | 72.50 tok/s | 73.20 tok/s | +0.97% |
| Mean | **68.25 tok/s** | **71.42 tok/s** | **+4.65%** |

The candidate won all three pairs. Every run generated 1,024 tokens with zero
repair and zero compiled-verifier fallback.

## Step 5 production boundary

Step 5 keeps the routed q4/g32 and shared q8/g64 down projections on the tuned
MLX quantized-matmul path. One small Metal kernel then combines routed scores,
the exact ten-row BF16 reduction, the shared gate, and the final add. It
removes dispatches without replacing the efficient quantized matmuls.

| Seed | Step 4 control | Step 5 combine tail | Paired change | Control accepted/drafted | Candidate accepted/drafted |
|---:|---:|---:|---:|---:|---:|
| `20260829` | 64.62 tok/s | 70.32 tok/s | +8.82% | 615 / 1,062 | 641 / 979 |
| `20260830` | 66.70 tok/s | 67.57 tok/s | +1.31% | 657 / 1,044 | 654 / 1,053 |
| `20260831` | 72.14 tok/s | 74.10 tok/s | +2.71% | 590 / 921 | 602 / 915 |
| Mean | **67.82 tok/s** | **70.66 tok/s** | **+4.19%** | 1,862 / 3,027 total | 1,897 / 2,947 total |

The candidate won all three pairs. Mean decode time fell from 15.131 to 14.512
seconds. Normalized verifier-forward cost fell from 32.04 to 31.20 ms per
call. Every run generated 1,024 tokens, traced once, and had zero repair and
zero compiled-verifier fallback. Temperature-1 runs can follow different
sampling and acceptance paths. The promotion decision therefore uses the
three same-seed pairs, not a raw comparison with an earlier campaign mean.

## Step 6 production boundary

Step 6 reduces the native MTP draft-head work. Qwen4 native MTP uses the
model's Q8 output head, not the configured Q4 draft-only head. Replacing it
with Q4 made proposal acceptance worse. Step 6 instead keeps the original Q8,
group-64, affine arithmetic and gathers only 65,536 code-ranked rows. It writes
those logits into their original positions in a full-vocabulary proposal and
sets all other proposal logits to a zero-probability sentinel. Target
verification still uses every vocabulary row.

| Seed | Step 5 control | Step 6 ranked Q8 head | Paired change | Control accepted/drafted | Candidate accepted/drafted |
|---:|---:|---:|---:|---:|---:|
| `20260829` | 72.26 tok/s | 77.54 tok/s | +7.30% | 632 / 937 | 660 / 921 |
| `20260830` | 69.79 tok/s | 76.13 tok/s | +9.09% | 667 / 1,014 | 662 / 972 |
| `20260831` | 74.51 tok/s | 76.47 tok/s | +2.63% | 604 / 910 | 605 / 907 |
| Mean | **72.19 tok/s** | **76.71 tok/s** | **+6.27%** | 1,903 / 2,861 total | 1,927 / 2,800 total |

Mean draft time fell from 2.481 to 1.903 seconds, a 23.32% reduction. Mean
verifier time also fell from 11.460 to 11.198 seconds. Every run
generated 1,024 tokens with zero repair and zero compiled-verifier fallback.

The built-in row list comes from a generic code corpus that excludes benchmark
fixtures and tests. It covers 99.6446% of a held-out 1,005,404-token code set.
The packaged JSON SHA-256 is
`950adfea038612e28a3839c98c9be73f76f422fcde0596bb4588ac774e7c1fba`.
Installation validates the native Q8/group-64/affine contract once and then
binds the proposal callable directly. It does not add a per-token eligibility
check or fallback.

## Step 7 production boundary

Step 7 reduces the small host gaps between stochastic draft depths. It makes
two construction-bound changes that work together:

- It compiles the fixed B1/S1 stateless work before QSA and installs the graph
  only after exact eager-versus-compiled parity. History updates remain eager,
  and the live QSA cache keeps the same owner.
- It selects the required top 20 proposal rows directly. The old path selected
  80 rows only to prove deterministic ownership of exact ties at the top-k
  cutoff. The full-vocabulary normalizer, top-p filter, support order, and
  NumPy `rng.choice` remain unchanged. An exact cutoff tie may select a
  different tied row.

The result uses the same frozen prompt, three seeds, and production settings as
Step 6.

| Seed | Step 6 control | Step 7 reduced-gap path | Accepted / drafted |
|---:|---:|---:|---:|
| `20260829` | 77.54 tok/s | 79.26 tok/s | 661 / 918 |
| `20260830` | 76.13 tok/s | 75.32 tok/s | 653 / 999 |
| `20260831` | 76.47 tok/s | 78.73 tok/s | 604 / 910 |
| Mean | **76.71 tok/s** | **77.77 tok/s** | 1,918 / 2,827 total |

The arithmetic mean improves by 1.38%. Mean draft time falls from 1.903 to
1.679 seconds, an 11.76% reduction. All runs generated 1,024 tokens with zero
repair and zero compiled-verifier fallback. The promotion rule allows exact
cutoff-tie flips, so the decision uses the fixed three-seed throughput mean and
does not require matching output digests.

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
| 3 | `ccf817e0` | Stage exact fixed-M4 n-gram sidecar inputs | **68.01 tok/s promotion mean** | +3.25% matched three-run mean |
| 4 | `03bc460e` | Select fixed-M4 n-gram rows from the host token ledger | **71.42 tok/s promotion mean** | +4.65% matched three-run mean |
| 5 | `3fe8da54` | Fuse the fixed-M4 combine tail after stock quantized matmuls | **70.66 tok/s promotion mean** | +4.19% matched three-run mean |
| 6 | `b9e1a3dd` | Bind a full-domain proposal over 65,536 code-ranked native Q8 rows | **76.71 tok/s promotion mean** | +6.27% matched three-run mean; draft time -23.32% |
| 7 | `4f0e8604` | Compile fixed draft preparation and remove the 4k cutoff-tie proof superset | **77.77 tok/s promotion mean** | +1.38% fixed three-seed mean; draft time -11.76% |

Invalid cache settings, duplicate defaults, profiler-only runs, rejected
experiments, and unconfirmed samples do not become hill-climb rows. A confirmed
optimization is added here after it is committed.

## Remaining GPU idle time

The MLX profiler measured the same production path. The profiler run is a
diagnostic and is not a promotion result.

| Decode measurement | Result |
|---|---:|
| GPU timeline | 15.165 s |
| Metal GPU work | 12.457 s |
| Metal GPU idle | 2.708 s |
| GPU use | 82.14% |
| Host-late submission | 2.170 s |
| Encoded or driver-side gaps | 0.502 s |
| Explicit GPU-drain wait | 0.025 ms |

These are many small gaps, not one long wait. The largest named transition is
stochastic draft sampling into the next embedding gather: 0.926 seconds across
992 transitions.

Step 5 was then traced with the same MLX 0.32.2 profiler. Its temperature-1
path used 362 compiled windows, compared with 328 in the Step 4 trace. The
comparison must therefore use cost per compiled window.

| Decode profile per compiled window | Step 4 | Step 5 | Change |
|---|---:|---:|---:|
| GPU timeline | 46.24 ms | 45.22 ms | **-2.19%** |
| GPU busy | 37.98 ms | 37.32 ms | **-1.74%** |
| GPU idle | 8.26 ms | 7.90 ms | **-4.29%** |
| Host-late submission | 6.62 ms | 6.27 ms | **-5.22%** |
| Command buffers | 172.2 | 160.9 | **-6.60%** |

Raw GPU use also rose from 82.14% to 82.53%. The largest draft-sampling to
next-gather gap fell from 0.934 to 0.810 ms per transition. The fixed-M4
combine tail therefore reduced both GPU work and host-late starvation. The
remaining sampling transition is still the largest named idle family.

Step 6 removes 0.579 seconds of mean draft-head time in the production runs.
It has not yet received a new MLX timeline trace, so no Step 6 GPU-idle claim
is recorded. The next trace must measure the retained Step 6 stack and compare
per-window GPU work, host-late submission, and the sampling-to-gather gap.

Step 7 directly shortens the CPU work in that sampling-to-gather transition by
sorting 20 proposal rows instead of 80. Its production mean is confirmed, but
it does not yet have a new MLX timeline trace. A later trace must measure the
remaining per-transition gap before recording a new GPU-use percentage.

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
MTPLX_FRSPEC_DRAFT=1 \
MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k \
MTPLX_QWEN4_COMPILED_MTP_PREPARE=1 \
MTPLX_QWEN4_RELAXED_DRAFT_TIES=1 \
mtplx serve \
  --model ~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --model-id mtplx-flash-next-optimized-speed
```

The model defaults select Turbo, native MTP depth 3, batched verification,
`xhigh` reasoning, the temperature-1 sampler, and the bounded 1 GiB hot-row
cache. The two environment variables enable the Step 6 ranked draft head. The
built-in list has a fixed size of 65,536 rows. The two Qwen4 variables enable
the Step 7 fixed-shape preparation graph and the relaxed cutoff-tie path.

## Review rule

Restack one optimization at a time. Check construction-time invariants and
focused parity before the production benchmark. Keep and commit only matched
wins. Record each retained win in the hill-climb table and in the pull request.
