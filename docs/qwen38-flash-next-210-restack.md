# Qwen3.8 Flash-Next optimization on MTPLX 2.10

This branch establishes the production baseline for a clean optimization stack
on MTPLX 2.10. It uses the official
`Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` model. At this revision,
the branch contains the audit and benchmark record. It does not contain a
retained runtime optimization.

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

## Current production boundary

Decode throughput excludes prefill. All runs below generated 1,024 tokens with
the same output digest, the same 578 / 1,282 MTP acceptance path, 432 verifier
calls, and zero repair time.

| Result | Decode throughput | End-to-end time | Notes |
|---|---:|---:|---|
| Current control 1 | 52.73 tok/s | 34.21 s | Unchanged MTPLX 2.10 |
| Current control 2 | 53.86 tok/s | 33.36 s | Unchanged MTPLX 2.10 |
| Current control mean | **53.29 tok/s** | **33.78 s** | Standard production baseline |

An earlier unchanged-control bracket measured 51.03 and 51.55 tok/s, with a
51.29 tok/s mean. The current controls set `MTPLX_COMPILED_GDN=1` explicitly,
but the MTPLX 2.10 `qwen4_exp` construction path already sets this value. The
53.29 tok/s mean is therefore the current standard baseline, not a compiled-GDN
optimization.

## Accepted optimization hill climb

This table records only changes that remain in the stack. Each new row must use
the production configuration above and must show a matched, repeatable win
against the unchanged prior step.

| Step | Commit | Retained stack | Production evidence | Result |
|---:|---|---|---:|---|
| 0 | `4ce96908` | Unchanged MTPLX 2.10 | **53.29 tok/s mean** | Production baseline |

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
