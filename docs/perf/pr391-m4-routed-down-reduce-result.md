# PR391 fixed-M4 routed-down reduction result

## Hypothesis and exact target adaptation

The saved exact-workload census contains 18,336 routed-down operations: 48
target layers x 382 physical-M4 verifier passes. Stock MLX materializes the
BF16 routed-down result as `[4, 10, 2560]`; the following combine tail rereads
it, multiplies by normalized route scores, and applies the exact ten-slot BF16
reduction tree.

This candidate emits the already weighted and reduced BF16 `[4, 2560]` result
directly. It is derived from the actual Qwen3.8 dispatch rather than ported from
an adjacent MoE kernel:

- MLX sees routed hidden state `[1, 4, 10, 1, 640]` as `B=40, M=1`, not M10;
- the exact down kernel is generic
  `affine_gather_qmv_bfloat16_t_gs_32_b_4`, not QMM or fast QMV;
- its 64-thread threadgroup owns eight output columns and performs two full
  256-element q4/group-32 blocks plus the safe 128-element K tail;
- q4 unpacking, affine bias contribution, FP32 accumulation, `simd_sum`, and
  the single BF16 narrowing of every expert dot match MLX 0.32.2;
- normalized BF16 route scores remain outside the custom kernel, and each dot
  is BF16-weighted before reproducing `0+8`, `1+9`, then `2..7`, with the same
  BF16 boundaries;
- the shared q8/group-64 projection and sigmoid remain stock and join through a
  separate exact tail. The expensive routed kernel therefore gains no new
  dependency on the independent shared branch;
- only the construction-installed physical-M4 class calls the candidate.
  Non-M4 rows retain the parent path without an enabled-path eligibility check
  or fallback.

The first construction attempt exposed a Metal declaration error before any
benchmark: program-scope constants needed the `constant` address space. A
failing regression test was added, the declarations were corrected, and the
real-artifact gate was rerun from a new commit.

Implementation commits:

- `ca4441ac3bd49083bc2fa3e6c78794dad6c68e50`
- `b2993f67f73e3430e19e890240298e990f391bc7`

## Correctness gates

- Focused guarded tests: `39 passed` across
  `tests/test_qwen4_m4_stage3.py` and `tests/test_profiles.py`.
- Independent source review confirmed weight/metadata strides, safe-tail
  bounds, arithmetic ordering, 2,560-column coverage, and the separate shared
  dependency.
- The exact artifact compiled the custom kernel and installed all 48 layers
  only after the candidate was array-equal to the stock gathered-QMV plus
  combine path at every layer. The receipt reported `max_abs_diff: 0.0`.
- Candidate and control matched the required response digest, acceptance
  counts, draft counts, and compiled-verifier counts within both seeds.

The passing 16-token construction receipt is:

- `.benchmark-artifacts/pr391/rebench3-2807-m4-routed-down-reduce-smoke-fixed-16k-output16.json`

The rejected pre-benchmark compile attempt was sequence 2806 and produced no
performance result.

## Same-commit order-reversed validation

The user-requested policy disabled the old 40 C wait. Every arm used source
commit `b2993f67f73e3430e19e890240298e990f391bc7`; only
`MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE` changed. Seed 20260829 ran candidate then
control, while seed 20260830 reversed the order.

| Seed/order | Candidate decode s | Control decode s | Seconds saved | Candidate tok/s | Control tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260829, C then A | 15.553158 | 15.722822 | 0.169664 | 65.838721 | 65.128257 |
| 20260830, A then C | 15.788076 | 15.926112 | 0.138037 | 64.859075 | 64.296921 |
| Mean | 15.670617 | 15.824467 | 0.153851 | 65.348898 | 64.712589 |

The mean decode-time reduction is 0.97 percent. Candidate peak memory remained
87,393,881,048 bytes; the controls were 87,393,881,048 and 87,393,848,280
bytes.

Artifacts:

- `.benchmark-artifacts/pr391/rebench3-2808-m4-routed-down-reduce-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-2809-m4-routed-down-reduce-control-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-2810-m4-routed-down-reduce-control-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-2811-m4-routed-down-reduce-seeds-16k-1k.json`

## Decision

Retain as an isolated positive candidate on
`experiments/pr391-m4-routed-down-reduce`. Both orderings won with exact
outputs, so this is not another scheduling-only transplant. Do not fold it
into the research branch or combine it with other candidates until the
remaining hypothesis queue is exhausted and an unchanged-control combination
gate is planned.

The mean candidate is still 2.87 seconds above the 12.8-second/80-tok/s target
and about 22 percent short in throughput. This candidate is useful evidence,
not completion of the performance goal.
