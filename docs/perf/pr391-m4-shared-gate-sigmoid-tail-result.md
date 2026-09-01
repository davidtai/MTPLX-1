# PR391 fixed-M4 shared-gate sigmoid-tail result

## Hypothesis and target adaptation

The saved exact-workload census shows a four-element shared-expert sigmoid in
the command buffer immediately before the active M4 combine tail. This
candidate passes the raw q8/group-64 scalar-gate output into that tail and
reproduces MLX's BF16 sigmoid expression there, removing one dispatch and its
four-element factor materialization per target layer and verifier cycle.

This is adapted to the Qwen3.8 physical-M4 path rather than transplanted by
topology:

- the scalar gate remains the stock q8/group-64 projection;
- the routed q4/group-32 and shared q8/group-64 down projections are unchanged;
- the ten-expert BF16 multiply and reduction tree is unchanged;
- MLX's pinned BF16 sigmoid arithmetic and narrowing points are copied exactly;
- the existing 40-threadgroup combine geometry is retained, with one sigmoid
  computed into threadgroup memory per group instead of once per output;
- only the construction-installed M4 class consumes the raw gate. Non-M4 rows
  retain the original parent route, with no eligibility check or fallback in
  the enabled M4 hot path.

The first draft naively evaluated sigmoid in every output thread: 10,240
exponentials per layer for four gate values. Review caught that before model
loading. The benchmarked implementation uses 40 exponentials and 40 barriers
per layer while preserving combine occupancy. A four-threadgroup row-owned
version would reduce this to four exponentials but would also cut combine
parallelism tenfold and primarily test another scheduling change.

Branch implementation commit: `7054adc4c12cec8b8e5353d2bf2de7f8b8efd0ac`.

## Correctness gates

- The new broadcast regression test failed against the per-output draft before
  the threadgroup adaptation was implemented.
- Focused guarded tests: `38 passed` across
  `tests/test_qwen4_m4_stage3.py` and `tests/test_profiles.py`.
- The exact artifact compiled the kernel and installed all 48 layers only after
  the raw-gate tail was array-equal to the existing MLX-sigmoid tail at every
  layer. The construction receipt reported `max_abs_diff: 0.0`.
- The full candidate and flag-off control produced the required seed-20260829
  digest
  `e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc`
  with identical acceptance, draft, and compiled-verifier counts.

The 16-token construction receipt is:

- `.benchmark-artifacts/pr391/rebench3-2803-m4-shared-gate-sigmoid-tail-smoke-fixed-16k-output16.json`

## Same-commit seed-20260829 kill gate

The user-requested policy disabled the old 40 C wait. Both arms used the same
source commit; only `MTPLX_QWEN4_M4_SHARED_GATE_SIGMOID_TAIL` changed.

| Metric | Raw-gate candidate | Flag-off control | Candidate delta |
| --- | ---: | ---: | ---: |
| Decode seconds | 15.701681 | 15.742497 | -0.040816 |
| Decode tok/s | 65.215948 | 65.046858 | +0.169089 |
| Wall seconds | 29.105578 | 28.988609 | +0.116969 |
| Prefill tok/s | 1224.370324 | 1238.961033 | -14.590709 |
| Peak memory bytes | 87,393,881,048 | 87,393,881,048 | 0 |

Artifacts:

- `.benchmark-artifacts/pr391/rebench3-2804-m4-shared-gate-sigmoid-tail-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-2805-m4-shared-gate-sigmoid-tail-control-seeds-16k-1k.json`

## Decision

Park. The candidate is exact, but its 40.82-millisecond decode lead is only
0.26 percent and is not material relative to the 3.2-second gap from the
12.8-second/80-tok/s target. It also lost wall time because of unrelated
prefill variation. Do not promote it, combine it with other candidates, or
spend more seeds on the four-threadgroup variant.

The next MoE hypothesis should materially reduce work: add an exact
weighted-reduction epilogue to the stock-derived q4 gathered-down kernel so the
`[4, 10, 2560]` intermediate is never materialized or reread. That candidate
must preserve the target q4/group-32 dot accumulation, BF16 weighting, and the
current ten-slot reduction tree by construction.
