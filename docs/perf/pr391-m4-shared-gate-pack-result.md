# PR391 fixed-M4 shared-gate pack result

## Hypothesis

The saved dispatch census contains one eight-operation MoE frontier for every
target verifier layer and cycle: 18,336 occurrences = 48 layers x 382 cycles.
Its whole-command-buffer ceiling is 1.887227457 seconds. One operation is the
q8/group-64 scalar `shared_expert_gate` projection, independent of the existing
1,280-row packed shared gate/up projection.

This candidate construction-packs that one row with the shared gate/up rows,
forming a 1,281-row q8 projection. The physical-M4 hot path consumes the gate
and gate/up outputs directly. Other widths retain the original 1,280-row
projection through the contiguous tail view. Both routed and shared down
projections and the exact M4 combine reduction remain unchanged.

Branch implementation commit: `e3d237c78d2e4bda5637773dfedec2dd2a9b0ca6`.

## Correctness gates

- Focused tests: `40 passed` across `tests/test_qwen4_m4_stage3.py` and
  `tests/test_profiles.py`.
- The exact artifact installed all 48 layers only after the packed 1,281-row
  output was array-equal to the separate 1,280-row and one-row projections.
- The installed M4 self-check reported `max_abs_diff: 0.0`.
- The full candidate and control both produced the required seed-20260829
  response digest
  `e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc`
  with identical acceptance and verifier-cycle counts.

The 16-token construction receipt is:

- `.benchmark-artifacts/pr391/rebench3-2800-m4-shared-gate-pack-smoke-fixed-16k-output16.json`

## Same-commit seed-20260829 kill gate

The user-requested benchmark policy disabled the old 40 C wait. Both arms used
the same source commit; only `MTPLX_QWEN4_M4_SHARED_GATE_PACK` changed.

| Metric | Packed candidate | Flag-off control | Candidate delta |
| --- | ---: | ---: | ---: |
| Decode seconds | 15.781021 | 15.744557 | +0.036464 |
| Decode tok/s | 64.888069 | 65.038350 | -0.150281 |
| Wall seconds | 29.087080 | 29.215514 | -0.128434 |
| Prefill tok/s | 1233.323987 | 1218.271398 | +15.052589 |
| Peak memory bytes | 87,394,733,016 | 87,393,881,048 | +851,968 |

Artifacts:

- `.benchmark-artifacts/pr391/rebench3-2801-m4-shared-gate-pack-candidate-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-2802-m4-shared-gate-pack-control-seeds-16k-1k.json`

## Decision

Park. The candidate is exact but lost 36.46 milliseconds, or 0.23 percent of
decode time, in the matched kill gate. The 1,281-row qmm and its strided
gate/up tail cost more than the removed one-row q8 dispatch. Do not combine it
with other candidates or spend three seeds on it.

The next individually attributable MoE candidate is to pass the raw scalar
gate into the existing exact combine-tail kernel and reproduce MLX's BF16
sigmoid there. That removes only the sigmoid dispatch and factor
materialization without changing quantized projection geometry.
