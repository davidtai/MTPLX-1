# Qwen 3.8 Challenge Port Plan

## Goal and fixed boundary

Port every transferable mechanism whose accepted Yukon step improved the
previous accepted score by more than 0.10 percent. All work stays on
`perf/qwen38-challenge-port` and lands in one PR to `main`.

The control and production target are always
`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`; no bare model or sibling Qwen
artifact may substitute for it. Source is pinned to challenge commit
`eb5eadc7a165047d4321ce883b9ff30894d8bd19`; MTPLX starts from
`bd4421567f9e16ce957c6ef97708b072dcd73937`.

The immutable source inventory contains 82 accepted submissions and 54 steps
above the percentage threshold. The row-by-row mapping is in the design spec;
the consolidated, evidence-backed disposition is in
`docs/perf/qwen38-challenge-port-ledger.md`.

## Target-shape adaptation contract

Every candidate must be derived for the installed model rather than copied by
mechanism name:

- BF16 hidden width 5,120, intermediate width 17,408, 64 target layers.
- 24 query heads, 4 K/V heads, head dimension 256, rotary dimension 64,
  theta 10,000,000.
- GDN has 48 value heads, 16 key heads, and head dimension 128.
- The target trunk is affine Q4/group-32 with Q8/group-64 islands; packed
  lanes use MLX unsigned-32 little-endian layout.
- Actual speculative target-verify widths are 2 through 9; the currently
  measured Optimized-Speed policy uses depth 3.
- The MTP sidecar has one full-attention draft layer. Cache changes must
  preserve the exact committed-history K/V state and cache offset.

An implementation that does not match these dimensions, quantization islands,
cache ownership, compile boundary, or actual call phase is rejected before its
performance result is interpreted.

## Candidate plan and disposition

| Mechanism cluster | Plan | Final disposition |
| --- | --- | --- |
| Target top-2/readout reuse | Inspect the MTPLX consumer path before porting. | Skip: MTPLX consumes target argmax and has no top-2 ledger; the port would add work. |
| Compact Q2 clustered proposal head | Build only the final surviving descendant from the installed Turbo Q4 draft head; measure it last against the accumulated winner stack. | Rejected against S1 at 16K: -1.3792%. Code removed. |
| Cross-row affine-4 QMV | Re-derive for group-32 trunk plus group-64 islands and bind only to target verify widths 2..9. | Rejected against S1 at 16K: +0.0618%, below the 0.10% materiality floor. Code removed. |
| Fused target Q/K/V projection | Compare the current packed-concat path with the source mechanism. | Already present: MTPLX already issues the fused affine projection and contiguous splits. |
| GDN projection/satellite fusion | Inspect installed Qwen3-Next and the compiled verify/capture path. | Already present/source-specific: `in_proj_qkvz`, `in_proj_ba`, whole-verify compile, and recurrence replay already own this work. |
| One-forward SDPA width bridge | Check whether the source Swift width wall exists in the actual Python/MLX route. | Skip as source-specific: packed GQA already processes the actual widths in one target forward. |
| Q/K RMSNorm plus partial RoPE | Adapt to 24Q/4KV, head 256, rotary 64 and measure against S1. | Rejected against S1 at 16K: -0.2034%. Code removed. |
| Verify-hidden reuse and attention gate | Inspect the current target-verify boundary and compile ownership. | Already present: post-norm hidden is reused and the complete verify graph is compiled. |
| Boundary residual/RMSNorm fusion | Derive for BF16 width 5,120 across the mixed 64-layer trunk and measure against S1. | Rejected against S1 at 16K: -0.0130%, with higher peak memory. Code removed. |
| Dual pre-FC RMSNorm fusion | Derive for two BF16 width-5,120 rows, eliminate the concatenation output, and measure against S1. | Rejected against corrected S1 at 16K: +0.0505%, below the 0.10% floor. Code removed. |
| K/V-only committed-history append | Specialize to the installed one-layer full-attention MTP sidecar; prove cache tensor equality while forbidding dead Q/O/MLP/final-norm work. | Retain at original request context >=16K: +1.9268%, exact parity, flat memory. |
| Precision-island compact artifact | Verify topology, provenance, and dependency outcome. | Skip: wrong head, no redistribution declaration, and its required compact descendant loses. |
| Depth/warm calibration | Map the source floor/cap to the current measured policy and check existing warmups. | Already/source-specific: source depth 6-7 is not the depth-3 route; MTPLX already warms and has its own cost/EV controller. |

Historical resamples, no-ops, weak <=0.10 percent steps, and mechanisms removed
by later accepted submissions receive no independent port. A below-threshold
row is used only when it is a dependency of a surviving final mechanism.

## Measurement gate for every implemented candidate

Use the guarded exclusive GPU lane. Build exact-token Python prompts from
`mtplx/generation.py` and keep the intact instruction from
`mtplx/benchmarks/prompts/python_modules_long.jsonl` at the tail.

Each route receives one full 1,024-output conditioning generation followed by
exactly four timed ABBA arms at 16,384 input tokens and 1,024 generated tokens.

Both target and draft use temperature 1.0, top-p 0.95, top-k 20, seed 42,
Turbo/Q4 draft construction, depth 3, persistent cache, committed history,
capture/commit verification, and `linear-gdn-from-conv-tape`. Record per-arm
prefill tok/s, decode tok/s, peak memory, generated count, wall time, route
fingerprint, counters, token hash, and acceptance schedule. A short generation
is invalid and must be rerun.

Exact token/schedule parity passes immediately. A deterministic tie or small
numeric shift is not rejected by token hash alone; performance remains the
promotion criterion. The checked-in QMV audit covers cache equality, top-1
stability, top-k overlap, error magnitude relative to decision margin, and
sampling total variation.

## Execution order

- [x] Freeze and reproduce all 82 accepted rows and the 54-row source gate.
- [x] Add the exact immutable Qwen3.8 27B route/identity contract.
- [x] Inspect and close already-present, challenge-only, superseded, and weak
  rows before writing duplicate code.
- [x] Implement, shape-correct, and independently gate compact-head, QMV,
  Q/K-RoPE, boundary-norm, dual-norm, and K/V-history candidates.
- [x] Measure each candidate chronologically against the accumulated winner
  stack at the required 16K window.
- [x] Retain only K/V history and remove compact-head, QMV, Q/K-RoPE,
  boundary-norm, and dual-norm production code.
- [x] Replace the pending final table with cumulative prefill/decode/memory and
  generated-count results.
- [x] Remove exploratory prompt/receipt artifacts, run focused and full tests,
  Ruff, inventory reproduction, and diff checks.
- [x] Review the final diff, commit, push this one branch, and open exactly one
  PR to `main`.
