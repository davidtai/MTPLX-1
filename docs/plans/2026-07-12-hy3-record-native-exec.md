# Hy3 Record-Native Sparse Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and, only if it wins, integrate a two-stage Hy3 affine-Q4 primitive that consumes the existing v1 expert record and reduces weighted top-8 down outputs directly.
**Architecture:** Build on PR 1. Keep cache policy and I/O on the host, preserve the gate/up-to-down dependency as two device stages, and make each kernel arm independently selectable. Reuse only the verified affine-Q4 primitives from the kernel spike; do not inherit its conclusions or wire its monolithic API.
**Tech Stack:** Python, MLX `mx.fast.metal_kernel`, optional native MLX extension after proof, pytest, checked-in micro/end-to-end benchmarks.
**Assumptions:** Assumes Hy3 Q4/group-64 and the manifest v1 nine-component order; this will NOT accept arbitrary quantization. Assumes a prototype can use JIT Metal for proof; promotion will NOT keep per-call JIT construction if its host cost erases the gain.

---

## Files

- Create `mtplx/kernels/moe_record_q4.py`: v1 layout validation and two-stage kernel API.
- Modify `mtplx/kernels/__init__.py`: explicit experimental export.
- Modify `mtplx/models/expert_mlx.py`: opt-in execution arm and weighted reduction boundary.
- Modify `mtplx/models/hy3_mlx.py`: pass scores into a fused-combine-capable switch without changing the fallback.
- Create `scripts/benchmark_hy3_record_q4.py` and `tests/test_moe_record_q4.py`.
- Extend streamed model tests for cache, pin, repeated-expert, and fallback behavior.

### Task 1: Lock the v1 record and combine contracts

- [x] Write failing tests for exact component offsets, shapes, dtypes, total 10,616,832-byte coverage, Q4/group-64 rejection, top-8 shape, activation-dtype scores, and deterministic fallback.
- [x] Verify RED: `pytest -q tests/test_moe_record_q4.py -k contract` fails because the module does not exist.
- [x] Implement immutable `RecordQ4Layout` and `validate_hy3_record_q4(record, spec)`; no Metal code yet.
- [x] Verify GREEN and commit `test(hy3): lock record-native Q4 contract`.

### Task 2: Measure Stage B headroom before building an arena

**Premise update:** The prior kernel spike measured stock `mx.gather_qmm` at 95.6% of its single-dispatch memory roofline, the full custom Q4 arm as a statistical tie, and pure reassembly-tail removal at only about 1.4-1.5 ms/token. The resource-safe #29 layer control averages 2.9545 tok/s, or 338.472 ms/token, so a 5% throughput win now requires saving 16.118 ms/token. The previously measured tail-elision mechanism covers at most about 9.3% of that budget; Stage B must therefore show additional fused-QMV savings rather than being retained for tail removal alone.

- [ ] Write failing raw-bit parity tests for a Stage-B-only weighted reduction over existing component-bank down weights: BF16/FP16, permuted execution rows, adversarial score order, repeated experts, and B=1/2/4/8.
- [ ] Add a device-side bounds/generation violation flag whose negative-index, `>=192`, and max-`uint32` mutation tests fail if the guard is removed. Routing data must never become an unchecked output address.
- [ ] Keep precomputed Stage-A hidden inputs identical between stock and candidate arms.
- [ ] Implement `component_down_weighted_reduce_q4` by selectively extracting the verified affine-Q4 dot-product/build-guard scaffold from the prior spike. Do not cherry-pick its divergent runtime integration.
- [ ] Require zero raw-bit mismatches before printing any timing. Kernel/build failures after dispatch must propagate; they are not fallback eligibility.
- [ ] Benchmark at least 150 paired, interleaved, rotated iterations with exactly one real `mx.eval` per arm and warm JIT state.
- [ ] Kill Stage B if it is slower in two independent paired runs or its upper-confidence savings cannot contribute enough to the freshly computed 5% time budget.
- [ ] Commit the parity harness, raw paired payload, exact commands, and measured go/no-go result before any arena work.

### Task 3: Measure loader destination layout independently

- [ ] Write tests for one fixed-stride record allocation, exact byte ownership, close behavior, and no duplicate component-bank allocation.
- [ ] Add a failure-path regression proving any outstanding shared-arena row write blocks Metal reads from every sibling row, then releases aggregate ownership exactly once.
- [ ] Preserve the resource-level safety rule from `058b40b`: all external writes into a shared arena must finish before Metal reads any sibling row, unless a proved event/fence mechanism replaces the barrier.
- [ ] Compare a contiguous v1 record destination with the current one-`preadv` scatter into nine component rows. Describe the mechanism as destination/layout efficiency, not syscall elimination.
- [ ] Hold sidecar offsets, record hashes, bytes, `F_NOCACHE`, and record sequence constant; measure QD1 and runtime-representative QD32 plus adjacent-record batches.
- [ ] Run three repeats and report throughput plus p50/p95 record latency.
- [ ] Kill the arena if service time has no significant improvement or p95 regresses.

### Task 4: Add Stage A only if combined headroom is plausible

- [ ] Sum conservative Stage-B and loader upper bounds. Stop if they cannot plausibly save the freshly computed 5% end-to-end budget.
- [ ] Write failing parity tests against `mx.gather_qmm` for BF16/FP16, scattered slot indices, repeated experts, and B=1/2/4/8.
- [ ] Implement gate and up as two specialized QMV dispatches plus SwiGLU producing `[assignments,1536]`.
- [ ] Keep one-dispatch gate/up fusion as a separate arm; the prior split-layout experiment regressed 10.5% because MLX already overlaps the independent dispatches.
- [ ] Require raw-bit equality and a fallback-disabled build guard for every retained variant.
- [ ] Measure Stage A independently before combining it with Stage B.

### Task 5: Integrate only a gate-passing execution arm

- [ ] Write failing integration tests proving the arm is off by default, preserves router authority/order, releases pins after errors, rejects unsupported records before allocation, and leaves logits unchanged on the fallback.
- [ ] Define an explicit storage/backend compatibility matrix. `record-native` cannot silently allocate both an arena and component banks.
- [ ] Add `expert_q4_backend="component"|"record-native"` plus the allocator/CLI/runtime seams needed to select exactly one storage representation.
- [ ] Add a Hy3-only `run_weighted(x, indices, scores)` switch; keep `SparseMLP`'s existing multiply/sum path byte-for-byte as the default and keep GLM on its existing switch.
- [ ] Preserve the shared expert as a separate arm. Do not fold shared final-add integration into the first routed result.
- [ ] Preserve exact reduction order across route waves; per-wave BF16 accumulation is not an acceptable substitute when B>1 spans transient waves.
- [ ] Verify stale generation, overwrite fence, table-version fence, pin release, repeated assignments, and B=1/2/4/8 behavior.

### Task 6: End-to-end promotion gate and PR

- [ ] Run short all-hit and controlled-miss end-to-end pairs first. Skip sustained runs if the upper confidence bound is below +5%.
- [ ] If still viable, run two alternating-order natural-stop B1 pairs with identical router IDs, 1,905-token output, and token SHA-256 `484e182a68604821f69d56d0b15488d26723e6123f6a57f8158f8b20a4c6ed1c`.
- [ ] Run B=2/4/8 and mixed prefill/decode secondary lanes with identical bytes/cache budget and no tail-latency regression above 2%.
- [ ] Retain the arm only if every B1 pair and their mean improve by at least 5%, memory is bounded, and every parity/generation guard passes.
- [ ] If it loses, leave only useful unwired test/benchmark scaffolding and document the negative result.
- [ ] Push `experiment/hy3-record-native-exec` and open a draft PR against `experiment/hy3-cache-scheduling`, linking #30.
