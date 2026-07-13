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
- Create `benchmarks/results/hy3-record-native-stageb-premise-20260713.{md,json}` for the evidence-first Stage B decision.
- Create `scripts/benchmark_hy3_record_destinations.py`, its contract tests, and `benchmarks/results/hy3-record-destination-layout-issue30-20260713.{md,json}` for the independent loader gate.
- Extend streamed model tests for cache, pin, repeated-expert, and fallback behavior.

### Task 1: Lock the v1 record and combine contracts

- [x] Write failing tests for exact component offsets, shapes, dtypes, total 10,616,832-byte coverage, Q4/group-64 rejection, top-8 shape, activation-dtype scores, and deterministic fallback.
- [x] Verify RED: `pytest -q tests/test_moe_record_q4.py -k contract` fails because the module does not exist.
- [x] Implement immutable `RecordQ4Layout` and `validate_hy3_record_q4(record, spec)`; no Metal code yet.
- [x] Verify GREEN and commit `test(hy3): lock record-native Q4 contract`.

### Task 2: Bound Stage B headroom before writing a kernel

**Premise update:** The prior kernel spike measured stock `mx.gather_qmm` at 95.6% of its single-dispatch memory roofline, the full custom Q4 arm as a statistical tie, and pure reassembly-tail removal at 1.4862 ms/token. The resource-safe #29 layer control averages 2.9545 tok/s, or 338.472 ms/token, so a 5% throughput win requires saving 16.118 ms/token.

- [x] Recompute the +5% time budget from the resource-safe #29 control.
- [x] Bound the entire measured down projection at 5.6485 ms/token and the entire measured tail at 1.4862 ms/token.
- [x] Record the deliberately impossible free-down-plus-free-tail ceiling: 7.1347 ms/token, only 44.27% of the requirement.
- [x] Check in the exact evidence chain and go/no-go payload before adding kernel code.
- [x] Defer Stage B until an independent loader/scheduling result supplies at least 8.983 ms/token of conservative headroom; the evidence-based realistic gap is about 13.2 ms/token.

If Stage B is reopened, its first slice remains a component-bank weighted down reduction with raw-bit BF16/FP16 parity at B=1/2/4/8, permuted and repeated assignments, and a device-visible assignment/slot/generation guard. Negative, `>=192`, and max-`uint32` mutations must trip the guard. Routing data must never become an unchecked output address. Use precomputed identical Stage-A inputs, warm JIT, fallback disabled, at least 150 paired/interleaved/rotated iterations, and exactly one real `mx.eval` per arm.

### Task 3: Measure loader destination layout independently

- [x] Write tests for one fixed-stride record allocation, exact byte ownership, close behavior, and no duplicate component-bank allocation.
- [ ] Add a failure-path regression proving any outstanding shared-arena row write blocks Metal reads from every sibling row, then releases aggregate ownership exactly once.
- [x] Preserve the resource-level safety rule from `058b40b`: the isolated harness materializes all MLX allocations before external writes and issues no Metal reads; no runtime arena is introduced.
- [x] Compare a contiguous v1 record destination with the current one-`preadv` scatter into nine component rows. Describe the mechanism as destination/layout efficiency, not syscall elimination.
- [x] Hold sidecar offsets, record hashes, bytes, `F_NOCACHE`, and record sequence constant; measure host read concurrency 1 and 32 plus adjacent-record batches.
- [x] Run four balanced AB/BA repeats and report throughput plus p50/p95 record latency.
- [x] Kill the arena: host concurrency 32 is inconclusive, adjacent p95 regresses, and the all-miss mean extrapolation cannot close the Stage B headroom gate.

The sibling-row failure regression remains intentionally unchecked because the gate rejected the arena before runtime integration. The benchmark explicitly records `metal_reads_issued=false` and `claims_runtime_fence_safety=false`; it is not evidence for an unimplemented runtime fence.

### Task 4: Add Stage A only if combined headroom is plausible

- [x] Sum the Stage-B and loader bounds and stop. Even the mean 632-miss loader extrapolation plus the impossible Stage-B ceiling implies only +4.9097%; the conservative loader estimate is lower.
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
- [x] If it loses, leave only useful unwired test/benchmark scaffolding and document the negative result.
- [ ] Push `experiment/hy3-record-native-exec` and open a draft PR against `experiment/hy3-cache-scheduling`, linking #30.
