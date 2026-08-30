# Qwen3.8 mlx-serve Candidate Design

## Scope

Build four independent Qwen3.8 Flash-Next candidates on the Step 8 control:

1. Fuse the fixed-M4 GatedDeltaNet verification chain.
2. Fold a hyper-connection write into the following fused read.
3. Expose the existing masked QSA attention kernel as an isolated Step 8 candidate.
4. Make the n-gram `pread` worker count construction-time configurable for a 16/32/48 sweep.

Each candidate gets its own commit and construction-time route. A combined stack is built only after each candidate beats the unchanged control on the exact 1024-output, 16K-prefill production workload.

## Non-goals

- Do not change sampling, MTP acceptance, tie handling, or output semantics.
- Do not add hot-path eligibility checks, fallback branches, counters, or environment reads.
- Do not port mlx-serve kernels by topology alone.
- Do not optimize vanity, concurrent-request throughput, ANE prefill, or other model families.

## Contracts

### Fixed-M4 GDN verification

Preserve Qwen3.8's sequential four-row recurrence, fp32 state ownership, convolution history, rounding boundaries, capture-commit rows, and accepted-prefix commit behavior. Bind the M4 callable only after model geometry and dtype validation. The enabled lane executes directly.

### Deferred hyper-connection write

Represent the pending write as the exact `(stream, block_output, injection)` inputs already owned by the layer. The next fused read consumes them once and emits the same stream, mixed input, and injection values as the current write-then-read sequence. Flush explicitly at boundaries where no following read exists.

### Masked QSA attention

Reuse the existing `qsa_flash_skip` implementation. Bind it only for the measured Qwen3.8 head-dim-256, fixed-M4, long-context route. Do not change the selected-block index, mask, RoPE, K/V ownership, or attention scale.

### N-gram `pread` workers

Resolve the worker count when the sidecar gather is constructed. Keep the 1 GiB hot-row LRU unchanged. Worker selection must not execute inside a gather. Benchmark 16, 32, and 48 workers after clearing the LRU between runs.

## Verification

For each candidate:

1. Run a focused parity test against its unchanged operation chain at the exact production geometry.
2. Spot-check generated output for finite values, valid text, zero repair, and zero verifier fallback.
3. Run the guarded 16K/1K, temperature-1, xhigh benchmark three times against an interleaved unchanged control.
4. Keep the candidate only if mean decode TPS improves without a wall-time, memory, or correctness regression.
5. Record confirmed wins on PR #391; do not publish failed experiments as wins.

## Failure-mode check

- **Critical: recurrence or rounding drift changes MTP acceptance.** Require exact state/row comparison where possible and output-distribution parity before benchmarking.
- **Critical: a microbenchmark win adds graph construction or CPU launch cost in generation.** Measure only installed, prebound callables on the full production workload.
- **Critical: QSA topology mismatch corrupts sparse attention.** Reuse the existing implementation and validate the exact selected indices and outputs before enabling its fixed route.
- **Minor: more `pread` workers increase cold-start CPU or SSD contention.** Keep it isolated and choose the winner from guarded full-workload measurements.

## Rollout

No candidate becomes a default merely because it compiles or wins a microbenchmark. Promote only a verified full-workload win, one commit at a time, then run the full requested benchmark matrix on the final stack.
