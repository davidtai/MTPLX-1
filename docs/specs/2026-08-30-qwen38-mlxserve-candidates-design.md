# Qwen3.8 mlx-serve Candidate Design

## Scope

Build four independent Qwen3.8 Flash-Next candidates on the Step 8 control:

1. Fuse the fixed-M4 GatedDeltaNet verification chain.
2. Fold a hyper-connection write into the following fused read.
3. Expose the existing masked QSA attention kernel as an isolated Step 8 candidate.
4. Make the n-gram `pread` worker count construction-time configurable for a 16/32/48 sweep.

Each candidate gets its own commit and construction-time route. A combined stack is built only after each candidate beats the unchanged control on the exact 1024-output, 16K-prefill comparison workload. That fixed output is an A/B control, not a claim that production responses have a canonical length; production output remains variable and stop-driven.

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
3. Run the guarded 16K/1K, temperature-1, xhigh comparison cell three times against an interleaved unchanged control, then separately prove variable-length stopping and capacity growth.
4. Keep the candidate only if mean wall time improves repeatably without a memory or correctness regression. Report decode TPS as a secondary metric.
5. Record confirmed wins on PR #391; do not publish failed experiments as wins.

## Failure-mode check

- **Critical: recurrence or rounding drift changes MTP acceptance.** Require exact state/row comparison where possible and output-distribution parity before benchmarking.
- **Critical: a microbenchmark win adds graph construction or CPU launch cost in generation.** Measure only installed, prebound callables on the full production workload.
- **Critical: QSA topology mismatch corrupts sparse attention.** Reuse the existing implementation and validate the exact selected indices and outputs before enabling its fixed route.
- **Minor: more `pread` workers increase cold-start CPU or SSD contention.** Keep it isolated and choose the winner from guarded full-workload measurements.

## Rollout

No candidate becomes a default merely because it compiles or wins a microbenchmark. Promote only a verified full-workload win, one commit at a time, then run the full requested benchmark matrix on the final stack.

## Audit outcome

The comparison used official `ddalcu/mlx-serve` source at
`cd93a2b00253218dff96fdb42d457bfb190b12de`. Each applicable technique was
adapted to Qwen3.8 arithmetic in an isolated worktree; no mlx-serve candidate
was stacked into the production branch because none beat the unchanged route
on the guarded 16K/1K comparison cell.

| Candidate | Correctness result | Production result | Decision |
| --- | --- | --- | --- |
| Fixed-M4 GDN fusion | Exact target arithmetic and focused tests passed | 28.8403 s mean wall, 69.3701 decode tok/s versus 28.3865 s, 72.9196 tok/s control | Reject |
| QSA RoPE reuse | Focused parity passed | 29.262 s mean wall, 72.646 tok/s versus 28.114 s, 76.464 tok/s control | Reject |
| Direct QSA attention | Focused parity passed | Best candidate 27.328 s wall versus 26.767 s warmed control | Reject |
| Hyper-connection routes | Focused parity passed | 31.250 s row-shared and 28.491 s stock-projection-tail versus 28.3865 s control | Reject |
| N-gram per-region `pread` | Focused suite passed | 16/32/48 workers: 29.3662/29.3420/29.4158 s mean wall; unchanged control: 29.1134 s | Reject |

The existing MTPLX routes already cover the useful topology from the compared
mlx-serve kernels, while preserving Qwen3.8's different rounding, cache, QSA,
and fixed-M4 ownership contracts. The retained change is therefore the
variable-length fixed-M4 verifier correction itself, not an unproven kernel
transplant.

## Variable-length verifier evidence

The rebased production source was exercised under the exclusive GPU guard with
a 16K prompt. An early-stop request configured for 2048 tokens stopped at 62
tokens and exactly matched the corresponding baseline prefix. A real
generation-final SessionBank snapshot restored 16,408 cached tokens, prefetched
only the seven-token suffix, and then generated the requested 97 tokens. Both
requests stayed on compiled M4 with zero fallback, demotion, growth demotion, or
repair. The receipt is
`.benchmark-artifacts/pr391/pr391-variable-length-production-proof.json`.
