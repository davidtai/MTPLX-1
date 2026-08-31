# PR391 recent Qwen3.8 upstream candidate deduplication

Window: 2026-08-17 through 2026-08-31. Target: Qwen3.8 Flash-Next
125B-A6B, exact 16K prompt/1024 output, physical M4/native D3. The acceptance
target remains 80 decode tok/s (12.8 seconds); arithmetic-changing results are
not performance candidates.

Sources surveyed independently: MLX/mlx-lm, MLXServe, and oMLX. This table
merges candidates by the boundary they actually change, rather than by PR name.

## Ranked queue, easiest to hardest

| rank | deduplicated boundary | upstream source | credible 16K/M4 gain | integration difficulty | evidence and disposition |
| ---: | --- | --- | ---: | --- | --- |
| 1 | QSA incomplete-block/padding semantics | mlx-lm #1788 | 0 | test-only | PR391 B1 already intersects selected complete blocks plus the visible tail with per-row causality. Retain focused parity tests; no transplant. |
| 2 | small-M wide quantized matmul routing | MLX #4171 | already in baseline | easy audit | Closed: installed MLX 0.32.2 contains the merged BM32 route, and the exact q8/group-64 target `lm_head` executes at M4, K2560, N248320. No integration or candidate A/B remains. |
| 3 | fixed-M4 QSA score layout | oMLX #3244 | none measured | easy | Flattened exact FP32 score GEMM was tested at 64.3626 tok/s versus 65.1364 same-commit control. Parked. |
| 4 | QSA prefill score/gather | MLXServe QSA adaptation; mlx-lm #1454 | no decode gain | easy/medium | 16K candidate changed the digest and reduced prefill from a recent 1232.9 to 1085.4 tok/s. Rejected; separate prefill lane only. |
| 5 | fixed-M4 GDN norm/gate epilogue | MLXServe GDN split | unproven under thermal drift | medium | Exact A/B/A was 63.67, 61.66, then 51.33 tok/s with the 40 C wait disabled. No repeatable gain; parked. |
| 6 | fixed-M4 GDN g/beta preparation | MLXServe GDN split | invalid apparent +2.4% | medium | Candidate changed the production digest. Rejected regardless of apparent speed. |
| 7 | fixed-M4 QSA BM8 native score/mask | oMLX #3320 | low single-digit ceiling | medium/high | Shape-compatible score primitive, but oMLX's direct verify route starts at 65K and evidence is around 150K/D5. Requires native ABI work and replaces only an already-accelerated matmul. Skipped at 16K. |
| 8 | direct PLE `pread` row delivery | oMLX #3287; MLXServe worker sweep | negative | medium | PR391 already tested exact synchronous all-miss and early-window direct payload delivery: 64.35 and 64.62 versus 64.88 control. Worker counts 16/32/48 also lost. Deduplicated and rejected. |
| 9 | exact native QSA top-k | oMLX #3244 | <2% ceiling | high | Do not transplant: oMLX cutoff ties prefer higher ids, while PR391 must preserve adjusted-score plus stable eager ordering. A bespoke selector is unjustified without a score/top-k profile. |
| 10 | remaining stock GDN recurrence/conv geometry | mlx-lm #1559/#1741; MLX #4409 | likely 0 | high | PR391 already installs fixed-M4 GDN kernels. Audit for remaining stock recurrence time before adapting another packed kernel. |
| 11 | compile trace-node retention | MLX #4440 | 0 tok/s | dependency/core | Soak-memory fix only. Revisit only after reproducing multi-output trace growth. |

## Do not integrate into the canonical exact lane

- oMLX direct sparse GQA: its tests allow reduction drift up to `5e-3`.
- oMLX native top-k unchanged: wrong tie ownership.
- oMLX direct verify QSA at 16K: its own crossover gate is 65K.
- oMLX resident PLE: the approximately 32 GiB table violates the 128 GiB
  memory plan.
- oMLX ANE, HOBBIT, or lossy expert streaming: only cosine/perplexity parity.
- MLX GQA12/16 decode kernels #4077/#4380/#4431: Qwen3.8 QSA uses D256 and is
  outside their D64/D128 contract.
- MLX #4352 inactive-SIMD MoE change: decode unchanged and a draft revert cites
  reliability concerns.
- MLX #4048 completion-handler scheduling: recent 320B evidence was effectively
  flat (34.78 versus 34.76 tok/s).
- MTP, n-gram drafting, batching, tensor/pipeline-parallel, and cache-ABI PRs:
  they change a different workload or replace PR391's exact D3 state contract.

## MLX #4171 engagement audit

The worktree has MLX 0.32.2 installed. Its release tag is 20 commits ahead of
the #4171 merge, so the `M <= 32 ? BM32 : BM64` quantized-matmul dispatch is
already active. The exact artifact declares the shared target `lm_head` as
8-bit affine/group-64 with hidden size 2560 and vocabulary size 248320.
`_forward_fixed_m4_suffix` sends all four verifier rows through that head.
Consequently its production geometry is M4/K2560/N248320 and directly selects
the new BM32 kernel. This upstream gain is already included in every recent
control and cannot be integrated a second time.

## Net assessment

The recent upstream work supplies correctness oracles and long-context/prefill
ideas, but no exact candidate has a credible path from approximately 65 to 80
tok/s on 16K/M4. Scheduling-only changes have repeatedly moved the same work
between host-late and GPU-late intervals. Continue only with boundaries that a
real-shape trace shows consume material time; do not abandon starvation as a
diagnostic, but stop treating generic queue reshuffling as its remedy.
