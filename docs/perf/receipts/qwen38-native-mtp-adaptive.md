# Qwen3.8 native-MTP adaptive-depth receipt

This receipt covers the native-MTP-only port of the adaptive depth work from
[PR #335](https://github.com/youssofal/MTPLX/pull/335). It does not route through
the external DFlash2 drafter.

## Candidates

- `control`: the Optimized-Speed model's stock BF16 native-MTP blocks at fixed
  K=3.
- `adaptive`: the same BF16 native-MTP blocks with
  `--adaptive-policy position_ema` (`r11_position_ema`).
- `q4 adaptive`: the same policy with the retained experimental Q4 MTP block
  (`r11_position_ema+r28_q4_mtp_block`). This artifact is benchmark-only and is
  not installed or selected by this PR.

The benchmarked runtime and harness commit is
`e29eff720ebaa1f21451d5e5c61736ab7cb34a49`. MLX and Metal were both 0.32.2.
Every paired matrix used a fresh process per arm under the parent GPU lock, in
the symmetric order `control-adaptive-q4-q4-adaptive-control`. The 128K xhigh
row is one pass per candidate, by explicit benchmark-plan decision.

The PR335-derived Python workload uses the same prompt for conditioning and the
timed arm, 1,024 conditioning output tokens, temperature 1.0, top-p 0.95,
top-k 20, and seed 42. The first matrix times exactly 1,024 output tokens. The
xhigh matrix times exactly 16,384 output tokens. All receipt invariant lists
were empty and paired arms were token-deterministic within each candidate.

Positive wall numbers mean faster than fixed K=3. Prefill and decode columns
are throughput deltas, so positive means faster. Memory is peak-memory delta,
so negative means less memory.

## Exact 1,024-output matrix

| Context | Candidate | Wall (s) | Wall | Prefill | Decode | Peak memory |
|---:|---|---:|---:|---:|---:|---:|
| 1K | adaptive | 17.603 | -3.86% | -14.84% | -2.93% | +0.00% |
| 1K | q4 adaptive | 17.681 | -4.28% | -10.00% | -3.87% | +1.04% |
| 16K | adaptive | 40.167 | -1.71% | -0.04% | -3.44% | +0.00% |
| 16K | q4 adaptive | 38.624 | +2.22% | -0.01% | +4.80% | +0.78% |
| 64K | adaptive | 123.737 | -0.07% | -0.06% | -0.15% | -0.00% |
| 64K | q4 adaptive | 121.348 | +1.90% | -0.07% | +10.65% | +0.78% |
| 128K | adaptive | 277.224 | -0.38% | -0.06% | -2.83% | -0.00% |
| 128K | q4 adaptive | 272.285 | +1.42% | -0.02% | +15.01% | +6.02% |

Fixed-K3 control wall times were 16.923 s, 39.481 s, 123.654 s, and
276.162 s at 1K, 16K, 64K, and 128K respectively.

## Xhigh 16,384-output matrix

| Context | Candidate | Wall (s) | Wall | Prefill | Decode | Peak memory |
|---:|---|---:|---:|---:|---:|---:|
| 1K | adaptive | 331.590 | -0.23% | -0.86% | -0.23% | -0.09% |
| 1K | q4 adaptive | 320.817 | +3.12% | -2.63% | +3.15% | +0.97% |
| 16K | adaptive | 387.691 | -0.77% | +0.83% | -0.86% | -0.00% |
| 16K | q4 adaptive | 364.597 | +5.52% | +2.08% | +5.72% | +0.78% |
| 64K | adaptive | 576.230 | -2.74% | +1.40% | -3.59% | -2.54% |
| 64K | q4 adaptive | 549.620 | +1.97% | +1.41% | +2.11% | -1.76% |
| 128K | adaptive | 822.537 | +3.60% | +0.15% | +5.05% | -0.09% |
| 128K | q4 adaptive | 809.628 | +5.25% | -0.02% | +7.56% | +0.42% |

Fixed-K3 control wall times were 330.841 s, 384.721 s, 560.455 s, and
852.112 s. The Q4 adaptive candidate won all four xhigh wall-time rows. The
BF16 adaptive candidate won only the single-pass 128K row and regressed the
three paired rows, so this receipt does not support making BF16 adaptive the
default.

## 128K adaptive-depth distribution

These are shares of speculative decode cycles grouped by accepted depth, not
shares of wall time or the policy's requested depth.

| Candidate | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| control | 3.17% | 0.02% | 0.00% | 96.81% |
| adaptive | 2.53% | 1.59% | 16.57% | 79.31% |
| q4 adaptive | 2.90% | 2.93% | 15.92% | 78.25% |

## DFlash2 comparison

One clean, isolated DFlash2 adaptive run used the same 1K prompt, 1,024-token
conditioning output, xhigh template, sampler, seed, and exact 16,384-token
timed output. It ran on MLX/Metal 0.32.2 and completed in 642.484 s at
25.553 decode tok/s. That is 94.20% slower by wall time than fixed-K3 native
MTP and 100.26% slower than Q4 adaptive native MTP; decode throughput was
48.62% and 50.19% lower respectively.

An earlier MLX 0.32.0 run and an MLX 0.32.2 run performed while the launchd
server was resident are excluded from every comparison.

## Raw-receipt identities

The full receipts contain generated output and per-cycle data, so their
content-addressed identities are recorded here rather than copying megabytes
of generation text into the repository.

| Workload | SHA-256 |
|---|---|
| 1K / 1,024 output | `cf66b709b80e1cf61d4e69546b2b063c33572a3ae22f57e59db477d074a09e58` |
| 16K / 1,024 output | `a2add44fe0888608dc9b171cdb690456bc99ba14dbd9c3d57a7255f798d2b146` |
| 64K / 1,024 output | `9981de6997995457c08410271f20f38a41175ab3b10771653606cb8bd3c8ea90` |
| 128K / 1,024 output | `6c2d15a8826187b5434d492d33d1f20549cfdf989e18d8858bcc555c57dad81b` |
| 1K / xhigh / 16,384 output | `7fe11714134f393f976849917b42912460a69505b47fb8428b7b2005681f38e1` |
| 16K / xhigh / 16,384 output | `2a6e0a31dffd4b4aeafb0fdd4531e3dbe541135a1cd2fa081b0de69c63bfc53c` |
| 64K / xhigh / 16,384 output | `a2db62235e58f77306f9a9b03f9a33d0e43af3d50c9e771bc94a07c7e3b0adde` |
| 128K / xhigh / 16,384 output, single pass | `6ef5b4e8486b0de2cf6c5b667b29fc852b32991067e52fd1ef352006816e8e90` |
| DFlash2 1K / xhigh / 16,384 output | `55e106e4d62358935cf02ce83c351d266bb8da53620bbb68b18b8ca2821e9025` |
