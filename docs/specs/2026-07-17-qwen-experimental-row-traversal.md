# Experimental Qwen Row Traversal: Exact-M Verify Tiles

## Status

Experimental and default-off. Based on public upstream `a391973` (`v2.1.0`).
Two independent flags are added, both off by default and mutually exclusive:

- `--experimental-qwen-row-traversal`
- `--experimental-qwen-fast-sigmoid`

Neither changes any default profile.

Row traversal has two cooperating parts, both engaged by the one flag:

1. **Fused gate/up+SwiGLU MLP tile** — one Metal kernel per exact verify tile
   (M2-M6) on the vk_k split-K geometry, replacing the two dense-MLP
   projection launches plus the SwiGLU launch, so the full `gate` and `up`
   BF16 intermediates are never materialized.
2. **Exact-M QuantizedLinear routing** — every other eligible 4-bit verify
   matmul at M in {2, 3, 5} (attention, GDN, `down_proj`, lm_head at M3/M5)
   is routed through the proven vk_k split-K codegen compiled at the true row
   count (`MTPLX_QWEN_ROW_TRAVERSAL_QLINEAR` overrides this half explicitly
   in either direction for A/B work).

## Where the wins actually come from

The shipped turbo profile already routes 4-bit `QuantizedLinear` calls with
M in 4..16 (non-prefill) through the MTPLX verify kernels
(`install_nax_qlinear_patch`): M4 runs the vk_k split-K tile, M5/M6 run the
M6 split-K tile (M5 zero-padded). **M1-M3 stay on stock MLX `qmv`, which
re-reads the full weight matrix once per output row**, and M5 pays the
padded-M6 waste.

Speculative depth K verifies M = K+1 rows, so the per-depth in-context
baselines differ structurally:

| depth | verify tile | turbo in-context baseline | row-traversal change |
| :---: | :---: | :--- | :--- |
| K1 | M2 | stock `qmv` (2x weight reads) | fused MLP tile + exact-M2 routing |
| K2 | M3 | stock `qmv` (3x weight reads) | fused MLP tile + exact-M3 routing |
| K3 | M4 | NAX vk_k pair (already 1x) | fused MLP tile (merges 2 launches) |
| K4 | M5 | NAX padded-M6 tile | fused MLP tile + exact-M5 routing |
| K5 | M6 | NAX M6 tile (already 1x) | fused MLP tile (merges 2 launches) |

This is why the original MLP-only measurement was large at K1/K2, moderate at
K4, and flat at K3/K5: the flat cells were already served by the tuned vk_k
family, so merging two already-fast launches buys launch overhead only. The
earlier draft attributed the K3/K5 flatness to attention dilution; the NAX
coverage map above is the verified cause (control verify-forward time at K3,
40.9 ms, is *lower* than K2's 49.7 ms despite one more row — the M3->M4
baseline kernel handoff is visible in control itself).

For the same reason, the isolated MLP microbench's `stock` arm (raw MLX
modules) is the correct in-context baseline **only at M2/M3**; at M4-M6 the
shipped baseline is the vk_k family, so the +33..42% isolated columns there do
not translate end-to-end.

## Verify-call census (live shapes)

QuantizedLinear call census over real generations (turbo control config,
1K context, per depth; counts per verify forward, 64-layer trunk):

| shape (K -> N) | calls/verify fwd | role | exact-M eligible |
| :--- | ---: | :--- | :---: |
| 5120 -> 17408 | 128 | MLP gate+up (fused MLP tile handles these) | via MLP tile |
| 17408 -> 5120 | 64 | MLP down_proj | yes |
| 6144 -> 5120 | 64 | GDN/attention out projections | yes |
| 5120 -> 6144 | 48 | GDN in projections | yes |
| 5120 -> 10240 | 48 | GDN in projections | yes |
| 5120 -> 12288 | 16 | full-attention QKV | yes |
| 5120 -> 248320 | 1 | quantized lm_head | M3/M5 only |
| 5120 -> 48 | 96 | GDN b/a projections | no (tiny-N floor) |
| 5120 -> 1024 | 32 | KV projections | no (tiny-N floor) |

The lazy-bonus commit forwards also run M in {2, 3, 5} at depths 2-5
("unknown" phase, same shapes) and are covered by the same routing — including
M3 commits at the shipped depth-3 configuration.

## Design

- Reads the original Q4 arrays directly; no packed duplicate, no
  peak-memory increase.
- MLP tile: output-tile-major traversal, BN=2, two fixed K parts, one kernel
  per exact verify tile (M2-M6). Each output row reduces in its own FP32
  accumulator, so live rows are bit-identical whether or not the tile is
  padded; removing the pad removes the zero-padded activation copy the fixed
  M4/M6 tiles materialized on every call. BN=4 needs 32 FP32 accumulators
  (over the register ceiling) and measured slower.
- Exact-M routing: `verify_kernels._build_ksplit_kernel` (the shipped vk_k
  codegen, already parameterized over M) compiled at M in {2, 3, 5} with the
  K constant baked (vk_k morphology), grid `N/4` column tiles.
- Eligibility floors are measured, not guessed: tiny-N projections (N=48 GDN
  b/a, N=1024 KV) lose to stock qmv at every M (launch-bound), and M2 loses
  on lm_head-class N (>= 100k), so those stay stock. Everything else with
  N >= 2048 wins.
- Install order: the exact-M wrapper installs after the NAX patch so exact-M5
  takes precedence over the padded-M6 NAX route; M4/M6/M7-16 fall through to
  NAX unchanged, everything else to stock.
- Autoregressive decode (M1) and prefill are untouched in both halves.

## Numerical contract and measured characterization

The row traversal does not reduce dtype precision: Q4 storage, BF16/FP16
projection boundaries, FP32 products and accumulators are unchanged. It does
change the FP32 addition tree, and FP32 addition is non-associative, so
low-order bits can differ from stock. This is the same accepted numerics class
as the shipped turbo vk_k verify kernels ("argmax- and
sampler-distribution-validated, not bit-exact vs stock"), and every lane
self-validates at model load through `kernel_selfcheck` with the same
disable-on-mismatch contract as the NAX lanes (`qwen_rt_qmm_m2/m3/m5`,
`qwen_rt_gateup`).

Measured characterization of the divergence (overflow/saturation probe, real
gate/up weights from layers 0/31/63, fp64 numpy reference, activation-RMS
ladder 0.03..30 at M2/M3):

- Zero non-finite values in either arm at any magnitude; pre-activations up
  to 130 and outputs up to ~9.6e3 stay far inside BF16/FP32 range.
- Fused-vs-fp64 max error: 0.78-3.23 BF16 ULP; stock-vs-fp64: 1.64-3.95 ULP.
  The fused tile is at or slightly below stock's own rounding error, flat
  across a 1000x magnitude range (no error growth with contraction length,
  no clipping signature).
- Fused-vs-stock disagreement: 1.4-5.2 ULP — symmetric low-order rounding.

At greedy temperature this low-order noise can cross a decision boundary and
fork the generation. In the diverged cells the forked texts were decoded and
inspected: both arms coherent and on-task, the fused arm with equal-or-higher
distinct-token ratio (0.74-0.75 vs 0.69-0.73) and no repetition degeneracy.
Verify-forward times improve identically in parity and diverged cells of the
same depth, so diverged-cell tok/s deltas are trajectory effects (different
text, different draft acceptance) and are disclosed, never credited.

## Isolated microbenches (queued lane)

Fused MLP tile, real dense-MLP shape K=5120, N=17408, Q4 gs=64. `stock` is
two raw MLX projections + SwiGLU — the in-context baseline only at M2/M3 (see
above); `padded` is the fixed-M4/M6 fused tile.

| verify tile | K | stock us | padded us | exact us | exact vs padded | exact vs stock |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | 1 | 151.74 | 191.48 | 156.35 | +18.34% | -3.04% |
| M3 | 2 | 223.45 | 210.59 | 184.68 | +12.30% | +17.35% |
| M4 | 3 | 360.87 | 241.75 | 241.97 | -0.09% | +32.95% |
| M5 | 4 | 473.99 | 316.66 | 268.07 | +15.35% | +43.44% |
| M6 | 5 | 582.60 | 348.93 | 337.16 | +3.37% | +42.13% |

Exact-M QuantizedLinear routing vs stock `qmv` on the live census shapes
(medians; `padded M6` shown at M5 where it is the shipped turbo route):

| shape | M2 | M3 | M5 | M5 vs padded M6 |
| :--- | ---: | ---: | ---: | ---: |
| 5120 -> 6144 | +17.9% | +26.2% | +31.8% | 63.5 -> 58.8 us |
| 5120 -> 10240 | +8.3% | +27.2% | +39.1% | 98.4 -> 83.6 us |
| 5120 -> 12288 | +9.5% | +27.8% | +39.4% | 116.7 -> 99.5 us |
| 5120 -> 17408 | +13.1% | +28.5% | +40.1% | 166.9 -> 143.5 us |
| 6144 -> 5120 | +8.9% | +25.4% | +38.9% | 69.5 -> 62.5 us |
| 17408 -> 5120 | +18.7% | +32.2% | +41.4% | 167.1 -> 152.1 us |
| 5120 -> 248320 (lm_head) | -6.4% (stays stock) | +34.0% | +45.0% | 2704.8 -> 2134.4 us |
| 5120 -> 1024 | -3.1% (stays stock) | -6.2% (stays stock) | +10.2% (stays stock, floor) | — |
| 5120 -> 48 | -72% (stays stock) | -95% (stays stock) | -81% (stays stock) | — |

## End-to-end: definitive quiet pairing (16K, K0-K4)

Total-quiet protocol: exclusive GPU window (serving booted out under the
guard), the rANS converter SIGSTOPped for the duration (verified `T` state),
no other process above 50% CPU, arms in alternating fresh processes
(control, fused, control, fused, ...) so drift cannot favor an arm, three
repeats, greedy seed 51, 96 tokens, fresh 16K prefill per generation.
Control drift across the window: at most 1.34% (K1), 0.0% at K0. Repeat
spread within each cell: under 0.6%. Fused = the full flag (fused MLP tile +
exact-M2/M3/M5 routing). Engagement verified in-window by dispatch counters.

| K @16K | control tok/s | fused tok/s | delta | verify/fwd | acceptance | tokens |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| K0 | 28.06 [28.05, 28.06] | 27.75 [27.75, 27.92] | -1.08% | 35.1 -> 35.5 ms | - | identical |
| K1 | 41.27 [40.87, 41.43] | 47.15 [47.13, 47.20] | +14.25% | 40.9 -> 37.8 ms (-7.7%) | 44/52 -> 46/49 | fork |
| K2 | 44.25 [43.99, 44.29] | 57.91 [57.89, 57.94] | +30.86% | 51.7 -> 40.3 ms (-21.9%) | 59/74 -> 61/70 | fork |
| K3 | 63.72 [63.56, 63.77] | 63.40 [63.28, 63.54] | -0.51% | 44.0 -> 44.1 ms | 67/84 = | identical |
| K4 | 56.76 [56.51, 56.84] | 62.01 [61.96, 62.14] | +9.23% | 59.1 -> 53.0 ms (-10.3%) | 72/94 = | identical |

Reading it honestly:

- K4 is the cleanest cell in the whole investigation: +9.23% end-to-end with
  bitwise-identical tokens and identical acceptance — pure kernel effect
  (exact-M5 routing replacing the padded-M6 route plus the M5 MLP tile).
- K1/K2 combine two effects. The kernel-real component is the verify-forward
  drop (-7.7% and -21.9%): stock `qmv` re-reads the full weight matrix per
  output row at M2/M3, so depths 1-2 are where the routing bites hardest.
  The rest of the tok/s delta is the acceptance ratchet on the forked
  trajectory (below) and is workload-dependent serving throughput, not
  kernel speed.
- K0 (-1.08%) and K3 (-0.51%) price the Python dispatch wrappers: at depths
  where no new kernel engages (M1 decode, M4 already NAX-served) the patch
  costs ~0.2-0.4 ms/forward of per-call eligibility checks. Acceptable for
  an experimental flag; a phase-first short-circuit would trim it if the
  flag ever graduates.
- The production serving depth is K3 under turbo, where this flag is net
  ~0.5% negative today. The flag pays at K1/K2/K4 and any config where
  depth-1/2 tiles dominate (short-depth speculative serving, draft-heavy
  regimes).

### Context sweep (4K/8K/16K, corroboration only)

The full K0-K5 x three-context matrices from the earlier (non-quiet) windows
reproduce the same shape: M2/M3-heavy depths win large, K3/K5 flat-positive,
parity in exactly the same cells every window. Those absolute deltas carried
up to 4-7% cross-window drift (a CPU-heavy rANS conversion ran on the box
and arms were sequential), which is why the quiet interleaved pairing above
is the citable table. 8K quiet spot-check (same protocol, alternating fresh
processes, 3 repeats):

| K @8K | control tok/s | fused tok/s | delta | verify/fwd | acceptance | tokens |
| ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| K1 | 45.15 | 48.75 | +7.95% | 38.8 -> 35.7 ms (-8.2%) | 46/50 = | identical |
| K2 | 47.73 | 59.66 | +25.00% | 49.0 -> 38.0 ms (-22.5%) | 60/72 = | identical |

These two cells are the flag's cleanest end-to-end statement: bitwise
token parity, identical acceptance, and +8/+25% wall speed from kernel time
alone — at 8K no near-tie happens to sit on the trajectory, so the M2/M3
association difference never surfaces in the tokens.

### Method note: how the box fooled us, twice

Two failure modes surfaced during verification and are now protocol here:

1. Environmental contamination: sequential arm blocks measured 25+ minutes
   apart, with a CPU-bound converter sharing the box, moved controls by
   4-7% between windows — the same order as several claimed wins. Rule: no
   end-to-end claim without same-window interleaved arms under the quiet
   protocol (bootout + SIGSTOP + hog check + drift anchors).
2. Harness self-deception: an in-process arm-toggle harness produced a
   plausible-looking all-flat grid because the fused path silently never
   engaged (verify-forward times identical to control to 0.0% was the tell;
   dispatch counters confirmed). Rule: every arm must carry an engagement
   signature (counter deltas or the mechanical verify/fwd drop), and a
   too-clean null is a bug until proven otherwise.

## Divergence: mechanism, measurement, and the acceptance ratchet

The exact-M kernels are bit-identical to the padded tiles they replace at
M4/M5/M6 (proven by construction and by token parity at K3/K4/K5 in every
window). At M2/M3 the association differs from stock `qmv` (lane-strided
8-value packs, split-K partials, factored-vs-per-element dequant), so
low-order bits differ, and at greedy temperature a near-tie can fork.

Measured at the actual forks (logit dumps at the first diverging verify
call, stock and fused arms replayed to the same state):

- 16K fork: stock's top-2 margin was 0.25 logits (`<|im_end|>` 20.625 vs
  `\n\n` 20.375); fused swapped the same two candidates (19.0 vs 18.75). The
  fused pick was stock's runner-up. 4K fork: same junction character, 2.0
  margin crossed.
- In the same verify forward, the non-fork row agreed within 0.25 logits
  while the fork row reshuffled by ~4 — junction-local amplification of
  sub-ULP per-layer noise through 64 layers, not kernel distortion. The
  fp64 ladder bounds the per-call error at or below stock's own (0.8-3.2
  vs 1.6-4.0 BF16 ULP), flat across a 1000x magnitude range.
- Stock replay is bit-identical run-to-run (logit delta 0.0), and the BN2
  and BN4 tile layouts are bit-identical to each other (the association is
  layout-invariant): the fork is a deterministic property of the two valid
  association classes, not nondeterminism.

The acceptance ratchet explains why forked cells post outsized tok/s gains:
the MTP draft's disagreements with the target concentrate at the target's
near-ties, so target-side tie-flips convert prior draft-rejections into
acceptances about half the time while rarely breaking prior agreements.
Acceptance can only drift up under divergence. Measured: diverged cells
jumped +7.4 to +9.3 acceptance points, the jump anti-correlated with base
acceptance (92% base -> +1.9pt; 80-85% base -> +7-9pt); all parity cells
moved 0.0pt. Forked-cell tok/s is therefore disclosed as workload throughput
and the verify-forward time is reported alongside as the kernel-real metric.

Forked texts were decoded and inspected in every diverged cell: both arms
coherent and on-task, fused with equal-or-higher distinct-token ratio, no
repetition degeneracy.

### Bit-parity option (prototype)

A stock-order variant of both kernels — reproducing stock `qmv_fast`'s
association exactly (contiguous 16-value spans per lane in block order,
factored qdot with the separate Sigma-x accumulator, one sequential fp32
accumulator per output, single `simd_sum`, T-arithmetic quad sums) while
still amortizing the weight read across the m verify rows — was prototyped
and gate-tested:

- Single-matrix QMM: **bit-exact vs `mx.quantized_matmul` at M2 and M3**
  across all six live trunk shapes, three activation scales, `mx.array_equal`
  — 36/36 checks in two independent runs. Kernel speed: +9-11% vs stock at
  M2 (within 5-7% of the vk_k variant), +15-16% vs stock at M3 (vk_k keeps a
  ~19-20% lead there — the register cost of stock's 16-value contiguous
  spans at m rows is real).
- M5-vs-stock-qmv mismatches at all scales: expected and moot — MLX
  dispatches a different kernel family above M3/M4, and the turbo baseline
  at M5 is the NAX padded route, against which the shipped exact-M5 kernel
  is bitwise-identical (below).
- Gate/up stock-order tile: bit-exact at M2 in the first build; the
  register-restructured build regressed large-activation inputs (RMS ~2,
  where gate pre-activations reach the measured 130 range) — defect known,
  unresolved, so the tile half is prototype-only.

Separately, the structural parity of the SHIPPED kernels against the turbo
baseline was gate-tested with `mx.array_equal` on real shapes:

- exact-M5 QMM vs NAX padded-M6: **bitwise equal (dmax 0.0)**.
- fused MLP tile at M4 vs turbo vk_k pair + compiled swiglu: **bitwise
  equal (dmax 0.0)**; same at M6.

So K3/K4/K5 token parity is bit-parity **by construction**, and the flag's
entire numerics delta versus the turbo baseline is confined to the M2/M3
tiles. A follow-up could ship stock-order kernels for exactly those two
row counts (QMM half proven today) and make the whole flag bit-identical
to baseline at a quantified kernel-speed cost.

## Standalone fast sigmoid

Fast sigmoid is deliberately separate and not bundled with row traversal. A
Vec8 standalone SwiGLU keeps both projections on their stock paths (so it
keeps the two intermediates) and changes only precise `metal::exp` to
`metal::fast::exp`. It is the counter-example to the fusion: it adds a launch
that reads the intermediates instead of removing them.

Measured end-to-end (4K, K0-K5, non-quiet window): deltas within noise
(-3.0% to +3.2%), while introducing K1/K2 forks of its own (fast::exp is a
different numerics class from the MLX `Sigmoid` op). No speed, new
divergence: not worth pursuing. The flag stays available and default-off.

## Decisions

Keep both flags experimental and default-off. Row traversal ships as the
fused exact-M MLP tile plus the exact-M2/M3/M5 QuantizedLinear routing under
the one flag: the fusion eliminates the gate/up intermediates, and the
routing removes the per-row weight re-reads that stock `qmv` pays at the
depth-1/depth-2 verify tiles and the padded-M6 waste at depth 4. The
established turbo numerics class (vk family, load-time self-validated)
applies. Not proposed for any default profile: at the production serving
depth (K3 turbo) the flag is ~0.5% negative from wrapper overhead; its wins
live at K1/K2/K4. If a serving config ever moves to those depths, the
stock-order bit-parity variant should be re-evaluated as the default
numerics for the M2/M3 tiles (kernel-level tradeoff quantified above).
