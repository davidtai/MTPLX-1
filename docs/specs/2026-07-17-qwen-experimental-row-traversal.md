# Experimental Qwen Gate/Up Row-Traversal Fusion

## Status

Experimental and default-off. Based on public upstream `a391973` (`v2.1.0`).
Two independent flags are added, both off by default and mutually exclusive:

- `--experimental-qwen-row-traversal`
- `--experimental-qwen-fast-sigmoid`

Neither changes any default profile.

## What it does

Qwen3-Next's dense MLP runs `down_proj(swiglu(gate_proj(x), up_proj(x)))`. On
the speculative-verify path the stock activation launches two separate Q4
projection kernels and a third SwiGLU kernel, materializing the full `gate` and
`up` projections as BF16 intermediates between launches.

The row-traversal flag fuses `gate_proj + up_proj + precise SwiGLU` into one
Metal kernel built on the existing small-M split-K (`vk_k`) ownership. One
output tile is owned by one threadgroup, which reads the two original Q4 arrays,
reduces both projections through the same fixed split-K tree in FP32 registers,
applies precise SwiGLU at the projection boundary, and writes only the activated
tile. `down_proj` stays on its existing path.

The point of the fusion is to keep the whole gate/up + SwiGLU chain in registers
and threadgroup memory so the two BF16 projection intermediates are never
materialized. There is no packed weight copy and no peak-memory increase.

## Design

- Reads the original gate/up Q4 arrays directly; no packed duplicate.
- Output-tile-major traversal with BN=2 and two fixed K parts.
- Compiles one kernel per exact verify tile (M2-M6). Owning the true row count
  keeps every live row's reduction in its own FP32 accumulator and removes the
  zero-padded activation copy a fixed M4/M6 tile would otherwise materialize on
  every call. Because each output row reduces over an independent accumulator,
  the live rows are bit-identical whether or not the tile is padded.
- Keeps the current Q4 storage, BF16/FP16 activation/output types, and FP32
  products and accumulators.
- Falls back to the stock path outside the narrow Q4 affine M2-M6 eligibility
  contract (autoregressive M1 / K0 always runs stock).

BN=4 was measured slower than the two-call path: carrying both projections at
BN=4 needs 32 FP32 accumulators per thread, over the register ceiling. BN=2
stays at 16 accumulators for the M4 tile and 24 for M6, at the ceiling, and
restores occupancy while retaining one dispatch.

## Numerical contract

The fusion does not reduce dtype precision: Q4 storage, BF16/FP16 projection
boundaries, and FP32 accumulation are unchanged. It does change the FP32
addition tree relative to the two stock projection kernels. Floating-point
addition is non-associative, so regrouping the same FP32 products can change
low-order bits even though precision is unchanged. "Same precision" therefore
does not promise bitwise identity with stock.

In practice most verify cells stay bit-parity with stock. In a few cells a
low-order rounding difference crosses a greedy decision boundary and changes the
downstream accepted/drafted workload; those cells are reported for completeness
and not credited as isolated kernel speedups.

The exact-tile change is numerically inert: it only removes the zero-pad, so its
output is bit-identical to the padded tile on the live rows, and it does not move
any cell across the parity boundary.

## Isolated kernel microbench (queued lane)

Real dense-MLP shape K=5120, N=17408, Q4 affine group_size=64. Median per-call
microseconds on the queued lane (fill the command buffer with independent calls,
single synchronize). `stock` is two Q4 projections + SwiGLU; `padded` is the
fixed-M4/M6 tile with the zero-pad copy; `exact` is the per-tile kernel.

| verify tile | K | stock us | padded us | exact us | exact vs padded | exact vs stock |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | 1 | 151.74 | 191.48 | 156.35 | +18.34% | -3.04% |
| M3 | 2 | 223.45 | 210.59 | 184.68 | +12.30% | +17.35% |
| M4 | 3 | 360.87 | 241.75 | 241.97 | -0.09% | +32.95% |
| M5 | 4 | 473.99 | 316.66 | 268.07 | +15.35% | +43.44% |
| M6 | 5 | 582.60 | 348.93 | 337.16 | +3.37% | +42.13% |

Removing the zero-pad recovers 12-18% of kernel time at the padded tiles
(M2/M3/M5). M4 and M6 are never padded, so exact and padded coincide within
measurement noise, which doubles as a harness sanity check. Against stock, the
fused kernel is faster everywhere except M2, where the two stock projections are
already cheap.

## End-to-end K0-K5 x 4K/8K/16K matrix

Control (stock) and the exact-tile candidate were measured back to back in one
exclusive GPU window (turbo profile, greedy, seed 51, 96 generated tokens, three
repeats after one warmup per cell), on public upstream `a391973`.

| context | K | control tok/s | exact-M tok/s | delta | verify delta | accepted/drafted (control -> exact-M) | token parity |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | :---: |
| 4096 | 0 | 29.486 | 29.317 | -0.57% | +0.58% | 0/0 -> 0/0 | yes (stock fallback) |
| 4096 | 1 | 45.665 | 49.182 | +7.70% | -7.61% | 46/50 -> 46/49 | no |
| 4096 | 2 | 47.470 | 56.465 | +18.95% | -17.34% | 59/72 -> 61/68 | no |
| 4096 | 3 | 72.002 | 71.958 | -0.06% | -0.31% | 69/80 -> 69/80 | yes |
| 4096 | 4 | 65.570 | 68.053 | +3.79% | -4.61% | 73/91 -> 73/91 | yes |
| 4096 | 5 | 71.371 | 71.536 | +0.23% | -0.31% | 76/98 -> 76/98 | yes |
| 8192 | 0 | 29.174 | 29.106 | -0.23% | +0.26% | 0/0 -> 0/0 | yes (stock fallback) |
| 8192 | 1 | 44.376 | 46.777 | +5.41% | -5.66% | 46/50 -> 46/50 | yes |
| 8192 | 2 | 46.332 | 51.397 | +10.93% | -11.25% | 60/72 -> 60/72 | yes |
| 8192 | 3 | 64.552 | 64.002 | -0.85% | +0.54% | 67/87 -> 67/87 | yes |
| 8192 | 4 | 44.862 | 46.389 | +3.40% | -4.45% | 65/123 -> 65/123 | yes |
| 8192 | 5 | 59.933 | 59.901 | -0.05% | +0.09% | 74/107 -> 74/107 | yes |
| 16384 | 0 | 28.077 | 27.873 | -0.73% | +0.76% | 0/0 -> 0/0 | yes (stock fallback) |
| 16384 | 1 | 40.394 | 45.067 | +11.57% | -10.79% | 44/52 -> 46/49 | no |
| 16384 | 2 | 42.096 | 49.315 | +17.15% | -15.95% | 59/74 -> 61/70 | no |
| 16384 | 3 | 61.790 | 61.415 | -0.61% | +0.62% | 67/84 -> 67/84 | yes |
| 16384 | 4 | 54.783 | 56.325 | +2.81% | -3.47% | 72/94 -> 72/94 | yes |
| 16384 | 5 | 54.325 | 53.679 | -1.19% | +1.33% | 74/105 -> 74/105 | yes |

Every exact-M cell produced tokens bit-identical to the prior padded fused run
(18/18), confirming the exact-tile change is numerically inert and that the
fused kernel engaged rather than silently falling back to stock.

K0 is a stock-fallback control (the fusion applies only to M2-M6); its small
deltas are run noise. Among the affected cells with identical tokens and
acceptance, the padded verify shapes where the zero-pad was removed improve
clearly: K1/8K went from -0.95% in the prior padded run to +5.41% here, K2/8K is
+10.93%, and K4 is +2.8 to +3.8% across contexts. The direct M4/M6 shapes (K3,
K5) land flat within +/-1.2%. The isolated microbench shows the fused kernel is
30-42% faster than stock at those shapes, but the MLP is a small fraction of
per-token time -- especially at 8K/16K where attention over the KV cache
dominates -- so the kernel win dilutes to noise at the token level. No parity
cell regresses beyond run-to-run noise, and the exact-tile change removed the
only regression the padded path had shown (K1/8K), so no shape is scoped out.

The four non-parity cells are K1/K2 at 4K and 16K. Each arm is deterministic,
but the changed FP32 reduction tree moved a low-order value across a greedy
decision boundary and changed the downstream acceptance workload, so their
throughput deltas are reported for completeness, not credited as isolated kernel
speedups.

## Standalone fast sigmoid

Fast sigmoid is deliberately separate and not bundled with row traversal. A
Vec8 standalone SwiGLU keeps both projections on their stock paths (so it keeps
the two intermediates) and changes only precise `metal::exp` to
`metal::fast::exp`. It is the counter-example to the fusion: it adds a launch
that reads the intermediates instead of removing them, and it does not show a
matched-cell win. It is exposed only for explicit A/B work behind
`--experimental-qwen-fast-sigmoid` and remains default-off.

## Decisions

Keep both paths experimental and default-off. The exact-tile fusion is the
adopted design because it eliminates the gate/up intermediates, removes the
zero-pad copy, and is no worse than the padded tile everywhere while recovering
12-18% at the padded verify shapes. Context-dependent numerical differences from
the changed FP32 reduction tree remain, so this is not proposed for any default
profile.
