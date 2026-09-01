# PR391 fixed-M4 paired routed-GLU result

## Candidate boundary

Commit `61f0bebc075719bbe45f4a8efc83b5d00be737d4` replaces only the
physical-M4 routed fused gate/up projection plus `SiLU * up` producer. The
active MLX 0.32.2 route is affine q4/group-32 `gather_qmv_fast`, not NAX
gather-QMM: the four verifier rows and ten selected experts are represented as
40 batched M=1 products.

The paired kernel preserves that route's 64-thread ownership, two SIMD groups,
16 BF16 inputs per lane, five 512-value K blocks, FP32 affine accumulation,
`simd_sum`, and BF16 gate/up dot boundaries. It pairs fused-pack rows `j` and
`640 + j`, reuses each hidden-input tile, applies the stock stable BF16
sigmoid/SILU/product boundaries, and emits only `[4,10,640]` routed
activations. Router/top-k, route scores, the shared path, the retained routed
down reduction, and the retained residual tail are unchanged.

The literal whole routed-expert kernel was rejected before implementation.
Down-output-tile ownership would recompute every gate/up dot about 320 times;
expert-slot ownership would materialize `[4,10,2560]` or require an inexact
cross-threadgroup reduction; row ownership would expose only four work units.

The candidate is construction-bound by `MTPLX_QWEN4_M4_ROUTED_GLU=1`, requires
the retained routed-down and residual-tail lanes, validates all 48 exact storage
contracts before mutation, and installs a distinct physical-M4 decoder class.
The enabled path has no environment read, eligibility fallback, exception
fallback, or engagement counter.

Branch: `experiments/pr391-whole-routed-expert`

## Minimal gates

The focused CPU/static suite passed `65 passed`. A guarded production-shape
Metal oracle used the exact M4/K2560/N1280 fused-GU geometry and all 40 selected
row-expert lanes. Every `[4,10,640]` BF16 activation bit matched stock
`gather_qmm -> split -> SiLU * up`; feeding both results through the retained
routed-down/residual-tail callable also matched exactly.

Full-model construction then self-checked all 48 layers with
`max_abs_diff: 0.0`, `paired_routed_glu_layers: 48`, and no partial install.

## Strict 40 C arm

The exact 16,384-token prompt / 1,024-token output arm used seed `20260829`,
artifact revision `29ba90f82124961d0d902a9ea9bbb1034972af2f`, D3/M4, fused QSA
K/V gather, full 65,536-row draft support, and compiled MTP preparation.

| Arm | Ready C | Decode s | Decode tok/s | Wall s | Verify forward s | Peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paired routed GLU | 39.5737 | 15.099175 | 67.818277 | 28.893670 | 11.695744 | 87,393,848,312 |
| Retained same-seed reference | 39.7640 | 15.354728 | 66.689556 | 29.025788 | 11.938083 | 87,393,848,280 |

Against the same-seed retained reference, the candidate saved `0.255554 s`
decode time, gained `1.128721 tok/s` (`1.6925%`), saved `0.132118 s` wall
time, and saved `0.242339 s` verifier-forward time. Peak memory changed by
only `32 bytes`.

Correctness and engagement were exact and non-vacuous: the paired route was
reported expected and observed, the response-token digest remained
`e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc`,
accepted/drafted-by-depth remained `[259,187,120]` / `[382,382,382]`, all 382
M4 target calls compiled, and fallback, demotion, repair, capacity-transition,
and route-transition counts were zero.

Receipt:

- `.benchmark-artifacts/pr391/rebench3-1788287001-paired-routed-glu-candidate-seeds-16k-1k-seeds-16k-1k.json`
  (`be4dc5c45afe5678e1a5182064790d5bdb28f204796ec8a127ac19f84bc1a0cd`)

The production service was restored healthy and idle with background warmup
complete and default fan mode; the GPU lock had no owner after the arm.

## Decision

Retain the paired routed-GLU producer as a measured positive candidate. One
strict arm is sufficient for composition because it is bit-exact, improves
decode, verifier-forward, and wall time simultaneously, and does not increase
memory materially. Do not spend another full-model arm on it before composing
the next attributable boundary; require a matched reverse-order pair only for
final promotion of the composed lane.
