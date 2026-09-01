# PR391 fixed-M4 routed-down plus residual-tail result

## Candidate boundary

This candidate combines the retained fixed-M4 routed-down reduction with the
MLP hyper-residual write tail. It is construction-bound to physical M4, and
rows 1 through 3 remain on the stock decoder path. The installed boundary is
reported as `routed_q4g32_reduce_shared_add_mlp_residual`.

The candidate uses exactly two Metal dispatches:

1. The first dispatch performs the routed q4/group-32 FP32 dot products,
   narrows each dot to BF16, multiplies by the BF16 route score, and reduces in
   the exact existing order: `0 + 8`, `1 + 9`, combine those pairs, then
   left-fold slots 2 through 7 with a BF16 narrowing at every add.
2. The second dispatch consumes only the reduced routed result plus the stock
   shared-down result and factor, hyper state, and inject values. It performs
   the BF16 shared product, BF16 block add, BF16 inject product, and BF16 hyper
   add, then emits `[1, 4, 10240]`.

Shared, hyper, and inject work is not pulled into the routed q4 kernel. The
shared q8/group-64 down projection remains stock. Installation requires the
routed-down reduction flag, validates the exact decoder and MoE owners for all
48 layers before binding or mutation, and installs one prebound callable for
all 48 layers. There is no enabled-path environment read, eligibility
fallback, proof counter, or arithmetic fallback.

Branch: `experiments/pr391-m4-routed-down-residual-tail`

- implementation commit: `2431d5e6`
- reviewed test/ownership head: `199704f0`

## Test and review gates

Strict TDD recorded RED failures for the missing combined source, two-dispatch
binding, physical-M4 decoder route, construction flag, exact-owner admission,
and failure-atomic all-layer installation. The final focused suite was GREEN
with `62 passed`.

The tests pin every BF16 boundary, exact second-dispatch input identity,
two-dispatch topology, physical-M4-only routing, rows 1 through 3 stock
behavior, the construction-bound flag dependency, all-48 validation before
mutation, and the exact report boundary and layer counts. An adversarial CPU
oracle also proves that omitted BF16 narrowing or reassociation changes the
combined result.

Two independent final reviews passed the implementation and ownership checks.

## Bounded Metal micro gate

The bounded Metal micro exercised the two-dispatch callable without loading
the full model. Its result was exact and finite with output shape
`[1, 4, 10240]`; peak memory was `524,911,140` bytes.

## Full-model correctness receipt

Artifact:

`/Users/davidtai/projects/OpenSourceWTF/.benchmark-artifacts/pr391/rebench3-2930-m4-routed-down-residual-tail-correctness-seeds-16k-1k.json`

Artifact SHA256:

`43c031a40e264a96aeb23a2db9b99d90cffe47f8df6194820a08cc42f8efb0a1`

The exact model revision was
`29ba90f82124961d0d902a9ea9bbb1034972af2f`. The 16K-prompt/1K-generation
receipt used seed `20260829` and produced peak memory of `87,393,979,832`
bytes.

The candidate installed the exact route on 48 layers with
`max_abs_diff: 0.0`, `exact_layers: 48`, and
`combined_residual_tail_layers: 48`. The compiled verifier recorded
`fallback_calls: 0`; the installed candidate has no runtime fallback route.

Seed-20260829 parity was exact:

| Evidence | Exact value |
| --- | --- |
| Response-token SHA256 | `e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc` |
| Drafted by depth | `[382, 382, 382]` |
| Accepted by depth | `[259, 187, 120]` |
| Drafted tokens | `1146` |
| Accepted drafts | `566` |
| Bonus tokens | `119` |
| Correction tokens | `269` |
| Verify calls | `392` |
| Start PCG64 state SHA256 | `da7887652f7f11899e362b48661512dcf59e404f9b6bd58720f93095ccccdb3e` |
| Final PCG64 state SHA256 | `d291d8dcea3fcd76576a911b34c194ee03c740bb70b38c7d911ac1b8edce6f65` |

The receipt records digest, depth, counter, exact-reference, and RNG-state
matches, with zero differing token positions and zero edit distance.

## Retention status

Retain. A later matched-order campaign restored the strict
`max(cpu_temp_avg, gpu_temp_avg) <= 40.0 C` admission gate. Both orders won
against the same-commit routed-down parent with exact output and state work.

| Seed/order | Candidate decode s | Control decode s | Candidate TPS | Control TPS | Candidate wall s | Control wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260829, candidate then control | 15.354728 | 15.499028 | 66.689556 | 66.068660 | 29.025788 | 29.214168 |
| 20260830, control then candidate | 15.608835 | 15.718491 | 65.603872 | 65.146203 | 29.264978 | 29.345030 |
| Mean | 15.481782 | 15.608759 | 66.146714 | 65.607431 | 29.145383 | 29.279599 |

The two-pair mean saves `0.126978 s` of decode time and `0.134216 s` of wall
time. That is a `0.8135%` decode-time reduction, `0.8220%` TPS increase, and
`0.4584%` wall-time reduction. Both individual wall comparisons favor the
candidate.

Every timed arm recorded an initial and ready temperature. Candidate admission
temperatures were `39.764 C` and `39.751 C`; control admissions were `39.712 C`
and `39.532 C`. Peak memory did not increase: all retained arms stayed at or
below `87,393,864,660` bytes.

The paired receipts are:

- `.benchmark-artifacts/pr391/rebench3-1788263303-m4-routed-down-residual-tail-40c-candidate-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-1788263304-m4-routed-down-residual-tail-40c-control-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-1788263305-m4-routed-down-residual-tail-40c-control-seeds-16k-1k.json`
- `.benchmark-artifacts/pr391/rebench3-1788263306-m4-routed-down-residual-tail-40c-candidate-seeds-16k-1k.json`

The interrupted/un-gated sequence `1788263301` remains diagnostic only and is
not included in these claims.
