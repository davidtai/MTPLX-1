# Python and MLX optimization map

This map covers the streamed Hy3 and GLM-5.2 decode path on the stacked
`codex/default-streamline` base. It records which Python and MLX operations are
on the critical path, which local substitutions were tested, and which larger
mechanisms remain credible.

The compact raw result summary is
[`benchmarks/results/python-mlx-optimization-gate-20260712.json`](../benchmarks/results/python-mlx-optimization-gate-20260712.json).

## Execution map

```text
resident layer
  -> router projection and top-k selection
  -> mx.eval(indices) and GPU-to-host route transfer
  -> Python cache lookup and hit/miss planning
  -> SSD reads for misses
  -> grouped component-bank gather_qmm
       gate Q4 + up Q4 -> SwiGLU -> down Q4
  -> concatenate grouped rows
  -> argsort(output_positions) + take
  -> router-score multiply + top-k reduction
  -> shared-expert add
  -> next layer
```

The route synchronization is authoritative: the host must know the selected
experts before it can issue SSD reads. The grouped Q4 path deliberately emits
rows in execution order because grouping and ready-miss processing are what
preserve QMM efficiency and I/O overlap.

## Measurement lane

All end-to-end measurements used one MLX process, unloaded Qwen, the pinned
local sidecars, deterministic generation, `component-banks`, `F_NOCACHE`, a
112 GiB memory limit, a 78 GiB expert-cache cap, 32 transient slots, and a
64 MiB read chunk. Candidate outputs were token-identical to their bases unless
the candidate was rejected at an earlier parity probe.

Hy3 used both a three-repeat 256-token screen and a sustained natural-stop
gate. The sustained base generated 1,905 completion tokens at 6.442 raw decode
tok/s. GLM-5.2 used a three-repeat 64-token screen because its baseline is
1.75-1.81 decode tok/s.

## Rejected local substitutions

| Candidate | Intended removal | Result | Decision |
|---|---|---:|---|
| Broadcast the one decode token to grouped assignments | Per-group index array and `mx.take` | 6.358 vs 6.442 Hy3 sustained tok/s (-1.30%) | Revert |
| Compute the inverse row permutation in Python | Tiny Metal `argsort` | 6.350 vs 6.472 tok/s for the prior broadcast-only short arm | Revert |
| Scatter grouped rows with `mx.put_along_axis` | `argsort + take` | 6.158 vs 6.442 Hy3 sustained tok/s (-4.42%) | Revert |
| Cache all GLM router weights in FP32 | Per-token BF16-to-FP32 router casts | 1.759 vs 1.812 tok/s (-2.96%) and +225 MiB resident | Revert |
| Batched score-by-expert matmul | Score multiply plus top-k reduction | Not bit-exact; max BF16 difference 0.125 | Reject before timing |
| Compile Hy3 selection subgraph | Python graph construction around sigmoid/top-k | About 2.9 us/layer, or 0.23 ms/token estimated | Reject below significance |

The short Hy3 screens sometimes gave the scatter or broadcast arms an apparent
sub-2% win. The sustained gates reversed those results. A short microbenchmark
or operation-count reduction is therefore not sufficient for promotion.

GLM's `group_expert_select` is already compiled by `mlx_lm`. The visible
`weight.astype(mx.float32)` also did not behave like 675 MiB/token of removable
critical-path traffic in the end-to-end lane; MLX's lazy execution already
handles it more efficiently than the source expression suggests.

## Retained conclusions

### 1. Fuse weighted reassembly, not standalone sorting

The useful kernel boundary begins after the down projection. A native Stage B
primitive can consume:

- down-projection rows in execution order;
- the assignment-position mapping;
- router scores in original top-k order;

and produce the final `[tokens, hidden]` routed output directly. One grid
element per `(token, hidden)` can loop over top-k in reference order, avoiding
atomics while eliminating the top-k output tensor, reorder gather, score
multiply, and separate reduction. This is tracked in
[#30](https://github.com/davidtai/MTPLX/issues/30).

Exactness must cover Hy3's activation-dtype reduction, optional FP32 combine,
GLM's final cast, repeated experts, and B=1/2/4/8 assignment mappings.

### 2. The per-layer synchronization boundary is larger than Python cleanup

Every streamed sparse layer evaluates router indices and transfers them to the
host before the next expert dispatch. This prevents cross-layer Metal
pipelining. The cache-miss path genuinely needs that decision; an all-hit path
can avoid it only if routing and residency remain device-visible without
weakening the authoritative-router contract. This is tracked in
[#27](https://github.com/davidtai/MTPLX/issues/27).

### 3. Stock gathered Q4 is already near its local memory roofline

The custom expert-kernel spike in
[#28](https://github.com/davidtai/MTPLX/issues/28) measured stock
`mx.gather_qmm` close to the single-dispatch bandwidth floor. Replacing its
arithmetic or dispatch shape without removing bytes is unlikely to produce a
stable end-to-end gain. Future kernel work should remove an intermediate,
remove a synchronization boundary, or change physical bytes—not merely rename
an MLX operation.

## Promotion rule

A Python, MLX, or Metal change should be retained only after:

1. behavior-locking tests cover route IDs, scores, assignment order, dtypes,
   slot lifetime, and deterministic output;
2. the candidate beats an immediate predecessor under an exclusive-machine
   matched lane;
3. a sustained decode gate confirms the short screen;
4. memory-plan changes are explicitly accounted; and
5. the result is large enough to exceed observed run-to-run variance.

No runtime change from this investigation passed those gates. The branch keeps
the evidence and map rather than shipping a plausible-looking regression.
