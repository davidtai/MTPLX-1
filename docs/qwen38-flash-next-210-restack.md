# Qwen 3.8 Flash-Next optimization restack

This branch restarts the Flash-Next optimization campaign from MTPLX 2.10
instead of carrying the historical PR #368 stack forward. Every retained
runtime change must beat the unchanged 2.10 control on the exact production
workload.

## Production benchmark contract

- Runtime base: MTPLX 2.10.0, commit
  `4ce96908cd2e5ebcb5c66ee2541d12fdf5d0423f`
- Model: `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`
- Model revision: `29ba90f82124961d0d902a9ea9bbb1034972af2f`
- MLX: 0.32.2
- Mode: native MTP, default depth 3 / physical M4, stock batched verifier
- Workload: exactly 16,384 prompt tokens and 1,024 generated tokens from the
  naturalistic Python generation-module task
- Thinking: enabled, default reasoning effort `xhigh`
- Target and draft sampler: temperature 1.0, top-p 0.95, top-k 20, min-p 0,
  presence penalty 0, repetition penalty 1
- Seed: 20260829
- Measurement: one full warmup followed by one measured request

The earlier depth-1/low-effort results are retained only as compatibility
diagnostics. They are not the production control and are not used to accept
optimizations.

## Benchmark history

The n-gram table is a 29.8 GiB file-backed mmap in every arm. Mapping the file
does not make 29.8 GiB resident: rows are faulted on demand and file-backed
pages remain reclaimable by macOS. `hot 1 GiB` adds the bounded MTPLX 2.10
`OrderedDict` row cache on top; `mmap only` does not add that second cache.
The A/B/A runs used the same prompt, seed, output, acceptance trajectory, and
432 verify calls. Repair time was zero throughout.

| Arm | Decode tok/s | End-to-end wall | Accepted / drafted | Result |
|---|---:|---:|---:|---|
| 2.10 hot 1 GiB r1 | 51.0265 | 34.4313 s | 578 / 1,282 | control |
| mmap only | **52.4803** | **33.0354 s** | 578 / 1,282 | diagnostic |
| 2.10 hot 1 GiB r2 | 51.5451 | 34.6328 s | 578 / 1,282 | control |
| hot 1 GiB mean | 51.2858 | 34.5321 s | 578 / 1,282 | control mean |

Mmap-only measured 2.33% higher decode throughput than the bracketed 1 GiB
mean and 4.33% lower wall time. This is diagnostic evidence only: the branch
does not alter MTPLX 2.10's n-gram table or cache behavior.

Temperature-zero short-prompt results are vanity measurements only and must
not be used to accept an optimization.

## Remaining GPU starvation

An MLX 0.32.2 dispatch census was collected on the same D3/M4, xhigh,
temperature-1, 16,384/1,024 workload with the production 1 GiB hot-row cache.
The trace was complete (`dropped_rows=0`) and reproduced the control's exact
578 / 1,282 acceptance trajectory across 432 verify calls. The instrumented
53.3425 tok/s result is diagnostic only; profiler throughput is not a promotion
number.

| Decode interval measurement | Result |
|---|---:|
| Decode wall | 19.1967 s |
| Metal GPU busy | 15.8964 s |
| Metal GPU idle | **3.3003 s** |
| GPU utilization | **82.81%** |
| Idle gaps at least 10 us | 4,320 gaps / 3.2239 s |
| Host-late portion of all gaps | 2.6561 s |
| Largest decode gap | 8.056 ms |
| `waitUntilCompleted` | 1 wait / 0.023 ms |
| Generation lock wait | 0.003 ms |

This is recurring micro-starvation, not one three-second halt. The dominant
boundary appears 1,713 times: the GPU drains before the host submits the next
three-array gather-front command buffer. Those gaps account for 2.6123 s of
idle time, including 2.2957 s where the next commit itself was late. The next
largest repeated boundary follows verifier router/sort work 577 times and
accounts for 0.1998 s.

The scheduler also reports 36,625 cap/backpressure waits totaling 5.69 s.
That is overlapping host wait while the GPU queue is full, not GPU idle, and
must not be added to the 3.3003 s idle total. The trace therefore confirms both
sides of the earlier hypothesis: substantial CPU backpressure exists while
Metal is busy, but a separate per-forward control/graph-construction boundary
still starves Metal between model calls.

Removing every measured idle gap without changing GPU work would raise this
trace's ceiling only to 64.42 tok/s (`1024 / 15.8964`). Reaching 90 tok/s also
requires about a 28.4% reduction in GPU work after the scheduling gaps are
removed, through verifier/kernel work reduction and/or better accepted work per
round.

## Use it

Pull the exact artifact:

```bash
mtplx pull Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --revision 29ba90f82124961d0d902a9ea9bbb1034972af2f
```

Start the supported Flash-Next path with its model defaults:

```bash
mtplx serve \
  --model ~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --model-id mtplx-flash-next-optimized-speed
```

This resolves to Turbo, native MTP depth 3, batched verification, xhigh
reasoning, and the pack's temperature-1 sampler. The complete 29.8 GiB n-gram
table stays file-backed through mmap rather than consuming 29.8 GiB of
resident RAM. The production 1 GiB hot-row cache remains bounded on top. Full
table residency engages only on machines with at least 160 GiB RAM or with
`MTPLX_NGRAM_RESIDENT=1`. This branch leaves the 2.10 cache defaults unchanged.

## What was already in 2.10

The old stack was audited against the official `qwen4_exp` D3/M4 path before
restacking:

| Area | 2.10 / official artifact result | Restack decision |
|---|---|---|
| QSA/cache ownership | Pooled keys, selected-row attention, corrected row-major gather, and prefix trim/commit are already present | Do not duplicate |
| Attention projections | Official QKV fusion is already present; the old fixed-M2 quantized hyper kernels require storage this artifact does not have | Do not transplant |
| Generation scheduling | Old resident-loader and M2-only target-array changes do not reach D3/M4; queue-first async was previously measured slower | Reject |
| MoE | 2.10 already fuses gate/up in 96 modules; the old whole-MoE lane requires dense-BF16 router plus q8/g128 shared projections, while this pack is q8/g64 | Incompatible |
| GDN | Fused input projection is already installed on 36 layers; norm/scale/sigmoid epilogue is not fused | Re-derive for BF16/M4 first |
| Hyper boundary | No equivalent fuses the attention residual write with the following grouped RMSNorm | Re-derive for BF16/M4 second |

Nothing is transplanted by commit name or old geometry.

## Restack rule

Restack one optimization at a time. Validate construction-time invariants,
run focused parity checks, and benchmark the exact production workload against
the unchanged control. Keep and commit only matched wins. Rejected experiments
remain in the benchmark history or PR discussion, not in the enabled hot path.
