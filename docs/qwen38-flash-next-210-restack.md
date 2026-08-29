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

The n-gram table is a 29.8 GiB file-backed mmap in every arm. `hot 1 GiB`
adds the MTPLX 2.10 default `OrderedDict` row cache on top; `mmap only` does
not add a second cache. The A/B/A runs used the same prompt, seed, output,
acceptance trajectory, and 432 verify calls. Repair time was zero throughout.

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
table stays file-backed through mmap. This branch leaves the 2.10 cache
defaults unchanged.

## What was already in 2.10

The old stack was audited against the official `qwen4_exp` D3/M4 path before
restacking:

- already present: QSA pooled-cache ownership, selected-row QSA attention,
  corrected row-major gather geometry, official QKV fusion, fused GDN input
  projection, and family-owned verifier capture/commit;
- obsolete for this artifact: the resident-loader generation changes,
  fixed-M2 whole-MoE kernels with incompatible router/shared-projection
  storage, fixed-M2 quantized-hyper kernels, and the rejected async draft
  ticket experiment;
- worth re-deriving for official BF16/M4: the GDN norm/gate epilogue fusion
  and the verifier hyper-boundary residual-plus-RMSNorm fusion.

Nothing is transplanted by commit name or old geometry.

## Restack rule

Restack one optimization at a time. Validate construction-time invariants,
run focused parity checks, and benchmark the exact production workload against
the unchanged control. Keep and commit only matched wins. Rejected experiments
remain in the benchmark history or PR discussion, not in the enabled hot path.
