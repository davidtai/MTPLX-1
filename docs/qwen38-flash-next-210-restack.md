# Qwen 3.8 Flash-Next optimization restack

This branch restarts the Flash-Next optimization campaign from the untouched
MTPLX 2.10 release instead of carrying the historical PR #368 stack forward.
Every runtime change must beat the unchanged 2.10 control on the exact
production workload before it remains in the branch.

## Pinned baseline

- Runtime: MTPLX 2.10.0, commit `4ce96908cd2e5ebcb5c66ee2541d12fdf5d0423f`
- Model: `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`
- Model revision: `29ba90f82124961d0d902a9ea9bbb1034972af2f`
- MLX: 0.32.2
- Mode: native MTP, depth 1, stock batched verifier with the family
  capture-commit lane
- Workload: exactly 16,384 prompt tokens and 1,024 generated tokens from the
  naturalistic Python generation-module task
- Thinking: enabled, reasoning effort `low`
- Target and draft sampler: temperature 1.0, top-p 0.95, top-k 20, min-p 0,
  presence penalty 0, repetition penalty 1
- Seed: 42
- N-gram hot-row cache: 1 GiB (`MTPLX_NGRAM_HOT_MB=1024`)
- Measurement: one full warmup followed by one measured request, with the
  machine-wide GPU lock held for both

The two measured responses have the same prompt digest, output digest,
acceptance trajectory, verify count, and zero repair time.

| Run | Decode tok/s | End-to-end wall | Accepted / drafted | Verify calls | Repair |
|---|---:|---:|---:|---:|---:|
| 2.10 control r1 | 59.4323 | 31.1804 s | 448 / 545 | 555 | 0 s |
| 2.10 control r2 | 59.6381 | 30.6170 s | 448 / 545 | 555 | 0 s |
| Mean | **59.5352** | **30.8987 s** | **448 / 545** | **555** | **0 s** |

Temperature-zero short-prompt results are vanity measurements only and must
not be used to accept an optimization.

## Run the stock service

Pull the exact artifact:

```bash
mtplx pull Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --revision 29ba90f82124961d0d902a9ea9bbb1034972af2f
```

Start it with the supported 2.10 Flash-Next path and the default 1 GiB hot-row
cache:

```bash
MTPLX_NGRAM_HOT_MB=1024 mtplx serve \
  --model ~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --model-id mtplx-flash-next-optimized-speed \
  --profile turbo \
  --mtp --generation-mode mtp --depth 3 \
  --verify-strategy batched \
  --reasoning-effort low
```

The service uses the pack's recommended sampler (temperature 1.0, top-p 0.95,
top-k 20). The optimization benchmark overrides request depth to 1 so every
candidate is compared with the same physical two-row verifier shape.

## Restack rule

Restack one optimization at a time. Validate its construction-time invariants,
run focused parity checks, and then run the pinned workload against the
unchanged control. Keep and commit only improvements with matching workload
and correctness receipts. Rejected experiments remain in the benchmark table
or PR discussion, not in the enabled runtime path.
