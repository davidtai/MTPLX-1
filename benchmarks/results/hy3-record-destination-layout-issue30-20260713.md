# Hy3 record-destination layout gate (#30)

## Decision

**No-go for integrating the v1 record arena or reopening the weighted Stage B prototype.** A contiguous destination is modestly faster than nine component rows for individual reads at host concurrency 1, but it does not supply the headroom required by the prior Stage B gate. Host concurrency 32 is inconclusive, and the adjacent 32-record lane fails the p95 gate.

The harness compares destination/layout efficiency only. Both arms issue the same `os.preadv` calls with `F_NOCACHE`, offsets, bytes, workload, and precomputed order. It does not claim syscall elimination, issue Metal reads, or prove runtime sibling-row fences.

## Paired result

Positive values mean lower wall time or latency for the record arena. Intervals are paired two-sided 95% Student-t intervals across four balanced AB/BA repeats.

| Lane | Wall-time improvement | p50 improvement | p95 improvement |
|---|---:|---:|---:|
| Individual, host concurrency 1 | **1.469%** [1.012, 1.926] | 1.338% [1.022, 1.654] | 1.952% [0.979, 2.926] |
| Individual, host concurrency 32 | 1.290% [-1.909, 4.489] | 2.879% [-1.878, 7.635] | 4.969% [-3.804, 13.742] |
| One synchronous 32-record adjacent `preadv` | **-1.974%** [-8.903, 4.956] | 0.216% [0.171, 0.262] | **-6.332%** [-27.089, 14.424] |

The adjacent lane's p95 values are its maximum of three batches per repeat. Its first record-arena repeat regressed 25.898%; the other three differed by less than 0.28%. That instability still fails the declared rule to reject the arena when p95 regresses. Host concurrency 32 also has no statistically supported wall-time improvement.

## Headroom closure

At host concurrency 1, the paired wall-time saving is 0.0137746 ms per record, with a 95% interval of 0.0094354-0.0181139 ms. Hy3 has at most 79 routed layers times eight selected experts, or 632 routed assignments per token. Extrapolating the mean to the deliberately impossible case where every assignment requires a physical read yields 8.7056 ms/token; the conservative lower confidence bound is 5.9632 ms/token. Real cache hits reduce both numbers.

The Stage B premise required the loader to supply at least 8.9830 ms/token. Even the mean all-miss extrapolation is short. Combining it with the deliberately impossible 7.1347 ms/token Stage B ceiling reaches 15.8403 ms/token versus the 16.1177 ms/token required for +5%, an implied best-case gain of 4.9097%. This combination assumes all 632 reads miss and the entire down projection plus reassembly tail become free, so it materially overstates any realizable path.

Therefore Stage A, Stage B, the device slot table, and runtime record-arena integration stop here. The unwired layout contract and benchmark remain useful evidence and do not allocate a duplicate runtime cache.

## Reproduction and provenance

- Harness commit: `9187e7b0cf9a2c9caa78ced8fc876fc9e4393ac9`; harness SHA-256 `867ba8cf6faa292e4ac7d122c40139137e9ae25b025d4734ae3900f7593ab398`; clean worktree; MLX 0.31.2.
- Four repeats, 96 measured and 32 warmup records per lane, host concurrencies 1 and 32, adjacent batch width 32, seed 30.
- Exact pinned 79x192 manifest, 10,616,832-byte records, and 161,036,107,776-byte sidecar. First/middle/last arena, component, and manifest record hashes matched. The full sidecar digest is declared by the pinned manifest but was not recomputed during this run.
- Raw payload: `benchmarks/raw/expert-io/issue30-record-destinations-20260713T084805Z-9187e7b0cf9a/result.json`, SHA-256 `c8f80ddffe67cd41fcb81ee3096bde9919040af4e259448089faa402f9374dc0`.
- Qwen was unloaded for the exclusive run, then restored; `/v1/models` returned `mtplx-qwen36-27b-optimized-speed`.

```bash
uv run --frozen --extra dev --extra server python \
  scripts/benchmark_hy3_record_destinations.py \
  "$MODEL" "$MODEL/expert-manifest-sidecar.json" \
  --operations 96 --warmup-operations 32 \
  --host-read-concurrencies 1,32 --batch-records 32 \
  --repeats 4 --seed 30 --output "$OUT"
```
