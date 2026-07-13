# Hy3 packed-stream floor result (#31)

## Decision

**No-go.** Four process-isolated capacity-102 repeats measured a weighted three-projection packed-layout speedup of **0.023%**, with a process-level 95% confidence interval of **-0.444% to +0.491%**. Because even the upper bound is far below issue #31's fixed +5% gate, the 144-byte packed layout stops before an exact packed-QMV kernel, codec, sidecar, or runtime integration.

This is a same-byte stream-floor result, not an exact-QMV or end-to-end throughput result.

## Contract

- Layout per four Q4/group-64 groups: 128 weight bytes, 8 BF16 scale bytes, and 8 BF16 bias bytes; 144 bytes total with no padding.
- Row strides: 2,304 bytes for gate/up and 864 bytes for down, identical to split storage.
- Both kernels use the same geometry and XOR-only load-retention work; only split versus packed addressing differs.
- Capacity 102 keeps 360,972,288 bytes per projection representation, beyond a small cache-resident probe.
- Each process uses 25 warmups and 150 measured samples, with 16 independent dispatches per `mx.eval`; no flush is subtracted.
- The decision uses four independent process summaries. Promotion requires a lower 95% confidence bound of at least +5% and no projection p95 regression above 2%.
- Split and packed checksums match for gate/up and down in all processes, including repeated and high slots `[101, 1, 101, 0, 100, 2, 3, 4]`.

## Results

| Geometry | Split mean process median | Packed mean process median | Packed speedup | 95% CI | Worst process p95 regression |
|---|---:|---:|---:|---:|---:|
| gate/up | 0.055071 ms | 0.055173 ms | -0.183% | [-0.785%, +0.419%] | +0.385% |
| down | 0.056230 ms | 0.055989 ms | +0.430% | [-0.055%, +0.914%] | -0.017% |
| weighted `2*gate/up + down` | 0.166372 ms | 0.166334 ms | **+0.023%** | **[-0.444%, +0.491%]** | +0.385% |

The one-time MLX packing median was 0.0798 ms/expert for gate/up and 0.0763 ms/expert for down. It is excluded from the timed stream floor and does not affect the no-go.

## Provenance

- Exact clean benchmark commit: `a55d5399b91d1f1086b275abc3177163c717c43a`
- Harness SHA-256: `a676f8f60341319ad458f2750c02d1395b483d8f290a2bffafb9e37b35bac357`
- Raw result: `benchmarks/raw/artifact-speculative/issue31-packed-stream-20260713T093116Z-a55d539/result.json`
- Raw SHA-256: `a33b40df2bd63c31989af5126e57e4f9126baebd44c2c266f7b76a05e8ca5078`
- MLX 0.31.2, Python 3.12.13, macOS 26.5.2 arm64
- Qwen was stopped only for the timing window, then restored and verified through `/v1/models` as `mtplx-qwen36-27b-optimized-speed`.

