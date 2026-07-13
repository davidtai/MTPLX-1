# Hy3 weighted Stage B premise gate (#30)

## Decision

**No-go for implementing Stage B as the next standalone optimization.** The resource-safe #29 layer control averages 2.9545 tok/s, or 338.472 ms/token. Reaching the issue's +5% promotion gate requires reducing that to 322.354 ms/token: a 16.118 ms/token saving.

Prior same-machine kernel evidence already provides a harder upper bound than a new microbenchmark could. The measured stock down projection costs about 0.0715 ms/layer, or 5.6485 ms across 79 routed layers. The measured component-bank reassembly tail costs 1.4862 ms/token. Even an impossible candidate that makes the complete down projection free and deletes the entire tail can therefore save at most 7.1347 ms/token.

| Quantity | ms/token |
|---|---:|
| Saving required for +5% | 16.1177 |
| Entire measured down projection | 5.6485 |
| Entire measured reassembly tail | 1.4862 |
| Deliberately impossible Stage B ceiling | **7.1347** |
| Unexplained remainder | **8.9830** |

That ceiling covers only 44.27% of the required saving. Applied literally, it would raise the safe control to about 3.0181 tok/s, or +2.153%, and it overstates what a real kernel can do because Stage B must still read all down weights and perform the affine-Q4 dot products.

The evidence-based plausible ceiling is lower. The prior spike measured about 1.4 ms/token for all three specialized QMVs and 1.4-1.5 ms/token for pure tail elision. Adding those already overlapping estimates generously yields 2.9 ms/token, leaving about 13.2 ms/token that Stage B cannot explain.

## Evidence chain

- Safe control: commit `058b40b28de97ddce23184e2c62ed1f41cb6ba2d`, two exact-parity repeats at 2.9332 and 2.9757 tok/s. Curated payload: `benchmarks/results/hy3-cache-scheduling-issue29-20260713.{md,json}`.
- Kernel spike: commit `2671ba78c3e80e18d412787ffaa343d971ff7b1f`, `docs/METAL_KERNEL_SPIKE.md`. Stock `mx.gather_qmm` reached 95.6% of the single-dispatch memory roofline; specialized QMV saved about 4.5-5% GPU-side, roughly 1.4 ms/token across all three projections.
- Tail payload: `benchmarks/results/expert-exec-fusion-v4.json` at the same spike commit. Component banks plus tail cost 27.6812836892 ms/token; the all-hit component fast path cost 26.1950963686 ms/token. Difference: 1.4861873206 ms/token.
- Down timing: the spike's isolated production-shape down QMV measured about 0.0715 ms/layer. Multiplying the complete stock stage by 79 gives the deliberately generous 5.6485 ms/token ceiling used above.

The old spike worktree was verified clean at `2671ba7` before extracting these values. No result is inferred from the invalid no-chat #29 launch.

## Next gate

Measure loader destination and resource-safe scheduling before writing a weighted kernel. Keep the current one-`preadv` scatter arm and a contiguous v1 destination arm identical in sidecar offsets, bytes, record sequence, `F_NOCACHE`, and queue depth.

- If loader/scheduling cannot conservatively save at least 8.983 ms/token, even the impossible Stage B ceiling cannot combine with it to reach +5%; stop the execution experiment.
- A realistic Stage B case still needs another mechanism to supply roughly 13.2 ms/token.
- Reopen a capped Stage B probe only if the independent loader result supplies that missing headroom. It must still pass raw-bit parity, device-visible assignment/slot/generation guards, and two independent paired runs.

This decision does not claim that a weighted reduction kernel is impossible. It says the kernel is not the next rational investment under the issue's declared end-to-end gate.
