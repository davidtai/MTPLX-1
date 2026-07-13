# Hy3 artifact and speculative experiment map (#31)

## Decision

**Partial no-go; issue #31 remains open.** The capacity-102 packed-stream probe rejects the same-byte v2 layout premise. Existing evidence also rejects the exact measured 75-to-100-slot reinvestment and the current MTP speed configuration. Trace-local route recall is insufficient to justify runtime promotion, but does not establish that all hint-only prediction fails. The lower-bit arm has arithmetic capacity upside but no artifact identity or quality method, so any speed number today would be incomplete.

No sub-5% mechanisms are combined into a claimed win. No v2 sidecar, runtime prefetch API, MTP route adapter, or lower-bit artifact is added.

## Track map

| Track | Best applicable evidence | Decision | Repository action |
|---|---|---|---|
| GPU-oriented v2 layout | The capacity-102 packed stream floor measured +0.023% weighted speedup with a process-level 95% CI of [-0.444%, +0.491%]. Gate/up was -0.183%; down was +0.430%. | **No-go for the 144-byte same-byte packed layout.** Even the upper CI is far below +5%. | Stop before an exact packed QMV, codec, 161 GB sidecar, or runtime integration. |
| Q8 KV budget exchange | The repaired 75-to-100-slot experiment cut expert bytes 27.03%, but decode fell 4.7052 to 0.9896 tok/s (-78.97%), rolling p95 rose 351.5%, and peak MLX grew 20.97 GB with only 376.9 MB headroom. | **No-go for the exact measured full 75-to-100-slot arm.** Fixed-slot BF16-versus-Q8 and other separately designed exchanges remain open. | Do not reintroduce the rejected 100-slot plan. Require a true long-context attention/quality method for fixed-slot Q8. |
| Hint-only prediction | Prior-token same-trace recall is 29.51%. Prompt-trained top-16 reaches 52.72% at 2x nominal requests; top-32 reaches 67.91% at 4x. Physical byte amplification is unmeasured. Q4 MTP acceptance is 20.79%, and decode is 3.8072 versus 6.0481 tok/s AR (-37.05%). | **Current MTP speed no-go; offline predictors deferred.** Same-trace recall is insufficient for runtime promotion. | Do not add a runtime API until an offline held-out candidate passes physical byte amplification, cache, interference, lead-time, and net-latency gates. Router authority remains unchanged. |
| Q3/Q2 cold tier | Group-64 arithmetic implies 8,257,536 bytes for Q3 (-22.22%) and 5,898,240 for Q2 (-44.44%) versus Q4, but no Hy3 lower-bit artifact or quality report exists. | **Deferred.** Byte arithmetic is not a quality or performance result. | Require pinned BF16 conversion, explicit artifact identity, native-kernel fail-closed behavior, and declared quality gates before a pilot. |

## Evidence chain

### Track 1: layout

The kernel spike at commit `2671ba78c3e80e18d412787ffaa343d971ff7b1f` measured top-8 stock at 0.1977 ms/layer and a zero-arithmetic same-byte floor at 0.1871 ms/layer. It also measured 482.6 GB/s for stock against a 504.6 GB/s single-dispatch roofline. The only unmeasured physical-layout idea is four-group 144-byte weight/scale/bias packing, which keeps the exact byte count while reducing projection streams from nine to three (`git show 2671ba7:docs/METAL_KERNEL_SPIKE.md`, sections 1 and S7).

The roofline ratio `504.6 / 482.6 - 1 = 4.5586%` is useful context, but it was not a hard ceiling: the source reports about 20% run-to-run bandwidth variance and no confidence interval for that cross-run ratio. The directly comparable historical stream-floor values left `0.1977 / 0.1871 - 1 = 5.665%` isolated headroom. Their direct delta is `0.0106 ms/layer * 79 = 0.8374 ms/token`, or about 0.247% of the resource-safe 338.4721 ms/token #30 baseline. That justified one bounded probe, not a full sidecar.

The clean-commit capacity-102 probe at `a55d539` compared two checksum-equivalent Metal stream floors whose only difference was split versus 144-byte packed addressing. It used four isolated processes, 25 warmups and 150 samples per process, 16 dispatches per `mx.eval`, and no flush subtraction. The weighted `2 * gate/up + down` result was 0.166372 ms split versus 0.166334 ms packed: +0.0231%, 95% CI [-0.4443%, +0.4906%]. Gate/up was -0.1831% and down was +0.4296%; all parity checks matched. The upper weighted bound cannot reach the fixed +5% gate, so this layout is closed before exact-QMV work. Full evidence is in `hy3-packed-stream-floor-issue31-20260713.{md,json}`.

### Track 2: KV exchange

`docs/MOE_RUNTIME_PR_BENCHMARKS.md` records the repaired PR #12 experiment. Both arms used the same physical 21,831,680,000-byte Q8 cache so that only the accounting-driven 75-to-100 expert-slot change differed. The larger plan improved hit rate and cut reads, but serialized route/read/compute work: an instrumented lane fell from 4.7846 to 0.7272 tok/s while SSD throughput fell from 6.670 to 0.913 GiB/s. This is a measured rejection of the exact full 75-to-100-slot reinvestment on this host and configuration, not every possible budget exchange and not a Q8-versus-BF16 quality comparison.

Q8 at fixed expert capacity remains an open, separately scoped representation/quality question. The existing workload reserved 128K capacity but did not fill a true 128K context, and both arms were physically Q8, so those results cannot be presented as Q8 quality evidence. Q4 KV remains out of scope until Q8 has an accepted quality method.

### Track 3: route hints

The checked route analysis at commit `cd1c5b4` (`benchmarks/results/hy3-q4-route-analysis-1k-64.json`) reports 39,936 assignment requests: top-8 recall 37.35%, top-16 recall 52.72% at 2x nominal request amplification, and top-32 recall 67.91% at 4x. Previous-token recall is 29.51%. These are same-trace recall values. The multipliers are derived from requested IDs versus top-8, not measured physical I/O after cache hits, deduplication, cancellation, or contention. They are insufficient to justify a runtime API but do not prove that every predictor fails.

The MTP gate at evidence commit `a2a1cb6` is directly measured: 296 accepted of 1,424 Q4 drafts (20.7865%), 3.8072 decode tok/s versus 6.0481 AR. MTP therefore supplies neither a low-cost predictor nor a current speed path in the measured configuration. The evidence does not cover held-out prompts, a zero-prediction scheduling control, physical cache amplification, interference, lead time, or net latency, so the general offline experiment remains open. Any future predictor remains hint-only; it cannot select experts, skip authoritative misses, or alter generated tokens.

### Track 4: lower precision

Local MLX 0.31.2 advertises affine kernels for 2, 3, 4, 5, 6, and 8 bits, but the deployed Hy3 manifest and record-native contracts are intentionally Q4-only. The Q3/Q2 sizes above come from the checked group-64 byte formula, not an artifact or benchmark. No perplexity, deterministic continuation, reasoning, tool-use, or long-generation evidence exists for a mixed tier.

If this track is reopened, the smallest defensible pilot is Q3, not Q2: quantize one routed layer from the pinned BF16 source; keep a training-trace hot set at Q4; evaluate held-out routes and activations; report physical bytes, conversion cost, p50/p95 layer latency, and a predeclared weighted-output error envelope. Passing that pilot would authorize a full separately branded artifact/quality experiment, not production promotion.

## Worktree and machine hygiene

- The old `codex/moe-optimization-research` worktree remains untouched at `8eba442`; its six modified files and four untracked research artifacts were inspected read-only.
- This decision branch starts from #30 commit `dfbefe965ff0cd256247327e41fd5b555ad245f6`.
- Qwen was stopped only for the packed-stream timing window. The benchmark completed and wrote its raw artifact; the outer zsh cleanup handler then hit a read-only-variable error, so Qwen was immediately restored explicitly and `/v1/models` verified `mtplx-qwen36-27b-optimized-speed` before work continued.
