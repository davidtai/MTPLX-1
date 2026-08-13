# DeepSeek-V4 Flash 0731 DSpark receipt

This is the scrubbed, tracked performance receipt for the construction-bound
DeepSeek-V4 Flash 0731 DSpark K2 lane. Raw generation artifacts remain local;
their hashes are listed below without model paths, generated text, service
details, or machine-local process data.

## Fixed conditions

- Machine: Apple M5 Max MacBook Pro, 128 GB, macOS 26.5.2.
- Runtime: Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3.
- Model: `mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed` at source revision
  `10001e0065f8394e03e968e652cbbe7cd2ca122c`.
- Model identity: config SHA-256
  `44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f`;
  index SHA-256
  `f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861`.
- Sampler: greedy, `temperature=0`, `top_p=1`, `top_k=0`, seed 0.
- Prompt: `Explain why speculative decoding can preserve greedy output.`
  through the model chat template, 14 prompt tokens.
- Output: forced 128-token budget, two identical cases in one model load. The
  second case is the warmed comparison; the first exposes one-time compilation.
- PR lane: explicit `deepseek_v4_0731_k2=True`, fixed proposal width K2,
  persistent cache, cycle history, batched native verification, stock verify and
  draft cores.
- Benchmarked commit: `8a57b9adb4030d1334ee5440e160a06c55555643`.

## Current PR bracket

Memory growth is measured from the post-load active-memory baseline of
86.4561 GiB. Peak memory is the process-wide MLX peak and therefore includes
load-time allocation; it is identical across these in-load arms.

| case | depth | target prefill tok/s | decode tok/s | end-to-end tok/s | active GiB | growth MiB | peak GiB | accepted / drafted | exact vs K0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cold compile | K0 | 0.191 | 25.264 | 1.630 | 86.4561 | 0.0180 | 139.7061 | - | reference |
| cold compile | K2 | 103.592 | 28.954 | 28.085 | 86.4561 | 0.0190 | 139.7061 | 68 / 119 | yes |
| warmed | K0 | 103.584 | **32.358** | **31.289** | 86.4561 | 0.0180 | 139.7061 | - | reference |
| warmed | K2 | 103.315 | 29.240 | 28.356 | 86.4561 | 0.0190 | 139.7061 | 68 / 119 | yes |

The warmed K2 lane is exact in both cases, but it does **not** beat warmed AR:
29.240 versus 32.358 decode tok/s, a 9.6% loss. First- and second-position
acceptance were 70.0% and 44.1%. The cold K0 prefill result is compilation time,
not model prefill throughput, so it is disclosed rather than used as a speedup
claim.

## Historical K-depth diagnostic

Before the K2-only construction contract was pinned, the native DSpark harness
ran one simple 9-prompt-token, 64-output-token K0-K3 sweep on MLX 0.31.2. It did
not record prefill TPS or memory growth, so those fields are unavailable. Active
and peak memory remain useful as measured.

| depth | decode tok/s | end-to-end tok/s | active GiB | peak GiB | accepted / drafted | exact vs K0 |
|---|---:|---:|---:|---:|---:|---|
| K0 | **24.565** | **23.312** | 86.4561 | 86.5079 | - | reference |
| K1 | 19.193 | 18.584 | 86.4561 | 86.5175 | 27 / 36 | yes |
| K2 | 19.640 | 19.010 | 86.4561 | 86.5240 | 34 / 58 | yes |
| K3 | 21.413 | 20.609 | 86.4561 | 86.5392 | 37 / 76 | **no** |

This older chart is diagnostic, not a promotion result: K1 and K2 were exact but
slower than AR, while K3 was faster than K1/K2 but diverged from greedy AR. The
current public lane therefore stays construction-pinned to K2 rather than
silently widening to an unqualified K1/K3 route.

## Raw-artifact manifest

| local artifact | SHA-256 |
|---|---|
| `0731-pr-optimized-k2-128-20260812.json` | `e3e8ab454a5a6860578eb022e85297de9143b5bd5588229bb795e472ba5395c2` |
| `0731-dspark-width123-64tok-20260809.json` | `1f60e529e4c172642fa461c41f5cd5dd11f28048c571f875cff04ee73cae9a3f` |

Profiler dispatch censuses and physical-M3 diagnostics are not used as TPS
proof here. They are discovery evidence only and remain separate from these
uninstrumented generation timings.
