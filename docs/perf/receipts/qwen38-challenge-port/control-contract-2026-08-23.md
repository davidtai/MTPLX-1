# Qwen 3.8 Challenge Port Control Contract

Status: static contract captured; real-model timing deferred until the shared
GPU lock is free.

## Source and environment

- MTPLX control: `bd4421567f9e16ce957c6ef97708b072dcd73937`
  (`youssofal/MTPLX` v2.9.1).
- Challenge source: `eb5eadc7a165047d4321ce883b9ff30894d8bd19`.
- Python: 3.12.13.
- MLX: 0.32.0.
- mlx-lm: 0.31.3.
- Host: Apple M5 Max, 128 GiB, macOS 26.5.2 (25F84).
- GPU ownership at capture: `/tmp/mtplx-gpu-exclusive.lock` was held by PID
  56537 for an unrelated DeepSeek V4 benchmark. No model load or GPU timing was
  attempted.

The benchmark runner must use a clean control worktree at the control commit or
record every dirty path. The feature worktree was dirty only with the approved
plan, design, inventory, and receipt files while this contract was captured.

## Model artifact

- Repository: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`.
- Cached Hub revision: `123db8bcc7101455b00d9aad36c0e760c6e7de02`.
- Local path:
  `/Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`.
- `config.json` SHA-256:
  `533e833dedb9e7b6a8ee22ab4f2fc034bcf6ded9d8693e5ebcc9d5f159b62a3b`.
- `mtplx_runtime.json` SHA-256:
  `3bac7e807762841d1fa4c32680e9c1f2f7b633ddc88c321036356803b8c7d1a8`.
- MTP weights: `mtp.safetensors`, 849,400,403 bytes, SHA-256
  `4468f39621de68a19ffd0bcb2e2e2f352205def7436a625b3427e3752866c287`.

Text topology: 64 layers, hidden width 5120, MLP width 17408, 24 query heads,
4 KV heads, head dimension 256, vocabulary 248320, and full attention every
fourth layer. Linear-attention topology is 16 key heads and 48 value heads with
128-wide keys and values.

The trunk default is affine 4-bit/group-32. The exact 8-bit/group-64 affine
overrides are `embed_tokens`, `lm_head`, every linear-attention `out_proj`
(layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,
32,33,34,36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62),
and MLP gate/up/down projections in layers 56 through 63. The configuration
digest above is authoritative for the full map.

The native MTP sidecar is BF16: `fc` is 5120x10240; Q/K/V projections are
12288x5120, 1024x5120, and 1024x5120; output projection is 5120x6144; MLP
gate/up/down are 17408x5120, 17408x5120, and 5120x17408; normalization vectors
are 5120 except Q/K norm at 256. The runtime contract is `post_norm`, concat
order `embedding_hidden`, local MTP positions, and maximum/default depth 3.

## Runtime route

- Profile: `turbo` / `native_mtp_turbo`.
- Draft depth: 3.
- Draft sampler: temperature 1.0, top-p 0.95, top-k 20.
- MTP history: `committed`; last-window fallback is not active.
- Target verification: NAX enabled with M4 route `vk_k`; compiled verify
  enabled through context 32768.
- Packed GQA attention: enabled from context 8192.
- Warmup ladder: 512,2560.
- Long-context MTP depth policy: off.

Candidate routes must preserve the exact control behavior or identify their
route in the output receipt. No candidate may silently fall back on a shape,
packing, dtype, head, or cache-contract miss.

## Matched measurement contract

- Prompt suite: `mtplx/benchmarks/prompts/flappy.jsonl` at SHA-256
  `d9f32acb3d56cc645d58c26e1c3a0bc799d1d8e2f2f160bb8730ad73fcc4bd7a`
  and `mtplx/benchmarks/prompts/python_modules_long.jsonl` at SHA-256
  `ca2054913c5c27c24c983ed27e3ee4eff1d01d456a73e71377fdaea3cbf8c140`.
- Seed: 42 for every matched arm.
- Prompt length: native encoded length for the first short gate; fixed context
  rungs are introduced only as named, token-identical follow-up cases.
- Decode gates: one verify cycle, then 64 tokens, then 512 tokens.
- Warmup: construct and self-check each route, then one unmeasured generation
  per prompt/route before its first timed arm.
- Primary timing boundary: after tokenization and prompt-state construction,
  immediately before the first decode cycle through completion of the final
  emitted token. Model load, tokenizer load, receipt serialization, and the
  unmeasured warmup are excluded. Prompt-history construction is reported
  separately so cache-append work cannot be hidden.
- Ordering: at least ABBA and BAAB with identical process lifecycle and thermal
  gate; reject at the first exactness failure or matched material regression.
- Required raw output: per-prompt wall times, output-token hashes, acceptance
  by depth, attempted depth schedule, route/kernel/compile counters, peak
  memory, thermal observations, environment, source revisions, and lock owner.

This document defines the control. It contains no performance result.
