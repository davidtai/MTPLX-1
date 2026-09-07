# Qwen3.8 Flash-Next long-context decode memory

`mtplx serve` OOM'd on the pack's advertised 262,144-token context on a 128 GB
M5 Max under turbo / server defaults. The full source + log diagnosis is in
`.benchmark-artifacts/over100-reports/oom255k/report.md`; this note records the
optimization it produced.

## Optimization — head-chunk the small-q_len verify SDPA

### Problem

At 261,120 tokens the **prefill completes** (TTFT ~240 s) and the request OOMs a
few decode tokens later with a Metal command-buffer execution failure
(`[METAL] ... kIOGPUCommandBufferCallbackErrorOutOfMemory`). Every arm — release
2.11.2, mask-fuse, score-tiler, `MTPLX_MLX_CACHE_LIMIT=2GiB`, session-off — hits
the identical failure at a ~100.8 GB sampled peak, so it is neither the prefill,
the allocator pool, nor the session bank.

The cause is the **speculative-verify decode step**. Of the 48 layers, 12 are
full-attention (`full_attention_interval: 4`) and keep dense KV over the whole
context; the model is GQA with `num_attention_heads: 24`, `num_key_value_heads:
2` (GQA factor 12). A multi-row verify has `q_len` = 3–8. MLX's fused
vector-attention kernel serves at most `q_len * GQA ≤ 32` rows per dispatch;
`q_len 5 × GQA 12 = 60 > 32`, so SDPA falls to an **unfused path that
materializes the `[24, q_len, T]` fp32 score plane (O(T)) and GQA-expands k/v**.
Across the 12 dense layers held in the fixed-M4 verify command buffer, that
transient — several GB at T = 261,120, and linear in T — lands on top of a
~93.8 GiB steady resident (weights 77.3 GiB + full KV 6.0 GiB + QSA aux + graph
buffers), tipping the single command buffer past the 100 GiB Metal wired limit.
At 131,072 tokens the same terms are ~half and the request fits; at 261,120 it
does not. The QSA layers avoid this via their bounded rows-gather selection; the
12 dense layers cannot, and the fixed-M4 verify path additionally gates off the
score-tiler and rows-gather (`not fixed_capacity`), which is why every prior
lever was a no-op.

### Change

`_verify_sdpa` (`mtplx/models/qwen4_exp.py`) replaces the dense-fallback SDPA
call in `Attention.__call__`. When `q_len * GQA > 32` and `q_len ≤ 32` (the
verify / small-tail band MLX will not route to its flash kernel), it splits the
**query heads** into chunks small enough that each fused call satisfies
`q_len * heads_per_chunk ≤ 32` (e.g. q_len 6 → chunks of 5 heads, q_len 4 → 8
heads), pairs each chunk with its **own single kv head** (a slice, so k/v is
**never GQA-expanded**), calls `mx.fast.scaled_dot_product_attention` per chunk,
and concatenates along the head axis. Every dispatch stays on the bounded fused
kernel, so no `[24, q_len, T]` plane is formed and the working set is one chunk
regardless of T. Wide prefill chunks (`q_len > 32`) are untouched and keep their
single flash call.

### Effect

The O(T) verify transient in the 12 dense layers is removed; the 261K decode
working set no longer scales with context, so it no longer tips past the wired
limit. Applies uniformly to the fixed-M4 verify, the ordinary verify, and any
small prefill-tail chunk (all reach the one dense-fallback SDPA). Decode `q_len
1` (12 ≤ 32) and prefill wide chunks are unaffected.

### Exactness

The heads in one chunk all belong to the same kv head, so each chunk's SDPA is
exactly the full GQA attention's arithmetic for those heads on the same fused
kernel — chunked-fused is **bit-identical to an unchunked fused call** (verified
< 1e-6 on fp32 random tensors against a GQA-expanded MHA reference). Versus the
unfused path it replaces, it is rounding-class (fused vs unfused accumulation),
the same class as the mask-fuse lane.

### Files

- `mtplx/models/qwen4_exp.py` — `_verify_sdpa`, `_sdpa_head_chunked`,
  `_verify_sdpa_head_chunk_plan`, `_verify_sdpa_head_chunk_enabled` (env at
  use), engagement latch + `verify_sdpa_head_chunk_report`; call site in
  `Attention.__call__`.
- `mtplx/server/openai.py` — `_qwen4_install_reports` surfaces the engagement
  receipt at `/health` under `qwen4_install_reports.verify_sdpa_head_chunk`.
- `tests/test_qwen4_verify_sdpa_head_chunk.py` — CPU tests.

### Switch / observability

- `MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK` — default ON (engages only in the
  `q_len × GQA > 32`, `q_len ≤ 32` band); `0`/`off` opts out. Resolved at use.
- Log line on first engagement: `[mtplx] verify SDPA head-chunked: q_len=S
  heads_per_chunk=H chunks=N`.
- `/health` → `qwen4_install_reports.verify_sdpa_head_chunk`:
  `{engaged, q_len, heads_per_chunk, chunks, n_kv_heads}` once it has fired.
