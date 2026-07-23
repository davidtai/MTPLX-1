---
license: apache-2.0
license_link: https://huggingface.co/tencent/Hy3/blob/main/LICENSE
base_model: tencent/Hy3
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mtplx
- moe
- quantized
- imatrix
- 2-bit
- streaming
- hunyuan
---

# Hy3 oQ2e MTPLX-streaming

2-bit Hunyuan 3.0 (295B-A21B MoE) for Apple Silicon, packaged for
SSD-streamed serving with [MTPLX](https://github.com/davidtai/MTPLX).
Weights stream from SSD under a configurable memory envelope, so the model
runs on Macs that cannot hold it resident. Measured envelopes: 64, 80, and
88 GiB of weight budget on a 128 GB M5 Max.

Quantization: omlx oQe level-2 imatrix experts (2.44 bpw, 2.50 bpw
effective with q8 residents), from
[mlx-community/Hy3-oQ2e](https://huggingface.co/mlx-community/Hy3-oQ2e).

## Repo contents (~91 GiB)

| files | size | what |
|---|---|---|
| `model-resident-*.safetensors` (2) + `model.safetensors.index.json` | 8.7 GiB | attention, routers, shared experts, embeddings, dense layer 0 (q8-gs64); the index covers exactly these resident tensors |
| `experts.bin` + `expert-manifest.json` | 75 GiB | all 15,168 routed expert records (2-bit gs128) as an aligned, hash-pinned bank; the manifest is sidecar-authoritative (references resident shards + the bank only) |
| `expert-manifest.shards.json` | — | provenance manifest pinning sha256 for the pre-repack safetensors shards and every expert record |
| `mtp/layer80-bf16.safetensors`, `mtp/layer80-residents-q.safetensors` | 7.1 GiB | Hy3 MTP head (speculative decoding) |
| `island-placement.json`, `route-census.json` | — | per-machine seeds; MTPLX regenerates them on first serve |

This is the MTPLX streaming layout: routed experts ship only in
`experts.bin`, so nothing is downloaded twice. Download everything:

```bash
hf download <REPO_ID>
```

**Plain-MLX resident serving (128 GB Mac):** use the upstream safetensors
checkpoint at
[mlx-community/Hy3-oQ2e](https://huggingface.co/mlx-community/Hy3-oQ2e);
this repo packages the same bytes for streaming.

**llama.cpp:** not served. MLX affine quantization does not convert to
GGML blocks in either direction.

## Measured decode (M5 Max, 614 GB/s)

Real long-code prefill (exact-token gated), greedy, decode 256, bf16 KV,
3 reps. Cell = decode tok/s, mode of record = faster of AR / K1 (MTP
speculative depth 1).

| weight envelope | 1024 ctx | 16k | 64k |
|---|---|---|---|
| 88 GiB (full residency) | **48.0** K1 | **27.6** K1 | — (KV exceeds the 112 GiB budget) |
| 80 GiB | **21.6** K1 | **15.9** AR | **9.2** AR |
| 64 GiB | **11.1** K1 | **7.0** AR | **4.0** AR |

Short-prompt arm (320-token code prompt): K1 47.7 / 22.3 / 10.8 on
88/80/64 — K1 wins every envelope there.

Two regimes govern the mode choice. K1 wins at short context on every
envelope, and at every context under full residency. AR wins at 16k+ on
streaming envelopes: speculative verify multiplies expert-miss servicing
under partial residency. Acceptance is 0.89–0.90 on code in every cell,
so the crossover is a decode-cost effect, not an acceptance effect.

## Quality (serving config: rq4 projections, bf16 KV)

| eval | score |
|---|---|
| MBPP full-974 | 0.8004 (identical to the q8-resident reference) |
| HumanEval-164 | 0.8659 (q8 reference 0.8720, McNemar exact p = 1.0) |

## Serving notes

- Presets `hy3-oq2e-rq4-{48..96}` ship in the MTPLX repo (main branch).
- **bf16 KV is required for MTP.** kv4 was measured and reversed:
  acceptance collapses 0.898 → 0.125 and every MTP lane loses to AR.
- Recommended: K1 for short context, AR for 16k+ on streaming envelopes.
- Residents are pre-quantized q8-gs64 and load via the config-driven
  path; never requantize them at load.

## Provenance and integrity

- Base model [tencent/Hy3](https://huggingface.co/tencent/Hy3)
  (Apache-2.0); quantized bank mlx-community/Hy3-oQ2e rev `1979c306`.
- Re-sharded here into resident/expert groups for selective download.
  Tensor bytes are unchanged; every tensor was sha256-verified at repack.
- `expert-manifest.json` (sidecar-authoritative, what the engine loads)
  pins the resident shards and `experts.bin`; it is what makes the
  expert-shard-free selective download servable.
- `expert-manifest.shards.json` pins sha256 for all 18 safetensors
  shards, all 15,168 expert records, and `experts.bin`.
- Two metadata corrections vs the published checkpoint (originals kept as
  `*.orig-published`): index `total_size` (overstated upstream by
  345,252 B) and config `num_nextn_predict_layers` 0→1, which the bundled
  MTP head requires.

## Not measured

32k context. 48/32 GiB envelopes (presets exist, never benchmarked).
kv4 beyond the single 88 GiB × 16k cell. Prose-heavy workloads: MTP
acceptance is content-dependent (0.52 measured on prose vs 0.89–0.90 on
code); the speed table is a code-workload table.
