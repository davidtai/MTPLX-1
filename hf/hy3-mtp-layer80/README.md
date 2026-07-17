---
base_model: tencent/Hy3
tags:
- mlx
- mtplx
- mtp
- speculative-decoding
---
# hy3-mtp-layer80 (mtplx MTP layer artifact)

Multi-token-prediction (MTP) layer weights extracted from the upstream
checkpoint so the mtplx runtime can attach speculative decoding to the
expert-streaming models without downloading the full source checkpoint.

## Provenance

- Extracted from the pinned upstream checkpoint (see the mtplx extraction scripts); no artifact manifest is present yet

## Contents

- Architecture: `HYV3ForCausalLM`
- MTP layer tensors as safetensors (see file list below)

## Files

| file | size | role |
| --- | ---: | --- |
| `config.json` | 1195 B | metadata |
| `layer80-bf16.safetensors` | 7.51 GB | mtp layer weights |
| `layer80-q4.safetensors` | 2.30 GB | mtp layer weights |
| `layer80-residents-q.safetensors` | 121.07 MB | mtp layer weights |

Total: 9.92 GB.

## Integrity

No per-tensor integrity manifest accompanies this artifact yet; generate one before relying on the upload.

## Usage

These weights are served by the [mtplx](https://github.com/davidtai/MTPLX)
expert-streaming runtime for Apple Silicon (MLX): resident weights load into
memory while routed experts stream on demand from the sidecar via bounded
positional I/O (dense per-layer "island" banks, slot caches, or mmap-banked
execution). Download the repository contents into one directory and point
the runtime at it; the sharded sidecar layout is read directly - no
reassembly step. This artifact is **not** loadable with plain
`mlx_lm.load()`.

> License note: this repository inherits the upstream model license; set the
> `license` front-matter field before publishing.
