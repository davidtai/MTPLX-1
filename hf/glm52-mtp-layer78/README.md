---
base_model: zai-org/GLM-5.2
tags:
- mlx
- mtplx
- mtp
- speculative-decoding
---
# glm52-mtp-layer78 (mtplx MTP layer artifact)

Multi-token-prediction (MTP) layer weights extracted from the upstream
checkpoint so the mtplx runtime can attach speculative decoding to the
expert-streaming models without downloading the full source checkpoint.

## Provenance

- Extracted from `zai-org/GLM-5.2` source shards (see `mtp-artifact-manifest.json` for per-file SHA-256 pins over 7 source files)
- Producer: mtplx commit `32c83967c0bf1a521975374c9116593a1e9b10a8`
- Artifact SHA-256: `56a19e9c0328f3b8f9ec32569f17d76aef7c20081334971d8855514e409746a6`
- Manifest digest: `e6fbcb0a673b5072080eb9cf8efb0a0b7e3a9355f9d7fd1d198197b052902c67`

## Contents

- Architecture: `unknown`
- MTP layer tensors as safetensors (see file list below)

## Files

| file | size | role |
| --- | ---: | --- |
| `mtp-artifact-manifest.json` | 247987 B | mtp manifest |
| `layer78-bf16.safetensors` | 19.91 GB | mtp layer weights |

Total: 19.91 GB.

## Integrity

Every tensor carries a SHA-256 in the manifest (`artifact.tensors[]`), alongside its exact source shard and byte range; the manifest itself is digest-pinned.

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
