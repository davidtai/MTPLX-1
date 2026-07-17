---
base_model: tencent/Hy3
tags:
- mlx
- moe
- mtplx
- expert-streaming
- q4
---
# hy3-expert-only-q4 (mtplx expert-streaming artifact)

Expert-only affine Q4 MoE artifact for the mtplx
expert-streaming runtime: dense (resident) weights as safetensors shards plus
all routed expert weights repacked into a record-major sidecar for SSD
streaming on Apple Silicon.

## Provenance

- Quantized source (manifest): `local/hy3-expert-only-mlx-q4` @
  `716aa7241bd6d95896be4ebfc761162a9c4d49ef`
- Upstream checkpoint: `tencent/Hy3` @ `716aa7241bd6d95896be4ebfc761162a9c4d49ef`
- Manifest digest: `507ca09cebb9ef5180c46401db7b61d8a9759ffd04ffbc97c5dbba0e9ef89f43`

## Quantization

- Routed experts: **4-bit affine**, group size
  64, mode `affine`
- 79 routed layers (layers 1-79),
  192 experts per layer, 15168 expert records
- Expert record: 10.62 MB
  (10616832 bytes); routed total
  161.04 GB
- Resident (non-expert) weights: 17.49 GB

## Expert sidecar format

`experts-*.bin` (or a monolithic `experts.bin`) is a **record-major** sidecar
of the routed expert weights, described and trust-anchored by
`expert-manifest.json` (`mtplx-expert-manifest-v1`):

- one record per `(layer, expert)`; records are sorted by `(layer, expert)`
  and each record's bytes are contiguous;
- every record starts on a **16 KiB-aligned** offset (`sidecar_alignment`);
- a record is the nine quantized components in fixed order
  `gate_proj/up_proj/down_proj` x `weight/scales/biases`
  (`U32` packed weights, `BF16` scales/biases), laid out back to back;
- manifest **segment offsets are shard-absolute**: each segment names its
  file and the byte offset inside that file;
- the sharded layout cuts the sidecar at record boundaries into
  `experts-00001-of-000NN.bin`; each record then carries `sidecar_shard`
  plus its shard-relative offset, and alignment is preserved because shards
  begin exactly on record boundaries;
- every record has a SHA-256 in the manifest; the runtime verifies it on
  read (fail-closed), and the manifest itself is digest-pinned
  (`manifest_sha256`).

## Upload sharding (planned)

`experts.bin` is 161.04 GB, above the Hub's 50 GB per-file cap, so the published repository ships it as 10 record-aligned shards produced by `scripts/shard_expert_sidecar.py` (<= 16 GiB each):

- `experts-00001-of-00010.bin` (17.18 GB)
- `experts-00002-of-00010.bin` (17.18 GB)
- `experts-00003-of-00010.bin` (17.18 GB)
- `experts-00004-of-00010.bin` (17.18 GB)
- `experts-00005-of-00010.bin` (17.18 GB)
- `experts-00006-of-00010.bin` (17.18 GB)
- `experts-00007-of-00010.bin` (17.18 GB)
- `experts-00008-of-00010.bin` (17.18 GB)
- `experts-00009-of-00010.bin` (17.18 GB)
- `experts-00010-of-00010.bin` (6.43 GB)

The mtplx runtime reads the sharded layout directly through the updated `expert-manifest.json`.

## Files

| file | size | role |
| --- | ---: | --- |
| `expert-manifest.json` | 50.96 MB | expert manifest |
| `config.json` | 1015 B | metadata |
| `generation_config.json` | 204 B | metadata |
| `tokenizer.json` | 9.53 MB | metadata |
| `tokenizer_config.json` | 165971 B | metadata |
| `special_tokens_map.json` | 491 B | metadata |
| `chat_template.jinja` | 10223 B | metadata |
| `model.safetensors.index.json` | 82152 B | metadata |
| `conversion-provenance.json` | 36966 B | metadata |
| `model-00001-of-00018.safetensors` | 1.07 GB | resident weights |
| `model-00002-of-00018.safetensors` | 1.07 GB | resident weights |
| `model-00003-of-00018.safetensors` | 939.53 MB | resident weights |
| `model-00004-of-00018.safetensors` | 1.07 GB | resident weights |
| `model-00005-of-00018.safetensors` | 1.07 GB | resident weights |
| `model-00006-of-00018.safetensors` | 1.06 GB | resident weights |
| `model-00007-of-00018.safetensors` | 1.06 GB | resident weights |
| `model-00008-of-00018.safetensors` | 1.06 GB | resident weights |
| `model-00009-of-00018.safetensors` | 1.04 GB | resident weights |
| `model-00010-of-00018.safetensors` | 1.02 GB | resident weights |
| `model-00011-of-00018.safetensors` | 939.53 MB | resident weights |
| `model-00012-of-00018.safetensors` | 1.01 GB | resident weights |
| `model-00013-of-00018.safetensors` | 1.01 GB | resident weights |
| `model-00014-of-00018.safetensors` | 1.01 GB | resident weights |
| `model-00015-of-00018.safetensors` | 998.25 MB | resident weights |
| `model-00016-of-00018.safetensors` | 266.14 MB | resident weights |
| `model-00017-of-00018.safetensors` | 989.86 MB | resident weights |
| `model-00018-of-00018.safetensors` | 822.09 MB | resident weights |
| `experts.bin` | 161.04 GB | expert sidecar |

Total: 178.59 GB.

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
