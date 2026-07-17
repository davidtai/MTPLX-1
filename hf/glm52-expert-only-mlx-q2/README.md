---
base_model: mlx-community/GLM-5.2-4bit
tags:
- mlx
- moe
- mtplx
- expert-streaming
- q2
---
# glm52-expert-q2 (mtplx expert-streaming artifact)

Expert-only affine Q2 MoE artifact for the mtplx
expert-streaming runtime: dense (resident) weights as safetensors shards plus
all routed expert weights repacked into a record-major sidecar for SSD
streaming on Apple Silicon.

## Provenance

- Quantized source (manifest): `mlx-community/GLM-5.2-4bit` @
  `6b347a6472d46bf55de65ee34032136a3929d778`
- Upstream checkpoint: `mlx-community/GLM-5.2-4bit` @ `6b347a6472d46bf55de65ee34032136a3929d778`
- Converted by mtplx commit `e2aa268313f3930adf3038411025c8decef6d1e7` (schema `mtplx-glm52-expert-q2-conversion-v1`)
- Manifest digest: `d06a3eb5b9449948918297859922904e797a2815666a69f21295f9fd36edfe5c`

## Quantization

- Routed experts: **2-bit affine**, group size
  64, mode `affine`
- 75 routed layers (layers 3-77),
  256 experts per layer, 19200 expert records
- Expert record: 11.80 MB
  (11796480 bytes); routed total
  226.49 GB
- Resident (non-expert) weights: 10.63 GB

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

`experts.bin` is 226.49 GB, above the Hub's 50 GB per-file cap, so the published repository ships it as 14 record-aligned shards produced by `scripts/shard_expert_sidecar.py` (<= 16 GiB each):

- `experts-00001-of-00014.bin` (17.18 GB)
- `experts-00002-of-00014.bin` (17.18 GB)
- `experts-00003-of-00014.bin` (17.18 GB)
- `experts-00004-of-00014.bin` (17.18 GB)
- `experts-00005-of-00014.bin` (17.18 GB)
- `experts-00006-of-00014.bin` (17.18 GB)
- `experts-00007-of-00014.bin` (17.18 GB)
- `experts-00008-of-00014.bin` (17.18 GB)
- `experts-00009-of-00014.bin` (17.18 GB)
- `experts-00010-of-00014.bin` (17.18 GB)
- `experts-00011-of-00014.bin` (17.18 GB)
- `experts-00012-of-00014.bin` (17.18 GB)
- `experts-00013-of-00014.bin` (17.18 GB)
- `experts-00014-of-00014.bin` (3.21 GB)

The mtplx runtime reads the sharded layout directly through the updated `expert-manifest.json`.

## Files

| file | size | role |
| --- | ---: | --- |
| `expert-manifest.json` | 61.58 MB | expert manifest |
| `config.json` | 4669 B | metadata |
| `generation_config.json` | 194 B | metadata |
| `tokenizer.json` | 20.22 MB | metadata |
| `tokenizer_config.json` | 821 B | metadata |
| `chat_template.jinja` | 5076 B | metadata |
| `model.safetensors.index.json` | 240324 B | metadata |
| `conversion-manifest.json` | 343746 B | metadata |
| `resident-00001-of-00003.safetensors` | 4.28 GB | resident weights |
| `resident-00002-of-00003.safetensors` | 4.27 GB | resident weights |
| `resident-00003-of-00003.safetensors` | 2.09 GB | resident weights |
| `experts.bin` | 226.49 GB | expert sidecar |

Total: 237.21 GB.

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
