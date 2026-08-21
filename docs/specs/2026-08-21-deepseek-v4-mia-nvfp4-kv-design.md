# DeepSeek V4 Mia `stock432` NVFP4 K/V Design

**Date:** 2026-08-21

**Status:** Approved for execution

## Goal

Replace the temporary MLX affine-int4 cache in the exact Mia/Sero K216 target
and K64 DSpark draft with Mia's native `stock432` NVFP4 cache contract, then
consume that representation directly from the bounded sparse-attention path.

This corrects both storage and arithmetic.  The current affine lane stores one
already-rotated 512-wide row and uses it as both key and value.  Mia's packaged
oracle instead uses an unrotated 512-wide latent as the value and forms the key
from the first 448 latent values plus a separately rotated 64-wide RoPE tail.

## Pinned Source Contract

The authoritative layout is the 432-byte record exercised by
`MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark@d1dc9e70d277746e4e369cc68f54d5c67a6afae8`
in `image-patch/selftest_extend.py` and read by the patched SparkInfer sparse-MLA
prefill/decode path:

```text
bytes   0..255   512 E2M1 values, low nibble first
bytes 256..287    32 E4M3 scales, one per group of 16 latent values
bytes 288..303    16 zero padding bytes
bytes 304..431    64 BF16 GPT-J-interleaved rotated-RoPE values
```

Dequantization is:

```text
latent[d] = e2m1(record[d / 2], d % 2) * e4m3(record[256 + d / 16])
value      = latent[0:512]
key        = concat(latent[0:448], bf16(record[304:432]))
```

The record has no affine zero point and is not MLX `mxfp4`.  It is a single
`uint8` owner, not three parallel packed/scale/bias arrays.

## Architecture

### Record owner and codec

`mtplx/deepseek_v4_nvfp4_kv.py` owns the fixed constants, a Metal pack kernel,
an oracle decoder, and `MiaNVFP4Rows`.  Construction fixes width 512, NoPE 448,
RoPE 64, group size 16, and record size 432.  Invalid geometry fails before the
owner is installed.

The owner accepts raw normalized latent rows and their already-computed rotated
RoPE tails.  Append, replacement, truncation, eviction, and state restoration
operate on whole records, preserving the existing target rollback and DSpark
ring ownership contracts.

### Target arithmetic

The target attention route keeps `kv_norm(wkv(x))` unrotated.  It computes the
64-wide rotated tail separately, inserts `(latent, rope)` into both the sliding
window and compressed cache, and never stores a shared rotated K/V row.

The attention compressor exposes the normalized pooled latent before RoPE and
the separately rotated compressed tail.  The indexer compressor and rollback
journals remain their existing auxiliary state; they are not reclassified as
NVFP4 attended K/V.

### DSpark arithmetic

Each of the three DSpark stages owns a distinct `MiaNVFP4Rows` ring.  Context
prefill and authoritative-main commits insert raw latent plus rotated RoPE.
Proposal-local rows remain ephemeral.  Attention reconstructs distinct K and V;
the inverse-output-RoPE workaround is removed.

### Direct sparse Metal consumption

`mtplx/kernels/deepseek_v4_nvfp4_mla.py` reads `stock432` records directly.
For every query/head it:

1. visits only the causal 128-row sliding interval;
2. visits the indexer's selected compressed rows, capped by the model's fixed
   `index_topk` contract;
3. decodes E2M1 and E4M3 in registers;
4. forms QK with the stored BF16 RoPE tail and accumulates PV from the raw latent;
5. includes the learned per-head sink in the online-softmax denominator; and
6. writes one BF16 512-wide output without materializing a score matrix or a
   dense dequantized cache.

The target route is installed once at construction.  It has no enabled-path
fallback, environment read, eligibility check, or engagement counter.  Prefill
continues through 1,024-query chunks, but each chunk's attention allocation is
bounded by selected rows rather than total context length.

## Migration

- Replace `DeepseekV4AffineInt4Cache` with `DeepseekV4NVFP4Cache` for the exact
  Mia artifact route.
- Replace every DSpark `AffineInt4Rows` ring with `MiaNVFP4Rows`.
- Update DFlash2 construction checks to require `stock432` owners.
- Remove the obsolete affine-int4 module and its affine-only tests once no live
  reference remains.
- Existing receipts describing affine-int4 K/V are superseded and cannot be
  published as final Mia evidence.

## Direct Verification Only

1. A fixed record contract check proves offsets, E2M1 nibble order, E4M3
   group-16 scaling, BF16 RoPE bytes, and distinct reconstructed K/V.
2. Target and all three draft cache constructors prove `stock432` ownership and
   exact rollback/ring replacement behavior.
3. One bounded Metal sparse-attention comparison checks the direct consumer
   against an oracle built from the same records.
4. The exact real model must pass one DSpark epoch, target-only/DSpark committed
   token parity, the Python service prompt, and the requested cold 1K/16K/64K
   matrix with peak memory.

No unrelated compatibility tests, generic NVFP4 framework, alternate record
layouts, or fallback routes are in scope.

## Failure-Mode Check

- **Critical: byte-compatible records but wrong values.**  The record gate checks
  decoded K and V independently and includes scale-boundary values before model
  construction.
- **Critical: long prefill still scales with full context.**  The direct consumer
  accepts selected indices and bounded window ranges; a whole-context score or
  dequantized tensor is not part of its interface.
- **Critical: rollback corrupts record alignment.**  replacement and truncation
  operate only in 432-byte row units and the existing rejection-repair gate must
  pass before benchmarking.
- **Minor: `stock432` is larger than the temporary affine record.**  This is an
  accepted cost of matching Mia's distinct K/V arithmetic and fused sparse
  consumer; it remains smaller than Mia's 584-byte padded record.

## Non-Goals

- MLX `mxfp4`, affine int4, Mia's 368-byte FP8-RoPE mode, or the alternate
  UE8M0/360-byte community layout.
- NVFP4 checkpoint-weight conversion.
- Replacing DFlash2, DSpark acceptance, event handling, or service architecture.
- Quantizing indexer scoring rows or compressor rollback journals as K/V.
- Any benchmark or PR claim from the superseded affine cache lane.
