# DeepSeek V4 DSpark Port Source Pins

## Clean implementation base

- MTPLX repository: `https://github.com/youssofal/MTPLX.git`
- Base: `upstream/main@2b0360ca1af5c383a797a9d96999540f3197f182`
- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-deepseek-v4-dspark-k5`
- Branch: `feat/deepseek-v4-dspark-k5`

## Behavior references

- MiaAI-Lab launcher and patches: `MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark@d1dc9e70d277746e4e369cc68f54d5c67a6afae8`
- MiaAI target artifact: `0xSero/deepseek-v4-flash-0731-spark@22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- MiaAI runtime image: `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`
- DeepSpec: `MiaAI-Lab/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2`
- Official DSpark inference source: `DeepSeek-V4-Flash-DSpark@aa22cb07426656189b2573b8e77a9b7333b8ae0f`

The official `DSparkAttention` source establishes that persistent stage K/V is
context K/V projected from accepted target taps. The five neural draft rows are
combined with that context only for the proposal attention call and are not
retained in the stage cache. The MLX cycle therefore trims rejected target-M6
rows and inserts only the retained target-tap prefix into each stage's affine-int4
context ring.

## Phase 1 checkpoint

- Directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-2.4bit-mixed`
- `config.json` SHA-256: `44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f`
- `model.safetensors.index.json` SHA-256: `f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861`
- DSpark: block size 5, Markov rank 256, noise token 128799, target taps 40/41/42, stage namespaces 0/1/2.
- K/V: affine int4, group size 64, quantized from cache offset zero for the target and all three DSpark stages.

## Historical implementation boundary

- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/deepseek-v4-0731-dspark`
- HEAD: `ea0c9d3968f8cac8dfc58805d92965c46943b45e`
- State at Phase 1 start: 141 porcelain-status rows; preserved without modification.

The historical implementation is evidence only. It conditioned DSpark on the previous target token, forced the authoritative primary into the first proposal row, and never implemented useful K5 with physical-M6 verification. Reuse only arithmetic and weight-layout facts independently re-established from the pinned sources above.

## Phase 1 route boundary

- Explicit daemon route: `mtplx/server/openai.py --generation-mode dspark`
- Fixed contract: `--depth 5 --temperature 0 --load-mtp`
- Runtime loader: `mtplx.runtime.load(..., dspark=True)` validates the artifact
  before model construction and does not invoke generic MTP injection.
- Worktree Python: `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python`
  because the clean worktree intentionally has no separate virtual environment.
