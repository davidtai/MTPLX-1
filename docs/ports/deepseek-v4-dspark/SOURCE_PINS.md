# DeepSeek V4 DSpark Port Source Pins

## Clean implementation base

- MTPLX repository: `https://github.com/youssofal/MTPLX.git`
- Base: `upstream/main@2b0360ca1af5c383a797a9d96999540f3197f182`
- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-deepseek-v4-dspark-k5`
- Branch: `feat/deepseek-v4-dspark-k5`

## Behavior references

- MiaAI-Lab launcher and patches: `MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark@d4ba142bc1d971eb73a911e207e3e963bbb3c455`
- MiaAI target artifact: `0xSero/deepseek-v4-flash-0731-spark@22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- MiaAI runtime image: `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`
- Image vLLM tree: `local-inference-lab/vllm@30038602b71395f481ef4a6edfe4fcf8551d9c15`
- Image SparkInfer tree: `local-inference-lab/sparkinfer@272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`
- DeepSpec: `MiaAI-Lab/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2`
- Official DSpark inference source: `DeepSeek-V4-Flash-DSpark@aa22cb07426656189b2573b8e77a9b7333b8ae0f`

## Reused DFlash2 runtime

- Existing MTPLX DFlash2 branch: `perf/qwen38-dflash2@c3487dc56de6c734c71508c1e293a44731ff025f`
- DFlash2 dependency: `davidtai/dflash-mlx@308672c08a04184cd075742db6db83ef6233296c`
- Imported MTPLX bridge commit: `4d3d03aa`
- Runtime authority: `dflash_mlx.engine.spec_epoch.SpeculativeSession`
- Generation authority: `dflash_mlx.runtime.stream_dflash_generate`

DeepSeek adds target and draft protocol adapters only. DFlash2 continues to own
prefill orchestration, speculative epochs, physical verification, acceptance,
target rollback calls, next-primary selection, events, and cleanup.

The official `DSparkAttention` source establishes that persistent stage K/V is
context K/V projected from accepted target taps. The five neural draft rows are
combined with that context only for the proposal attention call and are not
retained in the stage cache. The MLX cycle therefore trims rejected target-M6
rows and inserts only the retained target-tap prefix into each stage's Mia stock432
context ring.

## Exact local Mia artifact

- Pinned source directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI`
- Source `config.json` SHA-256: `b001ec8308044aa11daa0e624f5aea5e5362a63c05879a83a7be046b00eada82`
- Source `model.safetensors.index.json` SHA-256: `61af5c0782a8651ef893004e84369d2281a0fc316c8bcefc0bd8f76244224649`
- Those two hashes match the files served by the pinned `22f28d32...`
  Hugging Face revision.
- TP1 rank-sliced directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1`
- TP1 `config.json` SHA-256: `39f3a9e158019dc34dd943b64f874cfc43e9e392e6ce9215a56f2e183d661d90`
- TP1 `model.safetensors.index.json` SHA-256: `b7a450f88c99aee7f6d44ecb127e91e45ab5ccb1a0dad49ca9eabb90b400c304`
- TP1 rank-slice manifest SHA-256: `cee5b97698e16433f88e7ca23ab529acaa13628ae4af3ea18590ba4060c1203e`
- Separately derived K64 DSpark directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-dspark-k64`
- K64 draft weight index SHA-256: `c0d0e18e8c84fe6f1b7dc6991a4ba5765d1965f21f8892887aa01169fc2ba2b3`
- DSpark: block size 5, Markov rank 256, noise token 128799, target taps 40/41/42, stage namespaces 0/1/2.
- Persistent target and draft K/V use Mia-compatible 432-byte `stock432`
  NVFP4 records. Ratio-4 indexer records use Mia's 132-byte E4M3 plus FP32
  power-of-two scale layout.

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
- Worktree Python: `.venv/bin/python`, with the exact DFlash dependency installed
  from the pinned Git revision above.
