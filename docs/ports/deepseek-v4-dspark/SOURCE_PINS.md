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
- DFlash2 dependency: `davidtai/dflash-mlx@d67e6e4788f82c114b8f4efee8c62501d6cf3386`
- Imported MTPLX bridge commit: `4d3d03aa`
- Runtime authority: `dflash_mlx.engine.spec_epoch.SpeculativeSession`
- Generation authority: `dflash_mlx.runtime.stream_dflash_generate`

DeepSeek adds target and draft protocol adapters plus a construction-qualified
fixed-linear capability.  DFlash2 continues to own prefill orchestration,
speculative epochs, physical verification, acceptance, target rollback calls,
next-primary selection, events, and cleanup.  The pinned dependency selects a
direct fixed-linear subset once per request for this capability: snapshots,
sparse prompt positions, adaptive depth, diagnostics, cache clearing, generic
AR fallback, and CopySpec history are absent, while the existing verification,
greedy acceptance, accepted-feature commit, rollback, asynchronous next-draft
launch, stop, and event ordering are unchanged.
The fixed capability additionally certifies that acceptance restore trims the
installed target pages directly, so the shared loop does not run a disabled
per-cycle rollback-arm call or repeat the staged-primary proof.

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
- Mia's fused FP8 indexer quantizes post-RoPE BF16 Q/K directly.  The pinned
  compressor stores a `rotate` member but never consumes it, so the exact lane
  does not apply the generic DwarfStar Hadamard before FP8 quantization.
- Mia prefill consumes compact top-k indices/lengths and never materializes a
  full query-by-context boolean selection matrix.  SparkInfer's CUDA path then
  uses 16-head groups, native BF16 QK over the dequantized E2M1/E4M3 records,
  one FP32 online softmax across the SWA/indexed-cache union, and BF16 P.V with
  FP32 accumulation.  MTPLX maps that ownership to M5 NAX M16xN32xK16 tiles:
  16 heads per threadgroup, 32 candidates per tile, four QK K-splits, and eight
  PV SIMD groups covering the 512-value row.  The exact route installs that
  engine for prefill and retains the measured one-head direct kernel for
  decode.  A minimal installation-time dispatch compiles the fixed BF16 NAX
  pipeline before serving, matching vLLM's compile/warm lifecycle. Phase is the
  only hot-path route, neither lane silently falls back, and a request cannot
  become the first compiler trigger.

## Historical implementation boundary

- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/deepseek-v4-0731-dspark`
- HEAD: `ea0c9d3968f8cac8dfc58805d92965c46943b45e`
- State at Phase 1 start: 141 porcelain-status rows; preserved without modification.

The historical implementation is evidence only. It conditioned DSpark on the previous target token, forced the authoritative primary into the first proposal row, and never implemented useful K5 with physical-M6 verification. Reuse only arithmetic and weight-layout facts independently re-established from the pinned sources above.

## Phase 1 route boundary

- Explicit daemon route: `mtplx/server/openai.py --generation-mode dspark`
- Fixed contract: `--depth 5 --temperature 0 --load-mtp`
- Runtime loader: `mtplx.runtime.load(..., dspark=True)` validates the artifact
  before model construction and does not invoke generic MTP injection,
  projection requantization, packed-MoE/NAX patching, or the unrelated generic
  post-load installer stack after the engine has been sealed.
- Worktree Python: `.venv/bin/python`, with the exact DFlash dependency installed
  from the pinned Git revision above.

## Closed source-to-installed-route inventory

This inventory is the implementation gate for the Mia lane.  Paths under
`vllm/` refer to the pinned image vLLM tree; paths under `sparkinfer/` refer to
the pinned SparkInfer tree.  Every MTPLX route below is selected and checked by
`load_mia_exl3_dspark_model` and `build_mia_engine_plan` before a request can be
created.  The enabled callables do not probe eligibility or fall back.

### 1. B12X compressed sparse MLA

- **Source:** `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`,
  `sparkinfer/attention/_shared/mla/prefill.py`, `prefill_mg.py`, `kernel.py`,
  `kv_cache.py`, and Mia's patched
  `image-patch/sparkinfer/moe/_shared/kernels/tiny_decode.py`.
- **Arithmetic and layout:** 64 query heads, 512-wide latent rows, 64 RoPE
  dimensions, a 128-token SWA unioned with selected compressed rows, stock432
  E2M1 plus E4M3 scales, BF16 QK/P.V operands, and one FP32 online softmax with
  the learned sink.  The 432-byte record stores all 512 compressed latent
  values in bytes `[0,256)`, their 32 E4M3 group scales in `[256,288)`, zero
  padding in `[288,304)`, and a separate 64-value BF16 RoPE tail in
  `[304,432)`; the RoPE tail never replaces the final 64 latent values used by
  P.V.  Attention scaling is applied to the completed FP32 QK dot, not to each
  BF16 query element.  Prefill owns 16 heads per group, 32 candidates per M5
  NAX tile, four QK K-splits, and eight P.V SIMD groups.
- **MTPLX implementation:**
  `mtplx/kernels/deepseek_v4_nvfp4_mla.py`.
- **Construction owner / installed callable:**
  `DeepseekV4Attention.install_mia_nvfp4_attention` installs
  `_mia_cached_forward_uncompressed`, `_mia_cached_forward_ratio4`, or
  `_mia_cached_forward_ratio128` from the layer's immutable compression ratio;
  the enabled target lane never enters the generic cache/no-cache branch.  It
  also installs
  `_run_paged_nvfp4_prefill_mla` for prefill and
  `_run_paged_nvfp4_sparse_mla` for decode/verification.  Uncompressed target
  layers and DSpark stages install the corresponding direct-record callables.
  `MiaMLAWorkspace` owns invariant empty operands once.
  `DeepseekV4TargetOps` explicitly labels every DFlash prompt chunk as prefill
  and every physical M6 target call as decode/verification, so the external
  DFlash engine cannot leave the phase at `unknown`.
- **Disposition:** source-derived Metal port.  Phase is the only runtime route.

### 2. B12X sparse indexer

- **Source:** `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`,
  `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`,
  `sparkinfer/attention/nsa_indexer/fused_indexer.py`, `tiled_topk.py`, and
  `paged.py`.
- **Arithmetic and layout:** post-RoPE BF16 Q/K is quantized directly to E4M3;
  each 128-wide row is a 132-byte record with a FP32 power-of-two scale.  Scores
  are `sum_h(max(q_h dot k, 0) * w_h)`: tiled prefill decodes the raw E4M3
  operands into exactly representable FP16 values, completes the dot in FP32,
  then applies the Q and K row scales before ReLU and head weighting.  Selection
  is exact top-512 over only causal ratio-4 rows, represented as compact indices
  plus lengths.  The bounded prefill route carries candidates across score
  tiles; decode produces candidates in one fused pass and both use the same
  four-pass radix fold.
- **MTPLX implementation:** `mtplx/deepseek_v4_paged_indexer.py` and the
  `Indexer` installation seam in `mtplx/models/deepseek_v4.py`.
- **Construction owner / installed callable:** `MiaIndexerWorkspace` is sized
  for 8,224 query rows and top-512 by `build_mia_engine_plan`.
  `Indexer.install_mia_paged_topk` binds
  `_run_installed_indexer_query_records` and
  `_run_installed_paged_indexer_phase_topk`; the installed query finalizer
  accepts the construction-qualified record shape and a prebound FP32 weight
  scalar directly, and the selector routes only on prefill versus
  decode/verification.
- **Disposition:** source-derived Metal port.  Generic `argsort`,
  `argpartition`, a full score history, Hadamard rotation, and a
  query-by-context boolean mask are not reachable from the Mia route.

### 3. Fused compressor and cache insertion

- **Source:**
  `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` and the
  compressor/cache wiring in `vllm/models/deepseek_v4/nvidia/model.py`.
- **Arithmetic and layout:** FP32 projection outputs and gate logits are folded
  with per-dimension window softmax; ratio-4 includes the preceding half-window
  and ratio-128 does not.  The result takes the source BF16 boundary, RMSNorm,
  compressor RoPE, then direct stock432 NVFP4 packing.  The indexer copy packs
  direct post-RoPE Mia132 E4M3 records.  Full and incremental paths preserve the
  same completed-window frontier.
- **MTPLX implementation:** `mtplx/kernels/deepseek_v4_compressor.py`,
  `Compressor.mia_records`, `Compressor.step_records`, and paged
  `append_records` owners.
- **Construction owner / installed callable:** every compressed target
  attention installs `_nvfp4_record_impl`; every ratio-4 indexer installs
  `_indexer_record_impl`.  `DeepseekV4NVFP4Cache` owns the frontier and writes
  already-finalized records into its fixed pages.  The exact single-sequence
  route replaces vLLM's full request state pages with an arithmetic-equivalent
  absolute-position circular state cache sized only to the live compressor
  window plus the installed 64-row rollback allowance: `[21,72,2,1024]` for
  ratio-4 target state, `[21,72,2,256]` for its indexer state, and
  `[20,192,2,512]` for ratio-128.
  Incremental compression gathers its unfinished window from these fixed rows;
  it does not concatenate a retained Python journal or prepend the previous
  ratio-sized window to the current projected batch.  The fused finalizer takes
  the previous and current windows as separate device views and selects the
  previous source only for output row zero.
- **Disposition:** source-derived Metal port.  The generic pool/norm/RoPE and
  repack chain is not reachable with a Mia cache.

### 4. B12X mHC

- **Source:** `vllm/models/deepseek_v4/nvidia/model.py` and
  `sparkinfer/norm/mhc/_kernels.py`.
- **Arithmetic and layout:** hidden size 4,096, four residual streams, 20
  Sinkhorn iterations, FP32 routing matrices, the source Gram-trick norm, BF16
  carried values, fused RMSNorm at the following branch, final post, and head
  collapse.  Target taps are reconstructed after layers 40, 41, and 42 without
  breaking carried residual ownership; the layer-42 reconstruction is reused
  as the final trunk state exactly as in the pinned vLLM path.  The large-M
  repeated `post_pre` route follows SparkInfer's prefill split: one BF16
  POST/Gram producer, a BF16 matrix projection with FP32 accumulators, and the
  compact Gram finalize.  Initial pre and final head retain the source FP32
  split reduction.  The SM120 default selects the sibling TF32 projection at
  this boundary; Metal has no TF32 matrix operand mode, so the port selects
  SparkInfer's co-shipped BF16 projection and the same construction-time
  FP32-to-BF16 routing-weight conversion used by the pinned vLLM model.
  Metal uses its existing 8x8 simdgroup matrix primitive with BM64, BK32, and
  padded N32 for the source N24 projection; M below 384 retains the source
  split-32 FP32 route used by M6 verification and K5 drafting.  Target and
  draft construction also reproduce `finalize_mhc_broadcast_weights`: the
  first attention `fn` is summed over its four identical input streams once,
  and initial pre consumes the resulting FP32 `[24,4096]` matrix directly.
- **MTPLX implementation:** `mtplx/kernels/deepseek_v4_mhc.py` and the fixed
  `_run_mia_hc_target_tail_taps` / `_mia_propose_k5` state machines.
- **Construction owner / installed callable:** one `MiaMHCPlan` for the
  43-layer target and one for the three DSpark stages.  The target binds
  `_mia_hc_hidden` plus `_mia_collapse`; the draft binds `_mia_propose_k5`.
  Construction materializes each block's source FP32 `fn` matrix's BF16 MMA
  view once, seals M384/BM64, and owns compact `[M,11]` Gram and `[M,24]`
  projection outputs at repeated large-M post-pre boundaries.  Initial/head
  FP32 partials and the M<384 post-pre partials remain explicit source routes;
  the initial route owns the precollapsed FP32 broadcast matrix.
- **Disposition:** source-derived Metal port for target prefill, target M6
  verification, and draft K5.

### 5. EXL3 Trellis target MoE and routing

- **Source:** `vllm/models/deepseek_v4/nvidia/model.py`,
  `sparkinfer/gemm/trellis_linear/_small_m.py`, and
  `sparkinfer/moe/fused_moe/_impl.py`; the weight format is pinned by the target
  `EXL3_MANIFEST.json` and its ExLlamaV3 revision.
- **Arithmetic and layout:** K216, top-6 sqrt-softplus routing with correction
  bias except for the first three token-hash layers.  Softplus retains the
  source's exact `x > 20 ? x : log1p(exp(x))` arithmetic before the square
  root; the correction bias affects selection only, and unbiased selected
  scores are normalized then scaled by 1.5.  Source histogram, prefix, and
  expert-major route packing feed EXL3 MCG trellis payloads with H128 input
  transforms; fused clamped gate/up activation; down projection; route weight
  reduction and shared-expert addition.  The launcher's exact decode graph
  widths use BM8 and the construction-owned large-prefill arena uses BM64.
- **MTPLX implementation:** `mtplx/kernels/deepseek_v4_moe_router.py` and the
  installed `EXL3SwitchGLU.fused` path in `mtplx/deepseek_v4_exl3.py`.
- **Construction owner / installed callable:** each target `DeepseekV4MoE`
  binds `_mia_exl3_forward`; each gate binds `_mia_hash_route` or
  `_mia_score_route`; `EXL3SwitchGLU.install_trellis_runtime` binds the BM8 and
  BM64 plans.  `_pack_trellis_routes` is the enabled packer.
- **Disposition:** source-derived Metal W4A16 port.  The older QMV and generic
  `mx.argsort` implementations remain explicit stock/oracle code and are not
  reachable from `_mia_exl3_forward`.

### 6. WO inverse-RoPE and two projections

- **Source:** `sparkinfer/gemm/_shared/wo_mxfp8.py` and
  `sparkinfer/gemm/wo_projection/api.py`.
- **Arithmetic and layout:** inverse RoPE is fused into the grouped WO-A input;
  WO-A and WO-B consume the artifact's byte-identical E4M3 weights with E8M0
  group scales and retain the model BF16 boundary.
- **MTPLX implementation:** `_MiaInverseRopeGatherOLora` and
  `_mia_inverse_rope_output_kernel` in `mtplx/models/deepseek_v4.py`.
- **Construction owner / installed callable:**
  `install_deepseek_v4_o_lora_routes(mode="gather_qmm")` installs one direct
  object on all 43 target and three draft attention modules after the weights
  have been converted to the native MLX group-32 MXFP8 view.
- **Disposition:** fused inverse-RoPE Metal port plus the existing native MLX
  MXFP8 projections; the unfused output route is not selected.

### 7. General non-expert FP8 linears

- **Source:** the FP8 module declarations in
  `vllm/models/deepseek_v4/nvidia/model.py` and the artifact's dynamic E4M3,
  128-by-128 E8M0 scale contract.
- **Arithmetic and layout:** E4M3 payload bytes stay unchanged.  Each 128-by-128
  E8M0 source scale is repeated into the equivalent native group-32 scale view;
  weights are viewed as `uint32` without decoding or requantizing.
- **MTPLX implementation:** `_expand_mia_fp8_block_scales`, target/draft
  sanitizers, and `_quantize_loaded_modules` in `mtplx/deepseek_v4_exl3.py`.
- **Construction owner / installed callable:** the loader replaces every
  scale-bearing non-expert linear with `nn.QuantizedLinear(mode="mxfp8")` once,
  records the complete module map, and installs the 106 GB target through five
  bounded carried shards plus one complete 2 GB EXL3 layer shard at a time.
  Source arrays are evaluated into their destination and released before the
  next shard; `build_mia_engine_plan` rejects a missing streaming-load receipt
  or any module-map mismatch.  Request execution calls the installed native
  MXFP8 operator.
- **Disposition:** existing exact native storage/operator implementation; no
  request-path dequantized weight copy exists.

### 8. K64 fixed-K5 DSpark and DFlash2

- **Source:** `vllm/models/deepseek_v4/nvidia/dspark.py`, Mia's patched
  `image-patch/vllm/models/deepseek_v4/nvidia/dspark.py`, and the pinned
  DFlash2 `SpeculativeSession` / `stream_dflash_generate` runtime.
- **Arithmetic and layout:** post-layer target taps 40/41/42 feed one main
  projection; three draft stages own K64 routed experts; proposal input is the
  accepted primary plus four noise IDs; the five sequential Markov-biased
  argmax rows are future tokens.  Persistent draft K/V contains only accepted
  target context in a chronological stock432 128-row ring.  Each proposal sees
  that context plus five temporary neural rows.  DFlash2 physically verifies
  primary plus five futures (M6) and commits only the accepted prefix.
- **MTPLX implementation:** `mtplx/models/deepseek_v4_dspark.py`,
  `mtplx/deepseek_v4_dflash2.py`, and the DeepSeek binding in
  `mtplx/benchmarks/dflash2_runtime.py`.
- **Construction owner / installed callable:** the separately validated K64
  package constructs `DeepseekV4DSparkOwner`; its three expert banks use native
  group-32 MXFP4, its gates use `_mia_score_route`, and its proposal callable is
  `_mia_propose_k5`.  Each attention stage binds `_run_k5` and the direct
  `_run_pack_stock432` finalizer.  Its installed
  `_run_dspark_k5_nvfp4_mla` consumes the persistent 128-row ring and five
  proposal-local records as separate inputs to one online softmax, using
  `absolute_position % 128` for physical ring ownership.  It therefore does
  not concatenate a 133-row cache view, build visibility indices or lengths,
  allocate a temporary cache owner, or revalidate static record geometry.
  `DeepseekV4DSparkBackend.draft_greedy` is the DFlash2 backend;
  `DeepseekV4StreamingTargetFeatureStore` carries the three target taps as a
  tuple-native MLX tree and releases each prompt chunk after slicing its raw
  taps to the final 128 rows.  Only that at-most-128-row tail is concatenated
  for the main projection before inserting the three context-K/V copies; the
  full 1,024-by-12,288 tap tensor is never materialized.  Neither the projected
  DSpark context nor stage K/V is materialized for discarded prompt rows.
- **Disposition:** source-faithful DSpark/DFlash protocol port reusing the
  existing DFlash2 engine.  There is no full-prompt target-feature store.

### 9. Pages and bounded workspace ownership

- **Source:** vLLM's fixed KV block tables in
  `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`, SparkInfer's paged MLA and
  indexer modules above, and `sparkinfer/attention/sparse_mla/_scratch.py`.
- **Arithmetic and layout:** capacity is fixed at 384,000 logical tokens.  Each
  target layer owns an 8,416-row physical circular stock432 arena: the 8,224
  maximum input batch plus the logical 128-row attention window and M6 rollback
  allowance.  Every query exposes only its causal 128-row window.  Ratio-4 and
  ratio-128 layers own `ceil(capacity / ratio)` persistent stock432 pages;
  ratio-4 layers additionally own the same number of Mia132 index pages.
  DSpark owns three persistent physical 128-row rings.
- **MTPLX implementation:** `mtplx/paged_cache.py`, paged owners in
  `mtplx/deepseek_v4_nvfp4_kv.py`, and `MiaDeepseekV4EnginePlan`.
- **Construction owner / installed callable:** the immutable engine plan owns
  page geometry, shared indexer carry seeds, invariant MLA operands, and the
  fixed Metal threadgroup geometry, fixed compressor state rings, one persistent
  target page lease, and one persistent DSpark ring lease.  Per-dispatch Metal
  result arrays are bounded functional outputs, not growing histories.  Cache
  writes address the installed pages directly; request cleanup resets logical
  frontiers without reallocating their physical storage.
- **Disposition:** reusable fixed-capacity MTPLX paging specialized by exact
  DeepSeek record specs.  No geometric cache growth is on the Mia route.

### 10. Warmup and graph/callable ownership

- **Source:** `vllm/model_executor/warmup/b12x_sparse_indexer_warmup.py`,
  `deepseek_v4_compressor_warmup.py`, `kernel_warmup.py`, and Mia's eager
  launcher/capture configuration.
- **Arithmetic and layout:** the finite serving signatures are target prefill
  BM64, M384 mHC post-pre BF16 MMA BM64, sparse-indexer prefill,
  sparse-indexer decode, target M6 verification BM8, and three-stage DSpark K5
  BM8.  A 128-row target prefill reaches both compressor ratios; a direct
  384-row installed mHC call reaches the large-M projection without a second
  full target pass; a 513-row synthetic paged index view reaches top-512
  without a long model forward.
- **MTPLX implementation:** `MiaDeepseekV4EnginePlan.prewarm`.
- **Construction owner / installed callable:** the loader evaluates all weights
  and prewarms every signature before returning.  `DeepseekV4TargetOps` refuses
  to bind DFlash2 without a matching immutable-plan prewarm receipt.  The
  DeepSeek DFlash context fixes batch-one prompt chunks at Mia's 1,024-token
  long-prefill threshold while workspaces retain the 8,224-token scheduler
  cap; it fixes the draft window at 128, installs the dependency's fixed-linear
  M6 lane, and retains target/draft allocator state between chunks and requests.
- **Disposition:** MLX compile/prewarm equivalent of source eager warmup and
  piecewise capture.  The first request cannot become the compiler trigger.

## Source features intentionally absent on TP1 Metal

- **CUDA graph padding and replay repair:** these exist to keep CUDA graph
  addresses and batch buckets stable.  MTPLX has no CUDA graph or CUDA replay
  metadata.  The exact lane instead installs fixed batch-one K5/M6 callables,
  fixed page addresses, and eagerly compiles every MLX/Metal phase.  Copying the
  CUDA padding topology would add different arithmetic and state without a
  consumer.
- **TP/DCP collectives:** Mia's `entrypoint-no-download.sh` pins `TP_SIZE=1` and
  `DCP_SIZE=1`; DSpark non-causal attention also rejects DCP greater than one.
  The rank-coalesced artifact owns every K216 expert locally, so no collective
  or partial-output reduction exists in this route.
- **B12X A8 MoE activation path:** the NVIDIA `b12x-a8` lane is an SM120
  Tensor-Core/CuTe execution contract with dynamic FP8 activation tiles,
  N256/K128 geometry, and an in-place W4A8/QMMA weight representation.  Metal
  exposes neither that primitive nor that representation.  The target artifact
  separately pins EXL3 Trellis expert payloads, for which MTPLX installs the
  source-derived W4A16 Trellis route.  The K64 draft payload is native OCP MXFP4
  and maps byte-identically to MLX group-32 MXFP4.  Transplanting the A8 tile
  topology onto either format would change layout and arithmetic, so it is not
  part of the TP1 Metal port.
- **Confidence/capacity/dynamic-depth/draft-head FP8:** `start.sh` fixes
  `DSPARK_TOKENS=5` and `DSPARK_CAPACITY=0`; the packaged path does not enable
  dynamic depth or draft-head FP8.  No corresponding heads, counters, or hot
  branches are installed in MTPLX.
