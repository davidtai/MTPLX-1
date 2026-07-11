# Hy3 Q4 MLX + MTP handoff

- Date: 2026-07-11
- Research branch: `codex/hy3-q4-native-serialized`
- Implementation commit: `4bcb382`
- Performance gates: completed; see `hy3-q4-native-benchmark-matrix.md`

## Delivered artifact

- Artifact name: `hy3-q4-mlx-mtp`
- Intended Hugging Face repo: `davidtai/hy3-q4-mlx-mtp`
- Artifact root: `/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp`
- Manifest: `/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp/expert-manifest.json`
- Manifest digest: `47767d76039f6b9058af2bb9192c3ebf3c447df6c9602d9ed719ca8a55961ec5`
- Conversion provenance: `/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp/conversion-provenance.json`
- Conversion-provenance file digest: `5bffad3e9600bbe9e74c70587e32ef826e625181e2f90c6fd2e5ca96037e0187`
- Runtime contract: `/Users/davidtai/.cache/huggingface/hy3-q4-mlx-mtp/mtplx_runtime.json`

The source is `tencent/Hy3@716aa7241bd6d95896be4ebfc761162a9c4d49ef`.
All 99 official files were downloaded with revision metadata and full LFS
SHA-256 values. Their aggregate file size is 597,578,239,288 bytes. The
conversion provenance records every source filename, size, computed hash, and
expected LFS hash.

`pipenetwork/Hy3-4bit@160619d3f96c8470350b6dac0ef033a8381551e3` was used only
as the pinned resident tensor shape/dtype/sharding convention. Every produced
weight value was quantized from the official Tencent BF16 source.

## Storage and layout

- Model key: `hy3-q4-native`
- Affine Q4/gs64 experts: layers 1-80, 192 experts per layer
- Expert records: 15,360 records, exactly 10,616,832 bytes each
- Sidecar: `experts.bin`, 163,074,539,520 bytes, 16 KiB alignment
- Sidecar SHA-256: `eca7e961f6d4963fb60504ad4e2d2a8659d53e290cd73717426cc6fbb9fb0228`
- Resident Q4 tensor bytes: 5,073,424,896
- Routed expert bytes: 163,074,539,520
- Logical Q4 tensor bytes: 168,147,964,416
- Resident files: 34 trunk safetensors plus
  `layer80-residents-q.safetensors`
- Routers: affine Q8/gs64 with the FP32 correction bias preserved bit-exactly
- BF16 MTP auxiliary file: `layer80-bf16.safetensors`, 7,505,299,360 file
  bytes and 7,505,224,960 tensor bytes, 593 tensors
- BF16 MTP SHA-256: `7dced9436d54f12fb52600568055677a57dc85663a78ce99283f19347e433fcc`

The BF16 head is a same-volume hard link to the verified source head: source
and artifact have inode `92415251`, device `16777232`, and link count 2. This
ships the Forge-contract BF16 A/B path without allocating another 7.5 GB.

The artifact's apparent `du -sh` size is 164 GiB. It contains no full Q4
checkpoint and no `layer80-q4.safetensors`; layer-80 Q4 experts live only in
the shared expert sidecar.

## Verification completed

The converter verified the complete artifact before its atomic publication.
An independent verifier was then run against the published directory with
`--records --shards --sidecar`:

- `valid: true`
- checked records: 15,360
- checked shards/files: 36
- checked auxiliary files: 1
- full sidecar hash: verified

The physical layout audit passed with:

- manifest and model spec valid
- manifest layers exactly 1-80
- 2,356 manifest resident keys and 2,323 executable trunk parameter keys
- zero missing or extra resident keys
- zero routed tensors in `model.safetensors.index.json`
- resident shard payload bytes exactly 5,073,424,896
- resident-only shard/index mapping exact
- layer-80 Q4 residents share the indexed file
- BF16 and Q4 MTP inventories, byte counts, source repo, and revision exact

Fixed-vector validation used seed `20260711`, four BF16 vectors, expert 0,
and kernel tolerances `atol=rtol=1e-5`.

- Trunk layer 1: streamed/direct Q4 kernel match; projection weight cosines
  0.994999-0.995959; correction bias bit-exact; stored Q8 leaves, router IDs,
  and router scores exactly match a fresh affine-Q8/gs64 quantization.
- MTP layer 80: streamed/direct Q4 kernel match; projection weight cosines
  0.995785-0.995870; correction bias bit-exact; stored Q8 leaves, router IDs,
  and router scores exactly match a fresh affine-Q8/gs64 quantization.
- Layer-1 Q8 routing did not exactly match raw BF16 IDs on the random near-tied
  diagnostic. This is reported, not hidden: score cosine was 0.99999946 and
  max absolute score error was 0.00148654. The specified resident contract is
  Q8, and its leaves/IDs/scores matched exactly.
- BF16-vs-Q4 expert output error is reported in the JSON evidence. The repo's
  validation gate defines strict streamed-vs-resident Q4 tolerance, not a
  numeric BF16-vs-Q4 output threshold.

Evidence files:

- `benchmarks/results/hy3-q4-native-layer1-validation.json`
- `benchmarks/results/hy3-q4-native-layer80-validation.json`
- `benchmarks/results/hy3-q4-native-perplexity-sanity.json`

The 24-token quality sanity sample passed. Native and community artifacts both
produced mean NLL `4.417746067047119` and perplexity `82.90920281746611`, for a
native/community perplexity ratio of `1.0`. The result explicitly carries
`performance_claim: false`.

Repository validation completed with 2,029 tests collected: 2,025 passed and
four expected skips. The focused serializer/runtime/compactor suite passed 156
tests before the real build; the post-build validator suite passed 12 tests.
Ruff passed on every changed Python file. A repo-wide Ruff invocation still
reports five unrelated pre-existing findings outside this branch's diff.

## Generation gate summary

The main-session generation gates are complete. The full comparison is in
`benchmarks/results/hy3-q4-native-benchmark-matrix.md`, with raw provenance in
the three `hy3-q4-native-gate-*.json` files.

- Native AR: 1,905 tokens, 5.922 tok/s, 83.02 GiB peak.
- Native Q4 trunk plus BF16 MTP head: 316/1,404 accepted (22.51%),
  3.832 tok/s, 96.00 GiB peak.
- Fully quantized native Q4 trunk plus Q4 MTP head: 296/1,424 accepted
  (20.79%), 3.769 tok/s, 86.17 GiB peak.

Native AR token IDs exactly match the community AR gate. Native BF16-MTP token
IDs, accepted/drafted counts, acceptance, and peak memory exactly match the
community BF16-MTP gate. The new trunk therefore does not improve the MTP
acceptance stall in this gate, strongly refuting community trunk hiddens as
the dominant cause for this prompt and runtime. These are single-run
measurements and should not be read as throughput confidence intervals.

## Operational handoff and deviations

- Qwen was stopped only for quantization, GPU validation, and the serialized
  generation-gate window, then restored.
  Its `/v1/models` endpoint again serves
  `mtplx-qwen36-27b-optimized-speed`.
- The benchmark/probe run-lock was clear before compute and was checked during
  every quantization unit.
- The fixed realistic-prompt AR, BF16-MTP, and Q4-MTP generation gates were
  run in the main session. Thermal and long-context gates were not run.
- The official BF16 source directory remains intact at
  `/Users/davidtai/.cache/huggingface/hy3-mtp-layer80`.
- The duplicated community artifact remains intact. The branch includes a
  separately tested generic resident-only compactor for issue #8, but it was
  not run destructively against the community artifact.
- No source shard, cache entry, or old artifact was deleted automatically.
- The native artifact is local and was not uploaded to Hugging Face.

The artifact remains available under `hy3-q4-native`, using the artifact root
above and its authoritative manifest. For the Q4 MTP path, point
`mtp_artifacts` at the same artifact root; the BF16 MTP path uses the
manifest-bound BF16 auxiliary file.
