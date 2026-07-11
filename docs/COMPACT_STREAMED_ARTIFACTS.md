# Compact streamed MoE artifacts

The streamed runtime needs two kinds of model data:

- always-resident tensors such as attention, embeddings, norms, routers, and
  the LM head; and
- routed experts in the aligned `experts.bin` sidecar.

Keeping the original full checkpoint beside `experts.bin` stores every expert
twice. The compact layout repacks only resident tensors into safetensors and
makes the sidecar authoritative. It never deletes the source.

## Compact an existing Q4 artifact

Start with a complete, hash-bearing sidecar manifest. Existing research
artifacts normally call this `expert-manifest-sidecar.json`:

```bash
python scripts/compact_streamed_artifact.py \
  "$SOURCE" \
  "$SOURCE/expert-manifest-sidecar.json" \
  "$SOURCE/experts.bin" \
  "$OUTPUT"
```

The builder:

1. verifies the source manifest, expert records, sidecar, and source shards;
2. copies resident tensor byte ranges with bounded memory;
3. hard-links `experts.bin` and manifest-bound auxiliary files by default;
4. emits a resident-only safetensors index and authoritative manifest;
5. proves resident byte parity and that no routed tensor remains in a shard;
6. atomically renames a fully verified temporary directory to `$OUTPUT`.

If output is on another filesystem, hard-linking fails closed. Copying a large
sidecar requires explicit consent:

```bash
python scripts/compact_streamed_artifact.py \
  "$SOURCE" "$MANIFEST" "$SOURCE/experts.bin" "$OUTPUT" \
  --allow-sidecar-copy
```

`compaction-handoff.json` records whether each large file was linked or copied,
all verification results, and the source files that may become cleanup
candidates. The compactor itself has no deletion mode.

## Build native Hy3 Q4 directly from BF16

The native converter avoids both a full intermediate Q4 checkpoint and a
second layer-80 Q4 expert file. It writes trunk layers 1-79 and MTP layer 80
directly into `experts.bin`, writes only resident Q4 tensors to safetensors,
and binds the bit-exact BF16 MTP head as a manifest-hashed auxiliary file.

The source and layout revisions are immutable in official mode:

- `tencent/Hy3@716aa7241bd6d95896be4ebfc761162a9c4d49ef`
- `pipenetwork/Hy3-4bit@160619d3f96c8470350b6dac0ef033a8381551e3`

Inspect tensor headers and exact planned byte geometry first:

```bash
python scripts/build_hy3_native_q4.py \
  --source "$HY3_BF16" \
  --oracle "$HY3_Q4_LAYOUT_ORACLE" \
  --bf16-head "$HY3_BF16/layer80-bf16.safetensors" \
  --output "$HY3_NATIVE_Q4" \
  --plan-only
```

Run the passive full-shard hashing and BF16-head proof while other GPU work is
active:

```bash
python scripts/build_hy3_native_q4.py \
  --source "$HY3_BF16" \
  --oracle "$HY3_Q4_LAYOUT_ORACLE" \
  --bf16-head "$HY3_BF16/layer80-bf16.safetensors" \
  --output "$HY3_NATIVE_Q4" \
  --prepare-only
```

Before the final command, require the benchmark/probe run-lock to be empty and
announce the compute window. The converter checks the run-lock before MLX is
imported and throughout quantization. Rerunning the same command resumes its
fsynced journal:

```bash
pgrep -fl 'python.*(benchmark_streamed|probe_mtp)'

python scripts/build_hy3_native_q4.py \
  --source "$HY3_BF16" \
  --oracle "$HY3_Q4_LAYOUT_ORACLE" \
  --bf16-head "$HY3_BF16/layer80-bf16.safetensors" \
  --output "$HY3_NATIVE_Q4"
```

The default requires a same-volume hard link for the BF16 head. Use
`--copy-bf16-head` only when the extra allocation is intentional.

## Verify and load

Run the manifest verifier and physical layout audit before inference:

```bash
python scripts/verify_expert_manifest.py \
  "$OUTPUT" "$OUTPUT/expert-manifest.json" \
  --records --shards --sidecar

python scripts/audit_streamed_model_layout.py \
  --model hy3-q4-native \
  --artifact-root "$OUTPUT" \
  --manifest "$OUTPUT/expert-manifest.json"
```

Use `hy3-q4` for an existing community artifact compacted without
requantization. Use `hy3-q4-native` only for the pinned BF16-to-Q4 build.

For the native build, run the fixed-vector/router validator and the small
perplexity-class sanity check before making any performance claim:

```bash
python scripts/validate_hy3_native_quantization.py \
  "$HY3_BF16" "$HY3_NATIVE_Q4"

python scripts/evaluate_streamed_perplexity.py \
  "$HY3_NATIVE_Q4" "$HY3_NATIVE_Q4/expert-manifest.json" \
  --baseline-root "$HY3_Q4_COMMUNITY" \
  --baseline-manifest "$HY3_Q4_COMMUNITY/expert-manifest-sidecar.json"
```

These are correctness and quality-sanity gates, not throughput benchmarks.
Full benchmarks remain a separate idle-machine release gate.

## Archive and cleanup

A hard link is independently durable: unlinking the source pathname does not
remove data while the compact artifact still links the inode. Copying or
archiving the compact directory to another volume allocates the sidecar again.

Only consider source cleanup after all verification commands pass, a load test
succeeds, and `compaction-handoff.json` reports `cleanup.eligible: true`.
Review the listed paths and remove them manually. Never remove the only BF16
source if reproducibility or future requantization still requires it.
