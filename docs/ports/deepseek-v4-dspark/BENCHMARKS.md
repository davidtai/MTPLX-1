# Mia DSpark benchmark receipts

## Cold Python vocabulary ladder

These measurements use the exact local Mia/Sero package at
`/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1`, MTPLX source
`eba059982602387f2f868f80f6022cdf19cd3951`, and pinned DFlash revision
`54644e991039110f30140006c892c57734b9311e`.

Source lineage is documented against MiaAI's DGX Spark launcher and Sero's
packaged artifact, with the RTX6K Discord community
(`https://discord.gg/X54jjmcxWJ`) included in the references. Its related
RTX PRO 6000 / SM120 public wiki is pinned at
`local-inference-lab/rtx6kpro@3633c2c6028056729a6612126e9afe05c2e3cf08`.
These receipts are Apple Metal measurements and do not claim RTX PRO 6000
validation.

Every row is a separate process with an empty request cache. Model load is
reported separately and is not included in TTFT. The request prompt ends with
the same coherent 1,024-token Python repository task. Its prefix walks a
deterministic permutation of the tokenizer vocabulary, excludes special token
IDs, and avoids repeated filler IDs until the usable vocabulary is exhausted.
The 16K and 64K fillers contain no duplicate IDs. The 128K filler covers all
129,278 usable IDs before the 770 repeats that are mathematically required to
fill its remaining positions.

Each request generates exactly 1,024 tokens with physical M6 DSpark. TTFT is
wall time from request start through the first emitted token. The MLX peak is
the allocator high-water mark after an explicit reset immediately before the
request; it includes the already installed fixed physical cache arena.

| Cold prompt | Load | TTFT | Prefill | Prefill tok/s | Decode | Decode tok/s | Request | MLX peak | K5 accept | Cycles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 80.97 s | 87.02 s | 86.99 s | 188.35 | 50.21 s | 20.39 | 137.20 s | 103.424 GB / 96.321 GiB | 837/935 (89.52%) | 187 |
| 65,536 | 81.89 s | 371.39 s | 371.36 s | 176.48 | 64.76 s | 15.81 | 436.12 s | 103.424 GB / 96.321 GiB | 811/1,065 (76.15%) | 213 |
| 131,072 | 81.75 s | 808.37 s | 808.34 s | 162.15 | 58.81 s | 17.41 | 867.15 s | 103.424 GB / 96.321 GiB | 835/945 (88.36%) | 189 |

Raw receipts:

- [`mia-eba05998-python-vocab-cold-16384x1024.json`](../../../bench/deepseek-v4-mia/mia-eba05998-python-vocab-cold-16384x1024.json)
- [`mia-eba05998-python-vocab-cold-65536x1024.json`](../../../bench/deepseek-v4-mia/mia-eba05998-python-vocab-cold-65536x1024.json)
- [`mia-eba05998-python-vocab-cold-131072x1024.json`](../../../bench/deepseek-v4-mia/mia-eba05998-python-vocab-cold-131072x1024.json)

The command shape for each independent arm was:

```bash
python3 /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --child-timeout-seconds 1800 -- \
  .venv/bin/python scripts/deepseek_v4_dspark_k5_bench.py \
  --arm dspark --max-tokens 1024 \
  --prompt-tokens <16384|65536|131072> \
  --prompt-mode python-vocab --python-prompt-tokens 1024 \
  --out <receipt.json>
```

## What the ladder says

Cold prefill declines gradually from 188.35 tok/s at 16K to 162.15 tok/s at
128K. The fixed 384K target arena keeps allocator peak essentially flat: the
128K arm peaks only 460,940 bytes above the 16K arm. This is the intended
vLLM-style ownership result—request length advances logical page frontiers and
block tables instead of growing or materializing a contiguous cache. It also
means the roughly 103.4 GB peak is paid at installation rather than scaled to
the individual request.

Long-context decode is not monotonic in this single-run ladder because useful
tokens per physical M6 cycle vary with K5 acceptance. The 64K completion has
the lowest acceptance (76.15%), requires 213 cycles, and is therefore slower
than the 128K completion, which accepts 88.36% and finishes in 189 cycles. The
receipts should not be interpreted as an attention-only context-length curve.

## Physical-M6 full-acceptance gate

The retained physical-M6 target route compiles only its cache-free regions and
keeps all 43 cache-owning attention calls eager. Construction prewarms those
fixed regions and binds the route directly; decode does not perform an
eligibility check or silent fallback. On the same 1,024-token repeated-prompt,
single-cycle gate used for the earlier 39.74--39.79 tok/s comparison, the final
`5c382ad6` tree generated six tokens in one 137.833 ms physical cycle, accepted
all 5/5 DSpark drafts, and measured **43.531 tok/s**. The artifact, source pins,
stock432/Mia132 cache layout, and emitted token digest are recorded in the raw
receipt.

This is deliberately a full-acceptance cycle-cost gate, not a sustained or
nonrepetitive-prompt throughput claim. The cold Python-vocabulary ladder above
remains the sustained workload evidence, where acceptance depth determines how
many useful tokens each physical M6 cycle returns.

Raw receipt:

- [`mia-5c382ad6-piecewise-1024x6-full-accept.json`](../../../bench/deepseek-v4-mia/mia-5c382ad6-piecewise-1024x6-full-accept.json)

## Remaining performance headroom

The retained 16-byte Trellis staging change first raised matched short-cycle
decode from 30.84--30.87 tok/s to 39.74--39.79 tok/s with exact final bits and
5/5 accepted drafts. The construction-bound piecewise target route now clears
that same full-acceptance gate at 43.531 tok/s. This remains a cycle-cost result,
not a performance ceiling or a substitute for the sustained rows above. The
long-context receipts still expose meaningful work in paged MLA attention and
in the number of physical verification cycles.

Further optimization should start with a fresh profile of this post-staging
stack, then change only the largest measured bucket. Promising categories are
source-faithful long-context MLA/page scheduling and improvements that preserve
K5 acceptance while reducing verify work. MoE changes should be reconsidered
only if the new profile still shows a sufficient ceiling. Any follow-up must
retain Mia's arithmetic, stock432 NVFP4 records, Mia132 index layout, physical
M6 ownership, and construction-bound routing; no silent enabled-path fallback
or per-token eligibility checks belong in that work.
