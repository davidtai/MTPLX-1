# Mia DSpark benchmark receipts

## Cold Python vocabulary ladder

These measurements use the exact local Mia/Sero package at
`/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1`, MTPLX source
`cc6e1b1986a946fc7907b292a827e6d80e1aa316`, and pinned DFlash revision
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
The 1K row is the coherent Python task without a vocabulary prefix. The 16K
and 64K fillers contain no duplicate IDs. The 128K filler covers all 129,278
usable IDs before the 770 repeats that are mathematically required to fill its
remaining positions.

Each request generates exactly 1,024 tokens with physical M6 DSpark. TTFT is
wall time from request start through the first emitted token. The MLX peak is
the allocator high-water mark after an explicit reset immediately before the
request; it includes the already installed fixed physical cache arena.

| Cold prompt | Load | TTFT | Prefill | Prefill tok/s | Decode | Decode tok/s | Request | MLX peak | K5 accept | Cycles |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 82.03 s | 5.84 s | 5.81 s | 176.23 | 32.74 s | 31.28 | 38.55 s | 103.815 GB / 96.686 GiB | 784/1,200 (65.33%) | 240 |
| 16,384 | 81.95 s | 87.27 s | 87.24 s | 187.80 | 48.38 s | 21.16 | 135.63 s | 103.915 GB / 96.778 GiB | 837/935 (89.52%) | 187 |
| 65,536 | 81.70 s | 371.34 s | 371.31 s | 176.50 | 65.88 s | 15.54 | 437.18 s | 103.915 GB / 96.778 GiB | 811/1,065 (76.15%) | 213 |
| 131,072 | 81.72 s | 808.21 s | 808.17 s | 162.18 | 62.70 s | 16.33 | 870.88 s | 103.915 GB / 96.778 GiB | 835/945 (88.36%) | 189 |

Raw receipts:

- [`mia-cc6e1b19-python-vocab-cold-1024x1024.json`](../../../bench/deepseek-v4-mia/mia-cc6e1b19-python-vocab-cold-1024x1024.json)
- [`mia-cc6e1b19-python-vocab-cold-16384x1024.json`](../../../bench/deepseek-v4-mia/mia-cc6e1b19-python-vocab-cold-16384x1024.json)
- [`mia-cc6e1b19-python-vocab-cold-65536x1024.json`](../../../bench/deepseek-v4-mia/mia-cc6e1b19-python-vocab-cold-65536x1024.json)
- [`mia-cc6e1b19-python-vocab-cold-131072x1024.json`](../../../bench/deepseek-v4-mia/mia-cc6e1b19-python-vocab-cold-131072x1024.json)

The command shape for each independent arm was:

```bash
python3 /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --child-timeout-seconds 1800 -- \
  .venv/bin/python scripts/deepseek_v4_dspark_k5_bench.py \
  --arm dspark --max-tokens 1024 \
  --prompt-tokens <1024|16384|65536|131072> \
  --prompt-mode python-vocab --python-prompt-tokens 1024 \
  --out <receipt.json>
```

## What the ladder says

The 1K prefill measures 176.23 tok/s because fixed request work is material at
that size. Long-context prefill declines from 187.80 tok/s at 16K to 162.18
tok/s at 128K. The fixed 384K target arena keeps allocator peak essentially
flat: the 128K arm peaks only 458,752 bytes above the 16K arm. This is the
intended vLLM-style ownership result—request length advances logical page
frontiers and block tables instead of growing or materializing a contiguous
cache. It also means the roughly 103.9 GB peak is paid at installation rather
than scaled to the individual request.

Sustained decode is not monotonic in this single-run ladder because useful
tokens per physical M6 cycle vary with K5 acceptance. The 64K completion has
the lowest acceptance (76.15%), requires 213 cycles, and is therefore slower
than the 128K completion, which accepts 88.36% and finishes in 189 cycles. The
receipts should not be interpreted as an attention-only context-length curve.

## Remaining performance headroom

The retained 16-byte Trellis staging change preserves exact final bits. The
construction-bound piecewise target route compiles only cache-free regions and
keeps all 43 cache-owning attention calls eager. The full 1,024-output rows
above remain the only published throughput evidence. The long-context receipts
still expose meaningful work in paged MLA attention and in the number of
physical verification cycles.

Further optimization should start with a fresh profile of this post-staging
stack, then change only the largest measured bucket. Promising categories are
source-faithful long-context MLA/page scheduling and improvements that preserve
K5 acceptance while reducing verify work. MoE changes should be reconsidered
only if the new profile still shows a sufficient ceiling. Any follow-up must
retain Mia's arithmetic, stock432 NVFP4 records, Mia132 index layout, physical
M6 ownership, and construction-bound routing; no silent enabled-path fallback
or per-token eligibility checks belong in that work.
