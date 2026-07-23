# Hy3 MTPLX-streaming — envelope matrix claim ledger

Every publishable claim → its receipt + commit on `eval/hy3-q2-2p6bit`.
Source of truth for the blog post and HF card. All perf cells: real
long-code prefill (`realistic_programming_v1`, exact-token gated), decode
256, greedy, **bf16 KV**, 3 reps. Envelope = WEIGHTS budget; KV on top.
Mode of record per cell = faster of AR/K1 (both lanes always in receipt).

## Config of record

| item | value | source |
|---|---|---|
| bank | oq2e (imatrix 2.44 bpw, 2.50 bpw effective) | [[hy3-oq2e-integration]] |
| projections | proj-requant q4 (rq4) | champion |
| KV | bf16 (kv4 reversed — see below) | commit `db46f26`, #130 5047822076 |
| MTP depth | K1 locked matrix-wide | commit `2e2ed99`, #130 5048661802 |
| preset (88) | `hy3-oq2e-rq4-88e` (islands 79, cache-0) | commit `c513c6d` |
| knob | 112 GiB wired (`iogpu.wired_limit_mb=114688`) | [[never-exceed-the-memory-knob]] |

## Context cells — decode tok/s (mode of record bold)

| envelope | 1024 | 16k | 64k | peak GiB |
|---|---|---|---|---|
| 88 | **K1 48.0** / AR 41.4 | **K1 27.6** / AR 25.7 | — (>112 at bf16) | 88.4–93.0 |
| 80 | **K1 21.6** / AR 20.0 | AR 15.9 / **K1** 5.8→AR | AR 9.2 / K1 3.2→AR | 79.8–99.3 |
| 64 | **K1 11.1** / AR 9.1 | **AR 15.9→**7.0 / K1 3.6 | **AR 4.0** / K1 3.4 | 63.0–64.2 |

Note: at 80/64 the 16k+64k mode of record is AR (K1 inverts). Receipts:
- 88×16k: `evals/tier2/t3_kselect_88e16k_bf16_rep{1,2,3}.json` (`2e2ed99`)
- 88×1024: `evals/tier2/t3_88col_1024_rep{1,2,3}.json` (`05034f4`)
- 80 column: `evals/tier2/t3_80col_{1024,16384,65536}_rep*.json` (`0831fa6`)
- 64 column: `evals/tier2/t3_64col_{1024,16384,65536}_rep*.json` (`9441b42`)

## Small-code arm (320-token code prompt, decode-dominant)

| envelope | decode tok/s | mode |
|---|---|---|
| 88e | K1 47.7 / AR 43.3 | **K1** +10% |
| 80 | K1 22.3 / AR 21.2 | **K1** +5.5% |
| 64 | K1 10.8 / AR 9.2 | **K1** +17.6% |

Receipts: `evals/tier2/t3_smallcode_{88e,80,64}_rep{1,2,3}.json` (`067e2e2`).
Standard coding tail is 319 tokens → min gate-passing context 320.

## Two-regime finding (the headline)

1. **K1 (MTP) wins at short context, every envelope** — 1024 and the
   320-token code arm, full OR streaming residency.
2. **AR wins at long context (16k+) on streaming envelopes** — speculative
   verify multiplies expert-miss servicing under partial residency;
   deepens with context. Acceptance UNCHANGED (0.89–0.90) → decode-cost/
   residency effect, NOT acceptance collapse. At 88 (full residency) K1
   wins every code cell.
3. **Acceptance is content-dependent, not envelope-dependent:** 0.87
   (320 code) / 0.90 (long code) everywhere. Prose arm 0.523 =
   INCIDENTAL, not of record (out-of-distribution for a coding LLM; the
   harness wraps every prompt in a coding-agent persona regardless).

## Quality arms of record (rq4 + bf16 KV)

| eval | result | source |
|---|---|---|
| MBPP full-974 | 778/972 effective = **0.8004** (greedy seed 42) | `evals/tier2/mbpp_oq2e_rq4_bf16_full974.json` (`612758b`) |
| HumanEval-164 | rq4 **0.8659** (q8 0.8720, McNemar p=1.0) | `evals/tier2/humaneval_oq2e_full164_rq4.json` (`f5a30d8`) |
| MBPP vs q8 | identical (778/972) — rq4 gives up nothing at 2.50 bpw | footnote only |

## kv4 reversal (measured, not assumed)

88e×16k kv4 K-selection: K1 acceptance collapsed to 0.125 (vs bf16 0.898),
every MTP lane lost to AR (K1 16.5 < AR 26.1). bf16 K1 (27.6) beats kv4's
best lane at the same cell. Receipts: `t3_kselect_88e16k_kv4_rep*.json`
(`db46f26`), #130 5047822076.

## Not measured (state honestly if asked)

- kv4 at contexts other than 16k (1024, 320) — single cell only.
- 88×64k (structurally dead >112 at bf16); 32k skipped entirely.
- 48/32 envelopes skipped (scripts prepped `bdf4f8d`, never launched).

## HF publish gates (per [[hy3-hf-publish-plan]]) — status

Suffix `-MTPLX-streaming`; ship full serving root (experts.bin + manifest
+ MTP head). 5 gates: full-suite eval ✓ (MBPP+HumanEval of record),
bundle MTP head (pending), shard hashes (pending), clean-room rehearsal
(pending), license (pending). NOTHING uploaded yet.
