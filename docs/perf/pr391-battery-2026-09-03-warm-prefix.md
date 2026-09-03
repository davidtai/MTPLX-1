# PR 391 battery, warm-prefix run and served sanity check, 2026-09-03

Terms used in this document:

- MTP: multi-token prediction.
- TTFT: time to first token, measured by the client.
- KV: the key and value tensors of attention.
- GQA: grouped-query attention.
- QSA: Qwen Sparse Attention.
- GDN: Gated DeltaNet.
- PLE: per-layer embedding, fed from an n-gram sidecar table.
- FR-Spec: the frequency-ranked draft head over a pruned vocabulary.

This document holds two measurements that are not part of the headline
battery. Both change how two numbers of that battery read. Section 1 is the
warm-prefix run of the branch arm. Section 2 is the served check of the full
key set against the same server with every key off.

---

## 1. Warm-prefix branch run

Do not compare this run to the cold table. The session bank restored prompt
prefixes across cells, so the branch arm did not prefill what the control and
mlx-serve arms prefilled. Its prefill, TTFT and wall figures measure cache
restoration and not prefill throughput. The receipts are quarantined in their
own subdirectory, and the report readers do not recurse into it.

### 1.1 What the bank restored

`new_prefill_tokens` is what the server had to prefill. `cached_tokens` is
what the bank handed back. The values are in seed order: 20260829, 20260830
and 20260831.

| Cell | Prompt tok, per seed | New prefill tok, per seed | Cached tok, per seed | Mean restored |
| --- | --- | --- | --- | ---: |
| vanity | 87, 87, 87 | 87, 87, 87 | 0, 0, 0 | 0.0% |
| 1K | 1,023, 1,024, 1,024 | 1,023, 1,024, 1,024 | 0, 0, 0 | 0.0% |
| 8K | 8,192, 8,192, 8,192 | 8,192, 7,424, 7,424 | 0, 768, 768 | 6.2% |
| 16K | 16,384, 16,384, 16,384 | 8,448, 8,448, 8,448 | 7,936, 7,936, 7,936 | 48.4% |
| 32K | 32,768, 32,768, 32,768 | 20,736, 16,640, 16,640 | 12,032, 16,128, 16,128 | 45.1% |
| 64K | 65,536, 65,536, 65,536 | 33,024, 65,536, 65,536 | 32,512, 0, 0 | 16.5% |
| 128K | 131,072, 131,072 | 131,072, 131,072 | 0, 0 | 0.0% |
| 255K | 261,120 | 130,304 | 130,816 | 50.1% |

Read the 16K, 32K and 255K rows first: **16K prefilled 8,448 of 16,384 tokens on all three seeds** and **255K prefilled 130,304 of 261,120**. The cold run's branch arm prefilled every one of those tokens.

Read the 16K, 32K and 255K rows first. The 16K cell prefilled 8,448 of 16,384
tokens on all 3 seeds, and the 255K cell prefilled 130,304 of 261,120 tokens.
The cold run prefilled every one of those tokens.

### 1.2 Warm-prefix rows beside the cold rows

Both runs use the same renderer, cells, 3 seeds, flag files, profile, server
working directory, engine version and 21 request digests. One difference
remains between the two receipt sets, and it is the one that matters:

```
cold (b3, headline)  cli_env_overrides = {"MTPLX_SESSION_BLOCK_PREFIX_RESTORE": "0",
                                          "MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS": "999999999"}
warm (b2, this run)  cli_env_overrides = {}
```

The upstream and mlx-serve arms carry an empty `cli_env_overrides` in both
reports and restored nothing of their own accord. The branch arm needed the
explicit disable to be measured on the same footing. The control and
mlx-serve columns are byte-identical between the two reports across all 7
metric tables, so only the branch column moves.

**Prefill tok/s**

| Cell | branch, COLD (headline) | branch, WARM PREFIX (not like-for-like) |
| --- | ---: | ---: |
| vanity | 359±24 _n=3_ | 359±24 _n=3_ |
| 1K | 878±73 _n=3_ | 875±68 _n=3_ |
| 8K | 1,161±69 _n=3_ | 1,205±41 _n=3_ |
| 16K | 1,332±25 _n=3_ | 1,506±68 _n=3_ |
| 32K | 1,295±8 _n=3_ | 1,316±20 _n=3_ |
| 64K | 1,262±2 _n=3_ | 1,228±38 _n=3_ |
| 128K | 1,215±0 _n=2_ | 1,216±0 _n=2_ |
| 255K | 1,189 _n=1_ | 1,110 _n=1_ |

**Decode tok/s**

| Cell | branch, COLD (headline) | branch, WARM PREFIX (not like-for-like) |
| --- | ---: | ---: |
| vanity | 100.49 (95.78-104.11) _n=3_ | 100.53 (95.66-104.33) _n=3_ |
| 1K | 86.13 (82.27-93.65) _n=3_ | 86.33 (82.37-94.14) _n=3_ |
| 8K | 82.95 (78.87-86.17) _n=3_ | 82.55 (79.54-86.14) _n=3_ |
| 16K | 80.92 (77.13-83.75) _n=3_ | 79.61 (75.63-83.70) _n=3_ |
| 32K | 77.20 (74.96-78.58) _n=3_ | 78.01 (75.76-80.12) _n=3_ |
| 64K | 73.50 (69.23-79.93) _n=3_ | 76.92 (71.81-79.83) _n=3_ |
| 128K | 73.27 (70.42-76.11) _n=2_ | 73.36 (70.69-76.03) _n=2_ |
| 255K | 65.53 _n=1_ | 66.54 _n=1_ |

**TTFT s**

| Cell | branch, COLD (headline) | branch, WARM PREFIX (not like-for-like) |
| --- | ---: | ---: |
| vanity | 0.263±0.036 _n=3_ | 0.264±0.037 _n=3_ |
| 1K | 1.319±0.095 _n=3_ | 1.322±0.091 _n=3_ |
| 8K | 7.256±0.432 _n=3_ | 6.545±0.535 _n=3_ |
| 16K | 12.495±0.234 _n=3_ | 5.802±0.255 _n=3_ |
| 32K | 25.542±0.156 _n=3_ | 13.891±1.252 _n=3_ |
| 64K | 52.266±0.094 _n=3_ | 44.518±11.414 _n=3_ |
| 128K | 108.402±0.048 _n=2_ | 108.350±0.032 _n=2_ |
| 255K | 220.629 _n=1_ | 118.248 _n=1_ |

**Peak mem GB**

| Cell | branch, COLD (headline) | branch, WARM PREFIX (not like-for-like) |
| --- | ---: | ---: |
| vanity | 84.67±0.06 _n=3_ | 84.67±0.06 _n=3_ |
| 1K | 86.08±0.70 _n=3_ | 85.90±0.45 _n=3_ |
| 8K | 93.53±1.78 _n=3_ | 92.08±1.13 _n=3_ |
| 16K | 98.62±0.73 _n=3_ | 96.89±1.45 _n=3_ |
| 32K | 103.21±1.03 _n=3_ | 100.77±0.02 _n=3_ |
| 64K | 103.94 _n=3_ | 101.75±0.13 _n=3_ |
| 128K | 97.22±0.01 _n=2_ | 97.21±0.01 _n=2_ |
| 255K | 104.22 _n=1_ | 104.11 _n=1_ |

**Peak footprint GB**

| Cell | branch, COLD (headline) | branch, WARM PREFIX (not like-for-like) |
| --- | ---: | ---: |
| vanity | 86.00±0.03 _n=3_ | 86.00±0.03 _n=3_ |
| 1K | 88.20±0.73 _n=3_ | 88.02±0.48 _n=3_ |
| 8K | 97.99±2.11 _n=3_ | 96.43±1.41 _n=3_ |
| 16K | 103.69±1.12 _n=3_ | 102.36±0.94 _n=3_ |
| 32K | 108.24±0.02 _n=3_ | 108.15±0.36 _n=3_ |
| 64K | 108.31±0.02 _n=3_ | 108.28±0.03 _n=3_ |
| 128K | 104.43±1.56 _n=2_ | 103.99±0.27 _n=2_ |
| 255K | 108.42 _n=1_ | 108.34 _n=1_ |

**Prefill s (server)**

| Cell | branch, COLD (headline) | branch, WARM PREFIX (not like-for-like) |
| --- | ---: | ---: |
| vanity | 0.243±0.017 _n=3_ | 0.244±0.017 _n=3_ |
| 1K | 1.174±0.103 _n=3_ | 1.178±0.096 _n=3_ |
| 8K | 7.079±0.436 _n=3_ | 6.391±0.533 _n=3_ |
| 16K | 12.308±0.230 _n=3_ | 5.621±0.264 _n=3_ |
| 32K | 25.305±0.153 _n=3_ | 13.661±1.243 _n=3_ |
| 64K | 51.929±0.090 _n=3_ | 44.186±11.378 _n=3_ |
| 128K | 107.844±0.043 _n=2_ | 107.783±0.037 _n=2_ |
| 255K | 219.629 _n=1_ | 117.394 _n=1_ |

**Wall s** (with the generated-token count that produced it)

| Cell | branch, COLD (headline) | gen tok | branch, WARM PREFIX | gen tok |
| --- | ---: | ---: | ---: | ---: |
| vanity | 2.26 (1.97-2.49) _n=3_ | 199 (178-212) | 2.26 (1.96-2.49) _n=3_ | 199 (178-212) |
| 1K | 12.24 (9.74-13.64) _n=3_ | 932 (793-1,024) | 12.22 (9.71-13.63) _n=3_ | 932 (793-1,024) |
| 8K | 17.67 (14.11-19.75) _n=3_ | 870 (562-1,024) | 16.79 (13.97-19.19) _n=3_ | 850 (619-1,024) |
| 16K | 23.26 (20.25-25.34) _n=3_ | 875 (610-1,024) | 18.69 (17.84-19.19) _n=3_ | 1,024 (1,024-1,024) |
| 32K | 35.57 (31.41-38.87) _n=3_ | 774 (442-1,024) | 25.15 (24.77-25.57) _n=3_ | 875 (685-1,004) |
| 64K | 63.73 (58.53-67.42) _n=3_ | 828 (435-1,024) | 54.75 (39.75-65.26) _n=3_ | 780 (435-1,024) |
| 128K | 120.31 (117.75-122.88) _n=2_ | 808 (592-1,024) | 120.25 (117.68-122.83) _n=2_ | 808 (592-1,024) |
| 255K | 238.97 _n=1_ | 1,020 | 133.36 _n=1_ | 821 |

**Per-seed decode tok/s, branch arm only**

| Cell | COLD per-seed | COLD mean | WARM PREFIX per-seed | WARM PREFIX mean |
| --- | --- | ---: | --- | ---: |
| vanity | 95.78, 104.11, 101.58 | 100.49 | 95.66, 104.33, 101.60 | 100.53 |
| 1K | 82.27, 82.48, 93.65 | 86.13 | 82.37, 82.48, 94.14 | 86.33 |
| 8K | 86.17, 83.81, 78.87 | 82.95 | 86.14, 81.97, 79.54 | 82.55 |
| 16K | 81.88, 77.13, 83.75 | 80.92 | 79.49, 83.70, 75.63 | 79.61 |
| 32K | 78.58, 78.06, 74.96 | 77.20 | 75.76, 80.12, 78.16 | 78.01 |
| 64K | 69.23, 71.34, 79.93 | 73.50 | 79.13, 71.81, 79.83 | 76.92 |
| 128K | 76.11, 70.42 | 73.27 | 76.03, 70.69 | 73.36 |
| 255K | 65.53 | 65.53 | 66.54 | 66.54 |

Decode is the metric the warm bank barely touches: the two runs' branch decode means differ by well under the seed spread at every size. It is prefill, TTFT and wall that the restored prefix moved, which is exactly why the cold re-run was needed before the table could be published.

Decode is the metric the warm bank barely moves: the two branch decode means
differ by well under the seed spread at every size. The restored prefix moves
prefill, TTFT and wall time. That is why the cold run was needed before the
table could be published.

---

## 2. Served full stack against no flags

The battery answers the question "branch against upstream". This section
answers the narrower question "do the 20 keys do anything on the served path".
It runs the same server binary, profile and boot environment twice, and
changes only the flags file.

- FULL is the branch server with 12 decode keys and 8 prefill keys.
- BASE is the same server with an empty flags file, so every non-boot key is off.
- The profile and the boot environment apply to both arms. FR-Spec, the compiled MTP preparation and the 4 auto-armed M4 routes are on in both.
- Cells 1K and 16K, 1,024 generated tokens, 1 seed, the same request body on both arms.

| Cell | Metric | FULL | BASE | delta | delta % |
| --- | --- | ---: | ---: | ---: | ---: |
| 1K | Decode tok/s | 81.19 | 81.87 | -0.68 | -0.83% |
| 1K | Prefill tok/s | 781.5 | 776.6 | +5.0 | +0.64% |
| 1K | TTFT s | 1.438 | 1.458 | -0.019 | -1.32% |
| 1K | Wall s | 13.49 | 10.88 | +2.61 | +23.98% |
| 1K | Peak GB | 84.97 | 84.85 | +0.12 | +0.15% |
| 16K | Decode tok/s | 81.87 | 69.46 | +12.41 | +17.87% |
| 16K | Prefill tok/s | 1298.2 | 1148.2 | +150.0 | +13.07% |
| 16K | TTFT s | 12.796 | 14.465 | -1.669 | -11.54% |
| 16K | Wall s | 25.32 | 29.23 | -3.91 | -13.36% |
| 16K | Peak GB | 89.40 | 91.42 | -2.02 | -2.20% |

Assertions, as the harness printed them:

```
A1  PASS  16K decode tok/s: FULL 81.867 vs BASE 69.458 (floor 68.069 = BASE-2%), +17.87%
A2  PASS  1K prefill tok/s: FULL 781.524 vs BASE 776.556 (floor 761.025 = BASE-2%), +0.64%
A3  PASS  16K prefill tok/s: FULL 1298.204 vs BASE 1148.171 (floor 1125.208 = BASE-2%), +13.07%
A4  PASS  1K decode tok/s (sparse lane routes to stock below ~2,052 selected tokens; tolerance, not a win): FULL 81.192 vs BASE 81.868 (floor 80.231 = BASE-2%), -0.83%

note: branch-base engagement: frspec=True frspec_n=65536 gdn=True ladder_all_ok=True
note: branch-base resolved 0 fable key(s), default_used=False
note: branch-fullstack engagement: frspec=True frspec_n=65536 gdn=True ladder_all_ok=True
note: branch-fullstack resolved 20 fable key(s), default_used=False

SANITY PASS
```

Two rows are not wins, and the harness prints them as such. The 1K decode row
reads -0.83 %, because the sparse decode lane routes to stock attention below
about 2,052 selected tokens. At 1K the tuning set has nothing to do, so that
assertion is a no-regression tolerance. The 1K wall row reads +23.98 %,
because the two arms stopped at different lengths on this temperature-1
request: FULL generated 979 tokens and BASE generated 772 tokens, both with
finish reason `stop`. The 1K decode figure is per token and is flat.

Both arms of both cells ran cold, with `new_prefill_tokens` equal to
`prompt_tokens`: 1,023 tokens at 1K and 16,384 tokens at 16K. Both carried
the same `request_body_sha256` per cell, `bdd51de8b6bf...` at 1K and
`eb463f6ca7bb...` at 16K. Those are the two digests of the battery cells at
1K and 16K for seed 20260829.

An earlier attempt at the same check is a failure and stays in the record. Its
1K FULL arm produced no valid record, so 2 of the 4 assertions read MISSING.
Its 16K figures agree with the passing run to within one seed: decode
+17.73 % and prefill +13.38 %.

### 2.1 Install verdict for each of the 20 keys

Every key that the two flag files name has an install-time verdict from the
preflight. No key is off and no key is refused. `log` is the print in the
server log, and `receipt` is the block in the receipt.

| Key | Value | Verdict | log | receipt | Detail |
| --- | --- | --- | --- | --- | --- |
| `MTPLX_FABLE_BLOCK_VERIFY` | `1` | ARMED | ARMED | ARMED | lane=block_verify; engages_at an accept window with temperature>0, no target prefix; engagements=9 |
| `MTPLX_FABLE_DRAFT_K20_PRESCATTER` | `1` | ARMED | ARMED | ARMED | lane=draft_k20_prescatter; engages_at each request whose draft route the claim binds; engagements=1 |
| `MTPLX_FABLE_GRAPH_BUILD_OVERLAP` | `1` | ARMED | ARMED | - | |
| `MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS` | `3` | ARMED | ARMED | - | |
| `MTPLX_FABLE_HC_M4` | `1` | ARMED | ARMED | ARMED | |
| `MTPLX_FABLE_OPDIET` | `1` | ARMED | ARMED | ARMED | lane=opdiet; engagements=0 |
| `MTPLX_FABLE_PLE_FIRST_GATHER_EARLY` | `1` | ARMED | ARMED | ARMED | lane=ple_first_gather_early; engagements=0; declines={'model_declined_span': 1} |
| `MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD` | `1` | ARMED | ARMED | ARMED | lane=ple_prefill_lookahead; engagements=0; declines={'single_span': 1} |
| `MTPLX_FABLE_PREFILL_MASK_FUSE` | `1` | ARMED | ARMED | - | engaged on a prefill class: q_len 256 >= 9, class [causal-mask q_len 256 x GQA 12 at head_dim 256 bfloat16] (6 MTP-class refusal notes) |
| `MTPLX_FABLE_PREFILL_QSA_QUERY_TILE` | `2048` | ARMED | ARMED | ARMED | lane=prefill_qsa_query_tile; engages_at a prefill chunk wider than 2048 query rows; engagements=0 |
| `MTPLX_FABLE_QSA_SPARSE_DECODE` | `1` | ARMED | ARMED | - | |
| `MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS` | `17` | ARMED | ARMED | - | |
| `MTPLX_FABLE_QSA_SPARSE_DECODE_TILE` | `128:32` | ARMED | ARMED | - | |
| `MTPLX_FABLE_ROUTE_KERNEL` | `1` | ARMED | ARMED | ARMED | |
| `MTPLX_FABLE_VERIFY_GLUE` | `1` | ARMED | ARMED | - | |
| `MTPLX_FABLE_VERIFY_GLUE_ITEMS` | `qsa_rope,qsa_rope_idx` | ARMED | ARMED | - | |
| `MTPLX_GDN_BLOCKED_PREFILL` | `1` | ARMED | ARMED | - | |
| `MTPLX_PREFILL_CHUNK_SIZE` | `4096` | RESOLVED | RESOLVED | RESOLVED | lane=prefill_chunk_size; engagements=0 |
| `MTPLX_QSA_PREFILL_COMPILE_ROWS` | `4096` | RESOLVED | RESOLVED | RESOLVED | lane=qsa_prefill_compile_rows; engagements=0 |
| `MTPLX_SESSION_BANK_MAX_BYTES` | `8G` | RESOLVED | RESOLVED | RESOLVED | lane=session_bank_max_bytes; engagements=0 |

```
resolved=20 armed=17 resolved-knob=3 no-refusal=0 unprovable=0 fatal=0
fable_install_receipts: 9 lane(s) -> 10 key(s) mapped
harness markers (scan_engagement): {"compiled_mtp_prepare": true, "frspec_disabled": false, "frspec_installed": true, "gdn_blocked_prefill": true, "m4_fixed_verify": true, "m4_stage3": true, "prefill_chunk_override": true, "qsa_gather": false, "spec_stats": false}
frspec n=65536 ladder_all_ok=True logger_handler_installed=True
preflight problems: none

STACK OK: every key the two files name has an install-time verdict and none is off/refused (17 armed, 3 resolved-knob, 0 unprovable)
```

The quoted block above keeps the preflight output unchanged, except for one
internal task label removed from the receipts line.

Three findings of the preflight belong beside the table:

- Two lanes are armed but inert at startup. `MTPLX_FABLE_PLE_FIRST_GATHER_EARLY` declines with `model_declined_span`, and `MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD` declines with `single_span`. Both recorded 0 engagements on the warmup request, because neither condition existed there. Read the battery cell receipts before crediting either lane.
- The mask fuse refuses the MTP verify classes, at query lengths 3 to 6 with GQA 12. The vector-attention kernel of MLX requires query length times GQA of 32 or less. The refusal is per class, so the prefill classes are unaffected.
- The full-stack self-check reports `ok=False` at `phase=startup`, and the only unmet marker is `ladder_all_ok`. The scan of the finished log reads `ladder_all_ok=True` with both rungs at context 512 and context 2,560 valid. The startup snapshot predates the background ladder, so the stack is not implicated.
