# Qwen 3.8 Challenge Port: 54-Proposal Ledger

This is the authoritative campaign state for the 54 chronological Yukon rows
whose relative score improvement is strictly greater than 0.10%. It replaces
the invalid eight-bundle ledger. A later descendant, composite, or family gate
does not prove an earlier proposal. Every row receives its own exact source
diff review and one of: an individual 16K receipt, exact already-present proof,
target-shape no-op proof, or removed-later proof.

## Fixed benchmark and stacking contract

- Base: upstream Optimized-Speed `main` at
  `bd4421567f9e16ce957c6ef97708b072dcd73937` (v2.9.1), never stock MLX.
- Runtime: Turbo, compiled verify, packed GQA, Q4/group-64 draft head,
  persistent committed history, `capture_commit`, and
  `linear-gdn-from-conv-tape`.
- Workload: exactly 16,384 Python tokens assembled from `mtplx/generation.py`
  plus the intact `python_modules_long.jsonl` instruction tail; 1,024 output;
  target/draft temperature 1.0; top-p 0.95; top-k 20; seed 42; fixed D3.
- Timing: one conditioning generation per route, then exactly four timed ABBA
  arms under `/tmp/mtplx-gpu-exclusive.lock`.
- Promotion: strict matched wall-time improvement greater than 0.05%.
  Deterministic tie-breaking drift is recorded but is not a rejection reason.
- Stacking: proposal N's control is Optimized-Speed plus every retained row
  before N. A retained candidate becomes the next control. Percentages are
  never added or multiplied.
- Source: pinned challenge checkout
  `eb5eadc7a165047d4321ce883b9ff30894d8bd19`. The inventory test resolves and
  hashes each row's exact parent diff.

## Current measured results

Means are two timed arms per route; peak is the maximum timed arm.

| Row | Route | Prefill tok/s | Decode tok/s | Peak GiB | Mean wall s | Delta | Decision | Receipt |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 3 | Optimized-Speed main | 754.031 | 51.803 | 25.569 | 42.006 | - | control | `c1-packed-qkv-corrected-python16384in-1024out-t1-abba-2026-08-23.json` |
| 3 | + packed target Q/K/V | 761.534 | 46.971 | 25.569 | 43.785 | **-4.0629%** | REJECTED, removed | same |
| 8 | Optimized-Speed main + rows 2-7 dispositions | 750.099 | 52.478 | 24.885 | 41.870 | - | cumulative control | `chrono-r08-device-draft-python16384in-1024out-t1-abba-2026-08-23.json` |
| 8 | + device-resident fixed-D3 draft chain | 779.772 | 54.038 | 24.885 | 40.413 | **+3.6049%** | **RETAINED** | same |
| 9 | Optimized-Speed main + retained row 8 | 744.715 | 52.555 | 24.885 | 41.977 | - | cumulative control | `chrono-r09-paired-qmv-g32-m4-on-r08-python16384in-1024out-t1-abba-2026-08-23.json` |
| 9 | + paired G32/M4 target QMV | 761.698 | 47.465 | 24.885 | 43.651 | **-3.8350%** | REJECTED, remove | same |
| 10 | Optimized-Speed main + retained row 8 | 737.536 | 54.151 | 25.149 | 41.640 | - | cumulative control | `chrono-r10-compact-vocab-on-r08-python16384in-1024out-t1-abba-2026-08-23.json` |
| 10 | + compact proposal vocabulary | 765.005 | 55.240 | 25.149 | 40.411 | **+3.0411%** | **RETAINED** | same |
| 13 | Optimized-Speed main + retained rows 8, 10 | 765.003 | 56.424 | 27.507 | 40.066 | - | cumulative control | `chrono-r13-gdn-inproj-s9-on-r08-r10-python16384in-1024out-t1-abba-2026-08-23.json` |
| 13 | + four-way GDN input projection through S=9 | 778.485 | 54.008 | 27.507 | 40.469 | **-0.9949%** | REJECTED, removed | same |
| 18 | Optimized-Speed main + retained rows 8, 10 | 742.390 | 54.957 | 25.14946 | 41.190 | - | cumulative control | `chrono-r18-gdn-decay-memo-on-r08-r10-python16384in-1024out-t1-abba-2026-08-23.json` |
| 18 | + per-layer GDN `-exp(A_log)` memo | 758.449 | 54.816 | 25.14946 | 40.743 | **+1.0970%** | **RETAINED** | same |
| 18 addendum | retained row-18 decay control | 720.656 | 54.118 | 32.70319 | 42.161 | - | cumulative control with packed weights resident | `chrono-r18-mlp-gate-up-addendum-on-r08-r10-r18memo-python16384in-1024out-t1-abba-2026-08-23.json` |
| 18 addendum | + S≤9 packed MLP gate/up | 729.819 | 52.624 | 32.70319 | 42.388 | **-0.5360%** | REJECTED, removed | same |
| 20 | Optimized-Speed main + retained rows 8, 10, 18 | 744.888 | 56.926 | 25.16900 | 40.497 | - | cumulative control | `chrono-r20-kv-only-history-on-r08-r10-r18-python16384in-1024out-t1-abba-2026-08-23.json` |
| 20 | + packed K/V-only committed-history append | 780.840 | 54.742 | 25.16900 | 39.793 | **+1.7697%** | **RETAINED** | same |
| 21 | Optimized-Speed main + retained rows 8, 10, 18, 20 | 733.350 | 53.475 | 25.16899 | 41.645 | - | cumulative control | `chrono-r21-qk-rms-rope-on-r08-r10-r18-r20-python16384in-1024out-t1-abba-2026-08-23.json` |
| 21 | + fused Q/K RMSNorm + partial RoPE | 726.746 | 52.912 | 25.16899 | 42.018 | **-0.8870%** | REJECTED, removed | same |
| 24 | Optimized-Speed main + retained rows 8, 10, 18, 20 | 745.937 | 55.524 | 25.16900 | 40.554 | - | cumulative control | `chrono-r24-eval-ladder-on-r08-r10-r18-r20-python16384in-1024out-t1-abba-2026-08-23.json` |
| 24 | + target trunk evaluation ladder | 778.710 | 54.530 | 25.23538 | 39.915 | **+1.6019%** | **RETAINED** | same |

Row 8's source-only four-way GDN projection fuse is separately proven a
target-shape no-op: its `S <= 2` eligibility never fires in the fixed-D3
`S = 4` verify. Two diagnostic ABBA runs are preserved with zero fused-call
engagement and are explicitly invalidated rather than interpreted as timing
results. Row 13's later `S <= 9` expansion is the first applicable form.

Rows 8, 10, 18, 20, and 24 are therefore part of every later timed control. Row 9 regressed
despite 82,880 timed kernel engagements and remains absent. Row 13 executed
7,104 fused four-way projection calls in each timed candidate arm and was
deterministic within route; its loss is measured, not a no-op or parity veto.

Row 20's first diagnostic ABBA is preserved under an `invalid-` receipt name.
It measured the dead-output K/V-only path but showed `packed_calls=0` because
the live MTP attention K/V projections are BF16 rather than quantized. That
receipt is not promotion evidence; the corrected gate requires both K/V-only
history calls and the source patch's packed K+V projection to execute.

## Exact 54-row campaign state

`Patch` is the SHA-256 prefix of the full binary parent diff; `Stats` is that
same exact diff. `PENDING` means no disposition has been inferred from a later
row or from the earlier bundle campaign.

| Row | PR | Yukon delta | Source | Patch | Stats | Individual status |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 2 | 13 | 23.2995% | `97921a3fc5dd` | `bbabc5b7c18a` | +83/-26 | ALREADY PRESENT: checkpoint/rejection fast path is generalized by capture/commit; source D1 policy is superseded by fixed D3. Exact engagement proof pending. |
| 3 | 18 | 1.5662% | `c9e32f70dac8` | `758626e5d0ea` | +47/-3 | REJECTED at -4.0629%; individual receipt above. |
| 4 | 24 | 1.2868% | `ec0ba7d9ce42` | `2c67706857e2` | +542/-21 | ALREADY PRESENT: seed-tail logits and lazy capture/commit boundary are broader in main. Exact engagement proof pending. |
| 5 | 29 | 6.7162% | `ab62ceab428a` | `158a74067412` | +200/-7 | ALREADY PRESENT/ADAPTED: exact device top-k20 sampling supersets source top-2 for the fixed stochastic shape. Exact proof pending. |
| 6 | 37 | 1.7673% | `5c2441b5f08b` | `061942094f64` | +10/-8 | TARGET-SHAPE NO-OP: there is no second target argmax consumer to reuse under top-k20 sampling. Exact proof pending. |
| 7 | 38 | 20.4283% | `fe8829244cd9` | `9e25f5798c47` | +227/-22 | ALREADY PRESENT: persistent committed MTP history. Row-8 receipt exercises it; focused proof pending. |
| 8 | 41 | 17.9183% | `11670086c1b9` | `52c2ac2b4934` | +901/-822 | RETAINED in part: device-resident D3 draft-chain adaptation, +3.6049%. Its four-way GDN fuse is a fixed-shape no-op because source limits it to S<=2 while this campaign verifies S=4; both diagnostic receipts record zero fused calls. |
| 9 | 55 | 6.8764% | `b6c725144b56` | `9193949c4c87` | +340/-0 | REJECTED: adjacent-row shared-weight QMV adapted from source G64 to live target G32/M4; parity passed, but 16K wall regressed 3.8350% on row 8. |
| 10 | 59 | 5.3467% | `61936f26547d` | `c44c6fd53fb6` | +89/-7 | RETAINED: compact Q4/group-64 proposal vocabulary (98,330 reachable rows, padded to 98,336) with on-device target-ID mapping; exact parity and +3.0411% on retained row 8. |
| 11 | 63 | 5.7425% | `62174dbbca88` | `40a33f553244` | +230/-16 | TARGET-SHAPE NO-OP/COVERED: adaptive depth is disabled by fixed D3, exact 16K conditioning warms the seed shape, and retained row 8 has no host-built first draft to early-flush. Focused evidence receipt pending. |
| 12 | 70 | 0.8266% | `09eda55a08b1` | `96bb2be6fbe1` | +304/-24 | REMOVED NEXT ROW: row 13 deletes the prefix-replay tape and restores eager per-boundary checkpoints; never enters the cumulative target stack. |
| 13 | 71 | 2.0944% | `3e157ad981bb` | `83215ffbd861` | +25/-305 | REJECTED: source S<=9 four-way GDN input fusion was adapted exactly to live group-32 fixed-D3 S=4 verification, engaged 7,104 times per timed arm, and regressed wall throughput 0.9949% on retained rows 8+10. Deterministic tie drift was allowed; implementation removed. Prefix-replay removal is moot because row 12 was not retained. |
| 14 | 77 | 0.9839% | `d81964127281` | `0b3dba1ea446` | +304/-24 | ALREADY PRESENT: source reintroduces recurrent prefix replay for wide verifies; Optimized-Speed already runs the broader target `capture_commit` + `linear-gdn-from-conv-tape` boundary. Row-18 receipt directly records that strategy/core and 47.6 ms mean nonzero capture-commit work in both control and candidate arms. |
| 15 | 95 | 3.7597% | `08897af24b57` | `8e803fafd868` | +216/-53 | TARGET-SHAPE NO-OP: every source code path is gated to verify widths 6...9 or raises an adaptive depth cap. This campaign is fixed D3, hence target verify S=4; widths 2...5 are explicitly unchanged by the source patch. |
| 16 | 103 | 0.5808% | `8f41fa6d4f67` | `114e6ca13e03` | +149/-32 | ALREADY PRESENT: source separately compiles the GDN g/beta elementwise prologue and gated post-norm. Optimized-Speed compiles the entire route-specific S=4 target capture graph, which contains both regions; a nested subcompile adds no missing dispatch boundary. |
| 17 | 126 | 7.5460% | `deb63ad0d170` | `2dbcb36ee10e` | +6/-14 | ALREADY PRESENT: manifest-only switch from BF16 to Q4/group-64 proposal head. Optimized-Speed loads Q4/group-64 before every arm, and retained row 10 further reduces it to the reachable compact vocabulary. |
| 18 | 135 | 0.4535% | `b6ce964b16bb` | `2181386c97ac` | +324/-247 | RETAINED in part: adaptive depth, S>=6 attention segmentation, and synthetic 512-token warming are inapplicable to fixed D3/exact 16K conditioning. The adapted per-layer input-independent `-exp(A_log)` GDN memo was exact, engaged 6,912 times per timed arm, and improved wall throughput 1.0970% on rows 8+10. A chronology audit found the source also expands row 8's S<=2 packed MLP gate/up to S<=9; the corrected addendum engaged 9,628 times per candidate arm but regressed 0.5360%, so that component is removed. |
| 19 | 160 | 2.5391% | `1033e1ac5197` | `1a4f47311818` | +581/-97 | TARGET-SHAPE NON-TRANSFERABLE: source replaces compact projection + argmax + target-ID mapping with an argmax-only selector. The required temperature-1/top-k20 route must retain the complete sparse proposal distribution for exact speculative acceptance; substituting argmax changes the sampling law. Row 10 already retains the source's compact Q4/group-64 projection and on-device mapping. |
| 20 | 180 | 0.9180% | `cf350293feb4` | `b9b4300e973d` | +144/-8 | RETAINED: adapted to MTPLX's dead committed-history outputs by omitting Q/gate/attention/MLP for every appended row and packing the live BF16 K+V projections into one matmul. The packed path engaged 288 times per timed candidate arm over 17,125 history tokens and improved wall throughput 1.7697% on rows 8+10+18. Deterministic tie drift was allowed. |
| 21 | 186 | 1.5222% | `4eb54489fb51` | `df0b66eded6c` | +228/-5 | REJECTED: exact Qwen 3.8 BF16 H256/R64 fused Q/K RMSNorm + partial-RoPE engaged 2,416 times per timed candidate arm on rows 8+10+18+20, but regressed wall throughput 0.8870%. Deterministic tie drift was allowed; implementation removed. |
| 23 | 215 | 0.2964% | `df404e08fee2` | `597330a384fb` | +64/-41 | DEPENDENCY ABSENT/TARGET-SHAPE NON-TRANSFERABLE: this patch only retunes the row-19 argmax-only compact selector's reduction. Row 19 is absent because temperature-1/top-k20 acceptance requires the complete sparse proposal distribution, so row 23 has no retained call site and cannot affect this route. |
| 24 | 234 | 0.9658% | `7351e62674bc` | `849631b545f2` | +54/-14 | RETAINED in part: adaptive-margin depth is disabled by fixed D3; the Q/K width restriction modifies rejected row 21; and fused SwiGLU depends on row 18's rejected packed MLP gate/up. The remaining live mechanism, the source's target trunk evaluation ladder, engaged 144 times per candidate arm and improved wall throughput 1.6019% on rows 8+10+18+20 with exact tokens and schedules. |
| 25 | 270 | 0.5421% | `c7468c565a7c` | `e8898ba2afd6` | +1/-1 | PENDING |
| 26 | 276 | 0.1799% | `033f622755ac` | `47dc8c6d9b36` | +14/-6 | PENDING |
| 28 | 304 | 0.2525% | `6209702fba83` | `a6d69403cda0` | +6/-10 | PENDING |
| 30 | 350 | 0.4202% | `32b94cb67d2f` | `948f58d0f63b` | +120/-9 | PENDING |
| 32 | 365 | 0.1764% | `156b5b75bdfa` | `66b436ee06e7` | +58/-24 | PENDING |
| 33 | 401 | 1.7181% | `cbdc3a8d5fa9` | `9cd8e978d00a` | +92/-7 | PENDING |
| 34 | 405 | 0.6815% | `79683c633b13` | `aa0820c6217c` | +65/-114 | PENDING |
| 36 | 423 | 1.6826% | `ed4dfd6b0e95` | `12afdfd18be8` | +105/-40 | PENDING |
| 37 | 428 | 0.1477% | `be3361b96875` | `882e395797e2` | +33/-13 | PENDING |
| 38 | 430 | 0.6244% | `0824e0ec28e5` | `dc1e16093bb7` | +2/-2 | PENDING |
| 39 | 437 | 0.3030% | `1abe6368a882` | `2f3e81092f41` | +13/-33 | PENDING |
| 40 | 438 | 0.6643% | `d1530a409848` | `7a4e9cbc4c6b` | +4/-4 | PENDING |
| 41 | 450 | 0.4931% | `0d800b229e94` | `3272565fbdc7` | +6/-6 | PENDING |
| 42 | 472 | 1.4130% | `036fd9ca2a2c` | `be0fefb19a14` | +124/-9 | PENDING |
| 45 | 505 | 0.4408% | `868cde8f985a` | `3138ecedd936` | +309/-48 | PENDING |
| 47 | 530 | 0.5687% | `dccba745af5b` | `e89a06dfd673` | +195/-5 | PENDING |
| 48 | 543 | 0.1230% | `86fb1f020fc1` | `d2962993b6da` | +422/-240 | PENDING |
| 50 | 572 | 0.2736% | `c0e34afd857e` | `4b6eb22f8820` | +115/-0 | PENDING |
| 53 | 600 | 0.1577% | `0c90733d383f` | `39b6322daa32` | +11/-24 | PENDING |
| 59 | 843 | 1.4217% | `3e2530aeae21` | `a0b5e9aaa3c4` | +23/-2 | PENDING |
| 60 | 846 | 0.2224% | `88578f929552` | `c529a4989d0d` | +155/-27 | PENDING |
| 61 | 866 | 0.2836% | `8b54ff11c6d6` | `feeffa289cd4` | +129/-10 | PENDING |
| 63 | 911 | 0.1698% | `61612aa89dc6` | `0fd574d04a01` | +384/-14 | PENDING: this is a real fused quantized embedding/dual-norm/top-32 patch, not a resample. |
| 66 | 965 | 0.3080% | `ca0612472eb5` | `ddfede29ee90` | +1/-1 | PENDING |
| 67 | 968 | 0.3523% | `41bad1c6f124` | `76fa838f3cf2` | +93/-90 | PENDING |
| 69 | 1031 | 0.2133% | `fac135f222f9` | `863c65d8ae0b` | +412/-746 | PENDING |
| 70 | 1063 | 3.9125% | `6f1cd66fc214` | `0dd6cffb1309` | +1070/-496 | PENDING |
| 71 | 1066 | 0.7439% | `a0f8588668c6` | `8b0283f1e500` | +267/-11 | PENDING |
| 78 | 1123 | 4.3907% | `8849fad72cc3` | `53cc0c1c6e42` | +39/-14 | PENDING |
| 79 | 1130 | 0.2444% | `1d66bb36cda1` | `503f4fe8d56b` | +7/-10 | PENDING |
| 80 | 1139 | 0.3477% | `e8f14c444156` | `49484169aabc` | +8/-4 | PENDING |
| 82 | 1153 | 0.3733% | `eb5eadc7a165` | `3142edaa4070` | +5/-27 | PENDING |

The campaign remains incomplete until every `PENDING` cell is replaced by
direct evidence and every applicable retained implementation has its own 16K
receipt on the cumulative stack.
