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
  Process-latched environment candidates use one conditioning generation in
  each isolated arm process; the timed comparison remains exactly four ABBA
  arms, and every process loads the same cumulative stack.
- Promotion: strict matched wall-time improvement greater than 0.05%.
  Deterministic tie-breaking drift is recorded but is not a rejection reason.
- Stacking: proposal N's control is Optimized-Speed plus every retained row
  before N. A retained candidate becomes the next control. Percentages are
  never added or multiplied.
- Source: pinned challenge checkout
  `eb5eadc7a165047d4321ce883b9ff30894d8bd19`. The inventory test resolves and
  hashes each row's exact parent diff.

Adaptive MTP is now also required. Fixed-D3 receipts remain valid for mechanisms
that engage at D3, but a fixed-D3 no-op is no longer a final skip. Rows 11, 15,
18, 24, 25, 26, 32, 34, 36, 37, 38, 40, and 47 are reopened for surviving
adaptive-policy or newly reachable M5--M9 mechanisms. They will be tested
chronologically only after item 55 has replaced/merged the drafter with
DFlash2. The adaptive stack will start from that merged target/drafter and use
the DFlash2-supported depth range 1--8; changes genuinely removed by later rows
remain removed.

Custom MTP mechanisms are in scope as well: proposal-block variants, committed
history, rollback/warm paths, custom readouts, and shape-specific kernels must
be ported or given concrete already-present/removed/incompatible evidence. A
generic "custom path" label is not a skip reason.

## Current measured results

Means are two timed arms per route; peak is the maximum timed arm.

Chronology correction: row 17's declared Q4/group-64 artifact is
the complete one-layer MTP block, while row 10 quantizes only the separate
draft vocabulary projection. The earlier `ALREADY PRESENT` disposition
conflated those surfaces. Its corrected gate retained the block, so every old
row-18-and-later receipt below is historical and must be rerun on the corrected
cumulative control before that later decision can remain authoritative.

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
| 17 | Optimized-Speed main + retained rows 8, 10 | 750.137 | 56.805 | 25.37195 | 40.377 | - | cumulative control | `chrono-r17-q4-mtp-block-on-r08-r10-python16384in-1024out-t1-abba-2026-08-23.json` |
| 17 | + exact declared Q4/group-64 MTP block | 781.155 | 55.766 | 25.37197 | 39.808 | **+1.4299%** | **RETAINED** | same |
| 18 corrected | Optimized-Speed main + retained rows 8, 10, 17 | 736.928 | 56.265 | 25.37198 | 40.978 | - | corrected cumulative control | `chrono-r18-gdn-decay-memo-on-r08-r10-r17-python16384in-1024out-t1-abba-2026-08-23.json` |
| 18 corrected | + per-layer GDN `-exp(A_log)` memo | 763.072 | 54.971 | 25.37199 | 40.579 | **+0.9843%** | **RETAINED** | same |
| 18 addendum corrected | retained rows 8, 10, 17, 18-memo | 711.992 | 53.860 | 32.92571 | 42.569 | - | cumulative control with packed weights resident | `chrono-r18-mlp-gate-up-addendum-on-r08-r10-r17-r18memo-python16384in-1024out-t1-abba-2026-08-23.json` |
| 18 addendum corrected | + S≤9 packed MLP gate/up | 725.433 | 52.206 | 32.92571 | 42.702 | **-0.3108%** | REJECTED, removed | same |
| 20 corrected | Optimized-Speed main + retained rows 8, 10, 17, 18 | 735.167 | 55.625 | 25.37749 | 41.230 | - | corrected cumulative control | `chrono-r20-kv-only-history-on-r08-r10-r17-r18-python16384in-1024out-t1-abba-2026-08-23.json` |
| 20 corrected | + packed K/V-only committed-history append | 758.616 | 54.778 | 25.37749 | 40.392 | **+2.0739%** | **RETAINED** | same |
| 21 corrected | Optimized-Speed main + retained rows 8, 10, 17, 18, 20 | 720.881 | 54.455 | 25.37751 | 41.661 | - | corrected cumulative control | `chrono-r21-qk-rms-rope-on-r08-r10-r17-r18-r20-python16384in-1024out-t1-abba-2026-08-23.json` |
| 21 corrected | + fused Q/K RMSNorm + partial RoPE | 737.392 | 54.433 | 25.37751 | 41.132 | **+1.2860%** | **RETAINED** | same |
| 24 corrected | Optimized-Speed main + retained rows 8, 10, 17, 18, 20, 21 | 724.633 | 55.175 | 25.37750 | 41.303 | - | corrected cumulative control | `chrono-r24-full-on-r08-r10-r17-r18-r20-r21-python16384in-1024out-t1-abba-2026-08-23.json` |
| 24 corrected | + Q/K L≤16 bound and target evaluation ladder | 740.452 | 54.674 | 25.44388 | 40.958 | **+0.8435%** | **RETAINED** | same |
| 26 corrected | Optimized-Speed main + retained rows 8, 10, 17, 18, 20, 21, 24 | 753.929 | 56.866 | 25.44388 | 39.885 | - | corrected cumulative control | `chrono-r26-full-on-r08-r10-r17-r18-r20-r21-r24-python16384in-1024out-t1-abba-2026-08-23.json` |
| 26 corrected | + three-layer prefill evaluation cadence | 774.419 | 57.018 | 25.44388 | 39.226 | **+1.6797%** | **RETAINED** | same |
| 28 corrected | retained rows 8, 10, 17, 18, 20, 21, 24, 26 | 778.698 | 56.396 | 25.67190 | 39.299 | - | corrected cumulative control | `chrono-r28-full-on-r08-r10-r17-r18-r20-r21-r24-r26-python16384in-1024out-t1-abba-2026-08-23.json` |
| 28 corrected | replace row-17 block with alternate Q4/group-64 artifact | 783.596 | 55.874 | 25.67190 | 39.332 | **-0.0861%** | REJECTED, row 17 retained | same |
| 36 corrected | retained rows 8, 10, 17, 18, 20, 21, 24, 26 | 772.342 | 56.640 | 25.70974 | 39.399 | - | corrected cumulative control | `chrono-r36-full-on-r08-r10-r17-r18-r20-r21-r24-r26-python16384in-1024out-t1-abba-2026-08-23.json` |
| 36 corrected | replace row-17 block with Q4/group-64 + BF16 Q/K/V islands | 782.572 | 55.929 | 25.70974 | 39.341 | **+0.1479%** | **RETAINED**, replaces row 17 artifact | same |
| 48 corrected | retained rows 8, 10, 18, 20, 21, 24, 26, 36 | 735.798 | 55.947 | 25.48171 | 40.703 | - | corrected cumulative control | `chrono-r48-full-on-r08-r10-r18-r20-r21-r24-r26-r36-python16384in-1024out-t1-abba-2026-08-23.json` |
| 48 corrected | + fused residual/RMSNorm boundary chain | 760.347 | 54.792 | 25.48171 | 40.333 | **+0.9175%** | **RETAINED** | same |
| 50 corrected | retained rows 8, 10, 18, 20, 21, 24, 26, 36, 48 | 754.316 | 56.472 | 25.48171 | 39.990 | - | corrected cumulative control | `chrono-r50-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-python16384in-1024out-t1-abba-2026-08-23.json` |
| 50 corrected | + post-warm active-footprint wired residency | 771.058 | 55.787 | 25.48171 | 39.661 | **+0.8299%** | **RETAINED** | same |
| 53 corrected | retained rows 8, 10, 18, 20, 21, 24, 26, 36, 48, 50 | 754.872 | 55.414 | 25.48171 | 40.247 | - | isolated-process cumulative control | `chrono-r53-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-r50-python16384in-1024out-t1-abba-isolated-2026-08-23.json` |
| 53 corrected | + process-latched 512 MiB / 50-op command buffers | 754.912 | 55.979 | 32.72811 | 40.059 | **+0.4708%** | **RETAINED**, +7.24641 GiB peak | same |
| 61 corrected | retained rows 8, 10, 18, 20, 21, 24, 26, 36, 48, 50, 53 | 742.826 | 55.991 | 32.72812 | 40.438 | - | corrected cumulative control | `chrono-r61-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-r50-r53-python16384in-1024out-t1-abba-rerun-2026-08-23.json` |
| 61 corrected | + fused dual RMSNorm and output concatenate | 765.844 | 55.094 | 32.72812 | 40.037 | **+0.9996%** | **RETAINED** | same |
| 63 corrected | retained rows 8, 10, 18, 20, 21, 24, 26, 36, 48, 50, 53, 61 | 754.687 | 57.141 | 32.72812 | 39.698 | - | corrected cumulative control | `chrono-r63-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-r50-r53-r61-python16384in-1024out-t1-abba-2026-08-23.json` |
| 63 corrected | replace separate row-61 kernel with fused Q8 embedding/norm/concat | 768.177 | 55.700 | 32.72812 | 39.773 | **-0.1895%** | REJECTED, row 61 retained | same |
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
| 26 | Optimized-Speed main + retained rows 8, 10, 18, 20, 24 | 751.669 | 55.693 | 25.23538 | 40.325 | - | cumulative control | `chrono-r26-prefill-ladder3-on-r08-r10-r18-r20-r24-python16384in-1024out-t1-abba-2026-08-23.json` |
| 26 | + three-layer prefill evaluation cadence | 781.657 | 55.078 | 25.23538 | 39.650 | **+1.7032%** | **RETAINED** | same |

Row 8's source-only four-way GDN projection fuse is separately proven a
target-shape no-op: its `S <= 2` eligibility never fires in the fixed-D3
`S = 4` verify. Two diagnostic ABBA runs are preserved with zero fused-call
engagement and are explicitly invalidated rather than interpreted as timing
results. Row 13's later `S <= 9` expansion is the first applicable form.

Rows 8, 10, 18, 20, 21, 24, 26, 36, 48, 50, 53, and 61 are therefore part of every corrected
later timed control; row 36 supersedes row 17's artifact while preserving its
Q4/group-64 base tensors. Row 26's corrected candidate engaged 176 times per timed
arm and cleared the gate by 1.6797%. Row 9 regressed
despite 82,880 timed kernel engagements and remains absent. Row 13 executed
7,104 fused four-way projection calls in each timed candidate arm and was
deterministic within route; its loss is measured, not a no-op or parity veto.

Row 20's first diagnostic ABBA is preserved under an `invalid-` receipt name.
It measured the dead-output K/V-only path but showed `packed_calls=0` because
the live MTP attention K/V projections are BF16 rather than quantized. That
receipt is not promotion evidence; the corrected gate requires both K/V-only
history calls and the source patch's packed K+V projection to execute.

## Post-54 final stacked candidate

After every one of the 54 Yukon rows has a final disposition, merge the DFlash
2 logic as the campaign's final candidate. It must be adapted to the resulting
Qwen 3.8 target/MTP shape, stacked on every retained Yukon winner, and measured
with the same exact 16K Python, 1,024-output, four-arm guarded gate. Its result
gets its own receipt and metrics-table rows in this same PR; it does not replace
or retroactively bundle any of the 54 individual decisions. Item 55 is an
assembled DFlash stack, not a bare transplant: every retained target-side win
stays enabled, and every compatible survivor is separately adapted and gated
on both the Qwen target path and the DFlash2 drafter path. Every retained
proposer/custom-kernel win receives an explicit DFlash-equivalent adaptation or
concrete evidence that DFlash replaces or cannot dispatch that exact surface;
no survivor is considered covered merely because the base DFlash integration
works. The scheduled source
is the local `dflash-mlx` implementation at `54644e991039` and its declared
`z-lab/Qwen3.8-27B-DFlash2` checkpoint snapshot `50307d4c4cde`; implementation
will pin the complete artifact digest and geometry before the final gate rather
than silently following either moving ref. After the fixed DFlash2 stack and
its per-survivor gates are complete, adaptive DFlash2 depth 1--8 becomes the
next cumulative lane on the same branch and in the same PR.

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
| 11 | 63 | 5.7425% | `62174dbbca88` | `40a33f553244` | +230/-16 | STAGED FOR ADAPTIVE MTP: MTPLX's existing streak, expected-value, and wall-cost policies do not implement this source schedule. The exact per-position conditional-acceptance EMA, 0.15 update, 0.20 marginal head-cost rule, optimism transfer, hard depth cap, and true depth-0 target-only skip are now ported as the opt-in `position_ema` policy. Serial skips preserve committed-history transitions in a deferred backlog rather than executing the proposal head. Its chronological adaptive 16K gate remains pending. Seed warming and row-8 early-flush portions remain covered by the common conditioner/retained stack. |
| 12 | 70 | 0.8266% | `09eda55a08b1` | `96bb2be6fbe1` | +304/-24 | REMOVED NEXT ROW: row 13 deletes the prefix-replay tape and restores eager per-boundary checkpoints; never enters the cumulative target stack. |
| 13 | 71 | 2.0944% | `3e157ad981bb` | `83215ffbd861` | +25/-305 | REJECTED: source S<=9 four-way GDN input fusion was adapted exactly to live group-32 fixed-D3 S=4 verification, engaged 7,104 times per timed arm, and regressed wall throughput 0.9949% on retained rows 8+10. Deterministic tie drift was allowed; implementation removed. Prefix-replay removal is moot because row 12 was not retained. |
| 14 | 77 | 0.9839% | `d81964127281` | `0b3dba1ea446` | +304/-24 | ALREADY PRESENT: source reintroduces recurrent prefix replay for wide verifies; Optimized-Speed already runs the broader target `capture_commit` + `linear-gdn-from-conv-tape` boundary. Row-18 receipt directly records that strategy/core and 47.6 ms mean nonzero capture-commit work in both control and candidate arms. |
| 15 | 95 | 3.7597% | `08897af24b57` | `8e803fafd868` | +216/-53 | REOPENED FOR ADAPTIVE MTP: its verify-width 6...9 paths and raised adaptive cap become reachable and require a real adaptive-stack disposition. |
| 16 | 103 | 0.5808% | `8f41fa6d4f67` | `114e6ca13e03` | +149/-32 | ALREADY PRESENT: source separately compiles the GDN g/beta elementwise prologue and gated post-norm. Optimized-Speed compiles the entire route-specific S=4 target capture graph, which contains both regions; a nested subcompile adds no missing dispatch boundary. |
| 17 | 126 | 7.5460% | `deb63ad0d170` | `2dbcb36ee10e` | +6/-14 | RETAINED: exact declared Q4/group-64 complete MTP block, distinct from row 10's compact Q4 draft vocabulary projection. The artifact is pinned by source revision, manifest digest, file digest, byte count, tensor set, shape, and dtype; it improved wall throughput 1.4299% on rows 8+10. Both routes were deterministic; proposal/acceptance tie drift was allowed. |
| 18 | 135 | 0.4535% | `b6ce964b16bb` | `2181386c97ac` | +324/-247 | RETAINED in part on corrected rows 8+10+17: adaptive depth, S>=6 attention segmentation, and synthetic 512-token warming are inapplicable to fixed D3/exact 16K conditioning. The adapted per-layer input-independent `-exp(A_log)` GDN memo was exact, engaged 7,584 times per timed arm, and improved wall throughput 0.9843%. The source's S<=9 packed MLP gate/up addendum engaged 9,887 times per candidate arm but regressed 0.3108% on the corrected stack, so it is removed. Deterministic proposal/acceptance drift was allowed. |
| 19 | 160 | 2.5391% | `1033e1ac5197` | `1a4f47311818` | +581/-97 | TARGET-SHAPE NON-TRANSFERABLE: source replaces compact projection + argmax + target-ID mapping with an argmax-only selector. The required temperature-1/top-k20 route must retain the complete sparse proposal distribution for exact speculative acceptance; substituting argmax changes the sampling law. Row 10 already retains the source's compact Q4/group-64 projection and on-device mapping. |
| 20 | 180 | 0.9180% | `cf350293feb4` | `b9b4300e973d` | +144/-8 | RETAINED on corrected rows 8+10+17+18: adapted to MTPLX's dead committed-history outputs by omitting Q/gate/attention/MLP for every appended row and packing the live BF16 K+V projections into one matmul. The packed path engaged 286 times per candidate arm over 17,112 history tokens and improved wall throughput 2.0739% with exact tokens and schedules. |
| 21 | 186 | 1.5222% | `4eb54489fb51` | `df0b66eded6c` | +228/-5 | RETAINED on corrected rows 8+10+17+18+20: exact Qwen 3.8 BF16 H256/R64 fused Q/K RMSNorm + partial-RoPE engaged 2,512 times per candidate arm and improved wall throughput 1.2860%. Both routes were deterministic; proposal/acceptance tie drift was allowed. The old -0.8870% pre-row17 result is historical. |
| 23 | 215 | 0.2964% | `df404e08fee2` | `597330a384fb` | +64/-41 | DEPENDENCY ABSENT/TARGET-SHAPE NON-TRANSFERABLE: this patch only retunes the row-19 argmax-only compact selector's reduction. Row 19 is absent because temperature-1/top-k20 acceptance requires the complete sparse proposal distribution, so row 23 has no retained call site and cannot affect this route. |
| 24 | 234 | 0.9658% | `7351e62674bc` | `849631b545f2` | +54/-14 | RETAINED in part on corrected rows 8+10+17+18+20+21: adaptive-margin depth is disabled by fixed D3 and fused SwiGLU depends on row 18's rejected packed MLP gate/up. The now-live Q/K L<=16 bound fell back 176 times per candidate arm, the target evaluation ladder engaged 144 times, and the combined candidate improved wall throughput 0.8435% with exact tokens and schedules. |
| 25 | 270 | 0.5421% | `c7468c565a7c` | `e8898ba2afd6` | +1/-1 | REOPENED FOR ADAPTIVE MTP: the single streak-gate constant now has a live adaptive-policy call site and must be gated chronologically. |
| 26 | 276 | 0.1799% | `033f622755ac` | `47dc8c6d9b36` | +14/-6 | RETAINED in part on corrected rows 8+10+17+18+20+21+24: debug-only validation removal is irrelevant in release Python, while adaptive-depth and the deeper-width kernel edits remain reopened for the adaptive stack. The fixed-D3 live change tightens retained row 24's long-prefill target-evaluation cadence from every fourth layer to every third; it engaged 176 times per candidate arm and improved wall throughput 1.6797% with exact tokens and schedules. |
| 28 | 304 | 0.2525% | `6209702fba83` | `a6d69403cda0` | +6/-10 | REJECTED on corrected rows 8+10+17+18+20+21+24+26: source's eager recurrent-state evaluation is already absent from MTPLX capture/commit rollback. Replacing retained row 17 with the pinned alternate Q4/group-64 MTP artifact kept exact tokens and schedules but regressed wall throughput 0.0861%, so the alternate artifact is removed and row 17 remains active. |
| 30 | 350 | 0.4202% | `32b94cb67d2f` | `948f58d0f63b` | +120/-9 | ALREADY PRESENT/TARGET-COMPILED: source publishes the target forward's existing post-final-norm block and reuses it for committed MTP history. MTPLX's Qwen forward already returns `post_norm` for `base_hidden_variant=post_norm`, and capture/commit appends those returned rows directly without another final RMSNorm. Source's separately compiled full-attention sigmoid/multiply is already enclosed by MTPLX's whole route-specific compiled target graph. There is no missing live dispatch boundary to port. |
| 32 | 365 | 0.1764% | `156b5b75bdfa` | `66b436ee06e7` | +58/-24 | REOPENED FOR ADAPTIVE MTP: both the M=8 Q4/group-64 retune and adaptive streak policy become live at deeper selected widths and require a chronological adaptive-stack gate. |
| 33 | 401 | 1.7181% | `cbdc3a8d5fa9` | `9cd8e978d00a` | +92/-7 | REMOVED NEXT ROW: proposal-only BF16 Q/K/V precision islands and their declared artifact are deleted wholesale by row 34 before they can enter the retained chronological stack. The mechanism is reintroduced later by row 36 and is handled there. |
| 34 | 405 | 0.6815% | `79683c633b13` | `aa0820c6217c` | +65/-114 | TARGET-SHAPE NO-OP/REMOVAL: deletes row 33's transient precision-island artifact, restores the independent row-28 Q4/group-64 block, and changes direct-nibble QMV only at M=6 and M=9. Fixed-D3 verification is M=4, so the surviving kernel edits have no live call site. |
| 36 | 423 | 1.6826% | `ed4dfd6b0e95` | `12afdfd18be8` | +105/-40 | RETAINED on corrected rows 8+10+17+18+20+21+24+26: the pinned Q4/group-64 block plus proposal-only BF16 precision islands executed a mean 7 Q and 299 each K/V correction calls per timed candidate arm and improved wall throughput 0.1479%. Both routes were internally deterministic; proposal/acceptance and emitted-token drift at temperature 1 were allowed. The source M=8 toggle remains reopened for adaptive depth, while removed warm expressions are covered by the common conditioner. Row 36 supersedes row 17's artifact in later controls. |
| 37 | 428 | 0.1477% | `be3361b96875` | `882e395797e2` | +33/-13 | CONDITIONER-COVERED/TARGET-SHAPE NO-OP: source warms the row-30 post-norm verify output and reverts row 36's M=8 direct-nibble toggle. Every route receives the required 1,024-token conditioning generation before its timed arms, while fixed D3 never dispatches M=8. |
| 38 | 430 | 0.6244% | `0824e0ec28e5` | `dc1e16093bb7` | +2/-2 | TARGET-SHAPE NO-OP: toggles direct-nibble extraction only for affine-4/group-64 M=8. The fixed-D3 target verify is M=4. |
| 39 | 437 | 0.3030% | `1abe6368a882` | `2f3e81092f41` | +13/-33 | REMOVED NEXT ROW: source changes affine-4/group-64 M=4 from one four-row input group to two two-row groups, then row 40 restores the M=4 incumbent exactly. The same two-row ownership idea was independently adapted to the live group-32 target at row 9 and rejected there; this removed snapshot does not enter the cumulative stack. |
| 40 | 438 | 0.6643% | `d1530a409848` | `7a4e9cbc4c6b` | +4/-4 | TARGET-SHAPE NO-OP/RESTORE: restores M=4 after row 39 and enables direct-nibble extraction only at M=7. Fixed D3 verifies at M=4, so the surviving M=7 edit is inactive. |
| 41 | 450 | 0.4931% | `0d800b229e94` | `3272565fbdc7` | +6/-6 | ALREADY PRESENT/SUPERSEDED: source changes the affine-4/group-64 M=3,4,5 stock QMV inner loop from masked nibbles with power-of-two-scaled activations to direct shifts and unscaled activations. Optimized-Speed's live Turbo M=4 verify dispatcher already uses direct `pack >> (ki*4) & 0xf` extraction in its stronger group-32 shape-specialized `vk_k` kernel; row 9 proved that replacing this incumbent M=4 lane with the source-style shared-row morphology regresses. M=3/5 are outside fixed-D3 target verification. |
| 42 | 472 | 1.4130% | `036fd9ca2a2c` | `be0fefb19a14` | +124/-9 | TARGET-SHAPE NON-TRANSFERABLE: replaces the row-36 full Q4/group-64 proposal readout with an affine-2 coarse full-vocabulary pass, keeps only its top 32 token IDs, and exactly reranks those 32 Q4 rows to choose one argmax proposal. That is valid for the source's greedy proposer but not for the required temperature-1/top-p0.95/top-k20 route: speculative acceptance needs the complete proposal probability distribution and cannot reconstruct it from a 32-row shortlist. The target model and verification path are untouched, so there is no independent stochastic-route mechanism to gate. |
| 45 | 505 | 0.4408% | `868cde8f985a` | `3138ecedd936` | +309/-48 | REMOVED NEXT ROW: its memoized norm constants, fused residual/RMSNorm boundary chain, and no-copy attention gate layout are deleted by row 46. The target-side mechanisms are reintroduced by row 48 and are handled as that later live candidate. |
| 47 | 530 | 0.5687% | `dccba745af5b` | `e89a06dfd673` | +195/-5 | DEPENDENCY ABSENT/TARGET-SHAPE NO-OP: adds a custom affine-2 M=1 kernel for row 42's argmax-only coarse proposal selector and changes Q4 M=8 grouping. Temperature-1/top-k20 speculative acceptance cannot replace the complete proposal distribution with that argmax shortlist, and fixed D3 does not dispatch M=8. |
| 48 | 543 | 0.1230% | `86fb1f020fc1` | `d2962993b6da` | +422/-240 | RETAINED on corrected rows 8+10+18+20+21+24+26+36: the source's Q/K scalar memo is already present in MTPLX capture, and its row-47 affine-2 removal is irrelevant because that argmax selector is absent. The adapted 64-layer BF16/5120 fused residual/RMSNorm boundary path executed a mean 151 forwards and 9,513 merged interior boundaries per timed candidate arm, kept exact tokens and schedules, and improved wall throughput 0.9175%. |
| 50 | 572 | 0.2736% | `c0e34afd857e` | `4b6eb22f8820` | +115/-0 | RETAINED on corrected rows 8+10+18+20+21+24+26+36+48: the adapted post-warm policy measured 21,317,046,640 active bytes and set a 21,384,155,504-byte wired limit (64 MiB slack), while control arms restored the zero baseline. Candidate-first then control conditioning prevented the one-time cache clear from making control cold. The route kept exact tokens/schedules and improved wall throughput 0.8299%. Item 55 must recompute this budget over the assembled target plus DFlash2 footprint. |
| 53 | 600 | 0.1577% | `0c90733d383f` | `39b6322daa32` | +11/-24 | RETAINED on corrected rows 8+10+18+20+21+24+26+36+48+50: the live source change force-sets MLX's process-latched command-buffer profile to 512 MiB/50 operations; its verify-concat warm-loop removal is conditioner-covered, and the older Swift-only 128-to-512 override has no separate Python call site. Four clean child processes loaded the identical cumulative stack in ABBA order and observed null/null for control versus exact 512/50 candidate values before any MLX import. Tokens and depth schedules were exact; wall throughput improved 0.4708%. Peak memory rose from 25.48171 to 32.72811 GiB (+7.24641 GiB), which is recorded but does not override the speed-first >0.05% promotion rule. Item 55 must re-gate this profile over the combined target/DFlash2 process rather than assuming transfer. |
| 59 | 843 | 1.4217% | `3e2530aeae21` | `a0b5e9aaa3c4` | +23/-2 | TARGET-SHAPE NO-OP/REMOVED NEXT ROW: only raises the adaptive SDPA width cap from depth 5 to 6 before the full-accept streak gate; row 60 restores the prior policy. This campaign fixes D3 and disables adaptive depth. |
| 60 | 846 | 0.2224% | `88578f929552` | `c529a4989d0d` | +155/-27 | REMOVED NEXT ROW: restores row 59's policy and introduces a two-output dual RMSNorm for the MTP embedding/hidden pair. Row 61 immediately replaces that kernel and its two outputs plus concatenate with the later single-output dual-RMSNorm-concat candidate, which is the live mechanism to gate. |
| 61 | 866 | 0.2836% | `8b54ff11c6d6` | `feeffa289cd4` | +129/-10 | RETAINED on corrected rows 8+10+18+20+21+24+26+36+48+50+53: the exact BF16 dual RMSNorm plus output concatenate engaged seven times in each timed candidate arm, preserved exact tokens/depth schedules and peak memory, and improved ABBA mean wall throughput 0.9996%. Candidate arms were 40.012/40.063 s; control endpoints were 41.047/39.828 s, whose directional drift is balanced by ABBA. The earlier 16.9%-spread diagnostic remains preserved under `invalid-unstable-r61-...json` and is not used for promotion. |
| 63 | 911 | 0.1698% | `61612aa89dc6` | `0fd574d04a01` | +384/-14 | REJECTED IN PART on corrected rows 8+10+18+20+21+24+26+36+48+50+53+61: the source's argmax-only top-32 proposer remains non-transferable to temperature-1 full-distribution acceptance. The live fusion was correctly adapted to Optimized-Speed's Q8/group-64 embedding, compiled on the real artifact, superseded row 61 inside candidate arms, and engaged seven times per arm with exact tokens/depth schedules. It regressed wall throughput 0.1895% at unchanged peak memory, so the fused embedding variant is removed and row 61's separate dual-norm-concat kernel remains retained. |
| 66 | 965 | 0.3080% | `ca0612472eb5` | `ddfede29ee90` | +1/-1 | WEAK/NO-OP: changes only the human-readable artifact note by appending a resample-ticket annotation; model tensors and executable code are byte-for-byte unchanged. |
| 67 | 968 | 0.3523% | `41bad1c6f124` | `76fa838f3cf2` | +93/-90 | TARGET-SHAPE NON-TRANSFERABLE: replaces the gather-QMM plus reducer for row 42's argmax-only 32-row proposal rerank with a direct selected-row Metal kernel. Temperature-1/top-k20 acceptance requires the complete proposal probability distribution, so this selector has no valid call site in the required route. |
| 69 | 1031 | 0.2133% | `fac135f222f9` | `863c65d8ae0b` | +412/-746 | TARGET-SHAPE NON-TRANSFERABLE: replaces two selection stages inside the row-42 affine-2 clustered retrieval proposer with single-dispatch exact BF16 selectors, then feeds the same 32-row Q4 argmax rerank. It also deletes older selector experiments and trace fields. None touches target verification, cache state, MTP block arithmetic, or a full proposal distribution; the required temperature-1/top-p0.95/top-k20 route has no clustered shortlist call site. |
| 70 | 1063 | 3.9125% | `6f1cd66fc214` | `0dd6cffb1309` | +1070/-496 | STAGED IN PART: the clustered argmax selector and island-arm research controls are non-transferable, but the candidate-owned affine-4/group-64 wide QMV is live for MTP linears at M=3...9. The exact source arithmetic is ported: direct nibbles, source row partition, BF16 chunk sums, and a once-per-activation sum table consumed for M>=4; only exact Q4/group-64 BF16 cells with K divisible by 512 and N>=4096 route. Its first real-artifact conditioner exposed that MLX Python arrays have no public `.strides`; no timed arm ran. The corrected boundary delegates row-contiguity enforcement to `metal_kernel(ensure_row_contiguous=True)`, which preserves correctness for views and avoids relying on a nonexistent API. It remains scoped to the MTP block so the target's Q4/group-32 trunk is untouched; real Metal parity and the chronological 16K gate remain pending. |
| 71 | 1066 | 0.7439% | `a0f8588668c6` | `8b0283f1e500` | +267/-11 | TARGET-SHAPE NON-TRANSFERABLE: customizes the affine-2 centroid and selected-cluster QMV stages of the row-42/69 argmax-only retrieval index. The required stochastic route needs the complete proposal distribution and cannot dispatch this shortlist selector. |
| 78 | 1123 | 4.3907% | `8849fad72cc3` | `53cc0c1c6e42` | +39/-14 | STAGED AS A ROW-70 DEPENDENCY: replaces row 70's M-wide launch with the exact active input-group count (M3/4/5 -> 1, M6/7/8 -> 2, M9 -> 3), removing threadgroups that return before any read/write. Its unrelated selector fraction restoration is non-transferable. The route requires row 70 and has an explicit active-group configuration receipt; it can enter the cumulative stack only if row 70 is retained and this later 16K gate also clears. |
| 79 | 1130 | 0.2444% | `1d66bb36cda1` | `503f4fe8d56b` | +7/-10 | TARGET-SHAPE NON-TRANSFERABLE: only changes the row-69 argmax retrieval index's derived-cluster probe fraction from 0.25 to 0.15. That selector is absent under temperature-1/top-k20 full-distribution acceptance. |
| 80 | 1139 | 0.3477% | `e8f14c444156` | `49484169aabc` | +8/-4 | STAGED AS A ROW-70/78 DEPENDENCY: extends the same exact Q4/group-64 routed QMV and active-group launch to M=2. This is live under fixed D3 because retained committed-history updates can present two rows to wide MTP linears. The gate requires rows 70 and 78 and separately demands nonzero M2 calls before promotion. Kernel compilation/parity and the chronological 16K cumulative gate remain pending. |
| 82 | 1153 | 0.3733% | `eb5eadc7a165` | `3142edaa4070` | +5/-27 | CONDITIONER-COVERED/DEPENDENCY ABSENT: removes untimed folded-history warm shapes and skips construction of a fallback probe-sort factory while row 69's E87 argmax selector is active. The exact 1,024-token route conditioner already pays any live first touch before timed arms, and the argmax-only selector is absent from the required stochastic route. |

The campaign remains incomplete until every `PENDING` cell is replaced by
direct evidence and every applicable retained implementation has its own 16K
receipt on the cumulative stack.
