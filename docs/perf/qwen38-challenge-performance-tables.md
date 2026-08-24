# Qwen 3.8 Challenge Performance Tables

These are the canonical performance tables for the 54 Yukon proposals. Each
metric cell is `control → candidate`; untimed rows remain visible with the
reason they were not benchmarked. Means use the two arms for each route from
the exact 16,384-token Python / 1,024-output ABBA gate. Wall delta is
`control / candidate - 1`, so positive is faster. Peak is the maximum arm.
DFlash receipts record MLX's decimal-GB value; every DFlash peak displayed
here is mechanically converted to GiB (`bytes / 2^30`).

## Native MTP: all 54 proposal rows

| Row | Optimization | Prefill tok/s | Decode tok/s | Mean wall s | Wall delta | Peak GiB | Native-MTP result | Receipt / evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2 | Checkpoint/rejection fast path | — | — | — | — | — | Already present in broader capture/commit rollback | Row-8 route receipt |
| 3 | Packed target Q/K/V projection | 754.031 → 761.534 | 51.803 → 46.971 | 42.006 → 43.785 | **-4.0629%** | 25.569 → 25.569 | Rejected, removed | `c1-packed-qkv-corrected-python16384in-1024out-t1-abba-2026-08-23.json` |
| 4 | Seed-tail logits and lazy capture boundary | — | — | — | — | — | Already present, broader in Optimized-Speed | Row-8 route receipt |
| 5 | Device target top-k sampling | — | — | — | — | — | Already present/adapted for temperature 1, top-p .95, top-k 20 | Row-8 route receipt |
| 6 | Reuse second target argmax | — | — | — | — | — | Target-shape no-op: stochastic route has no second argmax | Source diff + row-8 route receipt |
| 7 | Persistent committed MTP history | — | — | — | — | — | Already present and engaged | Row-8 route receipt |
| 8 | Device-resident fixed-D3 draft chain | 750.099 → 779.772 | 52.478 → 54.038 | 41.870 → 40.413 | **+3.6049%** | 24.885 → 24.885 | Retained; S≤2 GDN sub-fuse was a D3 no-op | `chrono-r08-device-draft-python16384in-1024out-t1-abba-2026-08-23.json` |
| 9 | Paired shared-row G32/M4 target QMV | 744.715 → 761.698 | 52.555 → 47.465 | 41.977 → 43.651 | **-3.8350%** | 24.885 → 24.885 | Rejected, removed | `chrono-r09-paired-qmv-g32-m4-on-r08-python16384in-1024out-t1-abba-2026-08-23.json` |
| 10 | Compact Q4/G64 proposal vocabulary | 737.536 → 765.005 | 54.151 → 55.240 | 41.640 → 40.411 | **+3.0411%** | 25.149 → 25.149 | Retained | `chrono-r10-compact-vocab-on-r08-python16384in-1024out-t1-abba-2026-08-23.json` |
| 11 | Position-EMA adaptive-depth policy | — | — | — | — | — | Native policy staged; deployment adaptation tested in DFlash 1–8 lane and rejected | DFlash row 11 below |
| 12 | Recurrent prefix-replay tape | — | — | — | — | — | Removed by row 13 before entering stack | Source chronology |
| 13 | Four-way GDN input projection through S≤9 | 765.003 → 778.485 | 56.424 → 54.008 | 40.066 → 40.469 | **-0.9949%** | 27.507 → 27.507 | Rejected, removed | `chrono-r13-gdn-inproj-s9-on-r08-r10-python16384in-1024out-t1-abba-2026-08-23.json` |
| 14 | Reintroduced recurrent prefix replay | — | — | — | — | — | Already present in broader capture/commit path | Row-18 route receipt |
| 15 | Verify widths 6–9 and raised adaptive cap | — | — | — | — | — | Native wide policy moved to DFlash 1–8 deployment lane; retained there | DFlash row 15 below |
| 16 | Compiled GDN prologue/post-norm | — | — | — | — | — | Already enclosed by whole-route compilation | Source diff + route contract |
| 17 | Complete Q4/G64 MTP block | 750.137 → 781.155 | 56.805 → 55.766 | 40.377 → 39.808 | **+1.4299%** | 25.37195 → 25.37197 | Retained, later superseded by row 36 artifact | `chrono-r17-q4-mtp-block-on-r08-r10-python16384in-1024out-t1-abba-2026-08-23.json` |
| 18 | Memoized GDN decay | 736.928 → 763.072 | 56.265 → 54.971 | 40.978 → 40.579 | **+0.9843%** | 25.37198 → 25.37199 | Memo retained; packed MLP addendum rejected at -0.3108% | `chrono-r18-gdn-decay-memo-on-r08-r10-r17-python16384in-1024out-t1-abba-2026-08-23.json` |
| 19 | Argmax-only compact proposal selector | — | — | — | — | — | Non-transferable to stochastic full-distribution acceptance | Source diff |
| 20 | Packed K/V-only committed-history append | 735.167 → 758.616 | 55.625 → 54.778 | 41.230 → 40.392 | **+2.0739%** | 25.37749 → 25.37749 | Retained | `chrono-r20-kv-only-history-on-r08-r10-r17-r18-python16384in-1024out-t1-abba-2026-08-23.json` |
| 21 | Fused Q/K RMSNorm + partial RoPE | 720.881 → 737.392 | 54.455 → 54.433 | 41.661 → 41.132 | **+1.2860%** | 25.37751 → 25.37751 | Retained on corrected stack | `chrono-r21-qk-rms-rope-on-r08-r10-r17-r18-r20-python16384in-1024out-t1-abba-2026-08-23.json` |
| 23 | Retuned row-19 compact reduction | — | — | — | — | — | Dependency absent; row 19 is non-transferable | Source diff |
| 24 | Q/K L≤16 fence + target evaluation ladder | 724.633 → 740.452 | 55.175 → 54.674 | 41.303 → 40.958 | **+0.8435%** | 25.37750 → 25.44388 | Retained on corrected stack | `chrono-r24-full-on-r08-r10-r17-r18-r20-r21-python16384in-1024out-t1-abba-2026-08-23.json` |
| 25 | Adaptive streak-gate constant | — | — | — | — | — | Tested in DFlash 1–8 deployment lane; rejected | DFlash row 25 below |
| 26 | Three-layer prefill evaluation cadence | 753.929 → 774.419 | 56.866 → 57.018 | 39.885 → 39.226 | **+1.6797%** | 25.44388 → 25.44388 | Retained; deeper-width adaptive subrevision later rejected in DFlash lane | `chrono-r26-full-on-r08-r10-r17-r18-r20-r21-r24-python16384in-1024out-t1-abba-2026-08-23.json` |
| 28 | Alternate Q4/G64 block + eager recurrent state | 778.698 → 783.596 | 56.396 → 55.874 | 39.299 → 39.332 | **-0.0861%** | 25.67190 → 25.67190 | Rejected; incumbent block retained | `chrono-r28-full-on-r08-r10-r17-r18-r20-r21-r24-r26-python16384in-1024out-t1-abba-2026-08-23.json` |
| 30 | Reused post-final-norm target output | — | — | — | — | — | Already present/target-compiled | Source diff + capture contract |
| 32 | M=8 Q4/G64 retune + adaptive streak | — | — | — | — | — | Adaptive revision rejected in DFlash lane; interim M8 grouping superseded by row 47 | DFlash rows 32 and 47 below |
| 33 | Transient BF16 Q/K/V precision islands | — | — | — | — | — | Removed by row 34; live form returns at row 36 | Source chronology |
| 34 | M=6/M=9 direct-nibble edits and row-33 removal | — | — | — | — | — | Fixed-D3 no-op/removal; M6 DFlash adaptation tested and rejected, M9 outside cap | DFlash row 34 below |
| 36 | Q4/G64 block + BF16 Q/K/V islands | 772.342 → 782.572 | 56.640 → 55.929 | 39.399 → 39.341 | **+0.1479%** | 25.70974 → 25.70974 | Retained; supersedes row 17 | `chrono-r36-full-on-r08-r10-r17-r18-r20-r21-r24-r26-python16384in-1024out-t1-abba-2026-08-23.json` |
| 37 | Warm post-norm verify + revert M=8 toggle | — | — | — | — | — | Conditioner-covered/fixed-shape no-op; M8 revert replaced by row 38 | Source chronology |
| 38 | M=8 direct-nibble extraction | — | — | — | — | — | Fixed-D3 no-op; interim DFlash M8 form superseded and tested in final row-47 form | DFlash row 47 below |
| 39 | Two two-row M=4 input groups | — | — | — | — | — | Removed by row 40; related row-9 adaptation rejected | Source chronology + row-9 receipt |
| 40 | Restore M=4 + enable M=7 direct nibble | — | — | — | — | — | Fixed-D3 no-op/restore; complete M6+M7 DFlash state tested and rejected | DFlash row 40 below |
| 41 | Direct-nibble M=3/4/5 stock QMV | — | — | — | — | — | Already present/superseded by stronger G32 M=4 path | Source diff + row-9 receipt |
| 42 | Affine-2 coarse top-32 argmax proposer | — | — | — | — | — | Non-transferable to stochastic full-distribution acceptance | Source diff |
| 45 | Early fused residual/RMS boundary variant | — | — | — | — | — | Removed by row 46; live form returns at row 48 | Source chronology |
| 47 | Affine-2 M=1 selector + Q4 M=8 grouping | — | — | — | — | — | Selector dependency absent; complete final M6+M7+M8 DFlash state tested and rejected | DFlash row 47 below |
| 48 | Fused residual/RMSNorm boundary chain | 735.798 → 760.347 | 55.947 → 54.792 | 40.703 → 40.333 | **+0.9175%** | 25.48171 → 25.48171 | Retained | `chrono-r48-full-on-r08-r10-r18-r20-r21-r24-r26-r36-python16384in-1024out-t1-abba-2026-08-23.json` |
| 50 | Post-warm wired-residency budget | 754.316 → 771.058 | 56.472 → 55.787 | 39.990 → 39.661 | **+0.8299%** | 25.48171 → 25.48171 | Retained | `chrono-r50-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-python16384in-1024out-t1-abba-2026-08-23.json` |
| 53 | 512 MiB / 50-op command buffers | 754.872 → 754.912 | 55.414 → 55.979 | 40.247 → 40.059 | **+0.4708%** | 25.48171 → 32.72811 | Retained, +7.24640 GiB peak | `chrono-r53-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-r50-python16384in-1024out-t1-abba-isolated-2026-08-23.json` |
| 59 | Temporary adaptive SDPA cap 5→6 | — | — | — | — | — | Removed by row 60 | Source chronology |
| 60 | Two-output dual RMSNorm | — | — | — | — | — | Replaced by row 61 before entering stack | Source chronology |
| 61 | Fused dual RMSNorm + concatenate | 742.826 → 765.844 | 55.991 → 55.094 | 40.438 → 40.037 | **+0.9996%** | 32.72812 → 32.72812 | Retained | `chrono-r61-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-r50-r53-python16384in-1024out-t1-abba-rerun-2026-08-23.json` |
| 63 | Fused Q8 embedding/norm/concat + argmax proposer | 754.687 → 768.177 | 57.141 → 55.700 | 39.698 → 39.773 | **-0.1895%** | 32.72812 → 32.72812 | Live fusion rejected; argmax proposer non-transferable | `chrono-r63-full-on-r08-r10-r18-r20-r21-r24-r26-r36-r48-r50-r53-r61-python16384in-1024out-t1-abba-2026-08-23.json` |
| 66 | Resample-ticket artifact note | — | — | — | — | — | Weak/no-op: executable bytes unchanged | Source diff |
| 67 | Direct selected-row Q4 rerank kernel | — | — | — | — | — | Non-transferable argmax-selector dependency | Source diff |
| 69 | Fused clustered BF16 selectors | — | — | — | — | — | Non-transferable argmax retrieval path | Source diff |
| 70 | Clustered proposer + M3-wide QMV | — | — | — | — | — | Rejected incompatible after three real-artifact attempts | `diagnostic-r70-qmv-fixed-d3-incompatible-2026-08-23.json` |
| 71 | Custom centroid/selected-cluster QMV | — | — | — | — | — | Non-transferable argmax retrieval path | Source diff |
| 78 | Active-input-group row-70 launch | — | — | — | — | — | Dependency absent after row-70 rejection | Source diff |
| 79 | Cluster probe fraction .25→.15 | — | — | — | — | — | Non-transferable argmax retrieval path | Source diff |
| 80 | Extend rejected row-70 QMV to M=2 | — | — | — | — | — | Dependency absent after row-70 rejection | Source diff |
| 82 | Remove folded-history warm/probe-sort construction | — | — | — | — | — | Conditioner-covered; selector dependency absent | Source diff + conditioner contract |

## DFlash2 port stack: the same 54 proposal rows

The two `Base` rows are not Yukon proposal rows; they record the engine
replacement and the removal of the unused native-MTP object. Row 8 repeats the
engine-replacement metrics only to show which original drafter optimization it
replaces; that delta is composite and is not claimed as an isolated row-8 win.

The final production-entry verification is likewise supplemental, not a 55th
proposal row or a promotion bracket. After a 1,024-output conditioner, the
shipped bundle path completed 16,384 Python input tokens plus 1,024 output at
731.314 prefill tok/s, 69.489 decode tok/s, 37.173 s wall, and 32.921 GiB peak.
It proposed 1,161 tokens, accepted 823, and engaged the retained row-11+15
adaptive policy for 201 cycles over physical blocks 4--8. This supersedes the
invalid 60.261 tok/s integration measurement, whose production loader exposed
only block-5 draft capabilities and omitted the pre-import Turbo profile. See
`final-production-dflash2-entry-python16384in-1024out-t1-2026-08-23.json`.

| Row | Optimization / DFlash disposition | Prefill tok/s | Decode tok/s | Mean wall s | Wall delta | Peak GiB | DFlash2 result | Receipt / evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Base | Fixed native-MTP stack → fixed-depth-8 DFlash2 | 742.781 → 733.786 | 55.441 → 65.858 | 40.604 → 37.917 | **+7.0860%** | 32.72811 → 32.42313 | Retained replacement base | `item55-dflash2-static8-on-full-fixed-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| Base | Dual-loaded DFlash2 → target-only DFlash2 | 732.129 → 747.429 | 68.140 → 66.839 | 37.444 → 37.275 | **+0.4543%** | 31.94559 → 30.48846 | Retained; native MTP never constructed | `item55-dflash2-target-only-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 2 | Checkpoint/rejection fast path remains target-side | — | — | — | — | — | Already present; no DFlash-specific transplant | Native-MTP row 2 evidence |
| 3 | Packed target Q/K/V projection | — | — | — | — | — | Not ported: native candidate rejected | Native-MTP row 3 receipt |
| 4 | Lazy capture/commit target boundary | — | — | — | — | — | Already present in target stack | Native-MTP row 4 evidence |
| 5 | Exact stochastic target sampler | — | — | — | — | — | Active in every DFlash arm | DFlash workload contract |
| 6 | Second target argmax reuse | — | — | — | — | — | No-op under stochastic target sampling | Native-MTP row 6 evidence |
| 7 | Persistent native-MTP history | — | — | — | — | — | Replaced by DFlash target-feature/rollback ownership | DFlash cache contract |
| 8 | Device-resident native draft chain | 742.781 → 733.786 | 55.441 → 65.858 | 40.604 → 37.917 | **+7.0860%** | 32.72811 → 32.42313 | Replaced by DFlash2 eight-token proposal block; composite base delta | DFlash replacement-base receipt |
| 9 | Paired target QMV | — | — | — | — | — | Not ported: native candidate rejected | Native-MTP row 9 receipt |
| 10 | Compact native-MTP vocabulary | — | — | — | — | — | Incompatible: DFlash selector can emit outside the 98,330-ID set | Vocabulary geometry evidence |
| 11 | Position-EMA adaptive depth, mapped to DFlash blocks 1–8 | 749.632 → 759.566 | 67.048 → 57.642 | 37.162 → 39.370 | **-5.6076%** | 32.92223 → 32.92006 | Rejected; engaged 281 cycles/arm across blocks 1–5 | `item55-dflash2-a11-position-ema-on-r21-r24-r26-r48-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 12 | Prefix-replay tape | — | — | — | — | — | Removed by row 13 | Source chronology |
| 13 | Four-way GDN input projection | — | — | — | — | — | Not ported: native candidate rejected | Native-MTP row 13 receipt |
| 14 | Recurrent prefix replay | — | — | — | — | — | Replaced by DFlash rollback/cache path | DFlash cache contract |
| 15 | Wide adaptive cap, clamped to DFlash blocks 1–8 | 763.417 → 752.280 | 66.733 → 69.670 | 36.833 → 36.528 | **+0.8344%** | 32.92222 → 32.92107 | Retained complete row-11+15 revision; blocks 4–8 engaged | `item55-dflash2-a15-wide-position-ema-on-fixed-stack-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 16 | Compiled GDN prologue/post-norm | — | — | — | — | — | Already enclosed by target execution graphs | Native-MTP row 16 evidence |
| 17 | Complete Q4/G64 native MTP block | — | — | — | — | — | Replaced by pinned DFlash W4/G64 checkpoint; superseded by row 36 | Checkpoint manifest |
| 18 | Memoized GDN decay; adaptive streak-3/optimism-cap revision | Memo: 736.544 → 717.003<br>Adaptive: 768.382 → 768.819 | Memo: 67.051 → 66.759<br>Adaptive: 68.231 → 48.624 | Memo: 37.559 → 38.221<br>Adaptive: 36.363 → 42.415 | Memo: **-1.7342%**<br>Adaptive: **-14.2695%** | Memo: 30.48846 → 30.48847<br>Adaptive: 32.92107 → 32.92011 | Both rejected; adaptive revision shifted 201 → 334 cycles/arm | `item55-dflash2-r18-gdn-decay-memo-on-target-only-python16384in-1024out-t1-isolated-abba-2026-08-23.json`<br>`item55-dflash2-a18-streak3-optimism-cap-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 19 | Argmax-only compact selector | — | — | — | — | — | Non-transferable to stochastic DFlash acceptance | Native-MTP row 19 evidence |
| 20 | Native K/V-only history append | — | — | — | — | — | Replaced by DFlash feature and rollback caches | DFlash cache contract |
| 21 | Fused Q/K RMSNorm + partial RoPE | 694.696 → 729.811 | 66.374 → 65.775 | 39.111 → 38.046 | **+2.8001%** | 30.48846 → 30.48845 | Retained; 3,184 calls/arm | `item55-dflash2-r21-qk-rms-rope-on-target-only-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 23 | Retuned row-19 reduction | — | — | — | — | — | Dependency absent | Native-MTP row 23 evidence |
| 24 | Q/K L≤16 fence + target evaluation ladder; adaptive first-margin clamp | Fixed: 749.960 → 770.175<br>Adaptive: 742.601 → 754.103 | Fixed: 68.026 → 67.199<br>Adaptive: 67.440 → 59.544 | Fixed: 36.944 → 36.537<br>Adaptive: 37.305 → 38.962 | Fixed: **+1.1148%**<br>Adaptive: **-4.2531%** | Fixed: 30.48846 → 33.27748<br>Adaptive: 32.92107 → 32.92098 | Fixed target changes retained; adaptive margin revision rejected | `item55-dflash2-r24-eval-ladder-on-r21-python16384in-1024out-t1-isolated-abba-2026-08-23.json`<br>`item55-dflash2-a24-first-margin-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 25 | Adaptive deep-cap streak gate 3→2 | 763.352 → 748.011 | 68.197 → 61.277 | 36.511 → 38.652 | **-5.5393%** | 32.92107 → 32.91940 | Rejected; candidate used 239 cycles/arm versus incumbent 201 | `item55-dflash2-a25-streak2-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 26 | Three-layer prefill cadence; cap-5/two-margin adaptive revision | Fixed: 752.388 → 754.599<br>Adaptive: 733.361 → 748.585 | Fixed: 67.593 → 68.557<br>Adaptive: 68.185 → 65.847 | Fixed: 36.965 → 36.689<br>Adaptive: 37.404 → 37.471 | Fixed: **+0.7544%**<br>Adaptive: **-0.1792%** | Fixed: 33.27749 → 31.95738<br>Adaptive: 32.92108 → 32.92325 | Fixed cadence retained; adaptive subrevision rejected | `item55-dflash2-r26-prefill-ladder3-on-r21-r24-python16384in-1024out-t1-isolated-abba-2026-08-23.json`<br>`item55-dflash2-a26-two-margin-cap5-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 28 | Alternate native Q4/G64 block | — | — | — | — | — | Not applicable: rejected native artifact; DFlash owns drafter | Native-MTP row 28 receipt |
| 30 | Reused target post-final-norm output | — | — | — | — | — | Already present in DFlash hidden-capture/logit path | DFlash target-ops contract |
| 32 | Final source-policy streak revision; interim M=8 retune later superseded | 739.288 → 751.009 | 69.437 → 50.860 | 36.963 → 41.984 | **-11.9594%** | 32.92108 → 32.92200 | Adaptive revision rejected; interim M=8 grouping deferred to final row 47 form | `item55-dflash2-a32-final-source-policy-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 33 | Transient precision islands | — | — | — | — | — | Removed by row 34 | Source chronology |
| 34 | M=6/M=9 direct-nibble edits, adapted to the DFlash W4/G64 drafter | 722.962 → 720.659 | 65.958 → 59.689 | 38.221 → 39.924 | **-4.2634%** | 32.92107 → 32.92416 | Rejected; M6 engaged 926 calls/arm, while M9 is outside DFlash's 1–8 block range | `item55-dflash2-c34-m6-direct-nibble-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 36 | Native Q4/G64 block + precision islands | — | — | — | — | — | Replaced by pinned DFlash W4/G64 checkpoint | Checkpoint manifest |
| 37 | Warm path + M=8 revert | — | — | — | — | — | Conditioner-covered; M8 revert replaced by row 38 | Source chronology |
| 38 | M=8 direct-nibble extraction | — | — | — | — | — | Interim grouping superseded by row 47; final M8 form measured there | DFlash row 47 receipt |
| 39 | Transient M=4 grouping | — | — | — | — | — | Removed by row 40 | Source chronology |
| 40 | Restore M=4 and add M=7 direct-nibble extraction; complete surviving M6+M7 DFlash state | 757.932 → 740.764 | 67.344 → 63.313 | 36.868 → 38.350 | **-3.8639%** | 32.92107 → 32.91969 | Rejected; M6/M7 engaged 2,655/1,589 calls per arm | `item55-dflash2-c40-m6-m7-direct-nibble-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 41 | Direct-nibble native target QMV | — | — | — | — | — | Already present/superseded target implementation | Native-MTP row 41 evidence |
| 42 | Affine-2 argmax proposer | — | — | — | — | — | Non-transferable to stochastic DFlash acceptance | Native-MTP row 42 evidence |
| 45 | Early boundary fusion | — | — | — | — | — | Removed; live form is row 48 | Source chronology |
| 47 | Affine-2 M=1 selector + final M=8 grouping; complete M6+M7+M8 DFlash state | 770.770 → 765.495 | 68.374 → 58.219 | 36.265 → 39.023 | **-7.0673%** | 32.92107 → 32.92177 | Rejected; selector absent, M6/M7/M8 engaged 919/1,749/3,928 calls per arm | `item55-dflash2-c47-final-m6-m7-m8-qmv-on-a15-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 48 | Fused residual/RMSNorm DFlash capture boundary | 729.581 → 742.123 | 66.704 → 68.098 | 37.899 → 37.171 | **+1.9584%** | 31.95738 → 32.92223 | Retained; 199 forwards and 12,537 merged boundaries/arm | `item55-dflash2-r48-boundary-fused-on-r21-r24-r26-python16384in-1024out-t1-isolated-abba-2026-08-23.json` |
| 50 | Wired-residency budget | — | — | — | — | — | Active and recomputed in every DFlash arm; no isolated delta | DFlash arm feature receipts |
| 53 | 512 MiB / 50-op command buffers | — | — | — | — | — | Active in every isolated DFlash arm; no isolated delta | DFlash arm feature receipts |
| 59 | Temporary adaptive SDPA cap | — | — | — | — | — | Removed by row 60 | Source chronology |
| 60 | Two-output dual RMSNorm | — | — | — | — | — | Replaced by row 61 | Source chronology |
| 61 | Native-MTP dual norm/concat | — | — | — | — | — | Replaced: removed native MTP block has no call site | DFlash replacement contract |
| 63 | Fused native embedding/norm + argmax proposer | — | — | — | — | — | Native fusion rejected; proposer non-transferable | Native-MTP row 63 receipt |
| 66 | Artifact-note-only change | — | — | — | — | — | Weak/no-op | Source diff |
| 67 | Selected-row argmax rerank | — | — | — | — | — | Non-transferable | Source diff |
| 69 | Clustered argmax selectors | — | — | — | — | — | Non-transferable | Source diff |
| 70 | Clustered proposer + native M3 QMV | — | — | — | — | — | Rejected incompatible before DFlash port | Native-MTP row 70 diagnostic |
| 71 | Cluster QMV kernels | — | — | — | — | — | Non-transferable | Source diff |
| 78 | Row-70 active-group launch | — | — | — | — | — | Dependency absent | Source diff |
| 79 | Cluster probe fraction | — | — | — | — | — | Non-transferable | Source diff |
| 80 | Row-70 M=2 extension | — | — | — | — | — | Dependency absent | Source diff |
| 82 | Removed warm/probe construction | — | — | — | — | — | Conditioner-covered; selector dependency absent | Source diff + conditioner contract |

## Post-54 DFlash2 target-shape candidates

These are cumulative follow-ups, not additional Yukon proposal rows. Every
valid row uses four isolated ABBA arms on the same 16K Python / 1,024-output
workload. DFlash memory values are decimal GB as reported by `dflash-mlx`.

| Candidate | Control → candidate stack | Prefill tok/s | Decode tok/s | Mean wall s | Wall delta | Peak GB | Result | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GQA widths 6--8 | Production → + GQA | 736.967 → 743.492 | 67.977 → 68.445 | 37.361 → 37.036 | **+0.8785%** | 35.34873 → 35.34683 | Retained | `post54-dflash2-gqa678-on-production-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| Cost-aligned adaptive widths | GQA → + cost alignment | 773.971 → 771.523 | 68.195 → 67.067 | 36.215 → 36.539 | **-0.8862%** | 35.34685 → 35.34978 | Rejected | `post54-dflash2-cost-aligned-on-gqa678-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| Exact-BM8 NAX output projection | GQA → + shape-gated M8 | 725.526 → 740.627 | 67.088 → 66.269 | 37.893 → 37.607 | **+0.7617%** | 35.34683 → 35.34683 | Retained; exact/deterministic, 1,056 live M8 routes per candidate arm | `post54-dflash2-m8-nax-island-live-routes-on-gqa678-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| 1,024 MiB command buffers, diagnostic 1 | GQA+M8 → + 1,024 MiB | 722.149 → 664.922 | 67.903 → 63.760 | 37.823 → 40.760 | **-7.2060%** | 35.34685 → 47.45433 | Invalid/superseded; queued guard overlap | `invalid-overlap-post54-dflash2-cb1024-on-gqa678-m8nax-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| 1,024 MiB command buffers, diagnostic 2 | GQA+M8 → + 1,024 MiB | 765.439 → 701.616 | 67.669 → 68.408 | 36.570 → 38.476 | **-4.9557%** | 35.34683 → 47.45434 | Invalid/superseded; Qwen restarted during window | `invalid-autorestart-post54-dflash2-cb1024-on-gqa678-m8nax-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| 1,024 MiB command buffers, canonical | GQA+M8 → + 1,024 MiB | 757.049 → 744.181 | 68.077 → 67.486 | 36.715 → 37.221 | **-1.3607%** | 35.34683 → 47.45433 | Rejected; exact direct-lock ABBA | `post54-dflash2-cb1024-on-gqa678-m8nax-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| Exact-BM8 NAX linear-attention `in_proj_z`, initial | GQA+M8 `o_proj` → + M8 linear-Z | 728.405 → 732.697 | 67.794 → 65.838 | 37.645 → 37.947 | **-0.7978%** | 35.34683 → 35.34683 | Superseded; configured projections but no per-kernel route counters | `post54-dflash2-m8-linear-z-on-gqa678-m8o-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| Exact-BM8 NAX linear-attention `in_proj_z`, live routes | GQA+M8 `o_proj` → + M8 linear-Z | 767.457 → 754.123 | 67.739 → 67.830 | 36.497 → 36.874 | **-1.0234%** | 35.34683 → 35.34683 | Rejected; 3,168 linear-Z plus 1,056 retained output M8 routes per candidate arm | `post54-dflash2-m8-linear-z-live-routes-on-gqa678-m8o-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| Width-7 `o_proj` padded to BM8 NAX | GQA+M8 `o_proj` → + M7-on-M8 output | 742.853 → 754.982 | 67.849 → 68.312 | 37.187 → 36.724 | **+1.2606%** | 35.34683 → 35.34683 | Retained; 992 M7-on-M8 plus 1,056 exact-M8 output routes per candidate arm | `post54-dflash2-m7-padded-m8-output-on-gqa678-m8o-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
| Width-7 linear-Z padded to BM8 NAX | GQA+M8 `o_proj`+M7 output → + M7-on-M8 linear-Z | 736.926 → 742.475 | 67.933 → 68.703 | 37.357 → 37.014 | **+0.9269%** | 35.34683 → 35.34684 | Retained; 2,976 new linear-Z routes with prior M7/M8 output routes live | `post54-dflash2-m7-padded-m8-linear-z-on-gqa678-m8o-m7o-python16384in-1024out-t1-isolated-abba-2026-08-24.json` |
