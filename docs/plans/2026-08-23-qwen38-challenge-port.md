# Qwen 3.8 Challenge Port Corrective Plan

> **Execution:** inline and serial. GPU measurements share one exclusive lock,
> and every retained winner changes the next candidate's control.

**Goal:** Account for all 54 accepted Yukon improvements above 0.10%, port
every surviving transferable mechanism to the exact Optimized-Speed shape,
and land the measured cumulative winners in the existing single draft PR.

**Fixed target:** `Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed` at MTPLX
base `bd4421567f9e16ce957c6ef97708b072dcd73937`, Turbo, Q4/group-64 draft,
depth 3, temperature 1.0, top-p 0.95, top-k 20, seed 42.

**Benchmark:** exactly 16,384 Python tokens from `mtplx/generation.py` with
the intact `python_modules_long.jsonl` instruction at the tail; exactly 1,024
output tokens; one full-output conditioning run per route; exactly four timed
ABBA arms; exclusive `/tmp/mtplx-gpu-exclusive.lock` ownership.

## Correction to the first plan

The first pass produced only six candidate receipts. It incorrectly called
two early mechanisms already present even though both are disabled in the
measured Turbo environment:

- row 3 packed target Q/K/V (`MTPLX_PACKED_PROJ_CONCATS` was unset);
- row 13 fused GDN input projections (`MTPLX_FUSE_GDN_PROJECTIONS` was unset).

It also built the final compact candidate without the precision-island tensors
that the source's final compact artifact explicitly preserves. PR #335 is a
draft until these omissions are corrected. If C1 or C2 wins, every later
candidate is remeasured against the changed cumulative stack.

## All 54 qualifying submissions

`Benchmark Cn` means a real 16K cumulative gate. `Covered by Cn` means the row
is an intermediate form that is later replaced while its surviving mechanism
is measured in Cn. Every skip states the concrete reason.

| Row | PR | Source improvement | Mechanism | Corrected action |
| ---: | ---: | ---: | --- | --- |
| 2 | 13 | 23.2995% | Infrastructure resubmission | Skip: harness/worker infrastructure, not an inference mechanism. |
| 3 | 18 | 1.5662% | Packed affine-4 target Q/K/V | **Benchmark C1.** Isolate attention Q/K/V only; do not include the module's unrelated MLP concat. |
| 4 | 24 | 1.2868% | K=1 fused verify/lazy boundary | Skip already present: Turbo whole-verify compilation and lazy verify logits own the broader boundary. |
| 5 | 29 | 6.7162% | Hierarchical exact target top-2 | Skip challenge-only: MTPLX has no trusted-worker target top-2 ledger consumer. |
| 6 | 37 | 1.7673% | Reuse target top-2 top-1 as argmax | Skip challenge-only: no duplicate target argmax exists in MTPLX. |
| 7 | 38 | 20.4283% | Committed MTP history | Skip already present: persistent committed draft history is the measured control. |
| 8 | 41 | 17.9183% | History/sync/fusion/checkpoint/depth bundle | Split: existing history/sync/checkpoints; projection pieces go through C1/C2; source depth policy is inapplicable to fixed D3. |
| 9 | 55 | 6.8764% | Paired affine QMV | Superseded by C7, which was benchmarked and rejected locally. |
| 10 | 59 | 5.3467% | Warmed compact vocabulary | Covered by final faithful compact artifact C8. |
| 11 | 63 | 5.7425% | Cost model, seed warm, early flush | Existing controller/warm framework; surviving early-flush work is measured in K/V history C3. |
| 12 | 70 | 0.8266% | Lazy exact prefix replay | Skip already present: capture/commit replay and recurrent repair are control behavior. |
| 13 | 71 | 2.0944% | Fused GDN QKV+Z and B+A inputs | **Benchmark C2** using the source's two packed affine projections, not the optional unmeasured four-to-one variant. |
| 14 | 77 | 0.9839% | Prefix replay restack | Skip duplicate of already-present row 12 behavior. |
| 15 | 95 | 3.7597% | Width-6..9 chunking | Superseded by rejected C7; fixed D3 never reaches source verify widths 6..9. |
| 16 | 103 | 0.5808% | Compiled GDN g/beta/post-norm satellites | Skip already present: the complete target verify is one compiled graph and the promoted tape backend carries recurrence inputs into replay. Add an engagement receipt proving both paths. |
| 17 | 126 | 7.5460% | Q4/group-64 draft head | Skip already present: this is the installed Turbo draft construction. |
| 18 | 135 | 0.4535% | One-forward SDPA width bridge | Skip target-shape no-op: source fixes widths 6..9; fixed D3 target verify uses widths 2..4. |
| 19 | 160 | 2.5391% | Wider cross-row QMV/readout | Target-QMV portion rejected through C7; compact proposal portion retained through C8. |
| 20 | 180 | 0.9180% | K/V-only committed-history append | **Benchmark C3** against the C1+C2 winner stack. |
| 21 | 186 | 1.5222% | Fused Q/K RMSNorm plus partial RoPE | **Benchmark C4** against the retained stack through C3. |
| 23 | 215 | 0.2964% | Two-level compact selector | Covered by final compact C8. |
| 24 | 234 | 0.9658% | Confidence-aware scheduling | Skip target-policy no-op: the requested benchmark fixes depth 3 and does not call this source controller. |
| 25 | 270 | 0.5421% | Open deep cap sooner | Skip superseded by row 59 and inapplicable to fixed D3. |
| 26 | 276 | 0.1799% | Composite restack | Split across already-present replay/controller behavior and C1-C4; no independent surviving operation. |
| 28 | 304 | 0.2525% | Q4 head requantization | Skip already present: installed Turbo head is affine Q4/group-64. |
| 30 | 350 | 0.4202% | Verify hidden reuse and compiled output gate | Skip already present: `post_norm` is returned from the norm already used by logits, and the output gate sits inside whole-verify compilation. Add an engagement receipt. |
| 32 | 365 | 0.1764% | Streak-gate restore | Skip superseded by row 59 and no-op under fixed D3. |
| 33 | 401 | 1.7181% | Q4 head plus BF16 Q/K/V precision islands | Covered by C8, which must use the source-pinned island tensors rather than reconstructing a plain Q4 head. |
| 34 | 405 | 0.6815% | Direct-nibble QMV M=6,9 | Superseded by rejected C7. |
| 36 | 423 | 1.6826% | Precision islands plus direct-nibble M=8 | Island portion retained by C8; target-QMV portion rejected through C7. |
| 37 | 428 | 0.1477% | Precision-island restack | Covered by C8; no independent operation. |
| 38 | 430 | 0.6244% | Reuse `HiddenAndNormed` | Skip already present: MTPLX's `post_norm` result is shared between target logits and committed draft history. Add an engagement receipt. |
| 39 | 437 | 0.3030% | Affine-4 QMV M=4 | Superseded by rejected C7. |
| 40 | 438 | 0.6643% | Direct-nibble QMV M=7 | Superseded by rejected C7. |
| 41 | 450 | 0.4931% | Direct-nibble QMV M=3,4,5,7 | Superseded by rejected C7. |
| 42 | 472 | 1.4130% | Affine-2 shortlist plus affine-4 rerank | Covered by final faithful compact C8. |
| 45 | 505 | 0.4408% | Boundary residual/RMSNorm fusion | **Benchmark C5.** |
| 47 | 530 | 0.5687% | 32-value/lane affine-2 QMV | Covered by final faithful compact C8. |
| 48 | 543 | 0.1230% | Remove residency/command-buffer mechanisms | Skip removed later by qualifying row 50. |
| 50 | 572 | 0.2736% | Restore wired residency/command buffers | Skip challenge-only: Swift/Metal worker ownership has no Python MTPLX equivalent. |
| 53 | 600 | 0.1577% | Force 512 MiB command-buffer profile | Skip challenge-only: worker command-buffer policy is not owned by MTPLX. |
| 59 | 843 | 1.4217% | Final floor 6 under cap 7 | Skip target-policy no-op: fixed D3 is already shallower than both source constants and has no adaptive floor transition. |
| 60 | 846 | 0.2224% | Dual pre-FC RMSNorm | Combined into its surviving concat-free descendant C6. |
| 61 | 866 | 0.2836% | Concat-free dual pre-FC RMSNorm output | **Benchmark C6.** |
| 63 | 911 | 0.1698% | Controlled resample | Skip no-op: no source code change to port. |
| 66 | 965 | 0.3080% | Variance resample | Skip no-op: no source code change to port. |
| 67 | 968 | 0.3523% | Selected affine-4 rerank | Covered by final faithful compact C8. |
| 69 | 1031 | 0.2133% | E87 two-dispatch shortlist | Covered by final faithful compact C8. |
| 70 | 1063 | 3.9125% | Hoisted QMV activation chunk sums | Superseded by rejected C7. |
| 71 | 1066 | 0.7439% | E121 affine-2 cluster QMV | Covered by final faithful compact C8. |
| 78 | 1123 | 4.3907% | Tight active-group QMV launch | Superseded by rejected C7. |
| 79 | 1130 | 0.2444% | Restore compact probe fraction | Covered by final faithful compact C8. |
| 80 | 1139 | 0.3477% | Tight QMV launch for M=2 | **C7 rejected** at -0.7253% on the clean C3+C6 regate. |
| 82 | 1153 | 0.3733% | Skip unused compact probe-sort JIT | **C8 retained** at +4.2007% on C3+C6, preserving the pinned precision islands. |

Count: 54 qualifying rows = 8 real cumulative candidates + 23 rows covered
by a later measured descendant + 23 evidence-backed skips.

## Chronological cumulative candidate queue

| Candidate | Source endpoint | Candidate route | Promotion rule |
| --- | ---: | --- | --- |
| C1 | row 3 | target attention Q/K/V packed concat only | Rejected at -4.0629%. |
| C2 | row 13 | GDN packed QKV+Z and B+A projections | Rejected at -7.2989%. |
| C3 | row 20 | K/V-only committed-history append at original request >=16K | Retained at +2.3680%. |
| C4 | row 21 | fused Q/K norm + partial RoPE | Rejected at -0.2034%; tie drift was not the reason. |
| C5 | row 45 | boundary residual/RMSNorm | Rejected at -0.0130%. |
| C6 | row 61 | concat-free dual pre-FC RMSNorm | Retained at +1.7907%. |
| C7 | row 80 | final cross-row affine-4 QMV adapted to group-32 trunk and group-64 islands | Rejected at -0.7253% on C3+C6. |
| C8 | row 82 | final compact selector/reranker with source-pinned precision islands | Retained at +4.2007% on C3+C6. |

No percentage is added or multiplied. If a candidate wins, its implementation
remains enabled in both arms of every later bracket.

## Task 1: Make the benchmark express the real queue

**Files:** `scripts/qwen38_challenge_port_gate.py`,
`tests/test_qwen38_challenge_gate.py`, `mtplx/qwen38_challenge.py`.

- [x] Add route composition for C1-C8 and reject route strings that omit a
  previously retained winner.
- [x] Record packed-QKV calls, fused-GDN calls, compiled-verify engagement,
  recurrence-tape engagement, post-norm reuse, prefill/decode TPS, peak bytes,
  wall time, token hashes, schedules, and full output count per arm.
- [x] Keep deterministic cross-route tie drift as an audit field rather than a
  hash-only rejection.
- [x] Verify with:
  `uv run --frozen --with pytest pytest -q tests/test_qwen38_challenge_gate.py tests/test_qwen38_challenge_contract.py`.

## Task 2: Implement the three missing faithful mechanisms

**Files:** `mtplx/packed_concats.py`, `mtplx/gdn_capture.py`,
`mtplx/qwen38_challenge.py`, `mtplx/qwen38_compact_head.py`,
`tests/test_qwen38_challenge_contract.py`, plus focused kernel tests.

- [x] C1: expose an attention-only packed Q/K/V installer so the arm cannot
  silently include gate/up MLP fusion.
- [x] C2: bind the existing exact two-pair GDN projection packer explicitly to
  the Qwen 3.8 candidate route and materialize packed arrays before timing.
- [x] C8: load the immutable source compact/island artifact by declared
  revision and digest for measurement; bind the island corrections only to
  the proposal MTP attention. A promoted production route must require an
  explicit artifact declaration and must never download weights implicitly.
- [x] Prove tensor/cache equality for C1/C2 and proposal-only isolation for C8
  before running the GPU benchmark.

## Task 3: Run the serial 16K campaign

**Files:** `docs/perf/receipts/qwen38-challenge-port/*.json`,
`docs/perf/qwen38-challenge-port-ledger.md`.

- [x] Acquire the exclusive GPU lock and confirm no unrelated owner for the
  clean confirmation campaign.
- [x] Re-run retained C3 then C6 from the clean immutable candidate commit;
  promote each strict >0.05% winner immediately.
- [x] Re-run C7-C8 if the clean retained base differs from the recorded stack.
- [x] Use one conditioning generation per unique route and exactly four timed
  ABBA arms; do not add BAAB or additional timed arms.
- [x] Reject a candidate on matched wall regression or <=0.05% gain, not a
  deterministic tie-breaking hash difference.

## Task 4: Reduce to the winner stack and update the one PR

**Files:** production files touched by retained candidates, tests, this plan,
the ledger, raw receipts, and `NOTICE`.

- [x] Remove every rejected experimental implementation from production while
  preserving its receipt and row-level disposition.
- [x] Run `uv run --frozen --with pytest pytest -q`, inventory reproduction,
  focused Ruff, receipt schema checks, stub scan, and `git diff --check`. The
  unchanged cold-tier stats-cache race fails on both this branch and
  `upstream/main` (19/20 base reproductions); the full suite excluding only
  that exact base flake passes.
- [x] Update PR #335 in place. Do not open another PR.

## Task 5: Raise the retained DFlash2 stack above 70 decode tok/s

**Files:** `mtplx/qwen38_challenge_kernels.py`,
`mtplx/backends/dflash2.py`, `scripts/qwen38_challenge_dflash_gate.py`,
`scripts/qwen38_challenge_dflash_stack_gate.py`, focused tests, the DFlash
performance table, and a raw receipt.

- [x] Install a construction-validated width router that keeps the existing
  width-4 and width-5 routes, sends width 6--7 to asynchronous per-head GQA,
  and sends width 8 to per-head GQA at the exact Qwen3.8 target shape.
- [x] Expose control/candidate installation in the isolated stack gate without
  adding hot-path counters; prove engagement from the route receipt and the
  adaptive block histogram.
- [x] Run the exact guarded 16K Python / 1,024-output four-arm gate on top of
  the retained production stack. Retain only a strict matched wall win.
- [x] If the retained candidate remains at or below 70 decode tok/s, stack and
  gate the next evidence-backed candidate rather than multiplying estimates.
- [x] Update PR #335 in place with only the retained winner, receipt, table,
  tests, and final production-bundle verification.

## Task 6: Exhaust the live M5--M8 decode widths

**Files:** `mtplx/nax_verify.py`, the Qwen3.8 DFlash gate scripts, focused
tests, the DFlash performance table, and raw receipts.

- [ ] Recheck exact-M5 and the selected M6 K-partition routes as removal gates
  on the final retained stack, using the locked 16K Python / 1,024-output,
  four-arm ABBA protocol.
- [ ] Screen untested M5/M6 morphologies by live shape: barrier-free K=1,
  exact-M5 K partitions, small-N K=4, and only then compile-time-K or dual-pack
  variants. A microbenchmark is only a screen; promotion requires the full
  matched gate.
- [ ] Audit M7/M8 coverage from source and receipts. Test exact-M7 against the
  retained padded-M8 route and screen any remaining shape-specific M8
  morphology; do not rerun a previously closed candidate unchanged.
- [ ] Stack every strict wall-time winner, record prefill/decode TPS, wall time,
  wall delta, peak memory, engagement, and disposition, then run a final
  cumulative confirmation and update PR #335 only.
