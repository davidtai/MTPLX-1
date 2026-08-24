# Qwen 3.8 Phase-Split Regating Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers-optimized:executing-plans because the GPU gates share a rolling
> control and must run serially.

**Goal:** Classify all 54 Yukon rows by their real MTPLX request phase, then
re-gate the live DFlash2 survivors at 1,024 and 16,384 Python input tokens with
1,024 output tokens against the current fully stacked PR.

**Architecture:** The immutable starting control is PR #335 commit
`2b61e79e989a8704a1b0321e51adf87481269cc4`, including DFlash2, retained rows
15/21/24/26/48/50/53, and the final target-shape M5--M8 decode kernels. Rows are
classified by the call site they can actually affect, not by movement in a
noisy metric. Shared rows 24 and 48 are split into independent prefill and
decode ablations; rows 50 and 53 remain shared because their process-wide MLX
policies cannot be safely split.

**Tech Stack:** Python 3.12, MLX 0.32, DFlash2, pytest, four-process ABBA runner,
JSON receipts, and Markdown performance tables.

**Assumptions:**

- The 16,384-token result remains the promotion gate; the 1,024-token result
  records short-context behavior and cannot rescue a 16K loss.
- Every prompt is Python source. Every arm generates exactly 1,024 tokens and
  every isolated process receives one 1,024-token conditioner.
- Temperature is 1.0, top-p is 0.95, top-k is 20, and seed is 42.
- Each prompt length gets exactly four timed ABBA arms: control, candidate,
  candidate, control. The two prompt lengths are separate gates.
- Every GPU run owns `/tmp/mtplx-gpu-exclusive.lock`; `com.tea.qwen` is restored
  and health-checked when the campaign ends.
- Rejected, removed, replaced, dependency-absent, non-transferable, and true
  no-op rows stay visible but are not rerun unchanged.

---

## Exact 54-row phase inventory

`Both` means independently reachable prefill and decode work, or an indivisible
process-wide policy affecting both. The two `Non-runtime` rows remain counted.

| Row | Optimization | Phase | Current-stack treatment |
| ---: | --- | --- | --- |
| 2 | Checkpoint/rejection fast path | Decode | Already present in broader DFlash rollback |
| 3 | Packed target Q/K/V projection | Decode | Rejected verify-only route |
| 4 | Seed-tail logits and lazy capture boundary | Both | Already present; seed/prefill plus decode boundary |
| 5 | Device target top-k sampling | Decode | Required stochastic sampler |
| 6 | Reuse second target argmax | Decode | No-op under stochastic sampling |
| 7 | Persistent committed MTP history | Decode | Replaced by DFlash ownership |
| 8 | Device-resident fixed-D3 draft chain | Decode | Replaced by current DFlash2 control |
| 9 | Paired shared-row target QMV | Decode | Rejected verify-only route |
| 10 | Compact proposal vocabulary | Decode | Incompatible with DFlash selector |
| 11 | Position-EMA adaptive depth | Decode | Superseded by row-11+15 policy |
| 12 | Recurrent prefix-replay tape | Decode | Removed by row 13 |
| 13 | Four-way GDN input projection | Decode | Rejected verify-only route |
| 14 | Reintroduced recurrent prefix replay | Decode | Replaced by DFlash rollback/cache path |
| 15 | Wide adaptive cap | Decode | **Gate retained adaptive policy by removal** |
| 16 | Compiled GDN prologue/post-norm | Decode | Already enclosed by execution graphs |
| 17 | Complete Q4/G64 native MTP block | Decode | Replaced by DFlash checkpoint |
| 18 | Memoized GDN decay and adaptive revision | Decode | Rejected on DFlash |
| 19 | Argmax-only compact selector | Decode | Non-transferable |
| 20 | Packed K/V-only committed-history append | Decode | Replaced by DFlash caches |
| 21 | Fused Q/K RMSNorm plus partial RoPE | Decode | **Gate retained decode hook by removal** |
| 23 | Retuned row-19 reduction | Decode | Dependency absent |
| 24 | Q/K fence, target evaluation ladder, adaptive margin | Both | **Gate prefill and decode legs independently** |
| 25 | Adaptive streak-gate constant | Decode | Rejected on DFlash |
| 26 | Three-layer prefill evaluation cadence | Prefill | **Gate retained prefill route by removal** |
| 28 | Alternate Q4/G64 block | Decode | Rejected; DFlash owns drafter |
| 30 | Reused post-final-norm target output | Decode | Already present in decode capture/history path |
| 32 | Adaptive streak and M=8 retune | Decode | Adaptive and final M8 forms rejected |
| 33 | Transient BF16 proposal Q/K/V islands | Decode | Removed/replaced |
| 34 | M=6/M=9 direct-nibble edits | Decode | M6 rejected; M9 outside 1--8 |
| 36 | Q4/G64 block plus proposal precision islands | Decode | Replaced by DFlash checkpoint |
| 37 | Warm post-norm verify and M=8 revert | Decode | Conditioner-covered/superseded |
| 38 | M=8 direct-nibble extraction | Decode | Superseded; final form rejected |
| 39 | Two two-row M=4 input groups | Decode | Removed by row 40 |
| 40 | Restore M=4 and add M=7 direct nibble | Decode | Rejected on DFlash |
| 41 | Direct-nibble M=3/4/5 target QMV | Decode | Already present/superseded |
| 42 | Affine-2 coarse top-32 proposer | Decode | Non-transferable |
| 45 | Early fused residual/RMS boundary | Both | Removed; live shared form is row 48 |
| 47 | Affine-2 selector and final M=8 grouping | Decode | Non-transferable/rejected |
| 48 | Fused residual/RMSNorm boundary chain | Both | **Gate prefill and decode fusion independently** |
| 50 | Post-warm wired-residency budget | Both | **Gate as indivisible process policy** |
| 53 | 512 MiB / 50-op command buffers | Both | **Gate as indivisible process policy** |
| 59 | Temporary adaptive SDPA cap | Decode | Removed by row 60 |
| 60 | Two-output dual RMSNorm | Decode | Replaced by row 61 |
| 61 | Fused dual RMSNorm plus concatenate | Decode | Native MTP path replaced by DFlash |
| 63 | Fused embedding/norm plus argmax proposer | Decode | Fusion rejected; proposer non-transferable |
| 66 | Resample-ticket artifact note | Non-runtime | Executable bytes unchanged |
| 67 | Direct selected-row Q4 rerank | Decode | Non-transferable |
| 69 | Fused clustered BF16 selectors | Decode | Non-transferable |
| 70 | Clustered proposer plus M3-wide QMV | Decode | Rejected incompatible |
| 71 | Custom centroid/selected-cluster QMV | Decode | Non-transferable |
| 78 | Active-input-group row-70 launch | Decode | Dependency absent |
| 79 | Cluster probe fraction | Decode | Non-transferable |
| 80 | Extend rejected row-70 QMV to M=2 | Decode | Dependency absent |
| 82 | Remove folded-history warm/probe construction | Non-runtime | Conditioner-covered; selector absent |

Totals: **1 prefill-only, 45 decode-only, 6 both, 2 non-runtime = 54**.

## Live gate queue

| Order | Row | Candidate | Lane |
| ---: | ---: | --- | --- |
| 1 | 26 | Disable stride-3 prefill ladder | Prefill |
| 2 | 24 | Disable prefill evaluation ladder only | Prefill |
| 3 | 48 | Disable prefill boundary fusion only | Prefill |
| 4 | 15 | Disable retained row-11+15 adaptive policy | Decode |
| 5 | 21 | Disable fused Q/K RMSNorm plus partial RoPE | Decode |
| 6 | 24 | Disable decode evaluation ladder/fence only | Decode |
| 7 | 48 | Disable decode boundary fusion only | Decode |
| 8 | 50 | Remove wired-residency budget | Both, indivisible |
| 9 | 53 | Restore stock command-buffer policy | Both, indivisible |

Each entry produces two receipts: `python1024in-1024out` and
`python16384in-1024out`.

### Task 1: Freeze and attest the control

**Files:**
- Create: `docs/perf/receipts/qwen38-challenge-port/phase-split-control-2026-08-24.json`
- Modify: `docs/perf/qwen38-challenge-performance-tables.md`

**Security flag:** none

- [ ] Record local HEAD, both remote heads, model/draft revisions, MLX versions,
  sampler contract, and the nine live route receipts.
- [ ] Assert the starting HEAD is exactly
  `2b61e79e989a8704a1b0321e51adf87481269cc4`.
- [ ] Add the inventory above to the canonical table and mechanically verify 54
  unique rows with totals 1/45/6/2.
- [ ] Run `uv run --frozen --with pytest pytest -q tests/test_qwen38_challenge_inventory.py`
  and `git diff --check`; both must pass.

### Task 2: Permit both approved Python prompt lengths

**Files:**
- Modify: `scripts/qwen38_challenge_dflash_stack_gate.py`
- Modify: `tests/test_qwen38_dflash_stack_gate.py`

**Security flag:** none

**Does NOT cover:** Prompt sizes other than 1,024 and 16,384, output sizes other
than 1,024, non-Python prompts, more than four arms, or shared-process timing.

- [ ] Add a failing parameterized test accepting prompt lengths 1,024 and
  16,384 with `max_tokens=1024`, and rejecting 512, 2,048, or non-1,024 output.
- [ ] Introduce `_validate_workload(prompt_tokens, max_tokens)` with exactly
  `{1024, 16_384}` allowed, replacing the hard-coded 16K assertion.
- [ ] Record prompt length and prompt hash in each receipt filename and payload.
- [ ] Run `uv run --frozen --with pytest pytest -q tests/test_qwen38_dflash_stack_gate.py`;
  it must pass.

### Task 3: Add missing phase-isolated ablations

**Files:**
- Modify: `scripts/qwen38_challenge_dflash_stack_gate.py`
- Modify: `mtplx/backends/dflash2.py`
- Modify: `tests/test_qwen38_dflash_stack_gate.py`
- Modify: `tests/test_dflash2_backend.py`

**Security flag:** none

**Does NOT cover:** Reintroducing rejected source kernels, changing sampling, or
changing the DFlash block range.

- [ ] Add route/engagement tests for row 26 prefill disable, row 15 adaptive
  disable, row 21 decode disable, row 24 decode fence plus ladder disable, row
  50 removal, and row 53 stock restoration.
- [ ] Reuse the existing independent row-24 and row-48 prefill/decode controls.
- [ ] Implement only missing environment-controlled ablations. Production
  defaults must remain the current fully retained stack.
- [ ] Assert the disabled phase has zero candidate engagement while unrelated
  retained routes remain engaged.
- [ ] Run the two focused test files; all tests must pass.

### Task 4: Run nine two-context gates serially

**Files:**
- Create: `docs/perf/receipts/qwen38-challenge-port/phase-split-*.json`
- Modify: `docs/perf/qwen38-challenge-performance-tables.md`

**Security flag:** none

- [ ] Acquire the exclusive GPU lock before stopping `com.tea.qwen`.
- [ ] For every queue entry, run one four-arm ABBA gate at 1,024 Python input
  tokens and one at 16,384 Python input tokens.
- [ ] Record prefill TPS, decode TPS, wall time, wall delta, peak memory,
  generated count, token hashes, engagement, source commit, and exact control.
- [ ] Judge prefill legs on 16K prefill TPS, decode legs on 16K decode TPS, and
  indivisible shared policies on 16K wall time. Require a strict gain above
  0.05%; the 1K result is mandatory context-sensitivity evidence.
- [ ] Accept deterministic per-route ties; reject wrong count, fallback,
  wrong-phase, or missing-engagement failures.
- [ ] Restore and health-check `com.tea.qwen` after the final arm.

Expected: 18 receipts, each with exactly four timed arms and 1,024 generated
tokens per arm.

### Task 5: Assemble and verify the two phase stacks

**Files:**
- Modify: `mtplx/backends/dflash2.py`
- Modify: `docs/perf/qwen38-challenge-performance-tables.md`
- Modify: `docs/perf/qwen38-challenge-port-ledger.md`
- Modify: `tests/test_dflash2_backend.py`

**Security flag:** none

**Does NOT cover:** Post-54 M5--M8 discovery; those kernels remain in the
starting control and are already exhausted.

- [ ] Keep only independently measured 16K winners in their phase routes; do
  not add or multiply percentages.
- [ ] Run final 1K/1K and 16K/1K four-arm confirmations comparing the assembled
  phase-corrected stack with the immutable starting control.
- [ ] Publish separate prefill and decode tables with all required metrics and
  receipt paths.
- [ ] Run focused tests, the full repository suite, Ruff on changed Python, and
  `git diff --check`; all must pass.

### Task 6: Update the existing PR only

**Files:**
- Modify: PR #335 description or comment only

**Security flag:** none

- [ ] Commit classification, harness changes, receipts, final routing, and
  tables to `perf/qwen38-challenge-port`.
- [ ] Push the same commit to the existing `origin` and `mtplx1` mirrors.
- [ ] Update PR #335 with both phase tables and final cumulative receipts.
- [ ] Do not create another branch or pull request.

