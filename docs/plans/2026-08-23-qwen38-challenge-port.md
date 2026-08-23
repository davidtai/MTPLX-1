# Qwen 3.8 Challenge Optimization Port Implementation Plan

> **For implementation:** REQUIRED SUB-SKILL: use
> `superpowers-optimized:executing-plans` and execute tasks in order. Do not
> start kernel or model work until the inventory and unchanged-control gates
> are committed.

**Goal:** Port the qualifying, transferable, still-live Qwen 3.8 challenge
optimizations into MTPLX, retain only independently verified MTPLX wins, and
open one PR from `perf/qwen38-challenge-port` to `youssofal/MTPLX:main`.

**Architecture:** Add a construction-selected Qwen 3.8 optimization lane that
binds immutable proposal, QMV, projection, cache-append, and
policy callables after validating the real model contract. Reuse MTPLX's
existing committed history, capture/commit verification, NAX/compiled verify,
session identity, and adaptive-policy infrastructure. Every candidate has an
off/control route for matched A/B, but production generation does not perform
per-token feature eligibility checks or silent fallback.

**Tech stack:** Python 3.12, MLX 0.32.x, mlx-lm 0.31.x, Metal kernels through
`mx.fast.metal_kernel`, pytest, Ruff, guarded M5 Max benchmarks, GitHub CLI.

**Source design:**
`docs/specs/2026-08-23-qwen38-challenge-port-design.md`

**Pinned inputs:**

- MTPLX base `bd4421567f9e16ce957c6ef97708b072dcd73937`.
- Challenge source `eb5eadc7a165047d4321ce883b9ff30894d8bd19`.
- Yukon board snapshot rendered 2026-08-23 with 82 accepted rows.
- Source filter `(score / previous promoted score - 1) * 100 > 0.10`.

**Execution constraints:**

- Use `/tmp/mtplx-gpu-exclusive.lock` for all real-model GPU work.
- Keep Qwen 3.8 work pinned to the named 27B artifacts. Do not substitute a
  Qwen 3.5/3.6 checkpoint or infer compatibility from topology.
- Preserve shape, dtype, group size, bit packing, ownership, tiling, cache
  layout, compilation behavior, and workload timing boundaries.
- Reject a candidate at the first matched A/B regression or parity failure.
- Give every implemented candidate a first sanity bracket on
  `python_modules_long.jsonl` at approximately 100 generated tokens. Require
  identical token hashes before any longer or broader benchmark. Also require
  identical attempted/accepted depth schedules for candidates that do not
  intentionally change proposal selection; for proposal-only candidates,
  record the schedule and acceptance delta and reject an acceptance collapse.
- Do not retain code merely because its source challenge score improved.
- Do not push or open the PR until all retained candidates and the cumulative
  stack pass the final gate.

---

## Planned file structure

- Create `mtplx/qwen38_challenge.py`: exact model-contract validation,
  immutable route specification, installer, route fingerprint, and counters.
- Create `mtplx/qwen38_challenge_kernels.py`: final wide QMV, compact selector,
  projection, GDN, RMSNorm, and attention kernels that earn
  promotion.
- Create `mtplx/qwen38_compact_head.py`: compact-vocabulary manifest loading,
  cluster mapping, coarse selection, exact reranking, and proposal-only API.
- Modify `mtplx/qwen3_5_mtp_patch.py`: Qwen 3.8 K/V-only MTP cache update and
  construction-bound optimized MTP methods.
- Modify `mtplx/runtime.py`: install and expose the immutable Qwen 3.8 route.
- Modify `mtplx/generation.py`: consume bound proposal/cache/policy callables;
  preserve existing history and rollback contracts.
- Modify `mtplx/nax_verify.py`: compose, rather than conflict, with promoted
  width routes.
- Modify `mtplx/kernel_selfcheck.py`: exact per-lane construction self-checks.
- Modify `mtplx/profiles.py`: explicit candidate and retained-stack profile
  wiring after measurement.
- Modify `mtplx/session_bank.py` and policy fingerprint plumbing: include head,
  kernel, and policy route identity.
- Create `scripts/qwen38_challenge_port_gate.py`: locked matched A/B runner.
- Create `scripts/qwen38_challenge_inventory.py`: reproduce the 82-row source
  ledger from a pinned Yukon export and challenge commit map.
- Create `tests/test_qwen38_challenge_contract.py`.
- Create `tests/test_qwen38_challenge_kernels.py`.
- Create `tests/test_qwen38_compact_head.py`.
- Create `tests/test_qwen38_challenge_generation.py`.
- Update `NOTICE`, `docs/perf/qwen38-challenge-port-ledger.md`, and receipt
  documents under `docs/perf/receipts/qwen38-challenge-port/`.

## Task 1: Freeze the source inventory and unchanged MTPLX control

**Files:**

- Create: `scripts/qwen38_challenge_inventory.py`
- Create: `tests/test_qwen38_challenge_inventory.py`
- Create: `docs/perf/qwen38-challenge-port-ledger.md`
- Modify: `docs/specs/2026-08-23-qwen38-challenge-port-design.md` only if the
  reproducible inventory finds a mismatch

**Does not cover:** No runtime or benchmark behavior changes in this task.

- [x] **Step 1: Write a failing inventory test**

The fixture must assert:

- exactly 82 accepted/promoted Yukon rows;
- chronological submission IDs, official scores, and source commits;
- relative deltas computed from consecutive official scores;
- exactly 54 rows above 0.10 percent;
- every row has one disposition from the approved design ledger;
- every `PORT` or `DEPENDENCY` row maps to a challenge PR and source commit.

- [x] **Step 2: Run the red test**

```bash
PY=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
"$PY" -m pytest -q tests/test_qwen38_challenge_inventory.py
```

Expected: import or fixture failure because the inventory tool and checked-in
source receipt do not exist.

- [x] **Step 3: Implement deterministic inventory generation**

The tool accepts only explicit local inputs: a saved Yukon payload, the pinned
challenge checkout, and the approved threshold. It must not silently refetch a
changing leaderboard during plan execution. Emit the full 82-row table, the 54
qualifying rows, PR/commit mappings, dispositions, and SHA-256 of every input.

- [x] **Step 4: Capture the unchanged control contract**

Record:

- MTPLX commit, dirty state, Python/MLX/mlx-lm versions;
- exact Qwen 3.8 model and MTP-head identities;
- quantization map, real M/K/N widths, group size, packing, and dtype;
- enabled profile, NAX/compiled-verify routes, history and policy settings;
- prompt set, seed, prompt length, decode length, warmup, and timing boundary.

Do not run the real model if the GPU lock is held by another owner.

- [x] **Step 5: Verify and commit the inventory only**

```bash
"$PY" -m pytest -q tests/test_qwen38_challenge_inventory.py
"$PY" scripts/qwen38_challenge_inventory.py --check
git diff --check
git add scripts/qwen38_challenge_inventory.py \
  tests/test_qwen38_challenge_inventory.py \
  docs/perf/qwen38-challenge-port-ledger.md \
  docs/specs/2026-08-23-qwen38-challenge-port-design.md
git commit -m "Document Qwen 3.8 challenge port inventory"
```

## Task 2: Add an immutable Qwen 3.8 route contract

**Files:**

- Create: `mtplx/qwen38_challenge.py`
- Create: `tests/test_qwen38_challenge_contract.py`
- Modify: `mtplx/runtime.py`
- Modify: `mtplx/session_bank.py`
- Modify: `mtplx/server/openai.py`

**Does not cover:** The control route remains behavior-identical and no new
kernel is available yet.

- [x] **Step 1: Write failing construction and identity tests**

Cover:

- exact Qwen 3.8 27B topology and quantization acceptance;
- rejection of wrong family, dimensions, dtype, bit width, group size, or
  packing without fallback;
- one immutable route spec selected before generation;
- route fingerprint changes with compact-head digest, kernel set, or policy;
- session-bank restore rejects incompatible route identity;
- health and completion receipts report selected route and self-check status.

- [x] **Step 2: Run the red tests**

```bash
"$PY" -m pytest -q \
  tests/test_qwen38_challenge_contract.py \
  tests/test_qwen38_family.py \
  tests/test_session_bank.py
```

- [x] **Step 3: Implement the route dataclass and installer**

The bound spec contains callables for proposal readout, QMV
widths, MTP cache append, projection fusions, and policy selection. Missing
promoted callables fail construction for the candidate route. The unchanged
route binds current MTPLX functions directly.

- [x] **Step 4: Verify no behavior change**

```bash
"$PY" -m pytest -q \
  tests/test_qwen38_challenge_contract.py \
  tests/test_qwen38_family.py \
  tests/test_qwen3_5_mtp_backend.py \
  tests/test_mtp_alias_load_path.py \
  tests/test_session_bank.py
git diff --check
```

- [x] **Step 5: Commit the contract seam**

```bash
git add mtplx/qwen38_challenge.py mtplx/runtime.py mtplx/session_bank.py \
  mtplx/server/openai.py tests/test_qwen38_challenge_contract.py
git commit -m "Add Qwen 3.8 optimization route contract"
```

## Task 3: Close hierarchical target top-2 as non-transferable

**Source rows:** 5 and 6.

The source win removed duplicate work only because the trusted challenge worker
was required to materialize a per-row target top-2 ledger. MTPLX's production
greedy verifier consumes one target argmax and has no target top-2 consumer.
Porting the two-stage reducer would therefore add two dispatches instead of
removing work.

- [x] **Step 1: Inspect the exact source dependency**

Confirmed PR #29 introduced mandatory hierarchical target top-2 production and
PR #37 reused its first ID in place of a separate target argmax.

- [x] **Step 2: Inspect the current MTPLX target acceptance path**

Confirmed `generation.py` computes a single target `mx.argmax`; no target
top-2 values or IDs are consumed by generation, history, policy, or receipts.

- [x] **Step 3: Record the source rows as challenge-only**

Mark rows 5 and 6 `CHALLENGE-ONLY` / `SKIP`. Do not add a kernel, dispatch,
feature flag, test-only implementation, or performance claim.

## Task 4: Remove dead full-layer work from MTP history append

**Source rows:** remaining portion of 11 and row 20.

**Files:**

- Modify: `mtplx/mtp_patch.py`
- Modify: `mtplx/qwen38_challenge.py`
- Modify: `mtplx/runtime.py`
- Modify: `tests/test_mtp_depth_n.py`
- Create: `scripts/qwen38_challenge_port_gate.py`

- [x] **Step 1: Write failing cache-update tests**

Assert that the optimized Qwen 3.8 cache update:

- performs embedding normalization, hidden normalization, fusion, input
  normalization, K/V projection, RoPE, and the cache write;
- does not invoke Q/gate, attention, output projection, MoE, MTP norm, or the
  vocabulary head;
- produces the exact same K/V cache state as the current full forward;
- never appends rejected speculative rows;
- preserves session and prefix restore behavior.

- [x] **Step 2: Run the red tests**

```bash
"$PY" -m pytest -q \
  tests/test_mtp_depth_n.py \
  tests/test_qwen38_challenge_contract.py \
  -k 'kv_only_history or kv_only_candidate'
```

- [x] **Step 3: Add a K/V-only MTP history method**

Bind the optimized method only for the validated Qwen 3.8 route. Do not change
generic MTP or other Qwen families.

- [x] **Step 4: Close early first-draft submission as already present**

The host must materialize each sampled proposal ID before it can build the next
MTP step, so the first proposal is already submitted at that dependency
boundary. No additional async dispatch or cache owner is introduced.

- [x] **Step 5: Verify and benchmark**

The exact cache test and focused suites pass. The required 100-token Python
sanity brackets preserve token hash
`485303a13e681058a2d25bf216898ec321dbd45e3a107e12a6d87276cbad2388`
and `[23,20,17] / [25,25,25]` accepted/drafted counts in both orderings,
improving wall time by 3.56% in ABBA and 1.69% in BAAB. Locked longer brackets
also preserve token hashes and depth schedules.

- [x] **Step 6: Commit only retained changes**

```bash
git add mtplx/mtp_patch.py mtplx/qwen38_challenge.py mtplx/runtime.py \
  tests/test_mtp_depth_n.py tests/test_qwen38_challenge_contract.py \
  scripts/qwen38_challenge_port_gate.py \
  docs/perf/qwen38-challenge-port-ledger.md \
  docs/perf/receipts/qwen38-challenge-port/kv-only-history-*.json
git commit -m "Optimize Qwen 3.8 committed MTP cache updates"
```

## Task 5: Port the compact proposal-head stack

**Source rows:** 10, 42, 47, 67, 69, 71, 79, 82; row 46 dependency.

**Files:**

- Create: `mtplx/qwen38_compact_head.py`
- Create: `tests/test_qwen38_compact_head.py`
- Modify: `mtplx/qwen38_challenge_kernels.py`
- Modify: `mtplx/qwen38_challenge.py`
- Modify: `mtplx/draft_lm_head.py`
- Modify: `mtplx/kernel_selfcheck.py`
- Modify: `mtplx/session_bank.py`

- [ ] **Step 1: Write failing manifest and mapping tests**

Validate digest, byte size, source revision, vocabulary size, cluster count,
Q2/Q4 packing, group size, row mapping, exact-rerank width, and head identity.
Reject missing or incompatible artifacts at construction.

- [ ] **Step 2: Write failing proposal-boundary tests**

Assert:

- coarse affine-2 selection and exact affine-4 rerank stay proposal-only;
- candidate IDs are in range and deterministic;
- E87 and reference selector candidate sets match on the supported contract;
- probe fraction is exactly 0.15 after integer rounding;
- the unused fallback sorter is not constructed when E87 is selected;
- target-emitted tokens remain identical even when proposal IDs differ.

- [ ] **Step 3: Run the red tests**

```bash
"$PY" -m pytest -q tests/test_qwen38_compact_head.py
```

- [ ] **Step 4: Implement artifact loading and final selector**

Implement the final stack directly; do not reproduce superseded full sort,
centroid redraw, or intermediate selector variants. Construction selects E87
or fails closed for the candidate route.

- [ ] **Step 5: Implement the 32-value/lane affine-2 cluster QMV**

Match real N=12292/non-multiple-of-eight behavior, 3073-tile gather, packing,
and dtype. Test tail rows explicitly.

- [ ] **Step 6: Verify proposal safety and locked performance**

Run unit/self-check tests, a deterministic prompt corpus parity test, then
locked A/B for proposal time, full MTP time, acceptance-by-depth, and output
hashes. Reject the stack if target output changes or end-to-end performance is
neutral/regressing.

- [ ] **Step 7: Commit only if retained**

```bash
git add mtplx/qwen38_compact_head.py mtplx/qwen38_challenge_kernels.py \
  mtplx/qwen38_challenge.py mtplx/draft_lm_head.py \
  mtplx/kernel_selfcheck.py mtplx/session_bank.py \
  tests/test_qwen38_compact_head.py \
  docs/perf/qwen38-challenge-port-ledger.md
git commit -m "Add compact Qwen 3.8 proposal head"
```

## Task 6: Port the final wide affine-4/group-64 QMV family

**Source rows:** 19, 34, 36, 39, 40, 41, 70, 78, 80. Rows 9, 15, and 44 are
superseded and must not be recreated.

**Files:**

- Modify: `mtplx/qwen38_challenge_kernels.py`
- Modify: `mtplx/qwen38_challenge.py`
- Modify: `mtplx/nax_verify.py`
- Modify: `mtplx/kernel_selfcheck.py`
- Modify: `tests/test_qwen38_challenge_kernels.py`
- Modify: `tests/test_nax_verify.py`

- [ ] **Step 1: Write failing shape/ownership/packing tests**

For M=2..9 and every real Qwen 3.8 projection shape, compare with stock MLX.
Cover non-divisible N, last group, chunk-sum table boundaries, direct-nibble
packing, and exact active-group counts. Assert no missing or overlapping row
ownership.

- [ ] **Step 2: Inventory current NAX overlap**

Record which M/K/N shapes current Turbo already routes through NAX or
`verify_kernels`. For an overlapping width, the challenge port competes against
the existing MTPLX winner; it does not replace it by default.

- [ ] **Step 3: Implement one final width-plan table**

The plan owns input rows per threadgroup, chooses direct versus reusable sums,
and computes `ceil(M / inputs_per_group)` active groups. Unsupported widths
fail construction. Both table and direct paths share the same geometry witness.

- [ ] **Step 4: Add reusable activation chunk sums**

Build once per activation row set and reuse only where the table amortizes.
Keep the source stack's minimum-width boundary; do not enable the table at M=2.

- [ ] **Step 5: Run exactness and isolated width benchmarks**

```bash
"$PY" -m pytest -q \
  tests/test_qwen38_challenge_kernels.py \
  tests/test_nax_verify.py \
  tests/test_kernel_selfcheck.py
```

Then run locked stock-vs-NAX-vs-challenge brackets per real width. Retain the
fastest exact/approved route per shape. Stop at the first regression for each
candidate width.

- [ ] **Step 6: Run mixed-model one-cycle and 512-token A/B**

An isolated kernel win is insufficient. Verify command scheduling, acceptance,
thermal behavior, and cumulative decode TPS with the real mixed graph.

- [ ] **Step 7: Commit the retained width table only**

```bash
git add mtplx/qwen38_challenge_kernels.py mtplx/qwen38_challenge.py \
  mtplx/nax_verify.py mtplx/kernel_selfcheck.py \
  tests/test_qwen38_challenge_kernels.py tests/test_nax_verify.py \
  docs/perf/qwen38-challenge-port-ledger.md
git commit -m "Optimize Qwen 3.8 wide verification QMV"
```

## Task 7: Port retained Qwen projection and GDN fusions

**Source rows:** 3, 13, 16, and 30.

**Files:**

- Modify: `mtplx/qwen38_challenge_kernels.py`
- Modify: `mtplx/qwen38_challenge.py`
- Modify: `mtplx/qwen3_5_mtp_patch.py`
- Modify: `mtplx/kernel_selfcheck.py`
- Modify: `tests/test_qwen38_challenge_kernels.py`
- Modify: `tests/test_qwen38_challenge_generation.py`

- [ ] **Step 1: Write failing boundary tests**

Lock the current output at Q/K/V projection, each GDN input projection, GDN
satellite expression, attention output gate, and target verify boundary. Test
M=1..9 and real layer-specific layouts.

- [ ] **Step 2: Port one fusion at a time**

Order:

1. fused affine-4 Q/K/V;
2. fused GDN input projection;
3. compiled GDN satellites;
4. schedule-neutral verify reuse/output gate.

After each item, run focused exactness and a locked one-cycle A/B. Do not bundle
an unmeasured fusion with the next item.

- [ ] **Step 3: Verify target arithmetic and cache capture**

Run GDN capture, snapshot-free repair, target-prefix, compiled verify, and
Qwen-family suites. Reject any fusion that changes a target decision outside
the currently approved MTPLX tolerance.

- [ ] **Step 4: Commit each retained fusion separately**

Use one commit per retained mechanism so the final cumulative regression can be
bisected and a losing fusion can be removed without disturbing others.

## Task 8: Port retained attention and normalization fusions

**Source rows:** 18, 21, 45, 60, and 61.

**Files:**

- Modify: `mtplx/qwen38_challenge_kernels.py`
- Modify: `mtplx/qwen38_challenge.py`
- Modify: `mtplx/qwen3_5_mtp_patch.py`
- Modify: `mtplx/kernel_selfcheck.py`
- Modify: `tests/test_qwen38_challenge_kernels.py`
- Modify: `tests/test_qwen38_challenge_generation.py`

- [ ] **Step 1: Write failing exactness tests for each boundary**

Cover SDPA width bridging, Q/K RMSNorm, partial RoPE, residual/RMSNorm, copy
elision, dual pre-fc RMSNorm, and concatenation-free output. Include first and
later decode windows and every active verify width.

- [ ] **Step 2: Port and gate in source order**

Keep arithmetic and reduction order. A copy may be removed only when ownership
and lifetime tests show no later mutation or alias consumer.

- [ ] **Step 3: Test composition with packed-GQA and compiled verify**

The current Turbo profile remains the control. Benchmark both short and
later-window contexts so a local launch reduction does not hide an SDPA or
compile regression.

- [ ] **Step 4: Commit each retained mechanism and ledger result**

Rejected fusions leave documentation and receipts only.

## Task 9: Evaluate the precision-island compact-head artifact

**Source rows:** 33 and 36.

**Files:**

- Modify: `mtplx/qwen38_compact_head.py`
- Modify: `mtplx/profiles.py`
- Modify: `NOTICE`
- Create: `docs/perf/receipts/qwen38-challenge-port/head-manifest.json`
- Modify: `tests/test_qwen38_compact_head.py`

- [ ] **Step 1: Write failing provenance and artifact tests**

Require immutable source revision, SHA-256, byte size, license, complete tensor
map, exact precision-island rows, Q2/Q4 packing, and model-family identity.

- [ ] **Step 2: Reproduce the artifact outside the timed benchmark**

Do not runtime-requantize the head. Verify the generated artifact equals the
declared digest and loads without rewriting target weights.

- [ ] **Step 3: Run proposal and target parity gates**

Measure acceptance and proposal cost while asserting identical target output.
Compare against MTPLX's existing Q4 draft head and the retained compact-head
code without islands.

- [ ] **Step 4: Keep only the winning artifact choice**

If the artifact does not independently improve the MTPLX bracket, retain the
code-compatible manifest support only if another promoted feature requires it;
otherwise remove it and record rejection.

## Task 10: Bind final policy and warm shapes without adding a controller

**Source rows:** remaining portions of 11, 38, and 59.

**Files:**

- Modify: `mtplx/adaptive.py` only if a reusable parameter is missing
- Modify: `mtplx/qwen38_challenge.py`
- Modify: `mtplx/profiles.py`
- Modify: `mtplx/background_warmup.py`
- Modify: `tests/test_cost_depth_policy.py`
- Create/modify: `tests/test_qwen38_challenge_generation.py`

- [ ] **Step 1: Write failing policy-construction tests**

Assert one existing MTPLX policy object receives Qwen 3.8-specific immutable
parameters. Test legal zero-draft escape, caps, EMA update, boundary prices,
and fingerprint identity. No environment read occurs in the decode hot path.

- [ ] **Step 2: Write failing warm-shape tests**

Assert the exact retained `callWithHiddenAndNormed` and seed shapes warm before
timing, while unrelated deep context buckets remain lazy. This must not restore
the previously removed boot-time full warm ladder.

- [ ] **Step 3: Implement construction-only calibration**

Do not recreate historical streak-gate variants. Bind only the final candidate
calibration and retain current MTPLX defaults as the control.

- [ ] **Step 4: Run schedule-replay and locked A/B gates**

Compare per-prompt depth schedules, non-drafting rounds, acceptance-by-position,
cycle time, and output hashes. Reject hidden-prompt overfit that loses on the
MTPLX prompt corpus.

- [ ] **Step 5: Commit only if retained**

```bash
git add mtplx/adaptive.py mtplx/qwen38_challenge.py mtplx/profiles.py \
  mtplx/background_warmup.py tests/test_cost_depth_policy.py \
  tests/test_qwen38_challenge_generation.py \
  docs/perf/qwen38-challenge-port-ledger.md
git commit -m "Calibrate Qwen 3.8 challenge route"
```

## Task 11: Run the cumulative promotion funnel

**Files:**

- Create: `scripts/qwen38_challenge_port_gate.py`
- Create: `tests/test_qwen38_challenge_port_gate.py`
- Update: `docs/perf/qwen38-challenge-port-ledger.md`
- Create/update: `docs/perf/receipts/qwen38-challenge-port/*`

- [ ] **Step 1: Write failing runner-contract tests**

The runner must require:

- exact model/head/source revisions;
- clean or explicitly recorded worktree state;
- exclusive GPU lock ownership;
- fixed prompt/seed/decode/timing contract;
- unchanged control and named candidate routes;
- raw per-prompt records, output hashes, acceptance histograms, depth schedules,
  kernel counters, compile counters, thermal data, memory data, and environment;
- early-stop rejection after the first matched regression gate.

- [ ] **Step 2: Run the red test and implement the runner**

```bash
"$PY" -m pytest -q tests/test_qwen38_challenge_port_gate.py
```

- [ ] **Step 3: Run candidate-by-candidate matched brackets**

For each retained commit:

1. exactness/self-check;
2. one-cycle real-shape timing;
3. approximately 100 generated tokens on `python_modules_long.jsonl`, with
   identical token hashes plus either identical schedules or a documented,
   non-collapsing proposal-only schedule delta;
4. longer matched A/B only after the earlier gates pass.

Remove candidates that fail. Do not average a regression into a later bundle.

- [ ] **Step 4: Run the cumulative stack against unchanged main**

Use at least two ordering reversals (ABBA/BAAB), identical warmup, and the same
process lifecycle. Report median, every pair, and uncertainty/noise caveat.
Verify output hashes and target-token parity for all prompts.

- [ ] **Step 5: Run the full focused and general test suites**

```bash
"$PY" -m pytest -q \
  tests/test_qwen38_challenge_inventory.py \
  tests/test_qwen38_challenge_contract.py \
  tests/test_qwen38_challenge_kernels.py \
  tests/test_qwen38_compact_head.py \
  tests/test_qwen38_challenge_generation.py \
  tests/test_qwen38_family.py \
  tests/test_qwen3_5_mtp_backend.py \
  tests/test_nax_verify.py \
  tests/test_cost_depth_policy.py \
  tests/test_kernel_selfcheck.py \
  tests/test_session_bank.py
"$PY" -m pytest -q
"$PY" -m ruff check mtplx tests scripts
git diff --check
```

Expected: all tests pass; any unrelated pre-existing failure is recorded with
an unchanged-main reproduction before proceeding.

## Task 12: Attribution, documentation, review, and separate PR

**Files:**

- Modify: `NOTICE`
- Modify: `docs/perf/qwen38-challenge-port-ledger.md`
- Modify: `README.md` or `docs/turbo-verify.md` only for retained public behavior
- Update: plan/spec status and final receipts

- [ ] **Step 1: Add MIT provenance**

Name `Layr-Labs/qwen-3.8-mtp-challenge`, its pinned commit, MIT license, and the
specific adapted modules. Preserve any source-level copyright notices.

- [ ] **Step 2: Review the final diff against the 82-row ledger**

Every production change must map to a `PORT` row or an explicitly named
`DEPENDENCY`. Confirm no `WEAK`, `SUPERSEDED`, or `CHALLENGE-ONLY` mechanism
survived accidentally.

- [ ] **Step 3: Run verification-before-completion**

Capture fresh test, lint, worktree, commit, benchmark, and receipt output. Run a
code review focused on arithmetic, cache ownership, route identity, artifact
provenance, and benchmark validity. Address findings and rerun affected gates.

- [ ] **Step 4: Prepare the PR**

The PR body must include:

- base and challenge source pins;
- the 82-row ledger and 54-row source gate summary;
- retained candidate ledger with commits and receipts;
- rejected/skip summary;
- exact test commands and results;
- matched A/B raw receipt links;
- licensing attribution;
- caveats separating challenge scores from MTPLX measurements.

- [ ] **Step 5: Push and open one PR against main**

Use the authenticated user's MTPLX fork. Confirm the head branch is
`perf/qwen38-challenge-port`, the base is `youssofal/MTPLX:main`, and no other
campaign's commits are present. Do not merge unless separately requested.
