# PR391 Metal D3 MTP Draft Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaust the measurable optimization opportunity inside the fixed Metal D1-selector-D2-selector-D3-selector drafting lane without changing target verification, acceptance, cache-commit, or production model routes.

**Architecture:** The installed request route continues to call `rt.draft_mtp` three times inside one compiled MLX dependency graph, preserving QSA and FRSpec. The graph returns three packed leaves instead of twelve separate leaves, and the host converts each packed leaf once. Route proof is derived after timing from existing event tags and compiled-verifier statistics rather than aggregate drafted-token counts.

**Tech Stack:** Python 3.12, NumPy 2.4.4, MLX 0.32.2, `mx.compile`, `mx.fast.metal_kernel`, pytest, guarded macOS Metal benchmark runner.

**Assumptions:**

- Assumes the approved experiment remains fixed at D3, temperature 1, top-k 20, top-p 0.95 — it will NOT handle adaptive depth, penalties, constraints, or steering.
- Assumes the production QSA/FRSpec callables installed in `rt.draft_mtp` remain authoritative — it will NOT replace model kernels with a whole-model megakernel.
- Assumes float32 is benchmark-only — it will NOT be retained as exact PR391 evidence.
- Assumes target verification and acceptance are outside this goal — it will NOT move target sampling or cache commit onto device.

---

## File Structure

- `mtplx/generation.py`: pack compiled D3 tensor outputs and perform one host conversion per packed leaf.
- `scripts/pr391_metal_choice_benchmark_launcher.py`: build and validate an honest post-timer D3 route receipt from existing generation events and compiled-verifier statistics.
- `tests/test_pr391_float32_d3_core.py`: lock packed shapes, three-depth chaining, one evaluation, and unchanged QSA/FRSpec ownership.
- `tests/test_pr391_metal_choice_benchmark_launcher.py`: lock route partitioning for D3, context-copy, fixed-M4, and shortened cycles.
- `.benchmark-artifacts/pr391/`: store guarded smoke, A/B, profiler, drift, and memory receipts; never commit artifacts.

### Task 1: Correct D3 Route Attribution

**Files:**
- Modify: `scripts/pr391_metal_choice_benchmark_launcher.py`
- Test: `tests/test_pr391_metal_choice_benchmark_launcher.py`

**Security flag:** none

**Does NOT cover:** It does not add generation-loop counters or change route selection; it analyzes already-recorded output events after timing.

- [ ] **Step 1: Write failing mixed-route test**

```python
def test_route_receipt_partitions_actual_d3_and_context_copy_events() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import build_d3_route_counts

    events = [
        {"drafts": [{"draft_core": "pr391-float32-d3-test-only"}] * 3},
        {"drafts": [{"draft_core": "context_copy"}]},
        {"context_copy": {"lane": "batched", "block": 8}},
        {"drafts": [{"draft_core": "pr391-float32-d3-test-only"}] * 3},
    ]
    assert build_d3_route_counts(events) == {
        "d3_cycles": 2,
        "d3_rows": 6,
        "context_copy_substitutions": 1,
        "context_copy_block_rounds": 1,
        "other_draft_rows": 0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_pr391_metal_choice_benchmark_launcher.py::test_route_receipt_partitions_actual_d3_and_context_copy_events`

Expected: FAIL because `build_d3_route_counts` does not exist.

- [ ] **Step 3: Implement post-timer partition and receipt validation**

```python
def build_d3_route_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "d3_cycles": 0,
        "d3_rows": 0,
        "context_copy_substitutions": 0,
        "context_copy_block_rounds": 0,
        "other_draft_rows": 0,
    }
    for event in events:
        if isinstance(event.get("context_copy"), Mapping):
            counts["context_copy_block_rounds"] += 1
        drafts = event.get("drafts") or ()
        d3_rows = sum(
            draft.get("draft_core") == "pr391-float32-d3-test-only"
            for draft in drafts
            if isinstance(draft, Mapping)
        )
        if d3_rows:
            if d3_rows != 3:
                raise RuntimeError("PR391 D3 event must contain exactly three rows")
            counts["d3_cycles"] += 1
            counts["d3_rows"] += d3_rows
        for draft in drafts:
            if not isinstance(draft, Mapping):
                continue
            route = draft.get("draft_core")
            if route == "context_copy":
                counts["context_copy_substitutions"] += 1
            elif route != "pr391-float32-d3-test-only":
                counts["other_draft_rows"] += 1
    return counts
```

Pass `output.events` into `finish_receipt`, store the partition under
`route_counts`, and validate `calls == d3_rows`; do not compare D3 calls with
aggregate `drafted_tokens`.

- [ ] **Step 4: Run focused launcher tests**

Run: `pytest -q tests/test_pr391_metal_choice_benchmark_launcher.py`

Expected: PASS.

### Task 2: Pack the D3 Graph Outputs

**Files:**
- Modify: `mtplx/generation.py`
- Test: `tests/test_pr391_float32_d3_core.py`

**Security flag:** none

**Does NOT cover:** It does not alter QSA arithmetic, FRSpec ordering, selector arithmetic, RNG consumption, target verification, or host acceptance.

- [ ] **Step 1: Change the focused test to require three packed leaves**

Add `concatenate` shape support to `_FakeMX`, then require:

```python
assert len(result) == 3
assert result[0].shape == (1, 3)
assert result[1].shape == (3, 20)
assert result[2].shape == (3, 20)
assert len(fake_mx.eval_calls) == 1
assert fake_mx.eval_calls[0] == result
```

Also inspect `_pr391_decode_float32_d3_outputs` and require exactly three
`np.asarray` calls and no `.item()` call.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_pr391_float32_d3_core.py::test_d3_chain_keeps_selected_tokens_lazy_and_evaluates_once`

Expected: FAIL because the graph currently returns twelve leaves.

- [ ] **Step 3: Return packed tokens, IDs, and probabilities**

Inside `chain_fn`, stop exporting unused raw values and return:

```python
return (
    mx.concatenate(selected_tokens, axis=1),
    mx.concatenate(raw_ids_by_depth, axis=0),
    mx.concatenate(raw_probs_by_depth, axis=0),
)
```

Decode the packed results with one conversion per leaf:

```python
token_rows = np.asarray(result[0], dtype=np.uint32).reshape(3)
id_rows = np.asarray(result[1], dtype=np.uint32).reshape(3, 20)
prob_rows = np.asarray(result[2], dtype=np.float32).reshape(3, 20)
tokens = [int(token) for token in token_rows]
```

Build the same three `SparseDistribution` values from the converted matrices;
do not change their float64 constructor normalization.

- [ ] **Step 4: Run focused D3 tests**

Run: `pytest -q tests/test_pr391_float32_d3_core.py tests/test_tensor_offset_qsa_compile_state.py`

Expected: PASS.

### Task 3: Verify the Complete D3-only CPU Contract

**Files:**
- Verify only: `mtplx/generation.py`
- Verify only: `mtplx/graphbank.py`
- Verify only: `mtplx/pcg64_tape.py`
- Verify only: `scripts/pr391_metal_choice_benchmark_launcher.py`

**Security flag:** none

**Does NOT cover:** This task performs no MLX/Metal execution and makes no performance claim.

- [ ] **Step 1: Run the focused test set**

Run:

```bash
pytest -q \
  tests/test_pr391_float32_d3_core.py \
  tests/test_pr391_metal_choice_benchmark_launcher.py \
  tests/test_pcg64_uniform_tape.py \
  tests/test_tensor_offset_qsa_compile_state.py
```

Expected: PASS.

- [ ] **Step 2: Check the hot-path source contract**

Run:

```bash
git diff --check
rg -n "os\.environ|try:|fallback|mx\.random" \
  mtplx/generation.py scripts/pr391_metal_choice_benchmark_launcher.py
```

Expected: no new environment read, try/fallback branch, or MLX RNG in the PR391
D3 functions; unrelated pre-existing matches are reviewed by function context.

- [ ] **Step 3: Review the scoped diff**

Run:

```bash
git diff -- \
  mtplx/generation.py \
  scripts/pr391_metal_choice_benchmark_launcher.py \
  tests/test_pr391_float32_d3_core.py \
  tests/test_pr391_metal_choice_benchmark_launcher.py
```

Expected: only route-receipt and packed-D3 changes.

### Task 4: Guarded Metal Proof and D3 Optimization Loop

**Files:**
- Create artifact: `.benchmark-artifacts/pr391/pr391-metal-d3-packed-smoke.json`
- Create artifact: `.benchmark-artifacts/pr391/pr391-metal-d3-packed-seeds-16k-1k.json`
- Create artifact when needed: `.benchmark-artifacts/pr391/pr391-metal-d3-packed-profile.jsonl`

**Security flag:** none

**Does NOT cover:** It does not update the PR body, retain float32 behavior, or alter the production service configuration.

- [ ] **Step 1: Inspect GPU/service/memory state read-only**

Run the read-only preflight commands from
`docs/GPU_LOCK_AND_SERVICE_RUNBOOK.md`. Resolve the exact service command,
current lock owner, wired memory, swap, and candidate peak requirement.

Expected: candidate peak remains within the previously observed approximately
89.2 GB bound plus safe headroom. Otherwise do not load the full model.

- [ ] **Step 2: Run one guarded route smoke**

Use `bench/laguna/run_guarded.py` so the parent holds
`/tmp/mtplx-gpu-exclusive.lock` before any load/unload. Run one short candidate
request sufficient to compile D3 and require:

- route partition has D3 cycles and exactly three rows per D3 cycle;
- the compiled fixed-M4 route has zero fallback/demotion;
- full FRSpec, QSA gather, fused QSA KV gather, compiled MTP prepare, and M4
  stage-3 configuration match the unchanged candidate;
- service is restored and healthy before guard completion.

- [ ] **Step 3: Run exact guarded three-seed candidate**

Run the reviewed launcher for 16,384 prompt tokens, 1,024 generated tokens,
temperature 1, top-k 20, top-p 0.95, seeds 20260829/30/31. Record individual
decode seconds/TPS, aggregate TPS, `draft_time_s`, route partitions, verifier
statistics, hit/miss, drift, RNG state, and peak memory.

Expected: a complete artifact with no guard, route, service, or memory failure.

- [ ] **Step 4: Compare against the corrected control and previous D3 candidate**

Controls:

- corrected exact mean: 15.368160792 seconds, 66.63126537 TPS;
- previous float32 D3 mean: 15.338961292 seconds, 66.758105748 TPS;
- previous exact draft mean: 1.796658398 seconds;
- previous float32 D3 draft mean: 1.782673473 seconds.

Promotion requires a repeatable reduction in D3 `draft_time_s`; end-to-end TPS
is reported but target-stage time is not attributed to D3.

- [ ] **Step 5: Profile only if the D3 result is still materially below its bound**

Capture one guarded seed with `MLX_DISPATCH_CENSUS`. Classify gaps between D1,
D2, D3 selector command buffers and the packed terminal copy. Iterate only on a
directly attributed D3 boundary, rerunning the focused test and individual A/B
after each change. Stop when no D3-local host-late interval larger than 0.1
milliseconds per cycle remains or the next change would replace an existing
QSA/FRSpec model optimization.

- [ ] **Step 6: Restore and verify production service**

Expected: exact prior service command restored, `/v1/models` healthy, fan mode
default, owned GPU child absent, lock released, and swap not materially worse.

