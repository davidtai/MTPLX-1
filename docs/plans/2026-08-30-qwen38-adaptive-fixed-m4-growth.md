# Qwen3.8 Adaptive Fixed-M4 Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task.

**Goal:** Reduce capacity-transition overhead on the 16K-prefill/1K-output TPS cell while preserving variable-output correctness through 32K.

**Architecture:** Keep physical verification width M=4 and all acceptance arithmetic unchanged. Change only the construction-owned fixed-M4 capacity policy: large requests initially reserve 1,024 tokens; after an overrun, subsequent grants double through 2K, 4K, 8K, and a capped 16K. The existing host-owned boundary transition grows every QSA leaf, rebinds the capacity-owned gather, and reinstalls the compiled graph.

**Tech Stack:** Python, MLX, pytest, guarded Apple Metal benchmarks.

**Assumptions:** Assumes the observed transition/retrace work contributes measurable wall time — the change will not meet the TPS goal if stochastic acceptance and per-window kernel cost dominate. Assumes QSA backing shapes may change only at committed host boundaries — this does not support device-owned mid-window growth.

---

## File Structure

- `mtplx/graphbank.py`: owns initial reserve resolution and fixed-M4 capacity transitions.
- `tests/test_graphbank_compiled_verify.py`: pins pure grant progression, defaults, request tightening, and operator overrides.
- `tests/test_qwen4_exp_capture_commit.py`: proves the installed QSA route grows and remains compiled.
- `docs/specs/2026-08-30-qwen38-adaptive-fixed-m4-growth-design.md`: records the approved policy and measured outcome.

### Task 1: Pin the adaptive policy with failing tests

**Files:**
- Modify: `tests/test_graphbank_compiled_verify.py`
- Test: `tests/test_graphbank_compiled_verify.py`

**Security flag:** none

**Does NOT cover:** Generic compiled-verifier demotion, non-Qwen4 families, or physical verifier widths other than M=4.

- [ ] **Step 1: Write failing policy tests**

```python
@pytest.mark.parametrize(
    ("current", "expected"),
    [(1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 16384)],
)
def test_fixed_m4_growth_grant_doubles_to_8k_cap(current, expected):
    from mtplx.graphbank import _next_fixed_m4_growth_tokens

    assert _next_fixed_m4_growth_tokens(current) == expected


def test_fixed_m4_capacity_growth_clamps_to_reachable_request_end():
    from mtplx.graphbank import _fixed_m4_capacity_growth

    assert _fixed_m4_capacity_growth(
        capacity=20_000,
        required_end=20_004,
        growth_tokens=8_192,
        capacity_limit=24_000,
    ) == (24_000, 8_192)
```

Update `test_fixed_m4_strict_lane_uses_bounded_generation_headroom` to expect
`growth_reserve_tokens == 1024`, and add a 97-token request assertion expecting
101 tokens so small budgets still tighten the default.

- [ ] **Step 2: Run the focused tests under the GPU guard and verify RED**

Run:

```bash
/opt/homebrew/bin/python3 /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --timeout-seconds 900 --child-timeout-seconds 300 -- \
  /bin/zsh -lc 'cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-flash-next-210-restack && .venv/bin/python -m pytest -q tests/test_graphbank_compiled_verify.py -k "growth_grant or capacity_growth_clamps or bounded_generation_headroom"'
```

Expected: failures because the helpers do not exist and the default remains 512.

### Task 2: Implement the construction-owned growth ladder

**Files:**
- Modify: `mtplx/graphbank.py`
- Test: `tests/test_graphbank_compiled_verify.py`
- Test: `tests/test_qwen4_exp_capture_commit.py`

**Security flag:** none

**Does NOT cover:** Response-length prediction, per-request history, full-budget reservation, or eager fallback.

- [ ] **Step 1: Implement the grant helpers and 1K default**

```python
_FIXED_M4_MAX_GROWTH_TOKENS = 16384


def _next_fixed_m4_growth_tokens(current: int) -> int:
    current = max(1, int(current))
    return min(current * 2, max(current, _FIXED_M4_MAX_GROWTH_TOKENS))


def _fixed_m4_capacity_growth(
    *, capacity: int, required_end: int, growth_tokens: int,
    capacity_limit: int | None,
) -> tuple[int, int]:
    next_capacity = max(int(required_end), int(capacity) + int(growth_tokens))
    if capacity_limit is not None:
        next_capacity = max(int(required_end), min(next_capacity, int(capacity_limit)))
    return next_capacity, _next_fixed_m4_growth_tokens(growth_tokens)
```

Add a fixed-M4-only initial-reserve resolver that defaults to 1024 while
leaving `_compiled_verify_growth_reserve()` at 512 for generic banks. An
explicit `MTPLX_COMPILED_VERIFY_GROWTH_RESERVE` remains authoritative.

- [ ] **Step 2: Install and advance the fixed-M4 grant**

Store the reachable capacity limit as
`base_offset + request_max_tokens + speculative_headroom` when the request
budget is known. Initialize `dispatch["growth_tokens"]` to the next grant after
the first-promotion reserve. In `_transition_fixed_m4_generation`, call
`_fixed_m4_capacity_growth`, grow every QSA entry to the returned capacity, and
store the returned next grant only after successful common-capacity growth.

- [ ] **Step 3: Extend the installed-route test**

In `test_fixed_m4_capacity_grows_without_leaving_the_installed_lane`, preserve
the explicit four-token diagnostic reserve and assert the dispatch grant
advances from 8 to 16 after its capacity transition. Keep the existing zero
fallback, compact-demotion, and common-capacity assertions.

- [ ] **Step 4: Run focused tests under the guard and verify GREEN**

Run the Task 1 command plus:

```bash
.venv/bin/python -m pytest -q \
  tests/test_graphbank_compiled_verify.py \
  tests/test_qwen4_exp_capture_commit.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mtplx/graphbank.py tests/test_graphbank_compiled_verify.py \
  tests/test_qwen4_exp_capture_commit.py
git commit -m "perf(qwen4): expand fixed verifier capacity adaptively"
```

### Task 3: Measure the exact TPS cell

**Files:**
- Modify: `docs/specs/2026-08-30-qwen38-adaptive-fixed-m4-growth-design.md`

**Security flag:** none

**Does NOT cover:** Promotion based on a vanity prompt, a single favorable seed, prefill TPS, or a fixed-output claim about production serving.

- [ ] **Step 1: Run matched guarded controls**

Use the unchanged pre-change commit and exact model revision for three seeds on
the 16,384-prefill/1,024-output, temperature-1, xhigh cell. Record wall time,
decode TPS, token digest, acceptance/work counts, transitions, traces, and peak
memory.

- [ ] **Step 2: Run the candidate with the same command and seeds**

Expected engagement: zero capacity transitions for the 1K output, one compiled
shape, zero fallback/demotion/repair, and identical token/work receipts to its
matched control.

- [ ] **Step 3: Apply the promotion gate**

Retain only if lowest/mean wall time improves repeatably without digest, work,
or memory regression. If mean decode remains below 80 TPS, keep the goal open
and profile the retained or unchanged winner's per-window costs before designing
the next isolated candidate.

- [ ] **Step 4: Record and commit the result**

```bash
git add docs/specs/2026-08-30-qwen38-adaptive-fixed-m4-growth-design.md
git commit -m "docs(qwen4): record adaptive growth benchmark"
```

### Task 4: Regression verification

**Files:**
- Test: `tests/`

**Security flag:** none

**Does NOT cover:** Claiming the >80 TPS goal unless the exact guarded benchmark proves it.

- [ ] **Step 1: Run the focused verifier/QSA/server suite under the guard**

```bash
.venv/bin/python -m pytest -q \
  tests/test_graphbank_compiled_verify.py \
  tests/test_qwen4_exp_capture_commit.py \
  tests/test_qwen4_qsa_m4_fused_kv_gather.py \
  tests/test_qsa_prefill_gather_tier.py \
  tests/test_server_openai.py
```

Expected: all pass.

- [ ] **Step 2: Preserve long-output evidence**

If the candidate is retained, rerun at least the forced 2K boundary proof. The
existing final-branch 16K and 32K receipts remain valid only if the production
growth code is unchanged; otherwise rerun the affected long-output gate before
PR delivery.
