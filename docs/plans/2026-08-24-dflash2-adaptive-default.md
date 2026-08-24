# DFlash2 Adaptive Drafting Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Qwen3.8 DFlash2 adaptive M=1–8 drafting enabled at every context length by default, with an explicit process-level fixed-M8 opt-out.

**Architecture:** Store the immutable selection on `DFlash2RuntimeConfig`, pass it through runtime loading and every CLI/server launch boundary, and have the serialized generation route reapply that selection without consulting prompt length. The benchmark harness requests adaptive mode explicitly and records requested/effective state and observed widths.

**Tech Stack:** Python 3.12, argparse BooleanOptionalAction, MLX, dflash-mlx, pytest.

**Assumptions:** Assumes DFlash2 generation remains serialized by `_DFLASH2_GENERATION_LOCK` — this does not support per-request mutation. Assumes fixed fallback remains physical M=8 — it does not select another fixed width.

---

## File structure

- `mtplx/backends/dflash2.py`: runtime configuration and effective policy.
- `mtplx/runtime.py`: direct-load propagation.
- `mtplx/cli.py`, `mtplx/commands/public.py`, `mtplx/server/openai.py`: public flag and child/server propagation.
- `scripts/qwen38_final_benchmark_arm.py`, `scripts/qwen38_final_benchmark_matrix.py`: corrected, attested benchmark.
- `tests/test_dflash2_backend.py`, `tests/test_dflash2_server.py`: production tests.
- `tests/test_qwen38_final_benchmark_arm.py`, `tests/test_qwen38_final_benchmark_matrix.py`: benchmark tests.

### Task 1: Runtime policy defaults to adaptive at every length

**Files:** Modify `mtplx/backends/dflash2.py`, `mtplx/runtime.py`; test `tests/test_dflash2_backend.py`.

**Security flag:** none

**Does NOT cover:** CLI parsing or child propagation; Task 2 owns those.

- [ ] **Step 1: Write failing tests.** Assert `DFlash2RuntimeConfig.from_paths(...)` defaults `draft_adaptive=True`; route application at 1K, 16K, 64K, and 128K passes `active=True`; an explicit `draft_adaptive=False` passes `active=False` at all lengths. Assert receipts contain `requested_adaptive`, `effective_adaptive`, and `fixed_block_size`.
- [ ] **Step 2: Verify RED.** Run `.venv/bin/python -m pytest -q tests/test_dflash2_backend.py`. Expect missing-field and old-cutoff failures.
- [ ] **Step 3: Implement.** Add `draft_adaptive: bool = True` to `DFlash2RuntimeConfig`, `draft_adaptive` to `load_dflash2_bundle`, and `dflash2_draft_adaptive: bool | None = None` to `mtplx.runtime.load`. Apply `bool(runtime.config.draft_adaptive)` instead of a prompt-length decision and preserve short/long route IDs only for other phase telemetry.
- [ ] **Step 4: Verify GREEN.** Repeat the Task 1 test command; expect pass.
- [ ] **Step 5: Commit.** `git add mtplx/backends/dflash2.py mtplx/runtime.py tests/test_dflash2_backend.py && git commit -m "feat: default DFlash2 to adaptive drafting"`

### Task 2: Expose and propagate the Boolean CLI flag

**Files:** Modify `mtplx/cli.py`, `mtplx/commands/public.py`, `mtplx/server/openai.py`; test `tests/test_dflash2_server.py`.

**Security flag:** none

**Does NOT cover:** request-level toggles; policy remains process-level.

- [ ] **Step 1: Write failing tests.** For `run`, `ask`, `serve`, and `start`, assert default and `--dflash2-adaptive` are true while `--no-dflash2-adaptive` is false. Assert the generated server command forwards exactly one spelling and runtime loading receives the same Boolean.
- [ ] **Step 2: Verify RED.** Run `.venv/bin/python -m pytest -q tests/test_dflash2_server.py`. Expect parser rejection or missing namespace values.
- [ ] **Step 3: Implement.** Add `--dflash2-adaptive` with `argparse.BooleanOptionalAction`, `default=True`, to the shared DFlash helper and hidden server parser. Forward the resolved spelling to child servers and pass `dflash2_draft_adaptive=bool(args.dflash2_adaptive)` during runtime load.
- [ ] **Step 4: Verify GREEN.** Repeat the Task 2 test command; expect pass.
- [ ] **Step 5: Commit.** `git add mtplx/cli.py mtplx/commands/public.py mtplx/server/openai.py tests/test_dflash2_server.py && git commit -m "feat: add DFlash2 adaptive drafting flag"`

### Task 3: Correct and attest the final benchmark contract

**Files:** Modify `scripts/qwen38_final_benchmark_arm.py`, `scripts/qwen38_final_benchmark_matrix.py`; test `tests/test_qwen38_final_benchmark_arm.py`, `tests/test_qwen38_final_benchmark_matrix.py`.

**Security flag:** none

**Does NOT cover:** changing coding loads from exact 1,024 generated tokens; only the palindrome headline uses natural EOS under a 1,024-token cap.

- [ ] **Step 1: Complete failing tests.** Require `stop_token_ids_for_prompt("is_palindrome") is None`, `stop_token_ids_for_prompt("coding") == set()`, every PR child command to include `--dflash2-adaptive`, and receipts to contain requested/effective adaptive state plus observed widths.
- [ ] **Step 2: Verify RED.** Run `.venv/bin/python -m pytest -q tests/test_qwen38_final_benchmark_arm.py tests/test_qwen38_final_benchmark_matrix.py`. Expect missing helper, conditioner-mode, and receipt-field failures.
- [ ] **Step 3: Implement.** Keep the palindrome prompt simple. Use natural EOS for it, greedy target/draft sampling, same-prompt warm decode, fresh generation state, and `max_tokens=1024`. Preserve exact coding output. Construct candidate runtime with `draft_adaptive=True` and fail if the effective receipt is not adaptive.
- [ ] **Step 4: Verify GREEN.** Repeat the Task 3 test command; expect pass.
- [ ] **Step 5: Commit.** `git add scripts/qwen38_final_benchmark_arm.py scripts/qwen38_final_benchmark_matrix.py tests/test_qwen38_final_benchmark_arm.py tests/test_qwen38_final_benchmark_matrix.py && git commit -m "bench: attest adaptive final matrix"`

### Task 4: Focused verification and real-model matrix

**Files:** Create `benchmarks/results/qwen38-final-cold-prefill-matrix-20260824/main-vs-pr-cold-prefill-matrix.json`; update PR #335 body inline.

**Security flag:** none

**Does NOT cover:** another PR or unrelated services.

- [ ] **Step 1: Run focused verification.** Execute `.venv/bin/python -m pytest -q tests/test_dflash2_backend.py tests/test_dflash2_server.py tests/test_dflash2_runtime.py tests/test_qwen38_dflash_adaptive.py tests/test_qwen38_final_benchmark_arm.py tests/test_qwen38_final_benchmark_matrix.py`, compile both benchmark scripts, and run `git diff --check`. Expect all clean.
- [ ] **Step 2: Run real matrix.** With `com.tea.qwen` stopped and `/tmp/mtplx-gpu-exclusive.lock` available, run `.venv/bin/python scripts/qwen38_final_benchmark_matrix.py --output benchmarks/results/qwen38-final-cold-prefill-matrix-20260824/main-vs-pr-cold-prefill-matrix.json`. Expect five four-arm ABBA scenarios, exact cold prefill sizes, no prefix/session cache, and adaptive effective in every PR scenario.
- [ ] **Step 3: Validate and embed.** Validate with `jq`; embed main/PR prefill TPS, decode TPS, wall, wall delta, peak GiB, actual output tokens, adaptive mode, and observed M directly in PR #335.
- [ ] **Step 4: Finish.** Run fresh verification, commit the receipt/table changes, push the existing branch, verify remote head and inline PR body, and restore `com.tea.qwen` to its pre-benchmark state.
