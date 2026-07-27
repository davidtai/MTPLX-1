# Laguna Dual-Lane Fixed-M2 Implementation Plan

> **Execution workflow:** Use `superpowers-optimized:subagent-driven-development`
> to execute these tasks in order, with the test and review gates shown below.

**Goal:** Serve one Cline request and one OpenSource Leaderboard request in the
same Laguna B2 autoregressive decode cohort, reserve one logical slot for each
traffic class, and promote a fixed-M2 router projection only if it beats the
unchanged B2 control without changing Laguna's arithmetic contract.

**Approved design:**
[`docs/specs/2026-07-27-laguna-dual-lane-fixed-m2-design.md`](../specs/2026-07-27-laguna-dual-lane-fixed-m2-design.md)

**Primary MTPLX worktree:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf`

**Other scoped repositories and paths:**

- Leaderboard repository:
  `/Users/davidtai/projects/OpenSourceWTF/opensource-leaderboard`
- Clean leaderboard worktree to create:
  `/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-dual-lane-leaderboard`
- Local benchmark repository:
  `/Users/davidtai/projects/OpenSourceWTF/bench`
- Existing user-owned Laguna launch script used as operational evidence:
  `/Users/davidtai/projects/qwen36-server/scripts/start-laguna-s21.sh`
- Clean MTPLX-MoE port worktree to create:
  `/Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-moe-laguna-dual-lane`

**Architecture:** Classify requests once at job construction, enforce one
non-borrowable active slot for `cline` and one for `background`, and retain the
existing `BatchGenerator` AR batching path. Install a Laguna-checkpoint-specific
router route at model construction: M1 calls the characterized one-row kernel,
M2 calls a new expert-owned kernel that loads each BF16 router weight once for
both rows, and larger logical M uses the explicit stock phase route. Run route
overlap collection only in an offline benchmark wrapper; routed-expert
coalescing requires a later design and is not implemented here.

**Technology:** Python 3.11+, pytest, MLX/Metal custom kernels, FastAPI service
scheduler, Node.js test runner, pnpm, launchd, and the existing exclusive Metal
benchmark guard.

**Security review:** Headers are scheduling hints only. They do not authorize
requests, expose prompt text, or change network access. The route-overlap
artifact stores prompt hashes and aggregate routing measurements, never prompt
contents.

## Execution constraints

- Preserve the dirty `opensource-leaderboard` checkout. Create and use the
  clean worktree named above; never deploy from the dirty checkout.
- Preserve all unrelated changes in `/Users/davidtai/projects/qwen36-server`.
  Its Laguna launch script is untracked user state and is input to the
  repository-owned launcher. Never stage or commit inside that repository.
- Commit a portable `scripts/start-laguna-s21.sh` with tests and documentation
  on `perf/laguna-batch-kernels`. After promotion, port the accepted scheduler,
  kernel, installer, and launcher commits to a clean branch based on
  `moe/main`; do not leave `mtplx-moe` with a launcher that refers to code it
  does not contain.
- The portable launcher must contain an explicit note that Blackwellboy's
  operational changes were applied to make Laguna serving work correctly.
  Preserve the attribution in both repository copies.
- Preserve the current Laguna router's reduction order and BF16 output
  rounding for each row. No tolerance-only correctness gate is acceptable.
- Validate checkpoint, layer count, dimensions, dtype, bias, routing
  configuration, kernel compatibility, and self-check results once during
  installation.
- Once the fixed-M2 route is installed, M1 and M2 call fixed entrypoints
  directly. Do not add eligibility checks, environment reads, exception
  fallback, or diagnostic counters to the enabled per-layer path.
- Run Metal benchmarks sequentially under the exclusive guard. No two agents
  may use the hardware lane concurrently.
- Do not proceed to Task 7 unless Task 6 records a passing promotion decision.

## Task 1: Enforce strict Cline/background admission

**Working directory:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf`

**Files:**

- Modify: `mtplx/server/openai.py`
- Modify: `tests/test_server_openai.py`

**Security:** Request labels remain untrusted scheduling metadata. Unknown,
missing, or malformed values must classify as `background`.

**Does not cover:** Kernel behavior, client header configuration, deployment,
or physical row ordering inside `BatchGenerator`.

- [ ] **Step 1: Write failing classification, reservation, lifecycle, and
  snapshot tests**

  Add focused tests with the existing `object.__new__`,
  `types.SimpleNamespace`, and condition-variable fixtures:

  ```python
  def test_ar_batch_traffic_class_recognizes_only_explicit_cline():
      assert _ar_batch_traffic_class(
          {"request_client_hint": "cline"}
      ) == "cline"
      assert _ar_batch_traffic_class(
          {"request_client_hint": " CLINE "}
      ) == "cline"
      assert _ar_batch_traffic_class(
          {"request_client_hint": "opensource-leaderboard"}
      ) == "background"
      assert _ar_batch_traffic_class({}) == "background"
      assert _ar_batch_traffic_class(None) == "background"

  def test_ar_batch_admits_one_job_from_each_class_at_capacity_two(): ...
  def test_ar_batch_does_not_borrow_idle_cline_slot_for_background(): ...
  def test_ar_batch_does_not_borrow_idle_background_slot_for_cline(): ...
  def test_ar_batch_cancelled_pending_job_does_not_occupy_class_slot(): ...
  def test_ar_batch_completion_releases_only_its_class_slot(): ...
  def test_ar_batch_snapshot_reports_aggregate_and_per_class_counts(): ...
  ```

  Construct pending jobs in mixed FIFO order so the test proves selection is
  oldest-within-class rather than the first two entries in the global list.

- [ ] **Step 2: Run the focused tests and confirm RED**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_server_openai.py \
    -k 'ar_batch_traffic_class or ar_batch_admits_one_job_from_each_class or ar_batch_does_not_borrow or ar_batch_cancelled_pending_job or ar_batch_completion_releases or ar_batch_snapshot_reports'
  ```

  Expected failure: `_ar_batch_traffic_class` and per-class ownership do not
  exist, and the current FIFO admission permits two jobs from the same class.

- [ ] **Step 3: Add construction-time classification and immutable job
  ownership**

  In `mtplx/server/openai.py`, add:

  ```python
  TrafficClass = Literal["cline", "background"]

  def _ar_batch_traffic_class(
      request_observability: Mapping[str, Any] | None,
  ) -> TrafficClass:
      hint = str((request_observability or {}).get(
          "request_client_hint", ""
      )).strip().lower()
      return "cline" if hint == "cline" else "background"
  ```

  Add `traffic_class: TrafficClass` to `_BatchedARJob`, compute it once where
  the request's observability object is already available, and never inspect
  headers or observability from the decode path.

- [ ] **Step 4: Replace capacity-only FIFO admission with class-owned slots**

  Add a helper that:

  1. Computes occupied classes from non-finished active jobs.
  2. Removes cancelled pending jobs without consuming a slot.
  3. Scans the pending list in FIFO order.
  4. Selects at most one oldest job for each free class.
  5. Never fills a free slot with a second job from the occupied class.

  Keep `_pending` as the source of FIFO truth; do not create a second
  independently mutable queue. Update `snapshot()` to retain `pending` and
  `active` and add:

  ```python
  "pending_by_class": {"cline": cline_pending, "background": background_pending},
  "active_by_class": {"cline": cline_active, "background": background_active},
  ```

- [ ] **Step 5: Run focused and full scheduler tests**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_server_openai.py \
    -k 'ar_batch_traffic_class or ar_batch_admits_one_job_from_each_class or ar_batch_does_not_borrow or ar_batch_cancelled_pending_job or ar_batch_completion_releases or ar_batch_snapshot_reports'

  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_server_openai.py
  ```

  Expected: all focused tests and the entire server test module pass.

- [ ] **Step 6: Commit only the scheduler change**

  ```bash
  git diff --check
  git add mtplx/server/openai.py tests/test_server_openai.py
  git commit -m "feat: reserve Laguna AR slots by traffic class"
  ```

## Task 2: Label OpenSource Leaderboard requests in an isolated worktree

**Working directory after setup:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-dual-lane-leaderboard`

**Files:**

- Modify: `apps/api/server/scan/qwen.js`
- Modify: `apps/api/test/qwen.test.js`

**Security:** The label affects scheduling only. It must not replace or weaken
the existing authorization header.

**Does not cover:** Cline configuration, production deployment, or any dirty
frontend files in the existing leaderboard checkout.

- [ ] **Step 1: Create a clean worktree without touching the dirty checkout**

  ```bash
  git -C /Users/davidtai/projects/OpenSourceWTF/opensource-leaderboard \
    status --short --branch
  git -C /Users/davidtai/projects/OpenSourceWTF/opensource-leaderboard \
    worktree add \
    -b perf/laguna-dual-lane-client-header \
    /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-dual-lane-leaderboard \
    HEAD
  git -C /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-dual-lane-leaderboard \
    status --short --branch
  ```

  Expected: the new worktree is clean. If the branch or path already exists,
  inspect and reuse it only if its HEAD and status are understood; do not
  delete it.

- [ ] **Step 2: Add a failing header assertion**

  Extend the existing request-options assertion in
  `apps/api/test/qwen.test.js`:

  ```javascript
  assert.equal(
    seenOpts.headers['x-mtplx-client'],
    'opensource-leaderboard',
  );
  ```

- [ ] **Step 3: Run the exact Node test and confirm RED**

  ```bash
  pnpm --filter @osl/api exec node --test test/qwen.test.js
  ```

  Expected failure: `x-mtplx-client` is absent.

- [ ] **Step 4: Add the client label beside existing request headers**

  In `apps/api/server/scan/qwen.js`, preserve content type and optional
  authorization and add:

  ```javascript
  'x-mtplx-client': 'opensource-leaderboard',
  ```

- [ ] **Step 5: Run focused and API tests**

  ```bash
  pnpm --filter @osl/api exec node --test test/qwen.test.js
  pnpm --filter @osl/api test
  ```

- [ ] **Step 6: Commit only the two client files**

  ```bash
  git diff --check
  git add apps/api/server/scan/qwen.js apps/api/test/qwen.test.js
  git diff --cached --name-only
  git commit -m "feat: label leaderboard Laguna requests"
  ```

  Expected staged file list: exactly the two paths above.

## Task 3: Build the expert-owned fixed-M2 router kernel

**Working directory:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf`

**Files:**

- Modify: `mtplx/kernels/laguna_decode.py`
- Modify: `tests/test_laguna_fused.py`
- Modify: `/Users/davidtai/projects/OpenSourceWTF/bench/laguna/laguna_kernel_check.py`

**Security:** No external input reaches Metal source generation. Dimensions are
fixed by the construction-time checkpoint contract.

**Does not cover:** Model installation, serving activation, routed-expert
coalescing, or benchmark promotion.

- [ ] **Step 1: Write source-contract tests before adding the kernel**

  Add a testable source builder and assertions that prove the physical
  ownership contract:

  ```python
  def test_router_gemv_m2_source_owns_one_expert_and_two_rows():
      source = _router_gemv_m2_source(experts=256, dims=3072)
      assert "uint expert = thread_position_in_grid.x;" in source
      assert "uint row" not in source
      assert "float acc_row0" in source
      assert "float acc_row1" in source
      assert source.count("bf16_t w = gate_weight[") == 1
      assert "acc_row0 += float(w) * float(x[k]);" in source
      assert "acc_row1 += float(w) * float(x[3072 + k]);" in source
  ```

  Also add host-entrypoint tests that reject shapes other than exactly
  `[2, 3072]` and `[256, 3072]` before dispatch. These checks belong in the
  public construction/testing entrypoint; the installed hot-path callable in
  Task 4 will bind the validated kernel directly.

- [ ] **Step 2: Run source and shape tests and confirm RED**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_laguna_fused.py \
    -k 'router_gemv_m2_source or router_gemv_m2_shape'
  ```

- [ ] **Step 3: Implement the fixed-M2 source and cached kernel**

  In `mtplx/kernels/laguna_decode.py`:

  - Preserve `_ROUTER_GEMV_SPLIT = 8`,
    `_ROUTER_GEMV_SIMD = 32`, and 256 threads.
  - Launch grid `(256, 1, 1)`, one threadgroup per expert.
  - Load each `bf16_t w` once inside the K traversal and update
    `acc_row0` and `acc_row1`.
  - Apply the existing `simd_sum` reduction independently to both rows.
  - Have the same owner combine the eight partials in ascending order.
  - Round each result to BF16 at the same point as the existing M1 kernel,
    then write widened FP32 logits at `[row, expert]`.
  - Cache the compiled kernel by the fixed geometry.

  Expose:

  ```python
  def router_gemv_logits_m2(
      x: mx.array,
      gate_weight: mx.array,
  ) -> mx.array:
      """Fixed [2, 3072] x [256, 3072] BF16 Laguna router projection."""
  ```

  Add a private direct launcher that accepts the already-proven installed
  arrays and performs no eligibility decision or fallback.

- [ ] **Step 4: Add real-Metal bitwise correctness checks**

  Extend `laguna_kernel_check.py` with deterministic random, all-zero,
  alternating-sign, large-magnitude, and repeated-value inputs. Compare:

  ```python
  control = mx.concatenate([
      router_gemv_logits(x[0:1], gate_weight),
      router_gemv_logits(x[1:2], gate_weight),
  ], axis=0)
  candidate = router_gemv_logits_m2(x, gate_weight)
  mx.eval(control, candidate)
  assert np.array_equal(np.array(candidate), np.array(control))
  ```

  Run the same comparison through sigmoid, correction bias, top-10,
  normalization, and scale; assert exact expert IDs and bitwise-equal route
  weights.

- [ ] **Step 5: Run CPU/source tests and guarded Metal correctness**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_laguna_fused.py \
    -k 'router_gemv_m2_source or router_gemv_m2_shape'

  cd /Users/davidtai/projects/OpenSourceWTF
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
    -- /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    /Users/davidtai/projects/OpenSourceWTF/bench/laguna/laguna_kernel_check.py \
    --check fixed-m2-router
  ```

  Expected: every M2 logits comparison, expert-ID comparison, and route-weight
  comparison is exact. If the guard wrapper's command interface differs,
  inspect `run_guarded.py` and use its documented equivalent without bypassing
  exclusivity.

- [ ] **Step 6: Commit the MTPLX kernel and tests**

  The benchmark repository is separate and already contains user state. Do
  not stage it in the MTPLX commit.

  ```bash
  cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf
  git diff --check
  git add mtplx/kernels/laguna_decode.py tests/test_laguna_fused.py
  git commit -m "feat: add Laguna fixed-M2 router projection"
  ```

## Task 4: Install a validated direct M1/M2 model route

**Working directory:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf`

**Files:**

- Modify: `mtplx/models/laguna_fused.py`
- Modify: `tests/test_laguna_fused.py`
- Modify: `tests/test_laguna_model.py`

**Security:** A requested fixed-M2 configuration with a contract mismatch or
self-check failure aborts installation before serving.

**Does not cover:** Dynamic eligibility, production fallback, per-token
engagement counters, or M greater than two in the enabled fixed-M2 decode
lane.

- [ ] **Step 1: Write failing contract, self-check, route, and failure tests**

  Add tests covering:

  ```python
  def test_fixed_m2_install_rejects_wrong_layer_count(): ...
  def test_fixed_m2_install_rejects_wrong_router_shape(): ...
  def test_fixed_m2_install_rejects_non_bf16_router(): ...
  def test_fixed_m2_install_rejects_router_bias(): ...
  def test_fixed_m2_install_rejects_wrong_topk_or_scaling(): ...
  def test_fixed_m2_install_selfchecks_all_47_distinct_router_weights(): ...
  def test_fixed_m2_route_calls_bound_m1_entrypoint_for_one_row(): ...
  def test_fixed_m2_route_calls_bound_m2_entrypoint_for_two_rows(): ...
  def test_fixed_m2_route_uses_explicit_stock_route_for_larger_prefill_m(): ...
  def test_fixed_m2_route_has_no_exception_fallback(): ...
  def test_fixed_m2_requested_install_failure_is_fatal(): ...
  ```

  Use call-recording fakes to prove M1 and M2 make exactly one direct call and
  do not invoke eligibility helpers. Make the M2 fake raise a sentinel error
  and assert it propagates rather than entering stock code.

- [ ] **Step 2: Run the focused tests and confirm RED**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_laguna_fused.py tests/test_laguna_model.py \
    -k 'fixed_m2'
  ```

- [ ] **Step 3: Add the construction-time contract and dedicated flag**

  Add:

  ```python
  ENV_FIXED_M2_ROUTER = "MTPLX_LAGUNA_FIXED_M2_ROUTER"

  class LagunaFixedM2ConfigError(RuntimeError):
      pass
  ```

  `install_fixed_m2_router(model)` must prove, before replacing any callable:

  - exactly 47 MoE blocks;
  - hidden size 3072, 256 experts, top-k 10;
  - BF16 router weight `[256, 3072]` and no router bias;
  - current Laguna sigmoid, correction-bias, normalization, and scale
    contract;
  - routed Q4 group-128 and shared Q8 group-128 model layout;
  - Metal/custom-kernel availability;
  - all 47 router weights are distinct installed arrays.

  Read the environment only in `install_from_env`. If the flag is requested,
  any mismatch raises `LagunaFixedM2ConfigError` with the exact property;
  there is no skip report.

- [ ] **Step 4: Bind fixed callables and run all-layer installation self-check**

  For every MoE block, construct an immutable pack containing validated arrays
  and bound direct M1/M2 projection-plus-top-k callables. Compare the M2
  candidate against two M1 calls for all 47 router matrices using fixed
  deterministic vectors. Require bitwise logits and route-weight equality and
  exact expert IDs before installing any block.

  Install only after every block passes, so a partial model cannot escape
  construction.

- [ ] **Step 5: Replace the enabled per-layer path with direct logical-M routes**

  The installed call path may inspect only the flattened logical row count:

  ```python
  if logical_m == 1:
      routing = pack.route_m1(x2d)
  elif logical_m == 2:
      routing = pack.route_m2(x2d)
  else:
      return pack.prefill_stock(x)
  return pack.apply_routed_and_shared_experts(x, routing)
  ```

  `route_m1` and `route_m2` are already bound to exact kernels and top-k
  configuration. Do not read environment state, re-check dtype/shape/model
  metadata, call `is_router_gemv_eligible`, catch kernel exceptions, or
  update diagnostic counters here.

- [ ] **Step 6: Run focused tests, all Laguna tests, and server tests**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_laguna_fused.py tests/test_laguna_model.py \
    -k 'fixed_m2'

  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest \
    tests/test_laguna_fused.py \
    tests/test_laguna_model.py \
    tests/test_laguna_compiled_step.py \
    tests/test_server_openai.py
  ```

- [ ] **Step 7: Commit the installed route**

  ```bash
  git diff --check
  git add \
    mtplx/models/laguna_fused.py \
    tests/test_laguna_fused.py \
    tests/test_laguna_model.py
  git commit -m "feat: install direct Laguna M1 and M2 router routes"
  ```

## Task 5: Measure real Cline/leaderboard expert-route overlap offline

**Working directory:**
`/Users/davidtai/projects/OpenSourceWTF/bench`

**Files:**

- Create: `laguna/laguna_route_overlap_probe.py`
- Create: `laguna/test_laguna_route_overlap_probe.py`

**Security:** Store SHA-256 prompt hashes, token counts, layer/step overlap,
unique-expert counts, and histograms. Never store prompt text, completions,
authorization values, or request headers.

**Does not cover:** A routed-expert coalescing kernel. Any such implementation
requires a separate approved design based on this artifact.

- [ ] **Step 1: Write failing pure-function tests**

  Test a pure summary API with synthetic route traces:

  ```python
  def summarize_route_overlap(
      cline_trace: list[list[set[int]]],
      background_trace: list[list[set[int]]],
  ) -> dict[str, object]:
      ...
  ```

  Cover zero overlap, complete overlap, partial overlap, unequal trajectory
  lengths, 47-layer validation, unique-expert count, per-layer histograms, and
  aggregate histograms. Add an artifact test proving prompt text is absent and
  only its SHA-256 digest is emitted.

- [ ] **Step 2: Run tests and confirm RED**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest laguna/test_laguna_route_overlap_probe.py
  ```

- [ ] **Step 3: Implement benchmark-only route capture**

  Load the real checkpoint through `laguna_lane.py`. Wrap each block's
  `switch_mlp` only inside the probe process and record its existing top-10
  `indices` output. Run each B1 prompt independently, then zip the deterministic
  token/layer trajectories offline; do not modify MTPLX production code.

  Required CLI:

  ```text
  --cline-prompt-file PATH
  --leaderboard-prompt-file PATH
  --leaderboard-class normal|large
  --max-tokens N
  --output PATH
  ```

  Reject an output record if it contains either input prompt string.

- [ ] **Step 4: Run unit tests and three deterministic pairs per class**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest laguna/test_laguna_route_overlap_probe.py
  ```

  Under the exclusive Metal guard, run six pair captures: at least three
  Cline/normal-leaderboard pairs and three Cline/large-leaderboard pairs.
  Provide prompt files at execution time from approved local samples. Use the
  same seed, tokenizer, and decode length across paired trajectories.

- [ ] **Step 5: Review the artifact and record the decision boundary**

  Report median and tail overlap counts, per-layer distribution, and
  union-expert counts separately for normal and large background traffic.
  Conclude only one of:

  - overlap is too low to justify a routed-expert design; or
  - overlap is sufficient to justify a separate design and benchmark plan.

  Do not infer Q4 fixed-weight reuse from router reuse, and do not implement a
  routed-expert candidate in this task.

- [ ] **Step 6: Preserve local benchmark ownership**

  `laguna/` is currently user-owned untracked state in the benchmark
  repository. Leave these two files local and unstaged unless the user
  explicitly requests publication. Record their exact paths and `git status`
  visibility in the handoff.

## Task 6: Benchmark the fixed-M2 route against unchanged controls

**Working directory:**
`/Users/davidtai/projects/OpenSourceWTF/bench`

**Files:**

- Create: `laguna/laguna_fixed_m2_bench.py`
- Create: `laguna/laguna_fixed_m2_window.sh`
- Modify: `laguna/laguna_kernel_check.py`
- Create: `laguna/test_laguna_fixed_m2_bench.py`

**Security:** Benchmark artifacts contain model/config hashes, timings, token
digests, thermal state, and dispatch metadata, not prompt text.

**Does not cover:** Live service restart or promotion without all gates.

- [ ] **Step 1: Write failing harness-integrity tests**

  Add pure tests proving:

  - control and candidate use the same model, tokenizer, prompts, context
    length 1024, decode length 96, seed, and sampling configuration;
  - the only candidate delta is
    `MTPLX_LAGUNA_FIXED_M2_ROUTER=1`;
  - B1 and B2 arms are paired in alternating order for four repetitions;
  - every cell records exact token digests and completion counts;
  - the promotion function requires correctness, B2 median cycle improvement
    of at least 1%, at least three improved B2 pairs out of four, and no more
    than 1% B1 median decode-throughput regression.

  ```python
  def decide_promotion(cells: list[BenchCell]) -> PromotionDecision:
      ...
  ```

- [ ] **Step 2: Run harness tests and confirm RED**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest laguna/test_laguna_fixed_m2_bench.py
  ```

- [ ] **Step 3: Implement paired unchanged-control and candidate arms**

  Reuse `laguna_lane.py` for model load, prompt construction, memory guard, and
  result schema. Reuse the reset/install discipline from
  `laguna_ab_bench.py`. Define:

  ```text
  control: current full Laguna stack, fixed-M2 flag absent
  candidate: identical stack, fixed-M2 flag enabled
  shapes: B1 and B2
  context: 1024
  decode: 96
  repetitions: 4 paired
  ```

  Record prefill time, decode time, decode cycles, aggregate tok/s, per-row
  tok/s, exact token digests, peak memory, thermal state, and installed-route
  report.

- [ ] **Step 4: Add the chained 47-layer router microbenchmark**

  Measure existing B2 router projection versus fixed-M2 across 47 distinct
  real router weights, with warmup outside timed cycles and `mx.eval` at the
  same boundary in both arms. The artifact must include:

  - grid/threadgroup geometry;
  - current and candidate cycle distributions;
  - exact logits/IDs/weights result;
  - expected router traversal of 3 MiB versus 1.5 MiB per layer.

- [ ] **Step 5: Run benchmark unit tests and guarded benchmark window**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest laguna/test_laguna_fixed_m2_bench.py

  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
    -- bash /Users/davidtai/projects/OpenSourceWTF/bench/laguna/laguna_fixed_m2_window.sh
  ```

  Inspect the guard's documented argument handling before execution. Use the
  equivalent guarded invocation if needed; never run around an occupied Metal
  lane.

- [ ] **Step 6: Run dispatch census only after the candidate passes timing**

  Use the existing MLX profiler flow to prove the M2 projection launches 256
  expert-owned threadgroups per layer and the intended fixed-M2 kernel is
  present. Keep profiler instrumentation outside timed cells.

- [ ] **Step 7: Record a mechanical promotion decision**

  Promote only when all are true:

  ```text
  exact correctness: PASS
  B2 median decode-cycle improvement: >= 1.00%
  improved paired B2 repetitions: >= 3 of 4
  B1 median decode-throughput regression: <= 1.00%
  dispatch census: 256 expert-owned M2 threadgroups per layer
  ```

  If any item fails, retain the scheduler/header commits, leave the fixed-M2
  flag disabled, record the rejection artifact, and stop before Task 7.

- [ ] **Step 8: Preserve local benchmark ownership**

  Leave benchmark files and artifacts local and unstaged unless the user
  explicitly requests their publication. Report exact artifact paths and
  visibility.

## Task 7: Package and cross-port the supported Laguna launcher

**Gate:** Execute the MTPLX-MoE port only after Task 6 records `PROMOTE`.

**Primary working directory:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf`

**MTPLX-MoE working directory after setup:**
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-moe-laguna-dual-lane`

**Files on both code lines:**

- Create: `scripts/start-laguna-s21.sh`
- Create: `tests/test_start_laguna_s21_script.py`
- Modify: `README.md`

**Security:** The launcher binds to `127.0.0.1` by default, quotes every
operator-provided path/value, never prints authorization material, and refuses
ambiguous duplicate instances. Network exposure remains an explicit operator
override.

**Does not cover:** Live launchd cutover, direct commits to `moe/main`, or a
launcher-only MTPLX-MoE port that lacks the accepted dual-lane implementation.

- [ ] **Step 1: Write failing launcher contract tests on the upstream Laguna
  development branch**

  Add tests that read the script as text and execute non-serving preflight
  modes. Require:

  ```python
  def test_laguna_launcher_has_no_machine_specific_paths(): ...
  def test_laguna_launcher_derives_repository_root_from_script(): ...
  def test_laguna_launcher_defaults_to_loopback_and_exact_checkpoint(): ...
  def test_laguna_launcher_enables_strict_dual_lane_flags(): ...
  def test_laguna_launcher_enables_promoted_fixed_m2_route(): ...
  def test_laguna_launcher_preserves_single_instance_and_memory_guards(): ...
  def test_laguna_launcher_has_blackwellboy_provenance_note(): ...
  def test_laguna_launcher_print_config_does_not_start_server(): ...
  ```

  The provenance assertion must require wording equivalent to:

  ```text
  Blackwellboy's operational changes were applied to make the Laguna serving
  path work correctly.
  ```

- [ ] **Step 2: Run tests and confirm RED**

  ```bash
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_start_laguna_s21_script.py
  ```

  Expected failure: the repository has no supported launcher.

- [ ] **Step 3: Convert the proven local script into a portable product
  entrypoint**

  Preserve the behavior proven by the existing qwen36 launcher:

  - duplicate-listener and duplicate-loading-process refusal;
  - reclaimable-memory floor and bounded wait;
  - import-path verification;
  - startup readiness watchdog;
  - hidden tool-stream guard ceilings;
  - fatal startup behavior for an enabled fixed-M2 installation failure.

  Replace machine-specific constants with documented overrides:

  ```bash
  SCRIPT_DIR=${0:A:h}
  REPO_ROOT=${SCRIPT_DIR:h}
  PYTHON=${MTPLX_PYTHON:-python3}
  MODEL=${MTPLX_LAGUNA_MODEL:-mlx-community/Laguna-S-2.1-oQ4e}
  HOST=${MTPLX_LAGUNA_HOST:-127.0.0.1}
  PORT=${MTPLX_LAGUNA_PORT:-8080}
  MIN_AVAIL_GIB=${MTPLX_LAGUNA_MIN_AVAIL_GIB:-60}
  ```

  Add `--print-config` so tests and operators can inspect the resolved command
  without loading the model. The launched command must include:

  ```text
  --scheduler-mode ar_batch
  --batching-preset latency
  --max-active-requests 2
  --decode-batch-max 2
  --prefill-chunk-tokens 1024
  --batch-wait-ms 0
  ```

  Export `MTPLX_LAGUNA_FIXED_M2_ROUTER=1` only in the post-promotion version
  committed by this task.

- [ ] **Step 4: Document the supported entrypoint and attribution**

  In the Laguna section of `README.md`, show:

  ```bash
  ./scripts/start-laguna-s21.sh
  ```

  Document the Python/model/host/port/memory overrides, the two required client
  headers, strict non-borrowing behavior, and rollback by disabling the
  fixed-M2 export plus restoring serial scheduler arguments.

  State directly that Blackwellboy's changes were applied to make the
  real-server startup and serving path work correctly. Do not attribute the
  fixed-M2 kernel or its benchmark result to those operational changes.

- [ ] **Step 5: Verify and commit the launcher on
  `perf/laguna-batch-kernels`**

  ```bash
  chmod +x scripts/start-laguna-s21.sh
  zsh -n scripts/start-laguna-s21.sh
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest tests/test_start_laguna_s21_script.py
  ./scripts/start-laguna-s21.sh --print-config
  rg -n '/Users/davidtai|qwen36-server|\\.worktrees/laguna-perf' \
    scripts/start-laguna-s21.sh README.md
  ```

  Expected `rg` result: no machine-specific path.

  ```bash
  git diff --check
  git add \
    scripts/start-laguna-s21.sh \
    tests/test_start_laguna_s21_script.py \
    README.md
  git commit -m "feat: ship the supported Laguna dual-lane launcher"
  ```

- [ ] **Step 6: Create a clean branch from `moe/main`**

  ```bash
  git worktree add \
    -b perf/laguna-dual-lane-fixed-m2-moe \
    /Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-moe-laguna-dual-lane \
    moe/main
  git -C /Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-moe-laguna-dual-lane \
    status --short --branch
  ```

  If the path or branch exists, inspect it and reuse only understood state; do
  not delete or reset it.

- [ ] **Step 7: Port the complete accepted stack, not only the launcher**

  Cherry-pick, in dependency order, the Task 1 scheduler commit, Task 3 kernel
  commit, Task 4 installer commit, and Task 7 launcher commit. Resolve branch
  drift by preserving `moe/main`'s arithmetic, ownership, layout, and existing
  features; never take a whole-file checkout from the upstream Laguna branch.

  ```bash
  UPSTREAM_LAGUNA=/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf
  SCHEDULER_COMMIT=$(git -C "$UPSTREAM_LAGUNA" log -1 --format=%H \
    --grep='^feat: reserve Laguna AR slots by traffic class$')
  KERNEL_COMMIT=$(git -C "$UPSTREAM_LAGUNA" log -1 --format=%H \
    --grep='^feat: add Laguna fixed-M2 router projection$')
  INSTALLER_COMMIT=$(git -C "$UPSTREAM_LAGUNA" log -1 --format=%H \
    --grep='^feat: install direct Laguna M1 and M2 router routes$')
  LAUNCHER_COMMIT=$(git -C "$UPSTREAM_LAGUNA" log -1 --format=%H \
    --grep='^feat: ship the supported Laguna dual-lane launcher$')
  test -n "$SCHEDULER_COMMIT"
  test -n "$KERNEL_COMMIT"
  test -n "$INSTALLER_COMMIT"
  test -n "$LAUNCHER_COMMIT"
  git cherry-pick \
    "$SCHEDULER_COMMIT" \
    "$KERNEL_COMMIT" \
    "$INSTALLER_COMMIT" \
    "$LAUNCHER_COMMIT"
  ```

  Record the four resolved commit IDs before cherry-picking. If any conflict
  changes a performance path, rerun the fixed-M2 construction self-check and
  benchmark on the resolved MTPLX-MoE code; do not infer parity from the
  upstream result.

- [ ] **Step 8: Verify the MTPLX-MoE port**

  ```bash
  zsh -n scripts/start-laguna-s21.sh
  python3 -m pytest \
    tests/test_start_laguna_s21_script.py \
    tests/test_server_openai.py \
    tests/test_laguna_fused.py \
    tests/test_laguna_model.py \
    tests/test_laguna_compiled_step.py
  ./scripts/start-laguna-s21.sh --print-config
  git diff --check
  git status --short --branch
  ```

  Require the same Blackwellboy provenance note, launcher behavior, exact
  kernel self-check, scheduler reservation tests, and clean worktree.

- [ ] **Step 9: Record both distributable commits**

  Report:

  - the launcher commit on `perf/laguna-batch-kernels`;
  - the accepted port commit range on
    `perf/laguna-dual-lane-fixed-m2-moe`, based on `moe/main`;
  - exact test output from both branches;
  - whether publication to the corresponding remotes still requires a branch
    push or PR decision.

  Do not describe the feature as available out of the box until both remote
  code lines contain the verified commits.

## Task 8: Roll out and verify dual-lane serving

**Gate:** Execute only after Task 6 records `PROMOTE`.

**Working directories:**

- `/Users/davidtai/projects/qwen36-server`
- `/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-dual-lane-leaderboard`
- `/Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf`

**Files:**

- Sync from the committed canonical launcher into the existing operational
  path without staging:
  `/Users/davidtai/projects/qwen36-server/scripts/start-laguna-s21.sh`
- Deploy the committed leaderboard header from its clean isolated worktree.
- Configure Cline through its Custom OpenAI Headers UI.

**Security:** Bindings, authentication, and service identities remain
unchanged. Verify headers only through normalized request observability; never
log authorization values or prompt bodies.

**Does not cover:** Editing Cline's live VS Code global state directly or
deploying from a dirty checkout.

- [ ] **Step 1: Capture rollback and pre-change service evidence**

  Record:

  ```bash
  ps -axo pid,etime,command | rg 'mtplx.server.openai|Laguna-S-2.1'
  curl -fsS http://127.0.0.1:8880/health
  curl -fsS http://127.0.0.1:8880/v1/models
  tail -n 200 /Users/davidtai/projects/qwen36-server/logs/qwen.log
  ```

  Preserve the exact prior serial command and a copy of the scoped launch
  script diff as the rollback record.

- [ ] **Step 2: Sync the verified repository launcher into the operational
  path**

  Use the `perf/laguna-batch-kernels` committed launcher as the source of
  truth. Preserve only local deployment values through its documented
  environment overrides; do not fork the scheduler/kernel configuration in
  qwen36-server. Verify with:

  ```bash
  git -C /Users/davidtai/projects/qwen36-server diff -- \
    scripts/start-laguna-s21.sh
  git -C /Users/davidtai/projects/qwen36-server status --short
  ```

  Do not stage or commit any file in this repository.

- [ ] **Step 3: Configure the Cline label through the supported UI**

  In Cline's OpenAI-compatible provider settings, set:

  ```text
  X-MTPLX-Client: cline
  ```

  Do not edit VS Code/Cline global state while the extension is running. If
  the setting cannot be made during execution, report it as the sole manual
  rollout blocker and do not claim live dual-lane acceptance.

- [ ] **Step 4: Deploy the leaderboard header from the clean worktree**

  First prove the deploy source is clean and contains the header commit:

  ```bash
  git status --short --branch
  git log -1 --oneline
  pnpm --filter @osl/api exec node --test test/qwen.test.js
  make deploy
  ```

  `make deploy` requires the operator's existing `DROPLET` environment. Do not
  invent or change a production target. Abort if the clean-worktree preflight
  is not green.

- [ ] **Step 5: Restart Laguna and prove construction-time installation**

  ```bash
  launchctl kickstart -k gui/$(id -u)/com.tea.qwen
  curl -fsS http://127.0.0.1:8880/health
  curl -fsS http://127.0.0.1:8880/v1/models
  tail -n 300 /Users/davidtai/projects/qwen36-server/logs/qwen.log
  ```

  Required startup evidence:

  - scheduler is `ar_batch`;
  - active and decode batch limits are both two;
  - `fixed_m2_router` is installed;
  - 47 MoE router weights were validated and self-checked;
  - no construction error or fallback report is present.

- [ ] **Step 6: Verify class reservation under saturation**

  With background backlog present, submit:

  1. two background-labelled requests and prove only one becomes active;
  2. one Cline-labelled request and prove it joins as the second active row;
  3. a second Cline request and prove it remains pending;
  4. an unlabelled request and prove it is background.

  Poll the existing scheduler statistics surface and capture:

  ```json
  {
    "active": 2,
    "active_by_class": {"cline": 1, "background": 1},
    "pending_by_class": {"cline": 1, "background": 2}
  }
  ```

  Exact pending counts may differ with concurrent production completions; the
  invariant is that neither active class count exceeds one.

- [ ] **Step 7: Verify simultaneous B2 decode and live acceptance**

  Send one deterministic Cline-labelled prompt and one deterministic
  leaderboard-labelled prompt concurrently. Capture timestamps, class labels,
  admission time, first-token time, completion time, decode token counts, and
  aggregate decode throughput.

  Acceptance requires:

  - both requests are simultaneously active;
  - the model reports a B2 decode cohort;
  - Cline joins without waiting for the running background decode to finish;
  - aggregate B2 decode throughput exceeds the unchanged B1 aggregate;
  - token digests match the corresponding deterministic control;
  - leaderboard saturation never occupies the Cline slot.

- [ ] **Step 8: Run final verification and preserve rollback**

  ```bash
  cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-perf
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-support/.venv/bin/python \
    -m pytest \
    tests/test_server_openai.py \
    tests/test_laguna_fused.py \
    tests/test_laguna_model.py \
    tests/test_laguna_compiled_step.py
  git status --short --branch

  cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/laguna-dual-lane-leaderboard
  pnpm --filter @osl/api test
  git status --short --branch
  ```

  Keep the previous serial launch command available. If live acceptance fails,
  restore only the scoped launch-script lines and restart the service; do not
  revert unrelated repository state.

## Final handoff evidence

Before claiming completion, report:

- MTPLX commits and exact changed files.
- Portable launcher commit on the upstream Laguna development line and the
  verified MTPLX-MoE port commit range.
- The README and launcher note stating that Blackwellboy's operational changes
  were applied to make Laguna serving work correctly.
- Leaderboard header commit and clean deploy source.
- Uncommitted launch-script diff and why it remains uncommitted.
- Focused and full test command outputs.
- Fixed-M2 exact-correctness result.
- Four paired B1/B2 benchmark cells and mechanical promotion result.
- Dispatch census evidence.
- Route-overlap artifact path and prompt-redaction check.
- Live per-class scheduler snapshot showing one Cline and one background
  active.
- Live Cline and leaderboard timing/throughput comparison.
- Current health/model endpoints and rollback command.
