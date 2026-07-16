# Hy3 Row-Owned Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cross-threadgroup last-arrival router with an M1-M8 row-owned Metal kernel that is correct by construction and contains no device-wide synchronization protocol.

**Architecture:** One Metal threadgroup owns one logical router row and is the only writer of that row's expert IDs and route weights. Twelve SIMD groups cover the 192 experts in 16-expert tiles, loop over the 16 K partitions locally, reproduce the existing balanced FP32 reduction, materialize the 192 precise sigmoid scores in threadgroup memory, and let SIMD group zero perform the exact top-eight selection. The kernel uses only hardware-defined threadgroup barriers; it has no atomics, epochs, readiness flags, winner election, device scratch, checker, or fallback.

**Tech Stack:** Python 3.11, MLX, Metal Shading Language, Apple Metal Performance Primitives tensor operations, pytest, Ruff.

**Assumptions:**

- Assumes G17s permits a 384-thread threadgroup with roughly 15 KiB of threadgroup storage — compilation will fail if the device limit is lower.
- Assumes an MPP M8 descriptor with A/C row extents of one preserves the existing per-row FP32 products — bitwise parity will fail if edge lowering changes arithmetic.
- Assumes extra router-weight reads are tolerable because each row-owned threadgroup rereads the 1.5 MiB resident router matrix — performance will not improve if memory bandwidth dominates more than the removed device synchronization.
- Assumes row ownership is used only for M1-M8 decode and verification calls — M greater than eight remains a separate bulk-matmul workload and is not implemented by this kernel.

---

## File Structure

- Create `mtplx/hy3_router_row_owned.py`: source generator, cached Metal builder, resident qualification, and direct row-owned dispatch.
- Modify `mtplx/models/hy3_mlx.py`: add the explicit row-owned selector and direct M1-M8 model path without epoch state.
- Modify `mtplx/runtime.py`: ensure row-owned configuration reserves zero target and MTP epoch slots.
- Modify `benchmarks/hy3_router_last_arrival_timing.py`: add an explicitly named row-owned candidate arm while retaining #59 as the control and last-arrival as a historical comparison.
- Create `tests/test_hy3_router_row_owned.py`: source-contract, dispatch-shape, exactness, and hardware parity coverage.
- Modify `tests/test_hy3_router_last_arrival_runtime.py`: selector and runtime integration coverage.
- Modify `tests/test_hy3_router_last_arrival_timing.py`: benchmark identity and row-owned invocation coverage.

### Task 1: Build the row-owned Metal source and direct dispatch

**Files:**
- Create: `mtplx/hy3_router_row_owned.py`
- Create: `tests/test_hy3_router_row_owned.py`

**Security flag:** `none`

**Does NOT cover:** Runtime model selection, full-model benchmarks, or M greater than eight.

- [ ] **Step 1: Write failing source-contract and dispatch tests**

```python
@pytest.mark.parametrize("rows", range(1, 9))
def test_row_owned_source_has_static_ownership_and_no_device_protocol(rows):
    source = row_owned.hy3_router_row_owned_source(rows=rows)
    assert f"constexpr int ROWS = {rows};" in source
    assert "constexpr int SIMD_GROUPS = 12;" in source
    assert "uint row = threadgroup_position_in_grid.x;" in source
    assert "threadgroup float partials[P * N];" in source
    assert "threadgroup float unbiased_scores[N];" in source
    assert "threadgroup float selection_scores[N];" in source
    assert "atomic_" not in source
    assert "thread_scope_device" not in source
    assert "epoch" not in source
    assert "ready" not in source
    assert "elected" not in source
    assert "scratch" not in source


@pytest.mark.parametrize("rows", range(1, 9))
def test_row_owned_dispatch_launches_one_threadgroup_per_row(rows, monkeypatch):
    captured = {}
    monkeypatch.setattr(row_owned, "_build_hy3_router_row_owned_kernel", lambda *_: FakeKernel(captured, rows))
    output = row_owned._dispatch_hy3_router_row_owned(value(rows), weight(), bias(), scaling_factor=2.826)
    assert captured["grid"] == (rows * 384, 1, 1)
    assert captured["threadgroup"] == (384, 1, 1)
    assert captured["output_shapes"] == [(rows, 8), (rows, 8)]
    assert len(captured["inputs"]) == 3
    assert output.dispatch_count == 1
```

- [ ] **Step 2: Run tests and confirm the module is absent**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q tests/test_hy3_router_row_owned.py
```

Expected: collection fails because `mtplx.hy3_router_row_owned` does not exist.

- [ ] **Step 3: Implement the source generator and direct dispatch**

```python
_ROWS_MAX = 8
_EXPERTS = 192
_TOP_K = 8
_K_PARTS = 16
_K_SLICE = 256
_N_TILE = 16
_SIMD_GROUPS = 12
_THREADS = _SIMD_GROUPS * 32


def hy3_router_row_owned_source(*, rows: int, scaling_factor: float = 2.826) -> str:
    logical_rows = _validate_rows(rows)
    reduction = _balanced_splitk_reduction_source(_K_PARTS)
    scaling = _router_scaling_literal(scaling_factor)
    return f"""
        using namespace metal;
        using namespace mpp::tensor_ops;
        constexpr int ROWS = {logical_rows};
        constexpr int MPP_ROWS = 8;
        constexpr int N = 192;
        constexpr int P = 16;
        constexpr int KS = 256;
        constexpr int BN = 16;
        constexpr int SIMD_GROUPS = 12;
        uint row = threadgroup_position_in_grid.x;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tid = thread_index_in_threadgroup;
        threadgroup float a_tile[KS];
        threadgroup float partials[P * N];
        threadgroup float unbiased_scores[N];
        threadgroup float selection_scores[N];
        // Each SIMD group owns one 16-expert tile for this row. It loops over
        // all P K slices, then the threadgroup finishes only its own row.
    """


@lru_cache(maxsize=16)
def _build_hy3_router_row_owned_kernel(rows: int, scaling_factor: float):
    return mx.fast.metal_kernel(
        name=f"mtplx_hy3_router_row_owned_m{rows}_g12_precise",
        input_names=["x", "weight", "expert_bias"],
        output_names=["expert_ids", "router_scores"],
        header="#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
        source=hy3_router_row_owned_source(rows=rows, scaling_factor=scaling_factor),
        ensure_row_contiguous=True,
    )


def _dispatch_hy3_router_row_owned(value, weight, expert_bias, *, scaling_factor):
    rows = int(value.shape[1])
    kernel = _build_hy3_router_row_owned_kernel(rows, float(scaling_factor))
    ids, scores = kernel(
        inputs=[value.reshape(rows, 4096), weight, expert_bias],
        grid=(rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, 8), (rows, 8)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return Hy3RouterRowOwnedOutput(ids.reshape(1, rows, 8), scores.reshape(1, rows, 8))
```

The completed Metal body must use the existing `_balanced_splitk_reduction_source(16)` expression, precise `exp`, later-index tie breaking, and the same top-eight output order as #59.

- [ ] **Step 4: Run focused tests**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q tests/test_hy3_router_row_owned.py
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check mtplx/hy3_router_row_owned.py tests/test_hy3_router_row_owned.py
```

Expected: all host tests pass; hardware-marked tests skip.

- [ ] **Step 5: Commit**

```bash
git add mtplx/hy3_router_row_owned.py tests/test_hy3_router_row_owned.py
git commit -m "feat(router): add row-owned fused router kernel"
```

### Task 2: Prove Metal compilation and exact arithmetic on G17s

**Files:**
- Modify: `tests/test_hy3_router_row_owned.py`

**Security flag:** `none`

**Does NOT cover:** Performance or production selection.

- [ ] **Step 1: Add an explicitly enabled hardware parity test**

```python
@pytest.mark.skipif(
    os.environ.get("MTPLX_RUN_ISSUE58_ROW_OWNED_HARDWARE") != "1",
    reason="row-owned parity requires the exclusive Metal window",
)
@pytest.mark.parametrize("rows", range(1, 9))
def test_row_owned_matches_issue59_bitwise(rows):
    mx.random.seed(58_600_000 + rows)
    source_weight = mx.random.normal((192, 4096)).astype(mx.bfloat16)
    resident_weight = prepare_hy3_router_fp32_weight(source_weight)
    expert_bias = (mx.random.normal((192,)) * 0.01).astype(mx.float32)
    hidden = mx.random.normal((1, rows, 4096)).astype(mx.float32)
    expected_ids, expected_weights = hy3_router_fp32_route(
        hidden,
        resident_weight,
        expert_bias,
        available=True,
        n_tile=16,
        grid_k_parts=16,
        operand_mode="grouped-direct",
        simd_groups_per_threadgroup=4,
        finalizer_mode="simd",
        simd_groups=6,
        sigmoid_mode="precise",
    )
    observed = row_owned.hy3_router_row_owned_route(
        hidden, resident_weight, expert_bias, available=True
    )
    mx.eval(expected_ids, expected_weights, observed.expert_ids, observed.route_weights)
    assert bool(mx.array_equal(observed.expert_ids, expected_ids).item())
    assert bool(mx.array_equal(observed.route_weights, expected_weights).item())
```

- [ ] **Step 2: Run the test inside the exclusive MLX window**

Run the test under `exclusive_mlx_window` with:

```bash
MTPLX_RUN_ISSUE58_ROW_OWNED_HARDWARE=1 \
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q \
tests/test_hy3_router_row_owned.py::test_row_owned_matches_issue59_bitwise
```

Expected: eight Metal specializations compile and all eight outputs match bitwise. Qwen must be restored and the lock unheld after the parent context exits.

- [ ] **Step 3: Record any compile-only correction and rerun host tests**

Any Metal correction must preserve all forbidden-token assertions from Task 1. Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q tests/test_hy3_router_row_owned.py
git diff --check
```

Expected: host suite passes and the diff is clean.

- [ ] **Step 4: Commit**

```bash
git add mtplx/hy3_router_row_owned.py tests/test_hy3_router_row_owned.py
git commit -m "fix(router): compile row-owned router on g17s"
```

Skip this commit if no source correction was needed.

### Task 3: Add the explicit runtime selector without epochs or fallback

**Files:**
- Modify: `mtplx/models/hy3_mlx.py`
- Modify: `mtplx/runtime.py`
- Modify: `tests/test_hy3_router_last_arrival_runtime.py`

**Security flag:** `none`

**Does NOT cover:** M greater than eight; those calls retain the separate bulk router path.

- [ ] **Step 1: Write failing selector and epoch-allocation tests**

```python
def test_row_owned_selector_dispatches_directly_without_epoch(monkeypatch):
    router = _router()
    report = router.configure_kernel("mpp-r1-row-owned-fused-r2", available=True)
    calls = []
    monkeypatch.setattr(hy3_mlx, "_dispatch_hy3_router_row_owned", fake_row_owned(calls))
    ids, weights = router(mx.zeros((1, 4, 4096), dtype=mx.bfloat16))
    assert calls == [(1, 4, 4096)]
    assert report["ownership"] == "one-threadgroup-per-row"
    assert report["device_synchronization"] == "none"
    assert tuple(ids.shape) == (1, 4, 8)
    assert tuple(weights.shape) == (1, 4, 8)


def test_row_owned_runtime_reserves_zero_router_epochs(monkeypatch):
    model = configured_model("mpp-r1-row-owned-fused-r2")
    runtime = MTPLXRuntime(model=model, tokenizer=object(), model_path=Path("hy3"), mtp_enabled=True, contract=MTPContract())
    monkeypatch.setattr(runtime, "_new_hy3_router_epoch_block", lambda *_: pytest.fail("row-owned router allocated epochs"))
    runtime.forward_ar(mx.zeros((1, 4), dtype=mx.int32))
```

- [ ] **Step 2: Run tests and confirm the selector is rejected**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q \
tests/test_hy3_router_last_arrival_runtime.py -k row_owned
```

Expected: selector configuration fails because `mpp-r1-row-owned-fused-r2` is unknown.

- [ ] **Step 3: Implement load-time qualification and direct routing**

Add `mpp-r1-row-owned-fused-r2` to `Router.configure_kernel`. Prepare the same K-major BF16 resident once, validate it once, and record:

```python
report.update(
    ownership="one-threadgroup-per-row",
    supported_rows="1-8",
    dispatch_count=1,
    device_synchronization="none",
    precise_sigmoid=True,
)
```

In `Router.__call__`, route eligible calls directly:

```python
if state.selector == "mpp-r1-row-owned-fused-r2":
    rows = math.prod(int(d) for d in x.shape[:-1])
    if 1 <= rows <= 8:
        output = _dispatch_hy3_router_row_owned(
            x.reshape(1, rows, 4096).astype(mx.float32),
            state.prepared_weight,
            self.expert_bias,
            scaling_factor=self.router_scaling_factor,
        )
        shape = (*x.shape[:-1], 8)
        return output.expert_ids.reshape(shape), output.route_weights.reshape(shape)
```

Do not call the public qualification wrapper from this branch. Do not inspect attention phase, allocate an epoch, run #59 as a checker, or add an eligible-shape fallback.

- [ ] **Step 4: Run runtime and replay suites**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q \
tests/test_hy3_router_row_owned.py \
tests/test_hy3_router_last_arrival_runtime.py \
tests/test_hy3_router_kernel_selection.py \
tests/test_graphbank_compiled_verify.py \
tests/test_generation_sustained.py
```

Expected: all host tests pass; hardware tests skip.

- [ ] **Step 5: Commit**

```bash
git add mtplx/models/hy3_mlx.py mtplx/runtime.py tests/test_hy3_router_last_arrival_runtime.py
git commit -m "feat(router): select row-owned router without epochs"
```

### Task 4: Add a readable paired comparison

**Files:**
- Modify: `benchmarks/hy3_router_last_arrival_timing.py`
- Modify: `tests/test_hy3_router_last_arrival_timing.py`

**Security flag:** `none`

**Does NOT cover:** Full-model decode, cache hit rate, MTP accuracy, or reader saturation.

- [ ] **Step 1: Write failing benchmark identity tests**

```python
def test_parser_names_all_router_implementations_explicitly():
    args = module._parser().parse_args(["--output-json", "/tmp/x.json", "--candidate", "row-owned"])
    config = module._config(args)
    assert config["control"]["id"] == "issue59-g6-materialized"
    assert config["candidate"]["id"] == "issue58-row-owned-g12"
    assert config["candidate"]["ownership"] == "one-threadgroup-per-row"
    assert config["candidate"]["device_synchronization"] == "none"
```

- [ ] **Step 2: Run tests and confirm `--candidate` is absent**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q \
tests/test_hy3_router_last_arrival_timing.py -k candidate
```

Expected: argument parsing fails because the row-owned candidate is not registered.

- [ ] **Step 3: Add explicit candidate selection**

Add:

```python
parser.add_argument(
    "--candidate",
    choices=("last-arrival", "row-owned"),
    default="row-owned",
)
```

The row-owned arm must call `hy3_router_row_owned_route` and must not allocate or pass epochs. Render exact IDs and descriptions in JSON:

```python
"control": {
    "id": "issue59-g6-materialized",
    "description": "two-dispatch N16/P16/SG4 grouped-direct R1 plus precise G6 R2",
},
"candidate": {
    "id": "issue58-row-owned-g12",
    "description": "one dispatch; one 12-SIMD-group threadgroup owns each logical row",
    "ownership": "one-threadgroup-per-row",
    "device_synchronization": "none",
},
```

Keep ABBA ordering as a measurement method but call the result a paired comparison, not a runtime gate.

- [ ] **Step 4: Run harness tests and static checks**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q tests/test_hy3_router_last_arrival_timing.py
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check \
benchmarks/hy3_router_last_arrival_timing.py tests/test_hy3_router_last_arrival_timing.py
```

Expected: all harness tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/hy3_router_last_arrival_timing.py tests/test_hy3_router_last_arrival_timing.py
git commit -m "bench(router): compare row-owned router explicitly"
```

### Task 5: Measure, select, and integrate the winner

**Files:**
- Modify: `docs/plans/2026-07-16-hy3-row-owned-router.md`
- External update: GitHub Issues #58 and #51

**Security flag:** `none`

**Does NOT cover:** Promotion if any correctness mismatch occurs or if the row-owned implementation is slower in realistic decode.

- [ ] **Step 1: Run M1-M8 paired router comparisons under one-candidate-at-a-time exclusive ownership**

For each M from one through eight, run:

```bash
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
benchmarks/hy3_router_last_arrival_timing.py \
--candidate row-owned --rows M --warmups 12 --repeats 100 \
--bootstrap-resamples 10000 \
--output-json /tmp/issue58-row-owned-mM-r1.json
```

Expected: every artifact records bitwise exact IDs and route weights, the actual lock receipt, Qwen restoration, and unambiguous control/candidate IDs.

- [ ] **Step 2: Run the realistic 1024/1024 K0-K7 comparison if router timing is competitive**

Use the repository's realistic prompt generator and the Issue #51 benchmark runner with selector `mpp-r1-row-owned-fused-r2`, one candidate per exclusive window. Record ingest tokens/s, prefill time, decode tokens/s, cache hit rate, MTP accuracy, and reader saturation for K0-K7.

Expected: one complete JSON artifact and a rendered table with no mixed selectors.

- [ ] **Step 3: Publish exact results**

Update Issue #58 with a table containing full candidate IDs, per-M router means, paired ratios, confidence intervals, and correctness. Update Issue #51 only after the realistic K0-K7 run, with the requested unified throughput/cache/MTP/reader table.

- [ ] **Step 4: Complete the plan checklist and run final verification**

Run:

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/pytest -q \
tests/test_hy3_router_row_owned.py \
tests/test_hy3_router_last_arrival_runtime.py \
tests/test_hy3_router_last_arrival_timing.py \
tests/test_hy3_router_kernel_selection.py \
tests/test_graphbank_compiled_verify.py \
tests/test_generation_sustained.py
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/ruff check \
mtplx/hy3_router_row_owned.py mtplx/models/hy3_mlx.py mtplx/runtime.py \
benchmarks/hy3_router_last_arrival_timing.py \
tests/test_hy3_router_row_owned.py \
tests/test_hy3_router_last_arrival_runtime.py \
tests/test_hy3_router_last_arrival_timing.py
git diff --check
```

Expected: all host tests pass, hardware-only tests skip outside the exclusive window, Ruff passes, and the worktree is clean after the final commit.

