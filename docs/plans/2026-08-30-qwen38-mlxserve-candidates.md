# Qwen3.8 mlx-serve Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the four relevant mlx-serve techniques into independently measurable Qwen3.8 Flash-Next candidates without changing the Step 8 control.

**Architecture:** Every candidate is validated and bound when the Qwen3.8 model is installed. The generation path keeps only shape decisions that vary at runtime and directly invokes the selected callable. Each task ends with a guarded, interleaved three-run 16K/1K production A/B and an isolated commit; only confirmed wins may be stacked or reported on PR #391.

**Tech Stack:** Python 3.13, MLX 0.32.2, `mx.fast.metal_kernel`, pytest, MTPLX guarded benchmark runner.

**Assumptions:**

- The target is exactly `qwen4_exp`, bf16 activations, 48 layers, hidden size 2560, fixed physical M4 verification, and the current Optimized-Speed q4 artifact. The kernels will not work for other geometry.
- The Step 8 benchmark runner and `/tmp/mtplx-gpu-exclusive.lock` remain authoritative. No Metal command runs without the guarded runner.
- The 1 GiB n-gram LRU remains fixed. This plan changes only cold-row dispatch geometry, not cache capacity or table residency.

---

## File structure

- `mtplx/kernels/gdn_verify_m4_fused.py`: exact four-row sequential GDN recurrence and capture outputs.
- `mtplx/kernels/hyper_connection_pending.py`: fused pending HC write plus following HC read.
- `mtplx/models/qwen4_exp.py`: construction-time installation and variable-shape routing for all four candidates.
- `mtplx/profiles.py`, `mtplx/server/openai.py`: register candidate configuration keys before model construction.
- `tests/test_gdn_verify_m4_fused.py`: GDN state, capture-row, and module-route parity.
- `tests/test_hyper_connection_pending.py`: pending-write arithmetic and layer-boundary parity.
- `tests/test_qsa_flash_skip.py`: construction-bound QSA route and exact fixed-M4 output parity.
- `tests/test_ngram_table_memory.py`: worker construction and decode-site scheduling.

### Task 1: Fixed-M4 GDN verification fusion

**Files:**
- Create: `mtplx/kernels/gdn_verify_m4_fused.py`
- Create: `tests/test_gdn_verify_m4_fused.py`
- Modify: `mtplx/models/qwen4_exp.py`
- Modify: `mtplx/profiles.py`
- Modify: `mtplx/server/openai.py`

**Security flag:** `none`

**Does NOT cover:** Single-row decode, masked/ragged batches, widths other than four, non-fp32 recurrent state, or model families other than the installed Qwen3.8 geometry.

- [ ] **Step 1: Add a failing exact-geometry parity test**

```python
def test_m4_kernel_matches_four_sequential_reference_rows(family_gdn):
    inputs, cache = fixed_m4_inputs(family_gdn, seed=20260830)
    expected, expected_cache, expected_capture = run_reference_rows(
        family_gdn, inputs, cache
    )
    actual, actual_cache, actual_capture = run_candidate_m4(
        family_gdn, inputs, cache
    )
    mx.eval(expected, actual, *expected_cache, *actual_cache)
    assert_close_at_stock_boundaries(actual, expected)
    assert_state_equal(actual_cache, expected_cache)
    assert_capture_equal(actual_capture, expected_capture)
```

- [ ] **Step 2: Confirm the test fails because the candidate module is absent**

Run: `.venv/bin/python -m pytest -q tests/test_gdn_verify_m4_fused.py`

Expected: import failure for `mtplx.kernels.gdn_verify_m4_fused`.

- [ ] **Step 3: Implement the exact four-row kernel boundary**

```python
def fused_gdn_verify_m4(
    qkv_rows: mx.array,
    z_rows: mx.array,
    a_rows: mx.array,
    b_rows: mx.array,
    conv_state: mx.array,
    conv_weight: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    delta_state: mx.array,
    norm_weight: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Return y, conv state, delta state, q rows, k rows, and v rows."""
```

The Metal program must process rows 0 through 3 sequentially inside each value-head threadgroup, retain fp32 recurrent state between rows, preserve every bf16 boundary documented in `gdn_step_fused.py`, and surface the exact q/k/v rows required by capture-commit.

- [ ] **Step 4: Install a construction-bound route**

During Qwen3.8 model installation, validate all invariant geometry, dtype, convolution, norm, and cache requirements once. Store either the M4 callable or the stock callable in the layer. At execution, route only on `B == 1`, `S == 4`, mask presence, and capture mode; never reread environment or model metadata.

- [ ] **Step 5: Run focused correctness checks**

Run: `.venv/bin/python -m pytest -q tests/test_gdn_verify_m4_fused.py tests/test_gdn_step_fused.py tests/test_gdn_conv_norm_rows.py tests/test_env_flag_parsing.py tests/test_server_openai.py`

Expected: all pass, including exact state/capture comparison and construction-route pinning.

- [ ] **Step 6: Benchmark and commit only the isolated candidate**

Run the unchanged Step 8 control and candidate in ABBA order, three completed candidate runs and three completed control runs, under `bench/laguna/run_guarded.py`, using 16K prefill, 1024 output, xhigh thinking, temperature 1.0, top-p 0.95, top-k 20, and the fixed prompt.

Commit: `perf(qwen4): fuse fixed-m4 gdn verification`

### Task 2: Deferred hyper-connection writes

**Files:**
- Create: `mtplx/kernels/hyper_connection_pending.py`
- Create: `tests/test_hyper_connection_pending.py`
- Modify: `mtplx/models/qwen4_exp.py`
- Modify: `mtplx/profiles.py`
- Modify: `mtplx/server/openai.py`

**Security flag:** `none`

**Does NOT cover:** Prefill-sized rows, quantized HC weights, a PLE boundary that changes the stream before the next HC read, or any geometry other than hc=4/hidden=2560.

- [ ] **Step 1: Add failing write-then-read parity tests**

```python
@pytest.mark.parametrize("rows", [1, 4])
def test_pending_write_read_matches_materialized_chain(rows, family_hc_pair):
    stream, block_out, inject = pending_inputs(rows, seed=20260830)
    expected_stream = stream + (
        block_out[..., None, :] * inject[..., :, None]
    ).reshape(stream.shape)
    expected = family_hc_pair(expected_stream)
    actual = family_hc_pair.read_pending(stream, block_out, inject)
    mx.eval(expected_stream, *expected, *actual)
    assert_hc_boundary_parity(actual, expected, expected_stream)
```

- [ ] **Step 2: Confirm the test fails on the missing pending interface**

Run: `.venv/bin/python -m pytest -q tests/test_hyper_connection_pending.py`

Expected: `GatedResidual` has no `read_pending` method.

- [ ] **Step 3: Implement the fused pending boundary**

```python
def fused_hyper_read_pending(
    stream: mx.array,
    block_output: mx.array,
    injection: mx.array,
    norm_weight: mx.array,
    packed_read_weights: tuple[mx.array, ...],
) -> tuple[mx.array, mx.array, mx.array]:
    """Return logical written stream, next mixed input, and next injection."""
```

The kernel must reproduce the current multiplication, reshape, addition, grouped RMS normalization, low-rank mix, and injection rounding boundaries. `Qwen4ExpTextModel` carries a pending tuple across HC reads and flushes it at the PLE layer and final mixer boundary.

- [ ] **Step 4: Bind the route at model installation**

Prepack and bind the pending callable only when both adjacent HC modules satisfy the exact invariant. The enabled route executes directly. The stock model route remains an explicit construction-time alternative.

- [ ] **Step 5: Run focused correctness checks**

Run: `.venv/bin/python -m pytest -q tests/test_hyper_connection_pending.py tests/test_hyper_v3.py tests/test_qwen4_mtp.py tests/test_env_flag_parsing.py tests/test_server_openai.py`

Expected: all pass; full-layer output and final HC stream remain within the existing bf16 parity envelope.

- [ ] **Step 6: Benchmark and commit the isolated candidate**

Use the same guarded ABBA 16K/1K production contract as Task 1. Reject a microbenchmark-only win.

Commit: `perf(qwen4): fold hyper writes into following reads`

### Task 3: Construction-bound masked QSA attention

**Files:**
- Modify: `mtplx/models/qwen4_exp.py`
- Modify: `tests/test_qsa_flash_skip.py`
- Modify: `tests/test_env_flag_parsing.py`

**Security flag:** `none`

**Does NOT cover:** Short context, head dimensions other than 256, query widths outside fixed M4, or changes to index selection, block sorting, RoPE, K/V storage, or attention scaling.

- [ ] **Step 1: Add a failing route-pinning test**

```python
def test_qsa_flash_route_is_fixed_at_construction(monkeypatch, family_attention):
    monkeypatch.setenv("MTPLX_QSA_FLASH", "1")
    attention = family_attention()
    installed = attention._qsa_rows_attention
    monkeypatch.setenv("MTPLX_QSA_FLASH", "0")
    assert attention._qsa_rows_attention is installed
```

- [ ] **Step 2: Confirm the test fails because `_qsa_flash_enabled()` is read during execution**

Run: `.venv/bin/python -m pytest -q tests/test_qsa_flash_skip.py`

Expected: the route is not installed as a fixed callable.

- [ ] **Step 3: Install the existing kernel once**

Replace the per-forward environment check with a construction/install function that validates the exact Qwen3.8 QSA geometry and binds `qsa_flash_skip` to `_qsa_rows_attention`. Keep the runtime branch only for query width and context length.

- [ ] **Step 4: Verify exact QSA outputs**

Run: `.venv/bin/python -m pytest -q tests/test_qsa_flash_skip.py tests/test_qsa_gather.py tests/test_qwen4_qsa_m4_fused_kv_gather.py tests/test_env_flag_parsing.py`

Expected: all pass for fixed M4 and long-context block selections.

- [ ] **Step 5: Benchmark and commit the isolated candidate**

Run the guarded ABBA production benchmark at 16K first. If it wins, repeat at 64K and 128K because QSA work scales with context.

Commit: `perf(qwen4): prebind fixed-m4 masked qsa attention`

### Task 4: mlx-serve-style n-gram `pread` scheduling

**Files:**
- Modify: `mtplx/models/qwen4_exp.py`
- Modify: `mtplx/profiles.py`
- Modify: `mtplx/server/openai.py`
- Modify: `tests/test_ngram_table_memory.py`
- Modify: `tests/test_env_flag_parsing.py`

**Security flag:** `none`

**Does NOT cover:** LRU size, table format, residency, row hashing, dequantization, or large-prefill chunk scheduling.

- [ ] **Step 1: Add failing worker and site-scheduling tests**

```python
def test_decode_prefetch_schedules_one_job_per_row_region(monkeypatch, sidecar):
    pool = RecordingPool()
    sidecar._pool = pool
    sidecar._warm(np.arange(16, dtype=np.int64))
    assert len(pool.jobs) == 48


def test_prefetch_worker_count_is_resolved_at_construction(monkeypatch, sidecar_factory):
    monkeypatch.setenv("MTPLX_NGRAM_PREFETCH_WORKERS", "48")
    sidecar = sidecar_factory()
    monkeypatch.setenv("MTPLX_NGRAM_PREFETCH_WORKERS", "16")
    assert sidecar._prefetch_workers == 48
```

- [ ] **Step 2: Confirm both tests fail on the fixed 16-worker, chunk-only implementation**

Run: `.venv/bin/python -m pytest -q tests/test_ngram_table_memory.py`

Expected: no worker-count field and fewer than 48 decode-site jobs.

- [ ] **Step 3: Implement the two scheduling routes**

At `_SidecarGather` construction, parse `MTPLX_NGRAM_PREFETCH_WORKERS`, accept integers 1 through 64, and default to 16. For decode-sized gathers of at most 16 unique rows, submit one job per `(row, weight/scales/biases region)` so 48 workers can issue all reads concurrently. Retain the existing chunked `_warm` route for larger prefill gathers.

- [ ] **Step 4: Verify cache and table behavior**

Run: `.venv/bin/python -m pytest -q tests/test_ngram_table_memory.py tests/test_env_flag_parsing.py tests/test_server_openai.py`

Expected: all pass; the 1 GiB default, clearing receipt, row values, and large-gather route are unchanged.

- [ ] **Step 5: Sweep and commit the best isolated geometry**

Run guarded ABBA production benchmarks at 16, 32, and 48 workers. Clear the 1 GiB hot LRU before every run and use three runs per value. Keep the fastest mean only if output, wall time, memory, and reset receipts remain valid.

Commit: `perf(qwen4): parallelize decode ngram row reads`

After the four tasks, stack only confirmed wins, run the full vanity plus 16K/64K/128K xhigh/low matrix, update the professional PR table and graph, push the commits, restore the service, verify health, and confirm the GPU lock is free.
