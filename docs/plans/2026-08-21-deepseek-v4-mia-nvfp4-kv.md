# DeepSeek V4 Mia `stock432` NVFP4 K/V Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:executing-plans`.  Execute serially because the record,
> arithmetic, attention, and real-model gates share one evolving contract.

**Goal:** Replace the exact Mia target and draft affine-int4 K/V lane with native
432-byte NVFP4 records and direct bounded sparse Metal attention.

**Architecture:** `MiaNVFP4Rows` owns the exact byte records.  Target and DSpark
produce raw V latent plus a separate rotated K tail.  A construction-installed
Metal consumer reads selected records directly and performs online-softmax MLA
without whole-cache dequantization or whole-context scores.

**Tech Stack:** Python 3.11, MLX 0.32, `mx.fast.metal_kernel`, pytest, existing
DeepSeek V4/DFlash2 adapters.

**Assumptions:** This plan targets only Mia `stock432`, head width 512, NoPE 448,
RoPE 64, group 16, batch 1, and the pinned K216/K64 artifacts.  It will NOT work
for MLX `mxfp4`, Mia `rope368`, the community UE8M0/360-byte format, other head
geometry, or another model family.

**Design:** `docs/specs/2026-08-21-deepseek-v4-mia-nvfp4-kv-design.md`

---

## File Structure

- Create `mtplx/deepseek_v4_nvfp4_kv.py`: exact record constants, pack/decode
  kernels, oracle decode, and appendable row owner.
- Create `mtplx/kernels/deepseek_v4_nvfp4_mla.py`: direct record-consuming sparse
  online-softmax attention for target and fixed-window DSpark inputs.
- Modify `mtplx/models/deepseek_v4.py`: raw-latent/separate-RoPE target arithmetic,
  NVFP4 cache owner, selected-index route, and construction binding.
- Modify `mtplx/models/deepseek_v4_dspark.py`: three NVFP4 rings and distinct K/V.
- Modify `mtplx/deepseek_v4_dflash2.py`: construction checks and bounded prefill.
- Modify `tests/test_deepseek_v4_affine_kv.py`: replace the superseded affine
  contract with the required `stock432` record/cache contract and rename it.
- Modify `tests/test_deepseek_v4_dspark_model.py` and
  `tests/test_deepseek_v4_dflash2_adapter.py`: update only required owner and
  integration assertions.
- Modify `scripts/deepseek_v4_dspark_k5_bench.py`: report `stock432` rather than
  affine-int4 in exact-model receipts.

### Task 1: Exact `stock432` record owner

**Files:**
- Create: `mtplx/deepseek_v4_nvfp4_kv.py`
- Rename: `tests/test_deepseek_v4_affine_kv.py` to
  `tests/test_deepseek_v4_nvfp4_kv.py`

**Security flag:** `none`

**Does NOT cover:** model attention, DSpark wiring, alternate NVFP4 layouts, or
fused sparse consumption.

- [ ] **Step 1: Write the failing record contract check**

```python
def test_mia_stock432_record_reconstructs_distinct_key_and_value():
    latent = fixed_bf16_rows(shape=(1, 2, 512))
    rope = fixed_bf16_rows(shape=(1, 2, 64))
    rows = MiaNVFP4Rows()
    rows.append(latent, rope)
    key, value = rows.decode()
    assert rows.records.shape == (1, 2, 432)
    assert rows.records.dtype == mx.uint8
    assert mx.array_equal(rows.records[..., 288:304], mx.zeros((1, 2, 16), mx.uint8))
    assert bytes(rows.records[0, 0, 304:432]) == bf16_bytes(rope[0, 0])
    assert_allclose(key[..., :448], value[..., :448])
    assert_allclose(key[..., 448:], rope)
    assert not mx.array_equal(value[..., 448:], rope)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_deepseek_v4_nvfp4_kv.py
```

Expected: import failure for `mtplx.deepseek_v4_nvfp4_kv`.

- [ ] **Step 3: Implement the fixed codec and owner**

```python
NVFP4_HEAD_DIM = 512
NVFP4_NOPE_DIM = 448
NVFP4_ROPE_DIM = 64
NVFP4_GROUP_SIZE = 16
NVFP4_PACKED_BYTES = 256
NVFP4_SCALE_BYTES = 32
NVFP4_ROPE_OFFSET = 304
NVFP4_RECORD_BYTES = 432

class MiaNVFP4Rows:
    records: mx.array | None
    def append(self, latent: mx.array, rope: mx.array) -> None: ...
    def replace(self, start: int, latent: mx.array, rope: mx.array) -> None: ...
    def drop_first(self, count: int) -> None: ...
    def truncate(self, length: int) -> None: ...
    def decode(self, start: int = 0, stop: int | None = None) -> tuple[mx.array, mx.array]: ...
```

The Metal pack kernel calculates one group-16 scale from `amax / 6`, encodes it
as finite E4M3, uses that decoded scale for nearest/saturating E2M1 packing, writes
zero padding, and copies the supplied rotated tail as BF16 bytes.  The decoder
uses the fixed E2M1 table and E4M3 bit decoder already established by
`mtplx.compressed_tensors`.

- [ ] **Step 4: Verify GREEN and owner mutations**

Run the same test file.  Expected: record, replacement, truncation, and state
round-trip checks pass.

- [ ] **Step 5: Commit**

```bash
git add mtplx/deepseek_v4_nvfp4_kv.py tests/test_deepseek_v4_nvfp4_kv.py docs/specs docs/plans
git commit -m "feat: add Mia stock432 NVFP4 cache records"
```

### Task 2: Correct target and DSpark K/V semantics

**Files:**
- Modify: `mtplx/models/deepseek_v4.py`
- Modify: `mtplx/models/deepseek_v4_dspark.py`
- Modify: `mtplx/deepseek_v4_dflash2.py`
- Modify: `tests/test_deepseek_v4_nvfp4_kv.py`
- Modify: `tests/test_deepseek_v4_dspark_model.py`
- Modify: `tests/test_deepseek_v4_dflash2_adapter.py`

**Security flag:** `none`

**Does NOT cover:** direct sparse Metal attention; this task uses record decode as
the required arithmetic bring-up gate and is not considered the finished lane.

- [ ] **Step 1: Write failing construction and arithmetic checks**

The checks require `DeepseekV4NVFP4Cache` for all 43 target layers,
`MiaNVFP4Rows` for all three draft rings, raw latent as V, first-448 plus stored
RoPE as K, and exact target/draft trim/replace behavior.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_deepseek_v4_nvfp4_kv.py \
  tests/test_deepseek_v4_dspark_model.py \
  tests/test_deepseek_v4_dflash2_adapter.py
```

Expected: failures naming affine owners or the old shared rotated K/V result.

- [ ] **Step 3: Install the source-correct owners and arithmetic**

```python
class DeepseekV4NVFP4Cache(DeepseekV4Cache):
    window: MiaNVFP4Rows
    compressed: MiaNVFP4Rows
    def update_window(self, latent, rope): ...
    def update_compressed(self, latent, rope): ...
    def attention_window_records(self): ...
    def attention_compressed_records(self): ...
```

Split attention projection into `(latent, rotated_rope)`, split compressor output
at the normalized pooled latent boundary, reconstruct distinct K/V for the
bring-up path, and remove output inverse RoPE.  Bind `DeepseekV4NVFP4Cache` and
the three NVFP4 draft rings once at exact-artifact construction.  DFlash2 rejects
any non-`stock432` owner before generation.

- [ ] **Step 4: Verify GREEN**

Run the three files above.  Expected: all direct cache/arithmetic/adapter contracts
pass; no affine owner remains on the enabled exact-Mia route.

- [ ] **Step 5: Commit**

```bash
git add mtplx/models/deepseek_v4.py mtplx/models/deepseek_v4_dspark.py \
  mtplx/deepseek_v4_dflash2.py tests/test_deepseek_v4_nvfp4_kv.py \
  tests/test_deepseek_v4_dspark_model.py tests/test_deepseek_v4_dflash2_adapter.py
git commit -m "fix: restore Mia NVFP4 key value arithmetic"
```

### Task 3: Direct bounded sparse Metal consumer

**Files:**
- Create: `mtplx/kernels/deepseek_v4_nvfp4_mla.py`
- Modify: `mtplx/models/deepseek_v4.py`
- Modify: `mtplx/models/deepseek_v4_dspark.py`
- Modify: `tests/test_deepseek_v4_nvfp4_kv.py`

**Security flag:** `none`

**Does NOT cover:** generic attention dimensions, batching beyond one request,
other record layouts, or a runtime fallback.

- [ ] **Step 1: Write the failing direct-consumer comparison**

Use fixed `stock432` records, 64 query heads, a 128-row causal window, selected
compressed indices, learned sinks, and both M1 and M6 query shapes.  Compare the
Metal output with an online-softmax oracle reconstructed from the same records.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest -q tests/test_deepseek_v4_nvfp4_kv.py -k sparse_attention
```

Expected: import failure for `mtplx.kernels.deepseek_v4_nvfp4_mla`.

- [ ] **Step 3: Implement the fixed Metal kernel**

```python
def nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
) -> mx.array: ...
```

The kernel uses one 32-thread SIMD group per `(query, head)`, decodes one
16-value group per lane, initializes online softmax with the head sink and zero
numerator, loops only the 128-row causal window plus selected compressed rows,
and writes BF16 output.  Target and DSpark receive prebound callables at
construction; enabled execution has no fallback or invariant checks.

- [ ] **Step 4: Verify GREEN and the bounded-allocation contract**

Run the sparse-attention check and the target/DSpark files.  Inspect the callable's
interface to confirm it cannot receive a whole-context mask or score tensor.

- [ ] **Step 5: Commit**

```bash
git add mtplx/kernels/deepseek_v4_nvfp4_mla.py mtplx/models/deepseek_v4.py \
  mtplx/models/deepseek_v4_dspark.py tests/test_deepseek_v4_nvfp4_kv.py
git commit -m "perf: consume Mia NVFP4 cache in sparse Metal MLA"
```

### Task 4: Exact-model execution, receipts, and PR correction

**Files:**
- Modify: `scripts/deepseek_v4_dspark_k5_bench.py`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-1024x1024.json`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-16384x1024-cold.json`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-65536x1024-cold.json`

**Security flag:** `none`

**Does NOT cover:** unrelated models, concurrency, sampling, additional context
lengths, or optimization experiments not nominated by these executions.

- [ ] **Step 1: Update receipt identity and run focused non-GPU verification**

Require `kv_cache_format=nvfp4_stock432`, K216 target, K64 draft, pinned model and
source revisions.  Run lint plus only the DeepSeek NVFP4/DSpark/DFlash2 suites.

- [ ] **Step 2: Run one guarded real epoch and committed-token parity gate**

Acquire `/tmp/mtplx-gpu-exclusive.lock` through the existing service-restoration
guard.  Stop at the first record, output, rollback, or token mismatch.

- [ ] **Step 3: Run the guarded Python service prompt**

Serve the exact Mia/Sero target with the packaged K64 draft through DFlash2,
generate roughly 100 tokens, then restore and verify the prior service.

- [ ] **Step 4: Run the requested cold matrix**

Run exact `1024/1024`, `16384/1024`, and `65536/1024`.  Each receipt records
prefill tok/s, decode tok/s, generated count, acceptance, peak/active memory,
output digest, source commit, artifact revisions, and `stock432` identity.

- [ ] **Step 5: Correct and publish PR #312**

Remove superseded affine/wrong-model claims, include only exact-model NVFP4
evidence, push the implementation and receipts, update the PR body, and make it
ready only after every required gate succeeds.

- [ ] **Step 6: Final verification**

Run `git diff --check`, focused ruff, the directly relevant suite, inspect the
three receipts, confirm the branch/remote SHA, and verify the original Qwen
service is healthy after the last guarded run.
