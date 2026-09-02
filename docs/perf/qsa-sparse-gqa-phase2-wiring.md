# QSA direct-index sparse GQA — phase 2 (lane wiring) design note

Phase 1 (this branch) built the kernel standalone. Nothing in the model calls
it. This note is the plan for the wiring, and it records the phase-1 findings
that constrain that plan.

Program rows: **B3** (prefill), **K-Q2** (M=4 verify), **K-D6** (M=1 draft) in
`scratchpad/M-holistic-tps-program.md`.

## 0. What phase 1 landed

| file | what |
| --- | --- |
| `native_extensions/qsa_sparse_gqa/sparse_gqa/steel_qsa_sparse_gqa.h` | the ported Steel-MMA kernel (oMLX `7467dce8`, Jonathan Spangler, Apache-2.0; K/V staging in turn from mlx-serve's MIT `msv_attn_p256`) |
| `…/sparse_gqa/qsa_sparse_gqa_params.h` | one host/device parameter block, included by both sides |
| `…/sparse_gqa/qsa_sparse_gqa.{h,cpp}` | the MLX `Primitive`, the encoder, and the contract check |
| `…/sparse_gqa/qsa_sparse_gqa.metal` | 16 instantiations: {fp16, bf16} × {uint32, int32} × (BK,DC) ∈ {(128,32), (256,32), (64,64), (128,64)} |
| `mtplx/native/__init__.py` | `qsa_sparse_gqa(...)`, `qsa_sparse_gqa_supported`, `qsa_sparse_gqa_unsupported_reason`, `native_qsa_available` |
| `scripts/fable/micro_qsa_sparse_gqa.py` | parity + visible-set identity + queued-lane timing (guarded; not yet run) |
| `tests/test_qsa_sparse_gqa_native.py` | CPU-only gates for the contract and the harness's scoring logic |

**The Steel headers are reachable from the installed wheel.** mlx 0.32.2 ships
`include/mlx/backend/metal/kernels/steel/attn/{attn,mma,params,loader,transforms}.h`
and `.../steel/attn/kernels/steel_attention.h`, with the full API the kernel
needs (`BaseMMAFrag<float,8,8>`, `MMATile`, `tile_matmad`, `row_reduce`,
`row_bin_op`, `store_safe`, `Limits`). **Nothing had to be vendored.** The one
file oMLX's copy references but does not contain is its own params struct,
which is why this port defines it in a shared header instead.

Deviations from oMLX, all mechanical, all recorded in the kernel header:
symbol renames; the shared params header; an `int32` `IndexT` instantiation so
`_select_eager`'s int32 ids need no `astype` (an 8 MB copy per layer per
4,096-row chunk); and `params->kL` is a caller-supplied **logical** key length
rather than `keys.shape(2)`, because MTPLX hands the kernel the full cache
backing `[1, 2, capacity, 256]` and attends to the first `total_tokens` rows.

### Build

CMake + nanobind, same shape as `verify_mlp`. `python setup.py build_ext
--inplace` is the intended path but the qwen38 venv has no `setuptools` (which
is also why `verify_mlp` has never been built on this box), so phase 1 drove
CMake directly:

```
cd native_extensions/qsa_sparse_gqa
cmake -S . -B build \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/mtplx_native_qsa/ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DPython_EXECUTABLE=<venv>/bin/python
cmake --build build -j 8
```

Metal compiles clean under the MLX metallib flags (`-Wall -Wextra
-fno-fast-math -Wmetal-addr-spaces`) and the emitted AIR contains
`air.simdgroup_matrix_8x8_multiply_accumulate` — real Steel MMA, not a scalar
fallback. Phase 2 should decide whether the artifact is built in CI or checked
in; the current `.gitignore` keeps `*.so/*.dylib/*.metallib` out of the tree,
matching `verify_mlp`.

## 1. The wiring point

`mtplx/models/qwen4_exp.py`, `Attention.__call__`, the `"flash_prefill"` branch
(~3894-3960 on this branch). It already receives exactly what the kernel wants
and already has the fallback ladder:

```
flash_prefill  -> qsa_prefill_flash (NAX)      if supported
               -> _qsa_prefill_gather_attention if MTPLX_QSA_PREFILL_GATHER
               -> _qsa_blocks_to_dense_mask + dense SDPA
```

The native consumer becomes a new rung. **Order matters and is a measurement,
not a preference**: run the harness first. Its verdict decides whether native
goes *above* `qsa_prefill_flash` (replacing it on M5) or *below* it (as the
portable Metal-3 tier that also serves M1-M4 machines, where the NAX kernel
cannot run at all — `qsa_indexer_select_nax_available()` is False there and
today those machines drop to the gather tier).

Selection between rungs should be one env knob with an explicit auto:

```
MTPLX_QSA_NATIVE_SPARSE = 0 | 1 | auto     (default auto)
MTPLX_QSA_NATIVE_SPARSE_TILE = 128:32      (BK:DC, from the harness sweep)
```

`auto` = on where `native_qsa_available()` and the harness's winning tile beat
whatever else the box supports. Same fail-closed discipline as the existing
lane: ask `qsa_sparse_gqa_supported(...)` before routing, never catch an
exception to change the algorithm.

## 2. Feeding the existing top-512 ids

No change to the selector is required, and none should be made.

`_select_eager` returns `("flash_prefill", block_ids, block_valid)` with
`block_ids` `[S, 512]` int32 and `block_valid` `[S, 512]` bool. The native
binding accepts that shape directly (reshape to `[1,1,S,512]` is a view on
contiguous data) and that dtype directly (int32 instantiation). **The kernel
never reads `block_valid`** — it derives validity in-kernel as *"the first
`min(512, (pos+1)//4)` slots of the row are valid"*.

That is identical to the selector's output **by construction**, and the
argument has to be written down because the kernel's correctness rests on it:
`_select_eager` sorts the raw top-k ascending, and validity there is the
threshold predicate `id < complete_blocks`, so the valid entries are exactly a
leading prefix, and their count is exactly `min(512, complete_blocks)` (every
visible block outranks every masked one, whose score is a true `-inf`). The
subsequent `mx.where(block_valid, block_ids, 0)` overwrites only slots the
kernel already skips.

This is an invariant of the **selector**, not of the kernel, so phase 2 must
not let it drift silently:

* the harness asserts it on every row of every cell (A1-A4, plus a full
  token-list comparison on sampled rows) — see its docstring;
* add the same four assertions to the existing selector tests, so a future
  change to `_select_eager`'s ordering or tie-break fails there rather than
  producing quietly wrong attention;
* if a future selector ever emits a non-prefix validity mask, the honest fix
  is to pass `block_valid` into the kernel as a fifth buffer, not to re-sort
  on the host.

`block_ids` could also be produced as uint32 directly
(`mx.sort(top_idx.astype(mx.uint32))`) — the metallib handles both, so this is
a free choice, not a requirement.

## 3. The 2,048 crossover

oMLX goes sparse as soon as `offset + rows > 2048` (`language.py:1171`).
MTPLX's selector gate is `_qsa_prefill_min_context()` = **32,768** and the
attention-consumer gate `_qsa_prefill_flash_min_context()` is also 32,768. The
B3 win is mostly in that 2K-32K span, which MTPLX currently serves densely.

Two gates move, and they are not the same decision:

1. **Selector crossover** (`_qsa_prefill_min_context`, default 32,768 → 2,049).
   Lowering this makes the indexer's score/top-k pipeline run from 2K. That
   pipeline has its own fixed cost, which is why the default is where it is —
   the 32,768 default is a *measured* ABBA result (2026-08-30), so this must be
   re-measured, not assumed.
2. **Consumer crossover** (`_qsa_prefill_flash_min_context`, and the new native
   equivalent). The kernel itself is correct from `total_tokens // 4 > 512`,
   i.e. 2,052 tokens — below that the selector emits fewer than 512 columns and
   the shape check refuses the call. So the floor is structural at 2,052; the
   crossover above it is an economics question.

Both already have env overrides, so the ABBA can sweep them without a code
change. Suggested rungs: 2,049 / 4,096 / 8,192 / 16,384 / 32,768 at 16K and 32K
prompts.

## 4. Decode: K-Q2 (M=4) and K-D6 (M=1) — read this before costing them

The program note treats decode as "0 extra days if B3 lands". Phase 1 says
that is optimistic, for a reason visible without running anything:

**The dispatch grid is `(qL, kv_heads, 1)` threadgroups of 64 threads.** At
prefill (4,096 rows) that is 8,192 threadgroups — fine. At M=4 it is **8
threadgroups, 512 threads**; at M=1 it is **2 threadgroups, 128 threads**. On
an M5 Max that is single-digit-percent occupancy, and each of those few
threadgroups still walks ~2,051 keys × 256 dims serially. The kernel is
parallelised over *query rows*, and decode has almost none.

So:

* the harness's `decode-m4-16k` and `decode-m1-16k` cells are there to
  **measure this**, not to confirm a win. Expect the M=1 cell to lose;
* if it does, K-Q2/K-D6 need a **kernel change**, not wiring: split the ~2,051
  selected keys across several threadgroups per (row, KV head) and combine the
  partial online-softmax states in a second pass (the standard flash-decoding
  split-K), or fold the two KV heads and the 24 query heads differently. That
  is a real 3-5 day item on top of B3, and it should be priced as one;
* the existing decode lanes (`qsa_flash_skip` for M=1, the rows-gather lane for
  M=4) stay the default until a split-K variant beats them.

Separately, decode does not currently produce the block-id contract at all:
`_select_eager` reaches its `flash_prefill` branch only when `S > 1` **and**
`_qsa_large_prefill_enabled(S, total)`, which requires
`current_attention_phase() == "prefill"`. Wiring decode therefore also means
emitting `(block_ids, block_valid)` from the decode/verify selector path —
which is cheap (it is the same `top_idx`, sorted) but is a second change.

## 5. Fail-closed shapes

Follow oMLX and refuse rather than approximate:

* **mRoPE / vision.** `Attention.__call__` already sets `sel_mask = None` when
  `vrope is not None`, so multimodal requests never reach the sparse branch and
  run dense-causal. Nothing to add — but the phase-2 tests should pin it, since
  the failure mode is silent (oMLX's `#3355` exists precisely because a
  gathered-prefill eligibility check drifted across an mRoPE rebind).
* **Batched / B>1.** The primitive refuses `q.shape(0) != 1`. Batched decode
  must not reach it.
* **Non-production geometry.** 24 q heads, 2 KV heads, head_dim 256, block size
  4, K=512 are compiled-in. Anything else raises.
* **Strides.** The kernel reads K/V/Q with 128-bit `uint4` loads, so every
  leading stride must be a whole number of 16-byte words and the last dimension
  must be unit stride. The builder checks the nominal strides; `eval_gpu`
  re-checks the final ones and throws. The base pointers must also be 16-byte
  aligned — true for MLX allocations and for the `[B,S,H,D] → [B,H,S,D]`
  transposed Q view, but a future sliced K/V view could break it, so do not
  hand this kernel an offset slice of a cache.

## 6. Acceptance

1. Run `scripts/fable/micro_qsa_sparse_gqa.py` in one guarded window (command
   in its docstring). Gate: parity within the stated bf16-ULP tolerance on
   every cell, identity assertions clean, and **≥3× faster than
   `qsa_prefill_flash`** at 4,096 × {16K, 32K, 64K} on the queued lane ⇒ GO
   (the program note's own falsifier).
2. Pick the winning `(BK, DC)` from the same run; do not carry all four into
   the lane.
3. Wire the rung behind `MTPLX_QSA_NATIVE_SPARSE`, default off.
4. Crossover sweep, then a 32K ABBA against the shipped lane, then the
   agreement screen against the HC_M4 reference (this is a rounding-class
   change, so greedy-token agreement is the quality gate, not bit-exactness).
5. Only then flip the default, and only on the machine classes the receipts
   cover.

Do **not** report a win from the microbench alone: prior QSA work on this
runtime has had an isolated −1.9 ms turn into 0 end-to-end (W16), and the
lightning lane lost to dense at 16K after looking good in isolation (W19).
