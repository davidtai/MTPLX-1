# QSA decode attention: split-K direct-index sparse GQA (K-Q2, K-D6)

`MTPLX_FABLE_QSA_SPARSE_DECODE` (M=4 verify) and `MTPLX_FABLE_QSA_SPARSE_DRAFT`
(M=1). Both default off. Both rounding-class, so both HumanEval-gated.

## 0. The premise this item started with was wrong, and the correction matters

The task was framed as *"the QSA family moves 446 MB per decode cycle at
207-232 GB/s, about a third of the bandwidth the LM head (613) and the routed
experts (~700) achieve"* — i.e. a bandwidth-efficiency problem with ~3x of
headroom.

That number is not a measurement. `scripts/fable/census_retained_stack.py`
records per-**command-buffer** GPU intervals, fits four global coefficients
(`cb_floor`, `per_dispatch`, `weight GB/s`, `activation GB/s`) by NNLS over
every buffer, then splits each buffer's measured duration across its ops *in
proportion to modelled cost* (`census_retained_stack.py:566-660`). The
`achieved_GBs` column is `family_bytes / family_modelled_ns`, so it is monotone
in the bytes the classifier assumes and says nothing about efficiency. For QSA
specifically the classifier prices the fused K/V gather at a flat 4.19 MB and
gives the score, softmax and P@V dispatches **zero** bytes, while MLX's
transposed-K copy lands in the Copy family. So both "232 GB/s" and the
counter-claim "actually ~540 GB/s" that a corrected byte model produces are
artifacts of the byte model, not findings.

### What the census JSONL can and cannot measure directly

Grouping `op` records by `command_buffer_index` and keeping each buffer's
measured `gpu_end_ns - gpu_start_ns` (1,853,269 ops / 110,617 buffers / 383
lm_head cycles in `w58-retained-control-census-1788370322.jsonl`):

* **No command buffer isolates the QSA attention chain.** The dominant shape is
  4,202 buffers of exactly 50 ops (10.97 per cycle) plus 382 of 51 ops (1.00
  per cycle) — 12 per cycle, one per QSA layer. Median duration **397.3 µs**
  (p10 372.2, p90 425.2).
* Each of those buffers is *the QSA attention tail plus the next MoE block's
  routing head*: ops 0-23 are the token build → fused gather → score gemv →
  mask → softmax → cast → P@V → gate, and ops 24-49 are the MoE router GEMV,
  softmax, top-10, and one `paired_routed_glu` dispatch. The QSA q/k/v/o
  projections and the indexer score GEMM are in *other* buffers.
* The JSONL has no per-op timing, so the QSA attention chain **cannot** be
  measured directly from this artifact. The 397.3 µs is real but includes a
  routed-expert GLU dispatch that dominates it.

(Incidental but load-bearing for anyone re-reading that buffer: ops 0-3 are the
*stock* four-dispatch token build, not `qsa_m4_row_tokens`. This census ran
with `MTPLX_QSA_M4_FUSED_KV_GATHER` on but `MTPLX_FABLE_QSA_M4` off.)

### The measurement that does exist, and does decide

`scripts/fable/micro_qsa_m4.py`, run 2026-09-01 in a guarded window on the
compiled lane over one verify cycle (12 QSA layers), measured the shipped
`gather_stock` arm — the fused K/V gather **plus** the
`k_sel.swapaxes(-1,-2).reshape(...)` MLX materialises for the score operand —
at **1.501 ms per verify cycle**, 125 µs per layer. That is a direct GPU
number, not a modelled split.

Its bytes are concrete tensor sizes, not classifier weights:

| | per layer | per cycle |
| --- | ---: | ---: |
| gather reads the selected K and V rows | 16.8 MB | 202 MB |
| gather writes `k_sel`,`v_sel` `[1,2,4,2052,256]` bf16 | 16.8 MB | 202 MB |
| MLX copies the transposed K view (read + write) | 16.8 MB | 202 MB |
| **total** | **50.4 MB** | **605 MB** |

605 MB in 1.501 ms is **403 GB/s**, 74 % of this machine's measured 544 GB/s
(`TEST_MACHINES.md`). So that arm is *reasonably* close to the byte floor **for
the bytes it moves** — and every one of those bytes except the first 202 MB is
avoidable. The split-K kernel reads the same K and V rows, in place, inside the
attention that consumes them, and writes no gathered tensor and no transposed
copy at all.

**Verdict: continue.** The path is not near the byte floor *of the operation*;
it is ~4x above it. The lever is bytes, not bandwidth.

## 1. Per-layer anatomy at M=4 (from the census buffer above)

Six dispatches, ~71 MB, per QSA layer per verify cycle. Byte figures are tensor
sizes: rows = 4, selected width 2,052, 2 KV heads, head_dim 256, bf16, so
`k_sel` and `v_sel` are 8.40 MB each.

| # | census kernel (grid) | what | bytes |
| --- | --- | --- | ---: |
| 10 | `custom_kernel_mtplx_qwen4_qsa_m4_fused_kv_gather_c17408` `[1050624,1,1]` | gathers K and V rows into `[1,2,4,2052,256]` | 33.6 MB |
| 13 | `gg2_copybfloat16bfloat16` `[192,96,1]` | MLX's contiguous copy of `k_sel.swapaxes(-1,-2)` | 16.8 MB |
| 14 | `gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc1_axpby0` `[129,1,96]` | scores, 96 batches (2 KV x 12 GQA x 4 rows), K=256, N=2052 | 9.2 MB |
| 15 | `...OSelect...` `[8208,24,1]` | `where(token_ok, scores, -inf)` | 1.6 MB |
| 16 | `block_softmax_float32` `[52224,1,1]` | 96 rows x 2052, 544 threads/row | 1.6 MB |
| 17 | `vn_copyfloat32bfloat16` `[49248,1,1]` | `probs.astype(bf16)` | 1.2 MB |
| 18 | `gemv_t_bfloat16_bm1_bn2_sm8_sn4_tm4_tn4_nc1_axpby0` `[8,1,96]` | P@V, 8 x 32 = 256 outputs | 8.8 MB |
| | | **total** | **~71 MB** |

Two structural facts follow, and they answer "why is this path slow" better
than any GB/s number:

1. **It materialises what it is about to read.** 33.6 MB of gather traffic and
   16.8 MB of transposed copy exist only so that two GEMMs can re-read 16.8 MB
   of the same values. The operation's actual working set is the 16.8 MB of
   cache rows.
2. **Occupancy is NOT the problem in the shipped path.** The gather dispatches
   1,050,624 threads and the score gemv 12,384 threadgroups. The shipped lane is
   well parallelised; it is doing four times the memory work.

## 2. Per-layer anatomy at M=1

There is none on the retained stack. The census's once-per-cycle counts are
exactly 36 GDN `out_proj` and 48 MoE `shared down_proj` dispatches, i.e. **the
full stack runs once per verify cycle, at M=4**; the draft chain runs the MTP
block, not the twelve QSA layers. The kernels the census classifier files under
QSA at ~3 dispatches/cycle (`gemv...[160,1,1]`, `[320,1,1]`, `[80,1,1]`,
`block_softmax_precise_bfloat16 [128,1,1]`) are one-per-draft-depth ops of the
MTP block and the MoE router, not QSA attention. (Relatedly:
`block_softmax_precise_bfloat16 [512,1,1]` at 48.4/cycle is the **MoE router**
softmax over 48 layers, misfiled into QSA by the classifier's `block_softmax`
catch-all — 51 of the QSA family's 194.5 dispatches/cycle are not QSA.)

So `MTPLX_FABLE_QSA_SPARSE_DRAFT` exists for the non-speculative decode path
and for a future full-stack draft. It cannot move the 16K speculative ABBA and
must not be credited with doing so.

## 3. Why split-K, and what the phase-1 kernel could not do

W50's port parallelises over query rows: grid `(qL, kv_heads, 1)` threadgroups
of 64 threads. At M=4 that is **8 threadgroups** on a 40-core M5 Max; at M=1,
**2**. Each still walks all 2,051 selected keys x 256 dims. W50's own note
(`qsa-sparse-gqa-phase2-wiring.md` section 4) predicted the loss and priced the
split-K variant as a separate item.

It is also the shape MTPLX already knows it needs: a hand-written
`mx.fast.metal_kernel` SDPA lost to stock at long N because MLX's production
SDPA switches to a KV-split two-pass path there. A single-pass sparse kernel
would repeat that with a shorter but equally serial walk.

The variant: grid `(qL, kv_heads, n_splits)`. Each threadgroup owns one
(query row, KV head, contiguous range of BK-tiles), keeps its own fp32 online
softmax `(m, l, O)` in registers, and writes that state to
`[n_splits, 24, M, 258]` fp32 — the row is `[O(256) | m | l]`. A second pass,
one threadgroup per (head, row) and one thread per head dim, rescales by
`exp2(m_s - max m)` and normalises. Both dispatches live in one MLX command
encoder, so there is no host sync between them.

### Bytes, per layer, at M=4

| | shipped | split-K (BK=128, splits target 8 -> 6 splits) |
| --- | ---: | ---: |
| K/V reads | 16.8 MB (gather) + 16.8 MB (GEMM re-reads) | 16.8 MB |
| gathered K/V writes | 16.8 MB | — |
| transposed K copy | 16.8 MB | — |
| score / mask / softmax / cast round trips | 3.6 MB | — |
| partial states (write + read) | — | 1.2 MB |
| Q, ids, output | 0.15 MB | 0.11 MB |
| **total** | **~71 MB** | **~18 MB** |
| dispatches | 6 | 2 |

**3.9x fewer bytes, 637 MB/cycle removed, 48 fewer dispatches/cycle.**

### MEASURED, 2026-09-02 (second guarded run, M=4 / 16K / queued lane)

| arm | ms/layer | ms/cycle (12 layers) | GB/s of 544 |
| --- | ---: | ---: | ---: |
| `portable_take_reference` — parity reference only, NOT a baseline | 0.531 | 6.375 | 134 |
| **`production_gather_kernel`** — the baseline the ABBA replaces | **0.226** | **2.712** | 315 |
| **`native_bk128_dc32_s17`** — the split-K kernel | **0.094–0.099** | **1.13–1.19** | 204–216 |

**2.4x the production attention chain, −1.58 ms/cycle**, or 4.0 % of a 39.7 ms
cycle: about +2.7 tok/s at 68. That lands inside the 1.3–2.1 ms range predicted
from the byte model before anything ran.

#### It is occupancy, exactly as feared, and the split is what fixed it

| threadgroups | BK:DC | splits | ms/layer | x baseline | ms x tgs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | 128:32 | 4 | 0.325 | 0.70 | 10.4 |
| 48 | 128:32 | 8 | 0.210 | 1.08 | 10.1 |
| 72 | 128:32 | 16 | 0.149 | 1.52 | 10.7 |
| 136 | 128:32 | 17 / 32 | 0.099 / 0.094 | 2.28 / 2.41 | 13.5 / 12.8 |
| 136 | 64:64 | 17 / 32 | 0.098 / 0.090 | 2.30 / 2.51 | 13.3 / 12.3 |

`ms x threadgroups` is flat to 72 and rises only ~25 % at 136 — near-perfect
inverse scaling in the grid size. So the kernel is **occupancy-bound, not
bandwidth-bound**: at 136 threadgroups it achieves 216 GB/s of 544 and still
wins 2.4x, purely because it moves 0.28x the bytes. Below 4 threadgroups per
core, cores sit idle; W50's single-pass grid would have been 8 threadgroups and
is off the bottom of this table.

`bk128_dc64` is the outlier at 136 tgs (0.123 ms). Its threadgroup memory is
`max(BK*LDV, DC*LDK)*2 + Qs + selected` = 18.4 + 2.3 + 0.5 = **21 KB**, against
~12 KB for `128:32` and ~11.8 KB for `64:64`, so fewer threadgroups stay
resident per core. Same grid, worse residency.

#### The noise floor, measured for free

At BK=128 there are 17 tiles over the 2,051 selected keys, so split targets 17
and 32 **clamp to the same 17-split, 136-threadgroup grid** — identical work,
measured twice. They differ by 5.3 %. BK=64 s17/s32 likewise (8.2 %); BK=256
s16/s17/s32 likewise (3.4 %).

**The bench's noise floor is 3–8 %.** Averaging the duplicate pairs gives
`64:64` at 0.0941 and `128:32` at 0.0966 — a 2.6 % gap, well inside it. They
are the same performance class and no arm may be called a winner on a margin
under ~8 %.

#### Untested and worth one more sweep point

BK=64 has 33 tiles, so 17 and 32 both clamp to 17 splits; **33 is the first
value that reaches 33 splits = 264 threadgroups**, and the first sweep never
covered it. Given the inverse scaling above it is the obvious next point, and
it is now in `SPLIT_TARGETS`. It does not block the ABBA.

### Parity: the delta is the REFERENCE's, and the gate now says so

Every one of the twenty configurations reported the same parity to four
significant figures — max abs `1.953e-3` (= `2**-9` exactly), rel L2
`4.78e-3`, top-1 `1.0000` — across BK 64/128/256, DC 32/64 and splits 4..32.
DC changes the fp32 score contraction order and the split count changes the
online-softmax merge tree. A delta that does not move across either is not the
kernel's.

The shipped path carries two bf16 roundings this kernel does not:

1. **the scores.** `mx.matmul(q_view, k_view)` has bf16 operands, so its output
   is bf16 — the shipped path rounds the scores *before* `.astype(float32) *
   scale` and the softmax (the census shows this as `gemv_bfloat16...` feeding
   `block_softmax_float32`). A relative score error `u = 2**-9` shifts a logit
   by `u*|x|`, and with scaled logits of order 5 that is ~`2e-2` relative on
   the probabilities.
2. **the probabilities.** `probs.astype(bfloat16)` before P@V (census op 17,
   `vn_copyfloat32bfloat16`): another `u = 2e-3`.

So (1) should dominate (2) by roughly an order of magnitude, and the measured
`4.78e-3` sits between the two predictions. That is a prediction, so it is
**tested rather than asserted**: the micro now runs a three-rung reference
ladder — `shipped` (both casts), `shipped_fp32_probs` (probability cast
removed), `fp32` (both removed) — and reports the gaps.

The gate follows from that, and it is two gates because a threshold on
kernel-vs-shipped is really a threshold on how much bf16 rounding the *shipped*
path does, which no improvement to this kernel can move:

| vs | max abs | rel L2 | top-1 | why |
| --- | ---: | ---: | ---: | --- |
| `fp32_reference` (DECIDES) | 2 bf16 ulp | 5e-4 | 0.98 | the only differences left are fp32 reassociation (`sqrt(2051)*2**-24` = 2.7e-6 relative) and one bf16 store, so both sides round the same real number and agree except where it straddles a boundary |
| `stock_reference` (SANITY) | — | 5e-2 | 0.98 | bounds the shipped path's own quantisation, derived at ~2e-2 above; an order of magnitude over the measured 4.78e-3 |

The tight bar is deliberately **below** the measured 4.78e-3: if the
attribution is wrong and the kernel really does carry that delta, the fp32
comparison fails and says so. A gate set above it would certify nothing.

Neither is a quality gate. This kernel is rounding class; whether the
difference matters is answered by model-level greedy-token agreement plus a
full HumanEval run.

### M=1 confirmed dead, by crash

The M=1 cell died at `k_sel.swapaxes(-1,-2).reshape(...)`:
`Cannot reshape array of size 4202496 into shape (1,2,1,1,256,2052)`. 4,202,496
is `2*4*2052*256` — the shipped fused K/V gather compiles `_ROWS = 4`
(`kernels/qwen4_qsa_m4_fused_kv_gather.py:82`) and emits four rows whatever it
is handed. There is no M=1 production attention to baseline against, which is
the same conclusion section 2 reached from the census. The arm now refuses
`rows != 4` up front and the M=1 cell is opt-in (`--include-m1`).

### The ABBA candidate

```
MTPLX_FABLE_QSA_SPARSE_DECODE=1
MTPLX_FABLE_QSA_SPARSE_DECODE_TILE=128:32
MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS=17
```

`128:32` over the statistically tied `64:64` because it is the tile the prefill
lane already defaults to, and because at BK=128 `splits=17` is exactly one tile
per threadgroup — the smallest value that reaches the full 136-threadgroup grid
and the point past which the knob stops doing anything. Both are now the
built-in defaults, so the two `_TILE`/`_SPLITS` lines are belt-and-braces and
`MTPLX_FABLE_QSA_SPARSE_DECODE=1` alone selects the same configuration.

`MTPLX_FABLE_QSA_SPARSE_DRAFT` stays off: no target.

## 3a. The nanobind ABI defect (found 2026-09-02, inherited from W50)

The first guarded run built the extension cleanly, timed both baseline arms,
and then died at the first call of the new kernel:

```
TypeError: qsa_sparse_gqa_decode(): incompatible function arguments.
  1. qsa_sparse_gqa_decode(queries: mlx::core::array, ...)
  Invoked with types: mlx.core.array, mlx.core.array, ...
```

The initial reading — that nanobind rejected the explicit `stream=None` — is
wrong, and checking it mattered: the same failure reproduces with `stream`
omitted, and on **W50's** `qsa_sparse_gqa_unsupported_reason`, which is a pure
host predicate taking no stream at all. W50's kernel had never been callable
either; its phase-1 micro never reached a call.

The tell is in the signature: nanobind printed the raw C++ name
`mlx::core::array` where `mlx.core`'s own bindings print `array` (compare
`mx.take`'s error, which reads `a: array` and `stream: StreamOrDevice`). That
means the type was never resolved. Two nanobind modules share a type registry
only when they agree on the capsule key `__nb_internals_<abi_tag>_<domain>__`
(nanobind `src/nb_internals.cpp`, `nb_module_exec`). The domain is right on
both sides (`NB_DOMAIN=mlx`). The tag is not:

| | nanobind | `NB_INTERNALS_VERSION` | ABI tag |
| --- | --- | ---: | --- |
| `mlx.core` (wheel 0.32.2) | — | 21 | `v21_system_libcpp_abi1` |
| `_ext` (qwen38 venv) | 2.12.0 | 19 | `v19_system_libcpp_abi1` |

Separate capsules, separate registries, so every function taking an
`mx::array` rejects every call. The build succeeds and the module imports,
which is what makes it expensive: it costs a whole guarded GPU window to find.

**Remedy, no install required.** nanobind 2.15.0 (internals v21) is already on
this box at
`/Users/davidtai/.local/share/uv/tools/mtplx/lib/python3.13/site-packages/nanobind`.
Pass it as `-DMTPLX_NANOBIND_DIR=$NB` and delete the stale `build/` first.

**Do not** define `NB_INTERNALS_VERSION=21` on the v19 sources to force the tag
to match. That macro guards the layout of the shared `nb_internals` struct;
forcing agreement while the struct differs makes two modules write through one
capsule into each other's memory.

Three things now stop this recurring:

* `CMakeLists.txt` reads `NB_INTERNALS_VERSION` from the nanobind it is about
  to use and the ABI tag out of `mlx.core`'s `.so`, and `FATAL_ERROR`s on a
  mismatch with the remedy in the message — a configure-time error instead of
  a runtime one.
* `scripts/fable/check_native_qsa_abi.py` answers the same question with no
  build and no GPU, and can be run before queueing a window.
* `mtplx/native/__init__.py` checks the built artifact against `mlx.core` at
  load and reports the mismatch as the lane's unsupported reason, so the
  install gate raises a diagnosis rather than a bare `TypeError`.

## 4. The correctness difference from the prefill kernel

**Validity is per-slot, not a leading prefix.** This is the one thing in the
port that is not mechanical.

`_select_eager` (prefill) sorts its top-k ascending, so its valid entries
really are a leading prefix and W50's kernel may cut at
`min(512, complete_blocks)`. `QSAIndexer._select_m4` (decode) does **not**
sort: it hands `mx.argpartition`'s raw output to `qsa_m4_row_tokens`, whose
predicate is `block < visible_blocks` evaluated on every slot. Applying a
prefix cut to an unsorted row drops visible blocks and admits invisible ones.
`tests/test_fable_qsa_sparse_decode.py::test_the_prefix_cut_the_prefill_kernel_uses_is_wrong_here`
builds a row where they differ.

Nor is `candidate <= q_abs` a legal substitute, though it is implied by the
real predicate: for `block == complete_blocks` it admits the incomplete block's
tokens, which the tail slots already contribute, and double-counts them in the
softmax denominator. Pinned by
`test_causal_only_predicate_would_double_count_the_tail`.

The kernel walks `topk*ratio + ratio - 1` = 2,051 slots where the shipped lane
builds 2,052. The dropped one is the tail's fourth, token
`((pos+1)//4)*4 + 3`, which exceeds `pos` in every residue class. Pinned by
`test_the_dropped_2052nd_slot_is_always_invalid`.

The query offset arrives as a one-element int32 **device buffer**, because the
fixed-M4 verify carries a tensor-valued cache offset (`TensorOffsetKVCache`)
and reading it on the host would synchronize the graph.

## 5. Numerics and the gate

Identical visible set, different arithmetic: fp32 online softmax in `exp2`
(scale pre-multiplied by `M_LOG2E`) instead of fp32 `exp` over a materialised
score row; fp32 probabilities into an fp32 P@V instead of a bf16 cast and a
bf16 P@V; Steel-MMA reassociation of the 256-term score contraction; one
split-K rescale per row. Rounding class, and not fixable — adopt on greedy
token agreement plus a full HumanEval run, exactly like `MTPLX_FABLE_HC_M4`.

The gate is asymmetric, per `mtplx/kernels/qwen4_m4_route.py`:

* **Contract failure RAISES.** Decided in `TensorOffsetQSACache.__init__` — cache
  install, model build time, outside any `mx.compile` trace.
* **Parity failure DISABLES** for the process and records the deltas. The install
  probe runs synthetic cells (one long-context and one just past the crossover
  so invisible ids really appear) against BOTH references — the tight bar
  against `fp32_reference` decides, the loose one against `stock_reference` is
  a sanity bound. Thresholds and their derivations are in the "Parity" section
  above and in the module's own gate note. These are sanity gates, not the
  quality gate.

Engagement line: `mtplx.kernels.qsa_sparse_decode.engagement()` reports
`verify_kernel` / `draft_kernel` call counts, the install verdict and the probe
statistics. If an ABBA reports a win and those counters are zero, the win came
from somewhere else.
