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

### What that is worth, against a MEASURED baseline

The 2026-09-02 guarded run timed the production attention chain end to end at
M=4 / 16K on the queued lane:

| arm | ms/layer | ms/cycle (12 layers) | GB/s of 544 |
| --- | ---: | ---: | ---: |
| `production_gather_kernel` — fused gather + score + mask + softmax + cast + P@V | **0.230** | **2.764** | 309 |
| `portable_take_reference` — same math and bytes, gathering via two `mx.take` | 0.542 | 6.504 | 131 |

**`production_gather_kernel` is the baseline; the 6.5 ms figure is not.** The
portable arm exists only as the numerics reference for parity: it moves the
same bytes but through MLX's generic gather instead of the one-dispatch fused
kernel, and is 2.35x slower for that reason alone. Quoting a speedup against it
would inflate the result by more than the result.

Cross-check: `micro_qsa_m4.py` attributes 1.501 ms/cycle of that 2.764 to the
fused gather plus the transposed-K copy — **54 % of the production attention
chain is the materialisation this kernel deletes outright.**

Scaling the measured baseline by the byte ratio (217 vs 852 MB/cycle):

| if the kernel achieves | its ms/cycle | saving vs 2.764 | share of a 39.7 ms cycle |
| --- | ---: | ---: | ---: |
| 309 GB/s (the baseline's own rate) | 0.70 | 2.06 ms | 5.2 % |
| 200 GB/s | 1.09 | 1.68 ms | 4.2 % |
| 150 GB/s | 1.45 | 1.32 ms | 3.3 % |

So roughly **+2.2 to +3.6 tok/s at 68 tok/s**, and the 54 % that is pure
materialisation is a firm floor on the opportunity.

The risk is the other direction. At M=4 with 6 splits the grid is 48
threadgroups of 64 threads on 40 cores — better than 8, still thin. If the
kernel turns out latency-bound rather than bandwidth-bound, the deleted bytes
do not convert. That is what the split sweep in
`scripts/fable/micro_qsa_sparse_decode.py` is for, and it is why nothing here
may be reported as a win before the 16K ABBA: W16 turned an isolated -1.9 ms
into 0 end-to-end, and W19's lightning lane lost to dense at 16K after looking
good in isolation.

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
  probe runs four synthetic cells (M=4 and M=1, one long-context and one just
  past the crossover so invisible ids really appear) against
  `qsa_sparse_decode.stock_reference`, a transcription of the shipped
  rows-gather lane. Gates: max abs diff <= 8 bf16 ulp at the reference's own
  magnitude, relative L2 <= 2e-3, head-dim top-1 agreement >= 0.98. These are a
  sanity gate, not the quality gate.

Engagement line: `mtplx.kernels.qsa_sparse_decode.engagement()` reports
`verify_kernel` / `draft_kernel` call counts, the install verdict and the probe
statistics. If an ABBA reports a win and those counters are zero, the win came
from somewhere else.
