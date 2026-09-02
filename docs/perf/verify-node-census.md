# W69 — node census of the compiled fixed-M4 verify body

Branch `worker/w69-verify-node-census` (from `experiments/fable-qwen38-80tps` @ `7943d845`).
Runtime only, no code path changed, no GPU touched: this is an inventory the next workers are
funded from.

Tool: `scripts/fable/census_verify_nodes.py`. Tests: `tests/test_fable_verify_node_census.py`
(37, pure python on a synthetic census).

```
python scripts/fable/census_verify_nodes.py \
  .benchmark-artifacts/pr391/w58-retained-composed-census-1788370553.jsonl \
  --json /tmp/w69-composed.json --dump-body /tmp/w69-body.tsv
```

One streaming pass, 2.5 s on the 483 MB file, nothing held but a 5,000-dispatch deque.

## 0. What this measures, and what nobody had measured

`census_retained_stack` reduces a whole cycle by **kernel family** (bytes, GPU busy).
`census_verify_opener` (W63) finds the **command buffer** the compiled verify body opens in.
Neither says how many nodes the body has, what they are, or which of them a kernel could delete
— and the body is replayed on the host every single cycle, so every node in it is a recurring
host cost.

**The compiled verify body is 2,751 dispatches per cycle on the current composed stack**
(3,669 on the control). It is the whole 48-layer target forward plus the head, in one
`mx.compile`'d graph: `install_fixed_m4_split`'s prefix/suffix partition exists in `graphbank.py`
but is **not on the retained decode route**, so there is no separate layer-0 prefix to subtract.
The "~5,090-node suffix" the W63 doc carries is an estimate inherited from the pre-OPDIET
2410 census's *whole-cycle* dispatch count; it is superseded here by **2,751 measured dispatch
nodes plus ~1,684 source-counted view nodes ≈ 4,400 tape nodes per cycle** (§5).

## 1. How the body is located — measured, with a falsifier

A compiled graph replays the same nodes in the same order every cycle. So, anchoring on the
once-per-cycle target `lm_head` (grid `[1,31040,1]`, the body's **last** dispatch) and walking
backwards, the body is the **longest common `(kernel name, grid)` suffix across every cycle in
the file**. If a file has no such suffix — or if the suffix fills the whole search window — the
tool raises rather than guessing.

| census | cycles | body, measured | opening dispatch |
| --- | ---: | ---: | --- |
| `w58-retained-control-census-1788370322` | 382 | **3,669** | `gather_frontbfloat16_int32_int_2` |
| `w58-retained-composed-census-1788370553` | 394 | **2,751** | `gather_frontbfloat16_int32_int_2` |

Three independent checks that this is the right object:

1. **3,669 reproduces W63 exactly.** W63 anchored on command-buffer boundaries and found the
   body "opens at a fixed offset of 3,668 dispatches before the cycle's `lm_head`, in 382 of 382
   cycles". Offset 3,668 inclusive of the head is 3,669 dispatches. Different method, same number.
2. **The opening kernel is the same one W63 names** — `gather_frontbfloat16_int32_int_2`.
   Its five-dispatch prologue is the quantised token embedding (three gathers + one
   `affine_dequantize` of 4x2560) and the `mx.tile` to `hc_count`, i.e. exactly
   `_forward_fixed_m4_prefix`'s first two statements. The PLE q4 auxiliary is dequantised in its
   own command buffer *before* the body and is not in it.
3. **3,669 − 2,751 = 918**, against the **926** dispatches W64 attributes to HC_M4 + OPDIET for
   the whole cycle. The 8-dispatch remainder is outside the body.

## 2. The inventory

### 2.1 By block

The body cuts cleanly at the hyper-connection read that opens every block (three custom kernels
on the composed stack, a nine-dispatch eager form on the control). The block's own contents name
it: GDN fused `in_proj` `[1,2060,1]`, QSA `qkv` `[1,1536,1]`, MoE router `[1,64,1]`, the PLE conv,
the `lm_head`. The cut count is checked against that anchor census, so a wrong opener fails loudly.

| block | n | dispatches each | total | share of body |
| --- | ---: | ---: | ---: | ---: |
| **QSA (full attention)** | 12 | 111 (one 119) | **1,340** | **48.7 %** |
| MoE | 47 | 19 | 893 | 32.5 % |
| GDN (linear attention) | 36 | 13 | 468 | 17.0 % |
| MoE + PLE (layer 1) | 1 | 34 | 34 | 1.2 % |
| head | 1 | 11 | 11 | 0.4 % |
| prologue (token embedding + HC tile) | 1 | 5 | 5 | 0.2 % |
| | | | **2,751** | |

**Twelve QSA layers are half the body.** They are 25 % of the model's attention blocks and
48.7 % of its host-replayed nodes. Of a QSA block's 111 dispatches, **ten** are the load-bearing
compute (5 qmv: `qkv`, `o_proj`, indexer q/k, the indexer gate; 3 gemm: the indexer score
`steel_gemm` and the two attention gemvs; 1 softmax; the fused KV gather) and three are the
hyper-connection read. **The other 98 are glue** — rope, mask, cache offsets, index arithmetic
and layout copies.

### 2.2 By op class

| class | n | share | H if the whole class vanished |
| --- | ---: | ---: | ---: |
| copy / layout | 546 | 19.8 % | 0.207 ms |
| elementwise, `mx.compile`-fused | 545 | 19.8 % | 0.207 ms |
| custom kernel | 519 | 18.9 % | 0.197 ms |
| matmul / qmv | 325 | 11.8 % | 0.123 ms |
| elementwise, single op | 169 | 6.1 % | 0.064 ms |
| sort / top-k | 108 | 3.9 % | 0.041 ms |
| norm (`rms`) | 87 | 3.2 % | 0.033 ms |
| gather / scatter | 75 | 2.7 % | 0.028 ms |
| reduce | 73 | 2.7 % | 0.028 ms |
| cache offset (`compute_dynamic_offset`) | 72 | 2.6 % | 0.027 ms |
| cache append / slice (`*_dynamic_copy`) | 72 | 2.6 % | 0.027 ms |
| softmax | 60 | 2.2 % | 0.023 ms |
| index / `arange` | 60 | 2.2 % | 0.023 ms |
| matmul / gemm | 39 | 1.4 % | 0.015 ms |
| dequantize | 1 | 0.0 % | — |

### 2.3 By block × class

| class | QSA | MoE | MoE+PLE | GDN | head | prologue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| copy / layout | **408** | 94 | 4 | 36 | 3 | 1 |
| elementwise (fused) | **336** | 94 | 7 | 108 | 0 | 0 |
| custom kernel | 48 | 282 | 6 | 180 | 3 | 0 |
| matmul / qmv | 60 | 188 | 4 | 72 | 1 | 0 |
| elementwise | 80 | 47 | 2 | 36 | 4 | 0 |
| sort / top-k | 60 | 47 | 1 | 0 | 0 | 0 |
| norm | 48 | 0 | 3 | 36 | 0 | 0 |
| gather / scatter | 24 | 47 | 1 | 0 | 0 | 3 |
| reduce | 24 | 47 | 2 | 0 | 0 | 0 |
| cache offset | **72** | 0 | 0 | 0 | 0 | 0 |
| cache append / slice | **72** | 0 | 0 | 0 | 0 | 0 |
| softmax | 12 | 47 | 1 | 0 | 0 | 0 |
| index / `arange` | **60** | 0 | 0 | 0 | 0 | 0 |
| matmul / gemm | 36 | 0 | 3 | 0 | 0 | 0 |

Every dispatch in the body is host-replayed; the **cache offset**, **cache append/slice** and
**index/arange** classes exist only in QSA blocks, and 74 % of the body's copies are there too.

## 3. Prices, and the two calibrations

From `docs/perf/decode-critical-path.md` §1-2 (production lane, composed stack):

| item | price | what it buys |
| --- | ---: | --- |
| one replayed dispatch, **exposed** host encode | **0.379 µs** | H. 3.22 µs is encoded per dispatch but 88 % of it overlaps a busy GPU and is free. |
| one **dependent** GPU launch removed | **1.83 µs** | G, in full. A sibling launch hides under MLX's concurrent encoder and is worth nothing here. |
| 1 ms/cycle removed | +1.75 tok/s | 2.6806 tok/window at 39.637 ms |

The 0.43 µs/op figure from the earlier price list is **not** used: it measures Python-issued,
*uncompiled* graph construction at a sync (the draft loop), not compiled-body replay.

**Funding bar.** A group must remove ≥ ~150 dependent launches (0.3 ms of G) or ≥ ~800 sibling
nodes (0.3 ms of H). The 72-node (W59) and 168-node (W60) groups already tried were unmeasurable
by construction: 0.027 and 0.064 ms against a 0.08-0.32 ms A/A noise floor.

### 3.1 Calibration A — the route kernel (end to end)

The MoE routing head is exactly eight dispatches per block that survive OPDIET, and
`mtplx/kernels/qwen4_m4_route.py` replaces them with two:

```
384 matched − 96 emitted = 288 removed → predicted 0.636 ms;  measured −0.92 ms/cycle
```

**The method reads 31 % low on the only lever with an end-to-end number.** Every row in §4 is
therefore a floor, not a ceiling. (Against the ten targets the kernel's own docstring counts on
the pre-OPDIET Step-8 census, the prediction is 0.85 ms and the gap is 8 %.)

### 3.2 Calibration B — pre-body GPU idle scales with body node count (new here)

`census_verify_opener` run on both files with the body length this tool measured:

| stack | body dispatches | GPU idle immediately before the body | host-late | cycles |
| --- | ---: | ---: | ---: | ---: |
| control | 3,669 | **1.817 ms/cycle** (median 1.657) | 88.0 % | 382/382 |
| composed | 2,751 | **1.545 ms/cycle** (median 1.350) | 90.9 % | 394/394 |

918 fewer body dispatches → 0.272 ms less idle (medians: 0.307 ms) =
**0.30–0.33 µs per body node**, at the one sync where the body's replay is exposed.

That is an **independent in-file confirmation of the 0.379 µs price**, within 20 %, from a
different clock. It also says the two are the *same* microseconds — the pre-body idle is the
body's own replay/encode — so a removed node must be charged **once**, not twice.

For scale: 2,751 × 0.379 µs = **1.043 ms**, i.e. the body is **62 %** of the composed cycle's
entire 1.676 ms exposed-encode budget.

## 4. Ranked fusable groups

Groups are disjoint by construction — the tool raises if two claim the same dispatch — and
`removable = matched − what the replacement mechanism still emits`. `dep` is the share of the
removed dispatches that sit on a read-after-write chain and therefore also return GPU time.

| # | group | matched | removable | dep | H ms | G ms | **total ms** | tok/s | verdict |
| --: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| — | `moe_router_glue` | 384 | 288 | 288 | 0.109 | 0.527 | *0.636* | *1.11* | **CALIBRATION — already built** |
| 1 | `qsa_rope` | 288 | 240 | 204 | 0.091 | 0.373 | **0.464** | 0.81 | **FUND** |
| 2 | `hc_triple` | 291 | 194 | 194 | 0.074 | 0.355 | **0.429** | 0.75 | **FUND** |
| 3 | `gdn_gate_glue` | 180 | 180 | 180 | 0.068 | 0.329 | **0.398** | 0.70 | **FUND** |
| 4 | `qsa_row_list` | 204 | 198 | 158 | 0.075 | 0.289 | **0.364** | 0.64 | **FUND** |
| 5 | `qsa_indexer_select` | 156 | 144 | 130 | 0.055 | 0.238 | 0.292 | 0.51 | marginal |
| 6 | `qsa_head_layout` | 156 | 132 | 132 | 0.050 | 0.242 | 0.292 | 0.51 | marginal |
| 7 | `qsa_cache_offset` | 180 | 108 | 108 | 0.041 | 0.198 | 0.239 | 0.42 | marginal |
| 8 | `moe_expert_id_copies` | 96 | 96 | 96 | 0.036 | 0.176 | 0.212 | 0.37 | marginal |
| 9 | `moe_shared_gate` | 48 | 48 | 48 | 0.018 | 0.088 | 0.106 | 0.19 | below floor |
| 10 | `attn_residual_add` | 48 | 48 | 48 | 0.018 | 0.088 | 0.106 | 0.19 | below floor |
| 11 | `qsa_indexer_proj` | 48 | 0 | 0 | — | — | — | — | not a node lever (weight bytes) |

**Sum of the four FUND rows: 1.655 ms/cycle, +2.9 tok/s** at the conservative price; at the
route kernel's measured 1.45× conversion, 2.4 ms. They are additive in node count (disjoint) but
their GPU halves are not guaranteed additive — see §6.

### 4.1 The four worth building

**1. `qsa_rope` — 240 nodes, 0.46 ms.** Four rope applications per QSA layer, each a
cast → `sin`/`cos` → two fused multiplies → three-copy `concatenate`, 24 dispatches per layer.
Neighbours: 288 of 288 sit next to `qwen4_qsa_m4_fused_kv_gather`.
*Mechanism:* one rope call per application — `mx.fast.rope`, or a rope epilogue on the fused KV
gather, which already owns the tensors.
*Exactness:* numerically equivalent, **not** bit-exact (a fused kernel keeps f32 where the chain
round-trips through bf16).
*Trap, do not skip:* `mx.fast.rope` wrote **only row 0 at T=1** on mlx 0.31.2. The M4 window is
four rows. Any port needs a length-B offset vector and a four-row A/B before anything else.

**2. `hc_triple` — 194 nodes, 0.43 ms.** `hc_norm → hc_down → hc_up` runs at all 97 blocks. All
three are already custom kernels in `mtplx/kernels/qwen4_m4_hyper_read.py`, and the chain is
strict on a `[4,4,2560]` tensor.
*Mechanism:* one kernel. Fusing three into one removes two of every three — 194, not 291.
*Exactness:* bit-exact **only if** each hand-off's bf16 rounding is reproduced in registers; a
f32-carried fusion is not bit-exact and needs a task eval, not a diff.
*Cheapest of the four to build* — no new tensors, no new indexing, one owner.

**3. `gdn_gate_glue` — 180 nodes, 0.40 ms.** Five dispatches per GDN layer wrapped around two
kernels that already own their inputs: the a/b gate (`gn1_Sigmoid`, fused `exp(-x)`) feeding
`gated_delta_step`, and the output `rms` + fused SiLU gate feeding `out_proj`. Neighbours: 144
next to `gated_delta_step`, 108 next to `gdn_conv_norm_rows`.
*Mechanism:* two epilogues, no new kernel.
*Exactness:* **bit-exact is reachable** — the ops are elementwise on the kernel's own output and
the intermediate is read by nothing else. The only group here where that is true.
*Coordinate with W66* (GDN keep-mask fold) — adjacent kernels, different dispatches.

**4. `qsa_row_list` — 198 nodes, 0.36 ms.** Seventeen dispatches per QSA layer build the
per-row token list the fused KV gather consumes: `arange`, the `tok_blocks` index arithmetic,
`blocks_ok`/`tail_ok`, two `concatenate`s (four bool/int32 copies) and one `where`
(`qwen4_exp.py:2912-2925`, the fixed-capacity rows-gather branch).
*Mechanism:* fold the build into `qsa_m4_fused_kv_gather`'s index math — it already consumes
`(token_idx, token_ok)`, so hand it `top_idx` and `qpos` instead. Separately, the layer-invariant
half (`arange(ratio)`, `tail_tok`, `tail_ok`) hoists to the prologue: twelve rebuilds become one.
*Exactness:* **exact** — identical int32 indices and bool validity. Structurally the safest group
in the table.
*Correction to an obvious-looking claim:* this is **not** a causal-mask hoist. `tok_blocks` and
`blocks_ok` depend on **this layer's own `top_idx`** and are not layer-invariant; only the
tail/causal half is. A worker that assumes otherwise will build the wrong thing.

### 4.2 The marginal four, and why they are not first

* `qsa_indexer_select` (0.29) — **overlaps W68's sparse M=4 attention**; do not fund both without
  coordinating. Tie-break sensitive: a different top-k order changes which rows attend.
* `qsa_head_layout` (0.29 on nodes) — but these are the body's **biggest-byte** copies
  (`vn_copy` at 139 k / 278 k / 49 k elements). Its real value is in GPU bytes, which W68 prices;
  the node count understates it.
* `qsa_cache_offset` (0.24) — six `compute_dynamic_offset` + six dynamic slices + three scalar
  ops per QSA layer. W64 already moved the two *per-cycle* offset reads on device; this is the
  same lever applied per-layer inside the body, and the six slice copies survive it.
* `moe_expert_id_copies` (0.21) — rides on the route kernel (make it emit the layout
  `paired_routed_glu` and `routed_down_reduce` read). Not worth a worker; worth a line of the
  route kernel's next revision.

## 5. Nodes that emit no dispatch

The census records dispatches. MLX's view/metadata primitives — `Reshape`, `Transpose` when it
stays a view, `broadcast_to`, unit-stride `Slice`, `Split`, `expand_dims`, `squeeze`, a no-op
`astype` — are nodes on the replayed tape but produce nothing the census can see, so they are
counted from source.

**They are not part of the ranking above, and the reason is the pricing, not the counting:**
a node with no dispatch has no encode to expose and no launch to serialise. It costs host
pointer-shuffling inside the replay, and that only reaches the cycle where the replay itself is
exposed — the one sync in §3.2. Calibration B already prices *that whole sync* at
0.30-0.33 µs per **dispatch**, a slope fitted with view nodes present in both arms; folding a
separate view-node charge on top would double-count them.

So the correct use of the source count is as a **plausibility bound on the fusions in §4**, not
as an extra saving: a fusion that removes a dispatch usually removes the reshape/transpose glue
around it too, which is part of why the route kernel converted 1.45× better than its dispatch
count predicted.

> Source-side count: see §5.1. Where a "view" is forced to a copy (MLX inserts `Copy` when a
> reshape's result must be contiguous) it is **already** in the census counts above and must not
> be added again — `concatenate` in particular is three `copy` dispatches per rope application
> and is counted in `qsa_rope`.

### 5.1 Per-block view-only node counts

Counted from the decode path the composed flags actually take:
`graphbank._make_fixed_m4_suffix_step` → `qwen4_fixed_verify._forward_fixed_m4_suffix` →
`qwen4_m4_stage3._m4_paired_routed_glu_residual_tail_layer_forward` →
`qwen4_exp.GatedDeltaNet` / `Attention` / `SparseMoeBlock`.

| block | view-only nodes each | × | subtotal | confidence | dominant sites |
| --- | ---: | ---: | ---: | --- | --- |
| QSA (own math) | ~38 | 12 | **456** | med | rope `_rope_cos_sin` / `_apply_partial_rope` = **14 of the 38** (`qwen4_exp.py:292,336,341-356`, called at `:3971-3990`); `_select_m4`'s `pooled_f32_view` transpose+expand+slice (`graphbank.py:965`); `_qsa_rows_gather_attention`'s 3 reshapes + 2 squeezes + 2 expand_dims (`qwen4_exp.py:3706-3725`) |
| MoE (own math) | ~9 | 48 | **432** | med | `rows = x.reshape(ROWS, HIDDEN)`, in-source comment "one metadata-only view" (`qwen4_m4_stage3.py:484`); the `route is None` branch's slice + 3 reshapes (`:488-498`); `mx.split(shared_gu, 2, -1)` (`:534`) |
| GDN (own math) | ~10 | 36 | **360** | med-high | `mx.split` in `_FusedGDNInProj` (`qwen4_exp.py:1176`); 6 reshapes bracketing `fused_gdn_conv_norm_rows` (`:647-654`); `out.reshape(B,S,-1)` before `out_proj` (`:732`) |
| hyper-connection read | 3 (2 at the head) | 97 | **290** | high | `x2 = hyper_input.reshape(-1, hc*hidden)`, `mixed.reshape(...)`, `inject.reshape(...)` (`qwen4_exp.py:901,910,913`). **Zero** inside `fused_hc_read_m4` itself |
| attention-side residual write | 3 | 48 | **144** | high | two `ExpandDims` + one reshape in `_hyper_residual_write` (`qwen4_exp.py:511-513`). The **MoE-side** residual write is fused into `routed_down_residual_tail`'s kernel and contributes **0** |
| prologue | ~2 | 1 | 2 | low | `mx.tile` → expand_dims + broadcast_to, then a forced merge copy already in the census |
| | | | **~1,684** | | |

**So the compiled body is ~2,751 dispatch nodes + ~1,684 view nodes ≈ 4,400 tape nodes per
cycle.** (The W63 doc's "~5,090" was a whole-cycle *dispatch* count from the pre-OPDIET census
reused as a node count; it landed in the right region for the wrong reason.)

Three findings from the source read that change what the ranking means:

* **`_hyper_residual_write` costs 3 view nodes at all 48 attention blocks but 0 at the 48 MoE
  blocks**, because `routed_down_residual_tail` already absorbed the MoE side. That is the exact
  shape `attn_residual_add` (§4, row 10) proposes and evidence that it works.
* **The cache-commit machinery is graph-free.** `TensorOffsetKVCache.update_and_fetch` /
  `TensorOffsetQSACache.write_raw` / `write_pooled` are exactly the six dynamic-slice pairs the
  census already shows; the rollback and `state_leaves` plumbing around them
  (`graphbank.py:941-953`) is Python list and attribute bookkeeping over existing tracer arrays
  and adds **zero** graph nodes. `qsa_cache_offset`'s ceiling is therefore its census count and
  no more.
* **Three "views" are forced copies and are already counted in §2** — do not add them again:
  `k_sel.swapaxes(-1,-2)` (the fused-KV-gather module's own docstring says "that is the copy the
  census sees", `kernels/qwen4_qsa_m4_fused_kv_gather.py:33-42`); `out.transpose(0,2,1,3)
  .reshape(B,S,-1)` before `o_proj` (`qwen4_exp.py:4155` — the axis merge needs
  `stride_H == D·stride_D`, true only at `S==1`, and the verify width is 4); and
  `gate.reshape(B,S,-1)` (`:3945`, a last-axis half-slice of a wider fused tensor). All three sit
  in `qsa_head_layout`.

**How to use the view counts.** Calibration B's 0.30-0.33 µs *per dispatch* was fitted on a change
(HC_M4 + OPDIET) that removed dispatches **and** the view glue around them, so a typical view/
dispatch ratio is already inside the price. A group with a **higher** ratio than that baseline
converts better than its row says; a lower one, worse. `qsa_rope` is the standout: 20 removed
dispatches per QSA layer come with ~14 removed view nodes, so it should over-deliver — which is
also the direction the route kernel over-delivered in Calibration A.

**Two source-side uncertainties, stated rather than hidden.** (1) Whether MLX's `Reshape` does
generalised per-axis-split view detection or requires full C-contiguity: the QSA count assumes
the former, and under the latter several of the ~38 become forced copies that are already in the
census, so the total moves *down*, never up. (2) Multi-`None` index expressions
(`token_ok[None,None,None]`) are counted as one node each; if MLX lowers each axis separately the
QSA and residual-write rows rise by up to 2-3×. Neither moves the §4 ranking, because §4 is
priced on dispatches.

## 6. Uncertainty

1. **Dependent vs sibling is reasoned, not measured.** The census records `buffer_binds` but not
   *which* buffers, so MLX's barrier insertion cannot be recovered from it. `dep` comes from the
   dataflow shape visible in the dispatch stream (a `sin → multiply → concat` chain is dependent;
   three concat copies writing disjoint regions are siblings). If a group is more sibling-heavy
   than assumed, its G column shrinks toward zero and only the H column survives — for the four
   FUND rows that would be 0.07-0.09 ms each, i.e. **below the floor**. This is the single
   biggest risk in the table and the reason each row names its chain fraction.
2. **The prices themselves are one measurement each**, and the one lever with an end-to-end
   number came in 31 % better than they predict. Direction is favourable; magnitude is not pinned.
3. **The four FUND rows are not guaranteed additive on the G side.** Each removes launches from a
   different block type, but the GPU is byte-bound 81.7 % of the cycle; removing launches only
   converts while the launches are what the GPU is waiting on. Their H halves *are* additive.
4. **`qsa_head_layout` and `qsa_indexer_select` are priced on nodes and owned by bytes.** W68 must
   price them, not this table.
5. **This is one census file per stack, one seed, 16 K / 1 K.** The body is bit-identical across
   394 cycles within the file, so the *inventory* is exact for that configuration; a different
   context length changes the QSA indexer's grids (`c17408` is in the kernel name) but not the
   node counts, which are shape-independent.
6. **`MTPLX_FABLE_QSA_M4` is armed** in the measured stack — the source read could only infer
   it from the six cache-offset pairs, but the census settles it: the body contains
   `custom_kernel_mtplx_qwen4_qsa_m4_fused_kv_gather_c17408`, twelve of them, one per QSA layer.
7. **The composed census predates the route kernel.** Today's body is 2,751 − 288 = **~2,463
   dispatches**; every group above except `moe_router_glue` is unaffected, and the totals in §2
   should be read as "the body as captured", not "the body today".

## 7. Reproduce

```bash
# the inventory (one pass, 2.5 s)
python scripts/fable/census_verify_nodes.py \
  .benchmark-artifacts/pr391/w58-retained-composed-census-1788370553.jsonl

# the control, for the 3,669 cross-check against W63
python scripts/fable/census_verify_nodes.py \
  .benchmark-artifacts/pr391/w58-retained-control-census-1788370322.jsonl

# calibration B — W63's tool, which lives on worker/w63-graph-build-overlap:
#   git show worker/w63-graph-build-overlap:scripts/fable/census_verify_opener.py > /tmp/o.py
# then, with this worktree's census_retained_stack.py beside it, body length + 1:
python /tmp/o.py <composed census.jsonl> 2752
python /tmp/o.py <control census.jsonl>  3670

# tests
python -m pytest tests/test_fable_verify_node_census.py -q
```

`--dump-body body.tsv` writes the whole 2,751-row body with block, command buffer, op class,
kernel and grid — the working file for whoever builds one of the four.
