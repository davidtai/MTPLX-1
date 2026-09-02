# W63 — the compiled verify graph's host construction, re-derived on the retained stack

Branch `worker/w63-graph-build-overlap` (from `worker/w62-ple-boundary`, which carries W56
and W46 and is merged with `experiments/fable-qwen38-80tps`). Runtime only.
Flag: `MTPLX_FABLE_GRAPH_BUILD_OVERLAP`, instrument via
`MTPLX_FABLE_GRAPH_BUILD_OVERLAP_ITEMS=timing`. Default off.

## 0. The brief's premise, checked first

W62 measured gap B — the compiled verify graph's own host replay — on the **PR391 device-D3**
census (`current-exact-early-d3-census-2410.jsonl`): 1.780 ms/cycle, 90.9 % host-late,
374/374 cycles. The retained-stack control census landed mid-task with a reduce that
reports the whole "D3 → PLE dequant → target gather" boundary at only **0.613 ms/cycle in
112/382 cycles** — which reads like the target evaporated.

It did not. Reduced from
`.benchmark-artifacts/pr391/w58-retained-control-census-1788370322.jsonl` with
`census_retained_stack`'s own definitions (10 µs gap floor, union-busy timeline, lm_head
cycle marker, `host_late` split), window ops 63,477–1,853,245, 382 cycles,
32.402 busy + 7.301 idle ms/cycle:

**The compiled verify body opens at a fixed offset of 3,668 dispatches before the cycle's
`lm_head`, in 382 of 382 cycles, and the GPU idle immediately before that command buffer is:**

| | retained stack (w58) | PR391 device-D3 (D/2410) |
|---|---:|---:|
| idle before the verify body | **1.934 ms/cycle** | 1.780 ms/cycle |
| host-late share | 86.9 % | 90.9 % |
| cycles in which it appears | **382 / 382** | 374 / 374 |
| median / min / max | 1.690 / 1.257 / 20.5 ms | 1.717 median |
| share of that stack's idle | **26.5 %** of 7.301 | 27.8 % of 6.395 |

**It grew.** The 0.613 ms/cycle figure is a **classifier artifact**, not a disappearance:
`census_retained_stack.is_ple_boundary` keys on the *kernel pair*, and on the retained stack
the buffer that precedes the verify body ends in `g1_copybfloat16bfloat16` in 270 cycles and
in `affine_dequantize_..._gs_32_b_4` in only 112 — MLX batches the PLE q4 dequant into a
larger command buffer on most cycles instead of giving it its own. Both families are the same
event, and they sum:

```
  269 ev   1.267 ms/cyc   host-late 87.4 %   g1_copy...        -> gather_front(bf16,int32,2)
  112 ev   0.613 ms/cyc   host-late 85.0 %   affine_deq gs32b4 -> gather_front(bf16,int32,2)
                          -----------------
                          1.880 ms/cyc  (the 10 µs floor drops the remaining 0.054)
```

Reproduce with `scripts/fable/census_verify_opener.py <census.jsonl> 3669` — it streams the
file once, imports `census_retained_stack`'s own definitions, and anchors on the verify body's
fixed dispatch offset instead of the kernel pair, so it reports one row per cycle with no
10 µs floor.

### Re-ranked idle on the retained stack (7.301 ms/cycle)

| term | ms/cycle | share | host-late | owner |
|---|---:|---:|---:|---|
| draft loop's per-depth host syncs (`v_Exp →` gather/copy, ~3.7 ev/cycle) | 3.843 | 52.6 % | 87 %, 81 % | **not this lane** — the census's own verdict is "SERIAL per-depth loop WITH a host sync per depth" |
| **compiled verify graph host construction** | **1.934** | **26.5 %** | 86.9 % | **this lane** |
| `gather_front(uint32) → affine_dequantize gs64 b8` (1,135 ev) | 0.554 | 7.6 % | **39.2 %** | mostly driver latency, not a host lever |
| PLE auxiliary submission (W62's gap A) | 0.496 | 6.8 % | 88.2 % | W62's `warm_skip` / `primary_vectorized` |
| everything else | ~0.47 | 6.5 % | | |

So the answer to "is the graph-build gap gone or small here" is **no on both counts** — it is
the second-largest idle term on the retained stack and larger in absolute terms than it was
on the census the brief pointed at. Continued as briefed.

## 1. The per-cycle host timeline, and what depends on the drafted tokens

The retained stack (`--target-mode batched`, no `--d3-softfloat64-route`) drafts with the
**stock per-depth loop**, not the PR391 joint D3 core: three `mx.eval` + `np.asarray` +
`rng.choice` round trips, one per depth. There is no single "D3 sync" to hoist in front of;
the window's ids arrive incrementally and the last one arrives at the last depth's sync.

| # | host step | site | needs the drafted ids? | GPU meanwhile |
|---|---|---|---|---|
| 1 | draft depth 1/2/3, each ending in one host sync | `generation.py` per-depth loop | produces them | draft head, then idle for the sync (3.843 ms/cycle total) |
| 2 | `record_adaptive_width_event`, event dicts | ~12.6k | no | idle |
| 3 | `before_verify = snapshot_untrimmable_cache_lazy(cache)` | 12608 | no | idle (lazy views, cheap) |
| 4 | `verify_input = [primary] + draft_tokens`; `verify_input_array = mx.array([verify_input])` | 12653/12680 | **yes** — first statement that owns the whole window | idle |
| 5 | **← W63 enqueues the layer-0 prefix here** | 12701 | takes the array from 4 unchanged | **layer 0, ~0.53 ms** |
| 6 | `lazy_bonus_verify` / `speculative_bonus` event dicts, `set_native_mlp_context` | 12706–12740 | no | (5) |
| 7 | `_transition_fixed_m4_generation`, layers-1..47 state gather | `graphbank` | no | (5) |
| 8 | `prepare_aux(...)` → n-gram row ids, sidecar row read, `mx.array`/`mx.dequantize` | `qwen4_fixed_verify._FixedM4SidecarAux.__call__` | **yes** — host NumPy over the window's token values | (5) |
| 9 | `mx.async_eval(compiled_aux, *state_in)` | `graphbank` | — | PLE q4 dequant, 12 µs |
| 10 | **the suffix replay: ~5,090 nodes, ~1.9 ms** | `split["suffix_fn"](...)` | needs (8)'s array | **idle — this is the 1.934 ms** |
| 11 | `_unpack_fixed_m4_outputs` + state/capture rebind (~350 attribute writes) | `graphbank` | — | idle |
| 12 | `mx.async_eval(*outputs)` | `graphbank` | — | verify body, 3,668 dispatches |

**What depends on the sync and what does not.** Steps 2, 3, 6, 7 are token-independent but
cost nothing. Step 8 is genuinely token-dependent: the n-gram row ids are host NumPy over the
window's token *values*. Step 10 is token-dependent only **transitively** — `compiled_aux` is
an input to layer 1, and every node from layer 1 to the head is a dataflow descendant of it,
so their replay cannot be hoisted in front of step 8.

Step 5 is the exception, and it is the whole lever: **layer 0 reads no PLE auxiliary.** The
production config has `ple_layer_ids = [2]` (one-indexed) — one PLE layer, at index 1 — and
`qwen4_fixed_verify._forward_fixed_m4_prefix` accordingly runs layer 0 *outside*
`compiled_verify_ple_scope`. So layer 0 is exactly the aux-independent part of the window,
and it is the only part that can be submitted before the aux exists.

## 2. What was built

`install_fixed_m4_split` (already in `graphbank.py`, unwired) compiles exactly this partition:
`prefix_fn` = embedding + layer 0, `suffix_fn` = layers 1..47 + head, with construction-time
assertions that layer 0 carries 2 state leaves and 6 capture leaves and that the remainder is
132 / 213 — which is the production census (48 layers, 36 linear / 12 full attention, one PLE
layer at index 1: 35×2 + 1×4 + 12×5 = 134 state leaves, 36×6 + 3 = 219 capture leaves).

The two compiled graphs fit. **Its transaction wrappers do not**, and are left alone:

* `enqueue_fixed_m4_prefix` / `forward_fixed_m4_suffix` belong to PR391's device-committed
  split lane, whose invariants `tests/test_qwen4_fixed_host_tokens_static.py` pins by source
  (and which is red on this branch's base — 5 pre-existing failures, someone else's lane in
  flight). Editing them would be taking over that lane.
* They also never publish `entry._mtplx_verify_rows` / `_mtplx_verify_ple` /
  `_mtplx_verify_compiled_aux`, which the unchanged `_bind_fixed_m4_device_commit` reads for
  every GDN layer including layer 0, so wiring them as-is would break the commit.
* `FixedM4Prefix` retains `state_in` and `_held_fixed_m4_split_refs` is appended to without a
  trim — one layer-0 state + capture set pinned per window for the whole generation, and a
  live Python reference on an input buffer at `async_eval` time, which is exactly the shape
  that defeats MLX donation (`TensorOffsetKVCache.update_and_fetch`'s rollback slice, the
  17 MB/step bug the brief warned about).

So the new plumbing is minimal and separate: `FixedM4OverlapPrefix` (no `state_in`),
`arm_fixed_m4_graph_build_overlap`, `enqueue_fixed_m4_overlap_prefix`,
`forward_fixed_m4_overlap`, `discard_fixed_m4_overlap_prefix`, one single-slot field.
`forward_fixed_m4_overlap` publishes exactly what `_forward_installed_fixed_m4` publishes,
in the same order, and falls back to `_forward_installed_fixed_m4` — same call, same graph —
whenever no usable prefix is queued.

### Files

| file | change |
|---|---|
| `mtplx/graph_build_overlap.py` | new: flag, items, receipt counters, engagement line |
| `mtplx/graphbank.py` | new dataclass + 5 methods; `demote()` drops a queued prefix |
| `mtplx/generation.py` | one import, one setup block, one hook, `_fixed_m4_verify` indirection |
| `scripts/fable/abba_driver.py` | `graph_build_overlap` / `_armed` in the `ple_hot_rows` receipt block |
| `scripts/fable/micro_graph_build_overlap.py` | new (guarded) |
| `tests/test_fable_graph_build_overlap.py` | new, 26 tests, pure Python |

### Invariants the compiled capture imposes, and how they are kept

* **`MTPLX_FAMILY_CAPTURE_COMMIT`** — the family lane rebinds cache slots and never
  setitem-mutates, which is what makes `snapshot_untrimmable_cache_lazy` COW-safe. The
  overlap path only rebinds (`entry.cache[slot] = ...`, `entry.kv.cache[i] = ...`), never
  mutates, so the pre-verify snapshot the commit replays from is unaffected.
* **the materialization boundary** — `_prepare_compiled_verify_aux`'s docstring requires the
  aux to cross `async_eval` before becoming an `mx.compile` input, "otherwise MLX sees the row
  graph's cache leaves as uncaptured inputs to the verifier graph". `forward_fixed_m4_overlap`
  keeps `mx.async_eval(compiled_aux, *state_in)` under the same `boundary in ("both","pre")`
  condition, on the layers-1..47 state only. A test asserts the aux is the first argument of
  that submission.
* **donation** — `_clear_shadow_leaf_refs()` runs once per cycle, at the prefix (the first of
  the two submissions), where the monolithic route runs it before its single one; the join
  rebinds every state slot before `mx.async_eval(*outputs)` and drops `state_in`, as
  `_forward_installed_fixed_m4` does. The prefix never retains its input leaves.
* **capacity / route generations** — `_transition_fixed_m4_generation` recompiles
  `dispatch["fn"]` and rebuilds the shadow but knows nothing about `dispatch["split"]`;
  `_refresh_fixed_m4_split` recompiles the pair when the generation moves, and a prefix
  stamped with a stale generation (or a different `committed_count`) is refused and the
  window falls back to the monolithic route.

## 3. Expected ms/cycle, honestly

The verify body is 3,668 of the cycle's 4,685 dispatches and about 25.4 of its 32.4 ms of
GPU, so **one of 48 layers is ~0.53 ms**. The idle it can hide under is 1.934 ms, so the
saving is bounded by the prefix, not by the idle:

```
saving = min(prefix GPU 0.53, host build 1.93) = 0.53 ms/cycle
```

On a 37.4 ms production window that is **~1.4 %** — about 2× the ABBA within-seed floor
(0.3–0.7 % = 0.11–0.26 ms). Measurable, not comfortable. Two things eat into it: the extra
command buffer pays its own commit→start latency (the census measures 0.25 ms of the 1.93 ms
as driver, and gap A's one-kernel buffer paid 362.8 µs on census D), and the split costs a
second tree-flatten and a second `async_eval` of host time.

**The bigger prize, as design, not code.** Nothing forces the prefix to stop at layer 0 — only
the aux dependency does, and `prepare_aux` could be hoisted ahead of the prefix (it is host
work the cycle already pays). With an N-layer prefix the saving is
`min(N × 0.53, (48−N)/48 × 1.93)`, which peaks at **N = 3–4, ~1.7–1.8 ms/cycle (~4.5 %)**.
That needs (a) `prepare_aux` moved in front of the enqueue, (b) a generalized split —
`install_fixed_m4_split` hard-codes the layer-0 partition and its census assertions, and
`_forward_fixed_m4_prefix` hard-codes `inner.layers[0]` — and (c) one more `mx.compile` fusion
seam, i.e. one more place a last bit can move. Not built here.

**And the smallest change to the input contract that would make the *rest* of gap B
separable.** The suffix's ~5,090 nodes are unreachable because `compiled_aux` is a host-built
array: `_ngram_rows_np` computes 64 row ids from the window's token *values* in NumPy, and
`_SidecarGather` reads those rows out of a memmap. If instead the n-gram hash were computed
**on device** from the device-resident drafted ids and the rows selected on device out of a
resident `[4, 20, 16, 100]` candidate tensor (the shape
`docs/perf/ple-candidate-prefetch-phase2.md` already specifies), then `compiled_aux` would be
a *lazy device array available before any host sync* — and the entire 5,200-node replay could
be done while the draft chain is still on the GPU, removing the 1.93 ms **and** the
0.496 ms of gap A. That is W62's design 2. It is a Metal kernel plus a gather, i.e. a
model-path change, out of this program's runtime-only scope.

## 4. Exactness

Everything the two graphs are fed is what the monolithic route feeds one graph:
the **same `verify_input_array` object** (the hook does not rebuild it), the same
`compiled_aux` from the same unchanged `prepare_aux` and `host_input_ids`, the same state
leaves in the same order. `tests/test_fable_graph_build_overlap.py` runs both paths against
the same fake graphs and asserts the resulting cache census is equal entry by entry and slot
by slot — every cache slot, every `_mtplx_verify_rows`, every `_mtplx_verify_ple`, the
`_mtplx_verify_compiled_aux` identity, and every QSA `rollback_state` — rather than asserting
the argument.

**That is an argument about inputs, not a proof about bits.** The change puts an `mx.compile`
boundary between layer 0 and layer 1. MLX fuses element-wise chains inside a compiled graph,
and a fused chain can hold an intermediate in a wider register where an unfused one
round-trips through bf16 memory. `_bind_fixed_m4_device_commit` carries a comment about
exactly this class of effect ("the outer graph changes the arithmetic schedule in
rounding-sensitive states, first observed at seed 31, cycle 91"), so a last-bit difference at
the seam is possible. **The gate is the ABBA driver's `response_token_sha256` and its
per-cycle `accepted` list.** If they diverge from the control, this lever is dead as written.
Nothing in this lane touches `output.tokens` or `stats.events`, so that receipt is produced
unchanged.

## 5. How to run it

### Micro (guarded — loads the production model, runs two real decodes)

```zsh
W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w63-graph-build-overlap
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
RG=/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py
PLIST=/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist
mkdir -p $W/.benchmark-artifacts/fable
env PYTHONPATH=$W $PY $RG --plist $PLIST --lock-timeout-seconds 3600 \
    --child-timeout-seconds 3600 \
  -- env PYTHONPATH=$W $PY $W/scripts/fable/micro_graph_build_overlap.py \
       --source $W --prompt-tokens 16384 --max-tokens 192 --reps 2 \
       --json $W/.benchmark-artifacts/fable/micro-graph-build-overlap.json
```

Read `windows` first: an `overlap` row whose `graph_build_overlap.suffix_joined` is below its
window count ran the shipped monolithic route for the difference, and its delta is diluted by
exactly that fraction. `monolithic_build_ms` on the stock arm is the 1.93 ms measured from
inside the process; `prefix_build_ms + suffix_build_ms` on the overlap arm is the same host
work split, and it should be **equal or slightly larger** — if it is much larger, the split's
host cost is eating the win and the lever is not worth its complexity.

### ABBA — 16K decode, control `--prewarm-ngram-table`, candidate `MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1`

```zsh
W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w63-graph-build-overlap
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
S=/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/1b1e4a52-8af8-4acc-a173-0bf81c785447/scratchpad
mkdir -p $S/w63 $W/.benchmark-artifacts/fable
/bin/zsh $S/w8/retry_guarded.zsh $S/w63/abba.log "env PYTHONPATH=$W $PY" 9000 \
  -- $PY $W/scripts/fable/abba_window.py \
       --sequence 1788400622 --order ABBA \
       --label-prefix fable-w63-graph-build-overlap \
       --source $W --python $PY \
       --prompt-tokens 16384 --max-tokens 1024 \
       --control-flag=--prewarm-ngram-table \
       --candidate-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1
```

`--python $PY` is required: this worktree has no `.venv` of its own.
`--control-flag=--prewarm-ngram-table` puts the prewarm on **both** arms, so the candidate
measures the lever and not the prewarm.

Follow-up arm, only if the first is positive or ambiguous — the host-phase split on BOTH arms,
so the two replays can be attributed inside the process rather than from a census:

```zsh
--control-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1 \
--control-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP_ITEMS=timing \
--candidate-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1 \
--candidate-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP_ITEMS=timing
```

(That arms the lever on both arms and measures only the instrument's cost — use it to read
`prefix_build_ms` / `suffix_build_ms`, not to decide the lever.)

## 6. Reading the receipt

Under each row's `ple_hot_rows` block:

* `graph_build_overlap_armed` — whether the flag reached the process at all. A candidate with
  this `false` measured the control while wearing the candidate's label.
* `suffix_joined` vs `monolithic_windows` — the verdict gate. Every `monolithic_window` is a
  window that ran the shipped route.
* `prefix_discarded` — should be 0. Every one is a layer-0 forward computed and thrown away.
* `split_rebuilds` — capacity/route generation transitions; a handful over a 1,024-token run
  is normal, a large number means the pair is being recompiled inside the measured window.
* `prefix_build_ms` / `suffix_build_ms` — populated only with the `timing` item.

## 7. What was NOT verified

* **Nothing was run on the GPU.** No arm, no micro, no test that evaluates an MLX array. The
  26 new tests replace `graphbank.mx` with a recorder and every array with a sentinel that
  raises on `__int__` / `__iter__` / `__array__`.
* **Bit-identity.** See §4 — it is the ABBA's job and it is a real risk, not a formality.
* **The 0.53 ms layer-0 GPU estimate** is `verify-body GPU / 48` from the census's aggregate,
  not a measurement of layer 0. If layer 0 is cheaper than the average layer (it has no PLE
  add), the win is smaller.
* `tests/test_graphbank_compiled_verify.py` and `tests/test_runtime_obs_graphbank.py` touch
  `mtplx/graphbank.py` and were **not** run — they build and evaluate MLX arrays.
  `tests/test_qwen4_fixed_host_tokens_static.py` was run: 5 failures, byte-identical to the
  base branch's 5 (the PR391 split lane in flight), none new. Full pure-Python set after the
  merge with the main tip: **166 passed, 5 pre-existing failed**.
* `install_fixed_m4_split`'s four census assertions **are** covered — one test drives the real
  method against the production geometry (48 layers, 36 linear / 12 full attention, PLE at
  index 1) with `mx.compile` stubbed to identity, so an arm cannot fail at request setup on a
  census drift without that test going red first. What is *not* covered is whether MLX's
  tracer accepts the two closures on the real runtime; that first happens on the first cycle
  of a real arm.
* **Non-default `MTPLX_COMPILED_VERIFY_BOUNDARY`.** The join keeps the aux submission under
  the same `boundary in ("both", "pre")` condition as the shipped route, but the prefix roots
  its own outputs unconditionally. Under `boundary=none` with donation off — a combination
  the ABBA never runs — that schedules work the monolithic route would have left lazy. Still
  exact (`async_eval` only schedules), but untested.
