# W67 — the N-layer prefix

Branch `worker/w67-nlayer-prefix`, from `worker/w63-graph-build-overlap`
(tip `65e040ed`), merged with `experiments/fable-qwen38-80tps`. Runtime only.
Flag: `MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1` +
`MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS=N` (default **1** = W63's partition).
Read `docs/perf/graph-build-overlap.md` first — it owns the measurement this
lane is trying to spend.

## 0. What W63 left on the table

The retained-stack control census (`w58-retained-control-census-1788370322`,
382 cycles) puts **1.934 ms/cycle** of GPU idle immediately before the compiled
verify body, 86.9 % host-late, in 382 of 382 cycles — the host replay of the
`mx.compile`d physical-M4 verify graph. W63 hid `min(0.53, 1.93) = 0.53 ms` of
it by submitting layer 0 early, because layer 0 is the only layer that reads no
PLE auxiliary.

The rest was not unreachable, only unbuilt: `min(N × 0.53, (48−N)/48 × 1.93)`
peaks at **N = 3–4, 1.59–1.77 ms/cycle**.

## 1. The dependency table, from the code

`ple_layer_ids = [2]` (one-indexed) and `full_attention_interval = 4`, read out
of the production `config.json`
(`~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`):
48 layers, layers 3/7/…/47 full attention, one PLE layer at **index 1**.

`compiled_aux` reaches the model through exactly one contextvar,
`qwen4_exp._COMPILED_VERIFY_PLE`, and exactly one reader of it is on the verify
path: `NGramEmbedding.__call__` (`qwen4_exp.py:5087`), which returns the
compiled array outright when the scope is set. The only caller is
`PLELayer.__call__` → `self.ple_embedding(...)`, and the only caller of *that*
is `DecoderLayer.__call__`'s first statement, guarded by `if "ple" in self`.
`DecoderLayer.__init__` installs `self.ple` only when
`(layer_idx + 1) in args.ple_layer_ids`. (The other `_COMPILED_VERIFY_PLE.get()`
site, `Qwen4ExpTextModel._forward:5450`, is the *staging* guard on the eager
whole-model path; the split path calls layers directly and never reaches it.)

So:

| layer | type | has `ple` | reads `compiled_aux` **directly** | needs the prefix's output | state leaves | capture leaves |
|---:|---|---|---|---|---:|---:|
| 0 | linear (GDN) | no | **no** | — (reads `embed_tokens`) | 2 | 6 |
| 1 | linear (GDN) | **yes** | **YES** — `hidden + self.ple(hidden, ids, cache)` | layer 0's `hidden` | 4 | 9 |
| 2 | linear (GDN) | no | no | layer 1's `hidden` | 2 | 6 |
| 3 | **full attention (QSA)** | no | no | layer 2's `hidden` | 5 | 0 |
| 4 | linear (GDN) | no | no | layer 3's `hidden` | 2 | 6 |
| 5 | linear (GDN) | no | no | layer 4's `hidden` | 2 | 6 |

Layers 2..47 are aux-dependent only **transitively**, through layer 1's output.
That is what makes N > 1 a hoist question and not a dataflow question: exactly
one layer needs the auxiliary, and everything after it needs that layer.

Cumulative prefix census (asserted by `install_fixed_m4_overlap_split` and by
`tests/…::test_the_prefix_owns_exactly_layers_0_to_n_minus_1`):

| N | prefix state leaves | prefix capture leaves | prefix capture-plan entries | prefix reads the aux |
|---:|---:|---:|---:|---|
| 1 | 2 | 6 | 1 | no |
| 2 | 6 | 15 | 2 | **yes** |
| 3 | 8 | 21 | 3 | yes |
| 4 | 13 | 21 | 3 | yes |

(Totals: 134 state / 219 capture leaves — `35×2 + 1×4 + 12×5` and `36×6 + 3`.)

## 2. Is the hoist possible? Yes, and here is the proof obligation it meets

`prepare_aux` is `dispatch["prepare_aux"](input_ids, host_input_ids,
completion_tokens, committed_count)`. Both production implementations
(`_FixedM4SidecarAux.__call__`, `_FixedM4ExperimentalSidecarAux.__call__`)
**ignore `input_ids` entirely** — the parameter is literally spelled
`_input_ids` — and compute from `host_input_ids` (the Python list of the
window's four token values), `completion_tokens`, `committed_count` and the
construction-time `prompt_tail`. The materialized fallback
(`_prepare_compiled_verify_aux`) does read the device `input_ids` and the live
PLE cache entry, but both exist unchanged at the enqueue: the split path does
not rebind any live cache slot until the join.

At the enqueue site in `generation.py` all three host arguments already exist:
`verify_input` (the list `verify_input_array` was just built from), `tokens`,
and `len(tokens) - 1`. `tokens` is not appended to, extended, or rebound
anywhere between the enqueue and the verify — a source-level test pins that.

**Nothing `prepare_aux` needs is produced by the suffix.** The hoist is legal.

What it costs: the aux's host work (n-gram row ids in NumPy plus the sidecar
gather) now runs *before* the prefix submission instead of after it, so the
prefix starts that much later. That is why the hoist is taken only when it buys
something — at N = 1 the prefix does not read the auxiliary and the join builds
it exactly where W63 built it, so **the default arm is byte-for-byte W63's
schedule**.

The aux is built **once per window at every depth**. When the join refuses a
hoisted prefix (capacity/route generation moved, or the window advanced), the
fallback is handed the already-built object through
`_forward_installed_fixed_m4(..., compiled_aux=…)` rather than running
`prepare_aux` a second time — which would repeat its owned-row install and its
candidate-prefetch resolve inside one cycle.

## 3. Where the seam sits

The seam is one `mx.compile` boundary between layer `N-1` and layer `N`. The
value crossing it is always the same kind of thing — the `hidden` returned by
`_hyper_residual_write` (the MLP hyper-connection write: an elementwise
multiply-add ending in a reshape). What differs is the **consumer**:

| N | last prefix layer | first suffix operation on the seam value | elementwise chain cut? |
|---:|---|---|---|
| 1 | 0 (GDN) | `hidden + self.ple(hidden, ids, cache)` | **yes** |
| 2 | 1 (GDN, the PLE layer) | `attn_hyper_connection(hidden)` → `GroupedRMSNorm` / HC-M4 kernel | no |
| 3 | 2 (GDN) | `attn_hyper_connection(hidden)` | no |
| 4 | 3 (QSA) | `attn_hyper_connection(hidden)` | no |

`GatedResidual.__call__` opens with `hc_norm`, a `GroupedRMSNorm` over
`mx.fast.rms_norm` — or, under `MTPLX_FABLE_HC_M4`, a hand-written Metal kernel
reading `hyper_input` directly. Neither fuses with an elementwise producer, so
at N ≥ 2 the seam falls where MLX would not have fused across it anyway.
**N ≥ 2 cuts at a cleaner seam than N = 1 does.** That is an argument about
what MLX can fuse, not a proof about bits; see §6.

## 4. What was built

`install_fixed_m4_split` (PR391's, whose layer-0 census
`tests/test_qwen4_fixed_host_tokens_static` pins by source, including
`install.count("mx.compile(") == 2`) and `_forward_fixed_m4_prefix` /
`_forward_fixed_m4_suffix` (whose `inner.layers[0]` and
`range(1, len(inner.layers))` the same file pins) are **untouched**. W67 adds a
parallel pair.

| file | change |
|---|---|
| `mtplx/graph_build_overlap.py` | `LAYERS_ENV`, `layers()`, `MAX_LAYERS=8`, counters `prefix_layers` / `aux_hoisted`, depth-aware engagement line |
| `mtplx/qwen4_fixed_verify.py` | `_collect_fixed_m4_captures`, `_forward_fixed_m4_overlap_prefix(layer_count, compiled_aux)`, `_forward_fixed_m4_overlap_suffix(start)`, both bound on the runtime |
| `mtplx/graphbank.py` | `_fixed_m4_ple_layer_index`, `_make_fixed_m4_overlap_prefix_step`, `_make_fixed_m4_overlap_suffix_step`, `install_fixed_m4_overlap_split(N)` → `dispatch["overlap_split"]`; enqueue/join generalized; `FixedM4OverlapPrefix.compiled_aux` / `.layer_count`; `_forward_installed_fixed_m4(..., compiled_aux=None)` |
| `mtplx/generation.py` | the hook passes `host_input_ids` / `completion_tokens`; the engagement line names the installed depth |
| `scripts/fable/abba_driver.py` | `graph_build_overlap_layers` in the receipt block |
| `scripts/fable/micro_graph_build_overlap.py` | `--layers 1,2,3,4`, one overlap arm per depth, per-depth verdict table |
| `tests/test_fable_graph_build_overlap.py` | 101 tests, pure Python |

### The two arities

At depth 1 the traced prefix closure is `prefix_step(input_ids, *state_in)`; at
any depth containing the PLE layer it is
`prefix_step(input_ids, compiled_aux, *state_in)`. Passing `None` positionally
into the first form would change its signature and its trace, so the closure is
chosen at install time, not at call time.

### Invariants kept

* **`FixedM4OverlapPrefix` retains no input leaves.** No `state_in` field, and
  `_held_fixed_m4_split_refs` (PR391's untrimmed list, the shape that defeats
  donation) is untouched. A test walks the object graph and asserts no held
  value is one of the state-plan input leaves.
* **`MTPLX_FAMILY_CAPTURE_COMMIT`** — the join only rebinds cache slots
  (`entry.cache[slot] = …`, `entry.kv.cache[i] = …`) and never setitem-mutates,
  at every depth, so `snapshot_untrimmable_cache_lazy`'s COW pre-verify
  snapshot is unaffected. Every capture row the shipped route publishes is
  published, from whichever side of the seam owns it: the capture *plan* is
  split on the same layer boundary as the state plan, and the plan's offsets
  are cumulative, so `start` indexes the prefix's flat tuple below the shift
  and the suffix's above it — no guessing which entry owns which rows. The
  central test compares the whole cache census entry by entry and slot by slot
  against the monolithic route, at N = 1, 2, 3 and 4.
* **The materialization boundary** — `_prepare_compiled_verify_aux`'s docstring
  requires the aux to cross `async_eval` before becoming an `mx.compile` input.
  Whichever graph consumes it *first* is the one it must precede: the suffix at
  depth 1, the **prefix** at any depth past the PLE layer. Tested both ways,
  under `boundary` `both` and `pre`.
* **`demote()` drops a queued prefix** — unchanged from W63 and now tested
  directly: a promoted-cache demotion discards the prefix, counts it, and
  resets `_fixed_m4_split_generation` so the pair recompiles against the new
  shadow.
* **A depth knob without the lever refuses.**
  `MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS` set while
  `MTPLX_FABLE_GRAPH_BUILD_OVERLAP` is off raises at request setup — an arm
  labelled "N=3" that measured the control is the same lie the existing
  unsupported-route refusal exists to prevent. Re-arming the *same* depth on
  the same bank does not retrace; only a depth change forces the pair to
  recompile.
* **Install gate raises.** Depth outside `[1, layers-1]`, a drifted 134/219
  production census, more than one PLE layer in the capture layout, and the
  `raw_q4` aux contract at any depth past the PLE layer all raise from
  `arm_fixed_m4_graph_build_overlap` at the request boundary. (The `raw_q4`
  refusal is a real limitation, not a bug: that contract dequantizes the
  auxiliary *inside* the consuming graph and returns the expanded array as
  `_mtplx_verify_compiled_aux`, so moving the PLE layer into the prefix would
  move that dequantization and that output with it. The retained stack is on
  `materialized`, where `returns_aux` is False.)

## 5. Expected ms/cycle

`min(N × 0.53, (48−N)/48 × 1.934)`, with 0.53 = verify-body GPU / 48 from the
W63 census (an *average* layer, not a measurement of any particular one):

| N | prefix GPU | host build left to hide under | predicted saving | share of a 37.4 ms window |
|---:|---:|---:|---:|---:|
| 1 | 0.53 | 1.89 | **0.53 ms** | 1.4 % |
| 2 | 1.06 | 1.85 | **1.06 ms** | 2.8 % |
| 3 | 1.59 | 1.81 | **1.59 ms** | 4.3 % |
| 4 | 2.12 | 1.77 | **1.77 ms** | 4.7 % |
| 5 | 2.65 | 1.73 | 1.73 ms | 4.6 % |

N = 4 is the crossover; past it the host build runs out before the prefix does
and the curve turns over. **Measured** (§8): −0.33 / +0.62 / +0.78 / +0.70 ms
at N = 1/2/3/4 — the same shape, peaking at the same place, at roughly half
the predicted magnitude, and with N = 1 actually negative. Both terms are census aggregates, so the sweep is the
point: `--layers 1,2,3,4` prices all four in one guarded window, and the shape
of the curve (does it rise 1→3? does it flatten at 4?) is more informative than
any single number. Three things eat into it: the second command buffer pays its
own commit→start latency (the census attributes 0.25 ms of the 1.93 to driver
time, and W62's one-kernel buffer paid 362.8 µs), the split costs a second
tree-flatten and a second `async_eval`, and at N ≥ 2 the aux hoist delays the
prefix by the auxiliary's own host cost.

## 6. Exactness — still the gate, and still unproven

Everything the two graphs are fed is what the monolithic route feeds one: the
same `verify_input_array` object, the same `compiled_aux` from the same
unchanged `prepare_aux` and the same host arguments, the same state leaves in
the same order. The tests assert the resulting cache census is equal entry by
entry and slot by slot, at every depth.

**That is an argument about inputs, not a proof about bits.** §3 argues the
N ≥ 2 seam is cleaner than N = 1's, but `_bind_fixed_m4_device_commit` carries a
comment about exactly this class of effect ("the outer graph changes the
arithmetic schedule in rounding-sensitive states, first observed at seed 31,
cycle 91"). **The gate is the ABBA driver's `response_token_sha256` and its
per-cycle `accepted` list.** If they diverge from the control, the depth that
diverged is dead as written.

## 7. How to run it

### Micro sweep (guarded — loads the production model, runs 1 + 4 real decodes per rep)

```zsh
W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w67-nlayer-prefix
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
RG=/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py
PLIST=/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist
mkdir -p $W/.benchmark-artifacts/fable
env PYTHONPATH=$W $PY $RG --plist $PLIST --lock-timeout-seconds 5400 \
    --child-timeout-seconds 5400 \
  -- env PYTHONPATH=$W $PY $W/scripts/fable/micro_graph_build_overlap.py \
       --source $W --prompt-tokens 16384 --max-tokens 192 --reps 2 \
       --layers 1,2,3,4 \
       --json $W/.benchmark-artifacts/fable/micro-nlayer-prefix.json
```

Read `summary.by_depth` first. Per row: `installed_layers` must be `[N]` (a
row whose installed depth is not its label measured a partition it does not
name); `monolithic_windows` and `prefix_discarded` must be 0 (every one is a
window that ran the control's code or a prefix computed and thrown away);
`aux_hoisted` must equal the window count at N ≥ 2 and 0 at N = 1;
`measured_saving_ms` is read against `predicted_saving_ms`.

`prefix_discarded` matters slightly more at N ≥ 2 than it did at N = 1: a
discarded prefix at those depths also throws away an auxiliary whose
`_install_owned_rows` and candidate-prefetch `resolve` have already been
consumed, so the *next* window's hot-row lookup misses. Still correct — the
sidecar gather is the fallback — but it is a quiet quality-of-prefetch
regression that only `prefix_discarded > 0` would reveal. The hook and the
verify site are guarded identically and sit in the same loop iteration with no
`tokens` mutation between them, so the count should be 0.

### ABBA — 16K decode, the verdict

Substitute the best N from the micro sweep. Control carries the prewarm so the
candidate measures the lever and not the prewarm; `--python` is required
because this worktree has no `.venv` of its own.

```zsh
W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w67-nlayer-prefix
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
S=/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/1b1e4a52-8af8-4acc-a173-0bf81c785447/scratchpad
mkdir -p $S/w67 $W/.benchmark-artifacts/fable
/bin/zsh $S/w8/retry_guarded.zsh $S/w67/abba-n3.log "env PYTHONPATH=$W $PY" 9000 \
  -- $PY $W/scripts/fable/abba_window.py \
       --sequence 1788400622 --order ABBA \
       --label-prefix fable-w67-nlayer-prefix-n3 \
       --source $W --python $PY \
       --prompt-tokens 16384 --max-tokens 1024 \
       --control-flag=--prewarm-ngram-table \
       --candidate-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1 \
       --candidate-extra-env MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS=3
```

On the receipt, under each row's `ple_hot_rows` block:
`graph_build_overlap_armed` (a candidate with this `false` measured the
control), `graph_build_overlap_layers` (what the arm asked for) against
`graph_build_overlap.prefix_layers` (what it installed), then `suffix_joined`
vs `monolithic_windows`, `prefix_discarded`, `aux_hoisted`, `split_rebuilds`.

## 8. Measured — and the prefill question it raised

Depth sweep (real model, guarded, 192-token decodes), saving per cycle:
**N=1 −0.33 ms (worse than stock), N=2 +0.62, N=3 +0.78, N=4 +0.70**; token
digest matched stock at every depth. N=1 losing is the seam argument in §3
paying off exactly where it predicted: layer 0's residual→PLE-add is the one
cut that severs a fusible elementwise chain, and the split's own overhead
eats the 0.53 ms it buys.

Stacked 16K ABBA at N=3 (seq 1788400622, on HC_M4 + OPDIET + BLOCK_VERIFY +
ROUTE_KERNEL + PRESCATTER, max fans): cycle **36.13 → 35.57 ms (−0.558 ms,
−1.54 %**; seeds −2.34 / −0.69 / −1.61 %), decode 76.2 → 77.35 tok/s,
**digests identical 3/3**, `prefix_layers` 3,
`aux_hoisted == prefix_enqueued == suffix_joined ==` window count,
`prefix_discarded` 0, `split_rebuilds` 1 per arm. Retained for decode.

### The prefill side effect, attributed

The candidate arms' `prompt_eval_time_s` rose (+0.62 s mean, +0.20 s median)
and TTFT with it. Reduced from the twelve receipts, per arm, in seconds:

| | control (n=6) | candidate (n=6) | Δ mean | Δ median |
|---|---:|---:|---:|---:|
| prefill **chunk 0** | 2.157 | 2.719 | **+0.562** | **+0.118** |
| prefill chunks 1–7 | 11.258 | 11.321 | +0.063 | +0.046 |
| **`outside`** = `prompt_eval` − Σ chunk walls | **0.5364** | **0.5344** | **−0.0020** | **−0.0005** |
| `prompt_eval_time_s` | 13.951 | 14.574 | +0.623 | +0.202 |
| `ttft_s` − `prompt_eval_time_s` | 0.01209 | 0.01359 | +0.0015 | |
| `pre_first_token_setup_s` | 0.00326 | 0.00367 | **+0.00041** | |

**The whole regression is inside prefill chunk 0. None of it is inside the
part of the prefill span where construction could land.** `outside_s` — the
prefill span minus chunk compute — is identical to half a millisecond. And
`pre_first_token_setup_s`, the span that *does* contain both
`install_fixed_m4` and `arm_fixed_m4_graph_build_overlap`, moved by
**+0.41 ms**: that is the lane's entire construction cost, and it is 0.07 % of
the +0.62 s that was blamed on it.

Chunk 0 is the process-cold chunk and it is noisy on **both** arms: control
spread 1.910–2.591 s (0.68 s), candidate 1.990–5.283 s. One candidate arm
(B1/s20260830) hit 5.283 s; drop that single excursion and the candidate's
chunk-0 mean is 2.206 s against the control's 2.157 s (+49 ms, inside the
control's own spread). The second-largest chunk 0 in the whole matrix,
2.591 s, is a **control**.

### Where construction actually happens

`mtplx/generation.py`, in order, and pinned by
`test_construction_happens_after_the_prefill_span_not_inside_it`:

1. `restore_or_prefill_prompt_state(...)` — **`prompt_eval_time_s` is measured
   inside this call.** Nothing in this lane is reachable from it: a test
   asserts no W67 symbol appears in `qwen4_exp.py`, `fable_prefill_chunk.py`
   or `ple_prefill_lookahead.py`, and the decode loop's hook is the lane's
   only call site.
2. `pre_first_token_setup_started = time.perf_counter()` — after the prefill.
3. `compiled_verify_bank.install_fixed_m4(...)` — PR391's, on **both** arms.
4. `compiled_verify_bank.arm_fixed_m4_graph_build_overlap()` → the split
   install. **+0.41 ms**, measured.
5. `pre_first_token_setup_s = ...` — the span closes.
6. Decode cycle 1: the first `prefix_fn` / `suffix_fn` call. **This is where
   `mx.compile` actually traces** — `mx.compile(f)` only wraps; it traces on
   first call. So the trace is inside `decode_elapsed_s`, never prefill. The
   receipts confirm it from the other side: the control shows
   `compiled_verify.traces == 1` (it traced the monolithic graph) and the
   candidate shows **0** in 6/6 — the monolithic graph is never traced on the
   candidate, because every window joins the split.

So construction cannot move to model load: it needs `dispatch`, which
`install_fixed_m4(cache, prompt_ids=...)` builds from the *request's*
prefilled cache. It is already as early as it can be, and it is already
outside the prefill span.

### The real construction bug this found

The bank is constructed **per request**. `install_fixed_m4_overlap_split` was
building two fresh closures and wrapping them in `mx.compile` each time, so
`mx.compile` would re-trace **both graphs on the first cycle of every
request** — where the shipped monolithic route traces once per *process* via
`_SHARED_VERIFY_STEPS` (whose docstring prices one trace at ~1 s wall at 7k
leaves). Invisible in a one-request-per-process A/B; two full re-traces per
request in a served process.

Fixed by giving the pair the same sharing: `_SHARED_OVERLAP_SPLITS`, keyed on
`(runtime id, capture backend, capture layout, aux presence, spec, **depth**,
**needs_aux**, hidden variant, QSA gather route, aux contract,
exact-verify route)`, with the same `trace_host["bank"]` indirection so a
shared closure always reads the live bank, the same `weakref` guard against a
recycled runtime id after a model swap, and the same
`MTPLX_COMPILED_VERIFY_SHARED_TRACES` off-switch. Leaf-shape changes
(capacity growth) stay `mx.compile`'s own retrace dimension, as on the
monolithic path.

### New receipt fields

| field | what it answers |
|---|---|
| `graph_build_overlap.construction_ms` / `_calls` | what `install_fixed_m4_overlap_split` cost (inside `pre_first_token_setup_s`) |
| `graph_build_overlap.first_prefix_build_ms` / `first_suffix_build_ms` | the two graphs' one-time `mx.compile` traces, in decode cycle 1 — always on, first-call only, so no `timing` item is needed and no A/B carries an instrument on one arm |
| `graph_build_overlap.split_shared_hits` | installs that reused the process-wide pair. On a served process every install after the first must be a hit |
| `prefill_split` = `{chunk0_s, rest_s, outside_s, chunks}` | attributes any prefill delta to the cold first chunk vs the rest vs the non-chunk remainder, without a twelve-receipt hand-reduce |

## 9. What was NOT verified

* **Nothing was run on the GPU.** No arm, no micro, no test that evaluates an
  MLX array. The 101 tests replace `graphbank.mx` with a recorder and every
  "array" with a sentinel that raises on `__int__` / `__iter__` / `__array__`.
* **Bit-identity.** See §6. The seam argument is about what MLX fuses, not
  about what it emits.
* **Whether MLX's tracer accepts the new closures on the real runtime.** The
  census and partition arithmetic is covered against the production geometry
  with `mx.compile` stubbed to identity; the tracer first sees the real
  closures on the first cycle of a real arm. In particular the two-arity prefix
  (`compiled_aux` as an explicit compiled input at N ≥ 2) has never been traced.
* **Non-default `MTPLX_COMPILED_VERIFY_BOUNDARY`.** Under `post` or `none` the
  hoisted auxiliary is handed to the prefix graph without crossing
  `async_eval` first — exactly what the shipped monolithic route does under
  those settings, so the risk profile is unchanged, but it has never been run
  and the ABBA does not run it.
* **That chunk 0's excursion is unrelated to this lane.** The evidence is
  strong (the delta is entirely in the process-cold chunk, the control's own
  chunk-0 spread covers most of it, the second-largest chunk 0 in the matrix
  is a control, and the non-chunk part of the prefill span is identical) but
  it is an attribution, not a controlled experiment. A repeat ABBA now
  carries `prefill_split` and the construction counters, so the next run
  settles it from the receipt alone.
* **The process-wide pair cache has never been exercised on the real
  runtime.** Its key, its host re-pointing and its weakref guard are covered
  pure-Python against the production census; whether MLX reuses the trace as
  the monolithic path does first shows on a two-request process.
* **The per-layer GPU estimate.** 0.53 ms is `verify-body GPU / 48` from the
  census aggregate. Layers are not equal: layer 1 carries the PLE block (more
  than average) and the QSA layers are a different shape from the GDN ones, so
  the N = 2 and N = 4 predictions in §5 are the loosest in the table.
* **`tests/test_qwen4_exp_capture_commit.py` and
  `tests/test_pr391_float32_d3_core.py`** touch `mtplx/graphbank.py` and build
  and evaluate MLX arrays; they were **not** run.
  `tests/test_qwen4_fixed_host_tokens_static.py` was run: 5 failures,
  message-for-message identical to the base branch's 5 (the PR391 split lane in
  flight), none new.
