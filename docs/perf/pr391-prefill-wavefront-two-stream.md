# Wavefront prefill on two Metal queues (report M, row B4)

Status: **NEEDS-DATA, seam only.** The falsifier is written and not run;
nothing in the serving path is wired. This note says what would have to be
built, what it would cost, why it is bit-exact, and the three failure modes
that decide it.

## The shape

Chunked prefill walks a (chunk, layer) grid. Today it walks it one whole
chunk at a time: `_prefill_committed_mtp_history_streaming`
(`mtplx/generation.py:6442`) loops over `_iter_prefill_chunk_spans`, calls
`rt.forward_ar(chunk, cache=cache)` for all 48 layers, then blocks the host on
`_eval(logits_chunk, hidden_chunk)`.

The grid has exactly two dependency edges:

```
(k, L) -> (k, L+1)     hidden state
(k, L) -> (k+1, L)     KV + GDN recurrent state (cache entry L)
```

Both point strictly backwards along the anti-diagonals. So `(k, L+1)` and
`(k+1, L)` are independent — they share no cache entry and no hidden state —
and a schedule that issues anti-diagonals in order is legal. That is the
wavefront: chunk k at layer L+1 runs alongside chunk k+1 at layer L.

MLX has the machinery. `mx.new_stream(mx.gpu)` makes a second stream,
`mx.stream(s)` scopes op placement, and MLX inserts the cross-stream
dependency when an array produced on one stream is consumed on another. The
repo already uses this shape once, in `mtplx/cache_state.py:637`
(`async_per_head`).

## What is actually being bought

The 16K prefill census puts roughly 1.9 s in bandwidth-bound families —
elementwise, copies, norms, routing gathers, softmax/mask. Those queue behind
the compute-bound GEMMs today. A second queue could run one lane's
bandwidth-bound tail against the other lane's GEMM.

That is the *only* thing the wavefront buys. It does not improve MoE grouped
GEMM tile occupancy — each lane still routes 2,048 rows over 512 experts (40
rows per expert), the same as today. Widening the chunk to 4,096 is the lever
for occupancy, and it is a different lever at the same memory price (below).

## The falsifier

`scripts/fable/micro_two_stream_prefill.py`. One GDN `DecoderLayer` and one
QSA `DecoderLayer` at production shapes, 2,048 rows, synthetic 4-bit weights
under a 2 GiB budget. Arms:

| arm | what |
| --- | --- |
| `serial` | the four-node tile on one stream — today's shape |
| `wavefront` | same four nodes, same order, `{n1, n2}` on two streams |
| `independent` | the two bodies with no dependency, one per stream — the ceiling and the concurrency probe |

GO if `wavefront` is ≥15% faster than `serial` **and** `independent` shows
positive overlap **and** the tile's own arithmetic ceiling clears 15%. A
wavefront win with a flat `independent` has no mechanism behind it and the
script refuses it. The script also asserts arm `wavefront` is bit-identical to
arm `serial` and prints the differing-element count.

Two things about how it measures, both of which would otherwise hand the
wavefront a free win:

* **Verdicts read total wall time (host build + encode + GPU), not eval
  alone.** `mx.async_eval` deliberately moves GPU work out of the eval window
  and into the build window, so an eval-only comparison would credit the
  wavefront for work it merely relocated. The build/eval split stays in the
  receipt because it is the diagnosis: a wavefront that loses on total while
  its build doubles is host-bound, which is prior 2.
* **A 2×2 tile has exactly one overlappable pair**, so the largest saving it
  can show is `min(t_gdn, t_qsa) / (2·t_gdn + 2·t_qsa)` — 25% when the two
  bodies cost the same, 12.5% when one is 3× the other. If that ceiling is
  under 15% the run is *inconclusive for the row*, not a NO-GO for it, and the
  script says INCONCLUSIVE rather than pretending it measured something.

Guarded command is in the script docstring; `--self-test` and `--shapes`
import no MLX and are safe off-window.

### Three priors, all pointing the same way

1. `async_per_head` — the repo's existing multi-stream route — measured
   **4–7× slower** than single-stream below 64K context (issue #228, quoted
   at `mtplx/commands/public.py:342`). Extra Metal queues on this box have
   already cost more than they bought once.
2. `mtplx/prefill_rungs.py` says the prefill GPU idle is *host-build lag*:
   "MLX builds a whole prefill-chunk forward lazily and dispatches only at the
   end-of-chunk eval, so the GPU idles while the host walks 64 layers of graph
   construction." MLX streams are thread-local
   (`mtplx/backends/gemma4_assistant.py:2340`), so both wavefront lanes are
   built by the same Python thread. If the idle is host lag, the wavefront
   makes it worse.
3. A 2,048-row layer body is not a small kernel. If the routed MoE GEMM
   already saturates the GPU there is no idle width to fill.

### The cheaper adjacent lever

`MTPLX_PREFILL_ASYNC_RUNGS` (`mtplx/prefill_rungs.py`) already implements the
one-stream version of this idea: `mx.async_eval` every Nth layer inside one
chunk, so the GPU executes layer k while the host builds layer k+1. It is off
by default and **only installed for the `qwen3_5` classes — it was never
wired to `qwen4_exp`.** It needs no cache reordering, no PLE staging change,
no boundary-snapshot change, and `async_eval` changes scheduling and never
values. If prefill idle is host lag, that is a one-file port with none of the
hazards below, and it should be priced before anyone builds the wavefront.

## The schedule, if it is built

`mtplx/fable_prefill_wavefront.py` (pure arithmetic, no MLX, nothing in the
serving path imports it) owns it. The wavefront is **grouped, not
continuous**: chunks are cut into groups of `lanes` (default 2), each group
runs its own diagonal, and the pipeline drains between groups.

```
chunk_groups(8, lanes=2, tail_solo_chunk=True)
  -> [(0, 1), (2, 3), (4, 5), (6,), (7,)]

wavefront_steps(2, 2, lanes=2)
  -> [[(0, 0)], [(0, 1), (1, 0)], [(1, 1)]]
```

A continuous wavefront over an 8-chunk prompt reaches **8 lanes live**
(`lanes_live(8, 48, lanes=0) == 8`), which multiplies the QSA prefill
transient by 8. That is not a memory question, it is an OOM. Grouping bounds
it at exactly `lanes`.

Draining costs almost nothing: a group of `lanes` chunks over `layers` layers
takes `layers + lanes - 1` steps instead of `layers`, so at lanes=2,
layers=48 the schedule is two wide for **95.9%** of its steps
(`overlappable_step_fraction(8, 48, lanes=2)`). And the drain is what makes
the per-chunk bookkeeping safe — see the boundary-capture failure mode.

## Memory model

`plan_prefill_chunk_memory` now takes `live_lanes` (default 1, so the shipped
path is unchanged). The QSA dense prefill transient is 12.75 B per (chunk row
× context token) with 4 layers live (`mtplx/memory_plan.py:178`), and it is
linear in the live query rows, so `lanes` enters as a plain multiplier: each
live lane materializes its own attention/indexer chain, and by construction
they are live at the same moment.

At the production cell (16,384 prompt tokens, 90 GiB wired limit, 2 GiB
margin, census resident 87.39 GB):

| geometry | projected peak | attention work term |
| --- | ---: | ---: |
| 8 × 2,048, serial (shipped) | 87.39 GB | 150,994,944 |
| **2 lanes × 2,048, wavefront** | **89.11 GB** | **150,994,944** |
| 4 × 4,096, serial | 89.11 GB | 167,772,160 (+11.1%) |

That is the one genuinely attractive property: **a 2-lane wavefront costs
exactly what widening to 4,096 costs in memory, and none of what it costs in
attention work**, because each 2,048-row chunk still attends only over its own
context. The two levers are not substitutes; they buy different things at the
same price. `guard_wavefront_geometry` refuses a geometry that overruns the
budget and names both the serial and the wavefront projection in the message.

Composing the wavefront with a widened chunk lands on the same razor margin
the 8,192 geometry already sits on: 2 lanes × 4,096 projects **92.53 GB, 1.96
GB spare** — identical to 1 lane × 8,192, which is the arithmetic you would
expect. It fits; it is not comfortable.
`MTPLX_FABLE_PREFILL_QSA_QUERY_TILE` is the escape (it caps the live query
rows, so it caps the per-lane transient), and the same `query_tile` argument
already threads through the wavefront plan.

The unbounded continuous wavefront is the case the guard exists for: 8 lanes
at the shipped 2,048 width projects **99.37 GB** against a 90 GiB limit.
`guard_wavefront_geometry` refuses it and names both projections.

## Exactness

The claim is bit-exactness against the serial control, and it is structural
rather than statistical.

* **Ops per chunk are unchanged.** Every `(k, L)` node runs the same layer
  body on the same input as it does today. No op is added, removed, fused, or
  re-shaped.
* **Order within a chunk is unchanged.** Layers still run 0…47 for each
  chunk.
* **Order across chunks is unchanged where it matters.** The only shared
  mutable state is cache entry `L`, and `(k+1, L)` is issued at step `t+1`
  while `(k, L)` was issued at step `t`. The Python-side read of `cache[L]`
  therefore still happens after the write, and MLX's lazy arrays carry the
  dependency to the device. GDN recurrent state hands off through exactly the
  same in-place `cache[idx] = value` chain it uses today
  (`OwnedRecurrentStateCache.__setitem__`, `mtplx/cache_state.py:3550`).
* **Streams change scheduling, never values.** `mx.stream` annotates
  placement; `mx.async_eval` starts work early. Neither reassociates a
  reduction.

So digests must match the control, and the falsifier asserts it directly
(`numerics(wave_out, serial_out)` must print `0` differing elements). A
nonzero count would mean MLX reassociated something across a stream boundary,
and that alone kills the row — it is a finding, not a rounding allowance.

**This half is already measured.**
`tests/test_fable_prefill_wavefront.py::test_wavefront_tile_is_bit_identical_to_the_serial_tile`
runs the falsifier's own `run_tile` on a tiny Flash-Next config, serial versus
two `mx.new_stream(mx.cpu)` streams, and gets `max|diff| == 0.0` with `0`
differing elements across both outputs and all four cache leaves. Two CPU
streams are not two Metal queues, so this does not settle the GPU case — but
it does settle that the wavefront's cross-stream dependency wiring is correct
and that MLX's stream annotation moves no value. What is left for the GPU run
is purely the performance question.

Note what is *not* claimed: the GDN kernel's own chunk-split invariance is
tolerance-based, not bit-exact
(`tests/test_gdn_blocked_prefill.py::test_blocked_prefill_state_chains_like_stock`
asserts `<= 0.05`). The wavefront does not change the chunk split, so that
tolerance is untouched — but it is why the exactness argument is "same ops,
same order", not "same math, different order".

## Wiring, and why it is not clean

The chunk loop cannot do this on its own. `rt.forward_ar` runs all 48 layers
for one chunk; interleaving two chunks needs a new model-level entry point
that carries two hidden states through one layer loop, offset by one layer.
Concretely, on `Qwen4ExpTextModel`:

```
for L in range(layers + 1):
    if L < layers:      h0 = layers[L](h0, cache=cache[L])      # stream A
    if L >= 1:          h1 = layers[L-1](h1, cache=cache[L-1])  # stream B
```

Everything below is a thing that breaks if you write that loop naively.

1. **PLE staging.** `Qwen4ExpTextModel._forward` calls
   `ple.ple_embedding.stage(inputs, cache[_ple_stage_idx], NGRAM_IDX)` once
   per forward, and the staged rows live on the cache. Two chunks in flight
   means chunk k+1's `stage` would overwrite rows chunk k's PLE layer has not
   consumed yet. Needs per-lane staging slots, or `stage` for lane B deferred
   until step `_ple_stage_idx + 1`.
2. **`self._last_widened`.** Set once per forward and consumed by the MTP
   head. Two lanes clobber it; it has to become per-lane and be read in chunk
   order at the drain.
3. **MTP history append.** `_append_mtp_history` must see chunk k's hidden
   before chunk k+1's. At a drain both are available; append them in order.
4. **Chunk receipts and callbacks.** `_record_prefill_chunk` and
   `chunk_callback` carry per-chunk `wall_s`. Under a wavefront a chunk's wall
   time is not separable — two chunks share the window. Either the receipt
   becomes per-group, or `wall_s` becomes a group time with the group's span.
   Silently reporting a halved per-chunk wall time would corrupt the A/B
   harness that reads `prefill_chunk_records()`.
5. **PLE prefill lookahead.** `MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD` prepares
   chunk k+1's sidecar gather on a host thread while chunk k's forward runs
   (one slot, `PrefillLookahead.submit`/`take`). Under a 2-lane wavefront
   chunk k+1 is already in flight, so the lookahead must run one further
   ahead (prepare k+2) or it stops overlapping anything. The two features
   compose only after that change.

That is five coupled changes across the model forward and the generation
loop, gated on a mechanism question that three independent priors say will
come back negative. **Not clean. No prototype.** The falsifier reports first.

## Failure modes

**Stream priority.** MLX exposes no stream priority and Metal will not
guarantee one queue's progress. The trailing lane (chunk k+1) can starve the
leading lane (chunk k), which lengthens the critical path. Aggregate GPU busy
can go up while TTFT goes *down* — and TTFT is what prefill is judged on. Any
A/B has to read TTFT, not GPU utilization.

**GDN boundary capture — the torn snapshot.** This is the one that is a
correctness bug rather than a performance disappointment.
`_capture_gdn_boundary` (`mtplx/generation.py:3911`) snapshots the *whole*
cache at a chunk end, and `snapshot_untrimmable_cache` clones every recurrent
entry. Inside a wavefront group, when chunk k finishes layer 47, chunk k+1 has
already run layers 0…46 — so cache entries 0…46 hold chunk k+1's state and
47 holds chunk k's. A snapshot there is a state no forward ever produced, and
a warm restore from it resumes from garbage. The grouped schedule exists
precisely so that capture happens only at drains, where every cache entry is
at the same token count.
`assert_boundary_capture_compatible` refuses the non-draining combination.

The cost is stated, not discovered: boundary records go from one per chunk to
one per group. At 8 chunks and 2 lanes with a solo tail chunk that is 8 → 5
(`boundary_records_per_prompt`). `MTPLX_FABLE_PREFILL_WAVEFRONT_TAIL_SOLO`
(default on) keeps the last chunk unpaired so the boundary tail grid
(`MTPLX_GDN_BOUNDARY_TAIL_INTERVAL`, default 256) keeps its resolution near
the prompt tail, which is where warm restores land. Generated tokens stay
bit-exact either way; what changes is warm-restore granularity.

**Host build doubling.** MLX streams are thread-local, so both lanes' graphs
are built by one Python thread. Every wavefront step costs the host two layer
builds instead of one. If prefill is host-bound — which
`mtplx/prefill_rungs.py` and the PLE lookahead's 2,313 ms of measured
host-late GPU idle both suggest — the wavefront pays double host cost for a
GPU overlap the GPU may not have room for. The falsifier reports `build_ms`
separately from `eval_ms` for exactly this reason.

## What would close this row

* **NO-GO** if the falsifier's `independent` arm shows no concurrency, or the
  wavefront arm misses 15%. Write the number into this file, delete nothing,
  and take the `MTPLX_PREFILL_ASYNC_RUNGS` port instead.
* **GO** needs the falsifier to clear both gates, then a re-run with
  `--no-share-moe --allow-large` to prove the shared expert bank was not doing
  the work, then the five wiring items above, then a real TTFT A/B on the
  16,384/1,024 cell with prefill digests compared against the control.

## Files

| file | what |
| --- | --- |
| `scripts/fable/micro_two_stream_prefill.py` | the falsifier (not run) |
| `mtplx/fable_prefill_wavefront.py` | schedule, memory model, refusals — nothing imports it from the serving path |
| `mtplx/fable_prefill_chunk.py` | `plan_prefill_chunk_memory(live_lanes=...)`, default 1 |
| `tests/test_fable_prefill_wavefront.py` | pure arithmetic + a CPU-stream wiring proof |
