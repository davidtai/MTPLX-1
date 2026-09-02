# W62 — the decode-cycle PLE boundary: timeline, ranking, and what no host lever can reach

Branch `worker/w62-ple-boundary` (from `worker/w56-ple-candidate-prefetch`, which already
carries `worker/w46-first-ple-gather`), merged with `experiments/fable-qwen38-80tps`.
Runtime only. Flag: `MTPLX_FABLE_PLE_BOUNDARY`, items via `MTPLX_FABLE_PLE_BOUNDARY_ITEMS`,
probe width via `MTPLX_FABLE_PLE_BOUNDARY_PROBE_ROWS`. All default off.

## 1. Where the 4.002 ms/cycle actually goes

Reduced in one streaming pass over `.benchmark-artifacts/pr391/current-exact-early-d3-census-2410.jsonl`
(383 cycles, window ops 59,700–2,128,337) with `census_retained_stack.py`'s own definitions —
the 10 µs gap floor, the union-busy timeline, the lm_head cycle marker, and `is_ple_boundary`
verbatim. Reproduces the reference exactly: 14.090 s busy / 2.443 s idle, 6.380 ms idle/cycle,
`ple_boundary_ms_per_cycle` 4.002 (reference constant: (851.0 + 681.8)/382 = 4.013).

**The boundary is exactly two gaps, once each, in this order, in 374 of 374 cycles.**

| gap | previous kernel → next kernel | ms/cycle | host-late | driver |
|---|---|---:|---:|---:|
| **A** | `gg1_copyuint32uint32` → `affine_dequantize_bfloat16_t_gs_32_b_4` | 2.222 | 84.1 % | 15.9 % |
| **B** | `affine_dequantize_bfloat16_t_gs_32_b_4` → `gather_frontbfloat16_int32_int_2` | 1.780 | 90.9 % | 9.1 % |
| | **total** | **4.002** | **87.1 %** | 12.9 % |

Split by what the host was doing — (previous GPU end → host starts encoding) + (encode) +
(encode end → GPU start), means over 374 cycles:

| gap | pre-encode host | encode | commit → GPU start | median gap |
|---|---:|---:|---:|---:|
| A | **1890.7 µs** | 22.0 µs | 362.8 µs | 2413 µs |
| B | **1640.8 µs** | 15.7 µs | 166.5 µs | 1717 µs |

`encode_start − prev_gpu_end` was **positive on 374/374 gaps in both families** — the host never
once had the next buffer ready. So the boundary is **3.53 ms/cycle of serial host time**,
0.53 ms/cycle of Metal commit→start latency, and 0.04 ms of encoding. It is not a driver problem
and it is not a GPU problem.

### The cycle, in dispatch-sequence order

Offsets are op-seq offsets from the cycle's target `lm_head` (grid `[1,31040,1]`), which is the
*last* dispatch of the previous verify graph. Median cycle ≈ 5,200 dispatches / 165 command buffers.

| offset | command buffer | GPU | idle before | host state |
|---:|---|---:|---:|---|
| 0 | target `lm_head` | | | |
| 1–467 | accept / commit / GDN keep-replay, 51-op buffers | ~2.3 ms | 0.6–0.8 µs | host **1.3–3.2 ms ahead** |
| 468 | `v_copyuint32int32 → gather_frontuint32…` | 3.8 µs | **465 µs** | first sync (target dist → host), 354 µs host |
| 551 | `gather_front… → gather_front…` | 2.6 µs | **467 µs** | second host round trip, 371 µs host |
| 745 / 1002 / 1262 | FRSpec draft head (grid `[1,8192,1]`), depths 1/2/3 | ~4.7 ms | ~0 interior | |
| 1314 | 26-op buffer ending `gg1_copyuint32uint32` (the D3 ids) | 139 µs | 0.7 µs | |
| **1340** | **PLE q4 dequant — a 1-op buffer** | 12 µs | **2620 µs** | **gap A: 2165 µs host** |
| **1341** | **`gather_frontbfloat16_int32_int_2` — a 3-op buffer** | 2.8 µs | **1677 µs** | **gap B: 1568 µs host** |
| 1344 … 5200 | the compiled verify body, ~3,860 dispatches | ~28 ms | 140 µs then 0.6 µs | host ahead again, and stays ahead |

### What runs in each gap (read off the code, not guessed)

**Gap A** — between the draft chain's last dispatch and `mx.async_eval(compiled_aux, *state_in)`
in `graphbank._forward_installed_fixed_m4`:

1. the D3 host sync (the drafted ids must reach `verify_input`);
2. `_FixedM4SidecarAux.__call__` → `install_owned_rows(pending_warm)` — joins the 16 primary-row
   pread futures;
3. `_ngram_rows_np` on a `[1,4]` window → 64 row ids: ~33 NumPy calls on 6-element arrays, **~50–65 µs**;
4. `gather_np(flat)` → `_SidecarGather._rows_matrices`, hot-LRU branch:
   `np.unique` → 64 dict probes → **`self._warm(miss_np)`, a BLOCKING threaded `os.pread` pass** →
   3 memmap fancy indexes → 48 LRU inserts → 64 `move_to_end` → `_stack_hot_rows` ×3 → `[inverse]` ×3;
5. 3 × `mx.array(...)` + `.view()` + `mx.dequantize` graph build;
6. `mx.async_eval` → 22 µs encode → 363 µs commit→start → 12 µs of GPU.

**Gap B** — `dispatch["fn"](input_ids, compiled_aux, *state_in)`, i.e. the mx.compile'd verify
graph's host-side replay over ~5,200 nodes and ~137 array inputs, plus `_unpack_fixed_m4_outputs`
and the state/capture rebind (~350 Python attribute writes) before `mx.async_eval(*outputs)`.
**No sync happens here** — `async_eval` does not block and the preceding buffer is a 12 µs dequant
the host had already submitted. **Nothing PLE-shaped runs in gap B.** The census family name is
about the two Metal kernels that bracket it, not about what the host was doing.

### (a) page faults / (b) Python-host / (c) GPU→host sync

| class | evidence | ms/cycle |
|---|---|---:|
| **(c) sync + submit latency** | the two clean host round trips in the same cycle (offsets 468, 551) cost 354 and 371 µs of host time for a trivial decision, so that is this box's sync-wake + resubmit floor; gap A contains exactly one such round trip | ~0.35–0.40 of gap A, + 0.53 commit→start over both gaps |
| **(a) page faults / IO syscalls** | `_rows_matrices`'s hot branch preads every missing row before reading it: ~48 misses × 3 maps = **~144 `os.pread`** at W46's measured **5.03 µs of GIL-contended Python each** (32,768 rows × 1 map = 164.8 ms), plus ~24 `pool.submit` + 24 `future.result`. The fancy index behind it is ~13 ns/row warm. | **0.7–1.1** (micro: 0.72–1.03) |
| **(b) Python / NumPy host** | row arithmetic ~50–65 µs; LRU bookkeeping + `_stack_hot_rows` ~0.1–0.2 µs × 192; `mx.array` construction ×3; **and all 1.64 ms of gap B**, which is MLX graph construction | ~0.3 in gap A, **1.64 in gap B** |

The headline: **the single largest identified term in gap A is a pread warm pass that a
page-cache-warm table does not need** — the same inversion `mtplx/ple_row_gather.py` documents for
prefill (`164.8 ms` warm pass vs `0.44 ms` fancy index per 32,768 rows), which the **decode** branch
never got. `warm_decision` is only reachable from the `len(uniq) > _HOT_PATH_MAX_ROWS` branch;
every decode gather takes the hot-LRU branch, which calls `self._warm(miss_np)` unconditionally.

**Answer to the brief's explicit question — "check whether the decode side still does per-row
pread": yes, unconditionally, ~144 syscalls per cycle, blocking, with no residency probe.**

## 2. Ranking

Cost basis: −1 ms/window = +2.7 % (Report M §0). ABBA within-seed noise floor 0.3–0.7 % =
0.11–0.26 ms/window. Micro column is `scripts/fable/micro_ple_boundary.py --self-test`
(synthetic table at production row geometry, warm pages, 120 reps, probe widths 4/8/32) —
**pre-evidence, not the verdict**; the guarded run against the real table is.

| # | lever | mechanism | predicted | micro (synthetic) | exactness | verdict |
|---|---|---|---:|---:|---|---|
| 1 | **`warm_skip`** | `mincore`-gate the decode `_warm` pass; take the fancy index directly when the pages are resident | 0.75–1.15 | **−0.72 / −1.03 / −0.95** | **provable and total** — `_warm` preads into a throwaway buffer and returns `None`; the values come only from `gather_matrices`. Skipping it cannot change a byte. | **BUILT, in the default arm** |
| 2 | **`primary_vectorized`** | the 16 primary rows are read inline as one fancy index instead of 16 pool tasks / 48 GIL-held preads | 0.20–0.40 | **−0.22 / −0.27 / −0.27** | same payload bytes, same LRU insert order, owned copies (`np.array`, never a memmap view) | **BUILT, in the default arm** |
| 3 | `hot_block` | assemble the 3 output matrices with 3 scatters + ~16 row copies instead of `_stack_hot_rows`'s 192 | 0.10–0.20 | **−0.003 / −0.017 / +0.032** → noise | same values, same positions, same LRU order and eviction point | **BUILT, selectable, NOT in the default arm** — it measured below the floor and a candidate should carry only the code whose win it claims |
| 4 | `rows_fast` (pure-Python `_ngram_rows_np`) | replace ~33 small-array NumPy calls with Python ints | ~0.05 | — | would need int64 wraparound emulation (`mult` products overflow), so ~200–300 masked Python int ops ≈ 30–50 µs vs ~50–65 µs | **NOT BUILT** — a wash by arithmetic; the EOS-free constant-mask variant saves ~0.02 ms, an order below the floor |
| 5 | `MTPLX_COMPILED_VERIFY_BOUNDARY=post` (no code) | drop the separate aux `async_eval` so gaps A and B merge and one 363 µs commit→start disappears | ~0.17–0.36 | — | **UNSAFE as written**: `_prepare_compiled_verify_aux`'s own docstring says the aux must cross the materialization boundary before becoming an `mx.compile` input, "otherwise MLX sees the row graph's cache leaves as uncaptured inputs to the verifier graph". The separate dequant buffer is structural. | **REJECTED, documented** |
| 6 | bigger pread chunks on the cold fallback | `step = max(1, min(64, (n+31)//32))` gives 24 tasks for 48 rows | 0.2–0.4, **cold path only** | — | page warming only | **NOT BUILT** — W46 measured the syscall count, not the task count, as the cost (164.8 ms at 512 tasks vs 175.6 at 16); the cold path is not the production regime (`--prewarm-ngram-table`) |

`warm_skip` + `primary_vectorized` = **−0.94 to −1.30 ms/cycle** on the micro, i.e. **+2.5 to +3.5 %**
of a 37.4 ms window, against the 4.002 ms/cycle the census puts on the boundary. Probe width 8 is
the default: at 4 and 8 the probe is ~23/46 µs, at 32 it costs ~0.25 ms and eats a quarter of the win.

### Composition with W56

`MTPLX_FABLE_PLE_CANDIDATE_PREFETCH` moves the *whole* gather to a worker thread when its buffer
hits; `warm_skip` makes the gather cheap when it misses, and W56's own `_fill` already takes the
`vectorized` path on a warm table, so the two do not overlap. `primary_vectorized` sits underneath
both — the candidate aux uses the same `submit_warm` / `install_owned_rows` pair. Both flags are
armable in one process (tested).

## 3. What NO host lever in this module can reach, and what would

**Gap B's 1.64 ms/cycle is not PLE work and no gather change touches it.** It is the compiled verify
graph's host construction: ~5,200 lazy nodes and ~137 inputs built on the generation thread while
the GPU holds nothing but a finished 12 µs dequant. Three designs would remove it; none is code here.

1. **Split submission (cheapest, and the machinery already exists).** `graphbank.install_fixed_m4_split`
   already compiles a prefix graph (embedding + layer 0) and a suffix graph (layers 1–47), and
   `enqueue_fixed_m4_prefix` / `FixedM4Prefix` / `FixedM4Split` are built but not on the retained
   decode route. Submitting the prefix the moment it is constructed gives the GPU layer 0 to run —
   the first 51-op buffer of the census cycle is 694 µs of GPU — while the host builds the suffix.
   Ceiling: **min(prefix GPU, suffix host build) ≈ 0.7 ms/cycle**, ~1.9 % of the window. Exactness is
   the split lane's own question, and `test_qwen4_fixed_host_tokens_static.py` already carries its
   invariants (currently red on this branch's base, i.e. someone else's lane in flight). **Not mine
   to take over** — flagged for whoever owns PR391's split.
2. **Remove the D3 host sync entirely** (`MTPLX_FABLE_DEVICE_K20`'s stated blocker: "the fixed-M4
   verify needing `host_input_ids`"). If the n-gram row ids were computed **on device** from the
   device-resident drafted ids, and the PLE rows selected on device out of a resident
   `[4,20,16,100]` candidate tensor (`docs/perf/ple-candidate-prefetch-phase2.md`), gap A collapses
   to a submit: no sync, no host row arithmetic, no host read. That deletes ~2.2 ms/cycle **and**
   unblocks W42/W28's device-K20 merge. It is a model-path change (a Metal kernel for the n-gram
   hash + a gather), not runtime-only, so it is out of this program's scope as written — but it is
   the only lever that reaches the whole of gap A.
3. **Speculative two-window submission.** Submit the verify window for the *most likely* accepted
   prefix before the D3 sync resolves, and discard it on a miss. Acceptance is α₁ 0.699, so ~30 %
   of windows would be thrown away — 28.5 ms of wasted GPU against 4 ms of saved idle. **Arithmetic
   says no**; recorded so it is not re-proposed.

Also unreachable from here: the **0.53 ms/cycle of commit→start latency** (363 µs on gap A alone,
for a *one-kernel* buffer). That is Metal restarting a drained queue, and the only host-side way to
avoid it is to not drain the queue — which is design 1 again.

## 4. How to run it

### Micro (guarded — reads the real table's pages)

```zsh
W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w62-ple-boundary
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
RG=/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py
PLIST=/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist
mkdir -p $W/.benchmark-artifacts/fable
env PYTHONPATH=$W $PY $RG --plist $PLIST --lock-timeout-seconds 3600 \
    --child-timeout-seconds 900 \
  -- env PYTHONPATH=$W $PY $W/scripts/fable/micro_ple_boundary.py \
       --reps 200 --probe-rows 8 \
       --json $W/.benchmark-artifacts/fable/micro-ple-boundary.json
```

Read the `resident` column first: at ~1.000 the run measured the production (prewarmed) regime and
the `warm_skip` delta is the lever; at ~0.000 it measured a cold table, where `warm_skip` correctly
declines and the delta should be ≈0 minus the probe.

### ABBA — 16K decode, control `--prewarm-ngram-table`, candidate `MTPLX_FABLE_PLE_BOUNDARY=1`

```zsh
W=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w62-ple-boundary
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python
S=/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/1b1e4a52-8af8-4acc-a173-0bf81c785447/scratchpad
mkdir -p $S/w62 $W/.benchmark-artifacts/fable
/bin/zsh $S/w8/retry_guarded.zsh $S/w62/abba.log "env PYTHONPATH=$W $PY" 9000 \
  -- $PY $W/scripts/fable/abba_window.py \
       --sequence 1788400621 --order ABBA \
       --label-prefix fable-w62-ple-boundary \
       --source $W --python $PY \
       --prompt-tokens 16384 --max-tokens 1024 \
       --control-flag=--prewarm-ngram-table \
       --candidate-extra-env MTPLX_FABLE_PLE_BOUNDARY=1
```

`--python $PY` is required: this worktree has no `.venv` of its own, so without it every arm would
be launched with a python that does not exist. `--control-flag=--prewarm-ngram-table` puts the
prewarm on **both** arms (the shared baseline), which is what makes the candidate measure the lever
and not the prewarm — the same shape W46's window used.

`--prompt-tokens 16384` / `--max-tokens 1024` are `abba_window`'s defaults and are written out only
so the cell is unambiguous in the log. The candidate arm's default item set is
`warm_skip,primary_vectorized`; the receipt lands under `ple_boundary` / `ple_boundary_armed` in
each row's PLE hot-row block.

Follow-up arms, only if the first is positive or ambiguous:

```zsh
# attribution: which item carries the delta
--candidate-extra-env MTPLX_FABLE_PLE_BOUNDARY=1 \
--candidate-extra-env MTPLX_FABLE_PLE_BOUNDARY_ITEMS=warm_skip
--candidate-extra-env MTPLX_FABLE_PLE_BOUNDARY_ITEMS=primary_vectorized

# instrument: the host phase split on BOTH arms (levers disarmed), so gap A and
# gap B can be attributed inside the process rather than from the census
--control-extra-env MTPLX_FABLE_PLE_BOUNDARY=1 \
--control-extra-env MTPLX_FABLE_PLE_BOUNDARY_ITEMS=timing \
--candidate-extra-env MTPLX_FABLE_PLE_BOUNDARY=1 \
--candidate-extra-env MTPLX_FABLE_PLE_BOUNDARY_ITEMS=timing
```

## 5. Reading the receipt

`ple_boundary.warm_skipped` vs `warm_taken` is the verdict gate: an arm that claims `warm_skip` and
shows `warm_taken` on every cycle probed a **cold** table, took the shipped pread pass, and its
delta is not the lever. `primary_inline` counts cycles whose 16 primary rows never touched the pool.
`graph_build_ms / graph_build_calls` (under the `timing` item) is gap B's 1.64 ms measured from
inside the process.
