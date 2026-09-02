# fable ABBA harness (Qwen3.8 Flash-Next)

Matched A/B/B/A decode benchmarks on the production cell: 16,384-token coding
prompt, 1,024 output tokens, temperature 1 / top-p 0.95 / top-k 20, reasoning
`xhigh`, seeds `20260829 20260830 20260831`, all inside one guarded GPU window.

| File | What it is |
| --- | --- |
| `abba_driver.py` | One arm. Loads the model once, runs the requested seeds, writes a receipt JSON. |
| `abba_window.py` | The bracket. Runs as the guard's direct child and spawns one `abba_driver.py` per arm. |
| `humaneval_screen.py` | The quality gate. One guarded window = one server on :8091 + full HumanEval pass@1 + a receipt. |

Ported from the reviewed PR391 driver `/private/tmp/pr391_fixed_d3_abba.py`
(SHA-256 `0ae20c7c4028cea83d9b9084d29067925d6dca08ff0ca2ce5a4ea9d73b9bb7d0`).
The `/private/tmp` SHA pin is gone (this is now our own driver); the guard
consumption, memory wait, thermal gate, turbo-profile environment check, and
`reset_run_caches` are all kept.

## Control

Arm A defaults to the retained 67.818 tok/s "paired routed GLU" arm from
`docs/perf/pr391-m4-paired-routed-glu-result.md`
(receipt `.benchmark-artifacts/pr391/rebench3-1788287001-paired-routed-glu-candidate-seeds-16k-1k-seeds-16k-1k.json`):

```
--target-mode batched --require-compiled-verify --m4-stage3 \
--qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare --max-tokens 1024 \
--candidate-env MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE=1 \
--candidate-env MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL=1 \
--candidate-env MTPLX_QWEN4_M4_ROUTED_GLU=1
```

`--control-flag` / `--control-env` / `--control-extra-env` move the shared
baseline: they land on *both* arms. Arm B is then arm A plus whatever you add
with `--candidate-flag` / `--candidate-env` / `--candidate-extra-env`. A
candidate setting that repeats a control key replaces it on arm B rather than
being passed twice.

## 1. Control-only smoke, one seed

Roughly 5-10 minutes of GPU plus the thermal wait. It stops the production
Qwen service and restores it.

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/abba_window.py \
    --sequence 1788400001 \
    --order AB \
    --seed 20260829 \
    --label-prefix fable-smoke
```

`--order AB` with no candidate override runs two identical control arms, which
is the cheapest way to see the harness end to end and to read the run-to-run
noise floor. For a genuinely single-arm smoke, drop `--order AB` for
`--order AB --seed 20260829` and read only arm 0.

Preview the exact arm command lines without touching the GPU:

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/abba_window.py \
    --sequence 1788400001 --dry-run
```

### Measuring a different prompt length

`--prompt-tokens N` resizes the measured cell's prompt on **both** arms.
Accepted values are 1024, 8192, 16384 (default), 32768, 65536, 131072 and
262144. Only the prompt moves: the labels, the receipt paths, the sampler and
the summary table are identical at every length, so a window at another length
is read exactly like a 16K one -- it is just not comparable to a 16K receipt.

The default reproduces the pinned production prompt byte for byte. Every other
value is built to exactly N tokens by `abba_driver.build_exact_coding_prompt_ids`
from the same SHA-pinned fixture pair, which is how the benchmark matrix already
builds its 64K and 128K cells -- so `--prompt-tokens 65536` is the matrix's
`coding-64k-1k-xhigh-t1` prompt. The arm receipt records `prompt_tokens` and
`prompt_fixture_sha256`; `prompt_content_sha256` stays pinned only at 16384,
where the prompt bytes themselves are pinned.

`--require-reference-token-parity` is refused away from 16384 (the PR391
reference rows were recorded against the production prompt), and the window
refuses it while planning, before it takes the GPU lock.

Pair it with `--prefill-only` for a prefill-attribution window: `--max-tokens`
drops to 64 and the driver's unmeasured graph warm-up cell runs first at the
**same** N, so the measured run's first prefill chunk is not the cold one.
Per-chunk wall and PLE-gather seconds land on every row as `prefill_chunks`,
on control and candidate alike.

## 2. Full ABBA, three production seeds

Twelve arms (4 per seed x 3 seeds), each a fresh model load plus a thermal
wait. Budget several hours and give `--child-timeout-seconds` real headroom.

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 36000 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/abba_window.py \
    --sequence 1788400001 \
    --order ABBA \
    --label-prefix fable-abba
```

Outputs land in
`/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/`
(gitignored):

* one `<label>-<sequence>.json` arm receipt per arm,
* `abba-window-<sequence>-<label-prefix>.json` (rows, per-seed paired deltas,
  adjacent-pair deltas, means and medians, the outer command line, every arm's
  argv and exit code),
* `abba-window-<sequence>-<label-prefix>.md` (the markdown table, also printed
  to stdout).

## 3. Passing a candidate

Construction-time MTPLX settings (arm B only):

```
    --candidate-env MTPLX_QWEN4_M4_PLE_PREFIX_REUSE=1
```

Driver flags (arm B only). Argparse eats bare `--flag` values, so **use the
`=` form**:

```
    --candidate-flag=--nax-verify
    --candidate-flag=--frspec-n --candidate-flag=32768
```

Non-MTPLX process environment applied before the mlx import (arm B only):

```
    --candidate-extra-env MLX_MAX_OPS_PER_BUFFER=8
```

`MTPLX_FABLE_*` is the fable namespace and rides this same raw passthrough --
`parse_key_values` exempts it from the `--candidate-env` MTPLX check, so it must
go on `--candidate-extra-env` / `--control-extra-env`:

```
    --candidate-extra-env MTPLX_FABLE_MOE_SORTED=1
```

`MTPLX_FABLE_MOE_SORTED=1` argsorts the forty routed `(row, expert)` pairs by
expert id before the M4 routed gathers, so the ~12 duplicate pairs per layer
cycle land adjacent, then un-permutes the results. It is a pure permutation of
independent M=1 rows: bit-identical, worth 5-8% on the MoE in
`micro_moe_dedup.py` variant `b1`. It only bites on the `_m4_forward` and
routed-down-reduce routes, which still issue `mx.gather_qmm`; the retained
paired-routed-GLU arm gathers inside its Metal kernels, so on that arm the gate
is a no-op and the win needs the kernel-level equivalent.

Three MTPLX settings are *neither* profile overrides *nor* `MTPLX_FABLE_*`:
they are read with a bare `os.environ.get` at their use site, so
`apply_profile_env` refuses them on `--candidate-env` while the raw `--env`
namespace check used to refuse them too -- they were unreachable from the
harness on both channels. They now ride the raw passthrough through a named
allowlist (`abba_driver.RAW_ENV_MTPLX_KEYS`, mirrored in `abba_window.py` so a
mis-routed key fails during planning rather than after an arm takes the GPU
lock):

| Key | Read by | Default |
| --- | --- | --- |
| `MTPLX_CONTEXT_COPY_K` | `mtplx/context_copy.py:context_copy_block_k()` | 24 |
| `MTPLX_CONTEXT_COPY_PROBATION_K` | `mtplx/context_copy.py:context_copy_probation_k()` | 8 |
| `MTPLX_SESSION_BANK_MAX_BYTES` | `mtplx/engine_session.py` | model-aware auto |

Every *other* `MTPLX_*` key on `--*-extra-env` still fails loudly, and the two
above are refused on `--*-env` with a message naming the right channel. Both
are recorded in the arm receipt under `process_environment_overrides`, with the
requested value alongside the value actually in force.

### Recipe: start the first prefill chunk's PLE gather at request arrival

```
    --candidate-extra-env MTPLX_FABLE_PLE_FIRST_GATHER_EARLY=1
```

`MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1` hides chunks 2..n of the PLE n-gram
gather behind the previous chunk's forward. Chunk 1 has nothing to hide behind:
the scope submits it and `stage()` blocks on it microseconds later, which the
w35 receipts price at `prefill_chunks[0].ple_gather_s` 0.627 s against 0.0006 s
for chunks 2-4. On a single-chunk prompt the lookahead is inert by
construction, so that exposed gather is the whole PLE prefill cost.

This flag does three things, all off without it:

1. **Starts chunk 1's gather at request arrival** -- at the top of
   `restore_or_prefill_prompt_state`, before the session-bank lookup and the
   prefill graph setup -- on a process-wide worker pool. The chunked prefill
   then *adopts* that in-flight future as the lookahead's slot 0 rather than
   preparing the span twice; a single-chunk prefill consumes it directly. Row
   ids are a pure function of the prompt ids, so the first chunk's span is
   predicted from the prompt and the two span planners the prefill loop
   chooses between -- and the lane declines rather than guesses when they
   disagree (a banked short prompt, a stable-prefix edge).
2. **Replaces the per-row `os.pread` warm pass with the memmap fancy index**
   for every big gather, on the worker and on the generation thread alike, when
   `mincore(2)` says the rows' pages are already in core (~165 ms per 32,768
   rows against 0.44 ms). A demand-faulted mmap is flat at 1.40 GiB/s against
   pooled pread's 12.9, so the probe is what makes this safe: an unavailable or
   below-threshold probe takes the shipped pread path, which is never wrong.
   `vectorized_gathers` / `pread_gathers` in the receipt say which ran.
3. **Pre-touches the rest of the prompt's rows** behind chunk 1, chained off
   its future, so later chunks' gathers find their pages warm.

The receipt records `ple_first_gather_early`:
`started_at_ms_before_layer2` (the head start the lane bought -- submit to
first need), `rows`, `path`, `outcome` (`adopted_hit`, `hit`, or a named miss)
and `prefetch_rest_rows`.

Two companion knobs, also `MTPLX_FABLE_*` and also off by default:

| Key | Effect |
| --- | --- |
| `MTPLX_FABLE_NGRAM_MADVISE=random\|normal\|sequential` | Overrides the n-gram maps' `madvise`. Default is `random` (shipped) with the lane off and `normal` with it on: `MADV_RANDOM` suppresses readahead around a *mapping fault*, which under this lane is the ascending pre-touch and the vectorised gather's residual misses -- the two cases readahead helps. `pread(2)` never consulted the advice at all. |
| `MTPLX_NGRAM_PREWARM=auto\|all\|off\|<GiB>` | How much of the n-gram table to pre-read at model load. **On by default in `auto`** = `min(table, free - KV reservation - 6 GiB margin)`; `all` reads all 29.8 GiB (~2.5 s at ~12 GiB/s), `off` serves at the as-found page-cache rate. First-class option: `mtplx serve --ngram-prewarm ...`, and the CLI wins over the env. `--ngram-prewarm-order` / `<model>/ngram-hotness.npy` (built by `scripts/fable/ngram_row_hotness.py`) decides WHICH rows a partial budget warms. `MTPLX_FABLE_NGRAM_PREWARM_AT_LOAD` is a deprecated boolean alias. See `docs/server.md`. The driver's `--prewarm-ngram-table` did this for benchmarks only; the daemon had no equivalent -- a 1.9 s vs 4.4 s first chunk in the w22 window, and 56 vs 68.8 tok/s on decode. |

Prefill-only ABBA window at 16K, on top of the retained prefill stack:

```
PYTHONPATH=<worktree> <venv>/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --child-timeout-seconds 36000 -- \
  <venv>/bin/python <worktree>/scripts/fable/abba_window.py \
    --sequence <seq> --order ABBA --label-prefix fable-w46-firstgather \
    --source <worktree> --prefill-only \
    --control-env MTPLX_GDN_BLOCKED_PREFILL=1 \
    --control-env MTPLX_PREFILL_CHUNK_SIZE=4096 \
    --control-env MTPLX_QSA_PREFILL_COMPILE_ROWS=4096 \
    --control-env MTPLX_QSA_PREFILL_DEBUG=1 \
    --control-extra-env MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1 \
    --control-extra-env MTPLX_FABLE_PREFILL_QSA_QUERY_TILE=2048 \
    --candidate-extra-env MTPLX_FABLE_PLE_FIRST_GATHER_EARLY=1
```

Add `--prompt-tokens 1024` for the short-prompt cell, where the lookahead is
inert and this lane is the only thing hiding the gather.

### Recipe: prefetch the draft chain's candidate PLE rows (K-P1)

```
    --candidate-extra-env MTPLX_FABLE_PLE_CANDIDATE_PREFETCH=1
```

Decode reads 16 n-gram rows x 100 B per token out of the 32 GB memory-mapped
table.  The rows are 85-93 % novel so no LRU covers them, and the table cannot
go resident next to ~85 GB of wired weights.  This lane predicts instead: the
row ids are a pure function of the token ids, so the moment a draft depth's
K20 support exists on the host, the 16 rows for each of its 20 candidates are
known (320 rows = 32 KB/depth) and a worker gathers them while the GPU runs
the next depth.  When the sampled token is chosen its rows are already in a
host buffer, and `_FixedM4CandidateSidecarAux` hands them to `gather_np(prepared=...)`
instead of reading the table on the critical path.

Bytes are identical by construction, not by comparison: the buffer is
content-addressed by table row id and every payload in it is a literal read of
the same three memmaps the shipped gather reads.  A window with one uncovered
row falls back whole.

Read the receipt's `ple_hot_rows.ple_candidate_prefetch`:

| field | means |
| --- | --- |
| `depths` | candidate buckets submitted (primary + one per draft depth) |
| `candidate_rows` / `bytes` | rows and bytes the lane actually read |
| `hits` / `misses` | verify WINDOWS served from the buffer / fell back whole |
| `rows_served` / `rows_missing` | the row detail behind those |
| `vectorized_buckets` / `pread_buckets` / `cold_declines` | which read the worker took |
| `worker_wait_ms` | owner time blocked joining the workers |

`worker_wait_ms` is the honest half of the receipt: if it is not far below the
gather it replaced, the lane bought nothing.  `cold_declines` is the other
one -- a 320-row bucket on cold pages is 960 GIL-contended `os.pread` calls
(~4.8 ms) against a ~12 us warm fancy index, so a cold bucket is DECLINED and
the shipped gather runs.  A run with `cold_declines` high has no candidate
lane, whatever the flag says.

Two caveats worth stating before the window runs:

* The retained PR391 joint-D3 core keeps its per-depth K20 arrays
  verifier-resident (only the packed token vector is synced), so on that lane
  the *speculative* hook has nothing to read.  The lane still runs there, with
  the three resolved tokens as one-candidate depths, which only moves the read
  into the pool -- a smaller win than the speculative form.  The stock serial
  draft loop is where the 20-candidate hook fires.
* Removing the draft -> target sync itself is phase 2 and is NOT built; see
  `docs/perf/ple-candidate-prefetch-phase2.md`.

Full ABBA, prewarmed table on both arms (the regime the retained stack serves):

```
PYTHONPATH=<worktree> <venv>/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --child-timeout-seconds 36000 -- \
  <venv>/bin/python <worktree>/scripts/fable/abba_window.py \
    --sequence <seq> --order ABBA --label-prefix fable-w56-candprefetch \
    --source <worktree> --prewarm-ngram-table --warm-graph --retain-events \
    --candidate-extra-env MTPLX_FABLE_PLE_CANDIDATE_PREFETCH=1
```

And the as-found page-cache regime, where the lane should matter most -- the
same window with `--prewarm-ngram-table` dropped from BOTH arms, so neither
arm gets the driver's 29.8 GiB pre-read and the control pays real faults on
the critical path:

```
    ... --warm-graph --retain-events \
    --candidate-extra-env MTPLX_FABLE_PLE_CANDIDATE_PREFETCH=1
```

`MTPLX_NGRAM_PREWARM` is on by default at model *load*, and it is a profile
override key (`mtplx/profiles.py`), so the as-found cell needs it turned off on
BOTH arms through `--control-env` / `--candidate-env`, not `--extra-env`:

```
    --control-env MTPLX_NGRAM_PREWARM=0 --candidate-env MTPLX_NGRAM_PREWARM=0 \
    --candidate-extra-env MTPLX_FABLE_PLE_CANDIDATE_PREFETCH=1
```

Without that, both arms are prewarmed by the load path and the two cells
measure the same thing.  Expect `cold_declines` to dominate in the cold cell
unless the lane is paired with `MTPLX_FABLE_PLE_FIRST_GATHER_EARLY=1` on both
arms, which is what warms the pages the candidate buckets then read -- so the
cold cell is really a THREE-way question (cold control, cold candidate, cold
candidate + pre-touch) and the third arm is the one that can win.

### Recipe: open the context-copy block cap

```
    --candidate-extra-env MTPLX_CONTEXT_COPY_K=48
```

Context-copy (prompt-lookup) decoding is on by default and already supplies
~9.4% of the output. `MTPLX_CONTEXT_COPY_K` caps how many verbatim tokens a
round may propose; the confidence ladder (8/12/16/24/32) picks a rung under
that cap, and `MTPLX_CONTEXT_COPY_PROBATION_K` (default 8) holds rounds short
until the acceptance EMA proves the content pays. Six of twenty-one production
rounds were cut by the cap with 4-8 more verbatim tokens still matching the
prompt -- roughly 13 tokens per run left on the table, worth about +1% once
those rows are cheaper than the M4 windows they replace.

**K=48 and K=32 are the same experiment.** `block_for_ext` picks a rung of the
ladder and only *then* clamps to the cap, so the cap binds only below the top
rung: any K above 32 proposes exactly the blocks K=32 does. Against the default
24 it is still a real change -- it unlocks that top rung, which is the 4-8
tokens the cap was cutting off the strongest matches -- but do not expect
48-token blocks. For those you need RAMP (`MTPLX_RAMP_ENABLED=1
MTPLX_RAMP_BLOCK=48`), which replaces the ladder with a fixed length and is a
different, larger change.

**Raising it is legal on the fixed-M4 lane.** K never enters the physical-M4
graph: `install_fixed_m4` keys on `(4, hidden_variant, route_key, aux_contract)`
and hard-codes four rows everywhere. The copy round is a *separate* forward, so
changing K changes that forward's width and nothing about the M4 window.

**With `MTPLX_FABLE_COMPILED_COPY_ROUND=1` it also sets the compiled copy
round's traced width**, to `1 + copy_round_max_block(K)` -- 33 rows at K=32 *or*
K=48, not 49, precisely because of the ladder ceiling above. Sizing the graph
from the raw K instead would pad every round out to 49 rows for a block that
can never exceed 33: sixteen dead rows of MoE traffic per round, more than
compiling the round is worth. Two consequences still worth planning for: the
reserved KV window per round grows with the width
(`_transition_fixed_m4_generation` runs at install, so the cost lands in setup
rather than mid-decode), and every round pays the full width whatever rung the
ladder drew -- padding is free of *correctness* cost but not of *time*, so a
wide graph with a low hit rate is a worse trade than a wide one with a high hit
rate. Judge it on `ms_per_m4_window_net` with a re-fitted
`--copy-token-cost-s`, never on tok/s.

Pair it with `--candidate-extra-env MTPLX_CONTEXT_COPY_PROBATION_K=16` to let
proven lanes reach the wider cap sooner; run six seeds, because the effect sits
near the corrected metric's 0.3-0.7% noise floor.

### Recipe: fuse the dense QSA prefill attention

```
    --candidate-extra-env MTPLX_FABLE_PREFILL_MASK_FUSE=1 \
    --candidate-env MTPLX_QSA_PREFILL_DEBUG=1
```

At `head_dim` 256 MLX's own heuristic declines the fused `steel_attention`
kernel, so every dense QSA prefill layer materializes an `[H, S, T]` bf16
score tensor, masks it, softmaxes it twice and re-reads it for `P@V`. The
flag passes `force_fused=True` instead. Two arms, one flag, both counted by
`MTPLX_QSA_PREFILL_DEBUG=1`:

* `mask_fuse_causal` — the indexer returned **no** selection, which happens
  exactly when the chunk's post-update context is inside the block budget
  (`T <= (block_topk + 1) * ratio - 1` = **2,051** tokens on the production
  pack) or on a vision request. The visible set is then exactly causal, so
  the lane passes MLX the string `"causal"` (documented lower-right aligned:
  the last query is the last key, which is precisely the chunked-prefill
  offset case) and builds no tensor at all.
* `mask_fuse_bool` — a real top-k selection, handed to the fused kernel as
  the bool array it already is.
* `mask_fuse_unavailable` — the one-shot probe found no fused kernel for
  this `(mask kind, head_dim, dtype)`; the arm says so on stderr and stays
  on the dense route for the rest of the process instead of quietly
  measuring the control under a candidate label.

`mask_causal_eligible` counts the exactly-causal regime whether or not the
flag is armed. **At the retained prefill width (4,096) it is zero at 16K and
at 32K** — no chunk's context is under 2,051 — so at those cells the flag's
whole effect is `mask_fuse_bool`, and `mask_fuse_causal` is expected to read
0. It is chunk 0 and only chunk 0 at the shipped 2,048 width.

Neither arm is bit-identical: the fused kernel runs an fp32 online softmax
over the same visible set, so this is the rounding class the long-prompt
agreement screen gates, not an approximation.

The sparse-QSA crossover is a documented knob and is **not** touched by any
of this: `MTPLX_QSA_PREFILL_MIN_CONTEXT` (default 32,768, floor 2,049,
compared against `total_tokens - rows`) and
`MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT` are both in
`MODEL_RUNTIME_ENV_OVERRIDE_KEYS`, i.e. reachable from `--candidate-env`.
At 16K and at 32K the gate is unreachable at any chunk width (its maximum
`total_tokens - rows` is 30,720 at 32K), which is why both cells are wholly
dense and why the mask-fuse arm applies to every chunk of them.

`--candidate-env` keys must be members of
`mtplx.profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS`, or `apply_profile_env`
refuses the arm with `runtime_env_overrides has unsupported key: <KEY>`. That
is the intended fail-closed behaviour: a knob the profile does not know about
would otherwise be silently stomped back to the profile default and the arm
would measure the control twice. If you add a new construction-time knob to
`mtplx/`, add it to that set in the same commit. (For example
`MTPLX_QWEN4_M4_PLE_PREFIX_REUSE` is *not* in the set on this branch, so it
cannot be driven from `--candidate-env` here yet.) The same applies to driver
flags that set an unsupported key: `--lazy-d3`, `--fixed-d3-step`,
`--compiled-draft-tail`, `--compiled-mtp-mlp`, `--pipelined-mtp-hidden` and
friends were carried over from the reviewed driver and will raise on this
branch until their keys are added to the set.

Another agent commits to this branch while you work. `abba_driver.py` refuses a
dirty source tree, which is the right default for a benchmark, but it means an
in-flight edit elsewhere in the worktree aborts the arm. Pass
`--allow-dirty-source` (the dirty entries are recorded in the receipt) or
`--expected-file <path>=<sha256>` if you need to run against a working tree.

Move the shared baseline for both arms with `--control-env` / `--control-flag`
/ `--control-extra-env`; all three land on arm A *and* arm B, so what the
bracket measures stays the candidate-only difference. Flags the window supplies
itself (`--label`, `--sequence`, `--seed`, `--receipt-path`, `--guard-mode`,
`--source`, `--expected-source`, `--candidate-env`, `--env`,
`--thermal-gate-max-c`) are rejected as arm flags.

Other window options worth knowing:

| Flag | Effect |
| --- | --- |
| `--thermal-gate-max-c 45` | Moves the gate threshold. **There is no flag that disables the gate.** Default 40 C, recorded in every receipt. |
| `--prewarm-ngram-table` | Reads `ngram-table.safetensors` (32 GB) sequentially once before the timed cells; records seconds and bytes and stamps `page_cache_regime: "prewarmed"`. Default is `"as-found"`. |
| `--retain-events` | Sets `MTPLX_DROP_EVENTS=0` so per-cycle arrays (step, accepted, attributed ms) reach the receipt. Costs host time per cycle, so it is off by default and the arrays are marked unavailable rather than fabricated. |
| `--d3-softfloat64-route` | Installs the PR391 exact Metal softfloat64 D3 selector/verifier route. **Off by default** -- see the note below. |
| `--require-reference-token-parity` | Turns the recorded token-digest comparison against the PR391 reference rows into a hard failure. |
| `--expected-source <sha>` | Pins the source commit. Without it the observed HEAD is recorded, and a dirty tree is refused unless you pass `--allow-dirty-source` or `--expected-file`. |

### The D3 softfloat64 route is off by default

`scripts/pr391_metal_choice_benchmark_launcher.transform_metal_choice_driver`
injects the exact softfloat64 selector/verifier-decision route into the
reviewed driver's source text. `abba_driver.py` **ports that injected code
directly** and imports every non-trivial object it uses
(`prebind_softfloat64_choice_kernel`, `prewarm_softfloat64_verifier_decision`,
`PR391DirectSoftFloat64D3Route`, `validate_metal_choice_receipt`,
`build_exact_output_parity_receipt`, `build_hit_miss_receipt`) from that same
module, so the route and its fail-closed receipts are the same code rather
than a copy. Re-running the text transform was rejected because it also
replaces the thermal gate with `{"disabled": True}`.

The route is opt-in because **the retained control arm did not use it**. The
67.818 tok/s receipt has `candidate_files: {}` and its measured row carries no
`metal_softfloat64_choice_route` key; the arms that did use it
(`rebench3-17882610*`) landed around 64.2-64.4 tok/s. Turning it on by default
would mean the control could not reproduce its own retained number.

## 4. Restoration checks after the window

`run_guarded.py` restores the `com.tea.qwen` service itself. Confirm it before
walking away:

```
launchctl print gui/501/com.tea.qwen | head -20

curl -s --max-time 10 http://127.0.0.1:8080/v1/models

curl -s --max-time 20 http://127.0.0.1:8080/health

curl -s --max-time 120 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx-flash-next-optimized-speed","messages":[{"role":"user","content":"Reply with the single word READY."}],"max_tokens":16,"temperature":0}'

ls -l /tmp/mtplx-gpu-exclusive.lock && \
  /usr/sbin/lsof /tmp/mtplx-gpu-exclusive.lock || echo "lock has no owner"
```

Healthy means: `/v1/models` lists exactly `mtplx-flash-next-optimized-speed`,
`/health` reports `warmup.background.state == "done"` with
`active_requests == 0`, the chat call returns `READY`, and the flock has no
owner.

## Op diet: per-item A/B and the microbench

`MTPLX_FABLE_OPDIET=1` arms four independently written, bitwise-exact
rewrites of the compiled fixed-M4 verify graph.

Dispatch count is not GPU time: a rewrite can swap contiguous vectorized
kernels for broadcast/general ones and lose more than it saves. The first
shipped `bank` spelling did exactly that — it removed two dispatches per QSA
layer and gave most of the win back to a general strided select. The 3-seed
A/B of that set was neutral (-2.9 / +0.2 / +0.1%). `micro_opdiet.py` then
priced the items separately and `bank` moved to the `rowsel` spelling, which
issues *more* dispatches and runs twice as fast.

`MTPLX_FABLE_OPDIET_ITEMS` selects which rewrites are live, so a result can
be attributed to one item instead of the whole flag. Unset means all four.

| item | what it changes |
| --- | --- |
| `bank` | QSA fixed pooled-bank conditional write (`_extend_pooled_fixed`) |
| `rope` | half-width RoPE tables, shared per forward, split-half rotation |
| `resid` | hyper-connection residual write fused into one kernel |
| `k20` | eager K20 target/draft support (fused deterministic+ordered pair) |

`MTPLX_FABLE_*` is the diagnostic namespace, so both ride the raw
passthrough, **not** `--candidate-env`:

```
    --candidate-extra-env MTPLX_FABLE_OPDIET=1 \
    --candidate-extra-env MTPLX_FABLE_OPDIET_ITEMS=resid
```

An unknown item name raises at import rather than being dropped — a typo that
silently disabled the item under test would make the arm measure the control
twice.

### micro_opdiet.py

Before spending an ABBA window per item, price the three compiled-graph
rewrites directly. `micro_opdiet.py` times one verify cycle's worth of each
(12 QSA layers for `bank`/`rope`, 96 sites for `resid`) at the production
shapes, stock vs rewrite, **both eager and under `mx.compile`** — the compiled
lane is the one that matters, because the real verify step is one compiled
graph and MLX only fuses elementwise chains under compile. It prints per
variant: median/p10/p90 eval ms, us per layer/site, delta% vs stock, the
dispatch count of the built graph, and max-abs-diff against stock (every
shipped rewrite must print 0).

Measured 2026-09-01, compiled lane, per verify cycle:

| family | stock | rewrite | |
| --- | --- | --- | --- |
| bank | 0.492 ms | `bank_select` 0.392 (-20%) | `bank_rowsel` 0.253 (**-49%**, shipped) |
| rope | 1.068 ms | `rope_half` 0.738 (**-31%**, shipped) | |
| resid | 0.395 ms | `resid_fused` 0.366 (**-7%**, shipped) | |

`bank_select` is kept in the bench as the rejected alternative: it issues the
fewest dispatches of the three and is the slowest rewrite, which is the whole
reason this bench exists.

~30 s of GPU. Still a guarded window: it issues Metal work.

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/agent-a228fc55ff545c481 \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 900 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/agent-a228fc55ff545c481/scripts/fable/micro_opdiet.py \
    --out /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/micro-opdiet.json
```

(`PYTHONPATH` and the script path point at whichever checkout carries the
branch; the script itself imports no `mtplx`.)

Do **not** read `--donatable-bank` as the production number: it threads one
bank through all 12 updates so `mx.slice_update` can donate its input, which
the real graph — where the verifier bank holds every pooled leaf — cannot do.
It exists only to show how much of the stock spelling's cost is that copy.

## PYTHONPATH

Always export
`PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps`
on the wrapper invocation: the editable install can otherwise resolve `mtplx`
from a different worktree. `run_guarded.py` still loads its own guard from the
`eval-hy3-q2-2p6bit` worktree because it inserts that path at `sys.path[0]`
before importing `mtplx.qwen_guard`, so the two do not fight.

`abba_driver.py` asserts `mtplx.__file__` lives under `--source` (this
worktree) before it loads the model, and fails loudly with the resolved path
if it does not.

## Guard chain

```
run_guarded.py                 flock + stop com.tea.qwen + one-shot attestation pipe
  └── abba_window.py           consumes the pipe once -> publishes a 0400 window receipt
        ├── abba_driver.py     re-verifies the receipt (ancestry + live flock) before mlx
        ├── abba_driver.py
        └── ...
```

The window receipt mechanism is `scripts/deepseek_v4_guard_window.py`, reused
verbatim; it travels to the arms in `MTPLX_DSV4_GUARD_WINDOW_PATH` and
`MTPLX_DSV4_GUARD_WINDOW_SHA256`. Arms run with `--guard-mode window`;
`abba_driver.py` also accepts `--guard-mode attestation` if you ever want to
run a single arm as the guard's direct child.

## Microbenchmarks

`micro_dispatch_overhead.py`, `micro_moe_dedup.py`, `micro_expert_major.py`,
`micro_hc_read.py`, `micro_dependent_launch.py` and `micro_route_kernel.py`
price one site at the fixed-M4 verifier's shapes without loading the model. They import MLX and therefore need
the SAME guarded window as an ABBA arm; none of them touch `com.tea.qwen`, so
they can share a window.

`micro_hc_read.py` prices the gated-residual read (`GatedResidual.__call__`,
97 reads/cycle, 11 dispatches and 13.21 MB of bf16 mix weights each). Variant
`a` is the eager chain the compiled verifier runs today, `b` is
`kernels/qwen4_m4_hyper_read.py` (`MTPLX_FABLE_HC_M4`), `c` is the existing
`(1024, S, 1)` `fused_hyper_read` for reference. `bn`/`bd` are stage prefixes
of `b`, so the printed down/up GB/s are differences of measured cycles.

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --child-timeout-seconds 1800 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/micro_hc_read.py \
    --rows 4 --calls 97 --reps 20 \
    --sweep 4:256:8,4:512:8,6:256:8,12:256:8,4:256:16 \
    --out /tmp/micro-hc-read.json
```

Adoption bar: `b` at least 40% under `a` in ms/cycle, `down_gbps` >= 500, and
the numerics block a rounding-only class (the kernel docstring names the three
sources and why bit-equality is not reachable). Then confirm on the verifier
with an ABBA arm carrying `--candidate-env MTPLX_FABLE_HC_M4=1`; the gate is
acceptance parity, not a digest. Quality is a separate window: see the
HumanEval quality screen below.

### micro_route_kernel.py

Prices the MoE routing head (census item 4 plus the two GEMVs around it):
`block.gate` q8 GEMV, precise softmax, `argpartition` top-10,
`take_along_axis`, the bf16 renormalise, the shared-gate GEMV and its sigmoid
-- **ten dispatches per layer, 480 per cycle, for 44 numbers and 40 indices**.
`mtplx/kernels/qwen4_m4_route.py` emits the same
`(expert_ids, route_scores, shared_factor)` tuple in two.

Arms: `stock` (the shipped head), `k1` (the kernel at MLX's own `qmv_wide`
thread layout, 4,160 threads), `k4` (each verifier vector on its own lane
octet, 16,640 threads -- bit-identical to `k1` by construction, since the
per-vector accumulation sequence is untouched). The script is a `VEC_LANES`
decision, not a go/no-go: the -8 dispatches/layer land either way.

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --child-timeout-seconds 3600 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/micro_route_kernel.py \
    --layers 48 --reps 200 \
    --out /tmp/micro-route-kernel.json
```

Adoption bar: **parity must print `EXACT` on every arm** -- the routing head
decides which experts run, so `set_diff`, `order_diff`, `ids`, `scores` and
`shared` must all be 0, and the script exits non-zero otherwise. With that
clean, arm the winning `VEC_LANES` on the verifier through an ABBA window
carrying `--candidate-env MTPLX_QWEN4_M4_ROUTED_GLU=1
--candidate-env MTPLX_FABLE_ROUTE_KERNEL=1` (the route kernel replaces the
paired routed-GLU lane's head and refuses to install without it). Because the
tuple is bit-exact there is no quality screen: acceptance and the digest are
unchanged by construction, and `install_qwen4_m4_stage3` proves that per layer
on the real packs before the first token.

## HumanEval quality screen (non-bit-exact kernels)

`humaneval_screen.py` is the quality half of the harness. ABBA measures speed;
a candidate whose numerics are a rounding class away from the eager chain
(`MTPLX_FABLE_HC_M4=1` is the first — see the NUMERICS section of
`mtplx/kernels/qwen4_m4_hyper_read.py`) cannot be gated on an output digest,
so the gate is David's rule instead: the **full HumanEval, 164 problems**,
pass@1 looking decent and within noise of the control. `--n 20` is a smoke,
never a verdict.

One guarded window per arm. The child starts ONE MTPLX server on **:8091**
from this worktree's venv (never :8080 — the guard has that one stopped and
restores it on exit), waits for `/health` + background warmup + a READY chat,
generates greedily, stops the server, and only then scores on CPU.

Control (no `--env`):

```
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/humaneval_screen.py \
    --label control --n 164 --port 8091
```

Candidate (`MTPLX_FABLE_HC_M4=1`):

```
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/humaneval_screen.py \
    --label hc-m4 --n 164 --port 8091 --env MTPLX_FABLE_HC_M4=1
```

`--dry-run` prints both the outer command and the server argv without touching
the GPU. Run the two arms as two windows and compare the receipts; the two
`--env` sets are the ONLY difference between them (there is a test for that).

Expected runtime, per arm: model load + warmup ~5-10 min, 164 greedy
completions at ~70 tok/s ~10-20 min, CPU scoring ~1 min. Budget 25-45 min and
keep `--child-timeout-seconds 5400`. `--n 20` is roughly 8-12 min, almost all
of it model load.

### Sampler

Greedy, `temperature 0`, `n=1` — the pass@1 standard, and this lane accepts it.
The only temperature-1 requirement in the tree is the PR391 float32 D3 core
(`generation.py:_pr391_make_float32_d3_core`), which is opt-in through
`abba_driver.py --d3-softfloat64-route` and is not reachable from the server.
The server's own MTP path handles `temperature <= 0` throughout and even
couples the draft sampler to greedy (`_couple_draft_sampler_to_greedy_target`),
and the fixed-M4 verifier still runs its 4-row verify, so the kernel under test
is exercised either way. Reasoning is off (`--reasoning-mode off`, plus
`enable_thinking: false` on every request, asserted against
`/v1/mtplx/settings` before the first problem) so 164 problems do not each burn
a thousand thinking tokens.

### Environment

The lane is `CONTROL_FAMILY_ENV` in the script: the ABBA control arm's family
overrides, stated in full. Most of them the server family-defaults itself
(`_server_runtime_env_overrides`); the four it does not — `COMPILED_MTP_PREPARE`,
the frspec pair, and the three routed-down/GLU keys — come from
`abba_window.CONTROL_FLAGS` / `CONTROL_CANDIDATE_ENV`. The server env starts by
**stripping every inherited `MTPLX_*`**, so a leftover export from a previous
arm cannot move the control's lane.

`MTPLX_NAX_VERIFY` is deliberately never exported. Turbo sets it to `1` and it
is not operator-overridable, so an exported `0` gets stomped back to `1`;
leaving it unset is what lets the server's own override (applied after the
profile) set it to `0`, which is what production serves.
`test_turbo_profile_cannot_stomp_the_family_env` runs the real
`apply_profile_env` to keep that true.

### Scoring

`evalplus.evaluate` 0.3.1 from `/Users/davidtai/projects/evalplus/.venv`,
dataset hash `fe585eb4df8c88d844eeb463ea4d0302`, ground truth already cached in
`~/Library/Caches/evalplus`. It runs after the server is stopped, so no model
is resident. It insists on a samples file covering all 164 problems, so a
`--n 20` run pads the rest with an empty solution into `samples_scored.jsonl`;
padding rows are never counted.

HumanEval+ pass@1 is `base_status == plus_status == pass`, not `plus_status`
alone — the plus tests are the extra inputs only. Feeding the retained
2026-08-24 native-MTP samples through this path reproduces that receipt exactly:
151/164 base (0.9207) and 148/164 plus (0.9024).

Receipts land in `.benchmark-artifacts/fable/evals/<label>.json` with the flag
sets, the model revision from `mtplx_runtime.json` / `.mtplx-source.json`, the
sampler, pass@1, the per-problem pass list, timings and a server log tail;
samples and per-request receipts sit in `evals/<label>/`.

### Caveat on `--n 20`

On this model the first 20 HumanEval problems all pass, so the smoke has no
discriminating power at the top of the range — it proves the pipe works, not
that the kernel is safe. Only the 164 run is a verdict.

### micro_qsa_m4.py

`MTPLX_FABLE_QSA_M4` collapses four QSA glue chains into Metal kernels
(`kernels/qwen4_qsa_m4_indexer.py`, plus the transposed-key output of
`kernels/qwen4_qsa_m4_fused_kv_gather.py`). Compiled-lane dispatch map per
QSA layer, measured on the CPU stream by
`tests/test_fable_qsa_m4.py::test_dispatch_map_before_after` against the
op-diet-armed baseline:

| sub-chain | stock | fused | |
| --- | ---: | ---: | --- |
| `_prepare_queries_eager` (RMSNorm + partial RoPE) | 12 | 1 | the SHIPPED prepare kernel; the fixed lane never called it |
| `_extend_pooled_fixed` (bank row) | 24 | 4 | 1 kernel + the diet's `mx.slice_update` |
| scoring epilogue (relu/sum/scale/mask/tie) | 9 | 1 | |
| rows-gather token build | 18 | 1 | bit-exact, integers only |
| **per QSA layer** | **63** | **7** | **-56, x12 layers = -672/cycle** |

Untouched on purpose: the score GEMM, `mx.argpartition` (5 dispatches), and
the fused K/V gather. The top-k is not fusable *equivalently* — `token_idx`
carries `top_idx`'s ORDER into the gathered K/V rows, so any reordering
changes the softmax denominator's and the PV product's accumulation order.

`micro_qsa_m4.py` prices all five families (`prep`, `bank`, `score`, `tokens`,
`gather`) over one verify cycle (12 QSA layers) at the production shapes,
eager and under `mx.compile`, and prints max-abs-diff AND a differing-element
count against each stock spelling.

Measured 2026-09-01, compiled lane:

| family | stock | fused | differing elements |
| --- | ---: | ---: | ---: |
| `prep` | 0.557 ms | 0.199 (**-64%**) | 0 |
| `bank` | 1.153 ms | 0.251 (**-78%**) | 0 |
| `score` | 0.500 ms | 0.283 (**-44%**) | 0 |
| `tokens` | 0.628 ms | 0.205 (**-67%**) | 0 |
| `gather` | 1.501 ms | 1.657 (**+10%**) | 104, max abs 0.125 — **REJECTED** |

`MTPLX_FABLE_QSA_M4` is therefore the first four: bit-exact, and faster on
every one. The transposed-key gather is quarantined behind its own
`MTPLX_FABLE_QSA_M4_KT` (default off) and stays in the bench as the rejected
alternative, the way `bank_select` stays in `micro_opdiet.py`. Its 0.125 is
exactly one bf16 ulp at a score in [16,32): MLX's score GEMM takes a different
fp32 accumulation path for a natively-transposed B operand than for a
`swapaxes` view it has just made contiguous, so the layout is a rounding-class
change to attention output — and the 32x32 tiled transpose that produces it is
slower than the MLX copy it removes (a transpose cannot vectorize both sides,
so it trades a `vec<T,4>` streaming copy for scalar 2-byte accesses through
threadgroup memory plus a barrier).

```
PYTHONPATH=<branch checkout> \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 --child-timeout-seconds 1200 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  <branch checkout>/scripts/fable/micro_qsa_m4.py \
    --reps 20 --out /tmp/micro-qsa-m4.json
```

Adoption bar, per family: the fused spelling under stock in compiled ms/cycle
AND 0 differing elements. All four shipped families clear it. Confirm on the
verifier with an ABBA arm carrying `--candidate-env MTPLX_FABLE_QSA_M4=1`; the
gate is acceptance parity. The lane does NOT require
`MTPLX_QSA_M4_FUSED_KV_GATHER` — only `MTPLX_FABLE_QSA_M4_KT` does, and arming
KT without it RAISES at cache install rather than binding a stock gather and
looking inert.

## K20 row logging and the offline acceptance scorers

`MTPLX_FABLE_K20_LOG=<path.npz>` captures, per verify window, the K20 rows the
accept decision holds — draft rows for every depth and target rows for every
position — plus the drafted tokens, the primary, the PCG64 draws, and the
decision outputs. About 0.66 MB per 1,024-token request. Read once at import,
default off; see `mtplx/fable_k20_log.py` for the field list and the cost note.

**It captures whichever lane the run uses**, and records which — the two lanes
hold their rows at different stages of shaping:

| layout | lane | rows are | raw logits |
|---|---|---|---|
| `pr391_raw` | opt-in softfloat64 D3/M4 device kernel (`--d3-softfloat64-route`) | the kernel's **raw** input; top-p 0.95 and the double renormalisation happen inside it | yes |
| `stock_prepared` | the **retained** stock native-MTP host accept loop (the default) | **already** temperature/top-p/top-k shaped and renormalised | no — `values` is `log(prob)` |
| `stock_prepared_bv` | the same stock lane with `MTPLX_FABLE_BLOCK_VERIFY=1` | identical to `stock_prepared`, plus the window's block ladder | no |

On the pr391 lane the rows ride the decision's existing `mx.eval`, so an armed
run adds no new synchronisation — just a ~1.7 kB device-to-host copy per cycle.
On the stock lane there is no device work at all: every array copied is one the
host already built for its own decision. Either way the copies sit on the
critical path, so **an instrumented run is a data run, not a timing run.** Read
tok/s off an un-instrumented arm.

An armed run that captures nothing **exits non-zero** and says why — a silent
empty file is the failure this instrumentation exists to prevent. A greedy run
(`temperature <= 0`) builds no distributions at all, so its windows are recorded
with `greedy = 1` and no rows, and the scorers skip them and say so.

```
# Either lane; the log records which one fired.
MTPLX_FABLE_K20_LOG=/tmp/rows.npz <the usual ABBA / benchmark command>

# H §Option B — block verification vs the shipped law, same logged uniforms
python scripts/fable/offline_block_verification.py /tmp/rows.npz \
    --ms-per-window 37.47
python scripts/fable/offline_block_verification.py /tmp/rows.npz --cap one

# H §Option D — draft temperature / top-p / top-k sweep against sum(min(p, q))
python scripts/fable/offline_draft_temperature.py /tmp/rows.npz --tail lump
python scripts/fable/offline_draft_temperature.py /tmp/rows.npz --tail drop
```

Both scorers are pure NumPy, never import mlx, and load either layout through
one path — they branch only on whether the rows still need the kernel's
preparation. `offline_block_verification` replays the **shipped** law first and
fails if it disagrees with the decision the lane logged (on the stock layout,
every field except the correction id, which is drawn off the live generator and
so is not reproducible offline) — that check is what makes the block-law number
trustworthy. It fails again if block verification diverges on a window whose
reach credit stayed at 1, where the two laws are provably identical, or if no
window in the log is complete enough to score.

Two stock-lane caveats the scorers print rather than hide: accept coins past
the first rejection were never drawn, so a counterfactual that accepts deeper
uses a stream seeded from the window's own logged PCG64 state (the same stream
for both laws); and the temperature sweep can only re-temper the **retained**
support, since the shaping dropped the tail before the host saw it — so for
T > 1 its number is a lower bound.

Quote the `E[tok/win]` column (accept coins integrated out, paired standard
error), not the `replay` column; the replay is the exactness proof, and its
arm-to-arm difference is noise-dominated. Per H §1.4 a live A/B cannot resolve
either option at all.


## Block verification on the stock lane (`MTPLX_FABLE_BLOCK_VERIFY`)

H §Option B, Sun et al. 2024 (arXiv:2403.10444). Read once at import, default
off; when off the accept loop evaluates exactly the expressions it evaluated
before — same acceptance probability, same residual, same uniforms, same order.
Measured offline on 381 real windows: **+1.85% tokens/window, 2.487 → 2.533.**

The shipped law decides depth `d` from `x_d` alone and is already saturated at
depth 1. The whole M4 forward is finished before the accept loop runs, so every
target row is on the host when depth 1 is decided, and the decision may legally
look at `rho_2` and `rho_3`. One clip moves — the **running product** is clipped
at 1 instead of each factor — and the resulting budget is water-filled across
the depth `d+1` draft support, so it is spent on the realisations whose next
drafted token the target likes:

```
c_0 = w_0 = 1
for d in 1..D:
    rho_d = p_d(x_d) / q_d(x_d)
    A_d   = min(1, c_{d-1} * rho_d)                      # reach budget
    w_d   = min(w_{d-1}, min(1, A_d*rho_{d+1}(x_{d+1})) + lam_d)   # d < D
    w_D   = A_D
    a_d   = w_d / w_{d-1}                                # the accept coin
    reject -> sample from normalise((c_{d-1}*p_d - q_d)+)  # SCALED residual
bonus ~ p_{D+1}                                          # unchanged
```

`lam_d` is the water-fill level that holds `E_{x_{d+1}~q_{d+1}}[w_d]` at `A_d`,
which is what keeps the law exact. The cap is `w_{d-1}`, not H's literal
`min(1, .)`: with H's cap `a_d` can exceed 1 and is not a probability. When the
ladder never leaves 1 the two laws coincide token for token (43–47% of windows),
which is the partial parity check.

**It draws nothing extra.** Same accept coin per depth, same one `rng.choice`
for a correction, same one for the bonus; the water-fill is deterministic. It
adds no verifier row, no draft step, and no device work — the ladder is ~70 µs
of NumPy on 20-wide rows against a 30 ms forward.

A window arms only when all `D` draft rows and all `D` target rows are already
on the host, so the flag never forces the lazy per-row path to materialise a row
it meant to skip; a window that cannot arm runs the shipped law and records
`block_valid = 0`. Both laws are exact samplers of the same target
distribution, so mixing them per window changes nothing about the output.

```
# arm it, and log the rows so the run can be checked
MTPLX_FABLE_BLOCK_VERIFY=1 MTPLX_FABLE_K20_LOG=/tmp/bv.npz <benchmark command>

# the exactness check: must report EXACT on both lines
python scripts/fable/offline_block_verification.py /tmp/bv.npz
```

On a `stock_prepared_bv` log the scorer replays the **block** law (per window,
under the law that window's `block_valid` says ran) and additionally recomputes
the ladder from the rows and demands entry-for-entry equality with the one the
lane wrote. Both must print `EXACT`; anything else means
`mtplx/fable_block_verify.py` and `scripts/fable/offline_block_verification.py`
have drifted apart, and the run's numbers are worthless.

Receipts change meaning in exactly one place: `drafts[].accept_probability` and
`accept_probability_sum_by_depth` carry `a_d`, the conditional accept
probability, which is the same operational quantity `min(1, p/q)` was but is no
longer an estimator of the TV overlap `beta_d`. Read `alpha_uncensored` out of
the scorer for `beta`.

## Tests

```
cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps
.venv/bin/python -m unittest tests.test_fable_abba_window -v
```

(`pytest` is not installed in this venv; the test file also runs under
`pytest tests/test_fable_abba_window.py -q` where it is available.)

`tests/test_fable_qsa_m4.py` runs entirely on the CPU stream (66 cases): the
flag's gating and narrowing, every eligibility raise, CPU references of each
Metal body pinned against the stock chain it replaces, and the dispatch-map
receipt above.

`tests/test_fable_route_kernel.py` runs entirely on the CPU stream (51 cases):
the routing head's tie rule derived from MLX 0.32.2's `sort.h` and proved
against `mx.argsort` on constructed exact ties, the four rounding boundaries
the Metal source transcribes (bf16 sum order, bf16 divide, the bf16 sigmoid
decomposition, precise-softmax dtype), the geometry invariants that pin
`k_lanes = 8`, source tripwires on the load-bearing spellings, and every
construction-time refusal. It cannot execute the kernel; the bit-exactness
claim itself is gated per layer inside `install_qwen4_m4_stage3`.

`tests/test_compiled_copy_round.py` is pure Python (`mtplx.context_copy`
imports only `functools` and `os`): the read-once
`MTPLX_FABLE_COMPILED_COPY_ROUND` flag, and the padding law that lets one
traced graph serve every ladder rung -- exact width, prompt-continuation
content, deterministic tail fill, and the invariant that the logical block is
an untouched prefix of the padded rows, which is why acceptance cannot observe
the padding. Compiled-vs-eager output equality is a device claim and needs a
GPU window.

## TTFT screen (multi-turn coding-agent traffic)

`ttft_screen.py` is the latency half for *warm* turns. ABBA measures decode
tok/s and `humaneval_screen.py` measures quality; neither measures the number
an agent actually feels on turn N: how long the server takes to emit its first
token when the conversation is long and the client re-rendered part of the
transcript. That is the failure mode the oMLX PR #3330 audit targets, and the
one MTPLX answers with `near_prefix_candidates` -> `restore_entry_prefix_cache`
-> boundary-true restore -> suffix-only prefill.

One guarded window per arm. The child starts ONE MTPLX server on **:8092**
(never :8080, never :8091 — the HumanEval screen owns that one) from this
worktree's venv, with `PYTHONPATH` pinned to the worktree so the server imports
this checkout's `mtplx`.

### The three scenarios

Each repeat runs one conversation, in order, against a fresh salt:

| Scenario | Prompt | What it isolates |
| --- | --- | --- |
| `cold` | `POST /admin/cache/clear` + a per-repeat salted ~16K-token workspace dump | Upper bound: full prefill |
| `matching_terminal` | the same turn plus the model's own reply and one more user turn | The ordinary warm case (audit arm E) — the banked terminal is an exact prefix |
| `rerendered_terminal` | identical, except the prior **assistant** turn is re-rendered | **The target** (audit arm D) — divergence lands *inside* the banked terminal while the opening prompt stays an exact prefix |

The re-render is whitespace and markdown only: `- ` bullets become `* `, fence
info strings are dropped, tabs expand to four spaces, non-empty lines gain a
trailing space. Same information, different bytes. The harness **refuses to
run** scenario 3 if the transform came out a no-op — otherwise arm D would
silently measure arm E.

### Session headers (undocumented in `docs/server.md`)

This is the trap the audit names: `_session_keep_live_refs_for_request`
(`mtplx/server/openai.py`) returns False for an anonymous session with no
tools, so a naive `curl` harness measures the snapshot-only path and
**understates** MTPLX's current baseline.

`EngineSessionManager.resolve_session_id` (`mtplx/engine_session.py`) reads
these headers, case-insensitively, first match wins, in this order:

| Header | Notes |
| --- | --- |
| `x-mtplx-session-id` | MTPLX's own name; checked first |
| `x-session-affinity` | OpenCode stamps this on every request |
| `x-session-id` | OpenCode stamps this too |
| `x-openwebui-chat-id` | Open WebUI |
| `x-openwebui-user-id` | Open WebUI |

Any of them resolves `session_source` to `header.<name>`, which beats
prompt-prefix inference *and* arms the live-reference lease. Failing that, the
lease is armed by `user` / `chat_id` / `conversation_id`, by a `tools:` array
containing coding-agent tool names (`bash`, `edit`, `glob`, `grep`, `read`,
`todowrite`, `write`, …), or by `MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS=1`.

`--live-ref {header,tools,both,env}` picks exactly one, default `header`. It is
pinned into the arm identity, so two arms can never differ in it. Prefer
`header`: a `tools:` array also activates the server's tool-call contracts,
which rewrites the prompt template and can make the model answer with a tool
call instead of text (nothing to time).

### Arms

Control (no `--env`) is production: the launcher exports no `MTPLX_*` at all,
so neither does the control arm.

```
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/ttft_screen.py \
    --label control --repeats 3 --port 8092
```

Candidates — same command with one of:

```
    --label nearprefix --env MTPLX_FABLE_NEAR_PREFIX_RESTORE=1
    --label qsastage   --env MTPLX_FABLE_QSA_RESTORE_STAGING=1
    --label protterm   --env MTPLX_FABLE_PROTECTED_TERMINAL=1
    --label boundary32 --gdn-boundary-max 32
```

**Run `nearprefix` first.** The 2026-09-01 control receipt
(`.benchmark-artifacts/fable/ttft/control.json`) measured
`rerendered_terminal` at **15.79 s against `matching_terminal`'s 0.217 s**, on
all three repeats, with `cached=0` and `session_restore_mode: "cold"`. Root
cause, from that receipt's own `session_bank` block:

* the bank auto-sized to its **1 GiB floor** (`session-bank budget: 1.0G total
  (auto: machine memory plan...), model weights 107.1G` — production's
  resolution on this box too);
* a 19K-token entry's base snapshot is ~711 MB and each GDN boundary record
  ~87–101 MB, so an 8-record payload is ~700–810 MB;
* `SessionBank.put` counts that payload into `entry_nbytes` and then refuses
  the **entire** entry — `skipped_oversized_snapshot` at 1,398,321,776 and
  1,520,850,304 bytes against a 1,073,741,824 budget — while the same turn's
  boundary-**less** commit (710,255,120) was admitted;
* so the bank only ever held boundary-less entries (the survivor reports
  `gdn_boundaries: []`), `recurrent_boundary_at_or_below()` returned None, the
  near-prefix lane rejected every candidate with `boundary_not_better:0`, and
  the turn fell through to a cold prefill.

`MTPLX_FABLE_NEAR_PREFIX_RESTORE=1` sheds boundary records until the entry
fits instead of dropping it. Watch `session_bank.boundary_shed_puts` in the
receipt: > 0 means entries that used to be refused are now being admitted.

`--gdn-boundary-max 32` is the audit's suggested "cheapest lever", and under
that bug it makes things **worse** — a bigger payload triggers the refusal
harder. Only run it *after* `nearprefix`, on top of it.

`--dry-run` prints the exact outer command and the server argv without
touching the GPU.

### What the receipt records

Receipts land in `.benchmark-artifacts/fable/ttft/<label>.json`; the server log
and the arm-identity claim live in `.../ttft/<label>/`.

Per request: client `first_chunk_s` / `first_token_s` (visible TTFT) **and**
the server's own `mtplx_stats.ttft_s` (model TTFT) — the audit insists these be
reported separately, and oMLX's own two numbers differ by ~0.4 s. Plus
`prompt_tokens`, `cached_tokens`, `new_prefill_tokens`, `session_cache_hit`,
`cache_miss_reason`, `session_restore_mode`, `session_restore_served` (the
served-entry ground truth: `entry_prefix_len`, `requested_matched`,
`actual_restore_point`, `boundary_restore`, `storage_restore_mode`),
`peak_memory_bytes`, and `accepted_by_depth` / `drafted_by_depth`.

Per scenario: median, min, max and **p95**. p95 is not decoration — the
`b5fac4ac` phase-3 falsification was a tail result (27.6 s worst stall) that
medians hid completely.

Parity: `output_sha256_by_repeat` is the cross-arm key — ordered by repeat,
never a set, because each repeat carries its own salt and therefore its own
prompt. `assistant_turn_sha256` is recorded for the cold turn as well: turns 2
and 3 embed that text, so two arms are only comparable when it matches. Check
both before reading any TTFT delta.

`--salt-seed` must be **identical across arms** (it defaults to the fixed
`fable-ttft-v1`, deliberately not a timestamp — a per-run seed makes two arms
measure two different prompts). Change it only to force genuinely cold SSD
rows, and then change it for every arm.

The receipt also carries the `session_bank` block from `/v1/mtplx/snapshot` —
`prefixes[].gdn_boundaries` (the field that read `[]` in the control run),
`recent_evictions` (where `skipped_oversized_snapshot` and `shed_gdn_boundaries`
appear), `protect_newest_extending` / `protected_rejections` for
`MTPLX_FABLE_PROTECTED_TERMINAL`, and `boundary_shed_puts` /
`boundary_shed_records` for `MTPLX_FABLE_NEAR_PREFIX_RESTORE`.

One reporting trap to know: `cache_miss_reason` on a missed warm turn reads
`ssd_prefix_miss`, which is stamped by the SSD lookup inside
`session_bank.restore()` — that runs *after* the RAM near-prefix lane has
already returned None (generation.py tries near-prefix first). The SSD reason
therefore MASKS the real RAM-lane verdict. Use
`--env MTPLX_DEBUG_PREFIX_DIVERGENCE=1` to see it.

Add `--env MTPLX_DEBUG_PREFIX_DIVERGENCE=1` on a first pass: the server log
then prints `boundary-miss: entry_len=… matched=… boundary_positions=[…]` and
`near-prefix reject: … reason=…`, which tells you in one run whether the
rerendered arm is boundary-limited or identity-limited.

### Tests

```
cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps
.venv/bin/python -m pytest tests/test_fable_ttft_screen.py -q
```

Pure host tests: argv/env construction, the session-header names checked
against `mtplx/engine_session.py` and the tool names against
`mtplx/server/openai.py` (so a rename breaks the test rather than the
measurement), scenario prefix relationships, the re-render no-op refusal, and
the SSE folding arithmetic. No server, no MLX, no Metal.

---

## Long-prompt agreement screen (prefill numerics)

`longprompt_agreement_screen.py` is the quality gate for changes to the
**prefill** path — the chunk width (`MTPLX_PREFILL_CHUNK_SIZE`), the blocked
GDN prefill (`MTPLX_GDN_BLOCKED_PREFILL`), the QSA query tile
(`MTPLX_FABLE_PREFILL_QSA_QUERY_TILE`), mask fusion. The HumanEval screen
cannot gate any of them: every HumanEval prompt is a few hundred tokens, so it
fits inside ONE prefill chunk and never crosses a boundary. HumanEval passing
says nothing about a chunk-boundary bug.

The measurement is greedy agreement. At `temperature 0` the server takes a bare
`mx.argmax` (`_make_sampler`), so the continuation is a deterministic function
of the prefill: numeric drift big enough to flip one argmax shows up as a
divergence, and everything after it is permanently different. The metric is
therefore the **exact-match prefix length**, not "agreement over 256
positions" — after the first mismatch the two arms are answering different
questions.

### The prompts

Eight per run by default, built by the same generator and from the same pinned
fixture as the ABBA `coding-16k-1k-xhigh-t1` cell
(`abba_driver.production_prompt_content`, SHA-256 pinned; the screen calls it
purely as the drift gate before building its own):

| name | seed | target tokens | at 2,048 | at 4,096 |
| --- | --- | --- | --- | --- |
| `long-16k-s20260829`..`s20260834` | 20260829..20260834 | 16,384 | 8 chunks | 4 chunks |
| `mid-9k` | 20260835 | 9,216 | 4 + 1,024 | 2 + 1,024 |
| `short-4k5` | 20260836 | 4,608 | 2 + 512 | 1 + 512 |

`abba_window.PRODUCTION_SEEDS` are **sampler** seeds and the ABBA cell is one
fixed prompt; greedy decoding has no sampler seed, so this screen reuses those
integers as **prompt** seeds — each rotates the pinned coding context by a
different number of lines before the length-targeting builder cuts it. Same
material, same builder, six distinct 16K prompts, so one divergence cannot be
a single prompt's fluke. Seven of the eight land exactly on their target
(`--dry-run` prints the real counts, tokenizer only, no GPU); `short-4k5`
oscillates between 4,607 and 4,609 because one context token can be worth two
at the seam, which `PROMPT_TOKEN_TOLERANCE = 8` accepts.

Both short prompts end in a ragged chunk, which the 16K cell — a multiple of
both widths — never exercises. What these sizes do NOT give is a final chunk
of a different WIDTH in the two layouts: both tails are 512-aligned, so the
last chunk starts at the same offset either way. That needs `n % 4096 >= 2048`
(7,168, say: a 1,024 tail at 2,048 against a 3,072 tail at 4,096) and is a
reasonable thing to add to `SHORT_PROMPTS` if a boundary bug is suspected.

### The three arms

One guarded window per arm, ONE MTPLX server on **:8093** (8091 is the
HumanEval screen, 8092 the TTFT screen; never :8080). `--print-commands`
prints these verbatim:

```
# control
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/longprompt_agreement_screen.py \
    --label control --n 6 --port 8093

# candidate: prefill chunk 4,096
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/longprompt_agreement_screen.py \
    --label chunk4096 --n 6 --port 8093 \
    --env MTPLX_PREFILL_CHUNK_SIZE=4096 --env MTPLX_QSA_PREFILL_COMPILE_ROWS=4096

# candidate: blocked GDN prefill
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/longprompt_agreement_screen.py \
    --label gdnblocked --n 6 --port 8093 --env MTPLX_GDN_BLOCKED_PREFILL=1
```

Run the control TWICE (`--label control` and `--label control-2`) if you want
the noise floor; see the scoring section.

`MTPLX_PREFILL_CHUNK_SIZE` and `MTPLX_QSA_PREFILL_COMPILE_ROWS` **must move
together**. `mtplx/models/qwen4_exp.py` only serves the compiled QSA prefill
graph when `rows == _qsa_prefill_compile_rows()`, so moving the width alone
demotes every full chunk to the eager selector and the arm scores a lane
nobody proposed. `assert_candidate_env_coherent` refuses the pair before the
model loads (`mtplx.fable_prefill_chunk.assert_prefill_chunk_coherent` refuses
it again inside the server).

### Environment and memory

The lane is `humaneval_screen.CONTROL_FAMILY_ENV` — the ABBA control family,
stated in full — because these candidates are ABBA candidates and
`abba_window` measures their SPEED against exactly that lane. `--env` keys
must start with `MTPLX_` (`humaneval_screen.parse_env_settings`); there is no
`MTPLX_FABLE_*`-only restriction to work around, and all three keys used here
are registered in `mtplx.profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS`.
`MTPLX_PREFILL_CHUNK_SIZE` is also in `PROFILE_ENV_USER_OVERRIDE_KEYS`, so the
turbo profile's `auto` cannot stomp the candidate's `4096`; the other two are
absent from the turbo profile entirely.

On top of the family the screen exports the **Metal allocator caps**, the same
values `abba_driver` uses:

```
MTPLX_MEMORY_LIMIT_BYTES = 96 GiB
MTPLX_WIRED_LIMIT_BYTES  = 90 GiB
```

Neither `humaneval_screen` nor `ttft_screen` sets these, which is safe at their
prompt sizes and **not** safe here. Unset, `_apply_metal_memory_caps` lands on
60% of 128 GiB = 76.8 GiB, raised to the qwen4_exp resident floor (~83.3 GiB),
while a 16K prefill at chunk 4,096 peaks around 92.7 GB (~86.3 GiB) — over the
cap, and Metal's forced eviction is the documented ~10x serve collapse. The
candidate arm would have been scored on a machine that was swapping. Setting
the wired limit also ARMS `mtplx.fable_prefill_chunk`'s construction-time
geometry guard, whose budget is exactly `MTPLX_WIRED_LIMIT_BYTES` when no
explicit budget is set: 90 GiB minus the 2 GiB margin = 88 GiB, which the
projected peak clears. Unset, that guard is inert.

`--ssd-session-cache off` matters more here than in the other screens: a
cross-request session cache could serve arm B's 16K prefill out of arm A's KV,
which is the one thing that would make a broken prefill look identical.

### logprobs: not available on this build, and why not faked

`/v1/chat/completions` **rejects** `logprobs`/`top_logprobs` with a 400
("support is planned" — `mtplx/server/openai.py`), so the screen requests
`n=1` greedy tokens only and records that fact in the receipt's `logprobs`
block. It probes once per run, so the day chat logprobs land the screen starts
recording top-5 without an edit.

The one endpoint that does emit per-position top-K is `/v1/completions` with
`echo=true, logprobs=k, max_tokens=0`. It is the **wrong instrument here**:
`mtplx/generation.py:score_prompt_logprobs` teacher-forces the prompt in its
own fixed 256-token chunks, so it never reaches the prefill chunker or the
4,096-row QSA graph this screen exists to gate (and it caps out at
`MTPLX_PROMPT_SCORE_MAX_TOKENS`, 8,192, well under a 16K prompt). Wiring it in
would have produced confident-looking numbers about code that did not run.

### Scoring and the verdict rule

Pure Python, no GPU, no MLX, no tokenizer:

```
.venv/bin/python scripts/fable/longprompt_agreement_screen.py \
  --score .benchmark-artifacts/fable/longprompt/control.json \
          .benchmark-artifacts/fable/longprompt/chunk4096.json \
          [.benchmark-artifacts/fable/longprompt/control-2.json]
```

Per prompt: exact-match prefix length, whether the two arms are identical, and
— when the divergence position has logprobs behind it — the control's **top-2
margin** there, which is the rounding-class signature (two candidates the model
could not separate, so which one wins is decided by the last bit of the
reduction). Aggregate: median / min prefix length, the fraction of prompts with
full agreement, and the mean per-position top-1 logprob |Δ| over the agreeing
prefixes.

**VERDICT**

- With logprobs: **PASS** if every divergence sits at a near-tie (control top-2
  margin `< 0.05` nats, `--near-tie-nats`) **AND** the mean top-1 logprob |Δ|
  over the agreeing prefixes is `< 0.02` (`--max-logprob-delta`). Otherwise
  **FLAG**, naming the prompt and position.
- Without logprobs (this build): **PASS** only when every prompt agrees for its
  full length. A divergence with no margin behind it cannot be shown to be a
  near-tie, so it is FLAGged for human review rather than excused. The FLAG
  carries the character window around the split — shared tail, then each arm's
  next 80 characters — so the reviewer can see what actually changed.

Exit code is 0 on PASS and 1 on FLAG. `--json` prints the full report.

A third receipt is read as a **second control** and reported as the run-to-run
noise floor. Greedy decoding is deterministic, so control-vs-control that is
not 100% identical means the lane itself is nondeterministic and the candidate
comparison above it is void — the scorer says exactly that.

Receipts land in `.benchmark-artifacts/fable/longprompt/<label>.json`: the flag
sets (candidate, family, Metal caps), the model revision, the guard receipt,
the prompt table with per-prompt SHA-256, the per-prompt completions with the
server's `usage`, and a server log tail. `results.jsonl` in
`longprompt/<label>/` is the resume file; `--label` plus `arm.json` refuses to
append one arm's completions to another's.

### Tests

```
cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps
.venv/bin/python -m pytest tests/test_fable_longprompt_screen.py -q
```

Pure host tests (73): argv/env construction, the chunk-width/compile-rows
coherence rule, the Metal caps checked against `abba_driver.py`, the candidate
keys checked against `mtplx/profiles.py` (so a rename breaks the test rather
than the measurement), the chunk-boundary arithmetic that justifies the prompt
sizes, prompt sizing against a fake tokenizer, the logprobs probe in all three
outcomes, and every branch of the scorer and the verdict rule. No server, no
MLX, no Metal, no network.

---

## micro_k20_select.py — is the K20 selector exposed, or is it graph tail?

`MTPLX_FABLE_DEVICE_K20` (`mtplx/fable_device_k20.py`) replaces four per-cycle
top-20 selections on the stock native-MTP lane — three draft rows and the
4-row target support — with the exact device selector parked on
`experiments/pr391-target-lmhead-top20`, and samples the drafted token on
device so the three draft syncs collapse into one.  Whether that is worth
anything depends on a number the receipts cannot separate:
`verify_target_distribution_time_s` is 3.15 ms/window, but the
`MLX_DISABLE_COMPILE` arm shows the same code at 0.70 ms with
`verify_forward` correspondingly larger — i.e. most of the 3.15 is the
compiled verify graph's TAIL being awaited at the first sync, not selection.

This bench removes the graph.  With no model in front of it, whatever the
selector costs here is selection.

```
PYTHONPATH=<branch checkout> \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 900 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  <branch checkout>/scripts/fable/micro_k20_select.py \
    --json /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/micro-k20-select.json
```

Shapes: `[4, 248320]` (the target support), `[1, 248320]` (one draft depth on
the FRSpec scatter) and `[1, 65536]` (the compact FRSpec domain, for sizing the
follow-up that keeps the draft row unscattered).  Variants: `stock_serial`
(production, host tail and sync included), `stock_deterministic` (the all-device
selector the spill path falls back to), `device_k20`, `device_k20_host` (the
production route: kernel + logsumexp + one sync + the float64 top-p mask) and
`read_floor` (`mx.sum`, the bandwidth floor).

`--lane queued` (default) issues `--reps` graphs and evaluates once;
`--lane eager` evaluates each call.  Per `queued-vs-eager-metal-microbench` the
eager lane charges every call a host sync and can invert a verdict for
microsecond kernels, so promote on the queued number — but note that the
production draft site really is eager, which is the whole point of the change.

**Decision rule.** `stock_serial - device_k20_host` at `[4, 248320]`:
≥ 0.5 ms means the target-side lever is real; ≤ 0.15 ms means the receipts'
3.15 ms is graph tail and only the draft side (three `[1, V]` selections, each
followed by a sync) is worth building.

Every run also prints two parity blocks, so exactness is measured rather than
asserted: `parity` counts rows where the device selector disagrees with the
kernel-free NumPy oracle or with `_device_serial_support_arrays`, and
`choice_parity` counts rows where the device draft sampler disagrees with its
own CPU oracle, where the host mirror `prepare_draft_row_f32` disagrees with
`_prepare_reference_row`, and where a sampled token fell outside the `q` the
accept loop would score it against.  All five must be 0.

`--self-test` runs the NumPy oracle against a brute-force sort with no MLX and
no lock.

### With `MTPLX_FABLE_DEPTH4_PROBE`

The two compose. The device chain skips the per-depth loop that normally
captures the probe's inputs, so it captures them itself, from the materialised
result after its single sync — the probe still fires on all-accept windows and
still reads depth 3's own hidden, token and MTP cache. The probe's own `q_4`
row keeps going through the stock host shaping (`_distribution_from_mlx_logits`)
on purpose: it is measuring the MODEL, not the selector, so its rows stay
directly comparable between an armed and an unarmed device run.

The log layouts `stock_device_k20` / `stock_device_k20_bv` are stock layouts
for every consumer — they carry `gate_q` and the optional `probe_*` block, and
both offline scorers accept them. `gate_q` is then `q(x_d)` under the law the
device sampled from, which is the right gate feature for L §D.

### Tests

```
cd <branch checkout>
PYTHONPATH=. /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  -m pytest tests/test_fable_device_k20.py -q
```

Pure host, no MLX import: the flag-off source contract by AST inspection, the
selector oracle against brute force (ties, signed zero, NaN, compact-domain id
remapping), the oracle and the host tail against a NumPy transcription of
`_device_serial_support_arrays`, `prepare_draft_row_f32` bit-identical to the
choice kernel's own CPU oracle, the draw accounting against a flag-off
`rng.choice` chain, and the `stock_device_k20` logger round trip.

---

## micro_draft_k20.py — the FR-Spec draft row, scattered vs compact

`MTPLX_FABLE_DRAFT_K20_PRESCATTER` (`mtplx/fable_draft_k20_prescatter.py`)
builds each draft step's K20 support from the FR-Spec head's 65,536-row output
instead of the 248,320-wide scatter it is padded into, and maps the selected
LOCAL rows back to real token ids through the ranked table. Because MLX is
lazy and nothing under this route evaluates the scattered array, the
`put_along_axis` is built and dropped rather than run — so the step loses the
scatter AND shrinks both device passes (`argpartition` to 80, full-vocabulary
`logsumexp`) by 3.79x.

This bench prices that with no model in front of it, and re-measures on the
Metal stream the one exactness claim the CPU tests cannot settle: whether the
two DIFFERENT-WIDTH `logsumexp` reductions associate their float32 partials
identically. The sentinel terms are exactly `+0.0` (float32 `exp` underflows
below ~-103.97 and the pad is `-1.67e30` after the temperature divide), so the
sums are equal as real numbers; only the reduction tree is in question, and a
residual ULP there cannot change the SUPPORT.

```
PYTHONPATH=<branch checkout> \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 900 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  <branch checkout>/scripts/fable/micro_draft_k20.py \
    --lane queued --reps 200 --parity-rows 32 \
    --json /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/micro-draft-k20.json
```

Default shape: `65536x248320`, the production FR-Spec pair. Variants:
`stock_scatter_serial` (what the lane pays: scatter + the shipped builder over
248,320, host tail and sync included), `stock_serial_only` (the builder alone,
on an already-materialised row), `scatter_only` (`mx.full` + `put_along_axis`,
device-terminated), `prescatter_serial`, `stock_read` / `prescatter_read` (the
whole draft read including `_serial_row_distribution` and the one `rng.choice`
draw) and two `read_floor`s.

**Decision rule.** The draft chain is three sync-terminated steps per window.
`stock_read - prescatter_read` in 0.3–0.6 ms/step (1.0–1.8 ms/window) is the
expectation; below ~0.15 ms the gate is not worth carrying.

Parity is counted, never asserted: `support_ids_differing`,
`support_prob_bits_differing` and `draw_rows_differing` must be 0, while
`logsumexp_ulp_nonzero_rows` / `logsumexp_ulp_max_abs` bound the reduction-tree
residual rather than gating on it. `--self-test` runs the ranked-table and ULP
helpers with no MLX and no lock.

---

## Confidence-gated depth 4: the probe (`MTPLX_FABLE_DEPTH4_PROBE`)

L §D. H killed adaptive depth from *history* (acceptance is memoryless across
windows). Within a window it is not: the drafter's own probability of the token
it drafted predicts the target's acceptance strongly — `q(x_d) >= 0.95` gives
`a` of 0.96/0.91/0.89, `q(x_d) < 0.2` gives 0.54/0.46/0.43. Gating a 4th draft
step on `q(x_3) > 0.8` fires on 30% of windows, 52% of which accepted all three
drafts, and is worth **+0.147 tok/window for +0.89 ms (+3.5%)** — *if* `alpha_4`
on the gated windows is at least 0.75. Ungated depth 4 is −0.5%, exactly as H
found. The whole 5–8 day M=5 program hangs off one number nobody has measured.

**The measurement needs no M=5 verify graph.** After a normal M4 cycle whose
three drafts were all accepted, the target's bonus row is
`p(. | primary, d1, d2, d3)` — which *is* the distribution a fourth draft would
be verified against. So the probe runs one extra `rt.draft_mtp(..., mtp_depth=4)`
from the d3 hidden/token, shapes it with the draft sampler, and logs the row
(`q_4`) next to that bonus row; the scorer pairs them and reports
`alpha_4 = sum_x min(p_3(x), q_4(x))`, which is `E_{x~q}[min(1, p/q)]` with the
accept coin integrated out. The depth 1..3 ladder is reported in the same form
so the four columns are comparable.

The probe is a **pure read**: it samples nothing, draws no uniform, commits no
token, and restores the MTP cache offset in a `finally`, so the emitted stream
and the RNG stream are bit-identical to an unarmed run. `mtplx/generation.py`
runs it inside the all-accept branch, after the verify decision and before the
MTP history commit; it is gated on the stock host accept lane, `temperature > 0`,
no target prefix, a full-depth window, and the persistent MTP cache.

It costs ~1.6 ms on the ~31% of windows that reach it, self-timed into
`event["timing_s"]["fable_depth4_probe"]`. **An armed run is a data run, not a
timing run** — read tok/s off an unarmed arm.

The log gains `gate_q` (`q(x_d)` for every depth of *every* window — the
denominator of every gate) and, only when the probe recorded something, the
optional `probe_valid` / `probe_ids` / `probe_values` / `probe_probs` /
`probe_trimmed` columns. An unprobed log keeps exactly the schema the existing
scorers were written against.

```
# 3 seeds in ONE driver process, under the guard, from this worktree
python scripts/fable/abba_driver.py \
    --source $PWD --label fable-w25-depth4probe --sequence <seq> \
    --seed 20260829 --seed 20260830 --seed 20260831 \
    --target-mode batched --require-compiled-verify --m4-stage3 \
    --qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare --max-tokens 1024 \
    --candidate-env MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE=1 \
    --candidate-env MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL=1 \
    --candidate-env MTPLX_QWEN4_M4_ROUTED_GLU=1 \
    --env MTPLX_FABLE_DEPTH4_PROBE=1 \
    --env MTPLX_FABLE_K20_LOG=$PWD/.benchmark-artifacts/fable/k20-depth4-3seeds.npz

# the go/no-go
python scripts/fable/offline_depth4_gate.py \
    .benchmark-artifacts/fable/k20-depth4-3seeds.npz --ms-per-window 38.7
```

`MTPLX_FABLE_*` rides `--env`, not `--candidate-env` (`abba_driver.parse_key_values`
enforces that split). The arm is the retained control stack, unchanged, so the
`gate_q` / `P(all 3 | G)` columns are directly comparable to `L_gate_out.txt`.

The scorer prints one line of verdict: **GO** when
`alpha_4 | q(x_3) > 0.8 >= 0.75` on the probed windows, **NO-GO** otherwise, and
`NO-GO (undetermined)` when no probed window reached that gate. It reports the
projection under both marginal row costs the ledger holds — 1.8 ms (H §2.4, what
L's table used) and 1.4 ms (K's fit for a *compiled* fixed-width row) — because
neither is measured on an M=5 graph, and deciding whether to build one is the
entire point.

## micro_qsa_sparse_gqa.py — the native direct-index sparse-GQA kernel (B3)

Standalone parity + timing for `mtplx.native.qsa_sparse_gqa`, MTPLX's port of
oMLX's Steel-MMA `qwen4_qsa_sparse_gqa` (Jonathan Spangler, jundot/omlx
`7467dce8`, Apache-2.0). Nothing in the model calls the kernel yet; this is the
phase-1 falsifier from `scratchpad/M-holistic-tps-program.md` §B3, and the same
kernel at M=4 / M=1 is K-Q2 / K-D6, so those cells are in the sweep too.

Cells: `prefill-{16k,32k,64k}` at 4,096 rows, `decode-m4-16k`, `decode-m1-16k`.
Arms, all fed one selection: the ported kernel per `(BK, DC)` tile, the shipped
NAX `qsa_prefill_flash`, the portable `_qsa_prefill_gather_attention`, and the
dense lane (`_qsa_blocks_to_dense_mask` + `_qsa_dense_attention`), which is
also the numerics reference.

Selection is the production `QSAIndexer._select_eager`, imported, not copied.
The kernel derives per-slot validity in-kernel instead of reading `block_valid`,
so the bench **asserts** the selector invariant that makes those identical
(ascending ids, prefix validity, count `min(512, (pos+1)//4)`) on every row, and
materialises both token lists in full on sampled rows, before it trusts a parity
number. Tolerance is a stated bf16-ULP bar, not "looks close": this is a
rounding-class change (fp32 online softmax vs bf16 scores + precise softmax).

The extension has to be built first — CPU-only cmake, see
`mtplx/native/__init__.py`. Phase-2 wiring plan:
`docs/perf/qsa-sparse-gqa-phase2-wiring.md`.

Guarded window (the default sweep is minutes of GPU, dominated by the dense arm;
transient estimate per cell and refuses anything over `--max-transient-gb`):

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w50-qsa-sparse-gqa \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 \
  --child-timeout-seconds 3600 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w50-qsa-sparse-gqa/scripts/fable/micro_qsa_sparse_gqa.py \
    --out /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/micro-qsa-sparse-gqa.json
```

GO bar (the program note's own falsifier): parity inside the ULP bar on every
cell, identity assertions clean, and >=3x faster than `qsa_prefill_flash` on the
queued lane at 4,096 rows. Then, and only then, the 32K ABBA.

### Tests

`tests/test_qsa_sparse_gqa_native.py` — CPU-only. Covers the binding's
shape/dtype/position contract (MLX arrays are built but never evaluated) and
the harness's own scoring logic on numpy inputs, including the cases where the
identity check must fail.

---

## Shadow-draft acceptance (`shadow_draft_harness.py`)

Instrument I3. It exists so that a change to the **draft proposal** — indexer
reuse across depths, row K-D2, any cheaper way to compute the same draft step —
is judged on acceptance from rows on disk instead of by a live A/B. The per-run
acceptance spread is ±4.2 % (1σ, H §7): a 3-seed A/B cannot see a +2 % proposal
effect, and every attempt to look for one costs two guarded windows.

The number it produces, per depth, is the accept probability with **both** noise
sources integrated out — the accept coin and the draw of the drafted token:

```
alpha_d = sum_y min(p_d(y), q_d(y))  =  E_{x~q_d}[min(1, p_d(x)/q_d(x))]
```

the same estimator `offline_depth4_gate.py` reports, so the columns line up.

### How it runs: capture once, score forever

**capture** (GPU, guarded) replays a `MTPLX_FABLE_K20_LOG` trajectory. The log
carries `primary`, `draft_tokens`, `accepted` and `selected_token` per window,
which *is* the committed stream, so the trajectory — and the request boundaries
of a 3-seed log — reconstruct from the log alone. At every window the replay
runs the draft chain **once per variant from the identical hidden state**,
teacher-forced to the logged draft tokens, and writes each chain's rows.

**score** (pure NumPy, no GPU, no lock) pairs each variant's rows against the
same logged `p` rows and reports α₁..₃, the reach ladder, E[tokens/window] and
the realised accept count under the logged uniform tape, with paired standard
errors — i.i.d. and a moving-block bootstrap that keeps blocks inside a request,
because windows within one request are serially correlated and the i.i.d. SE is
a floor, not the interval.

### Teacher forcing is the definition, not a shortcut

`p_d` is `p(· | primary, x_1..x_{d-1})`: the logged target row at depth `d` is
conditioned on the tokens the **logged** chain drafted. A free-running candidate
that drafts a different token at depth 1 would be verified at depth 2 against
the wrong distribution, and every number past that point would be fiction. So
the candidate is forced onto the logged tokens: only `q_d` moves, `p_d` stays
exactly the row it is verified against, every depth is scored exactly.

Forcing isolates the divergence rather than hiding it. The report prints
`P(diverge)` per depth — how often the candidate would have drafted a different
token — from the logged draft tape on the `stock_device_k20` / `pr391_raw`
layouts, and from the row's argmax (a weaker question, labelled as such) on the
host stock layouts, which consume the draw inside `rng.choice` and never surface
it. Near zero means measured; 0.3 means the depth-2 and depth-3 numbers describe
a chain the model would often not have taken.

### What it can and cannot judge

**Can** — anything that alters only the draft proposal `q`: the drafter's
arithmetic, fusion, caching or scheduling (`MTPLX_FABLE_INDEXER_REUSE`, row
K-D2, compiled draft support, a cheaper QSA indexer, a different MTP hidden
variant); draft-side shaping; anything whose whole claim is "same output law,
cheaper or better `q`".

**Cannot**, and it does not pretend to:

- anything that changes **`p`** — the target model, its quantisation, its
  shaping, the verify graph's numerics. The logged rows would no longer be the
  rows the candidate faces and nothing here would notice.
- anything that changes the **accept law** — block verification, a different
  clip, a residual change. That is `offline_block_verification.py`, which
  replays laws against fixed rows: the mirror image of this harness.
- anything that changes the **window shape** — depth, adaptive width, gated
  stop, a 4th draft step. That is `offline_depth4_gate.py`.
- **wall time.** Acceptance is not tok/s. A candidate that wins α and costs
  2 ms/window loses. This prices nothing; multiply E[tok/win] by the ledger's
  ms/window yourself.
- a change that only fires **off** the logged trajectory — a different prompt,
  sampler or context length.
- a **greedy** run: `temperature <= 0` builds no distributions, so those windows
  are skipped and counted.

`offline_draft_temperature.py` already answers the draft-temperature question
from the log alone, with no replay and no GPU at all. Prefer it for that one.

### The gate that makes the rest trustworthy

Variant 0 is always `stock`: the same proposal path the logged run used,
re-drafted through the replay. Its rows must reproduce the logged `q` rows. The
scorer measures that as a total-variation distance per row and **withholds the
verdict** above `--fidelity-tol` (default 1e-6). A wrong hidden state, cache
offset, shaping or id space moves that distance far above the tolerance, so a
broken replay reports FAIL rather than a number. `build_replay_hooks` — the one
function in the file that touches MLX — was written without a GPU, and this gate
is what validates it on first run; the piece most likely to need adjustment is
the MTP-history restage in `advance`, which production does through
`generate_mtpk`'s nested `reconcile_mtp_indexer_history` (`generation.py:9650`).

The same gate catches the opposite failure. A candidate whose rows are
bit-identical to stock on every window reports **DID NOT ARM**: its flag is read
at import or at runtime construction, not per draft call, so the env scope
around the chain did nothing. Those need `--variant-module dotted:factory` (a
`ProposalVariant` with a `call` context manager) or two capture processes scored
against one log.

### Running it

Capture is a GPU job and rides the same guard as everything else here. Score is
pure NumPy — re-run it as often as you like, anywhere, off the lock.

```
# 1. capture the trajectory (the usual 3-seed ABBA run, with the K20 log armed)
python scripts/fable/abba_driver.py \
    --source $PWD --label fable-w47-shadow --sequence <seq> \
    --seed 20260829 --seed 20260830 --seed 20260831 \
    --target-mode batched --require-compiled-verify --m4-stage3 \
    --qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare --max-tokens 1024 \
    --candidate-env MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE=1 \
    --candidate-env MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL=1 \
    --candidate-env MTPLX_QWEN4_M4_ROUTED_GLU=1 \
    --env MTPLX_FABLE_K20_LOG=$PWD/.benchmark-artifacts/fable/k20-shadow-3seeds.npz

# 2. price the shadow capture before you book the window (no GPU)
python scripts/fable/shadow_draft_harness.py \
    .benchmark-artifacts/fable/k20-shadow-3seeds.npz \
    --variant indexer-reuse=MTPLX_FABLE_INDEXER_REUSE=1 --budget
```

The guarded capture-and-score, 3 seeds, with a placeholder variant env — the
flag does not have to exist yet; `DID NOT ARM` is the answer you get if it is
not read per draft call:

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/shadow_draft_harness.py \
    /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/k20-shadow-3seeds.npz \
    --capture-to /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/shadow-3seeds.npz \
    --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
    --variant indexer-reuse=MTPLX_FABLE_INDEXER_REUSE=1 \
    --expect-segments 3 \
    --budget \
    --json /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/shadow-3seeds.json
```

`--expect-segments 3` asserts the reconstruction found three requests; if it
finds two the trajectory does not match the run and the harness refuses to
replay it. Re-scoring afterwards needs no GPU and no guard:

```
python scripts/fable/shadow_draft_harness.py \
    .benchmark-artifacts/fable/k20-shadow-3seeds.npz \
    --rows .benchmark-artifacts/fable/shadow-3seeds.npz --budget
```

Budget, at the defaults (38.7 ms/window verify forward from L §3, 5 ms/draft
chain, 4 s prefill) over ~1,110 windows in 3 segments: ~22 s of GPU per seed for
stock + one candidate, ~66 s total, of which the *marginal* cost of adding a
second candidate is ~5.6 s. All three numbers are assumptions, printed with the
estimate; `--ms-per-window` / `--chain-ms` / `--prefill-s` move them.

### Tests

`tests/test_fable_shadow_draft_harness.py`, pure host, no MLX import — and one
test proves none is possible: every device-side import in the module lives
inside `build_replay_hooks`, checked by AST. It covers the segment
reconstruction (carry-in break, stop-token end, three seeds), the acceptance law
(exact match, disjoint support, a hand-computed Σ min, exact ties), the
depth-conditional reach ladder against the marginal α, `E[tokens/window]` with
and without a bonus, the realised accept count against a uniform tape, the
fidelity gate in both directions (PASS, FAIL → verdict withheld, DID NOT ARM),
the zero-variance paired delta a uniform candidate shift must produce, the
bootstrap's determinism, the budget arithmetic, the replay's call ordering under
a stub hook, and the report's rendering.

---

## Indexer reuse across the draft chain (`MTPLX_FABLE_INDEXER_REUSE`)

Row K-D2. Every depth of the 3-step MTP draft chain re-derives the QSA block
selection for a single query row — query norm + partial RoPE, a score GEMM over
all ~4,352 pooled blocks at 16K, relu/head-sum/mask/tie-break, `argpartition` to
top-512, blocks→tokens, the K/V gather. ~40 dependent dispatches, three times a
cycle, on a chain where nothing overlaps. Depths 2 and 3 pay it in full to
re-rank a history that grew by one token.

Armed, depth 1 selects normally and depths 2..3 are handed
`S_d = S_1 ∪ {b : nb_1 ≤ b < nb_d}` — the depth-1 block set plus the block the
chain's own tokens completed — and skip the preparation, the GEMM and the top-k.
`nb_d = (pos_start_d + 1) // ratio`; with `ratio = 4` and depth 3 at most one
block can complete inside a cycle, so one extra slot is exact rather than an
approximation, and the flag raises past `depth - 1 ≤ ratio`. The raw-key write
and the pooled-block bank update still run at every depth, so the cache the
verifier and the next cycle read is unchanged. Design and the causal /
valid / superset argument: `mtplx/fable_indexer_reuse.py`.

This moves the draft proposal `q` only — not `p`, not the verify graph, not the
accept law — so exact speculative sampling still holds and the output
distribution is unchanged. **Output digests WILL differ between the ABBA arms.**
A different `q` draws different tokens from the same law; a digest mismatch here
is the expected result, not a defect, and it is the reason the acceptance
question is answered offline instead of by reading tok/s alone.

An armed flag that meets a lane it cannot serve (`MTPLX_QSA_FLASH`,
`MTPLX_QSA_GATHER_DECODE`, the compiled or legacy-fused indexer,
`MTPLX_FABLE_COMPILED_DRAFT`) **raises**. It never reverts to the stock chain,
because a silent revert would put a stock number under the flag's label.

### (a) Acceptance — the guarded shadow capture, 3 seeds

`--variant NAME=KEY=VAL` arms the env around each draft-chain call, and the gate
is read per call precisely so that works; a `DID NOT ARM` verdict here would
mean the flag had been cached at import or at construction. Capture the
trajectory first (step 1 of the shadow-draft section above), then:

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/shadow_draft_harness.py \
    /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/k20-shadow-3seeds.npz \
    --capture-to /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/shadow-indexer-reuse-3seeds.npz \
    --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
    --variant indexer-reuse=MTPLX_FABLE_INDEXER_REUSE=1 \
    --expect-segments 3 \
    --budget \
    --json /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.benchmark-artifacts/fable/shadow-indexer-reuse-3seeds.json
```

Read the fidelity line first: above `--fidelity-tol` the verdict is withheld and
the α numbers mean nothing. Re-scoring needs no GPU and no guard.

### (b) Cycle time — ABBA

```
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 36000 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/abba_window.py \
    --sequence 1788500001 \
    --order ABBA \
    --label-prefix fable-indexer-reuse \
    --candidate-extra-env MTPLX_FABLE_INDEXER_REUSE=1
```

Each arm receipt row carries `indexer_reuse: {armed, cycles, steps_reused}`.
On the candidate `steps_reused` must be `2 * cycles` (depth 3, two reusing
depths); a shortfall means some cycles re-anchored and the ms/window delta is
not the lane's full effect. On the control both counters are 0, which is what
"the control really was the control" looks like.

Expected: **−0.5..−0.8 ms/window** if acceptance holds within noise. Acceptance
is not tok/s: a candidate that wins ms/window and loses α can still lose
end-to-end, so read (a) and (b) together — E[tokens/window] × ms/window.
