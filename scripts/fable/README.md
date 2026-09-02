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

`micro_dispatch_overhead.py`, `micro_moe_dedup.py`, `micro_expert_major.py` and
`micro_hc_read.py` price one site at the fixed-M4 verifier's shapes without
loading the model. They import MLX and therefore need the SAME guarded window
as an ABBA arm; none of them touch `com.tea.qwen`, so they can share a window.

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
