# fable ABBA harness (Qwen3.8 Flash-Next)

Matched A/B/B/A decode benchmarks on the production cell: 16,384-token coding
prompt, 1,024 output tokens, temperature 1 / top-p 0.95 / top-k 20, reasoning
`xhigh`, seeds `20260829 20260830 20260831`, all inside one guarded GPU window.

| File | What it is |
| --- | --- |
| `abba_driver.py` | One arm. Loads the model once, runs the requested seeds, writes a receipt JSON. |
| `abba_window.py` | The bracket. Runs as the guard's direct child and spawns one `abba_driver.py` per arm. |

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
acceptance parity, not a digest.

## Tests

```
cd /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps
.venv/bin/python -m unittest tests.test_fable_abba_window -v
```

(`pytest` is not installed in this venv; the test file also runs under
`pytest tests/test_fable_abba_window.py -q` where it is available.)
