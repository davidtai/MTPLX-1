# PR-391: running the measured stack

The configuration the PR-391 battery measured is now the configuration a
Qwen3.8 Flash-Next serve gets by default. This page is the runbook: how to
serve it, how to check it engaged, how to turn any one optimization off, and
how to reproduce the screens and the battery.

Everything here is a command you can paste. Where a command is assembled from
a tool's flags rather than quoted from an existing runbook, it says so.

---

## What changed, in one paragraph

Twenty-three env keys — the FR-Spec draft head and its 64k code vocabulary,
compiled Qwen4 MTP preparation, twelve retained decode keys and eight
retained prefill keys — used to reach a serve only if you exported two files
by hand. `mtplx serve` now arms all twenty-three **by default** for a served
Flash-Next pack, by the same `setdefault`-behind-a-config-predicate mechanism
that already armed the sixteen family keys above it in
`_server_runtime_env_overrides`. **An operator export always wins**, so
turning one off is an ordinary `MTPLX_FABLE_X=0`, and every key prints an
install-time verdict saying who won. No other model family is affected.

The two files are committed as the canonical record:

| file | keys |
| --- | --- |
| [`docs/perf/pr391-stack.flags`](pr391-stack.flags) | the 12 retained decode keys |
| [`docs/perf/pr391-prefill.flags`](pr391-prefill.flags) | the 8 retained prefill keys |

`mtplx/full_stack_env.py` is the source of truth in code;
`tests/test_full_stack_profile.py::test_each_flag_file_equals_its_registry_group`
asserts the two are equal, so neither can drift.

---

## 0. Prerequisites

| thing | value |
| --- | --- |
| model (MTPLX) | `/Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` |
| model (mlx-serve) | `/Users/davidtai/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit` |
| mlx-serve binary | `/Users/davidtai/projects/OpenSourceWTF/.tools/mlx-serve-macos-arm64/mlx-serve` (not on `$PATH`) |
| bench harness | worktree `/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w40-server-bench` (branch `worker/w40-server-bench`) |
| evalplus | `/Users/davidtai/projects/evalplus/.venv/bin/evalplus.evaluate` (EvalPlus 0.3.1, a pre-existing checkout) |
| GPU lock | `bench/laguna/run_guarded.py` — every Metal command goes through it |
| memory cap | `MTPLX_MEMORY_LIMIT_BYTES=107374182400` and `MTPLX_WIRED_LIMIT_BYTES=107374182400` (100 GiB) |

`mlx-serve --version` checks the binary:

```bash
/Users/davidtai/projects/OpenSourceWTF/.tools/mlx-serve-macos-arm64/mlx-serve --version
```

Every GPU command below is written bare for readability. **Run it under the
guard**, which holds the exclusive GPU flock (`/tmp/mtplx-gpu-exclusive.lock`)
and stops the production daemon first:

```bash
PYTHONPATH=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 1800 \
  --child-timeout-seconds 5400 \
  -- \
  <the actual GPU command>
```

---

## 1. Serve the measured stack

Nothing to opt into. On a Flash-Next pack this is the whole stack:

```bash
mtplx serve \
  --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --generation-mode mtp --depth 3 --port 8095 --no-auth
```

`mtplx serve` re-execs the daemon as `python -m mtplx.server.openai`; that is
also the form the bench harness uses, and it works directly:

```bash
MTPLX_MEMORY_LIMIT_BYTES=107374182400 \
MTPLX_WIRED_LIMIT_BYTES=107374182400 \
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  -m mtplx.server.openai \
  --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --model-id mtplx-flash-next-optimized-speed \
  --host 127.0.0.1 --port 8095 \
  --generation-mode mtp --load-mtp --depth 3 \
  --scheduler-mode serial --ssd-session-cache off --no-auth
```

`--profile turbo-full-stack` still selects the same 23 keys by name. On a
Flash-Next serve it now adds nothing the server was not already going to do;
it stays useful for a non-server caller (`mtplx/prefill_bench.py`, the ABBA
driver) and as an explicit label.

### The exact default env

Twenty-three keys, sorted. Nothing else in the tree sets any of them.

```
MTPLX_FABLE_BLOCK_VERIFY=1
MTPLX_FABLE_DRAFT_K20_PRESCATTER=1
MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1
MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS=3
MTPLX_FABLE_HC_M4=1
MTPLX_FABLE_OPDIET=1
MTPLX_FABLE_PLE_FIRST_GATHER_EARLY=1
MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1
MTPLX_FABLE_PREFILL_MASK_FUSE=1
MTPLX_FABLE_PREFILL_QSA_QUERY_TILE=2048
MTPLX_FABLE_QSA_SPARSE_DECODE=1
MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS=17
MTPLX_FABLE_QSA_SPARSE_DECODE_TILE=128:32
MTPLX_FABLE_ROUTE_KERNEL=1
MTPLX_FABLE_VERIFY_GLUE=1
MTPLX_FABLE_VERIFY_GLUE_ITEMS=qsa_rope,qsa_rope_idx
MTPLX_FRSPEC_DRAFT=1
MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k
MTPLX_GDN_BLOCKED_PREFILL=1
MTPLX_PREFILL_CHUNK_SIZE=4096
MTPLX_QSA_PREFILL_COMPILE_ROWS=4096
MTPLX_QWEN4_COMPILED_MTP_PREPARE=1
MTPLX_SESSION_BANK_MAX_BYTES=8G
```

`MTPLX_PREFILL_CHUNK_SIZE=4096` is the one value that replaces a turbo
default (`auto`, i.e. 2048 either way). It travels with
`MTPLX_QSA_PREFILL_COMPILE_ROWS=4096`: the QSA prefill graph bank captures a
single row width, and `assert_prefill_chunk_coherent` refuses the mismatched
pair at the request boundary. Change them together or not at all — the
`prefill_chunk` lane below does exactly that.

`MTPLX_SESSION_BANK_MAX_BYTES=8G` is a **serving memory budget, not a speed
key**. Unset, the session bank auto-sizes from the machine memory plan; the
retained set pins it so two arms cannot see different banks. Retune per
machine with `MTPLX_SESSION_BANK_MAX_BYTES=24G`, or hand it back with
`MTPLX_SESSION_BANK_MAX_BYTES=auto`. (`server_cell_bench.py`'s own
`COMMON_ENV` exports `auto`, so a battery run through that harness uses the
auto-sizer — an export beating a default, which is the mechanism working.)

---

## 2. Turn one optimization off

Three spellings, all equivalent, and they compose.

**By key** — any non-empty value takes the key away from the defaults:

```bash
MTPLX_FABLE_QSA_SPARSE_DECODE=0 mtplx serve --model ... --generation-mode mtp
```

**By lane name**, in the environment:

```bash
MTPLX_FABLE_DISABLE=qsa_sparse_decode,opdiet mtplx serve --model ... --generation-mode mtp
```

**By lane name**, on the command line (repeatable):

```bash
mtplx serve --model ... --generation-mode mtp \
  --disable-optimization qsa_sparse_decode \
  --disable-optimization opdiet
```

**The stock path** — every lane off, the shipped reader defaults everywhere:

```bash
mtplx serve --model ... --generation-mode mtp --disable-optimization all
```

Disabling a lane leaves its keys **unset** rather than stamping `0`: nine of
the twenty-three are widths, budgets, name lists or a vocabulary path, and
`MTPLX_PREFILL_CHUNK_SIZE=0` is not a configuration. Unset is what restores
each reader's own default.

An unknown lane name raises rather than disabling nothing — a typo that
silently left the optimization on would make an A/B measure the same arm
twice.

### The lanes

`mtplx profiles` prints this list.

| lane | keys |
| --- | --- |
| `compiled_mtp_prepare` | `MTPLX_QWEN4_COMPILED_MTP_PREPARE` |
| `frspec` | `MTPLX_FRSPEC_DRAFT` `MTPLX_FRSPEC_VOCAB` |
| `hc_m4` | `MTPLX_FABLE_HC_M4` |
| `opdiet` | `MTPLX_FABLE_OPDIET` |
| `block_verify` | `MTPLX_FABLE_BLOCK_VERIFY` |
| `route_kernel` | `MTPLX_FABLE_ROUTE_KERNEL` |
| `draft_k20_prescatter` | `MTPLX_FABLE_DRAFT_K20_PRESCATTER` |
| `graph_build_overlap` | `MTPLX_FABLE_GRAPH_BUILD_OVERLAP` `MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS` |
| `verify_glue` | `MTPLX_FABLE_VERIFY_GLUE` `MTPLX_FABLE_VERIFY_GLUE_ITEMS` |
| `qsa_sparse_decode` | `MTPLX_FABLE_QSA_SPARSE_DECODE` `..._TILE` `..._SPLITS` |
| `ple_prefill_lookahead` | `MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD` |
| `ple_first_gather_early` | `MTPLX_FABLE_PLE_FIRST_GATHER_EARLY` |
| `prefill_chunk` | `MTPLX_PREFILL_CHUNK_SIZE` `MTPLX_QSA_PREFILL_COMPILE_ROWS` |
| `prefill_qsa_query_tile` | `MTPLX_FABLE_PREFILL_QSA_QUERY_TILE` |
| `gdn_blocked_prefill` | `MTPLX_GDN_BLOCKED_PREFILL` |
| `prefill_mask_fuse` | `MTPLX_FABLE_PREFILL_MASK_FUSE` |
| `session_bank_max_bytes` | `MTPLX_SESSION_BANK_MAX_BYTES` |

---

## 3. Read back what actually engaged

Arming a flag is not engaging a lane. Three surfaces say what happened, and
all three come from receipts the runtime already publishes.

**The startup lines**, on stdout at model load:

```
[full-stack] startup 23/23 retained-stack keys armed by default (mtp, qwen4_exp fixed-M4 pack)
[full-stack] startup 41/41 driver-stack keys armed (mtp, qwen4_exp fixed-M4 pack) [profile 23, server_auto_arm 17, server_forced 1]
[full-stack] startup engagement frspec_installed ([frspec] install report): satisfied (installed n=65536)
...
```

With something turned off, the first line names it:

```
[full-stack] startup 21/23 retained-stack keys armed by default (...); lanes off: opdiet; operator off: MTPLX_FABLE_QSA_SPARSE_DECODE=0
```

**The per-key install verdicts**, one line per armed flag:

```bash
grep '\[fable\]' <server log>
```

```
[fable] opdiet armed: default: MTPLX_FABLE_OPDIET=1 ...
[fable] qsa_sparse_decode off (operator: MTPLX_FABLE_QSA_SPARSE_DECODE=0)
```

The `default:` / `operator:` prefix is the point — it says who won, not just
what the value is.

All twenty-three keys prove themselves: thirteen through an
`mtplx/fable_install_receipts.py` lane (`route_kernel`,
`graph_build_overlap` and `gdn_blocked_prefill` were added for this change),
and ten through a receipt their own lane already printed — the `[frspec]`
install report, `[qwen4-compiled-MTP-prepare]`, the `qwen4_hc_m4` pack
validation, `fable_verify_glue.install()`, `qsa_sparse_decode.engagement_line()`
and the mask-fuse `engaged:` lines. `EnvKeySpec.receipt` in
`mtplx/full_stack_env.py` names the second group, and
`tests/test_fable_defaults.py::test_every_defaulted_key_has_an_install_receipt`
fails on a key with neither.

**`GET /health`**, after boot:

```bash
curl -s localhost:8095/health | python3 -m json.tool | less
```

- `engagement_reports.fable_defaults` — `{armed_by_default: [...], operator_off: [...], operator_pinned: [...], disabled_lanes: [...], model_gate: ...}`
- `engagement_reports.full_stack_selfcheck.stack` — one row per key of the 41-key measured stack: wanted, observed, ok, who owns it
- `fable_install_receipts` — the same per-key verdicts as the `[fable]` lines, with live engagement counters

---

## 4. Sanity check before a battery

The harness's own preflight boots the server once with logging on, proves what
installed, and shuts down — no timed request. Run it once per server build.

```bash
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w40-server-bench
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  scripts/fable/server_cell_bench.py \
  --mode preflight --stack branch --server branch-fullstack \
  --server-python /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  --server-cwd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w93-profile-full-stack \
  --port 8095 --require-full-stack
```

`--require-full-stack` aborts unless the log proves FR-Spec installed at
`n=65536`, an M4 route installed, and an all-ok warmup ladder.

Then one 16K request through the same server. `--stop-after-context 16384` is
the harness default, so a plain `--mode run` will not walk past 16K:

```bash
/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  scripts/fable/server_cell_bench.py \
  --mode run --stack branch --server branch-fullstack \
  --server-python /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  --server-cwd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w93-profile-full-stack \
  --port 8095 --contexts 16384 --cells sweep --seeds 20260829 --repeats 1 \
  --require-full-stack
```

> Assembled from the harness's flags (`server_cell_bench.py:4287-4429`), not
> quoted from an existing runbook — the file carries no worked example.
> `--dry-run` prints the exact argv, env and per-cell plan without touching
> the GPU; run that first.

---

## 5. HumanEval screen

Quoted verbatim from `scripts/fable/README.md`. The screen boots its own
server, so it goes through the guard. `--n 164` is the only verdict-grade
size; `--n 20` is a smoke test. **No `--env` at all is the control arm.**

```bash
/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --lock-timeout-seconds 3600 --timeout-seconds 900 --child-timeout-seconds 5400 \
  -- \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python \
  /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/scripts/fable/humaneval_screen.py \
    --label control --n 164 --port 8091
```

Since the retained stack is armed by default, the control arm of a screen is
now **the stack**, and a candidate arm is a lane turned off:

```bash
  ... humaneval_screen.py --label no-qsa-sparse --n 164 --port 8091 \
    --env MTPLX_FABLE_QSA_SPARSE_DECODE=0
```

Receipts land in `.benchmark-artifacts/fable/evals/`.

---

## 6. The three-engine battery

`--stack` selects the engine and `--server` labels the arm; the labels must be
distinct or `check_arm_label` refuses to pool the receipts.

```bash
cd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w40-server-bench
BENCH=scripts/fable/server_cell_bench.py
PY=/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps/.venv/bin/python

# 1/3 -- this branch
$PY $BENCH --mode run --stack branch --server branch-fullstack \
  --server-python $PY \
  --server-cwd /Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.claude/worktrees/w93-profile-full-stack \
  --port 8095 --require-full-stack \
  --seeds 20260829,20260830,20260831 --repeats 3 --cells both

# 2/3 -- upstream control
$PY $BENCH --mode run --stack upstream --server upstream-2.10.2 \
  --server-python <upstream venv python> --server-cwd <upstream checkout> \
  --port 8095 --seeds 20260829,20260830,20260831 --repeats 3 --cells both

# 3/3 -- mlx-serve (no --server-python / --server-cwd; it runs its own binary)
$PY $BENCH --mode run --stack mlx-serve --server mlx-serve \
  --port 8095 --seeds 20260829,20260830,20260831 --repeats 3 --cells both

# then render
$PY $BENCH --mode report
```

> Assembled from the harness's flags, not quoted. Verify with `--dry-run`
> first. `--stop-after-context 16384` gates the long cells; raise it
> deliberately when the 16K rungs are clean. The default battery is
> `--contexts 1024,8192,16384,32768,65536,131072,262144` plus the vanity cell.

Memory parity across engines is `MTPLX_MEMORY_LIMIT_BYTES` /
`MTPLX_WIRED_LIMIT_BYTES` = `107374182400` on the two MTPLX arms and
`MLX_SERVE_CACHE_LIMIT=107374182400` on mlx-serve; the harness sets all three
itself (`server_cell_bench.py:1821-1826`, `:2775-2777`). mlx-serve's own
shipped default pool cap is 8 GB, so those rows are not out-of-the-box
mlx-serve numbers.

Results: [`pr391-battery-2026-09-03.md`](pr391-battery-2026-09-03.md) (cold)
and [`pr391-battery-2026-09-03-warm-prefix.md`](pr391-battery-2026-09-03-warm-prefix.md).

---

## 7. Why nine keys are armed before the server starts

Nine of the twenty-three are read **once at module import** —
`mtplx.runtime_options`, `mtplx.fable_block_verify` and
`mtplx.fable_draft_k20_prescatter` freeze them in module constants so the hot
path never touches `os.environ`. The server's own `setdefault` block runs
inside `_load`, thousands of imported lines later, so for those nine it would
change the environment and no reader would ever look again.

`mtplx/server/__init__.py` therefore stamps exactly those nine, from
`full_stack_env.stamp_import_time_defaults`. Python executes that package
`__init__` while resolving `python -m mtplx.server.openai` — before the first
line of `openai.py`. It applies the same model gate, yields to the same
operator exports and disabled lanes, and never raises.
`MTPLX_PROFILE_EARLY_ENV=0` turns it off.

`tests/test_fable_defaults.py::test_the_stamp_lands_before_the_readers_freeze`
proves it end to end in a subprocess, and the test beside it shows the same
env set one import too late reading back as off.

---

## 8. Where each thing lives

| what | where |
| --- | --- |
| the key registry (source of truth) | `mtplx/full_stack_env.py` |
| the committed record | `docs/perf/pr391-stack.flags`, `docs/perf/pr391-prefill.flags` |
| the defaults, armed | `mtplx/server/openai.py:_server_runtime_env_overrides` |
| the nine import-bound keys | `mtplx/server/__init__.py` |
| per-key install verdicts | `mtplx/fable_install_receipts.py` |
| lane engagement markers | `mtplx/full_stack_selfcheck.py` |
| the profile | `mtplx/profiles.py:TURBO_FULL_STACK_PROFILE` |
| tests | `tests/test_fable_defaults.py`, `tests/test_full_stack_profile.py` |
