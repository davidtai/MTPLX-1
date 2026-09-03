# PR 391 run guide

Terms used in this document:

- MTP: multi-token prediction.
- QSA: Qwen Sparse Attention.
- GDN: Gated DeltaNet.
- PLE: per-layer embedding, fed from an n-gram sidecar table.
- FR-Spec: the frequency-ranked draft head over a pruned vocabulary.

This guide tells an operator how to serve the measured stack, how to prove
that it engaged, how to turn one optimization off, and how to reproduce the
battery and the quality screen.

The server arms the stack by default for a served Qwen3.8 Flash-Next pack. An
operator export always wins over a default. No other model family is affected.

---

## 1. Keys and files

The server stamps 26 environment keys by default. Two committed files hold the
23 tuning keys, and `mtplx/full_stack_env.py` is the source of truth in code.
`tests/test_full_stack_profile.py` asserts that each file equals its registry
group, so the file and the code cannot drift.

| File | Keys |
| --- | --- |
| `docs/perf/pr391-stack.flags` | the 15 decode keys |
| `docs/perf/pr391-prefill.flags` | the 8 prefill keys |

The other 3 default keys are `MTPLX_FRSPEC_DRAFT`, `MTPLX_FRSPEC_VOCAB` and
`MTPLX_QWEN4_COMPILED_MTP_PREPARE`.

Four keys need a note:

- The three `MTPLX_QWEN4_M4_ROUTED_*` keys are one chain with the route kernel. The residual tail requires the reduction, the paired gate and up producer requires the residual tail, and `MTPLX_FABLE_ROUTE_KERNEL` requires that producer. Each link is exact. All 4 keys share the `route_kernel` lane, so `--disable-optimization route_kernel` returns the whole tail to stock. Turn one link off on its own, and the server raises at model install and names the key to unset.
- `MTPLX_PREFILL_CHUNK_SIZE=4096` travels with `MTPLX_QSA_PREFILL_COMPILE_ROWS=4096`. The prefill graph bank captures one row width, and the runtime refuses a mismatched pair at the request boundary. Change both keys together, or change neither.
- `MTPLX_SESSION_BANK_MAX_BYTES=8G` is a serving memory budget and not a speed key. Unset, the bank sizes itself from the machine memory plan. Retune it per machine, or hand it back with the value `auto`.

`MTPLX_FABLE_PREFILL_CHUNK_ALLOW_COMPILE_ROWS_MISMATCH` is not a default, even
though the benchmark harness sets it on every branch arm. It waives the
coherence refusal, and the defaults never reach that refusal. A full chunk is
4,096 rows and the compiled row width is 4,096 rows, so the check returns at
its first branch. The narrower warm-up chunks are not configured full widths,
so the check returns at its second branch.

---

## 2. Install the model

Run the pull once:

```bash
mtplx pull Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --revision 29ba90f82124961d0d902a9ea9bbb1034972af2f
```

---

## 3. Build the native extension

The sparse decode lane (`MTPLX_FABLE_QSA_SPARSE_DECODE`) needs a native extension. The build artifacts are not in git. Build them once per checkout.

1. Find a nanobind whose internals version matches the one `mlx.core` was built with. The CMake step prints the required version when they differ. On this machine the matching package is the `nanobind` package of the venv that provides `mlx`.
2. Run the build:

```bash
cd native_extensions/qsa_sparse_gqa
cmake -S . -B build \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/mtplx_native_qsa/ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DPython_EXECUTABLE=<venv>/bin/python \
  -DMTPLX_NANOBIND_DIR=<nanobind package root>
cmake --build build -j 8
```

3. Check the three artifacts in `mtplx_native_qsa/`: `_ext.cpython-312-darwin.so`, `libmtplx_native_qsa_ext.dylib`, `mtplx_native_qsa_ext.metallib`.

Without the extension the server does not start with the default stack: the sparse decode lane fails at install with the message `MTPLX_FABLE_QSA_SPARSE_DECODE requires the built native extension`. Build the extension, or start the server with `--disable-optimization qsa_sparse_decode`. Without that lane the 16,384-token decode figure of this pull request is not reached.

## 4. Serve the measured stack

Start the server. There is nothing to opt into:

```bash
mtplx serve \
  --model <model directory> \
  --model-id mtplx-flash-next-optimized-speed \
  --generation-mode mtp --depth 3 --port 8095
```

`mtplx serve` re-executes the daemon as `python -m mtplx.server.openai`. The
benchmark harness uses that second form directly:

```bash
MTPLX_MEMORY_LIMIT_BYTES=107374182400 \
MTPLX_WIRED_LIMIT_BYTES=107374182400 \
python -m mtplx.server.openai \
  --model <model directory> \
  --model-id mtplx-flash-next-optimized-speed \
  --host 127.0.0.1 --port 8095 \
  --generation-mode mtp --load-mtp --depth 3 \
  --scheduler-mode serial --ssd-session-cache off
```

`--profile turbo-full-stack` selects the same 26 keys by name. On a served
Flash-Next pack it adds nothing that the server does not already do. It stays
useful for a caller that is not the server, and as an explicit label.

---

## 5. Read back what engaged

An armed key is not an engaged lane. Three surfaces report what happened.

1. Read the startup lines on standard output at model load. The first line
   counts the armed keys and names any lane that is off.
2. Read the per-key install verdicts in the server log. Filter with
   `grep '\[fable\]' <server log>`. Each line names the winner as `default:`
   or as `operator:`.
3. Read `GET /health`. Use `curl -s localhost:8095/health | python3 -m json.tool`.

`GET /health` carries three blocks:

- `engagement_reports.fable_defaults` lists `armed_by_default`, `operator_off`, `operator_pinned`, `disabled_lanes` and `model_gate`.
- `engagement_reports.full_stack_selfcheck.stack` gives one row per key: wanted, observed, valid, and the owner.
- `fable_install_receipts` repeats the per-key verdicts with live engagement counters.

A launch environment is never proof that a lane ran.

---

## 6. Turn one optimization off

Three spellings do the same thing, and they compose.

Turn a key off by name:

```bash
MTPLX_FABLE_QSA_SPARSE_DECODE=0 mtplx serve --model <model directory> --generation-mode mtp
```

Turn one or more lanes off through the environment:

```bash
MTPLX_FABLE_DISABLE=qsa_sparse_decode,opdiet mtplx serve --model <model directory> --generation-mode mtp
```

Turn one or more lanes off on the command line. The option repeats:

```bash
mtplx serve --model <model directory> --generation-mode mtp \
  --disable-optimization qsa_sparse_decode \
  --disable-optimization opdiet
```

A disabled lane leaves its keys unset instead of writing `0`. Nine of the 26
keys are widths, budgets, name lists or a vocabulary path, and `0` is not a
valid value for those. An unknown lane name raises an error, so a typo cannot
silently leave the optimization on.

Lane names, as `mtplx profiles` prints them:

| Lane | Keys |
| --- | --- |
| `compiled_mtp_prepare` | `MTPLX_QWEN4_COMPILED_MTP_PREPARE` |
| `frspec` | `MTPLX_FRSPEC_DRAFT`, `MTPLX_FRSPEC_VOCAB` |
| `hc_m4` | `MTPLX_FABLE_HC_M4` |
| `opdiet` | `MTPLX_FABLE_OPDIET` |
| `block_verify` | `MTPLX_FABLE_BLOCK_VERIFY` |
| `route_kernel` | `MTPLX_FABLE_ROUTE_KERNEL`, `MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE`, `MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL`, `MTPLX_QWEN4_M4_ROUTED_GLU` |
| `draft_k20_prescatter` | `MTPLX_FABLE_DRAFT_K20_PRESCATTER` |
| `graph_build_overlap` | `MTPLX_FABLE_GRAPH_BUILD_OVERLAP`, `MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS` |
| `verify_glue` | `MTPLX_FABLE_VERIFY_GLUE`, `MTPLX_FABLE_VERIFY_GLUE_ITEMS` |
| `qsa_sparse_decode` | `MTPLX_FABLE_QSA_SPARSE_DECODE`, `..._TILE`, `..._SPLITS` |
| `ple_prefill_lookahead` | `MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD` |
| `ple_first_gather_early` | `MTPLX_FABLE_PLE_FIRST_GATHER_EARLY` |
| `prefill_chunk` | `MTPLX_PREFILL_CHUNK_SIZE`, `MTPLX_QSA_PREFILL_COMPILE_ROWS` |
| `prefill_qsa_query_tile` | `MTPLX_FABLE_PREFILL_QSA_QUERY_TILE` |
| `gdn_blocked_prefill` | `MTPLX_GDN_BLOCKED_PREFILL` |
| `prefill_mask_fuse` | `MTPLX_FABLE_PREFILL_MASK_FUSE` |
| `session_bank_max_bytes` | `MTPLX_SESSION_BANK_MAX_BYTES` |

---

## 7. Serve the stock path

Turn every lane off in one option:

```bash
mtplx serve --model <model directory> --generation-mode mtp --disable-optimization all
```

`--profile turbo` is not the opt-out. The defaults live in the environment
resolution of the server, behind the served-config gate, and not in a profile.
They apply under every profile. Use `--disable-optimization all` for the stock
path.

---

## 8. Sanity check before a battery

The harness lives on branch `worker/w40-server-bench`. Its preflight boots the
server once with logging on, proves what installed, and shuts down. It sends
no timed request. Run it once per server build.

1. Change to the harness worktree.
2. Run `scripts/fable/server_cell_bench.py --mode preflight --stack branch
   --server branch-fullstack --port 8095 --require-full-stack`.
3. Pass `--server-python <interpreter>` and `--server-cwd <served tree>`.
4. Read the verdict. `--require-full-stack` aborts unless the log proves
   FR-Spec installed at 65,536 rows, one M4 route installed, and a valid
   warmup ladder.

Then send one request at 16,384 tokens through the same server. Add
`--mode run --contexts 16384 --cells sweep --seeds 20260829 --repeats 1`.
Pass `--dry-run` first: it prints the argument vector, the environment and the
per-cell plan, and it does not touch the GPU.

---

## 9. Reproduce the three-engine battery

`--stack` selects the engine and `--server` labels the arm. The labels must
differ, or the harness refuses to pool the receipts.

1. Run the branch arm: `--mode run --stack branch --server branch-fullstack
   --require-full-stack`.
2. Run the control arm: `--mode run --stack upstream --server upstream-2.10.2`.
   Give it the upstream interpreter and the upstream checkout.
3. Run the third arm: `--mode run --stack mlx-serve --server mlx-serve`. It
   runs its own binary, so it needs neither option.
4. Give every arm `--seeds 20260829,20260830,20260831 --repeats 3 --cells both`.
5. Render the tables with `scripts/fable/server_cell_report.py` over the
   receipt directory.
6. Render the charts with `scripts/fable/server_cell_charts.py` over the same
   directory.

The default battery is `--contexts 1024,8192,16384,32768,65536,131072,262144`
plus the vanity cell. `--stop-after-context 16384` gates the long cells, so
raise it deliberately once the 16K rungs are clean.

The harness sets memory parity itself. It exports
`MTPLX_MEMORY_LIMIT_BYTES=107374182400` and
`MTPLX_WIRED_LIMIT_BYTES=107374182400` on the two MTPLX arms, and
`MLX_SERVE_CACHE_LIMIT=107374182400` on mlx-serve. The shipped default pool
cap of mlx-serve is 8 GB, so those rows are not out-of-the-box mlx-serve
figures.

Results are in `pr391-battery-2026-09-03.md`,
`pr391-battery-2026-09-03-cells.md` and
`pr391-battery-2026-09-03-warm-prefix.md`.

---

## 10. Run the quality screen

The screen boots its own server and runs all 164 HumanEval problems greedy.
`--n 164` is the only verdict-grade size, and `--n 20` is a smoke test.

1. Run `scripts/fable/humaneval_screen.py --label <label> --n 164 --port 8091`.
2. Add `--env <KEY>=0` to measure one lane turned off.
3. Read the receipt under `.benchmark-artifacts/fable/evals/`.

The stack is armed by default, so a screen with no `--env` measures the stack.
A candidate arm is now a lane turned off.

---

## 11. Where each part lives

| Part | Location |
| --- | --- |
| Key registry, the source of truth | `mtplx/full_stack_env.py` |
| Committed key record | `docs/perf/pr391-stack.flags`, `docs/perf/pr391-prefill.flags` |
| Default arming | `mtplx/server/openai.py`, `_server_runtime_env_overrides` |
| Import-time key stamping | `mtplx/server/__init__.py` |
| Per-key install verdicts | `mtplx/fable_install_receipts.py` |
| Lane engagement markers | `mtplx/full_stack_selfcheck.py` |
| Profile | `mtplx/profiles.py` |
| Tests | `tests/test_fable_defaults.py`, `tests/test_full_stack_profile.py` |
| Battery harness | branch `worker/w40-server-bench`, `scripts/fable/server_cell_bench.py`, `scripts/fable/server_cell_report.py`, `scripts/fable/server_cell_charts.py` |
| Screens and micro-benchmarks | `scripts/fable/` |

Nine of the 26 keys are read once at module import, so the server stamps those
9 before the readers freeze them. `mtplx/server/__init__.py` applies the same
model gate, yields to operator exports and disabled lanes, and never raises.
`MTPLX_PROFILE_EARLY_ENV=0` turns that stamping off.
