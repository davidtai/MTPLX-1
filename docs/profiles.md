# Profiles

| Profile | Purpose |
|---|---|
| `turbo` | Default for the quantized 27B and 9B flagships (the Qwen 3.8 trio, Optimized-Speed, Optimized-Quality, the legacy Optimized hybrid, and their FP16 siblings, plus the 9B Speed pair): Sustained plus the NAX verify kernels and context-routed compiled verify. Fastest decode profile; matches the macOS app's launch presets. |
| `sustained` | Default `mtplx start` mode for every other model: native-MTP long-context path with chunked prefill, final-token logits, request-sized paged KV, and the normal Apple fan controller. |
| `sustained` + `--max` | Sustained Max: the same long-context path with ThermalForge/TG Pro fans pinned while MTPLX runs. |
| `performance-cold` + `--max` | Burst: old max-fan headline lane, not recommended beyond 8K context. |
| `performance-cold` | Legacy burst path without fan boost. Kept for explicit flags and compatibility; not shown in first-run onboarding. |
| `stable` | Hidden conservative alias for the exact/staged long-reply path and compatibility fallback. |
| `exact` | QA and release exactness checks. |
| `turbo-full-stack` | Opt-in. Turbo plus the Qwen3.8 Flash-Next decode restack the in-process benchmark drivers arm. Nothing selects it automatically; see below. |
| `max-diagnostic` | Fan-control diagnostics only. Onboarding modes are Auto (recommended; the engine resolves the profile per model), Sustained, Sustained Max, and Burst. |

`--max` is separate from profiles. It is opt-in and must restore fan state on exit when supported.

## The Flash-Next fast decode path is opt-in (`turbo-full-stack`)

On Qwen3.8 Flash-Next, `--profile turbo` does **not** arm the full decode
stack the in-process benchmark drivers measure. Those switches
(`scripts/fable/abba_driver.py`, and the `FULL_STACK_ENV` block in
`scripts/fable/server_cell_bench.py`) are env-gated and default-off, and only
the drivers set them — so a served turbo process ran without FR-Spec and
logged `[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)` while the same code
measured faster in-process.

Select the stack explicitly:

```bash
mtplx serve --profile turbo-full-stack --model <flash-next-pack> \
  --generation-mode mtp --load-mtp --depth 3
mtplx start --profile turbo-full-stack
```

`full-stack` is accepted as an alias.

What it changes, relative to `turbo`:

- arms the FR-Spec draft head (`MTPLX_FRSPEC_DRAFT=1`,
  `MTPLX_FRSPEC_VOCAB=builtin:qwen38-code-64k`), the compiled fixed-M4
  verifier and its stage-3 combine tail, compiled Qwen4 MTP preparation,
  relaxed draft ties, the fused GDN / MoE gate+up / hyper-connection lanes,
  the QSA rows-gather (plus its one-dispatch fixed-M4 K/V gather), and the
  pipelined AR + compiled GDN decode lane;
- resolves the only three keys the drivers and `turbo` disagree about in the
  **drivers'** favour: `MTPLX_BATCH_TARGET_ARRAYS=1` (turbo 0),
  `MTPLX_LAZY_TARGET_DISTRIBUTIONS=0` (turbo 1) and
  `MTPLX_SKIP_VERIFY_SNAPSHOT=0` (turbo 1). The first two are one decision:
  the batched arm is runtime-dead while the lazy strategy is on.

Nothing else moves. `turbo`, `sustained` and every other profile keep the
exact env they had, and no default changes anywhere.

### Scope

The `MTPLX_QWEN4_*` / `MTPLX_QSA_M4_*` keys are read only under the
`qwen4_exp` model type, but `MTPLX_FRSPEC_DRAFT` is family-agnostic and the
loader **raises** if the FR-Spec head cannot install. Do not select this
profile for a non-Flash-Next model.

### Checking that it actually engaged

Arming a flag is not the same as engaging a lane. With this profile selected
the server prints one `[full-stack] startup engagement …: satisfied|missing`
line per receipt, and repeats the set as `post-warmup` once the background
warmup ladder finishes:

| Marker | Receipt it reads | Armed by |
|---|---|---|
| `frspec_installed` | `[frspec] install report` (expects `n=65536`) | `MTPLX_FRSPEC_DRAFT` |
| `qwen4_fixed_m4_verify` | `[qwen4-fixed-M4-verify]` | `MTPLX_QWEN4_FIXED_M4_VERIFY` |
| `qwen4_m4_stage3` | `[qwen4-M4-stage3]` | `MTPLX_QWEN4_M4_STAGE3` |
| `qwen4_compiled_mtp_prepare` | `[qwen4-compiled-MTP-prepare]` | `MTPLX_QWEN4_COMPILED_MTP_PREPARE` |
| `ladder_all_ok` | background warmup ladder steps | `MTPLX_WARMUP_LADDER` |

The three `[qwen4-…]` reports are `logger.info` in `mtplx/runtime.py` and are
invisible under `python -m mtplx.server.openai`, so the self-check prints
their contents rather than pointing at a line you cannot see.

### Typo protection

`MTPLX_QWEN4_*`, `MTPLX_QSA_*` and `MTPLX_FRSPEC_*` keys are mostly read by a
single bare `os.environ.get` with a default, so a misspelling is silence, not
an error. Every key of this stack is declared in `mtplx/full_stack_env.py`
with its type, its unset default, its reader and the profile that sets it; at
startup — on **every** profile — a family-prefixed key that nothing in MTPLX
reads gets one `WARNING` line naming it (and the nearest known spelling). It
is advisory: nothing raises and no value changes.
