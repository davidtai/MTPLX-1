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
| `turbo-full-stack` | Opt-in. Turbo plus the three Qwen3.8 Flash-Next decode switches no server path sets (FR-Spec and compiled MTP prepare). Requires `--generation-mode mtp`. Nothing selects it automatically; see below. |
| `max-diagnostic` | Fan-control diagnostics only. Onboarding modes are Auto (recommended; the engine resolves the profile per model), Sustained, Sustained Max, and Burst. |

`--max` is separate from profiles. It is opt-in and must restore fan state on exit when supported.

## The Flash-Next fast decode path is opt-in (`turbo-full-stack`)

On Qwen3.8 Flash-Next, `--profile turbo` gets you *most* of the decode stack
the ABBA benchmark control arm uses, but not all of it — and the part it
misses includes FR-Spec, which is why a served turbo process logs
`[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)` while the same code measures
faster in-process.

### What "the stack" means here

The reference is the **retained-stack control arm** — the configuration every
ABBA window measures its candidate against — i.e. one invocation of
`scripts/fable/abba_driver.py` carrying `scripts/fable/abba_window.py`'s
`CONTROL_FLAGS`:

```bash
python scripts/fable/abba_driver.py \
  --label <arm> --sequence <n> --seed <s> \
  --receipt-path <path> --guard-mode window \
  --target-mode batched --require-compiled-verify --m4-stage3 \
  --qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare \
  --max-tokens 1024
```

That resolves to `build_family_overrides(args)` (19 keys) plus the
`--full-frspec` block (2 keys) = **21 keys**. The registry declares the same
21 and a test derives them from the driver itself, so both sides are real.

Deriving it rather than transcribing it changed two things:
`MTPLX_QWEN4_RELAXED_DRAFT_TIES` is **not** in the control arm
(`--relaxed-draft-ties` is not in `CONTROL_FLAGS` and `--compiled-mtp-prepare`
does not imply it — it is a candidate-arm flag), and `MTPLX_COMPILED_VERIFY=on`
plus `MTPLX_NAX_VERIFY=0` **are**.

### Who sets what today

`mtplx serve` already supplies 18 of the 21 by itself, in
`mtplx/server/openai.py:_server_runtime_env_overrides`:

| Who | Count | Keys | Rule |
|---|---|---|---|
| server auto-arm, predicate `_served_model_type_is_qwen4_exp(args)` | 13 | `AR_PIPELINE`, `COMPILED_GDN`, `FAMILY_CAPTURE_COMMIT`, `FUSED_HC_V3`, `FUSED_GDN_INPROJ`, `FUSED_GATE_UP`, `FUSED_GDN_CONVNORM`, `FUSED_GDN_STEP`, `FUSED_CONVNORM_VERIFY`, `QSA_GATHER`, `BATCH_TARGET_ARRAYS=1`, `LAZY_TARGET_DISTRIBUTIONS=0`, `NAX_VERIFY=0` | set only when unset — **an operator export wins** |
| server auto-arm, predicate `_served_model_is_qwen4_fixed_m4(args)` | 4 | `COMPILED_VERIFY`, `QWEN4_FIXED_M4_VERIFY`, `QWEN4_M4_STAGE3`, `QSA_M4_FUSED_KV_GATHER` | same, via pop-on-operator |
| server **forced**, `mtp` + qwen4_exp | 1 | `SKIP_VERIFY_SNAPSHOT=0` | plain assignment — beats profile *and* operator |
| nobody | **3** | `MTPLX_FRSPEC_DRAFT`, `MTPLX_FRSPEC_VOCAB`, `MTPLX_QWEN4_COMPILED_MTP_PREPARE` | — |

Those last three are the gap, and they are all this profile sets.

It deliberately does **not** restate the 18. Server overrides are applied
after the profile env, so the value would not change — but a profile-owned key
*stomps* an operator's explicit export, while the server's auto-arm steps
aside for one. Restating them would take the A/B switch away from operators on
exactly the keys the server chose to leave them.

The profile's own three keys are in `PROFILE_ENV_USER_OVERRIDE_KEYS` for the
same reason: **`MTPLX_FRSPEC_DRAFT=0` is the kill switch**, and it wins over
the profile (the launch announces that it did). That matters because the
FR-Spec installer raises rather than falling back.

It also means this profile introduces **no conflict with turbo**. Every key
where turbo and the control arm disagree — `BATCH_TARGET_ARRAYS` (turbo 0),
`LAZY_TARGET_DISTRIBUTIONS` (turbo 1), `SKIP_VERIFY_SNAPSHOT` (turbo 1) and
`NAX_VERIFY` (turbo 1) — is already resolved control-wins by the server for a
qwen4_exp `mtp` serve, under turbo too. (`COMPILED_VERIFY` is not a conflict:
turbo's `1` and the control arm's `on` resolve to the same graphbank mode.)

### Selecting it

```bash
mtplx serve --profile turbo-full-stack --model <flash-next-pack> \
  --generation-mode mtp --load-mtp --depth 3
mtplx start --profile turbo-full-stack
```

`full-stack` is accepted as an alias.

### Requirements, enforced at selection

The profile **requires `--generation-mode mtp` and no `--mtp-adapter`**. Both
of its Qwen4 lanes raise inside `runtime.load` otherwise — compiled MTP
preparation needs the native draft head, and FR-Spec's installer raises rather
than falling back — which would be a traceback *after* the weights are mapped.
The server therefore refuses the profile at selection, naming the profile and
the reason, before anything loads. Use `turbo` for a non-MTP or adapter serve.

### Scope

The server's auto-arm is predicated on the served config, so on a non-qwen4_exp
model this profile arms FR-Spec and one Qwen4 key and nothing else — and
`MTPLX_FRSPEC_DRAFT` is family-agnostic. Do not select this profile for a
non-Flash-Next model.

### Checking that it actually engaged

Arming a flag is not the same as engaging a lane, and because 16 of the 20
keys depend on a server predicate that reads the served config, "the profile
was selected" is not proof the stack is armed either. With this profile
selected the server answers both, and repeats the set as `post-warmup` once
the background warmup ladder finishes.

First the env level, printed once — how much of the control stack is armed, by
whom, and against which serve shape, so a predicate that did not hold is
visible:

```
[full-stack] startup 21/21 driver-stack keys armed (mtp, qwen4_exp fixed-M4 pack) [profile 3, server_auto_arm 17, server_forced 1]
```

A partial stack names each missing key, the value wanted, and whose job it
was, e.g. `MTPLX_QSA_GATHER=None want '1' [server_auto_arm:
_served_model_type_is_qwen4_exp(args)]` — which, next to a shape of
`not a qwen4_exp pack`, says the served pack did not match the predicate
rather than that you mistyped something. A key counted as `reader default`
is satisfied only because nothing set it and the reader's own default happens
to match — nobody is holding it there.

Then one `[full-stack] … engagement …: satisfied|missing` line per receipt:

| Marker | Receipt it reads | Armed by |
|---|---|---|
| `frspec_installed` | `[frspec] install report` (expects `n=65536`) | `MTPLX_FRSPEC_DRAFT` |
| `qwen4_fixed_m4_verify` | `[qwen4-fixed-M4-verify]` | `MTPLX_QWEN4_FIXED_M4_VERIFY` |
| `qwen4_m4_stage3` | `[qwen4-M4-stage3]` | `MTPLX_QWEN4_M4_STAGE3` |
| `qwen4_compiled_mtp_prepare` | `[qwen4-compiled-MTP-prepare]` | `MTPLX_QWEN4_COMPILED_MTP_PREPARE` |
| `ladder_all_ok` | background warmup ladder steps | `MTPLX_WARMUP_LADDER` |

The three `[qwen4-…]` receipts are printed to stderr at install time on
**every** profile (`mtplx/runtime.py`), the same way `[frspec] install
report` always was — before that they were `logger.info` under a server that
configures no handler, so they appeared zero times in a real server log. The
self-check prints their contents too, so the verdict and its evidence sit on
one line.

After boot the same reports are readable at `GET /health` under
`engagement_reports`, next to `draft_lm_head`:

```json
{
  "engagement_reports": {
    "qwen4_fixed_m4_verify": {"installed": true, "linear_layers": 36, "rows": 4},
    "qwen4_m4_stage3": {"installed": true, "layers": 48, "...": "..."},
    "qwen4_compiled_mtp_prepare": {"installed": true, "...": "..."},
    "laguna_fused": null,
    "full_stack_selfcheck": {"phase": "post-warmup", "ok": true, "markers": []},
    "unknown_family_env_keys": []
  }
}
```

`null` means the lane did not install — its env key was not set, or the model
is not the family it belongs to.

### Typo protection

`MTPLX_QWEN4_*`, `MTPLX_QSA_*` and `MTPLX_FRSPEC_*` keys are mostly read by a
single bare `os.environ.get` with a default, so a misspelling is silence, not
an error. Every key of this stack is declared in `mtplx/full_stack_env.py`
with its type, its unset default, its reader and the profile that sets it; at
startup — on **every** profile — a family-prefixed key that nothing in MTPLX
reads gets one `WARNING` line naming it (and the nearest known spelling). It
is advisory: nothing raises and no value changes.
