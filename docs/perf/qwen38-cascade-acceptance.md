# Speculative-cascade acceptance for Qwen3.8 Flash-Next

Speculative-cascade acceptance is a second opt-in, lossy decode-verify rule
beside typical acceptance. Per draft position it decides whether the draft token
is good enough to keep, or whether to defer to the exact target law. It is OFF by
default, mutually exclusive with typical acceptance, and NOT distribution-exact.

Citation: Narasimhan, Jitkrittum, Rawat, Kim, Gupta, Menon, and Kumar, "Faster Cascades via
Speculative Decoding," arXiv:2405.19261 v2 (2024). This implements the
r-hat_OPT deferral rule of Section 4.3, Equation (10): the plug-in ESTIMATOR of
the optimal speculative-cascade deferral rule (Lemma 4, Equation (9)), replacing
that rule's ground-truth expected 0-1 losses with one minus each model's max
probability. It is not the optimal rule and not an oracle. The separate Diff rule
(Equation (5), max q < max p - alpha, no total-variation term) is the
SEQUENTIAL-cascade oracle and is not implemented here. Executed with the
speculative decoding of Algorithm 4.

Terms:

- p: the target distribution at a position (the truncated top-p / top-k target
  row already materialized for the exact rule).
- q: the draft distribution at a position (the native MTP head's scored rows).
- D_TV(p, q) = sum_v max(0, p(v) - q(v)) over the scored top-k support.
- defer (r = 1): use the large / target model here; accept (r = 0): keep the
  small / draft model's token.
- alpha: the deferral cost, the single operator knob.

## Problem

The exact speculative law commits the longest draft prefix the target would have
produced under the same coins, and no more. Even when the draft head closely
tracks the target it still spends a rejection whenever the coin `min(1, p/q)`
falls, which caps tokens per cycle. Typical acceptance loosens this with a
per-row typicality floor, but it is blind to how well the draft itself matched
the target: it can reject a token the draft was confidently and correctly
proposing, and it resamples from the target row rather than the residual.

## Change

A speculative-cascade verify rule that, at each draft position with target p and
draft q, applies Equation (10):

    defer (r = 1)  <=>  max_v q(v) < max_v p(v) - alpha * D_TV(p, q)

If it does NOT defer, the draft is "good enough" (the speculative-cascade target
is pi = q, so Algorithm 4's accept probability min(1, pi/q) = 1): accept the
draft token with no coin. If it DOES defer, pi = p and the code runs the exact
speculative law unchanged, accept with `min(1, p(x_t)/q(x_t))` and, on a coin
rejection, resample the residual `norm(max(0, p - q))`. So the deferral test uses
the total variation over the scored top-k, and the deferred path uses the token's
own target probability p(x_t); the cascade path is a strict superset of the exact
rule with a draft-accept shortcut. The decision is deterministic (it consumes no
uniform); the coin only appears on the deferred path, so with the lane off the
RNG stream and verify path are byte-identical.

What q is. On the served Turbo path the draft is the native MTP head. Its scored
rows reach the verify loop as `draft_probs[depth_index]`, a `SparseDistribution`
over the head's scored vocabulary, which for this pack is the FR-Spec
frequency-ranked subset (the same q the exact rule already uses in `min(1, p/q)`
and the residual). The rule reads q's peak `max_v q` and q's mass on the scored
top-k for D_TV from that object; it adds no draft distribution and no draft
forward. q is required, so the lane is a temperature > 0, MTP-on rule, like
typical acceptance.

The Lemma 3 note, in plain words. In Equation (10) the cost term `alpha * D_TV`
is SUBTRACTED from the target's confidence. So more disagreement between the
draft and the target LOWERS the bar for accepting the draft, which means the rule
defers LESS when they disagree, not more. The paper's reason (their Lemma 3) is
that a large disagreement makes the verification step itself expensive, so the
rule only pays that cost when the target is clearly better; it accepts a
draft that is confident, even on a token the target would not have picked. The
intuitive "reject when the draft diverges" behaviour still holds for the common
case, a diverging draft that has also lost its peak confidence, which defers; a
draft that stays confident on a wrong token is accepted. This is the paper's rule
as written; the tests pin it explicitly so a future sign change is caught.

## Effect

Higher acceptance when the draft distribution agrees with the target, including on
atypical tokens that typical acceptance would reject, at the cost of exactness
(the accepted draft token is sampled from q, not p). It is a speed / quality
dial, judged on task-quality evals like typical acceptance, never on
distribution-exactness.

## Exactness

NOT distribution-exact when on. The accepted-draft (non-defer) positions emit q's
token, which differs from the target law. The deferred positions ARE exact
(min(1, p/q) coin + residual). With the lane OFF (knob unset) the verify path is
byte-identical to the exact rule: the cascade branches are gated behind
`_cascade_active`, which is false, so neither the deterministic test nor any coin
runs.

## Files

- `mtplx/sampling.py`: `cascade_defer_decision`, `total_variation`,
  `_peak_probability`.
- `mtplx/generation.py`: `_cascade_accept_alpha` / `_cascade_accept_enabled`
  (env at use), `_assert_lossy_verify_rules_exclusive`, the two verify-loop
  decision branches (batched + lazy target), the `VerifyStats` cascade fields,
  and the `[cascade-accept]` verdict line.
- `mtplx/server/openai.py`: `--cascade-threshold`, the env stamp + fail-loud
  mutual exclusion with `--typical-threshold`, and the `/health`
  `cascade_acceptance` install report.
- `tests/test_cascade_acceptance.py`, `tests/test_cascade_threshold_cli_health_cpu.py`.

## Switch

- `--cascade-threshold ALPHA` (env `MTPLX_FABLE_CASCADE_THRESHOLD`): the deferral
  cost alpha. UNSET (or blank) = OFF = exact speculative sampling (the default).
  Any set value, including an explicit 0, turns the lane on, so the off switch is
  UNSETTING the key, not setting it to 0 (alpha = 0 still defers whenever the
  target is strictly more confident than the draft). Higher alpha widens the
  accept band, so fewer positions defer. Resolved at use, per request.
- Mutually exclusive with `--typical-threshold` (`MTPLX_FABLE_TYPICAL_THRESHOLD`
  > 0). Setting both fails loud at serve startup (SystemExit) and in the verify
  setup (ValueError).
- Observability: `/health` -> `cascade_acceptance` (`enabled`, `alpha`, the rule
  and divergence definitions, `conflict`); and a per-request verdict line
  `[cascade-accept] NOT distribution-exact; threshold=A alpha=A positions=N
  accepted=N resamples=N accept_rate=R mean_divergence=D ...` in the same format
  as `[typical-accept]`.

## Recommended alpha grid for the 16K sweep

The paper varies alpha continuously as a "lenience parameter" to trace a
quality-versus-latency Pareto curve (Section 6, Figure 2); it fixes no grid, and
reports temperatures T in {0, 0.1, 0.5, 1.0} and block sizes gamma in {3, 5, 7}
(Appendix E.1). So the grid is set from Equation (10)'s structure and this
model's measured scale.

Scale on this model, from the saved #478 receipts and verdict lines at 16,384
tokens, temperature 1, depth 3:

- Exact MTP acceptance rate is 0.43 to 0.47. The expected exact speculative
  acceptance rate equals the overlap `sum_v min(p, q) = 1 - D_TV(p, q)`, so
  `D_TV ~= 0.53 to 0.57` per position on average.
- Target-row entropy on the `[typical-accept]` lines is 0.83 to 1.81 nats, so the
  target peak `max_v p` sits around 0.35 to 0.65.

The deferral boundary is `max_v p - max_v q = alpha * D_TV`, so the useful alpha
range is set by the plausible target-versus-draft peak gap (at most ~max_v p,
about 0.6) divided by D_TV (~0.55): alpha up to about 1 already reaches the point
where the cost term dominates any confidence edge, and alpha ~2 defers almost
never. The four-value grid, from most deferral (least lossy, closest to the exact
rule) to near all-accept (fastest, lossiest):

| alpha | alpha * D_TV (D_TV ~= 0.55) | expected behaviour |
| --- | --- | --- |
| 0.0 | 0.00 | defer whenever the target is strictly more confident than the draft; maximal deferral |
| 0.5 | ~0.28 | defer only when the target's peak leads the draft's by more than ~0.28; moderate |
| 1.0 | ~0.55 | defer only on a large target-confidence lead; light |
| 2.0 | ~1.10 | defer essentially never; near all-accept |

Single alpha for the HumanEval cell: **alpha = 0.5**. It is the moderate point,
accepting the draft where the draft and target peaks are close and deferring
where the target clearly leads, which is the "gain speed while holding quality"
operating point the sweep should confirm. If HumanEval drops, fall back toward
alpha = 0 (more deferral); if quality holds, push toward alpha = 1.0 for more
speed.

## Provenance and rule-diff proof (arm G)

The arm G 16K sweep and the pending HumanEval cell were measured on served code
`2eac2fee` (cascade stacked on the #478 typical head). This branch re-parents the
cascade mode onto the #475 served base `27d5ff6b`, the same base #478 is built on,
so #478 (typical) and this PR (cascade) are alternative-mode PEERS on one base
rather than a stack. This is the same provenance bridge #475 uses (docs commits
over the served base); the measurements transfer because the cascade RULE is
byte-identical on the new base.

Per-function sha256 (first 16 hex) of the cascade computation, this branch vs the
measured `2eac2fee`:

| symbol | sha256 | vs 2eac2fee |
| --- | --- | --- |
| `mtplx/sampling.py::_peak_probability` | `f3bba8b637a1ff9e` | identical |
| `mtplx/sampling.py::total_variation` | `1b960bfacfb37ce9` | identical |
| `mtplx/sampling.py::cascade_defer_decision` | `3ba053b7097fa91e` | identical |
| `mtplx/generation.py::_cascade_accept_alpha` | `b81b13739d2166b7` | identical |
| `mtplx/generation.py::_cascade_accept_enabled` | `1c5fae48d7f81e85` | identical |
| batched-target cascade verify branch | `6d2749c7d09e8cbb` | identical |
| lazy-target cascade verify body | `1ab16b485087dce2` | identical |
| `[cascade-accept]` verdict block | `78d7c3e84c84f6c1` | identical |
| `/health` `_cascade_acceptance_health_payload` (feature commit) | `7a9006f74a187d70` | identical |
| `--cascade-threshold` argparse block | (identical) | identical |

The accept/defer/coin/residual, the total-variation and peak-probability
computations, the RNG draws, and the `[cascade-accept]` telemetry are therefore
byte-identical to the measured code. Three forced glue deltas remain, none of
which changes what a cascade run computes (so arm G stands):

1. Lazy-target dispatch keyword: `elif _cascade_active:` on `2eac2fee` (it chained
   off typical's `if _typical_active:`) becomes `if _cascade_active:` here, because
   typical is dropped. The branch body is byte-identical.
2. The exact-path lazy block is re-indented one level under the new `else:` and
   `target_p_for_cache = target_p` is hoisted above the dispatch (both idempotent
   / non-behavioral; the cascade-off output is unchanged).
3. `_assert_lossy_verify_rules_exclusive` reads `MTPLX_FABLE_TYPICAL_THRESHOLD`
   from the environment directly instead of calling the (now-absent)
   `_typical_accept_threshold()`.

Mutual exclusion (option a): the guard reads the typical env name defensively. It
is INERT on this branch, retained defensively; #478 and cascade are alternative
modes and were never intended to be armed together, so nothing here arms a typical
threshold, but the guard still fails loud (ValueError in the verify setup,
SystemExit at serve start) if an operator ever exports both keys.
