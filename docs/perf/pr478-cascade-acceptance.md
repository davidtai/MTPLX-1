# Speculative-cascade acceptance for Qwen3.8 Flash-Next

Speculative-cascade acceptance is a second opt-in, lossy decode-verify rule
beside typical acceptance. It decides per draft position whether the draft token
is good enough to keep, or whether to defer to the exact target law. It is OFF by
default, mutually exclusive with typical acceptance, and NOT distribution-exact.

Citation: Narasimhan, Mreddy, Jitkrittum, Rawat, Kumar, "Faster Cascades via
Speculative Decoding", ICLR 2025 / arXiv:2405.19261 v2. This implements the
plug-in deferral rule of Section 4.3, Equation (10), with the speculative
execution of Algorithm 4.

Terms:

- p: the target distribution at a position (the truncated top-p/top-k target
  row already materialized for the exact rule).
- q: the draft distribution at a position (the native MTP head's scored rows;
  see "What q is" below).
- D_TV(p, q) = sum_v max(0, p(v) - q(v)) over the scored top-k support.
- defer (r = 1): use the large/target model at this position; accept (r = 0):
  keep the small/draft model's token.
- alpha: the deferral cost (the operator knob).

## Problem

The exact speculative law commits the longest draft prefix the target would have
produced with the same coins, and no more. When the draft head is close to the
target it still spends a rejection whenever the exact coin `min(1, p/q)` happens
to fall, capping tokens per cycle. Typical acceptance loosens this by keeping
tokens that are "typical" under the target row, but it is blind to how well the
DRAFT itself matched the target: it can reject a token the draft was confidently
and correctly proposing, and it resamples from the target row rather than the
residual.

## Change

A speculative-cascade verify rule that, at each draft position with target p and
draft q, applies Equation (10):

    defer (r = 1)  <=>  max_v q(v) < max_v p(v) - alpha * D_TV(p, q)

If it does NOT defer, the draft is "good enough" (the speculative-cascade target
is pi = q, so Algorithm 4's accept probability min(1, pi/q) = 1): accept the
draft token with no coin. If it DOES defer, pi = p and the code runs the exact
speculative law unchanged -- accept with `min(1, p(x_t)/q(x_t))`, and on a coin
rejection resample the residual `norm(max(0, p - q))`. So the deferral test uses
the total variation over the scored top-k plus, on the deferred path, the token's
own target probability p(x_t), and the cascade path is a strict superset of the
exact rule with a draft-accept shortcut.

The decision is deterministic (it consumes no uniform); the coin only appears on
the deferred exact path, so with the lane off the RNG stream is byte-identical.

What q is: on the served Turbo path the draft is the native MTP head. Its scored
rows reach the verify loop as `draft_probs[depth_index]` -- a `SparseDistribution`
over the head's scored vocabulary, which for this pack is the FR-Spec
frequency-ranked subset (the same q the exact rule already uses in `min(1, p/q)`
and residual). The cascade rule reads q's peak (`max_v q`) and q's mass on the
scored top-k for D_TV from exactly that object; it introduces no new draft
distribution and needs no extra draft forward. q is required (the lane raises if
a non-greedy MTP position has no draft distribution), so the lane is a
temperature > 0, MTP-on rule, like typical acceptance.

Sign note (paper Lemma 3): alpha * D_TV is SUBTRACTED, so a larger disagreement
LOWERS the bar and defers LESS. A diverging draft that also loses peak confidence
(the realistic case) defers; a draft that stays confident on a wrong token is
accepted. This is the paper's documented behavior and is pinned in the tests.

## Effect

Higher acceptance when the draft distribution agrees with the target, including
on atypical tokens that typical acceptance would reject, at the cost of exactness
(the accepted draft is sampled from q, not p). It is a speed/quality dial, gated
on task-quality evals like typical acceptance, never on distribution-exactness.

## Exactness

NOT distribution-exact when on. The accepted-draft (non-defer) positions emit q's
token, which differs from the target law. The deferred positions ARE exact
(min(1, p/q) coin + residual). With the lane OFF (knob unset) the verify path is
byte-identical to the exact rule: the cascade branches are gated behind
`_cascade_active`, which is false, so neither the deterministic test nor any
extra coin runs.

## Files

- `mtplx/sampling.py` -- `cascade_defer_decision`, `total_variation`,
  `_peak_probability`.
- `mtplx/generation.py` -- `_cascade_accept_alpha` / `_cascade_accept_enabled`
  (env at use), `_assert_lossy_verify_rules_exclusive`, the two verify-loop
  decision branches (batched + lazy target), the `VerifyStats` cascade fields,
  and the `[cascade-accept]` verdict line.
- `mtplx/server/openai.py` -- `--cascade-threshold`, the env stamp + fail-loud
  mutual exclusion with `--typical-threshold`, and the `/health`
  `cascade_acceptance` install report.
- `tests/test_cascade_acceptance.py`, `tests/test_cascade_threshold_cli_health_cpu.py`.

## Switch

- `--cascade-threshold ALPHA` (env `MTPLX_FABLE_CASCADE_THRESHOLD`): the
  deferral cost alpha. UNSET (or blank) = OFF = exact speculative sampling (the
  default). Any set value -- including an explicit 0 -- turns the lane on, so the
  off switch is UNSETTING the key, not setting it to 0 (alpha = 0 still defers
  whenever the target is strictly more confident than the draft). Higher alpha
  widens the accept band, so fewer positions defer. Resolved at use, per request.
- Mutually exclusive with `--typical-threshold` (`MTPLX_FABLE_TYPICAL_THRESHOLD`
  > 0). Setting both fails loud at serve startup (SystemExit) and in the verify
  setup (ValueError).
- Observability: `/health` -> `cascade_acceptance` (`enabled`, `alpha`, the rule
  and divergence definitions, `conflict`); and a per-request verdict line
  `[cascade-accept] NOT distribution-exact; threshold=A alpha=A positions=N
  accepted=N resamples=N accept_rate=R mean_divergence=D ...` in the same format
  as `[typical-accept]`.
