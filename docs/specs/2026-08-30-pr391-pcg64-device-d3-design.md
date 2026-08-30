# PR #391 Exact PCG64 Device-D3 Design

## Objective

Remove the three draft-sampling GPU-to-host decision boundaries in each
stochastic M=4 verification cycle without changing the request's sampling,
acceptance, verifier, cache, or output semantics.  The first promotion unit is
the exact depth-three device draft chain alone.  Target acceptance and
correction remain on the host until this unit has passed parity and a corrected
16K-prefill/1K-output A/B at temperature 1.

The corrected control is 15.414148 seconds of decode and 66.5219 decode tokens
per second across three seeds.  Reaching 80 tokens per second requires a 12.8
second decode and approximately 2.614 seconds of total savings.  The measured
draft-boundary associated-time ceiling is 1.268243 seconds, including 1.022003
seconds classified as host-late.  This ceiling motivates the experiment but is
not a predicted saving and cannot reach the final target by itself.

## Semantic Contract

The installed lane must preserve:

- the NumPy PCG64 request stream and its observable cursor after every draw;
- exactly one uniform draw for every weighted choice;
- candidate ranking/filtering by descending logits and ascending token IDs for
  equal logits, followed by the current `SparseDistribution` canonicalization
  into ascending token-ID order before inverse-CDF selection;
- the already approved freedom at the top-k cutoff tie only;
- cumulative-before top-p filtering, positive-mass filtering, normalization,
  and `numpy.searchsorted(..., side="right")` selection;
- first-rejection behavior, residual correction, and all-accepted bonus choice;
- variable-length stopping, physical verifier width M=4, and a 16K production
  output cap;
- exact verifier-call counts, accepted-prefix ownership, cache state, and output
  digest on every no-cutoff-tie comparison.

The lane must not use `mx.random`, float32 uniforms, or an independent random
stream.  A non-tie sampling difference is a correctness failure, not a
performance tradeoff.

## Construction-Owned Random Tape

At request construction, copy the request PCG64 state into a second NumPy
generator and generate one contiguous float64 tape.  The conservative bound is
seven draws for every possible verification cycle plus one cycle of headroom:

```text
tape_draws = 7 * (max_output_tokens + 1)
```

Three draws cover the fixed D3 proposal chain.  At most four more cover three
accept tests plus either correction or bonus selection.  Every completed cycle
commits at least one token, so this bound covers a 16K request.  It occupies
917,560 bytes for 16,384 output tokens, plus the same device allocation: less
than 1 MiB on each side.

The cloned generator advances only to populate the tape.  The authoritative
request generator remains at its initial state.  A request-owned tape cursor
then advances both the logical tape position and the real NumPy generator at
runtime:

- `reserve_device_choices(3)` advances the real generator with
  `rng.random(3)` and returns the tape offset consumed by the compiled D3 chain;
- host acceptance continues to call the real generator's `random()`;
- host correction and bonus selection continue to call the real generator's
  `choice(..., p=...)`;
- every host draw advances the same logical tape cursor by one.

Consequently, the real generator state remains correct during generation,
after early stopping, and after cancellation.  The next D3 cycle indexes the
preinstalled tape after all conditionally consumed host draws from preceding
cycles.  No end-of-request state repair is required.

The request-owned cursor is functional state, not proof instrumentation.  The
exact lane receives it at construction and is the sole RNG capability used by
that lane.  Unsupported bit generators, an output allowance above 16K, sampler
geometry outside the installed contract, and incompatible draft features fail
once before measured decode.  There is no enabled-lane fallback.

## Exact Device D3 Chain

The D3 strategy is selected and installed before the decode loop.  Installation
validates the model, MTP depth, committed-history cache, sampler geometry, tape
bound, and device arithmetic once.  The hot cycle calls the installed strategy
directly; it contains no eligibility branch, environment read, route repair,
exception fallback, or engagement counter.

Installation promotes the single QSA MTP-history cache once with capacity for
the request's bounded output plus D3 headroom.  This prevents ordinary cycles
from comparing cache signatures or recompiling after a backing-shape change.
For the exact Qwen3.8 geometry, the attention K/V, raw index key, and pooled
index key banks cost 2,368 bytes per reserved logical token: a 16K grant is
37.0 MiB total and 27.75 MiB larger than the existing 4K device-core grant.
Construction must verify these leaf shapes/dtypes and report actual allocation
bytes before the full-model benchmark.  Requests configured to replace this
history cache during the reachable output span cannot install the lane.

For levels one through three, one compiled graph performs:

1. the level's MTP forward;
2. exact rank-order support construction and top-p filtering, followed by the
   current ascending-token-ID sparse-support canonicalization;
3. exact probability normalization and cumulative distribution construction;
4. inverse-CDF selection from the installed tape at `cursor + level - 1`;
5. direct use of the selected token as the following MTP level's input.

The graph returns the three tokens and their proposal distributions for the
unchanged host target-verification stage.  It evaluates once after the third
level rather than materializing each level's support and token separately.

The existing `_device_draft_q_arrays` route is not presumed correct for this
lane: its float32 normalization is only approximately equal to the current
host path.  Before the strategy can be installed, a guarded feasibility test
must prove that Metal-backed arithmetic reproduces the host support, cumulative
probabilities, and selected token on captured production rows.  Exposing
`float64` in the MLX API is not proof that the Metal graph supports it or that
it matches NumPy.

If native device arithmetic cannot meet this contract, Stage 1 stops without a
fallback.  A custom two-word integer or fixed-point CDF would be a separately
designed experiment; silently casting to float32 is prohibited.

## Implementation Boundaries

Stage 1 owns:

- a small request-tape/cursor type in the sampling layer;
- construction-time installation of the exact D3 strategy;
- a dedicated stochastic D3 entrypoint in generation;
- CPU cursor/choice regression tests and guarded device-D3 parity tests;
- an isolated corrected control/candidate benchmark receipt.

Stage 1 does not move accept/reject, first rejection, residual correction,
bonus selection, or the following commit graph to the device.  Those operations
form Stage 2 and are designed only after Stage 1 has an individual A/B result.
The unattributed 112-event handoff and compact FRSpec sampling remain later,
separate candidates.

## Test-First Gates

Before production code changes, add failing tests for:

1. cloned-tape values versus the authoritative PCG64 stream;
2. token and subsequent PCG64-state equality between `rng.choice` and the exact
   inverse-CDF rule over support sizes 1 through 40, including singleton and
   zero-mass-filtered supports;
3. cursor progression across draft, partial acceptance plus correction, full
   acceptance plus bonus, EOS, early stop, and cancellation sequences;
4. the 16K tape bound and rejection of larger production output allowances;
5. construction-time refusal of incompatible RNG and sampler contracts;
6. absence of `mx.random` and runtime fallback from the installed D3 strategy;
7. exact device rank order, cumulative-before top-p, positive filtering,
   final ascending-token-ID support order, normalization, selected tokens, and
   returned proposal distributions;
8. exact three-level tokens, real PCG64 state, verifier count, accepted-prefix
   trajectory, cache leaves, and output digest against the host draft loop.

Approximate probability assertions are insufficient for the promotion gate.
Cutoff-tie fixtures must separately prove that any allowed divergence begins at
the exact top-k cutoff tie.

## Measurement and Promotion

All MLX work follows `docs/GPU_LOCK_AND_SERVICE_RUNBOOK.md`: acquire
`/tmp/mtplx-gpu-exclusive.lock` before unloading or loading, account for wired
memory and the candidate peak, keep the guard alive through the child, restore
the exact production service, verify health, and only then release the
workflow.

After focused parity, run an interleaved corrected control/candidate comparison
on the exact model and artifact:

- 16K prefill;
- 1K generated output;
- temperature 1, top-k 20, top-p 0.95;
- adaptive verifier reserve starting at 1,024;
- three matching seeds.

Report every decode time/TPS, mean wall time/TPS, verifier and compiled-M4
counts, acceptance/correction census, output digest, RNG-state digest, active
memory, and peak memory.  Retain Stage 1 only if same-work performance improves
repeatably with exact parity.  A favorable trajectory with different verifier
work is not a win.

The PR body remains unchanged until the final corrected stack exceeds 80 TPS
on this three-seed gate with parity evidence.  Stage 1 success alone is not
completion because its measured ceiling is below the required total saving.

## Failure-Mode Review

- **Critical: device probability arithmetic differs from NumPy.** Prove exact
  captured-row CDF and token parity before integration; stop the candidate if
  Metal cannot provide it.
- **Critical: a host RNG call bypasses the tape cursor.** Give the installed
  lane one request-owned RNG capability and cover every conditional sequence
  with state comparisons.
- **Critical: predrawing changes the externally visible PCG64 state.** Generate
  with a clone and advance the authoritative generator incrementally for each
  consumed tape element.
- **Critical: the D3 graph changes proposal distributions used by acceptance.**
  Require exact returned support/probability arrays, accepted-prefix trajectory,
  verifier counts, and digest parity.
- **Critical: MTP-history cache replacement invalidates the compiled graph.**
  Reserve the reachable 16K span once and refuse reset/rebase policies that can
  replace the captured cache during the request; never poll signatures in the
  ordinary cycle.
- **Critical: a hot fallback hides installation or shape failure.** Validate
  once, install a fixed strategy, and fail before decode if its invariants do
  not hold.
- **Minor: construction reserve increases memory.** The bounded tape is below
  1 MiB per host/device copy; the Qwen3.8 MTP cache is 37.0 MiB at 16K, or
  27.75 MiB above the former 4K grant. Both must be verified and reported.
- **Minor: predrawing itself does not improve TPS.** Attribute any win to the
  removed dependency boundaries; RNG generation was measured at only about
  3.5 milliseconds for all 1,146 draft choices.
