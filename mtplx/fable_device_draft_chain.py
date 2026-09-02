"""Compiled, device-resident D1->D3 draft chain (``MTPLX_FABLE_DEVICE_DRAFT_CHAIN``).

Read ONCE at import, default OFF.  Flag-off, :func:`is_enabled` is False, no
plan is ever built, every call site in ``generation.py`` stays behind a
module-level constant, and the stock lane evaluates exactly the expressions it
evaluated before this module existed.

The measurement this answers
----------------------------
Two dispatch censuses of the retained/composed stack (2026-09-02) agree on the
shape of the decode cycle's idle time::

    draft-head dispatches/cycle 3.00 in 3.00 command buffers/cycle
    idle gaps strictly inside the draft window: 4.07/cycle, 1.199 ms/cycle
    v_Expfloat32float32 -> gather_frontbfloat16_int32_int_2
        1188 events / 394 cycles = 3.01 per cycle, 1030.6 ms = 2.62 ms/cycle

``v_Exp`` is the last op of a draft depth's K20 support
(``fast_sampling._device_serial_support_arrays``'s
``mx.exp(cand_vals - log_total)``); ``gather_front(bfloat16, int32)`` is the
FIRST op of the next forward (the token embedding gather).  So each of those
three gaps is one whole host round trip: GPU drain, ``np.asarray``, host
lexsort + top-p mask + ``rng.choice``, **and the Python construction of the
next depth's MTP graph**.

Two prior attempts removed only part of that
--------------------------------------------
* *restack "lazy stochastic D3"* built all three depths' lazy graphs before one
  device read: 64.99 vs 67.63 tok/s (-3.9%).  It moved the three Python graph
  constructions to the front of the cycle but did not remove them, and it
  handed ``mx.eval`` a 3x deeper uncompiled graph to plan.
* *W24/W28b ``MTPLX_FABLE_DEVICE_K20``* (``fable_device_k20.DeviceDraftChain``)
  really did collapse the three per-depth ``mx.eval``/``np.asarray`` syncs into
  one -- the code has no per-depth readback -- and still cost +1.14% cycle time
  (ABBA, 3 seeds, exact digests).  What it did **not** remove is the ~300
  host-issued MLX ops per depth, and what it **added** was an exact two-stage
  device selector plus a full ``logsumexp`` over the **248,320-lane scattered**
  FR-Spec row (the ``put_along_axis`` output), three times per cycle, on the
  serial critical path -- plus a per-cycle ``Fraction``-based PCG64 descriptor
  build (``build_pcg64_midpoint_descriptors`` evaluates
  ``_descriptor_for_grid_integer`` **twice** per uniform: once to build, once
  to re-validate).  That last one is small: measured 0.033 ms/cycle at depth 3
  against 0.015 ms for the vectorised form here, so ~0.017 ms of W28b's
  +0.43 ms.  It is fixed because it is free to fix, not because it explains
  anything.

So the readback was never the whole 2.6 ms.  The dominant term is host graph
construction, and neither attempt touched it.

What this module does differently
---------------------------------
1. **Compiles the per-depth body.**  ``fable_compiled_draft`` already builds
   exactly this (one ``mx.compile`` trace, replayed ``depth`` times, MTP cache
   captured as both ``inputs=`` and ``outputs=``) -- but its only call site is
   ``generation._pr391_make_float32_d3_core``, and the PR391 route is installed
   *only* by ``scripts/pr391_metal_choice_benchmark_launcher.py``.  On the
   serving / ABBA lane the flag has therefore always been **inert**: window 6's
   "+0.15% neutral" and window 33's composed stack both measured a no-op (W15
   found the same thing for the K20 logger).  This module is the first thing
   that puts a compiled draft body on the measured lane.
2. **Selects on the 65,536-row pre-scatter row.**  W42/W42b proved the FR-Spec
   head's compact row yields the identical K20 support (strictly ascending
   ranked table => value-desc/id-asc tie-break preserved; ``-1e30`` sentinels
   underflow to exactly ``+0.0`` in float32 so the ``logsumexp`` normaliser is
   value-identical).  Taking the stash inside the **traced** body wires the
   compiled graph straight to the compact row, so the 248,320-lane
   ``mx.full`` + ``put_along_axis`` is built and dropped, never executed.
3. **Samples on device from the PCG64 tape.**  The W24 sampler, parity-clean on
   all five counters at window 28a.
4. **One readback per cycle** -- ``mx.eval`` of the four small chain outputs.
5. **A vectorised descriptor build** (:func:`fast_midpoint_descriptors`),
   bit-identical to ``build_pcg64_midpoint_descriptors`` and free of
   ``Fraction``.

Modes
-----
``MTPLX_FABLE_DEVICE_DRAFT_CHAIN=1`` (or ``chain``) -- the full chain above,
ONE readback per cycle.

``MTPLX_FABLE_DEVICE_DRAFT_CHAIN=body`` -- the **attribution arm**: the same
compiled per-depth body, but the K20 support comes back to the host at every
depth and the token is drawn by the stock host sampler.  Three readbacks per
cycle, so it prices the *compiled-body* lever alone against the *one-readback*
lever, and it is the fallback if ``chain`` loses.  It is NOT a bit-identical
arm -- see below; an earlier version of this docstring said it was, and window
1788400641 falsified that on 3/3 seeds.

Exactness -- NEITHER MODE IS BIT-IDENTICAL
------------------------------------------
Both modes are **rounding class**, the same bucket as HC_M4, prefill chunk
4096 and GDN-blocked prefill.  They must be judged on the program's
rounding-class gate (HumanEval n=164, or the long-prompt agreement screen),
never on digest equality.  Window 1788400641 (``body``, 3 seeds) confirmed the
digests differ; that is the expected behaviour, not a defect.

Three independent rounding sources, largest first:

1. **``mx.compile`` on the MTP DecoderLayer forward** (BOTH modes).
   ``fable_compiled_draft``'s own contract says so: "the same draft tokens for
   the same inputs, *up to ``mx.compile`` rounding* (bf16 regrouping is
   expected and accepted; **bit-for-bit digest parity is not a goal of this
   flag**)".  It is unavoidable, because the compiled body *is* the lever --
   the fusion that removes the host issuance is the same fusion that regroups
   the arithmetic.  Excluding the fusable elementwise tail would exclude the
   norms, the gates and both residual mixers, i.e. the body.
2. **``logsumexp`` over 65,536 lanes instead of 248,320** (both modes).  W42's
   one admitted residual: the sentinel terms are exactly ``+0.0`` but the two
   reductions partition the real terms differently.
   ``scripts/fable/micro_draft_k20.py`` measured one 1-ULP row in 32 on Metal.
   It can only move a knife-edge top-p ``cumulative_before`` comparison.
3. **The float32 row preparation and CDF walk** (``chain`` only).  W24's
   documented divergence, kept here: the device prepares the row in float32
   where the host prepares it in float64.

*What IS exact.*  The K20 support SET and its ORDER:
``_deterministic_mlx_top_k_support`` + ``_order_bounded_mlx_top_k_support`` is
the exact (value desc, id asc) selector over the whole row -- the same contract
the stock builder's ``argpartition``-to-80 hot path implements, and the same
selector the stock builder falls back to when its superset spills -- and local
rows are mapped through the strictly ascending ranked table, which preserves
the order and every tie-break (W42 point 3).  ``host_support_tail`` then
reproduces the stock float64 tail exactly on those rows.  So given the same
support row, ``body`` mode draws the same token as stock; what differs is the
row, because the forward that produced it was compiled.  That is what
``tests/test_fable_device_draft_chain.py`` pins, and it is all it pins.

*Observed.*  On window 1788400641 the perturbation is broad and small rather
than a rare knife-edge: ``accepted_by_depth`` moved on every seed
(259,187,120 -> 265,187,119 and 275,184,117 -> 277,187,120), depth-1 accepts up
on both clean seeds, tok/M4win 2.5079 -> 2.4987 and 2.5064 -> 2.5349 --
directionally neutral, which is what a rounding-class proposal change looks
like.

*Draw accounting (both modes).*  Flag-off the stock lane draws exactly one
float64 per draft depth, in depth order, inside ``rng.choice``
(``sampling.py:303`` -> NumPy ``Generator.choice`` with ``p=`` and
``size=None`` -> one ``self.random()``), before any accept coin.  Both modes
here take ``rng.random(depth)`` up front -- the same doubles, in the same
order, from the same stream -- so the PCG64 cursor after a cycle is where
flag-off leaves it and every later accept coin, residual correction and bonus
draw is unshifted.  A cycle the chain does NOT run (a context-copy streak
substitution) draws nothing, exactly as flag-off.

*Token law.*  Every source above changes only the draft PROPOSAL ``q``, never
the accept/correct law: the accept loop is fed the same row the draft was
sampled from (``host_support_tail`` in ``body``,
:func:`fable_device_k20.draft_distribution` in ``chain``), so
``q_sample == q_test`` and the emitted distribution stays the target's.
Speculative sampling admits a different ``q`` for free -- it costs acceptance,
not correctness -- which is why the gate is a task eval and not a digest.

NO device work happens at import.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .fable_device_k20 import (
    draft_distribution as _draft_distribution,
    draw_draft_uniforms as _draw_draft_uniforms,
)
from .fable_claim_contract import (
    ClaimDeclined,
    decline as _decline,
    declined_receipt,
)
from .sampling import SamplerConfig, SparseDistribution


_ENV_VAR = "MTPLX_FABLE_DEVICE_DRAFT_CHAIN"

#: ``chain`` = one readback per cycle (device sampler).
#: ``body``  = compiled body only, stock host selection + sampler per depth.
MODE_CHAIN = "chain"
MODE_BODY = "body"
MODE_OFF = "off"

#: The only ranked-table width W42's exactness argument covers.
FRSPEC_ROWS = 65_536

#: The fixed draft support width this route serves.
TOP_K = 20

#: The choice kernel and the PCG64 tape are both pinned to this NumPy.
REQUIRED_NUMPY_VERSION = "2.4.4"

#: Extra MTP rows the promotion reserves beyond ``max_tokens``: the cycle's own
#: ``depth`` draft rows plus the primary.  Mirrors PR391's ``max_tokens + 4``.
_RESERVE_SLACK = 4

_TRUTHY = {"1", "true", "yes", "on"}


def _resolve_mode(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return MODE_OFF
    if value in _TRUTHY or value == MODE_CHAIN:
        return MODE_CHAIN
    if value == MODE_BODY:
        return MODE_BODY
    if value in {"0", "false", "no", "off"}:
        return MODE_OFF
    raise ValueError(
        f"{_ENV_VAR}={raw!r} is not a mode; use 1/chain, body, or leave it unset"
    )


#: Read exactly once, at import.
_MODE = _resolve_mode(os.environ.get(_ENV_VAR))


#: Engagement counters.  Zero on every flag-off run; a receipt with
#: ``cycles == 0`` on an armed run means the route never engaged.
COUNTERS: dict[str, int] = {
    "claims": 0,
    "cycles": 0,
    "depths": 0,
    "readbacks": 0,
    "tape_draws": 0,
    "cache_promotions": 0,
    "cache_demotions": 0,
    "chain_builds": 0,
    "cache_rebinds": 0,
    # Cycles the route declined because `cycle_depth` was shorter than the
    # depth the body was traced at (the last cycle of a request, when
    # `max_tokens - len(tokens) < planned_depth`).  Those run the stock loop.
    "short_cycles": 0,
    # The two per-cycle MTP-offset reads the cache promotion would otherwise
    # turn into command-queue drains (see the section at the bottom of this
    # module).  `declined` counts the cycles that fell back to the stock
    # host-read rollback because `rollback_state` held row snapshots.
    "offset_reads_kept_on_device": 0,
    "offset_rollbacks_on_device": 0,
    "offset_rollback_declined": 0,
    "refusals": 0,
    # Requests this lane stood aside for because their SHAPE is not one it
    # serves (greedy, penalties, a competing owner).  Those run the stock
    # draft loop; `refusals` counts only install-time contract violations,
    # which raise.  See `mtplx.fable_claim_contract`.
    "declines": 0,
}


def reset_counters() -> None:
    """Zero the engagement counters (tests and per-process receipts)."""

    for key in COUNTERS:
        COUNTERS[key] = 0


def mode() -> str:
    """``"chain"``, ``"body"`` or ``"off"`` -- resolved once, at import."""

    return _MODE


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_DEVICE_DRAFT_CHAIN`` armed a mode at import."""

    return _MODE != MODE_OFF


def _configure_for_test(mode_name: str) -> None:
    """Flip the import-time gate (tests only)."""

    global _MODE
    if mode_name not in {MODE_CHAIN, MODE_BODY, MODE_OFF}:
        raise ValueError(f"unknown mode {mode_name!r}")
    _MODE = mode_name


class DeviceDraftChainIneligible(RuntimeError):
    """The armed flag cannot work in THIS PROCESS at all.

    Reserved for install-time contract violations -- the pinned NumPy, the
    PCG64 tape, the FR-Spec head's presence/shape/liveness -- plus the
    mid-decode guards that arm after a claim already succeeded.  Every request
    would fail identically on the first class.

    A request whose SHAPE this lane does not serve (greedy, penalties, a
    competing owner of the draft chain, ...) DECLINES to the stock draft loop
    instead -- see :mod:`mtplx.fable_claim_contract`.  Raising on those turns
    every ineligible request into an HTTP 500 in serving.
    """


def _refuse(reason: str) -> None:
    """Install-time contract violation: no request here could be served."""

    COUNTERS["refusals"] += 1
    raise DeviceDraftChainIneligible(f"{_ENV_VAR}: {reason}")


# ---------------------------------------------------------------------------
# PCG64 midpoint descriptors, without Fraction
# ---------------------------------------------------------------------------


_F32_SIGNIFICAND_BITS = 23
_F32_EXPONENT_BIAS = 150  # 127 + 23
_PCG64_HIGH_MASK = (1 << 21) - 1


def fast_midpoint_descriptors(uniforms: np.ndarray) -> np.ndarray:
    """Vectorised ``build_pcg64_midpoint_descriptors``.

    The shipped builder spends two ``Fraction`` constructions and a
    ``validate`` pass -- which recomputes the whole descriptor -- per uniform,
    i.e. six exact-rational descriptor derivations per decode cycle, on the
    serial critical path.  This is the same function in closed form.

    Derivation.  ``u`` is on the exact 53-bit PCG64 grid, so ``float64(u)`` is
    exact and ``Fraction(u) == Fraction(int(u * 2**53), 2**53)``; the shipped
    ``_fraction_from_float(rounded) > exact_uniform`` test is therefore exactly
    ``float64(float32(u)) > u``.  ``lower`` and ``upper`` are adjacent
    binary32s, and adjacent binary32s always satisfy ``upper = lower + 2**e``
    where ``e`` is ``lower``'s dyadic exponent -- including the
    significand-carry case (``0xFFFFFF * 2**e`` -> ``0x800000 * 2**(e+1)``) and
    the subnormal/zero case (both use ``e = -149``).  Hence

        midpoint = lower + 2**(e-1) = (2 * significand(lower) + 1) * 2**(e-1)

    whose numerator is odd, so the fraction is already in lowest terms and the
    denominator is a power of two -- the two invariants the shipped builder
    asserts.

    ``tests/test_fable_device_draft_chain.py`` pins this against the shipped
    builder over random and adversarial uniforms, word for word.
    """

    values = np.asarray(uniforms)
    if values.dtype != np.dtype(np.float64):
        raise ValueError("uniforms must have dtype float64")
    if values.ndim != 1:
        raise ValueError("uniforms must be one-dimensional")
    if values.size and (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values >= 1.0)
    ):
        raise ValueError("uniforms must be finite and in [0, 1)")

    scaled = np.ldexp(values, 53)
    integers = scaled.astype(np.uint64)
    if not np.array_equal(scaled, integers.astype(np.float64)):
        raise ValueError("uniforms must lie on the exact PCG64 53-bit grid")

    rounded = values.astype(np.float32)
    widened = rounded.astype(np.float64)
    # `rounded` is never negative and never inf: values < 1 round to <= 1.0f.
    step_up = widened <= values
    upper = np.where(
        step_up,
        np.nextafter(rounded, np.float32(np.inf), dtype=np.float32),
        rounded,
    ).astype(np.float32)
    lower = np.where(
        step_up,
        rounded,
        np.nextafter(rounded, np.float32(-np.inf), dtype=np.float32),
    ).astype(np.float32)

    lower_bits = lower.view(np.uint32)
    exponent_bits = (lower_bits >> _F32_SIGNIFICAND_BITS) & np.uint32(0xFF)
    fraction_bits = lower_bits & np.uint32((1 << _F32_SIGNIFICAND_BITS) - 1)
    subnormal = exponent_bits == 0
    lower_significand = np.where(
        subnormal,
        fraction_bits.astype(np.uint64),
        (fraction_bits | np.uint32(1 << _F32_SIGNIFICAND_BITS)).astype(np.uint64),
    )
    lower_exponent = np.where(
        subnormal,
        np.int64(-149),
        exponent_bits.astype(np.int64) - np.int64(_F32_EXPONENT_BIAS),
    )

    significand = (lower_significand * np.uint64(2) + np.uint64(1)).astype(np.uint64)
    exponent = (lower_exponent - np.int64(1)).astype(np.int64)
    if significand.size and int(significand.max()) > np.iinfo(np.uint32).max:
        raise AssertionError("binary32 midpoint significand must fit uint32")

    upper_even = ((upper.view(np.uint32) & np.uint32(1)) == 0).astype(np.uint32)

    descriptors = np.empty((values.size, 5), dtype=np.uint32)
    descriptors[:, 0] = (
        (integers >> np.uint64(32)) & np.uint64(_PCG64_HIGH_MASK)
    ).astype(np.uint32)
    descriptors[:, 1] = (integers & np.uint64(0xFFFF_FFFF)).astype(np.uint32)
    descriptors[:, 2] = significand.astype(np.uint32)
    descriptors[:, 3] = exponent.astype(np.int32).view(np.uint32)
    descriptors[:, 4] = upper_even
    return descriptors


# ---------------------------------------------------------------------------
# Host tail -- the stock draft read, on an exact support
# ---------------------------------------------------------------------------


def host_support_tail(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``_device_serial_support_arrays``' host tail, on ``top_k`` exact rows.

    Transcribes ``fast_sampling.py``'s tail exactly: under top-p the float32
    probabilities widen to float64 and the float64 ``cumulative_before`` mask
    zeroes everything at or past the nucleus; otherwise the ``k`` values are
    softmaxed in float64 after subtracting the row max.  The rows arrive
    already ordered (value desc, id asc), which is what the stock lane's
    ``np.lexsort`` produces, so no re-sort is needed or wanted.
    """

    token_rows = np.asarray(ids, dtype=np.int64).reshape(1, -1)[:, :top_k]
    if 0.0 < float(top_p) < 1.0:
        prob_rows = np.asarray(probs, dtype=np.float64).reshape(1, -1)[
            :, :top_k
        ].copy()
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(cumulative_before < float(top_p), prob_rows, 0.0)
    else:
        vals64 = (
            np.asarray(values, dtype=np.float32)
            .reshape(1, -1)[:, :top_k]
            .astype(np.float64)
        )
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)
    return token_rows, prob_rows


# ---------------------------------------------------------------------------
# Construction-time plan
# ---------------------------------------------------------------------------


@dataclass
class DeviceDraftChainPlan:
    """One request's bound compiled draft-chain route.

    Construction validates every request-invariant term and arms the FR-Spec
    head; the compiled body itself is bound LAZILY, to the MTP cache container
    the decode loop actually drafts into.  That container is not known at
    construction (``generation.generate_mtpk`` picks it per cycle) and it can
    legitimately be REPLACED mid-request -- by the committed-history live reset
    (``mtp_history_cache = rt.make_mtp_cache()``) or by a prefix rebase
    (``mtp_history_cache = rebased.committed_mtp_cache``).  ``mx.compile``
    captures the container, so a swap must rebind, not silently keep writing
    the old cache.  :meth:`ensure_bound` does that and counts it.
    """

    mode: str
    depth: int
    top_k: int
    draft_temperature: float
    draft_top_p: float
    frspec_rows: int
    vocab_rows: int
    head: Any
    route: str
    reserve_tokens: int
    build_chain: Callable[[Any, Any], dict[str, Any]]
    state_tree_fn: Callable[[Any], Any]
    promote_fn: Callable[..., tuple[int, dict[str, int]]]
    mtp_cache: Any = None
    chain: dict[str, Any] | None = None
    promoted: int = 0
    builds: int = 0
    released: bool = False
    receipt_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def top_p_active(self) -> bool:
        return 0.0 < float(self.draft_top_p) < 1.0

    @property
    def readbacks_per_cycle(self) -> int:
        return 1 if self.mode == MODE_CHAIN else int(self.depth)

    def ensure_bound(self, mtp_cache: Any) -> None:
        """Bind (or rebind) the compiled body to ``mtp_cache``."""

        if self.mtp_cache is mtp_cache and self.chain is not None:
            return
        if not mtp_cache or len(mtp_cache) != 1:
            _refuse("this route requires exactly one MTP cache entry")
        rebind = self.chain is not None
        promoted, failures = self.promote_fn(
            mtp_cache,
            reserve_tokens=self.reserve_tokens,
            preserve_paged=True,
            initial_reserve_tokens=self.reserve_tokens,
        )
        if failures:
            _refuse(f"MTP cache promotion failures: {failures}")
        COUNTERS["cache_promotions"] += int(promoted)
        self.promoted += int(promoted)
        self.mtp_cache = mtp_cache
        self.chain = self.build_chain(mtp_cache, self.state_tree_fn(mtp_cache))
        self.builds += 1
        COUNTERS["chain_builds"] += 1
        if rebind:
            COUNTERS["cache_rebinds"] += 1
            print(
                "[mtplx] fable-device-draft-chain rebind: the live MTP cache "
                f"container was replaced; rebuilt the compiled body "
                f"(builds={self.builds})",
                file=sys.stderr,
                flush=True,
            )

    def to_dict(self) -> dict[str, Any]:
        stats = self.chain.get("trace_stats", {}) if self.chain else {}
        return {
            "installed": True,
            "mode": str(self.mode),
            "depth": int(self.depth),
            "top_k": int(self.top_k),
            "frspec_rows": int(self.frspec_rows),
            "vocab_rows": int(self.vocab_rows),
            "route": str(self.route),
            "reserve_tokens": int(self.reserve_tokens),
            "cache_promotions": int(self.promoted),
            "chain_builds": int(self.builds),
            "readbacks_per_cycle": int(self.readbacks_per_cycle),
            "body_traces": int(stats.get("body_traces", 0)),
            "counters": dict(COUNTERS),
            **self.receipt_extra,
        }


def claim_request_route(
    *,
    receipt: dict[str, object] | None = None,
    **request: Any,
) -> DeviceDraftChainPlan | None:
    """Bind the compiled draft chain to one generation.

    Returns ``None`` when the flag is off, and ``None`` again when the flag is
    on but this REQUEST's shape is not one the route serves -- a decline, not
    a failure: the stock draft loop runs and produces the same tokens.  When
    ``receipt`` is passed it is filled in with why.  Only an INSTALL-time
    contract violation raises :class:`DeviceDraftChainIneligible`.
    ``MTPLX_FABLE_STRICT_CLAIMS=1`` turns declines back into that exception
    for a measured arm that must prove the lane ran.

    The request terms are spelled out on :func:`_claim_request_route`; this
    wrapper forwards them untouched so there is exactly one list of them.
    """

    if _MODE == MODE_OFF:
        return None
    try:
        return _claim_request_route(**request)
    except ClaimDeclined as declined:
        COUNTERS["declines"] += 1
        stamped = declined_receipt(
            _ENV_VAR, declined, ineligible=DeviceDraftChainIneligible
        )
        if receipt is not None:
            receipt.clear()
            receipt.update(stamped)
        return None


def _claim_request_route(
    *,
    rt: Any,
    state_tree_fn: Callable[[Any], Any],
    promote_fn: Callable[..., tuple[int, dict[str, int]]],
    mtp_hidden_variant: str,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    speculative_depth: int,
    request_max_tokens: int,
    rng: Any,
    draft_core: str,
    mtp_cache_policy: str,
    mtp_history_policy: str,
    mtp_position_mode: str,
    target_prefix_verify: bool,
    lazy_target_distributions: bool,
    lazy_bonus_verify_requested: bool,
    batch_target_arrays: bool,
    steer_active: bool,
    penalties_active: bool,
    relaxed_draft_ties: bool,
    qsa_mtp_precompute_active: bool,
    constraint: Any,
    adaptive_policy: Any,
    adaptive_width_policy: Any,
    mtp_corrector: Any,
    mtp_topk_reranker: Any,
    draft_margin_threshold: float | None,
    wants_policy_metrics: bool,
    draft_confidence_needed: bool,
    online_hidden_corrector_alpha: float,
    online_correction_cache: bool,
    prompt_correction_cache: bool,
    adapter_ensemble_q: bool,
    combine_greedy_draft_read: bool,
    greedy_chain_enabled: bool,
    adaptive_dtemp_active: bool,
    frspec_legacy_ids: Any,
    late_depth_switch_after: int,
    a3b_target_prefix_route: Any,
    pr391_route: Any,
    device_k20_route: Any,
    draft_k20_prescatter_route: Any,
    depth4_probe_active: bool,
    k20_log_active: bool,
    ple_candidate_submit: Any,
) -> DeviceDraftChainPlan | None:
    """The claim body.  ``_refuse`` raises; ``_decline`` stands aside."""

    if _MODE == MODE_OFF:
        return None

    # -- environment ------------------------------------------------------
    if np.__version__ != REQUIRED_NUMPY_VERSION:
        _refuse(
            f"the PCG64 tape is pinned to NumPy {REQUIRED_NUMPY_VERSION}; "
            f"found {np.__version__}"
        )
    if type(rng) is not np.random.Generator:
        _refuse("this route requires a numpy.random.Generator")
    if type(rng.bit_generator) is not np.random.PCG64:
        _refuse("this route requires a PCG64 bit generator")

    # -- competing owners of the draft chain -------------------------------
    if pr391_route is not None:
        _decline(
            "pr391_owns_chain",
            "the PR391 float32 D3 route already owns the draft chain",
        )
    if a3b_target_prefix_route is not None or target_prefix_verify:
        _decline(
            "target_prefix_verify",
            "target-prefix verification samples drafts on device",
        )
    if device_k20_route is not None:
        _decline(
            "device_k20_owns_selector",
            "MTPLX_FABLE_DEVICE_K20 owns the draft selector",
        )
    if draft_k20_prescatter_route is not None:
        _decline(
            "prescatter_owns_stash",
            "MTPLX_FABLE_DRAFT_K20_PRESCATTER owns the FR-Spec pre-scatter "
            "stash; this route consumes it inside the compiled body",
        )
    if str(draft_core) != "stock":
        _decline(
            "non_stock_draft_core",
            f"this route requires the stock draft selector (got {draft_core!r})",
        )
    if greedy_chain_enabled or combine_greedy_draft_read:
        _decline(
            "greedy_chain", "the greedy draft chain owns the per-depth read"
        )

    # -- sampler contract ---------------------------------------------------
    if float(sampler.temperature) <= 0.0 or float(draft_sampler.temperature) <= 0.0:
        _decline(
            "greedy_request",
            "this is a sampled-lane route (temperature > 0); this request is "
            f"target t={float(sampler.temperature)!r} / draft "
            f"t={float(draft_sampler.temperature)!r}",
        )
    if int(draft_sampler.top_k) != TOP_K:
        _decline("top_k_mismatch", f"this route is fixed at top_k={TOP_K}")
    if not 0.0 < float(draft_sampler.top_p) <= 1.0:
        _decline("top_p_range", "this route requires 0 < draft top_p <= 1")
    if (
        float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
        or penalties_active
        or steer_active
    ):
        _decline(
            "steer_or_penalties",
            "steering/penalty overlays index by real token id",
        )
    if adaptive_dtemp_active:
        _decline(
            "adaptive_dtemp",
            "MTPLX_ADAPTIVE_DTEMP rebinds the draft sampler mid-generation; "
            "the compiled body and the choice kernel bake one temperature",
        )
    if relaxed_draft_ties:
        _decline(
            "relaxed_draft_ties",
            "MTPLX_QWEN4_RELAXED_DRAFT_TIES installs a different builder",
        )

    # -- loop shape ---------------------------------------------------------
    depth = int(speculative_depth)
    if depth < 1:
        _decline("non_positive_depth", "this route requires a positive draft depth")
    if int(late_depth_switch_after) != 0:
        _decline(
            "late_depth_switch", "this route does not admit late-depth switching"
        )
    if str(mtp_cache_policy) != "persistent":
        _decline(
            "mtp_cache_policy",
            "the compiled body captures ONE MTP cache; "
            f"mtp_cache_policy={mtp_cache_policy!r} makes a fresh one per depth",
        )
    if str(mtp_history_policy) != "committed":
        _decline(
            "mtp_history_policy",
            "this route requires the committed MTP history policy "
            f"(got {mtp_history_policy!r})",
        )
    if str(mtp_position_mode or "default").strip().lower().replace("-", "_") not in {
        "",
        "default",
        "cache",
    }:
        _decline(
            "mtp_position_mode",
            "the compiled body forwards no position_offset; "
            f"MTPLX_MTP_POSITION_MODE={mtp_position_mode!r} needs one",
        )
    if qsa_mtp_precompute_active:
        _decline(
            "qsa_mtp_precompute",
            "the QSA MTP indexer precompute stages replay caches per cycle; "
            "the compiled trace captures a fixed state tree",
        )
    # Deliberately NOT refused: `batch_target_arrays`, `lazy_target_distributions`
    # and `lazy_bonus_verify_requested`.  All three are TARGET-side, and this
    # route replaces only the draft chain: it hands the loop the same
    # `draft_tokens` (host ints) and `draft_probs` (SparseDistribution) the
    # per-depth loop would have produced, at the same point in the cycle, so
    # everything downstream -- including the lazy-bonus width decision, which
    # reads `draft_tokens[:-1]` -- sees an indistinguishable cycle.
    # `fable_device_k20` refuses them because it also replaced the TARGET
    # support builder; this one does not.  The receipt records what the request
    # had so a reader can still tell the arms apart.
    target_side = {
        "batch_target_arrays": bool(batch_target_arrays),
        "lazy_target_distributions": bool(lazy_target_distributions),
        "lazy_bonus_verify_requested": bool(lazy_bonus_verify_requested),
    }

    unsupported = {
        "adapter_ensemble_q": bool(adapter_ensemble_q),
        "adaptive_policy": adaptive_policy is not None,
        "adaptive_width_policy": adaptive_width_policy is not None,
        "constraint": constraint is not None,
        "depth4_probe": bool(depth4_probe_active),
        "draft_confidence": bool(draft_confidence_needed),
        "draft_margin_threshold": draft_margin_threshold is not None,
        "frspec_legacy": frspec_legacy_ids is not None,
        "k20_log": bool(k20_log_active),
        "mtp_corrector": mtp_corrector is not None,
        "mtp_topk_reranker": mtp_topk_reranker is not None,
        "online_correction_cache": bool(online_correction_cache),
        "online_hidden_corrector": float(online_hidden_corrector_alpha) != 0.0,
        "ple_candidate_prefetch": ple_candidate_submit is not None,
        "policy_metrics": bool(wants_policy_metrics),
        "prompt_correction_cache": bool(prompt_correction_cache),
    }
    named = sorted(name for name, active in unsupported.items() if active)
    if named:
        _decline(
            "unsupported_features", "unsupported features: " + ", ".join(named)
        )

    # -- the FR-Spec head ---------------------------------------------------
    from .fable_draft_k20_prescatter import _live_draft_route

    text = getattr(getattr(rt, "model", None), "language_model", None)
    if text is None:
        text = getattr(rt, "model", None)
    head = getattr(text, "_mtplx_frspec_draft_head", None)
    if head is None:
        _refuse("no FR-Spec draft head is installed (need MTPLX_FRSPEC_DRAFT=1)")
    route = _live_draft_route(text, head)
    if route is None:
        _refuse(
            "the FR-Spec full-vocabulary head is installed but is not on the "
            f"live draft route (head={type(head).__name__})"
        )
    if not hasattr(head, "arm_prescatter_capture") or not hasattr(
        head, "take_prescatter_row"
    ):
        _refuse(
            "the live draft head has no pre-scatter capture surface "
            f"({type(head).__name__})"
        )
    ids = getattr(head, "_ids", None)
    if ids is None:
        _refuse("the FR-Spec head carries no ranked id table")
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    frspec_rows = int(ids_np.shape[0])
    if frspec_rows != FRSPEC_ROWS:
        _refuse(
            f"ranked table is {frspec_rows} rows; this route is proven only at "
            f"{FRSPEC_ROWS}"
        )
    if not bool(np.all(ids_np[1:] > ids_np[:-1])):
        _refuse(
            "ranked id table is not strictly ascending; the local->real id map "
            "must be monotone for the tie-break contract to hold"
        )
    vocab_rows = int(getattr(head, "_vocab_rows", 0))
    if vocab_rows <= frspec_rows or int(ids_np[-1]) >= vocab_rows:
        _refuse(f"full vocabulary width {vocab_rows} does not admit the table")

    # -- the compiled body --------------------------------------------------
    max_tokens = int(request_max_tokens)
    if max_tokens <= 0:
        _decline(
            "non_positive_max_tokens",
            f"request_max_tokens must be positive (got {max_tokens})",
        )
    reserve = max_tokens + depth + _RESERVE_SLACK

    from .fable_compiled_draft import build_compiled_draft_chain
    from .kernels.qwen4_frspec_k20_float32_choice import (
        bind_qwen4_frspec_k20_float32_choice,
    )

    selector = bind_qwen4_frspec_k20_float32_choice(
        top_p=float(draft_sampler.top_p)
    )

    def compact_row_fn(dense: Any) -> Any:
        """The 65,536-wide pre-scatter row behind ``dense``.

        Runs only while ``mx.compile`` traces the body, so the compiled graph
        is wired to the compact row and the 248,320-lane ``mx.full`` +
        ``put_along_axis`` behind ``dense`` is never an output and never runs.
        Identity-checked against the array the head returned for THIS call, so
        a stale stash raises instead of silently scoring the wrong step.
        """

        stashed = head.take_prescatter_row(dense)
        if stashed is None:
            raise DeviceDraftChainIneligible(
                f"{_ENV_VAR}: the FR-Spec head did not capture a pre-scatter "
                "row for this draft step (the live draft head changed)"
            )
        if int(stashed.shape[-1]) != frspec_rows:
            raise DeviceDraftChainIneligible(
                f"{_ENV_VAR}: pre-scatter row is {int(stashed.shape[-1])} wide, "
                f"expected {frspec_rows}"
            )
        return stashed

    # MTPLX_FAMILY_CAPTURE_COMMIT interaction: the layer-owned capture is a
    # ContextVar (`qwen4_exp._VERIFY_CAPTURE`) that `rt.model.verify_capture_scope()`
    # sets ONLY around the target verify forward.  The draft block runs outside
    # that scope, so the compiled body is both traced and replayed with capture
    # OFF -- the one thing that would matter (a trace taken inside the scope and
    # replayed outside it) cannot happen from this call site.
    def build_chain(cache: Any, state_tree: Any) -> dict[str, Any]:
        return build_compiled_draft_chain(
            rt=rt,
            mtp_cache=cache,
            state_tree=state_tree,
            mtp_hidden_variant=mtp_hidden_variant,
            selector=selector,
            frspec_ids=getattr(head, "_ids", None),
            depth=depth,
            top_k=TOP_K,
            request_max_tokens=max_tokens,
            compact_row_fn=compact_row_fn,
        )

    head.arm_prescatter_capture(True)
    COUNTERS["claims"] += 1
    plan = DeviceDraftChainPlan(
        mode=_MODE,
        depth=depth,
        top_k=TOP_K,
        draft_temperature=float(draft_sampler.temperature),
        draft_top_p=float(draft_sampler.top_p),
        frspec_rows=frspec_rows,
        vocab_rows=vocab_rows,
        head=head,
        route=str(route),
        reserve_tokens=int(reserve),
        build_chain=build_chain,
        state_tree_fn=state_tree_fn,
        promote_fn=promote_fn,
        receipt_extra={"target_side": target_side},
    )
    print(
        f"[mtplx] fable-device-draft-chain {{'installed': True, "
        f"'mode': {plan.mode!r}, 'depth': {plan.depth}, "
        f"'frspec_rows': {plan.frspec_rows}, 'route': {plan.route!r}, "
        f"'readbacks_per_cycle': {plan.readbacks_per_cycle}, "
        f"'reserve_tokens': {plan.reserve_tokens}}}",
        file=sys.stderr,
        flush=True,
    )
    return plan


def release(plan: DeviceDraftChainPlan | None, *, compiled_verify_bank: Any = None) -> None:
    """Disarm the head capture and restore the stock MTP cache container."""

    if plan is None or plan.released:
        return
    plan.released = True
    plan.head.arm_prescatter_capture(False)
    if (
        plan.mtp_cache is not None
        and compiled_verify_bank is not None
        and hasattr(compiled_verify_bank, "demote")
    ):
        demoted = int(compiled_verify_bank.demote(plan.mtp_cache))
        COUNTERS["cache_demotions"] += demoted


# ---------------------------------------------------------------------------
# The per-cycle run
# ---------------------------------------------------------------------------


@dataclass
class DraftChainResult:
    tokens: list[int]
    distributions: list[SparseDistribution]
    uniforms: np.ndarray
    readbacks: int


def prewarm(
    plan: DeviceDraftChainPlan,
    hidden: Any,
    *,
    mtp_cache: Any,
    rollback: Callable[[Any, int], None],
    cache_offset: Callable[[Any], int],
) -> None:
    """Promote, build and trace the chain once, then restore the MTP history.

    Called at construction, from ``generate_mtpk`` before the decode loop, for
    the same reason ``_pr391_prewarm_float32_d3_core`` is: the ``mx.compile``
    trace and the one-off cache promotion (which reallocates the QSA banks to
    ``offset + reserve_tokens``) are tens to hundreds of milliseconds, and a
    measured window that carried them in cycle 1 would charge them to every
    cycle's average.

    ``hidden`` is the prompt's last widened MTP recursion state and the token
    is 0: the trace needs shapes, not values, and everything it produces is
    discarded.  What it does leave behind is ``depth`` rows on the MTP history,
    which the rollback removes -- and the offset is re-read afterwards, so a
    cache that could not be restored raises here rather than drifting the draft
    history through the whole request.
    """

    import mlx.core as mx

    plan.ensure_bound(mtp_cache)
    base_offset = int(cache_offset(plan.mtp_cache))
    descriptors = fast_midpoint_descriptors(np.zeros(plan.depth, dtype=np.float64))
    result = tuple(
        plan.chain["chain_fn"](
            hidden,
            mx.zeros((1, 1), dtype=mx.uint32),
            mx.array(descriptors),
        )
    )
    mx.eval(*result)
    _clear_rollback(plan)
    rollback(plan.mtp_cache, base_offset)
    if int(cache_offset(plan.mtp_cache)) != base_offset:
        raise DeviceDraftChainIneligible(
            f"{_ENV_VAR}: prewarm did not restore the MTP history offset"
        )
    # The trace consumed one stash per depth; re-arm so nothing stale survives.
    plan.head.arm_prescatter_capture(True)


def _clear_rollback(plan: DeviceDraftChainPlan) -> None:
    """Drop ``update_and_fetch``'s trace-time rollback tracers.

    They are not captured state; anything that evaluated them would raise.
    ``fable_compiled_draft.chain_fn`` clears them per depth already -- this is
    the ``body``-mode and post-``chain_fn`` equivalent.
    """

    if plan.chain is None:
        return
    entry_kv = plan.chain.get("entry_kv")
    if entry_kv is not None and hasattr(entry_kv, "rollback_state"):
        entry_kv.rollback_state[:] = [None, None, None]


def run_cycle(
    plan: DeviceDraftChainPlan,
    *,
    hidden: Any,
    primary: int,
    rng: Any,
    cycle_depth: int,
    live_mtp_cache: Any,
) -> DraftChainResult:
    """Draft ``cycle_depth`` tokens.  ONE ``mx.eval`` in ``chain`` mode.

    ``live_mtp_cache`` is the container the decode loop is about to draft into;
    it must be the very list the compiled body captured, or the replay would
    silently write a different cache.
    """

    import mlx.core as mx

    if plan.released:
        raise DeviceDraftChainIneligible(f"{_ENV_VAR}: the route was released")
    # The previous cycle's `_append_mtp_history` ran the MTP layer -- including
    # the FR-Spec head -- so the head is holding a compact/dense pair nobody
    # will consume, and the dense half keeps that append's whole unevaluated
    # scatter graph alive.  Two attribute writes drop it.  (Re-arming rather
    # than a dedicated clear keeps ONE spelling of "the stash is armed and
    # empty"; the compiled body only ever stashes while tracing.)
    plan.head.arm_prescatter_capture(True)
    # First cycle binds; a later container swap (committed-history live reset
    # or a prefix rebase) rebinds.  Never silently drafts into a stale cache.
    plan.ensure_bound(live_mtp_cache)
    if int(cycle_depth) != int(plan.depth):
        raise DeviceDraftChainIneligible(
            f"{_ENV_VAR}: met cycle_depth={cycle_depth}, route depth={plan.depth}"
        )

    uniforms = _draw_draft_uniforms(rng, int(cycle_depth))
    COUNTERS["tape_draws"] += int(cycle_depth)
    descriptors = fast_midpoint_descriptors(uniforms)
    first_token = mx.array([[int(primary)]], dtype=mx.uint32)

    if plan.mode == MODE_CHAIN:
        result = tuple(
            plan.chain["chain_fn"](hidden, first_token, mx.array(descriptors))
        )
        mx.eval(*result)
        COUNTERS["readbacks"] += 1
        _clear_rollback(plan)
        tokens_np = np.asarray(result[0], dtype=np.uint32).reshape(-1)
        ids_np = np.asarray(result[1], dtype=np.uint32).reshape(cycle_depth, -1)
        values_np = np.asarray(result[2], dtype=np.float32).reshape(cycle_depth, -1)
        probs_np = np.asarray(result[3], dtype=np.float32).reshape(cycle_depth, -1)
        tokens: list[int] = []
        distributions: list[SparseDistribution] = []
        for index in range(int(cycle_depth)):
            distribution, _ = _draft_distribution(
                ids_np[index],
                values_np[index],
                probs_np[index],
                top_p=plan.draft_top_p,
                vocab_size=plan.vocab_rows,
            )
            tokens.append(int(tokens_np[index]))
            distributions.append(distribution)
        COUNTERS["cycles"] += 1
        COUNTERS["depths"] += int(cycle_depth)
        return DraftChainResult(
            tokens=tokens,
            distributions=distributions,
            uniforms=uniforms,
            readbacks=1,
        )

    # -- MODE_BODY: compiled body, stock host tail, one readback per depth ---
    from .fast_sampling import _serial_row_distribution

    body = plan.chain["compiled_body"]
    state_shapes = plan.chain["state_shapes"]
    from .fable_compiled_draft import CompiledDraftStateChanged, state_leaf_shapes

    live_shapes = state_leaf_shapes(plan.chain["state_slots"])
    if live_shapes != state_shapes:
        raise CompiledDraftStateChanged(
            f"{_ENV_VAR}: captured MTP state changed shape "
            f"({state_shapes} -> {live_shapes})"
        )

    next_hidden = hidden
    next_token = first_token
    tokens = []
    distributions = []
    readbacks = 0
    for index in range(int(cycle_depth)):
        ids_arr, values_arr, probs_arr, produced_hidden = body(next_hidden, next_token)
        _clear_rollback(plan)
        mx.eval(ids_arr, values_arr, probs_arr, produced_hidden)
        readbacks += 1
        COUNTERS["readbacks"] += 1
        token_rows, prob_rows = host_support_tail(
            np.asarray(ids_arr, dtype=np.uint32).reshape(-1).astype(np.int64),
            np.asarray(values_arr, dtype=np.float32).reshape(-1),
            np.asarray(probs_arr, dtype=np.float32).reshape(-1),
            top_p=plan.draft_top_p,
            top_k=plan.top_k,
        )
        distribution = _serial_row_distribution(
            token_rows[0], prob_rows[0], int(plan.vocab_rows)
        )
        if distribution is None:
            raise DeviceDraftChainIneligible(
                f"{_ENV_VAR}: draft row carried no finite positive mass"
            )
        # Exactly the draw the stock lane takes here: one float64, in depth
        # order.  `uniforms` was pre-drawn from the same stream, so this reuses
        # the tape rather than advancing it a second time.
        token = int(_choice_from_uniform(distribution, float(uniforms[index])))
        tokens.append(token)
        distributions.append(distribution)
        next_hidden = produced_hidden
        next_token = mx.array([[token]], dtype=mx.uint32)
    COUNTERS["cycles"] += 1
    COUNTERS["depths"] += int(cycle_depth)
    return DraftChainResult(
        tokens=tokens,
        distributions=distributions,
        uniforms=uniforms,
        readbacks=readbacks,
    )


def _choice_from_uniform(distribution: SparseDistribution, uniform: float) -> int:
    """``rng.choice(ids, p=probs)`` with the double already drawn.

    NumPy's ``Generator.choice`` with ``p=`` and ``size=None`` is exactly::

        cdf = p.cumsum(); cdf /= cdf[-1]
        idx = cdf.searchsorted(self.random(), side="right")

    Reproducing it from a pre-drawn uniform keeps ``body`` mode's token
    bit-identical to the stock lane while letting the whole cycle's tape be
    drawn in one ``rng.random(depth)`` call -- the same doubles, in the same
    order, from the same stream.
    """

    probs = np.asarray(distribution.probs, dtype=np.float64)
    cdf = probs.cumsum()
    cdf /= cdf[-1]
    index = int(cdf.searchsorted(np.float64(uniform), side="right"))
    index = min(index, int(probs.shape[0]) - 1)
    return int(np.asarray(distribution.token_ids, dtype=np.int64)[index])


__all__ = [
    "COUNTERS",
    "DeviceDraftChainIneligible",
    "DeviceDraftChainPlan",
    "DraftChainResult",
    "FRSPEC_ROWS",
    "MODE_BODY",
    "MODE_CHAIN",
    "MODE_OFF",
    "TOP_K",
    "claim_request_route",
    "device_cycle_offset",
    "fast_midpoint_descriptors",
    "host_support_tail",
    "is_enabled",
    "mode",
    "prewarm",
    "release",
    "reset_counters",
    "rollback_to_device_offset",
    "run_cycle",
]


# ---------------------------------------------------------------------------
# The MTP-offset syncs the promotion introduces
# ---------------------------------------------------------------------------
#
# Measured on windows 1788400641 (body) and 1788400642 (chain): the candidate
# arms carry +0.41..+0.44 ms/cycle in the part of the decode loop that NO timer
# covers -- six arms, two modes, three seeds, spread 0.03 ms -- while every
# named span (verify, accept, snapshot, capture_commit) is flat and
# `draft_time_s` falls by 0.34..0.42 ms/cycle.  The saving and the cost are the
# same size, which is why the cycle barely moved.
#
# The cause is the cache promotion this route needs.  `mx.compile` re-traces
# when a captured leaf changes shape, so the compiled body requires a
# fixed-capacity MTP cache, and `promote_kv_cache_offsets` supplies one by
# swapping `QSACache` (whose `offset` is `kv.offset`, a plain python int) for
# `TensorOffsetQSACache` (whose `offset` is `cache[2]`, an `mx.array`).  Two
# per-cycle reads then stop being free:
#
#   generation.py  `cycle_mtp_offset = _mtp_cache_offset(mtp_cache)`
#   generation.py  `_rollback_mtp_cache(mtp_cache, cycle_mtp_offset + 1)`
#                  -> `current = int(getattr(cache, "offset", 0))`
#
# Neither is inside a span, and neither is a cheap scalar readback: `int()` on
# a lazy `mx.array` evaluates it, which flushes the command queue and waits for
# the GPU -- once at the top of the cycle and once right after the verify, at
# two points where the stock lane never stalls.  ~0.21 ms each.
#
# The fix keeps the whole quantity ON DEVICE.  On this route
# `cycle_mtp_offset`'s only consumer is the rollback (`reconcile_mtp_indexer_history`
# is refused with `qsa_mtp_precompute`, and the `int(cycle_mtp_offset)`
# lookahead site is PR391's), so nothing ever needs its host value.
#
# Equivalence.  `_rollback_mtp_cache(cache, target)` computes
# `trim = max(0, current - target)` and calls `cache.trim(trim)`, which -- on a
# cleared `rollback_state`, which this route guarantees -- is
# `cache[2] = max(cache[2] - trim, 0)`.  Substituting:
#
#     max(current - max(0, current - target), 0)
#         == max(min(current, target), 0)
#         == min(current, target)                     for target >= 0
#
# so the whole operation is one in-graph `mx.minimum`.  No host read, no
# host-side offset arithmetic to drift, and no tracking state to get wrong --
# which is why this is preferred over carrying a python int the way
# `_pr391_carried_d3["cycle_offset"]` does.


def _tensor_offset_entry(mtp_cache: Any) -> Any:
    """The promoted KV entry whose ``cache[2]`` is a device offset, else None."""

    if not mtp_cache:
        return None
    entry = mtp_cache[0]
    kv = getattr(entry, "kv", entry)
    slots = getattr(kv, "cache", None)
    if not isinstance(slots, list) or len(slots) < 3:
        return None
    if not hasattr(slots[2], "dtype"):
        return None  # a stock python-int offset: the reads are already free
    return kv


def device_cycle_offset(plan: DeviceDraftChainPlan | None, mtp_cache: Any) -> Any:
    """This cycle's MTP offset as a LAZY device array, or None.

    None means "no device path here" and the caller keeps
    ``_mtp_cache_offset``.  Nothing is evaluated.
    """

    if plan is None or plan.released or plan.chain is None:
        return None
    if mtp_cache is not plan.mtp_cache:
        return None
    kv = _tensor_offset_entry(mtp_cache)
    if kv is None:
        return None
    COUNTERS["offset_reads_kept_on_device"] += 1
    return kv.cache[2]


def rollback_to_device_offset(
    plan: DeviceDraftChainPlan | None,
    mtp_cache: Any,
    target: Any,
) -> bool:
    """Trim the MTP history to ``target`` in-graph.  True when it applied.

    False means the caller must fall back to ``_rollback_mtp_cache``: either
    this is not the route's promoted cache, or ``rollback_state`` holds the
    row-restoring snapshot whose fast path is NOT equivalent to moving the
    offset.  Fail-open to the stock path, never to a wrong trim.
    """

    if plan is None or plan.released or plan.chain is None:
        return False
    if mtp_cache is not plan.mtp_cache:
        return False
    kv = _tensor_offset_entry(mtp_cache)
    if kv is None or not hasattr(target, "dtype"):
        return False
    rollback_state = getattr(kv, "rollback_state", None)
    if rollback_state is not None and any(
        slot is not None for slot in rollback_state
    ):
        # The row-restoring fast path in `TensorOffsetKVCache.trim` puts KEYS
        # and VALUES back, not just the offset.  Equivalence does not hold; let
        # the stock rollback run.
        COUNTERS["offset_rollback_declined"] += 1
        return False

    import mlx.core as mx

    kv.cache[2] = mx.minimum(kv.cache[2], target.astype(kv.cache[2].dtype))
    COUNTERS["offset_rollbacks_on_device"] += 1
    return True
