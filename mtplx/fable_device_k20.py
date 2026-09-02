"""Device-side K20 selection + draft sampling for the STOCK native-MTP lane.

``MTPLX_FABLE_DEVICE_K20`` -- read ONCE at import, default OFF.  With the flag
unset :func:`is_enabled` is False, no plan is ever built, and every call site in
``generation.py`` stays behind a module-level constant, so the flag-off lane
evaluates exactly the expressions it evaluated before this module existed.

The problem
-----------
``L-fable-decode-ideas.md`` §0.1 / §A.  Between the end of one compiled verify
graph and the start of the next, the stock lane runs a strictly serial,
sync-terminated chain:

===  ==========================================  =========================
#    step                                        sync
===  ==========================================  =========================
1    draft depth 1: MTP fwd -> FRSpec head ->     ``mx.eval`` +
     ``_device_serial_support_arrays`` ->         ``np.asarray``
     host ``rng.choice``                          (``fast_sampling.py:420``)
2    draft depth 2: same                          same
3    draft depth 3: same                          same
4    target: 4-row ``_device_serial_support_arrays``  same
===  ==========================================  =========================

Four GPU-drain/host-wake bubbles per cycle, every one of them on the dependent
chain.  ``verify_target_distribution_time_s`` is 3.15 ms/window on every
control receipt; the three draft selections sit inside ``draft_time_s``.

What this module does
---------------------
* **Selection.**  :func:`mtplx.kernels.fable_device_k20.device_top_k` replaces
  the ``argpartition``-to-80 + host ``np.lexsort`` support builder with the
  parked PR391 two-stage exact selector, shape-parameterised.  Same ordering
  contract (value desc, real id asc, signed zero collapsed, NaN last).
* **Draft sampling.**  ``kernels/qwen4_frspec_k20_float32_choice`` samples the
  drafted token ON DEVICE from a PCG64 midpoint descriptor, so depth d+1's MTP
  step can consume depth d's token as a device scalar and the three per-depth
  syncs collapse into one.
* **One materialisation.**  :meth:`DeviceDraftChain.materialize` is the single
  ``mx.eval`` for the whole chain.

Sync accounting, honestly
-------------------------
The stock fixed-M4 verify (``generation.py:11913`` ->
``graphbank.forward_fixed_m4(..., host_input_ids=verify_input, ...)`` ->
``dispatch["prepare_aux"]``) needs the drafted ids ON THE HOST to build its
n-gram/PLE aux.  So on that route the chain materialises just before the verify
and the cycle keeps **two** syncs (draft chain, then target support) instead of
four.  Routes that take the device ``verify_input_array`` unchanged
(``forward_ar_capture`` / ``forward_ar``) can defer the chain materialisation
into the target sync and reach **one**; :attr:`DeviceK20Plan.fused_verify_input`
records which one a request got, and the receipt carries it.

Exactness
---------
*Target rows -- bit-identical.*  ``_device_serial_support_arrays``'s hot path
selects the top-k by (float32 value desc, id asc) over ``logits/T``, takes
``exp(v - logsumexp(full row))`` in float32, widens to float64 and applies the
top-p ``cumulative_before`` mask.  :func:`finalize_target_support` does exactly
that on exactly those numbers -- same ``scaled``, same ``mx.logsumexp``, same
float32 ``exp``, same float64 mask.  ``tests/test_fable_device_k20.py``
``test_device_support_equals_stock_hot_path`` pins it against a NumPy
transcription of the shipped builder.

The one place the two can differ is the stock **spill** fallback
(``fast_sampling.py``, ``if spill.any():``).  When a cutoff tie reaches past
the 80-candidate superset the stock hot path's answer is simply wrong, so it
throws the superset away and re-derives with
``_deterministic_mlx_top_k_support``.  This lane is exact over the whole
vocabulary and lands on that selector's SET directly -- no fallback, no
re-derivation.  What can still differ is the ORDER those k entries are in:
under top-p the stock fallback re-sorts by *probability* descending while this
lane keeps *value* descending.  Since ``exp`` is monotone the two orders
coincide unless two distinct float32 values in one top-20 round to the same
float32 probability, and the order is normalised away downstream anyway
(``BatchedSparseDistributions._from_execution_arrays`` sorts by token id); it
survives only into which entries the ``cumulative_before`` mask zeroes.

*Draft rows -- distribution-preserving, not bit-identical.*  The device sampler
prepares its row in float32 (top-p cut, then two normalisations, then an
exact-rational RN32 CDF walk against the uniform's midpoint descriptor); the
stock host sampler prepares it in float64 and walks NumPy's float64 CDF.  Both
are valid samplers of the same shaped support, and the accept loop is fed the
SAME row the device sampled from (:func:`draft_distribution`, which mirrors the
kernel's own float32 preparation bit for bit and hands the accept loop the
float64-exact difference law of the device's float32 CDF).  So ``q_sample`` and
``q_test`` still agree -- to float64, not to float32 -- and speculative
sampling stays exact in the only sense that matters: the emitted law is the
target's, deviating by O(2**-24) in total variation instead of O(2**-53).
It is a *different proposal q*, which the accept/correct law admits for free.

Draw accounting
---------------
Flag-off, the stock lane draws exactly ONE float64 from ``rng`` per draft depth
(``sampling.py:303``, ``rng.choice(ids, p=probs)`` -> one ``self.random()``),
in depth order, before any accept coin.  Flag-on, :func:`draw_draft_uniforms`
takes ``rng.random(depth)`` up front -- the same doubles, in the same order,
from the same stream -- and the accept loop's coins and corrections follow
unchanged.  The stream position after a cycle is therefore identical to
flag-off's.  The uniforms are handed to ``fable_k20_log`` (layout
``stock_device_k20``) so an armed run can replay the selection offline.

Composing with MTPLX_FABLE_DEPTH4_PROBE
--------------------------------------
The chain replaces the per-depth loop that normally captures the probe's
inputs, so ``generation.py``'s device branch captures them itself -- after
``materialize()``, so the token is a host int exactly as on the stock lane --
and the probe still fires from the all-accept branch on depth 3's own hidden,
token and MTP cache.  The probe's own ``q_4`` row deliberately keeps the stock
host shaping: it measures the model, not the selector.  The
``stock_device_k20`` / ``stock_device_k20_bv`` layouts are members of
``fable_k20_log.STOCK_LAYOUTS``, so ``gate_q``, the ``probe_*`` block and both
offline scorers read a device log unchanged.

NO device work happens at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .fable_claim_contract import (
    ClaimDeclined,
    decline as _decline,
    declined_receipt,
)
from .sampling import SamplerConfig, SparseDistribution


_ENV_VAR = "MTPLX_FABLE_DEVICE_K20"

#: The layout name an armed ``MTPLX_FABLE_K20_LOG`` run writes for this lane.
K20_LOG_LAYOUT = "stock_device_k20"

#: The choice kernel and the PCG64 tape are both pinned to this NumPy.
REQUIRED_NUMPY_VERSION = "2.4.4"

TOP_K = 20

_F32_ONE_BITS = int(np.float32(1.0).view(np.uint32))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: Read exactly once, at import.
_ENABLED = _env_truthy(_ENV_VAR)


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_DEVICE_K20`` was set at import."""

    return _ENABLED


def _configure_for_test(enabled: bool) -> None:
    """Flip the import-time gate (tests only)."""

    global _ENABLED
    _ENABLED = bool(enabled)


class DeviceK20Ineligible(RuntimeError):
    """The armed flag cannot work in THIS PROCESS at all.

    Reserved for install-time contract violations -- the pinned NumPy is
    missing, the rng is not the PCG64 tape this route replays, the runtime
    published a draft id map without a compact width.  Every request would
    fail identically.

    A request whose SHAPE this lane does not serve (greedy, top_k != 20,
    penalties, a competing owner of the draft chain, ...) DECLINES to the
    stock selector instead -- see :mod:`mtplx.fable_claim_contract`.  Raising
    on those turns every ineligible request into an HTTP 500 in serving.
    """


# ---------------------------------------------------------------------------
# Construction-time plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceK20Plan:
    """One request's bound device-K20 route."""

    depth: int
    top_k: int
    vocab_size: int
    target_rows: int
    temperature: float
    top_p: float
    draft_temperature: float
    draft_top_p: float
    draft_vocab_size: int
    fused_verify_input: bool
    draft_choice: Any = None
    draft_id_map: Any = None

    @property
    def top_p_active(self) -> bool:
        return 0.0 < float(self.top_p) < 1.0

    @property
    def draft_top_p_active(self) -> bool:
        return 0.0 < float(self.draft_top_p) < 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": K20_LOG_LAYOUT,
            "depth": int(self.depth),
            "top_k": int(self.top_k),
            "vocab_size": int(self.vocab_size),
            "target_rows": int(self.target_rows),
            "draft_vocab_size": int(self.draft_vocab_size),
            "fused_verify_input": bool(self.fused_verify_input),
            "compact_draft_domain": bool(
                self.draft_vocab_size != self.vocab_size
            ),
        }


def claim_request_route(
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    speculative_depth: int,
    rng: Any,
    fused_verify_input: bool,
    target_prefix_verify: bool,
    lazy_target_distributions: bool,
    lazy_bonus_verify_requested: bool,
    batch_target_arrays: bool,
    steer_active: bool,
    penalties_active: bool,
    constraint: Any,
    adaptive_policy: Any,
    adaptive_width_policy: Any,
    mtp_corrector: Any,
    mtp_topk_reranker: Any,
    draft_margin_threshold: float | None,
    online_hidden_corrector_alpha: float,
    online_correction_cache: bool,
    prompt_correction_cache: bool,
    adapter_ensemble_q: bool,
    combine_greedy_draft_read: bool,
    greedy_chain_enabled: bool,
    draft_confidence_needed: bool,
    frspec_legacy_ids: Any,
    late_depth_switch_after: int,
    a3b_target_prefix_route: Any,
    pr391_route: Any,
    adaptive_dtemp_active: bool = False,
    vocab_size: int = 0,
    draft_id_map: Any = None,
    draft_vocab_size: int | None = None,
    draft_core: str = "stock",
    receipt: dict[str, object] | None = None,
) -> DeviceK20Plan | None:
    """Bind the device-K20 route to one generation.

    Returns ``None`` when the flag is off, and ``None`` again when the flag is
    on but this REQUEST's shape is not one the route serves -- a decline, not
    a failure: the stock selector runs and produces the same tokens.  Only an
    INSTALL-time contract violation raises :class:`DeviceK20Ineligible`.
    ``MTPLX_FABLE_STRICT_CLAIMS=1`` turns declines back into that exception.
    """

    if not _ENABLED:
        return None
    try:
        return _claim_request_route(
            sampler=sampler,
            draft_sampler=draft_sampler,
            speculative_depth=speculative_depth,
            rng=rng,
            fused_verify_input=fused_verify_input,
            target_prefix_verify=target_prefix_verify,
            lazy_target_distributions=lazy_target_distributions,
            lazy_bonus_verify_requested=lazy_bonus_verify_requested,
            batch_target_arrays=batch_target_arrays,
            steer_active=steer_active,
            penalties_active=penalties_active,
            constraint=constraint,
            adaptive_policy=adaptive_policy,
            adaptive_width_policy=adaptive_width_policy,
            mtp_corrector=mtp_corrector,
            mtp_topk_reranker=mtp_topk_reranker,
            draft_margin_threshold=draft_margin_threshold,
            online_hidden_corrector_alpha=online_hidden_corrector_alpha,
            online_correction_cache=online_correction_cache,
            prompt_correction_cache=prompt_correction_cache,
            adapter_ensemble_q=adapter_ensemble_q,
            combine_greedy_draft_read=combine_greedy_draft_read,
            greedy_chain_enabled=greedy_chain_enabled,
            draft_confidence_needed=draft_confidence_needed,
            frspec_legacy_ids=frspec_legacy_ids,
            late_depth_switch_after=late_depth_switch_after,
            a3b_target_prefix_route=a3b_target_prefix_route,
            pr391_route=pr391_route,
            adaptive_dtemp_active=adaptive_dtemp_active,
            vocab_size=vocab_size,
            draft_id_map=draft_id_map,
            draft_vocab_size=draft_vocab_size,
            draft_core=draft_core,
        )
    except ClaimDeclined as declined:
        stamped = declined_receipt(
            _ENV_VAR, declined, ineligible=DeviceK20Ineligible
        )
        if receipt is not None:
            receipt.clear()
            receipt.update(stamped)
        return None


def _claim_request_route(
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    speculative_depth: int,
    rng: Any,
    fused_verify_input: bool,
    target_prefix_verify: bool,
    lazy_target_distributions: bool,
    lazy_bonus_verify_requested: bool,
    batch_target_arrays: bool,
    steer_active: bool,
    penalties_active: bool,
    constraint: Any,
    adaptive_policy: Any,
    adaptive_width_policy: Any,
    mtp_corrector: Any,
    mtp_topk_reranker: Any,
    draft_margin_threshold: float | None,
    online_hidden_corrector_alpha: float,
    online_correction_cache: bool,
    prompt_correction_cache: bool,
    adapter_ensemble_q: bool,
    combine_greedy_draft_read: bool,
    greedy_chain_enabled: bool,
    draft_confidence_needed: bool,
    frspec_legacy_ids: Any,
    late_depth_switch_after: int,
    a3b_target_prefix_route: Any,
    pr391_route: Any,
    adaptive_dtemp_active: bool,
    vocab_size: int,
    draft_id_map: Any,
    draft_vocab_size: int | None,
    draft_core: str,
) -> DeviceK20Plan:
    """The claim body.  ``raise`` is install-time; ``_decline`` is per request."""

    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise DeviceK20Ineligible(
            f"device K20 sampling is pinned to NumPy {REQUIRED_NUMPY_VERSION}; "
            f"found {np.__version__}"
        )
    if type(rng) is not np.random.Generator:
        raise DeviceK20Ineligible("device K20 requires a numpy.random.Generator")
    if type(rng.bit_generator) is not np.random.PCG64:
        raise DeviceK20Ineligible("device K20 requires a PCG64 bit generator")

    if str(draft_core) != "stock":
        _decline(
            "non_stock_draft_core",
            "device K20 requires the stock draft route selector "
            f"(got {draft_core!r})",
        )
    if a3b_target_prefix_route is not None or pr391_route is not None:
        _decline("competing_owner", "device K20 owns the stock lane only")
    if target_prefix_verify:
        _decline(
            "target_prefix_verify",
            "device K20 does not admit target_prefix verify",
        )
    if not batch_target_arrays:
        _decline(
            "eager_target_arrays",
            "device K20 requires the batched target-array support route",
        )
    if lazy_target_distributions:
        _decline(
            "lazy_target_distributions",
            "device K20 requires eager target support rows",
        )
    if lazy_bonus_verify_requested:
        _decline(
            "lazy_bonus_verify",
            "device K20 cannot decide lazy-bonus width without host draft ids",
        )
    if steer_active or penalties_active:
        _decline(
            "steer_or_penalties",
            "device K20 does not admit steering or sampler penalties",
        )

    if float(sampler.temperature) <= 0.0 or float(draft_sampler.temperature) <= 0.0:
        _decline(
            "greedy_request",
            "device K20 is a sampled-lane route (temperature > 0); this "
            f"request is target t={float(sampler.temperature)!r} / draft "
            f"t={float(draft_sampler.temperature)!r}",
        )
    if int(sampler.top_k) != TOP_K or int(draft_sampler.top_k) != TOP_K:
        _decline("top_k_not_20", "device K20 is fixed at top_k=20")
    for value in (sampler.top_p, draft_sampler.top_p):
        if not 0.0 < float(value) <= 1.0:
            _decline("top_p_range", "device K20 requires 0 < top_p <= 1")
    if (
        sampler.presence_penalty
        or sampler.frequency_penalty
        or draft_sampler.presence_penalty
        or draft_sampler.frequency_penalty
    ):
        _decline(
            "sampler_penalties", "device K20 does not admit sampler penalties"
        )

    if int(speculative_depth) <= 0:
        _decline("non_positive_depth", "device K20 requires a positive draft depth")
    if int(late_depth_switch_after) != 0:
        _decline(
            "late_depth_switch", "device K20 does not admit late-depth switching"
        )

    unsupported = {
        "constraint": constraint is not None,
        "adaptive_policy": adaptive_policy is not None,
        "adaptive_width_policy": adaptive_width_policy is not None,
        "mtp_corrector": mtp_corrector is not None,
        "mtp_topk_reranker": mtp_topk_reranker is not None,
        "draft_margin_threshold": draft_margin_threshold is not None,
        "online_hidden_corrector": float(online_hidden_corrector_alpha) != 0.0,
        "online_correction_cache": bool(online_correction_cache),
        "prompt_correction_cache": bool(prompt_correction_cache),
        "adapter_ensemble_q": bool(adapter_ensemble_q),
        "combine_greedy_draft_read": bool(combine_greedy_draft_read),
        "greedy_chain": bool(greedy_chain_enabled),
        "draft_confidence": bool(draft_confidence_needed),
        "frspec_legacy": frspec_legacy_ids is not None,
        # MTPLX_ADAPTIVE_DTEMP rebinds `draft_sampler` mid-generation; the
        # plan bakes the draft temperature into the choice kernel's bound
        # top-p route and its own `draft_temperature`, so a transition would
        # silently desync the proposal from the row the accept loop scores.
        "adaptive_dtemp": bool(adaptive_dtemp_active),
    }
    named = sorted(name for name, active in unsupported.items() if active)
    if named:
        _decline(
            "unsupported_features",
            "device K20 received unsupported features: " + ", ".join(named),
        )

    draft_width = int(draft_vocab_size or vocab_size or 0)
    if draft_id_map is not None and not draft_width:
        raise DeviceK20Ineligible(
            "device K20 draft id map requires an explicit compact draft width"
        )

    from .kernels.qwen4_frspec_k20_float32_choice import (
        bind_qwen4_frspec_k20_float32_choice,
    )

    return DeviceK20Plan(
        depth=int(speculative_depth),
        top_k=TOP_K,
        vocab_size=int(vocab_size),
        target_rows=int(speculative_depth) + 1,
        temperature=float(sampler.temperature),
        top_p=float(sampler.top_p),
        draft_temperature=float(draft_sampler.temperature),
        draft_top_p=float(draft_sampler.top_p),
        draft_vocab_size=draft_width,
        fused_verify_input=bool(fused_verify_input),
        draft_choice=bind_qwen4_frspec_k20_float32_choice(
            top_p=float(draft_sampler.top_p)
        ),
        draft_id_map=draft_id_map,
    )


# ---------------------------------------------------------------------------
# Target support -- bit-identical to _device_serial_support_arrays' hot path
# ---------------------------------------------------------------------------


def target_support_device(logits: Any, plan: DeviceK20Plan) -> tuple[Any, Any, Any]:
    """Queue the exact K20 support for the verify rows.  No sync."""

    import mlx.core as mx

    from .kernels.fable_device_k20 import device_top_k

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    scaled = rows * (1.0 / float(plan.temperature))
    ids, values = device_top_k(scaled, top_k=plan.top_k)
    if plan.top_p_active:
        log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
        probs = mx.exp(values - log_total)
    else:
        probs = None
    return ids, values, probs


def finalize_target_support(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray | None,
    plan: DeviceK20Plan,
) -> tuple[np.ndarray, np.ndarray]:
    """Host tail of ``_device_serial_support_arrays``, on exact rows.

    Mirrors ``fast_sampling.py:445-460`` exactly: under top-p the float32
    probabilities widen to float64 and the ``cumulative_before`` mask zeroes
    everything at or past the nucleus; otherwise the k values are softmaxed in
    float64 after subtracting the row max.
    """

    token_rows = np.asarray(ids, dtype=np.int64)
    if plan.top_p_active:
        if probs is None:
            raise ValueError("top-p support requires device probabilities")
        prob_rows = np.asarray(probs, dtype=np.float64)
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(
            cumulative_before < float(plan.top_p), prob_rows, 0.0
        )
    else:
        vals64 = np.asarray(values, dtype=np.float32)[:, : plan.top_k].astype(
            np.float64
        )
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)
    return token_rows, prob_rows


def reference_top_k(
    rows: np.ndarray,
    *,
    top_k: int = TOP_K,
    id_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-free NumPy oracle for :func:`device_top_k`.

    Same contract the stock selector documents: value descending, then real
    token id ascending, signed zero collapsed, NaN last.  Used by the CPU
    tests and by ``scripts/fable/micro_k20_select.py`` to count differences
    against the stock support builder.
    """

    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("reference_top_k expects a 2-D [rows, vocab] block")
    k = int(top_k)
    vocab_size = int(values.shape[1])
    if id_map is None:
        real = np.arange(vocab_size, dtype=np.int64)
    else:
        real = np.asarray(id_map, dtype=np.int64).reshape(-1)
        if real.shape[0] != vocab_size:
            raise ValueError("reference_top_k id_map must be [vocab_size]")
    # `-values` collapses signed zero (0.0 == -0.0 under comparison) and keeps
    # NaN last, matching fdk_value_before.
    negated = -values
    out_ids = np.empty((values.shape[0], k), dtype=np.int64)
    out_vals = np.empty((values.shape[0], k), dtype=np.float32)
    for row in range(values.shape[0]):
        order = np.lexsort((real, negated[row]))[:k]
        out_ids[row] = real[order]
        out_vals[row] = values[row][order]
    return out_ids, out_vals


# ---------------------------------------------------------------------------
# Draft row preparation -- the float32 law the device sampler walks
# ---------------------------------------------------------------------------


def _pairwise_sum_f32(values: np.ndarray) -> np.float32:
    """NumPy 2.4.4's float32 pairwise reduction.

    Verified bit-identical to the choice kernel's own
    ``_numpy_pairwise_sum`` oracle for every length 1..20 in
    ``tests/test_fable_device_k20.py``.
    """

    return np.float32(np.sum(np.asarray(values, dtype=np.float32), dtype=np.float32))


def prepare_draft_row_f32(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    top_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised mirror of the choice kernel's ``_prepare_reference_row``.

    Returns ``(retained ids ascending uint32, normalized float32)`` -- the
    exact array the Metal kernel builds its CDF over.  Bit-identical to
    ``mtplx.kernels.qwen4_frspec_k20_float32_choice._prepare_reference_row``
    (asserted in the CPU tests); the python original costs ~36 us per row and
    this costs ~3.
    """

    ids = np.asarray(ids, dtype=np.uint32)
    values = np.asarray(values, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    rounded_top_p = np.float32(float(top_p))

    order = np.lexsort((ids.astype(np.int64), -values))
    sorted_ids = ids[order]
    sorted_probs = probs[order]

    # `cumulative_before[i]` is the sequential float32 running sum of every
    # probability RANKED BEFORE i -- kept in rank order, including entries the
    # nucleus later drops, exactly as the kernel accumulates it.
    running = np.cumsum(sorted_probs, dtype=np.float32)
    cumulative_before = np.empty_like(running)
    cumulative_before[0] = np.float32(0.0)
    cumulative_before[1:] = running[:-1]

    keep = sorted_probs > np.float32(0.0)
    if rounded_top_p != np.float32(1.0):
        keep &= cumulative_before < rounded_top_p

    retained_ids = sorted_ids[keep]
    retained_probs = sorted_probs[keep]
    if retained_ids.size == 0:
        raise ValueError("draft row retained no probability mass")

    first_total = _pairwise_sum_f32(retained_probs)

    token_order = np.argsort(retained_ids, kind="stable")
    ordered_ids = retained_ids[token_order]
    normalized = (retained_probs[token_order] / first_total).astype(np.float32)
    normalized = np.where(
        np.isfinite(normalized) & (normalized > np.float32(0.0)),
        normalized,
        np.float32(0.0),
    ).astype(np.float32)

    second_total = _pairwise_sum_f32(normalized)
    if int(second_total.view(np.uint32)) != _F32_ONE_BITS:
        normalized = (normalized / second_total).astype(np.float32)
    return ordered_ids, normalized


def draft_distribution(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float,
    vocab_size: int,
) -> tuple[SparseDistribution, np.ndarray]:
    """The proposal the DEVICE sampled from, as the accept loop's ``q``.

    The device selects ``argmin_j { cdf_j / cdf_last > u }`` under exact
    rational comparison, so its law is the float64-exact difference law of the
    float32 CDF -- which is what this returns.  Feeding the accept loop this
    row (rather than the stock float64 preparation of the same support) is what
    keeps ``q_sample == q_test`` and the emitted distribution the target's.
    """

    ordered_ids, normalized = prepare_draft_row_f32(ids, values, probs, top_p)
    cdf = np.cumsum(normalized, dtype=np.float32).astype(np.float64)
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("draft row float32 CDF has no positive mass")
    weights = np.diff(cdf, prepend=np.float64(0.0)) / total
    return (
        SparseDistribution(
            token_ids=ordered_ids.astype(np.int64),
            probs=weights,
            vocab_size=int(vocab_size),
        ),
        normalized,
    )


# ---------------------------------------------------------------------------
# Draw accounting
# ---------------------------------------------------------------------------


def draw_draft_uniforms(rng: Any, depth: int) -> np.ndarray:
    """Take the SAME doubles ``rng.choice`` would have taken, in the same order.

    Flag-off the stock lane draws one ``self.random()`` double per depth,
    inside ``sample_from_distribution`` -> ``rng.choice`` (``sampling.py:303``,
    NumPy ``Generator.choice`` with ``p=`` and ``size=None``).  ``rng.random(n)``
    fills from the same stream in the same order, so a cycle leaves the PCG64
    cursor exactly where flag-off leaves it and every later accept coin,
    correction and bonus draw is unshifted.
    """

    values = np.asarray(rng.random(int(depth), dtype=np.float64), dtype=np.float64)
    if values.shape != (int(depth),):
        raise RuntimeError("draft uniform draw returned the wrong shape")
    return values


def build_uniform_descriptors(uniforms: np.ndarray) -> np.ndarray:
    """Audited PCG64 midpoint descriptors for the device CDF walk."""

    from .kernels.qwen4_frspec_k20_float32_choice import (
        build_pcg64_midpoint_descriptors,
    )

    return build_pcg64_midpoint_descriptors(
        np.asarray(uniforms, dtype=np.float64).reshape(-1)
    )


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


@dataclass
class DraftChainResult:
    tokens: list[int]
    distributions: list[SparseDistribution]
    uniforms: np.ndarray
    normalized_rows: list[np.ndarray]


class DeviceDraftChain:
    """Accumulate ``depth`` device draft selections and sample them on device.

    ``step`` queues one depth and returns the drafted token as a lazy
    ``[1, 1] int32`` device array, ready to feed the next ``draft_mtp`` with no
    host round trip.  ``materialize`` is the chain's ONLY sync.
    """

    __slots__ = (
        "plan",
        "uniforms",
        "depth",
        "vocab_size",
        "_descriptors",
        "_ids",
        "_values",
        "_probs",
        "_selected",
    )

    def __init__(self, plan: DeviceK20Plan, uniforms: np.ndarray) -> None:
        if plan is None:
            raise ValueError("DeviceDraftChain requires a bound plan")
        uniforms = np.asarray(uniforms, dtype=np.float64).reshape(-1)
        if not 0 < uniforms.shape[0] <= plan.depth:
            raise ValueError("draft uniform count must be in 1..plan.depth")
        self.plan = plan
        self.uniforms = uniforms
        self.depth = int(uniforms.shape[0])
        self.vocab_size = int(plan.vocab_size)
        self._descriptors = build_uniform_descriptors(uniforms)
        self._ids: list[Any] = []
        self._values: list[Any] = []
        self._probs: list[Any] = []
        self._selected: list[Any] = []

    @property
    def depth_done(self) -> int:
        return len(self._selected)

    def step(self, draft_logits: Any, *, compact_row: Any = None) -> Any:
        """Queue one draft depth.  Returns the token as ``[1, 1] int32``."""

        import mlx.core as mx

        from .kernels.fable_device_k20 import device_top_k

        plan = self.plan
        depth_index = len(self._selected)
        if depth_index >= self.depth:
            raise RuntimeError("device draft chain is already full")

        if compact_row is not None:
            row = compact_row.reshape(1, -1).astype(mx.float32)
            id_map = plan.draft_id_map
        else:
            row = draft_logits[:, -1, :].reshape(1, -1).astype(mx.float32)
            id_map = None
            if not self.vocab_size:
                # The stock FRSpec head already scatters its 65,536 rows into
                # the full vocabulary (``frspec_draft.py:_FullVocabDraftHead``,
                # sentinel -1e30), so the row width IS the vocabulary width.
                self.vocab_size = int(row.shape[-1])
        scaled = row * (1.0 / float(plan.draft_temperature))
        ids, values = device_top_k(scaled, top_k=plan.top_k, id_map=id_map)
        if plan.draft_top_p_active:
            probs = mx.exp(
                values - mx.logsumexp(scaled, axis=-1, keepdims=True)
            )
        else:
            # top_p == 1 disables the nucleus cut, and the kernel normalises
            # what it keeps, so any positive scaling of the true probabilities
            # gives the same law.  Subtracting the row max keeps it finite.
            probs = mx.exp(values - mx.max(values, axis=-1, keepdims=True))

        descriptor = mx.array(
            self._descriptors[depth_index : depth_index + 1], dtype=mx.uint32
        )
        selected, _, _, _ = plan.draft_choice(ids, values, probs, descriptor)

        self._ids.append(ids)
        self._values.append(values)
        self._probs.append(probs)
        self._selected.append(selected)
        return selected.reshape(1, 1).astype(mx.int32)

    def device_token_ids(self) -> Any:
        """The drafted ids as one lazy ``[1, depth] int32`` row."""

        import mlx.core as mx

        return mx.concatenate(
            [token.reshape(1, 1) for token in self._selected], axis=1
        ).astype(mx.int32)

    def leaves(self) -> tuple[Any, ...]:
        return (*self._ids, *self._values, *self._probs, *self._selected)

    def materialize(self, *extra: Any) -> DraftChainResult:
        """The chain's ONE sync.  ``extra`` rides along in the same eval."""

        import mlx.core as mx

        plan = self.plan
        if len(self._selected) != self.depth:
            raise RuntimeError("device draft chain was not run to full depth")
        if not self.vocab_size:
            raise RuntimeError("device draft chain never resolved a vocabulary width")
        mx.eval(*self.leaves(), *[leaf for leaf in extra if leaf is not None])

        tokens: list[int] = []
        distributions: list[SparseDistribution] = []
        normalized_rows: list[np.ndarray] = []
        for index in range(self.depth):
            ids = np.asarray(self._ids[index], dtype=np.uint32).reshape(-1)
            values = np.asarray(self._values[index], dtype=np.float32).reshape(-1)
            probs = np.asarray(self._probs[index], dtype=np.float32).reshape(-1)
            token = int(np.asarray(self._selected[index]).reshape(-1)[0])
            distribution, normalized = draft_distribution(
                ids,
                values,
                probs,
                top_p=plan.draft_top_p,
                vocab_size=self.vocab_size,
            )
            tokens.append(token)
            distributions.append(distribution)
            normalized_rows.append(normalized)
        return DraftChainResult(
            tokens=tokens,
            distributions=distributions,
            uniforms=self.uniforms,
            normalized_rows=normalized_rows,
        )


def selfcheck() -> dict[str, Any]:
    """Deterministic CPU-only check of the host mirror against the oracle."""

    from .kernels.qwen4_frspec_k20_float32_choice import (
        _prepare_reference_row,
        selfcheck_qwen4_frspec_k20_float32_choice,
    )

    rng = np.random.default_rng(20260901)
    checked = 0
    for _ in range(64):
        ids = rng.choice(1 << 17, size=TOP_K, replace=False).astype(np.uint32)
        values = (rng.standard_normal(TOP_K) * 4.0).astype(np.float32)
        raw = np.exp(values.astype(np.float64) - values.max())
        probs = (raw / raw.sum() * rng.uniform(0.2, 1.0)).astype(np.float32)
        want_ids, want_probs = _prepare_reference_row(
            ids, values, probs, np.float32(0.95)
        )
        got_ids, got_probs = prepare_draft_row_f32(ids, values, probs, 0.95)
        # The oracle returns the CDF; rebuild it from the mirror's weights.
        got_cdf = np.cumsum(got_probs, dtype=np.float32)
        if not np.array_equal(np.asarray(want_ids, dtype=np.uint32), got_ids):
            raise RuntimeError("device K20 host mirror disagrees on retained ids")
        if not np.array_equal(
            np.asarray(want_probs, dtype=np.float32).view(np.uint32),
            got_cdf.view(np.uint32),
        ):
            raise RuntimeError("device K20 host mirror disagrees on the float32 CDF")
        checked += 1
    return {
        "rows_checked": checked,
        "choice": selfcheck_qwen4_frspec_k20_float32_choice(),
        "layout": K20_LOG_LAYOUT,
    }


__all__ = [
    "DeviceDraftChain",
    "DeviceK20Ineligible",
    "DeviceK20Plan",
    "DraftChainResult",
    "K20_LOG_LAYOUT",
    "build_uniform_descriptors",
    "claim_request_route",
    "draft_distribution",
    "draw_draft_uniforms",
    "finalize_target_support",
    "is_enabled",
    "prepare_draft_row_f32",
    "reference_top_k",
    "selfcheck",
    "target_support_device",
]
