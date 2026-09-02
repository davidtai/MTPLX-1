"""MTPLX_FABLE_DRAFT_K20_PRESCATTER -- equivalence and construction proofs.

The lane claims the draft K20 support built from the FR-Spec head's 65,536-row
output is the SAME support the shipped builder produces from the 248,320-wide
scattered row.  These tests hold the two next to each other on random rows and
on rows with constructed exact ties, and compare RAW BITS (through an integer
view, so a one-ulp drift or a sign-flipped zero fails).

They run entirely on the CPU stream: no Metal, no model, no kernels.  The one
claim that is stream-dependent -- whether the two DIFFERENT-WIDTH reductions
behind ``mx.logsumexp`` associate their float32 partial sums identically -- is
pinned here on the CPU stream and re-measured on the Metal stream by
``scripts/fable/micro_draft_k20.py``, which prints the differing-row counters
instead of assuming.  See ``mtplx/fable_draft_k20_prescatter`` for why a
residual ULP there could not change the SUPPORT.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from types import SimpleNamespace

import mtplx.fable_draft_k20_prescatter as prescatter
import mtplx.fast_sampling as fast_sampling
import mtplx.frspec_draft as frspec_draft
from mtplx.fable_draft_k20_prescatter import (
    DraftK20PrescatterIneligible,
    DraftK20PrescatterPlan,
)
from mtplx.sampling import SamplerConfig, sample_from_distribution


TOP_K = 20
TEMPERATURE = 0.6
TOP_P = 0.95


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ranked_ids(rows: int, vocab_rows: int, seed: int) -> np.ndarray:
    """A strictly ascending ranked table -- the shipped artifact's shape."""

    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(vocab_rows, size=rows, replace=False)
    ).astype(np.int64)


def _plan(rows: int, vocab_rows: int, seed: int, head=None) -> DraftK20PrescatterPlan:
    return DraftK20PrescatterPlan(
        head=head,
        ids_np=_ranked_ids(rows, vocab_rows, seed),
        rows=rows,
        vocab_rows=vocab_rows,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        route="native_mtp_head",
    )


def _compact_row(rows: int, seed: int, *, dtype=mx.float32) -> mx.array:
    """A logit row shaped like a real one: a heavy head, a long tail."""

    rng = np.random.default_rng(seed)
    base = (rng.standard_normal(rows) * 2.0).astype(np.float32)
    peaks = rng.choice(rows, size=min(64, rows), replace=False)
    base[peaks] += rng.uniform(6.0, 14.0, size=peaks.shape[0]).astype(np.float32)
    return mx.array(base).astype(dtype)


def _scatter(compact: mx.array, plan: DraftK20PrescatterPlan) -> mx.array:
    """Exactly ``frspec_draft._FullVocabDraftHead.__call__``'s scatter."""

    subset = compact.reshape(1, -1)
    output = mx.full((1, plan.vocab_rows), -1.0e30, dtype=subset.dtype)
    index = mx.broadcast_to(
        mx.array(plan.ids_np, dtype=mx.int32).reshape(1, -1), subset.shape
    )
    return mx.put_along_axis(output, index, subset, axis=-1).reshape(-1)


def _config(*, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K) -> SamplerConfig:
    return SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    assert a.dtype == b.dtype
    width = {4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return bool(np.array_equal(a.view(width), b.view(width)))


# ---------------------------------------------------------------------------
# 1. The sentinel contributes exactly zero
# ---------------------------------------------------------------------------


def test_sentinel_underflows_to_exact_zero_in_float32():
    """``exp(sentinel/T - max)`` is exactly +0.0, so the sum is unchanged."""

    sentinel = mx.array([-1.0e30], dtype=mx.float32) * (1.0 / TEMPERATURE)
    assert bool(mx.isfinite(sentinel).item()), "the pad must not be an inf"
    for peak in (-50.0, 0.0, 40.0, 1e4):
        term = mx.exp(sentinel - mx.array([peak], dtype=mx.float32))
        value = np.asarray(term, dtype=np.float32)[0]
        assert value == np.float32(0.0)
        # +0.0, not -0.0: adding it cannot even flip a signed zero.
        assert value.view(np.uint32) == np.uint32(0)
    # x + 0.0 == x exactly, for the whole float32 range the sum can hold.
    for magnitude in (1e-38, 1.0, 3.14159, 1e30):
        x = np.float32(magnitude)
        assert (x + np.float32(0.0)).view(np.uint32) == x.view(np.uint32)


def test_sentinel_survives_a_bfloat16_head_output():
    """A bf16 head row keeps the pad finite and far below every real logit."""

    pad = mx.array([-1.0e30], dtype=mx.bfloat16)
    assert bool(mx.isfinite(pad).item())
    widened = np.asarray(pad.astype(mx.float32), dtype=np.float32)[0]
    assert widened < np.float32(-9.9e29)
    term = mx.exp(pad.astype(mx.float32) * (1.0 / TEMPERATURE) - mx.array([0.0]))
    assert np.asarray(term, dtype=np.float32)[0].view(np.uint32) == np.uint32(0)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_full_vocab_logsumexp_equals_compact_logsumexp(seed):
    """The normaliser the pre-scatter row computes is the full row's."""

    plan = _plan(65_536, 248_320, seed)
    compact = _compact_row(plan.rows, seed + 900)
    dense = _scatter(compact, plan)
    scale = 1.0 / TEMPERATURE
    compact_lse = mx.logsumexp(compact.astype(mx.float32) * scale, axis=-1)
    dense_lse = mx.logsumexp(dense.astype(mx.float32) * scale, axis=-1)
    mx.eval(compact_lse, dense_lse)
    assert _same_bits(
        np.asarray(compact_lse, dtype=np.float32),
        np.asarray(dense_lse, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 2. Support equivalence -- stock builder on the dense row vs pre-scatter
# ---------------------------------------------------------------------------


def _assert_support_equivalent(plan, compact, config):
    dense = _scatter(compact, plan)
    want_ids, want_probs, want_vocab = fast_sampling._device_serial_support_arrays(
        dense.astype(mx.float32), config
    )
    got_ids, got_probs, got_vocab = prescatter.prescatter_serial_support_arrays(
        plan, compact.astype(mx.float32), config
    )
    assert got_vocab == want_vocab == plan.vocab_rows
    assert np.array_equal(got_ids, want_ids), (want_ids[0][:6], got_ids[0][:6])
    assert _same_bits(got_probs, want_probs)
    return want_ids, want_probs


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_support_matches_stock_at_production_shape(seed):
    plan = _plan(65_536, 248_320, seed)
    compact = _compact_row(plan.rows, seed + 7)
    _assert_support_equivalent(plan, compact, _config())


@pytest.mark.parametrize("top_p", [0.95, 1.0, 0.0])
def test_support_matches_stock_across_top_p_branches(top_p):
    """``top_p`` in (0,1) takes the logsumexp branch; 1.0/0.0 take softmax."""

    plan = _plan(8_192, 32_768, 55)
    compact = _compact_row(plan.rows, 56)
    _assert_support_equivalent(plan, compact, _config(top_p=top_p))


def test_support_matches_stock_with_exact_ties_inside_the_support():
    """Exact float ties inside the top-20 must break the same way."""

    plan = _plan(8_192, 32_768, 71)
    compact = np.asarray(_compact_row(plan.rows, 72), dtype=np.float32)
    order = np.argsort(-compact)
    # Flatten ranks 3..8 onto one value: six exact ties well inside the top-20.
    compact[order[3:9]] = compact[order[3]]
    # And a second tie group straddling ranks 18..22, i.e. the k-th cutoff.
    compact[order[18:23]] = compact[order[18]]
    _assert_support_equivalent(plan, mx.array(compact), _config())


def test_support_matches_stock_with_a_cutoff_tie_beyond_the_superset():
    """The stock spill fallback and the pre-scatter one agree.

    ``_device_serial_support_arrays`` re-derives with the deterministic device
    selector when the cutoff tie group may reach past the 4k candidate
    superset.  A flat row makes every lane tie, which forces that branch on
    both rows (the condition reads only the sorted candidate VALUES, which are
    identical).
    """

    plan = _plan(4_096, 16_384, 83)
    compact = mx.full((plan.rows,), 1.25, dtype=mx.float32)
    config = _config()
    dense = _scatter(compact, plan)
    scaled_dense = dense.astype(mx.float32) * (1.0 / TEMPERATURE)
    cand = mx.argpartition(-scaled_dense.reshape(1, -1), kth=79, axis=-1)[:, :80]
    vals = mx.take_along_axis(scaled_dense.reshape(1, -1), cand, axis=-1)
    mx.eval(vals)
    values = np.asarray(vals, dtype=np.float32)
    assert np.nanmin(values, axis=1)[0] >= values[0, TOP_K - 1], "spill not armed"
    _assert_support_equivalent(plan, compact, config)


def test_support_matches_stock_from_a_bfloat16_head_row():
    plan = _plan(8_192, 32_768, 91)
    compact = _compact_row(plan.rows, 92, dtype=mx.bfloat16)
    _assert_support_equivalent(plan, compact, _config())


# ---------------------------------------------------------------------------
# 3. Distribution + sampled token equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [401, 402])
def test_distribution_and_draw_match_the_stock_draft_read(seed):
    plan = _plan(65_536, 248_320, seed)
    compact = _compact_row(plan.rows, seed + 3)
    dense = _scatter(compact, plan)
    config = _config()

    want = fast_sampling.sparse_distribution_from_mlx_logits(
        dense.astype(mx.float32), config
    )
    got = prescatter.prescatter_sparse_distribution(
        plan, compact.astype(mx.float32), config
    )
    assert want is not None
    assert got.vocab_size == want.vocab_size == plan.vocab_rows
    assert np.array_equal(got.token_ids, want.token_ids)
    assert _same_bits(
        np.asarray(got.probs, dtype=np.float64),
        np.asarray(want.probs, dtype=np.float64),
    )
    # And the same one rng draw, from the same stream position.
    assert sample_from_distribution(
        got, np.random.default_rng(5150)
    ) == sample_from_distribution(want, np.random.default_rng(5150))


def test_greedy_read_matches_the_stock_argmax():
    plan = _plan(8_192, 32_768, 611)
    compact = _compact_row(plan.rows, 612)
    dense = _scatter(compact, plan)
    config = _config(temperature=0.0)
    head = _FakeHead(dense, compact)
    plan = DraftK20PrescatterPlan(**{**plan.__dict__, "head": head})

    token, distribution = prescatter.read_draft(
        plan, dense, config, np.random.default_rng(1), need_distribution=True
    )
    assert token == int(mx.argmax(dense.astype(mx.float32), axis=-1).item())
    assert distribution.vocab_size == plan.vocab_rows
    assert list(distribution.token_ids) == [token]


def test_greedy_argmax_tie_picks_the_lowest_real_id():
    """``argmax``'s lowest-LOCAL-index tie-break maps to the lowest real id."""

    plan = _plan(1_024, 4_096, 613)
    compact = np.full(plan.rows, -3.0, dtype=np.float32)
    compact[[17, 900]] = 9.0  # an exact tie between two local rows
    compact = mx.array(compact)
    dense = _scatter(compact, plan)
    head = _FakeHead(dense, compact)
    plan = DraftK20PrescatterPlan(**{**plan.__dict__, "head": head})
    token, _ = prescatter.read_draft(
        plan, dense, _config(temperature=0.0), None, need_distribution=False
    )
    assert token == int(plan.ids_np[17])
    assert token == int(mx.argmax(dense.astype(mx.float32), axis=-1).item())


def test_sampled_read_returns_the_row_it_sampled_from():
    """The sampled branch always returns q, as ``_sample_draft_from_logits``
    does (it tail-calls ``_sample_from_logits``, ignoring need_distribution)."""

    plan = _plan(4_096, 16_384, 621)
    compact = _compact_row(plan.rows, 622)
    dense = _scatter(compact, plan)
    head = _FakeHead(dense, compact)
    plan = DraftK20PrescatterPlan(**{**plan.__dict__, "head": head})
    _, distribution = prescatter.read_draft(
        plan, dense, _config(), np.random.default_rng(3), need_distribution=False
    )
    assert distribution is not None
    assert distribution.vocab_size == plan.vocab_rows


class _FakeHead:
    """The head surface ``take_compact_row`` needs, without a model."""

    def __init__(self, dense, compact):
        self._dense = dense
        self._compact = compact
        self.armed = False

    def arm_prescatter_capture(self, enabled):
        self.armed = bool(enabled)

    def take_prescatter_row(self, dense):
        if dense is not self._dense:
            return None
        return self._compact


def test_take_compact_row_refuses_a_stale_stash():
    plan = _plan(4_096, 16_384, 631)
    compact = _compact_row(plan.rows, 632)
    dense = _scatter(compact, plan)
    plan = DraftK20PrescatterPlan(
        **{**plan.__dict__, "head": _FakeHead(dense, compact)}
    )
    other = _scatter(compact, plan)
    with pytest.raises(DraftK20PrescatterIneligible, match="did not capture"):
        prescatter.take_compact_row(plan, other)


# ---------------------------------------------------------------------------
# 4. The head's capture surface
# ---------------------------------------------------------------------------


def _full_vocab_head(rows: int, vocab_rows: int, seed: int):
    import mlx.nn as nn

    ids = mx.array(_ranked_ids(rows, vocab_rows, seed), dtype=mx.int32)
    inner = nn.Linear(8, rows, bias=False)
    return frspec_draft._full_vocab_head(inner, ids, vocab_rows), ids


def test_head_capture_is_disarmed_by_default():
    head, _ = _full_vocab_head(256, 1_024, 7)
    x = mx.zeros((1, 1, 8))
    dense = head(x)
    assert int(dense.shape[-1]) == 1_024
    assert head.take_prescatter_row(dense) is None


def test_head_capture_stashes_the_compact_row_and_clears_on_take():
    head, _ = _full_vocab_head(256, 1_024, 8)
    head.arm_prescatter_capture(True)
    x = mx.random.normal((1, 1, 8))
    dense = head(x)
    row = head.take_prescatter_row(dense)
    assert row is not None
    assert int(row.reshape(-1).shape[0]) == 256
    # Consumed: a second take cannot silently succeed.
    assert head.take_prescatter_row(dense) is None
    # And the compact row really is the pre-scatter one.
    assert _same_bits(
        np.asarray(row.reshape(-1), dtype=np.float32),
        np.asarray(mx.take(dense.reshape(-1), head._ids), dtype=np.float32),
    )


def test_head_capture_identity_rejects_another_call_dense_row():
    head, _ = _full_vocab_head(256, 1_024, 9)
    head.arm_prescatter_capture(True)
    first = head(mx.random.normal((1, 1, 8)))
    head(mx.random.normal((1, 1, 8)))
    assert head.take_prescatter_row(first) is None


def test_head_disarm_drops_the_stash():
    head, _ = _full_vocab_head(256, 1_024, 10)
    head.arm_prescatter_capture(True)
    dense = head(mx.random.normal((1, 1, 8)))
    head.arm_prescatter_capture(False)
    assert head.take_prescatter_row(dense) is None


# ---------------------------------------------------------------------------
# 5. Construction-bound install / refusal
# ---------------------------------------------------------------------------


_ELIGIBLE = dict(
    draft_core="stock",
    target_prefix_verify=False,
    a3b_target_prefix_route=None,
    pr391_route=None,
    device_k20_route=None,
    frspec_legacy_ids=None,
    adaptive_width_policy=None,
    combine_greedy_draft_read=False,
    draft_confidence_needed=False,
    draft_margin_threshold=None,
    wants_policy_metrics=False,
    correction_cache_enabled=False,
    adapter_ensemble_q=False,
    mtp_topk_reranker=None,
    relaxed_draft_ties=False,
    penalties_active=False,
    steer_active=False,
)


class _TextModelStandIn:
    """``models/qwen4_exp.TextModel``'s draft surface, on a plain object.

    Only the three members the FR-Spec install and this module's liveness
    probe touch: the draft-head hook ``mtp_forward`` projects through, the
    default binding ``TextModel.__init__`` puts there, and the bind hook the
    install calls.  Nothing here evaluates.
    """

    def __init__(self) -> None:
        # qwen4_exp.TextModel.__init__
        self._mtp_draft_head_logits = self._head_logits

    def _head_logits(self, h):  # pragma: no cover - never called in these tests
        raise AssertionError("the unpruned full-vocab projection was called")

    def _mtplx_bind_draft_lm_head(self, head) -> None:
        # qwen4_exp.TextModel._mtplx_bind_draft_lm_head
        self._mtp_draft_head_logits = head.__call__


def _configured_head():
    """The unpruned quantized head the install leaves at the legacy stamp."""

    import mlx.nn as nn

    return nn.QuantizedLinear(64, 32, bias=False, group_size=64, bits=8)


def _installed_text_model(head, *, route="native_mtp_head"):
    """A text model shaped exactly as ``install_frspec_draft_head`` leaves it.

    ``native_mtp_head`` is the shipped ``--full-frspec`` lane (install report
    ``source: native_mtp_head``, ``legacy_swap: False``): the wrapper is
    published at ``_mtplx_frspec_draft_head`` and bound onto the native MTP
    draft-head hook, while ``_mtplx_draft_lm_head`` KEEPS the unpruned
    ``QuantizedLinear`` for legacy consumers.  ``legacy_swap`` is
    ``MTPLX_FRSPEC_LEGACY=1`` on a generic ``mtp_patch`` model: no bind hook,
    the wrapper swapped in globally.  ``none`` is an install whose wrapper
    never reached either route.
    """

    text = _TextModelStandIn()
    text._mtplx_frspec_draft_head = head
    text._mtplx_frspec_full_vocab = int(head._vocab_rows)
    text._mtplx_frspec_ids = head._ids
    text._mtplx_draft_lm_head = _configured_head()
    if route == "native_mtp_head":
        text._mtplx_bind_draft_lm_head(head)
    elif route == "legacy_swap":
        text._mtplx_frspec_saved_head = text._mtplx_draft_lm_head
        text._mtplx_draft_lm_head = head
    elif route != "none":  # pragma: no cover - test typo guard
        raise AssertionError(f"unknown route {route!r}")
    return text


def _runtime(
    rows=prescatter.FRSPEC_ROWS,
    vocab_rows=248_320,
    *,
    ascending=True,
    route="native_mtp_head",
):
    head, ids = _full_vocab_head(rows, vocab_rows, 4242)
    if not ascending:
        shuffled = np.asarray(ids, dtype=np.int64)[::-1].copy()
        object.__setattr__(head, "_ids", mx.array(shuffled, dtype=mx.int32))
    text = _installed_text_model(head, route=route)
    return SimpleNamespace(model=SimpleNamespace(language_model=text)), head, text


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(prescatter, "_ENABLED", True)
    return True


def test_claim_returns_none_when_the_flag_is_off():
    rt, head, _ = _runtime(rows=256, vocab_rows=1_024)
    assert prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE) is None
    assert head._prescatter_capture is False


@pytest.mark.parametrize("route", ["native_mtp_head", "legacy_swap"])
def test_claim_installs_and_arms_at_the_production_width(armed, route):
    rt, head, text = _runtime(route=route)
    plan = prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    assert plan is not None
    assert plan.rows == prescatter.FRSPEC_ROWS
    assert plan.vocab_rows == 248_320
    assert plan.head is head
    assert plan.route == route
    assert head._prescatter_capture is True
    receipt = plan.to_dict()
    assert receipt["installed"] is True
    assert receipt["rows"] == prescatter.FRSPEC_ROWS
    assert receipt["route"] == route
    prescatter.release_draft_route(plan)
    assert head._prescatter_capture is False


def test_claim_succeeds_while_the_legacy_stamp_holds_a_quantized_head(armed):
    """The shipped lane's shape: live via the bind hook, legacy stamp unpruned.

    This is the exact structure the 2026-09-02 ABBA candidate arm refused on
    (``live=QuantizedLinear``) while its own load log said the FR-Spec head was
    installed in full-vocabulary output mode.
    """

    rt, head, text = _runtime(route="native_mtp_head")
    assert type(text._mtplx_draft_lm_head).__name__ == "QuantizedLinear"
    assert text._mtplx_draft_lm_head is not head
    assert text._mtp_draft_head_logits.__self__ is head
    plan = prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    assert plan is not None and plan.route == "native_mtp_head"
    prescatter.release_draft_route(plan)


def test_claim_refuses_a_non_frspec_width(armed):
    rt, _, _ = _runtime(rows=4_096, vocab_rows=16_384)
    with pytest.raises(DraftK20PrescatterIneligible, match="proven only at"):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)


def test_claim_refuses_a_non_ascending_ranked_table(armed):
    rt, _, _ = _runtime(ascending=False)
    with pytest.raises(DraftK20PrescatterIneligible, match="strictly ascending"):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)


def test_claim_refuses_without_an_frspec_head(armed):
    rt = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace()))
    with pytest.raises(DraftK20PrescatterIneligible, match="no FR-Spec draft head"):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)


def test_claim_refuses_when_the_frspec_head_is_not_live(armed):
    """Published but on neither route: no bind hook fired, no legacy swap."""

    rt, head, text = _runtime(route="none")
    with pytest.raises(
        DraftK20PrescatterIneligible, match="not on the live draft route"
    ) as excinfo:
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    message = str(excinfo.value)
    # The refusal names what each probe actually saw.
    assert "_mtp_draft_head_logits=_TextModelStandIn._head_logits" in message
    assert "_mtplx_draft_lm_head=QuantizedLinear" in message
    assert head._prescatter_capture is False


def test_claim_refuses_when_another_head_owns_the_native_hook(armed):
    """The hook is live, but bound to somebody else's head."""

    rt, head, text = _runtime(route="none")
    other, _ = _full_vocab_head(256, 1_024, 99)
    text._mtplx_bind_draft_lm_head(other)
    with pytest.raises(
        DraftK20PrescatterIneligible, match="not on the live draft route"
    ) as excinfo:
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **_ELIGIBLE)
    assert "_FullVocabDraftHead.__call__" in str(excinfo.value)
    assert head._prescatter_capture is False
    assert other._prescatter_capture is False


def test_the_real_install_produces_the_route_the_claim_probes():
    """Pin the probe to ``install_frspec_draft_head``'s ACTUAL output.

    Runs the shipped install at a toy width against the qwen4 draft surface,
    so a future change to how the wrapper is made live fails here rather than
    at the next benchmark.  No model, no Metal: a 1,024-row toy head on the
    CPU stream.
    """

    import mlx.nn as nn

    ids = sorted(int(i) for i in _ranked_ids(256, 1_024, 77))
    text = _TextModelStandIn()
    native = nn.QuantizedLinear(
        64, 1_024, bias=False, group_size=64, bits=8, mode="affine"
    )
    text._mtplx_native_mtp_draft_head = lambda: native
    # draft_lm_head._install_draft_lm_head stamps the configured draft head
    # here before it calls the FR-Spec install.
    text._mtplx_draft_lm_head = _configured_head()

    saved = frspec_draft.load_frspec_ids
    frspec_draft.load_frspec_ids = lambda: list(ids)
    try:
        report = frspec_draft.install_frspec_draft_head(text)
    finally:
        frspec_draft.load_frspec_ids = saved

    assert report["installed"] is True
    assert report["source"] == "native_mtp_head"
    assert report["output_mode"] == "full"
    assert report["legacy_swap"] is False

    head = text._mtplx_frspec_draft_head
    # The install leaves the unpruned head at the legacy stamp -- probing THAT
    # for liveness is the bug this test guards against.
    assert text._mtplx_draft_lm_head is not head
    assert isinstance(text._mtplx_draft_lm_head, nn.QuantizedLinear)
    assert prescatter._live_draft_route(text, head) == "native_mtp_head"
    # And the array the wrapper returns is the one the forward hands the
    # reader, so the stash identity check holds on this route.
    head.arm_prescatter_capture(True)
    dense = text._mtp_draft_head_logits(mx.zeros((1, 1, 64)))
    row = head.take_prescatter_row(dense)
    assert row is not None and int(row.reshape(-1).shape[0]) == 256
    head.arm_prescatter_capture(False)


def test_the_real_install_legacy_swap_is_also_recognised(monkeypatch):
    """``MTPLX_FRSPEC_LEGACY=1`` swaps the wrapper in globally instead."""

    import mlx.nn as nn

    ids = sorted(int(i) for i in _ranked_ids(256, 1_024, 78))
    text = SimpleNamespace()  # no bind hook: the generic mtp_patch surface
    text._mtplx_draft_lm_head = nn.QuantizedLinear(
        64, 1_024, bias=False, group_size=64, bits=8, mode="affine"
    )
    monkeypatch.setattr(frspec_draft, "load_frspec_ids", lambda: list(ids))
    monkeypatch.setattr(frspec_draft, "frspec_legacy_enabled", lambda: True)

    report = frspec_draft.install_frspec_draft_head(text)
    assert report["installed"] is True
    assert report["legacy_swap"] is True
    head = text._mtplx_frspec_draft_head
    assert prescatter._live_draft_route(text, head) == "legacy_swap"


@pytest.mark.parametrize(
    "override, match",
    [
        ({"draft_core": "device"}, "stock draft selector"),
        ({"relaxed_draft_ties": True}, "RELAXED_DRAFT_TIES"),
        ({"frspec_legacy_ids": np.arange(4)}, "FRSPEC_LEGACY"),
        ({"device_k20_route": object()}, "DEVICE_K20"),
        ({"pr391_route": object()}, "PR391"),
        ({"target_prefix_verify": True}, "target-prefix"),
        ({"a3b_target_prefix_route": object()}, "target-prefix"),
        ({"adaptive_width_policy": object()}, "adaptive-width"),
        ({"combine_greedy_draft_read": True}, "dense row"),
        ({"draft_confidence_needed": True}, "dense row"),
        ({"draft_margin_threshold": 0.5}, "dense row"),
        ({"wants_policy_metrics": True}, "dense row"),
        ({"correction_cache_enabled": True}, "correction cache"),
        ({"adapter_ensemble_q": True}, "adapter ensemble"),
        ({"mtp_topk_reranker": object()}, "reranker"),
        ({"penalties_active": True}, "real token id"),
        ({"steer_active": True}, "real token id"),
    ],
)
def test_claim_refuses_every_unsupported_request_term(armed, override, match):
    rt, head, _ = _runtime()
    kwargs = {**_ELIGIBLE, **override}
    with pytest.raises(DraftK20PrescatterIneligible, match=match):
        prescatter.claim_draft_route(rt, draft_sampler=_config(), **kwargs)
    assert head._prescatter_capture is False


def test_claim_refuses_draft_sampler_penalties(armed):
    rt, _, _ = _runtime()
    config = SamplerConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        presence_penalty=0.2,
    )
    with pytest.raises(DraftK20PrescatterIneligible, match="penalties"):
        prescatter.claim_draft_route(rt, draft_sampler=config, **_ELIGIBLE)


def test_claim_refuses_without_top_k(armed):
    rt, _, _ = _runtime()
    with pytest.raises(DraftK20PrescatterIneligible, match="top-k"):
        prescatter.claim_draft_route(
            rt, draft_sampler=_config(top_k=0), **_ELIGIBLE
        )


# ---------------------------------------------------------------------------
# 6. Wiring: the gate is off by default and the receipt exists
# ---------------------------------------------------------------------------


def test_gate_is_off_by_default_in_generation():
    from mtplx import generation

    assert generation._FABLE_DRAFT_K20_PRESCATTER is False
    assert "draft_k20_prescatter" in generation.GenerationStats.__dataclass_fields__


def test_the_draft_loop_reads_the_plan_before_the_stock_reader():
    """The one hot-loop site is guarded by ``is not None``."""

    import inspect

    from mtplx import generation

    source = inspect.getsource(generation.generate_mtpk)
    assert "elif _draft_k20_prescatter_plan is not None:" in source
    assert "_fable_draft_k20_prescatter_read(" in source
    assert "draft_k20_prescatter=_draft_k20_prescatter_receipt," in source
