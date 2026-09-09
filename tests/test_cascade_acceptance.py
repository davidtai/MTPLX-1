"""CPU tests for speculative-cascade acceptance (a second lossy verify rule).

Rule implemented: Narasimhan, Jitkrittum, Rawat, Kim, Gupta, Menon, and Kumar, "Faster
Cascades via Speculative Decoding" (arXiv:2405.19261 v2, 2024), Section 4.3
Equation (10), the plug-in approximation to the optimal deferral rule of
Equation (8):

    r_OPT(x_<t) = 1  <=>  max_v q(v) < max_v p(v) - alpha * D_TV(p, q)

r = 1 DEFERS to the large/target model p; r = 0 accepts the small/draft model q.
D_TV(p, q) = sum_v max(0, p(v) - q(v)) over the scored top-k support. On defer
the code takes the exact speculative law (min(1, p/q) coin + residual), so the
cascade path is a superset of the exact rule with a draft-accept shortcut.

Note on the divergence sign (Lemma 3): alpha * D_TV is SUBTRACTED, so a larger
disagreement LOWERS the bar and defers LESS. The intuitive "reject when the
draft diverges" holds for a draft that also loses peak confidence (the realistic
diverging MTP head); a draft that stays confident on a wrong token is accepted,
which is the paper's documented behavior and is pinned explicitly below.
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

from mtplx.sampling import (
    SparseDistribution,
    acceptance_probability,
    cascade_defer_decision,
    residual_distribution,
    total_variation,
)

VOCAB = 128


def sp(pairs) -> SparseDistribution:
    ids = np.array([t for t, _ in pairs], dtype=np.int64)
    probs = np.array([p for _, p in pairs], dtype=np.float64)
    return SparseDistribution(ids, probs, VOCAB)


# ---------------------------------------------------------------------------
# The rule (Eq. 10).
# ---------------------------------------------------------------------------


def test_total_variation_matches_paper_definition():
    p = sp([(0, 0.8), (1, 0.2)])
    q = sp([(0, 0.5), (1, 0.5)])
    # sum_v max(0, p-q) = max(0, 0.3) + max(0, -0.3) = 0.3
    assert total_variation(p, q) == pytest.approx(0.3)
    # union support: token the draft scores but the target truncated away counts
    p2 = sp([(0, 1.0)])
    q2 = sp([(0, 0.5), (9, 0.5)])
    assert total_variation(p2, q2) == pytest.approx(0.5)


def test_accepts_when_distributions_agree_even_for_an_atypical_token():
    # q == p: max_q == max_p and D_TV == 0, so the bar max_p - alpha*0 == max_p
    # is not strictly above max_q -> do NOT defer -> accept the draft. The
    # decision uses only max_q/max_p/D_TV, so it is token-independent: an
    # atypical proposed token (low mass) is accepted just the same, which is the
    # cascade advantage over typical acceptance (which would reject it).
    p = sp([(0, 0.9), (1, 0.06), (7, 0.04)])
    q = sp([(0, 0.9), (1, 0.06), (7, 0.04)])
    for alpha in (0.0, 0.1, 0.5, 2.0):
        defer, tv = cascade_defer_decision(p, q, alpha=alpha)
        assert defer is False, alpha
        assert tv == pytest.approx(0.0)


def test_defers_when_draft_diverges_and_loses_confidence():
    # Confident target (peak 0.9 on token 0); the draft has diverged to a spread
    # distribution on other tokens (peak 0.3, high D_TV). With a modest alpha the
    # subtracted penalty is small, so max_q (0.3) < max_p - alpha*D_TV -> defer.
    p = sp([(0, 0.9), (1, 0.1)])
    q = sp([(3, 0.3), (4, 0.25), (5, 0.25), (6, 0.2)])
    defer, tv = cascade_defer_decision(p, q, alpha=0.1)
    assert defer is True
    assert tv > 0.5


def test_alpha_zero_defers_iff_target_strictly_more_confident():
    # alpha=0 removes the divergence penalty: pure peak comparison.
    p = sp([(0, 0.6), (1, 0.4)])
    q_less = sp([(0, 0.5), (1, 0.5)])   # max_q 0.5 < max_p 0.6 -> defer
    q_more = sp([(0, 0.7), (1, 0.3)])   # max_q 0.7 > max_p 0.6 -> accept
    assert cascade_defer_decision(p, q_less, alpha=0.0)[0] is True
    assert cascade_defer_decision(p, q_more, alpha=0.0)[0] is False


def test_higher_alpha_defers_less():
    # Borderline case: raising alpha widens the accept band (monotone), so a
    # position that defers at low alpha stops deferring at high alpha.
    p = sp([(0, 0.7), (1, 0.3)])
    q = sp([(2, 0.5), (3, 0.5)])   # max_q 0.5 < max_p 0.7, D_TV = 0.7
    assert cascade_defer_decision(p, q, alpha=0.0)[0] is True     # 0.5 < 0.7
    assert cascade_defer_decision(p, q, alpha=1.0)[0] is False    # 0.5 < 0.7-0.7=0.0 -> False


def test_paper_counterintuitive_confident_disagreement_is_accepted():
    # Both models confident (peak 0.95) but on OPPOSITE tokens: D_TV = 0.9. Per
    # Eq. (10) with alpha=0.5 the bar is 0.95 - 0.5*0.9 = 0.5, and max_q 0.95 is
    # not below it, so the draft is ACCEPTED. This is the paper's documented
    # "defer less when disagreement is large" behavior (Lemma 3); pinned so a
    # future change to the sign is caught.
    p = sp([(0, 0.95), (1, 0.05)])
    q = sp([(1, 0.95), (0, 0.05)])
    defer, tv = cascade_defer_decision(p, q, alpha=0.5)
    assert tv == pytest.approx(0.9)
    assert defer is False


# ---------------------------------------------------------------------------
# Superset of the exact rule: the deferred path IS the exact law.
# ---------------------------------------------------------------------------


def test_deferred_path_equals_exact_speculative_law():
    # When the rule defers, the caller runs min(1, p/q) + residual(p, q) -- the
    # exact Leviathan-Chen law. Pin that the primitives the cascade branch uses
    # on defer are exactly the exact-rule primitives (so cascade is a strict
    # superset: an exact accept/residual, gated behind the deferral test).
    p = sp([(0, 0.6), (1, 0.3), (2, 0.1)])
    q = sp([(0, 0.2), (1, 0.5), (2, 0.3)])
    token = 1
    # exact accept probability and residual, computed the way both the exact
    # branch and the cascade-defer branch compute them.
    assert acceptance_probability(p, q, token) == pytest.approx(min(1.0, 0.3 / 0.5))
    resid = residual_distribution(p, q)
    # residual mass is norm(max(0, p-q)); token 1 has p<q so it drops out.
    assert resid.probability(1) == pytest.approx(0.0)
    assert resid.probability(0) > 0.0


# ---------------------------------------------------------------------------
# Arming: read at use (served order), default off, mutual exclusion.
# ---------------------------------------------------------------------------


def _clear():
    os.environ.pop("MTPLX_FABLE_CASCADE_THRESHOLD", None)
    os.environ.pop("MTPLX_FABLE_TYPICAL_THRESHOLD", None)


@pytest.fixture(autouse=True)
def _clean_env():
    _clear()
    try:
        yield
    finally:
        _clear()


def test_default_off_and_any_value_including_zero_turns_on():
    from mtplx.generation import _cascade_accept_alpha, _cascade_accept_enabled

    assert _cascade_accept_alpha() is None
    assert _cascade_accept_enabled() is False
    os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = "0"
    assert _cascade_accept_alpha() == 0.0
    assert _cascade_accept_enabled() is True  # explicit 0 is ON, not the off switch
    os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = "0.5"
    assert _cascade_accept_alpha() == pytest.approx(0.5)


def test_served_order_reader_read_at_use():
    # Reproduce the served order: import the generation and server modules
    # FIRST (before any auto-arm/setdefault), then set the env. A reader frozen
    # at import would miss it; read-at-use sees it.
    gen = importlib.import_module("mtplx.generation")
    importlib.import_module("mtplx.server.openai")
    assert gen._cascade_accept_alpha() is None
    os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = "0.25"
    assert gen._cascade_accept_alpha() == pytest.approx(0.25)


def test_mutual_exclusion_fails_loud():
    from mtplx.generation import _assert_lossy_verify_rules_exclusive

    # Either alone is fine.
    os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = "0.3"
    _assert_lossy_verify_rules_exclusive()
    _clear()
    os.environ["MTPLX_FABLE_TYPICAL_THRESHOLD"] = "0.09"
    _assert_lossy_verify_rules_exclusive()
    # Both set -> fail loud.
    os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = "0.3"
    with pytest.raises(ValueError, match="mutually exclusive"):
        _assert_lossy_verify_rules_exclusive()


def test_typical_zero_does_not_conflict_with_cascade():
    # Typical is OFF at delta 0, so cascade + typical=0 is not a conflict.
    from mtplx.generation import _assert_lossy_verify_rules_exclusive

    os.environ["MTPLX_FABLE_TYPICAL_THRESHOLD"] = "0"
    os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = "0.3"
    _assert_lossy_verify_rules_exclusive()  # must not raise


def test_exact_mode_when_off_takes_neither_lossy_branch():
    # Flag off (env unset): cascade disabled, so the verify path is the exact
    # speculative law unchanged. This is the cascade-only PEER of #478; the
    # typical lane's code is not present on this branch (the two are alternative
    # modes), so the exact-off contract is just "cascade off" plus the inert
    # mutual-exclusion guard.
    from mtplx.generation import (
        _assert_lossy_verify_rules_exclusive,
        _cascade_accept_enabled,
    )

    assert _cascade_accept_enabled() is False
    _assert_lossy_verify_rules_exclusive()  # inert here, must not raise


# ---------------------------------------------------------------------------
# Served-order regression guard (coordinator 2026-09-08): the re-parent that
# dropped the typical lane also deleted the [cascade-accept] verdict block from
# generate_mtpk (the served path) -- it survived only in generate_mtpa. The rule
# still engaged (cascade_* counters moved) but the verdict never printed, so the
# arm-G sweep's engagement gate (which reads threshold=/positions= off that line)
# saw nothing. This test drives generate_mtpk on CPU with the cascade lane armed
# via the env reader AT USE (served order: env set here, after import) and a
# temperature>0 mocked verify loop, and asserts the line prints once with
# threshold == alpha and positions > 0. It fails on any tree where the emission
# is missing from the SERVED function.
import re as _re
from pathlib import Path as _Path
from types import SimpleNamespace as _SNS

import mlx.core as _mx
import pytest as _pytest

from mtplx.generation import generate_mtpk as _generate_mtpk
from mtplx.mtp_patch import MTPContract as _MTPContract
from mtplx.runtime import MTPLXRuntime as _MTPLXRuntime
from mtplx.sampling import SamplerConfig as _SamplerConfig


class _VerdictTinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(t)) for t in tokens)


class _VerdictAcceptingMTPModel:
    """Minimal MTP model: deterministic logits favouring token 1 so the draft
    and target agree and the cascade branch accepts (positions > 0)."""

    def __init__(self):
        self.calls = []
        self.mtp = _SNS(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, *, mtp_cache=None,
                         concat_order=None, position_offset=None):
        return hidden_states

    def mtp_forward(self, hidden_states, next_token_ids, *, mtp_cache=None,
                    concat_order=None, return_hidden=False,
                    mtp_hidden_variant=None, position_offset=None):
        length = int(next_token_ids.shape[1])
        hidden = _mx.zeros((1, length, 2), dtype=_mx.float32)
        logits = _mx.zeros((1, length, 4), dtype=_mx.float32) + _mx.array(
            [0.0, 1.0, 0.0, 0.0], dtype=_mx.float32)
        return (logits, hidden) if return_hidden else logits

    def __call__(self, input_ids, *, cache=None, return_hidden=False,
                 hidden_variant=None, emit_logits=True, logits_keep=None):
        self.calls.append(int(input_ids.shape[1]))
        length = int(input_ids.shape[1])
        hidden = _mx.zeros((1, length, 2), dtype=_mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = _mx.zeros((1, keep, 4), dtype=_mx.float32) + _mx.array(
            [0.0, 1.0, 0.0, 0.0], dtype=_mx.float32)
        return (logits, hidden) if return_hidden else logits


def _verdict_runtime(model):
    return _MTPLXRuntime(
        model=model,
        tokenizer=_VerdictTinyTokenizer(),
        model_path=_Path("tiny"),
        mtp_enabled=True,
        contract=_MTPContract(),
    )


def test_served_path_emits_cascade_verdict_with_threshold_and_positions(
    capsys, monkeypatch
):
    previous = _mx.default_device()
    _mx.set_default_device(_mx.cpu)
    try:
        alpha = 0.5
        # Served order: the lane is armed via the env AFTER import; the reader
        # resolves it at use inside generate_mtpk. Batched target arrays make the
        # cascade branch reachable (target_distribution_batch is not None).
        monkeypatch.setenv("MTPLX_FABLE_CASCADE_THRESHOLD", str(alpha))
        monkeypatch.setenv("MTPLX_BATCH_TARGET_ARRAYS", "1")
        monkeypatch.delenv("MTPLX_FABLE_TYPICAL_THRESHOLD", raising=False)

        model = _VerdictAcceptingMTPModel()
        out = _generate_mtpk(
            _verdict_runtime(model),
            [0],
            max_tokens=5,
            sampler=_SamplerConfig(temperature=0.6, top_p=1.0, top_k=1),
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="batched",
            stop_token_ids=set(),
        )
    finally:
        _mx.set_default_device(previous)

    # The rule engaged: VerifyStats cascade fields populate on the served path.
    assert out.stats.cascade_accept_enabled is True
    assert out.stats.cascade_alpha == _pytest.approx(alpha)
    assert out.stats.cascade_positions > 0

    # The verdict line printed once on stderr, with threshold == alpha and
    # positions > 0 (the exact fields the arm-G engagement gate parses).
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if "[cascade-accept]" in ln]
    assert len(lines) == 1, f"expected exactly one verdict line, got {lines!r}"
    line = lines[0]
    m_thr = _re.search(r"threshold=([0-9.eE+-]+)", line)
    m_alpha = _re.search(r"alpha=([0-9.eE+-]+)", line)
    m_pos = _re.search(r"positions=(\d+)", line)
    assert m_thr and m_alpha and m_pos, line
    assert float(m_thr.group(1)) == _pytest.approx(alpha)
    assert float(m_alpha.group(1)) == _pytest.approx(alpha)
    assert float(m_thr.group(1)) == float(m_alpha.group(1))  # threshold == alpha
    assert int(m_pos.group(1)) > 0
    assert int(m_pos.group(1)) == out.stats.cascade_positions
