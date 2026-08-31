from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@dataclass(frozen=True)
class FakeSparseDistribution:
    token_ids: np.ndarray
    probs: np.ndarray
    vocab_size: int = 100


def fake_support(logits, config):
    del logits
    return config.distribution


def fake_sample(probs, rng):
    return int(rng.choice(probs.token_ids, p=probs.probs))


def _source_hash(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _generation(*, support=fake_support, sample=fake_sample):
    return SimpleNamespace(
        sparse_distribution_from_mlx_logits_relaxed_ties=support,
        sample_from_distribution=sample,
    )


def _distribution(probs=(0.1, 0.2, 0.7)) -> FakeSparseDistribution:
    values = np.asarray(probs, dtype=np.float64)
    values = values / np.sum(values, dtype=np.float64)
    values = values / np.sum(values, dtype=np.float64)
    return FakeSparseDistribution(
        token_ids=np.asarray([11, 22, 33], dtype=np.int64),
        probs=values,
    )


def _install_for_fake(generation, *, arm, expected_seed):
    import scripts.pr391_numpy_choice_routes as routes

    with (
        patch.object(
            routes, "REVIEWED_SUPPORT_SOURCE_SHA256", _source_hash(fake_support)
        ),
        patch.object(
            routes, "REVIEWED_SAMPLE_SOURCE_SHA256", _source_hash(fake_sample)
        ),
    ):
        return routes.NumpyChoiceRoute.install(
            generation, arm=arm, expected_seed=expected_seed
        )


def test_import_is_cpu_only():
    import scripts.pr391_numpy_choice_routes as routes

    assert "mlx" not in routes.__dict__
    assert "mtplx" not in routes.__dict__


def test_install_source_gate_is_unconditional_and_public_api_has_no_override():
    from scripts.pr391_numpy_choice_routes import (
        NumpyChoiceRoute,
        ChoiceRouteSourceMismatch,
    )

    assert (
        "expected_source_sha256"
        not in inspect.signature(NumpyChoiceRoute.install).parameters
    )
    generation = _generation()
    original_support = generation.sparse_distribution_from_mlx_logits_relaxed_ties
    original_sample = generation.sample_from_distribution
    with pytest.raises(ChoiceRouteSourceMismatch, match="source SHA-256"):
        NumpyChoiceRoute.install(generation, arm="control", expected_seed=1)
    assert (
        generation.sparse_distribution_from_mlx_logits_relaxed_ties is original_support
    )
    assert generation.sample_from_distribution is original_sample


def test_install_pins_expected_seed_and_does_not_accept_request_rng():
    from scripts.pr391_numpy_choice_routes import NumpyChoiceRoute

    parameters = inspect.signature(NumpyChoiceRoute.install).parameters
    assert "expected_seed" in parameters
    assert "rng" not in parameters


@pytest.mark.parametrize("mode", ["wrong_seed", "advanced_state"])
def test_first_primary_passthrough_validates_expected_pcg64_start_state(mode):
    from scripts.pr391_numpy_choice_routes import ChoiceRouteRNGError

    generation = _generation()
    route = _install_for_fake(generation, arm="control", expected_seed=41)
    rng = np.random.default_rng(42 if mode == "wrong_seed" else 41)
    if mode == "advanced_state":
        rng.random()
    try:
        with pytest.raises(ChoiceRouteRNGError, match="expected seed.*start state"):
            generation.sample_from_distribution(_distribution(), rng)
    finally:
        route.close()


def test_finish_requires_observed_request_rng_and_installed_restored_lifecycle():
    from scripts.pr391_numpy_choice_routes import ChoiceRouteRNGError

    generation = _generation()
    route = _install_for_fake(generation, arm="control", expected_seed=43)
    route.close()
    with pytest.raises(ChoiceRouteRNGError, match="request RNG was observed"):
        route.finish_receipt(
            stats={
                "drafted_tokens": 0,
                "accepted_drafts": 0,
                "verify_calls": 0,
                "correction_tokens": 0,
                "bonus_tokens": 0,
            }
        )


def test_install_requires_exact_numpy_2_4_4(monkeypatch):
    import scripts.pr391_numpy_choice_routes as routes

    monkeypatch.setattr(np, "__version__", "2.4.5")
    with pytest.raises(RuntimeError, match="exact NumPy 2.4.4"):
        routes.NumpyChoiceRoute.install(_generation(), arm="control", expected_seed=1)


def test_proposal_identity_is_unchanged_and_mismatch_fails_closed():
    from scripts.pr391_numpy_choice_routes import ChoiceRouteAssociationError

    distribution = _distribution()
    other = _distribution()
    generation = _generation()
    rng = np.random.default_rng(3)
    route = _install_for_fake(generation, arm="control", expected_seed=3)
    try:
        generation.sample_from_distribution(distribution, rng)
        observed = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            object(), SimpleNamespace(distribution=distribution)
        )
        assert observed is distribution
        with pytest.raises(
            ChoiceRouteAssociationError, match="pending object identity"
        ):
            generation.sample_from_distribution(other, rng)
        snapshot = route.diagnostic_snapshot(
            stats={
                "drafted_tokens": 1,
                "accepted_drafts": 0,
                "verify_calls": 1,
                "correction_tokens": 1,
                "bonus_tokens": 0,
            }
        )
        assert snapshot["receipt_kind"] == "failed_diagnostic"
        assert "current_pcg64_state_sha256" in snapshot
        assert "final_pcg64_state_sha256" not in snapshot
        assert snapshot["route_counts"]["failures"] == 1
        assert snapshot["route_counts"]["pending"] == 1
        route.close()
        with pytest.raises(ChoiceRouteAssociationError, match="pending|failures"):
            route.finish_receipt(
                stats={
                    "drafted_tokens": 1,
                    "accepted_drafts": 0,
                    "verify_calls": 1,
                    "correction_tokens": 1,
                    "bonus_tokens": 0,
                }
            )
    finally:
        route.close()


def test_second_proposal_before_immediate_sample_fails_closed():
    from scripts.pr391_numpy_choice_routes import ChoiceRouteAssociationError

    generation = _generation()
    route = _install_for_fake(generation, arm="control", expected_seed=4)
    try:
        config = SimpleNamespace(distribution=_distribution())
        generation.sparse_distribution_from_mlx_logits_relaxed_ties(object(), config)
        with pytest.raises(ChoiceRouteAssociationError, match="immediate pending"):
            generation.sparse_distribution_from_mlx_logits_relaxed_ties(
                object(), config
            )
    finally:
        route.close()


def test_control_calls_original_and_preserves_rng_state():
    distribution = _distribution()
    route_rng = np.random.default_rng(20)
    reference_rng = np.random.default_rng(20)
    generation = _generation()
    route = _install_for_fake(generation, arm="control", expected_seed=20)
    try:
        generation.sample_from_distribution(distribution, route_rng)
        fake_sample(distribution, reference_rng)
        returned = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            object(), SimpleNamespace(distribution=distribution)
        )
        selected = generation.sample_from_distribution(returned, route_rng)
        expected = fake_sample(distribution, reference_rng)
        assert selected == expected
        assert route_rng.bit_generator.state == reference_rng.bit_generator.state
    finally:
        route.close()


def test_matched_sample_callable_is_prebound_without_hot_arm_dispatch():
    from scripts.pr391_numpy_choice_routes import NumpyChoiceRoute

    source = inspect.getsource(NumpyChoiceRoute._sample_wrapper)
    assert "if self._arm" not in source
    assert "if self._rng" not in source
    assert "RouteArm." not in source
    assert "self._matched_sample" in source
    assert "self._rng_observer(rng)" in source


def test_wrappers_preserve_original_keyword_api():
    distribution = _distribution()
    route_rng = np.random.default_rng(21)
    reference_rng = np.random.default_rng(21)
    generation = _generation()
    route = _install_for_fake(generation, arm="control", expected_seed=21)
    try:
        generation.sample_from_distribution(probs=distribution, rng=route_rng)
        fake_sample(distribution, reference_rng)
        returned = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            logits=object(), config=SimpleNamespace(distribution=distribution)
        )
        selected = generation.sample_from_distribution(probs=returned, rng=route_rng)
        assert selected == fake_sample(distribution, reference_rng)
    finally:
        route.close()


@pytest.mark.parametrize("seed", range(100))
def test_reduced_exact_float64_matches_original_token_and_pcg64_cursor(seed):
    distribution = _distribution((0.10000000149011612, 0.20000000298023224, 0.7))
    route_rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed)
    generation = _generation()
    route = _install_for_fake(
        generation, arm="reduced-exact-float64", expected_seed=seed
    )
    try:
        generation.sample_from_distribution(distribution, route_rng)
        fake_sample(distribution, reference_rng)
        returned = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            object(), SimpleNamespace(distribution=distribution)
        )
        selected = generation.sample_from_distribution(returned, route_rng)
        assert selected == fake_sample(distribution, reference_rng)
        assert route_rng.bit_generator.state == reference_rng.bit_generator.state
    finally:
        route.close()


def test_reduced_float32_is_private_exact_uniform_schedule_and_does_not_mutate_q():
    from scripts.pr391_float32_choice_drift import (
        ReducedFloat32Row,
        select_reduced_float32_token,
    )

    distribution = _distribution((0.1, 0.2, 0.7))
    original_ids = distribution.token_ids.copy()
    original_q_bits = distribution.probs.view(np.uint64).copy()
    route_rng = np.random.default_rng(391)
    reference_rng = np.random.default_rng(391)
    fake_sample(distribution, reference_rng)
    uniform = float(reference_rng.random())
    private = distribution.probs.astype(np.float32)
    first_total = np.sum(private, dtype=np.float32)
    private = (private / first_total).astype(np.float32)
    second_total = np.sum(private, dtype=np.float32)
    if second_total.view(np.uint32) != np.float32(1.0).view(np.uint32):
        private = (private / second_total).astype(np.float32)
    expected_row = ReducedFloat32Row(
        token_ids=distribution.token_ids,
        probabilities=private,
        raw_cdf=np.cumsum(private, dtype=np.float32),
        second_normalization_skipped=bool(
            second_total.view(np.uint32) == np.float32(1.0).view(np.uint32)
        ),
    )
    expected = select_reduced_float32_token(expected_row, uniform, cast_uniform=False)

    generation = _generation()
    route = _install_for_fake(
        generation, arm="reduced-float32-test-only", expected_seed=391
    )
    try:
        generation.sample_from_distribution(distribution, route_rng)
        returned = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            object(), SimpleNamespace(distribution=distribution)
        )
        assert generation.sample_from_distribution(returned, route_rng) == expected
        np.testing.assert_array_equal(distribution.token_ids, original_ids)
        np.testing.assert_array_equal(
            distribution.probs.view(np.uint64), original_q_bits
        )
        assert route_rng.bit_generator.state == reference_rng.bit_generator.state
    finally:
        route.close()


def test_nonpending_sample_passthrough_and_close_restore_originals():
    generation = _generation()
    original_support = generation.sparse_distribution_from_mlx_logits_relaxed_ties
    original_sample = generation.sample_from_distribution
    route_rng = np.random.default_rng(8)
    reference_rng = np.random.default_rng(8)
    route = _install_for_fake(generation, arm="reduced-exact-float64", expected_seed=8)
    assert generation.sample_from_distribution(
        _distribution(), route_rng
    ) == fake_sample(_distribution(), reference_rng)
    route.close()
    assert (
        generation.sparse_distribution_from_mlx_logits_relaxed_ties is original_support
    )
    assert generation.sample_from_distribution is original_sample


def test_matched_route_requires_installed_request_rng_identity():
    from scripts.pr391_numpy_choice_routes import ChoiceRouteRNGError

    generation = _generation()
    route_rng = np.random.default_rng(10)
    route = _install_for_fake(generation, arm="control", expected_seed=10)
    try:
        generation.sample_from_distribution(_distribution(), route_rng)
        returned = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            object(), SimpleNamespace(distribution=_distribution())
        )
        with pytest.raises(ChoiceRouteRNGError, match="object identity"):
            generation.sample_from_distribution(returned, np.random.default_rng(10))
    finally:
        route.close()


def test_receipt_hashes_states_counts_and_derives_exact_hit_miss_definitions():
    distribution = _distribution()
    rng = np.random.default_rng(12)
    initial_state = copy.deepcopy(rng.bit_generator.state)
    generation = _generation()
    route = _install_for_fake(generation, arm="reduced-exact-float64", expected_seed=12)
    generation.sample_from_distribution(distribution, rng)
    returned = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
        object(), SimpleNamespace(distribution=distribution)
    )
    generation.sample_from_distribution(returned, rng)
    with pytest.raises(Exception, match="closed|restored"):
        route.finish_receipt(
            stats={
                "drafted_tokens": 2,
                "accepted_drafts": 1,
                "verify_calls": 2,
                "correction_tokens": 1,
                "bonus_tokens": 2,
            }
        )
    route.close()
    receipt = route.finish_receipt(
        stats={
            "drafted_tokens": 2,
            "accepted_drafts": 1,
            "verify_calls": 2,
            "correction_tokens": 1,
            "bonus_tokens": 2,
        }
    )

    assert receipt["arm"] == "reduced-exact-float64"
    assert receipt["receipt_kind"] == "final_success"
    assert receipt["start_pcg64_state_sha256"] != receipt["final_pcg64_state_sha256"]
    assert "current_pcg64_state_sha256" not in receipt
    assert len(receipt["start_pcg64_state_sha256"]) == 64
    assert receipt["route_counts"] == {
        "matched": 1,
        "passthrough": 1,
        "arm": {
            "control": 0,
            "reduced-exact-float64": 1,
            "reduced-float32-test-only": 0,
        },
        "pending": 0,
        "failures": 0,
    }
    assert receipt["hit_miss"] == {
        "definitions": {
            "draft_opportunities": "stats.drafted_tokens",
            "draft_hits": "identity_matched_route_calls",
            "verifier_choice_opportunities": (
                "stats.correction_tokens + stats.bonus_tokens"
            ),
            "verifier_hits": "always_zero_draft_only_constructor_route",
        },
        "draft": {"opportunities": 2, "hits": 1, "misses": 1, "hit_rate": 0.5},
        "verifier": {
            "opportunities": 3,
            "hits": 0,
            "misses": 3,
            "hit_rate": 0.0,
        },
    }
    assert receipt["stats"]["accepted_drafts"] == 1
    assert initial_state != rng.bit_generator.state


def test_float32_receipt_is_explicitly_test_only_and_ineligible_for_retention():
    generation = _generation()
    route = _install_for_fake(
        generation,
        arm="reduced-float32-test-only",
        expected_seed=14,
    )
    generation.sample_from_distribution(_distribution(), np.random.default_rng(14))
    route.close()
    receipt = route.finish_receipt(
        stats={
            "drafted_tokens": 0,
            "accepted_drafts": 0,
            "verify_calls": 0,
            "correction_tokens": 0,
            "bonus_tokens": 0,
        }
    )
    assert receipt["policy"]["evaluation_scope"] == "benchmark_experiment_only"
    assert receipt["policy"]["retention_eligible"] is False
    assert receipt["schedule"]["uniform"] == "direct_one_draw_numpy_pcg64_float64"
    assert receipt["schedule"]["cast_uniform"] is False
