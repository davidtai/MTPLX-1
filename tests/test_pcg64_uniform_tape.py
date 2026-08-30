from __future__ import annotations

import json

import numpy as np
import pytest

from mtplx.pcg64_tape import PCG64UniformTape
from mtplx.sampling import SparseDistribution, sample_from_distribution


def _state(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state, sort_keys=True)


def _inverse_cdf_choice(
    values: np.ndarray,
    probabilities: np.ndarray,
    uniform: float,
) -> int:
    cdf = np.cumsum(probabilities) / np.sum(probabilities)
    return int(values[np.searchsorted(cdf, uniform, side="right")])


@pytest.mark.parametrize("support", range(1, 41))
def test_reserved_uniform_matches_weighted_choice_and_state(support: int) -> None:
    probabilities = np.arange(1, support + 1, dtype=np.float64)
    probabilities /= probabilities.sum()
    values = np.arange(100, 100 + support, dtype=np.int64)

    for seed in (0, 1, 7, 29, 20260830, 2**63 + support):
        reference = np.random.default_rng(seed)
        authoritative = np.random.default_rng(seed)
        tape = PCG64UniformTape.build(authoritative, max_output_tokens=32)

        expected = int(reference.choice(values, p=probabilities))
        offset = tape.reserve_device_choices(1)
        actual = _inverse_cdf_choice(values, probabilities, tape.values[offset])

        assert actual == expected
        assert tape.cursor == 1
        assert _state(authoritative) == _state(reference)


@pytest.mark.parametrize("seed", [0, 1, 7, 29, 20260830])
def test_zero_mass_filtered_support_matches_weighted_choice(seed: int) -> None:
    all_values = np.array([3, 5, 8, 13, 21, 34, 55], dtype=np.int64)
    all_probabilities = np.array(
        [0.0, 0.125, 0.0, 0.375, 0.0, 0.5, 0.0], dtype=np.float64
    )
    keep = all_probabilities > 0
    values = all_values[keep]
    probabilities = all_probabilities[keep]
    reference = np.random.default_rng(seed)
    authoritative = np.random.default_rng(seed)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=8)

    expected = int(reference.choice(all_values, p=all_probabilities))
    offset = tape.reserve_device_choices(1)
    actual = _inverse_cdf_choice(values, probabilities, tape.values[offset])

    assert actual == expected
    assert _state(authoritative) == _state(reference)


def test_build_exposes_initial_tape_without_advancing_authoritative_rng() -> None:
    authoritative = np.random.default_rng(20260830)
    reference = np.random.default_rng(20260830)
    expected_values = reference.random(7 * (12 + 1), dtype=np.float64)
    initial_state = _state(authoritative)

    tape = PCG64UniformTape.build(authoritative, max_output_tokens=12)

    assert tape.rng is authoritative
    assert tape.cursor == 0
    assert tape.values.dtype == np.float64
    np.testing.assert_array_equal(tape.values, expected_values)
    assert _state(authoritative) == initial_state


def test_random_and_choice_delegate_to_authoritative_generator() -> None:
    reference = np.random.default_rng(17)
    authoritative = np.random.default_rng(17)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=8)
    values = np.array([4, 9], dtype=np.int64)
    probabilities = np.array([0.25, 0.75], dtype=np.float64)

    assert tape.random() == float(reference.random())
    assert tape.choice(values, p=probabilities) == reference.choice(
        values, p=probabilities
    )
    assert tape.cursor == 2
    assert _state(authoritative) == _state(reference)


@pytest.mark.parametrize("sparse", [False, True])
def test_sampling_protocol_uses_tape_weighted_choice(sparse: bool) -> None:
    reference = np.random.default_rng(43)
    authoritative = np.random.default_rng(43)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=8)
    probabilities = np.array([0.1, 0.0, 0.3, 0.6], dtype=np.float64)
    if sparse:
        distribution = SparseDistribution(
            np.array([4, 9, 17, 23], dtype=np.int64), probabilities, 32
        )
    else:
        distribution = probabilities

    expected = sample_from_distribution(distribution, reference)
    actual = sample_from_distribution(distribution, tape)

    assert actual == expected
    assert tape.cursor == 1
    assert _state(authoritative) == _state(reference)


def test_d3_accept_then_correction_keeps_one_global_cursor() -> None:
    reference = np.random.default_rng(29)
    authoritative = np.random.default_rng(29)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=16)
    correction_values = np.array([4, 9], dtype=np.int64)
    correction_probabilities = np.array([0.25, 0.75], dtype=np.float64)

    assert tape.reserve_device_choices(3) == 0
    reference.random(3)
    assert tape.random() == float(reference.random())
    assert tape.choice(
        correction_values, p=correction_probabilities
    ) == reference.choice(correction_values, p=correction_probabilities)

    assert tape.cursor == 5
    assert tape.reserve_device_choices(3) == 5
    reference.random(3)
    assert tape.cursor == 8
    assert _state(authoritative) == _state(reference)


def test_d3_all_accepted_then_bonus_keeps_one_global_cursor() -> None:
    reference = np.random.default_rng(31)
    authoritative = np.random.default_rng(31)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=16)
    bonus_values = np.array([2, 7, 8], dtype=np.int64)
    bonus_probabilities = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    assert tape.reserve_device_choices(3) == 0
    reference.random(3)
    for _ in range(3):
        assert tape.random() == float(reference.random())
    assert tape.choice(bonus_values, p=bonus_probabilities) == reference.choice(
        bonus_values, p=bonus_probabilities
    )

    assert tape.cursor == 7
    assert tape.reserve_device_choices(3) == 7
    reference.random(3)
    assert tape.cursor == 10
    assert _state(authoritative) == _state(reference)


@pytest.mark.parametrize("host_draws", [0, 1, 2, 3])
def test_early_stop_or_cancellation_needs_no_state_repair(host_draws: int) -> None:
    reference = np.random.default_rng(101 + host_draws)
    authoritative = np.random.default_rng(101 + host_draws)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=16)

    assert tape.reserve_device_choices(3) == 0
    reference.random(3)
    for _ in range(host_draws):
        assert tape.random() == float(reference.random())

    assert tape.cursor == 3 + host_draws
    assert _state(authoritative) == _state(reference)
    assert authoritative.random() == reference.random()


def test_exact_pcg64_and_numpy_version_are_required(monkeypatch) -> None:
    class DerivedPCG64(np.random.PCG64):
        pass

    with pytest.raises(TypeError, match="numpy.random.PCG64"):
        PCG64UniformTape.build(
            np.random.Generator(np.random.Philox(0)), max_output_tokens=8
        )
    with pytest.raises(TypeError, match="numpy.random.PCG64"):
        PCG64UniformTape.build(
            np.random.Generator(DerivedPCG64(0)), max_output_tokens=8
        )

    monkeypatch.setattr(np, "__version__", "2.4.5")
    with pytest.raises(RuntimeError, match="2.4.4"):
        PCG64UniformTape.build(np.random.default_rng(0), max_output_tokens=8)


def test_output_bound_accepts_zero_and_non_boolean_integrals() -> None:
    zero_tape = PCG64UniformTape.build(
        np.random.default_rng(0), max_output_tokens=0
    )
    numpy_integer_tape = PCG64UniformTape.build(
        np.random.default_rng(0), max_output_tokens=np.int64(2)
    )

    assert zero_tape.values.shape == (7,)
    assert numpy_integer_tape.values.shape == (21,)


@pytest.mark.parametrize(
    "invalid_max_output_tokens",
    [1.5, -0.2, 16_384.9, "2", True, False],
)
def test_output_bound_rejects_non_exact_integers(invalid_max_output_tokens) -> None:
    with pytest.raises(TypeError, match="integer"):
        PCG64UniformTape.build(
            np.random.default_rng(0),
            max_output_tokens=invalid_max_output_tokens,
        )


def test_direct_construction_cannot_bypass_build_validation() -> None:
    with pytest.raises(TypeError, match=r"\.build"):
        PCG64UniformTape(
            rng=np.random.default_rng(0),
            values=np.zeros(7, dtype=np.float64),
        )
    with pytest.raises(TypeError, match=r"\.build"):
        PCG64UniformTape()


def test_16k_bound_is_under_one_mib_and_oversize_is_rejected() -> None:
    tape = PCG64UniformTape.build(
        np.random.default_rng(0), max_output_tokens=16_384
    )

    assert tape.values.shape == (7 * 16_385,)
    assert tape.values.nbytes == 917_560
    with pytest.raises(ValueError, match="16,384"):
        PCG64UniformTape.build(
            np.random.default_rng(0), max_output_tokens=-1
        )
    with pytest.raises(ValueError, match="16,384"):
        PCG64UniformTape.build(
            np.random.default_rng(0), max_output_tokens=16_385
        )
