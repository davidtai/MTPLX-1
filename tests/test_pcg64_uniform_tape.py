from __future__ import annotations

import json

import numpy as np
import pytest

from mtplx.pcg64_tape import (
    DRAWS_PER_CYCLE,
    PCG64UniformTape,
    UniformTapeExhausted,
    _select_weighted_choice_index,
)
from mtplx.sampling import SparseDistribution, sample_from_distribution


def _state(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state, sort_keys=True)


def test_choice_uses_numpy_sum_normalization_at_known_boundary() -> None:
    probabilities = np.array(
        [
            0.2927264538460277,
            0.05644252671156259,
            0.11093093277275656,
            0.009337598431672221,
            0.06682097159765857,
            0.043447601838550544,
            0.2068876536160516,
            0.21340626118572034,
        ],
        dtype=np.float64,
    )
    uniform = np.float64(4_144_211_596_455_494 / 2**53)
    assert _select_weighted_choice_index(probabilities, uniform) == 2


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
        index = _select_weighted_choice_index(probabilities, float(tape.values[offset]))
        actual = int(values[index])

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
    index = _select_weighted_choice_index(probabilities, float(tape.values[offset]))
    actual = int(values[index])

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
    assert tape.values.flags.c_contiguous
    assert tape.values.flags.writeable is False
    assert tape.device_values.flags.writeable is False
    np.testing.assert_array_equal(tape.values, expected_values)
    assert _state(authoritative) == initial_state
    with pytest.raises(ValueError, match="read-only"):
        tape.values[0] = 0.0
    with pytest.raises(ValueError):
        tape.values.setflags(write=True)


def test_device_reservation_exposes_offset_values_and_legacy_indexing() -> None:
    reference = np.random.default_rng(391)
    authoritative = np.random.default_rng(391)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=1)

    reservation = tape.reserve_device_choices(3)

    assert reservation.offset == 0
    assert reservation.values.flags.writeable is False
    np.testing.assert_array_equal(reservation.values, reference.random(3))
    assert tape.values[reservation] == reservation.values[0]
    assert _state(authoritative) == _state(reference)


@pytest.mark.parametrize("committed", [1, 2, 3, 4])
def test_peek_then_commit_advances_only_device_reported_draws(committed: int) -> None:
    reference = np.random.default_rng(391 + committed)
    authoritative = np.random.default_rng(391 + committed)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=4)

    reservation = tape.peek_device_choices(4)

    assert reservation.offset == 0
    np.testing.assert_array_equal(
        reservation.values,
        np.random.default_rng(391 + committed).random(4),
    )
    assert tape.cursor == 0
    assert _state(authoritative) == _state(reference)

    tape.commit_device_choices(reservation, committed)
    reference.random(committed)

    assert tape.cursor == committed
    assert _state(authoritative) == _state(reference)


def test_peek_commit_rejects_stale_or_oversized_device_report() -> None:
    tape = PCG64UniformTape.build(np.random.default_rng(391), max_output_tokens=4)
    reservation = tape.peek_device_choices(4)

    with pytest.raises(ValueError, match="reported draw count"):
        tape.commit_device_choices(reservation, 0)
    with pytest.raises(ValueError, match="reported draw count"):
        tape.commit_device_choices(reservation, 5)

    tape.commit_device_choices(reservation, 2)
    with pytest.raises(RuntimeError, match="stale"):
        tape.commit_device_choices(reservation, 1)


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

    class DerivedGenerator(np.random.Generator):
        pass

    with pytest.raises(TypeError, match="numpy.random.PCG64"):
        PCG64UniformTape.build(
            np.random.Generator(np.random.Philox(0)), max_output_tokens=8
        )
    with pytest.raises(TypeError, match="numpy.random.PCG64"):
        PCG64UniformTape.build(
            np.random.Generator(DerivedPCG64(0)), max_output_tokens=8
        )
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        PCG64UniformTape.build(
            DerivedGenerator(np.random.PCG64(0)), max_output_tokens=8
        )

    monkeypatch.setattr(np, "__version__", "2.4.5")
    with pytest.raises(RuntimeError, match="2.4.4"):
        PCG64UniformTape.build(np.random.default_rng(0), max_output_tokens=8)


def test_output_bound_accepts_zero_and_non_boolean_integrals() -> None:
    zero_tape = PCG64UniformTape.build(np.random.default_rng(0), max_output_tokens=0)
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
    tape = PCG64UniformTape.build(np.random.default_rng(0), max_output_tokens=16_384)

    assert tape.values.shape == (7 * 16_385,)
    assert tape.values.nbytes == 917_560
    with pytest.raises(ValueError, match="16,384"):
        PCG64UniformTape.build(np.random.default_rng(0), max_output_tokens=-1)
    with pytest.raises(ValueError, match="16,384"):
        PCG64UniformTape.build(np.random.default_rng(0), max_output_tokens=16_385)


def test_exhaustion_fails_before_rng_or_cursor_mutation() -> None:
    tape = PCG64UniformTape.build(np.random.default_rng(7), max_output_tokens=0)
    tape.reserve_device_choices(DRAWS_PER_CYCLE)
    state = _state(tape.rng)

    with pytest.raises(UniformTapeExhausted, match="requested 1"):
        tape.random()
    with pytest.raises(UniformTapeExhausted, match="requested 1"):
        tape.choice(np.array([1]), p=np.array([1.0]))
    with pytest.raises(UniformTapeExhausted, match="requested 2"):
        tape.reserve_device_choices(2)

    assert tape.cursor == DRAWS_PER_CYCLE
    assert _state(tape.rng) == state


@pytest.mark.parametrize(
    "values, probabilities",
    [
        (np.array([1, 2]), np.array([0.5])),
        (np.array([1, 2]), np.array([[0.5, 0.5]])),
        (np.array([1, 2]), np.array([0.5, -0.5])),
        (np.array([1, 2]), np.array([0.5, np.nan])),
        (np.array([1, 2]), np.array([0.4, 0.5])),
        (np.array([], dtype=np.int64), np.array([], dtype=np.float64)),
    ],
)
def test_manual_choice_rejects_the_same_malformed_probabilities_as_numpy(
    values: np.ndarray, probabilities: np.ndarray
) -> None:
    reference = np.random.default_rng(13)
    tape = PCG64UniformTape.build(np.random.default_rng(13), max_output_tokens=0)
    with pytest.raises(ValueError):
        reference.choice(values, p=probabilities)
    state = _state(tape.rng)
    with pytest.raises(ValueError):
        tape.choice(values, p=probabilities)
    assert tape.cursor == 0
    assert _state(tape.rng) == state


def test_choice_uses_numpy_probability_tolerance_for_original_float32_dtype() -> None:
    values = np.array([11, 22], dtype=np.int64)
    accepted = np.array([0.5, 0.4999], dtype=np.float32)
    rejected = np.array([0.5, 0.499], dtype=np.float32)
    reference = np.random.default_rng(391)
    authoritative = np.random.default_rng(391)
    tape = PCG64UniformTape.build(authoritative, max_output_tokens=0)

    assert tape.choice(values, p=accepted) == reference.choice(values, p=accepted)
    assert tape.cursor == 1
    assert _state(authoritative) == _state(reference)

    with pytest.raises(ValueError, match="sum to 1"):
        reference.choice(values, p=rejected)
    state = _state(authoritative)
    with pytest.raises(ValueError, match="sum to 1"):
        tape.choice(values, p=rejected)
    assert tape.cursor == 1
    assert _state(authoritative) == state
