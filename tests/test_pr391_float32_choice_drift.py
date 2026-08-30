from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_exact_host_row_preserves_operation_order() -> None:
    from scripts.pr391_float32_choice_drift import prepare_exact_host_row

    row = prepare_exact_host_row(
        candidate_ids=np.array([9, 7, 2, 5], dtype=np.int64),
        candidate_values=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        candidate_probs=np.array([0.375, 0.625, 0.0, 0.0], dtype=np.float32),
        top_p=0.65,
    )

    # Rank is value-descending: 9, 7, 2, 5. Cumulative-before retains 9 and 7
    # (the crossing item), positive filtering removes zero-mass rows, and the
    # final categorical support is reordered to ascending token ID.
    np.testing.assert_array_equal(row.token_ids, np.array([7, 9], dtype=np.int64))
    np.testing.assert_allclose(row.probabilities, [0.625, 0.375], rtol=0, atol=0)
    np.testing.assert_allclose(row.cdf, [0.625, 1.0], rtol=0, atol=0)


def test_exact_host_sums_in_rank_order_before_token_id_sort() -> None:
    from scripts.pr391_float32_choice_drift import prepare_exact_host_row

    row = prepare_exact_host_row(
        candidate_ids=np.array([102, 101, 100], dtype=np.int64),
        candidate_values=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        candidate_probs=np.array(
            [
                1.0439006430107423e-12,
                3.3137047793230234e-12,
                0.004913488402962685,
            ],
            dtype=np.float32,
        ),
        top_p=1.0,
    )

    # Literal _serial_row_distribution semantics: the first total is reduced
    # in value-ranked order (102, 101, 100), then the support is sorted. Moving
    # that first reduction after the sort changes this boundary by one binary64
    # ULP: 0x1.fffffff861c2d -> 0x1.fffffff861c2e.
    assert float(row.cdf[0]).hex() == "0x1.fffffff861c2ep-1"


def test_top_p_uses_cumulative_before_and_keeps_crossing_item() -> None:
    from scripts.pr391_float32_choice_drift import prepare_exact_host_row

    row = prepare_exact_host_row(
        candidate_ids=np.array([30, 10, 20], dtype=np.int64),
        candidate_values=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        candidate_probs=np.array([0.4, 0.4, 0.2], dtype=np.float32),
        top_p=0.5,
    )

    # Token 10 starts below 0.5 and is retained even though it crosses 0.5.
    np.testing.assert_array_equal(row.token_ids, np.array([10, 30]))


def test_top_p_one_bypasses_nucleus_filter_for_both_schedules() -> None:
    from scripts.pr391_float32_choice_drift import (
        prepare_exact_host_row,
        prepare_float32_row,
    )

    ids = np.array([1, 2, 3], dtype=np.int64)
    values = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs = np.array([0.6, 0.4000001, 1e-8], dtype=np.float32)

    exact = prepare_exact_host_row(ids, values, probs, top_p=1.0)
    candidate = prepare_float32_row(ids, values, probs, top_p=1.0)

    np.testing.assert_array_equal(exact.token_ids, ids)
    np.testing.assert_array_equal(candidate.token_ids, ids)


def test_equal_values_are_ranked_by_ascending_token_id_before_top_p() -> None:
    from scripts.pr391_float32_choice_drift import prepare_exact_host_row

    row = prepare_exact_host_row(
        candidate_ids=np.array([8, 3, 5], dtype=np.int64),
        candidate_values=np.ones(3, dtype=np.float32),
        candidate_probs=np.array([0.6, 0.3, 0.1], dtype=np.float32),
        top_p=0.2,
    )

    # All values tie, so token 3 is the first (and only retained) candidate.
    np.testing.assert_array_equal(row.token_ids, np.array([3]))


def _boundary_drift_fixture():
    from scripts.pr391_float32_choice_drift import (
        prepare_exact_host_row,
        prepare_float32_row,
    )

    ids = np.array([11, 22, 33], dtype=np.int64)
    values = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    exact = prepare_exact_host_row(ids, values, probs, top_p=1.0)
    candidate = prepare_float32_row(ids, values, probs, top_p=1.0)
    return exact, candidate


def test_float32_has_a_nonzero_categorical_boundary_drift() -> None:
    from scripts.pr391_float32_choice_drift import row_disagreement

    exact, candidate = _boundary_drift_fixture()
    expected = np.sum(np.abs(exact.cdf[:-1] - candidate.cdf[:-1].astype(np.float64)))
    disagreement, max_shift = row_disagreement(exact, candidate)

    assert disagreement == pytest.approx(expected, rel=0, abs=1e-18)
    assert disagreement > 0.0
    assert max_shift > 0.0


def test_float32_reduced_normalization_schedule_is_bit_identical() -> None:
    from scripts.pr391_float32_choice_drift import prepare_float32_row

    ids = np.array([11, 22, 33], dtype=np.int64)
    values = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    first_total = np.sum(probs, dtype=np.float32)
    first_normalized = (probs / first_total).astype(np.float32)
    sparse_total = np.sum(first_normalized, dtype=np.float32)
    assert sparse_total.view(np.uint32) == np.float32(1.0).view(np.uint32)

    # The reduced schedule skips this division when sparse_total is exactly 1.
    # The old unconditional divide-by-one and the explicit skip are bit-equal.
    unconditional_second = (first_normalized / sparse_total).astype(np.float32)
    explicit_skip = first_normalized
    np.testing.assert_array_equal(
        unconditional_second.view(np.uint32), explicit_skip.view(np.uint32)
    )

    row = prepare_float32_row(ids, values, probs, top_p=1.0)
    np.testing.assert_array_equal(
        row.probabilities.view(np.uint32), explicit_skip.view(np.uint32)
    )
    predivide_cdf = np.cumsum(explicit_skip, dtype=np.float32)
    expected_boundaries = (predivide_cdf / predivide_cdf[-1]).astype(np.float32)
    np.testing.assert_array_equal(
        row.cdf.view(np.uint32), expected_boundaries.view(np.uint32)
    )


def test_float32_second_normalization_runs_for_nonunit_sum_bits() -> None:
    from scripts.pr391_float32_choice_drift import analyze_arrays, prepare_float32_row

    ids = np.array([11, 22], dtype=np.int64)
    values = np.array([2.0, 1.0], dtype=np.float32)
    probs = np.array([0.7369865, 0.8431228], dtype=np.float32)
    first_normalized = (probs / np.sum(probs, dtype=np.float32)).astype(np.float32)
    sparse_total = np.sum(first_normalized, dtype=np.float32)
    assert sparse_total.view(np.uint32) != np.float32(1.0).view(np.uint32)
    expected = (first_normalized / sparse_total).astype(np.float32)

    row = prepare_float32_row(ids, values, probs, top_p=1.0)
    np.testing.assert_array_equal(
        row.probabilities.view(np.uint32), expected.view(np.uint32)
    )

    report = analyze_arrays(
        ids[None, :],
        values[None, :],
        probs[None, :],
        np.array([0.25], dtype=np.float64),
        top_p=1.0,
    )
    reduced = report["division_accounting"]["reduced_float32"]
    assert reduced["passes_per_row"] == [3]
    assert reduced["division_count_per_row"] == [6]
    assert reduced["second_normalization_skip_count"] == 0


def test_uniform_inside_boundary_shift_changes_exact_uniform_choice() -> None:
    from scripts.pr391_float32_choice_drift import select_token

    exact, candidate = _boundary_drift_fixture()
    lower, upper = sorted((float(exact.cdf[0]), float(candidate.cdf[0])))
    uniform = lower + (upper - lower) / 2.0

    assert select_token(exact, uniform, cast_uniform=False) == 11
    assert select_token(candidate, uniform, cast_uniform=False) == 22


def test_uniform_outside_boundary_shift_keeps_choice() -> None:
    from scripts.pr391_float32_choice_drift import select_token

    exact, candidate = _boundary_drift_fixture()

    assert select_token(exact, 0.05, cast_uniform=False) == 11
    assert select_token(candidate, 0.05, cast_uniform=False) == 11


def test_float32_uniform_cast_is_measured_separately() -> None:
    from scripts.pr391_float32_choice_drift import select_token

    _, candidate = _boundary_drift_fixture()
    boundary = float(candidate.cdf[0])
    uniform = float(np.nextafter(boundary, 0.0))

    assert select_token(candidate, uniform, cast_uniform=False) == 11
    assert select_token(candidate, uniform, cast_uniform=True) == 22


def test_sub_grid_boundary_shift_has_zero_pcg64_grid_disagreement() -> None:
    from scripts.pr391_float32_choice_drift import (
        PreparedRow,
        pcg64_grid_disagreement_count,
        row_disagreement,
    )

    boundary = 0.1
    shifted = float(np.nextafter(boundary, 1.0))
    exact = PreparedRow(
        np.array([1, 2], dtype=np.int64),
        np.array([boundary, 1.0 - boundary], dtype=np.float64),
        np.array([boundary, 1.0], dtype=np.float64),
    )
    candidate = PreparedRow(
        np.array([1, 2], dtype=np.int64),
        np.array([shifted, 1.0 - shifted], dtype=np.float64),
        np.array([shifted, 1.0], dtype=np.float64),
    )

    assert row_disagreement(exact, candidate)[0] > 0.0
    assert (
        pcg64_grid_disagreement_count(exact, candidate, candidate_cast_uniform=False)
        == 0
    )


def test_one_grid_boundary_shift_has_exactly_one_pcg64_mismatch() -> None:
    from scripts.pr391_float32_choice_drift import (
        PCG64_GRID_SIZE,
        PreparedRow,
        pcg64_grid_disagreement_count,
    )

    boundary = (PCG64_GRID_SIZE // 3) / PCG64_GRID_SIZE
    shifted = float(np.nextafter(boundary, 1.0))
    exact = PreparedRow(
        np.array([1, 2], dtype=np.int64),
        np.array([boundary, 1.0 - boundary], dtype=np.float64),
        np.array([boundary, 1.0], dtype=np.float64),
    )
    candidate = PreparedRow(
        np.array([1, 2], dtype=np.int64),
        np.array([shifted, 1.0 - shifted], dtype=np.float64),
        np.array([shifted, 1.0], dtype=np.float64),
    )

    assert (
        pcg64_grid_disagreement_count(exact, candidate, candidate_cast_uniform=False)
        == 1
    )


def test_pcg64_grid_count_handles_changed_support() -> None:
    from scripts.pr391_float32_choice_drift import (
        PCG64_GRID_SIZE,
        PreparedRow,
        pcg64_grid_disagreement_count,
    )

    exact = PreparedRow(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([0.25, 0.5, 0.25], dtype=np.float64),
        np.array([0.25, 0.75, 1.0], dtype=np.float64),
    )
    candidate = PreparedRow(
        np.array([1, 3], dtype=np.int64),
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.5, 1.0], dtype=np.float32),
    )

    expected = PCG64_GRID_SIZE // 2
    assert (
        pcg64_grid_disagreement_count(exact, candidate, candidate_cast_uniform=False)
        == expected
    )
    assert (
        pcg64_grid_disagreement_count(exact, candidate, candidate_cast_uniform=True)
        == expected
    )


def test_float32_uniform_cast_grid_count_is_not_approximated() -> None:
    from scripts.pr391_float32_choice_drift import (
        PreparedRow,
        pcg64_grid_disagreement_count,
    )

    boundary = np.float32(0.1)
    exact = PreparedRow(
        np.array([1, 2], dtype=np.int64),
        np.array([boundary, 1.0 - boundary], dtype=np.float64),
        np.array([boundary, 1.0], dtype=np.float64),
    )
    candidate = PreparedRow(
        np.array([1, 2], dtype=np.int64),
        np.array([boundary, np.float32(1.0) - boundary], dtype=np.float32),
        np.array([boundary, 1.0], dtype=np.float32),
    )

    assert (
        pcg64_grid_disagreement_count(exact, candidate, candidate_cast_uniform=False)
        == 0
    )
    assert (
        pcg64_grid_disagreement_count(exact, candidate, candidate_cast_uniform=True) > 0
    )


def test_float32_top_p_can_change_retained_membership() -> None:
    from scripts.pr391_float32_choice_drift import (
        prepare_exact_host_row,
        prepare_float32_row,
    )

    ids = np.array([11, 22, 33], dtype=np.int64)
    values = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    exact_cumulative = float(probs[0]) + float(probs[1])
    float32_cumulative = float(np.cumsum(probs, dtype=np.float32)[1])
    top_p = exact_cumulative + (float32_cumulative - exact_cumulative) / 2.0

    exact = prepare_exact_host_row(ids, values, probs, top_p=top_p)
    candidate = prepare_float32_row(ids, values, probs, top_p=top_p)

    np.testing.assert_array_equal(exact.token_ids, [11, 22, 33])
    np.testing.assert_array_equal(candidate.token_ids, [11, 22])


def test_cli_reports_both_uniform_variants_and_boundary_metrics(tmp_path: Path) -> None:
    exact, candidate = _boundary_drift_fixture()
    lower, upper = sorted((float(exact.cdf[0]), float(candidate.cdf[0])))
    inside = lower + (upper - lower) / 2.0
    fixture = tmp_path / "captured_rows.npz"
    np.savez(
        fixture,
        candidate_ids=np.array([[11, 22, 33], [11, 22, 33]], dtype=np.int64),
        candidate_values=np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype=np.float32),
        candidate_probs=np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]], dtype=np.float32),
        uniforms=np.array([inside, 0.05], dtype=np.float64),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/pr391_float32_choice_drift.py",
            str(fixture),
            "--top-p",
            "1.0",
            "--max-mismatch-indices",
            "1",
            "--benchmark-cpu",
            "--timing-warmups",
            "1",
            "--timing-repeats",
            "2",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["schema_version"] == 2
    assert report["rows"] == 2
    assert report["top_p"] == 1.0
    assert set(report["actual_mismatches"]) == {
        "exact_uniform",
        "float32_uniform",
    }
    assert report["actual_mismatches"]["exact_uniform"] == 1
    assert report["mismatch_indices"]["exact_uniform"] == [0]
    assert len(report["mismatch_indices"]["float32_uniform"]) <= 1
    assert set(report["continuous_exact_uniform_disagreement"]) == {
        "sum",
        "mean",
        "max",
    }
    assert report["continuous_exact_uniform_disagreement"]["sum"] > 0.0
    assert report["pcg64_grid_disagreement"]["grid_size"] == str(2**53)
    assert report["pcg64_grid_disagreement"]["aggregate_denominator"] == str(2 * 2**53)
    for variant in ("exact_uniform", "float32_uniform"):
        grid = report["pcg64_grid_disagreement"][variant]
        assert set(grid) == {"count", "probability"}
        assert grid["count"].isdigit()
        assert 0.0 <= grid["probability"] <= 1.0
    assert report["max_boundary_shift"] > 0.0
    assert report["support_top_p_membership_mismatch_count"] == 0
    assert report["contract"] == {
        "candidate_ids_dtype": "int64",
        "candidate_probs_dtype": "float32",
        "candidate_values_dtype": "float32",
        "max_candidates": 20,
        "uniforms_dtype": "float64",
    }
    assert report["arithmetic"] == {
        "candidate": "float32",
        "candidate_schedule": {
            "id": "pr391_rn32_norm1_norm2_if_sum_ne_one_cdf_midpoint_v1",
            "first_normalization": "always_divide_by_rank_order_rn32_sum",
            "second_normalization": "divide_iff_rn32_sum_bits_not_one",
            "final_boundary_reference": "rn32_cumsum_then_rn32_divide",
            "selection_equivalence": "future_midpoint_predicate",
            "raw_q_collapse": False,
        },
        "exact_host": "float64",
        "uniform_variants": ["float32_uniform", "exact_uniform"],
    }
    accounting = report["division_accounting"]
    assert accounting["literal_host"]["passes_per_row"] == [3, 3]
    assert accounting["literal_host"]["division_count_per_row"] == [9, 9]
    assert accounting["reduced_float32"]["passes_per_row"] == [2, 2]
    assert accounting["reduced_float32"]["division_count_per_row"] == [6, 6]
    assert accounting["reduced_float32"]["second_normalization_skip_count"] == 2
    assert accounting["reduced_float32"]["second_normalization_skip_rate"] == 1.0
    timing = report["diagnostic_cpu_timing"]
    assert timing["label"] == "diagnostic_cpu_python_numpy_not_gpu_or_tps"
    assert timing["warmups"] == 1
    assert timing["repeats"] == 2
    assert timing["rows_per_repeat"] == 2
    for schedule in ("literal_host", "reduced_float32"):
        result = timing[schedule]
        assert len(result["samples_ns"]) == 2
        assert all(sample > 0 for sample in result["samples_ns"])
        assert result["mean_ns"] > 0.0
        assert result["median_ns"] > 0.0
        assert result["mean_ns_per_row"] > 0.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"candidate_ids": np.array([[1, 2]], dtype=np.int32)}, "int64"),
        ({"candidate_values": np.array([[1.0, 0.0]], dtype=np.float64)}, "float32"),
        ({"candidate_probs": np.array([[0.5, np.nan]], dtype=np.float32)}, "finite"),
        ({"candidate_probs": np.zeros((1, 2), dtype=np.float32)}, "positive mass"),
        ({"uniforms": np.array([1.0], dtype=np.float64)}, r"\[0, 1\)"),
        ({"uniforms": np.array([0.1, 0.2], dtype=np.float64)}, "shape"),
    ],
)
def test_array_contract_validation_fails_clearly(override, message) -> None:
    from scripts.pr391_float32_choice_drift import analyze_arrays

    arrays = {
        "candidate_ids": np.array([[1, 2]], dtype=np.int64),
        "candidate_values": np.array([[1.0, 0.0]], dtype=np.float32),
        "candidate_probs": np.array([[0.6, 0.4]], dtype=np.float32),
        "uniforms": np.array([0.25], dtype=np.float64),
    }
    arrays.update(override)

    with pytest.raises(ValueError, match=message):
        analyze_arrays(**arrays)


def test_array_contract_rejects_more_than_twenty_candidates() -> None:
    from scripts.pr391_float32_choice_drift import analyze_arrays

    with pytest.raises(ValueError, match="at most 20"):
        analyze_arrays(
            candidate_ids=np.arange(21, dtype=np.int64)[None, :],
            candidate_values=np.arange(21, dtype=np.float32)[None, :],
            candidate_probs=np.ones((1, 21), dtype=np.float32),
            uniforms=np.array([0.25], dtype=np.float64),
        )
