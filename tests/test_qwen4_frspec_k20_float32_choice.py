from __future__ import annotations

from fractions import Fraction
import importlib
import sys
import types

import numpy as np
import pytest


choice = importlib.import_module("mtplx.kernels.qwen4_frspec_k20_float32_choice")


def _grid_uniform(integer: int) -> np.float64:
    return np.ldexp(np.float64(integer), -53)


def _expected_midpoint(uniform: np.float64) -> tuple[int, int, int]:
    integer = int(np.ldexp(uniform, 53))
    exact = Fraction(integer, 1 << 53)
    rounded = np.float32(uniform)
    rounded_fraction = Fraction(*float(rounded).as_integer_ratio())
    if rounded_fraction > exact:
        upper = rounded
        lower = np.nextafter(rounded, np.float32(-np.inf), dtype=np.float32)
    else:
        lower = rounded
        upper = np.nextafter(rounded, np.float32(np.inf), dtype=np.float32)
    midpoint = (
        Fraction(*float(lower).as_integer_ratio())
        + Fraction(*float(upper).as_integer_ratio())
    ) / 2
    denominator_power = midpoint.denominator.bit_length() - 1
    assert midpoint.denominator == 1 << denominator_power
    return midpoint.numerator, -denominator_power, int(upper.view(np.uint32) & 1 == 0)


def _fixture_rows(rows: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.tile(np.arange(100, 120, dtype=np.uint32), (rows, 1))
    values = np.tile(np.linspace(2.0, -2.0, 20, dtype=np.float32), (rows, 1))
    probs = np.tile(np.linspace(0.12, 0.001, 20, dtype=np.float32), (rows, 1))
    return ids, values, probs


def test_descriptor_layout_preserves_grid_integer_and_exact_midpoint() -> None:
    integers = np.array([0, 1, (1 << 52), (1 << 53) - 1], dtype=np.uint64)
    uniforms = np.ldexp(integers.astype(np.float64), -53)

    descriptors = choice.build_pcg64_midpoint_descriptors(uniforms)

    assert descriptors.dtype == np.dtype(np.uint32)
    assert descriptors.shape == (4, 5)
    for row, (integer, uniform) in enumerate(zip(integers, uniforms, strict=True)):
        observed_integer = (int(descriptors[row, 0]) << 32) | int(descriptors[row, 1])
        significand, exponent, upper_even = _expected_midpoint(uniform)
        assert observed_integer == int(integer)
        assert int(descriptors[row, 2]) == significand
        assert descriptors[row, 3].view(np.int32) == exponent
        assert int(descriptors[row, 4]) == upper_even


@pytest.mark.parametrize(
    "uniforms, message",
    [
        (np.array([0.5], dtype=np.float32), "dtype float64"),
        (np.array([[0.5]], dtype=np.float64), "one-dimensional"),
        (np.array([-0.0], dtype=np.float64), "non-negative PCG64"),
        (np.array([1.0], dtype=np.float64), r"in \[0, 1\)"),
        (np.array([np.nan], dtype=np.float64), "finite"),
        (np.array([np.nextafter(0.0, 1.0)], dtype=np.float64), "53-bit grid"),
    ],
)
def test_descriptor_builder_rejects_non_pcg64_inputs(
    uniforms: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        choice.build_pcg64_midpoint_descriptors(uniforms)


def test_descriptor_validator_fails_closed_on_corruption() -> None:
    descriptor = choice.build_pcg64_midpoint_descriptors(
        np.array([_grid_uniform(123456789)], dtype=np.float64)
    )
    descriptor[0, 2] ^= np.uint32(1)

    with pytest.raises(ValueError, match="inconsistent midpoint descriptor"):
        choice.validate_pcg64_midpoint_descriptors(descriptor)


def test_descriptor_builder_and_binder_require_exact_numpy_244(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(choice.np, "__version__", "2.4.3")

    with pytest.raises(RuntimeError, match="requires exact NumPy 2.4.4"):
        choice.build_pcg64_midpoint_descriptors(np.array([0.5], dtype=np.float64))
    with pytest.raises(RuntimeError, match="requires exact NumPy 2.4.4"):
        choice.bind_qwen4_frspec_k20_float32_choice()


def test_reference_orders_value_desc_id_asc_and_returns_raw_inputs() -> None:
    ids, values, probs = _fixture_rows()
    ids[0, :4] = np.array([9, 3, 7, 1], dtype=np.uint32)
    values[0, :4] = np.float32(10.0)
    probs[0] = np.float32(0.0)
    probs[0, :4] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    raw = tuple(array.copy() for array in (ids, values, probs))
    descriptor = choice.build_pcg64_midpoint_descriptors(
        np.array([_grid_uniform(0)], dtype=np.float64)
    )

    selected, raw_ids, raw_values, raw_probs = (
        choice.reference_qwen4_frspec_k20_float32_choice(
            ids, values, probs, descriptor, top_p=1.0
        )
    )

    # Equal-valued candidates rank by token ID, then retained support is also
    # canonical ascending ID order. At u=0 the smallest ID wins.
    assert selected.tolist() == [1]
    for observed, expected in zip((raw_ids, raw_values, raw_probs), raw, strict=True):
        assert observed.dtype == expected.dtype
        assert observed.tobytes() == expected.tobytes()


def test_cumulative_before_top_p_keeps_crossing_token_and_drops_next() -> None:
    ids, values, probs = _fixture_rows()
    ids[0] = np.arange(100, 120, dtype=np.uint32)
    ids[0, :3] = np.array([1, 2, 3], dtype=np.uint32)
    values[0] = np.linspace(0.0, -2.0, 20, dtype=np.float32)
    values[0, :3] = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs[0] = np.float32(0.0)
    probs[0, :3] = np.array([0.60, 0.40, 0.25], dtype=np.float32)
    descriptor = choice.build_pcg64_midpoint_descriptors(
        np.array([_grid_uniform((1 << 53) - 1)], dtype=np.float64)
    )

    selected, *_ = choice.reference_qwen4_frspec_k20_float32_choice(
        ids, values, probs, descriptor, top_p=0.95
    )

    # The 0.40 row crosses .95 but is retained because its cumulative-before
    # is .60. The following .25 row sees cumulative-before 1.0 and is removed.
    assert selected.tolist() == [2]


def test_top_p_one_bypasses_nucleus_and_keeps_all_positive_tokens() -> None:
    ids, values, probs = _fixture_rows()
    ids[0, :3] = np.array([7, 8, 9], dtype=np.uint32)
    values[0, :3] = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs[0] = np.float32(0.0)
    probs[0, :3] = np.array([0.6, 0.4000001, 1e-8], dtype=np.float32)
    descriptor = choice.build_pcg64_midpoint_descriptors(
        np.array([_grid_uniform((1 << 53) - 1)], dtype=np.float64)
    )

    selected, *_ = choice.reference_qwen4_frspec_k20_float32_choice(
        ids, values, probs, descriptor, top_p=1.0
    )
    ordered_ids, _ = choice._prepare_reference_row(
        ids[0], values[0], probs[0], np.float32(1.0)
    )

    assert ordered_ids.tolist() == [7, 8, 9]
    # The final 1e-8 addition rounds away in RN32, so side-right selects the
    # preceding token at the largest PCG64 grid value despite retained ID 9.
    assert selected.tolist() == [8]


def test_reduced_midpoint_selection_matches_literal_rn32_division() -> None:
    rng = np.random.default_rng(20260830)
    for _ in range(300):
        ids = np.stack([rng.permutation(20).astype(np.uint32)])
        values = rng.normal(size=(1, 20)).astype(np.float32)
        probs = rng.random(size=(1, 20), dtype=np.float32)
        integer = int(rng.integers(0, 1 << 53, dtype=np.uint64))
        uniform = _grid_uniform(integer)
        descriptor = choice.build_pcg64_midpoint_descriptors(
            np.array([uniform], dtype=np.float64)
        )

        selected, *_ = choice.reference_qwen4_frspec_k20_float32_choice(
            ids, values, probs, descriptor, top_p=0.95
        )
        literal = choice.reference_literal_divided_cdf_token(
            ids[0], values[0], probs[0], uniform, top_p=0.95
        )
        assert int(selected[0]) == literal


def test_reduced_schedule_uses_numpy_244_pairwise_sum_order_at_k20() -> None:
    from scripts.pr391_float32_choice_drift import prepare_reduced_float32_row

    rng = np.random.default_rng(73)
    ids = rng.permutation(20).astype(np.uint32)
    values = rng.normal(size=20).astype(np.float32)
    probs = rng.random(20, dtype=np.float32)
    expected = prepare_reduced_float32_row(
        ids.astype(np.int64), values, probs, top_p=1.0
    )

    ordered_ids, raw_cdf = choice._prepare_reference_row(
        ids, values, probs, np.float32(1.0)
    )

    np.testing.assert_array_equal(ordered_ids.astype(np.int64), expected.token_ids)
    np.testing.assert_array_equal(
        raw_cdf.view(np.uint32), expected.raw_cdf.view(np.uint32)
    )


def test_reference_validates_fixed_k20_contract() -> None:
    ids, values, probs = _fixture_rows()
    descriptor = choice.build_pcg64_midpoint_descriptors(
        np.array([0.5], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="candidate_ids must be uint32"):
        choice.reference_qwen4_frspec_k20_float32_choice(
            ids.astype(np.int64), values, probs, descriptor
        )
    with pytest.raises(ValueError, match=r"shape \[rows, 20\]"):
        choice.reference_qwen4_frspec_k20_float32_choice(
            ids[:, :-1], values[:, :-1], probs[:, :-1], descriptor
        )
    broken = probs.copy()
    broken[0, 0] = np.float32(-1.0)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        choice.reference_qwen4_frspec_k20_float32_choice(
            ids, values, broken, descriptor
        )


class _FakeKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> tuple[str, str, str, str]:
        self.calls.append(kwargs)
        return ("selected", "raw_ids", "raw_values", "raw_probs")


def test_binder_builds_batched_metal_kernel_and_hot_call_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, object] = {}
    fake_kernel = _FakeKernel()

    def metal_kernel(**kwargs: object) -> _FakeKernel:
        built.update(kwargs)
        return fake_kernel

    fake_core = types.ModuleType("mlx.core")
    fake_core.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    fake_core.uint32 = "uint32"
    fake_core.float32 = "float32"
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    apply = choice.bind_qwen4_frspec_k20_float32_choice(top_p=0.95)
    ids, values, probs = _fixture_rows(rows=7)
    descriptors = choice.build_pcg64_midpoint_descriptors(
        np.linspace(0.0, 0.75, 7, dtype=np.float64)
    )

    assert apply(ids, values, probs, descriptors) == (
        "selected",
        "raw_ids",
        "raw_values",
        "raw_probs",
    )
    assert built["input_names"] == [
        "candidate_ids",
        "candidate_values",
        "candidate_probs",
        "uniform_descriptors",
    ]
    assert built["output_names"] == [
        "selected_tokens",
        "raw_candidate_ids",
        "raw_candidate_values",
        "raw_candidate_probs",
    ]
    assert fake_kernel.calls == [
        {
            "inputs": [ids, values, probs, descriptors],
            "grid": (7 * 32, 1, 1),
            "threadgroup": (32, 1, 1),
            "output_shapes": [(7,), (7, 20), (7, 20), (7, 20)],
            "output_dtypes": ["uint32", "uint32", "float32", "float32"],
        }
    ]


def test_metal_source_has_required_fixed_schedule_and_no_forbidden_rng_or_division() -> (
    None
):
    source = choice.METAL_HEADER + choice.METAL_SOURCE

    assert "K = 20" in source
    assert "sorted_values[j - 1u] == value" in source
    assert "sorted_ids[j - 1u] > token" in source
    assert "cumulative_before < top_p" in source
    assert "numpy_pairwise_sum" in source
    assert "threadgroup_position_in_grid.x" in source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in source
    assert "lane < retained_count" in source
    assert "ratio_gt_midpoint" in source
    assert "raw_candidate_ids[base + lane] = candidate_ids[base + lane]" in source
    assert "raw_candidate_values[base + lane] = candidate_values[base + lane]" in source
    assert "raw_candidate_probs[base + lane] = candidate_probs[base + lane]" in source
    assert "mx.random" not in source
    assert "uniform_float" not in source
    assert "/ final_cdf" not in source
    assert source.count("{") == source.count("}")


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_binder_rejects_invalid_top_p_before_importing_mlx(top_p: float) -> None:
    with pytest.raises(ValueError, match=r"top_p must be finite and in \(0, 1\]"):
        choice.bind_qwen4_frspec_k20_float32_choice(top_p=top_p)


def test_cpu_selfcheck_covers_descriptor_and_selector_contract() -> None:
    receipt = choice.selfcheck_qwen4_frspec_k20_float32_choice()

    assert receipt == {
        "schedule_id": choice.SCHEDULE_ID,
        "k": 20,
        "descriptor_words": 5,
        "cases": choice.SELFCHECK_CASES,
    }
