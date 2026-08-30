from __future__ import annotations

import math
import struct
from fractions import Fraction

import numpy as np
import pytest


_PCG64_GRID_DENOMINATOR = 1 << 53
_MAX_PCG64_GRID_INTEGER = _PCG64_GRID_DENOMINATOR - 1


def _float64_bits(value: float | np.float64) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(np.float64(value))))[0]


def _float64_from_bits(bits: int) -> np.float64:
    return np.float64(struct.unpack(">d", struct.pack(">Q", bits))[0])


def _pcg64_uniform(grid_integer: int) -> np.float64:
    assert 0 <= grid_integer <= _MAX_PCG64_GRID_INTEGER
    return np.ldexp(np.float64(grid_integer), -53)


def _upper_endpoint_is_even(upper: np.float64) -> bool:
    return (_float64_bits(upper) & 1) == 0


def _rounded_ratio_gt_threshold(
    numerator: np.float64,
    denominator: np.float64,
    lower: np.float64,
) -> bool:
    """Return whether RN64(numerator / denominator) is above ``lower``.

    The comparison is division-free: it compares the exact ratio with the
    midpoint between adjacent binary64 values and resolves a midpoint tie from
    the least-significant bit of the two endpoints.
    """
    assert math.isfinite(numerator) and numerator > 0
    assert math.isfinite(denominator) and denominator > 0
    assert math.isfinite(lower) and lower >= 0

    upper = np.nextafter(lower, np.float64(math.inf), dtype=np.float64)
    assert math.isfinite(upper)
    numerator_fraction = Fraction.from_float(float(numerator))
    denominator_fraction = Fraction.from_float(float(denominator))
    lower_fraction = Fraction.from_float(float(lower))
    upper_fraction = Fraction.from_float(float(upper))

    twice_numerator = 2 * numerator_fraction
    twice_midpoint_times_denominator = (
        lower_fraction + upper_fraction
    ) * denominator_fraction
    if twice_numerator != twice_midpoint_times_denominator:
        return twice_numerator > twice_midpoint_times_denominator
    return _upper_endpoint_is_even(upper)


def _rounded_ratio_gt_uniform(
    numerator: np.float64,
    denominator: np.float64,
    grid_integer: int,
) -> bool:
    return _rounded_ratio_gt_threshold(
        numerator,
        denominator,
        _pcg64_uniform(grid_integer),
    )


def _numpy_ratio_gt_uniform(
    numerator: np.float64,
    denominator: np.float64,
    grid_integer: int,
) -> bool:
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        rounded = np.divide(
            np.float64(numerator),
            np.float64(denominator),
            dtype=np.float64,
        )
    return bool(rounded > _pcg64_uniform(grid_integer))


def _raw_ratio_gt_uniform(
    numerator: np.float64,
    denominator: np.float64,
    grid_integer: int,
) -> bool:
    numerator_fraction = Fraction.from_float(float(numerator))
    denominator_fraction = Fraction.from_float(float(denominator))
    uniform_fraction = Fraction(grid_integer, _PCG64_GRID_DENOMINATOR)
    return numerator_fraction > uniform_fraction * denominator_fraction


def test_raw_ratio_cross_product_is_not_the_rounded_division_predicate() -> None:
    numerator = np.float64(float.fromhex("0x1.3e2696dbb9f90p+52"))
    denominator = np.float64(float.fromhex("0x1.7125232522932p+52"))
    grid_integer = 7_762_929_414_711_601
    uniform = _pcg64_uniform(grid_integer)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        rounded_ratio = np.divide(numerator, denominator, dtype=np.float64)

    assert uniform.hex() == "0x1.b9457da2e2931p-1"
    assert rounded_ratio.hex() == "0x1.b9457da2e2931p-1"
    assert _float64_bits(rounded_ratio) == _float64_bits(uniform)
    assert _raw_ratio_gt_uniform(numerator, denominator, grid_integer)
    assert not _rounded_ratio_gt_uniform(
        numerator,
        denominator,
        grid_integer,
    )
    assert not _numpy_ratio_gt_uniform(numerator, denominator, grid_integer)


@pytest.mark.parametrize(
    "grid_integer",
    [1, 2, (1 << 51) - 1, 1 << 51, 1 << 52, _MAX_PCG64_GRID_INTEGER],
)
def test_strict_side_right_keeps_a_ratio_rounded_equal_to_uniform(
    grid_integer: int,
) -> None:
    uniform = _pcg64_uniform(grid_integer)

    assert _rounded_ratio_gt_uniform(uniform, np.float64(1.0), grid_integer) is False
    assert _numpy_ratio_gt_uniform(uniform, np.float64(1.0), grid_integer) is False


def test_zero_uniform_obeys_the_underflow_midpoint() -> None:
    minimum_subnormal = _float64_from_bits(1)

    assert not _rounded_ratio_gt_uniform(
        minimum_subnormal,
        np.float64(2.0),
        0,
    )
    assert not _numpy_ratio_gt_uniform(
        minimum_subnormal,
        np.float64(2.0),
        0,
    )
    assert _rounded_ratio_gt_uniform(
        minimum_subnormal,
        np.float64(1.0),
        0,
    )
    assert _numpy_ratio_gt_uniform(
        minimum_subnormal,
        np.float64(1.0),
        0,
    )


def test_maximum_uniform_handles_equality_and_next_value() -> None:
    maximum_uniform = _pcg64_uniform(_MAX_PCG64_GRID_INTEGER)

    assert maximum_uniform.hex() == "0x1.fffffffffffffp-1"
    assert not _rounded_ratio_gt_uniform(
        maximum_uniform,
        np.float64(1.0),
        _MAX_PCG64_GRID_INTEGER,
    )
    assert _rounded_ratio_gt_uniform(
        np.float64(1.0),
        np.float64(1.0),
        _MAX_PCG64_GRID_INTEGER,
    )


def test_midpoint_tie_uses_the_even_endpoint() -> None:
    minimum_subnormal = _float64_from_bits(1)
    second_subnormal = _float64_from_bits(2)
    third_subnormal = _float64_from_bits(3)

    assert not _upper_endpoint_is_even(minimum_subnormal)
    assert _upper_endpoint_is_even(second_subnormal)

    with np.errstate(under="ignore"):
        lower_even_result = np.divide(
            minimum_subnormal,
            np.float64(2.0),
            dtype=np.float64,
        )
        upper_even_result = np.divide(
            third_subnormal,
            np.float64(2.0),
            dtype=np.float64,
        )

    assert _float64_bits(lower_even_result) == 0
    assert _float64_bits(upper_even_result) == 2
    assert not _rounded_ratio_gt_threshold(
        minimum_subnormal,
        np.float64(2.0),
        np.float64(0.0),
    )
    assert _rounded_ratio_gt_threshold(
        third_subnormal,
        np.float64(2.0),
        minimum_subnormal,
    )


@pytest.mark.parametrize(
    ("numerator", "denominator", "grid_integer"),
    [
        ("0x1.0000000000000p-1022", "0x1.0000000000000p+0", 1),
        ("0x1.fffffffffffffp+1023", "0x1.0000000000000p+0", 1 << 52),
        ("0x1.0000000000000p-1022", "0x1.fffffffffffffp+1023", 0),
        ("0x1.fffffffffffffp+1023", "0x1.0000000000000p-1022", 1 << 52),
        ("0x1.0000000000001p-500", "0x1.fffffffffffffp-501", 1 << 52),
        ("0x1.23456789abcdep+400", "0x1.edcba98765432p+401", 1 << 51),
    ],
)
def test_exponent_variation_matches_numpy_float64_division(
    numerator: str,
    denominator: str,
    grid_integer: int,
) -> None:
    x = np.float64(float.fromhex(numerator))
    z = np.float64(float.fromhex(denominator))

    assert _rounded_ratio_gt_uniform(x, z, grid_integer) == _numpy_ratio_gt_uniform(
        x,
        z,
        grid_integer,
    )


def test_random_positive_finite_operands_match_numpy_float64_division() -> None:
    rng = np.random.default_rng(20260830)

    for _ in range(10_000):
        numerator_bits = int(rng.integers(1, 0x7FF0_0000_0000_0000, dtype=np.uint64))
        denominator_bits = int(rng.integers(1, 0x7FF0_0000_0000_0000, dtype=np.uint64))
        grid_integer = int(rng.integers(0, _PCG64_GRID_DENOMINATOR, dtype=np.uint64))
        numerator = _float64_from_bits(numerator_bits)
        denominator = _float64_from_bits(denominator_bits)

        assert math.isfinite(numerator) and numerator > 0
        assert math.isfinite(denominator) and denominator > 0
        assert _rounded_ratio_gt_uniform(
            numerator,
            denominator,
            grid_integer,
        ) == _numpy_ratio_gt_uniform(numerator, denominator, grid_integer)
