#!/usr/bin/env python3
"""Measure categorical choice drift from float32 draft-sampler arithmetic."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np


PCG64_GRID_SIZE = 1 << 53
REQUIRED_NUMPY_VERSION = "2.4.4"
CANDIDATE_SCHEDULE_ID = "pr391_rn32_norm1_norm2_if_sum_ne_one_cdf_midpoint_v1"
REDUCED_EXACT_SCHEDULE_ID = "pr391_rn64_norm1_norm2_if_sum_ne_one_midpoint_v1"


@dataclass(frozen=True)
class PreparedRow:
    """One retained, token-id-ordered categorical distribution."""

    token_ids: np.ndarray
    probabilities: np.ndarray
    cdf: np.ndarray


@dataclass(frozen=True)
class ReducedExactRow:
    """Exact float64 probabilities plus an unnormalized sequential CDF."""

    token_ids: np.ndarray
    probabilities: np.ndarray
    raw_cdf: np.ndarray
    second_normalization_skipped: bool


@dataclass(frozen=True)
class ReducedFloat32Row:
    """Reduced float32 probabilities plus an unnormalized sequential CDF."""

    token_ids: np.ndarray
    probabilities: np.ndarray
    raw_cdf: np.ndarray
    second_normalization_skipped: bool


def _validate_top_p(top_p: float) -> float:
    top_p = float(top_p)
    if not np.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1]")
    return top_p


def _require_numpy_version() -> None:
    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise RuntimeError(
            f"PR391 drift analysis requires exact NumPy {REQUIRED_NUMPY_VERSION}; "
            f"found {np.__version__}"
        )


def _validate_row_inputs(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids)
    values = np.asarray(candidate_values)
    probs = np.asarray(candidate_probs)
    if ids.dtype != np.dtype(np.int64):
        raise ValueError("candidate_ids must have dtype int64")
    if values.dtype != np.dtype(np.float32):
        raise ValueError("candidate_values must have dtype float32")
    if probs.dtype != np.dtype(np.float32):
        raise ValueError("candidate_probs must have dtype float32")
    if ids.ndim != 1 or values.ndim != 1 or probs.ndim != 1:
        raise ValueError("row candidate arrays must be one-dimensional")
    if ids.shape != values.shape or ids.shape != probs.shape:
        raise ValueError("row candidate array shape mismatch")
    if not 1 <= ids.size <= 20:
        raise ValueError("each row must contain between 1 and at most 20 candidates")
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError("candidate_ids must be unique non-negative token IDs")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate_values must be finite")
    if not np.all(np.isfinite(probs)):
        raise ValueError("candidate_probs must be finite")
    if np.any(probs < 0.0):
        raise ValueError("candidate_probs must be in [0, 1]")
    if np.any(probs > 1.0):
        raise ValueError("candidate_probs must be in [0, 1]")
    if not float(np.sum(probs, dtype=np.float64)) > 0.0:
        raise ValueError("each candidate_probs row must have positive mass")
    return ids, values, probs


def _rank_and_filter(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((ids, -values))
    ranked_ids = ids[order]
    ranked_probs = probs[order].astype(dtype, copy=False)
    if 0.0 < top_p < 1.0:
        cumulative_before = np.concatenate(
            (
                np.zeros(1, dtype=dtype),
                np.cumsum(ranked_probs[:-1], dtype=dtype),
            )
        )
        threshold = float(top_p) if dtype == np.dtype(np.float64) else np.float32(top_p)
        keep = (cumulative_before < threshold) & (ranked_probs > 0.0)
    else:
        # Production activates nucleus filtering only for strict 0 < top_p < 1.
        keep = ranked_probs > 0.0
    retained_ids = ranked_ids[keep]
    retained_probs = ranked_probs[keep]
    return retained_ids, retained_probs


def _normalize_and_cdf(
    token_ids: np.ndarray,
    retained_probs: np.ndarray,
    *,
    dtype: np.dtype,
    skip_unit_second_normalization: bool = False,
) -> PreparedRow:
    # Match _serial_row_distribution exactly: reduce retained probabilities in
    # value-ranked order, then reorder the support by token ID before dividing.
    total = np.sum(retained_probs, dtype=dtype)
    token_order = np.argsort(token_ids)
    token_ids = token_ids[token_order]
    retained_probs = retained_probs[token_order]
    normalized_once = (retained_probs / total).astype(dtype, copy=False)
    sanitized = np.where(
        np.isfinite(normalized_once) & (normalized_once > 0.0),
        normalized_once,
        dtype.type(0.0),
    )
    sparse_total = np.sum(sanitized, dtype=dtype)
    if skip_unit_second_normalization and sparse_total == dtype.type(1.0):
        # Reduced candidate schedule: division by rounded one is bit-identical.
        # This is not the unapproved raw-q collapse; the first normalization is
        # always performed, and non-unit sparse totals are still renormalized.
        normalized = sanitized
    else:
        normalized = (sanitized / sparse_total).astype(dtype, copy=False)
    cdf = np.cumsum(normalized, dtype=dtype)
    cdf = (cdf / cdf[-1]).astype(dtype, copy=False)
    return PreparedRow(token_ids, normalized, cdf)


def prepare_exact_host_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float = 0.95,
) -> PreparedRow:
    """Reproduce the current NumPy host path for one pre-top-p row."""

    ids, values, probs = _validate_row_inputs(
        candidate_ids, candidate_values, candidate_probs
    )
    top_p = _validate_top_p(top_p)
    retained_ids, retained_probs = _rank_and_filter(
        ids,
        values,
        probs.astype(np.float64),
        top_p=top_p,
        dtype=np.dtype(np.float64),
    )
    return _normalize_and_cdf(retained_ids, retained_probs, dtype=np.dtype(np.float64))


def prepare_reduced_exact_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float = 0.95,
) -> ReducedExactRow:
    """Prepare literal float64 probabilities without final CDF division."""

    ids, values, probs = _validate_row_inputs(
        candidate_ids, candidate_values, candidate_probs
    )
    top_p = _validate_top_p(top_p)
    retained_ids, retained_probs = _rank_and_filter(
        ids,
        values,
        probs.astype(np.float64),
        top_p=top_p,
        dtype=np.dtype(np.float64),
    )
    first_total = np.sum(retained_probs, dtype=np.float64)
    token_order = np.argsort(retained_ids)
    retained_ids = retained_ids[token_order]
    retained_probs = retained_probs[token_order]
    normalized_once = (retained_probs / first_total).astype(np.float64, copy=False)
    sanitized = np.where(
        np.isfinite(normalized_once) & (normalized_once > 0.0),
        normalized_once,
        np.float64(0.0),
    )
    sparse_total = np.sum(sanitized, dtype=np.float64)
    skip_second = bool(sparse_total.view(np.uint64) == np.float64(1.0).view(np.uint64))
    probabilities = (
        sanitized
        if skip_second
        else (sanitized / sparse_total).astype(np.float64, copy=False)
    )
    raw_cdf = np.cumsum(probabilities, dtype=np.float64)
    return ReducedExactRow(
        token_ids=retained_ids,
        probabilities=probabilities,
        raw_cdf=raw_cdf,
        second_normalization_skipped=skip_second,
    )


def prepare_float32_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float = 0.95,
) -> PreparedRow:
    """Run the proposed device selector's arithmetic entirely in float32."""

    ids, values, probs = _validate_row_inputs(
        candidate_ids, candidate_values, candidate_probs
    )
    top_p = _validate_top_p(top_p)
    retained_ids, retained_probs = _rank_and_filter(
        ids,
        values,
        probs,
        top_p=top_p,
        dtype=np.dtype(np.float32),
    )
    return _normalize_and_cdf(
        retained_ids,
        retained_probs,
        dtype=np.dtype(np.float32),
        skip_unit_second_normalization=True,
    )


def prepare_reduced_float32_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float = 0.95,
) -> ReducedFloat32Row:
    """Prepare reduced float32 probabilities without final CDF division."""

    ids, values, probs = _validate_row_inputs(
        candidate_ids, candidate_values, candidate_probs
    )
    top_p = _validate_top_p(top_p)
    retained_ids, retained_probs = _rank_and_filter(
        ids,
        values,
        probs,
        top_p=top_p,
        dtype=np.dtype(np.float32),
    )
    first_total = np.sum(retained_probs, dtype=np.float32)
    token_order = np.argsort(retained_ids)
    retained_ids = retained_ids[token_order]
    retained_probs = retained_probs[token_order]
    normalized_once = (retained_probs / first_total).astype(np.float32, copy=False)
    sanitized = np.where(
        np.isfinite(normalized_once) & (normalized_once > 0.0),
        normalized_once,
        np.float32(0.0),
    )
    sparse_total = np.sum(sanitized, dtype=np.float32)
    skip_second = bool(sparse_total.view(np.uint32) == np.float32(1.0).view(np.uint32))
    probabilities = (
        sanitized
        if skip_second
        else (sanitized / sparse_total).astype(np.float32, copy=False)
    )
    raw_cdf = np.cumsum(probabilities, dtype=np.float32)
    return ReducedFloat32Row(
        token_ids=retained_ids,
        probabilities=probabilities,
        raw_cdf=raw_cdf,
        second_normalization_skipped=skip_second,
    )


def select_token(row: PreparedRow, uniform: float, *, cast_uniform: bool) -> int:
    """Select with right-sided search, safely mapping rounded 1.0 to the tail."""

    if not np.isfinite(uniform) or not 0.0 <= float(uniform) < 1.0:
        raise ValueError("uniform must be finite and in [0, 1)")
    if cast_uniform:
        search_cdf = row.cdf.astype(np.float32, copy=False)
        search_uniform = np.float32(uniform)
    else:
        # Explicit promotion isolates CDF drift from narrowing the 53-bit tape.
        search_cdf = row.cdf.astype(np.float64)
        search_uniform = np.float64(uniform)
    index = int(np.searchsorted(search_cdf, search_uniform, side="right"))
    index = min(index, int(row.token_ids.size) - 1)
    return int(row.token_ids[index])


def _pcg64_grid_integer(uniform: float) -> int:
    uniform = float(uniform)
    if not np.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be finite and in [0, 1)")
    scaled = uniform * PCG64_GRID_SIZE
    integer = int(scaled)
    if float(integer / PCG64_GRID_SIZE) != uniform:
        raise ValueError("uniform must lie on the exact PCG64 53-bit grid")
    return integer


def _fraction_from_float(value: float) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


def _rounded_ratio_greater_than_grid(
    numerator: float,
    denominator: float,
    integer: int,
) -> bool:
    """Test RN64(numerator/denominator) > integer/2**53 sans division."""

    ratio = _fraction_from_float(numerator) / _fraction_from_float(denominator)
    return _rounded_fraction_greater_than_grid(ratio, integer)


def _rounded_fraction_greater_than_grid(ratio: Fraction, integer: int) -> bool:
    """Apply the RN-even midpoint predicate to an exact rational value."""

    uniform = np.float64(integer / PCG64_GRID_SIZE)
    successor = np.nextafter(uniform, np.float64(np.inf))
    midpoint = (_fraction_from_float(uniform) + _fraction_from_float(successor)) / 2
    if ratio != midpoint:
        return ratio > midpoint
    # At the exact midpoint, IEEE round-to-nearest-even chooses the endpoint
    # whose significand has an even low bit. Positive adjacent float bit
    # patterns differ by one, so successor's low bit identifies that winner.
    successor_bits = int(successor.view(np.uint64))
    return successor_bits & 1 == 0


def select_reduced_exact_token(row: ReducedExactRow, uniform: float) -> int:
    """Select exactly like final RN64 CDF division plus side-right search."""

    integer = _pcg64_grid_integer(uniform)
    final = float(row.raw_cdf[-1])
    for index, boundary in enumerate(row.raw_cdf[:-1]):
        if _rounded_ratio_greater_than_grid(float(boundary), final, integer):
            return int(row.token_ids[index])
    return int(row.token_ids[-1])


def _rounded_fraction_float32_greater_than_grid(
    ratio: Fraction,
    integer: int,
    *,
    cast_uniform: bool,
) -> bool:
    """Test RN32(ratio) against an exact or float32-cast PCG64 uniform."""

    uniform_fraction = Fraction(integer, PCG64_GRID_SIZE)
    rounded_uniform = np.float32(integer / PCG64_GRID_SIZE)
    if cast_uniform:
        predecessor = rounded_uniform
        threshold = np.nextafter(rounded_uniform, np.float32(np.inf), dtype=np.float32)
    elif _fraction_from_float(rounded_uniform) > uniform_fraction:
        threshold = rounded_uniform
        predecessor = np.nextafter(
            rounded_uniform, np.float32(-np.inf), dtype=np.float32
        )
    else:
        predecessor = rounded_uniform
        threshold = np.nextafter(rounded_uniform, np.float32(np.inf), dtype=np.float32)
    midpoint = (_fraction_from_float(predecessor) + _fraction_from_float(threshold)) / 2
    if ratio != midpoint:
        return ratio > midpoint
    return int(threshold.view(np.uint32)) & 1 == 0


def _rounded_ratio_float32_greater_than_grid(
    numerator: float,
    denominator: float,
    integer: int,
    *,
    cast_uniform: bool,
) -> bool:
    ratio = _fraction_from_float(numerator) / _fraction_from_float(denominator)
    return _rounded_fraction_float32_greater_than_grid(
        ratio, integer, cast_uniform=cast_uniform
    )


def select_reduced_float32_token(
    row: ReducedFloat32Row,
    uniform: float,
    *,
    cast_uniform: bool,
) -> int:
    """Select like RN32 final division without evaluating that division."""

    integer = _pcg64_grid_integer(uniform)
    final = float(row.raw_cdf[-1])
    for index, boundary in enumerate(row.raw_cdf[:-1]):
        if _rounded_ratio_float32_greater_than_grid(
            float(boundary),
            final,
            integer,
            cast_uniform=cast_uniform,
        ):
            return int(row.token_ids[index])
    return int(row.token_ids[-1])


def _cdf_at_token_boundaries(row: PreparedRow, tokens: np.ndarray) -> np.ndarray:
    boundaries = np.empty(tokens.size, dtype=np.float64)
    for index, token in enumerate(tokens):
        row_index = int(np.searchsorted(row.token_ids, token, side="right")) - 1
        boundaries[index] = 0.0 if row_index < 0 else float(row.cdf[row_index])
    return boundaries


def row_disagreement(
    exact: PreparedRow,
    candidate: PreparedRow,
) -> tuple[float, float]:
    """Return inverse-CDF disagreement measure and maximum boundary shift."""

    exact_cdf = exact.cdf.astype(np.float64)
    candidate_cdf = candidate.cdf.astype(np.float64)
    breakpoints = np.unique(
        np.clip(
            np.concatenate(
                (
                    np.array([0.0, 1.0]),
                    exact_cdf[:-1],
                    candidate_cdf[:-1],
                )
            ),
            0.0,
            1.0,
        )
    )
    disagreement = 0.0
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        if upper <= lower:
            continue
        exact_token = select_token(exact, float(lower), cast_uniform=False)
        candidate_token = select_token(candidate, float(lower), cast_uniform=False)
        if exact_token != candidate_token:
            disagreement += float(upper - lower)

    union_tokens = np.union1d(exact.token_ids, candidate.token_ids)
    exact_boundaries = _cdf_at_token_boundaries(exact, union_tokens)
    candidate_boundaries = _cdf_at_token_boundaries(candidate, union_tokens)
    max_shift = float(np.max(np.abs(exact_boundaries - candidate_boundaries)))
    return disagreement, max_shift


def _exact_uniform_grid_transition(boundary: float) -> int:
    """First R whose exact R/2**53 is at or above one CDF boundary."""

    numerator, denominator = float(boundary).as_integer_ratio()
    scaled_numerator = numerator * PCG64_GRID_SIZE
    transition = (scaled_numerator + denominator - 1) // denominator
    return min(max(int(transition), 0), PCG64_GRID_SIZE)


def _float32_uniform_grid_transition(boundary: float) -> int:
    """First R whose float32(R/2**53) reaches one float32 boundary."""

    target = np.float32(boundary)
    lower = 0
    upper = PCG64_GRID_SIZE
    while lower < upper:
        midpoint = (lower + upper) // 2
        # midpoint / 2**53 is exact in binary64. The one narrowing conversion
        # is therefore exactly the proposed float32 tape-consumption behavior.
        observed = np.float32(midpoint / PCG64_GRID_SIZE)
        if observed >= target:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def _grid_transitions(row: PreparedRow, *, cast_uniform: bool) -> tuple[int, ...]:
    transition = (
        _float32_uniform_grid_transition
        if cast_uniform
        else _exact_uniform_grid_transition
    )
    return tuple(transition(float(boundary)) for boundary in row.cdf[:-1])


def _reduced_exact_grid_transition(boundary: float, final: float) -> int:
    """First PCG64 integer not kept below one reduced-exact boundary."""

    lower = 0
    upper = PCG64_GRID_SIZE
    while lower < upper:
        midpoint = (lower + upper) // 2
        if _rounded_ratio_greater_than_grid(boundary, final, midpoint):
            lower = midpoint + 1
        else:
            upper = midpoint
    return lower


def _reduced_exact_grid_transitions(row: ReducedExactRow) -> tuple[int, ...]:
    final = float(row.raw_cdf[-1])
    return tuple(
        _reduced_exact_grid_transition(float(boundary), final)
        for boundary in row.raw_cdf[:-1]
    )


def _reduced_float32_grid_transition(
    boundary: float,
    final: float,
    *,
    cast_uniform: bool,
) -> int:
    lower = 0
    upper = PCG64_GRID_SIZE
    while lower < upper:
        midpoint = (lower + upper) // 2
        if _rounded_ratio_float32_greater_than_grid(
            boundary,
            final,
            midpoint,
            cast_uniform=cast_uniform,
        ):
            lower = midpoint + 1
        else:
            upper = midpoint
    return lower


def _reduced_float32_grid_transitions(
    row: ReducedFloat32Row,
    *,
    cast_uniform: bool,
) -> tuple[int, ...]:
    final = float(row.raw_cdf[-1])
    return tuple(
        _reduced_float32_grid_transition(
            float(boundary), final, cast_uniform=cast_uniform
        )
        for boundary in row.raw_cdf[:-1]
    )


def _grid_token(
    row: PreparedRow | ReducedExactRow | ReducedFloat32Row,
    transitions: tuple[int, ...],
    value: int,
) -> int:
    index = bisect_right(transitions, value)
    return int(row.token_ids[min(index, int(row.token_ids.size) - 1)])


def pcg64_grid_disagreement_count(
    exact: PreparedRow | ReducedExactRow,
    candidate: PreparedRow | ReducedFloat32Row,
    *,
    candidate_cast_uniform: bool,
) -> int:
    """Count exact PCG64 R values producing different selected token IDs."""

    exact_transitions = (
        _reduced_exact_grid_transitions(exact)
        if isinstance(exact, ReducedExactRow)
        else _grid_transitions(exact, cast_uniform=False)
    )
    candidate_transitions = (
        _reduced_float32_grid_transitions(
            candidate, cast_uniform=candidate_cast_uniform
        )
        if isinstance(candidate, ReducedFloat32Row)
        else _grid_transitions(candidate, cast_uniform=candidate_cast_uniform)
    )
    endpoints = sorted(
        {
            0,
            PCG64_GRID_SIZE,
            *exact_transitions,
            *candidate_transitions,
        }
    )
    disagreement = 0
    for lower, upper in zip(endpoints[:-1], endpoints[1:]):
        if upper <= lower:
            continue
        exact_token = _grid_token(exact, exact_transitions, lower)
        candidate_token = _grid_token(candidate, candidate_transitions, lower)
        if exact_token != candidate_token:
            disagreement += upper - lower
    return disagreement


def _cross_check_reduced_exact(
    literal: PreparedRow,
    reduced: ReducedExactRow,
    *,
    row_index: int,
) -> None:
    """Fail closed unless reduced preparation and all grid transitions agree."""

    if not np.array_equal(literal.token_ids, reduced.token_ids):
        raise RuntimeError(
            f"reduced exact token support diverged from literal host at row {row_index}"
        )
    if not np.array_equal(
        literal.probabilities.view(np.uint64),
        reduced.probabilities.view(np.uint64),
    ):
        raise RuntimeError(
            f"reduced exact probability bits diverged from literal host at row {row_index}"
        )
    literal_raw_cdf = np.cumsum(literal.probabilities, dtype=np.float64)
    if not np.array_equal(
        literal_raw_cdf.view(np.uint64), reduced.raw_cdf.view(np.uint64)
    ):
        raise RuntimeError(
            f"reduced exact raw CDF bits diverged from literal host at row {row_index}"
        )
    literal_transitions = _grid_transitions(literal, cast_uniform=False)
    reduced_transitions = _reduced_exact_grid_transitions(reduced)
    if literal_transitions != reduced_transitions:
        raise RuntimeError(
            f"reduced exact PCG64 transitions diverged from literal host at row {row_index}"
        )


def _cross_check_reduced_float32(
    reference: PreparedRow,
    reduced: ReducedFloat32Row,
    *,
    row_index: int,
) -> None:
    """Fail closed unless reduced float32 and divided-CDF reference agree."""

    if not np.array_equal(reference.token_ids, reduced.token_ids):
        raise RuntimeError(
            f"reduced float32 token support diverged from divided reference at row {row_index}"
        )
    if not np.array_equal(
        reference.probabilities.view(np.uint32),
        reduced.probabilities.view(np.uint32),
    ):
        raise RuntimeError(
            f"reduced float32 probability bits diverged from divided reference at row {row_index}"
        )
    reference_raw_cdf = np.cumsum(reference.probabilities, dtype=np.float32)
    if not np.array_equal(
        reference_raw_cdf.view(np.uint32), reduced.raw_cdf.view(np.uint32)
    ):
        raise RuntimeError(
            f"reduced float32 raw CDF bits diverged from divided reference at row {row_index}"
        )
    for cast_uniform in (False, True):
        reference_transitions = _grid_transitions(reference, cast_uniform=cast_uniform)
        reduced_transitions = _reduced_float32_grid_transitions(
            reduced, cast_uniform=cast_uniform
        )
        if reference_transitions != reduced_transitions:
            variant = "float32" if cast_uniform else "exact"
            raise RuntimeError(
                f"reduced float32 {variant}-uniform PCG64 transitions diverged "
                f"from divided reference at row {row_index}"
            )


def _float32_division_accounting(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float,
) -> tuple[int, int, bool]:
    """Return support size, division passes, and second-pass skip decision."""

    retained_ids, retained_probs = _rank_and_filter(
        candidate_ids,
        candidate_values,
        candidate_probs,
        top_p=top_p,
        dtype=np.dtype(np.float32),
    )
    first_total = np.sum(retained_probs, dtype=np.float32)
    token_order = np.argsort(retained_ids)
    first_normalized = (retained_probs[token_order] / first_total).astype(
        np.float32, copy=False
    )
    sanitized = np.where(
        np.isfinite(first_normalized) & (first_normalized > 0.0),
        first_normalized,
        np.float32(0.0),
    )
    sparse_total = np.sum(sanitized, dtype=np.float32)
    skip_second = bool(sparse_total.view(np.uint32) == np.float32(1.0).view(np.uint32))
    # This counts probability-array divisions only. The RN32 final-boundary
    # reference is recorded separately from midpoint-equivalent selection.
    passes = 1 + int(not skip_second)
    return int(retained_ids.size), passes, skip_second


def benchmark_prepared_rows(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    """Interleaved CPU timing of all row preparers; never a GPU/TPS claim."""

    if isinstance(warmups, bool) or int(warmups) < 0:
        raise ValueError("timing warmups must be a non-negative integer")
    if isinstance(repeats, bool) or int(repeats) < 1:
        raise ValueError("timing repeats must be a positive integer")
    warmups = int(warmups)
    repeats = int(repeats)
    rows = int(candidate_ids.shape[0])
    schedules = (
        ("literal_float64", prepare_exact_host_row),
        ("reduced_exact_float64", prepare_reduced_exact_row),
        ("reduced_float32", prepare_reduced_float32_row),
    )

    def run(schedule) -> None:
        for row_index in range(rows):
            schedule(
                candidate_ids[row_index],
                candidate_values[row_index],
                candidate_probs[row_index],
                top_p=top_p,
            )

    for _ in range(warmups):
        for _, schedule in schedules:
            run(schedule)

    samples: dict[str, list[int]] = {name: [] for name, _ in schedules}
    for repeat in range(repeats):
        ordered = schedules if repeat % 2 == 0 else tuple(reversed(schedules))
        for name, schedule in ordered:
            started = time.perf_counter_ns()
            run(schedule)
            samples[name].append(time.perf_counter_ns() - started)

    report: dict[str, object] = {
        "label": "diagnostic_cpu_python_numpy_not_gpu_or_tps",
        "scope": "prepared_row_functions_only_selection_predicates_not_timed",
        "warmups": warmups,
        "repeats": repeats,
        "rows_per_repeat": rows,
    }
    for name, _ in schedules:
        values = samples[name]
        report[name] = {
            "samples_ns": values,
            "mean_ns": float(np.mean(values)),
            "median_ns": float(np.median(values)),
            "mean_ns_per_row": float(np.mean(values)) / rows,
        }
    return report


def _validate_arrays(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids)
    values = np.asarray(candidate_values)
    probs = np.asarray(candidate_probs)
    uniforms = np.asarray(uniforms)
    if ids.dtype != np.dtype(np.int64):
        raise ValueError("candidate_ids must have dtype int64")
    if values.dtype != np.dtype(np.float32):
        raise ValueError("candidate_values must have dtype float32")
    if probs.dtype != np.dtype(np.float32):
        raise ValueError("candidate_probs must have dtype float32")
    if uniforms.dtype != np.dtype(np.float64):
        raise ValueError("uniforms must have dtype float64")
    if ids.ndim != 2 or values.ndim != 2 or probs.ndim != 2:
        raise ValueError("candidate arrays must have shape [N, K]")
    if ids.shape != values.shape or ids.shape != probs.shape:
        raise ValueError("candidate array shape mismatch")
    rows, candidates = ids.shape
    if rows < 1:
        raise ValueError("candidate arrays must contain at least one row")
    if not 1 <= candidates <= 20:
        raise ValueError("candidate rows must contain between 1 and at most 20 entries")
    if uniforms.shape != (rows,):
        raise ValueError(f"uniforms shape must be ({rows},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate_values must be finite")
    if not np.all(np.isfinite(probs)):
        raise ValueError("candidate_probs must be finite")
    if np.any(probs < 0.0):
        raise ValueError("candidate_probs must be in [0, 1]")
    if np.any(probs > 1.0):
        raise ValueError("candidate_probs must be in [0, 1]")
    row_mass = np.sum(probs, axis=1, dtype=np.float64)
    if np.any(row_mass <= 0.0):
        raise ValueError("each candidate_probs row must have positive mass")
    if np.any(ids < 0):
        raise ValueError("candidate_ids must be non-negative")
    if any(np.unique(row).size != candidates for row in ids):
        raise ValueError("candidate_ids must be unique within each row")
    if not np.all(np.isfinite(uniforms)) or np.any(
        (uniforms < 0.0) | (uniforms >= 1.0)
    ):
        raise ValueError("uniforms must be finite and in [0, 1)")
    return ids, values, probs, uniforms


def analyze_arrays(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    uniforms: np.ndarray,
    *,
    top_p: float = 0.95,
    max_mismatch_indices: int = 64,
) -> dict[str, object]:
    """Analyze captured pre-top-p rows and return a stable JSON-ready receipt."""

    _require_numpy_version()
    ids, values, probs, uniforms = _validate_arrays(
        candidate_ids, candidate_values, candidate_probs, uniforms
    )
    top_p = _validate_top_p(top_p)
    if isinstance(max_mismatch_indices, bool) or int(max_mismatch_indices) < 0:
        raise ValueError("max_mismatch_indices must be a non-negative integer")
    max_mismatch_indices = int(max_mismatch_indices)

    mismatch_indices = {"float32_uniform": [], "exact_uniform": []}
    mismatch_counts = {"float32_uniform": 0, "exact_uniform": 0}
    disagreement_measures: list[float] = []
    grid_disagreement_counts = {"float32_uniform": 0, "exact_uniform": 0}
    max_boundary_shift = 0.0
    support_mismatches = 0
    literal_normalization_passes: list[int] = []
    literal_selection_passes: list[int] = []
    literal_divisions_per_row: list[int] = []
    reduced_exact_normalization_passes: list[int] = []
    reduced_exact_selection_passes: list[int] = []
    reduced_exact_divisions_per_row: list[int] = []
    reduced_exact_second_skips = 0
    candidate_normalization_passes: list[int] = []
    candidate_selection_passes: list[int] = []
    candidate_numpy_boundary_passes: list[int] = []
    candidate_divisions_per_row: list[int] = []
    candidate_reference_divisions_per_row: list[int] = []
    candidate_second_skips = 0

    for row_index in range(ids.shape[0]):
        literal = prepare_exact_host_row(
            ids[row_index], values[row_index], probs[row_index], top_p=top_p
        )
        reduced_exact = prepare_reduced_exact_row(
            ids[row_index], values[row_index], probs[row_index], top_p=top_p
        )
        candidate_reference = prepare_float32_row(
            ids[row_index], values[row_index], probs[row_index], top_p=top_p
        )
        candidate = prepare_reduced_float32_row(
            ids[row_index], values[row_index], probs[row_index], top_p=top_p
        )
        _cross_check_reduced_exact(
            literal,
            reduced_exact,
            row_index=row_index,
        )
        _cross_check_reduced_float32(
            candidate_reference,
            candidate,
            row_index=row_index,
        )

        literal_support = int(literal.token_ids.size)
        literal_normalization_passes.append(2)
        literal_selection_passes.append(1)
        literal_divisions_per_row.append(3 * literal_support)
        reduced_support = int(reduced_exact.token_ids.size)
        reduced_passes = 1 + int(not reduced_exact.second_normalization_skipped)
        reduced_exact_normalization_passes.append(reduced_passes)
        reduced_exact_selection_passes.append(0)
        reduced_exact_divisions_per_row.append(reduced_passes * reduced_support)
        reduced_exact_second_skips += int(reduced_exact.second_normalization_skipped)
        candidate_support, candidate_passes, candidate_skipped = (
            _float32_division_accounting(
                ids[row_index],
                values[row_index],
                probs[row_index],
                top_p=top_p,
            )
        )
        candidate_normalization_passes.append(candidate_passes)
        candidate_selection_passes.append(0)
        candidate_numpy_boundary_passes.append(1)
        candidate_divisions_per_row.append(candidate_passes * candidate_support)
        candidate_reference_divisions_per_row.append(
            (candidate_passes + 1) * candidate_support
        )
        candidate_second_skips += int(candidate_skipped)
        expected = select_reduced_exact_token(reduced_exact, float(uniforms[row_index]))
        variants = {
            "float32_uniform": select_reduced_float32_token(
                candidate, float(uniforms[row_index]), cast_uniform=True
            ),
            "exact_uniform": select_reduced_float32_token(
                candidate, float(uniforms[row_index]), cast_uniform=False
            ),
        }
        for name, observed in variants.items():
            if observed != expected:
                mismatch_counts[name] += 1
                if len(mismatch_indices[name]) < max_mismatch_indices:
                    mismatch_indices[name].append(row_index)
        if not np.array_equal(reduced_exact.token_ids, candidate.token_ids):
            support_mismatches += 1
        # Continuous measure is a literal-boundary proxy only. Actual samples
        # and exact grid counts below use the reduced-exact midpoint selector.
        disagreement, boundary_shift = row_disagreement(literal, candidate_reference)
        disagreement_measures.append(disagreement)
        max_boundary_shift = max(max_boundary_shift, boundary_shift)
        grid_disagreement_counts["exact_uniform"] += pcg64_grid_disagreement_count(
            reduced_exact, candidate, candidate_cast_uniform=False
        )
        grid_disagreement_counts["float32_uniform"] += pcg64_grid_disagreement_count(
            reduced_exact, candidate, candidate_cast_uniform=True
        )

    measures = np.asarray(disagreement_measures, dtype=np.float64)
    aggregate_grid_denominator = int(ids.shape[0]) * PCG64_GRID_SIZE
    return {
        "schema_version": 3,
        "rows": int(ids.shape[0]),
        "top_p": top_p,
        "comparison_baseline": {
            "schedule_id": REDUCED_EXACT_SCHEDULE_ID,
            "selection": "reduced_exact_float64",
        },
        "reference_cross_checks": {
            "required": True,
            "literal_float64": {
                "rows_checked": int(ids.shape[0]),
                "status": "pass",
            },
            "divided_float32": {
                "rows_checked": int(ids.shape[0]),
                "status": "pass",
            },
        },
        "actual_mismatches": mismatch_counts,
        "mismatch_indices": mismatch_indices,
        "continuous_literal_boundary_proxy": {
            "sum": float(np.sum(measures)),
            "mean": float(np.mean(measures)),
            "max": float(np.max(measures)),
        },
        "pcg64_grid_disagreement": {
            "grid_size": str(PCG64_GRID_SIZE),
            "aggregate_denominator": str(aggregate_grid_denominator),
            "exact_uniform": {
                "count": str(grid_disagreement_counts["exact_uniform"]),
                "probability": (
                    grid_disagreement_counts["exact_uniform"]
                    / aggregate_grid_denominator
                ),
            },
            "float32_uniform": {
                "count": str(grid_disagreement_counts["float32_uniform"]),
                "probability": (
                    grid_disagreement_counts["float32_uniform"]
                    / aggregate_grid_denominator
                ),
            },
        },
        "max_boundary_shift": max_boundary_shift,
        "support_top_p_membership_mismatch_count": support_mismatches,
        "division_accounting": {
            "label": (
                "rn_probability_array_divisions_only_excludes_fraction_midpoint_"
                "work_not_gpu_instructions"
            ),
            "literal_float64": {
                "normalization_passes_per_row": literal_normalization_passes,
                "selection_passes_per_row": literal_selection_passes,
                "division_count_per_row": literal_divisions_per_row,
                "aggregate_normalization_passes": sum(literal_normalization_passes),
                "aggregate_selection_passes": sum(literal_selection_passes),
                "aggregate_division_count": sum(literal_divisions_per_row),
            },
            "reduced_exact_float64": {
                "normalization_passes_per_row": reduced_exact_normalization_passes,
                "selection_passes_per_row": reduced_exact_selection_passes,
                "division_count_per_row": reduced_exact_divisions_per_row,
                "aggregate_normalization_passes": sum(
                    reduced_exact_normalization_passes
                ),
                "aggregate_selection_passes": 0,
                "aggregate_division_count": sum(reduced_exact_divisions_per_row),
                "second_normalization_skip_count": reduced_exact_second_skips,
                "second_normalization_skip_rate": reduced_exact_second_skips
                / int(ids.shape[0]),
            },
            "reduced_float32": {
                "normalization_passes_per_row": candidate_normalization_passes,
                "selection_passes_per_row": candidate_selection_passes,
                "division_count_per_row": candidate_divisions_per_row,
                "aggregate_normalization_passes": sum(candidate_normalization_passes),
                "aggregate_selection_passes": 0,
                "aggregate_division_count": sum(candidate_divisions_per_row),
                "second_normalization_skip_count": candidate_second_skips,
                "second_normalization_skip_rate": candidate_second_skips
                / int(ids.shape[0]),
            },
            "divided_float32_reference": {
                "normalization_passes_per_row": candidate_normalization_passes,
                "selection_passes_per_row": candidate_numpy_boundary_passes,
                "division_count_per_row": candidate_reference_divisions_per_row,
                "aggregate_normalization_passes": sum(candidate_normalization_passes),
                "aggregate_selection_passes": sum(candidate_numpy_boundary_passes),
                "aggregate_division_count": sum(candidate_reference_divisions_per_row),
                "purpose": "required_fail_closed_cross_check_only",
            },
        },
        "contract": {
            "candidate_ids_dtype": "int64",
            "candidate_values_dtype": "float32",
            "candidate_probs_dtype": "float32",
            "uniforms_dtype": "float64",
            "max_candidates": 20,
            "numpy_version": REQUIRED_NUMPY_VERSION,
        },
        "arithmetic": {
            "exact_host": "float64",
            "reduced_exact_schedule": {
                "id": REDUCED_EXACT_SCHEDULE_ID,
                "first_normalization": "always_divide_by_rank_order_rn64_sum",
                "second_normalization": ("divide_iff_rn64_sum_bits_not_positive_one"),
                "raw_cdf": "sequential_rn64_cumsum",
                "selection": "nextafter_midpoint_ties_to_even",
                "final_cdf_division": "not_evaluated_on_selection_path",
                "raw_ratio_shortcut": False,
            },
            "candidate": "float32",
            "candidate_schedule": {
                "id": CANDIDATE_SCHEDULE_ID,
                "first_normalization": "always_divide_by_rank_order_rn32_sum",
                "second_normalization": "divide_iff_rn32_sum_bits_not_one",
                "raw_cdf": "sequential_rn32_cumsum",
                "selection": "rn32_nextafter_midpoint_ties_to_even",
                "final_cdf_division": "not_evaluated_on_selection_path",
                "divided_cdf_reference": ("required_fail_closed_cross_check_only"),
                "evaluation_scope": "benchmark_experiment_only",
                "retention_eligible": False,
                "raw_q_collapse": False,
            },
            "uniform_variants": ["float32_uniform", "exact_uniform"],
        },
    }


def load_and_analyze(
    path: Path,
    *,
    top_p: float = 0.95,
    max_mismatch_indices: int = 64,
    benchmark_cpu: bool = False,
    timing_warmups: int = 2,
    timing_repeats: int = 7,
) -> dict[str, object]:
    """Load the fixed NPZ contract without coercing any input dtype."""

    _require_numpy_version()
    required = ("candidate_ids", "candidate_values", "candidate_probs", "uniforms")
    with np.load(path, allow_pickle=False) as capture:
        missing = [name for name in required if name not in capture]
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {', '.join(missing)}")
        arrays = {name: capture[name] for name in required}
    report = analyze_arrays(
        **arrays,
        top_p=top_p,
        max_mismatch_indices=max_mismatch_indices,
    )
    if benchmark_cpu:
        report["diagnostic_cpu_timing"] = benchmark_prepared_rows(
            arrays["candidate_ids"],
            arrays["candidate_values"],
            arrays["candidate_probs"],
            top_p=top_p,
            warmups=timing_warmups,
            repeats=timing_repeats,
        )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="captured top-20 NPZ")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-mismatch-indices", type=int, default=64)
    parser.add_argument("--benchmark-cpu", action="store_true")
    parser.add_argument("--timing-warmups", type=int, default=2)
    parser.add_argument("--timing-repeats", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = load_and_analyze(
        args.capture,
        top_p=args.top_p,
        max_mismatch_indices=args.max_mismatch_indices,
        benchmark_cpu=args.benchmark_cpu,
        timing_warmups=args.timing_warmups,
        timing_repeats=args.timing_repeats,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
