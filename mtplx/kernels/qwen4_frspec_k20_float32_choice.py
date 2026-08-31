"""Fixed-K20 float32 FRSpec choice kernel with exact PCG64 comparisons.

The hot callable is intentionally narrow. Its candidate contract and the
five-word uniform descriptors are validated when the experimental route is
constructed; the installed call only dispatches the fixed Metal program.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import importlib
import math
from typing import Callable

import numpy as np


K20 = 20
MIDPOINT_DESCRIPTOR_WORDS = 5
PCG64_GRID_SIZE = 1 << 53
PCG64_HIGH_MASK = (1 << 21) - 1
REQUIRED_NUMPY_VERSION = "2.4.4"
SCHEDULE_ID = "qwen4_frspec_k20_rn32_reduced_exact_pcg64_midpoint_v1"
SELFCHECK_CASES = 4


METAL_HEADER = f"""
    using namespace metal;
    constant constexpr uint K = {K20};
    constant constexpr uint DESCRIPTOR_WORDS = {MIDPOINT_DESCRIPTOR_WORDS};

    inline uint bit_length_u64(ulong value) {{
        return 64u - uint(clz(value));
    }}

    inline float numpy_pairwise_sum(
        threadgroup const float* values,
        uint count
    ) {{
        if (count < 8u) {{
            float result = -0.0f;
            for (uint i = 0; i < count; ++i) result += values[i];
            return result;
        }}
        float partials[8];
        for (uint i = 0; i < 8u; ++i) partials[i] = values[i];
        uint i = 8u;
        uint vector_end = count - (count % 8u);
        for (; i < vector_end; i += 8u) {{
            for (uint lane = 0; lane < 8u; ++lane) {{
                partials[lane] += values[i + lane];
            }}
        }}
        float result =
            ((partials[0] + partials[1]) + (partials[2] + partials[3])) +
            ((partials[4] + partials[5]) + (partials[6] + partials[7]));
        for (; i < count; ++i) result += values[i];
        return result;
    }}

    inline void positive_float32_parts(
        uint bits,
        thread ulong& significand,
        thread int& exponent
    ) {{
        uint exponent_bits = (bits >> 23) & 0xffu;
        uint fraction_bits = bits & 0x7fffffu;
        if (exponent_bits == 0u) {{
            significand = ulong(fraction_bits);
            exponent = -149;
        }} else {{
            significand = ulong(0x800000u | fraction_bits);
            exponent = int(exponent_bits) - 150;
        }}
    }}

    inline int compare_positive_dyadics(
        ulong left_significand,
        int left_exponent,
        ulong right_significand,
        int right_exponent
    ) {{
        int left_top = left_exponent + int(bit_length_u64(left_significand));
        int right_top = right_exponent + int(bit_length_u64(right_significand));
        if (left_top != right_top) {{
            return left_top > right_top ? 1 : -1;
        }}
        if (left_exponent > right_exponent) {{
            left_significand <<= uint(left_exponent - right_exponent);
        }} else if (right_exponent > left_exponent) {{
            right_significand <<= uint(right_exponent - left_exponent);
        }}
        if (left_significand == right_significand) return 0;
        return left_significand > right_significand ? 1 : -1;
    }}

    inline bool ratio_gt_midpoint(
        float numerator,
        float denominator,
        uint midpoint_significand,
        int midpoint_exponent,
        bool upper_endpoint_even
    ) {{
        if (as_type<uint>(numerator) == 0u) return false;
        ulong numerator_significand;
        ulong denominator_significand;
        int numerator_exponent;
        int denominator_exponent;
        positive_float32_parts(
            as_type<uint>(numerator),
            numerator_significand,
            numerator_exponent
        );
        positive_float32_parts(
            as_type<uint>(denominator),
            denominator_significand,
            denominator_exponent
        );
        ulong right_significand =
            denominator_significand * ulong(midpoint_significand);
        int comparison = compare_positive_dyadics(
            numerator_significand,
            numerator_exponent,
            right_significand,
            denominator_exponent + midpoint_exponent
        );
        return comparison > 0 || (comparison == 0 && upper_endpoint_even);
    }}
"""


METAL_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint base = row * K;
    uint descriptor_base = row * DESCRIPTOR_WORDS;

    threadgroup uint sorted_ids[K];
    threadgroup float sorted_values[K];
    threadgroup float sorted_probs[K];
    threadgroup uint retained_ids[K];
    threadgroup float retained_probs[K];
    threadgroup float normalized[K];
    threadgroup uint retained_count;
    threadgroup float first_total;
    threadgroup float second_total;
    threadgroup uint needs_second_normalization;

    if (lane < K) {
        // Preserve the raw pre-sort candidates for the proposal distribution.
        raw_candidate_ids[base + lane] = candidate_ids[base + lane];
        raw_candidate_values[base + lane] = candidate_values[base + lane];
        raw_candidate_probs[base + lane] = candidate_probs[base + lane];
    }

    if (lane == 0u) {
        for (uint i = 0; i < K; ++i) {
            uint token = candidate_ids[base + i];
            float value = candidate_values[base + i];
            float probability = candidate_probs[base + i];
            uint j = i;
            while (
                j > 0u &&
                (
                    sorted_values[j - 1u] < value ||
                    (
                        sorted_values[j - 1u] == value &&
                        sorted_ids[j - 1u] > token
                    )
                )
            ) {
                sorted_ids[j] = sorted_ids[j - 1u];
                sorted_values[j] = sorted_values[j - 1u];
                sorted_probs[j] = sorted_probs[j - 1u];
                --j;
            }
            sorted_ids[j] = token;
            sorted_values[j] = value;
            sorted_probs[j] = probability;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        retained_count = 0u;
        float cumulative_before = 0.0f;
        float top_p = as_type<float>(TOP_P_BITS);
        bool nucleus_enabled = TOP_P_BITS != 0x3f800000u;
        for (uint i = 0; i < K; ++i) {
            float probability = sorted_probs[i];
            bool keep = probability > 0.0f &&
                (!nucleus_enabled || cumulative_before < top_p);
            if (keep) {
                retained_ids[retained_count] = sorted_ids[i];
                retained_probs[retained_count] = probability;
                ++retained_count;
            }
            cumulative_before += probability;
        }

        // Complete the first reduction in rank order, before ID sorting.
        first_total = numpy_pairwise_sum(retained_probs, retained_count);

        // Canonical SparseDistribution support order: ascending token ID.
        for (uint i = 1; i < retained_count; ++i) {
            uint token = retained_ids[i];
            float probability = retained_probs[i];
            uint j = i;
            while (j > 0u && retained_ids[j - 1u] > token) {
                retained_ids[j] = retained_ids[j - 1u];
                retained_probs[j] = retained_probs[j - 1u];
                --j;
            }
            retained_ids[j] = token;
            retained_probs[j] = probability;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < retained_count) {
        float probability = retained_probs[lane] / first_total;
        probability = isfinite(probability) && probability > 0.0f
            ? probability
            : 0.0f;
        normalized[lane] = probability;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        second_total = numpy_pairwise_sum(normalized, retained_count);
        needs_second_normalization =
            as_type<uint>(second_total) != 0x3f800000u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < retained_count && needs_second_normalization != 0u) {
        normalized[lane] /= second_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        float raw_cdf[K];
        float cumulative = 0.0f;
        for (uint i = 0; i < retained_count; ++i) {
            cumulative += normalized[i];
            raw_cdf[i] = cumulative;
        }

        uint midpoint_significand = uniform_descriptors[descriptor_base + 2u];
        int midpoint_exponent = as_type<int>(
            uniform_descriptors[descriptor_base + 3u]
        );
        bool upper_endpoint_even =
            uniform_descriptors[descriptor_base + 4u] != 0u;
        float final_cdf = raw_cdf[retained_count - 1u];
        uint selected = retained_ids[retained_count - 1u];
        for (uint i = 0; i + 1u < retained_count; ++i) {
            if (ratio_gt_midpoint(
                raw_cdf[i],
                final_cdf,
                midpoint_significand,
                midpoint_exponent,
                upper_endpoint_even
            )) {
                selected = retained_ids[i];
                break;
            }
        }
        selected_tokens[row] = selected;
    }
"""


def _require_numpy_version() -> None:
    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise RuntimeError(
            f"Qwen4 K20 float32 choice requires exact NumPy "
            f"{REQUIRED_NUMPY_VERSION}; found {np.__version__}"
        )


def _validate_top_p(top_p: float) -> np.float32:
    value = float(top_p)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1]")
    rounded = np.float32(value)
    if not np.isfinite(rounded) or not np.float32(0.0) < rounded <= np.float32(1.0):
        raise ValueError("top_p must remain finite and positive in float32")
    return rounded


def _grid_integer(uniform: np.float64) -> int:
    value = np.float64(uniform)
    if not np.isfinite(value):
        raise ValueError("uniforms must be finite")
    if np.signbit(value):
        raise ValueError("uniforms must be non-negative PCG64 values")
    if not np.float64(0.0) <= value < np.float64(1.0):
        raise ValueError("uniforms must be in [0, 1)")
    scaled = np.ldexp(value, 53)
    integer = int(scaled)
    if scaled != np.float64(integer):
        raise ValueError("uniforms must lie on the exact PCG64 53-bit grid")
    return integer


def _fraction_from_float(value: float | np.floating) -> Fraction:
    return Fraction(*float(value).as_integer_ratio())


def _descriptor_for_grid_integer(integer: int) -> tuple[int, int, int, int, int]:
    if not 0 <= integer < PCG64_GRID_SIZE:
        raise ValueError("PCG64 grid integer must be in [0, 2**53)")
    uniform = np.ldexp(np.float64(integer), -53)
    exact_uniform = Fraction(integer, PCG64_GRID_SIZE)
    rounded_uniform = np.float32(uniform)
    if _fraction_from_float(rounded_uniform) > exact_uniform:
        upper = rounded_uniform
        lower = np.nextafter(rounded_uniform, np.float32(-np.inf), dtype=np.float32)
    else:
        lower = rounded_uniform
        upper = np.nextafter(rounded_uniform, np.float32(np.inf), dtype=np.float32)
    midpoint = (_fraction_from_float(lower) + _fraction_from_float(upper)) / 2
    denominator_power = midpoint.denominator.bit_length() - 1
    if midpoint.denominator != 1 << denominator_power:
        raise AssertionError("binary32 midpoint denominator must be a power of two")
    significand = midpoint.numerator
    if not 0 < significand <= np.iinfo(np.uint32).max:
        raise AssertionError("binary32 midpoint significand must fit uint32")
    exponent = -denominator_power
    upper_even = int((int(upper.view(np.uint32)) & 1) == 0)
    return (
        (integer >> 32) & PCG64_HIGH_MASK,
        integer & 0xFFFF_FFFF,
        significand,
        exponent,
        upper_even,
    )


def build_pcg64_midpoint_descriptors(uniforms: np.ndarray) -> np.ndarray:
    """Encode exact PCG64 uniforms as audited division-free RN32 thresholds."""

    _require_numpy_version()
    values = np.asarray(uniforms)
    if values.dtype != np.dtype(np.float64):
        raise ValueError("uniforms must have dtype float64")
    if values.ndim != 1:
        raise ValueError("uniforms must be one-dimensional")
    descriptors = np.empty((values.size, MIDPOINT_DESCRIPTOR_WORDS), dtype=np.uint32)
    for row, uniform in enumerate(values):
        high, low, significand, exponent, upper_even = _descriptor_for_grid_integer(
            _grid_integer(uniform)
        )
        descriptors[row] = np.array(
            [high, low, significand, np.uint32(np.int32(exponent)), upper_even],
            dtype=np.uint32,
        )
    validate_pcg64_midpoint_descriptors(descriptors)
    return descriptors


def validate_pcg64_midpoint_descriptors(descriptors: np.ndarray) -> None:
    """Fail closed if a descriptor is malformed or internally inconsistent."""

    _require_numpy_version()
    observed = np.asarray(descriptors)
    if observed.dtype != np.dtype(np.uint32):
        raise ValueError("uniform descriptors must have dtype uint32")
    if observed.ndim != 2 or observed.shape[1:] != (MIDPOINT_DESCRIPTOR_WORDS,):
        raise ValueError("uniform descriptors must have shape [rows, 5]")
    for row in range(observed.shape[0]):
        high = int(observed[row, 0])
        if high > PCG64_HIGH_MASK:
            raise ValueError("uniform descriptor R must be smaller than 2**53")
        integer = (high << 32) | int(observed[row, 1])
        expected = _descriptor_for_grid_integer(integer)
        expected_words = np.array(
            [
                expected[0],
                expected[1],
                expected[2],
                np.uint32(np.int32(expected[3])),
                expected[4],
            ],
            dtype=np.uint32,
        )
        if not np.array_equal(observed[row], expected_words):
            raise ValueError(f"inconsistent midpoint descriptor at row {row}")


def _validate_reference_inputs(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids)
    values = np.asarray(candidate_values)
    probs = np.asarray(candidate_probs)
    if ids.dtype != np.dtype(np.uint32):
        raise ValueError("candidate_ids must be uint32")
    if values.dtype != np.dtype(np.float32):
        raise ValueError("candidate_values must be float32")
    if probs.dtype != np.dtype(np.float32):
        raise ValueError("candidate_probs must be float32")
    if ids.ndim != 2 or ids.shape[1:] != (K20,):
        raise ValueError("candidate arrays must have shape [rows, 20]")
    if values.shape != ids.shape or probs.shape != ids.shape:
        raise ValueError("candidate arrays must share shape [rows, 20]")
    if ids.shape[0] == 0:
        raise ValueError("candidate arrays must contain at least one row")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate_values must be finite")
    if not np.all(np.isfinite(probs)):
        raise ValueError("candidate_probs must be finite")
    if np.any(probs < np.float32(0.0)) or np.any(probs > np.float32(1.0)):
        raise ValueError("candidate_probs must be in [0, 1]")
    if np.any(np.sum(probs, axis=1, dtype=np.float64) <= 0.0):
        raise ValueError("each candidate row must have positive probability mass")
    for row in ids:
        if np.unique(row).size != K20:
            raise ValueError("candidate_ids must be unique within each row")
    validate_pcg64_midpoint_descriptors(descriptors)
    if descriptors.shape[0] != ids.shape[0]:
        raise ValueError("uniform descriptor row count must match candidates")
    return ids, values, probs, descriptors


def _add32(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(left + right)


def _numpy_pairwise_sum(values: list[np.float32] | np.ndarray) -> np.float32:
    """Reproduce NumPy 2.4.4's float pairwise sum for the bounded K20 row."""

    count = len(values)
    if count < 8:
        result = np.float32(-0.0)
        for value in values:
            result = _add32(result, np.float32(value))
        return result
    partials = [np.float32(values[index]) for index in range(8)]
    index = 8
    vector_end = count - (count % 8)
    while index < vector_end:
        for lane in range(8):
            partials[lane] = _add32(partials[lane], np.float32(values[index + lane]))
        index += 8
    left = _add32(_add32(partials[0], partials[1]), _add32(partials[2], partials[3]))
    right = _add32(_add32(partials[4], partials[5]), _add32(partials[6], partials[7]))
    result = _add32(left, right)
    while index < count:
        result = _add32(result, np.float32(values[index]))
        index += 1
    return result


def _prepare_reference_row(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    top_p: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(
        range(K20), key=lambda index: (-float(values[index]), int(ids[index]))
    )
    cumulative_before = np.float32(0.0)
    retained_ids: list[np.uint32] = []
    retained_probs: list[np.float32] = []
    nucleus_enabled = top_p != np.float32(1.0)
    for index in ranked:
        probability = np.float32(probs[index])
        if probability > np.float32(0.0) and (
            not nucleus_enabled or cumulative_before < top_p
        ):
            retained_ids.append(np.uint32(ids[index]))
            retained_probs.append(probability)
        cumulative_before = _add32(cumulative_before, probability)

    first_total = _numpy_pairwise_sum(retained_probs)
    token_order = sorted(range(len(retained_ids)), key=retained_ids.__getitem__)
    ordered_ids = np.array([retained_ids[index] for index in token_order], np.uint32)
    normalized = np.empty(len(token_order), dtype=np.float32)
    for output, index in enumerate(token_order):
        probability = np.float32(retained_probs[index] / first_total)
        if not np.isfinite(probability) or probability <= np.float32(0.0):
            probability = np.float32(0.0)
        normalized[output] = probability
    second_total = _numpy_pairwise_sum(normalized)
    if second_total.view(np.uint32) != np.float32(1.0).view(np.uint32):
        for index in range(normalized.size):
            normalized[index] = np.float32(normalized[index] / second_total)
    raw_cdf = np.empty_like(normalized)
    cumulative = np.float32(0.0)
    for index, probability in enumerate(normalized):
        cumulative = _add32(cumulative, probability)
        raw_cdf[index] = cumulative
    return ordered_ids, raw_cdf


def _positive_float32_parts(value: np.float32) -> tuple[int, int]:
    bits = int(np.float32(value).view(np.uint32))
    exponent_bits = (bits >> 23) & 0xFF
    fraction_bits = bits & 0x7F_FFFF
    if exponent_bits == 0:
        return fraction_bits, -149
    return 0x80_0000 | fraction_bits, exponent_bits - 150


def _compare_dyadics(
    left_significand: int,
    left_exponent: int,
    right_significand: int,
    right_exponent: int,
) -> int:
    left_top = left_exponent + left_significand.bit_length()
    right_top = right_exponent + right_significand.bit_length()
    if left_top != right_top:
        return 1 if left_top > right_top else -1
    if left_exponent > right_exponent:
        left_significand <<= left_exponent - right_exponent
    elif right_exponent > left_exponent:
        right_significand <<= right_exponent - left_exponent
    return (left_significand > right_significand) - (
        left_significand < right_significand
    )


def _ratio_gt_descriptor(
    numerator: np.float32,
    denominator: np.float32,
    descriptor: np.ndarray,
) -> bool:
    numerator_significand, numerator_exponent = _positive_float32_parts(numerator)
    denominator_significand, denominator_exponent = _positive_float32_parts(denominator)
    midpoint_significand = int(descriptor[2])
    midpoint_exponent = int(descriptor[3].view(np.int32))
    comparison = _compare_dyadics(
        numerator_significand,
        numerator_exponent,
        denominator_significand * midpoint_significand,
        denominator_exponent + midpoint_exponent,
    )
    return comparison > 0 or (comparison == 0 and bool(descriptor[4]))


def reference_qwen4_frspec_k20_float32_choice(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    uniform_descriptors: np.ndarray,
    *,
    top_p: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU construction/self-check oracle for the fixed Metal schedule."""

    _require_numpy_version()
    ids, values, probs, descriptors = _validate_reference_inputs(
        candidate_ids, candidate_values, candidate_probs, uniform_descriptors
    )
    rounded_top_p = _validate_top_p(top_p)
    selected = np.empty(ids.shape[0], dtype=np.uint32)
    for row in range(ids.shape[0]):
        ordered_ids, raw_cdf = _prepare_reference_row(
            ids[row], values[row], probs[row], rounded_top_p
        )
        token = ordered_ids[-1]
        for index, boundary in enumerate(raw_cdf[:-1]):
            if _ratio_gt_descriptor(boundary, raw_cdf[-1], descriptors[row]):
                token = ordered_ids[index]
                break
        selected[row] = token
    return selected, ids.copy(), values.copy(), probs.copy()


def reference_literal_divided_cdf_token(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    uniform: np.float64,
    *,
    top_p: float = 0.95,
) -> int:
    """Literal final-RN32-division oracle used only by CPU self-checks."""

    descriptor = build_pcg64_midpoint_descriptors(np.array([uniform], dtype=np.float64))
    ids, values, probs, _ = _validate_reference_inputs(
        np.asarray(candidate_ids)[None, :],
        np.asarray(candidate_values)[None, :],
        np.asarray(candidate_probs)[None, :],
        descriptor,
    )
    ordered_ids, raw_cdf = _prepare_reference_row(
        ids[0], values[0], probs[0], _validate_top_p(top_p)
    )
    for index, boundary in enumerate(raw_cdf[:-1]):
        divided = np.float32(boundary / raw_cdf[-1])
        if np.float64(divided) > uniform:
            return int(ordered_ids[index])
    return int(ordered_ids[-1])


@lru_cache(maxsize=None)
def _metal_kernel(top_p_bits: int):
    mx = importlib.import_module("mlx.core")
    header = (
        METAL_HEADER + f"\nconstant constexpr uint TOP_P_BITS = 0x{top_p_bits:08x}u;\n"
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen4_frspec_k20_float32_choice_p{top_p_bits:08x}",
        input_names=[
            "candidate_ids",
            "candidate_values",
            "candidate_probs",
            "uniform_descriptors",
        ],
        output_names=[
            "selected_tokens",
            "raw_candidate_ids",
            "raw_candidate_values",
            "raw_candidate_probs",
        ],
        header=header,
        source=METAL_SOURCE,
        ensure_row_contiguous=True,
    )


def bind_qwen4_frspec_k20_float32_choice(
    *, top_p: float = 0.95
) -> Callable[[object, object, object, object], tuple[object, object, object, object]]:
    """Bind one fixed-top-p K20 route; returned callable only dispatches it."""

    _require_numpy_version()
    rounded_top_p = _validate_top_p(top_p)
    top_p_bits = int(rounded_top_p.view(np.uint32))
    mx = importlib.import_module("mlx.core")
    kernel = _metal_kernel(top_p_bits)

    def apply(
        candidate_ids: object,
        candidate_values: object,
        candidate_probs: object,
        uniform_descriptors: object,
    ) -> tuple[object, object, object, object]:
        rows = int(candidate_ids.shape[0])
        outputs = kernel(
            inputs=[
                candidate_ids,
                candidate_values,
                candidate_probs,
                uniform_descriptors,
            ],
            grid=(rows * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (rows,),
                (rows, K20),
                (rows, K20),
                (rows, K20),
            ],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.float32],
        )
        return outputs[0], outputs[1], outputs[2], outputs[3]

    return apply


def selfcheck_qwen4_frspec_k20_float32_choice() -> dict[str, int | str]:
    """Run deterministic CPU-only descriptor and midpoint-selection checks."""

    ids = np.tile(np.arange(K20, dtype=np.uint32), (SELFCHECK_CASES, 1))
    values = np.tile(
        np.linspace(2.0, -2.0, K20, dtype=np.float32), (SELFCHECK_CASES, 1)
    )
    base_probs = np.array(
        [0.21, 0.17, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02] + [0.01] * 10,
        dtype=np.float32,
    )
    probs = np.tile(base_probs, (SELFCHECK_CASES, 1))
    integers = np.array([0, 1, 1 << 52, PCG64_GRID_SIZE - 1], dtype=np.uint64)
    uniforms = np.ldexp(integers.astype(np.float64), -53)
    descriptors = build_pcg64_midpoint_descriptors(uniforms)
    selected, *_ = reference_qwen4_frspec_k20_float32_choice(
        ids, values, probs, descriptors
    )
    literal = np.array(
        [
            reference_literal_divided_cdf_token(
                ids[row], values[row], probs[row], uniforms[row]
            )
            for row in range(SELFCHECK_CASES)
        ],
        dtype=np.uint32,
    )
    if not np.array_equal(selected, literal):
        raise RuntimeError(
            "K20 reduced midpoint selector failed literal RN32 self-check"
        )
    return {
        "schedule_id": SCHEDULE_ID,
        "k": K20,
        "descriptor_words": MIDPOINT_DESCRIPTOR_WORDS,
        "cases": SELFCHECK_CASES,
    }
