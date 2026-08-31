"""Exact binary64 controller for the fixed PR391 D3/M4 experiment."""

from __future__ import annotations

import math
from functools import lru_cache
import importlib
from typing import Callable

import numpy as np

from ._metal_softfloat64_v0_1_1 import (
    METAL_SOFTFLOAT_COMMIT,
    METAL_SOFTFLOAT_SOURCE,
    METAL_SOFTFLOAT_VERSION,
)


K20 = 20
DEPTH = 3
SELECTED_NONE = 0
SELECTED_CORRECTION = 1
SELECTED_BONUS = 2


def _validate_top_p(top_p: float) -> np.float64:
    value = np.float64(top_p)
    if not math.isfinite(float(value)) or not np.float64(0.0) < value <= np.float64(1.0):
        raise ValueError("top_p must be finite and in (0, 1]")
    return value


def _prepare_batched_candidate_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(candidate_ids)
    values = np.asarray(candidate_values)
    probs = np.asarray(candidate_probs)
    if ids.dtype != np.dtype(np.uint32) or ids.ndim != 1:
        raise ValueError("candidate_ids must be one-dimensional uint32")
    if values.dtype != np.dtype(np.float32) or values.shape != ids.shape:
        raise ValueError("candidate_values must be float32 with the ID shape")
    if probs.dtype != np.dtype(np.float32) or probs.shape != ids.shape:
        raise ValueError("candidate_probs must be float32 with the ID shape")
    if ids.size == 0 or ids.size > K20 or np.unique(ids).size != ids.size:
        raise ValueError("candidate IDs must contain between one and 20 unique tokens")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate values must be finite")
    if not np.all(np.isfinite(probs)) or np.any(probs < np.float32(0.0)):
        raise ValueError("candidate probabilities must be finite and non-negative")

    bounded_top_p = _validate_top_p(top_p)
    probabilities = probs.astype(np.float64)
    rank = np.lexsort((ids.astype(np.uint64), -values.astype(np.float64)))
    ranked_ids = ids[rank]
    ranked_probs = probabilities[rank]
    cumulative_before = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(ranked_probs[:-1], dtype=np.float64))
    )
    retained = ranked_probs > np.float64(0.0)
    if bounded_top_p < np.float64(1.0):
        retained &= cumulative_before < bounded_top_p
    retained_ids = ranked_ids[retained]
    retained_probs = ranked_probs[retained]
    first_total = np.sum(retained_probs, dtype=np.float64)
    if not np.isfinite(first_total) or first_total <= np.float64(0.0):
        raise ValueError("candidate row must retain positive finite mass")

    token_order = np.argsort(retained_ids)
    prepared_ids = retained_ids[token_order]
    normalized_once = retained_probs[token_order] / first_total
    return prepared_ids.astype(np.uint32, copy=False), normalized_once


def _renormalize_sparse_probabilities(probabilities: np.ndarray) -> np.ndarray:
    sanitized = np.where(
        np.isfinite(probabilities) & (probabilities > np.float64(0.0)),
        probabilities,
        np.float64(0.0),
    )
    second_total = np.sum(sanitized, dtype=np.float64)
    return sanitized / second_total


def _prepare_candidate_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    top_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare one SparseDistribution row, including its second normalization."""

    prepared_ids, normalized_once = _prepare_batched_candidate_row(
        candidate_ids,
        candidate_values,
        candidate_probs,
        top_p=top_p,
    )
    return prepared_ids, _renormalize_sparse_probabilities(normalized_once)


def reference_select_candidate_row(
    candidate_ids: np.ndarray,
    candidate_values: np.ndarray,
    candidate_probs: np.ndarray,
    uniform: np.float64,
    *,
    top_p: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Literal NumPy-2.4.4 row preparation and right-sided selection."""

    draw = np.float64(uniform)
    if not np.isfinite(draw) or not np.float64(0.0) <= draw < np.float64(1.0):
        raise ValueError("uniform must be finite and in [0, 1)")
    prepared_ids, prepared_probs = _prepare_candidate_row(
        candidate_ids,
        candidate_values,
        candidate_probs,
        top_p=top_p,
    )
    cdf = np.cumsum(prepared_probs, dtype=np.float64)
    cdf /= np.sum(prepared_probs, dtype=np.float64)
    index = min(
        int(np.searchsorted(cdf, draw, side="right")),
        int(prepared_ids.size) - 1,
    )
    return (
        int(prepared_ids[index]),
        prepared_ids,
        prepared_probs.view(np.uint64),
    )


def _sample_prepared(
    token_ids: np.ndarray,
    probabilities: np.ndarray,
    uniform: np.float64,
) -> int:
    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf /= np.sum(probabilities, dtype=np.float64)
    index = min(
        int(np.searchsorted(cdf, np.float64(uniform), side="right")),
        int(token_ids.size) - 1,
    )
    return int(token_ids[index])


def _lookup_prepared(
    token_ids: np.ndarray,
    probabilities: np.ndarray,
    token: int,
) -> np.float64:
    hits = np.nonzero(token_ids == np.uint32(token))[0]
    return (
        np.float64(0.0)
        if hits.size == 0
        else np.float64(probabilities[int(hits[0])])
    )


def _prepare_residual(
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    union_ids = np.union1d(target_ids, draft_ids).astype(np.uint32, copy=False)
    residual = np.array(
        [
            max(
                _lookup_prepared(target_ids, target_probs, int(token))
                - _lookup_prepared(draft_ids, draft_probs, int(token)),
                np.float64(0.0),
            )
            for token in union_ids
        ],
        dtype=np.float64,
    )
    residual = np.where(
        np.isfinite(residual) & (residual > np.float64(0.0)),
        residual,
        np.float64(0.0),
    )
    keep = residual > np.float64(0.0)
    first_total = np.sum(residual[keep], dtype=np.float64)
    if not np.isfinite(first_total) or first_total <= np.float64(0.0):
        return None
    normalized_once = residual[keep] / first_total
    sanitized = np.where(
        np.isfinite(normalized_once) & (normalized_once > np.float64(0.0)),
        normalized_once,
        np.float64(0.0),
    )
    second_total = np.sum(sanitized, dtype=np.float64)
    return union_ids[keep], sanitized / second_total


def reference_pr391_softfloat64_verifier_decision(
    draft_tokens: np.ndarray,
    draft_ids: np.ndarray,
    draft_values: np.ndarray,
    draft_probs: np.ndarray,
    target_ids: np.ndarray,
    target_values: np.ndarray,
    target_probs: np.ndarray,
    uniforms: np.ndarray,
    stop_ids: np.ndarray,
    *,
    stop_count: int,
    bonus_allowed: bool,
) -> tuple[np.ndarray, ...]:
    """Literal NumPy reference for the exact fixed-D3/M4 controller."""

    tokens = np.asarray(draft_tokens)
    draft_support = np.asarray(draft_ids)
    draft_scores = np.asarray(draft_values)
    draft_masses = np.asarray(draft_probs)
    target_support = np.asarray(target_ids)
    target_scores = np.asarray(target_values)
    target_masses = np.asarray(target_probs)
    draws = np.asarray(uniforms)
    stops = np.asarray(stop_ids)
    if tokens.dtype != np.dtype(np.uint32) or tokens.shape != (DEPTH,):
        raise ValueError("draft_tokens must be uint32 with shape [3]")
    if draft_support.shape != (DEPTH, K20):
        raise ValueError("draft candidate arrays must have shape [3, 20]")
    if target_support.shape != (DEPTH + 1, K20):
        raise ValueError("target candidate arrays must have shape [4, 20]")
    if draft_scores.shape != draft_support.shape or draft_masses.shape != draft_support.shape:
        raise ValueError("draft candidate arrays must share one shape")
    if target_scores.shape != target_support.shape or target_masses.shape != target_support.shape:
        raise ValueError("target candidate arrays must share one shape")
    if draws.dtype != np.dtype(np.float64) or draws.shape != (DEPTH + 1,):
        raise ValueError("uniforms must be float64 with shape [4]")
    if not np.all(np.isfinite(draws)) or np.any(draws < 0.0) or np.any(draws >= 1.0):
        raise ValueError("uniforms must be finite and in [0, 1)")
    if stops.dtype != np.dtype(np.uint32) or stops.ndim != 1:
        raise ValueError("stop_ids must be one-dimensional uint32")
    bounded_stop_count = int(stop_count)
    if not 0 <= bounded_stop_count <= stops.size:
        raise ValueError("stop_count must be in [0, len(stop_ids)]")

    prepared_draft = [
        _prepare_candidate_row(
            draft_support[row],
            draft_scores[row],
            draft_masses[row],
            top_p=0.95,
        )
        for row in range(DEPTH)
    ]
    prepared_target = [
        _prepare_batched_candidate_row(
            target_support[row],
            target_scores[row],
            target_masses[row],
            top_p=0.95,
        )
        for row in range(DEPTH + 1)
    ]

    accepted = np.zeros(1, dtype=np.uint32)
    first_reject = np.array([-1], dtype=np.int32)
    selected_token = np.zeros(1, dtype=np.uint32)
    selected_kind = np.zeros(1, dtype=np.uint32)
    selected_present = np.zeros(1, dtype=np.uint32)
    draws_used = np.zeros(1, dtype=np.uint32)
    accept_probability_bits = np.zeros(DEPTH, dtype=np.uint64)
    active_stops = {int(token) for token in stops[:bounded_stop_count]}

    for depth in range(DEPTH):
        token = int(tokens[depth])
        draft_row_ids, draft_row_probs = prepared_draft[depth]
        target_row_ids, target_row_probs = prepared_target[depth]
        p_value = _lookup_prepared(target_row_ids, target_row_probs, token)
        q_value = _lookup_prepared(draft_row_ids, draft_row_probs, token)
        if q_value <= np.float64(0.0):
            accept_probability = np.float64(1.0 if p_value > 0.0 else 0.0)
        else:
            accept_probability = np.minimum(
                np.float64(1.0), np.float64(p_value / q_value)
            )
        accept_probability_bits[depth] = accept_probability.view(np.uint64)
        if draws[depth] <= accept_probability:
            accepted[0] = np.uint32(depth + 1)
            if token in active_stops:
                draws_used[0] = np.uint32(depth + 1)
                return (
                    accepted,
                    first_reject,
                    selected_token,
                    selected_kind,
                    selected_present,
                    draws_used,
                    accept_probability_bits,
                )
            continue

        first_reject[0] = np.int32(depth)
        sparse_target_probs = _renormalize_sparse_probabilities(target_row_probs)
        residual = _prepare_residual(
            target_row_ids,
            sparse_target_probs,
            draft_row_ids,
            draft_row_probs,
        )
        correction_ids, correction_probs = (
            (target_row_ids, sparse_target_probs) if residual is None else residual
        )
        selected_token[0] = np.uint32(
            _sample_prepared(correction_ids, correction_probs, draws[depth + 1])
        )
        selected_kind[0] = np.uint32(SELECTED_CORRECTION)
        selected_present[0] = np.uint32(1)
        draws_used[0] = np.uint32(depth + 2)
        return (
            accepted,
            first_reject,
            selected_token,
            selected_kind,
            selected_present,
            draws_used,
            accept_probability_bits,
        )

    accepted[0] = np.uint32(DEPTH)
    draws_used[0] = np.uint32(DEPTH)
    if bonus_allowed:
        bonus_ids, bonus_probs = prepared_target[DEPTH]
        selected_token[0] = np.uint32(
            _sample_prepared(bonus_ids, bonus_probs, draws[DEPTH])
        )
        selected_kind[0] = np.uint32(SELECTED_BONUS)
        selected_present[0] = np.uint32(1)
        draws_used[0] = np.uint32(DEPTH + 1)
    return (
        accepted,
        first_reject,
        selected_token,
        selected_kind,
        selected_present,
        draws_used,
        accept_probability_bits,
    )


METAL_HELPERS = r"""
    using namespace metal;
    constant constexpr uint D = 3;
    constant constexpr uint K = 20;
    constant constexpr uint RESIDUAL_CAPACITY = 2 * K;
    constant constexpr uint SELECTED_NONE = 0;
    constant constexpr uint SELECTED_CORRECTION = 1;
    constant constexpr uint SELECTED_BONUS = 2;
    constant constexpr ulong F64_ZERO = 0x0000000000000000UL;
    constant constexpr ulong F64_NEGATIVE_ZERO = 0x8000000000000000UL;
    constant constexpr ulong F64_ONE = 0x3ff0000000000000UL;
    constant constexpr ulong F64_TOP_P = 0x3fee666666666666UL;

    inline ulong f32_bits_to_f64(float value) {
        return __softfloat64_cvt_f32_to_f64(as_type<uint>(value));
    }

    inline ulong numpy_pairwise_sum_f64(
        thread const ulong* values,
        uint count
    ) {
        if (count < 8u) {
            ulong result = F64_NEGATIVE_ZERO;
            for (uint index = 0; index < count; ++index) {
                result = __softfloat64_fadd(result, values[index], 0u);
            }
            return result;
        }
        ulong partials[8];
        for (uint lane = 0; lane < 8u; ++lane) partials[lane] = values[lane];
        uint index = 8u;
        uint vector_end = count - (count % 8u);
        for (; index < vector_end; index += 8u) {
            for (uint lane = 0; lane < 8u; ++lane) {
                partials[lane] = __softfloat64_fadd(
                    partials[lane], values[index + lane], 0u
                );
            }
        }
        ulong left = __softfloat64_fadd(
            __softfloat64_fadd(partials[0], partials[1], 0u),
            __softfloat64_fadd(partials[2], partials[3], 0u),
            0u
        );
        ulong right = __softfloat64_fadd(
            __softfloat64_fadd(partials[4], partials[5], 0u),
            __softfloat64_fadd(partials[6], partials[7], 0u),
            0u
        );
        ulong result = __softfloat64_fadd(left, right, 0u);
        for (; index < count; ++index) {
            result = __softfloat64_fadd(result, values[index], 0u);
        }
        return result;
    }

    inline void prepare_batched_candidate_row(
        device const uint* candidate_ids,
        device const float* candidate_values,
        device const float* candidate_probs,
        uint base,
        thread uint* prepared_ids,
        thread ulong* prepared_probs,
        thread uint& prepared_count
    ) {
        uint sorted_ids[K];
        float sorted_values[K];
        ulong sorted_probs[K];
        for (uint index = 0; index < K; ++index) {
            uint token = candidate_ids[base + index];
            float value = candidate_values[base + index];
            ulong probability = f32_bits_to_f64(candidate_probs[base + index]);
            uint insertion = index;
            while (
                insertion > 0u &&
                (
                    sorted_values[insertion - 1u] < value ||
                    (
                        sorted_values[insertion - 1u] == value &&
                        sorted_ids[insertion - 1u] > token
                    )
                )
            ) {
                sorted_ids[insertion] = sorted_ids[insertion - 1u];
                sorted_values[insertion] = sorted_values[insertion - 1u];
                sorted_probs[insertion] = sorted_probs[insertion - 1u];
                --insertion;
            }
            sorted_ids[insertion] = token;
            sorted_values[insertion] = value;
            sorted_probs[insertion] = probability;
        }

        prepared_count = 0u;
        ulong cumulative_before = F64_ZERO;
        for (uint index = 0; index < K; ++index) {
            ulong probability = sorted_probs[index];
            bool keep = __softfloat64_fgt(probability, F64_ZERO) &&
                __softfloat64_flt(cumulative_before, F64_TOP_P);
            if (keep) {
                prepared_ids[prepared_count] = sorted_ids[index];
                prepared_probs[prepared_count] = probability;
                ++prepared_count;
            }
            cumulative_before = __softfloat64_fadd(
                cumulative_before, probability, 0u
            );
        }

        ulong first_total = numpy_pairwise_sum_f64(
            prepared_probs, prepared_count
        );
        for (uint index = 1u; index < prepared_count; ++index) {
            uint token = prepared_ids[index];
            ulong probability = prepared_probs[index];
            uint insertion = index;
            while (insertion > 0u && prepared_ids[insertion - 1u] > token) {
                prepared_ids[insertion] = prepared_ids[insertion - 1u];
                prepared_probs[insertion] = prepared_probs[insertion - 1u];
                --insertion;
            }
            prepared_ids[insertion] = token;
            prepared_probs[insertion] = probability;
        }
        for (uint index = 0u; index < prepared_count; ++index) {
            prepared_probs[index] = __softfloat64_fdiv(
                prepared_probs[index], first_total, 0u
            );
        }
    }

    inline void renormalize_sparse_row(
        thread ulong* prepared_probs,
        uint prepared_count
    ) {
        ulong second_total = numpy_pairwise_sum_f64(
            prepared_probs, prepared_count
        );
        for (uint index = 0u; index < prepared_count; ++index) {
            prepared_probs[index] = __softfloat64_fdiv(
                prepared_probs[index], second_total, 0u
            );
        }
    }

    inline void prepare_candidate_row(
        device const uint* candidate_ids,
        device const float* candidate_values,
        device const float* candidate_probs,
        uint base,
        thread uint* prepared_ids,
        thread ulong* prepared_probs,
        thread uint& prepared_count
    ) {
        prepare_batched_candidate_row(
            candidate_ids,
            candidate_values,
            candidate_probs,
            base,
            prepared_ids,
            prepared_probs,
            prepared_count
        );
        renormalize_sparse_row(prepared_probs, prepared_count);
    }

    inline ulong lookup_prepared_probability(
        thread const uint* ids,
        thread const ulong* probs,
        uint count,
        uint token
    ) {
        for (uint index = 0u; index < count; ++index) {
            if (ids[index] == token) return probs[index];
        }
        return F64_ZERO;
    }

    inline uint sample_prepared_row(
        thread const uint* ids,
        thread const ulong* probs,
        uint count,
        ulong uniform
    ) {
        ulong total = numpy_pairwise_sum_f64(probs, count);
        ulong cumulative = F64_ZERO;
        for (uint index = 0u; index + 1u < count; ++index) {
            cumulative = __softfloat64_fadd(cumulative, probs[index], 0u);
            ulong boundary = __softfloat64_fdiv(cumulative, total, 0u);
            // searchsorted(..., side="right"): equality advances.
            if (__softfloat64_flt(uniform, boundary)) return ids[index];
        }
        return ids[count - 1u];
    }

    inline void prepare_residual_row(
        thread const uint* target_ids,
        thread const ulong* target_probs,
        uint target_count,
        thread const uint* draft_ids,
        thread const ulong* draft_probs,
        uint draft_count,
        thread uint* residual_ids,
        thread ulong* residual_probs,
        thread uint& residual_count
    ) {
        uint union_ids[RESIDUAL_CAPACITY];
        uint union_count = 0u;
        for (uint index = 0u; index < target_count; ++index) {
            union_ids[union_count++] = target_ids[index];
        }
        for (uint index = 0u; index < draft_count; ++index) {
            union_ids[union_count++] = draft_ids[index];
        }
        for (uint index = 1u; index < union_count; ++index) {
            uint token = union_ids[index];
            uint insertion = index;
            while (insertion > 0u && union_ids[insertion - 1u] > token) {
                union_ids[insertion] = union_ids[insertion - 1u];
                --insertion;
            }
            union_ids[insertion] = token;
        }

        residual_count = 0u;
        uint previous = 0u;
        bool have_previous = false;
        for (uint index = 0u; index < union_count; ++index) {
            uint token = union_ids[index];
            if (have_previous && token == previous) continue;
            previous = token;
            have_previous = true;
            ulong target_probability = lookup_prepared_probability(
                target_ids, target_probs, target_count, token
            );
            ulong draft_probability = lookup_prepared_probability(
                draft_ids, draft_probs, draft_count, token
            );
            ulong residual = __softfloat64_fsub(
                target_probability, draft_probability, 0u
            );
            if (__softfloat64_fgt(residual, F64_ZERO)) {
                residual_ids[residual_count] = token;
                residual_probs[residual_count] = residual;
                ++residual_count;
            }
        }
        if (residual_count == 0u) return;

        ulong first_total = numpy_pairwise_sum_f64(
            residual_probs, residual_count
        );
        for (uint index = 0u; index < residual_count; ++index) {
            residual_probs[index] = __softfloat64_fdiv(
                residual_probs[index], first_total, 0u
            );
        }
        ulong second_total = numpy_pairwise_sum_f64(
            residual_probs, residual_count
        );
        for (uint index = 0u; index < residual_count; ++index) {
            residual_probs[index] = __softfloat64_fdiv(
                residual_probs[index], second_total, 0u
            );
        }
    }
"""


METAL_BODY = r"""
    if (thread_position_in_grid.x != 0u) return;

    accepted_count[0] = 0u;
    first_reject[0] = -1;
    selected_token[0] = 0u;
    selected_kind[0] = SELECTED_NONE;
    selected_present[0] = 0u;
    draws_used[0] = 0u;
    for (uint depth = 0u; depth < D; ++depth) {
        accept_probability_bits[depth] = F64_ZERO;
    }

    for (uint depth = 0u; depth < D; ++depth) {
        uint prepared_draft_ids[K];
        ulong prepared_draft_probs[K];
        uint prepared_draft_count;
        uint prepared_target_ids[K];
        ulong prepared_target_probs[K];
        uint prepared_target_count;
        prepare_candidate_row(
            draft_ids,
            draft_values,
            draft_probs,
            depth * K,
            prepared_draft_ids,
            prepared_draft_probs,
            prepared_draft_count
        );
        prepare_batched_candidate_row(
            target_ids,
            target_values,
            target_probs,
            depth * K,
            prepared_target_ids,
            prepared_target_probs,
            prepared_target_count
        );

        uint token = draft_tokens[depth];
        ulong p_value = lookup_prepared_probability(
            prepared_target_ids,
            prepared_target_probs,
            prepared_target_count,
            token
        );
        ulong q_value = lookup_prepared_probability(
            prepared_draft_ids,
            prepared_draft_probs,
            prepared_draft_count,
            token
        );
        ulong accept_probability;
        if (!__softfloat64_fgt(q_value, F64_ZERO)) {
            accept_probability = __softfloat64_fgt(p_value, F64_ZERO)
                ? F64_ONE
                : F64_ZERO;
        } else {
            ulong ratio = __softfloat64_fdiv(p_value, q_value, 0u);
            accept_probability = __softfloat64_flt(ratio, F64_ONE)
                ? ratio
                : F64_ONE;
        }
        accept_probability_bits[depth] = accept_probability;

        if (__softfloat64_fle(uniform_bits[depth], accept_probability)) {
            accepted_count[0] = depth + 1u;
            bool stopped = false;
            for (uint stop_index = 0u; stop_index < stop_count[0]; ++stop_index) {
                if (token == stop_ids[stop_index]) {
                    stopped = true;
                    break;
                }
            }
            if (stopped) {
                draws_used[0] = depth + 1u;
                return;
            }
            continue;
        }

        first_reject[0] = int(depth);
        renormalize_sparse_row(
            prepared_target_probs, prepared_target_count
        );
        uint residual_ids[RESIDUAL_CAPACITY];
        ulong residual_probs[RESIDUAL_CAPACITY];
        uint residual_count;
        prepare_residual_row(
            prepared_target_ids,
            prepared_target_probs,
            prepared_target_count,
            prepared_draft_ids,
            prepared_draft_probs,
            prepared_draft_count,
            residual_ids,
            residual_probs,
            residual_count
        );
        selected_token[0] = residual_count == 0u
            ? sample_prepared_row(
                prepared_target_ids,
                prepared_target_probs,
                prepared_target_count,
                uniform_bits[depth + 1u]
            )
            : sample_prepared_row(
                residual_ids,
                residual_probs,
                residual_count,
                uniform_bits[depth + 1u]
            );
        selected_kind[0] = SELECTED_CORRECTION;
        selected_present[0] = 1u;
        draws_used[0] = depth + 2u;
        return;
    }

    accepted_count[0] = D;
    draws_used[0] = D;
    if (bonus_allowed[0] == 0u) return;

    uint bonus_ids[K];
    ulong bonus_probs[K];
    uint bonus_count;
    prepare_batched_candidate_row(
        target_ids,
        target_values,
        target_probs,
        D * K,
        bonus_ids,
        bonus_probs,
        bonus_count
    );
    selected_token[0] = sample_prepared_row(
        bonus_ids, bonus_probs, bonus_count, uniform_bits[D]
    );
    selected_kind[0] = SELECTED_BONUS;
    selected_present[0] = 1u;
    draws_used[0] = D + 1u;
"""


METAL_SELECTOR_BODY = r"""
    uint row = thread_position_in_grid.x;
    uint base = row * K;
    for (uint index = 0u; index < K; ++index) {
        raw_candidate_ids[base + index] = candidate_ids[base + index];
        raw_candidate_values[base + index] = candidate_values[base + index];
        raw_candidate_probs[base + index] = candidate_probs[base + index];
    }
    uint prepared_ids[K];
    ulong prepared_probs[K];
    uint prepared_count;
    prepare_candidate_row(
        candidate_ids,
        candidate_values,
        candidate_probs,
        base,
        prepared_ids,
        prepared_probs,
        prepared_count
    );
    selected_tokens[row] = sample_prepared_row(
        prepared_ids,
        prepared_probs,
        prepared_count,
        uniform_bits[row]
    );
"""


METAL_CONTROLLER_SOURCE = METAL_HELPERS + "\n" + METAL_BODY
METAL_SOURCE = METAL_SOFTFLOAT_SOURCE + "\n" + METAL_CONTROLLER_SOURCE


@lru_cache(maxsize=1)
def _metal_kernel():
    mx = importlib.import_module("mlx.core")
    return mx.fast.metal_kernel(
        name="mtplx_pr391_softfloat64_verifier_decision_d3_k20",
        input_names=[
            "draft_tokens",
            "draft_ids",
            "draft_values",
            "draft_probs",
            "target_ids",
            "target_values",
            "target_probs",
            "uniform_bits",
            "stop_ids",
            "stop_count",
            "bonus_allowed",
        ],
        output_names=[
            "accepted_count",
            "first_reject",
            "selected_token",
            "selected_kind",
            "selected_present",
            "draws_used",
            "accept_probability_bits",
        ],
        header=METAL_SOFTFLOAT_SOURCE + "\n" + METAL_HELPERS,
        source=METAL_BODY,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def bind_pr391_softfloat64_verifier_decision() -> Callable[..., tuple[object, ...]]:
    """Bind the exact fixed-D3/M4 controller before request execution."""

    mx = importlib.import_module("mlx.core")
    kernel = _metal_kernel()

    def apply(
        draft_tokens: object,
        draft_ids: object,
        draft_values: object,
        draft_probs: object,
        target_ids: object,
        target_values: object,
        target_probs: object,
        uniform_bits: object,
        stop_ids: object,
        stop_count: object,
        bonus_allowed: object,
    ) -> tuple[object, ...]:
        return tuple(
            kernel(
                inputs=[
                    draft_tokens,
                    draft_ids,
                    draft_values,
                    draft_probs,
                    target_ids,
                    target_values,
                    target_probs,
                    uniform_bits,
                    stop_ids,
                    stop_count,
                    bonus_allowed,
                ],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[(1,), (1,), (1,), (1,), (1,), (1,), (DEPTH,)],
                output_dtypes=[
                    mx.uint32,
                    mx.int32,
                    mx.uint32,
                    mx.uint32,
                    mx.uint32,
                    mx.uint32,
                    mx.uint64,
                ],
            )
        )

    return apply


@lru_cache(maxsize=1)
def _metal_selector_kernel():
    mx = importlib.import_module("mlx.core")
    return mx.fast.metal_kernel(
        name="mtplx_pr391_softfloat64_candidate_selector_k20",
        input_names=[
            "candidate_ids",
            "candidate_values",
            "candidate_probs",
            "uniform_bits",
        ],
        output_names=[
            "selected_tokens",
            "raw_candidate_ids",
            "raw_candidate_values",
            "raw_candidate_probs",
        ],
        header=METAL_SOFTFLOAT_SOURCE + "\n" + METAL_HELPERS,
        source=METAL_SELECTOR_BODY,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def bind_pr391_softfloat64_candidate_selector() -> Callable[..., tuple[object, ...]]:
    """Bind exact K20/top-p=.95 choice with raw-candidate passthrough."""

    mx = importlib.import_module("mlx.core")
    kernel = _metal_selector_kernel()

    def apply(
        candidate_ids: object,
        candidate_values: object,
        candidate_probs: object,
        uniform_bits: object,
    ) -> tuple[object, ...]:
        rows = int(candidate_ids.shape[0])
        return tuple(
            kernel(
                inputs=[
                    candidate_ids,
                    candidate_values,
                    candidate_probs,
                    uniform_bits,
                ],
                grid=(rows, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[
                    (rows,),
                    (rows, K20),
                    (rows, K20),
                    (rows, K20),
                ],
                output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.float32],
            )
        )

    return apply


__all__ = [
    "DEPTH",
    "K20",
    "METAL_SOFTFLOAT_COMMIT",
    "METAL_SOFTFLOAT_VERSION",
    "METAL_SOURCE",
    "METAL_CONTROLLER_SOURCE",
    "SELECTED_BONUS",
    "SELECTED_CORRECTION",
    "SELECTED_NONE",
    "bind_pr391_softfloat64_candidate_selector",
    "bind_pr391_softfloat64_verifier_decision",
    "reference_pr391_softfloat64_verifier_decision",
    "reference_select_candidate_row",
]
