"""Benchmark-only fixed-D3/K20 float32 speculative verifier decision.

The installed callable is deliberately narrow: construction validates the
fixed experimental geometry, then the hot callable only dispatches. Tokens,
sparse distributions, uniform values, stop tokens, and bonus eligibility all
remain runtime inputs. Float32 decision drift is permitted only for the PR391
performance experiment; this module is not the exact production verifier.
"""

from __future__ import annotations

from functools import lru_cache
import importlib
from numbers import Integral
from typing import Callable

import numpy as np


DEPTH = 3
K20 = 20
SELECTED_NONE = 0
SELECTED_CORRECTION = 1
SELECTED_BONUS = 2


METAL_HEADER = """
    using namespace metal;
    constant constexpr uint D = 3;
    constant constexpr uint K = 20;
    constant constexpr uint UNION_CAPACITY = 2 * K;
    constant constexpr uint SELECTED_NONE = 0;
    constant constexpr uint SELECTED_CORRECTION = 1;
    constant constexpr uint SELECTED_BONUS = 2;

    inline float sparse_probability(
        device const uint* ids,
        device const float* probs,
        uint base,
        uint token
    ) {
        for (uint index = 0; index < K; ++index) {
            if (ids[base + index] == token) return probs[base + index];
        }
        return 0.0f;
    }

    inline uint sample_positive_in_order(
        thread uint* ids,
        thread float* weights,
        uint count,
        float uniform
    ) {
        uint positive_count = 0;
        float total = 0.0f;
        for (uint index = 0; index < count; ++index) {
            float weight = weights[index];
            if (isfinite(weight) && weight > 0.0f) {
                ids[positive_count] = ids[index];
                weights[positive_count] = weight;
                ++positive_count;
                total += weight;
            }
        }

        float cumulative = 0.0f;
        for (uint index = 0; index + 1 < positive_count; ++index) {
            cumulative += weights[index];
            // searchsorted(cdf, uniform, side="right"): equality advances.
            if (uniform < cumulative / total) return ids[index];
        }
        return ids[positive_count - 1];
    }
"""


METAL_BODY = """
    if (thread_position_in_grid.x != 0u) return;

    accepted_count[0] = 0u;
    first_reject[0] = -1;
    selected_token[0] = 0u;
    selected_kind[0] = SELECTED_NONE;
    selected_present[0] = 0u;
    draws_used[0] = 0u;
    for (uint depth = 0; depth < D; ++depth) accept_probs[depth] = 0.0f;

    for (uint depth = 0; depth < D; ++depth) {
        uint token = draft_tokens[depth];
        uint base = depth * K;
        float p_value = sparse_probability(target_ids, target_probs, base, token);
        float q_value = sparse_probability(draft_ids, draft_probs, base, token);
        float accept_probability = q_value <= 0.0f
            ? (p_value > 0.0f ? 1.0f : 0.0f)
            : min(1.0f, p_value / q_value);
        accept_probs[depth] = accept_probability;

        if (uniforms[depth] <= accept_probability) {
            accepted_count[0] = depth + 1u;
            bool stopped = false;
            for (uint stop_index = 0; stop_index < stop_count[0]; ++stop_index) {
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
        uint union_tokens[UNION_CAPACITY];
        for (uint index = 0; index < K; ++index) {
            union_tokens[index] = target_ids[base + index];
            union_tokens[K + index] = draft_ids[base + index];
        }
        for (uint index = 1; index < UNION_CAPACITY; ++index) {
            uint token_to_insert = union_tokens[index];
            uint insertion = index;
            while (
                insertion > 0 && union_tokens[insertion - 1] > token_to_insert
            ) {
                union_tokens[insertion] = union_tokens[insertion - 1];
                --insertion;
            }
            union_tokens[insertion] = token_to_insert;
        }

        uint residual_ids[UNION_CAPACITY];
        float residual_weights[UNION_CAPACITY];
        uint residual_count = 0u;
        uint previous = 0u;
        bool have_previous = false;
        for (uint index = 0; index < UNION_CAPACITY; ++index) {
            uint union_token = union_tokens[index];
            if (have_previous && union_token == previous) continue;
            previous = union_token;
            have_previous = true;
            float target_probability = sparse_probability(
                target_ids, target_probs, base, union_token
            );
            float draft_probability = sparse_probability(
                draft_ids, draft_probs, base, union_token
            );
            float weight = max(target_probability - draft_probability, 0.0f);
            if (isfinite(weight) && weight > 0.0f) {
                residual_ids[residual_count] = union_token;
                residual_weights[residual_count] = weight;
                ++residual_count;
            }
        }

        if (residual_count == 0u) {
            for (uint index = 0; index < K; ++index) {
                residual_ids[index] = target_ids[base + index];
                residual_weights[index] = target_probs[base + index];
            }
            residual_count = K;
        }
        selected_token[0] = sample_positive_in_order(
            residual_ids,
            residual_weights,
            residual_count,
            uniforms[depth + 1u]
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
    float bonus_weights[K];
    for (uint index = 0; index < K; ++index) {
        bonus_ids[index] = target_ids[3u * K + index];
        bonus_weights[index] = target_probs[3u * K + index];
    }
    selected_token[0] = sample_positive_in_order(
        bonus_ids, bonus_weights, K, uniforms[3]
    );
    selected_kind[0] = SELECTED_BONUS;
    selected_present[0] = 1u;
    draws_used[0] = 4u;
"""

# Combined source is public for CPU-only source audits. The binder supplies the
# declarations as ``header`` and the kernel body as ``source`` to MLX.
METAL_SOURCE = METAL_HEADER + "\n" + METAL_BODY


def validate_pr391_float32_verifier_contract(
    *, depth: int = DEPTH, top_k: int = K20
) -> None:
    """Validate fixed route geometry before constructing the hot callable."""

    if isinstance(depth, bool) or not isinstance(depth, Integral) or int(depth) != DEPTH:
        raise ValueError("PR391 float32 verifier decision requires depth=3")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or int(top_k) != K20:
        raise ValueError("PR391 float32 verifier decision requires top_k=20")


def _validate_reference_inputs(
    draft_tokens: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    uniforms: np.ndarray,
    stop_ids: np.ndarray,
    stop_count: int,
    bonus_allowed: bool,
) -> tuple[np.ndarray, ...]:
    tokens = np.asarray(draft_tokens)
    draft_support = np.asarray(draft_ids)
    draft_weights = np.asarray(draft_probs)
    target_support = np.asarray(target_ids)
    target_weights = np.asarray(target_probs)
    draws = np.asarray(uniforms)
    stops = np.asarray(stop_ids)

    if tokens.dtype != np.dtype(np.uint32):
        raise ValueError("draft_tokens must be uint32")
    if tokens.shape != (DEPTH,):
        raise ValueError("draft_tokens must have shape [3]")
    if draft_support.dtype != np.dtype(np.uint32):
        raise ValueError("draft_ids must be uint32")
    if draft_support.shape != (DEPTH, K20):
        raise ValueError("draft_ids must have shape [3, 20]")
    if draft_weights.dtype != np.dtype(np.float32):
        raise ValueError("draft_probs must be float32")
    if draft_weights.shape != (DEPTH, K20):
        raise ValueError("draft_probs must have shape [3, 20]")
    if target_support.dtype != np.dtype(np.uint32):
        raise ValueError("target_ids must be uint32")
    if target_support.shape != (DEPTH + 1, K20):
        raise ValueError("target_ids must have shape [4, 20]")
    if target_weights.dtype != np.dtype(np.float32):
        raise ValueError("target_probs must be float32")
    if target_weights.shape != (DEPTH + 1, K20):
        raise ValueError("target_probs must have shape [4, 20]")
    if draws.dtype != np.dtype(np.float32):
        raise ValueError("uniforms must be float32")
    if draws.shape != (DEPTH + 1,):
        raise ValueError("uniforms must have shape [4]")
    if not np.all(np.isfinite(draws)) or np.any(draws < np.float32(0.0)) or np.any(
        draws > np.float32(1.0)
    ):
        raise ValueError("uniforms must be finite and in [0, 1]")
    if stops.dtype != np.dtype(np.uint32):
        raise ValueError("stop_ids must be uint32")
    if stops.ndim != 1:
        raise ValueError("stop_ids must be one-dimensional")
    if isinstance(stop_count, bool) or not isinstance(stop_count, Integral):
        raise ValueError("stop_count must be an integer")
    bounded_stop_count = int(stop_count)
    if not 0 <= bounded_stop_count <= stops.size:
        raise ValueError("stop_count must be in [0, len(stop_ids)]")
    if not isinstance(bonus_allowed, (bool, np.bool_)):
        raise ValueError("bonus_allowed must be boolean")

    for support_name, support in (
        ("draft_ids", draft_support),
        ("target_ids", target_support),
    ):
        for row in support:
            if np.unique(row).size != K20:
                raise ValueError(f"{support_name} must be unique within each row")
    for weights in (draft_weights, target_weights):
        if not np.all(np.isfinite(weights)) or np.any(weights < np.float32(0.0)):
            raise ValueError("probabilities must be finite and non-negative")
        if np.any(np.sum(weights, axis=1, dtype=np.float32) <= np.float32(0.0)):
            raise ValueError("each probability row must have positive mass")

    return (
        tokens,
        draft_support,
        draft_weights,
        target_support,
        target_weights,
        draws,
        stops,
        np.int64(bounded_stop_count),
        np.bool_(bonus_allowed),
    )


def _lookup_probability(ids: np.ndarray, probs: np.ndarray, token: int) -> np.float32:
    for index in range(ids.size):
        if int(ids[index]) == token:
            return np.float32(probs[index])
    return np.float32(0.0)


def _sample_positive_in_order(
    ids: np.ndarray, probs: np.ndarray, uniform: np.float32
) -> np.uint32:
    positive = np.isfinite(probs) & (probs > np.float32(0.0))
    ordered_ids = ids[positive]
    ordered_probs = probs[positive]
    total = np.float32(0.0)
    for probability in ordered_probs:
        total = np.float32(total + np.float32(probability))
    cumulative = np.float32(0.0)
    for index in range(ordered_ids.size - 1):
        cumulative = np.float32(cumulative + np.float32(ordered_probs[index]))
        if uniform < np.float32(cumulative / total):
            return np.uint32(ordered_ids[index])
    return np.uint32(ordered_ids[-1])


def reference_pr391_float32_verifier_decision(
    draft_tokens: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    uniforms: np.ndarray,
    stop_ids: np.ndarray,
    *,
    stop_count: int,
    bonus_allowed: bool,
) -> tuple[np.ndarray, ...]:
    """Literal CPU float32 reference for construction tests and drift audits."""

    (
        tokens,
        draft_support,
        draft_weights,
        target_support,
        target_weights,
        draws,
        stops,
        bounded_stop_count,
        allow_bonus,
    ) = _validate_reference_inputs(
        draft_tokens,
        draft_ids,
        draft_probs,
        target_ids,
        target_probs,
        uniforms,
        stop_ids,
        stop_count,
        bonus_allowed,
    )

    accepted = np.zeros(1, dtype=np.uint32)
    first_reject = np.array([-1], dtype=np.int32)
    selected_token = np.zeros(1, dtype=np.uint32)
    selected_kind = np.zeros(1, dtype=np.uint32)
    selected_present = np.zeros(1, dtype=np.uint32)
    draws_used = np.zeros(1, dtype=np.uint32)
    accept_probs = np.zeros(DEPTH, dtype=np.float32)

    active_stops = {int(token) for token in stops[: int(bounded_stop_count)]}
    for depth in range(DEPTH):
        token = int(tokens[depth])
        p_value = _lookup_probability(
            target_support[depth], target_weights[depth], token
        )
        q_value = _lookup_probability(
            draft_support[depth], draft_weights[depth], token
        )
        if q_value <= np.float32(0.0):
            accept_probability = np.float32(
                1.0 if p_value > np.float32(0.0) else 0.0
            )
        else:
            accept_probability = np.minimum(
                np.float32(1.0), np.float32(p_value / q_value)
            )
        accept_probs[depth] = accept_probability
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
                    accept_probs,
                )
            continue

        first_reject[0] = np.int32(depth)
        union_ids = np.union1d(
            target_support[depth], draft_support[depth]
        ).astype(np.uint32, copy=False)
        residual = np.empty(union_ids.size, dtype=np.float32)
        for index, union_token in enumerate(union_ids):
            target_probability = _lookup_probability(
                target_support[depth], target_weights[depth], int(union_token)
            )
            draft_probability = _lookup_probability(
                draft_support[depth], draft_weights[depth], int(union_token)
            )
            residual[index] = np.maximum(
                np.float32(target_probability - draft_probability), np.float32(0.0)
            )
        if np.any(residual > np.float32(0.0)):
            correction = _sample_positive_in_order(
                union_ids, residual, draws[depth + 1]
            )
        else:
            correction = _sample_positive_in_order(
                target_support[depth], target_weights[depth], draws[depth + 1]
            )
        selected_token[0] = correction
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
            accept_probs,
        )

    accepted[0] = np.uint32(DEPTH)
    draws_used[0] = np.uint32(DEPTH)
    if bool(allow_bonus):
        selected_token[0] = _sample_positive_in_order(
            target_support[DEPTH], target_weights[DEPTH], draws[DEPTH]
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
        accept_probs,
    )


@lru_cache(maxsize=1)
def _metal_kernel():
    mx = importlib.import_module("mlx.core")
    return mx.fast.metal_kernel(
        name="mtplx_pr391_float32_verifier_decision_d3_k20",
        input_names=[
            "draft_tokens",
            "draft_ids",
            "draft_probs",
            "target_ids",
            "target_probs",
            "uniforms",
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
            "accept_probs",
        ],
        header=METAL_HEADER,
        source=METAL_BODY,
        ensure_row_contiguous=True,
    )


def bind_pr391_float32_verifier_decision(
    *, depth: int = DEPTH, top_k: int = K20
) -> Callable[..., tuple[object, ...]]:
    """Bind the fixed benchmark route; the returned callable only dispatches."""

    validate_pr391_float32_verifier_contract(depth=depth, top_k=top_k)
    mx = importlib.import_module("mlx.core")
    kernel = _metal_kernel()

    def apply(
        draft_tokens: object,
        draft_ids: object,
        draft_probs: object,
        target_ids: object,
        target_probs: object,
        uniforms: object,
        stop_ids: object,
        stop_count: object,
        bonus_allowed: object,
    ) -> tuple[object, ...]:
        return tuple(
            kernel(
                inputs=[
                    draft_tokens,
                    draft_ids,
                    draft_probs,
                    target_ids,
                    target_probs,
                    uniforms,
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
                    mx.float32,
                ],
            )
        )

    return apply


__all__ = [
    "DEPTH",
    "K20",
    "METAL_SOURCE",
    "SELECTED_BONUS",
    "SELECTED_CORRECTION",
    "SELECTED_NONE",
    "bind_pr391_float32_verifier_decision",
    "reference_pr391_float32_verifier_decision",
    "validate_pr391_float32_verifier_contract",
]
