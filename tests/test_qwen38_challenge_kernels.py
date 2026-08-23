from __future__ import annotations

import mlx.core as mx

from mtplx.qwen38_challenge_kernels import (
    qwen38_dual_rms_norm_concat,
    qwen38_qk_rms_rope,
)


def test_dual_rms_norm_concat_matches_two_stock_norms() -> None:
    a = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    b = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    a_weight = mx.random.normal((5120,)).astype(mx.bfloat16)
    b_weight = mx.random.normal((5120,)).astype(mx.bfloat16)
    expected = mx.concatenate(
        (
            mx.fast.rms_norm(a, a_weight, 1e-6),
            mx.fast.rms_norm(b, b_weight, 1e-6),
        ),
        axis=-1,
    )
    actual = qwen38_dual_rms_norm_concat(a, b, a_weight, b_weight, 1e-6)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


def test_qk_rms_rope_matches_stock_qwen38_partial_rope_at_fixed_d3_width() -> None:
    queries = mx.random.normal((1, 4, 24, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 4, 4, 256)).astype(mx.bfloat16)
    q_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    q_norm = mx.fast.rms_norm(queries, q_weight, 1e-6).transpose(0, 2, 1, 3)
    k_norm = mx.fast.rms_norm(keys, k_weight, 1e-6).transpose(0, 2, 1, 3)
    q_expected = mx.fast.rope(
        q_norm,
        64,
        traditional=False,
        base=10_000_000.0,
        scale=1.0,
        offset=37,
    )
    k_expected = mx.fast.rope(
        k_norm,
        64,
        traditional=False,
        base=10_000_000.0,
        scale=1.0,
        offset=37,
    )

    q_actual, k_actual = qwen38_qk_rms_rope(
        queries,
        keys,
        q_weight,
        k_weight,
        1e-6,
        37,
    )
    mx.eval(q_expected, k_expected, q_actual, k_actual)

    assert mx.array_equal(q_actual, q_expected).item()
    assert mx.array_equal(k_actual, k_expected).item()
