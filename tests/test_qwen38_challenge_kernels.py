from __future__ import annotations

import mlx.core as mx

from mtplx.qwen38_challenge_kernels import (
    qwen38_dual_rms_norm_concat,
    qwen38_row9_paired_qmv_g32_m4,
)


def test_row9_paired_qmv_matches_stock_at_the_target_g32_m4_shape() -> None:
    mx.random.seed(9)
    dense = mx.random.normal((4096, 512)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(
        dense,
        group_size=32,
        bits=4,
        mode="affine",
    )
    x = mx.random.normal((4, 512)).astype(mx.bfloat16)
    expected = mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=32,
        bits=4,
        mode="affine",
    )

    actual = qwen38_row9_paired_qmv_g32_m4(x, weight, scales, biases)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=0.02, atol=1.0).item()


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
