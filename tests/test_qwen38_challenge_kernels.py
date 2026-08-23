from __future__ import annotations

import mlx.core as mx

from mtplx.attention_context import attention_phase, model_forward_kind
from mtplx.qwen38_challenge_kernels import (
    _qmv_route_active,
    qwen38_active_input_groups,
    qwen38_affine4_qmv,
    qwen38_dual_rms_norm_concat,
)


def test_final_qmv_width_plan_has_only_live_input_groups() -> None:
    assert {
        width: qwen38_active_input_groups(width) for width in range(2, 10)
    } == {2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}


def test_qmv_is_limited_to_the_target_verify_forward() -> None:
    assert _qmv_route_active() is False
    with attention_phase("decode_verify"), model_forward_kind("target_verify"):
        assert _qmv_route_active() is True
    with attention_phase("prefill"), model_forward_kind("target_verify"):
        assert _qmv_route_active() is False


def test_final_qmv_matches_bf16_stock_for_group32_and_group64() -> None:
    mx.random.seed(7)
    dense = mx.random.normal((4096, 512)).astype(mx.bfloat16)
    for group_size in (32, 64):
        weight, scales, biases = mx.quantize(
            dense,
            group_size=group_size,
            bits=4,
            mode="affine",
        )
        for width in range(2, 10):
            x = mx.random.normal((width, 512)).astype(mx.bfloat16)
            expected = mx.quantized_matmul(
                x,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=group_size,
                bits=4,
                mode="affine",
            )
            actual = qwen38_affine4_qmv(
                x,
                weight,
                scales,
                biases,
                group_size=group_size,
            )
            mx.eval(expected, actual)
            assert mx.allclose(actual, expected, rtol=0.02, atol=1.0).item(), (
                f"group {group_size}, width {width}"
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
