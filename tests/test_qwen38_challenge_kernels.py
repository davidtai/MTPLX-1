from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.qwen3_5 import MLP

from mtplx.qwen38_challenge_kernels import (
    configure_qwen38_row18_mlp_gate_up,
    qwen38_dual_rms_norm_concat,
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


def test_row18_packed_dense_gate_up_matches_stock_at_fixed_d3_width() -> None:
    mlp = MLP(64, 128)
    x = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
    expected = mlp(x)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
    )

    report = configure_qwen38_row18_mlp_gate_up(model, active=True)
    actual = mlp(x)
    mx.eval(expected, actual)

    assert report == {"eligible_modules": 1, "active_modules": 1}
    assert mx.array_equal(actual, expected).item()
