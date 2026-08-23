from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx import gdn_capture


def _quantized_linear(in_features: int, out_features: int) -> nn.QuantizedLinear:
    dense = nn.Linear(in_features, out_features, bias=False)
    dense.weight = mx.arange(
        out_features * in_features,
        dtype=mx.float32,
    ).reshape(out_features, in_features).astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(
        dense,
        group_size=32,
        bits=4,
        mode="affine",
    )


def _gdn() -> SimpleNamespace:
    return SimpleNamespace(
        in_proj_qkv=_quantized_linear(32, 64),
        in_proj_z=_quantized_linear(32, 32),
        in_proj_b=_quantized_linear(32, 8),
        in_proj_a=_quantized_linear(32, 8),
    )


@pytest.mark.parametrize("width", (1, 2))
def test_row8_four_way_projection_fusion_is_exact_at_source_widths(width: int) -> None:
    gdn = _gdn()
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(linear_attn=gdn)],
            )
        )
    )
    x = mx.arange(width * 32, dtype=mx.float32).reshape(1, width, 32).astype(
        mx.bfloat16
    )
    expected = tuple(
        projection(x)
        for projection in (
            gdn.in_proj_qkv,
            gdn.in_proj_z,
            gdn.in_proj_b,
            gdn.in_proj_a,
        )
    )

    report = gdn_capture.configure_qwen38_row8_gdn_projection_fusion(
        model,
        active=True,
    )
    actual = gdn_capture._gdn_input_projections(gdn, x)
    mx.eval(*expected, *actual)

    assert report == {"configured_modules": 1, "active_modules": 1, "max_width": 2}
    assert all(mx.array_equal(left, right).item() for left, right in zip(expected, actual))


def test_row8_projection_fusion_keeps_width_three_on_stock_calls(monkeypatch) -> None:
    calls: list[str] = []
    gdn = SimpleNamespace(
        _mtplx_gdn_projection_mode="all",
        _mtplx_gdn_projection_max_width=2,
        in_proj_qkv=lambda _x: calls.append("qkv") or "qkv",
        in_proj_z=lambda _x: calls.append("z") or "z",
        in_proj_b=lambda _x: calls.append("b") or "b",
        in_proj_a=lambda _x: calls.append("a") or "a",
    )
    monkeypatch.setattr(
        gdn_capture,
        "_fused_quantized_many",
        lambda *_args, **_kwargs: pytest.fail("width-three fusion must not run"),
    )

    result = gdn_capture._gdn_input_projections(
        gdn,
        mx.zeros((1, 3, 32), dtype=mx.bfloat16),
    )

    assert result == ("qkv", "z", "b", "a")
    assert calls == ["qkv", "z", "b", "a"]
