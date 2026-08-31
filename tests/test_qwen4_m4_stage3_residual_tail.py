from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest


class _ArraySpec:
    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype

    def reshape(self, *shape):
        return _ArraySpec(shape, self.dtype)


def _bf16(values):
    """Round float32 values to BF16 with round-to-nearest-even."""

    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _cpu_residual_tail(block_out, hyper, inject):
    block_out = _bf16(block_out)
    hyper = _bf16(hyper)
    inject = _bf16(inject)
    products = _bf16(block_out[:, None, :] * inject[:, :, None])
    return _bf16(hyper + products).reshape(hyper.shape[0], -1)


def _valid_layer():
    return SimpleNamespace(
        mlp=object(),
        mlp_hyper_connection=SimpleNamespace(
            hc_count=4,
            hidden_size=2560,
            hc_norm=SimpleNamespace(
                weight=_ArraySpec((4 * 2560,), mx.bfloat16),
            ),
            block_inject_weight=SimpleNamespace(
                weight=_ArraySpec((4, 4 * 2560), mx.bfloat16),
            ),
        ),
    )


def test_residual_source_preserves_combine_and_bf16_decoder_order() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import residual_source, source

    stock = source()
    fused = residual_source()

    reduction = "bfloat routed_products[TOP_K]"
    assert reduction in stock
    assert reduction in fused
    assert "hyper" not in stock
    assert "inject" not in stock
    assert "bfloat block_out = bfloat(" in fused
    assert (
        "bfloat product = bfloat(\n"
        "                float(block_out) * float(inject_value));"
    ) in fused
    assert (
        "output[hidden_index] = bfloat(\n"
        "                float(hyper[hidden_index]) + float(product));"
    ) in fused
    assert fused.index("bfloat product") < fused.index("output[hidden_index]")


def test_cpu_reference_detects_reassociated_product_and_maps_streams() -> None:
    block_out = np.array([[-3.0, 1.0]], dtype=np.float32)
    inject = np.array([[-2.915, 0.5, -1.0, 2.0]], dtype=np.float32)
    hyper = np.array(
        [
            [
                [-2.845, 10.0],
                [1.0, 20.0],
                [2.0, 30.0],
                [3.0, 40.0],
            ]
        ],
        dtype=np.float32,
    )

    actual = _cpu_residual_tail(block_out, hyper, inject)
    staged_first_value = np.float32(5.90625)
    assert actual.shape == (1, 8)
    assert actual[0, 0] == staged_first_value
    assert actual.reshape(1, 4, 2)[0, :, 1].tolist() == [
        7.0625,
        20.5,
        29.0,
        42.0,
    ]

    no_product_round = _bf16(
        _bf16(hyper[0, 0, 0])
        + _bf16(block_out[0, 0]) * _bf16(inject[0, 0])
    )
    assert no_product_round == np.float32(5.9375)
    assert actual[0, 0] != no_product_round


def test_residual_tail_geometry_reuses_one_thread_per_m4_hidden_value() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import (
        launch_geometry,
        residual_launch_geometry,
    )

    assert launch_geometry() == ((4 * 2560, 1, 1), (256, 1, 1))
    assert residual_launch_geometry() == launch_geometry()


def test_residual_binding_is_six_input_bf16_construction_callable(
    monkeypatch,
) -> None:
    from mtplx.kernels import qwen4_m4_stage3 as kernel_module

    definition = {}
    launch = {}

    def fake_metal_kernel(**kwargs):
        definition.update(kwargs)

        def run(**kwargs):
            launch.update(kwargs)
            return (
                _ArraySpec(
                    kwargs["output_shapes"][0],
                    kwargs["output_dtypes"][0],
                ),
            )

        return run

    monkeypatch.setattr(kernel_module, "_RESIDUAL_KERNEL", None)
    monkeypatch.setattr(kernel_module.mx.fast, "metal_kernel", fake_metal_kernel)
    residual_tail = kernel_module.bind_residual_tail()
    inputs = tuple(object() for _ in range(4)) + (
        _ArraySpec((1, 4, 4 * 2560), mx.bfloat16),
        _ArraySpec((1, 4, 4), mx.bfloat16),
    )

    output = residual_tail(*inputs)
    assert output.shape == inputs[4].shape
    assert output.dtype == mx.bfloat16
    assert definition["input_names"] == [
        "routed_down",
        "shared_down",
        "route_scores",
        "shared_factor",
        "hyper",
        "inject",
    ]
    assert definition["ensure_row_contiguous"] is True
    assert launch["inputs"] == list(inputs)
    assert launch["grid"] == (4 * 2560, 1, 1)
    assert launch["threadgroup"] == (256, 1, 1)
    assert launch["output_shapes"] == [(4, 4 * 2560)]
    assert launch["output_dtypes"] == [mx.bfloat16]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda layer: setattr(layer.mlp_hyper_connection, "hc_count", 2),
            "four hyper streams",
        ),
        (
            lambda layer: setattr(
                layer.mlp_hyper_connection.block_inject_weight.weight,
                "dtype",
                mx.float16,
            ),
            "BF16 inject ownership",
        ),
        (
            lambda layer: setattr(
                layer.mlp_hyper_connection.block_inject_weight.weight,
                "shape",
                (4, 2560),
            ),
            "BF16 inject ownership",
        ),
    ],
)
def test_residual_tail_construction_admission_rejects_wrong_hyper_contract(
    monkeypatch, mutate, match
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = _valid_layer()
    mutate(layer)
    monkeypatch.setattr(stage3_module, "_validate_block_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(stage3_module, "bind_residual_tail", lambda: object())

    with pytest.raises(ValueError, match=match):
        stage3_module.bind_qwen4_m4_residual_tail(layer, index=7)


def test_residual_tail_construction_admission_returns_prebound_callable(
    monkeypatch,
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = _valid_layer()
    candidate = object()
    seen = []
    monkeypatch.setattr(
        stage3_module,
        "_validate_block_contract",
        lambda block, *, index: seen.append((block, index)),
    )
    monkeypatch.setattr(stage3_module, "bind_residual_tail", lambda: candidate)

    assert stage3_module.bind_qwen4_m4_residual_tail(layer, index=7) is candidate
    assert seen == [(layer.mlp, 7)]
