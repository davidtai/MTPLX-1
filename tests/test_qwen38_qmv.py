from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

import mtplx.qwen38_qmv as qmv
from mtplx.qwen38_qmv import qwen38_qmv_active_input_groups


def test_active_input_groups_match_rows_78_and_80_width_table() -> None:
    assert {
        width: qwen38_qmv_active_input_groups(width)
        for width in range(2, 10)
    } == {2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}


def test_row70_routes_real_mlx_arrays_without_a_strides_attribute(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    contiguous_inputs: list[object] = []
    original_contiguous = mx.contiguous

    def contiguous(value):
        contiguous_inputs.append(value)
        return original_contiguous(value)

    def replica(**kwargs):
        calls.append(kwargs)
        return (mx.zeros((3, 4096), dtype=mx.bfloat16),)

    monkeypatch.setattr(qmv, "_kernels", lambda: (replica, None, None))
    monkeypatch.setattr(mx, "contiguous", contiguous)
    linear = SimpleNamespace(
        _mtplx_qwen38_qmv_active=True,
        _mtplx_qwen38_qmv_min_width=3,
        _mtplx_qwen38_qmv_active_groups=False,
        bits=4,
        group_size=64,
        mode="affine",
        bias=None,
        weight=mx.zeros((4096, 64), dtype=mx.uint32),
        scales=mx.zeros((4096, 8), dtype=mx.bfloat16),
        biases=mx.zeros((4096, 8), dtype=mx.bfloat16),
    )

    result = qmv.qwen38_qmv(
        linear,
        mx.zeros((3, 512), dtype=mx.bfloat16),
    )

    assert result is not None
    assert result.shape == (3, 4096)
    assert len(calls) == 1
    assert len(contiguous_inputs) == 4


def test_dflash_width_filter_keeps_unselected_physical_blocks_stock() -> None:
    linear = SimpleNamespace(
        _mtplx_qwen38_qmv_active=True,
        _mtplx_qwen38_qmv_min_width=2,
        _mtplx_qwen38_qmv_allowed_widths=(6,),
        _mtplx_qwen38_qmv_active_groups=False,
        bits=4,
        group_size=64,
        mode="affine",
        bias=None,
        weight=mx.zeros((4096, 64), dtype=mx.uint32),
        scales=mx.zeros((4096, 8), dtype=mx.bfloat16),
        biases=mx.zeros((4096, 8), dtype=mx.bfloat16),
    )

    assert qmv.qwen38_qmv(
        linear,
        mx.zeros((7, 512), dtype=mx.bfloat16),
    ) is None


def test_dflash_config_marks_only_q4_group64_draft_linears() -> None:
    eligible = nn.QuantizedLinear(
        512,
        4096,
        bias=False,
        group_size=64,
        bits=4,
    )
    wrong_group = nn.QuantizedLinear(
        512,
        4096,
        bias=False,
        group_size=32,
        bits=4,
    )
    draft_model = SimpleNamespace(modules=lambda: [eligible, wrong_group])

    report = qmv.configure_qwen38_dflash_qmv(
        draft_model,
        active=True,
        allowed_widths=(6,),
    )

    assert report == {
        "eligible_modules": 1,
        "active_modules": 1,
        "allowed_widths": [6],
    }
    assert eligible._mtplx_qwen38_qmv_active is True
    assert eligible._mtplx_qwen38_qmv_allowed_widths == (6,)
    assert eligible._mtplx_qwen38_qmv_use_table is False
    assert eligible._mtplx_qwen38_qmv_min_output_size == 0
    assert wrong_group._mtplx_qwen38_qmv_active is False


def test_dflash_config_routes_verify_linear_private_dispatch(monkeypatch) -> None:
    class DraftVerifyLinear(nn.QuantizedLinear):
        def __init__(self) -> None:
            super().__init__(
                512,
                4096,
                bias=False,
                group_size=64,
                bits=4,
            )
            object.__setattr__(self, "_call_fn", lambda _x: "stock")

        def __call__(self, x):
            return self._call_fn(x)

    routed = object()
    monkeypatch.setattr(
        qmv,
        "qwen38_qmv",
        lambda linear, _x: routed
        if linear._mtplx_qwen38_qmv_active
        else None,
    )
    linear = DraftVerifyLinear()
    draft_model = SimpleNamespace(modules=lambda: [linear])

    qmv.configure_qwen38_dflash_qmv(
        draft_model,
        active=True,
        allowed_widths=(6,),
    )

    assert linear(mx.zeros((1, 6, 512), dtype=mx.bfloat16)) is routed


def test_dflash_direct_nibble_route_does_not_add_row70_sum_table(monkeypatch) -> None:
    calls = []

    def replica(**kwargs):
        calls.append("replica")
        return (mx.zeros((6, 4096), dtype=mx.bfloat16),)

    def table(**kwargs):
        calls.append("table")
        return (mx.zeros((6, 4096), dtype=mx.bfloat16),)

    monkeypatch.setattr(qmv, "_kernels", lambda: (replica, table, object()))
    linear = SimpleNamespace(
        _mtplx_qwen38_qmv_active=True,
        _mtplx_qwen38_qmv_min_width=6,
        _mtplx_qwen38_qmv_allowed_widths=(6,),
        _mtplx_qwen38_qmv_active_groups=True,
        _mtplx_qwen38_qmv_use_table=False,
        bits=4,
        group_size=64,
        mode="affine",
        bias=None,
        weight=mx.zeros((4096, 64), dtype=mx.uint32),
        scales=mx.zeros((4096, 8), dtype=mx.bfloat16),
        biases=mx.zeros((4096, 8), dtype=mx.bfloat16),
    )

    assert qmv.qwen38_qmv(
        linear,
        mx.zeros((6, 512), dtype=mx.bfloat16),
    ) is not None
    assert calls == ["replica"]
