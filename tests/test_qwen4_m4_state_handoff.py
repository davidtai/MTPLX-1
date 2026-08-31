from __future__ import annotations

import numpy as np
import pytest

import mtplx.kernels.qwen4_m4_state_handoff as handoff
from mtplx.kernels.qwen4_m4_state_handoff import (
    QWEN4_M4_ACTIVATION_WIDTH,
    reference_qwen4_m4_state_handoff,
)


def _raw_bf16_rows(rows: int, *, base: int = 0) -> np.ndarray:
    values = np.arange(base, base + rows, dtype=np.uint16)
    return np.broadcast_to(
        values.reshape(1, rows, 1),
        (1, rows, QWEN4_M4_ACTIVATION_WIDTH),
    ).copy()


def _production_layout(
    *,
    ple_names: tuple[str, ...] = (
        "ple_conv_rows",
        "ple_ids",
    ),
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    gdn_names = ("qkv", "q", "k", "v", "a", "b")
    return tuple(
        (index, gdn_names + (ple_names if index == 0 else ()))
        for index in range(36)
    )


def test_production_binding_validates_layout_and_binds_selector_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = object()
    bind_calls = 0

    def fake_bind() -> object:
        nonlocal bind_calls
        bind_calls += 1
        return selector

    monkeypatch.setattr(handoff, "bind_qwen4_m4_state_handoff", fake_bind)

    binding = handoff.bind_qwen4_m4_production_state_handoff(
        linear_layer_indices=tuple(range(36)),
        qsa_layer_indices=tuple(range(36, 48)),
        ple_layer_index=0,
        capture_layout=_production_layout(),
    )

    assert binding.select_windows is selector
    assert binding.gdn_layer_indices == tuple(range(36))
    assert binding.qsa_layer_indices == tuple(range(36, 48))
    assert binding.ple_layer_index == 0
    assert binding.gdn_recurrent_replay_required is True
    assert binding.qsa_device_trim_required is True
    assert binding.ple_exact_width_replay_required is True
    assert bind_calls == 1


def test_production_binding_rejects_layer_hidden_as_ple_conv_capture() -> None:
    with pytest.raises(
        handoff.Qwen4M4StateHandoffContractError,
        match="PLE layer 0 missing exact capture ple_conv_rows",
    ):
        handoff.bind_qwen4_m4_production_state_handoff(
            linear_layer_indices=tuple(range(36)),
            qsa_layer_indices=tuple(range(36, 48)),
            ple_layer_index=0,
            capture_layout=_production_layout(
                ple_names=("ple_hidden", "ple_ids")
            ),
        )


def test_production_binding_rejects_incomplete_gdn_capture() -> None:
    layout = dict(_production_layout())
    layout[7] = tuple(name for name in layout[7] if name != "k")

    with pytest.raises(
        handoff.Qwen4M4StateHandoffContractError,
        match="GDN layer 7 missing captures: k",
    ):
        handoff.bind_qwen4_m4_production_state_handoff(
            linear_layer_indices=tuple(range(36)),
            qsa_layer_indices=tuple(range(36, 48)),
            ple_layer_index=0,
            capture_layout=tuple(layout.items()),
        )


@pytest.mark.parametrize(
    ("linear", "qsa", "ple", "match"),
    [
        (tuple(range(35)), tuple(range(35, 48)), 0, "36 GDN layers"),
        (tuple(range(36)), tuple(range(35, 47)), 0, "disjoint partition"),
        (tuple(range(36)), tuple(range(36, 48)), 47, "PLE layer must be GDN"),
    ],
)
def test_production_binding_rejects_non_production_topology(
    linear: tuple[int, ...],
    qsa: tuple[int, ...],
    ple: int,
    match: str,
) -> None:
    with pytest.raises(handoff.Qwen4M4StateHandoffContractError, match=match):
        handoff.bind_qwen4_m4_production_state_handoff(
            linear_layer_indices=linear,
            qsa_layer_indices=qsa,
            ple_layer_index=ple,
            capture_layout=_production_layout(),
        )


@pytest.mark.parametrize("accepted_count", range(4))
def test_reference_selects_fixed_m4_windows_from_device_keep(
    accepted_count: int,
) -> None:
    result = reference_qwen4_m4_state_handoff(
        accepted_count=np.asarray([accepted_count], dtype=np.uint32),
        gdn_conv_pre=_raw_bf16_rows(3, base=10),
        gdn_qkv=_raw_bf16_rows(4, base=20),
        ple_conv_pre=_raw_bf16_rows(9, base=30),
        ple_conv_rows=_raw_bf16_rows(4, base=50),
        ple_history_pre=np.asarray([[60, 61]], dtype=np.int64),
        verify_ids=np.asarray([[70, 71, 72, 73]], dtype=np.int32),
        verify_hidden=_raw_bf16_rows(4, base=80),
    )

    keep = accepted_count + 1
    expected_gdn = np.asarray([10, 11, 12, 20, 21, 22, 23], dtype=np.uint16)[
        keep : keep + 3
    ]
    expected_ple = np.asarray(
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 50, 51, 52, 53],
        dtype=np.uint16,
    )[keep : keep + 9]
    expected_history = np.asarray([60, 61, 70, 71, 72, 73], dtype=np.int64)[
        keep : keep + 2
    ]

    assert result.keep == keep
    np.testing.assert_array_equal(result.gdn_conv[0, :, 0], expected_gdn)
    np.testing.assert_array_equal(result.ple_conv[0, :, 0], expected_ple)
    np.testing.assert_array_equal(result.ple_history[0], expected_history)
    np.testing.assert_array_equal(
        result.selected_hidden[0, :, 0],
        np.asarray([80 + accepted_count], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        result.gdn_keep_mask,
        np.arange(4, dtype=np.uint32)[None, :] < keep,
    )


def test_reference_rejects_non_device_accepted_count_contract() -> None:
    kwargs = {
        "gdn_conv_pre": _raw_bf16_rows(3),
        "gdn_qkv": _raw_bf16_rows(4),
        "ple_conv_pre": _raw_bf16_rows(9),
        "ple_conv_rows": _raw_bf16_rows(4),
        "ple_history_pre": np.zeros((1, 2), dtype=np.int64),
        "verify_ids": np.zeros((1, 4), dtype=np.int32),
        "verify_hidden": _raw_bf16_rows(4),
    }

    with pytest.raises(ValueError, match="accepted_count"):
        reference_qwen4_m4_state_handoff(
            accepted_count=np.asarray([4], dtype=np.uint32), **kwargs
        )
    with pytest.raises(TypeError, match="uint32"):
        reference_qwen4_m4_state_handoff(
            accepted_count=np.asarray([0], dtype=np.int32), **kwargs
        )


def test_reference_rejects_wrong_fixed_shape_or_storage_dtype() -> None:
    kwargs = {
        "accepted_count": np.asarray([0], dtype=np.uint32),
        "gdn_conv_pre": _raw_bf16_rows(3),
        "gdn_qkv": _raw_bf16_rows(4),
        "ple_conv_pre": _raw_bf16_rows(9),
        "ple_conv_rows": _raw_bf16_rows(4),
        "ple_history_pre": np.zeros((1, 2), dtype=np.int64),
        "verify_ids": np.zeros((1, 4), dtype=np.int32),
        "verify_hidden": _raw_bf16_rows(4),
    }

    with pytest.raises(ValueError, match="gdn_qkv"):
        reference_qwen4_m4_state_handoff(
            **{**kwargs, "gdn_qkv": _raw_bf16_rows(3)}
        )
    with pytest.raises(TypeError, match="ple_history_pre"):
        reference_qwen4_m4_state_handoff(
            **{**kwargs, "ple_history_pre": np.zeros((1, 2), dtype=np.int32)}
        )
    with pytest.raises(TypeError, match="BF16 storage"):
        reference_qwen4_m4_state_handoff(
            **{**kwargs, "verify_hidden": np.zeros((1, 4, 10240), dtype=np.float32)}
        )
