"""Exact fixed-M4 state-window selection for Qwen4 verifier handoff.

The enabled device path is construction-bound to the production Qwen4 geometry.
It consumes the verifier's device ``accepted_count`` directly; callers must not
materialize that scalar on the host before invoking the bound graph.

The NumPy reference represents BF16 arrays as their raw uint16 storage.  Window
selection is bit-preserving, so no floating-point arithmetic is required to
establish the selector contract in CPU-only tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

QWEN4_M4_VERIFY_WIDTH = 4
QWEN4_M4_ACTIVATION_WIDTH = 10_240
QWEN4_M4_GDN_CONV_ROWS = 3
QWEN4_M4_PLE_CONV_ROWS = 9
QWEN4_M4_PLE_HISTORY_ROWS = 2
QWEN4_M4_TOTAL_LAYERS = 48
QWEN4_M4_GDN_LAYERS = 36
QWEN4_M4_QSA_LAYERS = 12
QWEN4_M4_GDN_CAPTURE_NAMES = ("qkv", "q", "k", "v", "a", "b")
QWEN4_M4_PLE_CAPTURE_NAMES = ("ple_conv_rows", "ple_ids")


class Qwen4M4StateHandoffContractError(ValueError):
    """The captured verifier state cannot support an exact device handoff."""


@dataclass(frozen=True, slots=True)
class Qwen4M4StateHandoff:
    keep: int
    gdn_conv: np.ndarray
    ple_conv: np.ndarray
    ple_history: np.ndarray
    selected_hidden: np.ndarray
    gdn_keep_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class Qwen4M4ProductionStateHandoffBinding:
    """Construction-validated fixed-M4 primitives for a graph-bank binding.

    ``select_windows`` preserves BF16 and integer state bits by selection.
    Graph-bank integration must still replay every GDN recurrent state under
    the returned keep mask and trim each QSA cache from the same device keep;
    neither state is representable by the window selector alone.
    """

    select_windows: Callable[..., tuple[Any, ...]]
    gdn_layer_indices: tuple[int, ...]
    qsa_layer_indices: tuple[int, ...]
    ple_layer_index: int
    gdn_recurrent_replay_required: bool = True
    qsa_device_trim_required: bool = True
    ple_exact_width_replay_required: bool = True


def bind_qwen4_m4_production_state_handoff(
    *,
    linear_layer_indices: tuple[int, ...],
    qsa_layer_indices: tuple[int, ...],
    ple_layer_index: int,
    capture_layout: tuple[tuple[int, tuple[str, ...]], ...],
) -> Qwen4M4ProductionStateHandoffBinding:
    """Validate the production capture contract and bind its selector once.

    The exact PLE contract requires rows entering ``PLELayer._short_conv``.
    A ``ple_hidden`` capture is not interchangeable with those rows.  Current
    callers must add ``ple_conv_rows`` before this binding can be installed;
    this function fails at construction rather than inventing that state.
    """

    linear = tuple(linear_layer_indices)
    qsa = tuple(qsa_layer_indices)
    if len(linear) != QWEN4_M4_GDN_LAYERS:
        raise Qwen4M4StateHandoffContractError(
            f"fixed-M4 production handoff requires {QWEN4_M4_GDN_LAYERS} "
            f"GDN layers; got {len(linear)}"
        )
    if len(qsa) != QWEN4_M4_QSA_LAYERS:
        raise Qwen4M4StateHandoffContractError(
            f"fixed-M4 production handoff requires {QWEN4_M4_QSA_LAYERS} "
            f"QSA layers; got {len(qsa)}"
        )
    expected = set(range(QWEN4_M4_TOTAL_LAYERS))
    if (
        len(set(linear)) != len(linear)
        or len(set(qsa)) != len(qsa)
        or set(linear) & set(qsa)
        or set(linear) | set(qsa) != expected
    ):
        raise Qwen4M4StateHandoffContractError(
            "GDN and QSA layers must be a disjoint partition of layers 0..47"
        )
    if ple_layer_index not in set(linear):
        raise Qwen4M4StateHandoffContractError("PLE layer must be GDN")

    capture_items = tuple(capture_layout)
    capture_by_layer = dict(capture_items)
    if len(capture_by_layer) != len(capture_items):
        raise Qwen4M4StateHandoffContractError(
            "capture layout contains duplicate layer indices"
        )
    for index in linear:
        names = set(capture_by_layer.get(index, ()))
        missing = tuple(
            name for name in QWEN4_M4_GDN_CAPTURE_NAMES if name not in names
        )
        if missing:
            raise Qwen4M4StateHandoffContractError(
                f"GDN layer {index} missing captures: {', '.join(missing)}"
            )

    ple_names = set(capture_by_layer[ple_layer_index])
    for name in QWEN4_M4_PLE_CAPTURE_NAMES:
        if name not in ple_names:
            raise Qwen4M4StateHandoffContractError(
                f"PLE layer {ple_layer_index} missing exact capture {name}"
            )
    return Qwen4M4ProductionStateHandoffBinding(
        select_windows=bind_qwen4_m4_state_handoff(),
        gdn_layer_indices=linear,
        qsa_layer_indices=qsa,
        ple_layer_index=ple_layer_index,
    )


def _require_array(
    name: str,
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    dtype_label: str | None = None,
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        label = dtype_label or expected.name
        raise TypeError(f"{name} must use {label}; got {value.dtype}")


def reference_qwen4_m4_state_handoff(
    *,
    accepted_count: np.ndarray,
    gdn_conv_pre: np.ndarray,
    gdn_qkv: np.ndarray,
    ple_conv_pre: np.ndarray,
    ple_conv_rows: np.ndarray,
    ple_history_pre: np.ndarray,
    verify_ids: np.ndarray,
    verify_hidden: np.ndarray,
) -> Qwen4M4StateHandoff:
    """Literal CPU oracle for the production fixed-M4 selector.

    Activation arrays use uint16 BF16 storage bits.  ``verify_ids`` are the
    existing target input IDs (int32); the committed PLE history remains int64.
    """

    width = QWEN4_M4_ACTIVATION_WIDTH
    _require_array(
        "accepted_count", accepted_count, shape=(1,), dtype=np.dtype(np.uint32)
    )
    _require_array(
        "gdn_conv_pre",
        gdn_conv_pre,
        shape=(1, QWEN4_M4_GDN_CONV_ROWS, width),
        dtype=np.dtype(np.uint16),
        dtype_label="BF16 storage (uint16)",
    )
    _require_array(
        "gdn_qkv",
        gdn_qkv,
        shape=(1, QWEN4_M4_VERIFY_WIDTH, width),
        dtype=np.dtype(np.uint16),
        dtype_label="BF16 storage (uint16)",
    )
    _require_array(
        "ple_conv_pre",
        ple_conv_pre,
        shape=(1, QWEN4_M4_PLE_CONV_ROWS, width),
        dtype=np.dtype(np.uint16),
        dtype_label="BF16 storage (uint16)",
    )
    _require_array(
        "ple_conv_rows",
        ple_conv_rows,
        shape=(1, QWEN4_M4_VERIFY_WIDTH, width),
        dtype=np.dtype(np.uint16),
        dtype_label="BF16 storage (uint16)",
    )
    _require_array(
        "ple_history_pre",
        ple_history_pre,
        shape=(1, QWEN4_M4_PLE_HISTORY_ROWS),
        dtype=np.dtype(np.int64),
    )
    _require_array(
        "verify_ids",
        verify_ids,
        shape=(1, QWEN4_M4_VERIFY_WIDTH),
        dtype=np.dtype(np.int32),
    )
    _require_array(
        "verify_hidden",
        verify_hidden,
        shape=(1, QWEN4_M4_VERIFY_WIDTH, width),
        dtype=np.dtype(np.uint16),
        dtype_label="BF16 storage (uint16)",
    )

    accepted = int(accepted_count[0])
    if accepted >= QWEN4_M4_VERIFY_WIDTH:
        raise ValueError(
            "accepted_count must be in [0, 3] for fixed M4; "
            f"got {accepted}"
        )
    keep = accepted + 1

    gdn_window = np.concatenate((gdn_conv_pre, gdn_qkv), axis=1)
    ple_window = np.concatenate((ple_conv_pre, ple_conv_rows), axis=1)
    history = np.concatenate(
        (ple_history_pre, verify_ids.astype(np.int64)), axis=1
    )
    return Qwen4M4StateHandoff(
        keep=keep,
        gdn_conv=gdn_window[:, keep : keep + QWEN4_M4_GDN_CONV_ROWS, :].copy(),
        ple_conv=ple_window[:, keep : keep + QWEN4_M4_PLE_CONV_ROWS, :].copy(),
        ple_history=history[
            :, keep : keep + QWEN4_M4_PLE_HISTORY_ROWS
        ].copy(),
        selected_hidden=verify_hidden[:, accepted : accepted + 1, :].copy(),
        gdn_keep_mask=(
            np.arange(QWEN4_M4_VERIFY_WIDTH, dtype=np.uint32)[None, :] < keep
        ),
    )


def bind_qwen4_m4_state_handoff() -> Callable[..., tuple[Any, ...]]:
    """Bind the branch-free MLX selector for the validated production route."""

    import mlx.core as mx

    def select(
        accepted_count,
        gdn_conv_pre,
        gdn_qkv,
        ple_conv_pre,
        ple_conv_rows,
        ple_history_pre,
        verify_ids,
        verify_hidden,
    ):
        accepted = accepted_count.reshape(-1)[0].astype(mx.int32)
        keep = accepted + 1
        gdn_indices = keep + mx.arange(QWEN4_M4_GDN_CONV_ROWS, dtype=mx.int32)
        ple_indices = keep + mx.arange(QWEN4_M4_PLE_CONV_ROWS, dtype=mx.int32)
        history_indices = keep + mx.arange(
            QWEN4_M4_PLE_HISTORY_ROWS, dtype=mx.int32
        )

        gdn_conv = mx.take(
            mx.concatenate((gdn_conv_pre, gdn_qkv), axis=1),
            gdn_indices,
            axis=1,
        )
        ple_conv = mx.take(
            mx.concatenate((ple_conv_pre, ple_conv_rows), axis=1),
            ple_indices,
            axis=1,
        )
        ple_history = mx.take(
            mx.concatenate(
                (ple_history_pre, verify_ids.astype(mx.int64)), axis=1
            ),
            history_indices,
            axis=1,
        )
        selected_hidden = mx.take(
            verify_hidden, accepted.reshape(1), axis=1
        )
        gdn_keep_mask = (
            mx.arange(QWEN4_M4_VERIFY_WIDTH, dtype=mx.int32)[None, :] < keep
        )
        return (
            gdn_conv,
            ple_conv,
            ple_history,
            selected_hidden,
            gdn_keep_mask,
        )

    return mx.compile(select)


def replay_qwen4_m4_gdn_state(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    gdn_keep_mask,
):
    """Replay the exact stock recurrence under the device keep mask.

    This is intentionally only a thin graph-building primitive.  The future
    graph-bank integration owns the 36 construction-validated layer bindings.
    """

    from mlx_lm.models.gated_delta import gated_delta_update

    _unused_y, selected_state = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        gdn_keep_mask,
        use_kernel=True,
    )
    return selected_state


__all__ = [
    "QWEN4_M4_ACTIVATION_WIDTH",
    "QWEN4_M4_GDN_CAPTURE_NAMES",
    "QWEN4_M4_GDN_CONV_ROWS",
    "QWEN4_M4_GDN_LAYERS",
    "QWEN4_M4_PLE_CAPTURE_NAMES",
    "QWEN4_M4_PLE_CONV_ROWS",
    "QWEN4_M4_PLE_HISTORY_ROWS",
    "QWEN4_M4_QSA_LAYERS",
    "QWEN4_M4_TOTAL_LAYERS",
    "QWEN4_M4_VERIFY_WIDTH",
    "Qwen4M4ProductionStateHandoffBinding",
    "Qwen4M4StateHandoff",
    "Qwen4M4StateHandoffContractError",
    "bind_qwen4_m4_production_state_handoff",
    "bind_qwen4_m4_state_handoff",
    "reference_qwen4_m4_state_handoff",
    "replay_qwen4_m4_gdn_state",
]
