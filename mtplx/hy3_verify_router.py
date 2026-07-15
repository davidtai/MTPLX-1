"""Fixed-row compiled router seam for streamed Hy3 verification.

The streamed expert runtime must materialize router indices on the host before
it can make authoritative cache, slot-generation, SSD, pin, and fence
decisions.  An outer ``mx.compile`` therefore cannot legally contain a Hy3
sparse layer.  This module narrows compilation to the pure router array
function and returns to the unchanged streamed runtime at that host seam.

The experiment is disabled by default and is eligible only for a fixed-width
``decode_verify`` call.  ``parity`` mode double-runs the pure router and fails
closed on any route-order or route-weight difference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx

from .attention_context import current_attention_phase


_COMPILE_ENV = "MTPLX_HY3_VERIFY_ROUTER_COMPILE"
_ROWS_ENV = "MTPLX_HY3_VERIFY_ROUTER_ROWS"


def _fresh_stats() -> dict[str, Any]:
    return {
        "eligible_calls": 0,
        "compiled_calls": 0,
        "compiled_router_count": 0,
        "traces": 0,
        "initial_traces": 0,
        "retraces": 0,
        "failures": 0,
        "fallback_reasons": {},
        "parity_checks": 0,
        "parity_failures": 0,
        "max_route_weight_error": 0.0,
        "last_failure": None,
    }


_STATS = _fresh_stats()
_ROUTER_IDS: set[int] = set()


@dataclass
class _CompiledRouter:
    function: Callable[[mx.array], tuple[mx.array, mx.array]]
    host: dict[str, Any]


def _mode() -> str:
    raw = (os.environ.get(_COMPILE_ENV) or "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on"}:
        return "on"
    if raw == "parity":
        return raw
    raise ValueError(f"{_COMPILE_ENV} must be off, on, or parity")


def _target_rows() -> int:
    raw = (os.environ.get(_ROWS_ENV) or "4").strip()
    try:
        rows = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ROWS_ENV} must be an integer") from exc
    if not 2 <= rows <= 8:
        raise ValueError(f"{_ROWS_ENV} must be within [2, 8]")
    return rows


def _record_fallback(reason: str) -> None:
    reasons = _STATS["fallback_reasons"]
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _record_failure(exc: BaseException, *, phase: str) -> None:
    _STATS["failures"] = int(_STATS["failures"]) + 1
    _STATS["last_failure"] = {
        "phase": str(phase),
        "type": type(exc).__name__,
        "message": " ".join(str(exc).split())[:512],
    }


def reset_hy3_verify_router_stats() -> None:
    """Reset request-local evidence without discarding compiled functions."""

    global _STATS
    _STATS = _fresh_stats()
    _ROUTER_IDS.clear()


def hy3_verify_router_enabled() -> bool:
    """Return whether the opt-in seam is active (and validate its config)."""

    enabled = _mode() != "off"
    if enabled:
        _target_rows()
    return enabled


def hy3_verify_router_stats() -> dict[str, Any]:
    """Return a serialization-safe snapshot of the fixed-row router evidence."""

    result = dict(_STATS)
    result["fallback_reasons"] = dict(_STATS["fallback_reasons"])
    failure = _STATS["last_failure"]
    result["last_failure"] = dict(failure) if isinstance(failure, dict) else None
    result["mode"] = _mode()
    result["target_rows"] = _target_rows()
    return result


def _compiled_router(
    router: Any,
    stock_forward: Callable[[mx.array], tuple[mx.array, mx.array]],
    *,
    rows: int,
) -> _CompiledRouter:
    records = getattr(router, "_mtplx_verify_router_compiled", None)
    if not isinstance(records, dict):
        records = {}
        object.__setattr__(router, "_mtplx_verify_router_compiled", records)
    record = records.get(rows)
    if isinstance(record, _CompiledRouter):
        record.host["stats"] = _STATS
        return record

    host: dict[str, Any] = {"stats": _STATS, "trace_count": 0}

    def fixed_router(value):
        # This Python body runs only when MLX traces/retraces the fixed shape.
        stats = host["stats"]
        traces = int(host["trace_count"])
        stats["traces"] = int(stats["traces"]) + 1
        if traces:
            stats["retraces"] = int(stats["retraces"]) + 1
        else:
            stats["initial_traces"] = int(stats["initial_traces"]) + 1
        host["trace_count"] = traces + 1
        return stock_forward(value)

    record = _CompiledRouter(mx.compile(fixed_router), host)
    records[rows] = record
    return record


def maybe_compile_hy3_verify_router(
    router: Any,
    x: mx.array,
    stock_forward: Callable[[mx.array], tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array]:
    """Dispatch one pure Hy3 router through the fixed verification seam."""

    mode = _mode()
    if mode == "off":
        return stock_forward(x)
    phase = current_attention_phase()
    if phase != "decode_verify":
        _record_fallback(f"attention_phase:{phase}")
        return stock_forward(x)
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[0]) != 1:
        _record_fallback("shape")
        return stock_forward(x)
    rows = _target_rows()
    actual_rows = int(shape[-2])
    if actual_rows != rows:
        _record_fallback(f"rows:{actual_rows}")
        return stock_forward(x)

    _STATS["eligible_calls"] = int(_STATS["eligible_calls"]) + 1
    try:
        record = _compiled_router(router, stock_forward, rows=rows)
        record.host["stats"] = _STATS
        compiled_indices, compiled_weights = record.function(x)
        _ROUTER_IDS.add(id(router))
        _STATS["compiled_router_count"] = len(_ROUTER_IDS)
        _STATS["compiled_calls"] = int(_STATS["compiled_calls"]) + 1
        if mode != "parity":
            return compiled_indices, compiled_weights

        stock_indices, stock_weights = stock_forward(x)
        mx.eval(
            compiled_indices,
            compiled_weights,
            stock_indices,
            stock_weights,
        )
        _STATS["parity_checks"] = int(_STATS["parity_checks"]) + 1
        route_match = bool(mx.array_equal(compiled_indices, stock_indices).item())
        weight_match = bool(mx.array_equal(compiled_weights, stock_weights).item())
        max_error = float(
            mx.max(
                mx.abs(
                    compiled_weights.astype(mx.float32)
                    - stock_weights.astype(mx.float32)
                )
            ).item()
        )
        _STATS["max_route_weight_error"] = max(
            float(_STATS["max_route_weight_error"]),
            max_error,
        )
        if not route_match:
            raise RuntimeError("compiled Hy3 router route-order parity mismatch")
        if not weight_match:
            raise RuntimeError(
                "compiled Hy3 router route-weight parity mismatch "
                f"(max_error={max_error:.9g})"
            )
        return compiled_indices, compiled_weights
    except BaseException as exc:
        if mode == "parity":
            _STATS["parity_failures"] = int(_STATS["parity_failures"]) + 1
        _record_failure(exc, phase="router_dispatch")
        raise
