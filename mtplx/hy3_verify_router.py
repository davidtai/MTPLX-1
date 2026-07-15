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
        "compiled_graph_count": 0,
        "shared_graph_calls": 0,
        "per_router_graph_calls": 0,
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
_GRAPH_IDS: set[int] = set()


@dataclass
class _CompiledRouter:
    function: Callable[..., tuple[mx.array, mx.array]]
    host: dict[str, Any]


@dataclass(frozen=True)
class Hy3VerifyRouterConfig:
    """Immutable model-load selector used by every routed-layer hot call."""

    mode: str
    target_rows: int

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


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
    _GRAPH_IDS.clear()


def hy3_verify_router_enabled() -> bool:
    """Return whether the opt-in seam is active (and validate its config)."""

    enabled = _mode() != "off"
    if enabled:
        _target_rows()
    return enabled


def hy3_verify_router_config() -> Hy3VerifyRouterConfig:
    """Read and validate the selector once while constructing the model."""

    mode = _mode()
    rows = _target_rows() if mode != "off" else 4
    return Hy3VerifyRouterConfig(mode=mode, target_rows=rows)


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
    shared_group: dict[object, object] | None,
    linear_weight: mx.array | None,
    expert_bias: mx.array | None,
) -> tuple[_CompiledRouter, tuple[mx.array, ...], bool]:
    if (
        shared_group is not None
        and linear_weight is not None
        and expert_bias is not None
    ):
        cached_shared = getattr(router, "_mtplx_verify_router_shared_record", None)
        if isinstance(cached_shared, _CompiledRouter):
            cached_shared.host["stats"] = _STATS
            return cached_shared, (linear_weight, expert_bias), True
        key = (
            "fp32-linear-router",
            rows,
            int(linear_weight.shape[0]),
            int(linear_weight.shape[1]),
            int(router.top_k),
            bool(router.route_norm),
            float(router.router_scaling_factor),
        )
        shared_record = shared_group.get(key)
        if isinstance(shared_record, _CompiledRouter):
            shared_record.host["stats"] = _STATS
            object.__setattr__(
                router,
                "_mtplx_verify_router_shared_record",
                shared_record,
            )
            return shared_record, (linear_weight, expert_bias), True

        host: dict[str, Any] = {"stats": _STATS, "trace_count": 0}
        top_k = int(router.top_k)
        route_norm = bool(router.route_norm)
        scaling = float(router.router_scaling_factor)

        def fixed_linear_router(value, weight, bias):
            # One architecture-specialized graph is shared by all 79 trunk
            # routers. Weights remain dynamic inputs, so each layer keeps its
            # own parameters without retracing an otherwise identical graph.
            stats = host["stats"]
            traces = int(host["trace_count"])
            stats["traces"] = int(stats["traces"]) + 1
            if traces:
                stats["retraces"] = int(stats["retraces"]) + 1
            else:
                stats["initial_traces"] = int(stats["initial_traces"]) + 1
            host["trace_count"] = traces + 1
            logits = (value.astype(mx.float32) @ weight.T).astype(mx.float32)
            scores = mx.sigmoid(logits)
            selection_scores = scores + bias.astype(mx.float32)
            indices = mx.argpartition(
                selection_scores,
                kth=-top_k,
                axis=-1,
            )[..., -top_k:]
            weights = mx.take_along_axis(scores, indices, axis=-1)
            if route_norm:
                weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
            return indices, weights * scaling

        record = _CompiledRouter(mx.compile(fixed_linear_router), host)
        shared_group[key] = record
        object.__setattr__(router, "_mtplx_verify_router_shared_record", record)
        return record, (linear_weight, expert_bias), True

    records = getattr(router, "_mtplx_verify_router_compiled", None)
    if not isinstance(records, dict):
        records = {}
        object.__setattr__(router, "_mtplx_verify_router_compiled", records)
    record = records.get(rows)
    if isinstance(record, _CompiledRouter):
        record.host["stats"] = _STATS
        return record, (), False

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
    return record, (), False


def maybe_compile_hy3_verify_router(
    router: Any,
    x: mx.array,
    stock_forward: Callable[[mx.array], tuple[mx.array, mx.array]],
    *,
    mode: str,
    rows: int,
    shared_group: dict[object, object] | None = None,
    linear_weight: mx.array | None = None,
    expert_bias: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Dispatch one pure Hy3 router through the fixed verification seam."""

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
    actual_rows = int(shape[-2])
    if actual_rows != rows:
        _record_fallback(f"rows:{actual_rows}")
        return stock_forward(x)

    _STATS["eligible_calls"] = int(_STATS["eligible_calls"]) + 1
    try:
        record, dynamic_inputs, shared = _compiled_router(
            router,
            stock_forward,
            rows=rows,
            shared_group=shared_group,
            linear_weight=linear_weight,
            expert_bias=expert_bias,
        )
        record.host["stats"] = _STATS
        compiled_indices, compiled_weights = record.function(x, *dynamic_inputs)
        _ROUTER_IDS.add(id(router))
        _GRAPH_IDS.add(id(record))
        _STATS["compiled_router_count"] = len(_ROUTER_IDS)
        _STATS["compiled_graph_count"] = len(_GRAPH_IDS)
        call_key = "shared_graph_calls" if shared else "per_router_graph_calls"
        _STATS[call_key] = int(_STATS[call_key]) + 1
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
