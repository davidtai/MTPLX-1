"""Behavior contract for the Hy3 fixed-K3 / Rows4 target verifier.

This module deliberately contains no MLX kernels.  It is the narrow ownership
boundary between the four-row target sweep owned by Issue #63, the authoritative
router owned by Issues #59/#58, the streamed expert wave owned by Issue #65,
and the accept/commit consumer owned by Issue #64.

The contract makes the critical performance invariant explicit: one current
token plus three drafts enter one authoritative target capture, and acceptance
builds a commit request from that capture without invoking the target again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


HY3_FIXED_K3_DRAFT_ROWS = 3
HY3_FIXED_K3_TARGET_ROWS = HY3_FIXED_K3_DRAFT_ROWS + 1
HY3_FIXED_K3_TOP_K = 8
HY3_FIXED_K3_HIDDEN_SIZE = 4096
HY3_FIXED_K3_ASSIGNMENTS = HY3_FIXED_K3_TARGET_ROWS * HY3_FIXED_K3_TOP_K


class FixedK3ContractError(ValueError):
    """Raised before dispatch when a value leaves the fixed-K3 contract."""


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise FixedK3ContractError(f"{label} must expose a shape")
    try:
        shape = tuple(int(dimension) for dimension in raw)
    except (TypeError, ValueError) as exc:
        raise FixedK3ContractError(f"{label} has a non-integral shape") from exc
    if any(dimension <= 0 for dimension in shape):
        raise FixedK3ContractError(f"{label} dimensions must be positive")
    return shape


def _rows4_top8_shape(value: Any, *, label: str) -> tuple[int, ...]:
    shape = _shape(value, label=label)
    if len(shape) < 2 or shape[-2] != HY3_FIXED_K3_TARGET_ROWS:
        raise FixedK3ContractError(
            f"Rows4 router {label} must end in [{HY3_FIXED_K3_TARGET_ROWS}, 8]"
        )
    if shape[-1] != HY3_FIXED_K3_TOP_K:
        raise FixedK3ContractError(f"Rows4 router {label} must preserve top-8")
    return shape


def _rows4_hidden_shape(value: Any, *, label: str) -> tuple[int, ...]:
    shape = _shape(value, label=label)
    if (
        len(shape) < 2
        or shape[-2] != HY3_FIXED_K3_TARGET_ROWS
        or shape[-1] != HY3_FIXED_K3_HIDDEN_SIZE
    ):
        raise FixedK3ContractError(
            f"{label} must end in [4, {HY3_FIXED_K3_HIDDEN_SIZE}] hidden rows"
        )
    return shape


@dataclass(frozen=True, slots=True)
class Rows4RouterOutput:
    """Authoritative M4 top-8 routes from the #59/#58 router seam.

    ``dispatch_count`` is part of the contract because #58 owns the tagged
    last-arrival projection-to-finalizer handoff.  A two-dispatch projection
    plus finalizer remains a benchmark/control, not the fixed-K3 fused seam.
    """

    expert_ids: Any
    route_weights: Any
    dispatch_count: int = 1
    _batch_shape: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ids_shape = _rows4_top8_shape(self.expert_ids, label="expert IDs")
        weights_shape = _rows4_top8_shape(
            self.route_weights,
            label="route weights",
        )
        if ids_shape[:-2] != weights_shape[:-2]:
            raise FixedK3ContractError(
                "router expert IDs and route weights must have equal batch prefixes"
            )
        if int(self.dispatch_count) != 1:
            raise FixedK3ContractError(
                "the #58 Rows4 router seam requires exactly one dispatch"
            )
        object.__setattr__(self, "_batch_shape", ids_shape[:-2])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def rows(self) -> int:
        return HY3_FIXED_K3_TARGET_ROWS

    @property
    def top_k(self) -> int:
        return HY3_FIXED_K3_TOP_K

    @property
    def assignment_count(self) -> int:
        return HY3_FIXED_K3_ASSIGNMENTS


@dataclass(frozen=True, slots=True)
class Rows4ExpertWaveRequest:
    """The M4 x top-8 routed work presented to Issue #65's expert wave."""

    hidden_rows: Any
    routes: Rows4RouterOutput
    _batch_shape: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        hidden_shape = _rows4_hidden_shape(
            self.hidden_rows,
            label="expert-wave hidden rows",
        )
        batch_shape = hidden_shape[:-2]
        if batch_shape != self.routes.batch_shape:
            raise FixedK3ContractError(
                "expert-wave hidden rows and routes must have equal batch prefixes"
            )
        object.__setattr__(self, "_batch_shape", batch_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def assignment_count(self) -> int:
        return self.routes.assignment_count


@dataclass(frozen=True, slots=True)
class Rows4ExpertWaveOutput:
    """Route-reduced M4 hidden rows plus optional opaque stage evidence.

    The intermediate gate/up/SwiGLU tensor is intentionally not required: the
    promoted #65 implementation may keep it entirely inside a fused/tightly
    pipelined device path.  ``stage_capture`` is an optional diagnostic handle.
    """

    hidden_rows: Any
    stage_capture: Any = None
    _batch_shape: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        hidden_shape = _rows4_hidden_shape(
            self.hidden_rows,
            label="expert-wave output hidden rows",
        )
        batch_shape = hidden_shape[:-2]
        if batch_shape != (1,):
            raise FixedK3ContractError(
                "fixed-K3 expert output batch prefixes must be exactly [1]"
            )
        object.__setattr__(self, "_batch_shape", batch_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def rows(self) -> int:
        return HY3_FIXED_K3_TARGET_ROWS

    @property
    def hidden_size(self) -> int:
        return HY3_FIXED_K3_HIDDEN_SIZE


@runtime_checkable
class Rows4Router(Protocol):
    """Callable protocol that #59/#58 can satisfy without importing #63."""

    def __call__(self, hidden_rows: Any) -> Rows4RouterOutput: ...


@runtime_checkable
class Rows4ExpertWave(Protocol):
    """Callable protocol for the complete #65 gate/up/SiLU/down/reduce wave."""

    def __call__(self, request: Rows4ExpertWaveRequest) -> Rows4ExpertWaveOutput: ...


@dataclass(frozen=True, slots=True)
class FixedK3LayerCapture:
    """Opaque-but-shaped routing and expert evidence for one sparse layer."""

    layer_index: int
    routes: Rows4RouterOutput
    expert_wave: Rows4ExpertWaveOutput

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) or int(self.layer_index) < 0:
            raise FixedK3ContractError("layer_index must be a non-negative integer")
        if self.routes.batch_shape != self.expert_wave.batch_shape:
            raise FixedK3ContractError(
                "layer routes and expert output must have equal batch prefixes"
            )


@dataclass(frozen=True, slots=True)
class FixedK3SweepPayload:
    """All authoritative artifacts produced by one captured M4 target sweep."""

    logits: Any
    hidden_rows: Any
    cache_capture: Any
    layer_captures: tuple[FixedK3LayerCapture, ...] = ()

    def __post_init__(self) -> None:
        logits_shape = _shape(self.logits, label="fixed-K3 logits")
        if (
            len(logits_shape) != 3
            or logits_shape[0] != 1
            or logits_shape[1] != HY3_FIXED_K3_TARGET_ROWS
        ):
            raise FixedK3ContractError(
                "fixed-K3 logits must have shape [1, 4, vocabulary]"
            )
        hidden_shape = _rows4_hidden_shape(
            self.hidden_rows,
            label="fixed-K3 hidden rows",
        )
        if hidden_shape[:-2] != (1,):
            raise FixedK3ContractError(
                "fixed-K3 hidden rows must have batch prefix [1]"
            )
        if self.cache_capture is None:
            raise FixedK3ContractError(
                "fixed-K3 target capture must retain cache capture state"
            )
        captures = tuple(self.layer_captures)
        layer_indices = tuple(capture.layer_index for capture in captures)
        if len(layer_indices) != len(set(layer_indices)):
            raise FixedK3ContractError("layer capture indices must be unique")
        object.__setattr__(self, "layer_captures", captures)


@dataclass(frozen=True, slots=True)
class FixedK3CommitRequest:
    """One-shot capture view handed to #64; it performs no target work."""

    logits: Any
    hidden_rows: Any
    cache_capture: Any
    layer_captures: tuple[FixedK3LayerCapture, ...]
    accepted_drafts: int
    correction_token: int | None
    keep_tokens: int
    selected_row: int
    verified_tokens: int = HY3_FIXED_K3_TARGET_ROWS


@dataclass(slots=True)
class CapturedFixedK3TargetSweep:
    """One authoritative M4 capture awaiting exactly one #64 decision."""

    payload: FixedK3SweepPayload
    target_forward_calls: int = 1
    _consumed: bool = field(default=False, init=False, repr=False)

    @property
    def consumed(self) -> bool:
        return self._consumed

    def build_commit_request(
        self,
        *,
        accepted_drafts: int,
        correction_token: int | None,
    ) -> FixedK3CommitRequest:
        """Consume the capture once without calling an ordinary target forward."""

        if self._consumed:
            raise FixedK3ContractError("fixed-K3 target capture was already consumed")
        if isinstance(accepted_drafts, bool) or not isinstance(accepted_drafts, int):
            raise FixedK3ContractError("accepted_drafts must be an integer")
        if not 0 <= accepted_drafts <= HY3_FIXED_K3_DRAFT_ROWS:
            raise FixedK3ContractError("accepted_drafts must be within [0, 3]")
        accepted_all = accepted_drafts == HY3_FIXED_K3_DRAFT_ROWS
        if not accepted_all and correction_token is None:
            raise FixedK3ContractError(
                "a rejected fixed-K3 prefix requires a correction token"
            )
        if accepted_all and correction_token is not None:
            raise FixedK3ContractError(
                "an accept-all fixed-K3 prefix cannot carry a correction token"
            )

        keep_tokens = 1 + accepted_drafts
        request = FixedK3CommitRequest(
            logits=self.payload.logits,
            hidden_rows=self.payload.hidden_rows,
            cache_capture=self.payload.cache_capture,
            layer_captures=self.payload.layer_captures,
            accepted_drafts=accepted_drafts,
            correction_token=correction_token,
            keep_tokens=keep_tokens,
            selected_row=keep_tokens - 1,
        )
        self._consumed = True
        return request


TargetCapture = Callable[[Any, Any], FixedK3SweepPayload]


def capture_fixed_k3_target_sweep(
    input_ids: Any,
    *,
    cache: Any,
    target_capture: TargetCapture,
) -> CapturedFixedK3TargetSweep:
    """Invoke the authoritative target capture exactly once for ``[1, 4]`` IDs.

    The callable is intentionally a capture-only seam.  #64 receives the
    returned object and can select an accepted prefix/correction row, but the
    returned object's commit path has no runtime/forward callback to invoke.
    """

    input_shape = _shape(input_ids, label="fixed-K3 input ids")
    if input_shape != (1, HY3_FIXED_K3_TARGET_ROWS):
        raise FixedK3ContractError("fixed-K3 input ids must have shape [1, 4]")
    payload = target_capture(input_ids, cache)
    if not isinstance(payload, FixedK3SweepPayload):
        raise FixedK3ContractError(
            "fixed-K3 target capture must return FixedK3SweepPayload"
        )
    return CapturedFixedK3TargetSweep(payload=payload)
