"""Admission and ownership contract for the opt-in Hy3 Q4 128K lane."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


HY3_Q4_CONTEXT_WINDOW = 131_072


@dataclass(frozen=True)
class ContextAdmission:
    model_context_limit_tokens: int
    rendered_input_tokens: int
    requested_output_tokens: int
    admitted_total_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "model_context_limit_tokens": self.model_context_limit_tokens,
            "rendered_input_tokens": self.rendered_input_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "admitted_total_tokens": self.admitted_total_tokens,
        }


class ContextLimitExceeded(ValueError):
    """Raised when rendered input plus requested output exceeds the lane."""


def _exact_nonnegative_token_count(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def admit_hy3_q4_context(
    *,
    rendered_input_tokens: int,
    requested_output_tokens: int,
) -> ContextAdmission:
    """Admit an exact rendered-input plus requested-output token budget."""

    rendered = _exact_nonnegative_token_count(
        "rendered_input_tokens", rendered_input_tokens
    )
    requested = _exact_nonnegative_token_count(
        "requested_output_tokens", requested_output_tokens
    )
    total = rendered + requested
    if total > HY3_Q4_CONTEXT_WINDOW:
        raise ContextLimitExceeded(
            "Hy3 Q4 dynamic context request totals "
            f"{total} tokens ({rendered} rendered input + {requested} requested "
            f"output), exceeding the {HY3_Q4_CONTEXT_WINDOW}-token limit"
        )
    return ContextAdmission(
        model_context_limit_tokens=HY3_Q4_CONTEXT_WINDOW,
        rendered_input_tokens=rendered,
        requested_output_tokens=requested,
        admitted_total_tokens=total,
    )


class SequenceBusyError(RuntimeError):
    """Raised when the single physical Q4 sequence already has an owner."""


class SequenceLease:
    def __init__(
        self,
        gate: SingleSequenceGate,
        *,
        token: object,
        request_id: str,
    ) -> None:
        self._gate = gate
        self._token = token
        self.request_id = request_id
        self._release_lock = Lock()
        self._released = False

    def hold(self) -> SequenceLeaseHold:
        """Retain physical ownership for an active generation worker."""

        with self._release_lock:
            if self._released:
                raise RuntimeError("cannot hold a released sequence lease")
            return self._gate._hold(self._token, self.request_id)

    def release(self, reason: str) -> bool:
        """Release ownership once; subsequent terminal paths are no-ops."""

        del reason
        with self._release_lock:
            if self._released:
                return False
            released = self._gate._release(self._token)
            self._released = True
            return released


class SequenceLeaseHold:
    """Worker-side retention that drains independently of the HTTP owner."""

    def __init__(
        self,
        gate: SingleSequenceGate,
        *,
        owner_token: object,
        hold_token: object,
        request_id: str,
    ) -> None:
        self._gate = gate
        self._owner_token = owner_token
        self._hold_token = hold_token
        self.request_id = request_id
        self._release_lock = Lock()
        self._released = False

    def release(self, reason: str) -> bool:
        """Release this worker retention once."""

        del reason
        with self._release_lock:
            if self._released:
                return False
            released = self._gate._release_hold(
                self._owner_token,
                self._hold_token,
            )
            self._released = True
            return released


class SingleSequenceGate:
    """Fail-fast ownership gate for the lane's sole physical KV sequence."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_token: object | None = None
        self._active_owner_released = False
        self._active_hold_tokens: set[object] = set()
        self._active_request_id: str | None = None
        self._active_admission: ContextAdmission | None = None
        self._last_admission: ContextAdmission | None = None

    @property
    def active_request_id(self) -> str | None:
        with self._lock:
            return self._active_request_id

    def acquire(
        self,
        request_id: str,
        *,
        admission: ContextAdmission,
    ) -> SequenceLease:
        normalized_request_id = str(request_id or "unknown")
        with self._lock:
            if self._active_token is not None:
                raise SequenceBusyError(
                    "Hy3 Q4 dynamic context sequence is owned by "
                    f"{self._active_request_id}"
                )
            token = object()
            self._active_token = token
            self._active_owner_released = False
            self._active_hold_tokens.clear()
            self._active_request_id = normalized_request_id
            self._active_admission = admission
        return SequenceLease(
            self,
            token=token,
            request_id=normalized_request_id,
        )

    def _hold(self, token: object, request_id: str) -> SequenceLeaseHold:
        with self._lock:
            if token is not self._active_token or self._active_owner_released:
                raise RuntimeError("cannot hold an inactive sequence lease")
            hold_token = object()
            self._active_hold_tokens.add(hold_token)
        return SequenceLeaseHold(
            self,
            owner_token=token,
            hold_token=hold_token,
            request_id=request_id,
        )

    def _release(self, token: object) -> bool:
        with self._lock:
            if token is not self._active_token or self._active_owner_released:
                return False
            self._active_owner_released = True
            if not self._active_hold_tokens:
                self._finish_release_locked()
            return True

    def _release_hold(self, owner_token: object, hold_token: object) -> bool:
        with self._lock:
            if (
                owner_token is not self._active_token
                or hold_token not in self._active_hold_tokens
            ):
                return False
            self._active_hold_tokens.remove(hold_token)
            if self._active_owner_released and not self._active_hold_tokens:
                self._finish_release_locked()
            return True

    def _finish_release_locked(self) -> None:
        self._last_admission = self._active_admission
        self._active_token = None
        self._active_owner_released = False
        self._active_hold_tokens.clear()
        self._active_request_id = None
        self._active_admission = None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            active = self._active_admission
            last = self._last_admission
            selected = active or last
            return {
                "admission_state": (
                    "active"
                    if active is not None
                    else "last"
                    if last is not None
                    else "none"
                ),
                "active_request_id": self._active_request_id,
                "model_context_limit_tokens": HY3_Q4_CONTEXT_WINDOW,
                "rendered_input_tokens": (
                    selected.rendered_input_tokens if selected is not None else None
                ),
                "requested_output_tokens": (
                    selected.requested_output_tokens if selected is not None else None
                ),
                "admitted_total_tokens": (
                    selected.admitted_total_tokens if selected is not None else None
                ),
                "active_admission": active.to_dict() if active is not None else None,
                "last_admission": last.to_dict() if last is not None else None,
            }
