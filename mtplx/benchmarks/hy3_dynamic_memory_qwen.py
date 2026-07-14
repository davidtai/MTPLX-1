"""JSON subprocess adapter for exact Qwen isolation around issue #46.

The state transitions intentionally reuse the hardened issue-30 Qwen guard.
Every command emits exactly one JSON object on stdout, and unload, restore, and
verify consume the captured state object on stdin.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.run_issue30_starvation_attribution import (
    EXCLUSIVE_LANE,
    EXPECTED_QWEN_MODELS,
    QwenState,
    _acquire_lane,
    _assert_qwen_state_restored,
    _capture_qwen,
    _restore_qwen,
    _stop_qwen,
)


_LANE_OWNER_NAME = "issue46-owner.json"
_OWNER_TOKEN_ENV = "MTPLX_ISSUE46_CAMPAIGN_OWNER_TOKEN"
_OWNER_PID_ENV = "MTPLX_ISSUE46_CAMPAIGN_PID"


class QwenIsolationError(RuntimeError):
    """Raised when subprocess input or observed Qwen state is ambiguous."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise QwenIsolationError(f"duplicate JSON state key {key!r}")
        result[key] = value
    return result


def _state_from_value(
    value: object,
    *,
    expected_models: tuple[str, ...],
) -> QwenState:
    if not isinstance(value, Mapping) or set(value) != {"loaded", "models"}:
        raise QwenIsolationError(
            "captured Qwen state must contain exactly loaded and models"
        )
    loaded = value["loaded"]
    models = value["models"]
    if not isinstance(loaded, bool):
        raise QwenIsolationError("captured Qwen loaded state must be boolean")
    if (
        isinstance(models, (str, bytes, bytearray))
        or not isinstance(models, Sequence)
        or any(not isinstance(model, str) or not model for model in models)
    ):
        raise QwenIsolationError("captured Qwen models must be an array of model IDs")
    normalized = tuple(models)
    if len(normalized) != len(set(normalized)):
        raise QwenIsolationError("captured Qwen models contain duplicates")
    if loaded and normalized != expected_models:
        raise QwenIsolationError(
            "captured loaded Qwen model list differs from --expected-model"
        )
    if not loaded and normalized:
        raise QwenIsolationError("captured unloaded Qwen state contains models")
    return QwenState(loaded=loaded, models=normalized)


def _read_state(*, expected_models: tuple[str, ...]) -> QwenState:
    try:
        value = json.loads(sys.stdin.read(), object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise QwenIsolationError(
            f"stdin is not one captured Qwen state: {exc}"
        ) from exc
    return _state_from_value(value, expected_models=expected_models)


def _state_payload(state: QwenState) -> dict[str, object]:
    return {"loaded": state.loaded, "models": list(state.models)}


def _assert_positive(value: float, *, field: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise QwenIsolationError(f"{field} must be positive")
    return value


def _campaign_owner_identity() -> dict[str, object]:
    token = os.environ.get(_OWNER_TOKEN_ENV, "")
    if re.fullmatch(r"[0-9a-fA-F]{64}", token) is None:
        raise QwenIsolationError(
            f"{_OWNER_TOKEN_ENV} must be one explicit 256-bit hexadecimal token"
        )
    raw_pid = os.environ.get(_OWNER_PID_ENV, "")
    if not raw_pid.isascii() or not raw_pid.isdecimal():
        raise QwenIsolationError(f"{_OWNER_PID_ENV} must be a positive process ID")
    campaign_pid = int(raw_pid)
    if campaign_pid <= 0 or campaign_pid == os.getpid():
        raise QwenIsolationError(f"{_OWNER_PID_ENV} must identify the parent campaign")
    completed = subprocess.run(
        ("ps", "-o", "uid=", "-p", str(campaign_pid)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    raw_uid = completed.stdout.strip()
    if completed.returncode != 0 or not raw_uid.isdecimal():
        raise QwenIsolationError("cannot identify the live campaign owner process")
    uid = os.getuid()
    if int(raw_uid) != uid:
        raise QwenIsolationError("campaign owner process belongs to a different user")
    return {
        "uid": uid,
        "campaign_pid": campaign_pid,
        "owner_token_sha256": hashlib.sha256(token.lower().encode("ascii")).hexdigest(),
    }


def _lane_owner_path() -> Path:
    return EXCLUSIVE_LANE / _LANE_OWNER_NAME


def _acquire_owned_lane(*, poll_seconds: float) -> dict[str, object]:
    owner = _campaign_owner_identity()
    _acquire_lane(
        poll_seconds=poll_seconds,
        exclude_pids=(int(owner["campaign_pid"]),),
    )
    owner_path = _lane_owner_path()
    descriptor: int | None = None
    owner_created = False
    try:
        descriptor = os.open(
            owner_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        owner_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(owner, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if owner_created:
            owner_path.unlink(missing_ok=True)
        try:
            EXCLUSIVE_LANE.rmdir()
        except OSError:
            pass
        raise
    return owner


def _release_owned_lane() -> dict[str, object]:
    owner_path = _lane_owner_path()
    try:
        raw_owner = json.loads(
            owner_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenIsolationError(f"exclusive lane owner is unavailable: {exc}") from exc
    if not isinstance(raw_owner, dict) or set(raw_owner) != {
        "uid",
        "campaign_pid",
        "owner_token_sha256",
    }:
        raise QwenIsolationError("exclusive lane owner identity is malformed")
    expected = _campaign_owner_identity()
    if raw_owner != expected:
        raise QwenIsolationError(
            "exclusive lane owner differs from the releasing campaign"
        )
    owner_path.unlink()
    EXCLUSIVE_LANE.rmdir()
    return expected


def _run_action(args: argparse.Namespace) -> dict[str, object]:
    expected_models = tuple(args.expected_models or EXPECTED_QWEN_MODELS)
    if len(expected_models) != len(set(expected_models)):
        raise QwenIsolationError("--expected-model values must be unique")
    api_timeout = _assert_positive(
        args.api_timeout_seconds, field="api timeout seconds"
    )
    timeout = _assert_positive(args.qwen_timeout_seconds, field="Qwen timeout seconds")
    poll = _assert_positive(args.poll_seconds, field="poll seconds")
    common = {
        "uid": os.getuid(),
        "api_url": args.qwen_api_url,
    }

    if args.action == "acquire":
        owner = _acquire_owned_lane(poll_seconds=poll)
        return {"acquired": True, "lane": str(EXCLUSIVE_LANE), "owner": owner}
    if args.action == "release":
        owner = _release_owned_lane()
        return {"lane": str(EXCLUSIVE_LANE), "released": True, "owner": owner}
    if args.action == "capture":
        state = _capture_qwen(
            **common,
            api_timeout=api_timeout,
            expected_models=expected_models,
        )
        return _state_payload(state)

    state = _read_state(expected_models=expected_models)
    if state.loaded and not args.qwen_plist.expanduser().is_file():
        raise QwenIsolationError(
            f"captured loaded Qwen state requires plist {args.qwen_plist.expanduser()}"
        )
    transition = {
        **common,
        "plist": args.qwen_plist.expanduser().resolve(),
        "timeout_seconds": timeout,
        "poll_seconds": poll,
    }
    if args.action == "unload":
        _stop_qwen(state, **transition)
        stopped = _capture_qwen(
            **common,
            api_timeout=api_timeout,
            expected_models=expected_models,
        )
        expected_stopped = QwenState(loaded=False, models=())
        _assert_qwen_state_restored(expected_stopped, stopped)
        return {**_state_payload(stopped), "unloaded": True}

    if args.action == "restore":
        _restore_qwen(state, **transition)
        restored = _capture_qwen(
            **common,
            api_timeout=api_timeout,
            expected_models=state.models or expected_models,
        )
        _assert_qwen_state_restored(state, restored)
        return {**_state_payload(restored), "restored": True}

    if args.action == "verify":
        restored = _capture_qwen(
            **common,
            api_timeout=api_timeout,
            expected_models=state.models or expected_models,
        )
        _assert_qwen_state_restored(state, restored)
        return {**_state_payload(restored), "restored": True}
    raise AssertionError(args.action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact-state Qwen isolation hook for the issue #46 campaign"
    )
    parser.add_argument(
        "action",
        choices=("acquire", "release", "capture", "unload", "restore", "verify"),
    )
    parser.add_argument(
        "--qwen-api-url",
        default="http://127.0.0.1:8080/v1/models",
    )
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--expected-model", dest="expected_models", action="append")
    parser.add_argument("--api-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--qwen-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # The issue-30 guard prints wait diagnostics. Preserve the JSON-only
        # subprocess protocol by routing all transition output to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = _run_action(args)
        rendered = json.dumps(result, sort_keys=True) + "\n"
    except Exception as exc:
        print(f"Qwen isolation rejected: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(rendered)
    return 0


__all__ = [
    "QwenIsolationError",
    "build_parser",
    "main",
]
