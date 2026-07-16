#!/usr/bin/env python3
"""Run the evidence-gated Hy3 Q4 dynamic-memory campaign.

All hardware operations are argv arrays in a JSON spec.  Commands exchange one
JSON object on stdout; Qwen state is passed to unload/restore/verify on stdin.
No command is evaluated through a shell.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    CONTEXT_MATRIX_TOKENS,
    DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS,
    DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS,
    BenchmarkGateError,
    QwenIsolationHooks,
    balanced_campaign_schedule,
    run_exclusive_hardware_window,
    run_json_subprocess,
    run_subprocess_campaign,
    run_subprocess_diagnostic_arm,
)


QWEN_RECOVERY_SCHEMA = "mtplx-issue46-qwen-recovery-v1"
QWEN_EVIDENCE_SCHEMA = "mtplx-issue46-qwen-isolation-v1"
QWEN_RECOVERY_NAME = "issue46-recovery.json"
LEGACY_EXCLUSIVE_LANE_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkGateError(f"{field} must be an object")
    return value


def _acquire_legacy_lane_lock(path: Path, *, wait_seconds: float) -> int:
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        deadline = time.monotonic() + float(wait_seconds)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise BenchmarkGateError(
                        "legacy exclusive GPU lane is active; refusing overlapping "
                        "hardware work"
                    ) from exc
                time.sleep(min(0.25, remaining))
    except BaseException:
        os.close(descriptor)
        raise


def _release_legacy_lane_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _command(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BenchmarkGateError(f"{field} must be an argv array")
    command = tuple(value)
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise BenchmarkGateError(f"{field} must be a nonempty argv array")
    return command


def _exact_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkGateError(f"{field} must be an integer >= {minimum}")
    return value


def _parse_contexts(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from exc
    expected = tuple(
        context for context in CONTEXT_MATRIX_TOKENS if context in set(contexts)
    )
    if not contexts or contexts != expected or contexts[0] != 4_096:
        raise argparse.ArgumentTypeError(
            "contexts must start with 4096 and be an ordered subset of "
            + ",".join(str(context) for context in CONTEXT_MATRIX_TOKENS)
        )
    return contexts


def _git(
    repo_root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkGateError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _require_clean_source(repo_root: Path) -> str:
    root = repo_root.expanduser().resolve()
    actual = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if actual != root:
        raise BenchmarkGateError(
            f"campaign cwd must be the exact source worktree root: {actual}"
        )
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise BenchmarkGateError("source Git commit is not a full hexadecimal hash")
    index_entries = _git(root, "ls-files", "-v").stdout.splitlines()
    assume_unchanged = [entry[2:] for entry in index_entries if entry[:1].islower()]
    skip_worktree = [entry[2:] for entry in index_entries if entry.startswith("S ")]
    if assume_unchanged:
        raise BenchmarkGateError(
            "source worktree contains assume-unchanged index entries: "
            + ", ".join(assume_unchanged)
        )
    if skip_worktree:
        raise BenchmarkGateError(
            "source worktree contains skip-worktree index entries: "
            + ", ".join(skip_worktree)
        )
    dirty = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout.strip()
    if dirty:
        raise BenchmarkGateError(
            f"hardware campaign requires a clean source worktree; dirty: {dirty}"
        )
    return commit


def _source_commit(repo_root: Path) -> str:
    root = repo_root.expanduser().resolve()
    actual = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if actual != root:
        raise BenchmarkGateError(
            f"campaign cwd must be the exact source worktree root: {actual}"
        )
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise BenchmarkGateError("source Git commit is not a full hexadecimal hash")
    return commit


def _require_tracked_file(
    repo_root: Path,
    path: Path,
    *,
    description: str,
) -> Path:
    root = repo_root.resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise BenchmarkGateError(
            f"{description} must be a tracked source inside the campaign worktree"
        ) from exc
    tracked = _git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative.as_posix(),
        check=False,
    )
    if tracked.returncode != 0:
        raise BenchmarkGateError(
            f"{description} is not a tracked source: {relative.as_posix()}"
        )
    return resolved


def _command_source(command: Sequence[str], *, field: str) -> Path:
    candidate_indices = [
        index for index, item in enumerate(command) if item.endswith(".py")
    ]
    if len(candidate_indices) != 1:
        raise BenchmarkGateError(
            f"{field} must invoke exactly one tracked Python script command source"
        )
    source_index = candidate_indices[0]
    if source_index == 0:
        raise BenchmarkGateError(f"{field} has no Python interpreter before its source")
    interpreter = Path(command[source_index - 1]).name
    if not interpreter.startswith("python"):
        raise BenchmarkGateError(
            f"{field} must execute its command source as a Python script"
        )
    prefix = tuple(command[:source_index])
    direct_python = source_index == 1 and interpreter.startswith("python")
    uv_python = (
        source_index == 3
        and Path(prefix[0]).name == "uv"
        and prefix[1] == "run"
        and interpreter.startswith("python")
    )
    if not (direct_python or uv_python):
        raise BenchmarkGateError(
            f"{field} must use a direct Python script command source"
        )
    return Path(command[source_index])


def _campaign_commands(
    spec: Mapping[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    commands: list[tuple[str, tuple[str, ...]]] = [
        (
            "spec.artifact_verify_command",
            _command(
                spec["artifact_verify_command"],
                field="spec.artifact_verify_command",
            ),
        ),
        (
            "spec.probe_command",
            _command(spec["probe_command"], field="spec.probe_command"),
        ),
        (
            "spec.quality_command",
            _command(spec["quality_command"], field="spec.quality_command"),
        ),
        (
            "spec.arm_command_template",
            _command(spec["arm_command_template"], field="spec.arm_command_template"),
        ),
    ]
    qwen = _mapping(spec["qwen"], field="spec.qwen")
    for field in (
        "acquire_lane_command",
        "release_lane_command",
        "capture_command",
        "unload_command",
        "restore_command",
        "verify_command",
    ):
        commands.append(
            (f"spec.qwen.{field}", _command(qwen[field], field=f"spec.qwen.{field}"))
        )
    return commands


def _require_campaign_command_shapes(spec: Mapping[str, object]) -> None:
    for field, command in _campaign_commands(spec):
        _command_source(command, field=field)


def _require_campaign_sources(
    spec: Mapping[str, object],
    *,
    repo_root: Path,
) -> None:
    commands = _campaign_commands(spec)
    for field, command in commands:
        source = _command_source(command, field=field)
        resolved = _require_tracked_file(
            repo_root,
            source,
            description=f"{field} command source",
        )
        try:
            source_bytes = resolved.read_bytes()
        except OSError as exc:
            raise BenchmarkGateError(
                f"cannot read {field} command source: {exc}"
            ) from exc
        if source_bytes != _head_file_bytes(repo_root, resolved):
            raise BenchmarkGateError(
                f"{field} command source differs from its tracked HEAD bytes"
            )


def _command_option(command: Sequence[str], option: str, *, field: str) -> str:
    positions = [index for index, item in enumerate(command) if item == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise BenchmarkGateError(f"{field} must declare exactly one {option}")
    value = command[positions[0] + 1]
    if not value or value.startswith("-"):
        raise BenchmarkGateError(f"{field} has no value for {option}")
    return value


def _hardware_hooks_config_path(
    spec: Mapping[str, object],
    *,
    repo_root: Path,
) -> Path:
    fields = (
        "artifact_verify_command",
        "probe_command",
        "quality_command",
        "arm_command_template",
    )
    paths = []
    for field in fields:
        command = _command(spec[field], field=f"spec.{field}")
        raw_path = _command_option(command, "--hooks-config", field=f"spec.{field}")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        paths.append(candidate.resolve())
    if len(set(paths)) != 1:
        raise BenchmarkGateError(
            "artifact verification, probe, quality, and arm commands must use "
            "the same hardware hooks config"
        )
    return paths[0]


def _load_frozen_tracked_file(
    repo_root: Path,
    path: Path,
    *,
    description: str,
) -> tuple[Path, bytes, str]:
    resolved = _require_tracked_file(repo_root, path, description=description)
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        raise BenchmarkGateError(f"cannot read {description}: {exc}") from exc
    if raw_bytes != _head_file_bytes(repo_root, resolved):
        raise BenchmarkGateError(f"{description} differs from its tracked HEAD bytes")
    return resolved, raw_bytes, hashlib.sha256(raw_bytes).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkGateError(f"benchmark spec has duplicate key {key!r}")
        result[key] = value
    return result


def _parse_spec_bytes(raw_bytes: bytes, *, path: Path) -> dict[str, object]:
    try:
        raw = json.loads(raw_bytes, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkGateError(f"cannot load benchmark spec {path}: {exc}") from exc
    spec = dict(_mapping(raw, field="spec"))
    for field in (
        "artifact_verify_command",
        "probe_command",
        "quality_command",
        "arm_command_template",
        "legacy_exclusive_lane_lock",
        "legacy_exclusive_lane_wait_seconds",
        "repetitions",
        "qwen",
    ):
        if field not in spec:
            raise BenchmarkGateError(f"spec is missing required field {field}")
    spec["probe_command"] = _command(spec["probe_command"], field="spec.probe_command")
    spec["quality_command"] = _command(
        spec["quality_command"], field="spec.quality_command"
    )
    spec["artifact_verify_command"] = _command(
        spec["artifact_verify_command"], field="spec.artifact_verify_command"
    )
    spec["arm_command_template"] = _command(
        spec["arm_command_template"], field="spec.arm_command_template"
    )
    legacy_lock = spec["legacy_exclusive_lane_lock"]
    if not isinstance(legacy_lock, str) or not legacy_lock:
        raise BenchmarkGateError(
            "spec.legacy_exclusive_lane_lock must be a nonempty path"
        )
    if Path(legacy_lock) != LEGACY_EXCLUSIVE_LANE_LOCK:
        raise BenchmarkGateError(
            "spec.legacy_exclusive_lane_lock must use the shared MTPLX GPU lock"
        )
    spec["legacy_exclusive_lane_wait_seconds"] = _exact_int(
        spec["legacy_exclusive_lane_wait_seconds"],
        field="spec.legacy_exclusive_lane_wait_seconds",
        minimum=0,
    )
    repetitions = _exact_int(spec["repetitions"], field="spec.repetitions", minimum=2)
    if repetitions % 2:
        raise BenchmarkGateError("spec.repetitions must be even for balanced ordering")
    spec["repetitions"] = repetitions
    spec["bootstrap_resamples"] = _exact_int(
        spec.get("bootstrap_resamples", 10_000),
        field="spec.bootstrap_resamples",
        minimum=100,
    )
    spec["bootstrap_seed"] = _exact_int(
        spec.get("bootstrap_seed", 46), field="spec.bootstrap_seed", minimum=0
    )
    spec["workload_subprocess_timeout_seconds"] = _exact_int(
        spec.get(
            "workload_subprocess_timeout_seconds",
            DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS,
        ),
        field="spec.workload_subprocess_timeout_seconds",
        minimum=1,
    )
    spec["qwen_control_timeout_seconds"] = _exact_int(
        spec.get("qwen_control_timeout_seconds", 300),
        field="spec.qwen_control_timeout_seconds",
        minimum=1,
    )
    spec["subprocess_termination_grace_seconds"] = _exact_int(
        spec.get(
            "subprocess_termination_grace_seconds",
            DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS,
        ),
        field="spec.subprocess_termination_grace_seconds",
        minimum=1,
    )
    qwen = dict(_mapping(spec["qwen"], field="spec.qwen"))
    for field in (
        "acquire_lane_command",
        "release_lane_command",
        "capture_command",
        "unload_command",
        "restore_command",
        "verify_command",
    ):
        if field not in qwen:
            raise BenchmarkGateError(f"spec.qwen is missing required field {field}")
        qwen[field] = _command(qwen[field], field=f"spec.qwen.{field}")
    spec["qwen"] = qwen
    return spec


def _read_spec_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BenchmarkGateError(f"cannot load benchmark spec {path}: {exc}") from exc


def _load_spec(path: Path) -> dict[str, object]:
    return _parse_spec_bytes(_read_spec_bytes(path), path=path)


def _head_file_bytes(repo_root: Path, path: Path) -> bytes:
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    completed = subprocess.run(
        ("git", "show", f"HEAD:{relative}"),
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkGateError(f"cannot read tracked source file from HEAD: {detail}")
    return completed.stdout


def _load_frozen_spec(
    path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, object], bytes, str]:
    raw_bytes = _read_spec_bytes(path)
    if raw_bytes != _head_file_bytes(repo_root, path):
        raise BenchmarkGateError(
            "campaign spec bytes differ from the tracked HEAD source"
        )
    return (
        _parse_spec_bytes(raw_bytes, path=path),
        raw_bytes,
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def _plan(
    spec: Mapping[str, object],
    *,
    contexts: Sequence[int] = CONTEXT_MATRIX_TOKENS,
) -> dict[str, object]:
    repetitions = int(spec["repetitions"])
    return {
        "schema": "mtplx-hy3-dynamic-memory-campaign-plan-v1",
        "context_matrix_tokens": list(contexts),
        "available_context_matrix_tokens": list(CONTEXT_MATRIX_TOKENS),
        "repetitions": repetitions,
        "schedule": [
            asdict(entry)
            for entry in balanced_campaign_schedule(
                repetitions=repetitions,
                contexts=contexts,
            )
        ],
        "probe_command": list(spec["probe_command"]),
        "quality_command": list(spec["quality_command"]),
        "artifact_verify_command": list(spec["artifact_verify_command"]),
        "arm_command_template": list(spec["arm_command_template"]),
        "legacy_exclusive_lane_lock": str(spec["legacy_exclusive_lane_lock"]),
        "legacy_exclusive_lane_wait_seconds": int(
            spec["legacy_exclusive_lane_wait_seconds"]
        ),
        "workload_subprocess_timeout_seconds": int(
            spec["workload_subprocess_timeout_seconds"]
        ),
        "qwen_control_timeout_seconds": int(spec["qwen_control_timeout_seconds"]),
        "subprocess_termination_grace_seconds": int(
            spec["subprocess_termination_grace_seconds"]
        ),
        "qwen_isolation_configured": True,
        "execution_mode": "foreground-synchronous",
        "background_workloads": False,
    }


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _invalidate_output(path: Path) -> None:
    """Remove prior evidence before any new actual-run validation can fail."""

    try:
        existed = path.exists() or path.is_symlink()
        path.unlink(missing_ok=True)
        if existed:
            _fsync_directory(path.parent)
    except OSError as exc:
        raise BenchmarkGateError(
            f"cannot invalidate prior campaign output: {exc}"
        ) from exc


def _campaign_rejected(value: Mapping[str, object]) -> bool:
    acceptance = value.get("acceptance")
    return value.get("status") == "rejected" or (
        isinstance(acceptance, Mapping) and acceptance.get("passed") is False
    )


def _qwen_state(value: object, *, field: str) -> dict[str, object]:
    raw = _mapping(value, field=field)
    if set(raw) != {"loaded", "models"} or not isinstance(raw["loaded"], bool):
        raise BenchmarkGateError(f"{field} must contain exact loaded/models state")
    models = raw["models"]
    if (
        isinstance(models, (str, bytes, bytearray))
        or not isinstance(models, Sequence)
        or any(not isinstance(model, str) or not model for model in models)
    ):
        raise BenchmarkGateError(f"{field}.models must be an array of model IDs")
    normalized = list(models)
    if len(normalized) != len(set(normalized)):
        raise BenchmarkGateError(f"{field}.models contains duplicates")
    if raw["loaded"] is False and normalized:
        raise BenchmarkGateError(f"{field} has models while Qwen is unloaded")
    return {"loaded": raw["loaded"], "models": normalized}


def _qwen_transition(
    value: object,
    *,
    field: str,
    marker: str,
    expected_state: Mapping[str, object],
) -> dict[str, object]:
    raw = dict(_mapping(value, field=field))
    if raw.pop(marker, None) is not True:
        raise BenchmarkGateError(f"{field}.{marker} must be true")
    state = _qwen_state(raw, field=field)
    if state != dict(expected_state):
        raise BenchmarkGateError(f"{field} differs from the captured state")
    return {**state, marker: True}


def _write_all(descriptor: int, payload: bytes) -> None:
    cursor = 0
    while cursor < len(payload):
        try:
            written = os.write(descriptor, payload[cursor:])
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("short write while persisting Qwen recovery journal")
        cursor += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_qwen_recovery_journal(
    *,
    lane: Path,
    owner: Mapping[str, object],
    captured_state: Mapping[str, object],
) -> dict[str, object]:
    if not lane.is_dir():
        raise BenchmarkGateError("Qwen acquire did not create its exclusive lane")
    payload_value = {
        "schema": QWEN_RECOVERY_SCHEMA,
        "owner": dict(owner),
        "captured_state": dict(captured_state),
    }
    payload = (
        json.dumps(payload_value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    journal = lane / QWEN_RECOVERY_NAME
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            journal,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        created = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(lane)
    except BaseException as exc:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            journal.unlink(missing_ok=True)
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, BenchmarkGateError):
            raise
        raise BenchmarkGateError(
            f"cannot persist Qwen recovery journal before unload: {exc}"
        ) from exc
    return {
        "path": str(journal),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "durable_before_unload": True,
        "removed_after_restore_verification": False,
    }


def _remove_qwen_recovery_journal(evidence: dict[str, object]) -> None:
    journal = Path(str(evidence["path"]))
    try:
        journal.unlink()
        _fsync_directory(journal.parent)
    except OSError as exc:
        raise BenchmarkGateError(
            f"cannot remove verified Qwen recovery journal: {exc}"
        ) from exc
    evidence["removed_after_restore_verification"] = True


def _canonical_release_lane(value: object, *, acquired_lane: Path | None) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkGateError("Qwen release result omitted its lane path")
    if acquired_lane is None:
        raise BenchmarkGateError("Qwen release ran before lane acquisition")
    released_lane = Path(value).expanduser().resolve()
    if released_lane != acquired_lane:
        raise BenchmarkGateError("Qwen release lane differs from acquisition")
    return released_lane


def run_spec(
    spec: Mapping[str, object],
    *,
    cwd: Path,
    contexts: Sequence[int] = CONTEXT_MATRIX_TOKENS,
    diagnostic_arm: str | None = None,
) -> dict[str, object]:
    """Execute a validated command spec inside the exact-Qwen restore window."""

    if diagnostic_arm is not None:
        if diagnostic_arm not in {"static", "dynamic"}:
            raise BenchmarkGateError("diagnostic arm must be static or dynamic")
        if len(contexts) != 1:
            raise BenchmarkGateError(
                "diagnostic mode requires exactly one selected context"
            )

    qwen = _mapping(spec["qwen"], field="spec.qwen")
    workload_timeout_seconds = _exact_int(
        spec.get(
            "workload_subprocess_timeout_seconds",
            DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS,
        ),
        field="spec.workload_subprocess_timeout_seconds",
        minimum=1,
    )
    qwen_control_timeout_seconds = _exact_int(
        spec.get("qwen_control_timeout_seconds", 300),
        field="spec.qwen_control_timeout_seconds",
        minimum=1,
    )
    termination_grace_seconds = _exact_int(
        spec.get(
            "subprocess_termination_grace_seconds",
            DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS,
        ),
        field="spec.subprocess_termination_grace_seconds",
        minimum=1,
    )
    owner_env = {
        "MTPLX_ISSUE46_CAMPAIGN_OWNER_TOKEN": secrets.token_hex(32),
        "MTPLX_ISSUE46_CAMPAIGN_PID": str(os.getpid()),
    }
    qwen_evidence: dict[str, object] = {"schema": QWEN_EVIDENCE_SCHEMA}
    journal_evidence: dict[str, object] | None = None
    captured_state: dict[str, object] | None = None
    acquired_lane: Path | None = None
    acquired_owner: dict[str, object] | None = None
    legacy_lock_descriptor: int | None = None
    legacy_lock_value = spec.get("legacy_exclusive_lane_lock")
    legacy_lock_path = (
        None
        if legacy_lock_value is None
        else Path(str(legacy_lock_value)).expanduser().resolve()
    )
    legacy_lock_wait_seconds = _exact_int(
        spec.get("legacy_exclusive_lane_wait_seconds", 0),
        field="spec.legacy_exclusive_lane_wait_seconds",
        minimum=0,
    )

    def acquire_legacy_lane_lock() -> None:
        nonlocal legacy_lock_descriptor
        if legacy_lock_path is None:
            return
        legacy_lock_descriptor = _acquire_legacy_lane_lock(
            legacy_lock_path,
            wait_seconds=legacy_lock_wait_seconds,
        )
        qwen_evidence["legacy_exclusive_lane_lock"] = {
            "path": str(legacy_lock_path),
            "acquired": True,
            "released": False,
        }

    def release_legacy_lane_lock() -> None:
        nonlocal legacy_lock_descriptor
        descriptor = legacy_lock_descriptor
        if descriptor is None:
            return
        legacy_lock_descriptor = None
        _release_legacy_lane_lock(descriptor)
        evidence = qwen_evidence.get("legacy_exclusive_lane_lock")
        if isinstance(evidence, dict):
            evidence["released"] = True

    def command(field: str, *, payload: object | None = None) -> Mapping[str, object]:
        return run_json_subprocess(
            _command(qwen[field], field=f"spec.qwen.{field}"),
            cwd=cwd,
            input_payload=payload,
            env=owner_env,
            timeout_seconds=qwen_control_timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )

    def acquire_lane() -> None:
        nonlocal acquired_lane, acquired_owner
        try:
            if legacy_lock_path is not None and legacy_lock_descriptor is None:
                raise BenchmarkGateError(
                    "legacy exclusive GPU lane was not acquired before Qwen handoff"
                )
            result = dict(command("acquire_lane_command"))
            if result.get("acquired") is not True:
                raise BenchmarkGateError(
                    "Qwen exclusive lane acquisition was not proven"
                )
            lane_value = result.get("lane")
            if not isinstance(lane_value, str) or not lane_value:
                raise BenchmarkGateError("Qwen acquire result omitted its lane path")
            owner = dict(_mapping(result.get("owner"), field="Qwen acquire owner"))
            acquired_lane = Path(lane_value).expanduser().resolve()
            acquired_owner = owner
            qwen_evidence["acquire"] = {
                "acquired": True,
                "lane": str(acquired_lane),
                "owner": owner,
            }
        except BaseException:
            release_legacy_lane_lock()
            raise

    def capture() -> dict[str, object]:
        nonlocal captured_state, journal_evidence
        if acquired_lane is None or acquired_owner is None:
            raise BenchmarkGateError("Qwen capture ran before lane acquisition")
        captured_state = _qwen_state(command("capture_command"), field="Qwen capture")
        journal_evidence = _write_qwen_recovery_journal(
            lane=acquired_lane,
            owner=acquired_owner,
            captured_state=captured_state,
        )
        qwen_evidence["capture"] = dict(captured_state)
        qwen_evidence["recovery_journal"] = journal_evidence
        return captured_state

    def unload(state: object) -> None:
        if captured_state is None or dict(_mapping(state, field="Qwen state")) != (
            captured_state
        ):
            raise BenchmarkGateError("Qwen unload state differs from captured state")
        stopped = {"loaded": False, "models": []}
        qwen_evidence["unload"] = _qwen_transition(
            command("unload_command", payload=state),
            field="Qwen unload",
            marker="unloaded",
            expected_state=stopped,
        )

    def restore(state: object) -> None:
        if captured_state is None:
            raise BenchmarkGateError("Qwen restore has no captured state")
        qwen_evidence["restore"] = _qwen_transition(
            command("restore_command", payload=state),
            field="Qwen restore",
            marker="restored",
            expected_state=captured_state,
        )

    def verify_restored(state: object) -> bool:
        if captured_state is None:
            raise BenchmarkGateError("Qwen verification has no captured state")
        qwen_evidence["verify"] = _qwen_transition(
            command("verify_command", payload=state),
            field="Qwen verify",
            marker="restored",
            expected_state=captured_state,
        )
        return True

    def release_lane() -> None:
        if journal_evidence is not None:
            _remove_qwen_recovery_journal(journal_evidence)
        result = dict(command("release_lane_command"))
        if result.get("released") is not True:
            raise BenchmarkGateError("Qwen exclusive lane release was not proven")
        released_lane = _canonical_release_lane(
            result.get("lane"),
            acquired_lane=acquired_lane,
        )
        if acquired_owner is not None and result.get("owner") != acquired_owner:
            raise BenchmarkGateError("Qwen release owner differs from acquisition")
        result["lane"] = str(released_lane)
        qwen_evidence["release"] = result
        release_legacy_lane_lock()

    hooks = QwenIsolationHooks(
        acquire_lane=acquire_lane,
        release_lane=release_lane,
        capture=capture,
        unload=unload,
        restore=restore,
        verify_restored=verify_restored,
    )

    def workload() -> dict[str, object]:
        if diagnostic_arm is not None:
            return run_subprocess_diagnostic_arm(
                probe_command=_command(
                    spec["probe_command"], field="spec.probe_command"
                ),
                artifact_verify_command=_command(
                    spec["artifact_verify_command"],
                    field="spec.artifact_verify_command",
                ),
                arm_command_template=_command(
                    spec["arm_command_template"], field="spec.arm_command_template"
                ),
                arm=diagnostic_arm,
                context_tokens=int(contexts[0]),
                cwd=cwd,
                subprocess_timeout_seconds=workload_timeout_seconds,
                subprocess_termination_grace_seconds=termination_grace_seconds,
            )
        return run_subprocess_campaign(
            probe_command=_command(spec["probe_command"], field="spec.probe_command"),
            quality_command=_command(
                spec["quality_command"], field="spec.quality_command"
            ),
            artifact_verify_command=_command(
                spec["artifact_verify_command"],
                field="spec.artifact_verify_command",
            ),
            arm_command_template=_command(
                spec["arm_command_template"], field="spec.arm_command_template"
            ),
            repetitions=int(spec["repetitions"]),
            contexts=contexts,
            cwd=cwd,
            bootstrap_resamples=int(spec["bootstrap_resamples"]),
            bootstrap_seed=int(spec["bootstrap_seed"]),
            subprocess_timeout_seconds=workload_timeout_seconds,
            subprocess_termination_grace_seconds=termination_grace_seconds,
        )

    acquire_legacy_lane_lock()
    try:
        result = dict(run_exclusive_hardware_window(workload, hooks=hooks))
    except BaseException:
        if "acquire" not in qwen_evidence:
            release_legacy_lane_lock()
        raise
    required_evidence = {
        "schema",
        "acquire",
        "capture",
        "recovery_journal",
        "unload",
        "restore",
        "verify",
        "release",
    }
    if legacy_lock_path is not None:
        required_evidence.add("legacy_exclusive_lane_lock")
    if set(qwen_evidence) != required_evidence:
        raise BenchmarkGateError("Qwen isolation evidence is incomplete")
    result["qwen_isolation"] = qwen_evidence
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the issue #46 Hy3 Q4 dynamic-memory evidence campaign"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contexts",
        type=_parse_contexts,
        default=CONTEXT_MATRIX_TOKENS,
        help="comma-separated ordered subset: 4096,32768,65536,131072",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print the exact schedule without running commands",
    )
    parser.add_argument(
        "--diagnostic-arm",
        choices=("static", "dynamic"),
        help=(
            "run exactly one non-qualifying arm under the normal lock and Qwen "
            "lifecycle; requires one --contexts value"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = args.spec.expanduser().resolve()
    repo_root = args.cwd.expanduser().resolve()
    if args.plan_only:
        if args.diagnostic_arm is not None:
            raise BenchmarkGateError(
                "--diagnostic-arm cannot be combined with --plan-only"
            )
        if args.output_json is not None:
            raise BenchmarkGateError("--output-json cannot be used with --plan-only")
        _require_tracked_file(repo_root, spec_path, description="campaign spec")
        source_commit = _source_commit(repo_root)
        spec, _frozen_spec_bytes, spec_sha256 = _load_frozen_spec(
            spec_path,
            repo_root=repo_root,
        )
        _require_campaign_sources(spec, repo_root=repo_root)
        hooks_config_path = _hardware_hooks_config_path(spec, repo_root=repo_root)
        _hooks_path, _hooks_bytes, hooks_config_sha256 = _load_frozen_tracked_file(
            repo_root,
            hooks_config_path,
            description="hardware hooks config",
        )
        result = _plan(spec, contexts=args.contexts)
        result["campaign_spec_sha256"] = spec_sha256
        result["hardware_hooks_config_sha256"] = hooks_config_sha256
        result["source_git_commit"] = source_commit
        exit_code = 0
    else:
        if args.output_json is None:
            raise BenchmarkGateError("--output-json is required unless --plan-only")
        output_path = args.output_json.expanduser().resolve()
        _invalidate_output(output_path)
        _require_tracked_file(
            repo_root,
            spec_path,
            description="campaign spec",
        )
        source_commit = _require_clean_source(repo_root)
        spec, frozen_spec_bytes, spec_sha256 = _load_frozen_spec(
            spec_path,
            repo_root=repo_root,
        )
        _require_campaign_sources(spec, repo_root=repo_root)
        hooks_config_path = _hardware_hooks_config_path(spec, repo_root=repo_root)
        (
            hooks_config_path,
            frozen_hooks_config_bytes,
            hooks_config_sha256,
        ) = _load_frozen_tracked_file(
            repo_root,
            hooks_config_path,
            description="hardware hooks config",
        )
        if args.diagnostic_arm is None:
            result = dict(run_spec(spec, cwd=repo_root, contexts=args.contexts))
        else:
            result = dict(
                run_spec(
                    spec,
                    cwd=repo_root,
                    contexts=args.contexts,
                    diagnostic_arm=args.diagnostic_arm,
                )
            )
        post_commit = _require_clean_source(repo_root)
        _require_tracked_file(
            repo_root,
            spec_path,
            description="campaign spec",
        )
        _require_campaign_sources(spec, repo_root=repo_root)
        post_hooks_path, post_hooks_bytes, _post_hooks_sha256 = (
            _load_frozen_tracked_file(
                repo_root,
                _hardware_hooks_config_path(spec, repo_root=repo_root),
                description="hardware hooks config",
            )
        )
        if post_commit != source_commit:
            raise BenchmarkGateError("source Git commit changed during the campaign")
        if (
            _read_spec_bytes(spec_path) != frozen_spec_bytes
            or _head_file_bytes(repo_root, spec_path) != frozen_spec_bytes
        ):
            raise BenchmarkGateError("campaign spec changed during the campaign")
        if (
            post_hooks_path != hooks_config_path
            or post_hooks_bytes != frozen_hooks_config_bytes
        ):
            raise BenchmarkGateError(
                "hardware hooks config changed during the campaign"
            )
        result["campaign_spec_sha256"] = spec_sha256
        result["hardware_hooks_config_sha256"] = hooks_config_sha256
        result["source_git_commit"] = source_commit
        _atomic_write_json(output_path, result)
        exit_code = 2 if _campaign_rejected(result) else 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
