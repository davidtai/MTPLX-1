#!/usr/bin/env python3
"""Run the evidence-gated Hy3 Q4 dynamic-memory campaign.

All hardware operations are argv arrays in a JSON spec.  Commands exchange one
JSON object on stdout; Qwen state is passed to unload/restore/verify on stdin.
No command is evaluated through a shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    CONTEXT_MATRIX_TOKENS,
    BenchmarkGateError,
    QwenIsolationHooks,
    balanced_campaign_schedule,
    run_exclusive_hardware_window,
    run_json_subprocess,
    run_subprocess_campaign,
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkGateError(f"{field} must be an object")
    return value


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
            "spec.probe_command",
            _command(spec["probe_command"], field="spec.probe_command"),
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
    for field in ("probe_command", "arm_command_template", "repetitions", "qwen"):
        if field not in spec:
            raise BenchmarkGateError(f"spec is missing required field {field}")
    spec["probe_command"] = _command(spec["probe_command"], field="spec.probe_command")
    spec["arm_command_template"] = _command(
        spec["arm_command_template"], field="spec.arm_command_template"
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
        raise BenchmarkGateError(
            f"cannot read tracked campaign spec from HEAD: {detail}"
        )
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


def _plan(spec: Mapping[str, object]) -> dict[str, object]:
    repetitions = int(spec["repetitions"])
    return {
        "schema": "mtplx-hy3-dynamic-memory-campaign-plan-v1",
        "context_matrix_tokens": list(CONTEXT_MATRIX_TOKENS),
        "repetitions": repetitions,
        "schedule": [
            asdict(entry)
            for entry in balanced_campaign_schedule(repetitions=repetitions)
        ],
        "probe_command": list(spec["probe_command"]),
        "arm_command_template": list(spec["arm_command_template"]),
        "qwen_isolation_configured": True,
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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_spec(
    spec: Mapping[str, object],
    *,
    cwd: Path,
) -> dict[str, object]:
    """Execute a validated command spec inside the exact-Qwen restore window."""

    qwen = _mapping(spec["qwen"], field="spec.qwen")
    owner_env = {
        "MTPLX_ISSUE46_CAMPAIGN_OWNER_TOKEN": secrets.token_hex(32),
        "MTPLX_ISSUE46_CAMPAIGN_PID": str(os.getpid()),
    }

    def command(field: str, *, payload: object | None = None) -> Mapping[str, object]:
        return run_json_subprocess(
            _command(qwen[field], field=f"spec.qwen.{field}"),
            cwd=cwd,
            input_payload=payload,
            env=owner_env,
        )

    hooks = QwenIsolationHooks(
        acquire_lane=lambda: command("acquire_lane_command"),
        release_lane=lambda: command("release_lane_command"),
        capture=lambda: command("capture_command"),
        unload=lambda state: command("unload_command", payload=state),
        restore=lambda state: command("restore_command", payload=state),
        verify_restored=lambda state: (
            command("verify_command", payload=state).get("restored") is True
        ),
    )

    def workload() -> dict[str, object]:
        return run_subprocess_campaign(
            probe_command=_command(spec["probe_command"], field="spec.probe_command"),
            arm_command_template=_command(
                spec["arm_command_template"], field="spec.arm_command_template"
            ),
            repetitions=int(spec["repetitions"]),
            cwd=cwd,
            bootstrap_resamples=int(spec["bootstrap_resamples"]),
            bootstrap_seed=int(spec["bootstrap_seed"]),
        )

    return run_exclusive_hardware_window(workload, hooks=hooks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the issue #46 Hy3 Q4 dynamic-memory evidence campaign"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print the exact schedule without running commands",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = args.spec.expanduser().resolve()
    repo_root = args.cwd.expanduser().resolve()
    if args.plan_only:
        spec = _load_spec(spec_path)
        _require_campaign_command_shapes(spec)
        result = _plan(spec)
    else:
        if args.output_json is None:
            raise BenchmarkGateError("--output-json is required unless --plan-only")
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
        result = dict(run_spec(spec, cwd=repo_root))
        post_commit = _require_clean_source(repo_root)
        _require_tracked_file(
            repo_root,
            spec_path,
            description="campaign spec",
        )
        _require_campaign_sources(spec, repo_root=repo_root)
        if post_commit != source_commit:
            raise BenchmarkGateError("source Git commit changed during the campaign")
        if (
            _read_spec_bytes(spec_path) != frozen_spec_bytes
            or _head_file_bytes(repo_root, spec_path) != frozen_spec_bytes
        ):
            raise BenchmarkGateError("campaign spec changed during the campaign")
        result["campaign_spec_sha256"] = spec_sha256
        result["source_git_commit"] = source_commit
        _atomic_write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
