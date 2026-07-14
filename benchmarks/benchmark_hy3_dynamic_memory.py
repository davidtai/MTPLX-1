#!/usr/bin/env python3
"""Run the evidence-gated Hy3 Q4 dynamic-memory campaign.

All hardware operations are argv arrays in a JSON spec.  Commands exchange one
JSON object on stdout; Qwen state is passed to unload/restore/verify on stdin.
No command is evaluated through a shell.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _load_spec(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
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

    def command(field: str, *, payload: object | None = None) -> Mapping[str, object]:
        return run_json_subprocess(
            _command(qwen[field], field=f"spec.qwen.{field}"),
            cwd=cwd,
            input_payload=payload,
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
    spec = _load_spec(args.spec.expanduser().resolve())
    if args.plan_only:
        result = _plan(spec)
    else:
        if args.output_json is None:
            raise BenchmarkGateError("--output-json is required unless --plan-only")
        result = run_spec(spec, cwd=args.cwd.expanduser().resolve())
        _atomic_write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
