#!/usr/bin/env python3
"""Four-process ABBA gate for process-latched Qwen 3.8 candidates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_GATE_SCRIPT = ROOT / "scripts/qwen38_challenge_port_gate.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "qwen38_challenge_port_gate",
    _GATE_SCRIPT,
)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load gate module: {_GATE_SCRIPT}")
gate = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(gate)


BUFFER_ENV = ("MLX_MAX_MB_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER")
GUARD_FD_ENV = "MTPLX_GUARD_ATTEST_FD"
GUARD_NONCE_ENV = "MTPLX_GUARD_ATTEST_NONCE"


def _environment_for_route(
    route_id: str,
    inherited: Mapping[str, str],
) -> dict[str, str]:
    features = gate._validate_route_id(route_id)
    environment = dict(inherited)
    for name in (*BUFFER_ENV, GUARD_FD_ENV, GUARD_NONCE_ENV):
        environment.pop(name, None)
    if "r53_command_buffers" in features:
        environment["MLX_MAX_MB_PER_BUFFER"] = "512"
        environment["MLX_MAX_OPS_PER_BUFFER"] = "50"
    return environment


@contextmanager
def _gpu_lock_scope(lock_path: Path) -> Iterator[str]:
    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _verify_parent_guard_attestation,
    )

    if _verify_parent_guard_attestation(lock_path):
        yield "attested_parent"
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"GPU lock is busy: {lock_path}") from exc
        yield "direct"
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_attested_child(
    command: list[str],
    *,
    environment: Mapping[str, str],
    lock_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Delegate the already-verified lock attestation to one direct child."""

    read_fd, write_fd = os.pipe()
    nonce = secrets.token_hex(32)
    child_env = dict(environment)
    child_env[GUARD_FD_ENV] = str(read_fd)
    child_env[GUARD_NONCE_ENV] = nonce
    process = subprocess.Popen(
        command,
        env=child_env,
        pass_fds=(read_fd,),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.close(read_fd)
    issued = time.monotonic_ns()
    observed = lock_path.resolve(strict=True).stat()
    payload = {
        "schema_version": 1,
        "nonce": nonce,
        "guard_pid": os.getpid(),
        "child_pid": process.pid,
        "lock_path": str(lock_path.resolve(strict=True)),
        "lock_device": observed.st_dev,
        "lock_inode": observed.st_ino,
        "issued_monotonic_ns": issued,
        "expires_monotonic_ns": issued + 60_000_000_000,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(write_fd, view)
            view = view[written:]
    finally:
        os.close(write_fd)
    stdout, _ = process.communicate()
    return subprocess.CompletedProcess(command, process.returncode, stdout, None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=gate.DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=gate.DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--context-file", type=Path, default=gate.DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--order", required=True)
    parser.add_argument("--control-route", required=True)
    parser.add_argument("--candidate-route", required=True)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--row17-artifact", type=Path)
    parser.add_argument("--row28-artifact", type=Path)
    parser.add_argument("--row36-artifact", type=Path)
    parser.add_argument("--lock", type=Path, default=gate.DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _child_command(
    args: argparse.Namespace,
    *,
    route_id: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/qwen38_challenge_port_gate.py"),
        "--model",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--context-file",
        str(args.context_file),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--seed",
        str(args.seed),
        "--target-temperature",
        str(args.target_temperature),
        "--order",
        route_id,
        "--lock",
        str(args.lock),
        "--output",
        str(output),
    ]
    for flag, value in (
        ("--draft-temperature", args.draft_temperature),
        ("--source-artifact", args.source_artifact),
        ("--row17-artifact", args.row17_artifact),
        ("--row28-artifact", args.row28_artifact),
        ("--row36-artifact", args.row36_artifact),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def _aggregate(
    args: argparse.Namespace,
    *,
    order: list[str],
    child_receipts: list[dict[str, Any]],
    lock_scope: str,
) -> dict[str, Any]:
    arms = [receipt["arms"][0] for receipt in child_receipts]
    warmups = [receipt["warmups"][0] for receipt in child_receipts]
    unique_routes = list(dict.fromkeys(order))
    correctness = gate._correctness_summary(
        arms,
        route_ids=unique_routes,
        max_tokens=args.max_tokens,
    )
    means = {
        route_id: sum(
            float(arm["wall_s"]) for arm in arms if arm["route_id"] == route_id
        )
        / sum(arm["route_id"] == route_id for arm in arms)
        for route_id in unique_routes
    }
    improvement_pct = (
        means[args.control_route] / means[args.candidate_route] - 1.0
    ) * 100.0
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    engagement_errors = gate._candidate_engagement_errors(
        args.candidate_route,
        warmups,
        arms,
    )
    promotion = gate._promotion_decision(
        order=order,
        control_id=args.control_route,
        candidate_id=args.candidate_route,
        improvement_pct=improvement_pct,
        correctness=correctness,
        source_status=source_status,
        engagement_errors=engagement_errors,
    )
    first = child_receipts[0]
    return {
        **{
            key: first[key]
            for key in (
                "model",
                "prompt_file",
                "context_file",
                "context_sha256",
                "prompt_id",
                "prompt_tokens",
                "prompt_token_sha256",
                "prompt_token_target",
                "max_tokens",
                "seed",
                "target_temperature",
                "draft_temperature",
                "optimized_speed_stack",
                "platform",
                "python",
                "mlx_version",
                "source_commit",
            )
        },
        "kind": "qwen38_challenge_port_isolated_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "isolation_reason": "row53_process_latched_command_buffer_environment",
        "conditioning_scope": "one_1024_token_generation_per_isolated_arm_process",
        "timed_arm_count": 4,
        "order": order,
        "gpu_lock_scope": lock_scope,
        "source_status": source_status,
        "exact": bool(
            correctness["cross_route_token_exact"]
            and correctness["cross_route_schedule_exact"]
        ),
        "token_exact": correctness["cross_route_token_exact"],
        "schedule_exact": correctness["cross_route_schedule_exact"],
        "correctness": correctness,
        "control_route_id": args.control_route,
        "candidate_route_id": args.candidate_route,
        "mean_wall_s": means,
        "candidate_improvement_pct": improvement_pct,
        "candidate_engagement_errors": engagement_errors,
        "promotion": promotion,
        "warmups": warmups,
        "arms": arms,
    }


def main() -> int:
    args = _parse_args()
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    expected = [
        args.control_route,
        args.candidate_route,
        args.candidate_route,
        args.control_route,
    ]
    if order != expected:
        raise ValueError("isolated gate requires exactly four ABBA routes")
    for route_id in order:
        gate._validate_route_id(route_id)

    child_receipts: list[dict[str, Any]] = []
    with _gpu_lock_scope(args.lock) as lock_scope:
        with tempfile.TemporaryDirectory(prefix="qwen38-r53-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, route_id in enumerate(order):
                child_output = temp_root / f"arm-{index}.json"
                result = _run_attested_child(
                    _child_command(args, route_id=route_id, output=child_output),
                    environment=_environment_for_route(route_id, os.environ),
                    lock_path=args.lock,
                )
                if result.returncode not in (0, 2) or not child_output.is_file():
                    raise RuntimeError(
                        f"isolated arm {index} failed ({result.returncode}):\n"
                        f"{result.stdout}"
                    )
                child_receipts.append(
                    json.loads(child_output.read_text(encoding="utf-8"))
                )

    receipt = _aggregate(
        args,
        order=order,
        child_receipts=child_receipts,
        lock_scope=lock_scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "exact": receipt["exact"],
                "candidate_improvement_pct": receipt[
                    "candidate_improvement_pct"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["promotion"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
