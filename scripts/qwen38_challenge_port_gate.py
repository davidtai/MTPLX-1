#!/usr/bin/env python3
"""Matched real-model gate for Qwen 3.8 challenge-port candidates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = Path.home() / (
    ".mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
)
DEFAULT_PROMPT = ROOT / "mtplx/benchmarks/prompts/python_modules_long.jsonl"
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")


def _read_prompt(path: Path) -> tuple[str, str]:
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return str(row["id"]), str(row["prompt"])


def _token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_arm(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    prompt_ids: list[int],
    *,
    route_id: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    from mtplx.generation import generate_mtpk
    from mtplx.qwen38_challenge import install_qwen38_route
    from mtplx.sampling import SamplerConfig

    route = install_qwen38_route(
        runtime,
        config,
        model_path,
        cache_route=route_id,
    )
    target_sampler = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    draft_sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    started = time.perf_counter()
    output = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=target_sampler,
        draft_sampler=draft_sampler,
        speculative_depth=3,
        seed=seed,
        verify_strategy="capture_commit",
        mtp_history_policy="committed",
    )
    wall_s = time.perf_counter() - started
    stats = output.stats
    return {
        "route_id": route.route_id,
        "route_fingerprint": route.fingerprint,
        "kernel_ids": list(route.kernel_ids),
        "wall_s": wall_s,
        "generated_tokens": int(stats.generated_tokens),
        "decode_tok_s": float(stats.decode_tok_s),
        "prompt_mtp_history_time_s": float(stats.prompt_mtp_history_time_s),
        "draft_time_s": float(stats.draft_time_s),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "attempted_depth_schedule": [
            int(event.get("depth", 0)) for event in stats.events
        ],
        "accepted_depth_schedule": [
            int(event.get("accepted_depths", 0)) for event in stats.events
        ],
        "token_hash": _token_hash(list(output.tokens)),
        "tokens": list(output.tokens),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", default="control,kv_only_history,kv_only_history,control")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    lock_handle = args.lock.open("a+")
    try:
        fcntl.lockf(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError(f"GPU lock is busy: {args.lock}") from exc

    from mtplx.artifacts import load_config
    from mtplx.runtime import load

    config = load_config(model_path)
    runtime = load(model_path, mtp=True)
    prompt_id, prompt = _read_prompt(args.prompt_file)
    prompt_ids = list(runtime.tokenizer.encode(prompt))
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    if not order or any(item not in {"control", "kv_only_history"} for item in order):
        raise ValueError("order must contain only control and kv_only_history")

    warmups = [
        _run_arm(
            runtime,
            config,
            model_path,
            prompt_ids,
            route_id=route_id,
            max_tokens=args.warmup_tokens,
            seed=args.seed,
        )
        for route_id in ("control", "kv_only_history")
    ]
    arms = [
        _run_arm(
            runtime,
            config,
            model_path,
            prompt_ids,
            route_id=route_id,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        for route_id in order
    ]
    hashes = {arm["token_hash"] for arm in arms}
    attempted = {tuple(arm["attempted_depth_schedule"]) for arm in arms}
    accepted = {tuple(arm["accepted_depth_schedule"]) for arm in arms}
    exact = len(hashes) == len(attempted) == len(accepted) == 1
    by_route = {
        route_id: [arm["wall_s"] for arm in arms if arm["route_id"] == route_id]
        for route_id in ("control", "kv_only_history")
    }
    means = {
        route_id: sum(values) / len(values)
        for route_id, values in by_route.items()
        if values
    }
    improvement_pct = (
        (means["control"] / means["kv_only_history"] - 1.0) * 100.0
        if set(means) == {"control", "kv_only_history"}
        else None
    )
    receipt = {
        "kind": "qwen38_challenge_port_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model_path),
        "prompt_file": str(args.prompt_file.resolve()),
        "prompt_id": prompt_id,
        "prompt_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "order": order,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_version": importlib.metadata.version("mlx"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
        ).splitlines(),
        "exact": exact,
        "mean_wall_s": means,
        "candidate_improvement_pct": improvement_pct,
        "warmups": warmups,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "exact": exact,
        "candidate_improvement_pct": improvement_pct,
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
