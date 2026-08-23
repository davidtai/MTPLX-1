#!/usr/bin/env python3
"""Matched real-model gate for Qwen 3.8 challenge-port candidates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
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
DEFAULT_CONTEXT = ROOT / "mtplx/generation.py"
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
PROMOTION_THRESHOLD_PCT = 0.05


def _read_prompt(path: Path) -> tuple[str, str]:
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return str(row["id"]), str(row["prompt"])


def _expand_prompt_to_token_count(
    tokenizer: Any,
    seed_prompt: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """Repeat a fixed seed and truncate its token IDs to an exact cold-prefill size."""

    if target_tokens <= 0:
        raise ValueError("prompt token target must be positive")
    unit = seed_prompt.rstrip() + "\n"
    repeats = 1
    token_ids = list(tokenizer.encode(unit))
    while len(token_ids) < target_tokens:
        repeats *= 2
        token_ids = list(tokenizer.encode(unit * repeats))
    token_ids = token_ids[:target_tokens]
    return str(tokenizer.decode(token_ids)), token_ids


def _context_prompt_to_token_count(
    tokenizer: Any,
    *,
    context: str,
    instruction: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """Fill an exact prompt budget with context and one intact tail instruction."""

    if target_tokens <= 0:
        raise ValueError("prompt token target must be positive")
    tail_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
    if len(tail_ids) >= target_tokens:
        raise ValueError("tail instruction does not fit inside prompt token target")
    context_unit = context.rstrip() + "\n"
    context_ids = list(tokenizer.encode(context_unit))
    if not context_ids:
        raise ValueError("context must encode to at least one token")
    context_budget = target_tokens - len(tail_ids)
    repeats = (context_budget + len(context_ids) - 1) // len(context_ids)
    token_ids = (context_ids * repeats)[:context_budget] + tail_ids
    return str(tokenizer.decode(token_ids)), token_ids


def _token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _generation_metrics(stats: Any) -> dict[str, int | float]:
    peak = int(stats.peak_memory_bytes)
    return {
        "prefill_tokens": int(stats.new_prefill_tokens),
        "prefill_time_s": float(stats.prompt_target_prefill_time_s),
        "prefill_tok_s": float(stats.prompt_target_prefill_tok_s),
        "decode_tok_s": float(stats.decode_tok_s),
        "peak_memory_bytes": peak,
        "peak_memory_gib": peak / 2**30,
    }


def _correctness_summary(
    arms: list[dict[str, Any]],
    *,
    route_ids: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Require exact output and schedule replay for the retained route."""

    cross_route_token_exact = len({arm["token_hash"] for arm in arms}) == 1

    def schedule_fingerprint(arm: dict[str, Any]) -> tuple[Any, ...] | None:
        attempted = tuple(arm.get("attempted_depth_schedule") or ())
        accepted = tuple(arm.get("accepted_depth_schedule") or ())
        if attempted or accepted:
            return ("events", attempted, accepted)
        drafted_by_depth = tuple(arm.get("drafted_by_depth") or ())
        accepted_by_depth = tuple(arm.get("accepted_by_depth") or ())
        if drafted_by_depth or accepted_by_depth:
            return ("depth_histograms", drafted_by_depth, accepted_by_depth)
        return None

    schedule_fingerprints = [schedule_fingerprint(arm) for arm in arms]
    cross_route_schedule_exact = bool(
        schedule_fingerprints
        and all(value is not None for value in schedule_fingerprints)
        and len(set(schedule_fingerprints)) == 1
    )
    per_route_deterministic = {
        route_id: len(
            {
                (arm["token_hash"], schedule_fingerprint(arm))
                for arm in arms
                if arm["route_id"] == route_id
            }
        )
        == 1
        and all(
            schedule_fingerprint(arm) is not None
            for arm in arms
            if arm["route_id"] == route_id
        )
        for route_id in route_ids
    }
    full_output = all(int(arm["generated_tokens"]) == max_tokens for arm in arms)
    deterministic = all(per_route_deterministic.values())
    passed = bool(full_output and deterministic)
    exact = bool(cross_route_token_exact and cross_route_schedule_exact)
    return {
        "passed": passed,
        "mode": "exact" if exact else ("deterministic_drift" if passed else "rejected"),
        "full_output": full_output,
        "cross_route_token_exact": cross_route_token_exact,
        "cross_route_schedule_exact": cross_route_schedule_exact,
        "schedule_capture": (
            "events"
            if schedule_fingerprints
            and all(value is not None and value[0] == "events" for value in schedule_fingerprints)
            else "depth_histograms"
            if schedule_fingerprints
            and all(
                value is not None and value[0] == "depth_histograms"
                for value in schedule_fingerprints
            )
            else "missing"
        ),
        "per_route_deterministic": per_route_deterministic,
    }


def _validate_route_id(route_id: str) -> set[str]:
    features = {item for item in route_id.split("+") if item}
    allowed = {
        "control",
        "kv_only_history",
        "dual_norm",
        "source_proposal",
        "r08_device_draft",
    }
    unknown = features - allowed
    if not features or unknown:
        raise ValueError(f"unknown route features: {sorted(unknown)}")
    if "control" in features and len(features) != 1:
        raise ValueError("control cannot be combined with candidate features")
    return features


def _route_execution_options(route_id: str) -> dict[str, Any]:
    """Translate chronological proposal features into one cumulative run."""

    features = _validate_route_id(route_id)
    source_rows: list[int] = []
    if "r08_device_draft" in features:
        source_rows.append(8)
    return {
        "cache_route": (
            "kv_only_history" if "kv_only_history" in features else "control"
        ),
        "dual_norm": "dual_norm" in features,
        "source_proposal": "source_proposal" in features,
        "draft_core": "device" if "r08_device_draft" in features else "stock",
        "source_rows": tuple(source_rows),
    }


def _promotion_decision(
    *,
    order: list[str],
    control_id: str | None,
    candidate_id: str | None,
    improvement_pct: float | None,
    correctness: dict[str, Any],
    source_status: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_order = (
        [control_id, candidate_id, candidate_id, control_id]
        if control_id is not None and candidate_id is not None
        else []
    )
    if order != expected_order:
        errors.append("gate requires exactly four timed ABBA arms")
    if control_id is None or candidate_id is None:
        errors.append("gate requires explicit control and candidate routes")
    else:
        control_features = _validate_route_id(control_id) - {"control"}
        candidate_features = _validate_route_id(candidate_id) - {"control"}
        if not control_features < candidate_features:
            errors.append("candidate route must strictly extend the cumulative control")
    if improvement_pct is None or improvement_pct <= PROMOTION_THRESHOLD_PCT:
        errors.append(
            "candidate improvement must be strictly greater than "
            f"{PROMOTION_THRESHOLD_PCT:.2f}%"
        )
    if not bool(correctness.get("passed")):
        errors.append("correctness/determinism gate did not pass")
    if source_status:
        errors.append("promotion receipt requires a clean source tree")
    return {
        "passed": not errors,
        "threshold_pct": PROMOTION_THRESHOLD_PCT,
        "errors": errors,
    }


def _projection_counter_snapshot() -> dict[str, dict[str, int]]:
    from mtplx.qwen38_challenge_kernels import qwen38_dual_norm_counter_snapshot
    from mtplx.qwen38_source_proposal import qwen38_source_counter_snapshot

    return {
        "dual_norm": {"calls": qwen38_dual_norm_counter_snapshot()},
        "source_proposal": qwen38_source_counter_snapshot(),
    }


def _counter_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        family: {
            key: int(value) - int(before.get(family, {}).get(key, 0))
            for key, value in values.items()
        }
        for family, values in after.items()
    }


def _load_optimized_speed_stack(
    model_path: Path,
    runtime_contract: dict[str, Any],
    *,
    apply_profile_env_fn: Any = None,
    load_runtime_fn: Any = None,
    install_draft_head_fn: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct the same Turbo/Q4 draft stack used by Optimized-Speed serving."""

    from mtplx.draft_lm_head import draft_lm_head_spec_from_runtime_contract
    from mtplx.draft_sampling import draft_sampler_spec_from_runtime_contract
    from mtplx.profiles import (
        get_profile,
        runtime_env_overrides_from_contract,
    )

    profile = get_profile("turbo")
    fallback_head = {
        "bits": int(profile.draft_lm_head.bits),
        "group_size": int(profile.draft_lm_head.group_size),
        "mode": str(profile.draft_lm_head.mode),
    }
    draft_head = draft_lm_head_spec_from_runtime_contract(
        runtime_contract,
        fallback=fallback_head,
    )
    if draft_head is None:  # pragma: no cover - Turbo always has this requirement
        raise RuntimeError("Turbo profile requires a draft-only LM head")
    draft_sampler = draft_sampler_spec_from_runtime_contract(runtime_contract)
    runtime_env_overrides = runtime_env_overrides_from_contract(runtime_contract)

    if apply_profile_env_fn is None:
        from mtplx.profiles import apply_profile_env

        apply_profile_env_fn = apply_profile_env
    apply_profile_env_fn(
        profile.name,
        runtime_env_overrides=runtime_env_overrides,
    )

    # Runtime modules that bind env-gated kernels are deliberately imported
    # only after the production profile has populated the process environment.
    if load_runtime_fn is None:
        from mtplx.runtime import load

        load_runtime_fn = load
    runtime = load_runtime_fn(model_path, mtp=True)
    if install_draft_head_fn is None:
        from mtplx.draft_lm_head import _install_draft_lm_head

        install_draft_head_fn = _install_draft_lm_head
    draft_head_report = install_draft_head_fn(
        runtime,
        bits=int(draft_head["bits"]),
        group_size=int(draft_head["group_size"]),
        mode=str(draft_head["mode"]),
    )
    return runtime, {
        "profile": profile.name,
        "runtime_profile": profile.runtime_profile,
        "runtime_env": {**profile.env_dict(), **runtime_env_overrides},
        "draft_lm_head": draft_head,
        "draft_lm_head_report": draft_head_report,
        "draft_sampler": draft_sampler,
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }


def _run_arm(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    prompt_ids: list[int],
    *,
    route_id: str,
    max_tokens: int,
    seed: int,
    target_temperature: float,
    draft_temperature: float,
    source_artifact_path: Path | None,
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.generation import generate_mtpk
    from mtplx.qwen38_challenge import install_qwen38_route
    from mtplx.sampling import SamplerConfig

    options = _route_execution_options(route_id)
    route = install_qwen38_route(
        runtime,
        config,
        model_path,
        cache_route=str(options["cache_route"]),
        dual_norm=bool(options["dual_norm"]),
        source_proposal=bool(options["source_proposal"]),
        source_artifact_path=source_artifact_path,
    )
    target_sampler = SamplerConfig(
        temperature=target_temperature,
        top_p=0.95,
        top_k=20,
    )
    draft_sampler = SamplerConfig(
        temperature=draft_temperature,
        top_p=0.95,
        top_k=20,
    )
    counters_before = _projection_counter_snapshot()
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=target_sampler,
        draft_sampler=draft_sampler,
        speculative_depth=3,
        seed=seed,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        draft_core=str(options["draft_core"]),
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        mtp_history_policy="committed",
    )
    wall_s = time.perf_counter() - started
    counters_after = _projection_counter_snapshot()
    stats = output.stats
    return {
        **_generation_metrics(stats),
        "route_id": route_id,
        "installed_route_id": route.route_id,
        "route_fingerprint": hashlib.sha256(
            f"{route.fingerprint}:{route_id}".encode()
        ).hexdigest(),
        "kernel_ids": list(route.kernel_ids),
        "feature_receipt": dict(
            getattr(runtime, "qwen38_feature_receipt", {}) or {}
        ),
        "source_rows": list(options["source_rows"]),
        "draft_core": str(options["draft_core"]),
        "engagement": _counter_delta(counters_before, counters_after),
        "wall_s": wall_s,
        "generated_tokens": int(stats.generated_tokens),
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
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument(
        "--order",
        default="control,kv_only_history,kv_only_history,control",
    )
    parser.add_argument("--control-route")
    parser.add_argument("--candidate-route")
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=1024,
        help="Full-output conditioning tokens per route before timed arms.",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # The ordinary runtime installs the retained production route and drops
    # its controls. This harness must switch both directions inside one process
    # for ABBA, so it installs candidates explicitly after the common load.
    os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] = "1"
    model_path = args.model.expanduser().resolve()
    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _verify_parent_guard_attestation,
    )

    guarded_by_parent = _verify_parent_guard_attestation(args.lock)
    lock_handle = None
    if not guarded_by_parent:
        lock_handle = args.lock.open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"GPU lock is busy: {args.lock}") from exc

    from mtplx.backends.registry import load_runtime_contract

    contract, contract_error = load_runtime_contract(model_path)
    if contract_error is not None:
        raise RuntimeError(f"invalid runtime contract: {contract_error}")
    runtime_contract = {} if contract is None else contract.raw
    runtime, optimized_stack = _load_optimized_speed_stack(
        model_path,
        runtime_contract,
    )
    from mtplx.artifacts import load_config

    config = load_config(model_path)
    draft_temperature = (
        float(args.draft_temperature)
        if args.draft_temperature is not None
        else float((optimized_stack.get("draft_sampler") or {}).get("temperature", 1.0))
    )
    prompt_id, prompt = _read_prompt(args.prompt_file)
    if args.prompt_tokens is None:
        prompt_ids = list(runtime.tokenizer.encode(prompt))
    elif args.context_file is not None:
        prompt, prompt_ids = _context_prompt_to_token_count(
            runtime.tokenizer,
            context=args.context_file.read_text(encoding="utf-8"),
            instruction=prompt,
            target_tokens=args.prompt_tokens,
        )
    else:
        prompt, prompt_ids = _expand_prompt_to_token_count(
            runtime.tokenizer,
            prompt,
            args.prompt_tokens,
        )
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    if not order:
        raise ValueError("order must contain at least one route")
    for item in order:
        _validate_route_id(item)
    unique_routes = list(dict.fromkeys(order))

    warmups = [
        _run_arm(
            runtime,
            config,
            model_path,
            prompt_ids,
            route_id=route_id,
            max_tokens=args.warmup_tokens,
            seed=args.seed,
            target_temperature=args.target_temperature,
            draft_temperature=draft_temperature,
            source_artifact_path=args.source_artifact,
        )
        for route_id in unique_routes
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
            target_temperature=args.target_temperature,
            draft_temperature=draft_temperature,
            source_artifact_path=args.source_artifact,
        )
        for route_id in order
    ]
    correctness = _correctness_summary(
        arms,
        route_ids=unique_routes,
        max_tokens=args.max_tokens,
    )
    exact = bool(
        correctness["cross_route_token_exact"]
        and correctness["cross_route_schedule_exact"]
    )
    by_route = {
        route_id: [arm["wall_s"] for arm in arms if arm["route_id"] == route_id]
        for route_id in unique_routes
    }
    means = {
        route_id: sum(values) / len(values)
        for route_id, values in by_route.items()
        if values
    }
    control_id = args.control_route
    candidate_id = args.candidate_route
    if control_id is None and candidate_id is None and len(unique_routes) == 2:
        control_id, candidate_id = unique_routes
    if (control_id is None) != (candidate_id is None):
        raise ValueError("control-route and candidate-route must be supplied together")
    if control_id is not None and (
        control_id not in unique_routes or candidate_id not in unique_routes
    ):
        raise ValueError("control-route and candidate-route must occur in order")
    improvement_pct = None
    if control_id is not None and candidate_id is not None:
        improvement_pct = (means[control_id] / means[candidate_id] - 1.0) * 100.0
    source_status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    promotion = _promotion_decision(
        order=order,
        control_id=control_id,
        candidate_id=candidate_id,
        improvement_pct=improvement_pct,
        correctness=correctness,
        source_status=source_status,
    )
    receipt = {
        "kind": "qwen38_challenge_port_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model_path),
        "prompt_file": str(args.prompt_file.resolve()),
        "context_file": (
            None if args.context_file is None else str(args.context_file.resolve())
        ),
        "context_sha256": (
            None
            if args.context_file is None
            else hashlib.sha256(args.context_file.read_bytes()).hexdigest()
        ),
        "prompt_id": prompt_id,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256": _token_hash(prompt_ids),
        "prompt_token_target": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "target_temperature": args.target_temperature,
        "draft_temperature": draft_temperature,
        "optimized_speed_stack": optimized_stack,
        "order": order,
        "gpu_lock_scope": "attested_parent" if guarded_by_parent else "direct",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_version": importlib.metadata.version("mlx"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_status": source_status,
        "exact": exact,
        "token_exact": correctness["cross_route_token_exact"],
        "schedule_exact": correctness["cross_route_schedule_exact"],
        "correctness": correctness,
        "control_route_id": control_id,
        "candidate_route_id": candidate_id,
        "mean_wall_s": means,
        "candidate_improvement_pct": improvement_pct,
        "promotion": promotion,
        "warmups": warmups,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "exact": exact,
                "candidate_improvement_pct": improvement_pct,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if promotion["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
