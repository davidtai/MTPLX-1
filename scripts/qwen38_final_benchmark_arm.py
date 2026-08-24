#!/usr/bin/env python3
"""One isolated arm for the final Qwen3.8 main-vs-DFlash2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
IS_PALINDROME_PROMPT = """Write Python code only. Implement `is_palindrome(text: str) -> bool` for a production utility. Ignore whitespace, punctuation, and letter case while preserving Unicode letters and digits. Use O(n) time, explain edge cases in docstrings, and include concise pytest tests for empty input, punctuation, mixed case, Unicode, and a clear negative case."""


def _sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(token)) for token in tokens).encode("ascii")
    ).hexdigest()


def _fit_prompt(tokenizer: Any, text: str, target_tokens: int) -> tuple[str, list[int]]:
    unit = text.rstrip() + "\n# Include robust tests.\n"
    ids = list(tokenizer.encode(unit))
    while len(ids) < target_tokens:
        unit += "# Keep the implementation readable and deterministic.\n"
        ids = list(tokenizer.encode(unit))
    ids = ids[:target_tokens]
    return str(tokenizer.decode(ids)), ids


def _build_exact_coding_prompt(
    tokenizer: Any,
    *,
    target_tokens: int,
    context: str,
    instruction: str,
) -> tuple[str, list[int]]:
    tail_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
    if len(tail_ids) >= target_tokens:
        raise ValueError("instruction does not fit inside prompt token target")
    context_ids = list(tokenizer.encode(context.rstrip() + "\n"))
    if not context_ids:
        raise ValueError("context must encode to at least one token")
    budget = target_tokens - len(tail_ids)
    repeats = (budget + len(context_ids) - 1) // len(context_ids)
    ids = (context_ids * repeats)[:budget] + tail_ids
    return str(tokenizer.decode(ids)), ids


def native_arm_metrics(output: Any, *, prompt_tokens: int, wall_s: float) -> dict[str, Any]:
    stats = output.stats
    peak = int(stats.peak_memory_bytes)
    cached_tokens = int(getattr(stats, "cached_tokens", 0) or 0)
    return {
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(stats.generated_tokens),
        "prefill_tokens": int(getattr(stats, "new_prefill_tokens", prompt_tokens) or prompt_tokens),
        "prefill_s": float(stats.prompt_target_prefill_time_s),
        "prefill_tps": float(stats.prompt_target_prefill_tok_s),
        "decode_elapsed_s": float(stats.decode_elapsed_s),
        "decode_tps": float(stats.decode_tok_s),
        "generation_elapsed_s": float(stats.elapsed_s),
        "wall_s": float(wall_s),
        "peak_memory_bytes": peak,
        "peak_memory_gib": peak / 2**30,
        "cached_tokens": cached_tokens,
        "prefix_cache_used": bool(cached_tokens or getattr(stats, "session_cache_hit", False)),
        "session_cache_hit": bool(getattr(stats, "session_cache_hit", False)),
        "cache_source": str(getattr(stats, "cache_source", "none")),
        "session_restore_mode": str(getattr(stats, "session_restore_mode", "cold")),
        "accepted_drafts": int(getattr(stats, "accepted_drafts", 0) or 0),
        "drafted_tokens": int(getattr(stats, "drafted_tokens", 0) or 0),
        "verify_calls": int(getattr(stats, "verify_calls", 0) or 0),
        "speculative_depth": int(getattr(stats, "speculative_depth", 0) or 0),
        "requested_speculative_depth": int(
            getattr(stats, "requested_speculative_depth", 0) or 0
        ),
        "token_sha256": _sha256_tokens(list(output.tokens)),
    }


def native_draft_sampler_values(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    contract: dict[str, Any] | None,
) -> tuple[float, float, int]:
    """Use the shipped draft sampler, except when full greedy is requested."""

    if temperature <= 0.0:
        return 0.0, float(top_p), int(top_k)
    values = contract or {}
    return (
        float(values.get("temperature", temperature)),
        float(values.get("top_p", top_p)),
        int(values.get("top_k", top_k)),
    )


def stop_token_ids_for_prompt(prompt_kind: str) -> set[int] | None:
    """Let the simple burst stop naturally; load simulations fill their cap."""

    return None if prompt_kind == "is_palindrome" else set()


def validate_candidate_adaptive_receipt(receipt: dict[str, Any]) -> None:
    route = dict(receipt.get("context_route") or {})
    if not (
        bool(route.get("requested_adaptive"))
        and bool(route.get("effective_adaptive"))
    ):
        raise RuntimeError("DFlash2 benchmark candidate is not effectively adaptive")


def _dflash_arm_metrics(
    output: Any,
    runtime: Any,
    *,
    prompt_tokens: int,
    wall_s: float,
) -> dict[str, Any]:
    stats = output.stats
    telemetry = runtime.telemetry
    peak = int(stats.peak_memory_bytes)
    return {
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(stats.generated_tokens),
        "prefill_tokens": int(prompt_tokens),
        "prefill_s": float(stats.prompt_eval_time_s),
        "prefill_tps": float(stats.prompt_tps),
        "decode_elapsed_s": float(stats.decode_elapsed_s),
        "decode_tps": float(stats.decode_tok_s),
        "generation_elapsed_s": float(stats.elapsed_s),
        "wall_s": float(wall_s),
        "peak_memory_bytes": peak,
        "peak_memory_gib": peak / 2**30,
        "cached_tokens": 0,
        "prefix_cache_used": False,
        "session_cache_hit": False,
        "cache_source": "none",
        "session_restore_mode": "cold",
        "accepted_drafts": int(stats.accepted_drafts),
        "drafted_tokens": int(stats.drafted_tokens),
        "verify_calls": int(stats.verify_calls),
        "requested_width": int(runtime.config.draft_block_size),
        "effective_widths": sorted(
            int(value) for value in (telemetry.adaptive_metrics.get("cycles_by_block") or {})
        ),
        "requested_adaptive": bool(
            runtime.qwen38_feature_receipt.get("context_route", {}).get(
                "requested_adaptive"
            )
        ),
        "effective_adaptive": bool(
            runtime.qwen38_feature_receipt.get("context_route", {}).get(
                "effective_adaptive"
            )
        ),
        "fallback_ar": False,
        "adaptive_metrics": dict(telemetry.adaptive_metrics),
        "token_sha256": _sha256_tokens(list(output.tokens)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("main_native_mtp", "pr_dflash2"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--prompt-kind", choices=("is_palindrome", "coding"), required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--conditioner-tokens", type=int, default=32)
    parser.add_argument(
        "--conditioner-mode",
        choices=("same_prompt", "unrelated_prompt"),
        default="unrelated_prompt",
    )
    parser.add_argument(
        "--dflash2-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace, tokenizer: Any) -> tuple[str, list[int]]:
    if args.prompt_kind == "is_palindrome":
        return _fit_prompt(tokenizer, IS_PALINDROME_PROMPT, args.prompt_tokens)
    row = json.loads(args.prompt_file.read_text(encoding="utf-8").splitlines()[0])
    return _build_exact_coding_prompt(
        tokenizer,
        target_tokens=args.prompt_tokens,
        context=args.context_file.read_text(encoding="utf-8"),
        instruction=str(row["prompt"]),
    )


def _load_native(model: Path) -> tuple[Any, dict[str, Any]]:
    from mtplx.backends.registry import load_runtime_contract
    from mtplx.draft_lm_head import (
        _install_draft_lm_head,
        draft_lm_head_spec_from_runtime_contract,
    )
    from mtplx.draft_sampling import draft_sampler_spec_from_runtime_contract
    from mtplx.profiles import apply_profile_env, get_profile, runtime_env_overrides_from_contract
    from mtplx.runtime import load

    contract, error = load_runtime_contract(model)
    if error is not None:
        raise RuntimeError(f"invalid runtime contract: {error}")
    raw = {} if contract is None else contract.raw
    profile = get_profile("turbo")
    apply_profile_env(
        profile.name,
        runtime_env_overrides=runtime_env_overrides_from_contract(raw),
    )
    runtime = load(model, mtp=True)
    head = draft_lm_head_spec_from_runtime_contract(
        raw,
        fallback={"bits": 4, "group_size": 64, "mode": "affine"},
    )
    if head is None:
        raise RuntimeError("Optimized-Speed model requires its Q4 draft head")
    head_report = _install_draft_lm_head(
        runtime,
        bits=int(head["bits"]),
        group_size=int(head["group_size"]),
        mode=str(head["mode"]),
    )
    return runtime, {
        "profile": profile.name,
        "draft_lm_head": head,
        "draft_lm_head_report": head_report,
        "draft_sampler": draft_sampler_spec_from_runtime_contract(raw),
    }


def _load_dflash(
    model: Path,
    draft: Path,
    *,
    draft_adaptive: bool,
) -> tuple[Any, dict[str, Any]]:
    from mtplx.profiles import apply_profile_env
    from mtplx.runtime import load

    apply_profile_env("turbo")
    os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] = "1"
    target_runtime = load(model, mtp=False)
    from mtplx.benchmarks.dflash2_runtime import bind_mtplx_dflash2_bundle
    from mtplx.backends.dflash2 import (
        DFlash2Runtime,
        DFlash2RuntimeConfig,
        _install_measured_qwen38_dflash_stack,
    )
    from dflash_mlx.runtime.context import build_offline_runtime_context

    bundle = bind_mtplx_dflash2_bundle(target_runtime, str(draft))
    config = DFlash2RuntimeConfig.from_paths(
        target_model_path=model,
        draft_model_path=draft,
        draft_block_size=8,
        draft_quantization="4bit",
        prefill_step_size=2048,
        draft_adaptive=bool(draft_adaptive),
    )
    runtime_context = build_offline_runtime_context(
        quantize_kv_cache=False,
        verify_mode="dflash",
        copyspec_mode="off",
        prefill_step_size=2048,
        verify_len_cap=8,
    )
    runtime = DFlash2Runtime(
        target_model=bundle.target_model,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        target_runtime=target_runtime,
        target_ops=bundle.target_ops,
        draft_backend=bundle.draft_backend,
        runtime_context=runtime_context,
        config=config,
    )
    runtime.qwen38_feature_receipt = _install_measured_qwen38_dflash_stack(runtime)
    validate_candidate_adaptive_receipt(runtime.qwen38_feature_receipt)
    return runtime, {
        "profile": "turbo",
        "native_mtp_loaded": False,
        "dflash_block_size": 8,
        "feature_receipt": runtime.qwen38_feature_receipt,
    }


def _generate_native(runtime: Any, prompt_ids: list[int], args: argparse.Namespace) -> Any:
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    sampler = SamplerConfig(args.temperature, args.top_p, args.top_k)
    draft_values = native_draft_sampler_values(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        contract=getattr(runtime, "_final_benchmark_draft_sampler", None),
    )
    draft_sampler = SamplerConfig(*draft_values)
    return generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=args.max_tokens,
        sampler=sampler,
        draft_sampler=draft_sampler,
        speculative_depth=3,
        seed=args.seed,
        stop_token_ids=stop_token_ids_for_prompt(args.prompt_kind),
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )


def _generate_dflash(runtime: Any, prompt_ids: list[int], args: argparse.Namespace) -> Any:
    from mtplx.backends.dflash2 import generate_dflash2
    from mtplx.sampling import SamplerConfig

    return generate_dflash2(
        runtime,
        prompt_ids,
        max_tokens=args.max_tokens,
        sampler=SamplerConfig(args.temperature, args.top_p, args.top_k),
        seed=args.seed,
        stop_token_ids=stop_token_ids_for_prompt(args.prompt_kind),
    )


def main() -> int:
    args = _parse_args()
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True
    ).strip()
    if observed_commit != args.source_commit:
        raise RuntimeError(
            f"source commit moved: expected {args.source_commit}, got {observed_commit}"
        )
    sys.path.insert(0, str(args.source_root.resolve()))
    imported_mtplx = __import__("mtplx")
    if args.source_root.resolve() not in Path(imported_mtplx.__file__).resolve().parents:
        raise RuntimeError(f"mtplx import escaped pinned source root: {imported_mtplx.__file__}")

    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _verify_parent_guard_attestation,
    )

    if not _verify_parent_guard_attestation(args.lock):
        raise RuntimeError("benchmark arm requires exclusive parent GPU lock attestation")
    runtime, stack = (
        _load_native(args.model)
        if args.engine == "main_native_mtp"
        else _load_dflash(
            args.model,
            args.draft,
            draft_adaptive=bool(args.dflash2_adaptive),
        )
    )
    if args.engine == "main_native_mtp":
        runtime._final_benchmark_draft_sampler = stack.get("draft_sampler")
    prompt_text, prompt_ids = _load_prompt(args, runtime.tokenizer)
    if len(prompt_ids) != args.prompt_tokens:
        raise RuntimeError("prompt builder missed exact cold-prefill token count")
    if args.conditioner_mode == "same_prompt":
        conditioner_text, conditioner_ids = prompt_text, list(prompt_ids)
    else:
        conditioner_text, conditioner_ids = _fit_prompt(
            runtime.tokenizer,
            "Write a Python function named clamp and include two tests.",
            64,
        )
    del prompt_text, conditioner_text

    conditioner_args = argparse.Namespace(**vars(args))
    conditioner_args.max_tokens = args.conditioner_tokens
    generate = _generate_native if args.engine == "main_native_mtp" else _generate_dflash
    generate(runtime, conditioner_ids, conditioner_args)

    import mlx.core as mx

    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = generate(runtime, prompt_ids, args)
    wall_s = time.perf_counter() - started
    if args.engine == "pr_dflash2":
        validate_candidate_adaptive_receipt(runtime.qwen38_feature_receipt)
    arm = (
        native_arm_metrics(output, prompt_tokens=len(prompt_ids), wall_s=wall_s)
        if args.engine == "main_native_mtp"
        else _dflash_arm_metrics(
            output, runtime, prompt_tokens=len(prompt_ids), wall_s=wall_s
        )
    )
    arm["engine"] = args.engine
    arm["route"] = (
        "upstream_main_optimized_speed_native_mtp"
        if args.engine == "main_native_mtp"
        else "pr_final_production_dflash2"
    )
    receipt = {
        "kind": "qwen38_final_cold_prefill_isolated_arm",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": args.engine,
        "source_root": str(args.source_root.resolve()),
        "source_commit": observed_commit,
        "mtplx_import": str(Path(imported_mtplx.__file__).resolve()),
        "model": str(args.model.resolve()),
        "draft": str(args.draft.resolve()) if args.engine == "pr_dflash2" else None,
        "workload": {
            "prompt_kind": args.prompt_kind,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": _sha256_tokens(prompt_ids),
            "output_limit": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "draft_sampler": (
                dict(
                    zip(
                        ("temperature", "top_p", "top_k"),
                        native_draft_sampler_values(
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            contract=stack.get("draft_sampler"),
                        ),
                    )
                )
                if args.engine == "main_native_mtp"
                else {"mode": "dflash2_internal_greedy_proposal"}
            ),
            "seed": args.seed,
            "cold_prefill": True,
            "prefix_cache_used": False,
            "conditioner_prompt_tokens": len(conditioner_ids),
            "conditioner_output_tokens": args.conditioner_tokens,
            "conditioner_mode": args.conditioner_mode,
            "conditioner_reuses_timed_prompt": args.conditioner_mode == "same_prompt",
            "output_count_forced": args.prompt_kind != "is_palindrome",
            "requested_adaptive": (
                bool(args.dflash2_adaptive)
                if args.engine == "pr_dflash2"
                else None
            ),
        },
        "stack": stack,
        "arm": arm,
        "mlx_version": importlib.metadata.version("mlx"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_lock_scope": str(args.lock.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "engine": args.engine,
                "prompt_tokens": len(prompt_ids),
                "prefill_tps": arm["prefill_tps"],
                "decode_tps": arm["decode_tps"],
                "wall_s": arm["wall_s"],
                "peak_memory_gib": arm["peak_memory_gib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
