#!/usr/bin/env python3
"""Matched 16K ABBA gate for replacing fixed-D3 MTP with DFlash2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen38_challenge_port_gate import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_LOCK,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    PROMOTION_THRESHOLD_PCT,
    _context_prompt_to_token_count,
    _load_optimized_speed_stack,
    _projection_counter_snapshot,
    _counter_delta,
    _read_prompt,
    _route_execution_options,
    _run_arm,
)


DFLASH_REPO = "z-lab/Qwen3.8-27B-DFlash2"
DFLASH_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DFLASH_SOURCE_COMMIT = "54644e991039110f30140006c892c57734b9311e"
STATIC_WIDTH = 8
FULL_RETAINED_ROUTE = (
    "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
    "r20_kv_only_history+r21_qk_rms_rope+r24_eval_ladder+"
    "r26_prefill_ladder_3+r36_qkv_islands+r48_boundary_fused+"
    "r50_wired_residency+r53_command_buffers+r61_dual_norm_concat"
)
DEFAULT_DFLASH_SNAPSHOT = Path.home() / (
    ".cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/"
    f"snapshots/{DFLASH_REVISION}"
)
DEFAULT_ROW36_ARTIFACT = Path.home() / (
    ".cache/huggingface/hub/"
    "models--amal-david--qwen38-mtp-head-q4-qkv-islands-v1/"
    "blobs/517bb133d7ca6e228a5129710b3cb2c25aa9944753b9f9a225fa1e8135df5e65"
)


def _token_hash(tokens: list[int] | tuple[int, ...]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _dflash_target_sampling(*, seed: int) -> Iterator[None]:
    """Install the campaign's exact target sampler at DFlash's posterior seam.

    DFlash2 still proposes greedily.  Each verified target row is sampled from
    temperature 1 / top-p .95 / top-k 20, and drafted tokens are accepted only
    while they match those target samples.  On the first mismatch the sampled
    target token is emitted, preserving the target sampler distribution.
    """

    import mlx.core as mx

    from dflash_mlx.engine import spec_epoch
    from mtplx.fast_sampling import sample_token_ids_from_mlx_logits
    from mtplx.sampling import SamplerConfig

    sampler = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    original = spec_epoch.greedy_tokens_with_mask

    def sample_target_rows(logits, suppress_token_mask=None):
        if suppress_token_mask is not None:
            raise RuntimeError(
                "Qwen 3.8 DFlash benchmark does not permit token suppression"
            )
        sampled = sample_token_ids_from_mlx_logits(logits, sampler)
        if sampled is None:
            raise RuntimeError("DFlash target sampler could not stay on device")
        return sampled.astype(mx.uint32)

    mx.random.seed(int(seed))
    spec_epoch.greedy_tokens_with_mask = sample_target_rows
    try:
        yield
    finally:
        spec_epoch.greedy_tokens_with_mask = original


def _install_retained_route(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    *,
    row36_artifact: Path,
) -> Any:
    from mtplx.qwen38_challenge import install_qwen38_route

    options = _route_execution_options(FULL_RETAINED_ROUTE)
    return install_qwen38_route(
        runtime,
        config,
        model_path,
        cache_route=str(options["cache_route"]),
        dual_norm=bool(options["dual_norm"]),
        source_proposal=False,
        row10_compact_vocab=bool(options["row10_compact_vocab"]),
        mtp_block_variant=options["mtp_block_variant"],
        mtp_block_artifact_path=row36_artifact,
        row18_gdn_decay_memo=bool(options["row18_gdn_decay_memo"]),
        row21_qk_rms_rope=bool(options["row21_qk_rms_rope"]),
        row24_eval_ladder=bool(options["row24_eval_ladder"]),
        row26_prefill_ladder_3=bool(options["row26_prefill_ladder_3"]),
        row48_boundary_fused=bool(options["row48_boundary_fused"]),
        row50_wired_residency=bool(options["row50_wired_residency"]),
        row63_q8_embedding_dual_norm=False,
        row70_qmv_sumtable=False,
        row78_qmv_active_groups=False,
        row80_qmv_m2=False,
    )


def _run_dflash_arm(
    bundle: Any,
    config: dict[str, Any],
    model_path: Path,
    prompt_ids: list[int],
    runtime_context: Any,
    *,
    max_tokens: int,
    seed: int,
    row36_artifact: Path,
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.benchmarks.runners.dflash2_depth_sweep import (
        run_dflash2_candidate,
    )

    route = _install_retained_route(
        bundle.runtime,
        config,
        model_path,
        row36_artifact=row36_artifact,
    )
    counters_before = _projection_counter_snapshot()
    mx.reset_peak_memory()
    started = time.perf_counter()
    with _dflash_target_sampling(seed=seed):
        arm = run_dflash2_candidate(
            bundle,
            prompt_ids,
            STATIC_WIDTH,
            runtime_context,
            max_tokens=max_tokens,
        )
    wall_s = time.perf_counter() - started
    counters_after = _projection_counter_snapshot()
    tokens = tuple(int(token) for token in arm.pop("tokens"))
    return {
        **arm,
        "engine": "dflash2",
        "route_id": FULL_RETAINED_ROUTE,
        "installed_route_id": route.route_id,
        "wall_s": wall_s,
        "token_hash": _token_hash(tokens),
        "tokens": list(tokens),
        "engagement": _counter_delta(counters_before, counters_after),
        "feature_receipt": dict(
            getattr(bundle.runtime, "qwen38_feature_receipt", {}) or {}
        ),
    }


def _run_mtp_arm(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    seed: int,
    row36_artifact: Path,
) -> dict[str, Any]:
    arm = _run_arm(
        runtime,
        config,
        model_path,
        prompt_ids,
        route_id=FULL_RETAINED_ROUTE,
        max_tokens=max_tokens,
        seed=seed,
        target_temperature=1.0,
        draft_temperature=1.0,
        source_artifact_path=None,
        row17_artifact_path=None,
        row28_artifact_path=None,
        row36_artifact_path=row36_artifact,
    )
    arm["engine"] = "mtp_fixed_d3"
    arm["prefill_tps"] = float(arm["prefill_tok_s"])
    arm["decode_tps"] = float(arm["decode_tok_s"])
    arm["peak_memory_gb"] = float(arm["peak_memory_gib"])
    return arm


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DFLASH_SNAPSHOT)
    parser.add_argument("--row36-artifact", type=Path, default=DEFAULT_ROW36_ARTIFACT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prompt_tokens != 16_384 or args.max_tokens != 1024:
        raise ValueError("item 55 requires exactly 16K input and 1024 output tokens")
    model_path = args.model.expanduser().resolve()
    draft_path = args.draft.expanduser().resolve()
    row36_artifact = args.row36_artifact.expanduser().resolve()
    if not draft_path.is_dir():
        raise FileNotFoundError(f"pinned DFlash snapshot is absent: {draft_path}")
    if not row36_artifact.is_file():
        raise FileNotFoundError(f"row 36 artifact is absent: {row36_artifact}")

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

    os.environ["MTPLX_QWEN38_DISABLE_SOURCE_AUTO"] = "1"
    from mtplx.backends.registry import load_runtime_contract

    runtime_contract, contract_error = load_runtime_contract(model_path)
    if contract_error is not None:
        raise RuntimeError(f"invalid runtime contract: {contract_error}")
    runtime, optimized_stack = _load_optimized_speed_stack(
        model_path,
        {} if runtime_contract is None else runtime_contract.raw,
    )

    from mtplx.artifacts import load_config
    from mtplx.benchmarks.dflash2_runtime import bind_mtplx_dflash2_bundle
    from mtplx.benchmarks.runners.dflash2_depth_sweep import (
        build_fixed_dflash_runtime_context,
    )

    config = load_config(model_path)
    bundle = bind_mtplx_dflash2_bundle(runtime, str(draft_path))
    runtime_context = build_fixed_dflash_runtime_context()
    prompt_id, instruction = _read_prompt(args.prompt_file)
    prompt_text, prompt_ids = _context_prompt_to_token_count(
        runtime.tokenizer,
        context=args.context_file.read_text(encoding="utf-8"),
        instruction=instruction,
        target_tokens=args.prompt_tokens,
    )
    del prompt_text

    def run(engine: str, tokens: int) -> dict[str, Any]:
        if engine == "mtp_fixed_d3":
            return _run_mtp_arm(
                runtime,
                config,
                model_path,
                prompt_ids,
                max_tokens=tokens,
                seed=args.seed,
                row36_artifact=row36_artifact,
            )
        return _run_dflash_arm(
            bundle,
            config,
            model_path,
            prompt_ids,
            runtime_context,
            max_tokens=tokens,
            seed=args.seed,
            row36_artifact=row36_artifact,
        )

    warmups = [
        run("mtp_fixed_d3", args.warmup_tokens),
        run("dflash2", args.warmup_tokens),
    ]
    order = ["mtp_fixed_d3", "dflash2", "dflash2", "mtp_fixed_d3"]
    arms = [run(engine, args.max_tokens) for engine in order]
    by_engine = {
        engine: [arm for arm in arms if arm["engine"] == engine]
        for engine in ("mtp_fixed_d3", "dflash2")
    }
    deterministic = {
        engine: len({arm["token_hash"] for arm in rows}) == 1
        for engine, rows in by_engine.items()
    }
    generated_exact = all(
        int(arm["generated_tokens"]) == args.max_tokens for arm in arms
    )
    dflash_contract = all(
        int(arm["requested_width"]) == STATIC_WIDTH
        and int(arm["effective_width"]) == STATIC_WIDTH
        and not bool(arm["fallback_ar"])
        for arm in by_engine["dflash2"]
    )
    mean_wall = {
        engine: _mean(rows, "wall_s") for engine, rows in by_engine.items()
    }
    improvement_pct = (
        mean_wall["mtp_fixed_d3"] / mean_wall["dflash2"] - 1.0
    ) * 100.0
    summary = {
        engine: {
            "prefill_tps": _mean(rows, "prefill_tps"),
            "decode_tps": _mean(rows, "decode_tps"),
            "peak_memory_gb": _mean(rows, "peak_memory_gb"),
            "wall_s": mean_wall[engine],
        }
        for engine, rows in by_engine.items()
    }
    exact = bool(generated_exact and dflash_contract and all(deterministic.values()))
    promoted = bool(exact and improvement_pct > PROMOTION_THRESHOLD_PCT)
    receipt = {
        "kind": "qwen38_challenge_dflash2_item55_abba",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model_path),
        "dflash": {
            "repo_id": DFLASH_REPO,
            "revision": DFLASH_REVISION,
            "snapshot": str(draft_path),
            "source_commit": DFLASH_SOURCE_COMMIT,
            "block_size": 8,
            "static_width": STATIC_WIDTH,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
        "workload": {
            "prompt_id": prompt_id,
            "prompt_tokens": len(prompt_ids),
            "prompt_token_sha256": _token_hash(prompt_ids),
            "generated_tokens": args.max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "seed": args.seed,
            "conditioning_tokens_per_engine": args.warmup_tokens,
            "timed_order": order,
        },
        "optimized_speed_stack": optimized_stack,
        "retained_route": FULL_RETAINED_ROUTE,
        "mlx_version": importlib.metadata.version("mlx"),
        "dflash_mlx_version": importlib.metadata.version("dflash-mlx"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "warmups": warmups,
        "arms": arms,
        "summary": summary,
        "correctness": {
            "per_engine_deterministic": deterministic,
            "generated_count_exact": generated_exact,
            "dflash_width_and_fallback_exact": dflash_contract,
            "cross_engine_token_exact": (
                by_engine["mtp_fixed_d3"][0]["token_hash"]
                == by_engine["dflash2"][0]["token_hash"]
            ),
            "cross_engine_token_exact_required": False,
            "exact": exact,
        },
        "candidate_improvement_pct": improvement_pct,
        "promotion": {
            "threshold_pct": PROMOTION_THRESHOLD_PCT,
            "passed": promoted,
            "reason": (
                "strict wall improvement above threshold"
                if promoted
                else "correctness or strict wall threshold failed"
            ),
        },
        "gpu_lock_scope": str(args.lock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    print(f"candidate_improvement_pct={improvement_pct:.6f}")
    print(f"promotion_passed={promoted}")
    if lock_handle is not None:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
