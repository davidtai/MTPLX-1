#!/usr/bin/env python3
"""Guarded real-checkpoint gates for the fixed DeepSeek V4 DSpark K5 port."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_MODEL = Path("/Users/davidtai/models/DeepSeek-V4-Flash-0731-2.4bit-mixed")
DEFAULT_PROMPT = "Write a Python function that returns the first n Fibonacci numbers."


def _guard_before_mlx() -> dict:
    from deepseek_v4_guard_window import (
        WINDOW_PATH_ENV,
        WINDOW_SHA256_ENV,
        issue_guard_window,
        load_verified_guard_window,
    )

    path, digest = issue_guard_window()
    os.environ[WINDOW_PATH_ENV] = str(path)
    os.environ[WINDOW_SHA256_ENV] = digest
    return load_verified_guard_window()


def _memory(getter: str) -> int:
    import mlx.core as mx

    fn = getattr(mx, getter, None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), getter, None)
    return int(fn()) if callable(fn) else 0


def _evaluate_cache_roots(caches) -> None:
    import mlx.core as mx
    from mtplx.deepseek_v4_dspark_generation import _cache_arrays

    arrays = _cache_arrays(caches)
    if arrays:
        mx.eval(*arrays)


def _cache_contract(runtime) -> dict:
    target = runtime.deepseek_v4_dspark_runtime.make_target_cache()
    draft = runtime.deepseek_v4_dspark_runtime.make_dspark_cache()
    target_ok = all(
        cache.window.bits == 4
        and cache.window.group_size == 64
        and cache.compressed.bits == 4
        and cache.compressed.group_size == 64
        for cache in target
    )
    draft_ok = all(
        cache.ring.bits == 4 and cache.ring.group_size == 64 for cache in draft
    )
    if not target_ok or not draft_ok:
        raise RuntimeError("loaded DSpark runtime does not own affine-int4 K/V")
    return {
        "target_kv": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "start": 0,
            "layers": len(target),
        },
        "dspark_kv": {
            "mode": "affine",
            "bits": 4,
            "group_size": 64,
            "start": 0,
            "stages": len(draft),
        },
    }


def _prompt_ids(runtime, text: str) -> list[int]:
    values = runtime.tokenizer.encode(text)
    return [int(value) for value in values]


def _target_ar(runtime, prompt_ids: list[int], max_tokens: int) -> dict:
    import mlx.core as mx

    installed = runtime.deepseek_v4_dspark_runtime
    cache = installed.make_target_cache()
    started = time.perf_counter()
    prompt_started = time.perf_counter()
    prefetched = installed.target_prefill(
        mx.array([prompt_ids], dtype=mx.int32),
        cache,
    )
    mx.eval(prefetched.logits)
    _evaluate_cache_roots(cache)
    prompt_time = time.perf_counter() - prompt_started
    carried = prefetched.logits[:, -1]
    tokens = []
    stop = {int(runtime.tokenizer.eos_token_id)}
    decode_started = time.perf_counter()
    while len(tokens) < max_tokens:
        token_array = mx.argmax(carried, axis=-1).astype(mx.int32)
        mx.eval(token_array)
        token = int(token_array.item())
        tokens.append(token)
        if token in stop:
            break
        stepped = installed.target_m6(token_array[:, None], cache)
        mx.eval(stepped.logits)
        _evaluate_cache_roots(cache)
        carried = stepped.logits[:, -1]
    decode_time = time.perf_counter() - decode_started
    elapsed = time.perf_counter() - started
    return {
        "tokens": tokens,
        "token_digest": hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(tokens),
        "prompt_time_s": prompt_time,
        "decode_time_s": decode_time,
        "elapsed_s": elapsed,
        "decode_tok_s": len(tokens) / decode_time if decode_time > 0 else 0.0,
        "milliseconds_per_token": (
            1000.0 * decode_time / len(tokens) if tokens else 0.0
        ),
    }


def _dspark(runtime, prompt_ids: list[int], max_tokens: int) -> dict:
    from mtplx.deepseek_v4_dspark_generation import generate_dspark

    output = generate_dspark(runtime, prompt_ids, max_tokens=max_tokens)
    stats = output.stats.to_dict()
    summary = stats["events"][0]
    return {
        "tokens": output.tokens,
        "token_digest": hashlib.sha256(
            json.dumps(output.tokens, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(output.tokens),
        "prompt_time_s": stats["prompt_eval_time_s"],
        "decode_time_s": stats["decode_elapsed_s"],
        "elapsed_s": stats["elapsed_s"],
        "decode_tok_s": stats["decode_tok_s"],
        "milliseconds_per_token": (
            1000.0 * stats["decode_elapsed_s"] / len(output.tokens)
            if output.tokens
            else 0.0
        ),
        "verify_width_histogram": summary["verify_width_histogram"],
        "accepted_depth_histogram": summary["accepted_depth_histogram"],
        "accepted_future_tokens": stats["accepted_drafts"],
        "drafted_future_tokens": stats["drafted_tokens"],
    }


def _one_cycle(runtime, prompt_ids: list[int]) -> dict:
    import mlx.core as mx
    from mtplx.deepseek_v4_dspark_generation import DeepseekV4DSparkCycle

    installed = runtime.deepseek_v4_dspark_runtime
    target_cache = installed.make_target_cache()
    draft_caches = installed.make_dspark_cache()
    prefetched = installed.target_prefill(
        mx.array([prompt_ids], dtype=mx.int32),
        target_cache,
    )
    installed.prefill_dspark(prefetched.taps, draft_caches)
    mx.eval(prefetched.logits)
    _evaluate_cache_roots(target_cache)
    _evaluate_cache_roots(draft_caches)
    target_before = int(target_cache[0].size())
    draft_before = [int(cache.prefill_length) for cache in draft_caches]
    cycle = DeepseekV4DSparkCycle(
        propose_k5=lambda primary, start_pos: installed.proposal_k5(
            primary, draft_caches, start_pos
        ),
        verify_m6=lambda ids: installed.target_m6(ids, target_cache),
        trim_target=lambda count: installed.trim_target(target_cache, count),
        commit_dspark=lambda taps, start_pos: installed.commit_dspark(
            taps, draft_caches, start_pos
        ),
    )
    result = cycle(prefetched.logits[:, -1], start_pos=len(prompt_ids))
    mx.eval(result.committed_tokens, result.next_primary)
    _evaluate_cache_roots(target_cache)
    _evaluate_cache_roots(draft_caches)
    return {
        "primary_token": int(result.primary.item()),
        "future_tokens": [int(value) for value in result.future_tokens[0].tolist()],
        "future_draft_count": int(result.future_tokens.shape[1]),
        "verify_ids": [int(value) for value in result.verify_ids[0].tolist()],
        "physical_verify_width": result.physical_verify_width,
        "accepted_future_tokens": result.accepted_future_tokens,
        "committed_tokens": [
            int(value) for value in result.committed_tokens[0].tolist()
        ],
        "next_primary": int(result.next_primary.item()),
        "target_cache_before": target_before,
        "target_cache_after": int(target_cache[0].size()),
        "target_cache_delta": int(target_cache[0].size()) - target_before,
        "dspark_cache_before": draft_before,
        "dspark_cache_after": [
            int(cache.prefill_length) for cache in draft_caches
        ],
        "dspark_cache_delta": [
            int(cache.prefill_length) - before
            for cache, before in zip(draft_caches, draft_before)
        ],
    }


def _write(payload: dict, output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(output)


def main() -> int:
    guard = _guard_before_mlx()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--arm",
        choices=("construct", "one-cycle", "exact-stream", "bracket"),
        required=True,
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mtplx.deepseek_v4_dspark_artifact import open_verified_dspark_artifact
    from mtplx.runtime import load

    artifact = open_verified_dspark_artifact(args.model)
    load_started = time.perf_counter()
    runtime = load(args.model, mtp=True, dspark=True)
    load_time = time.perf_counter() - load_started
    parameters = tree_flatten(runtime.model.parameters())
    resident_bytes = sum(int(value.nbytes) for _name, value in parameters)
    common = {
        "schema_version": 1,
        "kind": "deepseek_v4_dspark_k5_phase1",
        "arm": args.arm,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "config_sha256": artifact.config_sha256,
        "index_sha256": artifact.index_sha256,
        "dspark_stages": len(runtime.model.dspark.stages),
        "target_taps": list(artifact.config.target_layer_ids),
        "load_time_s": load_time,
        "resident_parameter_bytes": resident_bytes,
        "active_memory_bytes": _memory("get_active_memory"),
        "peak_memory_bytes": _memory("get_peak_memory"),
        "mlx_version": mx.__version__,
        "guard_window_id": guard["window_id"],
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        **_cache_contract(runtime),
    }
    prompt_ids = _prompt_ids(runtime, args.prompt)
    status = 0
    if args.arm == "construct":
        payload = common
    elif args.arm == "one-cycle":
        payload = {**common, "prompt_tokens": len(prompt_ids), "cycle": _one_cycle(runtime, prompt_ids)}
    elif args.arm == "exact-stream":
        ar = _target_ar(runtime, prompt_ids, args.max_tokens)
        dspark = _dspark(runtime, prompt_ids, args.max_tokens)
        exact = ar["tokens"] == dspark["tokens"]
        status = 0 if exact else 2
        payload = {**common, "exact": exact, "ar": ar, "dspark": dspark}
    else:
        control_0 = _target_ar(runtime, prompt_ids, args.max_tokens)
        candidate = _dspark(runtime, prompt_ids, args.max_tokens)
        control_1 = _target_ar(runtime, prompt_ids, args.max_tokens)
        exact = control_0["tokens"] == candidate["tokens"] == control_1["tokens"]
        status = 0 if exact else 2
        controls = (control_0["decode_tok_s"] + control_1["decode_tok_s"]) / 2
        payload = {
            **common,
            "exact": exact,
            "controls_mean_decode_tok_s": controls,
            "candidate_delta_percent": (
                100.0 * (candidate["decode_tok_s"] / controls - 1.0)
                if controls > 0
                else 0.0
            ),
            "arms": {"C0": control_0, "K5": candidate, "C1": control_1},
        }
    _write(payload, args.out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
