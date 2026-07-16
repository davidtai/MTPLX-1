#!/usr/bin/env python3
"""Probe MLX compile buckets after SessionBank serving QA is green.

This is intentionally diagnostic. It checks two CUDA-graph-like bucket ideas
without making either one a prerequisite for the SessionBank fix:

* fixed prefill chunk shapes, defaulting to 128/256/512/1024 tokens
* D3/D4 graphbank verify buckets through the existing capture-commit path
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.benchmarks.runners.mtp_depth_sweep import _parse_depths
from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.generation import generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig
from mtplx.draft_lm_head import _install_draft_lm_head


FAST_PATH_ENV = {
    "MTPLX_LAZY_VERIFY_LOGITS": "1",
    "MTPLX_BATCH_TARGET_ARRAYS": "1",
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
}


def _csv_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out or any(item < 1 for item in out):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return out


def _eval_tree(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    elif isinstance(value, list):
        mx.eval(*value)
    else:
        mx.eval(value)


def _time_samples(
    fn: Callable[[Any], Any], input_ids: Any, *, repeats: int
) -> dict[str, Any]:
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        result = fn(input_ids)
        _eval_tree(result)
        samples.append(time.perf_counter() - started)
    return {
        "samples_s": samples,
        "mean_s": statistics.mean(samples),
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
    }


def _extend_tokens(token_ids: list[int], needed: int) -> tuple[list[int], bool]:
    if len(token_ids) >= needed:
        return token_ids, False
    if not token_ids:
        raise ValueError("encoded prompt produced no tokens")
    repeated = list(token_ids)
    while len(repeated) < needed:
        repeated.extend(token_ids)
    return repeated[:needed], True


def _probe_prefill_chunks(
    rt: Any,
    token_ids: list[int],
    *,
    chunks: list[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_chunk = max(chunks)
    tokens, synthetic_repeat = _extend_tokens(token_ids, max_chunk)

    for chunk in chunks:
        input_ids = mx.array([tokens[:chunk]], dtype=mx.int32)
        router_epoch_slots = int(getattr(rt, "_hy3_target_router_epoch_slots", 0))
        router_epoch_enabled = router_epoch_slots > 0 and 1 <= int(chunk) <= 8

        def eager_fn(ids):
            return rt.forward_ar(ids, cache=None, return_hidden=False)

        if router_epoch_enabled:

            def compiled_body(ids, router_epoch_block):
                return rt.forward_ar(
                    ids,
                    cache=None,
                    return_hidden=False,
                    _hy3_router_epoch_block=router_epoch_block,
                )

        else:

            def compiled_body(ids):
                return rt.forward_ar(ids, cache=None, return_hidden=False)

        compiled_error = None
        compile_time_s = None
        compiled_fn = None
        try:
            started = time.perf_counter()
            compiled_fn = mx.compile(compiled_body)
            if router_epoch_enabled:

                def compiled_call(ids):
                    return compiled_fn(
                        ids,
                        rt._new_hy3_router_epoch_block(router_epoch_slots),
                    )

            else:
                compiled_call = compiled_fn
            for _ in range(max(1, warmup)):
                _eval_tree(compiled_call(input_ids))
            compile_time_s = time.perf_counter() - started
        except Exception as exc:  # pragma: no cover - depends on local MLX compiler
            compiled_error = repr(exc)

        eager = _time_samples(eager_fn, input_ids, repeats=repeats)
        compiled = (
            _time_samples(compiled_call, input_ids, repeats=repeats)
            if compiled_fn is not None and compiled_error is None
            else None
        )
        rows.append(
            {
                "chunk_tokens": chunk,
                "synthetic_repeat": synthetic_repeat,
                "compile_time_s": compile_time_s,
                "compiled_error": compiled_error,
                "dynamic_router_epoch_input": router_epoch_enabled,
                "eager": eager,
                "compiled": compiled,
                "speedup_compiled_vs_eager": (
                    eager["median_s"] / compiled["median_s"]
                    if compiled is not None and compiled["median_s"] > 0
                    else None
                ),
            }
        )
    return rows


def _probe_verify_buckets(
    rt: Any,
    token_ids: list[int],
    *,
    depths: list[int],
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    verify_core: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sampler = SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)
    for depth in depths:
        out = generate_mtpk(
            rt,
            token_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            speculative_depth=depth,
            seed=seed,
            mtp_hidden_variant="post_norm",
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
            verify_strategy="graphbank_capture_commit",
            verify_core=verify_core,
        )
        stats = out.stats.to_dict()
        rows.append(
            {
                "depth": depth,
                "generated_tokens": stats.get("generated_tokens"),
                "tok_s": stats.get("tok_s"),
                "verify_calls": stats.get("verify_calls"),
                "verify_time_s": stats.get("verify_time_s"),
                "accepted_by_depth": stats.get("accepted_by_depth"),
                "drafted_by_depth": stats.get("drafted_by_depth"),
                "correction_tokens": stats.get("correction_tokens"),
                "bonus_tokens": stats.get("bonus_tokens"),
                "graphbank": stats.get("graphbank"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP"
    )
    parser.add_argument("--prompts", default="mtplx/benchmarks/prompts/long_code.jsonl")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument(
        "--prefill-chunks", type=_csv_ints, default=[128, 256, 512, 1024]
    )
    parser.add_argument("--depths", default="3,4")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for key, value in FAST_PATH_ENV.items():
        os.environ.setdefault(key, value)

    rt = load(args.model, mtp=True, contract=MTPContract())
    draft_lm_head = _install_draft_lm_head(rt, bits=4, group_size=64, mode="affine")
    cases = load_prompt_suite(args.prompts)
    case = cases[int(args.prompt_index)]
    prompt_ids = encode_prompt_case(
        rt.tokenizer,
        case,
        chat_template=True,
        enable_thinking=False if args.disable_thinking else None,
    )

    result = {
        "model": args.model,
        "prompt_suite": str(args.prompts),
        "prompt_index": int(args.prompt_index),
        "prompt_id": case.id,
        "prompt_tokens": len(prompt_ids),
        "fast_path_env": {key: os.environ.get(key) for key in FAST_PATH_ENV},
        "draft_lm_head": draft_lm_head,
        "prefill_chunks": []
        if args.skip_prefill
        else _probe_prefill_chunks(
            rt,
            prompt_ids,
            chunks=list(args.prefill_chunks),
            repeats=args.repeats,
            warmup=args.warmup,
        ),
        "verify_buckets": []
        if args.skip_verify
        else _probe_verify_buckets(
            rt,
            prompt_ids,
            depths=_parse_depths(args.depths),
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            verify_core=args.verify_core,
        ),
        "notes": [
            "diagnostic only",
            "mx.compile is not a prerequisite for the SessionBank serving fix",
        ],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
