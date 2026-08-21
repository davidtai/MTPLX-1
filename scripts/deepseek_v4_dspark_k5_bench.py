#!/usr/bin/env python3
"""Guarded real-checkpoint gates for DeepSeek DSpark through DFlash2."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_MODEL = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1"
)
DEFAULT_PROMPT = "Write a Python function that returns the first n Fibonacci numbers."
MIA_SOURCE_REVISION = "d4ba142bc1d971eb73a911e207e3e963bbb3c455"
MIA_MODEL_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
MIA_SOURCE_CONFIG_SHA256 = (
    "b001ec8308044aa11daa0e624f5aea5e5362a63c05879a83a7be046b00eada82"
)
MIA_SOURCE_INDEX_SHA256 = (
    "61af5c0782a8651ef893004e84369d2281a0fc316c8bcefc0bd8f76244224649"
)
MIA_IMAGE_DIGEST = (
    "sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
)
DFLASH_REVISION = "308672c08a04184cd075742db6db83ef6233296c"


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


def _reset_peak_before_arm() -> bool:
    """Drop probe temporaries, then measure the cold generation arm itself."""

    import mlx.core as mx

    gc.collect()
    clear = getattr(mx, "clear_cache", None)
    if callable(clear):
        clear()
    reset = getattr(mx, "reset_peak_memory", None)
    if not callable(reset):
        return False
    reset()
    return True


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_contract(bundle, *, capacity_tokens: int) -> dict:
    target = bundle.target_ops.make_cache(
        bundle.target_model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        cache_capacity_tokens=int(capacity_tokens),
    )
    draft = bundle.draft_backend.make_cache(
        draft_model=bundle.draft_model,
        sink_size=0,
        window_size=int(bundle.draft_model.args.sliding_window),
        allow_full_context_layers=False,
    )
    target_ok = all(
        cache.window.mode == "nvfp4_stock432"
        and cache.window.record_bytes == 432
        and (
            cache.compress_ratio == 0
            or (
                cache.compressed.mode == "nvfp4_stock432_paged"
                and cache.compressed.record_bytes == 432
                and cache.compressed.capacity
                == (int(capacity_tokens) + cache.compress_ratio - 1)
                // cache.compress_ratio
            )
        )
        and (
            cache.compress_ratio != 4
            or (
                cache.index_compressed.mode
                == "fp8_e4m3_ue8m0_scale132_paged"
                and cache.index_compressed.record_bytes == 132
            )
        )
        for cache in target
    )
    draft_ok = all(
        cache.ring.mode == "nvfp4_stock432" and cache.ring.record_bytes == 432
        for cache in draft
    )
    contract = {
        "target_kv": {
            "mode": "nvfp4_stock432",
            "record_bytes": 432,
            "start": 0,
            "layers": len(target),
            "capacity_tokens": int(capacity_tokens),
            "paged_compressed_layers": sum(
                int(cache.compress_ratio > 0) for cache in target
            ),
            "paged_indexer_layers": sum(
                int(cache.compress_ratio == 4) for cache in target
            ),
            "indexer_record_bytes": 132,
        },
        "dspark_kv": {
            "mode": "nvfp4_stock432",
            "record_bytes": 432,
            "start": 0,
            "stages": len(draft),
        },
    }
    bundle.target_ops.cleanup_generation_caches(target, draft)
    if not target_ok or not draft_ok:
        raise RuntimeError("DeepSeek DFlash2 bundle does not own Mia stock432 K/V")
    return contract


def _prompt_ids(bundle, text: str, target_tokens: int | None = None) -> list[int]:
    encoded = [int(value) for value in bundle.tokenizer.encode(text)]
    if target_tokens is None:
        return encoded
    if target_tokens <= 0 or not encoded:
        raise ValueError("prompt token target requires positive size and non-empty text")
    return (encoded * ((target_tokens + len(encoded) - 1) // len(encoded)))[
        :target_tokens
    ]


def _arm_payload(output) -> dict:
    stats = output.stats.to_dict()
    prompt_tokens = int((stats["events"][-1] or {}).get("prompt_token_count", 0))
    prompt_time = stats["prompt_eval_time_s"]
    return {
        "tokens": list(output.tokens),
        "token_digest": _token_digest(list(output.tokens)),
        "generated_tokens": len(output.tokens),
        "prompt_tokens": prompt_tokens,
        "prompt_time_s": prompt_time,
        "prefill_tok_s": prompt_tokens / prompt_time if prompt_time > 0 else 0.0,
        "decode_time_s": stats["decode_elapsed_s"],
        "elapsed_s": stats["elapsed_s"],
        "decode_tok_s": stats["decode_tok_s"],
        "milliseconds_per_token": (
            1000.0 * stats["decode_elapsed_s"] / len(output.tokens)
            if output.tokens
            else 0.0
        ),
        "peak_memory_bytes": stats["peak_memory_bytes"],
        "accepted_future_tokens": stats["accepted_drafts"],
        "drafted_future_tokens": stats["drafted_tokens"],
        "events": stats["events"],
    }


def _target_ar(bundle, prompt_ids: list[int], max_tokens: int) -> dict:
    from mtplx.generation import generate_ar
    from mtplx.sampling import SamplerConfig

    return _arm_payload(
        generate_ar(
            bundle.runtime,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            seed=0,
            stop_token_ids=set(),
        )
    )


def _dspark(bundle, prompt_ids: list[int], max_tokens: int, context) -> dict:
    from mtplx.deepseek_v4_dflash2 import generate_deepseek_v4_dflash2

    return _arm_payload(
        generate_deepseek_v4_dflash2(
            bundle,
            prompt_ids,
            max_tokens=max_tokens,
            stop_token_ids=[],
            runtime_context=context,
        )
    )


def _first_epoch(bundle, prompt_ids: list[int], context) -> dict:
    from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
    from dflash_mlx.engine.events import CycleCompleteEvent, SummaryEvent
    from dflash_mlx.runtime import stream_dflash_generate

    profiled = replace(
        context,
        diagnostics=DiagnosticsConfig(
            mode="full",
            trace=TraceConfig(cycle_events=True),
        ),
    )
    first = None
    summary = None
    for event in stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=prompt_ids,
        prompt="",
        use_chat_template=False,
        max_new_tokens=6,
        block_tokens=6,
        stop_token_ids=[],
        quantize_kv_cache=False,
        runtime_context=profiled,
    ):
        if first is None and isinstance(event, CycleCompleteEvent):
            first = event
        if isinstance(event, SummaryEvent):
            summary = event
    if first is None or summary is None:
        raise RuntimeError("DFlash2 did not emit a cycle and summary")
    payload = first.to_payload()
    payload["physical_verify_width"] = int(first.verify_token_count or 0)
    payload["future_draft_count"] = max(0, len(first.proposed_ids or ()) - 1)
    payload["summary"] = summary.to_payload()
    return payload


def _write(payload: dict, output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(output)


def _deepseek_quality_gate(candidate_tokens: list[int], control_tokens: list[int]) -> dict:
    """Reuse the established real-weight DeepSeek bf16/fp32 parity policy."""

    from deepseek_v4_mtpk_bench import _divergence, _exactness_is_enforced

    gate = _divergence(candidate_tokens, control_tokens)
    gate["enforced"] = _exactness_is_enforced(False)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--arm",
        choices=("construct", "one-cycle", "dspark", "exact-stream", "bracket"),
        required=True,
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--profile-cycles",
        action="store_true",
        help="Enable stock DFlash2 cycle and top-logit diagnostics for a failing gate.",
    )
    args = parser.parse_args()

    guard = _guard_before_mlx()
    if args.profile_cycles:
        os.environ["DFLASH_CAPTURE_LOGITS"] = "1"
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mtplx.benchmarks.dflash2_runtime import (
        build_deepseek_v4_dflash2_runtime_context,
        load_mtplx_deepseek_v4_dflash2_bundle,
    )
    from mtplx.deepseek_v4_dspark_artifact import open_verified_dspark_artifact

    artifact = open_verified_dspark_artifact(args.model)
    load_started = time.perf_counter()
    bundle = load_mtplx_deepseek_v4_dflash2_bundle(str(args.model))
    load_time = time.perf_counter() - load_started
    context = build_deepseek_v4_dflash2_runtime_context()
    if args.profile_cycles:
        from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig

        context = replace(
            context,
            diagnostics=DiagnosticsConfig(
                mode="full",
                trace=TraceConfig(cycle_events=True),
            ),
        )
    parameters = tree_flatten(bundle.target_model.parameters())
    resident_bytes = sum(int(value.nbytes) for _name, value in parameters)
    prompt_ids = _prompt_ids(bundle, args.prompt, args.prompt_tokens)
    cache_capacity_tokens = len(prompt_ids) + max(0, int(args.max_tokens))
    os.environ["MTPLX_CONTEXT_WINDOW_TOKENS"] = str(cache_capacity_tokens)
    common = {
        "schema_version": 2,
        "kind": "deepseek_v4_dspark_dflash2_k5",
        "arm": args.arm,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "mia_source": "MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark",
        "mia_source_revision": MIA_SOURCE_REVISION,
        "mia_model": "0xSero/deepseek-v4-flash-0731-spark",
        "mia_model_revision": MIA_MODEL_REVISION,
        "mia_source_config_sha256": MIA_SOURCE_CONFIG_SHA256,
        "mia_source_index_sha256": MIA_SOURCE_INDEX_SHA256,
        "mia_runtime_image_digest": MIA_IMAGE_DIGEST,
        "config_sha256": artifact.config_sha256,
        "index_sha256": artifact.index_sha256,
        "target_artifact_index_sha256": _file_digest(
            args.model / "model.safetensors.index.json"
        ),
        "draft_artifact_index_sha256": artifact.index_sha256,
        "engine": "dflash_mlx_0_1_10",
        "dflash_revision": DFLASH_REVISION,
        "physical_verify_width": bundle.checkpoint_block_size,
        "future_draft_count": bundle.checkpoint_block_size - 1,
        "dspark_stages": len(bundle.target_model.dspark.stages),
        "target_taps": list(bundle.target_layer_ids),
        "load_time_s": load_time,
        "resident_parameter_bytes": resident_bytes,
        "active_memory_bytes_after_load": _memory("get_active_memory"),
        "peak_memory_bytes_after_load": _memory("get_peak_memory"),
        "mlx_version": mx.__version__,
        "fp32_activations": (
            (os.environ.get("MTPLX_DSV4_FP32_ACTIVATIONS") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        "guard_window_id": guard["window_id"],
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        **_cache_contract(bundle, capacity_tokens=cache_capacity_tokens),
    }
    common["peak_memory_reset_before_arm"] = _reset_peak_before_arm()
    common["active_memory_bytes_before_arm"] = _memory("get_active_memory")
    common["peak_memory_bytes_before_arm"] = _memory("get_peak_memory")
    status = 0
    if args.arm == "construct":
        payload = common
    elif args.arm == "one-cycle":
        payload = {
            **common,
            "prompt_tokens": len(prompt_ids),
            "cycle": _first_epoch(bundle, prompt_ids, context),
        }
    elif args.arm == "dspark":
        payload = {
            **common,
            "prompt_tokens": len(prompt_ids),
            "dspark": _dspark(bundle, prompt_ids, args.max_tokens, context),
        }
    elif args.arm == "exact-stream":
        ar = _target_ar(bundle, prompt_ids, args.max_tokens)
        dspark = _dspark(bundle, prompt_ids, args.max_tokens, context)
        quality = _deepseek_quality_gate(dspark["tokens"], ar["tokens"])
        status = 0 if quality["pass"] or not quality["enforced"] else 2
        payload = {
            **common,
            "exact": quality["pass"],
            "quality_gate": quality,
            "ar": ar,
            "dspark": dspark,
        }
    else:
        control_0 = _target_ar(bundle, prompt_ids, args.max_tokens)
        candidate = _dspark(bundle, prompt_ids, args.max_tokens, context)
        control_1 = _target_ar(bundle, prompt_ids, args.max_tokens)
        controls_exact = control_0["tokens"] == control_1["tokens"]
        quality = _deepseek_quality_gate(candidate["tokens"], control_0["tokens"])
        status = (
            0
            if controls_exact and (quality["pass"] or not quality["enforced"])
            else 2
        )
        controls = (control_0["decode_tok_s"] + control_1["decode_tok_s"]) / 2
        payload = {
            **common,
            "exact": controls_exact and quality["pass"],
            "controls_exact": controls_exact,
            "quality_gate": quality,
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
