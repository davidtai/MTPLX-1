#!/usr/bin/env python3
"""Run a small AR cross-entropy sanity check for a streamed artifact.

This intentionally performs one teacher-forced sample, not a generation or
throughput benchmark.  When a baseline artifact is supplied, the gate compares
perplexity class rather than requiring token-for-token identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.expert_runtime import ExpertStreamingConfig, parse_memory_bytes  # noqa: E402
from mtplx.runtime import load  # noqa: E402


DEFAULT_TEXT = (
    "Open source model infrastructure is easiest to trust when every weight "
    "transformation is pinned, reproducible, and checked against a reference. "
    "A compact artifact should preserve the model's routing semantics while "
    "storing each expert exactly once."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-key", default="hy3-q4-native")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--baseline-model-key", default="hy3-q4")
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--memory-limit", default="116GiB")
    parser.add_argument("--runtime-reserve", default="8GiB")
    parser.add_argument("--max-ppl", type=float, default=1_000.0)
    parser.add_argument("--max-ppl-ratio", type=float, default=2.0)
    return parser


def _run_lock_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-fl", r"python.*(benchmark_streamed|probe_mtp)"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"pgrep failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _text(args: argparse.Namespace) -> str:
    if args.text is not None and args.text_file is not None:
        raise SystemExit("choose only one of --text or --text-file")
    if args.text_file is not None:
        value = args.text_file.read_text(encoding="utf-8")
    else:
        value = args.text if args.text is not None else DEFAULT_TEXT
    if not value.strip():
        raise SystemExit("text sample is empty")
    return value


def _config(model_key: str, args: argparse.Namespace) -> ExpertStreamingConfig:
    return ExpertStreamingConfig(
        model_key=model_key,
        memory_limit_bytes=parse_memory_bytes(args.memory_limit),
        max_live_kv_tokens=args.max_tokens,
        runtime_reserve_bytes=parse_memory_bytes(args.runtime_reserve),
        expert_cache_limit_bytes=0,
        transient_slots=8,
        prefer_sidecar=True,
        verify_record_hashes=True,
        verify_artifact_headers=True,
        slot_layout="component-banks",
        prefill_admission=False,
    )


def _perplexity(mean_nll: float) -> float | None:
    """Return a JSON-safe perplexity value, or ``None`` for invalid logits."""

    if (
        not math.isfinite(mean_nll)
        or mean_nll < 0
        or mean_nll > math.log(sys.float_info.max)
    ):
        return None
    value = math.exp(mean_nll)
    return value if math.isfinite(value) and value > 0 else None


def _evaluate(
    root: Path,
    manifest: Path,
    *,
    model_key: str,
    text: str,
    args: argparse.Namespace,
) -> dict[str, float | int | str | None]:
    import mlx.core as mx

    runtime = load(
        root,
        mtp=False,
        expert_streaming_config=_config(model_key, args),
        expert_manifest=manifest,
    )
    try:
        token_ids = [int(token) for token in runtime.tokenizer.encode(text)]
        token_ids = token_ids[: args.max_tokens]
        if len(token_ids) < 3:
            raise RuntimeError("text sample encoded to fewer than three tokens")
        inputs = mx.array([token_ids[:-1]], dtype=mx.int32)
        targets = mx.array(token_ids[1:], dtype=mx.int32)
        runtime.expert_streaming.reset()
        cache = runtime.make_cache()
        with runtime.admit_kv_tokens(len(token_ids) - 1):
            logits = runtime.forward_ar(inputs, cache=cache)[0].astype(mx.float32)
            target_logits = mx.take_along_axis(
                logits,
                targets[:, None],
                axis=-1,
            )[:, 0]
            nll = mx.logsumexp(logits, axis=-1) - target_logits
            mean_nll = mx.mean(nll)
            mx.eval(mean_nll)
        value = float(mean_nll.item())
        return {
            "model_key": model_key,
            "tokens": len(token_ids),
            "mean_nll": value if math.isfinite(value) else None,
            "perplexity": _perplexity(value),
        }
    finally:
        runtime.close(timeout=30.0)
        try:
            mx.clear_cache()
        except AttributeError:
            pass


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens < 3:
        raise SystemExit("--max-tokens must be at least 3")
    if not math.isfinite(args.max_ppl) or args.max_ppl <= 0:
        raise SystemExit("--max-ppl must be finite and positive")
    if not math.isfinite(args.max_ppl_ratio) or args.max_ppl_ratio < 1:
        raise SystemExit("--max-ppl-ratio must be finite and at least 1")
    if (args.baseline_root is None) != (args.baseline_manifest is None):
        raise SystemExit(
            "--baseline-root and --baseline-manifest are required together"
        )
    active = _run_lock_processes()
    if active:
        raise SystemExit(
            "benchmark/probe run-lock is occupied; refusing GPU validation:\n"
            + "\n".join(active)
        )
    print("run-lock clear; starting perplexity sanity compute", file=sys.stderr)

    sample = _text(args)
    baseline = None
    if args.baseline_root is not None:
        baseline = _evaluate(
            args.baseline_root.expanduser().resolve(),
            args.baseline_manifest.expanduser().resolve(),
            model_key=args.baseline_model_key,
            text=sample,
            args=args,
        )
    native = _evaluate(
        args.artifact_root.expanduser().resolve(),
        args.manifest.expanduser().resolve(),
        model_key=args.model_key,
        text=sample,
        args=args,
    )
    native_value = native["perplexity"]
    native_ppl = (
        float(native_value)
        if isinstance(native_value, (int, float)) and not isinstance(native_value, bool)
        else None
    )
    passed = (
        native_ppl is not None
        and math.isfinite(native_ppl)
        and 0 < native_ppl <= args.max_ppl
    )
    ratio = None
    if baseline is not None:
        baseline_value = baseline["perplexity"]
        baseline_ppl = (
            float(baseline_value)
            if isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
            else None
        )
        if (
            native_ppl is not None
            and baseline_ppl is not None
            and math.isfinite(baseline_ppl)
            and baseline_ppl > 0
        ):
            ratio = native_ppl / baseline_ppl
            passed = passed and (
                math.isfinite(ratio)
                and 1 / args.max_ppl_ratio <= ratio <= args.max_ppl_ratio
            )
        else:
            passed = False
    report = {
        "schema": "mtplx-streamed-perplexity-sanity-v1",
        "passed": passed,
        "pid": os.getpid(),
        "sample_sha256": hashlib.sha256(sample.encode()).hexdigest(),
        "max_tokens": args.max_tokens,
        "max_ppl": args.max_ppl,
        "max_ppl_ratio": args.max_ppl_ratio,
        "native": native,
        "baseline": baseline,
        "native_to_baseline_ppl_ratio": ratio,
        "performance_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
