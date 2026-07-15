#!/usr/bin/env python3
"""Benchmark the isolated Hy3 Q2 expert E1 gate+up+SwiGLU frontier."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mtplx.expert_manifest import load_expert_manifest, read_expert_record
from mtplx.hy3_expert_fused_e1 import (
    HY3_E1_GROUP_SIZE,
    HY3_E1_HIDDEN_SIZE,
    HY3_E1_INTERMEDIATE_SIZE,
    HY3_E1_K_TILES,
    HY3_E1_OUTPUT_N_TILES,
    HY3_E1_ROWS,
    HY3_E1_SILU_MODES,
    HY3_E1_SIMD_GROUPS,
    HY3_E1_TOP_K,
    Hy3ExpertE1Candidate,
    build_hy3_expert_fused_e1_kernel,
    hy3_e1_component_bank_shapes,
    hy3_e1_correctness_metrics,
    hy3_e1_supported_candidates,
    hy3_expert_fused_e1,
    hy3_expert_fused_e1_reference,
)
from mtplx.qwen_guard import DEFAULT_MLX_LOCK_PATH, exclusive_mlx_window


def select_candidates(
    *,
    n_tiles: tuple[int, ...] = HY3_E1_OUTPUT_N_TILES,
    k_tiles: tuple[int, ...] = HY3_E1_K_TILES,
    simd_groups: tuple[int, ...] = HY3_E1_SIMD_GROUPS,
    silu_modes: tuple[str, ...] = HY3_E1_SILU_MODES,
) -> tuple[Hy3ExpertE1Candidate, ...]:
    """Filter without collapsing the surviving multi-variant frontier."""

    candidates = tuple(
        candidate
        for candidate in hy3_e1_supported_candidates(silu_modes=silu_modes)
        if candidate.output_n_tile in n_tiles
        and candidate.k_tile in k_tiles
        and candidate.simd_groups in simd_groups
    )
    if not candidates:
        raise ValueError("E1 candidate filters select no legal variants")
    return candidates


def exact_candidate_key(candidate: Hy3ExpertE1Candidate) -> tuple[int, int, int]:
    """Identify matching exact/fast variants without mixing SiLU semantics."""

    return (
        candidate.output_n_tile,
        candidate.k_tile,
        candidate.simd_groups,
    )


def fast_silu_prerequisite_met(
    candidate: Hy3ExpertE1Candidate,
    exact_passed: set[tuple[int, int, int]],
) -> bool:
    """Allow a fast-SiLU ablation only after its exact twin passes."""

    if candidate.silu_mode != "fast":
        raise ValueError("fast-SiLU prerequisite applies only to fast candidates")
    return exact_candidate_key(candidate) in exact_passed


def classify_candidate_failure(exc: BaseException, *, phase: str) -> str:
    """Keep compile, resource, and other dispatch failures attributable."""

    message = str(exc).lower()
    resource_markers = (
        "threadgroup memory",
        "resource limit",
        "too many threads",
        "max total threads",
        "register pressure",
    )
    compile_markers = (
        "compile",
        "compiler",
        "failed to build program",
        "metal library",
        "syntax error",
    )
    if any(marker in message for marker in resource_markers):
        return "resource"
    if phase == "kernel_build" or any(marker in message for marker in compile_markers):
        return "compile"
    return "dispatch"


def _decode_component(
    payload: bytes, *, dtype: str, shape: tuple[int, ...]
) -> mx.array:
    if dtype == "U32":
        host = np.frombuffer(payload, dtype="<u4").copy().reshape(shape)
        return mx.array(host)
    if dtype == "BF16":
        words = np.frombuffer(payload, dtype="<u2").copy().reshape(shape)
        return mx.array(words).view(mx.bfloat16)
    raise ValueError(f"unsupported E1 component dtype: {dtype}")


def _load_e1_component_bank(
    model: Path,
    *,
    layer: int,
    experts: tuple[int, ...],
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """Load eight real records into the preserved component-major Q2 layout."""

    if len(experts) != HY3_E1_TOP_K or len(set(experts)) != HY3_E1_TOP_K:
        raise ValueError("E1 benchmark requires eight distinct experts")
    manifest_path = model / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    if (
        manifest.model_key != "hy3-expert-q2"
        or manifest.quant_bits != 2
        or manifest.quant_group_size != HY3_E1_GROUP_SIZE
        or manifest.quant_mode != "affine"
    ):
        raise ValueError("E1 benchmark requires the authoritative Hy3 affine-Q2 model")
    wanted = (
        "gate_proj.weight",
        "gate_proj.scales",
        "gate_proj.biases",
        "up_proj.weight",
        "up_proj.scales",
        "up_proj.biases",
    )
    component_rows: dict[str, list[mx.array]] = {name: [] for name in wanted}
    source_metadata = []
    for expert in experts:
        record = manifest.record(layer, expert)
        payload = read_expert_record(manifest, model, layer, expert)
        cursor = 0
        record_metadata = {"expert": expert, "logical_bytes": record.logical_bytes}
        for segment in record.segments:
            component_payload = payload[cursor : cursor + segment.length]
            cursor += segment.length
            if segment.component in component_rows:
                component_rows[segment.component].append(
                    _decode_component(
                        component_payload,
                        dtype=segment.dtype,
                        shape=segment.shape,
                    )
                )
        if cursor != len(payload):
            raise ValueError("E1 expert record components do not cover its payload")
        source_metadata.append(record_metadata)
    bank = {name: mx.stack(values, axis=0) for name, values in component_rows.items()}
    mx.eval(*bank.values())
    expected = hy3_e1_component_bank_shapes(len(experts))
    for name, value in bank.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(f"real E1 component {name} has unexpected geometry")
    return bank, {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
        "model_key": manifest.model_key,
        "layer": int(layer),
        "experts": list(experts),
        "records": source_metadata,
        "component_shapes": {name: list(value.shape) for name, value in bank.items()},
    }


def _token_rows(rows: int, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((rows, HY3_E1_HIDDEN_SIZE)).astype(mx.bfloat16)


def _expert_slots(rows: int) -> mx.array:
    slots = np.stack(
        [np.roll(np.arange(HY3_E1_TOP_K, dtype=np.int32), -row) for row in range(rows)]
    )
    return mx.array(slots)


def _evaluate(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples or any(value <= 0.0 for value in samples):
        raise ValueError("E1 timing samples must be nonempty and positive")
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _paired_bootstrap(
    reference_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    if (
        reference.ndim != 1
        or candidate.ndim != 1
        or reference.size == 0
        or reference.size != candidate.size
        or np.any(reference <= 0.0)
        or np.any(candidate <= 0.0)
        or bootstrap_resamples <= 0
    ):
        raise ValueError("paired E1 samples must be equal positive vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        reference.size,
        size=(bootstrap_resamples, reference.size),
    )
    boot = reference[indices].mean(axis=1) / candidate[indices].mean(axis=1)
    ratios = reference / candidate
    return {
        "reference_over_candidate_ratio_of_means": float(
            reference.mean() / candidate.mean()
        ),
        "paired_ratio_mean": float(ratios.mean()),
        "paired_ratio_median": float(np.median(ratios)),
        "bootstrap_mean_ratio_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
    }


def _measure_pair(
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if min(warmups, repeats, bootstrap_resamples) <= 0:
        raise ValueError("E1 measurement counts must be positive")
    functions = {"component_bank_reference": reference, "fused_e1": candidate}
    for function in functions.values():
        for _ in range(warmups):
            _evaluate(function())
    samples = {name: [] for name in functions}
    names = tuple(functions)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter_ns()
            _evaluate(functions[name]())
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "arms": {name: _sample_summary(values) for name, values in samples.items()},
        "comparison": _paired_bootstrap(
            samples["component_bank_reference"],
            samples["fused_e1"],
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def _candidate_call(
    token_rows: mx.array,
    slots: mx.array,
    bank: dict[str, mx.array],
    candidate: Hy3ExpertE1Candidate,
) -> mx.array:
    return hy3_expert_fused_e1(
        token_rows,
        slots,
        bank["gate_proj.weight"],
        bank["gate_proj.scales"],
        bank["gate_proj.biases"],
        bank["up_proj.weight"],
        bank["up_proj.scales"],
        bank["up_proj.biases"],
        candidate=candidate,
    )


def _reference_call(
    token_rows: mx.array,
    slots: mx.array,
    bank: dict[str, mx.array],
) -> mx.array:
    return hy3_expert_fused_e1_reference(
        token_rows,
        slots,
        bank["gate_proj.weight"],
        bank["gate_proj.scales"],
        bank["gate_proj.biases"],
        bank["up_proj.weight"],
        bank["up_proj.scales"],
        bank["up_proj.biases"],
    )


def preflight_candidates(
    candidates: Sequence[Hy3ExpertE1Candidate],
) -> dict[str, Any]:
    """Construct every lazy kernel object without dispatching or short-circuiting."""

    results = []
    failures: dict[str, list[dict[str, str]]] = {
        "compile": [],
        "resource": [],
        "dispatch": [],
    }
    for candidate in candidates:
        try:
            build_hy3_expert_fused_e1_kernel(candidate)
        except Exception as exc:
            kind = classify_candidate_failure(exc, phase="kernel_build")
            failure = {
                "candidate": candidate.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures[kind].append(failure)
            results.append(
                {**candidate.to_dict(), "preflight_status": f"{kind}_failure"}
            )
        else:
            results.append(
                {**candidate.to_dict(), "preflight_status": "kernel_object_ready"}
            )
    return {
        "schema": "mtplx-issue51-hy3-q2-expert-e1-compile-preflight-v1",
        "scope": "lazy kernel-object construction only; no Metal dispatch",
        "candidates": results,
        "failures": failures,
    }


def run_benchmark(
    *,
    model: Path,
    layer: int,
    experts: tuple[int, ...],
    candidates: tuple[Hy3ExpertE1Candidate, ...],
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    bank, component_metadata = _load_e1_component_bank(
        model,
        layer=layer,
        experts=experts,
    )
    failures: dict[str, list[dict[str, Any]]] = {
        "compile": [],
        "resource": [],
        "dispatch": [],
        "correctness": [],
    }
    result_rows = []
    for rows in HY3_E1_ROWS:
        token_rows = _token_rows(rows, seed=551_000 + rows)
        slots = _expert_slots(rows)
        reference = _reference_call(token_rows, slots, bank)
        _evaluate(reference)
        variants = []
        exact_passed: set[tuple[int, int, int]] = set()
        for candidate_index, candidate in enumerate(candidates):
            base = candidate.to_dict()
            if candidate.silu_mode == "fast" and not fast_silu_prerequisite_met(
                candidate,
                exact_passed,
            ):
                exact_name = Hy3ExpertE1Candidate(
                    candidate.output_n_tile,
                    candidate.k_tile,
                    candidate.simd_groups,
                    "exact",
                ).name
                variants.append(
                    {
                        **base,
                        "status": "blocked_by_exact_correctness",
                        "exact_prerequisite": exact_name,
                    }
                )
                continue
            try:
                build_hy3_expert_fused_e1_kernel(candidate)
            except Exception as exc:
                kind = classify_candidate_failure(exc, phase="kernel_build")
                failure = {
                    "m": rows,
                    "candidate": candidate.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures[kind].append(failure)
                variants.append(
                    {**base, "status": f"{kind}_failure", "failure": failure}
                )
                continue

            def candidate_function(
                candidate: Hy3ExpertE1Candidate = candidate,
            ) -> mx.array:
                return _candidate_call(token_rows, slots, bank, candidate)

            try:
                observed = candidate_function()
                _evaluate(observed)
            except Exception as exc:
                kind = classify_candidate_failure(exc, phase="first_eval")
                failure = {
                    "m": rows,
                    "candidate": candidate.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures[kind].append(failure)
                variants.append(
                    {**base, "status": f"{kind}_failure", "failure": failure}
                )
                continue
            correctness = hy3_e1_correctness_metrics(
                observed,
                reference,
                silu_mode=candidate.silu_mode,
            )
            if candidate.silu_mode == "exact" and not correctness["passes"]:
                failure = {
                    "m": rows,
                    "candidate": candidate.name,
                    "max_abs_error": correctness["max_abs_error"],
                    "normalized_rmse": correctness["normalized_rmse"],
                }
                failures["correctness"].append(failure)
                variants.append(
                    {
                        **base,
                        "status": "correctness_failure",
                        "correctness": correctness,
                    }
                )
                continue
            if candidate.silu_mode == "exact":
                exact_passed.add(exact_candidate_key(candidate))
            try:
                timing = _measure_pair(
                    lambda: _reference_call(token_rows, slots, bank),
                    candidate_function,
                    warmups=warmups,
                    repeats=repeats,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=552_000 + rows * 100 + candidate_index,
                )
            except Exception as exc:
                kind = classify_candidate_failure(exc, phase="timing")
                failure = {
                    "m": rows,
                    "candidate": candidate.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures[kind].append(failure)
                variants.append(
                    {
                        **base,
                        "status": f"{kind}_failure",
                        "correctness": correctness,
                        "failure": failure,
                    }
                )
                continue
            variants.append(
                {
                    **base,
                    "status": (
                        "measured_exact_candidate"
                        if candidate.silu_mode == "exact"
                        else "measured_fast_silu_ablation"
                    ),
                    "correctness": correctness,
                    "timing": timing,
                }
            )
        result_rows.append(
            {
                "m": rows,
                "token_expert_assignments": rows * HY3_E1_TOP_K,
                "variants": variants,
            }
        )
        print(f"completed isolated E1 M={rows}", file=sys.stderr, flush=True)
    return {
        "schema": "mtplx-issue51-hy3-q2-expert-fused-e1-v1",
        "scope": "Stage E1 only: Q2 gate+up+SwiGLU; E2/down excluded",
        "model": str(model),
        "layer": int(layer),
        "component_bank": component_metadata,
        "geometry": {
            "rows": list(HY3_E1_ROWS),
            "selected_experts_per_token": HY3_E1_TOP_K,
            "hidden_size": HY3_E1_HIDDEN_SIZE,
            "intermediate_size": HY3_E1_INTERMEDIATE_SIZE,
            "quantization": {
                "bits": 2,
                "group_size": HY3_E1_GROUP_SIZE,
                "mode": "affine",
            },
            "output_dtype": "bfloat16",
        },
        "correctness_boundary": {
            "reference": "component-bank gather_qmm gate/up plus exact SwiGLU",
            "exact_silu_only": True,
            "max_abs_error": 0.5,
            "max_normalized_rmse": 0.02,
            "fast_silu": "separately attributable ablation; never promotion eligible",
            "fast_silu_prerequisite": "matching exact topology must pass first",
        },
        "measurement": {
            "warmups": int(warmups),
            "paired_interleaved_repeats": int(repeats),
            "bootstrap_resamples": int(bootstrap_resamples),
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
        "failures": failures,
        "rows": result_rows,
        "environment": {"device": mx.metal.device_info()},
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _csv_ints(
    value: str,
    *,
    allowed: tuple[int, ...] | None = None,
    count: int | None = None,
) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    if count is not None and len(result) != count:
        raise argparse.ArgumentTypeError(f"must contain exactly {count} values")
    if allowed is not None and set(result) - set(allowed):
        raise argparse.ArgumentTypeError(f"values must be selected from {allowed}")
    return result


def _experts(value: str) -> tuple[int, ...]:
    result = _csv_ints(value, count=HY3_E1_TOP_K)
    if min(result) < 0 or max(result) >= 192:
        raise argparse.ArgumentTypeError("experts must be in [0, 191]")
    return result


def _n_tiles(value: str) -> tuple[int, ...]:
    return _csv_ints(value, allowed=HY3_E1_OUTPUT_N_TILES)


def _k_tiles(value: str) -> tuple[int, ...]:
    return _csv_ints(value, allowed=HY3_E1_K_TILES)


def _simd_groups(value: str) -> tuple[int, ...]:
    return _csv_ints(value, allowed=HY3_E1_SIMD_GROUPS)


def _silu_modes(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("SiLU modes must be nonempty and unique")
    if set(result) - set(HY3_E1_SILU_MODES):
        raise argparse.ArgumentTypeError("SiLU modes must be exact and/or fast")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--experts", type=_experts, default=tuple(range(8)))
    parser.add_argument("--n-tiles", type=_n_tiles, default=HY3_E1_OUTPUT_N_TILES)
    parser.add_argument("--k-tiles", type=_k_tiles, default=HY3_E1_K_TILES)
    parser.add_argument(
        "--simd-groups",
        type=_simd_groups,
        default=HY3_E1_SIMD_GROUPS,
    )
    parser.add_argument("--silu-modes", type=_silu_modes, default=HY3_E1_SILU_MODES)
    parser.add_argument("--warmups", type=_positive_int, default=4)
    parser.add_argument("--repeats", type=_positive_int, default=20)
    parser.add_argument(
        "--bootstrap-resamples",
        "--bootstrap",
        dest="bootstrap_resamples",
        type=_positive_int,
        default=10_000,
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument("--lock-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--output-json", type=Path)
    return parser


def _render(result: dict[str, Any], output_json: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output_json is not None:
        output = output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidates = select_candidates(
        n_tiles=args.n_tiles,
        k_tiles=args.k_tiles,
        simd_groups=args.simd_groups,
        silu_modes=args.silu_modes,
    )
    if args.compile_only:
        _render(preflight_candidates(candidates), args.output_json)
        return 0
    with exclusive_mlx_window(
        plist=args.qwen_plist,
        lock_path=args.lock_path,
        lock_timeout_seconds=args.lock_timeout_seconds,
    ) as receipt:
        print(
            f"acquired exclusive MLX window at {receipt.lock_path}",
            file=sys.stderr,
            flush=True,
        )
        result = run_benchmark(
            model=args.model.expanduser().resolve(),
            layer=args.layer,
            experts=args.experts,
            candidates=candidates,
            warmups=args.warmups,
            repeats=args.repeats,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        result["exclusive_window"] = {"lock_path": str(receipt.lock_path)}
    _render(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
