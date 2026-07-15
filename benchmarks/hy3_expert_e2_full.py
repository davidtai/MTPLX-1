#!/usr/bin/env python3
"""Screen/refine issue #66 full-core E2 candidates on real Hy3-Q2 tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mtplx.expert_manifest import load_expert_manifest, read_expert_record
from mtplx.hy3_expert_e2_full import (
    HY3_E2_ACTIVATION_MODES,
    HY3_E2_BITS,
    HY3_E2_GROUP_SIZE,
    HY3_E2_HIDDEN_SIZE,
    HY3_E2_INTERMEDIATE_SIZE,
    HY3_E2_K_TILES,
    HY3_E2_OUTPUT_N_TILES,
    HY3_E2_ROWS,
    HY3_E2_SIMD_GROUPS,
    HY3_E2_TOP_K,
    Hy3ExpertE2Candidate,
    hy3_e2_correctness_metrics,
    hy3_e2_supported_candidates,
    hy3_e2_swiglu,
    hy3_expert_e2_fused,
    hy3_expert_e2_reference,
)
from mtplx.qwen_guard import DEFAULT_MLX_LOCK_PATH, exclusive_mlx_window


FAMILIES = ("stock_eager", "stock_compiled", "fused_e2")
FIXTURES = ("router_like", "reverse_order", "peaked")


def _decode_component(
    payload: bytes,
    *,
    dtype: str,
    shape: tuple[int, ...],
) -> mx.array:
    if dtype == "U32":
        return mx.array(np.frombuffer(payload, dtype="<u4").copy().reshape(shape))
    if dtype == "BF16":
        words = np.frombuffer(payload, dtype="<u2").copy().reshape(shape)
        return mx.array(words).view(mx.bfloat16)
    raise ValueError(f"unsupported E2 component dtype: {dtype}")


def _load_source_bank(
    model: Path,
    *,
    layer: int,
    experts: tuple[int, ...],
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    if len(experts) != HY3_E2_TOP_K or len(set(experts)) != HY3_E2_TOP_K:
        raise ValueError("E2 requires eight distinct component-bank experts")
    manifest_path = model / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    if (
        manifest.model_key != "hy3-expert-q2"
        or manifest.quant_bits != HY3_E2_BITS
        or manifest.quant_group_size != HY3_E2_GROUP_SIZE
        or manifest.quant_mode != "affine"
    ):
        raise ValueError("E2 requires the authoritative Hy3 affine-Q2 model")
    wanted = tuple(
        f"{projection}.{component}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for component in ("weight", "scales", "biases")
    )
    rows: dict[str, list[mx.array]] = {name: [] for name in wanted}
    records = []
    for expert in experts:
        record = manifest.record(layer, expert)
        payload = read_expert_record(manifest, model, layer, expert)
        cursor = 0
        seen: set[str] = set()
        for segment in record.segments:
            component = payload[cursor : cursor + segment.length]
            cursor += segment.length
            if segment.component in rows:
                rows[segment.component].append(
                    _decode_component(
                        component,
                        dtype=segment.dtype,
                        shape=segment.shape,
                    )
                )
                seen.add(segment.component)
        if cursor != len(payload) or seen != set(wanted):
            raise ValueError("E2 record does not contain one complete expert core")
        records.append(
            {
                "expert": int(expert),
                "logical_bytes": int(record.logical_bytes),
                "record_sha256": record.sha256,
            }
        )
    bank = {name: mx.stack(values, axis=0) for name, values in rows.items()}
    mx.eval(*bank.values())
    return bank, {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
        "model_key": manifest.model_key,
        "layer": int(layer),
        "experts": list(experts),
        "records": records,
        "component_shapes": {name: list(value.shape) for name, value in bank.items()},
        "component_bytes": {name: int(value.nbytes) for name, value in bank.items()},
        "total_component_bytes": sum(int(value.nbytes) for value in bank.values()),
    }


def _token_rows(rows: int, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((rows, HY3_E2_HIDDEN_SIZE)).astype(mx.bfloat16)


def _expert_slots(rows: int, *, fixture: str) -> mx.array:
    base = np.stack(
        [np.roll(np.arange(HY3_E2_TOP_K, dtype=np.int32), -row) for row in range(rows)]
    )
    if fixture == "reverse_order":
        base = base[:, ::-1].copy()
    elif fixture == "peaked":
        permutation = np.array([7, 0, 6, 1, 5, 2, 4, 3], dtype=np.int32)
        base = base[:, permutation]
    elif fixture != "router_like":
        raise ValueError(f"unknown E2 correctness fixture: {fixture}")
    return mx.array(base)


def _route_weights(rows: int, *, fixture: str, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    if fixture in {"router_like", "reverse_order"}:
        logits = rng.normal(0.0, 1.4, size=(rows, HY3_E2_TOP_K)).astype(np.float32)
        weights = 1.0 / (1.0 + np.exp(-logits))
        weights /= weights.sum(axis=-1, keepdims=True)
        if fixture == "reverse_order":
            weights = weights[:, ::-1].copy()
    elif fixture == "peaked":
        weights = np.broadcast_to(
            np.array(
                [
                    0.875,
                    0.0625,
                    0.03125,
                    0.015625,
                    0.0078125,
                    0.00390625,
                    0.001953125,
                    0.001953125,
                ],
                dtype=np.float32,
            ),
            (rows, HY3_E2_TOP_K),
        ).copy()
        weights /= weights.sum(axis=-1, keepdims=True)
    else:
        raise ValueError(f"unknown E2 correctness fixture: {fixture}")
    return mx.array(weights).astype(mx.bfloat16)


def _selected_inputs(
    token_rows: mx.array, slots: mx.array
) -> tuple[mx.array, mx.array]:
    rows = int(token_rows.shape[0])
    assignments = rows * HY3_E2_TOP_K
    selected = mx.broadcast_to(
        token_rows[:, None, :],
        (rows, HY3_E2_TOP_K, HY3_E2_HIDDEN_SIZE),
    ).reshape(assignments, 1, 1, HY3_E2_HIDDEN_SIZE)
    return selected, slots.reshape(assignments, 1)


def _stock_preactivations(
    token_rows: mx.array,
    slots: mx.array,
    source: dict[str, mx.array],
) -> tuple[mx.array, mx.array]:
    rows = int(token_rows.shape[0])
    selected, indices = _selected_inputs(token_rows, slots)

    def qmm(projection: str) -> mx.array:
        output = mx.gather_qmm(
            selected,
            source[f"{projection}.weight"],
            source[f"{projection}.scales"],
            source[f"{projection}.biases"],
            rhs_indices=indices,
            transpose=True,
            group_size=HY3_E2_GROUP_SIZE,
            bits=HY3_E2_BITS,
            mode="affine",
        )
        return output.reshape(rows, HY3_E2_TOP_K, HY3_E2_INTERMEDIATE_SIZE)

    return qmm("gate_proj"), qmm("up_proj")


def _stock_hidden(
    token_rows: mx.array,
    slots: mx.array,
    source: dict[str, mx.array],
) -> mx.array:
    gate, up = _stock_preactivations(token_rows, slots, source)
    return (nn.silu(gate) * up).astype(mx.bfloat16)


def _stock_full(
    token_rows: mx.array,
    slots: mx.array,
    route_weights: mx.array,
    source: dict[str, mx.array],
) -> mx.array:
    hidden = _stock_hidden(token_rows, slots, source)
    _per_expert, final = hy3_expert_e2_reference(
        hidden,
        slots,
        route_weights,
        source["down_proj.weight"],
        source["down_proj.scales"],
        source["down_proj.biases"],
    )
    return final


def _compiled_stock(
    source: dict[str, mx.array],
) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    names = tuple(source)

    def graph(
        token_rows: mx.array,
        slots: mx.array,
        route_weights: mx.array,
        *arrays: mx.array,
    ) -> mx.array:
        live = dict(zip(names, arrays, strict=True))
        return _stock_full(token_rows, slots, route_weights, live)

    compiled = mx.compile(graph)

    def call(
        token_rows: mx.array, slots: mx.array, route_weights: mx.array
    ) -> mx.array:
        return compiled(
            token_rows,
            slots,
            route_weights,
            *(source[name] for name in names),
        )

    return call


def _candidate_hidden(
    token_rows: mx.array,
    slots: mx.array,
    source: dict[str, mx.array],
    *,
    activation_mode: str,
) -> mx.array:
    gate, up = _stock_preactivations(token_rows, slots, source)
    if activation_mode == "exact":
        return (nn.silu(gate) * up).astype(mx.bfloat16)
    return hy3_e2_swiglu(gate, up, activation_mode="fast")


def _candidate_full(
    token_rows: mx.array,
    slots: mx.array,
    route_weights: mx.array,
    source: dict[str, mx.array],
    *,
    candidate: Hy3ExpertE2Candidate,
) -> mx.array:
    hidden = _candidate_hidden(
        token_rows,
        slots,
        source,
        activation_mode=candidate.activation_mode,
    )
    output = hy3_expert_e2_fused(
        hidden,
        slots,
        route_weights,
        source["down_proj.weight"],
        source["down_proj.scales"],
        source["down_proj.biases"],
        candidate=candidate,
    )
    assert isinstance(output, mx.array)
    return output


def _evaluate(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def _tensor_metrics(observed: mx.array, reference: mx.array) -> dict[str, Any]:
    if tuple(observed.shape) != tuple(reference.shape):
        return {
            "shape_exact": False,
            "dtype": str(observed.dtype),
            "array_equal": False,
            "all_finite": False,
            "max_abs_error": float("inf"),
            "normalized_rmse": float("inf"),
        }
    observed_f32 = observed.astype(mx.float32)
    reference_f32 = reference.astype(mx.float32)
    difference = observed_f32 - reference_f32
    rmse = float(mx.sqrt(mx.mean(mx.square(difference))).item())
    reference_rms = float(mx.sqrt(mx.mean(mx.square(reference_f32))).item())
    return {
        "shape_exact": True,
        "dtype": str(observed.dtype),
        "array_equal": bool(mx.array_equal(observed, reference).item()),
        "all_finite": bool(
            mx.all(mx.isfinite(observed_f32)).item()
            and mx.all(mx.isfinite(reference_f32)).item()
        ),
        "max_abs_error": float(mx.max(mx.abs(difference)).item()),
        "normalized_rmse": rmse / max(reference_rms, 1e-12),
    }


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _paired_bootstrap(
    control: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    control_array = np.asarray(control, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, control_array.size, size=(resamples, control_array.size))
    ratios = control_array[indices].mean(axis=1) / candidate_array[indices].mean(axis=1)
    return {
        "control_over_candidate_ratio_of_means": float(
            control_array.mean() / candidate_array.mean()
        ),
        "paired_ratio_mean": float(np.mean(control_array / candidate_array)),
        "paired_ratio_median": float(np.median(control_array / candidate_array)),
        "bootstrap_mean_ratio_95_ci": [
            float(np.quantile(ratios, 0.025)),
            float(np.quantile(ratios, 0.975)),
        ],
    }


def _measure_pair(
    control: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    functions = {"stock": control, "candidate": candidate}
    for function in functions.values():
        for _ in range(warmups):
            _evaluate(function())
    samples: dict[str, list[float]] = {name: [] for name in functions}
    names = tuple(functions)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in order:
            started = time.perf_counter_ns()
            _evaluate(functions[name]())
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "stock": _sample_summary(samples["stock"]),
        "candidate": _sample_summary(samples["candidate"]),
        "speedup": _paired_bootstrap(
            samples["stock"],
            samples["candidate"],
            resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def classify_candidate_failure(error: BaseException, *, phase: str) -> str:
    message = str(error).lower()
    if (
        phase in {"kernel_build", "compile"}
        or "compiler" in message
        or "build program" in message
    ):
        return "compile"
    if any(
        token in message
        for token in ("resource", "threadgroup", "too many threads", "register")
    ):
        return "resource"
    return "dispatch"


def classify_stage_discrepancy(
    *,
    per_expert_array_equal: bool,
    fused_matches_stock_reduce: bool,
) -> str:
    """Keep down-dot and ordered-router discrepancies independently visible."""

    if per_expert_array_equal and fused_matches_stock_reduce:
        return "bit_exact"
    if not per_expert_array_equal and fused_matches_stock_reduce:
        return "down_qmm_reduction_order_only"
    if per_expert_array_equal and not fused_matches_stock_reduce:
        return "router_reduction_order_only"
    return "down_qmm_and_router_reduction_order"


def exact_candidate_key(candidate: Hy3ExpertE2Candidate) -> tuple[int, int, int]:
    return candidate.output_n_tile, candidate.k_tile, candidate.simd_groups


def fast_prerequisite_met(
    candidate: Hy3ExpertE2Candidate,
    passing_exact: set[tuple[int, int, int]],
) -> bool:
    if candidate.activation_mode != "fast":
        return True
    return exact_candidate_key(candidate) in passing_exact


def select_candidates(
    *,
    n_tiles: tuple[int, ...],
    k_tiles: tuple[int, ...],
    simd_groups: tuple[int, ...],
    activation_modes: tuple[str, ...],
    candidate_names: tuple[str, ...] = (),
) -> tuple[Hy3ExpertE2Candidate, ...]:
    candidates = tuple(
        candidate
        for candidate in hy3_e2_supported_candidates(activation_modes=activation_modes)
        if candidate.output_n_tile in n_tiles
        and candidate.k_tile in k_tiles
        and candidate.simd_groups in simd_groups
        and (not candidate_names or candidate.name in candidate_names)
    )
    unknown = set(candidate_names) - {candidate.name for candidate in candidates}
    if unknown:
        raise ValueError(f"unknown or filtered E2 candidate names: {sorted(unknown)}")
    if not candidates:
        raise ValueError("E2 filters select no supported candidates")
    return candidates


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1]
    files = (
        root / "mtplx" / "hy3_expert_e2_full.py",
        Path(__file__),
        root / "tests" / "test_hy3_expert_e2_full.py",
    )
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _correctness_fixture(
    *,
    rows: int,
    fixture: str,
    token_rows: mx.array,
    source: dict[str, mx.array],
    candidate: Hy3ExpertE2Candidate,
    seed: int,
) -> tuple[dict[str, Any], bool]:
    slots = _expert_slots(rows, fixture=fixture)
    route_weights = _route_weights(rows, fixture=fixture, seed=seed)
    gate, up = _stock_preactivations(token_rows, slots, source)
    exact_hidden = (nn.silu(gate) * up).astype(mx.bfloat16)
    candidate_hidden = (
        exact_hidden
        if candidate.activation_mode == "exact"
        else hy3_e2_swiglu(gate, up, activation_mode="fast")
    )
    reference_per_expert, reference_final = hy3_expert_e2_reference(
        exact_hidden,
        slots,
        route_weights,
        source["down_proj.weight"],
        source["down_proj.scales"],
        source["down_proj.biases"],
    )
    observed = hy3_expert_e2_fused(
        candidate_hidden,
        slots,
        route_weights,
        source["down_proj.weight"],
        source["down_proj.scales"],
        source["down_proj.biases"],
        candidate=candidate,
        diagnostic=True,
    )
    assert isinstance(observed, tuple)
    observed_per_expert, observed_final = observed
    stock_reduce_from_observed = (
        (observed_per_expert * route_weights[..., None])
        .sum(axis=-2)
        .astype(mx.bfloat16)
    )
    _evaluate(
        (
            gate,
            up,
            exact_hidden,
            candidate_hidden,
            reference_per_expert,
            reference_final,
            observed_per_expert,
            observed_final,
            stock_reduce_from_observed,
        )
    )
    metrics = hy3_e2_correctness_metrics(
        observed_per_expert,
        observed_final,
        reference_per_expert,
        reference_final,
        activation_mode=candidate.activation_mode,
    )
    metrics["activation"] = _tensor_metrics(candidate_hidden, exact_hidden)
    metrics["stock_reduce_from_candidate_down"] = _tensor_metrics(
        stock_reduce_from_observed,
        reference_final,
    )
    metrics["fused_reduce_given_candidate_down"] = _tensor_metrics(
        observed_final,
        stock_reduce_from_observed,
    )
    metrics["stage_classification"] = classify_stage_discrepancy(
        per_expert_array_equal=bool(metrics["per_expert"]["array_equal"]),
        fused_matches_stock_reduce=bool(
            metrics["fused_reduce_given_candidate_down"]["array_equal"]
        ),
    )
    metrics["fixture"] = fixture
    metrics["route_weights"] = {
        "dtype": str(route_weights.dtype),
        "min": float(mx.min(route_weights.astype(mx.float32)).item()),
        "max": float(mx.max(route_weights.astype(mx.float32)).item()),
        "row_sum_max_abs_from_one": float(
            mx.max(
                mx.abs(
                    route_weights.astype(mx.float32).sum(axis=-1)
                    - mx.array(1.0, dtype=mx.float32)
                )
            ).item()
        ),
    }
    exact_ok = candidate.activation_mode == "exact" and bool(metrics["passes"])
    return metrics, exact_ok


def run_campaign(
    *,
    phase: str,
    model: Path,
    layer: int,
    experts: tuple[int, ...],
    rows_selected: tuple[int, ...],
    candidates: tuple[Hy3ExpertE2Candidate, ...],
    families: tuple[str, ...],
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
    adversarial: bool,
) -> dict[str, Any]:
    source, component_metadata = _load_source_bank(
        model,
        layer=layer,
        experts=experts,
    )
    compiled_stock = _compiled_stock(source) if "stock_compiled" in families else None
    exact_candidates = tuple(
        candidate for candidate in candidates if candidate.activation_mode == "exact"
    )
    fast_candidates = tuple(
        candidate for candidate in candidates if candidate.activation_mode == "fast"
    )
    ordered_candidates = exact_candidates + fast_candidates
    passing_exact_by_row: dict[int, set[tuple[int, int, int]]] = {
        rows: set() for rows in rows_selected
    }
    failures: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []

    for rows in rows_selected:
        token_rows = _token_rows(rows, seed=6600 + rows)
        primary_slots = _expert_slots(rows, fixture="router_like")
        primary_weights = _route_weights(rows, fixture="router_like", seed=6610 + rows)
        exact_hidden = _stock_hidden(token_rows, primary_slots, source)
        stock_per_expert, stock_final = hy3_expert_e2_reference(
            exact_hidden,
            primary_slots,
            primary_weights,
            source["down_proj.weight"],
            source["down_proj.scales"],
            source["down_proj.biases"],
        )
        _evaluate((exact_hidden, stock_per_expert, stock_final))

        def stock_full() -> mx.array:
            return _stock_full(
                token_rows,
                primary_slots,
                primary_weights,
                source,
            )

        def stock_e2() -> mx.array:
            return hy3_expert_e2_reference(
                exact_hidden,
                primary_slots,
                primary_weights,
                source["down_proj.weight"],
                source["down_proj.scales"],
                source["down_proj.biases"],
            )[1]

        row_result: dict[str, Any] = {
            "rows": rows,
            "primary_target": rows == 4,
            "stock": {
                "per_expert_shape": list(stock_per_expert.shape),
                "final_shape": list(stock_final.shape),
                "dtype": str(stock_final.dtype),
                "intermediate_bytes": rows * HY3_E2_TOP_K * HY3_E2_HIDDEN_SIZE * 2,
            },
            "families": [],
            "candidates": [],
        }
        if compiled_stock is not None:

            def compiled_call() -> mx.array:
                return compiled_stock(
                    token_rows,
                    primary_slots,
                    primary_weights,
                )

            try:
                compiled_value = compiled_call()
                _evaluate(compiled_value)
                row_result["families"].append(
                    {
                        "family": "stock_compiled",
                        "correctness": _tensor_metrics(compiled_value, stock_final),
                        "full_core_timing": _measure_pair(
                            stock_full,
                            compiled_call,
                            warmups=warmups,
                            repeats=repeats,
                            bootstrap_resamples=bootstrap_resamples,
                            seed=6620 + rows,
                        ),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "rows": rows,
                        "family": "stock_compiled",
                        "phase": "compile",
                        "classification": classify_candidate_failure(
                            error, phase="compile"
                        ),
                        "error": repr(error),
                    }
                )

        for candidate in ordered_candidates:
            if not fast_prerequisite_met(candidate, passing_exact_by_row[rows]):
                row_result["candidates"].append(
                    {
                        **candidate.to_dict(rows=rows),
                        "status": "skipped_fast_without_matching_exact_pass",
                    }
                )
                continue
            fixtures = ["router_like"]
            if adversarial and rows == 4:
                fixtures.extend(("reverse_order", "peaked"))
            correctness: list[dict[str, Any]] = []
            exact_ok = True
            try:
                for fixture_index, fixture in enumerate(fixtures):
                    metrics, fixture_ok = _correctness_fixture(
                        rows=rows,
                        fixture=fixture,
                        token_rows=token_rows,
                        source=source,
                        candidate=candidate,
                        seed=6630 + rows * 10 + fixture_index,
                    )
                    correctness.append(metrics)
                    if candidate.activation_mode == "exact":
                        exact_ok = exact_ok and fixture_ok
            except Exception as error:
                failure = {
                    "rows": rows,
                    "candidate": candidate.name,
                    "phase": "first_eval",
                    "classification": classify_candidate_failure(
                        error, phase="first_eval"
                    ),
                    "error": repr(error),
                }
                failures.append(failure)
                row_result["candidates"].append(
                    {
                        **candidate.to_dict(rows=rows),
                        "status": "failed",
                        "failure": failure,
                    }
                )
                mx.clear_cache()
                continue
            if candidate.activation_mode == "exact" and exact_ok:
                passing_exact_by_row[rows].add(exact_candidate_key(candidate))
            candidate_result: dict[str, Any] = {
                **candidate.to_dict(rows=rows),
                "status": (
                    "correctness_pass"
                    if candidate.activation_mode == "exact" and exact_ok
                    else "fast_ablation"
                    if candidate.activation_mode == "fast"
                    else "correctness_reject"
                ),
                "correctness": correctness,
            }
            if "fused_e2" in families:

                def candidate_e2() -> mx.array:
                    hidden = (
                        exact_hidden
                        if candidate.activation_mode == "exact"
                        else _candidate_hidden(
                            token_rows,
                            primary_slots,
                            source,
                            activation_mode="fast",
                        )
                    )
                    output = hy3_expert_e2_fused(
                        hidden,
                        primary_slots,
                        primary_weights,
                        source["down_proj.weight"],
                        source["down_proj.scales"],
                        source["down_proj.biases"],
                        candidate=candidate,
                    )
                    assert isinstance(output, mx.array)
                    return output

                def candidate_full() -> mx.array:
                    return _candidate_full(
                        token_rows,
                        primary_slots,
                        primary_weights,
                        source,
                        candidate=candidate,
                    )

                # Exact E2-only compares equal E1 values. Fast E2-only includes
                # its activation cost and is therefore labelled full-path only.
                if candidate.activation_mode == "exact":
                    candidate_result["e2_only_timing"] = _measure_pair(
                        stock_e2,
                        candidate_e2,
                        warmups=warmups,
                        repeats=repeats,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=6640 + rows,
                    )
                candidate_result["full_core_timing"] = _measure_pair(
                    stock_full,
                    candidate_full,
                    warmups=warmups,
                    repeats=repeats,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=6650 + rows,
                )
            row_result["candidates"].append(candidate_result)
            print(
                f"completed E2 {candidate.name} R{rows} status={candidate_result['status']}",
                file=sys.stderr,
                flush=True,
            )
        output_rows.append(row_result)
        mx.clear_cache()

    return {
        "schema": "mtplx-issue66-hy3-q2-e2-full-v1",
        "issue": 66,
        "phase": phase,
        "scope": (
            "full expert core gate+up -> exact/fast SwiGLU -> Q2 down -> "
            "BF16 router-order weighted reduce; benchmark-only, no runtime integration"
        ),
        "model": str(model),
        "component_bank": component_metadata,
        "geometry": {
            "rows": list(rows_selected),
            "selected_experts_per_token": HY3_E2_TOP_K,
            "hidden_size": HY3_E2_HIDDEN_SIZE,
            "intermediate_size": HY3_E2_INTERMEDIATE_SIZE,
            "quantization": {
                "bits": HY3_E2_BITS,
                "group_size": HY3_E2_GROUP_SIZE,
                "mode": "affine",
            },
        },
        "contract": {
            "e1_output": "BF16 [R,8,1536]",
            "per_expert_down_reference": "BF16 [R,8,4096]",
            "route_weights": "BF16 [R,8]",
            "multiply_boundary": "BF16",
            "accumulation_order": "existing router slot order 0 through 7",
            "fused_output": "BF16 [R,4096]",
            "fast_math_policy": "ablation only; never promotion eligible",
        },
        "measurement": {
            "warmups": warmups,
            "paired_interleaved_repeats": repeats,
            "bootstrap_resamples": bootstrap_resamples,
            "primary_row": "R4 corresponding to K3/M4",
            "correctness_fixtures": list(FIXTURES if adversarial else FIXTURES[:1]),
            "adversarial_fixtures_run_at": "R4 only",
        },
        "families": list(families),
        "candidates": [candidate.to_dict(rows=4) for candidate in candidates],
        "failures": failures,
        "rows": output_rows,
        "source_sha256": _source_hashes(),
        "environment": {"device": mx.device_info()},
    }


def _csv_ints(value: str, *, allowed: tuple[int, ...]) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from error
    if not result or len(set(result)) != len(result) or set(result) - set(allowed):
        raise argparse.ArgumentTypeError(
            f"values must be unique selections from {allowed}"
        )
    return result


def _csv_strings(value: str, *, allowed: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result) or set(result) - set(allowed):
        raise argparse.ArgumentTypeError(
            f"values must be unique selections from {allowed}"
        )
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _experts(value: str) -> tuple[int, ...]:
    result = _csv_ints(value, allowed=tuple(range(192)))
    if len(result) != HY3_E2_TOP_K:
        raise argparse.ArgumentTypeError("experts must contain eight unique values")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "refine"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--experts", type=_experts, default=tuple(range(8)))
    parser.add_argument(
        "--rows",
        type=lambda value: _csv_ints(value, allowed=HY3_E2_ROWS),
        default=HY3_E2_ROWS,
    )
    parser.add_argument(
        "--n-tiles",
        type=lambda value: _csv_ints(value, allowed=HY3_E2_OUTPUT_N_TILES),
        default=HY3_E2_OUTPUT_N_TILES,
    )
    parser.add_argument(
        "--k-tiles",
        type=lambda value: _csv_ints(value, allowed=HY3_E2_K_TILES),
        default=HY3_E2_K_TILES,
    )
    parser.add_argument(
        "--simd-groups",
        type=lambda value: _csv_ints(value, allowed=HY3_E2_SIMD_GROUPS),
        default=HY3_E2_SIMD_GROUPS,
    )
    parser.add_argument(
        "--activation-modes",
        type=lambda value: _csv_strings(value, allowed=HY3_E2_ACTIVATION_MODES),
        default=HY3_E2_ACTIVATION_MODES,
    )
    parser.add_argument(
        "--families",
        type=lambda value: _csv_strings(value, allowed=FAMILIES),
        default=FAMILIES,
    )
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument("--warmups", type=_positive_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=2_000)
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument("--lock-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidates = select_candidates(
        n_tiles=args.n_tiles,
        k_tiles=args.k_tiles,
        simd_groups=args.simd_groups,
        activation_modes=args.activation_modes,
        candidate_names=tuple(args.candidate_name),
    )
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
        result = run_campaign(
            phase=args.phase,
            model=args.model.expanduser().resolve(),
            layer=args.layer,
            experts=args.experts,
            rows_selected=args.rows,
            candidates=candidates,
            families=args.families,
            warmups=args.warmups,
            repeats=args.repeats,
            bootstrap_resamples=args.bootstrap_resamples,
            adversarial=not args.no_adversarial,
        )
        result["exclusive_window"] = {
            "lock_path": str(receipt.lock_path),
            "captured_qwen_loaded": receipt.qwen_state.loaded,
            "captured_qwen_models": list(receipt.qwen_state.models),
        }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
