#!/usr/bin/env python3
"""Diagnose E1 r1 and benchmark packed-MLX E1 r2 on real Hy3-Q2 tensors."""

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
import mlx.nn as nn
import numpy as np

from mtplx.expert_manifest import load_expert_manifest, read_expert_record
from mtplx.hy3_expert_e1_r2 import (
    HY3_E1_R2_ASSIGNMENT_ORDERS,
    HY3_E1_R2_EXECUTIONS,
    HY3_E1_R2_LAYOUTS,
    Hy3E1R2Candidate,
    PackedHy3E1Bank,
    classify_stage_discrepancy,
    compile_packed_hy3_e1,
    hy3_e1_r2_candidates,
    pack_hy3_e1_component_bank,
    packed_hy3_e1,
    packed_hy3_e1_qmm,
    r1_hy3_e1_preactivations,
    scalar_affine_q2_dot,
    simulate_r1_affine_q2_dot,
    split_packed_hy3_e1_output,
    unpack_affine_q2_row,
)
from mtplx.hy3_expert_fused_e1 import (
    HY3_E1_BITS,
    HY3_E1_GROUP_SIZE,
    HY3_E1_HIDDEN_SIZE,
    HY3_E1_INTERMEDIATE_SIZE,
    HY3_E1_ROWS,
    HY3_E1_TOP_K,
    Hy3ExpertE1Candidate,
    hy3_expert_fused_e1,
)
from mtplx.qwen_guard import DEFAULT_MLX_LOCK_PATH, exclusive_mlx_window


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
    raise ValueError(f"unsupported E1 r2 component dtype: {dtype}")


def _load_source_bank(
    model: Path,
    *,
    layer: int,
    experts: tuple[int, ...],
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    if len(experts) != HY3_E1_TOP_K or len(set(experts)) != HY3_E1_TOP_K:
        raise ValueError("E1 r2 requires eight distinct component-bank experts")
    manifest_path = model / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    if (
        manifest.model_key != "hy3-expert-q2"
        or manifest.quant_bits != HY3_E1_BITS
        or manifest.quant_group_size != HY3_E1_GROUP_SIZE
        or manifest.quant_mode != "affine"
    ):
        raise ValueError("E1 r2 requires the authoritative Hy3 affine-Q2 model")
    wanted = (
        "gate_proj.weight",
        "gate_proj.scales",
        "gate_proj.biases",
        "up_proj.weight",
        "up_proj.scales",
        "up_proj.biases",
    )
    rows: dict[str, list[mx.array]] = {name: [] for name in wanted}
    records = []
    for expert in experts:
        record = manifest.record(layer, expert)
        payload = read_expert_record(manifest, model, layer, expert)
        cursor = 0
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
        if cursor != len(payload):
            raise ValueError("E1 r2 record segments do not cover the payload")
        records.append(
            {
                "expert": int(expert),
                "logical_bytes": int(record.logical_bytes),
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
        "source_component_bytes": sum(int(value.nbytes) for value in bank.values()),
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


def _stock_preactivations(
    token_rows: mx.array,
    slots: mx.array,
    source: dict[str, mx.array],
) -> tuple[mx.array, mx.array]:
    rows = int(token_rows.shape[0])
    assignments = rows * HY3_E1_TOP_K
    selected = mx.broadcast_to(
        token_rows[:, None, :],
        (rows, HY3_E1_TOP_K, HY3_E1_HIDDEN_SIZE),
    ).reshape(assignments, 1, 1, HY3_E1_HIDDEN_SIZE)
    indices = slots.reshape(assignments, 1)

    def qmm(projection: str) -> mx.array:
        output = mx.gather_qmm(
            selected,
            source[f"{projection}.weight"],
            source[f"{projection}.scales"],
            source[f"{projection}.biases"],
            rhs_indices=indices,
            transpose=True,
            group_size=HY3_E1_GROUP_SIZE,
            bits=HY3_E1_BITS,
            mode="affine",
        )
        return output.reshape(rows, HY3_E1_TOP_K, HY3_E1_INTERMEDIATE_SIZE)

    return qmm("gate_proj"), qmm("up_proj")


def _stock_e1(
    token_rows: mx.array,
    slots: mx.array,
    source: dict[str, mx.array],
) -> mx.array:
    gate, up = _stock_preactivations(token_rows, slots, source)
    return nn.silu(gate) * up


def _tensor_metrics(observed: mx.array, reference: mx.array) -> dict[str, Any]:
    shape_exact = tuple(observed.shape) == tuple(reference.shape)
    if not shape_exact:
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
    indices = rng.integers(
        0,
        control_array.size,
        size=(resamples, control_array.size),
    )
    boot = control_array[indices].mean(axis=1) / candidate_array[indices].mean(axis=1)
    return {
        "control_over_candidate_ratio_of_means": float(
            control_array.mean() / candidate_array.mean()
        ),
        "paired_ratio_mean": float(np.mean(control_array / candidate_array)),
        "paired_ratio_median": float(np.median(control_array / candidate_array)),
        "bootstrap_mean_ratio_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
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
        "arms": {name: _sample_summary(values) for name, values in samples.items()},
        "comparison": _paired_bootstrap(
            samples["stock"],
            samples["candidate"],
            resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def _to_numpy_f32(value: mx.array) -> np.ndarray:
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def _to_numpy_u32(value: mx.array) -> np.ndarray:
    return np.asarray(value, dtype=np.uint32)


def _bf16_scalar(value: float) -> float:
    array = mx.array([value], dtype=mx.float32).astype(mx.bfloat16).astype(mx.float32)
    return float(array.item())


def _scalar_probe(
    *,
    token_rows: mx.array,
    slots: mx.array,
    source: dict[str, mx.array],
    stock_gate: mx.array,
    stock_up: mx.array,
    r1_gate: mx.array,
    r1_up: mx.array,
    stock_final: mx.array,
    r1_final: mx.array,
    candidate: Hy3ExpertE1Candidate,
) -> dict[str, Any]:
    difference = mx.abs(r1_final.astype(mx.float32) - stock_final.astype(mx.float32))
    flat = int(mx.argmax(difference.reshape(-1)).item())
    row, selected_position, output_index = np.unravel_index(flat, difference.shape)
    expert = int(slots[row, selected_position].item())
    activation = _to_numpy_f32(token_rows[row])
    projection_results: dict[str, Any] = {}
    decoded_matches = True
    reduction_matches = True
    for projection, stock_values, r1_values in (
        ("gate_proj", stock_gate, r1_gate),
        ("up_proj", stock_up, r1_up),
    ):
        words_mx = source[f"{projection}.weight"][expert, output_index]
        scales_mx = source[f"{projection}.scales"][expert, output_index]
        biases_mx = source[f"{projection}.biases"][expert, output_index]
        words = _to_numpy_u32(words_mx)
        scales = _to_numpy_f32(scales_mx)
        biases = _to_numpy_f32(biases_mx)
        decoded = unpack_affine_q2_row(words, scales, biases)
        mlx_decoded = mx.dequantize(
            words_mx[None, :],
            scales_mx[None, :],
            biases_mx[None, :],
            group_size=HY3_E1_GROUP_SIZE,
            bits=HY3_E1_BITS,
            mode="affine",
            dtype=mx.float32,
        )
        _evaluate(mlx_decoded)
        decoded_max_abs = float(
            np.max(np.abs(decoded - _to_numpy_f32(mlx_decoded).reshape(-1)))
        )
        decoded_matches = decoded_matches and decoded_max_abs == 0.0
        scalar = scalar_affine_q2_dot(activation, words, scales, biases)
        simulated = simulate_r1_affine_q2_dot(
            activation,
            words,
            scales,
            biases,
            k_tile=candidate.k_tile,
        )
        stock_value = float(stock_values[row, selected_position, output_index].item())
        r1_value = float(r1_values[row, selected_position, output_index].item())
        scalar_bf16 = _bf16_scalar(scalar)
        simulated_bf16 = _bf16_scalar(simulated)
        projection_match = simulated_bf16 == r1_value
        reduction_matches = reduction_matches and projection_match
        projection_results[projection] = {
            "decoded_max_abs_vs_mx_dequantize_f32": decoded_max_abs,
            "scalar_float64_dot": scalar,
            "scalar_bf16": scalar_bf16,
            "stock_gather_qmm_bf16": stock_value,
            "r1_bf16": r1_value,
            "simulated_r1_float32": simulated,
            "simulated_r1_bf16": simulated_bf16,
            "stock_matches_scalar_bf16": stock_value == scalar_bf16,
            "r1_matches_simulated_bf16": projection_match,
        }
    return {
        "location": {
            "token_row": int(row),
            "selected_position": int(selected_position),
            "expert_slot": int(expert),
            "output_index": int(output_index),
            "final_abs_error": float(
                difference[row, selected_position, output_index].item()
            ),
        },
        "decoded_values_match": decoded_matches,
        "r1_matches_simulated_reduction": reduction_matches,
        "projections": projection_results,
    }


def run_diagnostic(
    *,
    model: Path,
    layer: int,
    experts: tuple[int, ...],
) -> dict[str, Any]:
    source, component_metadata = _load_source_bank(model, layer=layer, experts=experts)
    candidate = Hy3ExpertE1Candidate(32, 256, 8, "exact")
    result_rows = []
    classifications = []
    for rows in HY3_E1_ROWS:
        token_rows = _token_rows(rows, seed=551_000 + rows)
        slots = _expert_slots(rows)
        stock_gate, stock_up = _stock_preactivations(token_rows, slots, source)
        r1_gate, r1_up = r1_hy3_e1_preactivations(
            token_rows,
            slots,
            source,
            candidate=candidate,
        )
        stock_silu = nn.silu(stock_gate)
        r1_silu = nn.silu(r1_gate)
        stock_final = stock_silu * stock_up
        r1_from_parts = r1_silu * r1_up
        r1_fused = hy3_expert_fused_e1(
            token_rows,
            slots,
            source["gate_proj.weight"],
            source["gate_proj.scales"],
            source["gate_proj.biases"],
            source["up_proj.weight"],
            source["up_proj.scales"],
            source["up_proj.biases"],
            candidate=candidate,
        )
        _evaluate(
            (
                stock_gate,
                stock_up,
                r1_gate,
                r1_up,
                stock_silu,
                r1_silu,
                stock_final,
                r1_from_parts,
                r1_fused,
            )
        )
        scalar_probe = _scalar_probe(
            token_rows=token_rows,
            slots=slots,
            source=source,
            stock_gate=stock_gate,
            stock_up=stock_up,
            r1_gate=r1_gate,
            r1_up=r1_up,
            stock_final=stock_final,
            r1_final=r1_fused,
            candidate=candidate,
        )
        activation_matches = bool(mx.array_equal(r1_from_parts, r1_fused).item())
        bf16_cast_matches = r1_fused.dtype == mx.bfloat16
        classification = classify_stage_discrepancy(
            decoded_values_match=scalar_probe["decoded_values_match"],
            r1_matches_simulated_reduction=scalar_probe[
                "r1_matches_simulated_reduction"
            ],
            activation_matches_from_preactivations=activation_matches,
            bf16_cast_matches=bf16_cast_matches,
        )
        classifications.append(classification)
        result_rows.append(
            {
                "m": int(rows),
                "gate_preactivation": _tensor_metrics(r1_gate, stock_gate),
                "up_preactivation": _tensor_metrics(r1_up, stock_up),
                "silu": _tensor_metrics(r1_silu, stock_silu),
                "multiplication_from_preactivations": _tensor_metrics(
                    r1_from_parts,
                    stock_final,
                ),
                "inline_fused_vs_r1_preactivation_graph": _tensor_metrics(
                    r1_fused,
                    r1_from_parts,
                ),
                "inline_fused_vs_stock": _tensor_metrics(r1_fused, stock_final),
                "scalar_probe": scalar_probe,
                "classification": classification,
            }
        )
        print(f"completed E1 r1 stage diagnostic R{rows}", file=sys.stderr, flush=True)
    return {
        "schema": "mtplx-issue65-hy3-q2-e1-r1-stage-diagnostic-v1",
        "issues": {"e1_arithmetic": 65, "component_layout": 68},
        "scope": "r1 attribution only; no integration or promotion",
        "model": str(model),
        "component_bank": component_metadata,
        "candidate": candidate.to_dict(),
        "rows": result_rows,
        "classification_summary": {
            value: classifications.count(value)
            for value in sorted(set(classifications))
        },
        "environment": {"device": mx.device_info()},
    }


def _candidate_call(
    candidate: Hy3E1R2Candidate,
    packed: PackedHy3E1Bank,
    compiled: Callable[[mx.array, mx.array], mx.array] | None,
    token_rows: mx.array,
    slots: mx.array,
) -> mx.array:
    if candidate.execution == "compiled":
        if compiled is None:
            raise RuntimeError("compiled E1 candidate has no compiled callable")
        return compiled(token_rows, slots)
    return packed_hy3_e1(
        token_rows,
        slots,
        packed,
        assignment_order=candidate.assignment_order,
    )


def run_profile(
    *,
    model: Path,
    layer: int,
    experts: tuple[int, ...],
    candidates: tuple[Hy3E1R2Candidate, ...],
    warmups: int,
    repeats: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    source, component_metadata = _load_source_bank(model, layer=layer, experts=experts)
    layouts = tuple(dict.fromkeys(candidate.layout for candidate in candidates))
    layout_metadata: dict[str, Any] = {}
    rows_by_m: dict[int, dict[str, Any]] = {
        rows: {
            "m": rows,
            "token_expert_assignments": rows * HY3_E1_TOP_K,
            "variants": [],
        }
        for rows in HY3_E1_ROWS
    }
    failures: dict[str, list[dict[str, Any]]] = {
        "construction": [],
        "compile": [],
        "dispatch": [],
        "correctness": [],
    }
    for layout_index, layout in enumerate(layouts):
        construction_started = time.perf_counter_ns()
        try:
            packed = pack_hy3_e1_component_bank(source, layout=layout)
        except Exception as exc:
            failures["construction"].append(
                {"layout": layout, "error_type": type(exc).__name__, "error": str(exc)}
            )
            continue
        construction_ms = (time.perf_counter_ns() - construction_started) / 1_000_000
        layout_metadata[layout] = {
            "packed_shape": list(packed.weight.shape),
            "source_bytes": packed.source_bytes,
            "packed_bytes": packed.packed_bytes,
            "steady_state_duplicate_bytes": packed.packed_bytes - packed.source_bytes,
            "construction_ms": construction_ms,
            "production_policy": "replace gate/up source arrays after validation",
            "relationship": "layout mechanics tracked separately by issue #68",
        }
        layout_candidates = tuple(
            candidate for candidate in candidates if candidate.layout == layout
        )
        for rows in HY3_E1_ROWS:
            token_rows = _token_rows(rows, seed=561_000 + rows)
            slots = _expert_slots(rows)
            stock_gate, stock_up = _stock_preactivations(token_rows, slots, source)
            stock_final = nn.silu(stock_gate) * stock_up
            _evaluate((stock_gate, stock_up, stock_final))
            component_cache: dict[tuple[str, str], dict[str, Any]] = {}
            compiled_cache: dict[str, Callable[[mx.array, mx.array], mx.array]] = {}
            compile_ms: dict[str, float] = {}
            for candidate_index, candidate in enumerate(layout_candidates):
                key = (candidate.layout, candidate.assignment_order)
                if key not in component_cache:
                    packed_output = packed_hy3_e1_qmm(
                        token_rows,
                        slots,
                        packed,
                        assignment_order=candidate.assignment_order,
                    )
                    packed_gate, packed_up = split_packed_hy3_e1_output(
                        packed_output,
                        layout=layout,
                    )
                    packed_gate = packed_gate.reshape(stock_gate.shape)
                    packed_up = packed_up.reshape(stock_up.shape)
                    packed_activation = nn.silu(packed_gate) * packed_up
                    _evaluate(
                        (packed_output, packed_gate, packed_up, packed_activation)
                    )

                    def stock_qmm() -> tuple[mx.array, mx.array]:
                        return _stock_preactivations(token_rows, slots, source)

                    def candidate_qmm(
                        packed: PackedHy3E1Bank = packed,
                        candidate: Hy3E1R2Candidate = candidate,
                    ) -> mx.array:
                        return packed_hy3_e1_qmm(
                            token_rows,
                            slots,
                            packed,
                            assignment_order=candidate.assignment_order,
                        )

                    def stock_activation() -> mx.array:
                        return nn.silu(stock_gate) * stock_up

                    def candidate_activation() -> mx.array:
                        live_gate, live_up = split_packed_hy3_e1_output(
                            packed_output,
                            layout=layout,
                        )
                        return nn.silu(live_gate) * live_up

                    component_cache[key] = {
                        "gate_preactivation": _tensor_metrics(packed_gate, stock_gate),
                        "up_preactivation": _tensor_metrics(packed_up, stock_up),
                        "activation": _tensor_metrics(packed_activation, stock_final),
                        "packed_qmm_only": _measure_pair(
                            stock_qmm,
                            candidate_qmm,
                            warmups=warmups,
                            repeats=repeats,
                            bootstrap_resamples=bootstrap_resamples,
                            seed=565_000 + layout_index * 10_000 + rows * 100,
                        ),
                        "split_exact_swiglu_only": _measure_pair(
                            stock_activation,
                            candidate_activation,
                            warmups=warmups,
                            repeats=repeats,
                            bootstrap_resamples=bootstrap_resamples,
                            seed=566_000 + layout_index * 10_000 + rows * 100,
                        ),
                    }
                compiled: Callable[[mx.array, mx.array], mx.array] | None = None
                if candidate.execution == "compiled":
                    try:
                        started = time.perf_counter_ns()
                        compiled = compile_packed_hy3_e1(
                            packed,
                            assignment_order=candidate.assignment_order,
                        )
                        first = compiled(token_rows, slots)
                        _evaluate(first)
                        compile_ms[candidate.name] = (
                            time.perf_counter_ns() - started
                        ) / 1_000_000
                        compiled_cache[candidate.name] = compiled
                    except Exception as exc:
                        failure = {
                            "m": rows,
                            "candidate": candidate.name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failures["compile"].append(failure)
                        rows_by_m[rows]["variants"].append(
                            {
                                **candidate.to_dict(),
                                "status": "compile_failure",
                                "failure": failure,
                            }
                        )
                        continue

                def candidate_complete(
                    candidate: Hy3E1R2Candidate = candidate,
                    compiled: Callable[[mx.array, mx.array], mx.array]
                    | None = compiled,
                    packed: PackedHy3E1Bank = packed,
                ) -> mx.array:
                    return _candidate_call(
                        candidate,
                        packed,
                        compiled,
                        token_rows,
                        slots,
                    )

                try:
                    observed = candidate_complete()
                    _evaluate(observed)
                except Exception as exc:
                    failure = {
                        "m": rows,
                        "candidate": candidate.name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures["dispatch"].append(failure)
                    rows_by_m[rows]["variants"].append(
                        {
                            **candidate.to_dict(),
                            "status": "dispatch_failure",
                            "failure": failure,
                        }
                    )
                    continue
                correctness = _tensor_metrics(observed, stock_final)
                if not correctness["array_equal"]:
                    failure = {
                        "m": rows,
                        "candidate": candidate.name,
                        "max_abs_error": correctness["max_abs_error"],
                        "normalized_rmse": correctness["normalized_rmse"],
                    }
                    failures["correctness"].append(failure)
                    rows_by_m[rows]["variants"].append(
                        {
                            **candidate.to_dict(),
                            "status": "correctness_failure",
                            "correctness": correctness,
                            "component_profile": component_cache[key],
                        }
                    )
                    continue
                try:
                    complete_timing = _measure_pair(
                        lambda: _stock_e1(token_rows, slots, source),
                        candidate_complete,
                        warmups=warmups,
                        repeats=repeats,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=(
                            567_000
                            + layout_index * 10_000
                            + rows * 100
                            + candidate_index
                        ),
                    )
                except Exception as exc:
                    failure = {
                        "m": rows,
                        "candidate": candidate.name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures["dispatch"].append(failure)
                    rows_by_m[rows]["variants"].append(
                        {
                            **candidate.to_dict(),
                            "status": "timing_failure",
                            "failure": failure,
                        }
                    )
                    continue
                rows_by_m[rows]["variants"].append(
                    {
                        **candidate.to_dict(),
                        "status": "measured_exact_candidate",
                        "correctness": correctness,
                        "compile_and_first_eval_ms": compile_ms.get(candidate.name),
                        "component_profile": component_cache[key],
                        "complete_e1": complete_timing,
                    }
                )
            print(
                f"completed packed E1 layout={layout} R{rows}",
                file=sys.stderr,
                flush=True,
            )
        del packed
        mx.clear_cache()
    return {
        "schema": "mtplx-issue65-hy3-q2-e1-packed-r2-v1",
        "issues": {"e1_arithmetic": 65, "component_layout": 68},
        "scope": "E1 gate+up+exact-SwiGLU only; no E2 or runtime integration",
        "model": str(model),
        "component_bank": component_metadata,
        "geometry": {
            "rows": list(HY3_E1_ROWS),
            "selected_experts_per_token": HY3_E1_TOP_K,
            "hidden_size": HY3_E1_HIDDEN_SIZE,
            "packed_output_size": HY3_E1_INTERMEDIATE_SIZE * 2,
            "quantization": {
                "bits": HY3_E1_BITS,
                "group_size": HY3_E1_GROUP_SIZE,
                "mode": "affine",
            },
        },
        "measurement": {
            "warmups": warmups,
            "paired_interleaved_repeats": repeats,
            "bootstrap_resamples": bootstrap_resamples,
            "primary_row": "R4 corresponding to K3",
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
        "layouts": layout_metadata,
        "failures": failures,
        "rows": [rows_by_m[rows] for rows in HY3_E1_ROWS],
        "environment": {"device": mx.device_info()},
    }


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
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "experts must be comma-separated integers"
        ) from exc
    if len(result) != HY3_E1_TOP_K or len(set(result)) != HY3_E1_TOP_K:
        raise argparse.ArgumentTypeError("experts must contain eight unique values")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("diagnose", "profile"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--experts", type=_experts, default=tuple(range(8)))
    parser.add_argument(
        "--layouts",
        type=lambda value: _csv_strings(value, allowed=HY3_E1_R2_LAYOUTS),
        default=HY3_E1_R2_LAYOUTS,
    )
    parser.add_argument(
        "--assignment-orders",
        type=lambda value: _csv_strings(value, allowed=HY3_E1_R2_ASSIGNMENT_ORDERS),
        default=HY3_E1_R2_ASSIGNMENT_ORDERS,
    )
    parser.add_argument(
        "--executions",
        type=lambda value: _csv_strings(value, allowed=HY3_E1_R2_EXECUTIONS),
        default=HY3_E1_R2_EXECUTIONS,
    )
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument("--warmups", type=_positive_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=2_000)
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument("--lock-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _select_candidates(args: argparse.Namespace) -> tuple[Hy3E1R2Candidate, ...]:
    candidates = tuple(
        candidate
        for candidate in hy3_e1_r2_candidates()
        if candidate.layout in args.layouts
        and candidate.assignment_order in args.assignment_orders
        and candidate.execution in args.executions
        and (not args.candidate_name or candidate.name in args.candidate_name)
    )
    if not candidates:
        raise ValueError("E1 r2 filters select no candidates")
    unknown = set(args.candidate_name) - {candidate.name for candidate in candidates}
    if unknown:
        raise ValueError(
            f"unknown or filtered E1 r2 candidate names: {sorted(unknown)}"
        )
    return candidates


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidates = _select_candidates(args) if args.phase == "profile" else ()
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
        common = {
            "model": args.model.expanduser().resolve(),
            "layer": args.layer,
            "experts": args.experts,
        }
        if args.phase == "diagnose":
            result = run_diagnostic(**common)
        else:
            result = run_profile(
                **common,
                candidates=candidates,
                warmups=args.warmups,
                repeats=args.repeats,
                bootstrap_resamples=args.bootstrap_resamples,
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
