#!/usr/bin/env python3
"""Measure byte-neutral Hy3-Q2 expert layouts for issue #68."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from mtplx.expert_manifest import load_expert_manifest, read_expert_record
from mtplx.hy3_q2_layout_coalescing import (
    HY3_Q2_BLOCK_SIZES,
    HY3_Q2_LAYOUTS,
    HY3_Q2_PRIMARY_ROW,
    HY3_Q2_ROWS,
    HY3_Q2_TOP_K,
    HY3_Q2_VECTOR_WIDTHS,
    Hy3Q2LayoutCandidate,
    Hy3Q2SlotIdentity,
    Hy3Q2SourceControl,
    hy3_q2_coalescing_evidence,
    hy3_q2_component_shapes,
    hy3_q2_components_to_mlx,
    hy3_q2_layout_candidates,
    hy3_q2_source_controls,
    pack_hy3_q2_bank,
    scan_hy3_q2_packed,
    scan_hy3_q2_source,
)
from mtplx.qwen_guard import DEFAULT_MLX_LOCK_PATH, exclusive_mlx_window


def _csv_ints(value: str, *, allowed: tuple[int, ...]) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/huggingface/hy3-expert-only-mlx-q2",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--experts",
        type=lambda value: _csv_ints(value, allowed=tuple(range(192))),
        default=tuple(range(8)),
    )
    parser.add_argument(
        "--rows",
        type=lambda value: _csv_ints(value, allowed=HY3_Q2_ROWS),
        default=HY3_Q2_ROWS,
    )
    parser.add_argument("--primary-row", type=int, default=HY3_Q2_PRIMARY_ROW)
    parser.add_argument(
        "--layouts",
        type=lambda value: _csv_strings(value, allowed=HY3_Q2_LAYOUTS),
        default=HY3_Q2_LAYOUTS,
    )
    parser.add_argument(
        "--block-sizes",
        type=lambda value: _csv_ints(value, allowed=HY3_Q2_BLOCK_SIZES),
        default=HY3_Q2_BLOCK_SIZES,
    )
    parser.add_argument(
        "--vector-widths",
        type=lambda value: _csv_ints(value, allowed=HY3_Q2_VECTOR_WIDTHS),
        default=HY3_Q2_VECTOR_WIDTHS,
    )
    parser.add_argument("--screen-repeats", type=_positive_int, default=3)
    parser.add_argument("--refine-repeats", type=_positive_int, default=11)
    parser.add_argument("--warmups", type=_positive_int, default=2)
    parser.add_argument("--finalists", type=_positive_int, default=6)
    parser.add_argument("--bootstrap-resamples", type=_positive_int, default=10_000)
    parser.add_argument(
        "--qwen-plist",
        type=Path,
        default=Path.home() / "Library/LaunchAgents/com.tea.qwen.plist",
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_MLX_LOCK_PATH)
    parser.add_argument("--lock-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _select_candidates(args: argparse.Namespace) -> tuple[Hy3Q2LayoutCandidate, ...]:
    selected = tuple(
        candidate
        for candidate in hy3_q2_layout_candidates()
        if candidate.layout in args.layouts
        and candidate.vector_width in args.vector_widths
        and (
            not candidate.layout.startswith("tiled_")
            or candidate.block_size in args.block_sizes
        )
    )
    if not selected:
        raise ValueError("layout filters selected no candidates")
    return selected


def _paired_summary(
    control: list[float],
    candidate: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    control_array = np.asarray(control, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if (
        control_array.ndim != 1
        or candidate_array.shape != control_array.shape
        or control_array.size == 0
        or not np.all(np.isfinite(control_array))
        or not np.all(np.isfinite(candidate_array))
        or np.any(control_array <= 0)
        or np.any(candidate_array <= 0)
    ):
        raise ValueError("paired timing samples must be equal positive finite vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        control_array.size,
        size=(int(resamples), control_array.size),
    )
    bootstrap = control_array[indices].mean(axis=1) / candidate_array[indices].mean(
        axis=1
    )
    ratios = control_array / candidate_array
    return {
        "control_samples_ms": control_array.tolist(),
        "candidate_samples_ms": candidate_array.tolist(),
        "control_mean_ms": float(control_array.mean()),
        "candidate_mean_ms": float(candidate_array.mean()),
        "control_median_ms": float(np.median(control_array)),
        "candidate_median_ms": float(np.median(candidate_array)),
        "control_over_candidate_ratio_of_means": float(
            control_array.mean() / candidate_array.mean()
        ),
        "paired_ratio_mean": float(ratios.mean()),
        "paired_ratio_median": float(np.median(ratios)),
        "bootstrap_mean_ratio_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _promotion_gate(
    *,
    byte_reconstruction_exact: bool,
    scan_checksum_exact: bool,
    expert_output_exact: bool,
    resident_byte_overhead: int,
    primary_ratio_ci: list[float],
) -> bool:
    return (
        bool(byte_reconstruction_exact)
        and bool(scan_checksum_exact)
        and bool(expert_output_exact)
        and int(resident_byte_overhead) == 0
        and len(primary_ratio_ci) == 2
        and all(math.isfinite(float(value)) for value in primary_ratio_ci)
        and float(primary_ratio_ci[0]) > 1.0
    )


def _decode_component(
    payload: bytes, *, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    if dtype == "U32":
        return np.frombuffer(payload, dtype="<u4").copy().reshape(shape)
    if dtype == "BF16":
        return np.frombuffer(payload, dtype="<u2").copy().reshape(shape)
    raise ValueError(f"unsupported Hy3-Q2 component dtype: {dtype}")


def _load_source_bank(
    model: Path,
    *,
    layer: int,
    experts: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(experts) != HY3_Q2_TOP_K or len(set(experts)) != HY3_Q2_TOP_K:
        raise ValueError("issue #68 requires exactly eight distinct experts")
    manifest_path = model / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    if (
        manifest.model_key != "hy3-expert-q2"
        or manifest.quant_bits != 2
        or manifest.quant_group_size != 64
        or manifest.quant_mode != "affine"
        or manifest.sidecar is None
    ):
        raise ValueError("issue #68 requires the authoritative Hy3 affine-Q2 sidecar")
    wanted = tuple(hy3_q2_component_shapes())
    rows: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    records: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    for slot, expert in enumerate(experts):
        record = manifest.record(layer, expert)
        payload = read_expert_record(manifest, model, layer, expert)
        cursor = 0
        found: set[str] = set()
        for segment in record.segments:
            component = payload[cursor : cursor + segment.length]
            cursor += segment.length
            if segment.component in rows:
                if segment.component in found:
                    raise ValueError("duplicate component in Hy3-Q2 expert record")
                rows[segment.component].append(
                    _decode_component(
                        component,
                        dtype=segment.dtype,
                        shape=segment.shape,
                    )
                )
                found.add(segment.component)
        if cursor != len(payload) or found != set(wanted):
            raise ValueError("Hy3-Q2 record does not exactly cover all nine components")
        records.append(
            {
                "slot": slot,
                "expert": expert,
                "generation": 1,
                "record_sha256": record.sha256,
                "logical_bytes": record.logical_bytes,
            }
        )
    load_ms = (time.perf_counter_ns() - started) / 1_000_000
    bank = {name: np.stack(values, axis=0) for name, values in rows.items()}
    return bank, {
        "model": str(model),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
        "sidecar": manifest.sidecar.file,
        "sidecar_sha256": manifest.sidecar.sha256,
        "sidecar_bytes": manifest.sidecar.size,
        "layer": layer,
        "experts": list(experts),
        "records": records,
        "record_load_ms": load_ms,
        "loaded_bytes": sum(record["logical_bytes"] for record in records),
        "component_shapes": {name: list(value.shape) for name, value in bank.items()},
        "component_bytes": {name: int(value.nbytes) for name, value in bank.items()},
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _component_hashes(components: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: _array_sha256(value) for name, value in sorted(components.items())}


def _identities(experts: tuple[int, ...]) -> tuple[Hy3Q2SlotIdentity, ...]:
    return tuple(
        Hy3Q2SlotIdentity(slot=slot, expert=expert, generation=1)
        for slot, expert in enumerate(experts)
    )


def _slots(rows: int) -> mx.array:
    values = np.stack(
        [np.roll(np.arange(HY3_Q2_TOP_K, dtype=np.int32), -row) for row in range(rows)]
    )
    return mx.array(values)


def _tokens(rows: int, *, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((rows, 4096)).astype(mx.bfloat16)


def _evaluate(value: Any) -> None:
    if isinstance(value, tuple):
        mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def _measure_single(
    function: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        _evaluate(function())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        _evaluate(function())
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _measure_pair(
    control: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    for function in (control, candidate):
        for _ in range(warmups):
            _evaluate(function())
    samples = {"control": [], "candidate": []}
    functions = {"control": control, "candidate": candidate}
    for repeat in range(repeats):
        order = (
            ("control", "candidate") if repeat % 2 == 0 else ("candidate", "control")
        )
        for name in order:
            started = time.perf_counter_ns()
            _evaluate(functions[name]())
            samples[name].append((time.perf_counter_ns() - started) / 1_000_000)
    return _paired_summary(
        samples["control"],
        samples["candidate"],
        resamples=resamples,
        seed=seed,
    )


def _reference_components(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value if name.endswith(".weight") else value.view(mx.bfloat16)
        for name, value in raw.items()
    }


def _stock_expert_output(
    token_rows: mx.array,
    slots: mx.array,
    raw_components: dict[str, Any],
) -> mx.array:
    components = _reference_components(raw_components)
    rows = int(token_rows.shape[0])
    assignments = rows * HY3_Q2_TOP_K
    selected_tokens = mx.broadcast_to(
        token_rows[:, None, :],
        (rows, HY3_Q2_TOP_K, 4096),
    ).reshape(assignments, 1, 1, 4096)
    indices = slots.reshape(assignments, 1)

    def qmm(projection: str, selected: mx.array) -> mx.array:
        return mx.gather_qmm(
            selected,
            components[f"{projection}.weight"],
            components[f"{projection}.scales"],
            components[f"{projection}.biases"],
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=2,
            mode="affine",
        )

    gate = qmm("gate_proj", selected_tokens)
    up = qmm("up_proj", selected_tokens)
    activated = swiglu(gate, up).astype(mx.bfloat16)
    down = qmm("down_proj", activated)
    return down.reshape(rows, HY3_Q2_TOP_K, 4096).astype(mx.bfloat16)


def _mlx_sha256(value: mx.array) -> str:
    _evaluate(value)
    words = np.array(value.view(mx.uint16), copy=True).astype("<u2", copy=False)
    return hashlib.sha256(words.tobytes()).hexdigest()


def _git_state(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()

    status = run("status", "--short", "--untracked-files=all")
    return {
        "base_commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _pack_layout(
    source: dict[str, np.ndarray],
    candidate: Hy3Q2LayoutCandidate,
    identities: tuple[Hy3Q2SlotIdentity, ...],
    *,
    repetitions: int = 3,
) -> tuple[Any, dict[str, Any]]:
    samples = []
    packed = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        packed = pack_hy3_q2_bank(source, candidate, identities=identities)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    assert packed is not None
    restored = packed.reconstruct_components()
    source_hashes = _component_hashes(source)
    restored_hashes = _component_hashes(restored)
    replacement_safe = packed.validate_replacement(identities)
    return packed, {
        "construction_samples_ms": samples,
        "construction_mean_ms": statistics.fmean(samples),
        "construction_median_ms": statistics.median(samples),
        "source_component_bytes": packed.source_component_bytes,
        "packed_bytes": packed.packed_bytes,
        "resident_byte_overhead": packed.packed_bytes - packed.source_component_bytes,
        "steady_state_duplicate_bytes": packed.steady_state_duplicate_bytes,
        "source_component_sha256": source_hashes,
        "restored_component_sha256": restored_hashes,
        "byte_reconstruction_exact": source_hashes == restored_hashes,
        "slot_identity_generation_safe": replacement_safe,
        "production_policy": "replace source arrays after validated construction",
    }


def _source_controls(
    source_mlx: dict[str, Any],
    slots: mx.array,
    *,
    warmups: int,
    repeats: int,
) -> tuple[Hy3Q2SourceControl, dict[str, Any], mx.array]:
    results: dict[str, Any] = {}
    outputs: dict[str, mx.array] = {}
    for control in hy3_q2_source_controls():

        def call(control: Hy3Q2SourceControl = control) -> mx.array:
            return scan_hy3_q2_source(source_mlx, slots, control=control)

        output = call()
        _evaluate(output)
        outputs[control.name] = output
        results[control.name] = _measure_single(
            call,
            warmups=warmups,
            repeats=repeats,
        )
    reference = outputs["source_components_v1"]
    for name, output in outputs.items():
        results[name]["checksum_exact_to_v1"] = bool(
            mx.array_equal(output, reference).item()
        )
    winner = min(
        hy3_q2_source_controls(),
        key=lambda item: float(results[item.name]["mean_ms"]),
    )
    return winner, results, outputs[winner.name]


def run_campaign(
    *,
    model: Path,
    layer: int,
    experts: tuple[int, ...],
    rows: tuple[int, ...],
    primary_row: int,
    candidates: tuple[Hy3Q2LayoutCandidate, ...],
    warmups: int,
    screen_repeats: int,
    refine_repeats: int,
    finalists: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if primary_row not in rows:
        raise ValueError("primary row must be included in scaling rows")
    source, source_metadata = _load_source_bank(model, layer=layer, experts=experts)
    identities = _identities(experts)
    source_hashes = _component_hashes(source)
    source_mlx = hy3_q2_components_to_mlx(source)
    _evaluate(tuple(source_mlx.values()))
    primary_slots = _slots(primary_row)
    control, control_results, primary_control_output = _source_controls(
        source_mlx,
        primary_slots,
        warmups=warmups,
        repeats=max(screen_repeats, 3),
    )

    screen: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[Hy3Q2LayoutCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.layout, candidate.block_size), []).append(
            candidate
        )
    for layout_index, layout_candidates in enumerate(grouped.values()):
        representative = layout_candidates[0]
        try:
            packed_base, construction = _pack_layout(
                source,
                representative,
                identities,
            )
        except Exception as exc:
            for candidate in layout_candidates:
                screen.append(
                    {
                        "candidate": candidate.name,
                        "status": "construction_failure",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue
        for candidate_index, candidate in enumerate(layout_candidates):
            packed = replace(packed_base, candidate=candidate)
            result: dict[str, Any] = {
                "candidate": candidate.name,
                "layout": candidate.layout,
                "block_size": candidate.block_size,
                "vector_width": candidate.vector_width,
                "construction": construction,
                "coalescing": hy3_q2_coalescing_evidence(candidate),
            }
            try:
                mlx_packed = packed.to_mlx()
                _evaluate((mlx_packed.gate_up, mlx_packed.down))
                observed = scan_hy3_q2_packed(mlx_packed, primary_slots)
                _evaluate(observed)
                checksum_exact = bool(
                    mx.array_equal(observed, primary_control_output).item()
                )

                def candidate_call(mlx_packed: Any = mlx_packed) -> mx.array:
                    return scan_hy3_q2_packed(mlx_packed, primary_slots)

                def control_call(
                    control: Hy3Q2SourceControl = control,
                ) -> mx.array:
                    return scan_hy3_q2_source(
                        source_mlx, primary_slots, control=control
                    )

                timing = _measure_pair(
                    control_call,
                    candidate_call,
                    warmups=warmups,
                    repeats=screen_repeats,
                    resamples=bootstrap_resamples,
                    seed=680_000 + layout_index * 100 + candidate_index,
                )
                result.update(
                    {
                        "status": "screened",
                        "scan_checksum_exact": checksum_exact,
                        "timing": timing,
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "status": "screen_failure",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            screen.append(result)
            mx.clear_cache()

    eligible = [
        item
        for item in screen
        if item.get("status") == "screened"
        and item.get("scan_checksum_exact") is True
        and item["construction"]["byte_reconstruction_exact"] is True
        and item["construction"]["resident_byte_overhead"] == 0
    ]
    eligible.sort(
        key=lambda item: float(item["timing"]["control_over_candidate_ratio_of_means"]),
        reverse=True,
    )
    finalist_names = [
        item["candidate"] for item in eligible[: min(finalists, len(eligible))]
    ]
    candidate_by_name = {candidate.name: candidate for candidate in candidates}

    reference_tokens = _tokens(primary_row, seed=680_004)
    source_expert_output = _stock_expert_output(
        reference_tokens, primary_slots, source_mlx
    )
    _evaluate(source_expert_output)
    source_output_sha256 = _mlx_sha256(source_expert_output)

    refinement: list[dict[str, Any]] = []
    for finalist_index, name in enumerate(finalist_names):
        candidate = candidate_by_name[name]
        packed, construction = _pack_layout(source, candidate, identities)
        restored = packed.reconstruct_components()
        restored_mlx = hy3_q2_components_to_mlx(restored)
        restored_output = _stock_expert_output(
            reference_tokens, primary_slots, restored_mlx
        )
        _evaluate(restored_output)
        expert_output_exact = bool(
            mx.array_equal(restored_output, source_expert_output).item()
        )
        mlx_packed = packed.to_mlx()
        _evaluate((mlx_packed.gate_up, mlx_packed.down))
        scaling = []
        all_checksums_exact = True
        for row in rows:
            row_slots = _slots(row)
            expected = scan_hy3_q2_source(source_mlx, row_slots, control=control)
            observed = scan_hy3_q2_packed(mlx_packed, row_slots)
            _evaluate((expected, observed))
            checksum_exact = bool(mx.array_equal(observed, expected).item())
            all_checksums_exact = all_checksums_exact and checksum_exact
            timing = _measure_pair(
                lambda row_slots=row_slots: scan_hy3_q2_source(
                    source_mlx, row_slots, control=control
                ),
                lambda row_slots=row_slots: scan_hy3_q2_packed(mlx_packed, row_slots),
                warmups=warmups,
                repeats=refine_repeats,
                resamples=bootstrap_resamples,
                seed=681_000 + finalist_index * 100 + row,
            )
            scaling.append(
                {
                    "row": row,
                    "mtp_k": row - 1,
                    "assignments": row * HY3_Q2_TOP_K,
                    "scan_checksum_exact": checksum_exact,
                    "timing": timing,
                }
            )
        primary = next(item for item in scaling if item["row"] == primary_row)
        primary_ci = primary["timing"]["bootstrap_mean_ratio_95_ci"]
        promotable = _promotion_gate(
            byte_reconstruction_exact=construction["byte_reconstruction_exact"],
            scan_checksum_exact=all_checksums_exact,
            expert_output_exact=expert_output_exact,
            resident_byte_overhead=construction["resident_byte_overhead"],
            primary_ratio_ci=primary_ci,
        )
        refinement.append(
            {
                "candidate": candidate.name,
                "construction": construction,
                "coalescing": hy3_q2_coalescing_evidence(candidate),
                "expert_output_exact": expert_output_exact,
                "source_expert_output_sha256": source_output_sha256,
                "restored_expert_output_sha256": _mlx_sha256(restored_output),
                "all_scan_checksums_exact": all_checksums_exact,
                "scaling": scaling,
                "microbenchmark_promotion_gate": promotable,
            }
        )
        mx.clear_cache()

    refinement.sort(
        key=lambda item: float(
            next(row for row in item["scaling"] if row["row"] == primary_row)["timing"][
                "control_over_candidate_ratio_of_means"
            ]
        ),
        reverse=True,
    )
    winner = refinement[0] if refinement else None
    any_promotable = any(item["microbenchmark_promotion_gate"] for item in refinement)
    repository = Path(__file__).resolve().parents[1]
    return {
        "schema": "mtplx-issue68-hy3-q2-layout-coalescing-v1",
        "issue": 68,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "construction-time byte-neutral expert layout and raw-consumption "
            "coalescing only; no runtime, E1/E2 arithmetic, or decode integration"
        ),
        "qualification_scope": (
            "microbenchmark only; the mandatory 1024/1024 K0-K7 decode matrix "
            "is still required before PR promotion"
        ),
        "model": source_metadata,
        "geometry": {
            "rows": list(rows),
            "primary_row": primary_row,
            "primary_mtp_k": primary_row - 1,
            "top_k": HY3_Q2_TOP_K,
            "component_shapes": {
                name: list(shape) for name, shape in hy3_q2_component_shapes().items()
            },
            "source_component_sha256": source_hashes,
        },
        "measurement": {
            "warmups": warmups,
            "screen_paired_repeats": screen_repeats,
            "refine_paired_repeats": refine_repeats,
            "bootstrap_resamples": bootstrap_resamples,
            "paired_order": "alternating control-candidate / candidate-control",
            "metric": "complete raw affine-Q2 record consumption plus checksum write",
        },
        "source_controls": {
            "selected": control.name,
            "results": control_results,
        },
        "candidate_count": len(candidates),
        "screen": screen,
        "finalist_count": len(refinement),
        "refinement": refinement,
        "decision": {
            "microbenchmark_candidate": None if winner is None else winner["candidate"],
            "positive_primary_interval": any_promotable,
            "status": "advance_to_decode_integration" if any_promotable else "hold",
            "decode_promotion": False,
            "reason": (
                "layout microbenchmark cannot satisfy the issue promotion contract "
                "without isolated 1024/1024 K0-K7 generation evidence"
            ),
        },
        "provenance": {
            "git": _git_state(repository),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "device": mx.device_info(),
            },
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidates = _select_candidates(args)
    if len(args.experts) != HY3_Q2_TOP_K or len(set(args.experts)) != HY3_Q2_TOP_K:
        raise ValueError("--experts must contain exactly eight unique expert IDs")
    if args.primary_row not in args.rows:
        raise ValueError("--primary-row must appear in --rows")
    lock_started_ns = time.time_ns()
    with exclusive_mlx_window(
        plist=args.qwen_plist,
        lock_path=args.lock_path,
        lock_timeout_seconds=args.lock_timeout_seconds,
    ) as receipt:
        holder_pid = os.getpid()
        lock_receipt = hashlib.sha256(
            f"{receipt.lock_path}:{holder_pid}:{lock_started_ns}".encode()
        ).hexdigest()
        result = run_campaign(
            model=args.model.expanduser().resolve(),
            layer=args.layer,
            experts=args.experts,
            rows=args.rows,
            primary_row=args.primary_row,
            candidates=candidates,
            warmups=args.warmups,
            screen_repeats=args.screen_repeats,
            refine_repeats=args.refine_repeats,
            finalists=args.finalists,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        qwen_state = {
            "loaded": receipt.qwen_state.loaded,
            "models": list(receipt.qwen_state.models),
        }
        result["exclusive_window"] = {
            "lock_path": str(receipt.lock_path),
            "holder_pid": holder_pid,
            "receipt": lock_receipt,
            "qwen_before": qwen_state,
            "qwen_after": qwen_state,
            "restored_exactly": False,
            "held_through_restore": False,
        }
    result["exclusive_window"]["restored_exactly"] = True
    result["exclusive_window"]["held_through_restore"] = True
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    sha256 = _sha256_file(output)
    receipt_path = output.with_suffix(output.suffix + ".sha256")
    receipt_path.write_text(f"{sha256}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(output),
                "sha256": sha256,
                "receipt": str(receipt_path),
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
