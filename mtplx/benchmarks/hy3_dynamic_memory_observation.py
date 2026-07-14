"""Fail-closed producer for one issue #46 hardware-arm observation.

The producer owns evidence ordering and validation. Hardware-specific model,
runtime, and allocator operations are supplied by an injected hooks object so
the same transaction drives both the static control and dynamic candidate.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    BenchmarkGateError,
    CONTEXT_MATRIX_TOKENS,
    HY3_Q4_KV_BLOCK_BYTES,
    ISSUE46_RESOURCE_FIELDS,
    ISSUE46_RESOURCE_INTEGER_FIELDS,
    ISSUE46_RESOURCE_SIGNED_INTEGER_FIELDS,
    SCHEMA_OBSERVATION,
    bind_expert_route_evidence,
    canonical_sha256,
    normalize_arm_config,
    validate_campaign_observation,
)


MIN_HOLD_SAMPLES = 3
DEFAULT_HOLD_INTERVAL_SECONDS = 0.5


class ArmObservationError(RuntimeError):
    """Raised when a hook cannot provide complete physical evidence."""


@dataclass(frozen=True)
class ArmRequest:
    """One exact balanced-campaign arm request."""

    arm: Literal["static", "dynamic"]
    context_tokens: int
    repetition: int

    def __post_init__(self) -> None:
        if self.arm not in ("static", "dynamic"):
            raise ArmObservationError("arm must be static or dynamic")
        if self.context_tokens not in CONTEXT_MATRIX_TOKENS:
            raise ArmObservationError(
                f"context_tokens must be in {CONTEXT_MATRIX_TOKENS}"
            )
        _exact_int(self.repetition, field="repetition")


class HardwareArmLane(Protocol):
    """Hardware lane used by :func:`produce_arm_observation`."""

    def identity(self) -> Mapping[str, object]: ...

    def physical_ledger(self) -> Mapping[str, object]: ...

    def reclaim_experts_for_q4(self, context_tokens: int) -> None: ...

    def prepare_q4_context(self, context_tokens: int) -> None: ...

    def kv_growth_steps(self) -> Sequence[Mapping[str, object]]: ...

    def invoke_context(self, context_tokens: int) -> Mapping[str, object]: ...

    def sample_hold_performance(self) -> Mapping[str, object]: ...

    def reset_q4_context(self) -> None: ...

    def trigger_future_expert_demand(self) -> None: ...

    def close(self) -> None: ...


class HardwareArmHooks(Protocol):
    """Injectable loaders for the static and dynamic hardware lanes."""

    def load_static_lane(self, request: ArmRequest) -> HardwareArmLane: ...

    def load_dynamic_lane(self, request: ArmRequest) -> HardwareArmLane: ...


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArmObservationError(
            f"git {' '.join(args)} failed ({completed.returncode}): {detail}"
        )
    return completed.stdout.strip()


def _full_git_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64):
        raise ArmObservationError(f"{field} must be a full hexadecimal Git commit")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise ArmObservationError(f"{field} must be a full hexadecimal Git commit")
    return lowered


def require_clean_source(repo_root: Path | str) -> str:
    """Return HEAD only for an exact, clean Git worktree."""

    root = Path(repo_root).expanduser().resolve()
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ArmObservationError(
            f"repo root {root} differs from Git top level {top_level}"
        )
    commit = _full_git_commit(
        _git_output(root, "rev-parse", "HEAD"),
        field="source Git commit",
    )
    index_entries = _git_output(root, "ls-files", "-v").splitlines()
    assume_unchanged = [entry[2:] for entry in index_entries if entry[:1].islower()]
    skip_worktree = [entry[2:] for entry in index_entries if entry.startswith("S ")]
    if assume_unchanged:
        raise ArmObservationError(
            "source worktree contains assume-unchanged index entries: "
            + ", ".join(assume_unchanged)
        )
    if skip_worktree:
        raise ArmObservationError(
            "source worktree contains skip-worktree index entries: "
            + ", ".join(skip_worktree)
        )
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ArmObservationError(
            "hardware evidence requires a clean source worktree; "
            f"first dirty entry: {status.splitlines()[0]}"
        )
    return commit


def _require_tracked_file(repo_root: Path, path: Path) -> None:
    root = repo_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ArmObservationError(
            f"hook factory source must be inside the source repository: {resolved}"
        ) from exc
    try:
        tracked = _git_output(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            str(relative),
        )
    except ArmObservationError as exc:
        raise ArmObservationError(
            f"hook factory source is not tracked: {relative}"
        ) from exc
    if tracked != str(relative):
        raise ArmObservationError(
            f"hook factory source tracking is ambiguous: {relative}"
        )


def load_tracked_hooks(
    factory_spec: str,
    config: Mapping[str, object],
    *,
    repo_root: Path | str,
) -> HardwareArmHooks:
    """Load ``module:factory`` only when its source is tracked in ``repo_root``."""

    if not isinstance(factory_spec, str) or factory_spec.count(":") != 1:
        raise ArmObservationError("hooks must be specified as module:factory")
    module_name, factory_name = factory_spec.split(":", 1)
    if not module_name or not factory_name:
        raise ArmObservationError("hooks must be specified as module:factory")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise ArmObservationError(
            f"cannot load hook factory {factory_spec}: {exc}"
        ) from exc
    if not callable(factory):
        raise ArmObservationError(f"hook factory {factory_spec} is not callable")
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ArmObservationError(
            f"hook factory {factory_spec} has no inspectable Python source"
        )
    _require_tracked_file(Path(repo_root), Path(source))
    hooks = factory(dict(config))
    for method_name in ("load_static_lane", "load_dynamic_lane"):
        if not callable(getattr(hooks, method_name, None)):
            raise ArmObservationError(
                f"hook factory result is missing callable {method_name}"
            )
    return hooks


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArmObservationError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArmObservationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ArmObservationError(f"{field} must be a finite number")
    if strictly_positive and result <= 0.0:
        raise ArmObservationError(f"{field} must be positive")
    if minimum is not None and result < minimum:
        raise ArmObservationError(f"{field} must be >= {minimum}")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArmObservationError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ArmObservationError(f"{field} must be an array")
    return value


def _required(
    value: Mapping[str, object],
    fields: Sequence[str],
    *,
    context: str,
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ArmObservationError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _build_identity(
    raw_value: object,
    *,
    request: ArmRequest,
    source_git_commit: str,
) -> dict[str, object]:
    raw = _mapping(raw_value, field="lane identity")
    required = (
        "model_key",
        "model_artifact_id",
        "model_artifact_sha256",
        "expert_manifest_id",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "resident_payload_bytes",
        "resident_payload_sha256",
        "source_git_commit",
        "arm_config",
        "kv_quantization",
        "kv_block_size_tokens",
        "total_context_tokens",
    )
    _required(raw, required, context="lane identity")
    if raw["source_git_commit"] != source_git_commit:
        raise ArmObservationError(
            "lane identity source_git_commit differs from the clean producer source"
        )
    arm_config = dict(_mapping(raw["arm_config"], field="identity.arm_config"))
    expected_dynamic = request.arm == "dynamic"
    if arm_config.get("dynamic_memory") is not expected_dynamic:
        raise ArmObservationError(
            "identity.arm_config.dynamic_memory does not match the selected arm"
        )
    normalized = normalize_arm_config(arm_config)
    return {
        "model_key": raw["model_key"],
        "model_artifact_id": raw["model_artifact_id"],
        "model_artifact_sha256": raw["model_artifact_sha256"],
        "expert_manifest_id": raw["expert_manifest_id"],
        "expert_manifest_sha256": raw["expert_manifest_sha256"],
        "artifact_pins_sha256": raw["artifact_pins_sha256"],
        "artifact_stat_sha256": raw["artifact_stat_sha256"],
        "resident_payload_bytes": raw["resident_payload_bytes"],
        "resident_payload_sha256": raw["resident_payload_sha256"],
        "source_git_commit": raw["source_git_commit"],
        "arm_config": arm_config,
        "arm_config_sha256": canonical_sha256(arm_config),
        "normalized_config": normalized,
        "normalized_config_sha256": canonical_sha256(normalized),
        "kv_quantization": raw["kv_quantization"],
        "kv_block_size_tokens": raw["kv_block_size_tokens"],
        "total_context_tokens": raw["total_context_tokens"],
    }


def _timeline_point(
    lane: HardwareArmLane,
    *,
    phase: str,
    monotonic_ns: Callable[[], int],
) -> dict[str, object]:
    raw = _mapping(lane.physical_ledger(), field=f"{phase} physical ledger")
    fields = (
        "allocator_active_bytes",
        "allocator_cache_bytes",
        "allocator_peak_bytes",
        "expert_slab_physical_bytes",
        "kv_physical_bytes",
        "kv_allocated_blocks",
        "slot_health",
        *ISSUE46_RESOURCE_FIELDS,
    )
    _required(raw, fields, context=f"{phase} physical ledger")
    slot_health = dict(_mapping(raw["slot_health"], field=f"{phase}.slot_health"))
    expected_health = {
        "active_routes",
        "pins",
        "loading",
        "failed",
        "integrity_errors",
        "completion_fence_failures",
        "global_device_synchronizations",
    }
    if set(slot_health) != expected_health:
        raise ArmObservationError(
            f"{phase}.slot_health must contain exactly {sorted(expected_health)}"
        )
    for field, value in slot_health.items():
        _exact_int(value, field=f"{phase}.slot_health.{field}")
    numeric = {
        field: _exact_int(raw[field], field=f"{phase}.{field}")
        for field in (
            "allocator_active_bytes",
            "allocator_cache_bytes",
            "allocator_peak_bytes",
            "expert_slab_physical_bytes",
            "kv_physical_bytes",
            "kv_allocated_blocks",
            *ISSUE46_RESOURCE_INTEGER_FIELDS,
        )
    }
    for field in ISSUE46_RESOURCE_SIGNED_INTEGER_FIELDS:
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArmObservationError(f"{phase}.{field} must be an integer")
        numeric[field] = value
    swap_delta = raw["system_swap_delta_bytes"]
    if isinstance(swap_delta, bool) or not isinstance(swap_delta, int):
        raise ArmObservationError(f"{phase}.system_swap_delta_bytes must be an integer")
    representation = raw["kv_representation"]
    if representation != "q4":
        raise ArmObservationError(f"{phase}.kv_representation must be q4")
    failed_closed = raw["failed_closed"]
    if not isinstance(failed_closed, bool):
        raise ArmObservationError(f"{phase}.failed_closed must be a boolean")
    failure_reason = raw["failure_reason"]
    if failure_reason is not None and (
        not isinstance(failure_reason, str) or not failure_reason
    ):
        raise ArmObservationError(
            f"{phase}.failure_reason must be null or a nonempty string"
        )
    if failed_closed and failure_reason is None:
        raise ArmObservationError(
            f"{phase}.failure_reason is required after a fail-closed event"
        )
    if numeric["allocator_peak_bytes"] < numeric["allocator_active_bytes"]:
        raise ArmObservationError(
            f"{phase}.allocator_peak_bytes is below allocator_active_bytes"
        )
    expected_kv_bytes = numeric["kv_allocated_blocks"] * HY3_Q4_KV_BLOCK_BYTES
    if numeric["kv_physical_bytes"] != expected_kv_bytes:
        raise ArmObservationError(
            f"{phase} Q4 block geometry is contradictory: "
            f"{numeric['kv_allocated_blocks']} blocks require "
            f"{expected_kv_bytes} physical bytes"
        )
    classified_bytes = (
        numeric["resident_model_bytes"]
        + numeric["kv_physical_bytes"]
        + numeric["expert_slab_physical_bytes"]
        + numeric["inflight_expert_staging_bytes"]
        + numeric["runtime_workspace_bytes"]
    )
    if numeric["classified_bytes"] != classified_bytes:
        raise ArmObservationError(
            f"{phase}.classified_bytes must equal the five steady pools"
        )
    if numeric["classified_target_bytes"] != (
        numeric["operating_target_bytes"] - numeric["allocator_headroom_bytes"]
    ):
        raise ArmObservationError(
            f"{phase}.classified_target_bytes must preserve allocator headroom"
        )
    expected_charged = classified_bytes + numeric["allocator_cache_charged_bytes"]
    if numeric["charged_bytes"] != expected_charged:
        raise ArmObservationError(
            f"{phase}.charged_bytes must equal the six-pool additive ledger"
        )
    if numeric["charged_residual_bytes"] != (
        numeric["operating_target_bytes"] - numeric["charged_bytes"]
    ):
        raise ArmObservationError(
            f"{phase}.charged_residual_bytes must match charged memory"
        )
    if numeric["allocator_cache_charged_bytes"] < numeric["allocator_cache_bytes"]:
        raise ArmObservationError(
            f"{phase}.allocator_cache_charged_bytes is below raw MLX cache"
        )
    raw_allocator_footprint = (
        numeric["allocator_active_bytes"] + numeric["allocator_cache_bytes"]
    )
    if numeric["charged_bytes"] < raw_allocator_footprint:
        raise ArmObservationError(
            f"{phase}.charged_bytes is below the raw MLX allocator footprint"
        )
    captured_timestamp = raw.get("captured_monotonic_ns")
    observed_at = (
        monotonic_ns()
        if captured_timestamp is None
        else _exact_int(
            captured_timestamp,
            field=f"{phase}.captured_monotonic_ns",
        )
    )
    point = {
        "phase": phase,
        "monotonic_ns": _exact_int(observed_at, field=f"{phase}.monotonic_ns"),
        **numeric,
        "system_swap_delta_bytes": swap_delta,
        "kv_representation": representation,
        "failed_closed": failed_closed,
        "failure_reason": failure_reason,
        "slot_health": slot_health,
        "slot_health_sha256": canonical_sha256(slot_health),
    }
    return point


def _bound_expert_routes(
    *,
    route_trace: object,
    expert_hashes: object,
    expert_manifest_sha256: object,
    field: str,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    try:
        return bind_expert_route_evidence(
            route_trace=route_trace,
            expert_hashes=expert_hashes,
            expert_manifest_sha256=expert_manifest_sha256,
            field=field,
        )
    except BenchmarkGateError as exc:
        raise ArmObservationError(str(exc)) from exc


def _invocation_evidence(
    raw_value: object,
    *,
    expert_manifest_sha256: object,
) -> dict[str, object]:
    raw = _mapping(raw_value, field="lane invocation")
    fields = (
        "prompt_token_ids",
        "generated_token_ids",
        "route_trace",
        "expert_hashes",
        "elapsed_seconds",
    )
    _required(raw, fields, context="lane invocation")
    prompt_tokens = [
        _exact_int(token, field=f"prompt_token_ids[{index}]")
        for index, token in enumerate(
            _sequence(raw["prompt_token_ids"], field="prompt_token_ids")
        )
    ]
    generated_tokens = [
        _exact_int(token, field=f"generated_token_ids[{index}]")
        for index, token in enumerate(
            _sequence(raw["generated_token_ids"], field="generated_token_ids")
        )
    ]
    if not generated_tokens:
        raise ArmObservationError("generated_token_ids must not be empty")
    route_trace, expert_hashes, expert_route_binding = _bound_expert_routes(
        route_trace=raw["route_trace"],
        expert_hashes=raw["expert_hashes"],
        expert_manifest_sha256=expert_manifest_sha256,
        field="lane invocation",
    )
    elapsed = _finite_number(
        raw["elapsed_seconds"],
        field="elapsed_seconds",
        strictly_positive=True,
    )
    return {
        "prompt_token_ids": prompt_tokens,
        "prompt_sha256": canonical_sha256(prompt_tokens),
        "generated_token_ids": generated_tokens,
        "generated_token_sha256": canonical_sha256(generated_tokens),
        "route_trace": route_trace,
        "route_trace_sha256": canonical_sha256(route_trace),
        "expert_hashes": expert_hashes,
        "expert_route_binding": expert_route_binding,
        "elapsed_seconds": elapsed,
    }


def _performance_sample(
    raw_value: object,
    *,
    index: int,
    expert_manifest_sha256: object,
) -> dict[str, object]:
    raw = _mapping(raw_value, field=f"performance sample {index}")
    fields = (
        "tokens_per_second",
        "expert_hit_rate",
        "ssd_bytes_per_token",
        "p50_token_latency_ms",
        "p95_token_latency_ms",
        "generated_token_ids",
        "route_trace",
        "expert_hashes",
    )
    _required(raw, fields, context=f"performance sample {index}")
    generated_tokens = [
        _exact_int(token, field=f"performance sample {index}.generated_token_ids")
        for token in _sequence(
            raw["generated_token_ids"],
            field=f"performance sample {index}.generated_token_ids",
        )
    ]
    if not generated_tokens:
        raise ArmObservationError(
            f"performance sample {index}.generated_token_ids must not be empty"
        )
    route_trace, expert_hashes, expert_route_binding = _bound_expert_routes(
        route_trace=raw["route_trace"],
        expert_hashes=raw["expert_hashes"],
        expert_manifest_sha256=expert_manifest_sha256,
        field=f"performance sample {index}",
    )
    result: dict[str, object] = {
        "tokens_per_second": _finite_number(
            raw["tokens_per_second"],
            field=f"performance sample {index}.tokens_per_second",
            strictly_positive=True,
        ),
        "expert_hit_rate": _finite_number(
            raw["expert_hit_rate"],
            field=f"performance sample {index}.expert_hit_rate",
            minimum=0.0,
        ),
        "ssd_bytes_per_token": _finite_number(
            raw["ssd_bytes_per_token"],
            field=f"performance sample {index}.ssd_bytes_per_token",
            minimum=0.0,
        ),
        "p50_token_latency_ms": _finite_number(
            raw["p50_token_latency_ms"],
            field=f"performance sample {index}.p50_token_latency_ms",
            strictly_positive=True,
        ),
        "p95_token_latency_ms": _finite_number(
            raw["p95_token_latency_ms"],
            field=f"performance sample {index}.p95_token_latency_ms",
            strictly_positive=True,
        ),
        "generated_token_ids": generated_tokens,
        "generated_token_sha256": canonical_sha256(generated_tokens),
        "route_trace": route_trace,
        "route_trace_sha256": canonical_sha256(route_trace),
        "expert_hashes": expert_hashes,
        "expert_route_binding": expert_route_binding,
    }
    if float(result["expert_hit_rate"]) > 1.0:
        raise ArmObservationError(
            f"performance sample {index}.expert_hit_rate must be <= 1"
        )
    if float(result["p95_token_latency_ms"]) < float(result["p50_token_latency_ms"]):
        raise ArmObservationError(
            f"performance sample {index} p95 latency is below p50"
        )
    return result


def produce_arm_observation(
    *,
    hooks: HardwareArmHooks,
    request: ArmRequest,
    source_git_commit: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    hold_samples: int = MIN_HOLD_SAMPLES,
    hold_interval_seconds: float = DEFAULT_HOLD_INTERVAL_SECONDS,
) -> dict[str, object]:
    """Run one hardware arm and return a validated schema-v1 observation."""

    count = _exact_int(hold_samples, field="hold_samples", minimum=MIN_HOLD_SAMPLES)
    configured_hold_samples = getattr(hooks, "hold_samples", count)
    if (
        _exact_int(
            configured_hold_samples,
            field="hooks hold sample count",
            minimum=MIN_HOLD_SAMPLES,
        )
        != count
    ):
        raise ArmObservationError(
            "requested hold sample count differs from the hardware prompt reserve"
        )
    interval = _finite_number(
        hold_interval_seconds,
        field="hold_interval_seconds",
        minimum=0.0,
    )
    loader = (
        hooks.load_dynamic_lane if request.arm == "dynamic" else hooks.load_static_lane
    )
    lane = loader(request)
    timeline: list[dict[str, object]] = []
    performance_samples: list[dict[str, object]] = []
    try:
        identity = _build_identity(
            lane.identity(),
            request=request,
            source_git_commit=source_git_commit,
        )
        timeline.append(
            _timeline_point(lane, phase="pre_growth", monotonic_ns=monotonic_ns)
        )
        if request.arm == "dynamic":
            lane.reclaim_experts_for_q4(request.context_tokens)
            timeline.append(
                _timeline_point(
                    lane,
                    phase="post_expert_reclaim",
                    monotonic_ns=monotonic_ns,
                )
            )
        lane.prepare_q4_context(request.context_tokens)
        kv_growth_steps = [dict(step) for step in lane.kv_growth_steps()]
        timeline.append(
            _timeline_point(
                lane,
                phase="post_kv_growth",
                monotonic_ns=monotonic_ns,
            )
        )
        invocation = _invocation_evidence(
            lane.invoke_context(request.context_tokens),
            expert_manifest_sha256=identity["expert_manifest_sha256"],
        )
        for index in range(count):
            performance_samples.append(
                _performance_sample(
                    lane.sample_hold_performance(),
                    index=index,
                    expert_manifest_sha256=identity["expert_manifest_sha256"],
                )
            )
            timeline.append(
                _timeline_point(lane, phase="hold", monotonic_ns=monotonic_ns)
            )
            if index + 1 < count:
                sleep(interval)
        lane.reset_q4_context()
        timeline.append(
            _timeline_point(lane, phase="post_reset", monotonic_ns=monotonic_ns)
        )
        if request.arm == "dynamic":
            lane.trigger_future_expert_demand()
            timeline.append(
                _timeline_point(lane, phase="post_regrow", monotonic_ns=monotonic_ns)
            )

        generated_tokens = invocation["generated_token_ids"]
        assert isinstance(generated_tokens, list)
        elapsed = float(invocation["elapsed_seconds"])
        observation: dict[str, object] = {
            "schema": SCHEMA_OBSERVATION,
            "arm": request.arm,
            "context_tokens": request.context_tokens,
            "repetition": request.repetition,
            "cache_start_state": {
                "kind": (
                    "static-128k-reserved-q4"
                    if request.arm == "static"
                    else "dynamic-declared-q4"
                ),
                "kv_physical_bytes": timeline[0]["kv_physical_bytes"],
                "kv_blocks": timeline[0]["kv_allocated_blocks"],
            },
            "identity": identity,
            "prompt_token_ids": invocation["prompt_token_ids"],
            "prompt_sha256": invocation["prompt_sha256"],
            "generated_token_ids": generated_tokens,
            "generated_token_sha256": invocation["generated_token_sha256"],
            "route_trace": invocation["route_trace"],
            "route_trace_sha256": invocation["route_trace_sha256"],
            "expert_hashes": invocation["expert_hashes"],
            "expert_route_binding": invocation["expert_route_binding"],
            "kv_growth_steps": kv_growth_steps,
            "timeline": timeline,
            "lifecycle": {
                "reset_observed": True,
                "future_demand_invoked": request.arm == "dynamic",
                "post_regrow_observed": request.arm == "dynamic",
            },
            "metrics": {
                "generated_tokens": len(generated_tokens),
                "elapsed_seconds": elapsed,
                "tokens_per_second": len(generated_tokens) / elapsed,
                "peak_charged_bytes": max(
                    int(point["charged_bytes"]) for point in timeline
                ),
                "stress_peak_charged_bytes": max(
                    max(
                        int(point["allocator_peak_bytes"])
                        + int(point["allocator_cache_bytes"]),
                        int(point["charged_bytes"]),
                    )
                    for point in timeline
                ),
                "hold_performance_samples": [
                    sample["tokens_per_second"] for sample in performance_samples
                ],
                "performance_samples": performance_samples,
            },
        }
        validate_campaign_observation(observation)
        return observation
    finally:
        lane.close()


def _strict_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArmObservationError(f"duplicate hooks-config key {key!r}")
        result[key] = value
    return result


def _load_hooks_config(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArmObservationError(f"cannot load hooks config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArmObservationError("hooks config must be one JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce one fail-closed issue #46 Hy3 hardware-arm observation"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--hooks", required=True, metavar="MODULE:FACTORY")
    parser.add_argument("--hooks-config", type=Path, required=True)
    parser.add_argument("--arm", choices=("static", "dynamic"), required=True)
    parser.add_argument(
        "--context-tokens",
        type=int,
        choices=CONTEXT_MATRIX_TOKENS,
        required=True,
    )
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--hold-samples", type=int, default=MIN_HOLD_SAMPLES)
    parser.add_argument(
        "--hold-interval-seconds",
        type=float,
        default=DEFAULT_HOLD_INTERVAL_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.repo_root.expanduser().resolve()
        config_path = args.hooks_config.expanduser().resolve()
        source_commit = require_clean_source(root)
        _require_tracked_file(root, config_path)
        config = _load_hooks_config(config_path)
        request = ArmRequest(
            arm=args.arm,
            context_tokens=args.context_tokens,
            repetition=args.repetition,
        )
        # Hook code may use ordinary print calls for load diagnostics. Preserve
        # the one-JSON-value stdout protocol required by run_json_subprocess.
        with contextlib.redirect_stdout(sys.stderr):
            hooks = load_tracked_hooks(args.hooks, config, repo_root=root)
            observation = produce_arm_observation(
                hooks=hooks,
                request=request,
                source_git_commit=source_commit,
                hold_samples=args.hold_samples,
                hold_interval_seconds=args.hold_interval_seconds,
            )
        rendered = json.dumps(observation, indent=2, sort_keys=True) + "\n"
    except Exception as exc:
        print(f"Hy3 hardware arm observation rejected: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(rendered)
    return 0


__all__ = [
    "ArmObservationError",
    "ArmRequest",
    "HardwareArmHooks",
    "HardwareArmLane",
    "build_parser",
    "load_tracked_hooks",
    "main",
    "produce_arm_observation",
    "require_clean_source",
]
