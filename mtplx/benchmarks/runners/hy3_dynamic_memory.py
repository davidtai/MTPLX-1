"""Evidence gates for the Hy3 Q4 dynamic unified-memory experiment.

The module deliberately has no MLX or server imports.  Hardware integration is
supplied through callbacks (or JSON-emitting subprocesses), which keeps the
ordering, identity, restoration, and statistical gates independently testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from mtplx.memory_broker import (
    HY3_Q4_ALLOCATOR_HEADROOM_BYTES,
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BLOCK_TOKENS,
    HY3_Q4_KV_BYTES_PER_TOKEN,
    HY3_Q4_KV_LAYERS,
    hy3_q4_kv_physical_geometry,
)
from mtplx.runtime_options import HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV

CONTEXT_MATRIX_TOKENS = (4_096, 32_768, 65_536, 131_072)
HY3_Q4_TOTAL_CONTEXT_TOKENS = 131_072
HY3_Q4_KV_BLOCK_SIZE_TOKENS = HY3_Q4_KV_BLOCK_TOKENS
HY3_Q4_MAX_BLOCKS = hy3_q4_kv_physical_geometry(
    HY3_Q4_TOTAL_CONTEXT_TOKENS
).physical_blocks
MIN_STABLE_HOLD_SAMPLES = 3
MIN_STABLE_HOLD_DURATION_NS = 1_000_000_000
OPERATING_TARGET_BYTES = 110 * 1024**3
HARD_CEILING_BYTES = 112 * 1024**3
MAX_PROCESS_COMPRESSED_GROWTH_BYTES = 512 * 1024**2
MAX_128K_PERFORMANCE_REGRESSION = 0.05
DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS = 12 * 60 * 60
DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS = 30
SCHEMA_OBSERVATION = "mtplx-hy3-dynamic-memory-observation-v1"
SCHEMA_PROBE = "mtplx-hy3-allocator-release-probe-v1"
SCHEMA_CAMPAIGN = "mtplx-hy3-dynamic-memory-campaign-v1"
SCHEMA_ARTIFACT_ATTESTATION = "mtplx-hy3-artifact-attestation-v1"
SCHEMA_EXPERT_ROUTE_BINDING = "mtplx-hy3-expert-route-binding-v1"
EXPERT_ROUTE_PRODUCER_SCOPE = "route-map-from-loaded-manifest"
EXPERT_ROUTE_OFFLINE_SCOPE = "structural-binding-only"

_PROBE_BOUND_IDENTITY_FIELDS = (
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
    "kv_quantization",
    "kv_block_size_tokens",
    "total_context_tokens",
)

ISSUE46_RESOURCE_INTEGER_FIELDS = (
    "operating_target_bytes",
    "hard_ceiling_bytes",
    "allocator_headroom_bytes",
    "classified_target_bytes",
    "classified_bytes",
    "charged_bytes",
    "resident_model_bytes",
    "kv_logical_tokens",
    "expert_logical_records",
    "expert_active_records",
    "expert_resident_records",
    "expert_logical_slabs",
    "expert_active_slabs",
    "expert_draining_slabs",
    "expert_released_slabs",
    "pinned_expert_bytes",
    "inflight_expert_bytes",
    "speculative_expert_bytes",
    "runtime_workspace_bytes",
    "inflight_expert_staging_bytes",
    "requested_reclaim_bytes",
    "reclaimed_bytes",
    "regrown_bytes",
    "evicted_expert_records",
    "evicted_expert_slabs",
    "resize_duration_ns",
    "total_resize_duration_ns",
    "max_resize_duration_ns",
    "blocked_by_pin_bytes",
    "admission_failures",
    "resize_failures",
    "allocator_cache_charged_bytes",
    "process_rss_bytes",
    "process_compressed_bytes",
)
ISSUE46_RESOURCE_SIGNED_INTEGER_FIELDS = ("charged_residual_bytes",)
ISSUE46_RESOURCE_FIELDS = (
    *ISSUE46_RESOURCE_INTEGER_FIELDS,
    *ISSUE46_RESOURCE_SIGNED_INTEGER_FIELDS,
    "system_swap_delta_bytes",
    "kv_representation",
    "failed_closed",
    "failure_reason",
)

_T = TypeVar("_T")


class BenchmarkGateError(RuntimeError):
    """Raised when evidence is incomplete, ambiguous, or physically invalid."""


class _SubprocessCleanupNotProven(BenchmarkGateError):
    """Raised when an isolated subprocess group may still own the GPU lane."""


_SUBPROCESS_CLEANUP_UNPROVEN: ContextVar[bool] = ContextVar(
    "mtplx_issue46_subprocess_cleanup_unproven",
    default=False,
)


def _subprocess_cleanup_not_proven(message: str) -> _SubprocessCleanupNotProven:
    _SUBPROCESS_CLEANUP_UNPROVEN.set(True)
    return _SubprocessCleanupNotProven(message)


def canonical_sha256(value: object) -> str:
    """Return the lowercase SHA-256 of canonical JSON evidence."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_arm_config(value: Mapping[str, object]) -> dict[str, object]:
    """Remove only the declared static/dynamic intervention fields."""

    normalized = dict(value)
    normalized.pop("dynamic_memory", None)
    normalized.pop("planned_persistent_slots", None)
    raw_streaming = normalized.get("expert_streaming_config")
    if isinstance(raw_streaming, Mapping):
        streaming = dict(raw_streaming)
        streaming.pop("dynamic_expert_slabs", None)
        normalized["expert_streaming_config"] = streaming
    return normalized


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkGateError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkGateError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise BenchmarkGateError(f"{field} must be a {qualifier}finite number")
    return result


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkGateError(f"{field} must be a nonempty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise BenchmarkGateError(
            f"{field} must be exactly 64 lowercase hexadecimal digits"
        )
    return text


def _git_commit(value: object, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    if len(text) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise BenchmarkGateError(f"{field} must be a full hexadecimal Git commit")
    return text


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkGateError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BenchmarkGateError(f"{field} must be an array")
    return value


def _require_fields(
    value: Mapping[str, object],
    fields: Sequence[str],
    *,
    context: str,
) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise BenchmarkGateError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _normalize_route_trace(
    value: object,
    *,
    field: str,
) -> tuple[list[dict[str, object]], set[str]]:
    raw_routes = _sequence(value, field=field)
    if not raw_routes:
        raise BenchmarkGateError(f"{field} must not be empty")
    normalized: list[dict[str, object]] = []
    routed_pairs: set[str] = set()
    routed_fields = {
        "phase",
        "layer",
        "expert_ids",
        "trace_epoch",
        "token_count",
        "decode_step",
    }
    reset_fields = {"phase", "previous_trace_epoch", "trace_epoch"}
    for index, raw_route in enumerate(raw_routes):
        prefix = f"{field}[{index}]"
        route = _mapping(raw_route, field=prefix)
        has_layer = "layer" in route
        has_experts = "expert_ids" in route
        if has_layer or has_experts:
            if not has_layer or not has_experts:
                raise BenchmarkGateError(f"{prefix} is an incomplete routed entry")
            unknown = set(route) - routed_fields
            if unknown:
                raise BenchmarkGateError(
                    f"{prefix} has unvalidated field(s): {', '.join(sorted(unknown))}"
                )
            phase = _nonempty_string(route.get("phase"), field=f"{prefix}.phase")
            layer = _exact_int(route["layer"], field=f"{prefix}.layer")
            raw_experts = _sequence(route["expert_ids"], field=f"{prefix}.expert_ids")
            if not raw_experts:
                raise BenchmarkGateError(f"{prefix}.expert_ids must not be empty")
            experts = [
                _exact_int(expert, field=f"{prefix}.expert_ids[{expert_index}]")
                for expert_index, expert in enumerate(raw_experts)
            ]
            result: dict[str, object] = {
                "phase": phase,
                "layer": layer,
                "expert_ids": experts,
            }
            for optional in ("trace_epoch", "token_count", "decode_step"):
                if optional in route:
                    result[optional] = _exact_int(
                        route[optional], field=f"{prefix}.{optional}"
                    )
            normalized.append(result)
            routed_pairs.update(f"{layer}:{expert}" for expert in experts)
            continue
        if set(route) != reset_fields or route.get("phase") != "reset":
            raise BenchmarkGateError(
                f"{prefix} must be a routed expert object or exact reset object"
            )
        normalized.append(
            {
                "phase": "reset",
                "previous_trace_epoch": _exact_int(
                    route["previous_trace_epoch"],
                    field=f"{prefix}.previous_trace_epoch",
                ),
                "trace_epoch": _exact_int(
                    route["trace_epoch"], field=f"{prefix}.trace_epoch"
                ),
            }
        )
    if not routed_pairs:
        raise BenchmarkGateError(f"{field} has no routed expert pairs")
    return normalized, routed_pairs


def _normalize_expert_hashes(
    value: object,
    *,
    field: str,
    routed_pairs: set[str],
) -> dict[str, str]:
    raw_hashes = _mapping(value, field=field)
    if not raw_hashes:
        raise BenchmarkGateError(f"{field} must not be empty")
    normalized: dict[str, str] = {}
    for raw_key, raw_hash in raw_hashes.items():
        key = _nonempty_string(raw_key, field=f"{field} key")
        parts = key.split(":")
        if (
            len(parts) != 2
            or not all(part.isdecimal() for part in parts)
            or f"{int(parts[0])}:{int(parts[1])}" != key
        ):
            raise BenchmarkGateError(f"{field} key {key!r} must be layer:expert")
        digest = _sha256(raw_hash, field=f"{field}[{key!r}]")
        if digest == "0" * 64:
            raise BenchmarkGateError(f"{field}[{key!r}] has a zero SHA-256")
        normalized[key] = digest
    if set(normalized) != routed_pairs:
        raise BenchmarkGateError(
            f"{field} expert hashes do not exactly cover routed expert pairs"
        )
    return normalized


def bind_expert_route_evidence(
    *,
    route_trace: object,
    expert_hashes: object,
    expert_manifest_sha256: object,
    field: str,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    """Validate live route evidence and bind it to the loaded expert manifest."""

    routes, routed_pairs = _normalize_route_trace(
        route_trace, field=f"{field}.route_trace"
    )
    hashes = _normalize_expert_hashes(
        expert_hashes,
        field=f"{field}.expert_hashes",
        routed_pairs=routed_pairs,
    )
    manifest_hash = _sha256(
        expert_manifest_sha256,
        field=f"{field}.expert_manifest_sha256",
    )
    if manifest_hash == "0" * 64:
        raise BenchmarkGateError(f"{field}.expert_manifest_sha256 is zero")
    binding: dict[str, object] = {
        "schema": SCHEMA_EXPERT_ROUTE_BINDING,
        "producer_verification_scope": EXPERT_ROUTE_PRODUCER_SCOPE,
        "offline_verification_scope": EXPERT_ROUTE_OFFLINE_SCOPE,
        "expert_manifest_sha256": manifest_hash,
        "route_trace_sha256": canonical_sha256(routes),
        "expert_hashes_sha256": canonical_sha256(hashes),
    }
    return routes, hashes, binding


def _validate_expert_route_binding(
    *,
    route_trace: object,
    expert_hashes: object,
    binding: object,
    expert_manifest_sha256: object,
    field: str,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    routes, hashes, expected = bind_expert_route_evidence(
        route_trace=route_trace,
        expert_hashes=expert_hashes,
        expert_manifest_sha256=expert_manifest_sha256,
        field=field,
    )
    raw_binding = _mapping(binding, field=f"{field}.expert_route_binding")
    if set(raw_binding) != set(expected):
        raise BenchmarkGateError(
            f"{field}.expert_route_binding must contain the exact binding schema"
        )
    for name, expected_value in expected.items():
        if raw_binding[name] != expected_value:
            raise BenchmarkGateError(
                f"{field}.expert_route_binding.{name} differs from bound evidence"
            )
    return routes, hashes, expected


@dataclass(frozen=True)
class AllocatorSample:
    """One active/cache/peak allocator observation."""

    active_bytes: int
    cache_bytes: int
    peak_bytes: int

    def __post_init__(self) -> None:
        _exact_int(self.active_bytes, field="allocator.active_bytes")
        _exact_int(self.cache_bytes, field="allocator.cache_bytes")
        _exact_int(self.peak_bytes, field="allocator.peak_bytes")
        if self.peak_bytes < self.active_bytes:
            raise BenchmarkGateError(
                "allocator.peak_bytes cannot be below allocator.active_bytes"
            )

    @property
    def charged_bytes(self) -> int:
        return self.active_bytes + self.cache_bytes

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, field: str
    ) -> AllocatorSample:
        _require_fields(
            value,
            ("active_bytes", "cache_bytes", "peak_bytes"),
            context=field,
        )
        return cls(
            active_bytes=_exact_int(
                value["active_bytes"], field=f"{field}.active_bytes"
            ),
            cache_bytes=_exact_int(value["cache_bytes"], field=f"{field}.cache_bytes"),
            peak_bytes=_exact_int(value["peak_bytes"], field=f"{field}.peak_bytes"),
        )


@dataclass(frozen=True)
class ProbeSlab:
    """A real, independently releasable component slab used by the probe."""

    slab_id: str
    registered_physical_bytes: int

    def __post_init__(self) -> None:
        _nonempty_string(self.slab_id, field="probe slab id")
        _exact_int(
            self.registered_physical_bytes,
            field=f"probe slab {self.slab_id}.registered_physical_bytes",
            minimum=1,
        )


@dataclass(frozen=True)
class CacheStartState:
    """Declared physical KV state immediately before one arm observation."""

    kind: str
    kv_physical_bytes: int
    kv_blocks: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CacheStartState:
        _require_fields(
            value,
            ("kind", "kv_physical_bytes", "kv_blocks"),
            context="cache_start_state",
        )
        state = cls(
            kind=_nonempty_string(value["kind"], field="cache_start_state.kind"),
            kv_physical_bytes=_exact_int(
                value["kv_physical_bytes"],
                field="cache_start_state.kv_physical_bytes",
            ),
            kv_blocks=_exact_int(
                value["kv_blocks"], field="cache_start_state.kv_blocks"
            ),
        )
        if state.kv_physical_bytes != state.kv_blocks * HY3_Q4_KV_BLOCK_BYTES:
            raise BenchmarkGateError(
                "cache_start_state.kv_physical_bytes does not match exact Q4 geometry"
            )
        return state


@dataclass(frozen=True)
class MemoryTimelinePoint:
    phase: str
    monotonic_ns: int
    allocator_active_bytes: int
    allocator_cache_bytes: int
    allocator_peak_bytes: int
    expert_slab_physical_bytes: int
    kv_physical_bytes: int
    kv_allocated_blocks: int
    slot_health: Mapping[str, int]
    slot_health_sha256: str
    resource_evidence: Mapping[str, object]

    @property
    def charged_allocator_bytes(self) -> int:
        return self.allocator_active_bytes + self.allocator_cache_bytes

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        index: int,
    ) -> MemoryTimelinePoint:
        prefix = f"timeline[{index}]"
        _require_fields(
            value,
            (
                "phase",
                "monotonic_ns",
                "allocator_active_bytes",
                "allocator_cache_bytes",
                "allocator_peak_bytes",
                "expert_slab_physical_bytes",
                "kv_physical_bytes",
                "kv_allocated_blocks",
                "slot_health",
                "slot_health_sha256",
                *ISSUE46_RESOURCE_FIELDS,
            ),
            context=prefix,
        )
        raw_health = _mapping(value["slot_health"], field=f"{prefix}.slot_health")
        required_health = (
            "active_routes",
            "pins",
            "loading",
            "failed",
            "integrity_errors",
            "completion_fence_failures",
            "global_device_synchronizations",
        )
        _require_fields(raw_health, required_health, context=f"{prefix}.slot_health")
        health = {
            field: _exact_int(raw_health[field], field=f"{prefix}.slot_health.{field}")
            for field in required_health
        }
        health_hash = _sha256(
            value["slot_health_sha256"], field=f"{prefix}.slot_health_sha256"
        )
        if canonical_sha256(health) != health_hash:
            raise BenchmarkGateError(
                f"{prefix}.slot_health_sha256 does not match exact slot health"
            )
        resource_evidence: dict[str, object] = {
            field: _exact_int(value[field], field=f"{prefix}.{field}")
            for field in ISSUE46_RESOURCE_INTEGER_FIELDS
        }
        for field in ISSUE46_RESOURCE_SIGNED_INTEGER_FIELDS:
            signed = value[field]
            if isinstance(signed, bool) or not isinstance(signed, int):
                raise BenchmarkGateError(f"{prefix}.{field} must be an integer")
            resource_evidence[field] = signed
        swap_delta = value["system_swap_delta_bytes"]
        if isinstance(swap_delta, bool) or not isinstance(swap_delta, int):
            raise BenchmarkGateError(
                f"{prefix}.system_swap_delta_bytes must be an integer"
            )
        resource_evidence["system_swap_delta_bytes"] = swap_delta
        representation = _nonempty_string(
            value["kv_representation"], field=f"{prefix}.kv_representation"
        )
        if representation != "q4":
            raise BenchmarkGateError(f"{prefix}.kv_representation must be q4")
        resource_evidence["kv_representation"] = representation
        failed_closed = value["failed_closed"]
        if not isinstance(failed_closed, bool):
            raise BenchmarkGateError(f"{prefix}.failed_closed must be a boolean")
        resource_evidence["failed_closed"] = failed_closed
        failure_reason = value["failure_reason"]
        if failure_reason is not None:
            failure_reason = _nonempty_string(
                failure_reason, field=f"{prefix}.failure_reason"
            )
        resource_evidence["failure_reason"] = failure_reason
        if resource_evidence["operating_target_bytes"] != OPERATING_TARGET_BYTES:
            raise BenchmarkGateError(
                f"{prefix}.operating_target_bytes must be exactly 110 GiB"
            )
        if resource_evidence["hard_ceiling_bytes"] != HARD_CEILING_BYTES:
            raise BenchmarkGateError(
                f"{prefix}.hard_ceiling_bytes must be exactly 112 GiB"
            )
        if (
            resource_evidence["allocator_headroom_bytes"]
            != HY3_Q4_ALLOCATOR_HEADROOM_BYTES
        ):
            raise BenchmarkGateError(
                f"{prefix}.allocator_headroom_bytes must be exactly 1 GiB"
            )
        if failed_closed and failure_reason is None:
            raise BenchmarkGateError(
                f"{prefix}.failure_reason is required after a fail-closed event"
            )
        point = cls(
            phase=_nonempty_string(value["phase"], field=f"{prefix}.phase"),
            monotonic_ns=_exact_int(
                value["monotonic_ns"], field=f"{prefix}.monotonic_ns"
            ),
            allocator_active_bytes=_exact_int(
                value["allocator_active_bytes"],
                field=f"{prefix}.allocator_active_bytes",
            ),
            allocator_cache_bytes=_exact_int(
                value["allocator_cache_bytes"],
                field=f"{prefix}.allocator_cache_bytes",
            ),
            allocator_peak_bytes=_exact_int(
                value["allocator_peak_bytes"],
                field=f"{prefix}.allocator_peak_bytes",
            ),
            expert_slab_physical_bytes=_exact_int(
                value["expert_slab_physical_bytes"],
                field=f"{prefix}.expert_slab_physical_bytes",
            ),
            kv_physical_bytes=_exact_int(
                value["kv_physical_bytes"], field=f"{prefix}.kv_physical_bytes"
            ),
            kv_allocated_blocks=_exact_int(
                value["kv_allocated_blocks"],
                field=f"{prefix}.kv_allocated_blocks",
            ),
            slot_health=health,
            slot_health_sha256=health_hash,
            resource_evidence=resource_evidence,
        )
        if point.allocator_peak_bytes < point.allocator_active_bytes:
            raise BenchmarkGateError(
                f"{prefix}.allocator_peak_bytes is below active bytes"
            )
        expected_kv_bytes = point.kv_allocated_blocks * HY3_Q4_KV_BLOCK_BYTES
        if point.kv_physical_bytes != expected_kv_bytes:
            raise BenchmarkGateError(
                f"{prefix}.kv_physical_bytes does not match exact Q4 geometry"
            )
        classified_bytes = (
            int(resource_evidence["resident_model_bytes"])
            + point.kv_physical_bytes
            + point.expert_slab_physical_bytes
            + int(resource_evidence["inflight_expert_staging_bytes"])
            + int(resource_evidence["runtime_workspace_bytes"])
        )
        if int(resource_evidence["classified_bytes"]) != classified_bytes:
            raise BenchmarkGateError(
                f"{prefix}.classified_bytes does not match the five steady pools"
            )
        if int(resource_evidence["classified_target_bytes"]) != (
            int(resource_evidence["operating_target_bytes"])
            - int(resource_evidence["allocator_headroom_bytes"])
        ):
            raise BenchmarkGateError(
                f"{prefix}.classified_target_bytes does not preserve allocator headroom"
            )
        if classified_bytes > int(resource_evidence["classified_target_bytes"]):
            raise BenchmarkGateError(
                f"{prefix}.classified_bytes exceeds the classified target"
            )
        expected_charged = classified_bytes + int(
            resource_evidence["allocator_cache_charged_bytes"]
        )
        if int(resource_evidence["charged_bytes"]) != expected_charged:
            raise BenchmarkGateError(
                f"{prefix}.charged_bytes does not match the six-pool additive ledger"
            )
        if int(resource_evidence["charged_residual_bytes"]) != (
            int(resource_evidence["operating_target_bytes"])
            - int(resource_evidence["charged_bytes"])
        ):
            raise BenchmarkGateError(
                f"{prefix}.charged_residual_bytes does not match charged memory"
            )
        if (
            int(resource_evidence["allocator_cache_charged_bytes"])
            < point.allocator_cache_bytes
        ):
            raise BenchmarkGateError(
                f"{prefix}.allocator_cache_charged_bytes is below raw MLX cache"
            )
        if int(resource_evidence["charged_bytes"]) < point.charged_allocator_bytes:
            raise BenchmarkGateError(
                f"{prefix}.charged_bytes is below the raw MLX allocator footprint"
            )
        return point


@dataclass(frozen=True)
class CampaignObservation:
    arm: str
    context_tokens: int
    repetition: int
    cache_start_state: CacheStartState
    identity: Mapping[str, object]
    prompt_sha256: str
    generated_token_ids: tuple[int, ...]
    generated_token_sha256: str
    route_trace: object
    route_trace_sha256: str
    expert_hashes: Mapping[str, str]
    kv_growth_steps: tuple[Mapping[str, object], ...]
    timeline: tuple[MemoryTimelinePoint, ...]
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class CampaignScheduleEntry:
    context_tokens: int
    repetition: int
    arm: str
    order_index: int

    @property
    def arms(self) -> tuple[str, str]:
        return (
            ("static", "dynamic") if self.repetition % 2 == 0 else ("dynamic", "static")
        )


@dataclass(frozen=True)
class QwenIsolationHooks:
    """Injectable exclusive-lane and exact-Qwen-state operations."""

    acquire_lane: Callable[[], None]
    release_lane: Callable[[], None]
    capture: Callable[[], object]
    unload: Callable[[object], None]
    restore: Callable[[object], None]
    verify_restored: Callable[[object], bool]


def balanced_campaign_schedule(
    *,
    repetitions: int,
    contexts: Sequence[int] = CONTEXT_MATRIX_TOKENS,
) -> tuple[CampaignScheduleEntry, ...]:
    """Build an AB/BA schedule with equal first-position counts per context."""

    count = _exact_int(repetitions, field="repetitions", minimum=1)
    if count % 2:
        raise ValueError("repetitions must be a positive even integer")
    normalized_contexts = tuple(
        _exact_int(value, field="context_tokens", minimum=1) for value in contexts
    )
    if normalized_contexts != CONTEXT_MATRIX_TOKENS:
        raise ValueError(
            f"contexts must be the issue #46 matrix {CONTEXT_MATRIX_TOKENS}"
        )
    entries: list[CampaignScheduleEntry] = []
    for context_tokens in normalized_contexts:
        for repetition in range(count):
            arms = (
                ("static", "dynamic") if repetition % 2 == 0 else ("dynamic", "static")
            )
            entries.extend(
                CampaignScheduleEntry(
                    context_tokens=context_tokens,
                    repetition=repetition,
                    arm=arm,
                    order_index=order_index,
                )
                for order_index, arm in enumerate(arms)
            )
    return tuple(entries)


def _validate_identity(
    value: Mapping[str, object], *, context: str
) -> dict[str, object]:
    fields = (
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
        "arm_config_sha256",
        "normalized_config",
        "normalized_config_sha256",
        "kv_quantization",
        "kv_block_size_tokens",
        "total_context_tokens",
    )
    _require_fields(value, fields, context=context)
    result: dict[str, object] = {
        "model_key": _nonempty_string(value["model_key"], field=f"{context}.model_key"),
        "model_artifact_id": _nonempty_string(
            value["model_artifact_id"], field=f"{context}.model_artifact_id"
        ),
        "model_artifact_sha256": _sha256(
            value["model_artifact_sha256"],
            field=f"{context}.model_artifact_sha256",
        ),
        "expert_manifest_id": _nonempty_string(
            value["expert_manifest_id"], field=f"{context}.expert_manifest_id"
        ),
        "expert_manifest_sha256": _sha256(
            value["expert_manifest_sha256"],
            field=f"{context}.expert_manifest_sha256",
        ),
        "artifact_pins_sha256": _sha256(
            value["artifact_pins_sha256"],
            field=f"{context}.artifact_pins_sha256",
        ),
        "artifact_stat_sha256": _sha256(
            value["artifact_stat_sha256"],
            field=f"{context}.artifact_stat_sha256",
        ),
        "resident_payload_bytes": _exact_int(
            value["resident_payload_bytes"],
            field=f"{context}.resident_payload_bytes",
            minimum=1,
        ),
        "resident_payload_sha256": _sha256(
            value["resident_payload_sha256"],
            field=f"{context}.resident_payload_sha256",
        ),
        "source_git_commit": _git_commit(
            value["source_git_commit"], field=f"{context}.source_git_commit"
        ),
        "arm_config": dict(
            _mapping(value["arm_config"], field=f"{context}.arm_config")
        ),
        "arm_config_sha256": _sha256(
            value["arm_config_sha256"], field=f"{context}.arm_config_sha256"
        ),
        "normalized_config": dict(
            _mapping(value["normalized_config"], field=f"{context}.normalized_config")
        ),
        "normalized_config_sha256": _sha256(
            value["normalized_config_sha256"],
            field=f"{context}.normalized_config_sha256",
        ),
        "kv_quantization": _nonempty_string(
            value["kv_quantization"], field=f"{context}.kv_quantization"
        ),
        "kv_block_size_tokens": _exact_int(
            value["kv_block_size_tokens"],
            field=f"{context}.kv_block_size_tokens",
            minimum=1,
        ),
        "total_context_tokens": _exact_int(
            value["total_context_tokens"],
            field=f"{context}.total_context_tokens",
            minimum=1,
        ),
    }
    if result["model_key"] != "hy3-q4":
        raise BenchmarkGateError(f"{context}.model_key must be hy3-q4")
    if result["kv_quantization"] != "q4":
        raise BenchmarkGateError(f"{context}.kv_quantization must be q4")
    if result["kv_block_size_tokens"] != HY3_Q4_KV_BLOCK_SIZE_TOKENS:
        raise BenchmarkGateError(f"{context} has the wrong Q4 KV block size")
    if result["total_context_tokens"] != HY3_Q4_TOTAL_CONTEXT_TOKENS:
        raise BenchmarkGateError(f"{context} has the wrong total context contract")
    if canonical_sha256(result["arm_config"]) != result["arm_config_sha256"]:
        raise BenchmarkGateError(
            f"{context}.arm_config_sha256 does not match exact resolved config"
        )
    if (
        canonical_sha256(result["normalized_config"])
        != result["normalized_config_sha256"]
    ):
        raise BenchmarkGateError(
            f"{context}.normalized_config_sha256 does not match normalized config"
        )
    expected_normalized = normalize_arm_config(result["arm_config"])
    if expected_normalized != result["normalized_config"]:
        raise BenchmarkGateError(
            f"{context}.normalized_config does not match arm intervention normalization"
        )
    arm_config = _mapping(result["arm_config"], field=f"{context}.arm_config")
    attention_runtime_env = _mapping(
        arm_config.get("attention_runtime_env"),
        field=f"{context}.arm_config.attention_runtime_env",
    )
    if dict(attention_runtime_env) != HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV:
        raise BenchmarkGateError(
            f"{context}.arm_config.attention_runtime_env must pin the exact "
            "full-history Q4 attention route"
        )
    streaming = _mapping(
        arm_config.get("expert_streaming_config"),
        field=f"{context}.arm_config.expert_streaming_config",
    )
    if (
        _exact_int(
            streaming.get("allocator_headroom_bytes"),
            field=f"{context}.arm_config.expert_streaming_config.allocator_headroom_bytes",
        )
        != HY3_Q4_ALLOCATOR_HEADROOM_BYTES
    ):
        raise BenchmarkGateError(
            f"{context} must pin exactly 1 GiB of allocator headroom"
        )
    if (
        _exact_int(
            streaming.get("kv_bytes_per_token_override"),
            field=(
                f"{context}.arm_config.expert_streaming_config."
                "kv_bytes_per_token_override"
            ),
        )
        != HY3_Q4_KV_BYTES_PER_TOKEN
    ):
        raise BenchmarkGateError(f"{context} must pin exact Q4 KV physical geometry")
    expected_streaming = {
        "memory_limit_bytes": OPERATING_TARGET_BYTES,
        "max_live_kv_tokens": HY3_Q4_TOTAL_CONTEXT_TOKENS,
        "runtime_reserve_bytes": 8 * 1024**3,
        "transient_slots": 32,
        "cache_scope": "global",
        "slot_layout": "component-banks",
        "expert_slab_slots": 32,
    }
    for field, expected in expected_streaming.items():
        if streaming.get(field) != expected:
            raise BenchmarkGateError(
                f"{context}.arm_config.expert_streaming_config.{field} "
                f"must be {expected!r}"
            )
    dynamic_memory = arm_config.get("dynamic_memory")
    if not isinstance(dynamic_memory, bool):
        raise BenchmarkGateError(f"{context}.arm_config.dynamic_memory must be boolean")
    if streaming.get("dynamic_expert_slabs") is not dynamic_memory:
        raise BenchmarkGateError(
            f"{context}.arm_config dynamic expert mode contradicts the selected arm"
        )
    expected_slots = 9_696 if dynamic_memory else 8_673
    if arm_config.get("planned_persistent_slots") != expected_slots:
        raise BenchmarkGateError(
            f"{context}.arm_config.planned_persistent_slots must be {expected_slots}"
        )
    return result


def _phase_once(
    timeline: Sequence[MemoryTimelinePoint],
    phase: str,
) -> MemoryTimelinePoint:
    matches = [point for point in timeline if point.phase == phase]
    if len(matches) != 1:
        raise BenchmarkGateError(f"timeline must contain exactly one {phase} sample")
    return matches[0]


def _assert_final_slot_health(point: MemoryTimelinePoint) -> None:
    nonzero = tuple(
        field for field, value in point.slot_health.items() if int(value) != 0
    )
    if nonzero:
        raise BenchmarkGateError(
            f"{point.phase} slot health is not quiescent and failure-free: "
            + ", ".join(nonzero)
        )


def _validate_timeline(
    arm: str,
    context_tokens: int,
    cache_start_state: CacheStartState,
    timeline: Sequence[MemoryTimelinePoint],
) -> None:
    if not timeline:
        raise BenchmarkGateError("timeline must not be empty")
    if any(
        later.monotonic_ns <= earlier.monotonic_ns
        for earlier, later in zip(timeline, timeline[1:], strict=False)
    ):
        raise BenchmarkGateError("timeline monotonic_ns values must strictly increase")

    pre = _phase_once(timeline, "pre_growth")
    growth = _phase_once(timeline, "post_kv_growth")
    warmup = _phase_once(timeline, "hold_warmup")
    reset = _phase_once(timeline, "post_reset")
    if cache_start_state.kv_physical_bytes != (
        cache_start_state.kv_blocks * HY3_Q4_KV_BLOCK_BYTES
    ):
        raise BenchmarkGateError("cache_start_state does not match exact Q4 geometry")
    holds = [point for point in timeline if point.phase == "hold"]
    if len(holds) < MIN_STABLE_HOLD_SAMPLES:
        raise BenchmarkGateError(
            f"timeline needs at least {MIN_STABLE_HOLD_SAMPLES} stable hold samples"
        )
    if not (
        pre.monotonic_ns
        < growth.monotonic_ns
        < warmup.monotonic_ns
        < holds[0].monotonic_ns
        <= holds[-1].monotonic_ns
        < reset.monotonic_ns
    ):
        raise BenchmarkGateError(
            "timeline must order pre-growth, growth, hold_warmup, every hold, "
            "then reset"
        )
    fixed_pool_fields = (
        "resident_model_bytes",
        "inflight_expert_staging_bytes",
        "runtime_workspace_bytes",
    )
    fixed_pool_baseline = tuple(
        timeline[0].resource_evidence[field] for field in fixed_pool_fields
    )
    if any(
        tuple(point.resource_evidence[field] for field in fixed_pool_fields)
        != fixed_pool_baseline
        for point in timeline
    ):
        raise BenchmarkGateError("fixed physical memory pools changed during the arm")
    for point in timeline:
        logical_tokens = int(point.resource_evidence["kv_logical_tokens"])
        physical_capacity = point.kv_allocated_blocks * HY3_Q4_KV_BLOCK_SIZE_TOKENS
        if logical_tokens > physical_capacity:
            raise BenchmarkGateError(
                f"{point.phase} logical KV tokens exceed physical Q4 capacity"
            )
    if any(
        int(point.resource_evidence["kv_logical_tokens"]) < context_tokens
        for point in (growth, warmup, *holds)
    ):
        raise BenchmarkGateError(
            "post-growth and hold logical KV tokens do not cover the requested context"
        )
    record_evictions = [
        int(point.resource_evidence["evicted_expert_records"])
        for point in (warmup, *holds)
    ]
    if any(
        later < earlier
        for earlier, later in zip(record_evictions, record_evictions[1:], strict=False)
    ):
        raise BenchmarkGateError(
            "evicted_expert_records decreased during the hold window: "
            f"{record_evictions}"
        )
    stable_fields = (
        "allocator_active_bytes",
        "allocator_cache_bytes",
        "expert_slab_physical_bytes",
        "kv_physical_bytes",
        "kv_allocated_blocks",
        "slot_health_sha256",
    )
    stable_drift = {
        field: [getattr(point, field) for point in holds]
        for field in stable_fields
        if any(getattr(point, field) != getattr(holds[0], field) for point in holds[1:])
    }
    if stable_drift:
        raise BenchmarkGateError(
            "stable hold samples changed physical memory or slot health: "
            f"{json.dumps(stable_drift, sort_keys=True)}"
        )
    stable_resource_fields = (
        "allocator_headroom_bytes",
        "classified_target_bytes",
        "classified_bytes",
        "charged_bytes",
        "charged_residual_bytes",
        "resident_model_bytes",
        "inflight_expert_staging_bytes",
        "runtime_workspace_bytes",
        "allocator_cache_charged_bytes",
        "requested_reclaim_bytes",
        "reclaimed_bytes",
        "regrown_bytes",
        # Ordinary record-level LRU churn is expected while physical slabs,
        # charged bytes, and resize telemetry remain stable.
        "evicted_expert_slabs",
        "resize_duration_ns",
        "total_resize_duration_ns",
        "max_resize_duration_ns",
        "blocked_by_pin_bytes",
    )
    resource_drift = {
        field: [point.resource_evidence[field] for point in holds]
        for field in stable_resource_fields
        if any(
            point.resource_evidence[field] != holds[0].resource_evidence[field]
            for point in holds[1:]
        )
    }
    if resource_drift:
        raise BenchmarkGateError(
            "stable hold samples changed the classified or charged memory ledger: "
            f"{json.dumps(resource_drift, sort_keys=True)}"
        )
    if holds[-1].monotonic_ns - holds[0].monotonic_ns < MIN_STABLE_HOLD_DURATION_NS:
        raise BenchmarkGateError("stable hold duration is shorter than one second")
    for hold in holds:
        _assert_final_slot_health(hold)

    if cache_start_state.kv_physical_bytes != pre.kv_physical_bytes:
        raise BenchmarkGateError(
            "cache_start_state does not match pre_growth physical KV bytes"
        )
    if cache_start_state.kv_blocks != pre.kv_allocated_blocks:
        raise BenchmarkGateError(
            "cache_start_state does not match pre_growth allocated KV blocks"
        )
    if reset.kv_physical_bytes >= holds[-1].kv_physical_bytes:
        raise BenchmarkGateError("post_reset did not physically release KV bytes")
    if reset.kv_allocated_blocks >= holds[-1].kv_allocated_blocks:
        raise BenchmarkGateError("post_reset did not release allocated KV blocks")
    kv_release = holds[-1].kv_physical_bytes - reset.kv_physical_bytes
    if holds[-1].charged_allocator_bytes - reset.charged_allocator_bytes < kv_release:
        raise BenchmarkGateError(
            "post_reset KV release did not reduce charged allocator bytes"
        )
    _assert_final_slot_health(reset)

    if arm == "static":
        if cache_start_state.kind != "static-128k-reserved-q4":
            raise BenchmarkGateError(
                "static control must declare static-128k-reserved-q4 cache start"
            )
        if cache_start_state.kv_blocks != HY3_Q4_MAX_BLOCKS:
            raise BenchmarkGateError(
                "static control must reserve all 8192 Q4 KV blocks"
            )
        if growth.kv_physical_bytes != pre.kv_physical_bytes:
            raise BenchmarkGateError(
                "static 128K-reserved control changed physical KV capacity"
            )
        if growth.kv_allocated_blocks != HY3_Q4_MAX_BLOCKS:
            raise BenchmarkGateError(
                "static 128K-reserved control lost its declared KV blocks"
            )
        return

    if cache_start_state.kind != "dynamic-declared-q4":
        raise BenchmarkGateError(
            "dynamic arm must declare dynamic-declared-q4 cache start"
        )
    if cache_start_state.kv_blocks != 1:
        raise BenchmarkGateError(
            "dynamic arm must start from exactly one physical Q4 block"
        )
    reclaim = _phase_once(timeline, "post_expert_reclaim")
    regrow = _phase_once(timeline, "post_regrow")
    for point in (pre, reclaim):
        if int(point.resource_evidence["kv_logical_tokens"]) != 1:
            raise BenchmarkGateError(
                f"{point.phase} logical KV ownership must remain exactly 1 "
                "before physical growth completes"
            )
    if int(growth.resource_evidence["kv_logical_tokens"]) != context_tokens:
        raise BenchmarkGateError(
            "post_kv_growth logical KV ownership must equal the requested context"
        )
    if not (pre.monotonic_ns < reclaim.monotonic_ns < growth.monotonic_ns):
        raise BenchmarkGateError(
            "physical expert decrease must be observed before KV increase"
        )
    if reclaim.expert_slab_physical_bytes >= pre.expert_slab_physical_bytes:
        raise BenchmarkGateError(
            "physical expert decrease must be observed before KV increase"
        )
    if reclaim.kv_physical_bytes != pre.kv_physical_bytes:
        raise BenchmarkGateError(
            "physical expert decrease must be observed before KV increase"
        )
    if reclaim.kv_allocated_blocks != pre.kv_allocated_blocks:
        raise BenchmarkGateError(
            "physical expert decrease must be observed before KV block allocation"
        )
    expert_release = pre.expert_slab_physical_bytes - reclaim.expert_slab_physical_bytes
    if pre.charged_allocator_bytes - reclaim.charged_allocator_bytes < expert_release:
        raise BenchmarkGateError(
            "expert registration fell without the same charged allocator release"
        )
    if growth.kv_physical_bytes <= reclaim.kv_physical_bytes:
        raise BenchmarkGateError("post_kv_growth did not physically grow Q4 KV")
    if growth.kv_allocated_blocks <= reclaim.kv_allocated_blocks:
        raise BenchmarkGateError("post_kv_growth did not allocate Q4 KV blocks")
    if growth.expert_slab_physical_bytes > reclaim.expert_slab_physical_bytes:
        raise BenchmarkGateError("expert slabs regrew during the protected KV growth")
    required_final_blocks = hy3_q4_kv_physical_geometry(context_tokens).physical_blocks
    if growth.kv_allocated_blocks != required_final_blocks:
        raise BenchmarkGateError(
            "post_kv_growth does not exactly cover the requested context"
        )
    if growth.kv_allocated_blocks - pre.kv_allocated_blocks < 2:
        raise BenchmarkGateError(
            "dynamic growth did not cross multiple KV block boundaries"
        )
    if not (reset.monotonic_ns < regrow.monotonic_ns):
        raise BenchmarkGateError("post_regrow must follow post_reset")
    if regrow.kv_physical_bytes != reset.kv_physical_bytes:
        raise BenchmarkGateError("lazy expert regrow changed reset KV bytes")
    if regrow.kv_allocated_blocks != reset.kv_allocated_blocks:
        raise BenchmarkGateError("lazy expert regrow changed reset KV blocks")
    if regrow.expert_slab_physical_bytes <= reset.expert_slab_physical_bytes:
        raise BenchmarkGateError(
            "post_regrow did not physically restore expert capacity"
        )
    _assert_final_slot_health(regrow)


def _validate_kv_growth_steps(
    value: object,
    *,
    arm: str,
    context_tokens: int,
    cache_start_state: CacheStartState,
    timeline: Sequence[MemoryTimelinePoint],
) -> tuple[Mapping[str, object], ...]:
    raw_steps = _sequence(value, field="observation.kv_growth_steps")
    if arm == "static":
        if raw_steps:
            raise BenchmarkGateError("static control must not report KV growth steps")
        return ()
    if len(raw_steps) != 2:
        raise BenchmarkGateError(
            "dynamic arm must retain exactly two physical KV growth steps"
        )
    final_blocks = hy3_q4_kv_physical_geometry(context_tokens).physical_blocks
    expected_tokens = ((final_blocks - 1) * HY3_Q4_KV_BLOCK_SIZE_TOKENS, context_tokens)
    expected_blocks = (final_blocks - 1, final_blocks)
    normalized: list[Mapping[str, object]] = []
    prior_after: MemoryTimelinePoint | None = None
    prior_after_ns: int | None = None
    step_ledgers: list[
        tuple[MemoryTimelinePoint, MemoryTimelinePoint, MemoryTimelinePoint]
    ] = []
    step_timestamps: list[tuple[int, int, int]] = []

    def ledger(
        raw: object,
        *,
        field: str,
        monotonic_ns: int,
        captured: bool,
    ) -> MemoryTimelinePoint:
        item = _mapping(raw, field=field)
        _require_fields(
            item,
            (
                "allocator_active_bytes",
                "allocator_cache_bytes",
                "allocator_peak_bytes",
                "expert_slab_physical_bytes",
                "kv_physical_bytes",
                "kv_allocated_blocks",
                "slot_health",
                *ISSUE46_RESOURCE_FIELDS,
            ),
            context=field,
        )
        if captured:
            captured_ns = _exact_int(
                item.get("captured_monotonic_ns"),
                field=f"{field}.captured_monotonic_ns",
            )
            if captured_ns != monotonic_ns:
                raise BenchmarkGateError(
                    f"{field}.captured_monotonic_ns differs from growth timestamp"
                )
        candidate = dict(item)
        candidate.update(
            phase=field,
            monotonic_ns=monotonic_ns,
            slot_health_sha256=canonical_sha256(item["slot_health"]),
        )
        try:
            point = MemoryTimelinePoint.from_mapping(candidate, index=0)
        except BenchmarkGateError as exc:
            raise BenchmarkGateError(f"{field}: {exc}") from exc
        classified = int(point.resource_evidence["classified_bytes"])
        charged = int(point.resource_evidence["charged_bytes"])
        hard_ceiling = int(point.resource_evidence["hard_ceiling_bytes"])
        transient = max(
            point.allocator_peak_bytes + point.allocator_cache_bytes,
            charged,
        )
        if classified >= hard_ceiling:
            raise BenchmarkGateError(f"{field} classified memory reached 112 GiB")
        if charged >= hard_ceiling:
            raise BenchmarkGateError(f"{field} charged memory reached 112 GiB")
        if transient >= hard_ceiling:
            raise BenchmarkGateError(f"{field} transient memory reached 112 GiB")
        return point

    for index, raw_step in enumerate(raw_steps):
        field = f"observation.kv_growth_steps[{index}]"
        step = _mapping(raw_step, field=field)
        if (
            _exact_int(step.get("sequence_index"), field=f"{field}.sequence_index")
            != index
        ):
            raise BenchmarkGateError("KV growth step sequence is not exact")
        if (
            _exact_int(
                step.get("requested_tokens"),
                field=f"{field}.requested_tokens",
                minimum=1,
            )
            != expected_tokens[index]
        ):
            raise BenchmarkGateError("KV growth step token checkpoint differs")
        if (
            _exact_int(
                step.get("target_blocks"),
                field=f"{field}.target_blocks",
                minimum=1,
            )
            != expected_blocks[index]
        ):
            raise BenchmarkGateError("KV growth step block checkpoint differs")
        before_ns = _exact_int(
            step.get("before_monotonic_ns"), field=f"{field}.before_monotonic_ns"
        )
        reclaim_ns = _exact_int(
            step.get("reclaim_monotonic_ns"), field=f"{field}.reclaim_monotonic_ns"
        )
        after_ns = _exact_int(
            step.get("after_monotonic_ns"), field=f"{field}.after_monotonic_ns"
        )
        if not before_ns < reclaim_ns < after_ns:
            raise BenchmarkGateError(
                "KV growth step ordering is not strictly monotonic"
            )
        if prior_after_ns is not None and before_ns <= prior_after_ns:
            raise BenchmarkGateError("KV growth steps overlap or run out of order")
        before = ledger(
            step.get("before"),
            field=f"{field}.before",
            monotonic_ns=before_ns,
            captured=False,
        )
        reclaim = ledger(
            step.get("reclaim_gap"),
            field=f"{field}.reclaim_gap",
            monotonic_ns=reclaim_ns,
            captured=True,
        )
        after = ledger(
            step.get("after"),
            field=f"{field}.after",
            monotonic_ns=after_ns,
            captured=False,
        )
        for ledger_name, point in (
            ("before", before),
            ("reclaim_gap", reclaim),
            ("after", after),
        ):
            if int(point.resource_evidence["kv_logical_tokens"]) != 1:
                raise BenchmarkGateError(
                    f"{field}.{ledger_name} logical KV ownership must remain "
                    "exactly 1 until physical growth completes"
                )
        if index == 0 and (
            before.kv_allocated_blocks != cache_start_state.kv_blocks
            or before.kv_physical_bytes != cache_start_state.kv_physical_bytes
        ):
            raise BenchmarkGateError("first KV growth step differs from cache start")
        if prior_after is not None and any(
            getattr(before, name) != getattr(prior_after, name)
            for name in (
                "expert_slab_physical_bytes",
                "kv_physical_bytes",
                "kv_allocated_blocks",
            )
        ):
            raise BenchmarkGateError("KV growth steps do not form one physical chain")
        if (
            reclaim.kv_physical_bytes != before.kv_physical_bytes
            or reclaim.kv_allocated_blocks != before.kv_allocated_blocks
        ):
            raise BenchmarkGateError("KV rose before the reclaim gap was captured")
        if reclaim.expert_slab_physical_bytes > before.expert_slab_physical_bytes:
            raise BenchmarkGateError("expert slabs regrew inside KV growth")
        if after.kv_allocated_blocks != expected_blocks[index]:
            raise BenchmarkGateError("KV growth step missed its exact block target")
        if after.kv_physical_bytes <= before.kv_physical_bytes:
            raise BenchmarkGateError("KV growth step did not increase physical bytes")
        if after.expert_slab_physical_bytes > reclaim.expert_slab_physical_bytes:
            raise BenchmarkGateError("expert slabs regrew before KV growth committed")
        reclaimed = (
            before.expert_slab_physical_bytes - reclaim.expert_slab_physical_bytes
        )
        kv_growth = after.kv_physical_bytes - before.kv_physical_bytes
        if (
            _exact_int(
                step.get("reclaimed_expert_bytes"),
                field=f"{field}.reclaimed_expert_bytes",
            )
            != reclaimed
        ):
            raise BenchmarkGateError("KV growth step reclaimed-byte evidence differs")
        if index == 0 and reclaimed <= 0:
            raise BenchmarkGateError(
                "first KV growth step did not reclaim expert memory"
            )
        if (
            _exact_int(
                step.get("kv_growth_bytes"), field=f"{field}.kv_growth_bytes", minimum=1
            )
            != kv_growth
        ):
            raise BenchmarkGateError("KV growth step physical-byte evidence differs")
        if (
            _exact_int(
                step.get("steady_delta_bytes"),
                field=f"{field}.steady_delta_bytes",
                minimum=1,
            )
            != kv_growth
        ):
            raise BenchmarkGateError(
                "KV growth reservation differs from physical growth"
            )
        transient_delta = _exact_int(
            step.get("max_transient_delta_bytes"),
            field=f"{field}.max_transient_delta_bytes",
            minimum=1,
        )
        if HY3_Q4_KV_BLOCK_BYTES % HY3_Q4_KV_LAYERS:
            raise BenchmarkGateError("Q4 transient geometry is not layer-exact")
        expected_transient_delta = expected_blocks[index] * (
            HY3_Q4_KV_BLOCK_BYTES // HY3_Q4_KV_LAYERS
        )
        if transient_delta != expected_transient_delta:
            raise BenchmarkGateError(
                "KV growth transient does not match exact per-entry Q4 geometry"
            )
        if (
            int(reclaim.resource_evidence["charged_bytes"]) + transient_delta
            >= HARD_CEILING_BYTES
        ):
            raise BenchmarkGateError(
                "KV growth charged transient reached the 112 GiB hard ceiling"
            )
        normalized.append(dict(step))
        step_ledgers.append((before, reclaim, after))
        step_timestamps.append((before_ns, reclaim_ns, after_ns))
        prior_after = after
        prior_after_ns = after_ns

    pre = _phase_once(timeline, "pre_growth")
    timeline_reclaim = _phase_once(timeline, "post_expert_reclaim")
    timeline_growth = _phase_once(timeline, "post_kv_growth")
    first_before, first_reclaim, _first_after = step_ledgers[0]
    _last_before, _last_reclaim, final_after = step_ledgers[-1]
    first_before_ns, first_reclaim_ns, _first_after_ns = step_timestamps[0]
    _last_before_ns, _last_reclaim_ns, final_after_ns = step_timestamps[-1]
    if not pre.monotonic_ns < first_before_ns:
        raise BenchmarkGateError(
            "timeline pre_growth does not precede first growth ledger"
        )
    if timeline_reclaim.monotonic_ns != first_reclaim_ns:
        raise BenchmarkGateError(
            "timeline reclaim timestamp differs from captured growth ledger"
        )
    if not final_after_ns < timeline_growth.monotonic_ns:
        raise BenchmarkGateError(
            "final growth ledger does not precede timeline post_kv_growth"
        )
    physical_fields = (
        "expert_slab_physical_bytes",
        "kv_physical_bytes",
        "kv_allocated_blocks",
    )
    if any(
        getattr(first_before, name) != getattr(pre, name) for name in physical_fields
    ):
        raise BenchmarkGateError(
            "first growth ledger differs from timeline pre_growth physical state"
        )
    reclaim_fields = (
        "allocator_active_bytes",
        "allocator_cache_bytes",
        "allocator_peak_bytes",
        *physical_fields,
    )
    if (
        any(
            getattr(first_reclaim, name) != getattr(timeline_reclaim, name)
            for name in reclaim_fields
        )
        or first_reclaim.resource_evidence != timeline_reclaim.resource_evidence
    ):
        raise BenchmarkGateError(
            "captured growth reclaim ledger differs from timeline reclaim state"
        )
    if any(
        getattr(final_after, name) != getattr(timeline_growth, name)
        for name in physical_fields
    ):
        raise BenchmarkGateError(
            "final growth ledger differs from timeline post_kv_growth physical state"
        )
    return tuple(normalized)


def validate_campaign_observation(value: Mapping[str, object]) -> CampaignObservation:
    """Validate one arm result as evidence rather than trusting runner claims."""

    _require_fields(
        value,
        (
            "schema",
            "arm",
            "context_tokens",
            "repetition",
            "cache_start_state",
            "identity",
            "prompt_sha256",
            "generated_token_ids",
            "generated_token_sha256",
            "route_trace",
            "route_trace_sha256",
            "expert_hashes",
            "expert_route_binding",
            "kv_growth_steps",
            "timeline",
            "metrics",
        ),
        context="observation",
    )
    if value["schema"] != SCHEMA_OBSERVATION:
        raise BenchmarkGateError(f"observation.schema must be {SCHEMA_OBSERVATION}")
    arm = _nonempty_string(value["arm"], field="observation.arm")
    if arm not in ("static", "dynamic"):
        raise BenchmarkGateError("observation.arm must be static or dynamic")
    context_tokens = _exact_int(
        value["context_tokens"], field="observation.context_tokens", minimum=1
    )
    if context_tokens not in CONTEXT_MATRIX_TOKENS:
        raise BenchmarkGateError(
            f"observation.context_tokens must be in {CONTEXT_MATRIX_TOKENS}"
        )
    repetition = _exact_int(value["repetition"], field="observation.repetition")
    cache_start_state = CacheStartState.from_mapping(
        _mapping(value["cache_start_state"], field="observation.cache_start_state")
    )
    identity = _validate_identity(
        _mapping(value["identity"], field="observation.identity"),
        context="observation.identity",
    )
    dynamic_flag = identity["arm_config"].get("dynamic_memory")
    if not isinstance(dynamic_flag, bool) or dynamic_flag is not (arm == "dynamic"):
        raise BenchmarkGateError(
            "observation.identity.arm_config dynamic_memory does not match arm"
        )
    prompt_hash = _sha256(value["prompt_sha256"], field="observation.prompt_sha256")

    raw_tokens = _sequence(
        value["generated_token_ids"], field="observation.generated_token_ids"
    )
    tokens = tuple(
        _exact_int(token, field=f"generated_token_ids[{index}]")
        for index, token in enumerate(raw_tokens)
    )
    token_hash = _sha256(
        value["generated_token_sha256"],
        field="observation.generated_token_sha256",
    )
    if canonical_sha256(list(tokens)) != token_hash:
        raise BenchmarkGateError(
            "observation.generated_token_sha256 does not match exact generated tokens"
        )

    route_hash = _sha256(
        value["route_trace_sha256"], field="observation.route_trace_sha256"
    )
    if canonical_sha256(value["route_trace"]) != route_hash:
        raise BenchmarkGateError(
            "observation.route_trace_sha256 does not match exact routes"
        )
    route_trace, expert_hashes, _expert_route_binding = _validate_expert_route_binding(
        route_trace=value["route_trace"],
        expert_hashes=value["expert_hashes"],
        binding=value["expert_route_binding"],
        expert_manifest_sha256=identity["expert_manifest_sha256"],
        field="observation",
    )
    if canonical_sha256(route_trace) != route_hash:
        raise BenchmarkGateError(
            "observation.route_trace_sha256 does not match exact routes"
        )

    raw_timeline = _sequence(value["timeline"], field="observation.timeline")
    timeline = tuple(
        MemoryTimelinePoint.from_mapping(
            _mapping(item, field=f"observation.timeline[{index}]"),
            index=index,
        )
        for index, item in enumerate(raw_timeline)
    )
    _validate_timeline(arm, context_tokens, cache_start_state, timeline)
    kv_growth_steps = _validate_kv_growth_steps(
        value["kv_growth_steps"],
        arm=arm,
        context_tokens=context_tokens,
        cache_start_state=cache_start_state,
        timeline=timeline,
    )

    raw_metrics = _mapping(value["metrics"], field="observation.metrics")
    _require_fields(
        raw_metrics,
        (
            "generated_tokens",
            "elapsed_seconds",
            "tokens_per_second",
            "peak_charged_bytes",
            "stress_peak_charged_bytes",
            "hold_performance_samples",
            "hold_warmup_sample",
            "performance_samples",
        ),
        context="observation.metrics",
    )
    generated_tokens = _exact_int(
        raw_metrics["generated_tokens"],
        field="observation.metrics.generated_tokens",
    )
    if generated_tokens != len(tokens):
        raise BenchmarkGateError(
            "observation.metrics.generated_tokens differs from exact token IDs"
        )
    elapsed = _finite_number(
        raw_metrics["elapsed_seconds"],
        field="observation.metrics.elapsed_seconds",
        positive=True,
    )
    tokens_per_second = _finite_number(
        raw_metrics["tokens_per_second"],
        field="observation.metrics.tokens_per_second",
        positive=True,
    )
    if not math.isclose(
        tokens_per_second,
        generated_tokens / elapsed,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        raise BenchmarkGateError(
            "observation tokens_per_second does not match tokens / elapsed_seconds"
        )
    peak_charged = _exact_int(
        raw_metrics["peak_charged_bytes"],
        field="observation.metrics.peak_charged_bytes",
    )
    observed_peak = max(
        int(point.resource_evidence["charged_bytes"]) for point in timeline
    )
    if peak_charged != observed_peak:
        raise BenchmarkGateError(
            "observation.metrics.peak_charged_bytes must equal timeline evidence"
        )
    stress_peak_charged = _exact_int(
        raw_metrics["stress_peak_charged_bytes"],
        field="observation.metrics.stress_peak_charged_bytes",
    )
    observed_stress_peak = max(
        max(
            point.allocator_peak_bytes + point.allocator_cache_bytes,
            int(point.resource_evidence["charged_bytes"]),
        )
        for point in timeline
    )
    if stress_peak_charged != observed_stress_peak:
        raise BenchmarkGateError(
            "observation.metrics.stress_peak_charged_bytes must equal timeline evidence"
        )
    raw_hold_performance = _sequence(
        raw_metrics["hold_performance_samples"],
        field="observation.metrics.hold_performance_samples",
    )
    hold_performance = tuple(
        _finite_number(
            sample,
            field=f"observation.metrics.hold_performance_samples[{index}]",
            positive=True,
        )
        for index, sample in enumerate(raw_hold_performance)
    )
    if len(hold_performance) < MIN_STABLE_HOLD_SAMPLES:
        raise BenchmarkGateError(
            "observation.metrics.hold_performance_samples needs at least three samples"
        )
    median_performance = statistics.median(hold_performance)
    if (max(hold_performance) - min(hold_performance)) / median_performance > 0.10:
        raise BenchmarkGateError(
            "observation.metrics.hold_performance_samples are not stable within 10%: "
            f"{json.dumps(list(hold_performance))}"
        )
    raw_performance_samples = _sequence(
        raw_metrics["performance_samples"],
        field="observation.metrics.performance_samples",
    )
    if len(raw_performance_samples) != len(hold_performance):
        raise BenchmarkGateError(
            "observation.metrics.performance_samples must match hold_performance_samples"
        )
    raw_warmup_sample = _mapping(
        raw_metrics["hold_warmup_sample"],
        field="observation.metrics.hold_warmup_sample",
    )
    samples_to_validate: list[tuple[object, str, float | None]] = [
        (
            raw_warmup_sample,
            "observation.metrics.hold_warmup_sample",
            None,
        )
    ]
    samples_to_validate.extend(
        (
            raw_sample,
            f"observation.metrics.performance_samples[{index}]",
            hold_performance[index],
        )
        for index, raw_sample in enumerate(raw_performance_samples)
    )
    required_sample_fields = (
        "tokens_per_second",
        "expert_hit_rate",
        "ssd_bytes_per_token",
        "p50_token_latency_ms",
        "p95_token_latency_ms",
        "generated_token_ids",
        "generated_token_sha256",
        "route_trace",
        "route_trace_sha256",
        "expert_hashes",
        "expert_route_binding",
    )
    hold_warmup_sample: dict[str, object] | None = None
    performance_samples: list[dict[str, object]] = []
    for raw_sample, sample_prefix, expected_tps in samples_to_validate:
        sample = _mapping(raw_sample, field=sample_prefix)
        _require_fields(
            sample,
            required_sample_fields,
            context=sample_prefix,
        )
        sample_tps = _finite_number(
            sample["tokens_per_second"],
            field=f"{sample_prefix}.tokens_per_second",
            positive=True,
        )
        hit_rate = _finite_number(
            sample["expert_hit_rate"],
            field=f"{sample_prefix}.expert_hit_rate",
        )
        if not 0.0 <= hit_rate <= 1.0:
            raise BenchmarkGateError(
                f"{sample_prefix}.expert_hit_rate must be in [0, 1]"
            )
        ssd_bytes = _finite_number(
            sample["ssd_bytes_per_token"],
            field=f"{sample_prefix}.ssd_bytes_per_token",
        )
        if ssd_bytes < 0.0:
            raise BenchmarkGateError(
                f"{sample_prefix}.ssd_bytes_per_token must be nonnegative"
            )
        p50_ms = _finite_number(
            sample["p50_token_latency_ms"],
            field=f"{sample_prefix}.p50_token_latency_ms",
            positive=True,
        )
        p95_ms = _finite_number(
            sample["p95_token_latency_ms"],
            field=f"{sample_prefix}.p95_token_latency_ms",
            positive=True,
        )
        if p95_ms < p50_ms:
            raise BenchmarkGateError(
                f"{sample_prefix}.p95_token_latency_ms is below p50_token_latency_ms"
            )
        if expected_tps is not None and not math.isclose(
            sample_tps, expected_tps, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise BenchmarkGateError(
                f"{sample_prefix}.tokens_per_second differs from "
                "hold_performance_samples"
            )
        raw_sample_tokens = _sequence(
            sample["generated_token_ids"],
            field=f"{sample_prefix}.generated_token_ids",
        )
        sample_tokens = [
            _exact_int(token, field=f"{sample_prefix}.generated_token_ids[{offset}]")
            for offset, token in enumerate(raw_sample_tokens)
        ]
        if not sample_tokens:
            raise BenchmarkGateError(
                f"{sample_prefix}.generated_token_ids must not be empty"
            )
        sample_token_hash = _sha256(
            sample["generated_token_sha256"],
            field=f"{sample_prefix}.generated_token_sha256",
        )
        if canonical_sha256(sample_tokens) != sample_token_hash:
            raise BenchmarkGateError(
                f"{sample_prefix}.generated_token_sha256 does not match exact tokens"
            )
        sample_route_hash = _sha256(
            sample["route_trace_sha256"],
            field=f"{sample_prefix}.route_trace_sha256",
        )
        if canonical_sha256(sample["route_trace"]) != sample_route_hash:
            raise BenchmarkGateError(
                f"{sample_prefix}.route_trace_sha256 does not match exact routes"
            )
        (
            sample_route_trace,
            sample_expert_hashes,
            sample_expert_route_binding,
        ) = _validate_expert_route_binding(
            route_trace=sample["route_trace"],
            expert_hashes=sample["expert_hashes"],
            binding=sample["expert_route_binding"],
            expert_manifest_sha256=identity["expert_manifest_sha256"],
            field=sample_prefix,
        )
        if canonical_sha256(sample_route_trace) != sample_route_hash:
            raise BenchmarkGateError(
                f"{sample_prefix}.route_trace_sha256 does not match exact routes"
            )
        normalized_sample = {
            "tokens_per_second": sample_tps,
            "expert_hit_rate": hit_rate,
            "ssd_bytes_per_token": ssd_bytes,
            "p50_token_latency_ms": p50_ms,
            "p95_token_latency_ms": p95_ms,
            "generated_token_ids": sample_tokens,
            "generated_token_sha256": sample_token_hash,
            "route_trace": sample_route_trace,
            "route_trace_sha256": sample_route_hash,
            "expert_hashes": sample_expert_hashes,
            "expert_route_binding": sample_expert_route_binding,
        }
        if expected_tps is None:
            hold_warmup_sample = normalized_sample
        else:
            performance_samples.append(normalized_sample)
    if hold_warmup_sample is None:
        raise BenchmarkGateError("observation.metrics.hold_warmup_sample is missing")
    metrics: dict[str, object] = {
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "peak_charged_bytes": peak_charged,
        "stress_peak_charged_bytes": stress_peak_charged,
        "hold_performance_samples": list(hold_performance),
        "hold_warmup_sample": hold_warmup_sample,
        "performance_samples": performance_samples,
    }
    return CampaignObservation(
        arm=arm,
        context_tokens=context_tokens,
        repetition=repetition,
        cache_start_state=cache_start_state,
        identity=identity,
        prompt_sha256=prompt_hash,
        generated_token_ids=tokens,
        generated_token_sha256=token_hash,
        route_trace=route_trace,
        route_trace_sha256=route_hash,
        expert_hashes=expert_hashes,
        kv_growth_steps=kv_growth_steps,
        timeline=timeline,
        metrics=metrics,
    )


def validate_allocator_probe(value: Mapping[str, object]) -> dict[str, object]:
    """Require registered slab bytes to disappear from active plus cache memory."""

    _require_fields(
        value,
        (
            "schema",
            "manifest",
            "before_allocation",
            "before_release",
            "after_release",
            "untouched_slab_executable",
            "error",
        ),
        context="allocator probe",
    )
    if value["schema"] != SCHEMA_PROBE:
        raise BenchmarkGateError(f"allocator probe.schema must be {SCHEMA_PROBE}")
    if value["error"] is not None:
        raise BenchmarkGateError(f"allocator-release probe failed: {value['error']}")
    manifest = _mapping(value["manifest"], field="allocator probe.manifest")
    _require_fields(
        manifest,
        (
            "identity",
            "selected_slab_id",
            "selected_registered_physical_bytes",
            "untouched_slab_id",
            "allocated_slab_count",
            "slabs",
        ),
        context="allocator probe.manifest",
    )
    _validate_identity(
        _mapping(manifest["identity"], field="allocator probe.manifest.identity"),
        context="allocator probe.manifest.identity",
    )
    selected = _nonempty_string(
        manifest["selected_slab_id"],
        field="allocator probe.manifest.selected_slab_id",
    )
    untouched = _nonempty_string(
        manifest["untouched_slab_id"],
        field="allocator probe.manifest.untouched_slab_id",
    )
    if selected == untouched:
        raise BenchmarkGateError(
            "allocator probe selected and untouched slab are identical"
        )
    slab_bytes = _exact_int(
        manifest["selected_registered_physical_bytes"],
        field="allocator probe.manifest.selected_registered_physical_bytes",
        minimum=1,
    )
    if (
        _exact_int(
            manifest["allocated_slab_count"],
            field="allocator probe.manifest.allocated_slab_count",
        )
        < 2
    ):
        raise BenchmarkGateError(
            "allocator probe must allocate at least two real slabs"
        )
    raw_slabs = _sequence(manifest["slabs"], field="allocator probe.manifest.slabs")
    slabs = tuple(
        ProbeSlab(
            slab_id=_nonempty_string(
                _mapping(item, field=f"allocator probe.manifest.slabs[{index}]").get(
                    "slab_id"
                ),
                field=f"allocator probe.manifest.slabs[{index}].slab_id",
            ),
            registered_physical_bytes=_exact_int(
                _mapping(item, field=f"allocator probe.manifest.slabs[{index}]").get(
                    "registered_physical_bytes"
                ),
                field=(
                    f"allocator probe.manifest.slabs[{index}].registered_physical_bytes"
                ),
                minimum=1,
            ),
        )
        for index, item in enumerate(raw_slabs)
    )
    if len(slabs) != manifest["allocated_slab_count"] or len(slabs) < 2:
        raise BenchmarkGateError("allocator probe slab manifest count differs")
    if len({slab.slab_id for slab in slabs}) != len(slabs):
        raise BenchmarkGateError("allocator probe slab IDs must be unique")
    slab_by_id = {slab.slab_id: slab for slab in slabs}
    if selected not in slab_by_id or untouched not in slab_by_id:
        raise BenchmarkGateError("allocator probe selected slab IDs are not manifested")
    if slab_by_id[selected].registered_physical_bytes != slab_bytes:
        raise BenchmarkGateError("allocator probe selected slab byte count drifted")
    before_allocation = AllocatorSample.from_mapping(
        _mapping(value["before_allocation"], field="allocator probe.before_allocation"),
        field="allocator probe.before_allocation",
    )
    before = AllocatorSample.from_mapping(
        _mapping(value["before_release"], field="allocator probe.before_release"),
        field="allocator probe.before_release",
    )
    registered_total = sum(slab.registered_physical_bytes for slab in slabs)
    if before.charged_bytes - before_allocation.charged_bytes < registered_total:
        raise BenchmarkGateError(
            "allocator probe allocation did not add all registered slab bytes"
        )
    after = AllocatorSample.from_mapping(
        _mapping(value["after_release"], field="allocator probe.after_release"),
        field="allocator probe.after_release",
    )
    charged_release = before.charged_bytes - after.charged_bytes
    if charged_release < slab_bytes:
        raise BenchmarkGateError(
            "charged allocator bytes did not fall by the selected registered slab bytes"
        )
    if value["untouched_slab_executable"] is not True:
        raise BenchmarkGateError("allocator probe untouched slab is not executable")
    result = dict(value)
    result["charged_release_bytes"] = charged_release
    result["gate_passed"] = True
    return result


def _validate_artifact_attestation(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the independent full-payload attestation after the campaign."""

    fields = (
        "schema",
        "model_artifact_sha256",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "sidecar_fingerprint",
        "resident_payload_bytes",
        "resident_payload_sha256",
        "resident_shard_fingerprints",
        "payload_hash_verified",
        "payload_hash_io_mode",
    )
    _require_fields(value, fields, context="post-campaign artifact attestation")
    if set(value) != set(fields):
        raise BenchmarkGateError(
            "post-campaign artifact attestation has unknown fields"
        )
    if value["schema"] != SCHEMA_ARTIFACT_ATTESTATION:
        raise BenchmarkGateError(
            "post-campaign artifact attestation.schema must be "
            f"{SCHEMA_ARTIFACT_ATTESTATION}"
        )
    if value["payload_hash_verified"] is not True:
        raise BenchmarkGateError(
            "post-campaign artifact attestation.payload_hash_verified must be true"
        )
    if value["payload_hash_io_mode"] != "f-nocache":
        raise BenchmarkGateError(
            "post-campaign artifact attestation.payload_hash_io_mode must be f-nocache"
        )
    fingerprint = _mapping(
        value["sidecar_fingerprint"],
        field="post-campaign artifact attestation.sidecar_fingerprint",
    )
    fingerprint_fields = ("device", "inode", "size", "mtime_ns", "ctime_ns")
    _require_fields(
        fingerprint,
        fingerprint_fields,
        context="post-campaign artifact attestation.sidecar_fingerprint",
    )
    if set(fingerprint) != set(fingerprint_fields):
        raise BenchmarkGateError(
            "post-campaign artifact attestation.sidecar_fingerprint has unknown fields"
        )
    parsed_fingerprint = {
        field: _exact_int(
            fingerprint[field],
            field=(f"post-campaign artifact attestation.sidecar_fingerprint.{field}"),
            minimum=0,
        )
        for field in fingerprint_fields
    }
    resident_payload_bytes = _exact_int(
        value["resident_payload_bytes"],
        field="post-campaign artifact attestation.resident_payload_bytes",
        minimum=1,
    )
    resident_payload_sha256 = _sha256(
        value["resident_payload_sha256"],
        field="post-campaign artifact attestation.resident_payload_sha256",
    )
    raw_resident_fingerprints = _sequence(
        value["resident_shard_fingerprints"],
        field="post-campaign artifact attestation.resident_shard_fingerprints",
    )
    if not raw_resident_fingerprints:
        raise BenchmarkGateError(
            "post-campaign artifact attestation.resident_shard_fingerprints "
            "must not be empty"
        )
    resident_fingerprints: list[dict[str, int | str]] = []
    for index, item in enumerate(raw_resident_fingerprints):
        raw = _mapping(
            item,
            field=(
                "post-campaign artifact attestation."
                f"resident_shard_fingerprints[{index}]"
            ),
        )
        fields_with_name = ("name", *fingerprint_fields)
        _require_fields(
            raw,
            fields_with_name,
            context=(
                "post-campaign artifact attestation."
                f"resident_shard_fingerprints[{index}]"
            ),
        )
        if set(raw) != set(fields_with_name):
            raise BenchmarkGateError(
                "post-campaign artifact attestation resident fingerprint has "
                "unknown fields"
            )
        resident_fingerprints.append(
            {
                "name": _nonempty_string(
                    raw["name"],
                    field=(
                        "post-campaign artifact attestation."
                        f"resident_shard_fingerprints[{index}].name"
                    ),
                ),
                **{
                    field: _exact_int(
                        raw[field],
                        field=(
                            "post-campaign artifact attestation."
                            f"resident_shard_fingerprints[{index}].{field}"
                        ),
                        minimum=0,
                    )
                    for field in fingerprint_fields
                },
            }
        )
    names = [str(item["name"]) for item in resident_fingerprints]
    if names != sorted(set(names)):
        raise BenchmarkGateError(
            "post-campaign resident shard fingerprints must be sorted and unique"
        )
    artifact_stat_sha256 = _sha256(
        value["artifact_stat_sha256"],
        field="post-campaign artifact attestation.artifact_stat_sha256",
    )
    if (
        canonical_sha256(
            {
                "sidecar": parsed_fingerprint,
                "resident_shards": resident_fingerprints,
            }
        )
        != artifact_stat_sha256
    ):
        raise BenchmarkGateError(
            "post-campaign artifact attestation.artifact_stat_sha256 differs "
            "from the payload fingerprints"
        )
    return {
        "schema": SCHEMA_ARTIFACT_ATTESTATION,
        "model_artifact_sha256": _sha256(
            value["model_artifact_sha256"],
            field="post-campaign artifact attestation.model_artifact_sha256",
        ),
        "expert_manifest_sha256": _sha256(
            value["expert_manifest_sha256"],
            field="post-campaign artifact attestation.expert_manifest_sha256",
        ),
        "artifact_pins_sha256": _sha256(
            value["artifact_pins_sha256"],
            field="post-campaign artifact attestation.artifact_pins_sha256",
        ),
        "artifact_stat_sha256": artifact_stat_sha256,
        "sidecar_fingerprint": parsed_fingerprint,
        "resident_payload_bytes": resident_payload_bytes,
        "resident_payload_sha256": resident_payload_sha256,
        "resident_shard_fingerprints": resident_fingerprints,
        "payload_hash_verified": True,
        "payload_hash_io_mode": "f-nocache",
    }


def run_allocator_release_probe(
    *,
    identity: Mapping[str, object],
    allocate_slabs: Callable[[], Sequence[ProbeSlab]],
    evaluate_slabs: Callable[[Sequence[ProbeSlab]], None],
    sample_allocator: Callable[[], AllocatorSample],
    release_slab: Callable[[ProbeSlab], None],
    execute_slab: Callable[[ProbeSlab], bool],
) -> dict[str, object]:
    """Run the focused two-slab physical-release gate and emit JSON-safe evidence."""

    validated_identity = _validate_identity(identity, context="probe identity")
    result: dict[str, object] = {
        "schema": SCHEMA_PROBE,
        "manifest": {"identity": validated_identity},
        "gate_passed": False,
        "error": None,
    }
    try:
        before_allocation = sample_allocator()
        slabs = tuple(allocate_slabs())
        if len(slabs) < 2:
            raise BenchmarkGateError(
                "allocator probe must allocate at least two real slabs"
            )
        if len({slab.slab_id for slab in slabs}) != len(slabs):
            raise BenchmarkGateError("allocator probe slab IDs must be unique")
        evaluate_slabs(slabs)
        before_release = sample_allocator()
        selected, untouched = slabs[:2]
        release_slab(selected)
        after_release = sample_allocator()
        untouched_executable = execute_slab(untouched) is True
        result.update(
            {
                "manifest": {
                    "identity": validated_identity,
                    "selected_slab_id": selected.slab_id,
                    "selected_registered_physical_bytes": (
                        selected.registered_physical_bytes
                    ),
                    "untouched_slab_id": untouched.slab_id,
                    "allocated_slab_count": len(slabs),
                    "slabs": [asdict(slab) for slab in slabs],
                },
                "before_allocation": asdict(before_allocation),
                "before_release": asdict(before_release),
                "after_release": asdict(after_release),
                "untouched_slab_executable": untouched_executable,
            }
        )
        return validate_allocator_probe(result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["gate_passed"] = False
        return result


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise BenchmarkGateError("cannot compute a confidence interval without samples")
    count = _exact_int(resamples, field="bootstrap_resamples", minimum=100)
    rng = random.Random(_exact_int(seed, field="bootstrap_seed"))
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(count)
    )
    low_index = max(0, math.floor(0.025 * count))
    high_index = min(count - 1, math.ceil(0.975 * count) - 1)
    return means[low_index], means[high_index]


def _metric_summary(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    low, high = _bootstrap_mean_ci(values, resamples=resamples, seed=seed)
    return {
        "samples": list(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "confidence_interval_95": [low, high],
        "confidence_interval_method": "paired bootstrap mean percentile",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


_HOLD_PERFORMANCE_FIELDS = (
    "tokens_per_second",
    "expert_hit_rate",
    "ssd_bytes_per_token",
    "p50_token_latency_ms",
    "p95_token_latency_ms",
)


def _hold_performance_means(
    observation: CampaignObservation,
) -> dict[str, float]:
    raw_samples = _sequence(
        observation.metrics["performance_samples"],
        field="observation.metrics.performance_samples",
    )
    samples = [
        _mapping(sample, field=f"performance_samples[{index}]")
        for index, sample in enumerate(raw_samples)
    ]
    return {
        field: statistics.fmean(float(sample[field]) for sample in samples)
        for field in _HOLD_PERFORMANCE_FIELDS
    }


def _hold_expert_physical_bytes(observation: CampaignObservation) -> int:
    holds = [point for point in observation.timeline if point.phase == "hold"]
    if not holds:
        raise BenchmarkGateError("observation has no stable hold timeline")
    return holds[0].expert_slab_physical_bytes


def _hold_expert_slab_bytes(observation: CampaignObservation) -> int:
    holds = [point for point in observation.timeline if point.phase == "hold"]
    if not holds:
        raise BenchmarkGateError("observation has no stable hold timeline")
    point = holds[0]
    active_records = int(point.resource_evidence["expert_active_records"])
    if active_records <= 0:
        raise BenchmarkGateError("stable hold has no active expert record telemetry")
    physical_bytes = point.expert_slab_physical_bytes
    record_bytes, remainder = divmod(physical_bytes, active_records)
    if record_bytes <= 0 or remainder:
        raise BenchmarkGateError(
            "stable hold expert bytes do not divide into active physical records"
        )
    arm_config = _mapping(
        observation.identity["arm_config"], field="observation.identity.arm_config"
    )
    streaming = _mapping(
        arm_config.get("expert_streaming_config"),
        field="observation.identity.arm_config.expert_streaming_config",
    )
    slab_slots = _exact_int(
        streaming.get("expert_slab_slots"),
        field="expert_streaming_config.expert_slab_slots",
        minimum=1,
    )
    return slab_slots * record_bytes


def _campaign_acceptance_reasons(
    *,
    validated: Mapping[tuple[int, int, str], CampaignObservation],
    summaries: Mapping[str, Mapping[str, object]],
    repetitions: int,
) -> list[str]:
    reasons: list[str] = []
    for key, observation in validated.items():
        context_tokens, repetition, arm = key
        prefix = f"{context_tokens}/{repetition}/{arm}"
        if int(observation.metrics["peak_charged_bytes"]) > OPERATING_TARGET_BYTES:
            reasons.append(f"{prefix}: normal charged memory exceeded 110 GiB")
        if int(observation.metrics["stress_peak_charged_bytes"]) >= HARD_CEILING_BYTES:
            reasons.append(f"{prefix}: stress charged memory reached 112 GiB")
        compressed = [
            int(point.resource_evidence["process_compressed_bytes"])
            for point in observation.timeline
        ]
        if max(compressed) - min(compressed) > MAX_PROCESS_COMPRESSED_GROWTH_BYTES:
            reasons.append(f"{prefix}: process compressor growth exceeded 512 MiB")
        for point in observation.timeline:
            resource = point.resource_evidence
            if int(resource["classified_bytes"]) > int(
                resource["classified_target_bytes"]
            ):
                reasons.append(f"{prefix}: classified memory exceeded 109 GiB")
                break
            if int(resource["charged_bytes"]) > OPERATING_TARGET_BYTES:
                reasons.append(f"{prefix}: sampled charged memory exceeded 110 GiB")
                break
            if int(resource["system_swap_delta_bytes"]) > 0:
                reasons.append(f"{prefix}: system swap grew during the arm")
                break
            if bool(resource["failed_closed"]):
                reasons.append(f"{prefix}: runtime reported a fail-closed event")
                break
            if int(resource["admission_failures"]) > 0:
                reasons.append(f"{prefix}: request admission failed")
                break
            if int(resource["resize_failures"]) > 0:
                reasons.append(f"{prefix}: expert resize failed")
                break

    for context_tokens in CONTEXT_MATRIX_TOKENS:
        summary = summaries[str(context_tokens)]
        for repetition in range(repetitions):
            static = validated[(context_tokens, repetition, "static")]
            dynamic = validated[(context_tokens, repetition, "dynamic")]
            fixed_fields = (
                "resident_model_bytes",
                "inflight_expert_staging_bytes",
                "runtime_workspace_bytes",
            )
            static_fixed = sum(
                int(static.timeline[0].resource_evidence[field])
                for field in fixed_fields
            )
            dynamic_fixed = sum(
                int(dynamic.timeline[0].resource_evidence[field])
                for field in fixed_fields
            )
            if static_fixed != dynamic_fixed:
                reasons.append(
                    f"{context_tokens}/{repetition}: paired fixed physical pools differ"
                )
        if context_tokens < HY3_Q4_TOTAL_CONTEXT_TOKENS:
            for repetition in range(repetitions):
                static = validated[(context_tokens, repetition, "static")]
                dynamic = validated[(context_tokens, repetition, "dynamic")]
                if _hold_expert_physical_bytes(dynamic) <= _hold_expert_physical_bytes(
                    static
                ):
                    reasons.append(
                        f"{context_tokens}/{repetition}: dynamic expert capacity "
                        "did not exceed the static control"
                    )
            hit_summary = _mapping(
                summary["dynamic_minus_static_expert_hit_rate"],
                field="expert hit-rate summary",
            )
            hit_interval = _sequence(
                hit_summary["confidence_interval_95"],
                field="expert hit-rate confidence interval",
            )
            ssd_interval = _sequence(
                _mapping(
                    summary["dynamic_minus_static_ssd_bytes_per_token"],
                    field="SSD summary",
                )["confidence_interval_95"],
                field="SSD confidence interval",
            )
            tps_interval = _sequence(
                _mapping(
                    summary["dynamic_vs_static_tps_ratio"],
                    field="TPS summary",
                )["confidence_interval_95"],
                field="TPS confidence interval",
            )
            measurable = (
                float(hit_interval[0]) > 0.0
                or float(ssd_interval[1]) < 0.0
                or float(tps_interval[0]) > 1.0
            )
            if not measurable:
                reasons.append(
                    f"{context_tokens}: extra expert capacity produced no "
                    "measurable hit-rate, SSD, or TPS improvement"
                )
            continue

        for repetition in range(repetitions):
            static = validated[(context_tokens, repetition, "static")]
            dynamic = validated[(context_tokens, repetition, "dynamic")]
            capacity_delta = abs(
                _hold_expert_physical_bytes(dynamic)
                - _hold_expert_physical_bytes(static)
            )
            slab_tolerance = min(
                _hold_expert_slab_bytes(static),
                _hold_expert_slab_bytes(dynamic),
            )
            if capacity_delta >= slab_tolerance:
                reasons.append(
                    f"{context_tokens}/{repetition}: expert capacity did not "
                    "converge to the static control"
                )
            static_perf = _hold_performance_means(static)
            dynamic_perf = _hold_performance_means(dynamic)
            if (
                dynamic_perf["tokens_per_second"]
                < (1.0 - MAX_128K_PERFORMANCE_REGRESSION)
                * static_perf["tokens_per_second"]
                or dynamic_perf["p50_token_latency_ms"]
                > (1.0 + MAX_128K_PERFORMANCE_REGRESSION)
                * static_perf["p50_token_latency_ms"]
                or dynamic_perf["p95_token_latency_ms"]
                > (1.0 + MAX_128K_PERFORMANCE_REGRESSION)
                * static_perf["p95_token_latency_ms"]
                or dynamic_perf["expert_hit_rate"]
                < static_perf["expert_hit_rate"] - MAX_128K_PERFORMANCE_REGRESSION
                or dynamic_perf["ssd_bytes_per_token"]
                > (1.0 + MAX_128K_PERFORMANCE_REGRESSION)
                * static_perf["ssd_bytes_per_token"]
            ):
                reasons.append(
                    f"{context_tokens}/{repetition}: stable 128K decode performance "
                    "regressed by more than 5%"
                )
    return list(dict.fromkeys(reasons))


def _paired_equal(static: CampaignObservation, dynamic: CampaignObservation) -> None:
    common_identity_fields = (
        "model_key",
        "model_artifact_id",
        "model_artifact_sha256",
        "expert_manifest_id",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "source_git_commit",
        "normalized_config_sha256",
        "kv_quantization",
        "kv_block_size_tokens",
        "total_context_tokens",
    )
    for field in common_identity_fields:
        if static.identity[field] != dynamic.identity[field]:
            raise BenchmarkGateError(f"paired identity drifted at {field}")
    if static.prompt_sha256 != dynamic.prompt_sha256:
        raise BenchmarkGateError("paired prompt identity differs")
    if static.generated_token_ids != dynamic.generated_token_ids:
        raise BenchmarkGateError("paired generated tokens differ")
    if static.generated_token_sha256 != dynamic.generated_token_sha256:
        raise BenchmarkGateError("paired generated token hash differs")
    if static.route_trace != dynamic.route_trace:
        raise BenchmarkGateError("paired exact routes differ")
    if static.route_trace_sha256 != dynamic.route_trace_sha256:
        raise BenchmarkGateError("paired route hash differs")
    if static.expert_hashes != dynamic.expert_hashes:
        raise BenchmarkGateError("paired expert hashes differ")
    static_warmup = _mapping(
        static.metrics["hold_warmup_sample"], field="static hold warm-up sample"
    )
    dynamic_warmup = _mapping(
        dynamic.metrics["hold_warmup_sample"], field="dynamic hold warm-up sample"
    )
    if (
        static_warmup["generated_token_ids"] != dynamic_warmup["generated_token_ids"]
        or static_warmup["generated_token_sha256"]
        != dynamic_warmup["generated_token_sha256"]
    ):
        raise BenchmarkGateError("paired warm-up tokens differ")
    if (
        static_warmup["route_trace"] != dynamic_warmup["route_trace"]
        or static_warmup["route_trace_sha256"] != dynamic_warmup["route_trace_sha256"]
    ):
        raise BenchmarkGateError("paired warm-up routes differ")
    if static_warmup["expert_hashes"] != dynamic_warmup["expert_hashes"]:
        raise BenchmarkGateError("paired warm-up expert hashes differ")
    static_samples = _sequence(
        static.metrics["performance_samples"], field="static performance_samples"
    )
    dynamic_samples = _sequence(
        dynamic.metrics["performance_samples"], field="dynamic performance_samples"
    )
    if len(static_samples) != len(dynamic_samples):
        raise BenchmarkGateError("paired hold sample count differs")
    for index, (static_raw, dynamic_raw) in enumerate(
        zip(static_samples, dynamic_samples, strict=True)
    ):
        static_sample = _mapping(static_raw, field=f"static performance sample {index}")
        dynamic_sample = _mapping(
            dynamic_raw, field=f"dynamic performance sample {index}"
        )
        if (
            static_sample["generated_token_ids"]
            != dynamic_sample["generated_token_ids"]
            or static_sample["generated_token_sha256"]
            != dynamic_sample["generated_token_sha256"]
        ):
            raise BenchmarkGateError(f"paired hold tokens differ at sample {index}")
        if (
            static_sample["route_trace"] != dynamic_sample["route_trace"]
            or static_sample["route_trace_sha256"]
            != dynamic_sample["route_trace_sha256"]
        ):
            raise BenchmarkGateError(f"paired hold routes differ at sample {index}")
        if static_sample["expert_hashes"] != dynamic_sample["expert_hashes"]:
            raise BenchmarkGateError(
                f"paired hold expert hashes differ at sample {index}"
            )
    if static.timeline[-1].slot_health != dynamic.timeline[-1].slot_health:
        raise BenchmarkGateError("paired final slot health differs")


def _require_probe_arm_identity(
    probe_identity: Mapping[str, object],
    arm_identity: Mapping[str, object],
) -> None:
    for field in _PROBE_BOUND_IDENTITY_FIELDS:
        if probe_identity[field] != arm_identity[field]:
            raise BenchmarkGateError(f"allocator probe identity drifted at {field}")


def run_balanced_campaign(
    *,
    allocator_probe: Mapping[str, object],
    execute_arm: Callable[[str, int, int], Mapping[str, object]],
    repetitions: int,
    contexts: Sequence[int] = CONTEXT_MATRIX_TOKENS,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 46,
) -> dict[str, object]:
    """Run the exact issue #46 matrix only after the allocator probe passes."""

    if allocator_probe.get("gate_passed") is not True:
        raise BenchmarkGateError(
            "allocator-release probe did not pass; refusing the full campaign"
        )
    validated_probe = validate_allocator_probe(allocator_probe)
    probe_manifest = _mapping(
        validated_probe["manifest"], field="allocator probe.manifest"
    )
    probe_identity = _validate_identity(
        _mapping(
            probe_manifest["identity"],
            field="allocator probe.manifest.identity",
        ),
        context="allocator probe.manifest.identity",
    )
    schedule = balanced_campaign_schedule(
        repetitions=repetitions,
        contexts=contexts,
    )
    raw_observations: list[dict[str, object]] = []
    validated: dict[tuple[int, int, str], CampaignObservation] = {}
    for campaign_order_index, entry in enumerate(schedule):
        raw = execute_arm(entry.arm, entry.context_tokens, entry.repetition)
        observation = validate_campaign_observation(raw)
        _require_probe_arm_identity(probe_identity, observation.identity)
        if (
            observation.arm != entry.arm
            or observation.context_tokens != entry.context_tokens
            or observation.repetition != entry.repetition
        ):
            raise BenchmarkGateError(
                "arm executor returned observation metadata for a different schedule row"
            )
        key = (entry.context_tokens, entry.repetition, entry.arm)
        if key in validated:
            raise BenchmarkGateError(f"duplicate campaign observation {key}")
        validated[key] = observation
        raw_copy = dict(raw)
        raw_copy["campaign_order_index"] = campaign_order_index
        raw_copy["arm_order_index"] = entry.order_index
        raw_observations.append(raw_copy)

    paired_samples: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for context_index, context_tokens in enumerate(CONTEXT_MATRIX_TOKENS):
        context_pairs: list[dict[str, object]] = []
        paired_metrics: dict[str, list[float]] = {
            "dynamic_vs_static_tps_ratio": [],
            "dynamic_minus_static_tps": [],
            "dynamic_minus_static_expert_hit_rate": [],
            "dynamic_minus_static_ssd_bytes_per_token": [],
            "dynamic_minus_static_p50_token_latency_ms": [],
            "dynamic_minus_static_p95_token_latency_ms": [],
        }
        for repetition in range(repetitions):
            static = validated[(context_tokens, repetition, "static")]
            dynamic = validated[(context_tokens, repetition, "dynamic")]
            _paired_equal(static, dynamic)
            static_performance = _hold_performance_means(static)
            dynamic_performance = _hold_performance_means(dynamic)
            static_tps = static_performance["tokens_per_second"]
            dynamic_tps = dynamic_performance["tokens_per_second"]
            ratio = dynamic_tps / static_tps
            delta = dynamic_tps - static_tps
            pair = {
                "context_tokens": context_tokens,
                "repetition": repetition,
                "physical_arm_order": list(
                    ("static", "dynamic")
                    if repetition % 2 == 0
                    else ("dynamic", "static")
                ),
                "static_tokens_per_second": static_tps,
                "dynamic_tokens_per_second": dynamic_tps,
                "dynamic_vs_static_tps_ratio": ratio,
                "dynamic_minus_static_tps": delta,
                "static_expert_hit_rate": static_performance["expert_hit_rate"],
                "dynamic_expert_hit_rate": dynamic_performance["expert_hit_rate"],
                "dynamic_minus_static_expert_hit_rate": (
                    dynamic_performance["expert_hit_rate"]
                    - static_performance["expert_hit_rate"]
                ),
                "static_ssd_bytes_per_token": static_performance["ssd_bytes_per_token"],
                "dynamic_ssd_bytes_per_token": dynamic_performance[
                    "ssd_bytes_per_token"
                ],
                "dynamic_minus_static_ssd_bytes_per_token": (
                    dynamic_performance["ssd_bytes_per_token"]
                    - static_performance["ssd_bytes_per_token"]
                ),
                "static_p50_token_latency_ms": static_performance[
                    "p50_token_latency_ms"
                ],
                "dynamic_p50_token_latency_ms": dynamic_performance[
                    "p50_token_latency_ms"
                ],
                "dynamic_minus_static_p50_token_latency_ms": (
                    dynamic_performance["p50_token_latency_ms"]
                    - static_performance["p50_token_latency_ms"]
                ),
                "static_p95_token_latency_ms": static_performance[
                    "p95_token_latency_ms"
                ],
                "dynamic_p95_token_latency_ms": dynamic_performance[
                    "p95_token_latency_ms"
                ],
                "dynamic_minus_static_p95_token_latency_ms": (
                    dynamic_performance["p95_token_latency_ms"]
                    - static_performance["p95_token_latency_ms"]
                ),
                "static_expert_slab_physical_bytes": (
                    _hold_expert_physical_bytes(static)
                ),
                "dynamic_expert_slab_physical_bytes": (
                    _hold_expert_physical_bytes(dynamic)
                ),
                "static_peak_charged_bytes": static.metrics["peak_charged_bytes"],
                "dynamic_peak_charged_bytes": dynamic.metrics["peak_charged_bytes"],
                "generated_token_sha256": static.generated_token_sha256,
                "route_trace_sha256": static.route_trace_sha256,
                "final_slot_health_sha256": static.timeline[-1].slot_health_sha256,
            }
            context_pairs.append(pair)
            paired_samples.append(pair)
            for metric in paired_metrics:
                paired_metrics[metric].append(float(pair[metric]))
        summaries[str(context_tokens)] = {
            "sample_count": repetitions,
            **{
                metric: _metric_summary(
                    values,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + context_index * 10 + metric_index,
                )
                for metric_index, (metric, values) in enumerate(paired_metrics.items())
            },
            "raw_pairs": context_pairs,
        }
    rejection_reasons = _campaign_acceptance_reasons(
        validated=validated,
        summaries=summaries,
        repetitions=repetitions,
    )
    return {
        "schema": SCHEMA_CAMPAIGN,
        "status": "passed" if not rejection_reasons else "rejected",
        "acceptance": {
            "passed": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "operating_target_bytes": OPERATING_TARGET_BYTES,
            "hard_ceiling_bytes": HARD_CEILING_BYTES,
            "max_process_compressed_growth_bytes": (
                MAX_PROCESS_COMPRESSED_GROWTH_BYTES
            ),
            "max_128k_performance_regression": (MAX_128K_PERFORMANCE_REGRESSION),
            "quality_gate_issue": 43,
            "quality_gate_status": "required-separately",
        },
        "contexts": list(CONTEXT_MATRIX_TOKENS),
        "repetitions": repetitions,
        "physical_arm_order": [entry.arm for entry in schedule],
        "allocator_release_probe": validated_probe,
        "raw_observations": raw_observations,
        "paired_samples": paired_samples,
        "paired_metrics_by_context": summaries,
    }


def run_exclusive_hardware_window(
    workload: Callable[[], _T],
    *,
    hooks: QwenIsolationHooks,
) -> _T:
    """Run work with exact Qwen restoration guaranteed in ``finally``.

    Restoration is attempted after ordinary unload and workload failures.  If
    subprocess-group cleanup cannot prove the GPU workload absent, Qwen stays
    unloaded and the recovery journal and exclusive lane are retained.  The
    lane is released only after restoration verification has run.
    """

    cleanup_state = _SUBPROCESS_CLEANUP_UNPROVEN.set(False)
    try:
        with _termination_cleanup_scope():
            lane_acquired = False
            captured = False
            state: object = None
            try:
                with _blocked_termination_signals():
                    hooks.acquire_lane()
                    lane_acquired = True
                state = hooks.capture()
                captured = True
                hooks.unload(state)
                return workload()
            finally:
                if not _SUBPROCESS_CLEANUP_UNPROVEN.get():
                    with _blocked_termination_signals():
                        if captured:
                            hooks.restore(state)
                            if hooks.verify_restored(state) is not True:
                                raise BenchmarkGateError(
                                    "Qwen state does not exactly match its captured state"
                                )
                        if lane_acquired:
                            hooks.release_lane()
    finally:
        _SUBPROCESS_CLEANUP_UNPROVEN.reset(cleanup_state)


@contextmanager
def _termination_cleanup_scope():
    previous_handlers: dict[int, object] = {}

    def terminate_after_cleanup(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, terminate_after_cleanup)
    try:
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


@contextmanager
def _blocked_termination_signals():
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return
    previous = pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + float(timeout_seconds)
    while _process_group_exists(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _signal_process_group(process_group: int, signum: int) -> None:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        pass


def _terminate_json_subprocess_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> tuple[str, str]:
    """Boundedly reap the leader and prove its isolated group is gone."""

    process_group = int(process.pid)
    stdout = ""
    stderr = ""
    _signal_process_group(process_group, signal.SIGTERM)
    leader_reaped = process.returncode is not None
    if not leader_reaped:
        try:
            stdout, stderr = process.communicate(timeout=grace_seconds)
            leader_reaped = True
        except subprocess.TimeoutExpired:
            pass

    group_exists = _process_group_exists(process_group)
    if group_exists and leader_reaped:
        group_exists = not _wait_for_process_group_exit(
            process_group,
            timeout_seconds=grace_seconds,
        )
    if not leader_reaped or group_exists:
        _signal_process_group(process_group, signal.SIGKILL)
        if not leader_reaped:
            try:
                stdout, stderr = process.communicate(timeout=grace_seconds)
                leader_reaped = True
            except subprocess.TimeoutExpired:
                pass
        if not _wait_for_process_group_exit(
            process_group,
            timeout_seconds=grace_seconds,
        ):
            raise _subprocess_cleanup_not_proven(
                "JSON subprocess process group survived SIGKILL"
            )
    if not leader_reaped:
        try:
            stdout, stderr = process.communicate(timeout=grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise _subprocess_cleanup_not_proven(
                "JSON subprocess leader could not be reaped after SIGKILL"
            ) from exc
    if _process_group_exists(process_group):
        raise _subprocess_cleanup_not_proven(
            "JSON subprocess process group remained after cleanup"
        )
    return stdout, stderr


def _terminate_surviving_json_subprocess_descendants(
    process_group: int,
    *,
    grace_seconds: float,
) -> None:
    """Remove descendants that outlived an already-reaped group leader."""

    _signal_process_group(process_group, signal.SIGTERM)
    if _wait_for_process_group_exit(
        process_group,
        timeout_seconds=grace_seconds,
    ):
        return
    _signal_process_group(process_group, signal.SIGKILL)
    if not _wait_for_process_group_exit(
        process_group,
        timeout_seconds=grace_seconds,
    ):
        raise _subprocess_cleanup_not_proven(
            "JSON subprocess descendants survived SIGKILL"
        )


def _run_json_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    input_payload: object | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Sequence[int] = (0,),
    timeout_seconds: float = DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS,
    termination_grace_seconds: float = (
        DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS
    ),
) -> tuple[Mapping[str, object], int]:
    """Run argv, retaining the exact code alongside its single JSON object."""

    argv = tuple(_nonempty_string(item, field="command argv") for item in command)
    if not argv:
        raise BenchmarkGateError("JSON subprocess command must not be empty")
    accepted_codes = tuple(
        _exact_int(code, field="allowed subprocess return code")
        for code in allowed_returncodes
    )
    if not accepted_codes or len(set(accepted_codes)) != len(accepted_codes):
        raise BenchmarkGateError(
            "allowed subprocess return codes must be nonempty and unique"
        )
    encoded_input = (
        None
        if input_payload is None
        else json.dumps(input_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    timeout = _finite_number(
        timeout_seconds,
        field="JSON subprocess timeout_seconds",
        positive=True,
    )
    grace = _finite_number(
        termination_grace_seconds,
        field="JSON subprocess termination_grace_seconds",
        positive=True,
    )
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=None if env is None else {**os.environ, **env},
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=encoded_input, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            with _blocked_termination_signals():
                cleanup_stdout, cleanup_stderr = _terminate_json_subprocess_group(
                    process,
                    grace_seconds=grace,
                )
        except BaseException as cleanup_error:
            raise cleanup_error from exc
        detail = cleanup_stderr.strip() or cleanup_stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise BenchmarkGateError(
            f"JSON subprocess timed out after {timeout:g} seconds: {argv!r}{suffix}"
        ) from exc
    except BaseException as exc:
        try:
            with _blocked_termination_signals():
                _terminate_json_subprocess_group(process, grace_seconds=grace)
        except BaseException as cleanup_error:
            raise cleanup_error from exc
        raise
    if _process_group_exists(process.pid):
        with _blocked_termination_signals():
            _terminate_surviving_json_subprocess_descendants(
                process.pid,
                grace_seconds=grace,
            )
        raise BenchmarkGateError(
            "JSON subprocess descendants remained after the leader exited"
        )
    returncode = process.returncode
    if returncode is None:
        raise BenchmarkGateError("JSON subprocess leader was not reaped")
    if returncode not in accepted_codes:
        detail = stderr.strip() or stdout.strip()
        raise BenchmarkGateError(f"JSON subprocess failed ({returncode}): {detail}")
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkGateError(
            "JSON subprocess stdout is not one JSON value"
        ) from exc
    return _mapping(value, field="JSON subprocess stdout"), returncode


def run_json_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    input_payload: object | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Sequence[int] = (0,),
    timeout_seconds: float = DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS,
    termination_grace_seconds: float = (
        DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS
    ),
) -> Mapping[str, object]:
    """Run an argv-only command and parse exactly one JSON object from stdout."""

    value, _returncode = _run_json_subprocess(
        command,
        cwd=cwd,
        input_payload=input_payload,
        env=env,
        allowed_returncodes=allowed_returncodes,
        timeout_seconds=timeout_seconds,
        termination_grace_seconds=termination_grace_seconds,
    )
    return value


def _require_quality_returncode(returncode: int, *, passed: bool) -> None:
    expected = 0 if passed else 2
    if returncode != expected:
        raise BenchmarkGateError(
            "KV quality subprocess return code contradicts recomputed acceptance"
        )


def format_arm_command(
    template: Sequence[str],
    *,
    arm: str,
    context_tokens: int,
    repetition: int,
) -> tuple[str, ...]:
    """Format an argv template without invoking a shell."""

    fields = {
        "arm": arm,
        "context_tokens": str(context_tokens),
        "repetition": str(repetition),
    }
    try:
        return tuple(
            _nonempty_string(item, field="arm command template").format_map(fields)
            for item in template
        )
    except KeyError as exc:
        raise BenchmarkGateError(
            f"unknown arm command placeholder: {exc.args[0]}"
        ) from exc


def run_subprocess_campaign(
    *,
    probe_command: Sequence[str],
    artifact_verify_command: Sequence[str],
    arm_command_template: Sequence[str],
    repetitions: int,
    quality_command: Sequence[str] | None = None,
    cwd: Path | str | None = None,
    command_runner: Callable[[Sequence[str]], Mapping[str, object]] | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 46,
    subprocess_timeout_seconds: float = DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS,
    subprocess_termination_grace_seconds: float = (
        DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS
    ),
) -> dict[str, object]:
    """Bridge JSON-emitting hardware commands into the pure campaign runner."""

    runner = command_runner or (
        lambda command: run_json_subprocess(
            command,
            cwd=cwd,
            timeout_seconds=subprocess_timeout_seconds,
            termination_grace_seconds=subprocess_termination_grace_seconds,
        )
    )
    probe = validate_allocator_probe(runner(probe_command))
    probe_identity = _validate_identity(
        _mapping(
            _mapping(
                probe["manifest"],
                field="allocator probe.manifest",
            )["identity"],
            field="allocator probe.manifest.identity",
        ),
        context="allocator probe.manifest.identity",
    )

    def execute(arm: str, context_tokens: int, repetition: int) -> Mapping[str, object]:
        return runner(
            format_arm_command(
                arm_command_template,
                arm=arm,
                context_tokens=context_tokens,
                repetition=repetition,
            )
        )

    result = run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=repetitions,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    if quality_command is not None:
        quality_returncode: int | None = None
        if command_runner is None:
            raw_quality, quality_returncode = _run_json_subprocess(
                quality_command,
                cwd=cwd,
                allowed_returncodes=(0, 2),
                timeout_seconds=subprocess_timeout_seconds,
                termination_grace_seconds=subprocess_termination_grace_seconds,
            )
        else:
            raw_quality = runner(quality_command)
        from mtplx.benchmarks.hy3_kv_quality import validate_quality_result

        quality = dict(raw_quality)
        quality_acceptance = dict(validate_quality_result(quality))
        quality_identity = _mapping(quality.get("identity"), field="quality identity")
        for field in (
            "model_artifact_id",
            "model_artifact_sha256",
            "expert_manifest_sha256",
            "artifact_pins_sha256",
            "artifact_stat_sha256",
            "resident_payload_bytes",
            "resident_payload_sha256",
            "source_git_commit",
        ):
            if quality_identity.get(field) != probe_identity[field]:
                raise BenchmarkGateError(
                    f"quality identity drifted from allocator probe at {field}"
                )
        quality_passed = quality_acceptance.get("passed")
        if not isinstance(quality_passed, bool):
            raise BenchmarkGateError("KV quality acceptance must declare passed")
        if quality_returncode is not None:
            _require_quality_returncode(quality_returncode, passed=quality_passed)
        result["kv_quality"] = quality
        campaign_acceptance = dict(
            _mapping(result.get("acceptance"), field="campaign acceptance")
        )
        campaign_acceptance["quality_gate_status"] = (
            "passed" if quality_passed else "rejected"
        )
        if not quality_passed:
            raw_reasons = quality_acceptance.get("rejection_reasons", ())
            quality_reasons = tuple(
                _nonempty_string(reason, field="KV quality rejection reason")
                for reason in _sequence(
                    raw_reasons,
                    field="KV quality rejection reasons",
                )
            )
            if not quality_reasons:
                quality_reasons = ("KV quality gate rejected",)
            existing_reasons = tuple(
                _nonempty_string(reason, field="campaign rejection reason")
                for reason in _sequence(
                    campaign_acceptance.get("rejection_reasons", ()),
                    field="campaign rejection reasons",
                )
            )
            campaign_acceptance["passed"] = False
            campaign_acceptance["rejection_reasons"] = list(
                dict.fromkeys((*existing_reasons, *quality_reasons))
            )
            result["status"] = "rejected"
        result["acceptance"] = campaign_acceptance
    post_attestation = _validate_artifact_attestation(runner(artifact_verify_command))
    for field in (
        "model_artifact_sha256",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "resident_payload_bytes",
        "resident_payload_sha256",
    ):
        if post_attestation[field] != probe_identity[field]:
            raise BenchmarkGateError(
                f"post-campaign artifact attestation drifted at {field}"
            )
    result["post_campaign_artifact_attestation"] = post_attestation
    return result


__all__ = [
    "CONTEXT_MATRIX_TOKENS",
    "DEFAULT_JSON_SUBPROCESS_TERMINATION_GRACE_SECONDS",
    "DEFAULT_JSON_SUBPROCESS_TIMEOUT_SECONDS",
    "HY3_Q4_KV_BLOCK_BYTES",
    "HY3_Q4_KV_BYTES_PER_TOKEN",
    "HY3_Q4_MAX_BLOCKS",
    "AllocatorSample",
    "BenchmarkGateError",
    "CacheStartState",
    "CampaignObservation",
    "CampaignScheduleEntry",
    "ProbeSlab",
    "QwenIsolationHooks",
    "balanced_campaign_schedule",
    "canonical_sha256",
    "format_arm_command",
    "normalize_arm_config",
    "run_allocator_release_probe",
    "run_balanced_campaign",
    "run_exclusive_hardware_window",
    "run_json_subprocess",
    "run_subprocess_campaign",
    "validate_allocator_probe",
    "validate_campaign_observation",
]
