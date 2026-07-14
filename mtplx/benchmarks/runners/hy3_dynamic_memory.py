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
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from mtplx.memory_broker import (
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BLOCK_TOKENS,
    HY3_Q4_KV_BYTES_PER_TOKEN,
)

CONTEXT_MATRIX_TOKENS = (4_096, 32_768, 65_536, 131_072)
HY3_Q4_TOTAL_CONTEXT_TOKENS = 131_072
HY3_Q4_KV_BLOCK_SIZE_TOKENS = HY3_Q4_KV_BLOCK_TOKENS
HY3_Q4_MAX_BLOCKS = HY3_Q4_TOTAL_CONTEXT_TOKENS // HY3_Q4_KV_BLOCK_SIZE_TOKENS
MIN_STABLE_HOLD_SAMPLES = 3
MIN_STABLE_HOLD_DURATION_NS = 1_000_000_000
OPERATING_TARGET_BYTES = 110 * 1024**3
HARD_CEILING_BYTES = 112 * 1024**3
MAX_PROCESS_COMPRESSED_GROWTH_BYTES = 512 * 1024**2
MAX_128K_PERFORMANCE_REGRESSION = 0.05
SCHEMA_OBSERVATION = "mtplx-hy3-dynamic-memory-observation-v1"
SCHEMA_PROBE = "mtplx-hy3-allocator-release-probe-v1"
SCHEMA_CAMPAIGN = "mtplx-hy3-dynamic-memory-campaign-v1"

_PROBE_BOUND_IDENTITY_FIELDS = (
    "model_key",
    "model_artifact_id",
    "model_artifact_sha256",
    "expert_manifest_id",
    "expert_manifest_sha256",
    "source_git_commit",
    "kv_quantization",
    "kv_block_size_tokens",
    "total_context_tokens",
)

ISSUE46_RESOURCE_INTEGER_FIELDS = (
    "operating_target_bytes",
    "hard_ceiling_bytes",
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
ISSUE46_RESOURCE_FIELDS = (
    *ISSUE46_RESOURCE_INTEGER_FIELDS,
    "system_swap_delta_bytes",
    "kv_representation",
    "failed_closed",
    "failure_reason",
)

_T = TypeVar("_T")


class BenchmarkGateError(RuntimeError):
    """Raised when evidence is incomplete, ambiguous, or physically invalid."""


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
        expected_charged = classified_bytes + int(
            resource_evidence["allocator_cache_charged_bytes"]
        )
        if int(resource_evidence["charged_bytes"]) != expected_charged:
            raise BenchmarkGateError(
                f"{prefix}.charged_bytes does not match the six-pool additive ledger"
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
    if any(point.slot_health.values()):
        raise BenchmarkGateError(
            f"{point.phase} slot health is not quiescent and failure-free"
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
    if not (growth.monotonic_ns < holds[0].monotonic_ns < reset.monotonic_ns):
        raise BenchmarkGateError(
            "stable hold samples must follow growth and precede reset"
        )
    stable_fields = (
        "allocator_active_bytes",
        "allocator_cache_bytes",
        "expert_slab_physical_bytes",
        "kv_physical_bytes",
        "kv_allocated_blocks",
        "slot_health_sha256",
    )
    baseline = tuple(getattr(holds[0], field) for field in stable_fields)
    if any(
        tuple(getattr(point, field) for field in stable_fields) != baseline
        for point in holds
    ):
        raise BenchmarkGateError(
            "stable hold samples changed physical memory or slot health"
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

    reclaim = _phase_once(timeline, "post_expert_reclaim")
    regrow = _phase_once(timeline, "post_regrow")
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
    minimum_final_blocks = math.ceil(context_tokens / HY3_Q4_KV_BLOCK_SIZE_TOKENS)
    if growth.kv_allocated_blocks < minimum_final_blocks:
        raise BenchmarkGateError("post_kv_growth does not cover the requested context")
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

    route_trace = value["route_trace"]
    _sequence(route_trace, field="observation.route_trace")
    route_hash = _sha256(
        value["route_trace_sha256"], field="observation.route_trace_sha256"
    )
    if canonical_sha256(route_trace) != route_hash:
        raise BenchmarkGateError(
            "observation.route_trace_sha256 does not match exact routes"
        )

    raw_expert_hashes = _mapping(
        value["expert_hashes"], field="observation.expert_hashes"
    )
    if not raw_expert_hashes:
        raise BenchmarkGateError("observation.expert_hashes must not be empty")
    expert_hashes = {
        _nonempty_string(key, field="expert hash key"): _sha256(
            item, field=f"observation.expert_hashes[{key!r}]"
        )
        for key, item in raw_expert_hashes.items()
    }

    raw_timeline = _sequence(value["timeline"], field="observation.timeline")
    timeline = tuple(
        MemoryTimelinePoint.from_mapping(
            _mapping(item, field=f"observation.timeline[{index}]"),
            index=index,
        )
        for index, item in enumerate(raw_timeline)
    )
    _validate_timeline(arm, context_tokens, cache_start_state, timeline)

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
            "observation.metrics.hold_performance_samples are not stable within 10%"
        )
    raw_performance_samples = _sequence(
        raw_metrics["performance_samples"],
        field="observation.metrics.performance_samples",
    )
    if len(raw_performance_samples) != len(hold_performance):
        raise BenchmarkGateError(
            "observation.metrics.performance_samples must match hold_performance_samples"
        )
    performance_samples: list[dict[str, object]] = []
    for index, raw_sample in enumerate(raw_performance_samples):
        sample = _mapping(
            raw_sample,
            field=f"observation.metrics.performance_samples[{index}]",
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
        )
        _require_fields(
            sample,
            required_sample_fields,
            context=f"observation.metrics.performance_samples[{index}]",
        )
        sample_prefix = f"observation.metrics.performance_samples[{index}]"
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
        if not math.isclose(
            sample_tps,
            hold_performance[index],
            rel_tol=1e-9,
            abs_tol=1e-12,
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
        sample_route_trace = list(
            _sequence(
                sample["route_trace"],
                field=f"{sample_prefix}.route_trace",
            )
        )
        if not sample_route_trace:
            raise BenchmarkGateError(f"{sample_prefix}.route_trace must not be empty")
        sample_route_hash = _sha256(
            sample["route_trace_sha256"],
            field=f"{sample_prefix}.route_trace_sha256",
        )
        if canonical_sha256(sample_route_trace) != sample_route_hash:
            raise BenchmarkGateError(
                f"{sample_prefix}.route_trace_sha256 does not match exact routes"
            )
        performance_samples.append(
            {
                "tokens_per_second": sample_tps,
                "expert_hit_rate": hit_rate,
                "ssd_bytes_per_token": ssd_bytes,
                "p50_token_latency_ms": p50_ms,
                "p95_token_latency_ms": p95_ms,
                "generated_token_ids": sample_tokens,
                "generated_token_sha256": sample_token_hash,
                "route_trace": sample_route_trace,
                "route_trace_sha256": sample_route_hash,
            }
        )
    metrics: dict[str, object] = {
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "peak_charged_bytes": peak_charged,
        "stress_peak_charged_bytes": stress_peak_charged,
        "hold_performance_samples": list(hold_performance),
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

    Restoration is attempted after both unload and workload failures.  The lane
    is released only after restoration verification has run.
    """

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
            with _blocked_termination_signals():
                if captured:
                    hooks.restore(state)
                    if hooks.verify_restored(state) is not True:
                        raise BenchmarkGateError(
                            "Qwen state does not exactly match its captured state"
                        )
                if lane_acquired:
                    hooks.release_lane()


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


def run_json_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    input_payload: object | None = None,
    env: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Run an argv-only command and parse exactly one JSON object from stdout."""

    argv = tuple(_nonempty_string(item, field="command argv") for item in command)
    if not argv:
        raise BenchmarkGateError("JSON subprocess command must not be empty")
    encoded_input = (
        None
        if input_payload is None
        else json.dumps(input_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=None if env is None else {**os.environ, **env},
        input=encoded_input,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkGateError(
            f"JSON subprocess failed ({completed.returncode}): {detail}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkGateError(
            "JSON subprocess stdout is not one JSON value"
        ) from exc
    return _mapping(value, field="JSON subprocess stdout")


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
    arm_command_template: Sequence[str],
    repetitions: int,
    cwd: Path | str | None = None,
    command_runner: Callable[[Sequence[str]], Mapping[str, object]] | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 46,
) -> dict[str, object]:
    """Bridge JSON-emitting hardware commands into the pure campaign runner."""

    runner = command_runner or (lambda command: run_json_subprocess(command, cwd=cwd))
    probe = runner(probe_command)

    def execute(arm: str, context_tokens: int, repetition: int) -> Mapping[str, object]:
        return runner(
            format_arm_command(
                arm_command_template,
                arm=arm,
                context_tokens=context_tokens,
                repetition=repetition,
            )
        )

    return run_balanced_campaign(
        allocator_probe=probe,
        execute_arm=execute,
        repetitions=repetitions,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


__all__ = [
    "CONTEXT_MATRIX_TOKENS",
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
