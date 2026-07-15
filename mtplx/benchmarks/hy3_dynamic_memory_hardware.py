"""Production Hy3 Q4 hardware hooks for the issue #46 arm producer.

This adapter intentionally uses the real streamed runtime, paged-Q4 cache,
physical broker, route tracer, verified expert manifest, and MLX allocator
telemetry. It does not simulate a hardware observation or derive physical
release from logical capacity.
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mtplx.cache_state import prepare_brokered_q4_cache_group
from mtplx.benchmarks.hy3_dynamic_memory_artifacts import (
    ArtifactAttestationError,
    ArtifactPins,
    Hy3ArtifactAttestation,
    attest_hy3_artifact,
)
from mtplx.benchmarks.hy3_dynamic_memory_observation import (
    ArmObservationError,
    ArmRequest,
    HOLD_WARMUP_SAMPLES,
)
from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    HY3_Q4_KV_BLOCK_BYTES,
    HY3_Q4_KV_BLOCK_SIZE_TOKENS,
    HY3_Q4_KV_BYTES_PER_TOKEN,
    HY3_Q4_MAX_BLOCKS,
)
from mtplx.memory_broker import (
    HY3_Q4_ALLOCATOR_HEADROOM_BYTES,
    hy3_q4_kv_physical_geometry,
)
from mtplx.runtime_options import HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV


GIB = 1024**3
TOTAL_CONTEXT_TOKENS = 131_072
_SLOT_HEALTH_FIELDS = (
    "active_routes",
    "pins",
    "loading",
    "failed",
    "integrity_errors",
    "completion_fence_failures",
    "global_device_synchronizations",
)


class _TaskVmInfo(ctypes.Structure):
    """Public Darwin ``task_vm_info_data_t`` through revision 7."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("device_peak", ctypes.c_uint64),
        ("internal", ctypes.c_uint64),
        ("internal_peak", ctypes.c_uint64),
        ("external", ctypes.c_uint64),
        ("external_peak", ctypes.c_uint64),
        ("reusable", ctypes.c_uint64),
        ("reusable_peak", ctypes.c_uint64),
        ("purgeable_volatile_pmap", ctypes.c_uint64),
        ("purgeable_volatile_resident", ctypes.c_uint64),
        ("purgeable_volatile_virtual", ctypes.c_uint64),
        ("compressed", ctypes.c_uint64),
        ("compressed_peak", ctypes.c_uint64),
        ("compressed_lifetime", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("min_address", ctypes.c_uint64),
        ("max_address", ctypes.c_uint64),
        ("ledger_phys_footprint_peak", ctypes.c_int64),
        ("ledger_purgeable_nonvolatile", ctypes.c_int64),
        ("ledger_purgeable_novolatile_compressed", ctypes.c_int64),
        ("ledger_purgeable_volatile", ctypes.c_int64),
        ("ledger_purgeable_volatile_compressed", ctypes.c_int64),
        ("ledger_tag_network_nonvolatile", ctypes.c_int64),
        ("ledger_tag_network_nonvolatile_compressed", ctypes.c_int64),
        ("ledger_tag_network_volatile", ctypes.c_int64),
        ("ledger_tag_network_volatile_compressed", ctypes.c_int64),
        ("ledger_tag_media_footprint", ctypes.c_int64),
        ("ledger_tag_media_footprint_compressed", ctypes.c_int64),
        ("ledger_tag_media_nofootprint", ctypes.c_int64),
        ("ledger_tag_media_nofootprint_compressed", ctypes.c_int64),
        ("ledger_tag_graphics_footprint", ctypes.c_int64),
        ("ledger_tag_graphics_footprint_compressed", ctypes.c_int64),
        ("ledger_tag_graphics_nofootprint", ctypes.c_int64),
        ("ledger_tag_graphics_nofootprint_compressed", ctypes.c_int64),
        ("ledger_tag_neural_footprint", ctypes.c_int64),
        ("ledger_tag_neural_footprint_compressed", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint_compressed", ctypes.c_int64),
        ("limit_bytes_remaining", ctypes.c_uint64),
        ("decompressions", ctypes.c_int32),
        ("ledger_swapins", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint_total", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint_peak", ctypes.c_int64),
    ]


class _XswUsage(ctypes.Structure):
    """Public Darwin ``struct xsw_usage`` returned by ``vm.swapusage``."""

    _fields_ = [
        ("total", ctypes.c_uint64),
        ("available", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
        ("page_size", ctypes.c_uint32),
        ("encrypted", ctypes.c_int32),
    ]


def _darwin_libsystem() -> ctypes.CDLL:
    if sys.platform != "darwin":
        raise ArmObservationError("exact Mach memory telemetry requires macOS")
    try:
        return ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    except OSError as exc:
        raise ArmObservationError("Darwin kernel telemetry is unavailable") from exc


def _process_mach_memory_bytes() -> tuple[int, int]:
    libsystem = _darwin_libsystem()
    task_info = libsystem.task_info
    task_info.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint32),
    )
    task_info.restype = ctypes.c_int32
    try:
        task = ctypes.c_uint32.in_dll(libsystem, "mach_task_self_").value
    except ValueError as exc:
        raise ArmObservationError("mach_task_self_ is unavailable") from exc
    info = _TaskVmInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    result = task_info(
        task,
        22,  # TASK_VM_INFO
        ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(count),
    )
    required_count = (
        _TaskVmInfo.phys_footprint.offset + ctypes.sizeof(ctypes.c_uint64)
    ) // ctypes.sizeof(ctypes.c_uint32)
    if result != 0 or count.value < required_count:
        raise ArmObservationError(
            f"task_info TASK_VM_INFO failed ({result}, count={count.value})"
        )
    return int(info.resident_size), int(info.compressed)


def _system_swap_used_bytes() -> int:
    libsystem = _darwin_libsystem()
    sysctlbyname = libsystem.sysctlbyname
    sysctlbyname.argtypes = (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    sysctlbyname.restype = ctypes.c_int32
    usage = _XswUsage()
    size = ctypes.c_size_t(ctypes.sizeof(usage))
    result = sysctlbyname(
        b"vm.swapusage",
        ctypes.byref(usage),
        ctypes.byref(size),
        None,
        0,
    )
    if result != 0 or size.value < ctypes.sizeof(usage):
        raise ArmObservationError(
            f"sysctlbyname vm.swapusage failed ({result}, size={size.value})"
        )
    return int(usage.used)


def _host_memory_health_snapshot(
    *,
    initial_swap_bytes: int | None = None,
) -> dict[str, int]:
    """Return exact process memory and system-swap delta for one arm."""

    rss, compressed = _process_mach_memory_bytes()
    current_swap = _system_swap_used_bytes()
    baseline = current_swap if initial_swap_bytes is None else int(initial_swap_bytes)
    return {
        "process_rss_bytes": int(rss),
        "process_compressed_bytes": int(compressed),
        "system_swap_delta_bytes": current_swap - baseline,
    }


_GPU_UTILIZATION_PATTERN = re.compile(
    r'"Device Utilization %"\s*=\s*([0-9]+(?:\.[0-9]+)?)'
)


def _parse_gpu_utilization_percent(output: str) -> float:
    values = [float(match) for match in _GPU_UTILIZATION_PATTERN.findall(output)]
    if not values or any(not 0.0 <= value <= 100.0 for value in values):
        raise ArmObservationError("AGX GPU utilization evidence is unavailable")
    return max(values)


def _gpu_utilization_percent() -> float:
    completed = subprocess.run(
        ("/usr/sbin/ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )
    if completed.returncode != 0:
        raise ArmObservationError(
            f"AGX GPU utilization sampling failed ({completed.returncode})"
        )
    return _parse_gpu_utilization_percent(completed.stdout)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, Mapping) else {}


def _first(
    sources: Sequence[Mapping[str, object]],
    *names: str,
    default: object = 0,
) -> object:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArmObservationError(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArmObservationError(f"{field} must be a nonempty string")
    return value


def _path(value: object, *, field: str, root: Path | None = None) -> Path:
    raw = Path(_string(value, field=field)).expanduser()
    if not raw.is_absolute() and root is not None:
        raw = root / raw
    return raw.resolve()


@dataclass(frozen=True)
class Hy3HardwareConfig:
    """Exact real-hardware configuration shared by both campaign arms."""

    repo_root: Path
    model_root: Path
    manifest: Path
    model_artifact_id: str
    artifact_pins: ArtifactPins
    memory_limit_bytes: int = 100 * GIB
    runtime_reserve_bytes: int = 8 * GIB
    allocator_headroom_bytes: int = GIB
    transient_slots: int = 32
    generated_tokens: int = 16
    hold_sample_count: int = 3
    hold_tokens: int = 8
    hold_warmup_tokens: int | None = None
    prompt_style: str = "coding-agent"
    prompt_tail: str = "State the earliest and latest marker exactly."
    prefill_chunk_size: int = 2048

    def __post_init__(self) -> None:
        if self.hold_warmup_tokens is None:
            object.__setattr__(self, "hold_warmup_tokens", self.hold_tokens)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Hy3HardwareConfig:
        required = {
            "repo_root",
            "model_root",
            "manifest",
            "model_artifact_id",
            "artifact_pins",
        }
        allowed = required | {
            "memory_limit_bytes",
            "runtime_reserve_bytes",
            "allocator_headroom_bytes",
            "transient_slots",
            "generated_tokens",
            "hold_sample_count",
            "hold_tokens",
            "hold_warmup_tokens",
            "prompt_style",
            "prompt_tail",
            "prefill_chunk_size",
        }
        missing = required - set(value)
        unknown = set(value) - allowed
        if missing:
            raise ArmObservationError(
                f"hardware hooks config is missing {sorted(missing)}"
            )
        if unknown:
            raise ArmObservationError(
                f"hardware hooks config has unknown keys {sorted(unknown)}"
            )
        root = _path(value["repo_root"], field="repo_root")
        try:
            artifact_pins = ArtifactPins.from_mapping(value["artifact_pins"])
        except ArtifactAttestationError as exc:
            raise ArmObservationError(str(exc)) from exc
        config = cls(
            repo_root=root,
            model_root=_path(value["model_root"], field="model_root", root=root),
            manifest=_path(value["manifest"], field="manifest", root=root),
            model_artifact_id=_string(
                value["model_artifact_id"], field="model_artifact_id"
            ),
            artifact_pins=artifact_pins,
            memory_limit_bytes=_exact_int(
                value.get("memory_limit_bytes", 100 * GIB),
                field="memory_limit_bytes",
                minimum=1,
            ),
            runtime_reserve_bytes=_exact_int(
                value.get("runtime_reserve_bytes", 8 * GIB),
                field="runtime_reserve_bytes",
            ),
            allocator_headroom_bytes=_exact_int(
                value.get("allocator_headroom_bytes", GIB),
                field="allocator_headroom_bytes",
            ),
            transient_slots=_exact_int(
                value.get("transient_slots", 32),
                field="transient_slots",
                minimum=1,
            ),
            generated_tokens=_exact_int(
                value.get("generated_tokens", 16),
                field="generated_tokens",
                minimum=1,
            ),
            hold_sample_count=_exact_int(
                value.get("hold_sample_count", 3),
                field="hold_sample_count",
                minimum=3,
            ),
            hold_tokens=_exact_int(
                value.get("hold_tokens", 8), field="hold_tokens", minimum=1
            ),
            hold_warmup_tokens=_exact_int(
                value.get("hold_warmup_tokens", value.get("hold_tokens", 8)),
                field="hold_warmup_tokens",
                minimum=1,
            ),
            prompt_style=_string(
                value.get("prompt_style", "coding-agent"), field="prompt_style"
            ),
            prompt_tail=_string(
                value.get(
                    "prompt_tail",
                    "State the earliest and latest marker exactly.",
                ),
                field="prompt_tail",
            ),
            prefill_chunk_size=_exact_int(
                value.get("prefill_chunk_size", 2048),
                field="prefill_chunk_size",
                minimum=1,
            ),
        )
        if config.allocator_headroom_bytes != HY3_Q4_ALLOCATOR_HEADROOM_BYTES:
            raise ArmObservationError(
                "allocator_headroom_bytes must be exactly 1 GiB for issue #46"
            )
        if config.completion_reserve_tokens >= 4096:
            raise ArmObservationError(
                "generated plus warm-up and measured hold tokens must fit "
                "the smallest 4K context"
            )
        if config.prompt_style not in {"coding-agent", "legacy-repeat"}:
            raise ArmObservationError(
                "prompt_style must be coding-agent or legacy-repeat"
            )
        return config

    @property
    def completion_reserve_tokens(self) -> int:
        return (
            self.generated_tokens
            + (self.hold_sample_count * self.hold_tokens)
            + (HOLD_WARMUP_SAMPLES * self.hold_warmup_tokens)
        )


def _q4_cache_entries(cache: Sequence[object]) -> tuple[Any, ...]:
    entries = tuple(
        entry
        for entry in cache
        if bool(getattr(entry, "kv_quant", False))
        and str(getattr(getattr(entry, "kv_quant_config", None), "normalized_mode", ""))
        == "q4"
    )
    if not entries:
        raise ArmObservationError("retained cache has no paged Q4 entries")
    for index, entry in enumerate(entries):
        if int(getattr(entry, "block_size", 0)) != HY3_Q4_KV_BLOCK_SIZE_TOKENS:
            raise ArmObservationError(f"Q4 cache entry {index} has wrong block size")
        if (
            getattr(entry, "_shape", None) is None
            or getattr(entry, "_dtypes", None) is None
        ):
            raise ArmObservationError(
                f"Q4 cache entry {index} has no evaluated physical geometry"
            )
    return entries


def _entry_target_bytes(entry: Any, *, blocks: int) -> int:
    planned = getattr(entry, "_planned_capacity_bytes", None)
    if not callable(planned):
        raise ArmObservationError("Q4 cache entry cannot attest planned bytes")
    return int(
        planned(
            num_blocks=blocks,
            shape=entry._shape,
            dtypes=entry._dtypes,
        )
    )


class _CapturingGrowthObserver:
    """Capture the reclaim gap inside one aggregate cache transaction."""

    def __init__(
        self,
        delegate: Any,
        *,
        physical_ledger: Callable[[], dict[str, object]],
        monotonic_ns: Callable[[], int],
    ) -> None:
        self.delegate = delegate
        self.physical_ledger = physical_ledger
        self.monotonic_ns = monotonic_ns
        self.captured: dict[str, object] | None = None
        self.group_reservations = 0
        self.declared_steady_delta_bytes = 0
        self.declared_max_transient_delta_bytes = 0

    def reserve_growth(
        self,
        *,
        cache_id: str,
        steady_delta_bytes: int,
        transient_delta_bytes: int,
    ) -> Any:
        del cache_id, steady_delta_bytes, transient_delta_bytes
        raise ArmObservationError(
            "dynamic Q4 hardware preflight attempted single-entry growth"
        )

    def reserve_growth_group(
        self,
        *,
        members: tuple[tuple[str, int, int], ...],
    ) -> Any:
        if self.group_reservations:
            raise ArmObservationError(
                "dynamic Q4 hardware step opened multiple aggregate groups"
            )
        group = self.delegate.reserve_growth_group(members=members)
        self.group_reservations = 1
        self.declared_steady_delta_bytes = sum(int(member[1]) for member in members)
        self.declared_max_transient_delta_bytes = max(
            (int(member[2]) for member in members),
            default=0,
        )
        try:
            captured = dict(self.physical_ledger())
            captured["captured_monotonic_ns"] = _exact_int(
                self.monotonic_ns(),
                field="captured reclaim monotonic_ns",
            )
            self.captured = captured
        except BaseException as capture_error:
            try:
                group.abort(observed_uncommitted_physical_bytes=0)
            except BaseException as abort_error:
                raise abort_error from capture_error
            raise
        return group

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def preflight_and_grow_dynamic_q4(
    *,
    runtime: Any,
    cache: Sequence[object],
    context_tokens: int,
    physical_ledger: Callable[[], dict[str, object]],
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, object]:
    """Reclaim, then exercise a near-final and final Q4 growth boundary."""

    tokens = _exact_int(context_tokens, field="context_tokens", minimum=1)
    geometry = hy3_q4_kv_physical_geometry(tokens)
    target_blocks = geometry.physical_blocks
    if target_blocks < 3:
        raise ArmObservationError(
            "dynamic Q4 campaign growth must cross multiple block boundaries"
        )
    entries = _q4_cache_entries(cache)
    expert_runtime = getattr(runtime, "expert_streaming", None)
    if expert_runtime is None or getattr(expert_runtime, "memory_broker", None) is None:
        raise ArmObservationError("dynamic Q4 arm has no physical memory broker")
    original_observers = tuple(
        getattr(entry, "allocation_observer", None) for entry in entries
    )
    if any(observer is not expert_runtime for observer in original_observers):
        raise ArmObservationError(
            "dynamic Q4 entries do not share the runtime memory broker"
        )
    checkpoint_tokens = (target_blocks - 1) * HY3_Q4_KV_BLOCK_SIZE_TOKENS
    growth_steps: list[dict[str, object]] = []
    try:
        for sequence_index, requested_tokens in enumerate((checkpoint_tokens, tokens)):
            step_geometry = hy3_q4_kv_physical_geometry(requested_tokens)
            before = dict(physical_ledger())
            before_monotonic_ns = _exact_int(
                monotonic_ns(), field="KV growth before monotonic_ns"
            )
            target_bytes = [
                _entry_target_bytes(entry, blocks=step_geometry.physical_blocks)
                for entry in entries
            ]
            current_bytes = [int(entry.nbytes) for entry in entries]
            steady_delta = sum(
                target - current
                for target, current in zip(target_bytes, current_bytes, strict=True)
            )
            if steady_delta <= 0:
                raise ArmObservationError(
                    "dynamic Q4 campaign step did not require physical growth"
                )
            observer = _CapturingGrowthObserver(
                expert_runtime,
                physical_ledger=physical_ledger,
                monotonic_ns=monotonic_ns,
            )
            for entry in entries:
                entry.allocation_observer = observer
            prepared = prepare_brokered_q4_cache_group(
                list(cache),
                required_tokens=requested_tokens,
            )
            if (
                int(prepared.get("entries", 0)) != len(entries)
                or int(prepared.get("grown_entries", 0)) != len(entries)
                or int(prepared.get("target_blocks", 0))
                != step_geometry.physical_blocks
            ):
                raise ArmObservationError(
                    "aggregate Q4 cache preparation returned contradictory geometry"
                )
            reclaim_gap = observer.captured
            if reclaim_gap is None:
                raise ArmObservationError(
                    "dynamic Q4 growth did not expose a real reservation gap"
                )
            if observer.group_reservations != 1:
                raise ArmObservationError(
                    "Q4 growth did not use exactly one aggregate reservation"
                )
            if observer.declared_steady_delta_bytes != steady_delta:
                raise ArmObservationError(
                    "Q4 growth reservations did not cover total bytes"
                )
            if observer.declared_max_transient_delta_bytes != max(target_bytes):
                raise ArmObservationError(
                    "Q4 growth reservations did not cover the largest replacement"
                )
            if reclaim_gap["kv_physical_bytes"] != before["kv_physical_bytes"]:
                raise ArmObservationError("Q4 bytes changed inside a reservation gap")
            if int(reclaim_gap["expert_cache_physical_bytes"]) > int(
                before["expert_cache_physical_bytes"]
            ):
                raise ArmObservationError(
                    "expert records reallocated inside protected Q4 growth"
                )
            expert_release = int(before["expert_cache_physical_bytes"]) - int(
                reclaim_gap["expert_cache_physical_bytes"]
            )
            allocator_release = (
                int(before["allocator_active_bytes"])
                + int(before["allocator_cache_bytes"])
                - int(reclaim_gap["allocator_active_bytes"])
                - int(reclaim_gap["allocator_cache_bytes"])
            )
            if allocator_release < expert_release:
                raise ArmObservationError(
                    "expert registration fell without matching allocator release"
                )
            after = dict(physical_ledger())
            after_monotonic_ns = _exact_int(
                monotonic_ns(), field="KV growth after monotonic_ns"
            )
            reclaim_monotonic_ns = int(reclaim_gap["captured_monotonic_ns"])
            if not (before_monotonic_ns < reclaim_monotonic_ns < after_monotonic_ns):
                raise ArmObservationError(
                    "KV growth evidence timestamps are not strictly ordered"
                )
            if int(after["kv_allocated_blocks"]) != step_geometry.physical_blocks:
                raise ArmObservationError("dynamic Q4 cache missed a growth checkpoint")
            if int(after["kv_physical_bytes"]) != step_geometry.physical_bytes:
                raise ArmObservationError(
                    "dynamic Q4 checkpoint reached contradictory bytes"
                )
            growth_steps.append(
                {
                    "sequence_index": sequence_index,
                    "requested_tokens": requested_tokens,
                    "target_blocks": step_geometry.physical_blocks,
                    "before_monotonic_ns": before_monotonic_ns,
                    "reclaim_monotonic_ns": reclaim_monotonic_ns,
                    "after_monotonic_ns": after_monotonic_ns,
                    "before": before,
                    "reclaim_gap": dict(reclaim_gap),
                    "after": after,
                    "steady_delta_bytes": steady_delta,
                    "max_transient_delta_bytes": (
                        observer.declared_max_transient_delta_bytes
                    ),
                    "reclaimed_expert_bytes": expert_release,
                    "kv_growth_bytes": int(after["kv_physical_bytes"])
                    - int(before["kv_physical_bytes"]),
                }
            )
    finally:
        for entry, original in zip(entries, original_observers, strict=True):
            entry.allocation_observer = original
    first = growth_steps[0]
    reclaimed = dict(_mapping(first["reclaim_gap"]))
    if int(first["reclaimed_expert_bytes"]) <= 0:
        raise ArmObservationError(
            "preflight reservation did not physically evict an expert record"
        )
    after = physical_ledger()
    if int(after["kv_allocated_blocks"]) != target_blocks:
        raise ArmObservationError("dynamic Q4 cache did not reach target blocks")
    if int(after["kv_physical_bytes"]) != geometry.physical_bytes:
        raise ArmObservationError("dynamic Q4 cache reached contradictory bytes")
    reclaimed["kv_growth_steps"] = growth_steps
    return reclaimed


def trigger_future_demand_record_rewarm(
    *,
    expert_runtime: Any,
    route_trace: Sequence[Mapping[str, object]],
    expert_physical_bytes: Callable[[], int],
) -> None:
    """Warm a released record only through real post-reset route demand."""

    route = next(
        (
            entry
            for entry in reversed(route_trace)
            if isinstance(entry.get("layer"), int)
            and isinstance(entry.get("expert_ids"), Sequence)
            and entry.get("expert_ids")
        ),
        None,
    )
    if route is None:
        raise ArmObservationError("no exact prior expert route can drive future demand")
    layer = int(route["layer"])
    expert_ids = tuple(int(value) for value in route["expert_ids"])
    before = int(expert_physical_bytes())
    ready = expert_runtime.ensure_route(layer, expert_ids, phase="decode")
    try:
        pass
    finally:
        ready.release(synchronize=False)
    after = int(expert_physical_bytes())
    if after <= before:
        raise ArmObservationError(
            "future route demand did not allocate an expert record"
        )


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    commit = completed.stdout.strip().lower()
    if (
        completed.returncode != 0
        or len(commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ArmObservationError("cannot resolve full source Git commit")
    return commit


class _EnvironmentLease:
    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = dict(values)
        self.previous: dict[str, str | None] = {}
        self.active = False

    def acquire(self) -> None:
        if self.active:
            raise ArmObservationError("hardware environment lease already active")
        self.previous = {name: os.environ.get(name) for name in self.values}
        os.environ.update(self.values)
        self.active = True

    def release(self) -> None:
        if not self.active:
            return
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.active = False


def _arm_environment(arm: str, config: Hy3HardwareConfig) -> dict[str, str]:
    return {
        **HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV,
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": (
            str(HY3_Q4_MAX_BLOCKS) if arm == "static" else "1"
        ),
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": "q4",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "0",
        "MTPLX_DYNAMIC_PAGED_KV": "0" if arm == "static" else "1",
        "MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS": "1",
        "MTPLX_DYNAMIC_PAGED_KV_TOKENS": "0",
        "MTPLX_DYNAMIC_PAGED_KV_PREVIOUS_HIGH_WATER": "0",
        "MTPLX_DYNAMIC_PAGED_KV_MARGIN": "0",
        "MTPLX_PREFILL_CHUNK_SIZE": str(config.prefill_chunk_size),
    }


def _initial_kv_admission_tokens(arm: str, context_tokens: int) -> int:
    """Keep dynamic pre-growth logical ownership inside its one-block cache."""

    if arm not in {"static", "dynamic"}:
        raise ArmObservationError("hardware arm must be static or dynamic")
    tokens = _exact_int(context_tokens, field="context_tokens", minimum=1)
    return tokens if arm == "static" else 1


def _build_runtime_config(config: Hy3HardwareConfig, *, arm: str) -> Any:
    from mtplx.expert_runtime import ExpertStreamingConfig

    return ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=config.memory_limit_bytes,
        max_live_kv_tokens=TOTAL_CONTEXT_TOKENS,
        kv_bytes_per_token_override=HY3_Q4_KV_BYTES_PER_TOKEN,
        runtime_reserve_bytes=config.runtime_reserve_bytes,
        allocator_headroom_bytes=config.allocator_headroom_bytes,
        transient_slots=config.transient_slots,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="direct-slots",
        dynamic_expert_cache=arm == "dynamic",
        verify_sidecar_hash_at_open=False,
        verify_record_hashes=True,
        resource_telemetry=True,
    )


def _static_post_load_memory_pools(
    runtime: Any,
    runtime_config: Any,
    mx: Any,
) -> dict[str, int]:
    """Freeze the static control's fixed pools at the post-load boundary."""

    from mtplx.expert_runtime import mlx_memory_telemetry

    expert_runtime = runtime.expert_streaming
    plan = runtime_config.memory_plan(expert_runtime.spec)
    allocator = mlx_memory_telemetry(mx)
    expert_bytes = int(
        expert_runtime.slots.persistent_cache_telemetry_snapshot()["physical_bytes"]
    )
    transient_bytes = int(plan.transient_bytes)
    staging_bytes = transient_bytes + int(plan.io_staging_bytes)
    execution_workspace = int(plan.execution_workspace_bytes)
    resident_bytes = max(
        int(plan.resident_bytes),
        int(allocator["active_memory_bytes"]) - expert_bytes - transient_bytes,
    )
    resident_growth = resident_bytes - int(plan.resident_bytes)
    if resident_growth > int(plan.runtime_reserve_bytes):
        raise ArmObservationError(
            "static post-load resident growth exceeded the planned runtime reserve"
        )
    return {
        "resident_model_bytes": resident_bytes,
        "in_flight_expert_staging_bytes": staging_bytes,
        "runtime_workspace_bytes": (
            execution_workspace + int(plan.runtime_reserve_bytes) - resident_growth
        ),
    }


def _attest_config_artifact(
    config: Hy3HardwareConfig,
    *,
    verify_payload_hash: bool = False,
) -> Hy3ArtifactAttestation:
    try:
        return attest_hy3_artifact(
            model_root=config.model_root,
            manifest_path=config.manifest,
            pins=config.artifact_pins,
            verify_payload_hash=verify_payload_hash,
        )
    except ArtifactAttestationError as exc:
        raise ArmObservationError(str(exc)) from exc


def _require_loaded_manifest_identity(expected: Any, observed: Any) -> None:
    if observed is not expected:
        raise ArmObservationError("runtime-loaded manifest differs from attestation")


def _build_prompt_ids(
    tokenizer: Any,
    *,
    request: ArmRequest,
    config: Hy3HardwareConfig,
) -> list[int]:
    from mtplx.prefill_bench import _prompt_build_for_context

    prompt_tokens = request.context_tokens - config.completion_reserve_tokens
    build = _prompt_build_for_context(
        tokenizer,
        prompt_tokens,
        prompt_style=config.prompt_style,
        prompt_tail=config.prompt_tail,
        prompt_format="raw",
        enable_thinking=False,
    )
    result = [int(token) for token in build.token_ids]
    if len(result) != prompt_tokens:
        raise ArmObservationError(
            "prompt builder did not produce the exact token count"
        )
    return result


def _forward_prefill(
    runtime: Any,
    cache: Sequence[object],
    token_ids: Sequence[int],
    *,
    chunk_size: int,
) -> Any:
    import mlx.core as mx

    from mtplx.attention_context import attention_phase
    from mtplx.generation import _eval_cache_roots, _prefill_cache_only_forward

    if not token_ids:
        raise ArmObservationError("prefill token sequence must not be empty")
    body = list(token_ids[:-1])
    for start in range(0, len(body), chunk_size):
        chunk = mx.array([body[start : start + chunk_size]])
        with attention_phase("prefill"):
            result = _prefill_cache_only_forward(runtime, chunk, cache)
        if result is None:
            _eval_cache_roots(cache)
        else:
            mx.eval(result)
    with attention_phase("prefill"):
        logits = runtime.forward_ar(
            mx.array([[int(token_ids[-1])]]),
            cache=cache,
            emit_logits=True,
            logits_keep=1,
        )
    mx.eval(logits)
    return logits[:, -1, :]


def _decode_tokens(
    runtime: Any,
    cache: Sequence[object],
    logits: Any,
    *,
    count: int,
) -> tuple[list[int], Any, list[float], float]:
    import mlx.core as mx

    from mtplx.attention_context import attention_phase

    tokens: list[int] = []
    latencies_ms: list[float] = []
    started_batch = time.perf_counter()
    current = logits
    for _index in range(count):
        started = time.perf_counter()
        token = int(mx.argmax(current, axis=-1).item())
        tokens.append(token)
        with attention_phase("ar_decode"):
            current = runtime.forward_ar(
                mx.array([[token]]),
                cache=cache,
                emit_logits=True,
                logits_keep=1,
            )
        mx.eval(current)
        current = current[:, -1, :]
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    elapsed = time.perf_counter() - started_batch
    if elapsed <= 0.0:
        raise ArmObservationError("decode timer did not advance")
    return tokens, current, latencies_ms, elapsed


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ArmObservationError("latency sample must not be empty")
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


class MlxHy3HardwareLane:
    """One retained real-MLX Hy3 Q4 arm."""

    def __init__(
        self,
        *,
        config: Hy3HardwareConfig,
        request: ArmRequest,
        runtime: Any,
        runtime_config: Any,
        manifest: Any,
        prompt_ids: list[int],
        cache: Sequence[object],
        logits: Any,
        admission: Any,
        environment: _EnvironmentLease,
        model_artifact_sha256: str,
        manifest_file_sha256: str,
        source_git_commit: str,
        artifact_pins_sha256: str,
        artifact_stat_sha256: str,
        initial_artifact_evidence: Mapping[str, object],
        initial_system_swap_bytes: int | None = None,
        fixed_memory_pools: Mapping[str, int] | None = None,
    ) -> None:
        self.config = config
        self.request = request
        self.runtime = runtime
        self.runtime_config = runtime_config
        self.manifest = manifest
        self.prompt_ids = prompt_ids
        self.cache: Sequence[object] | None = cache
        self.logits = logits
        self.admission = admission
        self.environment = environment
        self.model_artifact_sha256 = model_artifact_sha256
        self.manifest_file_sha256 = manifest_file_sha256
        self.artifact_pins_sha256 = artifact_pins_sha256
        self.artifact_stat_sha256 = artifact_stat_sha256
        self.initial_artifact_evidence = dict(initial_artifact_evidence)
        self.source_git_commit = source_git_commit
        self.initial_system_swap_bytes = initial_system_swap_bytes
        self.fixed_memory_pools = dict(fixed_memory_pools or {})
        self._reclaim_ledger: dict[str, object] | None = None
        self._return_reclaim_ledger = False
        self._invocation_route_trace: list[dict[str, object]] = []
        self._admission_released = False
        self._closed = False

    @classmethod
    def open(
        cls,
        config: Hy3HardwareConfig,
        request: ArmRequest,
    ) -> MlxHy3HardwareLane:
        environment = _EnvironmentLease(_arm_environment(request.arm, config))
        environment.acquire()
        runtime = None
        admission = None
        cache = None
        try:
            initial_system_swap_bytes = _system_swap_used_bytes()
            import mlx.core as mx

            from mtplx.attention_context import attention_phase
            from mtplx.runtime import load

            artifact_attestation = _attest_config_artifact(config)
            manifest = artifact_attestation.manifest
            runtime_config = _build_runtime_config(config, arm=request.arm)
            runtime = load(
                config.model_root,
                mtp=False,
                expert_streaming_config=runtime_config,
                expert_manifest=manifest,
                model_config=artifact_attestation.model_config,
            )
            _require_loaded_manifest_identity(
                manifest,
                runtime.expert_streaming.manifest,
            )
            post_load_attestation = _attest_config_artifact(config)
            if post_load_attestation.evidence() != artifact_attestation.evidence():
                raise ArmObservationError("artifact changed during runtime load")
            fixed_memory_pools = (
                _static_post_load_memory_pools(runtime, runtime_config, mx)
                if request.arm == "static"
                else None
            )
            # Route tracing is evidence-only and does not change memory policy.
            runtime.expert_streaming.config = replace(runtime_config, trace_routes=True)
            prompt_ids = _build_prompt_ids(
                runtime.tokenizer,
                request=request,
                config=config,
            )
            admission = runtime.admit_kv_tokens(
                _initial_kv_admission_tokens(request.arm, request.context_tokens)
            )
            admission.__enter__()
            cache = runtime.make_cache()
            with attention_phase("prefill"):
                logits = runtime.forward_ar(
                    mx.array([[prompt_ids[0]]]),
                    cache=cache,
                    emit_logits=True,
                    logits_keep=1,
                )
            mx.eval(logits)
            logits = logits[:, -1, :]
            lane = cls(
                config=config,
                request=request,
                runtime=runtime,
                runtime_config=runtime_config,
                manifest=manifest,
                prompt_ids=prompt_ids,
                cache=cache,
                logits=logits,
                admission=admission,
                environment=environment,
                model_artifact_sha256=(artifact_attestation.model_artifact_sha256),
                manifest_file_sha256=(artifact_attestation.manifest_file_sha256),
                source_git_commit=_source_commit(config.repo_root),
                artifact_pins_sha256=(artifact_attestation.artifact_pins_sha256),
                artifact_stat_sha256=(artifact_attestation.artifact_stat_sha256),
                initial_artifact_evidence=artifact_attestation.evidence(),
                initial_system_swap_bytes=initial_system_swap_bytes,
                fixed_memory_pools=fixed_memory_pools,
            )
            initial = lane._live_physical_ledger()
            expected_blocks = HY3_Q4_MAX_BLOCKS if request.arm == "static" else 1
            if initial["kv_allocated_blocks"] != expected_blocks:
                raise ArmObservationError(
                    f"{request.arm} arm started with unexpected Q4 blocks"
                )
            return lane
        except BaseException:
            if cache is not None:
                try:
                    from mtplx.cache_state import close_physical_kv_cache

                    close_physical_kv_cache(cache)
                except BaseException:
                    pass
            if admission is not None:
                try:
                    admission.release()
                except BaseException:
                    pass
            if runtime is not None:
                try:
                    runtime.close(timeout=10.0)
                except BaseException:
                    pass
            environment.release()
            raise

    def identity(self) -> Mapping[str, object]:
        plan = self.runtime_config.memory_plan(self.runtime.expert_streaming.spec)
        arm_config = {
            "dynamic_memory": self.request.arm == "dynamic",
            "attention_runtime_env": dict(HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV),
            "expert_streaming_config": self.runtime_config.to_dict(),
            "planned_persistent_slots": int(plan.persistent_slots),
        }
        return {
            "model_key": "hy3-q4",
            "model_artifact_id": self.config.model_artifact_id,
            "model_artifact_sha256": self.model_artifact_sha256,
            "expert_manifest_id": self.config.manifest.name,
            "expert_manifest_sha256": self.manifest_file_sha256,
            "artifact_pins_sha256": self.artifact_pins_sha256,
            "artifact_stat_sha256": self.artifact_stat_sha256,
            "resident_payload_bytes": self.config.artifact_pins.resident_payload_bytes,
            "resident_payload_sha256": (
                self.config.artifact_pins.resident_payload_sha256
            ),
            "source_git_commit": self.source_git_commit,
            "arm_config": arm_config,
            "kv_quantization": "q4",
            "kv_block_size_tokens": HY3_Q4_KV_BLOCK_SIZE_TOKENS,
            "total_context_tokens": TOTAL_CONTEXT_TOKENS,
        }

    def _expert_physical_bytes(self) -> int:
        snapshot = (
            self.runtime.expert_streaming.slots.persistent_cache_telemetry_snapshot()
        )
        return int(snapshot["physical_bytes"])

    def _live_physical_ledger(self) -> dict[str, object]:
        import mlx.core as mx

        from mtplx.expert_runtime import mlx_memory_telemetry

        expert_runtime = self.runtime.expert_streaming
        expert_runtime._raise_if_unhealthy()
        slot_pool = expert_runtime.slots
        cache_details = _mapping(slot_pool.persistent_cache_telemetry_snapshot())
        health = _mapping(slot_pool.health_telemetry_snapshot())
        io_health = _mapping(health.get("io"))
        if "integrity_errors" not in io_health:
            raise ArmObservationError(
                "expert reader telemetry omitted integrity_errors"
            )
        metrics = _mapping(health.get("metrics"))
        states = _mapping(health.get("states"))
        if self.cache is None:
            blocks = 0
            kv_bytes = 0
        else:
            entries = _q4_cache_entries(self.cache)
            block_counts = {int(entry.num_blocks) for entry in entries}
            if len(block_counts) != 1:
                raise ArmObservationError("Q4 cache entries disagree on block count")
            blocks = block_counts.pop()
            kv_bytes = sum(int(entry.nbytes) for entry in entries)
        expected_kv_bytes = blocks * HY3_Q4_KV_BLOCK_BYTES
        if kv_bytes != expected_kv_bytes:
            raise ArmObservationError(
                f"Q4 physical ledger has {kv_bytes} bytes for {blocks} blocks"
            )
        expert_bytes = int(cache_details["physical_bytes"])
        broker = getattr(expert_runtime, "memory_broker", None)
        broker_snapshot = None
        if broker is None:
            if self.request.arm == "dynamic":
                raise ArmObservationError("dynamic arm lost the unified-memory broker")
        else:
            broker_snapshot = broker.snapshot()
            if int(broker_snapshot.kv_physical_bytes) != kv_bytes:
                raise ArmObservationError("broker KV bytes differ from retained cache")
            if int(broker_snapshot.expert_cache_physical_bytes) != expert_bytes:
                raise ArmObservationError(
                    "broker expert bytes differ from direct-record registry"
                )
        allocator = mlx_memory_telemetry(mx)
        sampler = getattr(self.runtime, "expert_resource_telemetry_snapshot", None)
        resource = _mapping(sampler()) if callable(sampler) else {}
        python_control_cpu = _mapping(resource.get("python_control_cpu"))
        if not python_control_cpu:
            raise ArmObservationError("runtime omitted Python control CPU telemetry")
        dynamic = _mapping(resource.get("dynamic_memory"))
        kv_resource = _mapping(resource.get("kv"))
        resource_broker = _mapping(resource.get("memory_broker"))
        broker_values = _mapping(broker_snapshot) or resource_broker
        fixed_values = _mapping(self.fixed_memory_pools)
        plan = None
        memory_plan = getattr(self.runtime_config, "memory_plan", None)
        spec = getattr(expert_runtime, "spec", None)
        if callable(memory_plan) and spec is not None:
            plan = memory_plan(spec)
        plan_values = _mapping(plan)
        active_bytes = int(allocator["active_memory_bytes"])
        allocator_cache_bytes = int(allocator["cache_memory_bytes"])
        resident_model_bytes = int(
            _first(
                (broker_values, fixed_values, plan_values),
                "resident_model_bytes",
                "resident_bytes",
            )
        )
        runtime_workspace_bytes = int(
            _first(
                (broker_values, fixed_values, plan_values),
                "runtime_workspace_bytes",
                "runtime_reserve_bytes",
            )
        )
        inflight_staging_bytes = int(
            _first(
                (broker_values, fixed_values),
                "in_flight_expert_staging_bytes",
                default=(
                    int(plan_values.get("transient_bytes", 0))
                    + int(plan_values.get("io_staging_bytes", 0))
                ),
            )
        )
        classified_bytes = (
            resident_model_bytes
            + kv_bytes
            + expert_bytes
            + inflight_staging_bytes
            + runtime_workspace_bytes
        )
        observed_allocator_residual = max(
            allocator_cache_bytes,
            active_bytes + allocator_cache_bytes - classified_bytes,
            0,
        )
        if broker_snapshot is None:
            charged_allocator_cache_bytes = observed_allocator_residual
            charged_bytes = classified_bytes + charged_allocator_cache_bytes
        else:
            charged_allocator_cache_bytes = int(
                _first((broker_values,), "allocator_cache_bytes")
            )
            if charged_allocator_cache_bytes < observed_allocator_residual:
                raise ArmObservationError(
                    "broker allocator residual is below raw MLX physical evidence"
                )
            charged_bytes = int(getattr(broker_snapshot, "charged_bytes"))
            if charged_bytes != classified_bytes + charged_allocator_cache_bytes:
                raise ArmObservationError(
                    "broker charged bytes differ from the six-pool additive ledger"
                )
        budget = getattr(broker, "budget", None)
        memory_limit_bytes = int(
            getattr(budget, "memory_limit_bytes", self.config.memory_limit_bytes)
        )
        allocator_headroom_bytes = int(
            getattr(
                budget,
                "allocator_headroom_bytes",
                self.config.allocator_headroom_bytes,
            )
        )
        classified_limit_bytes = int(
            getattr(
                budget,
                "classified_limit_bytes",
                memory_limit_bytes - allocator_headroom_bytes,
            )
        )
        host = _host_memory_health_snapshot(
            initial_swap_bytes=self.initial_system_swap_bytes,
        )
        return {
            "allocator_active_bytes": active_bytes,
            "allocator_cache_bytes": allocator_cache_bytes,
            "allocator_peak_bytes": int(allocator["peak_memory_bytes"]),
            "expert_cache_physical_bytes": expert_bytes,
            "kv_physical_bytes": kv_bytes,
            "kv_allocated_blocks": blocks,
            "memory_limit_bytes": memory_limit_bytes,
            "allocator_headroom_bytes": allocator_headroom_bytes,
            "classified_limit_bytes": classified_limit_bytes,
            "classified_bytes": classified_bytes,
            "charged_bytes": charged_bytes,
            "charged_residual_bytes": memory_limit_bytes - charged_bytes,
            "resident_model_bytes": resident_model_bytes,
            "kv_representation": str(
                _first((kv_resource,), "representation", default="q4")
            ),
            "kv_logical_tokens": int(
                _first(
                    (kv_resource, resource),
                    "logical_tokens",
                    "live_kv_tokens",
                    default=blocks * HY3_Q4_KV_BLOCK_SIZE_TOKENS,
                )
            ),
            "expert_logical_records": int(
                _first(
                    (dynamic, cache_details),
                    "logical_expert_records",
                    "logical_record_capacity",
                    default=int(plan_values.get("persistent_slots", 0)),
                )
            ),
            "expert_allocated_records": int(
                _first((cache_details,), "allocated_record_count")
            ),
            "expert_active_records": int(
                _first(
                    (dynamic, cache_details),
                    "active_expert_records",
                    "allocated_record_count",
                )
            ),
            "expert_resident_records": int(
                _first(
                    (dynamic, cache_details),
                    "resident_expert_records",
                    "resident_record_count",
                )
            ),
            "record_allocations": int(_first((dynamic,), "record_allocations")),
            "record_reuses": int(_first((dynamic,), "record_reuses")),
            "record_evictions": int(_first((dynamic,), "record_evictions")),
            "record_releases": int(_first((dynamic,), "record_releases")),
            "pinned_expert_bytes": int(
                _first(
                    (broker_values, dynamic),
                    "pinned_expert_bytes",
                    default=(
                        int(cache_details.get("pinned_record_count", 0))
                        * int(getattr(spec, "expert_record_bytes", 0))
                    ),
                )
            ),
            "inflight_expert_bytes": int(
                _first(
                    (dynamic,),
                    "in_flight_expert_bytes",
                    default=(
                        int(cache_details.get("in_flight_record_count", 0))
                        * int(getattr(spec, "expert_record_bytes", 0))
                    ),
                )
            ),
            "speculative_expert_bytes": int(
                _first((broker_values, dynamic), "speculative_expert_bytes")
            ),
            "runtime_workspace_bytes": runtime_workspace_bytes,
            "inflight_expert_staging_bytes": inflight_staging_bytes,
            "admission_failures": int(
                _first(
                    (broker_values, dynamic),
                    "admission_failures",
                    "admission_failure_count",
                )
            ),
            "allocator_cache_charged_bytes": charged_allocator_cache_bytes,
            **host,
            "gpu_utilization_percent": _gpu_utilization_percent(),
            "python_control_cpu": dict(python_control_cpu),
            "failed_closed": bool(
                _first((broker_values, dynamic), "failed_closed", default=False)
            ),
            "failure_reason": _first(
                (broker_values, dynamic),
                "failure_reason",
                default=None,
            ),
            "slot_health": {
                "active_routes": int(metrics["active_routes"]),
                "pins": int(health["pins"]),
                "loading": int(states["loading"]),
                "failed": int(states["failed"]),
                "integrity_errors": int(io_health["integrity_errors"]),
                "completion_fence_failures": int(metrics["completion_fence_failures"]),
                "global_device_synchronizations": int(
                    metrics["global_device_synchronizations"]
                ),
            },
        }

    def physical_ledger(self) -> Mapping[str, object]:
        if self._return_reclaim_ledger:
            if self._reclaim_ledger is None:
                raise ArmObservationError("captured reclaim ledger is missing")
            self._return_reclaim_ledger = False
            return dict(self._reclaim_ledger)
        return self._live_physical_ledger()

    def reclaim_experts_for_q4(self, context_tokens: int) -> None:
        if self.request.arm != "dynamic" or self.cache is None:
            raise ArmObservationError("expert reclaim applies only to live dynamic arm")
        self._reclaim_ledger = preflight_and_grow_dynamic_q4(
            runtime=self.runtime,
            cache=self.cache,
            context_tokens=context_tokens,
            physical_ledger=self._live_physical_ledger,
        )
        self._return_reclaim_ledger = True

    def prepare_q4_context(self, context_tokens: int) -> None:
        if context_tokens != self.request.context_tokens or self.cache is None:
            raise ArmObservationError("Q4 preparation request identity drifted")
        entries = _q4_cache_entries(self.cache)
        expected_blocks = (
            HY3_Q4_MAX_BLOCKS
            if self.request.arm == "static"
            else hy3_q4_kv_physical_geometry(context_tokens).physical_blocks
        )
        if any(int(entry.num_blocks) != expected_blocks for entry in entries):
            raise ArmObservationError("Q4 cache capacity was not prepared exactly")
        if self.request.arm == "dynamic":
            self.admission.grow_to(context_tokens)
        self.logits = _forward_prefill(
            self.runtime,
            self.cache,
            self.prompt_ids[1:],
            chunk_size=self.config.prefill_chunk_size,
        )

    def kv_growth_steps(self) -> Sequence[Mapping[str, object]]:
        if self.request.arm == "static":
            return ()
        if self._reclaim_ledger is None:
            raise ArmObservationError("dynamic Q4 growth evidence is missing")
        raw_steps = self._reclaim_ledger.get("kv_growth_steps")
        if isinstance(raw_steps, (str, bytes, bytearray)) or not isinstance(
            raw_steps, Sequence
        ):
            raise ArmObservationError("dynamic Q4 growth evidence is malformed")
        return tuple(dict(_mapping(step)) for step in raw_steps)

    def _expert_hashes(
        self,
        route_trace: Sequence[Mapping[str, object]],
    ) -> dict[str, str]:
        pairs = {
            (int(entry["layer"]), int(expert))
            for entry in route_trace
            if isinstance(entry.get("layer"), int)
            and isinstance(entry.get("expert_ids"), Sequence)
            for expert in entry["expert_ids"]
        }
        result: dict[str, str] = {}
        for layer, expert in sorted(pairs):
            record = self.manifest.record(layer, expert)
            if record.sha256 is None:
                raise ArmObservationError(
                    f"expert record ({layer}, {expert}) has no verified SHA-256"
                )
            result[f"{layer}:{expert}"] = str(record.sha256)
        if not result:
            raise ArmObservationError("generation produced no exact expert routes")
        return result

    def invoke_context(self, context_tokens: int) -> Mapping[str, object]:
        if context_tokens != self.request.context_tokens or self.cache is None:
            raise ArmObservationError("invocation request identity drifted")
        tokens, self.logits, _latencies, elapsed = _decode_tokens(
            self.runtime,
            self.cache,
            self.logits,
            count=self.config.generated_tokens,
        )
        trace = [dict(entry) for entry in self.runtime.expert_streaming.route_trace()]
        self._invocation_route_trace = trace
        return {
            "prompt_token_ids": list(self.prompt_ids),
            "generated_token_ids": tokens,
            "route_trace": trace,
            "expert_hashes": self._expert_hashes(trace),
            "elapsed_seconds": elapsed,
        }

    def sample_hold_performance(self) -> Mapping[str, object]:
        return self._sample_hold_performance(count=self.config.hold_tokens)

    def _sample_hold_performance(self, *, count: int) -> Mapping[str, object]:
        if self.cache is None:
            raise ArmObservationError("hold sample has no retained Q4 cache")
        before = self.runtime.expert_streaming.snapshot()["cache"]
        trace_before = len(self.runtime.expert_streaming.route_trace())
        tokens, self.logits, latencies, elapsed = _decode_tokens(
            self.runtime,
            self.cache,
            self.logits,
            count=count,
        )
        after = self.runtime.expert_streaming.snapshot()["cache"]
        hits = int(after["expert_hits"]) - int(before["expert_hits"])
        misses = int(after["expert_misses"]) - int(before["expert_misses"])
        requests = hits + misses
        bytes_read = int(after["bytes_read"]) - int(before["bytes_read"])
        if requests <= 0 or bytes_read < 0:
            raise ArmObservationError("hold sample has contradictory expert counters")
        route_trace = [
            dict(entry)
            for entry in self.runtime.expert_streaming.route_trace()[trace_before:]
        ]
        if not route_trace:
            raise ArmObservationError("hold sample did not retain an exact route trace")
        return {
            "tokens_per_second": len(tokens) / elapsed,
            "expert_hit_rate": hits / requests,
            "ssd_bytes_per_token": bytes_read / len(tokens),
            "p50_token_latency_ms": statistics.median(latencies),
            "p95_token_latency_ms": _percentile(latencies, 0.95),
            "generated_token_ids": tokens,
            "route_trace": route_trace,
            "expert_hashes": self._expert_hashes(route_trace),
        }

    def stabilize_hold(self) -> Mapping[str, object]:
        # The first hold exercises the complete measurement path and lets MLX
        # settle allocator-cache bookkeeping before the three gated samples.
        # Its tokens are included in completion_reserve_tokens and its physical
        # state remains visible as the hold_warmup timeline point.
        return self._sample_hold_performance(count=self.config.hold_warmup_tokens)

    def reset_q4_context(self) -> None:
        if self.cache is None:
            raise ArmObservationError("Q4 cache was already reset")
        from mtplx.cache_state import close_physical_kv_cache

        close_physical_kv_cache(self.cache)
        self.cache = None
        self._release_admission()

    def _release_admission(self) -> None:
        if self._admission_released:
            return
        self.admission.release()
        self._admission_released = True

    def trigger_future_expert_demand(self) -> None:
        if self.request.arm != "dynamic":
            raise ArmObservationError(
                "future-demand record rewarm applies only to dynamic arm"
            )
        trigger_future_demand_record_rewarm(
            expert_runtime=self.runtime.expert_streaming,
            route_trace=self._invocation_route_trace,
            expert_physical_bytes=self._expert_physical_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        if self.cache is not None:
            try:
                from mtplx.cache_state import close_physical_kv_cache

                close_physical_kv_cache(self.cache)
                self.cache = None
            except BaseException as exc:
                first_error = exc
        try:
            self._release_admission()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            self.runtime.close(timeout=10.0)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            final_attestation = _attest_config_artifact(self.config)
            if final_attestation.evidence() != self.initial_artifact_evidence:
                raise ArmObservationError("artifact changed during hardware arm")
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self.environment.release()
        self._closed = True
        if first_error is not None:
            raise first_error


class ProductionHy3HardwareHooks:
    """Tracked factory result loaded by the JSON-only arm command."""

    def __init__(self, config: Hy3HardwareConfig) -> None:
        self.config = config
        self.hold_samples = config.hold_sample_count

    def load_static_lane(self, request: ArmRequest) -> MlxHy3HardwareLane:
        if request.arm != "static":
            raise ArmObservationError("static loader received a non-static request")
        return MlxHy3HardwareLane.open(self.config, request)

    def load_dynamic_lane(self, request: ArmRequest) -> MlxHy3HardwareLane:
        if request.arm != "dynamic":
            raise ArmObservationError("dynamic loader received a non-dynamic request")
        return MlxHy3HardwareLane.open(self.config, request)


def create_hooks(config: Mapping[str, object]) -> ProductionHy3HardwareHooks:
    """Tracked ``module:factory`` entrypoint for hardware campaign specs."""

    return ProductionHy3HardwareHooks(Hy3HardwareConfig.from_mapping(config))


__all__ = [
    "Hy3HardwareConfig",
    "MlxHy3HardwareLane",
    "ProductionHy3HardwareHooks",
    "create_hooks",
    "preflight_and_grow_dynamic_q4",
    "trigger_future_demand_record_rewarm",
]
