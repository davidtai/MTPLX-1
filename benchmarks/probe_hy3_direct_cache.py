#!/usr/bin/env python3
"""Prove Hy3 direct expert records allocate lazily, replace, and release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mtplx.benchmarks.runners.hy3_dynamic_memory import (  # noqa: E402
    HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV,
    AllocatorSample,
    canonical_sha256,
    normalize_arm_config,
    run_direct_cache_probe,
)
from mtplx.benchmarks.hy3_dynamic_memory_artifacts import (  # noqa: E402
    ArtifactAttestationError,
    attest_hy3_artifact,
)
from mtplx.benchmarks.hy3_dynamic_memory_hardware import (  # noqa: E402
    Hy3HardwareConfig,
    _build_runtime_config,
)
from mtplx.expert_manifest import (  # noqa: E402
    ExpertManifestError,
    verify_expert_manifest,
)


TOTAL_CONTEXT_TOKENS = 131_072


def _buffer_identity(buffer: object) -> int:
    """Track record identity without keeping the MLX buffer alive."""

    return id(buffer)


def _attest_probe_artifact(config: Hy3HardwareConfig):
    """Full-hash the sidecar once, then verify every source-shard header."""

    try:
        attestation = attest_hy3_artifact(
            model_root=config.model_root,
            manifest_path=config.manifest,
            pins=config.artifact_pins,
            verify_payload_hash=True,
            require_f_nocache=True,
        )
        report = verify_expert_manifest(
            attestation.manifest,
            config.model_root,
            verify_sidecar_hash=False,
        )
    except (ArtifactAttestationError, ExpertManifestError) as exc:
        raise RuntimeError(str(exc)) from exc
    if report.get("valid") is not True:
        raise RuntimeError("expert shard-header verification did not pass")
    try:
        final_attestation = attest_hy3_artifact(
            model_root=config.model_root,
            manifest_path=config.manifest,
            pins=config.artifact_pins,
            verify_payload_hash=False,
        )
    except ArtifactAttestationError as exc:
        raise RuntimeError(str(exc)) from exc
    binding_fields = (
        "model_artifact_sha256",
        "manifest_file_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "resident_payload_bytes",
        "resident_payload_sha256",
    )
    if any(
        getattr(final_attestation, field) != getattr(attestation, field)
        for field in binding_fields
    ):
        raise RuntimeError("artifact changed during shard-header verification")
    return attestation


def load_sidecar_record(
    root: Path,
    manifest: Any,
    record: Any,
    buffer: Any,
) -> None:
    """Copy one verified sidecar record into a contiguous direct slot."""

    sidecar = getattr(manifest, "sidecar", None)
    offset = getattr(record, "sidecar_offset", None)
    length = getattr(record, "sidecar_length", None)
    if sidecar is None or offset is None or length is None:
        raise RuntimeError("probe record has no sidecar ownership range")
    if int(length) != int(record.logical_bytes):
        raise RuntimeError("probe sidecar range differs from logical record bytes")
    expected_sha256 = getattr(record, "sha256", None)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError("probe record has no valid payload hash")
    view = memoryview(buffer)
    if view.readonly or not view.c_contiguous or view.nbytes != int(length):
        view.release()
        raise RuntimeError("probe direct record buffer is not writable and contiguous")
    descriptor = os.open(root / sidecar.file, os.O_RDONLY)
    try:
        payload = os.pread(descriptor, int(length), int(offset))
        if len(payload) != int(length):
            raise RuntimeError(
                f"short sidecar read: expected {int(length)}, got {len(payload)}"
            )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise RuntimeError("probe sidecar record hash mismatch")
        view[:] = payload
    finally:
        os.close(descriptor)
        view.release()


def build_probe_identity(
    *,
    model_artifact_id: str,
    model_artifact_sha256: str,
    expert_manifest_id: str,
    expert_manifest_sha256: str,
    artifact_pins_sha256: str,
    artifact_stat_sha256: str,
    resident_payload_bytes: int,
    resident_payload_sha256: str,
    source_git_commit: str,
    arm_config: Mapping[str, object],
) -> dict[str, object]:
    exact_arm = dict(arm_config)
    normalized = normalize_arm_config(exact_arm)
    return {
        "model_key": "hy3-q4",
        "model_artifact_id": model_artifact_id,
        "model_artifact_sha256": model_artifact_sha256,
        "expert_manifest_id": expert_manifest_id,
        "expert_manifest_sha256": expert_manifest_sha256,
        "artifact_pins_sha256": artifact_pins_sha256,
        "artifact_stat_sha256": artifact_stat_sha256,
        "resident_payload_bytes": resident_payload_bytes,
        "resident_payload_sha256": resident_payload_sha256,
        "source_git_commit": source_git_commit,
        "arm_config": exact_arm,
        "arm_config_sha256": canonical_sha256(exact_arm),
        "normalized_config": normalized,
        "normalized_config_sha256": canonical_sha256(normalized),
        "kv_quantization": "q4",
        "kv_block_size_tokens": 16,
        "total_context_tokens": TOTAL_CONTEXT_TOKENS,
    }


def _probe_arm_config(runtime_config: Any, plan: Any) -> dict[str, object]:
    return {
        "dynamic_memory": True,
        "attention_runtime_env": dict(HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV),
        "expert_streaming_config": runtime_config.to_dict(),
        "planned_persistent_slots": int(plan.persistent_slots),
    }


def _require_clean_source() -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("allocator evidence requires a clean source worktree")
    return commit


def run_real_probe(
    *,
    config: Hy3HardwareConfig,
) -> dict[str, object]:
    import mlx.core as mx

    from mtplx.expert_io import PositionalExpertReader
    from mtplx.expert_runtime import mlx_memory_telemetry
    from mtplx.expert_slots import ExpertSlotBinding, ExpertSlotPool
    from mtplx.expert_streaming_models import HY3_Q4
    from mtplx.models.expert_mlx import (
        _run_q4_expert,
        make_mlx_slot_buffer_allocator,
    )

    source_commit = _require_clean_source()
    attestation = _attest_probe_artifact(config)
    manifest = attestation.manifest
    runtime_config = _build_runtime_config(config, arm="dynamic")
    plan = runtime_config.memory_plan(HY3_Q4)
    arm_config = _probe_arm_config(runtime_config, plan)
    identity = build_probe_identity(
        model_artifact_id=config.model_artifact_id,
        model_artifact_sha256=attestation.model_artifact_sha256,
        expert_manifest_id=config.manifest.name,
        expert_manifest_sha256=attestation.manifest_file_sha256,
        artifact_pins_sha256=attestation.artifact_pins_sha256,
        artifact_stat_sha256=attestation.artifact_stat_sha256,
        resident_payload_bytes=attestation.resident_payload_bytes,
        resident_payload_sha256=attestation.resident_payload_sha256,
        source_git_commit=source_commit,
        arm_config=arm_config,
    )
    allocator = make_mlx_slot_buffer_allocator(plan, HY3_Q4)
    reader = PositionalExpertReader(
        config.model_root,
        max_open_files=runtime_config.max_open_files,
        max_read_chunk_bytes=runtime_config.max_read_chunk_bytes,
        bypass_page_cache=runtime_config.bypass_page_cache,
    )
    pool = ExpertSlotPool(
        HY3_Q4,
        plan,
        manifest,
        reader,
        buffer_allocator=allocator,
        max_inflight_io_bytes=runtime_config.max_inflight_io_bytes,
        prefer_sidecar=runtime_config.prefer_sidecar,
        verify_hashes=True,
        cache_scope="global",
        lazy_persistent_buffers=True,
    )
    records = tuple(
        record
        for record in manifest.records
        if record.layer == HY3_Q4.routed_layer_start and record.expert in {0, 1}
    )
    if len(records) != 2:
        raise RuntimeError("probe requires experts 0 and 1 in the first routed layer")
    first_record, replacement_record = sorted(records, key=lambda item: item.expert)
    label = "global-persistent-0"
    original_buffer_identity: int | None = None

    def sample_allocator() -> AllocatorSample:
        report = mlx_memory_telemetry(mx)
        try:
            return AllocatorSample(
                active_bytes=int(report["active_memory_bytes"]),
                cache_bytes=int(report["cache_memory_bytes"]),
                peak_bytes=int(report["peak_memory_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("MLX allocator telemetry is incomplete") from exc

    def allocate_first_record() -> int:
        nonlocal original_buffer_identity
        allocated_bytes = pool.allocate_persistent_slot(0)
        buffer = allocator.slots[label]
        original_buffer_identity = _buffer_identity(buffer)
        load_sidecar_record(config.model_root, manifest, first_record, buffer)
        return allocated_bytes

    def execute(record: Any) -> bool:
        buffer = allocator.slots.get(label)
        if buffer is None:
            return False
        binding = ExpertSlotBinding(
            layer=int(record.layer),
            expert=int(record.expert),
            logical_slot=0,
            generation=1,
            record=record,
            buffer=buffer,
        )
        values = mx.ones((1, HY3_Q4.hidden_size), dtype=mx.bfloat16)
        output = _run_q4_expert(
            values,
            binding,
            group_size=HY3_Q4.quant_group_size,
        )
        finite = mx.all(mx.isfinite(output))
        mx.eval(output, finite)
        return tuple(output.shape) == (1, HY3_Q4.hidden_size) and bool(finite.item())

    def replace_record() -> bool:
        buffer = allocator.slots.get(label)
        if buffer is None:
            return False
        load_sidecar_record(config.model_root, manifest, replacement_record, buffer)
        return (
            original_buffer_identity is not None
            and _buffer_identity(buffer) == original_buffer_identity
        )

    def release_record() -> int:
        released = pool.release_persistent_slots((0,))
        return int(released.physical_bytes)

    try:
        startup_bytes = int(
            pool.persistent_cache_telemetry_snapshot()["physical_bytes"]
        )
        return run_direct_cache_probe(
            identity=identity,
            backend=str(allocator.backend),
            startup_persistent_bytes=startup_bytes,
            sample_allocator=sample_allocator,
            allocate_first_record=allocate_first_record,
            execute_first_record=lambda: execute(first_record),
            replace_record=replace_record,
            execute_replacement_record=lambda: execute(replacement_record),
            release_record=release_record,
        )
    finally:
        pool.close()


def _strict_config_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"hooks config has duplicate key {key!r}")
        result[key] = value
    return result


def _load_hardware_config(path: Path) -> Hy3HardwareConfig:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_config_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load hardware hooks config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("hardware hooks config must be an object")
    return Hy3HardwareConfig.from_mapping(value)


def verify_artifact_only(config: Hy3HardwareConfig) -> dict[str, object]:
    _require_clean_source()
    return _attest_probe_artifact(config).evidence()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hooks-config", type=Path, required=True)
    parser.add_argument("--verify-artifact-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_hardware_config(args.hooks_config.expanduser().resolve())
    result = (
        verify_artifact_only(config)
        if args.verify_artifact_only
        else run_real_probe(config=config)
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.verify_artifact_only:
        return 0 if result.get("payload_hash_verified") is True else 1
    return 0 if result.get("gate_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
