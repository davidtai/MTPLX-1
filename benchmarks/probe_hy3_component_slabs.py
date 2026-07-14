#!/usr/bin/env python3
"""Prove that real Hy3 component slabs release charged MLX memory."""

from __future__ import annotations

import argparse
import gc
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
    AllocatorSample,
    ProbeSlab,
    canonical_sha256,
    normalize_arm_config,
    run_allocator_release_probe,
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
    slot: Any,
) -> None:
    """Copy one exact sidecar record into component-major slot views."""

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
    views = tuple(slot.record_views(record))
    if len(views) != len(record.segments):
        raise RuntimeError("probe component view count differs from record segments")
    descriptor = os.open(root / sidecar.file, os.O_RDONLY)
    try:
        cursor = 0
        digest = hashlib.sha256()
        payloads: list[bytes] = []
        for segment, view in zip(record.segments, views, strict=True):
            expected = int(segment.length)
            payload = os.pread(descriptor, expected, int(offset) + cursor)
            if len(payload) != expected:
                raise RuntimeError(
                    f"short sidecar read at component offset {cursor}: "
                    f"expected {expected}, got {len(payload)}"
                )
            if view.nbytes != expected:
                raise RuntimeError("component view length differs from manifest")
            digest.update(payload)
            payloads.append(payload)
            cursor += expected
        if cursor != int(length):
            raise RuntimeError("component segments do not cover the sidecar record")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("probe sidecar record hash mismatch")
        for view, payload in zip(views, payloads, strict=True):
            view[:] = payload
    finally:
        os.close(descriptor)
        for view in views:
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

    from mtplx.expert_runtime import mlx_memory_telemetry
    from mtplx.expert_slots import ExpertSlotBinding
    from mtplx.expert_streaming_models import HY3_Q4
    from mtplx.models.expert_mlx import (
        _run_component_bank_q4,
        make_mlx_component_bank_allocator,
    )

    source_commit = _require_clean_source()
    attestation = _attest_probe_artifact(config)
    manifest = attestation.manifest
    runtime_config = _build_runtime_config(config, arm="dynamic")
    plan = runtime_config.memory_plan(HY3_Q4)
    arm_config = {
        "dynamic_memory": True,
        "expert_streaming_config": runtime_config.to_dict(),
        "planned_persistent_slots": int(plan.persistent_slots),
        "probe_slab_ids": [0, 1],
    }
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
    allocator = make_mlx_component_bank_allocator(
        plan,
        HY3_Q4,
        manifest,
        persistent_slab_slots=config.expert_slab_slots,
    )
    layout = allocator.slab_layout()
    if 0 not in layout or 1 not in layout:
        raise RuntimeError("planned component-bank layout has fewer than two slabs")
    record = next(
        record
        for record in manifest.records
        if record.layer == HY3_Q4.routed_layer_start and record.expert == 0
    )
    untouched_slot: Any | None = None
    untouched_slot_id = int(layout[1][0])

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

    def allocate_slabs() -> Sequence[ProbeSlab]:
        nonlocal untouched_slot
        result = []
        for slab_id in (0, 1):
            allocated = allocator.allocate_slab(slab_id)
            if slab_id == 1:
                untouched_slot = allocated[untouched_slot_id]
            result.append(
                ProbeSlab(
                    slab_id=str(slab_id),
                    registered_physical_bytes=int(
                        allocator.slab_physical_bytes(slab_id)
                    ),
                )
            )
            del allocated
        return tuple(result)

    def evaluate_slabs(_slabs: Sequence[ProbeSlab]) -> None:
        if untouched_slot is None:
            raise RuntimeError("untouched probe slot was not allocated")
        load_sidecar_record(config.model_root, manifest, record, untouched_slot)
        arrays = [
            value for bank in allocator.banks.values() for value in bank.arrays.values()
        ]
        mx.eval(*arrays)

    def release_slab(slab: ProbeSlab) -> None:
        release = allocator.release_slab(int(slab.slab_id))
        if not release.destroyed or release.physical_bytes != (
            slab.registered_physical_bytes
        ):
            raise RuntimeError("component slab release receipt differs from manifest")
        gc.collect()

    def execute_slab(slab: ProbeSlab) -> bool:
        if slab.slab_id != "1" or untouched_slot is None:
            return False
        binding = ExpertSlotBinding(
            layer=int(record.layer),
            expert=int(record.expert),
            logical_slot=untouched_slot_id,
            generation=1,
            record=record,
            buffer=untouched_slot,
        )
        values = mx.ones((1, HY3_Q4.hidden_size), dtype=mx.bfloat16)
        output = _run_component_bank_q4(
            values,
            (binding,),
            group_size=HY3_Q4.quant_group_size,
        )
        finite = mx.all(mx.isfinite(output))
        mx.eval(output, finite)
        return tuple(output.shape) == (1, HY3_Q4.hidden_size) and bool(finite.item())

    try:
        return run_allocator_release_probe(
            identity=identity,
            allocate_slabs=allocate_slabs,
            evaluate_slabs=evaluate_slabs,
            sample_allocator=sample_allocator,
            release_slab=release_slab,
            execute_slab=execute_slab,
        )
    finally:
        allocator.close()


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
