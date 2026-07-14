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
    run_allocator_release_probe,
)


GIB = 1024**3
TOTAL_CONTEXT_TOKENS = 131_072


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


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
    views = tuple(slot.record_views(record))
    if len(views) != len(record.segments):
        raise RuntimeError("probe component view count differs from record segments")
    descriptor = os.open(root / sidecar.file, os.O_RDONLY)
    try:
        cursor = 0
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
            view[:] = payload
            cursor += expected
        if cursor != int(length):
            raise RuntimeError("component segments do not cover the sidecar record")
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
    source_git_commit: str,
    arm_config: Mapping[str, object],
) -> dict[str, object]:
    exact_arm = dict(arm_config)
    normalized = dict(exact_arm)
    normalized.pop("dynamic_memory", None)
    return {
        "model_key": "hy3-q4",
        "model_artifact_id": model_artifact_id,
        "model_artifact_sha256": model_artifact_sha256,
        "expert_manifest_id": expert_manifest_id,
        "expert_manifest_sha256": expert_manifest_sha256,
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


def _model_artifact_sha256(
    root: Path,
    manifest_path: Path,
    manifest: Any,
) -> tuple[str, str]:
    manifest_sha256 = _sha256_file(manifest_path)
    sidecar = getattr(manifest, "sidecar", None)
    if sidecar is None:
        raise RuntimeError("Hy3 allocator probe requires the verified sidecar")
    artifact = {
        "config_sha256": _sha256_file(root / "config.json"),
        "expert_manifest_sha256": manifest_sha256,
        "expert_payload_sha256": str(sidecar.sha256),
        "expert_payload_bytes": int(sidecar.size),
        "source_revision": str(manifest.source_revision),
    }
    return canonical_sha256(artifact), manifest_sha256


def run_real_probe(
    *,
    model_root: Path,
    manifest_path: Path,
    slab_slots: int,
) -> dict[str, object]:
    import mlx.core as mx

    from mtplx.expert_manifest import load_expert_manifest
    from mtplx.expert_runtime import (
        ExpertStreamingConfig,
        mlx_memory_telemetry,
    )
    from mtplx.expert_slots import ExpertSlotBinding
    from mtplx.expert_streaming_models import HY3_Q4
    from mtplx.models.expert_mlx import (
        _run_component_bank_q4,
        make_mlx_component_bank_allocator,
    )

    source_commit = _require_clean_source()
    manifest = load_expert_manifest(manifest_path, verify_digest=True)
    if (
        manifest.model_key != "hy3-q4"
        or manifest.source_revision != HY3_Q4.quant_revision
    ):
        raise RuntimeError("probe manifest is not the pinned Hy3 Q4 artifact")
    artifact_sha256, manifest_sha256 = _model_artifact_sha256(
        model_root,
        manifest_path,
        manifest,
    )
    config = ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=110 * GIB,
        max_live_kv_tokens=TOTAL_CONTEXT_TOKENS,
        runtime_reserve_bytes=8 * GIB,
        transient_slots=32,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="component-banks",
        dynamic_expert_slabs=True,
        expert_slab_slots=slab_slots,
    )
    plan = config.memory_plan(HY3_Q4)
    arm_config = {
        "dynamic_memory": True,
        "expert_streaming_config": config.to_dict(),
        "planned_persistent_slots": int(plan.persistent_slots),
        "probe_slab_ids": [0, 1],
    }
    identity = build_probe_identity(
        model_artifact_id=(f"pipenetwork/Hy3-4bit@{HY3_Q4.quant_revision}"),
        model_artifact_sha256=artifact_sha256,
        expert_manifest_id=manifest_path.name,
        expert_manifest_sha256=manifest_sha256,
        source_git_commit=source_commit,
        arm_config=arm_config,
    )
    allocator = make_mlx_component_bank_allocator(
        plan,
        HY3_Q4,
        manifest,
        persistent_slab_slots=slab_slots,
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
        load_sidecar_record(model_root, manifest, record, untouched_slot)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--slab-slots", type=int, default=32)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.slab_slots <= 0:
        raise SystemExit("--slab-slots must be positive")
    result = run_real_probe(
        model_root=args.model_root.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        slab_slots=args.slab_slots,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result.get("gate_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
