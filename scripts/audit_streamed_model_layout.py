#!/usr/bin/env python3
"""Audit streamed model parameters against a Hub or compact local artifact."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from mtplx.expert_manifest import (  # noqa: E402
    ExpertManifestError,
    MAX_SAFETENSORS_HEADER_BYTES,
    load_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import MODEL_SPECS, get_model_spec  # noqa: E402
from mtplx.hy3_mtp_patch import (  # noqa: E402
    HY3_MTP_BF16_FILE,
    HY3_MTP_RESIDENTS_FILE,
    expected_bf16_names,
    expected_resident_names,
)
from mtplx.resident_loader import (  # noqa: E402
    _quantize_resident_model,
    get_streaming_model_classes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download only config/index metadata, construct the parameter-free "
            "streamed model, and require an exact resident key match."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Audit a self-contained local compact artifact instead of the "
            "model spec's Hub quant_model."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Expert manifest for --artifact-root (default: expert-manifest.json).",
    )
    return parser


def _is_routed_tensor(name: str) -> bool:
    return ".switch_mlp." in name or ".mlp.experts." in name


def _manifest_path(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    preferred = root / "expert-manifest.json"
    if preferred.is_file():
        return preferred
    legacy = root / "expert-manifest-sidecar.json"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"no expert manifest found under {root}")


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.manifest is not None and args.artifact_root is None:
        raise SystemExit("--manifest requires --artifact-root")


def _safetensors_inventory(path: Path) -> tuple[set[str], int, dict[str, str]]:
    """Read names, tensor bytes, and string metadata without loading arrays."""

    try:
        with open(path, "rb") as handle:
            length_raw = handle.read(8)
            if len(length_raw) != 8:
                raise ValueError("truncated safetensors length")
            (header_length,) = struct.unpack("<Q", length_raw)
            if not 1 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("invalid safetensors header length")
            header_raw = handle.read(header_length)
            if len(header_raw) != header_length:
                raise ValueError("truncated safetensors header")
        header = json.loads(header_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not inspect {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} has malformed safetensors metadata")
    total_bytes = 0
    maximum_end = 0
    ranges: list[tuple[int, int, str]] = []
    for name, info in header.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            raise ValueError(f"{path} has a malformed tensor entry")
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise ValueError(f"{path} tensor {name} has invalid data offsets")
        start, end = offsets
        if start < 0 or end <= start:
            raise ValueError(f"{path} tensor {name} has invalid data range")
        total_bytes += end - start
        maximum_end = max(maximum_end, end)
        ranges.append((start, end, name))
    if not header:
        raise ValueError(f"{path} has no tensors")
    ranges.sort()
    expected_start = 0
    for start, end, name in ranges:
        if start != expected_start:
            raise ValueError(f"{path} tensor {name} leaves a gap or overlaps")
        expected_start = end
    data_bytes = path.stat().st_size - 8 - header_length
    if maximum_end != data_bytes:
        raise ValueError(f"{path} tensor ranges do not cover the file")
    return (
        set(header),
        total_bytes,
        {str(key): str(value) for key, value in metadata.items()},
    )


def _compact_manifest_audit(
    manifest: Any,
    spec: Any,
    index: dict[str, str],
) -> dict[str, Any]:
    expected_keys = {
        (layer, expert)
        for layer in spec.expert_record_layer_indices
        for expert in range(spec.expert_count)
    }
    actual_keys = {(record.layer, record.expert) for record in manifest.records}
    indexed_residents = {key for key in index if not _is_routed_tensor(key)}
    manifest_residents = {tensor.tensor for tensor in manifest.resident_tensors}
    indexed_resident_shards = {key: str(index[key]) for key in indexed_residents}
    manifest_resident_shards = {
        tensor.tensor: tensor.shard for tensor in manifest.resident_tensors
    }
    resident_shard_mapping_matches = indexed_resident_shards == manifest_resident_shards
    sidecar_file = None if manifest.sidecar is None else manifest.sidecar.file
    sidecar_authoritative = sidecar_file is not None and all(
        record.sidecar_offset is not None
        and record.sidecar_length == record.logical_bytes
        and all(segment.shard == sidecar_file for segment in record.segments)
        for record in manifest.records
    )
    resident_shard_payload_bytes = sum(
        shard.size - shard.header_bytes
        for shard in manifest.shards
        if shard.kind == "safetensors"
    )
    resident_shards_are_compact = (
        resident_shard_payload_bytes == manifest.resident_tensor_bytes
    )
    valid = (
        actual_keys == expected_keys
        and manifest.model_key == spec.key
        and manifest.source_repo == spec.manifest_repo
        and manifest.source_revision == spec.manifest_revision
        and manifest.quant_bits == spec.quant_bits
        and manifest.quant_group_size == spec.quant_group_size
        and manifest.quant_mode == "affine"
        and manifest.artifact_tensor_bytes == spec.total_tensor_bytes
        and manifest.resident_tensor_bytes == spec.resident_bytes
        and manifest.routed_expert_bytes == spec.routed_expert_bytes
        and manifest_residents == indexed_residents
        and resident_shard_mapping_matches
        and sidecar_authoritative
        and resident_shards_are_compact
    )
    return {
        "valid": valid,
        "record_keys": actual_keys,
        "expected_record_keys": expected_keys,
        "manifest_residents": manifest_residents,
        "indexed_residents": indexed_residents,
        "resident_shard_mapping_matches": resident_shard_mapping_matches,
        "sidecar_authoritative": sidecar_authoritative,
        "resident_shard_payload_bytes": resident_shard_payload_bytes,
        "resident_shards_are_compact": resident_shards_are_compact,
    }


def main() -> int:
    args = build_parser().parse_args()
    _validate_cli_args(args)
    spec = get_model_spec(args.model)
    artifact_root: Path | None = None
    if args.artifact_root is None:
        if spec.quant_model.startswith("local/"):
            raise SystemExit(
                f"{spec.key} has no published quant_model; pass --artifact-root"
            )
        config_path = Path(
            hf_hub_download(
                spec.quant_model,
                "config.json",
                revision=spec.quant_revision,
            )
        )
        index_path = Path(
            hf_hub_download(
                spec.quant_model,
                "model.safetensors.index.json",
                revision=spec.quant_revision,
            )
        )
    else:
        artifact_root = args.artifact_root.expanduser().resolve()
        config_path = artifact_root / "config.json"
        index_path = artifact_root / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise SystemExit(
                f"compact artifact is missing config/index under {artifact_root}"
            )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"]
    model_class, args_class = get_streaming_model_classes(config)
    model_args = args_class.from_dict(config)
    model = model_class(model_args)

    mtp_prefix = (
        None
        if spec.mtp_layer_index is None
        else f"model.layers.{spec.mtp_layer_index}."
    )
    mtp_resident = {
        key for key in index if mtp_prefix is not None and key.startswith(mtp_prefix)
    }
    routed_index = {key for key in index if _is_routed_tensor(key)}
    resident = {
        key: None
        for key in index
        if key not in routed_index and key not in mtp_resident
    }
    if hasattr(model, "sanitize"):
        resident = model.sanitize(resident)
    _quantize_resident_model(model, config, resident)
    parameter_keys = {key for key, _value in tree_flatten(model.parameters())}
    resident_keys = set(resident)
    missing = sorted(parameter_keys - resident_keys)
    extra = sorted(resident_keys - parameter_keys)
    routed = sorted(routed_index)
    expected_routed_tensors = spec.routed_layer_count * 9
    manifest_records = 0
    expected_manifest_records = 0
    manifest_layers: list[int] = []
    manifest_valid = True
    manifest_file_integrity = True
    manifest_file_error: str | None = None
    manifest_resident_keys = 0
    expected_manifest_resident_keys = 0
    missing_manifest_residents: list[str] = []
    extra_manifest_residents: list[str] = []
    sidecar_authoritative = False
    resident_shard_payload_bytes = 0
    resident_shards_are_compact = False
    resident_shard_mapping_matches = False
    mtp_artifacts_valid = True
    mtp_artifact_error: str | None = None
    mtp_q4_residents_shared_with_index = False
    mtp_bf16_tensor_bytes = 0
    mtp_q4_resident_tensor_bytes = 0
    mtp_missing: list[str] = []
    mtp_extra: list[str] = []
    if artifact_root is not None:
        manifest = load_expert_manifest(_manifest_path(artifact_root, args.manifest))
        manifest_audit = _compact_manifest_audit(manifest, spec, index)
        actual_keys = manifest_audit["record_keys"]
        expected_keys = manifest_audit["expected_record_keys"]
        actual_residents = manifest_audit["manifest_residents"]
        expected_residents = manifest_audit["indexed_residents"]
        manifest_records = len(actual_keys)
        expected_manifest_records = len(expected_keys)
        manifest_layers = sorted({layer for layer, _expert in actual_keys})
        manifest_resident_keys = len(actual_residents)
        expected_manifest_resident_keys = len(expected_residents)
        missing_manifest_residents = sorted(expected_residents - actual_residents)
        extra_manifest_residents = sorted(actual_residents - expected_residents)
        sidecar_authoritative = bool(manifest_audit["sidecar_authoritative"])
        resident_shard_payload_bytes = int(
            manifest_audit["resident_shard_payload_bytes"]
        )
        resident_shards_are_compact = bool(
            manifest_audit["resident_shards_are_compact"]
        )
        resident_shard_mapping_matches = bool(
            manifest_audit["resident_shard_mapping_matches"]
        )
        try:
            validate_expert_manifest_spec(manifest, spec)
            verify_expert_manifest(manifest, artifact_root)
        except ExpertManifestError as exc:
            manifest_file_integrity = False
            manifest_file_error = str(exc)
        manifest_valid = bool(manifest_audit["valid"]) and manifest_file_integrity
        if spec.mtp_included:
            expected_mtp = expected_resident_names(model_args)
            mtp_missing = sorted(expected_mtp - mtp_resident)
            mtp_extra = sorted(mtp_resident - expected_mtp)
            q4_path = artifact_root / HY3_MTP_RESIDENTS_FILE
            bf16_path = artifact_root / HY3_MTP_BF16_FILE
            try:
                q4_names, mtp_q4_resident_tensor_bytes, q4_metadata = (
                    _safetensors_inventory(q4_path)
                )
                bf16_names, mtp_bf16_tensor_bytes, bf16_metadata = (
                    _safetensors_inventory(bf16_path)
                )
                mapped_mtp_shards = {str(index[name]) for name in mtp_resident}
                if len(mapped_mtp_shards) == 1:
                    indexed_q4_path = artifact_root / mapped_mtp_shards.pop()
                    mtp_q4_residents_shared_with_index = q4_path.samefile(
                        indexed_q4_path
                    )
                mtp_artifacts_valid = (
                    q4_names == expected_mtp
                    and bf16_names == expected_bf16_names(model_args)
                    and mtp_q4_resident_tensor_bytes == spec.mtp_q4_resident_bytes
                    and mtp_bf16_tensor_bytes == spec.mtp_bf16_tensor_bytes
                    and q4_metadata.get("source_repo") == spec.manifest_repo
                    and q4_metadata.get("source_revision") == spec.manifest_revision
                    and bf16_metadata.get("source_repo") == spec.manifest_repo
                    and bf16_metadata.get("source_revision") == spec.manifest_revision
                    and mtp_q4_residents_shared_with_index
                )
            except (OSError, ValueError) as exc:
                mtp_artifacts_valid = False
                mtp_artifact_error = str(exc)
    report = {
        "model_key": spec.key,
        "quant_model": spec.quant_model,
        "quant_revision": spec.quant_revision,
        "parameter_keys": len(parameter_keys),
        "resident_index_keys": len(resident_keys),
        "routed_index_keys": len(routed),
        "expected_routed_index_keys": expected_routed_tensors,
        "routed_tensors_in_compact_index": sorted(routed),
        "manifest_records": manifest_records,
        "expected_manifest_records": expected_manifest_records,
        "manifest_layers": manifest_layers,
        "manifest_valid": manifest_valid,
        "manifest_file_integrity": manifest_file_integrity,
        "manifest_file_error": manifest_file_error,
        "manifest_resident_keys": manifest_resident_keys,
        "expected_manifest_resident_keys": expected_manifest_resident_keys,
        "missing_manifest_resident_keys": missing_manifest_residents,
        "extra_manifest_resident_keys": extra_manifest_residents,
        "sidecar_authoritative": sidecar_authoritative,
        "resident_shard_payload_bytes": resident_shard_payload_bytes,
        "resident_shards_are_compact": resident_shards_are_compact,
        "resident_shard_mapping_matches": resident_shard_mapping_matches,
        "mtp_artifacts_valid": mtp_artifacts_valid,
        "mtp_artifact_error": mtp_artifact_error,
        "mtp_q4_residents_shared_with_index": mtp_q4_residents_shared_with_index,
        "mtp_q4_resident_tensor_bytes": mtp_q4_resident_tensor_bytes,
        "mtp_bf16_tensor_bytes": mtp_bf16_tensor_bytes,
        "mtp_resident_index_keys": len(mtp_resident),
        "missing_mtp_resident_keys": mtp_missing,
        "extra_mtp_resident_keys": mtp_extra,
        "missing_parameter_keys": missing,
        "extra_resident_keys": extra,
        "valid": (
            not missing
            and not extra
            and not mtp_missing
            and not mtp_extra
            and manifest_valid
            and mtp_artifacts_valid
            and (
                (artifact_root is None and len(routed) == expected_routed_tensors)
                or (artifact_root is not None and not routed)
            )
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
