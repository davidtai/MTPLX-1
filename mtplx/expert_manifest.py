"""Strict safetensors manifests for SSD-streamed MoE experts.

The manifest is the trust boundary between a revision-pinned checkpoint and
the native slot loader.  It maps every ``(layer, expert)`` to the nine affine
Q4 leaves used by gate/up/down projections without materializing tensors.

Only Python's standard library is used here so manifests can be inspected and
verified on machines without MLX.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from .expert_streaming_models import ExpertStreamingModelSpec, get_model_spec


MANIFEST_FORMAT = "mtplx-expert-manifest-v1"
DEFAULT_ALIGNMENT = 16 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024 * 1024
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_SHARDS = 4_096
MAX_MANIFEST_RESIDENT_TENSORS = 1_000_000
MAX_MANIFEST_RECORDS = 100_000
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "scales", "biases")
_SHARD_KINDS = ("safetensors", "sidecar")
_COMPONENTS = tuple(
    f"{projection}.{leaf}" for projection in _PROJECTIONS for leaf in _LEAVES
)
_LAYER_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.mlp\."
    r"(?P<container>switch_mlp|experts)\.(?P<tail>.+)$"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ExpertManifestError(ValueError):
    """Raised when artifact provenance or layout fails closed validation."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExpertManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, *, source: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_strict_pairs)
    except ExpertManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExpertManifestError(f"invalid JSON in {source}: {exc}") from exc


def _load_json_file(path: Path, *, max_bytes: int = MAX_MANIFEST_BYTES) -> Any:
    try:
        payload = _read_file_nofollow(path, max_bytes=max_bytes)
    except ExpertManifestError:
        raise
    except OSError as exc:
        raise ExpertManifestError(f"could not read {path}: {exc}") from exc
    return _load_json_bytes(payload, source=str(path))


def _expect_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExpertManifestError(f"{label} must be an object")
    return value


def _expect_keys(
    value: dict[str, Any],
    *,
    label: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(value)
    unknown = set(value) - allowed
    if missing:
        raise ExpertManifestError(f"{label} is missing keys: {sorted(missing)}")
    if unknown:
        raise ExpertManifestError(f"{label} has unknown keys: {sorted(unknown)}")


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpertManifestError(f"{label} must be an exact integer")
    if value < minimum:
        raise ExpertManifestError(f"{label} must be at least {minimum}")
    return value


def _string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ExpertManifestError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, *, label: str) -> str:
    digest = _string(value, label=label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ExpertManifestError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ExpertManifestError(f"{label} must be an integer array")
    return tuple(_integer(item, label=f"{label}[]", minimum=1) for item in value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _readonly_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _read_file_nofollow(
    path: Path,
    *,
    max_bytes: int,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> bytes:
    chunks: list[bytes] = []
    fd = os.open(path, _readonly_flags())
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExpertManifestError(f"JSON source is not a regular file: {path}")
        if metadata.st_size > max_bytes:
            raise ExpertManifestError(
                f"JSON source {path} exceeds the {max_bytes}-byte limit"
            )
        total = 0
        while True:
            try:
                chunk = os.read(fd, min(chunk_bytes, max_bytes + 1 - total))
            except InterruptedError:
                continue
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ExpertManifestError(
                    f"JSON source {path} exceeds the {max_bytes}-byte limit"
                )
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _hash_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    fd: int | None = None
    try:
        fd = os.open(path, _readonly_flags())
        while True:
            try:
                chunk = os.read(fd, chunk_bytes)
            except InterruptedError:
                continue
            if not chunk:
                break
            digest.update(chunk)
    except OSError as exc:
        raise ExpertManifestError(f"could not hash {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_relative_name(name: str, *, label: str) -> str:
    value = _string(name, label=label)
    if "\\" in value:
        raise ExpertManifestError(f"{label} must use POSIX separators")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ExpertManifestError(f"unsafe {label}: {value!r}")
    normalized = pure.as_posix()
    if normalized in {"", "."}:
        raise ExpertManifestError(f"unsafe {label}: {value!r}")
    return normalized


def _resolve_member(root: Path, name: str, *, must_exist: bool = True) -> Path:
    relative = _safe_relative_name(name, label="artifact path")
    base = root.resolve()
    candidate = base.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise ExpertManifestError(
            f"could not resolve artifact member {relative}: {exc}"
        ) from exc
    if resolved != base and base not in resolved.parents:
        # Hugging Face's standard cache stores snapshots as symlinks of the
        # form ``snapshots/<revision>/<file> -> ../../blobs/<digest>``.  Keep
        # the general no-escape rule, but recognize that exact repository-local
        # content-addressed layout so a pinned cache snapshot is directly
        # usable without copying hundreds of gigabytes.  The blobs directory
        # itself must be a real directory, and only an actual snapshot symlink
        # may cross this boundary.
        snapshots = base.parent
        repository = snapshots.parent
        blob_root = repository / "blobs"
        try:
            resolved_blob_root = blob_root.resolve(strict=True)
        except OSError:
            resolved_blob_root = None
        hugging_face_blob = (
            snapshots.name == "snapshots"
            and candidate.is_symlink()
            and blob_root.is_dir()
            and not blob_root.is_symlink()
            and resolved_blob_root is not None
            and resolved_blob_root in resolved.parents
        )
        if not hugging_face_blob:
            raise ExpertManifestError(f"artifact member escapes root: {relative}")
    if must_exist and not resolved.is_file():
        raise ExpertManifestError(f"artifact member is not a file: {relative}")
    return resolved


def resolve_artifact_member(
    root: Path | str,
    name: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a member inside its root or a repository-local HF cache blob."""

    return _resolve_member(Path(root), name, must_exist=must_exist)


@dataclass(frozen=True)
class ShardInfo:
    name: str
    size: int
    header_bytes: int
    header_sha256: str
    sha256: str | None = None
    kind: str = "safetensors"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "size": self.size,
            "header_bytes": self.header_bytes,
            "header_sha256": self.header_sha256,
        }
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.kind != "safetensors":
            result["kind"] = self.kind
        return result

    @classmethod
    def from_dict(cls, value: Any) -> ShardInfo:
        obj = _expect_object(value, label="shard")
        _expect_keys(
            obj,
            label="shard",
            required=("name", "size", "header_bytes", "header_sha256"),
            optional=("sha256", "kind"),
        )
        kind = obj.get("kind", "safetensors")
        if kind not in _SHARD_KINDS:
            raise ExpertManifestError(f"unsupported shard kind {kind!r}")
        digest = obj.get("sha256")
        header_minimum = 0 if kind == "sidecar" else 1
        header_bytes = _integer(
            obj["header_bytes"],
            label="shard header_bytes",
            minimum=header_minimum,
        )
        header_sha256 = _string(obj["header_sha256"], label="shard header_sha256")
        if kind == "sidecar" and (header_bytes != 0 or header_sha256 != EMPTY_SHA256):
            raise ExpertManifestError(
                "sidecar shard must have an empty zero-byte header"
            )
        return cls(
            name=_safe_relative_name(obj["name"], label="shard name"),
            size=_integer(obj["size"], label="shard size", minimum=1),
            header_bytes=header_bytes,
            header_sha256=header_sha256,
            sha256=None if digest is None else _string(digest, label="shard sha256"),
            kind=kind,
        )


@dataclass(frozen=True)
class TensorSegment:
    component: str
    tensor: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "tensor": self.tensor,
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Any) -> TensorSegment:
        obj = _expect_object(value, label="segment")
        _expect_keys(
            obj,
            label="segment",
            required=(
                "component",
                "tensor",
                "shard",
                "offset",
                "length",
                "dtype",
                "shape",
            ),
        )
        component = _string(obj["component"], label="segment component")
        if component not in _COMPONENTS:
            raise ExpertManifestError(f"unsupported expert component {component!r}")
        dtype = _string(obj["dtype"], label="segment dtype")
        if dtype not in _DTYPE_BYTES:
            raise ExpertManifestError(f"unsupported safetensors dtype {dtype!r}")
        return cls(
            component=component,
            tensor=_string(obj["tensor"], label="segment tensor"),
            shard=_safe_relative_name(obj["shard"], label="segment shard"),
            offset=_integer(obj["offset"], label="segment offset"),
            length=_integer(obj["length"], label="segment length", minimum=1),
            dtype=dtype,
            shape=_shape(obj["shape"], label="segment shape"),
        )


@dataclass(frozen=True)
class ResidentTensor:
    tensor: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ResidentTensor:
        obj = _expect_object(value, label="resident tensor")
        _expect_keys(
            obj,
            label="resident tensor",
            required=("tensor", "shard", "offset", "length", "dtype", "shape"),
        )
        dtype = _string(obj["dtype"], label="resident dtype")
        if dtype not in _DTYPE_BYTES:
            raise ExpertManifestError(f"unsupported safetensors dtype {dtype!r}")
        return cls(
            tensor=_string(obj["tensor"], label="resident tensor name"),
            shard=_safe_relative_name(obj["shard"], label="resident shard"),
            offset=_integer(obj["offset"], label="resident offset"),
            length=_integer(obj["length"], label="resident length", minimum=1),
            dtype=dtype,
            shape=_shape(obj["shape"], label="resident shape"),
        )


@dataclass(frozen=True)
class ExpertRecord:
    layer: int
    expert: int
    logical_bytes: int
    segments: tuple[TensorSegment, ...]
    sha256: str | None = None
    sidecar_offset: int | None = None
    sidecar_length: int | None = None
    # Sharded sidecar layout only: the sidecar shard file holding this
    # record.  ``sidecar_offset`` is then relative to that shard file.
    sidecar_shard: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "layer": self.layer,
            "expert": self.expert,
            "logical_bytes": self.logical_bytes,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.sidecar_offset is not None:
            result["sidecar_offset"] = self.sidecar_offset
            result["sidecar_length"] = self.sidecar_length
        if self.sidecar_shard is not None:
            result["sidecar_shard"] = self.sidecar_shard
        return result

    @classmethod
    def from_dict(cls, value: Any) -> ExpertRecord:
        obj = _expect_object(value, label="expert record")
        _expect_keys(
            obj,
            label="expert record",
            required=("layer", "expert", "logical_bytes", "segments"),
            optional=("sha256", "sidecar_offset", "sidecar_length", "sidecar_shard"),
        )
        raw_segments = obj["segments"]
        if not isinstance(raw_segments, list):
            raise ExpertManifestError("expert record segments must be an array")
        if len(raw_segments) != len(_COMPONENTS):
            raise ExpertManifestError(
                "expert record must contain the nine ordered quantized components"
            )
        segments = tuple(TensorSegment.from_dict(item) for item in raw_segments)
        if tuple(segment.component for segment in segments) != _COMPONENTS:
            raise ExpertManifestError(
                "expert record must contain the nine ordered quantized components"
            )
        digest = obj.get("sha256")
        sidecar_offset = obj.get("sidecar_offset")
        sidecar_length = obj.get("sidecar_length")
        if (sidecar_offset is None) != (sidecar_length is None):
            raise ExpertManifestError(
                "sidecar_offset and sidecar_length must appear together"
            )
        sidecar_shard = obj.get("sidecar_shard")
        if sidecar_shard is not None:
            sidecar_shard = _safe_relative_name(
                sidecar_shard, label="record sidecar_shard"
            )
            if sidecar_offset is None:
                raise ExpertManifestError(
                    "record sidecar_shard requires sidecar_offset and sidecar_length"
                )
        return cls(
            layer=_integer(obj["layer"], label="record layer"),
            expert=_integer(obj["expert"], label="record expert"),
            logical_bytes=_integer(
                obj["logical_bytes"], label="record logical_bytes", minimum=1
            ),
            segments=segments,
            sha256=None if digest is None else _string(digest, label="record sha256"),
            sidecar_offset=(
                None
                if sidecar_offset is None
                else _integer(sidecar_offset, label="record sidecar_offset")
            ),
            sidecar_length=(
                None
                if sidecar_length is None
                else _integer(sidecar_length, label="record sidecar_length", minimum=1)
            ),
            sidecar_shard=sidecar_shard,
        )


@dataclass(frozen=True)
class SidecarInfo:
    file: str
    alignment: int
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "alignment": self.alignment,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SidecarInfo:
        obj = _expect_object(value, label="sidecar")
        _expect_keys(
            obj, label="sidecar", required=("file", "alignment", "size", "sha256")
        )
        alignment = _integer(obj["alignment"], label="sidecar alignment", minimum=1)
        if alignment & (alignment - 1):
            raise ExpertManifestError("sidecar alignment must be a power of two")
        return cls(
            file=_safe_relative_name(obj["file"], label="sidecar file"),
            alignment=alignment,
            size=_integer(obj["size"], label="sidecar size", minimum=1),
            sha256=_string(obj["sha256"], label="sidecar sha256"),
        )


@dataclass(frozen=True)
class ExpertManifest:
    model_key: str
    source_repo: str
    source_revision: str
    quant_bits: int
    quant_group_size: int
    quant_mode: str
    artifact_tensor_bytes: int
    resident_tensor_bytes: int
    routed_expert_bytes: int
    shards: tuple[ShardInfo, ...]
    resident_tensors: tuple[ResidentTensor, ...]
    records: tuple[ExpertRecord, ...]
    sidecar: SidecarInfo | None = None
    manifest_sha256: str | None = None
    format: str = MANIFEST_FORMAT
    # Sharded sidecar layout only: the record alignment shared by every
    # sidecar shard (records carry ``sidecar_shard`` and shard-relative
    # offsets; there is no single-file ``sidecar`` entry).
    sidecar_alignment: int | None = None

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": self.format,
            "model_key": self.model_key,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "quantization": {
                "bits": self.quant_bits,
                "group_size": self.quant_group_size,
                "mode": self.quant_mode,
            },
            "artifact": {
                "tensor_bytes": self.artifact_tensor_bytes,
                "resident_tensor_bytes": self.resident_tensor_bytes,
                "routed_expert_bytes": self.routed_expert_bytes,
                "shard_count": len(self.shards),
                "record_count": len(self.records),
            },
            "shards": [shard.to_dict() for shard in self.shards],
            "resident_tensors": [tensor.to_dict() for tensor in self.resident_tensors],
            "records": [record.to_dict() for record in self.records],
        }
        if self.sidecar is not None:
            result["sidecar"] = self.sidecar.to_dict()
        if self.sidecar_alignment is not None:
            result["sidecar_alignment"] = self.sidecar_alignment
        if include_digest and self.manifest_sha256 is not None:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def with_digest(self) -> ExpertManifest:
        digest = _sha256_bytes(_canonical_json(self.to_dict(include_digest=False)))
        return replace(self, manifest_sha256=digest)

    def record(self, layer: int, expert: int) -> ExpertRecord:
        for record in self.records:
            if record.layer == layer and record.expert == expert:
                return record
        raise ExpertManifestError(f"manifest has no expert record ({layer}, {expert})")

    @classmethod
    def from_dict(cls, value: Any, *, verify_digest: bool = True) -> ExpertManifest:
        obj = _expect_object(value, label="manifest")
        _expect_keys(
            obj,
            label="manifest",
            required=(
                "format",
                "model_key",
                "source_repo",
                "source_revision",
                "quantization",
                "artifact",
                "shards",
                "resident_tensors",
                "records",
            ),
            optional=("sidecar", "sidecar_alignment", "manifest_sha256"),
        )
        if obj["format"] != MANIFEST_FORMAT:
            raise ExpertManifestError(f"unsupported manifest format {obj['format']!r}")
        quant = _expect_object(obj["quantization"], label="quantization")
        _expect_keys(
            quant, label="quantization", required=("bits", "group_size", "mode")
        )
        artifact = _expect_object(obj["artifact"], label="artifact")
        _expect_keys(
            artifact,
            label="artifact",
            required=(
                "tensor_bytes",
                "resident_tensor_bytes",
                "routed_expert_bytes",
                "shard_count",
                "record_count",
            ),
        )
        raw_shards = obj["shards"]
        raw_resident = obj["resident_tensors"]
        raw_records = obj["records"]
        if not all(
            isinstance(value, list) for value in (raw_shards, raw_resident, raw_records)
        ):
            raise ExpertManifestError(
                "shards, resident_tensors, and records must be arrays"
            )
        shard_count = _integer(artifact["shard_count"], label="artifact shard_count")
        record_count = _integer(artifact["record_count"], label="artifact record_count")
        if shard_count > MAX_MANIFEST_SHARDS or len(raw_shards) > MAX_MANIFEST_SHARDS:
            raise ExpertManifestError("manifest shard count exceeds the shard limit")
        if (
            record_count > MAX_MANIFEST_RECORDS
            or len(raw_records) > MAX_MANIFEST_RECORDS
        ):
            raise ExpertManifestError("manifest record count exceeds the record limit")
        if len(raw_resident) > MAX_MANIFEST_RESIDENT_TENSORS:
            raise ExpertManifestError(
                "manifest resident tensor count exceeds the resident limit"
            )
        if shard_count != len(raw_shards) or record_count != len(raw_records):
            raise ExpertManifestError("artifact counts do not match manifest arrays")
        shards = tuple(ShardInfo.from_dict(item) for item in raw_shards)
        resident_tensors = tuple(
            ResidentTensor.from_dict(item) for item in raw_resident
        )
        records = tuple(ExpertRecord.from_dict(item) for item in raw_records)
        digest = obj.get("manifest_sha256")
        manifest = cls(
            format=MANIFEST_FORMAT,
            model_key=_string(obj["model_key"], label="model_key"),
            source_repo=_string(obj["source_repo"], label="source_repo"),
            source_revision=_string(obj["source_revision"], label="source_revision"),
            quant_bits=_integer(quant["bits"], label="quantization bits", minimum=1),
            quant_group_size=_integer(
                quant["group_size"], label="quantization group_size", minimum=1
            ),
            quant_mode=_string(quant["mode"], label="quantization mode"),
            artifact_tensor_bytes=_integer(
                artifact["tensor_bytes"], label="artifact tensor_bytes", minimum=1
            ),
            resident_tensor_bytes=_integer(
                artifact["resident_tensor_bytes"],
                label="resident_tensor_bytes",
                minimum=0,
            ),
            routed_expert_bytes=_integer(
                artifact["routed_expert_bytes"], label="routed_expert_bytes", minimum=1
            ),
            shards=shards,
            resident_tensors=resident_tensors,
            records=records,
            sidecar=None
            if obj.get("sidecar") is None
            else SidecarInfo.from_dict(obj["sidecar"]),
            sidecar_alignment=None
            if obj.get("sidecar_alignment") is None
            else _integer(
                obj["sidecar_alignment"], label="sidecar_alignment", minimum=1
            ),
            manifest_sha256=None
            if digest is None
            else _string(digest, label="manifest_sha256"),
        )
        manifest.validate_structure()
        if verify_digest:
            if manifest.manifest_sha256 is None:
                raise ExpertManifestError("manifest_sha256 is required")
            expected = manifest.with_digest().manifest_sha256
            if manifest.manifest_sha256 != expected:
                raise ExpertManifestError("manifest digest mismatch")
        return manifest

    def validate_structure(self) -> None:
        if self.quant_bits not in {2, 4} or self.quant_mode != "affine":
            raise ExpertManifestError(
                "only affine Q2 or Q4 expert manifests are supported"
            )
        if len(self.shards) > MAX_MANIFEST_SHARDS:
            raise ExpertManifestError("manifest shard count exceeds the shard limit")
        if len(self.resident_tensors) > MAX_MANIFEST_RESIDENT_TENSORS:
            raise ExpertManifestError(
                "manifest resident tensor count exceeds the resident limit"
            )
        if len(self.records) > MAX_MANIFEST_RECORDS:
            raise ExpertManifestError("manifest record count exceeds the record limit")
        if len({shard.name for shard in self.shards}) != len(self.shards):
            raise ExpertManifestError("duplicate shard names")
        shard_by_name: dict[str, ShardInfo] = {}
        for shard in self.shards:
            if _safe_relative_name(shard.name, label="shard name") != shard.name:
                raise ExpertManifestError(f"unsafe shard name: {shard.name!r}")
            if shard.kind not in _SHARD_KINDS:
                raise ExpertManifestError(f"unsupported shard kind {shard.kind!r}")
            _integer(shard.size, label="shard size", minimum=1)
            _string(shard.header_sha256, label="shard header_sha256")
            if shard.sha256 is not None:
                _string(shard.sha256, label="shard sha256")
            header_minimum = 0 if shard.kind == "sidecar" else 1
            _integer(
                shard.header_bytes,
                label="shard header_bytes",
                minimum=header_minimum,
            )
            if shard.kind == "sidecar":
                if shard.header_bytes != 0 or shard.header_sha256 != EMPTY_SHA256:
                    raise ExpertManifestError(
                        "sidecar shard must have an empty zero-byte header"
                    )
            else:
                if shard.header_bytes > shard.size:
                    raise ExpertManifestError("safetensors header exceeds its shard")
            shard_by_name[shard.name] = shard
        if (
            self.artifact_tensor_bytes
            != self.resident_tensor_bytes + self.routed_expert_bytes
        ):
            raise ExpertManifestError(
                "resident plus routed bytes do not equal artifact bytes"
            )
        keys = [(record.layer, record.expert) for record in self.records]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ExpertManifestError("expert records must be sorted and unique")
        resident_names = [tensor.tensor for tensor in self.resident_tensors]
        if resident_names != sorted(resident_names) or len(resident_names) != len(
            set(resident_names)
        ):
            raise ExpertManifestError("resident tensor names must be sorted and unique")
        shard_sizes = {shard.name: shard.size for shard in self.shards}
        routed_bytes = 0
        for record in self.records:
            if tuple(segment.component for segment in record.segments) != _COMPONENTS:
                raise ExpertManifestError(
                    "expert record must contain the nine ordered quantized components"
                )
            if (
                sum(segment.length for segment in record.segments)
                != record.logical_bytes
            ):
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) length mismatch"
                )
            if (
                record.sidecar_length is not None
                and record.sidecar_length != record.logical_bytes
            ):
                raise ExpertManifestError(
                    "sidecar record length must equal logical_bytes"
                )
            routed_bytes += record.logical_bytes
            for segment in record.segments:
                _safe_relative_name(segment.shard, label="segment shard")
                if segment.dtype not in _DTYPE_BYTES:
                    raise ExpertManifestError(
                        f"unsupported segment dtype {segment.dtype!r}"
                    )
                expected_length = _DTYPE_BYTES[segment.dtype]
                for dimension in segment.shape:
                    expected_length *= dimension
                if segment.length != expected_length:
                    raise ExpertManifestError(
                        f"segment for {segment.tensor} has inconsistent dtype/shape bytes"
                    )
                size = shard_sizes.get(segment.shard)
                if size is None or segment.offset + segment.length > size:
                    raise ExpertManifestError(
                        f"segment for {segment.tensor} exceeds its shard"
                    )
        if routed_bytes != self.routed_expert_bytes:
            raise ExpertManifestError("record bytes do not equal routed_expert_bytes")
        if (
            sum(tensor.length for tensor in self.resident_tensors)
            != self.resident_tensor_bytes
        ):
            raise ExpertManifestError(
                "resident tensor bytes do not match artifact metadata"
            )
        resident_shard_names: set[str] = set()
        for tensor in self.resident_tensors:
            shard = shard_by_name.get(tensor.shard)
            if shard is None or shard.kind != "safetensors":
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} must reference a safetensors shard"
                )
            expected_length = _DTYPE_BYTES.get(tensor.dtype)
            if expected_length is None:
                raise ExpertManifestError(
                    f"unsupported resident dtype {tensor.dtype!r}"
                )
            for dimension in tensor.shape:
                expected_length *= dimension
            if tensor.length != expected_length:
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} has inconsistent dtype/shape bytes"
                )
            if tensor.offset < shard.header_bytes:
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} overlaps its shard header"
                )
            if tensor.offset + tensor.length > shard.size:
                raise ExpertManifestError(
                    f"resident tensor {tensor.tensor} exceeds its shard"
                )
            resident_shard_names.add(tensor.shard)
        sharded_records = [
            record for record in self.records if record.sidecar_shard is not None
        ]
        if sharded_records and self.sidecar is not None:
            raise ExpertManifestError(
                "sharded sidecar records conflict with single-file sidecar metadata"
            )
        if self.sidecar_alignment is not None and not sharded_records:
            raise ExpertManifestError(
                "sidecar_alignment requires sharded sidecar records"
            )
        if (
            self.sidecar is None
            and not sharded_records
            and any(record.sidecar_offset is not None for record in self.records)
        ):
            raise ExpertManifestError("record sidecar offsets require sidecar metadata")
        if self.sidecar is not None:
            _safe_relative_name(self.sidecar.file, label="sidecar file")
            alignment = _integer(
                self.sidecar.alignment,
                label="sidecar alignment",
                minimum=1,
            )
            if alignment & (alignment - 1):
                raise ExpertManifestError("sidecar alignment must be a power of two")
            _integer(self.sidecar.size, label="sidecar size", minimum=1)
            _string(self.sidecar.sha256, label="sidecar sha256")
            ranges: list[tuple[int, int]] = []
            for record in self.records:
                if record.sidecar_offset is None or record.sidecar_length is None:
                    raise ExpertManifestError("every record requires a sidecar offset")
                if record.sidecar_offset % self.sidecar.alignment:
                    raise ExpertManifestError("sidecar record is not aligned")
                end = record.sidecar_offset + record.sidecar_length
                if end > self.sidecar.size:
                    raise ExpertManifestError("sidecar record exceeds file size")
                ranges.append((record.sidecar_offset, end))
            if ranges != sorted(ranges) or any(
                a[1] > b[0] for a, b in zip(ranges, ranges[1:])
            ):
                raise ExpertManifestError("sidecar records overlap or are unsorted")

        sidecar_shards = [shard for shard in self.shards if shard.kind == "sidecar"]
        if sharded_records:
            self._validate_sharded_sidecar(sidecar_shards, sharded_records)
            return
        if len(sidecar_shards) > 1:
            raise ExpertManifestError(
                "an authoritative manifest requires exactly one sidecar shard"
            )
        if sidecar_shards:
            if self.sidecar is None:
                raise ExpertManifestError(
                    "an authoritative sidecar shard requires sidecar metadata"
                )
            sidecar_shard = sidecar_shards[0]
            if (
                sidecar_shard.name != self.sidecar.file
                or sidecar_shard.size != self.sidecar.size
                or sidecar_shard.sha256 != self.sidecar.sha256
            ):
                raise ExpertManifestError("authoritative sidecar metadata mismatch")
            if sidecar_shard.sha256 is None:
                raise ExpertManifestError("authoritative sidecar requires a hash")
            _sha256(sidecar_shard.sha256, label="authoritative sidecar sha256")
            safetensors_shards = {
                shard.name for shard in self.shards if shard.kind == "safetensors"
            }
            if safetensors_shards != resident_shard_names:
                raise ExpertManifestError(
                    "authoritative manifest contains unreferenced safetensors shards"
                )
            for name in resident_shard_names:
                if shard_by_name[name].sha256 is None:
                    raise ExpertManifestError(
                        f"authoritative resident shard {name} requires a hash"
                    )
                _sha256(
                    shard_by_name[name].sha256,
                    label=f"authoritative resident shard {name} sha256",
                )
            for record in self.records:
                if record.sidecar_offset is None:
                    raise ExpertManifestError(
                        "authoritative record requires a sidecar offset"
                    )
                cursor = record.sidecar_offset
                for segment in record.segments:
                    if segment.shard != self.sidecar.file or segment.offset != cursor:
                        raise ExpertManifestError(
                            "authoritative record components must be contiguous in the sidecar"
                        )
                    cursor += segment.length
                if cursor != record.sidecar_offset + record.logical_bytes:
                    raise ExpertManifestError(
                        "authoritative record components do not cover the record"
                    )

    def _validate_sharded_sidecar(
        self,
        sidecar_shards: list[ShardInfo],
        sharded_records: list[ExpertRecord],
    ) -> None:
        """Validate the sharded sidecar layout (per-record sidecar shards)."""

        if len(sharded_records) != len(self.records):
            raise ExpertManifestError(
                "sharded sidecar layout requires sidecar_shard on every record"
            )
        alignment = self.sidecar_alignment
        if alignment is None:
            raise ExpertManifestError(
                "sharded sidecar records require sidecar_alignment"
            )
        if alignment & (alignment - 1):
            raise ExpertManifestError("sidecar_alignment must be a power of two")
        shard_map = {shard.name: shard for shard in sidecar_shards}
        for shard in sidecar_shards:
            if shard.sha256 is None:
                raise ExpertManifestError(
                    f"sidecar shard {shard.name} requires a full-file hash"
                )
            _sha256(shard.sha256, label=f"sidecar shard {shard.name} sha256")
        ranges_by_shard: dict[str, list[tuple[int, int]]] = {}
        for record in self.records:
            assert record.sidecar_shard is not None
            shard = shard_map.get(record.sidecar_shard)
            if shard is None:
                raise ExpertManifestError(
                    f"record sidecar shard {record.sidecar_shard!r} is not a "
                    "sidecar shard of this manifest"
                )
            if record.sha256 is None:
                raise ExpertManifestError(
                    "sharded sidecar records require record hashes"
                )
            _sha256(record.sha256, label="sharded record sha256")
            if record.sidecar_offset is None or record.sidecar_length is None:
                raise ExpertManifestError(
                    "every sharded record requires a sidecar offset"
                )
            if record.sidecar_offset % alignment:
                raise ExpertManifestError("sidecar record is not aligned")
            end = record.sidecar_offset + record.sidecar_length
            if end > shard.size:
                raise ExpertManifestError(
                    f"sidecar record exceeds shard {shard.name}"
                )
            cursor = record.sidecar_offset
            for segment in record.segments:
                if segment.shard != record.sidecar_shard or segment.offset != cursor:
                    raise ExpertManifestError(
                        "sharded record components must be contiguous in their "
                        "sidecar shard"
                    )
                cursor += segment.length
            if cursor != end:
                raise ExpertManifestError(
                    "sharded record components do not cover the record"
                )
            ranges_by_shard.setdefault(record.sidecar_shard, []).append(
                (record.sidecar_offset, end)
            )
        for name, ranges in ranges_by_shard.items():
            ordered = sorted(ranges)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ExpertManifestError(f"sidecar shard {name} records overlap")


@dataclass(frozen=True)
class _TensorInfo:
    name: str
    shard: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]


def load_expert_manifest(
    path: Path | str, *, verify_digest: bool = True
) -> ExpertManifest:
    value = _load_json_file(Path(path))
    return ExpertManifest.from_dict(value, verify_digest=verify_digest)


def save_expert_manifest(manifest: ExpertManifest, path: Path | str) -> ExpertManifest:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    finalized = manifest.with_digest()
    payload = json.dumps(finalized.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExpertManifestError(f"could not save manifest {target}: {exc}") from exc
    return finalized


def _read_safetensors_header(
    path: Path, *, relative_name: str
) -> tuple[ShardInfo, tuple[_TensorInfo, ...]]:
    try:
        fd = os.open(path, _readonly_flags())
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ExpertManifestError(
                    f"{relative_name} is not a regular safetensors file"
                )
            size = metadata.st_size
            length_raw = _pread_exact(fd, 0, 8)
            header_length = int.from_bytes(length_raw, "little")
            if not 1 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ExpertManifestError(
                    f"{relative_name} has an invalid header length"
                )
            header_raw = _pread_exact(fd, 8, header_length)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ExpertManifestError(f"could not inspect {relative_name}: {exc}") from exc
    header = _expect_object(
        _load_json_bytes(header_raw, source=relative_name),
        label=f"safetensors header {relative_name}",
    )
    if len(header) > MAX_MANIFEST_RESIDENT_TENSORS:
        raise ExpertManifestError(
            f"{relative_name} tensor count exceeds the header inventory limit"
        )
    data_start = 8 + header_length
    tensors: list[_TensorInfo] = []
    ranges: list[tuple[int, int, str]] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        info = _expect_object(raw, label=f"tensor {name}")
        _expect_keys(
            info, label=f"tensor {name}", required=("dtype", "shape", "data_offsets")
        )
        dtype = _string(info["dtype"], label=f"tensor {name} dtype")
        if dtype not in _DTYPE_BYTES:
            raise ExpertManifestError(f"tensor {name} has unsupported dtype {dtype!r}")
        shape = _shape(info["shape"], label=f"tensor {name} shape")
        offsets = info["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ExpertManifestError(f"tensor {name} has invalid data_offsets")
        start = _integer(offsets[0], label=f"tensor {name} start")
        end = _integer(offsets[1], label=f"tensor {name} end")
        if end <= start:
            raise ExpertManifestError(f"tensor {name} has an empty or reversed range")
        length = end - start
        expected = _DTYPE_BYTES[dtype]
        for dimension in shape:
            expected *= dimension
        if length != expected:
            raise ExpertManifestError(
                f"tensor {name} byte length {length} does not match dtype/shape {expected}"
            )
        absolute_start = data_start + start
        absolute_end = data_start + end
        if absolute_end > size:
            raise ExpertManifestError(f"tensor {name} exceeds {relative_name}")
        tensors.append(
            _TensorInfo(
                name=name,
                shard=relative_name,
                offset=absolute_start,
                length=length,
                dtype=dtype,
                shape=shape,
            )
        )
        ranges.append((absolute_start, absolute_end, name))
    if not tensors:
        raise ExpertManifestError(f"{relative_name} contains no tensors")
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ExpertManifestError(
                f"overlapping tensors in {relative_name}: {previous[2]} and {current[2]}"
            )
    if max(end for _start, end, _name in ranges) != size:
        raise ExpertManifestError(
            f"{relative_name} has trailing or unindexed payload bytes"
        )
    shard = ShardInfo(
        name=relative_name,
        size=size,
        header_bytes=data_start,
        header_sha256=_sha256_bytes(length_raw + header_raw),
    )
    return shard, tuple(tensors)


def _checkpoint_inventory(
    root: Path,
    *,
    hash_shards: bool,
) -> tuple[tuple[ShardInfo, ...], dict[str, _TensorInfo]]:
    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    declared_total_size: int | None = None
    if index_path.is_file() or index_path.is_symlink():
        index_path = _resolve_member(root, index_path.name)
        index = _expect_object(
            _load_json_file(index_path, max_bytes=MAX_INDEX_BYTES),
            label="safetensors index",
        )
        _expect_keys(
            index,
            label="safetensors index",
            required=("weight_map",),
            optional=("metadata",),
        )
        raw_map = _expect_object(index["weight_map"], label="safetensors weight_map")
        if len(raw_map) > MAX_MANIFEST_RESIDENT_TENSORS:
            raise ExpertManifestError(
                "safetensors weight_map exceeds the tensor inventory limit"
            )
        weight_map = {}
        for tensor, shard in raw_map.items():
            tensor_name = _string(tensor, label="weight_map tensor")
            shard_name = _safe_relative_name(shard, label="weight_map shard")
            weight_map[tensor_name] = shard_name
        if "metadata" in index:
            metadata = _expect_object(
                index["metadata"], label="safetensors index metadata"
            )
            if "total_size" in metadata:
                declared_total_size = _integer(
                    metadata["total_size"],
                    label="safetensors index metadata total_size",
                )
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = sorted(
            path.name for path in root.glob("model*.safetensors") if path.is_file()
        )
    if not shard_names:
        raise ExpertManifestError(f"no model safetensors found under {root}")

    shards: list[ShardInfo] = []
    tensors: dict[str, _TensorInfo] = {}
    for name in shard_names:
        path = _resolve_member(root, name)
        shard, shard_tensors = _read_safetensors_header(path, relative_name=name)
        if hash_shards:
            shard = replace(shard, sha256=_hash_file(path))
        shards.append(shard)
        for tensor in shard_tensors:
            if tensor.name in tensors:
                raise ExpertManifestError(
                    f"tensor {tensor.name!r} appears in multiple shards"
                )
            if weight_map is not None and weight_map.get(tensor.name) != name:
                raise ExpertManifestError(
                    f"index maps tensor {tensor.name!r} to the wrong shard"
                )
            tensors[tensor.name] = tensor
    if weight_map is not None and set(weight_map) != set(tensors):
        missing = sorted(set(weight_map) - set(tensors))
        extra = sorted(set(tensors) - set(weight_map))
        raise ExpertManifestError(
            f"index/header tensor mismatch; missing={missing[:4]}, extra={extra[:4]}"
        )
    if declared_total_size is not None and declared_total_size != sum(
        tensor.length for tensor in tensors.values()
    ):
        raise ExpertManifestError(
            "safetensors index total_size does not match header inventory"
        )
    return tuple(shards), tensors


def _expected_component_shape(
    spec: ExpertStreamingModelSpec,
    component: str,
) -> tuple[str, tuple[int, ...], int]:
    projection, leaf = component.split(".", 1)
    output = (
        spec.expert_hidden_size
        if projection in {"gate_proj", "up_proj"}
        else spec.hidden_size
    )
    input_size = (
        spec.hidden_size
        if projection in {"gate_proj", "up_proj"}
        else spec.expert_hidden_size
    )
    if leaf == "weight":
        dtype = "U32"
        shape = (output, input_size * spec.quant_bits // 32)
    else:
        dtype = "BF16"
        shape = (output, input_size // spec.quant_group_size)
    length = _DTYPE_BYTES[dtype]
    for dimension in shape:
        length *= dimension
    return dtype, shape, length


def _classify_expert_name(
    name: str,
    spec: ExpertStreamingModelSpec,
) -> tuple[int, int | None, str] | None:
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    layer = int(match.group("layer"))
    if layer not in spec.routed_layer_indices:
        return None
    parts = match.group("tail").split(".")
    expert: int | None = None
    if len(parts) == 2 and parts[0] in _PROJECTIONS and parts[1] in _LEAVES:
        projection, leaf = parts
    elif (
        len(parts) == 3
        and parts[0].isdigit()
        and parts[1] in _PROJECTIONS
        and parts[2] in _LEAVES
    ):
        expert = int(parts[0])
        projection, leaf = parts[1:]
    else:
        return None
    return layer, expert, f"{projection}.{leaf}"


def _classify_expert_tensor(
    tensor: _TensorInfo,
    spec: ExpertStreamingModelSpec,
) -> tuple[int, int | None, str] | None:
    return _classify_expert_name(tensor.name, spec)


def _expert_segments(
    tensors: dict[str, _TensorInfo],
    spec: ExpertStreamingModelSpec,
) -> tuple[tuple[ExpertRecord, ...], set[str]]:
    if spec.quant_bits not in {2, 4} or spec.quant_parameter_bytes != 2:
        raise ExpertManifestError(
            "manifest builder supports affine Q2/Q4 metadata with BF16 parameters"
        )
    grouped: dict[tuple[int, int, str], TensorSegment] = {}
    expert_tensor_names: set[str] = set()
    for tensor in tensors.values():
        classified = _classify_expert_tensor(tensor, spec)
        if classified is None:
            continue
        layer, numbered_expert, component = classified
        expected_dtype, per_expert_shape, per_expert_length = _expected_component_shape(
            spec, component
        )
        if tensor.dtype != expected_dtype:
            raise ExpertManifestError(
                f"{tensor.name} dtype {tensor.dtype} does not match {expected_dtype}"
            )
        if numbered_expert is None:
            expected_shape = (spec.expert_count, *per_expert_shape)
            if (
                tensor.shape != expected_shape
                or tensor.length != per_expert_length * spec.expert_count
            ):
                raise ExpertManifestError(
                    f"{tensor.name} shape/length does not match stacked {expected_shape}"
                )
            experts = range(spec.expert_count)
        else:
            if not 0 <= numbered_expert < spec.expert_count:
                raise ExpertManifestError(
                    f"{tensor.name} expert index is outside the model"
                )
            if tensor.shape != per_expert_shape or tensor.length != per_expert_length:
                raise ExpertManifestError(
                    f"{tensor.name} shape/length does not match {per_expert_shape}"
                )
            experts = (numbered_expert,)
        expert_tensor_names.add(tensor.name)
        for expert in experts:
            key = (layer, expert, component)
            if key in grouped:
                raise ExpertManifestError(f"duplicate expert component {key}")
            relative_index = expert if numbered_expert is None else 0
            grouped[key] = TensorSegment(
                component=component,
                tensor=tensor.name,
                shard=tensor.shard,
                offset=tensor.offset + relative_index * per_expert_length,
                length=per_expert_length,
                dtype=tensor.dtype,
                shape=per_expert_shape,
            )

    records: list[ExpertRecord] = []
    missing: list[tuple[int, int, str]] = []
    for layer in spec.routed_layer_indices:
        for expert in range(spec.expert_count):
            segments: list[TensorSegment] = []
            for component in _COMPONENTS:
                segment = grouped.get((layer, expert, component))
                if segment is None:
                    missing.append((layer, expert, component))
                else:
                    segments.append(segment)
            if len(segments) == len(_COMPONENTS):
                logical_bytes = sum(segment.length for segment in segments)
                if logical_bytes != spec.expert_record_bytes:
                    raise ExpertManifestError(
                        f"record ({layer}, {expert}) has {logical_bytes} bytes, expected "
                        f"{spec.expert_record_bytes}"
                    )
                records.append(
                    ExpertRecord(
                        layer=layer,
                        expert=expert,
                        logical_bytes=logical_bytes,
                        segments=tuple(segments),
                    )
                )
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        raise ExpertManifestError(
            f"checkpoint is missing {len(missing)} expert components: {preview}"
        )
    return tuple(records), expert_tensor_names


def _pread_exact(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    position = offset
    while remaining:
        try:
            chunk = os.pread(fd, remaining, position)
        except InterruptedError:
            continue
        if not chunk:
            raise ExpertManifestError(
                f"short positional read at offset {position}; wanted {remaining} more bytes"
            )
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_source_record(root: Path, record: ExpertRecord) -> bytes:
    descriptors: dict[str, int] = {}
    chunks: list[bytes] = []
    try:
        for segment in record.segments:
            fd = descriptors.get(segment.shard)
            if fd is None:
                path = _resolve_member(root, segment.shard)
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags)
                descriptors[segment.shard] = fd
            chunks.append(_pread_exact(fd, segment.offset, segment.length))
    except OSError as exc:
        raise ExpertManifestError(
            f"could not read record ({record.layer}, {record.expert}): {exc}"
        ) from exc
    finally:
        for fd in descriptors.values():
            try:
                os.close(fd)
            except OSError:
                pass
    return b"".join(chunks)


def build_expert_manifest(
    root: Path | str,
    spec: ExpertStreamingModelSpec | str,
    *,
    source_repo: str | None = None,
    source_revision: str | None = None,
    hash_records: bool = True,
    hash_shards: bool = False,
    require_pinned_tensor_bytes: bool = True,
) -> ExpertManifest:
    """Inspect a checkpoint and build a strict deterministic expert manifest."""

    artifact_root = Path(root).resolve()
    if not artifact_root.is_dir():
        raise ExpertManifestError(f"artifact root is not a directory: {artifact_root}")
    model_spec = get_model_spec(spec) if isinstance(spec, str) else spec
    if not isinstance(model_spec, ExpertStreamingModelSpec):
        raise TypeError("spec must be a model key or ExpertStreamingModelSpec")
    shards, tensors = _checkpoint_inventory(artifact_root, hash_shards=hash_shards)
    records, expert_tensor_names = _expert_segments(tensors, model_spec)
    tensor_bytes = sum(tensor.length for tensor in tensors.values())
    if require_pinned_tensor_bytes and tensor_bytes != model_spec.total_tensor_bytes:
        raise ExpertManifestError(
            f"artifact tensor bytes {tensor_bytes} do not match pinned "
            f"{model_spec.total_tensor_bytes}"
        )
    routed_bytes = sum(record.logical_bytes for record in records)
    if routed_bytes != model_spec.routed_expert_bytes:
        raise ExpertManifestError(
            f"routed bytes {routed_bytes} do not match pinned {model_spec.routed_expert_bytes}"
        )
    resident_tensors = tuple(
        ResidentTensor(
            tensor=tensor.name,
            shard=tensor.shard,
            offset=tensor.offset,
            length=tensor.length,
            dtype=tensor.dtype,
            shape=tensor.shape,
        )
        for tensor in sorted(tensors.values(), key=lambda item: item.name)
        if tensor.name not in expert_tensor_names
    )
    resident_bytes = sum(tensor.length for tensor in resident_tensors)
    if resident_bytes + routed_bytes != tensor_bytes:
        raise ExpertManifestError(
            "expert/resident classification does not cover the artifact"
        )
    if require_pinned_tensor_bytes and resident_bytes != model_spec.resident_bytes:
        raise ExpertManifestError(
            f"resident bytes {resident_bytes} do not match pinned {model_spec.resident_bytes}"
        )
    if hash_records:
        hashed: list[ExpertRecord] = []
        for record in records:
            payload = _read_source_record(artifact_root, record)
            hashed.append(replace(record, sha256=_sha256_bytes(payload)))
        records = tuple(hashed)
    manifest = ExpertManifest(
        model_key=model_spec.key,
        source_repo=source_repo or model_spec.quant_model,
        source_revision=source_revision or model_spec.quant_revision,
        quant_bits=model_spec.quant_bits,
        quant_group_size=model_spec.quant_group_size,
        quant_mode="affine",
        artifact_tensor_bytes=tensor_bytes,
        resident_tensor_bytes=resident_bytes,
        routed_expert_bytes=routed_bytes,
        shards=shards,
        resident_tensors=resident_tensors,
        records=records,
    ).with_digest()
    manifest.validate_structure()
    return manifest


def validate_expert_manifest_spec(
    manifest: ExpertManifest,
    spec: ExpertStreamingModelSpec,
    *,
    require_pinned_tensor_bytes: bool = True,
) -> None:
    """Require exact identity, record keys, bytes, and component geometry."""

    if not isinstance(manifest, ExpertManifest):
        raise TypeError("manifest must be an ExpertManifest")
    if not isinstance(spec, ExpertStreamingModelSpec):
        raise TypeError("spec must be an ExpertStreamingModelSpec")
    manifest.validate_structure()
    if manifest.quant_bits != spec.quant_bits:
        raise ExpertManifestError(
            f"manifest bits {manifest.quant_bits} do not match descriptor bits "
            f"{spec.quant_bits}"
        )
    if manifest.quant_group_size != spec.quant_group_size:
        raise ExpertManifestError(
            "manifest quantization group size does not match the descriptor"
        )
    if manifest.quant_mode != "affine":
        raise ExpertManifestError("manifest quantization mode must be affine")
    if manifest.model_key != spec.key:
        raise ExpertManifestError(
            f"manifest model key {manifest.model_key!r} does not match "
            f"descriptor {spec.key!r}"
        )
    if (
        manifest.source_repo != spec.quant_model
        or manifest.source_revision != spec.quant_revision
    ):
        raise ExpertManifestError(
            "manifest source identity does not match the pinned descriptor"
        )

    expected_keys = tuple(
        (layer, expert)
        for layer in spec.routed_layer_indices
        for expert in range(spec.expert_count)
    )
    actual_keys = tuple((record.layer, record.expert) for record in manifest.records)
    if actual_keys != expected_keys:
        raise ExpertManifestError(
            "manifest record keys do not match the descriptor Cartesian product"
        )
    for record in manifest.records:
        if record.logical_bytes != spec.expert_record_bytes:
            raise ExpertManifestError(
                f"record ({record.layer}, {record.expert}) bytes do not match "
                "the descriptor"
            )
        for component, segment in zip(_COMPONENTS, record.segments, strict=True):
            expected_dtype, expected_shape, expected_length = _expected_component_shape(
                spec, component
            )
            if (
                segment.component != component
                or segment.dtype != expected_dtype
                or segment.shape != expected_shape
                or segment.length != expected_length
            ):
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) component "
                    f"{component} does not match descriptor geometry"
                )
    if manifest.routed_expert_bytes != spec.routed_expert_bytes:
        raise ExpertManifestError(
            "manifest routed expert bytes do not match the descriptor"
        )
    if require_pinned_tensor_bytes and (
        manifest.resident_tensor_bytes != spec.resident_bytes
        or manifest.artifact_tensor_bytes != spec.total_tensor_bytes
    ):
        raise ExpertManifestError(
            "manifest resident or total tensor bytes do not match the descriptor"
        )

    mtp_pattern = None
    if spec.mtp_layer_index is not None:
        mtp_pattern = re.compile(rf"(?:^|\.)layers\.{spec.mtp_layer_index}(?:\.|$)")
    for tensor in manifest.resident_tensors:
        if _classify_expert_name(tensor.tensor, spec) is not None:
            raise ExpertManifestError(
                f"resident tensor {tensor.tensor!r} is a routed expert tensor"
            )
        if mtp_pattern is not None and mtp_pattern.search(tensor.tensor):
            raise ExpertManifestError(
                f"resident tensor {tensor.tensor!r} is an MTP layer tensor"
            )


def make_sidecar_authoritative(
    manifest: ExpertManifest,
    spec: ExpertStreamingModelSpec,
) -> ExpertManifest:
    """Retain referenced residents and rebase all expert segments to one sidecar."""

    if (
        manifest.manifest_sha256 is None
        or manifest.manifest_sha256 != manifest.with_digest().manifest_sha256
    ):
        raise ExpertManifestError("manifest digest mismatch")
    validate_expert_manifest_spec(manifest, spec)
    if manifest.sidecar is None:
        raise ExpertManifestError("authoritative conversion requires a sidecar")
    resident_names = {tensor.shard for tensor in manifest.resident_tensors}
    shard_by_name = {shard.name: shard for shard in manifest.shards}
    resident_shards: list[ShardInfo] = []
    for name in sorted(resident_names):
        shard = shard_by_name.get(name)
        if shard is None or shard.kind != "safetensors":
            raise ExpertManifestError(
                f"resident tensor shard {name!r} is not a safetensors shard"
            )
        if shard.sha256 is None:
            raise ExpertManifestError(
                f"authoritative resident shard {name} requires a full-file hash"
            )
        resident_shards.append(shard)
    if manifest.sidecar.file in resident_names:
        raise ExpertManifestError("sidecar file conflicts with a resident shard")

    records: list[ExpertRecord] = []
    for record in manifest.records:
        if (
            record.sha256 is None
            or record.sidecar_offset is None
            or record.sidecar_length is None
        ):
            raise ExpertManifestError(
                "authoritative records require hashes and complete sidecar ranges"
            )
        _sha256(record.sha256, label="authoritative record sha256")
        cursor = record.sidecar_offset
        segments: list[TensorSegment] = []
        for segment in record.segments:
            segments.append(
                replace(
                    segment,
                    shard=manifest.sidecar.file,
                    offset=cursor,
                )
            )
            cursor += segment.length
        if cursor != record.sidecar_offset + record.logical_bytes:
            raise ExpertManifestError(
                "record components do not exactly cover the sidecar record"
            )
        records.append(replace(record, segments=tuple(segments)))

    sidecar_shard = ShardInfo(
        name=manifest.sidecar.file,
        size=manifest.sidecar.size,
        header_bytes=0,
        header_sha256=EMPTY_SHA256,
        sha256=manifest.sidecar.sha256,
        kind="sidecar",
    )
    authoritative = replace(
        manifest,
        shards=tuple(resident_shards) + (sidecar_shard,),
        records=tuple(records),
        manifest_sha256=None,
    ).with_digest()
    validate_expert_manifest_spec(authoritative, spec)
    return authoritative


def verify_expert_manifest(
    manifest: ExpertManifest,
    root: Path | str,
    *,
    verify_records: bool = False,
    verify_shard_hashes: bool = False,
    verify_sidecar_hash: bool = False,
) -> dict[str, int | bool | str]:
    """Verify provenance, bounds, optional payload hashes, and sidecar state."""

    artifact_root = Path(root).resolve()
    manifest.validate_structure()
    if manifest.manifest_sha256 != manifest.with_digest().manifest_sha256:
        raise ExpertManifestError("manifest digest mismatch")
    authoritative = any(shard.kind == "sidecar" for shard in manifest.shards)
    if authoritative:
        if not (artifact_root / "model.safetensors.index.json").is_file():
            raise ExpertManifestError(
                "authoritative artifact requires a complete resident index"
            )
        expected_safetensors = {
            shard.name for shard in manifest.shards if shard.kind == "safetensors"
        }
        actual_safetensors = {path.name for path in artifact_root.glob("*.safetensors")}
        if actual_safetensors != expected_safetensors:
            extra = sorted(actual_safetensors - expected_safetensors)
            missing = sorted(expected_safetensors - actual_safetensors)
            raise ExpertManifestError(
                "authoritative resident inventory has unreferenced or missing "
                f"safetensors shards; extra={extra[:4]}, missing={missing[:4]}"
            )
        inventory_shards, inventory_tensors = _checkpoint_inventory(
            artifact_root,
            hash_shards=False,
        )
        if {shard.name for shard in inventory_shards} != expected_safetensors:
            raise ExpertManifestError("authoritative resident shard inventory mismatch")
        expected_residents = {
            tensor.tensor: tensor for tensor in manifest.resident_tensors
        }
        if set(inventory_tensors) != set(expected_residents):
            raise ExpertManifestError(
                "authoritative resident index/header inventory mismatch"
            )
        for name, tensor in inventory_tensors.items():
            expected = expected_residents[name]
            if (
                tensor.shard != expected.shard
                or tensor.offset != expected.offset
                or tensor.length != expected.length
                or tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
            ):
                raise ExpertManifestError(
                    f"authoritative resident metadata mismatch: {name}"
                )
    checked_shards = 0
    for shard in manifest.shards:
        path = _resolve_member(artifact_root, shard.name)
        if shard.kind == "sidecar":
            try:
                fd = os.open(path, _readonly_flags())
                try:
                    metadata = os.fstat(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise ExpertManifestError(
                    f"could not inspect sidecar {shard.name}: {exc}"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != shard.size:
                raise ExpertManifestError("sidecar size mismatch")
            checked_shards += 1
            continue
        current, _tensors = _read_safetensors_header(path, relative_name=shard.name)
        if (
            current.size != shard.size
            or current.header_bytes != shard.header_bytes
            or current.header_sha256 != shard.header_sha256
        ):
            raise ExpertManifestError(f"shard provenance mismatch: {shard.name}")
        if verify_shard_hashes:
            if shard.sha256 is None:
                raise ExpertManifestError(f"shard {shard.name} has no full-file hash")
            if _hash_file(path) != shard.sha256:
                raise ExpertManifestError(f"shard hash mismatch: {shard.name}")
        checked_shards += 1
    checked_records = 0
    if verify_records:
        for record in manifest.records:
            if record.sha256 is None:
                raise ExpertManifestError(
                    f"record ({record.layer}, {record.expert}) has no hash"
                )
            if (
                _sha256_bytes(_read_source_record(artifact_root, record))
                != record.sha256
            ):
                raise ExpertManifestError(
                    f"record hash mismatch: ({record.layer}, {record.expert})"
                )
            checked_records += 1
    sidecar_verified = False
    if manifest.sidecar is not None:
        sidecar_path = _resolve_member(artifact_root, manifest.sidecar.file)
        if sidecar_path.stat().st_size != manifest.sidecar.size:
            raise ExpertManifestError("sidecar size mismatch")
        if verify_sidecar_hash:
            if _hash_file(sidecar_path) != manifest.sidecar.sha256:
                raise ExpertManifestError("sidecar hash mismatch")
            sidecar_verified = True
    return {
        "valid": True,
        "model_key": manifest.model_key,
        "checked_shards": checked_shards,
        "checked_records": checked_records,
        "sidecar_verified": sidecar_verified,
    }


def read_expert_record(
    manifest: ExpertManifest,
    root: Path | str,
    layer: int,
    expert: int,
    *,
    prefer_sidecar: bool = True,
    verify_hash: bool = True,
) -> bytes:
    """Read one complete expert record from sidecar or source segments."""

    record = manifest.record(layer, expert)
    artifact_root = Path(root).resolve()
    if prefer_sidecar and manifest.sidecar is not None:
        if record.sidecar_offset is None or record.sidecar_length is None:
            raise ExpertManifestError("manifest sidecar record is incomplete")
        path = _resolve_member(artifact_root, manifest.sidecar.file)
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                payload = _pread_exact(fd, record.sidecar_offset, record.sidecar_length)
            finally:
                os.close(fd)
        except OSError as exc:
            raise ExpertManifestError(f"could not read sidecar record: {exc}") from exc
    else:
        payload = _read_source_record(artifact_root, record)
    if len(payload) != record.logical_bytes:
        raise ExpertManifestError("expert record length mismatch")
    if verify_hash:
        if record.sha256 is None:
            raise ExpertManifestError("record has no payload hash")
        if _sha256_bytes(payload) != record.sha256:
            raise ExpertManifestError(
                f"record hash mismatch: ({record.layer}, {record.expert})"
            )
    return payload


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _relative_output(root: Path, output: Path) -> tuple[str, Path]:
    base = root.resolve()
    parent = output.parent.resolve()
    try:
        relative_parent = parent.relative_to(base)
    except ValueError as exc:
        raise ExpertManifestError(
            "sidecar must be created inside the artifact root"
        ) from exc
    relative = (relative_parent / output.name).as_posix()
    return _safe_relative_name(relative, label="sidecar output"), parent / output.name


def build_expert_sidecar(
    manifest: ExpertManifest,
    root: Path | str,
    output: Path | str,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    resume: bool = True,
    overwrite: bool = False,
) -> ExpertManifest:
    """Build a resumable expert-major sidecar and return an updated manifest."""

    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise ExpertManifestError("alignment must be a positive integer")
    if alignment & (alignment - 1):
        raise ExpertManifestError("alignment must be a power of two")
    artifact_root = Path(root).resolve()
    relative_output, final_path = _relative_output(artifact_root, Path(output))
    if final_path.exists() and not overwrite:
        raise ExpertManifestError(f"sidecar already exists: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial = final_path.with_name(f".{final_path.name}.partial")
    if partial.exists() and not resume:
        partial.unlink()
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(partial, flags, 0o644)
    except OSError as exc:
        raise ExpertManifestError(
            f"could not open sidecar partial file: {exc}"
        ) from exc

    updated_records: list[ExpertRecord] = []
    cursor = 0
    try:
        for record in manifest.records:
            offset = _align_up(cursor, alignment)
            payload = read_expert_record(
                manifest,
                artifact_root,
                record.layer,
                record.expert,
                prefer_sidecar=False,
                verify_hash=record.sha256 is not None,
            )
            digest = _sha256_bytes(payload)
            current_size = os.fstat(fd).st_size
            reusable = False
            if resume and current_size >= offset + len(payload):
                reusable = (
                    _sha256_bytes(_pread_exact(fd, offset, len(payload))) == digest
                )
            if not reusable:
                os.ftruncate(fd, offset)
                if offset > os.fstat(fd).st_size:
                    os.ftruncate(fd, offset)
                position = offset
                view = memoryview(payload)
                while view:
                    try:
                        written = os.pwrite(fd, view, position)
                    except InterruptedError:
                        continue
                    if written <= 0:
                        raise ExpertManifestError("short positional sidecar write")
                    position += written
                    view = view[written:]
            updated_records.append(
                replace(
                    record,
                    sha256=digest,
                    sidecar_offset=offset,
                    sidecar_length=len(payload),
                )
            )
            cursor = offset + len(payload)
        os.ftruncate(fd, cursor)
        os.fsync(fd)
    except (OSError, ExpertManifestError) as exc:
        if isinstance(exc, ExpertManifestError):
            raise
        raise ExpertManifestError(f"sidecar build failed: {exc}") from exc
    finally:
        os.close(fd)

    sidecar_sha256 = _hash_file(partial)
    try:
        os.replace(partial, final_path)
        directory_fd = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ExpertManifestError(f"could not publish sidecar: {exc}") from exc
    updated = replace(
        manifest,
        records=tuple(updated_records),
        sidecar=SidecarInfo(
            file=relative_output,
            alignment=alignment,
            size=cursor,
            sha256=sidecar_sha256,
        ),
        manifest_sha256=None,
    ).with_digest()
    updated.validate_structure()
    return updated


def iter_record_keys(manifest: ExpertManifest) -> Iterator[tuple[int, int]]:
    for record in manifest.records:
        yield record.layer, record.expert


DEFAULT_SIDECAR_SHARD_TEMPLATE = "experts-{index:05d}-of-{count:05d}.bin"
DEFAULT_MAX_SIDECAR_SHARD_BYTES = 16 * 1024**3


@dataclass(frozen=True)
class SidecarShardPlan:
    """One planned sidecar shard: a record-aligned slice of ``experts.bin``."""

    name: str
    source_offset: int
    length: int
    record_indices: tuple[int, ...]


def plan_sidecar_shards(
    manifest: ExpertManifest,
    *,
    max_shard_bytes: int = DEFAULT_MAX_SIDECAR_SHARD_BYTES,
    name_template: str = DEFAULT_SIDECAR_SHARD_TEMPLATE,
) -> tuple[SidecarShardPlan, ...]:
    """Cut the single sidecar into record-boundary shards (never mid-record).

    Every shard starts exactly at a record's ``sidecar_offset`` (which is
    aligned), so shard-relative record offsets keep the sidecar alignment.
    Records are never split across shards; a shard may exceed
    ``max_shard_bytes`` only when a single record is larger than the limit,
    which is rejected instead.
    """

    if isinstance(max_shard_bytes, bool) or not isinstance(max_shard_bytes, int):
        raise ExpertManifestError("max_shard_bytes must be an integer")
    if max_shard_bytes <= 0:
        raise ExpertManifestError("max_shard_bytes must be positive")
    if manifest.sidecar is None:
        raise ExpertManifestError("sidecar sharding requires a single-file sidecar")
    manifest.validate_structure()
    for record in manifest.records:
        if record.sidecar_offset is None or record.sidecar_length is None:
            raise ExpertManifestError("every record requires a sidecar offset")
        if record.sidecar_length > max_shard_bytes:
            raise ExpertManifestError(
                f"record ({record.layer}, {record.expert}) is larger than "
                f"max_shard_bytes {max_shard_bytes}"
            )

    boundaries: list[list[int]] = []
    current: list[int] = []
    current_start = 0
    for index, record in enumerate(manifest.records):
        offset = int(record.sidecar_offset or 0)
        end = offset + int(record.sidecar_length or 0)
        if not current:
            current = [index]
            current_start = offset
            continue
        if end - current_start > max_shard_bytes:
            boundaries.append(current)
            current = [index]
            current_start = offset
        else:
            current.append(index)
    if current:
        boundaries.append(current)

    count = len(boundaries)
    plans: list[SidecarShardPlan] = []
    for shard_index, record_indices in enumerate(boundaries, start=1):
        first = manifest.records[record_indices[0]]
        last = manifest.records[record_indices[-1]]
        start = int(first.sidecar_offset or 0)
        end = int(last.sidecar_offset or 0) + int(last.sidecar_length or 0)
        name = _safe_relative_name(
            name_template.format(index=shard_index, count=count),
            label="sidecar shard name",
        )
        plans.append(
            SidecarShardPlan(
                name=name,
                source_offset=start,
                length=end - start,
                record_indices=tuple(record_indices),
            )
        )
    if len({plan.name for plan in plans}) != len(plans):
        raise ExpertManifestError("sidecar shard name template repeats names")
    return tuple(plans)


def write_sidecar_shards(
    manifest: ExpertManifest,
    root: Path | str,
    plans: tuple[SidecarShardPlan, ...],
    *,
    output_dir: Path | str | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> ExpertManifest:
    """Copy planned shard byte ranges out of the sidecar and re-manifest.

    Returns a sharded-layout manifest: records carry ``sidecar_shard`` and
    shard-relative offsets, segments are rebased into their shard file, the
    single ``sidecar`` entry is replaced by one ``kind="sidecar"`` shard per
    output file, and ``sidecar_alignment`` preserves the record alignment.
    The copy is byte-exact (interior alignment padding included) and every
    shard is hashed while it is written.
    """

    if manifest.sidecar is None:
        raise ExpertManifestError("sidecar sharding requires a single-file sidecar")
    if not plans:
        raise ExpertManifestError("sidecar sharding requires at least one shard plan")
    for record in manifest.records:
        if record.sha256 is None:
            raise ExpertManifestError(
                "sharded sidecar records require record hashes; rebuild the "
                "manifest with hash_records=True"
            )
    planned_indices = [index for plan in plans for index in plan.record_indices]
    if planned_indices != list(range(len(manifest.records))):
        raise ExpertManifestError("shard plans must cover every record exactly once")
    artifact_root = Path(root).resolve()
    source_path = _resolve_member(artifact_root, manifest.sidecar.file)
    target_root = (
        artifact_root if output_dir is None else Path(output_dir).resolve()
    )
    target_root.mkdir(parents=True, exist_ok=True)

    shard_infos: list[ShardInfo] = []
    updated_records: dict[int, ExpertRecord] = {}
    source_fd = os.open(source_path, _readonly_flags())
    try:
        for plan in plans:
            digest = hashlib.sha256()
            final_path = target_root / plan.name
            final_path.parent.mkdir(parents=True, exist_ok=True)
            partial = final_path.with_name(f".{final_path.name}.partial")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            flags |= getattr(os, "O_CLOEXEC", 0)
            out_fd = os.open(partial, flags, 0o644)
            try:
                remaining = plan.length
                position = plan.source_offset
                write_position = 0
                while remaining:
                    payload = _pread_exact(
                        source_fd, position, min(chunk_bytes, remaining)
                    )
                    digest.update(payload)
                    view = memoryview(payload)
                    while view:
                        try:
                            written = os.pwrite(out_fd, view, write_position)
                        except InterruptedError:
                            continue
                        if written <= 0:
                            raise ExpertManifestError(
                                "short positional sidecar shard write"
                            )
                        write_position += written
                        view = view[written:]
                    position += len(payload)
                    remaining -= len(payload)
                os.fsync(out_fd)
            except OSError as exc:
                raise ExpertManifestError(
                    f"sidecar shard write failed: {exc}"
                ) from exc
            finally:
                os.close(out_fd)
            try:
                os.replace(partial, final_path)
            except OSError as exc:
                raise ExpertManifestError(
                    f"could not publish sidecar shard {plan.name}: {exc}"
                ) from exc
            shard_infos.append(
                ShardInfo(
                    name=plan.name,
                    size=plan.length,
                    header_bytes=0,
                    header_sha256=EMPTY_SHA256,
                    sha256=digest.hexdigest(),
                    kind="sidecar",
                )
            )
            for index in plan.record_indices:
                record = manifest.records[index]
                assert record.sidecar_offset is not None
                new_offset = record.sidecar_offset - plan.source_offset
                cursor = new_offset
                segments: list[TensorSegment] = []
                for segment in record.segments:
                    segments.append(
                        replace(segment, shard=plan.name, offset=cursor)
                    )
                    cursor += segment.length
                updated_records[index] = replace(
                    record,
                    segments=tuple(segments),
                    sidecar_offset=new_offset,
                    sidecar_shard=plan.name,
                )
    finally:
        os.close(source_fd)

    retained_shards = tuple(
        shard
        for shard in manifest.shards
        if not (shard.kind == "sidecar" and shard.name == manifest.sidecar.file)
    )
    sharded = replace(
        manifest,
        shards=retained_shards + tuple(shard_infos),
        records=tuple(
            updated_records[index] for index in range(len(manifest.records))
        ),
        sidecar=None,
        sidecar_alignment=manifest.sidecar.alignment,
        manifest_sha256=None,
    ).with_digest()
    sharded.validate_structure()
    return sharded
