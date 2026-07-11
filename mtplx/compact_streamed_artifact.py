"""Build a self-contained streamed artifact without duplicating expert bytes.

The compactor deliberately operates on safetensors byte ranges described by an
expert manifest.  It never imports MLX or materializes a tensor: resident data
is copied in bounded chunks, while the already verified expert sidecar is
hard-linked into the new artifact whenever the filesystems permit it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

from .expert_manifest import (
    MAX_SAFETENSORS_HEADER_BYTES,
    ExpertManifest,
    ExpertManifestError,
    ResidentTensor,
    ShardInfo,
    load_expert_manifest,
    make_sidecar_authoritative,
    resolve_artifact_member,
    save_expert_manifest,
    validate_expert_manifest_spec,
    verify_expert_manifest,
)
from .expert_streaming_models import MODEL_SPECS, ExpertStreamingModelSpec


HANDOFF_FORMAT = "mtplx-compact-artifact-handoff-v1"
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_SHARD_BYTES = 5 * 1024 * 1024 * 1024
MANIFEST_NAME = "expert-manifest.json"
INDEX_NAME = "model.safetensors.index.json"
HANDOFF_NAME = "compaction-handoff.json"
RUNTIME_NAME = "mtplx_runtime.json"
MAX_RUNTIME_BYTES = 4 * 1024 * 1024


class CompactArtifactError(ExpertManifestError):
    """Raised when compaction cannot be completed without losing provenance."""


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mode=value.st_mode,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )


@dataclass(frozen=True)
class _SourceFile:
    relative: str
    path: Path
    identity: _FileIdentity


@dataclass(frozen=True)
class _ShardPlan:
    name: str
    tensors: tuple[ResidentTensor, ...]


def _safe_relative_name(name: str, *, label: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise CompactArtifactError(f"unsafe {label}: {name!r}")
    value = PurePosixPath(name)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise CompactArtifactError(f"unsafe {label}: {name!r}")
    return value.as_posix()


def _destination_member(root: Path, relative: str) -> Path:
    safe = _safe_relative_name(relative, label="output path")
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    if candidate == root or root not in candidate.parents:
        raise CompactArtifactError(f"output member escapes artifact: {relative!r}")
    return candidate


def _is_hf_blob_symlink(root: Path, candidate: Path, resolved: Path) -> bool:
    snapshots = root.parent
    blob_root = snapshots.parent / "blobs"
    try:
        resolved_blobs = blob_root.resolve(strict=True)
    except OSError:
        return False
    return (
        snapshots.name == "snapshots"
        and candidate.is_symlink()
        and blob_root.is_dir()
        and not blob_root.is_symlink()
        and resolved_blobs in resolved.parents
    )


def _resolve_source_member(
    root: Path,
    relative: str,
    *,
    allow_hf_blob_symlink: bool = True,
) -> Path:
    safe = _safe_relative_name(relative, label="source path")
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    current = root
    for part in PurePosixPath(safe).parts[:-1]:
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise CompactArtifactError(
                    f"source directory symlinks are not allowed: {safe}"
                )
        except FileNotFoundError as exc:
            raise CompactArtifactError(f"source member is missing: {safe}") from exc
    resolved = resolve_artifact_member(root, safe)
    if candidate.is_symlink() and not (
        allow_hf_blob_symlink and _is_hf_blob_symlink(root, candidate, resolved)
    ):
        raise CompactArtifactError(f"source file symlink is not allowed: {safe}")
    return resolved


def _identity(path: Path) -> _FileIdentity:
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise CompactArtifactError(
            f"could not inspect source file {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(value.st_mode):
        raise CompactArtifactError(f"source member is not a regular file: {path}")
    return _FileIdentity.from_stat(value)


def _capture_source_file(
    root: Path,
    relative: str,
    *,
    allow_hf_blob_symlink: bool = True,
) -> _SourceFile:
    path = _resolve_source_member(
        root,
        relative,
        allow_hf_blob_symlink=allow_hf_blob_symlink,
    )
    return _SourceFile(relative=relative, path=path, identity=_identity(path))


def _assert_source_unchanged(
    root: Path,
    source: _SourceFile,
    *,
    allow_link_ctime_change: bool = False,
) -> None:
    current_path = _resolve_source_member(root, source.relative)
    current = _identity(current_path)
    unchanged = current == source.identity
    if allow_link_ctime_change:
        unchanged = (
            current.device == source.identity.device
            and current.inode == source.identity.inode
            and current.size == source.identity.size
            and current.mode == source.identity.mode
            and current.mtime_ns == source.identity.mtime_ns
        )
    if current_path != source.path or not unchanged:
        raise CompactArtifactError(
            f"source file changed during compaction: {source.relative}"
        )


def _open_readonly(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CompactArtifactError(f"could not open source file {path}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise CompactArtifactError(f"source member is not a regular file: {path}")
    return fd


def _open_output(path: Path, *, mode: int = 0o644) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags, mode)
    except OSError as exc:
        raise CompactArtifactError(
            f"could not create output file {path}: {exc}"
        ) from exc


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise CompactArtifactError("short output write")
        view = view[written:]


def _copy_range(
    source_fd: int,
    output_fd: int,
    *,
    source_offset: int,
    length: int,
    chunk_bytes: int,
    digest: hashlib._Hash | None = None,
) -> None:
    position = source_offset
    remaining = length
    while remaining:
        wanted = min(remaining, chunk_bytes)
        try:
            payload = os.pread(source_fd, wanted, position)
        except InterruptedError:
            continue
        if not payload:
            raise CompactArtifactError(
                f"short source read at offset {position}; wanted {remaining} bytes"
            )
        _write_all(output_fd, payload)
        if digest is not None:
            digest.update(payload)
        position += len(payload)
        remaining -= len(payload)


def _copy_file_bounded(
    source: _SourceFile,
    destination: Path,
    *,
    chunk_bytes: int,
) -> str:
    source_fd = _open_readonly(source.path)
    output_fd = _open_output(
        destination, mode=stat.S_IMODE(source.identity.mode) & 0o777
    )
    digest = hashlib.sha256()
    try:
        _copy_range(
            source_fd,
            output_fd,
            source_offset=0,
            length=source.identity.size,
            chunk_bytes=chunk_bytes,
            digest=digest,
        )
        os.fsync(output_fd)
    finally:
        os.close(source_fd)
        os.close(output_fd)
    if _identity(source.path) != source.identity:
        raise CompactArtifactError(
            f"source file changed while copying: {source.relative}"
        )
    return digest.hexdigest()


def _hash_source_file(source: _SourceFile, *, chunk_bytes: int) -> str:
    """Hash one captured source file and reject replacement during the pass."""

    descriptor = _open_readonly(source.path)
    digest = hashlib.sha256()
    try:
        opened = _FileIdentity.from_stat(os.fstat(descriptor))
        if opened != source.identity:
            raise CompactArtifactError(
                f"source file changed before hashing: {source.relative}"
            )
        position = 0
        remaining = source.identity.size
        while remaining:
            wanted = min(remaining, chunk_bytes)
            try:
                payload = os.pread(descriptor, wanted, position)
            except InterruptedError:
                continue
            if not payload:
                raise CompactArtifactError(
                    f"short source read while hashing {source.relative}"
                )
            digest.update(payload)
            position += len(payload)
            remaining -= len(payload)
    finally:
        os.close(descriptor)
    if _identity(source.path) != source.identity:
        raise CompactArtifactError(
            f"source file changed while hashing: {source.relative}"
        )
    return digest.hexdigest()


def _bind_source_shard_hashes(
    manifest: ExpertManifest,
    sources: dict[str, _SourceFile],
    *,
    chunk_bytes: int,
) -> tuple[ExpertManifest, dict[str, Any]]:
    """Compute every source-shard hash and bind hashes absent in old manifests."""

    shards: list[ShardInfo] = []
    hashes: dict[str, str] = {}
    computed = 0
    matched = 0
    for shard in manifest.shards:
        if shard.kind != "safetensors":
            shards.append(shard)
            continue
        try:
            source = sources[shard.name]
        except KeyError as exc:  # Guarded by the source inventory construction.
            raise CompactArtifactError(
                f"manifest shard has no captured source: {shard.name}"
            ) from exc
        digest = _hash_source_file(source, chunk_bytes=chunk_bytes)
        hashes[shard.name] = digest
        if shard.sha256 is None:
            computed += 1
        elif shard.sha256 != digest:
            raise CompactArtifactError(f"source shard hash mismatch: {shard.name}")
        else:
            matched += 1
        shards.append(replace(shard, sha256=digest))
    bound = replace(
        manifest,
        shards=tuple(shards),
        manifest_sha256=None,
    ).with_digest()
    bound.validate_structure()
    return bound, {
        "valid": True,
        "bound_manifest_sha256": bound.manifest_sha256,
        "verified_shards": len(hashes),
        "computed_missing_hashes": computed,
        "matched_manifest_hashes": matched,
        "sha256": hashes,
    }


def _validate_known_model_spec(
    manifest: ExpertManifest,
) -> ExpertStreamingModelSpec | None:
    """Apply the complete pinned identity and record contract when recognized."""

    spec = MODEL_SPECS.get(manifest.model_key)
    if spec is None:
        return None
    mismatches: list[str] = []
    if manifest.source_repo != spec.manifest_repo:
        mismatches.append("source repository")
    if manifest.source_revision != spec.manifest_revision:
        mismatches.append("source revision")
    if manifest.quant_bits != spec.quant_bits:
        mismatches.append("quantization bits")
    if manifest.quant_group_size != spec.quant_group_size:
        mismatches.append("quantization group size")
    if manifest.quant_mode != "affine":
        mismatches.append("quantization mode")
    if manifest.artifact_tensor_bytes != spec.total_tensor_bytes:
        mismatches.append("artifact tensor bytes")
    if mismatches:
        raise CompactArtifactError(
            "manifest does not match pinned model descriptor: " + ", ".join(mismatches)
        )
    validate_expert_manifest_spec(manifest, spec)
    return spec


def _check_copy_space(root: Path, required_bytes: int) -> None:
    free = shutil.disk_usage(root).free
    if free < required_bytes:
        raise CompactArtifactError(
            f"copy fallback needs {required_bytes} bytes but only {free} are free"
        )


def _link_or_copy(
    source: _SourceFile,
    destination: Path,
    *,
    allow_copy: bool,
    chunk_bytes: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.path, destination, follow_symlinks=False)
    except OSError as exc:
        if not allow_copy:
            hint = "pass --allow-sidecar-copy to explicitly permit duplication"
            if exc.errno == errno.EXDEV:
                hint = f"different filesystems; {hint}"
            raise CompactArtifactError(
                f"could not hard-link {source.relative}: {exc}; {hint}"
            ) from exc
        _check_copy_space(destination.parent, source.identity.size)
        digest = _copy_file_bounded(
            source,
            destination,
            chunk_bytes=chunk_bytes,
        )
        if digest is None:  # pragma: no cover - keeps the result shape explicit
            raise AssertionError("bounded copy did not produce a digest")
        mode = "copied"
    else:
        mode = "hardlinked"

    destination_identity = _identity(destination)
    same_inode = (
        destination_identity.device == source.identity.device
        and destination_identity.inode == source.identity.inode
    )
    if mode == "hardlinked" and not same_inode:
        raise CompactArtifactError(
            f"hard-link did not preserve inode identity: {source.relative}"
        )
    if mode == "copied" and same_inode:
        raise CompactArtifactError(
            f"copy fallback unexpectedly aliases source inode: {source.relative}"
        )
    return {
        "mode": mode,
        "same_inode_as_source": same_inode,
        "size": destination_identity.size,
    }


def _plan_shards(
    tensors: Sequence[ResidentTensor],
    *,
    max_shard_bytes: int,
) -> tuple[_ShardPlan, ...]:
    if max_shard_bytes <= 0:
        raise CompactArtifactError("max_shard_bytes must be positive")
    if not tensors:
        raise CompactArtifactError("manifest contains no resident tensors")
    groups: list[list[ResidentTensor]] = []
    current: list[ResidentTensor] = []
    current_bytes = 0
    for tensor in tensors:
        if current and current_bytes + tensor.length > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += tensor.length
    if current:
        groups.append(current)
    count = len(groups)
    return tuple(
        _ShardPlan(
            name=f"model-resident-{index:05d}-of-{count:05d}.safetensors",
            tensors=tuple(group),
        )
        for index, group in enumerate(groups, 1)
    )


def _safetensors_header(
    tensors: Sequence[ResidentTensor],
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    header: dict[str, Any] = {}
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for tensor in tensors:
        start = cursor
        cursor += tensor.length
        header[tensor.tensor] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, cursor],
        }
        ranges[tensor.tensor] = (start, cursor)
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    if not 1 <= len(encoded) <= MAX_SAFETENSORS_HEADER_BYTES:
        raise CompactArtifactError("resident safetensors header is too large")
    prefix = len(encoded).to_bytes(8, "little")
    return prefix + encoded, ranges


def _write_resident_shard(
    root: Path,
    plan: _ShardPlan,
    sources: dict[str, _SourceFile],
    *,
    chunk_bytes: int,
) -> tuple[ShardInfo, tuple[ResidentTensor, ...]]:
    path = _destination_member(root, plan.name)
    header, ranges = _safetensors_header(plan.tensors)
    output_fd = _open_output(path)
    full_digest = hashlib.sha256()
    source_fds: dict[str, int] = {}
    output_tensors: list[ResidentTensor] = []
    try:
        _write_all(output_fd, header)
        full_digest.update(header)
        for tensor in plan.tensors:
            source = sources[tensor.shard]
            source_fd = source_fds.get(tensor.shard)
            if source_fd is None:
                source_fd = _open_readonly(source.path)
                source_fds[tensor.shard] = source_fd
            _copy_range(
                source_fd,
                output_fd,
                source_offset=tensor.offset,
                length=tensor.length,
                chunk_bytes=chunk_bytes,
                digest=full_digest,
            )
            relative_start, _relative_end = ranges[tensor.tensor]
            output_tensors.append(
                replace(
                    tensor,
                    shard=plan.name,
                    offset=len(header) + relative_start,
                )
            )
        os.fsync(output_fd)
    finally:
        for descriptor in source_fds.values():
            os.close(descriptor)
        os.close(output_fd)
    expected_size = len(header) + sum(tensor.length for tensor in plan.tensors)
    if path.stat().st_size != expected_size:
        raise CompactArtifactError(f"resident shard size mismatch: {plan.name}")
    shard = ShardInfo(
        name=plan.name,
        size=expected_size,
        header_bytes=len(header),
        header_sha256=hashlib.sha256(header).hexdigest(),
        sha256=full_digest.hexdigest(),
    )
    return shard, tuple(output_tensors)


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = _open_output(path)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _iter_source_files(root: Path) -> Iterator[str]:
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            candidate = directory_path / name
            if candidate.is_symlink():
                relative = candidate.relative_to(root).as_posix()
                raise CompactArtifactError(
                    f"source directory symlinks are not allowed: {relative}"
                )
        for name in files:
            candidate = directory_path / name
            yield candidate.relative_to(root).as_posix()


def _is_stale_artifact_control_file(relative: str) -> bool:
    """Return whether passthrough would preserve an obsolete build contract."""

    name = PurePosixPath(relative).name.lower()
    if "expert-manifest" in name or "expert_manifest" in name:
        return True
    if "conversion-checkpoint" in name or "compaction-checkpoint" in name:
        return True
    if name in {"conversion-provenance.json", "conversion-validation.json"}:
        return True
    if name.endswith("-handoff.json"):
        return True
    return name in {
        "conversion-checkpoint.jsonl",
        "compaction-checkpoint.jsonl",
        HANDOFF_NAME,
    }


def _sanitize_runtime_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_runtime_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, str) and key.endswith("_path") and isinstance(item, str):
            if key == "output_path":
                sanitized[key] = "."
            continue
        sanitized[key] = _sanitize_runtime_value(item)
    return sanitized


def _rewrite_runtime_contract(source: _SourceFile, destination: Path) -> None:
    if source.identity.size > MAX_RUNTIME_BYTES:
        raise CompactArtifactError(
            f"runtime contract exceeds {MAX_RUNTIME_BYTES} bytes"
        )
    descriptor = _open_readonly(source.path)
    try:
        payload = _read_exact(descriptor, 0, source.identity.size)
    finally:
        os.close(descriptor)
    if _identity(source.path) != source.identity:
        raise CompactArtifactError(
            f"source file changed while reading: {source.relative}"
        )
    try:
        loaded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactArtifactError(f"invalid {RUNTIME_NAME}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CompactArtifactError(f"{RUNTIME_NAME} must contain a JSON object")
    runtime = _sanitize_runtime_value(loaded)
    runtime["compaction_provenance"] = {
        "format": HANDOFF_FORMAT,
        "manifest": MANIFEST_NAME,
        "model_index": INDEX_NAME,
        "resident_only_shards": True,
    }
    _write_json(destination, runtime)


def _copy_passthrough_files(
    source_root: Path,
    output_root: Path,
    *,
    excluded: set[str],
    chunk_bytes: int,
) -> list[str]:
    copied: list[str] = []
    for relative in sorted(_iter_source_files(source_root)):
        relative = _safe_relative_name(relative, label="metadata path")
        if (
            relative in excluded
            or relative.endswith(".partial")
            or _is_stale_artifact_control_file(relative)
        ):
            continue
        if relative.endswith(".safetensors"):
            raise CompactArtifactError(
                f"unmanifested safetensors file would make output ambiguous: {relative}"
            )
        source = _capture_source_file(source_root, relative)
        destination = _destination_member(output_root, relative)
        if relative == RUNTIME_NAME:
            _rewrite_runtime_contract(source, destination)
        else:
            _copy_file_bounded(source, destination, chunk_bytes=chunk_bytes)
        copied.append(relative)
    return copied


def _read_exact(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    position = offset
    remaining = length
    while remaining:
        try:
            chunk = os.pread(fd, remaining, position)
        except InterruptedError:
            continue
        if not chunk:
            raise CompactArtifactError(f"short read at offset {position}")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _physical_tensor_names(path: Path) -> set[str]:
    fd = _open_readonly(path)
    try:
        size = os.fstat(fd).st_size
        length_raw = _read_exact(fd, 0, 8)
        header_length = int.from_bytes(length_raw, "little")
        if not 1 <= header_length <= MAX_SAFETENSORS_HEADER_BYTES:
            raise CompactArtifactError(f"invalid safetensors header: {path}")
        header = json.loads(_read_exact(fd, 8, header_length))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactArtifactError(
            f"invalid safetensors JSON in {path}: {exc}"
        ) from exc
    finally:
        os.close(fd)
    if not isinstance(header, dict):
        raise CompactArtifactError(f"safetensors header must be an object: {path}")
    names = {str(name) for name in header if name != "__metadata__"}
    if not names or size <= 8 + header_length:
        raise CompactArtifactError(f"safetensors file has no payload: {path}")
    return names


def _load_index(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactArtifactError(f"could not read compact index: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("weight_map"), dict):
        raise CompactArtifactError("compact index has no weight_map")
    result: dict[str, str] = {}
    for name, shard in value["weight_map"].items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise CompactArtifactError("compact index entries must be strings")
        result[name] = _safe_relative_name(shard, label="index shard")
    return result


def _audit_physical_inventory(
    root: Path,
    manifest: ExpertManifest,
) -> dict[str, Any]:
    resident_shards = {
        shard.name for shard in manifest.shards if shard.kind == "safetensors"
    }
    auxiliary_safetensors = {
        item.file
        for item in manifest.auxiliary_files
        if item.file.endswith(".safetensors")
    }
    physical = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.safetensors")
        if path.is_file()
    }
    expected_physical = resident_shards | auxiliary_safetensors
    if physical != expected_physical:
        raise CompactArtifactError(
            "physical safetensors inventory mismatch; "
            f"missing={sorted(expected_physical - physical)[:4]}, "
            f"extra={sorted(physical - expected_physical)[:4]}"
        )

    physical_names: set[str] = set()
    for shard in sorted(resident_shards):
        names = _physical_tensor_names(_destination_member(root, shard))
        overlap = physical_names & names
        if overlap:
            raise CompactArtifactError(
                f"resident tensors appear in multiple shards: {sorted(overlap)[:4]}"
            )
        physical_names.update(names)
    resident_names = {tensor.tensor for tensor in manifest.resident_tensors}
    routed_names = {
        segment.tensor for record in manifest.records for segment in record.segments
    }
    index = _load_index(root / INDEX_NAME)
    if physical_names != resident_names or set(index) != resident_names:
        raise CompactArtifactError(
            "resident physical/index inventories do not match the manifest"
        )
    if physical_names & routed_names or set(index) & routed_names:
        raise CompactArtifactError(
            "routed expert tensors remain in compact shards/index"
        )
    if set(index.values()) != resident_shards:
        raise CompactArtifactError(
            "compact index does not reference every resident shard"
        )
    return {
        "valid": True,
        "resident_shards": len(resident_shards),
        "resident_tensors": len(resident_names),
        "routed_tensors_in_shards": 0,
        "routed_tensors_in_index": 0,
        "physical_safetensors": sorted(physical),
    }


def _verify_resident_parity(
    source_root: Path,
    source_manifest: ExpertManifest,
    output_root: Path,
    output_manifest: ExpertManifest,
    *,
    chunk_bytes: int,
) -> dict[str, int | bool]:
    source_by_name = {
        tensor.tensor: tensor for tensor in source_manifest.resident_tensors
    }
    output_by_name = {
        tensor.tensor: tensor for tensor in output_manifest.resident_tensors
    }
    if set(source_by_name) != set(output_by_name):
        raise CompactArtifactError("resident parity inventory mismatch")
    source_fds: dict[str, int] = {}
    output_fds: dict[str, int] = {}
    checked_bytes = 0
    try:
        for name in sorted(source_by_name):
            source_tensor = source_by_name[name]
            output_tensor = output_by_name[name]
            if (
                source_tensor.length != output_tensor.length
                or source_tensor.dtype != output_tensor.dtype
                or source_tensor.shape != output_tensor.shape
            ):
                raise CompactArtifactError(f"resident metadata parity mismatch: {name}")
            source_fd = source_fds.get(source_tensor.shard)
            if source_fd is None:
                source_path = _resolve_source_member(source_root, source_tensor.shard)
                source_fd = _open_readonly(source_path)
                source_fds[source_tensor.shard] = source_fd
            output_fd = output_fds.get(output_tensor.shard)
            if output_fd is None:
                output_path = _destination_member(output_root, output_tensor.shard)
                output_fd = _open_readonly(output_path)
                output_fds[output_tensor.shard] = output_fd
            remaining = source_tensor.length
            source_offset = source_tensor.offset
            output_offset = output_tensor.offset
            while remaining:
                wanted = min(remaining, chunk_bytes)
                source_payload = _read_exact(source_fd, source_offset, wanted)
                output_payload = _read_exact(output_fd, output_offset, wanted)
                if source_payload != output_payload:
                    raise CompactArtifactError(f"resident byte parity mismatch: {name}")
                source_offset += wanted
                output_offset += wanted
                remaining -= wanted
                checked_bytes += wanted
    finally:
        for descriptor in (*source_fds.values(), *output_fds.values()):
            os.close(descriptor)
    return {
        "valid": True,
        "checked_tensors": len(source_by_name),
        "checked_bytes": checked_bytes,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_inputs(
    source_root: Path,
    manifest_path: Path,
    sidecar_path: Path,
    output_root: Path,
    *,
    chunk_bytes: int,
) -> tuple[Path, Path, Path, Path]:
    if chunk_bytes <= 0 or chunk_bytes > 64 * 1024 * 1024:
        raise CompactArtifactError("chunk_bytes must be between 1 and 67108864")
    if source_root.is_symlink():
        raise CompactArtifactError("source root may not be a symlink")
    source = source_root.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise CompactArtifactError(f"source root is not a directory: {source}")
    if manifest_path.is_symlink():
        raise CompactArtifactError("manifest path may not be a symlink")
    manifest = manifest_path.expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise CompactArtifactError(f"manifest is not a file: {manifest}")
    expanded_output = output_root.expanduser()
    lexical_output = Path(os.path.abspath(expanded_output))
    if source == lexical_output or source in lexical_output.parents:
        raise CompactArtifactError("output path must not be inside the source artifact")
    try:
        parent = expanded_output.parent.resolve(strict=True)
    except OSError as exc:
        raise CompactArtifactError(
            f"output parent must already exist: {expanded_output.parent}"
        ) from exc
    output = parent / expanded_output.name
    if output.exists() or output.is_symlink():
        raise CompactArtifactError(f"output path already exists: {output}")
    if source == output or source in output.parents:
        raise CompactArtifactError("output path must not be inside the source artifact")
    sidecar = sidecar_path.expanduser()
    if sidecar.is_symlink():
        raise CompactArtifactError("expert sidecar may not be a symlink")
    sidecar = sidecar.resolve(strict=True)
    return source, manifest, sidecar, output


def _manifest_relative_to_source(path: Path, source_root: Path) -> str | None:
    try:
        return path.relative_to(source_root).as_posix()
    except ValueError:
        return None


def compact_streamed_artifact(
    source_root: Path | str,
    manifest_path: Path | str,
    sidecar_path: Path | str,
    output_root: Path | str,
    *,
    allow_sidecar_copy: bool = False,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Build, fully verify, and atomically publish a compact streamed artifact.

    The source is never modified.  The returned handoff marks source cleanup as
    eligible only after record, shard, sidecar, resident-parity, index, and
    physical-inventory verification all succeed.
    """

    source, manifest_file, explicit_sidecar, output = _validate_inputs(
        Path(source_root),
        Path(manifest_path),
        Path(sidecar_path),
        Path(output_root),
        chunk_bytes=chunk_bytes,
    )
    manifest = load_expert_manifest(manifest_file)
    if manifest.sidecar is None:
        raise CompactArtifactError("complete manifest has no expert sidecar")
    if any(record.sha256 is None for record in manifest.records):
        raise CompactArtifactError("complete manifest requires every record hash")
    known_spec = _validate_known_model_spec(manifest)

    sidecar_source = _capture_source_file(
        source,
        manifest.sidecar.file,
        allow_hf_blob_symlink=False,
    )
    if explicit_sidecar != sidecar_source.path:
        raise CompactArtifactError(
            "explicit sidecar does not match the manifest-bound experts.bin"
        )
    if (
        sidecar_source.identity.size != manifest.sidecar.size
        or sidecar_source.relative != manifest.sidecar.file
    ):
        raise CompactArtifactError("expert sidecar size/path mismatch")

    model_sources = {
        shard.name: _capture_source_file(source, shard.name)
        for shard in manifest.shards
        if shard.kind == "safetensors"
    }
    auxiliary_sources = {
        item.file: _capture_source_file(source, item.file)
        for item in manifest.auxiliary_files
    }
    input_verification = verify_expert_manifest(
        manifest,
        source,
        verify_records=False,
        verify_shard_hashes=False,
        verify_sidecar_hash=False,
    )
    manifest, source_shard_hashes = _bind_source_shard_hashes(
        manifest,
        model_sources,
        chunk_bytes=chunk_bytes,
    )

    plans = _plan_shards(
        manifest.resident_tensors,
        max_shard_bytes=max_shard_bytes,
    )
    generated_names = {plan.name for plan in plans}
    generated_names.update({INDEX_NAME, MANIFEST_NAME, HANDOFF_NAME})
    conflicts = generated_names & (
        {manifest.sidecar.file} | {item.file for item in manifest.auxiliary_files}
    )
    if conflicts:
        raise CompactArtifactError(
            f"generated artifact paths collide with manifest files: {sorted(conflicts)}"
        )

    lock_path = output.parent / f".{output.name}.compaction.lock"
    lock_fd: int | None = None
    temporary: Path | None = None
    try:
        lock_fd = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        _write_all(lock_fd, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_fd)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.compacting-",
                dir=output.parent,
            )
        )

        sidecar_link = _link_or_copy(
            sidecar_source,
            _destination_member(temporary, manifest.sidecar.file),
            allow_copy=allow_sidecar_copy,
            chunk_bytes=chunk_bytes,
        )
        auxiliary_links: dict[str, dict[str, Any]] = {}
        for relative, source_file in sorted(auxiliary_sources.items()):
            auxiliary_links[relative] = _link_or_copy(
                source_file,
                _destination_member(temporary, relative),
                allow_copy=allow_sidecar_copy,
                chunk_bytes=chunk_bytes,
            )

        excluded = set(model_sources)
        excluded.update(auxiliary_sources)
        excluded.add(manifest.sidecar.file)
        excluded.update({INDEX_NAME, MANIFEST_NAME, HANDOFF_NAME})
        manifest_relative = _manifest_relative_to_source(manifest_file, source)
        if manifest_relative is not None:
            excluded.add(manifest_relative)
        metadata_files = _copy_passthrough_files(
            source,
            temporary,
            excluded=excluded,
            chunk_bytes=chunk_bytes,
        )

        output_shards: list[ShardInfo] = []
        output_residents: list[ResidentTensor] = []
        for plan in plans:
            shard, residents = _write_resident_shard(
                temporary,
                plan,
                model_sources,
                chunk_bytes=chunk_bytes,
            )
            output_shards.append(shard)
            output_residents.extend(residents)
        output_residents.sort(key=lambda tensor: tensor.tensor)

        weight_map = {tensor.tensor: tensor.shard for tensor in output_residents}
        _write_json(
            temporary / INDEX_NAME,
            {
                "metadata": {"total_size": manifest.resident_tensor_bytes},
                "weight_map": weight_map,
            },
        )

        intermediate = replace(
            manifest,
            shards=(*manifest.shards, *output_shards),
            resident_tensors=tuple(output_residents),
            manifest_sha256=None,
        ).with_digest()
        compact = make_sidecar_authoritative(intermediate)
        output_spec = _validate_known_model_spec(compact)
        if output_spec is not known_spec:
            raise CompactArtifactError(
                "model descriptor recognition changed during compaction"
            )
        compact = save_expert_manifest(compact, temporary / MANIFEST_NAME)

        resident_parity = _verify_resident_parity(
            source,
            manifest,
            temporary,
            compact,
            chunk_bytes=chunk_bytes,
        )
        output_verification = verify_expert_manifest(
            compact,
            temporary,
            verify_records=True,
            verify_shard_hashes=True,
            verify_sidecar_hash=True,
        )
        inventory = _audit_physical_inventory(temporary, compact)
        for source_file in model_sources.values():
            _assert_source_unchanged(source, source_file)
        for source_file in auxiliary_sources.values():
            _assert_source_unchanged(
                source,
                source_file,
                allow_link_ctime_change=True,
            )
        _assert_source_unchanged(
            source,
            sidecar_source,
            allow_link_ctime_change=True,
        )

        handoff: dict[str, Any] = {
            "format": HANDOFF_FORMAT,
            "model_key": compact.model_key,
            "source_repo": compact.source_repo,
            "source_revision": compact.source_revision,
            "source_root": str(source),
            "output_root": str(output),
            "manifest": MANIFEST_NAME,
            "manifest_sha256": compact.manifest_sha256,
            "sidecar": {
                "file": compact.sidecar.file if compact.sidecar else None,
                "sha256": compact.sidecar.sha256 if compact.sidecar else None,
                **sidecar_link,
            },
            "auxiliary_files": auxiliary_links,
            "metadata_files": metadata_files,
            "resident": {
                "tensor_count": len(compact.resident_tensors),
                "tensor_bytes": compact.resident_tensor_bytes,
                "shards": [shard.name for shard in output_shards],
            },
            "verification": {
                "input": input_verification,
                "source_shard_hashes": source_shard_hashes,
                "output": output_verification,
                "known_model_spec": {
                    "model_key": compact.model_key,
                    "verified": known_spec is not None,
                },
                "resident_parity": resident_parity,
                "physical_inventory": inventory,
            },
            "cleanup": {
                "eligible": True,
                "performed": False,
                "source_preserved": True,
                "source_model_shards": sorted(model_sources),
                "source_sidecar": manifest.sidecar.file,
                "instruction": (
                    "Review this handoff, load-test the compact artifact, then unlink "
                    "source files manually; this tool never deletes source data."
                ),
            },
        }
        _write_json(temporary / HANDOFF_NAME, handoff)
        _fsync_directory(temporary)
        if output.exists() or output.is_symlink():
            raise CompactArtifactError(f"output path appeared during build: {output}")
        os.rename(temporary, output)
        temporary = None
        _fsync_directory(output.parent)
        return handoff
    except BaseException:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "CompactArtifactError",
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_MAX_SHARD_BYTES",
    "HANDOFF_FORMAT",
    "compact_streamed_artifact",
]
