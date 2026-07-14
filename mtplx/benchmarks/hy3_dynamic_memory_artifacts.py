"""Frozen external-artifact attestation for the issue #46 hardware campaign."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mtplx.benchmarks.runners.hy3_dynamic_memory import canonical_sha256
from mtplx.expert_manifest import (
    ExpertManifest,
    ExpertManifestError,
    resolve_artifact_member,
)


SCHEMA_ARTIFACT_ATTESTATION = "mtplx-hy3-artifact-attestation-v1"


class ArtifactAttestationError(ValueError):
    """Raised when a pinned external artifact cannot be proven exactly."""


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactAttestationError(
            f"{field} must be exactly 64 lowercase hexadecimal digits"
        )
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactAttestationError(f"{field} must be a nonempty string")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactAttestationError(f"{field} must be a positive integer")
    return value


def _safe_member(value: object, *, field: str) -> str:
    text = _nonempty_string(value, field=field)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ArtifactAttestationError(f"{field} must be a safe relative path")
    return text


@dataclass(frozen=True)
class ArtifactPins:
    model_config_sha256: str
    manifest_file_sha256: str
    manifest_sha256: str
    sidecar_file: str
    sidecar_bytes: int
    sidecar_sha256: str
    resident_payload_bytes: int
    resident_payload_sha256: str
    source_revision: str

    @classmethod
    def from_mapping(cls, value: object) -> ArtifactPins:
        if not isinstance(value, Mapping):
            raise ArtifactAttestationError("artifact_pins must be an object")
        required = {
            "model_config_sha256",
            "manifest_file_sha256",
            "manifest_sha256",
            "sidecar_file",
            "sidecar_bytes",
            "sidecar_sha256",
            "resident_payload_bytes",
            "resident_payload_sha256",
            "source_revision",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required)
            raise ArtifactAttestationError(
                f"artifact_pins keys differ (missing={missing}, unknown={unknown})"
            )
        return cls(
            model_config_sha256=_sha256(
                value["model_config_sha256"], field="artifact_pins.model_config_sha256"
            ),
            manifest_file_sha256=_sha256(
                value["manifest_file_sha256"],
                field="artifact_pins.manifest_file_sha256",
            ),
            manifest_sha256=_sha256(
                value["manifest_sha256"], field="artifact_pins.manifest_sha256"
            ),
            sidecar_file=_safe_member(
                value["sidecar_file"], field="artifact_pins.sidecar_file"
            ),
            sidecar_bytes=_positive_int(
                value["sidecar_bytes"], field="artifact_pins.sidecar_bytes"
            ),
            sidecar_sha256=_sha256(
                value["sidecar_sha256"], field="artifact_pins.sidecar_sha256"
            ),
            resident_payload_bytes=_positive_int(
                value["resident_payload_bytes"],
                field="artifact_pins.resident_payload_bytes",
            ),
            resident_payload_sha256=_sha256(
                value["resident_payload_sha256"],
                field="artifact_pins.resident_payload_sha256",
            ),
            source_revision=_nonempty_string(
                value["source_revision"], field="artifact_pins.source_revision"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_config_sha256": self.model_config_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sidecar_file": self.sidecar_file,
            "sidecar_bytes": self.sidecar_bytes,
            "sidecar_sha256": self.sidecar_sha256,
            "resident_payload_bytes": self.resident_payload_bytes,
            "resident_payload_sha256": self.resident_payload_sha256,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True)
class ArtifactFileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> ArtifactFileFingerprint:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class Hy3ArtifactAttestation:
    manifest: ExpertManifest
    model_config: dict[str, Any]
    model_artifact_sha256: str
    manifest_file_sha256: str
    artifact_pins_sha256: str
    artifact_stat_sha256: str
    sidecar_fingerprint: ArtifactFileFingerprint
    resident_payload_bytes: int
    resident_payload_sha256: str
    resident_shard_fingerprints: tuple[dict[str, int | str], ...]
    payload_hash_verified: bool
    payload_hash_io_mode: str

    def evidence(self) -> dict[str, object]:
        return {
            "schema": SCHEMA_ARTIFACT_ATTESTATION,
            "model_artifact_sha256": self.model_artifact_sha256,
            "expert_manifest_sha256": self.manifest_file_sha256,
            "artifact_pins_sha256": self.artifact_pins_sha256,
            "artifact_stat_sha256": self.artifact_stat_sha256,
            "sidecar_fingerprint": self.sidecar_fingerprint.to_dict(),
            "resident_payload_bytes": self.resident_payload_bytes,
            "resident_payload_sha256": self.resident_payload_sha256,
            "resident_shard_fingerprints": [
                dict(value) for value in self.resident_shard_fingerprints
            ],
            "payload_hash_verified": self.payload_hash_verified,
            "payload_hash_io_mode": self.payload_hash_io_mode,
        }


def _fingerprint(value: os.stat_result, *, field: str) -> ArtifactFileFingerprint:
    if not stat.S_ISREG(value.st_mode):
        raise ArtifactAttestationError(f"{field} is not a regular file")
    return ArtifactFileFingerprint.from_stat(value)


def _open_resolved(path: Path, *, field: str) -> int:
    try:
        resolved = path.expanduser().resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ArtifactAttestationError(f"cannot open {field}: {exc}") from exc
    try:
        _fingerprint(os.fstat(descriptor), field=field)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_frozen(path: Path, *, field: str) -> bytes:
    descriptor = _open_resolved(path, field=field)
    try:
        before = _fingerprint(os.fstat(descriptor), field=field)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(descriptor, 8 * 1024**2)
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
        after = _fingerprint(os.fstat(descriptor), field=field)
    finally:
        os.close(descriptor)
    if before != after:
        raise ArtifactAttestationError(f"{field} changed while it was read")
    return b"".join(chunks)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactAttestationError(f"pinned JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _parse_manifest(payload: bytes, *, source: Path) -> ExpertManifest:
    try:
        value = json.loads(payload, object_pairs_hook=_strict_pairs)
        return ExpertManifest.from_dict(value, verify_digest=True)
    except ArtifactAttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ExpertManifestError) as exc:
        raise ArtifactAttestationError(
            f"cannot parse pinned manifest {source}: {exc}"
        ) from exc


def _parse_model_config(payload: bytes, *, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_strict_pairs)
    except ArtifactAttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactAttestationError(
            f"cannot parse pinned model config {source}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactAttestationError("pinned model config must be a JSON object")
    return dict(value)


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        try:
            chunk = os.read(descriptor, 8 * 1024**2)
        except InterruptedError:
            continue
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _enable_no_cache(descriptor: int, *, field: str) -> None:
    try:
        import fcntl

        operation = getattr(fcntl, "F_NOCACHE")
        fcntl.fcntl(descriptor, operation, 1)
    except (AttributeError, ImportError, OSError) as exc:
        raise ArtifactAttestationError(
            f"cannot enable F_NOCACHE for {field}: {exc}"
        ) from exc


def _pread_hash_range(
    descriptor: int,
    *,
    offset: int,
    length: int,
    digest: Any,
    field: str,
) -> None:
    cursor = 0
    while cursor < length:
        requested = min(8 * 1024**2, length - cursor)
        try:
            payload = os.pread(descriptor, requested, offset + cursor)
        except InterruptedError:
            continue
        if len(payload) != requested:
            raise ArtifactAttestationError(
                f"short read while hashing {field}: expected {requested}, "
                f"got {len(payload)}"
            )
        digest.update(payload)
        cursor += requested


def _resident_shard_sizes(manifest: ExpertManifest) -> dict[str, int]:
    shard_sizes = {str(shard.name): int(shard.size) for shard in manifest.shards}
    referenced = {str(tensor.shard) for tensor in manifest.resident_tensors}
    missing = sorted(referenced - set(shard_sizes))
    if missing:
        raise ArtifactAttestationError(
            f"resident tensors reference unmanifested shards {missing}"
        )
    return {name: shard_sizes[name] for name in sorted(referenced)}


def _open_resident_shards(
    model_root: Path,
    manifest: ExpertManifest,
) -> dict[str, tuple[int, ArtifactFileFingerprint]]:
    opened: dict[str, tuple[int, ArtifactFileFingerprint]] = {}
    try:
        for name, expected_size in _resident_shard_sizes(manifest).items():
            try:
                path = resolve_artifact_member(model_root, name)
            except ExpertManifestError as exc:
                raise ArtifactAttestationError(str(exc)) from exc
            descriptor = _open_resolved(path, field=f"resident shard {name}")
            fingerprint = _fingerprint(
                os.fstat(descriptor), field=f"resident shard {name}"
            )
            if fingerprint.size != expected_size:
                os.close(descriptor)
                raise ArtifactAttestationError(
                    f"resident shard {name} size differs from its manifest"
                )
            opened[name] = (descriptor, fingerprint)
    except BaseException:
        for descriptor, _fingerprint_value in opened.values():
            os.close(descriptor)
        raise
    return opened


def _resident_fingerprint_evidence(
    opened: Mapping[str, tuple[int, ArtifactFileFingerprint]],
) -> tuple[dict[str, int | str], ...]:
    return tuple(
        {"name": name, **fingerprint.to_dict()}
        for name, (_descriptor, fingerprint) in sorted(opened.items())
    )


def _hash_resident_payload(
    manifest: ExpertManifest,
    opened: Mapping[str, tuple[int, ArtifactFileFingerprint]],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for tensor in manifest.resident_tensors:
        length = int(tensor.length)
        _pread_hash_range(
            opened[str(tensor.shard)][0],
            offset=int(tensor.offset),
            length=length,
            digest=digest,
            field=f"resident tensor {tensor.tensor}",
        )
        total += length
    return total, digest.hexdigest()


def _current_fingerprint(path: Path, *, field: str) -> ArtifactFileFingerprint:
    descriptor = _open_resolved(path, field=field)
    try:
        return _fingerprint(os.fstat(descriptor), field=field)
    finally:
        os.close(descriptor)


def _require_logical_path_fingerprint(
    *,
    model_root: Path,
    member: str,
    expected: ArtifactFileFingerprint,
    field: str,
) -> None:
    try:
        current_path = resolve_artifact_member(model_root, member)
    except ExpertManifestError as exc:
        raise ArtifactAttestationError(str(exc)) from exc
    current = _current_fingerprint(current_path, field=field)
    if current != expected:
        raise ArtifactAttestationError(f"{field} path changed during attestation")


def attest_hy3_artifact(
    *,
    model_root: Path,
    manifest_path: Path,
    pins: ArtifactPins,
    verify_payload_hash: bool,
    require_f_nocache: bool = False,
) -> Hy3ArtifactAttestation:
    """Bind config, manifest, and one exact sidecar file instance to pins."""

    if require_f_nocache and not verify_payload_hash:
        raise ArtifactAttestationError(
            "require_f_nocache requires full payload verification"
        )

    config_path = model_root / "config.json"
    config_payload = _read_frozen(config_path, field="model config")
    manifest_payload = _read_frozen(manifest_path, field="expert manifest")
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    manifest_file_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if config_sha256 != pins.model_config_sha256:
        raise ArtifactAttestationError("model config differs from its pinned SHA-256")
    if manifest_file_sha256 != pins.manifest_file_sha256:
        raise ArtifactAttestationError(
            "expert manifest differs from its pinned SHA-256"
        )
    model_config = _parse_model_config(
        config_payload,
        source=model_root / "config.json",
    )
    manifest = _parse_manifest(manifest_payload, source=manifest_path)
    sidecar = manifest.sidecar
    if manifest.model_key != "hy3-q4":
        raise ArtifactAttestationError("pinned manifest is not Hy3 Q4")
    if manifest.manifest_sha256 != pins.manifest_sha256:
        raise ArtifactAttestationError("manifest canonical digest differs from its pin")
    if manifest.source_revision != pins.source_revision:
        raise ArtifactAttestationError("manifest source revision differs from its pin")
    if sidecar is None:
        raise ArtifactAttestationError("pinned manifest has no expert sidecar")
    if (
        sidecar.file != pins.sidecar_file
        or int(sidecar.size) != pins.sidecar_bytes
        or sidecar.sha256 != pins.sidecar_sha256
    ):
        raise ArtifactAttestationError("manifest sidecar descriptor differs from pins")
    if int(manifest.resident_tensor_bytes) != pins.resident_payload_bytes:
        raise ArtifactAttestationError(
            "manifest resident tensor bytes differ from their pin"
        )
    try:
        sidecar_path = resolve_artifact_member(model_root, pins.sidecar_file)
    except ExpertManifestError as exc:
        raise ArtifactAttestationError(str(exc)) from exc
    descriptor = _open_resolved(sidecar_path, field="expert sidecar")
    resident_shards: dict[str, tuple[int, ArtifactFileFingerprint]] = {}
    try:
        before = _fingerprint(os.fstat(descriptor), field="expert sidecar")
        if before.size != pins.sidecar_bytes:
            raise ArtifactAttestationError("expert sidecar size differs from its pin")
        resident_shards = _open_resident_shards(model_root, manifest)
        resident_fingerprints = _resident_fingerprint_evidence(resident_shards)
        if verify_payload_hash and require_f_nocache:
            _enable_no_cache(descriptor, field="expert sidecar")
            for name, (resident_descriptor, _fingerprint_value) in sorted(
                resident_shards.items()
            ):
                _enable_no_cache(
                    resident_descriptor,
                    field=f"resident shard {name}",
                )
        payload_sha256 = _hash_descriptor(descriptor) if verify_payload_hash else None
        resident_payload = (
            _hash_resident_payload(manifest, resident_shards)
            if verify_payload_hash
            else None
        )
        after = _fingerprint(os.fstat(descriptor), field="expert sidecar")
        for name, (resident_descriptor, resident_before) in resident_shards.items():
            resident_after = _fingerprint(
                os.fstat(resident_descriptor), field=f"resident shard {name}"
            )
            if resident_after != resident_before:
                raise ArtifactAttestationError(
                    f"resident shard {name} changed during attestation"
                )
    finally:
        os.close(descriptor)
        for resident_descriptor, _fingerprint_value in resident_shards.values():
            os.close(resident_descriptor)
    if before != after:
        raise ArtifactAttestationError("expert sidecar changed during attestation")
    if payload_sha256 is not None and payload_sha256 != pins.sidecar_sha256:
        raise ArtifactAttestationError("expert sidecar payload differs from its pin")
    if resident_payload is not None:
        resident_bytes, resident_sha256 = resident_payload
        if (
            resident_bytes != pins.resident_payload_bytes
            or resident_sha256 != pins.resident_payload_sha256
        ):
            raise ArtifactAttestationError(
                "resident tensor payload differs from its pin"
            )
    if verify_payload_hash:
        if _read_frozen(config_path, field="model config") != config_payload:
            raise ArtifactAttestationError("model config changed during attestation")
        if _read_frozen(manifest_path, field="expert manifest") != manifest_payload:
            raise ArtifactAttestationError("expert manifest changed during attestation")
        _require_logical_path_fingerprint(
            model_root=model_root,
            member=pins.sidecar_file,
            expected=before,
            field="expert sidecar",
        )
        resident_by_name = {
            str(value["name"]): ArtifactFileFingerprint(
                device=int(value["device"]),
                inode=int(value["inode"]),
                size=int(value["size"]),
                mtime_ns=int(value["mtime_ns"]),
                ctime_ns=int(value["ctime_ns"]),
            )
            for value in resident_fingerprints
        }
        for name, resident_before in resident_by_name.items():
            _require_logical_path_fingerprint(
                model_root=model_root,
                member=name,
                expected=resident_before,
                field=f"resident shard {name}",
            )
    artifact = {
        "config_sha256": config_sha256,
        "expert_manifest_sha256": manifest_file_sha256,
        "expert_payload_sha256": pins.sidecar_sha256,
        "expert_payload_bytes": pins.sidecar_bytes,
        "resident_payload_sha256": pins.resident_payload_sha256,
        "resident_payload_bytes": pins.resident_payload_bytes,
        "source_revision": pins.source_revision,
    }
    artifact_stat_sha256 = canonical_sha256(
        {
            "sidecar": before.to_dict(),
            "resident_shards": list(resident_fingerprints),
        }
    )
    return Hy3ArtifactAttestation(
        manifest=manifest,
        model_config=model_config,
        model_artifact_sha256=canonical_sha256(artifact),
        manifest_file_sha256=manifest_file_sha256,
        artifact_pins_sha256=canonical_sha256(pins.to_dict()),
        artifact_stat_sha256=artifact_stat_sha256,
        sidecar_fingerprint=before,
        resident_payload_bytes=pins.resident_payload_bytes,
        resident_payload_sha256=pins.resident_payload_sha256,
        resident_shard_fingerprints=resident_fingerprints,
        payload_hash_verified=verify_payload_hash,
        payload_hash_io_mode=(
            "f-nocache"
            if verify_payload_hash and require_f_nocache
            else "buffered"
            if verify_payload_hash
            else "not-run"
        ),
    )


__all__ = [
    "ArtifactAttestationError",
    "ArtifactPins",
    "Hy3ArtifactAttestation",
    "SCHEMA_ARTIFACT_ATTESTATION",
    "attest_hy3_artifact",
]
