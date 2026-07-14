from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import mtplx.benchmarks.hy3_dynamic_memory_artifacts as artifact_module
from mtplx.benchmarks.hy3_dynamic_memory_artifacts import (
    ArtifactAttestationError,
    ArtifactPins,
    attest_hy3_artifact,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, ArtifactPins]:
    root = tmp_path / "model"
    root.mkdir()
    config_payload = b'{"model_type":"hy_v3"}\n'
    manifest_payload = b'{"manifest":"frozen"}\n'
    sidecar_payload = b"exact streamed expert payload"
    resident_shard_payload = b"headerresident payloadtail"
    resident_payload = b"resident payload"
    (root / "config.json").write_bytes(config_payload)
    manifest_path = root / "expert-manifest-sidecar.json"
    manifest_path.write_bytes(manifest_payload)
    sidecar_path = root / "experts.bin"
    sidecar_path.write_bytes(sidecar_payload)
    resident_shard_path = root / "model-00001-of-00001.safetensors"
    resident_shard_path.write_bytes(resident_shard_payload)
    pins = ArtifactPins.from_mapping(
        {
            "model_config_sha256": _sha(config_payload),
            "manifest_file_sha256": _sha(manifest_payload),
            "manifest_sha256": "c" * 64,
            "sidecar_file": "experts.bin",
            "sidecar_bytes": len(sidecar_payload),
            "sidecar_sha256": _sha(sidecar_payload),
            "resident_payload_bytes": len(resident_payload),
            "resident_payload_sha256": _sha(resident_payload),
            "source_revision": "revision-a",
        }
    )
    manifest = SimpleNamespace(
        model_key="hy3-q4",
        manifest_sha256=pins.manifest_sha256,
        source_revision=pins.source_revision,
        sidecar=SimpleNamespace(
            file=pins.sidecar_file,
            size=pins.sidecar_bytes,
            sha256=pins.sidecar_sha256,
        ),
        resident_tensor_bytes=len(resident_payload),
        resident_tensors=(
            SimpleNamespace(
                tensor="resident.weight",
                shard=resident_shard_path.name,
                offset=len(b"header"),
                length=len(resident_payload),
            ),
        ),
        shards=(
            SimpleNamespace(
                name=resident_shard_path.name,
                size=len(resident_shard_payload),
            ),
        ),
    )
    monkeypatch.setattr(
        artifact_module.ExpertManifest,
        "from_dict",
        lambda *_args, **_kwargs: manifest,
    )
    return root, manifest_path, pins


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_config_sha256", "A" * 64),
        ("manifest_file_sha256", "b" * 63),
        ("manifest_sha256", "not-a-hash"),
        ("sidecar_file", "../experts.bin"),
        ("sidecar_bytes", True),
        ("sidecar_sha256", "z" * 64),
        ("resident_payload_bytes", 0),
        ("resident_payload_sha256", "z" * 64),
        ("source_revision", ""),
    ),
)
def test_artifact_pins_reject_ambiguous_values(field: str, value: object) -> None:
    raw: dict[str, object] = {
        "model_config_sha256": "a" * 64,
        "manifest_file_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "sidecar_file": "experts.bin",
        "sidecar_bytes": 1,
        "sidecar_sha256": "d" * 64,
        "resident_payload_bytes": 1,
        "resident_payload_sha256": "e" * 64,
        "source_revision": "revision-a",
    }
    raw[field] = value

    with pytest.raises(ArtifactAttestationError):
        ArtifactPins.from_mapping(raw)


def test_attestation_hashes_payload_only_when_explicitly_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    calls = 0
    real_hash = artifact_module._hash_descriptor

    def count_hash(descriptor: int) -> str:
        nonlocal calls
        calls += 1
        return real_hash(descriptor)

    monkeypatch.setattr(artifact_module, "_hash_descriptor", count_hash)

    cheap = attest_hy3_artifact(
        model_root=root,
        manifest_path=manifest_path,
        pins=pins,
        verify_payload_hash=False,
    )
    full = attest_hy3_artifact(
        model_root=root,
        manifest_path=manifest_path,
        pins=pins,
        verify_payload_hash=True,
    )

    assert calls == 1
    assert cheap.payload_hash_verified is False
    assert full.payload_hash_verified is True
    assert full.model_config == {"model_type": "hy_v3"}
    assert full.resident_payload_sha256 == pins.resident_payload_sha256
    assert "model_config" not in full.evidence()
    assert cheap.model_artifact_sha256 == full.model_artifact_sha256
    assert cheap.artifact_stat_sha256 == full.artifact_stat_sha256
    assert cheap.payload_hash_io_mode == "not-run"
    assert full.payload_hash_io_mode == "buffered"


def test_attestation_rejects_resident_payload_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    resident = root / "model-00001-of-00001.safetensors"
    resident.write_bytes(b"headertampered payloadtail")

    with pytest.raises(ArtifactAttestationError, match="resident tensor payload"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
        )


def test_attestation_rechecks_small_pins_after_the_long_payload_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    real_hash = artifact_module._hash_descriptor

    def mutate_config(descriptor: int) -> str:
        result = real_hash(descriptor)
        (root / "config.json").write_bytes(b'{"model_type":"tampered"}\n')
        return result

    monkeypatch.setattr(artifact_module, "_hash_descriptor", mutate_config)

    with pytest.raises(ArtifactAttestationError, match="changed during attestation"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
        )


def test_attestation_rechecks_manifest_after_the_long_payload_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    real_hash = artifact_module._hash_descriptor

    def mutate_manifest(descriptor: int) -> str:
        result = real_hash(descriptor)
        manifest_path.write_bytes(b'{"manifest":"tampered"}\n')
        return result

    monkeypatch.setattr(artifact_module, "_hash_descriptor", mutate_manifest)

    with pytest.raises(ArtifactAttestationError, match="changed during attestation"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
        )


def test_attestation_rechecks_sidecar_path_after_hashing_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    good = root / "good.bin"
    bad = root / "bad.bin"
    (root / "experts.bin").replace(good)
    bad.write_bytes(b"exact streamed expert payload")
    (root / "experts.bin").symlink_to(good.name)
    real_hash = artifact_module._hash_descriptor

    def replace_sidecar(descriptor: int) -> str:
        result = real_hash(descriptor)
        (root / "experts.bin").unlink()
        (root / "experts.bin").symlink_to(bad.name)
        return result

    monkeypatch.setattr(artifact_module, "_hash_descriptor", replace_sidecar)

    with pytest.raises(ArtifactAttestationError, match="path changed"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
        )


def test_full_attestation_can_require_f_nocache_before_payload_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    enabled: list[str] = []
    monkeypatch.setattr(
        artifact_module,
        "_enable_no_cache",
        lambda _descriptor, *, field: enabled.append(field),
    )

    result = attest_hy3_artifact(
        model_root=root,
        manifest_path=manifest_path,
        pins=pins,
        verify_payload_hash=True,
        require_f_nocache=True,
    )

    assert result.payload_hash_io_mode == "f-nocache"
    assert enabled == [
        "expert sidecar",
        "resident shard model-00001-of-00001.safetensors",
    ]


def test_required_f_nocache_failure_rejects_before_payload_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        artifact_module,
        "_enable_no_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ArtifactAttestationError("F_NOCACHE unavailable")
        ),
    )
    monkeypatch.setattr(
        artifact_module,
        "_hash_descriptor",
        lambda _descriptor: pytest.fail("payload reads must not begin"),
    )

    with pytest.raises(ArtifactAttestationError, match="F_NOCACHE"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
            require_f_nocache=True,
        )


def test_attestation_rechecks_resident_shard_logical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    logical = root / "model-00001-of-00001.safetensors"
    good = root / "resident-good.safetensors"
    bad = root / "resident-bad.safetensors"
    logical.replace(good)
    bad.write_bytes(b"headerresident payloadtail")
    logical.symlink_to(good.name)
    real_hash = artifact_module._hash_resident_payload

    def retarget(*args, **kwargs):
        result = real_hash(*args, **kwargs)
        logical.unlink()
        logical.symlink_to(bad.name)
        return result

    monkeypatch.setattr(artifact_module, "_hash_resident_payload", retarget)

    with pytest.raises(ArtifactAttestationError, match="path changed"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
        )


def test_attestation_rejects_small_file_drift_before_expensive_payload_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    (root / "config.json").write_bytes(b"tampered")
    monkeypatch.setattr(
        artifact_module,
        "_hash_descriptor",
        lambda _descriptor: pytest.fail("payload hash must not run after config drift"),
    )

    with pytest.raises(ArtifactAttestationError, match="model config"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=True,
        )


def test_attestation_rejects_ambiguous_model_config_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    payload = b'{"model_type":"hy_v3","model_type":"other"}\n'
    (root / "config.json").write_bytes(payload)
    pins = ArtifactPins(
        **{
            **pins.to_dict(),
            "model_config_sha256": _sha(payload),
        }
    )

    with pytest.raises(ArtifactAttestationError, match="duplicate key"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=False,
        )


def test_attestation_rejects_non_object_model_config_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path, pins = _fixture(tmp_path, monkeypatch)
    payload = b'["hy_v3"]\n'
    (root / "config.json").write_bytes(payload)
    pins = ArtifactPins(
        **{
            **pins.to_dict(),
            "model_config_sha256": _sha(payload),
        }
    )

    with pytest.raises(ArtifactAttestationError, match="JSON object"):
        attest_hy3_artifact(
            model_root=root,
            manifest_path=manifest_path,
            pins=pins,
            verify_payload_hash=False,
        )
