from __future__ import annotations

import hashlib
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "probe_hy3_direct_cache.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "probe_hy3_direct_cache",
        _SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_sidecar_record_copies_one_contiguous_record_exactly(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "experts.bin").write_bytes(b"prefix" + b"abcdefgh")
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="experts.bin"))
    record = SimpleNamespace(
        sidecar_offset=6,
        sidecar_length=8,
        logical_bytes=8,
        sha256=hashlib.sha256(b"abcdefgh").hexdigest(),
        segments=(SimpleNamespace(length=3), SimpleNamespace(length=5)),
    )
    buffer = bytearray(8)

    module.load_sidecar_record(tmp_path, manifest, record, buffer)

    assert bytes(buffer) == b"abcdefgh"


def test_probe_buffer_identity_does_not_retain_the_record() -> None:
    module = _load_module()

    class RecordBuffer:
        pass

    buffer = RecordBuffer()
    reference = weakref.ref(buffer)
    identity = module._buffer_identity(buffer)

    del buffer

    assert reference() is None
    assert isinstance(identity, int)


def test_load_sidecar_record_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "experts.bin").write_bytes(b"tampered")
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="experts.bin"))
    record = SimpleNamespace(
        sidecar_offset=0,
        sidecar_length=8,
        logical_bytes=8,
        sha256=hashlib.sha256(b"expected").hexdigest(),
        segments=(SimpleNamespace(length=8),),
    )

    with pytest.raises(RuntimeError, match="record hash mismatch"):
        module.load_sidecar_record(tmp_path, manifest, record, bytearray(8))


def test_load_sidecar_record_rejects_short_reads(tmp_path: Path) -> None:
    module = _load_module()
    (tmp_path / "experts.bin").write_bytes(b"abc")
    manifest = SimpleNamespace(sidecar=SimpleNamespace(file="experts.bin"))
    record = SimpleNamespace(
        sidecar_offset=0,
        sidecar_length=8,
        logical_bytes=8,
        sha256=hashlib.sha256(b"abcdefgh").hexdigest(),
        segments=(SimpleNamespace(length=8),),
    )

    with pytest.raises(RuntimeError, match="short sidecar read"):
        module.load_sidecar_record(tmp_path, manifest, record, bytearray(8))


def test_probe_artifact_attestation_hashes_payload_once_and_verifies_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest = object()
    binding = {
        "model_artifact_sha256": "a" * 64,
        "manifest_file_sha256": "b" * 64,
        "artifact_pins_sha256": "c" * 64,
        "artifact_stat_sha256": "d" * 64,
        "resident_payload_bytes": 1,
        "resident_payload_sha256": "e" * 64,
    }
    attestation = SimpleNamespace(manifest=manifest, **binding)
    final_attestation = SimpleNamespace(manifest=object(), **binding)
    calls: list[object] = []
    config = SimpleNamespace(
        model_root=tmp_path,
        manifest=tmp_path / "manifest.json",
        artifact_pins=object(),
    )

    def attest(**kwargs):
        calls.append(("attest", kwargs))
        return attestation if kwargs["verify_payload_hash"] else final_attestation

    def verify(actual_manifest, actual_root, *, verify_sidecar_hash=False):
        calls.append(("verify", actual_manifest, actual_root, verify_sidecar_hash))
        return {"valid": True, "checked_shards": 2}

    monkeypatch.setattr(module, "attest_hy3_artifact", attest)
    monkeypatch.setattr(module, "verify_expert_manifest", verify, raising=False)

    assert module._attest_probe_artifact(config) is attestation

    assert calls == [
        (
            "attest",
            {
                "model_root": tmp_path,
                "manifest_path": tmp_path / "manifest.json",
                "pins": config.artifact_pins,
                "verify_payload_hash": True,
                "require_f_nocache": True,
            },
        ),
        ("verify", manifest, tmp_path, False),
        (
            "attest",
            {
                "model_root": tmp_path,
                "manifest_path": tmp_path / "manifest.json",
                "pins": config.artifact_pins,
                "verify_payload_hash": False,
            },
        ),
    ]


def test_probe_artifact_attestation_rejects_drift_during_header_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    config = SimpleNamespace(
        model_root=tmp_path,
        manifest=tmp_path / "manifest.json",
        artifact_pins=object(),
    )
    bindings = iter(("a" * 64, "f" * 64))

    def attest(**kwargs):
        return SimpleNamespace(
            manifest=object(),
            model_artifact_sha256=next(bindings),
            manifest_file_sha256="b" * 64,
            artifact_pins_sha256="c" * 64,
            artifact_stat_sha256="d" * 64,
            resident_payload_bytes=1,
            resident_payload_sha256="e" * 64,
        )

    monkeypatch.setattr(module, "attest_hy3_artifact", attest)
    monkeypatch.setattr(
        module,
        "verify_expert_manifest",
        lambda *_args, **_kwargs: {"valid": True},
        raising=False,
    )

    with pytest.raises(RuntimeError, match="header verification"):
        module._attest_probe_artifact(config)


def test_probe_artifact_attestation_rejects_header_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    config = SimpleNamespace(
        model_root=tmp_path,
        manifest=tmp_path / "manifest.json",
        artifact_pins=object(),
    )
    monkeypatch.setattr(
        module,
        "attest_hy3_artifact",
        lambda **_kwargs: SimpleNamespace(manifest=object()),
    )
    monkeypatch.setattr(
        module,
        "verify_expert_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad header")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="bad header"):
        module._attest_probe_artifact(config)


def test_probe_arm_config_pins_exact_full_history_q4_attention() -> None:
    module = _load_module()
    runtime_config = SimpleNamespace(
        to_dict=lambda: {
            "memory_limit_bytes": 100 * 1024**3,
            "slot_layout": "direct-slots",
            "dynamic_expert_cache": True,
        },
    )

    arm_config = module._probe_arm_config(
        runtime_config,
        SimpleNamespace(persistent_slots=9_792),
    )

    assert arm_config == {
        "dynamic_memory": True,
        "attention_runtime_env": dict(module.HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV),
        "expert_streaming_config": {
            "memory_limit_bytes": 100 * 1024**3,
            "slot_layout": "direct-slots",
            "dynamic_expert_cache": True,
        },
        "planned_persistent_slots": 9_792,
    }


def test_probe_identity_hashes_exact_dynamic_and_normalized_configs() -> None:
    module = _load_module()
    arm_config = {
        "dynamic_memory": True,
        "expert_streaming_config": {
            "slot_layout": "direct-slots",
            "dynamic_expert_cache": True,
        },
        "total_context_tokens": 131_072,
    }

    identity = module.build_probe_identity(
        model_artifact_id="pipenetwork/Hy3-4bit@revision",
        model_artifact_sha256="a" * 64,
        expert_manifest_id="expert-manifest-sidecar.json",
        expert_manifest_sha256="b" * 64,
        artifact_pins_sha256="d" * 64,
        artifact_stat_sha256="e" * 64,
        resident_payload_bytes=3,
        resident_payload_sha256="f" * 64,
        source_git_commit="c" * 40,
        arm_config=arm_config,
    )

    assert identity["arm_config"] == arm_config
    assert identity["resident_payload_bytes"] == 3
    assert identity["resident_payload_sha256"] == "f" * 64
    assert identity["normalized_config"] == {
        "expert_streaming_config": {"slot_layout": "direct-slots"},
        "total_context_tokens": 131_072,
    }
    assert identity["arm_config_sha256"] == module.canonical_sha256(arm_config)
    assert identity["normalized_config_sha256"] == module.canonical_sha256(
        identity["normalized_config"]
    )
