from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "probe_hy3_component_slabs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "probe_hy3_component_slabs",
        _SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Slot:
    def __init__(self, sizes: tuple[int, ...]) -> None:
        self.parts = tuple(bytearray(size) for size in sizes)

    def record_views(self, _record):
        return tuple(memoryview(part) for part in self.parts)


def test_load_sidecar_record_copies_every_component_exactly(tmp_path: Path) -> None:
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
    slot = _Slot((3, 5))

    module.load_sidecar_record(tmp_path, manifest, record, slot)

    assert bytes(slot.parts[0]) == b"abc"
    assert bytes(slot.parts[1]) == b"defgh"


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
        module.load_sidecar_record(tmp_path, manifest, record, _Slot((8,)))


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
        module.load_sidecar_record(tmp_path, manifest, record, _Slot((8,)))


def test_verify_probe_sidecar_requests_full_payload_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest = object()
    calls: list[tuple[object, Path, bool]] = []

    def verify(actual_manifest, actual_root, *, verify_sidecar_hash=False):
        calls.append((actual_manifest, actual_root, verify_sidecar_hash))
        return {"sidecar_verified": True}

    monkeypatch.setattr(module, "verify_expert_manifest", verify, raising=False)

    module._verify_probe_sidecar(tmp_path, manifest)

    assert calls == [(manifest, tmp_path, True)]


def test_verify_probe_sidecar_rejects_unverified_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "verify_expert_manifest",
        lambda *_args, **_kwargs: {"sidecar_verified": False},
    )

    with pytest.raises(RuntimeError, match="sidecar payload was not fully verified"):
        module._verify_probe_sidecar(tmp_path, object())


def test_probe_identity_hashes_exact_dynamic_and_normalized_configs() -> None:
    module = _load_module()
    arm_config = {
        "dynamic_memory": True,
        "expert_slab_slots": 32,
        "total_context_tokens": 131_072,
    }

    identity = module.build_probe_identity(
        model_artifact_id="pipenetwork/Hy3-4bit@revision",
        model_artifact_sha256="a" * 64,
        expert_manifest_id="expert-manifest-sidecar.json",
        expert_manifest_sha256="b" * 64,
        source_git_commit="c" * 40,
        arm_config=arm_config,
    )

    assert identity["arm_config"] == arm_config
    assert identity["normalized_config"] == {
        "expert_slab_slots": 32,
        "total_context_tokens": 131_072,
    }
    assert identity["arm_config_sha256"] == module.canonical_sha256(arm_config)
    assert identity["normalized_config_sha256"] == module.canonical_sha256(
        identity["normalized_config"]
    )
