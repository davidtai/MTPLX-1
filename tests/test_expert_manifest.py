from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import mtplx.expert_manifest as expert_manifest_module
from mtplx.expert_manifest import (
    AuxiliaryFileInfo,
    ExpertManifest,
    ExpertManifestError,
    ExpertRecord,
    ResidentTensor,
    ShardInfo,
    build_expert_manifest,
    build_expert_sidecar,
    load_expert_manifest,
    make_sidecar_authoritative,
    read_expert_record,
    resolve_artifact_member,
    save_expert_manifest,
    validate_expert_manifest_spec,
    verify_auxiliary_file,
    verify_expert_manifest,
)
from mtplx.expert_streaming_models import ExpertStreamingModelSpec


COMPONENTS = (
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)


def _tiny_spec(*, resident_bytes: int = 8) -> ExpertStreamingModelSpec:
    record_bytes = 6_912
    return ExpertStreamingModelSpec(
        key="tiny-q4",
        display_name="Tiny affine Q4",
        source_model="test/tiny",
        source_revision="source-revision",
        quant_model="test/tiny-q4",
        quant_revision="quant-revision",
        total_tensor_bytes=2 * record_bytes + resident_bytes,
        total_layers=2,
        routed_layer_start=1,
        routed_layer_count=1,
        expert_count=2,
        top_k=1,
        hidden_size=64,
        expert_hidden_size=64,
        quant_bits=4,
        quant_group_size=64,
        quant_parameter_bytes=2,
        router_storage="bfloat16",
        router_matmul_dtype="float32",
        router_bytes=0,
        kv_bytes_per_token=0,
        mtp_layer_index=2,
        mtp_included=False,
    )


def _component_info(component: str, *, stacked: bool) -> tuple[str, list[int], int]:
    leaf = component.rsplit(".", 1)[1]
    if leaf == "weight":
        dtype = "U32"
        per_expert_shape = [64, 8]
        per_expert_bytes = 2_048
    else:
        dtype = "BF16"
        per_expert_shape = [64, 1]
        per_expert_bytes = 128
    shape = [2, *per_expert_shape] if stacked else per_expert_shape
    return dtype, shape, per_expert_bytes


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _make_checkpoint(
    root: Path, *, numbered: bool = False
) -> tuple[ExpertStreamingModelSpec, dict[int, bytes]]:
    root.mkdir()
    spec = _tiny_spec()
    shards: list[list[tuple[str, str, list[int], bytes]]] = [[], []]
    expected: dict[int, bytearray] = {0: bytearray(), 1: bytearray()}
    weight_map: dict[str, str] = {}
    for component_index, component in enumerate(COMPONENTS):
        dtype, shape, per_expert_bytes = _component_info(
            component, stacked=not numbered
        )
        projection, leaf = component.split(".")
        if numbered:
            for expert in range(2):
                name = f"model.layers.1.mlp.experts.{expert}.{projection}.{leaf}"
                raw = bytes([component_index * 2 + expert + 1]) * per_expert_bytes
                shard_index = (component_index + expert) % 2
                shards[shard_index].append((name, dtype, shape, raw))
                weight_map[name] = f"model-{shard_index + 1:05d}-of-00002.safetensors"
                expected[expert].extend(raw)
        else:
            name = f"model.layers.1.mlp.switch_mlp.{component}"
            raw_parts = [
                bytes([component_index * 2 + expert + 1]) * per_expert_bytes
                for expert in range(2)
            ]
            raw = b"".join(raw_parts)
            shard_index = component_index % 2
            shards[shard_index].append((name, dtype, shape, raw))
            weight_map[name] = f"model-{shard_index + 1:05d}-of-00002.safetensors"
            for expert, part in enumerate(raw_parts):
                expected[expert].extend(part)
    resident_name = "model.embed_tokens.weight"
    resident_raw = bytes(range(8))
    shards[0].append((resident_name, "F32", [2], resident_raw))
    weight_map[resident_name] = "model-00001-of-00002.safetensors"
    for index, tensors in enumerate(shards, 1):
        _write_safetensors(root / f"model-{index:05d}-of-00002.safetensors", tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return spec, {expert: bytes(value) for expert, value in expected.items()}


@pytest.mark.parametrize("numbered", [False, True])
def test_build_manifest_for_stacked_and_numbered_experts(
    tmp_path: Path,
    numbered: bool,
) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root, numbered=numbered)

    manifest = build_expert_manifest(
        root,
        spec,
        hash_records=True,
        hash_shards=True,
    )

    assert manifest.artifact_tensor_bytes == spec.total_tensor_bytes
    assert manifest.resident_tensor_bytes == 8
    assert manifest.routed_expert_bytes == spec.routed_expert_bytes
    assert len(manifest.shards) == 2
    assert len(manifest.resident_tensors) == 1
    assert len(manifest.records) == 2
    assert all(shard.sha256 for shard in manifest.shards)
    for expert in range(2):
        record = manifest.record(1, expert)
        assert tuple(segment.component for segment in record.segments) == COMPONENTS
        assert record.logical_bytes == spec.expert_record_bytes
        assert record.sha256 == hashlib.sha256(expected[expert]).hexdigest()
        assert read_expert_record(manifest, root, 1, expert) == expected[expert]

    report = verify_expert_manifest(
        manifest,
        root,
        verify_records=True,
        verify_shard_hashes=True,
    )
    assert report["valid"] is True
    assert report["checked_records"] == 2


def test_manifest_roundtrip_digest_and_unknown_fields_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    path = root / "expert-manifest.json"

    saved = save_expert_manifest(manifest, path)
    loaded = load_expert_manifest(path)

    assert loaded == saved
    tampered = loaded.to_dict()
    tampered["model_key"] = "wrong"
    with pytest.raises(ExpertManifestError, match="digest mismatch"):
        ExpertManifest.from_dict(tampered)
    unknown = loaded.to_dict()
    unknown["surprise"] = True
    with pytest.raises(ExpertManifestError, match="unknown keys"):
        ExpertManifest.from_dict(unknown, verify_digest=False)
    escaped = loaded.to_dict()
    escaped["records"][0]["segments"][0]["shard"] = "../escape"
    with pytest.raises(ExpertManifestError, match="unsafe"):
        ExpertManifest.from_dict(escaped, verify_digest=False)


def test_manifest_spec_validation_rejects_same_byte_wrong_record_geometry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    wrong = replace(
        manifest,
        records=(manifest.records[0], replace(manifest.records[1], layer=2)),
        manifest_sha256=None,
    ).with_digest()
    wrong.validate_structure()

    with pytest.raises(ExpertManifestError, match="record geometry"):
        validate_expert_manifest_spec(wrong, spec)


def test_auxiliary_file_is_digest_bound_and_hash_verified(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    auxiliary_path = root / "layer80-bf16.safetensors"
    _write_safetensors(
        auxiliary_path,
        [("model.layers.80.norm.weight", "F32", [2], bytes(range(8)))],
    )
    auxiliary_shard, _tensors = expert_manifest_module._read_safetensors_header(
        auxiliary_path,
        relative_name=auxiliary_path.name,
    )
    auxiliary = AuxiliaryFileInfo(
        file=auxiliary_path.name,
        role="mtp-bf16",
        size=auxiliary_shard.size,
        sha256=hashlib.sha256(auxiliary_path.read_bytes()).hexdigest(),
        header_bytes=auxiliary_shard.header_bytes,
        header_sha256=auxiliary_shard.header_sha256,
        source_repo="tencent/Hy3",
        source_revision="source-revision",
    )
    with_auxiliary = replace(
        manifest,
        auxiliary_files=(auxiliary,),
        manifest_sha256=None,
    ).with_digest()
    saved = save_expert_manifest(with_auxiliary, root / "expert-manifest.json")
    loaded = load_expert_manifest(root / "expert-manifest.json")
    assert loaded == saved
    assert verify_auxiliary_file(loaded, root, "mtp-bf16") == auxiliary

    fd = os.open(auxiliary_path, os.O_RDWR)
    try:
        os.pwrite(fd, b"\xff", auxiliary.header_bytes)
    finally:
        os.close(fd)
    with pytest.raises(ExpertManifestError, match="auxiliary file hash mismatch"):
        verify_auxiliary_file(loaded, root, "mtp-bf16")


def test_hugging_face_snapshot_blob_symlink_is_allowed_but_other_escapes_fail(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "models--org--model"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / ("b" * 64)
    blob.write_bytes(b"trusted content-addressed payload")
    member = snapshot / "model-00001-of-00001.safetensors"
    member.symlink_to(Path("..") / ".." / "blobs" / blob.name)

    assert resolve_artifact_member(snapshot, member.name) == blob

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    escaped = snapshot / "escaped.bin"
    escaped.symlink_to(outside)
    with pytest.raises(ExpertManifestError, match="escapes root"):
        resolve_artifact_member(snapshot, escaped.name)

    # A blobs-looking symlink is not trusted outside the exact snapshots/
    # sibling-blobs repository layout.
    ordinary_root = tmp_path / "ordinary"
    ordinary_root.mkdir()
    ordinary = ordinary_root / "model.bin"
    ordinary.symlink_to(blob)
    with pytest.raises(ExpertManifestError, match="escapes root"):
        resolve_artifact_member(ordinary_root, ordinary.name)


def test_manifest_builder_rejects_external_index_symlink(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    index = root / "model.safetensors.index.json"
    external = tmp_path / "external-index.json"
    index.replace(external)
    index.symlink_to(external)

    with pytest.raises(ExpertManifestError, match="escapes root"):
        build_expert_manifest(root, spec)


def test_aligned_sidecar_is_readable_and_verifiable(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)

    updated = build_expert_sidecar(manifest, root, root / "experts.bin")

    assert updated.sidecar is not None
    assert updated.sidecar.file == "experts.bin"
    assert updated.sidecar.alignment == 16_384
    assert updated.records[0].sidecar_offset == 0
    assert updated.records[1].sidecar_offset == 16_384
    assert updated.sidecar.size == 16_384 + spec.expert_record_bytes
    assert read_expert_record(updated, root, 1, 0) == expected[0]
    assert read_expert_record(updated, root, 1, 1) == expected[1]
    report = verify_expert_manifest(updated, root, verify_sidecar_hash=True)
    assert report["sidecar_verified"] is True


def _repack_fixture_resident_only(
    root: Path,
    legacy: ExpertManifest,
) -> tuple[ExpertManifest, ExpertManifest, Path]:
    resident_path = root / "resident-00001-of-00001.safetensors"
    _write_safetensors(
        resident_path,
        [("model.embed_tokens.weight", "F32", [2], bytes(range(8)))],
    )
    resident_shard, resident_infos = expert_manifest_module._read_safetensors_header(
        resident_path, relative_name=resident_path.name
    )
    resident_shard = replace(
        resident_shard,
        sha256=hashlib.sha256(resident_path.read_bytes()).hexdigest(),
    )
    resident_info = resident_infos[0]
    resident_tensor = ResidentTensor(
        tensor=resident_info.name,
        shard=resident_info.shard,
        offset=resident_info.offset,
        length=resident_info.length,
        dtype=resident_info.dtype,
        shape=resident_info.shape,
    )
    repacked = replace(
        legacy,
        shards=(*legacy.shards, resident_shard),
        resident_tensors=(resident_tensor,),
        manifest_sha256=None,
    ).with_digest()
    repacked.validate_structure()
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": resident_tensor.length},
                "weight_map": {resident_tensor.tensor: resident_tensor.shard},
            }
        ),
        encoding="utf-8",
    )
    return repacked, make_sidecar_authoritative(repacked), resident_path


def test_authoritative_sidecar_manifest_is_backward_compatible_and_source_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec, hash_shards=True)
    legacy = build_expert_sidecar(manifest, root, root / "experts.bin")

    repacked, compact, resident_path = _repack_fixture_resident_only(root, legacy)

    assert all("kind" not in shard.to_dict() for shard in legacy.shards)

    assert compact.source_repo == repacked.source_repo
    assert compact.source_revision == repacked.source_revision
    sidecar_shard = compact.shards[-1]
    assert sidecar_shard.kind == "sidecar"
    assert sidecar_shard.header_bytes == 0
    assert sidecar_shard.to_dict()["kind"] == "sidecar"
    assert ShardInfo.from_dict(sidecar_shard.to_dict()) == sidecar_shard
    for old_record, record in zip(legacy.records, compact.records, strict=True):
        offset = record.sidecar_offset
        assert offset is not None
        for old_segment, segment in zip(
            old_record.segments, record.segments, strict=True
        ):
            assert segment.component == old_segment.component
            assert segment.tensor == old_segment.tensor
            assert segment.shard == "experts.bin"
            assert segment.offset == offset
            offset += segment.length
        assert offset == record.sidecar_offset + record.logical_bytes

    compact_names = {shard.name for shard in compact.shards}
    for shard in legacy.shards:
        if shard.name not in compact_names:
            (root / shard.name).unlink()
    assert sorted(path.name for path in root.glob("*.safetensors")) == [
        resident_path.name
    ]
    saved = save_expert_manifest(compact, root / "expert-manifest.json")
    loaded = load_expert_manifest(root / "expert-manifest.json")
    assert loaded == saved

    report = verify_expert_manifest(
        loaded,
        root,
        verify_records=True,
        verify_shard_hashes=True,
        verify_sidecar_hash=True,
    )
    assert report["checked_records"] == 2
    assert report["checked_shards"] == len(compact.shards)
    assert report["sidecar_verified"] is True
    for expert in range(2):
        assert (
            read_expert_record(loaded, root, 1, expert, prefer_sidecar=False)
            == expected[expert]
        )


def test_authoritative_sidecar_corruption_fails_record_and_file_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec, hash_shards=True)
    legacy = build_expert_sidecar(manifest, root, root / "experts.bin")
    _repacked, compact, _resident_path = _repack_fixture_resident_only(root, legacy)
    first = compact.records[0]
    assert first.sidecar_offset is not None
    sidecar = root / "experts.bin"
    fd = os.open(sidecar, os.O_RDWR)
    try:
        original = os.pread(fd, 1, first.sidecar_offset)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), first.sidecar_offset)
    finally:
        os.close(fd)

    with pytest.raises(ExpertManifestError, match="record hash mismatch"):
        verify_expert_manifest(compact, root, verify_records=True)
    with pytest.raises(ExpertManifestError, match="shard hash mismatch"):
        verify_expert_manifest(compact, root, verify_shard_hashes=True)
    with pytest.raises(ExpertManifestError, match="sidecar hash mismatch"):
        verify_expert_manifest(compact, root, verify_sidecar_hash=True)


def test_authoritative_verifier_rejects_experts_hidden_in_resident_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec, hash_shards=True)
    legacy = build_expert_sidecar(manifest, root, root / "experts.bin")
    falsely_compact = make_sidecar_authoritative(legacy)

    with pytest.raises(ExpertManifestError, match="compact resident inventory"):
        verify_expert_manifest(falsely_compact, root)


def test_sidecar_resume_reuses_hashed_prefix_before_reading_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    complete = build_expert_sidecar(manifest, root, root / "experts.bin")
    first = complete.records[0]
    assert first.sidecar_offset is not None
    assert first.sidecar_length is not None
    final = root / "experts.bin"
    partial = root / ".experts.bin.partial"
    expected_sidecar = final.read_bytes()
    final.replace(partial)
    with partial.open("r+b") as handle:
        handle.truncate(first.sidecar_offset + first.sidecar_length)

    source_reads: list[tuple[int, int]] = []
    original_reader = expert_manifest_module._read_source_record

    def recording_reader(root: Path, record: ExpertRecord) -> bytes:
        source_reads.append((record.layer, record.expert))
        return original_reader(root, record)

    monkeypatch.setattr(expert_manifest_module, "_read_source_record", recording_reader)
    resumed = build_expert_sidecar(manifest, root, final)

    assert source_reads == [(1, 1)]
    assert resumed.sidecar == complete.sidecar
    assert final.read_bytes() == expected_sidecar


def test_sidecar_resume_completes_with_all_source_shards_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    complete = build_expert_sidecar(manifest, root, root / "experts.bin")
    final = root / "experts.bin"
    partial = root / ".experts.bin.partial"
    expected_sidecar = final.read_bytes()
    final.replace(partial)
    for shard in manifest.shards:
        (root / shard.name).unlink()

    resumed = build_expert_sidecar(manifest, root, final)

    assert resumed.sidecar == complete.sidecar
    assert final.read_bytes() == expected_sidecar
    assert read_expert_record(resumed, root, 1, 0) == expected[0]


def test_corrupt_payload_and_truncated_sidecar_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)
    record = manifest.records[0]
    segment = record.segments[0]
    shard = root / segment.shard
    fd = os.open(shard, os.O_RDWR)
    try:
        original = os.pread(fd, 1, segment.offset)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), segment.offset)
    finally:
        os.close(fd)
    with pytest.raises(ExpertManifestError, match="record hash mismatch"):
        read_expert_record(manifest, root, record.layer, record.expert)

    # Rebuild a clean model before testing the independent sidecar failure.
    clean_root = tmp_path / "clean"
    clean_spec, _ = _make_checkpoint(clean_root)
    clean_manifest = build_expert_manifest(clean_root, clean_spec)
    updated = build_expert_sidecar(
        clean_manifest, clean_root, clean_root / "experts.bin"
    )
    assert updated.sidecar is not None
    (clean_root / "experts.bin").write_bytes(b"short")
    with pytest.raises(ExpertManifestError, match="sidecar size mismatch"):
        verify_expert_manifest(updated, clean_root)


def test_missing_component_and_index_mismatch_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing_key = "model.layers.1.mlp.switch_mlp.down_proj.biases"
    del index["weight_map"][missing_key]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ExpertManifestError, match="wrong shard|mismatch"):
        build_expert_manifest(root, spec)


def test_sidecar_must_stay_inside_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    manifest = build_expert_manifest(root, spec)

    with pytest.raises(ExpertManifestError, match="inside the artifact root"):
        build_expert_sidecar(manifest, root, tmp_path / "outside.bin")


def test_manifest_builder_can_skip_pinned_total_only_for_fixtures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    spec, _expected = _make_checkpoint(root)
    wrong_total = replace(spec, total_tensor_bytes=spec.total_tensor_bytes + 4)

    with pytest.raises(ExpertManifestError, match="pinned"):
        build_expert_manifest(root, wrong_total)
    manifest = build_expert_manifest(
        root,
        wrong_total,
        require_pinned_tensor_bytes=False,
    )
    assert manifest.artifact_tensor_bytes == spec.total_tensor_bytes
