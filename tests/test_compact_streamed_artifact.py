from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import mtplx.compact_streamed_artifact as compact_module
from mtplx.compact_streamed_artifact import (
    CompactArtifactError,
    compact_streamed_artifact,
)
from mtplx.expert_manifest import (
    ExpertManifestError,
    build_expert_manifest,
    build_expert_sidecar,
    load_expert_manifest,
    save_expert_manifest,
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


def _tiny_spec(model_key: str) -> ExpertStreamingModelSpec:
    record_bytes = 6_912
    resident_bytes = 8
    return ExpertStreamingModelSpec(
        key=model_key,
        display_name=f"{model_key} affine Q4",
        source_model="test/source",
        source_revision="source-revision",
        quant_model="test/q4",
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


def _component_info(component: str) -> tuple[str, list[int], int]:
    leaf = component.rsplit(".", 1)[1]
    if leaf == "weight":
        return "U32", [2, 64, 8], 2_048
    return "BF16", [2, 64, 1], 128


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


def _make_source(
    tmp_path: Path,
    *,
    model_key: str = "hy3-test",
    hash_shards: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "source"
    root.mkdir()
    spec = _tiny_spec(model_key)
    tensors: list[tuple[str, str, list[int], bytes]] = []
    for component_index, component in enumerate(COMPONENTS):
        dtype, shape, per_expert_bytes = _component_info(component)
        name = f"model.layers.1.mlp.switch_mlp.{component}"
        raw = b"".join(
            bytes([component_index * 2 + expert + 1]) * per_expert_bytes
            for expert in range(2)
        )
        tensors.append((name, dtype, shape, raw))
    resident_name = "model.embed_tokens.weight"
    tensors.append((resident_name, "F32", [2], bytes(range(8))))
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors(root / shard_name, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard_name for name, *_rest in tensors},
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": model_key}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0"}),
        encoding="utf-8",
    )
    manifest = build_expert_manifest(
        root,
        spec,
        hash_records=True,
        hash_shards=hash_shards,
    )
    manifest = build_expert_sidecar(manifest, root, root / "experts.bin")
    manifest_path = root / "expert-manifest.json"
    save_expert_manifest(manifest, manifest_path)
    return root, manifest_path


@pytest.mark.parametrize("model_key", ["hy3-test", "glm-test"])
def test_compacts_hy3_and_glm_without_duplicate_experts(
    tmp_path: Path,
    model_key: str,
) -> None:
    source, manifest_path = _make_source(tmp_path, model_key=model_key)
    output = tmp_path / "compact"
    source_sidecar = source / "experts.bin"
    source_shard = source / "model-00001-of-00001.safetensors"
    source_shard_size = source_shard.stat().st_size

    report = compact_streamed_artifact(
        source,
        manifest_path,
        source_sidecar,
        output,
        chunk_bytes=257,
    )

    compact_sidecar = output / "experts.bin"
    assert report["model_key"] == model_key
    assert report["sidecar"]["mode"] == "hardlinked"
    assert report["sidecar"]["same_inode_as_source"] is True
    assert os.path.samestat(source_sidecar.stat(), compact_sidecar.stat())
    assert source_shard.exists() and source_shard.stat().st_size == source_shard_size
    assert json.loads((output / "config.json").read_text())["model_type"] == model_key
    assert (output / "tokenizer.json").is_file()
    assert not (output / source_shard.name).exists()
    assert sorted(path.name for path in output.glob("*.safetensors")) == [
        "model-resident-00001-of-00001.safetensors"
    ]

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"model.embed_tokens.weight"}
    compact = load_expert_manifest(output / "expert-manifest.json")
    assert {shard.kind for shard in compact.shards} == {"safetensors", "sidecar"}
    assert all(
        segment.shard == "experts.bin"
        for record in compact.records
        for segment in record.segments
    )
    assert report["verification"]["physical_inventory"]["routed_tensors_in_shards"] == 0
    assert report["verification"]["resident_parity"]["checked_bytes"] == 8
    source_hashes = report["verification"]["source_shard_hashes"]
    assert source_hashes["verified_shards"] == 1
    assert source_hashes["computed_missing_hashes"] == 1
    assert source_hashes["matched_manifest_hashes"] == 0
    assert source_hashes["sha256"] == {
        source_shard.name: hashlib.sha256(source_shard.read_bytes()).hexdigest()
    }
    assert report["cleanup"]["eligible"] is True
    assert report["cleanup"]["performed"] is False
    assert report["cleanup"]["source_preserved"] is True


def test_verifies_existing_source_shard_hash_before_compaction(tmp_path: Path) -> None:
    source, manifest_path = _make_source(tmp_path, hash_shards=True)
    manifest = load_expert_manifest(manifest_path)
    wrong = replace(
        manifest,
        shards=(replace(manifest.shards[0], sha256="0" * 64),),
        manifest_sha256=None,
    )
    save_expert_manifest(wrong, manifest_path)

    output = tmp_path / "compact"
    with pytest.raises(CompactArtifactError, match="source shard hash mismatch"):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            output,
        )
    assert not output.exists()


def test_known_model_spec_is_validated_when_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest_path = _make_source(tmp_path)
    monkeypatch.setitem(compact_module.MODEL_SPECS, "hy3-test", _tiny_spec("hy3-test"))

    report = compact_streamed_artifact(
        source,
        manifest_path,
        source / "experts.bin",
        tmp_path / "compact",
    )
    assert report["verification"]["known_model_spec"] == {
        "model_key": "hy3-test",
        "verified": True,
    }


def test_known_model_spec_identity_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest_path = _make_source(tmp_path)
    wrong_spec = replace(_tiny_spec("hy3-test"), quant_revision="wrong-revision")
    monkeypatch.setitem(compact_module.MODEL_SPECS, "hy3-test", wrong_spec)

    with pytest.raises(CompactArtifactError, match="source revision"):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            tmp_path / "compact",
        )


def test_stale_manifests_checkpoints_and_handoffs_are_not_passthrough(
    tmp_path: Path,
) -> None:
    source, manifest_path = _make_source(tmp_path)
    (source / "expert-manifest-sidecar.json").write_text("{}", encoding="utf-8")
    (source / "conversion-checkpoint.jsonl").write_text("stale\n", encoding="utf-8")
    (source / "conversion-provenance.json").write_text("{}", encoding="utf-8")
    (source / "conversion-validation.json").write_text("{}", encoding="utf-8")
    (source / "previous-compaction-handoff.json").write_text("{}", encoding="utf-8")
    (source / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "hy3-mtp",
                "mtp_sidecar_file": "layer80-bf16.safetensors",
                "forge_provenance": {
                    "forge_inputs": {
                        "source_path": "/old/source",
                        "layout_oracle_path": "/old/oracle",
                        "output_path": "/old/output",
                        "checkpoint_path": "/old/output/conversion-checkpoint.jsonl",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    metadata = source / "metadata"
    metadata.mkdir()
    (metadata / "expert_manifest_old.json").write_text("{}", encoding="utf-8")
    (metadata / "notes.json").write_text('{"keep": true}', encoding="utf-8")

    output = tmp_path / "compact"
    report = compact_streamed_artifact(
        source,
        manifest_path,
        source / "experts.bin",
        output,
    )

    manifest_like = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and (
            "expert-manifest" in path.name.lower()
            or "expert_manifest" in path.name.lower()
        )
    )
    assert manifest_like == ["expert-manifest.json"]
    assert not (output / "conversion-checkpoint.jsonl").exists()
    assert not (output / "conversion-provenance.json").exists()
    assert not (output / "conversion-validation.json").exists()
    assert not (output / "previous-compaction-handoff.json").exists()
    assert (output / "compaction-handoff.json").is_file()
    assert (output / "metadata" / "notes.json").is_file()
    assert "metadata/notes.json" in report["metadata_files"]
    runtime = json.loads((output / "mtplx_runtime.json").read_text())
    assert runtime["arch_id"] == "hy3-mtp"
    assert runtime["mtp_sidecar_file"] == "layer80-bf16.safetensors"
    assert runtime["forge_provenance"]["forge_inputs"] == {"output_path": "."}
    assert runtime["compaction_provenance"] == {
        "format": "mtplx-compact-artifact-handoff-v1",
        "manifest": "expert-manifest.json",
        "model_index": "model.safetensors.index.json",
        "resident_only_shards": True,
    }


def test_hardlink_failure_requires_explicit_copy_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest_path = _make_source(tmp_path)
    original_link = compact_module.os.link

    def cross_device_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(compact_module.os, "link", cross_device_link)
    refused = tmp_path / "refused"
    with pytest.raises(CompactArtifactError, match="allow-sidecar-copy"):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            refused,
        )
    assert not refused.exists()
    assert not list(tmp_path.glob(".refused.compacting-*"))

    copied = tmp_path / "copied"
    report = compact_streamed_artifact(
        source,
        manifest_path,
        source / "experts.bin",
        copied,
        allow_sidecar_copy=True,
    )
    assert report["sidecar"]["mode"] == "copied"
    assert report["sidecar"]["same_inode_as_source"] is False
    assert (copied / "experts.bin").read_bytes() == (
        source / "experts.bin"
    ).read_bytes()
    assert not os.path.samestat(
        (copied / "experts.bin").stat(),
        (source / "experts.bin").stat(),
    )
    monkeypatch.setattr(compact_module.os, "link", original_link)


def test_interruption_removes_temporary_output_and_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest_path = _make_source(tmp_path)
    original_sidecar = (source / "experts.bin").read_bytes()

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(compact_module, "_copy_range", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".interrupted.compacting-*"))
    assert not (tmp_path / ".interrupted.compaction.lock").exists()
    assert (source / "experts.bin").read_bytes() == original_sidecar
    assert (source / "model-00001-of-00001.safetensors").is_file()


def test_rejects_output_path_traversal_into_source(tmp_path: Path) -> None:
    source, manifest_path = _make_source(tmp_path)

    with pytest.raises(CompactArtifactError, match="inside the source"):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            source / "nested" / "compact",
        )


def test_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    source, manifest_path = _make_source(tmp_path)
    value = json.loads(manifest_path.read_text())
    value["sidecar"]["file"] = "../experts.bin"
    escaped = tmp_path / "escaped-manifest.json"
    escaped.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ExpertManifestError, match="unsafe"):
        compact_streamed_artifact(
            source,
            escaped,
            source / "experts.bin",
            tmp_path / "compact",
        )


def test_rejects_sidecar_and_metadata_symlinks(tmp_path: Path) -> None:
    source, manifest_path = _make_source(tmp_path)
    real_sidecar = source / "real-experts.bin"
    (source / "experts.bin").replace(real_sidecar)
    (source / "experts.bin").symlink_to(real_sidecar)
    with pytest.raises(CompactArtifactError, match="sidecar may not be a symlink"):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            tmp_path / "sidecar-symlink",
        )

    (source / "experts.bin").unlink()
    real_sidecar.replace(source / "experts.bin")
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    (source / "linked-config.json").symlink_to(external)
    with pytest.raises(ExpertManifestError, match="escapes root"):
        compact_streamed_artifact(
            source,
            manifest_path,
            source / "experts.bin",
            tmp_path / "metadata-symlink",
        )
    assert not (tmp_path / "metadata-symlink").exists()
