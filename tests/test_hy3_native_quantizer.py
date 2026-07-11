from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest

from mtplx.expert_manifest import (
    load_expert_manifest,
    read_expert_record,
    verify_expert_manifest,
)
from mtplx.hy3_native_quantizer import (
    CHECKPOINT_FORMAT,
    COMPONENTS,
    DEFAULT_ALIGNMENT,
    ORACLE_REVISION,
    REAL_EXPERT_RECORD_BYTES,
    SOURCE_REVISION,
    CheckpointJournal,
    Hy3Geometry,
    Hy3NativeConversionError,
    TensorPayload,
    TensorSpec,
    build_conversion_plan,
    execute_conversion,
    prepare_conversion,
)


DTYPE_BYTES = {"BF16": 2, "F32": 4, "U32": 4}


def _repeat(seed: bytes, length: int) -> bytes:
    return (seed * (length // len(seed) + 1))[:length]


def _raw(name: str, dtype: str, shape: tuple[int, ...]) -> bytes:
    length = DTYPE_BYTES[dtype]
    for dim in shape:
        length *= dim
    return _repeat(hashlib.sha256(name.encode()).digest(), length)


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _write_index(root: Path, weight_map: dict[str, str]) -> None:
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _write_download_metadata(root: Path, filename: str, digest: str) -> None:
    path = root / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{SOURCE_REVISION}\n{digest}\n0\n", encoding="utf-8")


def _q_leaves(
    prefix: str,
    *,
    rows: int,
    cols: int,
    bits: int,
    stacked: int | None = None,
) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    leading = () if stacked is None else (stacked,)
    shapes = {
        "weight": (*leading, rows, cols * bits // 32),
        "scales": (*leading, rows, cols // 64),
        "biases": (*leading, rows, cols // 64),
    }
    result = {}
    for leaf, shape in shapes.items():
        dtype = "U32" if leaf == "weight" else "BF16"
        name = f"{prefix}.{leaf}"
        result[name] = (dtype, shape, _raw(name, dtype, shape))
    return result


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "official"
    oracle = tmp_path / "oracle"
    source.mkdir()
    oracle.mkdir()
    (source / ".mtplx-source-revision").write_text(SOURCE_REVISION)
    (oracle / ".mtplx-source-revision").write_text(ORACLE_REVISION)
    config = {
        "model_type": "hy_v3",
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "first_k_dense_replace": 1,
        "num_experts": 2,
        "hidden_size": 64,
        "moe_intermediate_size": 64,
        "expert_hidden_dim": 64,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (oracle / "config.json").write_text(json.dumps(config), encoding="utf-8")

    source_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def add_source(name: str, dtype: str, shape: tuple[int, ...]) -> None:
        source_tensors[name] = (dtype, shape, _raw(name, dtype, shape))

    add_source("model.embed_tokens.weight", "BF16", (4, 64))
    add_source("model.layers.0.mlp.down_proj.weight", "BF16", (64, 64))
    for layer in (1, 2):
        add_source(f"model.layers.{layer}.input_layernorm.weight", "BF16", (64,))
        add_source(f"model.layers.{layer}.mlp.router.gate.weight", "BF16", (2, 64))
        add_source(f"model.layers.{layer}.mlp.expert_bias", "F32", (2,))
        if layer == 2:
            add_source("model.layers.2.eh_proj.weight", "BF16", (64, 128))
            add_source("model.layers.2.enorm.weight", "BF16", (64,))
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                add_source(
                    f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight",
                    "BF16",
                    (64, 64),
                )
    source_shard = source / "model.safetensors"
    _write_safetensors(source_shard, source_tensors)
    _write_index(source, {name: source_shard.name for name in source_tensors})
    _write_download_metadata(
        source,
        source_shard.name,
        hashlib.sha256(source_shard.read_bytes()).hexdigest(),
    )
    _write_download_metadata(
        source,
        "model.safetensors.index.json",
        hashlib.sha256(
            (source / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
    )

    oracle_tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    oracle_tensors["model.embed_tokens.weight"] = source_tensors[
        "model.embed_tokens.weight"
    ]
    oracle_tensors.update(
        _q_leaves("model.layers.0.mlp.down_proj", rows=64, cols=64, bits=4)
    )
    oracle_tensors["model.layers.1.input_layernorm.weight"] = source_tensors[
        "model.layers.1.input_layernorm.weight"
    ]
    oracle_tensors.update(
        _q_leaves(
            "model.layers.1.mlp.router.gate",
            rows=2,
            cols=64,
            bits=8,
        )
    )
    bias_name = "model.layers.1.mlp.router.expert_bias"
    oracle_tensors[bias_name] = (
        "F32",
        (2,),
        source_tensors["model.layers.1.mlp.expert_bias"][2],
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        oracle_tensors.update(
            _q_leaves(
                f"model.layers.1.mlp.switch_mlp.{projection}",
                rows=64,
                cols=64,
                bits=4,
                stacked=2,
            )
        )
    oracle_shard = oracle / "model.safetensors"
    _write_safetensors(oracle_shard, oracle_tensors)
    _write_index(oracle, {name: oracle_shard.name for name in oracle_tensors})

    layer2 = {
        name: value
        for name, value in source_tensors.items()
        if name.startswith("model.layers.2.")
    }
    head = source / "layer80-bf16.safetensors"
    _write_safetensors(
        head,
        layer2,
        metadata={
            "source_repo": "tencent/Hy3",
            "source_revision": SOURCE_REVISION,
            "extracted_prefix": "model.layers.2.",
        },
    )
    return source, oracle, head


class FakeQuantizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {"implementation": "FakeQuantizer", "code_sha256": "0" * 64}

    def quantize(
        self,
        source: TensorSpec,
        raw: bytes,
        *,
        bits: int,
        group_size: int,
    ) -> tuple[TensorPayload, TensorPayload, TensorPayload]:
        assert source.dtype == "BF16"
        assert group_size == 64
        self.calls.append(source.name)
        rows, cols = source.shape
        specs = (
            ("U32", (rows, cols * bits // 32), b"weight"),
            ("BF16", (rows, cols // group_size), b"scales"),
            ("BF16", (rows, cols // group_size), b"biases"),
        )
        payloads = []
        for dtype, shape, leaf in specs:
            length = DTYPE_BYTES[dtype]
            for dim in shape:
                length *= dim
            seed = hashlib.sha256(raw + leaf + bytes([bits])).digest()
            payloads.append(TensorPayload(dtype, shape, _repeat(seed, length)))
        return tuple(payloads)  # type: ignore[return-value]


def _tiny_plan(source: Path, oracle: Path):
    return build_conversion_plan(
        source,
        oracle,
        enforce_official_geometry=False,
    )


def test_plan_has_serialized_aligned_records_and_layer80_resident_file(
    tmp_path: Path,
) -> None:
    source, oracle, _head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)

    assert plan.geometry.routed_layers == (1, 2)
    assert len(plan.records) == 4
    assert plan.expert_record_bytes == 6_912
    assert [record.offset for record in plan.records] == [0, 16_384, 32_768, 49_152]
    assert all(record.offset % DEFAULT_ALIGNMENT == 0 for record in plan.records)
    assert all(
        tuple(component.component for component in record.components) == COMPONENTS
        for record in plan.records
    )
    # Safetensors/index keys are lexical: biases and scales appear before
    # weight. They must still be consumed as one canonical BF16-weight unit.
    assert sorted(("module.weight", "module.scales", "module.biases")) == [
        "module.biases",
        "module.scales",
        "module.weight",
    ]
    assert not any(
        unit.source_tensor.endswith((".biases", ".scales"))
        for unit in plan.resident_units
    )
    assert all(len(unit.targets) == 3 for unit in plan.resident_units if unit.quantized)
    layer80 = next(
        output
        for output in plan.resident_files
        if output.name == "layer80-residents-q.safetensors"
    )
    assert any(
        tensor.name == "model.layers.2.mlp.expert_bias" for tensor in layer80.tensors
    )
    assert not any(".experts." in tensor.name for tensor in layer80.tensors)


def test_complete_conversion_is_compact_provenanced_and_atomically_published(
    tmp_path: Path,
) -> None:
    source, oracle, head = _fixture(tmp_path)
    source_shard_before = (source / "model.safetensors").read_bytes()
    plan = _tiny_plan(source, oracle)
    quantizer = FakeQuantizer()
    output = tmp_path / "native"

    result = execute_conversion(
        plan,
        output,
        quantizer_factory=lambda: quantizer,
        bf16_head_source=head,
    )

    assert result == output
    assert output.is_dir()
    assert not (tmp_path / ".native.hy3-native-work").exists()
    assert os.path.samefile(head, output / "layer80-bf16.safetensors")
    assert not (output / "layer80-q4.safetensors").exists()
    assert not any(path.name.startswith("model-000") for path in output.iterdir())
    assert (source / "model.safetensors").read_bytes() == source_shard_before

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert not any(
        "switch_mlp" in name or ".experts." in name for name in index["weight_map"]
    )
    assert {
        filename
        for name, filename in index["weight_map"].items()
        if name.startswith("model.layers.2.")
    } == {"layer80-residents-q.safetensors"}

    config = json.loads((output / "config.json").read_text())
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["group_size"] == 64
    for layer in (1, 2):
        assert config["quantization"][f"model.layers.{layer}.mlp.router.gate"] == {
            "bits": 8,
            "group_size": 64,
        }

    runtime = json.loads((output / "mtplx_runtime.json").read_text())
    assert runtime["forge_provenance"]["source_format"] == "bf16_native"
    assert runtime["forge_provenance"]["forge_recipe"]["mtp_policy"] == "keep_bf16"
    assert "speed_evidence" not in runtime
    assert "verified_on" not in runtime

    provenance = json.loads((output / "conversion-provenance.json").read_text())
    assert provenance["source"]["revision"] == SOURCE_REVISION
    assert len(provenance["source"]["shards"]) == 1
    assert (
        provenance["source"]["shards"][0]["sha256"]
        == hashlib.sha256(source_shard_before).hexdigest()
    )
    assert provenance["layout"]["component_order"] == list(COMPONENTS)
    assert provenance["layout"]["layer80_q4_expert_intermediate_created"] is False
    assert provenance["source"]["automatic_source_deletion"] is False

    manifest = load_expert_manifest(output / "expert-manifest.json")
    assert manifest.sidecar is not None
    assert manifest.shards[-1].kind == "sidecar"
    assert manifest.auxiliary_files[0].role == "mtp-bf16"
    assert (
        tuple(segment.component for segment in manifest.records[0].segments)
        == COMPONENTS
    )
    assert read_expert_record(manifest, output, 1, 0)
    report = verify_expert_manifest(
        manifest,
        output,
        verify_records=True,
        verify_shard_hashes=True,
        verify_sidecar_hash=True,
    )
    assert report["valid"] is True
    assert report["checked_auxiliary_files"] == 1


def test_resume_verifies_accepted_record_and_repairs_corruption(tmp_path: Path) -> None:
    source, oracle, head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)
    output = tmp_path / "native"
    first = FakeQuantizer()

    def stop_after_first_record(event: dict[str, Any]) -> None:
        if event.get("event") == "expert_record":
            raise RuntimeError("simulated crash after durable checkpoint")

    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_conversion(
            plan,
            output,
            quantizer_factory=lambda: first,
            bf16_head_source=head,
            on_event=stop_after_first_record,
        )
    assert not output.exists()
    work = tmp_path / ".native.hy3-native-work"
    assert work.is_dir()
    journal = (work / "conversion-checkpoint.jsonl").read_text()
    assert '"event":"expert_record"' in journal

    # Same-size corruption must not be trusted merely because a checkpoint
    # event exists. Resume hashes the accepted range and requantizes it.
    experts = work / "experts.bin"
    fd = os.open(experts, os.O_RDWR)
    try:
        original = os.pread(fd, 1, 0)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), 0)
        os.fsync(fd)
    finally:
        os.close(fd)

    resumed = FakeQuantizer()
    execute_conversion(
        plan,
        output,
        quantizer_factory=lambda: resumed,
        bf16_head_source=head,
    )
    expert_calls = [name for name in resumed.calls if ".mlp.experts." in name]
    assert len(expert_calls) == 12  # repaired record plus three not-yet-built records
    assert load_expert_manifest(output / "expert-manifest.json").records[0].sha256


def test_prepare_only_never_quantizes_and_copy_resume_hashes_destination(
    tmp_path: Path,
) -> None:
    source, oracle, head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)
    output = tmp_path / "native"

    work = prepare_conversion(
        plan,
        output,
        bf16_head_source=head,
        copy_bf16_head=True,
    )
    assert work.is_dir() and not output.exists()
    copied = work / "layer80-bf16.safetensors"
    assert not os.path.samefile(head, copied)
    expected = hashlib.sha256(head.read_bytes()).hexdigest()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == expected

    fd = os.open(copied, os.O_RDWR)
    try:
        original = os.pread(fd, 1, copied.stat().st_size - 1)
        os.pwrite(fd, bytes([original[0] ^ 1]), copied.stat().st_size - 1)
        os.fsync(fd)
    finally:
        os.close(fd)
    prepare_conversion(
        plan,
        output,
        bf16_head_source=head,
        copy_bf16_head=True,
    )
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == expected


def test_real_hy3_expert_geometry_is_exact_and_alignment_compatible() -> None:
    from mtplx.expert_streaming_models import get_model_spec

    geometry = Hy3Geometry(80, 1, 1, 192, 4096, 1536)
    q4_gate_or_up = 1536 * 512 * 4 + 2 * (1536 * 64 * 2)
    q4_down = 4096 * 192 * 4 + 2 * (4096 * 24 * 2)
    assert 2 * q4_gate_or_up + q4_down == REAL_EXPERT_RECORD_BYTES
    assert REAL_EXPERT_RECORD_BYTES % DEFAULT_ALIGNMENT == 0
    assert len(geometry.routed_layers) * geometry.expert_count == 15_360
    native = get_model_spec("hy3-q4-native")
    assert native.expert_record_bytes == REAL_EXPERT_RECORD_BYTES
    assert native.routed_expert_bytes == 163_074_539_520
    assert native.resident_bytes == 5_073_424_896
    assert native.total_tensor_bytes == 168_147_964_416
    assert native.mtp_q4_resident_bytes == 121_070_848
    assert native.mtp_bf16_tensor_bytes == 7_505_224_960


def test_checkpoint_truncates_torn_tail_before_future_appends(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    journal = CheckpointJournal(path, "f" * 64)
    journal.accept({"event": "unit", "id": "one", "sha256": "1" * 64})
    with path.open("ab") as handle:
        handle.write(b'{"event":"unit","id":"torn"')
        handle.flush()
        os.fsync(handle.fileno())

    resumed = CheckpointJournal(path, "f" * 64)
    resumed.accept({"event": "unit", "id": "two", "sha256": "2" * 64})
    final = CheckpointJournal(path, "f" * 64)

    assert final.get("unit", "one") is not None
    assert final.get("unit", "two") is not None
    assert final.get("unit", "torn") is None
    assert json.loads(path.read_text().splitlines()[0])["format"] == CHECKPOINT_FORMAT


def test_official_mode_rejects_revision_overrides_before_work(tmp_path: Path) -> None:
    source, oracle, _head = _fixture(tmp_path)
    with pytest.raises(Hy3NativeConversionError, match="pinned to"):
        build_conversion_plan(
            source,
            oracle,
            source_revision="not-the-pinned-source",
            enforce_official_geometry=True,
        )
    with pytest.raises(Hy3NativeConversionError, match="layout oracle is pinned"):
        build_conversion_plan(
            source,
            oracle,
            oracle_revision="not-the-pinned-oracle",
            enforce_official_geometry=True,
        )


def test_work_root_symlink_and_stale_duplicate_are_rejected(tmp_path: Path) -> None:
    source, oracle, head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)
    output = tmp_path / "native"
    outside = tmp_path / "outside"
    outside.mkdir()
    work = tmp_path / ".native.hy3-native-work"
    work.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Hy3NativeConversionError, match="not a real directory"):
        prepare_conversion(plan, output, bf16_head_source=head)
    assert not any(outside.iterdir())

    work.unlink()
    prepared = prepare_conversion(plan, output, bf16_head_source=head)
    (prepared / "layer80-q4.safetensors").write_bytes(b"stale duplicate")
    with pytest.raises(Hy3NativeConversionError, match="unexpected files"):
        prepare_conversion(plan, output, bf16_head_source=head)


def test_bf16_source_cannot_alias_the_work_directory(tmp_path: Path) -> None:
    source, oracle, _head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)
    output = tmp_path / "native"
    work = tmp_path / ".native.hy3-native-work"
    with pytest.raises(Hy3NativeConversionError, match="cannot be inside"):
        prepare_conversion(
            plan,
            output,
            bf16_head_source=work / "layer80-bf16.safetensors",
        )


def test_output_cannot_be_nested_in_either_input_tree(tmp_path: Path) -> None:
    source, oracle, head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)

    with pytest.raises(Hy3NativeConversionError, match="official source"):
        prepare_conversion(
            plan,
            source / "native-output",
            bf16_head_source=head,
        )
    with pytest.raises(Hy3NativeConversionError, match="layout oracle"):
        prepare_conversion(
            plan,
            oracle / "native-output",
            bf16_head_source=head,
        )


def test_conversion_refuses_source_shard_stat_drift_before_publish(
    tmp_path: Path,
) -> None:
    source, oracle, head = _fixture(tmp_path)
    plan = _tiny_plan(source, oracle)
    output = tmp_path / "native"
    mutated = False

    def mutate_after_first_record(event: dict[str, Any]) -> None:
        nonlocal mutated
        if not mutated and event.get("event") == "expert_record":
            shard = source / "model.safetensors"
            current = shard.stat()
            os.utime(shard, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
            mutated = True

    with pytest.raises(Hy3NativeConversionError, match="changed during conversion"):
        execute_conversion(
            plan,
            output,
            quantizer_factory=FakeQuantizer,
            bf16_head_source=head,
            on_event=mutate_after_first_record,
        )
    assert mutated is True
    assert not output.exists()
