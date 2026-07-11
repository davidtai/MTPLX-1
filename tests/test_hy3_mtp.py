"""Hy3 layer-80 NextN head: structure, loading, and fail-closed integrity."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

import mtplx.expert_manifest as expert_manifest_module
from mtplx.expert_manifest import (
    AuxiliaryFileInfo,
    EMPTY_SHA256,
    ExpertManifest,
    ExpertRecord,
    ShardInfo,
    SidecarInfo,
    TensorSegment,
    load_expert_manifest,
    save_expert_manifest,
)
from mtplx.hy3_mtp_patch import (
    HY3_MTP_BF16_FILE,
    HY3_MTP_EXPERTS_FILE,
    HY3_MTP_SOURCE_REPO,
    HY3_MTP_RESIDENTS_FILE,
    Hy3MTPLoadError,
    build_hy3_mtp_module,
    expected_bf16_names,
    expected_expert_names,
    expected_resident_names,
    load_hy3_mtp_bf16_weights,
    load_hy3_mtp_weights,
)
from mtplx.models.hy3_mlx import Hy3MTP, Hy3MTPLayer
from mtplx.models.hy3_mlx import Model as Hy3Model
from mtplx.models.hy3_mlx import ModelArgs as Hy3Args

_RESIDENTS_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "quantize_mtp_layer80_residents.py"
)
TEST_REVISION = "test-revision"


def _load_residents_script():
    spec = importlib.util.spec_from_file_location(
        "quantize_mtp_layer80_residents", _RESIDENTS_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_args() -> Hy3Args:
    return Hy3Args(
        model_type="hy_v3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        vocab_size=128,
        max_position_embeddings=128,
        head_dim=16,
        router_scaling_factor=2.0,
    )


def _tiny_source_residents(prefix: str) -> dict[str, mx.array]:
    def bf16(*shape):
        return mx.random.normal(shape).astype(mx.bfloat16)

    tensors = {
        prefix + "self_attn.q_proj.weight": bf16(64, 64),
        prefix + "self_attn.k_proj.weight": bf16(32, 64),
        prefix + "self_attn.v_proj.weight": bf16(32, 64),
        prefix + "self_attn.o_proj.weight": bf16(64, 64),
        prefix + "mlp.shared_mlp.gate_proj.weight": bf16(64, 64),
        prefix + "mlp.shared_mlp.up_proj.weight": bf16(64, 64),
        prefix + "mlp.shared_mlp.down_proj.weight": bf16(64, 64),
        prefix + "mlp.router.gate.weight": bf16(2, 64),
        prefix + "eh_proj.weight": bf16(64, 128),
        prefix + "enorm.weight": bf16(64),
        prefix + "hnorm.weight": bf16(64),
        prefix + "input_layernorm.weight": bf16(64),
        prefix + "post_attention_layernorm.weight": bf16(64),
        prefix + "final_layernorm.weight": bf16(64),
        prefix + "self_attn.q_norm.weight": bf16(16),
        prefix + "self_attn.k_norm.weight": bf16(16),
        prefix + "mlp.expert_bias": mx.zeros((2,), dtype=mx.float32),
    }
    mx.eval(list(tensors.values()))
    return tensors


def _write_tiny_artifacts(tmp_path: Path, *, revision: str = TEST_REVISION) -> Path:
    """Author both layer-80 fixture artifacts with the packaging scripts' formats."""

    mx.random.seed(11)
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    metadata = {"source_repo": "test/tiny-hy3", "source_revision": revision}
    residents_mod = _load_residents_script()
    residents = residents_mod.quantize_resident_tensors(
        _tiny_source_residents(prefix), layer_prefix=prefix
    )
    mx.eval(list(residents.values()))
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_RESIDENTS_FILE), residents, metadata=metadata
    )

    experts: dict[str, mx.array] = {}
    for expert in range(args.num_experts):
        for projection, shape in (
            ("gate_proj", (64, 64)),
            ("up_proj", (64, 64)),
            ("down_proj", (64, 64)),
        ):
            source = mx.random.normal(shape).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(
                source, group_size=64, bits=4, mode="affine"
            )
            base = f"{prefix}mlp.experts.{expert}.{projection}"
            experts[base + ".weight"] = weight
            experts[base + ".scales"] = scales
            experts[base + ".biases"] = biases
    # The real expert artifact also carries BF16 resident pass-through copies;
    # the loader must ignore them rather than double-load residents.
    experts[prefix + "eh_proj.weight"] = mx.zeros((64, 128), dtype=mx.bfloat16)
    mx.eval(list(experts.values()))
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE), experts, metadata=metadata
    )
    return tmp_path


def _replace_q4_experts_with_compact_sidecar(tmp_path: Path) -> dict[str, mx.array]:
    """Serialize the tiny layer-80 records exactly as the native artifact does."""

    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    experts_path = tmp_path / HY3_MTP_EXPERTS_FILE
    experts = dict(mx.load(str(experts_path), format="safetensors"))
    sidecar_payload = bytearray()
    records: list[ExpertRecord] = []
    for expert in range(args.num_experts):
        record_offset = len(sidecar_payload)
        segments: list[TensorSegment] = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for leaf in ("weight", "scales", "biases"):
                component = f"{projection}.{leaf}"
                tensor = f"{prefix}mlp.experts.{expert}.{component}"
                value = experts[tensor]
                raw = memoryview(value).cast("B").tobytes()
                segment_offset = len(sidecar_payload)
                sidecar_payload.extend(raw)
                segments.append(
                    TensorSegment(
                        component=component,
                        tensor=tensor,
                        shard="experts.bin",
                        offset=segment_offset,
                        length=len(raw),
                        dtype="U32" if leaf == "weight" else "BF16",
                        shape=tuple(int(dimension) for dimension in value.shape),
                    )
                )
        logical_bytes = len(sidecar_payload) - record_offset
        record_payload = bytes(sidecar_payload[record_offset:])
        records.append(
            ExpertRecord(
                layer=args.num_hidden_layers,
                expert=expert,
                logical_bytes=logical_bytes,
                segments=tuple(segments),
                sha256=hashlib.sha256(record_payload).hexdigest(),
                sidecar_offset=record_offset,
                sidecar_length=logical_bytes,
            )
        )

    sidecar_path = tmp_path / "experts.bin"
    sidecar_path.write_bytes(sidecar_payload)
    sidecar_sha256 = hashlib.sha256(sidecar_payload).hexdigest()
    routed_bytes = len(sidecar_payload)
    manifest = ExpertManifest(
        model_key="tiny-hy3-q4-native",
        source_repo=HY3_MTP_SOURCE_REPO,
        source_revision=TEST_REVISION,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        artifact_tensor_bytes=routed_bytes,
        resident_tensor_bytes=0,
        routed_expert_bytes=routed_bytes,
        shards=(
            ShardInfo(
                name="experts.bin",
                size=routed_bytes,
                header_bytes=0,
                header_sha256=EMPTY_SHA256,
                sha256=sidecar_sha256,
                kind="sidecar",
            ),
        ),
        resident_tensors=(),
        records=tuple(records),
        sidecar=SidecarInfo(
            file="experts.bin",
            alignment=256,
            size=routed_bytes,
            sha256=sidecar_sha256,
        ),
    ).with_digest()
    save_expert_manifest(manifest, tmp_path / "expert-manifest.json")
    experts_path.unlink()
    return experts


def _write_tiny_bf16_artifact(tmp_path: Path, *, revision: str = TEST_REVISION) -> Path:
    """Author the layer-80 BF16 fixture, mirroring the Q4 artifact writer.

    The BF16 artifact is a straight extraction of the source checkpoint:
    every layer-80 tensor as one BF16 ``.weight`` leaf (per-expert, not
    stacked), except the F32 router correction bias.
    """

    mx.random.seed(13)
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    tensors = _tiny_source_residents(prefix)
    for expert in range(args.num_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}mlp.experts.{expert}.{projection}.weight"] = (
                mx.random.normal((64, 64)).astype(mx.bfloat16)
            )
    mx.eval(list(tensors.values()))
    metadata = {"source_repo": "test/tiny-hy3", "source_revision": revision}
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE), tensors, metadata=metadata
    )
    return tmp_path


def test_expected_name_tables_cover_the_packaged_artifacts(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    assert len(expected_resident_names(args)) == 8 * 3 + 9
    assert len(expected_expert_names(args)) == args.num_experts * 3 * 3


def test_build_hy3_mtp_module_loads_quantized_resident_head(tmp_path: Path) -> None:
    # precision="q4" must keep today's quantized head behavior unchanged.
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    mtp = build_hy3_mtp_module(
        tmp_path, args, expected_revision=TEST_REVISION, precision="q4"
    )

    assert isinstance(mtp, Hy3MTP)
    assert len(mtp.layers) == 1
    assert mtp.start_layer == args.num_hidden_layers
    layer = mtp.layers[0]
    assert isinstance(layer, Hy3MTPLayer)

    # Experts are one resident stacked SwitchGLU in the pinned Q4 format.
    switch = layer.mtp_block.mlp.switch_mlp
    assert isinstance(switch.gate_proj, QuantizedSwitchLinear)
    assert switch.gate_proj.bits == 4 and switch.gate_proj.group_size == 64
    assert tuple(switch.gate_proj.weight.shape) == (2, 64, 8)
    assert tuple(switch.down_proj.scales.shape) == (2, 64, 1)

    # Residents mirror trunk conventions: Q4 attention, Q8 router gate,
    # BF16 eh_proj/norms, F32 correction bias.
    attn = layer.mtp_block.self_attn
    assert isinstance(attn.q_proj, nn.QuantizedLinear) and attn.q_proj.bits == 4
    router = layer.mtp_block.mlp.router
    assert isinstance(router.gate, nn.QuantizedLinear) and router.gate.bits == 8
    assert router.expert_bias.dtype == mx.float32
    assert isinstance(layer.eh_proj, nn.Linear)
    assert not isinstance(layer.eh_proj, nn.QuantizedLinear)
    assert layer.eh_proj.weight.dtype == mx.bfloat16
    assert layer.final_layernorm.weight.dtype == mx.bfloat16

    # 33 resident leaves + 9 stacked expert leaves, nothing else.
    parameters = dict(tree_flatten(mtp.parameters()))
    assert len(parameters) == 42
    assert all(name.startswith("layers.0.") for name in parameters)


def test_hy3_mtp_layer_forward_produces_finite_logits_and_hidden(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    mtp = build_hy3_mtp_module(
        tmp_path, args, expected_revision=TEST_REVISION, precision="q4"
    )
    trunk = Hy3Model(args)

    token_ids = mx.array([[3, 5]], dtype=mx.int32)
    previous_hidden = mx.random.normal((1, 2, args.hidden_size)).astype(mx.bfloat16)
    logits, hidden = mtp.layers[0](
        token_ids,
        previous_hidden,
        embed_tokens=trunk.model.embed_tokens,
        lm_head=trunk.lm_head,
    )
    mx.eval(logits, hidden)

    assert logits.shape == (1, 2, args.vocab_size)
    assert hidden.shape == (1, 2, args.hidden_size)
    assert mx.all(mx.isfinite(logits.astype(mx.float32))).item()
    assert mx.all(mx.isfinite(hidden.astype(mx.float32))).item()


def test_loader_rejects_revision_mismatch(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path, revision="unexpected")
    with pytest.raises(Hy3MTPLoadError, match="revision"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_missing_artifact_files(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    (tmp_path / HY3_MTP_EXPERTS_FILE).rename(tmp_path / "renamed.safetensors")
    with pytest.raises(Hy3MTPLoadError, match="missing Hy3 MTP artifact"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_missing_and_unexpected_tensors(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)

    residents = dict(mx.load(str(tmp_path / HY3_MTP_RESIDENTS_FILE)))
    del residents[prefix + "mlp.router.gate.scales"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_RESIDENTS_FILE),
        residents,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="missing tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    del experts[prefix + "mlp.experts.1.down_proj.weight"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="missing expert tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    experts[prefix + "mlp.experts.7.gate_proj.weight"] = mx.zeros(
        (64, 8), dtype=mx.uint32
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="unexpected expert tensors"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_wrong_leaf_dtypes(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    residents = dict(mx.load(str(tmp_path / HY3_MTP_RESIDENTS_FILE)))
    residents[prefix + "mlp.expert_bias"] = mx.zeros((2,), dtype=mx.bfloat16)
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_RESIDENTS_FILE),
        residents,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="must be float32"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_loader_rejects_inconsistent_expert_shapes(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    experts = dict(mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE)))
    experts[prefix + "mlp.experts.1.up_proj.weight"] = mx.zeros(
        (32, 8), dtype=mx.uint32
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_EXPERTS_FILE),
        experts,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="differs from expert 0"):
        load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_stacked_expert_weights_match_artifact_order(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    weights = load_hy3_mtp_weights(tmp_path, args, expected_revision=TEST_REVISION)
    experts = mx.load(str(tmp_path / HY3_MTP_EXPERTS_FILE))
    stacked = weights["layers.0.mtp_block.mlp.switch_mlp.gate_proj.weight"]
    assert tuple(stacked.shape) == (2, 64, 8)
    for expert in range(2):
        source = experts[f"{prefix}mlp.experts.{expert}.gate_proj.weight"]
        assert mx.array_equal(stacked[expert], source).item()
    # Residents come from the residents artifact, not the expert artifact's
    # BF16 pass-through copy.
    assert not mx.array_equal(
        weights["layers.0.eh_proj.weight"],
        experts[prefix + "eh_proj.weight"],
    ).item()


def test_q4_loader_falls_back_to_compact_layer80_sidecar(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_artifacts(tmp_path)
    experts = _replace_q4_experts_with_compact_sidecar(tmp_path)

    weights = load_hy3_mtp_weights(
        tmp_path,
        args,
        expected_revision=TEST_REVISION,
    )

    for projection in ("gate_proj", "up_proj", "down_proj"):
        for leaf in ("weight", "scales", "biases"):
            stacked = weights[f"layers.0.mtp_block.mlp.switch_mlp.{projection}.{leaf}"]
            assert stacked.shape[0] == args.num_experts
            for expert in range(args.num_experts):
                source = experts[f"{prefix}mlp.experts.{expert}.{projection}.{leaf}"]
                assert mx.array_equal(stacked[expert], source).item()


def test_q4_loader_rejects_corrupt_compact_layer80_record(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    _replace_q4_experts_with_compact_sidecar(tmp_path)
    sidecar = tmp_path / "experts.bin"
    payload = bytearray(sidecar.read_bytes())
    payload[0] ^= 0xFF
    sidecar.write_bytes(payload)

    with pytest.raises(Hy3MTPLoadError, match="record hash mismatch"):
        load_hy3_mtp_weights(
            tmp_path,
            args,
            expected_revision=TEST_REVISION,
        )


def test_native_bf16_loader_requires_manifest_bound_full_hash(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_artifacts(tmp_path)
    _write_tiny_bf16_artifact(tmp_path)
    _replace_q4_experts_with_compact_sidecar(tmp_path)

    with pytest.raises(Hy3MTPLoadError, match="auxiliary file role"):
        load_hy3_mtp_bf16_weights(
            tmp_path,
            args,
            expected_revision=TEST_REVISION,
        )

    path = tmp_path / HY3_MTP_BF16_FILE
    shard, _tensors = expert_manifest_module._read_safetensors_header(
        path,
        relative_name=path.name,
    )
    auxiliary = AuxiliaryFileInfo(
        file=path.name,
        role="mtp-bf16",
        size=shard.size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        header_bytes=shard.header_bytes,
        header_sha256=shard.header_sha256,
        source_repo=HY3_MTP_SOURCE_REPO,
        source_revision=TEST_REVISION,
    )
    manifest_path = tmp_path / "expert-manifest.json"
    manifest = load_expert_manifest(manifest_path)
    manifest = replace(
        manifest,
        auxiliary_files=(auxiliary,),
        manifest_sha256=None,
    ).with_digest()
    save_expert_manifest(manifest, manifest_path)

    weights = load_hy3_mtp_bf16_weights(
        tmp_path,
        args,
        expected_revision=TEST_REVISION,
    )
    assert weights


def _module_leaves(mtp) -> list[tuple[str, object]]:
    leaves = []

    def visit(prefix, module):
        leaves.append((prefix, module))
        for name, child in module.children().items():
            if isinstance(child, list):
                for index, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        visit(f"{prefix}.{name}.{index}", item)
            elif isinstance(child, nn.Module):
                visit(f"{prefix}.{name}", child)

    visit("mtp", mtp)
    return leaves


def test_expected_bf16_names_cover_the_full_head(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
    names = expected_bf16_names(args)
    # 8 dense projections + 8 BF16 norms/eh_proj + expert_bias, plus one
    # .weight leaf per expert projection.
    assert len(names) == 8 + 8 + 1 + args.num_experts * 3
    written = set(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    assert written == names


def test_default_build_precision_is_bf16_with_no_quantized_modules(
    tmp_path: Path,
) -> None:
    """Forge contract section 6: quantizing the MTP head collapses MoE
    acceptance to 5-11% (vs 79-85% BF16), so bf16 is the default and only
    the BF16 artifact is required."""

    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)  # no Q4 artifacts on disk at all
    mtp = build_hy3_mtp_module(tmp_path, args, expected_revision=TEST_REVISION)

    assert isinstance(mtp, Hy3MTP)
    layer = mtp.layers[0]
    assert isinstance(layer, Hy3MTPLayer)

    # The whole head is plain BF16: no quantized module anywhere.
    for path, module in _module_leaves(mtp):
        assert not isinstance(
            module, (nn.QuantizedLinear, QuantizedSwitchLinear)
        ), path
    switch = layer.mtp_block.mlp.switch_mlp
    assert tuple(switch.gate_proj.weight.shape) == (2, 64, 64)
    assert switch.gate_proj.weight.dtype == mx.bfloat16
    attn = layer.mtp_block.self_attn
    assert isinstance(attn.q_proj, nn.Linear)
    assert attn.q_proj.weight.dtype == mx.bfloat16
    router = layer.mtp_block.mlp.router
    assert isinstance(router.gate, nn.Linear)
    assert router.gate.weight.dtype == mx.bfloat16
    assert router.expert_bias.dtype == mx.float32
    assert layer.eh_proj.weight.dtype == mx.bfloat16

    # 17 resident leaves + 3 stacked expert leaves, nothing else.
    parameters = dict(tree_flatten(mtp.parameters()))
    assert len(parameters) == 20
    assert all(name.startswith("layers.0.") for name in parameters)


def test_bf16_head_forward_produces_finite_logits_and_hidden(
    tmp_path: Path,
) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
    mtp = build_hy3_mtp_module(tmp_path, args, expected_revision=TEST_REVISION)
    trunk = Hy3Model(args)

    token_ids = mx.array([[3, 5]], dtype=mx.int32)
    previous_hidden = mx.random.normal((1, 2, args.hidden_size)).astype(mx.bfloat16)
    logits, hidden = mtp.layers[0](
        token_ids,
        previous_hidden,
        embed_tokens=trunk.model.embed_tokens,
        lm_head=trunk.lm_head,
    )
    mx.eval(logits, hidden)

    assert logits.shape == (1, 2, args.vocab_size)
    assert hidden.shape == (1, 2, args.hidden_size)
    assert mx.all(mx.isfinite(logits.astype(mx.float32))).item()
    assert mx.all(mx.isfinite(hidden.astype(mx.float32))).item()


def test_bf16_stacked_expert_weights_match_artifact_order(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."
    _write_tiny_bf16_artifact(tmp_path)
    weights = load_hy3_mtp_bf16_weights(
        tmp_path, args, expected_revision=TEST_REVISION
    )
    source = mx.load(str(tmp_path / HY3_MTP_BF16_FILE))
    stacked = weights["layers.0.mtp_block.mlp.switch_mlp.gate_proj.weight"]
    assert tuple(stacked.shape) == (2, 64, 64)
    for expert in range(2):
        original = source[f"{prefix}mlp.experts.{expert}.gate_proj.weight"]
        assert mx.array_equal(stacked[expert], original).item()
    # Residents map onto the same Hy3MTPLayer paths as the Q4 loader.
    assert "layers.0.eh_proj.weight" in weights
    assert "layers.0.mtp_block.mlp.router.expert_bias" in weights
    assert "layers.0.mtp_block.self_attn.q_proj.weight" in weights


def test_bf16_loader_fails_closed(tmp_path: Path) -> None:
    args = _tiny_args()
    prefix = f"model.layers.{args.num_hidden_layers}."

    # Missing artifact file (only the Q4 artifacts on disk).
    _write_tiny_artifacts(tmp_path)
    with pytest.raises(Hy3MTPLoadError, match="missing Hy3 MTP artifact"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Revision mismatch.
    _write_tiny_bf16_artifact(tmp_path, revision="unexpected")
    with pytest.raises(Hy3MTPLoadError, match="revision"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Missing tensor.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    del tensors[prefix + "mlp.experts.1.down_proj.weight"]
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="missing tensors"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Unexpected tensor.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "mlp.experts.7.gate_proj.weight"] = mx.zeros(
        (64, 64), dtype=mx.bfloat16
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="unexpected tensors"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Wrong dtypes: BF16-only weights and the F32 correction bias.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "mlp.expert_bias"] = mx.zeros((2,), dtype=mx.bfloat16)
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="must be float32"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "eh_proj.weight"] = mx.zeros((64, 128), dtype=mx.float32)
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="must be"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)

    # Inconsistent expert shapes.
    _write_tiny_bf16_artifact(tmp_path)
    tensors = dict(mx.load(str(tmp_path / HY3_MTP_BF16_FILE)))
    tensors[prefix + "mlp.experts.1.up_proj.weight"] = mx.zeros(
        (32, 64), dtype=mx.bfloat16
    )
    mx.save_safetensors(
        str(tmp_path / HY3_MTP_BF16_FILE),
        tensors,
        metadata={"source_revision": TEST_REVISION},
    )
    with pytest.raises(Hy3MTPLoadError, match="differs from expert 0"):
        load_hy3_mtp_bf16_weights(tmp_path, args, expected_revision=TEST_REVISION)


def test_build_rejects_unknown_precision(tmp_path: Path) -> None:
    args = _tiny_args()
    _write_tiny_bf16_artifact(tmp_path)
    with pytest.raises(Hy3MTPLoadError, match="precision"):
        build_hy3_mtp_module(
            tmp_path, args, expected_revision=TEST_REVISION, precision="fp8"
        )
