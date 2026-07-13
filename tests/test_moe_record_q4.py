from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import mlx.core as mx
import pytest

from mtplx.expert_manifest import ExpertRecord, TensorSegment
from mtplx.expert_streaming_models import HY3_Q4
from mtplx.kernels.moe_record_q4 import (
    HY3_RECORD_Q4_BYTES,
    RecordQ4ContractError,
    is_hy3_record_q4_supported,
    is_hy3_weighted_top8_statically_supported,
    validate_hy3_record_q4,
    validate_hy3_weighted_top8,
)


_COMPONENTS = (
    ("gate_proj.weight", "U32", (1536, 512), 3_145_728),
    ("gate_proj.scales", "BF16", (1536, 64), 196_608),
    ("gate_proj.biases", "BF16", (1536, 64), 196_608),
    ("up_proj.weight", "U32", (1536, 512), 3_145_728),
    ("up_proj.scales", "BF16", (1536, 64), 196_608),
    ("up_proj.biases", "BF16", (1536, 64), 196_608),
    ("down_proj.weight", "U32", (4096, 192), 3_145_728),
    ("down_proj.scales", "BF16", (4096, 24), 196_608),
    ("down_proj.biases", "BF16", (4096, 24), 196_608),
)


def _record() -> ExpertRecord:
    segments = tuple(
        TensorSegment(
            component=component,
            tensor=f"model.layers.1.mlp.switch_mlp.{component}",
            shard="model-00001-of-00034.safetensors",
            offset=index * 16_384,
            length=length,
            dtype=dtype,
            shape=shape,
        )
        for index, (component, dtype, shape, length) in enumerate(_COMPONENTS)
    )
    return ExpertRecord(
        layer=1,
        expert=0,
        logical_bytes=HY3_RECORD_Q4_BYTES,
        segments=segments,
        sidecar_offset=0,
        sidecar_length=HY3_RECORD_Q4_BYTES,
    )


def _replace_segment(
    record: ExpertRecord,
    index: int,
    **changes: object,
) -> ExpertRecord:
    segments = list(record.segments)
    segments[index] = replace(segments[index], **changes)
    return replace(record, segments=tuple(segments))


def test_contract_exposes_exact_v1_offsets_and_coverage() -> None:
    layout = validate_hy3_record_q4(_record(), HY3_Q4)

    assert layout.record_bytes == 10_616_832
    assert layout.group_size == 64
    assert layout.bits == 4
    assert layout.hidden_size == 4096
    assert layout.intermediate_size == 1536
    assert tuple(component.name for component in layout.components) == tuple(
        item[0] for item in _COMPONENTS
    )
    assert tuple(component.offset for component in layout.components) == (
        0,
        3_145_728,
        3_342_336,
        3_538_944,
        6_684_672,
        6_881_280,
        7_077_888,
        10_223_616,
        10_420_224,
    )
    assert layout.components[-1].stop == layout.record_bytes
    assert layout.component("down_proj.weight").shape == (4096, 192)


def test_contract_layout_is_immutable() -> None:
    layout = validate_hy3_record_q4(_record(), HY3_Q4)

    with pytest.raises(FrozenInstanceError):
        layout.record_bytes = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda record: replace(record, logical_bytes=10_616_831), "record bytes"),
        (
            lambda record: replace(record, logical_bytes=10_616_832.5),
            "exact integer",
        ),
        (lambda record: replace(record, logical_bytes=True), "exact integer"),
        (lambda record: replace(record, sidecar_length=10_616_831), "sidecar length"),
        (
            lambda record: replace(
                record,
                sidecar_offset=None,
                sidecar_length=None,
            ),
            "sidecar record",
        ),
        (lambda record: replace(record, sidecar_offset=1), "sidecar offset"),
        (lambda record: replace(record, sidecar_offset=-16_384), "at least 0"),
        (
            lambda record: _replace_segment(record, 0, dtype="BF16"),
            "gate_proj.weight dtype",
        ),
        (
            lambda record: _replace_segment(record, 0, shape=(1536, 511)),
            "gate_proj.weight shape",
        ),
        (
            lambda record: _replace_segment(record, 0, shape=(1536.0, 512.0)),
            "exact integer",
        ),
        (
            lambda record: _replace_segment(record, 0, shape=1536),
            "shape",
        ),
        (
            lambda record: _replace_segment(record, 0, length=3_145_724),
            "gate_proj.weight length",
        ),
        (
            lambda record: _replace_segment(record, 0, length=3_145_728.5),
            "exact integer",
        ),
        (
            lambda record: replace(
                record,
                segments=(record.segments[1], record.segments[0], *record.segments[2:]),
            ),
            "component order",
        ),
        (
            lambda record: replace(record, segments=record.segments[:-1]),
            "nine components",
        ),
    ],
)
def test_contract_rejects_layout_drift(mutate, match: str) -> None:
    with pytest.raises(RecordQ4ContractError, match=match):
        validate_hy3_record_q4(mutate(_record()), HY3_Q4)


def test_contract_rejects_non_hy3_or_non_group64_specs() -> None:
    non_hy3 = replace(HY3_Q4, key="hy3-compatible-q4")
    non_q4 = replace(HY3_Q4, quant_bits=2)
    # Group-32 doubles the affine parameter footprint. Keep the synthetic
    # model internally valid so the record-native contract performs the
    # intended rejection instead of the model-spec constructor rejecting it.
    non_group64 = replace(
        HY3_Q4,
        quant_group_size=32,
        total_tensor_bytes=(
            HY3_Q4.total_tensor_bytes
            + HY3_Q4.scale_bias_bytes * HY3_Q4.expert_count * HY3_Q4.routed_layer_count
        ),
    )
    non_bf16_params = replace(HY3_Q4, quant_parameter_bytes=1)
    non_expert_count = replace(HY3_Q4, expert_count=191)
    non_top8 = replace(HY3_Q4, top_k=7)

    with pytest.raises(RecordQ4ContractError, match="hy3-q4 only"):
        validate_hy3_record_q4(_record(), non_hy3)

    with pytest.raises(RecordQ4ContractError, match="affine Q4"):
        validate_hy3_record_q4(_record(), non_q4)

    with pytest.raises(RecordQ4ContractError, match="group-64"):
        validate_hy3_record_q4(_record(), non_group64)

    with pytest.raises(RecordQ4ContractError, match="BF16 scale and bias"):
        validate_hy3_record_q4(_record(), non_bf16_params)

    with pytest.raises(RecordQ4ContractError, match="192-expert"):
        validate_hy3_record_q4(_record(), non_expert_count)

    with pytest.raises(RecordQ4ContractError, match="top-8"):
        validate_hy3_record_q4(_record(), non_top8)

    assert not is_hy3_record_q4_supported(_record(), non_top8)

    for unsupported in (
        non_hy3,
        non_q4,
        non_group64,
        non_bf16_params,
        non_expert_count,
        non_top8,
    ):
        assert not is_hy3_record_q4_supported(_record(), unsupported)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"source_model": "other/Hy3"}, "pinned Hy3 source"),
        ({"source_revision": "untrusted"}, "pinned Hy3 source"),
        ({"quant_model": "other/Hy3-4bit"}, "pinned Hy3 source"),
        ({"quant_revision": "untrusted"}, "pinned Hy3 source"),
        ({"router_storage": "q8"}, "pinned Hy3 router"),
        ({"router_matmul_dtype": "bfloat16"}, "pinned Hy3 router"),
        ({"routed_layer_start": 0}, "pinned Hy3 router"),
        ({"routed_layer_count": 78}, "pinned Hy3 router"),
        ({"hidden_size": 2048}, "4096/1536"),
        ({"expert_hidden_size": 1024}, "4096/1536"),
    ],
)
def test_contract_rejects_artifact_router_or_geometry_drift(
    changes: dict[str, object],
    match: str,
) -> None:
    spec = replace(HY3_Q4, **changes)

    with pytest.raises(RecordQ4ContractError, match=match):
        validate_hy3_record_q4(_record(), spec)

    assert not is_hy3_record_q4_supported(_record(), spec)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("activation_dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("indices_dtype", [mx.int32, mx.uint32])
def test_weighted_top8_contract_accepts_batched_decode_shapes(
    batch: int,
    activation_dtype,
    indices_dtype,
) -> None:
    x = mx.zeros((batch, 1, HY3_Q4.hidden_size), dtype=activation_dtype)
    indices = mx.zeros((batch, 1, HY3_Q4.top_k), dtype=indices_dtype)
    scores = mx.ones((batch, 1, HY3_Q4.top_k), dtype=activation_dtype)

    contract = validate_hy3_weighted_top8(x, indices, scores, HY3_Q4)

    assert contract.tokens == batch
    assert contract.assignments == batch * HY3_Q4.top_k
    assert contract.activation_dtype == activation_dtype
    assert is_hy3_weighted_top8_statically_supported(x, indices, scores, HY3_Q4)


@pytest.mark.parametrize(
    ("indices", "scores", "match"),
    [
        (
            mx.zeros((1, 1, 7), dtype=mx.int32),
            mx.ones((1, 1, 7), dtype=mx.bfloat16),
            "top-8",
        ),
        (
            mx.zeros((1, 1, 8), dtype=mx.int32),
            mx.ones((1, 1, 8), dtype=mx.float32),
            "activation dtype",
        ),
        (
            mx.zeros((1, 1, 8), dtype=mx.int64),
            mx.ones((1, 1, 8), dtype=mx.bfloat16),
            "indices dtype",
        ),
        (
            mx.zeros((1, 8), dtype=mx.int32),
            mx.ones((1, 1, 8), dtype=mx.bfloat16),
            "shape",
        ),
    ],
)
def test_weighted_top8_contract_rejects_unsupported_inputs(
    indices: mx.array,
    scores: mx.array,
    match: str,
) -> None:
    x = mx.zeros((1, 1, HY3_Q4.hidden_size), dtype=mx.bfloat16)

    with pytest.raises(RecordQ4ContractError, match=match):
        validate_hy3_weighted_top8(x, indices, scores, HY3_Q4)

    assert not is_hy3_weighted_top8_statically_supported(x, indices, scores, HY3_Q4)


@pytest.mark.parametrize(
    ("x", "indices", "scores", "match"),
    [
        (
            mx.zeros((HY3_Q4.hidden_size,), dtype=mx.bfloat16),
            mx.zeros((8,), dtype=mx.int32),
            mx.zeros((8,), dtype=mx.bfloat16),
            "activation shape",
        ),
        (
            mx.zeros((1, 1, HY3_Q4.hidden_size - 1), dtype=mx.bfloat16),
            mx.zeros((1, 1, 8), dtype=mx.int32),
            mx.zeros((1, 1, 8), dtype=mx.bfloat16),
            "activation shape",
        ),
        (
            mx.zeros((1, 1, HY3_Q4.hidden_size), dtype=mx.float32),
            mx.zeros((1, 1, 8), dtype=mx.int32),
            mx.zeros((1, 1, 8), dtype=mx.float32),
            "activation dtype",
        ),
        (
            mx.zeros((1, 1, HY3_Q4.hidden_size), dtype=mx.bfloat16),
            mx.zeros((1, 1, 8), dtype=mx.int32),
            mx.zeros((1, 8), dtype=mx.bfloat16),
            "shape",
        ),
    ],
)
def test_weighted_top8_contract_rejects_activation_drift(
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    match: str,
) -> None:
    with pytest.raises(RecordQ4ContractError, match=match):
        validate_hy3_weighted_top8(x, indices, scores, HY3_Q4)

    assert not is_hy3_weighted_top8_statically_supported(x, indices, scores, HY3_Q4)


def test_weighted_top8_static_contract_defers_device_index_bounds() -> None:
    x = mx.zeros((1, 1, HY3_Q4.hidden_size), dtype=mx.bfloat16)
    scores = mx.ones((1, 1, HY3_Q4.top_k), dtype=mx.bfloat16)

    for indices in (
        mx.full((1, 1, HY3_Q4.top_k), -1, dtype=mx.int32),
        mx.full((1, 1, HY3_Q4.top_k), HY3_Q4.expert_count, dtype=mx.int32),
        mx.full((1, 1, HY3_Q4.top_k), 2**32 - 1, dtype=mx.uint32),
    ):
        assert is_hy3_weighted_top8_statically_supported(x, indices, scores, HY3_Q4)


def test_eligibility_is_a_deterministic_fallback_decision() -> None:
    valid = _record()
    invalid = _replace_segment(valid, 8, length=196_606)
    invalid_shape = _replace_segment(valid, 0, shape=1536)

    assert is_hy3_record_q4_supported(valid, HY3_Q4)
    assert not is_hy3_record_q4_supported(invalid, HY3_Q4)
    assert not is_hy3_record_q4_supported(invalid, HY3_Q4)
    assert not is_hy3_record_q4_supported(invalid_shape, HY3_Q4)


def test_weighted_top8_rejects_zero_token_grid() -> None:
    x = mx.zeros((0, HY3_Q4.hidden_size), dtype=mx.bfloat16)
    indices = mx.zeros((0, HY3_Q4.top_k), dtype=mx.int32)
    scores = mx.zeros((0, HY3_Q4.top_k), dtype=mx.bfloat16)

    with pytest.raises(RecordQ4ContractError, match="at least one token"):
        validate_hy3_weighted_top8(x, indices, scores, HY3_Q4)

    assert not is_hy3_weighted_top8_statically_supported(x, indices, scores, HY3_Q4)
