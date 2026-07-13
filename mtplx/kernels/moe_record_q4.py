"""Fail-closed contracts for experimental Hy3 v1 record-native execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import index
from typing import Any

import mlx.core as mx

from mtplx.expert_manifest import ExpertRecord
from mtplx.expert_streaming_models import ExpertStreamingModelSpec


HY3_RECORD_Q4_BYTES = 10_616_832
HY3_RECORD_ALIGNMENT = 16_384


class RecordQ4ContractError(ValueError):
    """Raised when an input cannot use the exact Hy3 record-native path."""


@dataclass(frozen=True)
class RecordQ4Component:
    """One logical component inside an expert-major v1 sidecar record."""

    name: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def stop(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class RecordQ4Layout:
    """Immutable device-addressing contract for one Hy3 Q4 record."""

    components: tuple[RecordQ4Component, ...]
    record_bytes: int = HY3_RECORD_Q4_BYTES
    alignment: int = HY3_RECORD_ALIGNMENT
    bits: int = 4
    group_size: int = 64
    hidden_size: int = 4096
    intermediate_size: int = 1536

    def component(self, name: str) -> RecordQ4Component:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)


@dataclass(frozen=True)
class WeightedTop8Contract:
    """Shape/dtype facts required by activation-dtype weighted reduction."""

    tokens: int
    assignments: int
    activation_dtype: Any


_COMPONENT_SPECS = (
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


def _expected_layout() -> RecordQ4Layout:
    offset = 0
    components: list[RecordQ4Component] = []
    for name, dtype, shape, length in _COMPONENT_SPECS:
        components.append(
            RecordQ4Component(
                name=name,
                offset=offset,
                length=length,
                dtype=dtype,
                shape=shape,
            )
        )
        offset += length
    if offset != HY3_RECORD_Q4_BYTES:  # pragma: no cover - module invariant
        raise AssertionError("Hy3 record component table does not cover the record")
    return RecordQ4Layout(components=tuple(components))


HY3_RECORD_Q4_LAYOUT = _expected_layout()


def _exact_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RecordQ4ContractError(f"{name} must be an exact integer")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise RecordQ4ContractError(f"{name} must be an exact integer") from exc
    if normalized < minimum:
        raise RecordQ4ContractError(f"{name} must be at least {minimum}")
    return normalized


def _exact_shape(name: str, value: object) -> tuple[int, ...]:
    try:
        dimensions = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RecordQ4ContractError(
            f"{name} shape must contain exact integer dimensions"
        ) from exc
    return tuple(
        _exact_integer(f"{name} shape dimension", dimension, minimum=1)
        for dimension in dimensions
    )


def _validate_hy3_spec(spec: ExpertStreamingModelSpec) -> None:
    if spec.key != "hy3-q4":
        raise RecordQ4ContractError("record-native Q4 supports hy3-q4 only")
    if spec.quant_bits != 4:
        raise RecordQ4ContractError("record-native Q4 requires affine Q4")
    if spec.quant_group_size != 64:
        raise RecordQ4ContractError("record-native Q4 requires group-64")
    if spec.quant_parameter_bytes != 2:
        raise RecordQ4ContractError("record-native Q4 requires BF16 scale and bias")
    if (
        spec.hidden_size,
        spec.expert_hidden_size,
        spec.expert_count,
        spec.top_k,
    ) != (4096, 1536, 192, 8):
        raise RecordQ4ContractError(
            "record-native Q4 requires Hy3 4096/1536/192-expert/top-8 geometry"
        )
    if (
        spec.source_model,
        spec.source_revision,
        spec.quant_model,
        spec.quant_revision,
    ) != (
        "tencent/Hy3",
        "716aa7241bd6d95896be4ebfc761162a9c4d49ef",
        "pipenetwork/Hy3-4bit",
        "160619d3f96c8470350b6dac0ef033a8381551e3",
    ):
        raise RecordQ4ContractError(
            "record-native Q4 requires the pinned Hy3 source and quant revisions"
        )
    if (
        spec.router_storage,
        spec.router_matmul_dtype,
        spec.routed_layer_start,
        spec.routed_layer_count,
    ) != ("affine-q8 with fp32 correction bias", "float32", 1, 79):
        raise RecordQ4ContractError(
            "record-native Q4 requires the pinned Hy3 router and routed layers"
        )
    if spec.expert_record_bytes != HY3_RECORD_Q4_BYTES:
        raise RecordQ4ContractError(
            f"record-native Q4 requires {HY3_RECORD_Q4_BYTES} record bytes"
        )


def validate_hy3_record_q4(
    record: ExpertRecord,
    spec: ExpertStreamingModelSpec,
) -> RecordQ4Layout:
    """Validate one manifest record against exact device byte offsets.

    ``TensorSegment.offset`` addresses the original safetensors shard and is
    deliberately not used here. The v1 sidecar record is the ordered
    concatenation of segment payloads, so its device offsets are cumulative.
    """

    _validate_hy3_spec(spec)
    record_bytes = _exact_integer("record bytes", record.logical_bytes, minimum=1)
    if record_bytes != HY3_RECORD_Q4_BYTES:
        raise RecordQ4ContractError(f"record bytes must equal {HY3_RECORD_Q4_BYTES}")
    if record.sidecar_offset is None or record.sidecar_length is None:
        raise RecordQ4ContractError("record-native Q4 requires a sidecar record")
    sidecar_offset = _exact_integer("sidecar offset", record.sidecar_offset)
    if sidecar_offset % HY3_RECORD_ALIGNMENT:
        raise RecordQ4ContractError(
            f"sidecar offset must be {HY3_RECORD_ALIGNMENT}-byte aligned"
        )
    sidecar_length = _exact_integer("sidecar length", record.sidecar_length, minimum=1)
    if sidecar_length != HY3_RECORD_Q4_BYTES:
        raise RecordQ4ContractError(f"sidecar length must equal {HY3_RECORD_Q4_BYTES}")
    if len(record.segments) != len(_COMPONENT_SPECS):
        raise RecordQ4ContractError("record must contain exactly nine components")

    actual_names = tuple(segment.component for segment in record.segments)
    expected_names = tuple(component[0] for component in _COMPONENT_SPECS)
    if actual_names != expected_names:
        raise RecordQ4ContractError("record component order differs from Hy3 v1")

    coverage = 0
    for segment, expected in zip(record.segments, _COMPONENT_SPECS, strict=True):
        name, dtype, shape, length = expected
        if segment.dtype != dtype:
            raise RecordQ4ContractError(
                f"{name} dtype must be {dtype}, got {segment.dtype}"
            )
        actual_shape = _exact_shape(name, segment.shape)
        if actual_shape != shape:
            raise RecordQ4ContractError(
                f"{name} shape must be {shape}, got {actual_shape}"
            )
        segment_length = _exact_integer(f"{name} length", segment.length, minimum=1)
        if segment_length != length:
            raise RecordQ4ContractError(
                f"{name} length must be {length}, got {segment.length}"
            )
        coverage += segment_length
    if coverage != HY3_RECORD_Q4_BYTES:
        raise RecordQ4ContractError(
            f"component coverage is {coverage}, expected {HY3_RECORD_Q4_BYTES}"
        )
    return HY3_RECORD_Q4_LAYOUT


def is_hy3_record_q4_supported(
    record: ExpertRecord,
    spec: ExpertStreamingModelSpec,
) -> bool:
    """Return the deterministic fallback decision without compiling a kernel."""

    try:
        validate_hy3_record_q4(record, spec)
    except RecordQ4ContractError:
        return False
    return True


def validate_hy3_weighted_top8(
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    spec: ExpertStreamingModelSpec,
) -> WeightedTop8Contract:
    """Validate exact routed shapes before weighted Stage B execution."""

    _validate_hy3_spec(spec)
    if x.ndim < 2 or int(x.shape[-1]) != spec.hidden_size:
        raise RecordQ4ContractError(
            f"activation shape must end in hidden width {spec.hidden_size}"
        )
    if indices.ndim < 1 or int(indices.shape[-1]) != 8:
        raise RecordQ4ContractError("expert indices must have a top-8 dimension")
    expected_route_shape = (*x.shape[:-1], 8)
    if tuple(indices.shape) != expected_route_shape or tuple(scores.shape) != tuple(
        indices.shape
    ):
        raise RecordQ4ContractError(
            "indices and scores shape must match activation leading dimensions"
        )
    if indices.dtype not in (mx.int32, mx.uint32):
        raise RecordQ4ContractError(
            f"indices dtype must be int32 or uint32, got {indices.dtype}"
        )
    if x.dtype not in (mx.bfloat16, mx.float16):
        raise RecordQ4ContractError(
            f"activation dtype must be bfloat16 or float16, got {x.dtype}"
        )
    if scores.dtype != x.dtype:
        raise RecordQ4ContractError(
            "router scores must use the activation dtype for non-FP32 combine"
        )
    tokens = prod(int(size) for size in x.shape[:-1])
    if tokens < 1:
        raise RecordQ4ContractError("weighted top-8 requires at least one token")
    return WeightedTop8Contract(
        tokens=tokens,
        assignments=tokens * 8,
        activation_dtype=x.dtype,
    )


def is_hy3_weighted_top8_statically_supported(
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    spec: ExpertStreamingModelSpec,
) -> bool:
    """Return whether shapes and dtypes permit a weighted Stage B dispatch.

    This intentionally does not synchronize device-resident expert indices.
    The Stage B kernel must validate index and generation bounds on-device
    before using routing data for any address calculation.
    """

    try:
        validate_hy3_weighted_top8(x, indices, scores, spec)
    except RecordQ4ContractError:
        return False
    return True


__all__ = [
    "HY3_RECORD_ALIGNMENT",
    "HY3_RECORD_Q4_BYTES",
    "HY3_RECORD_Q4_LAYOUT",
    "RecordQ4Component",
    "RecordQ4ContractError",
    "RecordQ4Layout",
    "WeightedTop8Contract",
    "is_hy3_record_q4_supported",
    "is_hy3_weighted_top8_statically_supported",
    "validate_hy3_record_q4",
    "validate_hy3_weighted_top8",
]
