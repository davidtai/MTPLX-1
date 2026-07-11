"""Resumable, compact Tencent Hy3 BF16-to-MLX-Q4 conversion.

This module deliberately never creates a quantized model checkpoint.  Routed
experts are quantized a projection at a time and written directly into the
aligned ``experts.bin`` records consumed by MTPLX.  Resident tensors are
written into safetensors files that contain no routed expert payloads.

The conversion is an append-only transaction:

* source shards are fully hashed before compute starts;
* every durable output work unit is fsynced before its checkpoint event;
* accepted checkpoint events are verified against bytes on resume; and
* the hidden work directory is renamed to the final artifact atomically.

MLX is imported lazily, after the benchmark/probe process gate is checked.
The planning, provenance, serialization, and resume helpers are therefore
testable without touching the GPU.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence


SOURCE_REPO = "tencent/Hy3"
SOURCE_REVISION = "716aa7241bd6d95896be4ebfc761162a9c4d49ef"
ORACLE_REPO = "pipenetwork/Hy3-4bit"
ORACLE_REVISION = "160619d3f96c8470350b6dac0ef033a8381551e3"
MODEL_KEY = "hy3-q4-native"

CHECKPOINT_FORMAT = "mtplx-hy3-native-checkpoint-v1"
PROVENANCE_FORMAT = "mtplx-hy3-native-provenance-v1"
PLAN_FORMAT = "mtplx-hy3-native-plan-v1"
CONVERTER_VERSION = 1

DEFAULT_ALIGNMENT = 16 * 1024
REAL_EXPERT_RECORD_BYTES = 10_616_832
RUN_LOCK_PATTERN = r"python.*(benchmark_streamed|probe_mtp)"
MAX_SAFETENSORS_HEADER = 128 * 1024 * 1024

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
LEAVES = ("weight", "scales", "biases")
COMPONENTS = tuple(
    f"{projection}.{leaf}" for projection in PROJECTIONS for leaf in LEAVES
)

_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

ANCILLARY_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Hy3NativeConversionError(RuntimeError):
    """Raised when conversion provenance or layout fails closed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pwrite_all(fd: int, payload: bytes | bytearray | memoryview, offset: int) -> None:
    view = memoryview(payload)
    position = offset
    while view:
        try:
            written = os.pwrite(fd, view, position)
        except InterruptedError:
            continue
        if written <= 0:
            raise Hy3NativeConversionError("short positional write")
        position += written
        view = view[written:]


def _pread_exact(fd: int, offset: int, length: int) -> bytes:
    chunks: list[bytes] = []
    position = offset
    remaining = length
    while remaining:
        try:
            chunk = os.pread(fd, remaining, position)
        except InterruptedError:
            continue
        if not chunk:
            raise Hy3NativeConversionError(
                f"short positional read at {position}; wanted {remaining} more bytes"
            )
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _safe_member(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise Hy3NativeConversionError(f"unsafe artifact member {relative!r}")
    path = root.joinpath(*candidate.parts)
    if not path.is_file():
        raise Hy3NativeConversionError(f"artifact member is missing: {path}")
    return path


def _tensor_nbytes(dtype: str, shape: Sequence[int]) -> int:
    if dtype not in DTYPE_BYTES:
        raise Hy3NativeConversionError(f"unsupported safetensors dtype {dtype!r}")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape
    ):
        raise Hy3NativeConversionError(f"invalid tensor shape {tuple(shape)!r}")
    return DTYPE_BYTES[dtype] * math.prod(shape)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise Hy3NativeConversionError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.dtype, self.shape)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class TensorLocation(TensorSpec):
    shard: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.length != self.nbytes:
            raise Hy3NativeConversionError(
                f"{self.name} has {self.length} bytes, expected {self.nbytes}"
            )


@dataclass(frozen=True)
class TensorPayload:
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        expected = _tensor_nbytes(self.dtype, self.shape)
        if len(self.data) != expected:
            raise Hy3NativeConversionError(
                f"tensor payload has {len(self.data)} bytes, expected {expected}"
            )

    def require(self, spec: TensorSpec) -> None:
        if self.dtype != spec.dtype or self.shape != spec.shape:
            raise Hy3NativeConversionError(
                f"quantizer returned {self.dtype}{self.shape} for {spec.name}; "
                f"oracle requires {spec.dtype}{spec.shape}"
            )


class SafeTensorInventory:
    """Header-only index plus positional tensor reads."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        index_path = self.root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise Hy3NativeConversionError(f"missing safetensors index: {index_path}")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Hy3NativeConversionError(f"invalid safetensors index: {exc}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise Hy3NativeConversionError("safetensors index has no weight_map")
        self.weight_map = {str(key): str(value) for key, value in weight_map.items()}
        self.index_metadata = index.get("metadata") or {}
        self.index_sha256 = _hash_file(index_path)
        self._headers: dict[str, tuple[int, dict[str, Any], int, str]] = {}
        self._locations: dict[str, TensorLocation] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.weight_map))

    @property
    def shards(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _header(self, shard: str) -> tuple[int, dict[str, Any], int, str]:
        cached = self._headers.get(shard)
        if cached is not None:
            return cached
        path = _safe_member(self.root, shard)
        size = path.stat().st_size
        with path.open("rb", buffering=0) as handle:
            length_raw = handle.read(8)
            if len(length_raw) != 8:
                raise Hy3NativeConversionError(f"short safetensors header: {path}")
            header_len = struct.unpack("<Q", length_raw)[0]
            if not 2 <= header_len <= MAX_SAFETENSORS_HEADER:
                raise Hy3NativeConversionError(
                    f"invalid safetensors header length {header_len}: {path}"
                )
            header_raw = handle.read(header_len)
        try:
            header = json.loads(header_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Hy3NativeConversionError(
                f"invalid safetensors header: {path}"
            ) from exc
        if not isinstance(header, dict):
            raise Hy3NativeConversionError(
                f"safetensors header is not an object: {path}"
            )
        cached = (
            8 + header_len,
            header,
            size,
            _sha256_bytes(length_raw + header_raw),
        )
        self._headers[shard] = cached
        return cached

    def info(self, name: str) -> TensorLocation:
        cached = self._locations.get(name)
        if cached is not None:
            return cached
        shard = self.weight_map.get(name)
        if shard is None:
            raise Hy3NativeConversionError(f"tensor is not indexed: {name}")
        data_start, header, size, _header_sha = self._header(shard)
        meta = header.get(name)
        if not isinstance(meta, dict):
            raise Hy3NativeConversionError(f"{name} is absent from {shard}'s header")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(dim, int) for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise Hy3NativeConversionError(f"invalid tensor metadata for {name}")
        start, end = offsets
        if start < 0 or end <= start or data_start + end > size:
            raise Hy3NativeConversionError(f"out-of-bounds tensor metadata for {name}")
        location = TensorLocation(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            shard=shard,
            offset=data_start + start,
            length=end - start,
        )
        self._locations[name] = location
        return location

    def read(self, name: str) -> bytes:
        info = self.info(name)
        path = _safe_member(self.root, info.shard)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            return _pread_exact(fd, info.offset, info.length)
        finally:
            os.close(fd)

    def shard_header(self, shard: str) -> tuple[int, int, str]:
        data_start, _header, size, header_sha = self._header(shard)
        return size, data_start, header_sha


class SafeTensorFileInventory:
    """Header inventory for one standalone safetensors file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise Hy3NativeConversionError(f"missing safetensors file: {self.path}")
        with self.path.open("rb", buffering=0) as handle:
            length_raw = handle.read(8)
            if len(length_raw) != 8:
                raise Hy3NativeConversionError(f"short safetensors header: {self.path}")
            header_len = struct.unpack("<Q", length_raw)[0]
            if not 2 <= header_len <= MAX_SAFETENSORS_HEADER:
                raise Hy3NativeConversionError("invalid standalone safetensors header")
            header_raw = handle.read(header_len)
        try:
            header = json.loads(header_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Hy3NativeConversionError(
                "invalid standalone safetensors JSON"
            ) from exc
        self.data_start = 8 + header_len
        self.header_sha256 = _sha256_bytes(length_raw + header_raw)
        metadata = header.pop("__metadata__", {})
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.locations: dict[str, TensorLocation] = {}
        size = self.path.stat().st_size
        for name, meta in header.items():
            if not isinstance(meta, dict):
                raise Hy3NativeConversionError(f"invalid metadata for {name}")
            offsets = meta.get("data_offsets")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or not isinstance(dtype, str)
                or not isinstance(shape, list)
                or not all(isinstance(dim, int) for dim in shape)
            ):
                raise Hy3NativeConversionError(f"invalid metadata for {name}")
            start, end = offsets
            if start < 0 or end <= start or self.data_start + end > size:
                raise Hy3NativeConversionError(f"out-of-bounds metadata for {name}")
            self.locations[name] = TensorLocation(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                shard=self.path.name,
                offset=self.data_start + start,
                length=end - start,
            )
        if not self.locations:
            raise Hy3NativeConversionError("standalone safetensors has no tensors")


@dataclass(frozen=True)
class Hy3Geometry:
    trunk_layers: int
    mtp_layers: int
    routed_layer_start: int
    expert_count: int
    hidden_size: int
    expert_hidden_size: int

    @property
    def mtp_layer_index(self) -> int:
        return self.trunk_layers

    @property
    def routed_layers(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.routed_layer_start,
                self.trunk_layers + self.mtp_layers,
            )
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Hy3Geometry":
        return cls(
            trunk_layers=int(config["num_hidden_layers"]),
            mtp_layers=int(config.get("num_nextn_predict_layers", 0)),
            routed_layer_start=int(config.get("first_k_dense_replace", 1)),
            expert_count=int(config["num_experts"]),
            hidden_size=int(config["hidden_size"]),
            expert_hidden_size=int(
                config.get("moe_intermediate_size", config["expert_hidden_dim"])
            ),
        )

    def require_official_hy3(self) -> None:
        expected = Hy3Geometry(80, 1, 1, 192, 4096, 1536)
        if self != expected:
            raise Hy3NativeConversionError(
                f"source config geometry {self} does not match pinned Hy3 {expected}"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "trunk_layers": self.trunk_layers,
            "mtp_layers": self.mtp_layers,
            "routed_layer_start": self.routed_layer_start,
            "expert_count": self.expert_count,
            "hidden_size": self.hidden_size,
            "expert_hidden_size": self.expert_hidden_size,
        }


@dataclass(frozen=True)
class ResidentUnit:
    unit_id: str
    source_tensor: str
    output_file: str
    targets: tuple[TensorSpec, ...]
    bits: int | None = None
    group_size: int = 64

    @property
    def quantized(self) -> bool:
        return self.bits is not None


@dataclass(frozen=True)
class ExpertComponentPlan:
    component: str
    source_tensor: str
    target: TensorSpec
    relative_offset: int


@dataclass(frozen=True)
class ExpertRecordPlan:
    layer: int
    expert: int
    offset: int
    components: tuple[ExpertComponentPlan, ...]

    @property
    def logical_bytes(self) -> int:
        return sum(component.target.nbytes for component in self.components)

    @property
    def record_id(self) -> str:
        return f"{self.layer}:{self.expert}"


@dataclass(frozen=True)
class SafeTensorOutputPlan:
    name: str
    tensors: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class NativeConversionPlan:
    source_root: Path
    oracle_root: Path
    source_revision: str
    oracle_revision: str
    source_index_sha256: str
    oracle_index_sha256: str
    source_config_sha256: str
    oracle_config_sha256: str
    geometry: Hy3Geometry
    alignment: int
    resident_files: tuple[SafeTensorOutputPlan, ...]
    resident_units: tuple[ResidentUnit, ...]
    records: tuple[ExpertRecordPlan, ...]
    bf16_head_tensors: tuple[str, ...]
    source_shards: tuple[str, ...]

    @property
    def expert_record_bytes(self) -> int:
        if not self.records:
            return 0
        sizes = {record.logical_bytes for record in self.records}
        if len(sizes) != 1:
            raise Hy3NativeConversionError(
                f"expert record sizes differ: {sorted(sizes)}"
            )
        return next(iter(sizes))

    @property
    def experts_file_size(self) -> int:
        if not self.records:
            return 0
        last = self.records[-1]
        return last.offset + last.logical_bytes

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        fixed = {
            "format": PLAN_FORMAT,
            "version": CONVERTER_VERSION,
            "source_revision": self.source_revision,
            "oracle_revision": self.oracle_revision,
            "source_index_sha256": self.source_index_sha256,
            "oracle_index_sha256": self.oracle_index_sha256,
            "source_config_sha256": self.source_config_sha256,
            "oracle_config_sha256": self.oracle_config_sha256,
            "geometry": self.geometry.to_dict(),
            "alignment": self.alignment,
            "source_shards": list(self.source_shards),
            "bf16_head_tensors": list(self.bf16_head_tensors),
        }
        digest.update(_canonical_json(fixed))
        for output in self.resident_files:
            digest.update(
                _canonical_json(
                    {
                        "file": output.name,
                        "tensors": [tensor.to_dict() for tensor in output.tensors],
                    }
                )
            )
        for unit in self.resident_units:
            digest.update(
                _canonical_json(
                    {
                        "unit_id": unit.unit_id,
                        "source": unit.source_tensor,
                        "output": unit.output_file,
                        "targets": [target.to_dict() for target in unit.targets],
                        "bits": unit.bits,
                        "group_size": unit.group_size,
                    }
                )
            )
        for record in self.records:
            digest.update(
                _canonical_json(
                    {
                        "layer": record.layer,
                        "expert": record.expert,
                        "offset": record.offset,
                        "components": [
                            {
                                "component": component.component,
                                "source": component.source_tensor,
                                "target": component.target.to_dict(),
                                "relative_offset": component.relative_offset,
                            }
                            for component in record.components
                        ],
                    }
                )
            )
        return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Hy3NativeConversionError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Hy3NativeConversionError(f"{path} must contain a JSON object")
    return value


def _source_name_for_resident(target: str) -> str:
    return target.replace(".mlp.router.expert_bias", ".mlp.expert_bias")


def _is_oracle_expert(name: str) -> bool:
    return ".mlp.switch_mlp." in name or _EXPERT_SOURCE_RE.match(name) is not None


def _quantized_targets(
    inventory: SafeTensorInventory,
    weight_name: str,
) -> tuple[TensorSpec, ...] | None:
    if not weight_name.endswith(".weight") or not inventory.has(weight_name):
        return None
    weight = inventory.info(weight_name)
    if weight.dtype != "U32":
        return None
    prefix = weight_name[: -len(".weight")]
    names = tuple(prefix + "." + leaf for leaf in LEAVES)
    if not all(inventory.has(name) for name in names):
        raise Hy3NativeConversionError(
            f"oracle quantized module is missing leaves: {prefix}"
        )
    return tuple(
        TensorSpec(
            name=name,
            dtype=inventory.info(name).dtype,
            shape=inventory.info(name).shape,
        )
        for name in names
    )


def _router_bits(name: str) -> int:
    return 8 if ".mlp.router.gate.weight" in name else 4


def _expert_component_specs(
    oracle: SafeTensorInventory,
    geometry: Hy3Geometry,
) -> dict[str, TensorSpec]:
    result: dict[str, TensorSpec] = {}
    reference_layer = geometry.routed_layer_start
    for component in COMPONENTS:
        name = f"model.layers.{reference_layer}.mlp.switch_mlp.{component}"
        info = oracle.info(name)
        if len(info.shape) < 2 or info.shape[0] != geometry.expert_count:
            raise Hy3NativeConversionError(
                f"oracle expert tensor has unexpected shape: {name} {info.shape}"
            )
        wanted_dtype = "U32" if component.endswith(".weight") else "BF16"
        if info.dtype != wanted_dtype:
            raise Hy3NativeConversionError(
                f"oracle expert tensor {name} is {info.dtype}, expected {wanted_dtype}"
            )
        result[component] = TensorSpec(
            name=name,
            dtype=info.dtype,
            shape=info.shape[1:],
        )

    for layer in range(geometry.routed_layer_start, geometry.trunk_layers):
        for component, reference in result.items():
            name = f"model.layers.{layer}.mlp.switch_mlp.{component}"
            info = oracle.info(name)
            if info.dtype != reference.dtype or info.shape[1:] != reference.shape:
                raise Hy3NativeConversionError(
                    f"oracle expert layout drift at {name}: {info.dtype}{info.shape}"
                )
    return result


def _layer_number(name: str) -> int | None:
    match = _LAYER_RE.match(name)
    return None if match is None else int(match.group(1))


def _build_resident_plan(
    source: SafeTensorInventory,
    oracle: SafeTensorInventory,
    geometry: Hy3Geometry,
) -> tuple[tuple[SafeTensorOutputPlan, ...], tuple[ResidentUnit, ...]]:
    oracle_residents = [name for name in oracle.keys if not _is_oracle_expert(name)]
    consumed: set[str] = set()
    groups: dict[str, list[TensorSpec]] = {}
    units: list[ResidentUnit] = []
    oracle_shards = sorted({oracle.weight_map[name] for name in oracle_residents})
    output_names = {
        shard: f"model-resident-{index:05d}-of-{len(oracle_shards):05d}.safetensors"
        for index, shard in enumerate(oracle_shards, 1)
    }
    for target_name in sorted(oracle_residents):
        if target_name in consumed:
            continue
        # Quantized parameter leaves sort before ``weight``.  They are emitted
        # together from the single BF16 source weight when that unit is
        # visited, never mistaken for independent source tensors.
        if target_name.endswith((".biases", ".scales")):
            prefix, _leaf = target_name.rsplit(".", 1)
            weight_name = prefix + ".weight"
            if oracle.has(weight_name) and oracle.info(weight_name).dtype == "U32":
                continue
        layer = _layer_number(target_name)
        if layer is not None and layer >= geometry.trunk_layers:
            raise Hy3NativeConversionError(
                f"layout oracle unexpectedly contains MTP layer {layer}"
            )
        output_file = output_names[oracle.weight_map[target_name]]
        quant_targets = _quantized_targets(oracle, target_name)
        if quant_targets is not None:
            source_name = _source_name_for_resident(target_name)
            source_info = source.info(source_name)
            if source_info.dtype != "BF16":
                raise Hy3NativeConversionError(
                    f"quantization source {source_name} must be BF16, found {source_info.dtype}"
                )
            target_names = {target.name for target in quant_targets}
            if any(
                oracle.weight_map[name] != oracle.weight_map[target_name]
                for name in target_names
            ):
                raise Hy3NativeConversionError(
                    f"oracle splits quantization leaves across shards: {target_name}"
                )
            groups.setdefault(output_file, []).extend(quant_targets)
            units.append(
                ResidentUnit(
                    unit_id=f"resident:{target_name}",
                    source_tensor=source_name,
                    output_file=output_file,
                    targets=quant_targets,
                    bits=_router_bits(target_name),
                )
            )
            consumed.update(target_names)
            continue
        target = TensorSpec(
            name=target_name,
            dtype=oracle.info(target_name).dtype,
            shape=oracle.info(target_name).shape,
        )
        source_name = _source_name_for_resident(target_name)
        source_info = source.info(source_name)
        if source_info.dtype != target.dtype or source_info.shape != target.shape:
            raise Hy3NativeConversionError(
                f"resident source/oracle mismatch for {target_name}: "
                f"{source_info.dtype}{source_info.shape} != {target.dtype}{target.shape}"
            )
        groups.setdefault(output_file, []).append(target)
        units.append(
            ResidentUnit(
                unit_id=f"resident:{target_name}",
                source_tensor=source_name,
                output_file=output_file,
                targets=(target,),
            )
        )
        consumed.add(target_name)

    # Layer 80 is intentionally one separately named resident file.  Common
    # modules inherit the layer-79 oracle contract; MTP-only projections/norms
    # pass through bit-exactly from the official source.
    layer80_file = "layer80-residents-q.safetensors"
    layer80_prefix = f"model.layers.{geometry.mtp_layer_index}."
    reference_prefix = f"model.layers.{geometry.trunk_layers - 1}."
    layer80_sources = [
        name
        for name in source.keys
        if name.startswith(layer80_prefix) and _EXPERT_SOURCE_RE.match(name) is None
    ]
    layer80_targets: list[TensorSpec] = []
    for source_name in sorted(layer80_sources):
        # The standalone Hy3 MTP loader intentionally consumes the source
        # spelling here and maps it onto ``mtp_block.mlp.router`` itself.
        # Trunk residents use the oracle's ``mlp.router.expert_bias`` spelling.
        target_name = source_name
        reference_name = reference_prefix + target_name[len(layer80_prefix) :]
        if oracle.has(reference_name):
            quant_targets = _quantized_targets(oracle, reference_name)
        else:
            quant_targets = None
        if quant_targets is not None:
            targets = tuple(
                TensorSpec(
                    name=layer80_prefix + target.name[len(reference_prefix) :],
                    dtype=target.dtype,
                    shape=target.shape,
                )
                for target in quant_targets
            )
            source_info = source.info(source_name)
            if source_info.dtype != "BF16":
                raise Hy3NativeConversionError(
                    f"layer-80 quantization source must be BF16: {source_name}"
                )
            layer80_targets.extend(targets)
            units.append(
                ResidentUnit(
                    unit_id=f"resident:{target_name}",
                    source_tensor=source_name,
                    output_file=layer80_file,
                    targets=targets,
                    bits=_router_bits(target_name),
                )
            )
            continue
        source_info = source.info(source_name)
        target = TensorSpec(
            name=target_name,
            dtype=source_info.dtype,
            shape=source_info.shape,
        )
        if oracle.has(reference_name):
            reference = oracle.info(reference_name)
            if reference.dtype != target.dtype or reference.shape != target.shape:
                raise Hy3NativeConversionError(
                    f"layer-80 resident differs from oracle convention: {target_name}"
                )
        layer80_targets.append(target)
        units.append(
            ResidentUnit(
                unit_id=f"resident:{target_name}",
                source_tensor=source_name,
                output_file=layer80_file,
                targets=(target,),
            )
        )

    files = [
        SafeTensorOutputPlan(
            name=name, tensors=tuple(sorted(tensors, key=lambda x: x.name))
        )
        for name, tensors in sorted(groups.items())
    ]
    files.append(
        SafeTensorOutputPlan(
            name=layer80_file,
            tensors=tuple(sorted(layer80_targets, key=lambda x: x.name)),
        )
    )
    if len({unit.unit_id for unit in units}) != len(units):
        raise Hy3NativeConversionError("resident conversion unit IDs are not unique")
    return tuple(files), tuple(sorted(units, key=lambda unit: unit.unit_id))


def _build_record_plan(
    source: SafeTensorInventory,
    oracle: SafeTensorInventory,
    geometry: Hy3Geometry,
    *,
    alignment: int,
) -> tuple[ExpertRecordPlan, ...]:
    reference = _expert_component_specs(oracle, geometry)
    records: list[ExpertRecordPlan] = []
    cursor = 0
    for layer in geometry.routed_layers:
        for expert in range(geometry.expert_count):
            offset = _align_up(cursor, alignment)
            relative = 0
            components: list[ExpertComponentPlan] = []
            for component in COMPONENTS:
                projection, leaf = component.split(".", 1)
                source_name = (
                    f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                )
                source_info = source.info(source_name)
                if source_info.dtype != "BF16":
                    raise Hy3NativeConversionError(
                        f"expert source must be BF16: {source_name}"
                    )
                target_ref = reference[component]
                target = TensorSpec(
                    name=f"model.layers.{layer}.mlp.switch_mlp.{component}",
                    dtype=target_ref.dtype,
                    shape=target_ref.shape,
                )
                components.append(
                    ExpertComponentPlan(
                        component=component,
                        source_tensor=source_name,
                        target=target,
                        relative_offset=relative,
                    )
                )
                relative += target.nbytes
            record = ExpertRecordPlan(
                layer=layer,
                expert=expert,
                offset=offset,
                components=tuple(components),
            )
            if (
                tuple(component.component for component in record.components)
                != COMPONENTS
            ):
                raise Hy3NativeConversionError("expert record leaf order drifted")
            records.append(record)
            cursor = offset + record.logical_bytes
    return tuple(records)


def build_conversion_plan(
    source_root: Path | str,
    oracle_root: Path | str,
    *,
    source_revision: str = SOURCE_REVISION,
    oracle_revision: str = ORACLE_REVISION,
    alignment: int = DEFAULT_ALIGNMENT,
    enforce_official_geometry: bool = True,
) -> NativeConversionPlan:
    """Inspect official/oracle headers and return a deterministic direct-write plan."""

    if enforce_official_geometry:
        if alignment != DEFAULT_ALIGNMENT:
            raise Hy3NativeConversionError(
                f"official native sidecar alignment must be {DEFAULT_ALIGNMENT}, "
                f"not {alignment}"
            )
        if source_revision != SOURCE_REVISION:
            raise Hy3NativeConversionError(
                f"official Hy3 conversion is pinned to {SOURCE_REVISION}, "
                f"not {source_revision}"
            )
        if oracle_revision != ORACLE_REVISION:
            raise Hy3NativeConversionError(
                f"official layout oracle is pinned to {ORACLE_REVISION}, "
                f"not {oracle_revision}"
            )
    source = SafeTensorInventory(source_root)
    oracle = SafeTensorInventory(oracle_root)
    source_config_path = source.root / "config.json"
    oracle_config_path = oracle.root / "config.json"
    source_config = _load_json(source_config_path)
    geometry = Hy3Geometry.from_config(source_config)
    if enforce_official_geometry:
        geometry.require_official_hy3()
        if len(source.shards) != 99:
            raise Hy3NativeConversionError(
                f"pinned Hy3 source must contain 99 indexed shards, found {len(source.shards)}"
            )
    if geometry.mtp_layers != 1:
        raise Hy3NativeConversionError(
            "native Hy3 packaging requires exactly one MTP layer"
        )
    _verify_revision_evidence(source.root, source_revision, source.shards)
    _verify_revision_evidence(oracle.root, oracle_revision, ())
    resident_files, resident_units = _build_resident_plan(source, oracle, geometry)
    records = _build_record_plan(
        source,
        oracle,
        geometry,
        alignment=alignment,
    )
    bf16_prefix = f"model.layers.{geometry.mtp_layer_index}."
    bf16_tensors = tuple(name for name in source.keys if name.startswith(bf16_prefix))
    if not bf16_tensors:
        raise Hy3NativeConversionError("official source has no layer-80 tensors")
    plan = NativeConversionPlan(
        source_root=source.root,
        oracle_root=oracle.root,
        source_revision=source_revision,
        oracle_revision=oracle_revision,
        source_index_sha256=source.index_sha256,
        oracle_index_sha256=oracle.index_sha256,
        source_config_sha256=_hash_file(source_config_path),
        oracle_config_sha256=_hash_file(oracle_config_path),
        geometry=geometry,
        alignment=alignment,
        resident_files=resident_files,
        resident_units=resident_units,
        records=records,
        bf16_head_tensors=bf16_tensors,
        source_shards=source.shards,
    )
    if (
        enforce_official_geometry
        and plan.expert_record_bytes != REAL_EXPERT_RECORD_BYTES
    ):
        raise Hy3NativeConversionError(
            f"expert records are {plan.expert_record_bytes:,} bytes, expected "
            f"{REAL_EXPERT_RECORD_BYTES:,}"
        )
    if enforce_official_geometry:
        from mtplx.expert_streaming_models import get_model_spec

        spec = get_model_spec(MODEL_KEY)
        resident_bytes = sum(
            tensor.nbytes for output in plan.resident_files for tensor in output.tensors
        )
        routed_bytes = sum(record.logical_bytes for record in plan.records)
        mtp_resident_bytes = sum(
            tensor.nbytes
            for output in plan.resident_files
            if output.name == "layer80-residents-q.safetensors"
            for tensor in output.tensors
        )
        bf16_bytes = sum(source.info(name).length for name in plan.bf16_head_tensors)
        expected = {
            "resident_bytes": spec.resident_bytes,
            "routed_bytes": spec.routed_expert_bytes,
            "total_tensor_bytes": spec.total_tensor_bytes,
            "mtp_q4_resident_bytes": spec.mtp_q4_resident_bytes,
            "mtp_bf16_tensor_bytes": spec.mtp_bf16_tensor_bytes,
        }
        actual = {
            "resident_bytes": resident_bytes,
            "routed_bytes": routed_bytes,
            "total_tensor_bytes": resident_bytes + routed_bytes,
            "mtp_q4_resident_bytes": mtp_resident_bytes,
            "mtp_bf16_tensor_bytes": bf16_bytes,
        }
        if actual != expected:
            raise Hy3NativeConversionError(
                f"planned native artifact bytes drifted from {MODEL_KEY}: "
                f"actual={actual}, expected={expected}"
            )
    return plan


def _metadata_path(root: Path, filename: str) -> Path:
    return root / ".cache" / "huggingface" / "download" / f"{filename}.metadata"


def _download_metadata(root: Path, filename: str) -> tuple[str, str | None] | None:
    path = _metadata_path(root, filename)
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Hy3NativeConversionError(
            f"could not read Hugging Face metadata {path}"
        ) from exc
    if not lines or not lines[0]:
        raise Hy3NativeConversionError(f"invalid Hugging Face metadata {path}")
    expected_hash = lines[1].lower() if len(lines) > 1 else None
    if expected_hash and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        expected_hash = None
    return lines[0], expected_hash


def _verify_revision_evidence(
    root: Path,
    revision: str,
    shard_names: Iterable[str],
) -> None:
    snapshot_evidence = root.parent.name == "snapshots" and root.name == revision
    index_metadata = _download_metadata(root, "model.safetensors.index.json")
    metadata_evidence = index_metadata is not None and index_metadata[0] == revision
    marker = root / ".mtplx-source-revision"
    marker_evidence = (
        marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision
    )
    if not (snapshot_evidence or metadata_evidence or marker_evidence):
        raise Hy3NativeConversionError(
            f"could not prove that {root} is revision {revision}"
        )
    for shard in shard_names:
        metadata = _download_metadata(root, shard)
        if metadata is not None and metadata[0] != revision:
            raise Hy3NativeConversionError(
                f"{shard} was downloaded from {metadata[0]}, not {revision}"
            )


class Quantizer(Protocol):
    @property
    def provenance(self) -> dict[str, Any]: ...

    def quantize(
        self,
        source: TensorSpec,
        raw: bytes,
        *,
        bits: int,
        group_size: int,
    ) -> tuple[TensorPayload, TensorPayload, TensorPayload]: ...


class GPUProcessRunLock:
    """Fail closed while benchmark/probe Python processes are running."""

    def __init__(self, pattern: str = RUN_LOCK_PATTERN) -> None:
        self.pattern = pattern

    def matching_pids(self) -> tuple[int, ...]:
        result = subprocess.run(
            ["pgrep", "-f", self.pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 1:
            return ()
        if result.returncode != 0:
            raise Hy3NativeConversionError(
                f"run-lock pgrep failed with exit status {result.returncode}"
            )
        return tuple(int(line) for line in result.stdout.splitlines() if line.strip())

    def require_clear(self) -> None:
        pids = self.matching_pids()
        if pids:
            raise Hy3NativeConversionError(
                "GPU compute is blocked by benchmark/probe processes: "
                + ", ".join(str(pid) for pid in pids)
            )


class MlxAffineQuantizer:
    """One-BF16-tensor-at-a-time MLX affine quantizer."""

    def __init__(self, *, run_lock: GPUProcessRunLock | None = None) -> None:
        self.run_lock = run_lock or GPUProcessRunLock()
        self.run_lock.require_clear()
        # Import only after the process gate.  Importing and constructing the
        # BF16 array are part of the protected compute phase.
        import mlx.core as mx
        import numpy as np

        self.mx = mx
        self.np = np
        self._provenance = {
            "implementation": f"{__name__}.MlxAffineQuantizer",
            "converter_version": CONVERTER_VERSION,
            "module_sha256": _hash_file(Path(__file__)),
            "mlx_version": importlib.metadata.version("mlx"),
            "numpy_version": importlib.metadata.version("numpy"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "mode": "affine",
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _raw(self, value: Any, dtype: str) -> bytes:
        mx = self.mx
        np = self.np
        if dtype == "BF16":
            array = np.asarray(value.view(mx.uint16), dtype="<u2")
        elif dtype == "U32":
            array = np.asarray(value, dtype="<u4")
        else:
            raise Hy3NativeConversionError(f"cannot serialize MLX dtype {dtype}")
        return array.tobytes(order="C")

    def quantize(
        self,
        source: TensorSpec,
        raw: bytes,
        *,
        bits: int,
        group_size: int,
    ) -> tuple[TensorPayload, TensorPayload, TensorPayload]:
        self.run_lock.require_clear()
        if source.dtype != "BF16":
            raise Hy3NativeConversionError(
                f"MLX affine source must be BF16, found {source.dtype}"
            )
        if bits not in {4, 8} or group_size != 64:
            raise Hy3NativeConversionError(
                f"unsupported native quantization q{bits}/gs{group_size}"
            )
        mx = self.mx
        np = self.np
        words = np.frombuffer(raw, dtype="<u2")
        value = mx.array(words).view(mx.bfloat16).reshape(source.shape)
        weight, scales, biases = mx.quantize(
            value,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        mx.eval(weight, scales, biases)
        payloads = (
            TensorPayload("U32", tuple(weight.shape), self._raw(weight, "U32")),
            TensorPayload("BF16", tuple(scales.shape), self._raw(scales, "BF16")),
            TensorPayload("BF16", tuple(biases.shape), self._raw(biases, "BF16")),
        )
        del value, weight, scales, biases
        mx.clear_cache()
        return payloads


class CheckpointJournal:
    """Append-only fsynced acceptance journal."""

    def __init__(self, path: Path, plan_fingerprint: str) -> None:
        self.path = path
        self.plan_fingerprint = plan_fingerprint
        self.events: dict[tuple[str, str], dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()
        else:
            header = {
                "format": CHECKPOINT_FORMAT,
                "event": "header",
                "plan_fingerprint": plan_fingerprint,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            fd = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o644,
            )
            try:
                os.write(fd, _canonical_json(header) + b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_directory(self.path.parent)

    def _load(self) -> None:
        try:
            mode = self.path.lstat().st_mode
        except OSError as exc:
            raise Hy3NativeConversionError(f"could not stat checkpoint: {exc}") from exc
        if not stat.S_ISREG(mode):
            raise Hy3NativeConversionError(
                "conversion checkpoint is not a regular file"
            )
        try:
            payload = self.path.read_bytes()
        except OSError as exc:
            raise Hy3NativeConversionError(f"could not read checkpoint: {exc}") from exc
        lines = payload.splitlines(keepends=True)
        if not lines:
            raise Hy3NativeConversionError("empty conversion checkpoint")
        if not lines[0].endswith(b"\n"):
            raise Hy3NativeConversionError("torn conversion checkpoint header")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise Hy3NativeConversionError(
                "invalid conversion checkpoint header"
            ) from exc
        if (
            header.get("format") != CHECKPOINT_FORMAT
            or header.get("event") != "header"
            or header.get("plan_fingerprint") != self.plan_fingerprint
        ):
            raise Hy3NativeConversionError(
                "checkpoint belongs to a different conversion plan"
            )
        valid_bytes = len(lines[0])
        for line_index, line in enumerate(lines[1:], 1):
            complete = line.endswith(b"\n")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A torn last append was never fsynced/accepted.
                if line_index == len(lines) - 1:
                    break
                raise Hy3NativeConversionError("invalid checkpoint event")
            if not complete:
                # Journal writers always append a newline before fsync.  Even
                # syntactically valid JSON without it is an unaccepted append.
                if line_index == len(lines) - 1:
                    break
                raise Hy3NativeConversionError("torn checkpoint event")
            kind = event.get("event")
            event_id = event.get("id")
            if isinstance(kind, str) and isinstance(event_id, str):
                self.events[(kind, event_id)] = event
            valid_bytes += len(line)
        if valid_bytes != len(payload):
            fd = os.open(
                self.path,
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.ftruncate(fd, valid_bytes)
                os.fsync(fd)
            finally:
                os.close(fd)

    def get(self, kind: str, event_id: str) -> dict[str, Any] | None:
        return self.events.get((kind, event_id))

    def accept(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        event_id = event.get("id")
        if not isinstance(kind, str) or not isinstance(event_id, str):
            raise Hy3NativeConversionError("checkpoint events require string event/id")
        payload = _canonical_json(event) + b"\n"
        fd = os.open(
            self.path,
            os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise Hy3NativeConversionError("short checkpoint write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self.events[(kind, event_id)] = dict(event)


def _safetensors_header(
    tensors: Sequence[TensorSpec],
    *,
    metadata: dict[str, str],
) -> tuple[bytes, dict[str, tuple[int, int]], int]:
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for tensor in sorted(tensors, key=lambda item: item.name):
        if tensor.name in header:
            raise Hy3NativeConversionError(f"duplicate output tensor {tensor.name}")
        end = cursor + tensor.nbytes
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [cursor, end],
        }
        offsets[tensor.name] = (cursor, end)
        cursor = end
    encoded = _canonical_json(header)
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    prefix = struct.pack("<Q", len(encoded)) + encoded
    return prefix, offsets, cursor


class SafeTensorDirectWriter:
    """Predeclared safetensors file supporting positional resumable writes."""

    def __init__(
        self,
        path: Path,
        tensors: Sequence[TensorSpec],
        *,
        metadata: dict[str, str],
    ) -> None:
        self.path = path
        self.tensors = {tensor.name: tensor for tensor in tensors}
        self.prefix, relative, payload_bytes = _safetensors_header(
            tensors,
            metadata=metadata,
        )
        self.data_start = len(self.prefix)
        self.offsets = {
            name: (self.data_start + start, self.data_start + end)
            for name, (start, end) in relative.items()
        }
        self.size = self.data_start + payload_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        self.fd = os.open(self.path, flags, 0o644)
        if not stat.S_ISREG(os.fstat(self.fd).st_mode):
            os.close(self.fd)
            raise Hy3NativeConversionError(
                f"resident output is not a regular file: {self.path}"
            )
        current = os.fstat(self.fd).st_size
        if current == 0:
            _pwrite_all(self.fd, self.prefix, 0)
            os.ftruncate(self.fd, self.size)
            os.fsync(self.fd)
            _fsync_directory(self.path.parent)
        else:
            if (
                current != self.size
                or _pread_exact(self.fd, 0, len(self.prefix)) != self.prefix
            ):
                os.close(self.fd)
                raise Hy3NativeConversionError(
                    f"partial safetensors plan mismatch: {self.path}"
                )

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def write(self, spec: TensorSpec, payload: TensorPayload) -> None:
        payload.require(spec)
        start, end = self.offsets[spec.name]
        if end - start != len(payload.data):
            raise Hy3NativeConversionError("direct writer payload size mismatch")
        _pwrite_all(self.fd, payload.data, start)

    def fsync(self) -> None:
        os.fsync(self.fd)

    def digest(self, specs: Sequence[TensorSpec]) -> str:
        digest = hashlib.sha256()
        for spec in specs:
            start, end = self.offsets[spec.name]
            digest.update(_canonical_json(spec.to_dict()))
            digest.update(_pread_exact(self.fd, start, end - start))
        return digest.hexdigest()


def _payload_digest(
    specs: Sequence[TensorSpec], payloads: Sequence[TensorPayload]
) -> str:
    digest = hashlib.sha256()
    for spec, payload in zip(specs, payloads, strict=True):
        payload.require(spec)
        digest.update(_canonical_json(spec.to_dict()))
        digest.update(payload.data)
    return digest.hexdigest()


def _source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _hash_source_shards(
    plan: NativeConversionPlan,
    journal: CheckpointJournal,
    *,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    require_hf_hashes = plan.geometry == Hy3Geometry(80, 1, 1, 192, 4096, 1536)
    for shard in plan.source_shards:
        path = _safe_member(plan.source_root, shard)
        stat = _source_stat(path)
        metadata = _download_metadata(plan.source_root, shard)
        if require_hf_hashes and (
            metadata is None
            or metadata[0] != plan.source_revision
            or metadata[1] is None
        ):
            raise Hy3NativeConversionError(
                f"official source shard lacks pinned LFS hash metadata: {shard}"
            )
        expected = metadata[1] if metadata is not None else None
        accepted = journal.get("source_shard", shard)
        if (
            accepted is not None
            and accepted.get("stat") == stat
            and (expected is None or accepted.get("sha256") == expected)
        ):
            result[shard] = accepted
            continue
        digest = _hash_file(path)
        if _source_stat(path) != stat:
            raise Hy3NativeConversionError(
                f"source shard changed while it was being hashed: {shard}"
            )
        if expected is not None and digest != expected:
            raise Hy3NativeConversionError(
                f"local source hash mismatch for {shard}: {digest} != {expected}"
            )
        event = {
            "event": "source_shard",
            "id": shard,
            "sha256": digest,
            "stat": stat,
            "expected_lfs_sha256": expected,
        }
        journal.accept(event)
        result[shard] = event
        if on_event is not None:
            on_event(event)
    return result


def _assert_source_shards_unchanged(
    plan: NativeConversionPlan,
    accepted_shards: dict[str, dict[str, Any]],
) -> None:
    """Prove the quantizer read the same pinned files hashed at preparation."""

    require_hf_hashes = plan.geometry == Hy3Geometry(80, 1, 1, 192, 4096, 1536)
    for shard in plan.source_shards:
        accepted = accepted_shards.get(shard)
        if accepted is None:
            raise Hy3NativeConversionError(
                f"source shard has no accepted preparation event: {shard}"
            )
        path = _safe_member(plan.source_root, shard)
        if _source_stat(path) != accepted.get("stat"):
            raise Hy3NativeConversionError(
                f"source shard changed during conversion: {shard}"
            )
        metadata = _download_metadata(plan.source_root, shard)
        if require_hf_hashes:
            if metadata is None or metadata[0] != plan.source_revision:
                raise Hy3NativeConversionError(
                    f"source shard lost pinned revision metadata: {shard}"
                )
            expected_hash = metadata[1]
            if expected_hash is None or expected_hash != accepted.get("sha256"):
                raise Hy3NativeConversionError(
                    f"source shard metadata/hash drifted during conversion: {shard}"
                )
        elif metadata is not None and metadata[0] != plan.source_revision:
            raise Hy3NativeConversionError(
                f"source shard revision drifted during conversion: {shard}"
            )


def _verify_and_bind_bf16_head(
    plan: NativeConversionPlan,
    work_root: Path,
    journal: CheckpointJournal,
    *,
    source_head: Path,
    copy_head: bool,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    source_head = source_head.expanduser().resolve()
    inventory = SafeTensorFileInventory(source_head)
    if inventory.metadata.get("source_repo") != SOURCE_REPO:
        raise Hy3NativeConversionError(
            "BF16 head sidecar has wrong source_repo metadata"
        )
    if inventory.metadata.get("source_revision") != plan.source_revision:
        raise Hy3NativeConversionError(
            "BF16 head sidecar has wrong source_revision metadata"
        )
    if set(inventory.locations) != set(plan.bf16_head_tensors):
        raise Hy3NativeConversionError(
            "BF16 head tensor set does not match official layer 80"
        )
    stat = _source_stat(source_head)
    accepted = journal.get("bf16_head", "layer80-bf16.safetensors")
    destination = work_root / "layer80-bf16.safetensors"
    wanted_storage = "copy" if copy_head else "hardlink"
    if (
        accepted is not None
        and accepted.get("source_stat") == stat
        and accepted.get("storage") == wanted_storage
        and accepted.get("source_repo") == SOURCE_REPO
        and accepted.get("source_revision") == plan.source_revision
        and destination.is_file()
    ):
        if destination.stat().st_size == stat["size"]:
            if copy_head and _hash_file(destination) == accepted.get("sha256"):
                return accepted
            if not copy_head and os.path.samefile(source_head, destination):
                return accepted

    source_inventory = SafeTensorInventory(plan.source_root)
    source_fds: dict[str, int] = {}
    head_fd = os.open(source_head, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    try:
        digest.update(_pread_exact(head_fd, 0, inventory.data_start))
        cursor = inventory.data_start
        for name, head_info in sorted(
            inventory.locations.items(), key=lambda item: item[1].offset
        ):
            if head_info.offset != cursor:
                raise Hy3NativeConversionError("BF16 head payload is not contiguous")
            official = source_inventory.info(name)
            if (
                head_info.dtype != official.dtype
                or head_info.shape != official.shape
                or head_info.length != official.length
            ):
                raise Hy3NativeConversionError(f"BF16 head layout mismatch for {name}")
            source_fd = source_fds.get(official.shard)
            if source_fd is None:
                source_fd = os.open(
                    _safe_member(plan.source_root, official.shard),
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                )
                source_fds[official.shard] = source_fd
            remaining = head_info.length
            head_position = head_info.offset
            source_position = official.offset
            while remaining:
                length = min(16 * 1024 * 1024, remaining)
                head_chunk = _pread_exact(head_fd, head_position, length)
                source_chunk = _pread_exact(source_fd, source_position, length)
                if head_chunk != source_chunk:
                    raise Hy3NativeConversionError(
                        f"BF16 head payload differs from official source: {name}"
                    )
                digest.update(head_chunk)
                head_position += length
                source_position += length
                remaining -= length
            cursor += head_info.length
        if cursor != stat["size"]:
            raise Hy3NativeConversionError("BF16 head has trailing unindexed bytes")
    finally:
        os.close(head_fd)
        for fd in source_fds.values():
            os.close(fd)
    if _source_stat(source_head) != stat:
        raise Hy3NativeConversionError(
            "BF16 head source changed while it was being verified"
        )

    if destination.exists():
        destination.unlink()
    if copy_head:
        temporary = destination.with_name(".layer80-bf16.safetensors.copying")
        temporary.unlink(missing_ok=True)
        with (
            source_head.open("rb", buffering=0) as src,
            temporary.open("xb", buffering=0) as out,
        ):
            shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, destination)
    else:
        try:
            os.link(source_head, destination)
        except OSError as exc:
            raise Hy3NativeConversionError(
                "could not hardlink layer80-bf16.safetensors; rerun with explicit "
                "copy_head=True only if the extra disk allocation is acceptable"
            ) from exc
    _fsync_directory(work_root)
    event = {
        "event": "bf16_head",
        "id": "layer80-bf16.safetensors",
        "sha256": digest.hexdigest(),
        "size": stat["size"],
        "header_bytes": inventory.data_start,
        "header_sha256": inventory.header_sha256,
        "source_stat": stat,
        "storage": "copy" if copy_head else "hardlink",
        "tensor_count": len(inventory.locations),
        "source_repo": SOURCE_REPO,
        "source_revision": plan.source_revision,
    }
    journal.accept(event)
    if on_event is not None:
        on_event(event)
    return event


def _resident_metadata(plan: NativeConversionPlan, output_name: str) -> dict[str, str]:
    return {
        "producer": f"{__name__} v{CONVERTER_VERSION}",
        "source_repo": SOURCE_REPO,
        "source_revision": plan.source_revision,
        "layout_oracle_repo": ORACLE_REPO,
        "layout_oracle_revision": plan.oracle_revision,
        "scope": "layer-80 residents"
        if output_name.startswith("layer80-")
        else "resident-only",
    }


def _write_residents(
    plan: NativeConversionPlan,
    work_root: Path,
    journal: CheckpointJournal,
    quantizer: Quantizer,
    *,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> None:
    source = SafeTensorInventory(plan.source_root)
    writers = {
        output.name: SafeTensorDirectWriter(
            work_root / output.name,
            output.tensors,
            metadata=_resident_metadata(plan, output.name),
        )
        for output in plan.resident_files
    }
    try:
        for unit in plan.resident_units:
            writer = writers[unit.output_file]
            accepted = journal.get("resident", unit.unit_id)
            if accepted is not None:
                if writer.digest(unit.targets) == accepted.get("sha256"):
                    continue
            source_info = source.info(unit.source_tensor)
            raw = source.read(unit.source_tensor)
            source_spec = TensorSpec(
                name=unit.source_tensor,
                dtype=source_info.dtype,
                shape=source_info.shape,
            )
            if unit.quantized:
                payloads = quantizer.quantize(
                    source_spec,
                    raw,
                    bits=int(unit.bits),
                    group_size=unit.group_size,
                )
            else:
                payloads = (TensorPayload(source_info.dtype, source_info.shape, raw),)
            if len(payloads) != len(unit.targets):
                raise Hy3NativeConversionError(
                    f"quantizer returned wrong leaf count for {unit.source_tensor}"
                )
            digest = _payload_digest(unit.targets, payloads)
            for target, payload in zip(unit.targets, payloads, strict=True):
                writer.write(target, payload)
            # Output bytes become durable before the acceptance event.
            writer.fsync()
            event = {
                "event": "resident",
                "id": unit.unit_id,
                "source_tensor": unit.source_tensor,
                "output_file": unit.output_file,
                "sha256": digest,
                "bits": unit.bits,
                "group_size": unit.group_size if unit.quantized else None,
            }
            journal.accept(event)
            if on_event is not None:
                on_event(event)
    finally:
        for writer in writers.values():
            writer.close()


def _record_payload(
    record: ExpertRecordPlan,
    source: SafeTensorInventory,
    quantizer: Quantizer,
) -> bytes:
    chunks: list[bytes] = []
    by_projection: dict[str, tuple[ExpertComponentPlan, ...]] = {}
    for component in record.components:
        projection = component.component.split(".", 1)[0]
        by_projection.setdefault(projection, tuple())
        by_projection[projection] += (component,)
    for projection in PROJECTIONS:
        components = by_projection[projection]
        if tuple(item.component.rsplit(".", 1)[1] for item in components) != LEAVES:
            raise Hy3NativeConversionError("expert projection leaf order drifted")
        source_name = components[0].source_tensor
        if any(component.source_tensor != source_name for component in components):
            raise Hy3NativeConversionError("expert projection has multiple sources")
        info = source.info(source_name)
        payloads = quantizer.quantize(
            TensorSpec(source_name, info.dtype, info.shape),
            source.read(source_name),
            bits=4,
            group_size=64,
        )
        for component, payload in zip(components, payloads, strict=True):
            payload.require(component.target)
            chunks.append(payload.data)
    result = b"".join(chunks)
    if len(result) != record.logical_bytes:
        raise Hy3NativeConversionError(
            f"record {record.record_id} has {len(result)} bytes, "
            f"expected {record.logical_bytes}"
        )
    return result


def _write_experts(
    plan: NativeConversionPlan,
    work_root: Path,
    journal: CheckpointJournal,
    quantizer: Quantizer,
    *,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> None:
    source = SafeTensorInventory(plan.source_root)
    path = work_root / "experts.bin"
    fd = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Hy3NativeConversionError("experts.bin is not a regular file")
        for record in plan.records:
            accepted = journal.get("expert_record", record.record_id)
            if (
                accepted is not None
                and os.fstat(fd).st_size >= record.offset + record.logical_bytes
            ):
                current = _pread_exact(fd, record.offset, record.logical_bytes)
                if _sha256_bytes(current) == accepted.get("sha256"):
                    continue
            payload = _record_payload(record, source, quantizer)
            digest = _sha256_bytes(payload)
            _pwrite_all(fd, payload, record.offset)
            os.fsync(fd)
            event = {
                "event": "expert_record",
                "id": record.record_id,
                "layer": record.layer,
                "expert": record.expert,
                "offset": record.offset,
                "length": record.logical_bytes,
                "sha256": digest,
            }
            journal.accept(event)
            if on_event is not None:
                on_event(event)
        os.ftruncate(fd, plan.experts_file_size)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(work_root)


def _patched_config(plan: NativeConversionPlan) -> dict[str, Any]:
    config = _load_json(plan.source_root / "config.json")
    quantization: dict[str, Any] = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    for layer in plan.geometry.routed_layers:
        quantization[f"model.layers.{layer}.mlp.router.gate"] = {
            "bits": 8,
            "group_size": 64,
        }
    config["quantization"] = quantization
    config["quantization_config"] = json.loads(json.dumps(quantization))
    return config


def _write_atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_TRUNC
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise Hy3NativeConversionError("short JSON write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _allowed_work_names(plan: NativeConversionPlan) -> set[str]:
    return {
        *(output.name for output in plan.resident_files),
        *ANCILLARY_FILES,
        "experts.bin",
        "layer80-bf16.safetensors",
        "conversion-checkpoint.jsonl",
        "model.safetensors.index.json",
        "config.json",
        "mtplx_runtime.json",
        "expert-manifest.json",
        "conversion-validation.json",
        "conversion-provenance.json",
    }


def _prepare_work_directory(output: Path) -> Path:
    """Open the deterministic work root without following stale symlinks."""

    work_root = output.parent / f".{output.name}.hy3-native-work"
    if os.path.lexists(work_root):
        mode = work_root.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise Hy3NativeConversionError(
                f"conversion work path is not a real directory: {work_root}"
            )
        entries = list(work_root.iterdir())
        checkpoint = work_root / "conversion-checkpoint.jsonl"
        if entries and not checkpoint.is_file():
            raise Hy3NativeConversionError(
                "non-empty conversion work directory has no checkpoint"
            )
        if checkpoint.is_symlink():
            raise Hy3NativeConversionError("conversion checkpoint cannot be a symlink")
        if checkpoint.is_file():
            # These are converter-owned names that can survive a crash before
            # their final atomic replace. No model/source payload is removed.
            transient = {
                work_root / ".layer80-bf16.safetensors.copying",
                *(work_root / f".{name}.copying" for name in ANCILLARY_FILES),
                *(
                    work_root / f".{name}.tmp"
                    for name in (
                        "model.safetensors.index.json",
                        "config.json",
                        "mtplx_runtime.json",
                        "conversion-validation.json",
                        "conversion-provenance.json",
                    )
                ),
            }
            for path in transient:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            for path in work_root.glob(".expert-manifest.json.*.tmp"):
                if path.is_file() and not path.is_symlink():
                    path.unlink()
    else:
        os.mkdir(work_root, 0o755)
        _fsync_directory(output.parent)
    return work_root


def _resolve_output_root(plan: NativeConversionPlan, output_root: Path | str) -> Path:
    """Resolve an output sibling that cannot overwrite either input tree."""

    expanded = Path(output_root).expanduser()
    lexical = Path(os.path.abspath(expanded))
    for label, source_root in (
        ("official source", plan.source_root),
        ("layout oracle", plan.oracle_root),
    ):
        if lexical == source_root or source_root in lexical.parents:
            raise Hy3NativeConversionError(
                f"output path cannot be inside the {label}: {lexical}"
            )
    lexical.parent.mkdir(parents=True, exist_ok=True)
    output = lexical.parent.resolve(strict=True) / lexical.name
    for label, source_root in (
        ("official source", plan.source_root),
        ("layout oracle", plan.oracle_root),
    ):
        if output == source_root or source_root in output.parents:
            raise Hy3NativeConversionError(
                f"output path cannot be inside the {label}: {output}"
            )
    return output


def _guard_work_inventory(
    plan: NativeConversionPlan,
    work_root: Path,
    *,
    final_ancillary: Iterable[str] | None = None,
) -> None:
    """Reject stale files that could shadow the compact native payload."""

    allowed = _allowed_work_names(plan)
    members = list(work_root.iterdir())
    if any(path.is_symlink() for path in members):
        raise Hy3NativeConversionError("conversion work directory contains a symlink")
    actual = {path.name for path in members}
    unexpected = actual - allowed
    if unexpected:
        raise Hy3NativeConversionError(
            f"conversion work directory contains unexpected files: {sorted(unexpected)}"
        )
    if any(path.is_dir() for path in members):
        raise Hy3NativeConversionError(
            "conversion work directory must contain only files"
        )
    if final_ancillary is None:
        return
    expected = {
        *(output.name for output in plan.resident_files),
        *final_ancillary,
        "experts.bin",
        "layer80-bf16.safetensors",
        "conversion-checkpoint.jsonl",
        "model.safetensors.index.json",
        "config.json",
        "mtplx_runtime.json",
        "expert-manifest.json",
        "conversion-validation.json",
        "conversion-provenance.json",
    }
    if actual != expected:
        raise Hy3NativeConversionError(
            "publish inventory does not exactly match the compact artifact; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _write_index_and_config(plan: NativeConversionPlan, work_root: Path) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    for output in plan.resident_files:
        for tensor in output.tensors:
            if tensor.name in weight_map:
                raise Hy3NativeConversionError(
                    f"duplicate resident index key {tensor.name}"
                )
            weight_map[tensor.name] = output.name
            total_size += tensor.nbytes
    if any(_is_oracle_expert(name) for name in weight_map):
        raise Hy3NativeConversionError("resident-only index contains routed experts")
    layer80_file = "layer80-residents-q.safetensors"
    if any(
        filename != layer80_file
        for name, filename in weight_map.items()
        if name.startswith(f"model.layers.{plan.geometry.mtp_layer_index}.")
    ):
        raise Hy3NativeConversionError(
            "layer-80 residents are split across output files"
        )
    _write_atomic_json(
        work_root / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    _write_atomic_json(work_root / "config.json", _patched_config(plan))


def _copy_ancillary(
    plan: NativeConversionPlan, work_root: Path
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for name in ANCILLARY_FILES:
        source = plan.source_root / name
        origin = SOURCE_REPO
        revision = plan.source_revision
        if not source.is_file():
            source = plan.oracle_root / name
            origin = ORACLE_REPO
            revision = plan.oracle_revision
        if not source.is_file():
            continue
        destination = work_root / name
        temporary = destination.with_name(f".{name}.copying")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        copied.append(
            {
                "file": name,
                "source_repo": origin,
                "source_revision": revision,
                "size": destination.stat().st_size,
                "sha256": _hash_file(destination),
            }
        )
    _fsync_directory(work_root)
    return copied


def _write_runtime_contract(
    plan: NativeConversionPlan,
    work_root: Path,
    final_output: Path,
    *,
    bf16_head_source: Path,
) -> dict[str, Any]:
    """Emit Forge-shaped provenance without verification or speed claims."""

    from mtplx.version import __version__

    recipe = {
        "body_bits": 4,
        "body_group_size": 64,
        "body_mode": "affine",
        "router_bits": 8,
        "router_group_size": 64,
        "mtp_policy": "keep_bf16",
    }
    runtime = {
        "mtplx_version": __version__,
        "arch_id": "hy3-mtp",
        "mtp_depth_max": 1,
        "mtp_sidecar": "bf16",
        "mtp_sidecar_file": "layer80-bf16.safetensors",
        "base_trunk": SOURCE_REPO,
        "artifact_role": "hy3-q4-native-compact",
        "forge_provenance": {
            "source_repo": SOURCE_REPO,
            "source_sha": plan.source_revision,
            "source_format": "bf16_native",
            "forge_recipe": recipe,
            "mtp_contract": {
                "mtp_layer_index": plan.geometry.mtp_layer_index,
                "mtp_precision_default": "bf16",
                "q4_ab_available": True,
            },
            "forge_inputs": {
                "source_path": str(plan.source_root),
                "layout_oracle_path": str(plan.oracle_root),
                "bf16_head_source_path": str(bf16_head_source),
                "output_path": str(final_output),
                "checkpoint_path": str(final_output / "conversion-checkpoint.jsonl"),
            },
            "forged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mtplx_version": __version__,
            "tool": f"{__name__} v{CONVERTER_VERSION}",
            "forged_locally": True,
            "published_to_hf": None,
        },
    }
    # Validation and benchmark evidence are deliberately absent.  Those gates
    # belong to the handoff session, after conversion is complete.
    _write_atomic_json(work_root / "mtplx_runtime.json", runtime)
    return runtime


def _inspect_output_safetensors(
    work_root: Path,
    outputs: Sequence[SafeTensorOutputPlan],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shards: list[dict[str, Any]] = []
    residents: list[dict[str, Any]] = []
    for output in outputs:
        inventory = SafeTensorFileInventory(work_root / output.name)
        if set(inventory.locations) != {tensor.name for tensor in output.tensors}:
            raise Hy3NativeConversionError(
                f"resident output header mismatch: {output.name}"
            )
        shards.append(
            {
                "name": output.name,
                "size": inventory.path.stat().st_size,
                "header_bytes": inventory.data_start,
                "header_sha256": inventory.header_sha256,
                "sha256": _hash_file(inventory.path),
                "kind": "safetensors",
            }
        )
        for name, info in sorted(inventory.locations.items()):
            residents.append(
                {
                    "tensor": name,
                    "shard": output.name,
                    "offset": info.offset,
                    "length": info.length,
                    "dtype": info.dtype,
                    "shape": list(info.shape),
                }
            )
    return shards, residents


def _construct_expert_manifest(
    plan: NativeConversionPlan,
    work_root: Path,
    journal: CheckpointJournal,
    bf16_head: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    """Construct the sidecar-authoritative manifest without Q4 expert shards."""

    from mtplx.expert_manifest import (
        EMPTY_SHA256,
        AuxiliaryFileInfo,
        ExpertManifest,
        ExpertRecord,
        ResidentTensor,
        ShardInfo,
        SidecarInfo,
        TensorSegment,
        save_expert_manifest,
    )

    shard_rows, resident_rows = _inspect_output_safetensors(
        work_root,
        plan.resident_files,
    )
    experts_path = work_root / "experts.bin"
    experts_sha = _hash_file(experts_path)
    sidecar_shard = ShardInfo(
        name="experts.bin",
        size=experts_path.stat().st_size,
        header_bytes=0,
        header_sha256=EMPTY_SHA256,
        sha256=experts_sha,
        kind="sidecar",
    )
    shards = tuple(
        ShardInfo(
            name=row["name"],
            size=row["size"],
            header_bytes=row["header_bytes"],
            header_sha256=row["header_sha256"],
            sha256=row["sha256"],
            kind="safetensors",
        )
        for row in shard_rows
    ) + (sidecar_shard,)
    residents = tuple(
        ResidentTensor(
            tensor=row["tensor"],
            shard=row["shard"],
            offset=row["offset"],
            length=row["length"],
            dtype=row["dtype"],
            shape=tuple(row["shape"]),
        )
        for row in sorted(resident_rows, key=lambda item: item["tensor"])
    )
    records: list[Any] = []
    for record_plan in plan.records:
        accepted = journal.get("expert_record", record_plan.record_id)
        if accepted is None:
            raise Hy3NativeConversionError(
                f"missing accepted expert record {record_plan.record_id}"
            )
        segments = tuple(
            TensorSegment(
                component=component.component,
                tensor=component.target.name,
                shard="experts.bin",
                offset=record_plan.offset + component.relative_offset,
                length=component.target.nbytes,
                dtype=component.target.dtype,
                shape=component.target.shape,
            )
            for component in record_plan.components
        )
        records.append(
            ExpertRecord(
                layer=record_plan.layer,
                expert=record_plan.expert,
                logical_bytes=record_plan.logical_bytes,
                segments=segments,
                sha256=accepted["sha256"],
                sidecar_offset=record_plan.offset,
                sidecar_length=record_plan.logical_bytes,
            )
        )
    resident_bytes = sum(tensor.length for tensor in residents)
    routed_bytes = sum(record.logical_bytes for record in records)
    bf16_auxiliary = AuxiliaryFileInfo(
        file="layer80-bf16.safetensors",
        role="mtp-bf16",
        size=bf16_head["size"],
        sha256=bf16_head["sha256"],
        header_bytes=bf16_head["header_bytes"],
        header_sha256=bf16_head["header_sha256"],
        source_repo=SOURCE_REPO,
        source_revision=plan.source_revision,
    )
    manifest = ExpertManifest(
        model_key=MODEL_KEY,
        source_repo=SOURCE_REPO,
        source_revision=plan.source_revision,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        artifact_tensor_bytes=resident_bytes + routed_bytes,
        resident_tensor_bytes=resident_bytes,
        routed_expert_bytes=routed_bytes,
        shards=shards,
        resident_tensors=residents,
        records=tuple(records),
        sidecar=SidecarInfo(
            file="experts.bin",
            alignment=plan.alignment,
            size=experts_path.stat().st_size,
            sha256=experts_sha,
        ),
        auxiliary_files=(bf16_auxiliary,),
    ).with_digest()
    manifest.validate_structure()
    save_expert_manifest(manifest, work_root / "expert-manifest.json")
    shard_rows.append(
        {
            "name": "experts.bin",
            "size": experts_path.stat().st_size,
            "header_bytes": 0,
            "header_sha256": EMPTY_SHA256,
            "sha256": experts_sha,
            "kind": "sidecar",
        }
    )
    return manifest, shard_rows


def _write_provenance(
    plan: NativeConversionPlan,
    work_root: Path,
    *,
    source_shards: dict[str, dict[str, Any]],
    bf16_head: dict[str, Any],
    quantizer: Quantizer,
    ancillary: list[dict[str, Any]],
    manifest: Any,
    output_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    source_rows = [
        {
            "file": shard,
            "size": source_shards[shard]["stat"]["size"],
            "sha256": source_shards[shard]["sha256"],
            "expected_lfs_sha256": source_shards[shard].get("expected_lfs_sha256"),
        }
        for shard in plan.source_shards
    ]
    if len(source_rows) != len(plan.source_shards):
        raise Hy3NativeConversionError("source provenance is incomplete")
    provenance = {
        "format": PROVENANCE_FORMAT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "converter": {
            "module": __name__,
            "version": CONVERTER_VERSION,
            "module_sha256": _hash_file(Path(__file__)),
            "plan_fingerprint": plan.fingerprint,
        },
        "source": {
            "repo": SOURCE_REPO,
            "revision": plan.source_revision,
            "index_sha256": plan.source_index_sha256,
            "config_sha256": plan.source_config_sha256,
            "shards": source_rows,
            "automatic_source_deletion": False,
        },
        "layout_oracle": {
            "repo": ORACLE_REPO,
            "revision": plan.oracle_revision,
            "index_sha256": plan.oracle_index_sha256,
            "config_sha256": plan.oracle_config_sha256,
        },
        "quantization": {
            "mode": "affine",
            "trunk_and_mtp_experts": {"bits": 4, "group_size": 64},
            "residents": {"bits": 4, "group_size": 64},
            "routers": {"bits": 8, "group_size": 64},
            "quantizer": quantizer.provenance,
        },
        "layout": {
            "alignment": plan.alignment,
            "record_bytes": plan.expert_record_bytes,
            "record_count": len(plan.records),
            "routed_layers": list(plan.geometry.routed_layers),
            "expert_count": plan.geometry.expert_count,
            "component_order": list(COMPONENTS),
            "resident_only_index": True,
            "layer80_q4_resident_file": "layer80-residents-q.safetensors",
            "layer80_q4_expert_intermediate_created": False,
        },
        "bf16_mtp_head": bf16_head,
        "ancillary_files": ancillary,
        "outputs": {
            "manifest": {
                "file": "expert-manifest.json",
                "sha256": _hash_file(work_root / "expert-manifest.json"),
                "manifest_sha256": manifest.manifest_sha256,
            },
            "shards": output_shards,
            "model_index": {
                "file": "model.safetensors.index.json",
                "sha256": _hash_file(work_root / "model.safetensors.index.json"),
            },
            "config": {
                "file": "config.json",
                "sha256": _hash_file(work_root / "config.json"),
            },
            "runtime": {
                "file": "mtplx_runtime.json",
                "sha256": _hash_file(work_root / "mtplx_runtime.json"),
            },
            "serialization_validation": {
                "file": "conversion-validation.json",
                "sha256": _hash_file(work_root / "conversion-validation.json"),
            },
        },
    }
    _write_atomic_json(work_root / "conversion-provenance.json", provenance)
    return provenance


def _verify_final_serialization(
    plan: NativeConversionPlan,
    work_root: Path,
    manifest: Any,
) -> dict[str, Any]:
    """Run the full manifest/spec/auxiliary audit before publication."""

    from mtplx.expert_manifest import (
        validate_expert_manifest_spec,
        verify_expert_manifest,
    )
    from mtplx.expert_streaming_models import get_model_spec

    spec_verified = False
    if plan.geometry == Hy3Geometry(80, 1, 1, 192, 4096, 1536):
        validate_expert_manifest_spec(
            manifest,
            get_model_spec(MODEL_KEY),
            require_pinned_tensor_bytes=True,
        )
        spec_verified = True
    report = verify_expert_manifest(
        manifest,
        work_root,
        verify_records=True,
        verify_shard_hashes=True,
        verify_sidecar_hash=True,
    )
    validation = {
        "format": "mtplx-hy3-native-serialization-validation-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_fingerprint": plan.fingerprint,
        "manifest": report,
        "model_spec_verified": spec_verified,
        "resident_only_index_verified": True,
        "component_order": list(COMPONENTS),
        "performance_claims": False,
        "deferred_validation_gates": [
            "fixed-vector resident parity",
            "router ID and score match",
            "perplexity-class sanity",
        ],
    }
    _write_atomic_json(work_root / "conversion-validation.json", validation)
    return validation


def prepare_conversion(
    plan: NativeConversionPlan,
    output_root: Path | str,
    *,
    bf16_head_source: Path | str | None = None,
    copy_bf16_head: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Run passive hashing/head binding only; never import or invoke MLX."""

    output = _resolve_output_root(plan, output_root)
    if os.path.lexists(output):
        raise Hy3NativeConversionError(f"output already exists: {output}")
    work_root = _prepare_work_directory(output)
    _guard_work_inventory(plan, work_root)
    journal = CheckpointJournal(
        work_root / "conversion-checkpoint.jsonl",
        plan.fingerprint,
    )
    _hash_source_shards(plan, journal, on_event=on_event)
    head_source = (
        (
            Path(bf16_head_source)
            if bf16_head_source is not None
            else plan.source_root / "layer80-bf16.safetensors"
        )
        .expanduser()
        .resolve()
    )
    if head_source == work_root or work_root in head_source.parents:
        raise Hy3NativeConversionError(
            "BF16 head source cannot be inside the conversion work directory"
        )
    _verify_and_bind_bf16_head(
        plan,
        work_root,
        journal,
        source_head=head_source,
        copy_head=copy_bf16_head,
        on_event=on_event,
    )
    _write_index_and_config(plan, work_root)
    _copy_ancillary(plan, work_root)
    _write_runtime_contract(
        plan,
        work_root,
        output,
        bf16_head_source=head_source,
    )
    return work_root


def execute_conversion(
    plan: NativeConversionPlan,
    output_root: Path | str,
    *,
    quantizer_factory: Callable[[], Quantizer] = MlxAffineQuantizer,
    bf16_head_source: Path | str | None = None,
    copy_bf16_head: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Resume or execute a plan, then atomically publish ``output_root``.

    ``on_event`` runs only after an event's output and checkpoint have both
    been fsynced.  It is useful for progress reporting and crash-injection
    tests; exceptions intentionally leave the hidden work directory resumable.
    """

    output = Path(os.path.abspath(Path(output_root).expanduser()))
    work_root = prepare_conversion(
        plan,
        output,
        bf16_head_source=bf16_head_source,
        copy_bf16_head=copy_bf16_head,
        on_event=on_event,
    )
    journal = CheckpointJournal(
        work_root / "conversion-checkpoint.jsonl",
        plan.fingerprint,
    )
    source_shards = {
        shard: journal.get("source_shard", shard) for shard in plan.source_shards
    }
    if any(event is None for event in source_shards.values()):
        raise Hy3NativeConversionError("passive source preparation is incomplete")
    head_source = (
        (
            Path(bf16_head_source)
            if bf16_head_source is not None
            else plan.source_root / "layer80-bf16.safetensors"
        )
        .expanduser()
        .resolve()
    )
    bf16_head = journal.get("bf16_head", "layer80-bf16.safetensors")
    if bf16_head is None:
        raise Hy3NativeConversionError("passive BF16 head preparation is incomplete")
    ancillary = _copy_ancillary(plan, work_root)

    print(
        "ANNOUNCE: starting Hy3 MLX quantization compute; "
        f"run-lock pattern is {RUN_LOCK_PATTERN!r}",
        flush=True,
    )
    quantizer = quantizer_factory()
    _write_residents(
        plan,
        work_root,
        journal,
        quantizer,
        on_event=on_event,
    )
    _write_experts(
        plan,
        work_root,
        journal,
        quantizer,
        on_event=on_event,
    )
    _assert_source_shards_unchanged(plan, source_shards)
    manifest, output_shards = _construct_expert_manifest(
        plan,
        work_root,
        journal,
        bf16_head,
    )
    _write_runtime_contract(
        plan,
        work_root,
        output,
        bf16_head_source=head_source,
    )
    _verify_final_serialization(plan, work_root, manifest)
    _write_provenance(
        plan,
        work_root,
        source_shards=source_shards,
        bf16_head=bf16_head,
        quantizer=quantizer,
        ancillary=ancillary,
        manifest=manifest,
        output_shards=output_shards,
    )
    _guard_work_inventory(
        plan,
        work_root,
        final_ancillary=(row["file"] for row in ancillary),
    )
    ready = {
        "event": "ready_to_publish",
        "id": "artifact",
        "manifest_sha256": manifest.manifest_sha256,
        "provenance_sha256": _hash_file(work_root / "conversion-provenance.json"),
    }
    journal.accept(ready)
    if on_event is not None:
        on_event(ready)
    _fsync_directory(work_root)
    if os.path.lexists(output):
        raise Hy3NativeConversionError(f"output path appeared during build: {output}")
    os.replace(work_root, output)
    _fsync_directory(output.parent)
    return output


def plan_summary(plan: NativeConversionPlan) -> dict[str, Any]:
    return {
        "plan_fingerprint": plan.fingerprint,
        "source_repo": SOURCE_REPO,
        "source_revision": plan.source_revision,
        "layout_oracle_repo": ORACLE_REPO,
        "layout_oracle_revision": plan.oracle_revision,
        "source_shards": len(plan.source_shards),
        "resident_files": len(plan.resident_files),
        "resident_units": len(plan.resident_units),
        "expert_records": len(plan.records),
        "expert_record_bytes": plan.expert_record_bytes,
        "experts_file_bytes": plan.experts_file_size,
        "alignment": plan.alignment,
        "mtp_layer": plan.geometry.mtp_layer_index,
        "mtp_included": True,
        "bf16_head_tensors": len(plan.bf16_head_tensors),
    }
