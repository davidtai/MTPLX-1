"""Isolated Hy3-Q2 expert-layout and coalescing experiments for issue #68."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

import numpy as np


HY3_Q2_ROWS = tuple(range(1, 9))
HY3_Q2_PRIMARY_ROW = 4
HY3_Q2_BITS = 2
HY3_Q2_GROUP_SIZE = 64
HY3_Q2_HIDDEN_SIZE = 4096
HY3_Q2_INTERMEDIATE_SIZE = 1536
HY3_Q2_TOP_K = 8
HY3_Q2_SCAN_OUTPUTS = 2 * HY3_Q2_INTERMEDIATE_SIZE + HY3_Q2_HIDDEN_SIZE
HY3_Q2_EXPERT_RECORD_BYTES = 5_898_240
HY3_Q2_PACKET_WORDS = 5

HY3_Q2_LAYOUTS = (
    "output_packet",
    "group_output_packet",
    "tiled_projection_field",
    "tiled_field_projection",
)
HY3_Q2_BLOCK_SIZES = (16, 32, 64, 128)
HY3_Q2_VECTOR_WIDTHS = (1, 2, 4)


class Hy3Q2LayoutError(ValueError):
    """Raised when packed-layout construction cannot preserve its contract."""


@dataclass(frozen=True, slots=True)
class Hy3Q2LayoutCandidate:
    """One byte-neutral packed layout and consumer vector width."""

    layout: str
    block_size: int
    vector_width: int

    def __post_init__(self) -> None:
        if self.layout not in HY3_Q2_LAYOUTS:
            raise ValueError(f"unknown Hy3-Q2 layout: {self.layout}")
        if self.vector_width not in HY3_Q2_VECTOR_WIDTHS:
            raise ValueError("vector width must be 1, 2, or 4")
        if self.layout.startswith("tiled_"):
            if self.block_size not in HY3_Q2_BLOCK_SIZES:
                raise ValueError("tiled layouts require block size 16, 32, 64, or 128")
        elif self.block_size != 0:
            raise ValueError("untiled packet layouts require block size zero")

    @property
    def name(self) -> str:
        block = f"b{self.block_size}" if self.block_size else "flat"
        return f"{self.layout}_{block}_v{self.vector_width}"

    @property
    def resident_byte_overhead(self) -> int:
        return 0

    @property
    def promotion_eligible(self) -> bool:
        return self.resident_byte_overhead == 0


@dataclass(frozen=True, slots=True)
class Hy3Q2SourceControl:
    """The authoritative split-component layout at one consumer width."""

    vector_width: int

    def __post_init__(self) -> None:
        if self.vector_width not in HY3_Q2_VECTOR_WIDTHS:
            raise ValueError("vector width must be 1, 2, or 4")

    @property
    def name(self) -> str:
        return f"source_components_v{self.vector_width}"


def hy3_q2_layout_candidates() -> tuple[Hy3Q2LayoutCandidate, ...]:
    """Return the compact search space without padded resident layouts."""

    untiled = tuple(
        Hy3Q2LayoutCandidate(layout, 0, vector_width)
        for layout, vector_width in product(HY3_Q2_LAYOUTS[:2], HY3_Q2_VECTOR_WIDTHS)
    )
    tiled = tuple(
        Hy3Q2LayoutCandidate(layout, block_size, vector_width)
        for layout, block_size, vector_width in product(
            HY3_Q2_LAYOUTS[2:], HY3_Q2_BLOCK_SIZES, HY3_Q2_VECTOR_WIDTHS
        )
    )
    return untiled + tiled


def hy3_q2_source_controls() -> tuple[Hy3Q2SourceControl, ...]:
    return tuple(Hy3Q2SourceControl(width) for width in HY3_Q2_VECTOR_WIDTHS)


@dataclass(frozen=True, slots=True)
class Hy3Q2SlotIdentity:
    """Fixed slot ownership carried through construction-time replacement."""

    slot: int
    expert: int
    generation: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.slot, self.expert, self.generation)
        ):
            raise TypeError("slot identity fields must be exact integers")
        if self.slot < 0 or self.expert < 0 or self.generation <= 0:
            raise ValueError("slot/expert must be nonnegative and generation positive")


@dataclass(frozen=True, slots=True)
class PackedHy3Q2Bank:
    """Packed buffers that replace, rather than retain, source components."""

    candidate: Hy3Q2LayoutCandidate
    identities: tuple[Hy3Q2SlotIdentity, ...]
    gate_up: np.ndarray
    down: np.ndarray
    source_component_bytes: int

    @property
    def packed_bytes(self) -> int:
        return int(self.gate_up.nbytes + self.down.nbytes)

    @property
    def steady_state_duplicate_bytes(self) -> int:
        return 0

    def reconstruct_components(self) -> dict[str, np.ndarray]:
        gate_up = _canonical_gate_up(self.gate_up, self.candidate)
        down = _canonical_down(self.down, self.candidate)
        result: dict[str, np.ndarray] = {}
        for projection, packets in (
            ("gate_proj", gate_up[:, 0]),
            ("up_proj", gate_up[:, 1]),
            ("down_proj", down),
        ):
            weight, scales, biases = _unpack_projection_packets(packets)
            result[f"{projection}.weight"] = weight
            result[f"{projection}.scales"] = scales
            result[f"{projection}.biases"] = biases
        return result

    def validate_replacement(self, expected: tuple[Hy3Q2SlotIdentity, ...]) -> bool:
        if len(expected) != len(self.identities):
            raise Hy3Q2LayoutError("slot capacity changed during replacement")
        for actual, wanted in zip(self.identities, expected, strict=True):
            for field in ("slot", "expert", "generation"):
                if getattr(actual, field) != getattr(wanted, field):
                    raise Hy3Q2LayoutError(
                        f"{field} changed during packed-slot replacement"
                    )
        return True

    def to_mlx(self) -> MlxPackedHy3Q2Bank:
        import mlx.core as mx

        return MlxPackedHy3Q2Bank(
            candidate=self.candidate,
            identities=self.identities,
            gate_up=mx.array(self.gate_up),
            down=mx.array(self.down),
        )


@dataclass(frozen=True, slots=True)
class MlxPackedHy3Q2Bank:
    candidate: Hy3Q2LayoutCandidate
    identities: tuple[Hy3Q2SlotIdentity, ...]
    gate_up: Any
    down: Any


def _validated_components(
    components: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    expected = hy3_q2_component_shapes()
    if set(components) != set(expected):
        raise ValueError("Hy3-Q2 component bank must contain exactly nine arrays")
    capacity: int | None = None
    validated: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(components[name])
        dtype = np.dtype(np.uint32 if name.endswith(".weight") else np.uint16)
        if value.ndim != 3 or tuple(value.shape[1:]) != shape or value.dtype != dtype:
            raise ValueError(f"wrong shape or dtype for {name}")
        if capacity is None:
            capacity = int(value.shape[0])
        elif int(value.shape[0]) != capacity:
            raise ValueError("all Hy3-Q2 components must have equal slot capacity")
        validated[name] = np.ascontiguousarray(value)
    if capacity is None or capacity <= 0:
        raise ValueError("Hy3-Q2 component bank capacity must be positive")
    return validated, capacity


def _projection_packets(
    components: Mapping[str, np.ndarray], projection: str
) -> np.ndarray:
    weight = components[f"{projection}.weight"]
    scales = components[f"{projection}.scales"]
    biases = components[f"{projection}.biases"]
    capacity, outputs, groups = scales.shape
    weight_groups = weight.reshape(capacity, outputs, groups, 4)
    metadata = scales.astype(np.uint32) | (biases.astype(np.uint32) << np.uint32(16))
    return np.concatenate((weight_groups, metadata[..., None]), axis=-1)


def _pack_gate_up(canonical: np.ndarray, candidate: Hy3Q2LayoutCandidate) -> np.ndarray:
    if candidate.layout == "output_packet":
        packed = canonical
    elif candidate.layout == "group_output_packet":
        packed = canonical.transpose(0, 1, 3, 2, 4)
    else:
        capacity, projections, outputs, groups, fields = canonical.shape
        block = candidate.block_size
        blocked = canonical.reshape(
            capacity, projections, outputs // block, block, groups, fields
        )
        if candidate.layout == "tiled_projection_field":
            packed = blocked.transpose(0, 2, 4, 1, 5, 3)
        else:
            packed = blocked.transpose(0, 2, 4, 5, 1, 3)
    return np.ascontiguousarray(packed)


def _pack_down(canonical: np.ndarray, candidate: Hy3Q2LayoutCandidate) -> np.ndarray:
    if candidate.layout == "output_packet":
        packed = canonical
    elif candidate.layout == "group_output_packet":
        packed = canonical.transpose(0, 2, 1, 3)
    else:
        capacity, outputs, groups, fields = canonical.shape
        block = candidate.block_size
        packed = canonical.reshape(
            capacity, outputs // block, block, groups, fields
        ).transpose(0, 1, 3, 4, 2)
    return np.ascontiguousarray(packed)


def _canonical_gate_up(
    packed: np.ndarray, candidate: Hy3Q2LayoutCandidate
) -> np.ndarray:
    if candidate.layout == "output_packet":
        canonical = packed
    elif candidate.layout == "group_output_packet":
        canonical = packed.transpose(0, 1, 3, 2, 4)
    elif candidate.layout == "tiled_projection_field":
        capacity, blocks, groups, projections, fields, block = packed.shape
        canonical = packed.transpose(0, 3, 1, 5, 2, 4).reshape(
            capacity, projections, blocks * block, groups, fields
        )
    else:
        capacity, blocks, groups, fields, projections, block = packed.shape
        canonical = packed.transpose(0, 4, 1, 5, 2, 3).reshape(
            capacity, projections, blocks * block, groups, fields
        )
    return np.ascontiguousarray(canonical)


def _canonical_down(packed: np.ndarray, candidate: Hy3Q2LayoutCandidate) -> np.ndarray:
    if candidate.layout == "output_packet":
        canonical = packed
    elif candidate.layout == "group_output_packet":
        canonical = packed.transpose(0, 2, 1, 3)
    else:
        capacity, blocks, groups, fields, block = packed.shape
        canonical = packed.transpose(0, 1, 4, 2, 3).reshape(
            capacity, blocks * block, groups, fields
        )
    return np.ascontiguousarray(canonical)


def _unpack_projection_packets(
    packets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capacity, outputs, groups, _fields = packets.shape
    weight = np.ascontiguousarray(
        packets[..., :4].reshape(capacity, outputs, groups * 4)
    )
    metadata = packets[..., 4]
    scales = np.ascontiguousarray((metadata & np.uint32(0xFFFF)).astype(np.uint16))
    biases = np.ascontiguousarray((metadata >> np.uint32(16)).astype(np.uint16))
    return weight, scales, biases


def pack_hy3_q2_bank(
    components: Mapping[str, np.ndarray],
    candidate: Hy3Q2LayoutCandidate,
    *,
    identities: tuple[Hy3Q2SlotIdentity, ...],
) -> PackedHy3Q2Bank:
    """Pack all Q2 components without retaining a source-array reference."""

    validated, capacity = _validated_components(components)
    if (
        len(identities) != capacity
        or len({item.slot for item in identities}) != capacity
    ):
        raise ValueError("slot identities must uniquely cover packed capacity")
    gate_up_canonical = np.stack(
        (
            _projection_packets(validated, "gate_proj"),
            _projection_packets(validated, "up_proj"),
        ),
        axis=1,
    )
    down_canonical = _projection_packets(validated, "down_proj")
    source_bytes = sum(int(value.nbytes) for value in validated.values())
    packed = PackedHy3Q2Bank(
        candidate=candidate,
        identities=identities,
        gate_up=_pack_gate_up(gate_up_canonical, candidate),
        down=_pack_down(down_canonical, candidate),
        source_component_bytes=source_bytes,
    )
    if packed.packed_bytes != source_bytes:
        raise ValueError("compact Hy3-Q2 packing changed resident byte count")
    return packed


def _cache_line_count(addresses: list[int], *, cache_line_bytes: int) -> int:
    return len({address // cache_line_bytes for address in addresses})


def _source_projection_cache_lines(
    *, groups: int, logical_outputs: int, cache_line_bytes: int
) -> int:
    total = 0
    weight_row_bytes = groups * 4 * 4
    for word in range(4):
        total += _cache_line_count(
            [output * weight_row_bytes + word * 4 for output in range(logical_outputs)],
            cache_line_bytes=cache_line_bytes,
        )
    parameter_row_bytes = groups * 2
    for _parameter in range(2):
        total += _cache_line_count(
            [output * parameter_row_bytes for output in range(logical_outputs)],
            cache_line_bytes=cache_line_bytes,
        )
    return total


def _candidate_projection_address(
    candidate: Hy3Q2LayoutCandidate,
    *,
    projection: int,
    projections: int,
    output: int,
    outputs: int,
    groups: int,
    group: int,
    field: int,
) -> int:
    if candidate.layout == "output_packet":
        index = (((projection * outputs + output) * groups + group) * 5) + field
    elif candidate.layout == "group_output_packet":
        index = (((projection * groups + group) * outputs + output) * 5) + field
    else:
        block_size = candidate.block_size
        block = output // block_size
        lane = output % block_size
        if candidate.layout == "tiled_projection_field":
            index = (
                (((block * groups + group) * projections + projection) * 5 + field)
                * block_size
            ) + lane
        else:
            index = (
                (((block * groups + group) * 5 + field) * projections + projection)
                * block_size
            ) + lane
    return index * 4


def _candidate_projection_cache_lines(
    candidate: Hy3Q2LayoutCandidate,
    *,
    projection: int,
    projections: int,
    outputs: int,
    groups: int,
    logical_outputs: int,
    cache_line_bytes: int,
) -> int:
    return sum(
        _cache_line_count(
            [
                _candidate_projection_address(
                    candidate,
                    projection=projection,
                    projections=projections,
                    output=output,
                    outputs=outputs,
                    groups=groups,
                    group=0,
                    field=field,
                )
                for output in range(logical_outputs)
            ],
            cache_line_bytes=cache_line_bytes,
        )
        for field in range(HY3_Q2_PACKET_WORDS)
    )


def hy3_q2_coalescing_evidence(
    candidate: Hy3Q2LayoutCandidate,
    *,
    simd_lanes: int = 32,
    cache_line_bytes: int = 128,
) -> dict[str, object]:
    """Model cache-line demand for one affine group across adjacent outputs."""

    logical_outputs = simd_lanes * candidate.vector_width
    source_gate_up = 2 * _source_projection_cache_lines(
        groups=64,
        logical_outputs=logical_outputs,
        cache_line_bytes=cache_line_bytes,
    )
    source_down = _source_projection_cache_lines(
        groups=24,
        logical_outputs=logical_outputs,
        cache_line_bytes=cache_line_bytes,
    )
    candidate_gate_up = sum(
        _candidate_projection_cache_lines(
            candidate,
            projection=projection,
            projections=2,
            outputs=HY3_Q2_INTERMEDIATE_SIZE,
            groups=64,
            logical_outputs=logical_outputs,
            cache_line_bytes=cache_line_bytes,
        )
        for projection in range(2)
    )
    candidate_down = _candidate_projection_cache_lines(
        candidate,
        projection=0,
        projections=1,
        outputs=HY3_Q2_HIDDEN_SIZE,
        groups=24,
        logical_outputs=logical_outputs,
        cache_line_bytes=cache_line_bytes,
    )
    return {
        "simd_lanes": simd_lanes,
        "logical_outputs": logical_outputs,
        "cache_line_bytes": cache_line_bytes,
        "logical_bytes_per_output_group": 3 * HY3_Q2_PACKET_WORDS * 4,
        "source": {
            "cache_lines": source_gate_up + source_down,
            "gate_up_cache_lines": source_gate_up,
            "down_cache_lines": source_down,
            "metadata_loads_per_output_group": 6,
        },
        "candidate": {
            "cache_lines": candidate_gate_up + candidate_down,
            "gate_up_cache_lines": candidate_gate_up,
            "down_cache_lines": candidate_down,
            "metadata_loads_per_output_group": 3,
        },
    }


def hy3_q2_scan_output_shape(rows: int) -> tuple[int, int]:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows not in HY3_Q2_ROWS:
        raise ValueError("Hy3-Q2 layout scan requires R1 through R8")
    return rows * HY3_Q2_TOP_K, HY3_Q2_SCAN_OUTPUTS


def render_hy3_q2_source_scan_source(control: Hy3Q2SourceControl) -> str:
    """Render the split-component control scan in expert-compute access order."""

    return f"""
        constexpr uint VECTOR_WIDTH = {control.vector_width};
        constexpr uint E1_OUTPUTS = {HY3_Q2_INTERMEDIATE_SIZE};
        constexpr uint E1_GROUPS = 64;
        constexpr uint E1_WEIGHT_WORDS = 256;
        constexpr uint DOWN_OUTPUTS = {HY3_Q2_HIDDEN_SIZE};
        constexpr uint DOWN_GROUPS = 24;
        constexpr uint DOWN_WEIGHT_WORDS = 96;
        constexpr uint SCAN_OUTPUTS = {HY3_Q2_SCAN_OUTPUTS};
        constexpr uint GATE_TASKS = E1_OUTPUTS / VECTOR_WIDTH;
        constexpr uint UP_TASKS = E1_OUTPUTS / VECTOR_WIDTH;
        constexpr uint DOWN_TASKS = DOWN_OUTPUTS / VECTOR_WIDTH;
        constexpr uint TASKS_PER_ASSIGNMENT = GATE_TASKS + UP_TASKS + DOWN_TASKS;

        uint gid = thread_position_in_grid.x;
        uint assignment = gid / TASKS_PER_ASSIGNMENT;
        uint local_task = gid - assignment * TASKS_PER_ASSIGNMENT;
        uint projection = local_task < GATE_TASKS
            ? 0u : (local_task < GATE_TASKS + UP_TASKS ? 1u : 2u);
        uint projection_task = projection == 0u
            ? local_task
            : (projection == 1u ? local_task - GATE_TASKS
                                : local_task - GATE_TASKS - UP_TASKS);
        uint output_base = projection_task * VECTOR_WIDTH;
        int slot = expert_slots[assignment];

        for (uint vector_lane = 0; vector_lane < VECTOR_WIDTH; ++vector_lane) {{
            uint output = output_base + vector_lane;
            uint checksum = 2166136261u;
            if (projection == 0u) {{
                for (uint group = 0; group < E1_GROUPS; ++group) {{
                    uint weight_offset =
                        (uint(slot) * E1_OUTPUTS + output) * E1_WEIGHT_WORDS
                        + group * 4u;
                    for (uint field = 0; field < 4u; ++field) {{
                        checksum = (checksum * 16777619u)
                            ^ gate_weight[weight_offset + field];
                    }}
                    uint parameter_offset =
                        (uint(slot) * E1_OUTPUTS + output) * E1_GROUPS + group;
                    uint metadata = uint(gate_scales[parameter_offset])
                        | (uint(gate_biases[parameter_offset]) << 16u);
                    checksum = (checksum * 16777619u) ^ metadata;
                }}
                checksums[assignment * SCAN_OUTPUTS + output] = checksum;
            }} else if (projection == 1u) {{
                for (uint group = 0; group < E1_GROUPS; ++group) {{
                    uint weight_offset =
                        (uint(slot) * E1_OUTPUTS + output) * E1_WEIGHT_WORDS
                        + group * 4u;
                    for (uint field = 0; field < 4u; ++field) {{
                        checksum = (checksum * 16777619u)
                            ^ up_weight[weight_offset + field];
                    }}
                    uint parameter_offset =
                        (uint(slot) * E1_OUTPUTS + output) * E1_GROUPS + group;
                    uint metadata = uint(up_scales[parameter_offset])
                        | (uint(up_biases[parameter_offset]) << 16u);
                    checksum = (checksum * 16777619u) ^ metadata;
                }}
                checksums[assignment * SCAN_OUTPUTS + E1_OUTPUTS + output]
                    = checksum;
            }} else {{
                for (uint group = 0; group < DOWN_GROUPS; ++group) {{
                    uint weight_offset =
                        (uint(slot) * DOWN_OUTPUTS + output) * DOWN_WEIGHT_WORDS
                        + group * 4u;
                    for (uint field = 0; field < 4u; ++field) {{
                        checksum = (checksum * 16777619u)
                            ^ down_weight[weight_offset + field];
                    }}
                    uint parameter_offset =
                        (uint(slot) * DOWN_OUTPUTS + output) * DOWN_GROUPS + group;
                    uint metadata = uint(down_scales[parameter_offset])
                        | (uint(down_biases[parameter_offset]) << 16u);
                    checksum = (checksum * 16777619u) ^ metadata;
                }}
                checksums[assignment * SCAN_OUTPUTS + 2u * E1_OUTPUTS + output]
                    = checksum;
            }}
        }}
    """


def _packed_gate_offset_expression(candidate: Hy3Q2LayoutCandidate) -> str:
    if candidate.layout == "output_packet":
        return (
            "((((uint(slot) * 2u + projection) * E1_OUTPUTS + output) "
            "* E1_GROUPS + group) * 5u + field)"
        )
    if candidate.layout == "group_output_packet":
        return (
            "((((uint(slot) * 2u + projection) * E1_GROUPS + group) "
            "* E1_OUTPUTS + output) * 5u + field)"
        )
    if candidate.layout == "tiled_projection_field":
        return (
            "((((((uint(slot) * E1_BLOCKS + block) * E1_GROUPS + group) "
            "* 2u + projection) * 5u + field) * BLOCK_SIZE) + lane)"
        )
    return (
        "((((((uint(slot) * E1_BLOCKS + block) * E1_GROUPS + group) "
        "* 5u + field) * 2u + projection) * BLOCK_SIZE) + lane)"
    )


def _packed_down_offset_expression(candidate: Hy3Q2LayoutCandidate) -> str:
    if candidate.layout == "output_packet":
        return (
            "(((uint(slot) * DOWN_OUTPUTS + output) * DOWN_GROUPS + group) "
            "* 5u + field)"
        )
    if candidate.layout == "group_output_packet":
        return (
            "(((uint(slot) * DOWN_GROUPS + group) * DOWN_OUTPUTS + output) "
            "* 5u + field)"
        )
    return (
        "(((((uint(slot) * DOWN_BLOCKS + block) * DOWN_GROUPS + group) "
        "* 5u + field) * BLOCK_SIZE) + lane)"
    )


def render_hy3_q2_packed_scan_source(candidate: Hy3Q2LayoutCandidate) -> str:
    """Render a packed scan with the same logical work as the source control."""

    block_size = candidate.block_size or 1
    gate_offset = _packed_gate_offset_expression(candidate)
    down_offset = _packed_down_offset_expression(candidate)
    return f"""
        constexpr uint VECTOR_WIDTH = {candidate.vector_width};
        constexpr uint BLOCK_SIZE = {block_size};
        constexpr uint E1_OUTPUTS = {HY3_Q2_INTERMEDIATE_SIZE};
        constexpr uint E1_GROUPS = 64;
        constexpr uint E1_BLOCKS = E1_OUTPUTS / BLOCK_SIZE;
        constexpr uint DOWN_OUTPUTS = {HY3_Q2_HIDDEN_SIZE};
        constexpr uint DOWN_GROUPS = 24;
        constexpr uint DOWN_BLOCKS = DOWN_OUTPUTS / BLOCK_SIZE;
        constexpr uint SCAN_OUTPUTS = {HY3_Q2_SCAN_OUTPUTS};
        constexpr uint GATE_TASKS = E1_OUTPUTS / VECTOR_WIDTH;
        constexpr uint UP_TASKS = E1_OUTPUTS / VECTOR_WIDTH;
        constexpr uint DOWN_TASKS = DOWN_OUTPUTS / VECTOR_WIDTH;
        constexpr uint TASKS_PER_ASSIGNMENT = GATE_TASKS + UP_TASKS + DOWN_TASKS;

        uint gid = thread_position_in_grid.x;
        uint assignment = gid / TASKS_PER_ASSIGNMENT;
        uint local_task = gid - assignment * TASKS_PER_ASSIGNMENT;
        uint projection = local_task < GATE_TASKS
            ? 0u : (local_task < GATE_TASKS + UP_TASKS ? 1u : 2u);
        uint projection_task = projection == 0u
            ? local_task
            : (projection == 1u ? local_task - GATE_TASKS
                                : local_task - GATE_TASKS - UP_TASKS);
        uint output_base = projection_task * VECTOR_WIDTH;
        int slot = expert_slots[assignment];

        for (uint vector_lane = 0; vector_lane < VECTOR_WIDTH; ++vector_lane) {{
            uint output = output_base + vector_lane;
            uint block = output / BLOCK_SIZE;
            uint lane = output - block * BLOCK_SIZE;
            uint checksum = 2166136261u;
            if (projection < 2u) {{
                for (uint group = 0; group < E1_GROUPS; ++group) {{
                    for (uint field = 0; field < 4u; ++field) {{
                        uint offset = {gate_offset};
                        checksum = (checksum * 16777619u) ^ gate_up_packed[offset];
                    }}
                    uint field = 4u;
                    uint metadata_offset = {gate_offset};
                    uint metadata = gate_up_packed[metadata_offset];
                    checksum = (checksum * 16777619u) ^ metadata;
                }}
                uint output_offset = projection == 0u ? 0u : E1_OUTPUTS;
                checksums[assignment * SCAN_OUTPUTS + output_offset + output]
                    = checksum;
            }} else {{
                for (uint group = 0; group < DOWN_GROUPS; ++group) {{
                    for (uint field = 0; field < 4u; ++field) {{
                        uint offset = {down_offset};
                        checksum = (checksum * 16777619u) ^ down_packed[offset];
                    }}
                    uint field = 4u;
                    uint metadata_offset = {down_offset};
                    uint metadata = down_packed[metadata_offset];
                    checksum = (checksum * 16777619u) ^ metadata;
                }}
                checksums[assignment * SCAN_OUTPUTS + 2u * E1_OUTPUTS + output]
                    = checksum;
            }}
        }}
    """


def hy3_q2_components_to_mlx(
    components: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    import mlx.core as mx

    validated, _capacity = _validated_components(components)
    return {name: mx.array(value) for name, value in validated.items()}


_SOURCE_SCAN_KERNELS: dict[Hy3Q2SourceControl, Any] = {}
_PACKED_SCAN_KERNELS: dict[Hy3Q2LayoutCandidate, Any] = {}


def build_hy3_q2_source_scan_kernel(control: Hy3Q2SourceControl) -> Any:
    import mlx.core as mx

    cached = _SOURCE_SCAN_KERNELS.get(control)
    if cached is None:
        cached = mx.fast.metal_kernel(
            name=f"mtplx_hy3_q2_layout_source_v{control.vector_width}",
            input_names=[
                "expert_slots",
                "gate_weight",
                "gate_scales",
                "gate_biases",
                "up_weight",
                "up_scales",
                "up_biases",
                "down_weight",
                "down_scales",
                "down_biases",
            ],
            output_names=["checksums"],
            source=render_hy3_q2_source_scan_source(control),
        )
        _SOURCE_SCAN_KERNELS[control] = cached
    return cached


def build_hy3_q2_packed_scan_kernel(candidate: Hy3Q2LayoutCandidate) -> Any:
    import mlx.core as mx

    cached = _PACKED_SCAN_KERNELS.get(candidate)
    if cached is None:
        cached = mx.fast.metal_kernel(
            name=f"mtplx_hy3_q2_layout_{candidate.name}",
            input_names=["expert_slots", "gate_up_packed", "down_packed"],
            output_names=["checksums"],
            source=render_hy3_q2_packed_scan_source(candidate),
        )
        _PACKED_SCAN_KERNELS[candidate] = cached
    return cached


def _validate_mlx_slots(expert_slots: Any, *, capacity: int) -> int:
    import mlx.core as mx

    if (
        getattr(expert_slots, "ndim", 0) != 2
        or int(expert_slots.shape[0]) not in HY3_Q2_ROWS
        or int(expert_slots.shape[1]) != HY3_Q2_TOP_K
        or expert_slots.dtype != mx.int32
    ):
        raise Hy3Q2LayoutError("expert slots require int32 shape [R1..R8, 8]")
    if capacity <= 0:
        raise Hy3Q2LayoutError("packed capacity must be positive")
    return int(expert_slots.shape[0])


def _scan_grid(rows: int, vector_width: int) -> tuple[int, int, int]:
    assignments, outputs = hy3_q2_scan_output_shape(rows)
    return assignments * (outputs // vector_width), 1, 1


def scan_hy3_q2_source(
    components: Mapping[str, Any],
    expert_slots: Any,
    *,
    control: Hy3Q2SourceControl,
) -> Any:
    import mlx.core as mx

    expected = hy3_q2_component_shapes()
    if set(components) != set(expected):
        raise Hy3Q2LayoutError("source scan requires exactly nine Q2 components")
    capacity = int(components["gate_proj.weight"].shape[0])
    for name, shape in expected.items():
        value = components[name]
        dtype = mx.uint32 if name.endswith(".weight") else mx.uint16
        if tuple(int(item) for item in value.shape) != (capacity, *shape):
            raise Hy3Q2LayoutError(f"wrong source scan shape for {name}")
        if value.dtype != dtype:
            raise Hy3Q2LayoutError(f"wrong source scan dtype for {name}")
    rows = _validate_mlx_slots(expert_slots, capacity=capacity)
    kernel = build_hy3_q2_source_scan_kernel(control)
    inputs = [expert_slots.reshape(-1)] + [
        components[name]
        for name in (
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
    ]
    (output,) = kernel(
        inputs=inputs,
        grid=_scan_grid(rows, control.vector_width),
        threadgroup=(256, 1, 1),
        output_shapes=[hy3_q2_scan_output_shape(rows)],
        output_dtypes=[mx.uint32],
    )
    return output


def scan_hy3_q2_packed(
    packed: PackedHy3Q2Bank | MlxPackedHy3Q2Bank,
    expert_slots: Any,
) -> Any:
    import mlx.core as mx

    mlx_packed = packed.to_mlx() if isinstance(packed, PackedHy3Q2Bank) else packed
    capacity = len(mlx_packed.identities)
    if (
        int(mlx_packed.gate_up.shape[0]) != capacity
        or int(mlx_packed.down.shape[0]) != capacity
    ):
        raise Hy3Q2LayoutError("packed buffers do not match slot capacity")
    if mlx_packed.gate_up.dtype != mx.uint32 or mlx_packed.down.dtype != mx.uint32:
        raise Hy3Q2LayoutError("packed scan buffers must be uint32")
    rows = _validate_mlx_slots(expert_slots, capacity=capacity)
    candidate = mlx_packed.candidate
    kernel = build_hy3_q2_packed_scan_kernel(candidate)
    (output,) = kernel(
        inputs=[expert_slots.reshape(-1), mlx_packed.gate_up, mlx_packed.down],
        grid=_scan_grid(rows, candidate.vector_width),
        threadgroup=(256, 1, 1),
        output_shapes=[hy3_q2_scan_output_shape(rows)],
        output_dtypes=[mx.uint32],
    )
    return output


def hy3_q2_component_shapes() -> dict[str, tuple[int, int]]:
    """Return the exact per-expert affine-Q2 shapes from the Hy3 manifest."""

    return {
        "gate_proj.weight": (1536, 256),
        "gate_proj.scales": (1536, 64),
        "gate_proj.biases": (1536, 64),
        "up_proj.weight": (1536, 256),
        "up_proj.scales": (1536, 64),
        "up_proj.biases": (1536, 64),
        "down_proj.weight": (4096, 96),
        "down_proj.scales": (4096, 24),
        "down_proj.biases": (4096, 24),
    }


def hy3_q2_component_bytes() -> dict[str, int]:
    """Return byte accounting for one authoritative Hy3-Q2 expert record."""

    return {
        name: rows * columns * (4 if name.endswith(".weight") else 2)
        for name, (rows, columns) in hy3_q2_component_shapes().items()
    }
