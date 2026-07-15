"""Measured E1 r2 primitives built on MLX's tuned affine-Q2 gather path.

This module is intentionally benchmark-only.  It keeps issue #65 arithmetic
separate from the construction-time component layout tracked by issue #68.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mtplx.hy3_expert_fused_e1 import (
    HY3_E1_BITS,
    HY3_E1_GROUP_SIZE,
    HY3_E1_HIDDEN_SIZE,
    HY3_E1_INTERMEDIATE_SIZE,
    HY3_E1_TOP_K,
    Hy3ExpertE1Candidate,
    render_hy3_expert_fused_e1_source,
)


HY3_E1_R2_LAYOUTS = (
    "gate_then_up",
    "output_interleaved",
    "block_interleaved_16",
    "block_interleaved_32",
    "block_interleaved_64",
    "block_interleaved_128",
    "block_interleaved_256",
)
HY3_E1_R2_ASSIGNMENT_ORDERS = ("original", "slot_sorted")
HY3_E1_R2_EXECUTIONS = ("eager", "compiled")


@dataclass(frozen=True, slots=True)
class Hy3E1R2Candidate:
    """One attributable packed-QMM graph construction."""

    layout: str
    assignment_order: str
    execution: str

    def __post_init__(self) -> None:
        if self.layout not in HY3_E1_R2_LAYOUTS:
            raise ValueError(f"unknown E1 r2 layout: {self.layout}")
        if self.assignment_order not in HY3_E1_R2_ASSIGNMENT_ORDERS:
            raise ValueError(f"unknown E1 r2 assignment order: {self.assignment_order}")
        if self.execution not in HY3_E1_R2_EXECUTIONS:
            raise ValueError(f"unknown E1 r2 execution: {self.execution}")

    @property
    def name(self) -> str:
        if self.layout == "gate_then_up":
            layout = "halves"
        elif self.layout == "output_interleaved":
            layout = "interleaved"
        else:
            layout = f"block{_layout_block_size(self.layout)}"
        order = "direct" if self.assignment_order == "original" else "sorted"
        return f"packed_{layout}_{order}_{self.execution}"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "layout": self.layout,
            "assignment_order": self.assignment_order,
            "execution": self.execution,
        }


def hy3_e1_r2_candidates() -> tuple[Hy3E1R2Candidate, ...]:
    return tuple(
        Hy3E1R2Candidate(layout, assignment_order, execution)
        for layout, assignment_order, execution in product(
            HY3_E1_R2_LAYOUTS,
            HY3_E1_R2_ASSIGNMENT_ORDERS,
            HY3_E1_R2_EXECUTIONS,
        )
    )


def _layout_block_size(layout: str) -> int | None:
    if layout == "gate_then_up":
        return None
    if layout == "output_interleaved":
        return 1
    prefix = "block_interleaved_"
    if layout.startswith(prefix):
        try:
            block = int(layout.removeprefix(prefix))
        except ValueError as exc:  # pragma: no cover - guarded by the catalog
            raise ValueError(f"unknown E1 r2 layout: {layout}") from exc
        if block > 0:
            return block
    raise ValueError(f"unknown E1 r2 layout: {layout}")


def pack_component_pair(
    gate: mx.array,
    up: mx.array,
    *,
    layout: str,
) -> mx.array:
    """Pack equal component rows without changing their total resident bytes."""

    if layout not in HY3_E1_R2_LAYOUTS:
        raise ValueError(f"unknown E1 r2 layout: {layout}")
    if gate.dtype != up.dtype or tuple(gate.shape) != tuple(up.shape):
        raise ValueError("gate/up component arrays must have equal shape and dtype")
    if gate.ndim < 2:
        raise ValueError("gate/up component arrays require a component-row axis")
    block = _layout_block_size(layout)
    if block is None:
        return mx.concatenate((gate, up), axis=1)
    shape = tuple(int(value) for value in gate.shape)
    if shape[1] % block:
        raise ValueError("component-row width must divide the interleave block")
    gate_blocks = gate.reshape(shape[0], shape[1] // block, block, *shape[2:])
    up_blocks = up.reshape(shape[0], shape[1] // block, block, *shape[2:])
    interleaved = mx.stack((gate_blocks, up_blocks), axis=2)
    return interleaved.reshape(shape[0], shape[1] * 2, *shape[2:])


def reconstruct_component_pair(
    packed: mx.array,
    *,
    split_at: int,
    layout: str,
) -> tuple[mx.array, mx.array]:
    """Expose source-order component views from one replacement packed array."""

    if layout not in HY3_E1_R2_LAYOUTS:
        raise ValueError(f"unknown E1 r2 layout: {layout}")
    split_at = int(split_at)
    if packed.ndim < 2 or split_at <= 0 or int(packed.shape[1]) != split_at * 2:
        raise ValueError("packed component axis does not match split_at")
    block = _layout_block_size(layout)
    if block is None:
        return packed[:, :split_at], packed[:, split_at:]
    if split_at % block:
        raise ValueError("component-row width must divide the interleave block")
    shape = tuple(int(value) for value in packed.shape)
    paired = packed.reshape(
        shape[0],
        split_at // block,
        2,
        block,
        *shape[2:],
    )
    gate = paired[:, :, 0].reshape(shape[0], split_at, *shape[2:])
    up = paired[:, :, 1].reshape(shape[0], split_at, *shape[2:])
    return gate, up


@dataclass(frozen=True, slots=True)
class PackedHy3E1Bank:
    """One N=3072 affine-Q2 component bank replacing gate/up source arrays."""

    weight: mx.array
    scales: mx.array
    biases: mx.array
    split_at: int
    layout: str
    source_bytes: int
    packed_bytes: int


def pack_hy3_e1_component_bank(
    source: dict[str, mx.array],
    *,
    layout: str,
) -> PackedHy3E1Bank:
    """Materialize one replacement gate/up bank and evaluate construction work."""

    names = (
        "gate_proj.weight",
        "gate_proj.scales",
        "gate_proj.biases",
        "up_proj.weight",
        "up_proj.scales",
        "up_proj.biases",
    )
    if set(source) != set(names):
        raise ValueError("E1 r2 source bank must contain exactly gate/up components")
    expected_tail = {
        "weight": (HY3_E1_INTERMEDIATE_SIZE, HY3_E1_HIDDEN_SIZE * HY3_E1_BITS // 32),
        "scales": (HY3_E1_INTERMEDIATE_SIZE, HY3_E1_HIDDEN_SIZE // HY3_E1_GROUP_SIZE),
        "biases": (HY3_E1_INTERMEDIATE_SIZE, HY3_E1_HIDDEN_SIZE // HY3_E1_GROUP_SIZE),
    }
    for projection in ("gate_proj", "up_proj"):
        for component, tail in expected_tail.items():
            value = source[f"{projection}.{component}"]
            wanted_dtype = mx.uint32 if component == "weight" else mx.bfloat16
            if value.ndim != 3 or tuple(int(x) for x in value.shape[1:]) != tail:
                raise ValueError(f"E1 r2 {projection}.{component} has invalid geometry")
            if value.dtype != wanted_dtype:
                raise ValueError(f"E1 r2 {projection}.{component} has invalid dtype")
    packed_weight = pack_component_pair(
        source["gate_proj.weight"], source["up_proj.weight"], layout=layout
    )
    packed_scales = pack_component_pair(
        source["gate_proj.scales"], source["up_proj.scales"], layout=layout
    )
    packed_biases = pack_component_pair(
        source["gate_proj.biases"], source["up_proj.biases"], layout=layout
    )
    mx.eval(packed_weight, packed_scales, packed_biases)
    source_bytes = sum(int(source[name].nbytes) for name in names)
    packed_bytes = sum(
        int(value.nbytes) for value in (packed_weight, packed_scales, packed_biases)
    )
    if packed_bytes != source_bytes:
        raise RuntimeError("packed E1 bank changed steady-state component bytes")
    return PackedHy3E1Bank(
        weight=packed_weight,
        scales=packed_scales,
        biases=packed_biases,
        split_at=HY3_E1_INTERMEDIATE_SIZE,
        layout=layout,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
    )


def reconstruct_hy3_e1_source_bank(packed: PackedHy3E1Bank) -> dict[str, mx.array]:
    gate_weight, up_weight = reconstruct_component_pair(
        packed.weight,
        split_at=packed.split_at,
        layout=packed.layout,
    )
    gate_scales, up_scales = reconstruct_component_pair(
        packed.scales,
        split_at=packed.split_at,
        layout=packed.layout,
    )
    gate_biases, up_biases = reconstruct_component_pair(
        packed.biases,
        split_at=packed.split_at,
        layout=packed.layout,
    )
    return {
        "gate_proj.weight": gate_weight,
        "gate_proj.scales": gate_scales,
        "gate_proj.biases": gate_biases,
        "up_proj.weight": up_weight,
        "up_proj.scales": up_scales,
        "up_proj.biases": up_biases,
    }


def _selected_inputs(
    token_rows: mx.array,
    expert_slots: mx.array,
    *,
    assignment_order: str,
) -> tuple[mx.array, mx.array, mx.array | None]:
    rows = int(token_rows.shape[0])
    assignments = rows * HY3_E1_TOP_K
    selected = mx.broadcast_to(
        token_rows[:, None, :],
        (rows, HY3_E1_TOP_K, HY3_E1_HIDDEN_SIZE),
    ).reshape(assignments, 1, 1, HY3_E1_HIDDEN_SIZE)
    indices = expert_slots.reshape(assignments, 1)
    if assignment_order == "original":
        return selected, indices, None
    if assignment_order != "slot_sorted":
        raise ValueError(f"unknown E1 r2 assignment order: {assignment_order}")
    order = mx.argsort(indices.reshape(assignments))
    inverse = mx.argsort(order)
    return selected[order], indices[order], inverse


def packed_hy3_e1_qmm(
    token_rows: mx.array,
    expert_slots: mx.array,
    packed: PackedHy3E1Bank,
    *,
    assignment_order: str,
) -> mx.array:
    """Run one tuned gather-QMM to the packed N=3072 boundary."""

    selected, indices, inverse = _selected_inputs(
        token_rows,
        expert_slots,
        assignment_order=assignment_order,
    )
    output = mx.gather_qmm(
        selected,
        packed.weight,
        packed.scales,
        packed.biases,
        rhs_indices=indices,
        transpose=True,
        group_size=HY3_E1_GROUP_SIZE,
        bits=HY3_E1_BITS,
        mode="affine",
        sorted_indices=assignment_order == "slot_sorted",
    )
    if inverse is not None:
        output = output[inverse]
    return output


def split_packed_hy3_e1_output(
    output: mx.array,
    *,
    layout: str,
) -> tuple[mx.array, mx.array]:
    if int(output.shape[-1]) != HY3_E1_INTERMEDIATE_SIZE * 2:
        raise ValueError("packed E1 QMM output must have N=3072")
    block = _layout_block_size(layout)
    if block is None:
        return mx.split(output, [HY3_E1_INTERMEDIATE_SIZE], axis=-1)
    if HY3_E1_INTERMEDIATE_SIZE % block:
        raise ValueError("E1 output width must divide the interleave block")
    prefix = tuple(int(value) for value in output.shape[:-1])
    paired = output.reshape(
        *prefix,
        HY3_E1_INTERMEDIATE_SIZE // block,
        2,
        block,
    )
    gate = paired[..., 0, :].reshape(*prefix, HY3_E1_INTERMEDIATE_SIZE)
    up = paired[..., 1, :].reshape(*prefix, HY3_E1_INTERMEDIATE_SIZE)
    return gate, up


def packed_hy3_e1(
    token_rows: mx.array,
    expert_slots: mx.array,
    packed: PackedHy3E1Bank,
    *,
    assignment_order: str,
) -> mx.array:
    """Run one packed gather-QMM followed by exact MLX SwiGLU."""

    output = packed_hy3_e1_qmm(
        token_rows,
        expert_slots,
        packed,
        assignment_order=assignment_order,
    )
    gate, up = split_packed_hy3_e1_output(output, layout=packed.layout)
    activated = nn.silu(gate) * up
    rows = int(token_rows.shape[0])
    return activated.reshape(rows, HY3_E1_TOP_K, HY3_E1_INTERMEDIATE_SIZE)


def compile_packed_hy3_e1(
    packed: PackedHy3E1Bank,
    *,
    assignment_order: str,
) -> Callable[[mx.array, mx.array], mx.array]:
    """Compile the QMM/split/exact-SwiGLU graph without changing its arrays."""

    def graph(
        token_rows: mx.array,
        expert_slots: mx.array,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ) -> mx.array:
        live = PackedHy3E1Bank(
            weight=weight,
            scales=scales,
            biases=biases,
            split_at=packed.split_at,
            layout=packed.layout,
            source_bytes=packed.source_bytes,
            packed_bytes=packed.packed_bytes,
        )
        return packed_hy3_e1(
            token_rows,
            expert_slots,
            live,
            assignment_order=assignment_order,
        )

    compiled = mx.compile(graph)

    def call(token_rows: mx.array, expert_slots: mx.array) -> mx.array:
        return compiled(
            token_rows,
            expert_slots,
            packed.weight,
            packed.scales,
            packed.biases,
        )

    return call


def unpack_affine_q2_row(
    packed_words: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    group_size: int = HY3_E1_GROUP_SIZE,
) -> np.ndarray:
    """CPU scalar/reference affine-Q2 dequantization in manifest bit order."""

    words = np.asarray(packed_words, dtype=np.uint32).reshape(-1)
    scales_f32 = np.asarray(scales, dtype=np.float32).reshape(-1)
    biases_f32 = np.asarray(biases, dtype=np.float32).reshape(-1)
    if scales_f32.shape != biases_f32.shape:
        raise ValueError("affine-Q2 scales and biases must align")
    shifts = np.arange(16, dtype=np.uint32) * np.uint32(2)
    quantized = ((words[:, None] >> shifts[None, :]) & np.uint32(3)).reshape(-1)
    if quantized.size != scales_f32.size * int(group_size):
        raise ValueError("affine-Q2 words do not match parameter groups")
    groups = np.arange(quantized.size) // int(group_size)
    return (
        quantized.astype(np.float32) * scales_f32[groups] + biases_f32[groups]
    ).astype(np.float32)


def scalar_affine_q2_dot(
    activation: np.ndarray,
    packed_words: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    group_size: int = HY3_E1_GROUP_SIZE,
) -> float:
    """Float64 accumulation over the explicit affine-Q2 scalar formula."""

    values = unpack_affine_q2_row(
        packed_words,
        scales,
        biases,
        group_size=group_size,
    )
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    if x.shape != values.shape:
        raise ValueError("activation and dequantized Q2 row must align")
    return float(np.dot(x.astype(np.float64), values.astype(np.float64)))


def _pairwise_float32_sum(values: np.ndarray) -> np.float32:
    current = np.asarray(values, dtype=np.float32).copy()
    if current.size == 0 or current.size & (current.size - 1):
        raise ValueError("pairwise reduction requires a power-of-two vector")
    while current.size > 1:
        current = np.add(current[::2], current[1::2], dtype=np.float32)
    return np.float32(current[0])


def simulate_r1_affine_q2_dot(
    activation: np.ndarray,
    packed_words: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    group_size: int = HY3_E1_GROUP_SIZE,
    k_tile: int,
) -> float:
    """Simulate r1's per-value FP32 accumulation and tiled SIMD reduction."""

    values = unpack_affine_q2_row(
        packed_words,
        scales,
        biases,
        group_size=group_size,
    )
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    k_tile = int(k_tile)
    if x.shape != values.shape or x.size % k_tile or k_tile % 32:
        raise ValueError("r1 simulation requires aligned K and 32-value vectors")
    total = np.float32(0.0)
    for k0 in range(0, x.size, k_tile):
        lanes = np.zeros(32, dtype=np.float32)
        for lane in range(k_tile // 32):
            start = k0 + lane * 32
            partial = np.float32(0.0)
            for offset in range(32):
                product_value = np.float32(x[start + offset] * values[start + offset])
                partial = np.float32(partial + product_value)
            lanes[lane] = partial
        total = np.float32(total + _pairwise_float32_sum(lanes))
    return float(total)


def classify_stage_discrepancy(
    *,
    decoded_values_match: bool,
    r1_matches_simulated_reduction: bool,
    activation_matches_from_preactivations: bool,
    bf16_cast_matches: bool,
) -> str:
    if not decoded_values_match:
        return "indexing_or_dequantization_error"
    if not r1_matches_simulated_reduction:
        return "unclassified_preactivation_error"
    if not activation_matches_from_preactivations:
        return "activation_arithmetic_error"
    if not bf16_cast_matches:
        return "bf16_cast_error"
    return "arithmetic_reduction_order"


_R1_PREACT_KERNELS: dict[Hy3ExpertE1Candidate, Any] = {}


def _r1_preactivation_kernel(candidate: Hy3ExpertE1Candidate) -> Any:
    cached = _R1_PREACT_KERNELS.get(candidate)
    if cached is not None:
        return cached
    source = render_hy3_expert_fused_e1_source(candidate)
    marker = "output_values[assignment * N + n] = T(silu_value * up_value);"
    replacement = (
        "gate_values[assignment * N + n] = gate_value;\n"
        "                up_values[assignment * N + n] = up_value;"
    )
    if source.count(marker) != 1:
        raise RuntimeError("r1 source no longer exposes the expected output boundary")
    source = source.replace(marker, replacement)
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_hy3_q2_e1_preact_{candidate.name}",
        input_names=[
            "token_rows",
            "expert_slots",
            "gate_weight",
            "gate_scales",
            "gate_biases",
            "up_weight",
            "up_scales",
            "up_biases",
        ],
        output_names=["gate_values", "up_values"],
        source=source,
    )
    _R1_PREACT_KERNELS[candidate] = kernel
    return kernel


def r1_hy3_e1_preactivations(
    token_rows: mx.array,
    expert_slots: mx.array,
    source: dict[str, mx.array],
    *,
    candidate: Hy3ExpertE1Candidate,
) -> tuple[mx.array, mx.array]:
    """Expose r1's BF16 gate/up boundary for stage attribution only."""

    rows = int(token_rows.shape[0])
    assignments = rows * HY3_E1_TOP_K
    n_tiles = HY3_E1_INTERMEDIATE_SIZE // candidate.output_n_tile
    threads = candidate.simd_groups * 32
    kernel = _r1_preactivation_kernel(candidate)
    gate, up = kernel(
        inputs=[
            mx.contiguous(token_rows),
            mx.contiguous(expert_slots.reshape(assignments)),
            source["gate_proj.weight"],
            source["gate_proj.scales"],
            source["gate_proj.biases"],
            source["up_proj.weight"],
            source["up_proj.scales"],
            source["up_proj.biases"],
        ],
        template=[("T", mx.bfloat16)],
        grid=(threads * assignments * n_tiles, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (assignments, HY3_E1_INTERMEDIATE_SIZE),
            (assignments, HY3_E1_INTERMEDIATE_SIZE),
        ],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    shape = (rows, HY3_E1_TOP_K, HY3_E1_INTERMEDIATE_SIZE)
    return gate.reshape(shape), up.reshape(shape)
