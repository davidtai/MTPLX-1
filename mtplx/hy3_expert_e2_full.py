"""Isolated Hy3-Q2 E2 down + router-reduction benchmark primitives.

The runtime deliberately does not import this module.  It is a shape-specific
measurement lane for issue #66: E1 produces BF16 ``[R, 8, 1536]`` values, E2
computes the eight affine-Q2 down projections in router order, applies BF16
router weights, and writes BF16 ``[R, 4096]`` directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any

import mlx.core as mx
import numpy as np


HY3_E2_ROWS = tuple(range(1, 9))
HY3_E2_TOP_K = 8
HY3_E2_HIDDEN_SIZE = 4096
HY3_E2_INTERMEDIATE_SIZE = 1536
HY3_E2_BITS = 2
HY3_E2_GROUP_SIZE = 64
HY3_E2_PACKED_VALUES_PER_WORD = 16
HY3_E2_PACKED_VALUES_PER_VECTOR = 32

HY3_E2_OUTPUT_N_TILES = (32, 64, 128)
HY3_E2_K_TILES = (32, 64, 96, 192)
HY3_E2_SIMD_GROUPS = (2, 4, 6, 8)
HY3_E2_ACTIVATION_MODES = ("exact", "fast")

HY3_E2_MAX_ABS_ERROR = 0.5
HY3_E2_MAX_NORMALIZED_RMSE = 0.02


class Hy3ExpertE2Ineligible(ValueError):
    """Raised before dispatch for inputs outside the issue-#66 lane."""


@dataclass(frozen=True, slots=True)
class Hy3ExpertE2Candidate:
    """One full-core topology with an independently labelled SiLU mode."""

    output_n_tile: int
    k_tile: int
    simd_groups: int
    activation_mode: str = "exact"

    def __post_init__(self) -> None:
        if self.output_n_tile not in HY3_E2_OUTPUT_N_TILES:
            raise ValueError("E2 output-N tile must be 32, 64, or 128")
        if self.k_tile not in HY3_E2_K_TILES:
            raise ValueError("E2 K tile must be 32, 64, 96, or 192")
        if self.simd_groups not in HY3_E2_SIMD_GROUPS:
            raise ValueError("E2 SIMDgroup count must be 2, 4, 6, or 8")
        if self.activation_mode not in HY3_E2_ACTIVATION_MODES:
            raise ValueError("E2 activation mode must be exact or fast")

    @property
    def name(self) -> str:
        return (
            f"n{self.output_n_tile}_k{self.k_tile}_"
            f"sg{self.simd_groups}_{self.activation_mode}"
        )

    @property
    def unsupported_reason(self) -> str | None:
        if self.output_n_tile % self.simd_groups:
            return "output-N tile must divide evenly across SIMDgroups"
        outputs_per_simdgroup = self.output_n_tile // self.simd_groups
        if not 4 <= outputs_per_simdgroup <= 16:
            return "outputs per SIMDgroup must stay between 4 and 16"
        if HY3_E2_HIDDEN_SIZE % self.output_n_tile:
            return "output-N tile must divide the Hy3 hidden width"
        if HY3_E2_INTERMEDIATE_SIZE % self.k_tile:
            return "K tile must divide the Hy3 intermediate width"
        if self.simd_groups * 32 > 256:
            return "threadgroup exceeds the 256-thread resource boundary"
        return None

    @property
    def supported(self) -> bool:
        return self.unsupported_reason is None

    @property
    def promotion_eligible(self) -> bool:
        return self.supported and self.activation_mode == "exact"

    @property
    def classification(self) -> str:
        if not self.supported:
            return "unsupported_geometry"
        if self.activation_mode == "fast":
            return "fast_swiglu_ablation_only"
        return "exact_e2_candidate"

    def resources(self, *, rows: int) -> dict[str, Any]:
        rows = int(rows)
        if rows not in HY3_E2_ROWS:
            raise ValueError("E2 resource accounting requires R1 through R8")
        stock_bytes = rows * HY3_E2_TOP_K * HY3_E2_HIDDEN_SIZE * 2
        fused_bytes = rows * HY3_E2_HIDDEN_SIZE * 2
        down_bytes_per_output = (
            HY3_E2_INTERMEDIATE_SIZE * HY3_E2_BITS // 8
            + 2 * (HY3_E2_INTERMEDIATE_SIZE // HY3_E2_GROUP_SIZE) * 2
        )
        down_weight_parameter_bytes_per_token = (
            HY3_E2_TOP_K * HY3_E2_HIDDEN_SIZE * down_bytes_per_output
        )
        threadgroups_per_token = HY3_E2_HIDDEN_SIZE // self.output_n_tile
        activation_load_bytes_per_token = (
            HY3_E2_TOP_K * HY3_E2_INTERMEDIATE_SIZE * 2 * threadgroups_per_token
        )
        outputs = (
            self.output_n_tile // self.simd_groups
            if self.output_n_tile % self.simd_groups == 0
            else 0
        )
        return {
            "threads_per_threadgroup": self.simd_groups * 32,
            "outputs_per_simdgroup": outputs,
            "threadgroup_activation_bytes": self.k_tile * 2,
            "accumulator_floats_per_lane": outputs,
            "packed_weight_vector": "uint2",
            "packed_values_per_vector": HY3_E2_PACKED_VALUES_PER_VECTOR,
            "stock_down_output_bytes": stock_bytes,
            "fused_down_output_bytes": fused_bytes,
            "avoided_intermediate_bytes": stock_bytes - fused_bytes,
            "stock_intermediate_write_read_bytes": stock_bytes * 2,
            "estimated_saved_write_read_bytes": stock_bytes * 2,
            "down_weight_parameter_bytes_per_token": (
                down_weight_parameter_bytes_per_token
            ),
            "candidate_activation_load_bytes_per_token": (
                activation_load_bytes_per_token
            ),
            "threadgroups_per_token": threadgroups_per_token,
            "expert_order": "router slot 0 through 7",
        }

    def to_dict(self, *, rows: int | None = None) -> dict[str, Any]:
        result = {
            "name": self.name,
            "output_n_tile": self.output_n_tile,
            "k_tile": self.k_tile,
            "simd_groups": self.simd_groups,
            "activation_mode": self.activation_mode,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "classification": self.classification,
            "promotion_eligible": self.promotion_eligible,
        }
        if rows is not None:
            result["resources"] = self.resources(rows=rows)
        return result


def hy3_e2_candidate_catalog() -> tuple[Hy3ExpertE2Candidate, ...]:
    """Return the complete requested cross-product, including illegal rows."""

    return tuple(
        Hy3ExpertE2Candidate(n_tile, k_tile, simd_groups, activation_mode)
        for activation_mode, n_tile, k_tile, simd_groups in product(
            HY3_E2_ACTIVATION_MODES,
            HY3_E2_OUTPUT_N_TILES,
            HY3_E2_K_TILES,
            HY3_E2_SIMD_GROUPS,
        )
    )


def hy3_e2_supported_candidates(
    *,
    activation_modes: tuple[str, ...] = HY3_E2_ACTIVATION_MODES,
) -> tuple[Hy3ExpertE2Candidate, ...]:
    unknown = set(activation_modes) - set(HY3_E2_ACTIVATION_MODES)
    if unknown:
        raise ValueError(f"unknown E2 activation modes: {sorted(unknown)}")
    return tuple(
        candidate
        for candidate in hy3_e2_candidate_catalog()
        if candidate.supported and candidate.activation_mode in activation_modes
    )


def hy3_e2_down_bank_shapes(capacity: int) -> dict[str, tuple[int, ...]]:
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("E2 component-bank capacity must be positive")
    return {
        "down_proj.weight": (
            capacity,
            HY3_E2_HIDDEN_SIZE,
            HY3_E2_INTERMEDIATE_SIZE * HY3_E2_BITS // 32,
        ),
        "down_proj.scales": (
            capacity,
            HY3_E2_HIDDEN_SIZE,
            HY3_E2_INTERMEDIATE_SIZE // HY3_E2_GROUP_SIZE,
        ),
        "down_proj.biases": (
            capacity,
            HY3_E2_HIDDEN_SIZE,
            HY3_E2_INTERMEDIATE_SIZE // HY3_E2_GROUP_SIZE,
        ),
    }


def hy3_e2_candidate_passes(
    *,
    activation_mode: str,
    per_expert_shape_exact: bool,
    final_shape_exact: bool,
    dtype_bf16: bool,
    all_finite: bool,
    per_expert_max_abs_error: float,
    final_max_abs_error: float,
    final_normalized_rmse: float,
    max_abs_tolerance: float = HY3_E2_MAX_ABS_ERROR,
    normalized_rmse_tolerance: float = HY3_E2_MAX_NORMALIZED_RMSE,
) -> bool:
    """Gate exact candidates at both down and routed-reduction boundaries."""

    if activation_mode not in HY3_E2_ACTIVATION_MODES:
        raise ValueError("E2 activation mode must be exact or fast")
    values = (
        float(per_expert_max_abs_error),
        float(final_max_abs_error),
        float(final_normalized_rmse),
    )
    return (
        activation_mode == "exact"
        and bool(per_expert_shape_exact)
        and bool(final_shape_exact)
        and bool(dtype_bf16)
        and bool(all_finite)
        and all(math.isfinite(value) and value >= 0.0 for value in values)
        and values[0] <= float(max_abs_tolerance)
        and values[1] <= float(max_abs_tolerance)
        and values[2] <= float(normalized_rmse_tolerance)
    )


def _round_numpy_bf16(value: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 with round-to-nearest-even, returned as float32."""

    array = np.asarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def simulate_hy3_bf16_route_reduce(
    per_expert_values: np.ndarray,
    route_weights: np.ndarray,
) -> np.ndarray:
    """Scalar oracle for the BF16 multiply and router-order accumulation."""

    values = np.asarray(per_expert_values, dtype=np.float32)
    weights = np.asarray(route_weights, dtype=np.float32)
    if values.ndim != 3 or weights.shape != values.shape[:2]:
        raise ValueError("route reduction requires [R,8,N] values and [R,8] weights")
    if values.shape[1] != HY3_E2_TOP_K:
        raise ValueError("route reduction requires exactly eight ordered experts")
    values = _round_numpy_bf16(values)
    weights = _round_numpy_bf16(weights)
    total = np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    total = _round_numpy_bf16(total)
    for expert in range(HY3_E2_TOP_K):
        weighted = _round_numpy_bf16(values[:, expert] * weights[:, expert, None])
        total = _round_numpy_bf16(total + weighted)
    return total


def render_hy3_e2_swiglu_source(activation_mode: str) -> str:
    """Render the separately attributable exact/fast elementwise E1 boundary."""

    if activation_mode not in HY3_E2_ACTIVATION_MODES:
        raise ValueError("E2 activation mode must be exact or fast")
    exp_call = (
        "metal::exp(metal::abs(gate_value))"
        if activation_mode == "exact"
        else "fast::exp(metal::abs(gate_value))"
    )
    return f"""
        uint index = thread_position_in_grid.x;
        if (index < uint(total_size)) {{
            T gate_value = gate_values[index];
            T up_value = up_values[index];
            T sigmoid_base = T(1) / (T(1) + T({exp_call}));
            T sigmoid_value = gate_value < T(0)
                ? sigmoid_base : T(1) - sigmoid_base;
            T silu_value = T(gate_value * sigmoid_value);
            output_values[index] = T(silu_value * up_value);
        }}
    """


_SWIGLU_KERNELS: dict[str, Any] = {}


def hy3_e2_swiglu(
    gate_values: mx.array,
    up_values: mx.array,
    *,
    activation_mode: str,
) -> mx.array:
    """Apply an explicit BF16 exact or fast-SiLU boundary for full-core runs."""

    if (
        gate_values.dtype != mx.bfloat16
        or up_values.dtype != mx.bfloat16
        or tuple(gate_values.shape) != tuple(up_values.shape)
    ):
        raise Hy3ExpertE2Ineligible("E2 SwiGLU requires equal BF16 gate/up arrays")
    total = int(gate_values.size)
    kernel = _SWIGLU_KERNELS.get(activation_mode)
    if kernel is None:
        source = render_hy3_e2_swiglu_source(activation_mode)
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_hy3_q2_e2_swiglu_{activation_mode}",
            input_names=["gate_values", "up_values", "total_size"],
            output_names=["output_values"],
            source=source,
        )
        _SWIGLU_KERNELS[activation_mode] = kernel
    (output,) = kernel(
        inputs=[
            mx.contiguous(gate_values.reshape(total)),
            mx.contiguous(up_values.reshape(total)),
            total,
        ],
        template=[("T", mx.bfloat16)],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.bfloat16],
    )
    return output.reshape(gate_values.shape)


def render_hy3_expert_e2_source(
    candidate: Hy3ExpertE2Candidate,
    *,
    diagnostic: bool,
) -> str:
    """Render fused affine-Q2 down and ordered BF16 route reduction."""

    if not candidate.supported:
        raise Hy3ExpertE2Ineligible(candidate.unsupported_reason or candidate.name)
    diagnostic_write = (
        "per_expert_values[(token * TOP_K + expert) * N + n] = expert_value;"
        if diagnostic
        else ""
    )
    return f"""
        using namespace metal;

        constexpr int K = {HY3_E2_INTERMEDIATE_SIZE};
        constexpr int N = {HY3_E2_HIDDEN_SIZE};
        constexpr int TOP_K = {HY3_E2_TOP_K};
        constexpr int GROUP_SIZE = {HY3_E2_GROUP_SIZE};
        constexpr int K_TILE = {candidate.k_tile};
        constexpr int N_TILE = {candidate.output_n_tile};
        constexpr int NUM_SIMDGROUPS = {candidate.simd_groups};
        constexpr int THREADS = NUM_SIMDGROUPS * 32;
        constexpr int OUTPUTS_PER_SIMDGROUP = N_TILE / NUM_SIMDGROUPS;
        constexpr int VALUES_PER_WORD = 16;
        constexpr int VALUES_PER_VECTOR = 32;
        constexpr int PACKED_WORDS_PER_ROW = K / VALUES_PER_WORD;
        constexpr int GROUPS_PER_ROW = K / GROUP_SIZE;
        constexpr int N_TILES = N / N_TILE;

        uint tid = thread_position_in_threadgroup.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint task = threadgroup_position_in_grid.x;
        uint token = task / N_TILES;
        uint n_tile_index = task - token * N_TILES;
        int n_base = int(n_tile_index) * N_TILE
            + int(simd_gid) * OUTPUTS_PER_SIMDGROUP;

        threadgroup T activation_tile[K_TILE];
        T routed_acc[OUTPUTS_PER_SIMDGROUP];
        _Pragma("unroll")
        for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
            routed_acc[output] = T(0);
        }}

        for (int expert = 0; expert < TOP_K; ++expert) {{
            int selected_slot = expert_slots[int(token) * TOP_K + expert];
            float down_acc[OUTPUTS_PER_SIMDGROUP];
            _Pragma("unroll")
            for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
                down_acc[output] = 0.0f;
            }}

            for (int k0 = 0; k0 < K; k0 += K_TILE) {{
                for (int local_k = int(tid); local_k < K_TILE;
                     local_k += THREADS) {{
                    activation_tile[local_k] = hidden_values[
                        (int(token) * TOP_K + expert) * K + k0 + local_k];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                int vector_k = int(lane) * VALUES_PER_VECTOR;
                bool active_vector = vector_k < K_TILE;
                _Pragma("unroll")
                for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
                    float partial = 0.0f;
                    if (active_vector) {{
                        int n = n_base + output;
                        int global_k = k0 + vector_k;
                        int packed_offset =
                            (selected_slot * N + n) * PACKED_WORDS_PER_ROW
                            + global_k / VALUES_PER_WORD;
                        uint2 packed = *((const device uint2*)(
                            down_weight + packed_offset));
                        int parameter_offset =
                            (selected_slot * N + n) * GROUPS_PER_ROW
                            + global_k / GROUP_SIZE;
                        float scale = float(down_scales[parameter_offset]);
                        float bias = float(down_biases[parameter_offset]);
                        _Pragma("unroll")
                        for (int packed_index = 0; packed_index < 2;
                             ++packed_index) {{
                            uint word = packed_index == 0 ? packed.x : packed.y;
                            _Pragma("unroll")
                            for (int value_index = 0; value_index < 16;
                                 ++value_index) {{
                                int local_value =
                                    packed_index * 16 + value_index;
                                float activation = float(
                                    activation_tile[vector_k + local_value]);
                                float weight_value = float(
                                    (word >> (value_index * 2)) & 0x3u
                                ) * scale + bias;
                                partial += activation * weight_value;
                            }}
                        }}
                    }}
                    down_acc[output] += simd_sum(partial);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            if (lane == 0) {{
                _Pragma("unroll")
                for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
                    int n = n_base + output;
                    T expert_value = T(down_acc[output]);
                    {diagnostic_write}
                    T weighted_value = T(expert_value * route_weights[
                        int(token) * TOP_K + expert]);
                    routed_acc[output] = T(routed_acc[output] + weighted_value);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (lane == 0) {{
            _Pragma("unroll")
            for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
                int n = n_base + output;
                routed_values[int(token) * N + n] = routed_acc[output];
            }}
        }}
    """


_E2_KERNELS: dict[tuple[int, int, int, bool], Any] = {}


def _build_e2_kernel(candidate: Hy3ExpertE2Candidate, *, diagnostic: bool) -> Any:
    # Activation is upstream of E2. Exact and fast full-core arms intentionally
    # share the same down kernel so their delta stays attributable to SwiGLU.
    key = (
        candidate.output_n_tile,
        candidate.k_tile,
        candidate.simd_groups,
        bool(diagnostic),
    )
    cached = _E2_KERNELS.get(key)
    if cached is not None:
        return cached
    output_names = ["routed_values"]
    if diagnostic:
        output_names.append("per_expert_values")
    kernel = mx.fast.metal_kernel(
        name=(
            f"mtplx_hy3_q2_e2_n{candidate.output_n_tile}_"
            f"k{candidate.k_tile}_sg{candidate.simd_groups}_"
            f"{'diagnostic' if diagnostic else 'fused'}"
        ),
        input_names=[
            "hidden_values",
            "expert_slots",
            "route_weights",
            "down_weight",
            "down_scales",
            "down_biases",
        ],
        output_names=output_names,
        source=render_hy3_expert_e2_source(candidate, diagnostic=diagnostic),
    )
    _E2_KERNELS[key] = kernel
    return kernel


def _validate_e2_arrays(
    hidden_values: mx.array,
    expert_slots: mx.array,
    route_weights: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
) -> tuple[int, int]:
    if (
        hidden_values.ndim != 3
        or int(hidden_values.shape[0]) not in HY3_E2_ROWS
        or tuple(int(value) for value in hidden_values.shape[1:])
        != (HY3_E2_TOP_K, HY3_E2_INTERMEDIATE_SIZE)
        or hidden_values.dtype != mx.bfloat16
    ):
        raise Hy3ExpertE2Ineligible(
            "E2 hidden values require BF16 shape [R1..R8, 8, 1536]"
        )
    rows = int(hidden_values.shape[0])
    if expert_slots.dtype != mx.int32 or tuple(
        int(value) for value in expert_slots.shape
    ) != (rows, HY3_E2_TOP_K):
        raise Hy3ExpertE2Ineligible("E2 expert slots require int32 shape [R, 8]")
    if route_weights.dtype != mx.bfloat16 or tuple(
        int(value) for value in route_weights.shape
    ) != (rows, HY3_E2_TOP_K):
        raise Hy3ExpertE2Ineligible("E2 route weights require BF16 shape [R, 8]")
    capacity = int(down_weight.shape[0]) if down_weight.ndim == 3 else 0
    expected = hy3_e2_down_bank_shapes(capacity) if capacity > 0 else {}
    arrays = {
        "down_proj.weight": down_weight,
        "down_proj.scales": down_scales,
        "down_proj.biases": down_biases,
    }
    for name, value in arrays.items():
        wanted_dtype = mx.uint32 if name.endswith(".weight") else mx.bfloat16
        if value.dtype != wanted_dtype or tuple(
            int(d) for d in value.shape
        ) != expected.get(name):
            raise Hy3ExpertE2Ineligible(
                f"E2 component bank {name} has the wrong shape or dtype"
            )
    return rows, capacity


def hy3_expert_e2_reference(
    hidden_values: mx.array,
    expert_slots: mx.array,
    route_weights: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
) -> tuple[mx.array, mx.array]:
    """Run tuned gather-QMM then the stock BF16 route combine boundary."""

    rows, _capacity = _validate_e2_arrays(
        hidden_values,
        expert_slots,
        route_weights,
        down_weight,
        down_scales,
        down_biases,
    )
    assignments = rows * HY3_E2_TOP_K
    output = mx.gather_qmm(
        hidden_values.reshape(assignments, 1, 1, HY3_E2_INTERMEDIATE_SIZE),
        down_weight,
        down_scales,
        down_biases,
        rhs_indices=expert_slots.reshape(assignments, 1),
        transpose=True,
        group_size=HY3_E2_GROUP_SIZE,
        bits=HY3_E2_BITS,
        mode="affine",
    )
    per_expert = output.reshape(rows, HY3_E2_TOP_K, HY3_E2_HIDDEN_SIZE).astype(
        mx.bfloat16
    )
    routed = (per_expert * route_weights[..., None]).sum(axis=-2).astype(mx.bfloat16)
    return per_expert, routed


def hy3_expert_e2_fused(
    hidden_values: mx.array,
    expert_slots: mx.array,
    route_weights: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
    *,
    candidate: Hy3ExpertE2Candidate,
    diagnostic: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Write `[R,4096]` directly; diagnostics also expose `[R,8,4096]`."""

    if not candidate.supported:
        raise Hy3ExpertE2Ineligible(candidate.unsupported_reason or candidate.name)
    rows, _capacity = _validate_e2_arrays(
        hidden_values,
        expert_slots,
        route_weights,
        down_weight,
        down_scales,
        down_biases,
    )
    threads = candidate.simd_groups * 32
    tasks = rows * (HY3_E2_HIDDEN_SIZE // candidate.output_n_tile)
    kernel = _build_e2_kernel(candidate, diagnostic=diagnostic)
    output_shapes = [(rows, HY3_E2_HIDDEN_SIZE)]
    output_dtypes = [mx.bfloat16]
    if diagnostic:
        output_shapes.append((rows, HY3_E2_TOP_K, HY3_E2_HIDDEN_SIZE))
        output_dtypes.append(mx.bfloat16)
    outputs = kernel(
        inputs=[
            mx.contiguous(hidden_values),
            mx.contiguous(expert_slots),
            mx.contiguous(route_weights),
            down_weight,
            down_scales,
            down_biases,
        ],
        template=[("T", mx.bfloat16)],
        grid=(threads * tasks, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    if diagnostic:
        routed, per_expert = outputs
        return per_expert, routed
    return outputs[0]


def _tensor_error_metrics(observed: mx.array, reference: mx.array) -> dict[str, Any]:
    shape_exact = tuple(observed.shape) == tuple(reference.shape)
    if not shape_exact:
        return {
            "shape_exact": False,
            "dtype": str(observed.dtype),
            "array_equal": False,
            "all_finite": False,
            "max_abs_error": math.inf,
            "normalized_rmse": math.inf,
        }
    observed_f32 = observed.astype(mx.float32)
    reference_f32 = reference.astype(mx.float32)
    difference = observed_f32 - reference_f32
    rmse = float(mx.sqrt(mx.mean(mx.square(difference))).item())
    reference_rms = float(mx.sqrt(mx.mean(mx.square(reference_f32))).item())
    return {
        "shape_exact": True,
        "dtype": str(observed.dtype),
        "array_equal": bool(mx.array_equal(observed, reference).item()),
        "all_finite": bool(
            mx.all(mx.isfinite(observed_f32)).item()
            and mx.all(mx.isfinite(reference_f32)).item()
        ),
        "max_abs_error": float(mx.max(mx.abs(difference)).item()),
        "normalized_rmse": rmse / max(reference_rms, 1e-12),
    }


def hy3_e2_correctness_metrics(
    observed_per_expert: mx.array,
    observed_final: mx.array,
    reference_per_expert: mx.array,
    reference_final: mx.array,
    *,
    activation_mode: str,
) -> dict[str, Any]:
    per_expert = _tensor_error_metrics(observed_per_expert, reference_per_expert)
    final = _tensor_error_metrics(observed_final, reference_final)
    passes = hy3_e2_candidate_passes(
        activation_mode=activation_mode,
        per_expert_shape_exact=bool(per_expert["shape_exact"]),
        final_shape_exact=bool(final["shape_exact"]),
        dtype_bf16=(
            observed_per_expert.dtype == mx.bfloat16
            and observed_final.dtype == mx.bfloat16
        ),
        all_finite=bool(per_expert["all_finite"] and final["all_finite"]),
        per_expert_max_abs_error=float(per_expert["max_abs_error"]),
        final_max_abs_error=float(final["max_abs_error"]),
        final_normalized_rmse=float(final["normalized_rmse"]),
    )
    return {"per_expert": per_expert, "final": final, "passes": passes}
