"""Shape-specific fused E1 probe for streamed Hy3 affine-Q2 experts.

E1 is deliberately limited to token-grouped gate + up + SwiGLU. The down
projection (E2) remains outside this module so its cost and correctness stay
independently attributable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any

import mlx.core as mx


HY3_E1_ROWS = tuple(range(1, 9))
HY3_E1_TOP_K = 8
HY3_E1_HIDDEN_SIZE = 4096
HY3_E1_INTERMEDIATE_SIZE = 1536
HY3_E1_BITS = 2
HY3_E1_GROUP_SIZE = 64
HY3_E1_PACKED_VALUES_PER_WORD = 16
HY3_E1_PACKED_VALUES_PER_VECTOR = 32

HY3_E1_OUTPUT_N_TILES = (32, 64, 96, 128)
HY3_E1_K_TILES = (32, 64, 128, 256)
HY3_E1_SIMD_GROUPS = (2, 4, 6, 8)
HY3_E1_SILU_MODES = ("exact", "fast")

HY3_E1_EXACT_MAX_ABS_ERROR = 0.5
HY3_E1_EXACT_MAX_NORMALIZED_RMSE = 0.02


class Hy3ExpertE1Ineligible(ValueError):
    """Raised before dispatch when an input is outside the isolated E1 lane."""


@dataclass(frozen=True, slots=True)
class Hy3ExpertE1Candidate:
    """One independently attributable E1 kernel configuration."""

    output_n_tile: int
    k_tile: int
    simd_groups: int
    silu_mode: str = "exact"

    def __post_init__(self) -> None:
        if self.output_n_tile not in HY3_E1_OUTPUT_N_TILES:
            raise ValueError("E1 output-N tile must be 32, 64, 96, or 128")
        if self.k_tile not in HY3_E1_K_TILES:
            raise ValueError("E1 K tile must be 32, 64, 128, or 256")
        if self.simd_groups not in HY3_E1_SIMD_GROUPS:
            raise ValueError("E1 SIMDgroup count must be 2, 4, 6, or 8")
        if self.silu_mode not in HY3_E1_SILU_MODES:
            raise ValueError("E1 SiLU mode must be exact or fast")

    @property
    def name(self) -> str:
        return (
            f"n{self.output_n_tile}_k{self.k_tile}_"
            f"sg{self.simd_groups}_{self.silu_mode}"
        )

    @property
    def unsupported_reason(self) -> str | None:
        if self.output_n_tile % self.simd_groups:
            return "output-N tile must divide evenly across SIMDgroups"
        outputs_per_simdgroup = self.output_n_tile // self.simd_groups
        if not 4 <= outputs_per_simdgroup <= 16:
            return "outputs per SIMDgroup must stay between 4 and 16"
        if HY3_E1_INTERMEDIATE_SIZE % self.output_n_tile:
            return "output-N tile must divide the Hy3 intermediate width"
        if HY3_E1_HIDDEN_SIZE % self.k_tile:
            return "K tile must divide the Hy3 hidden width"
        if self.simd_groups * 32 > 256:
            return "threadgroup exceeds the 256-thread resource boundary"
        if self.k_tile * 2 > 32 * 1024:
            return "threadgroup activation tile exceeds 32 KiB"
        return None

    @property
    def supported(self) -> bool:
        return self.unsupported_reason is None

    @property
    def promotion_eligible(self) -> bool:
        return self.supported and self.silu_mode == "exact"

    @property
    def classification(self) -> str:
        if not self.supported:
            return "unsupported_geometry"
        if self.silu_mode == "fast":
            return "fast_silu_ablation_only"
        return "exact_e1_candidate"

    def resources(self) -> dict[str, Any]:
        outputs_per_simdgroup = (
            self.output_n_tile // self.simd_groups
            if self.output_n_tile % self.simd_groups == 0
            else 0
        )
        return {
            "threads_per_threadgroup": self.simd_groups * 32,
            "outputs_per_simdgroup": outputs_per_simdgroup,
            "threadgroup_activation_bytes": self.k_tile * 2,
            "accumulator_floats_per_lane": outputs_per_simdgroup * 2,
            "packed_weight_vector": "uint2",
            "packed_values_per_vector": HY3_E1_PACKED_VALUES_PER_VECTOR,
            "activation_reuse": "threadgroup gate+up/output-column reuse",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_n_tile": self.output_n_tile,
            "k_tile": self.k_tile,
            "simd_groups": self.simd_groups,
            "silu_mode": self.silu_mode,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "classification": self.classification,
            "promotion_eligible": self.promotion_eligible,
            "resources": self.resources(),
        }


def hy3_e1_candidate_catalog() -> tuple[Hy3ExpertE1Candidate, ...]:
    """Return the full requested cross-product, including illegal geometry."""

    return tuple(
        Hy3ExpertE1Candidate(n_tile, k_tile, simd_groups, silu_mode)
        for silu_mode, n_tile, k_tile, simd_groups in product(
            HY3_E1_SILU_MODES,
            HY3_E1_OUTPUT_N_TILES,
            HY3_E1_K_TILES,
            HY3_E1_SIMD_GROUPS,
        )
    )


def hy3_e1_supported_candidates(
    *,
    silu_modes: tuple[str, ...] = HY3_E1_SILU_MODES,
) -> tuple[Hy3ExpertE1Candidate, ...]:
    """Retain every legal candidate; exact candidates remain ordered first."""

    unknown = set(silu_modes) - set(HY3_E1_SILU_MODES)
    if unknown:
        raise ValueError(f"unknown E1 SiLU modes: {sorted(unknown)}")
    return tuple(
        candidate
        for candidate in hy3_e1_candidate_catalog()
        if candidate.supported and candidate.silu_mode in silu_modes
    )


def hy3_e1_component_bank_shapes(capacity: int) -> dict[str, tuple[int, ...]]:
    """Describe the preserved component-major affine-Q2 bank contract."""

    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("E1 component-bank capacity must be positive")
    weight_shape = (
        capacity,
        HY3_E1_INTERMEDIATE_SIZE,
        HY3_E1_HIDDEN_SIZE * HY3_E1_BITS // 32,
    )
    parameter_shape = (
        capacity,
        HY3_E1_INTERMEDIATE_SIZE,
        HY3_E1_HIDDEN_SIZE // HY3_E1_GROUP_SIZE,
    )
    return {
        "gate_proj.weight": weight_shape,
        "gate_proj.scales": parameter_shape,
        "gate_proj.biases": parameter_shape,
        "up_proj.weight": weight_shape,
        "up_proj.scales": parameter_shape,
        "up_proj.biases": parameter_shape,
    }


def hy3_e1_candidate_passes(
    *,
    silu_mode: str,
    output_shape_exact: bool,
    output_dtype_bf16: bool,
    all_finite: bool,
    max_abs_error: float,
    normalized_rmse: float,
    max_abs_tolerance: float = HY3_E1_EXACT_MAX_ABS_ERROR,
    normalized_rmse_tolerance: float = HY3_E1_EXACT_MAX_NORMALIZED_RMSE,
) -> bool:
    """Apply the correctness-first BF16 boundary; fast SiLU never promotes."""

    if silu_mode not in HY3_E1_SILU_MODES:
        raise ValueError("E1 SiLU mode must be exact or fast")
    max_abs_error = float(max_abs_error)
    normalized_rmse = float(normalized_rmse)
    return (
        silu_mode == "exact"
        and bool(output_shape_exact)
        and bool(output_dtype_bf16)
        and bool(all_finite)
        and math.isfinite(max_abs_error)
        and math.isfinite(normalized_rmse)
        and 0.0 <= max_abs_error <= float(max_abs_tolerance)
        and 0.0 <= normalized_rmse <= float(normalized_rmse_tolerance)
    )


def render_hy3_expert_fused_e1_source(
    candidate: Hy3ExpertE1Candidate,
) -> str:
    """Render one Q2 kernel body without compiling or dispatching it."""

    if not candidate.supported:
        raise Hy3ExpertE1Ineligible(candidate.unsupported_reason or candidate.name)
    exp_call = (
        "metal::exp(metal::abs(gate_value))"
        if candidate.silu_mode == "exact"
        else "fast::exp(metal::abs(gate_value))"
    )
    return f"""
        using namespace metal;

        constexpr int K = {HY3_E1_HIDDEN_SIZE};
        constexpr int N = {HY3_E1_INTERMEDIATE_SIZE};
        constexpr int TOP_K = {HY3_E1_TOP_K};
        constexpr int GROUP_SIZE = {HY3_E1_GROUP_SIZE};
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
        uint assignment = task / N_TILES;
        uint n_tile_index = task - assignment * N_TILES;
        uint token = assignment / TOP_K;
        int selected_slot = expert_slots[assignment];
        int n_base = int(n_tile_index) * N_TILE
            + int(simd_gid) * OUTPUTS_PER_SIMDGROUP;

        threadgroup T activation_tile[K_TILE];
        float gate_acc[OUTPUTS_PER_SIMDGROUP];
        float up_acc[OUTPUTS_PER_SIMDGROUP];
        _Pragma("unroll")
        for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
            gate_acc[output] = 0.0f;
            up_acc[output] = 0.0f;
        }}

        for (int k0 = 0; k0 < K; k0 += K_TILE) {{
            for (int local_k = int(tid); local_k < K_TILE;
                 local_k += THREADS) {{
                activation_tile[local_k] = token_rows[int(token) * K + k0 + local_k];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int vector_k = int(lane) * VALUES_PER_VECTOR;
            bool active_vector = vector_k < K_TILE;
            _Pragma("unroll")
            for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
                float gate_partial = 0.0f;
                float up_partial = 0.0f;
                if (active_vector) {{
                    int n = n_base + output;
                    int global_k = k0 + vector_k;
                    int packed_offset =
                        (selected_slot * N + n) * PACKED_WORDS_PER_ROW
                        + global_k / VALUES_PER_WORD;
                    uint2 gate_packed = *((const device uint2*)(
                        gate_weight + packed_offset));
                    uint2 up_packed = *((const device uint2*)(
                        up_weight + packed_offset));
                    int parameter_offset =
                        (selected_slot * N + n) * GROUPS_PER_ROW
                        + global_k / GROUP_SIZE;
                    float gate_scale = float(gate_scales[parameter_offset]);
                    float gate_bias = float(gate_biases[parameter_offset]);
                    float up_scale = float(up_scales[parameter_offset]);
                    float up_bias = float(up_biases[parameter_offset]);
                    _Pragma("unroll")
                    for (int packed_index = 0; packed_index < 2; ++packed_index) {{
                        uint gate_word = packed_index == 0
                            ? gate_packed.x : gate_packed.y;
                        uint up_word = packed_index == 0
                            ? up_packed.x : up_packed.y;
                        _Pragma("unroll")
                        for (int value_index = 0; value_index < 16; ++value_index) {{
                            int local_value = packed_index * 16 + value_index;
                            float activation = float(
                                activation_tile[vector_k + local_value]);
                            float gate_weight_value =
                                float((gate_word >> (value_index * 2)) & 0x3u)
                                * gate_scale + gate_bias;
                            float up_weight_value =
                                float((up_word >> (value_index * 2)) & 0x3u)
                                * up_scale + up_bias;
                            gate_partial += activation * gate_weight_value;
                            up_partial += activation * up_weight_value;
                        }}
                    }}
                }}
                gate_acc[output] += simd_sum(gate_partial);
                up_acc[output] += simd_sum(up_partial);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (lane == 0) {{
            _Pragma("unroll")
            for (int output = 0; output < OUTPUTS_PER_SIMDGROUP; ++output) {{
                int n = n_base + output;
                T gate_value = T(gate_acc[output]);
                T up_value = T(up_acc[output]);
                T sigmoid_base = T(1) / (T(1) + T({exp_call}));
                T sigmoid_value = gate_value < T(0)
                    ? sigmoid_base : T(1) - sigmoid_base;
                T silu_value = T(gate_value * sigmoid_value);
                output_values[assignment * N + n] = T(silu_value * up_value);
            }}
        }}
    """


_KERNEL_CACHE: dict[Hy3ExpertE1Candidate, Any] = {}


def build_hy3_expert_fused_e1_kernel(candidate: Hy3ExpertE1Candidate) -> Any:
    """Construct one lazy Metal kernel object; no dispatch occurs here."""

    if not candidate.supported:
        raise Hy3ExpertE1Ineligible(candidate.unsupported_reason or candidate.name)
    cached = _KERNEL_CACHE.get(candidate)
    if cached is not None:
        return cached
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_hy3_q2_e1_{candidate.name}",
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
        output_names=["output_values"],
        source=render_hy3_expert_fused_e1_source(candidate),
    )
    _KERNEL_CACHE[candidate] = kernel
    return kernel


def _validate_e1_arrays(
    token_rows: mx.array,
    expert_slots: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
) -> tuple[int, int]:
    if (
        token_rows.ndim != 2
        or int(token_rows.shape[0]) not in HY3_E1_ROWS
        or int(token_rows.shape[1]) != HY3_E1_HIDDEN_SIZE
        or token_rows.dtype != mx.bfloat16
    ):
        raise Hy3ExpertE1Ineligible("E1 token rows require BF16 shape [R1..R8, 4096]")
    rows = int(token_rows.shape[0])
    if (
        expert_slots.ndim != 2
        or tuple(int(value) for value in expert_slots.shape) != (rows, HY3_E1_TOP_K)
        or expert_slots.dtype != mx.int32
    ):
        raise Hy3ExpertE1Ineligible("E1 expert slots require int32 shape [R, 8]")
    capacity = int(gate_weight.shape[0]) if gate_weight.ndim == 3 else 0
    expected = hy3_e1_component_bank_shapes(capacity) if capacity > 0 else {}
    arrays = {
        "gate_proj.weight": gate_weight,
        "gate_proj.scales": gate_scales,
        "gate_proj.biases": gate_biases,
        "up_proj.weight": up_weight,
        "up_proj.scales": up_scales,
        "up_proj.biases": up_biases,
    }
    for name, value in arrays.items():
        wanted_dtype = mx.uint32 if name.endswith(".weight") else mx.bfloat16
        if value.dtype != wanted_dtype or tuple(
            int(dimension) for dimension in value.shape
        ) != expected.get(name):
            raise Hy3ExpertE1Ineligible(
                f"E1 component bank {name} has the wrong shape or dtype"
            )
    return rows, capacity


def hy3_expert_fused_e1(
    token_rows: mx.array,
    expert_slots: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    *,
    candidate: Hy3ExpertE1Candidate,
) -> mx.array:
    """Run token-grouped Q2 gate+up+SwiGLU and return BF16 `[R,8,1536]`."""

    if not candidate.supported:
        raise Hy3ExpertE1Ineligible(candidate.unsupported_reason or candidate.name)
    rows, _capacity = _validate_e1_arrays(
        token_rows,
        expert_slots,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    )
    assignments = rows * HY3_E1_TOP_K
    n_tiles = HY3_E1_INTERMEDIATE_SIZE // candidate.output_n_tile
    threads = candidate.simd_groups * 32
    kernel = build_hy3_expert_fused_e1_kernel(candidate)
    (output,) = kernel(
        inputs=[
            mx.contiguous(token_rows),
            mx.contiguous(expert_slots.reshape(assignments)),
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
        ],
        template=[("T", mx.bfloat16)],
        grid=(threads * assignments * n_tiles, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(assignments, HY3_E1_INTERMEDIATE_SIZE)],
        output_dtypes=[mx.bfloat16],
    )
    return output.reshape(rows, HY3_E1_TOP_K, HY3_E1_INTERMEDIATE_SIZE)


def hy3_expert_fused_e1_reference(
    token_rows: mx.array,
    expert_slots: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
) -> mx.array:
    """Run the preserved component-bank gather-QMM E1 reference."""

    from mlx_lm.models.activations import swiglu

    rows, _capacity = _validate_e1_arrays(
        token_rows,
        expert_slots,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
    )
    assignments = rows * HY3_E1_TOP_K
    selected = mx.broadcast_to(
        token_rows[:, None, :],
        (rows, HY3_E1_TOP_K, HY3_E1_HIDDEN_SIZE),
    ).reshape(assignments, 1, 1, HY3_E1_HIDDEN_SIZE)
    indices = expert_slots.reshape(assignments, 1)

    def qmm(
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ) -> mx.array:
        return mx.gather_qmm(
            selected,
            weight,
            scales,
            biases,
            rhs_indices=indices,
            transpose=True,
            group_size=HY3_E1_GROUP_SIZE,
            bits=HY3_E1_BITS,
            mode="affine",
        )

    gate = qmm(gate_weight, gate_scales, gate_biases)
    up = qmm(up_weight, up_scales, up_biases)
    output = swiglu(gate, up).astype(mx.bfloat16)
    return output.reshape(rows, HY3_E1_TOP_K, HY3_E1_INTERMEDIATE_SIZE)


def hy3_e1_correctness_metrics(
    observed: mx.array,
    reference: mx.array,
    *,
    silu_mode: str,
) -> dict[str, Any]:
    """Measure the exact BF16 E1 boundary without conflating fast SiLU."""

    output_shape_exact = tuple(observed.shape) == tuple(reference.shape)
    output_dtype_bf16 = observed.dtype == mx.bfloat16
    if not output_shape_exact:
        return {
            "output_shape_exact": False,
            "output_dtype_bf16": output_dtype_bf16,
            "all_finite": False,
            "max_abs_error": math.inf,
            "normalized_rmse": math.inf,
            "passes": False,
        }
    observed_f32 = observed.astype(mx.float32)
    reference_f32 = reference.astype(mx.float32)
    difference = observed_f32 - reference_f32
    max_abs_error = float(mx.max(mx.abs(difference)).item())
    rmse = float(mx.sqrt(mx.mean(mx.square(difference))).item())
    reference_rms = float(mx.sqrt(mx.mean(mx.square(reference_f32))).item())
    normalized_rmse = rmse / max(reference_rms, 1e-12)
    all_finite = bool(
        mx.all(mx.isfinite(observed_f32)).item()
        and mx.all(mx.isfinite(reference_f32)).item()
    )
    return {
        "output_shape_exact": output_shape_exact,
        "output_dtype_bf16": output_dtype_bf16,
        "all_finite": all_finite,
        "max_abs_error": max_abs_error,
        "normalized_rmse": normalized_rmse,
        "passes": hy3_e1_candidate_passes(
            silu_mode=silu_mode,
            output_shape_exact=output_shape_exact,
            output_dtype_bf16=output_dtype_bf16,
            all_finite=all_finite,
            max_abs_error=max_abs_error,
            normalized_rmse=normalized_rmse,
        ),
    }
