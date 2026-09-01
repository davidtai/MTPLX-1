"""Expert-major fixed-M4 paired routed gate/up producer for Qwen4 Flash-Next.

``qwen4_m4_routed_glu`` is *lane*-major: its grid is
``(output tiles) x (ROWS * TOP_K)``, so an expert selected by two of the four
verifier rows has its q4 weight tile streamed twice.  The census over the real
workload puts the mean distinct-expert count at 28.07 of 40 per layer-cycle, so
30% of the routed gate/up bytes are re-reads.

This kernel keeps the *same* arithmetic and the *same* grid but re-indexes the
``y`` axis from "lane" to "plan entry": entry ``i`` is a leader when lane ``i``
is the first (or ``MEMBERS``-th, ...) occurrence of ``expert_ids[i]``, and it
produces the outputs for *every* lane in that occurrence run.  Non-leader
entries mask themselves off.  The expert's weight tile is loaded into registers
once per k-block and applied to all of the entry's rows, so the layer streams
one tile per DISTINCT expert instead of one per lane.

Bit-exactness with the lane-major kernel
----------------------------------------
Each output element is still ``simd_sum`` over 32 lanes of a five-term
sequential accumulation of the identical ``qdot_q4`` expression, in the
identical k order, with the identical lane -> data mapping.  Restructuring the
loop nest changes only *which threadgroup* evaluates a given (lane, output)
product and *where its weight bytes came from* (a register instead of a device
load) -- never the order or the shape of the floating-point expression.  The
one residual risk is that the Metal compiler contracts the identical expression
into FMAs differently when its ``ws`` operand is a ``thread ushort4`` rather
than a ``device ushort``; ``scripts/fable/micro_expert_major.py`` measures that
directly and reports differing-element counts against the lane-major kernel.

The plan is recomputed inside every threadgroup rather than passed in.  It is
two 40-iteration scalar scans over the 160 L1-resident bytes of ``expert_ids``,
uniform across the whole threadgroup, and it costs zero extra dispatches --
which matters, because a stock-op plan builder costs ~17 dispatches per layer
(~1.6 ms/cycle at this box's ~2.0 us/dispatch) against a ~2.9 ms/cycle prize.
``scripts/fable/expert_id_patterns.expert_major_plan`` is the MLX-free,
unit-tested statement of what the scan computes.

Why the down projection is not here yet
---------------------------------------
``qwen4_m4_routed_down`` carries a third of the routed bytes and has the same
duplicate-expert waste, but it is not a drop-in the way this is: its threadgroup
is ``(output tile, ROW)`` and it accumulates the row's ten weighted expert
products in registers, in a fixed ``SLOT_ORDER`` that reproduces MLX's BF16
``col_reduce_small`` tree.  An expert-major version cannot hold that reduction,
because one threadgroup no longer owns one row.  The bit-exact restructuring is
a two-pass split: pass 1 goes expert-major and writes the per-(row, slot)
``bfloat(down_value * route_score)`` products -- the exact values the current
kernel already forms -- to a ``[4, 10, 2560]`` bf16 scratch; pass 2 replays the
identical ``SLOT_ORDER`` fold over that scratch and keeps the shared/residual
tail unchanged.  That is bit-exact by construction (the reduce reads the same
bf16 products in the same order) and it trades ~12.3 MB/layer of streamed
weights for 0.41 MB of scratch traffic and one extra dispatch per layer (~2 us
against ~20 us saved).  It is deliberately deferred until the GLU numbers below
are in: it is the same bet, and there is no reason to pay for it twice before
the first one is priced.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


ROWS = 4
TOP_K = 10
SLOTS = ROWS * TOP_K
HIDDEN = 2560
INTERMEDIATE = 640
THREADS = 64
SIMD_SIZE = 32
#: Lanes one leader entry may own.  A top-k row holds distinct experts, so an
#: expert's occurrence run is at most one lane per row.
MEMBERS = ROWS
#: Outputs each simdgroup owns.  Bit-exactness is independent of this (it only
#: retiles the output axis); it trades registers against x-reload amortization,
#: which is why both supported values are benchmarked rather than assumed.
SUPPORTED_OUTPUTS_PER_SIMD = (4, 2)
DEFAULT_OUTPUTS_PER_SIMD = 4

_KERNELS: dict[int, Any] = {}


def _header(outputs_per_simd: int) -> str:
    return f"""
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint ROWS = {ROWS};
    constant constexpr uint TOP_K = {TOP_K};
    constant constexpr uint SLOTS = {SLOTS};
    constant constexpr uint MEMBERS = {MEMBERS};
    constant constexpr uint HIDDEN = {HIDDEN};
    constant constexpr uint INTERMEDIATE = {INTERMEDIATE};
    constant constexpr uint FUSED_OUTPUTS = 2 * INTERMEDIATE;
    constant constexpr uint GROUP_SIZE = 32;
    constant constexpr uint VALUES_PER_THREAD = 16;
    constant constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr uint OUTPUTS_PER_SIMD = {outputs_per_simd};
    constant constexpr uint OUTPUTS_PER_THREADGROUP = 2 * OUTPUTS_PER_SIMD;
    constant constexpr uint WEIGHT_BYTES_PER_ROW = HIDDEN / 2;
    constant constexpr uint GROUPS_PER_ROW = HIDDEN / GROUP_SIZE;

    // Byte-for-byte the lane-major kernel's loader.
    inline float load_q4_vector(
        const device bfloat* x,
        thread float* x_thread) {{
        float sum = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {{
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
        }}
        return sum;
    }}

    // The lane-major ``qdot_q4`` with its packed words already staged in
    // registers.  The expression tree -- masks, the /16 /256 /4096 pre-scaling
    // folded into ``x_thread``, the ``a + b + c + d`` grouping, the running
    // ``accum +=``, and the closing ``scale * accum + sum * bias`` -- is
    // unchanged, and so is the ``for i`` loop the lane-major kernel writes
    // (constant trip count; the compiler unrolls both identically).  Only the
    // address space of ``ws`` moves, device -> thread.
    inline float qdot_q4_staged(
        const thread ushort* ws,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {{
        float accum = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {{
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x000f) +
                 x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                 x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                 x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }}
        return scale * accum + sum * bias;
    }}
"""


_SOURCE = """
    const uint output_tile = threadgroup_position_in_grid.x;
    const uint entry = threadgroup_position_in_grid.y;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint output_base =
        output_tile * OUTPUTS_PER_THREADGROUP
        + simd_group * OUTPUTS_PER_SIMD;

    // --- plan, recomputed per threadgroup; uniform across all its threads ---
    // ``expert_ids`` is 160 bytes and stays in L1 for the whole layer.
    const int selected_expert = int(expert_ids[entry]);
    uint position = 0;
    for (uint j = 0; j < entry; ++j) {
        position += (int(expert_ids[j]) == selected_expert) ? 1u : 0u;
    }
    if (position % MEMBERS != 0) {
        // An earlier leader already owns this occurrence run.  Every thread of
        // this threadgroup takes the branch, so no simd op is left unbalanced.
        return;
    }
    // Every index into ``member`` is a constant after the MEMBERS-trip loops
    // unroll, so it stays in registers.  A ``member[member_count] = ...`` with
    // its dynamic index would sink the array into thread-local memory, which on
    // this GPU is device memory -- the opposite of the point of this kernel.
    int member[MEMBERS];
    uint member_count = 0;
    for (uint slot_index = 0; slot_index < MEMBERS; ++slot_index) {
        member[slot_index] = -1;
    }
    for (uint j = entry; j < SLOTS; ++j) {
        if (int(expert_ids[j]) != selected_expert || member_count >= MEMBERS) {
            continue;
        }
        for (uint slot_index = 0; slot_index < MEMBERS; ++slot_index) {
            member[slot_index] =
                (slot_index == member_count) ? int(j) : member[slot_index];
        }
        member_count += 1;
    }

    // --- addressing: identical to the lane-major kernel, expert-indexed ---
    const uint expert = uint(selected_expert);
    const size_t expert_weight_base =
        (size_t)expert * FUSED_OUTPUTS * WEIGHT_BYTES_PER_ROW;
    const size_t expert_metadata_base =
        (size_t)expert * FUSED_OUTPUTS * GROUPS_PER_ROW;
    const device uchar* gate_weights = (const device uchar*)weights
        + expert_weight_base
        + (size_t)output_base * WEIGHT_BYTES_PER_ROW
        + lane * (VALUES_PER_THREAD / 2);
    const device uchar* up_weights = (const device uchar*)weights
        + expert_weight_base
        + (size_t)(INTERMEDIATE + output_base) * WEIGHT_BYTES_PER_ROW
        + lane * (VALUES_PER_THREAD / 2);
    const device bfloat* gate_scales = scales
        + expert_metadata_base
        + (size_t)output_base * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);
    const device bfloat* gate_biases = biases
        + expert_metadata_base
        + (size_t)output_base * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);
    const device bfloat* up_scales = scales
        + expert_metadata_base
        + (size_t)(INTERMEDIATE + output_base) * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);
    const device bfloat* up_biases = biases
        + expert_metadata_base
        + (size_t)(INTERMEDIATE + output_base) * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);

    float gate_result[MEMBERS][OUTPUTS_PER_SIMD];
    float up_result[MEMBERS][OUTPUTS_PER_SIMD];
    for (uint m = 0; m < MEMBERS; ++m) {
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            gate_result[m][out] = 0.0f;
            up_result[m][out] = 0.0f;
        }
    }

    for (uint k = 0; k < HIDDEN; k += BLOCK_SIZE) {
        // The expert's tile for this k-block, read from device memory ONCE and
        // then applied to every row in the entry.  The reads are the same
        // 2-byte ``ushort`` reads at the same addresses the lane-major kernel
        // issues, so no alignment beyond that kernel's is assumed.
        ushort gate_words[OUTPUTS_PER_SIMD][VALUES_PER_THREAD / 4];
        ushort up_words[OUTPUTS_PER_SIMD][VALUES_PER_THREAD / 4];
        float gate_scale[OUTPUTS_PER_SIMD];
        float gate_bias[OUTPUTS_PER_SIMD];
        float up_scale[OUTPUTS_PER_SIMD];
        float up_bias[OUTPUTS_PER_SIMD];
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            const device ushort* gate_row = (const device ushort*)(
                gate_weights + out * WEIGHT_BYTES_PER_ROW);
            const device ushort* up_row = (const device ushort*)(
                up_weights + out * WEIGHT_BYTES_PER_ROW);
            for (uint word = 0; word < VALUES_PER_THREAD / 4; ++word) {
                gate_words[out][word] = gate_row[word];
                up_words[out][word] = up_row[word];
            }
            gate_scale[out] = float(gate_scales[out * GROUPS_PER_ROW]);
            gate_bias[out] = float(gate_biases[out * GROUPS_PER_ROW]);
            up_scale[out] = float(up_scales[out * GROUPS_PER_ROW]);
            up_bias[out] = float(up_biases[out * GROUPS_PER_ROW]);
        }

        // Predicated rather than ``break``-ed: MEMBERS is a compile-time
        // constant, and an early exit would stop the unroll that keeps
        // ``member``/``gate_result``/``up_result`` in registers.
        for (uint m = 0; m < MEMBERS; ++m) {
            if (m < member_count) {
                const uint row = uint(member[m]) / TOP_K;
                const device bfloat* x =
                    value + row * HIDDEN + lane * VALUES_PER_THREAD + k;
                float x_thread[VALUES_PER_THREAD];
                const float sum = load_q4_vector(x, x_thread);
                for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                    gate_result[m][out] += qdot_q4_staged(
                        gate_words[out],
                        x_thread,
                        gate_scale[out],
                        gate_bias[out],
                        sum);
                    up_result[m][out] += qdot_q4_staged(
                        up_words[out],
                        x_thread,
                        up_scale[out],
                        up_bias[out],
                        sum);
                }
            }
        }

        gate_weights += BLOCK_SIZE / 2;
        up_weights += BLOCK_SIZE / 2;
        gate_scales += BLOCK_SIZE / GROUP_SIZE;
        gate_biases += BLOCK_SIZE / GROUP_SIZE;
        up_scales += BLOCK_SIZE / GROUP_SIZE;
        up_biases += BLOCK_SIZE / GROUP_SIZE;
    }

    // ``member_count`` is threadgroup-uniform, so every lane of the simdgroup
    // agrees on which iterations run the simd_sum -- the reduction is never
    // entered by a partial simdgroup.
    for (uint m = 0; m < MEMBERS; ++m) {
        if (m < member_count) {
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                gate_result[m][out] = simd_sum(gate_result[m][out]);
                up_result[m][out] = simd_sum(up_result[m][out]);
            }
            if (lane == 0) {
                const uint slot = uint(member[m]);
                for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                    bfloat gate_value = bfloat(gate_result[m][out]);
                    bfloat up_value = bfloat(up_result[m][out]);
                    auto sigmoid_y =
                        1 / (1 + metal::exp(metal::abs(gate_value)));
                    bfloat sigmoid_value = gate_value < bfloat(0.0f)
                        ? bfloat(sigmoid_y) : bfloat(1 - sigmoid_y);
                    bfloat silu = bfloat(gate_value * sigmoid_value);
                    routed_h[slot * INTERMEDIATE + output_base + out] =
                        bfloat(silu * up_value);
                }
            }
        }
    }
"""


def _check_outputs_per_simd(outputs_per_simd: int) -> int:
    if outputs_per_simd not in SUPPORTED_OUTPUTS_PER_SIMD:
        raise ValueError(
            "qwen4 M4 expert-major GLU supports outputs_per_simd in "
            f"{SUPPORTED_OUTPUTS_PER_SIMD}, got {outputs_per_simd!r}"
        )
    outputs_per_threadgroup = 2 * outputs_per_simd
    if INTERMEDIATE % outputs_per_threadgroup:
        raise ValueError(
            f"INTERMEDIATE={INTERMEDIATE} is not a multiple of "
            f"{outputs_per_threadgroup}"
        )
    if THREADS != 2 * SIMD_SIZE:
        raise ValueError("expert-major GLU assumes two 32-lane simdgroups")
    return outputs_per_simd


def source(outputs_per_simd: int = DEFAULT_OUTPUTS_PER_SIMD) -> str:
    """Return the expert-major q4 QMV and BF16 GLU source."""

    return _header(_check_outputs_per_simd(outputs_per_simd)) + _SOURCE


def launch_geometry(
    outputs_per_simd: int = DEFAULT_OUTPUTS_PER_SIMD,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the static geometry: output tiles x SLOTS plan entries.

    ``SLOTS`` -- not the distinct-expert count -- is the ``y`` extent, so the
    compiled graph's shape never depends on the routing decision.  The entries
    a leader does not own return before touching memory.
    """

    _check_outputs_per_simd(outputs_per_simd)
    outputs_per_threadgroup = 2 * outputs_per_simd
    return (
        (INTERMEDIATE // outputs_per_threadgroup * THREADS, SLOTS, 1),
        (THREADS, 1, 1),
    )


def bind(
    outputs_per_simd: int = DEFAULT_OUTPUTS_PER_SIMD,
) -> Callable[..., mx.array]:
    """Bind the expert-major routed gate/up producer.

    The returned callable is signature-compatible with
    ``qwen4_m4_routed_glu.bind()``: ``(value, weights, scales, biases,
    expert_ids) -> [ROWS, TOP_K, INTERMEDIATE]`` bf16.
    """

    outputs_per_simd = _check_outputs_per_simd(outputs_per_simd)
    kernel = _KERNELS.get(outputs_per_simd)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_qwen4_m4_expert_major_glu_o{outputs_per_simd}",
            input_names=[
                "value",
                "weights",
                "scales",
                "biases",
                "expert_ids",
            ],
            output_names=["routed_h"],
            header=_header(outputs_per_simd),
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNELS[outputs_per_simd] = kernel
    grid, threadgroup = launch_geometry(outputs_per_simd)

    def routed_glu(value, weights, scales, biases, expert_ids):
        (routed_h,) = kernel(
            inputs=[value, weights, scales, biases, expert_ids],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ROWS, TOP_K, INTERMEDIATE)],
            output_dtypes=[mx.bfloat16],
        )
        return routed_h

    return routed_glu


__all__ = [
    "DEFAULT_OUTPUTS_PER_SIMD",
    "MEMBERS",
    "SUPPORTED_OUTPUTS_PER_SIMD",
    "bind",
    "launch_geometry",
    "source",
]
