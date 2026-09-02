"""One-dispatch K/V row gather for the Qwen4 fixed-M4 QSA verifier.

Two output layouts, same gathered VALUES:

``transposed_keys=False`` (default, and the one that ships) emits K and V both
as ``[1, H_KV, ROWS, SELECTED, HEAD_DIM]`` -- the layout the stock rows-gather
attention takes, which then builds its score operand with
``k_sel.swapaxes(-1, -2).reshape(...)``.

``transposed_keys=True`` (``MTPLX_FABLE_QSA_M4_KT``, default OFF) emits K as
``[1, H_KV, ROWS, HEAD_DIM, SELECTED]`` instead, so the caller reaches the
score operand with ``mx.expand_dims`` and the transposed view is never
materialized.

FALSIFIED, AND KEPT ONLY AS ITS OWN A/B ARM
-------------------------------------------
The transposed layout was the byte-side idea of the QSA M4 lane: the census
attributes an 8.4 MB transposed copy per QSA layer to ``qwen4_exp.py:2842``,
101 MB per verify cycle, and this gather is writing those bytes anyway. The
GPU microbench (2026-09-01, compiled lane, 12 QSA layers,
scripts/fable/micro_qsa_m4.py) falsified it on BOTH axes, while the lane's
other four rewrites passed on both:

    gather_stock  1.501 ms      gather_kt  1.657 ms  (+10%)
    104 differing elements, max abs 0.125

so it is gated separately from ``MTPLX_FABLE_QSA_M4``, which alone is the
bit-exact, uniformly faster set.

WHY IT IS NOT BIT-EXACT (the gather itself is; the GEMM after it is not)
------------------------------------------------------------------------
This kernel moves data and does no arithmetic, and the two layouts hold the
same bf16 bits -- pinned on the CPU stream in tests/test_fable_qsa_m4.py. The
difference appears one op later, in the score GEMM that consumes K.

MLX does not have one matmul: it selects a kernel and a tiling from the
operand's layout. A ``swapaxes`` view of a contiguous ``[.., S, K, D]`` array
reaches the GEMM as a B operand with the transpose flag set, and MLX makes it
contiguous (that is the copy the census sees) before running its
``no-transpose`` path over the 256-element contraction. A natively
``[.., S, D, K]`` array is already contiguous and takes the transposed-B path
directly. Both accumulate in fp32; they split and order the K dimension
differently, so the fp32 partial sums are reassociated. Reassociation of 256
products is a few fp32 ulp, which usually rounds to the same bf16 and
sometimes does not -- 104 of 4.2 M score elements here, and max abs 0.125 is
exactly ONE bf16 ulp at a score magnitude in [16, 32), which is the signature
of that class rather than of a data error.

Those scores feed the softmax and the PV product, so this is a rounding-class
change to attention output on the same terms as
kernels/qwen4_m4_hyper_read.py: it would have to be adopted on acceptance
parity, not on a digest. It is not worth that when it also LOSES 10%.

WHY IT IS SLOWER THAN THE COPY IT REMOVES
------------------------------------------
The untransposed gather is a pure streaming copy: one ``vec<T,4>`` load and
one ``vec<T,4>`` store per thread, both 8-byte aligned and fully coalesced, no
threadgroup memory and no barrier. The transposed one cannot vectorize either
side -- a transpose needs the element that is contiguous on the read to be
strided on the write -- so it drops to scalar 2-byte accesses staged through a
32x32 threadgroup tile plus a barrier: 4x the memory instructions for the same
bytes, plus the barrier and the tile traffic. It does not delete the
transpose, it relocates it from a specialized MLX copy kernel into a hand tile
that is worse at it. Same shape of result as ``bank_select`` in
scripts/fable/micro_opdiet.py: fewer dispatches, more time.

The tiled staging is still the right spelling for a transposed gather (the
naive version scatters 2-byte stores across a SELECTED-sized stride and is far
worse); the finding is that a transposed gather is the wrong thing to want
here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import mlx.core as mx


_H_KV = 2
_ROWS = 4
_SELECTED = 2052
_HEAD_DIM = 256
_VECTOR_WIDTH = 4
_VECTORS_PER_HEAD = _HEAD_DIM // _VECTOR_WIDTH
_ELEMENTS = _H_KV * _ROWS * _SELECTED * _VECTORS_PER_HEAD
_OUTPUT_SHAPE = (1, _H_KV, _ROWS, _SELECTED, _HEAD_DIM)
_KEYS_T_SHAPE = (1, _H_KV, _ROWS, _HEAD_DIM, _SELECTED)

_TILE = 32
#: Threadgroup rows.  A 32x32 tile with one thread per element would need
#: 1,024 threads, which a kernel is only granted if its register pressure
#: allows; 32x8 with a 4-step loop covers the same tile at the 256-thread
#: width the untransposed gather already uses.
_TILE_ROWS = 8
_TILE_STEPS = _TILE // _TILE_ROWS
_TOKEN_TILES = (_SELECTED + _TILE - 1) // _TILE
_DIM_TILES = _HEAD_DIM // _TILE


@lru_cache(maxsize=None)
def _fused_kv_gather_kernel(capacity: int):
    source = f"""
        constexpr uint H_KV = {_H_KV};
        constexpr uint ROWS = {_ROWS};
        constexpr uint SELECTED = {_SELECTED};
        constexpr uint HEAD_DIM = {_HEAD_DIM};
        constexpr uint VECTORS_PER_HEAD = {_VECTORS_PER_HEAD};
        constexpr uint CAPACITY = {capacity};

        uint output = thread_position_in_grid.x;
        if (output >= H_KV * ROWS * SELECTED * VECTORS_PER_HEAD) return;
        uint dimension = output % VECTORS_PER_HEAD;
        uint selected_offset = (output / VECTORS_PER_HEAD) % (ROWS * SELECTED);
        uint head = output / (VECTORS_PER_HEAD * ROWS * SELECTED);
        uint token = uint(token_idx[selected_offset]);
        uint input = (head * CAPACITY + token) * VECTORS_PER_HEAD + dimension;
        const device vec<T, 4>* keys4 =
            reinterpret_cast<const device vec<T, 4>*>(keys);
        const device vec<T, 4>* values4 =
            reinterpret_cast<const device vec<T, 4>*>(values);
        device vec<T, 4>* selected_keys4 =
            reinterpret_cast<device vec<T, 4>*>(selected_keys);
        device vec<T, 4>* selected_values4 =
            reinterpret_cast<device vec<T, 4>*>(selected_values);
        selected_keys4[output] = keys4[input];
        selected_values4[output] = values4[input];
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen4_qsa_m4_fused_kv_gather_c{capacity}",
        input_names=["keys", "values", "token_idx"],
        output_names=["selected_keys", "selected_values"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _fused_kv_gather_kernel_kt(capacity: int):
    """Same gather, K emitted [.., HEAD_DIM, SELECTED] via a 32x32 tile."""

    source = f"""
        constexpr uint ROWS = {_ROWS};
        constexpr uint SELECTED = {_SELECTED};
        constexpr uint HEAD_DIM = {_HEAD_DIM};
        constexpr uint CAPACITY = {capacity};
        constexpr uint TILE = {_TILE};
        constexpr uint TILE_ROWS = {_TILE_ROWS};
        constexpr uint TILE_STEPS = {_TILE_STEPS};
        constexpr uint DIM_TILES = {_DIM_TILES};

        // +1 column removes the 32-way bank conflict on the transposed read.
        threadgroup T tile[TILE][TILE + 1];

        const uint tx = thread_position_in_threadgroup.x;   // 0..TILE-1
        const uint ty = thread_position_in_threadgroup.y;   // 0..TILE_ROWS-1
        const uint dim_tile = threadgroup_position_in_grid.x % DIM_TILES;
        const uint token_tile = threadgroup_position_in_grid.y;
        const uint pair = threadgroup_position_in_grid.z;   // head * ROWS + row
        const uint head = pair / ROWS;
        const uint row = pair % ROWS;

        const uint d0 = dim_tile * TILE;
        const uint t0 = token_tile * TILE;

        // Load: consecutive tx -> consecutive head dims (coalesced). V is
        // written in the same pass with the same mapping, so the gathered
        // token row is read once and serves both outputs.
        for (uint step = 0; step < TILE_STEPS; ++step) {{
            const uint ly = ty + step * TILE_ROWS;
            const uint t_load = t0 + ly;
            T loaded = T(0);
            if (t_load < SELECTED) {{
                const uint token = uint(token_idx[row * SELECTED + t_load]);
                const uint src = (head * CAPACITY + token) * HEAD_DIM + d0 + tx;
                loaded = keys[src];
                const uint out_v =
                    ((head * ROWS + row) * SELECTED + t_load) * HEAD_DIM
                    + d0 + tx;
                selected_values[out_v] = values[src];
            }}
            tile[ly][tx] = loaded;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store: consecutive tx -> consecutive selected tokens (coalesced).
        // tile[tx][dy] holds the value the ty==tx thread loaded for this
        // token, so the guard below is the predicate that filled it.
        const uint t_store = t0 + tx;
        if (t_store < SELECTED) {{
            for (uint step = 0; step < TILE_STEPS; ++step) {{
                const uint dy = ty + step * TILE_ROWS;
                const uint out_k =
                    ((head * ROWS + row) * HEAD_DIM + d0 + dy) * SELECTED
                    + t_store;
                selected_keys[out_k] = tile[tx][dy];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen4_qsa_m4_fused_kv_gather_kt_c{capacity}",
        input_names=["keys", "values", "token_idx"],
        output_names=["selected_keys", "selected_values"],
        source=source,
        ensure_row_contiguous=True,
    )


def bind_qwen4_qsa_m4_fused_kv_gather(
    *, capacity: int, transposed_keys: bool = False
) -> Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Bind the exact 1K/16K fixed-M4 gather geometry once at cache install.

    ``transposed_keys`` selects the ``[.., HEAD_DIM, SELECTED]`` key layout
    (``MTPLX_FABLE_QSA_M4_KT``, off by default -- see the module docstring for
    the measured verdict). The bound callable carries ``keys_transposed`` so
    the attention site reads the contract off the binding instead of inferring
    it from a shape.
    """

    capacity = int(capacity)
    if capacity < 16_384 or capacity % 4:
        raise ValueError(
            "Qwen4 fixed-M4 fused K/V gather requires a 16K context cache "
            "whose capacity is divisible by the QSA ratio"
        )
    if not transposed_keys:
        kernel = _fused_kv_gather_kernel(capacity)

        def gather(
            keys: mx.array, values: mx.array, token_idx: mx.array
        ) -> tuple[mx.array, mx.array]:
            return kernel(
                inputs=[keys, values, token_idx],
                template=[("T", mx.bfloat16)],
                grid=(_ELEMENTS, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[_OUTPUT_SHAPE, _OUTPUT_SHAPE],
                output_dtypes=[mx.bfloat16, mx.bfloat16],
            )

        gather.keys_transposed = False
        return gather

    kernel_kt = _fused_kv_gather_kernel_kt(capacity)

    def gather_kt(
        keys: mx.array, values: mx.array, token_idx: mx.array
    ) -> tuple[mx.array, mx.array]:
        return kernel_kt(
            inputs=[keys, values, token_idx],
            template=[("T", mx.bfloat16)],
            grid=(_DIM_TILES * _TILE, _TOKEN_TILES * _TILE_ROWS, _H_KV * _ROWS),
            threadgroup=(_TILE, _TILE_ROWS, 1),
            output_shapes=[_KEYS_T_SHAPE, _OUTPUT_SHAPE],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )

    gather_kt.keys_transposed = True
    return gather_kt
