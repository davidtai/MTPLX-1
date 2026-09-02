"""One-dispatch K/V row gather for the Qwen4 fixed-M4 QSA verifier.

Two output layouts, same gathered values:

``transposed_keys=False`` (default) emits K and V both as
``[1, H_KV, ROWS, SELECTED, HEAD_DIM]`` -- the layout the stock rows-gather
attention takes, which then builds its score operand with
``k_sel.swapaxes(-1, -2).reshape(...)``.  On Metal that view is not a legal
GEMM operand, so MLX materializes it: one 8.4 MB transposed copy per QSA
layer, 12 per verify cycle (the census counts it at qwen4_exp.py:2842).

``transposed_keys=True`` (MTPLX_FABLE_QSA_M4) emits K as
``[1, H_KV, ROWS, HEAD_DIM, SELECTED]`` instead.  The gather is already
writing those bytes, so the transpose is free at the source and the copy
disappears; the caller then needs only ``mx.expand_dims`` to reach the score
operand's shape.  Bit-exact -- pure data movement, no arithmetic.

The transposed write is staged through threadgroup memory so both the read
of ``keys`` (consecutive threads -> consecutive head dims) and the write of
``selected_keys`` (consecutive threads -> consecutive selected tokens) stay
coalesced; the naive spelling would scatter 2-byte stores across a
SELECTED-sized stride.
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

    ``transposed_keys`` selects the ``[.., HEAD_DIM, SELECTED]`` key layout;
    the bound callable carries ``keys_transposed`` so the attention site reads
    the contract off the binding instead of inferring it from a shape.
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
