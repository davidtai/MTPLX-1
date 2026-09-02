"""Construction-bound physical-M4 stock-QMM combine route for Flash-Next."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .fable_expert_census import census as _census
from .kernels.qwen4_m4_stage3 import bind
from .kernels.qwen4_m4_expert_major_glu import (
    bind as bind_expert_major_routed_glu,
)
from .kernels.qwen4_m4_routed_down import (
    bind as bind_routed_down_reduce,
    bind_residual_tail,
)
from .kernels.qwen4_m4_routed_glu import bind as bind_routed_glu
from .kernels import qwen4_m4_route as _route_kernel
from .kernels import qwen4_m4_shared_lane as _shared_lane
from .models.qwen4_exp import (
    DecoderLayer,
    SparseMoeBlock,
    _FusedGateUpMLP,
    _FusedGateUpSwitchGLU,
    _hyper_residual_write,
)
from .runtime_options import env_bool


logger = logging.getLogger(__name__)

ROWS = 4
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640

FABLE_MOE_SORTED_ENV = "MTPLX_FABLE_MOE_SORTED"

_MOE_SORTED_CACHE: bool | None = None


def fable_moe_sorted_enabled() -> bool:
    """Return the ``MTPLX_FABLE_MOE_SORTED`` gate; read once, default off.

    Resolved lazily rather than at import so a serving profile that arms the
    flag after this module is imported is still observed, then memoized so the
    decode cycle never repeats the environment lookup.  The gate is a pure
    permutation of the routed gathers (see ``_routed_gather_plan``), so it can
    be flipped without changing any numeric contract.
    """

    global _MOE_SORTED_CACHE
    if _MOE_SORTED_CACHE is None:
        _MOE_SORTED_CACHE = env_bool(
            FABLE_MOE_SORTED_ENV,
            default=False,
            env=os.environ,
        )
    return _MOE_SORTED_CACHE


def reset_fable_moe_sorted_cache() -> None:
    """Drop the memoized gate.  Test-support only."""

    global _MOE_SORTED_CACHE
    _MOE_SORTED_CACHE = None


def _routed_gather_plan(
    x: mx.array,
    expert_ids: mx.array,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """Return ``(routed_input, gather_ids, inverse)`` for the routed gathers.

    With the gate off this is the historical layout: the hidden row broadcast
    over its ten slots by ``expand_dims``, the ``[1, 4, 10]`` ids exactly as
    routed, and no un-permute.

    With the gate on the forty ``(row, expert)`` pairs are argsorted by expert
    id, so the duplicates -- the census puts the real overlap at a mean of 28
    distinct experts per 40-pair layer cycle -- land in adjacent gather rows and
    the weight tile each pair needs is already in cache for its neighbour.  That
    is the same permutation ``mlx_lm``'s ``_gather_sort`` applies at large M,
    inlined here so it is part of the compiled verify graph.  It is a pure
    reorder of independent M=1 rows: every pair still does its own dot product
    against its own expert, so the result is bit-identical once un-permuted.

    ``sorted_indices`` deliberately stays ``False`` at every call site.  MLX
    only takes its sorted-rhs fast path when the batch-to-expert ratio is at
    least four; at 40 pairs over 512 experts it would not engage, and claiming
    sortedness we do not benefit from only risks a different kernel.

    ``inverse`` is ``None`` when the gate is off, which is the signal to skip
    the un-permute entirely rather than pay an identity take.
    """

    if not fable_moe_sorted_enabled():
        return mx.expand_dims(x, (-2, -3)), expert_ids, None
    flat = expert_ids.reshape(ROWS * TOP_K)
    order = mx.argsort(flat)
    routed_input = x.reshape(ROWS, 1, HIDDEN)[order // TOP_K]
    return routed_input, flat[order], mx.argsort(order)


def qwen4_m4_stage3_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_STAGE3", default=False)


def qwen4_m4_routed_down_reduce_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", default=False)


def qwen4_m4_routed_down_residual_tail_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", default=False)


def qwen4_m4_routed_glu_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_GLU", default=False)


#: Swap the lane-major paired routed GLU for the expert-major one, which reads
#: one q4 weight tile per DISTINCT expert instead of one per (row, expert) lane.
#: Off by default; it is a strict drop-in (same signature, same grid extent,
#: same output layout) whose bit-exactness with the retained kernel is asserted
#: per layer at construction, so an arm that is not exact fails loudly rather
#: than quietly invalidating whatever is being measured under it.
FABLE_MOE_EXPERT_MAJOR_ENV = "MTPLX_FABLE_MOE_EXPERT_MAJOR"

_MOE_EXPERT_MAJOR_CACHE: bool | None = None


def fable_moe_expert_major_enabled() -> bool:
    """Return the ``MTPLX_FABLE_MOE_EXPERT_MAJOR`` gate; read once, default off.

    Resolved lazily rather than at import so a serving profile that arms env
    flags after this module is imported is still observed, then cached.
    """

    global _MOE_EXPERT_MAJOR_CACHE
    if _MOE_EXPERT_MAJOR_CACHE is None:
        _MOE_EXPERT_MAJOR_CACHE = env_bool(
            FABLE_MOE_EXPERT_MAJOR_ENV, default=False
        )
    return _MOE_EXPERT_MAJOR_CACHE


def reset_fable_moe_expert_major_cache() -> None:
    """Drop the memoized gate. Test-support only."""

    global _MOE_EXPERT_MAJOR_CACHE
    _MOE_EXPERT_MAJOR_CACHE = None


#: Replace the ten-dispatch routing head (router q8 GEMV, precise softmax,
#: argpartition top-10, take_along_axis, bf16 renormalise, shared-gate GEMV and
#: its sigmoid) with the two kernels in ``kernels/qwen4_m4_route``.  Off by
#: default.  Bit-exactness is not asserted by argument: every layer's whole
#: ``(expert_ids, route_scores, shared_factor)`` tuple is compared against the
#: stock scaffold with ``mx.array_equal`` at install, and a mismatch raises.
#: A flipped near-tie changes WHICH experts run, so there is no tolerance to
#: fall back on and no silent fallback path.
FABLE_ROUTE_KERNEL_ENV = "MTPLX_FABLE_ROUTE_KERNEL"

#: Threads per output row in the router GEMV: ``1`` reproduces MLX's own
#: ``qmv_wide`` thread layout (4,160 threads), ``4`` gives each verifier vector
#: its own lane octet (16,640 threads).  Both run the identical per-vector
#: accumulation, so this is a scheduling knob only -- see
#: ``scripts/fable/micro_route_kernel.py``.
FABLE_ROUTE_KERNEL_VEC_LANES_ENV = "MTPLX_FABLE_ROUTE_KERNEL_VEC_LANES"

_ROUTE_KERNEL_CACHE: bool | None = None
_ROUTE_KERNEL_VEC_LANES_CACHE: int | None = None


def fable_route_kernel_enabled() -> bool:
    """Return the ``MTPLX_FABLE_ROUTE_KERNEL`` gate; read once, default off."""

    global _ROUTE_KERNEL_CACHE
    if _ROUTE_KERNEL_CACHE is None:
        _ROUTE_KERNEL_CACHE = env_bool(FABLE_ROUTE_KERNEL_ENV, default=False)
    return _ROUTE_KERNEL_CACHE


def fable_route_kernel_vec_lanes() -> int:
    """Return the route GEMV's vector-lane count; read once, default 4.

    Raises on anything the kernel is not built for rather than clamping: a
    typo'd sweep value must not silently measure the default arm.
    """

    global _ROUTE_KERNEL_VEC_LANES_CACHE
    if _ROUTE_KERNEL_VEC_LANES_CACHE is None:
        raw = os.environ.get(FABLE_ROUTE_KERNEL_VEC_LANES_ENV)
        if raw is None or raw.strip() == "":
            value = _route_kernel.DEFAULT_VEC_LANES
        else:
            try:
                value = int(raw.strip())
            except ValueError as exc:
                raise ValueError(
                    f"{FABLE_ROUTE_KERNEL_VEC_LANES_ENV}={raw!r} is not an "
                    "integer"
                ) from exc
            if value not in _route_kernel.VEC_LANES_CHOICES:
                raise ValueError(
                    f"{FABLE_ROUTE_KERNEL_VEC_LANES_ENV}={value} is not one of "
                    f"{_route_kernel.VEC_LANES_CHOICES}"
                )
        _ROUTE_KERNEL_VEC_LANES_CACHE = value
    return _ROUTE_KERNEL_VEC_LANES_CACHE


def reset_fable_route_kernel_cache() -> None:
    """Drop the memoized route-kernel gates.  Test-support only."""

    global _ROUTE_KERNEL_CACHE, _ROUTE_KERNEL_VEC_LANES_CACHE
    _ROUTE_KERNEL_CACHE = None
    _ROUTE_KERNEL_VEC_LANES_CACHE = None


#: Runs the physical-M4 shared-expert branch (q8/g64 gate/up GEMV, the fused
#: split + SiLU * up, and the q8/g64 down GEMV) on a SECOND ``mx.gpu`` stream so
#: those three dispatches overlap the paired routed GLU and the routed-down
#: reduce instead of occupying barrier waves of their own.  Off by default.
#:
#: Bit-exactness is a property of construction: the ops, their arguments and
#: their order are byte-for-byte the shipped ones (``_shared_lane._emit_branch``
#: is the single definition both arms call), and only the ``Stream`` they are
#: recorded on differs.  Install still proves it per layer with
#: ``mx.array_equal``; unlike the route kernel -- where a flipped near-tie would
#: change WHICH experts run -- a mismatch here means the stream changed a value
#: it cannot change, so the lane disables itself and logs rather than raising.
#: Contract failures (a shared expert this lane is not contracted for) always
#: raise: they mean the arm measured a different pack.
#:
#: See ``mtplx/kernels/qwen4_m4_shared_lane.py`` for the measured per-layer
#: dispatch anatomy this lane is derived from, why the shared rows cannot join
#: the paired routed GLU's q4/group-32 grid, and the fence cost the overlap has
#: to beat.
FABLE_SHARED_LANE_ENV = "MTPLX_FABLE_SHARED_LANE"

_SHARED_LANE_CACHE: bool | None = None


def fable_shared_lane_enabled() -> bool:
    """Return the ``MTPLX_FABLE_SHARED_LANE`` gate; read once, default off."""

    global _SHARED_LANE_CACHE
    if _SHARED_LANE_CACHE is None:
        _SHARED_LANE_CACHE = env_bool(FABLE_SHARED_LANE_ENV, default=False)
    return _SHARED_LANE_CACHE


def reset_fable_shared_lane_cache() -> None:
    """Drop the memoized shared-lane gate.  Test-support only."""

    global _SHARED_LANE_CACHE
    _SHARED_LANE_CACHE = None


def qwen4_m4_stage3_flags() -> tuple[bool, bool, bool, bool]:
    """Capture and validate the complete construction-time feature route."""

    stage3_enabled = qwen4_m4_stage3_enabled()
    routed_down_reduce_enabled = qwen4_m4_routed_down_reduce_enabled()
    routed_down_residual_tail_enabled = (
        qwen4_m4_routed_down_residual_tail_enabled()
    )
    routed_glu_enabled = qwen4_m4_routed_glu_enabled()
    route_kernel_enabled = fable_route_kernel_enabled()
    shared_lane_enabled = fable_shared_lane_enabled()
    if not stage3_enabled and (
        routed_down_reduce_enabled
        or routed_down_residual_tail_enabled
        or routed_glu_enabled
        or route_kernel_enabled
        or shared_lane_enabled
    ):
        raise ValueError("qwen4 M4 child routes require M4 stage3")
    _validate_feature_combination(
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        moe_expert_major_enabled=fable_moe_expert_major_enabled(),
        route_kernel_enabled=route_kernel_enabled,
        shared_lane_enabled=shared_lane_enabled,
    )
    if route_kernel_enabled:
        # Read now so a typo'd sweep value fails at flag capture, before any
        # weight is touched.
        fable_route_kernel_vec_lanes()
    return (
        stage3_enabled,
        routed_down_reduce_enabled,
        routed_down_residual_tail_enabled,
        routed_glu_enabled,
    )


def _validate_feature_combination(
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
    moe_expert_major_enabled: bool = False,
    route_kernel_enabled: bool = False,
    shared_lane_enabled: bool = False,
) -> None:
    if routed_down_residual_tail_enabled and not routed_down_reduce_enabled:
        raise ValueError(
            "qwen4 M4 routed-down residual tail requires routed-down reduction"
        )
    if routed_glu_enabled and not routed_down_residual_tail_enabled:
        raise ValueError("qwen4 M4 routed GLU requires routed residual tail")
    if moe_expert_major_enabled and not routed_glu_enabled:
        raise ValueError(
            f"{FABLE_MOE_EXPERT_MAJOR_ENV} replaces the paired routed GLU and "
            "requires MTPLX_QWEN4_M4_ROUTED_GLU"
        )
    if route_kernel_enabled and not routed_glu_enabled:
        raise ValueError(
            f"{FABLE_ROUTE_KERNEL_ENV} replaces the routing head of the paired "
            "routed-GLU lane and requires MTPLX_QWEN4_M4_ROUTED_GLU"
        )
    if shared_lane_enabled and not routed_glu_enabled:
        raise ValueError(
            f"{FABLE_SHARED_LANE_ENV} re-homes the shared-expert branch of the "
            "paired routed-GLU lane and requires MTPLX_QWEN4_M4_ROUTED_GLU"
        )


def _text_model(runtime: Any):
    return getattr(runtime.model, "language_model", runtime.model)


def _m4_forward(block: SparseMoeBlock, x: mx.array, stage3) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    if _census.enabled:
        _census.record(getattr(block, "_mtplx_m4_layer_index", -1), expert_ids)
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input, gather_ids, inverse = _routed_gather_plan(x, expert_ids)
    routed_gate, routed_up = routed._gu(
        routed_input,
        gather_ids,
        sorted_indices=False,
    )
    routed_h = nn.silu(routed_gate) * routed_up
    routed_down = routed.down_proj(
        routed_h,
        gather_ids,
        sorted_indices=False,
    )
    if inverse is not None:
        routed_down = routed_down[inverse]
    routed_down = routed_down.squeeze(-2).reshape(ROWS, TOP_K, HIDDEN)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    output = stage3(
        routed_down,
        shared_down,
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_factor.astype(mx.bfloat16),
    )
    return output.reshape(x.shape).astype(x.dtype)


def _m4_routed_down_reduce_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_down_reduce,
) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    if _census.enabled:
        _census.record(getattr(block, "_mtplx_m4_layer_index", -1), expert_ids)
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input, gather_ids, inverse = _routed_gather_plan(x, expert_ids)
    routed_gate, routed_up = routed._gu(
        routed_input,
        gather_ids,
        sorted_indices=False,
    )
    routed_h = nn.silu(routed_gate) * routed_up
    # Un-permute before the routed-down kernel: it indexes routed_h and
    # expert_ids by the same flat (row * TOP_K + slot), so it must see the
    # original slot order.
    if inverse is not None:
        routed_h = routed_h[inverse]
    routed_h = routed_h.reshape(ROWS, TOP_K, INTERMEDIATE)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    output = routed_down_reduce(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids.reshape(4, 10),
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
    )
    return output.reshape(x.shape).astype(x.dtype)


def _m4_routed_down_residual_tail_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_down_residual_tail,
    hyper: mx.array,
    inject: mx.array,
) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    if _census.enabled:
        _census.record(getattr(block, "_mtplx_m4_layer_index", -1), expert_ids)
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input, gather_ids, inverse = _routed_gather_plan(x, expert_ids)
    routed_gate, routed_up = routed._gu(
        routed_input,
        gather_ids,
        sorted_indices=False,
    )
    routed_h = nn.silu(routed_gate) * routed_up
    # Un-permute before the routed-down kernel: it indexes routed_h and
    # expert_ids by the same flat (row * TOP_K + slot), so it must see the
    # original slot order.
    if inverse is not None:
        routed_h = routed_h[inverse]
    routed_h = routed_h.reshape(ROWS, TOP_K, INTERMEDIATE)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    return routed_down_residual_tail(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids.reshape(4, 10),
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
        hyper,
        inject,
    )


def _m4_paired_routed_glu_residual_tail_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_gu_activation,
    routed_down_residual_tail,
    hyper: mx.array,
    inject: mx.array,
    route=None,
    shared_lane: bool = False,
) -> mx.array:
    """Retained lane: paired routed GLU producer plus the fused residual tail.

    ``MTPLX_FABLE_MOE_SORTED`` deliberately does not apply here.  Both routed
    gathers on this lane happen *inside* Metal kernels -- the paired GLU kernel
    resolves ``expert = expert_ids[selected]`` and walks the fused pack itself,
    and the routed-down kernel does the same over its fixed ``SLOT_ORDER``
    reduction tree.  There is no ``mx.gather_qmm`` left to reorder, so the
    sorted-adjacent win has to be taken kernel-side (order the ``selected``
    threadgroups by expert id, or hoist the sort into the kernel's own indexing)
    rather than by permuting tensors in this forward.  Permuting the ids handed
    to these kernels without matching the reduction tree would break the
    bit-exact accumulation order they are validated against.
    """

    # One metadata-only view, shared by the route head and the routed GLU.
    rows = x.reshape(ROWS, HIDDEN)
    if route is None:
        gates = mx.softmax(block.gate(x), axis=-1, precise=True)
        expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
            ..., -block.top_k :
        ]
        route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
        if block.norm_topk_prob:
            route_scores = route_scores / route_scores.sum(
                axis=-1, keepdims=True
            )
        expert_ids = expert_ids.reshape(ROWS, TOP_K)
        route_scores = route_scores.reshape(ROWS, TOP_K)
        shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(ROWS)
    else:
        # MTPLX_FABLE_ROUTE_KERNEL: the same ten dispatches in two, emitting
        # the tuple below directly. Validated bit-exact per layer at install.
        expert_ids, route_scores, shared_factor = route(
            rows,
            block.gate.weight,
            block.gate.scales,
            block.gate.biases,
            block.shared_expert_gate.weight,
            block.shared_expert_gate.scales,
            block.shared_expert_gate.biases,
        )
    if _census.enabled:
        _census.record(getattr(block, "_mtplx_m4_layer_index", -1), expert_ids)

    routed = block.switch_mlp
    routed_h = routed_gu_activation(
        rows,
        routed.gu_weight,
        routed.gu_scales,
        routed.gu_biases,
        expert_ids,
    )

    # MTPLX_FABLE_SHARED_LANE: identical ops, identical arguments, identical
    # order -- only the stream they are recorded on differs, so the result is
    # bit-exact by construction (and proved per layer at install).  On the
    # second stream the three shared dispatches overlap the paired routed GLU
    # and the routed-down reduce instead of sitting in barrier waves of their
    # own; MLX pays for that with one fence crossing in each direction.
    shared_down = (
        _shared_lane.shared_branch(block, x)
        if shared_lane
        else _shared_lane.stock_shared_branch(block, x)
    )

    return routed_down_residual_tail(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids,
        route_scores.astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
        hyper,
        inject,
    )


def _m4_routed_down_residual_tail_layer_forward(
    layer: DecoderLayer,
    hidden: mx.array,
    *,
    input_ids: mx.array,
    ssm_mask: mx.array,
    cache: Any,
) -> mx.array:
    if "ple" in layer:
        hidden = hidden + layer.ple(hidden, input_ids, cache)

    mixed, hyper, inject = layer.attn_hyper_connection(hidden)
    if layer.is_linear:
        block_out = layer.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = layer.self_attn(mixed, cache)
    hidden = _hyper_residual_write(hyper, block_out, inject)

    mixed, hyper, inject = layer.mlp_hyper_connection(hidden)
    return _m4_routed_down_residual_tail_forward(
        layer.mlp,
        mixed,
        layer._mtplx_m4_routed_down_residual_tail,
        hyper,
        inject,
    )


def _m4_paired_routed_glu_residual_tail_layer_forward(
    layer: DecoderLayer,
    hidden: mx.array,
    *,
    input_ids: mx.array,
    ssm_mask: mx.array,
    cache: Any,
) -> mx.array:
    if "ple" in layer:
        hidden = hidden + layer.ple(hidden, input_ids, cache)

    mixed, hyper, inject = layer.attn_hyper_connection(hidden)
    if layer.is_linear:
        block_out = layer.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = layer.self_attn(mixed, cache)
    hidden = _hyper_residual_write(hyper, block_out, inject)

    mixed, hyper, inject = layer.mlp_hyper_connection(hidden)
    return _m4_paired_routed_glu_residual_tail_forward(
        layer.mlp,
        mixed,
        layer._mtplx_m4_routed_glu,
        layer._mtplx_m4_routed_down_residual_tail,
        hyper,
        inject,
        route=getattr(layer, "_mtplx_m4_route", None),
        shared_lane=getattr(layer, "_mtplx_m4_shared_lane", False),
    )


class _M4Stage3SparseMoeBlock(SparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        rows = int(x.size // x.shape[-1])
        if rows == 4:
            return _m4_forward(self, x, self._mtplx_m4_stage3)
        return super().__call__(x)


class _M4RoutedDownReduceSparseMoeBlock(SparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        rows = int(x.size // x.shape[-1])
        if rows == 4:
            return _m4_routed_down_reduce_forward(
                self,
                x,
                self._mtplx_m4_routed_down_reduce,
            )
        return super().__call__(x)


class _M4RoutedDownResidualTailDecoderLayer(DecoderLayer):
    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        rows = int(hidden.size // hidden.shape[-1])
        if rows == 4:
            return _m4_routed_down_residual_tail_layer_forward(
                self,
                hidden,
                input_ids=input_ids,
                ssm_mask=ssm_mask,
                cache=cache,
            )
        return super().__call__(
            hidden,
            input_ids=input_ids,
            ssm_mask=ssm_mask,
            cache=cache,
        )


class _M4PairedRoutedGluResidualTailDecoderLayer(DecoderLayer):
    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        rows = int(hidden.size // hidden.shape[-1])
        if rows == 4:
            return _m4_paired_routed_glu_residual_tail_layer_forward(
                self,
                hidden,
                input_ids=input_ids,
                ssm_mask=ssm_mask,
                cache=cache,
            )
        return super().__call__(
            hidden,
            input_ids=input_ids,
            ssm_mask=ssm_mask,
            cache=cache,
        )


def _projection_contract(projection: Any) -> tuple[Any, ...]:
    return (
        int(projection.bits),
        int(projection.group_size),
        str(projection.mode),
        projection.weight.dtype,
        projection.scales.dtype,
        projection.biases.dtype,
        tuple(projection.weight.shape),
        tuple(projection.scales.shape),
        tuple(projection.biases.shape),
    )


def _fused_gu_contract(owner: Any) -> tuple[Any, ...]:
    return (
        int(owner.bits),
        int(owner.group_size),
        str(owner.mode),
        owner.gu_weight.dtype,
        owner.gu_scales.dtype,
        owner.gu_biases.dtype,
        tuple(owner.gu_weight.shape),
        tuple(owner.gu_scales.shape),
        tuple(owner.gu_biases.shape),
    )


_ROUTER_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 640),
    (512, 40),
    (512, 40),
)
_SHARED_GATE_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (1, 640),
    (1, 40),
    (1, 40),
)
_ROUTED_GU_CONTRACT = (
    4,
    32,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 1280, 320),
    (512, 1280, 80),
    (512, 1280, 80),
)
_SHARED_GU_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (1280, 640),
    (1280, 40),
    (1280, 40),
)
_ROUTED_DOWN_CONTRACT = (
    4,
    32,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 2560, 80),
    (512, 2560, 20),
    (512, 2560, 20),
)
_SHARED_DOWN_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (2560, 160),
    (2560, 10),
    (2560, 10),
)


def _validate_layer_owner(layer: Any, *, index: int) -> None:
    if type(layer) is not DecoderLayer:
        raise ValueError(f"qwen4 M4 stage3 layer {index} has wrong decoder owner")


def _validate_block_contract(block: Any, *, index: int) -> None:
    label = f"qwen4 M4 stage3 layer {index}"
    if type(block) is not SparseMoeBlock:
        raise ValueError(f"{label} has wrong MoE owner")
    if int(block.num_experts) != 512:
        raise ValueError(f"{label} requires 512 experts")
    if int(block.top_k) != 10:
        raise ValueError(f"{label} requires exact top-k 10")
    if not bool(block.norm_topk_prob):
        raise ValueError(f"{label} requires top-k probability normalization")
    if block.sharding_group is not None:
        raise ValueError(f"{label} does not support sharding")
    if type(block.switch_mlp) is not _FusedGateUpSwitchGLU:
        raise ValueError(f"{label} lacks exact routed fused GU owner")
    if type(block.shared_expert) is not _FusedGateUpMLP:
        raise ValueError(f"{label} lacks exact shared fused GU owner")
    if _projection_contract(block.gate) != _ROUTER_CONTRACT:
        raise ValueError(f"{label} router mismatch")
    if _projection_contract(block.shared_expert_gate) != _SHARED_GATE_CONTRACT:
        raise ValueError(f"{label} shared gate mismatch")
    if _fused_gu_contract(block.switch_mlp) != _ROUTED_GU_CONTRACT:
        raise ValueError(f"{label} routed fused GU mismatch")
    if _fused_gu_contract(block.shared_expert) != _SHARED_GU_CONTRACT:
        raise ValueError(f"{label} shared fused GU mismatch")
    if _projection_contract(block.switch_mlp.down_proj) != _ROUTED_DOWN_CONTRACT:
        raise ValueError(f"{label} routed down mismatch")
    if _projection_contract(block.shared_expert.down_proj) != _SHARED_DOWN_CONTRACT:
        raise ValueError(f"{label} shared down mismatch")


def _validate_input_contract(layer: Any, *, index: int) -> None:
    norm = getattr(getattr(layer, "mlp_hyper_connection", None), "hc_norm", None)
    weight = getattr(norm, "weight", None)
    if (
        weight is None
        or tuple(weight.shape) != (4 * 2560,)
        or weight.dtype != mx.bfloat16
    ):
        raise ValueError(
            f"qwen4 M4 stage3 layer {index} requires BF16 hidden input ownership"
        )


def _validate_residual_tail_contract(layer: Any, *, index: int) -> None:
    """Admit only the exact BF16 hyper geometry consumed by the fused store."""

    label = f"qwen4 M4 residual tail layer {index}"
    connection = getattr(layer, "mlp_hyper_connection", None)
    if (
        connection is None
        or int(getattr(connection, "hc_count", -1)) != 4
        or int(getattr(connection, "hidden_size", -1)) != 2560
    ):
        raise ValueError(f"{label} requires exactly four hyper streams")
    inject_projection = getattr(connection, "block_inject_weight", None)
    inject_weight = getattr(inject_projection, "weight", None)
    if (
        inject_weight is None
        or tuple(inject_weight.shape) != (4, 4 * 2560)
        or inject_weight.dtype != mx.bfloat16
    ):
        raise ValueError(f"{label} requires BF16 inject ownership")


def bind_qwen4_m4_residual_tail(layer: Any, *, index: int):
    """Validate and bind the optional M4 residual tail at construction."""

    _validate_input_contract(layer, index=index)
    _validate_residual_tail_contract(layer, index=index)
    _validate_block_contract(layer.mlp, index=index)
    return bind_residual_tail()


def _build_install_plans(
    layers,
    *,
    stage3,
    routed_down_reduce,
    routed_down_residual_tail_enabled: bool,
):
    plans = []
    for index, layer in enumerate(layers):
        block = layer.mlp
        if routed_down_residual_tail_enabled:
            _validate_layer_owner(layer, index=index)
        _validate_input_contract(layer, index=index)
        _validate_block_contract(block, index=index)
        if routed_down_residual_tail_enabled:
            _validate_residual_tail_contract(layer, index=index)
        plans.append((layer, block, stage3, routed_down_reduce))
    return tuple(plans)


def _installation_report(
    *,
    layer_count: int,
    max_delta: float,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
    moe_expert_major_enabled: bool = False,
    route_kernel_enabled: bool = False,
    route_kernel_vec_lanes: int | None = None,
    shared_lane_enabled: bool = False,
    shared_lane_layers: int = 0,
) -> dict[str, Any]:
    return {
        "installed": True,
        "layers": layer_count,
        "rows": 4,
        "max_abs_diff": max_delta,
        "boundary": (
            "expert_major_routed_q4g32_glu_reduce_shared_add_mlp_residual"
            if routed_glu_enabled and moe_expert_major_enabled
            else "paired_routed_q4g32_glu_reduce_shared_add_mlp_residual"
            if routed_glu_enabled
            else "routed_q4g32_reduce_shared_add_mlp_residual"
            if routed_down_residual_tail_enabled
            else "stock_qmm_combine_tail"
        ),
        "reference_boundary": (
            "retained_m4_routed_down_then_stock_residual"
            if routed_down_residual_tail_enabled
            else "stock_sparse_moe_block"
        ),
        "routed": "stock_q4/g32",
        "shared": "stock_q8/g64",
        "routed_down_reduce": routed_down_reduce_enabled,
        "routed_down_residual_tail": routed_down_residual_tail_enabled,
        "paired_routed_glu": routed_glu_enabled,
        "paired_routed_glu_layers": layer_count if routed_glu_enabled else 0,
        "moe_expert_major_glu": moe_expert_major_enabled,
        "moe_expert_major_glu_layers": (
            layer_count if moe_expert_major_enabled else 0
        ),
        "route_kernel": {
            "installed": bool(route_kernel_enabled),
            "layers": layer_count if route_kernel_enabled else 0,
            "vec_lanes": route_kernel_vec_lanes,
            "dispatches_per_layer": (
                _route_kernel.FUSED_DISPATCHES_PER_LAYER
                if route_kernel_enabled
                else _route_kernel.EAGER_DISPATCHES_PER_LAYER
            ),
        },
        "shared_lane": {
            # "armed" and "installed" are deliberately separate: a lane whose
            # per-layer exactness check failed disables itself, and an A/B that
            # read flat must be able to tell "ran and did nothing" from "never
            # ran".
            "armed": bool(shared_lane_enabled),
            "installed": bool(shared_lane_enabled and shared_lane_layers > 0),
            "layers": shared_lane_layers,
            "branch_dispatches_per_layer": (
                _shared_lane.BRANCH_DISPATCHES_PER_LAYER
            ),
            "fence_dispatches_per_layer": (
                _shared_lane.FENCE_DISPATCHES_PER_LAYER
                if shared_lane_layers
                else 0
            ),
            "weight_bytes_per_layer": _shared_lane.BYTES_PER_LAYER,
            "exactness_failures": _shared_lane.COUNTERS["exactness_failures"],
        },
        "exact_layers": layer_count,
        "combined_residual_tail_layers": (
            layer_count if routed_down_residual_tail_enabled else 0
        ),
    }


def _install_validated_plans(
    plans,
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
    routed_glu=None,
    route=None,
    shared_lane: bool = False,
) -> None:
    """Mutate only the complete plan set after validation and exact self-checks."""

    for index, (layer, block, stage3, routed_down_reduce) in enumerate(plans):
        block._mtplx_m4_stage3 = stage3
        # Plain int, so mlx's Module.__setattr__ keeps it off the parameter
        # dict. Only the opt-in expert census reads it.
        block._mtplx_m4_layer_index = index
        if routed_glu_enabled:
            layer._mtplx_m4_routed_glu = routed_glu
            layer._mtplx_m4_routed_down_residual_tail = routed_down_reduce
            # Plain callable or None; Module.__setattr__ keeps it off the
            # parameter dict, and the forward reads it with getattr so an
            # un-armed layer never pays a branch on a missing attribute.
            layer._mtplx_m4_route = route
            # Plain bool; read with getattr in the forward so an un-armed layer
            # never pays a branch on a missing attribute.
            layer._mtplx_m4_shared_lane = bool(shared_lane)
            layer.__class__ = _M4PairedRoutedGluResidualTailDecoderLayer
        elif routed_down_residual_tail_enabled:
            layer._mtplx_m4_routed_down_residual_tail = routed_down_reduce
            layer.__class__ = _M4RoutedDownResidualTailDecoderLayer
        elif routed_down_reduce_enabled:
            block._mtplx_m4_routed_down_reduce = routed_down_reduce
            block.__class__ = _M4RoutedDownReduceSparseMoeBlock
        else:
            block.__class__ = _M4Stage3SparseMoeBlock


def install_qwen4_m4_stage3(
    runtime: Any,
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
    routed_glu_enabled: bool = False,
) -> dict[str, Any]:
    """Validate every owner, self-check the kernel, then install M4 directly."""

    text = _text_model(runtime)
    layers = tuple(text.model.layers)
    if len(layers) != 48:
        raise ValueError(f"qwen4 M4 stage3 requires 48 layers, got {len(layers)}")

    expert_major_enabled = fable_moe_expert_major_enabled()
    route_kernel_enabled = fable_route_kernel_enabled()
    route_kernel_vec_lanes = (
        fable_route_kernel_vec_lanes() if route_kernel_enabled else None
    )
    # Read before validation, not next to its first use: an install reached
    # directly (rather than through qwen4_m4_stage3_flags) with the lane armed
    # and the paired routed-GLU lane off would otherwise install nothing and
    # report a flat A/B as "the lane did not help".
    shared_lane_enabled = fable_shared_lane_enabled()
    _validate_feature_combination(
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        moe_expert_major_enabled=expert_major_enabled,
        route_kernel_enabled=route_kernel_enabled,
        shared_lane_enabled=shared_lane_enabled,
    )
    validated_plans = _build_install_plans(
        layers,
        stage3=None,
        routed_down_reduce=None,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
    )
    if route_kernel_enabled:
        # Construction-bound: every pack the two kernels hardcode is checked
        # before anything is bound, with the offending field named. Runs after
        # _build_install_plans so an owner-type problem reports as one.
        for index, (_, block, _, _) in enumerate(validated_plans):
            _route_kernel.check_contract(block, index=index)
    if shared_lane_enabled:
        # Contract failures RAISE: the lane's whole justification is the
        # measured dispatch anatomy of the q8/group-64 shared expert, so a
        # different pack means the arm is measuring something else.
        for index, (_, block, _, _) in enumerate(validated_plans):
            _shared_lane.check_contract(block, index=index)
    stage3 = bind()
    routed_down_reduce = (
        bind_residual_tail()
        if routed_down_residual_tail_enabled
        else bind_routed_down_reduce()
        if routed_down_reduce_enabled
        else None
    )
    routed_glu = bind_routed_glu() if routed_glu_enabled else None
    # Both are bound when the expert-major arm is on: the retained kernel stays
    # live purely as the per-layer bit-exactness reference below.
    expert_major_glu = (
        bind_expert_major_routed_glu() if expert_major_enabled else None
    )
    route = (
        _route_kernel.bind(vec_lanes=route_kernel_vec_lanes)
        if route_kernel_enabled
        else None
    )
    plans = tuple(
        (layer, block, stage3, routed_down_reduce)
        for layer, block, _, _ in validated_plans
    )

    sample = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.0009765625)
    sample = sample.reshape(1, 4, 2560).astype(mx.bfloat16)
    max_delta = 0.0
    for index, (layer, block, stage3, routed_down_reduce) in enumerate(plans):
        if route is not None:
            # The route head decides WHICH experts run. A near-tie that flips
            # is not a rounding-class difference, so this gate is array_equal
            # on all three emitted tensors -- no tolerance, no fallback.
            _route_kernel.check_input(sample.reshape(4, 2560))
            reference_route = _route_kernel.stock_route(block, sample)
            candidate_route = route(
                sample.reshape(4, 2560),
                block.gate.weight,
                block.gate.scales,
                block.gate.biases,
                block.shared_expert_gate.weight,
                block.shared_expert_gate.scales,
                block.shared_expert_gate.biases,
            )
            names = ("expert_ids", "route_scores", "shared_factor")
            checks = tuple(
                mx.array_equal(want, got)
                for want, got in zip(reference_route, candidate_route)
            )
            mx.eval(checks)
            for name, ok in zip(names, checks):
                if not bool(ok.item()):
                    raise ValueError(
                        f"{FABLE_ROUTE_KERNEL_ENV} layer {index}: {name} is "
                        "not bit-exact with the stock routing scaffold"
                    )
        if shared_lane_enabled:
            # The lane re-homes three dispatches onto a second stream without
            # touching an operand, so this can only fail if a stream changed a
            # value it cannot change.  That is a defect in the runtime, not a
            # rounding-class trade, so the lane disables itself and says so
            # rather than raising and taking the whole model down with it.
            want = _shared_lane.stock_shared_branch(block, sample)
            got = _shared_lane.shared_branch(block, sample)
            same = mx.array_equal(want, got)
            mx.eval(same)
            if not bool(same.item()):
                _shared_lane.COUNTERS["exactness_failures"] += 1
                shared_lane_enabled = False
                logger.warning(
                    "%s layer %d: the second-stream shared branch is not "
                    "bit-exact with the stock branch; disabling the lane for "
                    "every layer (this arm now measures the stock path)",
                    FABLE_SHARED_LANE_ENV,
                    index,
                )
        stock = None if routed_down_residual_tail_enabled else block(sample)
        reference = _m4_forward(block, sample, stage3)
        if routed_down_residual_tail_enabled:
            hyper = mx.sin(
                mx.arange(4 * 4 * 2560, dtype=mx.float32) * 0.000244140625
            ).reshape(1, 4, 4 * 2560).astype(mx.bfloat16)
            inject = mx.cos(
                mx.arange(4 * 4, dtype=mx.float32) * 0.0625
            ).reshape(1, 4, 4).astype(mx.bfloat16)
            candidate = (
                _m4_paired_routed_glu_residual_tail_forward(
                    block,
                    sample,
                    routed_glu,
                    routed_down_reduce,
                    hyper,
                    inject,
                )
                if routed_glu_enabled
                else _m4_routed_down_residual_tail_forward(
                    block,
                    sample,
                    routed_down_reduce,
                    hyper,
                    inject,
                )
            )
            if expert_major_glu is not None:
                expert_major_candidate = (
                    _m4_paired_routed_glu_residual_tail_forward(
                        block,
                        sample,
                        expert_major_glu,
                        routed_down_reduce,
                        hyper,
                        inject,
                    )
                )
                exact = mx.array_equal(candidate, expert_major_candidate)
                mx.eval(exact)
                if not bool(exact.item()):
                    raise ValueError(
                        f"{FABLE_MOE_EXPERT_MAJOR_ENV} layer {index} is not "
                        "bit-exact with the retained paired routed GLU"
                    )
                candidate = expert_major_candidate
            reference = hyper + (
                reference[..., None, :] * inject[..., :, None]
            ).reshape(*hyper.shape)
        elif routed_down_reduce_enabled:
            candidate = _m4_routed_down_reduce_forward(
                block, sample, routed_down_reduce
            )
        else:
            candidate = reference
        if routed_down_reduce_enabled:
            exact = mx.array_equal(reference, candidate)
            mx.eval(exact)
            if not bool(exact.item()):
                raise ValueError(
                    f"qwen4 M4 stage3 layer {index} routed-down "
                    "reduction self-check failed"
                )
        comparison = reference if routed_down_residual_tail_enabled else stock
        delta = mx.max(
            mx.abs(comparison.astype(mx.float32) - candidate.astype(mx.float32))
        )
        finite = mx.all(mx.isfinite(candidate))
        mx.eval(delta, finite)
        observed = float(delta.item())
        if not bool(finite.item()) or observed > 0.001953125:
            raise ValueError(
                f"qwen4 M4 stage3 layer {index} self-check failed: dmax={observed}"
            )
        max_delta = max(max_delta, observed)

    _install_validated_plans(
        plans,
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        routed_glu=expert_major_glu if expert_major_glu is not None else routed_glu,
        route=route,
        shared_lane=shared_lane_enabled,
    )
    if shared_lane_enabled:
        _shared_lane.COUNTERS["installed_layers"] += len(plans)
    report = _installation_report(
        layer_count=len(plans),
        max_delta=max_delta,
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
        routed_glu_enabled=routed_glu_enabled,
        moe_expert_major_enabled=expert_major_enabled,
        route_kernel_enabled=route_kernel_enabled,
        route_kernel_vec_lanes=route_kernel_vec_lanes,
        shared_lane_enabled=shared_lane_enabled,
        shared_lane_layers=len(plans) if shared_lane_enabled else 0,
    )
    logger.info(
        "%s",
        _shared_lane.engagement_line(
            installed_layers=len(plans) if shared_lane_enabled else 0,
            enabled=shared_lane_enabled,
        ),
    )
    runtime.qwen4_m4_stage3_report = report
    return report


__all__ = [
    "FABLE_MOE_SORTED_ENV",
    "FABLE_MOE_EXPERT_MAJOR_ENV",
    "FABLE_ROUTE_KERNEL_ENV",
    "FABLE_ROUTE_KERNEL_VEC_LANES_ENV",
    "FABLE_SHARED_LANE_ENV",
    "bind_qwen4_m4_residual_tail",
    "fable_moe_sorted_enabled",
    "fable_moe_expert_major_enabled",
    "fable_route_kernel_enabled",
    "fable_route_kernel_vec_lanes",
    "fable_shared_lane_enabled",
    "install_qwen4_m4_stage3",
    "qwen4_m4_routed_down_reduce_enabled",
    "qwen4_m4_routed_down_residual_tail_enabled",
    "qwen4_m4_routed_glu_enabled",
    "qwen4_m4_stage3_enabled",
    "qwen4_m4_stage3_flags",
    "reset_fable_moe_sorted_cache",
    "reset_fable_moe_expert_major_cache",
    "reset_fable_route_kernel_cache",
    "reset_fable_shared_lane_cache",
]
