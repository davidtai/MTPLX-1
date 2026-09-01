"""Reduced-dispatch QSA indexer chain for the Qwen3.8 fixed-M4 verifier.

WHY THIS EXISTS
---------------
After the M4 hyper read (``MTPLX_FABLE_HC_M4``) and the op diet
(``MTPLX_FABLE_OPDIET``), the twelve QSA layers are about half of every
dispatch the compiled verify graph issues.  The QSA block is not
bandwidth-bound: it is a long DEPENDENT chain of tiny kernels around three
real operations (the index projection, the score GEMM, and the attention
pair).  Three of its sub-chains are pure glue:

  ``_extend_pooled_fixed``   24 dispatches to write ONE 128-element block row
  ``_prepare_queries_eager`` 12 dispatches of RMSNorm + partial RoPE
  scoring epilogue            9 dispatches of relu/sum/scale/mask/tie-break
  rows-gather token build    18 dispatches of integer index arithmetic

This module collapses each into one Metal dispatch:

  pooled row      24 -> 1 + the 3-dispatch mx.slice_update the diet already had
  query prep      12 -> 1  (the SHIPPED prepare kernel, which the fixed lane
                            simply never called)
  scoring epilogue 9 -> 1
  token build     18 -> 1
                  --------
                  63 -> 7  per QSA layer, 672 per verify cycle over 12 layers

Counts are compiled-lane node counts measured on the CPU stream by
tests/test_fable_qsa_m4.py::test_dispatch_map_before_after, against the
op-diet-armed baseline that ships today.

WHAT IS EXACT AND WHAT IS NOT
-----------------------------
``qsa_m4_row_tokens`` is BIT-EXACT by construction: it is integer and boolean
arithmetic only, and every value it emits is a closed-form function of
``top_idx`` and the row position that the stock chain computes the same way.

``qsa_m4_pooled_row`` reuses, line for line, the arithmetic of the SHIPPED
``qsa_indexer_pool_keys_metal`` (kernels/qsa_indexer_prepare.py), whose
bit-exactness against the eager mean -> RMSNorm -> partial-RoPE chain is
pinned for bf16/f16 in tests/test_qsa_indexer_prepare_metal.py.  The one
addition is the conditional merge with the bank's current row, which is a
select.  So: exact on the same terms as the kernel it inherits from --
head_dim <= 128 (MLX's ``rms_single_row`` 32-lane x 4-value reduction is
reproduced exactly only in that regime) and a non-float32 activation dtype.

``qsa_m4_index_scores`` folds relu -> head-sum -> scale -> validity mask ->
tie-break into one kernel.  Every step is elementwise except the head sum,
which is a 4-term fp32 accumulation.  This kernel accumulates head 0..H-1 in
order; MLX's ``col_reduce_small`` walks the same axis in the same order, so
the two agree on every input we can construct.  That is an ASSUMPTION about
an MLX implementation detail, not a theorem, and it is the only one in this
module -- if it ever fails the difference is a 1-ulp fp32 reassociation,
which can flip a top-k tie.  It is called out in the microbench
(scripts/fable/micro_qsa_m4.py prints max-abs-diff and a differing-element
count against the stock chain at the real 4x4352 shape).

WHAT IS DELIBERATELY NOT FUSED
------------------------------
The top-k itself (``mx.argpartition``, 5 dispatches).  A fused per-row
top-512 selector is easy to write and impossible to make equivalent: the
downstream ``token_idx`` layout is ``top_idx``'s ORDER, so the gathered K/V
rows -- and therefore the softmax denominator's and the PV product's
accumulation order -- depend on an unspecified detail of MLX's multi-block
partition.  Reordering the selected set is a rounding-class change to the
attention output for a 5-dispatch saving, so the stock selector stays.

No silent fallback: every entry point validates and RAISES.  The model's
construction-time gate decides whether this lane is eligible at all.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache

import mlx.core as mx

#: Same regime as kernels/qsa_indexer_prepare.py: MLX's rms_single_row uses
#: 32 lanes x 4 contiguous values, so head_dim beyond 128 is not reproducible.
SIMD = 32
MAX_EXACT_HEAD_DIM = SIMD * 4

_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)

#: Dispatches each helper replaces, for the microbench's count column.
#: (measured with mx.compile on the CPU stream; see tests/test_fable_qsa_m4.py)
STOCK_DISPATCHES = {
    "pooled_row": 24,
    "prepare_queries": 12,
    "index_scores": 9,
    "row_tokens": 18,
}

#: What each becomes.  ``pooled_row`` keeps the diet's mx.slice_update bank
#: write (1 kernel + 3 for the non-donatable dynamic update).
FUSED_DISPATCHES = {
    "pooled_row": 4,
    "prepare_queries": 1,
    "index_scores": 1,
    "row_tokens": 1,
}


def _f32_literal(value: float) -> str:
    """Emit the Metal literal for the fp32 MLX would weak-promote to.

    ``masked - blk * 1e-12`` and ``/ math.sqrt(128)`` both start life as
    python doubles; MLX narrows them to float32 before the op runs.  Round
    here the same way so the kernel divides and subtracts by the same bits.
    """

    bits = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    return f"{bits!r}f"


def _dtype_tag(dtype: mx.Dtype) -> str:
    return {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}[dtype]


def _tag(value: float) -> str:
    return (
        format(float(value), ".9g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "d")
    )


def _grid_1d(threads: int, group: int = 256) -> tuple[int, int]:
    """Round the 1-D grid up to a whole number of threadgroups.

    Every kernel here bounds-checks its own linear id, so the tail threads of
    the last group are inert; rounding up keeps the launch uniform instead of
    relying on non-uniform threadgroup dispatch.
    """

    group = max(1, min(int(group), int(threads)))
    return ((int(threads) + group - 1) // group) * group, group


def _as_i32_scalar(value, name: str) -> mx.array:
    """A one-element int32 leaf, without synchronizing a traced offset."""

    if isinstance(value, mx.array):
        if value.dtype != mx.int32 or int(value.size) != 1:
            raise TypeError(f"{name} must be one int32 value; got {value.dtype}")
        return value.reshape((1,))
    return mx.array([int(value)], dtype=mx.int32)


# ---------------------------------------------------------------------------
# 1 -- pooled bank row: gather + mean + RMSNorm + partial RoPE + merge
# ---------------------------------------------------------------------------
#
# The bank WRITE stays where the op diet put it.  ``mx.slice_update`` on a
# held state leaf is one contiguous full-bank copy plus its offset compute,
# and no custom kernel can beat that: mx.fast.metal_kernel cannot write into
# an input, so an "in-place" spelling would have to emit the whole bank as an
# output -- the same copy, with the gather bolted on.  What this kernel
# removes is everything the diet's ``bank_rowsel`` spelling still needs
# AROUND that copy: the dynamic raw-key slice, the fp32 mean, the RMSNorm,
# the RoPE table, the rotation, the old-row read and the select.
#
# It also emits ``safe_block`` so the caller needs no scalar op chain of its
# own: min(offset // ratio, capacity - 1) happens in the kernel.
_SRC_POOLED_ROW = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint lane_base = lane * 4u;
    threadgroup float rounded_means[HEAD_DIM];

    // block = offset // RATIO, clamped into the bank; write = the block the
    // stock chain would have completed this step.
    const int off = offset[0];
    const int block_raw = off / int(RATIO);
    const int block = block_raw < int(POOLED_CAP) - 1
        ? block_raw : int(POOLED_CAP) - 1;
    const int nb_total = (off + int(STEP_ROWS)) / int(RATIO);
    const bool write = nb_total > block_raw;
    if (lane == 0) { safe_block[0] = block; }

    // Stock: mean of RATIO raw rows in fp32, rounded to the activation dtype
    // BEFORE the norm (mx.mean(...).astype(dtype)).
    float square_sum = 0.0f;
    for (uint i = 0; i < 4u; ++i) {
        const uint dim = lane_base + i;
        if (dim < HEAD_DIM) {
            float sum = 0.0f;
            for (uint within = 0; within < RATIO; ++within) {
                const uint token = uint(block) * RATIO + within;
                sum += float(raw_keys[
                    (size_t)token * raw_keys_strides[1] +
                    (size_t)dim * raw_keys_strides[2]]);
            }
            const T rounded = static_cast<T>(sum / float(RATIO));
            const float mean_value = float(rounded);
            rounded_means[dim] = mean_value;
            square_sum += mean_value * mean_value;
        }
    }
    square_sum = simd_sum(square_sum);
    const float inverse_rms = metal::precise::rsqrt(
        square_sum / float(HEAD_DIM) + RMS_EPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t old_base =
        (size_t)block * pooled_strides[1];
    const float position = float(block * int(RATIO));
    for (uint pair = lane; pair < HALF_ROTARY; pair += 32u) {
        const T first_norm = norm_weight[
            (size_t)pair * norm_weight_strides[0]] *
            static_cast<T>(rounded_means[pair] * inverse_rms);
        const T second_norm = norm_weight[
            (size_t)(pair + HALF_ROTARY) * norm_weight_strides[0]] *
            static_cast<T>(
                rounded_means[pair + HALF_ROTARY] * inverse_rms);
        const float theta = position * float(inv_freq[
            (size_t)pair * inv_freq_strides[0]]);
        const float cosine =
            metal::precise::cos(theta) * ROPE_ATTENTION_SCALE;
        const float sine =
            metal::precise::sin(theta) * ROPE_ATTENTION_SCALE;
        const float first = float(first_norm);
        const float second = float(second_norm);
        // Distinct fp32 products: an FMA contraction here moves bf16 cutoff
        // values and can perturb an exact top-k set (same note as
        // kernels/qsa_indexer_prepare.py).
        const float first_cosine = first * cosine;
        const float second_sine = second * sine;
        const float second_cosine = second * cosine;
        const float first_sine = first * sine;
        const T lo = static_cast<T>(first_cosine - second_sine);
        const T hi = static_cast<T>(second_cosine + first_sine);
        merged[pair] = write ? lo : pooled[
            old_base + (size_t)pair * pooled_strides[2]];
        merged[pair + HALF_ROTARY] = write ? hi : pooled[
            old_base + (size_t)(pair + HALF_ROTARY) * pooled_strides[2]];
    }
    for (uint dim = ROTARY_DIM + lane; dim < HEAD_DIM; dim += 32u) {
        const T normalized = norm_weight[
            (size_t)dim * norm_weight_strides[0]] *
            static_cast<T>(rounded_means[dim] * inverse_rms);
        merged[dim] = write ? normalized : pooled[
            old_base + (size_t)dim * pooled_strides[2]];
    }
"""


@lru_cache(maxsize=64)
def _pooled_row_kernel(
    head_dim: int,
    rotary_dim: int,
    ratio: int,
    step_rows: int,
    pooled_cap: int,
    eps: float,
    attention_scaling: float,
    dtype: mx.Dtype,
):
    header = f"""
        #include <metal_stdlib>
        using namespace metal;
        constant constexpr uint HEAD_DIM = {head_dim};
        constant constexpr uint ROTARY_DIM = {rotary_dim};
        constant constexpr uint HALF_ROTARY = {rotary_dim // 2};
        constant constexpr uint RATIO = {ratio};
        constant constexpr uint STEP_ROWS = {step_rows};
        constant constexpr uint POOLED_CAP = {pooled_cap};
        constant constexpr float RMS_EPS = {float(eps)!r}f;
        constant constexpr float ROPE_ATTENTION_SCALE = {float(attention_scaling)!r}f;
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_m4_pooled_row_d{head_dim}_r{rotary_dim}_c{ratio}_"
            f"s{step_rows}_p{pooled_cap}_{_tag(attention_scaling)}_"
            f"{_dtype_tag(dtype)}"
        ),
        input_names=["raw_keys", "pooled", "norm_weight", "inv_freq", "offset"],
        output_names=["merged", "safe_block"],
        header=header,
        source=_SRC_POOLED_ROW,
        ensure_row_contiguous=False,
    )


def qsa_m4_pooled_row(
    raw_keys: mx.array,
    pooled: mx.array,
    norm_weight: mx.array,
    inv_freq: mx.array,
    offset,
    *,
    compress_ratio: int,
    step_rows: int,
    eps: float,
    attention_scaling: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """One dispatch for the fixed bank's newly completed block row.

    Returns ``(merged [1, 1, D], safe_block [] int32)``.  ``merged`` is the
    value the caller must ``mx.slice_update`` into ``pooled`` at
    ``safe_block``; when no block completes this step it is the bank's
    current row, so the update is a no-op write of the same bytes -- exactly
    what the stock ``mx.where(nb_total > block, ..., pooled)`` produces.
    """

    head_dim = check_pooled_row_shapes(
        raw_keys, pooled, norm_weight, inv_freq, compress_ratio=compress_ratio
    )
    rotary_dim = 2 * int(inv_freq.shape[0])
    kernel = _pooled_row_kernel(
        head_dim,
        rotary_dim,
        int(compress_ratio),
        int(step_rows),
        int(pooled.shape[1]),
        float(eps),
        float(attention_scaling),
        raw_keys.dtype,
    )
    merged, safe_block = kernel(
        inputs=[
            raw_keys,
            pooled,
            norm_weight,
            inv_freq,
            _as_i32_scalar(offset, "offset"),
        ],
        template=[("T", raw_keys.dtype)],
        grid=(SIMD, 1, 1),
        threadgroup=(SIMD, 1, 1),
        output_shapes=[(1, 1, head_dim), (1,)],
        output_dtypes=[raw_keys.dtype, mx.int32],
    )
    return merged, safe_block.reshape(())


def check_pooled_row_shapes(
    raw_keys: mx.array,
    pooled: mx.array,
    norm_weight: mx.array,
    inv_freq: mx.array,
    *,
    compress_ratio: int,
) -> int:
    """Validate the pooled-row contract and return head_dim.  Raises."""

    if raw_keys.ndim != 3 or int(raw_keys.shape[0]) != 1:
        raise ValueError(f"raw_keys must be [1,cap,D]; got {tuple(raw_keys.shape)}")
    if pooled.ndim != 3 or int(pooled.shape[0]) != 1:
        raise ValueError(f"pooled must be [1,blocks,D]; got {tuple(pooled.shape)}")
    head_dim = int(raw_keys.shape[2])
    if int(pooled.shape[2]) != head_dim:
        raise ValueError(
            "raw_keys and pooled must share head_dim; got "
            f"{head_dim} and {int(pooled.shape[2])}"
        )
    if raw_keys.dtype not in _SUPPORTED_DTYPES or pooled.dtype != raw_keys.dtype:
        raise TypeError(
            "the M4 pooled row is exact only for float16/bfloat16 banks of one "
            f"dtype; got raw={raw_keys.dtype}, pooled={pooled.dtype}"
        )
    if norm_weight.ndim != 1 or int(norm_weight.shape[0]) != head_dim:
        raise ValueError(
            f"norm_weight must be [{head_dim}]; got {tuple(norm_weight.shape)}"
        )
    if norm_weight.dtype != raw_keys.dtype:
        raise TypeError(
            "exact fused RMSNorm requires the norm weight to match the bank "
            f"dtype; got {norm_weight.dtype} and {raw_keys.dtype}"
        )
    if inv_freq.ndim != 1 or inv_freq.dtype != mx.float32:
        raise TypeError("inv_freq must be a 1-D float32 array")
    rotary_dim = 2 * int(inv_freq.shape[0])
    if not (0 < rotary_dim <= head_dim) or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim=2*len(inv_freq) must be even and <= head_dim; got "
            f"{rotary_dim} for head_dim={head_dim}"
        )
    if not (0 < head_dim <= MAX_EXACT_HEAD_DIM):
        raise ValueError(
            f"head_dim must be in [1,{MAX_EXACT_HEAD_DIM}]; got {head_dim}"
        )
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ValueError(f"compress_ratio must be positive; got {ratio}")
    if int(raw_keys.shape[1]) < ratio:
        raise ValueError("raw_keys capacity is smaller than one block")
    return head_dim


# ---------------------------------------------------------------------------
# 2 -- scoring epilogue: relu -> head sum -> scale -> mask -> tie-break
# ---------------------------------------------------------------------------
_SRC_INDEX_SCORES = r"""
    const uint gid = thread_position_in_grid.x;
    if (gid >= ROWS * BLOCKS) return;
    const uint b = gid % BLOCKS;
    const uint s = gid / BLOCKS;

    // scores is the [1, ROWS, HEADS, BLOCKS] fp32 GEMM output, row-major.
    float acc = 0.0f;
    for (uint h = 0; h < HEADS; ++h) {
        const float v = scores[((size_t)s * HEADS + h) * BLOCKS + b];
        // mx.maximum(x, 0.0): NaN propagates, and -0.0 loses to +0.0.
        const float t = metal::isnan(v) ? v : (v > 0.0f ? v : 0.0f);
        acc += t;
    }
    const float value = acc / SQRT_HEAD_DIM;

    const int qpos = offset[0] + int(s);
    const int visible_blocks = (qpos + 1) / int(RATIO);
    const float masked = (int(b) < visible_blocks)
        ? value : as_type<float>(0xff800000u);
    out[gid] = masked - float(b) * TIE_STEP;
"""


@lru_cache(maxsize=64)
def _index_scores_kernel(
    rows: int,
    heads: int,
    blocks: int,
    ratio: int,
    head_dim: int,
    tie_step: float,
):
    header = f"""
        #include <metal_stdlib>
        using namespace metal;
        constant constexpr uint ROWS = {rows};
        constant constexpr uint HEADS = {heads};
        constant constexpr uint BLOCKS = {blocks};
        constant constexpr uint RATIO = {ratio};
        constant constexpr float SQRT_HEAD_DIM = {_f32_literal(math.sqrt(head_dim))};
        constant constexpr float TIE_STEP = {_f32_literal(tie_step)};
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_m4_index_scores_r{rows}_h{heads}_b{blocks}_"
            f"c{ratio}_d{head_dim}"
        ),
        input_names=["scores", "offset"],
        output_names=["out"],
        header=header,
        source=_SRC_INDEX_SCORES,
        ensure_row_contiguous=True,
    )


def qsa_m4_index_scores(
    scores: mx.array,
    offset,
    *,
    compress_ratio: int,
    head_dim: int,
    tie_step: float = 1e-12,
) -> mx.array:
    """Fold the score epilogue into one dispatch; returns ``[rows, blocks]``.

    ``scores`` is the raw ``[1, rows, heads, blocks]`` fp32 GEMM output.
    """

    rows, heads, blocks = check_index_scores_shapes(scores)
    kernel = _index_scores_kernel(
        rows, heads, blocks, int(compress_ratio), int(head_dim), float(tie_step)
    )
    grid, group = _grid_1d(rows * blocks)
    return kernel(
        inputs=[scores, _as_i32_scalar(offset, "offset")],
        grid=(grid, 1, 1),
        threadgroup=(group, 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )[0]


def check_index_scores_shapes(scores: mx.array) -> tuple[int, int, int]:
    """Validate the score epilogue contract and return (rows, heads, blocks)."""

    if scores.ndim != 4 or int(scores.shape[0]) != 1:
        raise ValueError(
            f"scores must be [1,rows,heads,blocks]; got {tuple(scores.shape)}"
        )
    if scores.dtype != mx.float32:
        raise TypeError(
            "the score epilogue reproduces the fp32 reduce of the stock chain; "
            f"got {scores.dtype}"
        )
    rows, heads, blocks = (int(s) for s in scores.shape[1:])
    if rows <= 0 or heads <= 0 or blocks <= 0:
        raise ValueError(f"scores dims must be positive; got {tuple(scores.shape)}")
    return rows, heads, blocks


# ---------------------------------------------------------------------------
# 3 -- rows-gather token build (bit-exact: integers and booleans only)
# ---------------------------------------------------------------------------
_SRC_ROW_TOKENS = r"""
    const uint gid = thread_position_in_grid.x;
    if (gid >= ROWS * WIDTH) return;
    const uint c = gid % WIDTH;
    const uint s = gid / WIDTH;

    const int qpos = offset[0] + int(s);
    const int visible_blocks = (qpos + 1) / int(RATIO);

    int token;
    bool ok;
    if (c < TOPK * RATIO) {
        const uint slot = c / RATIO;
        const uint within = c % RATIO;
        const int block = int(top_idx[
            (size_t)s * top_idx_strides[0] +
            (size_t)slot * top_idx_strides[1]]);
        token = block * int(RATIO) + int(within);
        ok = block < visible_blocks;
    } else {
        const uint within = c - TOPK * RATIO;
        token = visible_blocks * int(RATIO) + int(within);
        ok = token <= qpos;
    }
    token_ok[gid] = ok;
    token_idx[gid] = ok ? token : 0;
"""


@lru_cache(maxsize=64)
def _row_tokens_kernel(rows: int, topk: int, ratio: int, index_dtype: mx.Dtype):
    width = topk * ratio + ratio
    header = f"""
        #include <metal_stdlib>
        using namespace metal;
        constant constexpr uint ROWS = {rows};
        constant constexpr uint TOPK = {topk};
        constant constexpr uint RATIO = {ratio};
        constant constexpr uint WIDTH = {width};
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_m4_row_tokens_r{rows}_k{topk}_c{ratio}_"
            f"{'u32' if index_dtype == mx.uint32 else 'i32'}"
        ),
        input_names=["top_idx", "offset"],
        output_names=["token_idx", "token_ok"],
        header=header,
        source=_SRC_ROW_TOKENS,
        ensure_row_contiguous=False,
    )


def qsa_m4_row_tokens(
    top_idx: mx.array,
    offset,
    *,
    compress_ratio: int,
) -> tuple[mx.array, mx.array]:
    """One dispatch for the rows-gather token list.  Bit-exact.

    Returns ``(token_idx [rows, topk*ratio + ratio] int32, token_ok bool)`` --
    the same two arrays the stock take_along_axis / repeat / concatenate /
    where chain builds, computed in closed form.
    """

    rows, topk = check_row_tokens_shapes(top_idx)
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ValueError(f"compress_ratio must be positive; got {ratio}")
    width = topk * ratio + ratio
    kernel = _row_tokens_kernel(rows, topk, ratio, top_idx.dtype)
    grid, group = _grid_1d(rows * width)
    return tuple(
        kernel(
            inputs=[top_idx, _as_i32_scalar(offset, "offset")],
            grid=(grid, 1, 1),
            threadgroup=(group, 1, 1),
            output_shapes=[(rows, width), (rows, width)],
            output_dtypes=[mx.int32, mx.bool_],
        )
    )


def check_row_tokens_shapes(top_idx: mx.array) -> tuple[int, int]:
    """Validate the token-build contract and return (rows, topk).  Raises."""

    if top_idx.ndim != 2:
        raise ValueError(f"top_idx must be [rows,topk]; got {tuple(top_idx.shape)}")
    if top_idx.dtype not in (mx.uint32, mx.int32):
        raise TypeError(
            f"top_idx must be uint32 or int32 block ids; got {top_idx.dtype}"
        )
    rows, topk = int(top_idx.shape[0]), int(top_idx.shape[1])
    if rows <= 0 or topk <= 0:
        raise ValueError(f"top_idx dims must be positive; got {tuple(top_idx.shape)}")
    return rows, topk
