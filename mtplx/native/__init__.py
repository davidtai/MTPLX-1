"""Python surface for MTPLX's native (CMake + nanobind) MLX primitives.

Today this is one kernel: :func:`qsa_sparse_gqa`, the direct-index sparse-GQA
attention ported from oMLX (see
``native_extensions/qsa_sparse_gqa/sparse_gqa/steel_qsa_sparse_gqa.h`` for
provenance).  It is Steel MMA, so it cannot live in ``mx.fast.metal_kernel``
(the Laguna full-port verdict: no ``mlx::steel`` MMA reachable from
``metal_kernel``) and has to be a real MLX primitive in a built extension.

Phase 1 is standalone: nothing in ``mtplx/models/qwen4_exp.py`` calls this yet.
The gate order below deliberately mirrors
``mtplx.kernels.qsa_prefill_flash._unsupported_reason`` so the two attention
consumers refuse the same shapes for the same stated reason.

Build (CPU-only; no Metal execution)::

    cd native_extensions/qsa_sparse_gqa
    cmake -S . -B build \\
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/mtplx_native_qsa/ \\
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \\
      -DPython_EXECUTABLE=<venv>/bin/python
    cmake --build build -j 8

``python setup.py build_ext --inplace`` is the same build through setuptools;
it needs ``setuptools`` in the venv, which the current qwen38 venv does not
have (which is also why ``verify_mlp`` has no built artifact on this box).
"""

from __future__ import annotations

import math
import operator
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

__all__ = [
    "native_qsa_available",
    "qsa_sparse_gqa",
    "qsa_sparse_gqa_supported",
    "qsa_sparse_gqa_unsupported_reason",
]

# Production Qwen3.8 Flash-Next QSA geometry.  These are the ONLY shapes the
# kernel is instantiated for; everything else fails closed rather than
# silently changing the attention algorithm.
_BATCH = 1
_Q_HEADS = 24
_KV_HEADS = 2
_GQA = 12
_HEAD_DIM = 256
_COMPRESS_RATIO = 4
_TOP_K_BLOCKS = 512
_MAX_CONTEXT = 1_048_576
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)
_SUPPORTED_ID_DTYPES = (mx.int32, mx.uint32)
#: (key_tile, dimension_tile) pairs the metallib instantiates.
_SUPPORTED_TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
_DEFAULT_TILE = (128, 32)


def _extension_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "native_extensions"
        / "qsa_sparse_gqa"
    )


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    """Import the built extension, or return the import error."""

    native_path = str(_extension_path())
    if native_path not in sys.path:
        sys.path.insert(0, native_path)
    try:
        import mtplx_native_qsa  # noqa: PLC0415

        return mtplx_native_qsa
    except Exception as exc:  # pragma: no cover - depends on build state
        return exc


def native_qsa_available() -> bool:
    """True when the built extension imports."""

    return not isinstance(_load_extension(), Exception)


def _on_metal_device() -> bool:
    """Metal availability is insufficient when MLX currently targets CPU."""

    try:
        return mx.metal.is_available() and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _normalized_block_ids(block_ids: mx.array, rows: int) -> mx.array | None:
    """Accept the selector's ``[S, 512]`` or the kernel ABI's ``[1,1,S,512]``.

    ``_select_eager`` emits ``[S, 512]``; the kernel wants ``[1, 1, S, 512]``.
    The reshape is a view on the contiguous selector output, not a copy, and
    the int32 dtype is accepted natively (the metallib instantiates both
    int32 and uint32) so the lane never pays an 8 MB astype per layer.
    """

    if block_ids.ndim == 2:
        if tuple(int(x) for x in block_ids.shape) != (rows, _TOP_K_BLOCKS):
            return None
        return block_ids.reshape(1, 1, rows, _TOP_K_BLOCKS)
    if block_ids.ndim == 4:
        if tuple(int(x) for x in block_ids.shape) != (
            _BATCH,
            1,
            rows,
            _TOP_K_BLOCKS,
        ):
            return None
        return block_ids
    return None


def qsa_sparse_gqa_unsupported_reason(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
) -> str | None:
    """``None`` when the call is on contract, else the precise reason."""

    extension = _load_extension()
    if isinstance(extension, Exception):
        return f"the native QSA extension is not built ({extension})"
    if not _on_metal_device():
        return "the active MLX device is not an available Metal GPU"

    arrays = (queries, keys, values, block_ids)
    if any(not isinstance(array, mx.array) for array in arrays):
        return "all tensor inputs must be MLX arrays"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "Q, K, and V must be rank four"
    if block_ids.ndim not in (2, 4):
        return "block ids must be rank two [S, 512] or rank four [1, 1, S, 512]"

    batch, query_heads, rows, head_dim = (int(x) for x in queries.shape)
    if (batch, query_heads, head_dim) != (_BATCH, _Q_HEADS, _HEAD_DIM):
        return "Q must have production shape [1, 24, S, 256]"
    if rows <= 0:
        return "Q must carry at least one query row"

    key_batch, kv_heads, capacity, key_dim = (int(x) for x in keys.shape)
    if (key_batch, kv_heads, key_dim) != (_BATCH, _KV_HEADS, _HEAD_DIM):
        return "K must have production shape [1, 2, capacity, 256]"
    if tuple(int(x) for x in values.shape) != tuple(int(x) for x in keys.shape):
        return "V must have the same full-backing shape as K"

    if queries.dtype not in _SUPPORTED_DTYPES:
        return "Q must be float16 or bfloat16"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "Q, K, and V dtypes must match"
    if block_ids.dtype not in _SUPPORTED_ID_DTYPES:
        return "block ids must be int32 or uint32"
    if _normalized_block_ids(block_ids, rows) is None:
        return "block ids must have shape [S, 512] or [1, 1, S, 512]"

    # Host scalars only: a traced scalar would make these comparisons
    # synchronize the graph.  Same contract as qsa_prefill_flash.
    if isinstance(pos_start, mx.array) or isinstance(total_tokens, mx.array):
        return "pos_start and total_tokens must be host integers"
    if isinstance(scale, mx.array):
        return "scale must be a host float"
    if isinstance(pos_start, bool) or isinstance(total_tokens, bool):
        return "pos_start and total_tokens cannot be bool"
    try:
        pos_start_i = operator.index(pos_start)
        total_tokens_i = operator.index(total_tokens)
    except TypeError:
        return "pos_start and total_tokens must be exact host integers"
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        return "scale must be a numeric host scalar"
    scale_f = float(scale)

    if pos_start_i < 0 or total_tokens_i <= 0:
        return "positions must describe a non-empty non-negative suffix"
    if pos_start_i + rows > total_tokens_i:
        return "Q must be a causal suffix inside total_tokens"
    if total_tokens_i > capacity:
        return "the logical token count exceeds the full K/V backing capacity"
    if total_tokens_i > _MAX_CONTEXT:
        return "the logical token count exceeds the production context limit"
    if total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS:
        return "the context has not crossed the dense/sparse boundary"
    if not math.isfinite(scale_f):
        return "scale must be finite"

    if (int(key_tile), int(dimension_tile)) not in _SUPPORTED_TILES:
        return (
            "(key_tile, dimension_tile) must be one of "
            + ", ".join(str(t) for t in _SUPPORTED_TILES)
        )
    return None


def qsa_sparse_gqa_supported(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
) -> bool:
    """Whether the exact production-only kernel contract is met."""

    return (
        qsa_sparse_gqa_unsupported_reason(
            queries,
            keys,
            values,
            block_ids,
            pos_start=pos_start,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dimension_tile,
        )
        is None
    )


def qsa_sparse_gqa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
    key_tile: int = _DEFAULT_TILE[0],
    dimension_tile: int = _DEFAULT_TILE[1],
    stream: Any = None,
) -> mx.array:
    """Direct-index sparse GQA attention over chronological QSA block ids.

    ``queries``  ``[1, 24, S, 256]``  fp16/bf16; the ``[B,H,S,D]`` transposed
                 view the Attention module already builds.
    ``keys``/``values``  ``[1, 2, capacity, 256]`` -- the FULL KV cache
                 backing, read in place at its allocation stride.  Never slice
                 it to ``total_tokens`` first: that copy is the whole context.
    ``block_ids``  ``[S, 512]`` int32 (``_select_eager``'s ``flash_prefill``
                 output) or ``[1, 1, S, 512]``.  Chronological, and the valid
                 entries must occupy the leading
                 ``min(512, (pos + 1) // 4)`` slots of each row -- which is
                 what the selector produces, because it sorts the raw top-k
                 ascending and validity there is the threshold predicate
                 ``id < complete_blocks``.  The kernel derives validity from
                 that invariant instead of reading ``block_valid``; the
                 standalone harness asserts it.
    ``total_tokens``  logical tokens in the cache (NOT ``capacity``).

    Returns ``[1, 24, S, 256]``, same dtype as ``queries``.

    Numerics: fp32 online softmax (exp2) and fp32 P@V over the same visible
    set as the dense lane -- a rounding-class difference, not an exactness
    one.  See ``scripts/fable/micro_qsa_sparse_gqa.py`` for the tolerance
    statement and the measured deltas.
    """

    reason = qsa_sparse_gqa_unsupported_reason(
        queries,
        keys,
        values,
        block_ids,
        pos_start=pos_start,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dimension_tile,
    )
    if reason is not None:
        raise ValueError(f"[mtplx.native.qsa_sparse_gqa] {reason}.")

    extension = _load_extension()
    selected = _normalized_block_ids(block_ids, int(queries.shape[2]))
    return extension.qsa_sparse_gqa_attention(
        queries,
        keys,
        values,
        selected,
        float(scale),
        int(pos_start),
        int(total_tokens),
        int(key_tile),
        int(dimension_tile),
        stream=stream,
    )
