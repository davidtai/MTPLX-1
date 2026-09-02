"""Split-K native sparse-GQA attention for the Qwen3.8 QSA DECODE lanes.

MTPLX_FABLE_QSA_SPARSE_DECODE (M=4 fixed verify) and
MTPLX_FABLE_QSA_SPARSE_DRAFT (M=1 single row).  The kernel itself lives in
``native_extensions/qsa_sparse_gqa`` and is reached through
``mtplx.native.qsa_sparse_gqa_decode``; this module is the lane: the gate,
the install probe, the engagement counters and the reference the probe
compares against.

WHAT IT REPLACES, AND WHY THE CENSUS UNDERSTATES IT
---------------------------------------------------
The retained fixed-M4 verify attends 4 query rows per QSA layer over the
indexer's selected top-512 pooled blocks.  Per layer, per verify cycle, the
shipped lane issues six dispatches:

    1  custom_kernel_..._qsa_m4_fused_kv_gather_c17408  [1050624,1,1]
    2  <MLX contiguous copy of k_sel.swapaxes(-1,-2)>   (Copy family)
    3  gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc1       [129,1,96]   scores
    4  block_softmax_float32                            [52224,1,1]
    5  <probs.astype(bf16)>
    6  gemv_t_bfloat16_bm1_bn2_sm8_sn4_tm4_tn4_nc1      [8,1,96]     P@V

Dispatch (1) materialises ``k_sel``/``v_sel`` as ``[1, 2, 4, 2052, 256]``
bf16 -- 8.40 MB each.  So per layer the lane WRITES 16.8 MB of gathered K/V,
MLX then copies 8.4 MB more for the transposed score operand, and dispatches
(3) and (6) read 8.4 MB each back.  Roughly 70 MB per layer, ~840 MB per
verify cycle, to attend 4 rows.

The dispatch census's QSA row (446 MB/cycle, 232 GB/s) does NOT show this:
its cost model prices the gather at a flat 4.19 MB and gives the score,
softmax and P@V dispatches zero bytes, and the transposed copy lands in the
Copy family.  Counted properly the QSA family moves ~1.05 GB in its 1.93
ms/cycle, i.e. about 540 GB/s -- right at this machine's measured 544 GB/s
ceiling.  **The lane is not bandwidth-starved; it is moving three times the
bytes it needs to.**  That is the thing this kernel changes: it reads the
cache rows once, in place, and never materialises them.

WHY SPLIT-K AND NOT THE PHASE-1 KERNEL
---------------------------------------
The phase-1 (prefill) kernel parallelises over query rows: grid
``(qL, kv_heads, 1)``.  At M=4 that is EIGHT threadgroups of 64 threads on a
40-core M5 Max, each walking all 2,051 selected keys.  Phase 1's own design
note priced the fix as its own item (docs/perf/qsa-sparse-gqa-phase2-wiring.md
section 4), and MTPLX has the general finding already: a hand-written
metal_kernel SDPA lost to stock at long N precisely because MLX's production
SDPA switches to a KV-split two-pass path there.  So decode gets the KV-split
variant: ``(qL, kv_heads, n_splits)`` threadgroups accumulating independent
online-softmax states, then a merge pass.

NUMERICS -- ROUNDING CLASS, HUMANEVAL-GATED
--------------------------------------------
The visible set is IDENTICAL to the shipped lane's, slot for slot (the
kernel applies the shipped predicate ``block < (pos+1)//4`` to every slot of
``top_idx``, which is what ``qsa_m4_row_tokens`` does; it makes no ordering
assumption, because ``_select_m4`` hands through ``mx.argpartition``'s raw
output unsorted).  The ARITHMETIC is not the same: fp32 online softmax in
exp2, fp32 probabilities into an fp32 P@V, Steel-MMA reassociation of the
256-term score contraction, and one split-K rescale per row.

So this lane is adopted on the same terms as ``MTPLX_FABLE_HC_M4``: greedy
token agreement plus a full HumanEval run, never on a digest.  The install
probe below is a numerical SANITY gate, not the quality gate -- it exists so
an armed flag that is quietly wrong disables itself instead of shipping.

GATE DISCIPLINE
---------------
* CONTRACT failure RAISES.  An armed flag on a pack the kernel cannot serve
  is a configuration error, not a reason to run the stock chain quietly --
  that is how MTPLX_FUSED_HC_V3 came to be armed-but-dead at M=4.
* PARITY failure DISABLES for the process, records the measured deltas, and
  lets the stock lane serve.  A parity miss is a numerical verdict about a
  rounding-class kernel; raising there would turn a measurement into an
  outage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx

#: Production Qwen3.8 Flash-Next QSA geometry -- compiled into the metallib.
Q_HEADS = 24
KV_HEADS = 2
GQA = 12
HEAD_DIM = 256
TOP_K = 512
COMPRESS_RATIO = 4
VERIFY_ROWS = 4
DRAFT_ROWS = 1
#: The kernel's own selected-token width; see ``mtplx.native`` for why this is
#: one less than the shipped lane's 2,052 and why the visible sets still agree.
SELECTED_TOKENS = TOP_K * COMPRESS_RATIO + (COMPRESS_RATIO - 1)

#: bf16 has an 8-bit significand, so its relative spacing is 2**-8.
_BF16_REL_ULP = 2.0**-8
#: Install-probe gates.  Deliberately loose: this is a sanity gate on a
#: rounding-class kernel, and the QUALITY gate is HumanEval + greedy-token
#: agreement at the model level.  A kernel that is merely reassociated lands
#: at a few ulp; one that is wrong lands orders of magnitude away.
PARITY_MAX_ABS_ULPS = 8.0
PARITY_MAX_REL_L2 = 2.0e-3
#: Fraction of (head, row) pairs whose argmax over the head dimension must
#: still agree.  A coarse discrete statistic, reported for continuity with
#: the "top-1 agreement" the program asks for at the kernel level; the
#: DECIDING top-1 number is model-level greedy-token agreement.
PARITY_MIN_TOP1 = 0.98

#: Probe geometry: capacity only has to clear the dense/sparse crossover
#: (``total // 4 > 512``), and the split geometry does not depend on context
#: length at all -- it is a function of SELECTED_TOKENS and the tile -- so a
#: 4,096-token probe exercises the same grid a 17,408-token cycle does.
PROBE_CAPACITY = 4096

_COUNTS: Dict[str, int] = {
    "verify_kernel": 0,
    "draft_kernel": 0,
    "probe_runs": 0,
    "probe_failures": 0,
}

#: ``None`` until the probe has run.  ``""`` once it has passed.  A non-empty
#: string is the reason the lane is disabled for this process.
_DISABLED_REASON: Optional[str] = None
_PROBE_REPORT: Dict[str, Any] = {}


def engagement() -> Dict[str, Any]:
    """Snapshot of the lane's engagement counters and install verdict.

    ``verify_kernel``/``draft_kernel`` are the ENGAGEMENT LINE: if an ABBA
    reports a win and these are zero, the win came from somewhere else.
    """

    report = dict(_COUNTS)
    report["installed"] = _DISABLED_REASON == ""
    report["disabled_reason"] = _DISABLED_REASON or None
    report["probe"] = dict(_PROBE_REPORT)
    return report


def disabled_reason() -> Optional[str]:
    """The reason the lane is off, or ``None`` while it is usable/pending."""

    return _DISABLED_REASON or None


def reset_for_tests() -> None:
    """Clear the process verdict and counters.  Tests only."""

    global _DISABLED_REASON
    _DISABLED_REASON = None
    _PROBE_REPORT.clear()
    for key in _COUNTS:
        _COUNTS[key] = 0


# ---------------------------------------------------------------------------
# The visible set -- one definition, shared by the kernel model, the
# reference and the tests.
# ---------------------------------------------------------------------------
def visible_block_count(q_abs: int) -> int:
    """``visible_blocks`` exactly as ``_SRC_ROW_TOKENS`` computes it."""

    return (int(q_abs) + 1) // COMPRESS_RATIO


def shipped_row_tokens(
    top_idx_row, q_abs: int, *, topk: int = TOP_K
) -> Tuple[list, list]:
    """Host model of ``qsa_m4_row_tokens`` for ONE row.  Integers only.

    Returns ``(token_idx, token_ok)`` of width ``topk*ratio + ratio``, the
    closed form the shipped Metal kernel writes:

        slot < topk*ratio : block = top_idx[slot // ratio]
                            token = block*ratio + slot % ratio
                            ok    = block < visible_blocks
        otherwise         : token = visible_blocks*ratio + (slot - topk*ratio)
                            ok    = token <= q_abs

    Note ``ok`` is evaluated PER SLOT against the block id, not against a
    prefix length: ``top_idx`` is ``mx.argpartition``'s output and is not
    sorted.
    """

    ratio = COMPRESS_RATIO
    visible = visible_block_count(q_abs)
    idx: list = []
    ok: list = []
    for slot in range(topk * ratio):
        block = int(top_idx_row[slot // ratio])
        token = block * ratio + (slot % ratio)
        good = block < visible
        ok.append(good)
        idx.append(token if good else 0)
    for within in range(ratio):
        token = visible * ratio + within
        good = token <= int(q_abs)
        ok.append(good)
        idx.append(token if good else 0)
    return idx, ok


def kernel_row_tokens(
    top_idx_row, q_abs: int, *, key_length: int, topk: int = TOP_K
) -> list:
    """Host model of the SPLIT KERNEL's per-slot selection for ONE row.

    Returns the kernel's ``selected[]`` array: the absolute key position for
    each of the ``topk*ratio + ratio - 1`` slots, or ``-1`` for a masked one.
    Written to be readable against the MSL in
    ``steel_qsa_sparse_gqa_decode.h``; ``tests/test_fable_qsa_sparse_decode.py``
    pins it against :func:`shipped_row_tokens`.
    """

    ratio = COMPRESS_RATIO
    visible = visible_block_count(q_abs)
    out: list = []
    for slot in range(topk * ratio):
        block = int(top_idx_row[slot // ratio])
        pos = -1
        if 0 <= block < visible:
            candidate = block * ratio + (slot % ratio)
            if 0 <= candidate < int(key_length):
                pos = candidate
        out.append(pos)
    for within in range(ratio - 1):
        candidate = visible * ratio + within
        pos = -1
        if 0 <= candidate < int(key_length) and candidate <= int(q_abs):
            pos = candidate
        out.append(pos)
    return out


def visible_sets_agree(
    top_idx_row, q_abs: int, *, key_length: int, topk: int = TOP_K
) -> bool:
    """True when kernel and shipped lane attend the SAME multiset of keys."""

    idx, ok = shipped_row_tokens(top_idx_row, q_abs, topk=topk)
    shipped = sorted(t for t, good in zip(idx, ok) if good)
    kernel = sorted(p for p in kernel_row_tokens(
        top_idx_row, q_abs, key_length=key_length, topk=topk
    ) if p >= 0)
    return shipped == kernel


# ---------------------------------------------------------------------------
# The reference the probe compares against: the shipped rows-gather lane,
# transcribed.
# ---------------------------------------------------------------------------
def _row_tokens_mx(top_idx: mx.array, q_offset, *, topk: int) -> Tuple[mx.array, mx.array]:
    """``qsa_m4_row_tokens``' closed form in plain MLX, for the reference."""

    ratio = COMPRESS_RATIO
    rows = int(top_idx.shape[0])
    offsets = mx.arange(rows, dtype=mx.int32)
    qpos = (
        q_offset.reshape(1).astype(mx.int32) + offsets
        if isinstance(q_offset, mx.array)
        else mx.array(int(q_offset), dtype=mx.int32) + offsets
    )
    visible = (qpos + 1) // ratio  # [rows]
    within = mx.arange(ratio, dtype=mx.int32)
    blocks = top_idx.astype(mx.int32)  # [rows, topk]
    block_tokens = (blocks[:, :, None] * ratio + within).reshape(rows, topk * ratio)
    block_ok = mx.repeat(blocks < visible[:, None], ratio, axis=1)
    tail_tokens = visible[:, None] * ratio + within
    tail_ok = tail_tokens <= qpos[:, None]
    token_idx = mx.concatenate([block_tokens, tail_tokens], axis=1)
    token_ok = mx.concatenate([block_ok, tail_ok], axis=1)
    token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
    return token_idx, token_ok


def stock_reference(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    top_idx: mx.array,
    *,
    query_offset,
    scale: float,
    topk: int = TOP_K,
) -> mx.array:
    """The shipped rows-gather attention, over the SAME visible set.

    Transcribed from ``mtplx/models/qwen4_exp.py::_qsa_rows_gather_attention``
    (score GEMM -> ``-inf`` on invalid -> fp32 softmax -> bf16 probabilities
    -> P@V) so the probe compares against ONE definition and does not import
    the model into a kernel module.  ``keys``/``values`` are the full
    ``[1, 2, capacity, 256]`` backing; every gathered index is an absolute
    row inside it, exactly as in the shipped lane.
    """

    token_idx, token_ok = _row_tokens_mx(top_idx, query_offset, topk=topk)
    rows = int(queries.shape[2])
    width = int(token_idx.shape[1])
    k_sel = mx.take(keys, token_idx.reshape(-1), axis=2).reshape(
        1, KV_HEADS, rows, width, HEAD_DIM
    )
    v_sel = mx.take(values, token_idx.reshape(-1), axis=2).reshape(
        1, KV_HEADS, rows, width, HEAD_DIM
    )
    neg = mx.array(-mx.inf, dtype=mx.float32)
    q_view = queries.reshape(1, KV_HEADS, GQA, rows, 1, HEAD_DIM)
    k_view = k_sel.swapaxes(-1, -2).reshape(1, KV_HEADS, 1, rows, HEAD_DIM, width)
    scores = mx.matmul(q_view, k_view).squeeze(-2).astype(mx.float32) * scale
    scores = mx.where(token_ok[None, None, None], scores, neg)
    probs = mx.softmax(scores, axis=-1).astype(queries.dtype)
    v_view = v_sel.reshape(1, KV_HEADS, 1, rows, width, HEAD_DIM)
    out = mx.matmul(probs[..., None, :], v_view).squeeze(-2)
    return out.reshape(1, Q_HEADS, rows, HEAD_DIM)


# ---------------------------------------------------------------------------
# Contract + install
# ---------------------------------------------------------------------------
def check_cache_contract(keys: mx.array, values: mx.array, ratio: int) -> None:
    """The cache half of the lane's contract.  RAISES; never returns False."""

    if int(ratio) != COMPRESS_RATIO:
        raise RuntimeError(
            "MTPLX_FABLE_QSA_SPARSE_DECODE is wired for the ratio-4 QSA lane; "
            f"got ratio={ratio}"
        )
    if not mx.metal.is_available():
        raise RuntimeError(
            "MTPLX_FABLE_QSA_SPARSE_DECODE is a Metal kernel and has no "
            "portable spelling"
        )
    from mtplx.native import native_qsa_available

    if not native_qsa_available():
        raise RuntimeError(
            "MTPLX_FABLE_QSA_SPARSE_DECODE requires the built native "
            "extension (native_extensions/qsa_sparse_gqa); build it with the "
            "cmake command in mtplx/native/__init__.py's docstring"
        )
    for name, arr in (("keys", keys), ("values", values)):
        if arr is None:
            raise RuntimeError(
                f"MTPLX_FABLE_QSA_SPARSE_DECODE requires a materialized {name} "
                "bank"
            )
        if arr.ndim != 4 or tuple(int(x) for x in arr.shape)[:2] != (1, KV_HEADS):
            raise RuntimeError(
                "MTPLX_FABLE_QSA_SPARSE_DECODE requires a "
                f"[1, {KV_HEADS}, capacity, {HEAD_DIM}] {name} bank; got "
                f"{tuple(arr.shape)}"
            )
        if int(arr.shape[3]) != HEAD_DIM:
            raise RuntimeError(
                "MTPLX_FABLE_QSA_SPARSE_DECODE is wired for head_dim "
                f"{HEAD_DIM}; got {int(arr.shape[3])}"
            )
        if arr.dtype not in (mx.bfloat16, mx.float16):
            raise RuntimeError(
                "MTPLX_FABLE_QSA_SPARSE_DECODE is wired for bf16/fp16 K/V; "
                f"got {name} dtype {arr.dtype}"
            )
    if keys.dtype != values.dtype:
        raise RuntimeError(
            "MTPLX_FABLE_QSA_SPARSE_DECODE requires one K/V dtype; got "
            f"{keys.dtype} and {values.dtype}"
        )


def _probe_cell(dtype: mx.Dtype, rows: int, total_tokens: int, q_offset: int, seed: int):
    """One synthetic probe cell on the production geometry."""

    mx.random.seed(seed)
    nb_total = total_tokens // COMPRESS_RATIO
    queries = mx.random.normal((1, Q_HEADS, rows, HEAD_DIM)).astype(dtype)
    keys = mx.random.normal((1, KV_HEADS, PROBE_CAPACITY, HEAD_DIM)).astype(dtype)
    values = mx.random.normal((1, KV_HEADS, PROBE_CAPACITY, HEAD_DIM)).astype(dtype)
    # Deliberately UNSORTED distinct block ids, drawn from the whole logical
    # range so cells with few complete blocks really do carry invisible ids --
    # which is the case a leading-prefix validity cut would get wrong.
    ids = mx.argsort(mx.random.uniform(shape=(rows, nb_total)), axis=-1)
    top_idx = ids[:, :TOP_K].astype(mx.int32)
    return queries, keys, values, top_idx, total_tokens, q_offset


def _compare(reference: mx.array, candidate: mx.array) -> Dict[str, float]:
    """Host-side parity statistics.  One eval, at install, never in the hot path."""

    ref = reference.astype(mx.float32)
    got = candidate.astype(mx.float32)
    diff = mx.abs(ref - got)
    ref_absmax = mx.max(mx.abs(ref))
    l2_diff = mx.sqrt(mx.sum(diff * diff))
    l2_ref = mx.sqrt(mx.sum(ref * ref))
    top1 = mx.mean(
        (mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1)).astype(mx.float32)
    )
    stats = mx.stack(
        [mx.max(diff), mx.mean(diff), ref_absmax, l2_diff, l2_ref, top1]
    )
    mx.eval(stats)
    max_abs, mean_abs, absmax, l2d, l2r, top1_f = (float(x) for x in stats.tolist())
    scale = max(absmax, 1e-3)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "ref_absmax": absmax,
        "max_abs_ulps": max_abs / (_BF16_REL_ULP * scale),
        "rel_l2": l2d / max(l2r, 1e-12),
        "top1": top1_f,
    }


def install(
    keys: mx.array,
    values: mx.array,
    *,
    compress_ratio: int,
    verify: bool,
    draft: bool,
) -> bool:
    """Contract-check and parity-probe the lane once per process.

    Returns True when the lane is usable.  RAISES on a contract failure;
    DISABLES (returns False, records the reason) on a parity failure.  Called
    from ``TensorOffsetQSACache`` at cache install -- model build time,
    outside any ``mx.compile`` trace, exactly where ``MTPLX_FABLE_QSA_M4``
    validates its own geometry.
    """

    global _DISABLED_REASON
    if _DISABLED_REASON is not None:
        return _DISABLED_REASON == ""
    if not (verify or draft):
        return False

    check_cache_contract(keys, values, compress_ratio)

    from mtplx.native import (
        qsa_sparse_gqa_decode,
        qsa_sparse_gqa_decode_unsupported_reason,
    )
    from mtplx.runtime_options import (
        fable_qsa_sparse_decode_splits,
        fable_qsa_sparse_decode_tile,
    )

    key_tile, dim_tile = fable_qsa_sparse_decode_tile()
    key_splits = fable_qsa_sparse_decode_splits()
    scale = float(HEAD_DIM) ** -0.5
    dtype = keys.dtype

    cells = []
    if verify:
        # A long-context cell (every selected block visible) and a
        # just-past-crossover cell (some selected ids are NOT visible).
        cells.append(("verify-4096", VERIFY_ROWS, 4093, 4089, 20260902))
        cells.append(("verify-crossover", VERIFY_ROWS, 2052, 2048, 20260903))
    if draft:
        cells.append(("draft-4096", DRAFT_ROWS, 4093, 4092, 20260904))
        cells.append(("draft-crossover", DRAFT_ROWS, 2052, 2051, 20260905))

    worst: Dict[str, Any] = {}
    for name, rows, total, offset, seed in cells:
        q, k, v, top_idx, total_tokens, q_offset = _probe_cell(
            dtype, rows, total, offset, seed
        )
        reason = qsa_sparse_gqa_decode_unsupported_reason(
            q,
            k,
            v,
            top_idx,
            query_offset=q_offset,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dim_tile,
            key_splits=key_splits,
        )
        if reason is not None:
            # A contract miss on the lane's OWN synthetic production geometry
            # is a configuration error, not a numerical verdict.
            raise RuntimeError(
                f"MTPLX_FABLE_QSA_SPARSE_DECODE cannot serve its own probe "
                f"cell {name!r}: {reason}"
            )
        _COUNTS["probe_runs"] += 1
        candidate = qsa_sparse_gqa_decode(
            q,
            k,
            v,
            top_idx,
            query_offset=q_offset,
            total_tokens=total_tokens,
            scale=scale,
            key_tile=key_tile,
            dimension_tile=dim_tile,
            key_splits=key_splits,
        )
        reference = stock_reference(
            q, k, v, top_idx, query_offset=q_offset, scale=scale
        )
        stats = _compare(reference, candidate)
        stats["cell"] = name
        if not worst or stats["max_abs_ulps"] > worst.get("max_abs_ulps", 0.0):
            worst = stats
        _PROBE_REPORT[name] = stats

    _PROBE_REPORT["worst"] = worst
    _PROBE_REPORT["tile"] = [key_tile, dim_tile]
    _PROBE_REPORT["key_splits"] = key_splits

    failures = []
    if worst.get("max_abs_ulps", math.inf) > PARITY_MAX_ABS_ULPS:
        failures.append(
            f"max abs diff {worst['max_abs']:.3e} = "
            f"{worst['max_abs_ulps']:.2f} bf16 ulp (limit {PARITY_MAX_ABS_ULPS})"
        )
    if worst.get("rel_l2", math.inf) > PARITY_MAX_REL_L2:
        failures.append(
            f"relative L2 {worst['rel_l2']:.3e} (limit {PARITY_MAX_REL_L2})"
        )
    if worst.get("top1", 0.0) < PARITY_MIN_TOP1:
        failures.append(
            f"head-dim top-1 agreement {worst['top1']:.4f} "
            f"(limit {PARITY_MIN_TOP1})"
        )

    if failures:
        _COUNTS["probe_failures"] += 1
        _DISABLED_REASON = (
            f"parity probe failed on cell {worst.get('cell')!r}: "
            + "; ".join(failures)
        )
        return False

    _DISABLED_REASON = ""
    return True


# ---------------------------------------------------------------------------
# Hot path
# ---------------------------------------------------------------------------
def attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    top_idx: mx.array,
    *,
    query_offset,
    total_tokens: int,
    scale: float,
    draft: bool = False,
) -> mx.array:
    """Run the split-K kernel for one QSA layer.  Raises on a contract miss.

    ``queries`` is the ``[1, 24, M, 256]`` transposed view the attention
    module already holds; ``keys``/``values`` are the FULL cache backing.
    ``top_idx`` is ``mx.argpartition``'s ``[M, 512]`` output in its own
    order.
    """

    from mtplx.native import (
        qsa_sparse_gqa_decode,
        qsa_sparse_gqa_decode_unsupported_reason,
    )
    from mtplx.runtime_options import (
        fable_qsa_sparse_decode_splits,
        fable_qsa_sparse_decode_tile,
    )

    key_tile, dim_tile = fable_qsa_sparse_decode_tile()
    key_splits = fable_qsa_sparse_decode_splits()
    reason = qsa_sparse_gqa_decode_unsupported_reason(
        queries,
        keys,
        values,
        top_idx,
        query_offset=query_offset,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dim_tile,
        key_splits=key_splits,
    )
    if reason is not None:
        raise RuntimeError(
            "MTPLX_FABLE_QSA_SPARSE_DECODE is armed but this call is off "
            f"contract: {reason}"
        )
    _COUNTS["draft_kernel" if draft else "verify_kernel"] += 1
    return qsa_sparse_gqa_decode(
        queries,
        keys,
        values,
        top_idx,
        query_offset=query_offset,
        total_tokens=total_tokens,
        scale=scale,
        key_tile=key_tile,
        dimension_tile=dim_tile,
        key_splits=key_splits,
    )


__all__ = [
    "COMPRESS_RATIO",
    "DRAFT_ROWS",
    "GQA",
    "HEAD_DIM",
    "KV_HEADS",
    "PARITY_MAX_ABS_ULPS",
    "PARITY_MAX_REL_L2",
    "PARITY_MIN_TOP1",
    "PROBE_CAPACITY",
    "Q_HEADS",
    "SELECTED_TOKENS",
    "TOP_K",
    "VERIFY_ROWS",
    "attention",
    "check_cache_contract",
    "disabled_reason",
    "engagement",
    "install",
    "kernel_row_tokens",
    "reset_for_tests",
    "shipped_row_tokens",
    "stock_reference",
    "visible_block_count",
    "visible_sets_agree",
]
