"""Faithful proposal-side Qwen 3.8 head from the final accepted challenge stack."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QWEN38_SOURCE_HEAD_REPO = "amal-david/qwen38-mtp-head-q2-q4-rerank-v1"
QWEN38_SOURCE_HEAD_REVISION = "ae6282749a52e052496dd5300b4aa441df7301e8"
QWEN38_SOURCE_HEAD_SHA256 = (
    "d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1"
)
QWEN38_SOURCE_HEAD_BYTES = 427_742_600
QWEN38_SOURCE_HEAD_FORMAT = (
    "qwen38-mtp-incumbent-q4-g64-plus-bf16-qkv-islands-v1"
)
QWEN38_COMPACT_PREFIX_ROWS = 98_304
QWEN38_COMPACT_CONTROL_START = 248_044
QWEN38_COMPACT_CONTROL_END = 248_070
QWEN38_COMPACT_REAL_ROWS = 98_330
QWEN38_COMPACT_PADDED_ROWS = 98_336
QWEN38_COMPACT_TOPK = 32
QWEN38_CLUSTER_ROWS = 8
QWEN38_CLUSTER_COUNT = QWEN38_COMPACT_PADDED_ROWS // QWEN38_CLUSTER_ROWS
QWEN38_CLUSTER_PROBES = math.ceil(QWEN38_CLUSTER_COUNT * 0.15)
QWEN38_HIDDEN_SIZE = 5_120
QWEN38_VOCAB_SIZE = 248_320
qwen38_source_counts = {
    "selector_calls": 0,
    "q_island_calls": 0,
    "k_island_calls": 0,
    "v_island_calls": 0,
    "row_top32_calls": 0,
    "e87_probe_calls": 0,
    "selected_q4_rerank_calls": 0,
}
_ROW_TOP32_KERNELS: tuple[Any, Any] | None = None
_PROBE_SELECT_KERNEL: Any | None = None
_RERANK_KERNEL: Any | None = None


class Qwen38SourceProposalError(RuntimeError):
    """The source artifact or loaded model does not match the measured route."""


@dataclass(frozen=True)
class Qwen38SourceArtifact:
    path: Path
    sha256: str
    bytes: int
    metadata: dict[str, str]


def compact_token_ids_to_full(ids: Any) -> Any:
    import mlx.core as mx

    return mx.where(
        ids < QWEN38_COMPACT_PREFIX_ROWS,
        ids,
        ids + (QWEN38_COMPACT_CONTROL_START - QWEN38_COMPACT_PREFIX_ROWS),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_qwen38_source_artifact(path: Path) -> Qwen38SourceArtifact:
    """Validate the immutable HF blob and every tensor used by C8."""

    from safetensors import safe_open

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise Qwen38SourceProposalError(f"source proposal artifact is missing: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != QWEN38_SOURCE_HEAD_BYTES:
        raise Qwen38SourceProposalError(
            f"source artifact size mismatch: {observed_bytes} != {QWEN38_SOURCE_HEAD_BYTES}"
        )
    observed_sha = _sha256(path)
    if observed_sha != QWEN38_SOURCE_HEAD_SHA256:
        raise Qwen38SourceProposalError(
            f"source artifact SHA mismatch: {observed_sha} != {QWEN38_SOURCE_HEAD_SHA256}"
        )
    required = {
        "draft_lm_head.weight": ([98_336, 320], "U32"),
        "draft_lm_head.scales": ([98_336, 80], "BF16"),
        "draft_lm_head.biases": ([98_336, 80], "BF16"),
        "precision_islands.q.weight": ([1_024, 5_120], "BF16"),
        "precision_islands.q.indices": ([1_024], "I32"),
        "precision_islands.k.weight": ([1_024, 5_120], "BF16"),
        "precision_islands.k.indices": ([1_024], "I32"),
        "precision_islands.v.weight": ([1_024, 5_120], "BF16"),
        "precision_islands.v.indices": ([1_024], "I32"),
    }
    with safe_open(path, framework="numpy") as handle:
        metadata = dict(handle.metadata() or {})
        if metadata.get("format") != QWEN38_SOURCE_HEAD_FORMAT:
            raise Qwen38SourceProposalError("source artifact format metadata mismatch")
        for name, (shape, dtype) in required.items():
            if name not in handle.keys():
                raise Qwen38SourceProposalError(f"source artifact is missing {name}")
            tensor = handle.get_slice(name)
            if tensor.get_shape() != shape or tensor.get_dtype() != dtype:
                raise Qwen38SourceProposalError(
                    f"source tensor {name} has {tensor.get_shape()}/{tensor.get_dtype()}"
                )
    return Qwen38SourceArtifact(path, observed_sha, observed_bytes, metadata)


def _compact_source_rows() -> Any:
    import mlx.core as mx

    return mx.concatenate(
        (
            mx.arange(QWEN38_COMPACT_PREFIX_ROWS, dtype=mx.int32),
            mx.arange(
                QWEN38_COMPACT_CONTROL_START,
                QWEN38_COMPACT_CONTROL_END,
                dtype=mx.int32,
            ),
            mx.arange(
                QWEN38_COMPACT_PADDED_ROWS - QWEN38_COMPACT_REAL_ROWS,
                dtype=mx.int32,
            ),
        )
    )


def _require_affine_head(head: Any, *, bits: int) -> None:
    import mlx.nn as nn

    if not isinstance(head, nn.QuantizedLinear):
        raise Qwen38SourceProposalError("proposal rerank head must be quantized")
    observed = (
        list(head.weight.shape),
        list(head.scales.shape),
        int(head.bits),
        int(head.group_size),
        str(head.mode),
    )
    expected = (
        [QWEN38_VOCAB_SIZE, QWEN38_HIDDEN_SIZE * bits // 32],
        [QWEN38_VOCAB_SIZE, QWEN38_HIDDEN_SIZE // 64],
        bits,
        64,
        "affine",
    )
    if observed != expected:
        raise Qwen38SourceProposalError(
            f"proposal rerank head mismatch: {observed!r} != {expected!r}"
        )


def _cluster_squared_distance(xf: Any, xn: Any, centres: Any) -> Any:
    return xn[..., None] - 2.0 * (xf @ centres.swapaxes(1, 2)) + (
        centres * centres
    ).sum(axis=2)[:, None, :]


def _cluster_furthest_pair(xf: Any, xn: Any) -> Any:
    import mlx.core as mx

    nodes, span, hidden = map(int, xf.shape)
    flat = xf.reshape(nodes * span, hidden)
    row_base = mx.arange(nodes, dtype=mx.int32) * span
    mean = xf.mean(axis=1)[:, None, :]
    first = mx.argmax(_cluster_squared_distance(xf, xn, mean)[..., 0], axis=1)
    centre_a = mx.take(flat, row_base + first.astype(mx.int32), axis=0)[:, None, :]
    second = mx.argmax(
        _cluster_squared_distance(xf, xn, centre_a)[..., 0], axis=1
    )
    centre_b = mx.take(flat, row_base + second.astype(mx.int32), axis=0)[:, None, :]
    return mx.concatenate((centre_a, centre_b), axis=1)


def _cluster_balanced_split(
    xf: Any,
    xn: Any,
    split: Any,
    *,
    iterations: int,
) -> Any:
    import mlx.core as mx

    nodes, span = int(xf.shape[0]), int(xf.shape[1])
    centres = _cluster_furthest_pair(xf, xn)
    order = mx.broadcast_to(mx.arange(span, dtype=mx.int32)[None, :], (nodes, span))
    for _ in range(iterations):
        distance = _cluster_squared_distance(xf, xn, centres)
        order = mx.argsort(distance[..., 0] - distance[..., 1], axis=1)
        rank = mx.argsort(order, axis=1)
        left = (rank < split[:, None]).astype(mx.float32)
        membership = mx.stack((left, 1.0 - left), axis=1)
        counts = mx.maximum(membership.sum(axis=2), 1.0)
        centres = (membership @ xf) / counts[..., None]
        mx.eval(centres, order)
    return order


def _bisecting_partition(rows: Any, *, rows_per_leaf: int = 8) -> Any:
    """Port the source's deterministic capacity-balanced bisecting 2-means."""

    import mlx.core as mx
    import numpy as np

    count, hidden = map(int, rows.shape)
    work = rows
    permutation = mx.arange(count, dtype=mx.int32)
    nodes: list[tuple[int, int, int]] = [(0, count, count // rows_per_leaf)]
    while any(leaves > 1 for _, _, leaves in nodes):
        by_span: dict[int, list[int]] = {}
        for index, (_, span, leaves) in enumerate(nodes):
            if leaves > 1:
                by_span.setdefault(span, []).append(index)
        next_order = np.arange(count, dtype=np.int32)
        cuts: dict[int, int] = {}
        for span in sorted(by_span):
            members = by_span[span]
            gather = np.concatenate(
                [np.arange(nodes[index][0], nodes[index][0] + span) for index in members]
            ).astype(np.int32)
            block = mx.take(work, mx.array(gather), axis=0).reshape(
                len(members), span, hidden
            ).astype(mx.float32)
            block_norm = (block * block).sum(axis=2)
            targets = mx.array(
                [rows_per_leaf * ((nodes[index][2] + 1) // 2) for index in members],
                dtype=mx.int32,
            )
            order = _cluster_balanced_split(
                block,
                block_norm,
                targets,
                iterations=8,
            )
            mx.eval(order)
            order_host = np.asarray(order).reshape(len(members), span)
            targets_host = np.asarray(targets)
            for member, index in enumerate(members):
                start = nodes[index][0]
                next_order[start : start + span] = start + order_host[member]
                cuts[index] = int(targets_host[member])
        reorder = mx.array(next_order)
        work = mx.take(work, reorder, axis=0)
        permutation = mx.take(permutation, reorder, axis=0)
        mx.eval(work, permutation)
        next_nodes: list[tuple[int, int, int]] = []
        for index, (start, span, leaves) in enumerate(nodes):
            if leaves <= 1:
                next_nodes.append((start, span, leaves))
                continue
            cut = cuts[index]
            next_nodes.append((start, cut, cut // rows_per_leaf))
            next_nodes.append(
                (start + cut, span - cut, leaves - cut // rows_per_leaf)
            )
        nodes = next_nodes
    return permutation


_TOP32_HEADER = r"""
    inline uint qwen_top32_ordinal(float v) {
        if (isnan(v))  { return 0xFFFFFFFFu; }
        if (v == 0.0f) { return 0x80000000u; }
        uint u = as_type<uint>(v);
        return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    }
"""


def _top32_partial_source(*, real_count: int, tiles: int) -> str:
    tg = 256
    stride = tiles * tg
    per_thread = (real_count + stride - 1) // stride
    return f"""
        constexpr uint REAL_COUNT = {real_count};
        constexpr uint TG_SIZE = {tg};
        constexpr uint STRIDE = {stride};
        constexpr uint PER_THREAD = {per_thread};
        constexpr uint TOPK = 32;
        constexpr uint NSIMD = 8;
        constexpr uint PB = 8;
        static_assert(PER_THREAD <= 32, "top32 thread slots");
        uint tile = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint ord[PER_THREAD]; uint idx[PER_THREAD];
        for (uint t = 0; t < PER_THREAD; ++t) {{ ord[t] = 0u; idx[t] = 0u; }}
        uint n = 0;
        for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {{
            ord[n] = qwen_top32_ordinal(float(logits[i])); idx[n] = i; n++;
        }}
        threadgroup uint sc_ord[256]; threadgroup uint sc_idx[256];
        uint taken = 0u;
        for (uint r = 0; r < TOPK; ++r) {{
            uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
            for (uint t = 0; t < PER_THREAD; ++t) {{
                if ((taken & (1u << t)) != 0u) continue;
                if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {{
                    bo = ord[t]; bi = idx[t]; bs = t;
                }}
            }}
            uint mo = simd_max(bo);
            uint mi = simd_max((bo == mo) ? bi : 0u);
            if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) taken |= 1u << bs;
            if (lane == 0) {{ sc_ord[sg * TOPK + r] = mo; sc_idx[sg * TOPK + r] = mi; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {{
            uint o2[PB]; uint i2[PB];
            for (uint t = 0; t < PB; ++t) {{
                uint p = t * 32u + lane; o2[t] = sc_ord[p]; i2[t] = sc_idx[p];
            }}
            uint tk2 = 0u;
            for (uint r = 0; r < TOPK; ++r) {{
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PB; ++t) {{
                    if ((tk2 & (1u << t)) != 0u) continue;
                    if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {{
                        bo = o2[t]; bi = i2[t]; bs = t;
                    }}
                }}
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) tk2 |= 1u << bs;
                if (lane == 0) {{ cand_ord[tile * TOPK + r] = mo; cand_idx[tile * TOPK + r] = mi; }}
            }}
        }}
    """


def _top32_finalize_source(*, candidates: int) -> str:
    per_thread = candidates // 256
    return f"""
        constexpr uint TG_SIZE = 256; constexpr uint PER_THREAD = {per_thread};
        constexpr uint TOPK = 32; constexpr uint NSIMD = 8; constexpr uint PB = 8;
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint ord[PER_THREAD]; uint idx[PER_THREAD];
        for (uint t = 0; t < PER_THREAD; ++t) {{
            uint p = t * TG_SIZE + tid; ord[t] = cand_ord[p]; idx[t] = cand_idx[p];
        }}
        threadgroup uint sc_ord[256]; threadgroup uint sc_idx[256];
        uint taken = 0u;
        for (uint r = 0; r < TOPK; ++r) {{
            uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
            for (uint t = 0; t < PER_THREAD; ++t) {{
                if ((taken & (1u << t)) != 0u) continue;
                if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {{ bo = ord[t]; bi = idx[t]; bs = t; }}
            }}
            uint mo = simd_max(bo); uint mi = simd_max((bo == mo) ? bi : 0u);
            if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) taken |= 1u << bs;
            if (lane == 0) {{ sc_ord[sg * TOPK + r] = mo; sc_idx[sg * TOPK + r] = mi; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {{
            uint o2[PB]; uint i2[PB];
            for (uint t = 0; t < PB; ++t) {{ uint p = t * 32u + lane; o2[t] = sc_ord[p]; i2[t] = sc_idx[p]; }}
            uint tk2 = 0u;
            for (uint r = 0; r < TOPK; ++r) {{
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PB; ++t) {{
                    if ((tk2 & (1u << t)) != 0u) continue;
                    if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {{ bo = o2[t]; bi = i2[t]; bs = t; }}
                }}
                uint mo = simd_max(bo); uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) tk2 |= 1u << bs;
                if (lane == 0) {{
                    uint cluster = probed[mi / 8u];
                    token_ids[TOPK - 1u - r] = uint(perm[cluster * 8u + (mi % 8u)]);
                }}
            }}
        }}
    """


def _row_top32(row_score: Any, probed: Any, perm: Any) -> Any:
    import mlx.core as mx

    global _ROW_TOP32_KERNELS
    tiles = 32
    candidates = tiles * 32
    if _ROW_TOP32_KERNELS is None:
        partial = mx.fast.metal_kernel(
            name="mtplx_qwen38_source_row_top32_partial_v1",
            input_names=["logits"],
            output_names=["cand_ord", "cand_idx"],
            source=_top32_partial_source(
                real_count=QWEN38_CLUSTER_PROBES * QWEN38_CLUSTER_ROWS,
                tiles=tiles,
            ),
            header=_TOP32_HEADER,
            ensure_row_contiguous=False,
        )
        finalize = mx.fast.metal_kernel(
            name="mtplx_qwen38_source_row_top32_finalize_v1",
            input_names=["cand_ord", "cand_idx", "probed", "perm"],
            output_names=["token_ids"],
            source=_top32_finalize_source(candidates=candidates),
            ensure_row_contiguous=False,
        )
        _ROW_TOP32_KERNELS = (partial, finalize)
    partial, finalize = _ROW_TOP32_KERNELS
    cand_ord, cand_idx = partial(
        inputs=[row_score],
        template=[],
        grid=(tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(candidates,), (candidates,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    (token_ids,) = finalize(
        inputs=[cand_ord, cand_idx, probed, perm],
        template=[],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(32,)],
        output_dtypes=[mx.uint32],
    )
    qwen38_source_counts["row_top32_calls"] += 1
    return token_ids


_E87_HEADER = r"""
    inline ushort qwen_e87_key16(float v) {
        if (isnan(v))  { return 0xFFFFu; }
        if (v == 0.0f) { return 0x8000u; }
        uint u = as_type<uint>(v);
        uint o = (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
        return ushort(o >> 16);
    }
"""


def _e87_probe_select(score: Any) -> Any:
    """One-dispatch E87 top-probe selection with source tie semantics."""

    import mlx.core as mx

    global _PROBE_SELECT_KERNEL
    if _PROBE_SELECT_KERNEL is None:
        _PROBE_SELECT_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_source_e87_probe_select_v1",
            input_names=["score"],
            output_names=["probed"],
            header=_E87_HEADER,
            ensure_row_contiguous=True,
            source=f"""
                constexpr uint CLUSTERS = {QWEN38_CLUSTER_COUNT};
                constexpr uint PROBES = {QWEN38_CLUSTER_PROBES};
                constexpr uint TG = 1024;
                constexpr uint PT = (CLUSTERS + TG - 1u) / TG;
                constexpr uint WORDS = (CLUSTERS + 31u) / 32u;
                constexpr uint NSIMD = TG / 32u;
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint base = tid * PT;
                threadgroup atomic_uint hist[256];
                threadgroup atomic_uint bits[WORDS];
                threadgroup uint sel[8];
                threadgroup uint sgsum[NSIMD];
                ushort key[PT];
                for (uint j = 0; j < PT; ++j) {{
                    const uint i = base + j;
                    key[j] = (i < CLUSTERS) ? qwen_e87_key16(float(score[i])) : ushort(0);
                }}
                for (uint x = tid; x < 256u; x += TG) {{
                    atomic_store_explicit(&hist[x], 0u, memory_order_relaxed);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = 0; j < PT; ++j) {{
                    if (base + j < CLUSTERS) {{
                        atomic_fetch_add_explicit(&hist[uint(key[j]) >> 8], 1u, memory_order_relaxed);
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {{
                    uint acc = 0u, b = 0u;
                    for (int x = 255; x >= 0; --x) {{
                        const uint c = atomic_load_explicit(&hist[x], memory_order_relaxed);
                        if (acc + c >= PROBES) {{ b = uint(x); break; }}
                        acc += c;
                    }}
                    sel[0] = b; sel[1] = acc;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint hi = sel[0];
                const uint k1 = PROBES - sel[1];
                for (uint x = tid; x < 256u; x += TG) {{
                    atomic_store_explicit(&hist[x], 0u, memory_order_relaxed);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = 0; j < PT; ++j) {{
                    if (base + j < CLUSTERS && (uint(key[j]) >> 8) == hi) {{
                        atomic_fetch_add_explicit(&hist[uint(key[j]) & 0xFFu], 1u, memory_order_relaxed);
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {{
                    uint acc = 0u, c = 0u;
                    for (int x = 255; x >= 0; --x) {{
                        const uint n = atomic_load_explicit(&hist[x], memory_order_relaxed);
                        if (acc + n >= k1) {{ c = uint(x); break; }}
                        acc += n;
                    }}
                    sel[2] = c; sel[3] = acc;
                    sel[4] = atomic_load_explicit(&hist[c], memory_order_relaxed);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const ushort threshold = ushort((hi << 8) | sel[2]);
                const uint k2 = k1 - sel[3];
                const uint eq = sel[4];
                uint idx_threshold = 0u;
                if (k2 < eq) {{
                    for (uint w = tid; w < WORDS; w += TG) {{
                        atomic_store_explicit(&bits[w], 0u, memory_order_relaxed);
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint j = 0; j < PT; ++j) {{
                        const uint i = base + j;
                        if (i < CLUSTERS && key[j] == threshold) {{
                            atomic_fetch_or_explicit(&bits[i >> 5], 1u << (i & 31u), memory_order_relaxed);
                        }}
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid == 0) {{
                        uint need = k2, selected = 0u;
                        for (int w = int(WORDS) - 1; w >= 0; --w) {{
                            uint value = atomic_load_explicit(&bits[w], memory_order_relaxed);
                            const uint count = popcount(value);
                            if (count >= need) {{
                                for (uint k = 0; k + 1u < need; ++k) {{
                                    value &= ~(1u << (31u - clz(value)));
                                }}
                                selected = uint(w) * 32u + (31u - clz(value));
                                break;
                            }}
                            need -= count;
                        }}
                        sel[5] = selected;
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    idx_threshold = sel[5];
                }}
                uint count = 0u;
                for (uint j = 0; j < PT; ++j) {{
                    const uint i = base + j;
                    if (i < CLUSTERS && (key[j] > threshold || (key[j] == threshold && i >= idx_threshold))) ++count;
                }}
                const uint inclusive = simd_prefix_inclusive_sum(count);
                if (lane == 31u) sgsum[sg] = inclusive;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0u) {{
                    const uint value = sgsum[lane];
                    sgsum[lane] = simd_prefix_exclusive_sum(value);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint out = sgsum[sg] + inclusive - count;
                for (uint j = 0; j < PT; ++j) {{
                    const uint i = base + j;
                    if (i < CLUSTERS && (key[j] > threshold || (key[j] == threshold && i >= idx_threshold))) {{
                        probed[out++] = i;
                    }}
                }}
            """,
        )
    (probed,) = _PROBE_SELECT_KERNEL(
        inputs=[score],
        template=[],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(QWEN38_CLUSTER_PROBES,)],
        output_dtypes=[mx.uint32],
    )
    qwen38_source_counts["e87_probe_calls"] += 1
    return probed


def _selected_q4_rerank(
    x: Any,
    candidate_ids: Any,
    weight: Any,
    scales: Any,
    biases: Any,
) -> Any:
    import mlx.core as mx

    global _RERANK_KERNEL
    if _RERANK_KERNEL is None:
        _RERANK_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen38_source_selected_q4_rerank_v1",
            input_names=["x", "candidate_ids", "weight", "scales", "biases"],
            output_names=["token_id"],
            header=r"""
                typedef bfloat16_t InT;
                inline bool qwen_rerank_better(
                    float cv, uint ci, float bv, uint bi
                ) {
                    bool cn = isnan(cv), bn = isnan(bv);
                    if (cn != bn) return !cn;
                    if (cv > bv) return true;
                    if (cv < bv) return false;
                    return ci < bi;
                }
            """,
            source=r"""
                constexpr uint TOPK = 32, K = 5120, K_WORDS = 640;
                constexpr uint K_GROUPS = 80, VALUES = 16, BLOCK = 512;
                uint lane = thread_index_in_simdgroup;
                uint sg = simdgroup_index_in_threadgroup;
                uint candidate_base = sg * 4;
                float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (uint k = 0; k < K; k += BLOCK) {
                    float xv[VALUES];
                    uint x_base = k + lane * VALUES;
                    float sum = 0.0f;
                    for (uint i = 0; i < VALUES; i += 4) {
                        sum += x[x_base + i] + x[x_base + i + 1]
                            + x[x_base + i + 2] + x[x_base + i + 3];
                        xv[i] = x[x_base + i];
                        xv[i + 1] = x[x_base + i + 1] / 16.0f;
                        xv[i + 2] = x[x_base + i + 2] / 256.0f;
                        xv[i + 3] = x[x_base + i + 3] / 4096.0f;
                    }
                    for (uint r = 0; r < 4; ++r) {
                        uint row = uint(candidate_ids[candidate_base + r]);
                        uint word_base = row * K_WORDS + k / 8 + lane * 2;
                        uint p0 = weight[word_base], p1 = weight[word_base + 1];
                        ushort packed[4] = {
                            ushort(p0 & 0xffffu), ushort(p0 >> 16),
                            ushort(p1 & 0xffffu), ushort(p1 >> 16)
                        };
                        uint group_index = row * K_GROUPS + k / 64 + lane / 4;
                        float scale = scales[group_index], bias = biases[group_index];
                        float accum = 0.0f;
                        for (uint i = 0; i < 4; ++i) {
                            accum += xv[4*i] * (packed[i] & 0x000f)
                                + xv[4*i+1] * (packed[i] & 0x00f0)
                                + xv[4*i+2] * (packed[i] & 0x0f00)
                                + xv[4*i+3] * (packed[i] & 0xf000);
                        }
                        result[r] += scale * accum + sum * bias;
                    }
                }
                threadgroup float exact_scores[TOPK];
                for (uint r = 0; r < 4; ++r) {
                    float reduced = simd_sum(result[r]);
                    if (lane == 0) exact_scores[candidate_base + r] = float(InT(reduced));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg == 0) {
                    float best_value = exact_scores[lane];
                    uint best_id = uint(candidate_ids[lane]);
                    for (uint offset = 16; offset > 0; offset >>= 1) {
                        float ov = simd_shuffle_down(best_value, offset);
                        uint oi = simd_shuffle_down(best_id, offset);
                        if (lane < offset && qwen_rerank_better(ov, oi, best_value, best_id)) {
                            best_value = ov; best_id = oi;
                        }
                    }
                    if (lane == 0) token_id[0] = int(
                        best_id < 98304u ? best_id : best_id + 149740u);
                }
            """,
            ensure_row_contiguous=False,
        )
    (token_id,) = _RERANK_KERNEL(
        inputs=[x, candidate_ids, weight, scales, biases],
        template=[],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1)],
        output_dtypes=[mx.int32],
    )
    qwen38_source_counts["selected_q4_rerank_calls"] += 1
    return token_id


class _ExactQRows:
    def __init__(self, base: Any, weight: Any, indices: Any):
        import mlx.core as mx

        self.base = base
        self.weight = weight
        self.indices = indices.astype(mx.int32)

    def __call__(self, x: Any) -> Any:
        import mlx.core as mx

        qwen38_source_counts["q_island_calls"] += 1
        base = self.base(x)
        exact = mx.matmul(x, self.weight.T)
        shape = [1] * (base.ndim - 1) + [int(self.indices.shape[0])]
        return mx.put_along_axis(
            base,
            self.indices.reshape(shape),
            exact,
            axis=-1,
        )


class _CountedDense:
    def __init__(self, layer: Any, counter: str):
        self.layer = layer
        self.counter = counter

    def __call__(self, x: Any) -> Any:
        qwen38_source_counts[self.counter] += 1
        return self.layer(x)


def _dense_linear(weight: Any, *, counter: str) -> Any:
    import mlx.nn as nn

    layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False)
    layer.weight = weight
    return _CountedDense(layer, counter)


def _candidate_mtp(text: Any, tensors: dict[str, Any]) -> Any:
    """Instantiate the source Q4 body and install its BF16 precision islands."""

    import mlx.core as mx
    import mlx.nn as nn

    control = text.mtp
    try:
        candidate = type(control)(text.args, len(control.layers))
    except TypeError as exc:
        raise Qwen38SourceProposalError(
            "loaded MTP module cannot be cloned for a matched route"
        ) from exc

    def quantization(path: str, module: Any) -> dict[str, Any] | bool:
        if not hasattr(module, "to_quantized"):
            return False
        return {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
        }

    nn.quantize(candidate, class_predicate=quantization)
    # Make FC explicit as well: its 10,240-input tensor has 160 affine groups,
    # therefore it is group-64 like the rest of this source artifact.
    candidate.fc = nn.QuantizedLinear(
        2 * QWEN38_HIDDEN_SIZE,
        QWEN38_HIDDEN_SIZE,
        bias=False,
        group_size=64,
        bits=4,
        mode="affine",
    )
    body = {
        key: value
        for key, value in tensors.items()
        if not key.startswith("draft_lm_head.")
        and not key.startswith("precision_islands.")
    }
    candidate.load_weights(list(body.items()), strict=False)
    mx.eval(candidate.parameters())

    attn = candidate.layers[0].self_attn
    q_weight = tensors["precision_islands.q.weight"]
    q_indices = tensors["precision_islands.q.indices"]
    attn.q_proj = _ExactQRows(attn.q_proj, q_weight, q_indices)
    k_order = mx.argsort(tensors["precision_islands.k.indices"])
    v_order = mx.argsort(tensors["precision_islands.v.indices"])
    k_weight = mx.take(tensors["precision_islands.k.weight"], k_order, axis=0)
    v_weight = mx.take(tensors["precision_islands.v.weight"], v_order, axis=0)
    mx.eval(q_weight, q_indices, k_weight, v_weight)
    attn.k_proj = _dense_linear(
        k_weight,
        counter="k_island_calls",
    )
    attn.v_proj = _dense_linear(
        v_weight,
        counter="v_island_calls",
    )
    return candidate


class Qwen38SourceProposalHead:
    """Source Q2 coarse shortlist followed by selected incumbent-Q4 rerank."""

    def __init__(self, incumbent_head: Any, tensors: dict[str, Any]):
        import mlx.core as mx

        _require_affine_head(incumbent_head, bits=4)
        self.incumbent_head = incumbent_head
        self.coarse_weight = tensors["draft_lm_head.weight"]
        self.coarse_scales = tensors["draft_lm_head.scales"]
        self.coarse_biases = tensors["draft_lm_head.biases"]
        rows = _compact_source_rows()
        self.exact_weight = mx.take(incumbent_head.weight, rows, axis=0)
        self.exact_scales = mx.take(incumbent_head.scales, rows, axis=0)
        self.exact_biases = mx.take(incumbent_head.biases, rows, axis=0)
        exact_rows = mx.dequantize(
            self.exact_weight,
            self.exact_scales,
            self.exact_biases,
            group_size=64,
            bits=4,
            mode="affine",
        ).astype(mx.bfloat16)
        mx.eval(exact_rows)
        order = _bisecting_partition(exact_rows)
        order = mx.sort(order.reshape(QWEN38_CLUSTER_COUNT, QWEN38_CLUSTER_ROWS), axis=1).reshape(-1)
        centroids = mx.take(exact_rows, order, axis=0).reshape(
            QWEN38_CLUSTER_COUNT,
            QWEN38_CLUSTER_ROWS,
            QWEN38_HIDDEN_SIZE,
        ).astype(mx.float32).mean(axis=1).astype(mx.bfloat16)
        centroid_weight, centroid_scales, centroid_biases = mx.quantize(
            centroids,
            group_size=64,
            bits=2,
            mode="affine",
        )
        real_count = mx.array(QWEN38_COMPACT_REAL_ROWS, dtype=mx.int32)
        self.cluster_weight = mx.take(self.coarse_weight, order, axis=0).reshape(
            QWEN38_CLUSTER_COUNT, QWEN38_CLUSTER_ROWS, 320
        )
        self.cluster_scales = mx.take(self.coarse_scales, order, axis=0).reshape(
            QWEN38_CLUSTER_COUNT, QWEN38_CLUSTER_ROWS, 80
        )
        self.cluster_biases = mx.take(self.coarse_biases, order, axis=0).reshape(
            QWEN38_CLUSTER_COUNT, QWEN38_CLUSTER_ROWS, 80
        )
        self.cluster_perm = mx.where(order >= real_count, order - real_count, order)
        self.centroid_weight = centroid_weight
        self.centroid_scales = centroid_scales
        self.centroid_biases = centroid_biases
        self.cluster_lhs = mx.zeros((QWEN38_CLUSTER_PROBES,), dtype=mx.uint32)
        mx.eval(
            self.cluster_weight,
            self.cluster_scales,
            self.cluster_biases,
            self.cluster_perm,
            self.centroid_weight,
            self.centroid_scales,
            self.centroid_biases,
            self.exact_weight,
            self.exact_scales,
            self.exact_biases,
        )
        del exact_rows, centroids, order
        mx.clear_cache()

    def __call__(self, x: Any) -> Any:
        import mlx.core as mx

        qwen38_source_counts["selector_calls"] += 1
        if list(x.shape) != [1, 1, QWEN38_HIDDEN_SIZE]:
            raise Qwen38SourceProposalError(
                f"source selector requires [1,1,5120], got {list(x.shape)}"
            )
        flat = x.reshape(1, QWEN38_HIDDEN_SIZE)
        centroid_score = mx.quantized_matmul(
            flat,
            self.centroid_weight,
            scales=self.centroid_scales,
            biases=self.centroid_biases,
            transpose=True,
            group_size=64,
            bits=2,
            mode="affine",
        )[0]
        probed = _e87_probe_select(centroid_score)
        row_score = mx.gather_qmm(
            flat.reshape(1, 1, QWEN38_HIDDEN_SIZE),
            self.cluster_weight,
            scales=self.cluster_scales,
            biases=self.cluster_biases,
            lhs_indices=self.cluster_lhs,
            rhs_indices=probed,
            transpose=True,
            group_size=64,
            bits=2,
            mode="affine",
            sorted_indices=True,
        ).reshape(-1)
        compact_ids = _row_top32(
            row_score,
            probed.astype(mx.uint32),
            self.cluster_perm,
        ).astype(mx.int32)
        winner = _selected_q4_rerank(
            flat.reshape(QWEN38_HIDDEN_SIZE),
            compact_ids.astype(mx.uint32),
            self.exact_weight,
            self.exact_scales,
            self.exact_biases,
        )
        token_axis = mx.arange(QWEN38_VOCAB_SIZE, dtype=mx.int32)
        logits = mx.where(token_axis == winner, 0.0, -mx.inf).astype(x.dtype)
        return logits.reshape(1, 1, QWEN38_VOCAB_SIZE)


def configure_qwen38_source_proposal(
    runtime: Any,
    *,
    active: bool,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Switch between the incumbent head and the source Q4+island/Q2 route."""

    text = getattr(runtime.model, "language_model", runtime.model)
    if not active and not hasattr(text, "_mtplx_qwen38_control_mtp"):
        return {
            "active": False,
            "installed": False,
            "artifact_sha256": None,
            "artifact_bytes": None,
            "precision_q_rows": 0,
            "precision_k_rows": 0,
            "precision_v_rows": 0,
            "selector": "control",
        }
    if not hasattr(text, "_mtplx_qwen38_control_mtp"):
        text._mtplx_qwen38_control_mtp = text.mtp
        text._mtplx_qwen38_control_draft_head = getattr(
            text, "_mtplx_draft_lm_head", None
        )
    control_mtp = text._mtplx_qwen38_control_mtp
    control_head = text._mtplx_qwen38_control_draft_head

    if active and not hasattr(text, "_mtplx_qwen38_source_mtp"):
        import mlx.core as mx

        if artifact_path is None:
            raise Qwen38SourceProposalError("source proposal route requires artifact_path")
        artifact = validate_qwen38_source_artifact(artifact_path)
        tensors = mx.load(str(artifact.path), format="safetensors")
        text._mtplx_qwen38_source_mtp = _candidate_mtp(text, tensors)
        text._mtplx_qwen38_source_draft_head = Qwen38SourceProposalHead(
            control_head,
            tensors,
        )
        text._mtplx_qwen38_source_artifact = artifact

    if active:
        text.mtp = text._mtplx_qwen38_source_mtp
        text._mtplx_draft_lm_head = text._mtplx_qwen38_source_draft_head
    else:
        text.mtp = control_mtp
        text._mtplx_draft_lm_head = control_head

    artifact = getattr(text, "_mtplx_qwen38_source_artifact", None)
    return {
        "active": bool(active),
        "installed": hasattr(text, "_mtplx_qwen38_source_mtp"),
        "artifact_sha256": getattr(artifact, "sha256", None),
        "artifact_bytes": getattr(artifact, "bytes", None),
        "precision_q_rows": 1_024 if active else 0,
        "precision_k_rows": 1_024 if active else 0,
        "precision_v_rows": 1_024 if active else 0,
        "selector": "cluster_q2_top32_q4_rerank" if active else "control",
    }


def qwen38_source_counter_snapshot() -> dict[str, int]:
    return dict(qwen38_source_counts)
