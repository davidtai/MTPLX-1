#!/usr/bin/env python3
"""W69: node census of the compiled fixed-M4 verify body, by block and op class.

Why this exists
---------------
One decode cycle is ``T = G + H_exposed``.  Every dispatch inside the *compiled*
verify body is replayed on the host each cycle, and W64's production-lane model
(``docs/perf/decode-critical-path.md``) prices it twice:

* **host encode** 3.22 µs per dispatch, of which only **0.379 µs is exposed**
  (88 % of encoding overlaps a busy GPU and is free);
* **GPU launch** 1.83 µs, charged in full **only when the launch is dependent**
  — a sibling launch hides under MLX's concurrent encoder.

Nobody had inventoried the body.  ``census_retained_stack`` reduces a whole
cycle by *kernel family* (bytes, GPU busy); ``census_verify_opener`` (W63)
locates the body's opening command buffer.  Neither says how many nodes are in
the body, what they are, or which of them a kernel could delete.  This module
does exactly that, from the same instrumented-MLX dispatch census.

How the body is located (measured, not assumed)
-----------------------------------------------
The verify body is one ``mx.compile``'d graph, so its dispatch sequence is
**bit-identical every cycle**.  Anchoring on the once-per-cycle target
``lm_head`` (grid ``[1,31040,1]``, the body's *last* dispatch) and walking
backwards, the body is the **longest common suffix of (kernel name, grid)
across every cycle in the file**.  That is a measurement with its own falsifier:
if the file has no such suffix the body is not fixed and this tool says so.

    w58 retained control   3,669 dispatches, 382/382 cycles
    w58 retained composed  2,751 dispatches, 394/394 cycles

3,669 reproduces W63's independently-derived "opens at a fixed offset of 3,668
dispatches before the cycle's lm_head" exactly, and 3,669 − 2,751 = 918 of the
926 dispatches W64 attributes to HC_M4 + OPDIET.  Two independent checks.

Block segmentation
------------------
Every block of the model opens with the hyper-connection read
(``custom_kernel_mtplx_qwen4_m4_hc_norm_*``), so the body cuts cleanly at those:
one block per attention, one per MoE, one for the final mixer before the head.
The block's own contents identify it — GDN fused ``in_proj`` at ``[1,2060,1]``,
QSA ``qkv`` at ``[1,1536,1]``, MoE router at ``[1,64,1]``, PLE conv, lm_head.

Usage
-----
    python scripts/fable/census_verify_nodes.py <census.jsonl> [--json out.json]
    python scripts/fable/census_verify_nodes.py <census.jsonl> --dump-body body.tsv

``--body-len N`` overrides the measured length (for a file with too few cycles).
The pass is single and streaming: the censuses run to 2 GB and only a deque of
``--window`` dispatches is ever held.

Nothing here touches MLX, Metal or the GPU.  It reads a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence

# --------------------------------------------------------------------------
# Price list (docs/perf/decode-critical-path.md §1-2, production lane)
# --------------------------------------------------------------------------
#: Exposed host encode per replayed dispatch: 1.776 ms / 4,685 dispatches.
#: This is what a *sibling* node is worth — it is charged whether or not the
#: launch is on the GPU's critical path.
EXPOSED_ENCODE_US = 0.379
#: A *dependent* zero-byte GPU launch, from the I1 micro on the queued lane.
#: Charged to G in full, and only for launches that actually serialise.
DEPENDENT_LAUNCH_US = 1.83
#: Host tape-replay cost per node for *Python-issued, uncompiled* graph
#: construction at a sync (W64's draft-loop measurement).  It does NOT apply to
#: the compiled body's replay; kept here only so the two are never confused.
UNCOMPILED_BUILD_US = 0.43
#: A group must clear this to be worth a worker.
FUNDABLE_MS = 0.30
#: The brief's softer "worth building" line.
NOTABLE_MS = 0.20

#: 1 ms/cycle removed = +1.75 tok/s (W64 §1, 2.6806 tok/window at 39.637 ms).
TOKS_PER_MS = 1.75

# --------------------------------------------------------------------------
# Census parsing
# --------------------------------------------------------------------------
#: The census schema is fixed-order, so one regex is far cheaper than
#: ``json.loads`` on ~2M lines.  Falls back to ``json.loads`` if it ever misses.
_OP_RE = re.compile(
    r'"seq":\s*(\d+),\s*"command_buffer_index":\s*(\d+).*?'
    r'"kernel_name":\s*"([^"]*)".*?'
    r'"grid":\s*\[\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\]'
)
_OP_MARK = '"record":"op"'

#: The target lm_head: grid ``[1, 31040, 1]`` = 248,320 vocab / 8 per thread.
#: Exactly one per verify cycle, and the body's last dispatch.
LM_HEAD_GRID = (1, 31040, 1)
LM_HEAD_KERNEL_SUBSTR = "affine_qmv_wide"

#: The hyper-connection read opens every block of the model.
HC_OPENER_SUBSTR = "hc_norm"


#: NamedTuple rather than a dataclass on purpose: this module is loaded by
#: ``importlib.util.spec_from_file_location`` in tests and ad-hoc reductions,
#: and ``dataclasses`` resolves field annotations through
#: ``sys.modules[cls.__module__]`` — which is ``None`` for a module that was
#: never registered.  NamedTuple has no such dependency.
class Op(NamedTuple):
    seq: int
    cb: int
    kernel: str
    grid: tuple


def iter_ops(path: Path) -> Iterator[Op]:
    """Stream ``record:"op"`` rows.  One pass, nothing held."""

    with path.open("r", buffering=1 << 22) as handle:
        for line in handle:
            if _OP_MARK not in line:
                continue
            match = _OP_RE.search(line)
            if match is None:  # pragma: no cover - schema drift guard
                row = json.loads(line)
                if row.get("record") != "op":
                    continue
                yield Op(
                    int(row["seq"]),
                    int(row["command_buffer_index"]),
                    str(row.get("kernel_name", "?")),
                    tuple(int(d) for d in (row.get("grid") or ())),
                )
                continue
            yield Op(
                int(match.group(1)),
                int(match.group(2)),
                match.group(3),
                (int(match.group(4)), int(match.group(5)), int(match.group(6))),
            )


def is_cycle_mark(op: Op) -> bool:
    return op.grid == LM_HEAD_GRID and LM_HEAD_KERNEL_SUBSTR in op.kernel


def collect_cycle_tails(path: Path, window: int) -> list[list[Op]]:
    """Every cycle's last ``window`` dispatches, in forward order.

    A tail is emitted only once the deque is full, so the first (partial,
    prefill-contaminated) cycle is dropped by construction.

    Holds every tail, so it is for **fixtures and tests only** — 394 cycles of a
    5,000-dispatch window is ~2 M objects.  ``scan_body`` is the production path.
    """

    ring: deque[Op] = deque(maxlen=window)
    tails: list[list[Op]] = []
    for op in iter_ops(path):
        ring.append(op)
        if is_cycle_mark(op) and len(ring) == window:
            tails.append(list(ring))
    return tails


def scan_body(path: Path, window: int) -> tuple[list[Op], int, int]:
    """``(reference tail, common-suffix length, cycles)`` in one bounded pass.

    Same answer as ``measure_body_length(collect_cycle_tails(...))``, but the
    suffix is narrowed incrementally against a single retained reference tail,
    so memory is O(window) rather than O(window x cycles).
    """

    ring: deque[Op] = deque(maxlen=window)
    reference: list[Op] | None = None
    length = window
    cycles = 0
    for op in iter_ops(path):
        ring.append(op)
        if not (is_cycle_mark(op) and len(ring) == window):
            continue
        cycles += 1
        if reference is None:
            reference = list(ring)
            continue
        tail = list(ring)
        # count consecutive matches from the END, capped at the suffix so far
        matched = 0
        while matched < length:
            index = window - 1 - matched
            if (tail[index].kernel, tail[index].grid) != (
                reference[index].kernel,
                reference[index].grid,
            ):
                break
            matched += 1
        length = matched
        if length == 0:
            raise RuntimeError(
                f"{path.name}: two cycles share no common suffix at all — the "
                "verify body is not a fixed compiled graph in this census"
            )
    if reference is None or cycles < 2:
        raise RuntimeError(
            f"only {cycles} full cycle tails — need at least 2 to measure a "
            "common suffix; capture a longer census or lower --window"
        )
    if length == window:
        raise RuntimeError(
            f"the common suffix filled the whole {window}-dispatch window — "
            "raise --window until it stops growing, or the body is not bounded"
        )
    return reference, length, cycles


def measure_body_length(tails: Sequence[Sequence[Op]]) -> int:
    """Longest common suffix of (kernel, grid) over every cycle tail.

    This is the compiled body: a compiled graph replays the same nodes in the
    same order every cycle, and the first dispatch that differs is the first
    one outside it (the draft/sampling lane, whose shapes move with accept
    length).
    """

    if len(tails) < 2:
        raise RuntimeError(
            f"only {len(tails)} full cycle tails — need at least 2 to measure a "
            "common suffix; capture a longer census or pass --body-len"
        )
    window = len(tails[0])
    reference = tails[0]
    length = 0
    while length < window:
        index = window - 1 - length
        key = (reference[index].kernel, reference[index].grid)
        if any(
            (tail[index].kernel, tail[index].grid) != key for tail in tails[1:]
        ):
            break
        length += 1
    if length == window:
        raise RuntimeError(
            f"the common suffix filled the whole {window}-dispatch window — "
            "raise --window until it stops growing, or the body is not bounded"
        )
    return length


# --------------------------------------------------------------------------
# Op classification
# --------------------------------------------------------------------------
#: ``mx.compile``'s JIT-fused elementwise kernels carry the fusion hash in the
#: name (``..._VV_V2V2_11160318154034397263_strided_2``).  Distinguishing them
#: from single-op elementwise kernels matters: a fused name is a chain MLX
#: ALREADY collapsed, so it is not a fusion candidate twice.
_FUSED_HASH_RE = re.compile(r"_\d{12,}_")

OP_CLASSES = (
    "custom kernel",
    "matmul/qmv",
    "matmul/gemm",
    "copy/layout",
    "cache append/slice",
    "cache offset",
    "gather/scatter",
    "sort/top-k",
    "softmax",
    "norm",
    "reduce",
    "dequantize",
    "index/arange",
    "elementwise (fused)",
    "elementwise",
)


def op_class(kernel: str) -> str:
    """The dispatch's op class.  Name-only: the census records no primitive id."""

    n = kernel
    if n.startswith("custom_kernel_"):
        return "custom kernel"
    if "affine_dequantize" in n or "affine_quantize" in n:
        return "dequantize"
    if n.startswith(("affine_qmv", "affine_qmm", "affine_gather_qmm")):
        return "matmul/qmv"
    if n.startswith(("gemv", "steel_gemm")) or "implicit_gemm" in n:
        return "matmul/gemm"
    if "mbsort" in n or "block_sort" in n or "argpartition" in n:
        return "sort/top-k"
    if "block_softmax" in n or "logsumexp" in n:
        return "softmax"
    if n.startswith("rms"):
        return "norm"
    if "reduce" in n:
        return "reduce"
    if n.startswith("gather") or "scatter" in n:
        return "gather/scatter"
    if "compute_dynamic_offset" in n:
        return "cache offset"
    if "dynamic_copy" in n:
        return "cache append/slice"
    if "copy" in n or "Copy" in n:
        return "copy/layout"
    if n.startswith("arange"):
        return "index/arange"
    return "elementwise (fused)" if _FUSED_HASH_RE.search(n) else "elementwise"


# --------------------------------------------------------------------------
# Block segmentation
# --------------------------------------------------------------------------
BLOCK_KINDS = ("QSA", "MoE", "MoE+PLE", "GDN", "HEAD", "prologue")

#: Anchors that name a block, checked in this order.
_GDN_INPROJ = ("gs_32_b_4", (1, 2060, 1))
_QSA_QKV = ("gs_32_b_4", (1, 1536, 1))
_MOE_ROUTER = ("gs_64_b_8", (1, 64, 1))


class Block:
    """One model block: its kind, its span in the body, and its dispatches."""

    __slots__ = ("kind", "start", "end", "ops")

    def __init__(self, kind, start, end, ops=None):
        self.kind = kind
        self.start = start
        self.end = end
        self.ops = list(ops or ())

    @property
    def size(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Block({self.kind!r}, {self.start}, {self.end})"


def block_kind(ops: Sequence[Op]) -> str:
    for op in ops:
        if op.grid == LM_HEAD_GRID:
            return "HEAD"
    for substr, grid in (_GDN_INPROJ, _QSA_QKV):
        if any(substr in op.kernel and op.grid == grid for op in ops):
            kind = "GDN" if grid == _GDN_INPROJ[1] else "QSA"
            break
    else:
        kind = "MoE"
    if any("implicit_gemm_conv" in op.kernel for op in ops):
        kind += "+PLE"
    return kind


def find_block_cuts(body: Sequence[Op]) -> tuple[list[int], str]:
    """Block start offsets, plus the name of the opener strategy that found them.

    Every block of this model opens with the hyper-connection read, and the read
    has two shapes on the two stacks in the artefact directory:

    * ``hc_m4`` — the composed stack's three custom kernels, the first of which
      (``hc_norm``) is the block's first dispatch;
    * ``stock_hc`` — the control stack's nine-dispatch eager form, whose only
      unambiguous marker is the low-rank down projection
      ``gemv_wide[1,80,1]``; the read opens two dispatches earlier, at its RMS
      norm.

    The cut set is checked against the model's own anchor census — one block per
    attention (GDN ``in_proj`` or QSA ``qkv``), one per MoE router, one for the
    final mixer before the head — so a wrong opener fails loudly instead of
    silently mis-attributing dispatches.
    """

    expected = 1 + sum(
        1
        for op in body
        if (_GDN_INPROJ[0] in op.kernel and op.grid == _GDN_INPROJ[1])
        or (_QSA_QKV[0] in op.kernel and op.grid == _QSA_QKV[1])
        or (_MOE_ROUTER[0] in op.kernel and op.grid == _MOE_ROUTER[1])
    )
    candidates: list[tuple[str, list[int]]] = [
        ("hc_m4", [i for i, op in enumerate(body) if HC_OPENER_SUBSTR in op.kernel]),
        (
            "stock_hc",
            [
                i - 2
                for i, op in enumerate(body)
                if op.kernel.startswith("gemv_wide") and op.grid == (1, 80, 1)
            ],
        ),
    ]
    for name, cuts in candidates:
        if len(cuts) == expected and cuts and cuts == sorted(cuts) and cuts[0] >= 0:
            return cuts, name
    raise RuntimeError(
        f"no block opener matched the anchor census: expected {expected} blocks, "
        + ", ".join(f"found {name}={len(cuts)}" for name, cuts in candidates)
    )


def segment(body: Sequence[Op]) -> list[Block]:
    """Cut the body into blocks at each hyper-connection read.

    The leading ops before the first cut are the graph prologue (the PLE
    auxiliary gather + dequant that layer 1 consumes).
    """

    cuts, _strategy = find_block_cuts(body)
    blocks: list[Block] = []
    if cuts[0] > 0:
        blocks.append(Block("prologue", 0, cuts[0], list(body[: cuts[0]])))
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(body)
        ops = list(body[start:end])
        blocks.append(Block(block_kind(ops), start, end, ops))
    return blocks


# --------------------------------------------------------------------------
# Fusable groups
# --------------------------------------------------------------------------
#: ``chain`` = the group's dispatches form a read-after-write chain, so removing
#: them removes dependent GPU launches (charged 1.83 µs each to G) as well as
#: their exposed encode.  ``sibling`` = independent fan-out that MLX's
#: concurrent encoder already overlaps: exposed encode only.  ``mixed`` gives
#: the chain fraction used for the credit.
class Group(NamedTuple):
    key: str
    title: str
    blocks: tuple
    #: (name substring, grid or None) pairs; an op matches if ANY pair matches.
    patterns: tuple
    chain_fraction: float
    mechanism: str
    exactness: str
    #: the share of the matched dispatches the replacement mechanism still
    #: emits.  ``removable = matched - round(matched * replacement_fraction)``:
    #: fusing three kernels into one removes two of them, not three, and no
    #: honest ranking may pretend otherwise.  A fraction rather than a count so
    #: the same group definition prices a 48-layer body and a test fixture.
    replacement_fraction: float = 0.0
    note: str = ""


def _p(*pairs: tuple[str, tuple[int, ...] | None]) -> tuple:
    return tuple(pairs)


GROUPS: tuple[Group, ...] = (
    Group(
        key="qsa_rope",
        title="QSA rope: sin/cos, the two fused rotate kernels, and the 3-copy concat, x4 per layer",
        blocks=("QSA",),
        patterns=_p(
            ("v_Sinfloat32float32", None),
            ("v_Cosfloat32float32", None),
            ("Ef4IAsTypeAFf4IBroadcastB", None),
            ("Ef4IBroadcastBFf4IMultiplyAE", None),
            ("gg1_copybfloat16bfloat16", (32, 1, 1)),
            ("gg1_copybfloat16bfloat16", (64, 1, 1)),
            ("gg2_copybfloat16bfloat16", None),
        ),
        chain_fraction=0.85,
        replacement_fraction=1 / 6,  # one rope call per application: 4 of 24 per layer
        mechanism=(
            "one rope primitive per application instead of "
            "split -> cast -> sin/cos -> 2 fused multiplies -> 3-copy concat; "
            "either mx.fast.rope or a rope epilogue on qsa_m4_fused_kv_gather"
        ),
        exactness=(
            "numerically equivalent, NOT bit-exact by construction (the fused "
            "kernel keeps f32 where the chain round-trips through bf16)"
        ),
        note=(
            "TRAP: mx.fast.rope wrote only row 0 at T=1 on mlx 0.31.2 "
            "(memory: mlx-rope-batch-decode-bug). The M4 window is 4 rows. "
            "Any rope port needs a length-B offset vector and a 4-row A/B."
        ),
    ),
    Group(
        key="qsa_cache_offset",
        title="QSA KV-cache dynamic offset + dynamic slice glue",
        blocks=("QSA",),
        patterns=_p(
            ("compute_dynamic_offset_int32", None),
            ("dynamic_copy", None),
            ("ss_Minimumint32", None),
            ("ss_Multiplyint32", None),
            ("Di4IDivideABEi4OAddDC", None),
        ),
        chain_fraction=1.0,
        replacement_fraction=0.4,  # the 6 dynamic slice copies per layer stay
        mechanism=(
            "the offset is a scalar the host already owns; a fixed-capacity "
            "cache with a device offset resolved ONCE per cycle replaces 6 "
            "offset+slice pairs per layer with one"
        ),
        exactness="exact (identical addresses, identical bytes)",
        note=(
            "W64 already moved the two per-cycle offset reads on-device; this "
            "is the same lever applied per-layer inside the body."
        ),
    ),
    Group(
        key="qsa_mask",
        title="QSA per-layer mask rebuild (arange, compare, bool copies, select)",
        blocks=("QSA",),
        patterns=_p(
            ("arangeint32", (4, 1, 1)),
            ("Ci4IBroadcastBDb1OLessEqualAC", None),
            ("Ci4IBroadcastADi4IBroadcastBEb1OLessCD", None),
            ("g2_copybool_bool_", None),
            ("gg2_copybool_bool_", None),
            ("gg2_copyint32int32", None),
            ("Di4IBroadcastBEi4IMultiplyADFi4IBroadcastEGi4IBroadcastCHi4OAddFG", None),
            ("Di4IBroadcastCEi4OSelectABD", None),
            ("gather_axisbool_int64_intcc", None),
            ("Ci4IBroadcastADi4IAddCBEf4OAsTypeD", None),
            ("Ci4IBroadcastADi4OAddCB", None),
        ),
        chain_fraction=0.8,
        replacement_fraction=1 / 34,  # the window mask, built once instead of per layer
        mechanism=(
            "the mask is layer-INVARIANT for a fixed 4-row window: build it "
            "once in the prologue and let all 12 QSA layers read it, or fold "
            "the comparison into qsa_m4_fused_kv_gather's index math"
        ),
        exactness="exact (same bool tensor, hoisted)",
    ),
    Group(
        key="qsa_indexer_select",
        title="QSA indexer top-k select (mbsort chain, gather, arange, compare/select)",
        blocks=("QSA",),
        patterns=_p(
            ("mbsort", None),
            ("v_copyuint32uint32", None),
            ("gather_frontfloat32_uint32_int_1", None),
            ("arangeint32", (4352, 1, 1)),
            ("v_copyint32float32", None),
            ("Cf4IBroadcastBDf4ODivideAC", None),
            ("Ff4IBroadcastCGf4ISelectABFH", None),
            ("Cf4IBroadcastBDf4OMaximumAC", None),
            ("Fi4IAddABGi4IDivideFBHb1IGreaterGC", None),
        ),
        chain_fraction=0.9,
        replacement_fraction=1 / 13,  # one select kernel per QSA layer
        mechanism=(
            "one select kernel (mtplx/kernels/qsa_indexer_select.py already "
            "exists) replacing the 5-stage mbsort + normalise chain"
        ),
        exactness=(
            "tie-break sensitive: a different top-k tie order changes which "
            "rows attend. Needs a bit-exact A/B on the selected index set."
        ),
        note="overlaps W68 sparse M=4 attention; coordinate before funding both.",
    ),
    Group(
        key="qsa_head_layout",
        title="QSA head-layout copies and dtype casts around the attention gemvs",
        blocks=("QSA",),
        patterns=_p(
            ("g2_copybfloat16bfloat16", (256, 96, 1)),
            ("g3_copybfloat16bfloat16", None),
            ("vn_copyfloat32bfloat16", None),
            ("vn_copybfloat16float32", None),
            ("vn_copybfloat16bfloat16", None),
            ("v_copybfloat16float32", None),
            ("g2_copybfloat16float32", None),
            ("g2_copybfloat16bfloat16", (512, 4, 1)),
        ),
        chain_fraction=1.0,
        replacement_fraction=2 / 13,  # two casts per layer survive any layout
        mechanism=(
            "store K/V in the layout the gemv wants and keep the softmax in "
            "f32 end-to-end, so the three big vn_copy casts disappear"
        ),
        exactness=(
            "exact only if the cast order is preserved; dropping a bf16 "
            "round-trip CHANGES values"
        ),
        note="these are the body's biggest-byte copies; W68 prices the bytes.",
    ),
    Group(
        key="moe_router_glue",
        title="MoE routing head — the route kernel's exact target, 8 dispatches per block",
        blocks=("MoE", "MoE+PLE"),
        patterns=_p(
            ("affine_qmv_wide_bfloat16_t_gs_64_b_8", (1, 64, 1)),
            ("block_softmax_precise_bfloat16", None),
            ("carg_block_sort", None),
            ("gather_axisbfloat16uint32_intcnc", None),
            ("affine_qmv_wide_bfloat16_t_gs_64_b_8", (1, 1, 1)),
            ("row_reduce_small_1_reduce_sumbfloat16", None),
            ("v_Sigmoidbfloat16bfloat16", (4, 1, 1)),
            ("CV2IBroadcastBDV2ODivideAC", (10, 4, 1)),
        ),
        chain_fraction=1.0,
        replacement_fraction=0.25,  # the route kernel is TWO dispatches per block, of 8
        mechanism="mtplx/kernels/qwen4_m4_route.py — ALREADY BUILT AND RETAINED",
        exactness="exact (micro verified, all counters 0)",
        note=(
            "CALIBRATION, NOT A CANDIDATE. These are the 8 of the kernel's 10 "
            "documented targets that survive OPDIET, and the kernel emits 2 in "
            "their place. Measured -0.92 ms/cycle. This method predicts "
            "0.64 ms against the 8 here and 0.85 ms against the 10 the kernel's "
            "own docstring counts on the Step-8 census — so the method reads "
            "8-30 % LOW on the one lever that has an end-to-end number."
        ),
    ),
    Group(
        key="moe_expert_id_copies",
        title="MoE expert-id relayout: two uint32 copies per block feeding the routed kernels",
        blocks=("MoE", "MoE+PLE"),
        patterns=_p(("g2_copyuint32uint32", (10, 4, 1)),),
        chain_fraction=1.0,
        mechanism=(
            "have the route kernel write the expert ids in the layout "
            "paired_routed_glu and routed_down_reduce read, so neither copy "
            "is needed"
        ),
        exactness="exact (same ids, same order)",
        note="rides on the route kernel; not worth a worker of its own.",
    ),
    Group(
        key="moe_shared_gate",
        title="MoE shared-expert gate application (one fused SiLU-multiply per block)",
        blocks=("MoE", "MoE+PLE"),
        patterns=_p(("CV2ISigmoidADV2IMultiplyACEV2OMultiplyDB", (640, 4, 1)),),
        chain_fraction=1.0,
        mechanism="fold into the shared-expert stream W65 is already measuring",
        exactness="numerically equivalent",
    ),
    Group(
        key="attn_residual_add",
        title="Hyper-connection residual write-back after every attention block",
        blocks=("QSA", "GDN"),
        patterns=_p(
            ("DV2IBroadcastBEV2IBroadcastCFV2IMultiplyDEGV2OAddAF", (2560, 4, 4)),
        ),
        chain_fraction=1.0,
        mechanism="epilogue on the attention out_proj",
        exactness="numerically equivalent",
    ),
    Group(
        key="qsa_indexer_proj",
        title="QSA indexer q/k projections and their norms",
        blocks=("QSA",),
        patterns=_p(
            ("rmsbfloat16", (512, 1, 1)),
            ("affine_qmv_wide_bfloat16_t_gs_32_b_4", (1, 64, 1)),
        ),
        chain_fraction=1.0,
        replacement_fraction=1.0,  # nothing is removable: these move weight bytes
        mechanism="none — these move weight bytes; they are real work, listed for completeness",
        exactness="n/a",
    ),
    Group(
        key="gdn_gate_glue",
        title="GDN gate/normalise glue around gdn_conv_norm_rows and gated_delta_step",
        blocks=("GDN",),
        patterns=_p(
            ("g2_copybfloat16bfloat16", (10240, 4, 1)),
            ("gn1_Sigmoidbfloat16bfloat16", None),
            ("Ef4IAsTypeAFf4IExpEGf4INegativeF", None),
            ("rmsbfloat16", (6144, 1, 1)),
            ("Cf4IAsTypeADf4ISigmoidCEf4IAsTypeBFf4IMultiplyDE", None),
        ),
        chain_fraction=1.0,
        mechanism=(
            "absorb the a/b gate (sigmoid, exp(-x)) into gdn_conv_norm_rows' "
            "epilogue and the output rms+SiLU gate into gated_delta_step's; "
            "both kernels already own the tensors"
        ),
        exactness=(
            "bit-exact is reachable — the ops are elementwise on the kernel's "
            "own output and the intermediate is never read by anything else"
        ),
        note="adjacent to W66's GDN keep-mask fold; different dispatches.",
    ),
    Group(
        key="hc_triple",
        title="Hyper-connection read: hc_norm + hc_down + hc_up, 3 kernels at every block",
        blocks=("QSA", "MoE", "MoE+PLE", "GDN", "HEAD"),
        patterns=_p(
            ("hc_norm", None),
            ("hc_down", None),
            ("hc_up", None),
        ),
        chain_fraction=1.0,
        replacement_fraction=1 / 3,  # one fused read per block, of three
        mechanism=(
            "one kernel: norm -> low-rank down -> up is a strict chain on a "
            "[4, 4, 2560] tensor, all three already in "
            "mtplx/kernels/qwen4_m4_hyper_read.py"
        ),
        exactness=(
            "bit-exact iff the bf16 rounding at each hand-off is reproduced "
            "in registers; a f32-carried fusion is NOT bit-exact"
        ),
        note="fusing 3 into 1 removes 2 of every 3, not all 3.",
        # only 2/3 of the matched dispatches can be removed
    ),
)

def match_group(group: Group, op: Op) -> bool:
    for substr, grid in group.patterns:
        if substr in op.kernel and (grid is None or op.grid == grid):
            return True
    return False


def neighbours(body: Sequence[Op], indices: Sequence[int]) -> Counter:
    """The nearest custom kernel on either side of each matched dispatch.

    This is the "which fusion boundary does it sit next to" column: if a group's
    neighbours are dominated by one custom kernel, that kernel's epilogue is the
    cheapest place to absorb it.
    """

    anchors = [
        i
        for i, op in enumerate(body)
        if op.kernel.startswith("custom_kernel_") or op.grid == LM_HEAD_GRID
    ]
    out: Counter = Counter()
    if not anchors:
        return out
    import bisect

    for i in indices:
        j = bisect.bisect_left(anchors, i)
        for k in (j - 1, j):
            if 0 <= k < len(anchors):
                name = body[anchors[k]].kernel
                out[_short(name)] += 1
    return out


def _short(kernel: str) -> str:
    name = kernel
    for prefix in ("custom_kernel_mtplx_qwen4_m4_", "custom_kernel_mtplx_", "custom_kernel_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.split("__")[0].split("_bfloat16")[0][:34]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def build_report(
    path: Path, window: int, body_len: int | None
) -> tuple[dict[str, Any], list[Op]]:
    """``(report, the reference body)``.  One streaming pass over the census."""

    reference, measured, cycle_count = scan_body(path, window)
    length = body_len or measured
    body = list(reference[-length:])
    blocks = segment(body)

    by_class: Counter = Counter()
    by_block_class: Counter = Counter()
    by_kind: Counter = Counter()
    kind_sizes: dict[str, list[int]] = defaultdict(list)
    for block in blocks:
        by_kind[block.kind] += 1
        kind_sizes[block.kind].append(block.size)
        for op in block.ops:
            cls = op_class(op.kernel)
            by_class[cls] += 1
            by_block_class[(block.kind, cls)] += 1

    kind_of = {}
    for block in blocks:
        for i in range(block.start, block.end):
            kind_of[i] = block.kind

    groups: list[dict[str, Any]] = []
    for group in GROUPS:
        indices = [
            i
            for i, op in enumerate(body)
            if kind_of.get(i) in group.blocks and match_group(group, op)
        ]
        matched = len(indices)
        removable = matched - round(matched * group.replacement_fraction)
        # rounded first, so the printed columns multiply out to the printed ms
        dependent = round(removable * group.chain_fraction)
        host_ms = removable * EXPOSED_ENCODE_US / 1000.0
        gpu_ms = dependent * DEPENDENT_LAUNCH_US / 1000.0
        groups.append(
            {
                "key": group.key,
                "title": group.title,
                "blocks": list(group.blocks),
                "matched": matched,
                "removable": removable,
                "dependent": dependent,
                "host_ms": host_ms,
                "gpu_ms": gpu_ms,
                "total_ms": host_ms + gpu_ms,
                "tok_s": (host_ms + gpu_ms) * TOKS_PER_MS,
                "chain_fraction": group.chain_fraction,
                "mechanism": group.mechanism,
                "exactness": group.exactness,
                "note": group.note,
                "neighbours": neighbours(body, indices).most_common(4),
                "classes": Counter(op_class(body[i].kernel) for i in indices).most_common(),
                "calibration": group.key == "moe_router_glue",
            }
        )
    groups.sort(key=lambda g: (not g["calibration"], -g["total_ms"]))

    grouped_indices: set[int] = set()
    claimed: dict[int, str] = {}
    for group in GROUPS:
        for i, op in enumerate(body):
            if kind_of.get(i) not in group.blocks or not match_group(group, op):
                continue
            if i in claimed:
                raise RuntimeError(
                    f"groups {claimed[i]} and {group.key} both claim dispatch "
                    f"{i} ({body[i].kernel} {body[i].grid}) — the ranking would "
                    "double-count it"
                )
            claimed[i] = group.key
            if group.key != "moe_router_glue":
                grouped_indices.add(i)

    _cuts, strategy = find_block_cuts(body)
    report = {
        "census": path.name,
        "opener_strategy": strategy,
        "cycles": cycle_count,
        "window": window,
        "body_len_measured": measured,
        "body_len_used": length,
        "blocks": {
            kind: {
                "count": by_kind[kind],
                "sizes": sorted(set(kind_sizes[kind])),
                "dispatches": sum(kind_sizes[kind]),
            }
            for kind in BLOCK_KINDS
            if by_kind[kind]
        },
        "by_class": by_class.most_common(),
        "by_block_class": {f"{k}|{c}": v for (k, c), v in by_block_class.items()},
        "groups": groups,
        "grouped_dispatches": len(grouped_indices),
        "prices": {
            "exposed_encode_us": EXPOSED_ENCODE_US,
            "dependent_launch_us": DEPENDENT_LAUNCH_US,
            "toks_per_ms": TOKS_PER_MS,
        },
    }
    return report, body


def print_report(report: dict[str, Any]) -> None:
    total = report["body_len_used"]
    print(f"census            {report['census']}")
    print(f"block opener      {report['opener_strategy']}")
    print(
        f"cycles            {report['cycles']}   window {report['window']}   "
        f"body length MEASURED {report['body_len_measured']}"
        + ("" if report["body_len_used"] == report["body_len_measured"] else f" (using {total})")
    )
    print()
    print(f"=== compiled verify body: {total} host-replayed dispatches per cycle")
    print(f"{'block':10s} {'n':>4s} {'size':>12s} {'dispatches':>11s} {'share':>7s}")
    for kind, info in report["blocks"].items():
        sizes = "/".join(str(s) for s in info["sizes"])
        print(
            f"{kind:10s} {info['count']:4d} {sizes:>12s} {info['dispatches']:11d} "
            f"{100*info['dispatches']/total:6.1f}%"
        )
    print()
    print("=== by op class")
    print(f"{'class':22s} {'n':>6s} {'share':>7s} {'H if all removed':>17s}")
    for cls, n in report["by_class"]:
        print(
            f"{cls:22s} {n:6d} {100*n/total:6.1f}% "
            f"{n*EXPOSED_ENCODE_US/1000:14.3f} ms"
        )
    print()
    print("=== by block x class")
    kinds = list(report["blocks"])
    header = f"{'class':22s}" + "".join(f"{k:>9s}" for k in kinds)
    print(header)
    for cls, _ in report["by_class"]:
        row = f"{cls:22s}"
        for k in kinds:
            row += f"{report['by_block_class'].get(f'{k}|{cls}', 0):9d}"
        print(row)
    print()
    print("=== method calibration: the one group with an end-to-end number")
    calib = next((g for g in report["groups"] if g["calibration"]), None)
    if calib is not None:
        print(
            f"  route kernel: {calib['matched']} matched - {calib['matched']-calib['removable']}"
            f" emitted = {calib['removable']} removed  ->  predicted "
            f"{calib['total_ms']:.3f} ms, measured -0.92 ms/cycle (K-M1 micro)"
        )
        print(
            f"  the method reads {100*(0.92-calib['total_ms'])/0.92:.0f} % LOW on "
            "the only lever that has been measured end to end, so every row "
            "below is a floor, not a ceiling."
        )
    print()
    print("=== fusable groups, priced on the production critical path")
    print(
        f"  sibling node = {EXPOSED_ENCODE_US} us of H   "
        f"dependent launch = {DEPENDENT_LAUNCH_US} us of G   "
        f"fundable >= {FUNDABLE_MS} ms"
    )
    print()
    print(
        f"{'group':22s}{'match':>6s}{'rm':>6s}{'dep':>6s}"
        f"{'H ms':>8s}{'G ms':>8s}{'total':>8s}{'tok/s':>7s}  verdict"
    )
    for g in report["groups"]:
        if g["calibration"]:
            verdict = "CALIBRATION (already built)"
        elif g["total_ms"] >= FUNDABLE_MS:
            verdict = "FUND"
        elif g["total_ms"] >= NOTABLE_MS:
            verdict = "marginal"
        else:
            verdict = "below floor"
        print(
            f"{g['key']:22s}{g['matched']:6d}{g['removable']:6d}{g['dependent']:6d}"
            f"{g['host_ms']:8.3f}{g['gpu_ms']:8.3f}{g['total_ms']:8.3f}"
            f"{g['tok_s']:7.2f}  {verdict}"
        )
    print()
    for g in report["groups"]:
        print(f"--- {g['key']}: {g['title']}")
        print(f"    classes:    {g['classes']}")
        print(f"    neighbours: {g['neighbours']}")
        print(f"    mechanism:  {g['mechanism']}")
        print(f"    exactness:  {g['exactness']}")
        if g["note"]:
            print(f"    note:       {g['note']}")
    print()
    print(
        f"grouped dispatches (excluding the calibration group): "
        f"{report['grouped_dispatches']} of {total} "
        f"({100*report['grouped_dispatches']/total:.1f}%)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("census", type=Path)
    parser.add_argument(
        "--window",
        type=int,
        default=5000,
        help="dispatches held per cycle tail; must exceed the body (default 5000)",
    )
    parser.add_argument("--body-len", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--dump-body", type=Path, default=None)
    args = parser.parse_args(argv)

    report, body = build_report(args.census, args.window, args.body_len)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json}")
    if args.dump_body:
        blocks = segment(body)
        kind_of = {}
        for block in blocks:
            for i in range(block.start, block.end):
                kind_of[i] = block.kind
        with args.dump_body.open("w") as handle:
            handle.write("offset\tblock\tcb\tclass\tkernel\tgrid\n")
            for i, op in enumerate(body):
                handle.write(
                    f"{i}\t{kind_of.get(i,'?')}\t{op.cb}\t{op_class(op.kernel)}\t"
                    f"{op.kernel}\t{','.join(str(d) for d in op.grid)}\n"
                )
        print(f"wrote {args.dump_body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
