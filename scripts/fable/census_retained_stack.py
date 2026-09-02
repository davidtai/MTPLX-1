#!/usr/bin/env python3
"""Dispatch census of the RETAINED Qwen3.8 Flash-Next decode stack (instrument I2).

Why this exists
---------------
The only per-family dispatch census anyone has is ChatGPT's, taken on
``current-exact-early-d3-census-2410.jsonl`` (2026-08-31 10:35), i.e. on the
stack as it stood *before* routed-down-reduce, the routed-down residual tail and
the paired routed-GLU landed.  Every family share quoted in
``scratchpad/D-profile-attribution.md`` §4.3 is therefore stale by three MoE
changes worth roughly 0.64 s of decode.  This module takes the same measurement
on the retained stack and prints the difference.

How the JSONL is captured (found, not invented)
-----------------------------------------------
The census is **not** an MLX profiler API, an Instruments trace or
``mx.metal.start_capture``.  It is an *instrumented MLX build*: the OpenSourceWTF
``mlx-profiler`` fork adds ``mlx/backend/common/dispatch_census.{h,cpp}``, which
writes one JSONL row per Metal dispatch when the environment variable
``MLX_DISPATCH_CENSUS`` names an output path (unset ⇒ instrumentation compiled
in but inert).  Schema, verbatim from the files:

    {"record":"op","seq":N,"command_buffer_index":C,"kind":"compute",
     "dispatch":"threads","kernel_name":"...","setBytes_calls":..,
     "setBytes_total_bytes":..,"buffer_binds":..,"grid":[x,y,z],
     "threadgroup":[x,y,z]}
    {"record":"cb","command_buffer_index":C,"op_count":N,"first_op_seq":A,
     "last_op_seq":B,"encode_start_ns":..,"encode_end_ns":..,
     "gpu_start_ns":..,"gpu_end_ns":..}
    {"record":"wait","bucket":"cap_wait|sched_backpressure|alloc_lock|...",
     "wait_ns":..,"at_ns":..}
    {"record":"summary","ops_total":..,"cbs_total":..,"dropped_rows":0,
     "complete":true,"buckets":{...}}

Per-kernel *names* and grids are recorded; per-kernel GPU *times* are not.  The
only timed unit is the command buffer.  That is the whole reason §4.3 needs a
fitted apportionment model rather than a sum.

The exact recipe that produced the 2410/2600 censuses is preserved in
``/private/tmp/pr391_step4_profile.sh`` and ``/private/tmp/pr391_step6_profile.sh``:

* instrumented build: worktree ``.worktrees/mlx-profiler-0322`` at commit
  ``dbb7208a623211ce92ace87a9659b491511710d6``, MLX **0.32.2** (the same MLX the
  runtime serves on, so kernel selection is not perturbed),
* installed as a *python overlay directory*
  ``/tmp/pr391-mlx0322-py312.b9e5wL`` whose ``mlx/core.cpython-312-darwin.so``
  hashes to ``62fac981…c1cb``,
* ``PYTHONPATH="<overlay>:<mtplx worktree>"`` so the overlay's ``mlx`` shadows
  the venv's stock wheel,
* ``MLX_DISPATCH_CENSUS=<out.jsonl>``,
* the whole thing as the single child of ``bench/laguna/run_guarded.py``.

**The overlay is installed and intact on this box** (checked by ``run`` before
anything touches the GPU), so no MLX rebuild is required.  If it ever goes
missing, ``run`` fails closed with the rebuild instructions and ``reduce`` still
works on any previously captured JSONL.

What ``run`` measures
---------------------
One 16,384-token / 1,024-output decode cell of ``scripts/fable/abba_driver.py``,
one seed, arm A flags exactly (``scripts/fable/abba_window.py`` CONTROL_FLAGS +
CONTROL_CANDIDATE_ENV — imported, not copied, so the control cannot drift from
the harness) plus ``--prewarm-ngram-table``.  ``--candidate-extra-env`` adds raw
process environment (the ``MTPLX_FABLE_*`` namespace) for a candidate lane.

``run`` does not fork the driver: it ``execve``s it, so the driver keeps this
process's pid and parent and therefore consumes the guard's one-shot attestation
FD as the guard's *direct* child (``--guard-mode auto`` → ``attestation``).  The
exec is also what makes ``PYTHONPATH`` take effect: the profiler overlay must be
on ``sys.path`` at interpreter start, and mutating ``os.environ`` in a running
interpreter is too late.

The census slows the run down (a JSONL row per dispatch).  Read the output as a
*structural* measurement — counts, shapes, per-family shares, gap anatomy — never
as a tok/s claim.

Reduction
---------
``reduce`` re-implements ``/private/tmp/pr391_census_decode.py`` (busy/idle,
gap families, host-late/driver split, wait buckets) and D's family classifier
and 4-parameter fit (``scratchpad/family.py`` plus the inline fit whose output is
``fam3-2410.json``), then answers the three open questions:

1. **Draft-chain anatomy.**  W42 reads the draft as a serial 3-step loop with a
   host sync per step; W56 reads it as one joint-D3 compiled graph.  The census
   settles it: count the command buffers the three FRSpec draft-head dispatches
   land in per cycle, and the ≥10 µs idle gaps *between* them.  One buffer / no
   interior gaps ⇒ joint graph.  Three buffers separated by host-late gaps ⇒
   serial loop with a sync per step.
2. **The 6 × 17.8 MB MTP bank copies** (``scratchpad/G-opdiet-census.md`` E3:
   ``vn_copybfloat16bfloat16[2228736,1,1]`` = 8,914,944 bf16 = 17.83 MB, the MTP
   QSA K/V banks recopied because the draft cache is not fixed-capacity).
3. **Residual idle per cycle**, split into the first-sync gap (target
   distribution → host) and the D3 → PLE-dequant → target-gather boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Capture mechanism (see the module docstring for provenance)
# --------------------------------------------------------------------------
#: The prebuilt instrumented-MLX python overlay.  This is a directory holding an
#: ``mlx/`` package whose ``core`` extension carries the dispatch census; putting
#: it first on PYTHONPATH shadows the venv's stock wheel.
PROFILER_OVERLAY = Path("/tmp/pr391-mlx0322-py312.b9e5wL")
#: SHA-256 of ``<overlay>/mlx/core.cpython-312-darwin.so``, pinned by
#: ``/private/tmp/pr391_step6_profile.sh``.  A different binary is a different
#: measurement, so this is a hard gate.
PROFILER_CORE_SHA256 = (
    "62fac981de851dbaf32d0e2128a484be29396ec1622119544191869d89a7c1cb"
)
#: The source the overlay was built from, for the rebuild instructions.
PROFILER_SOURCE = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/mlx-profiler-0322"
)
PROFILER_SOURCE_COMMIT = "dbb7208a623211ce92ace87a9659b491511710d6"
PROFILER_MLX_VERSION = "0.32.2"
CENSUS_ENV = "MLX_DISPATCH_CENSUS"

RUN_GUARDED = Path("/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py")
#: The branch venv.  A per-worker worktree usually has none of its own, so fall
#: back to the branch's canonical 3.12 environment — the overlay ships
#: ``core.cpython-312-darwin.so`` and will not import under any other minor.
FALLBACK_PYTHON = Path(
    "/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps"
    "/.venv/bin/python"
)


def default_python() -> Path:
    local = ROOT / ".venv" / "bin" / "python"
    return local if local.is_file() else FALLBACK_PYTHON

QWEN_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")
GPU_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
DEFAULT_SEED = 20260829
DEFAULT_ARTIFACT_DIR = ROOT / ".benchmark-artifacts" / "fable"

# --------------------------------------------------------------------------
# Reduction: the cost model (ported from scratchpad/family.py, verbatim maths)
# --------------------------------------------------------------------------
H = 2560  # hidden size


def q4(n: float, k: float) -> float:
    """q4/group-32 weight bytes: 4 bits + a bf16 scale and bias per 32."""
    return n * k * 0.625


def q8(n: float, k: float) -> float:
    """q8/group-64 weight bytes: 8 bits + a bf16 scale and bias per 64."""
    return n * k * 1.0625


def bf(n: float, k: float) -> float:
    return n * k * 2.0


#: Suffixes ``disambiguate`` appends to the ambiguous kernel name.  They can
#: never collide with a real Metal kernel name, which has no ``|``.
GDN_OUT_TAG = "|gdn_out_proj"
SHARED_DOWN_TAG = "|shared_down_proj"

#: ``grid[1] * 8 == N`` for MLX's quantized matvec kernels.  Verified by two
#: independent anchors: lm_head grid [1,31040,1] → N = 248,320 = vocab, and GDN
#: in_proj grid [1,2060,1] → N = 16,480 = 2048+2048+6144+6144+48+48.
TARGET_QMV: dict[tuple[str, tuple[int, ...]], tuple[str, float]] = {
    ("affine_qmv_wide_gs32_b4", (1, 2060, 1)): ("GDN", q4(16480, H)),
    ("affine_qmv_wide_gs32_b4", (1, 1536, 1)): ("QSA", q4(12288, H)),
    ("affine_qmv_wide_gs32_b4", (1, 320, 1)): ("QSA", q4(2560, 6144)),
    ("affine_qmv_wide_gs32_b4", (1, 64, 1)): ("QSA", q4(512, H)),
    ("affine_qmv_wide_gs64_b8", (1, 31040, 1)): ("LM head", q8(248320, H)),
    ("affine_qmv_wide_gs64_b8", (1, 160, 1)): ("MoE shared", q8(1280, H)),
    ("affine_qmv_wide_gs64_b8", (1, 64, 1)): ("MoE router", q8(512, H)),
    ("affine_qmv_wide_gs64_b8", (1, 1, 1)): ("MoE shared", q8(8, H)),
    ("affine_qmv_wide_gs64_b8", (1, 80, 1)): ("QSA", q8(640, H)),
    # Resolved by ``disambiguate`` below, never emitted by MLX: the q8/g64
    # [1,320,1] matvec is 36 GDN out_proj (K=6144) AND 48 shared down_proj
    # (K=640) per cycle, and only its position in the dispatch stream tells
    # them apart.
    ("affine_qmv_wide_gs64_b8" + GDN_OUT_TAG, (1, 320, 1)): ("GDN", q8(2560, 6144)),
    ("affine_qmv_wide_gs64_b8" + SHARED_DOWN_TAG, (1, 320, 1)): (
        "MoE shared",
        q8(2560, 640),
    ),
}
DRAFT_QMV: dict[tuple[str, tuple[int, ...]], tuple[str, float]] = {
    ("affine_qmv_fast_gs64_b8", (1, 8192, 1)): ("Draft head", q8(65536, H)),
    ("affine_qmv_fast_gs32_b4", (1, 1536, 1)): ("Draft MTP", q4(12288, H)),
    ("affine_qmv_fast_gs32_b4", (1, 320, 1)): ("Draft MTP", q4(2560, 6144)),
    ("affine_qmv_fast_gs32_b4", (1, 64, 1)): ("Draft MTP", q4(512, H)),
    ("affine_qmv_fast_gs32_b4", (1, 2060, 1)): ("Draft MTP", q4(16480, H)),
    ("affine_qmv_fast_gs64_b8", (1, 80, 1)): ("Draft MTP", q8(640, H)),
    ("affine_qmv_fast_gs64_b8", (1, 64, 1)): ("Draft MTP", q8(512, H)),
    ("affine_qmv_gs64_b8", (1, 320, 1)): ("Draft MTP", q8(2560, 640)),
    ("affine_qmv_gs64_b8", (1, 1, 1)): ("Draft MTP", q8(8, H)),
    ("affine_qmv_fast_gs64_b8", (1, 160, 1)): ("Draft MTP", q8(1280, H)),
    ("affine_qmv_fast_gs32_b4", (1, 160, 1)): ("Draft MTP", q4(1280, H)),
}
ROUTED_GU_BYTES = q4(1280, H)  # per selected (row, expert) lane, gate+up packed
ROUTED_DOWN_BYTES = q4(2560, 640)  # per selected (row, expert) lane
HYPER: dict[tuple[int, ...], float] = {  # bf16 hyper-connection / mix
    (1, 2560, 1): bf(10240, 320),
    (1, 80, 1): bf(320, H),
    (1, 640, 1): bf(2560, 320),
    (1, 1, 1): bf(8, H),
}
GDN_STATE = 48 * 128 * 128 * 4  # f32 recurrent state (read + write below)
QSA_KV_GATHER = 2048 * 2 * 256 * 2 * 2  # budget x kv heads x dim x bf16 x (K,V)

_QMV_RE = re.compile(
    r"affine_(gather_)?qmv(_wide|_fast)?_bfloat16_t_gs_(\d+)_b_(\d+)"
)
_COPY_RE = re.compile(r"(s|v|g|n|c|b|d|e|f)?[a-z0-9]{0,3}_?(copy|Copy)")

#: The one dispatch that happens exactly once per verify cycle: the target
#: lm_head matvec.  Used as the cycle marker for the automatic decode window.
LM_HEAD_KERNEL_GRID = (1, 31040, 1)
#: The FRSpec draft head (65,536-row q8), one dispatch per draft depth.
DRAFT_HEAD_GRID = (1, 8192, 1)


def classify(kernel: str, grid: Sequence[int]) -> tuple[str, float, float]:
    """``(family, weight_bytes, activation_bytes)`` for one dispatch.

    Ported unchanged from ``scratchpad/family.py`` so that this tool's table and
    D's table are the same measurement taken twice, and their difference is the
    stack's difference rather than a classifier's.
    """

    g = tuple(int(d) for d in grid)
    n = kernel
    act = 1.0
    for d in g:
        act *= d
    act = act * 2 * 3  # bf16, ~read + read + write

    tag = ""
    if "|" in n:
        n, _, suffix = n.partition("|")
        tag = "|" + suffix
    match = _QMV_RE.match(n)
    if match:
        gather, variant, gs, bits = (
            match.group(1),
            match.group(2) or "",
            match.group(3),
            match.group(4),
        )
        key = f"affine_{'gather_' if gather else ''}qmv{variant}_gs{gs}_b{bits}{tag}"
        if gather:
            rows = g[2] if len(g) > 2 else 1
            if len(g) > 1 and g[1] in (160, 80):
                return "MoE routed", ROUTED_GU_BYTES * rows, act
            return "MoE routed", ROUTED_DOWN_BYTES * rows, act
        for table in (TARGET_QMV, DRAFT_QMV):
            hit = table.get((key, g))
            if hit:
                return hit[0], hit[1], act
        # An unmapped quantized projection still moves real weight bytes; size
        # it from the verified grid[1]*8 x hidden decoding.
        bpe = 0.625 if bits == "4" else 1.0625
        return "Unknown qmv", (g[1] if len(g) > 1 else 0) * 8 * H * bpe, act
    if n.startswith("affine_qmm") or n.startswith("steel_gemm"):
        return "QSA", 0.0, act
    if n.startswith("gemv_wide_bfloat16"):
        return "Hyper/residual", HYPER.get(g, bf((g[1] if len(g) > 1 else 0) * 4, 320)), act
    if n.startswith("gemv_") or n.startswith("gemv_t_"):
        return "QSA", 0.0, act
    if "affine_dequantize" in n or "affine_quantize" in n:
        return "KV / dequant", 0.0, act
    if "gated_delta_step" in n:
        return "GDN", 2 * GDN_STATE, act
    if "gdn_conv_norm_rows" in n or "gdn_step_fused" in n:
        return "GDN", 0.0, act
    if "qsa_m4_fused_kv_gather" in n:
        return "QSA", QSA_KV_GATHER, act
    if "qwen4_m4_combine_tail" in n:
        return "MoE routed", 0.0, act
    if "hyper_v3" in n:
        return "Hyper/residual", 0.0, act
    if "softfloat64" in n or "candidate_selector" in n or "verifier_decision" in n:
        return "Sampling/verify", 0.0, act
    if "mbsort" in n or "block_sort" in n or "argmax" in n or "argpartition" in n:
        return "Sampling/verify", 0.0, act
    if "block_softmax" in n:
        return "QSA", 0.0, act
    if n.startswith("gather_front") or "gather_axis" in n or n.startswith("gather"):
        return "Gather/scatter", 0.0, act
    if "scatter" in n:
        return "Gather/scatter", 0.0, act
    if "rms" in n:
        return "Norm/elementwise", 0.0, act
    if _COPY_RE.match(n) or "copy" in n:
        return "Copy", 0.0, act
    return "Norm/elementwise", 0.0, act


AMBIGUOUS_GRID = (1, 320, 1)


def _is_ambiguous_320(kernel: str, grid: tuple[int, ...]) -> bool:
    return (
        grid == AMBIGUOUS_GRID
        and "affine_qmv_wide" in kernel
        and "gs_64_b_8" in kernel
    )


def disambiguate(buffers: Sequence["CommandBuffer"]) -> dict[str, int]:
    """Split the q8/g64 ``[1,320,1]`` matvec into GDN out_proj and shared down_proj.

    The two are the same kernel at the same grid and differ only by K (6,144 vs
    640) — a 26x byte difference, so getting it wrong is not cosmetic.  Command
    buffers cannot settle it: MLX cuts them every ~45-51 dispatches and 68 of
    the 84 per cycle land in a buffer that holds *both* a GDN layer and an MoE
    fragment.  (Attributing by buffer is what D's §4.3 did; it puts 72 of the 84
    on GDN and ~600 MB/cycle of shared-expert bytes with them.)

    Dispatch *order* does settle it, exactly: walking the ops in ``seq`` order,
    the nearest preceding anchor is the GDN fused ``in_proj`` ``[1,2060,1]``
    inside a GDN layer body and a routed gather-qmv or the shared gate+up
    ``[1,160,1]`` inside the MoE frontier.  On the 2410 census this yields
    36.00 and 48.00 per cycle — the model's 36 GDN and 48 MoE layers, and D's
    own §2 dispatch inventory.
    """

    ordered = sorted(buffers, key=lambda cb: cb.first_op_seq)
    anchor: str | None = None
    counts = {"GDN": 0, "MoE shared": 0, "unanchored": 0}
    for cb in ordered:
        ops = list(cb.ops)
        changed = False
        for index, (kernel, grid) in enumerate(ops):
            if "affine_gather_qmv" in kernel:
                anchor = "moe"
            elif "affine_qmv_wide" in kernel and "gs_64_b_8" in kernel and grid == (1, 160, 1):
                anchor = "moe"
            elif "affine_qmv_wide" in kernel and "gs_32_b_4" in kernel and grid == (1, 2060, 1):
                anchor = "gdn"
            elif _is_ambiguous_320(kernel, grid):
                if anchor == "gdn":
                    ops[index] = (kernel + GDN_OUT_TAG, grid)
                    counts["GDN"] += 1
                else:
                    ops[index] = (kernel + SHARED_DOWN_TAG, grid)
                    counts["MoE shared"] += 1
                    if anchor is None:
                        counts["unanchored"] += 1
                changed = True
        if changed:
            cb.ops = tuple(ops)
    return counts


def nnls(design: Sequence[Sequence[float]], target: Sequence[float], iters: int = 3000) -> list[float]:
    """Non-negative least squares by coordinate descent on the normal equations.

    Same solver D used, generalised to any width.  Deterministic, pure python,
    no numpy: this has to run identically wherever the census is reduced.
    """

    width = len(design[0]) if design else 0
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row, value in zip(design, target):
        for i in range(width):
            rhs[i] += row[i] * value
            for j in range(width):
                gram[i][j] += row[i] * row[j]
    solution = [0.0] * width
    for _ in range(iters):
        for j in range(width):
            if gram[j][j] <= 0:
                continue
            residual = rhs[j] - sum(gram[j][k] * solution[k] for k in range(width))
            solution[j] = max(0.0, solution[j] + residual / gram[j][j])
    return solution


# --------------------------------------------------------------------------
# Reduction: reading the census
# --------------------------------------------------------------------------
class CommandBuffer:
    """One measured command buffer plus the dispatches encoded into it."""

    __slots__ = (
        "index",
        "gpu_start_ns",
        "gpu_end_ns",
        "encode_start_ns",
        "encode_end_ns",
        "first_op_seq",
        "last_op_seq",
        "ops",
    )

    def __init__(
        self,
        index: int,
        gpu_start_ns: int,
        gpu_end_ns: int,
        encode_start_ns: int,
        encode_end_ns: int,
        first_op_seq: int,
        last_op_seq: int,
        ops: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> None:
        self.index = index
        self.gpu_start_ns = gpu_start_ns
        self.gpu_end_ns = gpu_end_ns
        self.encode_start_ns = encode_start_ns
        self.encode_end_ns = encode_end_ns
        self.first_op_seq = first_op_seq
        self.last_op_seq = last_op_seq
        self.ops = ops

    @property
    def duration_ns(self) -> int:
        return self.gpu_end_ns - self.gpu_start_ns


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the census.  The files run to 2 GB, so nothing is held whole."""

    with path.open("r", buffering=1 << 22) as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def read_census(
    path: Path, lo: int, hi: int
) -> tuple[list[CommandBuffer], list[dict[str, Any]], dict[str, Any] | None, Counter]:
    """``(command buffers, waits, final summary, kernel counts)`` inside a window.

    Window semantics are the existing reducer's, kept bit-compatible: a command
    buffer counts when ``lo < first_op_seq`` and ``last_op_seq <= hi`` and
    ``first_op_seq > 0``; an op counts when ``lo < seq <= hi``.
    """

    pending: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    kernels: Counter = Counter()
    buffers: list[CommandBuffer] = []
    waits: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    for row in iter_records(path):
        record = row.get("record")
        if record == "op":
            seq = int(row["seq"])
            if lo < seq <= hi:
                name = str(row.get("kernel_name", "?"))
                grid = tuple(int(d) for d in (row.get("grid") or ()))
                pending[int(row["command_buffer_index"])].append((name, grid))
                kernels[name] += 1
        elif record == "cb":
            index = int(row["command_buffer_index"])
            ops = tuple(pending.pop(index, ()))
            first = int(row.get("first_op_seq", 0))
            last = int(row.get("last_op_seq", 0))
            if first > lo and last <= hi and first > 0:
                if len(ops) != int(row["op_count"]):
                    raise RuntimeError(
                        f"command buffer {index}: {len(ops)} op rows but "
                        f"op_count={row['op_count']} — truncated or interleaved trace"
                    )
                buffers.append(
                    CommandBuffer(
                        index,
                        int(row["gpu_start_ns"]),
                        int(row["gpu_end_ns"]),
                        int(row["encode_start_ns"]),
                        int(row["encode_end_ns"]),
                        first,
                        last,
                        ops,
                    )
                )
        elif record == "wait":
            waits.append(row)
        elif record == "summary" and row.get("final"):
            summary = row
    if summary is None:
        raise RuntimeError(f"{path}: no final summary row — the trace is truncated")
    if summary.get("dropped_rows") or not summary.get("complete", False):
        raise RuntimeError(
            f"{path}: INVALID trace (dropped_rows={summary.get('dropped_rows')} "
            f"complete={summary.get('complete')})"
        )
    buffers.sort(key=lambda cb: cb.gpu_start_ns)
    return buffers, waits, summary, kernels


def find_cycle_marks(path: Path) -> list[int]:
    """Op seqs of the target lm_head matvec — exactly one per verify cycle.

    Prefiltered on the literal grid before parsing: the censuses run to 2 GB and
    a ``json.loads`` per line costs more than the rest of the reduction.
    """

    needle = '"grid":[' + ",".join(str(d) for d in LM_HEAD_KERNEL_GRID) + "]"
    marks: list[int] = []
    with path.open("r", buffering=1 << 22) as handle:
        for line in handle:
            if needle not in line:
                continue
            row = json.loads(line)
            if row.get("record") != "op":
                continue
            if tuple(int(d) for d in (row.get("grid") or ())) != LM_HEAD_KERNEL_GRID:
                continue
            if "affine_qmv" not in str(row.get("kernel_name", "")):
                continue
            marks.append(int(row["seq"]))
    return marks


def auto_window(marks: Sequence[int], ops_total: int) -> tuple[int, int, int]:
    """``(lo, hi, cycles)`` for a cycle-aligned steady-state decode window.

    D picked ``lo`` by hand (59,700 on the 2410 census, which lands inside the
    post-prefill KV-quantize block) and ``hi = ops_total``.  Anchoring on the
    once-per-cycle lm_head instead gives whole cycles with no prefill
    contamination and no partial tail, and it needs no per-run tuning.  Cycle
    *k* is then ``(marks[k], marks[k+1]]``.
    """

    if len(marks) < 3:
        raise RuntimeError(
            f"only {len(marks)} lm_head dispatches found — this census has no "
            "steady-state decode phase to window"
        )
    return marks[0], marks[-1], len(marks) - 1


def union_busy_ns(buffers: Sequence[CommandBuffer]) -> int:
    """GPU-busy time with overlapping command buffers merged, not double-counted."""

    total = 0
    end = 0
    for cb in sorted(buffers, key=lambda c: c.gpu_start_ns):
        if cb.gpu_end_ns <= end:
            continue
        total += cb.gpu_end_ns - max(cb.gpu_start_ns, end)
        end = cb.gpu_end_ns
    return total


# --------------------------------------------------------------------------
# Reduction: the family table
# --------------------------------------------------------------------------
def fit_cost_model(buffers: Sequence[CommandBuffer]) -> dict[str, float]:
    """Fit ``dur = CB_FLOOR + DISPATCH*n + wbytes/BW_w + abytes/BW_a``.

    Four global parameters over every measured buffer.  The per-kernel
    alternative (NNLS per kernel name) is *not identifiable* on a fixed decode
    graph — most kernels appear exactly ``48 x cycles`` times and always in the
    same buffers — which is why D reports it as a negative result and why this
    model is physically anchored instead.
    """

    design: list[list[float]] = []
    target: list[float] = []
    cache: dict[tuple[str, tuple[int, ...]], tuple[str, float, float]] = {}
    for cb in buffers:
        wbytes = abytes = 0.0
        for kernel, grid in cb.ops:
            hit = cache.get((kernel, grid))
            if hit is None:
                hit = classify(kernel, grid)
                cache[(kernel, grid)] = hit
            wbytes += hit[1]
            abytes += hit[2]
        design.append([1.0, float(len(cb.ops)), wbytes, abytes])
        target.append(float(cb.duration_ns))
    coefficients = nnls(design, target)
    return {
        "cb_intercept_ns": coefficients[0],
        "dispatch_ns": coefficients[1],
        "w_ns_per_B": coefficients[2],
        "a_ns_per_B": coefficients[3],
    }


def attribute(
    buffers: Sequence[CommandBuffer], fit: dict[str, float], cycles: int
) -> dict[str, Any]:
    """Split every buffer's *measured* duration over its ops by modelled cost."""

    cb_floor = fit["cb_intercept_ns"]
    per_dispatch = fit["dispatch_ns"]
    w_rate = fit["w_ns_per_B"]
    a_rate = fit["a_ns_per_B"]

    cache: dict[tuple[str, tuple[int, ...]], tuple[str, float, float]] = {}
    fam_ns: Counter = Counter()
    fam_ops: Counter = Counter()
    fam_wbytes: Counter = Counter()
    residual_num = residual_den = 0.0
    for cb in buffers:
        n_ops = len(cb.ops)
        if not n_ops:
            continue
        parts: list[tuple[str, float, float]] = []
        modelled_total = 0.0
        for kernel, grid in cb.ops:
            hit = cache.get((kernel, grid))
            if hit is None:
                hit = classify(kernel, grid)
                cache[(kernel, grid)] = hit
            family, wbytes, abytes = hit
            modelled = cb_floor / n_ops + per_dispatch + w_rate * wbytes + a_rate * abytes
            parts.append((family, modelled, wbytes))
            modelled_total += modelled
        duration = float(cb.duration_ns)
        residual_num += (duration - modelled_total) ** 2
        residual_den += duration * duration
        if modelled_total <= 0:
            share = duration / n_ops
            for family, _modelled, wbytes in parts:
                fam_ns[family] += share
                fam_ops[family] += 1
                fam_wbytes[family] += wbytes
        else:
            for family, modelled, wbytes in parts:
                fam_ns[family] += duration * modelled / modelled_total
                fam_ops[family] += 1
                fam_wbytes[family] += wbytes

    total_ns = sum(fam_ns.values())
    families = []
    for family, ns in fam_ns.most_common():
        families.append(
            {
                "family": family,
                "gpu_ns": ns,
                "share": ns / total_ns if total_ns else 0.0,
                "seconds": ns / 1e9,
                "ms_per_cycle": ns / cycles / 1e6,
                "dispatches": fam_ops[family],
                "dispatches_per_cycle": fam_ops[family] / cycles,
                "weight_MB_per_cycle": fam_wbytes[family] / cycles / 1e6,
                "achieved_GBs": (fam_wbytes[family] / ns) if ns else 0.0,
            }
        )
    return {
        "fit": dict(fit),
        "relative_rms_residual": math.sqrt(residual_num / residual_den)
        if residual_den
        else 0.0,
        "total_ns": total_ns,
        "cycles": cycles,
        "families": families,
    }


# --------------------------------------------------------------------------
# Reduction: idle gaps
# --------------------------------------------------------------------------
GAP_FLOOR_NS = 10_000


def idle_gaps(buffers: Sequence[CommandBuffer]) -> list[dict[str, Any]]:
    """Gaps >= 10 µs between the union-merged command-buffer intervals.

    ``host_late_ns`` is the part of the gap that ended before the *next* buffer
    had even finished encoding — the GPU was idle because the host had not
    handed it work yet.  The remainder is driver/scheduling latency.
    """

    ordered = sorted(buffers, key=lambda cb: cb.gpu_start_ns)
    gaps: list[dict[str, Any]] = []
    if not ordered:
        return gaps
    previous = ordered[0]
    covered_end = previous.gpu_end_ns
    for current in ordered[1:]:
        if current.gpu_start_ns > covered_end:
            gap = current.gpu_start_ns - covered_end
            if gap >= GAP_FLOOR_NS:
                host_late = max(
                    0, min(current.gpu_start_ns, current.encode_end_ns) - covered_end
                )
                gaps.append(
                    {
                        "gap_ns": gap,
                        "host_late_ns": host_late,
                        "driver_ns": gap - host_late,
                        "previous_cb": previous.index,
                        "next_cb": current.index,
                        "previous_kernel": previous.ops[-1][0] if previous.ops else "<empty>",
                        "next_kernel": current.ops[0][0] if current.ops else "<empty>",
                        "previous_last_op_seq": previous.last_op_seq,
                        "next_first_op_seq": current.first_op_seq,
                    }
                )
        if current.gpu_end_ns > covered_end:
            covered_end = current.gpu_end_ns
            previous = current
    return gaps


def gap_families(gaps: Iterable[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    total: Counter = Counter()
    host_late: Counter = Counter()
    driver: Counter = Counter()
    count: Counter = Counter()
    for gap in gaps:
        key = (gap["previous_kernel"], gap["next_kernel"])
        total[key] += gap["gap_ns"]
        host_late[key] += gap["host_late_ns"]
        driver[key] += gap["driver_ns"]
        count[key] += 1
    return [
        {
            "previous_kernel": key[0],
            "next_kernel": key[1],
            "count": count[key],
            "gap_ns": value,
            "host_late_ns": host_late[key],
            "driver_ns": driver[key],
        }
        for key, value in total.most_common(top)
    ]


def is_ple_boundary(previous_kernel: str, next_kernel: str) -> bool:
    """The D3-sample → PLE-q4-dequant → target-gather handoff.

    Two consecutive transitions in D §5 (851.0 ms + 681.8 ms = 64 % of all idle):
    the D3 selector's uint32 copy into the PLE gs32/b4 dequant, and that dequant
    into the target embedding gather.
    """

    pair = (previous_kernel, next_kernel)
    dequant = "affine_dequantize_bfloat16_t_gs_32_b_4"
    if pair[1].startswith(dequant) and (
        "copyuint32" in pair[0] or pair[0].startswith("gather_front")
    ):
        return True
    if pair[0].startswith(dequant) and pair[1].startswith("gather_front"):
        return True
    return False


def cycle_of(seq: int, marks: Sequence[int]) -> int:
    """Index of the cycle a dispatch seq belongs to (-1 before the first mark)."""

    lo, hi = 0, len(marks)
    while lo < hi:
        mid = (lo + hi) // 2
        if marks[mid] <= seq:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def idle_map(
    buffers: Sequence[CommandBuffer], marks: Sequence[int], cycles: int, top: int
) -> dict[str, Any]:
    """Per-cycle idle anatomy: totals, the first-sync gap, the PLE boundary."""

    gaps = idle_gaps(buffers)
    ordered = sorted(buffers, key=lambda cb: cb.gpu_start_ns)
    timeline = (ordered[-1].gpu_end_ns - ordered[0].gpu_start_ns) if ordered else 0
    busy = union_busy_ns(ordered)

    first_sync: list[int] = []
    seen_cycles: set[int] = set()
    ple_total = ple_host_late = 0
    ple_count = 0
    for gap in gaps:
        cycle = cycle_of(gap["previous_last_op_seq"], marks)
        gap["cycle"] = cycle
        # The first >=10 us gap after the target lm_head of a cycle is the
        # target-distribution -> host decision sync.
        if cycle >= 0 and cycle not in seen_cycles:
            seen_cycles.add(cycle)
            first_sync.append(gap["gap_ns"])
        if is_ple_boundary(gap["previous_kernel"], gap["next_kernel"]):
            ple_total += gap["gap_ns"]
            ple_host_late += gap["host_late_ns"]
            ple_count += 1

    buckets: Counter = Counter()
    bucket_counts: Counter = Counter()
    for gap in gaps:
        value = gap["gap_ns"]
        if value < 100_000:
            bucket = "10-100us"
        elif value < 1_000_000:
            bucket = "100us-1ms"
        elif value < 5_000_000:
            bucket = "1-5ms"
        else:
            bucket = ">=5ms"
        buckets[bucket] += value
        bucket_counts[bucket] += 1

    return {
        "gpu_timeline_ns": timeline,
        "gpu_busy_ns": busy,
        "gpu_idle_ns": timeline - busy,
        "gpu_utilization": (busy / timeline) if timeline else 0.0,
        "idle_ms_per_cycle": (timeline - busy) / cycles / 1e6,
        "gaps_ge_10us": {
            "count": len(gaps),
            "total_ns": sum(g["gap_ns"] for g in gaps),
            "host_late_ns": sum(g["host_late_ns"] for g in gaps),
            "driver_ns": sum(g["driver_ns"] for g in gaps),
            "buckets": {
                key: {"count": bucket_counts[key], "total_ns": value}
                for key, value in buckets.most_common()
            },
            "families": gap_families(gaps, top),
            "largest": sorted(gaps, key=lambda g: g["gap_ns"], reverse=True)[:top],
        },
        "first_sync_gap": {
            "cycles_with_a_gap": len(first_sync),
            "total_ns": sum(first_sync),
            "mean_ms": (sum(first_sync) / len(first_sync) / 1e6) if first_sync else 0.0,
            "median_ms": (
                sorted(first_sync)[len(first_sync) // 2] / 1e6 if first_sync else 0.0
            ),
        },
        "ple_boundary": {
            "count": ple_count,
            "total_ns": ple_total,
            "host_late_ns": ple_host_late,
            "ms_per_cycle": ple_total / cycles / 1e6,
            "share_of_idle": (ple_total / (timeline - busy)) if timeline > busy else 0.0,
        },
    }


# --------------------------------------------------------------------------
# Reduction: the three open questions
# --------------------------------------------------------------------------
def draft_chain_anatomy(
    buffers: Sequence[CommandBuffer], marks: Sequence[int], cycles: int
) -> dict[str, Any]:
    """Serial 3-step loop with a sync per step, or one joint-D3 compiled graph?

    Evidence, per cycle: how many *distinct command buffers* the FRSpec draft
    head lands in, and how many >=10 µs idle gaps fall strictly between the
    cycle's first and last draft-head dispatch.  A serial loop must sync between
    depths, so each depth gets its own buffer and the interior gaps are non-zero.
    A joint graph submits all depths together: one or two buffers, no interior
    idle.
    """

    per_cycle_buffers: dict[int, set[int]] = defaultdict(set)
    per_cycle_dispatches: Counter = Counter()
    span: dict[int, list[int]] = defaultdict(list)
    for cb in buffers:
        for kernel, grid in cb.ops:
            if grid != DRAFT_HEAD_GRID or "affine_qmv" not in kernel:
                continue
            cycle = cycle_of(cb.first_op_seq, marks)
            per_cycle_buffers[cycle].add(cb.index)
            per_cycle_dispatches[cycle] += 1
            span[cycle].append(cb.gpu_start_ns)
            span[cycle].append(cb.gpu_end_ns)

    # A gap lies *inside* a cycle's draft window when it opens at or after the
    # first draft-head dispatch started and closes at or before the last one
    # ended: that is exactly the interval a serial per-depth loop has to sync
    # across, and a joint graph never idles in.
    bounds = sorted(
        (min(values), max(values), cycle) for cycle, values in span.items()
    )
    cb_end = {cb.index: cb.gpu_end_ns for cb in buffers}
    cb_start = {cb.index: cb.gpu_start_ns for cb in buffers}
    interior_gaps: Counter = Counter()
    interior_ns: Counter = Counter()
    for gap in idle_gaps(buffers):
        opens = cb_end.get(gap["previous_cb"])
        closes = cb_start.get(gap["next_cb"])
        if opens is None or closes is None:
            continue
        for low, high, cycle in bounds:
            if low > opens:
                break
            if closes <= high:
                interior_gaps[cycle] += 1
                interior_ns[cycle] += gap["gap_ns"]
                break

    counted = [len(v) for v in per_cycle_buffers.values()]
    dispatches = list(per_cycle_dispatches.values())
    mean_buffers = (sum(counted) / len(counted)) if counted else 0.0
    mean_interior = (
        sum(interior_gaps.values()) / len(per_cycle_buffers)
        if per_cycle_buffers
        else 0.0
    )
    interior_ms = sum(interior_ns.values()) / cycles / 1e6 if cycles else 0.0
    if mean_buffers <= 1.5:
        verdict = (
            "JOINT compiled graph — the depths share one submission "
            f"({mean_buffers:.2f} command buffers/cycle), so there is no "
            "per-depth host boundary to reclaim"
        )
    elif mean_interior >= 1.5:
        verdict = (
            f"SERIAL per-depth loop WITH a host sync per depth — "
            f"{mean_buffers:.2f} command buffers/cycle and "
            f"{mean_interior:.2f} interior idle gaps/cycle costing "
            f"{interior_ms:.3f} ms/cycle"
        )
    else:
        verdict = (
            f"SERIAL per-depth SUBMISSION, but the host keeps up — "
            f"{mean_buffers:.2f} command buffers/cycle (one per depth) with only "
            f"{mean_interior:.2f} interior idle gaps/cycle "
            f"({interior_ms:.3f} ms/cycle). The chain's cost is dependent-launch "
            "latency inside the buffers, not a per-step sync"
        )
    return {
        "cycles_seen": len(per_cycle_buffers),
        "draft_head_dispatches_per_cycle": (
            sum(dispatches) / cycles if cycles else 0.0
        ),
        "command_buffers_per_cycle_mean": mean_buffers,
        "command_buffers_per_cycle_histogram": dict(Counter(counted).most_common()),
        "interior_idle_gaps_per_cycle": mean_interior,
        "interior_idle_ms_per_cycle": (
            sum(interior_ns.values()) / cycles / 1e6 if cycles else 0.0
        ),
        "verdict": verdict,
    }


#: ``vn_``-style MLX copy kernels process 4 elements per thread, so the element
#: count is ``grid[0] * 4``.  Calibrated against G-opdiet-census.md E3:
#: ``vn_copybfloat16bfloat16[2228736,1,1]`` = 8,914,944 bf16 = 17.83 MB.
_VECTOR4_COPY_RE = re.compile(r"^[a-z]{0,2}n\d?_copy")
_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "uint16": 2,
    "int16": 2,
    "uint8": 1,
    "int8": 1,
    "bool_": 1,
}


def copy_bytes(kernel: str, grid: Sequence[int]) -> float:
    """Bytes a copy dispatch moves (source only; the write doubles it)."""

    elements = 1.0
    for dim in grid:
        elements *= int(dim)
    if _VECTOR4_COPY_RE.match(kernel):
        elements *= 4
    width = 2
    for name, size in _DTYPE_BYTES.items():
        if name in kernel:
            width = size
            break
    return elements * width


def bank_copies(
    buffers: Sequence[CommandBuffer], cycles: int, min_mb: float = 8.0
) -> dict[str, Any]:
    """Large copy dispatches — the suspected MTP QSA K/V bank recopies.

    ``scratchpad/G-opdiet-census.md`` E3 measured 5.38 per cycle of
    ``vn_copybfloat16bfloat16[2228736,1,1]`` (17.83 MB each) on the 2410 census,
    attributed to the MTP draft cache not being fixed-capacity, so
    ``update_and_fetch`` cannot write in place.  K-D3 turns on whether they
    survive on the retained stack.
    """

    sizes: Counter = Counter()
    totals: dict[tuple[str, tuple[int, ...]], float] = {}
    for cb in buffers:
        for kernel, grid in cb.ops:
            if "copy" not in kernel and "Copy" not in kernel:
                continue
            size = copy_bytes(kernel, grid)
            if size < min_mb * 1e6:
                continue
            sizes[(kernel, grid)] += 1
            totals[(kernel, grid)] = size
    rows = [
        {
            "kernel": kernel,
            "grid": list(grid),
            "MB": totals[(kernel, grid)] / 1e6,
            "count": count,
            "per_cycle": count / cycles if cycles else 0.0,
            "MB_per_cycle": totals[(kernel, grid)] * count / cycles / 1e6
            if cycles
            else 0.0,
        }
        for (kernel, grid), count in sizes.most_common()
    ]
    return {
        "min_MB": min_mb,
        "distinct_shapes": len(rows),
        "total_per_cycle": sum(r["per_cycle"] for r in rows),
        "total_MB_per_cycle": sum(r["MB_per_cycle"] for r in rows),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# The reference table (D §4.3 / scratchpad/fam3-2410.json)
# --------------------------------------------------------------------------
#: ``current-exact-early-d3-census-2410.jsonl``, window ops 59,700–2,128,337,
#: 382 cycles, 65,470 command buffers, 14.089182 s busy / 2.443012 s idle.
#: Taken BEFORE routed-down-reduce, the routed-down residual tail and the paired
#: routed-GLU landed (~0.64 s of decode, essentially all inside MoE routed), so
#: a smaller MoE-routed row on the retained stack is the expected result, not a
#: measurement error.
REFERENCE = {
    "census": "current-exact-early-d3-census-2410.jsonl",
    "window": (59700, 2128337),
    "cycles": 382,
    "command_buffers": 65470,
    "command_buffers_per_cycle": 171.4,
    "dispatches": 2068637,
    "dispatches_per_cycle": 5415.0,
    "gpu_busy_s": 14.089182,
    "gpu_idle_s": 2.443012,
    "gpu_utilization": 0.8522,
    "idle_ms_per_cycle": 6.395,
    "weight_MB_per_cycle": 12270.0,
    "fit": {
        "cb_intercept_ns": 10404.73,
        "dispatch_ns": 2041.01,
        "w_ns_per_B": 0.0016714387,
        "a_ns_per_B": 0.0044920836,
        "relative_rms_residual": 0.2340,
    },
    "ple_boundary_ms_per_cycle": (851.0 + 681.8) / 382,
    "families": {
        # family: (seconds, ms/cycle, dispatches/cycle, weight MB/cycle, GB/s)
        "MoE routed": (4.445, 11.636, 156.0, 6610.9, 568.0),
        "Norm/elementwise": (3.156, 8.261, 2875.2, 0.0, 0.0),
        "GDN": (1.970, 5.157, 224.6, 2676.9, 519.0),
        "Copy": (1.152, 3.015, 1004.1, 0.0, 0.0),
        "Hyper/residual": (0.856, 2.240, 325.8, 863.1, 385.0),
        "QSA": (0.613, 1.604, 196.4, 446.4, 278.0),
        "LM head": (0.435, 1.140, 1.0, 675.4, 593.0),
        "Draft head": (0.375, 0.983, 3.1, 546.0, 556.0),
        "Sampling/verify": (0.329, 0.861, 265.0, 0.0, 0.0),
        "Gather/scatter": (0.265, 0.693, 162.6, 0.0, 0.0),
        "MoE shared": (0.252, 0.659, 109.8, 192.2, 292.0),
        "Draft MTP": (0.110, 0.287, 33.0, 152.5, 531.0),
        "MoE router": (0.087, 0.227, 48.0, 66.8, 294.0),
        "Unknown qmv": (0.034, 0.088, 4.4, 37.9, 429.0),
        "KV / dequant": (0.014, 0.036, 6.4, 0.0, 0.0),
    },
}


def diff_against_reference(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-family retained-minus-D deltas, in the units that survive a rescale.

    ms/cycle and dispatches/cycle are the comparable columns: total seconds also
    move with the cycle count and with the census's own overhead, so they are
    shown but never used as the verdict.
    """

    measured = {row["family"]: row for row in report["attribution"]["families"]}
    rows = []
    for family in sorted(
        set(measured) | set(REFERENCE["families"]),
        key=lambda f: -REFERENCE["families"].get(f, (0,))[0],
    ):
        ref = REFERENCE["families"].get(family)
        got = measured.get(family)
        rows.append(
            {
                "family": family,
                "ref_ms_per_cycle": ref[1] if ref else None,
                "ms_per_cycle": got["ms_per_cycle"] if got else None,
                "delta_ms_per_cycle": (got["ms_per_cycle"] - ref[1])
                if (ref and got)
                else None,
                "ref_dispatches_per_cycle": ref[2] if ref else None,
                "dispatches_per_cycle": got["dispatches_per_cycle"] if got else None,
                "ref_MB_per_cycle": ref[3] if ref else None,
                "MB_per_cycle": got["weight_MB_per_cycle"] if got else None,
                "ref_GBs": ref[4] if ref else None,
                "GBs": got["achieved_GBs"] if got else None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Reduction driver
# --------------------------------------------------------------------------
def reduce_census(
    path: Path,
    *,
    lo: int | None,
    hi: int | None,
    top: int,
    bank_copy_min_mb: float,
    disambiguate_320: bool = True,
    cycles_override: int | None = None,
) -> dict[str, Any]:
    marks_all = find_cycle_marks(path)
    if lo is None or hi is None:
        window_lo, window_hi, cycles = auto_window(marks_all, 0)
        window_mode = "auto (cycle-aligned on the once-per-cycle lm_head matvec)"
    else:
        window_lo, window_hi = lo, hi
        window_mode = "manual"
        cycles = sum(1 for m in marks_all if window_lo < m <= window_hi)
        if cycles < 1:
            raise RuntimeError("the requested window contains no verify cycle")
    if cycles_override is not None:
        # D divided the 2410 window by 382 (the drafted-by-depth count), while
        # 383 lm_head dispatches fall inside its hand-picked op range. The
        # 0.26 % denominator difference lands on every ms/cycle column, so the
        # override exists to make a diff against a published table exact.
        cycles = cycles_override
    marks = [m for m in marks_all if window_lo <= m <= window_hi]

    buffers, waits, summary, kernels = read_census(path, window_lo, window_hi)
    if not buffers:
        raise RuntimeError("the requested window contains no command buffer")

    ambiguous = (
        disambiguate(buffers)
        if disambiguate_320
        else {"GDN": 0, "MoE shared": 0, "unanchored": 0}
    )

    fit = fit_cost_model(buffers)
    attribution = attribute(buffers, fit, cycles)
    idle = idle_map(buffers, marks, cycles, top)

    dispatches = sum(len(cb.ops) for cb in buffers)
    wait_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "total_ns": 0}
    )
    window_start = min(cb.gpu_start_ns for cb in buffers)
    window_end = max(cb.gpu_end_ns for cb in buffers)
    for row in waits:
        at = int(row.get("at_ns", 0))
        if window_start <= at <= window_end:
            bucket = str(row.get("bucket", "?"))
            wait_totals[bucket]["count"] += 1
            wait_totals[bucket]["total_ns"] += int(row.get("wait_ns", 0))

    report = {
        "census": str(path),
        "capture": {
            "mechanism": (
                f"instrumented MLX build ({CENSUS_ENV}) — mlx-profiler fork "
                f"{PROFILER_SOURCE_COMMIT[:12]}, MLX {PROFILER_MLX_VERSION}"
            ),
            "profiler_overlay": str(PROFILER_OVERLAY),
        },
        "window": {
            "mode": window_mode,
            "ops_lo": window_lo,
            "ops_hi": window_hi,
            "cycles": cycles,
        },
        "totals": {
            "command_buffers": len(buffers),
            "command_buffers_per_cycle": len(buffers) / cycles,
            "dispatches": dispatches,
            "dispatches_per_cycle": dispatches / cycles,
            "distinct_kernels": len(kernels),
            "gpu_busy_s": idle["gpu_busy_ns"] / 1e9,
            "gpu_idle_s": idle["gpu_idle_ns"] / 1e9,
            "gpu_utilization": idle["gpu_utilization"],
            "busy_ms_per_cycle": idle["gpu_busy_ns"] / cycles / 1e6,
            "idle_ms_per_cycle": idle["idle_ms_per_cycle"],
        },
        "ambiguous_320_split": {
            "enabled": disambiguate_320,
            "GDN_out_proj_per_cycle": ambiguous["GDN"] / cycles,
            "shared_down_proj_per_cycle": ambiguous["MoE shared"] / cycles,
            "unanchored": ambiguous["unanchored"],
        },
        "attribution": attribution,
        "idle": idle,
        "draft_chain": draft_chain_anatomy(buffers, marks, cycles),
        "bank_copies": bank_copies(buffers, cycles, bank_copy_min_mb),
        "waits": dict(wait_totals),
        "top_kernels": [
            {"kernel": name, "count": count, "per_cycle": count / cycles}
            for name, count in kernels.most_common(top)
        ],
        "census_summary": {
            "ops_total": summary.get("ops_total"),
            "cbs_total": summary.get("cbs_total"),
            "dropped_rows": summary.get("dropped_rows"),
            "complete": summary.get("complete"),
        },
    }
    report["diff_vs_reference"] = diff_against_reference(report)
    return report


def _implied_gbs(ns_per_byte: float) -> str:
    """A fitted-to-zero rate means the term carried no time, not infinite speed."""

    if ns_per_byte <= 0:
        return "n/a GB/s"
    return f"{1.0 / ns_per_byte:.0f} GB/s"


def print_report(report: dict[str, Any]) -> None:
    window = report["window"]
    totals = report["totals"]
    fit = report["attribution"]["fit"]
    out = sys.stdout.write

    out(f"census   {report['census']}\n")
    out(f"capture  {report['capture']['mechanism']}\n")
    out(
        f"window   {window['mode']}: ops {window['ops_lo']:,}–{window['ops_hi']:,}, "
        f"{window['cycles']} verify cycles\n\n"
    )
    out(
        f"busy {totals['gpu_busy_s']:.4f} s / idle {totals['gpu_idle_s']:.4f} s "
        f"({totals['gpu_utilization'] * 100:.2f} % util)   "
        f"{totals['busy_ms_per_cycle']:.3f} + {totals['idle_ms_per_cycle']:.3f} ms/cycle\n"
    )
    out(
        f"{totals['command_buffers']:,} cbs ({totals['command_buffers_per_cycle']:.1f}/cyc), "
        f"{totals['dispatches']:,} dispatches ({totals['dispatches_per_cycle']:.0f}/cyc), "
        f"{totals['distinct_kernels']} distinct kernels\n"
    )
    out(
        f"fit: cb floor {fit['cb_intercept_ns'] / 1e3:.3f} us | per dispatch "
        f"{fit['dispatch_ns'] / 1e3:.3f} us | weight "
        f"{_implied_gbs(fit['w_ns_per_B'])} | activation "
        f"{_implied_gbs(fit['a_ns_per_B'])} | rel-rms "
        f"{report['attribution']['relative_rms_residual']:.4f}\n\n"
    )

    out(
        f"{'family':<20}{'s':>8}{'share':>8}{'ms/cyc':>9}{'disp/cyc':>10}"
        f"{'MB/cyc':>10}{'GB/s':>8}\n"
    )
    for row in report["attribution"]["families"]:
        out(
            f"{row['family']:<20}{row['seconds']:>8.3f}{row['share'] * 100:>7.1f}%"
            f"{row['ms_per_cycle']:>9.3f}{row['dispatches_per_cycle']:>10.1f}"
            f"{row['weight_MB_per_cycle']:>10.1f}{row['achieved_GBs']:>8.0f}\n"
        )

    split = report["ambiguous_320_split"]
    if split["enabled"]:
        out(
            f"q8/g64 [1,320,1] split by dispatch order: "
            f"{split['GDN_out_proj_per_cycle']:.2f} GDN out_proj + "
            f"{split['shared_down_proj_per_cycle']:.2f} shared down_proj per cycle "
            f"(model: 36 GDN + 48 MoE layers)\n"
        )

    out("\n-- diff vs D (current-exact-early-d3-census-2410, 382 cycles) --\n")
    out(
        f"{'family':<20}{'D ms/cyc':>10}{'now':>9}{'delta':>9}"
        f"{'D disp':>10}{'now':>9}{'D MB':>10}{'now':>10}\n"
    )
    for row in report["diff_vs_reference"]:
        def fmt(value: float | None, width: int, digits: int) -> str:
            return ("{:>%d}" % width).format("-") if value is None else (
                "{:>%d.%df}" % (width, digits)
            ).format(value)

        out(
            f"{row['family']:<20}{fmt(row['ref_ms_per_cycle'], 10, 3)}"
            f"{fmt(row['ms_per_cycle'], 9, 3)}{fmt(row['delta_ms_per_cycle'], 9, 3)}"
            f"{fmt(row['ref_dispatches_per_cycle'], 10, 1)}"
            f"{fmt(row['dispatches_per_cycle'], 9, 1)}"
            f"{fmt(row['ref_MB_per_cycle'], 10, 1)}{fmt(row['MB_per_cycle'], 10, 1)}\n"
        )
    out(
        f"\ntotals    D 36.886 busy + 6.395 idle ms/cycle, 5,415 disp/cycle, "
        f"171.4 cbs/cycle\n"
        f"          now {totals['busy_ms_per_cycle']:.3f} busy + "
        f"{totals['idle_ms_per_cycle']:.3f} idle ms/cycle, "
        f"{totals['dispatches_per_cycle']:.0f} disp/cycle, "
        f"{totals['command_buffers_per_cycle']:.1f} cbs/cycle\n"
    )

    idle = report["idle"]
    out("\n-- idle-gap map --\n")
    gaps = idle["gaps_ge_10us"]
    out(
        f"gaps >= 10 us: {gaps['count']:,} totalling {gaps['total_ns'] / 1e9:.3f} s "
        f"({gaps['host_late_ns'] / 1e9:.3f} s host-late / "
        f"{gaps['driver_ns'] / 1e9:.3f} s driver)\n"
    )
    first = idle["first_sync_gap"]
    out(
        f"first-sync gap (target distribution -> host): {first['mean_ms']:.3f} ms mean, "
        f"{first['median_ms']:.3f} ms median over {first['cycles_with_a_gap']} cycles\n"
    )
    ple = idle["ple_boundary"]
    out(
        f"D3 -> PLE dequant -> target gather: {ple['count']} events, "
        f"{ple['total_ns'] / 1e9:.3f} s ({ple['host_late_ns'] / 1e9:.3f} s host-late), "
        f"{ple['ms_per_cycle']:.3f} ms/cycle = {ple['share_of_idle'] * 100:.0f} % of idle "
        f"(D: {REFERENCE['ple_boundary_ms_per_cycle']:.3f} ms/cycle, 64 %)\n"
    )
    out(f"residual idle after the PLE boundary: "
        f"{idle['idle_ms_per_cycle'] - ple['ms_per_cycle']:.3f} ms/cycle\n")
    out("top gap families:\n")
    for family in gaps["families"][:8]:
        out(
            f"  {family['count']:>5}  {family['gap_ns'] / 1e6:>9.1f} ms  "
            f"{family['previous_kernel'][:44]} -> {family['next_kernel'][:44]}\n"
        )

    draft = report["draft_chain"]
    out("\n-- draft-chain anatomy (W42 serial loop vs W56 joint-D3 graph) --\n")
    out(
        f"draft-head dispatches/cycle {draft['draft_head_dispatches_per_cycle']:.2f} "
        f"(D: 3.1) in {draft['command_buffers_per_cycle_mean']:.2f} command "
        f"buffers/cycle {draft['command_buffers_per_cycle_histogram']}\n"
    )
    out(
        f"idle gaps strictly inside the draft window: "
        f"{draft['interior_idle_gaps_per_cycle']:.2f}/cycle, "
        f"{draft['interior_idle_ms_per_cycle']:.3f} ms/cycle\n"
    )
    out(f"VERDICT: {draft['verdict']}\n")

    banks = report["bank_copies"]
    out(
        f"\n-- large copy dispatches (>= {banks['min_MB']:.0f} MB); "
        "G E3 expected 6 x 17.8 MB MTP bank copies --\n"
    )
    if not banks["rows"]:
        out("  none: the 6 x 17.8 MB MTP QSA K/V bank copies are GONE\n")
    for row in banks["rows"][:12]:
        out(
            f"  {row['per_cycle']:>6.2f}/cyc  {row['MB']:>8.2f} MB  "
            f"{row['MB_per_cycle']:>9.1f} MB/cyc  {row['kernel'][:52]} {row['grid']}\n"
        )
    out(
        f"  total {banks['total_per_cycle']:.2f}/cycle, "
        f"{banks['total_MB_per_cycle']:.1f} MB/cycle\n"
    )


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
def verify_profiler_overlay() -> dict[str, str]:
    """Fail closed unless the pinned instrumented-MLX overlay is present."""

    core = PROFILER_OVERLAY / "mlx" / "core.cpython-312-darwin.so"
    if not core.is_file():
        raise SystemExit(
            f"the instrumented MLX overlay is missing: {core}\n"
            "The dispatch census needs the mlx-profiler build, not the stock "
            f"wheel. Rebuild from {PROFILER_SOURCE} at commit "
            f"{PROFILER_SOURCE_COMMIT} (MLX {PROFILER_MLX_VERSION}) and install "
            f"it as a python overlay at {PROFILER_OVERLAY}, or reduce an "
            "already-captured JSONL with the `reduce` subcommand instead."
        )
    digest = hashlib.sha256(core.read_bytes()).hexdigest()
    if digest != PROFILER_CORE_SHA256:
        raise SystemExit(
            f"instrumented MLX binary drifted: {core}\n"
            f"  expected {PROFILER_CORE_SHA256}\n  observed {digest}\n"
            "A different binary is a different measurement; refusing."
        )
    return {"overlay": str(PROFILER_OVERLAY), "core_sha256": digest}


def control_arm() -> tuple[list[str], list[str]]:
    """``(driver flags, candidate-env settings)`` of the retained control arm.

    Imported from ``abba_window`` rather than copied, so the census can never
    silently measure a different stack from the one the ABBA harness calls the
    control.
    """

    sys.path.insert(0, str(ROOT / "scripts" / "fable"))
    import abba_window  # noqa: PLC0415  (deliberately late; ROOT-relative)

    return list(abba_window.control_flags(1024)), list(
        abba_window.CONTROL_CANDIDATE_ENV
    )


def build_driver_argv(args: argparse.Namespace) -> list[str]:
    flags, candidate_env = control_arm()
    argv = [
        str(args.python),
        str(ROOT / "scripts" / "fable" / "abba_driver.py"),
        "--label",
        args.label,
        "--sequence",
        str(args.sequence),
        "--seed",
        str(args.seed),
        "--receipt-path",
        str(args.receipt_path),
        "--guard-mode",
        "attestation",
        "--prewarm-ngram-table",
        *flags,
    ]
    for setting in candidate_env:
        argv.extend(["--candidate-env", setting])
    for setting in args.candidate_env:
        argv.extend(["--candidate-env", setting])
    for setting in args.candidate_extra_env:
        argv.extend(["--env", setting])
    argv.extend(args.driver_flag)
    return argv


def build_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment[CENSUS_ENV] = str(args.census_out)
    parts = [str(PROFILER_OVERLAY), str(ROOT)]
    existing = environment.get("PYTHONPATH")
    if existing:
        parts.extend(p for p in existing.split(os.pathsep) if p not in parts)
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment


def guarded_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(RUN_GUARDED),
        "--plist",
        str(QWEN_PLIST),
        "--lock-path",
        str(GPU_LOCK),
        "--lock-timeout-seconds",
        "3600",
        "--child-timeout-seconds",
        "3600",
        "--",
        str(args.python),
        str(Path(__file__).resolve()),
        "run",
        "--census-out",
        str(args.census_out),
        "--receipt-path",
        str(args.receipt_path),
        "--label",
        args.label,
        "--sequence",
        str(args.sequence),
        "--seed",
        str(args.seed),
        *[x for s in args.candidate_env for x in ("--candidate-env", s)],
        *[x for s in args.candidate_extra_env for x in ("--candidate-extra-env", s)],
        *[x for s in args.driver_flag for x in ("--driver-flag", s)],
    ]


def do_run(args: argparse.Namespace) -> int:
    verify_profiler_overlay()
    if not Path(args.python).is_file():
        raise SystemExit(f"interpreter not found: {args.python}")
    if args.census_out.exists():
        raise SystemExit(f"census output already exists: {args.census_out}")
    args.census_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_driver_argv(args)
    environment = build_environment(args)
    print(f"[census] {CENSUS_ENV}={args.census_out}", file=sys.stderr, flush=True)
    print(f"[census] PYTHONPATH={environment['PYTHONPATH']}", file=sys.stderr, flush=True)
    print(f"[census] exec {' '.join(argv)}", file=sys.stderr, flush=True)
    if args.dry_run:
        return 0
    # execve, not fork: the driver must keep this pid and parent so it consumes
    # the guard's one-shot attestation FD as the guard's direct child, and the
    # profiler overlay only reaches sys.path through a fresh interpreter start.
    os.execve(argv[0], argv, environment)
    raise SystemExit("execve returned")  # unreachable


def do_reduce(args: argparse.Namespace) -> int:
    report = reduce_census(
        args.census,
        lo=args.lo,
        hi=args.hi,
        top=args.top,
        bank_copy_min_mb=args.bank_copy_min_mb,
        disambiguate_320=not args.no_disambiguate,
        cycles_override=args.cycles,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"[census] wrote {args.json_out}", file=sys.stderr)
    print_report(report)
    return 0


def do_command(args: argparse.Namespace) -> int:
    print(" \\\n  ".join(guarded_command(args)))
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--census-out", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--label", default="census-retained-stack")
    parser.add_argument("--sequence", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python(),
        help="Interpreter for the guard and the driver (must be 3.12: the "
        "instrumented overlay ships core.cpython-312-darwin.so).",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra construction-time MTPLX_* override on top of the control.",
    )
    parser.add_argument(
        "--candidate-extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Raw process environment (the MTPLX_FABLE_* fable namespace, "
        "MLX_MAX_OPS_PER_BUFFER, ...) passed through the driver's --env.",
    )
    parser.add_argument(
        "--driver-flag",
        action="append",
        default=[],
        metavar="FLAG",
        help="Extra abba_driver.py flag (use the =form: --driver-flag=--nax-verify).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run", help="capture one census (guard's direct child)")
    _add_run_arguments(run)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=do_run)

    command = sub.add_parser("command", help="print the guarded command line")
    _add_run_arguments(command)
    command.set_defaults(func=do_command)

    reduce_parser = sub.add_parser("reduce", help="reduce a captured census")
    reduce_parser.add_argument("census", type=Path)
    reduce_parser.add_argument(
        "--lo", type=int, default=None, help="manual window (default: cycle-aligned)"
    )
    reduce_parser.add_argument("--hi", type=int, default=None)
    reduce_parser.add_argument("--top", type=int, default=20)
    reduce_parser.add_argument("--bank-copy-min-mb", type=float, default=8.0)
    reduce_parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Force the per-cycle denominator (default: the lm_head dispatches "
        "inside the window). Use 382 to diff against D's published table exactly.",
    )
    reduce_parser.add_argument(
        "--no-disambiguate",
        action="store_true",
        help="Leave the q8/g64 [1,320,1] matvec in 'Unknown qmv', reproducing "
        "scratchpad/family.py's raw output for an apples-to-apples audit.",
    )
    reduce_parser.add_argument("--json-out", type=Path, default=None)
    reduce_parser.set_defaults(func=do_reduce)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode in ("run", "command"):
        if args.sequence is None:
            trailing = args.census_out.stem.rsplit("-", 1)[-1]
            args.sequence = int(trailing) if trailing.isdigit() else 1
        if args.receipt_path is None:
            args.receipt_path = (
                DEFAULT_ARTIFACT_DIR / f"{args.label}-{args.sequence}.json"
            )
    if args.mode == "reduce" and (args.lo is None) != (args.hi is None):
        raise SystemExit("--lo and --hi must be given together")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
