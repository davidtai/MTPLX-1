"""Second-GPU-stream lane for the physical-M4 shared expert (``MTPLX_FABLE_SHARED_LANE``).

What this lane changes
----------------------
Nothing arithmetic.  It emits the *identical* MLX ops, with the identical
arguments, on a second ``mx.gpu`` stream so that the shared expert's two
quantized matvecs run concurrently with the paired routed GLU and the routed
down reduction instead of in barrier waves of their own.  Bit-exactness is a
property of construction here, not of a transcription: the same
``mx.quantized_matmul`` / ``mx.split`` / ``nn.silu`` / ``down_proj`` calls run,
in the same order, over the same arrays.  Only the ``Stream`` they are recorded
on differs.  The install self-check still compares every layer with
``mx.array_equal`` (see ``qwen4_m4_stage3.install_qwen4_m4_stage3``).

The measured anatomy this lane is built from
--------------------------------------------
From the retained-stack dispatch census
(``.benchmark-artifacts/pr391/w58-retained-control-census-1788370322.jsonl``,
382 verify cycles; the seq window quoted below is one representative MoE
block, ops 400045-400062), one physical-M4 MoE layer issues, in this order:

======  ===========================================  =============================
 seq    kernel (grid)                                 what it is
======  ===========================================  =============================
400045  affine_qmv_wide gs_64_b_8   (1, 64, 1)        router GEMV, N=512, K=2560
400046  block_softmax_precise_bfloat16 (512, 1, 1)    fp32-precise softmax
400047  carg_block_sort bn128_tn4   (1, 4, 1)         argpartition top-10
400048  affine_qmv_wide gs_64_b_8   (1, 160, 1)       SHARED gate/up, N=1280, K=2560
400049  gather_axis                 (1, 10, 4)        take_along_axis(route scores)
400050  row_reduce_small_1_sum      (4, 1, 1)         route-score sum
400051  gemv_wide_bfloat16          (1, 1, 1)         hyper inject GEMV
400052  affine_qmv_wide gs_64_b_8   (1, 1, 1)         SHARED scalar gate, N=8(1)
400053  Compiled Sigmoid/Mul/Mul    (640, 4, 1)       SHARED split + SiLU(gate)*up
400054  Compiled Broadcast/Divide   (10, 4, 1)        route-score renormalise
400055  v_Sigmoid                   (4, 1, 1)         shared factor
400056  affine_qmv_wide gs_64_b_8   (1, 320, 1)       SHARED down, N=2560, K=640
400057  g2_copy uint32              (10, 4, 1)        expert-id contiguity copy
400058  CK paired_routed_glu        (5120, 40, 1)     routed gate/up + SiLU*up
400061  CK routed_down_reduce       (20480, 4, 1)     routed down + weighted reduce
400062  CK routed_shared_residual_t (10240, 1, 1)     shared add + hyper residual
======  ===========================================  =============================

So the shared expert is four dispatches per layer (three when
``MTPLX_FABLE_ROUTE_KERNEL`` folds the scalar gate away), and they form a
three-deep *dependent* chain: gate/up -> SiLU*up -> down.

Weight bytes per layer (q8/group-64 = 1 byte + a bf16 scale and bias per 64,
so 1.0625 B/weight):

* gate/up  1280 x 2560 x 1.0625 = 3.482 MB
* down     2560 x  640 x 1.0625 = 1.741 MB
* scalar gate                     0.003 MB

5.226 MB per layer, 250.8 MB per cycle over 48 layers -- which is the census's
"MoE shared 251.7 MB/cyc" row.  At 600 GB/s that is 8.7 us/layer, 0.42 ms/cyc.

Why the census's 135-203 GB/s for this family is NOT a measurement
-----------------------------------------------------------------
``scripts/fable/census_retained_stack.py`` does not time kernels -- the
instrumented MLX build records per-*command-buffer* GPU intervals only.  The
per-family ms/GB-s columns come from ``fit_cost_model`` + ``attribute``: four
global coefficients (cb floor, per-dispatch ns, weight ns/B, activation ns/B)
fitted by NNLS over all buffers, after which each buffer's measured duration is
split across its ops *in proportion to their modelled cost*.  Two consequences
make the shared family's rate an artifact:

1. The column is monotone in bytes-per-dispatch, not in memory efficiency.
   Under the control fit (cb floor 59.5 us, 2.208 us/dispatch, 570 GB/s
   weights) the shared expert's three dispatches model at 9.15 us of bytes plus
   12.7 us of floor+dispatch overhead; dividing 5.226 MB by that sum yields
   ~203 GB/s by construction.  A family with 61 MB per dispatch (MoE routed)
   lands near the fitted weight rate for exactly the same reason.

2. The same two censuses report the shared family with *identical* dispatch
   counts (144.0/cyc) and *identical* bytes (251.7 MB/cyc) but 203 GB/s
   (control) versus 135 GB/s (composed) -- a 50 % swing that can only come from
   the fit, not from the kernels.

Separately, the classifier has no entry for the three retained custom MoE
kernels (``paired_routed_glu``, ``routed_down_reduce``,
``routed_shared_residual_tail``); they fall through to
``("Norm/elementwise", 0.0 weight bytes)``.  Their real time -- the
routed-down reduce alone measures 79.1 us per layer in the 18,333 three-op
command buffers that isolate it, i.e. 3.8 ms/cycle -- is therefore unmodelled,
and ``attribute`` smears it across every other op in the same buffer,
inflating the small dispatches' apportioned share further.

Why the shared rows cannot join the paired routed GLU's grid
------------------------------------------------------------
``kernels/qwen4_m4_routed_glu.py`` is hard-specialized to **affine q4 /
group-32**: ``GROUP_SIZE = 32``, ``WEIGHT_BYTES_PER_ROW = HIDDEN / 2``,
``load_q4_vector``/``qdot_q4``.  The shared expert is **q8 / group-64**.  An
eleventh lane would need a second dequant path, a second scale/bias stride and
three more buffers inside a kernel whose bit-exactness is validated against
MLX's q4/g32 ``gather_qmv_fast`` arithmetic.  That is a different kernel, not
an extra lane.

Why one fused gate/up -> SiLU -> down kernel is not available either
-------------------------------------------------------------------
The down projection needs the whole ``[4, 640]`` activation before any of its
2,560 output rows can start, so a single dispatch would need either a grid-wide
barrier or one threadgroup per row.  Row ownership is four threadgroups on a
40-core GPU and re-reads the 3.482 MB gate/up pack four times -- the same
per-thread serial walk that made ``fused_hyper_read`` at grid (1024, S, 1)
collapse to 13 tok/s (program A'.0).

And why the arithmetic folds were already measured as losses
------------------------------------------------------------
* "shared scalar gate packed into the shared gate/up projection" --
  ``docs/perf/pr391-m4-shared-gate-pack-result.md``: +36 ms decode.  The
  1,281-row qmm plus its strided gate/up tail cost more than the removed
  one-row dispatch.
* "shared q8/g64 down projection fused with the shared-factor multiply and
  add" -- PR391 ledger: component +34 %.  That fold put a K=640 q8 dot inside
  the residual tail's 40-threadgroup / 256-thread geometry, which has no
  simd-level K parallelism to give it.

Both changed *arithmetic placement*.  This lane changes only *scheduling*, and
leaves both kernels exactly as measured.

NUMERICS: why a stream cannot move a value, and what the install gate does
    and does not cover
--------------------------------------------------------------------------
Both arms call one definition (``_emit_branch``), so the op set, the argument
list and the order are the same by construction.  The remaining question is
whether ``mx.compile`` can fuse the branch DIFFERENTLY once it is on another
stream, since a different fusion group could in principle change a rounding
boundary.  It cannot, for two reasons:

* ``mlx/compile.cpp::compile_fuse`` stops fusing on a stream mismatch
  (``a.primitive().stream() != s``), so a group never spans streams; and
* the branch's three elementwise ops are already their own group on the shipped
  lane.  The census shows them as exactly one kernel,
  ``CV2ISigmoidADV2IMultiplyACEV2OMultiplyDB...(640, 4, 1)`` = split +
  SiLU(gate) * up, bounded on both sides by quantized matvecs, which are not
  fusable.  There is nothing on the default stream for it to have been fused
  with, so moving it loses no fusion and gains none.

Limit worth stating: ``install_qwen4_m4_stage3``'s per-layer ``mx.array_equal``
proves the equivalence on the EAGER graph, on the real packs, before the first
token.  Equivalence inside the outer compiled verify graph rests on the two
points above plus the ABBA window's response digest, which is unchanged by
construction if they hold.  If a future MLX relaxes the stream-mismatch guard,
the digest is the thing that catches it.

What the lane costs
-------------------
MLX synchronises cross-stream dependencies with ``Fence`` (``mlx/fence.cpp``):
the producer stream emits a ``barrier()`` plus a one-thread ``fence_update``
dispatch, the consumer stream a one-thread ``fence_wait`` dispatch that spins
on a shared word.  This lane creates two crossings per layer -- ``x`` in,
``shared_down`` out -- so roughly four extra tiny dispatches per layer, 192 per
cycle.  At this program's measured launch costs (1.83 us dependent, 0.80 us
independent; ``scripts/fable/micro_dependent_launch.py``) that is ~0.25 ms per
cycle of overhead which the overlap has to beat.

``scripts/fable/micro_shared_lane.py`` is the falsifier: it times the two
shared matvecs in isolation on the queued lane and then times the stock versus
two-stream MoE block over 48 layers.  Nothing here should be promoted before
that micro reports the isolated shared-branch cost.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


ROWS = 4
HIDDEN = 2560
INTERMEDIATE = 640

#: q8/group-64 is the only pack this lane is contracted against.  It is not a
#: kernel constraint (there is no kernel) but a measurement constraint: the
#: dispatch anatomy above, and therefore the reason to overlap at all, is the
#: anatomy of *this* pack.  A different pack must re-derive the anatomy before
#: claiming the lane helps.
SHARED_BITS = 8
SHARED_GROUP_SIZE = 64
SHARED_MODE = "affine"

#: Weight bytes moved by the branch per layer, used by the micro and by the
#: engagement report.  1 byte per weight plus a bf16 scale and bias per group.
_BYTES_PER_WEIGHT = 1.0 + 2.0 * 2.0 / SHARED_GROUP_SIZE
GU_BYTES_PER_LAYER = int(2 * INTERMEDIATE * HIDDEN * _BYTES_PER_WEIGHT)
DOWN_BYTES_PER_LAYER = int(HIDDEN * INTERMEDIATE * _BYTES_PER_WEIGHT)
BYTES_PER_LAYER = GU_BYTES_PER_LAYER + DOWN_BYTES_PER_LAYER

#: Dispatches the branch emits with and without the lane.  Equal on purpose:
#: the lane removes no dispatch, it only re-homes three of them.  (The two
#: fence crossings MLX adds are counted separately in the report so a null A/B
#: cannot be read as "nothing changed".)
BRANCH_DISPATCHES_PER_LAYER = 3
FENCE_DISPATCHES_PER_LAYER = 4


#: Process-wide engagement counters.  The counters law: a lane that reads flat
#: in an A/B must still be able to prove it ran.
COUNTERS: dict[str, int] = {
    "contract_checks": 0,
    "installed_layers": 0,
    "exactness_failures": 0,
    "branch_calls": 0,
    "stock_calls": 0,
    "streams_created": 0,
}


def reset_counters() -> None:
    """Zero the engagement counters. Test-support only."""

    for key in COUNTERS:
        COUNTERS[key] = 0


class SharedLaneContractError(RuntimeError):
    """The lane was armed against a shared expert it is not contracted for.

    Raised, never swallowed: the flag exists to move a specific, measured
    three-dispatch branch onto a second stream, and arming it against a
    different pack means the arm measured something else.
    """


_STREAM: Any | None = None


def stream() -> Any:
    """The lane's second GPU stream, created once per process.

    A stream is a command queue; creating one per call would leak queues and
    would also defeat the point, since MLX keys its cross-stream fences by
    stream index.
    """

    global _STREAM
    if _STREAM is None:
        _STREAM = mx.new_stream(mx.default_device())
        COUNTERS["streams_created"] += 1
    return _STREAM


def reset_stream_cache() -> None:
    """Drop the memoized stream. Test-support only."""

    global _STREAM
    _STREAM = None


def _projection_fields(
    owner: Any, name: str, *, prefix: str = ""
) -> tuple[Any, Any, Any]:
    """``(weight, scales, biases)`` off an affine pack.

    ``prefix`` is ``"gu_"`` for the fused gate/up pack, which carries its three
    arrays on the shared-expert module itself rather than on a sub-projection.
    """

    fields = tuple(f"{prefix}{field}" for field in ("weight", "scales", "biases"))
    missing = [f for f in fields if getattr(owner, f, None) is None]
    if missing:
        raise SharedLaneContractError(
            f"shared expert {name} is not an affine-quantized projection: "
            f"{type(owner).__name__} has no {', '.join(missing)}"
        )
    return tuple(getattr(owner, f) for f in fields)


def check_contract(block: Any, *, index: int) -> None:
    """Validate one MoE block's shared expert against the lane's contract.

    Construction-bound and total: every field the anatomy above depends on is
    named here, so a mis-armed flag fails at install with the offending field
    rather than mid-forward or, worse, silently on a pack whose dispatch
    profile is different.
    """

    COUNTERS["contract_checks"] += 1
    where = f"MTPLX_FABLE_SHARED_LANE layer {index}"

    shared = getattr(block, "shared_expert", None)
    if shared is None:
        raise SharedLaneContractError(f"{where}: block has no shared_expert")
    for field in ("gu_weight", "gu_scales", "gu_biases", "down_proj"):
        if not hasattr(shared, field):
            raise SharedLaneContractError(
                f"{where}: shared_expert has no {field}; the lane is "
                "contracted against the fused gate/up shared MLP "
                "(_FusedGateUpMLP)"
            )

    bits = int(getattr(shared, "bits", -1))
    group_size = int(getattr(shared, "group_size", -1))
    mode = str(getattr(shared, "mode", ""))
    if (bits, group_size, mode) != (SHARED_BITS, SHARED_GROUP_SIZE, SHARED_MODE):
        raise SharedLaneContractError(
            f"{where}: shared gate/up pack is bits={bits} group_size="
            f"{group_size} mode={mode!r}; the lane is contracted against "
            f"bits={SHARED_BITS} group_size={SHARED_GROUP_SIZE} "
            f"mode={SHARED_MODE!r}"
        )

    down = shared.down_proj
    down_bits = int(getattr(down, "bits", -1))
    down_group = int(getattr(down, "group_size", -1))
    down_mode = str(getattr(down, "mode", ""))
    if (down_bits, down_group, down_mode) != (
        SHARED_BITS,
        SHARED_GROUP_SIZE,
        SHARED_MODE,
    ):
        raise SharedLaneContractError(
            f"{where}: shared down pack is bits={down_bits} group_size="
            f"{down_group} mode={down_mode!r}; the lane is contracted against "
            f"bits={SHARED_BITS} group_size={SHARED_GROUP_SIZE} "
            f"mode={SHARED_MODE!r}"
        )

    gu_weight, gu_scales, gu_biases = _projection_fields(
        shared, "gate/up", prefix="gu_"
    )
    want_gu_weight = (2 * INTERMEDIATE, HIDDEN * SHARED_BITS // 32)
    want_gu_meta = (2 * INTERMEDIATE, HIDDEN // SHARED_GROUP_SIZE)
    for name, array, want in (
        ("gu_weight", gu_weight, want_gu_weight),
        ("gu_scales", gu_scales, want_gu_meta),
        ("gu_biases", gu_biases, want_gu_meta),
    ):
        if tuple(array.shape) != want:
            raise SharedLaneContractError(
                f"{where}: {name} shape {tuple(array.shape)} is not {want}"
            )

    down_weight, down_scales, down_biases = _projection_fields(down, "down")
    want_down_weight = (HIDDEN, INTERMEDIATE * SHARED_BITS // 32)
    want_down_meta = (HIDDEN, INTERMEDIATE // SHARED_GROUP_SIZE)
    for name, array, want in (
        ("down_proj.weight", down_weight, want_down_weight),
        ("down_proj.scales", down_scales, want_down_meta),
        ("down_proj.biases", down_biases, want_down_meta),
    ):
        if tuple(array.shape) != want:
            raise SharedLaneContractError(
                f"{where}: {name} shape {tuple(array.shape)} is not {want}"
            )

    if gu_weight.dtype != mx.uint32 or down_weight.dtype != mx.uint32:
        raise SharedLaneContractError(
            f"{where}: packed weights are {gu_weight.dtype}/{down_weight.dtype}, "
            "expected uint32"
        )
    if getattr(down, "bias", None) is not None:
        raise SharedLaneContractError(
            f"{where}: shared down projection carries a bias; the lane's "
            "reference branch does not add one"
        )


def _emit_branch(block: Any, x: mx.array) -> mx.array:
    """The shared-expert branch, verbatim from the retained M4 forward.

    Kept in one place so the lane and its reference cannot drift: both call
    this, the only difference being the stream in force at call time.
    """

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
    return shared.down_proj(shared_h).reshape(ROWS, HIDDEN)


def stock_shared_branch(block: Any, x: mx.array) -> mx.array:
    """The branch on whatever stream is in force (the shipped behaviour)."""

    COUNTERS["stock_calls"] += 1
    return _emit_branch(block, x)


def shared_branch(block: Any, x: mx.array) -> mx.array:
    """The branch on the lane's second GPU stream.

    ``x`` is produced on the caller's stream and ``shared_down`` is consumed
    there, so MLX inserts one fence in each direction; that is the whole cost
    of the lane and it is why the micro exists.
    """

    COUNTERS["branch_calls"] += 1
    with mx.stream(stream()):
        return _emit_branch(block, x)


def engagement_line(*, installed_layers: int, enabled: bool) -> str:
    """One-line engagement receipt for the serving log."""

    if not enabled:
        return "[fable] shared-lane: off"
    return (
        "[fable] shared-lane: on, "
        f"layers={installed_layers}, "
        f"branch_dispatches/layer={BRANCH_DISPATCHES_PER_LAYER} moved to a "
        f"second gpu stream, fence_dispatches/layer~{FENCE_DISPATCHES_PER_LAYER}, "
        f"weight_bytes/layer={BYTES_PER_LAYER}, "
        f"exactness_failures={COUNTERS['exactness_failures']}"
    )


__all__ = [
    "BRANCH_DISPATCHES_PER_LAYER",
    "BYTES_PER_LAYER",
    "COUNTERS",
    "DOWN_BYTES_PER_LAYER",
    "FENCE_DISPATCHES_PER_LAYER",
    "GU_BYTES_PER_LAYER",
    "HIDDEN",
    "INTERMEDIATE",
    "ROWS",
    "SHARED_BITS",
    "SHARED_GROUP_SIZE",
    "SHARED_MODE",
    "SharedLaneContractError",
    "check_contract",
    "engagement_line",
    "reset_counters",
    "reset_stream_cache",
    "shared_branch",
    "stock_shared_branch",
    "stream",
]
