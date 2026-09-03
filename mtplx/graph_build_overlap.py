"""W63 -- overlap the fixed-M4 verify graph's host construction with GPU work.

What this lane targets, measured on the RETAINED stack
------------------------------------------------------
W62 (``docs/perf/ple-boundary.md``) found, on the PR391 device-D3 census, that
1.780 ms of the 4.002 ms/cycle "PLE boundary" was not PLE work at all: it was
``dispatch["fn"](input_ids, compiled_aux, *state_in)`` -- the ``mx.compile``d
physical-M4 verify graph's own host-side replay over ~5,200 lazy nodes and
~137 array inputs -- plus ``_unpack_fixed_m4_outputs`` and the state/capture
rebind.  No sync happens in it.  The GPU idles for the whole of it.

Re-derived on the retained-stack control census
(``w58-retained-control-census-1788370322.jsonl``, 382 cycles, 32.402 busy +
7.301 idle ms/cycle), the verify body opens at a fixed offset of 3,668
dispatches before the cycle's ``lm_head``, and the GPU idle immediately before
that buffer is:

===========================================  ==========
idle before the compiled verify body          1.934 ms/cycle
host-late share                               86.9 %
cycles in which it appears                    382 / 382
median / min                                  1.690 / 1.257 ms
share of the retained stack's 7.301 ms idle   26.5 %
===========================================  ==========

**It did not shrink -- it grew** (D: 1.780 ms/cycle).  The retained-stack
reduce reports the "D3 -> PLE dequant -> target gather" boundary as only
0.613 ms/cycle in 112/382 cycles, and that is a CLASSIFIER artifact:
``census_retained_stack.is_ple_boundary`` keys on the kernel PAIR, and on the
retained stack the buffer that precedes the verify body ends in
``g1_copybfloat16bfloat16`` in 270 cycles and in the PLE q4 dequant in only
112.  Summing both families (1.267 + 0.613) recovers 1.880 ms/cycle, which is
the same event.

Ranking of the retained stack's 7.301 ms/cycle of idle:

* 3.843 ms/cycle -- the draft loop's per-depth host syncs (``v_Exp`` ->
  ``gather_front`` / ``gg1_copy``, ~3.7 events/cycle).  Not this lane.
* **1.934 ms/cycle -- this lane's target.**
* 0.554 ms/cycle -- ``gather_front(uint32)`` -> ``affine_dequantize gs64 b8``,
  only 39 % host-late, i.e. mostly driver latency.
* 0.496 ms/cycle -- the PLE auxiliary's own submission gap (W62's gap A),
  distinct in 112 cycles and folded into a busy buffer in the other 270.

The mechanism
-------------
``mtplx/graphbank.py`` already compiles the window in two pieces:
``install_fixed_m4_split`` builds a **prefix** graph (target embedding +
layer 0) and a **suffix** graph (layers 1..47 + head).  The split point is not
arbitrary -- it is exactly the PLE boundary in the *dataflow*:

* the production config has ``ple_layer_ids = [2]`` (one-indexed), i.e. one
  single PLE layer, at index **1**;
* ``qwen4_fixed_verify._forward_fixed_m4_prefix`` therefore runs layer 0
  WITHOUT ``compiled_verify_ple_scope`` -- it never reads ``compiled_aux``;
* ``_forward_fixed_m4_suffix`` opens the PLE scope and runs 1..47.

So layer 0 is the whole of the window that is independent of the PLE
auxiliary, and the PLE auxiliary is the only part of the window that needs the
drafted tokens **on the host** (the n-gram row ids are host NumPy over the
window's token values).

This lane submits the prefix at the earliest statement that owns the window --
immediately after ``verify_input_array`` is built, which is ahead of
``prepare_aux``'s PLE row read, ahead of the ~1.9 ms suffix replay, and one
statement after the drafted ids arrive.  The GPU then has layer 0 to run
during host time it was already spending.

Ceiling: ``min(prefix GPU, host build)``.  The verify body is 3,668 of the
cycle's 4,685 dispatches and ~25.4 ms of its 32.4 ms of GPU, so one of 48
layers is **~0.53 ms/cycle**, which is the binding term (the idle is 1.93 ms).
On a 37.4 ms production window that is **~1.4 %** -- roughly 2x the ABBA
within-seed floor (0.3-0.7 % = 0.11-0.26 ms), so it is measurable but not
comfortable.

What this lane does NOT claim
-----------------------------
It does not make the host build shorter.  Two compiled calls replay the same
~5,200 nodes as one, plus a second tree-flatten and a second ``async_eval``;
the host side is very slightly *more* expensive.  Every millisecond claimed
here is GPU work moved under host time that was already being spent.

Nor does it reach the rest of the 1.93 ms.  Layers 1..47 are dataflow
descendants of ``compiled_aux``, which cannot exist before the host knows the
drafted token values, so their ~5,090 nodes cannot be replayed early.

W67: the N-layer prefix
-----------------------
W63 stopped at layer 0 because layer 0 is the only layer that reads no PLE
auxiliary.  ``MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS=N`` (default 1, i.e.
W63's partition) moves the seam to layer ``N-1 / N``.  Saving is
``min(N x 0.53, (48-N)/48 x 1.93)``, which peaks near **N = 3-4 at
~1.7-1.8 ms/cycle (~4.5 %)**.

Three things make N > 1 legal:

1. **The aux hoist.**  ``prepare_aux`` needs only ``host_input_ids`` (the
   window's four token VALUES, a Python list), ``completion_tokens`` and
   ``committed_count`` -- all three exist at the enqueue statement, and none
   of them is produced by any layer.  So for N > 1 the auxiliary is built at
   the ENQUEUE, before the prefix, and carried on the prefix object to the
   join.  It is built exactly once per window either way.
2. **A generalized split.**  ``install_fixed_m4_split`` (PR391's, whose
   layer-0 census ``tests/test_qwen4_fixed_host_tokens_static`` pins by
   source) is left alone; ``install_fixed_m4_overlap_split`` partitions the
   state plan, the capture layout and the layer range at an arbitrary N and
   RAISES at the request boundary on any census it does not recognise.
3. **One more fusion seam.**  See "Exactness" below.

At N = 1 the prefix takes no auxiliary at all (its ``mx.compile`` closure has
no ``compiled_aux`` parameter) and the aux is prepared in the join exactly
where W63 prepared it, so the default arm is W63's schedule.

Where the seam sits, per N (production geometry: 48 layers,
``full_attention_interval=4`` so layers 3, 7, ... are QSA; ``ple_layer_ids ==
[2]`` one-indexed, i.e. the single PLE layer is index 1):

======  ===========================  =================================
N       last prefix layer            first suffix op reading the seam
======  ===========================  =================================
1       0  (GDN, no PLE)             ``hidden + ple(hidden, ids)``
2       1  (GDN, **the PLE layer**)  ``attn_hyper_connection(hidden)``
3       2  (GDN)                     ``attn_hyper_connection(hidden)``
4       3  (**QSA**)                 ``attn_hyper_connection(hidden)``
======  ===========================  =================================

Every producer is ``_hyper_residual_write`` (the MLP hyper-connection write,
an elementwise multiply-add ending in a reshape).  At N = 1 the consumer is
another elementwise add, so the seam cuts an elementwise chain that MLX could
have fused.  At N >= 2 the consumer is ``GatedResidual``, whose first
operation is a ``GroupedRMSNorm`` (``mx.fast.rms_norm``) or, under
``MTPLX_FABLE_HC_M4``, a hand-written Metal kernel -- neither of which fuses
with an elementwise producer.  **N >= 2 therefore cuts at a cleaner seam than
N = 1 does.**  That is an argument, not a proof; the ABBA's token digest is
still the gate.

Exactness
---------
The values fed to the two graphs are the values the monolithic route feeds to
one:

* ``input_ids`` is the SAME ``verify_input_array`` object -- the hook does not
  rebuild it, so there is nothing to argue about.
* ``compiled_aux`` is produced by the unchanged ``dispatch["prepare_aux"]``
  from the unchanged ``host_input_ids``.
* the layer-0 state leaves, the layers-1..47 state leaves, the capture leaves
  and the ``async_eval`` boundaries are published in the same order and to the
  same attributes as ``_forward_installed_fixed_m4``, so the device commit
  (``qwen4_fixed_verify._bind_fixed_m4_device_commit``) sees an identical
  ``_mtplx_verify_rows`` / ``_mtplx_verify_ple`` / ``_mtplx_verify_compiled_aux``
  census.  ``tests/test_fable_graph_build_overlap`` asserts that entry by
  entry and slot by slot against the shipped path rather than asserting the
  argument.

**This is an argument about inputs, not a proof about bits.**  What changes is
that an ``mx.compile`` boundary now sits between layer 0 and layer 1.  MLX
fuses element-wise chains inside one compiled graph, and a fused chain can
keep an intermediate in a wider register where an unfused one round-trips
through bf16 memory -- ``_bind_fixed_m4_device_commit`` carries a comment about
exactly this class of effect ("the outer graph changes the arithmetic schedule
in rounding-sensitive states, first observed at seed 31, cycle 91").  A
last-bit difference at the layer-0/layer-1 seam is therefore *possible*, and
the gate is empirical: the ABBA driver's ``response_token_sha256`` and its
per-cycle ``accepted`` list must match the control exactly.  If they do not,
this lever is dead as written and the report must say so.

Flag
----
``MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1`` arms it; unset or ``0`` runs the shipped
monolithic route with a constant-``None`` hook in the decode loop.
``MTPLX_FABLE_GRAPH_BUILD_OVERLAP_ITEMS=timing`` additionally records the host
seconds spent in each of the two replays (an instrument, not a lever -- it is
not in :data:`DEFAULT_ITEMS`, so an A/B measures the lever alone).
``MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS=N`` (default 1) sets the prefix depth.
"""

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache

from .fable_claim_contract import STRICT_ENV, strict_claims

__all__ = [
    "DEFAULT_ITEMS",
    "DEFAULT_LAYERS",
    "ENV_FLAG",
    "ITEMS",
    "ITEMS_ENV",
    "LAYERS_ENV",
    "MAX_LAYERS",
    "TIMING",
    "bump",
    "enabled",
    "engagement_line",
    "item",
    "items",
    "last_receipt",
    "layers",
    "note_aux_hoisted",
    "note_construction",
    "note_first_prefix_build",
    "note_first_suffix_build",
    "note_prefix_build",
    "note_prefix_layers",
    "note_suffix_build",
    "reset_receipt",
    "timing_enabled",
]

ENV_FLAG = "MTPLX_FABLE_GRAPH_BUILD_OVERLAP"
ITEMS_ENV = "MTPLX_FABLE_GRAPH_BUILD_OVERLAP_ITEMS"
#: W67: how many leading target layers the prefix graph carries.  ``1`` is
#: W63's layer-0 prefix, i.e. the default is exactly the shipped behaviour.
LAYERS_ENV = "MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS"
DEFAULT_LAYERS = 1
#: A ceiling, not a recommendation.  ``min(N x per-layer GPU, (L-N)/L x host
#: build)`` peaks at N = 3-4 on the production geometry and falls off after,
#: and a prefix that swallows most of the window has no host build left to
#: hide under; refuse an obviously-wrong N at the flag rather than compile it.
MAX_LAYERS = 8

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Every selectable item.  A name not in here raises at resolution time: an
#: A/B whose candidate silently ran nothing is worse than one that fails.
ITEMS: tuple[str, ...] = ("timing",)

#: What ``MTPLX_FABLE_GRAPH_BUILD_OVERLAP=1`` alone means.  The lever is the
#: flag; ``timing`` is an instrument and costs two ``perf_counter`` pairs per
#: cycle, so it is opt-in by name.
DEFAULT_ITEMS: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def enabled() -> bool:
    """Resolve :data:`ENV_FLAG` once.  Unparseable values raise."""

    raw = (os.environ.get(ENV_FLAG) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{ENV_FLAG} must be one of {accepted}, got {os.environ.get(ENV_FLAG)!r}"
    )


@lru_cache(maxsize=1)
def items() -> frozenset[str]:
    """The armed item set, resolved once.  Empty when the flag is off."""

    if not enabled():
        return frozenset()
    raw = (os.environ.get(ITEMS_ENV) or "").strip()
    if not raw:
        return frozenset(DEFAULT_ITEMS)
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = tuple(name for name in names if name not in ITEMS)
    if unknown:
        raise ValueError(
            f"{ITEMS_ENV} names unknown item(s) {unknown!r}; "
            f"known items are {ITEMS}"
        )
    return frozenset(names)


@lru_cache(maxsize=1)
def layers() -> int:
    """Resolve :data:`LAYERS_ENV` once.  Unset means ``DEFAULT_LAYERS``.

    This is the requested prefix depth, not the installed one: whether the
    fixed-M4 plan can actually be partitioned there is
    ``CompiledVerifyBank.install_fixed_m4_overlap_split``'s question, and it
    raises at the request boundary rather than degrading silently.
    """

    raw = (os.environ.get(LAYERS_ENV) or "").strip()
    if not raw:
        return DEFAULT_LAYERS
    try:
        value = int(raw, 10)
    except ValueError:
        raise ValueError(
            f"{LAYERS_ENV} must be an integer in [1, {MAX_LAYERS}], got {raw!r}"
        ) from None
    if not 1 <= value <= MAX_LAYERS:
        raise ValueError(
            f"{LAYERS_ENV} must be an integer in [1, {MAX_LAYERS}], got {value}"
        )
    return value


def item(name: str) -> bool:
    """Whether one item is armed.  Unknown names raise."""

    if name not in ITEMS:
        raise ValueError(f"unknown {ENV_FLAG} item {name!r}; known items {ITEMS}")
    return name in items()


@lru_cache(maxsize=1)
def timing_enabled() -> bool:
    """Whether the two host build timers are armed."""

    return item("timing")


#: Module constant read ONCE at import by ``mtplx.graphbank``: with the flag
#: off this is False and the sites there are constant-False branches, so the
#: control arm runs the code it ran before this module existed.
try:
    TIMING = timing_enabled()
except ValueError:  # a malformed flag must fail at the lane, not at import
    TIMING = False


_RECEIPT_ZERO: dict[str, float] = {
    # Windows whose layer-0 prefix was queued ahead of the D3 host sync.
    "prefix_enqueued": 0,
    # Windows that joined a queued prefix with the suffix graph.
    "suffix_joined": 0,
    # Windows that ran the shipped monolithic route with no prefix queued.
    "monolithic_windows": 0,
    # A queued prefix abandoned without a join (capacity/route generation
    # moved under it, or the cycle never reached the M4 verify).
    "prefix_discarded": 0,
    # install_fixed_m4_overlap_split re-runs after a capacity/route transition.
    "split_rebuilds": 0,
    # Installs that reused the PROCESS-wide compiled pair instead of tracing
    # a fresh one.  On a served process every install after the first should
    # be a hit; a rising `split_rebuilds` with a flat `split_shared_hits`
    # means every request is paying two fresh mx.compile traces.
    "split_shared_hits": 0,
    # Host ms spent in install_fixed_m4_overlap_split itself (closure
    # construction + mx.compile wrapping + the census assertions).  Always
    # on: this is the number that answers "did arming the lane cost TTFT?".
    "construction_ms": 0.0,
    "construction_calls": 0,
    # The FIRST replay of each graph, which is where mx.compile actually
    # traces.  Always on and first-call only, so the one-time trace is
    # separable from steady state without arming the `timing` item.
    "first_prefix_build_ms": 0.0,
    "first_suffix_build_ms": 0.0,
    # W67: the prefix depth actually INSTALLED (not the one requested).  Set
    # once, by the install; a receipt whose `prefix_layers` is not the N the
    # arm asked for measured a different partition than its label claims.
    "prefix_layers": 0,
    # W67: windows whose PLE auxiliary was built at the ENQUEUE (the hoist),
    # i.e. windows whose prefix contains the PLE layer.  Zero at N=1.
    "aux_hoisted": 0,
    # timing item (ms, cumulative over the request).
    "prefix_build_ms": 0.0,
    "prefix_build_calls": 0,
    "suffix_build_ms": 0.0,
    "suffix_build_calls": 0,
    # Requests the lane stood aside for because the SHAPE of the request did
    # not offer the physical-M4 compiled verify to split (a capture_commit
    # verify strategy, compiled verify off for this context length).  Those
    # run the shipped verify.  See `mtplx.fable_claim_contract`.
    "declines": 0,
}

_RECEIPT: dict[str, float] = dict(_RECEIPT_ZERO)


def last_receipt() -> dict[str, float]:
    """The lane's cumulative engagement receipt for this process."""

    receipt = dict(_RECEIPT)
    receipt["items"] = ",".join(sorted(items()))
    return receipt


def reset_receipt() -> None:
    _RECEIPT.update(_RECEIPT_ZERO)


def bump(name: str, value: float = 1) -> None:
    """Fold one engagement event into the receipt.  Unknown names raise."""

    if name not in _RECEIPT_ZERO:
        raise KeyError(f"unknown {ENV_FLAG} receipt counter {name!r}")
    _RECEIPT[name] = _RECEIPT[name] + value


def decline(detail: str) -> None:
    """Stand aside for one request that offers nothing to overlap.

    The armed flag needs the installed physical-M4 compiled verify on the
    batched verify route.  Whether a request HAS one is request-shaped
    (``verify_strategy`` is a per-request argument, and the compiled verify
    can be off for this context length), so a miss must not raise: that turned
    every such request into an HTTP 500.  One warning per process, a receipt
    counter, and the shipped verify runs.
    """

    if strict_claims():
        raise RuntimeError(
            f"{ENV_FLAG}: {detail} "
            f"[{STRICT_ENV}=1 turns request-time declines into failures]"
        )
    _RECEIPT["declines"] = _RECEIPT["declines"] + 1
    if _RECEIPT["declines"] == 1:
        print(
            f"[{ENV_FLAG}] declined: {detail} -- this request runs the "
            f"shipped verify; set {STRICT_ENV}=1 to fail closed instead",
            file=sys.stderr,
            flush=True,
        )


def note_prefix_layers(count: int) -> None:
    """Record the prefix depth the install actually compiled."""

    _RECEIPT["prefix_layers"] = int(count)


def note_construction(seconds: float) -> None:
    """Fold one ``install_fixed_m4_overlap_split`` into the receipt."""

    _RECEIPT["construction_ms"] = _RECEIPT["construction_ms"] + seconds * 1000.0
    _RECEIPT["construction_calls"] = _RECEIPT["construction_calls"] + 1


def note_first_prefix_build(seconds: float) -> None:
    """The prefix graph's first replay -- i.e. its ``mx.compile`` trace."""

    _RECEIPT["first_prefix_build_ms"] = seconds * 1000.0


def note_first_suffix_build(seconds: float) -> None:
    """The suffix graph's first replay -- i.e. its ``mx.compile`` trace."""

    _RECEIPT["first_suffix_build_ms"] = seconds * 1000.0


def note_aux_hoisted() -> None:
    """Record one window whose PLE auxiliary was built before the prefix."""

    _RECEIPT["aux_hoisted"] = _RECEIPT["aux_hoisted"] + 1


def note_prefix_build(seconds: float) -> None:
    """Fold one N-layer prefix host replay into the receipt."""

    _RECEIPT["prefix_build_ms"] = _RECEIPT["prefix_build_ms"] + seconds * 1000.0
    _RECEIPT["prefix_build_calls"] = _RECEIPT["prefix_build_calls"] + 1


def note_suffix_build(seconds: float) -> None:
    """Fold one layers-N..47 suffix host replay into the receipt."""

    _RECEIPT["suffix_build_ms"] = _RECEIPT["suffix_build_ms"] + seconds * 1000.0
    _RECEIPT["suffix_build_calls"] = _RECEIPT["suffix_build_calls"] + 1


def engagement_line(installed_layers: int | None = None) -> str | None:
    """The one line a request prints when this lane is armed, else ``None``.

    ``installed_layers`` is what the bank actually compiled; when it differs
    from :func:`layers` the line says both, because a reader who sees only the
    requested N would attribute the arm's number to the wrong partition.
    """

    if not enabled():
        return None
    armed = ",".join(sorted(items())) or "-"
    requested = layers()
    depth = requested if installed_layers is None else int(installed_layers)
    mismatch = "" if depth == requested else f" (requested {requested})"
    return (
        f"[{ENV_FLAG}] armed: fixed-M4 verify split at layer {depth - 1}/{depth}"
        f"{mismatch}; layers 0..{depth - 1} queued at verify_input_array, "
        f"ahead of the suffix replay; items={armed}"
    )


def now() -> float:
    """``time.perf_counter`` behind one name so tests can see the call site."""

    return time.perf_counter()
