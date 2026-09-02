"""Keep-mask fold: defer the eager GDN prefix replay into the next M4 window.

WHAT IT REMOVES (measured, not argued)
--------------------------------------
On the retained control lane (``--target-mode batched --require-compiled-verify
--m4-stage3 --qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare`` plus
``MTPLX_FAMILY_CAPTURE_COMMIT=1`` and ``MTPLX_SKIP_VERIFY_SNAPSHOT=0``) each
decode cycle makes exactly TWO passes over every GDN layer's recurrent state:

  1. the compiled 4-row verify's own ``gated_delta_step`` (36 dispatches/cycle,
     ``state_in`` = the pre-window state, one 3.146 MB read + one 3.146 MB
     write per layer -> 226.5 MB/cycle).  Its state output is thrown away on
     every cycle that does not accept the whole window.
  2. ``Qwen4ExpTextModel.commit_verified_window``'s eager replay of the KEPT
     prefix from the pre-verify snapshot -- again 36 dispatches, again
     226.5 MB, and only on the cycles that rejected something.

The W58 retained-stack census (382 cycles, ``w58-retained-control-census``)
measures both directly.  ``custom_kernel_gated_delta_step__bfloat16_t_float_
128_128_16_48`` runs 62.09 times per cycle: 36.0 verify + 26.09 replay.  The
census also contains a natural A/B -- an all-accept cycle returns at
``generation.py`` before the commit, so it runs 36 and no replay:

    all-accept cycles      n=112   4,317.7 dispatches   30.008 ms GPU busy
    partial-accept cycles  n=262   4,548.8 dispatches   30.706 ms GPU busy
    delta                          +231.1 dispatches    +0.699 ms GPU busy
                                                        +1.569 ms wall (median)

So the replay costs **0.70 ms of GPU time and 1.57 ms of wall on 69 % of
cycles** = 0.48 ms GPU / 1.08 ms wall averaged over every cycle of a 37.4 ms
window.  Of the +231 dispatches, 35.53 are the state kernel itself (223.5 MB),
35.53 are ``compute_g``, ~36 are ``sigmoid(b)``, ~103 are copies (the conv
window commit plus the row slices) and 24 are the QSA ``trim`` -- only the
first is byte-bound, and only the first is what this module removes.

HOW THE FOLD WORKS
------------------
The recurrence state is float32 in registers inside
``mlx_lm.models.gated_delta``'s kernel and float32 in memory
(``StT = float``), and the kernel loads it once, iterates ``t = 0..T-1`` and
stores it once.  Therefore **splitting or merging the T loop is bit-exact**:
running rows ``[0..k)`` in one dispatch (state -> memory -> state) and rows
``[k..k+4)`` in a second is the same float arithmetic in the same order as
running all ``k+4`` in one dispatch.  Every fp32 store/load round trip in
between is the identity.

So instead of replaying the kept prefix now, we keep the pre-window state as
the base and hand the kept rows to the NEXT window's step kernel, which reads
and writes the state once for both.  The rejected rows are not sliced away:
they ride along under the kernel's own ``mask`` argument, whose ``else`` branch
writes ``y = 0`` and leaves the register state untouched -- an exact no-op on
the state.  That keeps every prefix tensor at the captured ``[1, 4, ...]``
shape, so the padded prefix is one fixed shape and the compiled verify graph is
traced exactly once.

FAIL-SAFE BY CONSTRUCTION
-------------------------
The deferred state is not "missing".  ``commit_verified_window`` still assigns
a real ``mx.array`` to ``cache[1]`` -- the *lazy, unevaluated* masked replay
from the base over the ring.  Anything that reads or evaluates that leaf (a
context-copy block round, a rollback re-forward, ``detach_cache_state``,
``MTPLX_EVAL_STATE_ROOTS_ON_COMMIT``, the session bank) gets exactly the state
it gets today, at exactly today's cost.  The fold is purely an optimisation the
next compiled window may take: it recognises its own pending leaf, passes the
base and the ring rows instead, and lets MLX drop the unevaluated replay.  A
leaf it does not recognise is passed through unchanged.  There is therefore no
flush protocol to get wrong and no site that can silently read a stale state.

RING AND FLUSH POLICY
---------------------
The ring holds whole verify windows (4 captured rows + a keep count each), so
the padded prefix is ``4 * MAX_WINDOWS`` rows and the folded step runs at
``T = 4 * MAX_WINDOWS + 4``.  It resets to empty on every all-accept cycle (the
graph's own state output is already the authoritative base, and
``commit_verified_window`` is not called at all).  When a partial accept
arrives with a full ring, the pending leaf becomes the new base -- i.e. MLX
evaluates one masked replay covering the whole ring, one state pass for
several windows instead of one per window.

With the measured accept law (P(all-accept) = 0.295) the ring is a Markov chain
whose flush rate is 0.206/cycle at MAX_WINDOWS = 2 and 0.112 at 3, against
today's 0.705 replays/cycle:

    MAX_WINDOWS   T    replays/cycle   state MB/cycle removed
        1         8       0.497              47
        2        12       0.206             113
        3        16       0.112             134
        4        20       0.068             144

At 318-394 GB/s (the census's fitted GDN family rate) 113 MB is 0.29-0.36 ms of
the 37.4 ms window.

THE FALSIFIER, AND WHY THIS LANE IS OFF BY DEFAULT
--------------------------------------------------
The fold trades ``0.499`` state passes per cycle for ``4 * MAX_WINDOWS`` extra
``t`` iterations in every one of the 36 verify step kernels, most of them
masked.  If ``gated_delta_step`` is state-bound its wall time is flat in T and
the trade is free; if it is T-bound the extra iterations cancel the win.
Nothing in the census answers that -- it records dispatch counts and command
buffer times, not per-kernel times.  ``scripts/fable/micro_gdn_keepmask_fold.py``
measures it directly (queued lane, under the flock) and also proves the
split-vs-merged bit-exactness on the production shape.  Run it BEFORE any ABBA
window: a flat T curve arms the lane, a linear one kills it.

STATUS -- WHAT IS WIRED AND WHAT IS NOT (read this before arming anything)
--------------------------------------------------------------------------
WIRED: the flag, the ring policy, the mask, the contract/install gate, the
counters, the exact fold primitives (``mtplx/kernels/gdn_keepmask_fold.py``)
and the guarded micro that prices them.

NOT WIRED: the compiled verify graph does not yet take a prefix.  That is a
deliberate stop, not an omission.  The fold only pays if
``gated_delta_step``'s wall time is flat in ``T`` -- it trades 0.499 state
passes per cycle (110 MB, 0.28-0.35 ms at 318-394 GB/s) for
``4 * ring_windows`` extra ``t`` iterations in all 36 verify step kernels.
The census measures dispatch counts and command-buffer times, not per-kernel
times, so nothing in it decides that; ``scripts/fable/micro_gdn_keepmask_fold
.py`` arm A does, in one guarded window.  Wiring 150 lines through the
compiled M4 graph before that number exists would be building on a coin flip.

WIRING PLAN (once arm A says state-bound)
-----------------------------------------
The design is fail-safe: there is no flush protocol and no site that can read
a stale state.

1. ``Qwen4ExpTextModel.commit_verified_window`` (mtplx/models/qwen4_exp.py):
   for the 35 non-PLE GDN layers, instead of
   ``gated_delta_update(q[:, :keep], ..., pre[1], None)``, bind
   ``entry.cache[1] = masked_replay_state(ring_rows, keeps, ..., base)``
   -- the SAME state, built lazily and left unevaluated -- and hang a
   ``FoldPending(base, rows, keeps, state)`` off the entry.  The conv-state
   commit, the PLE layer's exact-width replay and the QSA trims are unchanged.
   ``pre[1]`` is itself the previous cycle's pending leaf, so
   ``pending_for(entry)`` on it yields the base and the ring to extend;
   ``ring_after_commit`` decides extend vs flush, and a flush is just
   "treat the pending leaf as the new base" (MLX evaluates one masked replay
   covering the whole ring -- one state pass for several windows).

2. ``CompiledVerifyBank._forward_installed_fixed_m4`` (mtplx/graphbank.py):
   when a GDN entry's slot-1 leaf is its own live pending leaf, push ``base``
   in its place and append the layer's six padded prefix leaves
   (``q, k, v, a, b`` at ``[1, 4*W, ...]`` plus the ``[1, 4*W]`` bool mask) to
   the trailing args.  Otherwise pass the leaf as it does today.  After the
   call, publish slot 1 as it does today; ``commit_verified_window`` (partial
   accept) or the all-accept branch's early return (which never commits, so
   the graph's own full-window state stays and the ring is empty by
   definition) leaves the invariant intact.

3. ``CompiledVerifyBank._make_verify_step``: ``state_in`` is consumed
   positionally by ``spec``; the trailing prefix args are simply
   ``state_in[pos:]``.  Install them in a contextvar keyed by
   ``id(shadow_entry)`` before the forward.

4. ``GatedDeltaNet.__call__`` (mtplx/models/qwen4_exp.py): capture
   ``_mtplx_verify_rows`` from the NEW rows first (unchanged), then, if a
   prefix is in scope for this cache entry, run
   ``prefix_gated_delta_update`` (or ``folded_gated_delta_update`` on the
   no-new-Metal route) and keep only the window rows of ``y``.
   ``cache.advance(S)`` still advances by the window width.

5. Add a ``fold`` dimension to ``_SHARED_VERIFY_STEPS``'s global key so a
   folded trace can never be served to a bank that is not passing a prefix.

FAIL-SAFE PROPERTY.  Every non-fold consumer -- a context-copy block round, a
rollback re-forward, ``detach_cache_state``,
``MTPLX_EVAL_STATE_ROOTS_ON_COMMIT``, the session bank -- reads
``entry.cache[1]``, which is a real array holding the correct state; forcing
it costs exactly today's replay.  ``pending_for`` drops the descriptor the
moment anything else rebinds the leaf, so a rollback or a trim silently
degrades the fold to today's behaviour instead of corrupting it.

ENV
---
``MTPLX_FABLE_GDN_KEEPMASK_FOLD=1``            arm the lane (default 0)
``MTPLX_FABLE_GDN_KEEPMASK_FOLD_WINDOWS=N``    ring depth, 1..4 (default 2)
``MTPLX_FABLE_GDN_KEEPMASK_FOLD_LOG=0``        silence the engagement line
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

#: The one geometry this lane is wired for (Qwen3.8 Flash-Next 125B-A6B).
VERIFY_WIDTH = 4
NUM_V_HEADS = 48
NUM_K_HEADS = 16
HEAD_DIM = 128
#: float32 recurrent state bytes per GDN layer: 48 * 128 * 128 * 4.
STATE_BYTES = NUM_V_HEADS * HEAD_DIM * HEAD_DIM * 4
#: GDN layers in the production text model (the 48-layer trunk is 36 GDN +
#: 12 QSA); the single PLE-carrying GDN layer is excluded from the fold.
GDN_LAYERS = 36
PLE_LAYERS = 1
FOLDABLE_LAYERS = GDN_LAYERS - PLE_LAYERS

ENV_FLAG = "MTPLX_FABLE_GDN_KEEPMASK_FOLD"
ENV_WINDOWS = "MTPLX_FABLE_GDN_KEEPMASK_FOLD_WINDOWS"
ENV_LOG = "MTPLX_FABLE_GDN_KEEPMASK_FOLD_LOG"

DEFAULT_MAX_WINDOWS = 2
MAX_WINDOWS_CHOICES = (1, 2, 3, 4)


class GdnKeepMaskFoldContractError(RuntimeError):
    """The model/cache cannot support an exact keep-mask fold."""


# --------------------------------------------------------------------------
# Flags (read once at first use, never inside a measured cycle)
# --------------------------------------------------------------------------

_ENABLED: bool | None = None
_MAX_WINDOWS: int | None = None


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def fable_gdn_keepmask_fold_enabled() -> bool:
    """Return the ``MTPLX_FABLE_GDN_KEEPMASK_FOLD`` gate; default off.

    Memoised for the same reason every other ``MTPLX_FABLE_*`` gate is: the
    hot path must not touch ``os.environ``, and two traces of the same
    compiled verify graph must not disagree about which recurrence they hold.
    """

    global _ENABLED
    if _ENABLED is None:
        _ENABLED = _env_bool(ENV_FLAG, default=False)
    return _ENABLED


def fable_gdn_keepmask_fold_windows() -> int:
    """Ring depth in whole verify windows.  Raises on an unsupported value.

    A typo'd sweep value must fail at flag capture rather than silently
    measure the default arm.
    """

    global _MAX_WINDOWS
    if _MAX_WINDOWS is None:
        raw = os.environ.get(ENV_WINDOWS)
        if raw is None or raw.strip() == "":
            value = DEFAULT_MAX_WINDOWS
        else:
            try:
                value = int(raw.strip())
            except ValueError as exc:
                raise ValueError(f"{ENV_WINDOWS}={raw!r} is not an integer") from exc
            if value not in MAX_WINDOWS_CHOICES:
                raise ValueError(
                    f"{ENV_WINDOWS}={value} is not one of {MAX_WINDOWS_CHOICES}"
                )
        _MAX_WINDOWS = value
    return _MAX_WINDOWS


def prefix_rows() -> int:
    """Padded prefix width -- the fixed shape the compiled graph is traced at."""

    return VERIFY_WIDTH * fable_gdn_keepmask_fold_windows()


def reset_fable_gdn_keepmask_fold_cache() -> None:
    """Drop the memoised gates.  Test support only."""

    global _ENABLED, _MAX_WINDOWS
    _ENABLED = None
    _MAX_WINDOWS = None


# --------------------------------------------------------------------------
# Counters + one engagement line
# --------------------------------------------------------------------------

STATS: dict[str, Any] = {
    "armed": False,
    "installed": False,
    "install_status": "disabled",
    "install_error": None,
    "max_windows": 0,
    "prefix_rows": 0,
    "folded_layers": 0,
    "windows": 0,             # compiled M4 windows seen by the fold hook
    "folded_windows": 0,      # windows that took the folded recurrence
    "deferred_commits": 0,    # partial accepts that deferred instead of replaying
    "flushes": 0,             # ring-full events (one masked replay each)
    "ring_depth_hist": {},    # ring length at window entry -> count
    "state_passes_saved": 0,  # layer-level passes not dispatched
    "state_bytes_saved": 0,
}

_LOGGED = False


def _log_engagement(message: str) -> None:
    global _LOGGED
    if _LOGGED or not _env_bool(ENV_LOG, default=True):
        return
    _LOGGED = True
    print(f"[fable] gdn_keepmask_fold {message}", flush=True)


def reset_stats() -> None:
    """Test support: clear the counters and the one-shot log latch."""

    global _LOGGED
    _LOGGED = False
    STATS.update(
        {
            "armed": False,
            "installed": False,
            "install_status": "disabled",
            "install_error": None,
            "max_windows": 0,
            "prefix_rows": 0,
            "folded_layers": 0,
            "windows": 0,
            "folded_windows": 0,
            "deferred_commits": 0,
            "flushes": 0,
            "ring_depth_hist": {},
            "state_passes_saved": 0,
            "state_bytes_saved": 0,
        }
    )


def stats_snapshot() -> dict[str, Any]:
    """A copy of the counters, safe to embed in a receipt."""

    snapshot = dict(STATS)
    snapshot["ring_depth_hist"] = dict(STATS["ring_depth_hist"])
    return snapshot


# --------------------------------------------------------------------------
# Ring policy -- pure Python, no MLX, fully unit-testable
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RingDecision:
    """What one commit does to the ring.

    ``flush`` means the pending leaf becomes the next base: MLX evaluates one
    masked replay over the whole ring (one state pass) instead of one per
    window.  ``keeps`` is the post-decision ring, oldest first.
    """

    flush: bool
    keeps: tuple[int, ...]


def ring_after_commit(
    keeps: Sequence[int], accepted_keep: int, *, max_windows: int
) -> RingDecision:
    """Advance the ring by one committed window.

    ``accepted_keep`` is ``accepted_count + 1`` -- the number of rows of the
    just-verified window that were committed, in ``1..VERIFY_WIDTH``.  A full
    window (``VERIFY_WIDTH``) never reaches here: ``generation.py`` returns on
    the all-accept branch without committing, and the graph's own state output
    is already the authoritative base.
    """

    if not 1 <= int(accepted_keep) <= VERIFY_WIDTH:
        raise ValueError(
            f"accepted_keep must be in 1..{VERIFY_WIDTH}; got {accepted_keep}"
        )
    if int(accepted_keep) == VERIFY_WIDTH:
        raise ValueError(
            "a whole-window accept never commits through the fold; "
            "generation.py returns before commit_verified_window"
        )
    if int(max_windows) not in MAX_WINDOWS_CHOICES:
        raise ValueError(f"max_windows must be one of {MAX_WINDOWS_CHOICES}")
    current = tuple(int(k) for k in keeps)
    if len(current) >= int(max_windows):
        return RingDecision(flush=True, keeps=(int(accepted_keep),))
    return RingDecision(flush=False, keeps=current + (int(accepted_keep),))


def prefix_mask_rows(keeps: Sequence[int], *, max_windows: int) -> list[bool]:
    """The padded ``[4 * max_windows]`` boolean prefix mask, oldest window last.

    The ring's windows occupy the LAST ``4 * len(keeps)`` slots so the rows
    that feed the recurrence are adjacent to the new window's four rows; the
    leading pad slots are all False, and the kernel's ``else`` branch makes
    them exact no-ops on the state.  Within a window, row ``t`` is live iff
    ``t < keep``: rows after the rejection boundary were never committed.
    """

    width = VERIFY_WIDTH * int(max_windows)
    live: list[bool] = []
    for keep in keeps:
        if not 1 <= int(keep) <= VERIFY_WIDTH:
            raise ValueError(f"keep must be in 1..{VERIFY_WIDTH}; got {keep}")
        live.extend(index < int(keep) for index in range(VERIFY_WIDTH))
    if len(live) > width:
        raise ValueError(
            f"ring of {len(keeps)} windows exceeds max_windows={max_windows}"
        )
    return [False] * (width - len(live)) + live


def expected_state_passes_per_cycle(
    p_all_accept: float, *, max_windows: int
) -> float:
    """Stationary replay rate of the ring under an i.i.d. all-accept process.

    The chain on ring length ``l`` is: all-accept (probability ``p``) -> 0;
    partial -> ``l + 1`` while ``l < max_windows``, else flush and -> 1.  The
    return value is the expected number of MASKED REPLAY passes per cycle,
    against ``1 - p`` for today's eager replay.  Used by the tests and by the
    module docstring's table; it never runs in the hot path.
    """

    p = float(p_all_accept)
    if not 0.0 < p < 1.0:
        raise ValueError("p_all_accept must be strictly between 0 and 1")
    n = int(max_windows)
    if n not in MAX_WINDOWS_CHOICES:
        raise ValueError(f"max_windows must be one of {MAX_WINDOWS_CHOICES}")
    q = 1.0 - p
    # pi[0] = p.  For 1 <= l <= n: pi[l] = q * pi[l-1], except pi[1] which also
    # receives the flush return from pi[n].  Solve pi[1] in closed form.
    #   pi[1] = q * (pi[0] + pi[n]),  pi[n] = q**(n-1) * pi[1]
    pi1 = q * p / (1.0 - q**n)
    pin = (q ** (n - 1)) * pi1
    return q * pin


# --------------------------------------------------------------------------
# Pending descriptor -- what a deferred commit leaves on the cache entry
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FoldPending:
    """A deferred GDN commit for one layer.

    ``state`` is the lazy masked replay of ``rows`` over ``base``; it is what
    ``cache[1]`` holds, so every non-fold consumer sees the correct state and
    simply pays today's replay.  ``rows`` is oldest window first, each entry a
    ``(q, k, v, a, b)`` tuple of the FULL captured ``[1, 4, ...]`` rows -- the
    keep counts live in ``keeps`` and are applied by the mask, never by a
    slice.
    """

    base: Any
    rows: list[tuple[Any, Any, Any, Any, Any]] = field(default_factory=list)
    keeps: tuple[int, ...] = ()
    state: Any = None

    def depth(self) -> int:
        return len(self.rows)


def pending_for(entry: Any) -> FoldPending | None:
    """The live pending descriptor for a cache entry, or ``None``.

    Fail-safe: the descriptor is only honoured while it still owns the entry's
    state leaf.  Anything that rebinds ``cache[1]`` -- a rollback, a trim, a
    detach, a re-forward -- invalidates it, and the caller falls back to the
    plain leaf, which is already the correct state.
    """

    pending = getattr(entry, "_mtplx_fold_pending", None)
    if pending is None:
        return None
    try:
        if pending.state is not entry.cache[1]:
            entry._mtplx_fold_pending = None
            return None
    except Exception:
        entry._mtplx_fold_pending = None
        return None
    return pending


def clear_pending(entry: Any) -> None:
    """Drop any pending descriptor (the entry's leaf is authoritative again)."""

    if getattr(entry, "_mtplx_fold_pending", None) is not None:
        entry._mtplx_fold_pending = None


# --------------------------------------------------------------------------
# Contract validation + install gate
# --------------------------------------------------------------------------


def validate_layer_contract(gdn: Any, *, label: str) -> None:
    """Raise unless this GDN module is the geometry the fold is wired for."""

    observed = (
        int(getattr(gdn, "num_v_heads", -1)),
        int(getattr(gdn, "num_k_heads", -1)),
        int(getattr(gdn, "head_v_dim", -1)),
        int(getattr(gdn, "head_k_dim", -1)),
    )
    expected = (NUM_V_HEADS, NUM_K_HEADS, HEAD_DIM, HEAD_DIM)
    if observed != expected:
        raise GdnKeepMaskFoldContractError(
            f"{label}: keep-mask fold is wired for {expected} "
            f"(v_heads, k_heads, head_v, head_k); got {observed}"
        )
    if bool(getattr(gdn, "training", False)):
        raise GdnKeepMaskFoldContractError(
            f"{label}: keep-mask fold requires the inference recurrence "
            "(use_kernel=True)"
        )


def validate_state_contract(state: Any, *, label: str) -> None:
    """Raise unless the recurrent state is the f32 shape the fold assumes.

    float32 is the whole exactness argument: the kernel keeps the state in
    fp32 registers and stores it as ``StT``, so splitting the T loop is the
    identity only while ``StT`` is fp32.  A bf16 state would round at the
    split point and the fold would NOT be bit-exact.
    """

    shape = tuple(int(d) for d in getattr(state, "shape", ()))
    if shape != (1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM):
        raise GdnKeepMaskFoldContractError(
            f"{label}: recurrent state must be "
            f"{(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM)}; got {shape}"
        )
    dtype_name = str(getattr(getattr(state, "dtype", None), "__name__", "")) or str(
        getattr(state, "dtype", "")
    )
    if "float32" not in dtype_name:
        raise GdnKeepMaskFoldContractError(
            f"{label}: recurrent state must be float32 for the fold to be "
            f"bit-exact; got {dtype_name!r}"
        )


def install_gdn_keepmask_fold(
    *,
    gdn_layer_indices: Sequence[int],
    ple_layer_index: int,
    layer_modules: Sequence[Any],
    exactness_probe: Any = None,
) -> dict[str, Any]:
    """Validate the lane once and arm it.  Contract failures RAISE.

    Modelled on ``mtplx/kernels/qwen4_m4_route.py``: anything structural (the
    wrong layer count, the wrong head geometry, a non-f32 state) raises, so an
    armed-but-inert flag can never masquerade as a neutral A/B result.  The
    ``exactness_probe`` -- the only part that needs a GPU -- is the single
    failure that DISABLES instead: it compares a split recurrence against the
    merged one on synthetic rows, and a mismatch means this MLX build's kernel
    does not round-trip its state exactly, which is a portability fact about
    the machine rather than a configuration error.
    """

    STATS["armed"] = True
    STATS["max_windows"] = fable_gdn_keepmask_fold_windows()
    STATS["prefix_rows"] = prefix_rows()
    indices = tuple(int(i) for i in gdn_layer_indices)
    if len(indices) != GDN_LAYERS:
        raise GdnKeepMaskFoldContractError(
            f"keep-mask fold requires {GDN_LAYERS} GDN layers; got {len(indices)}"
        )
    if int(ple_layer_index) not in set(indices):
        raise GdnKeepMaskFoldContractError("the PLE layer must be a GDN layer")
    foldable = tuple(i for i in indices if i != int(ple_layer_index))
    if len(foldable) != FOLDABLE_LAYERS:
        raise GdnKeepMaskFoldContractError(
            f"keep-mask fold expects {FOLDABLE_LAYERS} foldable GDN layers; "
            f"got {len(foldable)}"
        )
    for index in foldable:
        layer = layer_modules[index]
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            raise GdnKeepMaskFoldContractError(
                f"GDN layer {index} has no linear_attn module"
            )
        validate_layer_contract(gdn, label=f"{ENV_FLAG} layer {index}")

    if exactness_probe is not None:
        try:
            ok, detail = exactness_probe()
        except Exception as exc:  # pragma: no cover - probe owns its own errors
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            STATS["installed"] = False
            STATS["install_status"] = "exactness_failed"
            STATS["install_error"] = str(detail)
            STATS["folded_layers"] = 0
            _log_engagement(
                f"DISABLED (split/merged recurrence mismatch: {detail})"
            )
            return stats_snapshot()

    STATS["installed"] = True
    STATS["install_status"] = "installed"
    STATS["install_error"] = None
    STATS["folded_layers"] = len(foldable)
    _log_engagement(
        f"armed: {len(foldable)} foldable GDN layers, ring "
        f"{STATS['max_windows']} window(s) = {STATS['prefix_rows']} prefix rows, "
        f"step T={STATS['prefix_rows'] + VERIFY_WIDTH}"
    )
    return stats_snapshot()


def note_window(ring_depth: int, *, folded: bool) -> None:
    """Count one compiled M4 window and the ring it entered with."""

    STATS["windows"] += 1
    key = str(int(ring_depth))
    STATS["ring_depth_hist"][key] = STATS["ring_depth_hist"].get(key, 0) + 1
    if folded:
        STATS["folded_windows"] += 1


def note_deferred_commit(*, layers: int, flushed: bool) -> None:
    """Count one deferred commit and the state passes it did not dispatch."""

    STATS["deferred_commits"] += 1
    if flushed:
        STATS["flushes"] += 1
    else:
        STATS["state_passes_saved"] += int(layers)
        STATS["state_bytes_saved"] += int(layers) * 2 * STATE_BYTES


__all__ = [
    "DEFAULT_MAX_WINDOWS",
    "ENV_FLAG",
    "ENV_LOG",
    "ENV_WINDOWS",
    "FOLDABLE_LAYERS",
    "FoldPending",
    "GDN_LAYERS",
    "GdnKeepMaskFoldContractError",
    "MAX_WINDOWS_CHOICES",
    "RingDecision",
    "STATE_BYTES",
    "STATS",
    "VERIFY_WIDTH",
    "clear_pending",
    "expected_state_passes_per_cycle",
    "fable_gdn_keepmask_fold_enabled",
    "fable_gdn_keepmask_fold_windows",
    "install_gdn_keepmask_fold",
    "note_deferred_commit",
    "note_window",
    "pending_for",
    "prefix_mask_rows",
    "prefix_rows",
    "reset_fable_gdn_keepmask_fold_cache",
    "reset_stats",
    "stats_snapshot",
    "validate_layer_contract",
    "validate_state_contract",
]
