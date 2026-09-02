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

STATUS -- WHAT IS WIRED (W66b, 2026-09-02)
------------------------------------------
Arm A of ``scripts/fable/micro_gdn_keepmask_fold.py`` answered the falsifier
on the production shape (35 layers, ring 2, 200 reps, guarded window):
``T = 12`` costs **+0.083 ms/cycle** over ``T = 4`` -- ``gated_delta_step`` is
STATE-BOUND and the fold's extra rows are ~free.  Arm B priced the two forms,
both bit-exact in every accept pattern: the KERNEL form
(``mtplx_gated_delta_step_prefix``) runs at 0.83-0.86x of today on a
single-window commit and 0.64-0.67x on a two-window ring, while the pure-MLX
concatenate form is 1.4x SLOWER on single windows.  So the kernel form is what
is wired.

Wired, all behind ``MTPLX_FABLE_GDN_KEEPMASK_FOLD=1`` (default off):

1. ``CompiledVerifyBank._resolve_gdn_keepmask_fold`` (mtplx/graphbank.py) --
   arms the lane once per fixed-M4 installation, BEFORE the verify graph is
   traced.  Structural mismatches (layer count, head geometry, a non-f32
   recurrent state, more than one PLE layer) RAISE; only the
   split-vs-merged exactness probe disables-and-logs.
2. ``CompiledVerifyBank._fold_window_prefix`` + ``_forward_installed_fixed_m4``
   -- every window pushes ``base`` into slot 1 in place of the deferred
   commit's lazy leaf and appends 5 padded row tensors per foldable layer plus
   one SHARED ``[1, 4*W]`` bool mask (176 leaves at W=2) to the trailing state
   args.  One arity, one set of shapes, on every window whatever the ring
   holds -- a depth-0 ring passes cached all-zero pads under an all-False
   mask, which is an exact no-op and zero dispatches.
3. ``CompiledVerifyBank._make_verify_step`` -- ``spec`` consumes ``state_in``
   positionally, so the prefix is ``state_in[pos:]``; it is bound into a
   contextvar keyed by shadow-entry identity for the duration of the traced
   forward.  ``_SHARED_VERIFY_STEPS``'s global key carries the fold dimension,
   so a folded trace can never be served to a bank that is not passing a
   prefix.
4. ``GatedDeltaNet.__call__`` (mtplx/models/qwen4_exp.py) -- captures the new
   rows first (unchanged), then runs ``prefix_gated_delta_update`` when a
   prefix is in scope.  ``y`` is already only the window's rows, so
   ``cache.advance(S)`` is untouched.
5. ``Qwen4ExpTextModel.commit_verified_window`` -- a partial accept whose
   verify was THIS window's compiled graph binds ``cache[1]`` to the lazy
   masked replay and hangs a ``FoldPending`` off the entry instead of
   replaying eagerly.  The conv-state commit, the PLE layer's exact-width
   replay and the QSA trims are byte-identical.

WHY THE COMMIT CANNOT RE-DERIVE THE RING FROM THE SNAPSHOT
----------------------------------------------------------
The family lane snapshots LAZILY: ``snapshot_untrimmable_cache_lazy`` retains
``leaf[...]``, a fresh view object, so ``pre[1] is pending.state`` is False
even when the two hold the same value.  The compiled window therefore STAMPS
the descriptor it consumed onto the entry (``set_active``) and the commit
honours it only for its own window (``active_for``).  Everything else --
a context-copy block round's ``forward_ar``, an eager AR round, a rollback
re-forward, a refused commit -- leaves no stamp and takes the shipped replay,
which is why those paths need no flush protocol of their own.

FAIL-SAFE PROPERTY.  Every non-fold consumer -- a context-copy block round, a
rollback re-forward, ``detach_cache_state``,
``MTPLX_EVAL_STATE_ROOTS_ON_COMMIT``, the session bank -- reads
``entry.cache[1]``, which is a real array holding the correct state; forcing
it costs exactly today's replay.  ``pending_for`` drops the descriptor the
moment anything else rebinds the leaf, so a rollback or a trim silently
degrades the fold to today's behaviour instead of corrupting it.

WHAT THE FOLD COSTS THAT TODAY DOES NOT
---------------------------------------
* The deferred base stays alive.  Today's ``state_in`` leaf is sole-referenced
  by the time the graph runs and MLX may donate its buffer; under the fold the
  descriptor holds a reference, so it is not donated.  Bound at two states per
  foldable layer, i.e. ~220 MB at W=2 -- 0.2% of the 100 GiB wired limit.
* One ``mx.concatenate`` per row tensor per commit at W >= 2 (175 lazy copies
  of ~163 kB a layer, riding the next window's pre-boundary ``async_eval``).
  At W = 1 there is none: ``padded_prefix_leaves`` returns the captured rows
  themselves.
* 35 cached all-zero pad tuples, one per foldable layer (~5.7 MB total,
  allocated once).  They are per layer rather than shared so a depth-0
  window hands the graph 175 DISTINCT arrays -- one array in 35 input
  positions would make the traced graph's input identity depend on the ring
  depth of whichever window happened to trace it.
* Every window runs its 36 step kernels at ``T = 4*W + 4`` rather than 4,
  including the ~29.5% that enter with an empty ring.  That is the +0.083
  ms/cycle arm A priced, and it buys the removal of 0.499 state passes/cycle.

THE ABBA WINDOW AND ITS RECEIPT GATE
------------------------------------
Run the 16K decode bracket from the merged main worktree
(``$ROOT`` = /Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-fable-80tps).
``abba_window.py`` must be the guard's direct child, so it goes through
``bench/laguna/run_guarded.py``; running it bare prints the exact outer line
and refuses::

    cd $ROOT && PYTHONPATH=$ROOT $ROOT/.venv/bin/python \
      /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
        --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
        --lock-timeout-seconds 1800 --child-timeout-seconds 36000 \
        -- $ROOT/.venv/bin/python $ROOT/scripts/fable/abba_window.py \
             --sequence <N> \
             --label-prefix w66b-gdn-keepmask-fold \
             --prompt-tokens 16384 \
             --order ABBA \
             --python $ROOT/.venv/bin/python \
             --control-flag=--prewarm-ngram-table \
             --candidate-extra-env MTPLX_FABLE_GDN_KEEPMASK_FOLD=1

Verified with ``--dry-run``: 12 arms (3 seeds x ABBA), every arm carrying the
retained stack (``--target-mode batched --require-compiled-verify --m4-stage3
--qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare --max-tokens 1024``
plus ``MTPLX_QWEN4_M4_ROUTED_{DOWN_REDUCE,DOWN_RESIDUAL_TAIL,GLU}=1``) and
``--prewarm-ngram-table``, and only the six B arms carrying
``--env MTPLX_FABLE_GDN_KEEPMASK_FOLD=1``.

A ring sweep is ``--candidate-extra-env
MTPLX_FABLE_GDN_KEEPMASK_FOLD_WINDOWS=1|3``; W=1 is the concatenate-free arm
(0.497 flushes/cycle but no padded prefix to build at all), W=3 trades a
deeper ring and ``T = 16`` for 0.112.

``--control-flag=--prewarm-ngram-table`` moves the SHARED baseline (both arms);
the current stack flags are the queue's, added from its own file, and
``CONTROL_FLAGS``/``CONTROL_CANDIDATE_ENV`` in ``abba_window.py`` already carry
the retained lane.  ``MTPLX_FABLE_*`` is the one MTPLX namespace
``--candidate-extra-env`` accepts, which is why the flag rides there rather
than ``--candidate-env``.

Read ``row["gdn_keepmask_fold"]`` on every candidate row BEFORE reading any
timing.  ``receipt_gate`` decides it; the arm counts as having run the lane
only when all of:

* ``installed`` is true and ``install_error`` is null (35 folded layers),
* ``windows == compiled_m4_calls`` -- a window on the shipped route took the
  control's commit and dilutes the delta by exactly its share,
* ``flushes / windows ~= 0.20`` at W = 2 (the ring policy's stationary rate
  under the census accept law; the gate allows 35%).  A rate near 0.70 means
  something is FORCING the deferred leaf every cycle,
* ``declines == 0``,
* ``response_token_sha256`` identical to the control arm's on the same seed.

Expected saving, from the micro: today's commit replay is 1.14-1.64 ms per
commit event at 0.70 events/cycle = 0.8-1.1 ms/cycle; the fold leaves
~0.20 flushes/cycle x ~1.0 ms ~= 0.2, i.e. **-0.6 to -0.9 ms/cycle** against a
~15 ms net M4 window, less the +0.083 ms/cycle of arm A's extra rows.

ENV
---
``MTPLX_FABLE_GDN_KEEPMASK_FOLD=1``            arm the lane (default 0)
``MTPLX_FABLE_GDN_KEEPMASK_FOLD_WINDOWS=N``    ring depth, 1..4 (default 2)
``MTPLX_FABLE_GDN_KEEPMASK_FOLD_LOG=0``        silence the engagement line
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

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


#: Row tensors a folded layer contributes to the compiled graph's trailing
#: args: ``q``, ``k``, ``v``, ``a``, ``b``.  The keep mask is NOT per layer --
#: the ring is one shared object across the 35 foldable layers, so one
#: ``[1, 4*W]`` bool leaf serves all of them and the graph grows by 176 inputs
#: rather than 210.
PREFIX_LEAVES_PER_LAYER = 5


def prefix_leaf_count(layers: int) -> int:
    """Trailing leaves the compiled verify carries for ``layers`` folded layers.

    One number, used by the producer (``_fold_window_prefix``) and by the
    consumer (``_make_verify_step``'s arity check) so a shape drift is a loud
    ValueError at the graph boundary rather than a silently misaligned prefix
    36 layers deep.
    """

    count = int(layers)
    if count < 0:
        raise ValueError(f"layers must be >= 0; got {layers}")
    return PREFIX_LEAVES_PER_LAYER * count + (1 if count else 0)


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
    "declines": 0,            # windows/commits that fell back to today's path
    "decline_reasons": {},    # reason -> count
    "bypassed_commits": 0,    # commits from a NON-M4 round (copy/AR/re-forward)
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
            "declines": 0,
            "decline_reasons": {},
            "bypassed_commits": 0,
        }
    )
    reset_window_seq()


def stats_snapshot() -> dict[str, Any]:
    """A copy of the counters, safe to embed in a receipt."""

    snapshot = dict(STATS)
    snapshot["ring_depth_hist"] = dict(STATS["ring_depth_hist"])
    snapshot["decline_reasons"] = dict(STATS["decline_reasons"])
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


# --------------------------------------------------------------------------
# Window sequence -- the stamp that ties a commit to ITS OWN verify
# --------------------------------------------------------------------------
#
# ``commit_verified_window`` cannot re-derive the ring from the snapshot: the
# family lane snapshots LAZILY (``snapshot_untrimmable_cache_lazy`` retains
# ``leaf[...]``, a fresh view object), so ``pre[1] is pending.state`` is False
# even when the two hold the same value.  Instead the compiled window stamps
# the descriptor it consumed onto the entry, and the commit honours it only
# when the stamp is THIS window's.  A copy round, a rollback re-forward, an
# eager AR forward or a refused commit all leave a stale stamp that the next
# window overwrites and that no commit can mistake for its own.

_WINDOW_SEQ = 0
ACTIVE_ATTR = "_mtplx_fold_active"


def next_window_seq() -> int:
    """Open a new compiled-window epoch and return its stamp."""

    global _WINDOW_SEQ
    _WINDOW_SEQ += 1
    return _WINDOW_SEQ


def current_window_seq() -> int:
    """The stamp of the most recently opened window."""

    return _WINDOW_SEQ


def reset_window_seq() -> None:
    """Test support: rewind the epoch counter."""

    global _WINDOW_SEQ
    _WINDOW_SEQ = 0


def set_active(entry: Any, pending: "FoldPending", seq: int) -> None:
    """Record the descriptor this window's graph consumed for ``entry``."""

    entry._mtplx_fold_active = (int(seq), pending)


def active_for(entry: Any, seq: int) -> "FoldPending | None":
    """The descriptor stamped for window ``seq``, or ``None``.

    ``None`` is always safe: the caller replays exactly as it does today.
    """

    stamped = getattr(entry, ACTIVE_ATTR, None)
    if stamped is None:
        return None
    try:
        stamp, pending = stamped
    except Exception:
        return None
    return pending if int(stamp) == int(seq) else None


def clear_active(entry: Any) -> None:
    """Drop the window stamp so one window's commit cannot be applied twice."""

    if getattr(entry, ACTIVE_ATTR, None) is not None:
        entry._mtplx_fold_active = None


# --------------------------------------------------------------------------
# Prefix scope -- how the compiled body's step kernel finds its prefix
# --------------------------------------------------------------------------
#
# The compiled verify's Python body runs at TRACE time only; replays bind the
# same graph positionally.  The scope therefore exists purely so
# ``GatedDeltaNet.__call__`` can wire the trailing prefix tracers into the
# right layer's step while the trace is being built.  It is keyed by the
# SHADOW entry's identity, so a layer with no prefix (the PLE-carrying GDN
# layer, or any layer under a declined window) simply misses the map and takes
# the stock ``gated_delta_update`` it takes today.

_PREFIX_SCOPE: ContextVar[dict[int, Any] | None] = ContextVar(
    "mtplx_gdn_keepmask_fold_prefix", default=None
)


@contextlib.contextmanager
def fold_prefix_scope(mapping: dict[int, Any] | None) -> Iterator[None]:
    """Bind ``id(cache entry) -> prefix leaves`` for one traced forward."""

    token = _PREFIX_SCOPE.set(mapping)
    try:
        yield
    finally:
        _PREFIX_SCOPE.reset(token)


def fold_prefix_for(entry: Any) -> Any:
    """This layer's ``(q, k, v, a, b, mask)`` prefix, or ``None``."""

    scope = _PREFIX_SCOPE.get()
    if not scope:
        return None
    return scope.get(id(entry))


# --------------------------------------------------------------------------
# Ring advance -- one committed window against the descriptor its verify used
# --------------------------------------------------------------------------


def advance_ring(
    pending: "FoldPending",
    window_rows: Any,
    accepted_keep: int,
    *,
    max_windows: int,
) -> tuple[Any, list[Any], tuple[int, ...], bool]:
    """``(base, rows, keeps, flushed)`` for the post-commit descriptor.

    On a flush the OLD pending leaf becomes the new base: MLX evaluates one
    masked replay covering the whole old ring (one state pass for several
    windows) the next time the base is read, which is the next window's
    ``state_in``.  That bounds the lazy chain at two levels -- the base handed
    to a graph is always evaluated by that graph's own pre-boundary
    ``async_eval``, so a flush can never stack a third.
    """

    decision = ring_after_commit(
        pending.keeps, accepted_keep, max_windows=max_windows
    )
    if decision.flush:
        if pending.state is None:
            raise GdnKeepMaskFoldContractError(
                "a ring flush needs a materialisable pending state"
            )
        return pending.state, [window_rows], decision.keeps, True
    return pending.base, [*pending.rows, window_rows], decision.keeps, False


def note_decline(reason: str) -> None:
    """Count one window or commit that fell back to today's exact path.

    A decline is never a correctness event -- it is today's answer at today's
    cost -- but a candidate arm with declines did not measure the lane, so the
    receipt gate fails on any non-zero count.
    """

    STATS["declines"] += 1
    key = str(reason)
    STATS["decline_reasons"][key] = STATS["decline_reasons"].get(key, 0) + 1


# --------------------------------------------------------------------------
# Receipt gate -- is a candidate ABBA arm's engagement the lane we priced?
# --------------------------------------------------------------------------

#: 112 all-accept of 374 classified cycles, W58 retained-control census.
CENSUS_P_ALL_ACCEPT = 112 / 374


def receipt_gate(
    snapshot: dict[str, Any],
    *,
    compiled_windows: int,
    p_all_accept: float = CENSUS_P_ALL_ACCEPT,
    tolerance: float = 0.35,
) -> dict[str, Any]:
    """Decide from the receipt alone whether an arm ran the priced lane.

    ``compiled_windows`` is the arm's ``compiled_verify.compiled_calls`` --
    the number of fixed-M4 windows the bank actually replayed.  The gate is
    deliberately about ENGAGEMENT, not speed: it answers "did this arm run the
    fold on every window, with the flush rate the ring policy predicts, and
    without ever falling back", so a null result can be read as a null result
    rather than as an inert flag.
    """

    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    installed = bool(snapshot.get("installed"))
    _check("installed", installed, snapshot.get("install_status"))
    _check("no_install_error", snapshot.get("install_error") is None,
           snapshot.get("install_error"))
    _check(
        "folded_layers",
        int(snapshot.get("folded_layers") or 0) == FOLDABLE_LAYERS,
        snapshot.get("folded_layers"),
    )
    declines = int(snapshot.get("declines") or 0)
    _check("fold_declined_zero", declines == 0, snapshot.get("decline_reasons"))

    windows = int(snapshot.get("windows") or 0)
    _check(
        "windows_cover_compiled_calls",
        windows == int(compiled_windows),
        {"fold_windows": windows, "compiled_calls": int(compiled_windows)},
    )

    expected_flushes = expected_state_passes_per_cycle(
        p_all_accept, max_windows=int(snapshot.get("max_windows") or 0) or 2
    )
    flushes = int(snapshot.get("flushes") or 0)
    observed = (flushes / windows) if windows else float("nan")
    within = (
        windows > 0
        and abs(observed - expected_flushes) <= tolerance * expected_flushes
    )
    _check(
        "flushes_per_cycle",
        within,
        {
            "observed": observed,
            "expected": expected_flushes,
            "tolerance_frac": tolerance,
        },
    )

    deferred = int(snapshot.get("deferred_commits") or 0)
    expected_commits = (1.0 - float(p_all_accept)) * windows if windows else 0.0
    commits_ok = windows > 0 and abs(deferred - expected_commits) <= (
        tolerance * max(1.0, expected_commits)
    )
    _check(
        "deferred_commits_track_partial_accepts",
        commits_ok,
        {"deferred_commits": deferred, "expected": expected_commits},
    )

    hist = dict(snapshot.get("ring_depth_hist") or {})
    max_windows = int(snapshot.get("max_windows") or 0)
    depth_ok = bool(hist) and all(
        0 <= int(key) <= max_windows for key in hist
    )
    _check("ring_depth_within_max", depth_ok, hist)

    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "expected_flushes_per_cycle": expected_flushes,
        "observed_flushes_per_cycle": observed,
    }


__all__ = [
    "ACTIVE_ATTR",
    "CENSUS_P_ALL_ACCEPT",
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
    "active_for",
    "advance_ring",
    "clear_active",
    "clear_pending",
    "current_window_seq",
    "expected_state_passes_per_cycle",
    "fable_gdn_keepmask_fold_enabled",
    "fable_gdn_keepmask_fold_windows",
    "fold_prefix_for",
    "fold_prefix_scope",
    "install_gdn_keepmask_fold",
    "next_window_seq",
    "note_decline",
    "note_deferred_commit",
    "note_window",
    "pending_for",
    "prefix_mask_rows",
    "PREFIX_LEAVES_PER_LAYER",
    "prefix_leaf_count",
    "prefix_rows",
    "receipt_gate",
    "reset_fable_gdn_keepmask_fold_cache",
    "reset_stats",
    "reset_window_seq",
    "set_active",
    "stats_snapshot",
    "validate_layer_contract",
    "validate_state_contract",
]
