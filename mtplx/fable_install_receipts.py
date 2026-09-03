"""W84 -- install-time verdicts for the armed flags that had none.

The program's rule is that every armed flag prints an install-time verdict to
stderr and exposes it in the server's engagement reports, so a benchmark can
prove the stack it measured.  Two lanes already do it right and are the model
for this module:

``mtplx/kernels/qsa_sparse_decode.py``
    ``engagement_line(enabled=...)`` renders ``[fable] qsa_sparse_decode
    armed: ...`` / ``[fable] qsa_sparse_decode: off (<reason>)`` and
    ``receipt()`` returns the compact dict the abba driver stores.
``mtplx/fable_verify_glue.py``
    ``install()`` emits one line per selected item through ``_emit`` (which
    goes to BOTH ``logger.info`` and stderr, because a driver run configures
    no logging handler and a lane that cannot prove it ran is unreadable) and
    ``receipt()`` carries the per-item hot-path call counters.

A pre-battery sanity stage over the served set found nine keys with no
install-time receipt anywhere in the tree -- bare ``os.environ.get`` reads at
request-time or construction-time sites:

===================================== ====================================
key                                   lane
===================================== ====================================
``MTPLX_FABLE_OPDIET`` (+``_ITEMS``)  ``opdiet``
``MTPLX_FABLE_BLOCK_VERIFY``          ``block_verify``
``MTPLX_FABLE_DRAFT_K20_PRESCATTER``  ``draft_k20_prescatter``
``MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD`` ``ple_prefill_lookahead``
``MTPLX_FABLE_PLE_FIRST_GATHER_EARLY````ple_first_gather_early``
``MTPLX_FABLE_PREFILL_QSA_QUERY_TILE````prefill_qsa_query_tile``
``MTPLX_PREFILL_CHUNK_SIZE``          ``prefill_chunk_size``
``MTPLX_QSA_PREFILL_COMPILE_ROWS``    ``qsa_prefill_compile_rows``
``MTPLX_SESSION_BANK_MAX_BYTES``      ``session_bank_max_bytes``
===================================== ====================================

WHAT A VERDICT IS, AND IS NOT
-----------------------------
A verdict OBSERVES.  Every function here reads state the owning module has
already decided -- the module-level constant it froze at import, the cached
``enabled()``, the counters it already keeps -- and never re-decides anything.
Arming, disarming, parsing and raising all stay where they are.  The one
deliberate exception is the ``refused`` state, which is a PRINTED verdict, not
an exception: an arm that can be shown from the code to do nothing in this
process says so at install instead of leaving the operator to discover it as a
missing delta.  Nothing here raises into a model load.

Four states, in the shape the other Fable lanes use::

    [fable] <lane> armed: <detail>
    [fable] <lane> armed, engages at <condition>: <detail>
    [fable] <lane> off (<reason>)
    [fable] <lane> refused (<reason>)
    [fable] <lane> resolved: <detail>

``armed, engages at ...`` is for a flag whose fate is settled at install but
whose lane only runs on a request that fits (a long prompt, a claimable draft
route).  Those lanes carry an ENGAGEMENT COUNTER, so "armed" and "ran" stay
separate claims; a per-request decline that is by design -- a short prompt, a
greedy window -- is counted, never printed.  ``resolved`` is for the three
non-Fable server knobs, which are not arms at all: they always have a value,
and the receipt's job is to say which one this process is running.

WHERE IT PRINTS
---------------
``emit_all`` is called once, at the end of ``mtplx/runtime.py:load()`` -- the
same place ``[qwen4-fixed-M4-verify]`` and friends publish their install
reports, and the one point every consumer (``mtplx serve``,
``scripts/fable/abba_driver.py``, ``mtplx/prefill_bench.py``) passes through.
Each lane prints at most once per process; ``emit`` is idempotent.

W61 COMPATIBILITY
-----------------
``worker/w61-restack-profile`` (tip 92571504) adds
``mtplx/full_stack_selfcheck.py`` with ``print_install_receipt(tag, report)``
-- the ``[frspec] install report`` spelling generalised -- and a ``/health``
``engagement_reports`` payload.  That branch is not merged, so this module
carries a minimal compatible :func:`print_install_receipt` that DELEGATES to
W61's the moment it exists.  Nothing here imports that module at module scope,
and nothing here depends on a file it adds.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TextIO

__all__ = [
    "LANES",
    "LANE_KEYS",
    "REGISTERED_KEYS",
    "STATE_ARMED",
    "STATE_OFF",
    "STATE_REFUSED",
    "STATE_RESOLVED",
    "Verdict",
    "counters",
    "emit",
    "emit_all",
    "emitted_lines",
    "lane_for_key",
    "note_decline",
    "note_engagement",
    "print_install_receipt",
    "receipts",
    "record",
    "recorded",
    "reset_for_tests",
    "verdict",
    "verdict_for_key",
]

#: Prefix every verdict line carries -- the same one the sparse-decode and
#: verify-glue lanes use, so one ``grep '\[fable\]'`` over a benchmark log
#: yields the whole armed stack.
LINE_PREFIX = "[fable]"

STATE_ARMED = "armed"
STATE_OFF = "off"
STATE_REFUSED = "refused"
STATE_RESOLVED = "resolved"

_TRUE = frozenset({"1", "true", "yes", "on"})


def print_install_receipt(
    tag: str, report: Any, *, stream: TextIO | None = None
) -> None:
    """Print one install-time engagement receipt, the way frspec always did.

    Delegates to ``mtplx.full_stack_selfcheck.print_install_receipt`` when that
    module exists (branch ``worker/w61-restack-profile``) so the two spellings
    never diverge, and otherwise runs W61's body verbatim.  Never raises: a
    receipt must not be able to fail a model load.
    """

    try:
        from .full_stack_selfcheck import (  # type: ignore[attr-defined]
            print_install_receipt as _w61_print,
        )
    except Exception:
        pass
    else:
        _w61_print(tag, report, stream=stream)
        return
    try:
        print(f"[{tag}] {report}", file=stream or sys.stderr, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Verdict:
    """One flag's install-time fate, as a line and as a dict."""

    lane: str
    keys: tuple[str, ...]
    state: str
    detail: str = ""
    reason: str | None = None
    engages_at: str | None = None
    decided_at: str = ""
    readers: tuple[str, ...] = ()
    fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def line(self) -> str:
        if self.state in (STATE_OFF, STATE_REFUSED):
            reason = self.reason or "no reason recorded"
            return f"{LINE_PREFIX} {self.lane} {self.state} ({reason})"
        head = self.state
        if self.state == STATE_ARMED and self.engages_at:
            head = f"{STATE_ARMED}, engages at {self.engages_at}"
        detail = self.detail or "no detail recorded"
        return f"{LINE_PREFIX} {self.lane} {head}: {detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "keys": list(self.keys),
            "state": self.state,
            "reason": self.reason,
            "engages_at": self.engages_at,
            "detail": self.detail,
            "decided_at": self.decided_at,
            "readers": list(self.readers),
            "env": {key: os.environ.get(key) for key in self.keys},
            "fields": dict(self.fields),
            "line": self.line,
        }


@dataclass
class _Entry:
    lane: str
    keys: tuple[str, ...]
    build: Callable[[Mapping[str, Any]], Verdict]
    #: Lanes that ALREADY count their own engagements read them through this
    #: instead of calling :func:`note_engagement`, so the receipt carries one
    #: uniform number and no lane pays a second counter on a hot path.
    engagements: Callable[[], int] | None = None


_REGISTRY: dict[str, _Entry] = {}
_KEY_TO_LANE: dict[str, str] = {}
_EMITTED: dict[str, str] = {}
_ENGAGEMENTS: dict[str, int] = {}
_DECLINES: dict[str, dict[str, int]] = {}
_RECORDED: dict[str, dict[str, Any]] = {}


def _register(
    lane: str,
    keys: Sequence[str],
    build: Callable[[Mapping[str, Any]], Verdict],
    *,
    engagements: Callable[[], int] | None = None,
) -> None:
    _REGISTRY[lane] = _Entry(lane, tuple(keys), build, engagements)
    for key in keys:
        _KEY_TO_LANE[key] = lane


def lane_for_key(key: str) -> str:
    """The lane that owns ``key``.  ``KeyError`` when nothing registered it."""

    return _KEY_TO_LANE[key]


# ---------------------------------------------------------------------------
# Engagement counters and recorded facts
# ---------------------------------------------------------------------------
def note_engagement(lane: str, amount: int = 1) -> None:
    """Count one request on which the lane actually ran.

    Called from the lane's single engagement point.  Never printed: an
    install verdict is a statement about the process, and per-request activity
    belongs in the receipt's counters.
    """

    _ENGAGEMENTS[lane] = _ENGAGEMENTS.get(lane, 0) + int(amount)


def note_decline(lane: str, reason: str, amount: int = 1) -> None:
    """Count one BY-DESIGN decline (a short prompt, a greedy window).

    A decline is not a failure and never prints; it is the other half of the
    engagement counter, and without it a zero-engagement receipt cannot be
    told from a lane that was never asked.
    """

    bucket = _DECLINES.setdefault(lane, {})
    bucket[str(reason)] = bucket.get(str(reason), 0) + int(amount)


def counters(lane: str) -> dict[str, Any]:
    """``{'engagements': n, 'declines': {...}}`` for one lane."""

    entry = _REGISTRY.get(lane)
    if entry is not None and entry.engagements is not None:
        try:
            count = int(entry.engagements())
        except Exception:
            count = int(_ENGAGEMENTS.get(lane, 0))
    else:
        count = int(_ENGAGEMENTS.get(lane, 0))
    return {"engagements": count, "declines": dict(_DECLINES.get(lane, {}))}


def record(lane: str, **fields: Any) -> None:
    """Record a fact the lane only learns later (a resolved budget).

    Pure bookkeeping: the value lands in the lane's receipt dict.  Used by
    ``mtplx/engine_session.py``, whose bank budget resolves after the model
    load that prints the verdict.
    """

    _RECORDED.setdefault(lane, {}).update(fields)


def recorded(lane: str) -> dict[str, Any]:
    return dict(_RECORDED.get(lane, {}))


# ---------------------------------------------------------------------------
# Building, emitting and reporting
# ---------------------------------------------------------------------------
def verdict(lane: str, context: Mapping[str, Any] | None = None) -> Verdict:
    """This process's verdict for ``lane``.  Never raises."""

    entry = _REGISTRY[lane]
    try:
        return entry.build(dict(context or {}))
    except Exception as exc:  # a receipt must not be able to fail a load
        return Verdict(
            lane=lane,
            keys=entry.keys,
            state=STATE_OFF,
            reason=f"verdict unavailable: {type(exc).__name__}: {exc}",
        )


def verdict_for_key(key: str, context: Mapping[str, Any] | None = None) -> Verdict:
    return verdict(lane_for_key(key), context)


def emit(
    lane: str,
    *,
    context: Mapping[str, Any] | None = None,
    stream: TextIO | None = None,
) -> str | None:
    """Print ``lane``'s verdict once per process; return the line, or None.

    Idempotent: a second call for a lane already emitted prints nothing and
    returns None, so "exactly once" holds however many install paths run.
    """

    if lane in _EMITTED:
        return None
    line = verdict(lane, context).line
    _EMITTED[lane] = line
    try:
        print(line, file=stream or sys.stderr, flush=True)
    except Exception:
        pass
    return line


def emit_all(
    *,
    context: Mapping[str, Any] | None = None,
    stream: TextIO | None = None,
) -> list[str]:
    """Emit every not-yet-emitted lane, in registration order."""

    lines = []
    for lane in LANES:
        line = emit(lane, context=context, stream=stream)
        if line is not None:
            lines.append(line)
    return lines


def emitted_lines() -> dict[str, str]:
    """``lane -> line`` for everything printed so far this process."""

    return dict(_EMITTED)


def receipts(context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The block the abba receipt and ``/health`` carry.

    Same shape as the other lanes' ``receipt()``: a plain dict of plain
    values, safe to ``json.dumps``, computed from state that already exists.
    """

    out: dict[str, Any] = {}
    for lane in LANES:
        block = verdict(lane, context).to_dict()
        block.update(counters(lane))
        block["recorded"] = recorded(lane)
        block["emitted"] = lane in _EMITTED
        out[lane] = block
    return out


def reset_for_tests() -> None:
    """Clear emission state, counters and recorded facts.  Tests only."""

    _EMITTED.clear()
    _ENGAGEMENTS.clear()
    _DECLINES.clear()
    _RECORDED.clear()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _raw(key: str) -> str | None:
    return os.environ.get(key)


def _env_note(key: str) -> str:
    """``KEY=value``, prefixed with WHO set it once that is knowable.

    The retained stack is armed by default for a Flash-Next serve
    (mtplx/full_stack_env.py), so ``MTPLX_FABLE_OPDIET='1'`` on its own no
    longer says whether the operator asked for it or simply did not turn it
    off -- and an operator who exports ``=0`` needs to read their own
    decision back off the verdict line. ``value_source`` answers that from
    what this process actually armed, and returns ``""`` in any process that
    armed no defaults, so every existing line keeps its spelling.
    """

    raw = _raw(key)
    if raw is None:
        return f"{key} unset"
    try:
        from .full_stack_env import value_source

        source = value_source(key)
    except Exception:  # a receipt must never fail on a diagnostic
        source = ""
    return f"{source}: {key}={raw}" if source else f"{key}={raw!r}"


def _gib(value: int) -> str:
    return f"{int(value) / 1024 ** 3:.1f}G"


def _qwen4_exp_model(context: Mapping[str, Any]) -> bool | None:
    """True/False when the served model's family is knowable, else None.

    Walks the same wrapper chain ``mtplx/generation.py:_resolve_ple_lookahead_hook``
    walks (runtime -> model -> language_model -> model), because the family
    classes live on the INNER text model.
    """

    runtime = context.get("runtime")
    node = getattr(runtime, "model", None)
    if node is None:
        return None
    seen: list[Any] = []
    for _ in range(4):
        if node is None or any(node is other for other in seen):
            break
        seen.append(node)
        if type(node).__module__.endswith("models.qwen4_exp"):
            return True
        node = getattr(node, "language_model", None) or getattr(node, "model", None)
    return False if seen else None


def _ple_stage_present(context: Mapping[str, Any]) -> tuple[bool | None, str]:
    """``(has_stage, detail)`` for the two PLE lanes.

    ``_ple_stage_idx is None`` is the silent-nothing case both lanes share:
    ``Model.ple_prefill_lookahead`` / ``ple_first_gather_early`` return None
    for every request without raising, so an armed flag is inert for the whole
    process.  Unknowable (no model in the context) returns ``None``.
    """

    runtime = context.get("runtime")
    node = getattr(runtime, "model", None)
    if node is None:
        return None, "no model in the install context"
    seen: list[Any] = []
    for _ in range(4):
        if node is None or any(node is other for other in seen):
            break
        seen.append(node)
        if hasattr(node, "_ple_stage_idx"):
            index = getattr(node, "_ple_stage_idx")
            if index is None:
                return False, f"{type(node).__name__} has no PLE stage layer"
            return True, f"PLE stage at layer {int(index)}"
        node = getattr(node, "language_model", None) or getattr(node, "model", None)
    return None, "no PLE-stage-bearing model under the runtime"


# ---------------------------------------------------------------------------
# 1. MTPLX_FABLE_OPDIET (+ MTPLX_FABLE_OPDIET_ITEMS)
# ---------------------------------------------------------------------------
OPDIET_KEYS = ("MTPLX_FABLE_OPDIET", "MTPLX_FABLE_OPDIET_ITEMS")

#: Which family each op-diet item's gated sites belong to.  ``bank``,
#: ``rope`` and ``resid`` are gated only inside ``mtplx/models/qwen4_exp.py``
#: and ``mtplx/kernels/qwen4_m4_rope.py``; ``k20`` is gated in
#: ``mtplx/generation.py`` and ``mtplx/fast_sampling.py``, which every family
#: runs.  An arm whose whole selection is qwen4_exp-only, on a model that is
#: not qwen4_exp, cannot execute one rewritten site -- that is the
#: ``refused`` case, and it is read off the call sites, not guessed.
OPDIET_ITEM_FAMILY = {
    "bank": "qwen4_exp",
    "rope": "qwen4_exp",
    "resid": "qwen4_exp",
    "k20": "any",
}


def _opdiet_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import runtime_options

    decided_at = "import of mtplx.runtime_options"
    readers = (
        "mtplx/runtime_options.py:fable_opdiet_enabled",
        "mtplx/models/qwen4_exp.py (bank, rope, resid)",
        "mtplx/kernels/qwen4_m4_rope.py (rope)",
        "mtplx/generation.py + mtplx/fast_sampling.py (k20)",
    )
    known = tuple(runtime_options.FABLE_OPDIET_ITEMS)
    armed = bool(runtime_options.fable_opdiet_enabled())
    selected = tuple(item for item in known if runtime_options.fable_opdiet_enabled(item)) if armed else ()
    fields = {
        "armed": armed,
        "items": list(selected),
        "known_items": list(known),
        "item_family": dict(OPDIET_ITEM_FAMILY),
    }
    if not armed:
        return Verdict(
            lane="opdiet",
            keys=OPDIET_KEYS,
            state=STATE_OFF,
            reason=f"{_env_note('MTPLX_FABLE_OPDIET')} at {decided_at}",
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    is_qwen4 = _qwen4_exp_model(context)
    fields["served_model_is_qwen4_exp"] = is_qwen4
    family_scoped = all(
        OPDIET_ITEM_FAMILY.get(item) == "qwen4_exp" for item in selected
    )
    if selected and family_scoped and is_qwen4 is False:
        return Verdict(
            lane="opdiet",
            keys=OPDIET_KEYS,
            state=STATE_REFUSED,
            reason=(
                f"items {','.join(selected)} are gated only inside "
                "mtplx/models/qwen4_exp.py and mtplx/kernels/qwen4_m4_rope.py, "
                "and the served model is not qwen4_exp: no rewritten site can "
                "execute in this process"
            ),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    items_raw = _raw("MTPLX_FABLE_OPDIET_ITEMS")
    selection = (
        f"items={','.join(selected)}"
        f" ({'MTPLX_FABLE_OPDIET_ITEMS unset -> all' if items_raw is None else _env_note('MTPLX_FABLE_OPDIET_ITEMS')})"
    )
    return Verdict(
        lane="opdiet",
        keys=OPDIET_KEYS,
        state=STATE_ARMED,
        detail=(
            f"{selection} of {','.join(known)}; "
            f"exact-preserving rewrites, read once at {decided_at}"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 2. MTPLX_FABLE_BLOCK_VERIFY
# ---------------------------------------------------------------------------
def _block_verify_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import fable_block_verify

    decided_at = "import of mtplx.fable_block_verify"
    readers = (
        "mtplx/fable_block_verify.py:is_enabled",
        "mtplx/generation.py:_FABLE_BLOCK_VERIFY (module constant)",
    )
    armed = bool(fable_block_verify.is_enabled())
    fields = {"armed": armed, "cap_mode": fable_block_verify.CAP_MODE}
    if not armed:
        return Verdict(
            lane="block_verify",
            keys=("MTPLX_FABLE_BLOCK_VERIFY",),
            state=STATE_OFF,
            reason=f"{_env_note('MTPLX_FABLE_BLOCK_VERIFY')} at {decided_at}",
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="block_verify",
        keys=("MTPLX_FABLE_BLOCK_VERIFY",),
        state=STATE_ARMED,
        engages_at=(
            "an accept window with temperature>0, no target prefix, and all "
            "draft+target rows already on the host"
        ),
        detail=(
            f"cap_mode={fable_block_verify.CAP_MODE}; host NumPy ladder, "
            f"read once at {decided_at}"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 3. MTPLX_FABLE_DRAFT_K20_PRESCATTER
# ---------------------------------------------------------------------------
def _draft_k20_prescatter_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import fable_draft_k20_prescatter as lane

    decided_at = "import of mtplx.fable_draft_k20_prescatter"
    readers = (
        "mtplx/fable_draft_k20_prescatter.py:is_enabled",
        "mtplx/generation.py:_FABLE_DRAFT_K20_PRESCATTER (module constant)",
        "mtplx/fable_draft_k20_prescatter.py:claim_draft_route (per request)",
    )
    armed = bool(lane.is_enabled())
    fields = {"armed": armed, "frspec_rows": int(lane.FRSPEC_ROWS)}
    if not armed:
        return Verdict(
            lane="draft_k20_prescatter",
            keys=("MTPLX_FABLE_DRAFT_K20_PRESCATTER",),
            state=STATE_OFF,
            reason=(
                f"{_env_note('MTPLX_FABLE_DRAFT_K20_PRESCATTER')} at {decided_at}"
            ),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="draft_k20_prescatter",
        keys=("MTPLX_FABLE_DRAFT_K20_PRESCATTER",),
        state=STATE_ARMED,
        engages_at="each request whose draft route the claim binds",
        detail=(
            f"frspec_rows={int(lane.FRSPEC_ROWS)}; claimed once per generation "
            "construction, a request shape it does not serve declines to the "
            f"stock draft read; read once at {decided_at}"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 4. MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD
# ---------------------------------------------------------------------------
def _ple_prefill_lookahead_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import ple_prefill_lookahead as lane

    key = lane.ENV_FLAG
    decided_at = (
        "mtplx/ple_prefill_lookahead.py:enabled() -- lru_cached, so the FIRST "
        "read in the process fixes it"
    )
    readers = (
        "mtplx/ple_prefill_lookahead.py:enabled",
        "mtplx/models/qwen4_exp.py:Model.ple_prefill_lookahead (construction)",
        "mtplx/generation.py:_ple_prefill_lookahead_scope (per request)",
    )
    armed = bool(lane.enabled())
    has_stage, stage_detail = _ple_stage_present(context)
    fields = {
        "armed": armed,
        "ple_stage": has_stage,
        "ple_stage_detail": stage_detail,
        "lane_counters": {
            name: int(value)
            for name, value in lane.snapshot_counters().items()
            if not name.startswith("early_")
        },
        "last_scope": dict(lane.last_scope_status()),
    }
    if not armed:
        return Verdict(
            lane="ple_prefill_lookahead",
            keys=(key,),
            state=STATE_OFF,
            reason=_env_note(key),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    if has_stage is False:
        return Verdict(
            lane="ple_prefill_lookahead",
            keys=(key,),
            state=STATE_REFUSED,
            reason=(
                f"{stage_detail}: Model.ple_prefill_lookahead returns None for "
                "every request on a model with no PLE stage, so the lane is "
                "inert for this whole process"
            ),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="ple_prefill_lookahead",
        keys=(key,),
        state=STATE_ARMED,
        engages_at=(
            "any prefill cut into 2+ chunks whose spans exceed the sidecar "
            "hot-row threshold -- a cold prompt's chunked prefill and a "
            "session-bank restore's suffix prefill alike"
        ),
        detail=(
            f"{stage_detail}; worker-thread preparation of chunk k+1's n-gram "
            "rows during chunk k's forward.  A prefill with only one chunk "
            "(a short prompt, a small restored suffix, the fused small-suffix "
            "lane) has nothing to look ahead to and declines as "
            "`single_span`; no request path raises"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 5. MTPLX_FABLE_PLE_FIRST_GATHER_EARLY
# ---------------------------------------------------------------------------
def _ple_first_gather_early_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import ple_prefill_lookahead as lane
    from mtplx import ple_row_gather

    key = lane.EARLY_ENV_FLAG
    decided_at = (
        "mtplx/ple_row_gather.py:enabled() -- lru_cached, so the FIRST read "
        "in the process fixes it"
    )
    readers = (
        "mtplx/ple_row_gather.py:enabled (lru_cache)",
        "mtplx/ple_prefill_lookahead.py:early_enabled",
        "mtplx/models/qwen4_exp.py:Model.ple_first_gather_early (construction)",
        "mtplx/generation.py:_ple_first_gather_early_scope (per request)",
    )
    armed = bool(ple_row_gather.enabled())
    has_stage, stage_detail = _ple_stage_present(context)
    fields = {
        "armed": armed,
        "early_enabled": bool(lane.early_enabled()),
        "ple_stage": has_stage,
        "ple_stage_detail": stage_detail,
        "lane_counters": {
            name: int(value)
            for name, value in lane.snapshot_counters().items()
            if name.startswith("early_")
        },
        "last_early": dict(lane.last_early_status()),
    }
    if not armed:
        return Verdict(
            lane="ple_first_gather_early",
            keys=(key,),
            state=STATE_OFF,
            reason=_env_note(key),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    if has_stage is False:
        return Verdict(
            lane="ple_first_gather_early",
            keys=(key,),
            state=STATE_REFUSED,
            reason=(
                f"{stage_detail}: Model.ple_first_gather_early returns None for "
                "every request on a model with no PLE stage, so the lane is "
                "inert for this whole process"
            ),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="ple_first_gather_early",
        keys=(key,),
        state=STATE_ARMED,
        engages_at=(
            "request arrival, when the first prefill span is predictable and "
            "above the sidecar hot-row threshold"
        ),
        detail=(
            f"{stage_detail}; also selects the vectorised sidecar gather "
            f"(resident_fraction >= {ple_row_gather.RESIDENT_FRACTION_THRESHOLD})."
            "  The span is predicted from the prompt alone, before the "
            "session-bank lookup, so a request the bank restores prefills a "
            "SUFFIX whose first chunk is not that span: the payload is "
            "refused on the span/plan comparison (`early_span_mismatch`, "
            "`early_miss_wrong_span`) and the owner pays the ordinary "
            "gather -- counted, never a raise"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 6. MTPLX_FABLE_PREFILL_QSA_QUERY_TILE
# ---------------------------------------------------------------------------
def _prefill_qsa_query_tile_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx.fable_prefill_chunk import (
        QUERY_TILE_ENV,
        configured_full_chunk_widths,
        resolve_query_tile_rows,
    )

    decided_at = (
        "mtplx/fable_prefill_chunk.py:resolve_query_tile_rows() -- an "
        "UNCACHED read on the attention path; this verdict is its value at "
        "install"
    )
    readers = (
        "mtplx/fable_prefill_chunk.py:resolve_query_tile_rows",
        "mtplx/models/qwen4_exp.py:_prefill_qsa_query_tile_rows",
        "mtplx/models/qwen4_exp.py:_qsa_dense_attention (per attention call)",
    )
    tile = int(resolve_query_tile_rows())
    widths = sorted(int(width) for width in configured_full_chunk_widths())
    try:
        from mtplx.models.qwen4_exp import qsa_prefill_engagement

        engaged = int(qsa_prefill_engagement().get("query_tile", 0))
    except Exception:
        engaged = 0
    fields = {
        "tile_rows": tile,
        "configured_full_chunk_widths": widths,
        "query_tile_calls": engaged,
    }
    if tile <= 0:
        return Verdict(
            lane="prefill_qsa_query_tile",
            keys=(QUERY_TILE_ENV,),
            state=STATE_OFF,
            reason=f"{_env_note(QUERY_TILE_ENV)} -> 0 rows, whole-chunk attention",
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    if widths and tile >= max(widths):
        return Verdict(
            lane="prefill_qsa_query_tile",
            keys=(QUERY_TILE_ENV,),
            state=STATE_REFUSED,
            reason=(
                f"tile={tile} is not narrower than the widest configured "
                f"prefill chunk ({max(widths)}): _qsa_dense_attention takes "
                "the untiled path for every chunk (`tile >= S`), so the arm "
                "cannot change a single attention call"
            ),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="prefill_qsa_query_tile",
        keys=(QUERY_TILE_ENV,),
        state=STATE_ARMED,
        engages_at=f"a prefill chunk wider than {tile} query rows",
        detail=(
            f"tile={tile} rows against configured chunk widths {widths}; "
            "attention only -- GDN, MoE and the projections still see the "
            "whole chunk"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


def _query_tile_engagements() -> int:
    """Tiled attention calls, from the lane's existing engagement counter."""

    try:
        from mtplx.models.qwen4_exp import qsa_prefill_engagement
    except Exception:
        return 0
    return int(qsa_prefill_engagement().get("query_tile", 0))


# ---------------------------------------------------------------------------
# 7. MTPLX_PREFILL_CHUNK_SIZE  (server knob: report the resolved value)
# ---------------------------------------------------------------------------
def _prefill_chunk_size_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx.fable_prefill_chunk import (
        CHUNK_SIZE_DENSE_ENV,
        CHUNK_SIZE_ENV,
        CHUNK_SIZE_REPAGE_ENV,
        COMPILE_ROWS_ENV,
        PrefillChunkGeometryError,
        assert_prefill_chunk_coherent,
        configured_full_chunk_widths,
    )

    decided_at = (
        "mtplx/generation.py:_prefill_chunk_size() -- read per request, and "
        "overridable per request by the ContextVar the warm-up ladder uses"
    )
    readers = (
        "mtplx/generation.py:_prefill_chunk_size",
        "mtplx/fable_prefill_chunk.py:configured_full_chunk_widths",
    )
    raw = _raw(CHUNK_SIZE_ENV)
    widths = sorted(int(width) for width in configured_full_chunk_widths())
    coherence: dict[str, Any] = {}
    verdicts: list[str] = []
    for width in widths:
        try:
            state = assert_prefill_chunk_coherent(width)
        except PrefillChunkGeometryError as exc:
            # Report, never raise: the request path already refuses this pair,
            # and a receipt that can fail a model load is not a receipt.  The
            # line carries the one-word verdict; the full refusal text -- which
            # names the fix -- rides the receipt's `fields`.
            coherence[str(width)] = f"incoherent: {exc}"
            verdicts.append(f"{width}=INCOHERENT")
        else:
            coherence[str(width)] = state
            verdicts.append(f"{width}={state}")
    fields = {
        "raw": raw,
        "widths": widths,
        "dense": _raw(CHUNK_SIZE_DENSE_ENV),
        "repage": _raw(CHUNK_SIZE_REPAGE_ENV),
        "coherence_vs_compile_rows": coherence,
    }
    if raw is None:
        source = f"{CHUNK_SIZE_ENV} unset -> shipped default"
    elif raw.strip().lower() == "auto":
        source = (
            f"{CHUNK_SIZE_ENV}='auto' -> per KV layout "
            f"({CHUNK_SIZE_DENSE_ENV}={_raw(CHUNK_SIZE_DENSE_ENV)!r}, "
            f"{CHUNK_SIZE_REPAGE_ENV}={_raw(CHUNK_SIZE_REPAGE_ENV)!r})"
        )
    else:
        source = _env_note(CHUNK_SIZE_ENV)
    return Verdict(
        lane="prefill_chunk_size",
        keys=(CHUNK_SIZE_ENV,),
        state=STATE_RESOLVED,
        detail=(
            f"full serving chunk widths {widths} ({source}); "
            f"coherence vs {COMPILE_ROWS_ENV} {' '.join(verdicts)}"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 8. MTPLX_QSA_PREFILL_COMPILE_ROWS  (server knob: report the resolved value)
# ---------------------------------------------------------------------------
def _qsa_prefill_compile_rows_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx.fable_prefill_chunk import (
        COMPILE_ROWS_ENV,
        DEFAULT_CHUNK_SIZE,
        _env_int,
    )

    decided_at = (
        "mtplx/models/qwen4_exp.py:_qsa_prefill_compile_rows() -- read per "
        "prefill chunk, uncached"
    )
    readers = (
        "mtplx/models/qwen4_exp.py:_qsa_prefill_compile_rows",
        "mtplx/fable_prefill_chunk.py:assert_prefill_chunk_coherent",
    )
    raw = _raw(COMPILE_ROWS_ENV)
    try:
        from mtplx.models.qwen4_exp import _qsa_prefill_compile_rows

        model_rows: int | None = int(_qsa_prefill_compile_rows())
    except Exception:
        model_rows = None
    guard_raw = _env_int(COMPILE_ROWS_ENV, DEFAULT_CHUNK_SIZE)
    guard_rows = (
        DEFAULT_CHUNK_SIZE if guard_raw is None else max(2, int(guard_raw))
    )
    fields = {
        "raw": raw,
        "model_rows": model_rows,
        "guard_rows": guard_rows,
        "default": int(DEFAULT_CHUNK_SIZE),
    }
    source = (
        f"{COMPILE_ROWS_ENV} unset -> shipped default {DEFAULT_CHUNK_SIZE}"
        if raw is None
        else _env_note(COMPILE_ROWS_ENV)
    )
    mismatch = (
        ""
        if model_rows is None or model_rows == guard_rows
        else (
            f"; WARNING readers disagree -- the model gate resolves "
            f"{model_rows} and the coherence guard resolves {guard_rows}"
        )
    )
    return Verdict(
        lane="qsa_prefill_compile_rows",
        keys=(COMPILE_ROWS_ENV,),
        state=STATE_RESOLVED,
        detail=(
            f"{model_rows if model_rows is not None else guard_rows} rows "
            f"({source}); the QSA prefill graph bank captures this width only"
            f"{mismatch}"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 9. MTPLX_SESSION_BANK_MAX_BYTES  (server knob: report the resolved value)
# ---------------------------------------------------------------------------
SESSION_BANK_LANE = "session_bank_max_bytes"
SESSION_BANK_KEY = "MTPLX_SESSION_BANK_MAX_BYTES"


def _session_bank_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import engine_session

    decided_at = (
        "mtplx/engine_session.py:resolve_session_bank_max_bytes(), called from "
        "EngineSessionManager.__init__ -- AFTER the model load that prints "
        "this line, so the resolved budget arrives as a recorded fact"
    )
    readers = (
        "mtplx/engine_session.py:resolve_session_bank_max_bytes",
        "mtplx/engine_session.py:_explicit_max_bytes_env_set",
        "mtplx/server/openai.py:_effective_ram_session_cache_settings (display)",
    )
    raw = _raw(SESSION_BANK_KEY)
    explicit = bool(engine_session._explicit_max_bytes_env_set())
    fields: dict[str, Any] = {
        "raw": raw,
        "explicit": explicit,
        "auto_floor_bytes": int(engine_session._AUTO_BUDGET_FLOOR_BYTES),
        "auto_cap_bytes": int(engine_session._AUTO_BUDGET_CAP_BYTES),
        "flat_default_bytes": int(engine_session.DEFAULT_MAX_BYTES),
    }
    if explicit:
        try:
            parsed = int(
                engine_session._bank_bytes_from_env(
                    SESSION_BANK_KEY, engine_session.DEFAULT_MAX_BYTES
                )
            )
        except Exception:
            parsed = int(engine_session.DEFAULT_MAX_BYTES)
        fields["resolved_bytes"] = parsed
        detail = f"{_gib(parsed)} explicit ({_env_note(SESSION_BANK_KEY)})"
    else:
        detail = (
            f"auto ({_env_note(SESSION_BANK_KEY)}; model-aware sizing at "
            f"session-bank construction, floor "
            f"{_gib(engine_session._AUTO_BUDGET_FLOOR_BYTES)} cap "
            f"{_gib(engine_session._AUTO_BUDGET_CAP_BYTES)})"
        )
    return Verdict(
        lane=SESSION_BANK_LANE,
        keys=(SESSION_BANK_KEY,),
        state=STATE_RESOLVED,
        detail=detail,
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 10. MTPLX_FABLE_ROUTE_KERNEL  (W93: defaults-on needs a verdict per key)
# ---------------------------------------------------------------------------
def _route_kernel_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import qwen4_m4_stage3 as lane

    key = lane.FABLE_ROUTE_KERNEL_ENV
    decided_at = (
        "mtplx/qwen4_m4_stage3.py:fable_route_kernel_enabled() -- cached in a "
        "lazy global, so the FIRST read in the process fixes it"
    )
    readers = (
        "mtplx/qwen4_m4_stage3.py:fable_route_kernel_enabled",
        "mtplx/qwen4_m4_stage3.py (stage-3 MoE combine tail)",
    )
    armed = bool(lane.fable_route_kernel_enabled())
    fields: dict[str, Any] = {"armed": armed}
    try:
        fields["vec_lanes"] = int(lane.fable_route_kernel_vec_lanes())
    except Exception as exc:  # a bad sweep value: report, never raise
        fields["vec_lanes"] = f"unresolvable: {exc}"
    if not armed:
        return Verdict(
            lane="route_kernel",
            keys=(key,),
            state=STATE_OFF,
            reason=f"{_env_note(key)} at {decided_at}",
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="route_kernel",
        keys=(key,),
        state=STATE_ARMED,
        engages_at="every stage-3 M4 MoE combine tail",
        detail=(
            f"route GEMV + top-k in two dispatches instead of ten, "
            f"vec_lanes={fields['vec_lanes']}; needs MTPLX_QWEN4_M4_STAGE3, "
            "which the runtime already requires (it raises on the "
            "child-routes-without-stage3 combination)"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 11. MTPLX_FABLE_GRAPH_BUILD_OVERLAP (+ _LAYERS)
# ---------------------------------------------------------------------------
GRAPH_BUILD_OVERLAP_KEYS = (
    "MTPLX_FABLE_GRAPH_BUILD_OVERLAP",
    "MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS",
)


def _graph_build_overlap_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx import graph_build_overlap as lane

    decided_at = (
        "mtplx/graph_build_overlap.py:enabled() -- lru_cached, so the FIRST "
        "read in the process fixes it"
    )
    readers = (
        "mtplx/graph_build_overlap.py:enabled",
        "mtplx/graph_build_overlap.py:layers",
        "mtplx/qwen4_fixed_verify.py (fixed-M4 overlap split, at install)",
    )
    try:
        armed = bool(lane.enabled())
    except ValueError as exc:
        # The reader raises on an unparseable spelling. Say so here rather
        # than letting the first request carry the traceback.
        return Verdict(
            lane="graph_build_overlap",
            keys=GRAPH_BUILD_OVERLAP_KEYS,
            state=STATE_REFUSED,
            reason=str(exc),
            decided_at=decided_at,
            readers=readers,
            fields={"armed": None},
        )
    fields: dict[str, Any] = {"armed": armed, "default_layers": int(lane.DEFAULT_LAYERS)}
    if not armed:
        return Verdict(
            lane="graph_build_overlap",
            keys=GRAPH_BUILD_OVERLAP_KEYS,
            state=STATE_OFF,
            reason=f"{_env_note(lane.ENV_FLAG)} at {decided_at}",
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    try:
        layers = int(lane.layers())
    except ValueError as exc:
        return Verdict(
            lane="graph_build_overlap",
            keys=GRAPH_BUILD_OVERLAP_KEYS,
            state=STATE_REFUSED,
            reason=str(exc),
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    fields["layers"] = layers
    fields["items"] = sorted(lane.items())
    return Verdict(
        lane="graph_build_overlap",
        keys=GRAPH_BUILD_OVERLAP_KEYS,
        state=STATE_ARMED,
        engages_at=(
            "every compiled fixed-M4 verify whose plan can be partitioned at "
            "the requested prefix depth"
        ),
        detail=(
            f"{layers}-layer compiled prefix ({_env_note(lane.LAYERS_ENV)}, "
            f"default {lane.DEFAULT_LAYERS}), items={fields['items'] or 'none'}; "
            "host graph build runs behind the prefix's GPU time"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# 12. MTPLX_GDN_BLOCKED_PREFILL
# ---------------------------------------------------------------------------
GDN_BLOCKED_PREFILL_KEY = "MTPLX_GDN_BLOCKED_PREFILL"


def _gdn_blocked_prefill_verdict(context: Mapping[str, Any]) -> Verdict:
    from mtplx.kernels import gdn_blocked_prefill as lane

    decided_at = (
        "mtplx/kernels/gdn_blocked_prefill.py:blocked_prefill_env_enabled() "
        "-- an UNCACHED read on the prefill path; this verdict is its value "
        "at install"
    )
    readers = (
        "mtplx/kernels/gdn_blocked_prefill.py:blocked_prefill_env_enabled",
        "mtplx/models/qwen4_exp.py (GDN prefill route selection)",
    )
    armed = bool(lane.blocked_prefill_env_enabled())
    fields: dict[str, Any] = {"armed": armed}
    try:
        fields["min_route_t"] = int(lane._min_route_t())
    except Exception as exc:
        fields["min_route_t"] = f"unresolvable: {exc}"
    if not armed:
        return Verdict(
            lane="gdn_blocked_prefill",
            keys=(GDN_BLOCKED_PREFILL_KEY,),
            state=STATE_OFF,
            reason=f"{_env_note(GDN_BLOCKED_PREFILL_KEY)}",
            decided_at=decided_at,
            readers=readers,
            fields=fields,
        )
    return Verdict(
        lane="gdn_blocked_prefill",
        keys=(GDN_BLOCKED_PREFILL_KEY,),
        state=STATE_ARMED,
        engages_at=(
            f"any GDN prefill chunk of at least {fields['min_route_t']} tokens"
        ),
        detail=(
            "blocked (chunked-scan) GDN prefill route in place of the "
            f"per-token recurrence; min_route_t={fields['min_route_t']}"
        ),
        decided_at=decided_at,
        readers=readers,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Registration.  Order is the print order.
# ---------------------------------------------------------------------------
_register("opdiet", OPDIET_KEYS, _opdiet_verdict)
_register("block_verify", ("MTPLX_FABLE_BLOCK_VERIFY",), _block_verify_verdict)
_register(
    "draft_k20_prescatter",
    ("MTPLX_FABLE_DRAFT_K20_PRESCATTER",),
    _draft_k20_prescatter_verdict,
)
_register(
    "ple_prefill_lookahead",
    ("MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD",),
    _ple_prefill_lookahead_verdict,
)
_register(
    "ple_first_gather_early",
    ("MTPLX_FABLE_PLE_FIRST_GATHER_EARLY",),
    _ple_first_gather_early_verdict,
)
_register(
    "prefill_qsa_query_tile",
    ("MTPLX_FABLE_PREFILL_QSA_QUERY_TILE",),
    _prefill_qsa_query_tile_verdict,
    # The tiled attention path already bumps `_qsa_prefill_count("query_tile")`
    # on every tiled call; reading it costs nothing and adds no second counter
    # to a per-layer, per-chunk path.
    engagements=_query_tile_engagements,
)
_register(
    "prefill_chunk_size", ("MTPLX_PREFILL_CHUNK_SIZE",), _prefill_chunk_size_verdict
)
_register(
    "qsa_prefill_compile_rows",
    ("MTPLX_QSA_PREFILL_COMPILE_ROWS",),
    _qsa_prefill_compile_rows_verdict,
)
_register(SESSION_BANK_LANE, (SESSION_BANK_KEY,), _session_bank_verdict)
# W93: the retained stack is armed by DEFAULT for a Flash-Next serve
# (mtplx/full_stack_env.py), and the program's rule is that every armed flag
# prints an install-time verdict.  These three lanes had none.
_register("route_kernel", ("MTPLX_FABLE_ROUTE_KERNEL",), _route_kernel_verdict)
_register(
    "graph_build_overlap",
    GRAPH_BUILD_OVERLAP_KEYS,
    _graph_build_overlap_verdict,
)
_register(
    "gdn_blocked_prefill",
    (GDN_BLOCKED_PREFILL_KEY,),
    _gdn_blocked_prefill_verdict,
)

#: Lanes in print order.
LANES: tuple[str, ...] = tuple(_REGISTRY)

#: ``lane -> keys``.
LANE_KEYS: dict[str, tuple[str, ...]] = {
    lane: entry.keys for lane, entry in _REGISTRY.items()
}

#: Every env key that has a registered install verdict.  The W84 sanity stage
#: reads THIS, so a flag added without a verdict fails CI rather than shipping
#: as an unprovable arm.
REGISTERED_KEYS: frozenset[str] = frozenset(_KEY_TO_LANE)
