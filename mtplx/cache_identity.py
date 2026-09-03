"""Cache-entry identity across a compiled run's container re-wrap.

WHY THIS EXISTS (W66d, 2026-09-02)
----------------------------------
``Qwen4ExpTextModel._decode_layers_compiled`` runs each contiguous run of
non-PLE GDN layers through ``_compiled_run_fn``, whose traced body builds a
THROWAWAY ``ArraysCache(size=2)`` per layer and hands THAT to the layer as
``cache``.  ``MTPLX_COMPILED_GDN=1`` is a family default -- set by
``abba_driver.build_family_overrides`` on every ABBA arm and by the server's
own runtime overrides -- so every ``S <= 4`` decode, the compiled physical-M4
verify included, goes through that re-wrap.

Any lane that keys per-layer state by the IDENTITY of the container it was
handed therefore misses inside a compiled run, and a miss is not always a
decline.  The keep-mask fold's miss was silent and wrong: the dispatch had
already substituted the deferred ring's BASE into state slot 1, so a layer
that failed to find its prefix ran the stock recurrence from a state missing
one or two committed windows.  Every host-side counter stayed perfectly
self-consistent; three ABBA seeds produced different text, diverging around
token ten, and nothing in the receipts could see it.

The fold was repaired lane-specifically (``fable_gdn_keepmask_fold``'s
``bind_fold_alias`` + ``assert_prefix_consumed``).  This module is the generic
version, so the NEXT lane that reads a cache container by identity cannot be
hidden by the same re-wrap:

1. **An explicit alias.**  ``_compiled_run_fn`` calls
   :func:`bind_rewrapped_entry` on each throwaway container, which stamps it
   with the real cache entry the run is standing in for plus that entry's
   layer index.  :func:`resolve_cache_entry` is the ONE helper every
   identity-keyed lane resolves through; it returns the object unchanged when
   there is no alias, so a lane that resolves correctly today keeps behaving
   exactly as it does.

2. **A declared expectation.**  A dispatch that has already committed to a
   layer consuming something -- the fold substituting a ring base into slot 1
   is the archetype -- registers ``expect(lane, layer_index)``.  The lane
   calls :func:`note_resolved` when it actually finds its state, and
   ``_decode_layers_compiled`` calls :func:`assert_satisfied` once per
   compiled run.  An expectation that was declared for a layer THIS RUN
   ACTUALLY RE-WRAPPED and never resolved raises
   :class:`CacheIdentityContractError` at trace time, naming the lane and the
   layer index.

3. **No ragged metadata across the re-wrap** (W77).  The throwaway carries
   neither ``lengths`` nor ``left_padding``, and neither is an input to the
   traced step, so a layer that branches on them -- ``GatedDeltaNet``'s ragged
   conv-state write, and all three ``_fused_*_applies`` refusals -- reads
   ``None`` inside a compiled run whatever the real entry holds.
   :func:`assert_no_ragged_metadata` refuses that at the compiled entry
   (once per run, on the run's head entry) and at trace time (per layer, on
   every real entry the run re-wraps).  It cannot fire today: ``_forward``
   only takes the compiled path when the ssm entry's ``make_mask`` returns
   ``None``, which it does only when both fields are unset there, and every
   producer sets them uniformly across the cache list.  That is a coincidence,
   not a contract, which is exactly what the guard is for.

Cost.  Both context variables default to ``None``.  With no lane registered,
:func:`assert_satisfied` and :func:`note_resolved` are a single
``ContextVar.get()`` and a return, and :func:`bind_rewrapped_entry` is one
``setattr`` -- and it runs at TRACE time only, because ``mx.compile`` replays
bind the traced graph positionally and never re-enter the Python body.  That
is also why the guard is a trace-time assertion and not a runtime one: a
replay executes no lane code at all, so :func:`assert_satisfied` only ever
inspects layers whose containers were re-wrapped during the call it is
checking.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "ALIAS_ATTR",
    "CacheIdentityContractError",
    "CacheIdentityExpectations",
    "RAGGED_METADATA_FIELDS",
    "RaggedCacheInCompiledRunError",
    "STASH_PREFIX",
    "assert_no_dropped_stash",
    "assert_no_ragged_metadata",
    "assert_satisfied",
    "bind_rewrapped_entry",
    "current_expectations",
    "expect",
    "expectations_scope",
    "note_resolved",
    "note_resolved_index",
    "pop_rewrap_source",
    "push_rewrap_source",
    "ragged_metadata_fields",
    "real_entry_for",
    "resolve_cache_entry",
    "rewrap_scope",
    "rewrapped_layer_index",
]


class CacheIdentityContractError(RuntimeError):
    """A compiled run re-wrapped a layer whose declared lane never resolved."""


class RaggedCacheInCompiledRunError(CacheIdentityContractError):
    """A compiled run was handed a cache entry carrying ragged-batch metadata.

    The throwaway container ``_compiled_run_fn`` builds carries neither
    ``lengths`` nor ``left_padding``, so inside a compiled run every predicate
    that reads them sees ``None`` and the layer takes a DIFFERENT branch from
    the eager path -- see :func:`assert_no_ragged_metadata`.
    """


#: Attribute the throwaway container carries back to the real cache entry.
ALIAS_ATTR = "_mtplx_cache_rewrap"


@dataclass(slots=True, frozen=True)
class _Rewrap:
    """What a throwaway container stands in for."""

    layer_index: int
    real: Any


# --------------------------------------------------------------------------
# The alias: throwaway container -> real cache entry
# --------------------------------------------------------------------------
#
# The source is the caller's own cache list, bound for the duration of one
# forward.  Passing the list rather than a per-run dict keeps the hot path to
# one ContextVar set/reset per forward: `_decode_layers_compiled` runs every
# decode step, while the traced body that reads it runs only on a retrace.

_REWRAP_SOURCE: ContextVar["Sequence[Any] | None"] = ContextVar(
    "mtplx_cache_rewrap_source", default=None
)


def push_rewrap_source(source: "Sequence[Any] | None") -> Token:
    """Bind the real cache list a compiled run's containers stand in for."""

    return _REWRAP_SOURCE.set(source)


def pop_rewrap_source(token: Token) -> None:
    """Undo :func:`push_rewrap_source`."""

    _REWRAP_SOURCE.reset(token)


@contextlib.contextmanager
def rewrap_scope(source: "Sequence[Any] | None") -> Iterator[None]:
    """Context-manager form of :func:`push_rewrap_source`."""

    token = push_rewrap_source(source)
    try:
        yield
    finally:
        pop_rewrap_source(token)


def bind_rewrapped_entry(container: Any, layer_index: int) -> Any:
    """Point ``container`` at the real cache entry for ``layer_index``.

    Called by every forward that hands a layer a container other than the one
    its caller owns -- today that is ``_compiled_run_fn``'s per-layer
    ``ArraysCache``.  Returns the real entry, or ``None`` outside a re-wrap
    scope (in which case any stale alias is REMOVED, so a recycled container
    can never inherit another forward's entry).
    """

    source = _REWRAP_SOURCE.get()
    index = int(layer_index)
    real = None
    if source is not None:
        try:
            real = source[index]
        except (IndexError, KeyError, TypeError):
            real = None
    if real is None:
        try:
            delattr(container, ALIAS_ATTR)
        except AttributeError:
            pass
        return None
    try:
        setattr(container, ALIAS_ATTR, _Rewrap(index, real))
    except AttributeError:
        # Containers with __slots__ and no room for the alias cannot be
        # guarded; they also cannot be resolved, so say so rather than
        # pretending the binding happened.
        return None
    expectations = _EXPECTATIONS.get()
    if expectations is not None:
        expectations.rewrapped.add(index)
    return real


def real_entry_for(obj: Any) -> Any:
    """The real cache entry ``obj`` stands in for, or ``None``."""

    alias = getattr(obj, ALIAS_ATTR, None)
    return None if alias is None else alias.real


def resolve_cache_entry(obj: Any) -> Any:
    """``obj``'s real cache entry -- ``obj`` itself when it is not a stand-in.

    THE one helper an identity-keyed lane resolves through.  A no-op for every
    forward that does not re-wrap, which is why adopting it cannot change the
    behaviour of a lane that resolves correctly today.
    """

    alias = getattr(obj, ALIAS_ATTR, None)
    return obj if alias is None else alias.real


def rewrapped_layer_index(obj: Any) -> "int | None":
    """The layer index ``obj`` was bound to, or ``None`` if it is not a stand-in."""

    alias = getattr(obj, ALIAS_ATTR, None)
    return None if alias is None else alias.layer_index


# --------------------------------------------------------------------------
# Expectations: what a dispatch has already committed the layer to consuming
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CacheIdentityExpectations:
    """One traced forward's ``(lane, layer index)`` obligations.

    ``pending`` is what a dispatch declared and no lane has resolved yet;
    ``rewrapped`` is the set of layer indices a compiled run actually handed a
    stand-in container this call.  The assertion checks the intersection, so a
    replay -- which runs no Python body and therefore re-wraps nothing --
    cannot raise, and an eager forward's layers are left to the lane's own
    end-of-forward check.
    """

    pending: set[tuple[str, int]] = field(default_factory=set)
    satisfied: set[tuple[str, int]] = field(default_factory=set)
    rewrapped: set[int] = field(default_factory=set)
    #: ``id(real cache entry) -> layer index``, so a lane that resolves in a
    #: forward that did NOT re-wrap can still be attributed to its layer.
    index_by_entry: dict[int, int] = field(default_factory=dict)

    def expect(self, lane: str, layer_index: int, entry: Any = None) -> None:
        index = int(layer_index)
        self.pending.add((str(lane), index))
        if entry is not None:
            self.index_by_entry[id(entry)] = index

    def satisfy(self, lane: str, layer_index: int) -> None:
        key = (str(lane), int(layer_index))
        self.pending.discard(key)
        self.satisfied.add(key)

    def unmet(self, indices: "Iterable[int] | None" = None) -> tuple[tuple[str, int], ...]:
        covered = (
            self.rewrapped
            if indices is None
            else self.rewrapped.intersection(int(index) for index in indices)
        )
        if not covered or not self.pending:
            return ()
        return tuple(
            sorted(key for key in self.pending if key[1] in covered)
        )

    def __len__(self) -> int:  # pragma: no cover - convenience only
        return len(self.pending)


_EXPECTATIONS: ContextVar["CacheIdentityExpectations | None"] = ContextVar(
    "mtplx_cache_identity_expectations", default=None
)


@contextlib.contextmanager
def expectations_scope(
    expectations: "CacheIdentityExpectations | None" = None,
) -> Iterator[CacheIdentityExpectations]:
    """Open one traced forward's expectation registry."""

    if expectations is None:
        expectations = CacheIdentityExpectations()
    token = _EXPECTATIONS.set(expectations)
    try:
        yield expectations
    finally:
        _EXPECTATIONS.reset(token)


def current_expectations() -> "CacheIdentityExpectations | None":
    """The registry for this traced forward, or ``None``."""

    return _EXPECTATIONS.get()


def expect(lane: str, layer_index: int, entry: Any = None) -> None:
    """Declare that ``lane`` MUST resolve ``layer_index`` in this forward.

    Pass ``entry`` -- the real cache entry -- so a forward that does not
    re-wrap can still attribute :func:`note_resolved` to its layer.  A no-op
    outside an expectations scope.
    """

    expectations = _EXPECTATIONS.get()
    if expectations is None:
        return
    expectations.expect(lane, layer_index, entry)


def note_resolved_index(lane: str, layer_index: int) -> None:
    """Record that ``lane`` resolved its state for ``layer_index``.

    For lanes that already know the layer index -- the keep-mask fold looks
    one up from its own scope.  A no-op outside an expectations scope.
    """

    expectations = _EXPECTATIONS.get()
    if expectations is None:
        return
    expectations.satisfy(lane, layer_index)


def note_resolved(lane: str, obj: Any) -> None:
    """Record that ``lane`` resolved its state for the layer behind ``obj``.

    ``obj`` may be the stand-in container (the layer index comes from the
    alias) or the real entry (it comes from the ``expect`` registration).  A
    no-op outside an expectations scope, and when neither route attributes a
    layer -- an unattributable resolve is left to the lane's own check rather
    than credited to the wrong layer.
    """

    expectations = _EXPECTATIONS.get()
    if expectations is None:
        return
    index = rewrapped_layer_index(obj)
    if index is None:
        index = expectations.index_by_entry.get(id(obj))
    if index is None:
        return
    expectations.satisfy(lane, index)


#: Namespace every mtplx lane uses to stash per-layer state on a cache entry
#: (``_mtplx_verify_rows``, ``_mtplx_verify_ple``, ``_mtplx_fold_pending``...).
STASH_PREFIX = "_mtplx_"


def assert_no_dropped_stash(
    container: Any,
    layer_index: int,
    *,
    forwarded: "Iterable[str]",
    label: str,
) -> None:
    """Raise if a layer stashed state on a stand-in nobody reads back.

    The identity lookup is one half of the re-wrap hazard; this is the other.
    ``GatedDeltaNet.__call__`` hangs the capture-commit rows off the container
    it is handed, and ``_compiled_run_fn`` only survives that because it
    explicitly re-reads ``_mtplx_verify_rows`` off the throwaway and surfaces
    it as a compiled output.  Any OTHER ``_mtplx_*`` stash written inside the
    run dies with the throwaway -- silently, and the reader downstream sees
    the value from a previous window instead.  Checked at trace time, once per
    layer per retrace, so the run either forwards a stash or refuses it.
    """

    state = getattr(container, "__dict__", None)
    if not state:
        return
    known = set(forwarded)
    known.add(ALIAS_ATTR)
    dropped = sorted(
        name
        for name, value in state.items()
        if name.startswith(STASH_PREFIX) and name not in known and value is not None
    )
    if not dropped:
        return
    raise CacheIdentityContractError(
        f"{label}: layer {int(layer_index)} stashed {', '.join(dropped)} on the "
        "throwaway cache container this run hands it, and the run does not "
        "read it back -- the value dies here and the reader downstream sees "
        "the previous window's. Surface it as a compiled output (as "
        "_mtplx_verify_rows is), or write it to the real entry via "
        "mtplx.cache_identity.resolve_cache_entry."
    )


#: The ragged-batch metadata a cache entry can carry.  Both are set by the
#: batch lanes (mlx-lm's ``ArraysCache.merge`` seeds ``left_padding`` on every
#: all-empty merge; ``PromptProcessingBatch.prompt`` sets ``lengths`` around a
#: right-padded prefill; ``restore_cache(..., restore_meta_state=True)``
#: reinstalls whatever a snapshot captured), and both are read by
#: ``GatedDeltaNet`` OUTSIDE the mask: the ragged conv-state write branches on
#: ``cache.lengths``, and all three ``_fused_*_applies`` predicates refuse on it.
RAGGED_METADATA_FIELDS = ("lengths", "left_padding")


def ragged_metadata_fields(entry: Any) -> tuple[str, ...]:
    """The ragged-batch metadata fields ``entry`` carries, in field order.

    ``()`` for the plain decode cache, which is every entry outside a padded
    or ragged batch.  Cold path: the hot callers inline the two ``getattr``
    tests and only call in here to build the message.
    """

    if entry is None:
        return ()
    return tuple(
        name
        for name in RAGGED_METADATA_FIELDS
        if getattr(entry, name, None) is not None
    )


def assert_no_ragged_metadata(entry: Any, layer_index: int, *, label: str) -> None:
    """Raise if a compiled run would drop ``entry``'s ragged-batch metadata.

    THE THIRD HALF of the re-wrap hazard (after the identity lookup and the
    dropped stash).  ``_compiled_run_fn``'s throwaway ``ArraysCache(size=2)``
    carries no ``lengths`` and no ``left_padding``, and neither field is an
    input to the traced step, so inside a compiled run:

    * ``GatedDeltaNet.__call__``'s conv-state write takes the DENSE tail
      (``conv_input[:, -n_keep:, :]``) instead of the ragged
      ``take_along_axis`` gather keyed on ``cache.lengths``; and
    * ``_fused_step_applies`` / ``_fused_conv_norm_applies`` /
      ``_fused_conv_norm_rows_applies`` stop refusing, so a padded batch can
      reach a kernel written for dense rows.

    Neither is a decline.  Both are silently different arithmetic from what
    the eager path would have run on the same cache -- W66d's failure mode.

    Today this cannot happen, and it is blocked by ONE coincidence rather than
    by a contract: ``Qwen4ExpTextModel._forward`` enters the compiled path only
    when ``create_ssm_mask(h, cache[self.ssm_idx])`` is ``None``, and
    ``make_mask`` returns an array whenever EITHER field is set on that one
    entry.  Every producer sets the fields uniformly across the cache list, so
    the ssm entry stands in for all of them.  The coincidence dies the moment
    a lane sets the metadata non-uniformly, hands the model a container that
    carries the fields without a ``make_mask``, or re-wraps the cache in a twin
    that drops the metadata (``graphbank._ensure_shadow`` already builds its
    GDN twins as ``type(entry)(len(entry.cache))``, which copies the state
    leaves and nothing else).  So the check is here, at the compiled entry, to
    fail loudly on the day it becomes reachable instead of diverging.
    """

    # The two fields are spelled out rather than looped over
    # RAGGED_METADATA_FIELDS because this is the hot path: one call, two
    # attribute loads and two identity tests per compiled run per decode
    # step.  Both are plain instance attributes on stock ArraysCache and
    # `_lengths is None` / `_left_padding is None` fast-path properties on the
    # vendored FixedArraysCache, so neither load folds or schedules anything
    # unless it is actually set -- and if it is set, this call raises.
    if (
        getattr(entry, "lengths", None) is None
        and getattr(entry, "left_padding", None) is None
    ):
        return
    fields = ragged_metadata_fields(entry)
    raise RaggedCacheInCompiledRunError(
        f"{label}: layer {int(layer_index)}'s cache entry carries "
        f"{' and '.join(fields)}, and the throwaway container a compiled run "
        "hands the layer carries neither -- the run would take the dense "
        "conv-state write and stop refusing the fused GDN kernels, which is "
        "not a decline but different arithmetic from the eager path on the "
        "same cache. Route ragged/left-padded forwards to the eager path "
        "(the ssm mask normally does this), or thread the metadata through "
        "the compiled step as an explicit input."
    )


def assert_satisfied(
    indices: "Iterable[int] | None" = None, *, label: str
) -> None:
    """Raise unless every lane that expected a re-wrapped layer resolved it.

    ``indices`` scopes the check to the layers a compiled run just executed;
    ``None`` checks every layer this forward re-wrapped.  A no-op when no lane
    declared an expectation.
    """

    expectations = _EXPECTATIONS.get()
    if expectations is None:
        return
    unmet = expectations.unmet(indices)
    if not unmet:
        return
    detail = ", ".join(f"{lane} @ layer {index}" for lane, index in unmet)
    raise CacheIdentityContractError(
        f"{label}: the compiled run re-wrapped the cache container for "
        f"{len(unmet)} declared lane/layer pair(s) that never resolved it "
        f"({detail}). The layer ran with state the dispatch had already "
        "substituted, which does not decline -- it is silently wrong. The "
        "lane must resolve through mtplx.cache_identity.resolve_cache_entry."
    )
