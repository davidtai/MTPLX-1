"""Compiled D1->D3 MTP draft-chain replay (``MTPLX_FABLE_COMPILED_DRAFT``).

Why
---
The PR391 float32-D3 route builds its three-depth draft graph node by node, in
plain Python, on every decode cycle (``mtplx.generation._pr391_make_float32_d3_core``
-> ``chain_fn``).  Each depth issues a full MTP ``DecoderLayer`` forward (QSA
attention + indexer + the 512-expert MoE + two gated-residual mixers), a ranked
draft-head gather, and the deterministic K20 support preparation -- on the order
of a hundred host-issued MLX ops per depth, ~300 per cycle.  None of that host
encode work is model compute; it is pure cycle-time lag.

``mx.compile`` removes it: the graph is traced once, at request construction,
and every later cycle replays the cached trace with a single host call per
depth.

Arming it
---------
``MTPLX_FABLE_COMPILED_DRAFT=1``, default off.  It is *not* a member of
``profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS`` and must not be put in the ABBA
driver's ``family_overrides`` (``apply_profile_env`` refuses the whole arm on an
unregistered key).  ``MTPLX_FABLE_*`` rides the driver's raw passthrough
instead::

    scripts/fable/abba_driver.py ... --env MTPLX_FABLE_COMPILED_DRAFT=1

which lands in ``os.environ`` before ``apply_profile_env`` runs and is never
popped by it, so the gate below observes it at the first core construction.

Design -- option A, one compiled per-depth body
----------------------------------------------
Exactly **one** depth body is compiled and invoked ``depth`` times, with the
softfloat64 candidate selector left eager between the calls.

The rejected alternative was compiling the whole three-depth chain *including*
the three ``mx.fast.metal_kernel`` selector dispatches.  Two reasons not to:

* the selector is already one host op per depth -- three ops out of ~300, so
  folding it in buys almost nothing; and
* ``mx.fast.metal_kernel`` composition with ``mx.compile`` cannot be exercised
  anywhere except on a Metal device, so a whole-chain trace would ship
  unverified.  The per-depth body is verifiable structurally (see
  ``tests/test_fable_compiled_draft.py``) and its only remaining unknown is
  numerical, not structural.

One body (rather than one per depth) is admissible only because ``mtp_depth`` is
inert for this model: ``mtp_adapter_depth`` moves ``active_depth`` on MTP LoRA
modules, and ``MTPLXRuntime.draft_mtp`` forwards ``mtp_depth`` into
``model.mtp_forward`` only when that signature accepts it.  Both are checked at
construction and both raise :class:`CompiledDraftUnsupported` when violated --
the flag never degrades to the eager chain silently.

State
-----
``draft_mtp`` mutates the MTP ``TensorOffsetQSACache`` in place.  The compiled
body therefore captures the same nested state tree the eager route advertises
(``_device_core_state_tree``) as both ``inputs=`` and ``outputs=``, which is the
pattern ``graphbank`` uses for the verifier: MLX reads the leaves out of the live
list containers before the call and writes the updated leaves back after it, so
three sequential calls chain their cache updates exactly as the eager loop does.

``TensorOffsetKVCache.update_and_fetch`` also writes ``rollback_state``, which is
deliberately *not* part of the captured tree (its shapes change per cycle).
Under a compiled body those writes happen only while tracing, so the slots would
retain trace-time tracers that raise on evaluation.  The chain clears them after
every depth, which is also what the eager route's callers do after the chain.

ABI
---
``build_compiled_draft_chain(...)["chain_fn"]`` is a drop-in replacement for the
eager ``chain_fn`` -- identical call signature, identical return structure and
ordering, so ``_pr391_run_float32_d3_core``, ``_pr391_queue_canonical_d3``,
``_pr391_queue_device_canonical_d3`` and ``_pr391_prewarm_float32_d3_core`` need
no change::

    chain_fn(hidden_states, first_token_ids, uniform_bit_rows) -> (
        selected_tokens,      # [1, depth]  uint32
        raw_candidate_ids,    # [depth, K]  uint32
        raw_candidate_values, # [depth, K]  float32
        raw_candidate_probs,  # [depth, K]  float32
    )

    hidden_states     [1, 1, W]  the trunk's widened MTP recursion state
    first_token_ids   [1, 1]     uint32, the primary this cycle drafts from
    uniform_bit_rows  [depth]    the request's PCG64 tape slice, one row per
                                 depth; row ``level`` is handed to the selector
                                 for depth ``level + 1``, exactly as eager.

The compiled inner body has its own, narrower ABI::

    compiled_body(hidden_states, token_ids) -> (
        candidate_ids,    # [1, K] uint32   -- FR-Spec remapped when installed
        candidate_values, # [1, K] float32
        candidate_probs,  # [1, K] float32
        next_hidden,      # [1, 1, W]       -- produced_hidden[:, -1:, :]
    )

plus the captured MTP cache state, which it advances by exactly one row.

What is preserved
-----------------
* the same draft tokens for the same inputs, up to ``mx.compile`` rounding
  (bf16 regrouping is expected and accepted; bit-for-bit digest parity is not a
  goal of this flag);
* the K20 proposal rows handed to the verifier decision -- same construction,
  same order, same dtypes;
* PCG64 uniform consumption -- the tape is host-side and untouched; the chain
  still consumes exactly one tape row per depth and the selector stays eager;
* MTP cache offsets -- advanced by the same in-graph slice updates;
* the carried / lookahead D3 mechanics in ``generation.py``, which only ever see
  ``core["fn"]``.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable

import mlx.core as mx

from .fable_indexer_reuse import ENV_FLAG as INDEXER_REUSE_ENV
from .fable_indexer_reuse import indexer_reuse_enabled
from .fast_sampling import (
    _deterministic_mlx_top_k_support,
    _order_bounded_mlx_top_k_support,
)

FABLE_COMPILED_DRAFT_ENV = "MTPLX_FABLE_COMPILED_DRAFT"

_TRUTHY = {"1", "true", "yes", "on"}
_ENABLED_CACHE: bool | None = None


class CompiledDraftUnsupported(RuntimeError):
    """The compiled draft replay was requested where it is not admissible.

    Raised at construction only.  The flag is opt-in and fails loudly: a
    request that asks for the compiled replay and cannot have it is an error,
    never a silent return to the eager chain (which would quietly invalidate
    any measurement taken under the flag).
    """


class CompiledDraftStateChanged(RuntimeError):
    """The captured MTP cache state changed shape after the trace.

    ``mx.compile`` keys its trace on the shapes of the captured state, so a QSA
    capacity growth mid-request would silently replay a stale graph.  The chain
    checks the captured leaf shapes once per cycle and raises instead.
    """


def compiled_draft_enabled() -> bool:
    """Return the ``MTPLX_FABLE_COMPILED_DRAFT`` gate; read once, default off.

    Resolved lazily rather than at import so a serving profile that arms env
    flags after ``mtplx.generation`` is imported is still observed, then cached
    so the decode cycle never pays a repeated environment lookup.
    """

    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        raw = os.environ.get(FABLE_COMPILED_DRAFT_ENV)
        _ENABLED_CACHE = bool(raw) and raw.strip().lower() in _TRUTHY
    return _ENABLED_CACHE


def reset_compiled_draft_flag_cache() -> None:
    """Drop the memoized gate. Test-support only."""

    global _ENABLED_CACHE
    _ENABLED_CACHE = None


def _is_array_leaf(node: Any) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def state_leaf_slots(tree: Any) -> tuple[tuple[list[Any], int], ...]:
    """Return ``(container, index)`` for every array leaf of a state tree.

    The slots are the exact positions ``mx.compile(outputs=...)`` rebinds after
    a call, so reading shapes back through them proves the compiled trace still
    matches the live cache rather than trusting a duck-typed cache property.
    """

    slots: list[tuple[list[Any], int]] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            for child in node.values():
                visit(child)
            return
        if isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                if _is_array_leaf(child):
                    if isinstance(node, tuple):
                        raise CompiledDraftUnsupported(
                            "MTP compiled draft state holds array leaves in an "
                            "immutable tuple; mx.compile cannot rebind them"
                        )
                    slots.append((node, index))
                else:
                    visit(child)
            return

    visit(tree)
    return tuple(slots)


def state_leaf_shapes(
    slots: tuple[tuple[list[Any], int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Shapes currently held in ``slots`` -- the compiled trace's identity."""

    return tuple(tuple(container[index].shape) for container, index in slots)


def _require_inert_mtp_depth(rt: Any) -> None:
    """Prove one compiled body may serve every depth of the chain.

    The eager chain passes ``mtp_depth=level``.  That argument has two possible
    effects, and both must be absent before a single trace can stand in for all
    three depths.
    """

    model = getattr(rt, "model", None)
    if model is None:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT requires a runtime with a model"
        )

    from .mtp_adapters import iter_mtp_lora_modules

    try:
        modules = iter_mtp_lora_modules(model)
    except RuntimeError:
        # No injected MTP module at all: there is nothing for mtp_depth to
        # select, and the draft chain itself would already have failed.
        modules = []
    if modules:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT cannot serve a depth-selected MTP LoRA "
            f"stack ({len(modules)} adapter module(s) switch weights per depth); "
            "one compiled body would bake a single depth's adapter"
        )

    forward = getattr(model, "mtp_forward", None)
    if forward is None:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT requires model.mtp_forward"
        )
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError) as error:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT could not introspect model.mtp_forward"
        ) from error
    if "mtp_depth" in parameters:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT cannot serve a model.mtp_forward that "
            "consumes mtp_depth; the per-depth body would bake depth 1"
        )


def _require_fixed_capacity_headroom(
    mtp_cache: Any,
    *,
    depth: int,
    request_max_tokens: int,
) -> None:
    """Reject a cache that could still grow while the trace is live.

    ``mx.compile`` re-traces when a captured leaf changes shape.  The QSA MTP
    cache grows through ``ensure_capacity``, so the compiled replay is only safe
    when the construction-time promotion already reserved every row this request
    can append.  Checked here, at construction, rather than discovered as a
    silent retrace (or a stale replay) in the middle of the measured loop.
    """

    if not mtp_cache or len(mtp_cache) != 1:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT requires exactly one promoted MTP cache"
        )
    entry = mtp_cache[0]
    if getattr(entry, "fixed_capacity", False) is not True:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT requires a fixed-capacity MTP QSA cache"
        )
    # Both reads materialize the device-resident offset, so this is the one
    # host sync the flag adds -- at construction, never in the decode cycle.
    capacity = int(entry.capacity)
    offset = int(entry.offset)
    required = offset + int(request_max_tokens) + int(depth)
    if capacity < required:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT requires a request-sized MTP reservation: "
            f"capacity {capacity} < offset {offset} + max_tokens "
            f"{int(request_max_tokens)} + depth {int(depth)}; the cache would "
            "grow mid-request and invalidate the compiled trace"
        )


def build_compiled_draft_chain(
    *,
    rt: Any,
    mtp_cache: Any,
    state_tree: Any,
    mtp_hidden_variant: str,
    selector: Callable[..., tuple[Any, ...]],
    frspec_ids: Any,
    depth: int,
    top_k: int,
    request_max_tokens: int,
) -> dict[str, Any]:
    """Build the compiled D1->D3 replacement for the eager draft chain.

    Returns ``{"chain_fn", "compiled_body", "state_slots", "state_shapes",
    "trace_stats", "depth", "top_k"}``.  ``chain_fn`` matches the eager chain's
    ABI exactly (see the module docstring); the caller installs it as
    ``core["fn"]``.  ``trace_stats["body_traces"]`` is an exact count of how
    many times the graph was traced -- 1 for a healthy prewarmed request.

    Raises :class:`CompiledDraftUnsupported` when any precondition fails.  The
    first call to ``chain_fn`` performs the trace, so callers must prewarm --
    ``_pr391_prewarm_float32_d3_core`` already does, at construction, outside
    the measured loop.
    """

    depth = int(depth)
    top_k = int(top_k)
    if indexer_reuse_enabled():
        # The reuse anchor is host state consulted once per draft call; under
        # one compiled body the depth-1 branch is baked into the trace and
        # depths 2/3 would replay it, so the flag would be silently inert.
        raise CompiledDraftUnsupported(
            f"{FABLE_COMPILED_DRAFT_ENV} and {INDEXER_REUSE_ENV} are mutually "
            "exclusive: the per-depth trace cannot observe the host-side "
            "reuse anchor, so an armed reuse flag would never fire"
        )
    if depth < 1:
        raise CompiledDraftUnsupported("compiled draft depth must be >= 1")
    if top_k < 1:
        raise CompiledDraftUnsupported("compiled draft top_k must be >= 1")
    if not callable(selector):
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT requires a prebound candidate selector"
        )

    _require_inert_mtp_depth(rt)
    _require_fixed_capacity_headroom(
        mtp_cache,
        depth=depth,
        request_max_tokens=request_max_tokens,
    )

    state_slots = state_leaf_slots(state_tree)
    if not state_slots:
        raise CompiledDraftUnsupported(
            "MTPLX_FABLE_COMPILED_DRAFT found no rebindable MTP state leaves"
        )
    traced_shapes = state_leaf_shapes(state_slots)

    entry = mtp_cache[0]
    entry_kv = entry.kv
    count_call = getattr(rt, "_count", None)
    # The python body runs only while mx.compile traces, so this counter is an
    # exact trace count and costs nothing per cycle. A prewarmed request should
    # report 1; anything that keeps climbing means the trace signature moves
    # (a changed hidden dtype, a regrown cache) and the flag is not paying off.
    trace_stats = {"body_traces": 0}

    def depth_body(hidden_states, token_ids):
        """One MTP depth: layer forward, ranked head gather, K20 support prep.

        ``mtp_depth`` is passed as ``None`` rather than the level: the
        construction gate proved the argument is inert for this model, so a
        single trace serves every depth.
        """

        trace_stats["body_traces"] += 1
        logits, produced_hidden = rt.draft_mtp(
            hidden_states,
            token_ids,
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=None,
        )
        row = logits[:, -1, :].reshape(-1)
        flat = row.astype(mx.float32)
        local_ids, q_values = _deterministic_mlx_top_k_support(flat, top_k)
        local_ids, q_values = _order_bounded_mlx_top_k_support(local_ids, q_values)
        q_probs = mx.exp(q_values - mx.logsumexp(flat, axis=-1, keepdims=True))
        if frspec_ids is not None and int(row.shape[0]) == int(frspec_ids.shape[0]):
            real_ids = mx.take(frspec_ids, local_ids)
        else:
            real_ids = local_ids
        return (
            real_ids.astype(mx.uint32).reshape(1, top_k),
            q_values.astype(mx.float32).reshape(1, top_k),
            q_probs.astype(mx.float32).reshape(1, top_k),
            produced_hidden[:, -1:, :],
        )

    compiled_body = mx.compile(depth_body, inputs=state_tree, outputs=state_tree)

    def chain_fn(hidden_states, first_token_ids, uniform_bit_rows):
        live_shapes = state_leaf_shapes(state_slots)
        if live_shapes != traced_shapes:
            raise CompiledDraftStateChanged(
                "MTPLX_FABLE_COMPILED_DRAFT captured MTP state changed shape "
                f"({traced_shapes} -> {live_shapes}); the compiled draft replay "
                "cannot serve a regrown cache"
            )
        next_hidden = hidden_states
        next_token = first_token_ids
        selected_tokens: list[Any] = []
        raw_ids_by_depth: list[Any] = []
        raw_values_by_depth: list[Any] = []
        raw_probs_by_depth: list[Any] = []
        for level in range(depth):
            if count_call is not None:
                count_call("draft_mtp_calls")
            candidate_ids, candidate_values, candidate_probs, produced_hidden = (
                compiled_body(next_hidden, next_token)
            )
            # update_and_fetch stashed trace-time tracers here; they are not
            # captured state and raise if anything ever evaluates them. Read
            # the list through the cache so a reassignment cannot orphan it.
            entry_kv.rollback_state[:] = [None, None, None]
            selected, raw_ids, raw_values, raw_probs = selector(
                candidate_ids,
                candidate_values,
                candidate_probs,
                uniform_bit_rows[level : level + 1],
            )
            selected = selected.reshape(1, 1)
            selected_tokens.append(selected)
            raw_ids_by_depth.append(raw_ids)
            raw_values_by_depth.append(raw_values)
            raw_probs_by_depth.append(raw_probs)
            next_hidden = produced_hidden
            next_token = selected
        return (
            mx.concatenate(selected_tokens, axis=1),
            mx.concatenate(raw_ids_by_depth, axis=0),
            mx.concatenate(raw_values_by_depth, axis=0),
            mx.concatenate(raw_probs_by_depth, axis=0),
        )

    return {
        "chain_fn": chain_fn,
        "compiled_body": compiled_body,
        "state_slots": state_slots,
        "state_shapes": traced_shapes,
        "trace_stats": trace_stats,
        "depth": depth,
        "top_k": top_k,
    }


def maybe_build_compiled_draft_chain(
    *,
    rt: Any,
    mtp_cache: Any,
    state_tree: Any,
    mtp_hidden_variant: str,
    selector: Callable[..., tuple[Any, ...]],
    frspec_ids: Any,
    depth: int,
    top_k: int,
    request_max_tokens: int,
) -> dict[str, Any] | None:
    """Build the compiled chain when the gate is armed, else ``None``.

    ``None`` means the flag is off, which is the default and keeps the eager
    route byte-identical.  An armed flag either yields the compiled chain or
    raises :class:`CompiledDraftUnsupported`.
    """

    if not compiled_draft_enabled():
        return None
    return build_compiled_draft_chain(
        rt=rt,
        mtp_cache=mtp_cache,
        state_tree=state_tree,
        mtp_hidden_variant=mtp_hidden_variant,
        selector=selector,
        frspec_ids=frspec_ids,
        depth=depth,
        top_k=top_k,
        request_max_tokens=request_max_tokens,
    )


__all__ = [
    "FABLE_COMPILED_DRAFT_ENV",
    "CompiledDraftStateChanged",
    "CompiledDraftUnsupported",
    "build_compiled_draft_chain",
    "compiled_draft_enabled",
    "maybe_build_compiled_draft_chain",
    "reset_compiled_draft_flag_cache",
    "state_leaf_shapes",
    "state_leaf_slots",
]
