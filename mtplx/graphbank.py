"""Speculative decode graph-bank scaffolding for MLX.

The first useful job of this module is to make graph-capture eligibility
explicit.  The current Qwen3.6 MLX cache keeps full-attention positions as
Python integers, so a safe compiled decode graph cannot replay across decode
steps until those offsets become tensor inputs/outputs.
"""

from __future__ import annotations

import os
import time
import weakref
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import mlx.core as mx

from .attention_context import attention_phase
from . import fable_gdn_keepmask_fold as _gdn_fold
from .fable_expert_census import census as _expert_census
from . import graph_build_overlap as _graph_build_overlap
from .graph_build_overlap import TIMING as _GRAPH_BUILD_OVERLAP_TIMING
from .ple_boundary import GRAPH_TIMING as _PLE_BOUNDARY_GRAPH_TIMING
from .ple_boundary import note_graph_build as _note_ple_boundary_graph_build
from .gdn_capture import resolve_gdn_capture_backend
from .runtime_options import (
    FABLE_QSA_M4_ROWS,
    fable_qsa_m4_enabled,
    fable_qsa_m4_kt_enabled,
    fable_qsa_sparse_decode_enabled,
    fable_qsa_sparse_draft_enabled,
)


def _prepare_fixed_m4_materialized(
    prepare_aux,
    cache,
    input_ids,
    _host_input_ids,
    _completion_tokens,
    _committed_count,
):
    """Adapt the materialized PLE route to the fixed-M4 host-input contract."""

    return prepare_aux(input_ids, cache)


def _fixed_m4_materialized_prefetch(
    _primary,
    _completion_tokens,
    _committed_count,
) -> None:
    """Construction-bound no-op for materialized fixed-M4 auxiliaries."""


def _fixed_m4_materialized_window_prefetch(
    _host_input_ids,
    _completion_tokens,
    _committed_count,
) -> None:
    """Construction-bound no-op for inline fixed-M4 window preparation."""


def _fixed_m4_no_candidate_prefetch(
    *,
    prefix_tokens,
    candidate_ids,
    completion_tokens,
    committed_count,
) -> int:
    """Construction-bound no-op when the K-P1 candidate lane is unarmed."""

    del prefix_tokens, candidate_ids, completion_tokens, committed_count
    return 0


def _format_compiled_verify_key(key) -> str:
    """Render the generic or fixed-M4 compiled graph key for receipts."""

    if len(key) == 3:
        length, variant, bucket = key
        return f"m{length}:{variant or 'default'}:b{bucket}"
    if len(key) == 4:
        length, variant, bucket, aux_contract = key
        return f"m{length}:{variant or 'default'}:b{bucket}:{aux_contract}"
    raise ValueError(f"unsupported compiled verify key: {key!r}")


def _unpack_fixed_m4_outputs(outputs, *, capture_leaves: int, returns_aux: bool):
    """Split the construction-selected fixed-M4 compiled output contract."""

    aux_offset = int(returns_aux)
    capture_start = 2 + aux_offset
    capture_end = capture_start + capture_leaves
    returned_aux = outputs[2] if returns_aux else None
    return (
        outputs[0],
        outputs[1],
        returned_aux,
        outputs[capture_start:capture_end],
        outputs[capture_end:],
    )


@dataclass(frozen=True)
class FixedM4Prefix:
    """Rooted construction-bound layer-0 result awaiting the suffix join."""

    input_ids: Any
    hidden: Any
    captures: tuple[Any, ...]
    state_in: tuple[Any, ...]
    state_out: tuple[Any, ...]
    outputs: tuple[Any, ...]


@dataclass(frozen=True)
class FixedM4Split:
    """Rooted split verifier outputs awaiting authoritative frontier commit."""

    prefix: FixedM4Prefix
    returned_aux: Any
    captures: tuple[Any, ...]
    state_in: tuple[Any, ...]
    state_out: tuple[Any, ...]
    outputs: tuple[Any, ...]


@dataclass(frozen=True)
class FixedM4OverlapPrefix:
    """W63: one layer-0 result queued ahead of this window's D3 host sync.

    Deliberately does NOT retain ``state_in``.  ``FixedM4Prefix`` (the PR391
    split lane's own transaction object) keeps the layer-0 input leaves alive
    for the whole transaction, which is the shape that defeats MLX donation --
    the same failure ``TensorOffsetKVCache.update_and_fetch``'s rollback slice
    has.  The join needs only the rooted results.
    """

    input_ids: Any
    hidden: Any
    captures: tuple[Any, ...]
    state_out: tuple[Any, ...]
    outputs: tuple[Any, ...]
    generation: int
    committed_count: int
    #: W67.  ``None`` at depth 1, where the prefix reads no PLE auxiliary and
    #: the join builds it exactly where the shipped route builds it.  At depth
    #: > 1 the prefix CONTAINS the PLE layer, so the auxiliary is built at the
    #: enqueue and carried here -- built once per window either way, and
    #: reused (not rebuilt) even when the join refuses the prefix.
    compiled_aux: Any = None
    #: The prefix depth this object was produced at, for the receipt.
    layer_count: int = 1


def _overlap_fold_scope(bank, fold_indices, state_in, pos, expected):
    """W66b: bind one half of a split verify's keep-mask prefix, at TRACE time.

    ``state_in`` is consumed positionally by the half's own state plan, so the
    prefix is everything after it: ``5`` row tensors per folded layer ON THIS
    SIDE plus the shared ``[1, 4*W]`` mask.  Both halves of a split carry
    their own copy of that mask because they are two graphs.  Returns ``None``
    when this half owns no folded layer, which is the same thing as the fold
    being off for it.

    The returned scope is keyed BOTH by layer index and by shadow-entry
    identity -- see ``fable_gdn_keepmask_fold.FoldPrefixScope`` for why the
    entry key alone is not enough under ``MTPLX_COMPILED_GDN``.
    """

    if not fold_indices:
        return None
    trailing = state_in[pos:]
    if len(trailing) != expected:
        raise ValueError(
            f"split verify half got {len(trailing)} keep-mask fold leaves, "
            f"expected {expected}"
        )
    return _gdn_fold.make_prefix_scope(
        fold_indices, trailing, lambda index: bank._shadow[index]
    )


@dataclass(frozen=True)
class FoldWindow:
    """W66b: one verify window's keep-mask ring, frozen for the whole cycle.

    The monolithic route needs this once; the W67 overlap pair needs it
    TWICE -- the enqueue owns layers ``0..N-1`` and the join owns
    ``N..last``, and the two must see the SAME ring, the same window stamp
    and the same mask or the two halves of one recurrence would disagree.
    So the record is built by whichever of the two runs first and reused by
    the other, and it is valid exactly while the live state leaves are the
    ones it was built from (``seen``): a commit, a rollback or a published
    state output all move those leaves and force a rebuild.

    ``rows`` is keyed by LAYER index, not by fold position, because the
    overlap split partitions on the layer boundary.
    """

    seq: int
    keeps: tuple[int, ...]
    order: tuple[int, ...]
    rows: dict[int, tuple[Any, ...]]
    bases: dict[int, Any]
    #: The same bases keyed by ``id(cache entry)``.  ``_fold_state_in`` walks
    #: the STATE PLAN, whose position equals the layer index only while no
    #: cache entry is ``None`` -- ``build_verify_state_spec`` skips those.
    #: Substituting a base into the wrong layer's slot 1 would be silent, so
    #: the substitution is addressed by the entry it belongs to.
    bases_by_entry: dict[int, Any]
    mask: Any
    entries: tuple[Any, ...]
    seen: tuple[Any, ...]

    @property
    def depth(self) -> int:
        return len(self.keeps)

    def is_live(self) -> bool:
        """True while every folded layer still holds the leaf it was built on."""

        return all(
            entry.cache[1] is leaf
            for entry, leaf in zip(self.entries, self.seen)
        )

    def layers_in(self, lo: int, hi: int | None) -> tuple[int, ...]:
        return tuple(
            index
            for index in self.order
            if index >= lo and (hi is None or index < hi)
        )

    def leaves(self, lo: int = 0, hi: int | None = None) -> list[Any]:
        """Flat trailing leaves for the folded layers in ``[lo, hi)``.

        ``5`` row tensors a layer plus ONE shared mask, and nothing at all
        when the range owns no folded layer.  The mask is repeated on both
        sides of a split because they are two graphs: 177 leaves across the
        pair against 176 on the monolithic body, which is one extra ``[1, 8]``
        bool input.
        """

        indices = self.layers_in(lo, hi)
        if not indices:
            return []
        out: list[Any] = []
        for index in indices:
            out.extend(self.rows[index])
        out.append(self.mask)
        return out


@dataclass
class GraphBankStats:
    calls: int = 0
    compiled_calls: int = 0
    fallback_calls: int = 0
    promoted_cache_entries: int = 0
    warmed_lengths: list[int] = field(default_factory=list)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    compile_errors: dict[str, int] = field(default_factory=dict)
    promotion_failures: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecDecodeGraphBank:
    """Fixed-length verify dispatcher with safe fallback instrumentation.

    `mx.compile` can capture array trees, but the stock MLX Qwen3.6 cache also
    stores decode offsets as Python integers.  Replaying a compiled closure that
    captured those integers would use stale RoPE/mask positions, so the safe
    backend refuses to compile until explicit tensor cache state lands.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        max_verify_len: int = 6,
        allow_python_cache_capture: bool = False,
        promote_tensor_offsets: bool = True,
        capture_backend: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.max_verify_len = max_verify_len
        self.allow_python_cache_capture = allow_python_cache_capture
        self.promote_tensor_offsets = promote_tensor_offsets
        self.capture_backend = resolve_gdn_capture_backend(capture_backend)
        self._capture_accepts_backend = _accepts_capture_backend(runtime)
        self.stats = GraphBankStats()
        self._compiled: dict[tuple[str, int, tuple[int, ...]], Any] = {}

    def forward_ar(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        return self._forward(
            "forward",
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )

    def forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        return self._forward(
            "capture",
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )

    def _forward(
        self,
        kind: str,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        started = time.perf_counter()
        self.stats.calls += 1
        length = _decode_length(input_ids)
        reason = self._fallback_reason(length, cache)
        if reason is not None:
            return self._fallback(
                kind,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=reason,
                started=started,
            )

        try:
            key = (kind, length, str(hidden_variant or ""), _cache_container_signature(cache))
            fn = self._compiled.get(key)
            if fn is None:
                if kind == "capture":
                    fn = self._compile_capture_length(
                        length,
                        cache=cache,
                        return_hidden=return_hidden,
                        hidden_variant=hidden_variant,
                    )
                else:
                    fn = self._compile_length(
                        length,
                        cache=cache,
                        return_hidden=return_hidden,
                        hidden_variant=hidden_variant,
                    )
                self._compiled[key] = fn
            result = fn(input_ids)
            self.stats.compiled_calls += 1
            self.stats.elapsed_s += time.perf_counter() - started
            return result
        except Exception as exc:  # pragma: no cover - exercised by real MLX cache probes
            key = type(exc).__name__
            self.stats.compile_errors[key] = self.stats.compile_errors.get(key, 0) + 1
            return self._fallback(
                kind,
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=f"compile_error:{key}",
                started=started,
            )

    def warm(
        self,
        lengths: range | list[int] | tuple[int, ...],
        *,
        cache_factory,
        token_factory,
    ) -> None:
        """Warm eligible shapes using caller-provided disposable cache/tokens."""
        for length in lengths:
            if length < 1 or length > self.max_verify_len:
                continue
            cache = cache_factory()
            tokens = token_factory(length)
            self.forward_ar(tokens, cache=cache, return_hidden=True)
            if length not in self.stats.warmed_lengths:
                self.stats.warmed_lengths.append(length)

    def to_dict(self) -> dict[str, Any]:
        data = self.stats.to_dict()
        data["max_verify_len"] = self.max_verify_len
        data["allow_python_cache_capture"] = self.allow_python_cache_capture
        data["promote_tensor_offsets"] = self.promote_tensor_offsets
        data["capture_backend"] = self.capture_backend
        data["compiled_lengths"] = sorted({length for _, length, _, _ in self._compiled})
        data["compiled_paths"] = [
            f"{kind}:{length}"
            for kind, length in sorted({(kind, length) for kind, length, _, _ in self._compiled})
        ]
        data["compiled_entry_count"] = len(self._compiled)
        return data

    def reset(self) -> None:
        """Drop compiled closures after cache container identity changes."""
        self._compiled.clear()

    def _fallback_reason(self, length: int, cache: Any) -> str | None:
        if length < 1:
            return "invalid_length"
        if length > self.max_verify_len:
            return "length_outside_graphbank"
        if cache is None:
            return None
        if self.allow_python_cache_capture:
            return None
        if self.promote_tensor_offsets:
            promoted, failures = promote_kv_cache_offsets(cache, reserve_tokens=length)
            self.stats.promoted_cache_entries += promoted
            for reason, count in failures.items():
                self.stats.promotion_failures[reason] = (
                    self.stats.promotion_failures.get(reason, 0) + count
                )
        if cache_has_python_offsets(cache):
            return "python_cache_offsets"
        return None

    def _fallback(
        self,
        kind: str,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        reason: str,
        started: float,
    ):
        self.stats.fallback_calls += 1
        self.stats.fallback_reasons[reason] = self.stats.fallback_reasons.get(reason, 0) + 1
        if kind == "capture":
            result = self._runtime_forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
        else:
            result = self.runtime.forward_ar(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
        self.stats.elapsed_s += time.perf_counter() - started
        return result

    def _compile_length(
        self,
        length: int,
        *,
        cache: Any,
        return_hidden: bool,
        hidden_variant: str | None,
    ):
        def verify_fn(input_ids):
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            return self.runtime.forward_ar(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )

        return mx.compile(
            verify_fn,
            inputs=cache_array_tree(cache),
            outputs=cache_array_tree(cache),
        )

    def _compile_capture_length(
        self,
        length: int,
        *,
        cache: Any,
        return_hidden: bool,
        hidden_variant: str | None,
    ):
        def verify_fn(input_ids):
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            return self._runtime_forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )

        return mx.compile(
            verify_fn,
            inputs=cache_array_tree(cache),
            outputs=cache_array_tree(cache),
        )

    def _runtime_forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        if self._capture_accepts_backend:
            return self.runtime.forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                capture_backend=self.capture_backend,
            )
        return self.runtime.forward_ar_capture(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )


def _decode_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    return int(shape[1])


def _cache_container_signature(cache: Any) -> tuple[int, ...]:
    if cache is None:
        return ()
    signature: list[int] = [id(cache)]
    for entry in cache:
        signature.append(id(entry))
        if entry is None:
            continue
        if hasattr(entry, "compile_state"):
            state = getattr(entry, "compile_state")
            if isinstance(state, list):
                signature.extend(id(item) for item in state)
            continue
        if hasattr(entry, "cache"):
            signature.append(id(getattr(entry, "cache")))
            continue
        state = getattr(entry, "state", None)
        if isinstance(state, list):
            signature.append(id(state))
    return tuple(signature)


def _accepts_capture_backend(runtime: Any) -> bool:
    import inspect

    try:
        signature = inspect.signature(runtime.forward_ar_capture)
    except (AttributeError, TypeError, ValueError):
        return False
    return "capture_backend" in signature.parameters


def _accepts_runtime_keyword(runtime: Any, name: str) -> bool:
    import inspect

    try:
        signature = inspect.signature(runtime.forward_ar_capture)
    except (AttributeError, TypeError, ValueError):
        return False
    return name in signature.parameters


def cache_has_python_offsets(cache: Any) -> bool:
    for entry in cache or []:
        if entry is None:
            continue
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            return True
        idx = getattr(entry, "_idx", None)
        if isinstance(idx, int):
            return True
    return False


class TensorOffsetKVCache:
    """Full-attention KV cache adapter with array-backed mutable offset.

    Stock `KVCache.offset` is a Python integer.  In a compiled verify graph that
    integer is graph-constant state, so RoPE and mask positions can silently go
    stale.  This adapter keeps the existing key/value buffers, stores the offset
    in `cache[2]`, and mutates the three-array state through operations visible
    to `mx.compile(inputs=..., outputs=...)`.
    """

    def __init__(
        self,
        keys: mx.array,
        values: mx.array,
        offset: int | mx.array,
        *,
        step: int = 256,
    ) -> None:
        offset_array = (
            offset
            if isinstance(offset, mx.array)
            else mx.array(offset, dtype=mx.int32)
        )
        self.cache = [keys, values, offset_array]
        self.rollback_state = [None, None, None]
        self.step = step
        # Growth-budget tracking (2026-07-03): the first promotion grants
        # headroom (`initial_reserve_tokens`); any capacity expansion AFTER
        # that grant means the compiled verify graph would retrace, so the
        # bank demotes the request to eager. Flag-based so the hot path never
        # adds extra offset evals.
        self._granted = False
        self.growth_after_grant = False

    @classmethod
    def from_kv_cache(cls, entry: Any, *, reserve_tokens: int) -> "TensorOffsetKVCache":
        cache = cls(
            entry.keys,
            entry.values,
            entry.offset,
            step=getattr(entry, "step", 256),
        )
        cache.ensure_capacity(int(entry.offset) + reserve_tokens)
        return cache

    @property
    def keys(self):
        return self.cache[0]

    @keys.setter
    def keys(self, value):
        self.cache[0] = value

    @property
    def values(self):
        return self.cache[1]

    @values.setter
    def values(self, value):
        self.cache[1] = value

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value):
        self.cache[2] = (
            value
            if isinstance(value, mx.array)
            else mx.array(value, dtype=mx.int32)
        )

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value

    @property
    def compile_state(self):
        return [self.cache, self.rollback_state]

    def ensure_capacity(self, needed: int) -> None:
        if self.keys is None or self.values is None:
            return
        capacity = int(self.keys.shape[2])
        if needed <= capacity:
            self._granted = True
            return
        if self._granted:
            self.growth_after_grant = True
        new_capacity = ((needed + self.step - 1) // self.step) * self.step
        extra = new_capacity - capacity
        k_shape = (*self.keys.shape[:2], extra, self.keys.shape[3])
        v_shape = (*self.values.shape[:2], extra, self.values.shape[3])
        self.keys = mx.concatenate(
            [self.keys, mx.zeros(k_shape, dtype=self.keys.dtype)],
            axis=2,
        )
        self.values = mx.concatenate(
            [self.values, mx.zeros(v_shape, dtype=self.values.dtype)],
            axis=2,
        )
        self._granted = True

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
        self.rollback_state[0] = self.cache[2]
        self.rollback_state[1] = mx.slice(
            self.cache[0],
            self.cache[2],
            axes=(2,),
            slice_size=keys.shape,
        )
        self.rollback_state[2] = mx.slice(
            self.cache[1],
            self.cache[2],
            axes=(2,),
            slice_size=values.shape,
        )
        self.cache[0] = mx.slice_update(
            self.cache[0],
            keys,
            self.cache[2],
            axes=(2,),
        )
        self.cache[1] = mx.slice_update(
            self.cache[1],
            values,
            self.cache[2],
            axes=(2,),
        )
        self.cache[2] = self.cache[2] + steps
        return self.cache[0], self.cache[1]

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        del return_array
        if self.keys is None:
            return None
        capacity = int(self.keys.shape[2])
        rinds = mx.arange(capacity)
        linds = self.cache[2] + mx.arange(N)
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + window_size)
        return mask

    def size(self):
        value = self.cache[2]
        mx.eval(value)
        return int(value.item())

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = int(n)
        if (
            self.rollback_state[0] is not None
            and self.rollback_state[1] is not None
            and self.rollback_state[2] is not None
            and int(self.rollback_state[1].shape[2]) == n
        ):
            self.cache[0] = mx.slice_update(
                self.cache[0],
                self.rollback_state[1],
                self.rollback_state[0],
                axes=(2,),
            )
            self.cache[1] = mx.slice_update(
                self.cache[1],
                self.rollback_state[2],
                self.rollback_state[0],
                axes=(2,),
            )
            self.cache[2] = self.rollback_state[0]
        else:
            self.cache[2] = mx.maximum(
                self.cache[2] - n,
                mx.array(0, dtype=self.cache[2].dtype),
            )
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes + self.cache[2].nbytes

    def demote(self):
        """Restore a stock ``KVCache`` from this adapter.

        The stock container receives the adapter's current key/value buffers
        (no copy) and the materialized integer offset, so downstream consumers
        that expect python-int offsets (postcommit, session bank snapshots)
        never see a tensor-offset adapter.
        """
        from mlx_lm.models.cache import KVCache

        entry = KVCache()
        entry.step = self.step
        entry.keys = self.cache[0]
        entry.values = self.cache[1]
        entry.offset = int(self.size()) if self.cache[0] is not None else 0
        return entry


class TensorOffsetQSACache:
    """Fixed-capacity Qwen4 QSA state for compiled target verification.

    QSA owns five graph leaves: attention keys, attention values, the logical
    token offset, raw index keys, and pooled index keys.  The logical pooled
    length is always ``offset // ratio`` and therefore does not need a second
    mutable offset.  All buffers are granted once when the verifier bank is
    constructed; the enabled path only performs fixed-shape slice updates.
    """

    fixed_capacity = True
    step = 256

    def __init__(
        self,
        kv: TensorOffsetKVCache,
        raw_keys: mx.array,
        pooled: mx.array,
        *,
        compress_ratio: int,
        rows_gather: bool = False,
        rows_gather_kv_m4: Any,
        rows_gather_enabled: bool = False,
        rows_gather_min_context: int = 0,
        fused_rows_gather_kv_m4: bool = False,
        fable_qsa_m4: bool = False,
        fable_qsa_m4_kt: bool = False,
        fable_qsa_sparse_decode: bool = False,
        fable_qsa_sparse_draft: bool = False,
    ) -> None:
        self.kv = kv
        self.aux = [raw_keys, pooled]
        self._compile_state = [self.kv.cache, self.aux]
        self.ratio = max(1, int(compress_ratio))
        self.step = int(getattr(kv, "step", 256))
        self.fixed_rows_gather = bool(rows_gather)
        self.rows_gather_kv_m4 = rows_gather_kv_m4
        self.rows_gather_enabled = bool(rows_gather_enabled)
        self.rows_gather_min_context = max(0, int(rows_gather_min_context))
        self.fused_rows_gather_kv_m4 = bool(fused_rows_gather_kv_m4)
        # MTPLX_FABLE_QSA_M4: validated once, here, at cache install. The
        # indexer reads ``fable_qsa_m4_rows`` and never re-derives the
        # decision, so one trace of the verify graph cannot disagree with the
        # next about which QSA chain it contains.
        self.fable_qsa_m4 = bool(fable_qsa_m4)
        self.fable_qsa_m4_rows = FABLE_QSA_M4_ROWS if fable_qsa_m4 else 0
        # Independent of the four bit-exact rewrites above: the transposed
        # key layout was measured slower AND not bit-exact, so it is its own
        # arm rather than a rider on the lane.
        self.fable_qsa_m4_kt = bool(fable_qsa_m4_kt)
        # MTPLX_FABLE_QSA_SPARSE_DECODE / _DRAFT: the native split-K
        # sparse-GQA attention lane.  Validated ONCE, here, at cache install:
        # this is model-build time and outside any mx.compile trace, which is
        # what lets the install run a real parity probe.  The indexer reads
        # the row counts below and never re-derives the decision, so one
        # trace of the verify graph cannot disagree with the next about which
        # attention it contains.
        #
        # The gate is asymmetric on purpose.  install() RAISES when the
        # contract cannot be met (an armed flag that cannot apply is a
        # configuration error) and DISABLES when the numerical probe fails
        # (this kernel is rounding-class; a parity miss is a measurement, and
        # turning a measurement into an outage helps nobody).
        self.fable_qsa_sparse_decode = bool(fable_qsa_sparse_decode)
        self.fable_qsa_sparse_draft = bool(fable_qsa_sparse_draft)
        self.fable_qsa_sparse_decode_rows = 0
        self.fable_qsa_sparse_draft_rows = 0
        if fable_qsa_sparse_decode or fable_qsa_sparse_draft:
            from .kernels import qsa_sparse_decode as _qsa_sparse

            if _qsa_sparse.install(
                self.kv.keys,
                self.kv.values,
                compress_ratio=self.ratio,
                verify=bool(fable_qsa_sparse_decode),
                draft=bool(fable_qsa_sparse_draft),
            ):
                self.fable_qsa_sparse_decode_rows = (
                    _qsa_sparse.VERIFY_ROWS if fable_qsa_sparse_decode else 0
                )
                self.fable_qsa_sparse_draft_rows = (
                    _qsa_sparse.DRAFT_ROWS if fable_qsa_sparse_draft else 0
                )

    @staticmethod
    def _fixed_bank(value: mx.array, capacity: int, axis: int) -> mx.array:
        current = int(value.shape[axis])
        if current == capacity:
            return value
        if current > capacity:
            slices = [slice(None)] * value.ndim
            slices[axis] = slice(0, capacity)
            return value[tuple(slices)]
        shape = list(value.shape)
        shape[axis] = capacity - current
        return mx.concatenate(
            [value, mx.zeros(tuple(shape), dtype=value.dtype)], axis=axis
        )

    @classmethod
    def from_qsa_cache(
        cls, entry: Any, *, reserve_tokens: int
    ) -> "TensorOffsetQSACache":
        reserve_tokens = max(1, int(reserve_tokens))
        offset = int(entry.offset)
        ratio = max(1, int(entry.ratio))
        if entry.raw_keys is None or entry.pooled is None:
            raise ValueError("QSA index state is empty")
        if entry.kv.keys is None or entry.kv.values is None:
            raise ValueError("QSA attention state is empty")

        logical_capacity = offset + reserve_tokens
        raw_capacity = ((logical_capacity + ratio - 1) // ratio) * ratio
        pooled_capacity = raw_capacity // ratio

        kv = TensorOffsetKVCache.from_kv_cache(
            entry.kv, reserve_tokens=reserve_tokens
        )
        kv.keys = cls._fixed_bank(kv.keys, raw_capacity, 2)
        kv.values = cls._fixed_bank(kv.values, raw_capacity, 2)
        raw = cls._fixed_bank(entry.raw_keys, raw_capacity, 1)
        pooled = cls._fixed_bank(entry.pooled, pooled_capacity, 1)
        from .models.qwen4_exp import (
            _qsa_gather_enabled,
            _qsa_gather_min_context,
        )

        rows_gather_enabled = _qsa_gather_enabled()
        rows_gather_min_context = _qsa_gather_min_context()
        rows_gather = rows_gather_enabled and offset >= rows_gather_min_context
        rows_gather_kv_m4 = entry.rows_gather_kv_m4
        fused_rows_gather_kv_m4 = _env_enabled("MTPLX_QSA_M4_FUSED_KV_GATHER")
        fable_qsa_m4 = fable_qsa_m4_enabled()
        fable_qsa_m4_kt = fable_qsa_m4_kt_enabled()
        fable_qsa_sparse_decode = fable_qsa_sparse_decode_enabled()
        fable_qsa_sparse_draft = fable_qsa_sparse_draft_enabled()
        if fable_qsa_m4_kt and not fused_rows_gather_kv_m4:
            # No silent fallback: there is no binding to transpose without it.
            raise RuntimeError(
                "MTPLX_FABLE_QSA_M4_KT requires MTPLX_QSA_M4_FUSED_KV_GATHER: "
                "the transposed key layout is an output mode of the fused "
                "K/V gather, not a standalone lane"
            )
        if fable_qsa_m4:
            # No silent fallback: an armed flag that cannot apply is a
            # configuration error, not a quiet revert to the stock chain.
            if ratio != 4:
                raise RuntimeError(
                    "MTPLX_FABLE_QSA_M4 is wired for the ratio-4 QSA lane; got "
                    f"ratio={ratio}"
                )
            if not mx.metal.is_available():
                raise RuntimeError(
                    "MTPLX_FABLE_QSA_M4 replaces four QSA sub-chains with "
                    "Metal kernels and has no portable spelling"
                )
            if pooled_capacity < 1:
                raise RuntimeError(
                    "MTPLX_FABLE_QSA_M4 requires a materialized pooled bank"
                )
        if fused_rows_gather_kv_m4:
            expected_shape = (1, 2, raw_capacity, 256)
            if not _env_enabled("MTPLX_QWEN4_FIXED_M4_VERIFY"):
                raise RuntimeError(
                    "QSA fused K/V gather requires the fixed-M4 verifier"
                )
            if not rows_gather_enabled or ratio != 4:
                raise RuntimeError(
                    "QSA fused K/V gather requires the fixed rows-gather ratio-4 lane"
                )
            if (
                tuple(kv.keys.shape) != expected_shape
                or tuple(kv.values.shape) != expected_shape
                or kv.keys.dtype != mx.bfloat16
                or kv.values.dtype != mx.bfloat16
            ):
                raise RuntimeError(
                    "QSA fused K/V gather requires BF16 "
                    "[1,2,capacity,256] cache ownership"
                )
            if rows_gather:
                from .kernels.qwen4_qsa_m4_fused_kv_gather import (
                    bind_qwen4_qsa_m4_fused_kv_gather,
                )

                rows_gather_kv_m4 = bind_qwen4_qsa_m4_fused_kv_gather(
                    capacity=raw_capacity,
                    transposed_keys=fable_qsa_m4_kt,
                )

        return cls(
            kv,
            raw,
            pooled,
            compress_ratio=ratio,
            rows_gather=rows_gather,
            rows_gather_kv_m4=rows_gather_kv_m4,
            rows_gather_enabled=rows_gather_enabled,
            rows_gather_min_context=rows_gather_min_context,
            fused_rows_gather_kv_m4=fused_rows_gather_kv_m4,
            fable_qsa_m4=fable_qsa_m4,
            fable_qsa_m4_kt=fable_qsa_m4_kt,
            fable_qsa_sparse_decode=fable_qsa_sparse_decode,
            fable_qsa_sparse_draft=fable_qsa_sparse_draft,
        )

    @property
    def compile_state(self) -> list[list[mx.array]]:
        return self._compile_state

    @property
    def raw_keys(self):
        return self.aux[0]

    @raw_keys.setter
    def raw_keys(self, value):
        self.aux[0] = value

    @property
    def pooled(self):
        return self.aux[1]

    @pooled.setter
    def pooled(self, value):
        self.aux[1] = value

    @property
    def capacity(self) -> int:
        return int(self.raw_keys.shape[1])

    def ensure_capacity(self, needed: int) -> bool:
        """Grow this installed QSA generation without changing its offset."""

        raw_capacity = (
            (max(1, int(needed)) + self.ratio - 1) // self.ratio
        ) * self.ratio
        if raw_capacity <= self.capacity:
            return False
        pooled_capacity = raw_capacity // self.ratio
        self.kv.keys = self._fixed_bank(self.kv.keys, raw_capacity, 2)
        self.kv.values = self._fixed_bank(self.kv.values, raw_capacity, 2)
        self.raw_keys = self._fixed_bank(self.raw_keys, raw_capacity, 1)
        self.pooled = self._fixed_bank(self.pooled, pooled_capacity, 1)
        self.kv._granted = True
        self.kv.growth_after_grant = False
        if self.fixed_rows_gather and self.fused_rows_gather_kv_m4:
            from .kernels.qwen4_qsa_m4_fused_kv_gather import (
                bind_qwen4_qsa_m4_fused_kv_gather,
            )

            self.rows_gather_kv_m4 = bind_qwen4_qsa_m4_fused_kv_gather(
                capacity=raw_capacity,
                transposed_keys=self.fable_qsa_m4_kt,
            )
        return True

    def activate_rows_gather(self, logical_end: int) -> bool:
        """Install the construction-validated sparse route at its threshold."""

        if (
            self.fixed_rows_gather
            or not self.rows_gather_enabled
            or int(logical_end) < self.rows_gather_min_context
        ):
            return False
        self.fixed_rows_gather = True
        if self.fused_rows_gather_kv_m4:
            from .kernels.qwen4_qsa_m4_fused_kv_gather import (
                bind_qwen4_qsa_m4_fused_kv_gather,
            )

            self.rows_gather_kv_m4 = bind_qwen4_qsa_m4_fused_kv_gather(
                capacity=self.capacity,
                transposed_keys=self.fable_qsa_m4_kt,
            )
        return True

    @property
    def offset(self):
        return self.kv.offset

    @property
    def pooled_len(self):
        return self.kv.offset // self.ratio

    @property
    def state_leaves(self) -> list[mx.array]:
        return [*self.kv.cache, *self.aux]

    def write_raw(self, keys: mx.array) -> None:
        self.raw_keys = mx.slice_update(
            self.raw_keys, keys, self.kv.offset, axes=(1,)
        )

    def write_pooled(self, blocks: mx.array, nb_start, nb_total) -> None:
        del nb_total
        self.pooled = mx.slice_update(
            self.pooled, blocks, nb_start, axes=(1,)
        )

    def pooled_f32_view(self, nb: int) -> mx.array:
        """Return the fixed pooled backing in the selector's fp32 layout.

        The ordinary QSA cache maintains a mutable mirror to avoid rebuilding
        it during eager decode.  This adapter is an explicit compiled-graph
        state carrier: deriving the view from its tracked pooled leaf keeps
        every replay self-contained and prevents an untracked mirror from
        becoming stale after a fixed-shape slice update.
        """

        return mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[:, None, :, :nb]

    def size(self) -> int:
        return self.kv.size()

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return self.kv.trim(n)

    @property
    def nbytes(self) -> int:
        return int(self.kv.nbytes + self.raw_keys.nbytes + self.pooled.nbytes)

    def demote(self):
        from .models.qwen4_exp import QSACache

        offset = self.kv.size()
        entry = QSACache(self.ratio)
        entry.kv = self.kv.demote()
        entry.raw_keys = self.raw_keys
        entry.pooled = self.pooled
        entry.pooled_len = min(int(self.pooled.shape[1]), offset // self.ratio)
        return entry


def promote_kv_cache_offsets(
    cache: Any,
    *,
    reserve_tokens: int,
    preserve_paged: bool | None = None,
    initial_reserve_tokens: int | None = None,
) -> tuple[int, dict[str, int]]:
    """Replace stock full-attention KV caches with tensor-offset adapters.

    ``preserve_paged`` controls what happens to ``VllmMetalPagedKVCache``
    entries.  When true they are promoted in place to
    ``TensorOffsetVllmMetalPagedKVCache`` (keeping the physical page buffers).
    When false the paged entry falls through to the dense promotion path,
    which reads ``entry.keys`` / ``entry.values`` — the ``.keys`` property on
    the paged cache densifies the whole cache, so paged storage is silently
    lost.  The default (``None``) preserves the historical behavior of the
    ``MTPLX_GRAPHBANK_PRESERVE_PAGED_KV`` env switch; callers that must never
    densify paged KV (e.g. ``CompiledVerifyBank``) pass ``True`` explicitly.
    """
    promoted = 0
    failures: dict[str, int] = {}
    if cache is None:
        return promoted, failures
    if preserve_paged is None:
        preserve_paged = _env_enabled("MTPLX_GRAPHBANK_PRESERVE_PAGED_KV")
    for idx, entry in enumerate(cache):
        if entry is None:
            continue
        if isinstance(entry, TensorOffsetQSACache):
            continue
        if isinstance(entry, TensorOffsetKVCache):
            entry.ensure_capacity(entry.size() + reserve_tokens)
            continue
        try:
            from .models.qwen4_exp import QSACache
        except Exception:  # pragma: no cover - optional model import
            QSACache = None
        if QSACache is not None and isinstance(entry, QSACache):
            try:
                cache[idx] = TensorOffsetQSACache.from_qsa_cache(
                    entry,
                    reserve_tokens=(
                        initial_reserve_tokens
                        if initial_reserve_tokens is not None
                        else reserve_tokens
                    ),
                )
            except (TypeError, ValueError):
                failures["auxiliary_qsa_state"] = (
                    failures.get("auxiliary_qsa_state", 0) + 1
                )
                continue
            promoted += 1
            continue
        if preserve_paged:
            try:
                from .cache_state import (
                    TensorOffsetQuantizedPagedKVCache,
                    TensorOffsetVllmMetalPagedKVCache,
                    VllmMetalPagedKVCache,
                )
            except Exception:  # pragma: no cover - import guard for minimal test envs
                TensorOffsetQuantizedPagedKVCache = None
                TensorOffsetVllmMetalPagedKVCache = None
                VllmMetalPagedKVCache = None
            if (
                VllmMetalPagedKVCache is not None
                and isinstance(entry, VllmMetalPagedKVCache)
            ):
                if entry.key_cache is None or entry.value_cache is None:
                    failures["empty_paged_kv_cache"] = (
                        failures.get("empty_paged_kv_cache", 0) + 1
                    )
                    continue
                if getattr(entry, "turboquant", False):
                    # TurboQuant pages depend on the external vLLM-Metal ops;
                    # no adapter understands them. Keep the eager refusal.
                    failures["quantized_paged_kv_cache"] = (
                        failures.get("quantized_paged_kv_cache", 0) + 1
                    )
                    continue
                if getattr(entry, "kv_quant", False):
                    # kv_quant pages promote to the quantized adapter
                    # (head-major banks + fp32 scale planes, stable leaf
                    # shapes/dtypes for the compiled graph). Fail-closed:
                    # geometry the packed-quant kernel refuses, or the env
                    # kill-switch, keeps the historical eager refusal.
                    if not _env_enabled(
                        "MTPLX_GRAPHBANK_QUANTIZED_PAGED", default=True
                    ):
                        failures["quantized_paged_kv_cache"] = (
                            failures.get("quantized_paged_kv_cache", 0) + 1
                        )
                        continue
                    if (
                        TensorOffsetQuantizedPagedKVCache is None
                        or not TensorOffsetQuantizedPagedKVCache.promotable(entry)
                    ):
                        failures["quantized_paged_kv_geometry"] = (
                            failures.get("quantized_paged_kv_geometry", 0) + 1
                        )
                        continue
                    cache[idx] = TensorOffsetQuantizedPagedKVCache.from_paged_cache(
                        entry
                    )
                    promoted += 1
                    continue
                cache[idx] = TensorOffsetVllmMetalPagedKVCache.from_paged_cache(entry)
                promoted += 1
                continue
        offset = getattr(entry, "offset", None)
        if not isinstance(offset, int):
            continue
        if getattr(entry, "_idx", None) is not None:
            failures["rotating_or_indexed_cache"] = (
                failures.get("rotating_or_indexed_cache", 0) + 1
            )
            continue
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            failures["empty_kv_cache"] = failures.get("empty_kv_cache", 0) + 1
            continue
        if (
            len(getattr(keys, "shape", ())) != 4
            or len(getattr(values, "shape", ())) != 4
        ):
            failures["unsupported_kv_shape"] = failures.get("unsupported_kv_shape", 0) + 1
            continue
        cache[idx] = TensorOffsetKVCache.from_kv_cache(
            entry,
            # First promotion may grant extra growth headroom so the compiled
            # verify graph keeps a stable leaf shape for the whole span of a
            # typical agent round; steady-state re-promotion calls above only
            # top up by `reserve_tokens` (the verify length).
            reserve_tokens=(
                initial_reserve_tokens
                if initial_reserve_tokens is not None
                else reserve_tokens
            ),
        )
        promoted += 1
    return promoted, failures


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cache_array_tree(cache: Any) -> list[Any]:
    """Return the arrays a compiled closure can legally capture."""
    tree: list[Any] = []
    for entry in cache or []:
        if entry is None:
            tree.append(None)
            continue
        if hasattr(entry, "compile_state"):
            tree.append(getattr(entry, "compile_state"))
            continue
        if hasattr(entry, "cache"):
            tree.append(getattr(entry, "cache"))
            continue
        leaves = []
        for name in ("keys", "values", "left_padding", "lengths", "_lengths"):
            if hasattr(entry, name):
                leaves.append(getattr(entry, name))
        if not leaves and hasattr(entry, "state"):
            leaves.append(entry.state)
        tree.append(leaves)
    return tree


# ---------------------------------------------------------------------------
# W2 compiled verify: pure-function verify step over a shadow cache.
#
# The June-12 poisoning failure compiled the side-effecting forward directly:
# tracer arrays were assigned into the *real* ArraysCache/paged cache lists and
# python offsets were baked into the trace as constants, so the next trace died
# with "eval an array without a primitive".  The firewall here is a persistent
# shadow cache owned by the bank: the compiled function re-seeds every shadow
# leaf from its explicit inputs BEFORE any read, runs the existing runtime
# forward against the shadow containers, and returns every leaf as an explicit
# output.  Tracers therefore never escape into the real cache; the dispatch
# wrapper mirror-commits materialized outputs into the real entries.
# ---------------------------------------------------------------------------

VERIFY_SPEC_KIND_FULL_ATTN = "fa"
VERIFY_SPEC_KIND_GDN = "gdn"
VERIFY_SPEC_KIND_QSA = "qsa"

TAPE_CAPTURE_KEYS = ("conv_states", "conv_out", "g", "state_in", "tape")
STANDARD_CAPTURE_KEYS = ("conv_states", "states")
_UNSUPPORTED_CAPTURE_BACKENDS = {
    "linear_gdn_final",  # emits {"final_only": True}; nothing to flatten
    "linear_gdn_from_conv_stream_skip0",  # capture_start-shifted layout
}


# Prewarm one-shot (F6, 2026-08-16). The shader/pipeline cache the ladder
# primes is process-global (and OS-persistent), so re-walking buckets that
# are already warm is pure waste — but the OLD one-shot boolean was spent by
# the FIRST compiled dispatch of the process, which is normally the 16-token
# boot warmup: its tiny cache clamped the walk (min paged capacity) and the
# deeper buckets then paid their ~1s compile inside the first MEASURED
# benchmark row. `_PREWARM_DONE` now means "no future walk can add
# coverage" (walk reached the router ceiling, or the cache is structurally
# ladder-free); until then, the first dispatch of each generation retries
# the walk and extends it with whatever new buckets the current cache
# capacity allows, skipping buckets already recorded in
# `_PREWARMED_BUCKETS`. A retry with nothing new to walk is a few python
# comparisons — no compiles, no kernel work.
_PREWARM_DONE = False

# Buckets already walked this process, keyed
# (runtime id, verify length, hidden variant, bucket). A recycled runtime
# id after a model swap can only SKIP a warmup walk (perf miss, never a
# correctness risk — the compiled callables themselves are guarded by the
# weakref check in _shared_or_new_verify_step).
_PREWARMED_BUCKETS: set[tuple[int, int, str, int]] = set()

# Importable prewarm truth for /health (read defensively via getattr).
# "done": no further walk can add coverage; "buckets": bucket sizes warmed
# this process; "walks": ladder walks that executed; "last_report": the most
# recent walk report (same shape as CompiledVerifyBank.stats["prewarm"]).
prewarm_status: dict[str, Any] = {
    "done": False,
    "buckets": [],
    "walks": 0,
    "last_report": None,
}

# Importable compiled-verify degradation truth for /health (F23a).
# "permanent_eager" tracks the most recently constructed bank (flipped True
# by any later runtime flip); "reason"/"flipped_at" keep the LAST flip
# forensics (sticky across requests); "flip_count" counts permanent flips
# process-wide (construction-gate flips count once per distinct reason, not
# once per request); "transient_exception_count" counts per-call exception
# fallbacks that did NOT flip the bank.
compiled_verify_status: dict[str, Any] = {
    "mode": None,
    "permanent_eager": False,
    "reason": None,
    "flipped_at": None,
    "flip_count": 0,
    "transient_exception_count": 0,
}

_PERMANENT_EAGER_LOGGED: set[str] = set()


def _record_permanent_eager(reason: str, *, once: bool = False) -> None:
    """Record (and log once per distinct reason) a permanent-eager flip.

    ``once=True`` marks deterministic construction-time flips (per-model
    quant gate): the first bank records and logs; subsequent per-request
    banks only re-assert ``permanent_eager`` without inflating the count.
    """
    already_logged = reason in _PERMANENT_EAGER_LOGGED
    compiled_verify_status["permanent_eager"] = True
    if once and already_logged:
        return
    compiled_verify_status["reason"] = reason
    compiled_verify_status["flipped_at"] = time.time()
    compiled_verify_status["flip_count"] = (
        int(compiled_verify_status.get("flip_count", 0)) + 1
    )
    if not already_logged:
        _PERMANENT_EAGER_LOGGED.add(reason)
        try:
            print(
                "[mtplx] compiled-verify permanent-eager: "
                + reason
                + " (verify runs the eager path from here)",
                flush=True,
            )
        except Exception:
            pass

# Process-global compiled verify callables, keyed by
# (runtime id, capture backend, state spec, verify length, hidden variant,
# bucket). The bank is per-generation; without sharing, every request pays a
# fresh trace. Values are (compiled_fn, trace_host) where trace_host["bank"]
# is re-pointed to the live bank before each dispatch so internal retraces
# (mx.compile re-traces on leaf-shape changes) always use live scratch
# containers. See CompiledVerifyBank._shared_or_new_verify_step.
_SHARED_VERIFY_STEPS: dict[tuple, tuple[Any, dict[str, Any]]] = {}

# W67: the same sharing, for the graph-build-overlap pair.  Without it the
# bank -- which is constructed per generation -- would build TWO fresh
# closures per request and mx.compile would re-trace both on the request's
# first cycle, where the shipped monolithic route pays that trace once per
# PROCESS.  (The monolithic docstring prices one trace at ~1 s wall at 7k
# leaves.)  Values are (prefix_fn, suffix_fn, trace_host, runtime weakref);
# trace_host["bank"] is re-pointed at the live bank on every hit.
_SHARED_OVERLAP_SPLITS: dict[tuple, tuple[Any, Any, dict[str, Any], Any]] = {}


def _prewarm_enabled() -> bool:
    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_PREWARM", "1")).strip().lower()
    return raw not in {"0", "false", "off", ""}


def compiled_verify_mode() -> str:
    """Resolve MTPLX_COMPILED_VERIFY into 'off' | 'on' | 'parity' | 'parity2'.

    ``parity``  — double-run with the eager leg authoritative; abort on the
                  first mismatch (Gate A: per-call bit-exactness).
    ``parity2`` — double-run with the COMPILED leg authoritative and an eager
                  clone tracking it; log mismatches, never abort (Gate B:
                  does compiled-committed state evolution diverge?).
    """
    raw = (os.environ.get("MTPLX_COMPILED_VERIFY") or "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"parity", "parity2"}:
        return raw
    return "on"


def _next_pow2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _owned_state_env_active(name: str) -> bool:
    """True when an owned-state wrapper env is set to any enabling value.

    These envs carry mode names (e.g. ``persistent_eval``) rather than plain
    booleans, so anything other than empty/off counts as active.
    """
    raw = (os.environ.get(name) or "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def build_verify_state_spec(cache: Any) -> tuple[list[tuple[int, str, int]] | None, str | None]:
    """Ordered (layer_idx, kind, n_leaves) spec over the cache list.

    Full-attention tensor-offset entries contribute every slot of their
    ``cache`` list (three for plain KV/paged adapters, five for the
    quantized paged adapter — payloads, offset, scale planes); GDN
    ``ArraysCache`` entries contribute their two slots.  ``None`` entries
    contribute nothing.  Any other container makes the cache non-compilable
    and returns ``(None, reason)``.
    """
    try:
        from mlx_lm.models.cache import ArraysCache
    except Exception:  # pragma: no cover - mlx_lm always present in product envs
        ArraysCache = None
    try:
        from .cache_state import TensorOffsetVllmMetalPagedKVCache
    except Exception:  # pragma: no cover - import guard for minimal test envs
        TensorOffsetVllmMetalPagedKVCache = None

    spec: list[tuple[int, str, int]] = []
    for idx, entry in enumerate(cache or []):
        if entry is None:
            continue
        if isinstance(entry, TensorOffsetQSACache):
            spec.append((idx, VERIFY_SPEC_KIND_QSA, 5))
            continue
        if isinstance(entry, TensorOffsetKVCache) or (
            TensorOffsetVllmMetalPagedKVCache is not None
            and isinstance(entry, TensorOffsetVllmMetalPagedKVCache)
        ):
            spec.append((idx, VERIFY_SPEC_KIND_FULL_ATTN, len(entry.cache)))
            continue
        if ArraysCache is not None and isinstance(entry, ArraysCache):
            if len(entry.cache) not in (2, 4):
                return None, f"unsupported_container:ArraysCache[{len(entry.cache)}]"
            n_leaves = len(entry.cache)
            if n_leaves == 4 and any(leaf is None for leaf in entry.cache):
                return None, "unsupported_container:ArraysCache[partial_ple]"
            spec.append((idx, VERIFY_SPEC_KIND_GDN, n_leaves))
            continue
        return None, f"unsupported_container:{type(entry).__name__}"
    return spec, None


def _paged_kernel_bucket_eligible(entry: Any, length: int, bucket: int) -> bool:
    """Best-effort eager mirror of the compiled paged-attention kernel gates.

    Plain adapters mirror ``sdpa_2pass_paged_tail_dynamic_offset``; the
    quantized adapter mirrors ``sdpa_gqa_packed_tail_quant``. A miss here is
    a performance decision, not a correctness one: inside the compiled
    function the kernel declining simply routes to the pure dense
    ``cache.state`` math, which stays trace-safe.
    """
    key_cache = entry.cache[0]
    value_cache = entry.cache[1]
    if key_cache is None or value_cache is None:
        return False
    if not mx.metal.is_available():
        return False
    try:
        from .cache_state import TensorOffsetQuantizedPagedKVCache
    except Exception:  # pragma: no cover - import guard for minimal test envs
        TensorOffsetQuantizedPagedKVCache = None
    if TensorOffsetQuantizedPagedKVCache is not None and isinstance(
        entry, TensorOffsetQuantizedPagedKVCache
    ):
        # Packed-quant kernel gates (head-major banks): two query banks cap
        # the verify window at 8 rows; payload dtype must match the bits;
        # head dims come from the adapter's own metadata. GQA legality
        # (32 * factor <= 1024) needs the query head count and stays a
        # per-call kernel gate inside the graph.
        if length > 8:
            return False
        bits = int(entry.kv_bits)
        expect_kv = mx.int8 if bits == 8 else mx.uint8
        if key_cache.dtype != expect_kv or value_cache.dtype != expect_kv:
            return False
        head_dim = int(entry.head_dims[0])
        if head_dim not in (64, 128, 256) or int(entry.head_dims[1]) != head_dim:
            return False
        from .kernels.sdpa_gqa_packed_quant import _static_blocks

        blocks = _static_blocks(int(entry.capacity), int(bucket) or None)
        return blocks > 0 and blocks % 32 == 0
    if key_cache.dtype not in (mx.bfloat16, mx.float16):
        return False
    if key_cache.dtype != value_cache.dtype:
        return False
    if int(entry.block_size) != int(key_cache.shape[1]):
        return False
    head_dim = int(key_cache.shape[3])
    if head_dim != int(value_cache.shape[3]) or head_dim not in {64, 96, 128, 256}:
        return False
    max_q = int(os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q", "16") or "16")
    if length > max_q:
        return False
    from .kernels.sdpa_2pass import _compute_blocks

    blocks = _compute_blocks(max(1, int(length)), int(bucket))
    return blocks > 0 and blocks % 32 == 0


def _as_numpy(value: Any):
    import numpy as np

    try:
        import mlx.core as mx

        if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
            # numpy has no bf16 buffer support; widening to float32 is exact
            # (every bf16 maps to a unique float32), so bit-equality on the
            # widened arrays is bit-equality on the originals.
            return np.asarray(value.astype(mx.float32))
    except Exception:
        pass
    return np.asarray(value)


def _copy_state_leaf(leaf: Any) -> Any:
    """Materialized copy of a cache state leaf.

    ``mx.array(existing)`` allocates a fresh buffer (dtype-preserving, immune
    to donation of the source), which is what lets the parity2 eager clone
    replay a verify step without sharing a single buffer with the live
    compiled-authoritative stream.
    """
    if isinstance(leaf, mx.array):
        return mx.array(leaf)
    return leaf


def _artifact_kind(name: str) -> str:
    """Map a compare_verify_outputs leaf name to its artifact family."""
    if name == "logits":
        return "logits"
    if name == "hidden":
        return "hidden"
    if name.startswith("capture["):
        return "capture"
    if name.startswith("state["):
        return "state"
    return "other"


def _leaf_max_abs_diff(reference: Any, candidate: Any) -> float | None:
    """Max-abs difference between two leaves, or None when incomparable."""
    import numpy as np

    if reference is None or candidate is None:
        return None
    if not hasattr(reference, "shape") or not hasattr(candidate, "shape"):
        return None
    ref_np = _as_numpy(reference)
    cand_np = _as_numpy(candidate)
    if ref_np.shape != cand_np.shape:
        return None
    try:
        diff = np.asarray(ref_np, dtype=np.float64) - np.asarray(
            cand_np, dtype=np.float64
        )
    except (TypeError, ValueError):
        return None
    if not diff.size:
        return 0.0
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.abs(diff)))


def _compiled_verify_max_context() -> int:
    """Context ceiling for the compiled verify step (tokens). Beyond it the
    bank falls back to eager for that call. Default 6144 = the highest
    context Gate A has proven bit-exact AND the ABBA showed +4.8%; past it
    the 2026-07-02 long-form pair measured -28% with a seed-0 trajectory
    fork (boundary materialization scales with context; bucket-crossing
    numerics untested). 0 disables the ceiling (experiments only)."""
    import os

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "6144")).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 6144
    return max(0, value)


def _compiled_verify_boundary() -> str:
    import os

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")).strip().lower()
    return raw if raw in ("both", "pre", "post", "none") else "both"


def _compiled_verify_donation_enabled() -> bool:
    """A2.1 commit-first ownership handoff (speed-war Lane A2, 2026-07-06).

    Donation of a KV buffer into its in-graph ``slice_update`` requires the
    graph to hold the ONLY reference when the graph is scheduled.  The
    historical dispatch order (async_eval outputs -> mirror-commit) kept the
    real cache entries and the ``state_in`` list alive at schedule time, so
    every compiled verify call materialized a full copy of every full-attn
    K and V buffer: measured 16.5 ms at 64k / ~33 ms at 128k per call
    (compiled_copy_tax_probe.py arms A vs G, 2026-07-06).  Committing the
    output leaves into the real cache FIRST and dropping the dispatcher
    reference before ``async_eval`` unblocks donation with byte-identical
    results (chained-pending + snapshot-COW proof:
    compiled_copy_tax_correctness.py).  Default ON; env kill-switch for
    A/B and emergency revert.
    """
    import os

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_DONATION", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _batch_paged_offsets_enabled() -> bool:
    """Batch-materialize paged-KV offsets before the bucket walk (#318 port).

    ``TensorOffsetVllmMetalPagedKVCache.size()`` does ``mx.eval(cache[2])``
    per entry, so after a trim/rollback (offsets left lazy) the bucket walk
    forces one serial host sync per full-attention entry.  Evaluating every
    offset in one ``mx.eval`` first turns N syncs into one; ``mx.eval``
    cannot change values, so the result is exact by construction.  Neutral
    on non-trimming workloads (offsets already materialized).  Ported from
    grzracz PR #318 with the env read hoisted out of the hot call.  Default
    ON since the night-20260822 round-4 ruling (n=4 counterbalanced ABBA
    blend +2.7% mean, byte-identity held greedy+sampled); "0" opts out.
    """
    import os

    raw = str(os.environ.get("MTPLX_BATCH_PAGED_OFFSETS", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


_BATCH_PAGED_OFFSETS = _batch_paged_offsets_enabled()

# Long-context fence for the #318 default (night-20260822 quad: the trio
# stack measured −2.9%/−2.7% at 16k/32k while short/mid rungs blend
# +2.5..+9.8). generation's per-request prebind sets this from the shared
# MTPLX_GREEDY_TRIO_MAX_CONTEXT fence; requests that never prebind (batch
# lane) keep the last-set/default value — that lane pays at most the
# pre-#318 serial-sync behavior, never a correctness change.

_PAGED_OFFSETS_CONTEXT_OK: ContextVar[bool] = ContextVar(
    "mtplx_paged_offsets_context_ok", default=True
)


def set_paged_offsets_context_ok(allowed: bool):
    """Per-request fence stamp from generation's trio prebind."""
    return _PAGED_OFFSETS_CONTEXT_OK.set(bool(allowed))


def paged_offsets_context_ok() -> bool:
    """Read the current request's fence stamp (receipts/trace)."""
    return _PAGED_OFFSETS_CONTEXT_OK.get()


def _ccopy_bank_max_len() -> int:
    """Ceiling for extended-window (context-copy block) compiled dispatch.

    Copy blocks are proposed at their native ladder lengths (block 8-32 ->
    T=9-33); the bank verifies them one-shot so the trajectory is byte-equal
    to the eager copy lane (v1's cap-to-bank-window changed the proposal and
    was falsified as a net win, MEASUREMENTS 2026-08-25 12:05). Default 33
    covers the full default ladder; longer custom MTPLX_CONTEXT_COPY_K
    proposals fall back eager per call.
    """
    raw = os.environ.get("MTPLX_CCOPY_BANK_MAX_LEN", "").strip()
    try:
        return max(1, int(raw)) if raw else 33
    except ValueError:
        return 33


def _compiled_verify_growth_reserve() -> int:
    """Dense-leaf growth headroom granted at first promotion (tokens).

    Sized so a typical agent tool round (40-500 generated tokens) completes
    inside one stable leaf shape: one trace per (length, capacity) class,
    zero mid-round retraces. Long generations exceed the grant and demote to
    eager for the request remainder.
    """

    raw = str(os.environ.get("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "512")).strip()
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 512


def _fixed_m4_initial_growth_reserve() -> int:
    """Construction-time reserve for the strict Qwen4 fixed-M4 lane."""

    if "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE" in os.environ:
        return _compiled_verify_growth_reserve()
    return 1024


_FIXED_M4_MAX_GROWTH_TOKENS = 16384


def _next_fixed_m4_growth_tokens(current: int) -> int:
    """Next construction-owned grant for an overrun fixed-M4 generation."""

    current = max(1, int(current))
    return min(
        current * 2,
        max(current, _FIXED_M4_MAX_GROWTH_TOKENS),
    )


def _fixed_m4_capacity_growth(
    *,
    capacity: int,
    required_end: int,
    growth_tokens: int,
    capacity_limit: int | None,
) -> tuple[int, int]:
    """Resolve one host-boundary capacity transition and its next grant."""

    next_capacity = max(
        int(required_end),
        int(capacity) + max(1, int(growth_tokens)),
    )
    if capacity_limit is not None:
        next_capacity = max(
            int(required_end),
            min(next_capacity, int(capacity_limit)),
        )
    return next_capacity, _next_fixed_m4_growth_tokens(growth_tokens)


def _post_restore_eager_rounds() -> int:
    """Verify rounds routed eager after a large session-bank restore (opt-in).

    A restored cache (clone or bank reference lease) arrives with exact-size
    KV buffers, so the first compiled-route promotion ensure_capacity ->
    mx.concatenate's the restored KV per full-attention layer before the
    round can run. Deferring the first round(s) to eager moves that copy off
    the TTFT path; promotion happens one round later, mid-stream.

    DEFAULT 0 (off). Clean-room A/B 2026-08-06 (4k restore, fresh server):
    the promotion copy measured sub-milliseconds at 4k context (the 08-05
    turbo warm anomaly was dominated by first-shape-in-process compile
    traces plus postcommit stacking, not the copy), while the deferral's
    eager->compiled transition introduced one novel verify-shape trace
    (~100-200ms once per process). Net: no receipt that the deferral helps
    at agent-scale contexts, one measured cost. The copy grows linearly
    with restored context (~2 GB at 32k), so the lever may still pay at
    16k+ restores — enable via env and gate before flipping any default.
    """

    raw = os.environ.get(
        "MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS", ""
    ).strip()
    if raw:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0
    return 0


def _post_restore_min_tokens() -> int:
    """Restored-prefix size below which the post-restore deferral stays off.

    Small restores copy little (a 512-token prefix is ~tens of MB across the
    full-attention layers); the deferral only earns its round for mid/long
    contexts where the concatenate cost is user-visible.
    """

    raw = os.environ.get(
        "MTPLX_COMPILED_VERIFY_POST_RESTORE_MIN_TOKENS", ""
    ).strip()
    if raw:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 2048
    return 2048


def _runtime_trunk_quant_bits(runtime: Any) -> int | None:
    """Bits of the first quantized trunk projection, or None if unquantized.

    Used by the turbo-profile per-model gate. 4-bit (Optimized-Speed) and
    8-bit (Optimized-Quality) trunks are measured wins with the growth-demote
    + shared-traces bank (2026-07-04 re-measure: q8 +10% bare / flat @7k /
    +6% rules-context, parity2 zero divergences — the 07-02 sprint's q8
    -15/-18% verdict was the per-request trace tax, since removed). Other
    quantizations (6-bit 9B) stay eager until measured.
    """

    try:
        model = getattr(runtime, "model", None)
        text_model = getattr(model, "language_model", model)
        inner = getattr(text_model, "model", text_model)
        for layer in getattr(inner, "layers", []) or []:
            for attr_path in (
                ("self_attn", "q_proj"),
                ("mlp", "gate_proj"),
                ("linear_attn", "in_proj_qkvz"),
            ):
                node = layer
                for name in attr_path:
                    node = getattr(node, name, None)
                    if node is None:
                        break
                bits = getattr(node, "bits", None)
                if bits is not None:
                    return int(bits)
        return None
    except Exception:
        return None


def _compiled_verify_bits_gate_ok(runtime: Any) -> bool:
    if _env_enabled("MTPLX_COMPILED_VERIFY_FORCE"):
        return True
    bits = _runtime_trunk_quant_bits(runtime)
    # Measured-win allowlist: 4-bit and 8-bit affine trunks engage;
    # unquantized (None) passes for test rigs and bf16 research models.
    # Unmeasured quantizations (e.g. the 6-bit 9B) stay eager.
    return bits is None or bits in (4, 8)


def compare_verify_outputs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_report_lines: int = 24,
) -> list[str]:
    """Exact-equality diff between two named verify output trees.

    Both arguments are flat mappings ``name -> leaf`` where leaves are arrays
    (mx or numpy) or plain python values.  Returns human-readable mismatch
    lines; an empty list means bit-exact agreement.
    """
    import numpy as np

    lines: list[str] = []

    def add(line: str) -> None:
        if len(lines) < max_report_lines:
            lines.append(line)
        elif len(lines) == max_report_lines:
            lines.append("... report truncated ...")

    for name in sorted(set(reference) | set(candidate)):
        if name not in reference:
            add(f"{name}: missing from reference output")
            continue
        if name not in candidate:
            add(f"{name}: missing from candidate output")
            continue
        ref = reference[name]
        cand = candidate[name]
        if ref is None or cand is None:
            if ref is not cand:
                add(f"{name}: one side is None ({type(ref).__name__} vs {type(cand).__name__})")
            continue
        if not hasattr(ref, "shape") and not hasattr(cand, "shape"):
            if ref != cand:
                add(f"{name}: value mismatch ({ref!r} vs {cand!r})")
            continue
        ref_np = _as_numpy(ref)
        cand_np = _as_numpy(cand)
        if ref_np.shape != cand_np.shape:
            add(f"{name}: shape mismatch ({ref_np.shape} vs {cand_np.shape})")
            continue
        if ref_np.dtype != cand_np.dtype:
            add(f"{name}: dtype mismatch ({ref_np.dtype} vs {cand_np.dtype})")
            continue
        if not np.array_equal(ref_np, cand_np):
            both = np.asarray(ref_np, dtype=np.float64) - np.asarray(cand_np, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                max_abs = float(np.nanmax(np.abs(both))) if both.size else 0.0
            mismatched = int(np.sum(ref_np != cand_np))
            add(
                f"{name}: value mismatch (elements={mismatched}/{ref_np.size}, "
                f"max_abs_diff={max_abs:.3e})"
            )
    return lines


class CompiledVerifyParityError(RuntimeError):
    """Raised in parity mode when compiled and eager verify outputs diverge."""

    def __init__(self, report: list[str]) -> None:
        self.report = list(report)
        super().__init__(
            "compiled verify parity mismatch:\n" + "\n".join(self.report)
        )


class CompiledVerifyBank:
    """Compiled speculative-verify dispatcher with a shadow-cache firewall.

    ``verify_step(input_ids, *state_in) -> (logits, hidden, *captures_flat,
    *state_out)`` is a pure function: every piece of cache state enters as an
    explicit input leaf and leaves as an explicit output leaf.  The dispatch
    wrapper reads the leaves from the real (promoted) cache entries, calls the
    compiled function, and mirror-commits the outputs back into the real
    entries with ``rollback_state`` cleared so the untouched accept
    (``commit_captured_prefix``) and reject (``rollback_after_verify`` ->
    offset-only ``trim``) paths keep working unchanged.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        max_verify_len: int | None = None,
        request_max_tokens: int | None = None,
        capture_backend: str | None = None,
        parity: bool = False,
        parity2: bool = False,
        restored_tokens: int = 0,
    ) -> None:
        self.runtime = runtime
        if max_verify_len is None:
            raw = os.environ.get("MTPLX_COMPILED_VERIFY_MAX_LEN", "").strip()
            max_verify_len = int(raw) if raw else 6
        self.max_verify_len = int(max_verify_len)
        self.request_max_tokens = (
            None if request_max_tokens is None else max(0, int(request_max_tokens))
        )
        self.speculative_headroom = (
            self.max_verify_len if self.request_max_tokens is not None else 0
        )
        self.strict_no_fallback = bool(
            getattr(runtime, "qwen4_fixed_m4_compiled_verify", False)
        )
        # Generic banks let the request budget only TIGHTEN the reserve; it
        # never raises it past the env ceiling. Server requests default max_tokens to the
        # whole remaining context window (~262k on a 256k model), and
        # granting that verbatim made every request materialize a
        # multi-gigabyte KV reserve across all promoted leaves at first
        # promotion: +17 GB active / 44 GB peak, decode opening at ~13 tok/s
        # for the first ~150 tokens of every turn, and 8.8x commit cost
        # (2.4.0 short-turn regression, root-caused 2026-07-31). A bounded
        # grant restores the growth-demotion contract below: agent-length
        # rounds run fully compiled, longer generations demote to eager for
        # the request remainder (measured flat vs eager-only). Explicit
        # small budgets still reserve exactly budget + one speculative
        # window; raise MTPLX_COMPILED_VERIFY_GROWTH_RESERVE to widen the
        # stable-capacity generation. The construction-owned Qwen4 fixed-M4
        # lane has its own 1K default, then grows and reinstalls its graph at
        # capacity boundaries instead of demoting to eager. An explicit env
        # reserve remains authoritative for both lanes.
        reserve_ceiling = (
            _fixed_m4_initial_growth_reserve()
            if self.strict_no_fallback
            else _compiled_verify_growth_reserve()
        )
        self.growth_reserve_tokens = (
            min(
                self.request_max_tokens + self.speculative_headroom,
                max(
                    reserve_ceiling,
                    self.max_verify_len,
                ),
            )
            if self.request_max_tokens is not None
            else reserve_ceiling
        )
        self.capture_backend = resolve_gdn_capture_backend(capture_backend)
        self.parity = bool(parity)
        self.parity2 = bool(parity2)
        if self.parity and self.parity2:
            raise ValueError(
                "CompiledVerifyBank: parity and parity2 are mutually exclusive"
            )
        self.permanent_eager = False
        self.permanent_eager_reason: str | None = None
        compiled_verify_status["mode"] = (
            "parity" if self.parity else ("parity2" if self.parity2 else "on")
        )
        compiled_verify_status["permanent_eager"] = False
        if not parity and not parity2 and not _compiled_verify_bits_gate_ok(runtime):
            # Per-model promotion gate: 4-bit and 8-bit affine trunks engage
            # (both parity2-validated; q8's early -15/-18% reading predated
            # the 2.4.0 compiled stack — measured 2026-07-31: q8 304/304
            # compiled, 0 fallbacks, 41.3 tok/s at league parity). Unmeasured
            # quantizations (e.g. the 6-bit 9B) stay eager.
            self.permanent_eager = True
            self.permanent_eager_reason = (
                f"quant_bits_gate:bits={_runtime_trunk_quant_bits(runtime)}"
            )
            _record_permanent_eager(self.permanent_eager_reason, once=True)
        self._capture_accepts_backend = _accepts_capture_backend(runtime)
        capture_layout = getattr(runtime, "_mtplx_capture_layout", None)
        self._capture_layout_override = (
            None
            if capture_layout is None
            else tuple(str(name) for name in capture_layout)
        )
        self._extra_capture_layout = tuple(
            (int(layer_index), tuple(str(name) for name in names))
            for layer_index, names in tuple(
                getattr(runtime, "_mtplx_capture_extra_layout", ()) or ()
            )
        )
        prepare_aux = getattr(runtime, "prepare_compiled_verify_aux", None)
        self._prepare_compiled_aux = prepare_aux if callable(prepare_aux) else None
        build_fixed_aux = getattr(runtime, "build_fixed_m4_compiled_verify_aux", None)
        self._build_fixed_m4_aux = (
            build_fixed_aux if callable(build_fixed_aux) else None
        )
        graph_fixed_aux = getattr(
            runtime, "dequantize_fixed_m4_compiled_verify_aux", None
        )
        self._graph_fixed_m4_aux = (
            graph_fixed_aux if callable(graph_fixed_aux) else None
        )
        commit_captures = getattr(runtime, "commit_compiled_verify_captures", None)
        self._commit_compiled_captures = (
            commit_captures if callable(commit_captures) else None
        )
        self._runtime_accepts_compiled_aux = _accepts_runtime_keyword(
            runtime, "compiled_aux"
        )
        if self._prepare_compiled_aux is not None and not self._runtime_accepts_compiled_aux:
            raise TypeError(
                "compiled verify auxiliary preparation requires a compiled_aux input"
            )
        self._compiled: dict[tuple[int, str, int], Any] = {}
        # MTPLX_FABLE_GDN_KEEPMASK_FOLD state, all resolved once by
        # `_resolve_gdn_keepmask_fold` at fixed-M4 install and never after.
        self._fold_layer_indices: tuple[int, ...] = ()
        self._fold_entries: tuple[Any, ...] = ()
        self._fold_windows: int = 0
        self._fold_dtype: Any = None
        #: The ring frozen for the window in flight; shared by the monolithic
        #: body and by both halves of the W67 overlap pair.
        self._fold_window: Any = None
        self._spec: list[tuple[int, str, int]] | None = None
        self._shadow: list[Any] | None = None
        self._shadow_signature: tuple[Any, ...] | None = None
        self._gdn_meta_cache: dict[int, dict[str, int] | None] = {}
        self._exception_failures = 0
        self._held_state_refs: list = []
        self._held_aux_refs: list = []
        self._held_fixed_m4_split_refs: list = []
        # MTPLX_FABLE_GRAPH_BUILD_OVERLAP (W63, default off).  One slot, not a
        # list: exactly one layer-0 prefix can be in flight, it is consumed by
        # the same cycle's verify, and an unconsumed one is discarded rather
        # than accumulated (``_held_fixed_m4_split_refs`` has no such trim and
        # would pin one layer-0 state + capture set per window for the whole
        # generation).
        self._fixed_m4_overlap_prefix: FixedM4OverlapPrefix | None = None
        self._fixed_m4_split_generation: int = -1
        # W67: the prefix depth this bank compiled its overlap pair at.  Set
        # by ``arm_fixed_m4_graph_build_overlap`` from
        # ``MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS``; 1 is W63's partition.
        self._fixed_m4_overlap_layers: int = (
            _graph_build_overlap.DEFAULT_LAYERS
        )
        # W67: time the FIRST replay of each graph -- where mx.compile
        # actually traces -- always, not only under the `timing` item.  After
        # the first cycle these are False and the sites are a bool test, so
        # the steady state pays what it paid before.
        self._fixed_m4_overlap_first_prefix: bool = True
        self._fixed_m4_overlap_first_suffix: bool = True
        # The Qwen4 fixed-M4 lane installs one construction-owned replay plan
        # after prompt prefill.  Production calls then bypass the generic
        # eligibility, promotion, bucket, shadow, and fallback machinery.
        # Parity modes intentionally stay on the generic dispatcher because
        # they need its eager comparison paths.
        self._fixed_m4_dispatch: dict[str, Any] | None = None
        # Context-copy block rounds run a SECOND, wider forward on the same
        # cache. On the fixed-M4 lane they were eager (see
        # ``install_copy_round``); this holds the construction-bound compiled
        # replay for them at one fixed physical width.
        self._copy_round_dispatch: dict[str, Any] | None = None
        # Generic lanes demote when dense leaves outgrow the initial grant.
        # The fixed-M4 lane instead performs explicit capacity-generation
        # transitions while keeping its installed direct route.
        self._growth_demoted = False
        self._dense_capacity_grant: dict[int, int] | None = None
        # Post-restore warmup: a session-bank restore hands this generation
        # exact-size KV buffers, so the first promotion concatenate-copies the
        # whole restored context (see _post_restore_eager_rounds). Parity
        # modes keep full compiled coverage for the exactness harnesses.
        self._post_restore_eager_remaining = (
            _post_restore_eager_rounds()
            if (
                int(restored_tokens or 0) >= _post_restore_min_tokens()
                and not parity
                and not parity2
                and not self.strict_no_fallback
            )
            else 0
        )
        self.stats: dict[str, Any] = {
            "calls": 0,
            "compiled_calls": 0,
            "extended_calls": 0,
            "fallback_calls": 0,
            "fallback_reasons": {},
            "buckets": {},
            "promoted": 0,
            "demotions": 0,
            "traces": 0,
            "parity_checks": 0,
            "parity_failures": 0,
            "parity2_calls": 0,
            "parity2_divergent_calls": 0,
            "parity2_first_divergence": None,
            "growth_demotions": 0,
            "growth_handoff_materializations": 0,
            "growth_handoff_state_leaves": 0,
            "growth_handoff_materialize_time_s": 0.0,
            "fixed_m4_capacity_transitions": 0,
            "fixed_m4_route_transitions": 0,
            "copy_round_compiled_calls": 0,
            "copy_round_width": None,
        }

    # -- public API ---------------------------------------------------------

    def install_fixed_m4(
        self,
        cache: Any,
        *,
        prompt_ids,
        hidden_variant: str | None,
    ) -> None:
        """Install the exact Qwen4 physical-M4 replay once after prefill."""

        if not self.strict_no_fallback:
            raise ValueError("fixed-M4 installation requires the Qwen4 runtime route")
        if self.parity or self.parity2:
            raise ValueError("fixed-M4 direct replay is disabled in parity modes")

        class _M4Shape:
            shape = (1, 4)

        reason = self._fallback_reason(_M4Shape(), cache, True)
        if reason is not None:
            raise RuntimeError(f"qwen4 fixed-M4 installation refused: {reason}")
        bucket = self._resolve_bucket(cache, 4)
        if bucket != 0:
            raise RuntimeError(
                f"qwen4 fixed-M4 installation requires dense state; bucket={bucket}"
            )
        self._ensure_shadow(cache)
        state_plan = tuple(
            (kind, cache[idx], n_leaves)
            for idx, kind, n_leaves in self._spec or ()
        )
        if not state_plan or any(
            kind not in (VERIFY_SPEC_KIND_QSA, VERIFY_SPEC_KIND_GDN)
            for kind, _entry, _n in state_plan
        ):
            raise RuntimeError("qwen4 fixed-M4 installation found unsupported state")
        qsa_entries = tuple(
            entry for kind, entry, _n in state_plan if kind == VERIFY_SPEC_KIND_QSA
        )
        if not qsa_entries:
            raise RuntimeError("qwen4 fixed-M4 installation found no QSA state")
        route_key = int(all(entry.fixed_rows_gather for entry in qsa_entries))
        pending_route_thresholds = tuple(
            entry.rows_gather_min_context
            for entry in qsa_entries
            if entry.rows_gather_enabled and not entry.fixed_rows_gather
        )

        capture_plan = []
        capture_pos = 0
        for idx, names in self._extra_capture_layout:
            capture_plan.append((cache[idx], capture_pos, len(names)))
            capture_pos += len(names)

        # MTPLX_FABLE_GDN_KEEPMASK_FOLD (W66b).  Resolved here, once, BEFORE
        # `_shared_or_new_verify_step` traces: the trace bakes in whether the
        # step kernels take a prefix, and the shared-trace key below carries
        # the same dimension so a folded trace can never be served to a bank
        # that is not passing one.  Contract failures raise; this is the
        # construction path, and an armed-but-inert flag would masquerade as a
        # neutral A/B result.
        self._resolve_gdn_keepmask_fold(cache)
        self._fold_window = None

        boundary = _compiled_verify_boundary()
        if self._build_fixed_m4_aux is not None and boundary in ("both", "pre"):
            prepare_aux = self._build_fixed_m4_aux(cache, prompt_ids)
            graph_aux = self._graph_fixed_m4_aux
            if graph_aux is None:
                returns_aux = False
                aux_contract = "materialized"
            else:
                returns_aux = True
                aux_contract = "raw_q4"
        else:
            prepare_aux = partial(
                _prepare_fixed_m4_materialized,
                self._prepare_compiled_aux,
                cache,
            )
            graph_aux = None
            returns_aux = False
            aux_contract = "materialized"
        key = (4, str(hidden_variant or ""), route_key, aux_contract)
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._shared_or_new_verify_step(
                key,
                4,
                hidden_variant,
                graph_aux=graph_aux,
                return_compiled_aux=returns_aux,
                fold_indices=self._fold_layer_indices,
            )
            self._compiled[key] = fn
        bind_device_commit = getattr(
            self.runtime, "bind_fixed_m4_device_commit", None
        )
        device_commit = (
            bind_device_commit(cache) if callable(bind_device_commit) else None
        )
        initial_growth_tokens = max(
            self.max_verify_len,
            self.growth_reserve_tokens,
        )
        capacity_limit = (
            None
            if self.request_max_tokens is None
            else (
                len(prompt_ids)
                + self.request_max_tokens
                + self.speculative_headroom
            )
        )
        prefetch_aux = _fixed_m4_materialized_prefetch
        if getattr(prepare_aux, "_submit_warm", None) is not None:
            prefetch_aux = prepare_aux.prefetch_primary
        prefetch_window_aux = _fixed_m4_materialized_window_prefetch
        if getattr(prepare_aux, "_prefetch_window_rows", None) is not None:
            prefetch_window_aux = prepare_aux.prefetch_window
        # MTPLX_FABLE_PLE_CANDIDATE_PREFETCH (default off): a bound `submit`
        # when the aux armed the lane, the no-op above otherwise.  Resolved
        # once here so the draft loop's hook is a plain call with no flag read
        # and no attribute walk per depth.
        candidate_prefetch = getattr(prepare_aux, "candidate_prefetch", None)
        submit_candidates_aux = (
            _fixed_m4_no_candidate_prefetch
            if candidate_prefetch is None
            else candidate_prefetch.submit
        )
        self._fixed_m4_dispatch = {
            "fn": fn,
            "prepare_aux": prepare_aux,
            "prefetch_aux": prefetch_aux,
            "prefetch_window_aux": prefetch_window_aux,
            "submit_candidates_aux": submit_candidates_aux,
            "state_plan": state_plan,
            "state_leaves": sum(n for _kind, _entry, n in state_plan),
            "capture_plan": tuple(capture_plan),
            "capture_leaves": capture_pos,
            "fold_entries": self._fold_entries,
            "fold_layer_indices": self._fold_layer_indices,
            "fold_dtype": self._fold_dtype,
            "fold_windows": self._fold_windows,
            "returns_aux": returns_aux,
            "aux_contract": aux_contract,
            "graph_aux": graph_aux,
            "boundary": boundary,
            "base_offset": len(prompt_ids),
            "capacity": min(entry.capacity for entry in qsa_entries),
            "growth_tokens": _next_fixed_m4_growth_tokens(
                initial_growth_tokens
            ),
            "capacity_limit": capacity_limit,
            "hidden_variant": hidden_variant,
            "qsa_entries": qsa_entries,
            "route_transition_at": (
                min(pending_route_thresholds)
                if pending_route_thresholds
                else None
            ),
            "donate": (
                _compiled_verify_donation_enabled()
                and boundary in ("both", "post")
            ),
            "device_commit": device_commit,
            "device_commit_width": getattr(
                device_commit, "commit_width", None
            ),
        }

    # -- MTPLX_FABLE_GDN_KEEPMASK_FOLD (W66b) -------------------------------

    def _fold_text_layers(self):
        """The production text model's layer list, or ``None``."""

        model = getattr(self.runtime, "model", None)
        text_model = getattr(model, "language_model", model)
        inner = getattr(text_model, "model", None)
        layers = getattr(inner, "layers", None)
        return None if layers is None else tuple(layers)

    def _resolve_gdn_keepmask_fold(self, cache: Any) -> None:
        """Arm (or leave disarmed) the keep-mask fold for this installation.

        The lane is decided ONCE per fixed-M4 installation, before the verify
        graph is traced, and every failure mode is loud:

        * flag off               -> no prefix leaves, byte-identical to today.
        * structural mismatch    -> ``GdnKeepMaskFoldContractError`` (raises).
        * non-f32 recurrent state-> raises; splitting the T loop is the
          identity only while the state round-trips through fp32 memory.
        * split/merged mismatch  -> DISABLED + logged (a property of this MLX
          build's kernel, not of this configuration).
        """

        self._fold_layer_indices = ()
        self._fold_entries = ()
        self._fold_windows = 0
        self._fold_dtype = None
        if not _gdn_fold.fable_gdn_keepmask_fold_enabled():
            return

        gdn_indices = tuple(
            int(idx) for idx, _names in self._extra_capture_layout
        )
        foldable = tuple(
            int(idx)
            for idx, names in self._extra_capture_layout
            if len(names) == 6
        )
        ple_candidates = tuple(
            int(idx)
            for idx, names in self._extra_capture_layout
            if len(names) != 6
        )
        if len(ple_candidates) != 1:
            raise _gdn_fold.GdnKeepMaskFoldContractError(
                "keep-mask fold expects exactly one PLE-carrying GDN layer; "
                f"got {len(ple_candidates)}"
            )
        layers = self._fold_text_layers()
        if layers is None:
            raise _gdn_fold.GdnKeepMaskFoldContractError(
                "keep-mask fold could not reach the text model's layer list"
            )

        entries = tuple(cache[idx] for idx in foldable)
        for idx, entry in zip(foldable, entries):
            leaf = entry.cache[1]
            if leaf is None:
                raise _gdn_fold.GdnKeepMaskFoldContractError(
                    f"{_gdn_fold.ENV_FLAG} layer {idx}: no recurrent state"
                )
            _gdn_fold.validate_state_contract(
                leaf, label=f"{_gdn_fold.ENV_FLAG} layer {idx}"
            )

        # The prefix rows are `q`/`k`/`v` straight out of the conv, so their
        # dtype is the conv weight's.  One template instantiation covers the
        # prefix and the window halves, so a mismatch would read the prefix
        # through the window's `InT` -- resolved here and asserted again on
        # every kernel call.
        first = layers[foldable[0]].linear_attn
        dtype = getattr(getattr(first, "conv1d", None), "weight", None)
        dtype = getattr(dtype, "dtype", None)
        if dtype is None:
            raise _gdn_fold.GdnKeepMaskFoldContractError(
                "keep-mask fold could not resolve the GDN row dtype"
            )

        from .kernels.gdn_keepmask_fold import default_exactness_probe

        windows = _gdn_fold.fable_gdn_keepmask_fold_windows()
        report = _gdn_fold.install_gdn_keepmask_fold(
            gdn_layer_indices=gdn_indices,
            ple_layer_index=ple_candidates[0],
            layer_modules=layers,
            exactness_probe=partial(
                default_exactness_probe, max_windows=windows
            ),
        )
        if not report.get("installed"):
            return
        self._fold_layer_indices = foldable
        self._fold_entries = entries
        self._fold_windows = int(windows)
        self._fold_dtype = dtype

    def _fold_window_build(self, dispatch) -> "FoldWindow":
        """Freeze one window's ring: bases, padded rows, mask, stamp.

        Every folded layer gets ``5`` row tensors at one fixed shape whatever
        the ring holds, so the compiled graph traces exactly once.  A ring the
        layers do not agree on (something outside the fold rebound one state
        leaf) DECLINES: every base becomes the entry's own leaf, which is the
        correct state, and the all-pad prefix makes the extra rows exact
        no-ops.  That is today's answer at today's cost, with the graph shape
        unchanged.
        """

        from .kernels.gdn_keepmask_fold import (
            empty_prefix_leaves,
            padded_prefix_leaves,
            prefix_mask_array,
        )

        entries = tuple(dispatch["fold_entries"])
        order = tuple(dispatch["fold_layer_indices"])
        if len(order) != len(entries):
            raise _gdn_fold.GdnKeepMaskFoldContractError(
                f"keep-mask fold has {len(order)} layer indices for "
                f"{len(entries)} cache entries"
            )
        windows = int(dispatch["fold_windows"])
        dtype = dispatch["fold_dtype"]
        seq = _gdn_fold.next_window_seq()

        pendings = [_gdn_fold.pending_for(entry) for entry in entries]
        rings = {() if p is None else tuple(p.keeps) for p in pendings}
        if len(rings) != 1:
            _gdn_fold.note_decline("ring_depth_disagreement")
            pendings = [None] * len(entries)
            keeps: tuple[int, ...] = ()
        else:
            keeps = rings.pop()

        rows_by_layer: dict[int, tuple[Any, ...]] = {}
        bases: dict[int, Any] = {}
        bases_by_entry: dict[int, Any] = {}
        seen: list[Any] = []
        for slot, (index, entry, pending) in enumerate(
            zip(order, entries, pendings)
        ):
            leaf = entry.cache[1]
            seen.append(leaf)
            if pending is None or not pending.keeps:
                rows = empty_prefix_leaves(
                    max_windows=windows, dtype=dtype, slot=slot
                )
                pending = _gdn_fold.FoldPending(
                    base=leaf, rows=[], keeps=(), state=leaf
                )
            else:
                rows = padded_prefix_leaves(
                    pending.rows,
                    pending.keeps,
                    max_windows=windows,
                    dtype=dtype,
                )
            rows_by_layer[int(index)] = tuple(rows)
            bases[int(index)] = pending.base
            bases_by_entry[id(entry)] = pending.base
            _gdn_fold.set_active(entry, pending, seq)

        window = FoldWindow(
            seq=seq,
            keeps=keeps,
            order=order,
            rows=rows_by_layer,
            bases=bases,
            bases_by_entry=bases_by_entry,
            mask=prefix_mask_array(keeps, max_windows=windows),
            entries=entries,
            seen=tuple(seen),
        )
        expected = _gdn_fold.prefix_leaf_count(len(entries))
        if len(window.leaves()) != expected:
            raise _gdn_fold.GdnKeepMaskFoldContractError(
                f"keep-mask fold built {len(window.leaves())} prefix leaves, "
                f"expected {expected}"
            )
        return window

    def _fold_window_open(self, dispatch) -> "FoldWindow":
        """The frozen ring for the window in flight, built on first use.

        The W67 overlap pair calls this twice per window -- once at the
        enqueue for layers ``0..N-1`` and once at the join for the rest -- and
        both must see one ring, one stamp and one mask.  The record survives a
        refused prefix (nothing committed in between, so the ring is
        unchanged) and is rebuilt the moment any folded layer's live state
        leaf moves, which is what a commit, a rollback and a published state
        output all do.
        """

        window = getattr(self, "_fold_window", None)
        if window is not None and window.is_live():
            return window
        window = self._fold_window_build(dispatch)
        self._fold_window = window
        return window

    def _fold_window_close(self) -> None:
        """Count the window and release its ring refs, once state is published.

        The COUNT lands here rather than at the build so that ``windows``
        tracks ``compiled_calls`` exactly: the overlap lane builds the ring at
        the enqueue and may then discard the prefix, and a window that never
        reached a verify must not appear in the receipt as one that did.
        """

        window = getattr(self, "_fold_window", None)
        if window is None:
            return
        self._fold_window = None
        _gdn_fold.note_window(window.depth, folded=bool(window.keeps))

    @staticmethod
    def _fold_state_in(plan, bases_by_entry: dict[int, Any]):
        """``state_in`` for one state-plan slice, with folded bases in slot 1.

        The deferred commit's lazy leaf stays on ``entry.cache[1]`` for every
        other consumer; only the graph that is about to re-derive the ring
        from ``base`` is handed the base instead.

        Keyed by ``id(entry)`` rather than by the plan position: the state
        plan skips ``None`` cache entries, so a position is the layer index
        only by convention, and a base substituted into the wrong layer's
        slot 1 would be silently wrong.
        """

        leaves: list[Any] = []
        for kind, entry, n_leaves in plan:
            if kind == VERIFY_SPEC_KIND_QSA:
                leaves.extend(
                    (
                        entry.kv.cache[0],
                        entry.kv.cache[1],
                        entry.kv.cache[2],
                        entry.raw_keys,
                        entry.pooled,
                    )
                )
                continue
            base = bases_by_entry.get(id(entry))
            if base is None:
                leaves.extend(entry.cache[:n_leaves])
            else:
                slots = list(entry.cache[:n_leaves])
                slots[1] = base
                leaves.extend(slots)
        return leaves

    def _make_fixed_m4_prefix_step(self):
        bank = self

        def prefix_step(input_ids, *state_in):
            entry = bank._shadow[0]
            for slot, leaf in enumerate(state_in):
                entry.cache[slot] = leaf
            hidden, captures = bank.runtime.forward_fixed_m4_prefix(
                input_ids,
                cache=bank._shadow,
            )
            return (hidden, *captures, *entry.cache[: len(state_in)])

        return prefix_step

    def _make_fixed_m4_suffix_step(self, dispatch):
        bank = self
        suffix_plan = dispatch["state_plan"][1:]
        suffix_capture = self._extra_capture_layout[1:]
        returns_aux = bool(dispatch["returns_aux"])
        graph_aux = dispatch["graph_aux"]

        def suffix_step(layer0_hidden, input_ids, compiled_aux, *state_in):
            pos = 0
            for (index, _spec_kind, _spec_leaves), (
                kind,
                _entry,
                n_leaves,
            ) in zip(bank._spec[1:], suffix_plan):
                shadow_entry = bank._shadow[index]
                if kind == VERIFY_SPEC_KIND_QSA:
                    shadow_entry.kv.cache[0] = state_in[pos]
                    shadow_entry.kv.cache[1] = state_in[pos + 1]
                    shadow_entry.kv.cache[2] = state_in[pos + 2]
                    shadow_entry.raw_keys = state_in[pos + 3]
                    shadow_entry.pooled = state_in[pos + 4]
                    for slot in range(len(shadow_entry.kv.rollback_state)):
                        shadow_entry.kv.rollback_state[slot] = None
                else:
                    for slot in range(n_leaves):
                        shadow_entry.cache[slot] = state_in[pos + slot]
                pos += n_leaves
            if graph_aux is not None:
                compiled_aux = graph_aux(compiled_aux)
            logits, hidden, captures = bank.runtime.forward_fixed_m4_suffix(
                layer0_hidden,
                input_ids,
                cache=bank._shadow,
                compiled_aux=compiled_aux,
            )
            captures_flat = []
            for index, names in suffix_capture:
                layer_capture = captures[index]
                captures_flat.extend(layer_capture[name] for name in names)
            state_out = []
            for (index, _kind, _spec_leaves), (
                _plan_kind,
                _entry,
                n_leaves,
            ) in zip(bank._spec[1:], suffix_plan):
                shadow_entry = bank._shadow[index]
                if _plan_kind == VERIFY_SPEC_KIND_QSA:
                    state_out.extend(shadow_entry.state_leaves)
                else:
                    state_out.extend(shadow_entry.cache[:n_leaves])
            if returns_aux:
                return (logits, hidden, compiled_aux, *captures_flat, *state_out)
            return (logits, hidden, *captures_flat, *state_out)

        return suffix_step

    def install_fixed_m4_split(self) -> None:
        """Install the PR391 layer-0 prefix and layers-1..47 suffix graphs."""

        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        layer_indices = tuple(index for index, _kind, _n in self._spec or ())
        if layer_indices != tuple(range(len(layer_indices))) or len(layer_indices) < 2:
            raise RuntimeError("fixed-M4 split requires one contiguous state plan")
        prefix_kind, prefix_entry, prefix_state_leaves = dispatch["state_plan"][0]
        if (
            prefix_kind != VERIFY_SPEC_KIND_GDN
            or prefix_state_leaves != 2
        ):
            raise RuntimeError("fixed-M4 split requires two layer-0 state leaves")
        prefix_capture_entry, prefix_start, prefix_capture_leaves = dispatch[
            "capture_plan"
        ][0]
        if (
            prefix_capture_entry is not dispatch["state_plan"][0][1]
            or prefix_start != 0
            or prefix_capture_leaves != 6
        ):
            raise RuntimeError("fixed-M4 split state or capture census changed")
        if len(layer_indices) == 48 and (
            dispatch["state_leaves"] - prefix_state_leaves != 132
            or dispatch["capture_leaves"] - prefix_capture_leaves != 213
        ):
            raise RuntimeError("fixed-M4 split production census changed")
        dispatch["split"] = {
            "prefix_fn": mx.compile(self._make_fixed_m4_prefix_step()),
            "suffix_fn": mx.compile(self._make_fixed_m4_suffix_step(dispatch)),
            "prefix_state_leaves": prefix_state_leaves,
            "prefix_capture_leaves": prefix_capture_leaves,
            "suffix_capture_leaves": dispatch["capture_leaves"]
            - prefix_capture_leaves,
        }

    def prefetch_fixed_m4_primary(
        self,
        primary,
        completion_tokens,
        committed_count: int,
    ) -> None:
        self._fixed_m4_dispatch["prefetch_aux"](
            primary,
            completion_tokens,
            committed_count,
        )

    def prefetch_fixed_m4_window(
        self,
        *,
        host_input_ids,
        completion_tokens,
        committed_count: int,
    ) -> None:
        self._fixed_m4_dispatch["prefetch_window_aux"](
            host_input_ids,
            completion_tokens,
            committed_count,
        )

    def submit_fixed_m4_candidates(
        self,
        *,
        prefix_tokens,
        candidate_ids,
        completion_tokens,
        committed_count: int,
    ) -> int:
        """Queue one window position's candidate PLE rows (K-P1).

        A no-op returning 0 unless ``MTPLX_FABLE_PLE_CANDIDATE_PREFETCH``
        armed the lane at aux construction.
        """

        return self._fixed_m4_dispatch["submit_candidates_aux"](
            prefix_tokens=prefix_tokens,
            candidate_ids=candidate_ids,
            completion_tokens=completion_tokens,
            committed_count=committed_count,
        )

    def _transition_fixed_m4_generation(
        self,
        cache: Any,
        *,
        committed_count: int,
        window: int = 4,
    ) -> None:
        """Grow or reroute one installed fixed-M4 capacity generation.

        The decision is host-owned: ``committed_count`` advances with the
        accepted completion prefix, so this boundary check never evaluates a
        device offset. Within a generation, replay stays branch-free.

        ``window`` is the number of rows the imminent forward will append. It
        is 4 for the physical-M4 verify and the compiled copy round's fixed
        physical width for a block round, which appends more rows than an M4
        window and must therefore reserve for them BEFORE dispatch.
        """

        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        required_end = (
            int(dispatch["base_offset"])
            + max(0, int(committed_count))
            + max(4, int(window))
        )
        capacity_needed = required_end > int(dispatch["capacity"])
        route_transition_at = dispatch["route_transition_at"]
        route_needed = (
            route_transition_at is not None
            and required_end >= int(route_transition_at)
        )
        if not capacity_needed and not route_needed:
            return

        qsa_entries = dispatch["qsa_entries"]
        capacity_changed = False
        next_growth_tokens = int(dispatch["growth_tokens"])
        if capacity_needed:
            next_capacity, next_growth_tokens = _fixed_m4_capacity_growth(
                capacity=int(dispatch["capacity"]),
                required_end=required_end,
                growth_tokens=int(dispatch["growth_tokens"]),
                capacity_limit=dispatch["capacity_limit"],
            )
            for entry in qsa_entries:
                capacity_changed = (
                    entry.ensure_capacity(next_capacity) or capacity_changed
                )
        route_changed = False
        if route_needed:
            for entry in qsa_entries:
                route_changed = (
                    entry.activate_rows_gather(required_end) or route_changed
                )
            pending_route_thresholds = tuple(
                entry.rows_gather_min_context
                for entry in qsa_entries
                if entry.rows_gather_enabled and not entry.fixed_rows_gather
            )
            dispatch["route_transition_at"] = (
                min(pending_route_thresholds)
                if pending_route_thresholds
                else None
            )
        if not capacity_changed and not route_changed:
            return

        self._clear_shadow_leaf_refs()
        self._held_state_refs.clear()
        self._shadow = None
        self._shadow_signature = None
        self._ensure_shadow(cache)
        route_key = int(all(entry.fixed_rows_gather for entry in qsa_entries))
        key = (
            4,
            str(dispatch["hidden_variant"] or ""),
            route_key,
            dispatch["aux_contract"],
        )
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._shared_or_new_verify_step(
                key,
                4,
                dispatch["hidden_variant"],
                graph_aux=dispatch["graph_aux"],
                return_compiled_aux=dispatch["returns_aux"],
                fold_indices=getattr(self, "_fold_layer_indices", ()),
            )
            self._compiled[key] = fn
        dispatch["fn"] = fn
        dispatch["capacity"] = min(entry.capacity for entry in qsa_entries)
        if capacity_changed:
            dispatch["growth_tokens"] = next_growth_tokens
            self.stats["fixed_m4_capacity_transitions"] += 1
        if route_changed:
            self.stats["fixed_m4_route_transitions"] += 1

    def _forward_installed_fixed_m4(
        self,
        input_ids,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache: Any,
        *,
        compiled_aux=None,
    ):
        """Run the installed monolithic physical-M4 verify graph.

        ``compiled_aux`` is W67's only concession: when the graph-build
        overlap lane already built THIS window's auxiliary at its enqueue and
        then refused its prefix, the fallback reuses that object rather than
        running ``prepare_aux`` a second time in one cycle (which would repeat
        its owned-row install and candidate resolve).  Left ``None`` -- as
        every shipped caller leaves it -- this method is unchanged.
        """

        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        self._transition_fixed_m4_generation(
            cache,
            committed_count=committed_count,
        )
        boundary = dispatch["boundary"]
        donate = dispatch["donate"]
        if donate:
            self._clear_shadow_leaf_refs()

        # MTPLX_FABLE_GDN_KEEPMASK_FOLD (W66b): the ring's rows, at one fixed
        # shape, plus the base each folded layer's recurrence must start from.
        # Resolved BEFORE `state_in` is built so slot 1 can carry the base in
        # place of the deferred commit's lazy leaf; the leaf itself stays on
        # `entry.cache[1]` for every other consumer, which forces it and gets
        # exactly today's state at exactly today's cost.
        fold_entries = dispatch.get("fold_entries") or ()
        fold_window = self._fold_window_open(dispatch) if fold_entries else None
        bases = {} if fold_window is None else fold_window.bases_by_entry

        state_in = self._fold_state_in(dispatch["state_plan"], bases)
        if fold_window is not None:
            state_in.extend(fold_window.leaves())

        if compiled_aux is None:
            compiled_aux = dispatch["prepare_aux"](
                input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
            )
        if boundary in ("both", "pre"):
            if dispatch["returns_aux"]:
                mx.async_eval(*state_in)
            else:
                mx.async_eval(compiled_aux, *state_in)
        # MTPLX_FABLE_PLE_BOUNDARY item `timing` (instrument, default off).
        # The census's gap-B host term -- the 1.64 ms/cycle the GPU idles
        # after the PLE dequant, which is this construction and not the PLE
        # -- measured from inside the process.  The constant is resolved at
        # import, so the control arm's branch is a constant False.
        if _PLE_BOUNDARY_GRAPH_TIMING:
            _graph_build_started = time.perf_counter()
            outputs = dispatch["fn"](input_ids, compiled_aux, *state_in)
            _note_ple_boundary_graph_build(
                time.perf_counter() - _graph_build_started
            )
        else:
            outputs = dispatch["fn"](input_ids, compiled_aux, *state_in)

        logits, hidden, returned_aux, captures_flat, state_out = (
            _unpack_fixed_m4_outputs(
                outputs,
                capture_leaves=dispatch["capture_leaves"],
                returns_aux=dispatch["returns_aux"],
            )
        )
        if not dispatch["returns_aux"]:
            returned_aux = compiled_aux
        else:
            self._held_aux_refs.append((compiled_aux, returned_aux))
            if len(self._held_aux_refs) > 3:
                self._held_aux_refs.pop(0)

        if not donate and boundary in ("both", "post"):
            mx.async_eval(*outputs)
            self._held_state_refs.clear()
        elif not donate:
            self._held_state_refs.append((state_in, compiled_aux))
            if len(self._held_state_refs) > 3:
                self._held_state_refs.pop(0)

        state_pos = 0
        for kind, entry, n_leaves in dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = state_out[state_pos]
                entry.kv.cache[1] = state_out[state_pos + 1]
                entry.kv.cache[2] = state_out[state_pos + 2]
                entry.raw_keys = state_out[state_pos + 3]
                entry.pooled = state_out[state_pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[state_pos + slot]
            state_pos += n_leaves

        for entry, start, count in dispatch["capture_plan"]:
            entry._mtplx_verify_rows = tuple(captures_flat[start : start + 6])
            if count > 6:
                entry._mtplx_verify_ple = tuple(
                    captures_flat[start + 6 : start + count]
                )
                entry._mtplx_verify_compiled_aux = returned_aux

        # Slot 1 now holds the graph's own post-window state, so the deferred
        # descriptor no longer owns the leaf and `pending_for` would drop it
        # anyway; clearing here releases the previous ring's base (3.1 MB a
        # layer) at the window that superseded it rather than at the next
        # commit.  `_mtplx_fold_active` is a DIFFERENT attribute and survives:
        # it is what this cycle's commit reads.
        for entry in fold_entries:
            _gdn_fold.clear_pending(entry)
        self._fold_window_close()

        if donate:
            state_in = None
            self._held_state_refs.clear()
            mx.async_eval(*outputs)

        self.stats["compiled_calls"] += 1
        self.stats["buckets"]["0"] = self.stats["buckets"].get("0", 0) + 1
        _expert_census.end_cycle()  # diagnostic: one M4 verify window closed
        return logits, hidden, {}

    @staticmethod
    def _fixed_m4_state_inputs(state_plan) -> tuple[Any, ...]:
        leaves = []
        for kind, entry, n_leaves in state_plan:
            if kind == VERIFY_SPEC_KIND_QSA:
                leaves.extend(
                    (
                        entry.kv.cache[0],
                        entry.kv.cache[1],
                        entry.kv.cache[2],
                        entry.raw_keys,
                        entry.pooled,
                    )
                )
            else:
                leaves.extend(entry.cache[:n_leaves])
        return tuple(leaves)

    def discard_fixed_m4_prefix(self, prefix: FixedM4Prefix) -> None:
        """Release exactly one abandoned split transaction without publishing it."""

        self._held_fixed_m4_split_refs[:] = [
            held
            for held in self._held_fixed_m4_split_refs
            if held is not prefix
            and not (
                isinstance(held, FixedM4Split)
                and held.prefix is prefix
            )
        ]

    def enqueue_fixed_m4_prefix(
        self,
        input_ids,
        *,
        cache,
    ) -> FixedM4Prefix:
        """Queue the fixed-M4 embedding/layer-0 graph without rebinding cache."""

        dispatch = self._fixed_m4_dispatch
        split = dispatch["split"]
        state_in = self._fixed_m4_state_inputs(dispatch["state_plan"][:1])
        outputs = tuple(split["prefix_fn"](input_ids, *state_in))
        mx.async_eval(*outputs)
        capture_end = 1 + split["prefix_capture_leaves"]
        prefix = FixedM4Prefix(
            input_ids=input_ids,
            hidden=outputs[0],
            captures=tuple(outputs[1:capture_end]),
            state_in=state_in,
            state_out=tuple(outputs[capture_end:]),
            outputs=outputs,
        )
        self._held_fixed_m4_split_refs.append(prefix)
        return prefix

    def forward_fixed_m4_suffix(
        self,
        prefix: FixedM4Prefix,
        *,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        """Join the queued layer-0 result with PLE and layers 1..47."""

        del return_hidden, hidden_variant
        try:
            dispatch = self._fixed_m4_dispatch
            split = dispatch["split"]
            self._transition_fixed_m4_generation(
                cache,
                committed_count=committed_count,
            )
            suffix_plan = dispatch["state_plan"][1:]
            state_in = self._fixed_m4_state_inputs(suffix_plan)
            compiled_aux = dispatch["prepare_aux"](
                prefix.input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
            )
            if dispatch["boundary"] in ("both", "pre"):
                if dispatch["returns_aux"]:
                    mx.async_eval(*state_in)
                else:
                    mx.async_eval(compiled_aux, *state_in)
            outputs = tuple(
                split["suffix_fn"](
                    prefix.hidden,
                    prefix.input_ids,
                    compiled_aux,
                    *state_in,
                )
            )
            logits, hidden, returned_aux, captures_flat, state_out = (
                _unpack_fixed_m4_outputs(
                    outputs,
                    capture_leaves=split["suffix_capture_leaves"],
                    returns_aux=dispatch["returns_aux"],
                )
            )
            if not dispatch["returns_aux"]:
                returned_aux = compiled_aux

            mx.async_eval(*prefix.outputs, *outputs)
            split_result = FixedM4Split(
                prefix=prefix,
                returned_aux=returned_aux,
                captures=tuple(captures_flat),
                state_in=state_in,
                state_out=tuple(state_out),
                outputs=outputs,
            )
            self._held_fixed_m4_split_refs.append(split_result)
            self.stats["calls"] += 1
            self.stats["compiled_calls"] += 1
            self.stats["buckets"]["0"] = self.stats["buckets"].get("0", 0) + 1
            _expert_census.end_cycle()  # diagnostic: split M4 window closed
            return logits, hidden, {}, split_result
        except Exception:
            self.discard_fixed_m4_prefix(prefix)
            raise

    def forward_fixed_m4(
        self,
        input_ids,
        *,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        """Run the installed physical-M4 route with host-owned n-gram inputs."""

        del return_hidden, hidden_variant
        self.stats["calls"] += 1
        return self._forward_installed_fixed_m4(
            input_ids,
            host_input_ids,
            completion_tokens,
            committed_count,
            cache,
        )

    # -- W63/W67 graph-build overlap (MTPLX_FABLE_GRAPH_BUILD_OVERLAP) -----
    #
    # The SAME partition idea as the PR391 split lane above
    # (``install_fixed_m4_split``) but its own compiled pair
    # (``install_fixed_m4_overlap_split``, depth
    # ``MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS``, default 1) and its own
    # transaction shape.  PR391's pair is left byte-identical.  The lane
    # above is a device-committed transaction that stays unpublished until
    # ``commit_fixed_m4_device_window`` selects the authoritative frontier;
    # this one is a pure SUBMISSION-TIMING change to the retained route --
    # it publishes exactly what ``_forward_installed_fixed_m4`` publishes, in
    # the same order, so the unchanged host/device commit sees an unchanged
    # cache census.  The two do not share state and neither is on the other's
    # path.

    def _fixed_m4_generation(self) -> int:
        """Capacity/route generation of the installed fixed-M4 plan."""

        return int(self.stats["fixed_m4_capacity_transitions"]) + int(
            self.stats["fixed_m4_route_transitions"]
        )

    def _fixed_m4_ple_layer_index(self) -> int:
        """Index of the single PLE layer, read off the capture layout.

        The capture layout is the runtime's own census
        (``_mtplx_capture_extra_layout``): every linear layer contributes the
        six GDN row names, and exactly one of them additionally contributes
        the three PLE names.  Deriving the index from it rather than from the
        model means a config with the PLE somewhere else partitions correctly
        and a config with more than one PLE layer RAISES here instead of
        producing a prefix that silently reads an auxiliary it was not given.
        """

        ple = tuple(
            int(index)
            for index, names in self._extra_capture_layout
            if "ple_hidden" in names
        )
        if len(ple) != 1:
            raise RuntimeError(
                "fixed-M4 overlap split requires exactly one PLE layer in the "
                f"capture layout; found {ple}"
            )
        return ple[0]

    def _make_fixed_m4_overlap_prefix_step(
        self,
        dispatch,
        *,
        layer_count: int,
        capture_len: int,
        needs_aux: bool,
        trace_host: dict[str, Any] | None = None,
        fold_indices: tuple[int, ...] = (),
    ):
        """W67: the traced closure for layers ``0..layer_count-1``.

        Structurally the suffix step's mirror image -- it seeds the shadow
        slots for its own layer range from its explicit inputs, runs the
        range, and returns ``(hidden, *captures, *state_out)`` in the layout
        the join unpacks.  ``needs_aux`` selects the arity: at depth 1 the
        prefix has no ``compiled_aux`` parameter at all, so ``mx.compile``
        traces exactly the graph W63 traced.
        """

        # ``prefix_plan``'s entries are read for `kind` and `n_leaves` only --
        # never for `_entry`, which would pin a dead request's cache -- and
        # both are part of the shared key's `spec_sig`.  Everything live
        # (`_spec`, `_shadow`, `runtime`) comes through the host.
        host = {"bank": self} if trace_host is None else trace_host
        prefix_plan = dispatch["state_plan"][:layer_count]
        prefix_capture = self._extra_capture_layout[:capture_len]
        # W66b: the folded GDN layers that live on THIS side of the split.
        # Part of the shared pair's key, so a pair traced with a prefix can
        # never be replayed by a lane that is not passing one.
        fold_indices = tuple(fold_indices)
        fold_leaf_count = _gdn_fold.prefix_leaf_count(len(fold_indices))

        def _run(input_ids, compiled_aux, state_in):
            bank = host["bank"]
            pos = 0
            for (index, _spec_kind, _spec_leaves), (
                kind,
                _entry,
                n_leaves,
            ) in zip(bank._spec[:layer_count], prefix_plan):
                shadow_entry = bank._shadow[index]
                if kind == VERIFY_SPEC_KIND_QSA:
                    shadow_entry.kv.cache[0] = state_in[pos]
                    shadow_entry.kv.cache[1] = state_in[pos + 1]
                    shadow_entry.kv.cache[2] = state_in[pos + 2]
                    shadow_entry.raw_keys = state_in[pos + 3]
                    shadow_entry.pooled = state_in[pos + 4]
                    for slot in range(len(shadow_entry.kv.rollback_state)):
                        shadow_entry.kv.rollback_state[slot] = None
                else:
                    for slot in range(n_leaves):
                        shadow_entry.cache[slot] = state_in[pos + slot]
                pos += n_leaves
            fold_scope = _overlap_fold_scope(
                bank, fold_indices, state_in, pos, fold_leaf_count
            )
            with _gdn_fold.fold_prefix_scope(fold_scope):
                hidden, captures = bank.runtime.forward_fixed_m4_overlap_prefix(
                    input_ids,
                    cache=bank._shadow,
                    layer_count=layer_count,
                    compiled_aux=compiled_aux,
                )
            _gdn_fold.assert_prefix_consumed(
                fold_scope, label="fixed-M4 overlap prefix half"
            )
            captures_flat = []
            for index, names in prefix_capture:
                layer_capture = captures[index]
                captures_flat.extend(layer_capture[name] for name in names)
            state_out = []
            for (index, _spec_kind, _spec_leaves), (
                plan_kind,
                _entry,
                n_leaves,
            ) in zip(bank._spec[:layer_count], prefix_plan):
                shadow_entry = bank._shadow[index]
                if plan_kind == VERIFY_SPEC_KIND_QSA:
                    state_out.extend(shadow_entry.state_leaves)
                else:
                    state_out.extend(shadow_entry.cache[:n_leaves])
            return (hidden, *captures_flat, *state_out)

        if needs_aux:

            def prefix_step(input_ids, compiled_aux, *state_in):
                return _run(input_ids, compiled_aux, state_in)

        else:

            def prefix_step(input_ids, *state_in):
                return _run(input_ids, None, state_in)

        return prefix_step

    def _make_fixed_m4_overlap_suffix_step(
        self,
        dispatch,
        *,
        start_layer: int,
        capture_start: int,
        trace_host: dict[str, Any] | None = None,
        fold_indices: tuple[int, ...] = (),
    ):
        """W67: the traced closure for layers ``start_layer..last`` + head."""

        host = {"bank": self} if trace_host is None else trace_host
        suffix_plan = dispatch["state_plan"][start_layer:]
        suffix_capture = self._extra_capture_layout[capture_start:]
        returns_aux = bool(dispatch["returns_aux"])
        graph_aux = dispatch["graph_aux"]
        fold_indices = tuple(fold_indices)
        fold_leaf_count = _gdn_fold.prefix_leaf_count(len(fold_indices))

        def suffix_step(prefix_hidden, input_ids, compiled_aux, *state_in):
            bank = host["bank"]
            pos = 0
            for (index, _spec_kind, _spec_leaves), (
                kind,
                _entry,
                n_leaves,
            ) in zip(bank._spec[start_layer:], suffix_plan):
                shadow_entry = bank._shadow[index]
                if kind == VERIFY_SPEC_KIND_QSA:
                    shadow_entry.kv.cache[0] = state_in[pos]
                    shadow_entry.kv.cache[1] = state_in[pos + 1]
                    shadow_entry.kv.cache[2] = state_in[pos + 2]
                    shadow_entry.raw_keys = state_in[pos + 3]
                    shadow_entry.pooled = state_in[pos + 4]
                    for slot in range(len(shadow_entry.kv.rollback_state)):
                        shadow_entry.kv.rollback_state[slot] = None
                else:
                    for slot in range(n_leaves):
                        shadow_entry.cache[slot] = state_in[pos + slot]
                pos += n_leaves
            if graph_aux is not None:
                compiled_aux = graph_aux(compiled_aux)
            fold_scope = _overlap_fold_scope(
                bank, fold_indices, state_in, pos, fold_leaf_count
            )
            with _gdn_fold.fold_prefix_scope(fold_scope):
                logits, hidden, captures = (
                    bank.runtime.forward_fixed_m4_overlap_suffix(
                        prefix_hidden,
                        input_ids,
                        cache=bank._shadow,
                        compiled_aux=compiled_aux,
                        start=start_layer,
                    )
                )
            _gdn_fold.assert_prefix_consumed(
                fold_scope, label="fixed-M4 overlap suffix half"
            )
            captures_flat = []
            for index, names in suffix_capture:
                layer_capture = captures[index]
                captures_flat.extend(layer_capture[name] for name in names)
            state_out = []
            for (index, _spec_kind, _spec_leaves), (
                plan_kind,
                _entry,
                n_leaves,
            ) in zip(bank._spec[start_layer:], suffix_plan):
                shadow_entry = bank._shadow[index]
                if plan_kind == VERIFY_SPEC_KIND_QSA:
                    state_out.extend(shadow_entry.state_leaves)
                else:
                    state_out.extend(shadow_entry.cache[:n_leaves])
            if returns_aux:
                return (logits, hidden, compiled_aux, *captures_flat, *state_out)
            return (logits, hidden, *captures_flat, *state_out)

        return suffix_step

    def install_fixed_m4_overlap_split(self, layer_count: int) -> None:
        """W67: compile the ``0..N-1`` / ``N..last`` pair for THIS lane.

        Separate from ``install_fixed_m4_split``, which is PR391's and whose
        hard-coded layer-0 census ``tests/test_qwen4_fixed_host_tokens_static``
        pins by source.  Every check below RAISES: it runs at the request
        boundary (``arm_fixed_m4_graph_build_overlap``), where an arm that
        cannot honour its own flag must fail loudly rather than quietly run
        the control while wearing the candidate's label.
        """

        _started = time.perf_counter()
        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        spec = tuple(self._spec or ())
        layer_indices = tuple(index for index, _kind, _n in spec)
        if layer_indices != tuple(range(len(layer_indices))) or (
            len(layer_indices) < 2
        ):
            raise RuntimeError(
                "fixed-M4 overlap split requires one contiguous state plan"
            )
        if len(dispatch["state_plan"]) != len(layer_indices):
            raise RuntimeError(
                "fixed-M4 overlap split requires one state-plan entry per layer"
            )
        count = int(layer_count)
        if not 1 <= count < len(layer_indices):
            raise RuntimeError(
                f"{_graph_build_overlap.LAYERS_ENV}={count} is outside "
                f"[1, {len(layer_indices) - 1}] for this "
                f"{len(layer_indices)}-layer verify plan"
            )
        if len(layer_indices) == 48 and (
            dispatch["state_leaves"] != 134
            or dispatch["capture_leaves"] != 219
        ):
            # 35*2 GDN + 1*4 PLE-GDN + 12*5 QSA = 134; 36*6 + 3 = 219.  The
            # partition arithmetic below is exact for any census, but a drift
            # here means the geometry this lane was priced on is gone.
            raise RuntimeError(
                "fixed-M4 overlap split production census changed: "
                f"{dispatch['state_leaves']} state / "
                f"{dispatch['capture_leaves']} capture leaves (want 134 / 219)"
            )
        ple_index = self._fixed_m4_ple_layer_index()
        needs_aux = ple_index < count
        if needs_aux and dispatch["returns_aux"]:
            # The raw-q4 contract dequantizes the auxiliary INSIDE the graph
            # that consumes it and returns the expanded array as the census's
            # `_mtplx_verify_compiled_aux`.  With the PLE layer in the prefix
            # that dequantization and that output would have to move to the
            # prefix, which is a second aux contract this lane has never run.
            raise RuntimeError(
                f"{_graph_build_overlap.LAYERS_ENV}={count} puts the PLE layer "
                f"(index {ple_index}) in the prefix, which requires the "
                "materialized auxiliary contract; this request is on "
                f"{dispatch['aux_contract']!r}"
            )
        capture_start = sum(
            1 for index, _names in self._extra_capture_layout if index < count
        )
        prefix_state_leaves = sum(
            n for _kind, _entry, n in dispatch["state_plan"][:count]
        )
        prefix_capture_leaves = sum(
            len(names)
            for _index, names in self._extra_capture_layout[:capture_start]
        )
        if not 0 < prefix_state_leaves < int(dispatch["state_leaves"]):
            raise RuntimeError(
                "fixed-M4 overlap split leaves no state on one side: "
                f"{prefix_state_leaves} of {dispatch['state_leaves']}"
            )
        if not 0 < prefix_capture_leaves < int(dispatch["capture_leaves"]):
            raise RuntimeError(
                "fixed-M4 overlap split leaves no capture rows on one side: "
                f"{prefix_capture_leaves} of {dispatch['capture_leaves']}"
            )
        # W66b: the keep-mask fold's 35 folded GDN layers partition on the
        # SAME boundary as the state and capture plans.  Whichever half owns a
        # layer carries that layer's five padded row tensors, and each half
        # that owns any carries its own copy of the shared mask.  The PLE
        # layer is not folded, so at N >= 2 it sits in the prefix without
        # contributing a leaf.
        fold_indices = tuple(dispatch.get("fold_layer_indices") or ())
        prefix_fold = tuple(index for index in fold_indices if index < count)
        suffix_fold = tuple(index for index in fold_indices if index >= count)
        if len(prefix_fold) + len(suffix_fold) != len(fold_indices):
            raise RuntimeError(
                "fixed-M4 overlap split lost a folded GDN layer across the "
                f"boundary: {len(prefix_fold)} + {len(suffix_fold)} != "
                f"{len(fold_indices)}"
            )
        prefix_fn, suffix_fn = self._shared_or_new_overlap_split(
            dispatch,
            layer_count=count,
            capture_start=capture_start,
            needs_aux=needs_aux,
            prefix_fold=prefix_fold,
            suffix_fold=suffix_fold,
        )
        dispatch["overlap_split"] = {
            "prefix_fn": prefix_fn,
            "suffix_fn": suffix_fn,
            "layer_count": count,
            "prefix_fold_layers": prefix_fold,
            "suffix_fold_layers": suffix_fold,
            "prefix_fold_leaves": _gdn_fold.prefix_leaf_count(
                len(prefix_fold)
            ),
            "suffix_fold_leaves": _gdn_fold.prefix_leaf_count(
                len(suffix_fold)
            ),
            "needs_aux": needs_aux,
            "ple_layer_index": ple_index,
            "prefix_plan_len": count,
            "prefix_capture_plan_len": capture_start,
            "prefix_state_leaves": prefix_state_leaves,
            "prefix_capture_leaves": prefix_capture_leaves,
            "suffix_capture_leaves": int(dispatch["capture_leaves"])
            - prefix_capture_leaves,
        }
        _graph_build_overlap.note_prefix_layers(count)
        _graph_build_overlap.note_construction(time.perf_counter() - _started)
        if fold_indices:
            _gdn_fold.note_overlap_split(
                layer_count=count,
                prefix_layers=len(prefix_fold),
                suffix_layers=len(suffix_fold),
            )

    def _shared_or_new_overlap_split(
        self,
        dispatch,
        *,
        layer_count: int,
        capture_start: int,
        needs_aux: bool,
        prefix_fold: tuple[int, ...] = (),
        suffix_fold: tuple[int, ...] = (),
    ):
        """Reuse one compiled overlap pair per process for a logical key.

        Mirrors ``_shared_or_new_verify_step`` exactly, and for the same
        reason: the bank is per-generation, so building fresh closures here
        would make ``mx.compile`` re-trace both graphs on the first cycle of
        EVERY request, where the shipped monolithic route traces once per
        process.  In a one-request-per-process A/B that costs nothing and is
        invisible; in a served process it is two full re-traces per request.

        The key carries everything the traced pair depends on that is not
        already inside ``spec_sig``: the prefix depth, whether the prefix
        takes the auxiliary, the QSA gather route, the aux contract and the
        exact-verify kernel route.  Leaf SHAPE changes (capacity growth) are
        ``mx.compile``'s own retrace dimension, as on the monolithic path.
        """

        if not _env_enabled("MTPLX_COMPILED_VERIFY_SHARED_TRACES", default=True):
            return (
                mx.compile(
                    self._make_fixed_m4_overlap_prefix_step(
                        dispatch,
                        layer_count=layer_count,
                        capture_len=capture_start,
                        needs_aux=needs_aux,
                        fold_indices=prefix_fold,
                    )
                ),
                mx.compile(
                    self._make_fixed_m4_overlap_suffix_step(
                        dispatch,
                        start_layer=layer_count,
                        capture_start=capture_start,
                        fold_indices=suffix_fold,
                    )
                ),
            )
        from .attention_context import exact_verify_required

        qsa_entries = dispatch["qsa_entries"]
        global_key = (
            id(self.runtime),
            self.capture_backend,
            self._capture_layout_override,
            self._extra_capture_layout,
            self._prepare_compiled_aux is not None,
            tuple(self._spec or []),
            int(layer_count),
            bool(needs_aux),
            str(dispatch["hidden_variant"] or ""),
            int(all(entry.fixed_rows_gather for entry in qsa_entries)),
            str(dispatch["aux_contract"]),
            bool(exact_verify_required()),
            # W66b keep-mask fold dimension.  BOTH halves' partitions, because
            # a pair traced with a prefix has a different input arity and a
            # different recurrence on each side from one traced without, and
            # the boundary decides which side owns which layer.
            tuple(prefix_fold),
            tuple(suffix_fold),
        )
        entry = _SHARED_OVERLAP_SPLITS.get(global_key)
        if entry is not None:
            prefix_fn, suffix_fn, host, runtime_ref = entry
            # id() can be recycled after a model swap frees the old runtime;
            # a stale pair would replay graphs bound to freed weights.
            if runtime_ref() is self.runtime:
                host["bank"] = self
                _graph_build_overlap.bump("split_shared_hits")
                return prefix_fn, suffix_fn
            _SHARED_OVERLAP_SPLITS.pop(global_key, None)
        host = {"bank": self}
        prefix_fn = mx.compile(
            self._make_fixed_m4_overlap_prefix_step(
                dispatch,
                layer_count=layer_count,
                capture_len=capture_start,
                needs_aux=needs_aux,
                trace_host=host,
                fold_indices=prefix_fold,
            )
        )
        suffix_fn = mx.compile(
            self._make_fixed_m4_overlap_suffix_step(
                dispatch,
                start_layer=layer_count,
                capture_start=capture_start,
                trace_host=host,
                fold_indices=suffix_fold,
            )
        )
        _SHARED_OVERLAP_SPLITS[global_key] = (
            prefix_fn,
            suffix_fn,
            host,
            weakref.ref(self.runtime),
        )
        return prefix_fn, suffix_fn

    def _refresh_fixed_m4_split(self) -> int:
        """Recompile the overlap pair when the plan changed generation.

        ``_transition_fixed_m4_generation`` rebuilds the shadow and recompiles
        ``dispatch["fn"]`` on a capacity or route transition, but knows nothing
        about ``dispatch["overlap_split"]``.  Recompiling here keeps the pair on
        the same generation as the monolithic graph without editing the shared
        transition (whose source ``tests/test_qwen4_fixed_host_tokens_static``
        pins for the PR391 lane).
        """

        generation = self._fixed_m4_generation()
        if generation != self._fixed_m4_split_generation:
            self.install_fixed_m4_overlap_split(self._fixed_m4_overlap_layers)
            self._fixed_m4_split_generation = generation
            _graph_build_overlap.bump("split_rebuilds")
        return generation

    def arm_fixed_m4_graph_build_overlap(
        self, layer_count: int | None = None
    ) -> int:
        """Compile the ``0..N-1`` / ``N..last`` pair once, at request setup.

        Called only when ``MTPLX_FABLE_GRAPH_BUILD_OVERLAP`` is armed.  Doing
        it here rather than lazily on the first cycle means an unsupported
        state/capture census, an out-of-range depth or an aux contract the
        prefix cannot carry raises at the request boundary, where the arm is
        readable, instead of mid-window.  Returns the installed depth.
        """

        if self._fixed_m4_dispatch is None:
            raise RuntimeError(
                "graph-build overlap requires an installed fixed-M4 verify"
            )
        requested = int(
            _graph_build_overlap.layers() if layer_count is None else layer_count
        )
        if requested != self._fixed_m4_overlap_layers:
            # A bank reused across requests (or a bench sweeping depths in one
            # process) must retrace the pair; one that is not asked for a new
            # depth must NOT, because `mx.compile` would retrace ~5,200 nodes
            # for nothing.
            self._fixed_m4_overlap_layers = requested
            self._fixed_m4_split_generation = -1
        self._refresh_fixed_m4_split()
        return int(self._fixed_m4_overlap_layers)

    def discard_fixed_m4_overlap_prefix(self) -> None:
        """Drop an unjoined layer-0 prefix.  Idempotent."""

        if self._fixed_m4_overlap_prefix is None:
            return
        self._fixed_m4_overlap_prefix = None
        _graph_build_overlap.bump("prefix_discarded")

    def enqueue_fixed_m4_overlap_prefix(
        self,
        input_ids,
        *,
        committed_count: int,
        cache,
        host_input_ids=None,
        completion_tokens=None,
    ) -> FixedM4OverlapPrefix:
        """Queue target embedding + layers ``0..N-1`` ahead of the window.

        ``input_ids`` is the SAME ``[1,4]`` array the monolithic route is
        handed, passed at the earliest statement that owns it -- ahead of the
        ~1.9 ms/cycle of suffix replay the retained-stack census measures the
        GPU idling through (382/382 cycles, 86.9 % host-late).  Nothing here
        reads it on the host, so an unevaluated array is fine: ``mx.compile``'s
        replay substitutes inputs into the traced graph by shape and dtype and
        never evaluates them.

        **The W67 hoist.**  At depth 1 the prefix reads no PLE auxiliary and
        this method never touches ``prepare_aux`` -- the join builds it where
        the shipped route builds it.  At depth > 1 the prefix CONTAINS the PLE
        layer, so the auxiliary is built HERE, from ``host_input_ids`` /
        ``completion_tokens`` / ``committed_count``, none of which any layer
        produces: the drafted token VALUES are all it needs and they arrived
        with the window.  It is built exactly once per window either way, and
        carried on the returned prefix so that even a window whose prefix the
        join refuses reuses it instead of paying for a second one.
        """

        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        # A prefix still in the slot belonged to a window that never reached
        # the verify.  Count it rather than overwriting it silently: the
        # receipt's `prefix_discarded` is how a reader learns the lane is
        # computing prefix forwards it throws away.
        self.discard_fixed_m4_overlap_prefix()
        self._transition_fixed_m4_generation(
            cache,
            committed_count=committed_count,
        )
        generation = self._refresh_fixed_m4_split()
        split = dispatch["overlap_split"]
        layer_count = int(split["layer_count"])
        needs_aux = bool(split["needs_aux"])
        if dispatch["donate"]:
            # Once per cycle, here rather than in the join: the traced prefix
            # re-seeds its own shadow slots from its explicit inputs, and the
            # suffix does the same for the rest, so clearing before the FIRST
            # of the two submissions is what the monolithic route does before
            # its single one.
            self._clear_shadow_leaf_refs()
        # W66b: freeze this window's ring HERE, before the first of the two
        # submissions.  The join reuses the same record, so both halves of the
        # split run the same recurrence from the same bases under the same
        # mask; a refused prefix leaves it live (nothing committed in between)
        # and the monolithic fallback reuses it too.
        fold_window = (
            self._fold_window_open(dispatch)
            if dispatch.get("fold_entries")
            else None
        )
        state_in = self._fold_state_in(
            dispatch["state_plan"][:layer_count],
            {} if fold_window is None else fold_window.bases_by_entry,
        )
        if fold_window is not None:
            state_in.extend(fold_window.leaves(0, layer_count))
        state_in = tuple(state_in)
        compiled_aux = None
        if needs_aux:
            if host_input_ids is None:
                raise RuntimeError(
                    "graph-build overlap past the PLE layer needs the "
                    "window's host token ids at the enqueue"
                )
            compiled_aux = dispatch["prepare_aux"](
                input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
            )
            _graph_build_overlap.note_aux_hoisted()
            if dispatch["boundary"] in ("both", "pre"):
                # ``_prepare_compiled_verify_aux``'s contract: the auxiliary
                # must cross the materialization boundary before it becomes an
                # mx.compile input.  With the PLE layer in the prefix, the
                # FIRST graph to consume it is the prefix, so the submission
                # moves here with it.  No ``returns_aux`` branch: the install
                # refuses that contract at any depth past the PLE layer, so
                # the shipped route's raw-payload spelling cannot be reached.
                mx.async_eval(compiled_aux, *state_in)
            outputs = self._replay_overlap_prefix(
                split["prefix_fn"], input_ids, compiled_aux, state_in
            )
        else:
            outputs = self._replay_overlap_prefix(
                split["prefix_fn"], input_ids, None, state_in
            )
        capture_end = 1 + split["prefix_capture_leaves"]
        prefix = FixedM4OverlapPrefix(
            input_ids=input_ids,
            hidden=outputs[0],
            captures=tuple(outputs[1:capture_end]),
            state_out=tuple(outputs[capture_end:]),
            outputs=outputs,
            generation=generation,
            committed_count=int(committed_count),
            compiled_aux=compiled_aux,
            layer_count=layer_count,
        )
        state_in = None
        # NOT rebinding the prefix layers' live cache slots here, unlike the
        # monolithic route's pre-``async_eval`` commit: a window that falls
        # back after this point must find those layers' PRE-verify state on
        # the live cache.  They are a few MB, and `before_verify`'s snapshot
        # pins them for the whole cycle anyway, so no donation is lost by
        # waiting.
        mx.async_eval(*outputs)
        self._fixed_m4_overlap_prefix = prefix
        _graph_build_overlap.bump("prefix_enqueued")
        return prefix

    def _replay_overlap_prefix(self, prefix_fn, input_ids, compiled_aux, state_in):
        """Replay the prefix graph, timed on the first call and under `timing`.

        The two arities are the point: at depth 1 the traced closure has no
        ``compiled_aux`` parameter at all, so passing ``None`` positionally
        would change its signature and its trace.
        """

        first = self._fixed_m4_overlap_first_prefix
        if _GRAPH_BUILD_OVERLAP_TIMING or first:
            _started = time.perf_counter()
            if compiled_aux is None:
                outputs = tuple(prefix_fn(input_ids, *state_in))
            else:
                outputs = tuple(prefix_fn(input_ids, compiled_aux, *state_in))
            _elapsed = time.perf_counter() - _started
            if first:
                self._fixed_m4_overlap_first_prefix = False
                _graph_build_overlap.note_first_prefix_build(_elapsed)
            if _GRAPH_BUILD_OVERLAP_TIMING:
                _graph_build_overlap.note_prefix_build(_elapsed)
            return outputs
        if compiled_aux is None:
            return tuple(prefix_fn(input_ids, *state_in))
        return tuple(prefix_fn(input_ids, compiled_aux, *state_in))

    def forward_fixed_m4_overlap(
        self,
        input_ids,
        *,
        host_input_ids,
        completion_tokens,
        committed_count: int,
        cache,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        """Join a queued ``0..N-1`` prefix with ``N..last`` and the head.

        Falls back to the shipped monolithic route -- same call, same graph --
        whenever no usable prefix is queued, so the lane can never run a stale
        prefix against a regrown capacity generation.
        """

        del return_hidden, hidden_variant
        self.stats["calls"] += 1
        dispatch = self._fixed_m4_dispatch
        assert dispatch is not None
        self._transition_fixed_m4_generation(
            cache,
            committed_count=committed_count,
        )
        prefix = self._fixed_m4_overlap_prefix
        self._fixed_m4_overlap_prefix = None
        if (
            prefix is None
            or prefix.generation != self._fixed_m4_generation()
            or prefix.committed_count != int(committed_count)
        ):
            if prefix is not None:
                _graph_build_overlap.bump("prefix_discarded")
            _graph_build_overlap.bump("monolithic_windows")
            return self._forward_installed_fixed_m4(
                input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
                cache,
                # W67: a refused prefix may already have paid for this
                # window's auxiliary (the hoist).  Reuse it -- building a
                # second one would run `prepare_aux`'s owned-row install and
                # candidate resolve twice in one cycle.
                compiled_aux=None if prefix is None else prefix.compiled_aux,
            )
        split = dispatch["overlap_split"]
        boundary = dispatch["boundary"]
        donate = dispatch["donate"]
        layer_count = int(split["layer_count"])

        suffix_plan = dispatch["state_plan"][layer_count:]
        # W66b: the SAME record the enqueue froze -- same ring, same stamp,
        # same mask -- now for the layers this half owns.
        fold_window = (
            self._fold_window_open(dispatch)
            if dispatch.get("fold_entries")
            else None
        )
        state_in = self._fold_state_in(
            suffix_plan,
            {} if fold_window is None else fold_window.bases_by_entry,
        )
        if fold_window is not None:
            state_in.extend(fold_window.leaves(layer_count, None))

        if bool(split["needs_aux"]) is not (prefix.compiled_aux is not None):
            # The prefix graph either consumes the auxiliary or it does not,
            # and the enqueue and the join must agree about which -- a prefix
            # that reached the PLE layer without one would have raised in the
            # runtime forward, and one carrying an auxiliary the join then
            # rebuilds would run `prepare_aux`'s side effects twice.
            raise RuntimeError(
                "graph-build overlap prefix/join auxiliary contract disagree: "
                f"needs_aux={split['needs_aux']!r}, "
                f"carried={prefix.compiled_aux is not None}"
            )
        if prefix.compiled_aux is None:
            compiled_aux = dispatch["prepare_aux"](
                prefix.input_ids,
                host_input_ids,
                completion_tokens,
                committed_count,
            )
            if boundary in ("both", "pre"):
                if dispatch["returns_aux"]:
                    mx.async_eval(*state_in)
                else:
                    mx.async_eval(compiled_aux, *state_in)
        else:
            # Hoisted: the enqueue built it AND (under the same boundary
            # condition) already put it across the materialization boundary,
            # because the prefix graph consumed it.  Only the suffix's own
            # state leaves are left to root here.
            compiled_aux = prefix.compiled_aux
            if boundary in ("both", "pre"):
                mx.async_eval(*state_in)
        _first_suffix = self._fixed_m4_overlap_first_suffix
        if _GRAPH_BUILD_OVERLAP_TIMING or _first_suffix:
            _started = time.perf_counter()
            outputs = tuple(
                split["suffix_fn"](
                    prefix.hidden,
                    prefix.input_ids,
                    compiled_aux,
                    *state_in,
                )
            )
            _elapsed = time.perf_counter() - _started
            if _first_suffix:
                self._fixed_m4_overlap_first_suffix = False
                _graph_build_overlap.note_first_suffix_build(_elapsed)
            if _GRAPH_BUILD_OVERLAP_TIMING:
                _graph_build_overlap.note_suffix_build(_elapsed)
        else:
            outputs = tuple(
                split["suffix_fn"](
                    prefix.hidden,
                    prefix.input_ids,
                    compiled_aux,
                    *state_in,
                )
            )

        logits, hidden, returned_aux, captures_flat, state_out = (
            _unpack_fixed_m4_outputs(
                outputs,
                capture_leaves=split["suffix_capture_leaves"],
                returns_aux=dispatch["returns_aux"],
            )
        )
        if not dispatch["returns_aux"]:
            returned_aux = compiled_aux
        else:
            self._held_aux_refs.append((compiled_aux, returned_aux))
            if len(self._held_aux_refs) > 3:
                self._held_aux_refs.pop(0)

        if not donate and boundary in ("both", "post"):
            mx.async_eval(*outputs)
            self._held_state_refs.clear()
        elif not donate:
            self._held_state_refs.append((state_in, compiled_aux))
            if len(self._held_state_refs) > 3:
                self._held_state_refs.pop(0)

        # Layers 0..N-1's leaves come from the prefix, N..last's from the
        # suffix; together they are byte-for-byte the census
        # ``_forward_installed_fixed_m4`` publishes from its single output
        # tuple, in the same order.
        prefix_pos = 0
        for kind, entry, n_leaves in dispatch["state_plan"][:layer_count]:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = prefix.state_out[prefix_pos]
                entry.kv.cache[1] = prefix.state_out[prefix_pos + 1]
                entry.kv.cache[2] = prefix.state_out[prefix_pos + 2]
                entry.raw_keys = prefix.state_out[prefix_pos + 3]
                entry.pooled = prefix.state_out[prefix_pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = prefix.state_out[prefix_pos + slot]
            prefix_pos += n_leaves
        state_pos = 0
        for kind, entry, n_leaves in suffix_plan:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = state_out[state_pos]
                entry.kv.cache[1] = state_out[state_pos + 1]
                entry.kv.cache[2] = state_out[state_pos + 2]
                entry.raw_keys = state_out[state_pos + 3]
                entry.pooled = state_out[state_pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[state_pos + slot]
            state_pos += n_leaves

        capture_shift = split["prefix_capture_leaves"]
        prefix_plan_len = int(split["prefix_capture_plan_len"])
        for plan_index, (entry, start, count) in enumerate(
            dispatch["capture_plan"]
        ):
            # ``install_fixed_m4_overlap_split`` splits the capture PLAN on
            # the same layer boundary it splits the state plan, and the plan's
            # offsets are cumulative over the whole layout, so `start` indexes
            # the prefix's flat tuple directly below the shift and the
            # suffix's above it.  Neither side is a guess about which entry
            # owns which rows.
            if plan_index < prefix_plan_len:
                rows = prefix.captures
                offset = start
            else:
                rows = captures_flat
                offset = start - capture_shift
            entry._mtplx_verify_rows = tuple(rows[offset : offset + 6])
            if count > 6:
                entry._mtplx_verify_ple = tuple(
                    rows[offset + 6 : offset + count]
                )
                entry._mtplx_verify_compiled_aux = returned_aux

        # Both halves' state is published, so the deferred descriptors no
        # longer own their leaves and the frozen ring is spent.  Exactly what
        # `_forward_installed_fixed_m4` does at the same point.
        for entry in dispatch.get("fold_entries") or ():
            _gdn_fold.clear_pending(entry)
        self._fold_window_close()

        if donate:
            state_in = None
            self._held_state_refs.clear()
            mx.async_eval(*outputs)

        self.stats["compiled_calls"] += 1
        self.stats["buckets"]["0"] = self.stats["buckets"].get("0", 0) + 1
        _graph_build_overlap.bump("suffix_joined")
        _expert_census.end_cycle()  # diagnostic: one M4 verify window closed
        return logits, hidden, {}

    # -- compiled context-copy block round ---------------------------------

    def install_copy_round(
        self,
        cache: Any,
        *,
        width: int,
        hidden_variant: str | None,
    ) -> None:
        """Install one fixed-width compiled replay for context-copy rounds.

        WHY THIS EXISTS.  ``mtplx/context_copy.py`` block rounds forward
        ``[primary, *block]`` through the target once per round; on the batched
        (qwen4_exp / Flash-Next) lane that call is
        ``rt.forward_ar(...)`` — the EAGER model path — while the fixed-M4
        verify next to it replays a compiled graph.  Two independent reasons,
        both removed here:

        1.  The batched lane never routed the block round through this bank at
            all.  The capture lane has an opt-in bank route
            (``MTPLX_CCOPY_BANK_ROUTE`` + ``extended_window=True``); the
            batched lane's block simply called the runtime.
        2.  Even routed, it would not have compiled: ``forward_ar_capture``
            short-circuits at its top whenever ``_fixed_m4_dispatch`` is
            installed — which is exactly this lane's state — and returns
            ``_runtime_forward``.  The installed physical-M4 replay makes the
            generic compiled dispatcher unreachable for every length but 4.

        The receipts show it: ``compiled_verify.calls == compiled_calls == 382``
        against ``verify_calls == 392`` with ``extended_calls == 0`` — the ten
        missing calls are the ten copy rounds, all eager.

        WIDTH IS FIXED AND THE BLOCK IS PADDED TO IT.  ``width`` is one physical
        row count (the block cap K plus the primary), and every round runs at
        that width regardless of the ladder length it proposed: the caller pads
        the proposal's rows and keeps its acceptance loop over the LOGICAL
        block only.  This is exact, not approximate, and for the same reason a
        physical-M4 window that accepts one token is exact:

          * the forward is causal, so rows 0..L-1 (the proposed block) cannot
            be influenced by the pad rows that follow them — same logits, same
            acceptance draws, same emitted stream;
          * ``commit_verified_window`` replays each GDN recurrence over
            ``rows[:, :keep_tokens]`` and restages PLE from
            ``ids[:, :keep_tokens]``, so the committed state reads only the
            accepted prefix, and trimmable (QSA) entries trim
            ``verified_tokens - keep_tokens`` rows — which is why the caller
            MUST pass the PADDED width as ``verified_tokens``, and why the
            "no trim needed on a full accept" shortcut on the non-family
            commit path stops being valid once rows are padded.

        The alternative — a small bank of graphs, one per ladder length
        (8/12/16/24/32 → widths 9/13/17/25/33) — was rejected: five extra
        traces cost five sets of graph buffers against an 87.4 GB peak under a
        90 GB wired limit, and the whole point of the ladder (spend less on a
        weak match) is worth little once a row is compiled-cheap.  One width,
        one trace.

        Construction-time eligibility only: everything that could refuse is
        checked HERE and raises.  There is no per-call fallback, so a round
        never silently reverts to the eager forward mid-generation and the
        A/B arm cannot be half-compiled.
        """

        dispatch = self._fixed_m4_dispatch
        if dispatch is None:
            raise RuntimeError(
                "compiled copy round requires an installed physical-M4 verifier"
            )
        if self.parity or self.parity2:
            raise ValueError("compiled copy round is disabled in parity modes")
        width = int(width)
        if width < 2:
            raise ValueError(f"compiled copy round width must be >= 2, got {width}")
        if width == 4:
            # (4,...) is the fixed-M4 verifier's own compiled key; a copy round
            # sharing it would inherit the raw-q4 sidecar aux contract.
            raise ValueError("compiled copy round width must differ from the M4 window")
        if self._prepare_compiled_aux is None:
            raise RuntimeError(
                "compiled copy round requires the length-generic PLE auxiliary"
            )
        if not self._extra_capture_layout:
            raise RuntimeError(
                "compiled copy round requires the family capture layout"
            )
        capacity_limit = dispatch["capacity_limit"]
        if capacity_limit is not None and (
            int(dispatch["base_offset"]) + width > int(capacity_limit)
        ):
            raise RuntimeError(
                f"compiled copy round width {width} exceeds the request's "
                f"KV capacity limit {capacity_limit}"
            )
        self._copy_round_dispatch = {
            "width": width,
            "hidden_variant": hidden_variant,
            "state_plan": dispatch["state_plan"],
            "capture_plan": dispatch["capture_plan"],
            "capture_leaves": dispatch["capture_leaves"],
            "qsa_entries": dispatch["qsa_entries"],
            "boundary": _compiled_verify_boundary(),
        }
        self.stats["copy_round_width"] = width
        # Reserve for the wider window NOW.  Without this the first copy round
        # of a generation trips the capacity transition, which rebuilds the
        # shadow and re-keys the compiled callables -- a one-off spike landing
        # inside a measured decode window instead of in setup.
        self._transition_fixed_m4_generation(
            cache, committed_count=0, window=width
        )

    def _copy_round_step(self, hidden_variant: str | None):
        """Resolve (compiling on first use) this width's verify callable.

        Re-resolved per call rather than pinned at install: a capacity or
        rows-gather transition re-keys the fixed-M4 callable the same way, and
        a stale copy-round callable would replay a graph bound to the previous
        generation's route.  The lookup is a dict hit on the steady state.
        """

        dispatch = self._copy_round_dispatch
        assert dispatch is not None
        route_key = int(
            all(entry.fixed_rows_gather for entry in dispatch["qsa_entries"])
        )
        key = (int(dispatch["width"]), str(hidden_variant or ""), route_key, "materialized")
        fn = self._compiled.get(key)
        if fn is None:
            fn = self._shared_or_new_verify_step(
                key, int(dispatch["width"]), hidden_variant
            )
            self._compiled[key] = fn
        return fn

    def forward_copy_round(
        self,
        input_ids,
        *,
        cache,
        committed_count: int,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
    ):
        """Replay one context-copy block round through the compiled graph.

        ``input_ids`` must already be padded to the installed width; the
        caller owns the padding and the logical/physical split (see
        ``install_copy_round``).  Returns ``(logits, hidden)`` shaped to the
        PADDED width -- the caller slices to its logical rows.
        """

        dispatch = self._copy_round_dispatch
        if dispatch is None:
            raise RuntimeError("compiled copy round is not installed")
        if not return_hidden:
            raise ValueError("compiled copy round always returns hidden states")
        width = int(dispatch["width"])
        length = _decode_length(input_ids)
        if length != width:
            raise ValueError(
                f"compiled copy round expects {width} padded rows, got {length}"
            )
        if hidden_variant is None:
            hidden_variant = dispatch["hidden_variant"]
        elif hidden_variant != dispatch["hidden_variant"]:
            raise ValueError(
                "compiled copy round hidden variant changed after installation"
            )

        self.stats["calls"] += 1
        # Reserve for the FULL padded window before dispatch: a block round
        # appends `width` rows, not 4.
        self._transition_fixed_m4_generation(
            cache,
            committed_count=committed_count,
            window=width,
        )
        fn = self._copy_round_step(hidden_variant)
        boundary = dispatch["boundary"]
        donate = (
            _compiled_verify_donation_enabled() and boundary in ("both", "post")
        )
        if donate:
            self._clear_shadow_leaf_refs()

        state_in = self._fixed_m4_state_inputs(dispatch["state_plan"])
        compiled_aux = self._prepare_compiled_aux(input_ids, cache)
        if boundary in ("both", "pre"):
            mx.async_eval(compiled_aux, *state_in)
        outputs = fn(input_ids, compiled_aux, *state_in)
        logits, hidden, _returned_aux, captures_flat, state_out = (
            _unpack_fixed_m4_outputs(
                outputs,
                capture_leaves=dispatch["capture_leaves"],
                returns_aux=False,
            )
        )
        if not donate and boundary in ("both", "post"):
            mx.async_eval(*outputs)
            self._held_state_refs.clear()
        elif not donate:
            # Same 3-generation input hold the generic dispatcher uses: a
            # pending graph must keep its input buffers alive.
            self._held_state_refs.append((state_in, compiled_aux))
            if len(self._held_state_refs) > 3:
                self._held_state_refs.pop(0)

        state_pos = 0
        for kind, entry, n_leaves in dispatch["state_plan"]:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = state_out[state_pos]
                entry.kv.cache[1] = state_out[state_pos + 1]
                entry.kv.cache[2] = state_out[state_pos + 2]
                entry.raw_keys = state_out[state_pos + 3]
                entry.pooled = state_out[state_pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[state_pos + slot]
            state_pos += n_leaves

        # The capture rows are what commit_verified_window replays from, so
        # they are published exactly as the fixed-M4 route publishes them --
        # at the PADDED width, which is the width the commit validates.
        for entry, start, count in dispatch["capture_plan"]:
            entry._mtplx_verify_rows = tuple(captures_flat[start : start + 6])
            if count > 6:
                entry._mtplx_verify_ple = tuple(
                    captures_flat[start + 6 : start + count]
                )
                entry._mtplx_verify_compiled_aux = compiled_aux

        if donate:
            # Commit-first ownership handoff, exactly as the fixed-M4 route:
            # the real cache now holds the output leaves, so dropping the
            # dispatcher's inputs lets MLX donate their buffers in-graph
            # instead of copying every KV leaf. (A pre-verify snapshot -- which
            # a block round always takes, because it is the only way to repair
            # a partial accept -- also references them, so donation degrades to
            # one COW rather than being unsafe.)
            state_in = None
            self._held_state_refs.clear()
            mx.async_eval(*outputs)

        self.stats["compiled_calls"] += 1
        self.stats["copy_round_compiled_calls"] += 1
        return logits, hidden

    def _publish_fixed_m4_selected_state(self, commit_plan) -> None:
        """Publish only the successfully enqueued authoritative frontier."""

        for kind, entry, selected_state in commit_plan:
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = selected_state[0]
                entry.kv.cache[1] = selected_state[1]
                entry.kv.cache[2] = selected_state[2]
                entry.raw_keys = selected_state[3]
                entry.pooled = selected_state[4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            else:
                for slot, leaf in enumerate(selected_state):
                    entry.cache[slot] = leaf
                entry._mtplx_verify_rows = None
                if len(entry.cache) > 2:
                    entry._mtplx_verify_ple = None
                    entry._mtplx_verify_compiled_aux = None

    def commit_fixed_m4_device_window(
        self,
        accepted_count,
        snapshot_states,
        verify_hidden,
    ):
        """Queue the construction-bound target state selection on device."""

        selected_hidden, state_roots = self._fixed_m4_dispatch["device_commit"](
            accepted_count,
            snapshot_states,
            verify_hidden,
        )
        mx.async_eval(selected_hidden, *state_roots)
        return selected_hidden

    def commit_fixed_m4_host_window(
        self,
        accepted_width,
        snapshot_states,
        verify_hidden,
    ):
        """Queue the same state selection for one host-known accepted width."""

        commit_width = self._fixed_m4_dispatch["device_commit_width"]
        if not callable(commit_width):
            raise RuntimeError(
                "fixed-M4 compact commit requires a construction-bound "
                "width-selected state commit"
            )
        selected_hidden, state_roots = commit_width(
            accepted_width,
            snapshot_states,
            verify_hidden,
        )
        mx.async_eval(selected_hidden, *state_roots)
        return selected_hidden

    def forward_ar_capture(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = True,
        hidden_variant: str | None = None,
        extended_window: bool = False,
    ):
        """Compiled verify dispatch.

        ``extended_window`` (context-copy block rounds, 2026-08-26 v2) admits
        lengths above ``max_verify_len`` up to ``MTPLX_CCOPY_BANK_MAX_LEN``.
        The extended lane changes ROUTING only, never the request's memory
        contract: the speculative reserve stays keyed to ``max_verify_len``,
        a dense-capacity preflight refuses (falls back eager) instead of
        growing a granted KV leaf, and paged capacity overflow falls back as
        before.  ``_paged_ineligibility`` is skipped for extended lengths:
        that gate is a performance router for windows whose eager alternative
        is cheap, while a block round's eager alternative costs ~380 ms flat
        at long context (MEASUREMENTS 2026-08-25 11:26) — inside the traced
        graph the paged kernel declining is shape-deterministic and routes to
        the same dense math the eager forward takes at the same T, so
        exactness is unaffected either way.
        """
        global _PREWARM_DONE
        if self._fixed_m4_dispatch is not None:
            self.stats["calls"] += 1
            if _decode_length(input_ids) == 4:
                raise RuntimeError(
                    "installed fixed-M4 replay requires forward_fixed_m4 host inputs"
                )
            return self._runtime_forward(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
            )
        if (
            not _PREWARM_DONE
            and not self.parity
            and not self.parity2
            and not extended_window
            and _prewarm_enabled()
        ):
            # First compiled dispatch of a generation while coverage is
            # incomplete (the first one of the process is normally the
            # startup warmup generation): walk the PAGED bucket ladder so
            # those graphs (and their Metal pipelines) exist before any
            # user-facing generation — paged bucket crossings were the bulk
            # of the −28% unrouted long-form cost (MEASUREMENTS 2026-07-02).
            # On the dense path this is a deliberate no-op
            # ("no_paged_entries", marks the walk complete): dense KV
            # retraces every 256 tokens of growth (5 traces per 1.3k-token
            # chat answer, measured 2026-07-02 21:25) and pre-walking ~24
            # shape classes to 6k is startup-prohibitive — the designed fix
            # there is pow2-bucketized dense leaves, not a longer prewarm.
            # F6 (2026-08-16): a walk CLAMPED by the current cache's paged
            # capacity (the 16-token boot warmup) no longer spends the
            # one-shot — later generations with more capacity (the server
            # warmup ladder rungs) extend the walk over the still-missing
            # buckets, so their compiles land in warmup, not in measured
            # rows. The walk is best-effort by design: a failure is
            # recorded visibly and the organic dispatch below handles the
            # same condition through its own fallback accounting.
            try:
                report = self.prewarm_ladder(
                    cache, input_ids, hidden_variant=hidden_variant
                )
            except Exception as exc:  # visible, never fatal (see docstring)
                report = {
                    "buckets": [],
                    "skipped": [f"walk_error:{type(exc).__name__}"],
                    "elapsed_s": 0.0,
                    "complete": False,
                }
            self.stats["prewarm"] = report
            _PREWARM_DONE = bool(report.get("complete"))
            prewarm_status["done"] = _PREWARM_DONE
            prewarm_status["walks"] = int(prewarm_status.get("walks", 0)) + 1
            prewarm_status["last_report"] = report
            prewarm_status["buckets"] = sorted(
                {bucket for _rt, _len, _var, bucket in _PREWARMED_BUCKETS}
            )
            if report.get("buckets") or int(prewarm_status["walks"]) == 1:
                # One line per walk that actually compiled something (plus
                # the first walk of the process); silent no-op retries stay
                # off the console.
                try:
                    import json as _json

                    print(
                        "[mtplx] compiled-verify prewarm " + _json.dumps(report),
                        flush=True,
                    )
                except Exception:
                    pass
        self.stats["calls"] += 1
        reason = self._fallback_reason(
            input_ids, cache, return_hidden, extended_window=extended_window
        )
        if reason is not None:
            return self._fallback(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=reason,
            )
        length = _decode_length(input_ids)
        try:
            bucket = self._resolve_bucket(cache, length)
            if bucket is None:
                return self._fallback(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    reason="capacity_overflow",
                )
            max_ctx = _compiled_verify_max_context()
            if max_ctx and getattr(self, "_last_context_estimate", 0) > max_ctx:
                # Context-scaled router: compiled verify is proven bit-exact
                # and +4.8% only up to ~6k ctx; beyond, eager wins and the
                # exactness corpus has no coverage. Fall back per call.
                return self._fallback(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    reason="context_above_threshold",
                )
            if not (extended_window and length > self.max_verify_len):
                # Extended block windows skip this performance router (see
                # the method docstring); every other call keeps it verbatim.
                ineligible = self._paged_ineligibility(cache, length, bucket)
                if ineligible is not None:
                    return self._fallback(
                        input_ids,
                        cache=cache,
                        return_hidden=return_hidden,
                        hidden_variant=hidden_variant,
                        reason=ineligible,
                    )
            self._ensure_shadow(cache)
            self._apply_bucket(cache, bucket)
            # Boundary policy (experiment knob, 2026-07-02 sprint):
            #   pre  — materialize pending input state with the eager kernels
            #          before entering the compiled function. Exactness
            #          boundary: a lazy upstream graph absorbed into compiled
            #          execution computes with fused-kernel numerics (~1e-6),
            #          breaking bit-parity with the eager reference.
            #   post — schedule evaluation of outputs while the input leaves
            #          are still referenced by the real cache. Buffer-safety
            #          boundary: without it, mirror-commit drops the last
            #          input references while the compiled graph is pending
            #          and the allocator reuses their buffers.
            # MTPLX_COMPILED_VERIFY_BOUNDARY = both (default) | pre | post |
            # none. When 'post' is dropped, buffer safety is preserved by
            # holding the input references until the NEXT dispatch instead
            # (self._held_state_refs) — no numerics cost, no forced batch.
            boundary = _compiled_verify_boundary()
            donate = (
                _compiled_verify_donation_enabled()
                and not self.parity
                and not self.parity2
                and boundary in ("both", "post")
            )
            if donate:
                # A2.1: the shadow twins hold promotion-time leaf refs that
                # (a) pin one full stale KV buffer set for the generation and
                # (b) alias the first call's input buffers, blocking their
                # donation. The traced body re-seeds every slot from the
                # explicit inputs before any read, so the held refs are dead.
                self._clear_shadow_leaf_refs()
            key = (length, str(hidden_variant or ""), int(bucket))
            fn = self._compiled.get(key)
            if fn is None:
                fn = self._shared_or_new_verify_step(key, length, hidden_variant)
                self._compiled[key] = fn
            state_in = self._read_state_leaves(cache)
            if state_in is None:
                return self._fallback(
                    input_ids,
                    cache=cache,
                    return_hidden=return_hidden,
                    hidden_variant=hidden_variant,
                    reason="empty_state_leaf",
                )
            compiled_aux = (
                self._prepare_compiled_aux(input_ids, cache)
                if self._prepare_compiled_aux is not None
                else None
            )
            if boundary in ("both", "pre"):
                mx.async_eval(
                    *((compiled_aux,) if compiled_aux is not None else ()),
                    *state_in,
                )
            outputs = (
                fn(input_ids, compiled_aux, *state_in)
                if compiled_aux is not None
                else fn(input_ids, *state_in)
            )
            logits, hidden, captures_flat, state_out = self._unpack_outputs(outputs)
            if donate:
                # A2.1 commit-first ownership handoff — commit + schedule
                # happen AFTER this fallback-safe block (see below): once the
                # real cache is rebound to the outputs, an eager fallback
                # would double-apply the verify window.
                pass
            elif boundary in ("both", "post"):
                mx.async_eval(*outputs)
                self._held_state_refs.clear()
            else:
                # Keep inputs alive across a 3-generation window: with the
                # deferred serve path, call N-1's graph may still be pending
                # when call N dispatches, so a single-slot hold can release
                # buffers the allocator then reuses. Three generations covers
                # the deepest deferred chain the serve path produces
                # (experiment probe; production would release on evidence).
                self._held_state_refs.append(state_in)
                if len(self._held_state_refs) > 3:
                    self._held_state_refs.pop(0)
        except Exception as exc:
            self._exception_failures += 1
            compiled_verify_status["transient_exception_count"] = (
                int(compiled_verify_status.get("transient_exception_count", 0)) + 1
            )
            if self._exception_failures >= 3:
                self.permanent_eager = True
                self.permanent_eager_reason = (
                    f"exception_streak:{type(exc).__name__}"
                )
                _record_permanent_eager(self.permanent_eager_reason)
            return self._fallback(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                reason=f"exception:{type(exc).__name__}",
            )
        self._exception_failures = 0
        self.stats["compiled_calls"] += 1
        if extended_window and length > self.max_verify_len:
            self.stats["extended_calls"] += 1
        bucket_key = str(int(bucket))
        self.stats["buckets"][bucket_key] = self.stats["buckets"].get(bucket_key, 0) + 1
        captures = self._rebuild_captures(captures_flat)
        if self.parity:
            return self._parity_check(
                input_ids,
                cache=cache,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
                state_in=state_in,
                compiled_logits=logits,
                compiled_hidden=hidden,
                compiled_captures=captures,
                compiled_state_out=state_out,
            )
        if self.parity2:
            return self._parity2_check(
                input_ids,
                cache=cache,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
                bucket=int(bucket),
                compiled_logits=logits,
                compiled_hidden=hidden,
                compiled_captures=captures,
                compiled_state_out=state_out,
            )
        self._mirror_commit(cache, state_out)
        if self._commit_compiled_captures is not None:
            self._commit_compiled_captures(cache, captures)
        if donate:
            # A2.1 commit-first ownership handoff: the real cache is already
            # rebound to the output leaves, so dropping the dispatcher's
            # ``state_in`` list makes the pending graph the ONLY holder of
            # each input KV buffer at schedule time.  MLX then donates the
            # buffer into the in-graph ``slice_update`` instead of
            # materializing a full copy of every full-attn K/V buffer per
            # verify call (measured 16.5 ms @64k, ~33 ms @128k — probe arms
            # A vs G, outputs/ivanbench-20260705/compiled_copy_tax_probe.py).
            # Byte-exactness across chained pending calls and snapshot-COW
            # pinning proven in compiled_copy_tax_correctness.py; buffers
            # shared with a bank entry (restore/postcommit views) simply COW
            # once, exactly as before.  (A freshly built shadow still holds
            # the promotion-time leaves, so the first call of a generation
            # pays one copy; calls 2+ donate because the shadow's stale refs
            # never alias the current inputs.)
            state_in = None
            self._held_state_refs.clear()
            mx.async_eval(*outputs)
        return logits, hidden, captures

    def prewarm_ladder(
        self,
        cache: Any,
        input_ids,
        *,
        hidden_variant: str | None = None,
        max_context: int | None = None,
    ) -> dict[str, Any]:
        """Compile-and-execute the verify step once per pow2 bucket up to
        the router boundary, priming the Metal shader cache.

        Outputs are discarded and state is never committed (`verify_step`
        is a pure function of its state leaves), so the caller's cache is
        untouched apart from the static bucket ceiling, which is restored
        to its natural value before returning. Failures are recorded per
        bucket and never flip ``permanent_eager`` — a bucket that cannot
        prewarm simply pays its organic compile later.

        ``report["complete"]`` is the one-shot verdict (F6): True when no
        future walk could add coverage (the ladder reached the router
        ceiling, or the cache is structurally ladder-free), False when the
        walk was clamped by the current cache's paged capacity or skipped
        for a transient reason — the trigger then retries on a later
        generation whose cache reaches further. Buckets warmed by earlier
        walks are skipped (``report["already"]``), so a retry with nothing
        new to add costs a few python comparisons.
        """
        report: dict[str, Any] = {
            "buckets": [],
            "skipped": [],
            "already": [],
            "elapsed_s": 0.0,
            "complete": False,
        }
        started = time.perf_counter()

        def _finish() -> dict[str, Any]:
            report["elapsed_s"] = round(time.perf_counter() - started, 3)
            return report

        if self.permanent_eager:
            # Structural for this process/model (quant gate) or already a
            # terminal degradation — nothing a later walk could add.
            report["skipped"].append("permanent_eager")
            report["complete"] = True
            return _finish()
        reason = self._fallback_reason(
            input_ids, cache, True, consume_post_restore=False
        )
        if reason is not None:
            report["skipped"].append(reason)
            return _finish()
        length = _decode_length(input_ids)
        try:
            natural = self._resolve_bucket(cache, length)
        except Exception as exc:
            report["skipped"].append(f"resolve:{type(exc).__name__}")
            return _finish()
        if not natural:
            report["skipped"].append(
                "capacity_overflow" if natural is None else "no_paged_entries"
            )
            # Dense caches have no paged bucket ladder by design (see the
            # trigger comment): the walk is complete, not clamped.
            report["complete"] = natural is not None
            return _finish()
        boundary = (
            int(max_context)
            if max_context is not None
            else _compiled_verify_max_context()
        )
        if boundary <= 0:
            # Router disabled: only the natural bucket is reachable cheaply;
            # deeper buckets appear at unbounded context growth and warming
            # them all is unbounded work.
            boundary = int(natural)
        min_capacity: int | None = None
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if hasattr(entry, "capacity"):
                cap = int(entry.capacity)
                min_capacity = cap if min_capacity is None else min(min_capacity, cap)
        ceiling = _next_pow2(boundary + length + 512)
        if int(natural) > ceiling:
            # This call's context is already above the compiled-verify
            # router: every dispatch of this generation falls back per call
            # ("context_above_threshold"), so walking (and compiling) its
            # bucket would burn ~1s on a graph no compiled row can use.
            report["skipped"].append("context_above_router")
            return _finish()
        ladder: list[int] = []
        bucket = int(natural)
        while True:
            if min_capacity is not None:
                bucket = min(bucket, min_capacity)
            if bucket not in ladder:
                ladder.append(bucket)
            if min_capacity is not None and bucket >= min_capacity:
                break
            if bucket >= ceiling:
                break
            bucket *= 2
        # Complete = the ladder reached the router ceiling. A walk clamped
        # below it by min_capacity leaves the one-shot unspent so a later,
        # larger cache (server warmup ladder rungs) extends the coverage.
        report["complete"] = bool(ladder) and int(ladder[-1]) >= ceiling
        variant_key = str(hidden_variant or "")
        runtime_id = id(self.runtime)
        pending = [
            bucket
            for bucket in ladder
            if (runtime_id, length, variant_key, int(bucket))
            not in _PREWARMED_BUCKETS
        ]
        report["already"] = [
            int(bucket) for bucket in ladder if bucket not in pending
        ]
        if not pending:
            return _finish()
        self._ensure_shadow(cache)
        state_in = self._read_state_leaves(cache)
        if state_in is None:
            report["skipped"].append("empty_state_leaf")
            report["complete"] = False
            return _finish()
        for bucket in pending:
            if self._paged_ineligibility(cache, length, bucket) is not None:
                report["skipped"].append(f"b{bucket}:paged_kernel_ineligible")
                continue
            try:
                self._apply_bucket(cache, bucket)
                key = (length, variant_key, int(bucket))
                fn = self._compiled.get(key)
                if fn is None:
                    # Shared-registry compile (F6): a bare per-bank
                    # mx.compile primed the Metal pipelines but kept the
                    # trace private to the warmup bank, so the first real
                    # request at the same shapes re-traced every bucket
                    # (~1s each) inside its measured row. The shared step
                    # is exactly what organic dispatch consults.
                    fn = self._shared_or_new_verify_step(key, length, hidden_variant)
                    self._compiled[key] = fn
                bucket_started = time.perf_counter()
                outputs = fn(input_ids, *state_in)
                # Synchronous eval: the compile cost is paid HERE, and no
                # graph is left pending, so no held-reference bookkeeping
                # is needed. Outputs are dropped, never committed.
                mx.eval(*outputs)
                report["buckets"].append(
                    {
                        "bucket": int(bucket),
                        "s": round(time.perf_counter() - bucket_started, 3),
                    }
                )
                _PREWARMED_BUCKETS.add((runtime_id, length, variant_key, int(bucket)))
            except Exception as exc:
                report["skipped"].append(f"b{bucket}:{type(exc).__name__}")
        try:
            restored = self._resolve_bucket(cache, length)
            if restored:
                self._apply_bucket(cache, restored)
        except Exception:
            pass
        return _finish()

    def prewarm_extended_lengths(
        self,
        cache: Any,
        lengths: list[int],
        *,
        hidden_variant: str | None = None,
    ) -> dict[str, Any]:
        """Trace the extended (context-copy block) windows ahead of use.

        Optional, driven by ``MTPLX_CCOPY_BANK_PREWARM`` at the ccopy site.
        Without it the first block round per (length, bucket) pays the fresh
        ``mx.compile`` trace organically — once per PROCESS (shared traces),
        which was the recurring-looking "~240 ms/call" in the 8-round v1 cell
        (one ~1s first-trace amortized over 8 rounds; the dispatch layer
        itself re-clones nothing, probe receipts 2026-08-26). A/B cells that
        time steady-state block rounds should enable this so first-trace cost
        lands in warmup, not in a measured row.

        Same firewall economics as ``prewarm_ladder``: the compiled function
        is pure, outputs are dropped and never mirror-committed, so the live
        cache is untouched. The dry run cannot donate its input buffers (the
        real cache still holds every leaf), so each traced length transiently
        materializes one copy of the full-attn KV set — the same one-time
        copy the first organic call of a generation pays.
        """
        report: dict[str, Any] = {"lengths": [], "skipped": [], "elapsed_s": 0.0}
        started = time.perf_counter()

        def _finish() -> dict[str, Any]:
            report["elapsed_s"] = round(time.perf_counter() - started, 3)
            self.stats["extended_prewarm"] = report
            return report

        if self.permanent_eager or self.parity or self.parity2:
            report["skipped"].append("bank_mode")
            return _finish()
        ceiling = max(self.max_verify_len, _ccopy_bank_max_len())
        variant_key = str(hidden_variant or "")
        runtime_id = id(self.runtime)
        for length in sorted({int(item) for item in lengths}):
            if length <= self.max_verify_len or length > ceiling:
                report["skipped"].append(f"m{length}:outside_extended_window")
                continue
            probe = mx.zeros((1, length), dtype=mx.int32)
            reason = self._fallback_reason(
                probe,
                cache,
                True,
                consume_post_restore=False,
                extended_window=True,
            )
            if reason is not None:
                report["skipped"].append(f"m{length}:{reason}")
                continue
            try:
                bucket = self._resolve_bucket(cache, length)
                if bucket is None:
                    report["skipped"].append(f"m{length}:capacity_overflow")
                    continue
                if (
                    runtime_id,
                    length,
                    variant_key,
                    int(bucket),
                ) in _PREWARMED_BUCKETS:
                    report["skipped"].append(f"m{length}:already")
                    continue
                self._ensure_shadow(cache)
                self._apply_bucket(cache, bucket)
                state_in = self._read_state_leaves(cache)
                if state_in is None:
                    report["skipped"].append(f"m{length}:empty_state_leaf")
                    continue
                key = (length, variant_key, int(bucket))
                fn = self._compiled.get(key)
                if fn is None:
                    fn = self._shared_or_new_verify_step(key, length, hidden_variant)
                    self._compiled[key] = fn
                length_started = time.perf_counter()
                outputs = fn(probe, *state_in)
                mx.eval(*outputs)
                report["lengths"].append(
                    {
                        "m": int(length),
                        "bucket": int(bucket),
                        "s": round(time.perf_counter() - length_started, 3),
                    }
                )
                _PREWARMED_BUCKETS.add((runtime_id, length, variant_key, int(bucket)))
            except Exception as exc:
                report["skipped"].append(f"m{length}:{type(exc).__name__}")
        try:
            # Politeness restore (mirrors prewarm_ladder): an eager forward
            # between this walk and the next dispatch should see a natural
            # static ceiling, not the last extended length's. Any ceiling
            # >= offset + T is topology-valid, and dispatch re-applies its
            # own bucket before every compiled call.
            restored = self._resolve_bucket(cache, 1)
            if restored:
                self._apply_bucket(cache, restored)
        except Exception:
            pass
        return _finish()

    def _materialize_growth_handoff_state(self, cache: Any) -> int:
        """Settle compiled state before the eager tail takes ownership.

        Compiled dispatch schedules every output asynchronously. Merely
        replacing the tensor-offset cache containers leaves their KV and
        recurrent leaves attached to that deferred graph. The eager tail then
        inherits the compiled dependency chain, so long generations pay the
        old work through later verify-output evaluations instead of crossing a
        clean ownership boundary.

        Growth demotion is a once-per-request transition. Evaluate the current
        state exactly once here, while the compiled state spec is still valid,
        then let ``demote`` replace the containers and release compiled refs.
        """
        state = self._read_state_leaves(cache)
        if state is None:
            raise RuntimeError(
                "compiled verify growth handoff has incomplete cache state"
            )
        leaves: list[mx.array] = []
        seen: set[int] = set()
        for leaf in state:
            if not isinstance(leaf, mx.array):
                continue
            identity = id(leaf)
            if identity in seen:
                continue
            seen.add(identity)
            leaves.append(leaf)
        started = time.perf_counter()
        if leaves:
            mx.eval(*leaves)
        self.stats["growth_handoff_materializations"] = (
            int(self.stats.get("growth_handoff_materializations", 0)) + 1
        )
        self.stats["growth_handoff_state_leaves"] = (
            int(self.stats.get("growth_handoff_state_leaves", 0)) + len(leaves)
        )
        self.stats["growth_handoff_materialize_time_s"] = float(
            self.stats.get("growth_handoff_materialize_time_s", 0.0)
        ) + (time.perf_counter() - started)
        return len(leaves)

    def demote(self, cache: Any) -> int:
        """Restore stock containers for every tensor-offset adapter in place.

        Mandatory before postcommit / final-state capture: downstream cache
        consumers must never see promoted adapters.
        """
        try:
            from .cache_state import TensorOffsetVllmMetalPagedKVCache
        except Exception:  # pragma: no cover - import guard for minimal test envs
            TensorOffsetVllmMetalPagedKVCache = None
        count = 0
        for idx, entry in enumerate(cache or []):
            if isinstance(entry, TensorOffsetQSACache):
                cache[idx] = entry.demote()
                count += 1
            elif isinstance(entry, TensorOffsetKVCache):
                cache[idx] = entry.demote()
                count += 1
            elif TensorOffsetVllmMetalPagedKVCache is not None and isinstance(
                entry, TensorOffsetVllmMetalPagedKVCache
            ):
                cache[idx] = entry.demote()
                count += 1
        if count:
            self.stats["demotions"] += count
            # Container identity changed; compiled closures bound the old
            # shadow, which no longer mirrors the cache list.
            self._clear_shadow_leaf_refs()
            self._held_state_refs.clear()
            self._held_fixed_m4_split_refs.clear()
            # W63: a queued layer-0 prefix was traced against the shadow that
            # is about to stop mirroring the cache; its generation stamp would
            # still match, so drop it explicitly.
            self.discard_fixed_m4_overlap_prefix()
            self._fixed_m4_split_generation = -1
            self._shadow = None
            self._shadow_signature = None
            self._spec = None
            self._compiled.clear()
        return count

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.stats)
        data["fallback_reasons"] = dict(self.stats["fallback_reasons"])
        data["buckets"] = dict(self.stats["buckets"])
        first_divergence = self.stats.get("parity2_first_divergence")
        data["parity2_first_divergence"] = (
            dict(first_divergence) if isinstance(first_divergence, dict) else None
        )
        if self.parity2:
            data["mode"] = "parity2"
        else:
            data["mode"] = "parity" if self.parity else "on"
        data["max_verify_len"] = self.max_verify_len
        data["request_max_tokens"] = self.request_max_tokens
        data["speculative_headroom"] = self.speculative_headroom
        data["growth_reserve_tokens"] = self.growth_reserve_tokens
        data["capture_backend"] = self.capture_backend
        data["permanent_eager"] = self.permanent_eager
        data["permanent_eager_reason"] = getattr(
            self, "permanent_eager_reason", None
        )
        data["compiled_entry_count"] = len(self._compiled)
        data["compiled_keys"] = [
            _format_compiled_verify_key(key) for key in sorted(self._compiled)
        ]
        data["copy_round_installed"] = self._copy_round_dispatch is not None
        return data

    # -- dispatch preconditions ----------------------------------------------

    def _fallback_reason(
        self,
        input_ids,
        cache,
        return_hidden: bool,
        *,
        consume_post_restore: bool = True,
        extended_window: bool = False,
    ) -> str | None:
        if self.permanent_eager:
            return "permanent_eager"
        if not return_hidden:
            return "hidden_not_requested"
        shape = getattr(input_ids, "shape", None)
        if shape is None or len(shape) != 2:
            return "invalid_input_shape"
        if int(shape[0]) != 1:
            return "batch_size"
        length = int(shape[1])
        if length < 1:
            return "invalid_length"
        window_ceiling = (
            max(self.max_verify_len, _ccopy_bank_max_len())
            if extended_window
            else self.max_verify_len
        )
        if length > window_ceiling:
            return "length_outside_bank"
        if extended_window and length > self.max_verify_len:
            # Dense-capacity preflight: an extended window must never grow a
            # granted dense KV leaf (`promote_kv_cache_offsets` below would
            # call `ensure_capacity(size + length)` and flip
            # `growth_after_grant`). When the window cannot fit the grant,
            # run the SAME once-per-request growth-demotion transition the
            # MTP lane runs at grant exhaustion — the MTP top-up would trip
            # it within `max_verify_len` tokens anyway — so the eager
            # fallback verifies against stock containers that grow natively.
            # (Falling back onto the still-granted adapter would overflow
            # its fixed buffer inside `update_and_fetch`; the route-off
            # eager lane shares that narrow dense-edge exposure today.)
            for entry in cache or []:
                if not isinstance(entry, TensorOffsetKVCache):
                    continue
                if entry.keys is None:
                    continue
                if entry.size() + length > int(entry.keys.shape[2]):
                    self._growth_demoted = True
                    self.stats["growth_demotions"] = (
                        int(self.stats.get("growth_demotions", 0)) + 1
                    )
                    self._materialize_growth_handoff_state(cache)
                    self.demote(cache)
                    return "block_window_capacity"
        if self.capture_backend in _UNSUPPORTED_CAPTURE_BACKENDS:
            return "unsupported_capture_backend"
        if _owned_state_env_active("MTPLX_OWNED_ATTN_KV"):
            return "owned_attn_kv_env"
        if _owned_state_env_active("MTPLX_OWNED_RECURRENT_STATE"):
            return "owned_recurrent_state_env"
        if cache is None:
            return "no_cache"
        if self._growth_demoted:
            # Cache was demoted back to stock entries when the growth budget
            # tripped; the plain eager path owns the rest of this request.
            return "growth_budget_exhausted"
        if self._post_restore_eager_remaining > 0:
            # Keep the restored cache unpromoted for the first round(s) so the
            # O(context) ensure_capacity copy lands after the first token is
            # already on the wire, not inside warm TTFT. Non-consuming probes
            # (prewarm eligibility) must not tick the counter — and must still
            # skip, or the probe itself would promote and pay the copy.
            if consume_post_restore:
                self._post_restore_eager_remaining -= 1
            return "post_restore_warmup"
        promoted, failures = promote_kv_cache_offsets(
            cache,
            reserve_tokens=length,
            preserve_paged=True,
            initial_reserve_tokens=max(length, self.growth_reserve_tokens),
        )
        self.stats["promoted"] += promoted
        for entry in cache:
            if isinstance(entry, TensorOffsetKVCache) and entry.growth_after_grant:
                # A dense leaf outgrew its first-promotion grant: every
                # further growth step would retrace the compiled graph, and
                # eager-on-adapter pays capacity-wide masks + non-donatable
                # slice updates (measured -15% vs clean eager at 7k). Demote
                # to stock entries NOW and stay eager for the rest of this
                # request (the bank is per-request, so the next round
                # re-grants fresh headroom).
                self._growth_demoted = True
                self.stats["growth_demotions"] = (
                    int(self.stats.get("growth_demotions", 0)) + 1
                )
                self._materialize_growth_handoff_state(cache)
                self.demote(cache)
                return "growth_budget_exhausted"
        if failures:
            if "quantized_paged_kv_cache" in failures:
                return "quantized_paged_kv"
            if "quantized_paged_kv_geometry" in failures:
                return "quantized_paged_kv_geometry"
            return "promotion_failure:" + ",".join(sorted(failures))
        if cache_has_python_offsets(cache):
            return "python_cache_offsets"
        spec, spec_reason = build_verify_state_spec(cache)
        if spec is None:
            return spec_reason or "unsupported_container"
        self._spec = spec
        if self.capture_backend == "linear_gdn_from_conv_tape":
            for idx, kind, _n in spec:
                if kind == VERIFY_SPEC_KIND_GDN and self._gdn_meta(idx) is None:
                    return "gdn_meta_unavailable"
        return None

    def _resolve_bucket(self, cache: Any, length: int) -> int | None:
        """Static paged-attention ceiling for this call, or None on overflow."""
        if _BATCH_PAGED_OFFSETS and _PAGED_OFFSETS_CONTEXT_OK.get():
            # One eval for every paged offset instead of a serial sync per
            # entry inside size() below (#318; helper docstring has the
            # mechanism). Mirrors this loop's own iteration exactly.
            paged_offsets = []
            for spec_idx, spec_kind, _n in self._spec or []:
                if spec_kind != VERIFY_SPEC_KIND_FULL_ATTN:
                    continue
                spec_entry = cache[spec_idx]
                if not hasattr(spec_entry, "capacity"):
                    continue
                entry_state = getattr(spec_entry, "cache", None)
                if isinstance(entry_state, (list, tuple)) and len(entry_state) > 2:
                    entry_offset = entry_state[2]
                    if isinstance(entry_offset, mx.array):
                        paged_offsets.append(entry_offset)
            if paged_offsets:
                mx.eval(*paged_offsets)
        max_needed = 0
        min_capacity: int | None = None
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if not hasattr(entry, "capacity"):
                continue  # dense adapter: grows via ensure_capacity instead
            offset = int(entry.size())
            capacity = int(entry.capacity)
            max_needed = max(max_needed, offset + length)
            min_capacity = capacity if min_capacity is None else min(min_capacity, capacity)
        self._last_context_estimate = max_needed
        if min_capacity is None:
            return 0  # no paged entries; bucket unused
        if max_needed > min_capacity:
            return None
        bucket = min(min_capacity, _next_pow2(max_needed + 512))
        if max_needed > bucket:  # hard precondition: offset+M <= bucket
            bucket = min_capacity
        return bucket

    def _paged_ineligibility(self, cache: Any, length: int, bucket: int) -> str | None:
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if not hasattr(entry, "capacity"):
                continue
            if not _paged_kernel_bucket_eligible(entry, length, bucket):
                return "paged_kernel_ineligible"
        return None

    def _apply_bucket(self, cache: Any, bucket: int) -> None:
        """Pin the per-instance static ceiling on shadow and real paged entries.

        The two-pass paged kernel's reduction topology depends on the static
        ceiling, so the real entries get the same bucket: eager fallback calls
        and parity's authoritative eager run then use the identical kernel
        shape, which is what makes bit-exact comparison meaningful.
        """
        if not bucket:
            return
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            if hasattr(entry, "static_max_offset"):
                entry.static_max_offset = int(bucket)
            shadow_entry = self._shadow[idx] if self._shadow else None
            if shadow_entry is not None and hasattr(shadow_entry, "static_max_offset"):
                shadow_entry.static_max_offset = int(bucket)

    # -- shadow cache ---------------------------------------------------------

    def _container_signature(self, cache: Any) -> tuple[Any, ...]:
        signature: list[Any] = []
        for entry in cache or []:
            if entry is None:
                signature.append(None)
                continue
            meta = (
                (int(entry.block_size), int(entry.num_blocks))
                if hasattr(entry, "num_blocks")
                else ()
            )
            signature.append((id(entry), type(entry).__name__, meta))
        return tuple(signature)

    def _ensure_shadow(self, cache: Any) -> None:
        signature = self._container_signature(cache)
        if self._shadow is not None and signature == self._shadow_signature:
            return
        from .cache_state import (
            TensorOffsetQuantizedPagedKVCache,
            TensorOffsetVllmMetalPagedKVCache,
        )

        shadow: list[Any] = [None] * len(cache)
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                kv = TensorOffsetKVCache(
                    entry.kv.cache[0],
                    entry.kv.cache[1],
                    entry.kv.cache[2],
                    step=entry.kv.step,
                )
                twin = TensorOffsetQSACache(
                    kv,
                    entry.raw_keys,
                    entry.pooled,
                    compress_ratio=entry.ratio,
                    rows_gather=entry.fixed_rows_gather,
                    rows_gather_kv_m4=entry.rows_gather_kv_m4,
                    rows_gather_enabled=entry.rows_gather_enabled,
                    rows_gather_min_context=entry.rows_gather_min_context,
                    fused_rows_gather_kv_m4=entry.fused_rows_gather_kv_m4,
                    # A twin that dropped these would silently revert to
                    # the stock QSA chain -- the armed-but-inert failure mode.
                    fable_qsa_m4=entry.fable_qsa_m4,
                    fable_qsa_m4_kt=entry.fable_qsa_m4_kt,
                    fable_qsa_sparse_decode=getattr(
                        entry, "fable_qsa_sparse_decode", False
                    ),
                    fable_qsa_sparse_draft=getattr(
                        entry, "fable_qsa_sparse_draft", False
                    ),
                )
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        entry.cache[0],
                        entry.cache[1],
                        entry.cache[2],
                        step=entry.step,
                    )
                elif isinstance(entry, TensorOffsetQuantizedPagedKVCache):
                    twin = TensorOffsetQuantizedPagedKVCache(
                        key_cache=entry.cache[0],
                        value_cache=entry.cache[1],
                        offset=entry.cache[2],
                        key_scale_cache=entry.cache[3],
                        value_scale_cache=entry.cache[4],
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                        kv_quant_config=entry.kv_quant_config,
                        source_dtypes=entry.source_dtypes,
                        head_dims=entry.head_dims,
                    )
                else:
                    twin = TensorOffsetVllmMetalPagedKVCache(
                        key_cache=entry.cache[0],
                        value_cache=entry.cache[1],
                        offset=entry.cache[2],
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                    )
            else:
                twin = type(entry)(len(entry.cache))
                for slot, leaf in enumerate(entry.cache):
                    twin[slot] = leaf
            shadow[idx] = twin
        self._shadow = shadow
        self._shadow_signature = signature
        # New shadow objects invalidate closures compiled over the old ones.
        self._compiled.clear()

    # -- compiled function ------------------------------------------------------

    def _shared_or_new_verify_step(
        self,
        key,
        length: int,
        hidden_variant: str | None,
        *,
        graph_aux=None,
        return_compiled_aux: bool = False,
        fold_indices: tuple[int, ...] = (),
    ):
        """Reuse one compiled verify callable per process for a logical key.

        The bank is constructed per generation, so a per-instance compile dict
        pays a fresh trace (~1s wall at 7k leaves, measured 2026-07-03 as the
        whole compiled-vs-eager gap on long generations) for every request.
        The traced graph depends only on the runtime, capture layout, state
        spec, verify length, and hidden variant — mx.compile re-traces
        internally when leaf shapes change and caches per shape signature —
        so callables are shared process-wide. The closure's shadow containers
        are trace-time scratch: the re-seed firewall assigns every leaf from
        the explicit inputs before any read, so a retrace under a different
        bank/request is safe. `_TRACE_HOSTS` keeps each callable's shadow and
        stats sink pointed at the LIVE bank so retraces never touch a dead
        request's containers.
        """

        if not _env_enabled("MTPLX_COMPILED_VERIFY_SHARED_TRACES", default=True):
            return mx.compile(
                self._make_verify_step(
                    length,
                    hidden_variant,
                    graph_aux=graph_aux,
                    return_compiled_aux=return_compiled_aux,
                    fold_indices=fold_indices,
                )
            )
        spec_sig = tuple(self._spec or [])
        from .attention_context import exact_verify_required

        global_key = (
            id(self.runtime),
            self.capture_backend,
            self._capture_layout_override,
            self._extra_capture_layout,
            self._prepare_compiled_aux is not None,
            spec_sig,
            int(length),
            str(hidden_variant or ""),
            int(key[2]),
            str(key[3]) if len(key) > 3 else "materialized",
            # Kernel-route dimension: a trace compiled under the sampled
            # (vk/nax) verify route bakes those kernels into the graph; a
            # greedy (t<=0, stock-route) request must never replay it, and
            # vice versa. Without this key a t=0.6 request's shared trace
            # would silently serve a t=0 request with non-exact kernels.
            bool(exact_verify_required()),
            # Keep-mask fold dimension (W66b): a trace whose GDN steps take a
            # prefix has a different input arity AND a different recurrence
            # from one that does not.  Without this key an armed bank could be
            # served the control's trace (silently inert) or, worse, an
            # unarmed bank could replay a folded graph with no prefix bound.
            # It is per-CALLER, not per-bank: only the fixed-M4 length-4
            # installation passes a prefix, so the fallback/other-length
            # traces on the SAME bank stay exactly what they are today.
            tuple(fold_indices),
            int(getattr(self, "_fold_windows", 0)) if fold_indices else 0,
        )
        entry = _SHARED_VERIFY_STEPS.get(global_key)
        if entry is not None:
            fn, host, runtime_ref = entry
            # id() can be recycled after a model swap frees the old runtime;
            # a stale callable would replay graphs bound to freed weights.
            if runtime_ref() is self.runtime:
                host["bank"] = self
                return fn
            _SHARED_VERIFY_STEPS.pop(global_key, None)
        host = {"bank": self}
        fn = mx.compile(
            self._make_verify_step(
                length,
                hidden_variant,
                trace_host=host,
                graph_aux=graph_aux,
                return_compiled_aux=return_compiled_aux,
                fold_indices=fold_indices,
            )
        )
        _SHARED_VERIFY_STEPS[global_key] = (fn, host, weakref.ref(self.runtime))
        return fn

    def _make_verify_step(
        self,
        length: int,
        hidden_variant: str | None,
        trace_host: dict[str, Any] | None = None,
        *,
        graph_aux=None,
        return_compiled_aux: bool = False,
        fold_indices: tuple[int, ...] = (),
    ):
        spec = list(self._spec or [])
        layout = self._capture_layout()
        bank = self
        static_host = {"bank": self}
        host = trace_host if trace_host is not None else static_host
        # W66b: closure-captured like `spec` and `layout`, and keyed into the
        # shared-trace key beside them, so a retrace under a different bank can
        # never disagree about whether the graph carries a prefix.  Only the
        # fixed-M4 length-4 installation asks for it -- the fallback and
        # other-length traces built on the same bank keep an empty tuple and
        # are byte-identical to today.
        fold_indices = tuple(fold_indices)
        fold_prefix_leaves = _gdn_fold.prefix_leaf_count(len(fold_indices))

        del bank

        def verify_step(input_ids, *args):
            # Python body executes at trace time only; replays skip it.
            live = host["bank"]
            shadow = live._shadow
            if live._prepare_compiled_aux is not None:
                compiled_aux, *state_in = args
            else:
                compiled_aux = None
                state_in = args
            if graph_aux is not None:
                compiled_aux = graph_aux(compiled_aux)
            live.stats["traces"] += 1
            if _decode_length(input_ids) != length:
                raise ValueError("compiled verify length mismatch")
            # (1) Re-seed firewall: every shadow leaf is assigned from the
            # explicit inputs BEFORE any read, so nothing stale and no tracer
            # from a previous trace can leak into this graph.
            pos = 0
            for idx, kind, n_leaves in spec:
                entry = shadow[idx]
                if kind == VERIFY_SPEC_KIND_QSA:
                    entry.kv.cache[0] = state_in[pos]
                    entry.kv.cache[1] = state_in[pos + 1]
                    entry.kv.cache[2] = state_in[pos + 2]
                    entry.raw_keys = state_in[pos + 3]
                    entry.pooled = state_in[pos + 4]
                    for slot in range(len(entry.kv.rollback_state)):
                        entry.kv.rollback_state[slot] = None
                elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    for slot in range(n_leaves):
                        entry.cache[slot] = state_in[pos + slot]
                    for slot in range(len(entry.rollback_state)):
                        entry.rollback_state[slot] = None
                else:
                    for slot in range(n_leaves):
                        entry.cache[slot] = state_in[pos + slot]
                pos += n_leaves
            # (1b) W66b keep-mask fold: `spec` consumes `state_in`
            # positionally, so the padded prefix is simply everything after
            # it -- 5 row tensors per foldable GDN layer plus one shared
            # `[1, 4*W]` mask, all at fixed shapes.  The scope is trace-time
            # scaffolding only: it exists so each layer's step wires the right
            # prefix tracers into the graph.  Replays bind the same graph
            # positionally and never run this body.
            fold_scope = None
            if fold_indices:
                trailing = state_in[pos:]
                if len(trailing) != fold_prefix_leaves:
                    raise ValueError(
                        f"compiled verify got {len(trailing)} keep-mask fold "
                        f"leaves, expected {fold_prefix_leaves}"
                    )
                fold_scope = _gdn_fold.make_prefix_scope(
                    fold_indices, trailing, lambda index: shadow[index]
                )
            # (2) The existing runtime forward, on shadow containers only.
            with _gdn_fold.fold_prefix_scope(fold_scope):
                with attention_phase("decode_verify"):
                    result = live._runtime_forward(
                        input_ids,
                        cache=shadow,
                        return_hidden=True,
                        hidden_variant=hidden_variant,
                        compiled_aux=compiled_aux,
                    )
            # W66d: the fold is only exact because the step kernel that was
            # handed the ring's BASE also replayed the ring.  A layer that
            # missed its prefix does not decline -- it runs the stock
            # recurrence from that base and silently drops committed windows,
            # which no downstream counter can see.  Checked once, at trace.
            _gdn_fold.assert_prefix_consumed(
                fold_scope, label="compiled fixed-M4 verify"
            )
            logits, hidden, captures = result
            # (3) Read every leaf back out and return it explicitly.
            captures_flat: list[Any] = []
            for idx, kind, _n in spec:
                if kind != VERIFY_SPEC_KIND_GDN:
                    continue
                layer_capture = captures[idx]
                for key_name in layout:
                    captures_flat.append(layer_capture[key_name])
            for idx, names in live._extra_capture_layout:
                layer_capture = captures[idx]
                for key_name in names:
                    captures_flat.append(layer_capture[key_name])
            state_out: list[Any] = []
            for idx, kind, _n in spec:
                entry = shadow[idx]
                if kind == VERIFY_SPEC_KIND_QSA:
                    state_out.extend(entry.state_leaves)
                elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                    state_out.extend(entry.cache[slot] for slot in range(_n))
                else:
                    state_out.extend(entry.cache[slot] for slot in range(_n))
            if return_compiled_aux:
                return (logits, hidden, compiled_aux, *captures_flat, *state_out)
            return (logits, hidden, *captures_flat, *state_out)

        return verify_step

    def _capture_layout(self) -> tuple[str, ...]:
        if self._capture_layout_override is not None:
            return self._capture_layout_override
        if self.capture_backend == "linear_gdn_from_conv_tape":
            return TAPE_CAPTURE_KEYS
        return STANDARD_CAPTURE_KEYS

    def _unpack_outputs(self, outputs):
        spec = self._spec or []
        layout = self._capture_layout()
        n_captures = sum(
            len(layout) for _idx, kind, _n in spec if kind == VERIFY_SPEC_KIND_GDN
        )
        n_captures += sum(len(names) for _idx, names in self._extra_capture_layout)
        n_state = sum(n for _idx, _kind, n in spec)
        expected = 2 + n_captures + n_state
        if len(outputs) != expected:
            raise ValueError(
                f"compiled verify returned {len(outputs)} leaves, expected {expected}"
            )
        logits = outputs[0]
        hidden = outputs[1]
        captures_flat = list(outputs[2 : 2 + n_captures])
        state_out = list(outputs[2 + n_captures :])
        return logits, hidden, captures_flat, state_out

    def _rebuild_captures(self, captures_flat: list[Any]) -> dict[int, dict[str, Any]]:
        layout = self._capture_layout()
        captures: dict[int, dict[str, Any]] = {}
        pos = 0
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_GDN:
                continue
            layer_capture = {
                key_name: captures_flat[pos + key_pos]
                for key_pos, key_name in enumerate(layout)
            }
            pos += len(layout)
            if self.capture_backend == "linear_gdn_from_conv_tape":
                layer_capture["gdn_meta"] = self._gdn_meta(idx)
            captures[idx] = layer_capture
        for idx, names in self._extra_capture_layout:
            layer_capture = captures.setdefault(idx, {})
            for key_name in names:
                layer_capture[key_name] = captures_flat[pos]
                pos += 1
        return captures

    def _gdn_meta(self, layer_idx: int) -> dict[str, int] | None:
        if layer_idx in self._gdn_meta_cache:
            return self._gdn_meta_cache[layer_idx]
        meta: dict[str, int] | None = None
        try:
            from .gdn_capture import _gdn_tape_meta

            model = getattr(self.runtime, "model", None)
            text_model = getattr(model, "language_model", model)
            inner = getattr(text_model, "model", None)
            layer = inner.layers[layer_idx]
            meta = _gdn_tape_meta(layer.linear_attn)
        except Exception:
            meta = None
        self._gdn_meta_cache[layer_idx] = meta
        return meta

    # -- state movement -----------------------------------------------------------

    def _read_state_leaves(self, cache: Any) -> list[Any] | None:
        leaves: list[Any] = []
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                layer_leaves = tuple(entry.state_leaves)
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                layer_leaves = tuple(entry.cache[slot] for slot in range(_n))
            else:
                layer_leaves = tuple(entry.cache[slot] for slot in range(_n))
            if any(leaf is None for leaf in layer_leaves):
                return None
            leaves.extend(layer_leaves)
        return leaves

    def _mirror_commit(self, cache: Any, state_out: list[Any]) -> None:
        pos = 0
        for idx, kind, n_leaves in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                entry.kv.cache[0] = state_out[pos]
                entry.kv.cache[1] = state_out[pos + 1]
                entry.kv.cache[2] = state_out[pos + 2]
                entry.raw_keys = state_out[pos + 3]
                entry.pooled = state_out[pos + 4]
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[pos + slot]
                # Cleared rollback forces trim() onto the offset-only branch,
                # which is the correct reject semantics for a batched verify.
                for slot in range(len(entry.rollback_state)):
                    entry.rollback_state[slot] = None
            else:
                for slot in range(n_leaves):
                    entry.cache[slot] = state_out[pos + slot]
            pos += n_leaves

    def _clear_shadow_leaf_refs(self) -> None:
        """Drop leaf references held by the shadow twins (A2.1 donation).

        The traced verify body re-seeds every shadow slot from the explicit
        inputs before any read, so whatever the twins hold between calls —
        promotion-time leaves right after ``_ensure_shadow``, stale tracers
        after a trace — is dead weight.  Promotion-time refs additionally
        alias the first call's input buffers, which would block their
        donation and pin one full stale KV/GDN buffer set for the whole
        generation.
        """
        for entry in self._shadow or []:
            if entry is None:
                continue
            if isinstance(entry, TensorOffsetQSACache):
                for slot in range(len(entry.kv.cache)):
                    entry.kv.cache[slot] = None
                for slot in range(len(entry.kv.rollback_state)):
                    entry.kv.rollback_state[slot] = None
                entry.raw_keys = None
                entry.pooled = None
                continue
            cache_list = getattr(entry, "cache", None)
            if isinstance(cache_list, list):
                for slot in range(len(cache_list)):
                    cache_list[slot] = None
            rollback = getattr(entry, "rollback_state", None)
            if isinstance(rollback, list):
                for slot in range(len(rollback)):
                    rollback[slot] = None

    # -- eager paths ---------------------------------------------------------------

    def _runtime_forward(
        self,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        compiled_aux=None,
    ):
        kwargs = (
            {"compiled_aux": compiled_aux}
            if self._runtime_accepts_compiled_aux
            else {}
        )
        if self._capture_accepts_backend:
            return self.runtime.forward_ar_capture(
                input_ids,
                cache=cache,
                return_hidden=return_hidden,
                hidden_variant=hidden_variant,
                capture_backend=self.capture_backend,
                **kwargs,
            )
        return self.runtime.forward_ar_capture(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            **kwargs,
        )

    def _fallback(
        self,
        input_ids,
        *,
        cache,
        return_hidden: bool,
        hidden_variant: str | None,
        reason: str,
    ):
        if self.strict_no_fallback:
            raise RuntimeError(f"qwen4 fixed-M4 verifier refused: {reason}")
        self.stats["fallback_calls"] += 1
        self.stats["fallback_reasons"][reason] = (
            self.stats["fallback_reasons"].get(reason, 0) + 1
        )
        return self._runtime_forward(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
        )

    def _parity_check(
        self,
        input_ids,
        *,
        cache,
        hidden_variant: str | None,
        compiled_aux,
        state_in: list[Any],
        compiled_logits,
        compiled_hidden,
        compiled_captures,
        compiled_state_out,
    ):
        """Double-run: compiled pure step already ran; eager is authoritative."""
        self.stats["parity_checks"] += 1
        with attention_phase("decode_verify"):
            eager_logits, eager_hidden, eager_captures = self._runtime_forward(
                input_ids,
                cache=cache,
                return_hidden=True,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
            )
        eager_state = []
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                eager_state.extend(entry.state_leaves)
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                eager_state.extend(entry.cache[slot] for slot in range(_n))
            else:
                eager_state.extend(entry.cache[slot] for slot in range(_n))
        reference = self._named_outputs(eager_logits, eager_hidden, eager_captures, eager_state)
        candidate = self._named_outputs(
            compiled_logits, compiled_hidden, compiled_captures, compiled_state_out
        )
        report = compare_verify_outputs(reference, candidate)
        if report:
            self.stats["parity_failures"] += 1
            raise CompiledVerifyParityError(report)
        return eager_logits, eager_hidden, eager_captures

    def _parity2_check(
        self,
        input_ids,
        *,
        cache,
        hidden_variant: str | None,
        compiled_aux,
        bucket: int,
        compiled_logits,
        compiled_hidden,
        compiled_captures,
        compiled_state_out,
    ):
        """Inverted parity: COMPILED is authoritative; an eager CLONE tracks it.

        Parity mode #1 proved per-call bit-exactness at fixed contexts, but its
        eager leg re-commits the real cache on every call, so compiled-committed
        state never compounds across steps — exactly the multi-step evolution
        the live-stream fork hypothesis points at.  Here the real stream keeps
        running on the compiled mirror-commit, and the eager reference replays
        the same single step on a fresh leaf-copy clone of the pre-step cache.
        The clone is rebuilt from the real entries every call, so accept-path
        commits/trims on the real cache between calls can never drift the clone
        structurally: each comparison is one verify step given identical
        (compiled-committed) inputs.  A mismatch is logged and counted — never
        raised — so streaming continues compiled-authoritative.
        """
        self.stats["parity2_calls"] += 1
        # Seed the clone BEFORE mirror-commit: the real entries still hold the
        # pre-step leaves here (the compiled step ran purely on the shadow).
        clone = self._parity2_clone_cache(cache, bucket)
        with attention_phase("decode_verify"):
            eager_logits, eager_hidden, eager_captures = self._runtime_forward(
                input_ids,
                cache=clone,
                return_hidden=True,
                hidden_variant=hidden_variant,
                compiled_aux=compiled_aux,
            )
        # Compiled is authoritative: the live stream advances on compiled state.
        self._mirror_commit(cache, compiled_state_out)
        clone_state: list[Any] = []
        for idx, kind, _n in self._spec or []:
            entry = clone[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                clone_state.extend(entry.state_leaves)
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                clone_state.extend(entry.cache[slot] for slot in range(_n))
            else:
                clone_state.extend(entry.cache[slot] for slot in range(_n))
        reference = self._named_outputs(
            eager_logits, eager_hidden, eager_captures, clone_state
        )
        candidate = self._named_outputs(
            compiled_logits, compiled_hidden, compiled_captures, compiled_state_out
        )
        # Uncapped compare so mismatched_leaves is a true count, not a preview.
        report = compare_verify_outputs(
            reference,
            candidate,
            max_report_lines=len(reference) + len(candidate) + 8,
        )
        if report:
            self._record_parity2_divergence(report, reference, candidate, cache)
        return compiled_logits, compiled_hidden, compiled_captures

    def _parity2_clone_cache(self, cache: Any, bucket: int) -> list[Any]:
        """Fresh eager-leg clone: real container classes over leaf COPIES.

        Mirrors ``_ensure_shadow``'s twin construction but with materialized
        ``mx.array`` copies instead of shared refs, so the eager forward's
        writes (functional slice_updates and slot reassignments) can never
        interact with the buffers the compiled-authoritative stream holds.
        """
        from .cache_state import (
            TensorOffsetQuantizedPagedKVCache,
            TensorOffsetVllmMetalPagedKVCache,
        )

        clone: list[Any] = [None] * len(cache)
        for idx, kind, _n in self._spec or []:
            entry = cache[idx]
            if kind == VERIFY_SPEC_KIND_QSA:
                kv = TensorOffsetKVCache(
                    _copy_state_leaf(entry.kv.cache[0]),
                    _copy_state_leaf(entry.kv.cache[1]),
                    _copy_state_leaf(entry.kv.cache[2]),
                    step=entry.kv.step,
                )
                twin = TensorOffsetQSACache(
                    kv,
                    _copy_state_leaf(entry.raw_keys),
                    _copy_state_leaf(entry.pooled),
                    compress_ratio=entry.ratio,
                    rows_gather=entry.fixed_rows_gather,
                    rows_gather_kv_m4=entry.rows_gather_kv_m4,
                    rows_gather_enabled=entry.rows_gather_enabled,
                    rows_gather_min_context=entry.rows_gather_min_context,
                    fused_rows_gather_kv_m4=entry.fused_rows_gather_kv_m4,
                    # A twin that dropped these would silently revert to
                    # the stock QSA chain -- the armed-but-inert failure mode.
                    fable_qsa_m4=entry.fable_qsa_m4,
                    fable_qsa_m4_kt=entry.fable_qsa_m4_kt,
                    fable_qsa_sparse_decode=getattr(
                        entry, "fable_qsa_sparse_decode", False
                    ),
                    fable_qsa_sparse_draft=getattr(
                        entry, "fable_qsa_sparse_draft", False
                    ),
                )
            elif kind == VERIFY_SPEC_KIND_FULL_ATTN:
                if isinstance(entry, TensorOffsetKVCache):
                    twin = TensorOffsetKVCache(
                        _copy_state_leaf(entry.cache[0]),
                        _copy_state_leaf(entry.cache[1]),
                        _copy_state_leaf(entry.cache[2]),
                        step=entry.step,
                    )
                elif isinstance(entry, TensorOffsetQuantizedPagedKVCache):
                    twin = TensorOffsetQuantizedPagedKVCache(
                        key_cache=_copy_state_leaf(entry.cache[0]),
                        value_cache=_copy_state_leaf(entry.cache[1]),
                        offset=_copy_state_leaf(entry.cache[2]),
                        key_scale_cache=_copy_state_leaf(entry.cache[3]),
                        value_scale_cache=_copy_state_leaf(entry.cache[4]),
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                        kv_quant_config=entry.kv_quant_config,
                        source_dtypes=entry.source_dtypes,
                        head_dims=entry.head_dims,
                    )
                else:
                    twin = TensorOffsetVllmMetalPagedKVCache(
                        key_cache=_copy_state_leaf(entry.cache[0]),
                        value_cache=_copy_state_leaf(entry.cache[1]),
                        offset=_copy_state_leaf(entry.cache[2]),
                        block_size=entry.block_size,
                        num_blocks=entry.num_blocks,
                    )
                if bucket and hasattr(twin, "static_max_offset"):
                    # Same static ceiling as the real/shadow entries so the
                    # eager paged kernel runs the identical reduction topology
                    # (what makes bit-exact comparison meaningful).
                    twin.static_max_offset = int(bucket)
            else:
                twin = type(entry)(len(entry.cache))
                for slot, leaf in enumerate(entry.cache[:_n]):
                    twin[slot] = _copy_state_leaf(leaf)
            clone[idx] = twin
        return clone

    def _record_parity2_divergence(
        self,
        report: list[str],
        reference: dict[str, Any],
        candidate: dict[str, Any],
        cache: Any,
    ) -> None:
        self.stats["parity2_divergent_calls"] += 1
        ordinal = int(self.stats["calls"])
        context = self._parity2_context_estimate(cache)
        # Split on ": " (not ":"): state leaf names embed a colon, e.g.
        # "state[1:fa].2: value mismatch (...)".
        first_name = report[0].split(": ", 1)[0]
        artifact = _artifact_kind(first_name)
        max_abs = _leaf_max_abs_diff(
            reference.get(first_name), candidate.get(first_name)
        )
        mismatched = sum(1 for line in report if not line.startswith("... "))
        record = {
            "call": ordinal,
            "context": context,
            "artifact": artifact,
            "leaf": first_name,
            "max_abs_diff": max_abs,
            "mismatched_leaves": mismatched,
        }
        if self.stats["parity2_first_divergence"] is None:
            self.stats["parity2_first_divergence"] = record
        count = int(self.stats["parity2_divergent_calls"])
        if count <= 10:
            max_abs_text = "n/a" if max_abs is None else f"{max_abs:.3e}"
            print(
                f"[parity2] divergence call={ordinal} context={context} "
                f"artifact={artifact} leaf={first_name} "
                f"max_abs_diff={max_abs_text} mismatched_leaves={mismatched}",
                flush=True,
            )
            if count == 10:
                print(
                    "[parity2] divergence log cap reached (10); further "
                    "divergent calls are counted in stats only "
                    "(parity2_divergent_calls)",
                    flush=True,
                )

    def _parity2_context_estimate(self, cache: Any) -> int:
        """Context/offset estimate for divergence reports (tokens).

        Paged entries already produced offset+M in ``_resolve_bucket``; dense
        adapters (no ``capacity``) fall through to the post-commit offset.
        Best-effort diagnostics only — never load-bearing.
        """
        estimate = int(getattr(self, "_last_context_estimate", 0) or 0)
        if estimate:
            return estimate
        best = 0
        for idx, kind, _n in self._spec or []:
            if kind != VERIFY_SPEC_KIND_FULL_ATTN:
                continue
            entry = cache[idx]
            try:
                best = max(best, int(entry.size()))
            except Exception:
                continue
        return best

    def _named_outputs(
        self,
        logits,
        hidden,
        captures: dict[int, dict[str, Any]],
        state_leaves: list[Any],
    ) -> dict[str, Any]:
        named: dict[str, Any] = {"logits": logits, "hidden": hidden}
        layout = self._capture_layout()
        for layer_idx in sorted(k for k in captures if isinstance(k, int)):
            layer_capture = captures[layer_idx]
            for key_name in layout:
                named[f"capture[{layer_idx}].{key_name}"] = layer_capture.get(key_name)
        pos = 0
        for idx, kind, n_leaves in self._spec or []:
            for leaf_idx in range(n_leaves):
                named[f"state[{idx}:{kind}].{leaf_idx}"] = state_leaves[pos + leaf_idx]
            pos += n_leaves
        return named
