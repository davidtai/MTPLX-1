"""Construction-bound fixed-M4 capture surface for Qwen4 Flash-Next."""

from __future__ import annotations

import os
from functools import partial
from types import MethodType
from typing import Any

import mlx.core as mx
import numpy as np

from . import ple_boundary, ple_candidate_prefetch
from .models.qwen4_exp import (
    _ngram_rows_np,
    compiled_verify_ple_scope,
    verify_capture_disabled_scope,
    verify_capture_scope,
)

_GDN_ROW_NAMES = ("qkv", "q", "k", "v", "a", "b")
_PLE_ROW_NAMES = ("ple_hidden", "ple_ids", "ple_conv_rows")


def is_qwen4_fixed_verify_config(config: dict[str, Any]) -> bool:
    """Identify the one measured Flash-Next fixed-M4 model geometry."""

    text = config.get("text_config")
    if not isinstance(text, dict):
        return False
    observed = (
        config.get("model_type"),
        text.get("model_type"),
        text.get("hidden_size"),
        text.get("num_hidden_layers"),
        text.get("hc_count"),
        text.get("hc_lowrank"),
        text.get("indexer_compress_ratio"),
        text.get("linear_num_key_heads"),
        text.get("linear_num_value_heads"),
        text.get("linear_key_head_dim"),
        text.get("linear_value_head_dim"),
        text.get("ple_layer_ids"),
        text.get("ngram_size"),
        text.get("ngram_vocab_size_base"),
        text.get("heads_per_ngram"),
        text.get("ple_embed_dim"),
        text.get("ngram_sidecar"),
        text.get("num_experts"),
        text.get("num_experts_per_tok"),
        text.get("moe_intermediate_size"),
        text.get("vocab_size"),
    )
    return observed == (
        "qwen4_exp",
        "qwen4_exp_text",
        2560,
        48,
        4,
        320,
        4,
        16,
        48,
        128,
        128,
        [2],
        3,
        20_000_000,
        8,
        2560,
        True,
        512,
        10,
        640,
        248_320,
    )


def qwen4_fixed_verify_enabled() -> bool:
    raw = os.environ.get("MTPLX_QWEN4_FIXED_M4_VERIFY", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_FABLE_COMPACT_COMMIT: bool | None = None


def fable_compact_commit_enabled() -> bool:
    """Read the compact width-a commit switch exactly once per process.

    The control and candidate arms must be the same binary, so the switch is
    an environment read that is cached at first use and never re-read inside a
    measured cycle.
    """

    global _FABLE_COMPACT_COMMIT
    if _FABLE_COMPACT_COMMIT is None:
        raw = os.environ.get("MTPLX_FABLE_COMPACT_COMMIT", "0").strip().lower()
        _FABLE_COMPACT_COMMIT = raw in {"1", "true", "yes", "on"}
    return _FABLE_COMPACT_COMMIT


def _text_model(runtime: Any):
    return getattr(runtime.model, "language_model", runtime.model)


def _inner(runtime: Any):
    return _text_model(runtime).model


def _prepare_compiled_verify_aux(self: Any, input_ids, cache) -> mx.array:
    """Materialize the one host-backed PLE embedding before graph dispatch."""

    inner = _inner(self)
    layer_index = int(inner._ple_stage_idx)
    ple = inner.layers[layer_index].ple
    entry = cache[layer_index]
    previous = entry[ple.NGRAM_IDX]
    embedding = ple.ple_embedding._graph_path(
        input_ids,
        None,
        ple.NGRAM_IDX,
        prev=previous,
    )
    # The SSD sidecar returns a lazy dequantization graph whose inputs are
    # host-acquired row payloads.  It must cross the materialization boundary
    # before becoming an explicit input to mx.compile; otherwise MLX sees the
    # row graph's cache leaves as uncaptured inputs to the verifier graph.
    mx.eval(embedding)
    return embedding


def _bind_fixed_m4_owned_row_prefetch(sidecar, *, all_miss: bool = False):
    """Bind the fixed-M4 pread payload handoff to one sidecar instance."""

    pool = sidecar._pool
    if pool is None:
        raise ValueError("qwen4 fixed-M4 row prefetch requires a worker pool")
    hot_capacity = int(sidecar._hot_cap_rows)
    required_hot_rows = 64 if all_miss else 16
    if hot_capacity < required_hot_rows:
        raise ValueError(
            "qwen4 fixed-M4 row prefetch requires hot capacity for "
            f"{required_hot_rows} rows"
        )

    fd = sidecar._fd
    hot = sidecar._hot
    specs = []
    for name in ("weight", "scales", "biases"):
        matrix = sidecar._maps[name][0]
        row_shape = tuple(matrix.shape[1:])
        row_count = int(np.prod(row_shape))
        row_bytes = row_count * int(matrix.dtype.itemsize)
        specs.append(
            (
                int(matrix.offset),
                row_bytes,
                matrix.dtype,
                row_count,
                row_shape,
            )
        )
    specs = tuple(specs)

    def fetch(row):
        row = int(row)
        payload = tuple(
            np.frombuffer(
                os.pread(fd, row_bytes, base + row * row_bytes),
                dtype=dtype,
                count=row_count,
            ).reshape(row_shape)
            for base, row_bytes, dtype, row_count, row_shape in specs
        )
        return row, payload

    def submit_primary(rows):
        return tuple(pool.submit(fetch, int(row)) for row in rows)

    def submit_missing(rows):
        requested = []
        seen = set()
        for value in np.asarray(rows).reshape(-1):
            row = int(value)
            if row in seen:
                continue
            seen.add(row)
            if row in hot:
                hot.move_to_end(row)
            else:
                requested.append(row)
        return tuple(pool.submit(fetch, row) for row in requested)

    def install(pending):
        for future in pending:
            row, payload = future.result()
            hot[row] = payload
            hot.move_to_end(row)
        while len(hot) > hot_capacity:
            hot.popitem(last=False)

    if not all_miss:

        def submit_missing(_rows):
            return ()

    return submit_primary, submit_missing, install


def _bind_fixed_m4_window_prefetch(
    *,
    prompt_tail,
    rows,
    submit_window,
    install_owned_rows,
    enabled: bool,
):
    """Bind either an early staged window or the unchanged inline resolver."""

    def resolve_rows(host_input_ids, completion_tokens, committed_count):
        ids_np = np.asarray((host_input_ids,), dtype=np.int64)
        previous = _fixed_m4_previous_tokens(
            prompt_tail,
            completion_tokens,
            committed_count,
        )
        prev_np = np.asarray((previous,), dtype=np.int64)
        resolved, _new_history = rows(ids_np, prev_np)
        return resolved.reshape(-1)

    if enabled:
        pending = [None]

        def prefetch(host_input_ids, completion_tokens, committed_count):
            resolved = resolve_rows(
                host_input_ids,
                completion_tokens,
                committed_count,
            )
            pending[0] = (resolved, submit_window(resolved))

        def consume(_host_input_ids, _completion_tokens, _committed_count):
            resolved, futures = pending[0]
            pending[0] = None
            install_owned_rows(futures)
            return resolved

        return prefetch, consume

    def prefetch(_host_input_ids, _completion_tokens, _committed_count):
        return None

    return prefetch, resolve_rows


class _FixedM4SidecarAux:
    """Monomorphic materialized host gather for the retained M4 route."""

    __slots__ = (
        "_gather",
        "_install_owned_rows",
        "_output_dim",
        "_pending_warm",
        "_prompt_tail",
        "_rows",
        "_submit_warm",
    )

    def __init__(
        self,
        *,
        prompt_tail,
        rows,
        gather,
        output_dim,
        submit_warm,
        install_owned_rows,
    ):
        self._prompt_tail = prompt_tail
        self._rows = rows
        self._gather = gather
        self._output_dim = int(output_dim)
        self._submit_warm = submit_warm
        self._install_owned_rows = install_owned_rows
        self._pending_warm = ()

    def prefetch_primary(
        self,
        primary,
        completion_tokens,
        committed_count,
    ) -> None:
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        ids_np = np.asarray(((int(primary),),), dtype=np.int64)
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        self._pending_warm = self._submit_warm(rows.reshape(-1))

    def __call__(
        self,
        _input_ids,
        host_input_ids,
        completion_tokens,
        committed_count,
    ) -> mx.array:
        pending_warm = self._pending_warm
        self._pending_warm = ()
        self._install_owned_rows(pending_warm)
        ids_np = np.asarray((host_input_ids,), dtype=np.int64)
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        return self._gather(rows.reshape(-1)).reshape(
            1,
            4,
            self._output_dim,
        )


class _FixedM4CandidateSidecarAux:
    """MTPLX_FABLE_PLE_CANDIDATE_PREFETCH's aux (K-P1).  Default off.

    A SEPARATE class rather than a branch inside ``_FixedM4SidecarAux``: the
    retained aux's ``__call__`` is deliberately monomorphic (see
    ``test_retained_materialized_aux_is_monomorphic_in_the_hot_path``), and an
    A/B whose control arm's hot path has been edited is not a control.  The
    lane's own call is monomorphic too -- the miss branch takes exactly the
    shipped expression.
    """

    __slots__ = (
        "_candidates",
        "_gather",
        "_install_owned_rows",
        "_output_dim",
        "_pending_warm",
        "_prompt_tail",
        "_rows",
        "_submit_warm",
    )

    def __init__(
        self,
        *,
        prompt_tail,
        rows,
        gather,
        output_dim,
        submit_warm,
        install_owned_rows,
        candidates,
    ):
        self._prompt_tail = prompt_tail
        self._rows = rows
        self._gather = gather
        self._output_dim = int(output_dim)
        self._submit_warm = submit_warm
        self._install_owned_rows = install_owned_rows
        self._pending_warm = ()
        self._candidates = candidates

    @property
    def candidate_prefetch(self):
        """The armed K-P1 buffer.  The draft loop's hook reaches it here."""

        return self._candidates

    def prefetch_primary(
        self,
        primary,
        completion_tokens,
        committed_count,
    ) -> None:
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        ids_np = np.asarray(((int(primary),),), dtype=np.int64)
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        self._pending_warm = self._submit_warm(rows.reshape(-1))
        # A cycle boundary: join and drop the previous cycle's buffer, then
        # seed it with the one window position that is not a candidate.  The
        # `_submit_warm` above still runs, so the hot-row LRU fallback keeps
        # exactly the coverage it has today.
        self._candidates.begin_cycle()
        self._candidates.submit(
            prefix_tokens=(),
            candidate_ids=(int(primary),),
            completion_tokens=completion_tokens,
            committed_count=committed_count,
        )

    def __call__(
        self,
        _input_ids,
        host_input_ids,
        completion_tokens,
        committed_count,
    ) -> mx.array:
        pending_warm = self._pending_warm
        self._pending_warm = ()
        self._install_owned_rows(pending_warm)
        ids_np = np.asarray((host_input_ids,), dtype=np.int64)
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        flat = rows.reshape(-1)
        prepared = self._candidates.resolve(flat)
        if prepared is None:
            return self._gather(flat).reshape(1, 4, self._output_dim)
        # The same `prepared` contract the prefill lookahead uses: row matrices
        # this lane already read out of the same three memmaps, so the only
        # thing that changed is WHEN they were read.
        return self._gather(flat, prepared=prepared).reshape(
            1,
            4,
            self._output_dim,
        )


class _FixedM4ExperimentalSidecarAux:
    """Construction-bound raw/window gather retained for composition tests."""

    __slots__ = (
        "_gather",
        "_install_owned_rows",
        "_pending_warm",
        "_prefetch_window_rows",
        "_prompt_tail",
        "_resolve_window_rows",
        "_rows",
        "_submit_warm",
    )

    def __init__(
        self,
        *,
        prompt_tail,
        rows,
        gather,
        submit_warm,
        install_owned_rows,
        prefetch_window_rows,
        resolve_window_rows,
    ):
        self._prompt_tail = prompt_tail
        self._rows = rows
        self._gather = gather
        self._submit_warm = submit_warm
        self._install_owned_rows = install_owned_rows
        self._prefetch_window_rows = prefetch_window_rows
        self._resolve_window_rows = resolve_window_rows
        self._pending_warm = ()

    def prefetch_primary(
        self,
        primary,
        completion_tokens,
        committed_count,
    ) -> None:
        previous = _fixed_m4_previous_tokens(
            self._prompt_tail,
            completion_tokens,
            committed_count,
        )
        ids_np = np.asarray(((int(primary),),), dtype=np.int64)
        prev_np = np.asarray((previous,), dtype=np.int64)
        rows, _new_history = self._rows(ids_np, prev_np)
        self._pending_warm = self._submit_warm(rows.reshape(-1))

    def prefetch_window(
        self,
        host_input_ids,
        completion_tokens,
        committed_count,
    ) -> None:
        self._prefetch_window_rows(
            host_input_ids,
            completion_tokens,
            committed_count,
        )

    def __call__(
        self,
        _input_ids,
        host_input_ids,
        completion_tokens,
        committed_count,
    ) -> tuple[mx.array, mx.array, mx.array]:
        pending_warm = self._pending_warm
        self._pending_warm = ()
        self._install_owned_rows(pending_warm)
        rows = self._resolve_window_rows(
            host_input_ids,
            completion_tokens,
            committed_count,
        )
        return self._gather(rows)


def _dequantize_fixed_m4_ple(raw_aux, *, output_dim: int) -> mx.array:
    """Expand the exact q4 sidecar payload inside the compiled M4 graph."""

    weight, scales, biases = raw_aux
    return mx.dequantize(
        weight,
        scales,
        biases,
        group_size=32,
        bits=4,
    ).reshape(1, 4, output_dim)


def _gather_fixed_m4_materialized(flat, *, gather, output_dim: int) -> mx.array:
    """Restore the physical 64-row gather to the logical M=4 PLE window."""

    return gather(flat).reshape(1, 4, output_dim)


def _fixed_m4_previous_tokens(
    prompt_tail: tuple[int, int], completion_tokens, committed_count: int
) -> tuple[int, int]:
    """Project the last two cache-owned tokens from the host token ledger."""

    if committed_count >= 2:
        return (
            int(completion_tokens[committed_count - 2]),
            int(completion_tokens[committed_count - 1]),
        )
    if committed_count == 1:
        return int(prompt_tail[1]), int(completion_tokens[0])
    return prompt_tail


def _build_fixed_m4_compiled_verify_aux(
    self: Any,
    cache,
    prompt_ids,
    *,
    raw_q4_aux: bool = True,
    owned_all_miss_rows: bool = False,
    early_window_prefetch: bool = False,
):
    """Validate and bind the production sidecar gather once after prefill."""

    inner = _inner(self)
    layer_index = int(inner._ple_stage_idx)
    ple = inner.layers[layer_index].ple
    embedding = ple.ple_embedding
    sidecar = embedding.ngram_embedding._sidecar
    entry = cache[layer_index]
    previous = entry[ple.NGRAM_IDX]
    observed = (
        int(embedding.context_len),
        int(embedding.ngram_size),
        int(embedding.heads_per_ngram),
        tuple(previous.shape),
        previous.dtype,
        int(inner.args.ple_embed_dim),
    )
    production = int(inner.args.hidden_size) == 2560
    exact = observed == (2, 3, 8, (1, 2), mx.int64, 2560)
    internally_consistent = (
        observed[0:2] == (2, 3)
        and observed[2] > 0
        and observed[3] == (1, 2)
        and observed[4] == mx.int64
        and observed[5] == int(inner.args.hidden_size)
    )
    if sidecar is None or (not exact if production else not internally_consistent):
        raise ValueError(
            f"qwen4 fixed-M4 sidecar auxiliary geometry mismatch: {observed}"
        )
    # MTPLX_FABLE_PLE_BOUNDARY items `warm_skip` / `hot_block` / `timing`
    # (default off).  Bound once per request, AFTER the geometry gate above:
    # an unvalidated sidecar must never get a replaced gather.  Returns None
    # and touches nothing when no item is armed.
    _boundary_engagement = ple_boundary.bind_sidecar(sidecar)
    if _boundary_engagement is not None:
        print(_boundary_engagement, flush=True)
    mult, sizes, offs = embedding._np_consts()
    const_shapes = tuple(tuple(value.shape) for value in (mult, sizes, offs))
    if const_shapes != ((3,), (2 * observed[2],), (2 * observed[2],)):
        raise ValueError(
            f"qwen4 fixed-M4 sidecar auxiliary constants mismatch: {const_shapes}"
        )
    if production:
        storage = (
            int(sidecar.bits),
            int(sidecar.group_size),
            tuple(
                (name, tuple(sidecar._maps[name][0].shape[1:]), sidecar._maps[name][1])
                for name in ("weight", "scales", "biases")
            ),
        )
        expected_storage = (
            4,
            32,
            (
                ("weight", (20,), "U32"),
                ("scales", (5,), "BF16"),
                ("biases", (5,), "BF16"),
            ),
        )
        if storage != expected_storage:
            raise ValueError(
                f"qwen4 fixed-M4 sidecar auxiliary storage mismatch: {storage}"
            )
    prompt_tail = tuple(
        [int(embedding.eos_id), int(embedding.eos_id)]
        + [int(token) for token in prompt_ids[-2:]]
    )[-2:]
    device_history = tuple(
        int(token)
        for token in np.asarray(previous, dtype=np.int64).reshape(-1)
    )
    if device_history != prompt_tail:
        raise ValueError(
            "qwen4 fixed-M4 prompt history does not match the prefetched cache"
        )
    rows = partial(
        _ngram_rows_np,
        mult=mult,
        sizes=sizes,
        offs=offs,
        eos=int(embedding.eos_id),
        ngram_size=int(embedding.ngram_size),
        heads_per_ngram=int(embedding.heads_per_ngram),
    )
    submit_owned_rows, submit_window_rows, install_owned_rows = (
        _bind_fixed_m4_owned_row_prefetch(
            sidecar,
            all_miss=owned_all_miss_rows,
        )
    )
    # MTPLX_FABLE_PLE_BOUNDARY item `primary_vectorized` (default off).  The
    # swap is here and not inside `_bind_fixed_m4_owned_row_prefetch` so that
    # the shipped binder stays a closed function of `(sidecar, all_miss)` --
    # `tests/test_qwen4_fixed_host_tokens_static.py` executes its SOURCE, and
    # a lane reference in it would break that check on the control arm.  With
    # the item disarmed the two names below are rebound to the same objects.
    submit_owned_rows, install_owned_rows, _boundary_prefetch_engagement = (
        ple_boundary.bind_owned_row_prefetch(
            sidecar,
            submit_primary=submit_owned_rows,
            install=install_owned_rows,
            names=("weight", "scales", "biases"),
        )
    )
    if _boundary_prefetch_engagement is not None:
        print(_boundary_prefetch_engagement, flush=True)
    prefetch_window_rows, resolve_window_rows = _bind_fixed_m4_window_prefetch(
        prompt_tail=prompt_tail,
        rows=rows,
        submit_window=submit_window_rows,
        install_owned_rows=install_owned_rows,
        enabled=early_window_prefetch,
    )
    gather = sidecar.gather_raw_np
    if not raw_q4_aux:
        gather = partial(
            _gather_fixed_m4_materialized,
            gather=sidecar.gather_np,
            output_dim=int(inner.args.ple_embed_dim),
        )
    if not raw_q4_aux and not owned_all_miss_rows and not early_window_prefetch:
        if ple_candidate_prefetch.enabled():
            return _FixedM4CandidateSidecarAux(
                prompt_tail=prompt_tail,
                rows=rows,
                gather=sidecar.gather_np,
                output_dim=int(inner.args.ple_embed_dim),
                submit_warm=submit_owned_rows,
                install_owned_rows=install_owned_rows,
                candidates=ple_candidate_prefetch.CandidateRowPrefetch(
                    rows=rows,
                    sidecar=sidecar,
                    prompt_tail=prompt_tail,
                ),
            )
        return _FixedM4SidecarAux(
            prompt_tail=prompt_tail,
            rows=rows,
            gather=sidecar.gather_np,
            output_dim=int(inner.args.ple_embed_dim),
            submit_warm=submit_owned_rows,
            install_owned_rows=install_owned_rows,
        )
    if ple_candidate_prefetch.enabled():
        # The lane's consumer lives on the retained monomorphic route only.
        # Arming it against a raw/window-prefetch aux would silently measure
        # the control while wearing the candidate's label -- the exact failure
        # the 2026-09-01 lookahead arm hit -- so refuse at construction.
        raise RuntimeError(
            f"{ple_candidate_prefetch.ENV_FLAG} requires the retained "
            "materialized fixed-M4 aux (raw_q4_aux / owned_all_miss_rows / "
            "early_window_prefetch must all be off)"
        )
    return _FixedM4ExperimentalSidecarAux(
        prompt_tail=prompt_tail,
        rows=rows,
        gather=gather,
        submit_warm=submit_owned_rows,
        install_owned_rows=install_owned_rows,
        prefetch_window_rows=prefetch_window_rows,
        resolve_window_rows=resolve_window_rows,
    )


def _forward_ar_capture(
    self: Any,
    input_ids,
    *,
    cache=None,
    return_hidden: bool = True,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
    compiled_aux: mx.array | None = None,
):
    del hidden_variant, capture_backend
    text = _text_model(self)
    if cache is None:
        cache = text.make_cache()
    with verify_capture_scope(), compiled_verify_ple_scope(compiled_aux):
        logits, hidden = text(
            input_ids,
            cache=cache,
            return_hidden=True,
        )

    captures: dict[int, dict[str, mx.array]] = {}
    for index, layer in enumerate(text.model.layers):
        if not layer.is_linear:
            continue
        entry = cache[index]
        rows = getattr(entry, "_mtplx_verify_rows", None)
        if rows is None or len(rows) != len(_GDN_ROW_NAMES):
            raise RuntimeError(
                f"qwen4 fixed-M4 capture missing GDN rows at layer {index}"
            )
        layer_capture = dict(zip(_GDN_ROW_NAMES, rows))
        if getattr(layer, "ple", None) is not None:
            ple_rows = getattr(entry, "_mtplx_verify_ple", None)
            if ple_rows is None or len(ple_rows) != len(_PLE_ROW_NAMES):
                ple_base = getattr(entry, "_mtplx_verify_ple", None)
                ple_conv_rows = getattr(
                    entry, "_mtplx_verify_ple_conv_rows", None
                )
                if ple_base is None or len(ple_base) != 2 or ple_conv_rows is None:
                    raise RuntimeError(
                        f"qwen4 fixed-M4 capture missing PLE rows at layer {index}"
                    )
                ple_rows = (*ple_base, ple_conv_rows)
            layer_capture.update(zip(_PLE_ROW_NAMES, ple_rows))
        captures[index] = layer_capture
    if return_hidden:
        return logits, hidden, captures
    return logits, captures


def _forward_fixed_m4_prefix(self: Any, input_ids, *, cache):
    """Run target embedding and layer 0 for the construction-bound M4 split."""

    inner = _inner(self)
    layer = inner.layers[0]
    entry = cache[0]
    hidden = inner.embed_tokens(input_ids)
    hidden = mx.tile(hidden, (1, 1, int(inner.args.hc_count)))
    with verify_capture_scope():
        hidden = layer(
            hidden,
            input_ids=input_ids,
            ssm_mask=None,
            cache=entry,
        )
    rows = getattr(entry, "_mtplx_verify_rows", None)
    if rows is None or len(rows) != len(_GDN_ROW_NAMES):
        raise RuntimeError("qwen4 fixed-M4 prefix missing layer-0 capture rows")
    return hidden, tuple(rows)


def _forward_fixed_m4_suffix(
    self: Any,
    layer0_hidden,
    input_ids,
    *,
    cache,
    compiled_aux,
):
    """Run layers 1..47 and the target head for the fixed-M4 split."""

    text = _text_model(self)
    inner = text.model
    hidden = layer0_hidden
    captures: dict[int, dict[str, mx.array]] = {}
    with verify_capture_scope(), compiled_verify_ple_scope(compiled_aux):
        for index in range(1, len(inner.layers)):
            layer = inner.layers[index]
            entry = cache[index]
            if getattr(layer, "ple", None) is not None:
                entry._mtplx_verify_ple = (hidden, input_ids)
            hidden = layer(
                hidden,
                input_ids=input_ids,
                ssm_mask=None,
                cache=entry,
            )

    for index in range(1, len(inner.layers)):
        layer = inner.layers[index]
        if not layer.is_linear:
            continue
        entry = cache[index]
        rows = getattr(entry, "_mtplx_verify_rows", None)
        if rows is None or len(rows) != len(_GDN_ROW_NAMES):
            raise RuntimeError(
                f"qwen4 fixed-M4 suffix missing GDN rows at layer {index}"
            )
        layer_capture = dict(zip(_GDN_ROW_NAMES, rows))
        if getattr(layer, "ple", None) is not None:
            ple_rows = getattr(entry, "_mtplx_verify_ple", None)
            if ple_rows is None or len(ple_rows) != len(_PLE_ROW_NAMES):
                ple_base = getattr(entry, "_mtplx_verify_ple", None)
                ple_conv_rows = getattr(entry, "_mtplx_verify_ple_conv_rows", None)
                if ple_base is None or len(ple_base) != 2 or ple_conv_rows is None:
                    raise RuntimeError(
                        f"qwen4 fixed-M4 suffix missing PLE rows at layer {index}"
                    )
                ple_rows = (*ple_base, ple_conv_rows)
            layer_capture.update(zip(_PLE_ROW_NAMES, ple_rows))
        captures[index] = layer_capture

    logits = text._head_logits(inner.hyper_connection_mixer(hidden))
    return logits, hidden, captures


def _collect_fixed_m4_captures(
    inner: Any, cache, indices, where: str
) -> dict[int, dict[str, mx.array]]:
    """Harvest one contiguous layer range's capture rows off the cache.

    Byte-for-byte the harvest ``_forward_fixed_m4_suffix`` does, factored out
    so the W67 N-layer prefix and suffix agree with it and with each other.
    ``_forward_fixed_m4_prefix`` / ``_forward_fixed_m4_suffix`` themselves are
    PR391's and are NOT routed through this (their source is pinned).
    """

    captures: dict[int, dict[str, mx.array]] = {}
    for index in indices:
        layer = inner.layers[index]
        if not layer.is_linear:
            continue
        entry = cache[index]
        rows = getattr(entry, "_mtplx_verify_rows", None)
        if rows is None or len(rows) != len(_GDN_ROW_NAMES):
            raise RuntimeError(
                f"qwen4 fixed-M4 {where} missing GDN rows at layer {index}"
            )
        layer_capture = dict(zip(_GDN_ROW_NAMES, rows))
        if getattr(layer, "ple", None) is not None:
            ple_rows = getattr(entry, "_mtplx_verify_ple", None)
            if ple_rows is None or len(ple_rows) != len(_PLE_ROW_NAMES):
                ple_base = getattr(entry, "_mtplx_verify_ple", None)
                ple_conv_rows = getattr(entry, "_mtplx_verify_ple_conv_rows", None)
                if ple_base is None or len(ple_base) != 2 or ple_conv_rows is None:
                    raise RuntimeError(
                        f"qwen4 fixed-M4 {where} missing PLE rows at layer {index}"
                    )
                ple_rows = (*ple_base, ple_conv_rows)
            layer_capture.update(zip(_PLE_ROW_NAMES, ple_rows))
        captures[index] = layer_capture
    return captures


def _forward_fixed_m4_overlap_prefix(
    self: Any,
    input_ids,
    *,
    cache,
    layer_count: int,
    compiled_aux=None,
):
    """W67: run target embedding and layers ``0..layer_count-1``.

    The generalization of ``_forward_fixed_m4_prefix``, which stays wired to
    layer 0 for PR391's device-committed split lane.  ``compiled_aux`` is
    required exactly when the single PLE layer falls inside the prefix, and
    it is then the SAME object the suffix and the monolithic route are handed
    -- the prefix does not build its own.
    """

    inner = _inner(self)
    count = int(layer_count)
    if not 1 <= count < len(inner.layers):
        raise ValueError(
            f"qwen4 fixed-M4 overlap prefix depth {count} is outside "
            f"[1, {len(inner.layers) - 1}]"
        )
    hidden = inner.embed_tokens(input_ids)
    hidden = mx.tile(hidden, (1, 1, int(inner.args.hc_count)))
    with verify_capture_scope(), compiled_verify_ple_scope(compiled_aux):
        for index in range(count):
            layer = inner.layers[index]
            entry = cache[index]
            if getattr(layer, "ple", None) is not None:
                if compiled_aux is None:
                    raise RuntimeError(
                        "qwen4 fixed-M4 overlap prefix reaches the PLE layer "
                        f"at index {index} without a compiled auxiliary"
                    )
                entry._mtplx_verify_ple = (hidden, input_ids)
            hidden = layer(
                hidden,
                input_ids=input_ids,
                ssm_mask=None,
                cache=entry,
            )
    captures = _collect_fixed_m4_captures(
        inner, cache, range(count), "overlap prefix"
    )
    return hidden, captures


def _forward_fixed_m4_overlap_suffix(
    self: Any,
    prefix_hidden,
    input_ids,
    *,
    cache,
    compiled_aux,
    start: int,
):
    """W67: run layers ``start..47`` and the target head on a prefix hidden."""

    text = _text_model(self)
    inner = text.model
    first = int(start)
    if not 1 <= first < len(inner.layers):
        raise ValueError(
            f"qwen4 fixed-M4 overlap suffix start {first} is outside "
            f"[1, {len(inner.layers) - 1}]"
        )
    hidden = prefix_hidden
    with verify_capture_scope(), compiled_verify_ple_scope(compiled_aux):
        for index in range(first, len(inner.layers)):
            layer = inner.layers[index]
            entry = cache[index]
            if getattr(layer, "ple", None) is not None:
                entry._mtplx_verify_ple = (hidden, input_ids)
            hidden = layer(
                hidden,
                input_ids=input_ids,
                ssm_mask=None,
                cache=entry,
            )

    captures = _collect_fixed_m4_captures(
        inner, cache, range(first, len(inner.layers)), "overlap suffix"
    )
    logits = text._head_logits(inner.hyper_connection_mixer(hidden))
    return logits, hidden, captures


def _commit_compiled_verify_captures(
    self: Any, cache, captures: dict[int, dict[str, mx.array]]
) -> None:
    """Install materialized capture leaves on the live family cache."""

    for index, layer in enumerate(_inner(self).layers):
        if not layer.is_linear:
            continue
        layer_capture = captures[index]
        cache[index]._mtplx_verify_rows = tuple(
            layer_capture[name] for name in _GDN_ROW_NAMES
        )
        if getattr(layer, "ple", None) is not None:
            cache[index]._mtplx_verify_ple = tuple(
                layer_capture[name] for name in _PLE_ROW_NAMES
            )


def _bind_fixed_m4_device_commit(self: Any, cache):
    """Bind the exact target-cache state transition once per request."""

    from .kernels.qwen4_m4_state_handoff import (
        QWEN4_M4_GDN_CONV_ROWS,
        QWEN4_M4_PLE_CONV_ROWS,
        QWEN4_M4_PLE_HISTORY_ROWS,
        QWEN4_M4_VERIFY_WIDTH,
        replay_qwen4_m4_gdn_state,
    )

    inner = _inner(self)
    binding = self._mtplx_qwen4_m4_state_handoff_binding
    plan = tuple(
        (
            "gdn" if layer.is_linear else "qsa",
            index,
            entry,
            getattr(layer, "linear_attn", None),
            getattr(layer, "ple", None),
        )
        for index, (layer, entry) in enumerate(zip(inner.layers, cache))
    )
    if tuple(index for kind, index, *_rest in plan if kind == "gdn") != (
        binding.gdn_layer_indices
    ):
        raise ValueError("fixed-M4 device commit GDN topology changed")
    if tuple(index for kind, index, *_rest in plan if kind == "qsa") != (
        binding.qsa_layer_indices
    ):
        raise ValueError("fixed-M4 device commit QSA topology changed")
    ple_rows = tuple(index for _kind, index, _entry, _gdn, ple in plan if ple)
    if ple_rows != (binding.ple_layer_index,):
        raise ValueError("fixed-M4 device commit PLE topology changed")

    # One constant GDN keep mask per accepted width. The all-width commit
    # rebuilds this from the device ``accepted_count`` every cycle; the
    # compact commit already owns the width on the host, so the four masks
    # are bound once and reused.
    keep_masks = tuple(
        mx.array(
            [[column < width for column in range(QWEN4_M4_VERIFY_WIDTH)]],
            dtype=mx.bool_,
        )
        for width in range(1, QWEN4_M4_VERIFY_WIDTH + 1)
    )

    def commit(accepted_count, snapshot_states, verify_hidden):
        accepted = accepted_count.reshape(-1)[0].astype(mx.int32)
        keep = accepted + 1
        conv_indices = keep + mx.arange(
            QWEN4_M4_GDN_CONV_ROWS, dtype=mx.int32
        )

        ple_entry = cache[binding.ple_layer_index]
        ple_pre = snapshot_states[binding.ple_layer_index]
        ple_qkv, *_ple_gdn_rows = ple_entry._mtplx_verify_rows
        (
            ple_hidden,
            ple_ids,
            ple_conv_rows,
        ) = ple_entry._mtplx_verify_ple
        compiled_aux = ple_entry._mtplx_verify_compiled_aux
        ple_layer = inner.layers[binding.ple_layer_index]
        logical_states = []
        # Build the three exact-width PLE cache candidates outside the outer
        # compiled M4 target graph.  The outer graph changes the arithmetic
        # schedule in rounding-sensitive states (first observed at seed 31,
        # cycle 91); this eager-lazy construction matches the CPU commit while
        # the device accepted-count still selects the authoritative candidate.
        for logical_width in range(1, QWEN4_M4_VERIFY_WIDTH):
            logical_cache = type(ple_entry)(len(ple_entry.cache))
            for slot, leaf in enumerate(ple_pre):
                logical_cache[slot] = leaf
            with (
                verify_capture_disabled_scope(),
                compiled_verify_ple_scope(compiled_aux[:, :logical_width]),
            ):
                logical_hidden = ple_hidden[:, :logical_width] + ple_layer.ple(
                    ple_hidden[:, :logical_width],
                    ple_ids[:, :logical_width],
                    logical_cache,
                )
                logical_mixed, _logical_hyper, _logical_inject = (
                    ple_layer.attn_hyper_connection(logical_hidden)
                )
                ple_layer.linear_attn(logical_mixed, None, logical_cache)
            logical_states.append(tuple(logical_cache.cache))
        ple_logical_states = tuple(logical_states)
        (
            ple_gdn_conv,
            selected_ple_conv,
            selected_ple_history,
            selected_hidden,
            gdn_keep_mask,
        ) = binding.select_windows(
            accepted_count,
            ple_pre[0],
            ple_qkv,
            ple_pre[2],
            ple_conv_rows,
            ple_pre[3],
            ple_ids.astype(mx.int32),
            verify_hidden,
        )

        for kind, index, entry, gdn, ple in plan:
            if kind == "qsa":
                # The fixed buffers already contain all four verify rows.
                # Only their device-owned logical frontier changes.
                entry.kv.cache[2] = entry.kv.cache[2] - (
                    QWEN4_M4_VERIFY_WIDTH - keep
                )
                entry.kv.rollback_state[:] = [None, None, None]
                continue

            pre = snapshot_states[index]
            qkv, q, k, v, a, b = entry._mtplx_verify_rows
            next_conv = (
                ple_gdn_conv
                if ple is not None
                else mx.take(
                    mx.concatenate((pre[0], qkv), axis=1),
                    conv_indices,
                    axis=1,
                )
            )
            next_delta = replay_qwen4_m4_gdn_state(
                q,
                k,
                v,
                a,
                b,
                gdn.A_log,
                gdn.dt_bias,
                pre[1],
                gdn_keep_mask,
            )
            if ple is not None:
                def select_logical(slot: int, physical_m4):
                    selected = physical_m4
                    for width_index in range(2, -1, -1):
                        selected = mx.where(
                            accepted == width_index,
                            ple_logical_states[width_index][slot],
                            selected,
                        )
                    return selected

                entry[0] = select_logical(0, next_conv)
                entry[1] = select_logical(1, next_delta)
            else:
                entry[0] = next_conv
                entry[1] = next_delta
            entry._mtplx_verify_rows = None
            if ple is not None:
                entry[2] = select_logical(2, selected_ple_conv)
                entry[3] = select_logical(3, selected_ple_history)
                entry._mtplx_verify_ple = None
                entry._mtplx_verify_compiled_aux = None
        state_roots = []
        for kind, _index, entry, _gdn, _ple in plan:
            if kind == "qsa":
                state_roots.extend(entry.state_leaves)
            else:
                state_roots.extend(entry.cache)
        return selected_hidden, tuple(state_roots)

    def commit_width(accepted_width, snapshot_states, verify_hidden):
        """Commit exactly the accepted width once the host owns the decision.

        Same committed state as ``commit`` for the accepted width, reached
        without building the three unselected candidates.  The caller has
        already materialized the verifier decision, so every window index is a
        Python constant: the PLE replay runs at one width (or not at all when
        the whole M4 window is accepted), the 12 ``mx.where`` selections are
        gone, and the GDN conv windows are static slices instead of a device
        ``take``.  The bf16 arithmetic order of the GDN recurrence is
        unchanged; the selection order is not, so a receipt digest may move.

        Still eager on purpose.  Folding the 48-layer loop into one
        ``mx.compile``'d function per width would need the ~300 per-cycle
        capture and snapshot leaves re-expressed as positional arguments
        (they arrive here as mutable cache attributes), a four-width prewarm
        on synthetic captures at every bank install, and a GPU run to
        establish that ``gated_delta_update(use_kernel=True)`` composes with
        ``mx.compile``.  The eager width-a form already removes the
        candidate construction, which is where the host ops are.
        """

        accepted = int(accepted_width)
        if not 0 <= accepted < QWEN4_M4_VERIFY_WIDTH:
            raise ValueError(
                "fixed-M4 compact commit requires an accepted width in [0, 3]"
            )
        keep = accepted + 1
        whole_window = keep == QWEN4_M4_VERIFY_WIDTH
        gdn_keep_mask = keep_masks[accepted]

        ple_entry = cache[binding.ple_layer_index]
        ple_pre = snapshot_states[binding.ple_layer_index]
        (
            ple_hidden,
            ple_ids,
            ple_conv_rows,
        ) = ple_entry._mtplx_verify_ple
        compiled_aux = ple_entry._mtplx_verify_compiled_aux
        ple_layer = inner.layers[binding.ple_layer_index]

        # Below the whole window the PLE layer's four slots come entirely from
        # one exact-width logical replay, so neither its GDN windows nor its
        # recurrence are built at all.
        ple_logical_state = None
        if not whole_window:
            logical_cache = type(ple_entry)(len(ple_entry.cache))
            for slot, leaf in enumerate(ple_pre):
                logical_cache[slot] = leaf
            with (
                verify_capture_disabled_scope(),
                compiled_verify_ple_scope(compiled_aux[:, :keep]),
            ):
                logical_hidden = ple_hidden[:, :keep] + ple_layer.ple(
                    ple_hidden[:, :keep],
                    ple_ids[:, :keep],
                    logical_cache,
                )
                logical_mixed, _logical_hyper, _logical_inject = (
                    ple_layer.attn_hyper_connection(logical_hidden)
                )
                ple_layer.linear_attn(logical_mixed, None, logical_cache)
            ple_logical_state = tuple(logical_cache.cache)

        selected_hidden = verify_hidden[:, accepted : accepted + 1, :]

        for kind, index, entry, gdn, ple in plan:
            if kind == "qsa":
                # The fixed buffers already contain all four verify rows.
                # Only their device-owned logical frontier changes, and the
                # whole window moves it by nothing at all.
                if not whole_window:
                    entry.kv.cache[2] = entry.kv.cache[2] - (
                        QWEN4_M4_VERIFY_WIDTH - keep
                    )
                entry.kv.rollback_state[:] = [None, None, None]
                continue

            if ple is not None and ple_logical_state is not None:
                entry[0] = ple_logical_state[0]
                entry[1] = ple_logical_state[1]
                entry[2] = ple_logical_state[2]
                entry[3] = ple_logical_state[3]
                entry._mtplx_verify_rows = None
                entry._mtplx_verify_ple = None
                entry._mtplx_verify_compiled_aux = None
                continue

            pre = snapshot_states[index]
            qkv, q, k, v, a, b = entry._mtplx_verify_rows
            entry[0] = mx.concatenate((pre[0], qkv), axis=1)[
                :, keep : keep + QWEN4_M4_GDN_CONV_ROWS
            ]
            entry[1] = replay_qwen4_m4_gdn_state(
                q,
                k,
                v,
                a,
                b,
                gdn.A_log,
                gdn.dt_bias,
                pre[1],
                gdn_keep_mask,
            )
            entry._mtplx_verify_rows = None
            if ple is not None:
                entry[2] = mx.concatenate((pre[2], ple_conv_rows), axis=1)[
                    :, keep : keep + QWEN4_M4_PLE_CONV_ROWS
                ]
                entry[3] = mx.concatenate(
                    (pre[3], ple_ids.astype(mx.int32).astype(mx.int64)),
                    axis=1,
                )[:, keep : keep + QWEN4_M4_PLE_HISTORY_ROWS]
                entry._mtplx_verify_ple = None
                entry._mtplx_verify_compiled_aux = None
        state_roots = []
        for kind, _index, entry, _gdn, _ple in plan:
            if kind == "qsa":
                state_roots.extend(entry.state_leaves)
            else:
                state_roots.extend(entry.cache)
        return selected_hidden, tuple(state_roots)

    commit.commit_width = commit_width
    return commit


def install_qwen4_fixed_verify_route(
    runtime: Any,
    *,
    raw_q4_aux: bool = False,
    owned_all_miss_rows: bool = False,
    early_window_prefetch: bool = False,
) -> dict[str, Any]:
    """Validate the Qwen4 state topology once and bind direct fixed-M4 hooks."""

    inner = _inner(runtime)
    layers = tuple(inner.layers)
    linear = tuple(index for index, layer in enumerate(layers) if layer.is_linear)
    qsa = tuple(index for index, layer in enumerate(layers) if not layer.is_linear)
    ple = tuple(
        index for index, layer in enumerate(layers) if getattr(layer, "ple", None)
    )
    if not linear or not qsa or len(ple) != 1 or not layers[ple[0]].is_linear:
        raise ValueError("qwen4 fixed-M4 verifier requires mixed QSA/GDN and one PLE")

    args = inner.args
    if int(args.hidden_size) == 2560:
        observed = (
            len(layers),
            len(linear),
            len(qsa),
            int(args.hc_count),
            int(args.indexer_compress_ratio),
        )
        if observed != (48, 36, 12, 4, 4):
            raise ValueError(f"qwen4 fixed-M4 production geometry mismatch: {observed}")

    extra = []
    for index in linear:
        names = _GDN_ROW_NAMES + (
            _PLE_ROW_NAMES if index == ple[0] else ()
        )
        extra.append((index, names))

    from .kernels.qwen4_m4_state_handoff import (
        bind_qwen4_m4_production_state_handoff,
    )

    runtime._mtplx_qwen4_m4_state_handoff_binding = (
        bind_qwen4_m4_production_state_handoff(
            linear_layer_indices=linear,
            qsa_layer_indices=qsa,
            ple_layer_index=ple[0],
            capture_layout=tuple(extra),
        )
    )

    runtime.forward_ar_capture = MethodType(_forward_ar_capture, runtime)
    runtime.forward_fixed_m4_prefix = MethodType(
        _forward_fixed_m4_prefix, runtime
    )
    runtime.forward_fixed_m4_suffix = MethodType(
        _forward_fixed_m4_suffix, runtime
    )
    # W67 (MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS): the N-layer pair.  Bound
    # alongside PR391's layer-0 pair, never in place of it -- the two lanes
    # compile different closures and neither is on the other's path.
    runtime.forward_fixed_m4_overlap_prefix = MethodType(
        _forward_fixed_m4_overlap_prefix, runtime
    )
    runtime.forward_fixed_m4_overlap_suffix = MethodType(
        _forward_fixed_m4_overlap_suffix, runtime
    )
    runtime.prepare_compiled_verify_aux = MethodType(
        _prepare_compiled_verify_aux, runtime
    )
    runtime.build_fixed_m4_compiled_verify_aux = partial(
        _build_fixed_m4_compiled_verify_aux,
        runtime,
        raw_q4_aux=raw_q4_aux,
        owned_all_miss_rows=owned_all_miss_rows,
        early_window_prefetch=early_window_prefetch,
    )
    ple_embedding = layers[ple[0]].ple.ple_embedding
    if raw_q4_aux:
        runtime.dequantize_fixed_m4_compiled_verify_aux = partial(
            _dequantize_fixed_m4_ple,
            output_dim=int(args.ple_embed_dim),
        )
    if ple_embedding.ngram_embedding._sidecar is None:
        del runtime.build_fixed_m4_compiled_verify_aux
    runtime.commit_compiled_verify_captures = MethodType(
        _commit_compiled_verify_captures, runtime
    )
    runtime.bind_fixed_m4_device_commit = MethodType(
        _bind_fixed_m4_device_commit, runtime
    )
    runtime._mtplx_capture_layout = ()
    runtime._mtplx_capture_extra_layout = tuple(extra)
    runtime.qwen4_fixed_m4_compiled_verify = True
    return {"installed": True, "linear_layers": len(linear), "rows": 4}


def enable_qwen4_fixed_verify_owned_window_prefetch(runtime: Any) -> None:
    """Rebind the installed PR391 route to early owned full-window reads."""

    if getattr(runtime, "qwen4_fixed_m4_compiled_verify", False) is not True:
        raise RuntimeError("owned window prefetch requires installed fixed-M4 verify")
    runtime.build_fixed_m4_compiled_verify_aux = partial(
        _build_fixed_m4_compiled_verify_aux,
        runtime,
        raw_q4_aux=False,
        owned_all_miss_rows=True,
        early_window_prefetch=True,
    )


__all__ = [
    "enable_qwen4_fixed_verify_owned_window_prefetch",
    "is_qwen4_fixed_verify_config",
    "install_qwen4_fixed_verify_route",
    "qwen4_fixed_verify_enabled",
]
