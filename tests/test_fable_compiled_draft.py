"""Compiled D1->D3 draft chain (MTPLX_FABLE_COMPILED_DRAFT).

Everything here runs on the MLX CPU stream over tiny synthetic tensors: the
point is the *structure* of the replay (output ordering, state advance, tape
consumption, guard behavior), which is device-independent.  The numerical
question this cannot answer -- whether the production MTP DecoderLayer traces
and replays identically on Metal -- needs a GPU parity run.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx

import mtplx.fable_compiled_draft as fcd


# --------------------------------------------------------------------------
# CPU-only fixture: no Metal work is ever issued from this module.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    # Hard stop: this module must never issue Metal work -- it shares the box
    # with guarded benchmark windows.
    assert mx.default_device() == mx.cpu
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _clean_flag(monkeypatch):
    monkeypatch.delenv(fcd.FABLE_COMPILED_DRAFT_ENV, raising=False)
    fcd.reset_compiled_draft_flag_cache()
    try:
        yield
    finally:
        fcd.reset_compiled_draft_flag_cache()


# --------------------------------------------------------------------------
# Tiny stand-ins for TensorOffsetKVCache / TensorOffsetQSACache.
#
# Only the surface the compiled chain depends on: a mutable list of state
# leaves (so mx.compile can rebind them), a device-resident offset, the
# rollback_state slots update_and_fetch writes into, and the fixed-capacity
# markers the construction gate reads.
# --------------------------------------------------------------------------

_DIM = 4
_AUX = 3
_VOCAB = 16
_TOP_K = 5
_DEPTH = 3


class _TinyKV:
    def __init__(self, capacity: int) -> None:
        self.cache = [
            mx.zeros((1, 1, capacity, _DIM)),
            mx.array(0, dtype=mx.int32),
        ]
        self.rollback_state = [None, None, None]

    @property
    def offset(self):
        return self.cache[1]

    def update_and_fetch(self, row):
        self.rollback_state[0] = self.cache[1]
        snapshot = mx.slice(
            self.cache[0], self.cache[1], axes=(2,), slice_size=row.shape
        )
        self.rollback_state[1] = snapshot
        self.rollback_state[2] = snapshot
        self.cache[0] = mx.slice_update(
            self.cache[0], row, self.cache[1], axes=(2,)
        )
        self.cache[1] = self.cache[1] + 1
        return self.cache[0]


class _TinyQSACache:
    fixed_capacity = True

    def __init__(self, capacity: int = 64) -> None:
        self.kv = _TinyKV(capacity)
        self.aux = [mx.zeros((1, capacity, _AUX))]
        self._compile_state = [self.kv.cache, self.aux]

    @property
    def compile_state(self):
        return self._compile_state

    @property
    def state_leaves(self):
        return [*self.kv.cache, *self.aux]

    @property
    def capacity(self) -> int:
        return int(self.kv.cache[0].shape[2])

    @property
    def offset(self):
        return self.kv.offset


def _state_tree(cache) -> list:
    """Same nesting generation._device_core_state_tree produces for one QSA."""

    return [[cache[0].compile_state]]


class _TinyRuntime:
    """A deterministic stand-in for the MTP draft surface.

    ``draft_mtp`` reads and writes the cache exactly the way the production
    path does -- one appended row per call, a device-resident offset, an
    auxiliary leaf that also advances -- so the compiled capture has real state
    to thread.
    """

    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.embed = mx.array(rng.standard_normal((_VOCAB, _DIM)).astype(np.float32))
        self.head = mx.array(rng.standard_normal((_DIM, _VOCAB)).astype(np.float32))
        self.model = SimpleNamespace(
            mtp=SimpleNamespace(),
            mtp_forward=lambda hidden_states, next_token_ids, **kwargs: None,
        )
        self.diagnostic_counters: dict[str, int] = {}
        self.depths_seen: list = []

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = (
            int(self.diagnostic_counters.get(key, 0)) + int(amount)
        )

    def draft_mtp(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache,
        return_hidden,
        mtp_hidden_variant,
        mtp_depth,
    ):
        assert return_hidden is True
        assert mtp_hidden_variant == "post_norm"
        self.depths_seen.append(mtp_depth)
        entry = mtp_cache[0]
        ids = next_token_ids.reshape(-1).astype(mx.int32)
        emb = mx.take(self.embed, ids, axis=0).reshape(1, 1, _DIM)
        fused = hidden_states + emb
        bank = entry.kv.update_and_fetch(fused.reshape(1, 1, 1, _DIM))
        entry.aux[0] = entry.aux[0] + 1.0
        context = mx.sum(bank, axis=2)
        produced = mx.tanh(fused + 0.125 * context)
        logits = mx.matmul(produced, self.head)
        return logits, produced


def _pure_mlx_selector(candidate_ids, candidate_values, candidate_probs, uniform_bits):
    """An eager, CPU-runnable stand-in for the softfloat64 choice kernel.

    Same ABI as ``bind_pr391_softfloat64_candidate_selector``'s ``apply``:
    (ids[1,K], values[1,K], probs[1,K], uniform[1]) ->
    (selected[1], raw_ids[1,K], raw_values[1,K], raw_probs[1,K]).
    Inverse-CDF over the proposal, so the draw actually depends on the tape
    row -- a chain that mis-orders the rows produces different tokens.
    """

    row = candidate_probs.reshape(-1)
    cdf = mx.cumsum(row / mx.sum(row))
    draw = uniform_bits.reshape(-1)[0].astype(mx.float32)
    pick = mx.minimum(mx.sum((cdf <= draw).astype(mx.int32)), _TOP_K - 1)
    selected = mx.take(candidate_ids.reshape(-1), pick.reshape(1)).astype(mx.uint32)
    return selected, candidate_ids, candidate_values, candidate_probs


def _eager_chain(
    *,
    rt,
    mtp_cache,
    selector,
    frspec_ids,
    mtp_hidden_variant="post_norm",
    depth=_DEPTH,
    top_k=_TOP_K,
):
    """Byte-for-byte mirror of generation.py's ``chain_fn`` body.

    Kept literal (not shared with production) so a drift in either copy shows
    up as a numeric mismatch in ``test_compiled_chain_matches_eager_outputs``
    rather than being papered over by a shared helper.
    """

    from mtplx.fast_sampling import (
        _deterministic_mlx_top_k_support,
        _order_bounded_mlx_top_k_support,
    )

    def chain_fn(hidden_states, first_token_ids, uniform_bit_rows):
        next_hidden = hidden_states
        next_token = first_token_ids
        selected_tokens = []
        raw_ids_by_depth = []
        raw_values_by_depth = []
        raw_probs_by_depth = []
        for level in range(1, depth + 1):
            logits, produced_hidden = rt.draft_mtp(
                next_hidden,
                next_token,
                mtp_cache=mtp_cache,
                return_hidden=True,
                mtp_hidden_variant=mtp_hidden_variant,
                mtp_depth=level,
            )
            row = logits[:, -1, :].reshape(-1)
            flat = row.astype(mx.float32)
            local_ids, q_values = _deterministic_mlx_top_k_support(flat, top_k)
            local_ids, q_values = _order_bounded_mlx_top_k_support(local_ids, q_values)
            q_probs = mx.exp(q_values - mx.logsumexp(flat, axis=-1, keepdims=True))
            if frspec_ids is not None and int(row.shape[0]) == int(
                frspec_ids.shape[0]
            ):
                real_ids = mx.take(frspec_ids, local_ids)
            else:
                real_ids = local_ids
            selected, raw_ids, raw_values, raw_probs = selector(
                real_ids.astype(mx.uint32).reshape(1, top_k),
                q_values.astype(mx.float32).reshape(1, top_k),
                q_probs.astype(mx.float32).reshape(1, top_k),
                uniform_bit_rows[level - 1 : level],
            )
            selected = selected.reshape(1, 1)
            selected_tokens.append(selected)
            raw_ids_by_depth.append(raw_ids)
            raw_values_by_depth.append(raw_values)
            raw_probs_by_depth.append(raw_probs)
            next_hidden = produced_hidden[:, -1:, :]
            next_token = selected
        return (
            mx.concatenate(selected_tokens, axis=1),
            mx.concatenate(raw_ids_by_depth, axis=0),
            mx.concatenate(raw_values_by_depth, axis=0),
            mx.concatenate(raw_probs_by_depth, axis=0),
        )

    return chain_fn


def _inputs(seed: int = 3):
    rng = np.random.default_rng(seed)
    hidden = mx.array(rng.standard_normal((1, 1, _DIM)).astype(np.float32))
    primary = mx.array([[5]], dtype=mx.uint32)
    uniform = mx.array(
        np.asarray([0.11, 0.63, 0.87], dtype=np.float32).reshape(_DEPTH)
    )
    return hidden, primary, uniform


def _build(rt, cache, *, frspec_ids=None, request_max_tokens=8):
    return fcd.build_compiled_draft_chain(
        rt=rt,
        mtp_cache=cache,
        state_tree=_state_tree(cache),
        mtp_hidden_variant="post_norm",
        selector=_pure_mlx_selector,
        frspec_ids=frspec_ids,
        depth=_DEPTH,
        top_k=_TOP_K,
        request_max_tokens=request_max_tokens,
    )


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def test_flag_defaults_off() -> None:
    assert fcd.compiled_draft_enabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_flag_accepts_the_usual_truthy_spellings(monkeypatch, raw) -> None:
    monkeypatch.setenv(fcd.FABLE_COMPILED_DRAFT_ENV, raw)
    fcd.reset_compiled_draft_flag_cache()
    assert fcd.compiled_draft_enabled() is True


@pytest.mark.parametrize("raw", ["", "0", "off", "no", "maybe"])
def test_flag_rejects_everything_else(monkeypatch, raw) -> None:
    monkeypatch.setenv(fcd.FABLE_COMPILED_DRAFT_ENV, raw)
    fcd.reset_compiled_draft_flag_cache()
    assert fcd.compiled_draft_enabled() is False


def test_flag_is_read_once_then_memoized(monkeypatch) -> None:
    monkeypatch.setenv(fcd.FABLE_COMPILED_DRAFT_ENV, "1")
    fcd.reset_compiled_draft_flag_cache()
    assert fcd.compiled_draft_enabled() is True
    monkeypatch.delenv(fcd.FABLE_COMPILED_DRAFT_ENV)
    assert fcd.compiled_draft_enabled() is True  # no second environment read


def test_disabled_gate_builds_nothing(monkeypatch) -> None:
    def explode(**_kwargs):  # pragma: no cover - must never run
        raise AssertionError("the disabled gate must not build a compiled chain")

    monkeypatch.setattr(fcd, "build_compiled_draft_chain", explode)
    assert (
        fcd.maybe_build_compiled_draft_chain(
            rt=None,
            mtp_cache=None,
            state_tree=None,
            mtp_hidden_variant="post_norm",
            selector=_pure_mlx_selector,
            frspec_ids=None,
            depth=_DEPTH,
            top_k=_TOP_K,
            request_max_tokens=8,
        )
        is None
    )


def test_armed_gate_delegates_to_the_builder(monkeypatch) -> None:
    monkeypatch.setenv(fcd.FABLE_COMPILED_DRAFT_ENV, "1")
    fcd.reset_compiled_draft_flag_cache()
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return {"chain_fn": "compiled"}

    monkeypatch.setattr(fcd, "build_compiled_draft_chain", record)
    built = fcd.maybe_build_compiled_draft_chain(
        rt="rt",
        mtp_cache="cache",
        state_tree="tree",
        mtp_hidden_variant="post_norm",
        selector=_pure_mlx_selector,
        frspec_ids=None,
        depth=_DEPTH,
        top_k=_TOP_K,
        request_max_tokens=8,
    )
    assert built == {"chain_fn": "compiled"}
    assert seen["rt"] == "rt"
    assert seen["state_tree"] == "tree"
    assert seen["depth"] == _DEPTH
    assert seen["request_max_tokens"] == 8


# --------------------------------------------------------------------------
# The generation.py wiring stays inert with the flag off
# --------------------------------------------------------------------------


def test_generation_installs_the_eager_chain_when_the_flag_is_off() -> None:
    import mtplx.generation as generation

    source = inspect.getsource(generation._pr391_make_float32_d3_core)

    assert "maybe_build_compiled_draft_chain(" in source
    assert '"fn": chain_fn if compiled_draft is None else compiled_draft["chain_fn"]' in source
    # The guards the retained exact route already carries must survive.
    assert "os.environ" not in source
    assert "mx.random" not in source
    assert "fallback" not in source.lower()
    assert "_eval(" not in source
    assert not any(
        isinstance(node, ast.Try) for node in ast.walk(ast.parse(source))
    )


def test_generation_imports_the_gate_helper_not_the_environment() -> None:
    import mtplx.generation as generation

    assert generation.maybe_build_compiled_draft_chain is (
        fcd.maybe_build_compiled_draft_chain
    )


# --------------------------------------------------------------------------
# Structural parity with the eager chain
# --------------------------------------------------------------------------


def test_compiled_chain_matches_eager_outputs() -> None:
    hidden, primary, uniform = _inputs()

    eager_rt = _TinyRuntime()
    eager_cache = [_TinyQSACache()]
    eager = _eager_chain(
        rt=eager_rt, mtp_cache=eager_cache, selector=_pure_mlx_selector,
        frspec_ids=None,
    )(hidden, primary, uniform)
    mx.eval(*eager)

    compiled_rt = _TinyRuntime()
    compiled_cache = [_TinyQSACache()]
    built = _build(compiled_rt, compiled_cache)
    compiled = built["chain_fn"](hidden, primary, uniform)
    mx.eval(*compiled)

    assert len(compiled) == len(eager) == 4
    for index, (got, want) in enumerate(zip(compiled, eager)):
        assert tuple(got.shape) == tuple(want.shape), index
        assert got.dtype == want.dtype, index
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            np.asarray(want, dtype=np.float64),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"compiled output {index} diverged from the eager chain",
        )


def test_compiled_chain_returns_the_pr391_output_structure() -> None:
    hidden, primary, uniform = _inputs()
    built = _build(_TinyRuntime(), [_TinyQSACache()])
    tokens, ids, values, probs = built["chain_fn"](hidden, primary, uniform)
    mx.eval(tokens, ids, values, probs)

    assert tuple(tokens.shape) == (1, _DEPTH)
    assert tokens.dtype == mx.uint32
    assert tuple(ids.shape) == (_DEPTH, _TOP_K)
    assert ids.dtype == mx.uint32
    assert tuple(values.shape) == (_DEPTH, _TOP_K)
    assert values.dtype == mx.float32
    assert tuple(probs.shape) == (_DEPTH, _TOP_K)
    assert probs.dtype == mx.float32


def test_compiled_chain_advances_the_captured_cache_like_the_eager_chain() -> None:
    hidden, primary, uniform = _inputs()

    eager_cache = [_TinyQSACache()]
    mx.eval(
        _eager_chain(
            rt=_TinyRuntime(), mtp_cache=eager_cache,
            selector=_pure_mlx_selector, frspec_ids=None,
        )(hidden, primary, uniform)
    )

    compiled_cache = [_TinyQSACache()]
    built = _build(_TinyRuntime(), compiled_cache)
    mx.eval(built["chain_fn"](hidden, primary, uniform))

    assert int(compiled_cache[0].offset) == int(eager_cache[0].offset) == _DEPTH
    np.testing.assert_allclose(
        np.asarray(compiled_cache[0].kv.cache[0], dtype=np.float64),
        np.asarray(eager_cache[0].kv.cache[0], dtype=np.float64),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_cache[0].aux[0], dtype=np.float64),
        np.asarray(eager_cache[0].aux[0], dtype=np.float64),
    )


def test_repeated_cycles_keep_chaining_the_captured_state() -> None:
    hidden, primary, uniform = _inputs()

    eager_cache = [_TinyQSACache()]
    eager_rt = _TinyRuntime()
    eager_chain = _eager_chain(
        rt=eager_rt, mtp_cache=eager_cache, selector=_pure_mlx_selector,
        frspec_ids=None,
    )
    compiled_cache = [_TinyQSACache()]
    compiled_chain = _build(_TinyRuntime(), compiled_cache)["chain_fn"]

    for _cycle in range(3):
        eager_out = eager_chain(hidden, primary, uniform)
        compiled_out = compiled_chain(hidden, primary, uniform)
        mx.eval(*eager_out, *compiled_out)
        np.testing.assert_allclose(
            np.asarray(compiled_out[0], dtype=np.int64),
            np.asarray(eager_out[0], dtype=np.int64),
        )

    assert int(compiled_cache[0].offset) == int(eager_cache[0].offset) == 3 * _DEPTH


def test_frspec_remap_is_applied_inside_the_compiled_body() -> None:
    hidden, primary, uniform = _inputs()
    frspec_ids = mx.array(
        np.asarray(list(reversed(range(_VOCAB))), dtype=np.uint32)
    )

    eager = _eager_chain(
        rt=_TinyRuntime(), mtp_cache=[_TinyQSACache()],
        selector=_pure_mlx_selector, frspec_ids=frspec_ids,
    )(hidden, primary, uniform)
    compiled = _build(
        _TinyRuntime(), [_TinyQSACache()], frspec_ids=frspec_ids
    )["chain_fn"](hidden, primary, uniform)
    mx.eval(*eager, *compiled)

    np.testing.assert_array_equal(
        np.asarray(compiled[1], dtype=np.int64),
        np.asarray(eager[1], dtype=np.int64),
    )
    # The remap must actually have moved the ids, or this proves nothing.
    unmapped = _build(_TinyRuntime(), [_TinyQSACache()])["chain_fn"](
        hidden, primary, uniform
    )
    mx.eval(*unmapped)
    assert not np.array_equal(
        np.asarray(compiled[1], dtype=np.int64),
        np.asarray(unmapped[1], dtype=np.int64),
    )


def test_one_tape_row_is_consumed_per_depth_in_order() -> None:
    hidden, primary, uniform = _inputs()
    seen = []

    def recording_selector(ids, values, probs, uniform_bits):
        seen.append(uniform_bits)
        return _pure_mlx_selector(ids, values, probs, uniform_bits)

    cache = [_TinyQSACache()]
    built = fcd.build_compiled_draft_chain(
        rt=_TinyRuntime(),
        mtp_cache=cache,
        state_tree=_state_tree(cache),
        mtp_hidden_variant="post_norm",
        selector=recording_selector,
        frspec_ids=None,
        depth=_DEPTH,
        top_k=_TOP_K,
        request_max_tokens=8,
    )
    seen.clear()
    mx.eval(built["chain_fn"](hidden, primary, uniform))

    assert len(seen) == _DEPTH
    for level, row in enumerate(seen):
        assert tuple(row.shape) == (1,)
        np.testing.assert_allclose(
            np.asarray(row, dtype=np.float64),
            np.asarray(uniform[level : level + 1], dtype=np.float64),
        )


def test_draft_mtp_call_counter_survives_the_compiled_replay() -> None:
    hidden, primary, uniform = _inputs()
    rt = _TinyRuntime()
    cache = [_TinyQSACache()]
    chain = _build(rt, cache)["chain_fn"]

    rt.diagnostic_counters.clear()
    for _cycle in range(2):
        mx.eval(chain(hidden, primary, uniform))
    assert rt.diagnostic_counters["draft_mtp_calls"] == 2 * _DEPTH


def test_compiled_body_traces_once_across_cycles() -> None:
    hidden, primary, uniform = _inputs()
    rt = _TinyRuntime()
    cache = [_TinyQSACache()]
    chain = _build(rt, cache)["chain_fn"]

    mx.eval(chain(hidden, primary, uniform))
    traced_after_prewarm = len(rt.depths_seen)
    for _cycle in range(4):
        mx.eval(chain(hidden, primary, uniform))

    # The whole point of the flag: the python body runs once, while mx.compile
    # traces.  One trace covers all three depths (their (hidden, token) input
    # signatures are identical) and every later cycle replays it, so the host
    # never re-issues the chain's ops.
    assert traced_after_prewarm == 1
    assert len(rt.depths_seen) == 1


def test_trace_stats_report_exactly_one_trace_for_a_prewarmed_request() -> None:
    hidden, primary, uniform = _inputs()
    built = _build(_TinyRuntime(), [_TinyQSACache()])

    assert built["trace_stats"]["body_traces"] == 0  # nothing traced yet
    mx.eval(built["chain_fn"](hidden, primary, uniform))  # the prewarm
    assert built["trace_stats"]["body_traces"] == 1
    for _cycle in range(5):
        mx.eval(built["chain_fn"](hidden, primary, uniform))
    assert built["trace_stats"]["body_traces"] == 1


def test_a_captured_leaf_rewritten_between_cycles_is_re_read() -> None:
    """The carried-D3 rewind mechanic: generation.py sets ``kv.cache[2]``
    between cycles, and the compiled replay must honor it rather than replay a
    trace-time constant."""

    hidden, primary, uniform = _inputs()
    cache = [_TinyQSACache()]
    chain = _build(_TinyRuntime(), cache)["chain_fn"]
    mx.eval(chain(hidden, primary, uniform))
    assert int(cache[0].offset) == _DEPTH

    # Logically rewind the frontier the way _pr391_finish_canonical_d3_queue does.
    cache[0].kv.cache[1] = cache[0].kv.cache[1] - _DEPTH
    mx.eval(chain(hidden, primary, uniform))
    assert int(cache[0].offset) == _DEPTH


def test_a_nested_compiled_inner_function_still_traces_once() -> None:
    """MTPLX_QWEN4_COMPILED_MTP_PREPARE installs an mx.compile'd
    ``_prepare_inputs_eager`` *inside* the MTP module; the outer per-depth
    compile has to swallow it rather than retrace per call."""

    hidden, primary, uniform = _inputs()
    rt = _TinyRuntime()
    inner_traces = {"n": 0}
    plain_draft_mtp = rt.draft_mtp

    def inner(x):
        inner_traces["n"] += 1
        return x * 1.0

    compiled_inner = mx.compile(inner)

    def wrapped(hidden_states, next_token_ids, **kwargs):
        return plain_draft_mtp(compiled_inner(hidden_states), next_token_ids, **kwargs)

    rt.draft_mtp = wrapped
    built = _build(rt, [_TinyQSACache()])
    for _cycle in range(4):
        mx.eval(built["chain_fn"](hidden, primary, uniform))

    assert built["trace_stats"]["body_traces"] == 1
    assert inner_traces["n"] == 1


def test_compiled_body_passes_an_inert_depth() -> None:
    hidden, primary, uniform = _inputs()
    rt = _TinyRuntime()
    cache = [_TinyQSACache()]
    mx.eval(_build(rt, cache)["chain_fn"](hidden, primary, uniform))
    assert set(rt.depths_seen) == {None}


def test_rollback_slots_never_retain_trace_time_tracers() -> None:
    hidden, primary, uniform = _inputs()
    cache = [_TinyQSACache()]
    chain = _build(_TinyRuntime(), cache)["chain_fn"]
    mx.eval(chain(hidden, primary, uniform))
    assert cache[0].kv.rollback_state == [None, None, None]


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_state_leaf_slots_finds_every_rebindable_leaf() -> None:
    cache = [_TinyQSACache()]
    slots = fcd.state_leaf_slots(_state_tree(cache))
    assert len(slots) == 3  # keys, offset, aux
    assert fcd.state_leaf_shapes(slots) == tuple(
        tuple(leaf.shape) for leaf in cache[0].state_leaves
    )


def test_state_leaf_slots_rejects_immutable_leaf_containers() -> None:
    with pytest.raises(fcd.CompiledDraftUnsupported, match="immutable tuple"):
        fcd.state_leaf_slots([[(mx.zeros((2,)), mx.zeros((2,)))]])


def test_state_shape_change_raises_instead_of_replaying_a_stale_graph() -> None:
    hidden, primary, uniform = _inputs()
    cache = [_TinyQSACache()]
    chain = _build(_TinyRuntime(), cache)["chain_fn"]
    mx.eval(chain(hidden, primary, uniform))

    # Simulate a mid-request ensure_capacity growth.
    cache[0].kv.cache[0] = mx.zeros((1, 1, 128, _DIM))
    with pytest.raises(fcd.CompiledDraftStateChanged, match="changed shape"):
        chain(hidden, primary, uniform)


def test_construction_rejects_a_depth_selected_lora_stack(monkeypatch) -> None:
    import mtplx.mtp_adapters as mtp_adapters

    monkeypatch.setattr(
        mtp_adapters, "iter_mtp_lora_modules", lambda _model: [("fc", object())]
    )
    with pytest.raises(fcd.CompiledDraftUnsupported, match="MTP LoRA"):
        _build(_TinyRuntime(), [_TinyQSACache()])


def test_construction_rejects_a_forward_that_consumes_mtp_depth() -> None:
    rt = _TinyRuntime()
    rt.model.mtp_forward = (
        lambda hidden_states, next_token_ids, mtp_depth=None, **kwargs: None
    )
    with pytest.raises(fcd.CompiledDraftUnsupported, match="consumes mtp_depth"):
        _build(rt, [_TinyQSACache()])


def test_construction_rejects_a_runtime_without_a_model() -> None:
    rt = _TinyRuntime()
    rt.model = None
    with pytest.raises(fcd.CompiledDraftUnsupported, match="a model"):
        _build(rt, [_TinyQSACache()])


def test_construction_rejects_a_model_without_mtp_forward() -> None:
    rt = _TinyRuntime()
    rt.model = SimpleNamespace(mtp=SimpleNamespace())
    with pytest.raises(fcd.CompiledDraftUnsupported, match="model.mtp_forward"):
        _build(rt, [_TinyQSACache()])


def test_construction_rejects_a_cache_that_can_still_grow() -> None:
    with pytest.raises(fcd.CompiledDraftUnsupported, match="request-sized"):
        _build(
            _TinyRuntime(),
            [_TinyQSACache(capacity=8)],
            request_max_tokens=4096,
        )


def test_construction_rejects_a_non_fixed_capacity_cache() -> None:
    cache = [_TinyQSACache()]
    cache[0].fixed_capacity = False
    with pytest.raises(fcd.CompiledDraftUnsupported, match="fixed-capacity"):
        _build(_TinyRuntime(), cache)


def test_construction_rejects_more_than_one_mtp_cache() -> None:
    with pytest.raises(fcd.CompiledDraftUnsupported, match="exactly one"):
        _build(_TinyRuntime(), [_TinyQSACache(), _TinyQSACache()])


def test_construction_rejects_an_empty_state_tree() -> None:
    cache = [_TinyQSACache()]
    with pytest.raises(fcd.CompiledDraftUnsupported, match="no rebindable"):
        fcd.build_compiled_draft_chain(
            rt=_TinyRuntime(),
            mtp_cache=cache,
            state_tree=[],
            mtp_hidden_variant="post_norm",
            selector=_pure_mlx_selector,
            frspec_ids=None,
            depth=_DEPTH,
            top_k=_TOP_K,
            request_max_tokens=8,
        )


def test_construction_rejects_a_non_callable_selector() -> None:
    cache = [_TinyQSACache()]
    with pytest.raises(fcd.CompiledDraftUnsupported, match="prebound"):
        fcd.build_compiled_draft_chain(
            rt=_TinyRuntime(),
            mtp_cache=cache,
            state_tree=_state_tree(cache),
            mtp_hidden_variant="post_norm",
            selector=None,
            frspec_ids=None,
            depth=_DEPTH,
            top_k=_TOP_K,
            request_max_tokens=8,
        )
