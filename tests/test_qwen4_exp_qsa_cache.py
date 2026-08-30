"""QSACache must be a full citizen of the cache contract.

The QSA indexer keeps its own raw-key stream (and derived pooled block keys)
next to the attention KV. The serve loop rolls caches back after every
speculative verify round (``rollback_after_verify``: trim for trimmable
entries, snapshot-restore for the rest) and resumes banked sessions through
``state``. A raw-key stream that only ever appends desyncs from the KV on the
first rollback; once the context crosses the indexer's engage threshold the
selection mask is built from the raw-stream length while attention keys come
from the KV — the ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash
OpenCode hit live at 3.7k ctx (2026-08-27). Below the threshold the same
desync corrupts pooled blocks silently instead of crashing.

All runs are CPU (M-series GPU fp32 matmul is reduced-precision; CPU is the
parity surface).
"""

import inspect

import mlx.core as mx
import pytest

import mtplx.graphbank as graphbank
from mtplx.cache_state import (
    rollback_after_verify,
    snapshot_untrimmable_cache,
)
from mtplx.models.qwen4_exp import Attention, QSACache, QSAIndexer, TextArgs


def _tiny_args(*, compress_ratio: int = 2) -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=compress_ratio,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


@pytest.fixture()
def ratio4_attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_tiny_args(compress_ratio=4))
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


PREFILL = 12  # engage threshold with budget=8/ratio=2 is >8 visible tokens
STEP = 4  # a depth-3 verify round: 1 committed + 3 drafts


def test_rollback_then_forward_matches_fresh_run(attn):
    """A rejected verify round must leave the QSA layer exactly where a run
    that never saw the rejected tokens would be."""
    x_pre = _hidden(PREFILL, seed=1)
    x_rejected = _hidden(STEP, seed=2)
    x_next = _hidden(STEP, seed=3)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    assert cache[0].offset == PREFILL
    out = attn(x_next, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert out.shape == golden.shape
    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_state_roundtrip_resumes_identically(attn):
    """Bank restore: ``state`` must carry everything the layer needs — a
    resumed session past the engage threshold selects the same blocks and
    produces the same output as the uninterrupted run."""
    x_pre = _hidden(PREFILL, seed=4)
    x_next = _hidden(STEP, seed=5)

    live = QSACache()
    attn(x_pre, live)
    golden = attn(x_next, live)

    donor = QSACache()
    attn(x_pre, donor)
    resumed = QSACache()
    resumed.state = donor.state
    assert resumed.offset == PREFILL
    out = attn(x_next, resumed)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_trim_contract(attn):
    """QSACache is trimmable: trim rolls the layer back token-exactly,
    including through a pooled-block boundary."""
    cache = QSACache()
    assert cache.is_trimmable()

    x_pre = _hidden(PREFILL, seed=6)
    x_tail = _hidden(3, seed=7)  # odd length: trims back through a block edge
    x_next = _hidden(STEP, seed=8)

    attn(x_pre, cache)
    attn(x_tail, cache)
    assert cache.trim(3) == 3
    assert cache.offset == PREFILL
    out = attn(x_next, cache)

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_rollback_below_engage_threshold_still_exact(attn):
    """The desync is silent below the engage threshold (dense mask hides it);
    the pooled stream must still be positionally correct once the session
    grows past it."""
    x_pre = _hidden(4, seed=9)
    x_rejected = _hidden(STEP, seed=10)
    # two accepted rounds carry the session across the threshold
    x_a = _hidden(STEP, seed=11)
    x_b = _hidden(STEP, seed=12)
    x_c = _hidden(STEP, seed=13)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    for chunk in (x_a, x_b, x_c):
        out = attn(chunk, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    for chunk in (x_a, x_b, x_c):
        golden = attn(chunk, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_tensor_offset_qsa_cache_trim_matches_stock(attn):
    """The compiled-verifier cache owns fixed banks without changing QSA math."""
    x_pre = _hidden(PREFILL, seed=14)
    x_rejected = _hidden(STEP, seed=15)
    x_next = _hidden(STEP, seed=16)

    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, cache[0])
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache,
        reserve_tokens=STEP,
        initial_reserve_tokens=16,
    )

    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL

    attn(x_rejected, cache[0])
    assert cache[0].trim(STEP) == STEP
    out = attn(x_next, cache[0])

    fresh = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_compiled_verify_bank_threads_qsa_state_without_fallback(attn):
    class TinyQSARuntime:
        def __init__(self):
            mx.random.seed(17)
            self.attn = attn
            self.embed = mx.random.normal((32, 64)).astype(mx.float32)
            self.head = mx.random.normal((64, 32)).astype(mx.float32)

        def forward_ar_capture(
            self,
            input_ids,
            *,
            cache,
            return_hidden=True,
            hidden_variant=None,
            capture_backend=None,
        ):
            del hidden_variant, capture_backend
            hidden = self.attn(self.embed[input_ids], cache[0])
            logits = hidden @ self.head
            return logits, hidden, {}

    rt = TinyQSARuntime()
    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    rt.forward_ar_capture(
        mx.arange(PREFILL, dtype=mx.int32).reshape(1, -1), cache=cache
    )
    bank = graphbank.CompiledVerifyBank(rt, request_max_tokens=16)

    bank.forward_ar_capture(mx.array([[1, 2, 3, 4]]), cache=cache)
    bank.forward_ar_capture(mx.array([[5, 6, 7, 8]]), cache=cache)

    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["traces"] == 1
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL + 8


def test_installed_fixed_m4_direct_pool_survives_chained_donation_growth(
    ratio4_attn,
    monkeypatch,
):
    class TinyFixedM4Runtime:
        qwen4_fixed_m4_compiled_verify = True

        def __init__(self):
            mx.random.seed(18)
            self.attn = ratio4_attn
            self.embed = mx.random.normal((32, 64)).astype(mx.float32)
            self.head = mx.random.normal((64, 32)).astype(mx.float32)

        def forward_ar_capture(
            self,
            input_ids,
            *,
            cache,
            return_hidden=True,
            hidden_variant=None,
            capture_backend=None,
            compiled_aux=None,
        ):
            del return_hidden, hidden_variant, capture_backend, compiled_aux
            hidden = self.attn(self.embed[input_ids], cache[0])
            logits = hidden @ self.head
            return logits, hidden, {}

        def prepare_compiled_verify_aux(self, input_ids, cache):
            del input_ids, cache
            return mx.array(0, dtype=mx.int32)

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "4")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_DONATION", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER", "0")
    runtime = TinyFixedM4Runtime()
    installed = [QSACache(compress_ratio=4)]
    reference = [QSACache(compress_ratio=4)]
    prompt = list(range(8))
    prompt_ids = mx.array([prompt], dtype=mx.int32)
    installed_prefill = runtime.forward_ar_capture(prompt_ids, cache=installed)
    reference_prefill = runtime.forward_ar_capture(prompt_ids, cache=reference)
    mx.eval(
        installed_prefill[0],
        installed_prefill[1],
        reference_prefill[0],
        reference_prefill[1],
    )

    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=4,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        installed,
        prompt_ids=prompt,
        hidden_variant=None,
    )
    reference_bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=4,
        request_max_tokens=16,
    )
    reference_bank.install_fixed_m4(
        reference,
        prompt_ids=prompt,
        hidden_variant=None,
    )

    def bind_compiled_pooled_route(compiled_bank, enabled):
        dispatch = compiled_bank._fixed_m4_dispatch
        for entry in dispatch["qsa_entries"]:
            entry.fixed_m4_direct_pooled = enabled
        for idx, kind, _n in compiled_bank._spec:
            if kind == graphbank.VERIFY_SPEC_KIND_QSA:
                compiled_bank._shadow[idx].fixed_m4_direct_pooled = enabled
        compiled_bank._compiled.clear()
        route_key = int(
            all(entry.fixed_rows_gather for entry in dispatch["qsa_entries"])
        )
        key = (4, "", route_key)
        fn = compiled_bank._shared_or_new_verify_step(key, 4, None)
        compiled_bank._compiled[key] = fn
        dispatch["fn"] = fn

    # Both installations initially own the direct route. Rebind the oracle
    # to a separately keyed marker-off compiled graph, then rebind direct last
    # so each shared trace host points at the bank that will execute it.
    bind_compiled_pooled_route(reference_bank, False)
    bind_compiled_pooled_route(bank, True)
    initial_capacity = installed[0].capacity
    assert reference[0].capacity == initial_capacity
    assert reference[0].fixed_m4_direct_pooled is False
    assert bank._fixed_m4_dispatch["donate"] is True
    assert reference_bank._fixed_m4_dispatch["donate"] is True

    completion_tokens = []
    for ordinal, host_ids in enumerate(([8, 9, 10, 11], [12, 13, 14, 15])):
        ids = mx.array([host_ids], dtype=mx.int32)
        completion_tokens.extend(host_ids)
        actual_logits, actual_hidden, captures = bank.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=ordinal * 4,
            cache=installed,
        )
        expected_logits, expected_hidden, reference_captures = (
            reference_bank.forward_fixed_m4(
                ids,
                host_input_ids=host_ids,
                completion_tokens=completion_tokens,
                committed_count=ordinal * 4,
                cache=reference,
            )
        )
        installed_leaves = installed[0].state_leaves
        reference_leaves = reference[0].state_leaves
        assert len(installed_leaves) == len(reference_leaves) == 5
        mx.eval(
            actual_logits,
            actual_hidden,
            expected_logits,
            expected_hidden,
            *installed_leaves,
            *reference_leaves,
        )
        assert captures == reference_captures == {}
        assert mx.array_equal(actual_logits, expected_logits).item()
        assert mx.array_equal(actual_hidden, expected_hidden).item()
        for installed_leaf, reference_leaf in zip(
            installed_leaves, reference_leaves
        ):
            assert installed_leaf.shape == reference_leaf.shape
            assert installed_leaf.dtype == reference_leaf.dtype
            assert mx.array_equal(installed_leaf, reference_leaf).item()
        if ordinal == 0:
            assert (
                installed[0].capacity
                == reference[0].capacity
                == initial_capacity
            )
            assert bank.stats["fixed_m4_capacity_transitions"] == 0
            assert reference_bank.stats["fixed_m4_capacity_transitions"] == 0

    assert installed[0].capacity == reference[0].capacity > initial_capacity
    assert installed[0].fixed_m4_direct_pooled is True
    assert bank._shadow[0].fixed_m4_direct_pooled is True
    assert reference[0].fixed_m4_direct_pooled is False
    assert reference_bank._shadow[0].fixed_m4_direct_pooled is False
    assert bank.stats["fixed_m4_capacity_transitions"] == 1
    assert reference_bank.stats["fixed_m4_capacity_transitions"] == 1
    assert bank.stats["compiled_calls"] == 2
    assert reference_bank.stats["compiled_calls"] == 2
    assert bank.stats["fallback_calls"] == 0
    assert reference_bank.stats["fallback_calls"] == 0


@pytest.mark.parametrize("residue", [0, 1, 2, 3])
def test_fixed_m4_pooled_update_matches_generic_for_every_ratio4_residue(
    ratio4_attn,
    monkeypatch,
    residue,
):
    """The direct one-block M4 update is byte-exact at every input frontier."""

    prefill = 8 + residue
    x_pre = _hidden(prefill, seed=30 + residue)
    x_step = _hidden(4, seed=40 + residue)
    direct = [QSACache(compress_ratio=4)]
    generic = [QSACache(compress_ratio=4)]
    ratio4_attn(x_pre, direct[0])
    ratio4_attn(x_pre, generic[0])
    for cache in (direct, generic):
        promoted, failures = graphbank.promote_kv_cache_offsets(
            cache,
            reserve_tokens=4,
            initial_reserve_tokens=16,
        )
        assert promoted == 1
        assert failures == {}
        assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
        assert cache[0].size() % 4 == residue

    class FixedM4Owner:
        qwen4_fixed_m4_compiled_verify = True

        def forward_ar_capture(
            self,
            input_ids,
            *,
            cache,
            return_hidden=True,
            hidden_variant=None,
            capture_backend=None,
        ):
            raise AssertionError("installation must not execute the verifier")

    bank = graphbank.CompiledVerifyBank(
        FixedM4Owner(),
        max_verify_len=4,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        direct,
        prompt_ids=list(range(prefill)),
        hidden_variant=None,
    )
    assert direct[0].fixed_m4_direct_pooled is True
    assert bank._shadow[0].fixed_m4_direct_pooled is True
    assert generic[0].fixed_m4_direct_pooled is False

    selections = []
    original_indexer_call = QSAIndexer.__call__

    def capture_selection(self, *args, **kwargs):
        result = original_indexer_call(self, *args, **kwargs)
        if isinstance(result, tuple):
            route_tag = result[0]
            tensors = result[1:]
        elif result is None:
            route_tag = "update_only"
            tensors = ()
        else:
            route_tag = "dense_mask"
            tensors = (result,)
        selections.append((route_tag, tensors))
        return result

    monkeypatch.setattr(QSAIndexer, "__call__", capture_selection)
    direct_out = ratio4_attn(x_step, direct[0])
    monkeypatch.setattr(
        QSAIndexer,
        "_extend_pooled_fixed",
        QSAIndexer._extend_pooled_fixed_generic,
    )
    generic_out = ratio4_attn(x_step, generic[0])

    direct_leaves = direct[0].state_leaves
    generic_leaves = generic[0].state_leaves
    assert len(selections) == 2
    (direct_tag, direct_selection), (generic_tag, generic_selection) = selections
    assert direct_tag == generic_tag
    assert len(direct_selection) == len(generic_selection)
    assert len(direct_leaves) == len(generic_leaves) == 5
    mx.eval(
        direct_out,
        generic_out,
        *direct_selection,
        *generic_selection,
        *direct_leaves,
        *generic_leaves,
    )
    for direct_tensor, generic_tensor in zip(
        direct_selection, generic_selection
    ):
        assert direct_tensor.shape == generic_tensor.shape
        assert direct_tensor.dtype == generic_tensor.dtype
        assert mx.array_equal(direct_tensor, generic_tensor).item()
    assert direct_out.shape == generic_out.shape
    assert direct_out.dtype == generic_out.dtype
    assert mx.array_equal(direct_out, generic_out).item()
    for direct_leaf, generic_leaf in zip(direct_leaves, generic_leaves):
        assert direct_leaf.shape == generic_leaf.shape
        assert direct_leaf.dtype == generic_leaf.dtype
        assert mx.array_equal(direct_leaf, generic_leaf).item()


def test_non_owner_fixed_m4_uses_generic_clamp_at_exhausted_capacity(
    ratio4_attn,
    monkeypatch,
):
    cache = [QSACache(compress_ratio=4)]
    ratio4_attn(_hidden(8, seed=50), cache[0])
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache,
        reserve_tokens=4,
        initial_reserve_tokens=4,
    )
    assert promoted == 1
    assert failures == {}
    fixed = cache[0]
    assert isinstance(fixed, graphbank.TensorOffsetQSACache)
    assert fixed.fixed_m4_direct_pooled is False

    generic_calls = []
    original_generic = QSAIndexer._extend_pooled_fixed_generic

    def observed_generic(self, cache, total):
        generic_calls.append(True)
        return original_generic(self, cache, total)

    monkeypatch.setattr(
        QSAIndexer,
        "_extend_pooled_fixed_generic",
        observed_generic,
    )
    fixed.kv.offset = fixed.capacity
    fixed._last_write_rows = 4
    pooled = ratio4_attn.indexer._extend_pooled_fixed(
        fixed,
        fixed.offset + 4,
    )
    mx.eval(pooled)

    assert generic_calls == [True]
    assert pooled.shape == fixed.pooled.shape


def test_fixed_m4_pooled_update_removes_only_direct_route_guards():
    route = inspect.getsource(QSAIndexer._extend_pooled_fixed)
    direct = inspect.getsource(QSAIndexer._extend_pooled_fixed_m4)
    generic = inspect.getsource(QSAIndexer._extend_pooled_fixed_generic)

    assert "cache.fixed_m4_direct_pooled" in route
    assert "step_rows == self.ratio == 4" in route
    assert "mx.minimum" not in direct
    assert "mx.where" not in direct
    assert "mx.slice_update" in direct
    assert "mx.minimum" in generic
    assert "mx.where" in generic


@pytest.mark.parametrize("shared_traces", ["0", "1"])
def test_shared_graph_refuses_mixed_qsa_direct_pool_ownership(
    ratio4_attn,
    monkeypatch,
    shared_traces,
):
    class FixedM4Owner:
        qwen4_fixed_m4_compiled_verify = True

    cache = [QSACache(compress_ratio=4), QSACache(compress_ratio=4)]
    prefill = _hidden(8, seed=51)
    for entry in cache:
        ratio4_attn(prefill, entry)
    bank = graphbank.CompiledVerifyBank(
        FixedM4Owner(),
        max_verify_len=4,
        request_max_tokens=16,
    )

    class M4Shape:
        shape = (1, 4)

    assert bank._fallback_reason(M4Shape(), cache, True) is None
    bank._ensure_shadow(cache)
    bank._shadow[0].fixed_m4_direct_pooled = True
    bank._shadow[1].fixed_m4_direct_pooled = False
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_SHARED_TRACES", shared_traces)

    with pytest.raises(RuntimeError, match="mixed QSA direct-pooled ownership"):
        bank._shared_or_new_verify_step((4, "", 0), 4, None)


def test_fixed_m4_aux_failure_does_not_publish_direct_pool_ownership(
    ratio4_attn,
    monkeypatch,
):
    class FailingAuxOwner:
        qwen4_fixed_m4_compiled_verify = True

        def build_fixed_m4_compiled_verify_aux(self, cache, prompt_ids):
            del cache, prompt_ids
            raise RuntimeError("aux construction failed")

    cache = [QSACache(compress_ratio=4)]
    ratio4_attn(_hidden(8, seed=52), cache[0])
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")
    bank = graphbank.CompiledVerifyBank(
        FailingAuxOwner(),
        max_verify_len=4,
        request_max_tokens=16,
    )
    with pytest.raises(RuntimeError, match="aux construction failed"):
        bank.install_fixed_m4(
            cache,
            prompt_ids=list(range(8)),
            hidden_variant=None,
        )

    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].fixed_m4_direct_pooled is False
    assert bank._shadow[0].fixed_m4_direct_pooled is False
    assert bank._fixed_m4_dispatch is None

    generic_calls = []
    original_generic = QSAIndexer._extend_pooled_fixed_generic

    def observed_generic(self, cache, total):
        generic_calls.append(True)
        return original_generic(self, cache, total)

    monkeypatch.setattr(
        QSAIndexer,
        "_extend_pooled_fixed_generic",
        observed_generic,
    )
    cache[0]._last_write_rows = 4
    pooled = ratio4_attn.indexer._extend_pooled_fixed(
        cache[0],
        cache[0].offset + 4,
    )
    mx.eval(pooled)
    assert generic_calls == [True]


def test_fixed_m4_compile_failure_rolls_back_markers_and_local_bank(
    ratio4_attn,
    monkeypatch,
):
    class FixedM4Owner:
        qwen4_fixed_m4_compiled_verify = True

    cache = [QSACache(compress_ratio=4)]
    ratio4_attn(_hidden(8, seed=53), cache[0])
    bank = graphbank.CompiledVerifyBank(
        FixedM4Owner(),
        max_verify_len=4,
        request_max_tokens=16,
    )
    prior_key = (1, "prior", 0)
    prior_fn = object()
    bank._compiled[prior_key] = prior_fn

    def fail_compile(*_args, **_kwargs):
        raise RuntimeError("compile selection failed")

    monkeypatch.setattr(bank, "_shared_or_new_verify_step", fail_compile)
    with pytest.raises(RuntimeError, match="compile selection failed"):
        bank.install_fixed_m4(
            cache,
            prompt_ids=list(range(8)),
            hidden_variant=None,
        )

    assert cache[0].fixed_m4_direct_pooled is False
    assert bank._shadow[0].fixed_m4_direct_pooled is False
    assert bank._compiled == {prior_key: prior_fn}
    assert bank._fixed_m4_dispatch is None


def test_fixed_m4_installer_owns_direct_pooled_route_before_compile():
    install = inspect.getsource(graphbank.CompiledVerifyBank.install_fixed_m4)
    shadow = inspect.getsource(graphbank.CompiledVerifyBank._ensure_shadow)
    parity_clone = inspect.getsource(
        graphbank.CompiledVerifyBank._parity2_clone_cache
    )
    shared_graph = inspect.getsource(
        graphbank.CompiledVerifyBank._shared_or_new_verify_step
    )
    compile_at = install.index("self._shared_or_new_verify_step")
    aux_at = install.index("self._build_fixed_m4_aux(cache, prompt_ids)")
    owner_at = install.index("fixed_m4_direct_pooled = True")
    clear_at = install.index("self._compiled.clear()")
    publish_at = install.index("self._fixed_m4_dispatch = dispatch")

    assert aux_at < owner_at < clear_at < compile_at < publish_at
    assert "shadow_entry.fixed_m4_direct_pooled = True" in install
    assert "entry.fixed_m4_direct_pooled = live_marker" in install
    assert "shadow_entry.fixed_m4_direct_pooled = shadow_marker" in install
    assert "self._compiled.update(prior_compiled)" in install
    assert (
        "fixed_m4_direct_pooled=entry.fixed_m4_direct_pooled" in shadow
    )
    assert "fixed_m4_direct_pooled=" not in parity_clone
    assert "fixed_m4_direct_pooled" in shared_graph
    assert "mixed QSA direct-pooled ownership" in shared_graph
