"""MTPLX_FABLE_MTP_KV_ONLY_APPEND -- gating, cache equality, skipped work.

The lane claims the per-cycle MTP history append can compute ONLY what the MTP
cache consumes and leave that cache bit-identical.  Everything below runs on
tiny tensors on the CPU stream (the GPU on a development box is usually
holding a guarded benchmark), which is enough to settle:

* **the cache contract.**  A random-input append through
  ``Qwen4ExpMTP.append_kv_only`` leaves every QSA leaf -- attention keys and
  values, the offset, the indexer's raw key stream, the pooled block bank, its
  fp32 mirror and ``pooled_len`` -- bitwise equal to the full
  ``fuse_and_run_history``, at every window width and across enough steps for
  the indexer's selector to engage on the full arm.
* **the skipped work.**  Counting shims over ``q_proj``, ``q_norm``,
  ``o_proj``, the MLP hyper read and the 512-expert MoE record zero calls on
  the KV-only arm and one per append on the full arm; ``mx.export_to_dot``
  over the cache leaves shows no attention/selection primitive survives.
* **the gating.**  Flag off is the pre-change path (no ``kv_only`` keyword
  ever reaches a backend), and an armed flag on an ineligible model RAISES
  with the missing piece named rather than reverting quietly.
* **the compiled route.**  ``write_only`` forces the indexer's existing
  ``update_only`` compiled mode -- the one whose graph shares
  ``raw_next``/``pooled_next`` with every selecting mode.
* **composition** with ``MTPLX_FABLE_DEVICE_K20`` and the depth-4 probe.

What is NOT settled here, and needs the GPU: the compiled ``update_only``
graph's numerics (``_compiled_route_supported`` requires Metal, so the CPU
stream always takes the eager oracle) and the ms/window saving.
"""

from __future__ import annotations

import io
import re

import mlx.core as mx
import numpy as np
import pytest

import mtplx.fable_mtp_kv_only as kv_only_mod
import mtplx.models.qwen4_exp as qwen4_exp


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # set_default_device leaks into every later-collected module (pytest
    # shares one process), so restore it.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _restore_gate():
    previous = kv_only_mod.is_enabled()
    try:
        yield
    finally:
        kv_only_mod._configure_for_test(previous)


RATIO = 2
HIDDEN = 32
HC = 4
WIDENED = HIDDEN * HC


def _args(**overrides):
    fields = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=HC,
        hc_lowrank=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        # budget // ratio == 4 completed blocks, so the selector engages once
        # the history passes 10 tokens -- inside the loops below.
        indexer_budget=8,
        indexer_compress_ratio=RATIO,
        partial_rotary_factor=0.5,
        full_attention_interval=2,
    )
    fields.update(overrides)
    return qwen4_exp.TextArgs(**fields)


def _head(seed: int = 7):
    mx.random.seed(seed)
    return qwen4_exp.Qwen4ExpMTP(_args())


def _rows(width: int, seed: int):
    mx.random.seed(seed)
    widened = mx.random.normal((1, width, WIDENED)).astype(mx.float32)
    embedding = mx.random.normal((1, width, HIDDEN)).astype(mx.float32)
    mx.eval(widened, embedding)
    return widened, embedding


def _leaves(cache):
    entry = cache[0]
    return {
        "offset": entry.offset,
        "pooled_len": entry.pooled_len,
        "keys": entry.kv.keys,
        "values": entry.kv.values,
        "raw_keys": entry.raw_keys,
        "pooled": entry.pooled,
        "pooled_f32_t": entry.pooled_f32_t,
    }


def _assert_same_leaves(full, kv):
    assert set(full) == set(kv)
    for name, expected in full.items():
        actual = kv[name]
        if isinstance(expected, mx.array):
            assert isinstance(actual, mx.array), name
            assert expected.shape == actual.shape, name
            assert expected.dtype == actual.dtype, name
            assert bool(mx.array_equal(expected, actual).item()), name
        else:
            assert expected == actual, name


# ---------------------------------------------------------------------------
# the cache contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_kv_only_append_leaves_cache_bit_identical(width):
    """Every QSA leaf matches the full append, at every committed width."""

    head = _head()
    full_cache = [qwen4_exp.QSACache(RATIO)]
    kv_cache = [qwen4_exp.QSACache(RATIO)]

    selector_engaged = False
    for step in range(12):
        widened, embedding = _rows(width, seed=100 + step)
        hidden = head.fuse_and_run_history(widened, embedding, full_cache)
        mx.eval(hidden)
        assert head.append_kv_only(widened, embedding, kv_cache) is None
        if full_cache[0].offset // RATIO > head.layers[0].self_attn.indexer.block_topk:
            selector_engaged = True

    assert selector_engaged, "history never passed the indexer's dense==sparse bound"
    _assert_same_leaves(_leaves(full_cache), _leaves(kv_cache))


def test_kv_only_append_matches_after_a_rollback():
    """Positional (not append-only) writes stay identical across a trim."""

    head = _head(seed=11)
    full_cache = [qwen4_exp.QSACache(RATIO)]
    kv_cache = [qwen4_exp.QSACache(RATIO)]

    for step in range(6):
        widened, embedding = _rows(3, seed=200 + step)
        mx.eval(head.fuse_and_run_history(widened, embedding, full_cache))
        head.append_kv_only(widened, embedding, kv_cache)

    # Speculative rows are trimmed and re-appended every rejection cycle.
    for cache in (full_cache, kv_cache):
        cache[0].kv.trim(2)
    replay_widened, replay_embedding = _rows(2, seed=999)
    mx.eval(head.fuse_and_run_history(replay_widened, replay_embedding, full_cache))
    head.append_kv_only(replay_widened, replay_embedding, kv_cache)

    _assert_same_leaves(_leaves(full_cache), _leaves(kv_cache))


# ---------------------------------------------------------------------------
# the skipped work
# ---------------------------------------------------------------------------
class _CountingShim:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._wrapped(*args, **kwargs)


_SKIPPED = (
    ("mlp", "layer"),
    ("mlp_hyper_connection", "layer"),
    ("q_proj", "attention"),
    ("q_norm", "attention"),
    ("o_proj", "attention"),
)


def _install_shims(head):
    layer = head.layers[0]
    owners = {"layer": layer, "attention": layer.self_attn}
    shims = {}
    for name, owner_key in _SKIPPED:
        owner = owners[owner_key]
        shim = _CountingShim(getattr(owner, name))
        owner[name] = shim
        shims[name] = shim
    return shims


def test_skipped_work_is_not_executed():
    """The op-count receipt: zero calls on the KV-only arm, one per append
    on the full arm, for every module the lane claims to skip."""

    head = _head(seed=13)
    shims = _install_shims(head)

    kv_cache = [qwen4_exp.QSACache(RATIO)]
    for step in range(5):
        widened, embedding = _rows(2, seed=300 + step)
        head.append_kv_only(widened, embedding, kv_cache)
    assert {name: shim.calls for name, shim in shims.items()} == {
        name: 0 for name, _ in _SKIPPED
    }

    full_cache = [qwen4_exp.QSACache(RATIO)]
    for step in range(5):
        widened, embedding = _rows(2, seed=300 + step)
        mx.eval(head.fuse_and_run_history(widened, embedding, full_cache))
    assert {name: shim.calls for name, shim in shims.items()} == {
        name: 5 for name, _ in _SKIPPED
    }


def test_indexer_selection_is_not_built():
    """The eager selector never runs under ``write_only``."""

    head = _head(seed=17)
    indexer = head.layers[0].self_attn.indexer
    calls = []
    stock_select = indexer._select_eager

    def _counted_select(*args, **kwargs):
        calls.append(1)
        return stock_select(*args, **kwargs)

    indexer._select_eager = _counted_select  # type: ignore[assignment]

    kv_cache = [qwen4_exp.QSACache(RATIO)]
    for step in range(8):
        widened, embedding = _rows(2, seed=400 + step)
        head.append_kv_only(widened, embedding, kv_cache)
    assert calls == []

    full_cache = [qwen4_exp.QSACache(RATIO)]
    for step in range(8):
        widened, embedding = _rows(2, seed=400 + step)
        mx.eval(head.fuse_and_run_history(widened, embedding, full_cache))
    assert calls, "the full append must still build a selection"


#: Primitives only the attention product, the indexer's selector or the MoE
#: can put in a graph.  ``test_cache_leaf_graphs_...`` asserts the full arm's
#: hidden DOES carry them, so the KV-only assertion cannot go vacuous.
_ATTENTION_PRIMITIVES = {
    "ScaledDotProductAttention",
    "Softmax",
    "ArgPartition",
    "Sort",
    "Gather",
    "GatherAxis",
    "GatherMM",
    "GatherQMM",
    "ScatterAxis",
}


def _graph_labels(*outputs):
    buffer = io.StringIO()
    mx.export_to_dot(buffer, *[o for o in outputs if isinstance(o, mx.array)])
    return set(re.findall(r'label ="([^"]+)"', buffer.getvalue()))


def test_cache_leaf_graphs_carry_no_attention_or_selection():
    """The queued graph the cache leaves depend on holds writes only."""

    head = _head(seed=19)
    kv_cache = [qwen4_exp.QSACache(RATIO)]
    for step in range(6):
        widened, embedding = _rows(2, seed=500 + step)
        head.append_kv_only(widened, embedding, kv_cache)
        # Leave the LAST append unevaluated so its graph is inspectable.
        if step < 5:
            mx.eval(*[v for v in _leaves(kv_cache).values() if isinstance(v, mx.array)])

    leaves = _leaves(kv_cache)
    labels = _graph_labels(*leaves.values())
    assert labels, "expected a live graph on the cache leaves"
    assert not (labels & _ATTENTION_PRIMITIVES), sorted(labels & _ATTENTION_PRIMITIVES)

    # The control: the full append's hidden carries exactly what the KV-only
    # leaves do not, so the assertion above is a real separation.
    full_cache = [qwen4_exp.QSACache(RATIO)]
    for step in range(6):
        widened, embedding = _rows(2, seed=500 + step)
        hidden = head.fuse_and_run_history(widened, embedding, full_cache)
        if step < 5:
            mx.eval(hidden)
    assert _graph_labels(hidden) & _ATTENTION_PRIMITIVES


# ---------------------------------------------------------------------------
# the compiled route
# ---------------------------------------------------------------------------
def test_write_only_forces_the_update_only_compiled_mode(monkeypatch):
    """``write_only`` picks the existing update_only graph, not a new one."""

    head = _head(seed=23)
    indexer = head.layers[0].self_attn.indexer
    seen = []

    monkeypatch.setattr(
        qwen4_exp.QSAIndexer, "_compiled_route_supported", lambda *a, **k: True
    )
    monkeypatch.setattr(
        qwen4_exp.QSAIndexer,
        "_call_rows_compiled",
        lambda self, hidden, pos_start, cache, qk_rows, *, mode: seen.append(mode),
    )
    monkeypatch.setattr(
        qwen4_exp.QSAIndexer,
        "_compiled_mode",
        lambda self, **kwargs: "dense_mask",
    )

    cache = qwen4_exp.QSACache(RATIO)
    hidden = mx.zeros((1, 2, HIDDEN), dtype=mx.float32)
    indexer(hidden, 0, cache, write_only=False)
    indexer(hidden, 0, cache, write_only=True)
    assert seen == ["dense_mask", "update_only"]


# ---------------------------------------------------------------------------
# the gating
# ---------------------------------------------------------------------------
class _FakeIndexer:
    def __call__(self, hidden, pos_start, cache, qk_rows=None, *, write_only=False):
        return None


class _FakeAttention:
    def __init__(self):
        self.indexer = _FakeIndexer()

    def __call__(self, x, cache, *, kv_only=False):
        return None


class _FakeLayer:
    is_linear = False

    def __init__(self):
        self.self_attn = _FakeAttention()

    def __call__(self, hidden, *, input_ids, ssm_mask, cache, kv_only=False):
        return None


class _FakeHead:
    def __init__(self):
        self.layers = [_FakeLayer()]

    def append_kv_only(self, widened, tok_emb, cache):
        return None


class _FakeInner:
    def __init__(self):
        self.mtp = _FakeHead()

    def mtp_update_cache(self, hidden_states, next_token_ids, kv_only=False):
        return None


class _FakeModel:
    def __init__(self):
        self.language_model = _FakeInner()


def test_flag_off_claims_nothing():
    kv_only_mod._configure_for_test(False)
    assert kv_only_mod.is_enabled() is False
    assert kv_only_mod.claim_model_route(_FakeModel()) is False
    # Even a model with no MTP at all is silent when the flag is off.
    assert kv_only_mod.claim_model_route(object()) is False


def test_armed_claim_accepts_the_eligible_shape():
    kv_only_mod._configure_for_test(True)
    assert kv_only_mod.claim_model_route(_FakeModel()) is True


def test_armed_claim_accepts_the_real_head():
    """The shipped ``Qwen4ExpMTP`` + ``TextModel`` signatures are eligible."""

    kv_only_mod._configure_for_test(True)

    class _RealShapedInner:
        def __init__(self, head):
            self.mtp = head

        mtp_update_cache = qwen4_exp.TextModel.mtp_update_cache

    class _RealShapedModel:
        def __init__(self, head):
            self.language_model = _RealShapedInner(head)

    assert kv_only_mod.claim_model_route(_RealShapedModel(_head(seed=29))) is True


def _model_without_update():
    model = _FakeModel()

    class _NoUpdate:
        def __init__(self, mtp):
            self.mtp = mtp

    model.language_model = _NoUpdate(_FakeHead())
    return model


def _model_update_without_kv_only():
    model = _FakeModel()

    class _NoKeyword:
        def __init__(self, mtp):
            self.mtp = mtp

        def mtp_update_cache(self, hidden_states, next_token_ids, mtp_cache=None):
            return None

    model.language_model = _NoKeyword(_FakeHead())
    return model


def _model_without_head():
    model = _FakeModel()
    model.language_model.mtp = None
    return model


def _model_with_two_layers():
    model = _FakeModel()
    model.language_model.mtp.layers = [_FakeLayer(), _FakeLayer()]
    return model


def _model_with_a_linear_layer():
    model = _FakeModel()

    class _LinearLayer(_FakeLayer):
        is_linear = True

    model.language_model.mtp.layers = [_LinearLayer()]
    return model


def _model_without_self_attn():
    model = _FakeModel()
    model.language_model.mtp.layers[0].self_attn = None
    return model


def _model_without_indexer():
    model = _FakeModel()
    model.language_model.mtp.layers[0].self_attn.indexer = None
    return model


@pytest.mark.parametrize(
    "build, expected",
    [
        (_model_without_update, "mtp_update_cache"),
        (_model_update_without_kv_only, "takes kv_only"),
        (_model_without_head, "attached MTP head"),
        (_model_with_two_layers, "one-DecoderLayer"),
        (_model_with_a_linear_layer, "linear attention"),
        (_model_without_self_attn, "self_attn"),
        (_model_without_indexer, "QSA indexer"),
    ],
)
def test_armed_claim_raises_on_every_missing_piece(build, expected):
    kv_only_mod._configure_for_test(True)
    with pytest.raises(RuntimeError, match=expected):
        kv_only_mod.claim_model_route(build())


def test_armed_claim_rejects_kwargs_only_backends():
    """``**kwargs`` is not evidence of a KV-only route."""

    kv_only_mod._configure_for_test(True)

    class _ForwardingInner:
        def __init__(self):
            self.mtp = _FakeHead()

        def mtp_update_cache(self, hidden_states, next_token_ids, **kwargs):
            return None

    class _ForwardingModel:
        def __init__(self):
            self.language_model = _ForwardingInner()

    with pytest.raises(RuntimeError, match="takes kv_only"):
        kv_only_mod.claim_model_route(_ForwardingModel())


# ---------------------------------------------------------------------------
# the runtime shim: flag off must not change any backend call
# ---------------------------------------------------------------------------
class _RecordingRuntimeModel:
    def __init__(self):
        self.seen = None

    def mtp_update_cache(self, hidden_states, next_token_ids, **kwargs):
        self.seen = dict(kwargs)
        return "hidden"


def _runtime_with(model):
    from mtplx.runtime import MTPLXRuntime

    runtime = MTPLXRuntime.__new__(MTPLXRuntime)
    runtime.model = model
    runtime.mtp_enabled = True
    runtime.contract = type(
        "_Contract", (), {"hidden_variant": "pre_mixer", "concat_order": "hidden_first"}
    )()
    runtime.diagnostic_counters = {}
    return runtime


def test_runtime_offers_kv_only_only_when_armed():
    model = _RecordingRuntimeModel()
    runtime = _runtime_with(model)

    runtime.update_mtp_cache("h", "t", mtp_cache=None)
    assert "kv_only" not in model.seen

    runtime.update_mtp_cache("h", "t", mtp_cache=None, kv_only=True)
    assert model.seen["kv_only"] is True


def test_runtime_raises_when_the_backend_cannot_take_kv_only():
    class _NoKwargs:
        def mtp_update_cache(self, hidden_states, next_token_ids, mtp_cache=None):
            return "hidden"

    runtime = _runtime_with(_NoKwargs())
    with pytest.raises(RuntimeError, match="does not accept kv_only"):
        runtime.update_mtp_cache("h", "t", mtp_cache=None, kv_only=True)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def test_composes_with_device_k20_and_the_depth4_probe():
    """Three independent module gates; the probe's offset contract holds over
    a KV-only append."""

    import mtplx.fable_depth4_probe as depth4
    import mtplx.fable_device_k20 as device_k20

    kv_only_mod._configure_for_test(True)
    saved_probe = depth4.is_enabled()
    saved_k20 = device_k20.is_enabled()
    depth4._configure_for_test(True)
    device_k20._configure_for_test(True)
    try:
        assert kv_only_mod.claim_model_route(_FakeModel()) is True
        assert depth4.is_enabled() is True
        assert device_k20.is_enabled() is True

        head = _head(seed=31)
        cache = [qwen4_exp.QSACache(RATIO)]
        for step in range(4):
            widened, embedding = _rows(2, seed=600 + step)
            head.append_kv_only(widened, embedding, cache)
        after_append = cache[0].offset

        # The probe stages one speculative row on the SAME cache the KV-only
        # append just advanced, and must hand it back untouched.
        def draft_step():
            widened, embedding = _rows(1, seed=700)
            head.fuse_and_run_history(widened, embedding, cache)
            return None

        class _Row:
            token_ids = np.array([1, 2], dtype=np.int64)
            probs = np.array([0.75, 0.25], dtype=np.float64)

        ids, probs, trimmed = depth4.run_probe(
            draft_step=draft_step,
            shape_row=lambda _row: _Row(),
            mtp_cache=cache,
            read_offset=lambda c: int(c[0].offset),
            rollback=lambda c, offset: c[0].kv.trim(int(c[0].offset) - offset),
        )
        assert trimmed is False
        assert list(ids) == [1, 2]
        assert cache[0].offset == after_append
    finally:
        depth4._configure_for_test(saved_probe)
        device_k20._configure_for_test(saved_k20)


# ---------------------------------------------------------------------------
# the generation wiring
# ---------------------------------------------------------------------------
def test_generation_wires_the_flag_once_and_defaults_off():
    """Flag off is the pre-change lane: constant False, kv_only never True."""

    import inspect

    import mtplx.generation as generation

    assert generation._FABLE_MTP_KV_ONLY_APPEND is False

    append_source = inspect.getsource(generation._append_mtp_history)
    assert "kv_only=kv_only" in append_source
    # force_eval + kv_only must materialize the CACHE, never `_eval(None)`.
    assert "_eval_cache_roots(mtp_cache)" in append_source

    loop_source = inspect.getsource(generation.generate_mtpk)
    assert "_fable_mtp_kv_only_claim(rt.model)" in loop_source
    assert "kv_only=_mtp_kv_only_append" in loop_source
    # Claimed exactly once per request, not per cycle.
    assert loop_source.count("_fable_mtp_kv_only_claim(") == 1

    # The two prefill appends stay on the full path (separate change,
    # separate ABBA) -- they are the only force_eval=True call sites.
    assert loop_source.count("kv_only=") == 1


def test_force_eval_materializes_the_cache_not_a_none_hidden(monkeypatch):
    """Regression: the KV-only append has no hidden for ``_eval`` to take."""

    import mtplx.generation as generation

    evaluated = []
    monkeypatch.setattr(
        generation, "_eval_cache_roots", lambda cache: evaluated.append(cache)
    )
    monkeypatch.setattr(
        generation,
        "_eval",
        lambda *values, **kwargs: pytest.fail("_eval must not see a None hidden"),
    )
    monkeypatch.setenv("MTPLX_LAZY_MTP_HISTORY_APPEND", "0")

    sentinel = ["mtp-cache"]

    class _Runtime:
        def update_mtp_cache(self, hidden_states, token_ids, **kwargs):
            assert kwargs["kv_only"] is True
            return None

    monkeypatch.setattr(generation, "_runtime_count", lambda *a, **k: None)
    elapsed = generation._append_mtp_history(
        _Runtime(),
        sentinel,
        mx.zeros((1, 2, HIDDEN)),
        [3, 4],
        phase="ar_decode",
        mtp_hidden_variant="pre_mixer",
        force_eval=True,
        kv_only=True,
    )
    assert elapsed >= 0.0
    assert evaluated == [sentinel]
