"""CPU parity and construction wiring for cache-only MTP history append."""

import inspect
import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import Model, QSACache, Qwen4ExpMTP, TextArgs
from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=32,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=2,
        hc_lowrank=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        partial_rotary_factor=0.5,
    )


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture()
def mtp(_cpu_only):
    mx.random.seed(41)
    module = Qwen4ExpMTP(_tiny_args())
    module.eval()
    mx.eval(module.parameters())
    return module


def _cache_arrays(cache: QSACache):
    return (
        cache.kv.keys,
        cache.kv.values,
        cache.raw_keys,
        cache.pooled,
        cache.pooled_f32_t,
    )


def _assert_cache_exact(actual: QSACache, expected: QSACache) -> None:
    assert actual.offset == expected.offset
    assert actual.pooled_len == expected.pooled_len
    assert actual._reserved_raw_capacity == expected._reserved_raw_capacity
    assert actual._reserved_pooled_capacity == expected._reserved_pooled_capacity
    for got, want in zip(_cache_arrays(actual), _cache_arrays(expected)):
        assert (got is None) == (want is None)
        if got is not None:
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            assert mx.array_equal(got, want).item()


def test_cache_only_history_matches_full_qsa_cache_after_incremental_appends(mtp):
    """The unused decoder output must not be needed to advance QSA state."""

    full_cache = [QSACache(compress_ratio=2)]
    cache_only = [QSACache(compress_ratio=2)]
    for rows in (3, 2, 4):
        widened = mx.random.normal((1, rows, 64)).astype(mx.float32)
        embeddings = mx.random.normal((1, rows, 32)).astype(mx.float32)

        unused_full = mtp.fuse_and_run_history(widened, embeddings, full_cache)
        unused_cache_only = mtp.fuse_and_update_history_cache(
            widened,
            embeddings,
            cache_only,
        )
        mx.eval(
            unused_full,
            unused_cache_only,
            *_cache_arrays(full_cache[0]),
            *_cache_arrays(cache_only[0]),
        )

        assert unused_cache_only.shape == widened.shape
        _assert_cache_exact(cache_only[0], full_cache[0])


def test_cache_only_history_matches_full_with_pre_reserved_backings(mtp):
    full_cache = [QSACache(compress_ratio=2)]
    cache_only = [QSACache(compress_ratio=2)]
    for cache in (full_cache[0], cache_only[0]):
        cache.reserve_indexer_capacity(raw_capacity=512, pooled_capacity=256)

    widened = mx.random.normal((1, 5, 64)).astype(mx.float32)
    embeddings = mx.random.normal((1, 5, 32)).astype(mx.float32)
    unused_full = mtp.fuse_and_run_history(widened, embeddings, full_cache)
    unused_cache_only = mtp.fuse_and_update_history_cache(
        widened,
        embeddings,
        cache_only,
    )
    mx.eval(
        unused_full,
        unused_cache_only,
        *_cache_arrays(full_cache[0]),
        *_cache_arrays(cache_only[0]),
    )

    _assert_cache_exact(cache_only[0], full_cache[0])
    assert cache_only[0].raw_keys.shape[1] == 512
    assert cache_only[0].pooled.shape[1] == 256


def test_constructor_keeps_full_history_even_when_env_is_set(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_CACHE_ONLY_MTP_HISTORY", "1")
    module = Qwen4ExpMTP(_tiny_args())

    assert module._mtp_history_append_impl.__func__ is Qwen4ExpMTP.fuse_and_run_history


def test_explicit_installer_binds_cache_only_once_after_construction():
    module = Qwen4ExpMTP(_tiny_args())
    receipt = module.install_cache_only_history()

    assert (
        module._mtp_history_append_impl.__func__
        is Qwen4ExpMTP.fuse_and_update_history_cache
    )
    assert receipt == {"installed": True, "layer_type": "full_attention"}


def test_installer_rejects_unsupported_non_qsa_topology():
    module = Qwen4ExpMTP(_tiny_args())
    module.layers[0].self_attn.indexer = None

    with pytest.raises(RuntimeError, match="QSA indexer"):
        module.install_cache_only_history()


def test_attach_mtp_installs_opt_in_after_load_and_before_publish():
    source = inspect.getsource(Model.attach_mtp)

    env_read = source.index("MTPLX_QWEN4_CACHE_ONLY_MTP_HISTORY")
    strict_load = source.index("mtp.load_weights")
    install = source.index("mtp.install_cache_only_history()")
    publish = source.index("self.language_model.mtp = mtp")
    assert strict_load < env_read < install < publish
    assert '{"1", "true", "yes", "on"}' in source


def test_cache_only_history_route_is_a_registered_runtime_override():
    assert (
        "MTPLX_QWEN4_CACHE_ONLY_MTP_HISTORY"
        in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    )


def test_cache_only_history_skips_decoder_output_stages_and_roots_cache(
    mtp, monkeypatch
):
    """Cache append stops after QSA K/V; attention output and MoE stay cold."""

    layer = mtp.layers[0]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache-only history executed an output-only stage")

    monkeypatch.setattr(layer.self_attn, "q_norm", forbidden)
    monkeypatch.setattr(layer.self_attn, "o_proj", forbidden)
    monkeypatch.setattr(layer, "mlp", forbidden)
    monkeypatch.setattr(layer, "mlp_hyper_connection", forbidden)

    captured = {}
    original_depends = mx.depends

    def record_depends(value, dependencies):
        captured["value"] = value
        captured["dependencies"] = tuple(dependencies)
        return original_depends(value, dependencies)

    monkeypatch.setattr(mx, "depends", record_depends)

    widened = mx.random.normal((1, 3, 64)).astype(mx.float32)
    embeddings = mx.random.normal((1, 3, 32)).astype(mx.float32)
    cache = [QSACache(compress_ratio=2)]
    result = mtp.fuse_and_update_history_cache(widened, embeddings, cache)
    mx.eval(result, *_cache_arrays(cache[0]))

    assert cache[0].offset == 3
    assert result.shape == captured["value"].shape
    assert {id(leaf) for leaf in captured["dependencies"]} == {
        id(cache[0].kv.keys),
        id(cache[0].kv.values),
        id(cache[0].raw_keys),
        id(cache[0].pooled),
        id(cache[0].pooled_f32_t),
    }


def test_cache_only_history_matches_full_after_trim_and_reappend(mtp):
    full_cache = [QSACache(compress_ratio=2)]
    cache_only = [QSACache(compress_ratio=2)]

    initial_widened = mx.random.normal((1, 5, 64)).astype(mx.float32)
    initial_embeddings = mx.random.normal((1, 5, 32)).astype(mx.float32)
    full_root = mtp.fuse_and_run_history(
        initial_widened,
        initial_embeddings,
        full_cache,
    )
    cache_only_root = mtp.fuse_and_update_history_cache(
        initial_widened,
        initial_embeddings,
        cache_only,
    )
    mx.eval(full_root, cache_only_root)

    assert full_cache[0].trim(2) == cache_only[0].trim(2) == 2
    replacement_widened = mx.random.normal((1, 3, 64)).astype(mx.float32)
    replacement_embeddings = mx.random.normal((1, 3, 32)).astype(mx.float32)
    full_root = mtp.fuse_and_run_history(
        replacement_widened,
        replacement_embeddings,
        full_cache,
    )
    cache_only_root = mtp.fuse_and_update_history_cache(
        replacement_widened,
        replacement_embeddings,
        cache_only,
    )
    mx.eval(
        full_root,
        cache_only_root,
        *_cache_arrays(full_cache[0]),
        *_cache_arrays(cache_only[0]),
    )

    _assert_cache_exact(cache_only[0], full_cache[0])
