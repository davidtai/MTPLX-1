from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import MiaNVFP4Rows  # noqa: E402
from mtplx.models import deepseek_v4 as target_module  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4NVFP4Cache,
    Model,
    ModelArgs,
    is_deepseek_v4_mtp_config,
)
import mtplx.models.deepseek_v4_dspark as dspark_module  # noqa: E402
from mtplx.models.deepseek_v4_dspark import (  # noqa: E402
    DSparkTargetRoute,
    DeepseekV4DSparkCache,
    _dspark_draft_positions,
    _dspark_visibility_indices,
    build_deepseek_v4_dspark,
    greedy_future_tokens,
)


class _SpyMarkovHead:
    def __init__(self) -> None:
        self.inputs: list[int] = []

    def __call__(self, token_ids: mx.array):
        self.inputs.append(int(token_ids.item()))
        batch = int(token_ids.shape[0])
        return mx.zeros((batch, 64)), mx.zeros((batch, 8))


def test_primary_token_conditions_dspark_row_zero_and_returns_five_future_tokens() -> None:
    primary = mx.array([29], dtype=mx.int32)
    neural_logits = mx.full((1, 5, 64), -100.0)
    wanted = (31, 32, 33, 34, 35)
    for row, token in enumerate(wanted):
        neural_logits[:, row, token] = 100.0
    markov = _SpyMarkovHead()

    future = greedy_future_tokens(neural_logits, primary, markov)

    assert tuple(future.shape) == (1, 5)
    assert tuple(np.array(future)[0]) == wanted
    assert markov.inputs == [29, 31, 32, 33, 34]
    assert 11 not in markov.inputs
    assert 29 not in tuple(np.array(future)[0])


def test_each_dspark_stage_owns_distinct_mia_nvfp4_cache() -> None:
    caches = [DeepseekV4DSparkCache(window_size=8, head_dim=512) for _ in range(3)]
    assert len({id(cache) for cache in caches}) == 3
    assert len({id(cache.ring) for cache in caches}) == 3
    assert all(isinstance(cache.ring, MiaNVFP4Rows) for cache in caches)
    assert all(cache.ring.mode == "nvfp4_stock432" for cache in caches)
    assert all(cache.ring.record_bytes == 432 for cache in caches)

    prompt_latent = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    prompt_rope = mx.zeros((1, 3, 64), dtype=mx.bfloat16)
    caches[0].prefill(prompt_latent, prompt_rope)
    assert len(caches[0].ring) == 8
    assert len(caches[1].ring) == 0
    assert len(caches[2].ring) == 0


def test_dspark_cache_commits_authoritative_main_row_without_dense_owner() -> None:
    cache = DeepseekV4DSparkCache(window_size=8, head_dim=512)
    cache.prefill(
        mx.zeros((1, 8, 512), dtype=mx.bfloat16),
        mx.zeros((1, 8, 64), dtype=mx.bfloat16),
    )
    replacement_latent = ((mx.arange(512, dtype=mx.float32) % 37) / 11).reshape(
        1, 1, 512
    ).astype(mx.bfloat16)
    replacement_rope = ((mx.arange(64, dtype=mx.float32) - 11) / 9).reshape(
        1, 1, 64
    ).astype(mx.bfloat16)

    cache.commit_main(
        start_pos=2,
        main_latent=replacement_latent,
        main_rope=replacement_rope,
    )

    visible_key, visible_value = cache.visible_rows()
    expected_key, expected_value = cache.ring.decode(2, 3)
    np.testing.assert_array_equal(
        np.array(visible_value[:, 2:3].astype(mx.float32)),
        np.array(expected_value.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(visible_key[:, 2:3].astype(mx.float32)),
        np.array(expected_key.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(visible_key[:, 2:3, 448:].astype(mx.float32)),
        np.array(replacement_rope.astype(mx.float32)),
    )
    assert not hasattr(cache, "dense_ring")


def test_dspark_decode_uses_the_committed_main_row_then_five_future_positions() -> None:
    np.testing.assert_array_equal(
        np.array(_dspark_visibility_indices(128, 5, 17)),
        np.concatenate([np.arange(18), 128 + np.arange(5)]),
    )
    np.testing.assert_array_equal(
        np.array(_dspark_draft_positions(17, 5)),
        np.arange(18, 23),
    )


class _Layer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id

    def __call__(self, hidden, mask=None, cache=None, input_ids=None):
        del mask, cache, input_ids
        return mx.full(hidden.shape, self.layer_id, dtype=hidden.dtype)


class _Embedding:
    def __call__(self, input_ids):
        return mx.zeros((*input_ids.shape, 2), dtype=mx.float32)


def test_target_route_returns_ordered_40_41_42_taps() -> None:
    owner = SimpleNamespace(
        args=SimpleNamespace(hc_mult=2),
        model=SimpleNamespace(
            embed_tokens=_Embedding(),
            layers=[_Layer(layer_id) for layer_id in range(43)],
        ),
    )
    route = DSparkTargetRoute((40, 41, 42))

    final_hidden, taps = route(owner, mx.array([[7]], dtype=mx.int32), cache=None)

    assert tuple(final_hidden.shape) == (1, 1, 2, 2)
    assert len(taps) == 3
    assert all(tuple(tap.shape) == (1, 1, 2) for tap in taps)
    assert tuple(float(tap[0, 0, 0].item()) for tap in taps) == (40.0, 41.0, 42.0)


def test_dspark_owner_constructs_three_stages_and_primary_plus_four_noise_inputs(
    monkeypatch,
) -> None:
    class _FakeStage:
        def __init__(self, args, stage_id):
            del args
            self.stage_id = stage_id
            self.attn = SimpleNamespace(window_size=128, head_dim=512)

    monkeypatch.setattr(dspark_module, "DeepseekV4DSparkStage", _FakeStage)
    args = SimpleNamespace(
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_markov_rank=256,
        num_hidden_layers=43,
        num_nextn_predict_layers=1,
        vocab_size=129280,
        compress_ratios=[0] * 46,
    )

    owner = build_deepseek_v4_dspark(args)
    draft_inputs = owner.draft_input_ids(mx.array([29], dtype=mx.int32))
    caches = owner.make_cache()

    assert tuple(stage.stage_id for stage in owner.stages) == (0, 1, 2)
    assert tuple(np.array(draft_inputs)[0]) == (29, 128799, 128799, 128799, 128799)
    assert len(caches) == 3
    assert len({id(cache) for cache in caches}) == 3
    assert all(cache.ring.mode == "nvfp4_stock432" for cache in caches)


def test_dspark_signature_installs_tap_route_and_mia_nvfp4_target_cache(monkeypatch) -> None:
    class _FakeAttention:
        window_size = 128
        compress_ratio = 0
        head_dim = 512

        def __init__(self) -> None:
            self.mia_nvfp4_installed = False

        def install_mia_nvfp4_attention(self) -> None:
            self.mia_nvfp4_installed = True

    class _FakeTarget(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embed_tokens = _Embedding()
            self.layers = [_Layer(layer_id) for layer_id in range(args.num_hidden_layers)]
            for layer in self.layers:
                layer.attn = _FakeAttention()

        def collapse(self, hidden):
            return mx.mean(hidden, axis=2)

        def hc_hidden(self, input_ids, cache=None):
            del input_ids, cache
            raise AssertionError("the DSpark target route must replace hc_hidden")

    fake_stages = [SimpleNamespace(stage_id=stage_id) for stage_id in range(3)]
    fake_owner = SimpleNamespace(stages=fake_stages)
    monkeypatch.setattr(target_module, "DeepseekV4Model", _FakeTarget)
    monkeypatch.setattr(dspark_module, "build_deepseek_v4_dspark", lambda args: fake_owner)

    args = ModelArgs(
        vocab_size=129280,
        hidden_size=2,
        num_hidden_layers=43,
        hc_mult=2,
        compress_ratios=[0] * 46,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_markov_rank=256,
        num_nextn_predict_layers=1,
    )
    model = Model(args)

    logits, taps = model(
        mx.array([[7]], dtype=mx.int32),
        return_hidden=True,
        emit_logits=False,
    )
    caches = model.make_cache()

    assert logits is None
    assert model.dspark is fake_owner
    assert model.mtp == fake_stages
    assert model.has_mtp is False
    assert tuple(float(tap[0, 0, 0].item()) for tap in taps) == (40.0, 41.0, 42.0)
    assert len(caches) == 43
    assert all(isinstance(cache, DeepseekV4NVFP4Cache) for cache in caches)
    assert all(layer.attn.mia_nvfp4_installed for layer in model.model.layers)
    assert is_deepseek_v4_mtp_config(
        {
            "model_type": "deepseek_v4",
            "num_nextn_predict_layers": 1,
            "dspark_block_size": 5,
            "dspark_markov_rank": 256,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40, 41, 42],
        }
    ) is False


def test_sanitize_flattens_real_dspark_grouped_o_lora_storage() -> None:
    model = SimpleNamespace(dspark=SimpleNamespace())
    grouped = mx.zeros((8, 1024, 768), dtype=mx.uint32)
    grouped_scales = mx.zeros((8, 1024, 32), dtype=mx.bfloat16)
    weights = {
        "model.layers.0.attn.wo_a.weight": grouped,
        "model.layers.0.attn.wo_a.scales": grouped_scales,
        "model.layers.0.attn.wo_a.biases": grouped_scales,
        "mtp.0.attn.wo_a.weight": grouped,
    }

    sanitized = Model.sanitize(model, weights)

    assert sanitized["model.layers.0.attn.wo_a.weight"].shape == (8192, 768)
    assert sanitized["model.layers.0.attn.wo_a.scales"].shape == (8192, 32)
    assert sanitized["model.layers.0.attn.wo_a.biases"].shape == (8192, 32)
    assert sanitized["mtp.0.attn.wo_a.weight"].shape == (8192, 768)
