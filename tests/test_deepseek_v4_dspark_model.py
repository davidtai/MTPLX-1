from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mtplx.deepseek_v4_affine_kv import AffineInt4Rows  # noqa: E402
from mtplx.models import deepseek_v4 as target_module  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4AffineInt4Cache,
    Model,
    ModelArgs,
    is_deepseek_v4_mtp_config,
)
import mtplx.models.deepseek_v4_dspark as dspark_module  # noqa: E402
from mtplx.models.deepseek_v4_dspark import (  # noqa: E402
    DSparkTargetRoute,
    DeepseekV4DSparkCache,
    build_deepseek_v4_dspark,
    greedy_future_tokens,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


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


def test_each_dspark_stage_owns_distinct_affine_int4_cache() -> None:
    caches = [DeepseekV4DSparkCache(window_size=8, head_dim=512) for _ in range(3)]
    assert len({id(cache) for cache in caches}) == 3
    assert len({id(cache.ring) for cache in caches}) == 3
    assert all(isinstance(cache.ring, AffineInt4Rows) for cache in caches)
    assert all(cache.ring.bits == 4 for cache in caches)
    assert all(cache.ring.group_size == 64 for cache in caches)

    prompt_kv = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    caches[0].prefill(prompt_kv)
    assert len(caches[0].ring) == 8
    assert len(caches[1].ring) == 0
    assert len(caches[2].ring) == 0


def test_dspark_cache_commits_authoritative_main_row_without_dense_owner() -> None:
    cache = DeepseekV4DSparkCache(window_size=8, head_dim=512)
    cache.prefill(mx.zeros((1, 8, 512), dtype=mx.bfloat16))
    replacement = ((mx.arange(512, dtype=mx.float32) % 37) / 11).reshape(
        1, 1, 512
    ).astype(mx.bfloat16)

    cache.commit_main(start_pos=2, main_kv=replacement)

    direct = mx.dequantize(
        *mx.quantize(replacement, group_size=64, bits=4, mode="affine"),
        group_size=64,
        bits=4,
        mode="affine",
    )
    visible = cache.visible_rows()
    np.testing.assert_array_equal(
        np.array(visible[:, 2:3].astype(mx.float32)),
        np.array(direct.astype(mx.float32)),
    )
    assert not hasattr(cache, "dense_ring")


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
    assert all(cache.ring.bits == 4 for cache in caches)


def test_dspark_signature_installs_tap_route_and_affine_target_cache(monkeypatch) -> None:
    class _FakeTarget(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embed_tokens = _Embedding()
            self.layers = [_Layer(layer_id) for layer_id in range(args.num_hidden_layers)]
            for layer in self.layers:
                layer.attn = SimpleNamespace(
                    window_size=128,
                    compress_ratio=0,
                    head_dim=512,
                )

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
    assert all(isinstance(cache, DeepseekV4AffineInt4Cache) for cache in caches)
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
