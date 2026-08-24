from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mtplx import qwen38_challenge_kernels as kernels
from mtplx.gdn_capture import (
    configure_qwen38_dflash_row48_boundary,
    qwen38_row48_boundary_counter_snapshot,
)
from mtplx.qwen38_challenge_kernels import (
    configure_qwen38_dflash_row24_eval_ladder,
    configure_qwen38_row21_qk_rms_rope,
    configure_qwen38_row24_qk_length_limit,
    qwen38_dual_rms_norm_concat,
    qwen38_qk_rms_rope,
)


def test_dual_rms_norm_concat_matches_two_stock_norms() -> None:
    a = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    b = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    a_weight = mx.random.normal((5120,)).astype(mx.bfloat16)
    b_weight = mx.random.normal((5120,)).astype(mx.bfloat16)
    expected = mx.concatenate(
        (
            mx.fast.rms_norm(a, a_weight, 1e-6),
            mx.fast.rms_norm(b, b_weight, 1e-6),
        ),
        axis=-1,
    )
    actual = qwen38_dual_rms_norm_concat(a, b, a_weight, b_weight, 1e-6)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


def test_qk_rms_rope_matches_stock_qwen38_partial_rope_at_fixed_d3_width() -> None:
    queries = mx.random.normal((1, 4, 24, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 4, 4, 256)).astype(mx.bfloat16)
    q_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    k_weight = mx.random.normal((256,)).astype(mx.bfloat16)
    q_norm = mx.fast.rms_norm(queries, q_weight, 1e-6).transpose(0, 2, 1, 3)
    k_norm = mx.fast.rms_norm(keys, k_weight, 1e-6).transpose(0, 2, 1, 3)
    q_expected = mx.fast.rope(
        q_norm,
        64,
        traditional=False,
        base=10_000_000.0,
        scale=1.0,
        offset=37,
    )
    k_expected = mx.fast.rope(
        k_norm,
        64,
        traditional=False,
        base=10_000_000.0,
        scale=1.0,
        offset=37,
    )

    q_actual, k_actual = qwen38_qk_rms_rope(
        queries,
        keys,
        q_weight,
        k_weight,
        1e-6,
        37,
    )
    mx.eval(q_expected, k_expected, q_actual, k_actual)

    assert mx.array_equal(q_actual, q_expected).item()
    assert mx.array_equal(k_actual, k_expected).item()


def test_row21_config_exposes_dflash_qk_prepare_callback(monkeypatch) -> None:
    rope = SimpleNamespace(
        dims=64,
        base=10_000_000.0,
        scale=1.0,
        traditional=False,
    )
    q_norm = SimpleNamespace(weight=object(), eps=1e-6)
    k_norm = SimpleNamespace(weight=object(), eps=1e-6)
    attention = SimpleNamespace(
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        rope=rope,
        q_norm=q_norm,
        k_norm=k_norm,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    )
    calls = []

    def fake_kernel(queries, keys, q_weight, k_weight, eps, offset):
        calls.append((queries, keys, q_weight, k_weight, eps, offset))
        return "prepared-q", "prepared-k"

    monkeypatch.setattr(kernels, "qwen38_qk_rms_rope", fake_kernel)

    report = configure_qwen38_row21_qk_rms_rope(model, active=True)
    assert report["dflash_modules"] == 1
    assert attention._dflash_qk_prepare("q", "k", 4096) == (
        "prepared-q",
        "prepared-k",
    )
    assert calls == [("q", "k", q_norm.weight, k_norm.weight, 1e-6, 4096)]

    inactive = configure_qwen38_row21_qk_rms_rope(model, active=False)
    assert inactive["dflash_modules"] == 0
    assert not hasattr(attention, "_dflash_qk_prepare")


def test_row24_config_exposes_dflash_qk_length_fallback() -> None:
    rope = SimpleNamespace(
        dims=64,
        base=10_000_000.0,
        scale=1.0,
        traditional=False,
    )
    attention = SimpleNamespace(
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        rope=rope,
        q_norm=SimpleNamespace(weight=object(), eps=1e-6),
        k_norm=SimpleNamespace(weight=object(), eps=1e-6),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    )
    before = kernels.qwen38_row24_qk_length_fallback_counter_snapshot()

    report = configure_qwen38_row24_qk_length_limit(
        model,
        active=True,
        max_length=16,
    )
    assert report["dflash_modules"] == 1
    assert attention._dflash_qk_max_length == 16
    attention._dflash_qk_fallback()
    assert kernels.qwen38_row24_qk_length_fallback_counter_snapshot() == before + 1

    inactive = configure_qwen38_row24_qk_length_limit(model, active=False)
    assert inactive["dflash_modules"] == 0
    assert not hasattr(attention, "_dflash_qk_max_length")
    assert not hasattr(attention, "_dflash_qk_fallback")


def test_row24_dflash_eval_ladder_uses_prefill_and_decode_rungs(monkeypatch) -> None:
    inner = SimpleNamespace(layers=[])
    model = SimpleNamespace(model=inner)
    calls = []
    monkeypatch.setattr(
        kernels,
        "qwen38_row24_async_eval",
        lambda value, *, row26=False: calls.append((tuple(value.shape), row26)),
    )

    report = configure_qwen38_dflash_row24_eval_ladder(
        model,
        active=True,
        prefill_stride=4,
    )
    assert report == {"active": 1, "prefill_stride": 4}
    decode = mx.zeros((1, 8, 16))
    for layer_index in range(3):
        inner._dflash_post_layer(decode, layer_index)
    prefill = mx.zeros((1, 512, 16))
    for layer_index in range(5):
        inner._dflash_post_layer(prefill, layer_index)

    assert calls == [
        ((1, 8, 16), False),
        ((1, 8, 16), False),
        ((1, 512, 16), False),
        ((1, 512, 16), False),
    ]

    inactive = configure_qwen38_dflash_row24_eval_ladder(
        model,
        active=False,
        prefill_stride=4,
    )
    assert inactive == {"active": 0, "prefill_stride": 0}
    assert not hasattr(inner, "_dflash_post_layer")


def test_row48_config_exposes_dflash_cross_layer_fusion(monkeypatch) -> None:
    layer = SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=object(), eps=1e-6),
        post_attention_layernorm=SimpleNamespace(weight=object(), eps=1e-6),
        mlp=object(),
        self_attn=object(),
    )
    inner = SimpleNamespace(layers=[layer, layer])
    model = SimpleNamespace(model=inner)
    calls = []

    from mtplx.kernels import fused_norm

    monkeypatch.setattr(
        fused_norm,
        "fused_add_rmsnorm",
        lambda base, delta, weight, eps, *, threadgroup_size: calls.append(
            (base, delta, weight, eps, threadgroup_size)
        )
        or ("hidden", "normed"),
    )
    before = qwen38_row48_boundary_counter_snapshot()

    report = configure_qwen38_dflash_row48_boundary(model, active=True)
    assert report == {"eligible_modules": 2, "active_modules": 2}
    inner._dflash_boundary_begin()
    assert inner._dflash_fused_add_rmsnorm(
        "base",
        "delta",
        "weight",
        1e-6,
        merged_boundary=True,
    ) == ("hidden", "normed")
    after = qwen38_row48_boundary_counter_snapshot()

    assert after["calls"] == before["calls"] + 1
    assert after["merged_boundaries"] == before["merged_boundaries"] + 1
    assert calls == [("base", "delta", "weight", 1e-6, 1024)]

    inactive = configure_qwen38_dflash_row48_boundary(model, active=False)
    assert inactive == {"eligible_modules": 2, "active_modules": 0}
    assert not hasattr(inner, "_dflash_boundary_begin")
    assert not hasattr(inner, "_dflash_fused_add_rmsnorm")
