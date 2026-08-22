from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors import safe_open

import mlx.core as mx

import mtplx.deepseek_v4_exl3 as exl3
from mtplx.deepseek_v4_exl3 import (
    EXL3SwitchGLU,
    decode_mcg_trellis_tile,
    exl3_mcg_grouped_mma,
    exl3_mcg_grouped_qmv,
    exl3_mcg_qmv,
    load_indexed_safetensors,
    load_mia_exl3_dspark_model,
    sanitize_mia_dspark_weights,
)
from mtplx.models.deepseek_v4 import DeepseekV4MoE, ModelArgs


def test_mia_loader_installs_stacked_projections_after_weights_before_plan() -> None:
    source = inspect.getsource(load_mia_exl3_dspark_model)

    draft_load = source.index(
        "model.load_weights(list(draft_weights.items()), strict=False)"
    )
    wo_install = source.index("install_mia_tp1_wo_projection_routes(")
    stacked_install = source.index("install_mia_stacked_projections(model)")
    qkv_install = source.index("install_mia_qkv_prologue_routes(model)")
    plan_build = source.index("engine_plan = build_mia_engine_plan(")

    assert draft_load < wo_install < stacked_install < qkv_install < plan_build


_MIA_EXACT_MODEL = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1"
)
_LAYER0 = _MIA_EXACT_MODEL / "exl3-layer-000-tp1-rank0.safetensors"
_MIA_K64_DRAFT = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-dspark-k64"
)
_W1_TRELLIS = "layers.0.ffn.experts.0.w1.rank0.trellis"
_W1_SUH = "layers.0.ffn.experts.0.w1.rank0.suh"
_W1_SVH = "layers.0.ffn.experts.0.w1.rank0.svh"


class _StaticArray:
    def __init__(self, shape, dtype=mx.float16):
        self.shape = tuple(shape)
        self.size = int(np.prod(self.shape))
        self.dtype = dtype

    def reshape(self, *_shape):
        return self

    def astype(self, _dtype):
        return self


_MIA_MHC_ROUTE_CONTRACT = (
    "broadcast_fn_fp32",
    "tiny_split32_fp32",
    "prefill_post_pre_bf16_mma_bm64_fp32",
    "compact_gram_finalize",
)


def _callable_name(value):
    return str(getattr(value, "__name__", type(value).__name__))


def _assert_mia_carried_mhc_contract(model) -> None:
    """Assert the installed carried route, not the retired layer-local route."""

    target_connections = tuple(
        connection
        for layer in model.model.layers
        for connection in (layer.attn_hc, layer.ffn_hc)
    )
    draft_connections = tuple(
        connection
        for stage in model.dspark.stages
        for connection in (stage.attn_hc, stage.ffn_hc)
    )
    assert len(target_connections) == 43 * 2
    assert len(draft_connections) == 3 * 2

    target_mhc = model.model._mia_mhc
    draft_mhc = model.dspark._mia_mhc
    assert target_mhc is not draft_mhc
    assert target_mhc.bound_hyper_connections == len(target_connections)
    assert draft_mhc.bound_hyper_connections == len(draft_connections)
    assert target_mhc.route_contract == _MIA_MHC_ROUTE_CONTRACT
    assert draft_mhc.route_contract == _MIA_MHC_ROUTE_CONTRACT
    assert all(
        actual is expected
        for actual, expected in zip(
            target_mhc._hyper_connections,
            target_connections,
            strict=True,
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            draft_mhc._hyper_connections,
            draft_connections,
            strict=True,
        )
    )

    for connections in (target_connections, draft_connections):
        bindings = tuple(connection._mia_mhc_weight for connection in connections)
        assert len({id(binding) for binding in bindings}) == len(bindings)
        assert all(binding.fn_bf16.dtype == mx.bfloat16 for binding in bindings)
        assert bindings[0].fn_broadcast.shape == (24, 4096)
        assert bindings[0].fn_broadcast.dtype == mx.float32
        assert all(binding.fn_broadcast is None for binding in bindings[1:])

    installed_hot_routes = (
        model.model._hc_hidden_impl,
        model.model._collapse_impl,
        model._target_forward_route,
        model.dspark._propose_impl,
    )
    assert tuple(map(_callable_name, installed_hot_routes)) == (
        "_mia_hc_hidden",
        "_mia_collapse",
        "_mia_target_forward",
        "_mia_propose_k5",
    )
    generic_sinkhorn_callables = {
        id(connection._sinkhorn_normalise)
        for connection in target_connections + draft_connections
    }
    assert all(id(route) not in generic_sinkhorn_callables for route in installed_hot_routes)


def _named_route(name):
    def route(*_args, **_kwargs):
        raise AssertionError("route execution is outside this construction test")

    route.__name__ = name
    return route


def test_carried_mhc_contract_owns_43_target_and_3_draft_layers():
    def connection():
        return SimpleNamespace(
            _sinkhorn_normalise=_named_route("stock"),
            _mia_mhc_weight=SimpleNamespace(
                fn_bf16=SimpleNamespace(dtype=mx.bfloat16),
                fn_broadcast=None,
            ),
        )

    target_layers = tuple(
        SimpleNamespace(attn_hc=connection(), ffn_hc=connection())
        for _ in range(43)
    )
    draft_stages = tuple(
        SimpleNamespace(attn_hc=connection(), ffn_hc=connection())
        for _ in range(3)
    )
    target_connections = tuple(
        owner
        for layer in target_layers
        for owner in (layer.attn_hc, layer.ffn_hc)
    )
    draft_connections = tuple(
        owner
        for stage in draft_stages
        for owner in (stage.attn_hc, stage.ffn_hc)
    )
    target_connections[0]._mia_mhc_weight.fn_broadcast = SimpleNamespace(
        shape=(24, 4096), dtype=mx.float32
    )
    draft_connections[0]._mia_mhc_weight.fn_broadcast = SimpleNamespace(
        shape=(24, 4096), dtype=mx.float32
    )
    target_mhc = SimpleNamespace(
        bound_hyper_connections=86,
        route_contract=_MIA_MHC_ROUTE_CONTRACT,
        _hyper_connections=target_connections,
    )
    draft_mhc = SimpleNamespace(
        bound_hyper_connections=6,
        route_contract=_MIA_MHC_ROUTE_CONTRACT,
        _hyper_connections=draft_connections,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=target_layers,
            _mia_mhc=target_mhc,
            _hc_hidden_impl=_named_route("_mia_hc_hidden"),
            _collapse_impl=_named_route("_mia_collapse"),
        ),
        dspark=SimpleNamespace(
            stages=draft_stages,
            _mia_mhc=draft_mhc,
            _propose_impl=_named_route("_mia_propose_k5"),
        ),
        _target_forward_route=_named_route("_mia_target_forward"),
    )

    _assert_mia_carried_mhc_contract(model)


def test_trellis_bm64_descriptors_and_launch_use_populated_block_bound(
    monkeypatch,
):
    """M1024/top-k6/K216 can populate at most 308 BM64 route blocks."""

    calls = {}

    def route_kernel(**kwargs):
        calls["route"] = kwargs
        return tuple(_StaticArray(shape) for shape in kwargs["output_shapes"])

    def mma_kernel(**kwargs):
        calls["mma"] = kwargs
        return (object(),)

    monkeypatch.setattr(exl3.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(
        exl3,
        "_trellis_route_pack_kernel",
        lambda _experts, _topk, _block_m: route_kernel,
    )
    monkeypatch.setattr(
        exl3,
        "_mcg_trellis_mma_kernel",
        lambda _size_k, _size_n, _experts, _block_m: mma_kernel,
    )

    tasks = 1024 * 6
    routes = exl3._pack_trellis_routes(
        _StaticArray((1024, 6)),
        experts=216,
        topk=6,
        block_m=64,
        kernel=route_kernel,
    )
    owner = SimpleNamespace(experts=216)
    bank = SimpleNamespace(
        input_dims=4096,
        output_dims=2048,
        trellis=object(),
    )
    exl3.EXL3SwitchGLU._trellis_mma(
        owner,
        bank,
        _StaticArray((tasks, 4096)),
        routes[3:],
        block_m=64,
        kernel=mma_kernel,
    )

    assert calls["route"]["output_shapes"] == [
        (tasks,),
        (tasks,),
        (tasks,),
        (308,),
        (308,),
        (308,),
        (1,),
    ]
    assert calls["mma"]["grid"] == (512, 64, 308)


def test_trellis_uses_measured_bm8_route_through_m127() -> None:
    source = inspect.getsource(EXL3SwitchGLU.fused)

    assert "self._trellis_plans[0 if rows <= 127 else 1]" in source


def test_trellis_swiglu_limit_is_a_valid_metal_float_literal(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(exl3.mx.fast, "metal_kernel", capture)
    exl3._trellis_activation_down_hadamard_kernel.__wrapped__(2048, 216, 10.0)

    assert "constant constexpr float LIMIT = 10.0f;" in captured["header"]


def test_installed_trellis_runtime_never_reenters_kernel_factories(monkeypatch):
    """BM8/BM64 execution must use only construction-bound Metal kernels."""

    def kernel(**kwargs):
        return tuple(
            _StaticArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        )

    monkeypatch.setattr(exl3, "_mma_route_pack_kernel", lambda _experts: kernel)
    monkeypatch.setattr(exl3.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(exl3, "_mcg_qmv_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_route_hadamard_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_mcg_grouped_mma_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_route_output_hadamard_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_trellis_route_pack_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_packed_route_hadamard_kernel", lambda *_args: kernel)
    monkeypatch.setattr(exl3, "_mcg_trellis_mma_kernel", lambda *_args: kernel)
    monkeypatch.setattr(
        exl3, "_trellis_activation_down_hadamard_kernel", lambda *_args: kernel
    )
    monkeypatch.setattr(exl3, "_trellis_final_reduce_kernel", lambda *_args: kernel)

    owner = EXL3SwitchGLU(128, 128, 2, 1, limit=0.0)
    owner.install_trellis_runtime(max_tokens=64)

    def forbidden_factory(*_args, **_kwargs):
        raise AssertionError("installed Trellis execution re-entered a kernel factory")

    monkeypatch.setattr(exl3, "_trellis_route_pack_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_packed_route_hadamard_kernel", forbidden_factory)
    monkeypatch.setattr(exl3, "_mcg_trellis_mma_kernel", forbidden_factory)
    monkeypatch.setattr(
        exl3, "_trellis_activation_down_hadamard_kernel", forbidden_factory
    )
    monkeypatch.setattr(exl3, "_trellis_final_reduce_kernel", forbidden_factory)

    for rows in (1, 33):
        owner.fused(
            mx.zeros((rows, 128), dtype=mx.float16),
            mx.zeros((rows, 1), dtype=mx.int32),
            mx.ones((rows, 1), dtype=mx.float32),
            mx.zeros((rows, 128), dtype=mx.float16),
        )


def _authentic_layer0_w1_tile() -> np.ndarray:
    if not _LAYER0.is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")
    with safe_open(_LAYER0, framework="np") as handle:
        return handle.get_tensor(_W1_TRELLIS)[0, 0]


def test_authentic_mia_mcg_tile_decodes_in_source_layout():
    """Gate the exact MCG decode, bit windows, and tensor-core permutation."""

    tile = decode_mcg_trellis_tile(_authentic_layer0_w1_tile())

    assert tile.shape == (16, 16)
    assert tile.dtype == np.float16
    assert sha256(tile.tobytes()).hexdigest() == (
        "9c5d060bb4bb9caca2d16886d0c2c1192755571d651668e9332225a0b808e954"
    )
    np.testing.assert_array_equal(
        tile[0],
        np.array(
            [
                1.169921875,
                2.23046875,
                -0.97265625,
                -0.3203125,
                0.533203125,
                0.54052734375,
                -0.8876953125,
                1.32421875,
                0.100341796875,
                0.420166015625,
                1.388671875,
                1.45703125,
                0.64013671875,
                -0.78564453125,
                -1.0830078125,
                -0.9833984375,
            ],
            dtype=np.float16,
        ),
    )


def _hadamard128(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    stride = 1
    while stride < 128:
        for start in range(0, 128, stride * 2):
            left = out[start : start + stride].copy()
            right = out[start + stride : start + 2 * stride].copy()
            out[start : start + stride] = left + right
            out[start + stride : start + 2 * stride] = left - right
        stride *= 2
    return out * np.float32(1.0 / np.sqrt(128.0))


def _authentic_w1_block():
    if not _LAYER0.is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")
    with safe_open(_LAYER0, framework="np") as handle:
        return (
            handle.get_tensor(_W1_TRELLIS)[:8, :8].copy(),
            handle.get_tensor(_W1_SUH)[:128].copy(),
            handle.get_tensor(_W1_SVH)[:128].copy(),
        )


def test_authentic_mia_projection_fuses_h128_signs_and_mcg_qmv():
    """The Metal operator must reproduce the pinned source projection order."""

    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    trellis, suh, svh = _authentic_w1_block()
    inner = np.empty((128, 128), dtype=np.float16)
    for tile_k in range(8):
        for tile_n in range(8):
            inner[
                tile_k * 16 : (tile_k + 1) * 16,
                tile_n * 16 : (tile_n + 1) * 16,
            ] = decode_mcg_trellis_tile(trellis[tile_k, tile_n])

    x = np.linspace(-1.0, 1.0, 128, dtype=np.float16)
    x_had = _hadamard128((x * suh).astype(np.float16)).astype(np.float16)
    projected = (x_had.astype(np.float32) @ inner.astype(np.float32)).astype(
        np.float16
    )
    expected = (
        _hadamard128(projected).astype(np.float16) * svh.astype(np.float16)
    ).astype(np.float16)

    actual = exl3_mcg_qmv(
        mx.array(x)[None],
        mx.array(trellis),
        mx.array(suh),
        mx.array(svh),
    )
    mx.eval(actual)

    assert tuple(actual.shape) == (1, 128)
    assert actual.dtype == mx.float16
    np.testing.assert_allclose(np.array(actual)[0], expected, rtol=2e-2, atol=2e-2)

    grouped = exl3_mcg_grouped_qmv(
        mx.array(x)[None],
        mx.array(trellis)[None],
        mx.array(suh)[None],
        mx.array(svh)[None],
        mx.array([[0]], dtype=mx.int32),
    )
    mx.eval(grouped)
    assert tuple(grouped.shape) == (1, 1, 128)
    np.testing.assert_array_equal(np.array(grouped)[0, 0], np.array(actual)[0])


def test_authentic_mia_grouped_mma_matches_exl3_projection():
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    trellis, suh, svh = _authentic_w1_block()
    rows = np.stack(
        [np.linspace(-1.0 + i / 32, 1.0 + i / 32, 128, dtype=np.float16)
         for i in range(9)]
    )
    ids = mx.zeros((9, 1), dtype=mx.int32)
    expected = exl3_mcg_grouped_qmv(
        mx.array(rows),
        mx.array(trellis)[None],
        mx.array(suh)[None],
        mx.array(svh)[None],
        ids,
    )
    actual = exl3_mcg_grouped_mma(
        mx.array(rows),
        mx.array(trellis)[None],
        mx.array(suh)[None],
        mx.array(svh)[None],
        ids,
    )
    mx.eval(expected, actual)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-2, atol=2e-2)


def test_exact_mia_config_installs_exl3_only_on_target_layers():
    if not (_MIA_EXACT_MODEL / "config.json").is_file():
        pytest.skip("exact MiaAI TP1 artifact is not installed")
    import json

    args = ModelArgs.from_dict(
        json.loads((_MIA_EXACT_MODEL / "config.json").read_text())
    )
    target = DeepseekV4MoE(args, 0)
    draft = DeepseekV4MoE(args, args.num_hidden_layers)

    assert isinstance(target.switch_mlp, EXL3SwitchGLU)
    assert tuple(target.switch_mlp.gate_proj.trellis.shape) == (216, 256, 128, 48)
    assert not isinstance(draft.switch_mlp, EXL3SwitchGLU)


def test_exact_mia_k64_draft_maps_native_fp4_and_fp8_storage():
    if not (_MIA_K64_DRAFT / "model.safetensors.index.json").is_file():
        pytest.skip("exact MiaAI K64 draft artifact is not installed")

    source = load_indexed_safetensors(_MIA_K64_DRAFT)
    mapped = sanitize_mia_dspark_weights(source, stages=3, experts=64)

    assert len(source) == 1249
    assert mapped["mtp.0.ffn.switch_mlp.gate_proj.weight"].shape == (
        64,
        2048,
        512,
    )
    assert mapped["mtp.0.ffn.switch_mlp.gate_proj.weight"].dtype == mx.uint32
    assert mapped["mtp.0.ffn.switch_mlp.gate_proj.scales"].shape == (
        64,
        2048,
        128,
    )
    assert mapped["mtp.0.main_proj.weight"].shape == (4096, 3072)
    assert mapped["mtp.0.main_proj.weight"].dtype == mx.uint32
    assert mapped["mtp.0.main_proj.scales"].shape == (4096, 384)
    assert "mtp.2.markov_head.markov_w1.weight" in mapped
    assert not any(".experts." in name for name in mapped)


def test_streaming_carried_split_fp8_pairs_keep_quantized_parameter_names():
    geometries = {
        "model.layers.20.ffn.shared_experts.down_proj": (128, 128),
        "model.layers.8.attn.wo_a": (128, 128),
    }
    scale_shard = {
        "layers.20.ffn.shared_experts.w2.scale": mx.zeros(
            (1, 1), dtype=mx.uint8
        ),
        "layers.8.attn.wo_a.scale": mx.zeros((1, 1), dtype=mx.uint8),
    }
    weight_shard = {
        "layers.20.ffn.shared_experts.w2.weight": mx.zeros(
            (128, 128), dtype=mx.uint8
        ),
        "layers.8.attn.wo_a.weight": mx.zeros(
            (128, 128), dtype=mx.uint8
        ),
    }

    mapped = {
        **exl3._map_mia_target_carried_shard(
            scale_shard,
            fp8_geometries=geometries,
        ),
        **exl3._map_mia_target_carried_shard(
            weight_shard,
            fp8_geometries=geometries,
        ),
    }

    assert set(mapped) == {
        "model.layers.20.ffn.shared_experts.down_proj.weight",
        "model.layers.20.ffn.shared_experts.down_proj.scales",
        "model.layers.8.attn.wo_a.weight",
        "model.layers.8.attn.wo_a.scales",
    }


def test_exact_mia_split_artifact_constructs_k216_target_and_k64_owner():
    if not (_MIA_K64_DRAFT / "model.safetensors.index.json").is_file():
        pytest.skip("exact MiaAI K64 draft artifact is not installed")

    model = load_mia_exl3_dspark_model(
        _MIA_EXACT_MODEL,
        draft_root=_MIA_K64_DRAFT,
        lazy=True,
    )

    assert model.args.n_routed_experts == 216
    assert model.args.dspark_block_size == 5
    assert model.args.dspark_target_layer_ids == [40, 41, 42]
    assert model._target_cache_type.__name__ == "DeepseekV4NVFP4Cache"
    assert model.dspark.args.n_routed_experts == 64
    assert len(model.dspark.stages) == 3
    _assert_mia_carried_mhc_contract(model)
    attention_owners = tuple(model.layers) + tuple(model.dspark.stages)
    assert len(attention_owners) == 46
    wo_plans = tuple(
        layer.attn._output_projection_impl for layer in attention_owners
    )
    assert len({id(plan) for plan in wo_plans}) == 46
    assert all(type(plan).__name__ == "MiaTP1WOMXFP8Plan" for plan in wo_plans)
    assert model._mia_wo_projection_receipt["plan_ids"] == tuple(
        id(plan) for plan in wo_plans
    )
    assert model.mtp[0].ffn.switch_mlp.gate_proj.mode == "mxfp4"
    assert model.mtp[0].main_proj.mode == "mxfp8"
