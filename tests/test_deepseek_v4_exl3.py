from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

import mlx.core as mx

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
    sinkhorn_owners = tuple(model.layers) + tuple(model.dspark.stages)
    assert len(sinkhorn_owners) == 46
    assert all(
        layer.attn_hc._sinkhorn_kernel and layer.ffn_hc._sinkhorn_kernel
        for layer in sinkhorn_owners
    )
    assert all(
        layer.attn.o_lora_mode == "gather_qmm" for layer in sinkhorn_owners
    )
    assert model.mtp[0].ffn.switch_mlp.gate_proj.mode == "mxfp4"
    assert model.mtp[0].main_proj.mode == "mxfp8"
