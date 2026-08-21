import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_affine_kv import AffineInt4Rows  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _rows() -> mx.array:
    values = (mx.arange(1024, dtype=mx.float32) % 97 - 48) / 13
    return values.reshape(1, 2, 512).astype(mx.bfloat16)


def _load_target_module():
    source = Path(__file__).parents[1] / "mtplx" / "models" / "deepseek_v4.py"
    spec = importlib.util.spec_from_file_location("dsv4_affine_kv_undertest", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_affine_int4_rows_store_only_packed_values_scales_and_biases() -> None:
    rows = _rows()
    store = AffineInt4Rows(width=512)

    store.append(rows[:, :1])
    store.append(rows[:, 1:])

    assert store.mode == "affine"
    assert store.bits == 4
    assert store.group_size == 64
    assert store.shape == (1, 2, 512)
    assert store.packed.dtype == mx.uint32
    assert store.scales.shape == (1, 2, 8)
    assert store.biases.shape == (1, 2, 8)
    assert not hasattr(store, "dense")
    assert not hasattr(store, "rows")

    direct = mx.quantize(rows, group_size=64, bits=4, mode="affine")
    expected = mx.dequantize(
        *direct,
        group_size=64,
        bits=4,
        mode="affine",
    )
    np.testing.assert_array_equal(
        np.array(store.dequantize().astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(store.dequantize(stop=1).astype(mx.float32)),
        np.array(expected[:, :1].astype(mx.float32)),
    )


def test_target_affine_int4_cache_owns_window_and_compressed_kv_from_zero() -> None:
    target = _load_target_module()
    cache = target.DeepseekV4AffineInt4Cache(
        window_size=4,
        compress_ratio=4,
        head_dim=512,
    )
    rows = _rows()

    visible, start = cache.update_window(rows)
    cache.compressed.append(rows[:, :1])

    assert start == 0
    assert visible.shape == rows.shape
    assert cache.offset == 0
    assert isinstance(cache.window, AffineInt4Rows)
    assert isinstance(cache.compressed, AffineInt4Rows)
    assert cache.window.bits == cache.compressed.bits == 4
    assert cache.window.group_size == cache.compressed.group_size == 64
    assert cache.n_compressed == 1
    assert cache.attention_compressed().shape == (1, 1, 512)
