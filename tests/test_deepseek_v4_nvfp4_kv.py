import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import (  # noqa: E402
    MIA_NVFP4_RECORD_BYTES,
    MiaNVFP4Rows,
)


def _exact_latent(rows: int = 2) -> mx.array:
    values = mx.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=mx.bfloat16,
    )
    row = mx.tile(values, 32)
    return mx.broadcast_to(row, (1, rows, 512))


def _rope(rows: int = 2) -> mx.array:
    values = (mx.arange(rows * 64, dtype=mx.float32) - 37.0) / 29.0
    return values.reshape(1, rows, 64).astype(mx.bfloat16)


def _as_numpy(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def test_mia_stock432_record_reconstructs_distinct_key_and_value() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    latent = _exact_latent()
    rope = _rope()
    rows = MiaNVFP4Rows()
    rows.append(latent[:, :1], rope[:, :1])
    rows.append(latent[:, 1:], rope[:, 1:])
    key, value = rows.decode()

    assert MIA_NVFP4_RECORD_BYTES == 432
    assert rows.shape == (1, 2, 432)
    assert rows.records.dtype == mx.uint8
    assert rows.nbytes == 2 * 432
    np.testing.assert_array_equal(
        np.array(rows.records[..., 288:304]),
        np.zeros((1, 2, 16), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.array(rows.records[..., 256:288]),
        np.full((1, 2, 32), 0x38, dtype=np.uint8),
    )
    assert int(rows.records[0, 0, 0].item()) == 0x10
    np.testing.assert_array_equal(_as_numpy(value), _as_numpy(latent))
    np.testing.assert_array_equal(_as_numpy(key[..., :448]), _as_numpy(latent[..., :448]))
    np.testing.assert_array_equal(_as_numpy(key[..., 448:]), _as_numpy(rope))
    assert not np.array_equal(_as_numpy(value[..., 448:]), _as_numpy(rope))


def test_mia_stock432_owner_replaces_truncates_and_restores_whole_records() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    rows = MiaNVFP4Rows()
    rows.append(_exact_latent(4), _rope(4))
    replacement = -_exact_latent(1)
    replacement_rope = -_rope(1)
    rows.replace(1, replacement, replacement_rope)
    saved = rows.state

    rows.drop_first(1)
    rows.truncate(2)
    assert rows.shape == (1, 2, 432)
    key, value = rows.decode()
    np.testing.assert_array_equal(_as_numpy(value[:, :1]), _as_numpy(replacement))
    np.testing.assert_array_equal(_as_numpy(key[:, :1, 448:]), _as_numpy(replacement_rope))

    rows.replace_state(saved)
    assert rows.shape == (1, 4, 432)
    restored_key, restored_value = rows.decode(1, 2)
    np.testing.assert_array_equal(_as_numpy(restored_value), _as_numpy(replacement))
    np.testing.assert_array_equal(
        _as_numpy(restored_key[..., 448:]),
        _as_numpy(replacement_rope),
    )
