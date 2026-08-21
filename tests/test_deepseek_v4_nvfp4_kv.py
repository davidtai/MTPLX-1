import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import (  # noqa: E402
    MIA_NVFP4_RECORD_BYTES,
    MiaNVFP4Rows,
)
from mtplx.models.deepseek_v4 import DeepseekV4NVFP4Cache  # noqa: E402
from mtplx.kernels.deepseek_v4_nvfp4_mla import nvfp4_sparse_mla  # noqa: E402


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


def test_target_cache_owns_distinct_mia_key_and_value_rows() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    cache = DeepseekV4NVFP4Cache(
        window_size=8,
        compress_ratio=0,
        head_dim=512,
    )
    latent = _exact_latent(3)
    rope = _rope(3)

    records, start = cache.update_window(latent, rope)
    key, value = cache.window.decode()

    assert start == 0
    assert isinstance(cache.window, MiaNVFP4Rows)
    assert cache.window.mode == "nvfp4_stock432"
    assert cache.window.shape == (1, 3, 432)
    assert records.shape == (1, 3, 432)
    np.testing.assert_array_equal(_as_numpy(value), _as_numpy(latent))
    np.testing.assert_array_equal(_as_numpy(key[..., :448]), _as_numpy(latent[..., :448]))
    np.testing.assert_array_equal(_as_numpy(key[..., 448:]), _as_numpy(rope))


@pytest.mark.parametrize("query_rows", [1, 6])
def test_sparse_attention_reads_stock432_records_directly(query_rows: int) -> None:
    if not mx.metal.is_available():
        pytest.skip("requires direct Metal NVFP4 attention")

    window_count = 128
    compressed_count = 17
    window_start = 10
    query_positions = mx.arange(131, 131 + query_rows, dtype=mx.int32)
    latent_values = (
        (mx.arange(window_count * 512, dtype=mx.float32) % 31) - 15
    ) / 6.0
    rope_values = (
        (mx.arange(window_count * 64, dtype=mx.float32) % 23) - 11
    ) / 7.0
    window = MiaNVFP4Rows()
    window.append(
        latent_values.reshape(1, window_count, 512).astype(mx.bfloat16),
        rope_values.reshape(1, window_count, 64).astype(mx.bfloat16),
    )
    compressed = MiaNVFP4Rows()
    compressed.append(
        (
            ((mx.arange(compressed_count * 512, dtype=mx.float32) % 19) - 9)
            / 5.0
        ).reshape(1, compressed_count, 512).astype(mx.bfloat16),
        (
            ((mx.arange(compressed_count * 64, dtype=mx.float32) % 13) - 6)
            / 4.0
        ).reshape(1, compressed_count, 64).astype(mx.bfloat16),
    )
    queries = (
        ((mx.arange(64 * query_rows * 512, dtype=mx.float32) % 29) - 14)
        / 17.0
    ).reshape(1, 64, query_rows, 512).astype(mx.bfloat16)
    sinks = mx.linspace(-0.75, 0.5, 64, dtype=mx.float32)
    selected = mx.broadcast_to(
        mx.array([0, 3, 5, 9, 14], dtype=mx.int32),
        (1, query_rows, 5),
    )
    lengths = mx.minimum(
        mx.arange(3, 3 + query_rows, dtype=mx.int32),
        5,
    )[None]
    scale = 512**-0.5

    output = nvfp4_sparse_mla(
        queries,
        window.records,
        window_start,
        query_positions,
        compressed.records,
        selected,
        lengths,
        sinks,
        scale,
    )

    window_key, window_value = window.decode()
    compressed_key, compressed_value = compressed.decode()
    expected_rows = []
    for query_row in range(query_rows):
        query_position = int(query_positions[query_row].item())
        absolute_window = np.arange(
            window_start,
            window_start + window_count,
        )
        valid_window = np.flatnonzero(
            (absolute_window <= query_position)
            & (absolute_window > query_position - 128)
        )
        valid_window = mx.array(valid_window, dtype=mx.int32)
        chosen = selected[0, query_row, : int(lengths[0, query_row].item())]
        key = mx.concatenate(
            [window_key[:, valid_window], compressed_key[:, chosen]],
            axis=1,
        )
        value = mx.concatenate(
            [window_value[:, valid_window], compressed_value[:, chosen]],
            axis=1,
        )
        query = queries[:, :, query_row : query_row + 1].astype(mx.float32)
        scores = (query * scale) @ mx.swapaxes(key[:, None].astype(mx.float32), -1, -2)
        maximum = mx.maximum(
            mx.max(scores, axis=-1, keepdims=True),
            sinks.reshape(1, 64, 1, 1),
        )
        weights = mx.exp(scores - maximum)
        denominator = mx.sum(weights, axis=-1, keepdims=True) + mx.exp(
            sinks.reshape(1, 64, 1, 1) - maximum
        )
        expected_rows.append(
            ((weights / denominator) @ value[:, None].astype(mx.float32)).astype(
                mx.bfloat16
            )
        )
    expected = mx.concatenate(expected_rows, axis=2)
    mx.eval(output, expected)

    assert output.shape == (1, 64, query_rows, 512)
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-2,
    )
