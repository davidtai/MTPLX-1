from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import (  # noqa: E402
    MIA_NVFP4_RECORD_BYTES,
    MiaNVFP4Rows,
    PagedMiaNVFP4Rows,
)
from mtplx.deepseek_v4_paged_indexer import (  # noqa: E402
    MiaTopKSelection,
    PagedMiaIndexerRows,
    _run_paged_indexer_topk,
    paged_indexer_scores,
    paged_indexer_tiled_scores,
)
from mtplx.attention_context import attention_phase  # noqa: E402
from mtplx.models import deepseek_v4 as deepseek_v4_module  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4NVFP4Cache,
    Indexer,
)
from mtplx.kernels.deepseek_v4_nvfp4_mla import (  # noqa: E402
    nvfp4_prefill_mla,
    nvfp4_sparse_mla,
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


def test_mia_stock432_record_quantizes_the_post_rope_row_for_key_and_value() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    latent = _exact_latent()
    rope = -latent[..., 448:]
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
    stored = mx.concatenate([latent[..., :448], rope], axis=-1)
    np.testing.assert_array_equal(_as_numpy(value), _as_numpy(stored))
    np.testing.assert_array_equal(_as_numpy(key[..., :448]), _as_numpy(latent[..., :448]))
    np.testing.assert_array_equal(_as_numpy(key[..., 448:]), _as_numpy(rope))
    np.testing.assert_array_equal(_as_numpy(value[..., 448:]), _as_numpy(rope))


def test_mia_stock432_owner_replaces_truncates_and_restores_whole_records() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    rows = MiaNVFP4Rows()
    rows.append(_exact_latent(4), _rope(4))
    replacement = -_exact_latent(1)
    replacement_rope = -replacement[..., 448:]
    rows.replace(1, replacement, replacement_rope)
    saved = rows.state

    rows.drop_first(1)
    rows.truncate(2)
    assert rows.shape == (1, 2, 432)
    key, value = rows.decode()
    expected_replacement = mx.concatenate(
        [replacement[..., :448], replacement_rope],
        axis=-1,
    )
    np.testing.assert_array_equal(
        _as_numpy(value[:, :1]),
        _as_numpy(expected_replacement),
    )
    np.testing.assert_array_equal(_as_numpy(key[:, :1, 448:]), _as_numpy(replacement_rope))

    rows.replace_state(saved)
    assert rows.shape == (1, 4, 432)
    restored_key, restored_value = rows.decode(1, 2)
    np.testing.assert_array_equal(
        _as_numpy(restored_value),
        _as_numpy(expected_replacement),
    )
    np.testing.assert_array_equal(
        _as_numpy(restored_key[..., 448:]),
        _as_numpy(replacement_rope),
    )


def test_paged_mia_stock432_owner_keeps_fixed_pages_across_writes() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    rows = PagedMiaNVFP4Rows(capacity_rows=8, block_size=4)
    pages = rows.pages
    rows.append(_exact_latent(3), _rope(3))
    rows.append(_exact_latent(2), _rope(2))

    assert rows.pages is pages
    assert rows.shape == (1, 5, 432)
    assert rows.paged_records.records is pages
    assert rows.paged_records.length == 5
    assert rows.paged_records.block_size == 4

    replacement = -_exact_latent(1)
    replacement_rope = -replacement[..., 448:]
    rows.replace(1, replacement, replacement_rope)
    rows.truncate(4)
    _key, value = rows.decode()
    expected_replacement = mx.concatenate(
        [replacement[..., :448], replacement_rope], axis=-1
    )
    np.testing.assert_array_equal(
        _as_numpy(value[:, 1:2]),
        _as_numpy(expected_replacement),
    )
    with pytest.raises(ValueError, match="capacity exceeded"):
        rows.append(_exact_latent(5), _rope(5))


def test_target_cache_owns_distinct_mia_key_and_value_rows() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    cache = DeepseekV4NVFP4Cache(
        window_size=8,
        compress_ratio=0,
        head_dim=512,
    )
    latent = _exact_latent(3)
    rope = -latent[..., 448:]

    records, start = cache.update_window(latent, rope)
    key, value = cache.window.decode()

    assert start == 0
    assert isinstance(cache.window, MiaNVFP4Rows)
    assert cache.window.mode == "nvfp4_stock432"
    assert cache.window.shape == (1, 3, 432)
    assert records.shape == (1, 3, 432)
    stored = mx.concatenate([latent[..., :448], rope], axis=-1)
    np.testing.assert_array_equal(_as_numpy(value), _as_numpy(stored))
    np.testing.assert_array_equal(_as_numpy(key[..., :448]), _as_numpy(latent[..., :448]))
    np.testing.assert_array_equal(_as_numpy(key[..., 448:]), _as_numpy(rope))


def test_target_compressed_cache_uses_fixed_stock432_pages() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal NVFP4 record packer")

    cache = DeepseekV4NVFP4Cache(
        window_size=128,
        compress_ratio=4,
        head_dim=512,
        capacity_tokens=33,
    )
    pages = cache.compressed.pages
    cache.compressed.append(_exact_latent(5), _rope(5))
    cache.compressed.append(_exact_latent(2), _rope(2))

    assert isinstance(cache.compressed, PagedMiaNVFP4Rows)
    assert cache.compressed.capacity == 9
    assert cache.compressed.pages is pages
    assert cache.attention_compressed().records is pages
    assert isinstance(cache.index_compressed, PagedMiaIndexerRows)
    assert cache.index_compressed.capacity == 9


def test_paged_mia_indexer_reads_132_byte_fp8_records_directly() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires Metal paged indexer")

    row_values = (
        ((mx.arange(7 * 128, dtype=mx.float32) % 23) - 11) / 9.0
    ).reshape(1, 7, 128).astype(mx.bfloat16)
    rows = PagedMiaIndexerRows(capacity_rows=16, block_size=4)
    pages = rows.pages
    rows.append(row_values[:, :3])
    rows.append(row_values[:, 3:])

    query = (
        ((mx.arange(2 * 64 * 128, dtype=mx.float32) % 29) - 14) / 13.0
    ).reshape(1, 2, 64, 128).astype(mx.bfloat16)
    weights = mx.linspace(-0.2, 0.3, 2 * 64).reshape(1, 2, 64)
    actual = paged_indexer_scores(query, weights, rows.paged_records)
    tiled = paged_indexer_tiled_scores(query, weights, rows.paged_records)

    query_rows = PagedMiaIndexerRows(capacity_rows=128, block_size=64)
    query_rows.append(query.reshape(1, 2 * 64, 128))
    quant_query = query_rows.decode().reshape(1, 2, 64, 128)
    quant_rows = rows.decode()
    dot = mx.einsum("bshd,btd->bsht", quant_query, quant_rows)
    expected = mx.sum(mx.maximum(dot, 0.0) * weights[..., None], axis=2)
    mx.eval(actual, tiled, expected)

    assert rows.pages is pages
    assert rows.paged_records.records is pages
    assert rows.paged_records.record_bytes == 132
    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.array(tiled),
        np.array(actual),
        rtol=2e-3,
        atol=2e-3,
    )


def test_mia_indexer_streams_bounded_score_slices_into_compact_topk(
    monkeypatch,
) -> None:
    score_slice_widths = []

    def fake_score_slice(q_records, weights, rows, row_start, row_count):
        del q_records, weights, rows
        score_slice_widths.append(row_count)
        scores = mx.arange(row_start, row_start + row_count, dtype=mx.float32)
        return mx.broadcast_to(scores[None, None], (1, 2, row_count))

    monkeypatch.setattr(
        "mtplx.deepseek_v4_paged_indexer._run_paged_indexer_score_slice",
        fake_score_slice,
    )
    monkeypatch.setattr(
        "mtplx.deepseek_v4_paged_indexer._pack_indexer132",
        lambda queries: queries,
    )
    rows = SimpleNamespace(length=300)
    selection = _run_paged_indexer_topk(
        mx.zeros((1, 2, 64, 128), dtype=mx.bfloat16),
        mx.zeros((1, 2, 64), dtype=mx.float32),
        mx.array([7, 1199], dtype=mx.int32),
        rows,
        topk=3,
        compress_ratio=4,
        score_chunk_rows=128,
    )

    assert isinstance(selection, MiaTopKSelection)
    assert score_slice_widths == [128, 128, 44]
    assert tuple(selection.indices.shape) == (1, 2, 3)
    assert tuple(selection.lengths.shape) == (1, 2)
    np.testing.assert_array_equal(
        np.array(selection.indices),
        np.array([[[0, 1, 300], [297, 298, 299]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.array(selection.lengths), [[2, 3]])


def test_mia_indexer_install_removes_the_non_source_hadamard(monkeypatch) -> None:
    def installed(*_args):
        return None

    monkeypatch.setattr(
        deepseek_v4_module,
        "install_paged_indexer_topk",
        lambda **_kwargs: installed,
    )
    indexer = Indexer.__new__(Indexer)
    indexer.n_heads = 64
    indexer.head_dim = 128
    indexer.index_topk = 512
    indexer.compress_ratio = 4
    indexer.compressor = SimpleNamespace(rotate=True)

    indexer.install_mia_paged_topk()

    query = mx.zeros((1, 1, 64, 128), dtype=mx.bfloat16)
    assert indexer.compressor.rotate is False
    assert indexer._prepare_query_rows(query) is query
    assert indexer._select_rows is installed


def test_mia_attention_routes_nax_prefill_by_phase(monkeypatch) -> None:
    prefill_result = mx.array([11], dtype=mx.int32)
    direct_result = mx.array([22], dtype=mx.int32)
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_nvfp4_prefill_mla",
        lambda **_kwargs: lambda *_args: prefill_result,
    )
    monkeypatch.setattr(
        deepseek_v4_module,
        "install_nvfp4_sparse_mla",
        lambda **_kwargs: lambda *_args: direct_result,
    )

    class FakeIndexer:
        def install_mia_paged_topk(self) -> None:
            return None

    attn = deepseek_v4_module.DeepseekV4Attention.__new__(
        deepseek_v4_module.DeepseekV4Attention
    )
    attn.head_dim = 512
    attn.rope_head_dim = 64
    attn.n_heads = 64
    attn.window_size = 128
    attn.compress_ratio = 4
    attn.attn_sink = mx.zeros((64,), dtype=mx.float32)
    attn.softmax_scale = 512**-0.5
    attn.indexer = FakeIndexer()
    attn.install_mia_nvfp4_attention()

    selection = MiaTopKSelection(
        indices=mx.zeros((1, 2, 1), dtype=mx.int32),
        lengths=mx.ones((1, 2), dtype=mx.int32),
    )

    def run(query_rows: int):
        return attn._mia_cached_attention(
            mx.zeros((1, 64, query_rows, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 432), dtype=mx.uint8),
            None,
            0,
            mx.arange(query_rows, dtype=mx.int32),
            4,
            MiaTopKSelection(
                indices=selection.indices[:, :query_rows],
                lengths=selection.lengths[:, :query_rows],
            ),
            None,
        )

    with attention_phase("prefill"):
        assert run(1) is prefill_result
        assert run(2) is prefill_result
    with attention_phase("ar_decode"):
        assert run(2) is direct_result


@pytest.mark.parametrize("query_rows", [1, 6])
@pytest.mark.parametrize("paged_compressed", [False, True])
@pytest.mark.parametrize(
    "attention_impl",
    [nvfp4_sparse_mla, nvfp4_prefill_mla],
    ids=["direct-decode", "nax-prefill"],
)
def test_sparse_attention_reads_stock432_records_directly(
    query_rows: int,
    paged_compressed: bool,
    attention_impl,
) -> None:
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
    compressed = (
        PagedMiaNVFP4Rows(capacity_rows=32, block_size=8)
        if paged_compressed
        else MiaNVFP4Rows()
    )
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

    output = attention_impl(
        queries,
        window.records,
        window_start,
        query_positions,
        compressed.paged_records if paged_compressed else compressed.records,
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
