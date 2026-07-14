from __future__ import annotations

import copy
import math
from pathlib import Path
import runpy

import mlx.core as mx
import pytest

from mtplx.attention_context import attention_phase
from mtplx.benchmarks.hy3_kv_quality import (
    COMPLETION_RESERVE_TOKENS,
    validate_quality_result,
)
from mtplx.cache_state import VllmMetalPagedKVCache
from mtplx.kv_quant import PagedKVQuantConfig


_QUALITY_TEST_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_hy3_kv_quality.py"))
)
_valid_quality_result = _QUALITY_TEST_HELPERS["_valid_result"]


def _q4_cache(offset: int) -> tuple[VllmMetalPagedKVCache, mx.array, mx.array]:
    mx.random.seed(4600 + offset)
    keys = 0.25 * mx.random.normal((1, 2, offset, 8), dtype=mx.float16)
    values = 0.25 * mx.random.normal((1, 2, offset, 8), dtype=mx.float16)
    cache = VllmMetalPagedKVCache(
        block_size=4,
        num_blocks=math.ceil((offset + 8) / 4),
        kv_quant_config=PagedKVQuantConfig("q4"),
    )
    cache.update_without_fetch(keys, values)
    return cache, keys, values


def _expected_prefill_coverage(prompt_tokens: int, chunk_size: int) -> tuple[int, int]:
    body_tokens = prompt_tokens - 1
    offsets = [
        min(start + chunk_size, body_tokens)
        for start in range(0, body_tokens, chunk_size)
    ]
    offsets.append(prompt_tokens)
    return len(offsets), sum(offsets)


def _truthful_q4_result() -> dict[str, object]:
    result = copy.deepcopy(_valid_quality_result())
    chunk_size = int(
        result["configuration"]["prefill_chunk_size"]  # type: ignore[index]
    )
    for arm_name, arm in result["arms"].items():  # type: ignore[union-attr]
        for row in arm["rows"]:
            context_tokens = int(row["context_tokens"])
            prompt_tokens = context_tokens - COMPLETION_RESERVE_TOKENS
            for entry in row["kv"]["entries"]:
                entry["active_array_calls"] = 0
                if arm_name != "q4":
                    continue
                prefill_calls, prefill_coverage = _expected_prefill_coverage(
                    prompt_tokens,
                    chunk_size,
                )
                decode_coverage = (
                    COMPLETION_RESERVE_TOKENS * prompt_tokens
                    + COMPLETION_RESERVE_TOKENS * (COMPLETION_RESERVE_TOKENS + 1) // 2
                )
                calls_by_phase = {
                    "prefill": prefill_calls,
                    "ar_decode": COMPLETION_RESERVE_TOKENS,
                }
                coverage_by_phase = {
                    "prefill": prefill_coverage,
                    "ar_decode": decode_coverage,
                }
                attention_calls = sum(calls_by_phase.values())
                expected_unique_tokens = sum(coverage_by_phase.values())
                entry.update(
                    {
                        "kv_quant_dequant_calls": 0,
                        "kv_quant_dequant_time_s": 0.0,
                        "kv_quant_dequant_tokens": 0,
                        "large_q_split_sdpa_fallback_calls": 0,
                        "paged_attention_calls": attention_calls,
                        "kv_quant_attention_calls": attention_calls,
                        "q4_chunked_dequant_attention_calls": attention_calls,
                        "q4_chunked_dequant_tokens": expected_unique_tokens,
                        "q4_chunked_dequant_attention_calls_by_phase": calls_by_phase,
                        "q4_chunked_dequant_attention_path_by_phase": {
                            "prefill": "q4_streaming_softmax",
                            "ar_decode": "q4_streaming_softmax",
                        },
                        "q4_chunked_dequant_expected_unique_tokens_by_phase": (
                            coverage_by_phase
                        ),
                        "q4_chunked_dequant_unique_tokens_by_phase": coverage_by_phase,
                        "q4_chunked_dequant_coverage_failures": 0,
                        "q4_chunked_dequant_configured_chunk_tokens": 1024,
                        "q4_chunked_dequant_realized_peak_chunk_tokens": 1024,
                        "q4_chunked_dequant_realized_peak_chunk_tokens_by_phase": {
                            "prefill": 1024,
                            "ar_decode": 1024,
                        },
                        "q4_chunked_dequant_realized_source_bytes_per_token": 4096,
                        "q4_chunked_dequant_realized_compute_bytes_per_token": 8192,
                        "q4_chunked_dequant_realized_bytes_per_token": 12288,
                        "q4_chunked_dequant_realized_peak_chunk_bytes": 1024 * 12288,
                        "q4_chunked_dequant_realized_peak_chunk_bytes_by_phase": {
                            "prefill": 1024 * 12288,
                            "ar_decode": 1024 * 12288,
                        },
                        "paged_attention_large_q_path": "",
                    }
                )
    return result


def test_q4_prefill_and_decode_never_materialize_whole_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_lm.models.base import scaled_dot_product_attention

    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE", "2")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE", "7")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "off")

    cache, _, _ = _q4_cache(32)
    prefill_queries = mx.random.normal((1, 8, 4, 8), dtype=mx.float16)
    dense_keys, dense_values = cache._paged_range(0, 32)
    expected_prefill = scaled_dot_product_attention(
        prefill_queries,
        dense_keys,
        dense_values,
        cache=None,
        scale=8**-0.5,
        mask="causal",
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Q4 attention attempted a whole-history active array")

    monkeypatch.setattr(cache, "_active_arrays", forbidden)
    monkeypatch.setattr(cache, "_active_attention_arrays", forbidden)
    with attention_phase("prefill"):
        actual_prefill = cache.paged_attention(
            prefill_queries,
            scale=8**-0.5,
            mask="causal",
            impl_override="fast_sdpa_gather",
        )
    assert actual_prefill is not None
    mx.eval(expected_prefill, actual_prefill)

    next_key = mx.random.normal((1, 2, 1, 8), dtype=mx.float16)
    next_value = mx.random.normal((1, 2, 1, 8), dtype=mx.float16)
    cache.update_without_fetch(next_key, next_value)
    decode_query = mx.random.normal((1, 8, 1, 8), dtype=mx.float16)
    decode_keys, decode_values = cache._paged_range(0, 33)
    expected_decode = scaled_dot_product_attention(
        decode_query,
        decode_keys,
        decode_values,
        cache=None,
        scale=8**-0.5,
        mask=None,
    )
    with attention_phase("ar_decode"):
        actual_decode = cache.paged_attention(
            decode_query,
            scale=8**-0.5,
            impl_override="fast_sdpa_gather",
        )
    assert actual_decode is not None
    mx.eval(expected_decode, actual_decode)

    prefill_diff = mx.max(
        mx.abs(expected_prefill.astype(mx.float32) - actual_prefill.astype(mx.float32))
    )
    decode_diff = mx.max(
        mx.abs(expected_decode.astype(mx.float32) - actual_decode.astype(mx.float32))
    )
    mx.eval(prefill_diff, decode_diff)
    assert float(prefill_diff.item()) <= 1e-3
    assert float(decode_diff.item()) <= 1e-3

    stats = cache.paged_stats()
    assert stats["active_array_calls"] == 0
    assert stats["kv_quant_dequant_calls"] == 0
    assert stats["kv_quant_dequant_tokens"] == 0
    assert stats["dense_fallback_calls"] == 0
    assert stats["large_q_split_sdpa_fallback_calls"] == 0
    assert stats["paged_attention_bailouts_by_phase_reason"] == {}
    assert stats["q4_chunked_dequant_attention_calls_by_phase"] == {
        "prefill": 1,
        "ar_decode": 1,
    }
    assert stats["q4_chunked_dequant_attention_path_by_phase"] == {
        "prefill": "q4_streaming_softmax",
        "ar_decode": "q4_streaming_softmax",
    }
    assert stats["q4_chunked_dequant_expected_unique_tokens_by_phase"] == {
        "prefill": 32,
        "ar_decode": 33,
    }
    assert stats["q4_chunked_dequant_unique_tokens_by_phase"] == {
        "prefill": 32,
        "ar_decode": 33,
    }
    assert stats["q4_chunked_dequant_coverage_failures"] == 0
    assert (
        stats["q4_chunked_dequant_realized_calls"] == stats["q4_chunked_dequant_calls"]
    )
    assert stats["q4_chunked_dequant_realized_peak_chunk_tokens"] == 7
    # The bounded lifetime contains both the BF16 dequantized K/V arrays and
    # their FP32 online-attention inputs until the online state is evaluated.
    # Count both simultaneously live representations, not only the FP32 pair.
    assert stats["q4_chunked_dequant_realized_source_bytes_per_token"] == 64
    assert stats["q4_chunked_dequant_realized_compute_bytes_per_token"] == 128
    assert stats["q4_chunked_dequant_realized_bytes_per_token"] == 192
    assert stats["q4_chunked_dequant_realized_peak_chunk_bytes"] == 7 * 192
    assert stats["q4_chunked_dequant_realized_peak_chunk_tokens_by_phase"] == {
        "prefill": 7,
        "ar_decode": 7,
    }
    assert stats["q4_chunked_dequant_realized_peak_chunk_bytes_by_phase"] == {
        "prefill": 7 * 192,
        "ar_decode": 7 * 192,
    }


def test_q4_live_chunk_geometry_matches_realized_mlx_arrays() -> None:
    cache, _, _ = _q4_cache(32)
    chunk_keys, chunk_values = cache._paged_range(0, 7)
    compute_keys = chunk_keys.astype(mx.float32)
    compute_values = chunk_values.astype(mx.float32)

    mx.eval(chunk_keys, chunk_values, compute_keys, compute_values)

    source_bytes = int(chunk_keys.nbytes) + int(chunk_values.nbytes)
    compute_bytes = int(compute_keys.nbytes) + int(compute_values.nbytes)
    assert source_bytes == 7 * 64
    assert compute_bytes == 7 * 128
    assert source_bytes + compute_bytes == 7 * 192


def test_q4_realized_chunk_peak_is_independent_of_total_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE", "7")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "off")

    observed: list[tuple[int, int]] = []
    for offset in (32, 96):
        cache, _, _ = _q4_cache(offset)
        query = mx.zeros((1, 8, 1, 8), dtype=mx.float16)
        with attention_phase("ar_decode"):
            output = cache.paged_attention(query, scale=8**-0.5)
        assert output is not None
        mx.eval(output)
        stats = cache.paged_stats()
        observed.append(
            (
                stats["q4_chunked_dequant_realized_peak_chunk_tokens"],
                stats["q4_chunked_dequant_realized_peak_chunk_bytes"],
            )
        )

    # Each token keeps the source-dtype K/V pair (64 bytes here) and FP32
    # compute pair (128 bytes) live together at the evaluated boundary.
    assert observed == [(7, 7 * 192), (7, 7 * 192)]


def test_quality_validator_accepts_only_bounded_q4_streaming_truth() -> None:
    result = _truthful_q4_result()

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is True
    assert acceptance["gates"]["attention_identity"] is True
    assert acceptance["gates"]["kv_representation_path"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("active_array_calls", 1),
        ("kv_quant_dequant_calls", 1),
        ("kv_quant_dequant_tokens", 4096),
        ("large_q_split_sdpa_fallback_calls", 1),
        ("q4_chunked_dequant_coverage_failures", 1),
        ("q4_chunked_dequant_realized_peak_chunk_tokens", 1025),
    ),
)
def test_quality_validator_rejects_unbounded_or_dense_q4_evidence(
    field: str,
    value: object,
) -> None:
    result = _truthful_q4_result()
    result["arms"]["q4"]["rows"][0]["kv"]["entries"][0][field] = value

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert "attention_identity" in acceptance["rejection_reasons"] or (
        "kv_representation_path" in acceptance["rejection_reasons"]
    )


def test_quality_validator_rejects_inexact_q4_unique_coverage() -> None:
    result = _truthful_q4_result()
    entry = result["arms"]["q4"]["rows"][0]["kv"]["entries"][0]
    entry["q4_chunked_dequant_unique_tokens_by_phase"]["ar_decode"] -= 1

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert "kv_representation_path" in acceptance["rejection_reasons"]


def test_quality_validator_rejects_impossible_q4_chunk_call_count() -> None:
    result = _truthful_q4_result()
    entry = result["arms"]["q4"]["rows"][0]["kv"]["entries"][0]
    impossible_calls = entry["q4_chunked_dequant_attention_calls"]
    entry["q4_chunked_dequant_calls"] = impossible_calls
    entry["q4_chunked_dequant_realized_calls"] = impossible_calls

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert "kv_representation_path" in acceptance["rejection_reasons"]
