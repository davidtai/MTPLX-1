from __future__ import annotations

from mtplx.qwen38_source_proposal import (
    QWEN38_SOURCE_HEAD_BYTES,
    QWEN38_SOURCE_HEAD_FORMAT,
    QWEN38_SOURCE_HEAD_REVISION,
    QWEN38_SOURCE_HEAD_SHA256,
    compact_token_ids_to_full,
)


def test_source_artifact_contract_is_the_immutable_huggingface_blob() -> None:
    assert QWEN38_SOURCE_HEAD_REVISION == (
        "ae6282749a52e052496dd5300b4aa441df7301e8"
    )
    assert QWEN38_SOURCE_HEAD_SHA256 == (
        "d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1"
    )
    assert QWEN38_SOURCE_HEAD_BYTES == 427_742_600
    assert QWEN38_SOURCE_HEAD_FORMAT == (
        "qwen38-mtp-incumbent-q4-g64-plus-bf16-qkv-islands-v1"
    )


def test_compact_token_mapping_preserves_prefix_and_maps_controls() -> None:
    import mlx.core as mx

    compact = mx.array([0, 98_303, 98_304, 98_329], dtype=mx.int32)
    mapped = compact_token_ids_to_full(compact)
    assert mapped.tolist() == [0, 98_303, 248_044, 248_069]
