from __future__ import annotations

from mtplx.qwen38_source_proposal import (
    QWEN38_SOURCE_HEAD_LOCAL_PATH,
    QWEN38_SOURCE_HEAD_BYTES,
    QWEN38_SOURCE_HEAD_FORMAT,
    QWEN38_SOURCE_HEAD_REVISION,
    QWEN38_SOURCE_HEAD_SHA256,
    Qwen38SourceProposalError,
    _validate_candidate_body,
    _validate_source_handle,
    compact_token_ids_to_full,
    resolve_qwen38_source_artifact,
)
import pytest


def test_source_handle_validation_uses_safe_open_keys_api() -> None:
    class Slice:
        @staticmethod
        def get_shape():
            return [2, 3]

        @staticmethod
        def get_dtype():
            return "BF16"

    class SafeOpenLike:
        @staticmethod
        def keys():
            return ["weight"]

        @staticmethod
        def get_slice(name):
            assert name == "weight"
            return Slice()

    _validate_source_handle(SafeOpenLike(), {"weight": ([2, 3], "BF16")})

    with pytest.raises(Qwen38SourceProposalError, match="missing missing"):
        _validate_source_handle(SafeOpenLike(), {"missing": ([2, 3], "BF16")})


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


def test_source_artifact_resolver_prefers_the_pulled_model_dependency(
    tmp_path,
) -> None:
    model_path = tmp_path / "model"
    artifact = model_path / QWEN38_SOURCE_HEAD_LOCAL_PATH
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"staged")

    assert resolve_qwen38_source_artifact(model_path) == artifact


def test_candidate_body_validation_rejects_missing_or_mismatched_weights() -> None:
    import mlx.core as mx

    class Candidate:
        @staticmethod
        def parameters():
            return {
                "fc": {"weight": mx.zeros((2, 2), dtype=mx.uint32)},
                "norm": {"weight": mx.zeros((2,), dtype=mx.float32)},
            }

    body = {
        "fc.weight": mx.zeros((2, 2), dtype=mx.uint32),
        "norm.weight": mx.zeros((2,), dtype=mx.bfloat16),
    }
    _validate_candidate_body(Candidate(), body)

    with pytest.raises(Qwen38SourceProposalError, match="missing=.*fc.weight"):
        _validate_candidate_body(Candidate(), {})
    with pytest.raises(Qwen38SourceProposalError, match="shape/dtype"):
        _validate_candidate_body(
            Candidate(),
            {
                "fc.weight": mx.zeros((2, 3), dtype=mx.uint32),
                "norm.weight": mx.zeros((2,), dtype=mx.bfloat16),
            },
        )
