from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.benchmarks.hy3_dynamic_memory_artifacts import (
    SCHEMA_ARTIFACT_ATTESTATION,
    ArtifactPins,
)
from mtplx.benchmarks.hy3_dynamic_memory_hardware import Hy3HardwareConfig
from mtplx.benchmarks.runners.hy3_dynamic_memory import canonical_sha256
from mtplx.benchmarks.hy3_kv_quality import (
    BF16_KV_BYTES_PER_TOKEN,
    COMPLETION_RESERVE_TOKENS,
    EXPERT_CACHE_LIMIT_BYTES,
    GIB,
    HY3_KV_QUALITY_SCHEMA,
    Q4_KV_BYTES_PER_TOKEN,
    QUALITY_CONTEXTS,
    QualityGateError,
    build_quality_prompt,
    build_quality_runtime_config,
    page_slots_for_context,
    quality_environment,
    score_retrieval_text,
    validate_quality_result,
)


class _CharacterTokenizer:
    """Small reversible tokenizer that exercises prompt geometry exactly."""

    vocab_size = 1_114_112

    @staticmethod
    def encode(text: str, **_kwargs: object) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int] | tuple[int, ...], **_kwargs: object) -> str:
        return "".join(chr(int(token)) for token in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs: object,
    ) -> list[int] | str:
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered) if tokenize else rendered


def _pins() -> ArtifactPins:
    return ArtifactPins.from_mapping(
        {
            "model_config_sha256": "a" * 64,
            "manifest_file_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "sidecar_file": "experts.bin",
            "sidecar_bytes": 1,
            "sidecar_sha256": "d" * 64,
            "resident_payload_bytes": 1,
            "resident_payload_sha256": "e" * 64,
            "source_revision": "revision-a",
        }
    )


def _hardware_config(tmp_path: Path) -> Hy3HardwareConfig:
    return Hy3HardwareConfig(
        repo_root=tmp_path,
        model_root=tmp_path / "model",
        manifest=tmp_path / "manifest.json",
        model_artifact_id="pipenetwork/Hy3-4bit@revision-a",
        artifact_pins=_pins(),
    )


_TEST_MANIFEST_FILE_SHA256 = "2" * 64
_TEST_MANIFEST_CONTENT_SHA256 = "7" * 64
_TEST_DECODE_SCOPE = "producer-decoded;offline-structural-tokenizer-files-required"
_TEST_ROUTE_SCOPE = "producer-verified-live-manifest;offline-structural"


def _test_tokenizer_identity() -> dict[str, object]:
    core: dict[str, object] = {
        "schema": "mtplx-tokenizer-identity-v1",
        "files": [
            {
                "name": name,
                "size": index + 1,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            for index, name in enumerate(
                (
                    "chat_template.jinja",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            )
        ],
        "runtime": {
            "class": "tests._CharacterTokenizer",
            "vocab_size": 1_114_112,
        },
        "offline_decode_verification": (
            "structural-only-tokenizer-files-required-for-redecode"
        ),
    }
    return {**core, "identity_sha256": canonical_sha256(core)}


def _decode_binding(token_ids: list[int], text: str) -> dict[str, object]:
    core: dict[str, object] = {
        "tokenizer_identity_sha256": _test_tokenizer_identity()["identity_sha256"],
        "generated_token_sha256": hashlib.sha256(
            b"".join(int(token).to_bytes(8, "little") for token in token_ids)
        ).hexdigest(),
        "generated_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "producer_decode_match": True,
        "verification_scope": _TEST_DECODE_SCOPE,
    }
    return {**core, "binding_sha256": canonical_sha256(core)}


def _route_evidence(route_trace: list[dict[str, object]]) -> dict[str, object]:
    expert_hashes = {
        "1:2": hashlib.sha256(b"synthetic-manifest-record-1:2").hexdigest(),
        "1:3": hashlib.sha256(b"synthetic-manifest-record-1:3").hexdigest(),
    }
    core: dict[str, object] = {
        "expert_manifest_sha256": _TEST_MANIFEST_FILE_SHA256,
        "expert_manifest_content_sha256": _TEST_MANIFEST_CONTENT_SHA256,
        "expert_hashes_sha256": canonical_sha256(expert_hashes),
        "routed_pair_count": len(expert_hashes),
        "live_manifest_records_verified": True,
        "verification_scope": _TEST_ROUTE_SCOPE,
    }
    return {
        "trace": route_trace,
        "trace_sha256": canonical_sha256(route_trace),
        "expert_hashes": expert_hashes,
        "manifest_binding": {**core, "binding_sha256": canonical_sha256(core)},
    }


def _routed_expert_attestation() -> dict[str, object]:
    record_hashes = {
        "1:2": hashlib.sha256(b"synthetic-manifest-record-1:2").hexdigest(),
        "1:3": hashlib.sha256(b"synthetic-manifest-record-1:3").hexdigest(),
    }
    core: dict[str, object] = {
        "schema": "mtplx-routed-expert-attestation-v1",
        "expert_manifest_content_sha256": _TEST_MANIFEST_CONTENT_SHA256,
        "record_hashes": record_hashes,
        "record_hashes_sha256": canonical_sha256(record_hashes),
        "routed_pair_count": len(record_hashes),
        "producer_manifest_records_verified": True,
        "verification_scope": _TEST_ROUTE_SCOPE,
    }
    return {**core, "binding_sha256": canonical_sha256(core)}


@pytest.mark.parametrize(
    ("context_tokens", "expected_slots"),
    ((1, 1), (16, 1), (17, 2), (4096, 256), (4097, 257), (131072, 8192)),
)
def test_page_slots_are_derived_from_each_context(
    context_tokens: int,
    expected_slots: int,
) -> None:
    assert page_slots_for_context(context_tokens) == expected_slots


def test_quality_environment_uses_exact_dynamic_geometry_per_row() -> None:
    q4 = quality_environment("q4", context_tokens=4097, prefill_chunk_size=1024)
    bf16 = quality_environment("bf16", context_tokens=32_768)

    assert q4["MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"] == "257"
    assert bf16["MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"] == "2048"
    assert q4["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] == "q4"
    assert q4["MTPLX_PAGED_KV_QUANT"] == "q4"
    assert bf16["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] == "off"
    assert bf16["MTPLX_PAGED_KV_QUANT"] == "off"
    assert q4["MTPLX_DYNAMIC_PAGED_KV"] == "0"
    assert q4["MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"] == "0"
    assert q4["MTPLX_PREFILL_CHUNK_SIZE"] == "1024"
    assert q4["MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK"] == "1"
    assert q4["MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS"] == "1"
    assert q4["MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK"] == "0"
    assert q4["MTPLX_ALLOW_PAGED_ACTIVE_ARRAY_SNAPSHOT"] == "0"
    assert bf16["MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK"] == "1"
    assert bf16["MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS"] == "1"


def test_dynamic_q4_quality_environment_starts_one_brokered_page() -> None:
    environment = quality_environment(
        "q4_dynamic",
        context_tokens=65_536,
        prefill_chunk_size=1024,
    )

    assert environment["MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"] == "1"
    assert environment["MTPLX_DYNAMIC_PAGED_KV"] == "1"
    assert environment["MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS"] == "1"
    assert environment["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] == "q4"
    assert environment["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] == "0"


@pytest.mark.parametrize(
    ("representation", "key_dtype", "value_dtype", "scale_dtype", "expected"),
    (
        (
            "bf16",
            mx.bfloat16,
            mx.bfloat16,
            None,
            {
                "key": "bfloat16",
                "value": "bfloat16",
                "key_scale": None,
                "value_scale": None,
            },
        ),
        (
            "q4",
            mx.uint8,
            mx.uint8,
            mx.float16,
            {
                "key": "uint8",
                "value": "uint8",
                "key_scale": "float16",
                "value_scale": "float16",
            },
        ),
    ),
)
def test_kv_evidence_attests_live_array_storage_dtypes(
    representation: str,
    key_dtype: object,
    value_dtype: object,
    scale_dtype: object | None,
    expected: dict[str, str | None],
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    key_scale = None if scale_dtype is None else mx.zeros((1,), dtype=scale_dtype)
    value_scale = None if scale_dtype is None else mx.zeros((1,), dtype=scale_dtype)
    entry = SimpleNamespace(
        key_cache=mx.zeros((1,), dtype=key_dtype),
        value_cache=mx.zeros((1,), dtype=value_dtype),
        key_scale_cache=key_scale,
        value_scale_cache=value_scale,
        paged_stats=lambda: {"bytes": 1},
    )

    evidence = hy3_kv_quality._kv_evidence(
        [entry],
        representation=representation,
        context_tokens=16,
        environment={},
        initial_stats=[{"num_blocks": 1, "offset": 0, "bytes": 0}],
    )

    assert evidence["entries"][0]["storage_dtypes"] == expected


def test_retrieval_prompt_is_exact_and_keeps_unique_context_specific_edges() -> None:
    tokenizer = _CharacterTokenizer()
    first = build_quality_prompt(tokenizer, context_tokens=4096)
    second = build_quality_prompt(tokenizer, context_tokens=32_768)

    assert len(first.token_ids) == 4096 - COMPLETION_RESERVE_TOKENS
    assert first.prompt_tokens == len(first.token_ids)
    assert first.earliest_marker != second.earliest_marker
    assert first.latest_marker != second.latest_marker
    assert first.rendered_prompt.count(first.earliest_marker) == 1
    assert first.rendered_prompt.count(first.latest_marker) == 1
    assert first.earliest_token_index < 256
    assert first.latest_token_index >= first.prompt_tokens - 512
    assert (
        first.prompt_sha256
        == hashlib.sha256(
            b"".join(int(token).to_bytes(8, "little") for token in first.token_ids)
        ).hexdigest()
    )


def test_tokenizer_identity_hashes_exact_decode_artifacts(tmp_path: Path) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    for index, name in enumerate(
        (
            "chat_template.jinja",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    ):
        (tmp_path / name).write_text(f"artifact-{index}\n", encoding="utf-8")

    identity = hy3_kv_quality._attest_tokenizer_identity(
        _CharacterTokenizer(),
        tmp_path,
    )

    assert [value["name"] for value in identity["files"]] == [
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert identity["runtime"]["vocab_size"] == _CharacterTokenizer.vocab_size
    assert len(identity["identity_sha256"]) == 64


def test_retrieval_scoring_requires_both_exact_markers_and_rejects_stale_values() -> (
    None
):
    score = score_retrieval_text(
        "EARLIEST=first-marker\nLATEST=last-marker\n",
        expected_earliest="first-marker",
        expected_latest="last-marker",
        forbidden_markers=("old-first", "old-last"),
    )
    stale = score_retrieval_text(
        "EARLIEST=first-marker\nLATEST=last-marker\nold-first",
        expected_earliest="first-marker",
        expected_latest="last-marker",
        forbidden_markers=("old-first",),
    )
    wrong = score_retrieval_text(
        "EARLIEST=first-marker\nLATEST=wrong-marker\n",
        expected_earliest="first-marker",
        expected_latest="last-marker",
    )

    assert score["passed"] is True
    assert score["earliest_exact"] is True
    assert score["latest_exact"] is True
    assert score["stale_markers"] == []
    assert stale["passed"] is False
    assert stale["stale_markers"] == ["old-first"]
    assert wrong["passed"] is False


def test_bf16_and_q4_use_one_identical_32_gib_expert_cap(tmp_path: Path) -> None:
    hardware = _hardware_config(tmp_path)
    bf16 = build_quality_runtime_config(hardware, "bf16")
    q4 = build_quality_runtime_config(hardware, "q4")

    from mtplx.expert_streaming_models import HY3_Q4

    bf16_plan = bf16.memory_plan(HY3_Q4)
    q4_plan = q4.memory_plan(HY3_Q4)
    assert bf16.expert_cache_limit_bytes == EXPERT_CACHE_LIMIT_BYTES == 32 * GIB
    assert q4.expert_cache_limit_bytes == EXPERT_CACHE_LIMIT_BYTES
    assert bf16.kv_bytes_per_token_override == BF16_KV_BYTES_PER_TOKEN
    assert q4.kv_bytes_per_token_override == Q4_KV_BYTES_PER_TOKEN
    assert bf16_plan.persistent_slots == q4_plan.persistent_slots
    assert bf16_plan.persistent_cache_bytes == q4_plan.persistent_cache_bytes


def test_dynamic_q4_quality_uses_same_expert_cap_and_enables_broker(
    tmp_path: Path,
) -> None:
    hardware = _hardware_config(tmp_path)
    dynamic = build_quality_runtime_config(hardware, "q4_dynamic")

    assert dynamic.expert_cache_limit_bytes == EXPERT_CACHE_LIMIT_BYTES
    assert dynamic.kv_bytes_per_token_override == Q4_KV_BYTES_PER_TOKEN
    assert dynamic.dynamic_expert_cache is True
    assert dynamic.slot_layout == "direct-slots"


def test_representation_runner_restores_environment_on_early_attestation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    key = "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"
    monkeypatch.setenv(key, "sentinel")
    monkeypatch.setattr(
        hy3_kv_quality,
        "_attest_config_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            QualityGateError("attestation failed")
        ),
    )

    with pytest.raises(QualityGateError, match="attestation failed"):
        hy3_kv_quality._run_quality_representation(
            hardware_config=_hardware_config(tmp_path),
            representation="bf16",
            prompts=None,
        )

    assert os.environ[key] == "sentinel"


def test_representation_runner_restores_environment_when_runtime_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mtplx.runtime as runtime_module
    from mtplx.benchmarks import hy3_kv_quality

    key = "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"
    manifest = object()
    runtime = SimpleNamespace(
        expert_streaming=SimpleNamespace(manifest=manifest),
        close=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("close failed")),
    )
    attestation = SimpleNamespace(manifest=manifest, model_config={})
    monkeypatch.setenv(key, "sentinel")
    monkeypatch.setattr(
        hy3_kv_quality,
        "_attest_config_artifact",
        lambda *_args, **_kwargs: attestation,
    )
    monkeypatch.setattr(runtime_module, "load", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(
        hy3_kv_quality,
        "_require_loaded_manifest_identity",
        lambda *_args: (_ for _ in ()).throw(QualityGateError("load gate failed")),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        hy3_kv_quality._run_quality_representation(
            hardware_config=_hardware_config(tmp_path),
            representation="bf16",
            prompts=None,
        )

    assert os.environ[key] == "sentinel"


def _healthy_row(
    representation: str,
    context_tokens: int,
    prompt: dict[str, object],
    *,
    generated_token_ids: list[int],
) -> dict[str, object]:
    page_slots = page_slots_for_context(context_tokens)
    capacity_tokens = page_slots * 16
    is_q4 = representation in {"q4", "q4_dynamic"}
    bytes_per_token = Q4_KV_BYTES_PER_TOKEN if is_q4 else BF16_KV_BYTES_PER_TOKEN
    earliest = str(prompt["earliest_marker"])
    latest = str(prompt["latest_marker"])
    generated_text = f"EARLIEST={earliest}\nLATEST={latest}\n"
    route_trace = [{"layer": 1, "expert_ids": [2, 3]}]
    decode_binding = _decode_binding(generated_token_ids, generated_text)
    prompt_tokens = context_tokens - COMPLETION_RESERVE_TOKENS
    prefill_chunk_size = 2048
    body_tokens = prompt_tokens - 1
    prefill_offsets = [
        min(start + prefill_chunk_size, body_tokens)
        for start in range(0, body_tokens, prefill_chunk_size)
    ]
    prefill_offsets.append(prompt_tokens)
    q4_calls_by_phase = {
        "prefill": len(prefill_offsets),
        "ar_decode": COMPLETION_RESERVE_TOKENS,
    }
    q4_coverage_by_phase = {
        "prefill": sum(prefill_offsets),
        "ar_decode": (
            COMPLETION_RESERVE_TOKENS * prompt_tokens
            + COMPLETION_RESERVE_TOKENS * (COMPLETION_RESERVE_TOKENS + 1) // 2
        ),
    }
    q4_attention_calls = sum(q4_calls_by_phase.values())
    q4_unique_tokens = sum(q4_coverage_by_phase.values())
    q4_chunk_calls = sum(math.ceil(offset / 1024) for offset in prefill_offsets) + sum(
        math.ceil(offset / 1024)
        for offset in range(prompt_tokens + 1, context_tokens + 1)
    )
    q4_peak_tokens_by_phase = {
        "prefill": min(1024, prompt_tokens),
        "ar_decode": min(1024, context_tokens),
    }
    entry = {
        "mode": ("vllm_metal_paged_kv_q4" if is_q4 else "vllm_metal_paged"),
        "block_size": 16,
        "num_blocks": page_slots,
        "capacity": capacity_tokens,
        "offset": context_tokens,
        "bytes": bytes_per_token * capacity_tokens,
        "kv_quant": int(is_q4),
        "kv_quant_mode": "q4" if is_q4 else "",
        "updates": 1,
        "cache_write_time_s": 1.0,
        "attention_time_s": 1.0,
        "kv_quant_dequant_calls": 0,
        "kv_quant_dequant_time_s": 0.0,
        "kv_quant_dequant_tokens": 0,
        "kv_quant_attention_calls": (q4_attention_calls if is_q4 else 0),
        "q4_chunked_dequant_attention_calls": (q4_attention_calls if is_q4 else 0),
        "q4_chunked_dequant_attention_calls_by_phase": (
            q4_calls_by_phase if is_q4 else {}
        ),
        "q4_chunked_dequant_attention_path_by_phase": (
            {
                "prefill": "q4_streaming_softmax",
                "ar_decode": "q4_streaming_softmax",
            }
            if is_q4
            else {}
        ),
        "q4_chunked_dequant_calls": q4_chunk_calls if is_q4 else 0,
        "q4_chunked_dequant_time_s": 1.0 if is_q4 else 0.0,
        "q4_chunked_dequant_tokens": (q4_unique_tokens if is_q4 else 0),
        "q4_chunked_dequant_expected_unique_tokens_by_phase": (
            q4_coverage_by_phase if is_q4 else {}
        ),
        "q4_chunked_dequant_unique_tokens_by_phase": (
            q4_coverage_by_phase if is_q4 else {}
        ),
        "q4_chunked_dequant_coverage_failures": 0,
        "q4_chunked_dequant_realized_calls": (q4_chunk_calls if is_q4 else 0),
        "q4_chunked_dequant_configured_chunk_tokens": (1024 if is_q4 else 0),
        "q4_chunked_dequant_realized_peak_chunk_tokens": (1024 if is_q4 else 0),
        "q4_chunked_dequant_realized_peak_chunk_tokens_by_phase": (
            q4_peak_tokens_by_phase if is_q4 else {}
        ),
        "q4_chunked_dequant_realized_source_bytes_per_token": (4096 if is_q4 else 0),
        "q4_chunked_dequant_realized_compute_bytes_per_token": (8192 if is_q4 else 0),
        "q4_chunked_dequant_realized_bytes_per_token": (12288 if is_q4 else 0),
        "q4_chunked_dequant_realized_peak_chunk_bytes": (1024 * 12288 if is_q4 else 0),
        "q4_chunked_dequant_realized_peak_chunk_bytes_by_phase": (
            {phase: tokens * 12288 for phase, tokens in q4_peak_tokens_by_phase.items()}
            if is_q4
            else {}
        ),
        "paged_attention_calls": q4_attention_calls if is_q4 else 1,
        "partitioned_paged_calls": 0 if is_q4 else 1,
        "active_array_calls": 0,
        "dense_fallback_calls": 0,
        "large_q_split_sdpa_fallback_calls": 0,
        "paged_attention_large_q_path": ("" if is_q4 else "partitioned_paged"),
        "paged_attention_bailouts_by_phase_reason": {},
        "sliding_window": 0,
        "storage_dtypes": (
            {
                "key": "uint8",
                "value": "uint8",
                "key_scale": "float16",
                "value_scale": "float16",
            }
            if is_q4
            else {
                "key": "bfloat16",
                "value": "bfloat16",
                "key_scale": None,
                "value_scale": None,
            }
        ),
    }
    physical_bytes = bytes_per_token * capacity_tokens
    if representation == "q4_dynamic":
        assert physical_bytes % 80 == 0
        per_entry_bytes = physical_bytes // 80
        entry["bytes"] = per_entry_bytes
        entries = [copy.deepcopy(entry) for _ in range(80)]
        owners = [
            {
                "entry_index": index,
                "cache_id": f"target:fixture:{index}",
                "allocation_count": page_slots,
                "committed_physical_bytes": per_entry_bytes,
                "allocation_handle_bytes": per_entry_bytes,
            }
            for index in range(80)
        ]
        broker = {
            "kv_physical_bytes": physical_bytes,
            "owned_kv_physical_bytes": physical_bytes,
            "unreconciled_kv_physical_bytes": 0,
            "pending_kv_ticket_id": None,
            "failed_closed": False,
        }
    else:
        entries = [entry]
        owners = []
        broker = None
    entry_count = len(entries)
    broker_release = (
        {
            "kv_physical_bytes": 0,
            "owned_kv_physical_bytes": 0,
            "unreconciled_kv_physical_bytes": 0,
            "pending_kv_ticket_id": None,
            "failed_closed": False,
        }
        if representation == "q4_dynamic"
        else None
    )
    return {
        "context_tokens": context_tokens,
        "prompt_tokens": context_tokens - COMPLETION_RESERVE_TOKENS,
        "completion_tokens": COMPLETION_RESERVE_TOKENS,
        "prompt_sha256": prompt["prompt_sha256"],
        "generated_token_ids": generated_token_ids,
        "generated_token_sha256": hashlib.sha256(
            b"".join(int(token).to_bytes(8, "little") for token in generated_token_ids)
        ).hexdigest(),
        "generated_text": generated_text,
        "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
        "decode_binding": decode_binding,
        "retrieval": {
            "expected_earliest": earliest,
            "expected_latest": latest,
            "observed_earliest": [earliest],
            "observed_latest": [latest],
            "earliest_exact": True,
            "latest_exact": True,
            "stale_markers": [],
            "passed": True,
        },
        "kv": {
            "representation": representation,
            "bytes_per_token": bytes_per_token,
            "block_size_tokens": 16,
            "required_page_slots": page_slots,
            "physical_capacity_tokens": capacity_tokens,
            "physical_bytes": physical_bytes,
            "entry_count": entry_count,
            "initial": {
                "entry_count": entry_count,
                "num_blocks": (
                    [1] * entry_count
                    if representation == "q4_dynamic"
                    else [page_slots] * entry_count
                ),
                "entry_offsets": [0] * entry_count,
                "entry_bytes": (
                    [Q4_KV_BYTES_PER_TOKEN * 16 // 80] * entry_count
                    if representation == "q4_dynamic"
                    else [0] * entry_count
                ),
                "physical_bytes": (
                    Q4_KV_BYTES_PER_TOKEN * 16 if representation == "q4_dynamic" else 0
                ),
            },
            "attention_environment": quality_environment(
                representation,
                context_tokens=context_tokens,
            ),
            "entries": entries,
            "ownership": {
                "policy": (
                    "brokered_target_group"
                    if representation == "q4_dynamic"
                    else "unbrokered_static"
                ),
                "owners": owners,
                "broker": broker,
            },
        },
        "routes": _route_evidence(route_trace),
        "health": {
            "active_routes": 0,
            "pins": 0,
            "loading": 0,
            "failed": 0,
            "integrity_errors": 0,
            "completion_fence_failures": 0,
            "global_device_synchronizations": 0,
        },
        "reset": {
            "entry_count": entry_count,
            "entry_offsets": [0] * entry_count,
            "entry_bytes": [0] * entry_count,
            "entries_closed": [True] * entry_count,
            "live_kv_tokens": 0,
            "broker_release": broker_release,
            "health": {
                "active_routes": 0,
                "pins": 0,
                "loading": 0,
                "failed": 0,
                "integrity_errors": 0,
                "completion_fence_failures": 0,
                "global_device_synchronizations": 0,
            },
            "passed": True,
        },
        "elapsed_seconds": 1.0,
    }


_PROMPT_FIXTURES: list[dict[str, object]] | None = None


def _prompt_fixtures() -> list[dict[str, object]]:
    global _PROMPT_FIXTURES
    if _PROMPT_FIXTURES is None:
        tokenizer = _CharacterTokenizer()
        _PROMPT_FIXTURES = [
            build_quality_prompt(tokenizer, context_tokens=context).to_dict()
            for context in QUALITY_CONTEXTS
        ]
    return copy.deepcopy(_PROMPT_FIXTURES)


def _artifact_evidence() -> dict[str, object]:
    sidecar = {
        "device": 1,
        "inode": 2,
        "size": 161_036_107_776,
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    resident_shards = [
        {
            "name": "model-00001-of-00001.safetensors",
            "device": 1,
            "inode": 5,
            "size": 4_952_354_048,
            "mtime_ns": 6,
            "ctime_ns": 7,
        }
    ]
    return {
        "schema": SCHEMA_ARTIFACT_ATTESTATION,
        "model_artifact_sha256": "1" * 64,
        "expert_manifest_sha256": "2" * 64,
        "artifact_pins_sha256": "3" * 64,
        "artifact_stat_sha256": canonical_sha256(
            {"sidecar": sidecar, "resident_shards": resident_shards}
        ),
        "sidecar_fingerprint": sidecar,
        "resident_payload_bytes": 123,
        "resident_payload_sha256": "6" * 64,
        "resident_shard_fingerprints": resident_shards,
        "payload_hash_verified": False,
        "payload_hash_io_mode": "not-run",
    }


def _valid_result() -> dict[str, object]:
    prompts = _prompt_fixtures()
    artifact = _artifact_evidence()
    hardware = _hardware_config(Path("/tmp/hy3-quality-fixture"))
    runtime_configs = {
        representation: build_quality_runtime_config(hardware, representation)
        for representation in ("bf16", "q4", "q4_dynamic")
    }
    from mtplx.expert_streaming_models import HY3_Q4

    plans = {
        representation: config.memory_plan(HY3_Q4)
        for representation, config in runtime_configs.items()
    }
    rows = {
        representation: [
            _healthy_row(
                representation,
                context,
                prompt,
                generated_token_ids=(
                    list(range(COMPLETION_RESERVE_TOKENS))
                    if context == QUALITY_CONTEXTS[0]
                    else [context] * COMPLETION_RESERVE_TOKENS
                ),
            )
            for context, prompt in zip(QUALITY_CONTEXTS, prompts, strict=True)
        ]
        for representation in ("bf16", "q4", "q4_dynamic")
    }
    routed_expert_attestation = _routed_expert_attestation()
    return {
        "schema": HY3_KV_QUALITY_SCHEMA,
        "identity": {
            "model_artifact_id": "pipenetwork/Hy3-4bit@revision-a",
            "model_artifact_sha256": "1" * 64,
            "expert_manifest_sha256": _TEST_MANIFEST_FILE_SHA256,
            "expert_manifest_content_sha256": _TEST_MANIFEST_CONTENT_SHA256,
            "artifact_pins_sha256": "3" * 64,
            "artifact_stat_sha256": artifact["artifact_stat_sha256"],
            "resident_payload_bytes": 123,
            "resident_payload_sha256": "6" * 64,
            "source_git_commit": "5" * 40,
            "payload_hash_verified": False,
            "tokenizer_identity": _test_tokenizer_identity(),
            "routed_expert_attestation": routed_expert_attestation,
        },
        "configuration": {
            "contexts": list(QUALITY_CONTEXTS),
            "completion_reserve_tokens": COMPLETION_RESERVE_TOKENS,
            "expert_cache_limit_bytes": EXPERT_CACHE_LIMIT_BYTES,
            "prefill_chunk_size": 2048,
            "representations": {
                "bf16": {
                    "kv_bytes_per_token": BF16_KV_BYTES_PER_TOKEN,
                    "expert_cache_limit_bytes": EXPERT_CACHE_LIMIT_BYTES,
                    "planned_persistent_slots": plans["bf16"].persistent_slots,
                    "persistent_cache_bytes": plans["bf16"].persistent_cache_bytes,
                    "attention_identity": {
                        "backend": "vllm_metal_paged",
                        "block_size_tokens": 16,
                        "sliding_window": 0,
                        "turboquant": False,
                        "dynamic_pages": False,
                        "page_slots": "ceil(context_tokens/16)",
                    },
                    "runtime_config": runtime_configs["bf16"].to_dict(),
                    "runtime_config_sha256": canonical_sha256(
                        runtime_configs["bf16"].to_dict()
                    ),
                },
                "q4": {
                    "kv_bytes_per_token": Q4_KV_BYTES_PER_TOKEN,
                    "expert_cache_limit_bytes": EXPERT_CACHE_LIMIT_BYTES,
                    "planned_persistent_slots": plans["q4"].persistent_slots,
                    "persistent_cache_bytes": plans["q4"].persistent_cache_bytes,
                    "attention_identity": {
                        "backend": "vllm_metal_paged",
                        "block_size_tokens": 16,
                        "sliding_window": 0,
                        "turboquant": False,
                        "dynamic_pages": False,
                        "page_slots": "ceil(context_tokens/16)",
                    },
                    "runtime_config": runtime_configs["q4"].to_dict(),
                    "runtime_config_sha256": canonical_sha256(
                        runtime_configs["q4"].to_dict()
                    ),
                },
                "q4_dynamic": {
                    "kv_bytes_per_token": Q4_KV_BYTES_PER_TOKEN,
                    "expert_cache_limit_bytes": EXPERT_CACHE_LIMIT_BYTES,
                    "planned_persistent_slots": plans["q4_dynamic"].persistent_slots,
                    "persistent_cache_bytes": plans[
                        "q4_dynamic"
                    ].persistent_cache_bytes,
                    "attention_identity": {
                        "backend": "vllm_metal_paged",
                        "block_size_tokens": 16,
                        "sliding_window": 0,
                        "turboquant": False,
                        "dynamic_pages": True,
                        "page_slots": "ceil(context_tokens/16)",
                    },
                    "runtime_config": runtime_configs["q4_dynamic"].to_dict(),
                    "runtime_config_sha256": canonical_sha256(
                        runtime_configs["q4_dynamic"].to_dict()
                    ),
                },
            },
        },
        "prompts": prompts,
        "arms": {
            representation: {
                "representation": representation,
                "model_artifact_id": "pipenetwork/Hy3-4bit@revision-a",
                "source_git_commit": "5" * 40,
                "expert_manifest_content_sha256": _TEST_MANIFEST_CONTENT_SHA256,
                "tokenizer_identity": _test_tokenizer_identity(),
                "routed_expert_attestation_sha256": (
                    routed_expert_attestation["binding_sha256"]
                ),
                "artifact_before": copy.deepcopy(artifact),
                "artifact_after": copy.deepcopy(artifact),
                "rows": rows[representation],
            }
            for representation in ("bf16", "q4", "q4_dynamic")
        },
    }


def test_result_validator_accepts_quality_and_geometry_evidence() -> None:
    result = _valid_result()

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is True
    assert acceptance["gates"]["all_retrieval_exact"] is True
    assert acceptance["gates"]["dynamic_page_geometry"] is True
    assert acceptance["gates"]["dynamic_target_owner_path"] is True
    assert acceptance["gates"]["short_control_token_parity"] is True
    assert acceptance["gates"]["static_dynamic_q4_token_parity"] is True
    assert acceptance["gates"]["cross_arm_retrieval_identity"] is True
    assert acceptance["rejection_reasons"] == []


def test_result_validator_rejects_text_edit_with_unchanged_generated_tokens() -> None:
    result = _valid_result()
    row = result["arms"]["q4"]["rows"][0]
    prompt = result["prompts"][0]
    edited_text = str(row["generated_text"]) + "\n"
    row["generated_text"] = edited_text
    row["generated_text_sha256"] = hashlib.sha256(edited_text.encode()).hexdigest()
    row["retrieval"] = score_retrieval_text(
        edited_text,
        expected_earliest=str(prompt["earliest_marker"]),
        expected_latest=str(prompt["latest_marker"]),
    )

    with pytest.raises(QualityGateError, match="decode binding"):
        validate_quality_result(result)


def test_result_validator_rejects_arbitrary_valid_expert_hash() -> None:
    result = _valid_result()
    expert_hashes = result["arms"]["q4"]["rows"][0]["routes"]["expert_hashes"]
    expert_hashes["1:2"] = hashlib.sha256(b"forged-expert-record").hexdigest()

    with pytest.raises(QualityGateError, match="manifest binding"):
        validate_quality_result(result)


def test_result_validator_rejects_zeroed_expert_hashes() -> None:
    result = _valid_result()
    expert_hashes = result["arms"]["q4_dynamic"]["rows"][0]["routes"]["expert_hashes"]
    for pair in tuple(expert_hashes):
        expert_hashes[pair] = "0" * 64

    with pytest.raises(QualityGateError, match="pinned manifest binding"):
        validate_quality_result(result)


def test_result_validator_rejects_unobserved_manifest_record_attestation() -> None:
    result = _valid_result()
    attestation = result["identity"]["routed_expert_attestation"]
    attestation["record_hashes"]["9:9"] = hashlib.sha256(
        b"unobserved-record"
    ).hexdigest()
    attestation["record_hashes_sha256"] = canonical_sha256(attestation["record_hashes"])
    attestation["routed_pair_count"] = len(attestation["record_hashes"])
    core = {key: value for key, value in attestation.items() if key != "binding_sha256"}
    attestation["binding_sha256"] = canonical_sha256(core)
    for arm in result["arms"].values():
        arm["routed_expert_attestation_sha256"] = attestation["binding_sha256"]

    with pytest.raises(QualityGateError, match="exact observed pairs"):
        validate_quality_result(result)


def test_result_validator_rejects_missing_routed_expert_hash() -> None:
    result = _valid_result()
    expert_hashes = result["arms"]["q4"]["rows"][0]["routes"]["expert_hashes"]
    del expert_hashes["1:3"]

    with pytest.raises(QualityGateError, match="exact routes"):
        validate_quality_result(result)


def test_result_validator_requires_dynamic_q4_hardware_arm() -> None:
    result = _valid_result()
    del result["arms"]["q4_dynamic"]

    with pytest.raises(QualityGateError, match="arms differ"):
        validate_quality_result(result)


def test_result_validator_rejects_non_target_dynamic_owner() -> None:
    result = _valid_result()
    owner = result["arms"]["q4_dynamic"]["rows"][0]["kv"]["ownership"]["owners"][0]
    owner["cache_id"] = "mtp:fixture:0"

    with pytest.raises(QualityGateError, match="not a target cache"):
        validate_quality_result(result)


def test_result_validator_rejects_preallocated_dynamic_context_maximum() -> None:
    result = _valid_result()
    initial = result["arms"]["q4_dynamic"]["rows"][0]["kv"]["initial"]
    initial["num_blocks"] = [8192] * initial["entry_count"]

    with pytest.raises(QualityGateError, match="one broker-owned 16-token page"):
        validate_quality_result(result)


def test_result_validator_rejects_dynamic_broker_ledger_drift() -> None:
    result = _valid_result()
    broker = result["arms"]["q4_dynamic"]["rows"][0]["kv"]["ownership"]["broker"]
    broker["owned_kv_physical_bytes"] -= 1

    with pytest.raises(QualityGateError, match="broker ownership"):
        validate_quality_result(result)


@pytest.mark.parametrize(
    ("representation", "field", "wrong_dtype"),
    (
        ("bf16", "key", "float16"),
        ("bf16", "value", "float32"),
        ("q4", "key", "int8"),
        ("q4", "value", "float16"),
        ("q4", "key_scale", "float32"),
        ("q4", "value_scale", "bfloat16"),
    ),
)
def test_result_validator_rejects_live_storage_dtype_drift(
    representation: str,
    field: str,
    wrong_dtype: str,
) -> None:
    result = _valid_result()
    storage = result["arms"][representation]["rows"][0]["kv"]["entries"][0][
        "storage_dtypes"
    ]
    storage[field] = wrong_dtype

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["kv_representation_path"] is False
    assert "kv_representation_path" in acceptance["rejection_reasons"]


def test_result_validator_requires_storage_dtype_evidence() -> None:
    result = _valid_result()
    del result["arms"]["q4"]["rows"][0]["kv"]["entries"][0]["storage_dtypes"]

    with pytest.raises(QualityGateError, match="storage_dtypes"):
        validate_quality_result(result)


@pytest.mark.parametrize(
    "field",
    (
        "dense_fallback_calls",
        "large_q_split_sdpa_fallback_calls",
        "partitioned_paged_calls",
        "q4_chunked_dequant_attention_calls",
        "paged_attention_bailouts_by_phase_reason",
    ),
)
def test_result_validator_requires_complete_attention_path_evidence(
    field: str,
) -> None:
    result = _valid_result()
    del result["arms"]["q4"]["rows"][0]["kv"]["entries"][0][field]

    with pytest.raises(QualityGateError, match=field):
        validate_quality_result(result)


@pytest.mark.parametrize(
    ("representation", "field", "value"),
    (
        ("q4", "q4_chunked_dequant_attention_calls", 0),
        ("q4", "large_q_split_sdpa_fallback_calls", 2),
        ("q4", "paged_attention_large_q_path", "large_q_split_sdpa_fallback"),
        ("q4", "partitioned_paged_calls", 1),
        ("bf16", "partitioned_paged_calls", 0),
        ("bf16", "paged_attention_large_q_path", "q4_chunked_dequant_attention"),
        ("bf16", "large_q_split_sdpa_fallback_calls", 1),
    ),
)
def test_result_validator_rejects_unexpected_attention_path_drift(
    representation: str,
    field: str,
    value: object,
) -> None:
    result = _valid_result()
    result["arms"][representation]["rows"][0]["kv"]["entries"][0][field] = value

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["attention_identity"] is False
    assert "attention_identity" in acceptance["rejection_reasons"]


@pytest.mark.parametrize(
    "field",
    (
        "updates",
        "cache_write_time_s",
        "attention_time_s",
        "q4_chunked_dequant_calls",
        "q4_chunked_dequant_time_s",
        "q4_chunked_dequant_tokens",
    ),
)
def test_result_validator_rejects_missing_q4_streaming_activity(
    field: str,
) -> None:
    result = _valid_result()
    result["arms"]["q4"]["rows"][0]["kv"]["entries"][0][field] = 0

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["kv_representation_path"] is False
    assert "kv_representation_path" in acceptance["rejection_reasons"]


@pytest.mark.parametrize(
    "field",
    (
        "kv_quant_attention_calls",
        "kv_quant_dequant_calls",
        "kv_quant_dequant_time_s",
        "kv_quant_dequant_tokens",
        "q4_chunked_dequant_calls",
        "q4_chunked_dequant_time_s",
        "q4_chunked_dequant_tokens",
    ),
)
def test_result_validator_rejects_quant_activity_in_bf16_control(field: str) -> None:
    result = _valid_result()
    result["arms"]["bf16"]["rows"][0]["kv"]["entries"][0][field] = 1

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["kv_representation_path"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result["arms"]["q4"]["rows"][0]["kv"].update(
                {"required_page_slots": 8192}
            ),
            "page slots",
        ),
        (
            lambda result: result["arms"]["q4"]["rows"][1]["retrieval"].update(
                {"passed": False, "stale_markers": ["early-4096"]}
            ),
            "retrieval",
        ),
        (
            lambda result: result["prompts"][0]["token_ids"].append(123),
            "prompt token",
        ),
        (
            lambda result: result["arms"]["q4"]["rows"][0].update(
                {"generated_token_sha256": "0" * 64}
            ),
            "generated token",
        ),
        (
            lambda result: result["arms"]["q4"]["rows"][0]["routes"].update(
                {"trace_sha256": "0" * 64}
            ),
            "route trace",
        ),
        (
            lambda result: result["arms"]["q4"]["rows"][0]["routes"][
                "expert_hashes"
            ].update({"1:2": "bad"}),
            "expert SHA-256",
        ),
        (
            lambda result: result["configuration"]["representations"]["q4"][
                "runtime_config"
            ].update({"expert_cache_limit_bytes": 31 * GIB}),
            "runtime config",
        ),
        (
            lambda result: result["arms"]["q4"]["rows"][0].update(
                {"elapsed_seconds": float("nan")}
            ),
            "elapsed",
        ),
        (
            lambda result: result["identity"].update({"resident_payload_bytes": 124}),
            "artifact attestation",
        ),
        (
            lambda result: result["arms"]["q4"]["artifact_before"].update(
                {"unknown": True}
            ),
            "attestation keys",
        ),
        (
            lambda result: result["arms"]["q4"]["artifact_before"].update(
                {"artifact_stat_sha256": "9" * 64}
            ),
            "artifact stat",
        ),
    ),
)
def test_result_validator_fails_closed_on_bad_evidence(mutation, message: str) -> None:
    result = _valid_result()
    mutation(result)

    with pytest.raises(QualityGateError, match=message):
        validate_quality_result(result)


def _set_generated_text(
    row: dict[str, object],
    text: str,
    *,
    forbidden_markers: tuple[str, ...] = (),
) -> None:
    row["generated_text"] = text
    row["generated_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    retrieval = row["retrieval"]
    assert isinstance(retrieval, dict)
    row["retrieval"] = score_retrieval_text(
        text,
        expected_earliest=str(retrieval["expected_earliest"]),
        expected_latest=str(retrieval["expected_latest"]),
        forbidden_markers=forbidden_markers,
    )
    row["decode_binding"] = _decode_binding(
        list(row["generated_token_ids"]),
        text,
    )


def _drift_artifact(result: dict[str, object]) -> None:
    after = result["arms"]["q4"]["artifact_after"]
    after["sidecar_fingerprint"]["ctime_ns"] += 1
    after["artifact_stat_sha256"] = canonical_sha256(
        {
            "sidecar": after["sidecar_fingerprint"],
            "resident_shards": after["resident_shard_fingerprints"],
        }
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result: result["arms"]["q4"]["rows"][0]["kv"]["entries"][0].update(
            {"dense_fallback_calls": 1}
        ),
        lambda result: result["arms"]["q4"]["rows"][0]["reset"].update(
            {"entry_bytes": [1], "passed": False}
        ),
        lambda result: result["arms"]["q4"]["rows"][0]["health"].update(
            {"global_device_synchronizations": 1}
        ),
        lambda result: result["arms"]["q4"]["rows"][0]["reset"]["health"].update(
            {"completion_fence_failures": 1}
        ),
        _drift_artifact,
        lambda result: result["configuration"]["representations"]["q4"][
            "attention_identity"
        ].update({"sliding_window": 1}),
    ),
)
def test_complete_negative_evidence_returns_a_valid_rejection(mutation) -> None:
    result = _valid_result()
    mutation(result)

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert any(value is False for value in acceptance["gates"].values())


def test_exact_retrieval_failure_is_a_valid_rejection() -> None:
    result = _valid_result()
    row = result["arms"]["q4"]["rows"][1]
    previous = result["prompts"][0]
    _set_generated_text(
        row,
        "EARLIEST=wrong\nLATEST=also-wrong\n",
        forbidden_markers=(
            str(previous["earliest_marker"]),
            str(previous["latest_marker"]),
        ),
    )

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["all_retrieval_exact"] is False
    assert "all_retrieval_exact" in acceptance["rejection_reasons"]


def test_result_validator_derives_retrieval_marker_from_probe_spec() -> None:
    result = _valid_result()
    prompt = result["prompts"][0]
    original = str(prompt["earliest_marker"])
    forged = "E46-004096-EARLIEST-forged00000000000000"
    prompt["earliest_marker"] = forged
    prompt["rendered_prompt"] = str(prompt["rendered_prompt"]).replace(
        original,
        forged,
    )
    for representation in ("bf16", "q4"):
        row = result["arms"][representation]["rows"][0]
        latest = str(result["prompts"][0]["latest_marker"])
        retrieval = row["retrieval"]
        assert isinstance(retrieval, dict)
        retrieval["expected_earliest"] = forged
        _set_generated_text(row, f"EARLIEST={forged}\nLATEST={latest}\n")

    with pytest.raises(QualityGateError, match="probe marker"):
        validate_quality_result(result)


def test_short_token_parity_failure_is_a_valid_rejection() -> None:
    result = _valid_result()
    row = result["arms"]["q4"]["rows"][0]
    token_ids = list(row["generated_token_ids"])
    token_ids[0] += 1
    row["generated_token_ids"] = token_ids
    row["generated_token_sha256"] = hashlib.sha256(
        b"".join(int(token).to_bytes(8, "little") for token in token_ids)
    ).hexdigest()
    row["decode_binding"] = _decode_binding(token_ids, str(row["generated_text"]))

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["short_control_token_parity"] is False
    assert "short_control_token_parity" in acceptance["rejection_reasons"]


def test_static_dynamic_q4_token_drift_is_a_valid_rejection() -> None:
    result = _valid_result()
    row = result["arms"]["q4_dynamic"]["rows"][1]
    token_ids = list(row["generated_token_ids"])
    token_ids[0] += 1
    row["generated_token_ids"] = token_ids
    row["generated_token_sha256"] = hashlib.sha256(
        b"".join(int(token).to_bytes(8, "little") for token in token_ids)
    ).hexdigest()
    row["decode_binding"] = _decode_binding(token_ids, str(row["generated_text"]))

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["static_dynamic_q4_token_parity"] is False
    assert "static_dynamic_q4_token_parity" in acceptance["rejection_reasons"]


def test_dynamic_q4_retrieval_drift_breaks_cross_arm_identity() -> None:
    result = _valid_result()
    row = result["arms"]["q4_dynamic"]["rows"][2]
    previous_markers = tuple(
        str(result["prompts"][index][edge])
        for index in range(2)
        for edge in ("earliest_marker", "latest_marker")
    )
    _set_generated_text(
        row,
        "EARLIEST=wrong\nLATEST=also-wrong\n",
        forbidden_markers=previous_markers,
    )

    acceptance = validate_quality_result(result)

    assert acceptance["passed"] is False
    assert acceptance["gates"]["all_retrieval_exact"] is False
    assert acceptance["gates"]["cross_arm_retrieval_identity"] is False


def test_integrated_runner_executes_all_three_quality_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    fixture = _valid_result()
    exact_prompts = [
        hy3_kv_quality.QualityPrompt(
            context_tokens=int(prompt["context_tokens"]),
            prompt_tokens=int(prompt["prompt_tokens"]),
            completion_reserve_tokens=int(prompt["completion_reserve_tokens"]),
            token_ids=tuple(prompt["token_ids"]),
            rendered_prompt=str(prompt["rendered_prompt"]),
            prompt_sha256=str(prompt["prompt_sha256"]),
            earliest_marker=str(prompt["earliest_marker"]),
            latest_marker=str(prompt["latest_marker"]),
            earliest_token_index=int(prompt["earliest_token_index"]),
            latest_token_index=int(prompt["latest_token_index"]),
        )
        for prompt in fixture["prompts"]
    ]
    calls: list[tuple[str, bool]] = []

    def fake_representation_runner(*, hardware_config, representation, prompts):
        del hardware_config
        calls.append((representation, prompts is None))
        returned_prompts = exact_prompts if prompts is None else list(prompts)
        return (
            copy.deepcopy(fixture["arms"][representation]),
            returned_prompts,
            copy.deepcopy(fixture["configuration"]["representations"][representation]),
        )

    monkeypatch.setattr(
        hy3_kv_quality,
        "_run_quality_representation",
        fake_representation_runner,
    )

    result = hy3_kv_quality.run_hy3_kv_quality(_hardware_config(tmp_path))

    assert calls == [("bf16", True), ("q4", False), ("q4_dynamic", False)]
    assert set(result["arms"]) == {"bf16", "q4", "q4_dynamic"}
    assert result["acceptance"]["passed"] is True


def test_cli_emits_exactly_one_json_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    config_path = tmp_path / "hooks.json"
    config_path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "model_root": str(tmp_path / "model"),
                "manifest": str(tmp_path / "manifest.json"),
                "model_artifact_id": "pipenetwork/Hy3-4bit@revision-a",
                "artifact_pins": _pins().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    result = _valid_result()

    def noisy_runtime(_config):
        print("captured runtime diagnostic")
        os.write(1, b"captured native diagnostic\n")
        ctypes.CDLL(None).printf(b"captured libc diagnostic\n")
        return result

    monkeypatch.setattr(hy3_kv_quality, "run_hy3_kv_quality", noisy_runtime)

    assert hy3_kv_quality.main(["--hooks-config", str(config_path)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {**result, "acceptance": result["acceptance"]}
    diagnostics = result["diagnostics"]
    assert diagnostics["captured_runtime_stdout_bytes"] > 0
    assert len(diagnostics["captured_runtime_stdout_sha256"]) == 64
    captured_tail = diagnostics["captured_runtime_stdout_tail"]
    assert "captured runtime diagnostic\n" in captured_tail
    assert "captured native diagnostic\n" in captured_tail
    assert "captured libc diagnostic\n" in captured_tail
    assert "captured_runtime_stdout" not in diagnostics
    assert output.out.count("\n") == 1


def test_cli_prints_complete_rejection_json_and_returns_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    config_path = tmp_path / "hooks.json"
    config_path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "model_root": str(tmp_path / "model"),
                "manifest": str(tmp_path / "manifest.json"),
                "model_artifact_id": "pipenetwork/Hy3-4bit@revision-a",
                "artifact_pins": _pins().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    result = _valid_result()
    result["arms"]["q4"]["rows"][0]["health"]["global_device_synchronizations"] = 1
    monkeypatch.setattr(
        hy3_kv_quality,
        "run_hy3_kv_quality",
        lambda _config: result,
    )

    assert hy3_kv_quality.main(["--hooks-config", str(config_path)]) == 2
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["acceptance"]["passed"] is False
    assert payload["acceptance"]["gates"]["runtime_health"] is False
    assert output.out.count("\n") == 1


def test_cli_writes_the_same_exclusive_evidence_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mtplx.benchmarks import hy3_kv_quality

    config_path = tmp_path / "hooks.json"
    output_path = tmp_path / "quality.json"
    config_path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "model_root": str(tmp_path / "model"),
                "manifest": str(tmp_path / "manifest.json"),
                "model_artifact_id": "pipenetwork/Hy3-4bit@revision-a",
                "artifact_pins": _pins().to_dict(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hy3_kv_quality,
        "run_hy3_kv_quality",
        lambda _config: _valid_result(),
    )

    assert (
        hy3_kv_quality.main(
            [
                "--hooks-config",
                str(config_path),
                "--output-json",
                str(output_path),
            ]
        )
        == 0
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    assert json.loads(output_path.read_text(encoding="utf-8")) == stdout_payload
    with pytest.raises(FileExistsError):
        hy3_kv_quality.main(
            [
                "--hooks-config",
                str(config_path),
                "--output-json",
                str(output_path),
            ]
        )
