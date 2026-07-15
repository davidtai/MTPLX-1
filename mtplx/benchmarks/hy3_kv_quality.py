"""BF16-control versus Q4-KV retrieval-quality gate for Hy3.

The issue #46 memory campaign compares static and brokered-dynamic Q4 policies.
This module runs the integrated BF16, static-Q4, and dynamic-Q4 quality lane
needed to prove that neither quantization nor dynamic ownership destroys
long-context retrieval.  It keeps one exact 32 GiB expert cap across all arms
and derives paged-KV capacity independently for every context row.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from mtplx.benchmarks.hy3_dynamic_memory_hardware import (
    Hy3HardwareConfig,
    _EnvironmentLease,
    _attest_config_artifact,
    _decode_tokens,
    _forward_prefill,
    _require_loaded_manifest_identity,
    _source_commit,
)
from mtplx.benchmarks.hy3_dynamic_memory_artifacts import (
    SCHEMA_ARTIFACT_ATTESTATION,
)
from mtplx.benchmarks.runners.hy3_dynamic_memory import canonical_sha256
from mtplx.memory_broker import (
    HY3_Q4_KV_HEAD_DIM,
    HY3_Q4_KV_HEADS,
    HY3_Q4_KV_LAYERS,
)
from mtplx.runtime_options import HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV


GIB = 1024**3
HY3_KV_QUALITY_SCHEMA = "mtplx-hy3-kv-quality-v1"
QUALITY_CONTEXTS = (4_096, 32_768, 65_536, 131_072)
COMPLETION_RESERVE_TOKENS = 64
EXPERT_CACHE_LIMIT_BYTES = 32 * GIB
KV_BLOCK_SIZE_TOKENS = 16
BF16_KV_BYTES_PER_TOKEN = 327_680
Q4_KV_BYTES_PER_TOKEN = 84_480
Q4_STREAMING_DEQUANT_SOURCE_BYTES_PER_TOKEN_PER_LAYER = (
    2 * HY3_Q4_KV_HEADS * HY3_Q4_KV_HEAD_DIM * 2
)
Q4_STREAMING_DEQUANT_COMPUTE_BYTES_PER_TOKEN_PER_LAYER = (
    2 * HY3_Q4_KV_HEADS * HY3_Q4_KV_HEAD_DIM * 4
)
Q4_STREAMING_DEQUANT_BYTES_PER_TOKEN_PER_LAYER = (
    Q4_STREAMING_DEQUANT_SOURCE_BYTES_PER_TOKEN_PER_LAYER
    + Q4_STREAMING_DEQUANT_COMPUTE_BYTES_PER_TOKEN_PER_LAYER
)
_PROMPT_HEAD_GUARD_TOKENS = 256
_PROMPT_TAIL_GUARD_TOKENS = 512
_HEALTH_FIELDS = (
    "active_routes",
    "pins",
    "loading",
    "failed",
    "integrity_errors",
    "completion_fence_failures",
    "global_device_synchronizations",
)
_REPRESENTATIONS = ("bf16", "q4", "q4_dynamic")
_Q4_REPRESENTATIONS = frozenset({"q4", "q4_dynamic"})
_STORAGE_DTYPES: dict[str, dict[str, str | None]] = {
    "bf16": {
        "key": "bfloat16",
        "value": "bfloat16",
        "key_scale": None,
        "value_scale": None,
    },
    "q4": {
        "key": "uint8",
        "value": "uint8",
        "key_scale": "float16",
        "value_scale": "float16",
    },
    "q4_dynamic": {
        "key": "uint8",
        "value": "uint8",
        "key_scale": "float16",
        "value_scale": "float16",
    },
}
ATTENTION_IDENTITY: dict[str, object] = {
    "backend": "vllm_metal_paged",
    "block_size_tokens": KV_BLOCK_SIZE_TOKENS,
    "sliding_window": 0,
    "turboquant": False,
    "dynamic_pages": False,
    "page_slots": "ceil(context_tokens/16)",
}
TOKENIZER_IDENTITY_SCHEMA = "mtplx-tokenizer-identity-v1"
TOKENIZER_IDENTITY_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
TOKENIZER_OFFLINE_SCOPE = "structural-only-tokenizer-files-required-for-redecode"
DECODE_VERIFICATION_SCOPE = (
    "producer-decoded;offline-structural-tokenizer-files-required"
)
ROUTE_VERIFICATION_SCOPE = "producer-verified-live-manifest;offline-structural"
ROUTED_EXPERT_ATTESTATION_SCHEMA = "mtplx-routed-expert-attestation-v1"


class QualityGateError(ValueError):
    """Raised when quality evidence is missing, ambiguous, or contradictory."""


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualityGateError(f"{field} must be an integer >= {minimum}")
    return value


def _nonnegative_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise QualityGateError(f"{field} must be a finite number >= 0")
    return float(value)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QualityGateError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise QualityGateError(f"{field} must be an array")
    return value


def _representation(value: object) -> str:
    if not isinstance(value, str) or value not in _REPRESENTATIONS:
        raise QualityGateError(
            "KV representation must be 'bf16', 'q4', or 'q4_dynamic'"
        )
    return value


def _hex_digest(value: object, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        expected = " or ".join(str(length) for length in lengths)
        raise QualityGateError(
            f"{field} must be {expected} lowercase hexadecimal characters"
        )
    return value


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while payload := handle.read(1024 * 1024):
                size += len(payload)
                digest.update(payload)
    except OSError as exc:
        raise QualityGateError(f"cannot hash tokenizer artifact {path}: {exc}") from exc
    if size <= 0:
        raise QualityGateError(f"tokenizer artifact is empty: {path}")
    return size, digest.hexdigest()


def _attest_tokenizer_identity(tokenizer: Any, model_root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for name in TOKENIZER_IDENTITY_FILES:
        size, digest = _file_sha256(model_root / name)
        files.append({"name": name, "size": size, "sha256": digest})
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        try:
            vocab_size = len(tokenizer)
        except (TypeError, AttributeError):
            vocab_size = 0
    vocab_size = _exact_int(vocab_size, field="tokenizer.vocab_size", minimum=1)
    core: dict[str, object] = {
        "schema": TOKENIZER_IDENTITY_SCHEMA,
        "files": files,
        "runtime": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab_size": vocab_size,
        },
        "offline_decode_verification": TOKENIZER_OFFLINE_SCOPE,
    }
    return {**core, "identity_sha256": canonical_sha256(core)}


def _validate_tokenizer_identity(value: object, *, field: str) -> dict[str, object]:
    identity = dict(_mapping(value, field=field))
    expected_keys = {
        "schema",
        "files",
        "runtime",
        "offline_decode_verification",
        "identity_sha256",
    }
    if set(identity) != expected_keys:
        raise QualityGateError(f"{field} keys differ")
    if identity.get("schema") != TOKENIZER_IDENTITY_SCHEMA:
        raise QualityGateError(f"{field} schema differs")
    files = _sequence(identity.get("files"), field=f"{field}.files")
    observed_names: list[str] = []
    normalized_files: list[dict[str, object]] = []
    for index, value in enumerate(files):
        item = dict(_mapping(value, field=f"{field}.files[{index}]"))
        if set(item) != {"name", "size", "sha256"}:
            raise QualityGateError(f"{field}.files[{index}] keys differ")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise QualityGateError(f"{field}.files[{index}].name is invalid")
        observed_names.append(name)
        normalized_files.append(
            {
                "name": name,
                "size": _exact_int(
                    item.get("size"), field=f"{field}.files[{index}].size", minimum=1
                ),
                "sha256": _hex_digest(
                    item.get("sha256"), field=f"{field}.files[{index}].sha256"
                ),
            }
        )
    if tuple(observed_names) != TOKENIZER_IDENTITY_FILES:
        raise QualityGateError(f"{field} tokenizer files differ")
    runtime = dict(_mapping(identity.get("runtime"), field=f"{field}.runtime"))
    if set(runtime) != {"class", "vocab_size"}:
        raise QualityGateError(f"{field}.runtime keys differ")
    runtime_class = runtime.get("class")
    if not isinstance(runtime_class, str) or not runtime_class:
        raise QualityGateError(f"{field}.runtime.class is invalid")
    normalized_runtime = {
        "class": runtime_class,
        "vocab_size": _exact_int(
            runtime.get("vocab_size"), field=f"{field}.runtime.vocab_size", minimum=1
        ),
    }
    if identity.get("offline_decode_verification") != TOKENIZER_OFFLINE_SCOPE:
        raise QualityGateError(f"{field} offline verification scope differs")
    core = {
        "schema": TOKENIZER_IDENTITY_SCHEMA,
        "files": normalized_files,
        "runtime": normalized_runtime,
        "offline_decode_verification": TOKENIZER_OFFLINE_SCOPE,
    }
    if _hex_digest(
        identity.get("identity_sha256"), field=f"{field}.identity_sha256"
    ) != canonical_sha256(core):
        raise QualityGateError(f"{field} SHA-256 contradicts tokenizer identity")
    return {**core, "identity_sha256": identity["identity_sha256"]}


def page_slots_for_context(
    context_tokens: int,
    *,
    block_size_tokens: int = KV_BLOCK_SIZE_TOKENS,
) -> int:
    """Return physical page slots for this row, never a campaign-wide maximum."""

    tokens = _exact_int(context_tokens, field="context_tokens", minimum=1)
    block = _exact_int(
        block_size_tokens,
        field="block_size_tokens",
        minimum=1,
    )
    return math.ceil(tokens / block)


def _bytes_per_token(representation: str) -> int:
    return (
        Q4_KV_BYTES_PER_TOKEN
        if _representation(representation) in _Q4_REPRESENTATIONS
        else BF16_KV_BYTES_PER_TOKEN
    )


def _attention_identity(representation: str) -> dict[str, object]:
    mode = _representation(representation)
    return {
        **ATTENTION_IDENTITY,
        "dynamic_pages": mode == "q4_dynamic",
    }


def quality_environment(
    representation: str,
    *,
    context_tokens: int,
    prefill_chunk_size: int = 2048,
) -> dict[str, str]:
    """Build exact paged-attention environment for one quality row."""

    mode = _representation(representation)
    pages = 1 if mode == "q4_dynamic" else page_slots_for_context(context_tokens)
    chunk = _exact_int(prefill_chunk_size, field="prefill_chunk_size", minimum=1)
    quant = "q4" if mode in _Q4_REPRESENTATIONS else "off"
    environment = {
        **HY3_Q4_EXACT_PAGED_ATTENTION_RUNTIME_ENV,
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS": str(pages),
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": quant,
        "MTPLX_PAGED_KV_QUANT": quant,
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT": "0",
        "MTPLX_DYNAMIC_PAGED_KV": "1" if mode == "q4_dynamic" else "0",
        "MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS": "1",
        "MTPLX_DYNAMIC_PAGED_KV_TOKENS": "0",
        "MTPLX_DYNAMIC_PAGED_KV_PREVIOUS_HIGH_WATER": "0",
        "MTPLX_DYNAMIC_PAGED_KV_MARGIN": "0",
        "MTPLX_PREFILL_CHUNK_SIZE": str(chunk),
        "MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK": "1",
        "MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS": "1",
        "MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK": "0",
        "MTPLX_ALLOW_PAGED_ACTIVE_ARRAY_SNAPSHOT": "0",
    }
    if mode == "q4_dynamic":
        environment["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] = "0"
    return environment


def _token_list(value: object, *, field: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = _sequence(value, field=field)
    result: list[int] = []
    for index, token in enumerate(values):
        if isinstance(token, Sequence) and not isinstance(
            token, (str, bytes, bytearray)
        ):
            if len(values) != 1:
                raise QualityGateError(f"{field} must be one-dimensional")
            return _token_list(token, field=field)
        result.append(_exact_int(token, field=f"{field}[{index}]"))
    return result


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    return _token_list(encoded, field="encoded text")


def _decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    values = [int(token) for token in token_ids]
    try:
        decoded = tokenizer.decode(values, skip_special_tokens=True)
    except TypeError:
        decoded = tokenizer.decode(values)
    if not isinstance(decoded, str):
        raise QualityGateError("tokenizer.decode did not return text")
    return decoded


def _decode_binding(
    *,
    tokenizer_identity: Mapping[str, object],
    generated_token_ids: Sequence[int],
    generated_text: str,
) -> dict[str, object]:
    core: dict[str, object] = {
        "tokenizer_identity_sha256": _hex_digest(
            tokenizer_identity.get("identity_sha256"),
            field="tokenizer identity SHA-256",
        ),
        "generated_token_sha256": _prompt_digest(generated_token_ids),
        "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
        "producer_decode_match": True,
        "verification_scope": DECODE_VERIFICATION_SCOPE,
    }
    return {**core, "binding_sha256": canonical_sha256(core)}


def _validate_decode_binding(
    value: object,
    *,
    field: str,
    tokenizer_identity_sha256: str,
    generated_token_sha256: str,
    generated_text_sha256: str,
) -> bool:
    binding = dict(_mapping(value, field=field))
    expected_keys = {
        "tokenizer_identity_sha256",
        "generated_token_sha256",
        "generated_text_sha256",
        "producer_decode_match",
        "verification_scope",
        "binding_sha256",
    }
    if set(binding) != expected_keys:
        raise QualityGateError(f"{field} decode binding keys differ")
    core = {name: binding[name] for name in expected_keys - {"binding_sha256"}}
    if (
        binding.get("tokenizer_identity_sha256") != tokenizer_identity_sha256
        or binding.get("generated_token_sha256") != generated_token_sha256
        or binding.get("generated_text_sha256") != generated_text_sha256
        or binding.get("producer_decode_match") is not True
        or binding.get("verification_scope") != DECODE_VERIFICATION_SCOPE
    ):
        raise QualityGateError(f"{field} decode binding contradicts row evidence")
    if _hex_digest(
        binding.get("binding_sha256"), field=f"{field}.binding_sha256"
    ) != canonical_sha256(core):
        raise QualityGateError(f"{field} decode binding SHA-256 differs")
    return True


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs: dict[str, object] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    return _token_list(encoded, field="chat-template tokens")


def _prompt_digest(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _marker(context_tokens: int, edge: str) -> str:
    digest = hashlib.sha256(
        f"issue-46:{context_tokens}:{edge}:hy3-kv-quality-v1".encode()
    ).hexdigest()[:24]
    return f"E46-{context_tokens:06d}-{edge.upper()}-{digest}"


def _distractor(context_tokens: int, index: int) -> str:
    digest = hashlib.sha256(
        f"issue-46:{context_tokens}:distractor:{index}".encode()
    ).hexdigest()
    return f"D{index:06d}={digest}\n"


@dataclass(frozen=True)
class QualityPrompt:
    context_tokens: int
    prompt_tokens: int
    completion_reserve_tokens: int
    token_ids: tuple[int, ...]
    rendered_prompt: str
    prompt_sha256: str
    earliest_marker: str
    latest_marker: str
    earliest_token_index: int
    latest_token_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "context_tokens": self.context_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_reserve_tokens": self.completion_reserve_tokens,
            "token_ids": list(self.token_ids),
            "rendered_prompt": self.rendered_prompt,
            "prompt_sha256": self.prompt_sha256,
            "earliest_marker": self.earliest_marker,
            "latest_marker": self.latest_marker,
            "earliest_token_index": self.earliest_token_index,
            "latest_token_index": self.latest_token_index,
        }


def build_quality_prompt(
    tokenizer: Any,
    *,
    context_tokens: int,
    completion_reserve_tokens: int = COMPLETION_RESERVE_TOKENS,
) -> QualityPrompt:
    """Build an exact-length prompt with protected unique head/tail markers."""

    context = _exact_int(context_tokens, field="context_tokens", minimum=1)
    reserve = _exact_int(
        completion_reserve_tokens,
        field="completion_reserve_tokens",
        minimum=1,
    )
    target = context - reserve
    if target <= _PROMPT_HEAD_GUARD_TOKENS + _PROMPT_TAIL_GUARD_TOKENS:
        raise QualityGateError("context is too small for protected retrieval markers")
    earliest = _marker(context, "earliest")
    latest = _marker(context, "latest")
    record_count = 64
    full_ids: list[int] | None = None
    while record_count <= 1_048_576:
        middle = "".join(_distractor(context, index) for index in range(record_count))
        user = (
            f"EARLIEST={earliest}\n"
            "The records below are distractors. Preserve the two exact markers.\n"
            f"{middle}"
            f"LATEST={latest}\n"
            "Reply with exactly two lines: EARLIEST=<value> then LATEST=<value>."
        )
        full_ids = _apply_chat_template(
            tokenizer,
            [
                {
                    "role": "system",
                    "content": "Retrieve both boundary markers exactly; do not infer.",
                },
                {"role": "user", "content": user},
            ],
        )
        if len(full_ids) >= target:
            break
        record_count *= 2
    if full_ids is None or len(full_ids) < target:
        raise QualityGateError("could not grow retrieval prompt to target length")
    excess = len(full_ids) - target
    if excess:
        removable = len(full_ids) - (
            _PROMPT_HEAD_GUARD_TOKENS + _PROMPT_TAIL_GUARD_TOKENS
        )
        if excess > removable:
            raise QualityGateError("prompt overshoot would remove a marker guard")
        removal_start = _PROMPT_HEAD_GUARD_TOKENS + (removable - excess) // 2
        token_ids = full_ids[:removal_start] + full_ids[removal_start + excess :]
    else:
        token_ids = full_ids
    if len(token_ids) != target:
        raise QualityGateError("prompt builder did not produce the exact token count")
    rendered = _decode_text(tokenizer, token_ids)
    if rendered.count(earliest) != 1 or rendered.count(latest) != 1:
        raise QualityGateError("exact prompt trimming damaged a retrieval marker")
    earliest_character = rendered.index(earliest)
    latest_character = rendered.index(latest)
    earliest_token = len(_encode_text(tokenizer, rendered[:earliest_character]))
    latest_token = len(_encode_text(tokenizer, rendered[:latest_character]))
    if earliest_token >= _PROMPT_HEAD_GUARD_TOKENS:
        raise QualityGateError("earliest marker escaped its protected head window")
    if latest_token < target - _PROMPT_TAIL_GUARD_TOKENS:
        raise QualityGateError("latest marker escaped its protected tail window")
    return QualityPrompt(
        context_tokens=context,
        prompt_tokens=target,
        completion_reserve_tokens=reserve,
        token_ids=tuple(token_ids),
        rendered_prompt=rendered,
        prompt_sha256=_prompt_digest(token_ids),
        earliest_marker=earliest,
        latest_marker=latest,
        earliest_token_index=earliest_token,
        latest_token_index=latest_token,
    )


_EARLIEST_LINE = re.compile(r"^\s*EARLIEST\s*[:=]\s*(\S+)\s*$", re.MULTILINE)
_LATEST_LINE = re.compile(r"^\s*LATEST\s*[:=]\s*(\S+)\s*$", re.MULTILINE)


def score_retrieval_text(
    text: str,
    *,
    expected_earliest: str,
    expected_latest: str,
    forbidden_markers: Sequence[str] = (),
) -> dict[str, object]:
    """Score decoded text and reject any marker retained from an earlier row."""

    if not isinstance(text, str):
        raise QualityGateError("generated retrieval output must be text")
    if not expected_earliest or not expected_latest:
        raise QualityGateError("expected retrieval markers must be nonempty")
    observed_earliest = _EARLIEST_LINE.findall(text)
    observed_latest = _LATEST_LINE.findall(text)
    stale = sorted(
        {
            marker
            for marker in forbidden_markers
            if isinstance(marker, str) and marker and marker in text
        }
    )
    earliest_exact = observed_earliest == [expected_earliest]
    latest_exact = observed_latest == [expected_latest]
    return {
        "expected_earliest": expected_earliest,
        "expected_latest": expected_latest,
        "observed_earliest": observed_earliest,
        "observed_latest": observed_latest,
        "earliest_exact": earliest_exact,
        "latest_exact": latest_exact,
        "stale_markers": stale,
        "passed": earliest_exact and latest_exact and not stale,
    }


def build_quality_runtime_config(
    hardware_config: Hy3HardwareConfig,
    representation: str,
) -> Any:
    """Use identical expert ownership while varying only KV representation."""

    from mtplx.expert_runtime import ExpertStreamingConfig

    mode = _representation(representation)
    if hardware_config.runtime_reserve_bytes != 8 * GIB:
        raise QualityGateError("quality lane requires the 8 GiB runtime reserve")
    if hardware_config.allocator_headroom_bytes != GIB:
        raise QualityGateError("quality lane requires 1 GiB allocator headroom")
    return ExpertStreamingConfig(
        model_key="hy3-q4",
        memory_limit_bytes=hardware_config.memory_limit_bytes,
        max_live_kv_tokens=max(QUALITY_CONTEXTS),
        kv_bytes_per_token_override=_bytes_per_token(mode),
        runtime_reserve_bytes=hardware_config.runtime_reserve_bytes,
        allocator_headroom_bytes=hardware_config.allocator_headroom_bytes,
        expert_cache_limit_bytes=EXPERT_CACHE_LIMIT_BYTES,
        transient_slots=hardware_config.transient_slots,
        cache_policy="lru",
        cache_scope="global",
        slot_layout="direct-slots",
        dynamic_expert_cache=mode == "q4_dynamic",
        verify_sidecar_hash_at_open=False,
        verify_record_hashes=True,
        resource_telemetry=True,
        trace_routes=True,
    )


def _sum_nested_counts(value: object, *, field: str) -> int:
    mapping = _mapping(value, field=field)
    total = 0
    for name, count in mapping.items():
        total += _exact_int(count, field=f"{field}.{name}")
    return total


def _exact_phase_counts(value: object, *, field: str) -> dict[str, int]:
    mapping = _mapping(value, field=field)
    result: dict[str, int] = {}
    for phase, count in mapping.items():
        if not isinstance(phase, str) or not phase:
            raise QualityGateError(f"{field} contains an invalid phase name")
        result[phase] = _exact_int(count, field=f"{field}.{phase}")
    return result


def _expected_q4_phase_evidence(
    *,
    context_tokens: int,
    prefill_chunk_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    prompt_tokens = context_tokens - COMPLETION_RESERVE_TOKENS
    body_tokens = prompt_tokens - 1
    prefill_offsets = [
        min(start + prefill_chunk_size, body_tokens)
        for start in range(0, body_tokens, prefill_chunk_size)
    ]
    prefill_offsets.append(prompt_tokens)
    calls = {
        "prefill": len(prefill_offsets),
        "ar_decode": COMPLETION_RESERVE_TOKENS,
    }
    coverage = {
        "prefill": sum(prefill_offsets),
        "ar_decode": (
            COMPLETION_RESERVE_TOKENS * prompt_tokens
            + COMPLETION_RESERVE_TOKENS * (COMPLETION_RESERVE_TOKENS + 1) // 2
        ),
    }
    return calls, coverage


def _expected_q4_realized_evidence(
    *,
    context_tokens: int,
    prefill_chunk_size: int,
    query_chunk_size: int,
    kv_chunk_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Derive physical chunk evaluations from the exact quality call sequence."""

    prompt_tokens = context_tokens - COMPLETION_RESERVE_TOKENS
    body_tokens = prompt_tokens - 1
    calls = {"prefill": 0, "ar_decode": 0}
    tokens = {"prefill": 0, "ar_decode": 0}

    def add_call(*, phase: str, offset: int, q_len: int) -> None:
        cached_prefix = offset - q_len
        for q_start in range(0, q_len, query_chunk_size):
            q_end = min(q_len, q_start + query_chunk_size)
            max_key = min(offset, cached_prefix + q_end)
            calls[phase] += math.ceil(max_key / kv_chunk_size)
            tokens[phase] += max_key

    offset = 0
    for start in range(0, body_tokens, prefill_chunk_size):
        q_len = min(prefill_chunk_size, body_tokens - start)
        offset += q_len
        add_call(phase="prefill", offset=offset, q_len=q_len)
    offset += 1
    add_call(phase="prefill", offset=offset, q_len=1)
    for _ in range(COMPLETION_RESERVE_TOKENS):
        offset += 1
        add_call(phase="ar_decode", offset=offset, q_len=1)
    return calls, tokens


def _validate_prompt(value: object, *, context_tokens: int) -> Mapping[str, object]:
    prompt = _mapping(value, field=f"prompt[{context_tokens}]")
    prompt_context = _exact_int(
        prompt.get("context_tokens"),
        field=f"prompt[{context_tokens}].context_tokens",
        minimum=1,
    )
    if prompt_context != context_tokens:
        raise QualityGateError("prompt contexts are not the exact quality matrix")
    expected_prompt_tokens = context_tokens - COMPLETION_RESERVE_TOKENS
    if (
        _exact_int(
            prompt.get("prompt_tokens"),
            field=f"prompt[{context_tokens}].prompt_tokens",
            minimum=1,
        )
        != expected_prompt_tokens
    ):
        raise QualityGateError("prompt token count differs from its context row")
    if (
        _exact_int(
            prompt.get("completion_reserve_tokens"),
            field=f"prompt[{context_tokens}].completion_reserve_tokens",
            minimum=1,
        )
        != COMPLETION_RESERVE_TOKENS
    ):
        raise QualityGateError("prompt completion reserve is not exact")
    token_ids = _token_list(
        prompt.get("token_ids"),
        field=f"prompt[{context_tokens}].token_ids",
    )
    if len(token_ids) != expected_prompt_tokens:
        raise QualityGateError("prompt token_ids length is not exact")
    expected_digest = _prompt_digest(token_ids)
    observed_digest = _hex_digest(
        prompt.get("prompt_sha256"),
        field=f"prompt[{context_tokens}].prompt_sha256",
    )
    if observed_digest != expected_digest:
        raise QualityGateError("prompt token SHA-256 differs from token_ids")
    earliest = prompt.get("earliest_marker")
    latest = prompt.get("latest_marker")
    rendered = prompt.get("rendered_prompt")
    if not isinstance(earliest, str) or not earliest:
        raise QualityGateError("prompt earliest marker is missing")
    if not isinstance(latest, str) or not latest or latest == earliest:
        raise QualityGateError("prompt latest marker is missing or ambiguous")
    if earliest != _marker(context_tokens, "earliest") or latest != _marker(
        context_tokens, "latest"
    ):
        raise QualityGateError("prompt retrieval probe marker differs from its spec")
    if not isinstance(rendered, str):
        raise QualityGateError("prompt rendered text is missing")
    if rendered.count(earliest) != 1 or rendered.count(latest) != 1:
        raise QualityGateError("prompt rendered text has ambiguous markers")
    if (
        _exact_int(
            prompt.get("earliest_token_index"),
            field=f"prompt[{context_tokens}].earliest_token_index",
        )
        >= _PROMPT_HEAD_GUARD_TOKENS
    ):
        raise QualityGateError("prompt earliest marker is outside the head window")
    if (
        _exact_int(
            prompt.get("latest_token_index"),
            field=f"prompt[{context_tokens}].latest_token_index",
        )
        < expected_prompt_tokens - _PROMPT_TAIL_GUARD_TOKENS
    ):
        raise QualityGateError("prompt latest marker is outside the tail window")
    return prompt


def _validate_route_evidence(
    value: object,
    *,
    field: str,
    expert_manifest_sha256: str,
    expert_manifest_content_sha256: str,
    routed_expert_record_hashes: Mapping[str, str],
) -> bool:
    routes = dict(_mapping(value, field=field))
    if set(routes) != {
        "trace",
        "trace_sha256",
        "expert_hashes",
        "manifest_binding",
    }:
        raise QualityGateError(f"{field} keys differ")
    trace = _sequence(routes.get("trace"), field=f"{field}.trace")
    if not trace:
        raise QualityGateError("route trace must be nonempty")
    routed_pairs: set[tuple[int, int]] = set()
    for index, raw_entry in enumerate(trace):
        entry = _mapping(raw_entry, field=f"{field}.trace[{index}]")
        layer = entry.get("layer")
        experts = entry.get("expert_ids")
        if layer is None and experts is None:
            continue
        exact_layer = _exact_int(layer, field=f"{field}.trace[{index}].layer")
        exact_experts = _sequence(
            experts,
            field=f"{field}.trace[{index}].expert_ids",
        )
        if not exact_experts:
            raise QualityGateError("route trace contains an empty expert route")
        for expert_index, expert in enumerate(exact_experts):
            routed_pairs.add(
                (
                    exact_layer,
                    _exact_int(
                        expert,
                        field=(f"{field}.trace[{index}].expert_ids[{expert_index}]"),
                    ),
                )
            )
    if not routed_pairs:
        raise QualityGateError("route trace contains no exact expert routes")
    observed_trace_hash = _hex_digest(
        routes.get("trace_sha256"),
        field=f"{field}.trace_sha256",
    )
    if observed_trace_hash != canonical_sha256(trace):
        raise QualityGateError("route trace SHA-256 differs from canonical trace")
    expert_hashes = _mapping(
        routes.get("expert_hashes"),
        field=f"{field}.expert_hashes",
    )
    expected_keys = {f"{layer}:{expert}" for layer, expert in routed_pairs}
    if set(expert_hashes) != expected_keys:
        raise QualityGateError("expert SHA-256 evidence does not cover exact routes")
    normalized_hashes = {
        str(pair): _hex_digest(digest, field=f"expert SHA-256 {pair}")
        for pair, digest in expert_hashes.items()
    }
    if any(
        routed_expert_record_hashes.get(pair) != digest
        for pair, digest in normalized_hashes.items()
    ):
        raise QualityGateError(
            f"{field} expert SHA-256 differs from pinned manifest binding"
        )
    binding = dict(
        _mapping(routes.get("manifest_binding"), field=f"{field}.manifest_binding")
    )
    expected_binding_keys = {
        "expert_manifest_sha256",
        "expert_manifest_content_sha256",
        "expert_hashes_sha256",
        "routed_pair_count",
        "live_manifest_records_verified",
        "verification_scope",
        "binding_sha256",
    }
    if set(binding) != expected_binding_keys:
        raise QualityGateError(f"{field} manifest binding keys differ")
    core = {
        name: binding[name]
        for name in expected_binding_keys
        if name != "binding_sha256"
    }
    if (
        binding.get("expert_manifest_sha256") != expert_manifest_sha256
        or binding.get("expert_manifest_content_sha256")
        != expert_manifest_content_sha256
        or binding.get("expert_hashes_sha256") != canonical_sha256(normalized_hashes)
        or _exact_int(
            binding.get("routed_pair_count"),
            field=f"{field}.manifest_binding.routed_pair_count",
            minimum=1,
        )
        != len(expected_keys)
        or binding.get("live_manifest_records_verified") is not True
        or binding.get("verification_scope") != ROUTE_VERIFICATION_SCOPE
    ):
        raise QualityGateError(f"{field} manifest binding contradicts routed experts")
    if _hex_digest(
        binding.get("binding_sha256"),
        field=f"{field}.manifest_binding.binding_sha256",
    ) != canonical_sha256(core):
        raise QualityGateError(f"{field} manifest binding SHA-256 differs")
    return True


def _validate_row(
    value: object,
    *,
    representation: str,
    context_tokens: int,
    prompt: Mapping[str, object],
    forbidden_markers: Sequence[str],
    prefill_chunk_size: int,
    tokenizer_identity_sha256: str,
    expert_manifest_sha256: str,
    expert_manifest_content_sha256: str,
    routed_expert_record_hashes: Mapping[str, str],
) -> dict[str, bool]:
    field = f"arms.{representation}.rows[{context_tokens}]"
    row = _mapping(value, field=field)
    if (
        _exact_int(
            row.get("context_tokens"), field=f"{field}.context_tokens", minimum=1
        )
        != context_tokens
    ):
        raise QualityGateError("quality row context order differs")
    prompt_tokens = context_tokens - COMPLETION_RESERVE_TOKENS
    if (
        _exact_int(row.get("prompt_tokens"), field=f"{field}.prompt_tokens", minimum=1)
        != prompt_tokens
    ):
        raise QualityGateError("quality row prompt token count differs")
    if (
        _exact_int(
            row.get("completion_tokens"),
            field=f"{field}.completion_tokens",
            minimum=1,
        )
        != COMPLETION_RESERVE_TOKENS
    ):
        raise QualityGateError("quality row completion token count differs")
    if row.get("prompt_sha256") != prompt.get("prompt_sha256"):
        raise QualityGateError("cross-arm prompt SHA-256 identity differs")

    generated_ids = _token_list(
        row.get("generated_token_ids"),
        field=f"{field}.generated_token_ids",
    )
    if len(generated_ids) != COMPLETION_RESERVE_TOKENS:
        raise QualityGateError("generated token length is not exact")
    generated_digest = _hex_digest(
        row.get("generated_token_sha256"),
        field=f"{field}.generated_token_sha256",
    )
    if generated_digest != _prompt_digest(generated_ids):
        raise QualityGateError("generated token SHA-256 differs from token_ids")
    generated_text = row.get("generated_text")
    if not isinstance(generated_text, str):
        raise QualityGateError("generated text evidence is missing")
    text_digest = _hex_digest(
        row.get("generated_text_sha256"),
        field=f"{field}.generated_text_sha256",
    )
    if text_digest != hashlib.sha256(generated_text.encode()).hexdigest():
        raise QualityGateError("generated text SHA-256 differs from decoded text")
    decode_binding_passed = _validate_decode_binding(
        row.get("decode_binding"),
        field=f"{field}.decode_binding",
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        generated_token_sha256=generated_digest,
        generated_text_sha256=text_digest,
    )
    recomputed_score = score_retrieval_text(
        generated_text,
        expected_earliest=_marker(context_tokens, "earliest"),
        expected_latest=_marker(context_tokens, "latest"),
        forbidden_markers=forbidden_markers,
    )
    recorded_score = _mapping(row.get("retrieval"), field=f"{field}.retrieval")
    if dict(recorded_score) != recomputed_score:
        raise QualityGateError("retrieval evidence contradicts decoded text")
    retrieval_passed = recomputed_score["passed"] is True

    kv = _mapping(row.get("kv"), field=f"{field}.kv")
    if kv.get("representation") != representation:
        raise QualityGateError("KV representation identity differs")
    bytes_per_token = _bytes_per_token(representation)
    if (
        _exact_int(
            kv.get("bytes_per_token"), field=f"{field}.kv.bytes_per_token", minimum=1
        )
        != bytes_per_token
    ):
        raise QualityGateError("KV byte geometry differs from representation")
    if (
        _exact_int(
            kv.get("block_size_tokens"),
            field=f"{field}.kv.block_size_tokens",
            minimum=1,
        )
        != KV_BLOCK_SIZE_TOKENS
    ):
        raise QualityGateError("KV block size differs")
    expected_pages = page_slots_for_context(context_tokens)
    if (
        _exact_int(
            kv.get("required_page_slots"),
            field=f"{field}.kv.required_page_slots",
            minimum=1,
        )
        != expected_pages
    ):
        raise QualityGateError("dynamic page slots differ from row context")
    capacity_tokens = expected_pages * KV_BLOCK_SIZE_TOKENS
    if (
        _exact_int(
            kv.get("physical_capacity_tokens"),
            field=f"{field}.kv.physical_capacity_tokens",
            minimum=1,
        )
        != capacity_tokens
    ):
        raise QualityGateError("KV physical capacity differs from dynamic page slots")
    expected_bytes = bytes_per_token * capacity_tokens
    if (
        _exact_int(
            kv.get("physical_bytes"),
            field=f"{field}.kv.physical_bytes",
            minimum=1,
        )
        != expected_bytes
    ):
        raise QualityGateError("KV aggregate bytes differ from exact page geometry")
    expected_environment = quality_environment(
        representation,
        context_tokens=context_tokens,
        prefill_chunk_size=prefill_chunk_size,
    )
    attention_environment_passed = (
        dict(
            _mapping(
                kv.get("attention_environment"),
                field=f"{field}.kv.attention_environment",
            )
        )
        == expected_environment
    )
    entries = _sequence(kv.get("entries"), field=f"{field}.kv.entries")
    entry_count = _exact_int(
        kv.get("entry_count"), field=f"{field}.kv.entry_count", minimum=1
    )
    if not entries or entry_count != len(entries):
        raise QualityGateError("paged KV entry count is missing or contradictory")
    initial = _mapping(kv.get("initial"), field=f"{field}.kv.initial")
    if (
        _exact_int(
            initial.get("entry_count"),
            field=f"{field}.kv.initial.entry_count",
            minimum=1,
        )
        != entry_count
    ):
        raise QualityGateError("initial paged KV entry count differs")
    initial_blocks = [
        _exact_int(value, field=f"{field}.kv.initial.num_blocks", minimum=1)
        for value in _sequence(
            initial.get("num_blocks"), field=f"{field}.kv.initial.num_blocks"
        )
    ]
    initial_offsets = [
        _exact_int(value, field=f"{field}.kv.initial.entry_offsets")
        for value in _sequence(
            initial.get("entry_offsets"),
            field=f"{field}.kv.initial.entry_offsets",
        )
    ]
    initial_bytes = [
        _exact_int(value, field=f"{field}.kv.initial.entry_bytes")
        for value in _sequence(
            initial.get("entry_bytes"), field=f"{field}.kv.initial.entry_bytes"
        )
    ]
    if not (
        len(initial_blocks) == len(initial_offsets) == len(initial_bytes) == entry_count
    ):
        raise QualityGateError("initial paged KV entry arrays differ")
    initial_physical_bytes = _exact_int(
        initial.get("physical_bytes"),
        field=f"{field}.kv.initial.physical_bytes",
    )
    if initial_physical_bytes != sum(initial_bytes):
        raise QualityGateError("initial paged KV bytes contradict their entries")
    if representation == "q4_dynamic":
        if (
            any(blocks != 1 for blocks in initial_blocks)
            or any(initial_offsets)
            or any(byte_count <= 0 for byte_count in initial_bytes)
            or initial_physical_bytes != Q4_KV_BYTES_PER_TOKEN * KV_BLOCK_SIZE_TOKENS
        ):
            raise QualityGateError(
                "dynamic Q4 must start at one broker-owned 16-token page"
            )
    elif (
        any(blocks != expected_pages for blocks in initial_blocks)
        or any(initial_offsets)
        or any(initial_bytes)
        or initial_physical_bytes != 0
    ):
        raise QualityGateError(
            "static quality arm initial page geometry is contradictory"
        )
    aggregate_bytes = 0
    geometry_passed = True
    attention_path_passed = attention_environment_passed
    representation_path_passed = True
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, field=f"{field}.kv.entries[{index}]")
        block_size = _exact_int(
            entry.get("block_size"),
            field=f"{field}.kv.entries[{index}].block_size",
            minimum=1,
        )
        num_blocks = _exact_int(
            entry.get("num_blocks"),
            field=f"{field}.kv.entries[{index}].num_blocks",
            minimum=1,
        )
        capacity = _exact_int(
            entry.get("capacity"),
            field=f"{field}.kv.entries[{index}].capacity",
            minimum=1,
        )
        offset = _exact_int(
            entry.get("offset"),
            field=f"{field}.kv.entries[{index}].offset",
            minimum=1,
        )
        if capacity != block_size * num_blocks:
            raise QualityGateError("paged KV entry capacity contradicts its pages")
        geometry_passed = geometry_passed and (
            block_size == KV_BLOCK_SIZE_TOKENS
            and num_blocks == expected_pages
            and capacity == capacity_tokens
            and offset == context_tokens
        )
        aggregate_bytes += _exact_int(
            entry.get("bytes"),
            field=f"{field}.kv.entries[{index}].bytes",
            minimum=1,
        )
        sliding_window = _exact_int(
            entry.get("sliding_window"),
            field=f"{field}.kv.entries[{index}].sliding_window",
        )
        paged_attention_calls = _exact_int(
            entry.get("paged_attention_calls"),
            field=f"{field}.kv.entries[{index}].paged_attention_calls",
        )
        updates = _exact_int(
            entry.get("updates"),
            field=f"{field}.kv.entries[{index}].updates",
        )
        cache_write_time_s = _nonnegative_number(
            entry.get("cache_write_time_s"),
            field=f"{field}.kv.entries[{index}].cache_write_time_s",
        )
        attention_time_s = _nonnegative_number(
            entry.get("attention_time_s"),
            field=f"{field}.kv.entries[{index}].attention_time_s",
        )
        dense_fallbacks = _exact_int(
            entry.get("dense_fallback_calls"),
            field=f"{field}.kv.entries[{index}].dense_fallback_calls",
        )
        active_array_calls = _exact_int(
            entry.get("active_array_calls"),
            field=f"{field}.kv.entries[{index}].active_array_calls",
        )
        split_fallbacks = _exact_int(
            entry.get("large_q_split_sdpa_fallback_calls"),
            field=(f"{field}.kv.entries[{index}].large_q_split_sdpa_fallback_calls"),
        )
        partitioned_paged_calls = _exact_int(
            entry.get("partitioned_paged_calls"),
            field=f"{field}.kv.entries[{index}].partitioned_paged_calls",
        )
        q4_chunked_attention_calls = _exact_int(
            entry.get("q4_chunked_dequant_attention_calls"),
            field=(f"{field}.kv.entries[{index}].q4_chunked_dequant_attention_calls"),
        )
        large_q_path = entry.get("paged_attention_large_q_path")
        if not isinstance(large_q_path, str):
            raise QualityGateError(
                f"{field}.kv.entries[{index}].paged_attention_large_q_path "
                "must be a string"
            )
        bailouts = _sum_nested_counts(
            entry.get("paged_attention_bailouts_by_phase_reason"),
            field=(
                f"{field}.kv.entries[{index}].paged_attention_bailouts_by_phase_reason"
            ),
        )
        attention_path_passed = attention_path_passed and (
            sliding_window == 0
            and paged_attention_calls > 0
            and active_array_calls == 0
            and dense_fallbacks == 0
            and bailouts == 0
        )
        quantized = _exact_int(
            entry.get("kv_quant", 0),
            field=f"{field}.kv.entries[{index}].kv_quant",
        )
        quant_calls = _exact_int(
            entry.get("kv_quant_attention_calls"),
            field=f"{field}.kv.entries[{index}].kv_quant_attention_calls",
        )
        dequant_calls = _exact_int(
            entry.get("kv_quant_dequant_calls"),
            field=f"{field}.kv.entries[{index}].kv_quant_dequant_calls",
        )
        dequant_time_s = _nonnegative_number(
            entry.get("kv_quant_dequant_time_s"),
            field=f"{field}.kv.entries[{index}].kv_quant_dequant_time_s",
        )
        dequant_tokens = _exact_int(
            entry.get("kv_quant_dequant_tokens"),
            field=f"{field}.kv.entries[{index}].kv_quant_dequant_tokens",
        )
        chunk_dequant_calls = _exact_int(
            entry.get("q4_chunked_dequant_calls"),
            field=f"{field}.kv.entries[{index}].q4_chunked_dequant_calls",
        )
        chunk_dequant_time_s = _nonnegative_number(
            entry.get("q4_chunked_dequant_time_s"),
            field=f"{field}.kv.entries[{index}].q4_chunked_dequant_time_s",
        )
        chunk_dequant_tokens = _exact_int(
            entry.get("q4_chunked_dequant_tokens"),
            field=f"{field}.kv.entries[{index}].q4_chunked_dequant_tokens",
        )
        storage_dtypes = dict(
            _mapping(
                entry.get("storage_dtypes"),
                field=f"{field}.kv.entries[{index}].storage_dtypes",
            )
        )
        representation_path_passed = representation_path_passed and (
            storage_dtypes == _STORAGE_DTYPES[representation]
        )
        if representation in _Q4_REPRESENTATIONS:
            attention_calls_by_phase = _exact_phase_counts(
                entry.get("q4_chunked_dequant_attention_calls_by_phase"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_attention_calls_by_phase"
                ),
            )
            attention_paths_by_phase = dict(
                _mapping(
                    entry.get("q4_chunked_dequant_attention_path_by_phase"),
                    field=(
                        f"{field}.kv.entries[{index}]."
                        "q4_chunked_dequant_attention_path_by_phase"
                    ),
                )
            )
            expected_unique_by_phase = _exact_phase_counts(
                entry.get("q4_chunked_dequant_expected_unique_tokens_by_phase"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_expected_unique_tokens_by_phase"
                ),
            )
            unique_by_phase = _exact_phase_counts(
                entry.get("q4_chunked_dequant_unique_tokens_by_phase"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_unique_tokens_by_phase"
                ),
            )
            coverage_failures = _exact_int(
                entry.get("q4_chunked_dequant_coverage_failures"),
                field=(
                    f"{field}.kv.entries[{index}].q4_chunked_dequant_coverage_failures"
                ),
            )
            realized_calls = _exact_int(
                entry.get("q4_chunked_dequant_realized_calls"),
                field=(
                    f"{field}.kv.entries[{index}].q4_chunked_dequant_realized_calls"
                ),
            )
            configured_chunk_tokens = _exact_int(
                entry.get("q4_chunked_dequant_configured_chunk_tokens"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_configured_chunk_tokens"
                ),
                minimum=1,
            )
            realized_peak_tokens = _exact_int(
                entry.get("q4_chunked_dequant_realized_peak_chunk_tokens"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_peak_chunk_tokens"
                ),
                minimum=1,
            )
            realized_peak_tokens_by_phase = _exact_phase_counts(
                entry.get("q4_chunked_dequant_realized_peak_chunk_tokens_by_phase"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_peak_chunk_tokens_by_phase"
                ),
            )
            realized_source_bytes_per_token = _exact_int(
                entry.get("q4_chunked_dequant_realized_source_bytes_per_token"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_source_bytes_per_token"
                ),
                minimum=1,
            )
            realized_compute_bytes_per_token = _exact_int(
                entry.get("q4_chunked_dequant_realized_compute_bytes_per_token"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_compute_bytes_per_token"
                ),
                minimum=1,
            )
            realized_bytes_per_token = _exact_int(
                entry.get("q4_chunked_dequant_realized_bytes_per_token"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_bytes_per_token"
                ),
                minimum=1,
            )
            realized_peak_bytes_by_phase = _exact_phase_counts(
                entry.get("q4_chunked_dequant_realized_peak_chunk_bytes_by_phase"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_peak_chunk_bytes_by_phase"
                ),
            )
            realized_peak_bytes = _exact_int(
                entry.get("q4_chunked_dequant_realized_peak_chunk_bytes"),
                field=(
                    f"{field}.kv.entries[{index}]."
                    "q4_chunked_dequant_realized_peak_chunk_bytes"
                ),
                minimum=1,
            )
            expected_calls_by_phase, expected_coverage_by_phase = (
                _expected_q4_phase_evidence(
                    context_tokens=context_tokens,
                    prefill_chunk_size=prefill_chunk_size,
                )
            )
            expected_chunk_tokens = int(
                expected_environment["MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE"]
            )
            expected_query_chunk_tokens = int(
                expected_environment["MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE"]
            )
            expected_attention_calls = sum(expected_calls_by_phase.values())
            expected_realized_calls_by_phase, expected_realized_tokens_by_phase = (
                _expected_q4_realized_evidence(
                    context_tokens=context_tokens,
                    prefill_chunk_size=prefill_chunk_size,
                    query_chunk_size=expected_query_chunk_tokens,
                    kv_chunk_size=expected_chunk_tokens,
                )
            )
            expected_realized_calls = sum(expected_realized_calls_by_phase.values())
            expected_realized_tokens = sum(expected_realized_tokens_by_phase.values())
            expected_peak_tokens_by_phase = {
                "prefill": min(expected_chunk_tokens, prompt_tokens),
                "ar_decode": min(expected_chunk_tokens, context_tokens),
            }
            expected_peak_bytes_by_phase = {
                phase: tokens * Q4_STREAMING_DEQUANT_BYTES_PER_TOKEN_PER_LAYER
                for phase, tokens in expected_peak_tokens_by_phase.items()
            }
            attention_path_passed = attention_path_passed and (
                q4_chunked_attention_calls == expected_attention_calls
                and paged_attention_calls == expected_attention_calls
                and quant_calls == expected_attention_calls
                and attention_calls_by_phase == expected_calls_by_phase
                and attention_paths_by_phase
                == {
                    "prefill": "q4_streaming_softmax",
                    "ar_decode": "q4_streaming_softmax",
                }
                and split_fallbacks == 0
                and partitioned_paged_calls == 0
                and large_q_path == ""
            )
            representation_path_passed = representation_path_passed and (
                quantized == 1
                and entry.get("kv_quant_mode") == "q4"
                and updates > 0
                and cache_write_time_s > 0
                and attention_time_s > 0
                and dequant_calls == 0
                and dequant_time_s == 0
                and dequant_tokens == 0
                and chunk_dequant_calls == expected_realized_calls
                and chunk_dequant_time_s > 0
                and chunk_dequant_tokens == expected_realized_tokens
                and realized_calls == chunk_dequant_calls
                and expected_unique_by_phase == expected_coverage_by_phase
                and unique_by_phase == expected_coverage_by_phase
                and coverage_failures == 0
                and configured_chunk_tokens == expected_chunk_tokens
                and realized_peak_tokens == max(expected_peak_tokens_by_phase.values())
                and realized_peak_tokens_by_phase == expected_peak_tokens_by_phase
                and realized_source_bytes_per_token
                == Q4_STREAMING_DEQUANT_SOURCE_BYTES_PER_TOKEN_PER_LAYER
                and realized_compute_bytes_per_token
                == Q4_STREAMING_DEQUANT_COMPUTE_BYTES_PER_TOKEN_PER_LAYER
                and realized_bytes_per_token
                == Q4_STREAMING_DEQUANT_BYTES_PER_TOKEN_PER_LAYER
                and realized_peak_bytes
                == realized_peak_tokens * realized_bytes_per_token
                and realized_peak_bytes_by_phase == expected_peak_bytes_by_phase
            )
        else:
            attention_path_passed = attention_path_passed and (
                split_fallbacks == 0
                and q4_chunked_attention_calls == 0
                and partitioned_paged_calls > 0
                and large_q_path == "partitioned_paged"
            )
            representation_path_passed = representation_path_passed and (
                quantized == 0
                and entry.get("kv_quant_mode", "") in {"", None}
                and updates > 0
                and cache_write_time_s > 0
                and attention_time_s > 0
                and quant_calls == 0
                and dequant_calls == 0
                and dequant_time_s == 0
                and dequant_tokens == 0
                and chunk_dequant_calls == 0
                and chunk_dequant_time_s == 0
                and chunk_dequant_tokens == 0
            )
    observed_physical_bytes = _exact_int(
        kv.get("physical_bytes"),
        field=f"{field}.kv.physical_bytes",
        minimum=1,
    )
    if aggregate_bytes != observed_physical_bytes:
        raise QualityGateError("paged KV entry bytes contradict aggregate bytes")
    geometry_passed = geometry_passed and aggregate_bytes == expected_bytes

    ownership = dict(_mapping(kv.get("ownership"), field=f"{field}.kv.ownership"))
    if set(ownership) != {"policy", "owners", "broker"}:
        raise QualityGateError("KV ownership evidence keys differ")
    raw_owners = _sequence(
        ownership.get("owners"),
        field=f"{field}.kv.ownership.owners",
    )
    if representation == "q4_dynamic":
        if ownership.get("policy") != "brokered_target_group":
            raise QualityGateError("dynamic Q4 did not use target-owner grouping")
        if entry_count != HY3_Q4_KV_LAYERS or len(raw_owners) != entry_count:
            raise QualityGateError(
                "dynamic Q4 target-owner evidence must cover all attention layers"
            )
        cache_ids: list[str] = []
        for index, raw_owner in enumerate(raw_owners):
            owner = dict(
                _mapping(
                    raw_owner,
                    field=f"{field}.kv.ownership.owners[{index}]",
                )
            )
            if set(owner) != {
                "entry_index",
                "cache_id",
                "allocation_count",
                "committed_physical_bytes",
                "allocation_handle_bytes",
            }:
                raise QualityGateError("dynamic Q4 target-owner keys differ")
            if (
                _exact_int(
                    owner.get("entry_index"),
                    field=f"{field}.kv.ownership.owners[{index}].entry_index",
                )
                != index
            ):
                raise QualityGateError("dynamic Q4 target-owner order differs")
            cache_id = owner.get("cache_id")
            if not isinstance(cache_id, str) or not cache_id.startswith("target:"):
                raise QualityGateError("dynamic Q4 owner is not a target cache")
            cache_ids.append(cache_id)
            expected_entry_bytes = _exact_int(
                _mapping(
                    entries[index],
                    field=f"{field}.kv.entries[{index}]",
                ).get("bytes"),
                field=f"{field}.kv.entries[{index}].bytes",
                minimum=1,
            )
            if (
                _exact_int(
                    owner.get("allocation_count"),
                    field=(f"{field}.kv.ownership.owners[{index}].allocation_count"),
                    minimum=1,
                )
                < 1
                or _exact_int(
                    owner.get("committed_physical_bytes"),
                    field=(
                        f"{field}.kv.ownership.owners[{index}].committed_physical_bytes"
                    ),
                    minimum=1,
                )
                != expected_entry_bytes
                or _exact_int(
                    owner.get("allocation_handle_bytes"),
                    field=(
                        f"{field}.kv.ownership.owners[{index}].allocation_handle_bytes"
                    ),
                    minimum=1,
                )
                != expected_entry_bytes
            ):
                raise QualityGateError(
                    "dynamic Q4 target-owner bytes contradict physical entries"
                )
        if len(set(cache_ids)) != len(cache_ids):
            raise QualityGateError("dynamic Q4 target-owner cache ids are not unique")
        broker = dict(
            _mapping(
                ownership.get("broker"),
                field=f"{field}.kv.ownership.broker",
            )
        )
        if set(broker) != {
            "kv_physical_bytes",
            "owned_kv_physical_bytes",
            "unreconciled_kv_physical_bytes",
            "pending_kv_ticket_id",
            "failed_closed",
        }:
            raise QualityGateError("dynamic Q4 broker evidence keys differ")
        if (
            _exact_int(
                broker.get("kv_physical_bytes"),
                field=f"{field}.kv.ownership.broker.kv_physical_bytes",
                minimum=1,
            )
            != expected_bytes
            or _exact_int(
                broker.get("owned_kv_physical_bytes"),
                field=f"{field}.kv.ownership.broker.owned_kv_physical_bytes",
                minimum=1,
            )
            != expected_bytes
            or _exact_int(
                broker.get("unreconciled_kv_physical_bytes"),
                field=(f"{field}.kv.ownership.broker.unreconciled_kv_physical_bytes"),
            )
            != 0
            or broker.get("pending_kv_ticket_id") is not None
            or broker.get("failed_closed") is not False
        ):
            raise QualityGateError(
                "dynamic Q4 broker ownership contradicts physical KV"
            )
    else:
        if (
            ownership.get("policy") != "unbrokered_static"
            or raw_owners
            or ownership.get("broker") is not None
        ):
            raise QualityGateError("static quality arm has unexpected KV ownership")
    ownership_passed = True

    route_integrity_passed = _validate_route_evidence(
        row.get("routes"),
        field=f"{field}.routes",
        expert_manifest_sha256=expert_manifest_sha256,
        expert_manifest_content_sha256=expert_manifest_content_sha256,
        routed_expert_record_hashes=routed_expert_record_hashes,
    )
    health = _mapping(row.get("health"), field=f"{field}.health")
    health_passed = True
    for health_field in _HEALTH_FIELDS:
        count = _exact_int(
            health.get(health_field),
            field=f"{field}.health.{health_field}",
        )
        health_passed = health_passed and count == 0

    reset = _mapping(row.get("reset"), field=f"{field}.reset")
    reset_count = _exact_int(
        reset.get("entry_count"), field=f"{field}.reset.entry_count", minimum=1
    )
    offsets = _sequence(
        reset.get("entry_offsets"), field=f"{field}.reset.entry_offsets"
    )
    byte_counts = _sequence(
        reset.get("entry_bytes"), field=f"{field}.reset.entry_bytes"
    )
    closed = _sequence(
        reset.get("entries_closed"), field=f"{field}.reset.entries_closed"
    )
    if not (
        reset_count == entry_count == len(offsets) == len(byte_counts) == len(closed)
    ):
        raise QualityGateError("reset entry count differs from live paged entries")
    reset_passed = not (
        any(
            _exact_int(offset, field=f"{field}.reset.offset") != 0 for offset in offsets
        )
        or any(
            _exact_int(byte_count, field=f"{field}.reset.bytes") != 0
            for byte_count in byte_counts
        )
        or any(value is not True for value in closed)
        or _exact_int(
            reset.get("live_kv_tokens"),
            field=f"{field}.reset.live_kv_tokens",
        )
        != 0
    )
    expected_broker_release = (
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
    if reset.get("broker_release") != expected_broker_release:
        raise QualityGateError("post-reset broker release evidence differs")
    if reset.get("passed") is not reset_passed:
        raise QualityGateError("reset pass flag contradicts exact reset evidence")
    post_reset_health = _mapping(
        reset.get("health"),
        field=f"{field}.reset.health",
    )
    post_reset_health_passed = True
    for health_field in _HEALTH_FIELDS:
        count = _exact_int(
            post_reset_health.get(health_field),
            field=f"{field}.reset.health.{health_field}",
        )
        post_reset_health_passed = post_reset_health_passed and count == 0
    elapsed = row.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed <= 0
    ):
        raise QualityGateError("quality row elapsed time is invalid")
    return {
        "retrieval": retrieval_passed,
        "decode_binding": decode_binding_passed,
        "route_integrity": route_integrity_passed,
        "geometry": geometry_passed,
        "ownership": ownership_passed,
        "attention_path": attention_path_passed,
        "representation_path": representation_path_passed,
        "health": health_passed and post_reset_health_passed,
        "reset": reset_passed,
    }


def _validate_runtime_config_binding(
    value: object,
    *,
    representation: str,
) -> tuple[tuple[int, int, int], bool, bool]:
    from mtplx.expert_runtime import ExpertStreamingConfig
    from mtplx.expert_streaming_models import HY3_Q4

    field = f"configuration.representations.{representation}"
    dynamic = representation == "q4_dynamic"
    rep_config = _mapping(value, field=field)
    runtime_mapping = dict(
        _mapping(rep_config.get("runtime_config"), field=f"{field}.runtime_config")
    )
    runtime_digest = _hex_digest(
        rep_config.get("runtime_config_sha256"),
        field=f"{field}.runtime_config_sha256",
    )
    if runtime_digest != canonical_sha256(runtime_mapping):
        raise QualityGateError("runtime config SHA-256 contradicts its payload")
    try:
        runtime_config = ExpertStreamingConfig(**runtime_mapping)
    except (TypeError, ValueError) as exc:
        raise QualityGateError(f"runtime config is invalid: {exc}") from exc
    plan = runtime_config.memory_plan(HY3_Q4)
    reported_summary = (
        _exact_int(
            rep_config.get("planned_persistent_slots"),
            field=f"{field}.planned_persistent_slots",
            minimum=1,
        ),
        _exact_int(
            rep_config.get("persistent_cache_bytes"),
            field=f"{field}.persistent_cache_bytes",
            minimum=1,
        ),
    )
    if reported_summary != (
        int(plan.persistent_slots),
        int(plan.persistent_cache_bytes),
    ):
        raise QualityGateError("runtime config memory plan contradicts its summary")
    summary = (*reported_summary, int(runtime_config.memory_limit_bytes))
    if (
        _exact_int(
            rep_config.get("kv_bytes_per_token"),
            field=f"{field}.kv_bytes_per_token",
            minimum=1,
        )
        != runtime_config.kv_bytes_per_token_override
    ):
        raise QualityGateError("runtime config KV bytes contradict their summary")
    if (
        _exact_int(
            rep_config.get("expert_cache_limit_bytes"),
            field=f"{field}.expert_cache_limit_bytes",
            minimum=1,
        )
        != runtime_config.expert_cache_limit_bytes
    ):
        raise QualityGateError("runtime config expert cap contradicts its summary")
    fixed_expert_passed = (
        runtime_config.model_key == "hy3-q4"
        and runtime_config.memory_limit_bytes > EXPERT_CACHE_LIMIT_BYTES
        and runtime_config.max_live_kv_tokens == max(QUALITY_CONTEXTS)
        and runtime_config.kv_bytes_per_token_override
        == _bytes_per_token(representation)
        and runtime_config.runtime_reserve_bytes == 8 * GIB
        and runtime_config.allocator_headroom_bytes == GIB
        and runtime_config.expert_cache_limit_bytes == EXPERT_CACHE_LIMIT_BYTES
        and runtime_config.cache_policy == "lru"
        and runtime_config.cache_scope == "global"
        and runtime_config.slot_layout == "direct-slots"
        and runtime_config.dynamic_expert_cache is dynamic
        and runtime_config.verify_sidecar_hash_at_open is False
        and runtime_config.verify_record_hashes is True
        and runtime_config.resource_telemetry is True
        and runtime_config.trace_routes is True
        and plan.persistent_cache_bytes <= EXPERT_CACHE_LIMIT_BYTES
    )
    attention_passed = dict(
        _mapping(
            rep_config.get("attention_identity"),
            field=f"{field}.attention_identity",
        )
    ) == _attention_identity(representation)
    return summary, fixed_expert_passed, attention_passed


def _validate_artifact_evidence(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    evidence = dict(_mapping(value, field=field))
    expected_keys = {
        "schema",
        "model_artifact_sha256",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "sidecar_fingerprint",
        "resident_payload_bytes",
        "resident_payload_sha256",
        "resident_shard_fingerprints",
        "payload_hash_verified",
        "payload_hash_io_mode",
    }
    if set(evidence) != expected_keys:
        missing = sorted(expected_keys - set(evidence))
        unknown = sorted(set(evidence) - expected_keys)
        raise QualityGateError(
            f"{field} attestation keys differ (missing={missing}, unknown={unknown})"
        )
    if evidence.get("schema") != SCHEMA_ARTIFACT_ATTESTATION:
        raise QualityGateError(f"{field} attestation schema differs")
    for name in (
        "model_artifact_sha256",
        "expert_manifest_sha256",
        "artifact_pins_sha256",
        "artifact_stat_sha256",
        "resident_payload_sha256",
    ):
        _hex_digest(evidence.get(name), field=f"{field}.{name}")
    _exact_int(
        evidence.get("resident_payload_bytes"),
        field=f"{field}.resident_payload_bytes",
        minimum=1,
    )
    fingerprint_keys = {"device", "inode", "size", "mtime_ns", "ctime_ns"}

    def fingerprint(raw: object, *, fingerprint_field: str) -> dict[str, int]:
        exact = dict(_mapping(raw, field=fingerprint_field))
        if set(exact) != fingerprint_keys:
            raise QualityGateError(f"{fingerprint_field} keys differ")
        return {
            name: _exact_int(
                exact[name],
                field=f"{fingerprint_field}.{name}",
                minimum=1 if name == "size" else 0,
            )
            for name in fingerprint_keys
        }

    sidecar = fingerprint(
        evidence.get("sidecar_fingerprint"),
        fingerprint_field=f"{field}.sidecar_fingerprint",
    )
    shard_values = _sequence(
        evidence.get("resident_shard_fingerprints"),
        field=f"{field}.resident_shard_fingerprints",
    )
    if not shard_values:
        raise QualityGateError(f"{field} has no resident shard fingerprints")
    resident_shards: list[dict[str, int | str]] = []
    for index, raw_shard in enumerate(shard_values):
        shard_field = f"{field}.resident_shard_fingerprints[{index}]"
        shard = dict(_mapping(raw_shard, field=shard_field))
        if set(shard) != fingerprint_keys | {"name"}:
            raise QualityGateError(f"{shard_field} keys differ")
        name = shard.get("name")
        if not isinstance(name, str) or not name:
            raise QualityGateError(f"{shard_field}.name must be nonempty")
        pure = PurePosixPath(name)
        if pure.is_absolute() or pure.as_posix() != name or ".." in pure.parts:
            raise QualityGateError(f"{shard_field}.name must be a safe relative path")
        resident_shards.append(
            {
                "name": name,
                **fingerprint(
                    {key: shard[key] for key in fingerprint_keys},
                    fingerprint_field=shard_field,
                ),
            }
        )
    names = [str(shard["name"]) for shard in resident_shards]
    if names != sorted(set(names)):
        raise QualityGateError(f"{field} resident shard fingerprints are not canonical")
    if not isinstance(evidence.get("payload_hash_verified"), bool):
        raise QualityGateError(f"{field}.payload_hash_verified must be bool")
    expected_modes = (
        {"buffered", "f-nocache"}
        if evidence["payload_hash_verified"] is True
        else {"not-run"}
    )
    if evidence.get("payload_hash_io_mode") not in expected_modes:
        raise QualityGateError(f"{field} payload hash mode contradicts its flag")
    recomputed_stat = canonical_sha256(
        {"sidecar": sidecar, "resident_shards": resident_shards}
    )
    if evidence.get("artifact_stat_sha256") != recomputed_stat:
        raise QualityGateError(
            f"{field} artifact stat SHA-256 contradicts fingerprints"
        )
    return evidence


def validate_quality_result(value: object) -> dict[str, object]:
    """Validate evidence structure and classify acceptance without hiding failures."""

    result = _mapping(value, field="quality result")
    if result.get("schema") != HY3_KV_QUALITY_SCHEMA:
        raise QualityGateError("quality result schema differs")
    identity = _mapping(result.get("identity"), field="identity")
    model_artifact_id = identity.get("model_artifact_id")
    if not isinstance(model_artifact_id, str) or not model_artifact_id:
        raise QualityGateError("model artifact identity is missing")
    for field, lengths in (
        ("model_artifact_sha256", (64,)),
        ("expert_manifest_sha256", (64,)),
        ("expert_manifest_content_sha256", (64,)),
        ("artifact_pins_sha256", (64,)),
        ("artifact_stat_sha256", (64,)),
        ("resident_payload_sha256", (64,)),
        ("source_git_commit", (40, 64)),
    ):
        _hex_digest(identity.get(field), field=f"identity.{field}", lengths=lengths)
    _exact_int(
        identity.get("resident_payload_bytes"),
        field="identity.resident_payload_bytes",
        minimum=1,
    )
    if not isinstance(identity.get("payload_hash_verified"), bool):
        raise QualityGateError("identity.payload_hash_verified must be bool")
    tokenizer_identity = _validate_tokenizer_identity(
        identity.get("tokenizer_identity"),
        field="identity.tokenizer_identity",
    )
    tokenizer_identity_sha256 = str(tokenizer_identity["identity_sha256"])
    expert_manifest_sha256 = str(identity["expert_manifest_sha256"])
    expert_manifest_content_sha256 = str(identity["expert_manifest_content_sha256"])
    routed_expert_attestation = _validate_routed_expert_attestation(
        identity.get("routed_expert_attestation"),
        field="identity.routed_expert_attestation",
        expert_manifest_content_sha256=expert_manifest_content_sha256,
    )
    routed_expert_record_hashes = _mapping(
        routed_expert_attestation["record_hashes"],
        field="identity.routed_expert_attestation.record_hashes",
    )

    configuration = _mapping(result.get("configuration"), field="configuration")
    contexts = tuple(
        _exact_int(context, field="configuration.contexts", minimum=1)
        for context in _sequence(
            configuration.get("contexts"), field="configuration.contexts"
        )
    )
    if contexts != QUALITY_CONTEXTS:
        raise QualityGateError("quality contexts differ from the exact matrix")
    if (
        _exact_int(
            configuration.get("completion_reserve_tokens"),
            field="configuration.completion_reserve_tokens",
            minimum=1,
        )
        != COMPLETION_RESERVE_TOKENS
    ):
        raise QualityGateError("quality completion reserve differs")
    top_expert_cap = _exact_int(
        configuration.get("expert_cache_limit_bytes"),
        field="configuration.expert_cache_limit_bytes",
        minimum=1,
    )
    prefill_chunk_size = _exact_int(
        configuration.get("prefill_chunk_size"),
        field="configuration.prefill_chunk_size",
        minimum=1,
    )
    representation_configs = _mapping(
        configuration.get("representations"),
        field="configuration.representations",
    )
    if set(representation_configs) != set(_REPRESENTATIONS):
        raise QualityGateError("quality representations differ")
    plan_identities: list[tuple[int, int, int]] = []
    fixed_config_results: list[bool] = []
    attention_config_results: list[bool] = []
    for representation in _REPRESENTATIONS:
        plan, fixed, attention = _validate_runtime_config_binding(
            representation_configs[representation],
            representation=representation,
        )
        plan_identities.append(plan)
        fixed_config_results.append(fixed)
        attention_config_results.append(attention)
    fixed_expert_ownership = (
        top_expert_cap == EXPERT_CACHE_LIMIT_BYTES
        and all(fixed_config_results)
        and len(plan_identities) == len(_REPRESENTATIONS)
        and len(set(plan_identities)) == 1
    )

    prompt_values = _sequence(result.get("prompts"), field="prompts")
    if len(prompt_values) != len(QUALITY_CONTEXTS):
        raise QualityGateError("quality prompt count differs")
    prompts = [
        _validate_prompt(prompt, context_tokens=context)
        for context, prompt in zip(QUALITY_CONTEXTS, prompt_values, strict=True)
    ]
    marker_set = {
        str(prompt[marker])
        for prompt in prompts
        for marker in ("earliest_marker", "latest_marker")
    }
    if len(marker_set) != len(prompts) * 2:
        raise QualityGateError("quality prompts reuse a retrieval marker")

    arms = _mapping(result.get("arms"), field="arms")
    if set(arms) != set(_REPRESENTATIONS):
        raise QualityGateError("quality result arms differ")
    artifact_before_values: list[dict[str, object]] = []
    artifact_results: list[bool] = []
    row_results: list[dict[str, bool]] = []
    rows_by_representation: dict[str, Sequence[object]] = {}
    top_artifact_binding = {
        name: identity[name]
        for name in (
            "model_artifact_sha256",
            "expert_manifest_sha256",
            "artifact_pins_sha256",
            "artifact_stat_sha256",
            "resident_payload_bytes",
            "resident_payload_sha256",
        )
    }
    for representation in _REPRESENTATIONS:
        arm = _mapping(arms[representation], field=f"arms.{representation}")
        if arm.get("representation") != representation:
            raise QualityGateError("quality arm representation differs")
        if arm.get("model_artifact_id") != model_artifact_id or arm.get(
            "source_git_commit"
        ) != identity.get("source_git_commit"):
            raise QualityGateError("quality arm identity contradicts top identity")
        if arm.get("expert_manifest_content_sha256") != expert_manifest_content_sha256:
            raise QualityGateError(
                "quality arm manifest content identity contradicts top identity"
            )
        if (
            arm.get("routed_expert_attestation_sha256")
            != routed_expert_attestation["binding_sha256"]
        ):
            raise QualityGateError(
                "quality arm routed experts contradict top manifest attestation"
            )
        arm_tokenizer_identity = _validate_tokenizer_identity(
            arm.get("tokenizer_identity"),
            field=f"arms.{representation}.tokenizer_identity",
        )
        if arm_tokenizer_identity != tokenizer_identity:
            raise QualityGateError(
                "quality arm tokenizer identity contradicts top identity"
            )
        before = _validate_artifact_evidence(
            arm.get("artifact_before"), field=f"arms.{representation}.artifact_before"
        )
        after = _validate_artifact_evidence(
            arm.get("artifact_after"), field=f"arms.{representation}.artifact_after"
        )
        if any(
            before.get(name) != digest for name, digest in top_artifact_binding.items()
        ):
            raise QualityGateError("arm artifact attestation contradicts top identity")
        artifact_before_values.append(before)
        artifact_results.append(
            before == after and before.get("payload_hash_verified") is False
        )
        rows = _sequence(arm.get("rows"), field=f"arms.{representation}.rows")
        if len(rows) != len(QUALITY_CONTEXTS):
            raise QualityGateError("quality row count differs")
        for context_index, (context, prompt, row) in enumerate(
            zip(
                QUALITY_CONTEXTS,
                prompts,
                rows,
                strict=True,
            )
        ):
            forbidden = [
                _marker(prior_context, edge)
                for prior_context in QUALITY_CONTEXTS[:context_index]
                for edge in ("earliest", "latest")
            ]
            row_results.append(
                _validate_row(
                    row,
                    representation=representation,
                    context_tokens=context,
                    prompt=prompt,
                    forbidden_markers=forbidden,
                    prefill_chunk_size=prefill_chunk_size,
                    tokenizer_identity_sha256=tokenizer_identity_sha256,
                    expert_manifest_sha256=expert_manifest_sha256,
                    expert_manifest_content_sha256=(expert_manifest_content_sha256),
                    routed_expert_record_hashes=routed_expert_record_hashes,
                )
            )
        rows_by_representation[representation] = rows

    observed_routed_expert_hashes: dict[str, str] = {}
    for representation in _REPRESENTATIONS:
        for row_index, raw_row in enumerate(rows_by_representation[representation]):
            routes = _mapping(
                _mapping(
                    raw_row,
                    field=f"arms.{representation}.rows[{row_index}]",
                ).get("routes"),
                field=f"arms.{representation}.rows[{row_index}].routes",
            )
            expert_hashes = _mapping(
                routes.get("expert_hashes"),
                field=(f"arms.{representation}.rows[{row_index}].routes.expert_hashes"),
            )
            for pair, digest in expert_hashes.items():
                if not isinstance(pair, str) or not isinstance(digest, str):
                    raise QualityGateError("routed expert attestation is not canonical")
                previous = observed_routed_expert_hashes.setdefault(pair, digest)
                if previous != digest:
                    raise QualityGateError("cross-arm routed expert hashes differ")
    if dict(sorted(observed_routed_expert_hashes.items())) != dict(
        routed_expert_record_hashes
    ):
        raise QualityGateError(
            "routed expert manifest attestation does not cover exact observed pairs"
        )

    for index, prompt in enumerate(prompts):
        expected_hash = prompt["prompt_sha256"]
        if any(
            _mapping(
                rows_by_representation[representation][index],
                field=f"arms.{representation}.rows[{index}]",
            ).get("prompt_sha256")
            != expected_hash
            for representation in _REPRESENTATIONS
        ):
            raise QualityGateError("cross-arm prompt identity differs")
    bf16_short = _mapping(rows_by_representation["bf16"][0], field="bf16 short")
    q4_short = _mapping(rows_by_representation["q4"][0], field="q4 short")
    q4_dynamic_short = _mapping(
        rows_by_representation["q4_dynamic"][0],
        field="dynamic Q4 short",
    )
    static_dynamic_q4_token_parity = all(
        _mapping(rows_by_representation["q4"][index], field="static Q4 row").get(
            "generated_token_ids"
        )
        == _mapping(
            rows_by_representation["q4_dynamic"][index],
            field="dynamic Q4 row",
        ).get("generated_token_ids")
        for index in range(len(QUALITY_CONTEXTS))
    )
    cross_arm_retrieval_identity = all(
        len(
            {
                canonical_sha256(
                    _mapping(
                        rows_by_representation[representation][index],
                        field=f"{representation} retrieval row",
                    ).get("retrieval")
                )
                for representation in _REPRESENTATIONS
            }
        )
        == 1
        for index in range(len(QUALITY_CONTEXTS))
    )
    gates = {
        "all_retrieval_exact": all(row["retrieval"] for row in row_results),
        "dynamic_page_geometry": all(row["geometry"] for row in row_results),
        "dynamic_target_owner_path": all(row["ownership"] for row in row_results),
        "sequential_reset_and_stale_marker_gate": all(
            row["reset"] and row["retrieval"] for row in row_results
        ),
        "fixed_expert_ownership": fixed_expert_ownership,
        "attention_identity": all(attention_config_results)
        and all(row["attention_path"] for row in row_results),
        "artifact_identity": all(artifact_results)
        and len({canonical_sha256(item) for item in artifact_before_values}) == 1
        and identity.get("payload_hash_verified") is False,
        "route_and_expert_integrity": all(
            row["route_integrity"] for row in row_results
        ),
        "tokenizer_decode_binding": all(row["decode_binding"] for row in row_results),
        "kv_representation_path": all(
            row["representation_path"] for row in row_results
        ),
        "runtime_health": all(row["health"] for row in row_results),
        "short_control_token_parity": (
            bf16_short.get("generated_token_ids")
            == q4_short.get("generated_token_ids")
            == q4_dynamic_short.get("generated_token_ids")
        ),
        "static_dynamic_q4_token_parity": static_dynamic_q4_token_parity,
        "cross_arm_retrieval_identity": cross_arm_retrieval_identity,
    }
    acceptance: dict[str, object] = {
        "passed": all(gates.values()),
        "gates": gates,
        "rejection_reasons": [
            name for name, passed in gates.items() if passed is False
        ],
    }
    if isinstance(value, dict):
        value["acceptance"] = acceptance
    return acceptance


def _paged_entries(cache: Sequence[object]) -> tuple[Any, ...]:
    entries = tuple(
        entry for entry in cache if callable(getattr(entry, "paged_stats", None))
    )
    if not entries:
        raise QualityGateError("runtime cache contains no paged attention entries")
    return entries


def _route_expert_hashes(
    manifest: Any,
    route_trace: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    pairs: set[tuple[int, int]] = set()
    for entry in route_trace:
        layer = entry.get("layer")
        expert_ids = entry.get("expert_ids")
        if isinstance(layer, int) and isinstance(expert_ids, Sequence):
            pairs.update((layer, int(expert)) for expert in expert_ids)
    if not pairs:
        raise QualityGateError("generation produced no exact expert routes")
    result: dict[str, str] = {}
    for layer, expert in sorted(pairs):
        record = manifest.record(layer, expert)
        digest = getattr(record, "sha256", None)
        result[f"{layer}:{expert}"] = _hex_digest(
            digest,
            field=f"expert SHA-256 {layer}:{expert}",
        )
    return result


def _build_route_evidence(
    *,
    manifest: Any,
    route_trace: Sequence[Mapping[str, object]],
    expert_manifest_sha256: str,
    expert_manifest_content_sha256: str,
) -> dict[str, object]:
    trace = [dict(entry) for entry in route_trace]
    expert_hashes = _route_expert_hashes(manifest, trace)
    binding_core: dict[str, object] = {
        "expert_manifest_sha256": _hex_digest(
            expert_manifest_sha256,
            field="expert manifest file SHA-256",
        ),
        "expert_manifest_content_sha256": _hex_digest(
            expert_manifest_content_sha256,
            field="expert manifest content SHA-256",
        ),
        "expert_hashes_sha256": canonical_sha256(expert_hashes),
        "routed_pair_count": len(expert_hashes),
        "live_manifest_records_verified": True,
        "verification_scope": ROUTE_VERIFICATION_SCOPE,
    }
    return {
        "trace": trace,
        "trace_sha256": canonical_sha256(trace),
        "expert_hashes": expert_hashes,
        "manifest_binding": {
            **binding_core,
            "binding_sha256": canonical_sha256(binding_core),
        },
    }


def _build_routed_expert_attestation(
    arms: Sequence[Mapping[str, object]],
    *,
    expert_manifest_content_sha256: str,
) -> dict[str, object]:
    record_hashes: dict[str, str] = {}
    for arm_index, arm in enumerate(arms):
        rows = _sequence(arm.get("rows"), field=f"arms[{arm_index}].rows")
        for row_index, row in enumerate(rows):
            routes = _mapping(
                _mapping(row, field=f"arms[{arm_index}].rows[{row_index}]").get(
                    "routes"
                ),
                field=f"arms[{arm_index}].rows[{row_index}].routes",
            )
            expert_hashes = _mapping(
                routes.get("expert_hashes"),
                field=f"arms[{arm_index}].rows[{row_index}].routes.expert_hashes",
            )
            for pair, raw_digest in expert_hashes.items():
                if not isinstance(pair, str):
                    raise QualityGateError("routed expert pair key must be text")
                digest = _hex_digest(
                    raw_digest,
                    field=f"routed expert SHA-256 {pair}",
                )
                if digest == "0" * 64:
                    raise QualityGateError("routed expert SHA-256 cannot be zero")
                previous = record_hashes.setdefault(pair, digest)
                if previous != digest:
                    raise QualityGateError(
                        "cross-arm routed expert hashes contradict the live manifest"
                    )
    if not record_hashes:
        raise QualityGateError("quality arms contain no routed expert records")
    record_hashes = dict(sorted(record_hashes.items()))
    record_hashes_sha256 = canonical_sha256(record_hashes)
    core: dict[str, object] = {
        "schema": ROUTED_EXPERT_ATTESTATION_SCHEMA,
        "expert_manifest_content_sha256": _hex_digest(
            expert_manifest_content_sha256,
            field="expert manifest content SHA-256",
        ),
        "record_hashes": record_hashes,
        "record_hashes_sha256": record_hashes_sha256,
        "routed_pair_count": len(record_hashes),
        "producer_manifest_records_verified": True,
        "verification_scope": ROUTE_VERIFICATION_SCOPE,
    }
    return {**core, "binding_sha256": canonical_sha256(core)}


def _validate_routed_expert_attestation(
    value: object,
    *,
    field: str,
    expert_manifest_content_sha256: str,
) -> dict[str, object]:
    from mtplx.expert_streaming_models import HY3_Q4

    attestation = dict(_mapping(value, field=field))
    expected_keys = {
        "schema",
        "expert_manifest_content_sha256",
        "record_hashes",
        "record_hashes_sha256",
        "routed_pair_count",
        "producer_manifest_records_verified",
        "verification_scope",
        "binding_sha256",
    }
    if set(attestation) != expected_keys:
        raise QualityGateError(f"{field} keys differ")
    raw_hashes = _mapping(
        attestation.get("record_hashes"), field=f"{field}.record_hashes"
    )
    record_hashes: dict[str, str] = {}
    for pair, raw_digest in raw_hashes.items():
        if not isinstance(pair, str) or pair.count(":") != 1:
            raise QualityGateError(f"{field} contains an invalid routed pair")
        layer, expert = pair.split(":")
        try:
            normalized_pair = f"{int(layer)}:{int(expert)}"
        except ValueError as exc:
            raise QualityGateError(
                f"{field} contains a non-integer routed pair"
            ) from exc
        if (
            normalized_pair != pair
            or int(layer) not in HY3_Q4.routed_layer_indices
            or not 0 <= int(expert) < HY3_Q4.expert_count
        ):
            raise QualityGateError(f"{field} contains a non-canonical routed pair")
        digest = _hex_digest(raw_digest, field=f"{field}.record_hashes.{pair}")
        if digest == "0" * 64:
            raise QualityGateError(f"{field} contains a zero expert SHA-256")
        record_hashes[pair] = digest
    record_hashes = dict(sorted(record_hashes.items()))
    record_hashes_sha256 = canonical_sha256(record_hashes)
    if (
        attestation.get("schema") != ROUTED_EXPERT_ATTESTATION_SCHEMA
        or attestation.get("expert_manifest_content_sha256")
        != expert_manifest_content_sha256
        or attestation.get("record_hashes_sha256") != record_hashes_sha256
        or _exact_int(
            attestation.get("routed_pair_count"),
            field=f"{field}.routed_pair_count",
            minimum=1,
        )
        != len(record_hashes)
        or attestation.get("producer_manifest_records_verified") is not True
        or attestation.get("verification_scope") != ROUTE_VERIFICATION_SCOPE
    ):
        raise QualityGateError(f"{field} contradicts the pinned manifest binding")
    core = {
        name: attestation[name] for name in expected_keys if name != "binding_sha256"
    }
    core["record_hashes"] = record_hashes
    if _hex_digest(
        attestation.get("binding_sha256"), field=f"{field}.binding_sha256"
    ) != canonical_sha256(core):
        raise QualityGateError(f"{field} binding SHA-256 differs")
    return {**core, "binding_sha256": attestation["binding_sha256"]}


def _runtime_health(runtime: Any) -> dict[str, int]:
    slot_pool = runtime.expert_streaming.slots
    health = _mapping(
        slot_pool.health_telemetry_snapshot(),
        field="expert slot health",
    )
    resource = _mapping(
        slot_pool.resource_telemetry_snapshot(),
        field="expert slot resources",
    )
    states = _mapping(health.get("states"), field="expert slot states")
    metrics = _mapping(health.get("metrics"), field="expert slot metrics")
    io = _mapping(resource.get("io"), field="expert slot io")
    return {
        "active_routes": _exact_int(
            metrics.get("active_routes"), field="health.active_routes"
        ),
        "pins": _exact_int(health.get("pins"), field="health.pins"),
        "loading": _exact_int(states.get("loading"), field="health.loading"),
        "failed": _exact_int(states.get("failed"), field="health.failed"),
        "integrity_errors": _exact_int(
            io.get("integrity_errors"), field="health.integrity_errors"
        ),
        "completion_fence_failures": _exact_int(
            metrics.get("completion_fence_failures"),
            field="health.completion_fence_failures",
        ),
        "global_device_synchronizations": _exact_int(
            metrics.get("global_device_synchronizations"),
            field="health.global_device_synchronizations",
        ),
    }


def _kv_evidence(
    entries: Sequence[Any],
    *,
    representation: str,
    context_tokens: int,
    environment: Mapping[str, str],
    initial_stats: Sequence[Mapping[str, object]] | None = None,
    runtime: Any | None = None,
) -> dict[str, object]:
    def storage_dtype(entry: Any, attribute: str) -> str | None:
        array = getattr(entry, attribute, None)
        if array is None:
            return None
        dtype = getattr(array, "dtype", None)
        if dtype is None:
            raise QualityGateError(f"live paged {attribute} has no dtype")
        return str(dtype).rsplit(".", 1)[-1]

    stats: list[dict[str, object]] = []
    for entry in entries:
        entry_stats = dict(entry.paged_stats())
        entry_stats["storage_dtypes"] = {
            "key": storage_dtype(entry, "key_cache"),
            "value": storage_dtype(entry, "value_cache"),
            "key_scale": storage_dtype(entry, "key_scale_cache"),
            "value_scale": storage_dtype(entry, "value_scale_cache"),
        }
        stats.append(entry_stats)
    pages = page_slots_for_context(context_tokens)
    capacity = pages * KV_BLOCK_SIZE_TOKENS
    physical_bytes = sum(
        _exact_int(entry.get("bytes"), field="paged entry bytes", minimum=1)
        for entry in stats
    )
    initial_values = [dict(value) for value in initial_stats or ()]
    if len(initial_values) != len(stats):
        raise QualityGateError("initial paged entry evidence count differs")
    initial_evidence = {
        "entry_count": len(initial_values),
        "num_blocks": [int(value.get("num_blocks", -1)) for value in initial_values],
        "entry_offsets": [int(value.get("offset", -1)) for value in initial_values],
        "entry_bytes": [int(value.get("bytes", -1)) for value in initial_values],
        "physical_bytes": sum(int(value.get("bytes", -1)) for value in initial_values),
    }
    dynamic = representation == "q4_dynamic"
    owners: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        observer = getattr(entry, "allocation_observer", None)
        if observer is None:
            continue
        allocations = tuple(getattr(entry, "_kv_allocations", ()))
        owners.append(
            {
                "entry_index": index,
                "cache_id": str(getattr(entry, "cache_id", "")),
                "allocation_count": len(allocations),
                "committed_physical_bytes": int(
                    getattr(entry, "_committed_physical_bytes", -1)
                ),
                "allocation_handle_bytes": sum(
                    int(getattr(allocation, "physical_bytes", -1))
                    for allocation in allocations
                ),
            }
        )
    expert_runtime = getattr(runtime, "expert_streaming", None)
    broker = getattr(expert_runtime, "memory_broker", None)
    if broker is None:
        broker_evidence = None
    else:
        snapshot = broker.snapshot()
        broker_evidence = {
            "kv_physical_bytes": int(snapshot.kv_physical_bytes),
            "owned_kv_physical_bytes": int(snapshot.owned_kv_physical_bytes),
            "unreconciled_kv_physical_bytes": int(
                snapshot.unreconciled_kv_physical_bytes
            ),
            "pending_kv_ticket_id": snapshot.pending_kv_ticket_id,
            "failed_closed": bool(snapshot.failed_closed),
        }
    return {
        "representation": representation,
        "bytes_per_token": _bytes_per_token(representation),
        "block_size_tokens": KV_BLOCK_SIZE_TOKENS,
        "required_page_slots": pages,
        "physical_capacity_tokens": capacity,
        "physical_bytes": physical_bytes,
        "entry_count": len(stats),
        "initial": initial_evidence,
        "attention_environment": dict(environment),
        "entries": stats,
        "ownership": {
            "policy": "brokered_target_group" if dynamic else "unbrokered_static",
            "owners": owners,
            "broker": broker_evidence,
        },
    }


def _close_and_attest_reset(
    runtime: Any,
    cache: list[object],
    entries: Sequence[Any],
    admission: Any,
    *,
    representation: str,
) -> dict[str, object]:
    first_error: BaseException | None = None
    for entry in entries:
        try:
            entry.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    offsets = [int(entry.paged_stats()["offset"]) for entry in entries]
    byte_counts = [int(entry.paged_stats()["bytes"]) for entry in entries]
    closed = [bool(getattr(entry, "_closed", False)) for entry in entries]
    cache.clear()
    try:
        admission.release()
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    expert_runtime = runtime.expert_streaming
    with expert_runtime._kv_lock:
        live_tokens = _exact_int(
            expert_runtime._live_kv_tokens,
            field="post-reset live_kv_tokens",
        )
    broker = getattr(expert_runtime, "memory_broker", None)
    if broker is None:
        broker_release = None
    else:
        snapshot = broker.snapshot()
        broker_release = {
            "kv_physical_bytes": int(snapshot.kv_physical_bytes),
            "owned_kv_physical_bytes": int(snapshot.owned_kv_physical_bytes),
            "unreconciled_kv_physical_bytes": int(
                snapshot.unreconciled_kv_physical_bytes
            ),
            "pending_kv_ticket_id": snapshot.pending_kv_ticket_id,
            "failed_closed": bool(snapshot.failed_closed),
        }
    broker_released = (
        broker_release is None
        if representation != "q4_dynamic"
        else broker_release
        == {
            "kv_physical_bytes": 0,
            "owned_kv_physical_bytes": 0,
            "unreconciled_kv_physical_bytes": 0,
            "pending_kv_ticket_id": None,
            "failed_closed": False,
        }
    )
    evidence = {
        "entry_count": len(entries),
        "entry_offsets": offsets,
        "entry_bytes": byte_counts,
        "entries_closed": closed,
        "live_kv_tokens": live_tokens,
        "health": _runtime_health(runtime),
        "broker_release": broker_release,
        "passed": (
            all(offset == 0 for offset in offsets)
            and all(count == 0 for count in byte_counts)
            and all(closed)
            and live_tokens == 0
            and broker_released
            and first_error is None
        ),
    }
    if first_error is not None:
        raise first_error
    return evidence


def _run_quality_row(
    *,
    runtime: Any,
    manifest: Any,
    representation: str,
    prompt: QualityPrompt,
    forbidden_markers: Sequence[str],
    prefill_chunk_size: int,
    tokenizer_identity: Mapping[str, object],
    expert_manifest_sha256: str,
    expert_manifest_content_sha256: str,
) -> dict[str, object]:
    environment_values = quality_environment(
        representation,
        context_tokens=prompt.context_tokens,
        prefill_chunk_size=prefill_chunk_size,
    )
    environment = _EnvironmentLease(environment_values)
    environment.acquire()
    admission: Any | None = None
    cache: list[object] | None = None
    entries: tuple[Any, ...] = ()
    try:
        admission = runtime.admit_kv_tokens(prompt.context_tokens)
        admission.__enter__()
        cache = runtime.make_cache()
        if not isinstance(cache, list):
            cache = list(cache)
        entries = _paged_entries(cache)
        initial_stats = [dict(entry.paged_stats()) for entry in entries]
        if representation == "q4_dynamic":
            if any(
                int(stats.get("num_blocks", 0)) != 1
                or int(stats.get("offset", -1)) != 0
                or int(stats.get("bytes", 0)) <= 0
                for stats in initial_stats
            ):
                raise QualityGateError(
                    "new dynamic Q4 cache did not start at one owned page"
                )
        elif any(
            int(stats.get("num_blocks", 0))
            != page_slots_for_context(prompt.context_tokens)
            or int(stats.get("offset", -1)) != 0
            or int(stats.get("bytes", -1)) != 0
            for stats in initial_stats
        ):
            raise QualityGateError(
                "new static paged cache did not start at zero physical bytes"
            )
        route_start = len(runtime.expert_streaming.route_trace())
        started = time.perf_counter()
        logits = _forward_prefill(
            runtime,
            cache,
            prompt.token_ids,
            chunk_size=prefill_chunk_size,
        )
        generated_ids, _logits, _latencies, _elapsed = _decode_tokens(
            runtime,
            cache,
            logits,
            count=COMPLETION_RESERVE_TOKENS,
        )
        elapsed = time.perf_counter() - started
        if elapsed <= 0:
            raise QualityGateError("quality row timer did not advance")
        generated_text = _decode_text(runtime.tokenizer, generated_ids)
        retrieval = score_retrieval_text(
            generated_text,
            expected_earliest=prompt.earliest_marker,
            expected_latest=prompt.latest_marker,
            forbidden_markers=forbidden_markers,
        )
        trace = [
            dict(entry)
            for entry in runtime.expert_streaming.route_trace()[route_start:]
        ]
        row: dict[str, object] = {
            "context_tokens": prompt.context_tokens,
            "prompt_tokens": prompt.prompt_tokens,
            "completion_tokens": COMPLETION_RESERVE_TOKENS,
            "prompt_sha256": prompt.prompt_sha256,
            "generated_token_ids": generated_ids,
            "generated_token_sha256": _prompt_digest(generated_ids),
            "generated_text": generated_text,
            "generated_text_sha256": hashlib.sha256(
                generated_text.encode()
            ).hexdigest(),
            "decode_binding": _decode_binding(
                tokenizer_identity=tokenizer_identity,
                generated_token_ids=generated_ids,
                generated_text=generated_text,
            ),
            "retrieval": retrieval,
            "kv": _kv_evidence(
                entries,
                representation=representation,
                context_tokens=prompt.context_tokens,
                environment=environment_values,
                initial_stats=initial_stats,
                runtime=runtime,
            ),
            "routes": _build_route_evidence(
                manifest=manifest,
                route_trace=trace,
                expert_manifest_sha256=expert_manifest_sha256,
                expert_manifest_content_sha256=expert_manifest_content_sha256,
            ),
            "health": _runtime_health(runtime),
            "elapsed_seconds": elapsed,
        }
        row["reset"] = _close_and_attest_reset(
            runtime,
            cache,
            entries,
            admission,
            representation=representation,
        )
        admission = None
        cache = None
        return row
    finally:
        if cache is not None:
            for entry in entries:
                try:
                    entry.close()
                except BaseException:
                    pass
            cache.clear()
        if admission is not None:
            try:
                admission.release()
            except BaseException:
                pass
        environment.release()


def _run_quality_representation(
    *,
    hardware_config: Hy3HardwareConfig,
    representation: str,
    prompts: Sequence[QualityPrompt] | None,
) -> tuple[dict[str, object], list[QualityPrompt], dict[str, object]]:
    from mtplx.runtime import load

    mode = _representation(representation)
    environment = _EnvironmentLease(
        quality_environment(
            mode,
            context_tokens=max(QUALITY_CONTEXTS),
            prefill_chunk_size=hardware_config.prefill_chunk_size,
        )
    )
    environment.acquire()
    runtime: Any | None = None
    try:
        before = _attest_config_artifact(
            hardware_config,
            verify_payload_hash=False,
        )
        runtime_config = build_quality_runtime_config(hardware_config, mode)
        runtime = load(
            hardware_config.model_root,
            mtp=False,
            expert_streaming_config=runtime_config,
            expert_manifest=before.manifest,
            model_config=before.model_config,
        )
        _require_loaded_manifest_identity(
            before.manifest,
            runtime.expert_streaming.manifest,
        )
        tokenizer_identity = _attest_tokenizer_identity(
            runtime.tokenizer,
            hardware_config.model_root,
        )
        expert_manifest_sha256 = _hex_digest(
            before.manifest_file_sha256,
            field="expert manifest file SHA-256",
        )
        expert_manifest_content_sha256 = _hex_digest(
            before.manifest.manifest_sha256,
            field="expert manifest content SHA-256",
        )
        if prompts is None:
            exact_prompts = [
                build_quality_prompt(runtime.tokenizer, context_tokens=context)
                for context in QUALITY_CONTEXTS
            ]
        else:
            exact_prompts = list(prompts)
            for prompt in exact_prompts:
                decoded = _decode_text(runtime.tokenizer, prompt.token_ids)
                if decoded != prompt.rendered_prompt:
                    raise QualityGateError(
                        "cross-arm tokenizer decoded the exact prompt differently"
                    )
        rows: list[dict[str, object]] = []
        for prompt_index, prompt in enumerate(exact_prompts):
            forbidden = [
                _marker(prior.context_tokens, edge)
                for prior in exact_prompts[:prompt_index]
                for edge in ("earliest", "latest")
            ]
            rows.append(
                _run_quality_row(
                    runtime=runtime,
                    manifest=before.manifest,
                    representation=mode,
                    prompt=prompt,
                    forbidden_markers=forbidden,
                    prefill_chunk_size=hardware_config.prefill_chunk_size,
                    tokenizer_identity=tokenizer_identity,
                    expert_manifest_sha256=expert_manifest_sha256,
                    expert_manifest_content_sha256=(expert_manifest_content_sha256),
                )
            )
        plan = runtime_config.memory_plan(runtime.expert_streaming.spec)
        plan_evidence = {
            "kv_bytes_per_token": _bytes_per_token(mode),
            "expert_cache_limit_bytes": EXPERT_CACHE_LIMIT_BYTES,
            "planned_persistent_slots": int(plan.persistent_slots),
            "persistent_cache_bytes": int(plan.persistent_cache_bytes),
            "attention_identity": _attention_identity(mode),
            "runtime_config": runtime_config.to_dict(),
            "runtime_config_sha256": canonical_sha256(runtime_config.to_dict()),
        }
    finally:
        try:
            if runtime is not None:
                runtime.close(timeout=30.0)
        finally:
            environment.release()
    after = _attest_config_artifact(hardware_config, verify_payload_hash=False)
    arm = {
        "representation": mode,
        "model_artifact_id": hardware_config.model_artifact_id,
        "source_git_commit": _source_commit(hardware_config.repo_root),
        "expert_manifest_content_sha256": expert_manifest_content_sha256,
        "tokenizer_identity": tokenizer_identity,
        "artifact_before": before.evidence(),
        "artifact_after": after.evidence(),
        "rows": rows,
    }
    return arm, exact_prompts, plan_evidence


def run_hy3_kv_quality(config: Hy3HardwareConfig) -> dict[str, object]:
    """Run BF16, static Q4, then dynamic Q4 as one validated quality lane."""

    bf16_arm, prompts, bf16_plan = _run_quality_representation(
        hardware_config=config,
        representation="bf16",
        prompts=None,
    )
    q4_arm, q4_prompts, q4_plan = _run_quality_representation(
        hardware_config=config,
        representation="q4",
        prompts=prompts,
    )
    q4_dynamic_arm, q4_dynamic_prompts, q4_dynamic_plan = _run_quality_representation(
        hardware_config=config,
        representation="q4_dynamic",
        prompts=prompts,
    )
    expected_prompt_ids = [prompt.token_ids for prompt in prompts]
    if [prompt.token_ids for prompt in q4_prompts] != expected_prompt_ids or [
        prompt.token_ids for prompt in q4_dynamic_prompts
    ] != expected_prompt_ids:
        raise QualityGateError("cross-arm prompt token identity changed")
    arm_values = (bf16_arm, q4_arm, q4_dynamic_arm)
    expert_manifest_content_sha256 = str(bf16_arm["expert_manifest_content_sha256"])
    routed_expert_attestation = _build_routed_expert_attestation(
        arm_values,
        expert_manifest_content_sha256=expert_manifest_content_sha256,
    )
    for arm in arm_values:
        arm["routed_expert_attestation_sha256"] = routed_expert_attestation[
            "binding_sha256"
        ]
    bf16_artifact = _mapping(
        bf16_arm["artifact_before"], field="BF16 artifact evidence"
    )
    identity = {
        "model_artifact_id": config.model_artifact_id,
        "model_artifact_sha256": bf16_artifact["model_artifact_sha256"],
        "expert_manifest_sha256": bf16_artifact["expert_manifest_sha256"],
        "expert_manifest_content_sha256": expert_manifest_content_sha256,
        "artifact_pins_sha256": bf16_artifact["artifact_pins_sha256"],
        "artifact_stat_sha256": bf16_artifact["artifact_stat_sha256"],
        "resident_payload_bytes": bf16_artifact["resident_payload_bytes"],
        "resident_payload_sha256": bf16_artifact["resident_payload_sha256"],
        "source_git_commit": bf16_arm["source_git_commit"],
        "payload_hash_verified": False,
        "tokenizer_identity": bf16_arm["tokenizer_identity"],
        "routed_expert_attestation": routed_expert_attestation,
    }
    result: dict[str, object] = {
        "schema": HY3_KV_QUALITY_SCHEMA,
        "identity": identity,
        "configuration": {
            "contexts": list(QUALITY_CONTEXTS),
            "completion_reserve_tokens": COMPLETION_RESERVE_TOKENS,
            "expert_cache_limit_bytes": EXPERT_CACHE_LIMIT_BYTES,
            "prefill_chunk_size": config.prefill_chunk_size,
            "representations": {
                "bf16": bf16_plan,
                "q4": q4_plan,
                "q4_dynamic": q4_dynamic_plan,
            },
        },
        "prompts": [prompt.to_dict() for prompt in prompts],
        "arms": {
            "bf16": bf16_arm,
            "q4": q4_arm,
            "q4_dynamic": q4_dynamic_arm,
        },
    }
    validate_quality_result(result)
    return result


def _load_hardware_config(path: Path) -> Hy3HardwareConfig:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityGateError(f"cannot load hooks config {path}: {exc}") from exc
    mapping = _mapping(value, field="hooks config")
    return Hy3HardwareConfig.from_mapping(mapping)


def _run_capturing_stdout(
    config: Hy3HardwareConfig,
) -> tuple[dict[str, object], str]:
    """Capture Python and file-descriptor stdout so only final JSON escapes."""

    def flush_c_stdout() -> None:
        try:
            function = ctypes.CDLL(None).fflush
            function.argtypes = (ctypes.c_void_p,)
            function.restype = ctypes.c_int
            function(None)
        except (AttributeError, OSError):
            pass

    python_stdout = io.StringIO()
    sys.stdout.flush()
    flush_c_stdout()
    saved_descriptor = os.dup(1)
    result: dict[str, object]
    try:
        with tempfile.TemporaryFile(mode="w+b") as native_stdout:
            os.dup2(native_stdout.fileno(), 1)
            try:
                with contextlib.redirect_stdout(python_stdout):
                    result = run_hy3_kv_quality(config)
            finally:
                flush_c_stdout()
                os.dup2(saved_descriptor, 1)
            native_stdout.seek(0)
            native_payload = native_stdout.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved_descriptor)
    return result, python_stdout.getvalue() + native_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Run the Hy3 BF16/static-Q4/dynamic-Q4 KV retrieval-quality gate.")
    )
    parser.add_argument("--hooks-config", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    output_path = (
        None if args.output_json is None else args.output_json.expanduser().resolve()
    )
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"quality evidence already exists: {output_path}")
    config = _load_hardware_config(args.hooks_config)
    result, captured_stdout = _run_capturing_stdout(config)
    if captured_stdout:
        captured_bytes = captured_stdout.encode()
        tail_bytes = captured_bytes[-4096:]
        result["diagnostics"] = {
            "captured_runtime_stdout_bytes": len(captured_bytes),
            "captured_runtime_stdout_sha256": hashlib.sha256(
                captured_bytes
            ).hexdigest(),
            "captured_runtime_stdout_tail": tail_bytes.decode(
                "utf-8", errors="replace"
            ),
            "captured_runtime_stdout_tail_bytes": len(tail_bytes),
            "captured_runtime_stdout_truncated": len(captured_bytes) > len(tail_bytes),
        }
    acceptance = validate_quality_result(result)
    rendered = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if output_path is not None:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    sys.stdout.write(rendered)
    return 0 if acceptance["passed"] is True else 2


__all__ = [
    "ATTENTION_IDENTITY",
    "BF16_KV_BYTES_PER_TOKEN",
    "COMPLETION_RESERVE_TOKENS",
    "EXPERT_CACHE_LIMIT_BYTES",
    "GIB",
    "HY3_KV_QUALITY_SCHEMA",
    "Q4_KV_BYTES_PER_TOKEN",
    "QUALITY_CONTEXTS",
    "QualityGateError",
    "QualityPrompt",
    "build_quality_prompt",
    "build_quality_runtime_config",
    "main",
    "page_slots_for_context",
    "quality_environment",
    "run_hy3_kv_quality",
    "score_retrieval_text",
    "validate_quality_result",
]
