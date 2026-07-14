from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from mtplx.hy3_q4_context import HY3_Q4_CONTEXT_WINDOW


KV_QUANT_MODES = ("off", "q8", "q4")


@dataclass(frozen=True)
class ResolvedAPIKey:
    value: str | None
    source: str

    @property
    def required(self) -> bool:
        return bool(self.value)


def normalize_paged_kv_quantization(value: object | None, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        return "off"
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "false", "0", "disabled", "disable"):
        return "off"
    if raw in ("off", "q8", "q4"):
        return raw
    if raw in ("8", "8bit", "int8", "uint8", "q8_0"):
        return "q8"
    if raw in ("4", "4bit", "int4", "uint4", "q4_0"):
        return "q4"
    choices = ", ".join(KV_QUANT_MODES)
    raise ValueError(f"unsupported paged KV quantization mode {value!r}; expected one of: {choices}")


def paged_kv_quantization_env(mode: object | None) -> dict[str, str]:
    canonical = normalize_paged_kv_quantization(mode)
    return {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": canonical,
        "MTPLX_PAGED_KV_QUANT": canonical,
    }


def apply_paged_kv_quantization_env(mode: object | None, env: dict[str, str] | None = None) -> str:
    canonical = normalize_paged_kv_quantization(mode)
    target = os.environ if env is None else env
    target.update(paged_kv_quantization_env(canonical))
    return canonical


def validate_hy3_q4_dynamic_context_options(
    args: object,
    expert_streaming_config: object | None,
    *,
    resolved_context_window: int | None = None,
) -> bool:
    """Validate the explicit, single-sequence Hy3 Q4 128K server lane."""

    if not bool(getattr(args, "hy3_q4_dynamic_context", False)):
        return False
    if (
        normalize_paged_kv_quantization(getattr(args, "paged_kv_quantization", "off"))
        != "q4"
    ):
        raise ValueError("--hy3-q4-dynamic-context requires --paged-kv-quantization q4")
    if int(getattr(args, "context_window", 0) or 0) != HY3_Q4_CONTEXT_WINDOW:
        raise ValueError("--hy3-q4-dynamic-context requires --context-window 131072")
    if (
        resolved_context_window is not None
        and int(resolved_context_window) != HY3_Q4_CONTEXT_WINDOW
    ):
        raise ValueError(
            "--hy3-q4-dynamic-context requires the loaded model to resolve "
            "a 131072-token context window"
        )
    if str(getattr(args, "scheduler_mode", "serial") or "serial") != "serial":
        raise ValueError("--hy3-q4-dynamic-context requires --scheduler-mode serial")

    max_active = getattr(args, "max_active_requests", None)
    decode_max = getattr(args, "decode_batch_max", None)
    if type(max_active) is not int or max_active != 1:
        raise ValueError(
            "--hy3-q4-dynamic-context requires --max-active-requests 1 exactly"
        )
    if type(decode_max) is not int or decode_max != 1:
        raise ValueError(
            "--hy3-q4-dynamic-context requires --decode-batch-max 1 exactly"
        )
    if bool(getattr(args, "session_bank_live_refs", True)):
        raise ValueError(
            "--hy3-q4-dynamic-context requires --no-session-bank-live-refs"
        )

    if (
        expert_streaming_config is None
        or getattr(expert_streaming_config, "model_key", None) != "hy3-q4"
    ):
        raise ValueError("--hy3-q4-dynamic-context requires Hy3 Q4 expert streaming")
    if (
        getattr(expert_streaming_config, "cache_scope", None) != "global"
        or getattr(expert_streaming_config, "slot_layout", None) != "component-banks"
    ):
        raise ValueError(
            "--hy3-q4-dynamic-context requires the global component-bank lane"
        )
    return True


_HY3_Q4_DYNAMIC_MEMORY_TUNING_FLAGS = (
    ("expert_slab_slots", "--expert-slab-slots"),
    (
        "expert_regrow_hysteresis_slabs",
        "--expert-regrow-hysteresis-slabs",
    ),
    ("expert_resize_min_interval_ms", "--expert-resize-min-interval-ms"),
)


def validate_hy3_q4_dynamic_memory_request_options(args: object) -> bool:
    """Reject ambiguous serving requests before reading or loading model weights."""

    enabled = bool(getattr(args, "hy3_q4_dynamic_memory", False))
    if not enabled:
        for attribute, flag in _HY3_Q4_DYNAMIC_MEMORY_TUNING_FLAGS:
            if getattr(args, attribute, None) is not None:
                raise ValueError(f"{flag} requires --hy3-q4-dynamic-memory")
        return False
    if not bool(getattr(args, "hy3_q4_dynamic_context", False)):
        raise ValueError("--hy3-q4-dynamic-memory requires --hy3-q4-dynamic-context")
    if not bool(
        getattr(args, "expert_streaming", False)
        or getattr(args, "expert_streaming_config", None)
        or getattr(args, "expert_manifest", None)
    ):
        raise ValueError("--hy3-q4-dynamic-memory requires expert streaming")
    return True


def validate_hy3_q4_dynamic_memory_options(
    args: object,
    expert_streaming_config: object | None,
) -> bool:
    """Attest the explicit serving opt-in and its resolved runtime config."""

    enabled = validate_hy3_q4_dynamic_memory_request_options(args)
    config_is_dynamic = bool(
        getattr(expert_streaming_config, "dynamic_expert_slabs", False)
    )
    if not enabled:
        if config_is_dynamic:
            raise ValueError(
                "dynamic expert slabs in serving require --hy3-q4-dynamic-memory"
            )
        return False
    if expert_streaming_config is None:
        raise ValueError("--hy3-q4-dynamic-memory requires expert streaming")
    if not config_is_dynamic:
        raise ValueError("--hy3-q4-dynamic-memory requires dynamic_expert_slabs=True")
    if not bool(getattr(expert_streaming_config, "resource_telemetry", False)):
        raise ValueError("--hy3-q4-dynamic-memory requires resource_telemetry=True")
    for attribute, flag in _HY3_Q4_DYNAMIC_MEMORY_TUNING_FLAGS:
        requested = getattr(args, attribute, None)
        if (
            requested is not None
            and getattr(expert_streaming_config, attribute, None) != requested
        ):
            raise ValueError(f"{flag} was not applied to ExpertStreamingConfig")
    return True


def resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    api_key_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAPIKey:
    explicit = _clean_secret(explicit_api_key)
    if explicit:
        return ResolvedAPIKey(explicit, "flag")

    if api_key_file:
        path = Path(api_key_file).expanduser()
        secret = _clean_secret(path.read_text(encoding="utf-8"))
        if not secret:
            raise ValueError(f"API key file is empty: {path}")
        return ResolvedAPIKey(secret, "file")

    source_env = os.environ if env is None else env
    api_key = _clean_secret(source_env.get("MTPLX_API_KEY"))
    if api_key:
        return ResolvedAPIKey(api_key, "env:MTPLX_API_KEY")

    legacy = _clean_secret(source_env.get("MTPLX_AUTH"))
    if legacy:
        return ResolvedAPIKey(legacy, "env:MTPLX_AUTH")

    return ResolvedAPIKey(None, "none")


def _clean_secret(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
