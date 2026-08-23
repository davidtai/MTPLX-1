"""Measured Qwen 3.8 27B optimization route and artifact contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .draft_lm_head import configure_qwen38_row10_compact_head
from .gdn_capture import configure_qwen38_row18_gdn_decay_memo
from .qwen38_challenge_kernels import configure_qwen38_row18_mlp_gate_up
from .qwen38_source_proposal import configure_qwen38_source_proposal

QWEN38_Q8_LINEAR_ATTN_LAYERS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45,
    46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62,
)
QWEN38_PACKING = "mlx_affine_u32_le"
DEFAULT_QWEN38_CACHE_ROUTE = "kv_only_history"
QWEN38_KV_ONLY_MIN_CONTEXT = 16_384
QWEN38_FINAL_ROUTE: Mapping[str, Any] = MappingProxyType(
    {
        "cache_route": DEFAULT_QWEN38_CACHE_ROUTE,
        "dual_norm": True,
        "source_proposal": True,
        "source_retain_control": False,
    }
)


def qwen38_final_route() -> dict[str, Any]:
    """Return the cumulative winner stack measured at 16K context."""

    route = dict(QWEN38_FINAL_ROUTE)
    if os.environ.get("MTPLX_QWEN38_DISABLE_SOURCE_AUTO") == "1":
        route.pop("source_proposal", None)
        route.pop("source_retain_control", None)
    return route


class Qwen38ContractError(RuntimeError):
    """The requested Qwen 3.8 route does not match its measured contract."""


@dataclass(frozen=True)
class Qwen38ModelContract:
    contract_id: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    dtype: str
    trunk_bits: int
    trunk_group_size: int
    trunk_mode: str
    packing: str


@dataclass(frozen=True)
class Qwen38RouteBindings:
    mtp_cache_append: Callable[..., Any]


@dataclass(frozen=True)
class Qwen38RouteSpec:
    route_id: str
    contract: Qwen38ModelContract
    bindings: Qwen38RouteBindings
    kernel_ids: tuple[str, ...] = ()
    min_context_tokens: int = 0
    policy_id: str = "current_mtplx"
    selfcheck_status: str = "control"
    selfcheck_passed: bool = True

    @property
    def fingerprint(self) -> str:
        payload = {
            "contract_id": self.contract.contract_id,
            "kernel_ids": list(self.kernel_ids),
            "min_context_tokens": self.min_context_tokens,
            "policy_id": self.policy_id,
            "route_id": self.route_id,
            "selfcheck_passed": self.selfcheck_passed,
            "selfcheck_status": self.selfcheck_status,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _qwen38_identity(config: Mapping[str, Any], model_path: Path) -> bool:
    runtime = config.get("mtplx_runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    sources = (
        str(model_path),
        str(runtime.get("base_trunk") or ""),
        str(runtime.get("source_repo") or ""),
        str(runtime.get("public_model_id") or ""),
    )
    return any(
        re.search(r"qwen(?:3)?[.\-_]?8[^/]*27b", source, re.IGNORECASE)
        for source in sources
    )


def is_qwen38_27b_candidate(config: Mapping[str, Any], model_path: Path) -> bool:
    """Return whether this is the one measured Optimized-Speed artifact."""

    runtime = config.get("mtplx_runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    sources = (
        str(model_path),
        str(model_path.resolve()),
        str(runtime.get("public_model_id") or ""),
    )
    return any(
        re.search(
            r"(?:qwen3[.]8-27b-mtplx-optimized-speed|"
            r"mtplx-qwen38-27b-optimized-speed)$",
            source,
            re.IGNORECASE,
        )
        for source in sources
    )


def _expected_q8_overrides() -> set[str]:
    names = {
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    }
    names.update(
        f"language_model.model.layers.{layer}.linear_attn.out_proj"
        for layer in QWEN38_Q8_LINEAR_ATTN_LAYERS
    )
    names.update(
        f"language_model.model.layers.{layer}.mlp.{projection}"
        for layer in range(56, 64)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    return names


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise Qwen38ContractError(
            f"Qwen 3.8 contract {label} mismatch: {actual!r} != {expected!r}"
        )


def validate_qwen38_27b_contract(
    config: Mapping[str, Any],
    model_path: Path,
    *,
    packing: str = QWEN38_PACKING,
) -> Qwen38ModelContract:
    """Validate the exact artifact used for the retained measurements."""

    if not _qwen38_identity(config, model_path):
        raise Qwen38ContractError("expected exact Qwen 3.8 27B identity")
    _require_equal(
        "architectures",
        config.get("architectures"),
        ["Qwen3_5ForConditionalGeneration"],
    )
    _require_equal("model_type", config.get("model_type"), "qwen3_5")
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise Qwen38ContractError("Qwen 3.8 contract text_config is missing")
    expected_text = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
    }
    for field, expected in expected_text.items():
        _require_equal(field, text.get(field), expected)
    extra = config.get("mlx_lm_extra_tensors")
    extra = extra if isinstance(extra, Mapping) else {}
    _require_equal("MTP sidecar", extra.get("mtp_file"), "mtp.safetensors")

    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        raise Qwen38ContractError("Qwen 3.8 trunk quantization is missing")
    trunk = (
        quantization.get("bits"),
        quantization.get("group_size"),
        quantization.get("mode"),
    )
    if trunk != (4, 32, "affine"):
        raise Qwen38ContractError(
            f"Qwen 3.8 trunk quantization mismatch: {trunk!r} != (4, 32, 'affine')"
        )
    overrides = {
        key for key, value in quantization.items() if isinstance(value, Mapping)
    }
    expected_overrides = _expected_q8_overrides()
    if overrides != expected_overrides:
        raise Qwen38ContractError(
            "Qwen 3.8 quantization override map mismatch: "
            f"missing={sorted(expected_overrides - overrides)}, "
            f"extra={sorted(overrides - expected_overrides)}"
        )
    for name in sorted(expected_overrides):
        spec = quantization[name]
        observed = (spec.get("bits"), spec.get("group_size"), spec.get("mode"))
        if observed != (8, 64, "affine"):
            raise Qwen38ContractError(
                f"Qwen 3.8 override {name} mismatch: {observed!r}"
            )
    if packing != QWEN38_PACKING:
        raise Qwen38ContractError(
            f"Qwen 3.8 packing mismatch: {packing!r} != {QWEN38_PACKING!r}"
        )

    contract_payload = {
        **expected_text,
        "packing": packing,
        "trunk_bits": 4,
        "trunk_group_size": 32,
        "trunk_mode": "affine",
    }
    contract_id = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen38ModelContract(
        contract_id=contract_id,
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        vocab_size=int(text["vocab_size"]),
        dtype=str(text["dtype"]),
        trunk_bits=4,
        trunk_group_size=32,
        trunk_mode="affine",
        packing=packing,
    )


def build_qwen38_route(
    config: Mapping[str, Any],
    model_path: Path,
    *,
    bindings: Qwen38RouteBindings,
    route_id: str,
    kernel_ids: tuple[str, ...] = (),
    min_context_tokens: int = 0,
    policy_id: str = "current_mtplx",
    selfcheck_status: str = "control",
    selfcheck_passed: bool = True,
) -> Qwen38RouteSpec:
    contract = validate_qwen38_27b_contract(config, model_path)
    if not callable(bindings.mtp_cache_append):
        raise Qwen38ContractError(
            f"route {route_id!r} is missing callable mtp_cache_append"
        )
    if route_id != "control" and not selfcheck_passed:
        raise Qwen38ContractError("challenge route self-check did not pass")
    return Qwen38RouteSpec(
        route_id=route_id,
        contract=contract,
        bindings=bindings,
        kernel_ids=tuple(kernel_ids),
        min_context_tokens=max(0, int(min_context_tokens)),
        policy_id=policy_id,
        selfcheck_status=selfcheck_status,
        selfcheck_passed=bool(selfcheck_passed),
    )


def control_bindings(runtime: Any) -> Qwen38RouteBindings:
    return Qwen38RouteBindings(
        mtp_cache_append=getattr(
            runtime.model,
            "mtp_update_cache",
            runtime.update_mtp_cache,
        )
    )


def install_qwen38_control_route(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
) -> Qwen38RouteSpec | None:
    return install_qwen38_route(runtime, config, model_path, cache_route="control")


def install_qwen38_route(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
    *,
    cache_route: str = DEFAULT_QWEN38_CACHE_ROUTE,
    dual_norm: bool = False,
    source_proposal: bool = False,
    row10_compact_vocab: bool = False,
    row18_gdn_decay_memo: bool = False,
    row18_mlp_gate_up: bool = False,
    source_artifact_path: Path | None = None,
    source_retain_control: bool = True,
) -> Qwen38RouteSpec | None:
    if not is_qwen38_27b_candidate(config, model_path):
        return None
    if not bool(getattr(runtime, "mtp_enabled", False)):
        return None
    cache_route_id = str(cache_route or "control").strip().lower()
    bindings = control_bindings(runtime)
    kernel_ids: list[str] = []
    route_features: list[str] = []
    feature_receipt: dict[str, dict[str, int]] = {}

    selfcheck_status = "control"
    min_context_tokens = 0
    if cache_route_id == "kv_only_history":
        implementation = getattr(
            runtime.model,
            "mtp_update_cache_kv_only_history",
            None,
        )
        if not callable(implementation):
            raise Qwen38ContractError(
                "Qwen 3.8 K/V-only history route is unavailable on the loaded model"
            )
        bindings = replace(bindings, mtp_cache_append=implementation)
        route_features.append("kv_only_history")
        kernel_ids.append("qwen38_mtp_kv_only_history_ge16384_v1")
        min_context_tokens = QWEN38_KV_ONLY_MIN_CONTEXT
        selfcheck_status = "passed:python16384:conditioned_abba"
    elif cache_route_id != "control":
        raise Qwen38ContractError(
            f"unknown Qwen 3.8 cache route: {cache_route!r}"
        )

    row18_gdn_report = configure_qwen38_row18_gdn_decay_memo(
        runtime.model,
        active=bool(row18_gdn_decay_memo),
    )
    if row18_gdn_decay_memo:
        if int(row18_gdn_report.get("active_modules", 0)) <= 0:
            raise Qwen38ContractError(
                "Qwen 3.8 row 18 GDN decay memo configured no modules"
            )
        route_features.append("r18_gdn_decay_memo")
        kernel_ids.append("qwen38_row18_gdn_neg_exp_a_log_memo_v1")
        feature_receipt["r18_gdn_decay_memo"] = row18_gdn_report

    row18_mlp_report = configure_qwen38_row18_mlp_gate_up(
        runtime.model,
        active=bool(row18_mlp_gate_up),
    )
    if row18_mlp_gate_up:
        if int(row18_mlp_report.get("active_modules", 0)) <= 0:
            raise Qwen38ContractError(
                "Qwen 3.8 row 18 packed MLP gate/up configured no modules"
            )
        route_features.append("r18_mlp_gate_up")
        kernel_ids.append("qwen38_row18_mlp_gate_up_s_le9_v1")
        feature_receipt["r18_mlp_gate_up"] = row18_mlp_report

    text = getattr(runtime.model, "language_model", runtime.model)
    text._mtplx_qwen38_dual_norm_concat = bool(dual_norm)
    if dual_norm:
        route_features.append("dual_norm")
        kernel_ids.append("qwen38_dual_rms_norm_concat_bf16_v1")
        feature_receipt["dual_norm"] = {"active": 1}

    row10_report = configure_qwen38_row10_compact_head(
        runtime,
        active=bool(row10_compact_vocab),
    )
    if row10_compact_vocab:
        if not bool(row10_report.get("installed")):
            raise Qwen38ContractError("Qwen 3.8 row 10 compact head was not installed")
        route_features.append("r10_compact_vocab")
        kernel_ids.append("qwen38_row10_compact_q4_g64_vocab_v1")
        feature_receipt["r10_compact_vocab"] = row10_report

    source_report = configure_qwen38_source_proposal(
        runtime,
        active=bool(source_proposal),
        artifact_path=source_artifact_path,
        retain_control=bool(source_retain_control),
    )
    if source_proposal:
        if not bool(source_report.get("installed")):
            raise Qwen38ContractError("Qwen 3.8 source proposal route was not installed")
        route_features.append("source_proposal")
        kernel_ids.extend(
            (
                "qwen38_source_q4_g64_bf16_qkv_islands_v1",
                "qwen38_source_q2_top32_q4_rerank_v1",
            )
        )
        feature_receipt["source_proposal"] = source_report

    route_id = "+".join(route_features) if route_features else "control"
    route = build_qwen38_route(
        config,
        model_path,
        bindings=bindings,
        route_id=route_id,
        kernel_ids=tuple(kernel_ids),
        min_context_tokens=min_context_tokens,
        selfcheck_status=selfcheck_status,
    )
    runtime.qwen38_route = route
    runtime.qwen38_feature_receipt = feature_receipt
    return route


def policy_fingerprint_with_qwen38_route(
    fingerprint: str,
    route: Qwen38RouteSpec | None,
) -> str:
    if route is None:
        return fingerprint
    return f"{fingerprint};qwen38_route={route.fingerprint}"


def qwen38_route_receipt(route: Qwen38RouteSpec | None) -> dict[str, Any] | None:
    if route is None:
        return None
    return {
        "route_id": route.route_id,
        "fingerprint": route.fingerprint,
        "contract_id": route.contract.contract_id,
        "kernel_ids": list(route.kernel_ids),
        "min_context_tokens": route.min_context_tokens,
        "policy_id": route.policy_id,
        "selfcheck": {
            "passed": bool(route.selfcheck_passed),
            "status": route.selfcheck_status,
        },
    }
