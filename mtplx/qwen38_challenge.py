"""Immutable construction contract for the Qwen 3.8 27B optimization lane."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


QWEN38_Q8_LINEAR_ATTN_LAYERS = (
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    28,
    29,
    30,
    32,
    33,
    34,
    36,
    37,
    38,
    40,
    41,
    42,
    44,
    45,
    46,
    48,
    49,
    50,
    52,
    53,
    54,
    56,
    57,
    58,
    60,
    61,
    62,
)
QWEN38_QMV_WIDTHS = tuple(range(2, 10))
QWEN38_PACKING = "mlx_affine_u32_le"
DEFAULT_QWEN38_CACHE_ROUTE = "kv_only_history"


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
    qmv_bits: int
    qmv_group_size: int
    packing: str


@dataclass(frozen=True)
class Qwen38RouteBindings:
    proposal_readout: Callable[..., Any]
    qmv_by_width: Mapping[int, Callable[..., Any]]
    mtp_cache_append: Callable[..., Any]
    projection_fusions: Mapping[str, Callable[..., Any]]
    policy_factory: Callable[..., Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qmv_by_width",
            MappingProxyType(dict(self.qmv_by_width)),
        )
        object.__setattr__(
            self,
            "projection_fusions",
            MappingProxyType(dict(self.projection_fusions)),
        )


@dataclass(frozen=True)
class Qwen38RouteSpec:
    route_id: str
    contract: Qwen38ModelContract
    bindings: Qwen38RouteBindings
    compact_head_digest: str | None = None
    kernel_ids: tuple[str, ...] = ()
    policy_id: str = "current_mtplx"
    selfcheck_status: str = "control"
    selfcheck_passed: bool = True

    @property
    def fingerprint(self) -> str:
        payload = {
            "compact_head_digest": self.compact_head_digest,
            "contract_id": self.contract.contract_id,
            "kernel_ids": list(self.kernel_ids),
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
    """Validate the exact measured artifact; every mismatch fails construction."""

    if not _qwen38_identity(config, model_path):
        raise Qwen38ContractError("expected exact Qwen 3.8 27B identity")
    _require_equal("architectures", config.get("architectures"), ["Qwen3_5ForConditionalGeneration"])
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
    overrides = {key for key, value in quantization.items() if isinstance(value, Mapping)}
    expected_overrides = _expected_q8_overrides()
    if overrides != expected_overrides:
        missing = sorted(expected_overrides - overrides)
        extra_overrides = sorted(overrides - expected_overrides)
        raise Qwen38ContractError(
            "Qwen 3.8 quantization override map mismatch: "
            f"missing={missing}, extra={extra_overrides}"
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
        "qmv_bits": 4,
        "qmv_group_size": 64,
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
        qmv_bits=4,
        qmv_group_size=64,
        packing=packing,
    )


def _validate_bindings(bindings: Qwen38RouteBindings, route_id: str) -> None:
    for name in (
        "proposal_readout",
        "mtp_cache_append",
        "policy_factory",
    ):
        if not callable(getattr(bindings, name)):
            raise Qwen38ContractError(f"route {route_id!r} is missing callable {name}")
    if route_id != "control" and set(bindings.qmv_by_width) != set(QWEN38_QMV_WIDTHS):
        raise Qwen38ContractError("challenge route requires QMV widths 2..9")
    if route_id != "control" and not bindings.projection_fusions:
        raise Qwen38ContractError("challenge route requires projection fusions")
    for width, implementation in bindings.qmv_by_width.items():
        if not callable(implementation):
            raise Qwen38ContractError(f"QMV width {width} is not callable")
    for name, implementation in bindings.projection_fusions.items():
        if not callable(implementation):
            raise Qwen38ContractError(f"projection fusion {name!r} is not callable")


def build_qwen38_route(
    config: Mapping[str, Any],
    model_path: Path,
    *,
    bindings: Qwen38RouteBindings,
    route_id: str,
    compact_head_digest: str | None = None,
    kernel_ids: tuple[str, ...] = (),
    policy_id: str = "current_mtplx",
    selfcheck_status: str = "control",
    selfcheck_passed: bool = True,
) -> Qwen38RouteSpec:
    contract = validate_qwen38_27b_contract(config, model_path)
    _validate_bindings(bindings, route_id)
    if route_id != "control" and not selfcheck_passed:
        raise Qwen38ContractError("challenge route self-check did not pass")
    return Qwen38RouteSpec(
        route_id=route_id,
        contract=contract,
        bindings=bindings,
        compact_head_digest=compact_head_digest,
        kernel_ids=tuple(kernel_ids),
        policy_id=policy_id,
        selfcheck_status=selfcheck_status,
        selfcheck_passed=bool(selfcheck_passed),
    )


def _stock_proposal_readout(logits: Any) -> Any:
    import mlx.core as mx

    return mx.argmax(logits, axis=-1)


def _stock_qmv(*args: Any, **kwargs: Any) -> Any:
    import mlx.core as mx

    return mx.quantized_matmul(*args, **kwargs)


def _stock_projection(module: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return module(*args, **kwargs)


def _stock_policy_factory(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return factory(*args, **kwargs)


def control_bindings(runtime: Any) -> Qwen38RouteBindings:
    stock_cache_append = getattr(
        runtime.model,
        "mtp_update_cache",
        runtime.update_mtp_cache,
    )
    return Qwen38RouteBindings(
        proposal_readout=_stock_proposal_readout,
        qmv_by_width={width: _stock_qmv for width in QWEN38_QMV_WIDTHS},
        mtp_cache_append=stock_cache_append,
        projection_fusions={"stock": _stock_projection},
        policy_factory=_stock_policy_factory,
    )


def install_qwen38_control_route(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
) -> Qwen38RouteSpec | None:
    return install_qwen38_route(
        runtime,
        config,
        model_path,
        cache_route="control",
    )


def install_qwen38_route(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
    *,
    cache_route: str = DEFAULT_QWEN38_CACHE_ROUTE,
    proposal_route: str = "control",
    compact_head_path: Path | None = None,
) -> Qwen38RouteSpec | None:
    if not is_qwen38_27b_candidate(config, model_path):
        return None
    if not bool(getattr(runtime, "mtp_enabled", False)):
        return None
    cache_route_id = str(cache_route or "control").strip().lower()
    proposal_route_id = str(proposal_route or "control").strip().lower()
    bindings = control_bindings(runtime)
    kernel_ids: list[str] = []
    selfcheck_status = "control"
    compact_head_digest = None
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
        kernel_ids.append("qwen38_mtp_kv_only_history_v1")
        selfcheck_status = "passed:cache_exact+64t_abba_baab"
    elif cache_route_id != "control":
        raise Qwen38ContractError(f"unknown Qwen 3.8 cache route: {cache_route!r}")

    if proposal_route_id == "compact_head":
        if compact_head_path is None:
            raise Qwen38ContractError("compact_head route requires compact_head_path")
        from .qwen38_compact_head import install_qwen38_compact_proposal_head

        artifact = install_qwen38_compact_proposal_head(
            runtime,
            compact_head_path,
            source_contract_id=validate_qwen38_27b_contract(
                config,
                model_path,
            ).contract_id,
        )
        text = getattr(runtime.model, "language_model", runtime.model)
        proposal = getattr(text, "_mtplx_draft_lm_head")
        bindings = replace(bindings, proposal_readout=proposal)
        compact_head_digest = artifact.sha256
        kernel_ids.extend(
            (
                "qwen38_compact_q2_shortlist_reference_v1",
                "qwen38_target_q8_selected_rerank_reference_v1",
            )
        )
        selfcheck_status = "pending:python100"
    elif proposal_route_id == "control":
        from .qwen38_compact_head import restore_qwen38_control_proposal_head

        restore_qwen38_control_proposal_head(runtime)
    else:
        raise Qwen38ContractError(
            f"unknown Qwen 3.8 proposal route: {proposal_route!r}"
        )

    route_parts = []
    if proposal_route_id != "control":
        route_parts.append(proposal_route_id)
    if cache_route_id != "control":
        route_parts.append(cache_route_id)
    route_id = "+".join(route_parts) or "control"
    route = build_qwen38_route(
        config,
        model_path,
        bindings=bindings,
        route_id=route_id,
        compact_head_digest=compact_head_digest,
        kernel_ids=tuple(kernel_ids),
        selfcheck_status=selfcheck_status,
    )
    runtime.qwen38_route = route
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
        "compact_head_digest": route.compact_head_digest,
        "kernel_ids": list(route.kernel_ids),
        "policy_id": route.policy_id,
        "selfcheck": {
            "passed": bool(route.selfcheck_passed),
            "status": route.selfcheck_status,
        },
    }
