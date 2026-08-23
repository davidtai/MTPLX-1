from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.qwen38_challenge import (
    QWEN38_Q8_LINEAR_ATTN_LAYERS,
    Qwen38ContractError,
    Qwen38RouteBindings,
    build_qwen38_route,
    control_bindings,
    install_qwen38_control_route,
    install_qwen38_route,
    is_qwen38_27b_candidate,
    policy_fingerprint_with_qwen38_route,
    qwen38_route_receipt,
    validate_qwen38_27b_contract,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.server.openai import _qwen38_challenge_route_payload
from mtplx.session_bank import CacheMissReason, SessionBank


MODEL_PATH = Path("models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed")


def _config() -> dict:
    quantization: dict[str, object] = {
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
        "language_model.model.embed_tokens": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        },
        "language_model.lm_head": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        },
    }
    for layer in QWEN38_Q8_LINEAR_ATTN_LAYERS:
        quantization[f"language_model.model.layers.{layer}.linear_attn.out_proj"] = {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        }
    for layer in range(56, 64):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            quantization[
                f"language_model.model.layers.{layer}.mlp.{projection}"
            ] = {"bits": 8, "group_size": 64, "mode": "affine"}
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
        "mtplx_runtime": {
            "arch_id": "qwen3-next-mtp",
            "base_trunk": "/models/Qwen--Qwen3.8-27B",
        },
        "quantization": quantization,
        "text_config": {
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
        },
    }


def _callable(*args, **kwargs):
    return args, kwargs


def _bindings() -> Qwen38RouteBindings:
    return Qwen38RouteBindings(
        proposal_readout=_callable,
        qmv_by_width={width: _callable for width in range(2, 10)},
        mtp_cache_append=_callable,
        projection_fusions={"stock": _callable},
        policy_factory=_callable,
    )


def test_exact_qwen38_27b_control_contract_is_accepted() -> None:
    contract = validate_qwen38_27b_contract(_config(), MODEL_PATH)

    assert contract.hidden_size == 5120
    assert contract.vocab_size == 248320
    assert contract.trunk_bits == 4
    assert contract.trunk_group_size == 32
    assert contract.qmv_group_size == 64
    assert contract.packing == "mlx_affine_u32_le"


@pytest.mark.parametrize(
    "sibling",
    [
        "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    ],
)
def test_unmeasured_qwen38_siblings_stay_outside_the_route(sibling: str) -> None:
    config = _config()
    config["mtplx_runtime"]["base_trunk"] = "/models/Qwen--Qwen3.8-27B"

    assert not is_qwen38_27b_candidate(config, Path(sibling))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family", "Qwen3.6-27B", "Qwen 3.8 27B identity"),
        ("hidden_size", 4096, "hidden_size"),
        ("dtype", "float16", "dtype"),
        ("bits", 3, "trunk quantization"),
        ("group_size", 64, "trunk quantization"),
        ("mode", "symmetric", "trunk quantization"),
        ("packing", "q4_k", "packing"),
    ],
)
def test_contract_misses_fail_loudly(field: str, value: object, message: str) -> None:
    config = _config()
    path = MODEL_PATH
    packing = "mlx_affine_u32_le"
    if field == "family":
        path = Path(f"models/{value}")
        config["mtplx_runtime"]["base_trunk"] = f"/models/{value}"
    elif field in config["text_config"]:
        config["text_config"][field] = value
    elif field in {"bits", "group_size", "mode"}:
        config["quantization"][field] = value
    elif field == "packing":
        packing = str(value)

    with pytest.raises(Qwen38ContractError, match=message):
        validate_qwen38_27b_contract(config, path, packing=packing)


def test_route_is_immutable_and_identity_covers_head_kernels_and_policy() -> None:
    base = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )

    with pytest.raises(FrozenInstanceError):
        base.route_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        base.bindings.qmv_by_width[2] = _callable  # type: ignore[index]

    assert replace(base, compact_head_digest="head-b").fingerprint != base.fingerprint
    assert replace(base, kernel_ids=("kernel-b",)).fingerprint != base.fingerprint
    assert replace(base, policy_id="policy-b").fingerprint != base.fingerprint


def test_candidate_route_requires_every_promoted_callable() -> None:
    bindings = replace(_bindings(), qmv_by_width={2: _callable})

    with pytest.raises(Qwen38ContractError, match="QMV widths 2..9"):
        build_qwen38_route(
            _config(), MODEL_PATH, bindings=bindings, route_id="challenge"
        )


def test_route_fingerprint_prevents_cross_route_session_restore() -> None:
    control = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )
    candidate = replace(control, route_id="challenge", kernel_ids=("top2-v1",))
    control_policy = policy_fingerprint_with_qwen38_route("base", control)
    candidate_policy = policy_fingerprint_with_qwen38_route("base", candidate)
    runtime = SimpleNamespace(model_path=MODEL_PATH, mtp_enabled=True)
    bank = SessionBank(max_entries=2, max_bytes=4096, per_session_max_bytes=4096)
    assert bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        policy_fingerprint=control_policy,
        nbytes_override=64,
    )

    assert (
        bank.restore(
            runtime,
            [1, 2, 3],
            policy_fingerprint=candidate_policy,
            cache_factory=lambda: [],
        )
        is None
    )
    assert bank.last_miss_reason == CacheMissReason.POLICY_MISMATCH.value


def test_health_and_completion_receipt_is_stable() -> None:
    route = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )

    receipt = qwen38_route_receipt(route)

    assert receipt == {
        "route_id": "control",
        "fingerprint": route.fingerprint,
        "contract_id": route.contract.contract_id,
        "compact_head_digest": None,
        "kernel_ids": [],
        "policy_id": "current_mtplx",
        "selfcheck": {"passed": True, "status": "control"},
    }


def test_runtime_installs_one_control_route_and_server_surfaces_it() -> None:
    runtime = MTPLXRuntime(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    assert runtime.qwen38_route is None

    route = install_qwen38_control_route(runtime, _config(), MODEL_PATH)

    assert route is runtime.qwen38_route
    assert _qwen38_challenge_route_payload(runtime) == qwen38_route_receipt(route)


def test_kv_only_candidate_binds_exact_route_and_runtime_dispatches_to_it() -> None:
    calls: list[str] = []

    def stock(*args, **kwargs):
        calls.append("stock")
        return "stock"

    def kv_only(*args, **kwargs):
        calls.append("kv_only")
        return "candidate"

    model = SimpleNamespace(
        mtp_update_cache=stock,
        mtp_update_cache_kv_only_history=kv_only,
    )
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
    )
    result = runtime.update_mtp_cache("hidden", "tokens", mtp_cache="cache")

    assert route is runtime.qwen38_route
    assert route.route_id == "kv_only_history"
    assert route.kernel_ids == ("qwen38_mtp_kv_only_history_v1",)
    assert result == "candidate"
    assert calls == ["kv_only"]


def test_non_cache_candidate_binding_uses_model_append_without_runtime_recursion() -> None:
    calls: list[str] = []

    def stock(*args, **kwargs):
        calls.append("stock")
        return "stock"

    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=stock),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    bindings = control_bindings(runtime)
    runtime.qwen38_route = build_qwen38_route(
        _config(),
        MODEL_PATH,
        bindings=bindings,
        route_id="proposal_only_test",
    )

    assert runtime.update_mtp_cache("hidden", "tokens", mtp_cache="cache") == "stock"
    assert calls == ["stock"]
