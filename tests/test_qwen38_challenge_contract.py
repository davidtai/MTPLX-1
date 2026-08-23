from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.attention_context import request_prompt_tokens
from mtplx.mtp_patch import MTPContract
from mtplx.qwen38_challenge import (
    QWEN38_FINAL_ROUTE,
    QWEN38_Q8_LINEAR_ATTN_LAYERS,
    Qwen38ContractError,
    Qwen38RouteBindings,
    build_qwen38_route,
    install_qwen38_control_route,
    install_qwen38_route,
    is_qwen38_27b_candidate,
    policy_fingerprint_with_qwen38_route,
    qwen38_final_route,
    qwen38_route_receipt,
    validate_qwen38_27b_contract,
)
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
        "mtplx_runtime": {"base_trunk": "/models/Qwen--Qwen3.8-27B"},
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
    return Qwen38RouteBindings(mtp_cache_append=_callable)


def test_exact_qwen38_27b_control_contract_is_accepted() -> None:
    contract = validate_qwen38_27b_contract(_config(), MODEL_PATH)

    assert contract.hidden_size == 5120
    assert contract.vocab_size == 248320
    assert contract.trunk_bits == 4
    assert contract.trunk_group_size == 32
    assert contract.packing == "mlx_affine_u32_le"


def test_final_route_is_only_the_chronological_winner_stack(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_QWEN38_DISABLE_SOURCE_AUTO", raising=False)
    assert dict(QWEN38_FINAL_ROUTE) == {
        "cache_route": "kv_only_history",
        "dual_norm": True,
        "source_proposal": True,
        "source_retain_control": False,
    }
    assert qwen38_final_route() == dict(QWEN38_FINAL_ROUTE)

    monkeypatch.setenv("MTPLX_QWEN38_DISABLE_SOURCE_AUTO", "1")
    assert qwen38_final_route() == {
        "cache_route": "kv_only_history",
        "dual_norm": True,
    }


@pytest.mark.parametrize(
    "sibling",
    [
        "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    ],
)
def test_unmeasured_qwen38_siblings_stay_outside_the_route(sibling: str) -> None:
    assert not is_qwen38_27b_candidate(_config(), Path(sibling))


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
    else:
        packing = str(value)

    with pytest.raises(Qwen38ContractError, match=message):
        validate_qwen38_27b_contract(config, path, packing=packing)


def test_route_is_immutable_and_fingerprint_covers_kernel_and_policy() -> None:
    base = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )

    with pytest.raises(FrozenInstanceError):
        base.route_id = "changed"  # type: ignore[misc]
    assert replace(base, kernel_ids=("kernel-b",)).fingerprint != base.fingerprint
    assert replace(base, policy_id="policy-b").fingerprint != base.fingerprint


def test_route_fingerprint_prevents_cross_route_session_restore() -> None:
    control = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )
    candidate = replace(control, route_id="kv_only_history", kernel_ids=("kv-v1",))
    runtime = SimpleNamespace(model_path=MODEL_PATH, mtp_enabled=True)
    bank = SessionBank(max_entries=2, max_bytes=4096, per_session_max_bytes=4096)
    assert bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        policy_fingerprint=policy_fingerprint_with_qwen38_route("base", control),
        nbytes_override=64,
    )

    assert bank.restore(
        runtime,
        [1, 2, 3],
        policy_fingerprint=policy_fingerprint_with_qwen38_route("base", candidate),
        cache_factory=list,
    ) is None
    assert bank.last_miss_reason == CacheMissReason.POLICY_MISMATCH.value


def test_kv_only_history_route_binds_the_target_shaped_append() -> None:
    calls: list[str] = []

    def stock(*args, **kwargs):
        calls.append("stock")
        return "stock"

    def kv_only(*args, **kwargs):
        calls.append("kv-only")
        return "kv-only"

    runtime = MTPLXRuntime(
        model=SimpleNamespace(
            mtp_update_cache=stock,
            mtp_update_cache_kv_only_history=kv_only,
        ),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    route = install_qwen38_route(runtime, _config(), MODEL_PATH)

    assert route is runtime.qwen38_route
    assert route.route_id == "kv_only_history"
    assert route.bindings.mtp_cache_append is kv_only
    assert route.min_context_tokens == 16_384
    short = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 1024)),
        mtp_cache=[SimpleNamespace(offset=0)],
    )
    long = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 16384)),
        mtp_cache=[SimpleNamespace(offset=0)],
    )
    continued_long = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 1)),
        mtp_cache=[SimpleNamespace(offset=16384)],
    )
    with request_prompt_tokens(16_384):
        windowed_long = runtime.update_mtp_cache(
            object(),
            SimpleNamespace(shape=(1, 8192)),
            mtp_cache=[SimpleNamespace(offset=0)],
        )

    assert (short, long, continued_long, windowed_long) == (
        "stock",
        "kv-only",
        "stock",
        "kv-only",
    )
    assert calls == ["stock", "kv-only", "stock", "kv-only"]
    assert route.kernel_ids == ("qwen38_mtp_kv_only_history_ge16384_v1",)
    assert route.selfcheck_status == "passed:python16384:conditioned_abba"


def test_control_route_and_receipt_are_stable() -> None:
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    route = install_qwen38_control_route(runtime, _config(), MODEL_PATH)

    assert route.route_id == "control"
    assert qwen38_route_receipt(route) == {
        "route_id": "control",
        "fingerprint": route.fingerprint,
        "contract_id": route.contract.contract_id,
        "kernel_ids": [],
        "min_context_tokens": 0,
        "policy_id": "current_mtplx",
        "selfcheck": {"passed": True, "status": "control"},
    }
    assert _qwen38_challenge_route_payload(runtime) == qwen38_route_receipt(route)
