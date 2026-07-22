"""Hy3 backend descriptor: registration, family detection, kv_quant policy.

mtplx/backends/descriptors.py is the app/API-facing metadata surface (used by
mtplx/server/openai.py and mtplx/commands/public.py) that tells clients what a
loaded model supports. Before this module gained an HY3_MTP_DESCRIPTOR entry,
any hy3 model fell through to NATIVE_CONTRACT_DESCRIPTOR, which advertises
kv_quant_policy(supported=False) -- inaccurate once the SSD expert-streaming
path's kv_quant knob (mtplx/expert_runtime.py ExpertStreamingConfig.kv_quant)
actually reaches Hy3Model.make_cache via mtplx/resident_loader.py.
"""

from __future__ import annotations

from mtplx.backends.descriptors import (
    DESCRIPTORS_BY_BACKEND_ID,
    HY3_MTP_DESCRIPTOR,
    NATIVE_CONTRACT_DESCRIPTOR,
    context_window_policy_for_model,
    descriptor_for_architecture_id,
    descriptor_for_backend_id,
    descriptor_from_inspection,
    kv_quant_policy_for_model,
    model_controls_for_descriptor,
    model_family_from_inspection,
    reasoning_policy_for_model,
)


def test_hy3_descriptor_is_registered_under_its_backend_id() -> None:
    assert DESCRIPTORS_BY_BACKEND_ID["hy_v3_mtp"] is HY3_MTP_DESCRIPTOR
    assert descriptor_for_backend_id("hy_v3_mtp") is HY3_MTP_DESCRIPTOR
    # Regression: an unregistered backend id must NOT silently resolve to
    # the Hy3 descriptor (or to any other specific family) -- it falls back
    # to the generic native-contract descriptor.
    assert descriptor_for_backend_id("not-a-real-backend") is NATIVE_CONTRACT_DESCRIPTOR


def test_hy3_descriptor_resolves_by_architecture_id() -> None:
    assert descriptor_for_architecture_id("hy-v3-mtp") is HY3_MTP_DESCRIPTOR


def test_hy3_family_detected_from_backend_id() -> None:
    inspection = {"recommended_backend": "hy_v3_mtp"}
    assert model_family_from_inspection(inspection) == "hy3"
    assert descriptor_from_inspection(inspection) is HY3_MTP_DESCRIPTOR


def test_hy3_family_detected_from_arch_id() -> None:
    inspection = {"mtp_arch": "hy-v3-mtp"}
    assert model_family_from_inspection(inspection) == "hy3"
    assert descriptor_from_inspection(inspection) is HY3_MTP_DESCRIPTOR


def test_hy3_family_detected_from_model_ref_text() -> None:
    assert model_family_from_inspection(None, model_ref="tencent/Hy3") == "hy3"
    assert model_family_from_inspection(None, model_ref="pipenetwork/Hy3-4bit") == "hy3"


def test_hy3_kv_quant_policy_is_supported_with_off_q8_q4() -> None:
    policy = kv_quant_policy_for_model(model_ref="tencent/Hy3")
    assert policy.supported is True
    assert policy.modes == ("off", "q8", "q4")
    payload = policy.to_dict()
    assert payload["supported"] is True
    assert payload["modes"] == ["off", "q8", "q4"]
    assert payload["disabled_reason"] is None


def test_unrelated_model_still_reports_kv_quant_unsupported() -> None:
    """Adding the Hy3 branch (and its "hy3"/"hy_v3"/"hy-v3" text markers)
    must not change resolution for a model that was already correctly
    unsupported -- regression guard against the new branch's placement
    accidentally widening or reordering the family if-chain."""

    policy = kv_quant_policy_for_model(model_ref="zai-org/GLM-5.2")
    assert model_family_from_inspection(None, model_ref="zai-org/GLM-5.2") == "glm"
    assert policy.supported is False


def test_hy3_reasoning_and_context_window_resolve_without_crashing() -> None:
    reasoning = reasoning_policy_for_model(model_ref="tencent/Hy3")
    assert reasoning.supported is False  # no verified Hy3 reasoning parser yet

    context = context_window_policy_for_model(model_ref="tencent/Hy3")
    assert context.maximum == 262_144
    assert context.default == 262_144


def test_model_controls_for_descriptor_reports_hy3_kv_quant_end_to_end() -> None:
    controls = model_controls_for_descriptor(
        HY3_MTP_DESCRIPTOR, model_ref="tencent/Hy3"
    )
    assert controls["model_family"] == "hy3"
    assert controls["backend_id"] == "hy_v3_mtp"
    assert controls["kv_quant"]["supported"] is True
    assert controls["kv_quant"]["modes"] == ["off", "q8", "q4"]
