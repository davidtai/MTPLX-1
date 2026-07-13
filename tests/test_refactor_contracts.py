from __future__ import annotations

import json

import pytest

from mtplx import (
    deepseek_mtp_patch,
    glm_mtp_patch,
    mimo_mtp_patch,
    nemotron_h_mtp_patch,
    step3p5_mtp_patch,
)
from mtplx.commands import public
from mtplx.server import openai


@pytest.mark.parametrize(
    "helper",
    [public._is_localhost_bind, openai._is_localhost_bind],
)
@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (None, True),
        ("  LOCALHOST  ", True),
        ("127.0.0.1", True),
        ("[::1]", True),
        ("[[::1]]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("example.com", False),
    ],
)
def test_localhost_bind_helpers_share_the_existing_contract(
    helper, host: str | None, expected: bool
) -> None:
    assert helper(host) is expected


@pytest.mark.parametrize(
    "module",
    [
        deepseek_mtp_patch,
        glm_mtp_patch,
        mimo_mtp_patch,
        nemotron_h_mtp_patch,
        step3p5_mtp_patch,
    ],
)
def test_mtp_patch_model_type_helpers_share_the_existing_contract(module) -> None:
    assert module._model_type({"model_type": "TOP"}) == "top"
    assert (
        module._model_type(
            {"model_type": "TOP", "text_config": {"model_type": "Nested"}}
        )
        == "nested"
    )
    assert module._model_type({}) == ""


@pytest.mark.parametrize(
    "module",
    [deepseek_mtp_patch, glm_mtp_patch, mimo_mtp_patch, step3p5_mtp_patch],
)
def test_matching_mtp_layer_count_helpers_preserve_key_precedence(module) -> None:
    assert (
        module._num_mtp_layers(
            {
                "num_nextn_predict_layers": 4,
                "text_config": {
                    "num_nextn_predict_layers": 2,
                    "mtp_num_hidden_layers": 3,
                },
            }
        )
        == 2
    )
    assert module._num_mtp_layers({"text_config": {"mtp_num_hidden_layers": 3}}) == 3
    assert module._num_mtp_layers({"mtp_num_hidden_layers": 5}) == 5
    assert (
        module._num_mtp_layers(
            {"mtp_num_hidden_layers": 5, "text_config": {}}
        )
        == 0
    )


def test_nemotron_layer_count_keeps_its_top_level_mtp_fallback() -> None:
    assert (
        nemotron_h_mtp_patch._num_mtp_layers(
            {"mtp_num_hidden_layers": 5, "text_config": {}}
        )
        == 5
    )


@pytest.mark.parametrize(
    ("module", "config", "weight_key"),
    [
        (
            deepseek_mtp_patch,
            {"num_hidden_layers": 10, "num_nextn_predict_layers": 1},
            "model.layers.10.input_layernorm.weight",
        ),
        (
            glm_mtp_patch,
            {"num_hidden_layers": 10, "num_nextn_predict_layers": 1},
            "model.layers.10.input_layernorm.weight",
        ),
        (
            mimo_mtp_patch,
            {"num_hidden_layers": 10, "num_nextn_predict_layers": 1},
            "model.mtp_layers.0.input_layernorm.weight",
        ),
        (
            nemotron_h_mtp_patch,
            {
                "num_hidden_layers": 10,
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*",
            },
            "backbone.layers.10.mixer.weight",
        ),
        (
            step3p5_mtp_patch,
            {"num_hidden_layers": 10, "num_nextn_predict_layers": 1},
            "language_model.model.layers.10.input_layernorm.weight",
        ),
    ],
)
def test_mtp_candidate_weight_files_preserve_adapter_prefix_policies(
    tmp_path, module, config: dict, weight_key: str
) -> None:
    index = {
        "weight_map": {
            "unrelated.weight": "model-00001-of-00003.safetensors",
            weight_key: "model-00002-of-00003.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert module._candidate_weight_files(tmp_path, config) == [
        tmp_path / "model-00002-of-00003.safetensors"
    ]


def test_mtp_candidate_weight_files_prefer_sidecar_and_fall_back_to_glob(
    tmp_path,
) -> None:
    config = {"num_hidden_layers": 10, "num_nextn_predict_layers": 1}
    sidecar = tmp_path / "mtp.safetensors"
    fallback_b = tmp_path / "model-b.safetensors"
    fallback_a = tmp_path / "model-a.safetensors"
    fallback_b.touch()
    fallback_a.touch()
    sidecar.touch()

    assert deepseek_mtp_patch._candidate_weight_files(tmp_path, config) == [sidecar]

    sidecar.unlink()
    (tmp_path / "model.safetensors.index.json").write_text("not json")
    assert deepseek_mtp_patch._candidate_weight_files(tmp_path, config) == [
        fallback_a,
        fallback_b,
    ]
