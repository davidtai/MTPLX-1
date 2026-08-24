from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mtplx.dflash2_bundle as dflash2_bundle
import pytest
from mtplx.dflash2_bundle import (
    DFLASH2_ALGORITHM_REPO,
    DFLASH2_ALGORITHM_REVISION,
    DFLASH2_DRAFT_REPO,
    DFLASH2_DRAFT_REVISION,
    DFLASH2_TARGET_BASE_REPO,
    DFLASH2_TARGET_LAYER_IDS,
    DFLASH2_TARGET_REPO,
    DFLASH2_TARGET_REVISION,
)
from mtplx.qwen38_challenge import _expected_q8_overrides


def exact_target_config(**overrides: Any) -> dict[str, Any]:
    quantization: dict[str, Any] = {
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
    }
    quantization.update({
        name: {"bits": 8, "group_size": 64, "mode": "affine"}
        for name in _expected_q8_overrides()
    })
    config: dict[str, Any] = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
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
    text_overrides = overrides.pop("text_config", {})
    config["text_config"].update(text_overrides)
    config.update(overrides)
    return config


def exact_draft_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": "qwen3",
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 5120,
        "vocab_size": 248320,
        "num_hidden_layers": 5,
        "num_target_layers": 64,
        "dflash_config": {
            "block_size": 8,
            "target_layer_ids": list(DFLASH2_TARGET_LAYER_IDS),
        },
    }
    dflash_overrides = overrides.pop("dflash_config", {})
    config["dflash_config"].update(dflash_overrides)
    config.update(overrides)
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exact_bundle(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    target_config: dict[str, Any] | None = None,
    draft_config: dict[str, Any] | None = None,
) -> Path:
    target = root / "target"
    draft = root / "dflash2"
    target.mkdir(parents=True)
    draft.mkdir()
    target_config_path = target / "config.json"
    draft_config_path = draft / "config.json"
    draft_weights_path = draft / "model.safetensors"
    target_config_path.write_text(json.dumps(target_config or exact_target_config()))
    draft_config_path.write_text(json.dumps(draft_config or exact_draft_config()))
    target_weights_path = target / "model.safetensors"
    target_weights_path.write_bytes(b"target")
    draft_weights_path.write_bytes(b"draft")
    (root / "mtplx_dflash2.json").write_text(json.dumps({
        "schemaVersion": 1,
        "backend": "dflash2",
        "layout": {"target": "target", "draft": "dflash2"},
        "block_size": 8,
        "target": {
            "repo": DFLASH2_TARGET_REPO,
            "base_model": DFLASH2_TARGET_BASE_REPO,
            "revision": DFLASH2_TARGET_REVISION,
        },
        "draft": {
            "repo": DFLASH2_DRAFT_REPO,
            "revision": DFLASH2_DRAFT_REVISION,
            "precision": "4bit",
        },
        "algorithm": {
            "repo": DFLASH2_ALGORITHM_REPO,
            "revision": DFLASH2_ALGORITHM_REVISION,
        },
        "checksums": {
            "target_config": {
                "path": "target/config.json",
                "sha256": _sha256(target_config_path),
            },
            "target_weights": {
                "path": "target/model.safetensors",
                "sha256": _sha256(target_weights_path),
            },
            "draft_config": {
                "path": "dflash2/config.json",
                "sha256": _sha256(draft_config_path),
            },
            "draft_weights": {
                "path": "dflash2/model.safetensors",
                "sha256": _sha256(draft_weights_path),
            },
        },
        "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    }))
    # Unit bundles are intentionally tiny.  Install their byte identity as the
    # process-local trust root so production validation stays fail-closed while
    # tests exercise the same code path without copying the 27B artifacts.
    monkeypatch.setattr(
        dflash2_bundle, "DFLASH2_TARGET_CONFIG_SHA256", _sha256(target_config_path)
    )
    monkeypatch.setattr(
        dflash2_bundle, "DFLASH2_DRAFT_CONFIG_SHA256", _sha256(draft_config_path)
    )
    monkeypatch.setattr(dflash2_bundle, "DFLASH2_TARGET_WEIGHT_SHA256", {
        target_weights_path.name: _sha256(target_weights_path),
    })
    monkeypatch.setattr(dflash2_bundle, "DFLASH2_DRAFT_WEIGHT_SHA256", {
        draft_weights_path.name: _sha256(draft_weights_path),
    })
    return root
