from __future__ import annotations

import hashlib
import json

from mtplx.artifacts import inspect_model
from mtplx.backends.descriptors import DFLASH2_DESCRIPTOR, descriptor_from_inspection
from mtplx.dflash2_bundle import (
    DFLASH2_ALGORITHM_REVISION,
    DFLASH2_ARCH_ID,
    DFLASH2_BACKEND,
    DFLASH2_DRAFT_REVISION,
    DFLASH2_TARGET_REVISION,
    resolve_dflash2_bundle_paths,
)
from tests.dflash2_test_bundle import (
    exact_draft_config,
    exact_target_config,
    write_exact_bundle,
)


def _write_bundle(
    tmp_path,
    monkeypatch,
    *,
    target_hidden=5120,
    draft_hidden=5120,
    target_vocab=248320,
    draft_vocab=None,
):
    draft_vocab = target_vocab if draft_vocab is None else draft_vocab
    return write_exact_bundle(
        tmp_path,
        monkeypatch=monkeypatch,
        target_config=exact_target_config(
            text_config={"hidden_size": target_hidden, "vocab_size": target_vocab}
        ),
        draft_config=exact_draft_config(
            hidden_size=draft_hidden,
            vocab_size=draft_vocab,
            # These are draft geometry, not a target compatibility equality.
            num_attention_heads=3,
            num_key_value_heads=1,
        ),
    )


def test_dflash2_bundle_resolves_and_selects_backend(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)

    paths = resolve_dflash2_bundle_paths(bundle)
    result = inspect_model(bundle)

    assert paths is not None
    assert paths["bundle_root"] == str(bundle)
    assert paths["target_model"] == str(bundle / "target")
    assert paths["draft_model"] == str(bundle / "dflash2")
    assert result.passes_primary_gate is True
    assert result.compatibility["recommended_backend"] == DFLASH2_BACKEND
    assert result.compatibility["runtime_compatibility"] == "dflash2-bundle-native"
    assert result.dflash2_bundle["target_revision"] == DFLASH2_TARGET_REVISION
    assert result.dflash2_bundle["draft_revision"] == DFLASH2_DRAFT_REVISION
    assert result.dflash2_bundle["algorithm_revision"] == DFLASH2_ALGORITHM_REVISION
    assert result.dflash2_bundle["draft_precision"] == "4bit"
    assert result.recommended_sampler == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert descriptor_from_inspection(result.to_dict()) is DFLASH2_DESCRIPTOR
    assert result.compatibility["arch_id"] == DFLASH2_ARCH_ID


def test_dflash2_bundle_does_not_require_attention_geometry_match(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    result = inspect_model(bundle)
    assert result.passes_primary_gate is True


def test_dflash2_bundle_rejects_hidden_size_mismatch(tmp_path, monkeypatch):
    result = inspect_model(_write_bundle(tmp_path, monkeypatch, draft_hidden=4096))

    assert result.passes_primary_gate is False
    assert "hidden_size mismatch" in result.compatibility["message"]


def test_dflash2_bundle_rejects_missing_weights_fail_closed(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    (bundle / "dflash2" / "model.safetensors").unlink()

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "no safetensors weights" in result.compatibility["message"]
    assert "checksum references missing" in result.compatibility["message"]


def test_dflash2_bundle_rejects_layer_and_vocab_mismatch(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch, draft_vocab=32000)
    config_path = bundle / "dflash2" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_target_layers"] = 63
    config["dflash_config"]["target_layer_ids"] = [1, 15, 30, 45, 64]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "vocab_size mismatch" in result.compatibility["message"]
    assert "num_target_layers mismatch" in result.compatibility["message"]
    assert "target_layer_ids must be within target layer range" in result.compatibility["message"]


def test_dflash2_bundle_rejects_invalid_checksum_and_path_traversal(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    manifest_path = bundle / "mtplx_dflash2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["draft_config"] = {
        "path": "../outside.json",
        "sha256": "not-a-sha256",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "draft_config checksum must be a 64-character SHA-256 hex digest" in result.compatibility["message"]


def test_dflash2_bundle_hashes_weights_and_rejects_unlisted_shards(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    (bundle / "dflash2/model.safetensors").write_bytes(b"substituted")
    (bundle / "dflash2/model-00002-of-00002.safetensors").write_bytes(b"extra")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "draft_weights checksum mismatch" in result.compatibility["message"]
    assert "draft weights checksum omits files" in result.compatibility["message"]


def test_dflash2_bundle_rejects_substitution_even_when_manifest_is_rewritten(
    tmp_path, monkeypatch
):
    bundle = _write_bundle(tmp_path, monkeypatch)
    weights_path = bundle / "dflash2/model.safetensors"
    weights_path.write_bytes(b"attacker-controlled-substitute")
    manifest_path = bundle / "mtplx_dflash2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["draft_weights"]["sha256"] = hashlib.sha256(
        weights_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "trusted draft weight digest mismatch" in result.compatibility["message"]


def test_dflash2_bundle_trust_root_pins_weight_paths(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    original = bundle / "dflash2/model.safetensors"
    relocated = bundle / "dflash2/nested/model.safetensors"
    relocated.parent.mkdir()
    original.rename(relocated)
    manifest_path = bundle / "mtplx_dflash2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["draft_weights"]["path"] = (
        "dflash2/nested/model.safetensors"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "trusted draft weight inventory mismatch" in result.compatibility["message"]


def test_dflash2_bundle_rejects_unmeasured_revisions(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    manifest_path = bundle / "mtplx_dflash2.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target"]["revision"] = "different-target"
    manifest["draft"]["revision"] = "different-draft"
    manifest["algorithm"]["revision"] = "different-runtime"
    manifest_path.write_text(json.dumps(manifest))

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "target.revision must equal the measured identity" in result.compatibility["message"]
    assert "draft.revision must equal the measured identity" in result.compatibility["message"]
    assert "algorithm.revision must equal the measured identity" in result.compatibility["message"]


def test_dflash2_bundle_rejects_non_qwen_or_non_dflash_configs(tmp_path, monkeypatch):
    bundle = _write_bundle(tmp_path, monkeypatch)
    target_config = json.loads((bundle / "target/config.json").read_text())
    target_config["model_type"] = "llama"
    target_config["architectures"] = ["LlamaForCausalLM"]
    (bundle / "target/config.json").write_text(json.dumps(target_config))
    draft_config = json.loads((bundle / "dflash2/config.json").read_text())
    draft_config["architectures"] = ["Qwen3ForCausalLM"]
    draft_config["model_type"] = "qwen3"
    (bundle / "dflash2/config.json").write_text(json.dumps(draft_config))

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "not the exact Optimized-Speed contract" in result.compatibility["message"]
    assert "not a DFlash2 draft" in result.compatibility["message"]
