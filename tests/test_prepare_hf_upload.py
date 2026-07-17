"""Upload preparation for Hugging Face (scripts/prepare_hf_upload.py).

Dry-run planning, size/limit findings, sampled hash spot-checks, sharded
layout staging, model card rendering, and the HF_TOKEN gate.  Everything
runs against the tiny fixture artifact; no network and no real uploads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mtplx.expert_manifest import (
    build_expert_sidecar,
    plan_sidecar_shards,
    save_expert_manifest,
    write_sidecar_shards,
)
from tests.test_expert_slots_runtime import _global_artifact


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_hf_upload.py"
    spec = importlib.util.spec_from_file_location("prepare_hf_upload", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("prepare_hf_upload", module)
    spec.loader.exec_module(module)
    return module


def _artifact_with_manifest(tmp_path):
    root, spec, manifest, expected = _global_artifact(tmp_path)
    manifest = build_expert_sidecar(manifest, root, root / "experts.bin")
    save_expert_manifest(manifest, root / "expert-manifest.json")
    (root / "config.json").write_text("{}\n")
    return root, spec, manifest, expected


def test_dry_run_plan_stages_manifest_metadata_and_weights(tmp_path) -> None:
    cli = _load_cli()
    root, _spec, manifest, _expected = _artifact_with_manifest(tmp_path)
    plan = cli.build_expert_plan(root)
    staged = {entry.path_in_repo for entry in plan.entries}
    assert "expert-manifest.json" in staged
    assert "config.json" in staged
    assert "experts.bin" in staged
    assert "source.bin" in staged  # fixture resident shard
    assert plan.total_bytes > 0
    # The fixture has no resident index; the plan must fail closed on it.
    assert any(
        "model.safetensors.index.json" in item.message for item in plan.errors
    )
    assert any(
        "spot-check passed" in item.message
        for item in plan.findings
        if item.severity == "info"
    )


def test_spot_check_detects_sidecar_corruption(tmp_path) -> None:
    cli = _load_cli()
    root, _spec, _manifest, _expected = _artifact_with_manifest(tmp_path)
    payload = bytearray((root / "experts.bin").read_bytes())
    payload[0] ^= 0xFF
    (root / "experts.bin").write_bytes(payload)
    plan = cli.build_expert_plan(root)
    assert any("spot-check failed" in item.message for item in plan.errors)


def test_sharded_layout_preferred_and_monolith_excluded(tmp_path) -> None:
    cli = _load_cli()
    root, _spec, manifest, _expected = _artifact_with_manifest(tmp_path)
    plans = plan_sidecar_shards(manifest, max_shard_bytes=24_000)
    sharded = write_sidecar_shards(manifest, root, plans)
    save_expert_manifest(sharded, root / "expert-manifest-sharded.json")
    plan = cli.build_expert_plan(root)
    staged = {entry.path_in_repo for entry in plan.entries}
    for shard_plan in plans:
        assert shard_plan.name in staged
    assert "experts.bin" not in staged
    manifest_entry = next(
        entry
        for entry in plan.entries
        if entry.path_in_repo == "expert-manifest.json"
    )
    assert "renamed" in manifest_entry.note
    excluded_names = {name for name, _size, _reason in plan.excluded}
    assert "experts.bin" in excluded_names


def test_oversize_monolith_gets_sharding_remediation(tmp_path, monkeypatch) -> None:
    cli = _load_cli()
    root, _spec, _manifest, _expected = _artifact_with_manifest(tmp_path)
    monkeypatch.setattr(cli, "HUB_HARD_LIMIT_BYTES", 10_000)
    monkeypatch.setattr(cli, "HUB_RECOMMENDED_BYTES", 5_000)
    plan = cli.build_expert_plan(root)
    assert any("hard cap" in item.message for item in plan.errors)
    assert any(
        "shard_expert_sidecar.py" in item.message for item in plan.errors
    )


def test_stray_experiment_files_reported_not_staged(tmp_path) -> None:
    cli = _load_cli()
    root, _spec, _manifest, _expected = _artifact_with_manifest(tmp_path)
    (root / "experts-huff-all.bin").write_bytes(b"x" * 64)
    (root / "experts-banked-39-59.bin").write_bytes(b"x" * 64)
    plan = cli.build_expert_plan(root)
    staged = {entry.path_in_repo for entry in plan.entries}
    assert "experts-huff-all.bin" not in staged
    assert "experts-banked-39-59.bin" not in staged
    reclamation = {
        name
        for name, _size, reason in plan.excluded
        if "reclamation" in reason
    }
    assert reclamation == {"experts-huff-all.bin", "experts-banked-39-59.bin"}


def test_render_expert_card_carries_provenance_and_format(tmp_path) -> None:
    cli = _load_cli()
    root, _spec, manifest, _expected = _artifact_with_manifest(tmp_path)
    card = cli.render_card(root)
    assert manifest.source_repo in card
    assert manifest.source_revision in card
    assert f"{manifest.quant_bits}-bit affine" in card
    assert "record-major" in card
    assert "16 KiB-aligned" in card
    assert "shard-absolute" in card
    assert "mtplx" in card
    assert "license" in card.lower()


def test_mtp_plan_and_card(tmp_path) -> None:
    cli = _load_cli()
    root = tmp_path / "mtp-artifact"
    root.mkdir()
    payload = b"\x00" * 128
    (root / "layer99-bf16.safetensors").write_bytes(payload)
    (root / "config.json").write_text(json.dumps({"architectures": ["Test"]}))
    manifest = {
        "schema": "mtplx-test-mtp-v1",
        "artifact": {
            "file": "layer99-bf16.safetensors",
            "file_bytes": len(payload),
            "header_bytes": 0,
            "tensors": [],
        },
        "producer": {"commit": "abc123"},
        "source": {"files": []},
        "manifest_sha256": "0" * 64,
    }
    (root / "mtp-artifact-manifest.json").write_text(json.dumps(manifest))
    plan = cli.build_plan(root)
    assert plan.kind == "mtp-layer"
    staged = {entry.path_in_repo for entry in plan.entries}
    assert "layer99-bf16.safetensors" in staged
    assert "mtp-artifact-manifest.json" in staged
    assert not plan.errors
    card = cli.render_card(root)
    assert "MTP layer" in card
    assert "abc123" in card


def test_execute_requires_hf_token_and_never_uploads_without_it(
    tmp_path, monkeypatch, capsys
) -> None:
    cli = _load_cli()
    root, _spec, _manifest, _expected = _artifact_with_manifest(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    exit_code = cli.main(
        [str(root), "--execute", "--repo-id", "example/does-not-matter"]
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "HF_TOKEN" in captured.err


def test_dry_run_exit_reflects_validation_errors(tmp_path, capsys) -> None:
    cli = _load_cli()
    root, _spec, _manifest, _expected = _artifact_with_manifest(tmp_path)
    exit_code = cli.main([str(root)])
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    # Fixture lacks the resident index -> dry run reports and exits nonzero.
    assert exit_code == 1
    assert "validation errors" in captured.err
