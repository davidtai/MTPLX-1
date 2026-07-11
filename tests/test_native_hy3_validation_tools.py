"""CLI and helper contracts for native Hy3 artifact validation tools."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize(
    "script_name",
    ["validate_hy3_native_quantization", "evaluate_streamed_perplexity"],
)
def test_run_lock_helper_matches_only_clear_or_occupied_processes(
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
) -> None:
    module = _load_script(script_name)
    calls = []

    def occupied(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="101 python scripts/benchmark_streamed_generation.py\n\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", occupied)
    assert module._run_lock_processes() == [
        "101 python scripts/benchmark_streamed_generation.py"
    ]
    assert calls[0][0] == [
        "pgrep",
        "-fl",
        r"python.*(benchmark_streamed|probe_mtp)",
    ]
    assert calls[0][1] == {
        "check": False,
        "capture_output": True,
        "text": True,
    }

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    assert module._run_lock_processes() == []

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(RuntimeError, match="permission denied"):
        module._run_lock_processes()


def test_native_quantization_cli_refuses_an_occupied_run_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("validate_hy3_native_quantization")
    monkeypatch.setattr(
        module,
        "_run_lock_processes",
        lambda: ["101 python scripts/probe_mtp_draft_rank.py"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_hy3_native_quantization.py", "/source", "/artifact"],
    )

    with pytest.raises(SystemExit, match="run-lock is occupied"):
        module.main()


def test_fixed_vector_binding_uses_a_writable_slot_buffer() -> None:
    module = _load_script("validate_hy3_native_quantization")
    segment = SimpleNamespace(component="gate_proj.weight", length=4)
    record = SimpleNamespace(segments=(segment,))

    binding = module._expert_binding(
        record,
        b"\x01\x02\x03\x04",
        layer=1,
        expert=2,
    )
    view = binding.component_view("gate_proj.weight")

    assert isinstance(binding.buffer, bytearray)
    assert view.readonly is False
    view[0] = 9
    assert binding.buffer[0] == 9


def test_router_helper_honors_route_norm() -> None:
    module = _load_script("validate_hy3_native_quantization")
    logits = mx.array([[0.0, 0.0, 0.0, 1.0]], dtype=mx.float32)
    correction = mx.array([0.0, 0.1, 0.2, 0.3], dtype=mx.float32)
    _ids, normalized = module._route(
        logits,
        correction,
        top_k=2,
        scale=2.0,
        route_norm=True,
    )
    _ids, raw = module._route(
        logits,
        correction,
        top_k=2,
        scale=2.0,
        route_norm=False,
    )
    mx.eval(normalized, raw)

    assert float(mx.sum(normalized).item()) == pytest.approx(2.0)
    assert float(mx.sum(raw).item()) > 2.4
    assert [float(value) for value in raw[0].tolist()] == pytest.approx([1.462117, 1.0])


def test_perplexity_helper_fails_cleanly_for_non_finite_or_overflowing_nll() -> None:
    module = _load_script("evaluate_streamed_perplexity")

    assert module._perplexity(0.0) == 1.0
    assert module._perplexity(math.log(4.0)) == pytest.approx(4.0)
    assert module._perplexity(-1.0) is None
    assert module._perplexity(float("nan")) is None
    assert module._perplexity(float("inf")) is None
    assert module._perplexity(math.log(sys.float_info.max) + 1.0) is None


def test_perplexity_text_helper_rejects_conflicts_and_empty_samples(
    tmp_path: Path,
) -> None:
    module = _load_script("evaluate_streamed_perplexity")
    text_file = tmp_path / "sample.txt"
    text_file.write_text("from file", encoding="utf-8")

    assert (
        module._text(argparse.Namespace(text=None, text_file=text_file)) == "from file"
    )
    with pytest.raises(SystemExit, match="choose only one"):
        module._text(argparse.Namespace(text="inline", text_file=text_file))
    with pytest.raises(SystemExit, match="empty"):
        module._text(argparse.Namespace(text=" \n", text_file=None))


def test_perplexity_cli_refuses_an_occupied_run_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("evaluate_streamed_perplexity")
    monkeypatch.setattr(
        module,
        "_run_lock_processes",
        lambda: ["202 python scripts/benchmark_streamed_generation.py"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_streamed_perplexity.py", "/artifact", "/manifest"],
    )

    with pytest.raises(SystemExit, match="run-lock is occupied"):
        module.main()


def test_layout_cli_helpers_are_fail_closed(tmp_path: Path) -> None:
    module = _load_script("audit_streamed_model_layout")
    root = tmp_path / "artifact"
    root.mkdir()
    legacy = root / "expert-manifest-sidecar.json"
    legacy.write_text("{}", encoding="utf-8")
    assert module._manifest_path(root, None) == legacy
    preferred = root / "expert-manifest.json"
    preferred.write_text("{}", encoding="utf-8")
    assert module._manifest_path(root, None) == preferred
    assert module._is_routed_tensor("model.layers.1.mlp.switch_mlp.gate_proj.weight")
    assert module._is_routed_tensor("model.layers.80.mlp.experts.0.up_proj.scales")
    assert not module._is_routed_tensor("model.layers.1.mlp.router.gate.weight")

    with pytest.raises(SystemExit, match="requires --artifact-root"):
        module._validate_cli_args(
            argparse.Namespace(manifest=preferred, artifact_root=None)
        )


def test_layout_inventory_reads_metadata_without_loading_tensors(
    tmp_path: Path,
) -> None:
    module = _load_script("audit_streamed_model_layout")
    header = {
        "__metadata__": {
            "source_repo": "source/repo",
            "source_revision": "revision",
        },
        "model.layers.80.enorm.weight": {
            "dtype": "U8",
            "shape": [4],
            "data_offsets": [0, 4],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path = tmp_path / "layer80-bf16.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"data")

    names, tensor_bytes, metadata = module._safetensors_inventory(path)

    assert names == {"model.layers.80.enorm.weight"}
    assert tensor_bytes == 4
    assert metadata == {
        "source_repo": "source/repo",
        "source_revision": "revision",
    }


def test_compact_layout_rejects_shards_that_still_embed_experts() -> None:
    module = _load_script("audit_streamed_model_layout")
    spec = SimpleNamespace(
        expert_record_layer_indices=(1,),
        expert_count=1,
        key="tiny-native",
        manifest_repo="source/repo",
        manifest_revision="revision",
        quant_bits=4,
        quant_group_size=64,
        total_tensor_bytes=12,
        resident_bytes=4,
        routed_expert_bytes=8,
    )
    record = SimpleNamespace(
        layer=1,
        expert=0,
        logical_bytes=8,
        sidecar_offset=0,
        sidecar_length=8,
        segments=(SimpleNamespace(shard="experts.bin"),),
    )
    manifest = SimpleNamespace(
        records=(record,),
        resident_tensors=(
            SimpleNamespace(
                tensor="model.embed_tokens.weight",
                shard="model.safetensors",
            ),
        ),
        sidecar=SimpleNamespace(file="experts.bin"),
        shards=(
            SimpleNamespace(
                kind="safetensors",
                size=14,
                header_bytes=10,
            ),
            SimpleNamespace(kind="sidecar", size=8, header_bytes=0),
        ),
        model_key=spec.key,
        source_repo=spec.manifest_repo,
        source_revision=spec.manifest_revision,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        artifact_tensor_bytes=12,
        resident_tensor_bytes=4,
        routed_expert_bytes=8,
    )

    report = module._compact_manifest_audit(
        manifest,
        spec,
        {"model.embed_tokens.weight": "model.safetensors"},
    )
    assert report["valid"] is True
    assert report["sidecar_authoritative"] is True
    assert report["resident_shards_are_compact"] is True
    assert report["resident_shard_mapping_matches"] is True

    mismatched = module._compact_manifest_audit(
        manifest,
        spec,
        {"model.embed_tokens.weight": "duplicated-residents.safetensors"},
    )
    assert mismatched["valid"] is False
    assert mismatched["resident_shard_mapping_matches"] is False

    manifest.shards = (
        SimpleNamespace(kind="safetensors", size=22, header_bytes=10),
        SimpleNamespace(kind="sidecar", size=8, header_bytes=0),
    )
    report = module._compact_manifest_audit(
        manifest,
        spec,
        {"model.embed_tokens.weight": "model.safetensors"},
    )
    assert report["valid"] is False
    assert report["resident_shards_are_compact"] is False
