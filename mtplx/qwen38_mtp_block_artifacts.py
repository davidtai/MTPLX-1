"""Chronological Qwen 3.8 MTP-block artifact candidates.

Rows 17 and 28 replace the complete one-layer MTP block, not the draft-only
vocabulary projection.  Keep those two surfaces separate so a Q4 vocabulary
head can never be mistaken for a Q4 MTP block again.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Qwen38MTPBlockArtifactError(RuntimeError):
    """Raised when a pinned challenge MTP-block artifact is not exact."""


@dataclass(frozen=True)
class Qwen38MTPBlockArtifactSpec:
    variant: str
    source_commit: str
    manifest_sha256: str
    file_sha256: str
    bytes: int


QWEN38_MTP_BLOCK_ARTIFACTS = {
    "r17": Qwen38MTPBlockArtifactSpec(
        variant="r17",
        source_commit="deb63ad0d1701d9d14cacd34d901ae7c0588c432",
        manifest_sha256="cc209e30d8a7def1fc4d785be22b0ec40e16ae6763f9591255a1996a34f08f0d",
        file_sha256="0e267a482e74c2664ce41dc4c4326f480020d015372fc9f7654ea3a136d62815",
        bytes=238_934_093,
    ),
    "r28": Qwen38MTPBlockArtifactSpec(
        variant="r28",
        source_commit="6209702fba83a744eb3deb598905d59978f9e5e7",
        manifest_sha256="7d62702795865b9036afe4bddcd16a2a8eb973c0caced15e5243139dda067f47",
        file_sha256="c934b40f1254858425cc0b5fdfe62b6ae13d1a4aff74da9d81606e92fdcf41ee",
        bytes=238_934_129,
    ),
}

_Q4_LINEAR_SHAPES = {
    "fc": ((5120, 1280), (5120, 160)),
    "layers.0.mlp.down_proj": ((5120, 2176), (5120, 272)),
    "layers.0.mlp.gate_proj": ((17408, 640), (17408, 80)),
    "layers.0.mlp.up_proj": ((17408, 640), (17408, 80)),
    "layers.0.self_attn.k_proj": ((1024, 640), (1024, 80)),
    "layers.0.self_attn.o_proj": ((5120, 768), (5120, 96)),
    "layers.0.self_attn.q_proj": ((12288, 640), (12288, 80)),
    "layers.0.self_attn.v_proj": ((1024, 640), (1024, 80)),
}
_NORM_SHAPES = {
    "layers.0.input_layernorm.weight": (5120,),
    "layers.0.post_attention_layernorm.weight": (5120,),
    "layers.0.self_attn.k_norm.weight": (256,),
    "layers.0.self_attn.q_norm.weight": (256,),
    "norm.weight": (5120,),
    "pre_fc_norm_embedding.weight": (5120,),
    "pre_fc_norm_hidden.weight": (5120,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tensors(tensors: dict[str, Any]) -> None:
    expected = set(_NORM_SHAPES)
    for prefix in _Q4_LINEAR_SHAPES:
        expected.update((f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"))
    if set(tensors) != expected:
        missing = sorted(expected - set(tensors))
        extra = sorted(set(tensors) - expected)
        raise Qwen38MTPBlockArtifactError(
            f"MTP block tensor set mismatch: missing={missing}, extra={extra}"
        )
    for prefix, (weight_shape, scale_shape) in _Q4_LINEAR_SHAPES.items():
        weight = tensors[f"{prefix}.weight"]
        scales = tensors[f"{prefix}.scales"]
        biases = tensors[f"{prefix}.biases"]
        if tuple(weight.shape) != weight_shape or str(weight.dtype) != "mlx.core.uint32":
            raise Qwen38MTPBlockArtifactError(f"invalid Q4 weight geometry for {prefix}")
        for name, value in (("scales", scales), ("biases", biases)):
            if tuple(value.shape) != scale_shape or str(value.dtype) != "mlx.core.bfloat16":
                raise Qwen38MTPBlockArtifactError(
                    f"invalid Q4 {name} geometry for {prefix}"
                )
    for key, shape in _NORM_SHAPES.items():
        value = tensors[key]
        if tuple(value.shape) != shape or str(value.dtype) != "mlx.core.bfloat16":
            raise Qwen38MTPBlockArtifactError(f"invalid BF16 norm geometry for {key}")


def configure_qwen38_mtp_block(
    runtime: Any,
    *,
    variant: str | None,
    artifact_path: Path | None,
) -> dict[str, Any]:
    """Switch the live one-layer MTP block between BF16 control and Q4 rows."""

    text = getattr(runtime.model, "language_model", runtime.model)
    if not hasattr(text, "_mtplx_qwen38_control_mtp_block"):
        text._mtplx_qwen38_control_mtp_block = text.mtp
        text._mtplx_qwen38_mtp_block_variants = {}

    control = text._mtplx_qwen38_control_mtp_block
    if variant is None:
        text.mtp = control
        if runtime.model is not text and hasattr(runtime.model, "mtp"):
            runtime.model.mtp = control
        return {
            "installed": True,
            "active": False,
            "variant": None,
            "bits": None,
            "group_size": None,
        }

    spec = QWEN38_MTP_BLOCK_ARTIFACTS.get(variant)
    if spec is None:
        raise Qwen38MTPBlockArtifactError(f"unknown MTP-block variant: {variant!r}")
    if artifact_path is None:
        raise Qwen38MTPBlockArtifactError(f"{variant} requires a pinned artifact path")
    path = artifact_path.expanduser().resolve()
    if not path.is_file():
        raise Qwen38MTPBlockArtifactError(f"MTP-block artifact is missing: {path}")
    size = path.stat().st_size
    if size != spec.bytes:
        raise Qwen38MTPBlockArtifactError(
            f"{variant} artifact bytes mismatch: expected {spec.bytes}, got {size}"
        )
    file_sha256 = _sha256(path)
    if file_sha256 != spec.file_sha256:
        raise Qwen38MTPBlockArtifactError(
            f"{variant} artifact sha256 mismatch: expected {spec.file_sha256}, got {file_sha256}"
        )

    variants = text._mtplx_qwen38_mtp_block_variants
    if variant not in variants:
        import mlx.core as mx
        import mlx.nn as nn

        tensors = mx.load(str(path), format="safetensors")
        _validate_tensors(tensors)
        candidate = copy.deepcopy(control)
        nn.quantize(candidate, group_size=64, bits=4, mode="affine")
        candidate.load_weights(list(tensors.items()), strict=True)
        mx.eval(candidate.parameters())
        variants[variant] = candidate

    candidate = variants[variant]
    text.mtp = candidate
    if runtime.model is not text and hasattr(runtime.model, "mtp"):
        runtime.model.mtp = candidate
    return {
        "installed": True,
        "active": True,
        "variant": variant,
        "source_commit": spec.source_commit,
        "manifest_sha256": spec.manifest_sha256,
        "file_sha256": file_sha256,
        "artifact_bytes": size,
        "artifact_path": str(path),
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "linear_modules": len(_Q4_LINEAR_SHAPES),
    }
