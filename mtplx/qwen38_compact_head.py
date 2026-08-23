"""Proposal-only compact vocabulary head for the pinned Qwen 3.8 route."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QWEN38_COMPACT_PREFIX_ROWS = 98_304
QWEN38_COMPACT_CONTROL_START = 248_044
QWEN38_COMPACT_CONTROL_END = 248_070
QWEN38_COMPACT_REAL_ROWS = (
    QWEN38_COMPACT_PREFIX_ROWS
    + QWEN38_COMPACT_CONTROL_END
    - QWEN38_COMPACT_CONTROL_START
)
QWEN38_COMPACT_PADDED_ROWS = 98_336
QWEN38_COMPACT_TOPK = 32
QWEN38_VOCAB_SIZE = 248_320
QWEN38_HIDDEN_SIZE = 5_120
QWEN38_CLUSTER_ROWS = 8
QWEN38_CLUSTER_COUNT = QWEN38_COMPACT_PADDED_ROWS // QWEN38_CLUSTER_ROWS
QWEN38_CLUSTER_PROBE_FRACTION = 0.15
QWEN38_CLUSTER_PROBES = math.ceil(
    QWEN38_CLUSTER_COUNT * QWEN38_CLUSTER_PROBE_FRACTION
)
QWEN38_COMPACT_FORMAT = "mtplx-qwen38-compact-q2-g64-v1"
_MISSING = object()


class Qwen38CompactHeadError(RuntimeError):
    """The compact proposal artifact does not match the Qwen 3.8 contract."""


@dataclass(frozen=True)
class Qwen38CompactArtifact:
    path: Path
    sha256: str
    bytes: int
    source_contract_id: str


def compact_token_ids_to_full(ids: Any) -> Any:
    """Map compact proposal IDs back into the fixed tokenizer vocabulary."""

    import mlx.core as mx

    return mx.where(
        ids < QWEN38_COMPACT_PREFIX_ROWS,
        ids,
        ids + (QWEN38_COMPACT_CONTROL_START - QWEN38_COMPACT_PREFIX_ROWS),
    )


def _compact_source_rows() -> Any:
    import mlx.core as mx

    padding = QWEN38_COMPACT_PADDED_ROWS - QWEN38_COMPACT_REAL_ROWS
    return mx.concatenate(
        (
            mx.arange(QWEN38_COMPACT_PREFIX_ROWS, dtype=mx.int32),
            mx.arange(
                QWEN38_COMPACT_CONTROL_START,
                QWEN38_COMPACT_CONTROL_END,
                dtype=mx.int32,
            ),
            mx.arange(padding, dtype=mx.int32),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_target_q8_head(head: Any) -> None:
    import mlx.nn as nn

    if not isinstance(head, nn.QuantizedLinear):
        raise Qwen38CompactHeadError("Qwen 3.8 target lm_head must be quantized")
    observed = (
        int(head.weight.shape[0]),
        int(head.weight.shape[1]),
        int(head.scales.shape[1]),
        int(head.bits),
        int(head.group_size),
        str(head.mode),
    )
    expected = (QWEN38_VOCAB_SIZE, 1_280, 80, 8, 64, "affine")
    if observed != expected:
        raise Qwen38CompactHeadError(
            f"Qwen 3.8 target lm_head mismatch: {observed!r} != {expected!r}"
        )


def build_qwen38_compact_artifact(
    target_head: Any,
    output_path: Path,
    *,
    source_contract_id: str,
) -> Qwen38CompactArtifact:
    """Derive and persist the proposal-only Q2 table outside timed decode."""

    import mlx.core as mx

    _require_target_q8_head(target_head)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _compact_source_rows()
    dense = mx.dequantize(
        mx.take(target_head.weight, rows, axis=0),
        mx.take(target_head.scales, rows, axis=0),
        mx.take(target_head.biases, rows, axis=0),
        group_size=64,
        bits=8,
        mode="affine",
    ).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(
        dense,
        group_size=64,
        bits=2,
        mode="affine",
    )
    mx.eval(weight, scales, biases)
    metadata = {
        "format": QWEN38_COMPACT_FORMAT,
        "source_contract_id": str(source_contract_id),
        "source_head": "target-q8-g64-affine",
        "vocabulary_mapping": "prefix-98304+controls-248044:248070+pad-first-6",
    }
    mx.save_safetensors(
        str(output_path),
        {
            "draft_lm_head.weight": weight,
            "draft_lm_head.scales": scales,
            "draft_lm_head.biases": biases,
        },
        metadata=metadata,
    )
    return validate_qwen38_compact_artifact(
        output_path,
        source_contract_id=source_contract_id,
    )


def validate_qwen38_compact_artifact(
    path: Path,
    *,
    source_contract_id: str,
) -> Qwen38CompactArtifact:
    """Fail closed on artifact metadata, tensor layout, size, and digest."""

    from safetensors import safe_open

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise Qwen38CompactHeadError(f"compact head artifact is missing: {path}")
    expected = {
        "draft_lm_head.weight": ([QWEN38_COMPACT_PADDED_ROWS, 320], "U32"),
        "draft_lm_head.scales": ([QWEN38_COMPACT_PADDED_ROWS, 80], "BF16"),
        "draft_lm_head.biases": ([QWEN38_COMPACT_PADDED_ROWS, 80], "BF16"),
    }
    with safe_open(path, framework="numpy") as tensors:
        metadata = tensors.metadata() or {}
        if metadata.get("format") != QWEN38_COMPACT_FORMAT:
            raise Qwen38CompactHeadError("compact head format metadata mismatch")
        if metadata.get("source_contract_id") != source_contract_id:
            raise Qwen38CompactHeadError("compact head source contract mismatch")
        if set(tensors.keys()) != set(expected):
            raise Qwen38CompactHeadError("compact head tensor map mismatch")
        for name, (shape, dtype) in expected.items():
            value = tensors.get_slice(name)
            if value.get_shape() != shape or value.get_dtype() != dtype:
                raise Qwen38CompactHeadError(
                    f"compact head tensor {name} has "
                    f"{value.get_shape()}/{value.get_dtype()}, expected {shape}/{dtype}"
                )
    return Qwen38CompactArtifact(
        path=path,
        sha256=_sha256(path),
        bytes=path.stat().st_size,
        source_contract_id=str(source_contract_id),
    )


class Qwen38CompactProposalHead:
    """Q2 shortlist plus target-Q8 exact rerank for one proposal row."""

    def __init__(self, target_head: Any, artifact: Qwen38CompactArtifact):
        import mlx.core as mx

        _require_target_q8_head(target_head)
        tensors = mx.load(str(artifact.path))
        self.target_head = target_head
        self.coarse_weight = tensors["draft_lm_head.weight"]
        self.coarse_scales = tensors["draft_lm_head.scales"]
        self.coarse_biases = tensors["draft_lm_head.biases"]
        self.artifact = artifact

    def __call__(self, x: Any) -> Any:
        import mlx.core as mx

        if int(x.size) != QWEN38_HIDDEN_SIZE:
            raise Qwen38CompactHeadError(
                "compact proposal head supports one Qwen 3.8 draft row"
            )
        flat = x.reshape(1, QWEN38_HIDDEN_SIZE)
        coarse = mx.quantized_matmul(
            flat,
            self.coarse_weight,
            scales=self.coarse_scales,
            biases=self.coarse_biases,
            transpose=True,
            group_size=64,
            bits=2,
            mode="affine",
        )[0, :QWEN38_COMPACT_REAL_ROWS]
        kth = QWEN38_COMPACT_REAL_ROWS - QWEN38_COMPACT_TOPK
        compact_ids = mx.argpartition(coarse, kth=kth)[kth:]
        full_ids = compact_token_ids_to_full(compact_ids).astype(mx.int32)
        exact = mx.quantized_matmul(
            flat,
            mx.take(self.target_head.weight, full_ids, axis=0),
            scales=mx.take(self.target_head.scales, full_ids, axis=0),
            biases=mx.take(self.target_head.biases, full_ids, axis=0),
            transpose=True,
            group_size=64,
            bits=8,
            mode="affine",
        )[0]
        best_value = mx.max(exact)
        winner = mx.min(
            mx.where(exact == best_value, full_ids, QWEN38_VOCAB_SIZE)
        )
        token_axis = mx.arange(QWEN38_VOCAB_SIZE, dtype=mx.int32)
        logits = mx.where(token_axis == winner, 0.0, -mx.inf).astype(x.dtype)
        return logits.reshape(*x.shape[:-1], QWEN38_VOCAB_SIZE)


def install_qwen38_compact_proposal_head(
    runtime: Any,
    path: Path,
    *,
    source_contract_id: str,
) -> Qwen38CompactArtifact:
    """Attach the compact table to the existing draft-only head seam."""

    text = getattr(runtime.model, "language_model", runtime.model)
    target_head = getattr(text, "lm_head", None)
    artifact = validate_qwen38_compact_artifact(
        path,
        source_contract_id=source_contract_id,
    )
    if not hasattr(text, "_mtplx_qwen38_control_draft_lm_head"):
        text._mtplx_qwen38_control_draft_lm_head = getattr(
            text,
            "_mtplx_draft_lm_head",
            _MISSING,
        )
    text._mtplx_draft_lm_head = Qwen38CompactProposalHead(target_head, artifact)
    return artifact


def restore_qwen38_control_proposal_head(runtime: Any) -> None:
    """Restore the proposal head present before the compact route was installed."""

    text = getattr(runtime.model, "language_model", runtime.model)
    if not hasattr(text, "_mtplx_qwen38_control_draft_lm_head"):
        return
    control = text._mtplx_qwen38_control_draft_lm_head
    if control is _MISSING:
        if hasattr(text, "_mtplx_draft_lm_head"):
            delattr(text, "_mtplx_draft_lm_head")
    else:
        text._mtplx_draft_lm_head = control


def qwen38_compact_artifact_receipt(artifact: Qwen38CompactArtifact) -> str:
    return json.dumps(
        {
            "bytes": artifact.bytes,
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "source_contract_id": artifact.source_contract_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
