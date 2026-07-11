#!/usr/bin/env python3
"""Validate a compact native Hy3 Q4 artifact against its pinned BF16 source.

This is a correctness probe, not a throughput benchmark.  It exercises one
sidecar expert on deterministic vectors, checks the streamed record against
the direct affine-Q4 kernels, reports BF16-vs-Q4 quantization error, and proves
the resident Q8 router matches a fresh quantization of the official weights.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.activations import swiglu  # noqa: E402

from mtplx.expert_manifest import load_expert_manifest, read_expert_record  # noqa: E402
from mtplx.expert_slots import ExpertSlotBinding  # noqa: E402
from mtplx.expert_streaming_models import get_model_spec  # noqa: E402
from mtplx.models.expert_mlx import _run_q4_expert  # noqa: E402


KERNEL_ATOL = 1e-5
KERNEL_RTOL = 1e-5
Q4_WEIGHT_COSINE_MIN = 0.99
Q8_WEIGHT_COSINE_MIN = 0.999


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-key", default="hy3-q4-native")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--vectors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_711)
    parser.add_argument("--kernel-atol", type=float, default=KERNEL_ATOL)
    parser.add_argument("--kernel-rtol", type=float, default=KERNEL_RTOL)
    return parser


def _run_lock_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-fl", r"python.*(benchmark_streamed|probe_mtp)"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"pgrep failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _index(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{path} has no weight_map")
    return {str(name): str(shard) for name, shard in weight_map.items()}


class _TensorReader:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.weight_map = _index(self.root)
        self._loaded: dict[str, dict[str, Any]] = {}

    def get(self, name: str) -> Any:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"{name} is absent from {self.root}") from exc
        tensors = self._loaded.get(shard)
        if tensors is None:
            tensors = mx.load(str(self.root / shard), format="safetensors")
            self._loaded[shard] = tensors
        try:
            return tensors[name]
        except KeyError as exc:
            raise KeyError(f"{name} is absent from {shard}") from exc


def _as_float(value: Any) -> Any:
    return value.astype(mx.float32)


def _cosine(left: Any, right: Any) -> float:
    a = _as_float(left).reshape((-1,))
    b = _as_float(right).reshape((-1,))
    result = mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))
    mx.eval(result)
    return float(result.item())


def _error_stats(actual: Any, reference: Any) -> dict[str, float]:
    delta = mx.abs(_as_float(actual) - _as_float(reference)).reshape((-1,))
    mx.eval(delta)
    host = np.asarray(delta, dtype=np.float32)
    return {
        "max_abs": float(host.max(initial=0.0)),
        "mean_abs": float(host.mean()) if host.size else 0.0,
        "p99_abs": float(np.quantile(host, 0.99)) if host.size else 0.0,
        "cosine": _cosine(actual, reference),
    }


def _decode_record(record: Any, payload: bytes) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    cursor = 0
    for segment in record.segments:
        end = cursor + segment.length
        raw = payload[cursor:end]
        if len(raw) != segment.length:
            raise ValueError(f"truncated {segment.component}")
        if segment.dtype == "U32":
            host = np.frombuffer(raw, dtype=np.dtype("<u4")).copy()
            value = mx.array(host)
        elif segment.dtype == "BF16":
            host = np.frombuffer(raw, dtype=np.dtype("<u2")).copy()
            value = mx.array(host).view(mx.bfloat16)
        else:
            raise ValueError(f"unsupported record dtype {segment.dtype!r}")
        arrays[segment.component] = value.reshape(segment.shape)
        cursor = end
    if cursor != len(payload):
        raise ValueError("record has trailing bytes")
    return arrays


def _direct_q4_expert(x: Any, arrays: dict[str, Any], *, group_size: int) -> Any:
    gate = mx.quantized_matmul(
        x,
        arrays["gate_proj.weight"],
        scales=arrays["gate_proj.scales"],
        biases=arrays["gate_proj.biases"],
        group_size=group_size,
        bits=4,
    )
    up = mx.quantized_matmul(
        x,
        arrays["up_proj.weight"],
        scales=arrays["up_proj.scales"],
        biases=arrays["up_proj.biases"],
        group_size=group_size,
        bits=4,
    )
    return mx.quantized_matmul(
        swiglu(gate, up),
        arrays["down_proj.weight"],
        scales=arrays["down_proj.scales"],
        biases=arrays["down_proj.biases"],
        group_size=group_size,
        bits=4,
    )


def _bf16_expert(x: Any, source: _TensorReader, *, layer: int, expert: int) -> Any:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
    gate = x @ source.get(prefix + "gate_proj.weight").T
    up = x @ source.get(prefix + "up_proj.weight").T
    return swiglu(gate, up) @ source.get(prefix + "down_proj.weight").T


def _route(
    logits: Any,
    correction: Any,
    *,
    top_k: int,
    scale: float,
    route_norm: bool,
) -> tuple[Any, Any]:
    scores = mx.sigmoid(_as_float(logits))
    ranked = scores + _as_float(correction)
    ids = mx.argsort(ranked, axis=-1)[:, -top_k:][:, ::-1]
    selected = mx.take_along_axis(scores, ids, axis=-1)
    weights = selected
    if route_norm:
        weights = selected / (mx.sum(selected, axis=-1, keepdims=True) + 1e-20)
    return ids, weights * float(scale)


def _arrays_equal(left: Any, right: Any) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left, right).item())


def _router_contract_passes(
    *,
    leaves_match: bool,
    q8_ids_match: bool,
    q8_scores_match: bool,
    correction_match: bool,
) -> bool:
    """Evaluate only the specified resident affine-Q8 parity contract."""

    return leaves_match and q8_ids_match and q8_scores_match and correction_match


def _artifact_correction_name(spec: Any, layer: int) -> str:
    if layer == spec.mtp_layer_index:
        return f"model.layers.{layer}.mlp.expert_bias"
    return f"model.layers.{layer}.mlp.router.expert_bias"


def _router_probe(
    source: _TensorReader,
    artifact: _TensorReader,
    *,
    layer: int,
    vectors: Any,
    group_size: int,
    top_k: int,
    scale: float,
    route_norm: bool,
    atol: float,
    rtol: float,
    artifact_correction_name: str,
) -> dict[str, Any]:
    base = f"model.layers.{layer}.mlp.router.gate"
    source_weight = source.get(base + ".weight")
    weight = artifact.get(base + ".weight")
    scales = artifact.get(base + ".scales")
    biases = artifact.get(base + ".biases")
    oracle = mx.quantize(
        source_weight,
        group_size=group_size,
        bits=8,
        mode="affine",
    )
    leaves_match = all(
        _arrays_equal(actual, expected)
        for actual, expected in zip((weight, scales, biases), oracle, strict=True)
    )
    dequantized = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=group_size,
        bits=8,
        mode="affine",
    )
    q8_logits = mx.quantized_matmul(
        vectors,
        weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=8,
    )
    oracle_logits = mx.quantized_matmul(
        vectors,
        oracle[0],
        scales=oracle[1],
        biases=oracle[2],
        group_size=group_size,
        bits=8,
    )
    bf16_logits = vectors @ source_weight.T
    # Tencent's BF16 checkpoint stores the correction beside ``mlp`` while
    # the pinned MLX resident layout stores it beside the router.  The native
    # converter deliberately follows each side's established spelling. MTP
    # layer 80 keeps the official spelling in its standalone resident file.
    source_correction = source.get(f"model.layers.{layer}.mlp.expert_bias")
    artifact_correction = artifact.get(artifact_correction_name)
    q8_ids, q8_scores = _route(
        q8_logits,
        artifact_correction,
        top_k=top_k,
        scale=scale,
        route_norm=route_norm,
    )
    oracle_ids, oracle_scores = _route(
        oracle_logits,
        source_correction,
        top_k=top_k,
        scale=scale,
        route_norm=route_norm,
    )
    bf16_ids, bf16_scores = _route(
        bf16_logits,
        source_correction,
        top_k=top_k,
        scale=scale,
        route_norm=route_norm,
    )
    mx.eval(q8_ids, q8_scores, oracle_ids, oracle_scores, bf16_ids, bf16_scores)
    q8_ids_match = _arrays_equal(q8_ids, oracle_ids)
    q8_scores_match = bool(
        mx.allclose(q8_scores, oracle_scores, atol=atol, rtol=rtol).item()
    )
    bf16_ids_match = _arrays_equal(q8_ids, bf16_ids)
    correction_match = _arrays_equal(source_correction, artifact_correction)
    return {
        # The artifact contract is affine Q8/gs64, so correctness is exact
        # parity with a fresh Q8 quantization of the pinned BF16 router. Raw
        # BF16 route drift is a useful near-tie diagnostic, not a Q8 failure.
        "passed": _router_contract_passes(
            leaves_match=leaves_match,
            q8_ids_match=q8_ids_match,
            q8_scores_match=q8_scores_match,
            correction_match=correction_match,
        ),
        "reference_contract": "fresh affine Q8/gs64 from pinned BF16",
        "quantized_leaves_match_fresh_oracle": leaves_match,
        "correction_bias_bit_exact": correction_match,
        "q8_oracle_ids_match": q8_ids_match,
        "q8_oracle_scores_match": q8_scores_match,
        "bf16_ids_match": bf16_ids_match,
        "bf16_ids_diagnostic_only": True,
        "bf16_score_error": _error_stats(q8_scores, bf16_scores),
        "weight_cosine": _cosine(dequantized, source_weight),
        "weight_cosine_min": Q8_WEIGHT_COSINE_MIN,
    }


def _expert_binding(
    record: Any,
    payload: bytes,
    *,
    layer: int,
    expert: int,
) -> ExpertSlotBinding:
    """Mirror a filled runtime slot with writable host storage."""

    return ExpertSlotBinding(
        layer,
        expert,
        0,
        1,
        record,
        bytearray(payload),
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.vectors <= 0:
        raise SystemExit("--vectors must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (args.kernel_atol, args.kernel_rtol)
    ):
        raise SystemExit("kernel tolerances must be finite and non-negative")
    spec = get_model_spec(args.model_key)
    if args.layer not in spec.expert_record_layer_indices:
        raise SystemExit(f"layer {args.layer} is not packaged as expert records")
    if not 0 <= args.expert < spec.expert_count:
        raise SystemExit("--expert is outside the model")
    active = _run_lock_processes()
    if active:
        raise SystemExit(
            "benchmark/probe run-lock is occupied; refusing GPU validation:\n"
            + "\n".join(active)
        )
    print("run-lock clear; starting fixed-vector GPU validation", file=sys.stderr)

    source_root = args.source_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else artifact_root / "expert-manifest.json"
    )
    manifest = load_expert_manifest(manifest_path)
    if (
        manifest.model_key != spec.key
        or manifest.source_repo != spec.manifest_repo
        or manifest.source_revision != spec.manifest_revision
        or manifest.quant_bits != spec.quant_bits
        or manifest.quant_group_size != spec.quant_group_size
        or manifest.quant_mode != "affine"
        or manifest.artifact_tensor_bytes != spec.total_tensor_bytes
    ):
        raise SystemExit("manifest does not match the requested native artifact")

    source = _TensorReader(source_root)
    artifact = _TensorReader(artifact_root)
    record = manifest.record(args.layer, args.expert)
    payload = read_expert_record(
        manifest,
        artifact_root,
        args.layer,
        args.expert,
        prefer_sidecar=True,
        verify_hash=True,
    )
    arrays = _decode_record(record, payload)

    mx.random.seed(args.seed)
    vectors = mx.random.normal((args.vectors, spec.hidden_size)).astype(mx.bfloat16)
    binding = _expert_binding(
        record,
        payload,
        layer=args.layer,
        expert=args.expert,
    )
    streamed = _run_q4_expert(vectors, binding, group_size=spec.quant_group_size)
    direct = _direct_q4_expert(vectors, arrays, group_size=spec.quant_group_size)
    bf16 = _bf16_expert(
        vectors,
        source,
        layer=args.layer,
        expert=args.expert,
    )
    mx.eval(streamed, direct, bf16)
    kernel_match = bool(
        mx.allclose(
            streamed,
            direct,
            atol=args.kernel_atol,
            rtol=args.kernel_rtol,
        ).item()
    )
    projection_cosines = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.{args.layer}.mlp.experts.{args.expert}.{projection}"
        dequantized = mx.dequantize(
            arrays[projection + ".weight"],
            arrays[projection + ".scales"],
            arrays[projection + ".biases"],
            group_size=spec.quant_group_size,
            bits=4,
            mode="affine",
        )
        projection_cosines[projection] = _cosine(
            dequantized, source.get(base + ".weight")
        )

    config = json.loads((artifact_root / "config.json").read_text(encoding="utf-8"))
    router = _router_probe(
        source,
        artifact,
        layer=args.layer,
        vectors=vectors,
        group_size=spec.quant_group_size,
        top_k=spec.top_k,
        scale=float(config["router_scaling_factor"]),
        route_norm=bool(config.get("route_norm", True)),
        atol=args.kernel_atol,
        rtol=args.kernel_rtol,
        artifact_correction_name=_artifact_correction_name(spec, args.layer),
    )
    q4_cosines_pass = min(projection_cosines.values()) > Q4_WEIGHT_COSINE_MIN
    router_cosine_pass = router["weight_cosine"] > Q8_WEIGHT_COSINE_MIN
    passed = (
        kernel_match and q4_cosines_pass and router_cosine_pass and router["passed"]
    )
    report = {
        "schema": "mtplx-hy3-native-quantization-validation-v1",
        "passed": passed,
        "model_key": spec.key,
        "source_repo": manifest.source_repo,
        "source_revision": manifest.source_revision,
        "fixed_vector": {
            "seed": args.seed,
            "vectors": args.vectors,
            "layer": args.layer,
            "expert": args.expert,
            "q4_kernel_match": kernel_match,
            "kernel_atol": args.kernel_atol,
            "kernel_rtol": args.kernel_rtol,
            "q4_projection_weight_cosines": projection_cosines,
            "q4_weight_cosine_min": Q4_WEIGHT_COSINE_MIN,
            "bf16_output_error": _error_stats(streamed, bf16),
        },
        "router": router,
        "notes": {
            "bf16_q4_output_threshold": (
                "reported only; the repository validation gate does not define "
                "a numeric BF16-vs-Q4 layer-output tolerance"
            ),
            "router_reference": (
                "the specified resident router is affine Q8/gs64; raw BF16 ID "
                "drift on near-tied routes is reported but is not an artifact "
                "parity failure"
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
