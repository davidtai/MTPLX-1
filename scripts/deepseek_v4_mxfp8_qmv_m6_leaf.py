#!/usr/bin/env python3
"""Guarded exact-weight leaf gate for the Mia physical-M6 ``wq_b`` lane.

The first gate covered all four MXFP8 geometries in a target pass and rejected
three.  This retained gate covers only the positive ``K=1024, N=32768`` shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx

from mtplx.kernels.deepseek_v4_mxfp8_qmv_m6 import (
    install_mia_m6_mxfp8_qmv,
)


DEFAULT_MODEL = Path("/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1")


def _expand_scales(scales: mx.array, n: int, k: int) -> mx.array:
    expected = ((n + 127) // 128, (k + 127) // 128)
    if tuple(scales.shape) != expected or scales.dtype != mx.uint8:
        raise ValueError(
            f"raw scale contract changed: {tuple(scales.shape)}/{scales.dtype} "
            f"!= {expected}/uint8"
        )
    return mx.contiguous(
        mx.repeat(mx.repeat(scales, 128, axis=0), 4, axis=1)[:n, : k // 32]
    )


def _projection(
    raw: dict[str, mx.array],
    prefixes: tuple[str, ...],
) -> SimpleNamespace:
    weights = []
    scales = []
    k = None
    for prefix in prefixes:
        weight = raw[prefix + ".weight"]
        scale = raw[prefix + ".scale"]
        if weight.dtype != mx.uint8 or weight.ndim != 2:
            raise ValueError(
                f"raw weight contract changed for {prefix}: "
                f"{tuple(weight.shape)}/{weight.dtype}"
            )
        n_part, k_part = (int(dimension) for dimension in weight.shape)
        if k is None:
            k = k_part
        elif k != k_part:
            raise ValueError(f"stacked K changed for {prefix}: {k_part} != {k}")
        weights.append(mx.contiguous(weight).view(mx.uint32))
        scales.append(_expand_scales(scale, n_part, k_part))
    assert k is not None
    weight = mx.contiguous(mx.concatenate(weights, axis=0))
    expanded_scales = mx.contiguous(mx.concatenate(scales, axis=0))
    mx.eval(weight, expanded_scales)
    return SimpleNamespace(
        weight=weight,
        scales=expanded_scales,
        bias=None,
        biases=None,
        group_size=32,
        bits=8,
        mode="mxfp8",
        k=k,
        n=int(expanded_scales.shape[0]),
    )


def _stock(projection: SimpleNamespace, values: mx.array) -> mx.array:
    return mx.quantized_matmul(
        values,
        projection.weight,
        scales=projection.scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


def _seconds(callable_, values: mx.array) -> float:
    started = time.perf_counter()
    output = callable_(values)
    mx.eval(output)
    return time.perf_counter() - started


def _gate(
    label: str,
    projection: SimpleNamespace,
    *,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    values = (
        mx.sin(mx.arange(6 * projection.k, dtype=mx.float32) * 0.001)
        .reshape(6, projection.k)
        .astype(mx.bfloat16)
    )
    candidate = install_mia_m6_mxfp8_qmv(
        projection,
        k=projection.k,
        n=projection.n,
    )
    for _ in range(warmup):
        mx.eval(_stock(projection, values), candidate(values))

    stock_output = _stock(projection, values)
    candidate_output = candidate(values)
    mx.eval(stock_output, candidate_output)
    exact = bool(mx.array_equal(stock_output, candidate_output).item())

    stock_seconds = []
    candidate_seconds = []
    for sample in range(samples):
        if sample % 2:
            candidate_seconds.append(_seconds(candidate, values))
            stock_seconds.append(_seconds(lambda x: _stock(projection, x), values))
        else:
            stock_seconds.append(_seconds(lambda x: _stock(projection, x), values))
            candidate_seconds.append(_seconds(candidate, values))

    stock_median = statistics.median(stock_seconds)
    candidate_median = statistics.median(candidate_seconds)
    return {
        "label": label,
        "k": projection.k,
        "n": projection.n,
        "physical_m": 6,
        "exact": exact,
        "stock_seconds": stock_seconds,
        "candidate_seconds": candidate_seconds,
        "stock_median_ms": stock_median * 1000.0,
        "candidate_median_ms": candidate_median * 1000.0,
        "candidate_delta_percent": (candidate_median / stock_median - 1.0) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 3:
        raise SystemExit("--warmup must be >=1 and --samples must be >=3")

    shard = args.model / "carried-001.safetensors"
    raw = mx.load(str(shard))
    layer = f"layers.{args.layer}"
    specs = (("wq_b", (f"{layer}.attn.wq_b",)),)
    results = []
    for label, prefixes in specs:
        projection = _projection(raw, prefixes)
        result = _gate(
            label,
            projection,
            warmup=args.warmup,
            samples=args.samples,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    receipt = {
        "model": str(args.model),
        "shard": str(shard),
        "layer": args.layer,
        "warmup": args.warmup,
        "samples": args.samples,
        "results": results,
        "all_exact": all(bool(result["exact"]) for result in results),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["all_exact"]:
        raise SystemExit("physical-M6 candidate failed exact parity")


if __name__ == "__main__":
    main()
