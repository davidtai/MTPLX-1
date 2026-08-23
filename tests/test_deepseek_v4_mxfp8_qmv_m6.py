from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.kernels import deepseek_v4_mxfp8_qmv_m6 as qmv_m6  # noqa: E402


class _StaticArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.flags = SimpleNamespace(row_contiguous=True)


class _MXFP8Projection:
    def __init__(self, *, k: int, n: int):
        self.weight = _StaticArray((n, k // 4), mx.uint32)
        self.scales = _StaticArray((n, k // 32), mx.uint8)
        self.bias = None
        self.biases = None
        self.group_size = 32
        self.bits = 8
        self.mode = "mxfp8"
        self.calls = []

    def __call__(self, values):
        self.calls.append(values)
        return ("stock", values)


def test_mia_m6_plan_binds_exact_shape_kernel_once(
    monkeypatch,
) -> None:
    k = 1024
    n = 32768
    input_shape = (1, 6, 1024)
    factories = []
    calls = []

    def factory(actual_k: int, actual_n: int):
        factories.append((actual_k, actual_n))

        def kernel(**kwargs):
            calls.append(kwargs)
            return tuple(
                _StaticArray(shape, dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"],
                    kwargs["output_dtypes"],
                    strict=True,
                )
            )

        return kernel

    monkeypatch.setattr(qmv_m6, "_mxfp8_qmv_m6_kernel", factory)
    projection = _MXFP8Projection(k=k, n=n)
    plan = qmv_m6.install_mia_m6_mxfp8_qmv(projection, k=k, n=n)

    assert factories == [(k, n)]
    assert plan.weight is projection.weight
    assert plan.scales is projection.scales

    monkeypatch.setattr(
        qmv_m6,
        "_mxfp8_qmv_m6_kernel",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("installed M6 execution re-entered the kernel factory")
        ),
    )
    inputs = _StaticArray(input_shape, mx.bfloat16)
    output = plan(inputs)

    assert output.shape == (*input_shape[:-1], n)
    assert output.dtype == mx.bfloat16
    assert len(calls) == 1
    assert calls[0]["inputs"] == [projection.weight, projection.scales, inputs]
    assert calls[0]["template"] == [("T", mx.bfloat16)]
    assert calls[0]["output_shapes"] == [(*input_shape[:-1], n)]
    assert calls[0]["output_dtypes"] == [mx.bfloat16]
    assert calls[0]["grid"] == (64 * (n // 4), 1, 1)
    assert calls[0]["threadgroup"] == (64, 1, 1)


def test_mia_wq_b_route_selects_candidate_only_for_physical_m6(
    monkeypatch,
) -> None:
    candidate_calls = []

    def factory(_actual_k: int, _actual_n: int):
        def kernel(**kwargs):
            candidate_calls.append(kwargs["inputs"][2])
            return ((_StaticArray(kwargs["output_shapes"][0], mx.bfloat16)),)

        return kernel

    monkeypatch.setattr(qmv_m6, "_mxfp8_qmv_m6_kernel", factory)
    projection = _MXFP8Projection(k=1024, n=32768)
    route = qmv_m6.install_mia_wq_b_m6_route(projection)
    decode = _StaticArray((1, 6, 1024), mx.bfloat16)
    prefill = _StaticArray((1, 1024, 1024), mx.bfloat16)
    multi_batch = _StaticArray((2, 6, 1024), mx.bfloat16)
    decode.size = 6 * 1024
    prefill.size = 1024 * 1024
    multi_batch.size = 2 * 6 * 1024

    candidate_output = route(decode)
    stock_output = route(prefill)
    multi_batch_output = route(multi_batch)

    assert candidate_output.shape == (1, 6, 32768)
    assert candidate_calls == [decode]
    assert stock_output == ("stock", prefill)
    assert multi_batch_output == ("stock", multi_batch)
    assert projection.calls == [prefill, multi_batch]


@pytest.mark.parametrize(
    ("k", "n"),
    (
        (4096, 1536),
        (4096, 2048),
        (2048, 4096),
    ),
)
def test_mia_m6_plan_rejects_leaf_regression_geometries(k: int, n: int) -> None:
    projection = _MXFP8Projection(k=k, n=n)

    with pytest.raises(ValueError, match="physical-M6 MXFP8"):
        qmv_m6.install_mia_m6_mxfp8_qmv(projection, k=k, n=n)


def test_mia_m6_plan_accepts_loader_style_mlx_arrays(monkeypatch) -> None:
    projection = _MXFP8Projection(k=1024, n=32768)
    projection.weight = mx.zeros((32768, 256), dtype=mx.uint32)
    projection.scales = mx.zeros((32768, 32), dtype=mx.uint8)
    monkeypatch.setattr(qmv_m6, "_mxfp8_qmv_m6_kernel", lambda *_args: object())

    plan = qmv_m6.install_mia_m6_mxfp8_qmv(
        projection,
        k=1024,
        n=32768,
    )

    assert plan.weight is projection.weight
    assert plan.scales is projection.scales


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("group_size", 64),
        ("bits", 4),
        ("mode", "affine"),
        ("weight", _StaticArray((32768, 255), mx.uint32)),
        ("scales", _StaticArray((32768, 31), mx.uint8)),
        ("biases", _StaticArray((32768,), mx.bfloat16)),
    ),
)
def test_mia_m6_plan_rejects_non_mxfp8_storage_at_construction(
    field: str,
    value,
) -> None:
    projection = _MXFP8Projection(k=1024, n=32768)
    setattr(projection, field, value)

    with pytest.raises(ValueError, match="Mia physical-M6 MXFP8"):
        qmv_m6.install_mia_m6_mxfp8_qmv(
            projection,
            k=1024,
            n=32768,
        )


def _stock_or_candidate_cpu_oracle(
    x: np.ndarray,
    weight: np.ndarray,
    scales: np.ndarray,
    *,
    vectors_per_weight_read: int,
) -> tuple[np.ndarray, int]:
    """Model MLX's per-vector FP32 association, not a dense-matmul oracle."""

    m, k = x.shape
    partial = np.zeros((m, 16), dtype=np.float32)
    weight_reads = 0
    for vector0 in range(0, m, vectors_per_weight_read):
        vector1 = min(vector0 + vectors_per_weight_read, m)
        for lane in range(16):
            for group in range(lane, k // 32, 16):
                k0 = group * 32
                scale = np.float32(qmv_m6.mia_ue8m0_decode_byte(scales[group]))
                decoded = np.asarray(
                    [qmv_m6.mia_e4m3_decode_byte(raw) for raw in weight[k0 : k0 + 32]],
                    dtype=np.float32,
                )
                weight_reads += 1
                for vector in range(vector0, vector1):
                    acc = np.float32(0.0)
                    for j in range(0, 32, 4):
                        dot4 = np.float32(0.0)
                        for i in range(4):
                            dot4 = np.float32(
                                dot4
                                + np.float32(decoded[j + i] * x[vector, k0 + j + i])
                            )
                        acc = np.float32(acc + dot4)
                    partial[vector, lane] = np.float32(
                        partial[vector, lane] + np.float32(scale * acc)
                    )

    output = np.empty((m,), dtype=np.float32)
    for vector in range(m):
        reduced = partial[vector].copy()
        for delta in (8, 4, 2, 1):
            previous = reduced.copy()
            for lane in range(16 - delta):
                reduced[lane] = np.float32(previous[lane] + previous[lane + delta])
        output[vector] = reduced[0]
    return output, weight_reads


def test_six_row_stream_preserves_mlx_nv3_arithmetic_and_halves_weight_reads() -> None:
    rng = np.random.default_rng(17)
    k = 4096
    x = rng.standard_normal((6, k), dtype=np.float32)
    weight = rng.integers(0, 255, size=(k,), dtype=np.uint8)
    scales = rng.integers(118, 136, size=(k // 32,), dtype=np.uint8)

    stock, stock_reads = _stock_or_candidate_cpu_oracle(
        x,
        weight,
        scales,
        vectors_per_weight_read=3,
    )
    candidate, candidate_reads = _stock_or_candidate_cpu_oracle(
        x,
        weight,
        scales,
        vectors_per_weight_read=6,
    )

    np.testing.assert_array_equal(candidate.view(np.uint32), stock.view(np.uint32))
    assert candidate_reads * 2 == stock_reads


def test_m6_metal_source_is_the_mlx_fp_qmv_wide_association_with_nv6() -> None:
    source = qmv_m6._MXFP8_QMV_M6_SOURCE

    assert "constexpr int M = 6;" in source
    assert "constexpr int K_LANES = 16;" in source
    assert "constant constexpr" not in source
    assert "float4 weight4[GROUP_SIZE / 4];" in source
    assert "acc += dot(weight4[j], float4(x4[j]));" in source
    assert "result[v] += scale * acc;" in source
    assert tuple(
        source.index(f"simd_shuffle_down(result[v], {delta})") for delta in (8, 4, 2, 1)
    ) == tuple(
        sorted(
            source.index(f"simd_shuffle_down(result[v], {delta})")
            for delta in (8, 4, 2, 1)
        )
    )
    assert "simd_sum" not in source
