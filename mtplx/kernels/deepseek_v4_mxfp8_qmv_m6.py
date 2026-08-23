"""Exact physical-M6 MXFP8 QMV for Mia's target ``wq_b`` projection.

MLX 0.32.1's ``fp_qmv_wide`` streams at most five vectors per threadgroup.
Physical M6 therefore selects ``nv=3`` and launches two weight-reading tiles.
This kernel preserves that implementation's group-32 E4M3/E8M0 decode, FP32
dot/accumulator association, 16-lane K reduction, and BF16 store, but compiles
the fixed verifier width as ``nv=6`` so one weight read feeds all six rows.

The pinned SparkInfer path binds ``expected_m`` when it constructs its MXFP8
linear owner.  Its activation quantization and tensor-core arithmetic are not
equivalent to this target and are deliberately not copied.  This candidate
instead applies the same construction-fixed-M principle to MLX's own QMV
association and native storage.

Exact-weight leaf gates rejected the other three target geometries.  Keeping
only ``K=1024, N=32768`` makes the measured boundary part of the installed
type rather than a hot eligibility check.

Source anchors inspected for this bounded candidate:

* MLX ``fp_quantized.h::fp_qmv_wide_impl`` at
  ``9d050847989bff93c6e8ff39f2b7be6f0f45109b``.
* B12X ``mxfp8_linear`` construction at
  ``272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f``.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import mlx.core as mx

from .deepseek_v4_wo_mxfp8 import (
    mia_e4m3_decode_byte as _mia_e4m3_decode_byte,
)


MIA_M6_MXFP8_SHAPES = frozenset({(1024, 32768)})


def mia_e4m3_decode_byte(raw: int) -> float:
    """Expose the already-qualified Mia/MLX E4M3 byte oracle."""

    return _mia_e4m3_decode_byte(raw)


def mia_ue8m0_decode_byte(raw: int) -> float:
    """Decode one MLX E8M0 scale byte with its two endpoint encodings."""

    raw = int(raw) & 0xFF
    if raw == 255:
        return math.inf
    return 2.0 ** (-127 if raw == 0 else raw - 127)


_MXFP8_QMV_M6_HEADER = r"""
using namespace metal;

inline float mia_m6_e4m3_decode(uchar raw) {
    ushort bits = ushort(uint(raw) & 0x7fu) << 7;
    half converted = as_type<half>(bits);
    converted *= 256.0;
    float value = float(converted);
    return (uint(raw) & 0x80u) != 0u ? -value : value;
}

inline float mia_m6_e8m0_decode(uchar raw) {
    uint bits = raw == uchar(0)
        ? 0x00400000u
        : uint(raw) << 23;
    return as_type<float>(bits);
}
"""


_MXFP8_QMV_M6_SOURCE = r"""
constexpr int M = 6;
constexpr int K = __K__;
constexpr int N = __N__;
constexpr int GROUP_SIZE = 32;
constexpr int K_LANES = 16;
constexpr int RESULTS_PER_SIMDGROUP = 32 / K_LANES;
constexpr int NUM_SIMDGROUPS = 2;
constexpr int ROWS_PER_THREADGROUP =
    RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
int k_lane = int(lane) % K_LANES;
int simdgroup_row = int(lane) / K_LANES;
int out_row = int(threadgroup_position_in_grid.x) * ROWS_PER_THREADGROUP
    + int(simdgroup) * RESULTS_PER_SIMDGROUP + simdgroup_row;
int row = min(out_row, N - 1);

const device uchar* weight_bytes =
    reinterpret_cast<const device uchar*>(w);
const device uchar* weight_row = weight_bytes + size_t(row) * K;
const device uchar* scale_row = scales + size_t(row) * (K / GROUP_SIZE);

float result[M] = {0.0f};
for (int group = k_lane; group < K / GROUP_SIZE; group += K_LANES) {
    int k0 = group * GROUP_SIZE;
    float scale = mia_m6_e8m0_decode(scale_row[group]);
    const device uchar* weight_group = weight_row + k0;

    float4 weight4[GROUP_SIZE / 4];
    _Pragma("unroll")
    for (int j = 0; j < GROUP_SIZE / 4; ++j) {
        weight4[j] = float4(
            mia_m6_e4m3_decode(weight_group[4 * j]),
            mia_m6_e4m3_decode(weight_group[4 * j + 1]),
            mia_m6_e4m3_decode(weight_group[4 * j + 2]),
            mia_m6_e4m3_decode(weight_group[4 * j + 3]));
    }

    _Pragma("unroll")
    for (int v = 0; v < M; ++v) {
        const device vec<T, 4>* x4 =
            reinterpret_cast<const device vec<T, 4>*>(x + v * K + k0);
        float acc = 0.0f;
        _Pragma("unroll")
        for (int j = 0; j < GROUP_SIZE / 4; ++j) {
            acc += dot(weight4[j], float4(x4[j]));
        }
        result[v] += scale * acc;
    }
}

_Pragma("unroll")
for (int v = 0; v < M; ++v) {
    result[v] += simd_shuffle_down(result[v], 8);
    result[v] += simd_shuffle_down(result[v], 4);
    result[v] += simd_shuffle_down(result[v], 2);
    result[v] += simd_shuffle_down(result[v], 1);
}

if (k_lane == 0 && out_row < N) {
    _Pragma("unroll")
    for (int v = 0; v < M; ++v) {
        y[v * N + out_row] = T(result[v]);
    }
}
"""


MIA_WQ_B_M6_DESCRIPTOR_SHA256 = hashlib.sha256(
    b"mtplx-mia-wq-b-m6-mxfp8-v1\0"
    + _MXFP8_QMV_M6_HEADER.encode("utf-8")
    + b"\0"
    + _MXFP8_QMV_M6_SOURCE.encode("utf-8")
    + b"\0k=1024,n=32768,m=6,grid=n/4,tg=64,kl=16"
).hexdigest()


@lru_cache(maxsize=1)
def _mxfp8_qmv_m6_kernel(k: int, n: int):
    geometry = (int(k), int(n))
    if geometry not in MIA_M6_MXFP8_SHAPES:
        raise ValueError(f"unsupported Mia physical-M6 MXFP8 geometry {geometry}")
    source = _MXFP8_QMV_M6_SOURCE.replace("__K__", str(k)).replace("__N__", str(n))
    return mx.fast.metal_kernel(
        name=f"mtplx_dsv4_mxfp8_qmv_m6_k{k}_n{n}",
        input_names=["w", "scales", "x"],
        output_names=["y"],
        header=_MXFP8_QMV_M6_HEADER,
        source=source,
    )


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _validate_projection(projection, *, k: int, n: int) -> None:
    geometry = (int(k), int(n))
    expected = (
        32,
        8,
        "mxfp8",
        (n, k // 4),
        mx.uint32,
        (n, k // 32),
        mx.uint8,
        None,
        None,
    )
    observed = (
        int(getattr(projection, "group_size", -1)),
        int(getattr(projection, "bits", -1)),
        str(getattr(projection, "mode", "")).lower(),
        _shape(getattr(projection, "weight", None)),
        getattr(getattr(projection, "weight", None), "dtype", None),
        _shape(getattr(projection, "scales", None)),
        getattr(getattr(projection, "scales", None), "dtype", None),
        getattr(projection, "bias", None),
        getattr(projection, "biases", None),
    )
    if geometry not in MIA_M6_MXFP8_SHAPES or observed != expected:
        raise ValueError(
            "Mia physical-M6 MXFP8 projection contract changed: "
            f"geometry={geometry!r}, observed={observed!r}"
        )


class MiaM6MXFP8QMVPlan:
    """Construction-bound direct lane for one fixed native MXFP8 projection."""

    __slots__ = (
        "descriptor_sha256",
        "k",
        "kernel",
        "n",
        "scales",
        "weight",
    )

    def __init__(self, projection, *, k: int, n: int) -> None:
        _validate_projection(projection, k=k, n=n)
        self.k = int(k)
        self.n = int(n)
        self.weight = projection.weight
        self.scales = projection.scales
        self.kernel = _mxfp8_qmv_m6_kernel(self.k, self.n)
        self.descriptor_sha256 = MIA_WQ_B_M6_DESCRIPTOR_SHA256

    def __call__(self, x: mx.array) -> mx.array:
        output_shape = (*x.shape[:-1], self.n)
        (output,) = self.kernel(
            inputs=[self.weight, self.scales, x],
            template=[("T", mx.bfloat16)],
            output_shapes=[output_shape],
            output_dtypes=[mx.bfloat16],
            grid=(64 * (self.n // 4), 1, 1),
            threadgroup=(64, 1, 1),
        )
        return output


class MiaWQBM6Route:
    """Construction-bound ``wq_b`` route with one genuine physical-M split."""

    __slots__ = ("m6", "stock")

    def __init__(self, projection) -> None:
        self.stock = projection
        self.m6 = MiaM6MXFP8QMVPlan(projection, k=1024, n=32768)

    def __call__(self, values: mx.array) -> mx.array:
        if int(values.size) == 6 * 1024:
            return self.m6(values)
        return self.stock(values)


def install_mia_m6_mxfp8_qmv(
    projection,
    *,
    k: int,
    n: int,
) -> MiaM6MXFP8QMVPlan:
    """Validate and bind one direct physical-M6 projection before generation."""

    return MiaM6MXFP8QMVPlan(projection, k=k, n=n)


def install_mia_wq_b_m6_route(projection) -> MiaWQBM6Route:
    """Bind the only exact-weight-positive M6 geometry before generation."""

    return MiaWQBM6Route(projection)
