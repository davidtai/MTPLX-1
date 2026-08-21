"""Mia ``stock432`` NVFP4 rows for DeepSeek V4 target and DSpark K/V."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


MIA_NVFP4_HEAD_DIM = 512
MIA_NVFP4_NOPE_DIM = 448
MIA_NVFP4_ROPE_DIM = 64
MIA_NVFP4_GROUP_SIZE = 16
MIA_NVFP4_PACKED_BYTES = 256
MIA_NVFP4_SCALE_BYTES = 32
MIA_NVFP4_PADDING_OFFSET = 288
MIA_NVFP4_ROPE_OFFSET = 304
MIA_NVFP4_RECORD_BYTES = 432


_NVFP4_HEADER = r"""
    using namespace metal;

    inline uchar mtplx_e4m3_encode_positive(float value) {
        if (!(value > 0.0f)) {
            return uchar(0);
        }
        value = min(value, 448.0f);
        constexpr float MIN_NORMAL = 0.015625f;
        constexpr float SUB_STEP = 0.001953125f;
        if (value < MIN_NORMAL) {
            uint mantissa = uint(rint(value / SUB_STEP));
            if (mantissa >= 8u) {
                return uchar(0x08);
            }
            return uchar(mantissa);
        }

        int exponent = int(floor(log2(value)));
        float step = exp2(float(exponent - 3));
        uint significand = uint(rint(value / step));
        if (significand >= 16u) {
            exponent += 1;
            significand = 8u;
        }
        uint stored_exponent = uint(exponent + 7);
        if (stored_exponent >= 15u) {
            stored_exponent = 15u;
            significand = min(significand, 14u);
        }
        uint mantissa = significand - 8u;
        return uchar((stored_exponent << 3) | mantissa);
    }

    inline float mtplx_e4m3_decode(uchar raw) {
        uint exponent = (uint(raw) >> 3) & 0x0fu;
        uint mantissa = uint(raw) & 0x07u;
        if (exponent == 0u) {
            return float(mantissa) * 0.001953125f;
        }
        return (1.0f + float(mantissa) * 0.125f)
            * exp2(float(int(exponent) - 7));
    }

    inline uchar mtplx_e2m1_encode(float value) {
        float magnitude = abs(value);
        uchar code;
        if (magnitude <= 0.25f) {
            code = uchar(0);
        } else if (magnitude < 0.75f) {
            code = uchar(1);
        } else if (magnitude <= 1.25f) {
            code = uchar(2);
        } else if (magnitude < 1.75f) {
            code = uchar(3);
        } else if (magnitude <= 2.5f) {
            code = uchar(4);
        } else if (magnitude < 3.5f) {
            code = uchar(5);
        } else if (magnitude <= 5.0f) {
            code = uchar(6);
        } else {
            code = uchar(7);
        }
        uint sign = as_type<uint>(value) >> 31;
        return uchar(code | uchar(sign << 3));
    }
"""


@lru_cache(maxsize=1)
def _stock432_pack_kernel():
    if not mx.metal.is_available():
        raise RuntimeError("Mia stock432 NVFP4 K/V requires Metal")
    source = r"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / 432u;
        uint byte = gid - row * 432u;
        const device T* latent_row = latent + size_t(row) * 512u;
        device uchar* record = records + size_t(row) * 432u;

        if (byte < 256u) {
            uint dim0 = byte * 2u;
            uint group = dim0 / 16u;
            float amax = 0.0f;
            for (uint i = 0; i < 16u; ++i) {
                amax = max(amax, abs(float(latent_row[group * 16u + i])));
            }
            uchar scale_byte = mtplx_e4m3_encode_positive(amax / 6.0f);
            float scale = mtplx_e4m3_decode(scale_byte);
            float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            uchar low = mtplx_e2m1_encode(float(latent_row[dim0]) * inv_scale);
            uchar high = mtplx_e2m1_encode(float(latent_row[dim0 + 1u]) * inv_scale);
            record[byte] = uchar(low | uchar(high << 4));
            return;
        }
        if (byte < 288u) {
            uint group = byte - 256u;
            float amax = 0.0f;
            for (uint i = 0; i < 16u; ++i) {
                amax = max(amax, abs(float(latent_row[group * 16u + i])));
            }
            record[byte] = mtplx_e4m3_encode_positive(amax / 6.0f);
            return;
        }
        if (byte < 304u) {
            record[byte] = uchar(0);
            return;
        }
        const device uchar* rope_bytes = reinterpret_cast<const device uchar*>(
            rope + size_t(row) * 64u
        );
        record[byte] = rope_bytes[byte - 304u];
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_mia_stock432_pack",
        input_names=["latent", "rope"],
        output_names=["records"],
        header=_NVFP4_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _pack_stock432(latent: mx.array, rope: mx.array) -> mx.array:
    if latent.dtype != mx.bfloat16 or rope.dtype != mx.bfloat16:
        raise ValueError("Mia stock432 insertion requires BF16 latent and RoPE rows")
    if latent.ndim < 2 or int(latent.shape[-1]) != MIA_NVFP4_HEAD_DIM:
        raise ValueError("Mia stock432 latent rows must end in width 512")
    if tuple(latent.shape[:-1]) != (*rope.shape[:-1],):
        raise ValueError("Mia stock432 latent and RoPE row prefixes differ")
    if int(rope.shape[-1]) != MIA_NVFP4_ROPE_DIM:
        raise ValueError("Mia stock432 RoPE rows must end in width 64")
    row_count = 1
    for size in latent.shape[:-1]:
        row_count *= int(size)
    records = _stock432_pack_kernel()(
        inputs=[mx.contiguous(latent), mx.contiguous(rope)],
        template=[("T", mx.bfloat16)],
        grid=(row_count * MIA_NVFP4_RECORD_BYTES, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*latent.shape[:-1], MIA_NVFP4_RECORD_BYTES)],
        output_dtypes=[mx.uint8],
    )[0]
    return records


_E2M1_TABLE = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=mx.float32,
)


def _decode_e4m3(raw_bytes: mx.array) -> mx.array:
    raw = raw_bytes.astype(mx.uint32)
    negative = (raw & 0x80) != 0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    subnormal = mantissa.astype(mx.float32) * (2.0**-9)
    normal = (1.0 + mantissa.astype(mx.float32) / 8.0) * mx.power(
        mx.array(2.0, dtype=mx.float32), exponent.astype(mx.float32) - 7.0
    )
    magnitude = mx.where(exponent == 0, subnormal, normal)
    return mx.where(negative, -magnitude, magnitude)


def decode_stock432(records: mx.array) -> tuple[mx.array, mx.array]:
    """Return source-semantics ``(key, value)`` reconstructed from records."""
    if records.dtype != mx.uint8 or records.ndim < 2:
        raise ValueError("Mia stock432 records must be rank-2-or-higher uint8")
    if int(records.shape[-1]) != MIA_NVFP4_RECORD_BYTES:
        raise ValueError("Mia stock432 records must end in 432 bytes")
    packed = records[..., :MIA_NVFP4_PACKED_BYTES]
    low = mx.take(_E2M1_TABLE, packed & 0xF)
    high = mx.take(_E2M1_TABLE, (packed >> 4) & 0xF)
    values = mx.stack([low, high], axis=-1).reshape(
        *records.shape[:-1], MIA_NVFP4_HEAD_DIM
    )
    scales = _decode_e4m3(records[..., 256:288])
    latent = (values * mx.repeat(scales, MIA_NVFP4_GROUP_SIZE, axis=-1)).astype(
        mx.bfloat16
    )
    rope_bytes = mx.contiguous(records[..., MIA_NVFP4_ROPE_OFFSET:])
    rope = rope_bytes.view(mx.bfloat16)
    key = mx.concatenate([latent[..., :MIA_NVFP4_NOPE_DIM], rope], axis=-1)
    return key, latent


class MiaNVFP4Rows:
    """Appendable DeepSeek V4 K/V rows in Mia's native 432-byte layout."""

    head_dim = MIA_NVFP4_HEAD_DIM
    nope_dim = MIA_NVFP4_NOPE_DIM
    rope_dim = MIA_NVFP4_ROPE_DIM
    group_size = MIA_NVFP4_GROUP_SIZE
    record_bytes = MIA_NVFP4_RECORD_BYTES
    mode = "nvfp4_stock432"

    def __init__(self) -> None:
        self.records: mx.array | None = None
        self._prefix_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return 0 if self.records is None else int(self.records.shape[-2])

    @property
    def shape(self) -> tuple[int, ...]:
        if self.records is None:
            return (0, self.record_bytes)
        return tuple(int(value) for value in self.records.shape)

    @property
    def state(self) -> mx.array | None:
        return self.records

    @property
    def nbytes(self) -> int:
        return 0 if self.records is None else int(self.records.nbytes)

    def _validate_rows(self, latent: mx.array, rope: mx.array) -> tuple[int, ...]:
        if latent.ndim < 2 or int(latent.shape[-1]) != self.head_dim:
            raise ValueError("Mia stock432 latent rows must end in width 512")
        if rope.ndim != latent.ndim or int(rope.shape[-1]) != self.rope_dim:
            raise ValueError("Mia stock432 RoPE rows must end in width 64")
        if tuple(latent.shape[:-1]) != tuple(rope.shape[:-1]):
            raise ValueError("Mia stock432 latent and RoPE row shapes differ")
        prefix = tuple(int(value) for value in latent.shape[:-2])
        if self._prefix_shape is not None and prefix != self._prefix_shape:
            raise ValueError(
                f"Mia stock432 prefix changed from {self._prefix_shape} to {prefix}"
            )
        return prefix

    def append(self, latent: mx.array, rope: mx.array) -> None:
        prefix = self._validate_rows(latent, rope)
        new_records = _pack_stock432(latent, rope)
        if self.records is None:
            self.records = new_records
            self._prefix_shape = prefix
        else:
            self.records = mx.concatenate([self.records, new_records], axis=-2)

    def decode(self, start: int = 0, stop: int | None = None) -> tuple[mx.array, mx.array]:
        if self.records is None:
            raise ValueError("cannot decode an empty Mia stock432 K/V store")
        begin = int(start)
        end = len(self) if stop is None else int(stop)
        if begin < 0 or end < begin or end > len(self):
            raise ValueError("Mia stock432 decode range is outside the store")
        return decode_stock432(self.records[..., begin:end, :])

    def replace(self, start: int, latent: mx.array, rope: mx.array) -> None:
        if self.records is None:
            raise ValueError("cannot replace rows in an empty Mia stock432 store")
        self._validate_rows(latent, rope)
        start = int(start)
        count = int(latent.shape[-2])
        if start < 0 or count <= 0 or start + count > len(self):
            raise ValueError("replacement Mia stock432 range is outside the store")
        replacement = _pack_stock432(latent, rope)
        self.records = mx.concatenate(
            [
                self.records[..., :start, :],
                replacement,
                self.records[..., start + count :, :],
            ],
            axis=-2,
        )

    def drop_first(self, count: int) -> None:
        count = max(0, int(count))
        if count == 0:
            return
        if count >= len(self):
            self.clear()
            return
        self.records = self.records[..., count:, :]

    def truncate(self, length: int) -> None:
        length = max(0, int(length))
        if length >= len(self):
            return
        if length == 0:
            self.clear()
            return
        self.records = self.records[..., :length, :]

    def clear(self) -> None:
        self.records = None
        self._prefix_shape = None

    def replace_state(self, state: mx.array | None) -> None:
        if state is None:
            self.clear()
            return
        if (
            state.dtype != mx.uint8
            or state.ndim < 2
            or int(state.shape[-1]) != self.record_bytes
        ):
            raise ValueError("invalid Mia stock432 K/V state")
        self.records = state
        self._prefix_shape = tuple(int(value) for value in state.shape[:-2])
