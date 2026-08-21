"""Affine-int4 row storage for DeepSeek V4 target and DSpark K/V caches."""

from __future__ import annotations

import mlx.core as mx


AFFINE_KV_BITS = 4
AFFINE_KV_GROUP_SIZE = 64
AFFINE_KV_MODE = "affine"


class AffineInt4Rows:
    """Appendable sequence rows stored only as MLX affine-int4 triples."""

    bits = AFFINE_KV_BITS
    group_size = AFFINE_KV_GROUP_SIZE
    mode = AFFINE_KV_MODE

    def __init__(self, *, width: int) -> None:
        width = int(width)
        if width <= 0 or width % self.group_size:
            raise ValueError(
                f"affine-int4 K/V width must be a positive multiple of "
                f"{self.group_size}; got {width}"
            )
        self.width = width
        self.packed: mx.array | None = None
        self.scales: mx.array | None = None
        self.biases: mx.array | None = None
        self._prefix_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        return 0 if self.packed is None else int(self.packed.shape[-2])

    @property
    def shape(self) -> tuple[int, ...]:
        if self._prefix_shape is None:
            return (0, self.width)
        return (*self._prefix_shape, len(self), self.width)

    @property
    def state(self) -> tuple[mx.array, mx.array, mx.array] | None:
        if self.packed is None:
            return None
        return self.packed, self.scales, self.biases

    @property
    def nbytes(self) -> int:
        if self.packed is None:
            return 0
        return int(self.packed.nbytes + self.scales.nbytes + self.biases.nbytes)

    def append(self, rows: mx.array) -> None:
        if rows.ndim < 2 or int(rows.shape[-1]) != self.width:
            raise ValueError(
                f"affine-int4 K/V rows must end in width {self.width}; "
                f"got shape {tuple(rows.shape)}"
            )
        prefix = tuple(int(v) for v in rows.shape[:-2])
        if self._prefix_shape is not None and prefix != self._prefix_shape:
            raise ValueError(
                f"affine-int4 K/V prefix changed from {self._prefix_shape} to {prefix}"
            )
        packed, scales, biases = mx.quantize(
            rows,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if self.packed is None:
            self.packed = packed
            self.scales = scales
            self.biases = biases
            self._prefix_shape = prefix
            return
        self.packed = mx.concatenate([self.packed, packed], axis=-2)
        self.scales = mx.concatenate([self.scales, scales], axis=-2)
        self.biases = mx.concatenate([self.biases, biases], axis=-2)

    def dequantize(self, start: int = 0, stop: int | None = None) -> mx.array:
        if self.packed is None:
            raise ValueError("cannot dequantize an empty affine-int4 K/V row store")
        end = len(self) if stop is None else int(stop)
        begin = int(start)
        return mx.dequantize(
            self.packed[..., begin:end, :],
            self.scales[..., begin:end, :],
            self.biases[..., begin:end, :],
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def replace(self, start: int, rows: mx.array) -> None:
        if self.packed is None:
            raise ValueError("cannot replace rows in an empty affine-int4 K/V store")
        start = int(start)
        count = int(rows.shape[-2])
        if (
            rows.ndim < 2
            or int(rows.shape[-1]) != self.width
            or tuple(int(v) for v in rows.shape[:-2]) != self._prefix_shape
        ):
            raise ValueError("replacement affine-int4 K/V rows have incompatible shape")
        if start < 0 or count <= 0 or start + count > len(self):
            raise ValueError("replacement affine-int4 K/V range is outside the store")
        packed, scales, biases = mx.quantize(
            rows,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

        def splice(current: mx.array, replacement: mx.array) -> mx.array:
            return mx.concatenate(
                [
                    current[..., :start, :],
                    replacement,
                    current[..., start + count :, :],
                ],
                axis=-2,
            )

        self.packed = splice(self.packed, packed)
        self.scales = splice(self.scales, scales)
        self.biases = splice(self.biases, biases)

    def drop_first(self, count: int) -> None:
        count = max(0, int(count))
        if count == 0:
            return
        if count >= len(self):
            self.clear()
            return
        self.packed = self.packed[..., count:, :]
        self.scales = self.scales[..., count:, :]
        self.biases = self.biases[..., count:, :]

    def truncate(self, length: int) -> None:
        length = max(0, int(length))
        if length >= len(self):
            return
        if length == 0:
            self.clear()
            return
        self.packed = self.packed[..., :length, :]
        self.scales = self.scales[..., :length, :]
        self.biases = self.biases[..., :length, :]

    def clear(self) -> None:
        self.packed = None
        self.scales = None
        self.biases = None
        self._prefix_shape = None

    def replace_state(
        self,
        state: tuple[mx.array, mx.array, mx.array] | None,
    ) -> None:
        if state is None:
            self.clear()
            return
        if not isinstance(state, (tuple, list)) or len(state) != 3:
            raise ValueError("affine-int4 K/V state must be packed/scales/biases")
        packed, scales, biases = state
        if packed.dtype != mx.uint32:
            raise ValueError(f"affine-int4 packed rows must be uint32; got {packed.dtype}")
        if packed.ndim < 2 or scales.ndim != packed.ndim or biases.shape != scales.shape:
            raise ValueError("invalid affine-int4 K/V state shapes")
        if int(packed.shape[-1]) * (32 // self.bits) != self.width:
            raise ValueError("affine-int4 packed width does not match the cache width")
        if int(scales.shape[-1]) * self.group_size != self.width:
            raise ValueError("affine-int4 scale width does not match the cache width")
        if tuple(packed.shape[:-1]) != tuple(scales.shape[:-1]):
            raise ValueError("affine-int4 packed and scale row shapes differ")
        self.packed = packed
        self.scales = scales
        self.biases = biases
        self._prefix_shape = tuple(int(v) for v in packed.shape[:-2])
