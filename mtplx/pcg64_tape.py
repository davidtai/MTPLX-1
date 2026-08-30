"""Request-owned PCG64 uniform tape for exact device weighted choices."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

MAX_OUTPUT_TOKENS = 16_384
DRAWS_PER_CYCLE = 7


@dataclass(init=False)
class PCG64UniformTape:
    rng: np.random.Generator
    values: np.ndarray
    cursor: int = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "PCG64UniformTape must be created with PCG64UniformTape.build(...)"
        )

    @classmethod
    def build(
        cls,
        rng: np.random.Generator,
        *,
        max_output_tokens: Integral,
    ) -> "PCG64UniformTape":
        if type(rng.bit_generator) is not np.random.PCG64:
            raise TypeError("exact device D3 requires numpy.random.PCG64")
        if np.__version__ != "2.4.4":
            raise RuntimeError("exact device D3 is pinned to NumPy 2.4.4")
        if isinstance(max_output_tokens, bool) or not isinstance(
            max_output_tokens, Integral
        ):
            raise TypeError("max_output_tokens must be a non-boolean integer")
        output_tokens = int(max_output_tokens)
        if not 0 <= output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError("exact device D3 supports at most 16,384 outputs")

        clone_bits = np.random.PCG64()
        clone_bits.state = deepcopy(rng.bit_generator.state)
        count = DRAWS_PER_CYCLE * (output_tokens + 1)
        values = np.random.Generator(clone_bits).random(count, dtype=np.float64)
        instance = cls.__new__(cls)
        instance.rng = rng
        instance.values = values
        instance.cursor = 0
        return instance

    def reserve_device_choices(self, count: int) -> int:
        offset = self.cursor
        self.rng.random(count)
        self.cursor += count
        return offset

    def random(self) -> float:
        value = float(self.rng.random())
        self.cursor += 1
        return value

    def choice(self, values: np.ndarray, /, *, p: np.ndarray):
        value = self.rng.choice(values, p=p)
        self.cursor += 1
        return value
