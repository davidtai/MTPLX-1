"""Request-owned PCG64 uniform tape for exact device weighted choices."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np


MAX_OUTPUT_TOKENS = 16_384
DRAWS_PER_CYCLE = 7
REQUIRED_NUMPY_VERSION = "2.4.4"
_PROBABILITY_SUM_TOLERANCE = np.sqrt(np.finfo(np.float64).eps)


def _select_weighted_choice_index(
    probabilities: np.ndarray,
    uniform: float,
) -> int:
    """Select from validated float64 probabilities with NumPy's reductions."""

    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf /= np.sum(probabilities, dtype=np.float64)
    return int(np.searchsorted(cdf, uniform, side="right"))


class UniformTapeExhausted(RuntimeError):
    """A request attempted to consume beyond its construction-time reserve."""


@dataclass(frozen=True, eq=False)
class DeviceUniformReservation:
    """One immutable contiguous range in the request's device input tape."""

    offset: int
    values: np.ndarray

    def __index__(self) -> int:
        return self.offset

    def __int__(self) -> int:
        return self.offset

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Integral):
            return self.offset == int(other)
        return self is other


class PCG64UniformTape:
    """Share one finite PCG64 cursor between device and host sampling."""

    __slots__ = ("_cursor", "_values", "rng")

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
        if type(rng) is not np.random.Generator:
            raise TypeError("exact device D3 requires numpy.random.Generator")
        if type(rng.bit_generator) is not np.random.PCG64:
            raise TypeError("exact device D3 requires numpy.random.PCG64")
        if np.__version__ != REQUIRED_NUMPY_VERSION:
            raise RuntimeError(
                f"exact device D3 is pinned to NumPy {REQUIRED_NUMPY_VERSION}"
            )
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
        values.setflags(write=False)

        instance = cls.__new__(cls)
        instance.rng = rng
        instance._values = values
        instance._cursor = 0
        return instance

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def remaining(self) -> int:
        return int(self._values.size) - self._cursor

    @property
    def values(self) -> np.ndarray:
        view = self._values.view()
        view.setflags(write=False)
        return view

    @property
    def device_values(self) -> np.ndarray:
        return self.values

    def reserve_device_choices(self, count: Integral) -> DeviceUniformReservation:
        requested = self._validated_reservation_count(count)
        offset = self._cursor
        values = self._consume(requested)
        return DeviceUniformReservation(offset=offset, values=values)

    def peek_device_choices(self, count: Integral) -> DeviceUniformReservation:
        """Expose a device window without advancing the authoritative cursor."""

        requested = self._validated_reservation_count(count)
        end = self._cursor + requested
        if end > int(self._values.size):
            raise UniformTapeExhausted(
                "request PCG64 uniform tape exhausted: "
                f"requested {requested}, remaining {self.remaining}"
            )
        values = self._values[self._cursor:end]
        values.setflags(write=False)
        return DeviceUniformReservation(offset=self._cursor, values=values)

    def commit_device_choices(
        self,
        reservation: DeviceUniformReservation,
        count: Integral,
    ) -> None:
        """Advance by the draw count returned from one device decision."""

        if not isinstance(reservation, DeviceUniformReservation):
            raise TypeError("device commit requires a DeviceUniformReservation")
        if reservation.offset != self._cursor:
            raise RuntimeError("device uniform reservation is stale")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("device reported draw count must be an integer")
        committed = int(count)
        if not 0 < committed <= int(reservation.values.size):
            raise ValueError("device reported draw count exceeds its reservation")
        expected = self._values[self._cursor : self._cursor + reservation.values.size]
        if not np.shares_memory(reservation.values, self._values) or not np.array_equal(
            reservation.values, expected
        ):
            raise RuntimeError("device uniform reservation ownership changed")
        self._consume(committed)

    def random(self) -> float:
        return float(self._consume(1)[0])

    def choice(
        self,
        values: np.ndarray,
        /,
        *,
        p: np.ndarray,
    ) -> Any:
        candidates = np.asarray(values)
        input_probabilities = np.asarray(p)
        if np.issubdtype(input_probabilities.dtype, np.floating):
            probability_sum_tolerance = np.sqrt(np.finfo(input_probabilities.dtype).eps)
        else:
            probability_sum_tolerance = _PROBABILITY_SUM_TOLERANCE
        probabilities = np.asarray(input_probabilities, dtype=np.float64)
        if candidates.ndim != 1:
            raise ValueError("weighted choice values must be one-dimensional")
        if candidates.size == 0:
            raise ValueError("weighted choice values must be non-empty")
        if probabilities.ndim != 1:
            raise ValueError("weighted choice p must be one-dimensional")
        if candidates.size != probabilities.size:
            raise ValueError("weighted choice values and p must have equal size")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("weighted choice probabilities must be finite")
        if np.any(probabilities < 0):
            raise ValueError("weighted choice probabilities must be non-negative")
        total = np.sum(probabilities, dtype=np.float64)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("weighted choice probabilities must have positive mass")
        if abs(float(total) - 1.0) > probability_sum_tolerance:
            raise ValueError("weighted choice probabilities do not sum to 1")

        uniform = float(self._consume(1)[0])
        index = _select_weighted_choice_index(probabilities, uniform)
        return candidates[index]

    @staticmethod
    def _validated_reservation_count(count: Integral) -> int:
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("device choice count must be a non-boolean integer")
        requested = int(count)
        if requested <= 0:
            raise ValueError("device choice count must be positive")
        return requested

    def _consume(self, count: int) -> np.ndarray:
        end = self._cursor + count
        if end > int(self._values.size):
            raise UniformTapeExhausted(
                "request PCG64 uniform tape exhausted: "
                f"requested {count}, remaining {self.remaining}"
            )
        start = self._cursor
        self.rng.random(count, dtype=np.float64)
        self._cursor = end
        values = self._values[start:end]
        values.setflags(write=False)
        return values
