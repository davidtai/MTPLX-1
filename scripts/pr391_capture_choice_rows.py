#!/usr/bin/env python3
"""Process-local capture of PR #391 weighted-choice inputs.

This module deliberately imports neither MLX nor MTPLX.  A benchmark process
installs the recorder at an explicit boundary by passing the already-imported
``mtplx.fast_sampling`` and ``mtplx.generation`` modules to
``ChoiceRowCapture.install``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping

import numpy as np


REVIEWED_SUPPORT_SOURCE_SHA256 = (
    "31d5f2d5b4be4dd83dff05b3ae65f70482b5022747d9755c7264f187c9fa8db3"
)
FIXED_COUNTER_FIELDS = (
    "drafted_tokens",
    "accepted_drafts",
    "verify_calls",
    "correction_tokens",
    "bonus_tokens",
)


class CaptureError(RuntimeError):
    """Base class for fail-closed capture errors."""


class SourceHashMismatch(CaptureError):
    """The production support function is not the audited implementation."""


class CaptureAssociationError(CaptureError):
    """A queued support row did not reach the expected sparse sampler call."""


class CaptureRNGError(CaptureError):
    """The sampler did not consume the audited single-draw PCG64 stream."""


class CaptureFinalizationError(CaptureError):
    """The caller's end-of-run evidence does not match the capture."""


@dataclass(frozen=True)
class _PendingRow:
    candidate_ids: np.ndarray
    candidate_values: np.ndarray
    candidate_probs: np.ndarray
    expected_ids: np.ndarray
    expected_probs: np.ndarray
    vocab_size: int


@dataclass(frozen=True)
class _CapturedRow:
    pending: _PendingRow
    uniform: float
    selected_token: int
    rng_pre_sha256: bytes
    rng_post_sha256: bytes


def _sha256_source(function: Any) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise SourceHashMismatch(
            "could not inspect support source for SHA-256 gate"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


def _canonical_state(state: Mapping[str, Any]) -> bytes:
    return json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _state_sha256(state: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(_canonical_state(state)).hexdigest().encode("ascii")


def _expected_sparse_row(
    token_ids: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keep = probs > 0
    kept_ids = np.asarray(token_ids[keep], dtype=np.int64)
    kept_probs = np.asarray(probs[keep], dtype=np.float64)
    total = kept_probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise CaptureAssociationError(
            "captured support row cannot produce a positive SparseDistribution"
        )
    order = np.argsort(kept_ids)
    kept_ids = kept_ids[order]
    kept_probs = kept_probs[order] / total
    # SparseDistribution.__post_init__ performs this second normalization.
    sanitized = np.where(np.isfinite(kept_probs) & (kept_probs > 0), kept_probs, 0.0)
    second_total = sanitized.sum()
    return kept_ids, sanitized / second_total


class ChoiceRowCapture:
    """Own process-local wrappers and serialize their captured rows."""

    def __init__(
        self,
        fast_sampling_module: Any,
        generation_module: Any,
    ) -> None:
        self._fast_sampling = fast_sampling_module
        self._generation = generation_module
        self._original_support = (
            fast_sampling_module._device_serial_support_arrays_relaxed_ties
        )
        self._original_sample = generation_module.sample_from_distribution
        self._expected_source_hash = REVIEWED_SUPPORT_SOURCE_SHA256
        self._actual_source_hash = _sha256_source(self._original_support)
        if self._actual_source_hash != self._expected_source_hash:
            raise SourceHashMismatch(
                "support source SHA-256 mismatch: expected "
                f"{self._expected_source_hash}, got {self._actual_source_hash}"
            )
        self._pending: deque[_PendingRow] = deque()
        self._captured: list[_CapturedRow] = []
        self._support_width: int | None = None
        self._installed = False
        self._finalized = False

    @classmethod
    def install(
        cls,
        fast_sampling_module: Any,
        generation_module: Any,
    ) -> "ChoiceRowCapture":
        recorder = cls(fast_sampling_module, generation_module)
        recorder._install()
        return recorder

    def _install(self) -> None:
        if self._installed:
            raise CaptureError("capture wrappers are already installed")
        self._fast_sampling._device_serial_support_arrays_relaxed_ties = (
            self._support_wrapper
        )
        try:
            self._generation.sample_from_distribution = self._sample_wrapper
        except BaseException:
            self._fast_sampling._device_serial_support_arrays_relaxed_ties = (
                self._original_support
            )
            raise
        self._installed = True

    def __enter__(self) -> "ChoiceRowCapture":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        if not self._installed:
            return
        self._fast_sampling._device_serial_support_arrays_relaxed_ties = (
            self._original_support
        )
        self._generation.sample_from_distribution = self._original_sample
        self._installed = False

    def _support_wrapper(self, logits: Any, config: Any):
        if self._finalized:
            raise CaptureFinalizationError("support called after capture finalization")

        # Audited literal copy of
        # _device_serial_support_arrays_relaxed_ties. The source-hash gate in
        # __init__ makes any production-source change fail before installation.
        # Capture happens at the sole device materialization boundary, before
        # float64 promotion, value ordering, or top-p masking.
        mx = self._fast_sampling.mx
        try:
            rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
            vocab_size = int(rows.shape[-1])
            k = min(int(config.top_k), vocab_size)
            scaled = rows * (1.0 / float(config.temperature))
            cand_idx = mx.argpartition(-scaled, kth=k - 1, axis=-1)[:, :k]
            cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
            top_p_active = 0.0 < float(config.top_p) < 1.0
            if not top_p_active:
                raise CaptureAssociationError(
                    "choice-row capture requires strict 0 < top_p < 1"
                )
            log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
            cand_probs = mx.exp(cand_vals - log_total)
            mx.eval(cand_idx, cand_vals, cand_probs)
            captured_ids = np.asarray(cand_idx, dtype=np.int64).copy()
            captured_values = np.asarray(cand_vals, dtype=np.float32).copy()
            captured_probs = np.asarray(cand_probs, dtype=np.float32).copy()
            token_rows = captured_ids.copy()
            value_rows = captured_values.copy()
            prob_rows = np.asarray(cand_probs, dtype=np.float64)
        except CaptureAssociationError:
            raise
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise CaptureAssociationError(
                "support inputs do not satisfy the audited relaxed-ties shape"
            ) from exc

        if (
            token_rows.ndim != 2
            or prob_rows.shape != token_rows.shape
            or value_rows.shape != token_rows.shape
        ):
            raise CaptureAssociationError(
                "support IDs, values, and probabilities must align exactly"
            )
        width = int(token_rows.shape[1])
        if self._support_width is None:
            self._support_width = width
        elif width != self._support_width:
            raise CaptureAssociationError(
                f"support width changed from {self._support_width} to {width}"
            )

        order = np.lexsort((token_rows, -value_rows), axis=1)
        token_rows = np.take_along_axis(token_rows, order, axis=1)
        value_rows = np.take_along_axis(value_rows, order, axis=1)
        prob_rows = np.take_along_axis(prob_rows, order, axis=1)
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(cumulative_before < float(config.top_p), prob_rows, 0.0)

        for raw_ids, raw_values, raw_probs, token_ids, probs in zip(
            captured_ids,
            captured_values,
            captured_probs,
            token_rows,
            prob_rows,
            strict=True,
        ):
            expected_ids, expected_probs = _expected_sparse_row(token_ids, probs)
            self._pending.append(
                _PendingRow(
                    candidate_ids=raw_ids.copy(),
                    candidate_values=raw_values.copy(),
                    candidate_probs=raw_probs.copy(),
                    expected_ids=expected_ids.copy(),
                    expected_probs=expected_probs.copy(),
                    vocab_size=int(vocab_size),
                )
            )
        return token_rows, prob_rows, vocab_size

    def _sample_wrapper(self, distribution: Any, rng: Any) -> int:
        if not self._pending:
            return int(self._original_sample(distribution, rng))

        pending = self._pending[0]
        try:
            actual_ids = np.asarray(distribution.token_ids)
            actual_probs = np.asarray(distribution.probs)
            actual_vocab_size = int(distribution.vocab_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise CaptureAssociationError(
                "next sampler input is not the pending support row's SparseDistribution"
            ) from exc
        if (
            actual_vocab_size != pending.vocab_size
            or actual_ids.dtype != np.dtype(np.int64)
            or actual_probs.dtype != np.dtype(np.float64)
            or not np.array_equal(actual_ids, pending.expected_ids)
            or not np.array_equal(actual_probs, pending.expected_probs)
        ):
            raise CaptureAssociationError(
                "next sampler input does not exactly match the pending support row"
            )

        if not isinstance(rng, np.random.Generator) or not isinstance(
            rng.bit_generator, np.random.PCG64
        ):
            raise CaptureRNGError("capture requires a NumPy Generator backed by PCG64")
        pre_state = copy.deepcopy(rng.bit_generator.state)
        clone_bit_generator = np.random.PCG64()
        clone_bit_generator.state = copy.deepcopy(pre_state)
        clone = np.random.Generator(clone_bit_generator)
        uniform = float(clone.random())
        expected_post_state = copy.deepcopy(clone.bit_generator.state)

        selected = int(self._original_sample(distribution, rng))
        actual_post_state = copy.deepcopy(rng.bit_generator.state)
        if _canonical_state(actual_post_state) != _canonical_state(expected_post_state):
            raise CaptureRNGError(
                "sampler must consume exactly one PCG64 random draw per weighted choice"
            )

        cdf = np.cumsum(actual_probs, dtype=np.float64)
        cdf /= cdf[-1]
        oracle_index = int(np.searchsorted(cdf, uniform, side="right"))
        oracle_index = min(oracle_index, int(actual_ids.size) - 1)
        oracle_token = int(actual_ids[oracle_index])
        if selected != oracle_token:
            raise CaptureRNGError(
                "selected token does not match the pinned cdf[-1]/side-right oracle"
            )

        self._pending.popleft()
        self._captured.append(
            _CapturedRow(
                pending=pending,
                uniform=uniform,
                selected_token=selected,
                rng_pre_sha256=_state_sha256(pre_state),
                rng_post_sha256=_state_sha256(actual_post_state),
            )
        )
        return selected

    def finalize(
        self,
        output_path: str | Path,
        *,
        metadata: Mapping[str, Any],
        expected_rows: int,
        observed_counters: Mapping[str, int] | None = None,
        expected_counters: Mapping[str, int] | None = None,
    ) -> Path:
        """Validate caller-owned run totals, then write a non-object NPZ."""
        if self._finalized:
            raise CaptureFinalizationError("capture has already been finalized")
        if self._pending:
            raise CaptureFinalizationError(
                f"cannot finalize with {len(self._pending)} pending support rows"
            )
        actual_rows = len(self._captured)
        if actual_rows != int(expected_rows):
            raise CaptureFinalizationError(
                f"expected {int(expected_rows)} rows, captured {actual_rows}"
            )
        normalized_observed = self._normalize_counters(
            observed_counters, label="observed"
        )
        normalized_expected = self._normalize_counters(
            expected_counters, label="expected"
        )
        if normalized_expected != normalized_observed:
            raise CaptureFinalizationError(
                "counter mismatch: expected "
                f"{normalized_expected}, observed {normalized_observed}"
            )
        if normalized_observed["drafted_tokens"] != actual_rows:
            raise CaptureFinalizationError(
                "observed drafted_tokens "
                f"{normalized_observed['drafted_tokens']} does not match captured "
                f"{actual_rows} rows"
            )
        path = Path(output_path)
        if path.suffix != ".npz":
            raise CaptureFinalizationError("capture output path must end in .npz")

        width = int(self._support_width or 0)
        candidate_ids = np.empty((actual_rows, width), dtype=np.int64)
        candidate_values = np.empty((actual_rows, width), dtype=np.float32)
        candidate_probs = np.empty((actual_rows, width), dtype=np.float32)
        uniforms = np.empty(actual_rows, dtype=np.float64)
        selected_tokens = np.empty(actual_rows, dtype=np.int64)
        rng_pre = np.empty(actual_rows, dtype="S64")
        rng_post = np.empty(actual_rows, dtype="S64")
        for index, row in enumerate(self._captured):
            candidate_ids[index] = row.pending.candidate_ids
            candidate_values[index] = row.pending.candidate_values
            candidate_probs[index] = row.pending.candidate_probs
            uniforms[index] = row.uniform
            selected_tokens[index] = row.selected_token
            rng_pre[index] = row.rng_pre_sha256
            rng_post[index] = row.rng_post_sha256

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "row_count": actual_rows,
                "support_width": width,
                "support_source_sha256": self._actual_source_hash,
                "observed_counters": normalized_observed,
                "expected_counters": normalized_expected,
            }
        )
        try:
            metadata_json = json.dumps(
                output_metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except (TypeError, ValueError) as exc:
            raise CaptureFinalizationError(
                "metadata must be JSON serializable"
            ) from exc

        np.savez_compressed(
            path,
            candidate_ids=candidate_ids,
            candidate_values=candidate_values,
            candidate_probs=candidate_probs,
            uniforms=uniforms,
            selected_tokens=selected_tokens,
            rng_pre_sha256=rng_pre,
            rng_post_sha256=rng_post,
            metadata_json=np.asarray(metadata_json, dtype=np.str_),
        )
        self._finalized = True
        return path

    @staticmethod
    def _normalize_counters(
        counters: Mapping[str, int] | None,
        *,
        label: str,
    ) -> dict[str, int]:
        if counters is None or set(counters) != set(FIXED_COUNTER_FIELDS):
            raise CaptureFinalizationError(
                f"{label} counters must use the complete counter schema "
                f"{FIXED_COUNTER_FIELDS}"
            )
        normalized = {field: int(counters[field]) for field in FIXED_COUNTER_FIELDS}
        if any(value < 0 for value in normalized.values()):
            raise CaptureFinalizationError(f"{label} counters must be non-negative")
        return normalized


__all__ = [
    "CaptureAssociationError",
    "CaptureError",
    "CaptureFinalizationError",
    "CaptureRNGError",
    "ChoiceRowCapture",
    "FIXED_COUNTER_FIELDS",
    "REVIEWED_SUPPORT_SOURCE_SHA256",
    "SourceHashMismatch",
]
