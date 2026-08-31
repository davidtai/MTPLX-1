#!/usr/bin/env python3
"""Process-local NumPy weighted-choice routes for PR #391 E2E benchmarks."""

from __future__ import annotations

from enum import Enum
import hashlib
import importlib
import inspect
import json
from types import TracebackType
from typing import Any, Mapping

import numpy as np


REQUIRED_NUMPY_VERSION = "2.4.4"
REVIEWED_SUPPORT_SOURCE_SHA256 = (
    "1caabc0070767ac2252e5f6511ed54b038e2d880527c28d8c4333503a4258822"
)
REVIEWED_SAMPLE_SOURCE_SHA256 = (
    "4dc740a4b320b54c5b38b3bc8ab13935d8445407d5690e9da9eac867cc9ecba7"
)
STATS_FIELDS = (
    "drafted_tokens",
    "accepted_drafts",
    "verify_calls",
    "correction_tokens",
    "bonus_tokens",
)


class RouteArm(str, Enum):
    CONTROL = "control"
    REDUCED_EXACT_FLOAT64 = "reduced-exact-float64"
    REDUCED_FLOAT32_TEST_ONLY = "reduced-float32-test-only"


class ChoiceRouteError(RuntimeError):
    """Base class for fail-closed benchmark-route errors."""


class ChoiceRouteSourceMismatch(ChoiceRouteError):
    """A production callable is not the reviewed implementation."""


class ChoiceRouteAssociationError(ChoiceRouteError):
    """The immediate proposal-to-sample identity chain diverged."""


class ChoiceRouteRNGError(ChoiceRouteError):
    """A matched proposal did not use the installed request PCG64."""


def _source_sha256(function: Any) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise ChoiceRouteSourceMismatch(
            "could not inspect callable source for SHA-256 gate"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


def _state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_api(function: Any, names: tuple[str, ...], *, label: str) -> None:
    try:
        parameters = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise ChoiceRouteSourceMismatch(f"could not inspect {label} API") from exc
    if parameters[: len(names)] != names:
        raise ChoiceRouteSourceMismatch(
            f"{label} API mismatch: expected leading parameters {names}, got {parameters}"
        )


class NumpyChoiceRoute:
    """Install one immutable weighted-choice arm into a generation module."""

    def __init__(
        self,
        generation_module: Any,
        *,
        arm: str | RouteArm,
        expected_seed: int,
    ) -> None:
        if np.__version__ != REQUIRED_NUMPY_VERSION:
            raise RuntimeError(
                f"PR391 choice routes require exact NumPy {REQUIRED_NUMPY_VERSION}; "
                f"found {np.__version__}"
            )
        try:
            installed_arm = RouteArm(arm)
        except ValueError as exc:
            raise ValueError(f"unknown PR391 choice-route arm: {arm!r}") from exc
        if isinstance(expected_seed, bool):
            raise ValueError("expected_seed must be an integer accepted by NumPy PCG64")
        try:
            expected_seed = int(expected_seed)
            expected_rng = np.random.default_rng(expected_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected_seed must be an integer accepted by NumPy PCG64"
            ) from exc

        try:
            original_support = (
                generation_module.sparse_distribution_from_mlx_logits_relaxed_ties
            )
            original_sample = generation_module.sample_from_distribution
        except AttributeError as exc:
            raise ChoiceRouteSourceMismatch(
                "generation module is missing the reviewed choice-route API"
            ) from exc
        support_hash = _source_sha256(original_support)
        sample_hash = _source_sha256(original_sample)
        if support_hash != REVIEWED_SUPPORT_SOURCE_SHA256:
            raise ChoiceRouteSourceMismatch(
                "support source SHA-256 mismatch: expected "
                f"{REVIEWED_SUPPORT_SOURCE_SHA256}, got {support_hash}"
            )
        if sample_hash != REVIEWED_SAMPLE_SOURCE_SHA256:
            raise ChoiceRouteSourceMismatch(
                "sample source SHA-256 mismatch: expected "
                f"{REVIEWED_SAMPLE_SOURCE_SHA256}, got {sample_hash}"
            )
        _require_api(original_support, ("logits", "config"), label="support")
        _require_api(original_sample, ("probs", "rng"), label="sample")

        self._generation = generation_module
        self._arm = installed_arm
        self._expected_seed = expected_seed
        self._expected_start_state_hash = _state_sha256(expected_rng)
        self._rng: np.random.Generator | None = None
        self._rng_observer = self._bind_request_rng
        self._original_support = original_support
        self._original_sample = original_sample
        self._support_hash = support_hash
        self._sample_hash = sample_hash
        self._start_state_hash: str | None = None
        self._pending: Any | None = None
        self._matched = 0
        self._passthrough = 0
        self._failures = 0
        self._arm_counts = {candidate.value: 0 for candidate in RouteArm}
        self._installed = False
        self._ever_installed = False
        self._restored = False

        self._exact_row_type = None
        self._float32_row_type = None
        self._exact_selector = None
        self._float32_selector = None
        self._schedule_id = "numpy_generator_choice_control"
        if installed_arm is not RouteArm.CONTROL:
            analyzer = importlib.import_module("scripts.pr391_float32_choice_drift")
            if installed_arm is RouteArm.REDUCED_EXACT_FLOAT64:
                self._exact_row_type = analyzer.ReducedExactRow
                self._exact_selector = analyzer.select_reduced_exact_token
                self._schedule_id = analyzer.REDUCED_EXACT_SCHEDULE_ID
            else:
                self._float32_row_type = analyzer.ReducedFloat32Row
                self._float32_selector = analyzer.select_reduced_float32_token
                self._schedule_id = analyzer.CANDIDATE_SCHEDULE_ID
        if installed_arm is RouteArm.CONTROL:
            self._matched_sample = self._sample_control
        elif installed_arm is RouteArm.REDUCED_EXACT_FLOAT64:
            self._matched_sample = self._sample_exact
        else:
            self._matched_sample = self._sample_float32
        self._arm_count_key = installed_arm.value

    @classmethod
    def install(
        cls,
        generation_module: Any,
        *,
        arm: str | RouteArm,
        expected_seed: int,
    ) -> "NumpyChoiceRoute":
        route = cls(generation_module, arm=arm, expected_seed=expected_seed)
        route._install()
        return route

    def _install(self) -> None:
        self._support_wrapper_callable = self._support_wrapper
        self._sample_wrapper_callable = self._sample_wrapper
        self._generation.sparse_distribution_from_mlx_logits_relaxed_ties = (
            self._support_wrapper_callable
        )
        try:
            self._generation.sample_from_distribution = self._sample_wrapper_callable
        except BaseException:
            self._generation.sparse_distribution_from_mlx_logits_relaxed_ties = (
                self._original_support
            )
            raise
        self._installed = True
        self._ever_installed = True
        self._restored = False

    def __enter__(self) -> "NumpyChoiceRoute":
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
        if (
            self._generation.sparse_distribution_from_mlx_logits_relaxed_ties
            is not self._support_wrapper_callable
            or self._generation.sample_from_distribution
            is not self._sample_wrapper_callable
        ):
            self._failures += 1
            raise ChoiceRouteAssociationError(
                "choice-route wrappers changed before restoration"
            )
        self._generation.sparse_distribution_from_mlx_logits_relaxed_ties = (
            self._original_support
        )
        self._generation.sample_from_distribution = self._original_sample
        self._installed = False
        self._restored = True

    def _support_wrapper(self, logits: Any, config: Any) -> Any:
        if self._pending is not None:
            self._failures += 1
            raise ChoiceRouteAssociationError(
                "support called while an immediate pending proposal remains"
            )
        proposal = self._original_support(logits, config)
        if proposal is not None:
            self._pending = proposal
        return proposal

    def _sample_wrapper(self, probs: Any, rng: Any = None) -> int:
        self._rng_observer(rng)
        if self._pending is None:
            self._passthrough += 1
            return int(self._original_sample(probs, rng))
        if probs is not self._pending:
            self._failures += 1
            raise ChoiceRouteAssociationError(
                "sample does not match the pending object identity"
            )
        try:
            selected = int(self._matched_sample(probs, rng))
        except BaseException:
            self._failures += 1
            raise

        self._pending = None
        self._matched += 1
        self._arm_counts[self._arm_count_key] += 1
        return selected

    def _bind_request_rng(self, rng: Any) -> None:
        if self._pending is not None:
            self._failures += 1
            raise ChoiceRouteRNGError(
                "request RNG must be observed on a primary passthrough before any draft"
            )
        if (
            type(rng) is not np.random.Generator
            or type(rng.bit_generator) is not np.random.PCG64
        ):
            self._failures += 1
            raise ChoiceRouteRNGError(
                "first sampler passthrough requires an exact NumPy Generator/PCG64"
            )
        observed_hash = _state_sha256(rng)
        if observed_hash != self._expected_start_state_hash:
            self._failures += 1
            raise ChoiceRouteRNGError(
                f"request RNG does not match expected seed {self._expected_seed} start state"
            )
        self._rng = rng
        self._start_state_hash = observed_hash
        self._rng_observer = self._check_request_rng_identity

    def _check_request_rng_identity(self, rng: Any) -> None:
        if rng is not self._rng:
            self._failures += 1
            raise ChoiceRouteRNGError(
                "sampler must keep using the observed request PCG64 object identity"
            )

    def _sample_control(self, distribution: Any, rng: Any) -> int:
        return int(self._original_sample(distribution, rng))

    def _sample_exact(self, distribution: Any, rng: Any) -> int:
        return self._select_exact(distribution, float(rng.random()))

    def _sample_float32(self, distribution: Any, rng: Any) -> int:
        return self._select_float32(distribution, float(rng.random()))

    def _select_exact(self, distribution: Any, uniform: float) -> int:
        probabilities = distribution.probs
        row = self._exact_row_type(
            token_ids=distribution.token_ids,
            probabilities=probabilities,
            raw_cdf=np.cumsum(probabilities, dtype=np.float64),
            second_normalization_skipped=True,
        )
        return int(self._exact_selector(row, uniform))

    def _select_float32(self, distribution: Any, uniform: float) -> int:
        private = np.asarray(distribution.probs, dtype=np.float32).copy()
        first_total = np.sum(private, dtype=np.float32)
        private = (private / first_total).astype(np.float32, copy=False)
        second_total = np.sum(private, dtype=np.float32)
        skip_second = bool(
            second_total.view(np.uint32) == np.float32(1.0).view(np.uint32)
        )
        if not skip_second:
            private = (private / second_total).astype(np.float32, copy=False)
        row = self._float32_row_type(
            token_ids=distribution.token_ids,
            probabilities=private,
            raw_cdf=np.cumsum(private, dtype=np.float32),
            second_normalization_skipped=skip_second,
        )
        return int(self._float32_selector(row, uniform, cast_uniform=False))

    def finish_receipt(self, *, stats: Mapping[str, int] | Any) -> dict[str, Any]:
        if self._installed or not self._ever_installed or not self._restored:
            raise ChoiceRouteAssociationError(
                "finish receipt requires installed-then-closed restored wrappers"
            )
        if self._rng is None or self._start_state_hash is None:
            raise ChoiceRouteRNGError(
                "finish receipt requires that the request RNG was observed"
            )
        if self._pending is not None or self._failures:
            raise ChoiceRouteAssociationError(
                "finish receipt requires empty pending state and zero failures"
            )
        receipt = self._build_snapshot(stats)
        receipt.update(
            {
                "receipt_kind": "final_success",
                "start_pcg64_state_sha256": self._start_state_hash,
                "final_pcg64_state_sha256": _state_sha256(self._rng),
            }
        )
        return receipt

    def diagnostic_snapshot(
        self,
        *,
        stats: Mapping[str, int] | Any,
    ) -> dict[str, Any]:
        snapshot = self._build_snapshot(stats)
        snapshot.update(
            {
                "receipt_kind": (
                    "failed_diagnostic" if self._failures else "current_diagnostic"
                ),
                "start_pcg64_state_sha256": self._start_state_hash,
                "current_pcg64_state_sha256": (
                    _state_sha256(self._rng) if self._rng is not None else None
                ),
            }
        )
        return snapshot

    def _build_snapshot(self, stats: Mapping[str, int] | Any) -> dict[str, Any]:
        normalized_stats = self._normalize_stats(stats)
        draft_opportunities = normalized_stats["drafted_tokens"]
        if self._matched > draft_opportunities:
            raise ChoiceRouteAssociationError(
                "identity-matched route calls exceed stats.drafted_tokens"
            )
        verifier_opportunities = (
            normalized_stats["correction_tokens"] + normalized_stats["bonus_tokens"]
        )
        policy = {
            "fixed_at_install": True,
            "evaluation_scope": "corrected_e2e_benchmark",
            "retention_eligible": True,
        }
        if self._arm is RouteArm.REDUCED_FLOAT32_TEST_ONLY:
            policy.update(
                {
                    "evaluation_scope": "benchmark_experiment_only",
                    "retention_eligible": False,
                }
            )
        schedule = {
            "id": self._schedule_id,
            "uniform": (
                "original_sampler"
                if self._arm is RouteArm.CONTROL
                else "direct_one_draw_numpy_pcg64_float64"
            ),
            "cast_uniform": None if self._arm is RouteArm.CONTROL else False,
            "proposal_distribution_mutated": False,
        }
        return {
            "schema_version": 1,
            "arm": self._arm.value,
            "expected_seed": self._expected_seed,
            "numpy_version": REQUIRED_NUMPY_VERSION,
            "source_sha256": {
                "support": self._support_hash,
                "sample": self._sample_hash,
            },
            "route_counts": {
                "matched": self._matched,
                "passthrough": self._passthrough,
                "arm": dict(self._arm_counts),
                "pending": int(self._pending is not None),
                "failures": self._failures,
            },
            "schedule": schedule,
            "policy": policy,
            "stats": normalized_stats,
            "hit_miss": {
                "definitions": {
                    "draft_opportunities": "stats.drafted_tokens",
                    "draft_hits": "identity_matched_route_calls",
                    "verifier_choice_opportunities": (
                        "stats.correction_tokens + stats.bonus_tokens"
                    ),
                    "verifier_hits": "always_zero_draft_only_constructor_route",
                },
                "draft": {
                    "opportunities": draft_opportunities,
                    "hits": self._matched,
                    "misses": draft_opportunities - self._matched,
                    "hit_rate": self._rate(self._matched, draft_opportunities),
                },
                "verifier": {
                    "opportunities": verifier_opportunities,
                    "hits": 0,
                    "misses": verifier_opportunities,
                    "hit_rate": self._rate(0, verifier_opportunities),
                },
            },
        }

    @staticmethod
    def _rate(hits: int, opportunities: int) -> float | None:
        return hits / opportunities if opportunities else None

    @staticmethod
    def _normalize_stats(stats: Mapping[str, int] | Any) -> dict[str, int]:
        if isinstance(stats, Mapping):
            source = stats
            missing = [field for field in STATS_FIELDS if field not in source]
            if missing:
                raise ValueError(f"route receipt stats missing: {', '.join(missing)}")
            normalized = {field: int(source[field]) for field in STATS_FIELDS}
        else:
            try:
                normalized = {
                    field: int(getattr(stats, field)) for field in STATS_FIELDS
                }
            except AttributeError as exc:
                raise ValueError("route receipt stats are incomplete") from exc
        if any(value < 0 for value in normalized.values()):
            raise ValueError("route receipt stats must be non-negative")
        return normalized


__all__ = [
    "ChoiceRouteAssociationError",
    "ChoiceRouteError",
    "ChoiceRouteRNGError",
    "ChoiceRouteSourceMismatch",
    "NumpyChoiceRoute",
    "REQUIRED_NUMPY_VERSION",
    "REVIEWED_SAMPLE_SOURCE_SHA256",
    "REVIEWED_SUPPORT_SOURCE_SHA256",
    "RouteArm",
]
