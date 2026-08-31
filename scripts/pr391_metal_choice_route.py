#!/usr/bin/env python3
"""Synchronous, test-only native Metal float32 draft-choice route for PR #391."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from types import TracebackType
from typing import Any, Mapping

import numpy as np


REQUIRED_NUMPY_VERSION = "2.4.4"
REVIEWED_SPARSE_SUPPORT_SHA256 = (
    "1caabc0070767ac2252e5f6511ed54b038e2d880527c28d8c4333503a4258822"
)
REVIEWED_SAMPLE_SHA256 = (
    "4dc740a4b320b54c5b38b3bc8ab13935d8445407d5690e9da9eac867cc9ecba7"
)
REVIEWED_KERNEL_BINDER_SHA256 = (
    "627b3906af3902f963a5e2d62943097280eb32fa9d445d645f3b2513b1b48488"
)
REVIEWED_DESCRIPTOR_BUILDER_SHA256 = (
    "250e4666ba5f2a7b38b2aed1c9a75475e5b5bdf0d5983cd8f86c59aa6e1f3659"
)
REVIEWED_KERNEL_SELFCHECK_SHA256 = (
    "07e17594e6681e4dc8096d322f8efebf29d6e9dae20f19333f0c89f0e9db9a78"
)
REVIEWED_METAL_SOURCE_SHA256 = (
    "4ad73064240ad78f95969d541eed9407aac65d265dee9dd19a4563b1fe7f3356"
)
ARM = "metal-float32-test-only"
STATS_FIELDS = (
    "drafted_tokens",
    "accepted_drafts",
    "verify_calls",
    "correction_tokens",
    "bonus_tokens",
)


class MetalChoiceRouteError(RuntimeError):
    """Base class for fail-closed synchronous Metal route errors."""


class MetalChoiceRouteConfigError(MetalChoiceRouteError):
    """Construction-time route configuration is not the reviewed workload."""


class MetalChoiceRouteSourceMismatch(MetalChoiceRouteError):
    """A callable or Metal program differs from its reviewed source."""


class MetalChoiceRouteAssociationError(MetalChoiceRouteError):
    """The immediate raw-support/proposal/sample ownership chain diverged."""


class MetalChoiceRouteRNGError(MetalChoiceRouteError):
    """The route did not observe the expected request-owned PCG64."""


def _source_sha256(function: Any) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise MetalChoiceRouteSourceMismatch(
            "could not inspect callable source for SHA-256 gate"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


def _require_source(function: Any, expected: str, *, label: str) -> str:
    observed = _source_sha256(function)
    if observed != expected:
        raise MetalChoiceRouteSourceMismatch(
            f"{label} source SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return observed


def _require_api(function: Any, names: tuple[str, ...], *, label: str) -> None:
    try:
        parameters = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise MetalChoiceRouteSourceMismatch(f"could not inspect {label} API") from exc
    if parameters[: len(names)] != names:
        raise MetalChoiceRouteSourceMismatch(
            f"{label} API mismatch: expected leading parameters {names}, "
            f"got {parameters}"
        )


def _state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _padded_distribution_inputs(
    distribution: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map an already top-p-filtered production q to the fixed K20 ABI.

    SparseDistribution stores support in ascending token-ID order. Descending
    synthetic values preserve that order through the kernel's first reduction;
    top_p=1 then makes the kernel exactly the final NumPy-f32 normalize/choice
    port, without applying production nucleus filtering a second time.
    """

    ids = np.asarray(distribution.token_ids)
    probs = np.asarray(distribution.probs)
    if (
        ids.dtype != np.dtype(np.int64)
        or probs.dtype != np.dtype(np.float64)
        or ids.ndim != 1
        or probs.shape != ids.shape
        or not 1 <= ids.size <= 20
        or np.any(ids < 0)
        or np.any(ids > np.iinfo(np.uint32).max)
        or np.unique(ids).size != ids.size
        or np.any(np.diff(ids) <= 0)
        or not np.all(np.isfinite(probs))
        or np.any(probs <= 0.0)
    ):
        raise MetalChoiceRouteAssociationError(
            "pending proposal does not satisfy the fixed sparse K20 contract"
        )
    ids_host = np.empty((1, 20), dtype=np.uint32)
    probs_host = np.zeros((1, 20), dtype=np.float32)
    values_host = np.arange(20, 0, -1, dtype=np.float32).reshape(1, 20)
    ids_host[0, : ids.size] = ids.astype(np.uint32)
    probs_host[0, : ids.size] = probs.astype(np.float32)
    used = {int(value) for value in ids}
    padding = np.iinfo(np.uint32).max
    for index in range(ids.size, 20):
        while padding in used:
            padding -= 1
        ids_host[0, index] = padding
        used.add(padding)
        padding -= 1
    return ids_host, values_host, probs_host


@dataclass
class PreboundMetalFloat32ChoiceKernel:
    """Construction-gated selector reused by every request in one benchmark."""

    mx: Any
    kernel_module: Any
    selector: Any
    selfcheck: Mapping[str, Any]
    source_sha256: Mapping[str, str]
    _prewarm_receipt: dict[str, Any] | None = None

    def prewarm_b1(self) -> dict[str, Any]:
        """Compile/evaluate B1 once and prove the kernel's raw outputs bit-exact."""

        if self._prewarm_receipt is not None:
            return dict(self._prewarm_receipt)
        ids_host = np.arange(100, 120, dtype=np.uint32).reshape(1, 20)
        values_host = np.linspace(2.0, -2.0, 20, dtype=np.float32).reshape(1, 20)
        probs_host = np.linspace(0.12, 0.001, 20, dtype=np.float32).reshape(1, 20)
        descriptor_host = self.kernel_module.build_pcg64_midpoint_descriptors(
            np.array([np.ldexp(np.float64(391), -53)], dtype=np.float64)
        )
        ids = self.mx.array(ids_host, dtype=self.mx.uint32)
        values = self.mx.array(values_host, dtype=self.mx.float32)
        probs = self.mx.array(probs_host, dtype=self.mx.float32)
        descriptors = self.mx.array(descriptor_host, dtype=self.mx.uint32)
        selected, raw_ids, raw_values, raw_probs = self.selector(
            ids, values, probs, descriptors
        )
        self.mx.eval(selected, raw_ids, raw_values, raw_probs)
        expected_selected, *_ = (
            self.kernel_module.reference_qwen4_frspec_k20_float32_choice(
                ids_host,
                values_host,
                probs_host,
                descriptor_host,
                top_p=1.0,
            )
        )
        selected_token_match = bool(
            np.array_equal(np.asarray(selected), expected_selected)
        )
        raw_bit_exact = all(
            np.asarray(observed).tobytes() == expected.tobytes()
            and np.asarray(observed).dtype == expected.dtype
            and np.asarray(observed).shape == expected.shape
            for observed, expected in zip(
                (raw_ids, raw_values, raw_probs),
                (ids_host, values_host, probs_host),
                strict=True,
            )
        )
        if not raw_bit_exact or not selected_token_match:
            raise MetalChoiceRouteConfigError(
                "B1 prewarm failed selected-token or raw-passthrough proof"
            )
        peak_getter = getattr(self.mx, "get_peak_memory", None)
        peak_memory = int(peak_getter()) if peak_getter is not None else None
        self._prewarm_receipt = {
            "status": "passed",
            "rows": 1,
            "raw_passthrough_bit_exact": True,
            "selected_token_match": True,
            "peak_memory_bytes": peak_memory,
            "schedule_id": str(self.selfcheck["schedule_id"]),
        }
        return dict(self._prewarm_receipt)

    @property
    def prewarm_receipt(self) -> dict[str, Any] | None:
        return None if self._prewarm_receipt is None else dict(self._prewarm_receipt)


def prebind_metal_float32_choice_kernel(
    *,
    mx_module: Any | None = None,
    kernel_module: Any | None = None,
) -> PreboundMetalFloat32ChoiceKernel:
    """Validate and bind the fixed K20/top-p=.95 selector before requests."""

    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise MetalChoiceRouteConfigError(
            f"native float32 choice requires exact NumPy {REQUIRED_NUMPY_VERSION}; "
            f"found {np.__version__}"
        )
    if kernel_module is None:
        kernel_module = importlib.import_module(
            "mtplx.kernels.qwen4_frspec_k20_float32_choice"
        )
    if mx_module is None:
        mx_module = importlib.import_module("mlx.core")
    source_hashes = {
        "kernel_binder": _require_source(
            kernel_module.bind_qwen4_frspec_k20_float32_choice,
            REVIEWED_KERNEL_BINDER_SHA256,
            label="kernel binder",
        ),
        "descriptor_builder": _require_source(
            kernel_module.build_pcg64_midpoint_descriptors,
            REVIEWED_DESCRIPTOR_BUILDER_SHA256,
            label="descriptor builder",
        ),
        "kernel_selfcheck": _require_source(
            kernel_module.selfcheck_qwen4_frspec_k20_float32_choice,
            REVIEWED_KERNEL_SELFCHECK_SHA256,
            label="kernel selfcheck",
        ),
    }
    _require_api(
        kernel_module.build_pcg64_midpoint_descriptors,
        ("uniforms",),
        label="descriptor builder",
    )
    metal_source = getattr(kernel_module, "METAL_HEADER", "") + getattr(
        kernel_module, "METAL_SOURCE", ""
    )
    if metal_source:
        metal_hash = hashlib.sha256(metal_source.encode()).hexdigest()
        if metal_hash != REVIEWED_METAL_SOURCE_SHA256:
            raise MetalChoiceRouteSourceMismatch(
                "Metal selector source SHA-256 mismatch: expected "
                f"{REVIEWED_METAL_SOURCE_SHA256}, got {metal_hash}"
            )
        source_hashes["metal"] = metal_hash
    selfcheck = dict(kernel_module.selfcheck_qwen4_frspec_k20_float32_choice())
    if (
        int(selfcheck.get("k", -1)) != 20
        or int(selfcheck.get("descriptor_words", -1)) != 5
        or not selfcheck.get("schedule_id")
    ):
        raise MetalChoiceRouteConfigError(
            "K20 float32 selector selfcheck contract is incomplete"
        )
    # Production top-p has already been applied to SparseDistribution.  The
    # Metal port must only reproduce the NumPy-f32 renormalization/choice.
    selector = kernel_module.bind_qwen4_frspec_k20_float32_choice(top_p=1.0)
    return PreboundMetalFloat32ChoiceKernel(
        mx=mx_module,
        kernel_module=kernel_module,
        selector=selector,
        selfcheck=selfcheck,
        source_sha256=source_hashes,
    )


class MetalFloat32ChoiceRoute:
    """Install the fixed synchronous Metal selector into one benchmark request."""

    def __init__(
        self,
        generation_module: Any,
        *,
        expected_seed: int,
        kernel_module: PreboundMetalFloat32ChoiceKernel,
        sampler: Any,
    ) -> None:
        if np.__version__ != REQUIRED_NUMPY_VERSION:
            raise MetalChoiceRouteConfigError(
                f"native float32 choice requires exact NumPy {REQUIRED_NUMPY_VERSION}; "
                f"found {np.__version__}"
            )
        if not isinstance(kernel_module, PreboundMetalFloat32ChoiceKernel):
            raise MetalChoiceRouteConfigError(
                "kernel_module must be a prebound Metal float32 choice kernel"
            )
        if kernel_module.prewarm_receipt is None:
            raise MetalChoiceRouteConfigError(
                "prebound Metal float32 choice kernel requires successful B1 prewarm"
            )
        if (
            int(sampler.top_k) != 20
            or float(sampler.top_p) != 0.95
            or float(sampler.temperature) != 1.0
        ):
            raise MetalChoiceRouteConfigError(
                "route requires exact top_k=20, top_p=0.95, temperature=1"
            )
        if isinstance(expected_seed, bool):
            raise MetalChoiceRouteConfigError(
                "expected_seed must be a PCG64 seed integer"
            )
        try:
            self._expected_seed = int(expected_seed)
            expected_rng = np.random.default_rng(self._expected_seed)
        except (TypeError, ValueError) as exc:
            raise MetalChoiceRouteConfigError(
                "expected_seed must be a PCG64 seed integer"
            ) from exc

        try:
            original_support = (
                generation_module.sparse_distribution_from_mlx_logits_relaxed_ties
            )
            original_sample = generation_module.sample_from_distribution
        except AttributeError as exc:
            raise MetalChoiceRouteSourceMismatch(
                "generation module is missing the reviewed relaxed-choice API"
            ) from exc
        source_hashes = {
            "sparse_support": _require_source(
                original_support,
                REVIEWED_SPARSE_SUPPORT_SHA256,
                label="sparse support",
            ),
            "sample": _require_source(
                original_sample,
                REVIEWED_SAMPLE_SHA256,
                label="sample",
            ),
        }
        _require_api(original_support, ("logits", "config"), label="sparse support")
        _require_api(original_sample, ("probs", "rng"), label="sample")

        self._generation = generation_module
        self._prebound = kernel_module
        self._mx = kernel_module.mx
        self._kernel = kernel_module.kernel_module
        self._selector = kernel_module.selector
        self._sampler = sampler
        self._original_support = original_support
        self._original_sample = original_sample
        self._source_hashes = source_hashes
        self._expected_start_hash = _state_sha256(expected_rng)
        self._rng: np.random.Generator | None = None
        self._start_hash: str | None = None
        self._pending_proposal: Any | None = None
        self._failures = 0
        self._installed = False
        self._ever_installed = False
        self._restored = False

    @classmethod
    def install(
        cls,
        generation_module: Any,
        *,
        expected_seed: int,
        kernel_module: PreboundMetalFloat32ChoiceKernel,
        sampler: Any,
    ) -> "MetalFloat32ChoiceRoute":
        route = cls(
            generation_module,
            expected_seed=expected_seed,
            kernel_module=kernel_module,
            sampler=sampler,
        )
        route._install()
        return route

    def _install(self) -> None:
        self._support_wrapper_callable = self._sparse_support_wrapper
        self._sample_wrapper_callable = self._sample_wrapper
        self._generation.sparse_distribution_from_mlx_logits_relaxed_ties = (
            self._support_wrapper_callable
        )
        self._generation.sample_from_distribution = self._sample_wrapper_callable
        self._installed = True
        self._ever_installed = True
        self._restored = False

    def __enter__(self) -> "MetalFloat32ChoiceRoute":
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
        wrappers_match = (
            self._generation.sparse_distribution_from_mlx_logits_relaxed_ties
            is self._support_wrapper_callable
            and self._generation.sample_from_distribution
            is self._sample_wrapper_callable
        )
        if not wrappers_match:
            self._failures += 1
            raise MetalChoiceRouteAssociationError(
                "Metal choice wrappers changed before restoration"
            )
        self._generation.sparse_distribution_from_mlx_logits_relaxed_ties = (
            self._original_support
        )
        self._generation.sample_from_distribution = self._original_sample
        self._installed = False
        self._restored = True

    def _sparse_support_wrapper(self, logits: Any, config: Any) -> Any:
        if self._pending_proposal is not None:
            self._failures += 1
            raise MetalChoiceRouteAssociationError(
                "sparse support called while an immediate pending proposal remains"
            )
        if config is not self._sampler:
            self._failures += 1
            raise MetalChoiceRouteAssociationError(
                "sparse support sampler identity differs from installed draft sampler"
            )
        proposal = self._original_support(logits, config)
        if proposal is None:
            self._failures += 1
            raise MetalChoiceRouteAssociationError(
                "reviewed sparse support did not produce one K20 proposal"
            )
        self._pending_proposal = proposal
        return proposal

    def _sample_wrapper(self, probs: Any, rng: Any = None) -> int:
        self._observe_rng(rng)
        if self._pending_proposal is None:
            return int(self._original_sample(probs, rng))
        if probs is not self._pending_proposal:
            self._failures += 1
            raise MetalChoiceRouteAssociationError(
                "sample does not match the pending proposal object identity"
            )
        ids_host, values_host, probs_host = _padded_distribution_inputs(probs)
        uniform = float(rng.random())
        descriptor_host = self._kernel.build_pcg64_midpoint_descriptors(
            np.array([uniform], dtype=np.float64)
        )
        raw_ids = self._mx.array(ids_host, dtype=self._mx.uint32)
        raw_values = self._mx.array(values_host, dtype=self._mx.float32)
        raw_probs = self._mx.array(probs_host, dtype=self._mx.float32)
        descriptors = self._mx.array(descriptor_host, dtype=self._mx.uint32)
        selected, _raw_ids, _raw_values, _raw_probs = self._selector(
            raw_ids, raw_values, raw_probs, descriptors
        )
        token = int(selected.item())
        self._pending_proposal = None
        return token

    def _observe_rng(self, rng: Any) -> None:
        if self._rng is None:
            if (
                type(rng) is not np.random.Generator
                or type(rng.bit_generator) is not np.random.PCG64
            ):
                self._failures += 1
                raise MetalChoiceRouteRNGError(
                    "first sampler call requires exact NumPy Generator/PCG64"
                )
            observed = _state_sha256(rng)
            if observed != self._expected_start_hash:
                self._failures += 1
                raise MetalChoiceRouteRNGError(
                    f"request RNG does not match expected seed "
                    f"{self._expected_seed} start state"
                )
            self._rng = rng
            self._start_hash = observed
        elif rng is not self._rng:
            self._failures += 1
            raise MetalChoiceRouteRNGError(
                "sampler must keep using the observed request PCG64 object identity"
            )

    def finish_receipt(self, *, stats: Mapping[str, int] | Any) -> dict[str, Any]:
        if self._installed or not self._ever_installed or not self._restored:
            raise MetalChoiceRouteAssociationError(
                "finish receipt requires an installed-then-closed route"
            )
        if self._rng is None or self._start_hash is None:
            raise MetalChoiceRouteRNGError(
                "finish receipt requires that the request RNG was observed"
            )
        if self._pending_proposal is not None:
            raise MetalChoiceRouteAssociationError(
                "finish receipt requires empty pending proposal state"
            )
        if self._failures:
            raise MetalChoiceRouteAssociationError(
                "finish receipt requires zero route failures"
            )
        normalized_stats = self._normalize_stats(stats)
        prewarm = self._prebound.prewarm_receipt
        if prewarm is None or prewarm.get("status") != "passed":
            raise MetalChoiceRouteConfigError(
                "final receipt requires successful B1 prewarm"
            )
        return {
            "schema_version": 1,
            "receipt_kind": "final_success",
            "arm": ARM,
            "expected_seed": self._expected_seed,
            "numpy_version": REQUIRED_NUMPY_VERSION,
            "source_sha256": {
                **self._source_hashes,
                **dict(self._prebound.source_sha256),
            },
            "start_pcg64_state_sha256": self._start_hash,
            "final_pcg64_state_sha256": _state_sha256(self._rng),
            "route_counts": {
                "calls": normalized_stats["drafted_tokens"],
                "matched_rows": normalized_stats["drafted_tokens"],
                "passthrough": None,
                "pending": 0,
                "failures": 0,
                "raw_passthrough_rows": normalized_stats["drafted_tokens"],
                "count_source": "stats.drafted_tokens_under_structural_route",
            },
            "selected_token_mismatches": None,
            "drift_observation": "external_output_digest_gate",
            "prebound": prewarm,
            "selector_memory": {
                "prewarm_peak_memory_bytes": prewarm["peak_memory_bytes"],
            },
            "schedule": {
                "id": str(self._prebound.selfcheck["schedule_id"]),
                "uniform": "direct_one_draw_numpy_pcg64_float64_descriptor",
                "kernel_top_p": 1.0,
                "input": "production_sparse_distribution_probs_cast_float32",
                "proposal_distribution_mutated": False,
            },
            "policy": {
                "fixed_at_install": True,
                "evaluation_scope": "benchmark_experiment_only",
                "retention_eligible": False,
                "sync_boundary": "one_selected_token_item_per_draft",
            },
            "stats": normalized_stats,
        }

    @staticmethod
    def _normalize_stats(stats: Mapping[str, int] | Any) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for field in STATS_FIELDS:
            try:
                raw = (
                    stats[field]
                    if isinstance(stats, Mapping)
                    else getattr(stats, field)
                )
                value = int(raw)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise MetalChoiceRouteAssociationError(
                    f"stats must provide integer {field}"
                ) from exc
            if value < 0:
                raise MetalChoiceRouteAssociationError(
                    f"stats.{field} must be non-negative"
                )
            normalized[field] = value
        return normalized
