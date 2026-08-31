from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@dataclass(frozen=True)
class FakeSparseDistribution:
    token_ids: np.ndarray
    probs: np.ndarray
    vocab_size: int


class FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, *shape):
        return FakeArray(self.value.reshape(*shape))

    def astype(self, dtype):
        return FakeArray(self.value.astype(dtype))

    def __getitem__(self, item):
        return FakeArray(self.value[item])

    def __neg__(self):
        return FakeArray(-self.value)

    def __mul__(self, other):
        return FakeArray(self.value * other)

    def __add__(self, other):
        return FakeArray(self.value + np.asarray(other))

    def __sub__(self, other):
        return FakeArray(self.value - np.asarray(other))

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(self.value, dtype=dtype)

    def item(self):
        return self.value.item()


class FakeMX:
    float32 = np.float32
    uint32 = np.uint32

    def __init__(self):
        self.evals = 0

    @staticmethod
    def array(value, dtype=None):
        return FakeArray(np.asarray(value, dtype=dtype))

    @staticmethod
    def argpartition(value, kth, axis):
        return FakeArray(np.argpartition(np.asarray(value), kth=kth, axis=axis))

    @staticmethod
    def take_along_axis(value, indices, axis):
        return FakeArray(
            np.take_along_axis(np.asarray(value), np.asarray(indices), axis=axis)
        )

    @staticmethod
    def logsumexp(value, axis, keepdims):
        rows = np.asarray(value)
        maximum = np.max(rows, axis=axis, keepdims=True)
        result = maximum + np.log(
            np.sum(np.exp(rows - maximum), axis=axis, keepdims=True)
        )
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return FakeArray(result)

    @staticmethod
    def exp(value):
        return FakeArray(np.exp(np.asarray(value)))

    def eval(self, *values):
        del values
        self.evals += 1

    @staticmethod
    def get_peak_memory():
        return 123_456


_device_serial_support_arrays_relaxed_ties = None


def fake_sparse_support(logits, config):
    token_rows, prob_rows, vocab_size = _device_serial_support_arrays_relaxed_ties(
        logits, config
    )
    keep = prob_rows[0] > 0
    ids = token_rows[0][keep]
    probs = prob_rows[0][keep]
    order = np.argsort(ids)
    probs = probs[order] / probs[order].sum()
    return FakeSparseDistribution(ids[order], probs, vocab_size)


def fake_sample(probs, rng=None):
    rng = rng or np.random.default_rng()
    return int(rng.choice(probs.token_ids, p=probs.probs))


class FakeKernelModule:
    K20 = 20
    MIDPOINT_DESCRIPTOR_WORDS = 5
    SCHEDULE_ID = "fake-metal-float32"

    def __init__(self):
        self.bind_calls = 0
        self.dispatches = 0

    @staticmethod
    def selfcheck_qwen4_frspec_k20_float32_choice():
        return {
            "schedule_id": "fake-metal-float32",
            "k": 20,
            "descriptor_words": 5,
            "cases": 4,
        }

    @staticmethod
    def build_pcg64_midpoint_descriptors(uniforms):
        from mtplx.kernels.qwen4_frspec_k20_float32_choice import (
            build_pcg64_midpoint_descriptors,
        )

        return build_pcg64_midpoint_descriptors(uniforms)

    @staticmethod
    def reference_qwen4_frspec_k20_float32_choice(*args, **kwargs):
        from mtplx.kernels.qwen4_frspec_k20_float32_choice import (
            reference_qwen4_frspec_k20_float32_choice,
        )

        return reference_qwen4_frspec_k20_float32_choice(*args, **kwargs)

    def bind_qwen4_frspec_k20_float32_choice(self, *, top_p=0.95):
        assert top_p == 1.0
        self.bind_calls += 1

        def select(ids, values, probs, descriptors):
            self.dispatches += 1
            host_ids = np.asarray(ids)
            host_probs = np.asarray(probs)
            descriptor_words = np.asarray(descriptors, dtype=np.uint32)
            selected = np.empty(host_ids.shape[0], dtype=np.uint32)
            for row in range(host_ids.shape[0]):
                private = host_probs[row][host_probs[row] > 0].astype(np.float32)
                private = (private / np.sum(private, dtype=np.float32)).astype(
                    np.float32
                )
                total = np.sum(private, dtype=np.float32)
                if total.view(np.uint32) != np.float32(1.0).view(np.uint32):
                    private = (private / total).astype(np.float32)
                raw_cdf = np.cumsum(private, dtype=np.float32)
                integer = (int(descriptor_words[row, 0]) << 32) | int(
                    descriptor_words[row, 1]
                )
                uniform = integer / float(1 << 53)
                picked = private.size - 1
                for index, boundary in enumerate(raw_cdf[:-1]):
                    if float(np.float32(boundary / raw_cdf[-1])) > uniform:
                        picked = index
                        break
                selected[row] = host_ids[row, picked]
            return FakeArray(selected), ids, values, probs

        return select


def _source_hash(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _fixture(seed=391):
    import scripts.pr391_metal_choice_route as route_module

    mx = FakeMX()
    kernel = FakeKernelModule()
    generation = SimpleNamespace(
        sparse_distribution_from_mlx_logits_relaxed_ties=fake_sparse_support,
        sample_from_distribution=fake_sample,
    )
    original_device = _reference_device_support(mx)
    globals()["_device_serial_support_arrays_relaxed_ties"] = original_device
    gates = (
        patch.object(
            route_module,
            "REVIEWED_SPARSE_SUPPORT_SHA256",
            _source_hash(fake_sparse_support),
        ),
        patch.object(route_module, "REVIEWED_SAMPLE_SHA256", _source_hash(fake_sample)),
        patch.object(
            route_module,
            "REVIEWED_KERNEL_BINDER_SHA256",
            _source_hash(kernel.bind_qwen4_frspec_k20_float32_choice),
        ),
        patch.object(
            route_module,
            "REVIEWED_DESCRIPTOR_BUILDER_SHA256",
            _source_hash(kernel.build_pcg64_midpoint_descriptors),
        ),
        patch.object(
            route_module,
            "REVIEWED_KERNEL_SELFCHECK_SHA256",
            _source_hash(kernel.selfcheck_qwen4_frspec_k20_float32_choice),
        ),
    )
    return route_module, mx, kernel, generation, original_device, gates, seed


def _reference_device_support(mx):
    def device(logits, config):
        rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        vocab_size = int(rows.shape[-1])
        k = min(int(config.top_k), vocab_size)
        scaled = rows * (1.0 / float(config.temperature))
        cand_idx = mx.argpartition(-scaled, kth=k - 1, axis=-1)[:, :k]
        cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
        log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
        cand_probs = mx.exp(cand_vals - log_total)
        mx.eval(cand_idx, cand_vals, cand_probs)
        token_rows = np.asarray(cand_idx, dtype=np.int64)
        value_rows = np.asarray(cand_vals, dtype=np.float32)
        prob_rows = np.asarray(cand_probs, dtype=np.float64)
        order = np.lexsort((token_rows, -value_rows), axis=1)
        token_rows = np.take_along_axis(token_rows, order, axis=1)
        prob_rows = np.take_along_axis(prob_rows, order, axis=1)
        before = np.concatenate(
            (np.zeros((prob_rows.shape[0], 1)), np.cumsum(prob_rows[:, :-1], axis=1)),
            axis=1,
        )
        prob_rows = np.where(before < float(config.top_p), prob_rows, 0.0)
        return token_rows, prob_rows, vocab_size

    return device


def _activate(gates):
    for gate in gates:
        gate.start()


def _deactivate(gates):
    for gate in reversed(gates):
        gate.stop()


def _stats(drafted=1):
    return {
        "drafted_tokens": drafted,
        "accepted_drafts": 0,
        "verify_calls": drafted,
        "correction_tokens": drafted,
        "bonus_tokens": 0,
    }


def test_import_is_cpu_only():
    import scripts.pr391_metal_choice_route as route_module

    assert "mlx" not in route_module.__dict__
    assert "mtplx" not in route_module.__dict__


def test_padded_metal_schedule_exactly_ports_numpy_float32_choice() -> None:
    from mtplx.kernels import qwen4_frspec_k20_float32_choice as kernel
    from scripts import pr391_float32_choice_drift as analyzer
    from scripts.pr391_metal_choice_route import _padded_distribution_inputs

    rng = np.random.default_rng(20260830)
    for support in range(1, 21):
        for _ in range(15):
            token_ids = np.sort(
                rng.choice(248_320, size=support, replace=False).astype(np.int64)
            )
            probabilities = rng.random(support)
            probabilities /= probabilities.sum()
            distribution = FakeSparseDistribution(
                token_ids, probabilities.astype(np.float64), 248_320
            )
            uniform = rng.random(1, dtype=np.float64)
            ids, values, probs = _padded_distribution_inputs(distribution)
            descriptors = kernel.build_pcg64_midpoint_descriptors(uniform)
            observed, *_ = kernel.reference_qwen4_frspec_k20_float32_choice(
                ids, values, probs, descriptors, top_p=1.0
            )

            private = probabilities.astype(np.float32)
            first_total = np.sum(private, dtype=np.float32)
            private = (private / first_total).astype(np.float32, copy=False)
            second_total = np.sum(private, dtype=np.float32)
            skip_second = bool(
                second_total.view(np.uint32) == np.float32(1.0).view(np.uint32)
            )
            if not skip_second:
                private = (private / second_total).astype(np.float32, copy=False)
            row = analyzer.ReducedFloat32Row(
                token_ids=token_ids,
                probabilities=private,
                raw_cdf=np.cumsum(private, dtype=np.float32),
                second_normalization_skipped=skip_second,
            )
            expected = analyzer.select_reduced_float32_token(
                row, float(uniform[0]), cast_uniform=False
            )
            assert int(observed[0]) == expected


def test_prebind_runs_exact_gates_selfcheck_once_and_prewarm_proves_raw_bits():
    route_module, mx, kernel, _, _, gates, _ = _fixture()
    _activate(gates)
    try:
        prebound = route_module.prebind_metal_float32_choice_kernel(
            mx_module=mx, kernel_module=kernel
        )
        receipt = prebound.prewarm_b1()
    finally:
        _deactivate(gates)

    assert kernel.bind_calls == 1
    assert receipt["raw_passthrough_bit_exact"] is True
    assert receipt["selected_token_match"] is True
    assert receipt["rows"] == 1
    assert receipt["peak_memory_bytes"] == 123_456


def test_install_rejects_noncanonical_sampler_before_publishing_wrappers():
    route_module, mx, kernel, generation, original_device, gates, seed = _fixture()
    _activate(gates)
    original_support = generation.sparse_distribution_from_mlx_logits_relaxed_ties
    original_sample = generation.sample_from_distribution
    try:
        prebound = route_module.prebind_metal_float32_choice_kernel(
            mx_module=mx, kernel_module=kernel
        )
        prebound.prewarm_b1()
        with pytest.raises(route_module.MetalChoiceRouteConfigError, match="top_k=20"):
            route_module.MetalFloat32ChoiceRoute.install(
                generation,
                expected_seed=seed,
                kernel_module=prebound,
                sampler=SimpleNamespace(top_k=19, top_p=0.95, temperature=1.0),
            )
    finally:
        _deactivate(gates)
    assert (
        generation.sparse_distribution_from_mlx_logits_relaxed_ties is original_support
    )
    assert generation.sample_from_distribution is original_sample
    assert globals()["_device_serial_support_arrays_relaxed_ties"] is original_device


def test_matched_route_preserves_q_consumes_one_uniform_and_dispatches_metal():
    route_module, mx, kernel, generation, original_device, gates, seed = _fixture()
    _activate(gates)
    rng = np.random.default_rng(seed)
    reference_rng = np.random.default_rng(seed)
    logits = FakeArray(np.linspace(-3.0, 3.0, 64, dtype=np.float32))
    sampler = SimpleNamespace(top_k=20, top_p=0.95, temperature=1.0)
    try:
        prebound = route_module.prebind_metal_float32_choice_kernel(
            mx_module=mx, kernel_module=kernel
        )
        prebound.prewarm_b1()
        route = route_module.MetalFloat32ChoiceRoute.install(
            generation,
            expected_seed=seed,
            kernel_module=prebound,
            sampler=sampler,
        )
        proposal = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            logits, sampler
        )
        ids_before = proposal.token_ids.copy()
        probs_before = proposal.probs.copy()
        expected_uniform = float(reference_rng.random())
        token = generation.sample_from_distribution(proposal, rng)
        private = proposal.probs.astype(np.float32)
        private = (private / np.sum(private, dtype=np.float32)).astype(np.float32)
        total = np.sum(private, dtype=np.float32)
        if total.view(np.uint32) != np.float32(1.0).view(np.uint32):
            private = (private / total).astype(np.float32)
        expected = int(
            proposal.token_ids[
                np.searchsorted(
                    np.cumsum(private, dtype=np.float32) / np.sum(private),
                    expected_uniform,
                    side="right",
                )
            ]
        )
        assert token == expected
        assert rng.bit_generator.state == reference_rng.bit_generator.state
        assert expected_uniform >= 0.0
        np.testing.assert_array_equal(proposal.token_ids, ids_before)
        np.testing.assert_array_equal(
            proposal.probs.view(np.uint64), probs_before.view(np.uint64)
        )
        route.close()
        assert (
            globals()["_device_serial_support_arrays_relaxed_ties"] is original_device
        )
        receipt = route.finish_receipt(stats=_stats())
    finally:
        if "route" in locals():
            route.close()
        _deactivate(gates)

    assert receipt["arm"] == "metal-float32-test-only"
    assert receipt["route_counts"] == {
        "calls": 1,
        "matched_rows": 1,
        "passthrough": None,
        "pending": 0,
        "failures": 0,
        "raw_passthrough_rows": 1,
        "count_source": "stats.drafted_tokens_under_structural_route",
    }
    assert receipt["selected_token_mismatches"] is None
    assert receipt["drift_observation"] == "external_output_digest_gate"
    assert receipt["policy"]["retention_eligible"] is False
    assert receipt["policy"]["sync_boundary"] == "one_selected_token_item_per_draft"


def test_passthrough_binds_rng_then_identity_mismatch_fails_closed():
    route_module, mx, kernel, generation, _, gates, seed = _fixture()
    _activate(gates)
    rng = np.random.default_rng(seed)
    distribution = FakeSparseDistribution(
        np.array([1, 2], dtype=np.int64), np.array([0.25, 0.75]), 3
    )
    sampler = SimpleNamespace(top_k=20, top_p=0.95, temperature=1.0)
    try:
        prebound = route_module.prebind_metal_float32_choice_kernel(
            mx_module=mx, kernel_module=kernel
        )
        prebound.prewarm_b1()
        route = route_module.MetalFloat32ChoiceRoute.install(
            generation,
            expected_seed=seed,
            kernel_module=prebound,
            sampler=sampler,
        )
        generation.sample_from_distribution(distribution, rng)
        logits = FakeArray(np.linspace(-3.0, 3.0, 64, dtype=np.float32))
        proposal = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            logits, sampler
        )
        with pytest.raises(route_module.MetalChoiceRouteRNGError, match="identity"):
            generation.sample_from_distribution(proposal, np.random.default_rng(seed))
    finally:
        if "route" in locals():
            route.close()
        _deactivate(gates)


def test_second_support_and_wrong_proposal_identity_fail_closed():
    route_module, mx, kernel, generation, _, gates, seed = _fixture()
    _activate(gates)
    sampler = SimpleNamespace(top_k=20, top_p=0.95, temperature=1.0)
    logits = FakeArray(np.linspace(-3.0, 3.0, 64, dtype=np.float32))
    try:
        prebound = route_module.prebind_metal_float32_choice_kernel(
            mx_module=mx, kernel_module=kernel
        )
        prebound.prewarm_b1()
        route = route_module.MetalFloat32ChoiceRoute.install(
            generation, expected_seed=seed, kernel_module=prebound, sampler=sampler
        )
        proposal = generation.sparse_distribution_from_mlx_logits_relaxed_ties(
            logits, sampler
        )
        with pytest.raises(
            route_module.MetalChoiceRouteAssociationError, match="pending"
        ):
            generation.sparse_distribution_from_mlx_logits_relaxed_ties(logits, sampler)
        with pytest.raises(
            route_module.MetalChoiceRouteAssociationError, match="identity"
        ):
            generation.sample_from_distribution(
                FakeSparseDistribution(
                    proposal.token_ids, proposal.probs, proposal.vocab_size
                ),
                np.random.default_rng(seed),
            )
    finally:
        if "route" in locals():
            route.close()
        _deactivate(gates)


def test_finish_requires_closed_empty_successful_route_and_exact_stats_match():
    route_module, mx, kernel, generation, _, gates, seed = _fixture()
    _activate(gates)
    try:
        prebound = route_module.prebind_metal_float32_choice_kernel(
            mx_module=mx, kernel_module=kernel
        )
        prebound.prewarm_b1()
        route = route_module.MetalFloat32ChoiceRoute.install(
            generation,
            expected_seed=seed,
            kernel_module=prebound,
            sampler=SimpleNamespace(top_k=20, top_p=0.95, temperature=1.0),
        )
        with pytest.raises(
            route_module.MetalChoiceRouteAssociationError, match="closed"
        ):
            route.finish_receipt(stats=_stats(0))
        if "route" in locals():
            route.close()
        with pytest.raises(route_module.MetalChoiceRouteRNGError, match="request RNG"):
            route.finish_receipt(stats=_stats(0))
    finally:
        route.close()
        _deactivate(gates)


def test_source_contains_no_hot_environment_reads_or_fallback():
    import scripts.pr391_metal_choice_route as route_module

    source = inspect.getsource(route_module.MetalFloat32ChoiceRoute._sample_wrapper)
    assert "os.environ" not in source
    assert "try:" not in source
    assert "fallback" not in source.lower()
    assert "self._calls" not in source
    assert "self._passthrough" not in source
    assert ".random()" in source
    assert ".item()" in source
