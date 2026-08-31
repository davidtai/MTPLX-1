from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


FIXED_COUNTERS = {
    "drafted_tokens": 1,
    "accepted_drafts": 1,
    "verify_calls": 1,
    "correction_tokens": 0,
    "bonus_tokens": 0,
}


def _counters(drafted_tokens: int) -> dict[str, int]:
    return {
        **FIXED_COUNTERS,
        "drafted_tokens": drafted_tokens,
        "accepted_drafts": drafted_tokens,
        "verify_calls": int(drafted_tokens > 0),
    }


class FakeMX:
    float32 = np.float32

    @staticmethod
    def argpartition(values, *, kth, axis):
        return np.argpartition(values, kth=kth, axis=axis)

    @staticmethod
    def take_along_axis(values, indices, *, axis):
        return np.take_along_axis(values, indices, axis=axis)

    @staticmethod
    def logsumexp(values, *, axis, keepdims):
        maximum = np.max(values, axis=axis, keepdims=True)
        result = maximum + np.log(
            np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
        )
        return result if keepdims else np.squeeze(result, axis=axis)

    exp = staticmethod(np.exp)

    @staticmethod
    def eval(*values):
        del values


@dataclass(frozen=True)
class FakeSparseDistribution:
    token_ids: np.ndarray
    probs: np.ndarray
    vocab_size: int


def fake_support(logits, config):
    del config
    ids = np.asarray([[4, 1, 3]], dtype=np.int64)
    probs = np.asarray([[0.6, 0.3, 0.1]], dtype=np.float64)
    return ids, probs, int(np.asarray(logits).shape[-1])


def fake_sample(distribution, rng):
    return int(rng.choice(distribution.token_ids, p=distribution.probs))


def _modules(*, support=fake_support, sample=fake_sample):
    return (
        SimpleNamespace(
            _device_serial_support_arrays_relaxed_ties=support,
            mx=FakeMX,
        ),
        SimpleNamespace(sample_from_distribution=sample),
    )


def _source_hash(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _install_for_fake(fast, generation, *, support=fake_support):
    from unittest.mock import patch

    import scripts.pr391_capture_choice_rows as capture

    with patch.object(
        capture,
        "REVIEWED_SUPPORT_SOURCE_SHA256",
        _source_hash(support),
    ):
        return capture.ChoiceRowCapture.install(fast, generation)


def _config(*, top_k=3, top_p=0.95):
    return SimpleNamespace(temperature=1.0, top_k=top_k, top_p=top_p)


def _matching_distribution() -> FakeSparseDistribution:
    raw_ids = np.asarray([4, 1, 3], dtype=np.int64)
    raw_probs = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)
    order = np.argsort(raw_ids)
    normalized = raw_probs[order] / raw_probs.sum()
    normalized = normalized / normalized.sum()
    return FakeSparseDistribution(
        token_ids=raw_ids[order],
        probs=normalized,
        vocab_size=6,
    )


def _distribution_from_support(result) -> FakeSparseDistribution:
    token_rows, prob_rows, vocab_size = result
    token_ids = np.asarray(token_rows[0], dtype=np.int64)
    probs = np.asarray(prob_rows[0], dtype=np.float64)
    keep = probs > 0.0
    token_ids = token_ids[keep]
    probs = probs[keep]
    order = np.argsort(token_ids)
    token_ids = token_ids[order]
    probs = probs[order] / np.sum(probs)
    probs = probs / np.sum(probs)
    return FakeSparseDistribution(token_ids, probs, int(vocab_size))


def test_import_has_no_mlx_or_mtplx_dependency():
    import scripts.pr391_capture_choice_rows as capture

    assert "mlx" not in capture.__dict__
    assert "mtplx" not in capture.__dict__


def test_source_hash_gate_rejects_before_patching():
    from scripts.pr391_capture_choice_rows import ChoiceRowCapture, SourceHashMismatch

    fast, generation = _modules()
    original_support = fast._device_serial_support_arrays_relaxed_ties
    original_sample = generation.sample_from_distribution

    with pytest.raises(SourceHashMismatch, match="support source SHA-256"):
        ChoiceRowCapture.install(fast, generation)

    assert fast._device_serial_support_arrays_relaxed_ties is original_support
    assert generation.sample_from_distribution is original_sample


def test_runtime_callers_cannot_override_reviewed_source_hash():
    from scripts.pr391_capture_choice_rows import ChoiceRowCapture

    assert (
        "expected_support_source_sha256"
        not in inspect.signature(ChoiceRowCapture.install).parameters
    )
    assert (
        "expected_support_source_sha256"
        not in inspect.signature(ChoiceRowCapture).parameters
    )

    fast, generation = _modules()
    with pytest.raises(TypeError, match="expected_support_source_sha256"):
        ChoiceRowCapture.install(
            fast,
            generation,
            expected_support_source_sha256=_source_hash(fake_support),
        )


def test_reviewed_production_support_source_hash_is_pinned():
    from scripts.pr391_capture_choice_rows import REVIEWED_SUPPORT_SOURCE_SHA256

    path = Path(__file__).parents[1] / "mtplx" / "fast_sampling.py"
    source = path.read_text()
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_device_serial_support_arrays_relaxed_ties"
    )
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[node.lineno - 1 : node.end_lineno])

    assert REVIEWED_SUPPORT_SOURCE_SHA256 == (
        "31d5f2d5b4be4dd83dff05b3ae65f70482b5022747d9755c7264f187c9fa8db3"
    )
    assert hashlib.sha256(function_source.encode()).hexdigest() == (
        REVIEWED_SUPPORT_SOURCE_SHA256
    )


def test_context_capture_peeks_uniform_without_extra_rng_advance_and_restores(tmp_path):

    fast, generation = _modules()
    original_support = fast._device_serial_support_arrays_relaxed_ties
    original_sample = generation.sample_from_distribution
    rng = np.random.Generator(np.random.PCG64(20260829))
    reference = np.random.Generator(np.random.PCG64(20260829))
    expected_uniform = float(reference.random())
    logits = np.asarray([[0.0, 1.25, -2.0, 0.75, 2.5, -9.0]], dtype=np.float32)
    config = _config()

    with _install_for_fake(fast, generation) as recorder:
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        distribution = _distribution_from_support(returned)
        cdf = np.cumsum(distribution.probs, dtype=np.float64)
        cdf /= cdf[-1]
        expected_index = int(np.searchsorted(cdf, expected_uniform, side="right"))
        expected_token = int(distribution.token_ids[expected_index])
        selected = generation.sample_from_distribution(distribution, rng)
        output = tmp_path / "rows.npz"
        recorder.finalize(
            output,
            metadata={"seeds": [20260829], "workload": "16k-prefill-1k-output"},
            expected_rows=1,
            observed_counters=FIXED_COUNTERS,
            expected_counters=FIXED_COUNTERS,
        )

    assert selected == expected_token
    assert rng.bit_generator.state == reference.bit_generator.state
    assert fast._device_serial_support_arrays_relaxed_ties is original_support
    assert generation.sample_from_distribution is original_sample

    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "candidate_ids",
            "candidate_values",
            "candidate_probs",
            "uniforms",
            "selected_tokens",
            "rng_pre_sha256",
            "rng_post_sha256",
            "metadata_json",
        }
        expected_ids = np.argpartition(-logits, kth=2, axis=-1)[:, :3]
        expected_values = np.take_along_axis(logits, expected_ids, axis=1)
        log_total = FakeMX.logsumexp(logits, axis=-1, keepdims=True)
        expected_probs = np.exp(expected_values - log_total)
        np.testing.assert_array_equal(saved["candidate_ids"], expected_ids)
        np.testing.assert_array_equal(
            saved["candidate_values"],
            expected_values.astype(np.float32),
        )
        np.testing.assert_array_equal(
            saved["candidate_probs"],
            expected_probs.astype(np.float32),
        )
        np.testing.assert_array_equal(saved["uniforms"], [expected_uniform])
        np.testing.assert_array_equal(saved["selected_tokens"], [expected_token])
        assert saved["rng_pre_sha256"].dtype == np.dtype("S64")
        assert saved["rng_post_sha256"].dtype == np.dtype("S64")
        assert all(saved[name].dtype != object for name in saved.files)
        metadata = json.loads(str(saved["metadata_json"].item()))
        assert metadata["row_count"] == 1
        assert metadata["support_width"] == 3
        assert metadata["seeds"] == [20260829]
        assert metadata["observed_counters"] == FIXED_COUNTERS
        assert metadata["expected_counters"] == FIXED_COUNTERS


def test_fifo_association_is_fail_closed():
    from scripts.pr391_capture_choice_rows import (
        CaptureAssociationError,
    )

    fast, generation = _modules()
    logits = np.arange(6, dtype=np.float32)[None, :]
    config = _config()
    wrong = FakeSparseDistribution(
        token_ids=np.asarray([1, 3, 5]),
        probs=np.asarray([0.3, 0.1, 0.6]),
        vocab_size=6,
    )

    with _install_for_fake(fast, generation):
        fast._device_serial_support_arrays_relaxed_ties(logits, config)
        with pytest.raises(CaptureAssociationError, match="pending support row"):
            generation.sample_from_distribution(wrong, np.random.default_rng(7))


def test_finalization_enforces_pending_rows_count_and_counters(tmp_path):
    from scripts.pr391_capture_choice_rows import (
        CaptureFinalizationError,
    )

    fast, generation = _modules()
    logits = np.arange(6, dtype=np.float32)[None, :]
    config = _config()

    with _install_for_fake(fast, generation) as recorder:
        fast._device_serial_support_arrays_relaxed_ties(logits, config)
        with pytest.raises(CaptureFinalizationError, match="1 pending"):
            recorder.finalize(tmp_path / "pending.npz", metadata={}, expected_rows=0)

    fast, generation = _modules()
    with _install_for_fake(fast, generation) as recorder:
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        generation.sample_from_distribution(
            _distribution_from_support(returned), np.random.default_rng(7)
        )
        with pytest.raises(CaptureFinalizationError, match="expected 3338 rows"):
            recorder.finalize(tmp_path / "count.npz", metadata={}, expected_rows=3338)
        with pytest.raises(CaptureFinalizationError, match="counter mismatch"):
            recorder.finalize(
                tmp_path / "counter.npz",
                metadata={},
                expected_rows=1,
                observed_counters=FIXED_COUNTERS,
                expected_counters={**FIXED_COUNTERS, "bonus_tokens": 2},
            )
        with pytest.raises(
            CaptureFinalizationError, match="drafted_tokens.*captured 1"
        ):
            wrong_row_count = _counters(4)
            recorder.finalize(
                tmp_path / "drafted-row-count.npz",
                metadata={},
                expected_rows=1,
                observed_counters=wrong_row_count,
                expected_counters=wrong_row_count,
            )


def test_capture_requires_pc64_and_exactly_one_rng_draw():
    from scripts.pr391_capture_choice_rows import CaptureRNGError

    logits = np.arange(6, dtype=np.float32)[None, :]
    config = _config()

    fast, generation = _modules()
    with _install_for_fake(fast, generation):
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        with pytest.raises(CaptureRNGError, match="PCG64"):
            generation.sample_from_distribution(
                _distribution_from_support(returned),
                np.random.Generator(np.random.Philox(3)),
            )

    def two_draw_sample(distribution, rng):
        rng.random()
        return int(rng.choice(distribution.token_ids, p=distribution.probs))

    fast, generation = _modules(sample=two_draw_sample)
    with _install_for_fake(fast, generation):
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        with pytest.raises(CaptureRNGError, match="exactly one PCG64 random draw"):
            generation.sample_from_distribution(
                _distribution_from_support(returned), np.random.default_rng(3)
            )


def test_batched_support_rows_are_queued_and_consumed_fifo(tmp_path):

    def batched_support(logits, config):
        del logits, config
        return (
            np.asarray([[4, 1], [2, 0]], dtype=np.int64),
            np.asarray([[0.75, 0.25], [0.4, 0.6]], dtype=np.float64),
            5,
        )

    fast, generation = _modules(support=batched_support)
    logits = np.asarray(
        [[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    with _install_for_fake(fast, generation, support=batched_support) as recorder:
        returned = fast._device_serial_support_arrays_relaxed_ties(
            logits, _config(top_k=2)
        )
        token_rows, prob_rows, vocab_size = returned
        distributions = [
            _distribution_from_support(
                (
                    token_rows[index : index + 1],
                    prob_rows[index : index + 1],
                    vocab_size,
                )
            )
            for index in range(token_rows.shape[0])
        ]
        for distribution in distributions:
            generation.sample_from_distribution(distribution, np.random.default_rng(11))
        output = recorder.finalize(
            tmp_path / "batched.npz",
            metadata={},
            expected_rows=2,
            observed_counters=_counters(2),
            expected_counters=_counters(2),
        )

    with np.load(output, allow_pickle=False) as saved:
        expected_ids = np.argpartition(-logits, kth=1, axis=-1)[:, :2]
        expected_values = np.take_along_axis(logits, expected_ids, axis=1)
        np.testing.assert_array_equal(saved["candidate_ids"], expected_ids)
        np.testing.assert_array_equal(saved["candidate_values"], expected_values)


def test_capture_records_raw_pre_top_p_mass_before_return_masking(tmp_path):

    def masked_support(logits, config):
        del logits, config
        return (
            np.asarray([[2, 1, 0]], dtype=np.int64),
            np.asarray([[0.6, 0.3, 0.0]], dtype=np.float64),
            4,
        )

    logits = np.log(np.asarray([[0.1, 0.3, 0.6, 1e-10]], dtype=np.float32)).astype(
        np.float32
    )
    config = SimpleNamespace(temperature=1.0, top_k=3, top_p=0.75)
    fast, generation = _modules(support=masked_support)

    with _install_for_fake(fast, generation, support=masked_support) as recorder:
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        excluded_index = int(np.flatnonzero(returned[0][0] == 0)[0])
        assert returned[1][0, excluded_index] == 0.0
        generation.sample_from_distribution(
            _distribution_from_support(returned), np.random.default_rng(17)
        )
        output = recorder.finalize(
            tmp_path / "raw-pre-top-p.npz",
            metadata={},
            expected_rows=1,
            observed_counters=FIXED_COUNTERS,
            expected_counters=FIXED_COUNTERS,
        )

    with np.load(output, allow_pickle=False) as saved:
        captured_index = int(np.flatnonzero(saved["candidate_ids"][0] == 0)[0])
        assert saved["candidate_probs"][0, captured_index] > 0.0


def test_capture_does_not_materialize_full_logits(tmp_path):

    class DeviceOnlyLogits:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        @property
        def shape(self):
            return self.values.shape

        def reshape(self, *shape):
            return type(self)(self.values.reshape(*shape))

        def astype(self, dtype):
            return type(self)(self.values.astype(dtype))

        def __mul__(self, scalar):
            return self.values * scalar

        def __array__(self, dtype=None, copy=None):
            del dtype, copy
            raise AssertionError("full logits must not be materialized on the host")

    def opaque_support(logits, config):
        del logits, config
        return (
            np.asarray([[2, 1, 0]], dtype=np.int64),
            np.asarray([[0.6, 0.3, 0.0]], dtype=np.float64),
            4,
        )

    logits = DeviceOnlyLogits(
        np.log(np.asarray([[0.1, 0.3, 0.6, 1e-10]], dtype=np.float32))
    )
    config = SimpleNamespace(temperature=1.0, top_k=3, top_p=0.75)
    fast, generation = _modules(support=opaque_support)
    with _install_for_fake(fast, generation, support=opaque_support) as recorder:
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        generation.sample_from_distribution(
            _distribution_from_support(returned), np.random.default_rng(17)
        )
        recorder.finalize(
            tmp_path / "no-full-logits.npz",
            metadata={},
            expected_rows=1,
            observed_counters=FIXED_COUNTERS,
            expected_counters=FIXED_COUNTERS,
        )


def test_finalize_requires_and_persists_complete_fixed_counters(tmp_path):
    from scripts.pr391_capture_choice_rows import (
        CaptureFinalizationError,
    )

    fast, generation = _modules()
    with _install_for_fake(fast, generation) as recorder:
        with pytest.raises(CaptureFinalizationError, match="complete counter schema"):
            recorder.finalize(
                tmp_path / "missing-counters.npz",
                metadata={},
                expected_rows=0,
                observed_counters={"drafted_tokens": 0},
                expected_counters={"drafted_tokens": 0},
            )

    fast, generation = _modules()
    with _install_for_fake(fast, generation) as recorder:
        zero_counters = _counters(0)
        output = recorder.finalize(
            tmp_path / "complete-counters.npz",
            metadata={},
            expected_rows=0,
            observed_counters=zero_counters,
            expected_counters=zero_counters,
        )

    with np.load(output, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
    assert metadata["observed_counters"] == zero_counters
    assert metadata["expected_counters"] == zero_counters


def test_capture_rejects_one_draw_sampler_returning_wrong_token():
    from scripts.pr391_capture_choice_rows import CaptureRNGError

    def wrong_token_sample(distribution, rng):
        del distribution
        rng.random()
        return 999

    fast, generation = _modules(sample=wrong_token_sample)
    logits = np.arange(6, dtype=np.float32)[None, :]
    config = SimpleNamespace(temperature=1.0, top_k=3, top_p=0.95)
    with _install_for_fake(fast, generation):
        returned = fast._device_serial_support_arrays_relaxed_ties(logits, config)
        with pytest.raises(CaptureRNGError, match="selected token.*side-right"):
            generation.sample_from_distribution(
                _distribution_from_support(returned), np.random.default_rng(19)
            )
