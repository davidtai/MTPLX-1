from __future__ import annotations

import importlib
import hashlib
import os
import sys
import types

import numpy as np
import pytest


softfloat = importlib.import_module(
    "mtplx.kernels.pr391_softfloat64_verifier_decision"
)


def _row(entries: list[tuple[int, float]], *, filler_base: int) -> tuple[np.ndarray, ...]:
    used = {token for token, _prob in entries}
    fillers: list[int] = []
    candidate = filler_base
    while len(entries) + len(fillers) < 20:
        if candidate not in used:
            fillers.append(candidate)
        candidate += 1
    ids = np.array([token for token, _prob in entries] + fillers, dtype=np.uint32)
    values = np.arange(20, 0, -1, dtype=np.float32)
    probs = np.array(
        [prob for _token, prob in entries] + [0.0] * len(fillers),
        dtype=np.float32,
    )
    return ids, values, probs


def _decision_fixture() -> tuple[np.ndarray, ...]:
    draft_tokens = np.array([11, 22, 33], dtype=np.uint32)
    draft_rows = [
        _row([(11, 0.5), (101, 0.5)], filler_base=1000),
        _row([(22, 0.5), (102, 0.5)], filler_base=1100),
        _row([(33, 0.5), (103, 0.5)], filler_base=1200),
    ]
    target_rows = [
        _row([(11, 0.25), (201, 0.75)], filler_base=2000),
        _row([(22, 0.25), (202, 0.75)], filler_base=2100),
        _row([(33, 0.25), (203, 0.75)], filler_base=2200),
        _row([(301, 0.25), (302, 0.75)], filler_base=2300),
    ]
    draft_ids = np.stack([row[0] for row in draft_rows])
    draft_values = np.stack([row[1] for row in draft_rows])
    draft_probs = np.stack([row[2] for row in draft_rows])
    target_ids = np.stack([row[0] for row in target_rows])
    target_values = np.stack([row[1] for row in target_rows])
    target_probs = np.stack([row[2] for row in target_rows])
    return (
        draft_tokens,
        draft_ids,
        draft_values,
        draft_probs,
        target_ids,
        target_values,
        target_probs,
    )


def test_reference_selector_matches_numpy_float64_boundary() -> None:
    ids = np.array([10, 20, 30], dtype=np.uint32)
    values = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    probs = np.array([0.61188227, 0.4436372, 0.22369266], dtype=np.float32)
    uniform = np.float64(7_432_132_623_180_982 / 2**53)

    selected, prepared_ids, prepared_bits = softfloat.reference_select_candidate_row(
        ids,
        values,
        probs,
        uniform,
        top_p=1.0,
    )

    assert selected == 30
    np.testing.assert_array_equal(prepared_ids, np.array([10, 20, 30], dtype=np.uint32))
    assert prepared_bits.dtype == np.dtype(np.uint64)


def test_metal_source_uses_vendored_binary64_arithmetic() -> None:
    source = softfloat.METAL_SOURCE

    assert softfloat.METAL_SOFTFLOAT_VERSION == "0.1.1"
    assert softfloat.METAL_SOFTFLOAT_COMMIT == "8b6c592e2e383040fe2778bed8dda7904df284b1"
    for operation in (
        "__softfloat64_fadd",
        "__softfloat64_fsub",
        "__softfloat64_fdiv",
        "__softfloat64_fle",
        "__softfloat64_flt",
    ):
        assert operation in source
    assert "uniform_bits[depth]" in source
    assert "device const float* uniforms" not in source


def test_vendored_metal_softfloat_is_the_complete_pinned_distribution() -> None:
    from mtplx.kernels._metal_softfloat64_v0_1_1 import METAL_SOFTFLOAT_SOURCE

    assert hashlib.sha256(METAL_SOFTFLOAT_SOURCE.encode()).hexdigest() == (
        "95b63a73c942ab9c49bce39af85e73338ac166ef35ca0cbbde961b219b0c2e26"
    )
    assert "SPDX-License-Identifier: BSD-3-Clause AND MIT" in METAL_SOFTFLOAT_SOURCE
    assert "kernel void __softfloat64_fadd_chain" in METAL_SOFTFLOAT_SOURCE


def test_reference_decision_rejects_and_samples_exact_residual() -> None:
    result = softfloat.reference_pr391_softfloat64_verifier_decision(
        *_decision_fixture(),
        np.array([0.75, 0.25, 0.25, 0.5], dtype=np.float64),
        np.array([999], dtype=np.uint32),
        stop_count=0,
        bonus_allowed=True,
    )

    assert tuple(int(value[0]) for value in result[:6]) == (
        0,
        0,
        201,
        softfloat.SELECTED_CORRECTION,
        1,
        2,
    )
    np.testing.assert_array_equal(
        result[6].view(np.float64),
        np.array([0.5, 0.0, 0.0], dtype=np.float64),
    )


def test_reference_acceptance_uses_once_normalized_batched_target_row() -> None:
    fixture = list(_decision_fixture())
    target_probs = fixture[6].copy()
    target_probs[0] = np.array(
        [
            0.012104385532438755,
            0.011631159111857414,
            0.07210824638605118,
            0.045167386531829834,
            0.05337757617235184,
            0.05441480502486229,
            0.0644269660115242,
            0.0025953599251806736,
            0.04392126202583313,
            0.013382192701101303,
            0.03632119670510292,
            0.08397100865840912,
            0.049552712589502335,
            0.0063706268556416035,
            0.04909956082701683,
            0.011740054003894329,
            0.06825051456689835,
            0.08579094707965851,
            0.08860597759485245,
            0.05625896900892258,
        ],
        dtype=np.float32,
    )
    fixture[6] = target_probs

    result = softfloat.reference_pr391_softfloat64_verifier_decision(
        *fixture,
        np.zeros(4, dtype=np.float64),
        np.array([], dtype=np.uint32),
        stop_count=0,
        bonus_allowed=False,
    )

    raw = target_probs[0].astype(np.float64)
    target_probability_once = raw[0] / np.sum(raw, dtype=np.float64)
    expected_acceptance = np.float64(target_probability_once / np.float64(0.5))
    assert result[6][0] == expected_acceptance.view(np.uint64)


def test_reference_decision_all_accepted_samples_bonus() -> None:
    result = softfloat.reference_pr391_softfloat64_verifier_decision(
        *_decision_fixture(),
        np.array([0.25, 0.25, 0.25, 0.5], dtype=np.float64),
        np.array([999], dtype=np.uint32),
        stop_count=0,
        bonus_allowed=True,
    )

    assert tuple(int(value[0]) for value in result[:6]) == (
        3,
        -1,
        302,
        softfloat.SELECTED_BONUS,
        1,
        4,
    )
    np.testing.assert_array_equal(
        result[6].view(np.float64),
        np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )


class _FakeKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> tuple[str, ...]:
        self.calls.append(kwargs)
        return tuple(f"output-{index}" for index in range(7))


def test_binder_installs_uint64_bit_abi_and_direct_hot_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, object] = {}
    fake_kernel = _FakeKernel()

    def metal_kernel(**kwargs: object) -> _FakeKernel:
        built.update(kwargs)
        return fake_kernel

    fake_core = types.ModuleType("mlx.core")
    fake_core.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    fake_core.uint32 = "uint32"
    fake_core.uint64 = "uint64"
    fake_core.int32 = "int32"
    fake_core.float32 = "float32"
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    apply = softfloat.bind_pr391_softfloat64_verifier_decision()
    inputs = [object() for _ in range(11)]

    assert apply(*inputs) == tuple(f"output-{index}" for index in range(7))
    assert built["input_names"] == [
        "draft_tokens",
        "draft_ids",
        "draft_values",
        "draft_probs",
        "target_ids",
        "target_values",
        "target_probs",
        "uniform_bits",
        "stop_ids",
        "stop_count",
        "bonus_allowed",
    ]
    assert built["output_names"][-1] == "accept_probability_bits"
    assert built["compile_options"] == {"math_mode": "safe"}
    assert fake_kernel.calls == [
        {
            "inputs": inputs,
            "grid": (1, 1, 1),
            "threadgroup": (1, 1, 1),
            "output_shapes": [(1,), (1,), (1,), (1,), (1,), (1,), (3,)],
            "output_dtypes": [
                "uint32",
                "int32",
                "uint32",
                "uint32",
                "uint32",
                "uint32",
                "uint64",
            ],
        }
    ]


def test_controller_source_carries_numpy_reductions_and_side_right() -> None:
    source = softfloat.METAL_CONTROLLER_SOURCE

    assert "numpy_pairwise_sum_f64" in source
    assert "RESIDUAL_CAPACITY = 2 * K" in source
    assert "prepare_candidate_row" in source
    assert "prepare_residual_row" in source
    assert "__softfloat64_flt(uniform, boundary)" in source
    assert "__softfloat64_fle(uniform_bits[depth], accept_probability)" in source


def test_guarded_metal_decision_matches_float64_reference() -> None:
    if not (
        os.environ.get("MTPLX_GUARD_ATTEST_FD")
        and os.environ.get("MTPLX_GUARD_ATTEST_NONCE")
    ):
        pytest.skip("real MLX/Metal execution requires the canonical GPU guard")

    import mlx.core as mx

    fixture = _decision_fixture()
    uniforms = np.array([0.75, 0.25, 0.25, 0.5], dtype=np.float64)
    stop_ids = np.array([999], dtype=np.uint32)
    expected = softfloat.reference_pr391_softfloat64_verifier_decision(
        *fixture,
        uniforms,
        stop_ids,
        stop_count=0,
        bonus_allowed=True,
    )
    apply = softfloat.bind_pr391_softfloat64_verifier_decision()
    device_inputs = [
        mx.array(fixture[0], dtype=mx.uint32),
        mx.array(fixture[1], dtype=mx.uint32),
        mx.array(fixture[2], dtype=mx.float32),
        mx.array(fixture[3], dtype=mx.float32),
        mx.array(fixture[4], dtype=mx.uint32),
        mx.array(fixture[5], dtype=mx.float32),
        mx.array(fixture[6], dtype=mx.float32),
        mx.array(np.ascontiguousarray(uniforms).view(np.uint64), dtype=mx.uint64),
        mx.array(stop_ids, dtype=mx.uint32),
        mx.array(np.array([0], dtype=np.uint32), dtype=mx.uint32),
        mx.array(np.array([1], dtype=np.uint32), dtype=mx.uint32),
    ]
    observed = apply(*device_inputs)
    mx.eval(*observed)

    for actual, reference in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), reference)


class _FakeSelectorKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> tuple[str, ...]:
        self.calls.append(kwargs)
        return ("selected", "raw-ids", "raw-values", "raw-probs")


def test_selector_binder_keeps_raw_candidates_and_uint64_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, object] = {}
    fake_kernel = _FakeSelectorKernel()

    def metal_kernel(**kwargs: object) -> _FakeSelectorKernel:
        built.update(kwargs)
        return fake_kernel

    fake_core = types.ModuleType("mlx.core")
    fake_core.fast = types.SimpleNamespace(metal_kernel=metal_kernel)
    fake_core.uint32 = "uint32"
    fake_core.uint64 = "uint64"
    fake_core.float32 = "float32"
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    selector = softfloat.bind_pr391_softfloat64_candidate_selector()
    ids = types.SimpleNamespace(shape=(1, 20))
    values = object()
    probs = object()
    uniform_bits = object()

    assert selector(ids, values, probs, uniform_bits) == (
        "selected",
        "raw-ids",
        "raw-values",
        "raw-probs",
    )
    assert built["input_names"] == [
        "candidate_ids",
        "candidate_values",
        "candidate_probs",
        "uniform_bits",
    ]
    assert fake_kernel.calls[0]["output_dtypes"] == [
        "uint32",
        "uint32",
        "float32",
        "float32",
    ]
    assert "prepare_candidate_row" in str(built["header"])
    assert "sample_prepared_row" in str(built["source"])


def test_guarded_metal_selector_matches_float64_reference() -> None:
    if not (
        os.environ.get("MTPLX_GUARD_ATTEST_FD")
        and os.environ.get("MTPLX_GUARD_ATTEST_NONCE")
    ):
        pytest.skip("real MLX/Metal execution requires the canonical GPU guard")

    import mlx.core as mx

    ids, values, probs = _row(
        [(10, 0.61188227), (20, 0.4436372), (30, 0.22369266)],
        filler_base=1000,
    )
    uniform = np.array([0.625], dtype=np.float64)
    expected, _prepared_ids, _prepared_bits = softfloat.reference_select_candidate_row(
        ids,
        values,
        probs,
        uniform[0],
        top_p=0.95,
    )
    selector = softfloat.bind_pr391_softfloat64_candidate_selector()
    observed = selector(
        mx.array(ids.reshape(1, 20), dtype=mx.uint32),
        mx.array(values.reshape(1, 20), dtype=mx.float32),
        mx.array(probs.reshape(1, 20), dtype=mx.float32),
        mx.array(np.ascontiguousarray(uniform).view(np.uint64), dtype=mx.uint64),
    )
    mx.eval(*observed)

    assert int(np.asarray(observed[0])[0]) == expected
    np.testing.assert_array_equal(np.asarray(observed[1]), ids.reshape(1, 20))
    np.testing.assert_array_equal(np.asarray(observed[2]), values.reshape(1, 20))
    np.testing.assert_array_equal(np.asarray(observed[3]), probs.reshape(1, 20))


def test_guarded_metal_selector_matches_production_single_row_dispatch() -> None:
    if not (
        os.environ.get("MTPLX_GUARD_ATTEST_FD")
        and os.environ.get("MTPLX_GUARD_ATTEST_NONCE")
    ):
        pytest.skip("real MLX/Metal execution requires the canonical GPU guard")

    import mlx.core as mx

    rows = 256
    rng = np.random.Generator(np.random.PCG64(20260829))
    ids = (
        np.arange(rows, dtype=np.uint32)[:, None] * np.uint32(20)
        + np.arange(20, dtype=np.uint32)[None, :]
    )
    values = rng.standard_normal((rows, 20), dtype=np.float32)
    shifted = values - np.max(values, axis=1, keepdims=True)
    probs = np.exp(shifted).astype(np.float32)
    probs *= (
        np.float32(0.94)
        / np.sum(probs, axis=1, dtype=np.float32, keepdims=True)
    )
    uniforms = rng.random(rows)
    expected = np.array(
        [
            softfloat.reference_select_candidate_row(
                ids[row],
                values[row],
                probs[row],
                uniforms[row],
                top_p=0.95,
            )[0]
            for row in range(rows)
        ],
        dtype=np.uint32,
    )

    selector = softfloat.bind_pr391_softfloat64_candidate_selector()
    device_uniforms = mx.array(
        np.ascontiguousarray(uniforms).view(np.uint64), dtype=mx.uint64
    )
    selected_rows = [
        selector(
            mx.array(ids[row : row + 1], dtype=mx.uint32),
            mx.array(values[row : row + 1], dtype=mx.float32),
            mx.array(probs[row : row + 1], dtype=mx.float32),
            device_uniforms[row : row + 1],
        )[0]
        for row in range(rows)
    ]
    mx.eval(*selected_rows)
    observed = np.array(
        [int(np.asarray(selected)[0]) for selected in selected_rows],
        dtype=np.uint32,
    )

    np.testing.assert_array_equal(observed, expected)


def test_guarded_metal_selector_row83_preparation_bits() -> None:
    if not (
        os.environ.get("MTPLX_GUARD_ATTEST_FD")
        and os.environ.get("MTPLX_GUARD_ATTEST_NONCE")
    ):
        pytest.skip("real MLX/Metal execution requires the canonical GPU guard")

    import mlx.core as mx

    rows = 4096
    rng = np.random.Generator(np.random.PCG64(20260829))
    ids = (
        np.arange(rows, dtype=np.uint32)[:, None] * np.uint32(20)
        + np.arange(20, dtype=np.uint32)[None, :]
    )
    values = rng.standard_normal((rows, 20), dtype=np.float32)
    shifted = values - np.max(values, axis=1, keepdims=True)
    probs = np.exp(shifted).astype(np.float32)
    probs *= (
        np.float32(0.94)
        / np.sum(probs, axis=1, dtype=np.float32, keepdims=True)
    )
    uniforms = rng.random(rows)
    row = 83
    expected_ids, expected_probs = softfloat._prepare_candidate_row(
        ids[row], values[row], probs[row], top_p=0.95
    )
    expected_cdf = np.cumsum(expected_probs, dtype=np.float64)
    expected_cdf /= np.sum(expected_probs, dtype=np.float64)

    kernel = mx.fast.metal_kernel(
        name="mtplx_pr391_softfloat64_selector_row_diagnostic",
        input_names=["candidate_ids", "candidate_values", "candidate_probs"],
        output_names=["prepared_ids_out", "prepared_probs_out", "cdf_out", "count_out"],
        header=softfloat.METAL_SOFTFLOAT_SOURCE + "\n" + softfloat.METAL_HELPERS,
        source=r"""
            uint prepared_ids[K];
            ulong prepared_probs[K];
            uint prepared_count;
            prepare_candidate_row(
                candidate_ids, candidate_values, candidate_probs, 0u,
                prepared_ids, prepared_probs, prepared_count
            );
            ulong total = numpy_pairwise_sum_f64(prepared_probs, prepared_count);
            ulong cumulative = F64_ZERO;
            for (uint index = 0u; index < K; ++index) {
                prepared_ids_out[index] = index < prepared_count ? prepared_ids[index] : 0u;
                prepared_probs_out[index] = index < prepared_count ? prepared_probs[index] : F64_ZERO;
                if (index < prepared_count) {
                    cumulative = __softfloat64_fadd(cumulative, prepared_probs[index], 0u);
                    cdf_out[index] = __softfloat64_fdiv(cumulative, total, 0u);
                } else {
                    cdf_out[index] = F64_ZERO;
                }
            }
            count_out[0] = prepared_count;
        """,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )
    observed = kernel(
        inputs=[
            mx.array(ids[row], dtype=mx.uint32),
            mx.array(values[row], dtype=mx.float32),
            mx.array(probs[row], dtype=mx.float32),
        ],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(20,), (20,), (20,), (1,)],
        output_dtypes=[mx.uint32, mx.uint64, mx.uint64, mx.uint32],
    )
    mx.eval(*observed)
    actual_ids = np.asarray(observed[0])
    actual_probs = np.asarray(observed[1])
    actual_cdf = np.asarray(observed[2])
    actual_count = int(np.asarray(observed[3])[0])

    assert actual_count == len(expected_ids)
    np.testing.assert_array_equal(actual_ids, expected_ids)
    for label, actual, expected in (
        ("probability", actual_probs, expected_probs.view(np.uint64)),
        ("cdf", actual_cdf, expected_cdf.view(np.uint64)),
    ):
        mismatch = np.flatnonzero(actual != expected)
        assert mismatch.size == 0, (
            f"{label} bit mismatch at {int(mismatch[0])}: "
            f"metal={int(actual[mismatch[0]]):#018x} "
            f"numpy={int(expected[mismatch[0]]):#018x} "
            f"uniform={int(np.float64(uniforms[row]).view(np.uint64)):#018x}"
        )
