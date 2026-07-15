from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


_BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "hy3_expert_fused_e1.py"


def _load_e1():
    return importlib.import_module("mtplx.hy3_expert_fused_e1")


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "hy3_expert_fused_e1_bench", _BENCHMARK
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e1_geometry_matches_real_hy3_q2_component_banks() -> None:
    module = _load_e1()

    assert module.HY3_E1_ROWS == tuple(range(1, 9))
    assert module.HY3_E1_TOP_K == 8
    assert module.HY3_E1_HIDDEN_SIZE == 4096
    assert module.HY3_E1_INTERMEDIATE_SIZE == 1536
    assert module.HY3_E1_BITS == 2
    assert module.HY3_E1_GROUP_SIZE == 64
    assert module.hy3_e1_component_bank_shapes(capacity=11) == {
        "gate_proj.weight": (11, 1536, 256),
        "gate_proj.scales": (11, 1536, 64),
        "gate_proj.biases": (11, 1536, 64),
        "up_proj.weight": (11, 1536, 256),
        "up_proj.scales": (11, 1536, 64),
        "up_proj.biases": (11, 1536, 64),
    }


def test_e1_candidate_catalog_exposes_the_requested_tuning_axes() -> None:
    module = _load_e1()

    catalog = module.hy3_e1_candidate_catalog()

    assert {candidate.output_n_tile for candidate in catalog} == {32, 64, 96, 128}
    assert {candidate.k_tile for candidate in catalog} == {32, 64, 128, 256}
    assert {candidate.simd_groups for candidate in catalog} == {2, 4, 6, 8}
    assert {candidate.silu_mode for candidate in catalog} == {"exact", "fast"}
    assert len(catalog) == 128


def test_e1_supported_matrix_retains_all_legal_variants() -> None:
    module = _load_e1()

    exact = module.hy3_e1_supported_candidates(silu_modes=("exact",))
    fast = module.hy3_e1_supported_candidates(silu_modes=("fast",))
    topology = {(candidate.output_n_tile, candidate.simd_groups) for candidate in exact}

    assert topology == {
        (32, 2),
        (32, 4),
        (32, 8),
        (64, 4),
        (64, 8),
        (96, 6),
        (96, 8),
        (128, 8),
    }
    assert len(exact) == 32
    assert len(fast) == 32
    assert all(candidate.supported for candidate in exact + fast)
    assert all(candidate.silu_mode == "exact" for candidate in exact)
    assert all(candidate.silu_mode == "fast" for candidate in fast)


def test_e1_candidate_resources_describe_packed_loads_and_activation_reuse() -> None:
    module = _load_e1()
    candidate = module.Hy3ExpertE1Candidate(
        output_n_tile=96,
        k_tile=128,
        simd_groups=6,
        silu_mode="exact",
    )

    resources = candidate.resources()

    assert candidate.supported
    assert candidate.name == "n96_k128_sg6_exact"
    assert resources == {
        "threads_per_threadgroup": 192,
        "outputs_per_simdgroup": 16,
        "threadgroup_activation_bytes": 256,
        "accumulator_floats_per_lane": 32,
        "packed_weight_vector": "uint2",
        "packed_values_per_vector": 32,
        "activation_reuse": "threadgroup gate+up/output-column reuse",
    }


def test_e1_candidate_reports_illegal_geometry_without_hiding_other_variants() -> None:
    module = _load_e1()
    unsupported = module.Hy3ExpertE1Candidate(
        output_n_tile=64,
        k_tile=64,
        simd_groups=2,
        silu_mode="exact",
    )

    assert unsupported.supported is False
    assert "outputs per SIMDgroup" in unsupported.unsupported_reason
    assert len(module.hy3_e1_supported_candidates()) == 64


def test_e1_exact_source_uses_q2_uint_vectors_and_threadgroup_activation() -> None:
    module = _load_e1()
    candidate = module.Hy3ExpertE1Candidate(32, 64, 2, "exact")

    source = module.render_hy3_expert_fused_e1_source(candidate)

    assert "threadgroup T activation_tile[K_TILE]" in source
    assert "uint2 gate_packed" in source
    assert "uint2 up_packed" in source
    assert "& 0x3u" in source
    assert "metal::exp" in source
    assert "fast::exp" not in source
    assert "T gate_value = T(gate_acc" in source
    assert "T up_value = T(up_acc" in source


def test_e1_fast_silu_is_source_and_policy_separated_as_an_ablation() -> None:
    module = _load_e1()
    candidate = module.Hy3ExpertE1Candidate(32, 64, 2, "fast")

    source = module.render_hy3_expert_fused_e1_source(candidate)

    assert "fast::exp" in source
    assert candidate.promotion_eligible is False
    assert candidate.classification == "fast_silu_ablation_only"


@pytest.mark.parametrize(
    (
        "silu_mode",
        "shape_exact",
        "dtype_bf16",
        "all_finite",
        "max_abs",
        "nrmse",
        "expected",
    ),
    (
        ("exact", True, True, True, 0.5, 0.02, True),
        ("exact", False, True, True, 0.0, 0.0, False),
        ("exact", True, False, True, 0.0, 0.0, False),
        ("exact", True, True, False, 0.0, 0.0, False),
        ("exact", True, True, True, 0.5001, 0.0, False),
        ("exact", True, True, True, 0.0, 0.0201, False),
        ("fast", True, True, True, 0.0, 0.0, False),
    ),
)
def test_e1_correctness_gate_is_exact_only_and_bf16_bounded(
    silu_mode: str,
    shape_exact: bool,
    dtype_bf16: bool,
    all_finite: bool,
    max_abs: float,
    nrmse: float,
    expected: bool,
) -> None:
    module = _load_e1()

    assert (
        module.hy3_e1_candidate_passes(
            silu_mode=silu_mode,
            output_shape_exact=shape_exact,
            output_dtype_bf16=dtype_bf16,
            all_finite=all_finite,
            max_abs_error=max_abs,
            normalized_rmse=nrmse,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("message", "phase", "expected"),
    (
        ("Metal compiler failed to build program", "first_eval", "compile"),
        ("threadgroup memory resource limit exceeded", "first_eval", "resource"),
        ("too many threads in threadgroup", "first_eval", "resource"),
        ("unexpected command buffer error", "first_eval", "dispatch"),
        ("builder raised", "kernel_build", "compile"),
    ),
)
def test_benchmark_classifies_compile_and_resource_failures_separately(
    message: str,
    phase: str,
    expected: str,
) -> None:
    benchmark = _load_benchmark()

    assert (
        benchmark.classify_candidate_failure(RuntimeError(message), phase=phase)
        == expected
    )


def test_benchmark_cli_can_filter_variants_or_preflight_without_dispatch() -> None:
    benchmark = _load_benchmark()

    args = benchmark._parser().parse_args(
        [
            "--model",
            "/tmp/hy3-q2",
            "--layer",
            "7",
            "--experts",
            "3,5,7,11,13,17,19,23",
            "--n-tiles",
            "32,96",
            "--k-tiles",
            "64,128",
            "--simd-groups",
            "2,6",
            "--silu-modes",
            "exact,fast",
            "--warmups",
            "2",
            "--repeats",
            "7",
            "--bootstrap",
            "123",
            "--compile-only",
            "--output-json",
            "/tmp/e1.json",
        ]
    )

    assert args.model == Path("/tmp/hy3-q2")
    assert args.layer == 7
    assert args.experts == (3, 5, 7, 11, 13, 17, 19, 23)
    assert args.n_tiles == (32, 96)
    assert args.k_tiles == (64, 128)
    assert args.simd_groups == (2, 6)
    assert args.silu_modes == ("exact", "fast")
    assert args.bootstrap_resamples == 123
    assert args.compile_only is True


def test_benchmark_filter_retains_multiple_matching_variants() -> None:
    benchmark = _load_benchmark()

    selected = benchmark.select_candidates(
        n_tiles=(32, 96),
        k_tiles=(64, 128),
        simd_groups=(2, 6),
        silu_modes=("exact", "fast"),
    )

    assert tuple(candidate.name for candidate in selected) == (
        "n32_k64_sg2_exact",
        "n32_k128_sg2_exact",
        "n96_k64_sg6_exact",
        "n96_k128_sg6_exact",
        "n32_k64_sg2_fast",
        "n32_k128_sg2_fast",
        "n96_k64_sg6_fast",
        "n96_k128_sg6_fast",
    )


def test_fast_silu_waits_for_its_matching_exact_candidate_to_pass() -> None:
    module = _load_e1()
    benchmark = _load_benchmark()
    exact = module.Hy3ExpertE1Candidate(96, 128, 6, "exact")
    fast = module.Hy3ExpertE1Candidate(96, 128, 6, "fast")

    assert not benchmark.fast_silu_prerequisite_met(fast, set())
    assert benchmark.fast_silu_prerequisite_met(
        fast,
        {benchmark.exact_candidate_key(exact)},
    )
