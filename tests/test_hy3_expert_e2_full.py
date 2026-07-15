from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "hy3_expert_e2_full.py"


def _module():
    from mtplx import hy3_expert_e2_full

    return hy3_expert_e2_full


def _benchmark():
    spec = importlib.util.spec_from_file_location(
        "hy3_expert_e2_full_bench", _BENCHMARK
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2_geometry_matches_real_hy3_q2_full_expert_bank() -> None:
    module = _module()

    assert module.HY3_E2_ROWS == tuple(range(1, 9))
    assert module.HY3_E2_TOP_K == 8
    assert module.HY3_E2_HIDDEN_SIZE == 4096
    assert module.HY3_E2_INTERMEDIATE_SIZE == 1536
    assert module.HY3_E2_BITS == 2
    assert module.HY3_E2_GROUP_SIZE == 64
    assert module.hy3_e2_down_bank_shapes(capacity=11) == {
        "down_proj.weight": (11, 4096, 96),
        "down_proj.scales": (11, 4096, 24),
        "down_proj.biases": (11, 4096, 24),
    }


def test_e2_catalog_is_exhaustive_and_keeps_fast_math_separate() -> None:
    module = _module()

    catalog = module.hy3_e2_candidate_catalog()

    assert len(catalog) == 96
    assert {candidate.output_n_tile for candidate in catalog} == {32, 64, 128}
    assert {candidate.k_tile for candidate in catalog} == {32, 64, 96, 192}
    assert {candidate.simd_groups for candidate in catalog} == {2, 4, 6, 8}
    assert {candidate.activation_mode for candidate in catalog} == {"exact", "fast"}
    exact = module.hy3_e2_supported_candidates(activation_modes=("exact",))
    fast = module.hy3_e2_supported_candidates(activation_modes=("fast",))
    assert len(exact) == 24
    assert len(fast) == 24
    assert all(candidate.promotion_eligible for candidate in exact)
    assert all(not candidate.promotion_eligible for candidate in fast)


def test_e2_resources_account_for_removed_intermediate_and_q2_loads() -> None:
    module = _module()
    candidate = module.Hy3ExpertE2Candidate(64, 96, 4, "exact")

    assert candidate.supported
    assert candidate.name == "n64_k96_sg4_exact"
    assert candidate.resources(rows=4) == {
        "threads_per_threadgroup": 128,
        "outputs_per_simdgroup": 16,
        "threadgroup_activation_bytes": 192,
        "accumulator_floats_per_lane": 16,
        "packed_weight_vector": "uint2",
        "packed_values_per_vector": 32,
        "stock_down_output_bytes": 262144,
        "fused_down_output_bytes": 32768,
        "avoided_intermediate_bytes": 229376,
        "stock_intermediate_write_read_bytes": 524288,
        "estimated_saved_write_read_bytes": 524288,
        "down_weight_parameter_bytes_per_token": 15728640,
        "candidate_activation_load_bytes_per_token": 1572864,
        "threadgroups_per_token": 64,
        "expert_order": "router slot 0 through 7",
    }


def test_e2_source_preserves_bf16_boundaries_and_router_order() -> None:
    module = _module()
    candidate = module.Hy3ExpertE2Candidate(64, 96, 4, "exact")

    source = module.render_hy3_expert_e2_source(candidate, diagnostic=True)

    assert "for (int expert = 0; expert < TOP_K; ++expert)" in source
    assert "T expert_value = T(down_acc[output]);" in source
    assert "T weighted_value = T(expert_value * route_weights" in source
    assert "routed_acc[output] = T(routed_acc[output] + weighted_value);" in source
    assert "per_expert_values" in source
    assert "uint2 packed" in source
    assert "& 0x3u" in source


def test_fast_swiglu_source_and_policy_are_explicitly_nonpromotable() -> None:
    module = _module()
    fast = module.Hy3ExpertE2Candidate(32, 64, 2, "fast")
    exact = module.Hy3ExpertE2Candidate(32, 64, 2, "exact")

    assert "fast::exp" in module.render_hy3_e2_swiglu_source("fast")
    assert "fast::exp" not in module.render_hy3_e2_swiglu_source("exact")
    assert fast.classification == "fast_swiglu_ablation_only"
    assert fast.promotion_eligible is False
    assert exact.classification == "exact_e2_candidate"


@pytest.mark.parametrize(
    ("activation_mode", "per_expert_max", "final_max", "nrmse", "expected"),
    (
        ("exact", 0.5, 0.5, 0.02, True),
        ("exact", 0.5001, 0.0, 0.0, False),
        ("exact", 0.0, 0.5001, 0.0, False),
        ("exact", 0.0, 0.0, 0.0201, False),
        ("fast", 0.0, 0.0, 0.0, False),
    ),
)
def test_e2_correctness_gate_requires_exact_per_expert_and_final_boundaries(
    activation_mode: str,
    per_expert_max: float,
    final_max: float,
    nrmse: float,
    expected: bool,
) -> None:
    module = _module()

    assert (
        module.hy3_e2_candidate_passes(
            activation_mode=activation_mode,
            per_expert_shape_exact=True,
            final_shape_exact=True,
            dtype_bf16=True,
            all_finite=True,
            per_expert_max_abs_error=per_expert_max,
            final_max_abs_error=final_max,
            final_normalized_rmse=nrmse,
        )
        is expected
    )


def test_adversarial_reference_reduction_is_order_sensitive_and_bf16_bounded() -> None:
    module = _module()
    expert_values = np.array(
        [[[512.0], [-512.0], [1.0], [0.5], [-0.5], [0.25], [-0.25], [0.125]]],
        dtype=np.float32,
    )
    weights = np.array(
        [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]],
        dtype=np.float32,
    )

    forward = module.simulate_hy3_bf16_route_reduce(expert_values, weights)
    reverse = module.simulate_hy3_bf16_route_reduce(
        expert_values[:, ::-1], weights[:, ::-1]
    )

    assert forward.dtype == np.float32
    assert reverse.dtype == np.float32
    assert not np.array_equal(forward, reverse)


@pytest.mark.parametrize(
    ("per_expert_equal", "route_reduce_equal", "expected"),
    (
        (True, True, "bit_exact"),
        (False, True, "down_qmm_reduction_order_only"),
        (True, False, "router_reduction_order_only"),
        (False, False, "down_qmm_and_router_reduction_order"),
    ),
)
def test_stage_attribution_keeps_down_and_route_reduction_separate(
    per_expert_equal: bool,
    route_reduce_equal: bool,
    expected: str,
) -> None:
    benchmark = _benchmark()

    assert (
        benchmark.classify_stage_discrepancy(
            per_expert_array_equal=per_expert_equal,
            fused_matches_stock_reduce=route_reduce_equal,
        )
        == expected
    )


def test_benchmark_cli_filters_topologies_rows_and_candidate_families() -> None:
    benchmark = _benchmark()
    args = benchmark._parser().parse_args(
        [
            "screen",
            "--model",
            "/tmp/hy3-q2",
            "--rows",
            "1,4,8",
            "--n-tiles",
            "32,64",
            "--k-tiles",
            "64,96",
            "--simd-groups",
            "2,4",
            "--activation-modes",
            "exact,fast",
            "--families",
            "stock_eager,stock_compiled,fused_e2",
            "--output-json",
            "/tmp/e2.json",
        ]
    )

    selected = benchmark.select_candidates(
        n_tiles=args.n_tiles,
        k_tiles=args.k_tiles,
        simd_groups=args.simd_groups,
        activation_modes=args.activation_modes,
    )
    assert args.rows == (1, 4, 8)
    assert args.families == ("stock_eager", "stock_compiled", "fused_e2")
    assert {candidate.name for candidate in selected} == {
        "n32_k64_sg2_exact",
        "n32_k96_sg2_exact",
        "n32_k64_sg4_exact",
        "n32_k96_sg4_exact",
        "n64_k64_sg4_exact",
        "n64_k96_sg4_exact",
        "n32_k64_sg2_fast",
        "n32_k96_sg2_fast",
        "n32_k64_sg4_fast",
        "n32_k96_sg4_fast",
        "n64_k64_sg4_fast",
        "n64_k96_sg4_fast",
    }


def test_fast_candidate_waits_for_matching_exact_correctness() -> None:
    benchmark = _benchmark()
    module = _module()
    fast = module.Hy3ExpertE2Candidate(64, 96, 4, "fast")
    exact = module.Hy3ExpertE2Candidate(64, 96, 4, "exact")

    assert benchmark.fast_prerequisite_met(fast, set()) is False
    assert benchmark.fast_prerequisite_met(fast, {benchmark.exact_candidate_key(exact)})
