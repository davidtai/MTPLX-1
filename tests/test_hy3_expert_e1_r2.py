from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx


def _module():
    from mtplx import hy3_expert_e1_r2

    return hy3_expert_e1_r2


def _benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "hy3_expert_e1_r2.py"
    spec = importlib.util.spec_from_file_location("hy3_expert_e1_r2_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_pair() -> tuple[mx.array, mx.array]:
    gate = mx.array(np.arange(2 * 256 * 4, dtype=np.uint32).reshape(2, 256, 4))
    up = gate + mx.array(100, dtype=mx.uint32)
    return gate, up


@pytest.mark.parametrize(
    "layout",
    (
        "gate_then_up",
        "output_interleaved",
        "block_interleaved_16",
        "block_interleaved_32",
        "block_interleaved_64",
        "block_interleaved_128",
        "block_interleaved_256",
    ),
)
def test_pack_and_reconstruct_component_pair_without_steady_duplicate(
    layout: str,
) -> None:
    module = _module()
    gate, up = _small_pair()

    packed = module.pack_component_pair(gate, up, layout=layout)
    restored_gate, restored_up = module.reconstruct_component_pair(
        packed,
        split_at=256,
        layout=layout,
    )
    mx.eval(packed, restored_gate, restored_up)

    assert tuple(packed.shape) == (2, 512, 4)
    assert int(packed.nbytes) == int(gate.nbytes) + int(up.nbytes)
    assert mx.array_equal(restored_gate, gate).item()
    assert mx.array_equal(restored_up, up).item()


def test_r2_candidate_catalog_keeps_layout_order_and_compile_attributable() -> None:
    module = _module()

    candidates = module.hy3_e1_r2_candidates()

    assert len(candidates) == 28
    assert {candidate.layout for candidate in candidates} == {
        "gate_then_up",
        "output_interleaved",
        "block_interleaved_16",
        "block_interleaved_32",
        "block_interleaved_64",
        "block_interleaved_128",
        "block_interleaved_256",
    }
    assert {candidate.assignment_order for candidate in candidates} == {
        "original",
        "slot_sorted",
    }
    assert {candidate.execution for candidate in candidates} == {"eager", "compiled"}
    assert len({candidate.name for candidate in candidates}) == len(candidates)


def test_r2_benchmark_cli_filters_one_refinement_candidate() -> None:
    benchmark = _benchmark()
    args = benchmark._parser().parse_args(
        [
            "profile",
            "--experts",
            "0,1,2,3,4,5,6,7",
            "--layouts",
            "gate_then_up",
            "--assignment-orders",
            "original",
            "--executions",
            "compiled",
            "--candidate-name",
            "packed_halves_direct_compiled",
            "--output-json",
            "/tmp/e1-r2.json",
        ]
    )

    selected = benchmark._select_candidates(args)

    assert args.experts == tuple(range(8))
    assert [candidate.name for candidate in selected] == [
        "packed_halves_direct_compiled"
    ]


def _pack_q2(values: np.ndarray) -> np.ndarray:
    assert values.ndim == 1 and values.size % 16 == 0
    words = np.zeros(values.size // 16, dtype=np.uint32)
    for index, value in enumerate(values.astype(np.uint32)):
        words[index // 16] |= value << np.uint32(2 * (index % 16))
    return words


def test_scalar_affine_q2_formula_decodes_bits_scale_and_bias() -> None:
    module = _module()
    quantized = np.tile(np.arange(4, dtype=np.uint32), 16)
    words = _pack_q2(quantized)
    scales = np.array([0.25], dtype=np.float32)
    biases = np.array([-0.5], dtype=np.float32)
    activation = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    decoded = module.unpack_affine_q2_row(
        words,
        scales,
        biases,
        group_size=64,
    )
    dot = module.scalar_affine_q2_dot(
        activation,
        words,
        scales,
        biases,
        group_size=64,
    )

    expected = quantized.astype(np.float32) * scales[0] + biases[0]
    np.testing.assert_array_equal(decoded, expected)
    assert dot == pytest.approx(float(np.dot(activation.astype(np.float64), expected)))


def test_r1_reduction_simulation_preserves_q2_formula_but_changes_order() -> None:
    module = _module()
    rng = np.random.default_rng(65)
    quantized = rng.integers(0, 4, size=256, dtype=np.uint32)
    words = _pack_q2(quantized)
    scales = rng.normal(0.0, 0.2, size=4).astype(np.float32)
    biases = rng.normal(0.0, 0.2, size=4).astype(np.float32)
    activation = rng.normal(size=256).astype(np.float32)

    scalar = module.scalar_affine_q2_dot(
        activation,
        words,
        scales,
        biases,
        group_size=64,
    )
    simulated = module.simulate_r1_affine_q2_dot(
        activation,
        words,
        scales,
        biases,
        group_size=64,
        k_tile=128,
    )

    assert np.isfinite(simulated)
    assert simulated == pytest.approx(scalar, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        (
            {
                "decoded_values_match": False,
                "r1_matches_simulated_reduction": False,
                "activation_matches_from_preactivations": True,
                "bf16_cast_matches": True,
            },
            "indexing_or_dequantization_error",
        ),
        (
            {
                "decoded_values_match": True,
                "r1_matches_simulated_reduction": True,
                "activation_matches_from_preactivations": True,
                "bf16_cast_matches": True,
            },
            "arithmetic_reduction_order",
        ),
        (
            {
                "decoded_values_match": True,
                "r1_matches_simulated_reduction": True,
                "activation_matches_from_preactivations": False,
                "bf16_cast_matches": True,
            },
            "activation_arithmetic_error",
        ),
        (
            {
                "decoded_values_match": True,
                "r1_matches_simulated_reduction": True,
                "activation_matches_from_preactivations": True,
                "bf16_cast_matches": False,
            },
            "bf16_cast_error",
        ),
    ),
)
def test_stage_classification_is_explicit(
    kwargs: dict[str, bool], expected: str
) -> None:
    module = _module()

    assert module.classify_stage_discrepancy(**kwargs) == expected
