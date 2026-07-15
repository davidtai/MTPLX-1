from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


_BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "hy3_q2_layout_coalescing.py"


def _benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "hy3_q2_layout_coalescing_benchmark", _BENCHMARK
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _actual_shape_components(capacity: int = 1) -> dict[str, np.ndarray]:
    from mtplx import hy3_q2_layout_coalescing as layout

    components: dict[str, np.ndarray] = {}
    for index, (name, shape) in enumerate(layout.hy3_q2_component_shapes().items()):
        dtype = np.uint32 if name.endswith(".weight") else np.uint16
        size = capacity * int(np.prod(shape))
        values = np.arange(size, dtype=dtype).reshape((capacity, *shape))
        components[name] = values + np.asarray(index, dtype=dtype)
    return components


def test_layout_geometry_matches_actual_hy3_q2_expert_record() -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    assert layout.HY3_Q2_ROWS == tuple(range(1, 9))
    assert layout.HY3_Q2_PRIMARY_ROW == 4
    assert layout.HY3_Q2_BITS == 2
    assert layout.HY3_Q2_GROUP_SIZE == 64
    assert layout.hy3_q2_component_shapes() == {
        "gate_proj.weight": (1536, 256),
        "gate_proj.scales": (1536, 64),
        "gate_proj.biases": (1536, 64),
        "up_proj.weight": (1536, 256),
        "up_proj.scales": (1536, 64),
        "up_proj.biases": (1536, 64),
        "down_proj.weight": (4096, 96),
        "down_proj.scales": (4096, 24),
        "down_proj.biases": (4096, 24),
    }
    assert layout.hy3_q2_component_bytes() == {
        "gate_proj.weight": 1_572_864,
        "gate_proj.scales": 196_608,
        "gate_proj.biases": 196_608,
        "up_proj.weight": 1_572_864,
        "up_proj.scales": 196_608,
        "up_proj.biases": 196_608,
        "down_proj.weight": 1_572_864,
        "down_proj.scales": 196_608,
        "down_proj.biases": 196_608,
    }
    assert layout.HY3_Q2_EXPERT_RECORD_BYTES == 5_898_240


def test_candidate_catalog_exhausts_packet_block_and_vector_refinements() -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    candidates = layout.hy3_q2_layout_candidates()

    assert len(candidates) == 30
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert {candidate.layout for candidate in candidates} == {
        "output_packet",
        "group_output_packet",
        "tiled_projection_field",
        "tiled_field_projection",
    }
    assert {candidate.vector_width for candidate in candidates} == {1, 2, 4}
    assert {
        candidate.block_size
        for candidate in candidates
        if candidate.layout.startswith("tiled_")
    } == {16, 32, 64, 128}
    assert all(candidate.resident_byte_overhead == 0 for candidate in candidates)
    assert all(candidate.promotion_eligible for candidate in candidates)
    assert {control.vector_width for control in layout.hy3_q2_source_controls()} == {
        1,
        2,
        4,
    }


@pytest.mark.parametrize(
    ("layout_name", "block_size"),
    [
        ("output_packet", 0),
        ("group_output_packet", 0),
        ("tiled_projection_field", 32),
        ("tiled_field_projection", 32),
    ],
)
def test_pack_reconstructs_all_actual_shape_components_byte_exact(
    layout_name: str, block_size: int
) -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    source = _actual_shape_components()
    candidate = layout.Hy3Q2LayoutCandidate(layout_name, block_size, 1)
    identity = layout.Hy3Q2SlotIdentity(slot=3, expert=17, generation=9)

    packed = layout.pack_hy3_q2_bank(source, candidate, identities=(identity,))
    restored = packed.reconstruct_components()

    assert packed.identities == (identity,)
    assert packed.source_component_bytes == layout.HY3_Q2_EXPERT_RECORD_BYTES
    assert packed.packed_bytes == layout.HY3_Q2_EXPERT_RECORD_BYTES
    assert packed.steady_state_duplicate_bytes == 0
    assert set(restored) == set(source)
    for name in source:
        np.testing.assert_array_equal(restored[name], source[name])


@pytest.mark.parametrize("changed_field", ["slot", "expert", "generation"])
def test_packed_replacement_fails_closed_on_slot_identity_or_generation_change(
    changed_field: str,
) -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    identity = layout.Hy3Q2SlotIdentity(slot=3, expert=17, generation=9)
    packed = layout.pack_hy3_q2_bank(
        _actual_shape_components(),
        layout.Hy3Q2LayoutCandidate("tiled_projection_field", 32, 1),
        identities=(identity,),
    )
    values = {"slot": 3, "expert": 17, "generation": 9}
    values[changed_field] += 1
    changed = layout.Hy3Q2SlotIdentity(**values)

    with pytest.raises(layout.Hy3Q2LayoutError, match=changed_field):
        packed.validate_replacement((changed,))

    assert packed.validate_replacement((identity,)) is True


def test_transaction_model_identifies_tiled_field_major_coalescing_gain() -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    output_packet = layout.hy3_q2_coalescing_evidence(
        layout.Hy3Q2LayoutCandidate("output_packet", 0, 1)
    )
    group_packet = layout.hy3_q2_coalescing_evidence(
        layout.Hy3Q2LayoutCandidate("group_output_packet", 0, 1)
    )
    tiled = layout.hy3_q2_coalescing_evidence(
        layout.Hy3Q2LayoutCandidate("tiled_projection_field", 32, 1)
    )

    assert output_packet["logical_bytes_per_output_group"] == 60
    assert group_packet["logical_bytes_per_output_group"] == 60
    assert tiled["logical_bytes_per_output_group"] == 60
    assert tiled["source"]["cache_lines"] == 536
    assert output_packet["candidate"]["cache_lines"] == 480
    assert group_packet["candidate"]["cache_lines"] == 75
    assert tiled["candidate"]["cache_lines"] == 15
    assert tiled["candidate"]["cache_lines"] < group_packet["candidate"]["cache_lines"]
    assert tiled["candidate"]["metadata_loads_per_output_group"] == 3
    assert tiled["source"]["metadata_loads_per_output_group"] == 6


@pytest.mark.parametrize("rows", range(1, 9))
def test_scan_shape_covers_every_r1_through_r8_assignment(rows: int) -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    assert layout.hy3_q2_scan_output_shape(rows) == (rows * 8, 7168)


def test_source_and_packed_scan_sources_cover_all_affine_components() -> None:
    from mtplx import hy3_q2_layout_coalescing as layout

    source = layout.render_hy3_q2_source_scan_source(
        layout.Hy3Q2SourceControl(vector_width=4)
    )
    candidate = layout.render_hy3_q2_packed_scan_source(
        layout.Hy3Q2LayoutCandidate("tiled_field_projection", 64, 4)
    )

    assert "constexpr uint VECTOR_WIDTH = 4" in source
    assert "gate_weight" in source
    assert "gate_scales" in source
    assert "gate_biases" in source
    assert "up_weight" in source
    assert "up_scales" in source
    assert "up_biases" in source
    assert "down_weight" in source
    assert "down_scales" in source
    assert "down_biases" in source
    assert "constexpr uint BLOCK_SIZE = 64" in candidate
    assert "gate_up_packed" in candidate
    assert "down_packed" in candidate
    assert "metadata" in candidate


def test_scan_dispatch_uses_identical_logical_shape_and_work(monkeypatch) -> None:
    import mlx.core as mx

    from mtplx import hy3_q2_layout_coalescing as layout

    source_np = _actual_shape_components()
    candidate = layout.Hy3Q2LayoutCandidate("tiled_projection_field", 32, 2)
    packed = layout.pack_hy3_q2_bank(
        source_np,
        candidate,
        identities=(layout.Hy3Q2SlotIdentity(0, 0, 1),),
    )
    source = layout.hy3_q2_components_to_mlx(source_np)
    slots = mx.zeros((1, 8), dtype=mx.int32)
    calls: list[dict] = []

    def fake_kernel(*, inputs, **kwargs):
        calls.append({"inputs": inputs, **kwargs})
        return (mx.zeros(kwargs["output_shapes"][0], dtype=mx.uint32),)

    monkeypatch.setattr(
        layout, "build_hy3_q2_source_scan_kernel", lambda _control: fake_kernel
    )
    monkeypatch.setattr(
        layout, "build_hy3_q2_packed_scan_kernel", lambda _candidate: fake_kernel
    )

    source_output = layout.scan_hy3_q2_source(
        source,
        slots,
        control=layout.Hy3Q2SourceControl(2),
    )
    packed_output = layout.scan_hy3_q2_packed(packed, slots)

    assert tuple(source_output.shape) == (8, 7168)
    assert tuple(packed_output.shape) == (8, 7168)
    assert calls[0]["grid"] == calls[1]["grid"]
    assert calls[0]["threadgroup"] == calls[1]["threadgroup"] == (256, 1, 1)


def test_benchmark_cli_filters_layout_block_vector_search_space() -> None:
    benchmark = _benchmark_module()
    args = benchmark._parser().parse_args(
        [
            "--layouts",
            "tiled_projection_field",
            "--block-sizes",
            "32,64",
            "--vector-widths",
            "2,4",
            "--screen-repeats",
            "3",
            "--refine-repeats",
            "9",
            "--output-json",
            "/tmp/issue68-layout.json",
        ]
    )

    candidates = benchmark._select_candidates(args)

    assert args.rows == tuple(range(1, 9))
    assert args.primary_row == 4
    assert args.screen_repeats == 3
    assert args.refine_repeats == 9
    assert {candidate.name for candidate in candidates} == {
        "tiled_projection_field_b32_v2",
        "tiled_projection_field_b32_v4",
        "tiled_projection_field_b64_v2",
        "tiled_projection_field_b64_v4",
    }


@pytest.mark.skipif(
    os.environ.get("MTPLX_RUN_HARDWARE_LAYOUT_TESTS") != "1",
    reason="explicit exclusive-MLX hardware lane required",
)
@pytest.mark.parametrize(
    ("layout_name", "block_size", "vector_width"),
    [
        ("output_packet", 0, 1),
        ("group_output_packet", 0, 2),
        ("tiled_projection_field", 32, 4),
        ("tiled_field_projection", 64, 4),
    ],
)
def test_hardware_scan_checksum_matches_source_for_every_layout_family(
    layout_name: str, block_size: int, vector_width: int
) -> None:
    import mlx.core as mx

    from mtplx import hy3_q2_layout_coalescing as layout

    source_np = _actual_shape_components()
    candidate = layout.Hy3Q2LayoutCandidate(layout_name, block_size, vector_width)
    packed = layout.pack_hy3_q2_bank(
        source_np,
        candidate,
        identities=(layout.Hy3Q2SlotIdentity(0, 0, 1),),
    ).to_mlx()
    source = layout.hy3_q2_components_to_mlx(source_np)
    slots = mx.zeros((1, 8), dtype=mx.int32)

    expected = layout.scan_hy3_q2_source(
        source,
        slots,
        control=layout.Hy3Q2SourceControl(vector_width),
    )
    observed = layout.scan_hy3_q2_packed(packed, slots)
    mx.eval(expected, observed)

    assert mx.array_equal(observed, expected).item()


def test_paired_summary_and_promotion_gate_require_positive_interval() -> None:
    benchmark = _benchmark_module()

    summary = benchmark._paired_summary(
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        resamples=500,
        seed=68,
    )

    assert summary["control_over_candidate_ratio_of_means"] == 2.0
    assert summary["bootstrap_mean_ratio_95_ci"] == [2.0, 2.0]
    assert (
        benchmark._promotion_gate(
            byte_reconstruction_exact=True,
            scan_checksum_exact=True,
            expert_output_exact=True,
            resident_byte_overhead=0,
            primary_ratio_ci=summary["bootstrap_mean_ratio_95_ci"],
        )
        is True
    )
    assert (
        benchmark._promotion_gate(
            byte_reconstruction_exact=True,
            scan_checksum_exact=True,
            expert_output_exact=True,
            resident_byte_overhead=0,
            primary_ratio_ci=[0.99, 1.2],
        )
        is False
    )
