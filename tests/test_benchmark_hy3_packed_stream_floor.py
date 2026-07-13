"""Contract tests for the issue #31 packed-stream floor benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "benchmark_hy3_packed_stream_floor.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_hy3_packed_stream_floor", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_metal(module) -> None:
    if module.mx is None:
        pytest.skip("MLX Metal is unavailable")
    try:
        available = bool(module.mx.metal.is_available())
    except (AttributeError, RuntimeError):
        pytest.skip("MLX Metal is unavailable")
    if not available:
        pytest.skip("MLX Metal is unavailable")


def _logical_projection_bytes(module, *, capacity: int, n: int, k: int):
    groups = k // module.GROUP_SIZE
    weight = bytes(index % 251 for index in range(capacity * n * groups * 32))
    scale = bytes((index + 17) % 251 for index in range(capacity * n * groups * 2))
    bias = bytes((index + 31) % 251 for index in range(capacity * n * groups * 2))
    return weight, scale, bias


@pytest.mark.parametrize(
    ("k", "expected_row_bytes", "expected_chunks"),
    [(4096, 2304, 16), (1536, 864, 6)],
)
def test_packed_layout_has_exact_aligned_production_strides(
    k: int, expected_row_bytes: int, expected_chunks: int
) -> None:
    module = _load_module()

    layout = module.projection_layout(k)

    assert layout.groups == k // 64
    assert layout.chunks == expected_chunks
    assert layout.split_row_bytes == expected_row_bytes
    assert layout.packed_row_bytes == expected_row_bytes
    assert layout.packed_row_bytes % 16 == 0
    assert module.CHUNK_BYTES == 144


def test_hy3_projection_and_record_byte_counts_are_exact() -> None:
    module = _load_module()

    gate = module.projection_bytes(capacity=1, n=1536, k=4096)
    down = module.projection_bytes(capacity=1, n=4096, k=1536)

    assert gate == 3_538_944
    assert down == 3_538_944
    assert 2 * gate + down == 10_616_832
    assert module.routed_layer_bytes(top_k=8) == 84_934_656


@pytest.mark.parametrize("k", [64, 128, 192, 320, 4100])
def test_layout_rejects_non_four_group_input_widths(k: int) -> None:
    module = _load_module()

    with pytest.raises(module.BenchmarkContractError, match="divisible by 256"):
        module.projection_layout(k)


@pytest.mark.parametrize(("n", "k"), [(3, 4096), (5, 1536)])
def test_raw_pack_round_trip_is_exact_without_padding(n: int, k: int) -> None:
    module = _load_module()
    capacity = 2
    weight, scale, bias = _logical_projection_bytes(module, capacity=capacity, n=n, k=k)

    packed = module.pack_projection_bytes(
        weight,
        scale,
        bias,
        capacity=capacity,
        n=n,
        k=k,
    )
    unpacked = module.unpack_projection_bytes(
        packed,
        capacity=capacity,
        n=n,
        k=k,
    )

    assert len(packed) == module.projection_bytes(capacity=capacity, n=n, k=k)
    assert unpacked == (weight, scale, bias)


def test_chunk_order_is_four_weight_groups_then_scale_and_bias() -> None:
    module = _load_module()
    weight = b"".join(bytes([group]) * 32 for group in range(4))
    scale = b"".join(bytes([40 + group, 50 + group]) for group in range(4))
    bias = b"".join(bytes([60 + group, 70 + group]) for group in range(4))

    packed = module.pack_projection_bytes(weight, scale, bias, capacity=1, n=1, k=256)

    assert packed[:128] == weight
    assert packed[128:136] == scale
    assert packed[136:144] == bias


@pytest.mark.parametrize("delta", [-1, 1])
def test_unpack_rejects_truncated_or_extra_bytes(delta: int) -> None:
    module = _load_module()
    packed = bytes(module.CHUNK_BYTES + delta)

    with pytest.raises(module.BenchmarkContractError, match="packed byte length"):
        module.unpack_projection_bytes(packed, capacity=1, n=1, k=256)


def test_mlx_packer_matches_the_raw_layout() -> None:
    module = _load_module()
    _require_metal(module)
    import mlx.core as mx

    weight, scale, bias = _logical_projection_bytes(module, capacity=1, n=2, k=256)
    weight_array = mx.array(list(weight), dtype=mx.uint8).reshape((1, 2, 128))
    scale_array = mx.array(list(scale), dtype=mx.uint8).reshape((1, 2, 4, 2))
    bias_array = mx.array(list(bias), dtype=mx.uint8).reshape((1, 2, 4, 2))

    packed = module.pack_projection_mlx(weight_array, scale_array, bias_array)
    mx.eval(packed)

    assert bytes(memoryview(packed)) == module.pack_projection_bytes(
        weight, scale, bias, capacity=1, n=2, k=256
    )


def test_paired_summary_and_weighted_gate_use_process_repeats() -> None:
    module = _load_module()
    gate = module.summarize_process_pairs(
        control_ms=[10.0, 10.1, 9.9, 10.0],
        packed_ms=[9.0, 9.1, 8.9, 9.0],
    )
    down = module.summarize_process_pairs(
        control_ms=[20.0, 20.2, 19.8, 20.0],
        packed_ms=[18.0, 18.2, 17.8, 18.0],
    )
    weighted = module.weighted_projection_summary(gate, down)

    assert gate["process_repeats"] == 4
    expected_speedup = (
        sum(
            (control / packed - 1) * 100
            for control, packed in zip(
                [10.0, 10.1, 9.9, 10.0], [9.0, 9.1, 8.9, 9.0], strict=True
            )
        )
        / 4
    )
    assert gate["speedup_percent"] == pytest.approx(expected_speedup)
    assert weighted["control_ms"] == pytest.approx(40.0)
    assert weighted["packed_ms"] == pytest.approx(36.0)
    assert module.classify_floor_gate(weighted, minimum_percent=5.0) == "go-exact-qmv"


def test_floor_gate_is_inconclusive_when_interval_crosses_five_percent() -> None:
    module = _load_module()
    summary = {
        "speedup_percent": 5.5,
        "speedup_percent_ci95": [4.8, 6.2],
        "packed_p95_regression_percent": 0.0,
    }

    assert module.classify_floor_gate(summary, minimum_percent=5.0) == "inconclusive"


def test_floor_gate_stops_when_upper_interval_cannot_reach_five_percent() -> None:
    module = _load_module()
    summary = {
        "speedup_percent": 3.0,
        "speedup_percent_ci95": [2.0, 4.0],
        "packed_p95_regression_percent": -1.0,
    }

    assert module.classify_floor_gate(summary, minimum_percent=5.0) == "no-go"


def test_floor_gate_rejects_projection_p95_regression() -> None:
    module = _load_module()
    summary = {
        "speedup_percent": 8.0,
        "speedup_percent_ci95": [6.0, 10.0],
        "packed_p95_regression_percent_worst_process": 2.1,
    }

    assert module.classify_floor_gate(summary, minimum_percent=5.0) == "no-go"


def test_kernel_checksums_match_for_production_geometries() -> None:
    module = _load_module()
    _require_metal(module)
    import mlx.core as mx

    for n, k in ((1536, 4096), (4096, 1536)):
        arrays = module.allocate_projection_arrays(
            capacity=9,
            n=n,
            k=k,
            seed=n + k,
        )
        slots = mx.array([8, 1, 8, 0, 7, 2, 6, 3], dtype=mx.int32)
        split_kernel = module.build_split_floor_kernel()
        packed_kernel = module.build_packed_floor_kernel()

        split = module.dispatch_split(split_kernel, arrays, slots, n=n, k=k)
        packed = module.dispatch_packed(packed_kernel, arrays, slots, n=n, k=k)
        mx.eval(split, packed)

        assert split.shape == packed.shape == (module.TOP_K, n)
        assert bool(mx.array_equal(split, packed))


def test_parser_defaults_lock_the_measurement_contract() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(["--output", "/tmp/result.json"])

    assert args.capacity == 102
    assert args.per_eval == 16
    assert args.warmup == 25
    assert args.iters == 150
    assert args.repeats == 4
    assert args.minimum_percent == 5.0


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (["--capacity", "16"], "capacity must be at least 102"),
        (["--repeats", "5"], "exactly 4 process repeats"),
        (["--minimum-percent", "4.9"], "exactly a 5.0% promotion gate"),
    ],
)
def test_cli_rejects_noncanonical_decision_parameters(
    arguments: list[str], match: str
) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        ["--output", "/tmp/result.json", *arguments]
    )

    with pytest.raises(module.BenchmarkContractError, match=match):
        module._validate_args(args)
