"""Contract tests for the Hy3 destination-layout benchmark harness."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "benchmark_hy3_record_destinations.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_hy3_record_destinations", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _allocate_bytes(shapes: list[tuple[int, ...]]):
    def allocate(shape: tuple[int, ...]):
        shapes.append(shape)
        size = 1
        for dimension in shape:
            size *= dimension
        return bytearray(size)

    return allocate


def test_destinations_have_exact_fixed_stride_and_component_row_ownership() -> None:
    module = _load_module()
    shapes: list[tuple[int, ...]] = []
    finalized: list[tuple[object, ...]] = []

    arena, rows = module.allocate_destinations(
        capacity=2,
        record_bytes=6,
        component_lengths=(2, 1, 3),
        allocate=_allocate_bytes(shapes),
        finalize=lambda *resources: finalized.append(resources),
    )

    assert shapes == [(2, 6), (2, 2), (2, 1), (2, 3)]
    assert len(finalized) == 1
    assert len(finalized[0]) == 4

    arena_slot = arena.record_views(1)
    row_slot = rows.record_views(1)
    try:
        assert [view.nbytes for view in arena_slot] == [6]
        assert [view.nbytes for view in row_slot] == [2, 1, 3]
        arena_slot[0][:] = b"ABCDEF"
        for view, value in zip(row_slot, (b"AB", b"C", b"DEF"), strict=True):
            view[:] = value
        assert arena.record_bytes_at(1) == b"ABCDEF"
        assert rows.record_bytes_at(1) == b"ABCDEF"
    finally:
        module.release_views(arena_slot)
        module.release_views(row_slot)

    arena.close()
    rows.close()
    with pytest.raises(module.BenchmarkContractError, match="closed"):
        arena.record_views(0)
    with pytest.raises(module.BenchmarkContractError, match="closed"):
        rows.record_views(0)


def test_hy3_layout_allocates_one_arena_and_exactly_nine_component_resources() -> None:
    module = _load_module()
    shapes: list[tuple[int, ...]] = []
    finalized: list[tuple[object, ...]] = []

    def allocate_shape_only(shape: tuple[int, ...]) -> object:
        shapes.append(shape)
        return object()

    arena, rows = module.allocate_destinations(
        capacity=32,
        record_bytes=module.HY3_RECORD_BYTES,
        component_lengths=module.HY3_COMPONENT_LENGTHS,
        allocate=allocate_shape_only,
        finalize=lambda *resources: finalized.append(resources),
    )

    try:
        assert len(shapes) == 10
        assert shapes[0] == (32, module.HY3_RECORD_BYTES)
        assert shapes[1:] == [(32, length) for length in module.HY3_COMPONENT_LENGTHS]
        assert sum(module.HY3_COMPONENT_LENGTHS) == module.HY3_RECORD_BYTES
        assert len(finalized) == 1
        assert len(finalized[0]) == 10
    finally:
        arena.close()
        rows.close()


def test_exact_preadv_and_parity_preflight_use_the_same_ranges(tmp_path: Path) -> None:
    module = _load_module()
    source = b"aaaaBBBBccccDDDD"
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(source)
    ranges = (
        module.RecordRange(offset=4, length=4, identity="record-1"),
        module.RecordRange(offset=8, length=4, identity="record-2"),
    )
    arena, rows = module.allocate_destinations(
        capacity=2,
        record_bytes=4,
        component_lengths=(1, 3),
        allocate=lambda shape: bytearray(shape[0] * shape[1]),
        finalize=lambda *_resources: None,
    )
    fd = os.open(sidecar, os.O_RDONLY)
    try:
        parity = module.run_parity_preflight(fd, ranges, arena, rows)
        assert parity[0]["sha256"] == hashlib.sha256(b"BBBB").hexdigest()
        assert parity[1]["sha256"] == hashlib.sha256(b"cccc").hexdigest()
        assert all(item["arena_sha256"] == item["component_sha256"] for item in parity)
    finally:
        os.close(fd)
        arena.close()
        rows.close()


def test_manifest_range_preserves_zero_sidecar_offset() -> None:
    module = _load_module()
    record = SimpleNamespace(
        layer=1,
        expert=0,
        sidecar_offset=0,
        sidecar_length=module.HY3_RECORD_BYTES,
        sha256="digest",
    )

    result = module.manifest_record_range(record)

    assert result.offset == 0
    assert result.length == module.HY3_RECORD_BYTES
    assert result.identity == "layer-1-expert-0"
    assert result.sha256 == "digest"


def test_real_mlx_destinations_expose_writable_bytes_after_finalize() -> None:
    module = _load_module()
    import mlx.core as mx

    arena, rows = module.allocate_destinations(
        capacity=2,
        record_bytes=6,
        component_lengths=(2, 1, 3),
        allocate=lambda shape: mx.zeros(shape, dtype=mx.uint8),
        finalize=mx.eval,
    )
    arena_views = arena.record_views(1)
    row_views = rows.record_views(1)
    try:
        arena_views[0][:] = b"ABCDEF"
        for view, value in zip(row_views, (b"AB", b"C", b"DEF"), strict=True):
            view[:] = value
        assert arena.record_bytes_at(1) == b"ABCDEF"
        assert rows.record_bytes_at(1) == b"ABCDEF"
    finally:
        module.release_views(arena_views)
        module.release_views(row_views)
        arena.close()
        rows.close()


def test_destination_caches_each_base_memoryview_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    calls: list[object] = []
    original = module._resource_bytes

    def counted(resource: object) -> memoryview:
        calls.append(resource)
        return original(resource)

    monkeypatch.setattr(module, "_resource_bytes", counted)
    arena, rows = module.allocate_destinations(
        capacity=2,
        record_bytes=6,
        component_lengths=(2, 1, 3),
        allocate=lambda shape: bytearray(shape[0] * shape[1]),
        finalize=lambda *_resources: None,
    )
    try:
        for destination in (arena, rows):
            first = destination.record_views(0)
            second = destination.record_views(1)
            module.release_views(first)
            module.release_views(second)
        assert len(calls) == 4
    finally:
        arena.close()
        rows.close()


def test_record_and_batch_latencies_include_destination_view_setup(
    tmp_path: Path,
) -> None:
    module = _load_module()
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(b"ABCDEFGH")

    class SlowDestination:
        def record_views(self, slot: int) -> tuple[memoryview, ...]:
            del slot
            time.sleep(0.01)
            return (memoryview(bytearray(4)),)

    destination = SlowDestination()
    records = (
        module.RecordRange(offset=0, length=4, identity="left"),
        module.RecordRange(offset=4, length=4, identity="right"),
    )
    fd = os.open(sidecar, os.O_RDONLY)
    try:
        record_ms, _batch_ms, _elapsed = module._measure_individual(
            fd, records[:1], destination, host_read_concurrency=1
        )
        adjacent_record_ms, adjacent_batch_ms, _elapsed = module._measure_adjacent(
            fd, (records,), destination
        )
    finally:
        os.close(fd)

    assert record_ms[0] >= 8.0
    assert adjacent_record_ms[0] >= 18.0
    assert adjacent_batch_ms[0] >= 18.0


def test_read_helpers_fail_closed_on_short_nonadjacent_and_wrong_coverage(
    tmp_path: Path,
) -> None:
    module = _load_module()
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(b"01234567")
    fd = os.open(sidecar, os.O_RDONLY)
    try:
        with pytest.raises(module.BenchmarkContractError, match="coverage"):
            module.preadv_record_exact(
                fd,
                module.RecordRange(offset=0, length=4, identity="one"),
                (memoryview(bytearray(3)),),
            )
        with pytest.raises(module.BenchmarkContractError, match="short read"):
            module.preadv_record_exact(
                fd,
                module.RecordRange(offset=6, length=4, identity="short"),
                (memoryview(bytearray(4)),),
            )
        with pytest.raises(module.BenchmarkContractError, match="adjacent"):
            module.preadv_adjacent_batch_exact(
                fd,
                (
                    module.RecordRange(offset=0, length=2, identity="left"),
                    module.RecordRange(offset=4, length=2, identity="right"),
                ),
                (
                    (memoryview(bytearray(2)),),
                    (memoryview(bytearray(2)),),
                ),
            )
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    ("operations", "queue_depths", "batch_records", "match"),
    [
        (0, (1, 32), 32, "operations"),
        (31, (1, 32), 32, "operations"),
        (32, (0, 32), 32, "host read concurrency"),
        (32, (1, 33), 32, "host read concurrency"),
        (32, (1, 32), 0, "batch records"),
        (64, (1, 32), 64, "batch records"),
    ],
)
def test_config_rejects_invalid_operations_and_queue_depths(
    operations: int,
    queue_depths: tuple[int, ...],
    batch_records: int,
    match: str,
) -> None:
    module = _load_module()
    with pytest.raises(module.BenchmarkContractError, match=match):
        module.validate_config(
            operations=operations,
            host_read_concurrencies=queue_depths,
            batch_records=batch_records,
            repeats=4,
        )


def test_config_requires_balanced_even_repeats_and_defaults_to_four() -> None:
    module = _load_module()

    defaults = module.build_parser().parse_args(["/model", "/manifest"])
    assert defaults.repeats == 4
    assert defaults.host_read_concurrencies == "1,32"
    with pytest.raises(module.BenchmarkContractError, match="even"):
        module.validate_config(
            operations=32,
            host_read_concurrencies=(1, 32),
            batch_records=32,
            repeats=3,
        )


def test_sequences_are_precomputed_once_and_arm_order_alternates() -> None:
    module = _load_module()
    ranges = tuple(
        module.RecordRange(offset=index * 4, length=4, identity=f"record-{index}")
        for index in range(40)
    )

    workload = module.precompute_workload(
        ranges,
        operations=64,
        batch_records=4,
        seed=17,
    )
    repeated = module.precompute_workload(
        ranges,
        operations=64,
        batch_records=4,
        seed=17,
    )

    assert workload == repeated
    assert len(workload.individual) == 64
    assert len(workload.adjacent_batches) == 16
    assert all(
        right.offset == left.offset + left.length
        for batch in workload.adjacent_batches
        for left, right in zip(batch, batch[1:], strict=False)
    )
    assert module.arm_order(0) == ("record-arena", "component-rows")
    assert module.arm_order(1) == ("component-rows", "record-arena")
    assert module.arm_order(2) == ("record-arena", "component-rows")


def _manifest_fixture(module):
    segments = tuple(
        SimpleNamespace(
            component=component.name,
            length=component.length,
            dtype=component.dtype,
            shape=component.shape,
        )
        for component in module.HY3_RECORD_Q4_LAYOUT.components
    )
    records = tuple(
        SimpleNamespace(
            layer=layer,
            expert=expert,
            logical_bytes=module.HY3_RECORD_BYTES,
            segments=segments,
            sidecar_offset=((layer - 1) * 192 + expert) * module.HY3_RECORD_BYTES,
            sidecar_length=module.HY3_RECORD_BYTES,
            sha256=f"digest-{layer}-{expert}",
        )
        for layer in range(1, 80)
        for expert in range(192)
    )
    return SimpleNamespace(
        model_key="hy3-q4",
        source_repo="pipenetwork/Hy3-4bit",
        source_revision="160619d3f96c8470350b6dac0ef033a8381551e3",
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        records=records,
        manifest_sha256="8263670ddb775ccfccb51cb266fe5ddc3ce706114fc9c26b40ec36df54c367b4",
        sidecar=SimpleNamespace(
            file="experts.bin",
            alignment=16_384,
            sha256="5ba698b9b2c51bca66254e5d8d35101325e37dfe40744294d4aa233c980472ae",
            size=161_036_107_776,
        ),
    )


def test_manifest_contract_pins_revision_quantization_and_every_record() -> None:
    module = _load_module()
    manifest = _manifest_fixture(module)

    assert module.validate_manifest_contract(manifest) == module.HY3_COMPONENT_LENGTHS

    wrong_group = SimpleNamespace(**{**vars(manifest), "quant_group_size": 32})
    with pytest.raises(module.BenchmarkContractError, match="pinned Hy3"):
        module.validate_manifest_contract(wrong_group)

    bad_segment = SimpleNamespace(**vars(manifest.records[1].segments[-1]))
    bad_segment.length -= 2
    bad_record = SimpleNamespace(
        **{
            **vars(manifest.records[1]),
            "segments": (*manifest.records[1].segments[:-1], bad_segment),
        }
    )
    bad_manifest = SimpleNamespace(
        **{
            **vars(manifest),
            "records": (manifest.records[0], bad_record, *manifest.records[2:]),
        }
    )
    with pytest.raises(module.BenchmarkContractError, match="record geometry"):
        module.validate_manifest_contract(bad_manifest)

    truncated = SimpleNamespace(**{**vars(manifest), "records": manifest.records[:-1]})
    with pytest.raises(module.BenchmarkContractError, match="record corpus"):
        module.validate_manifest_contract(truncated)

    wrong_offset = SimpleNamespace(
        **{**vars(manifest.records[1]), "sidecar_offset": module.HY3_RECORD_BYTES + 1}
    )
    bad_offsets = SimpleNamespace(
        **{
            **vars(manifest),
            "records": (manifest.records[0], wrong_offset, *manifest.records[2:]),
        }
    )
    with pytest.raises(
        module.BenchmarkContractError, match="record identity or offset"
    ):
        module.validate_manifest_contract(bad_offsets)

    wrong_digest = SimpleNamespace(
        **{
            **vars(manifest),
            "sidecar": SimpleNamespace(**{**vars(manifest.sidecar), "sha256": "wrong"}),
        }
    )
    with pytest.raises(module.BenchmarkContractError, match="pinned Hy3"):
        module.validate_manifest_contract(wrong_digest)


def test_harness_provenance_includes_source_and_environment_identity() -> None:
    module = _load_module()

    provenance = module.harness_provenance()

    assert provenance["path"].endswith("benchmark_hy3_record_destinations.py")
    assert len(provenance["sha256"]) == 64
    assert isinstance(provenance["git_dirty"], bool)
    assert provenance["cwd"]
    assert provenance["mlx_version"]


def test_artifact_provenance_marks_sidecar_digest_as_declared_not_fully_verified() -> (
    None
):
    module = _load_module()
    manifest = _manifest_fixture(module)

    provenance = module.artifact_provenance(
        manifest,
        Path("/manifest.json"),
        Path("/experts.bin"),
    )

    assert provenance["declared_sidecar_sha256"] == manifest.sidecar.sha256
    assert provenance["full_sidecar_sha256_verified"] is False
    assert "sidecar_sha256" not in provenance


def test_result_distinguishes_host_read_concurrency_from_batch_width() -> None:
    module = _load_module()

    result = module._result(
        repeat=0,
        arm="record-arena",
        lane="adjacent-batch",
        host_read_concurrency=1,
        batch_records=32,
        record_count=32,
        record_bytes=4,
        record_latencies=[1.0] * 32,
        batch_latencies=[1.0],
        elapsed=1.0,
    )

    assert result["host_read_concurrency"] == 1
    assert result["batch_records"] == 32
    assert "io_queue_depth" not in result


def test_payload_preserves_exact_command_config_provenance_and_latency_metrics() -> (
    None
):
    module = _load_module()
    command = ["uv", "run", "scripts/benchmark_hy3_record_destinations.py", "/m", "/x"]
    config = {
        "operations": 64,
        "host_read_concurrencies": [1, 32],
        "batch_records": 32,
        "repeats": 4,
    }
    provenance = {
        "manifest_sha256": "manifest",
        "declared_sidecar_sha256": "sidecar",
    }
    results = [
        {
            "arm": "record-arena",
            "lane": "individual",
            "record_latency_ms": module.summarize_latencies([1.0, 2.0, 3.0]),
            "batch_latency_ms": module.summarize_latencies([4.0, 5.0, 6.0]),
        }
    ]

    payload = module.build_payload(
        command=command,
        config=config,
        provenance=provenance,
        workload_sha256="workload",
        parity=[{"sha256": "record"}],
        results=results,
    )

    assert payload["schema"] == "mtplx-hy3-record-destination-benchmark-v1"
    assert payload["command"]["argv"] == command
    assert (
        payload["command"]["shell"]
        == "uv run scripts/benchmark_hy3_record_destinations.py /m /x"
    )
    assert payload["configuration"] == config
    assert payload["provenance"] == provenance
    assert payload["workload_sha256"] == "workload"
    assert payload["mechanism"] == "destination/layout efficiency"
    assert payload["io_contract"]["syscall"] == "os.preadv"
    assert payload["io_contract"]["claims_syscall_elimination"] is False
    assert payload["io_contract"]["metal_reads_issued"] is False
    assert payload["io_contract"]["claims_runtime_fence_safety"] is False
    assert payload["results"][0]["record_latency_ms"]["p50"] == 2.0
    assert payload["results"][0]["record_latency_ms"]["p95"] == 3.0


def test_open_sidecar_requires_and_applies_f_nocache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_module()
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(b"record")
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(module.fcntl, "F_NOCACHE", 48, raising=False)
    monkeypatch.setattr(
        module.fcntl,
        "fcntl",
        lambda fd, command, value: calls.append((fd, command, value)),
    )

    with module.open_sidecar_nocache(sidecar, expected_size=6) as fd:
        assert fd >= 0
    assert len(calls) == 1
    assert calls[0][1:] == (48, 1)

    with pytest.raises(module.BenchmarkContractError, match="sidecar size"):
        with module.open_sidecar_nocache(sidecar, expected_size=7):
            pass

    monkeypatch.delattr(module.fcntl, "F_NOCACHE", raising=False)
    with pytest.raises(module.BenchmarkContractError, match="F_NOCACHE"):
        with module.open_sidecar_nocache(sidecar):
            pass
