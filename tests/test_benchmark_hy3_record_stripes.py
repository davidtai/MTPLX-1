from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_hy3_record_stripes as module


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_partition_record_is_aligned_disjoint_and_complete(width: int) -> None:
    record_bytes = 16 * 16_384
    destination = memoryview(bytearray(record_bytes))

    stripes = module.partition_record(
        destination,
        source_offset=7 * 16_384,
        logical_bytes=record_bytes,
        stripe_count=width,
        alignment=16_384,
    )

    try:
        stripe_bytes = record_bytes // width
        assert [stripe.logical_offset for stripe in stripes] == [
            index * stripe_bytes for index in range(width)
        ]
        assert [stripe.source_offset for stripe in stripes] == [
            7 * 16_384 + index * stripe_bytes for index in range(width)
        ]
        assert all(stripe.length == stripe_bytes for stripe in stripes)
        assert sum(stripe.length for stripe in stripes) == record_bytes
        assert sum(stripe.destination.nbytes for stripe in stripes) == record_bytes
        assert all(stripe.source_offset % 16_384 == 0 for stripe in stripes)
        assert all(stripe.length % 16_384 == 0 for stripe in stripes)
    finally:
        module.release_stripes(stripes)
        destination.release()


@pytest.mark.parametrize("width", [True, 0, 3, 16])
def test_partition_record_rejects_unsupported_width(width: object) -> None:
    destination = memoryview(bytearray(16_384))
    try:
        with pytest.raises(module.BenchmarkContractError, match="stripe count"):
            module.partition_record(
                destination,
                source_offset=0,
                logical_bytes=16_384,
                stripe_count=width,  # type: ignore[arg-type]
                alignment=16_384,
            )
    finally:
        destination.release()


def test_partition_record_rejects_invalid_geometry_before_io() -> None:
    with pytest.raises(module.BenchmarkContractError, match="writable"):
        module.partition_record(
            memoryview(bytes(16_384)),
            source_offset=0,
            logical_bytes=16_384,
            stripe_count=1,
            alignment=16_384,
        )

    destination = memoryview(bytearray(3 * 16_384))
    try:
        with pytest.raises(module.BenchmarkContractError, match="alignment"):
            module.partition_record(
                destination,
                source_offset=0,
                logical_bytes=destination.nbytes,
                stripe_count=2,
                alignment=16_384,
            )
        with pytest.raises(module.BenchmarkContractError, match="source offset"):
            module.partition_record(
                destination,
                source_offset=1,
                logical_bytes=destination.nbytes,
                stripe_count=1,
                alignment=16_384,
            )
        with pytest.raises(module.BenchmarkContractError, match="coverage"):
            module.partition_record(
                destination,
                source_offset=0,
                logical_bytes=2 * 16_384,
                stripe_count=1,
                alignment=16_384,
            )
    finally:
        destination.release()


def test_preadv_exact_counts_actual_attempts_and_returned_bytes() -> None:
    destination = memoryview(bytearray(b"......"))
    returns = iter((2, 4))
    calls: list[tuple[int, int]] = []

    def partial_preadv(_fd: int, buffers, offset: int) -> int:
        target = buffers[0]
        returned = next(returns)
        target[:returned] = b"abcdef"[offset : offset + returned]
        calls.append((offset, returned))
        return returned

    try:
        accounting = module.preadv_exact(
            3,
            source_offset=0,
            destination=destination,
            preadv=partial_preadv,
        )
        assert bytes(destination) == b"abcdef"
        assert calls == [(0, 2), (2, 4)]
        assert accounting.host_calls == 2
        assert accounting.returned_bytes == 6
    finally:
        destination.release()


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_every_width_reads_and_hashes_the_complete_record(
    tmp_path: Path, width: int
) -> None:
    payload = bytes((index * 17) % 251 for index in range(16 * 16_384))
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(payload)
    destination = bytearray(len(payload))
    record = module.RecordRange(
        identity="layer-1-expert-0",
        offset=0,
        length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    with sidecar.open("rb", buffering=0) as source:
        executor = None if width == 1 else ThreadPoolExecutor(max_workers=width)
        try:
            outcome = module.read_and_validate_record(
                source.fileno(),
                record,
                destination,
                stripe_count=width,
                executor=executor,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    assert destination == payload
    assert outcome.sha256 == record.sha256
    assert outcome.logical_publications == 1
    assert outcome.accounting.host_calls >= width
    assert outcome.accounting.returned_bytes == len(payload)


def test_one_way_control_is_direct_and_never_submits(tmp_path: Path) -> None:
    payload = b"x" * (2 * 16_384)
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(payload)

    class RejectSubmit:
        def submit(self, *_args, **_kwargs):
            raise AssertionError("the direct control must not use an executor")

    with sidecar.open("rb", buffering=0) as source:
        outcome = module.read_and_validate_record(
            source.fileno(),
            module.RecordRange(
                identity="record",
                offset=0,
                length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
            bytearray(len(payload)),
            stripe_count=1,
            executor=RejectSubmit(),
        )

    assert outcome.accounting.host_calls == 1
    assert outcome.logical_publications == 1


def test_striped_failure_drains_accepted_siblings_before_raising(monkeypatch) -> None:
    sibling_started = threading.Event()
    sibling_completed = threading.Event()
    first_failed = threading.Event()

    def injected(_fd: int, *, source_offset: int, destination, preadv=None):
        if source_offset == 0:
            assert sibling_started.wait(timeout=2)
            first_failed.set()
            raise OSError("injected stripe failure")
        sibling_started.set()
        assert first_failed.wait(timeout=2)
        destination[:] = b"x" * destination.nbytes
        sibling_completed.set()
        return module.IOAccounting(host_calls=1, returned_bytes=destination.nbytes)

    monkeypatch.setattr(module, "preadv_exact", injected)
    destination = bytearray(2 * 16_384)
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(OSError, match="injected stripe failure"):
            module.read_record_exact(
                3,
                module.RecordRange("record", 0, len(destination), "0" * 64),
                destination,
                stripe_count=2,
                executor=executor,
            )
    assert sibling_completed.is_set()


def test_interrupted_wait_settles_worker_before_releasing_destination(
    monkeypatch,
) -> None:
    worker_started = threading.Event()
    worker_completed = threading.Event()
    allow_worker = threading.Event()
    worker_errors: list[BaseException] = []

    def injected(_fd: int, *, source_offset: int, destination, preadv=None):
        if source_offset == 0:
            worker_started.set()
            assert allow_worker.wait(timeout=2)
            try:
                destination[0] = 7
            except BaseException as exc:
                worker_errors.append(exc)
            finally:
                worker_completed.set()
        return module.IOAccounting(host_calls=1, returned_bytes=destination.nbytes)

    class InterruptOnceFuture:
        def __init__(self, inner: Future[module.IOAccounting]) -> None:
            self.inner = inner
            self.result_calls = 0

        def result(self):
            self.result_calls += 1
            if self.result_calls == 1:
                assert worker_started.wait(timeout=2)
                raise KeyboardInterrupt("injected caller interruption")
            return self.inner.result()

        def done(self) -> bool:
            return self.inner.done()

    class InterruptingExecutor:
        def __init__(self) -> None:
            self.pool = ThreadPoolExecutor(max_workers=2)
            self.first: InterruptOnceFuture | None = None
            self.calls = 0

        def submit(self, *args, **kwargs):
            self.calls += 1
            future = self.pool.submit(*args, **kwargs)
            if self.calls == 1:
                self.first = InterruptOnceFuture(future)
                return self.first
            return future

        def close(self) -> None:
            self.pool.shutdown(wait=True, cancel_futures=False)

    monkeypatch.setattr(module, "preadv_exact", injected)
    executor = InterruptingExecutor()
    timer = threading.Timer(0.1, allow_worker.set)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt, match="caller interruption"):
            module.read_record_exact(
                3,
                module.RecordRange("record", 0, 2 * 16_384, "0" * 64),
                bytearray(2 * 16_384),
                stripe_count=2,
                executor=executor,
            )
    finally:
        timer.cancel()
        allow_worker.set()
        executor.close()

    assert worker_completed.is_set()
    assert worker_errors == []
    assert executor.first is not None
    assert executor.first.result_calls >= 2


def test_stripe_submission_blocks_termination_signals(monkeypatch) -> None:
    events: list[str] = []
    blocked = False

    @contextmanager
    def fake_blocked_signals():
        nonlocal blocked
        blocked = True
        events.append("enter")
        try:
            yield
        finally:
            blocked = False
            events.append("exit")

    class ImmediateExecutor:
        def submit(self, *_args, **_kwargs):
            assert blocked
            events.append("submit")
            future: Future[module.IOAccounting] = Future()
            future.set_result(module.IOAccounting(host_calls=1, returned_bytes=16_384))
            return future

    monkeypatch.setattr(module, "_blocked_termination_signals", fake_blocked_signals)
    module.read_record_exact(
        3,
        module.RecordRange("record", 0, 2 * 16_384, "0" * 64),
        bytearray(2 * 16_384),
        stripe_count=2,
        executor=ImmediateExecutor(),
    )

    assert events == ["enter", "submit", "exit", "enter", "submit", "exit"]


def test_hash_failure_publishes_no_logical_record(tmp_path: Path) -> None:
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(b"x" * 16_384)
    with sidecar.open("rb", buffering=0) as source:
        with pytest.raises(module.BenchmarkContractError, match="hash") as error:
            module.read_and_validate_record(
                source.fileno(),
                module.RecordRange("record", 0, 16_384, "0" * 64),
                bytearray(16_384),
                stripe_count=1,
                executor=None,
            )
    assert getattr(error.value, "logical_publications", 0) == 0


def test_balanced_order_covers_every_position_in_both_phases() -> None:
    widths = (1, 2, 4, 8)
    orders = [module.balanced_order(repeat, widths) for repeat in range(16)]
    for phase in (orders[:8], orders[8:]):
        assert all(set(order) == set(widths) for order in phase)
        for position in range(4):
            assert sorted(order[position] for order in phase) == [
                1,
                1,
                2,
                2,
                4,
                4,
                8,
                8,
            ]


def test_result_row_reports_actual_io_service_distribution_and_hashes() -> None:
    outcomes = [
        module.RecordOutcome(module.IOAccounting(2, 1024), 1.0, "a" * 64, 1),
        module.RecordOutcome(module.IOAccounting(3, 1024), 2.0, "b" * 64, 1),
        module.RecordOutcome(module.IOAccounting(4, 1024), 4.0, "c" * 64, 1),
    ]
    row = module.result_row(
        repeat=9,
        order_position=2,
        stripe_count=4,
        record_bytes=1024,
        outcomes=outcomes,
        expected_sha256=["a" * 64, "b" * 64, "c" * 64],
        validated_wall_seconds=0.010,
    )

    assert row["phase"] == "confirmation"
    assert row["logical_records"] == 3
    assert row["logical_bytes"] == 3072
    assert row["logical_record_publications"] == 3
    assert row["host_calls"] == 9
    assert row["returned_bytes"] == 3072
    assert row["byte_amplification"] == 1.0
    assert row["service_time_ms"] == {
        "mean": pytest.approx(7 / 3),
        "p50": 2.0,
        "p95": 4.0,
        "max": 4.0,
        "total": 7.0,
    }
    assert row["useful_gib_per_second"] == pytest.approx(3072 / 1024**3 / 0.007)
    assert row["validated_wall_gib_per_second"] == pytest.approx(3072 / 1024**3 / 0.010)
    assert row["expected_sha256"] == ["a" * 64, "b" * 64, "c" * 64]
    assert row["observed_sha256"] == ["a" * 64, "b" * 64, "c" * 64]
    assert row["hash_parity"] is True


def _synthetic_row(
    repeat: int,
    width: int,
    *,
    bandwidth: float,
    mean_ms: float,
    p95_ms: float,
    byte_amplification: float = 1.0,
    parity: bool = True,
) -> dict[str, object]:
    logical_bytes = 1000
    return {
        "repeat": repeat,
        "phase": "selection" if repeat < 8 else "confirmation",
        "stripe_count": width,
        "logical_records": 10,
        "logical_bytes": logical_bytes,
        "logical_record_publications": 10 if parity else 9,
        "host_calls": width * 10,
        "returned_bytes": int(logical_bytes * byte_amplification),
        "byte_amplification": byte_amplification,
        "useful_gib_per_second": bandwidth,
        "service_time_ms": {
            "mean": mean_ms,
            "p50": mean_ms,
            "p95": p95_ms,
            "max": p95_ms,
            "total": mean_ms * 10,
        },
        "hash_parity": parity,
    }


def test_analysis_accepts_repeatable_sub_five_percent_gain_without_amplification() -> (
    None
):
    rows: list[dict[str, object]] = []
    for repeat in range(16):
        rows.extend(
            (
                _synthetic_row(repeat, 1, bandwidth=10.0, mean_ms=1.0, p95_ms=1.2),
                _synthetic_row(repeat, 2, bandwidth=10.2, mean_ms=0.98, p95_ms=1.15),
                _synthetic_row(repeat, 4, bandwidth=10.5, mean_ms=0.95, p95_ms=1.3),
                _synthetic_row(
                    repeat,
                    8,
                    bandwidth=11.0,
                    mean_ms=0.9,
                    p95_ms=1.0,
                    byte_amplification=1.1,
                ),
            )
        )

    analysis = module.analyze_results(rows, seed=30, bootstrap_resamples=1000)

    assert analysis["decision"] == "accept"
    assert analysis["selected_stripe_count"] == 2
    candidates = analysis["candidates"]
    assert candidates["2"]["decision"] == "accept"
    assert candidates["2"]["full"]["relative_useful_gib_per_second_delta_pct"][
        "mean"
    ] == pytest.approx(2.0)
    assert candidates["4"]["decision"] == "reject"
    assert "p95_regression" in candidates["4"]["failed_gates"]
    assert candidates["8"]["decision"] == "reject"
    assert "byte_amplification" in candidates["8"]["failed_gates"]


def test_analysis_rejects_a_gain_that_does_not_repeat_in_confirmation() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(16):
        candidate_bandwidth = 10.2 if repeat < 8 else 9.8
        candidate_ms = 0.98 if repeat < 8 else 1.02
        for width in (1, 2, 4, 8):
            rows.append(
                _synthetic_row(
                    repeat,
                    width,
                    bandwidth=10.0 if width == 1 else candidate_bandwidth,
                    mean_ms=1.0 if width == 1 else candidate_ms,
                    p95_ms=1.2 if width == 1 else 1.15,
                )
            )

    analysis = module.analyze_results(rows, seed=30, bootstrap_resamples=1000)

    assert analysis["decision"] == "reject"
    assert analysis["selected_stripe_count"] is None
    assert (
        "confirmation_useful_gib_per_second"
        in analysis["candidates"]["2"]["failed_gates"]
    )


def test_analysis_rejects_confirmation_p95_regression_hidden_by_selection() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(16):
        for width in (1, 2, 4, 8):
            candidate = width != 1
            rows.append(
                _synthetic_row(
                    repeat,
                    width,
                    bandwidth=10.2 if candidate else 10.0,
                    mean_ms=0.98 if candidate else 1.0,
                    p95_ms=((0.3 if repeat < 8 else 1.3) if candidate else 1.2),
                )
            )

    analysis = module.analyze_results(rows, seed=30, bootstrap_resamples=1000)

    assert analysis["decision"] == "reject"
    assert analysis["selected_stripe_count"] is None
    assert "confirmation_p95_regression" in analysis["candidates"]["2"]["failed_gates"]


def test_confirmation_cannot_reselect_after_frozen_candidate_fails() -> None:
    rows: list[dict[str, object]] = []
    for repeat in range(16):
        for width in (1, 2, 4, 8):
            if width == 1:
                bandwidth, mean_ms = 10.0, 1.0
            elif width == 2:
                bandwidth = 10.2 if repeat < 8 else 9.8
                mean_ms = 0.98 if repeat < 8 else 1.02
            elif width == 4:
                bandwidth, mean_ms = 10.3, 0.97
            else:
                bandwidth, mean_ms = 9.0, 1.1
            rows.append(
                _synthetic_row(
                    repeat,
                    width,
                    bandwidth=bandwidth,
                    mean_ms=mean_ms,
                    p95_ms=1.1 if width in (2, 4) else 1.2,
                )
            )

    analysis = module.analyze_results(rows, seed=30, bootstrap_resamples=1000)

    assert analysis["selection"]["selected_stripe_count"] == 2
    assert analysis["decision"] == "reject"
    assert analysis["selected_stripe_count"] is None
    assert "not_preselected" in analysis["candidates"]["4"]["failed_gates"]


def test_payload_states_scope_and_direct_control_contract() -> None:
    payload = module.build_payload(
        command=["python", "benchmark.py"],
        configuration={"stripe_counts": [1, 2, 4, 8]},
        workload=[{"identity": "record", "offset": 0, "sha256": "a" * 64}],
        provenance={"git_commit": "abc"},
        results=[],
        analysis={"decision": "reject"},
    )

    assert payload["schema"] == "mtplx-hy3-record-stripe-benchmark-v2"
    assert payload["io_contract"]["control"] == "direct synchronous preadv"
    assert payload["io_contract"]["host_call_semantics"] == "actual preadv attempts"
    assert payload["io_contract"]["returned_byte_semantics"] == "actual positive bytes"
    assert payload["io_contract"]["full_record_hash_per_operation"] is True
    assert payload["scope"]["runtime_integration_authorized"] is False


def test_config_requires_two_independently_balanced_phases() -> None:
    assert module.validate_config(
        stripe_counts=(1, 2, 4, 8),
        operations=96,
        warmup_operations=32,
        repeats=16,
        bootstrap_resamples=1000,
    ) == (1, 2, 4, 8)
    with pytest.raises(module.BenchmarkContractError, match="16"):
        module.validate_config(
            stripe_counts=(1, 2, 4, 8),
            operations=96,
            warmup_operations=32,
            repeats=8,
            bootstrap_resamples=1000,
        )
    with pytest.raises(module.BenchmarkContractError, match="exactly"):
        module.validate_config(
            stripe_counts=(1, 2, 8),
            operations=96,
            warmup_operations=32,
            repeats=16,
            bootstrap_resamples=1000,
        )


def test_manifest_records_require_exact_hy3_sidecar_geometry() -> None:
    records = [
        SimpleNamespace(
            layer=1,
            expert=0,
            logical_bytes=module.HY3_RECORD_BYTES,
            sidecar_offset=0,
            sidecar_length=module.HY3_RECORD_BYTES,
            sha256="a" * 64,
        ),
        SimpleNamespace(
            layer=1,
            expert=1,
            logical_bytes=module.HY3_RECORD_BYTES,
            sidecar_offset=module.HY3_RECORD_BYTES,
            sidecar_length=module.HY3_RECORD_BYTES,
            sha256="b" * 64,
        ),
    ]
    manifest = SimpleNamespace(
        model_key="hy3-q4",
        records=records,
        sidecar=SimpleNamespace(
            alignment=module.ALIGNMENT_BYTES,
            file="experts.bin",
            size=2 * module.HY3_RECORD_BYTES,
            sha256="c" * 64,
        ),
    )

    converted = module.manifest_record_ranges(manifest)

    assert converted == (
        module.RecordRange("layer-1-expert-0", 0, module.HY3_RECORD_BYTES, "a" * 64),
        module.RecordRange(
            "layer-1-expert-1",
            module.HY3_RECORD_BYTES,
            module.HY3_RECORD_BYTES,
            "b" * 64,
        ),
    )

    records[1].sidecar_offset += 1
    with pytest.raises(module.BenchmarkContractError, match="aligned"):
        module.manifest_record_ranges(manifest)


def test_measure_schedule_reuses_persistent_candidate_executors(monkeypatch) -> None:
    warm = (module.RecordRange("warm", 0, 16_384, "a" * 64),)
    measured = (module.RecordRange("measured", 0, 16_384, "a" * 64),)
    executors = {1: None, 2: object(), 4: object(), 8: object()}
    observed: list[tuple[int, object, tuple[module.RecordRange, ...]]] = []

    def fake_measure(_fd, records, _destination, *, stripe_count, executor):
        observed.append((stripe_count, executor, tuple(records)))
        outcome = module.RecordOutcome(
            module.IOAccounting(stripe_count, 16_384),
            1.0,
            "a" * 64,
            1,
        )
        return [outcome] * len(records), 0.001

    class ColdResidency:
        def try_evict(self, _offset: int, _length: int) -> bool:
            return True

        def residency(self, _offset: int, _length: int):
            return SimpleNamespace(pages=1, resident_pages=0, referenced_pages=0)

    monkeypatch.setattr(module, "measure_records", fake_measure)
    rows = module.measure_schedule(
        3,
        warm,
        measured,
        bytearray(16_384),
        stripe_counts=(1, 2, 4, 8),
        repeats=16,
        executors=executors,
        residency=ColdResidency(),
    )

    assert len(rows) == 64
    assert len(observed) == 128
    assert all(executor is executors[width] for width, executor, _ in observed)
    assert all(records in (warm, measured) for _width, _executor, records in observed)
    assert all(row["cache_before"]["verified_cold"] is True for row in rows)
    assert all(row["cache_after"]["verified_cold"] is True for row in rows)


def test_cold_preparation_evicts_and_verifies_every_record() -> None:
    records = (
        module.RecordRange("a", 0, 16_384, "a" * 64),
        module.RecordRange("b", 16_384, 16_384, "b" * 64),
    )

    class FakeResidency:
        def __init__(self) -> None:
            self.evicted: list[tuple[int, int]] = []

        def try_evict(self, offset: int, length: int) -> bool:
            self.evicted.append((offset, length))
            return True

        def residency(self, _offset: int, _length: int):
            return SimpleNamespace(pages=4, resident_pages=0, referenced_pages=0)

    residency = FakeResidency()
    snapshot = module.prepare_cold_records(residency, records)

    assert residency.evicted == [(0, 16_384), (16_384, 16_384)]
    assert snapshot["resident_fraction"] == 0.0
    assert snapshot["verified_cold"] is True


def test_cold_preparation_fails_closed_when_pages_remain_resident() -> None:
    record = module.RecordRange("a", 0, 16_384, "a" * 64)

    class WarmResidency:
        def try_evict(self, _offset: int, _length: int) -> bool:
            return False

        def residency(self, _offset: int, _length: int):
            return SimpleNamespace(pages=4, resident_pages=1, referenced_pages=0)

    with pytest.raises(module.BenchmarkContractError, match="cold"):
        module.prepare_cold_records(WarmResidency(), (record,))


def test_workload_selection_is_unique_disjoint_and_deterministic() -> None:
    records = tuple(
        module.RecordRange(f"record-{index}", index * 16_384, 16_384, "a" * 64)
        for index in range(20)
    )

    first = module.select_workload(records, operations=8, warmup_operations=4, seed=30)
    second = module.select_workload(records, operations=8, warmup_operations=4, seed=30)

    assert first == second
    warm, measured = first
    assert len(set(warm)) == 4
    assert len(set(measured)) == 8
    assert set(warm).isdisjoint(measured)


def test_nocache_sidecar_open_checks_exact_size(tmp_path: Path) -> None:
    sidecar = tmp_path / "experts.bin"
    sidecar.write_bytes(b"x" * 16_384)
    with module.open_sidecar_nocache(sidecar, expected_size=16_384) as fd:
        assert os.fstat(fd).st_size == 16_384
    with pytest.raises(module.BenchmarkContractError, match="size"):
        with module.open_sidecar_nocache(sidecar, expected_size=32_768):
            pass


def test_parser_defaults_to_the_frozen_phase_two_campaign() -> None:
    args = module.build_parser().parse_args(["model", "manifest.json"])
    assert args.stripe_counts == (1, 2, 4, 8)
    assert args.operations == 96
    assert args.warmup_operations == 32
    assert args.repeats == 16
    assert args.seed == 30
    assert args.bootstrap_resamples >= 10_000


def test_git_provenance_fails_closed_when_git_inspection_fails(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["git"], 5)

    monkeypatch.setattr(subprocess, "check_output", fail)
    with pytest.raises(module.BenchmarkContractError, match="git"):
        module.git_provenance()


def test_final_provenance_rejects_mid_campaign_harness_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "benchmark.py"
    source.write_text("before\n", encoding="utf-8")
    initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    initial_git = {"git_commit": "abc", "git_dirty": False, "git_status": ""}
    monkeypatch.setattr(module, "git_provenance", lambda: dict(initial_git))
    source.write_text("after\n", encoding="utf-8")

    with pytest.raises(module.BenchmarkContractError, match="changed during"):
        module.verify_final_provenance(
            initial_git=initial_git,
            initial_harness_sha256=initial_hash,
            source=source,
        )


def test_final_provenance_rejects_mid_campaign_git_change(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "benchmark.py"
    source.write_text("stable\n", encoding="utf-8")
    initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    initial_git = {"git_commit": "abc", "git_dirty": False, "git_status": ""}
    monkeypatch.setattr(
        module,
        "git_provenance",
        lambda: {"git_commit": "abc", "git_dirty": True, "git_status": " M x"},
    )

    with pytest.raises(module.BenchmarkContractError, match="changed during"):
        module.verify_final_provenance(
            initial_git=initial_git,
            initial_harness_sha256=initial_hash,
            source=source,
        )
