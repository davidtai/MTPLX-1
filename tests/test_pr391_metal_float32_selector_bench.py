from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROWS = 3338
WIDTH = 20
CAPTURE_SHA = "8b43a1734a756627790d8e4ac033731dc6b3705d690b400dc0800365c9782a6f"
CAPTURE_SOURCE_COMMIT = "340c153375de864740151c8c7a4c6368fd4af745"
BENCHMARK_SOURCE_COMMIT = "673104f5a64c5489547e729c581f413cb059e76e"


def test_guard_attestation_window_rejects_stale_oversized_and_boolean_fields() -> None:
    from scripts.pr391_metal_float32_selector_bench import GuardAttestationError
    from scripts.pr391_metal_float32_selector_bench import _validate_attestation_window

    now = 100_000_000_000
    valid = {
        "schema_version": 1,
        "guard_pid": 10,
        "child_pid": 11,
        "lock_device": 12,
        "lock_inode": 13,
        "issued_monotonic_ns": now - 1,
        "expires_monotonic_ns": now + 1,
    }
    _validate_attestation_window(valid, now=now)

    for updates in (
        {"expires_monotonic_ns": now - 1},
        {"issued_monotonic_ns": now + 1, "expires_monotonic_ns": now + 2},
        {
            "issued_monotonic_ns": now - 60_000_000_001,
            "expires_monotonic_ns": now + 1,
        },
        {"guard_pid": True},
    ):
        with pytest.raises(GuardAttestationError):
            _validate_attestation_window(valid | updates, now=now)
    missing = dict(valid)
    missing.pop("issued_monotonic_ns")
    with pytest.raises(GuardAttestationError):
        _validate_attestation_window(missing, now=now)


def _capture_arrays() -> dict[str, np.ndarray]:
    ids = np.broadcast_to(np.arange(WIDTH, dtype=np.int64), (ROWS, WIDTH)).copy()
    values = np.broadcast_to(
        np.linspace(1.0, -1.0, WIDTH, dtype=np.float32), (ROWS, WIDTH)
    ).copy()
    probs = np.zeros((ROWS, WIDTH), dtype=np.float32)
    probs[:, 0] = np.float32(0.75)
    probs[:, 1] = np.float32(0.25)
    uniforms = np.ldexp(
        (np.arange(ROWS, dtype=np.uint64) % np.uint64(1 << 20)).astype(np.float64),
        -53,
    )
    metadata = {
        "capture_kind": "diagnostic_pre_top_p_draft_choices",
        "driver_sha256": (
            "0ae20c7c4028cea83d9b9084d29067925d6dca08ff0ca2ce5a4ea9d73b9bb7d0"
        ),
        "expected_counters": {
            "accepted_drafts": 1656,
            "bonus_tokens": 342,
            "correction_tokens": 789,
            "drafted_tokens": ROWS,
            "verify_calls": 1146,
        },
        "float32_policy": "benchmark_experiment_only_not_retainable",
        "numpy_version": "2.4.4",
        "observed_counters": {
            "accepted_drafts": 1656,
            "bonus_tokens": 342,
            "correction_tokens": 789,
            "drafted_tokens": ROWS,
            "verify_calls": 1146,
        },
        "row_count": ROWS,
        "source_commit": CAPTURE_SOURCE_COMMIT,
        "support_width": WIDTH,
    }
    state_hash = np.full(ROWS, b"a" * 64, dtype="S64")
    return {
        "candidate_ids": ids,
        "candidate_values": values,
        "candidate_probs": probs,
        "uniforms": uniforms,
        "selected_tokens": np.zeros(ROWS, dtype=np.int64),
        "rng_pre_sha256": state_hash,
        "rng_post_sha256": state_hash.copy(),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def _write_capture(path: Path, **replacements: np.ndarray) -> str:
    arrays = _capture_arrays()
    arrays.update(replacements)
    np.savez(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_is_cpu_only_and_does_not_import_mlx(monkeypatch) -> None:
    import sys

    sys.modules.pop("scripts.pr391_metal_float32_selector_bench", None)
    before = set(sys.modules)
    module = importlib.import_module("scripts.pr391_metal_float32_selector_bench")
    added = set(sys.modules) - before

    assert "mlx" not in module.__dict__
    assert not any(name == "mlx" or name.startswith("mlx.") for name in added)
    assert not any(name.startswith("mtplx.") for name in added)


def test_capture_loader_hash_gates_and_accepts_exact_contract(tmp_path: Path) -> None:
    from scripts.pr391_metal_float32_selector_bench import load_capture

    path = tmp_path / "capture.npz"
    digest = _write_capture(path)

    capture = load_capture(
        path,
        expected_sha256=digest,
        expected_capture_source=CAPTURE_SOURCE_COMMIT,
        top_p=0.95,
    )

    assert capture.candidate_ids.shape == (ROWS, WIDTH)
    assert capture.candidate_ids.dtype == np.dtype(np.int64)
    assert capture.uniforms.dtype == np.dtype(np.float64)
    assert capture.metadata["row_count"] == ROWS
    assert capture.sha256 == digest


def test_capture_loader_rejects_hash_before_loading_npz(tmp_path: Path) -> None:
    from scripts.pr391_metal_float32_selector_bench import CaptureHashMismatch
    from scripts.pr391_metal_float32_selector_bench import load_capture

    path = tmp_path / "capture.npz"
    _write_capture(path)

    with pytest.raises(CaptureHashMismatch, match="SHA-256"):
        load_capture(
            path,
            expected_sha256="0" * 64,
            expected_capture_source=CAPTURE_SOURCE_COMMIT,
            top_p=0.95,
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"candidate_ids": np.zeros((ROWS, WIDTH), dtype=np.uint32)}, "int64"),
        ({"candidate_values": np.zeros((ROWS, WIDTH), dtype=np.float64)}, "float32"),
        ({"uniforms": np.zeros(ROWS - 1, dtype=np.float64)}, "3338"),
        (
            {"candidate_probs": np.full((ROWS, WIDTH), np.nan, dtype=np.float32)},
            "finite",
        ),
    ],
)
def test_capture_loader_rejects_schema_drift(
    tmp_path: Path, replacement: dict[str, np.ndarray], message: str
) -> None:
    from scripts.pr391_metal_float32_selector_bench import CaptureContractError
    from scripts.pr391_metal_float32_selector_bench import load_capture

    path = tmp_path / "capture.npz"
    digest = _write_capture(path, **replacement)
    with pytest.raises(CaptureContractError, match=message):
        load_capture(
            path,
            expected_sha256=digest,
            expected_capture_source=CAPTURE_SOURCE_COMMIT,
            top_p=0.95,
        )


def test_capture_loader_rejects_any_top_p_other_than_point_95(tmp_path: Path) -> None:
    from scripts.pr391_metal_float32_selector_bench import CaptureContractError
    from scripts.pr391_metal_float32_selector_bench import load_capture

    path = tmp_path / "capture.npz"
    digest = _write_capture(path)
    with pytest.raises(CaptureContractError, match="top_p"):
        load_capture(
            path,
            expected_sha256=digest,
            expected_capture_source=CAPTURE_SOURCE_COMMIT,
            top_p=0.9,
        )


def test_source_gate_requires_commit_and_both_reviewed_files(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.pr391_metal_float32_selector_bench as bench

    kernel = tmp_path / "mtplx/kernels/qwen4_frspec_k20_float32_choice.py"
    analyzer = tmp_path / "scripts/pr391_float32_choice_drift.py"
    helper = tmp_path / "scripts/pr391_metal_float32_selector_bench.py"
    kernel.parent.mkdir(parents=True)
    analyzer.parent.mkdir(parents=True)
    kernel.write_text("kernel\n")
    analyzer.write_text("analyzer\n")
    helper.write_text("helper\n")
    monkeypatch.setattr(bench, "_git_head", lambda source: BENCHMARK_SOURCE_COMMIT)
    expected = {
        str(kernel.relative_to(tmp_path)): hashlib.sha256(kernel.read_bytes()).hexdigest(),
        str(analyzer.relative_to(tmp_path)): hashlib.sha256(analyzer.read_bytes()).hexdigest(),
        str(helper.relative_to(tmp_path)): hashlib.sha256(helper.read_bytes()).hexdigest(),
    }

    receipt = bench.verify_source(tmp_path, BENCHMARK_SOURCE_COMMIT, expected)
    assert receipt["benchmark_commit"] == BENCHMARK_SOURCE_COMMIT
    assert receipt["files"] == expected

    with pytest.raises(bench.SourceContractError, match="required expected-file"):
        bench.verify_source(
            tmp_path,
            BENCHMARK_SOURCE_COMMIT,
            {
                str(kernel.relative_to(tmp_path)): expected[
                    str(kernel.relative_to(tmp_path))
                ]
            },
        )


def test_reviewed_capture_hash_is_not_caller_selectable() -> None:
    import scripts.pr391_metal_float32_selector_bench as bench

    bench.validate_reviewed_capture(
        expected_capture_sha256=CAPTURE_SHA,
    )
    with pytest.raises(bench.CaptureHashMismatch, match="reviewed capture"):
        bench.validate_reviewed_capture(
            expected_capture_sha256="1" * 64,
        )


def test_later_benchmark_head_is_distinct_from_capture_source_and_must_match(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.pr391_metal_float32_selector_bench as bench

    kernel = tmp_path / "mtplx/kernels/qwen4_frspec_k20_float32_choice.py"
    analyzer = tmp_path / "scripts/pr391_float32_choice_drift.py"
    helper = tmp_path / "scripts/pr391_metal_float32_selector_bench.py"
    for path in (kernel, analyzer, helper):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    expected_files = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (kernel, analyzer, helper)
    }
    monkeypatch.setattr(bench, "_git_head", lambda source: BENCHMARK_SOURCE_COMMIT)

    receipt = bench.verify_source(
        tmp_path,
        BENCHMARK_SOURCE_COMMIT,
        expected_files,
    )
    assert receipt["benchmark_commit"] == BENCHMARK_SOURCE_COMMIT
    assert BENCHMARK_SOURCE_COMMIT != CAPTURE_SOURCE_COMMIT
    with pytest.raises(bench.SourceContractError, match="source commit mismatch"):
        bench.verify_source(tmp_path, "1" * 40, expected_files)


class FakeMX:
    uint32 = np.uint32
    float32 = np.float32

    def __init__(self) -> None:
        self.compile_calls = 0
        self.eval_calls = 0

    @staticmethod
    def array(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def take(value, indices, *, axis):
        return np.take(value, np.asarray(indices, dtype=np.int64), axis=axis)

    def eval(self, *values):
        self.eval_calls += 1
        return values

    def compile(self, function):
        self.compile_calls += 1
        return function

    @staticmethod
    def reset_peak_memory():
        return None

    @staticmethod
    def get_active_memory():
        return 100

    @staticmethod
    def get_cache_memory():
        return 20

    @staticmethod
    def get_peak_memory():
        return 140


def _descriptor_builder(uniforms: np.ndarray) -> np.ndarray:
    rows = int(uniforms.shape[0])
    descriptor = np.zeros((rows, 5), dtype=np.uint32)
    descriptor[:, 2] = 1
    descriptor[:, 3] = np.asarray(-150, dtype=np.int32).view(np.uint32)
    return descriptor


def _expected_tokens(capture) -> SimpleNamespace:
    selected = np.zeros(capture.candidate_ids.shape[0], dtype=np.uint32)
    return SimpleNamespace(
        reduced_exact=selected.copy(),
        reduced_float32=selected.copy(),
    )


def _selector(ids, values, probs, descriptors):
    del descriptors
    selected = np.asarray(ids[:, 0], dtype=np.uint32)
    return selected, ids, values, probs


def test_cpu_fake_runs_all_conformance_and_timing_modes(tmp_path: Path) -> None:
    from scripts.pr391_metal_float32_selector_bench import load_capture
    from scripts.pr391_metal_float32_selector_bench import run_benchmark

    path = tmp_path / "capture.npz"
    digest = _write_capture(path)
    capture = load_capture(
        path,
        expected_sha256=digest,
        expected_capture_source=CAPTURE_SOURCE_COMMIT,
        top_p=0.95,
    )
    fake_mx = FakeMX()

    report = run_benchmark(
        mx=fake_mx,
        selector=_selector,
        descriptor_builder=_descriptor_builder,
        capture=capture,
        expected=_expected_tokens(capture),
        warmups=1,
        repeats=2,
    )

    assert report["status"] == "pass"
    assert report["conformance"]["rows_checked"] == ROWS
    assert report["conformance"]["selected_vs_reduced_float32"]["mismatches"] == 0
    assert report["conformance"]["raw_passthrough"]["ids_bit_exact"] is True
    assert report["conformance"]["raw_passthrough"]["values_bit_exact"] is True
    assert report["conformance"]["raw_passthrough"]["probs_bit_exact"] is True
    assert report["conformance"]["nonfinite_output_count"] == 0
    assert report["drift"]["reduced_float32_vs_reduced_exact"]["mismatches"] == 0
    assert report["timing"]["batch"]["rows_per_second"] > 0
    assert report["timing"]["b1"]["samples"] == 2
    assert report["timing"]["dependent_d3_single_eval"]["depth"] == 3
    assert report["timing"]["dependent_d3_single_eval"]["samples"] == 2
    assert fake_mx.compile_calls == 1
    assert report["memory"]["peak_bytes"] == 140
    assert report["memory"]["host_capture_bytes"] == sum(
        array.nbytes
        for array in (
            capture.candidate_ids,
            capture.candidate_values,
            capture.candidate_probs,
            capture.uniforms,
            capture.selected_tokens,
            capture.rng_pre_sha256,
            capture.rng_post_sha256,
        )
    )
    assert report["memory"]["device_input_static_bytes"] == (
        ROWS * WIDTH * (4 + 4 + 4) + ROWS * 5 * 4
    )
    assert report["memory"]["device_output_static_bytes"] == (
        ROWS * 4 + ROWS * WIDTH * (4 + 4 + 4)
    )


def test_hash_gated_module_import_rejects_preloaded_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    import scripts.pr391_metal_float32_selector_bench as bench

    name = "pr391_hash_gated_probe"
    path = tmp_path / f"{name}.py"
    path.write_text("VALUE = 7\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, name, raising=False)

    module = bench._import_hash_gated_module(tmp_path, name, path.name)
    assert module.VALUE == 7
    assert Path(module.__file__).resolve() == path.resolve()

    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(bench.SourceContractError, match="preloaded"):
        bench._import_hash_gated_module(tmp_path, name, path.name)


def test_metal_selection_drift_is_reported_and_fails_conformance(tmp_path: Path) -> None:
    from scripts.pr391_metal_float32_selector_bench import load_capture
    from scripts.pr391_metal_float32_selector_bench import run_benchmark

    path = tmp_path / "capture.npz"
    digest = _write_capture(path)
    capture = load_capture(
        path,
        expected_sha256=digest,
        expected_capture_source=CAPTURE_SOURCE_COMMIT,
        top_p=0.95,
    )

    def wrong_selector(ids, values, probs, descriptors):
        del descriptors
        selected = np.ones(ids.shape[0], dtype=np.uint32)
        return selected, ids, values, probs

    report = run_benchmark(
        mx=FakeMX(),
        selector=wrong_selector,
        descriptor_builder=_descriptor_builder,
        capture=capture,
        expected=_expected_tokens(capture),
        warmups=0,
        repeats=1,
    )

    assert report["status"] == "fail"
    mismatch = report["conformance"]["selected_vs_reduced_float32"]
    assert mismatch["mismatches"] == ROWS
    assert mismatch["indices"][:3] == [0, 1, 2]


def test_main_consumes_guard_before_source_or_gpu_loader(monkeypatch) -> None:
    import scripts.pr391_metal_float32_selector_bench as bench

    calls: list[str] = []

    def guard():
        calls.append("guard")
        raise bench.GuardAttestationError("stop")

    monkeypatch.setattr(bench, "consume_guard_attestation", guard)
    monkeypatch.setattr(
        bench,
        "verify_source",
        lambda *args, **kwargs: calls.append("source"),
    )
    monkeypatch.setattr(
        bench,
        "_load_gpu_api",
        lambda *args, **kwargs: calls.append("gpu"),
    )

    with pytest.raises(bench.GuardAttestationError, match="stop"):
        bench.main(
            [
                "--source",
                "/tmp/source",
                "--expected-benchmark-source",
                BENCHMARK_SOURCE_COMMIT,
                "--expected-file",
                "mtplx/kernels/qwen4_frspec_k20_float32_choice.py=" + "0" * 64,
                "--expected-file",
                "scripts/pr391_float32_choice_drift.py=" + "0" * 64,
                "--capture",
                "/tmp/capture.npz",
                "--expected-capture-sha256",
                CAPTURE_SHA,
            ]
        )

    assert calls == ["guard"]
