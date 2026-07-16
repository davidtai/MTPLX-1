from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "hy3_router_last_arrival_timing.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "hy3_router_last_arrival_timing_benchmark",
        _SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_correctness() -> dict[str, bool]:
    return {
        "ids_exact": True,
        "route_weights_bitwise_exact": True,
        "control_route_weights_finite": True,
        "candidate_route_weights_finite": True,
        "control_route_weights_normalized": True,
        "candidate_route_weights_normalized": True,
        "control_route_weights_in_range": True,
        "candidate_route_weights_in_range": True,
        "control_repeated_deterministic": True,
        "candidate_repeated_deterministic": True,
    }


def _valid_provenance(module) -> dict[str, object]:
    return {
        "commit": {
            "head": "a" * 40,
            "dirty": False,
            "status_lines": [],
            "capture_error": None,
        },
        "sources": {
            path: {"sha256": "b" * 64, "bytes": 1}
            for path in module.REQUIRED_PROVENANCE_SOURCES
        },
        "command": {
            "argv": ["python", "benchmark.py", "--output-json", "/tmp/x.json"],
            "shell": "python benchmark.py --output-json /tmp/x.json",
            "cwd": "/repo",
        },
        "host": {
            "platform": "macOS-26.2-arm64",
            "python": "3.12.0",
            "python_executable": "/usr/bin/python3",
        },
        "libraries": {"mlx": "0.29.0", "numpy": "2.3.0"},
        "device": {
            "qualified": True,
            "system": "Darwin",
            "macos_version": "26.2",
            "architecture": "applegpu_g17s",
        },
    }


def _passing_timing() -> dict[str, object]:
    return {
        "bootstrap_speed_ratio_95_ci": [1.01, 1.08],
        "raw_pairs": [
            {
                "repeat": index,
                "order": (
                    ["control", "candidate"]
                    if index % 2 == 0
                    else ["candidate", "control"]
                ),
            }
            for index in range(60)
        ],
        "abba_blocks": 30,
        "bootstrap_unit": "complete ABBA block (2 paired repeats)",
        "bootstrap_resamples": 10_000,
        "percentile_method": "numpy.quantile(method='linear')",
    }


def test_pair_order_is_abba_balanced() -> None:
    module = _load_script()

    observed = [module.paired_arm_order(repeat) for repeat in range(6)]

    assert observed == [
        ("control", "candidate"),
        ("candidate", "control"),
        ("control", "candidate"),
        ("candidate", "control"),
        ("control", "candidate"),
        ("candidate", "control"),
    ]


def test_measurement_interleaves_and_synchronizes_both_arms_through_one_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    calls: list[str] = []
    clock = iter(index * 1_000_000 for index in range(16))

    def arm(name: str):
        def execute():
            calls.append(name)

        return execute

    monkeypatch.setattr(module, "_execute_route", lambda function: function())
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(clock))

    observed = module._measure_pairs(
        {"control": arm("control"), "candidate": arm("candidate")},
        warmups=2,
        repeats=4,
    )

    assert calls == [
        "control",
        "candidate",
        "candidate",
        "control",
        "control",
        "candidate",
        "candidate",
        "control",
        "control",
        "candidate",
        "candidate",
        "control",
    ]
    assert [pair["order"] for pair in observed] == [
        ["control", "candidate"],
        ["candidate", "control"],
        ["control", "candidate"],
        ["candidate", "control"],
    ]
    assert all(pair["control_ms"] == 1.0 for pair in observed)
    assert all(pair["candidate_ms"] == 1.0 for pair in observed)


@pytest.mark.parametrize(("warmups", "repeats"), ((1, 2), (2, 3)))
def test_measurement_rejects_unbalanced_abba_blocks(
    warmups: int,
    repeats: int,
) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="even ABBA blocks"):
        module._measure_pairs(
            {"control": lambda: None, "candidate": lambda: None},
            warmups=warmups,
            repeats=repeats,
        )


def test_paired_statistics_preserve_raw_samples_and_report_speed_ratio() -> None:
    module = _load_script()
    samples = tuple(zip((2.0, 2.2, 1.8, 2.0), (1.0, 1.1, 0.9, 1.0), strict=True))
    pairs = [
        {
            "repeat": index,
            "order": list(module.paired_arm_order(index)),
            "control_ms": control,
            "candidate_ms": candidate,
        }
        for index, (control, candidate) in enumerate(samples * 15)
    ]

    observed = module.paired_timing_statistics(
        pairs,
        bootstrap_resamples=10_000,
        bootstrap_seed=58,
    )

    assert observed["raw_pairs"] == pairs
    assert observed["control"]["mean_ms"] == pytest.approx(2.0)
    assert observed["control"]["median_ms"] == pytest.approx(2.0)
    assert observed["control"]["p95_ms"] == pytest.approx(2.2)
    assert observed["candidate"]["mean_ms"] == pytest.approx(1.0)
    assert observed["speed_ratio_control_over_candidate"] == pytest.approx(2.0)
    assert observed["paired_speed_ratio_median"] == pytest.approx(2.0)
    assert observed["bootstrap_speed_ratio_95_ci"][0] > 1.0
    assert observed["percentile_method"] == "numpy.quantile(method='linear')"
    assert observed["abba_blocks"] == 30
    assert observed["bootstrap_unit"] == "complete ABBA block (2 paired repeats)"


def test_paired_statistics_reject_underpowered_confidence_intervals() -> None:
    module = _load_script()
    pairs = [
        {
            "repeat": index,
            "order": list(module.paired_arm_order(index)),
            "control_ms": 2.0,
            "candidate_ms": 1.0,
        }
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="ABBA blocks"):
        module.paired_timing_statistics(
            pairs,
            bootstrap_resamples=module.MIN_BOOTSTRAP_RESAMPLES,
            bootstrap_seed=58,
        )

    enough_pairs = [
        {
            "repeat": index,
            "order": list(module.paired_arm_order(index)),
            "control_ms": 2.0,
            "candidate_ms": 1.0,
        }
        for index in range(module.MIN_PAIRED_REPEATS)
    ]
    with pytest.raises(ValueError, match="bootstrap resamples"):
        module.paired_timing_statistics(
            enough_pairs,
            bootstrap_resamples=module.MIN_BOOTSTRAP_RESAMPLES - 1,
            bootstrap_seed=58,
        )


def test_router_arms_use_actual_issue59_r41_precise_g6_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    control_calls: list[dict[str, object]] = []
    candidate_calls: list[dict[str, object]] = []
    sentinel_ids = object()
    sentinel_weights = object()

    def control_route(*args, **kwargs):
        control_calls.append(kwargs)
        return sentinel_ids, sentinel_weights

    def candidate_route(*args, **kwargs):
        candidate_calls.append(kwargs)
        return SimpleNamespace(
            expert_ids=sentinel_ids,
            route_weights=sentinel_weights,
        )

    monkeypatch.setattr(module, "hy3_router_fp32_route", control_route)
    monkeypatch.setattr(module, "hy3_router_last_arrival_route", candidate_route)

    epochs = [object(), object()]
    allocation_calls: list[int] = []

    class EpochBlock:
        def __getitem__(self, index: int):
            return epochs[index]

    def allocate_epochs(slots: int):
        allocation_calls.append(slots)
        return EpochBlock()

    materialized: list[object] = []
    monkeypatch.setattr(
        module,
        "new_hy3_router_forward_epoch",
        allocate_epochs,
        raising=False,
    )
    monkeypatch.setattr(module.mx, "eval", lambda value: materialized.append(value))
    monkeypatch.setattr(module.mx, "synchronize", lambda: None)

    arms = module._router_arms(
        object(),
        object(),
        object(),
        candidate_invocations=2,
    )

    assert arms["control"]() == (sentinel_ids, sentinel_weights)
    assert arms["candidate"]() == (sentinel_ids, sentinel_weights)
    assert arms["candidate"]() == (sentinel_ids, sentinel_weights)
    assert allocation_calls == [2]
    assert len(materialized) == 1
    assert control_calls == [
        {
            "n_tile": 16,
            "grid_k_parts": 16,
            "operand_mode": "grouped-direct",
            "simd_groups_per_threadgroup": 4,
            "top_k": 8,
            "route_norm": True,
            "scaling_factor": 2.826,
            "finalizer_mode": "simd",
            "simd_groups": 6,
            "sigmoid_mode": "precise",
        }
    ]
    assert candidate_calls == [
        {
            "epoch": epochs[0],
            "top_k": 8,
            "route_norm": True,
            "scaling_factor": 2.826,
            "sigmoid_mode": "precise",
        },
        {
            "epoch": epochs[1],
            "top_k": 8,
            "route_norm": True,
            "scaling_factor": 2.826,
            "sigmoid_mode": "precise",
        },
    ]


def test_router_arms_fail_loudly_if_explicit_epoch_api_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module,
        "new_hy3_router_forward_epoch",
        None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="explicit forward-epoch API"):
        module._router_arms(
            object(),
            object(),
            object(),
            candidate_invocations=2,
        )


def test_config_names_implementations_without_claiming_runtime_selectors() -> None:
    module = _load_script()
    args = module._parser().parse_args(
        ["--output-json", "/tmp/result.json", "--warmups", "4", "--repeats", "60"]
    )

    config = module._config(args)

    assert config["control"]["arm"] == module.CONTROL_ARM
    assert config["control"]["issue59_candidate"] == (
        "n16_p16_sg4_grouped_direct_precise_g6"
    )
    assert config["control"]["finalizer_simd_groups"] == 6
    assert config["candidate"]["arm"] == module.CANDIDATE_ARM
    assert "selector" not in config["control"]
    assert "selector" not in config["candidate"]


def test_route_weight_parity_is_bitwise_not_numeric_equality() -> None:
    module = _load_script()
    positive_zero = np.asarray([0.0, 1.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0, 1.0], dtype=np.float32)

    assert np.array_equal(positive_zero, negative_zero)
    assert not module._weights_bitwise_equal(positive_zero, negative_zero)


@pytest.mark.parametrize(
    ("weights", "finite", "normalized", "in_range"),
    (
        (
            np.full((1, 1, 8), 2.826 / 8, dtype=np.float32),
            True,
            True,
            True,
        ),
        (
            np.full((1, 1, 8), 0.1, dtype=np.float32),
            True,
            False,
            True,
        ),
        (
            np.asarray(
                [
                    [
                        [
                            -0.1,
                            0.8065,
                            0.35325,
                            0.35325,
                            0.35325,
                            0.35325,
                            0.35325,
                            0.35325,
                        ]
                    ]
                ],
                dtype=np.float32,
            ),
            True,
            True,
            False,
        ),
        (
            np.asarray(
                [[[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                dtype=np.float32,
            ),
            False,
            False,
            False,
        ),
    ),
)
def test_route_weight_health_requires_finite_normalized_in_range_values(
    weights: np.ndarray,
    finite: bool,
    normalized: bool,
    in_range: bool,
) -> None:
    module = _load_script()

    observed = module._route_weight_health(weights)

    assert observed["finite"] is finite
    assert observed["normalized"] is normalized
    assert observed["in_range"] is in_range


def test_correctness_gate_rejects_bitwise_identical_nan_route_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    ids = np.zeros((1, 4, 8), dtype=np.int32)
    weights = np.full((1, 4, 8), 2.826 / 8, dtype=np.float32)
    weights[0, 0, 0] = np.nan
    monkeypatch.setattr(module, "_execute_route", lambda function: function())

    observed = module._correctness(
        lambda: (ids.copy(), weights.copy()),
        lambda: (ids.copy(), weights.copy()),
    )

    assert observed["ids_exact"] is True
    assert observed["route_weights_bitwise_exact"] is True
    assert observed["control_route_weights_finite"] is False
    assert observed["candidate_route_weights_finite"] is False
    assert observed["control_route_weights_normalized"] is False
    assert observed["candidate_route_weights_normalized"] is False
    assert observed["control_route_weights_in_range"] is False
    assert observed["candidate_route_weights_in_range"] is False
    assert (
        module.candidate_advances_to_full_matrix(
            correctness=observed,
            bootstrap_speed_ratio_95_ci=(1.01, 1.10),
        )
        is False
    )


@pytest.mark.parametrize(
    ("correctness", "interval", "expected"),
    (
        (
            _passing_correctness(),
            (1.01, 1.10),
            True,
        ),
        (
            {**_passing_correctness(), "ids_exact": False},
            (1.01, 1.10),
            False,
        ),
        (
            {**_passing_correctness(), "route_weights_bitwise_exact": False},
            (1.01, 1.10),
            False,
        ),
        (
            {
                **_passing_correctness(),
                "candidate_repeated_deterministic": False,
            },
            (1.01, 1.10),
            False,
        ),
        (
            _passing_correctness(),
            (0.99, 1.10),
            False,
        ),
    ),
)
def test_full_matrix_advance_policy_requires_parity_health_determinism_and_speed(
    correctness: dict[str, bool],
    interval: tuple[float, float],
    expected: bool,
) -> None:
    module = _load_script()

    assert (
        module.candidate_advances_to_full_matrix(
            correctness=correctness,
            bootstrap_speed_ratio_95_ci=interval,
        )
        is expected
    )


def test_parser_requires_checkpoint_and_exposes_reproducibility_controls() -> None:
    module = _load_script()
    parser = module._parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(
        [
            "--output-json",
            "/tmp/issue58-router.json",
            "--warmups",
            "4",
            "--repeats",
            "60",
            "--activation-seed",
            "5804",
            "--bootstrap-seed",
            "5859",
            "--bootstrap-resamples",
            "10000",
        ]
    )

    assert args.output_json == Path("/tmp/issue58-router.json")
    assert args.warmups == 4
    assert args.repeats == 60
    assert args.activation_seed == 5804
    assert args.bootstrap_seed == 5859
    assert args.bootstrap_resamples == 10_000
    assert str(args.lock_path) == "/tmp/mtplx-gpu-exclusive.lock"


def test_parser_rejects_nonpositive_measurement_counts() -> None:
    module = _load_script()
    parser = module._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--output-json", "/tmp/result.json", "--repeats", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--output-json", "/tmp/result.json", "--bootstrap-resamples", "-1"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-json", "/tmp/result.json", "--warmups", "3"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-json", "/tmp/result.json", "--repeats", "7"])


def test_parser_rejects_statistically_underpowered_evidence_counts() -> None:
    module = _load_script()
    parser = module._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-json",
                "/tmp/result.json",
                "--repeats",
                str(module.MIN_PAIRED_REPEATS - 2),
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-json",
                "/tmp/result.json",
                "--bootstrap-resamples",
                str(module.MIN_BOOTSTRAP_RESAMPLES - 1),
            ]
        )


def test_result_record_separates_execution_status_from_full_matrix_advance() -> None:
    module = _load_script()
    correctness = _passing_correctness()
    timing = _passing_timing()

    observed = module.completed_result(
        correctness=correctness,
        timing=timing,
        provenance=_valid_provenance(module),
        config={"repeats": 60},
    )

    assert observed["schema"] == "mtplx-issue58-router-paired-timing-v3"
    assert observed["status"] == "complete"
    assert observed["advance_to_full_matrix"]["passed"] is True
    assert observed["advance_to_full_matrix"]["decision"] == "advance"
    assert observed["correctness"] is correctness
    assert observed["timing"] is timing
    assert observed["timing_gate"] == {"passed": True, "failures": []}
    assert observed["provenance_gate"] == {"passed": True, "failures": []}
    assert observed["provenance"]["commit"]["head"] == "a" * 40


def test_result_only_advances_to_full_matrix_without_claiming_later_gates() -> None:
    module = _load_script()

    observed = module.completed_result(
        correctness=_passing_correctness(),
        timing=_passing_timing(),
        provenance=_valid_provenance(module),
        config={"repeats": module.MIN_PAIRED_REPEATS},
    )

    assert "promotion" not in observed
    assert observed["advance_to_full_matrix"] == {
        "passed": True,
        "decision": "advance",
        "scope": "authorizes the subsequent full validation matrix only",
        "requirements": observed["advance_to_full_matrix"]["requirements"],
        "not_claimed": list(module.NOT_CLAIMED_GATES),
    }
    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"promotion"' not in source
    assert '"decision": "promote"' not in source


def test_full_matrix_advance_fails_closed_for_unbalanced_timing_record() -> None:
    module = _load_script()
    timing = _passing_timing()
    timing["raw_pairs"] = timing["raw_pairs"][:1]

    observed = module.completed_result(
        correctness=_passing_correctness(),
        timing=timing,
        provenance=_valid_provenance(module),
        config={"repeats": 1},
    )

    assert observed["advance_to_full_matrix"]["passed"] is False
    assert observed["advance_to_full_matrix"]["decision"] == "hold"
    assert "unbalanced_abba_pairs" in observed["timing_gate"]["failures"]


def test_timing_gate_rejects_underpowered_abba_or_bootstrap_evidence() -> None:
    module = _load_script()
    timing = _passing_timing()
    timing["raw_pairs"] = timing["raw_pairs"][:2]
    timing["abba_blocks"] = 1
    timing["bootstrap_resamples"] = module.MIN_BOOTSTRAP_RESAMPLES - 1

    observed = module.timing_gate(timing)

    assert observed["passed"] is False
    assert "insufficient_abba_blocks" in observed["failures"]
    assert "insufficient_bootstrap_resamples" in observed["failures"]


def test_full_matrix_advance_fails_closed_for_malformed_pair_metadata() -> None:
    module = _load_script()
    timing = _passing_timing()
    timing["raw_pairs"][0] = {"repeat": "bad", "order": None}

    observed = module.completed_result(
        correctness=_passing_correctness(),
        timing=timing,
        provenance=_valid_provenance(module),
        config={"repeats": 60},
    )

    assert observed["advance_to_full_matrix"]["passed"] is False
    assert observed["advance_to_full_matrix"]["decision"] == "hold"
    assert "invalid_pair_order:0" in observed["timing_gate"]["failures"]


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    (
        (lambda provenance: provenance["commit"].update(dirty=True), "dirty_worktree"),
        (
            lambda provenance: provenance["commit"].update(head="unavailable"),
            "invalid_commit_head",
        ),
        (
            lambda provenance: provenance["libraries"].update(mlx="unavailable"),
            "missing_mlx_version",
        ),
        (
            lambda provenance: provenance["device"].update(qualified=False),
            "unqualified_device",
        ),
    ),
)
def test_full_matrix_advance_fails_closed_when_provenance_is_not_qualified(
    mutation,
    expected_failure: str,
) -> None:
    module = _load_script()
    provenance = _valid_provenance(module)
    mutation(provenance)

    observed = module.completed_result(
        correctness=_passing_correctness(),
        timing=_passing_timing(),
        provenance=provenance,
        config={"repeats": 60},
    )

    assert observed["advance_to_full_matrix"]["passed"] is False
    assert observed["advance_to_full_matrix"]["decision"] == "hold"
    assert expected_failure in observed["provenance_gate"]["failures"]


def test_collect_provenance_records_mlx_and_numpy_versions() -> None:
    module = _load_script()

    observed = module.collect_provenance(["benchmark.py", "--output-json", "/tmp/x"])

    assert observed["libraries"]["mlx"] not in {"", "unavailable"}
    assert observed["libraries"]["numpy"] == np.__version__


def test_correctness_failure_rejects_before_any_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    correctness = {**_passing_correctness(), "ids_exact": False}
    monkeypatch.setattr(
        module,
        "_load_router",
        lambda *args, **kwargs: (object(), object(), {"router": "metadata"}),
    )
    monkeypatch.setattr(
        module,
        "_activation",
        lambda *args, **kwargs: (object(), {"activation": "metadata"}),
    )
    monkeypatch.setattr(
        module,
        "_router_arms",
        lambda *args, **kwargs: {"control": object(), "candidate": object()},
    )
    monkeypatch.setattr(module, "_correctness", lambda *args: correctness)
    monkeypatch.setattr(
        module,
        "_measure_pairs",
        lambda *args, **kwargs: pytest.fail("timing must not run after rejection"),
    )

    observed = module.run_benchmark(
        model=Path("/model"),
        layer=1,
        warmups=4,
        repeats=module.MIN_PAIRED_REPEATS,
        activation_seed=1,
        bootstrap_seed=2,
        bootstrap_resamples=module.MIN_BOOTSTRAP_RESAMPLES,
        provenance=_valid_provenance(module),
        config={"measurement": {}},
        device={"qualified": True},
    )

    assert observed["status"] == "rejected"
    assert observed["rejection"] == {
        "phase": "correctness",
        "reasons": ["ids_exact"],
    }
    assert observed["timing"] == {
        "status": "not_run",
        "reason": "correctness_or_determinism_gate_failed",
    }
    assert observed["advance_to_full_matrix"]["passed"] is False
    assert observed["advance_to_full_matrix"]["decision"] == "hold"


@pytest.mark.parametrize(
    ("system", "macos", "architecture", "capability", "message"),
    (
        ("Linux", "", "applegpu_g17s", True, "requires Darwin"),
        ("Darwin", "26.1", "applegpu_g17s", True, "macOS 26.2"),
        ("Darwin", "26.2", "applegpu_g16", True, "G17"),
        ("Darwin", "26.2", "applegpu_g17s", False, "capability probe"),
    ),
)
def test_qualified_device_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    macos: str,
    architecture: str,
    capability: bool,
    message: str,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module.platform, "system", lambda: system)
    monkeypatch.setattr(module.platform, "mac_ver", lambda: (macos, (), ""))
    monkeypatch.setattr(
        module.mx,
        "device_info",
        lambda: {"architecture": architecture},
    )
    monkeypatch.setattr(module.mx.metal, "device_info", lambda: {"name": "M5 Max"})
    monkeypatch.setattr(module, "hy3_router_fp32_available", lambda: capability)

    with pytest.raises(RuntimeError, match=message):
        module._require_qualified_device()


def test_qualified_device_gate_records_real_device_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "mac_ver", lambda: ("26.2.1", (), ""))
    monkeypatch.setattr(
        module.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g17s"},
    )
    monkeypatch.setattr(
        module.mx.metal,
        "device_info",
        lambda: {"name": "Apple M5 Max"},
    )
    monkeypatch.setattr(module, "hy3_router_fp32_available", lambda: True)

    observed = module._require_qualified_device()

    assert observed["qualified"] is True
    assert observed["system"] == "Darwin"
    assert observed["macos_version"] == "26.2.1"
    assert observed["architecture"] == "applegpu_g17s"
    assert observed["metal"]["name"] == "Apple M5 Max"


def test_main_gates_device_inside_window_and_audits_successful_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    state = {"inside": False}

    @contextmanager
    def fake_window(**kwargs):
        audit = kwargs["audit"]
        audit.update(
            {
                "lock_acquired": True,
                "qwen_state_captured": True,
                "qwen_loaded_before": True,
                "qwen_models_before": ["qwen"],
                "qwen_restore_verified": False,
                "lock_release_verified": False,
            }
        )
        state["inside"] = True
        try:
            yield SimpleNamespace(
                lock_path=Path("/tmp/mtplx-gpu-exclusive.lock"),
                qwen_state=SimpleNamespace(loaded=True, models=("qwen",)),
            )
        finally:
            state["inside"] = False
            audit["qwen_restore_verified"] = True
            audit["lock_release_verified"] = True

    def qualified_device():
        assert state["inside"] is True
        return {"qualified": True}

    def run_benchmark(**kwargs):
        assert state["inside"] is True
        assert kwargs["device"] == {"qualified": True}
        return {
            "schema": module.SCHEMA,
            "status": "complete",
            "advance_to_full_matrix": {"passed": False, "decision": "hold"},
        }

    monkeypatch.setattr(module, "_audited_exclusive_mlx_window", fake_window)
    monkeypatch.setattr(module, "_require_qualified_device", qualified_device)
    monkeypatch.setattr(module, "run_benchmark", run_benchmark)
    monkeypatch.setattr(module, "collect_provenance", lambda argv: {"p": 1})
    monkeypatch.setattr(
        module,
        "finalize_result_provenance",
        lambda result, final_capture: result,
    )
    destination = tmp_path / "result.json"

    exit_code = module.main(
        [
            "--output-json",
            str(destination),
            "--warmups",
            "4",
            "--repeats",
            str(module.MIN_PAIRED_REPEATS),
        ]
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert saved["exclusive_window"]["lock_acquired"] is True
    assert saved["exclusive_window"]["qwen_state_captured"] is True
    assert saved["exclusive_window"]["qwen_restore_verified"] is True
    assert saved["exclusive_window"]["lock_release_verified"] is True


def test_main_interrupt_artifact_is_written_after_restoration_and_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    state = {"inside": False, "exited": False}

    @contextmanager
    def fake_window(**kwargs):
        audit = kwargs["audit"]
        audit.update(
            {
                "lock_acquired": True,
                "qwen_state_captured": True,
                "qwen_loaded_before": False,
                "qwen_models_before": [],
                "qwen_restore_verified": False,
                "lock_release_verified": False,
            }
        )
        state["inside"] = True
        try:
            yield SimpleNamespace(
                lock_path=Path("/tmp/mtplx-gpu-exclusive.lock"),
                qwen_state=SimpleNamespace(loaded=False, models=()),
            )
        finally:
            state["inside"] = False
            state["exited"] = True
            audit["qwen_restore_verified"] = True
            audit["lock_release_verified"] = True

    def interrupted(**kwargs):
        assert state["inside"] is True
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_audited_exclusive_mlx_window", fake_window)
    monkeypatch.setattr(
        module,
        "_require_qualified_device",
        lambda: {"qualified": True},
    )
    monkeypatch.setattr(module, "run_benchmark", interrupted)
    monkeypatch.setattr(module, "collect_provenance", lambda argv: {"p": 1})
    destination = tmp_path / "result.json"

    exit_code = module.main(
        [
            "--output-json",
            str(destination),
            "--warmups",
            "4",
            "--repeats",
            str(module.MIN_PAIRED_REPEATS),
        ]
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert state["exited"] is True
    assert exit_code == 130
    assert saved["status"] == "interrupted"
    assert saved["failure"]["error_type"] == "KeyboardInterrupt"
    assert saved["exclusive_window"]["qwen_restore_verified"] is True
    assert saved["exclusive_window"]["lock_release_verified"] is True


def test_main_guard_exit_failure_does_not_claim_restoration_or_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()

    @contextmanager
    def failing_exit(**kwargs):
        audit = kwargs["audit"]
        audit.update(
            {
                "lock_acquired": True,
                "qwen_state_captured": True,
                "qwen_loaded_before": True,
                "qwen_models_before": ["qwen"],
                "qwen_restore_verified": False,
                "lock_release_verified": False,
            }
        )
        yield SimpleNamespace(
            lock_path=Path("/tmp/mtplx-gpu-exclusive.lock"),
            qwen_state=SimpleNamespace(loaded=True, models=("qwen",)),
        )
        raise RuntimeError("restore verification failed")

    monkeypatch.setattr(module, "_audited_exclusive_mlx_window", failing_exit)
    monkeypatch.setattr(
        module,
        "_require_qualified_device",
        lambda: {"qualified": True},
    )
    monkeypatch.setattr(
        module,
        "run_benchmark",
        lambda **kwargs: {
            "schema": module.SCHEMA,
            "status": "complete",
            "advance_to_full_matrix": {"passed": False, "decision": "hold"},
        },
    )
    monkeypatch.setattr(module, "collect_provenance", lambda argv: {"p": 1})
    destination = tmp_path / "result.json"

    exit_code = module.main(
        [
            "--output-json",
            str(destination),
            "--warmups",
            "4",
            "--repeats",
            str(module.MIN_PAIRED_REPEATS),
        ]
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert saved["status"] == "failed"
    assert saved["failure"]["error"] == "restore verification failed"
    assert saved["exclusive_window"]["lock_acquired"] is True
    assert saved["exclusive_window"]["qwen_restore_verified"] is False
    assert saved["exclusive_window"]["lock_release_verified"] is False


def test_main_reports_lock_acquired_and_released_on_pre_yield_qwen_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()

    @contextmanager
    def fake_lock(**kwargs):
        yield Path(kwargs["lock_path"])

    @contextmanager
    def fail_before_qwen_yield(**kwargs):
        raise RuntimeError("Qwen stop failed after lock acquisition")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "_exclusive_mlx_lock", fake_lock, raising=False)
    monkeypatch.setattr(
        module,
        "qwen_stopped_for_mlx",
        fail_before_qwen_yield,
        raising=False,
    )
    monkeypatch.setattr(module, "collect_provenance", lambda argv: {"p": 1})
    destination = tmp_path / "result.json"

    exit_code = module.main(
        [
            "--output-json",
            str(destination),
            "--warmups",
            "4",
            "--repeats",
            str(module.MIN_PAIRED_REPEATS),
        ]
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert saved["failure"]["error"] == ("Qwen stop failed after lock acquisition")
    assert saved["exclusive_window"]["lock_acquired"] is True
    assert saved["exclusive_window"]["qwen_state_captured"] is False
    assert saved["exclusive_window"]["qwen_restore_verified"] is False
    assert saved["exclusive_window"]["lock_release_verified"] is True


def test_main_recomputes_advance_from_provenance_captured_after_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script()
    destination = tmp_path / "result.json"
    captures: list[tuple[bool, str | None]] = []
    initial = _valid_provenance(module)
    final = _valid_provenance(module)
    final["commit"]["head"] = "f" * 40

    def collect(argv):
        status = None
        if destination.exists():
            status = json.loads(destination.read_text(encoding="utf-8")).get("status")
        captures.append((destination.exists(), status))
        return initial if len(captures) == 1 else final

    @contextmanager
    def fake_window(**kwargs):
        audit = kwargs["audit"]
        audit.update(
            {
                "lock_acquired": True,
                "qwen_state_captured": True,
                "qwen_loaded_before": True,
                "qwen_models_before": ["qwen"],
                "qwen_restore_verified": True,
                "lock_release_verified": True,
            }
        )
        yield SimpleNamespace(
            lock_path=Path("/tmp/mtplx-gpu-exclusive.lock"),
            qwen_state=SimpleNamespace(loaded=True, models=("qwen",)),
        )

    def run_benchmark(**kwargs):
        return module.completed_result(
            correctness=_passing_correctness(),
            timing=_passing_timing(),
            provenance={**kwargs["provenance"], "device": kwargs["device"]},
            config=kwargs["config"],
        )

    monkeypatch.setattr(module, "collect_provenance", collect)
    monkeypatch.setattr(module, "_audited_exclusive_mlx_window", fake_window)
    monkeypatch.setattr(
        module,
        "_require_qualified_device",
        lambda: final["device"],
    )
    monkeypatch.setattr(module, "run_benchmark", run_benchmark)

    exit_code = module.main(
        [
            "--output-json",
            str(destination),
            "--warmups",
            "4",
            "--repeats",
            str(module.MIN_PAIRED_REPEATS),
        ]
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert captures == [(False, None), (True, "complete")]
    assert saved["provenance"]["commit"]["head"] == "f" * 40
    assert saved["advance_to_full_matrix"]["passed"] is True


def test_atomic_checkpoint_persists_structured_failure(tmp_path: Path) -> None:
    module = _load_script()
    destination = tmp_path / "result.json"
    failure = module.failure_result(
        phase="measure",
        error=RuntimeError("Metal dispatch failed"),
        provenance={"commit": {"head": "abc"}},
        config={"repeats": 8},
    )

    module.write_json_checkpoint(destination, failure)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["failure"] == {
        "phase": "measure",
        "error_type": "RuntimeError",
        "error": "Metal dispatch failed",
    }
    assert not list(tmp_path.glob("*.tmp"))
