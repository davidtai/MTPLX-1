from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import ANY

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_issue51_qualification.py"
_SHA = "a" * 64
_COMMIT = "b" * 40
_CONTROL_COMMIT = "c" * 40


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "validate_issue51_qualification", _SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache(*, hits: int, misses: int) -> dict:
    lookups = hits + misses
    return {
        "expert_hits": hits,
        "expert_misses": misses,
        "expert_requests": lookups,
        "hit_rate": hits / lookups,
    }


def _mtp_metrics(k: int) -> dict:
    if k == 0:
        aggregate = {
            "accepted": None,
            "proposed": None,
            "acceptance_rate": None,
            "draft_accuracy": None,
        }
        by_depth = []
    else:
        aggregate = {
            "accepted": 80,
            "proposed": 100,
            "acceptance_rate": 0.8,
            "draft_accuracy": 0.8,
        }
        by_depth = [
            {
                "depth": depth,
                "accepted": 8,
                "proposed": 10,
                "acceptance_rate": 0.8,
                "draft_accuracy": 0.8,
            }
            for depth in range(1, k + 1)
        ]
    return {"aggregate": aggregate, "by_depth": by_depth}


def _observation(k: int) -> dict:
    return {
        "requested_depth": k,
        "effective_depth": k,
        "context_tokens": 1024,
        "prompt_tokens": 1024,
        "new_prefill_tokens": 1024,
        "generated_tokens": 1024,
        "finish_reason": "length",
        "ingestion_tok_s": 31.5,
        "prompt_target_prefill_time_s": 32.5,
        "decode_tok_s": 6.0,
        "qualification_metrics": {"mtp": _mtp_metrics(k)},
        "expert_streaming_counters": _cache(hits=900, misses=100),
        "expert_streaming_counters_by_phase": {"decode": _cache(hits=950, misses=50)},
        "expert_resource_telemetry": {
            "decode": {
                "reader_pool": {
                    "worker_capacity": 56,
                    "mean_active_readers": 1.25,
                    "peak_active_readers": 14,
                    "active_capacity_fraction": 1.25 / 56,
                    "peak_capacity_fraction": 14 / 56,
                    "full_capacity_fraction": 0.0,
                    "mean_queued_reads": 0.03,
                    "queued_reads_p95": 1.0,
                    "queued_reads_max": 4,
                    "queue_wait_p50_s": 0.0002,
                    "queue_wait_p95_s": 0.001,
                }
            }
        },
        "peak_memory_bytes": 96_000_000_000,
        "ar_comparison": {"observed_token_sha256": _SHA},
        "gates": {
            "committed_history": True,
            "compiled_verify_evidence": True,
            "decode_expert_cache_metrics": True,
            "effective_depth_exact": True,
            "final_state_contract": True,
            "generated_count_consistent": True,
            "guards_disabled": True,
            "length_finish": True,
            "new_prefill_tokens_exact": True,
            "output_tokens_exact": True,
            "prompt_length_exact": True,
            "requested_depth_exact": True,
            "speculative_event_contract": True,
        },
    }


def _valid_document() -> dict:
    qwen_state = {
        "loaded": True,
        "service": "qwen",
        "model": "Qwen3.6-27B",
        "health": "healthy",
    }
    return {
        "schema": "mtplx-q2-bf16-mtp-depth-matrix-v3",
        "status": "passed",
        "passed": True,
        "configuration": {
            "contexts": [1024],
            "output_tokens": 1024,
            "measurement_lane": "issue51-qualification-resource-instrumented",
            "candidate": {
                "verify_strategy": "capture_commit",
                "compiled_verify_mode": "off",
            },
            "generation": {
                "prefill_mode": "sustained",
                "mtp_history_policy": "committed",
                "mtp_cache_policy": "persistent",
            },
            "runtime": {"resource_telemetry": True},
        },
        "qualification": {
            "profile": "issue51-hy3-q2-1024x1024-k0-k7-v1",
            "candidate": {
                "id": "exact-r2",
                "commit": _COMMIT,
            },
            "control": {
                "id": "stock",
                "commit": _CONTROL_COMMIT,
            },
            "tested_commit": _COMMIT,
            "command": [
                "python",
                "scripts/benchmark_q2_mtp_depth_matrix.py",
                "--qualification-profile",
                "issue51",
            ],
            "environment": {
                "hardware": "MacBookPro",
                "os": "macOS",
                "python": "3.12",
                "mlx": "0.29",
                "model_artifact": "hy3-expert-q2@sha256:fixture",
            },
            "exclusive_lock": {
                "path": "/tmp/mtplx-gpu-exclusive.lock",
                "holder_pid": 12345,
                "receipt": "lock-receipt-12345",
                "acquired": True,
                "held_through_qwen_restore": True,
            },
            "qwen": {
                "before": qwen_state,
                "after": dict(qwen_state),
                "restored": True,
                "health_verified": True,
            },
        },
        "models": [
            {
                "model": "hy3-q2",
                "model_key": "hy3-expert-q2",
                "depths": list(range(1, 8)),
                "passed": True,
                "observations": [_observation(k) for k in range(8)],
            }
        ],
    }


def _validate(module, document: dict, *, expected: str = _SHA, actual: str = _SHA):
    return module.validate_document(
        document,
        artifact_path=Path("result.json"),
        artifact_url=(f"https://artifacts.example/issue51/{expected}/result.json"),
        expected_sha256=expected,
        actual_sha256=actual,
    )


def _parent_for_path(document: dict, path: str):
    target = document
    parts = path.replace("]", "").replace("[", ".").split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    return target, parts[-1]


def test_complete_issue51_artifact_qualifies() -> None:
    module = _load_script()

    report = module.validate_document(
        _valid_document(),
        artifact_path=Path("result.json"),
        artifact_url=(
            "https://artifacts.example/issue51/"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
            "result.json"
        ),
        expected_sha256=_SHA,
        actual_sha256=_SHA,
    )

    assert report["schema"] == "mtplx-issue51-qualification-validation-v1"
    assert report["qualified"] is True
    assert report["errors"] == []
    assert report["summary"]["validated_depths"] == list(range(8))
    assert report["artifact"]["sha256"] == _SHA


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("models[0].model", "other-model"),
        ("models[0].model_key", "other-key"),
        ("configuration.contexts", [2048]),
        ("configuration.output_tokens", 1028),
        ("configuration.generation.prefill_mode", "chunked"),
        ("configuration.candidate.verify_strategy", "serial"),
        ("configuration.generation.mtp_history_policy", "uncommitted"),
        ("configuration.runtime.resource_telemetry", False),
        ("models[0].depths", list(range(1, 7))),
    ],
)
def test_fixed_profile_configuration_fails_at_exact_path(
    path: str, value: object
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    target: object = document
    parts = path.replace("]", "").replace("[", ".").split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = value

    report = _validate(module, document)

    assert report["qualified"] is False
    assert report["errors"] == [
        {
            "path": path,
            "code": "invalid_value",
            "message": ANY,
        }
    ]


def test_missing_required_field_is_reported_without_exception() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    del document["configuration"]["contexts"]

    report = _validate(module, document)

    assert report["qualified"] is False
    assert report["errors"] == [
        {
            "path": "configuration.contexts",
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_observation_matrix_requires_exactly_one_k0_through_k7(case: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    observations = document["models"][0]["observations"]
    if case == "missing":
        observations.pop(4)
    else:
        observations[7]["requested_depth"] = 3

    report = _validate(module, document)

    assert report["qualified"] is False
    assert report["errors"] == [
        {
            "path": "models[0].observations",
            "code": "invalid_matrix",
            "message": "require exactly one row for each K=0..7",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_tokens", 2048),
        ("prompt_tokens", 1023),
        ("new_prefill_tokens", 1023),
        ("generated_tokens", 1028),
        ("effective_depth", 2),
        ("finish_reason", "stop"),
    ],
)
def test_each_k_row_enforces_input_output_and_depth(field: str, value: object) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3][field] = value

    report = _validate(module, document)

    path = f"models[0].observations[3].{field}"
    assert report["qualified"] is False
    assert report["errors"] == [
        {
            "path": path,
            "code": "invalid_value",
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    "field_path",
    [
        ("ingestion_tok_s",),
        ("prompt_target_prefill_time_s",),
        ("decode_tok_s",),
        ("peak_memory_bytes",),
        ("ar_comparison", "observed_token_sha256"),
    ],
)
def test_required_performance_and_output_fields_cannot_be_missing(
    field_path: tuple[str, ...],
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    target = document["models"][0]["observations"][3]
    for key in field_path[:-1]:
        target = target[key]
    del target[field_path[-1]]

    report = _validate(module, document)

    suffix = ".".join(field_path)
    path = f"models[0].observations[3].{suffix}"
    assert report["qualified"] is False
    assert report["errors"] == [
        {
            "path": path,
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize(
    ("field_path", "value", "code"),
    [
        (("ingestion_tok_s",), 0.0, "invalid_value"),
        (("prompt_target_prefill_time_s",), float("nan"), "invalid_value"),
        (("decode_tok_s",), "6.0", "invalid_type"),
        (("peak_memory_bytes",), 1.5, "invalid_type"),
        (
            ("ar_comparison", "observed_token_sha256"),
            "A" * 64,
            "invalid_value",
        ),
    ],
)
def test_performance_memory_and_hash_values_are_validated(
    field_path: tuple[str, ...], value: object, code: str
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    target = document["models"][0]["observations"][3]
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value

    report = _validate(module, document)

    suffix = ".".join(field_path)
    assert report["errors"] == [
        {
            "path": f"models[0].observations[3].{suffix}",
            "code": code,
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    "field_path",
    [
        ("aggregate", "accepted"),
        ("aggregate", "proposed"),
        ("aggregate", "acceptance_rate"),
        ("aggregate", "draft_accuracy"),
        ("by_depth",),
    ],
)
def test_mtp_metrics_are_required_at_precise_paths(field_path: tuple[str, ...]) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    target = document["models"][0]["observations"][3]["qualification_metrics"]["mtp"]
    for key in field_path[:-1]:
        target = target[key]
    del target[field_path[-1]]

    report = _validate(module, document)

    suffix = ".".join(field_path)
    assert report["errors"] == [
        {
            "path": (f"models[0].observations[3].qualification_metrics.mtp.{suffix}"),
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize(
    "field", ["accepted", "proposed", "acceptance_rate", "draft_accuracy"]
)
def test_k0_mtp_metrics_require_explicit_null_not_zero(field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][0]["qualification_metrics"]["mtp"][
        "aggregate"
    ][field] = 0

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                f"models[0].observations[0].qualification_metrics.mtp.aggregate.{field}"
            ),
            "code": "na_required",
            "message": "K0 MTP metric must be explicit null, not zero",
        }
    ]


def test_k0_mtp_by_depth_must_be_an_explicit_empty_list() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][0]["qualification_metrics"]["mtp"][
        "by_depth"
    ] = [_mtp_metrics(1)["by_depth"][0]]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": ("models[0].observations[0].qualification_metrics.mtp.by_depth"),
            "code": "invalid_matrix",
            "message": "K0 requires an empty by-depth MTP list",
        }
    ]


def test_mtp_by_depth_requires_exactly_depths_one_through_k() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3]["qualification_metrics"]["mtp"][
        "by_depth"
    ].pop()

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": ("models[0].observations[3].qualification_metrics.mtp.by_depth"),
            "code": "invalid_matrix",
            "message": "require exactly one MTP metric for each depth 1..K",
        }
    ]


@pytest.mark.parametrize("field", ["acceptance_rate", "draft_accuracy"])
def test_mtp_rates_must_match_accepted_over_proposed(field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3]["qualification_metrics"]["mtp"][
        "aggregate"
    ][field] = 0.7

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                f"models[0].observations[3].qualification_metrics.mtp.aggregate.{field}"
            ),
            "code": "inconsistent",
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("whole_run", "expert_hits"),
        ("whole_run", "expert_misses"),
        ("whole_run", "expert_requests"),
        ("whole_run", "hit_rate"),
        ("decode", "expert_hits"),
        ("decode", "expert_misses"),
        ("decode", "expert_requests"),
        ("decode", "hit_rate"),
    ],
)
def test_cache_counts_and_rates_are_required(scope: str, field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    row = document["models"][0]["observations"][3]
    if scope == "whole_run":
        cache = row["expert_streaming_counters"]
        prefix = "expert_streaming_counters"
    else:
        cache = row["expert_streaming_counters_by_phase"]["decode"]
        prefix = "expert_streaming_counters_by_phase.decode"
    del cache[field]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"models[0].observations[3].{prefix}.{field}",
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("expert_hits", -1, "invalid_value"),
        ("expert_misses", 1.5, "invalid_type"),
        ("expert_requests", 999, "inconsistent"),
        ("hit_rate", 0.5, "inconsistent"),
    ],
)
def test_cache_metrics_are_numeric_and_internally_consistent(
    field: str, value: object, code: str
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3]["expert_streaming_counters"][field] = value

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (f"models[0].observations[3].expert_streaming_counters.{field}"),
            "code": code,
            "message": ANY,
        }
    ]


_READER_FIELDS = [
    "worker_capacity",
    "mean_active_readers",
    "peak_active_readers",
    "active_capacity_fraction",
    "peak_capacity_fraction",
    "full_capacity_fraction",
    "mean_queued_reads",
    "queued_reads_p95",
    "queued_reads_max",
    "queue_wait_p50_s",
    "queue_wait_p95_s",
]


@pytest.mark.parametrize("field", _READER_FIELDS)
def test_reader_saturation_and_queue_fields_are_required(field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    pool = document["models"][0]["observations"][3]["expert_resource_telemetry"][
        "decode"
    ]["reader_pool"]
    del pool[field]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                "models[0].observations[3].expert_resource_telemetry"
                f".decode.reader_pool.{field}"
            ),
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("full_capacity_fraction", 1.1, "invalid_value"),
        ("mean_queued_reads", -0.1, "invalid_value"),
        ("queued_reads_max", 1.5, "invalid_type"),
        ("queued_reads_p95", 5.0, "invalid_value"),
        ("queue_wait_p50_s", 0.002, "invalid_value"),
    ],
)
def test_reader_saturation_and_queue_values_are_validated(
    field: str, value: object, code: str
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3]["expert_resource_telemetry"]["decode"][
        "reader_pool"
    ][field] = value

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                "models[0].observations[3].expert_resource_telemetry"
                f".decode.reader_pool.{field}"
            ),
            "code": code,
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_capacity_fraction", 0.5),
        ("peak_capacity_fraction", 0.5),
    ],
)
def test_reader_saturation_fractions_match_configured_capacity(
    field: str, value: float
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3]["expert_resource_telemetry"]["decode"][
        "reader_pool"
    ][field] = value

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                "models[0].observations[3].expert_resource_telemetry"
                f".decode.reader_pool.{field}"
            ),
            "code": "inconsistent",
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("schema", "old-schema"),
        ("status", "failed"),
        ("passed", False),
        ("configuration.measurement_lane", "uncontrolled"),
        ("configuration.generation.mtp_cache_policy", "reset"),
        ("qualification.profile", "other-profile"),
        ("models[0].passed", False),
    ],
)
def test_qualification_status_and_execution_mode_are_fixed(
    path: str, value: object
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    parent, key = _parent_for_path(document, path)
    parent[key] = value

    report = _validate(module, document)

    assert report["errors"] == [{"path": path, "code": "invalid_value", "message": ANY}]


_REQUIRED_GATES = list(_observation(1)["gates"])


@pytest.mark.parametrize(
    ("case", "gate"), [("missing", _REQUIRED_GATES[0]), ("false", _REQUIRED_GATES[-1])]
)
def test_every_correctness_gate_must_be_present_and_true(case: str, gate: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    gates = document["models"][0]["observations"][3]["gates"]
    if case == "missing":
        del gates[gate]
        code = "missing"
    else:
        gates[gate] = False
        code = "invalid_value"

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"models[0].observations[3].gates.{gate}",
            "code": code,
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    "path",
    [
        "qualification.candidate.id",
        "qualification.candidate.commit",
        "qualification.control.id",
        "qualification.control.commit",
        "qualification.tested_commit",
        "qualification.command",
        "qualification.environment.hardware",
        "qualification.environment.os",
        "qualification.environment.python",
        "qualification.environment.mlx",
        "qualification.environment.model_artifact",
    ],
)
def test_candidate_control_command_and_environment_are_required(path: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    parent, key = _parent_for_path(document, path)
    del parent[key]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": path,
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        ("qualification.candidate.id", "", "invalid_value"),
        ("qualification.control.id", "exact-r2", "not_distinct"),
        ("qualification.candidate.commit", "deadbeef", "invalid_value"),
        ("qualification.control.commit", "C" * 40, "invalid_value"),
        ("qualification.tested_commit", _CONTROL_COMMIT, "inconsistent"),
        ("qualification.command", [], "invalid_value"),
        ("qualification.command", ["python", ""], "invalid_value"),
        ("qualification.environment.hardware", "", "invalid_value"),
    ],
)
def test_candidate_control_and_reproduction_identity_are_validated(
    path: str, value: object, code: str
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    parent, key = _parent_for_path(document, path)
    parent[key] = value

    report = _validate(module, document)

    assert report["errors"] == [{"path": path, "code": code, "message": ANY}]


@pytest.mark.parametrize(
    "field",
    [
        "path",
        "holder_pid",
        "receipt",
        "acquired",
        "held_through_qwen_restore",
    ],
)
def test_exclusive_lock_receipt_and_holder_are_required(field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    del document["qualification"]["exclusive_lock"][field]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"qualification.exclusive_lock.{field}",
            "code": "missing",
            "message": "required field is missing",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("path", "/tmp/other.lock", "invalid_value"),
        ("holder_pid", 0, "invalid_value"),
        ("receipt", "", "invalid_value"),
        ("acquired", False, "invalid_value"),
        ("held_through_qwen_restore", False, "invalid_value"),
    ],
)
def test_exclusive_lock_is_validated(field: str, value: object, code: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["qualification"]["exclusive_lock"][field] = value

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"qualification.exclusive_lock.{field}",
            "code": code,
            "message": ANY,
        }
    ]


@pytest.mark.parametrize("field", ["before", "after", "restored", "health_verified"])
def test_qwen_restore_and_health_evidence_is_required(field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    del document["qualification"]["qwen"][field]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"qualification.qwen.{field}",
            "code": "missing",
            "message": "required field is missing",
        }
    ]


def test_qwen_post_state_must_exactly_match_pre_state() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["qualification"]["qwen"]["after"]["model"] = "other"

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": "qualification.qwen.after",
            "code": "not_restored",
            "message": "post-benchmark Qwen state must exactly match pre-state",
        }
    ]


@pytest.mark.parametrize("field", ["restored", "health_verified"])
def test_qwen_restore_and_health_flags_must_be_true(field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["qualification"]["qwen"][field] = False

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"qualification.qwen.{field}",
            "code": "invalid_value",
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(
    ("scope", "field", "value", "code"),
    [
        ("aggregate", "accepted", -1, "invalid_value"),
        ("aggregate", "proposed", 0, "invalid_value"),
        ("by_depth", "accepted", -1, "invalid_value"),
        ("by_depth", "proposed", 0, "invalid_value"),
        ("by_depth", "accepted", 11, "invalid_value"),
    ],
)
def test_aggregate_and_per_depth_mtp_counts_are_validated(
    scope: str, field: str, value: object, code: str
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    mtp = document["models"][0]["observations"][3]["qualification_metrics"]["mtp"]
    if scope == "aggregate":
        mtp["aggregate"][field] = value
        suffix = f"aggregate.{field}"
    else:
        mtp["by_depth"][0][field] = value
        suffix = f"by_depth[0].{field}"

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (f"models[0].observations[3].qualification_metrics.mtp.{suffix}"),
            "code": code,
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(("case", "value"), [("missing", None), ("typed", "3")])
def test_requested_depth_is_fail_closed_without_exceptions(
    case: str, value: object
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    row = document["models"][0]["observations"][3]
    if case == "missing":
        del row["requested_depth"]
        code = "missing"
    else:
        row["requested_depth"] = value
        code = "invalid_type"

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": "models[0].observations[3].requested_depth",
            "code": code,
            "message": ANY,
        }
    ]


def test_artifact_must_contain_exactly_one_hy3_q2_model() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"].append(copy.deepcopy(document["models"][0]))

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": "models",
            "code": "invalid_matrix",
            "message": "require exactly one Hy3-Q2 model result",
        }
    ]


@pytest.mark.parametrize(
    ("expected", "actual", "url", "path", "code"),
    [
        (_SHA, "b" * 64, None, "artifact.sha256", "sha_mismatch"),
        ("A" * 64, _SHA, None, "artifact.expected_sha256", "invalid_value"),
        (_SHA, "xyz", None, "artifact.sha256", "invalid_value"),
        (
            _SHA,
            _SHA,
            "http://artifacts.example/result.json",
            "artifact.url",
            "immutable_url",
        ),
        (
            _SHA,
            _SHA,
            "https://artifacts.example/result.json",
            "artifact.url",
            "immutable_url",
        ),
    ],
)
def test_artifact_link_and_sha_are_immutable_and_matching(
    expected: str,
    actual: str,
    url: str | None,
    path: str,
    code: str,
) -> None:
    module = _load_script()
    artifact_url = url or f"https://artifacts.example/{expected}/result.json"

    report = module.validate_document(
        _valid_document(),
        artifact_path=Path("result.json"),
        artifact_url=artifact_url,
        expected_sha256=expected,
        actual_sha256=actual,
    )

    assert report["errors"] == [{"path": path, "code": code, "message": ANY}]


def test_cli_writes_a_qualified_machine_readable_report(tmp_path: Path) -> None:
    module = _load_script()
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps(_valid_document()), encoding="utf-8")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report_path = tmp_path / "validation.json"

    exit_code = module.main(
        [
            str(artifact),
            "--artifact-url",
            f"https://artifacts.example/{sha256}/result.json",
            "--expected-sha256",
            sha256,
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["qualified"] is True
    assert report["artifact"]["sha256"] == sha256
    assert report["summary"]["validated_depths"] == list(range(8))
    assert report_path.read_bytes().endswith(b"\n")


def test_cli_returns_one_and_reports_invalid_json(tmp_path: Path) -> None:
    module = _load_script()
    artifact = tmp_path / "broken.json"
    artifact.write_text("{broken", encoding="utf-8")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report_path = tmp_path / "validation.json"

    exit_code = module.main(
        [
            str(artifact),
            "--artifact-url",
            f"https://artifacts.example/{sha256}/broken.json",
            "--expected-sha256",
            sha256,
            "--report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["qualified"] is False
    assert report["errors"] == [{"path": "$", "code": "invalid_json", "message": ANY}]


@pytest.mark.parametrize(
    "field", ["depth", "accepted", "proposed", "acceptance_rate", "draft_accuracy"]
)
def test_each_per_depth_mtp_field_is_required_at_its_exact_path(
    field: str,
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    del document["models"][0]["observations"][3]["qualification_metrics"]["mtp"][
        "by_depth"
    ][1][field]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                "models[0].observations[3].qualification_metrics"
                f".mtp.by_depth[1].{field}"
            ),
            "code": "missing",
            "message": "required field is missing",
        }
    ]


def test_per_depth_mtp_rate_is_internally_consistent() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    document["models"][0]["observations"][3]["qualification_metrics"]["mtp"][
        "by_depth"
    ][1]["draft_accuracy"] = 0.7

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": (
                "models[0].observations[3].qualification_metrics"
                ".mtp.by_depth[1].draft_accuracy"
            ),
            "code": "inconsistent",
            "message": ANY,
        }
    ]


@pytest.mark.parametrize(("side", "field"), [("before", "model"), ("after", "health")])
def test_qwen_state_fields_are_required_at_precise_paths(side: str, field: str) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    del document["qualification"]["qwen"][side][field]

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": f"qualification.qwen.{side}.{field}",
            "code": "missing",
            "message": "required field is missing",
        }
    ]


def test_qwen_must_be_loaded_and_healthy_before_and_after() -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    for side in ("before", "after"):
        document["qualification"]["qwen"][side]["health"] = "unhealthy"

    report = _validate(module, document)

    assert report["errors"] == [
        {
            "path": "qualification.qwen.before.health",
            "code": "invalid_value",
            "message": ANY,
        },
        {
            "path": "qualification.qwen.after.health",
            "code": "invalid_value",
            "message": ANY,
        },
    ]


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        ("qualification.control.commit", _COMMIT, "not_distinct"),
        ("qualification.command", ["python", "other.py"], "invalid_value"),
    ],
)
def test_control_commit_and_benchmark_command_are_distinct_and_reproducible(
    path: str, value: object, code: str
) -> None:
    module = _load_script()
    document = copy.deepcopy(_valid_document())
    parent, key = _parent_for_path(document, path)
    parent[key] = value

    report = _validate(module, document)

    assert report["errors"] == [{"path": path, "code": code, "message": ANY}]
