#!/usr/bin/env python3
"""Fail-closed validator for Issue 51 Hy3-Q2 qualification artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPORT_SCHEMA = "mtplx-issue51-qualification-validation-v1"
_MISSING = object()


def validate_document(
    document: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_url: str,
    expected_sha256: str,
    actual_sha256: str,
) -> dict[str, Any]:
    """Return a machine-readable Issue 51 qualification report."""
    errors: list[dict[str, str]] = []

    def add_error(path: str, code: str, message: str) -> None:
        errors.append({"path": path, "code": code, "message": message})

    def read(path: str, *keys: str | int) -> Any:
        value: Any = document
        for key in keys:
            if isinstance(key, int):
                if not isinstance(value, list) or key >= len(value):
                    break
                value = value[key]
            else:
                if not isinstance(value, dict) or key not in value:
                    break
                value = value[key]
        else:
            return value
        add_error(path, "missing", "required field is missing")
        return _MISSING

    def require_equal(path: str, value: Any, expected: Any) -> None:
        if value is _MISSING:
            return
        if value != expected:
            add_error(
                path,
                "invalid_value",
                f"expected {expected!r}; received {value!r}",
            )

    def require_positive_number(path: str, value: Any) -> None:
        if value is _MISSING:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            add_error(path, "invalid_type", "expected a finite number")
        elif not math.isfinite(value) or value <= 0:
            add_error(path, "invalid_value", "expected a finite number > 0")

    def require_positive_integer(path: str, value: Any) -> None:
        if value is _MISSING:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            add_error(path, "invalid_type", "expected a positive integer")
        elif value <= 0:
            add_error(path, "invalid_value", "expected a positive integer")

    def require_sha256(path: str, value: Any) -> None:
        if value is _MISSING:
            return
        if not isinstance(value, str):
            add_error(path, "invalid_type", "expected a SHA-256 string")
        elif re.fullmatch(r"[0-9a-f]{64}", value) is None:
            add_error(path, "invalid_value", "expected 64 lowercase hex digits")

    def require_nonempty_string(path: str, value: Any) -> bool:
        if value is _MISSING:
            return False
        if not isinstance(value, str):
            add_error(path, "invalid_type", "expected a non-empty string")
            return False
        if not value.strip():
            add_error(path, "invalid_value", "string cannot be empty")
            return False
        return True

    def require_commit(path: str, value: Any) -> bool:
        if value is _MISSING:
            return False
        if not isinstance(value, str):
            add_error(path, "invalid_type", "expected a Git commit hash")
            return False
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            add_error(path, "invalid_value", "expected 40 lowercase hex digits")
            return False
        return True

    def validate_mtp_metrics(index: int, k: int) -> None:
        base = f"models[0].observations[{index}].qualification_metrics.mtp"
        aggregate: dict[str, Any] = {}
        for field in (
            "accepted",
            "proposed",
            "acceptance_rate",
            "draft_accuracy",
        ):
            path = f"{base}.aggregate.{field}"
            aggregate[field] = read(
                path,
                "models",
                0,
                "observations",
                index,
                "qualification_metrics",
                "mtp",
                "aggregate",
                field,
            )
        by_depth = read(
            f"{base}.by_depth",
            "models",
            0,
            "observations",
            index,
            "qualification_metrics",
            "mtp",
            "by_depth",
        )
        if k == 0:
            for field, value in aggregate.items():
                if value is not _MISSING and value is not None:
                    add_error(
                        f"{base}.aggregate.{field}",
                        "na_required",
                        "K0 MTP metric must be explicit null, not zero",
                    )
            if by_depth is not _MISSING and by_depth != []:
                add_error(
                    f"{base}.by_depth",
                    "invalid_matrix",
                    "K0 requires an empty by-depth MTP list",
                )
            return

        accepted = aggregate["accepted"]
        proposed = aggregate["proposed"]
        for field in ("accepted", "proposed"):
            value = aggregate[field]
            path = f"{base}.aggregate.{field}"
            if value is _MISSING:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                add_error(path, "invalid_type", "expected an integer count")
            elif value < 0 or (field == "proposed" and value == 0):
                add_error(path, "invalid_value", "expected a valid MTP count")
        if (
            isinstance(accepted, int)
            and not isinstance(accepted, bool)
            and isinstance(proposed, int)
            and not isinstance(proposed, bool)
            and proposed > 0
            and accepted >= 0
        ):
            if accepted > proposed:
                add_error(
                    f"{base}.aggregate.accepted",
                    "invalid_value",
                    "accepted cannot exceed proposed",
                )
            else:
                expected_rate = accepted / proposed
                for field in ("acceptance_rate", "draft_accuracy"):
                    value = aggregate[field]
                    path = f"{base}.aggregate.{field}"
                    if value is _MISSING:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        add_error(path, "invalid_type", "expected a rate")
                    elif not math.isfinite(value) or not 0 <= value <= 1:
                        add_error(path, "invalid_value", "rate must be in [0, 1]")
                    elif not math.isclose(
                        value, expected_rate, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        add_error(
                            path,
                            "inconsistent",
                            "rate does not equal accepted / proposed",
                        )

        if by_depth is _MISSING:
            return
        if not isinstance(by_depth, list):
            add_error(
                f"{base}.by_depth",
                "invalid_matrix",
                "require exactly one MTP metric for each depth 1..K",
            )
            return
        depth_values: list[int] = []
        depth_values_valid = True
        for depth_index, item in enumerate(by_depth):
            item_path = f"{base}.by_depth[{depth_index}]"
            if not isinstance(item, dict):
                add_error(
                    item_path,
                    "invalid_type",
                    "expected a per-depth MTP object",
                )
                depth_values_valid = False
                continue
            if "depth" not in item:
                add_error(
                    f"{item_path}.depth",
                    "missing",
                    "required field is missing",
                )
                depth_values_valid = False
            elif isinstance(item["depth"], bool) or not isinstance(item["depth"], int):
                add_error(
                    f"{item_path}.depth",
                    "invalid_type",
                    "depth must be an integer",
                )
                depth_values_valid = False
            else:
                depth_values.append(item["depth"])
        if not depth_values_valid:
            return
        if depth_values != list(range(1, k + 1)):
            add_error(
                f"{base}.by_depth",
                "invalid_matrix",
                "require exactly one MTP metric for each depth 1..K",
            )
            return
        for depth_index, item in enumerate(by_depth):
            item_base = f"{base}.by_depth[{depth_index}]"
            item_accepted = item.get("accepted", _MISSING)
            item_proposed = item.get("proposed", _MISSING)
            for field in (
                "accepted",
                "proposed",
                "acceptance_rate",
                "draft_accuracy",
            ):
                if field not in item:
                    add_error(
                        f"{item_base}.{field}",
                        "missing",
                        "required field is missing",
                    )
            counts_valid = True
            for field, value in (
                ("accepted", item_accepted),
                ("proposed", item_proposed),
            ):
                path = f"{item_base}.{field}"
                if value is _MISSING:
                    counts_valid = False
                elif isinstance(value, bool) or not isinstance(value, int):
                    add_error(path, "invalid_type", "expected an integer count")
                    counts_valid = False
                elif value < 0 or (field == "proposed" and value == 0):
                    add_error(path, "invalid_value", "expected a valid MTP count")
                    counts_valid = False
            if counts_valid and item_accepted > item_proposed:
                add_error(
                    f"{item_base}.accepted",
                    "invalid_value",
                    "accepted cannot exceed proposed",
                )
                counts_valid = False
            if (
                counts_valid
                and isinstance(item_accepted, int)
                and not isinstance(item_accepted, bool)
                and isinstance(item_proposed, int)
                and not isinstance(item_proposed, bool)
                and item_proposed > 0
                and 0 <= item_accepted <= item_proposed
            ):
                expected_rate = item_accepted / item_proposed
                for field in ("acceptance_rate", "draft_accuracy"):
                    value = item.get(field, _MISSING)
                    if value is _MISSING:
                        continue
                    path = f"{item_base}.{field}"
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        add_error(path, "invalid_type", "expected a rate")
                    elif not math.isfinite(value) or not 0 <= value <= 1:
                        add_error(path, "invalid_value", "rate must be in [0, 1]")
                    elif not math.isclose(
                        value, expected_rate, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        add_error(
                            path,
                            "inconsistent",
                            "rate does not equal accepted / proposed",
                        )

    def validate_cache(index: int, keys: tuple[str, ...], label: str) -> None:
        base = f"models[0].observations[{index}].{label}"
        values: dict[str, Any] = {}
        for field in (
            "expert_hits",
            "expert_misses",
            "expert_requests",
            "hit_rate",
        ):
            values[field] = read(
                f"{base}.{field}",
                "models",
                0,
                "observations",
                index,
                *keys,
                field,
            )
        valid_counts = True
        for field in ("expert_hits", "expert_misses", "expert_requests"):
            value = values[field]
            path = f"{base}.{field}"
            if value is _MISSING:
                valid_counts = False
            elif isinstance(value, bool) or not isinstance(value, int):
                add_error(path, "invalid_type", "expected an integer count")
                valid_counts = False
            elif value < 0:
                add_error(path, "invalid_value", "count cannot be negative")
                valid_counts = False
        rate = values["hit_rate"]
        valid_rate = True
        if rate is _MISSING:
            valid_rate = False
        elif isinstance(rate, bool) or not isinstance(rate, (int, float)):
            add_error(f"{base}.hit_rate", "invalid_type", "expected a rate")
            valid_rate = False
        elif not math.isfinite(rate) or not 0 <= rate <= 1:
            add_error(f"{base}.hit_rate", "invalid_value", "rate must be in [0, 1]")
            valid_rate = False
        if not valid_counts:
            return
        expected_requests = values["expert_hits"] + values["expert_misses"]
        if values["expert_requests"] != expected_requests:
            add_error(
                f"{base}.expert_requests",
                "inconsistent",
                "expert_requests must equal expert_hits + expert_misses",
            )
            return
        if expected_requests == 0:
            add_error(
                f"{base}.expert_requests",
                "invalid_value",
                "cache lookup count must be positive",
            )
        elif valid_rate and not math.isclose(
            rate,
            values["expert_hits"] / expected_requests,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            add_error(
                f"{base}.hit_rate",
                "inconsistent",
                "hit_rate must equal expert_hits / expert_requests",
            )

    def validate_readers(index: int) -> None:
        keys = (
            "expert_resource_telemetry",
            "decode",
            "reader_pool",
        )
        base = (
            f"models[0].observations[{index}]"
            ".expert_resource_telemetry.decode.reader_pool"
        )
        fields = (
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
        )
        values = {
            field: read(
                f"{base}.{field}",
                "models",
                0,
                "observations",
                index,
                *keys,
                field,
            )
            for field in fields
        }

        capacity = values["worker_capacity"]
        valid_capacity = True
        if capacity is _MISSING:
            valid_capacity = False
        elif isinstance(capacity, bool) or not isinstance(capacity, int):
            add_error(
                f"{base}.worker_capacity",
                "invalid_type",
                "expected a positive integer",
            )
            valid_capacity = False
        elif capacity <= 0:
            add_error(
                f"{base}.worker_capacity",
                "invalid_value",
                "configured reader capacity must be positive",
            )
            valid_capacity = False

        numeric_valid: dict[str, bool] = {}
        for field in (
            "mean_active_readers",
            "active_capacity_fraction",
            "peak_capacity_fraction",
            "full_capacity_fraction",
            "mean_queued_reads",
            "queued_reads_p95",
            "queue_wait_p50_s",
            "queue_wait_p95_s",
        ):
            value = values[field]
            numeric_valid[field] = True
            if value is _MISSING:
                numeric_valid[field] = False
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                add_error(f"{base}.{field}", "invalid_type", "expected a number")
                numeric_valid[field] = False
            elif not math.isfinite(value) or value < 0:
                add_error(
                    f"{base}.{field}",
                    "invalid_value",
                    "expected a finite nonnegative number",
                )
                numeric_valid[field] = False

        for field in ("peak_active_readers", "queued_reads_max"):
            value = values[field]
            numeric_valid[field] = True
            if value is _MISSING:
                numeric_valid[field] = False
            elif isinstance(value, bool) or not isinstance(value, int):
                add_error(
                    f"{base}.{field}",
                    "invalid_type",
                    "expected a nonnegative integer",
                )
                numeric_valid[field] = False
            elif value < 0:
                add_error(
                    f"{base}.{field}",
                    "invalid_value",
                    "expected a nonnegative integer",
                )
                numeric_valid[field] = False

        for field in (
            "active_capacity_fraction",
            "peak_capacity_fraction",
            "full_capacity_fraction",
        ):
            if numeric_valid[field] and values[field] > 1:
                add_error(
                    f"{base}.{field}",
                    "invalid_value",
                    "saturation fraction must be in [0, 1]",
                )
                numeric_valid[field] = False

        if valid_capacity and numeric_valid["mean_active_readers"]:
            if values["mean_active_readers"] > capacity:
                add_error(
                    f"{base}.mean_active_readers",
                    "invalid_value",
                    "mean active readers cannot exceed capacity",
                )
            elif numeric_valid["active_capacity_fraction"] and not math.isclose(
                values["active_capacity_fraction"],
                values["mean_active_readers"] / capacity,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                add_error(
                    f"{base}.active_capacity_fraction",
                    "inconsistent",
                    "fraction must equal mean_active_readers / worker_capacity",
                )
        if valid_capacity and numeric_valid["peak_active_readers"]:
            if values["peak_active_readers"] > capacity:
                add_error(
                    f"{base}.peak_active_readers",
                    "invalid_value",
                    "peak active readers cannot exceed capacity",
                )
            elif numeric_valid["peak_capacity_fraction"] and not math.isclose(
                values["peak_capacity_fraction"],
                values["peak_active_readers"] / capacity,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                add_error(
                    f"{base}.peak_capacity_fraction",
                    "inconsistent",
                    "fraction must equal peak_active_readers / worker_capacity",
                )
        if (
            numeric_valid["queued_reads_p95"]
            and numeric_valid["queued_reads_max"]
            and values["queued_reads_p95"] > values["queued_reads_max"]
        ):
            add_error(
                f"{base}.queued_reads_p95",
                "invalid_value",
                "queued-read p95 cannot exceed the maximum",
            )
        if (
            numeric_valid["queue_wait_p50_s"]
            and numeric_valid["queue_wait_p95_s"]
            and values["queue_wait_p50_s"] > values["queue_wait_p95_s"]
        ):
            add_error(
                f"{base}.queue_wait_p50_s",
                "invalid_value",
                "queue-wait p50 cannot exceed p95",
            )

    def validate_identity_and_environment() -> None:
        candidate_id = read(
            "qualification.candidate.id", "qualification", "candidate", "id"
        )
        control_id = read("qualification.control.id", "qualification", "control", "id")
        candidate_id_valid = require_nonempty_string(
            "qualification.candidate.id", candidate_id
        )
        control_id_valid = require_nonempty_string(
            "qualification.control.id", control_id
        )
        if candidate_id_valid and control_id_valid and candidate_id == control_id:
            add_error(
                "qualification.control.id",
                "not_distinct",
                "candidate and control identities must differ",
            )

        candidate_commit = read(
            "qualification.candidate.commit",
            "qualification",
            "candidate",
            "commit",
        )
        control_commit = read(
            "qualification.control.commit",
            "qualification",
            "control",
            "commit",
        )
        tested_commit = read(
            "qualification.tested_commit", "qualification", "tested_commit"
        )
        candidate_commit_valid = require_commit(
            "qualification.candidate.commit", candidate_commit
        )
        control_commit_valid = require_commit(
            "qualification.control.commit", control_commit
        )
        tested_commit_valid = require_commit(
            "qualification.tested_commit", tested_commit
        )
        if (
            candidate_commit_valid
            and tested_commit_valid
            and tested_commit != candidate_commit
        ):
            add_error(
                "qualification.tested_commit",
                "inconsistent",
                "tested_commit must equal the candidate commit",
            )
        if (
            candidate_commit_valid
            and control_commit_valid
            and control_commit == candidate_commit
        ):
            add_error(
                "qualification.control.commit",
                "not_distinct",
                "candidate and control commits must differ",
            )

        command = read("qualification.command", "qualification", "command")
        if command is not _MISSING:
            command_valid = (
                isinstance(command, list)
                and bool(command)
                and all(isinstance(arg, str) and bool(arg) for arg in command)
            )
            if not command_valid:
                add_error(
                    "qualification.command",
                    "invalid_value",
                    "command must be a non-empty argv list with no empty arguments",
                )
            elif "scripts/benchmark_q2_mtp_depth_matrix.py" not in command:
                add_error(
                    "qualification.command",
                    "invalid_value",
                    "command must invoke the Issue 51 qualification benchmark",
                )
        for field in ("hardware", "os", "python", "mlx", "model_artifact"):
            path = f"qualification.environment.{field}"
            require_nonempty_string(
                path,
                read(path, "qualification", "environment", field),
            )

    def validate_lock_and_qwen() -> None:
        require_equal(
            "qualification.exclusive_lock.path",
            read(
                "qualification.exclusive_lock.path",
                "qualification",
                "exclusive_lock",
                "path",
            ),
            "/tmp/mtplx-gpu-exclusive.lock",
        )
        holder = read(
            "qualification.exclusive_lock.holder_pid",
            "qualification",
            "exclusive_lock",
            "holder_pid",
        )
        require_positive_integer("qualification.exclusive_lock.holder_pid", holder)
        receipt = read(
            "qualification.exclusive_lock.receipt",
            "qualification",
            "exclusive_lock",
            "receipt",
        )
        require_nonempty_string("qualification.exclusive_lock.receipt", receipt)
        for field in ("acquired", "held_through_qwen_restore"):
            path = f"qualification.exclusive_lock.{field}"
            require_equal(
                path,
                read(path, "qualification", "exclusive_lock", field),
                True,
            )

        before = read("qualification.qwen.before", "qualification", "qwen", "before")
        after = read("qualification.qwen.after", "qualification", "qwen", "after")
        for field in ("restored", "health_verified"):
            path = f"qualification.qwen.{field}"
            require_equal(
                path,
                read(path, "qualification", "qwen", field),
                True,
            )

        def validate_qwen_state(side: str, state: Any) -> bool:
            if state is _MISSING:
                return False
            if not isinstance(state, dict):
                add_error(
                    f"qualification.qwen.{side}",
                    "invalid_type",
                    "expected a Qwen state object",
                )
                return False
            valid = True
            for field in ("loaded", "service", "model", "health"):
                if field not in state:
                    add_error(
                        f"qualification.qwen.{side}.{field}",
                        "missing",
                        "required field is missing",
                    )
                    valid = False
            if not valid:
                return False
            expected_values = {
                "loaded": True,
                "service": "qwen",
                "health": "healthy",
            }
            for field, expected in expected_values.items():
                if state[field] != expected:
                    add_error(
                        f"qualification.qwen.{side}.{field}",
                        "invalid_value",
                        f"expected {expected!r}; received {state[field]!r}",
                    )
                    valid = False
            if not isinstance(state["model"], str) or not state["model"].strip():
                add_error(
                    f"qualification.qwen.{side}.model",
                    "invalid_value",
                    "restored Qwen model identity cannot be empty",
                )
                valid = False
            return valid

        before_valid = validate_qwen_state("before", before)
        after_valid = validate_qwen_state("after", after)
        if before_valid and after_valid and before != after:
            add_error(
                "qualification.qwen.after",
                "not_restored",
                "post-benchmark Qwen state must exactly match pre-state",
            )

    def validate_artifact_identity() -> None:
        expected_valid = (
            isinstance(expected_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
        )
        actual_valid = (
            isinstance(actual_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", actual_sha256) is not None
        )
        if not expected_valid:
            add_error(
                "artifact.expected_sha256",
                "invalid_value",
                "expected SHA-256 must be 64 lowercase hex digits",
            )
        if not actual_valid:
            add_error(
                "artifact.sha256",
                "invalid_value",
                "actual SHA-256 must be 64 lowercase hex digits",
            )
        elif expected_valid and actual_sha256 != expected_sha256:
            add_error(
                "artifact.sha256",
                "sha_mismatch",
                "artifact SHA-256 does not match the expected digest",
            )
        if expected_valid:
            parsed = urlparse(artifact_url)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or expected_sha256 not in parsed.path.split("/")
            ):
                add_error(
                    "artifact.url",
                    "immutable_url",
                    "HTTPS artifact URL must contain the expected SHA as a path segment",
                )

    validate_artifact_identity()
    require_equal(
        "schema", read("schema", "schema"), "mtplx-q2-bf16-mtp-depth-matrix-v3"
    )
    require_equal("status", read("status", "status"), "passed")
    require_equal("passed", read("passed", "passed"), True)
    models = read("models", "models")
    if models is not _MISSING and (not isinstance(models, list) or len(models) != 1):
        add_error(
            "models",
            "invalid_matrix",
            "require exactly one Hy3-Q2 model result",
        )
    require_equal(
        "configuration.measurement_lane",
        read(
            "configuration.measurement_lane",
            "configuration",
            "measurement_lane",
        ),
        "issue51-qualification-resource-instrumented",
    )
    require_equal(
        "configuration.generation.mtp_cache_policy",
        read(
            "configuration.generation.mtp_cache_policy",
            "configuration",
            "generation",
            "mtp_cache_policy",
        ),
        "persistent",
    )
    require_equal(
        "qualification.profile",
        read("qualification.profile", "qualification", "profile"),
        "issue51-hy3-q2-1024x1024-k0-k7-v1",
    )
    require_equal(
        "models[0].model",
        read("models[0].model", "models", 0, "model"),
        "hy3-q2",
    )
    require_equal(
        "models[0].model_key",
        read("models[0].model_key", "models", 0, "model_key"),
        "hy3-expert-q2",
    )
    require_equal(
        "configuration.contexts",
        read("configuration.contexts", "configuration", "contexts"),
        [1024],
    )
    require_equal(
        "configuration.output_tokens",
        read("configuration.output_tokens", "configuration", "output_tokens"),
        1024,
    )
    require_equal(
        "configuration.generation.prefill_mode",
        read(
            "configuration.generation.prefill_mode",
            "configuration",
            "generation",
            "prefill_mode",
        ),
        "sustained",
    )
    require_equal(
        "configuration.candidate.verify_strategy",
        read(
            "configuration.candidate.verify_strategy",
            "configuration",
            "candidate",
            "verify_strategy",
        ),
        "capture_commit",
    )
    require_equal(
        "configuration.generation.mtp_history_policy",
        read(
            "configuration.generation.mtp_history_policy",
            "configuration",
            "generation",
            "mtp_history_policy",
        ),
        "committed",
    )
    require_equal(
        "configuration.runtime.resource_telemetry",
        read(
            "configuration.runtime.resource_telemetry",
            "configuration",
            "runtime",
            "resource_telemetry",
        ),
        True,
    )
    require_equal(
        "models[0].depths",
        read("models[0].depths", "models", 0, "depths"),
        list(range(1, 8)),
    )
    require_equal(
        "models[0].passed",
        read("models[0].passed", "models", 0, "passed"),
        True,
    )
    validate_identity_and_environment()
    validate_lock_and_qwen()
    observations = read("models[0].observations", "models", 0, "observations")
    depths: list[int] = []
    depth_fields_valid = True
    if observations is not _MISSING and not isinstance(observations, list):
        add_error(
            "models[0].observations",
            "invalid_type",
            "expected an observation list",
        )
        depth_fields_valid = False
    elif observations is not _MISSING:
        for index, row in enumerate(observations):
            path = f"models[0].observations[{index}].requested_depth"
            if not isinstance(row, dict):
                add_error(
                    f"models[0].observations[{index}]",
                    "invalid_type",
                    "expected an observation object",
                )
                depth_fields_valid = False
                continue
            depth = read(
                path,
                "models",
                0,
                "observations",
                index,
                "requested_depth",
            )
            if depth is _MISSING:
                depth_fields_valid = False
            elif isinstance(depth, bool) or not isinstance(depth, int):
                add_error(path, "invalid_type", "requested_depth must be an integer")
                depth_fields_valid = False
            else:
                depths.append(depth)
        depths.sort()
    if observations is not _MISSING and depth_fields_valid and depths != list(range(8)):
        add_error(
            "models[0].observations",
            "invalid_matrix",
            "require exactly one row for each K=0..7",
        )
    elif observations is not _MISSING and depth_fields_valid:
        for index, row in enumerate(observations):
            k = row["requested_depth"]
            expected_fields = {
                "context_tokens": 1024,
                "prompt_tokens": 1024,
                "new_prefill_tokens": 1024,
                "generated_tokens": 1024,
                "effective_depth": k,
                "finish_reason": "length",
            }
            for field, expected in expected_fields.items():
                path = f"models[0].observations[{index}].{field}"
                require_equal(
                    path,
                    read(
                        path,
                        "models",
                        0,
                        "observations",
                        index,
                        field,
                    ),
                    expected,
                )
            performance: dict[str, Any] = {}
            for field in (
                "ingestion_tok_s",
                "prompt_target_prefill_time_s",
                "decode_tok_s",
            ):
                path = f"models[0].observations[{index}].{field}"
                performance[field] = read(
                    path, "models", 0, "observations", index, field
                )
                require_positive_number(path, performance[field])
            memory_path = f"models[0].observations[{index}].peak_memory_bytes"
            memory = read(
                memory_path,
                "models",
                0,
                "observations",
                index,
                "peak_memory_bytes",
            )
            require_positive_integer(memory_path, memory)
            hash_path = (
                f"models[0].observations[{index}].ar_comparison.observed_token_sha256"
            )
            output_hash = read(
                hash_path,
                "models",
                0,
                "observations",
                index,
                "ar_comparison",
                "observed_token_sha256",
            )
            require_sha256(hash_path, output_hash)
            validate_mtp_metrics(index, k)
            validate_cache(
                index,
                ("expert_streaming_counters",),
                "expert_streaming_counters",
            )
            validate_cache(
                index,
                ("expert_streaming_counters_by_phase", "decode"),
                "expert_streaming_counters_by_phase.decode",
            )
            validate_readers(index)
            for gate in (
                "committed_history",
                "compiled_verify_evidence",
                "decode_expert_cache_metrics",
                "effective_depth_exact",
                "final_state_contract",
                "generated_count_consistent",
                "guards_disabled",
                "length_finish",
                "new_prefill_tokens_exact",
                "output_tokens_exact",
                "prompt_length_exact",
                "requested_depth_exact",
                "speculative_event_contract",
            ):
                gate_path = f"models[0].observations[{index}].gates.{gate}"
                require_equal(
                    gate_path,
                    read(
                        gate_path,
                        "models",
                        0,
                        "observations",
                        index,
                        "gates",
                        gate,
                    ),
                    True,
                )
    return {
        "schema": REPORT_SCHEMA,
        "qualified": not errors,
        "artifact": {
            "path": str(artifact_path),
            "url": artifact_url,
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
        },
        "summary": {
            "validated_depths": depths,
            "error_count": len(errors),
        },
        "errors": errors,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Validate one qualification artifact and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--artifact-url", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        artifact_bytes = args.artifact.read_bytes()
    except OSError as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "qualified": False,
            "artifact": {
                "path": str(args.artifact),
                "url": args.artifact_url,
                "sha256": None,
                "expected_sha256": args.expected_sha256,
            },
            "summary": {"validated_depths": [], "error_count": 1},
            "errors": [
                {
                    "path": "$",
                    "code": "read_error",
                    "message": str(exc),
                }
            ],
        }
    else:
        actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        try:
            document = json.loads(artifact_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            report = {
                "schema": REPORT_SCHEMA,
                "qualified": False,
                "artifact": {
                    "path": str(args.artifact),
                    "url": args.artifact_url,
                    "sha256": actual_sha256,
                    "expected_sha256": args.expected_sha256,
                },
                "summary": {"validated_depths": [], "error_count": 1},
                "errors": [
                    {
                        "path": "$",
                        "code": "invalid_json",
                        "message": str(exc),
                    }
                ],
            }
        else:
            report = validate_document(
                document,
                artifact_path=args.artifact,
                artifact_url=args.artifact_url,
                expected_sha256=args.expected_sha256,
                actual_sha256=actual_sha256,
            )
    try:
        _write_report(args.report, report)
    except OSError as exc:
        print(f"failed to write validation report: {exc}", file=sys.stderr)
        return 2
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
