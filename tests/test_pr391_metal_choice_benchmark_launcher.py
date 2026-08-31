from __future__ import annotations

import inspect

import pytest


FAKE_DRIVER = """#!/usr/bin/env python3
def main():
    runtime = object()
    cells = []
    after_load_memory = {
        "active_bytes": 1,
    }
    def run(cell, sequence, seed):
        thermal_receipt = wait_for_temperature()
        reset_receipt = reset_run_caches(runtime, mx)
        mx.reset_peak_memory()
        sampler = cell["sampler"]
        started = time.perf_counter()
        output = generate_mtpk(
            runtime,
            seed=seed,
        )
        row = stats_receipt(
            output,
            "arm",
            sequence,
            seed,
            time.perf_counter() - started,
        )
        row["pre_run_reset"] = reset_receipt
        return row
"""


def test_transform_prewarms_before_load_receipt_and_installs_each_request() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import (
        ROUTE_ARM,
        transform_metal_choice_driver,
    )

    transformed = transform_metal_choice_driver(FAKE_DRIVER)
    prebind = transformed.index("prebind_metal_float32_choice_kernel()")
    prewarm = transformed.index("_metal_choice_prebound.prewarm_b1()")
    after_load = transformed.index("after_load_memory")
    reset = transformed.index("reset_run_caches")
    install = transformed.index("MetalFloat32ChoiceRoute.install")
    started = transformed.index("started = time.perf_counter()")
    generate = transformed.index("output = generate_mtpk")
    row = transformed.index("row = stats_receipt")
    close = transformed.index("_metal_choice_route.close()")
    finish = transformed.index("_metal_choice_route.finish_receipt")
    attach = transformed.index('row["metal_float32_choice_route"]')

    assert prebind < prewarm < after_load
    assert reset < install < started < generate < row < close < finish < attach
    assert "expected_seed=seed" in transformed
    assert "kernel_module=_metal_choice_prebound" in transformed
    assert 'sampler=cell.get("draft_sampler", sampler)' in transformed
    assert transformed.count("MetalFloat32ChoiceRoute.install") == 1
    assert "draft-choice-arm" not in transformed
    assert ROUTE_ARM == "metal-float32-test-only"


@pytest.mark.parametrize(
    "missing",
    [
        "    after_load_memory = {",
        "        started = time.perf_counter()",
        '        row["pre_run_reset"] = reset_receipt',
    ],
)
def test_transform_fails_closed_when_reviewed_anchor_changes(missing: str) -> None:
    from scripts.pr391_capture_launcher import DriverSourceMismatch
    from scripts.pr391_metal_choice_benchmark_launcher import (
        transform_metal_choice_driver,
    )

    with pytest.raises(DriverSourceMismatch, match="anchor"):
        transform_metal_choice_driver(
            FAKE_DRIVER.replace(missing, "        changed = 1")
        )


def test_route_receipt_requires_full_engagement_and_passthrough() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import (
        validate_metal_choice_receipt,
    )

    receipt = {
        "arm": "metal-float32-test-only",
        "route_counts": {
            "calls": 1146,
            "matched_rows": 1146,
            "raw_passthrough_rows": 1146,
            "pending": 0,
            "failures": 0,
        },
        "prebound": {
            "status": "passed",
            "rows": 1,
            "raw_passthrough_bit_exact": True,
            "selected_token_match": True,
            "peak_memory_bytes": 1024,
            "schedule_id": "fixed-k20-test",
        },
    }
    validate_metal_choice_receipt(receipt, drafted_tokens=1146)

    for field, value in (
        ("calls", 1145),
        ("matched_rows", 1145),
        ("raw_passthrough_rows", 1145),
        ("pending", 1),
        ("failures", 1),
    ):
        changed = {**receipt, "route_counts": dict(receipt["route_counts"])}
        changed["route_counts"][field] = value
        with pytest.raises(RuntimeError, match="contract"):
            validate_metal_choice_receipt(changed, drafted_tokens=1146)

    with pytest.raises(RuntimeError, match="arm"):
        validate_metal_choice_receipt(
            {**receipt, "arm": "control"}, drafted_tokens=1146
        )

    for field, value in (
        ("status", "failed"),
        ("rows", 2),
        ("raw_passthrough_bit_exact", False),
        ("selected_token_match", False),
        ("schedule_id", ""),
    ):
        changed = {**receipt, "prebound": dict(receipt["prebound"])}
        changed["prebound"][field] = value
        with pytest.raises(RuntimeError, match="contract"):
            validate_metal_choice_receipt(changed, drafted_tokens=1146)


def test_output_drift_receipt_quantifies_digest_counters_and_depths() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import (
        REFERENCE_ROWS,
        build_float32_output_drift_receipt,
    )

    exact = {
        "seed": 20260829,
        "response_token_sha256": (
            "e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc"
        ),
        "drafted_tokens": 1146,
        "accepted_drafts": 566,
        "verify_calls": 392,
        "correction_tokens": 269,
        "bonus_tokens": 119,
        "accepted_by_depth": [259, 187, 120],
        "drafted_by_depth": [382, 382, 382],
        "metal_float32_choice_route": {
            "start_pcg64_state_sha256": REFERENCE_ROWS[20260829][
                "start_pcg64_state_sha256"
            ],
            "final_pcg64_state_sha256": REFERENCE_ROWS[20260829][
                "final_pcg64_state_sha256"
            ],
        },
    }
    receipt = build_float32_output_drift_receipt(exact)
    assert receipt["digest_match"] is True
    assert receipt["counter_deltas"] == {
        "drafted_tokens": 0,
        "accepted_drafts": 0,
        "verify_calls": 0,
        "correction_tokens": 0,
        "bonus_tokens": 0,
    }
    assert receipt["accepted_by_depth_deltas"] == [0, 0, 0]
    assert receipt["drafted_by_depth_deltas"] == [0, 0, 0]
    assert receipt["exact_reference_match"] is True
    assert receipt["rng_state_match"] is True
    assert receipt["token_level_severity"] == {
        "status": "zero_drift_by_exact_1024_token_digest",
        "first_divergence_index": None,
        "differing_positions": 0,
        "edit_distance": 0,
    }

    drifted = dict(exact)
    drifted["response_token_sha256"] = "0" * 64
    drifted["accepted_drafts"] = 564
    drifted["accepted_by_depth"] = [258, 186, 120]
    receipt = build_float32_output_drift_receipt(drifted)
    assert receipt["digest_match"] is False
    assert receipt["counter_deltas"]["accepted_drafts"] == -2
    assert receipt["accepted_by_depth_deltas"] == [-1, -1, 0]
    assert receipt["exact_reference_match"] is False
    assert receipt["token_level_severity"]["status"] == ("reference_token_ids_required")

    rng_drifted = dict(exact)
    rng_drifted["metal_float32_choice_route"] = {
        **exact["metal_float32_choice_route"],
        "final_pcg64_state_sha256": "0" * 64,
    }
    receipt = build_float32_output_drift_receipt(rng_drifted)
    assert receipt["rng_state_match"] is False


def test_hit_miss_receipt_has_per_depth_and_verifier_rates() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import (
        build_hit_miss_receipt,
    )

    receipt = build_hit_miss_receipt(
        {
            "drafted_tokens": 10,
            "accepted_drafts": 6,
            "accepted_by_depth": [4, 2],
            "drafted_by_depth": [5, 5],
            "verify_calls": 5,
            "bonus_tokens": 2,
            "correction_tokens": 3,
        }
    )
    assert receipt["draft_hit_rate"] == pytest.approx(0.6)
    assert receipt["depths"][0]["depth_hit_rate"] == pytest.approx(0.8)
    assert receipt["depths"][1]["depth_miss_rate"] == pytest.approx(0.6)
    assert receipt["resolved_verify_hit_rate"] == pytest.approx(0.4)
    assert receipt["resolved_verify_miss_rate"] == pytest.approx(0.6)


def test_launcher_has_no_arm_switch_and_requires_candidate_source_files() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import (
        REQUIRED_EXPECTED_FILES,
        _parse_args,
        validate_candidate_file_gates,
    )

    assert "draft-choice-arm" not in inspect.getsource(_parse_args)
    specs = [f"{path}={'1' * 64}" for path in sorted(REQUIRED_EXPECTED_FILES)]
    validate_candidate_file_gates(specs)
    with pytest.raises(RuntimeError, match="expected-file"):
        validate_candidate_file_gates(specs[1:])
