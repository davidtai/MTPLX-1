from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
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
        max_tokens = int(cell["max_tokens"])
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
    verifier_bind = transformed.index("bind_pr391_float32_verifier_decision()")
    verifier_prewarm = transformed.index("_metal_choice_verifier_prewarm")
    after_load = transformed.index("after_load_memory")
    reset = transformed.index("reset_run_caches")
    install = transformed.index("PR391DirectFloat32D3Route.install")
    started = transformed.index("started = time.perf_counter()")
    generate = transformed.index("output = generate_mtpk")
    row = transformed.index("row = stats_receipt")
    close = transformed.index("_metal_choice_route.close()")
    finish = transformed.index("_metal_choice_route.finish_receipt")
    attach = transformed.index('row["metal_float32_choice_route"]')

    assert prebind < prewarm < verifier_bind < verifier_prewarm < after_load
    assert reset < install < started < generate < row < close < finish < attach
    assert "thermal_receipt = wait_for_temperature()" not in transformed
    assert '"reason": "user_requested"' in transformed
    assert "expected_seed=seed" in transformed
    assert "max_output_tokens=max_tokens" in transformed
    assert "kernel_module=_metal_choice_prebound" in transformed
    assert "verifier_kernel=_metal_choice_verifier" in transformed
    assert "verifier_prewarm=_metal_choice_verifier_prewarm" in transformed
    assert "target_sampler=sampler" in transformed
    assert 'draft_sampler=cell.get("draft_sampler", sampler)' in transformed
    assert transformed.count("PR391DirectFloat32D3Route.install") == 1
    assert "draft-choice-arm" not in transformed
    assert ROUTE_ARM == "metal-float32-test-only"


def test_direct_route_builds_one_request_tape_and_one_descriptor_array() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import (
        PR391DirectFloat32D3Route,
    )
    from scripts.pr391_metal_choice_route import PreboundMetalFloat32ChoiceKernel

    class FakeMX:
        uint32 = "uint32"
        float32 = "float32"

        def __init__(self):
            self.arrays = []

        def array(self, value, *, dtype):
            result = np.asarray(value)
            self.arrays.append((result, dtype))
            return result

    class FakeKernel:
        @staticmethod
        def build_pcg64_midpoint_descriptors(uniforms):
            return np.zeros((len(uniforms), 5), dtype=np.uint32)

    class FakeGeneration:
        def __init__(self):
            self.installed = None

        def _pr391_install_float32_d3_request_route(self, route):
            self.installed = route

        def _pr391_uninstall_float32_d3_request_route(self, route):
            assert self.installed is route
            self.installed = None

    fake_mx = FakeMX()
    prebound = PreboundMetalFloat32ChoiceKernel(
        mx=fake_mx,
            kernel_module=FakeKernel(),
            selector=object(),
            selfcheck={"schedule_id": "test"},
        source_sha256={"kernel": "1" * 64},
        _prewarm_receipt={
            "status": "passed",
            "rows": 1,
            "raw_passthrough_bit_exact": True,
            "selected_token_match": True,
            "peak_memory_bytes": 0,
            "schedule_id": "test",
        },
    )
    generation = FakeGeneration()
    target_sampler = SimpleNamespace(temperature=1.0, top_k=20, top_p=0.95)
    draft_sampler = SimpleNamespace(temperature=1.0, top_k=20, top_p=0.95)

    route = PR391DirectFloat32D3Route.install(
        generation,
        expected_seed=391,
        max_output_tokens=8,
        kernel_module=prebound,
        verifier_kernel=lambda *_args: (),
        verifier_prewarm={
            "status": "passed",
            "case_count": 6,
            "cases": [
                "reject_d0",
                "reject_d1",
                "reject_d2",
                "all_accept_bonus",
                "accepted_stop",
                "bonus_disabled",
            ],
        },
        target_sampler=target_sampler,
        draft_sampler=draft_sampler,
    )

    assert generation.installed is route
    assert route.uniform_tape.cursor == 0
    assert route.descriptor_rows.shape == (7 * 9, 5)
    assert route.uniform_rows.shape == (7 * 9,)
    assert len(fake_mx.arrays) == 4
    assert route.verifier_kernel is not None
    assert route.preserve_paged is True
    assert route.sampler is target_sampler
    assert route.draft_sampler is draft_sampler
    route.claimed = True
    route.close()
    receipt = route.finish_receipt(
        stats={
            "drafted_tokens": 6,
            "drafted_by_depth": [2, 2, 2],
            "accepted_drafts": 0,
            "verify_calls": 2,
            "correction_tokens": 2,
            "bonus_tokens": 0,
            "context_copy_rounds": 0,
            "context_copy_drafted_tokens": 0,
        },
    )
    assert receipt["schedule"]["descriptor_device_installs"] == 1
    assert generation.installed is None


def test_verifier_prewarm_proves_all_device_outcomes_against_reference() -> None:
    from mtplx.kernels.pr391_float32_verifier_decision import (
        reference_pr391_float32_verifier_decision,
    )
    from scripts.pr391_metal_choice_benchmark_launcher import (
        prewarm_float32_verifier_decision,
    )

    class FakeMX:
        uint32 = np.uint32
        int32 = np.int32
        float32 = np.float32

        @staticmethod
        def array(value, *, dtype):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            return None

    def verifier(*values):
        return reference_pr391_float32_verifier_decision(
            *values[:7],
            stop_count=int(values[7][0]),
            bonus_allowed=bool(values[8][0]),
        )

    receipt = prewarm_float32_verifier_decision(FakeMX(), verifier)

    assert receipt["status"] == "passed"
    assert receipt["case_count"] == 6
    assert receipt["cases"] == [
        "reject_d0",
        "reject_d1",
        "reject_d2",
        "all_accept_bonus",
        "accepted_stop",
        "bonus_disabled",
    ]


def test_verifier_prewarm_fails_closed_on_device_drift() -> None:
    from mtplx.kernels.pr391_float32_verifier_decision import (
        reference_pr391_float32_verifier_decision,
    )
    from scripts.pr391_metal_choice_benchmark_launcher import (
        prewarm_float32_verifier_decision,
    )

    class FakeMX:
        uint32 = np.uint32
        int32 = np.int32
        float32 = np.float32

        @staticmethod
        def array(value, *, dtype):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            return None

    def corrupt_verifier(*values):
        outputs = list(
            reference_pr391_float32_verifier_decision(
                *values[:7],
                stop_count=int(values[7][0]),
                bonus_allowed=bool(values[8][0]),
            )
        )
        outputs[5] = outputs[5] + np.uint32(1)
        return tuple(outputs)

    with pytest.raises(RuntimeError, match="parity mismatch"):
        prewarm_float32_verifier_decision(FakeMX(), corrupt_verifier)


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
            "d3_cycles": 382,
            "d3_rows": 1146,
            "context_copy_substitutions": 0,
            "context_copy_block_rounds": 0,
            "other_draft_rows": 0,
        },
        "prebound": {
            "status": "passed",
            "rows": 1,
            "raw_passthrough_bit_exact": True,
            "selected_token_match": True,
            "peak_memory_bytes": 1024,
            "schedule_id": "fixed-k20-test",
        },
        "verifier_prebound": {
            "status": "passed",
            "case_count": 6,
        },
        "stats": {"drafted_tokens": 1146},
    }
    validate_metal_choice_receipt(receipt)

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
            validate_metal_choice_receipt(changed)

    with pytest.raises(RuntimeError, match="arm"):
        validate_metal_choice_receipt(
            {**receipt, "arm": "control"}
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
            validate_metal_choice_receipt(changed)

    aggregate_mismatch = {**receipt, "stats": {"drafted_tokens": 1145}}
    with pytest.raises(RuntimeError, match="contract"):
        validate_metal_choice_receipt(aggregate_mismatch)


def test_route_receipt_uses_fixed_d3_depth_ledger_without_events() -> None:
    from scripts.pr391_metal_choice_benchmark_launcher import build_d3_route_counts

    assert build_d3_route_counts(
        {
            "drafted_tokens": 1146,
            "drafted_by_depth": [382, 382, 382],
            "context_copy_rounds": 8,
            "context_copy_drafted_tokens": 112,
        }
    ) == {
        "d3_cycles": 382,
        "d3_rows": 1146,
        "shortened_d2_cycles": 0,
        "shortened_d1_cycles": 0,
        "context_copy_rounds": 8,
        "context_copy_drafted_tokens": 112,
        "other_draft_rows": 0,
        "count_source": "construction_claim_plus_stats.drafted_by_depth",
    }

    assert build_d3_route_counts(
        {
            "drafted_tokens": 1125,
            "drafted_by_depth": [376, 375, 374],
            "context_copy_rounds": 8,
            "context_copy_drafted_tokens": 112,
        }
    ) == {
        "d3_cycles": 374,
        "d3_rows": 1122,
        "shortened_d2_cycles": 1,
        "shortened_d1_cycles": 1,
        "context_copy_rounds": 8,
        "context_copy_drafted_tokens": 112,
        "other_draft_rows": 3,
        "count_source": "construction_claim_plus_stats.drafted_by_depth",
    }


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
