from __future__ import annotations

import pytest


FAKE_DRIVER = """#!/usr/bin/env python3
def main():
    runtime = object()
    cells = []
    after_load_memory = {"active": 1}
    def run(cell, sequence, seed):
        started = time.perf_counter()
        output = generate_mtpk(
            runtime,
            seed=seed,
        )
        if args.pipelined_mtp_hidden == "1":
            pass
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


@pytest.mark.parametrize(
    "arm",
    ["control", "reduced-exact-float64", "reduced-float32-test-only"],
)
def test_transform_installs_fixed_arm_outside_timing_and_finishes_after_run(
    arm: str,
) -> None:
    from scripts.pr391_choice_benchmark_launcher import transform_choice_driver

    transformed = transform_choice_driver(FAKE_DRIVER, arm=arm)
    install = transformed.index("NumpyChoiceRoute.install")
    started = transformed.index("started = time.perf_counter()")
    generate = transformed.index("output = generate_mtpk")
    close = transformed.index("_numpy_choice_route.close()")
    finish = transformed.index("_numpy_choice_route.finish_receipt")
    row = transformed.index("row = stats_receipt")
    attach = transformed.index('row["numpy_choice_route"]')

    assert install < started < generate < row < close < finish < attach
    assert f"arm={arm!r}" in transformed
    assert "expected_seed=seed" in transformed
    assert "draft_hit_rate" in transformed
    assert "depth_hit_rate" in transformed
    assert "resolved_verify_hit_rate" in transformed
    assert transformed.count("NumpyChoiceRoute.install") == 1


@pytest.mark.parametrize(
    "missing",
    [
        "        started = time.perf_counter()",
        '        row["pre_run_reset"] = reset_receipt',
    ],
)
def test_transform_rejects_changed_driver_anchors(missing: str) -> None:
    from scripts.pr391_choice_benchmark_launcher import DriverSourceMismatch
    from scripts.pr391_choice_benchmark_launcher import transform_choice_driver

    with pytest.raises(DriverSourceMismatch, match="anchor"):
        transform_choice_driver(
            FAKE_DRIVER.replace(missing, "        changed = 1"), arm="control"
        )


def test_transform_rejects_unknown_arm() -> None:
    from scripts.pr391_choice_benchmark_launcher import transform_choice_driver

    with pytest.raises(ValueError, match="unknown"):
        transform_choice_driver(FAKE_DRIVER, arm="bad")
