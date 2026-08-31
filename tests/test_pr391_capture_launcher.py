from __future__ import annotations

from pathlib import Path

import pytest


FAKE_DRIVER = """#!/usr/bin/env python3
def main():
    runtime = object()
    cells = []
    after_load_memory = {
        "active_bytes": 1,
    }
    rows = []
    after_run_memory = {
        "active_bytes": 2,
    }
    cell_receipts = []
    return 0
"""


def test_transform_installs_after_construction_and_finalizes_after_runs() -> None:
    from scripts.pr391_capture_launcher import transform_capture_driver

    transformed = transform_capture_driver(
        FAKE_DRIVER,
        capture_path=Path("/tmp/capture.npz"),
    )

    install = transformed.index("ChoiceRowCapture.install")
    after_load = transformed.index("after_load_memory")
    after_run = transformed.index("after_run_memory")
    finalize = transformed.index("_choice_capture.finalize")
    cells = transformed.index("cell_receipts")
    assert install < after_load
    assert after_run < finalize < cells
    assert "expected_rows=3338" in transformed
    assert "'drafted_tokens': 3338" in transformed
    assert "'accepted_drafts': 1656" in transformed
    assert "'verify_calls': 1146" in transformed
    assert "'correction_tokens': 789" in transformed
    assert "'bonus_tokens': 342" in transformed
    assert "draft_hit_rate" in transformed
    assert "resolved_verify_hit_rate" in transformed
    assert "depth_hit_rate" in transformed
    assert (
        "e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc"
        in transformed
    )
    assert (
        "e50c3361a12a34d0b410819658acfc125e1559537434c57060de2cd90af94f16"
        in transformed
    )
    assert (
        "dc1816e3628d3e585ae4fe64d6745ca1c9f7ed30bde3128a624b3d5ae715e501"
        in transformed
    )
    assert "capture baseline row drifted" in transformed
    assert '"source_commit": args.expected_source' in transformed
    assert '"source_commit": source_commit' not in transformed
    assert transformed.count("ChoiceRowCapture.install") == 1
    assert transformed.count("_choice_capture.finalize") == 1


@pytest.mark.parametrize(
    "missing",
    ["    after_load_memory = {", "    cell_receipts = []"],
)
def test_transform_fails_closed_when_driver_anchor_changes(missing: str) -> None:
    from scripts.pr391_capture_launcher import DriverSourceMismatch
    from scripts.pr391_capture_launcher import transform_capture_driver

    with pytest.raises(DriverSourceMismatch, match="anchor"):
        transform_capture_driver(
            FAKE_DRIVER.replace(missing, "    changed = {"),
            capture_path=Path("/tmp/capture.npz"),
        )


def test_load_driver_requires_reviewed_sha(tmp_path: Path) -> None:
    from scripts.pr391_capture_launcher import DriverSourceMismatch
    from scripts.pr391_capture_launcher import load_reviewed_driver

    driver = tmp_path / "driver.py"
    driver.write_text(FAKE_DRIVER)
    with pytest.raises(DriverSourceMismatch, match="SHA-256"):
        load_reviewed_driver(driver)


def _canonical_driver_argv() -> list[str]:
    return [
        "--source",
        "/Users/davidtai/projects/OpenSourceWTF/.worktrees/qwen38-flash-next-210-restack",
        "--expected-source",
        "1" * 40,
        "--expected-file",
        "mtplx/generation.py=" + "2" * 64,
        "--label",
        "capture",
        "--sequence",
        "1",
        "--seed",
        "20260829",
        "--seed",
        "20260830",
        "--seed",
        "20260831",
        "--target-mode",
        "batched",
        "--require-compiled-verify",
        "--m4-stage3",
        "--qsa-fused-kv-gather",
        "--full-frspec",
        "--compiled-mtp-prepare",
        "--relaxed-draft-ties",
        "--max-tokens",
        "1024",
    ]


def test_canonical_driver_argv_is_accepted() -> None:
    from scripts.pr391_capture_launcher import validate_driver_argv

    validate_driver_argv(_canonical_driver_argv())


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("20260831", "7"), "seeds"),
        (("1024", "2048"), "max-tokens"),
        (("batched", "lazy"), "target-mode"),
    ],
)
def test_noncanonical_workload_is_rejected_before_exec(
    replacement: tuple[str, str], message: str
) -> None:
    from scripts.pr391_capture_launcher import DriverArgumentMismatch
    from scripts.pr391_capture_launcher import validate_driver_argv

    argv = _canonical_driver_argv()
    index = len(argv) - 1 - argv[::-1].index(replacement[0])
    argv[index] = replacement[1]
    with pytest.raises(DriverArgumentMismatch, match=message):
        validate_driver_argv(argv)


def test_extra_driver_mode_is_rejected_before_exec() -> None:
    from scripts.pr391_capture_launcher import DriverArgumentMismatch
    from scripts.pr391_capture_launcher import validate_driver_argv

    with pytest.raises(DriverArgumentMismatch, match="canonical"):
        validate_driver_argv([*_canonical_driver_argv(), "--warm-graph"])
