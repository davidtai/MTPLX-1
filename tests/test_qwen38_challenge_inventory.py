from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import qwen38_challenge_inventory as inventory_tool
from scripts.qwen38_challenge_inventory import (
    CHALLENGE_COMMIT,
    QUALIFYING_RELATIVE_PERCENT,
    load_inventory,
    validate_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "docs"
    / "perf"
    / "receipts"
    / "qwen38-challenge-port"
    / "yukon-accepted-2026-08-23.json"
)
DESIGN = ROOT / "docs" / "specs" / "2026-08-23-qwen38-challenge-port-design.md"


def test_pinned_inventory_covers_every_accepted_submission() -> None:
    inventory = load_inventory(RECEIPT, DESIGN)

    assert CHALLENGE_COMMIT == "eb5eadc7a165047d4321ce883b9ff30894d8bd19"
    assert QUALIFYING_RELATIVE_PERCENT == 0.10
    assert len(inventory.rows) == 82
    assert [row.ordinal for row in inventory.rows] == list(range(1, 83))
    assert len({row.submission_id for row in inventory.rows}) == 82
    assert all(row.score > 0 for row in inventory.rows)
    assert all(row.submission_commit for row in inventory.rows)


def test_relative_threshold_and_dispositions_are_reproducible() -> None:
    inventory = load_inventory(RECEIPT, DESIGN)
    report = validate_inventory(inventory)

    assert report.errors == ()
    assert len(report.qualifying_rows) == 54
    assert inventory.rows[0].relative_percent is None
    assert all(
        row.relative_percent > QUALIFYING_RELATIVE_PERCENT
        for row in report.qualifying_rows
    )
    assert all(row.disposition for row in inventory.rows)


def test_every_port_or_dependency_has_pr_and_source_commit() -> None:
    inventory = load_inventory(RECEIPT, DESIGN)

    selected = [
        row
        for row in inventory.rows
        if row.disposition.startswith(("PORT", "DEPENDENCY"))
    ]
    assert selected
    assert all(row.pr_number > 0 for row in selected)
    assert all(len(row.source_commit) == 40 for row in selected)


def test_source_receipt_pr_must_match_the_approved_ledger(tmp_path: Path) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload["rows"][4]["pr_number"] = 9999
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_inventory(load_inventory(altered, DESIGN))

    assert "row 5: source PR 9999 does not match approved PR 29" in report.errors


def test_regeneration_requires_explicit_checkout_and_threshold() -> None:
    with pytest.raises(
        SystemExit,
        match="--emit requires --yukon-html, --github-pulls, --challenge-repo, "
        "and --threshold",
    ):
        inventory_tool.main(
            [
                "--emit",
                "--yukon-html",
                "/does/not/matter.html",
                "--github-pulls",
                "/does/not/matter.json",
            ]
        )
