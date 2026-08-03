"""Static safety contracts for scripts that must not be run in unit tests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_is_distinct_from_live_service() -> None:
    launch = (ROOT / "launch_candidate.sh").read_text(encoding="utf-8")
    plist = (ROOT / "com.tea.deepseek-v4-0731.candidate.plist").read_text(encoding="utf-8")
    assert "com.tea.deepseek-v4-0731.candidate" in launch + plist
    assert "--port 8081" in launch
    assert "8080" not in launch
    assert "launchctl" not in launch
    assert "/usr/bin/env -i" in launch
    assert "command environment override rejected" in launch
    assert "MTPLX_DSV4_0731_TEST_FIXTURE" in launch


def test_candidate_config_pins_all_installation_identities() -> None:
    config = json.loads((ROOT / "candidate.json").read_text(encoding="utf-8"))
    assert config["candidate_port"] == 8081
    assert config["candidate_label"] == "com.tea.deepseek-v4-0731.candidate"
    assert config["encoding_source_revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    for key in ("encoding_sha256", "model_config_sha256", "model_index_sha256", "trusted_python_sha256", "worktree_base_revision"):
        expected_length = 40 if key == "worktree_base_revision" else 64
        assert len(config[key]) == expected_length


def test_command_override_is_rejected_except_for_nonstarting_fixture() -> None:
    launcher = ROOT / "launch_candidate.sh"
    fixture_env = {"PATH": os.environ["PATH"], "MTPLX_DSV4_0731_TEST_FIXTURE": "1"}
    fixture = subprocess.run(
        [str(launcher), "--print-command"], env=fixture_env, check=True, capture_output=True, text=True
    )
    assert "--port 8081" in fixture.stdout

    rejected = subprocess.run(
        [str(launcher), "--print-command"],
        env={"PATH": os.environ["PATH"], "MTPLX_DSV4_0731_EXECUTABLE": "/bin/false"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "override rejected" in rejected.stderr


def test_cutover_requires_receipts_lock_identity_and_explicit_promotion() -> None:
    source = (ROOT / "promote_cutover.py").read_text(encoding="utf-8")
    for required in (
        "LOCK_NB",
        "--promote",
        "assert_candidate_receipt",
        "assert_live_identity",
        "/v1/models",
        "finish_reason",
        "SENSITIVE_KEY",
        "finally:",
        "_bootstrap(prior_plist)",
    ):
        assert required in source


@pytest.mark.parametrize("forbidden_key, forbidden_value", [
    ("stdout", "must never be retained"),
    ("prompt", "must never be retained"),
    ("tools", []),
    ("secret", "must never be retained"),
    ("argv", ["must never be retained"]),
    ("env", {"MUST_NEVER": "be retained"}),
    ("model_path", "/Users/davidtai/models/private"),
])
def test_candidate_receipt_rejects_sensitive_capture(forbidden_key: str, forbidden_value: object) -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from promote_cutover import PromotionError, assert_candidate_receipt  # noqa: PLC0415

    receipt = {
        "candidate_preflight": {
            "ok": True,
            "label": "com.tea.deepseek-v4-0731.candidate",
            "port": 8081,
            "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "a" * 64},
        },
        "candidate_smoke": {"ok": True, "models_ok": True, "ready": True, "finish_reason": "stop"},
        forbidden_key: forbidden_value,
    }
    with pytest.raises(PromotionError, match="sensitive"):
        assert_candidate_receipt(receipt)


def test_scrubbed_passing_candidate_receipt_is_accepted() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from promote_cutover import assert_candidate_receipt  # noqa: PLC0415

    assert_candidate_receipt(
        {
            "candidate_preflight": {
                "ok": True,
                "label": "com.tea.deepseek-v4-0731.candidate",
                "port": 8081,
                "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "b" * 64},
            },
            "candidate_smoke": {"ok": True, "models_ok": True, "ready": True, "finish_reason": "stop"},
        }
    )
