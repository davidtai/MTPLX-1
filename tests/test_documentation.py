from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def test_settings_guide_covers_all_scopes_and_precedence():
    text = _text("settings.md")
    for phrase in (
        "settings user set",
        "settings live set",
        "--set",
        "--settings",
        "Precedence",
        "settings explain",
        "API key file",
    ):
        assert phrase in text


def test_cli_guide_explains_settings_versus_operands():
    text = _text("cli.md")
    assert "Settings versus command inputs" in text
    assert "mtplx help advanced" in text
    assert "mtplx lab list" in text


def test_experiment_guide_covers_lifecycle_and_provenance():
    text = _text("experiments.md")
    for phrase in (
        "active",
        "retained",
        "rejected",
        "superseded",
        "expired",
        "SHA-256",
    ):
        assert phrase in text


def test_migration_guide_maps_every_compatibility_alias():
    text = _text("migration-settings.md")
    from mtplx.settings.builtins import BUILTIN_SETTINGS

    for spec in BUILTIN_SETTINGS:
        for alias in spec.aliases:
            if alias.source in {"cli", "env"}:
                assert alias.name in text
