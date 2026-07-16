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


def test_docs_index_links_required_user_journeys():
    text = _text("README.md")
    for target in (
        "getting-started.md",
        "settings.md",
        "cli.md",
        "experiments.md",
        "migration-settings.md",
        "advanced/ssd-streamed-moe.md",
    ):
        assert f"]({target})" in text


def test_streamed_moe_commands_live_in_advanced_guide_not_root_readme():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    advanced = (ROOT / "docs/advanced/ssd-streamed-moe.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_expert_manifest.py" not in root_readme
    assert "scripts/build_expert_manifest.py" in advanced
    assert "--expert-memory-limit 104GiB" in advanced


def test_root_readme_has_settings_native_normal_path():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "mtplx start",
        "mtplx settings user set runtime.profile=sustained",
        "mtplx start --set generation.temperature=0.7",
        "mtplx settings explain runtime.profile",
        "docs/settings.md",
        "docs/experiments.md",
    )
    for phrase in required:
        assert phrase in text


def test_root_readme_normal_sections_do_not_teach_legacy_runtime_flags():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normal = text.split("## Advanced and compatibility", 1)[0]
    forbidden = (
        "--profile sustained",
        "--default-temperature",
        "--default-top-p",
        "--adaptive-policy",
        "MTPLX_COMPILED_VERIFY=",
        "MTPLX_NAX_VERIFY=",
    )
    assert not [item for item in forbidden if item in normal]
