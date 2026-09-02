"""The opt-in ``turbo-full-stack`` profile and its typed env registry.

Pure Python: no MLX import, no model load, no GPU. Everything here is a
statement about what ``mtplx serve --profile turbo-full-stack`` resolves to
and about what the registry knows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx import full_stack_env
from mtplx.full_stack_env import (
    FULL_STACK_PROFILE_NAME,
    FULL_STACK_RESTACK_ENV,
    TRUE_TOKENS,
    flag_enabled,
    known_family_keys,
    registered_names,
    text_value,
    unknown_family_keys,
    warn_unknown_family_keys,
)
from mtplx.profiles import (
    PROFILE_CHOICES,
    PROFILES,
    apply_profile_env,
    get_profile,
    profile_env_status,
    resolve_profile_name,
)

#: The env block the in-process drivers arm, copied verbatim from
#: ``FULL_STACK_ENV`` in ``scripts/fable/server_cell_bench.py`` (branch
#: worker/w40-server-bench), which is itself what
#: ``scripts/fable/abba_driver.py:build_family_overrides`` produces from its
#: own flag defaults (--target-mode batched, no --skip-verify-snapshot).
#:
#: Duplicated on purpose. If either side moves, this test fails and someone
#: has to decide which one is right, instead of the server quietly serving a
#: different stack than the benchmark measured.
DRIVER_FULL_STACK_ENV = {
    "MTPLX_QWEN4_FIXED_M4_VERIFY": "1",
    "MTPLX_QWEN4_M4_STAGE3": "1",
    "MTPLX_QSA_M4_FUSED_KV_GATHER": "1",
    "MTPLX_QSA_GATHER": "1",
    "MTPLX_COMPILED_GDN": "1",
    "MTPLX_AR_PIPELINE": "1",
    "MTPLX_FAMILY_CAPTURE_COMMIT": "1",
    "MTPLX_FUSED_HC_V3": "1",
    "MTPLX_FUSED_GDN_INPROJ": "1",
    "MTPLX_FUSED_GATE_UP": "1",
    "MTPLX_FUSED_GDN_CONVNORM": "1",
    "MTPLX_FUSED_GDN_STEP": "1",
    "MTPLX_FUSED_CONVNORM_VERIFY": "1",
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE": "1",
    "MTPLX_QWEN4_RELAXED_DRAFT_TIES": "1",
    "MTPLX_FRSPEC_DRAFT": "1",
    "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
    # The three profile conflicts, resolved DRIVER-wins.
    "MTPLX_BATCH_TARGET_ARRAYS": "1",  # turbo sets 0
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "0",  # turbo sets 1
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "0",  # turbo sets 1
}

#: The only keys on which the driver stack and turbo disagree.
EXPECTED_CONFLICTS = {
    "MTPLX_BATCH_TARGET_ARRAYS": ("0", "1"),
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS": ("1", "0"),
    "MTPLX_SKIP_VERIFY_SNAPSHOT": ("1", "0"),
}

GOLDEN_PRE_CHANGE_ENV = (
    Path(__file__).parent / "golden" / "profiles" / "pre_full_stack_profile_env.json"
)


# ---------------------------------------------------------------------------
# (1) the profile resolves to the exact env block
# ---------------------------------------------------------------------------


def test_registry_block_matches_the_driver_stack_byte_for_byte() -> None:
    assert FULL_STACK_RESTACK_ENV == DRIVER_FULL_STACK_ENV
    assert len(FULL_STACK_RESTACK_ENV) == 20


def test_profile_is_selectable_by_name_and_by_harness_alias() -> None:
    assert FULL_STACK_PROFILE_NAME in PROFILE_CHOICES
    assert FULL_STACK_PROFILE_NAME in PROFILES
    for alias in ("full-stack", "full_stack", "turbo_full_stack"):
        assert resolve_profile_name(alias) == FULL_STACK_PROFILE_NAME


def test_profile_env_is_turbo_plus_the_driver_block() -> None:
    turbo = get_profile("turbo").env_dict()
    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()

    expected = dict(turbo)
    expected.update(DRIVER_FULL_STACK_ENV)
    assert full == expected

    # Every driver key is present at the driver's value, none dropped.
    for key, value in DRIVER_FULL_STACK_ENV.items():
        assert full[key] == value, key


def test_apply_profile_env_stamps_the_whole_block_on_a_clean_environ() -> None:
    environ: dict[str, str] = {}

    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    for key, value in DRIVER_FULL_STACK_ENV.items():
        assert environ[key] == value, key
    status = profile_env_status(FULL_STACK_PROFILE_NAME, environ=environ)
    assert all(entry["ok"] for entry in status.values())


def test_frspec_is_armed_where_turbo_leaves_it_unset() -> None:
    # The reported symptom: `[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)`.
    assert "MTPLX_FRSPEC_DRAFT" not in get_profile("turbo").env_dict()
    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()
    assert full["MTPLX_FRSPEC_DRAFT"] == "1"
    assert full["MTPLX_FRSPEC_VOCAB"] == "builtin:qwen38-code-64k"


# ---------------------------------------------------------------------------
# (2) conflicts resolve driver-wins
# ---------------------------------------------------------------------------


def test_exactly_three_keys_conflict_with_turbo() -> None:
    turbo = get_profile("turbo").env_dict()

    conflicts = {
        key: (turbo[key], value)
        for key, value in DRIVER_FULL_STACK_ENV.items()
        if key in turbo and turbo[key] != value
    }
    assert conflicts == EXPECTED_CONFLICTS


def test_conflicts_resolve_driver_wins() -> None:
    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()

    for key, (turbo_value, driver_value) in EXPECTED_CONFLICTS.items():
        assert full[key] == driver_value, key
        assert full[key] != turbo_value, key


def test_batched_target_arrays_is_not_runtime_dead_under_this_profile() -> None:
    # profiles.RUNTIME_GATED_ENV_PAIRS: BATCH_TARGET_ARRAYS has no effect
    # while LAZY_TARGET_DISTRIBUTIONS is truthy. Turbo ships exactly that
    # dead pair; the point of the driver-wins resolution is to undo it.
    from mtplx.profiles import RUNTIME_GATED_ENV_PAIRS, announce_runtime_gated_env

    environ: dict[str, str] = {}
    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    gated = announce_runtime_gated_env(environ, profile_name=FULL_STACK_PROFILE_NAME)
    dead = {entry["var"] for entry in gated}
    assert "MTPLX_BATCH_TARGET_ARRAYS" not in dead
    assert any(pair[0] == "MTPLX_BATCH_TARGET_ARRAYS" for pair in RUNTIME_GATED_ENV_PAIRS)

    turbo_environ: dict[str, str] = {}
    apply_profile_env("turbo", environ=turbo_environ)
    assert turbo_environ["MTPLX_BATCH_TARGET_ARRAYS"] == "0"
    assert turbo_environ["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"


def test_operator_env_still_beats_the_profile_on_overridable_keys() -> None:
    environ = {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"}

    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    assert environ["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"


# ---------------------------------------------------------------------------
# (3) the registry: types, defaults, parse fidelity, unknown-key warning
# ---------------------------------------------------------------------------


def test_every_registered_key_names_its_reader_and_its_profile() -> None:
    for spec in full_stack_env.FULL_STACK_KEYS:
        assert spec.set_by == (FULL_STACK_PROFILE_NAME,), spec.name
        assert ":" in spec.reader or "/" in spec.reader, spec.name
        assert spec.profile_value == DRIVER_FULL_STACK_ENV[spec.name], spec.name


def test_registered_names_are_exactly_the_block() -> None:
    assert set(registered_names()) == set(DRIVER_FULL_STACK_ENV)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("Yes", True),
        ("on", True),
        (" 1 ", True),
        ("garbage", False),
    ],
)
def test_lenient_keys_parse_exactly_as_their_call_site_did(
    raw: str | None, expected: bool
) -> None:
    # The pre-change expression at every "lenient" call site was
    #   (os.environ.get(K) or "0").strip().lower() in {"1","true","yes","on"}
    env = {} if raw is None else {"MTPLX_QSA_GATHER": raw}
    assert flag_enabled("MTPLX_QSA_GATHER", env=env) is expected
    reference = (env.get("MTPLX_QSA_GATHER") or "0").strip().lower() in TRUE_TOKENS
    assert flag_enabled("MTPLX_QSA_GATHER", env=env) is reference


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("TRUE", True),
        # The generation.py call site lowercases WITHOUT stripping. Preserved,
        # not "fixed": changing it would change behaviour for a padded value.
        (" 1 ", False),
    ],
)
def test_batch_target_arrays_keeps_its_nostrip_parse(
    raw: str | None, expected: bool
) -> None:
    env = {} if raw is None else {"MTPLX_BATCH_TARGET_ARRAYS": raw}
    assert flag_enabled("MTPLX_BATCH_TARGET_ARRAYS", env=env) is expected
    reference = env.get("MTPLX_BATCH_TARGET_ARRAYS", "").lower() in TRUE_TOKENS
    assert flag_enabled("MTPLX_BATCH_TARGET_ARRAYS", env=env) is reference


def test_text_key_reads_stripped_with_the_call_site_default() -> None:
    assert text_value("MTPLX_FRSPEC_VOCAB", env={}) == ""
    assert (
        text_value("MTPLX_FRSPEC_VOCAB", env={"MTPLX_FRSPEC_VOCAB": " a.json "})
        == "a.json"
    )
    with pytest.raises(TypeError):
        flag_enabled("MTPLX_FRSPEC_VOCAB", env={})


def test_unregistered_key_is_a_programming_error_not_a_silent_false() -> None:
    with pytest.raises(KeyError):
        flag_enabled("MTPLX_QWEN4_NOT_A_REAL_KEY", env={})


def test_unknown_family_key_warns_once_and_does_not_raise() -> None:
    lines: list[str] = []
    env = {
        "MTPLX_QWEN4_FIXED_M4_VERFIY": "1",  # transposed spelling
        "MTPLX_QSA_GATHER": "1",  # known
        "PATH": "/usr/bin",  # not ours
    }

    unknown = warn_unknown_family_keys(env, warn=lines.append)

    assert unknown == ["MTPLX_QWEN4_FIXED_M4_VERFIY"]
    assert len(lines) == 1
    assert "MTPLX_QWEN4_FIXED_M4_VERFIY" in lines[0]
    assert "WARNING" in lines[0]
    # The value is quoted back so the operator can see what they meant to set.
    assert "'1'" in lines[0]
    # And the nearest known key is offered.
    assert "MTPLX_QWEN4_FIXED_M4_VERIFY" in lines[0]


def test_known_family_keys_are_never_warned_about() -> None:
    lines: list[str] = []
    env = {key: "1" for key in known_family_keys()}

    assert warn_unknown_family_keys(env, warn=lines.append) == []
    assert lines == []


def test_unknown_detection_covers_all_three_family_prefixes() -> None:
    env = {
        "MTPLX_QWEN4_BOGUS": "1",
        "MTPLX_QSA_BOGUS": "1",
        "MTPLX_FRSPEC_BOGUS": "1",
        "MTPLX_TOTALLY_OTHER": "1",
    }

    assert unknown_family_keys(env) == [
        "MTPLX_FRSPEC_BOGUS",
        "MTPLX_QSA_BOGUS",
        "MTPLX_QWEN4_BOGUS",
    ]


def test_every_family_key_read_in_the_package_is_known_to_the_registry() -> None:
    """A new reader must land in the registry or the known list.

    Without this, adding an ``os.environ.get("MTPLX_QSA_SOMETHING")`` in the
    package would make that legitimate key start warning as unknown.
    """

    import re

    root = Path(__file__).resolve().parents[1] / "mtplx"
    pattern = re.compile(
        r'(?:os\.environ\.get|os\.environ\.pop|env_bool|_env_enabled|_env_truthy'
        r'|_env_int|_env_enabled_default_on|os\.environ\[)\(?\s*'
        r'["\'](MTPLX_(?:QWEN4|QSA|FRSPEC)_[A-Z0-9_]+)["\']'
    )
    read_keys: set[str] = set()
    for path in root.rglob("*.py"):
        read_keys.update(pattern.findall(path.read_text(errors="replace")))

    assert read_keys, "the reader scan found nothing; the pattern rotted"
    assert read_keys <= known_family_keys(), sorted(read_keys - known_family_keys())


# ---------------------------------------------------------------------------
# (4) existing profiles are untouched
# ---------------------------------------------------------------------------


def test_existing_profiles_resolve_byte_identically_to_the_pre_change_snapshot() -> None:
    """Golden snapshot taken from mtplx/profiles.py at 75673156 (the branch
    point). If an upstream change legitimately moves a shipped profile's env,
    regenerate this file; it must never move because of the full-stack work.
    """

    golden = json.loads(GOLDEN_PRE_CHANGE_ENV.read_text())

    assert set(golden) == {
        "stable",
        "performance-cold",
        "sustained",
        "turbo",
        "exact",
        "max-diagnostic",
    }
    for name, expected_env in golden.items():
        assert get_profile(name).env_dict() == expected_env, name


def test_the_new_profile_is_appended_and_displaces_nothing() -> None:
    assert PROFILE_CHOICES[:6] == (
        "stable",
        "performance-cold",
        "sustained",
        "turbo",
        "exact",
        "max-diagnostic",
    )
    assert PROFILE_CHOICES[6] == FULL_STACK_PROFILE_NAME
    assert len(PROFILE_CHOICES) == 7


def test_no_shipped_profile_gained_a_full_stack_key() -> None:
    for name in PROFILE_CHOICES:
        if name == FULL_STACK_PROFILE_NAME:
            continue
        env = get_profile(name).env_dict()
        leaked = {
            key
            for key, value in DRIVER_FULL_STACK_ENV.items()
            if env.get(key) == value and key not in EXPECTED_CONFLICTS
        }
        assert not leaked, f"{name} gained {sorted(leaked)}"


def test_the_default_profile_is_still_sustained() -> None:
    from mtplx.profiles import DEFAULT_PROFILE_NAME

    assert DEFAULT_PROFILE_NAME == "sustained"
    assert resolve_profile_name(None) == "sustained"
    assert resolve_profile_name("auto") == "sustained"
