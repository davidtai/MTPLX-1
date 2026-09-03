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
    CONTROL_ARM_ENV,
    FULL_STACK_PROFILE_ENV,
    FULL_STACK_PROFILE_NAME,
    FULL_STACK_RESTACK_ENV,
    GROUP_CONTROL_ARM,
    GROUP_FABLE_DECODE,
    GROUP_FABLE_PREFILL,
    GROUP_FLAG_FILES,
    LANES,
    LANE_KEYS,
    OWNER_PROFILE,
    OWNER_SERVER_AUTO,
    OWNER_SERVER_FORCED,
    flag_enabled,
    keys_owned_by,
    known_family_keys,
    registered_names,
    group_env,
    parse_flag_file,
    resolved_stack,
    stack_summary_line,
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

#: The ABBA retained-stack CONTROL ARM, derived from the real invocation
#: rather than transcribed: abba_window.CONTROL_FLAGS parsed by the driver's
#: own parser, fed to abba_driver.build_family_overrides, plus the
#: --full-frspec block. Both modules are pure Python (no MLX), so this is a
#: two-sided check: if either the driver's defaults or this registry moves,
#: it fails.
def _control_arm_env() -> dict[str, str]:
    """Exactly what one control arm sets, computed from the driver itself."""

    import sys

    fable = str(Path(__file__).resolve().parents[1] / "scripts" / "fable")
    if fable not in sys.path:
        sys.path.insert(0, fable)
    import abba_driver
    import abba_window

    argv = [
        "--label", "control", "--sequence", "1", "--seed", "20260829",
        *abba_window.CONTROL_FLAGS,
    ]
    args = abba_driver.build_parser().parse_args(argv)
    family, _candidate = abba_driver.build_family_overrides(args)
    env = dict(family)
    # abba_driver.py:1717-1733 -- --full-frspec with no --frspec-n.
    assert args.full_frspec and args.frspec_n is None
    env["MTPLX_FRSPEC_DRAFT"] = "1"
    env["MTPLX_FRSPEC_VOCAB"] = "builtin:qwen38-code-64k"
    return env


#: The gap: CONTROL-ARM keys no server path sets. Verified against
#: mtplx/server/openai.py:_server_runtime_env_overrides on 2026-09-02. The
#: profile stamps these PLUS both retained Fable groups (see
#: test_the_profile_stamps_every_key_nothing_else_sets).
EXPECTED_GAP = {
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE": "1",
    "MTPLX_FRSPEC_DRAFT": "1",
    "MTPLX_FRSPEC_VOCAB": "builtin:qwen38-code-64k",
}

#: Armed by _server_runtime_env_overrides with
#: ``if os.environ.get(key) is None: setdefault(...)`` (or the pop-on-operator
#: form), i.e. an operator export wins. The profile must not restate these.
EXPECTED_SERVER_AUTO = {
    "MTPLX_QWEN4_FIXED_M4_VERIFY",
    "MTPLX_QWEN4_M4_STAGE3",
    "MTPLX_QSA_M4_FUSED_KV_GATHER",
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_QSA_GATHER",
    "MTPLX_COMPILED_GDN",
    "MTPLX_AR_PIPELINE",
    "MTPLX_FAMILY_CAPTURE_COMMIT",
    "MTPLX_FUSED_HC_V3",
    "MTPLX_FUSED_GDN_INPROJ",
    "MTPLX_FUSED_GATE_UP",
    "MTPLX_FUSED_GDN_CONVNORM",
    "MTPLX_FUSED_GDN_STEP",
    "MTPLX_FUSED_CONVNORM_VERIFY",
    "MTPLX_BATCH_TARGET_ARRAYS",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
    "MTPLX_NAX_VERIFY",
}

#: Assigned unconditionally by the server (beats profile AND operator).
EXPECTED_SERVER_FORCED = {"MTPLX_SKIP_VERIFY_SNAPSHOT"}

#: Turbo values the control arm disagrees with. All server-owned.
EXPECTED_CONFLICTS = {
    "MTPLX_BATCH_TARGET_ARRAYS": ("0", "1"),
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS": ("1", "0"),
    "MTPLX_SKIP_VERIFY_SNAPSHOT": ("1", "0"),
    "MTPLX_NAX_VERIFY": ("1", "0"),
}


def _server_auto_arm() -> dict[str, str]:
    """What mtplx/server/openai.py adds for a qwen4_exp fixed-M4 mtp serve.

    Modelled from _server_runtime_env_overrides: every server-owned key at
    its stack value. Applied AFTER the profile env by apply_profile_env, so
    updating the dict last is the right order.
    """

    return {
        key: FULL_STACK_RESTACK_ENV[key]
        for key in EXPECTED_SERVER_AUTO | EXPECTED_SERVER_FORCED
    }


GOLDEN_PRE_CHANGE_ENV = (
    Path(__file__).parent / "golden" / "profiles" / "pre_full_stack_profile_env.json"
)


# ---------------------------------------------------------------------------
# (1) the profile resolves to the exact env block
# ---------------------------------------------------------------------------


def test_registry_block_equals_the_real_abba_control_arm() -> None:
    """Two real sides: the driver computes one, the registry declares one.

    Not a copied literal. If abba_window.CONTROL_FLAGS changes, or a driver
    flag default moves, or the registry drifts, this fails.
    """

    assert CONTROL_ARM_ENV == _control_arm_env()
    assert len(CONTROL_ARM_ENV) == 21
    # ... and the whole measured stack is that plus the two Fable groups.
    assert len(FULL_STACK_RESTACK_ENV) == 41
    assert FULL_STACK_RESTACK_ENV == {
        **CONTROL_ARM_ENV,
        **group_env(GROUP_FABLE_DECODE),
        **group_env(GROUP_FABLE_PREFILL),
    }


def test_relaxed_draft_ties_is_not_in_the_control_arm() -> None:
    """The flag is not in CONTROL_FLAGS and nothing else implies it.

    An earlier draft of this registry shipped it. It is a CANDIDATE-arm flag:
    arming it in a profile would turn on a lane the control never measured.
    """

    import sys

    fable = str(Path(__file__).resolve().parents[1] / "scripts" / "fable")
    if fable not in sys.path:
        sys.path.insert(0, fable)
    import abba_driver
    import abba_window

    assert "--relaxed-draft-ties" not in abba_window.CONTROL_FLAGS
    argv = [
        "--label", "control", "--sequence", "1", "--seed", "1",
        *abba_window.CONTROL_FLAGS,
    ]
    args = abba_driver.build_parser().parse_args(argv)
    assert args.relaxed_draft_ties is False
    assert "MTPLX_QWEN4_RELAXED_DRAFT_TIES" not in _control_arm_env()

    # Registered (so its read is typed and its spelling known) but not scored.
    entry = full_stack_env.spec("MTPLX_QWEN4_RELAXED_DRAFT_TIES")
    assert entry is not None and entry.in_stack is False
    assert "MTPLX_QWEN4_RELAXED_DRAFT_TIES" not in FULL_STACK_RESTACK_ENV
    assert "MTPLX_QWEN4_RELAXED_DRAFT_TIES" not in FULL_STACK_PROFILE_ENV
    assert (
        "MTPLX_QWEN4_RELAXED_DRAFT_TIES"
        not in get_profile(FULL_STACK_PROFILE_NAME).env_dict()
    )


def test_profile_is_selectable_by_name_and_by_harness_alias() -> None:
    assert FULL_STACK_PROFILE_NAME in PROFILE_CHOICES
    assert FULL_STACK_PROFILE_NAME in PROFILES
    for alias in ("full-stack", "full_stack", "turbo_full_stack"):
        assert resolve_profile_name(alias) == FULL_STACK_PROFILE_NAME


def test_the_registry_records_who_sets_every_stack_key() -> None:
    # Within the control arm, the profile owns exactly the three-key gap.
    assert set(keys_owned_by(OWNER_PROFILE, group=GROUP_CONTROL_ARM)) == set(
        EXPECTED_GAP
    )
    # Both Fable groups are the profile's in full: nothing else sets any of
    # them -- not the server's auto-arm, not a model pack's contract.
    assert set(keys_owned_by(OWNER_PROFILE, group=GROUP_FABLE_DECODE)) == set(
        group_env(GROUP_FABLE_DECODE)
    )
    assert set(keys_owned_by(OWNER_PROFILE, group=GROUP_FABLE_PREFILL)) == set(
        group_env(GROUP_FABLE_PREFILL)
    )
    assert set(keys_owned_by(OWNER_SERVER_AUTO)) == EXPECTED_SERVER_AUTO
    assert set(keys_owned_by(OWNER_SERVER_FORCED)) == EXPECTED_SERVER_FORCED
    # Every key is accounted for exactly once.
    owned = (
        set(keys_owned_by(OWNER_PROFILE))
        | EXPECTED_SERVER_AUTO
        | EXPECTED_SERVER_FORCED
    )
    assert owned == set(FULL_STACK_RESTACK_ENV)
    for entry in full_stack_env.FULL_STACK_KEYS:
        assert entry.owner_site.strip(), entry.name
        assert entry.owner_predicate.strip(), entry.name


def test_the_profile_stamps_every_key_nothing_else_sets() -> None:
    """The gap plus both retained Fable groups -- 23 keys, and no others.

    Before 2026-09-03 this was three keys and the twenty Fable ones rode as
    hand-exported files, which is how the measured configuration became one
    nobody serving actually got.
    """

    expected = {
        **EXPECTED_GAP,
        **group_env(GROUP_FABLE_DECODE),
        **group_env(GROUP_FABLE_PREFILL),
    }
    assert FULL_STACK_PROFILE_ENV == expected
    assert len(FULL_STACK_PROFILE_ENV) == 23

    turbo = get_profile("turbo").env_dict()
    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()

    assert full == {**turbo, **expected}
    added = {key: value for key, value in full.items() if turbo.get(key) != value}
    assert added == expected


def test_the_profile_does_not_restate_a_server_armed_key() -> None:
    """Restating one would take the A/B switch away from the operator.

    The server arms those keys only when the environment left them unset, so
    an operator export beats it. A profile-owned key is the opposite: it
    stomps whatever is in the environment (apply_profile_env yields only on
    PROFILE_ENV_USER_OVERRIDE_KEYS). Putting a server-armed key in the
    profile would silently win those A/Bs.
    """

    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()
    turbo = get_profile("turbo").env_dict()

    for key in EXPECTED_SERVER_AUTO | EXPECTED_SERVER_FORCED:
        # Present only if turbo already had it, and then at turbo's value.
        assert full.get(key) == turbo.get(key), key


def test_an_operator_export_survives_the_profile_on_server_owned_keys() -> None:
    # The operator's A/B arm: force a server-armed lane off for one launch.
    environ = {
        "MTPLX_QWEN4_M4_STAGE3": "0",
        "MTPLX_QSA_GATHER": "0",
        "MTPLX_FUSED_GDN_STEP": "0",
    }

    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    assert environ["MTPLX_QWEN4_M4_STAGE3"] == "0"
    assert environ["MTPLX_QSA_GATHER"] == "0"
    assert environ["MTPLX_FUSED_GDN_STEP"] == "0"


def test_apply_profile_env_stamps_the_gap_on_a_clean_environ() -> None:
    environ: dict[str, str] = {}

    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    for key, value in EXPECTED_GAP.items():
        assert environ[key] == value, key
    status = profile_env_status(FULL_STACK_PROFILE_NAME, environ=environ)
    assert all(entry["ok"] for entry in status.values())


def test_frspec_is_armed_where_nothing_else_arms_it() -> None:
    # The reported symptom: `[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)`.
    assert "MTPLX_FRSPEC_DRAFT" not in get_profile("turbo").env_dict()
    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()
    assert full["MTPLX_FRSPEC_DRAFT"] == "1"
    assert full["MTPLX_FRSPEC_VOCAB"] == "builtin:qwen38-code-64k"


# ---------------------------------------------------------------------------
# (2) the turbo conflicts are SERVER-owned, and the stack check proves it
# ---------------------------------------------------------------------------


def test_every_turbo_conflict_is_server_owned_but_one_the_profile_owns() -> None:
    turbo = get_profile("turbo").env_dict()

    conflicts = {
        key: (turbo[key], value)
        for key, value in FULL_STACK_RESTACK_ENV.items()
        if key in turbo and turbo[key] != value
    }
    # The one conflict the PROFILE resolves, deliberately: turbo's 'auto'
    # prefill chunk vs the measured stack's fixed 4096. It is not a
    # server-owned key and the server does not arbitrate it, so it is
    # excluded here and asserted in
    # test_the_new_profile_touches_only_the_retained_stack.
    assert conflicts.pop("MTPLX_PREFILL_CHUNK_SIZE") == ("auto", "4096")
    # MTPLX_COMPILED_VERIFY is NOT a conflict: turbo's "1" and the control
    # arm's "on" resolve to the same graphbank mode.
    assert full_stack_env.resolved_mode(
        "MTPLX_COMPILED_VERIFY", env={"MTPLX_COMPILED_VERIFY": "1"}
    ) == full_stack_env.resolved_mode(
        "MTPLX_COMPILED_VERIFY", env={"MTPLX_COMPILED_VERIFY": "on"}
    )
    assert {k: v for k, v in conflicts.items() if k != "MTPLX_COMPILED_VERIFY"} == (
        EXPECTED_CONFLICTS
    )
    # None of them is the profile's to resolve.
    assert set(conflicts) & set(FULL_STACK_PROFILE_ENV) == set()
    for key in conflicts:
        assert full_stack_env.spec(key).owner in (
            OWNER_SERVER_AUTO,
            OWNER_SERVER_FORCED,
        )


def test_a_plain_turbo_serve_is_short_the_whole_retained_stack() -> None:
    """What `mtplx serve --profile turbo` resolves on a Flash-Next pack.

    Turbo's env plus the server's auto-arm for a qwen4_exp fixed-M4 pack in
    mtp mode: 20 of the 41 keys. Everything missing is one of the 23 the
    profile (and, since 2026-09-03, the server's retained-stack defaults)
    supply -- minus the two whose reader default already happens to be the
    stack value, which is exactly the present-vs-ok distinction
    resolved_stack draws.
    """

    environ: dict[str, str] = {}
    apply_profile_env("turbo", environ=environ)
    environ.update(_server_auto_arm())

    rows = {row["name"]: row for row in resolved_stack(environ)}
    missing = {name for name, row in rows.items() if not row["ok"]}

    assert missing <= set(FULL_STACK_PROFILE_ENV)
    satisfied_by_reader_default = set(FULL_STACK_PROFILE_ENV) - missing
    assert satisfied_by_reader_default == {
        "MTPLX_FABLE_QSA_SPARSE_DECODE_TILE",
        "MTPLX_FABLE_QSA_SPARSE_DECODE_SPLITS",
    }
    for name in satisfied_by_reader_default:
        assert rows[name]["present"] is False
    assert "20/41 driver-stack keys armed" in stack_summary_line(environ)


def test_the_profile_plus_the_server_auto_arm_completes_the_stack() -> None:
    environ: dict[str, str] = {}
    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)
    environ.update(_server_auto_arm())

    assert all(row["ok"] for row in resolved_stack(environ))
    line = stack_summary_line(environ, shape="mtp, qwen4_exp fixed-M4 pack")
    assert "41/41 driver-stack keys armed" in line
    assert "(mtp, qwen4_exp fixed-M4 pack)" in line
    assert "profile 23" in line
    assert "server_auto_arm 17" in line
    assert "server_forced 1" in line


def test_a_predicate_that_did_not_hold_reads_as_unarmed() -> None:
    # Profile selected, but the served pack is not qwen4_exp, so the server
    # armed nothing. The stack line has to say so, naming the predicate.
    environ: dict[str, str] = {}
    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    line = stack_summary_line(environ, shape="mtp, not a qwen4_exp pack")
    assert "24/41 driver-stack keys armed" in line
    assert "(mtp, not a qwen4_exp pack)" in line
    assert "_served_model_type_is_qwen4_exp(args)" in line
    assert "MTPLX_FRSPEC_DRAFT" not in line  # the profile did arm that one


def test_batched_target_arrays_is_not_runtime_dead_once_the_server_arms_it():
    # profiles.RUNTIME_GATED_ENV_PAIRS: BATCH_TARGET_ARRAYS has no effect
    # while LAZY_TARGET_DISTRIBUTIONS is truthy. Turbo ships exactly that
    # dead pair; the server's auto-arm is what undoes it for this family.
    from mtplx.profiles import RUNTIME_GATED_ENV_PAIRS, announce_runtime_gated_env

    turbo_only: dict[str, str] = {}
    apply_profile_env("turbo", environ=turbo_only)
    assert turbo_only["MTPLX_BATCH_TARGET_ARRAYS"] == "0"
    assert turbo_only["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert any(
        pair[0] == "MTPLX_BATCH_TARGET_ARRAYS" for pair in RUNTIME_GATED_ENV_PAIRS
    )

    served: dict[str, str] = {}
    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=served)
    served.update(_server_auto_arm())
    gated = announce_runtime_gated_env(served, profile_name=FULL_STACK_PROFILE_NAME)
    assert "MTPLX_BATCH_TARGET_ARRAYS" not in {entry["var"] for entry in gated}


def test_operator_env_still_beats_the_profile_on_overridable_keys() -> None:
    environ = {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"}

    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    assert environ["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"


# ---------------------------------------------------------------------------
# (3) the registry: types, defaults, parse fidelity, unknown-key warning
# ---------------------------------------------------------------------------


def test_every_registered_key_names_its_reader_owner_and_stack_value() -> None:
    for entry in full_stack_env.FULL_STACK_KEYS:
        assert ":" in entry.reader or "/" in entry.reader, entry.name
        assert entry.stack_value == FULL_STACK_RESTACK_ENV[entry.name], entry.name
        assert entry.owner in full_stack_env.OWNERS, entry.name
        assert entry.stamped_by_profile is (entry.owner == OWNER_PROFILE), entry.name


def test_registered_names_cover_the_stack_plus_the_candidate_flag() -> None:
    assert set(registered_names()) == set(FULL_STACK_RESTACK_ENV)
    assert {entry.name for entry in full_stack_env.REGISTERED_KEYS} == (
        set(FULL_STACK_RESTACK_ENV) | {"MTPLX_QWEN4_RELAXED_DRAFT_TIES"}
    )


#: The expression each rerouted call site used BEFORE it was routed,
#: transcribed from mtplx/ at 75673156 (`git show 75673156:<file>`). This is
#: the independent side of the parse check -- deriving it from the registry
#: would only prove the registry agrees with itself.
#:
#:   "or0"    (os.environ.get(K) or "0").strip().lower() in {1,true,yes,on}
#:   "get0"   os.environ.get(K, "0").strip().lower()     in {...}
#:   "get''"  os.environ.get(K, "").strip().lower()      in {...}
#:   "nostrip" os.environ.get(K, "").lower()             in {...}
ORIGINAL_READS = {
    "MTPLX_FUSED_GATE_UP": "or0",
    "MTPLX_FUSED_GDN_INPROJ": "or0",
    "MTPLX_FUSED_GDN_CONVNORM": "or0",
    "MTPLX_FUSED_GDN_STEP": "or0",
    "MTPLX_FUSED_CONVNORM_VERIFY": "or0",
    "MTPLX_QSA_GATHER": "or0",
    "MTPLX_FUSED_HC_V3": "or0",
    "MTPLX_COMPILED_GDN": "get0",
    "MTPLX_QWEN4_FIXED_M4_VERIFY": "get0",
    "MTPLX_FRSPEC_DRAFT": "get''",
    "MTPLX_QWEN4_RELAXED_DRAFT_TIES": "get''",
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE": "get''",
    "MTPLX_BATCH_TARGET_ARRAYS": "nostrip",
}

#: The reviewer's value set: unset, empty, padded, tab/newline padded, a
#: spelling only the strict parser accepts, and whitespace-only.
PARSE_VALUES = (None, "", " 1 ", "\t1\n", "enable", "  ", "0", "TRUE", "junk")

_TRUE = {"1", "true", "yes", "on"}


def _original_read(shape: str, raw: str | None) -> bool:
    """Evaluate the pre-change expression for one raw value."""

    env = {} if raw is None else {"K": raw}
    if shape == "or0":
        return (env.get("K") or "0").strip().lower() in _TRUE
    if shape == "get0":
        return env.get("K", "0").strip().lower() in _TRUE
    if shape == "get''":
        return env.get("K", "").strip().lower() in _TRUE
    if shape == "nostrip":
        return env.get("K", "").lower() in _TRUE
    raise AssertionError(f"unknown shape {shape}")


@pytest.mark.parametrize("key", sorted(ORIGINAL_READS))
@pytest.mark.parametrize("raw", PARSE_VALUES)
def test_every_rerouted_key_parses_exactly_as_its_call_site_did(
    key: str, raw: str | None
) -> None:
    env = {} if raw is None else {key: raw}
    expected = _original_read(ORIGINAL_READS[key], raw)

    assert flag_enabled(key, env=env) is expected, (key, raw)


@pytest.mark.parametrize("key", sorted(ORIGINAL_READS))
@pytest.mark.parametrize("raw", PARSE_VALUES)
def test_the_hot_path_readers_agree_with_flag_enabled(
    key: str, raw: str | None, monkeypatch
) -> None:
    """flag_reader() is what the per-forward gates call; it must not differ."""

    reader = full_stack_env.flag_reader(key)
    monkeypatch.delenv(key, raising=False)
    if raw is not None:
        monkeypatch.setenv(key, raw)

    assert reader() is _original_read(ORIGINAL_READS[key], raw), (key, raw)


def test_every_routed_key_is_covered_by_the_parse_table() -> None:
    """No rerouted read may escape the table above."""

    routed = {
        entry.name
        for entry in full_stack_env.REGISTERED_KEYS
        if entry.routed and entry.kind == "bool"
    }
    assert routed == set(ORIGINAL_READS)
    # ...plus the one routed string key, checked separately below.
    routed_text = {
        entry.name
        for entry in full_stack_env.REGISTERED_KEYS
        if entry.routed and entry.kind != "bool"
    }
    assert routed_text == {"MTPLX_FRSPEC_VOCAB"}


@pytest.mark.parametrize("raw", PARSE_VALUES)
def test_the_routed_vocab_key_matches_its_original_expression(
    raw: str | None,
) -> None:
    env = {} if raw is None else {"MTPLX_FRSPEC_VOCAB": raw}
    expected = (env.get("MTPLX_FRSPEC_VOCAB") or "").strip()

    assert text_value("MTPLX_FRSPEC_VOCAB", env=env) == expected

    with pytest.raises(TypeError):
        flag_enabled("MTPLX_FRSPEC_VOCAB", env=env)


def test_readers_still_see_a_mid_process_env_flip() -> None:
    """The A/B pattern the GDN/QSA tests use: one object, env flipped.

    flag_reader() bakes in the key and the default, never the VALUE -- an
    in-process A/B flips these between arms on an already-constructed module.
    """

    reader = full_stack_env.flag_reader("MTPLX_QSA_GATHER")
    import os as _os

    before = _os.environ.get("MTPLX_QSA_GATHER")
    try:
        _os.environ["MTPLX_QSA_GATHER"] = "0"
        assert reader() is False
        _os.environ["MTPLX_QSA_GATHER"] = "1"
        assert reader() is True
    finally:
        if before is None:
            _os.environ.pop("MTPLX_QSA_GATHER", None)
        else:
            _os.environ["MTPLX_QSA_GATHER"] = before


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


def test_no_shipped_profile_gained_a_gap_key() -> None:
    for name in PROFILE_CHOICES:
        if name == FULL_STACK_PROFILE_NAME:
            continue
        env = get_profile(name).env_dict()
        leaked = set(EXPECTED_GAP) & set(env)
        assert not leaked, f"{name} gained {sorted(leaked)}"


def test_the_new_profile_touches_only_the_retained_stack() -> None:
    """It ADDS 22 keys and REPLACES exactly one turbo value.

    MTPLX_PREFILL_CHUNK_SIZE is the single deliberate overlap: turbo ships
    'auto' (2048 either way) and the measured prefill stack runs a fixed
    4096, paired with MTPLX_QSA_PREFILL_COMPILE_ROWS. Nothing else of
    turbo's moves.
    """

    turbo = get_profile("turbo").env_dict()
    full = get_profile(FULL_STACK_PROFILE_NAME).env_dict()

    added = set(full) - set(turbo)
    assert added == set(FULL_STACK_PROFILE_ENV) - {"MTPLX_PREFILL_CHUNK_SIZE"}
    replaced = {key for key in turbo if full[key] != turbo[key]}
    assert replaced == {"MTPLX_PREFILL_CHUNK_SIZE"}
    assert turbo["MTPLX_PREFILL_CHUNK_SIZE"] == "auto"
    assert full["MTPLX_PREFILL_CHUNK_SIZE"] == "4096"
    assert full["MTPLX_QSA_PREFILL_COMPILE_ROWS"] == "4096"


def test_the_default_profile_is_still_sustained() -> None:
    from mtplx.profiles import DEFAULT_PROFILE_NAME

    assert DEFAULT_PROFILE_NAME == "sustained"
    assert resolve_profile_name(None) == "sustained"
    assert resolve_profile_name("auto") == "sustained"


# ---------------------------------------------------------------------------
# (5) the profile's own keys are operator-overridable, and the profile states
#     the launch shape it requires
# ---------------------------------------------------------------------------


def test_the_profiles_own_keys_are_operator_overridable() -> None:
    """Without this the profile has no kill switch.

    The FR-Spec installer raises rather than falling back, and until these
    keys joined PROFILE_ENV_USER_OVERRIDE_KEYS the profile stomped
    MTPLX_FRSPEC_DRAFT=0 -- so an operator who hit the raise had no way out
    but to stop using the profile. Consistent with the other 18 stack keys,
    whose server auto-arm already steps aside for an export.
    """

    from mtplx.profiles import PROFILE_ENV_USER_OVERRIDE_KEYS

    for key in FULL_STACK_PROFILE_ENV:
        assert key in PROFILE_ENV_USER_OVERRIDE_KEYS, key


@pytest.mark.parametrize("key", sorted(FULL_STACK_PROFILE_ENV))
def test_an_operator_can_override_each_profile_key(key: str) -> None:
    """Every one of the 23, not a sample: each is somebody's A/B switch."""

    off = "0" if FULL_STACK_PROFILE_ENV[key] == "1" else "operator-value"
    assert off != FULL_STACK_PROFILE_ENV[key]
    environ = {key: off}

    apply_profile_env(FULL_STACK_PROFILE_NAME, environ=environ)

    assert environ[key] == off


def test_the_profile_states_the_launch_shape_it_requires() -> None:
    """The server refuses a non-MTP / adapter launch; say so where it is read.

    The refusal itself lives in mtplx/server/openai.py
    (_assert_full_stack_profile_is_servable) and cannot be exercised here
    without importing the server, which pulls MLX in. What is checkable
    purely is that the profile documents the requirement rather than leaving
    an operator to discover it as a traceback after the weights are mapped.
    """

    caveats = " ".join(get_profile(FULL_STACK_PROFILE_NAME).caveats)

    assert "--generation-mode mtp" in caveats
    assert "--mtp-adapter" in caveats
    assert "refuses the profile at selection" in caveats


def test_the_server_refusal_exists_and_is_called_before_the_model_loads() -> None:
    """Source-level check; importing mtplx.server.openai would pull MLX in."""

    source = (
        Path(__file__).resolve().parents[1] / "mtplx" / "server" / "openai.py"
    ).read_text()

    assert "def _assert_full_stack_profile_is_servable(" in source
    # Called right after the profile is resolved, and before the load call.
    select = source.index("self.profile = get_profile(args.profile)")
    guard = source.index("_assert_full_stack_profile_is_servable(args)", select)
    load = source.index("_startup_line(f\"[5/6] Loading model weights", select)
    assert select < guard < load


def test_the_dropped_runtime_profile_alias_is_gone() -> None:
    # Speculative: no config in the wild ever carried it, unlike the
    # native_mtp_60_cold / long_response_exact_staged aliases.
    with pytest.raises(ValueError):
        resolve_profile_name("native_mtp_turbo_full_stack")
    assert resolve_profile_name("full-stack") == FULL_STACK_PROFILE_NAME
    assert resolve_profile_name("turbo_full_stack") == FULL_STACK_PROFILE_NAME


# ---------------------------------------------------------------------------
# (6) the committed flag files and the registry cannot drift apart
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _flag_file(group: str) -> dict[str, str]:
    return parse_flag_file((REPO_ROOT / GROUP_FLAG_FILES[group]).read_text())


def test_the_committed_flag_files_exist_and_parse() -> None:
    for group, relative in GROUP_FLAG_FILES.items():
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        assert _flag_file(group), relative


@pytest.mark.parametrize("group", sorted(GROUP_FLAG_FILES))
def test_each_flag_file_equals_its_registry_group(group: str) -> None:
    """The record and the source of truth, asserted equal in both directions.

    The two files are the canonical record of what the PR-391 battery
    exported by hand; mtplx/full_stack_env.py is where the same set now
    lives as code. A key added to one and not the other is the drift this
    test exists to make impossible -- and it is the drift that would put an
    unmeasured lane in a serve, or leave a measured one out.
    """

    assert _flag_file(group) == group_env(group)


def test_the_profile_env_is_exactly_the_two_files_plus_the_gap() -> None:
    union = {
        **EXPECTED_GAP,
        **_flag_file(GROUP_FABLE_DECODE),
        **_flag_file(GROUP_FABLE_PREFILL),
    }

    assert FULL_STACK_PROFILE_ENV == union
    served = get_profile(FULL_STACK_PROFILE_NAME).env_dict()
    for key, value in union.items():
        assert served[key] == value, key


def test_the_flag_files_are_shell_sourceable() -> None:
    """`set -a; . docs/perf/pr391-stack.flags` has to actually work.

    No spaces, no quotes, no shell metacharacters -- the files are documented
    as exportable by hand for a run that bypasses the server, and a value
    that needed quoting would silently export something else.
    """

    for group in GROUP_FLAG_FILES:
        for key, value in _flag_file(group).items():
            assert key == key.strip() and " " not in key
            assert value == value.strip()
            assert not (set(value) & set(" \t\"'$`;&|<>()\\")), (key, value)


def test_render_flag_file_round_trips_through_the_parser() -> None:
    from mtplx.full_stack_env import render_flag_file

    for group in GROUP_FLAG_FILES:
        rendered = render_flag_file(group, header="regenerated")
        assert parse_flag_file(rendered) == group_env(group)


# ---------------------------------------------------------------------------
# (7) every optimization has a lane name, and the lanes cover the stack
# ---------------------------------------------------------------------------


def test_every_profile_key_belongs_to_exactly_one_lane() -> None:
    covered = [key for keys in LANE_KEYS.values() for key in keys]

    assert sorted(covered) == sorted(FULL_STACK_PROFILE_ENV)
    assert len(covered) == len(set(covered))
    assert set(LANES) == set(LANE_KEYS)


def test_the_prefill_chunk_pair_shares_one_lane() -> None:
    """Half of it is the incoherent configuration the guard refuses.

    The QSA prefill graph bank captures ONE row width. Turning the chunk
    width off without the compile rows (or the other way) is exactly the
    mismatch fable_prefill_chunk.assert_prefill_chunk_coherent raises on, so
    the two cannot be separate operator switches.
    """

    assert LANE_KEYS["prefill_chunk"] == (
        "MTPLX_PREFILL_CHUNK_SIZE",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS",
    )


def test_lane_names_match_the_install_receipt_lanes_where_both_exist() -> None:
    """The name an operator types is the name the verdict line prints."""

    from mtplx.fable_install_receipts import LANE_KEYS as RECEIPT_LANE_KEYS

    shared = set(LANE_KEYS) & set(RECEIPT_LANE_KEYS)
    assert shared, "the two lane vocabularies stopped overlapping"
    for lane in sorted(shared):
        assert set(RECEIPT_LANE_KEYS[lane]) >= set(LANE_KEYS[lane]) or set(
            LANE_KEYS[lane]
        ) >= set(RECEIPT_LANE_KEYS[lane]), lane
