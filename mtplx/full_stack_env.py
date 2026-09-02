"""Typed registry for the Qwen3.8 Flash-Next full-stack decode env keys.

Why this module exists
----------------------
The in-process benchmark drivers (``scripts/fable/abba_driver.py`` and the
bench-tune drivers) arm a stack of decode switches. ``mtplx serve`` reaches
most of them on its own -- ``mtplx/server/openai.py`` auto-arms them for the
served pack -- but not all, and the ones it misses are env-gated and
default-off, so they simply do not happen. The visible symptom is
``[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)`` in the server log while the
same code measures faster in-process.

The second half of the problem is spelling. Most of these keys are read by a
bare ``os.environ.get(name, default)`` at one call site, so a misspelled key
is not an error -- it is silence, and the lane it was meant to arm stays off
while every receipt still says "ok".

So this module is the ONE place that names each key of the stack, its type,
the value a reader sees when it is unset, the call site that reads it, and --
critically -- **who sets it**. Four things hang off that:

* :data:`FULL_STACK_PROFILE_ENV`, the keys the ``turbo-full-stack`` profile
  stamps: exactly the ones nothing else sets. The profile does not restate a
  key the server already arms, because restating it would also STOMP an
  operator's explicit export (a profile-owned key beats the environment,
  while the server's auto-arm deliberately yields to it);
* :data:`FULL_STACK_RESTACK_ENV`, the full 20-key driver block, kept as the
  reference the whole stack is checked against;
* :func:`resolved_stack`, which answers "is the stack actually armed, and by
  whom" against a live environment; and
* :func:`warn_unknown_family_keys`, which says at startup that an
  ``MTPLX_QWEN4_*`` / ``MTPLX_QSA_*`` / ``MTPLX_FRSPEC_*`` key in the
  environment is read by nothing in this package -- a WARNING, never a raise,
  and never a change to any default.

Who sets what (verified against mtplx/server/openai.py:730-890, 2026-09-02)
--------------------------------------------------------------------------
``_server_runtime_env_overrides`` builds runtime overrides BEFORE
``apply_profile_env`` runs and they are applied AFTER the profile env, so a
server override beats a profile value, and the server's own
``if os.environ.get(key) is None`` / ``pop`` guards are what let an operator
export beat the server. Precedence, weakest to strongest:

    reader default  <  profile env  <  server auto-arm  <  operator export
                                                        <  server FORCED

Hence :data:`OWNER_SERVER_AUTO` keys must NOT appear in the profile env: the
profile would win over the operator on exactly the keys the server chose to
let the operator own.

Parse fidelity
--------------
These call sites do NOT agree on how to parse a boolean, and this registry
does not "fix" that: routing a read through here must be behaviour-preserving
to the byte. Each entry records the parse its call site actually performs
(:class:`EnvKeySpec.parse`), and :func:`flag_enabled` reproduces it:

``lenient``          ``(env.get(name) or default).strip().lower() in TRUE_TOKENS``
``lenient_nostrip``  ``env.get(name, default).lower() in TRUE_TOKENS``
``strict``           ``mtplx.runtime_options.env_bool`` (raises on an unknown
                     spelling; accepts ``enable``/``enabled``)
``text``             the raw string, read with :func:`text_value`

``tests/test_full_stack_env.py`` pins each routed key's parse against the
expression the call site used before it was routed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

#: The boolean vocabulary the bare ``os.environ.get`` call sites accept.
#: Deliberately NARROWER than ``mtplx.runtime_options.ENV_TRUE_VALUES``
#: (which also takes ``enable``/``enabled``): these sites never accepted the
#: wider spelling and this registry does not widen them.
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

#: Env-key prefixes this registry is responsible for. An unregistered key
#: under one of these prefixes is what :func:`warn_unknown_family_keys`
#: reports.
FAMILY_PREFIXES = ("MTPLX_QWEN4_", "MTPLX_QSA_", "MTPLX_FRSPEC_")

#: Name of the opt-in profile that closes the gap. Imported by
#: ``mtplx.profiles``; kept here so the registry, not the profile table, is
#: the single place the stack is described.
FULL_STACK_PROFILE_NAME = "turbo-full-stack"

PARSE_KINDS = ("lenient", "lenient_nostrip", "strict", "text")

#: Nothing sets it. Serving gets the reader's own unset default.
OWNER_DEFAULT = "default"
#: mtplx/server/openai.py:_server_runtime_env_overrides arms it for the
#: served pack, but only when the operator has not exported it. Yields to an
#: operator export -- which is why the profile must not restate these.
OWNER_SERVER_AUTO = "server_auto_arm"
#: Same function, but assigned unconditionally: it beats an operator export
#: too, because the value is a correctness requirement rather than a knob.
OWNER_SERVER_FORCED = "server_forced"
#: The ``turbo-full-stack`` profile stamps it, because nothing else does.
OWNER_PROFILE = "profile"

OWNERS = (OWNER_DEFAULT, OWNER_SERVER_AUTO, OWNER_SERVER_FORCED, OWNER_PROFILE)


@dataclass(frozen=True)
class EnvKeySpec:
    """One env key of the full-stack decode lane.

    ``default`` is the value the READER sees when the key is unset -- i.e.
    the literal already written at the call site, not an aspiration.
    ``stack_value`` is the value the in-process drivers set, i.e. what a
    fully armed stack must resolve to, whoever supplies it. Nothing in this
    module changes a default.
    """

    name: str
    kind: str  # "bool" | "str"
    parse: str  # one of PARSE_KINDS
    default: str
    stack_value: str
    owner: str
    owner_site: str
    owner_predicate: str
    reader: str
    note: str
    routed: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("bool", "str"):
            raise ValueError(f"{self.name}: kind must be 'bool' or 'str'")
        if self.parse not in PARSE_KINDS:
            raise ValueError(f"{self.name}: parse must be one of {PARSE_KINDS}")
        if self.kind == "str" and self.parse != "text":
            raise ValueError(f"{self.name}: str keys must use parse='text'")
        if self.owner not in OWNERS:
            raise ValueError(f"{self.name}: owner must be one of {OWNERS}")
        if not self.reader.strip():
            raise ValueError(f"{self.name}: reader must name the call site")
        if not self.owner_site.strip():
            raise ValueError(f"{self.name}: owner_site must name who sets it")
        if not self.note.strip():
            raise ValueError(f"{self.name}: note must say what the key arms")

    @property
    def stamped_by_profile(self) -> bool:
        return self.owner == OWNER_PROFILE


_SERVER = "mtplx/server/openai.py:_server_runtime_env_overrides"
_QWEN4_EXP = "_served_model_type_is_qwen4_exp(args)"
_FIXED_M4 = "_served_model_is_qwen4_fixed_m4(args)"
_MTP_QWEN4 = 'generation_mode == "mtp" and _served_model_type_is_qwen4_exp(args)'
_PROFILE_SITE = "mtplx/profiles.py:TURBO_FULL_STACK_PROFILE"
_NOBODY = "nobody"

#: The stack, in the order ``scripts/fable/server_cell_bench.py``'s
#: ``FULL_STACK_ENV`` lists it (which is the order the ABBA driver's
#: ``build_family_overrides`` builds it in). Ownership verified against
#: mtplx/server/openai.py:730-890 on 2026-09-02.
FULL_STACK_KEYS: tuple[EnvKeySpec, ...] = (
    EnvKeySpec(
        name="MTPLX_QWEN4_FIXED_M4_VERIFY",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/qwen4_fixed_verify.py:qwen4_fixed_verify_enabled",
        note=(
            "Construction-bound fixed-M4 (4-row) compiled verifier. The "
            "server arms it for a fixed-M4 pack; an operator export makes the "
            "server drop its override entirely (pop, not setdefault)."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_M4_STAGE3",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/qwen4_m4_stage3.py:qwen4_m4_stage3_flags",
        note=(
            "Stage-3 M4 MoE combine tail. Already goes through "
            "runtime_options.env_bool, so it is registered but not rerouted."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_QSA_M4_FUSED_KV_GATHER",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/graphbank.py:_env_enabled",
        note="One-dispatch QSA selected-K/V gather for the fixed-M4 rows.",
    ),
    EnvKeySpec(
        name="MTPLX_QSA_GATHER",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_qsa_gather_enabled",
        note="QSA rows-gather decode lane (self-fenced to S 2..8 at KV>=16384).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_COMPILED_GDN",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:Qwen4ExpTextModel.__init__",
        note="Compiled GDN decode runs (paired with MTPLX_AR_PIPELINE).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_AR_PIPELINE",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_env_truthy",
        note="Pipelined AR decode lane. Read through generation's _env_truthy.",
    ),
    EnvKeySpec(
        name="MTPLX_FAMILY_CAPTURE_COMMIT",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_family_capture_commit_enabled",
        note=(
            "Layer-owned capture-commit (repair-free speculative rollback). "
            "Already reads through runtime_options.env_bool."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_HC_V3",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_hc_v3_enabled",
        note="Fused hyper-connection read v3.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_INPROJ",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_in_proj_enabled",
        note="GDN in_proj fusion (four input GEMVs to one).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GATE_UP",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gate_up_enabled",
        note="Sanitize-time MoE gate+up library merge.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_CONVNORM",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_conv_norm_enabled",
        note="Fused GDN conv+silu+l2norm between the GEMVs.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_STEP",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_step_enabled",
        note="One-dispatch GDN decode step (supersedes CONVNORM at decode).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_CONVNORM_VERIFY",
        kind="bool",
        parse="lenient",
        default="0",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/models/qwen4_exp.py:_fused_conv_norm_rows_enabled",
        note="Verify-width conv+silu+l2norm rows kernel (S<=6).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_COMPILED_MTP_PREPARE",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime.py:load",
        note=(
            "Compiled Qwen4 MTP preparation (driver flag: "
            "--compiled-mtp-prepare). GAP: no server path sets it."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_BATCH_TARGET_ARRAYS",
        kind="bool",
        parse="lenient_nostrip",
        default="",
        stack_value="1",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_batch_target_arrays_enabled",
        note=(
            "Batched target-distribution precompute. Turbo sets 0 and the "
            "server's auto-arm already overrides it to 1 for this family "
            "(the override is applied AFTER the profile env), so the "
            "turbo-vs-driver conflict is resolved by the server, not here. "
            "NOTE: this call site lowercases WITHOUT stripping, hence "
            "parse='lenient_nostrip'."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="0",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/generation.py:_lazy_target_distributions_enabled",
        note=(
            "Lazy per-row target distributions. Turbo sets 1, which would "
            "make MTPLX_BATCH_TARGET_ARRAYS runtime-dead "
            "(profiles.RUNTIME_GATED_ENV_PAIRS); the server's auto-arm "
            "already overrides it to 0 for this family."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_SKIP_VERIFY_SNAPSHOT",
        kind="bool",
        parse="strict",
        default="0",
        stack_value="0",
        owner=OWNER_SERVER_FORCED,
        owner_site=_SERVER,
        owner_predicate=_MTP_QWEN4,
        reader="mtplx/generation.py:_skip_verify_snapshot",
        note=(
            "Turbo sets 1; the server assigns 0 UNCONDITIONALLY for mtp on "
            "this family (plain assignment, not setdefault), so it beats both "
            "the profile and an operator export. Flash-Next rejection "
            "rollback replays from the recurrent-state snapshot -- this is a "
            "correctness requirement, not a speed knob."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FRSPEC_DRAFT",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/frspec_draft.py:frspec_enabled",
        note=(
            "FR-Spec row-pruned draft head. GAP: no server path sets it, and "
            "it is not in profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS either, so "
            "a model pack's runtime contract cannot request it. This is the "
            "switch whose absence the log reports as '[frspec] disabled "
            "(MTPLX_FRSPEC_DRAFT=None)'."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FRSPEC_VOCAB",
        kind="str",
        parse="text",
        default="",
        stack_value="builtin:qwen38-code-64k",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/frspec_draft.py:_vocab_path",
        note=(
            "FR-Spec vocabulary; 'builtin:qwen38-code-64k' is the 65,536-row "
            "table the engagement marker reports as n=65536. GAP: same as "
            "MTPLX_FRSPEC_DRAFT."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_RELAXED_DRAFT_TIES",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="1",
        owner=OWNER_PROFILE,
        owner_site=_PROFILE_SITE,
        owner_predicate="profile selected",
        reader="mtplx/runtime.py:load",
        note=(
            "Relaxed Qwen4 draft ties (driver flag: --relaxed-draft-ties). "
            "GAP: no server path sets it."
        ),
        routed=True,
    ),
)

_BY_NAME: dict[str, EnvKeySpec] = {spec.name: spec for spec in FULL_STACK_KEYS}

if len(_BY_NAME) != len(FULL_STACK_KEYS):  # pragma: no cover - construction check
    raise RuntimeError("full-stack env registry has duplicate key names")

#: The complete driver stack: what a fully armed serve must resolve to,
#: whoever supplies each value. This is the reference :func:`resolved_stack`
#: checks a live environment against; it is NOT what the profile stamps.
FULL_STACK_RESTACK_ENV: dict[str, str] = {
    spec.name: spec.stack_value for spec in FULL_STACK_KEYS
}

#: What the ``turbo-full-stack`` profile actually stamps: exactly the keys
#: nothing else sets. Restating a server-armed key here would also stomp an
#: operator's explicit export, because a profile-owned key beats the
#: environment while the server's auto-arm deliberately yields to it.
FULL_STACK_PROFILE_ENV: dict[str, str] = {
    spec.name: spec.stack_value for spec in FULL_STACK_KEYS if spec.stamped_by_profile
}

#: Family-prefixed keys that ARE read somewhere in the package but are not
#: part of this stack and are not registered in ``mtplx.profiles``. Kept
#: explicit (rather than discovered at import) so the known-set is reviewable
#: and cheap; ``tests/test_full_stack_profile.py`` re-derives it from the
#: source and fails if a new reader appears without landing here.
OTHER_KNOWN_FAMILY_KEYS: tuple[str, ...] = (
    "MTPLX_FRSPEC_LEGACY",
    "MTPLX_FRSPEC_N",
    "MTPLX_QSA_GATHER_DECODE",
    "MTPLX_QSA_SCORE_TILE_ROWS",
)


def spec(name: str) -> EnvKeySpec | None:
    """The registry entry for ``name``, or ``None`` when unregistered."""

    return _BY_NAME.get(name)


def registered_names() -> tuple[str, ...]:
    """Every key of the full-stack decode lane, in registry order."""

    return tuple(entry.name for entry in FULL_STACK_KEYS)


def keys_owned_by(owner: str) -> tuple[str, ...]:
    """Every key a given setter is responsible for."""

    if owner not in OWNERS:
        raise ValueError(f"unknown owner {owner!r}; expected one of {OWNERS}")
    return tuple(entry.name for entry in FULL_STACK_KEYS if entry.owner == owner)


def _require(name: str) -> EnvKeySpec:
    found = _BY_NAME.get(name)
    if found is None:
        raise KeyError(
            f"{name} is not in the full-stack env registry; add an EnvKeySpec "
            "for it rather than reading os.environ directly"
        )
    return found


def flag_enabled(name: str, *, env: Mapping[str, str] | None = None) -> bool:
    """Read a registered boolean key exactly the way its call site does.

    Reproduces the call site's own parse (see the module docstring): this is
    a routing helper, not a normalization pass. ``strict`` keys defer to
    ``mtplx.runtime_options.env_bool``, imported lazily so this module keeps
    no import-time coupling (``runtime_options`` parses flags at import and
    can raise on a bad spelling).
    """

    entry = _require(name)
    if entry.kind != "bool":
        raise TypeError(f"{name} is a {entry.kind} key; use text_value()")
    source = os.environ if env is None else env
    if entry.parse == "strict":
        from .runtime_options import env_bool

        return env_bool(
            name,
            default=entry.default.strip().lower() in TRUE_TOKENS,
            env=source,
        )
    if entry.parse == "lenient_nostrip":
        return str(source.get(name, entry.default)).lower() in TRUE_TOKENS
    return str(source.get(name) or entry.default).strip().lower() in TRUE_TOKENS


def text_value(name: str, *, env: Mapping[str, str] | None = None) -> str:
    """Read a registered string key, stripped, falling back to its default."""

    entry = _require(name)
    source = os.environ if env is None else env
    return str(source.get(name) or entry.default).strip()


def resolved_stack(env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Is the whole driver stack armed in ``env``, and who was responsible?

    One row per key: the value the stack needs, the value present, whether
    they agree, and the owner that was supposed to supply it. Comparison is
    on the PARSED value for booleans, so ``"true"`` and ``"1"`` agree.
    """

    source = os.environ if env is None else env
    rows: list[dict[str, Any]] = []
    for entry in FULL_STACK_KEYS:
        observed = source.get(entry.name)
        if entry.kind == "bool":
            try:
                actual = flag_enabled(entry.name, env=source)
            except ValueError:  # an unparseable operator spelling
                actual = None
            wanted = entry.stack_value.strip().lower() in TRUE_TOKENS
            ok = actual is wanted
        else:
            ok = text_value(entry.name, env=source) == entry.stack_value
        rows.append(
            {
                "name": entry.name,
                "wanted": entry.stack_value,
                "observed": observed,
                "ok": bool(ok),
                "owner": entry.owner,
                "owner_predicate": entry.owner_predicate,
            }
        )
    return rows


def stack_summary_line(env: Mapping[str, str] | None = None) -> str:
    """One line saying how much of the driver stack is armed, and by whom.

    Not a receipt for any lane -- the lanes have their own install reports
    (see mtplx/full_stack_selfcheck.py). This is the env-level answer, which
    is the one that explains a missing lane: a key the server's auto-arm
    skipped because its model predicate did not hold reads as unarmed here.
    """

    rows = resolved_stack(env)
    missing = [row for row in rows if not row["ok"]]
    by_owner: dict[str, int] = {}
    for row in rows:
        if row["ok"]:
            by_owner[row["owner"]] = by_owner.get(row["owner"], 0) + 1
    supplied = ", ".join(f"{owner} {count}" for owner, count in sorted(by_owner.items()))
    head = f"{len(rows) - len(missing)}/{len(rows)} driver-stack keys armed"
    if not missing:
        return f"{head} ({supplied})"
    detail = ", ".join(
        f"{row['name']}={row['observed']!r} want {row['wanted']!r} "
        f"[{row['owner']}: {row['owner_predicate']}]"
        for row in missing
    )
    return f"{head} ({supplied}); NOT armed: {detail}"


def known_family_keys() -> frozenset[str]:
    """Every family-prefixed key this package knows how to read or stamp.

    The union of the registry, the keys ``mtplx.profiles`` validates
    (``MODEL_RUNTIME_ENV_OVERRIDE_KEYS`` and
    ``PROFILE_ENV_USER_OVERRIDE_KEYS``), and
    :data:`OTHER_KNOWN_FAMILY_KEYS`. Imported lazily: ``mtplx.profiles``
    imports this module for the profile env block.
    """

    from .profiles import (
        MODEL_RUNTIME_ENV_OVERRIDE_KEYS,
        PROFILE_ENV_USER_OVERRIDE_KEYS,
    )

    known = set(_BY_NAME)
    known.update(OTHER_KNOWN_FAMILY_KEYS)
    for key in (*MODEL_RUNTIME_ENV_OVERRIDE_KEYS, *PROFILE_ENV_USER_OVERRIDE_KEYS):
        if key.startswith(FAMILY_PREFIXES):
            known.add(key)
    return frozenset(known)


def unknown_family_keys(env: Mapping[str, str] | None = None) -> list[str]:
    """Family-prefixed keys present in ``env`` that nothing in mtplx reads."""

    source = os.environ if env is None else env
    known = known_family_keys()
    return sorted(
        key for key in source if key.startswith(FAMILY_PREFIXES) and key not in known
    )


def _nearest_known(name: str, candidates: Iterable[str]) -> str | None:
    """Cheap 'did you mean' by longest shared prefix under the same family."""

    best: str | None = None
    best_score = 0
    for candidate in candidates:
        score = 0
        for left, right in zip(name, candidate):
            if left != right:
                break
            score += 1
        if score > best_score and score > len("MTPLX_QWEN4_"):
            best, best_score = candidate, score
    return best


def warn_unknown_family_keys(
    env: Mapping[str, str] | None = None,
    *,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """Log one WARNING per unreadable family key. Never raises, never mutates.

    Returns the unknown keys so a health surface can carry them. A key here
    is not an error -- it is a key that will silently do nothing, which is
    the failure this registry exists to make visible.
    """

    unknown = unknown_family_keys(env)
    if not unknown:
        return unknown
    source = os.environ if env is None else env
    emit = warn if warn is not None else _default_warn
    known = known_family_keys()
    for key in unknown:
        suggestion = _nearest_known(key, known)
        hint = f"; did you mean {suggestion}?" if suggestion else ""
        try:
            emit(
                f"[mtplx] WARNING: {key}={source.get(key)!r} is set but no "
                f"reader in mtplx consults it -- it will silently do "
                f"nothing{hint}"
            )
        except Exception:  # pragma: no cover - a warning must not break boot
            pass
    return unknown


def _default_warn(line: str) -> None:  # pragma: no cover - trivial sink
    import logging

    logging.getLogger("mtplx.full_stack_env").warning("%s", line)
    print(line, flush=True)


def registry_rows() -> list[dict[str, object]]:
    """The registry as plain data, for ``/health`` and docs generation."""

    return [
        {
            "name": entry.name,
            "kind": entry.kind,
            "parse": entry.parse,
            "default": entry.default,
            "stack_value": entry.stack_value,
            "owner": entry.owner,
            "owner_site": entry.owner_site,
            "owner_predicate": entry.owner_predicate,
            "reader": entry.reader,
            "routed": entry.routed,
            "note": entry.note,
        }
        for entry in FULL_STACK_KEYS
    ]


__all__ = [
    "EnvKeySpec",
    "FAMILY_PREFIXES",
    "FULL_STACK_KEYS",
    "FULL_STACK_PROFILE_ENV",
    "FULL_STACK_PROFILE_NAME",
    "FULL_STACK_RESTACK_ENV",
    "OTHER_KNOWN_FAMILY_KEYS",
    "OWNERS",
    "OWNER_DEFAULT",
    "OWNER_PROFILE",
    "OWNER_SERVER_AUTO",
    "OWNER_SERVER_FORCED",
    "TRUE_TOKENS",
    "flag_enabled",
    "keys_owned_by",
    "known_family_keys",
    "registered_names",
    "registry_rows",
    "resolved_stack",
    "spec",
    "stack_summary_line",
    "text_value",
    "unknown_family_keys",
    "warn_unknown_family_keys",
]
