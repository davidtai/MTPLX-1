"""Typed registry for the Qwen3.8 Flash-Next full-stack decode env keys.

Why this module exists
----------------------
The in-process benchmark drivers (``scripts/fable/abba_driver.py`` and the
bench-tune drivers) turn on a stack of decode switches that ``mtplx serve``
had no way to reach: they are env-gated, default-off, and only the drivers
set them. The measured consequence is that a user of ``mtplx serve
--profile turbo`` got noticeably less decode throughput than the same code
measured in-process, with the server log showing
``[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)``.

The second half of the problem is spelling. Most of these keys are read by a
bare ``os.environ.get(name, default)`` at one call site, so a misspelled key
is not an error -- it is silence, and the lane it was meant to arm stays off
while every receipt still says "ok". That is how a whole benchmark battery
ran without the compiled fixed-M4 verifier and without FR-Spec.

So this module is the ONE place that names each key of the stack, its type,
the value a reader sees when it is unset, which profile stamps it, and which
call site reads it. Two things hang off that:

* the ``turbo-full-stack`` profile's env block is generated from this
  registry (:data:`FULL_STACK_RESTACK_ENV`), so the profile and the registry
  cannot drift apart; and
* :func:`warn_unknown_family_keys` can say, at startup, that an
  ``MTPLX_QWEN4_*`` / ``MTPLX_QSA_*`` / ``MTPLX_FRSPEC_*`` key in the
  environment is read by nothing in this package -- a WARNING, never a raise,
  and never a change to any default.

Parse fidelity
--------------
These call sites do NOT agree on how to parse a boolean, and this registry
does not "fix" that: routing a read through here must be behaviour-preserving
to the byte. Each entry therefore records the parse its call site actually
performs (:class:`EnvKeySpec.parse`), and :func:`flag_enabled` reproduces it:

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
from typing import Callable, Iterable, Mapping

#: The boolean vocabulary the bare ``os.environ.get`` call sites accept.
#: Deliberately NARROWER than ``mtplx.runtime_options.ENV_TRUE_VALUES``
#: (which also takes ``enable``/``enabled``): these sites never accepted the
#: wider spelling and this registry does not widen them.
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

#: Env-key prefixes this registry is responsible for. An unregistered key
#: under one of these prefixes is what :func:`warn_unknown_family_keys`
#: reports.
FAMILY_PREFIXES = ("MTPLX_QWEN4_", "MTPLX_QSA_", "MTPLX_FRSPEC_")

#: Name of the opt-in profile that stamps this stack. Imported by
#: ``mtplx.profiles``; kept here so the registry, not the profile table, is
#: the single place the stack is described.
FULL_STACK_PROFILE_NAME = "turbo-full-stack"

PARSE_KINDS = ("lenient", "lenient_nostrip", "strict", "text")


@dataclass(frozen=True)
class EnvKeySpec:
    """One env key of the full-stack decode lane.

    ``default`` is the value the READER sees when the key is unset -- i.e.
    the literal already written at the call site, not an aspiration. Nothing
    in this module changes a default; the profile is what supplies a
    different value, and only when the operator selects it.
    """

    name: str
    kind: str  # "bool" | "str"
    parse: str  # one of PARSE_KINDS
    default: str
    profile_value: str
    set_by: tuple[str, ...]
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
        if not self.reader.strip():
            raise ValueError(f"{self.name}: reader must name the call site")
        if not self.note.strip():
            raise ValueError(f"{self.name}: note must say what the key arms")


_P = (FULL_STACK_PROFILE_NAME,)

#: The stack, in the order ``scripts/fable/server_cell_bench.py``'s
#: ``FULL_STACK_ENV`` lists it (which is the order the ABBA driver's
#: ``build_family_overrides`` builds it in). Verified against
#: ``scripts/fable/abba_driver.py:build_family_overrides`` on 2026-09-02.
FULL_STACK_KEYS: tuple[EnvKeySpec, ...] = (
    EnvKeySpec(
        name="MTPLX_QWEN4_FIXED_M4_VERIFY",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/qwen4_fixed_verify.py:qwen4_fixed_verify_enabled",
        note=(
            "Construction-bound fixed-M4 (4-row) compiled verifier. Driver "
            "flag: --require-compiled-verify."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_M4_STAGE3",
        kind="bool",
        parse="strict",
        default="0",
        profile_value="1",
        set_by=_P,
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
        profile_value="1",
        set_by=_P,
        reader="mtplx/graphbank.py:_env_enabled",
        note=(
            "One-dispatch QSA selected-K/V gather for the fixed-M4 rows. "
            "Read through graphbank's own _env_enabled helper."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_QSA_GATHER",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_qsa_gather_enabled",
        note="QSA rows-gather decode lane (self-fenced to S 2..8 at KV>=16384).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_COMPILED_GDN",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:Qwen4ExpModel.__init__",
        note="Compiled GDN decode runs (paired with MTPLX_AR_PIPELINE).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_AR_PIPELINE",
        kind="bool",
        parse="lenient",
        default="",
        profile_value="1",
        set_by=_P,
        reader="mtplx/generation.py:_env_truthy",
        note="Pipelined AR decode lane. Read through generation's _env_truthy.",
    ),
    EnvKeySpec(
        name="MTPLX_FAMILY_CAPTURE_COMMIT",
        kind="bool",
        parse="strict",
        default="0",
        profile_value="1",
        set_by=_P,
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
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_fused_hc_v3_enabled",
        note="Fused hyper-connection read v3.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_INPROJ",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_in_proj_enabled",
        note="GDN in_proj fusion (four input GEMVs to one).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GATE_UP",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_fused_gate_up_enabled",
        note="Sanitize-time MoE gate+up library merge.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_CONVNORM",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_conv_norm_enabled",
        note="Fused GDN conv+silu+l2norm between the GEMVs.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_GDN_STEP",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_fused_gdn_step_enabled",
        note="One-dispatch GDN decode step (supersedes CONVNORM at decode).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FUSED_CONVNORM_VERIFY",
        kind="bool",
        parse="lenient",
        default="0",
        profile_value="1",
        set_by=_P,
        reader="mtplx/models/qwen4_exp.py:_fused_conv_norm_rows_enabled",
        note="Verify-width conv+silu+l2norm rows kernel (S<=6).",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_COMPILED_MTP_PREPARE",
        kind="bool",
        parse="lenient",
        default="",
        profile_value="1",
        set_by=_P,
        reader="mtplx/runtime.py:load",
        note="Compiled Qwen4 MTP preparation. Driver flag: --compiled-mtp-prepare.",
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_BATCH_TARGET_ARRAYS",
        kind="bool",
        parse="lenient_nostrip",
        default="",
        profile_value="1",
        set_by=_P,
        reader="mtplx/generation.py:_batch_target_arrays_enabled",
        note=(
            "Batched target-distribution precompute. CONFLICT: turbo sets 0; "
            "the driver's default (--target-mode batched) sets 1 and wins "
            "here. Runtime-gated by MTPLX_LAZY_TARGET_DISTRIBUTIONS below, "
            "which is why the pair must move together. NOTE: this call site "
            "lowercases WITHOUT stripping, hence parse='lenient_nostrip'."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        kind="bool",
        parse="lenient",
        default="",
        profile_value="0",
        set_by=_P,
        reader="mtplx/generation.py:_lazy_target_distributions_enabled",
        note=(
            "Lazy per-row target distributions. CONFLICT: turbo sets 1; the "
            "driver sets 0 and wins here, because a 1 here makes "
            "MTPLX_BATCH_TARGET_ARRAYS runtime-dead "
            "(profiles.RUNTIME_GATED_ENV_PAIRS)."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_SKIP_VERIFY_SNAPSHOT",
        kind="bool",
        parse="strict",
        default="0",
        profile_value="0",
        set_by=_P,
        reader="mtplx/generation.py:_skip_verify_snapshot_enabled",
        note=(
            "CONFLICT: turbo sets 1; the driver's default (no "
            "--skip-verify-snapshot) sets 0 and wins here. Flash-Next "
            "rejection rollback REQUIRES the recurrent-state snapshot -- it "
            "is the pre-state the family capture-commit replays from. The "
            "server already forces 0 for mtp + qwen4_exp; the profile makes "
            "that explicit instead of implicit."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_FRSPEC_DRAFT",
        kind="bool",
        parse="lenient",
        default="",
        profile_value="1",
        set_by=_P,
        reader="mtplx/frspec_draft.py:frspec_enabled",
        note=(
            "FR-Spec row-pruned draft head. This is the switch whose absence "
            "the server log reports as '[frspec] disabled "
            "(MTPLX_FRSPEC_DRAFT=None)'."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_FRSPEC_VOCAB",
        kind="str",
        parse="text",
        default="",
        profile_value="builtin:qwen38-code-64k",
        set_by=_P,
        reader="mtplx/frspec_draft.py:_vocab_path",
        note=(
            "FR-Spec vocabulary. 'builtin:qwen38-code-64k' is the 65,536-row "
            "table the engagement marker reports as n=65536."
        ),
        routed=True,
    ),
    EnvKeySpec(
        name="MTPLX_QWEN4_RELAXED_DRAFT_TIES",
        kind="bool",
        parse="lenient",
        default="",
        profile_value="1",
        set_by=_P,
        reader="mtplx/runtime.py:load",
        note="Relaxed Qwen4 draft ties. Driver flag: --relaxed-draft-ties.",
        routed=True,
    ),
)

_BY_NAME: dict[str, EnvKeySpec] = {spec.name: spec for spec in FULL_STACK_KEYS}

if len(_BY_NAME) != len(FULL_STACK_KEYS):  # pragma: no cover - construction check
    raise RuntimeError("full-stack env registry has duplicate key names")

#: The exact env block the ``turbo-full-stack`` profile stamps, generated
#: from the registry so the two cannot drift.
FULL_STACK_RESTACK_ENV: dict[str, str] = {
    spec.name: spec.profile_value for spec in FULL_STACK_KEYS
}

#: Family-prefixed keys that ARE read somewhere in the package but are not
#: part of this stack and are not registered in ``mtplx.profiles``. Kept
#: explicit (rather than discovered at import) so the known-set is reviewable
#: and cheap; ``tests/test_full_stack_env.py`` re-derives it from the source
#: and fails if a new reader appears without landing here.
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

    return tuple(spec.name for spec in FULL_STACK_KEYS)


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


def unknown_family_keys(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Family-prefixed keys present in ``env`` that nothing in mtplx reads."""

    source = os.environ if env is None else env
    known = known_family_keys()
    return sorted(
        key
        for key in source
        if key.startswith(FAMILY_PREFIXES) and key not in known
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
            "profile_value": entry.profile_value,
            "set_by": list(entry.set_by),
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
    "FULL_STACK_PROFILE_NAME",
    "FULL_STACK_RESTACK_ENV",
    "OTHER_KNOWN_FAMILY_KEYS",
    "TRUE_TOKENS",
    "flag_enabled",
    "known_family_keys",
    "registered_names",
    "registry_rows",
    "spec",
    "text_value",
    "unknown_family_keys",
    "warn_unknown_family_keys",
]
