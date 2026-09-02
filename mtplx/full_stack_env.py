"""Typed registry for the Qwen3.8 Flash-Next full-stack decode env keys.

Why this module exists
----------------------
The in-process benchmark drivers arm a stack of decode switches. ``mtplx
serve`` reaches most of them on its own -- ``mtplx/server/openai.py``
auto-arms them for the served pack -- but not all, and the ones it misses are
env-gated and default-off, so they simply do not happen. The visible symptom
is ``[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)`` in the server log while the
same code measures faster in-process.

Where the reference stack comes from
------------------------------------
:data:`FULL_STACK_KEYS` is the RETAINED-STACK CONTROL ARM -- the configuration
every ABBA window measures its candidate against, i.e. exactly one invocation
of ``scripts/fable/abba_driver.py`` carrying
``scripts/fable/abba_window.py:CONTROL_FLAGS``::

    python scripts/fable/abba_driver.py \
        --label <arm> --sequence <n> --seed <s> \
        --receipt-path <path> --guard-mode window \
        --target-mode batched --require-compiled-verify --m4-stage3 \
        --qsa-fused-kv-gather --full-frspec --compiled-mtp-prepare \
        --max-tokens 1024

which resolves to ``build_family_overrides(args)`` (19 keys) plus the
``--full-frspec`` block at ``abba_driver.py:1717-1733`` (2 keys) = the 21 keys
below. ``tests/test_full_stack_profile.py`` imports both modules -- they are
pure Python, no MLX -- parses that argv with the driver's own parser and
asserts equality, so the reference has two real sides rather than a copied
literal.

Two consequences of deriving it from the real control arm rather than from a
transcribed block:

* ``MTPLX_QWEN4_RELAXED_DRAFT_TIES`` is NOT in it. ``--relaxed-draft-ties`` is
  absent from ``CONTROL_FLAGS`` and ``--compiled-mtp-prepare`` does not imply
  it (``args.relaxed_draft_ties`` is False for this invocation), so it is a
  CANDIDATE-arm flag, never part of the measured control. An earlier draft of
  this registry carried it; shipping it would have armed a lane the control
  never measured.
* ``MTPLX_COMPILED_VERIFY=on`` and ``MTPLX_NAX_VERIFY=0`` ARE in it. Both are
  already supplied by the server for this family, so neither changes what the
  profile stamps -- but leaving them out understated the stack the check is
  against.

The second half of the problem is spelling. Most of these keys are read by a
bare ``os.environ.get(name, default)`` at one call site, so a misspelled key
is not an error -- it is silence, and the lane it was meant to arm stays off
while every receipt still says "ok".

So this module is the ONE place that names each key of the stack, its type,
the value a reader sees when it is unset, the call site that reads it, and --
critically -- **who sets it**. Four things hang off that:

* :data:`FULL_STACK_PROFILE_ENV`, the three keys the ``turbo-full-stack``
  profile stamps: exactly the ones nothing else sets. The profile does not
  restate a
  key the server already arms, because restating it would also STOMP an
  operator's explicit export (a profile-owned key beats the environment
  unless it is in ``PROFILE_ENV_USER_OVERRIDE_KEYS``, while the server's
  auto-arm deliberately yields to one). The profile's own three keys ARE in
  that set, so an operator export still beats them -- ``MTPLX_FRSPEC_DRAFT=0``
  is the kill switch for a lane whose installer raises rather than falling
  back;
* :data:`FULL_STACK_RESTACK_ENV`, the full 21-key control-arm block, kept as
  the reference the whole stack is checked against;
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
``mode``             a multi-valued string knob, compared on its resolved
                     mode rather than its spelling (``MTPLX_COMPILED_VERIFY``
                     reads ``1`` and ``on`` as the same mode)

``tests/test_full_stack_profile.py`` pins EVERY routed key's parse against
the expression its call site used before it was routed.
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

PARSE_KINDS = ("lenient", "lenient_nostrip", "strict", "text", "mode")

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
    #: Part of the measured control arm. A key can be registered (so its
    #: reads are typed and it is a known spelling) without being part of the
    #: stack the self-check scores -- see MTPLX_QWEN4_RELAXED_DRAFT_TIES.
    in_stack: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("bool", "str"):
            raise ValueError(f"{self.name}: kind must be 'bool' or 'str'")
        if self.parse not in PARSE_KINDS:
            raise ValueError(f"{self.name}: parse must be one of {PARSE_KINDS}")
        if self.kind == "str" and self.parse not in ("text", "mode"):
            raise ValueError(f"{self.name}: str keys must use parse='text'/'mode'")
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
REGISTERED_KEYS: tuple[EnvKeySpec, ...] = (
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
        name="MTPLX_COMPILED_VERIFY",
        kind="str",
        parse="mode",
        default="",
        stack_value="on",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_FIXED_M4,
        reader="mtplx/graphbank.py:_compiled_verify_mode",
        note=(
            "Compiled verify. The control arm sets the string 'on' and turbo "
            "sets '1'; graphbank resolves anything outside "
            "{'', 0, false, no, off, parity, parity2} to the SAME 'on' mode, "
            "so the two spellings agree and this key is compared on the "
            "resolved mode, not the literal."
        ),
    ),
    EnvKeySpec(
        name="MTPLX_NAX_VERIFY",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="0",
        owner=OWNER_SERVER_AUTO,
        owner_site=_SERVER,
        owner_predicate=_QWEN4_EXP,
        reader="mtplx/nax_verify.py:nax_verify_enabled",
        note=(
            "27B NAX verify patch. Turbo sets 1; the control arm and the "
            "server's family override both set 0 (unmeasured and mostly "
            "bypassed on this family), and the server's override is applied "
            "after the profile env, so turbo already resolves to 0 here."
        ),
    ),
    # --- registered, but NOT part of the measured control arm -------------
    EnvKeySpec(
        name="MTPLX_QWEN4_RELAXED_DRAFT_TIES",
        kind="bool",
        parse="lenient",
        default="",
        stack_value="0",
        owner=OWNER_DEFAULT,
        owner_site=_NOBODY,
        owner_predicate="never armed by default",
        reader="mtplx/runtime.py:load",
        note=(
            "Relaxed Qwen4 draft ties. Registered so its read is typed and "
            "its spelling is known, but NOT in the stack: "
            "--relaxed-draft-ties is absent from abba_window.CONTROL_FLAGS "
            "and --compiled-mtp-prepare does not imply it, so the control "
            "arm never measured it. It stays a candidate-arm flag."
        ),
        routed=True,
        in_stack=False,
    ),
)

#: The measured control arm, in registry order.
FULL_STACK_KEYS: tuple[EnvKeySpec, ...] = tuple(
    entry for entry in REGISTERED_KEYS if entry.in_stack
)

_BY_NAME: dict[str, EnvKeySpec] = {spec.name: spec for spec in REGISTERED_KEYS}

if len(_BY_NAME) != len(REGISTERED_KEYS):  # pragma: no cover - construction check
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


def flag_reader(name: str) -> Callable[[], bool]:
    """A zero-overhead per-call reader for a HOT-PATH gate.

    :func:`flag_enabled` is the right call almost everywhere, but it costs an
    extra stack frame plus a registry lookup and a parse branch per call --
    measured +53 ns (206 vs 153) against the bare expression it replaced,
    which is ~0.08% of a token across 48 layers x 4 per-forward reads. That is
    small, but it is a control-vs-candidate delta in a decode measurement, and
    a measurement tool must not move the thing it measures.

    So the per-forward gates bind this instead: the key, the default and the
    token set are baked in at import, and the closure body is the same work
    the original in-line expression did, in the same single frame.

    The env is still read on EVERY call, deliberately. These gates live on
    modules that are constructed once and then A/B'd by flipping the env
    between arms in one process (tests/test_gdn_step_fused.py and friends do
    exactly that on a shared fixture), so caching at construction -- the
    pattern Qwen4ExpTextModel.__init__ uses for MTPLX_COMPILED_GDN, which is
    read once per model -- would silently change that behaviour.
    """

    entry = _require(name)
    if entry.kind != "bool":
        raise TypeError(f"{name} is a {entry.kind} key; use text_value()")
    default = entry.default
    tokens = TRUE_TOKENS
    environ = os.environ  # the mapping itself: setenv/delenv mutate in place

    if entry.parse == "lenient":

        def _read() -> bool:
            return str(environ.get(name) or default).strip().lower() in tokens

    elif entry.parse == "lenient_nostrip":

        def _read() -> bool:
            return str(environ.get(name, default)).lower() in tokens

    else:  # "strict" -- keep the raising parse, no fast path worth the risk

        def _read() -> bool:
            return flag_enabled(name)

    _read.__name__ = f"read_{name.lower()}"
    _read.__qualname__ = _read.__name__
    _read.__doc__ = f"Read {name} exactly as {entry.reader} always did."
    return _read


def text_value(name: str, *, env: Mapping[str, str] | None = None) -> str:
    """Read a registered string key, stripped, falling back to its default."""

    entry = _require(name)
    source = os.environ if env is None else env
    return str(source.get(name) or entry.default).strip()


#: ``mtplx/graphbank.py:_compiled_verify_mode`` off-spellings. Mirrored here
#: so this module compares the resolved mode, not the literal, without
#: importing graphbank (which pulls MLX).
_MODE_OFF_TOKENS = frozenset({"", "0", "false", "no", "off"})
_MODE_NAMED = frozenset({"parity", "parity2"})


def resolved_mode(name: str, *, env: Mapping[str, str] | None = None) -> str:
    """Resolve a ``parse="mode"`` key the way its reader does."""

    entry = _require(name)
    if entry.parse != "mode":
        raise TypeError(f"{name} is not a mode key")
    source = os.environ if env is None else env
    raw = str(source.get(name) or entry.default).strip().lower()
    if raw in _MODE_OFF_TOKENS:
        return "off"
    if raw in _MODE_NAMED:
        return raw
    return "on"


def resolved_stack(env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Is the whole driver stack armed in ``env``, and who was responsible?

    One row per key: the value the stack needs, the value present, whether
    the runtime will behave as the stack requires, and who was supposed to
    supply it. Comparison is on the PARSED value, so ``"true"`` and ``"1"``
    agree and ``MTPLX_COMPILED_VERIFY`` reads ``1`` and ``on`` as one mode.

    ``present``/``supplied_by`` are reported separately from ``ok`` because a
    want-"0" key (LAZY_TARGET_DISTRIBUTIONS, NAX_VERIFY, SKIP_VERIFY_SNAPSHOT)
    also reads as satisfied when it is simply ABSENT and the reader's own
    default happens to match. That is a real distinction: nobody armed it, so
    nothing holds it there if a default or a launcher lane later moves.
    """

    source = os.environ if env is None else env
    rows: list[dict[str, Any]] = []
    for entry in FULL_STACK_KEYS:
        observed = source.get(entry.name)
        present = bool(str(observed or "").strip())
        if entry.parse == "mode":
            wanted_mode = entry.stack_value.strip().lower()
            ok = resolved_mode(entry.name, env=source) == (
                wanted_mode if wanted_mode not in _MODE_OFF_TOKENS else "off"
            )
        elif entry.kind == "bool":
            try:
                actual = flag_enabled(entry.name, env=source)
            except ValueError:  # an unparseable operator spelling
                actual = None
            ok = actual is (entry.stack_value.strip().lower() in TRUE_TOKENS)
        else:
            ok = text_value(entry.name, env=source) == entry.stack_value
        rows.append(
            {
                "name": entry.name,
                "wanted": entry.stack_value,
                "observed": observed,
                "present": present,
                "ok": bool(ok),
                "supplied_by": entry.owner if present else "reader default",
                "owner": entry.owner,
                "owner_predicate": entry.owner_predicate,
            }
        )
    return rows


def stack_summary_line(
    env: Mapping[str, str] | None = None,
    *,
    shape: str | None = None,
) -> str:
    """One line saying how much of the driver stack is armed, and by whom.

    Not a receipt for any lane -- the lanes have their own install reports
    (see mtplx/full_stack_selfcheck.py). This is the env-level answer, which
    is the one that explains a missing lane: a key the server's auto-arm
    skipped because its model predicate did not hold reads as unarmed here.
    ``shape`` is the serve shape the caller knows and this module does not
    (generation mode, model family), so a partial stack can be read against
    the predicates that produced it.
    """

    rows = resolved_stack(env)
    missing = [row for row in rows if not row["ok"]]
    by_source: dict[str, int] = {}
    for row in rows:
        if row["ok"]:
            key = row["supplied_by"]
            by_source[key] = by_source.get(key, 0) + 1
    supplied = ", ".join(f"{name} {count}" for name, count in sorted(by_source.items()))
    head = f"{len(rows) - len(missing)}/{len(rows)} driver-stack keys armed"
    if shape:
        head = f"{head} ({shape})"
    if not missing:
        return f"{head} [{supplied}]"
    detail = ", ".join(
        f"{row['name']}={row['observed']!r} want {row['wanted']!r} "
        f"[{row['owner']}: {row['owner_predicate']}]"
        for row in missing
    )
    return f"{head} [{supplied}]; NOT armed: {detail}"


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
            "in_stack": entry.in_stack,
            "note": entry.note,
        }
        for entry in REGISTERED_KEYS
    ]


__all__ = [
    "EnvKeySpec",
    "FAMILY_PREFIXES",
    "FULL_STACK_KEYS",
    "REGISTERED_KEYS",
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
    "flag_reader",
    "keys_owned_by",
    "known_family_keys",
    "registered_names",
    "registry_rows",
    "resolved_mode",
    "resolved_stack",
    "spec",
    "stack_summary_line",
    "text_value",
    "unknown_family_keys",
    "warn_unknown_family_keys",
]
