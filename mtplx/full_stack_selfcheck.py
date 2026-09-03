"""Startup self-check for the ``turbo-full-stack`` profile's engagement.

Arming an env flag is not the same as engaging a lane: every key in
:mod:`mtplx.full_stack_env` is read by exactly one call site, most of them
with a bare ``os.environ.get`` and a default, so a key that never reaches
the reader -- misspelled, stomped, or refused because the model is not
qwen4_exp -- leaves the lane off while every other receipt still says ok.

Arming an env flag is also not the same as an operator having chosen it: the
26 retained keys are armed by default now, so the receipts have to say who
won, and ``mtplx.full_stack_env.value_source`` is what lets them.

The benchmark harness's ``--require-full-stack`` answers the same
question by grepping a finished log for three proofs: an FR-Spec install
report at ``n=65536``, an M4 route reported as installed, and an all-ok
warmup ladder. This module answers them in-process, from the install reports
the runtime already publishes, and prints one line per marker.

Where each marker's evidence comes from -- these are the EXISTING receipts,
not new ones:

``frspec_installed``   ``mtplx/draft_lm_head.py`` prints ``[frspec] install
                       report: {...}`` to stderr; the same dict is the
                       ``frspec`` section of the draft-head report the server
                       keeps on ``state.draft_lm_head``.
``qwen4_fixed_m4_verify``
                       ``mtplx/runtime.py`` prints ``[qwen4-fixed-M4-verify]``
                       with the ``install_qwen4_fixed_verify_route`` report,
                       kept as ``runtime.qwen4_fixed_verify_report``
                       (``{"installed": True, "linear_layers": n, "rows": 4}``).
``qwen4_m4_stage3``    ``mtplx/runtime.py`` prints ``[qwen4-M4-stage3]`` with
                       the ``install_qwen4_m4_stage3`` report, kept as
                       ``runtime.qwen4_m4_stage3_report``.
``qwen4_compiled_mtp_prepare``
                       ``mtplx/runtime.py`` prints
                       ``[qwen4-compiled-MTP-prepare]`` with the
                       ``install_compiled_prepare`` report, kept as
                       ``runtime.qwen4_compiled_mtp_prepare_report``.
``ladder_all_ok``      every ``{"kind": "ladder"}`` step of the server's
                       background warmup plan in state ``ok``.

Those three ``[qwen4-*]`` receipts are also printed to stderr at install
time by ``mtplx/runtime.py:_print_install_receipt``, exactly the way
``[frspec] install report`` always was. Before that they existed only as
``logger.info`` under a server that configures no handler, so the lanes
could be off with nothing in the log either way -- which is the failure this
self-check exists to end. The self-check still prints each report's contents
so the verdict and its evidence sit on one line. It does NOT re-emit the
driver's ``M4 route {...}`` spelling: that line belongs to
the benchmark driver, and a summary must not be mistakable for
the receipt it summarizes.

The same reports are readable after boot at ``GET /health`` under
``engagement_reports``.

These markers are the LANE level. The env level -- is each of the 44
measured-stack keys armed, and by whom -- is
``mtplx.full_stack_env.stack_summary_line``, and what the retained defaults
armed against what the operator turned off is
``mtplx.full_stack_env.defaults_summary_line``; the server prints both once,
just above these lines. They answer different questions: a lane can be
missing because the server's own auto-arm predicate did not match the served
pack, or because an operator exported the key off, and only the env lines can
say which.

Since 2026-09-03 these markers run on ANY Flash-Next serve, not only under
``--profile turbo-full-stack``: the retained stack is that family's default,
and a default nobody can see armed is exactly as unreadable as the opt-in was.
Per-KEY verdicts (as opposed to these per-LANE markers) are
``mtplx/fable_install_receipts.py``, whose ``[fable]`` lines now say whether
each value came from the defaults or from the operator.

Pure functions over plain dicts: the server passes what it has, nothing here
touches MLX, the model, or the GPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TextIO

#: The row count ``builtin:qwen38-code-64k`` installs, and the number
#: ``require_full_stack`` demands. A smaller ``n`` means MTPLX_FRSPEC_N
#: truncated the table or a different vocabulary was selected.
EXPECTED_FRSPEC_N = 65536

#: Marker name -> the env key that arms it and the receipt it is read from.
MARKER_SOURCES: dict[str, tuple[str, str]] = {
    "frspec_installed": ("MTPLX_FRSPEC_DRAFT", "[frspec] install report"),
    "qwen4_fixed_m4_verify": (
        "MTPLX_QWEN4_FIXED_M4_VERIFY",
        "[qwen4-fixed-M4-verify]",
    ),
    "qwen4_m4_stage3": ("MTPLX_QWEN4_M4_STAGE3", "[qwen4-M4-stage3]"),
    "qwen4_compiled_mtp_prepare": (
        "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
        "[qwen4-compiled-MTP-prepare]",
    ),
    "ladder_all_ok": ("MTPLX_WARMUP_LADDER", "background warmup ladder"),
}

#: Marker names in report order.
MARKER_NAMES = tuple(MARKER_SOURCES)

#: Prefix every self-check line carries. Deliberately NOT ``[frspec]``,
#: ``[qwen4-...]`` or a bare ``M4 route ... installed ... true``: those
#: spellings are the receipts themselves (and what the bench's log scanner
#: matches), and a summary line must not be mistaken for one.
LINE_PREFIX = "[full-stack]"


def print_install_receipt(
    tag: str,
    report: Any,
    *,
    stream: TextIO | None = None,
) -> None:
    """Print one install-time engagement receipt, the way frspec always did.

    ``mtplx/runtime.py`` logs ``[qwen4-fixed-M4-verify]`` and friends through
    ``logger.info``, and ``mtplx/server/openai.py`` installs no logging
    handler -- so on a real ``mtplx serve`` log those lines appeared zero
    times while ``[frspec] install report`` (a plain print) appeared once.
    Lanes could therefore install, or silently fail to install, with nothing
    to show either way.

    Called at INSTALL time only, once per model load: no per-request logging
    is enabled and nothing is printed from a timed request path. Never
    raises -- a receipt must not be able to fail a model load.
    """

    try:
        print(f"[{tag}] {report}", file=stream or sys.stderr, flush=True)
    except Exception:
        pass


@dataclass(frozen=True)
class MarkerStatus:
    """One engagement marker: did the lane actually install?"""

    name: str
    satisfied: bool
    detail: str

    @property
    def verdict(self) -> str:
        return "satisfied" if self.satisfied else "missing"

    @property
    def env_key(self) -> str:
        return MARKER_SOURCES[self.name][0]

    @property
    def receipt(self) -> str:
        return MARKER_SOURCES[self.name][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.name,
            "satisfied": self.satisfied,
            "env_key": self.env_key,
            "receipt": self.receipt,
            "detail": self.detail,
        }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _missing_report(name: str) -> MarkerStatus:
    env_key, receipt = MARKER_SOURCES[name]
    return MarkerStatus(
        name,
        False,
        f"no install report on the runtime; expected {env_key}=1 to produce "
        f'a "{receipt}" report on a qwen4_exp model',
    )


def _install_report_status(name: str, report: Any) -> MarkerStatus:
    """Generic ``{'installed': True, ...}`` runtime install report."""

    fields = _as_mapping(report)
    if not fields:
        return _missing_report(name)
    if not fields.get("installed"):
        return MarkerStatus(
            name,
            False,
            f"report says installed={fields.get('installed')!r}"
            + (f": {fields['reason']}" if "reason" in fields else ""),
        )
    rest = ", ".join(
        f"{key}={value!r}" for key, value in fields.items() if key != "installed"
    )
    return MarkerStatus(name, True, f"installed{': ' + rest if rest else ''}")


def _frspec_status(draft_lm_head: Any) -> MarkerStatus:
    name = "frspec_installed"
    env_key, receipt = MARKER_SOURCES[name]
    report = _as_mapping(draft_lm_head)
    frspec = _as_mapping(report.get("frspec"))
    if not frspec:
        reason = report.get("reason")
        detail = (
            "no frspec section in the draft-head install report"
            + (f" (draft head: {reason})" if reason else "")
            + f'; expected {env_key}=1 to produce "{receipt}: '
            + "{'installed': True, 'n': "
            + f"{EXPECTED_FRSPEC_N}"
            + '}"'
        )
        return MarkerStatus(name, False, detail)
    if not frspec.get("installed"):
        return MarkerStatus(
            name,
            False,
            f"frspec did not install: {frspec.get('reason', 'unknown reason')}",
        )
    observed_n = frspec.get("n")
    if observed_n != EXPECTED_FRSPEC_N:
        return MarkerStatus(
            name, False, f"frspec n={observed_n}, expected {EXPECTED_FRSPEC_N}"
        )
    return MarkerStatus(name, True, f"installed n={observed_n}")


def _ladder_steps(warmup_status: Any) -> list[Mapping[str, Any]]:
    background = _as_mapping(_as_mapping(warmup_status).get("background"))
    steps = background.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return []
    return [
        step
        for step in steps
        if isinstance(step, Mapping) and step.get("kind") == "ladder"
    ]


def _ladder_status(warmup_status: Any) -> MarkerStatus:
    name = "ladder_all_ok"
    steps = _ladder_steps(warmup_status)
    if not steps:
        background = _as_mapping(_as_mapping(warmup_status).get("background"))
        if not background:
            return MarkerStatus(
                name,
                False,
                "no background warmup plan (MTPLX_WARMUP_LADDER empty, "
                "extended warmup disabled, or warmup skipped)",
            )
        return MarkerStatus(name, False, "background warmup plan has no ladder rungs")
    states = [str(step.get("state")) for step in steps]
    contexts = [step.get("context") for step in steps]
    if all(state == "ok" for state in states):
        return MarkerStatus(name, True, f"rungs {contexts} all ok")
    pending = [
        state
        for state in states
        if state in ("pending", "running", "waiting_idle", "yielded")
    ]
    tail = (
        "; background warmup is still running, so this line is re-checked "
        "when it finishes"
        if pending
        else ""
    )
    pairs = ", ".join(f"{ctx}={state}" for ctx, state in zip(contexts, states))
    return MarkerStatus(name, False, f"rungs {pairs}{tail}")


def evaluate_full_stack_markers(
    *,
    draft_lm_head: Any = None,
    fixed_verify_report: Any = None,
    m4_stage3_report: Any = None,
    compiled_mtp_prepare_report: Any = None,
    warmup_status: Any = None,
) -> list[MarkerStatus]:
    """Did the lanes this profile arms actually install? One status each."""

    return [
        _frspec_status(draft_lm_head),
        _install_report_status("qwen4_fixed_m4_verify", fixed_verify_report),
        _install_report_status("qwen4_m4_stage3", m4_stage3_report),
        _install_report_status(
            "qwen4_compiled_mtp_prepare", compiled_mtp_prepare_report
        ),
        _ladder_status(warmup_status),
    ]


def markers_from_runtime(
    runtime: Any,
    *,
    draft_lm_head: Any = None,
    warmup_status: Any = None,
) -> list[MarkerStatus]:
    """Read the install reports straight off a loaded runtime object."""

    return evaluate_full_stack_markers(
        draft_lm_head=draft_lm_head,
        fixed_verify_report=getattr(runtime, "qwen4_fixed_verify_report", None),
        m4_stage3_report=getattr(runtime, "qwen4_m4_stage3_report", None),
        compiled_mtp_prepare_report=getattr(
            runtime, "qwen4_compiled_mtp_prepare_report", None
        ),
        warmup_status=warmup_status,
    )


def format_marker_lines(
    statuses: Sequence[MarkerStatus],
    *,
    phase: str = "startup",
) -> list[str]:
    """One line per marker, ``satisfied`` or ``missing``, with the evidence."""

    return [
        f"{LINE_PREFIX} {phase} engagement {status.name} "
        f"({status.receipt}): {status.verdict} ({status.detail})"
        for status in statuses
    ]


def selfcheck_payload(
    statuses: Sequence[MarkerStatus],
    *,
    phase: str = "startup",
) -> dict[str, Any]:
    """Plain-data form for ``/health`` and for tests."""

    return {
        "phase": phase,
        "ok": all(status.satisfied for status in statuses),
        "markers": [status.to_dict() for status in statuses],
    }


__all__ = [
    "EXPECTED_FRSPEC_N",
    "print_install_receipt",
    "LINE_PREFIX",
    "MARKER_NAMES",
    "MARKER_SOURCES",
    "MarkerStatus",
    "evaluate_full_stack_markers",
    "format_marker_lines",
    "markers_from_runtime",
    "selfcheck_payload",
]
