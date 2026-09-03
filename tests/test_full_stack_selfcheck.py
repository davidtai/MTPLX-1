"""The startup engagement self-check for ``turbo-full-stack``.

The markers are the receipts the code already produces:

* ``[frspec] install report: {'installed': True, 'n': 65536}``
  (mtplx/draft_lm_head.py -> the ``frspec`` section of the draft-head report)
* ``[qwen4-fixed-M4-verify]`` (mtplx/runtime.py -> qwen4_fixed_verify_report)
* ``[qwen4-M4-stage3]`` (mtplx/runtime.py -> qwen4_m4_stage3_report)
* ``[qwen4-compiled-MTP-prepare]``
  (mtplx/runtime.py -> qwen4_compiled_mtp_prepare_report)
* every ``{"kind": "ladder"}`` background warmup step in state ``ok``

Pure Python over plain dicts; no MLX, no model, no server boot.
"""

from __future__ import annotations

import re

from mtplx.full_stack_selfcheck import (
    EXPECTED_FRSPEC_N,
    MARKER_NAMES,
    MARKER_SOURCES,
    evaluate_full_stack_markers,
    format_marker_lines,
    markers_from_runtime,
    selfcheck_payload,
)

# Report shapes copied from the install functions they come from.
GOOD_DRAFT_HEAD = {
    "installed": True,
    "frspec": {"installed": True, "n": 65536, "vocab_rows": 248320},
}
# mtplx/qwen4_fixed_verify.py:install_qwen4_fixed_verify_route
GOOD_FIXED_VERIFY = {"installed": True, "linear_layers": 36, "rows": 4}
# mtplx/qwen4_m4_stage3.py:_installation_report
GOOD_M4_STAGE3 = {
    "installed": True,
    "layers": 48,
    "rows": 4,
    "boundary": "stock_qmm_combine_tail",
}
# mtplx/models/qwen4_exp.py:install_compiled_prepare
GOOD_MTP_PREPARE = {"installed": True, "shape": [1, 1, 2560], "dtype": "bfloat16"}
GOOD_WARMUP = {
    "background": {
        "state": "done",
        "steps": [
            {"kind": "gqa_packed_pipelines", "state": "ok"},
            {"kind": "ladder", "context": 512, "state": "ok"},
            {"kind": "ladder", "context": 2560, "state": "ok"},
        ],
    }
}

ALL_GOOD = dict(
    draft_lm_head=GOOD_DRAFT_HEAD,
    fixed_verify_report=GOOD_FIXED_VERIFY,
    m4_stage3_report=GOOD_M4_STAGE3,
    compiled_mtp_prepare_report=GOOD_MTP_PREPARE,
    warmup_status=GOOD_WARMUP,
)


class _FakeRuntime:
    def __init__(self, **reports):
        for key, value in reports.items():
            setattr(self, key, value)


def _by_name(statuses):
    return {status.name: status for status in statuses}


def test_expected_frspec_row_count_matches_the_harness() -> None:
    assert EXPECTED_FRSPEC_N == 65536


def test_marker_names_and_their_sources() -> None:
    assert MARKER_NAMES == (
        "frspec_installed",
        "qwen4_fixed_m4_verify",
        "qwen4_m4_stage3",
        "qwen4_compiled_mtp_prepare",
        "ladder_all_ok",
    )
    # Every marker names the env key that arms it and the receipt it reads.
    for name, (env_key, receipt) in MARKER_SOURCES.items():
        assert env_key.startswith("MTPLX_"), name
        assert receipt, name


def test_every_qwen4_marker_is_armed_by_a_registered_full_stack_key() -> None:
    from mtplx.full_stack_env import registered_names

    registered = set(registered_names())
    for name, (env_key, _receipt) in MARKER_SOURCES.items():
        if name == "ladder_all_ok":
            continue  # the ladder is a turbo key, not part of the restack
        assert env_key in registered, name


def test_all_markers_satisfied_on_a_fully_engaged_boot() -> None:
    statuses = evaluate_full_stack_markers(**ALL_GOOD)

    assert [status.name for status in statuses] == list(MARKER_NAMES)
    assert all(status.satisfied for status in statuses)
    assert selfcheck_payload(statuses)["ok"] is True


def test_markers_from_runtime_reads_the_reports_the_runtime_publishes() -> None:
    runtime = _FakeRuntime(
        qwen4_fixed_verify_report=GOOD_FIXED_VERIFY,
        qwen4_m4_stage3_report=GOOD_M4_STAGE3,
        qwen4_compiled_mtp_prepare_report=GOOD_MTP_PREPARE,
    )

    statuses = _by_name(
        markers_from_runtime(
            runtime,
            draft_lm_head=GOOD_DRAFT_HEAD,
            warmup_status=GOOD_WARMUP,
        )
    )

    assert all(status.satisfied for status in statuses.values())
    assert "linear_layers=36" in statuses["qwen4_fixed_m4_verify"].detail
    assert "layers=48" in statuses["qwen4_m4_stage3"].detail
    assert "dtype='bfloat16'" in statuses["qwen4_compiled_mtp_prepare"].detail


def test_a_runtime_with_no_reports_reports_every_qwen4_marker_missing() -> None:
    statuses = _by_name(markers_from_runtime(_FakeRuntime()))

    for name in (
        "qwen4_fixed_m4_verify",
        "qwen4_m4_stage3",
        "qwen4_compiled_mtp_prepare",
    ):
        assert statuses[name].satisfied is False, name
        # The line has to name the switch that would have installed it.
        assert MARKER_SOURCES[name][0] in statuses[name].detail, name


def test_the_reported_symptom_is_a_missing_frspec_marker() -> None:
    # `[frspec] disabled (MTPLX_FRSPEC_DRAFT=None)`: the draft head installed,
    # but with no frspec section at all.
    statuses = _by_name(
        evaluate_full_stack_markers(**{**ALL_GOOD, "draft_lm_head": {"installed": True}})
    )

    frspec = statuses["frspec_installed"]
    assert frspec.satisfied is False
    assert frspec.verdict == "missing"
    assert "MTPLX_FRSPEC_DRAFT=1" in frspec.detail


def test_a_truncated_frspec_table_is_missing_not_satisfied() -> None:
    statuses = _by_name(
        evaluate_full_stack_markers(
            **{
                **ALL_GOOD,
                "draft_lm_head": {"frspec": {"installed": True, "n": 32768}},
            }
        )
    )

    frspec = statuses["frspec_installed"]
    assert frspec.satisfied is False
    assert "n=32768" in frspec.detail
    assert str(EXPECTED_FRSPEC_N) in frspec.detail


def test_a_failed_frspec_install_reports_its_reason() -> None:
    statuses = _by_name(
        evaluate_full_stack_markers(
            draft_lm_head={"frspec": {"installed": False, "reason": "no_ids"}}
        )
    )

    assert statuses["frspec_installed"].satisfied is False
    assert "no_ids" in statuses["frspec_installed"].detail


def test_an_install_report_that_says_not_installed_is_missing() -> None:
    statuses = _by_name(
        evaluate_full_stack_markers(
            m4_stage3_report={"installed": False, "reason": "geometry_mismatch"}
        )
    )

    stage3 = statuses["qwen4_m4_stage3"]
    assert stage3.satisfied is False
    assert "geometry_mismatch" in stage3.detail


def test_a_ladder_still_running_is_missing_and_says_it_is_rechecked() -> None:
    warmup = {
        "background": {
            "steps": [
                {"kind": "ladder", "context": 512, "state": "ok"},
                {"kind": "ladder", "context": 2560, "state": "pending"},
            ]
        }
    }

    statuses = _by_name(evaluate_full_stack_markers(warmup_status=warmup))

    ladder = statuses["ladder_all_ok"]
    assert ladder.satisfied is False
    assert "2560=pending" in ladder.detail
    assert "re-checked" in ladder.detail


def test_a_failed_rung_is_missing_without_the_recheck_promise() -> None:
    warmup = {
        "background": {"steps": [{"kind": "ladder", "context": 512, "state": "failed"}]}
    }

    statuses = _by_name(evaluate_full_stack_markers(warmup_status=warmup))

    ladder = statuses["ladder_all_ok"]
    assert ladder.satisfied is False
    assert "512=failed" in ladder.detail
    assert "re-checked" not in ladder.detail


def test_no_background_plan_at_all_is_reported_plainly() -> None:
    statuses = _by_name(evaluate_full_stack_markers(warmup_status={}))

    assert statuses["ladder_all_ok"].satisfied is False
    assert "no background warmup plan" in statuses["ladder_all_ok"].detail


def test_evaluation_never_raises_on_junk_inputs() -> None:
    statuses = evaluate_full_stack_markers(
        draft_lm_head="not a dict",
        fixed_verify_report=[1, 2, 3],
        m4_stage3_report=object(),
        compiled_mtp_prepare_report=None,
        warmup_status={"background": {"steps": "nope"}},
    )

    assert [status.name for status in statuses] == list(MARKER_NAMES)
    assert not any(status.satisfied for status in statuses)


def test_lines_are_one_per_marker_and_do_not_impersonate_the_real_receipts() -> None:
    statuses = evaluate_full_stack_markers(**ALL_GOOD)
    lines = format_marker_lines(statuses, phase="startup")

    assert len(lines) == len(MARKER_NAMES)
    for line, name in zip(lines, MARKER_NAMES):
        assert line.startswith("[full-stack] startup engagement ")
        assert name in line
        assert MARKER_SOURCES[name][1] in line
        assert "satisfied" in line

    # A summary line must not be mistakable for the receipt it summarizes,
    # nor for the driver-only "M4 route" line the server never prints
    # (the benchmark harness's own ENGAGEMENT_MARKERS list).
    blob = "\n".join(lines)
    assert not re.search(r"\[frspec\] install report.*'installed': True", blob)
    assert not re.search(r"M4 route .*installed.*[Tt]rue", blob)


def test_payload_carries_the_phase_env_key_and_receipt_per_marker() -> None:
    payload = selfcheck_payload(
        evaluate_full_stack_markers(warmup_status=GOOD_WARMUP), phase="post-warmup"
    )

    assert payload["phase"] == "post-warmup"
    assert payload["ok"] is False
    assert [row["marker"] for row in payload["markers"]] == list(MARKER_NAMES)
    for row in payload["markers"]:
        assert row["env_key"] == MARKER_SOURCES[row["marker"]][0]
        assert row["receipt"] == MARKER_SOURCES[row["marker"]][1]


# ---------------------------------------------------------------------------
# Install-time receipts: the [qwen4-*] lines a real `mtplx serve` log shows
# ---------------------------------------------------------------------------


def test_install_receipt_prints_the_same_text_the_logger_would() -> None:
    import io

    from mtplx.full_stack_selfcheck import print_install_receipt

    stream = io.StringIO()
    print_install_receipt("qwen4-fixed-M4-verify", GOOD_FIXED_VERIFY, stream=stream)

    assert stream.getvalue() == f"[qwen4-fixed-M4-verify] {GOOD_FIXED_VERIFY}\n"


def test_install_receipt_never_raises() -> None:
    from mtplx.full_stack_selfcheck import print_install_receipt

    class _Broken:
        def write(self, _data):
            raise OSError("closed")

        def flush(self):
            raise OSError("closed")

    # Must not propagate: a receipt cannot be allowed to fail a model load.
    print_install_receipt("qwen4-M4-stage3", GOOD_M4_STAGE3, stream=_Broken())


def test_runtime_prints_a_receipt_for_every_qwen4_install_report() -> None:
    """Each report the self-check reads must also reach a plain server log.

    mtplx/server/openai.py installs no logging handler, so a logger.info-only
    receipt appears zero times in a real `mtplx serve` log. Read the source
    rather than importing mtplx.runtime (which pulls MLX in).
    """

    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "mtplx" / "runtime.py").read_text()

    assert "from .full_stack_selfcheck import print_install_receipt" in source
    for name, (_env_key, receipt) in MARKER_SOURCES.items():
        if not name.startswith("qwen4_"):
            continue
        tag = receipt.strip("[]")
        assert f'logger.info(\n                "[{tag}]' in source or (
            f'logger.info("[{tag}]' in source
        ), tag
        assert f'_print_install_receipt(\n                "{tag}"' in source or (
            f'_print_install_receipt("{tag}"' in source
        ), tag


def test_health_engagement_reports_registry_covers_every_runtime_report() -> None:
    """``/health`` must publish every install report the runtime publishes.

    ``_ENGAGEMENT_REPORT_ATTRS`` in mtplx/server/openai.py is the single
    registration point: one ``(health name, runtime attribute)`` row per
    lane, read with ``getattr`` only, so adding a row cannot change what the
    lane itself prints. A new install-time lane that sets
    ``runtime.<something>_report`` and is not registered here is invisible at
    ``/health`` -- which is the failure this registry exists to end.

    Both files are read as source: importing mtplx.runtime pulls MLX in.
    """

    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[1] / "mtplx"
    published = set(
        re.findall(r"runtime\.([a-z0-9_]+_report)\s*=", (root / "runtime.py").read_text())
    )
    registered = set(
        re.findall(
            r'\(\s*"[a-z0-9_]+"\s*,\s*"([a-z0-9_]+_report)"\s*\)',
            (root / "server" / "openai.py").read_text(),
        )
    )

    assert published, "the runtime scan found no install reports; the pattern rotted"
    assert published <= registered, sorted(published - registered)


def test_every_runtime_published_install_artifact_reaches_health() -> None:
    """The ``_report`` scan above is a suffix match; this one is not.

    The receipt registry publishes ``runtime.fable_install_receipts`` -- an install artifact
    with no ``_report`` suffix, so the scan above passes it by. It reaches
    ``/health`` through its OWN payload function rather than through
    ``_ENGAGEMENT_REPORT_ATTRS``, and that is correct: the two registries
    answer different questions (``engagement_reports`` reports LANES, keyed
    by lane, off a bare ``getattr``; ``fable_install_receipts`` reports
    FLAGS, and re-reads live so per-request engagement counters are current).
    Registering it in both would publish it twice under two names.

    So assert the invariant that actually matters -- every install artifact
    the runtime hands back is served somewhere at ``/health`` -- and name the
    route for each, so a third registry cannot be added without a decision.
    """

    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[1] / "mtplx"
    runtime_source = (root / "runtime.py").read_text()
    server_source = (root / "server" / "openai.py").read_text()

    published = set(re.findall(r"runtime\.([a-z0-9_]+)\s*=", runtime_source))
    install_artifacts = {
        name
        for name in published
        if name.endswith("_report") or name.endswith("_receipts")
    }
    assert "fable_install_receipts" in install_artifacts, sorted(install_artifacts)

    via_engagement_reports = set(
        re.findall(
            r'\(\s*"[a-z0-9_]+"\s*,\s*"([a-z0-9_]+_report)"\s*\)',
            server_source,
        )
    )
    # The receipts' own route: a top-level /health key fed by its payload function.
    via_own_payload = {"fable_install_receipts"}
    assert '"fable_install_receipts": _fable_install_receipts_payload(state)' in (
        server_source
    )
    assert '"engagement_reports": _engagement_reports_payload(state)' in server_source

    unserved = install_artifacts - via_engagement_reports - via_own_payload
    assert not unserved, sorted(unserved)


def test_hc_m4_install_report_reaches_health() -> None:
    """The MTPLX_FABLE_HC_M4 pack validation is a registered lane."""

    from mtplx.server.openai import _engagement_reports_payload

    class _Runtime:
        qwen4_hc_m4_report = {"armed": True, "validated": 49}

    class _State:
        runtime = _Runtime()

    payload = _engagement_reports_payload(_State())
    assert payload["qwen4_hc_m4"] == {"armed": True, "validated": 49}
    # Unset lanes stay None rather than disappearing.
    assert payload["qwen4_m4_stage3"] is None
