"""W84 -- every armed flag prints an install-time verdict, exactly once.

The program's rule is that a benchmark must be able to prove the stack it
measured.  Nine keys of the served set had no install-time receipt anywhere in
the tree; :mod:`mtplx.fable_install_receipts` gives each one a verdict line, a
receipt dict, and (for the lanes that only run on a request that fits) an
engagement counter.

The gate that matters most is :func:`test_every_served_key_has_a_verdict`: it
names the keys explicitly, so a flag added later without a verdict fails CI
rather than shipping as an unprovable arm.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mtplx import fable_install_receipts as receipts_mod


ROOT = Path(__file__).resolve().parents[1]

#: The nine keys the pre-battery sanity stage found with no install-time
#: receipt.  ``MTPLX_FABLE_OPDIET_ITEMS`` rides its master switch's lane.
SERVED_KEYS_WITHOUT_RECEIPTS = (
    "MTPLX_FABLE_OPDIET",
    "MTPLX_FABLE_OPDIET_ITEMS",
    "MTPLX_FABLE_BLOCK_VERIFY",
    "MTPLX_FABLE_DRAFT_K20_PRESCATTER",
    "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD",
    "MTPLX_FABLE_PLE_FIRST_GATHER_EARLY",
    "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE",
    "MTPLX_PREFILL_CHUNK_SIZE",
    "MTPLX_QSA_PREFILL_COMPILE_ROWS",
    "MTPLX_SESSION_BANK_MAX_BYTES",
)

#: Every key's lane, so the test names both halves of the mapping.
KEY_LANES = {
    "MTPLX_FABLE_OPDIET": "opdiet",
    "MTPLX_FABLE_OPDIET_ITEMS": "opdiet",
    "MTPLX_FABLE_BLOCK_VERIFY": "block_verify",
    "MTPLX_FABLE_DRAFT_K20_PRESCATTER": "draft_k20_prescatter",
    "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD": "ple_prefill_lookahead",
    "MTPLX_FABLE_PLE_FIRST_GATHER_EARLY": "ple_first_gather_early",
    "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE": "prefill_qsa_query_tile",
    "MTPLX_PREFILL_CHUNK_SIZE": "prefill_chunk_size",
    "MTPLX_QSA_PREFILL_COMPILE_ROWS": "qsa_prefill_compile_rows",
    "MTPLX_SESSION_BANK_MAX_BYTES": "session_bank_max_bytes",
    # W93 (2026-09-03): the retained stack is armed by DEFAULT for a
    # Flash-Next serve, and the program's rule is that every armed flag
    # prints an install-time verdict. These three had none. (Before W93 this
    # table was the nine-key/9-lane pre-battery set above.)
    "MTPLX_FABLE_ROUTE_KERNEL": "route_kernel",
    "MTPLX_FABLE_GRAPH_BUILD_OVERLAP": "graph_build_overlap",
    "MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS": "graph_build_overlap",
    "MTPLX_GDN_BLOCKED_PREFILL": "gdn_blocked_prefill",
}

#: The env that arms every armable key at once, for the one-line-per-lane and
#: exactly-once checks.  The two width knobs move together on purpose: the
#: coherence rule refuses a full serving width that does not match
#: ``MTPLX_QSA_PREFILL_COMPILE_ROWS``.
ARMED_ENV = {
    "MTPLX_FABLE_OPDIET": "1",
    "MTPLX_FABLE_OPDIET_ITEMS": "k20",
    "MTPLX_FABLE_BLOCK_VERIFY": "1",
    "MTPLX_FABLE_DRAFT_K20_PRESCATTER": "1",
    "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD": "1",
    "MTPLX_FABLE_PLE_FIRST_GATHER_EARLY": "1",
    "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE": "2048",
    "MTPLX_PREFILL_CHUNK_SIZE": "4096",
    "MTPLX_QSA_PREFILL_COMPILE_ROWS": "4096",
    "MTPLX_SESSION_BANK_MAX_BYTES": "8G",
    "MTPLX_FABLE_ROUTE_KERNEL": "1",
    "MTPLX_FABLE_GRAPH_BUILD_OVERLAP": "1",
    "MTPLX_FABLE_GRAPH_BUILD_OVERLAP_LAYERS": "3",
    "MTPLX_GDN_BLOCKED_PREFILL": "1",
}


@pytest.fixture(autouse=True)
def _clean_registry():
    receipts_mod.reset_for_tests()
    yield
    receipts_mod.reset_for_tests()


def _run_in_subprocess(source: str, env_overrides: dict[str, str]) -> tuple[str, str]:
    """Run ``source`` in a fresh interpreter; return ``(stdout, stderr)``.

    A subprocess is not optional here.  Five of the nine keys are resolved at
    import (``mtplx.runtime_options``, ``mtplx.fable_block_verify``,
    ``mtplx.fable_draft_k20_prescatter``) or behind an ``lru_cache``
    (``ple_row_gather.enabled``), which is exactly the property the verdicts
    report -- so an in-process ``monkeypatch.setenv`` would test a different
    stack from the one that ships.
    """

    env = dict(os.environ)
    for key in SERVED_KEYS_WITHOUT_RECEIPTS:
        env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout, completed.stderr


# ---------------------------------------------------------------------------
# The CI gate: no flag without a verdict
# ---------------------------------------------------------------------------
def test_every_served_key_has_a_verdict():
    """Each of the nine keys resolves to a registered lane with a verdict.

    This is the guard the W84 sanity stage exists to make permanent: a key
    added to the served set without an install-time verdict fails HERE.
    """

    missing = [
        key
        for key in SERVED_KEYS_WITHOUT_RECEIPTS
        if key not in receipts_mod.REGISTERED_KEYS
    ]
    assert not missing, f"keys with no registered install verdict: {missing}"
    for key, lane in KEY_LANES.items():
        assert receipts_mod.lane_for_key(key) == lane
        verdict = receipts_mod.verdict_for_key(key)
        assert verdict.state in {
            receipts_mod.STATE_ARMED,
            receipts_mod.STATE_OFF,
            receipts_mod.STATE_REFUSED,
            receipts_mod.STATE_RESOLVED,
        }
        assert verdict.line.startswith(f"[fable] {lane} ")
        assert key in verdict.keys


def test_registry_key_index_matches_the_lane_table():
    """No lane owns a key the tests do not know about, and vice versa."""

    assert set(receipts_mod.REGISTERED_KEYS) == set(KEY_LANES)
    assert set(receipts_mod.LANES) == set(KEY_LANES.values())
    flattened = {
        key for keys in receipts_mod.LANE_KEYS.values() for key in keys
    }
    assert flattened == set(KEY_LANES)


def test_every_lane_states_an_engagement_condition_or_needs_none():
    """A lane that engages per request must carry an engagement counter.

    ``armed, engages at ...`` is a promise that "armed" and "ran" are separate
    claims, and the receipt has to be able to tell them apart.
    """

    for lane in receipts_mod.LANES:
        block = receipts_mod.receipts()[lane]
        assert "engagements" in block
        assert isinstance(block["declines"], dict)


# ---------------------------------------------------------------------------
# One line per lane, exactly once
# ---------------------------------------------------------------------------
def test_armed_process_prints_each_verdict_exactly_once():
    """Every lane prints one line to stderr, even if emit_all runs twice."""

    _, err = _run_in_subprocess(
        """
        from mtplx import fable_install_receipts as r
        r.emit_all()
        r.emit_all()      # idempotent: a second install path prints nothing
        """,
        ARMED_ENV,
    )
    for lane in receipts_mod.LANES:
        prefix = f"[fable] {lane} "
        hits = [line for line in err.splitlines() if line.startswith(prefix)]
        assert len(hits) == 1, f"{lane}: expected one verdict line, got {hits}"


def test_armed_process_states_armed_or_resolved_for_every_key():
    """With everything armed, no lane reports itself off."""

    out, _ = _run_in_subprocess(
        """
        import json
        from mtplx import fable_install_receipts as r
        print(json.dumps({
            lane: block["state"] for lane, block in r.receipts().items()
        }))
        """,
        ARMED_ENV,
    )
    states = json.loads(out.strip().splitlines()[-1])
    assert states == {
        "opdiet": "armed",
        "block_verify": "armed",
        "draft_k20_prescatter": "armed",
        "ple_prefill_lookahead": "armed",
        "ple_first_gather_early": "armed",
        "prefill_qsa_query_tile": "armed",
        "prefill_chunk_size": "resolved",
        "qsa_prefill_compile_rows": "resolved",
        "session_bank_max_bytes": "resolved",
        # W93: the three lanes that had no verdict before the retained stack
        # became a served default.
        "route_kernel": "armed",
        "graph_build_overlap": "armed",
        "gdn_blocked_prefill": "armed",
    }


def test_unarmed_process_names_the_key_in_every_off_reason():
    """An ``off`` verdict says WHICH key was unset, not just that it is off."""

    out, err = _run_in_subprocess(
        """
        import json
        from mtplx import fable_install_receipts as r
        r.emit_all()
        print(json.dumps({
            lane: [block["state"], block["reason"]]
            for lane, block in r.receipts().items()
        }))
        """,
        {},
    )
    verdicts = json.loads(out.strip().splitlines()[-1])
    for lane, keys in receipts_mod.LANE_KEYS.items():
        state, reason = verdicts[lane]
        if state == "resolved":
            continue
        assert state == "off", f"{lane}: {state} ({reason})"
        assert keys[0] in reason, f"{lane} off reason does not name its key: {reason}"
        assert "unset" in reason
    for lane in receipts_mod.LANES:
        assert sum(
            line.startswith(f"[fable] {lane} ") for line in err.splitlines()
        ) == 1


def test_resolved_knobs_print_their_value_even_when_unset():
    """The three server knobs always have a value, so they always report one."""

    _, err = _run_in_subprocess(
        """
        from mtplx import fable_install_receipts as r
        r.emit_all()
        """,
        {},
    )
    lines = {
        line.split()[1]: line for line in err.splitlines() if line.startswith("[fable] ")
    }
    assert "2048" in lines["prefill_chunk_size"]
    assert "2048" in lines["qsa_prefill_compile_rows"]
    assert "auto" in lines["session_bank_max_bytes"]
    for lane in ("prefill_chunk_size", "qsa_prefill_compile_rows", "session_bank_max_bytes"):
        assert " resolved: " in lines[lane]


# ---------------------------------------------------------------------------
# Receipt dicts round-trip
# ---------------------------------------------------------------------------
def test_receipt_blocks_round_trip_through_json():
    block = receipts_mod.receipts()
    assert json.loads(json.dumps(block)) == block
    for lane, entry in block.items():
        assert entry["lane"] == lane
        assert entry["line"] == receipts_mod.verdict(lane).line
        assert set(entry) >= {
            "lane",
            "keys",
            "state",
            "reason",
            "engages_at",
            "detail",
            "decided_at",
            "readers",
            "env",
            "fields",
            "line",
            "engagements",
            "declines",
            "recorded",
            "emitted",
        }


def test_verdict_to_dict_matches_its_line():
    for lane in receipts_mod.LANES:
        verdict = receipts_mod.verdict(lane)
        assert verdict.to_dict()["line"] == verdict.line


def test_emitted_flag_tracks_the_printed_line(capsys):
    line = receipts_mod.emit("block_verify", stream=sys.stderr)
    assert line is not None
    assert receipts_mod.emit("block_verify", stream=sys.stderr) is None
    assert receipts_mod.emitted_lines()["block_verify"] == line
    assert receipts_mod.receipts()["block_verify"]["emitted"] is True
    assert capsys.readouterr().err.count(line) == 1


# ---------------------------------------------------------------------------
# Engagement counters
# ---------------------------------------------------------------------------
def test_engagement_and_decline_counters_reach_the_receipt():
    receipts_mod.note_engagement("block_verify")
    receipts_mod.note_engagement("block_verify")
    receipts_mod.note_decline("block_verify", "rows_not_all_on_host")
    block = receipts_mod.receipts()["block_verify"]
    assert block["engagements"] == 2
    assert block["declines"] == {"rows_not_all_on_host": 1}


def test_query_tile_engagements_read_the_lane_s_own_counter():
    """The tiled attention path already counts; the receipt must not double it."""

    from mtplx.models import qwen4_exp

    before = receipts_mod.counters("prefill_qsa_query_tile")["engagements"]
    qwen4_exp._qsa_prefill_count("query_tile")
    after = receipts_mod.counters("prefill_qsa_query_tile")["engagements"]
    assert after == before + 1


def test_recorded_facts_reach_the_receipt():
    receipts_mod.record("session_bank_max_bytes", bank_max_bytes=1234)
    assert receipts_mod.receipts()["session_bank_max_bytes"]["recorded"] == {
        "bank_max_bytes": 1234
    }


# ---------------------------------------------------------------------------
# Refusals: an arm that provably cannot engage says so at install
# ---------------------------------------------------------------------------
def test_query_tile_wider_than_the_chunk_is_refused():
    """``tile >= S`` makes ``_qsa_dense_attention`` take the untiled path."""

    _, err = _run_in_subprocess(
        """
        from mtplx import fable_install_receipts as r
        r.emit_all()
        """,
        {
            "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE": "4096",
            "MTPLX_PREFILL_CHUNK_SIZE": "2048",
        },
    )
    line = next(
        line
        for line in err.splitlines()
        if line.startswith("[fable] prefill_qsa_query_tile ")
    )
    assert " refused (" in line
    assert "4096" in line and "2048" in line


class _FakeInner:
    """A model node shaped like the qwen4 text model, with no PLE stage."""

    _ple_stage_idx = None


class _FakeRuntime:
    def __init__(self, model):
        self.model = model


def test_ple_lanes_refuse_a_model_with_no_ple_stage(monkeypatch):
    """``_ple_stage_idx is None`` is the silent-nothing case, so it prints.

    ``Model.ple_prefill_lookahead`` and ``Model.ple_first_gather_early`` both
    return ``None`` for every request when the model has no PLE stage layer --
    no raise, no counter, no log.  The verdict says ``refused`` instead of
    leaving the operator to find a missing delta.  It does NOT raise: turning
    this into a load failure is a behaviour change nobody asked for.
    """

    monkeypatch.setenv("MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD", "1")
    monkeypatch.setenv("MTPLX_FABLE_PLE_FIRST_GATHER_EARLY", "1")
    from mtplx import ple_prefill_lookahead, ple_row_gather

    ple_prefill_lookahead.enabled.cache_clear()
    ple_prefill_lookahead.early_enabled.cache_clear()
    ple_row_gather.enabled.cache_clear()
    try:
        context = {"runtime": _FakeRuntime(_FakeInner())}
        for lane in ("ple_prefill_lookahead", "ple_first_gather_early"):
            verdict = receipts_mod.verdict(lane, context)
            assert verdict.state == receipts_mod.STATE_REFUSED, verdict.line
            assert "no PLE stage" in verdict.reason
            assert verdict.line.startswith(f"[fable] {lane} refused (")
    finally:
        ple_prefill_lookahead.enabled.cache_clear()
        ple_prefill_lookahead.early_enabled.cache_clear()
        ple_row_gather.enabled.cache_clear()


def test_ple_lanes_do_not_refuse_when_the_model_is_unknown():
    """No model in the context is not evidence that the lane cannot run."""

    verdict = receipts_mod.verdict("ple_prefill_lookahead", {})
    assert verdict.state != receipts_mod.STATE_REFUSED


def test_opdiet_refuses_a_family_scoped_selection_on_another_family():
    """``bank``/``rope``/``resid`` are gated only inside qwen4_exp."""

    out, _ = _run_in_subprocess(
        """
        import json
        from mtplx import fable_install_receipts as r


        class Inner:
            pass


        class Runtime:
            model = Inner()


        print(json.dumps(r.verdict("opdiet", {"runtime": Runtime()}).to_dict()))
        """,
        {"MTPLX_FABLE_OPDIET": "1", "MTPLX_FABLE_OPDIET_ITEMS": "rope,resid"},
    )
    verdict = json.loads(out.strip().splitlines()[-1])
    assert verdict["state"] == "refused"
    assert "qwen4_exp" in verdict["reason"]
    assert verdict["line"].startswith("[fable] opdiet refused (")


def test_opdiet_does_not_refuse_a_selection_that_includes_k20():
    """``k20`` is gated in generation.py/fast_sampling.py: every family runs it."""

    out, _ = _run_in_subprocess(
        """
        import json
        from mtplx import fable_install_receipts as r


        class Inner:
            pass


        class Runtime:
            model = Inner()


        print(json.dumps(r.verdict("opdiet", {"runtime": Runtime()}).to_dict()))
        """,
        {"MTPLX_FABLE_OPDIET": "1", "MTPLX_FABLE_OPDIET_ITEMS": "k20"},
    )
    verdict = json.loads(out.strip().splitlines()[-1])
    assert verdict["state"] == "armed"


# ---------------------------------------------------------------------------
# The verdicts observe; they never change what a flag does
# ---------------------------------------------------------------------------
def test_a_broken_verdict_cannot_fail_a_model_load(monkeypatch):
    """``verdict`` swallows its own failure and says so in the reason."""

    entry = receipts_mod._REGISTRY["block_verify"]

    def _boom(_context):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(entry, "build", _boom)
    verdict = receipts_mod.verdict("block_verify")
    assert verdict.state == receipts_mod.STATE_OFF
    assert "probe exploded" in verdict.reason


def _guarded_calls(module_path: Path, guard_name: str) -> list[ast.Call]:
    """Every ``note_engagement``/``note_decline`` call under ``guard_name``."""

    tree = ast.parse(module_path.read_text())
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)
        }
        if guard_name not in names:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"note_engagement", "note_decline"}
            ):
                found.append(child)
    return found


def test_block_verify_counters_live_inside_the_flag_s_own_guard():
    """An unarmed process must not run one extra statement for the receipt.

    The block-verification counters sit in the accept loop, the hottest code
    the receipt touches.  They are inside ``if _FABLE_BLOCK_VERIFY and ...``,
    so a flag-off run short-circuits before either call.
    """

    generation = ROOT / "mtplx" / "generation.py"
    guarded = _guarded_calls(generation, "_FABLE_BLOCK_VERIFY")
    assert len(guarded) == 2, (
        "the block-verification counters must be the only two under "
        f"`if _FABLE_BLOCK_VERIFY ...`; found {len(guarded)}"
    )
    source = generation.read_text()
    assert source.count("_fable_install_receipts.note_engagement(") == 1
    assert source.count("_fable_install_receipts.note_decline(") == 1


def test_k20_prescatter_counters_live_behind_the_import_time_gate():
    """``claim_draft_route`` returns before either counter when unarmed."""

    source = (ROOT / "mtplx" / "fable_draft_k20_prescatter.py").read_text()
    gate = source.index("    if not _ENABLED:\n        return None")
    assert source.index("note_engagement(") > gate
    assert source.index("note_decline(") > gate
