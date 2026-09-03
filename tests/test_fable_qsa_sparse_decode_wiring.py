"""W68 -- the WIRING of ``MTPLX_FABLE_QSA_SPARSE_DECODE``, not its arithmetic.

``tests/test_fable_qsa_sparse_decode.py`` pins the kernel's selection model,
its split geometry and its parity gates.  This file pins the thing that
actually failed on 2026-09-02: the lane was armed, the cache installed, and
the kernel never ran, because the ONE call site that could reach the verify
width asked the DRAFT question.

THE DEFECT, reproduced here on the CPU stream with stub shapes.

``Indexer._call_rows`` takes the fused-M4 branch -- and therefore
``_select_m4``, the only call site that asked ``draft=False`` -- only when
``MTPLX_FABLE_QSA_M4`` is armed (``_m4_route``).  That flag is SEPARATE from
the fixed-M4 verifier and was not in the window's environment, so a
fixed-capacity S=4 verify fell through ``legacy_fused=False`` into
``_select_eager``, whose sparse-decode question was hard-coded ``draft=True``.
It read ``fable_qsa_sparse_draft_rows`` -- 0, because the draft flag was not
armed either -- returned False, and handed attention the rows-gather lane.

Every test below is host-side.  Nothing evaluates an MLX array, and no test
touches the GPU: the routing decision this file is about is decided entirely
from python ints and cache attributes, which is exactly why it could go wrong
without leaving a mark on a receipt.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.kernels import qsa_sparse_decode as lane
from mtplx.models.qwen4_exp import Attention, QSAIndexer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_lane():
    lane.reset_for_tests()
    yield
    lane.reset_for_tests()


@pytest.fixture
def armed_verify(monkeypatch):
    """Arm the verify width only, exactly as the 2026-09-02 window did."""

    from mtplx import runtime_options

    monkeypatch.setattr(runtime_options, "_FABLE_QSA_SPARSE_DECODE", True)
    monkeypatch.setattr(runtime_options, "_FABLE_QSA_SPARSE_DRAFT", False)
    return runtime_options


def fixed_cache(*, decode_rows: int = 4, draft_rows: int = 0) -> SimpleNamespace:
    """The state a ``TensorOffsetQSACache`` hands the routing predicate."""

    return SimpleNamespace(
        fixed_capacity=True,
        fable_qsa_sparse_decode_rows=int(decode_rows),
        fable_qsa_sparse_draft_rows=int(draft_rows),
    )


def indexer(*, ratio: int = 4, block_topk: int = 512) -> SimpleNamespace:
    """The two module attributes the predicate reads, bound to the real method."""

    stub = SimpleNamespace(ratio=int(ratio), block_topk=int(block_topk))
    stub._sparse_decode_route = QSAIndexer._sparse_decode_route.__get__(stub)
    return stub


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------
def test_the_verify_width_routes_on_the_stack_that_measured_the_control(
    armed_verify,
):
    """rows=4, verify armed, draft NOT armed: the kernel must serve."""

    assert indexer()._sparse_decode_route(
        fixed_cache(), rows=4, k_eff=512, draft=False, site="select_eager_verify"
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 1
    assert counters["route_sites"] == {"select_eager_verify": 1}


def test_the_draft_question_at_the_verify_width_is_what_declined(armed_verify):
    """The exact predicate the shipped code asked, and its exact answer.

    ``draft=True`` at 4 rows reads a width the flag never armed, so it can
    only ever be False -- and before this fix that was the ONLY question the
    fixed-M4 verify's selector asked.
    """

    assert not indexer()._sparse_decode_route(
        fixed_cache(),
        rows=4,
        k_eff=512,
        draft=True,
        site="select_eager_draft",
    )
    assert lane.route_counters()["route_hits"] == 0


def test_the_eager_selector_asks_the_verify_width_not_only_the_draft_one():
    """``_select_eager`` is the selector a fixed-M4 verify actually reaches."""

    source = inspect.getsource(QSAIndexer._select_eager)
    calls = re.findall(r"_sparse_decode_route\((.*?)\)\n", source, re.S)
    assert len(calls) == 2, calls
    joined = " ".join(calls)
    assert "draft=False" in joined
    assert "draft=True" in joined


def test_every_route_call_site_names_itself():
    """A single total would not have shown which site declined."""

    tree = ast.parse((ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute) and func.attr == "_sparse_decode_route"
        ):
            continue
        keywords = {kw.arg for kw in node.keywords}
        assert "site" in keywords, ast.dump(node)
        for kw in node.keywords:
            if kw.arg == "site":
                sites.append(kw.value.value)
    assert sorted(sites) == [
        "select_eager_draft",
        "select_eager_verify",
        "select_m4_verify",
    ]


def test_the_m4_selector_is_unreachable_without_its_own_flag():
    """Why the fixed-M4 verify never saw ``_select_m4``.

    ``_m4_route`` gates on ``MTPLX_FABLE_QSA_M4``, which is a different
    environment variable from the fixed-M4 verifier the window armed.
    """

    source = inspect.getsource(QSAIndexer._m4_route)
    assert "fable_qsa_m4_enabled()" in source
    call_rows = inspect.getsource(QSAIndexer._call_rows)
    assert "fused_m4 = fixed_capacity and self._m4_route(cache, S)" in call_rows
    assert "return self._select_m4(q, pos_start, cache, pooled)" in call_rows


# ---------------------------------------------------------------------------
# Routing vs failure -- the two kinds of "no"
# ---------------------------------------------------------------------------
def test_an_unarmed_width_is_one_cached_bool_test(armed_verify):
    """The OFF path must not do bookkeeping: it runs per QSA layer per forward."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(), rows=1, k_eff=512, draft=True, site="select_eager_draft"
    )
    assert lane.route_counters()["route_declines"] == {}
    assert lane.route_counters()["route_hits"] == 0


def test_the_off_path_reads_the_flag_before_anything_else():
    """Mirrors ``_m4_route``: an unarmed process pays one cached-bool test."""

    source = inspect.getsource(QSAIndexer._sparse_decode_route)
    body = source[source.index('"""', source.index('"""') + 3) + 3 :]
    flag = body.index("fable_qsa_sparse_decode_enabled()")
    assert flag < body.index("getattr(cache")
    assert flag < body.index("from mtplx.kernels import qsa_sparse_decode")


def test_a_growable_cache_is_routing_and_is_counted(armed_verify):
    growable = SimpleNamespace(fixed_capacity=False)
    assert not indexer()._sparse_decode_route(
        growable, rows=4, k_eff=512, draft=False, site="select_eager_verify"
    )
    assert lane.route_counters()["route_declines"] == {
        "select_eager_verify: growable cache": 1
    }


def test_a_width_the_predicate_was_not_asked_about_returns_quietly(armed_verify):
    """A 16 K prefill row count is neither width; no decline, no hit."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(), rows=16384, k_eff=512, draft=False, site="select_eager_verify"
    )
    assert lane.route_counters()["route_declines"] == {}


def test_a_cache_built_without_the_lane_raises_at_the_armed_width(armed_verify):
    with pytest.raises(RuntimeError, match="fable_qsa_sparse_decode_rows=0"):
        indexer()._sparse_decode_route(
            fixed_cache(decode_rows=0),
            rows=4,
            k_eff=512,
            draft=False,
            site="select_eager_verify",
        )


def test_a_geometry_the_metallib_is_not_built_for_raises(armed_verify):
    with pytest.raises(RuntimeError, match="ratio-4 top-512"):
        indexer(ratio=2)._sparse_decode_route(
            fixed_cache(), rows=4, k_eff=512, draft=False, site="select_eager_verify"
        )


# ---------------------------------------------------------------------------
# Short context: ROUTING, not failure.  A HumanEval prompt must be served.
# ---------------------------------------------------------------------------
def test_a_short_context_routes_to_stock_instead_of_returning_500(armed_verify):
    """The 2026-09-02 fullset regression, as a unit.

    The server saw ``selects 7 blocks`` at warmup and ``selects 33 blocks`` on
    a chat completion, and raised both times.  Context length is a per-request
    shape the server must accept and the kernel has no analogue below a full
    budget, so this is the routing case in the contract.
    """

    assert not indexer()._sparse_decode_route(
        fixed_cache(), rows=4, k_eff=33, draft=False, site="select_eager_verify"
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 0
    assert counters["short_context"] == 1
    assert counters["route_declines"] == {"select_eager_verify: short_context": 1}


def test_the_short_context_decline_key_stays_bounded(armed_verify):
    """One key per site, never one per context length; blocks ride min/max."""

    route = indexer()._sparse_decode_route
    for blocks in (7, 33, 511, 33):
        assert not route(
            fixed_cache(), rows=4, k_eff=blocks, draft=False,
            site="select_eager_verify",
        )
    counters = lane.route_counters()
    assert counters["route_declines"] == {"select_eager_verify: short_context": 4}
    assert counters["short_context"] == 4
    assert counters["short_context_blocks"] == {"min": 7, "max": 511}
    assert counters["short_context_tokens"] == 2052


def test_the_threshold_is_the_full_budget_the_abi_needs():
    assert lane.SHORT_CONTEXT_TOKENS == (lane.TOP_K + 1) * lane.COMPRESS_RATIO == 2052


def test_a_full_budget_forward_still_binds_after_a_short_one(armed_verify):
    """Short requests must not poison the long-context arm."""

    route = indexer()._sparse_decode_route
    assert not route(
        fixed_cache(), rows=4, k_eff=33, draft=False, site="select_eager_verify"
    )
    assert route(
        fixed_cache(), rows=4, k_eff=512, draft=False, site="select_eager_verify"
    )
    counters = lane.route_counters()
    assert counters["route_hits"] == 1
    assert counters["short_context"] == 1


def test_a_short_context_prints_nothing(armed_verify, capsys):
    """Once per QSA layer per short request: a print here is a log flood."""

    indexer()._sparse_decode_route(
        fixed_cache(), rows=4, k_eff=33, draft=False, site="select_eager_verify"
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_nothing_raises_when_the_flag_is_off():
    """The whole predicate is one host-side compare when nobody armed it."""

    assert not indexer()._sparse_decode_route(
        fixed_cache(decode_rows=0),
        rows=4,
        k_eff=17,
        draft=False,
        site="select_eager_verify",
    )
    assert lane.route_counters()["route_hits"] == 0


# ---------------------------------------------------------------------------
# The per-layer proof inside the traced verify body
# ---------------------------------------------------------------------------
def required(cache, rows, *, indexer_present=True):
    stub = SimpleNamespace(indexer=object() if indexer_present else None)
    return Attention._sparse_decode_required(stub, cache, rows)


def test_the_armed_verify_width_on_a_fixed_cache_is_required(armed_verify):
    assert required(fixed_cache(), 4)


def test_an_unarmed_draft_width_is_not_required(armed_verify):
    assert not required(fixed_cache(), 1)


def test_a_growable_cache_is_never_required(armed_verify):
    assert not required(SimpleNamespace(fixed_capacity=False), 4)


def test_a_layer_without_an_indexer_is_never_required(armed_verify):
    assert not required(fixed_cache(), 4, indexer_present=False)


def guard(sel_mask, *, rows=4, before=None):
    stub = SimpleNamespace()
    return Attention._require_sparse_decode_lane(
        stub,
        sel_mask,
        rows=rows,
        before=before or {"route_hits": 0, "short_context": 0},
    )


def test_the_guard_accepts_the_sparse_lane(armed_verify):
    guard(("sparse_blocks", object()))


def test_the_guard_accepts_a_forward_that_routed_for_short_context(armed_verify):
    """The HumanEval case: the indexer declined, and that is correct."""

    before = lane.route_snapshot()
    lane.note_short_context("select_eager_verify", 33)
    guard(("gather_rows", object(), object()), before=before)


def test_the_guard_refuses_a_full_budget_forward_on_another_lane(armed_verify):
    with pytest.raises(RuntimeError, match="gather_rows"):
        guard(("gather_rows", object(), object()))


@pytest.mark.parametrize(
    "sel_mask, name",
    [
        (("flash", object(), 0), "flash"),
        (("flash_prefill", object(), object()), "flash_prefill"),
        (None, "no_selection"),
        (object(), "dense_mask"),
    ],
)
def test_the_guard_names_the_lane_attention_actually_got(armed_verify, sel_mask, name):
    with pytest.raises(RuntimeError, match=name):
        guard(sel_mask)


def test_the_guard_sits_before_every_other_attention_branch():
    source = inspect.getsource(Attention.__call__)
    call = source.index("self._require_sparse_decode_lane(")
    for lane_name in ("flash", "flash_prefill", "sparse_blocks", "gather_rows"):
        assert call < source.index(f'sel_mask[0] == "{lane_name}"')


def test_the_guard_samples_before_the_indexer_runs():
    """The routing happens inside ``self.indexer(...)``, so the baseline must
    be taken above it or the delta would always be zero."""

    source = inspect.getsource(Attention.__call__)
    assert source.index("sparse_before = _sparse_route_snapshot()") < source.index(
        "self.indexer("
    )


def test_the_guard_skips_vision_requests():
    source = inspect.getsource(Attention.__call__)
    assert "if vrope is None and sparse_required:" in source


# ---------------------------------------------------------------------------
# The graph-level proof
# ---------------------------------------------------------------------------
ZERO = {"route_hits": 0, "short_context": 0}


def test_assert_traced_raises_when_the_armed_lane_missed_the_graph(armed_verify):
    with pytest.raises(lane.SparseDecodeContractError, match="not in the traced"):
        lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_assert_traced_passes_once_a_route_hit_landed(armed_verify):
    lane.note_route_hit("select_eager_verify")
    lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_assert_traced_accepts_a_forward_that_routed_for_short_context(
    armed_verify,
):
    """The stock lane IS correct below 2,052 tokens; the assertion says so."""

    lane.note_short_context("select_eager_verify", 33)
    lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_assert_traced_still_raises_on_an_unbound_full_budget_forward(
    armed_verify,
):
    """A short forward earlier in the run must not vouch for this one."""

    lane.note_short_context("select_eager_verify", 33)
    before = lane.route_snapshot()
    with pytest.raises(lane.SparseDecodeContractError, match="at or above 2052"):
        lane.assert_traced(4, before=before, where="compiled verify")


def test_assert_traced_measures_the_delta_not_the_total(armed_verify):
    """A hit from a PREVIOUS trace must not vouch for this one."""

    lane.note_route_hit("select_eager_verify")
    with pytest.raises(lane.SparseDecodeContractError):
        lane.assert_traced(
            4, before={"route_hits": 1, "short_context": 0}, where="compiled verify"
        )


def test_assert_traced_is_silent_off_the_armed_width(armed_verify):
    lane.assert_traced(1, before=ZERO, where="compiled verify")
    lane.assert_traced(16, before=ZERO, where="compiled verify")


def test_assert_traced_is_silent_when_nothing_is_armed():
    lane.assert_traced(4, before=ZERO, where="compiled verify")


def test_the_snapshot_carries_both_ways_a_forward_can_prove_engagement():
    assert lane.route_snapshot() == {"route_hits": 0, "short_context": 0}
    lane.note_route_hit("x")
    lane.note_short_context("x", 9)
    assert lane.route_snapshot() == {"route_hits": 1, "short_context": 1}


def test_the_compiled_verify_body_samples_and_asserts_across_the_forward():
    source = (ROOT / "mtplx" / "graphbank.py").read_text()
    body = source[source.index("def verify_step(input_ids, *args):") :]
    body = body[: body.index("        return verify_step")]
    sample = body.index("sparse_route_before = ")
    forward = body.index("result = live._runtime_forward(")
    check = body.index("_qsa_sparse_lane.assert_traced(")
    assert sample < forward < check


# ---------------------------------------------------------------------------
# Construction owns the "cache built without the lane" gate
# ---------------------------------------------------------------------------
def make_cache(monkeypatch, *, decode=False, draft=False, armed=False):
    from mtplx import graphbank

    monkeypatch.setattr(
        graphbank, "fable_qsa_sparse_decode_enabled", lambda: bool(armed)
    )
    monkeypatch.setattr(graphbank, "fable_qsa_sparse_draft_enabled", lambda: False)
    kv = SimpleNamespace(cache=[None, None, None], step=256, keys=None, values=None)
    return graphbank.TensorOffsetQSACache(
        kv,
        None,
        None,
        compress_ratio=4,
        rows_gather_kv_m4=None,
        fable_qsa_sparse_decode=decode,
        fable_qsa_sparse_draft=draft,
    )


def test_an_armed_flag_on_a_cache_built_without_the_lane_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="constructed without the lane"):
        make_cache(monkeypatch, decode=False, armed=True)


def test_an_unarmed_process_builds_the_cache_untouched(monkeypatch):
    cache = make_cache(monkeypatch, decode=False, armed=False)
    assert cache.fable_qsa_sparse_decode_rows == 0
    assert cache.fable_qsa_sparse_draft_rows == 0


def test_a_disabled_probe_now_fails_the_build_rather_than_the_arm(monkeypatch):
    monkeypatch.setattr(lane, "install", lambda *a, **k: False)
    monkeypatch.setattr(lane, "_DISABLED_REASON", "parity probe failed on 'x'")
    with pytest.raises(RuntimeError, match="parity probe failed"):
        make_cache(monkeypatch, decode=True, armed=True)


def test_a_passing_probe_wires_the_verify_rows(monkeypatch):
    monkeypatch.setattr(lane, "install", lambda *a, **k: True)
    cache = make_cache(monkeypatch, decode=True, armed=True)
    assert cache.fable_qsa_sparse_decode_rows == lane.VERIFY_ROWS
    assert cache.fable_qsa_sparse_draft_rows == 0


def test_the_shadow_twins_carry_the_lane_without_a_defaulting_getattr():
    """A twin that silently defaulted to False would revert to stock.

    Both twin sites build a ``TensorOffsetQSACache`` from an entry that always
    sets these attributes, so reading them directly (as the neighbouring
    ``fable_qsa_m4`` lines do) is what makes a site that forgot them loud.
    """

    source = (ROOT / "mtplx" / "graphbank.py").read_text()
    assert 'getattr(\n                        entry, "fable_qsa_sparse_decode"' not in source
    assert source.count("fable_qsa_sparse_decode=entry.fable_qsa_sparse_decode") == 2
    assert source.count("fable_qsa_sparse_draft=entry.fable_qsa_sparse_draft") == 2


# ---------------------------------------------------------------------------
# The evidence a receipt and a log carry
# ---------------------------------------------------------------------------
def test_the_install_verdict_reaches_stderr_not_only_the_logger(capsys):
    lane._emit("[fable] qsa_sparse_decode: probe line")
    assert "[fable] qsa_sparse_decode: probe line" in capsys.readouterr().err


def test_install_emits_both_the_line_and_the_json(armed_verify):
    source = inspect.getsource(lane.install)
    assert source.count("_emit(engagement_line(enabled=True))") == 1
    assert source.count("_emit(engagement_line(enabled=False))") == 1
    assert source.count('"[fable] qsa_sparse_decode install: "') == 2


def test_the_engagement_line_states_rows_tile_splits_and_the_probe(armed_verify):
    lane._PROBE_REPORT["worst"] = {
        "cell": "verify-4096",
        "vs_fp32": {"max_abs_ulps": 0.75, "rel_l2": 1.2e-5, "top1": 1.0},
        "vs_shipped": {"rel_l2": 4.78e-3},
    }
    lane._COUNTS["cache_installs"] = 12
    lane._COUNTS["probe_runs"] = 2
    line = lane.engagement_line(enabled=True)
    assert line.startswith("[fable] qsa_sparse_decode armed:")
    for token in ("rows=4", "tile=128:32", "splits=17", "caches=12", "probe_runs=2"):
        assert token in line
    assert "'verify-4096'" in line


def test_the_off_line_carries_the_reason(monkeypatch):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "parity probe failed on 'x'")
    assert lane.engagement_line(enabled=False) == (
        "[fable] qsa_sparse_decode: off (parity probe failed on 'x')"
    )


def test_the_receipt_never_raises_while_the_probe_is_pending(armed_verify):
    block = lane.receipt()
    assert block["armed"] is True
    assert block["pending"] is True
    assert block["installed"] is False
    assert block["disabled_reason"] is None


def test_the_receipt_carries_everything_the_owner_asked_for(armed_verify):
    block = lane.receipt()
    for key in (
        "armed",
        "installed",
        "disabled_reason",
        "tile",
        "splits",
        "probe",
        "route_hits",
        "route_sites",
        "route_declines",
        "kernel_calls",
        "cache_installs",
    ):
        assert key in block, key
    assert block["tile"] == [128, 32]
    assert block["splits"] == 17


def test_route_hits_and_kernel_calls_are_separate_counters(armed_verify):
    lane.note_route_hit("select_eager_verify")
    lane._COUNTS["verify_kernel"] += 3
    block = lane.receipt()
    assert block["route_hits"] == 1
    assert block["kernel_calls"] == {"verify_kernel": 3, "draft_kernel": 0}


def test_reset_clears_the_route_state_too():
    lane.note_route_hit("a")
    lane.note_route_decline("b")
    lane.note_short_context("a", 9)
    lane.reset_for_tests()
    assert lane.route_counters() == {
        "route_hits": 0,
        "route_sites": {},
        "route_declines": {},
        "short_context": 0,
        "short_context_blocks": {},
        "short_context_tokens": 2052,
    }


# ---------------------------------------------------------------------------
# The driver's engagement check
# ---------------------------------------------------------------------------
DRIVER_PATH = ROOT / "scripts" / "fable" / "abba_driver.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("abba_driver_w68", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unarmed_run_needs_no_proof():
    driver = load_driver()
    block = driver.fable_qsa_sparse_decode_block(require_calls=True)
    assert block["armed"] is False
    assert "problems" not in block


def test_an_armed_run_whose_probe_never_ran_is_refused(armed_verify):
    driver = load_driver()
    with pytest.raises(RuntimeError, match="install probe never ran"):
        driver.fable_qsa_sparse_decode_block(require_calls=True)


def test_an_armed_run_that_disabled_itself_is_refused(armed_verify, monkeypatch):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "parity probe failed on 'x'")
    driver = load_driver()
    with pytest.raises(RuntimeError, match="disabled itself"):
        driver.fable_qsa_sparse_decode_block(require_calls=True)


def test_an_installed_lane_with_no_route_hit_is_refused(armed_verify, monkeypatch):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "")
    lane.note_route_decline("select_eager_verify: growable cache")
    driver = load_driver()
    with pytest.raises(RuntimeError, match="no routing decision ever reached"):
        driver.fable_qsa_sparse_decode_block(require_calls=True)


def test_a_route_hit_with_no_kernel_call_is_refused(armed_verify, monkeypatch):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "")
    lane.note_route_hit("select_eager_verify")
    driver = load_driver()
    with pytest.raises(RuntimeError, match="zero kernel calls"):
        driver.fable_qsa_sparse_decode_block(require_calls=True)


def test_a_short_context_only_cell_is_refused_with_the_reason(
    armed_verify, monkeypatch
):
    """Not an inert flag -- a cell measured below the lane's own threshold."""

    monkeypatch.setattr(lane, "_DISABLED_REASON", "")
    lane.note_short_context("select_eager_verify", 33)
    driver = load_driver()
    with pytest.raises(RuntimeError, match="never reached 2052 tokens"):
        driver.fable_qsa_sparse_decode_block(require_calls=True)


def test_a_long_cell_that_also_served_short_requests_passes(
    armed_verify, monkeypatch
):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "")
    lane.note_short_context("select_eager_verify", 33)
    lane.note_route_hit("select_eager_verify")
    lane._COUNTS["verify_kernel"] += 12
    driver = load_driver()
    block = driver.fable_qsa_sparse_decode_block(require_calls=True)
    assert block["problems"] == []
    assert block["short_context"] == 1
    assert block["route_hits"] == 1


def test_a_fully_engaged_arm_passes(armed_verify, monkeypatch):
    monkeypatch.setattr(lane, "_DISABLED_REASON", "")
    lane.note_route_hit("select_eager_verify")
    lane._COUNTS["verify_kernel"] += 12
    driver = load_driver()
    block = driver.fable_qsa_sparse_decode_block(require_calls=True)
    assert block["problems"] == []
    assert block["route_sites"] == {"select_eager_verify": 1}


def test_load_time_does_not_require_calls(armed_verify):
    """Nothing has installed at load: the QSA cache is built at first verify."""

    driver = load_driver()
    block = driver.fable_qsa_sparse_decode_block(require_calls=False)
    assert block["pending"] is True
    assert block["problems"] == []


def test_non_strict_collects_instead_of_raising(armed_verify):
    driver = load_driver()
    block = driver.fable_qsa_sparse_decode_block(require_calls=True, strict=False)
    assert len(block["problems"]) == 1
    assert "install probe never ran" in block["problems"][0]


def test_the_receipt_block_is_written_before_the_run_is_refused():
    source = DRIVER_PATH.read_text()
    assert source.index('"qsa_sparse_decode": fable_qsa_sparse_decode_block(') < source.index(
        'print(f"[fable-abba] wrote {out}"'
    ) < source.index('(payload.get("qsa_sparse_decode") or {}).get("problems")')
