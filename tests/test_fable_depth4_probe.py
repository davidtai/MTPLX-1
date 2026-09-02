"""Pure-python coverage for the depth-4 probe and its offline gate scorer.

No MLX import happens here.  :func:`mtplx.fable_depth4_probe.run_probe` takes
every device-touching piece as an injected callable precisely so the part that
owns the MTP-cache offset contract can be driven with stubs; the wiring in
``generation.py`` is checked by source inspection, which is how the rest of
this lane is guarded (``tests/test_fable_k20_log.py``).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from mtplx import fable_depth4_probe as probe_mod
from mtplx import fable_k20_log as log_mod

from scripts.fable.offline_block_verification import load_log
from scripts.fable import offline_depth4_gate as gate_mod


REPO_ROOT = Path(__file__).resolve().parents[1]
K20 = 20


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Sparse:
    """The two attributes ``support_from_distribution`` reads off a row."""

    def __init__(self, token_ids, probs) -> None:
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        self.probs = np.asarray(probs, dtype=np.float64)


class _Cache:
    """One QSA cache reduced to the offset the probe must put back."""

    def __init__(self, offset: int = 100) -> None:
        self.offset = int(offset)
        self.history: list[int] = []

    def append(self) -> None:
        self.offset += 1
        self.history.append(self.offset)


class _ExplodingRng:
    """Any draw is a bug: the probe must consume no randomness."""

    def random(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("the depth-4 probe drew a uniform")

    def choice(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("the depth-4 probe drew a uniform")


def _read_offset(cache: _Cache) -> int:
    return int(cache.offset)


def _rollback(cache: _Cache, offset: int) -> None:
    cache.offset = min(int(cache.offset), int(offset))


@pytest.fixture
def k20(monkeypatch, tmp_path):
    """Point the K20 log singleton at a test path and restore it after."""

    def configure(name="rows.npz"):
        path = tmp_path / name
        log_mod._configure_for_test(str(path))
        return log_mod.k20_log

    monkeypatch.setattr(log_mod.k20_log, "_written", None, raising=False)
    yield configure
    log_mod._configure_for_test(None)


def _row(pairs, *, width=K20):
    """One logged K20 row from ``{id: probability}`` plus zero-prob padding."""

    ids = np.zeros(width, dtype=np.uint32)
    probs = np.zeros(width, dtype=np.float64)
    used = set(pairs)
    filler = (token for token in range(90000, 99999) if token not in used)
    for slot, (token, probability) in enumerate(sorted(pairs.items())):
        ids[slot] = token
        probs[slot] = probability
    for slot in range(len(pairs), width):
        ids[slot] = next(filler)
    values = np.where(probs > 0, np.log(np.maximum(probs, 1e-30)), -1e30).astype(
        np.float32
    )
    return ids, values, probs


def _generation_source() -> str:
    return (REPO_ROOT / "mtplx" / "generation.py").read_text()


_PROBE_END = "time.perf_counter() - _d4_started,"


def _probe_block(body: str, *, code_only: bool = False) -> str:
    """The hook, from its gate through the timing call that closes it.

    ``code_only`` drops comment lines.  The block's own prose says what it must
    not do ("commits no token", "all three drafts accepted"), so a naive
    substring search over the raw text finds those words in the explanation
    rather than in the code -- which would make the guard vacuous in exactly
    the direction that matters.
    """

    start = body.index("if (\n                _FABLE_DEPTH4_PROBE")
    end = body.index(_PROBE_END, start) + len(_PROBE_END)
    block = body[start:end]
    if code_only:
        block = "\n".join(
            line for line in block.splitlines() if not line.lstrip().startswith("#")
        )
    return block


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_probe_is_off_unless_the_env_var_is_set(monkeypatch):
    monkeypatch.delenv("MTPLX_FABLE_DEPTH4_PROBE", raising=False)
    assert probe_mod._env_truthy("MTPLX_FABLE_DEPTH4_PROBE") is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MTPLX_FABLE_DEPTH4_PROBE", value)
        assert probe_mod._env_truthy("MTPLX_FABLE_DEPTH4_PROBE") is True
    for value in ("0", "", "off", "no"):
        monkeypatch.setenv("MTPLX_FABLE_DEPTH4_PROBE", value)
        assert probe_mod._env_truthy("MTPLX_FABLE_DEPTH4_PROBE") is False


def test_the_gate_is_read_once_at_import():
    """No call site may re-read the env: an armed run must stay armed."""

    source = _generation_source()
    assert source.count("_FABLE_DEPTH4_PROBE = _fable_depth4_probe_enabled()") == 1
    assert "MTPLX_FABLE_DEPTH4_PROBE" not in source.split("# MTPLX_FABLE_DEPTH4_PROBE")[
        -1
    ].split("\n_FABLE_DEPTH4_PROBE")[-1]
    probe_source = inspect.getsource(probe_mod)
    assert probe_source.count("_ENABLED = _env_truthy(_ENV_VAR)") == 1


# ---------------------------------------------------------------------------
# Row shaping
# ---------------------------------------------------------------------------


def test_support_takes_a_shaped_sparse_row_as_is():
    ids, probs = probe_mod.support_from_distribution(
        _Sparse([7, 3, 9], [0.5, 0.2, 0.3])
    )
    assert ids.tolist() == [7, 3, 9]
    assert probs.tolist() == [0.5, 0.2, 0.3]


def test_support_drops_zero_probability_padding():
    ids, probs = probe_mod.support_from_distribution(
        _Sparse([7, 3, 9], [0.5, 0.0, 0.5])
    )
    assert ids.tolist() == [7, 9]
    assert probs.tolist() == [0.5, 0.5]


def test_support_reads_a_dense_row():
    dense = np.zeros(10, dtype=np.float64)
    dense[2] = 0.75
    dense[5] = 0.25
    ids, probs = probe_mod.support_from_distribution(dense)
    assert ids.tolist() == [2, 5]
    assert probs.tolist() == [0.75, 0.25]


def test_support_trims_a_wide_row_by_probability_then_id():
    # Two ties at 0.1: the lower id wins, mirroring the deterministic K20
    # selector's (value desc, id asc) contract.
    ids, probs = probe_mod.support_from_distribution(
        _Sparse([40, 10, 30, 20], [0.5, 0.1, 0.3, 0.1]), width=3
    )
    assert ids.tolist() == [40, 30, 10]
    assert probs.tolist() == [0.5, 0.3, 0.1]
    # Not renormalised: the offline scorer's prepared_row does that once, the
    # same way it does for the three real draft rows.
    assert probs.sum() == pytest.approx(0.9)


def test_support_refuses_a_row_with_no_mass():
    with pytest.raises(ValueError, match="retained no mass"):
        probe_mod.support_from_distribution(_Sparse([1, 2], [0.0, 0.0]))


def test_gate_feature_is_the_drafted_tokens_own_probability():
    ids, _values, probs = _row({11: 0.7, 12: 0.3})
    assert probe_mod.gate_feature(ids, probs, 11) == pytest.approx(0.7)
    assert probe_mod.gate_feature(ids, probs, 12) == pytest.approx(0.3)
    # A token outside the shaped support (correction cache / reranker
    # substitution) scores 0 and so is excluded from every gate.
    assert probe_mod.gate_feature(ids, probs, 13) == 0.0


# ---------------------------------------------------------------------------
# The offset contract
# ---------------------------------------------------------------------------


def test_run_probe_restores_the_cache_offset():
    cache = _Cache(offset=100)

    def draft_step():
        cache.append()
        return "logits"

    ids, probs, trimmed = probe_mod.run_probe(
        draft_step=draft_step,
        shape_row=lambda row: _Sparse([5, 6], [0.4, 0.6]),
        mtp_cache=cache,
        read_offset=_read_offset,
        rollback=_rollback,
    )
    assert cache.offset == 100, "the probe left a speculative row staged"
    assert cache.history == [101], "the probe did not run its draft step"
    assert ids.tolist() == [5, 6]
    assert probs.tolist() == [0.4, 0.6]
    assert trimmed is False


def test_run_probe_restores_the_offset_when_the_draft_step_raises():
    cache = _Cache(offset=42)

    def draft_step():
        cache.append()
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        probe_mod.run_probe(
            draft_step=draft_step,
            shape_row=lambda row: _Sparse([1], [1.0]),
            mtp_cache=cache,
            read_offset=_read_offset,
            rollback=_rollback,
        )
    assert cache.offset == 42


def test_run_probe_restores_the_offset_when_shaping_raises():
    cache = _Cache(offset=7)

    def draft_step():
        cache.append()
        return "logits"

    def shape_row(row):
        raise ValueError("degenerate row")

    with pytest.raises(ValueError, match="degenerate"):
        probe_mod.run_probe(
            draft_step=draft_step,
            shape_row=shape_row,
            mtp_cache=cache,
            read_offset=_read_offset,
            rollback=_rollback,
        )
    assert cache.offset == 7


def test_run_probe_consumes_no_randomness():
    """Neither the signature nor the body admits a generator."""

    assert "rng" not in inspect.signature(probe_mod.run_probe).parameters
    source = inspect.getsource(probe_mod)
    assert "rng." not in source
    assert "random(" not in source

    cache = _Cache()
    rng = _ExplodingRng()
    probe_mod.run_probe(
        draft_step=lambda: rng and "logits",  # the stub is reachable, unused
        shape_row=lambda row: _Sparse([1, 2], [0.5, 0.5]),
        mtp_cache=cache,
        read_offset=_read_offset,
        rollback=_rollback,
    )


def test_run_probe_remaps_ids_into_the_target_id_space():
    cache = _Cache()
    table = np.array([100, 200, 300, 400], dtype=np.int64)
    ids, _probs, _trimmed = probe_mod.run_probe(
        draft_step=lambda: "logits",
        shape_row=lambda row: _Sparse([3, 1], [0.6, 0.4]),
        mtp_cache=cache,
        read_offset=_read_offset,
        rollback=_rollback,
        remap_ids=lambda selected: table[selected],
    )
    assert ids.tolist() == [400, 200]


def test_run_probe_flags_a_trimmed_row():
    cache = _Cache()
    _ids, _probs, trimmed = probe_mod.run_probe(
        draft_step=lambda: "logits",
        shape_row=lambda row: _Sparse([1, 2, 3], [0.5, 0.3, 0.2]),
        mtp_cache=cache,
        read_offset=_read_offset,
        rollback=_rollback,
        width=2,
    )
    assert trimmed is True


# ---------------------------------------------------------------------------
# The log's optional columns
# ---------------------------------------------------------------------------


def _open_window(log, *, primary, tokens, draft_pairs, target_pairs):
    log.stock_open(
        primary=primary,
        draft_tokens=list(tokens),
        draft_probs=[_Sparse(*_pair_arrays(pairs)) for pairs in draft_pairs],
        target_list=[_Sparse(*_pair_arrays(pairs)) for pairs in target_pairs],
        bonus_allowed=True,
        greedy=False,
        rng=None,
    )


def _pair_arrays(pairs):
    ids = np.asarray(sorted(pairs), dtype=np.int64)
    probs = np.asarray([pairs[int(token)] for token in ids], dtype=np.float64)
    return ids, probs


def test_stock_depth4_writes_the_optional_probe_columns(k20):
    log = k20()
    _open_window(
        log,
        primary=1,
        tokens=[11, 12, 13],
        draft_pairs=[{11: 0.9, 99: 0.1}, {12: 0.8, 98: 0.2}, {13: 0.85, 97: 0.15}],
        target_pairs=[
            {11: 0.7, 99: 0.3},
            {12: 0.6, 98: 0.4},
            {13: 0.9, 97: 0.1},
            {14: 0.5, 15: 0.5},
        ],
    )
    for depth in range(3):
        log.stock_depth(
            depth, accept_prob=1.0, coin=0.1, accepted=True, correction=0
        )
    log.stock_depth4(ids=np.array([14, 15]), probs=np.array([0.7, 0.3]))
    log.stock_bonus(14)
    path = log.flush()

    stored = load_log(path)
    assert stored["probe_valid"].tolist() == [1]
    assert stored["probe_trimmed"].tolist() == [0]
    ids = stored["probe_ids"][0]
    probs = stored["probe_probs"][0]
    assert ids[:2].tolist() == [14, 15]
    assert probs[:2].tolist() == [0.7, 0.3]
    assert probs[2:].sum() == 0.0
    # log(prob) in `values`, the same convention every stock row uses.
    assert stored["probe_values"][0][0] == pytest.approx(np.log(0.7), rel=1e-6)
    # The gate feature is recorded for every depth of every window.
    assert stored["gate_q"][0].tolist() == pytest.approx([0.9, 0.8, 0.85])


def test_an_unprobed_log_carries_no_probe_columns(k20):
    log = k20()
    _open_window(
        log,
        primary=1,
        tokens=[11, 12, 13],
        draft_pairs=[{11: 0.9}, {12: 0.8}, {13: 0.7}],
        target_pairs=[{11: 0.5}, {12: 0.5}, {13: 0.5}, {14: 1.0}],
    )
    path = log.flush()
    stored = load_log(path)
    for key in ("probe_valid", "probe_ids", "probe_probs", "probe_trimmed"):
        assert key not in stored, f"{key} leaked into an unprobed log"
    # gate_q always rides a stock log: it is the denominator of every gate.
    assert "gate_q" in stored


def test_stock_depth4_refuses_a_window_that_rejected(k20):
    log = k20()
    _open_window(
        log,
        primary=1,
        tokens=[11, 12, 13],
        draft_pairs=[{11: 0.9}, {12: 0.8}, {13: 0.7}],
        target_pairs=[{11: 0.5}, {12: 0.5}, {13: 0.5}, {14: 1.0}],
    )
    log.stock_depth(0, accept_prob=0.1, coin=0.9, accepted=False, correction=77)
    with pytest.raises(RuntimeError, match="rejected"):
        log.stock_depth4(ids=np.array([1]), probs=np.array([1.0]))


def test_stock_depth4_is_a_no_op_when_the_log_is_off():
    log_mod._configure_for_test(None)
    assert log_mod.k20_log.stock_depth4(ids=np.array([1]), probs=np.array([1.0])) is None
    assert log_mod.k20_log.cycles == 0


# ---------------------------------------------------------------------------
# The offline scorer
# ---------------------------------------------------------------------------


def test_overlap_is_the_sum_of_the_pointwise_minimum():
    p_ids = np.array([1, 2, 3], dtype=np.uint32)
    p_probs = np.array([0.5, 0.3, 0.2])
    q_ids = np.array([2, 3, 4], dtype=np.uint32)
    q_probs = np.array([0.1, 0.6, 0.3])
    # min(0, 0.5) + min(0.1, 0.3) + min(0.6, 0.2) + min(0.3, 0) = 0.3
    assert gate_mod.overlap(p_ids, p_probs, q_ids, q_probs) == pytest.approx(0.3)


def test_overlap_of_a_row_with_itself_is_one():
    ids = np.array([4, 9], dtype=np.uint32)
    probs = np.array([0.25, 0.75])
    assert gate_mod.overlap(ids, probs, ids, probs) == pytest.approx(1.0)


def test_overlap_of_disjoint_rows_is_zero():
    assert (
        gate_mod.overlap(
            np.array([1], dtype=np.uint32),
            np.array([1.0]),
            np.array([2], dtype=np.uint32),
            np.array([1.0]),
        )
        == 0.0
    )


def _probe_log(k20, windows):
    """Write a log of ``(gate_q3, all_accepted, alpha4_overlap)`` windows."""

    log = k20()
    for gate_q3, all_accepted, bonus_overlap in windows:
        _open_window(
            log,
            primary=1,
            tokens=[11, 12, 13],
            draft_pairs=[
                {11: 0.9, 91: 0.1},
                {12: 0.9, 92: 0.1},
                {13: gate_q3, 93: round(1.0 - gate_q3, 6)},
            ],
            target_pairs=[
                {11: 0.9, 91: 0.1},
                {12: 0.9, 92: 0.1},
                {13: 0.9, 93: 0.1},
                # p_3 = bonus row; q_4 below overlaps it by `bonus_overlap`.
                {14: 1.0},
            ],
        )
        depths = 3 if all_accepted else 1
        for depth in range(depths):
            log.stock_depth(
                depth,
                accept_prob=1.0 if all_accepted else 0.0,
                coin=0.1 if all_accepted else 0.9,
                accepted=bool(all_accepted),
                correction=0 if all_accepted else 77,
            )
        if all_accepted:
            log.stock_depth4(
                ids=np.array([14, 15]),
                probs=np.array([bonus_overlap, round(1.0 - bonus_overlap, 6)]),
            )
            log.stock_bonus(14)
    return log.flush()


def test_score_measures_alpha4_against_the_bonus_row(k20):
    path = _probe_log(
        k20,
        [
            (0.95, True, 0.9),
            (0.85, True, 0.8),
            (0.5, True, 0.4),
            (0.95, False, 0.0),
        ],
    )
    result = gate_mod.score(load_log(path), thresholds=(0.6, 0.9))
    assert result["cycles"] == 4
    assert result["cycles_scoreable"] == 4
    assert result["probe_cycles"] == 3
    assert result["probe_cycles_scored"] == 3
    # p_3 is the point mass at 14, so alpha_4 = min(1, q_4(14)) = q_4(14).
    assert result["alpha4_ungated"] == pytest.approx((0.9 + 0.8 + 0.4) / 3)

    by_threshold = {gate["threshold"]: gate for gate in result["gates"]}
    # q(x_3) > 0.9 keeps the two 0.95 windows, one of which was probed.
    high = by_threshold[0.9]
    assert high["windows"] == 2
    assert high["p_gate"] == pytest.approx(0.5)
    assert high["p_all_accepted"] == pytest.approx(0.5)
    assert high["probed"] == 1
    assert high["alpha4"] == pytest.approx(0.9)
    # q(x_3) > 0.6 keeps 0.95, 0.85, 0.95.
    low = by_threshold[0.6]
    assert low["windows"] == 3
    assert low["probed"] == 2
    assert low["alpha4"] == pytest.approx(0.85)


def test_projection_charges_the_extra_row_only_on_gated_windows(k20):
    path = _probe_log(k20, [(0.95, True, 0.9), (0.1, True, 0.9)])
    result = gate_mod.score(load_log(path), thresholds=(0.8,))
    rows = gate_mod.project(
        result, ms_per_window=40.0, draft_step_ms=1.0, row_ms=2.0
    )
    (row,) = rows
    assert row["p_gate"] == pytest.approx(0.5)
    assert row["delta_cost_ms"] == pytest.approx(0.5 * 3.0)
    assert row["delta_tokens"] == pytest.approx(0.5 * 1.0 * 0.9)
    base = result["base_tokens_per_window"]
    assert row["tok_s"] == pytest.approx(
        1000.0 * (base + row["delta_tokens"]) / (40.0 + row["delta_cost_ms"])
    )


def test_report_says_go_above_the_bar_and_no_go_below(k20):
    high = gate_mod.score(
        load_log(_probe_log(k20, [(0.95, True, 0.9), (0.95, True, 0.86)])),
        thresholds=(0.8,),
    )
    text = gate_mod.report(
        high,
        ms_per_window=38.7,
        draft_step_ms=1.2,
        row_costs=(1.8, 1.4),
        go_gate=0.8,
        go_alpha=0.75,
    )
    assert text.startswith("layout ")
    assert "\nGO: alpha_4 | q(x_3) > 0.8" in text
    # Both row costs are reported, not just one.
    assert "M=5 row 1.80 ms" in text and "M=5 row 1.40 ms" in text

    low = gate_mod.score(
        load_log(_probe_log(k20, [(0.95, True, 0.5), (0.95, True, 0.5)], )),
        thresholds=(0.8,),
    )
    assert "\nNO-GO: alpha_4" in gate_mod.report(
        low,
        ms_per_window=38.7,
        draft_step_ms=1.2,
        row_costs=(1.8,),
        go_gate=0.8,
        go_alpha=0.75,
    )


def test_scorer_refuses_a_log_that_carries_no_probe(k20, tmp_path):
    log = k20("plain.npz")
    _open_window(
        log,
        primary=1,
        tokens=[11, 12, 13],
        draft_pairs=[{11: 0.9}, {12: 0.8}, {13: 0.7}],
        target_pairs=[{11: 0.5}, {12: 0.5}, {13: 0.5}, {14: 1.0}],
    )
    path = log.flush()
    with pytest.raises(SystemExit, match="MTPLX_FABLE_DEPTH4_PROBE"):
        gate_mod.score(load_log(path))


def test_main_exits_non_zero_when_nothing_was_scored(k20, capsys):
    path = _probe_log(k20, [(0.95, False, 0.0)])
    assert gate_mod.main([path]) == 1
    assert "FAIL: depth-4 gate: this log has no" in capsys.readouterr().err


def test_main_exits_zero_and_prints_the_verdict(k20, capsys):
    path = _probe_log(k20, [(0.95, True, 0.9), (0.5, True, 0.9)])
    assert gate_mod.main([path, "--ms-per-window", "38.7"]) == 0
    out = capsys.readouterr().out
    assert "depth-4 probe: 2 windows recorded, 2 scored" in out
    assert "GO:" in out or "NO-GO:" in out


# ---------------------------------------------------------------------------
# The generation.py wiring
# ---------------------------------------------------------------------------


def _generate_mtpk_body() -> str:
    source = _generation_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_mtpk"
    )
    return ast.get_source_segment(source, function)


def test_the_hook_is_gated_on_both_the_probe_and_the_log():
    block = _probe_block(_generate_mtpk_body())
    assert "_FABLE_DEPTH4_PROBE" in block
    assert "_FABLE_K20_LOG" in block
    # Only the stock host accept lane, only a sampled window, only a
    # full-depth one -- the three preconditions the bonus row needs.
    assert "_host_accept_drafts" in block
    assert "sampler.temperature > 0" in block
    assert "target_prefix_tokens is None" in block
    assert "len(draft_tokens) == cycle_depth" in block
    # Only the persistent MTP cache carries the window's real draft history,
    # and only a cycle the PR391 device handoff does not own.
    assert "_d4_probe_state[2] is mtp_cache" in block
    assert "not _pr391_mtp_handoff_owns_cycle" in block


def test_the_hook_runs_before_the_mtp_history_commit():
    body = _generate_mtpk_body()
    hook = body.index("if (\n                _FABLE_DEPTH4_PROBE")
    branch = body.index("if accepted_count == len(draft_tokens):")
    commit = body.index("_rollback_mtp_cache(mtp_cache, cycle_mtp_offset + 1)", branch)
    assert branch < hook < commit, (
        "the probe must run after the verify decision and before the commit "
        "rolls the MTP cache back"
    )


def test_the_hook_drafts_at_depth_four_and_restores_the_offset():
    block = _probe_block(_generate_mtpk_body())
    assert "mtp_depth=len(draft_tokens) + 1" in block
    assert "read_offset=_mtp_cache_offset" in block
    assert "rollback=_rollback_mtp_cache" in block
    # The position offset is recomputed, not reused from depth 3: a real 4th
    # step would read it after depth 3's append.
    assert "position_offset=mtp_position_offset_for_cache(_d4_cache)" in block


def test_the_hook_consumes_no_randomness_and_commits_no_token():
    block = _probe_block(_generate_mtpk_body(), code_only=True)
    assert "rng" not in block, "the probe touched the generator"
    for forbidden in ("tokens.append", "pending_primary", "committed", "accepted"):
        assert forbidden not in block, f"the probe touched {forbidden}"
    # Its only output is the log call.
    body = _generate_mtpk_body()
    assert body.count("k20_log.stock_depth4(") == 1


def test_the_accept_coin_draw_count_is_untouched():
    """The probe must not shift the RNG stream by a single draw."""

    body = _generate_mtpk_body()
    assert body.count("_k20_coin = float(rng.random())") == 2
    assert body.count("accepted_now = _k20_coin <= accept_prob") == 2
    assert "accepted_now = float(rng.random()) <= accept_prob" not in body


def test_the_probe_locals_are_never_read_outside_the_hook():
    body = _generate_mtpk_body()
    block = _probe_block(body)
    for name in ("_d4_ids", "_d4_probs", "_d4_trimmed", "_d4_hidden", "_d4_started"):
        assert body.count(name) == block.count(name), (
            f"{name} escapes the probe block; the probe must be a pure read"
        )


def test_the_probe_state_is_reset_every_cycle_and_captured_per_depth():
    body = _generate_mtpk_body()
    assert body.count("_d4_probe_state: tuple[Any, int, Any, bool] | None = None") == 1
    reset = body.index("_d4_probe_state: tuple[Any, int, Any, bool] | None = None")
    loop = body.index("for depth_index in range(", reset)
    capture = body.index("_d4_probe_state = (", loop)
    hook = body.index("if (\n                _FABLE_DEPTH4_PROBE")
    assert reset < loop < capture < hook


def test_the_probe_self_times_outside_the_production_timers():
    block = _probe_block(_generate_mtpk_body(), code_only=True)
    for timer in ("draft_time", "verify_time", "target_time", "commit_time"):
        assert timer not in block, f"the probe contaminated {timer}"
    assert "_add_timing(" in block
    assert '"fable_depth4_probe",' in block
