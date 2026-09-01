"""Pure-python coverage for the opt-in K20 row log and the two offline scorers.

No MLX import happens here.  The log's ``record`` only calls ``np.asarray`` on
whatever it is handed, so plain NumPy arrays stand in for the device rows; the
generation-side wiring is checked by source inspection, which is how the rest
of the PR391 lane is guarded (see ``tests/test_pr391_float32_d3_core.py``).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from mtplx import fable_k20_log as log_mod

from scripts.fable.offline_block_verification import (
    DEPTH,
    ONE,
    Window,
    _block_realised_reach,
    alpha_by_depth,
    decide_block,
    decide_current,
    prepare_batched_row,
    prepare_row,
    reach_ladder_block,
    reach_ladder_current,
    report,
    score,
    water_fill_lambda,
)
from scripts.fable import offline_draft_temperature as temp_mod


REPO_ROOT = Path(__file__).resolve().parents[1]
K20 = 20


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeTape:
    """The slice of ``PCG64UniformTape`` the log reads: an immutable tape."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        return self._values


class _FakeReservation:
    def __init__(self, offset: int) -> None:
        self.offset = int(offset)


@pytest.fixture
def k20(monkeypatch, tmp_path):
    """Point the module singleton at a test path and restore it after."""

    def configure(name="rows.npz"):
        path = tmp_path / name
        log_mod._configure_for_test(str(path))
        return log_mod.k20_log, path

    yield configure
    log_mod._configure_for_test(None)


def _row(seed: int, rows: int, columns: int = K20):
    rng = np.random.default_rng(seed)
    ids = np.stack(
        [rng.choice(4096, size=columns, replace=False) for _ in range(rows)]
    ).astype(np.uint32)
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    logits = values.astype(np.float64)
    probs = np.exp(logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 3.0))
    return ids, values, probs.astype(np.float32)


def _record_one(logger, *, offset=3, tape_size=32, accepted=2, first_reject=-1):
    draft_ids, draft_values, draft_probs = _row(1, DEPTH)
    target_ids, target_values, target_probs = _row(2, DEPTH + 1)
    tape = _FakeTape(np.linspace(0.0, 0.99, tape_size))
    logger.record(
        draft_result=(
            np.array([[11, 22, 33]], dtype=np.uint32),
            draft_ids,
            draft_values,
            draft_probs,
        ),
        target_support=(target_ids, target_values, target_probs),
        uniform_tape=tape,
        reservation=_FakeReservation(offset),
        primary=7,
        bonus_allowed=1,
        decision=(accepted, first_reject, 99, 1, True, 4, [0.5, 0.25, 0.0]),
    )
    return tape


# ---------------------------------------------------------------------------
# The npz writer
# ---------------------------------------------------------------------------


def test_disabled_log_records_nothing_and_writes_nothing(tmp_path):
    log_mod._configure_for_test(None)
    logger = log_mod.k20_log
    assert logger.enabled is False
    _record_one(logger)
    assert logger.cycles == 0
    assert logger.flush() is None
    assert not list(tmp_path.iterdir())


def test_record_captures_every_row_and_the_two_uniform_windows(k20):
    logger, path = k20()
    logger.set_stop_ids(np.array([151643, 151645], dtype=np.uint32))
    tape = _record_one(logger, offset=5)
    assert logger.cycles == 1

    out = logger.flush()
    assert out == str(path)
    with np.load(out) as handle:
        data = {key: handle[key] for key in handle.files}

    assert data["draft_ids"].shape == (1, DEPTH, K20)
    assert data["draft_values"].shape == (1, DEPTH, K20)
    assert data["draft_probs"].shape == (1, DEPTH, K20)
    assert data["target_ids"].shape == (1, DEPTH + 1, K20)
    assert data["target_values"].shape == (1, DEPTH + 1, K20)
    assert data["target_probs"].shape == (1, DEPTH + 1, K20)
    assert data["draft_ids"].dtype == np.uint32
    assert data["draft_values"].dtype == np.float32

    np.testing.assert_array_equal(data["draft_tokens"][0], [11, 22, 33])
    assert int(data["primary"][0]) == 7
    # The decision consumes tape[offset : offset + 4]; the three draft selects
    # consumed tape[offset - 3 : offset] immediately before it.
    np.testing.assert_array_equal(data["decision_uniforms"][0], tape.values[5:9])
    np.testing.assert_array_equal(data["draft_uniforms"][0], tape.values[2:5])
    assert int(data["descriptor_offset"][0]) == 5
    np.testing.assert_array_equal(data["stop_ids"], [151643, 151645])


def test_accept_probability_valid_marks_the_kernel_early_return(k20):
    logger, _ = k20()
    _record_one(logger, accepted=1, first_reject=1)
    _record_one(logger, accepted=3, first_reject=-1)
    out = logger.flush()
    with np.load(out) as handle:
        valid = handle["accept_probability_valid"]
        first_reject = handle["first_reject"]
    # The kernel returns as soon as a depth rejects, so only first_reject + 1
    # alpha entries are real; a full accept fills all DEPTH.
    assert list(valid) == [2, DEPTH]
    assert list(first_reject) == [1, -1]


def test_record_refuses_a_reservation_that_precedes_the_draft_selects(k20):
    logger, _ = k20()
    with pytest.raises(RuntimeError, match="precedes the tape"):
        _record_one(logger, offset=1)


def test_record_refuses_a_short_decision_reservation(k20):
    logger, _ = k20()
    with pytest.raises(RuntimeError, match="reservation is short"):
        _record_one(logger, offset=5, tape_size=8)


def test_flush_is_idempotent_and_accepts_a_json_path(k20):
    logger, path = k20(name="rows.json")
    _record_one(logger)
    first = logger.flush()
    second = logger.flush()
    assert first == second == str(path.with_suffix(".npz"))
    assert path.exists()
    with np.load(first) as handle:
        assert handle["draft_ids"].shape[0] == 1


# ---------------------------------------------------------------------------
# generation.py wiring -- source level, so no MLX is imported
# ---------------------------------------------------------------------------


def _generation_source() -> str:
    return (REPO_ROOT / "mtplx" / "generation.py").read_text()


def test_decode_keeps_one_sync_and_widens_it_only_when_armed():
    source = _generation_source()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pr391_decode_float32_verifier_decision"
    )
    body = ast.get_source_segment(source, function)
    # Off: literally the original single sync over the decision outputs.
    assert "mx.eval(*result)" in body
    # On: the same ONE sync, widened to retain the rows the kernel already
    # consumed -- never a second mx.eval.
    assert "mx.eval(*result, *_k20_draft_result, *_k20_target_support)" in body
    assert "k20_log.record(" in body

    syncs = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mx"
    ]
    # Exactly two spellings of the SAME sync, on the two arms of one `if`,
    # and nothing else on the device.
    assert [node.func.attr for node in syncs] == ["eval", "eval"]
    branch = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and not isinstance(node, ast.Expr)
    )
    assert len(branch.body) == 1 and len(branch.orelse) == 2


def test_both_decision_call_sites_pass_the_same_gated_capture():
    source = _generation_source()
    assert source.count("k20_capture=_pr391_k20_capture,") == 2
    gate = source.index("_pr391_k20_capture = (")
    # The capture is built by a conditional expression, so when the gate is
    # off nothing is allocated and no attribute on the rows is touched.
    assert "None\n                if not _FABLE_K20_LOG" in source[gate : gate + 400]
    first = source.index("k20_capture=_pr391_k20_capture,")
    assert gate < first


def test_the_gate_is_read_once_at_import():
    source = _generation_source()
    assert "_FABLE_K20_LOG = _fable_k20_log_enabled()" in source
    # generation.py never reads the variable itself; it mentions it only in
    # comments, and the one read lives at module scope in fable_k20_log.
    assert 'os.environ.get("MTPLX_FABLE_K20_LOG"' not in source
    assert "_env_truthy(\"MTPLX_FABLE_K20_LOG\")" not in source
    assert "os.environ.get(_ENV_VAR)" not in inspect.getsource(log_mod.K20RowLog)
    module = (REPO_ROOT / "mtplx" / "fable_k20_log.py").read_text()
    assert module.count("os.environ.get(_ENV_VAR)") == 1
    assert module.count('_ENV_VAR = "MTPLX_FABLE_K20_LOG"') == 1


def test_generation_flushes_explicitly_and_the_module_registers_atexit():
    source = _generation_source()
    assert "k20_log.flush()" in source
    assert "atexit.register(k20_log.flush)" in (
        REPO_ROOT / "mtplx" / "fable_k20_log.py"
    ).read_text()


# ---------------------------------------------------------------------------
# Offline scorers -- synthetic rows
# ---------------------------------------------------------------------------


def _make_row(pairs, *, width=K20):
    """Build one logged K20 row from ``{id: probability}``.

    Padding ids carry probability 0 so the kernel's ``probs > 0`` filter drops
    them, which is exactly what a short real row looks like.
    """

    ids = np.zeros(width, dtype=np.uint32)
    probs = np.zeros(width, dtype=np.float32)
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


def _window(draft_pairs, target_pairs, tokens, uniforms, *, bonus=True, stops=()):
    return Window(
        draft_tokens=tokens,
        draft_rows=[prepare_row(*_make_row(pairs)) for pairs in draft_pairs],
        target_rows=[prepare_batched_row(*_make_row(pairs)) for pairs in target_pairs],
        uniforms=np.asarray(uniforms, dtype=np.float64),
        bonus_allowed=bonus,
        stops=frozenset(int(token) for token in stops),
    )


_FLAT = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}


def test_water_fill_hits_the_budget_exactly():
    q = np.array([0.5, 0.3, 0.2])
    base = np.array([0.1, 0.4, 0.9])
    level = water_fill_lambda(q, base, np.float64(1.0), np.float64(0.6))
    assert float(np.sum(q * np.minimum(1.0, base + level))) == pytest.approx(0.6)
    # A budget already met at lambda = 0 must not push the level up.
    assert float(water_fill_lambda(q, base, np.float64(1.0), np.float64(0.3))) == 0.0
    # A saturating budget: everything clips at the cap.
    full = water_fill_lambda(q, base, np.float64(1.0), np.float64(1.0))
    assert float(np.sum(q * np.minimum(1.0, base + full))) == pytest.approx(1.0)


def test_water_fill_respects_a_cap_below_one():
    q = np.array([0.5, 0.5])
    base = np.array([0.0, 0.2])
    cap = np.float64(0.4)
    level = water_fill_lambda(q, base, cap, np.float64(0.35))
    assert float(np.sum(q * np.minimum(cap, base + level))) == pytest.approx(0.35)
    assert float(np.minimum(cap, base[0] + level)) <= float(cap)


@pytest.mark.parametrize("cap_mode", ["reach", "one"])
def test_block_is_identical_to_current_when_the_ladder_stays_at_one(cap_mode):
    """H §3.2: with c = 1 throughout, block verification IS the shipped law."""

    # rho >= 1 at every depth: the target likes each drafted token at least as
    # much as the draft did, so A_d = 1 and the ladder never leaves 1.
    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.9, 2: 0.1}, {2: 0.8, 3: 0.2}, {3: 0.7, 4: 0.3}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.9, 0.9, 0.9, 0.42])
    current = decide_current(window)
    block = decide_block(window, cap_mode=cap_mode)
    assert current.accepted == DEPTH
    assert block.ladder_all_one is True
    assert block.key() == current.key()
    assert block.tokens == current.tokens == DEPTH + 1


@pytest.mark.parametrize("cap_mode", ["reach", "one"])
def test_block_matches_on_a_final_depth_rejection_after_a_unit_ladder(cap_mode):
    """The last depth has no look-ahead row, so c = 1 there is still identity."""

    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.9, 2: 0.1}, {2: 0.8, 3: 0.2}, {3: 0.1, 4: 0.9}, {1: 1.0}]
    # rho_3 = 0.1 / 0.25 = 0.4; a draw of 0.8 rejects under both laws.
    window = _window(draft, target, [1, 2, 3], [0.5, 0.5, 0.8, 0.3])
    current = decide_current(window)
    block = decide_block(window, cap_mode=cap_mode)
    assert current.first_reject == 2
    assert block.ladder_all_one is True
    assert block.key() == current.key()


def test_block_diverges_when_the_ladder_drops_below_one():
    """A sub-1 rho at depth 1 opens the water-fill; the laws may differ."""

    draft = [_FLAT, _FLAT, _FLAT]
    # rho_1 = 0.1/0.25 = 0.4 -- under water. The depth-2 row is one the target
    # likes far more than the draft did, so block verification reaches for it.
    target = [{1: 0.1, 2: 0.9}, {2: 0.95, 3: 0.05}, {3: 0.5, 4: 0.5}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.55, 0.1, 0.1, 0.3])
    current = decide_current(window)
    block = decide_block(window, cap_mode="reach")
    assert block.ladder_all_one is False
    assert current.accepted == 0  # 0.55 > alpha_1 = 0.4
    assert block.accepted >= 1  # the boosted coin reaches depth 2
    assert block.tokens > current.tokens


def test_block_preserves_the_depth_one_budget_in_expectation():
    """Exactness check: E_{x2 ~ q2}[w_1] must equal A_1 = min(1, rho_1)."""

    draft2 = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
    draft = [_FLAT, draft2, _FLAT]
    target = [{1: 0.1, 2: 0.9}, {1: 0.05, 2: 0.5, 3: 0.4, 4: 0.05}, {3: 0.5, 4: 0.5}, {1: 1.0}]
    budget = 0.1 / 0.25
    realised = []
    for second in (1, 2, 3, 4):
        window = _window(draft, target, [1, second, 3], [0.0, 0.0, 0.0, 0.3])
        block = decide_block(window, cap_mode="reach")
        realised.append(block.accept_probability[0])
    # float32 row probabilities set the floor on how exactly the budget
    # can be reproduced; the law itself is float64 throughout.
    assert float(np.mean(realised)) == pytest.approx(budget, abs=1e-6)
    # And it is a genuine redistribution, not a constant.
    assert max(realised) > min(realised)


def test_current_law_owns_the_accept_tie_and_the_cdf_tie():
    """``u <= alpha`` accepts (kernel line 289); sampling is side='right'."""

    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.125, 2: 0.875}, {2: 1.0}, {3: 1.0}, {1: 0.5, 2: 0.5}]
    alpha1 = 0.125 / 0.25  # exactly 0.5
    accepted = decide_current(_window(draft, target, [1, 2, 3], [alpha1, 0.0, 0.0, 0.0]))
    assert accepted.accepted == DEPTH, "u == alpha must ACCEPT"
    rejected = decide_current(
        _window(draft, target, [1, 2, 3], [np.nextafter(alpha1, 1.0), 0.0, 0.0, 0.0])
    )
    assert rejected.accepted == 0

    # Bonus row {1: 0.5, 2: 0.5}: cdf = [0.5, 1.0]. side="right" sends a draw
    # of exactly 0.5 into the SECOND bucket.
    at_edge = decide_current(_window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, 0.5]))
    assert at_edge.selected_token == 2
    below = decide_current(
        _window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, np.nextafter(0.5, 0.0)])
    )
    assert below.selected_token == 1


def test_accepted_stop_token_ends_the_window_without_a_bonus():
    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.9, 2: 0.1}, {2: 0.8}, {3: 1.0}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, 0.0], stops=(2,))
    for outcome in (decide_current(window), decide_block(window, cap_mode="reach")):
        assert outcome.accepted == 2
        assert outcome.selected_present is False
        assert outcome.draws_used == 2
        assert outcome.tokens == 2


def test_zero_target_mass_is_an_unconditional_rejection():
    """H §1.2: 8-16% of drafts land outside the target's prepared row."""

    draft = [_FLAT, _FLAT, _FLAT]
    target = [{2: 1.0}, {2: 1.0}, {3: 1.0}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    current = decide_current(window)
    # alpha_1 = 0, and a draw of exactly 0.0 still accepts under `u <= alpha`;
    # any positive draw rejects. Use a positive draw for the real statement.
    strict = _window(draft, target, [1, 2, 3], [1e-12, 0.0, 0.0, 0.0])
    assert decide_current(strict).accepted == 0
    assert decide_block(strict, cap_mode="reach").accepted == 0
    assert current.accept_probability[0] == 0.0


# ---------------------------------------------------------------------------
# Offline scorers -- end to end over a written npz
# ---------------------------------------------------------------------------


def _synthetic_log(path, cycles=6):
    rng = np.random.default_rng(391)
    log_mod._configure_for_test(str(path))
    logger = log_mod.k20_log
    logger.set_stop_ids(np.zeros(0, dtype=np.uint32))
    tape = _FakeTape(rng.random(7 * (cycles + 2)))
    for cycle in range(cycles):
        offset = 7 * cycle + 3
        draft_ids, draft_values, draft_probs = _row(10 + cycle, DEPTH)
        target_ids, target_values, target_probs = _row(50 + cycle, DEPTH + 1)
        # Make the drafted tokens real members of each draft row so the replay
        # sees a q > 0, as production always does.
        tokens = draft_ids[:, 0].astype(np.uint32)
        logger.record(
            draft_result=(
                tokens.reshape(1, DEPTH),
                draft_ids,
                draft_values,
                draft_probs,
            ),
            target_support=(target_ids, target_values, target_probs),
            uniform_tape=tape,
            reservation=_FakeReservation(offset),
            primary=cycle,
            bonus_allowed=1,
            decision=(0, 0, 0, 0, False, 1, [0.0, 0.0, 0.0]),
        )
    out = logger.flush()
    log_mod._configure_for_test(None)
    return out


def test_score_runs_both_laws_over_a_written_log(tmp_path):
    path = _synthetic_log(tmp_path / "rows.npz")
    with np.load(path) as handle:
        data = {key: handle[key] for key in handle.files}
    result = score(data, cap_mode="reach")
    assert result["cycles"] == 6
    for name in ("current", "block"):
        row = result[name]
        assert 1.0 <= row["tokens_per_window"] <= DEPTH + 1
        assert 1.0 <= row["tokens_per_window_e"] <= DEPTH + 1
        # The reach ladder is non-increasing: you cannot accept through depth
        # d without having accepted through d - 1.
        assert row["reach_by_depth"] == sorted(row["reach_by_depth"], reverse=True)
    assert np.isfinite(result["paired_delta_tokens_per_window"])
    assert np.isfinite(result["paired_delta_sem"])
    # The synthetic decisions are placeholders, so the replay self-check is
    # expected to flag them -- that it flags them at all is the point.
    assert result["replay_mismatch_cycles"]
    if result["ladder_all_one_cycles"]:
        assert result["ladder_all_one_agree_fraction"] == 1.0
    assert "E[tok/win]" in report(result, ms_per_window=37.47)


def test_both_caps_hold_the_c_equals_one_identity_over_a_log(tmp_path):
    path = _synthetic_log(tmp_path / "rows.npz", cycles=12)
    with np.load(path) as handle:
        data = {key: handle[key] for key in handle.files}
    for cap in ("reach", "one"):
        result = score(data, cap_mode=cap)
        assert 0.0 <= result["agree_fraction"] <= 1.0
        if result["ladder_all_one_cycles"]:
            assert result["ladder_all_one_agree_fraction"] == 1.0


# ---------------------------------------------------------------------------
# The analytic (coin-free) reach ladder
# ---------------------------------------------------------------------------


def test_alpha_is_reported_past_the_first_rejection(tmp_path):
    """The kernel censors alpha at the first rejection; the rows do not."""

    draft = [_FLAT, _FLAT, _FLAT]
    # rho_1 = 0 (target zeroes the drafted token), so the shipped kernel would
    # stop at depth 1 and never compute alpha_2 or alpha_3. Both are here.
    target = [{2: 1.0}, {2: 0.5, 3: 0.5}, {3: 1.0}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.5, 0.5, 0.5, 0.5])
    alpha = alpha_by_depth(window)
    # rho = 0 / 0.25, 0.5 / 0.25, 1.0 / 0.25 -> alpha = 0, 1, 1 after the clip.
    np.testing.assert_allclose(alpha, [0.0, 1.0, 1.0], rtol=1e-6)
    # The coin-driven decision stops at depth 1, which is the censoring the
    # receipts suffer from.
    outcome = decide_current(window)
    assert outcome.first_reject == 0
    assert outcome.accept_probability[1] == 0.0  # never evaluated

    path = _synthetic_log(tmp_path / "rows.npz", cycles=5)
    with np.load(path) as handle:
        data = {key: handle[key] for key in handle.files}
    table = score(data, cap_mode="reach")["alpha_uncensored"]
    for key in ("mean", "p_zero", "p_one"):
        assert len(table[key]) == DEPTH
        assert all(0.0 <= value <= 1.0 for value in table[key])


def test_current_ladder_is_the_running_product_of_alpha():
    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.125, 2: 0.875}, {2: 0.125, 3: 0.875}, {3: 1.0}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    ladder = reach_ladder_current(window)
    # alpha = 0.125 / 0.25 = 0.5 at depths 1 and 2, then 1.0.
    np.testing.assert_allclose(ladder, [0.5, 0.25, 0.25], rtol=1e-6)
    # E[l] = sum_d w_d, and it never consults a uniform.
    assert float(ladder.sum()) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("cap_mode", ["reach", "one"])
def test_block_ladder_equals_the_current_ladder_at_c_equals_one(cap_mode):
    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.9, 2: 0.1}, {2: 0.8, 3: 0.2}, {3: 0.1, 4: 0.9}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        reach_ladder_block(window, cap_mode=cap_mode), reach_ladder_current(window)
    )


def test_saturated_water_fill_returns_the_cap_exactly():
    """The c = 1 identity is a float64 claim, not an approximate one.

    Solving for ``lam = cap - min(base)`` and adding it back returns
    0.9999999999999999 often enough to break the identity, so the saturated
    case short-circuits to the cap.
    """

    draft = [_FLAT, _FLAT, _FLAT]
    target = [{1: 0.9, 2: 0.1}, {2: 0.3, 3: 0.7}, {3: 0.11, 4: 0.89}, {1: 1.0}]
    window = _window(draft, target, [1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    realised = _block_realised_reach(
        window, 0, budget=ONE, reach=ONE, cap_mode="reach"
    )
    assert realised == 1.0
    assert float(realised) == 1.0  # exactly, not 1 - 1 ulp


# ---------------------------------------------------------------------------
# Draft temperature
# ---------------------------------------------------------------------------


def test_temper_row_is_the_identity_at_temperature_one():
    probs = np.array([0.4, 0.3, 0.2, 0.05], dtype=np.float64)
    out = temp_mod.temper_row(probs, temperature=1.0, tail="lump")
    np.testing.assert_allclose(out, probs, rtol=0, atol=1e-15)


def test_temper_row_sharpens_below_one_and_flattens_above():
    probs = np.array([0.5, 0.3, 0.15], dtype=np.float64)
    cold = temp_mod.temper_row(probs, temperature=0.5, tail="drop")
    hot = temp_mod.temper_row(probs, temperature=2.0, tail="drop")
    warm = temp_mod.temper_row(probs, temperature=1.0, tail="drop")
    assert cold[0] > warm[0] > hot[0]
    assert cold[-1] < warm[-1] < hot[-1]
    for row in (cold, hot, warm):
        assert float(row.sum()) == pytest.approx(1.0)


def test_drop_tail_is_never_below_lump_tail_when_sharpening():
    """``sum x^a <= (sum x)^a`` for ``a >= 1``: lumping over-states the tail."""

    probs = np.array([0.4, 0.3, 0.2], dtype=np.float64)  # head 0.9, tail 0.1
    drop = temp_mod.temper_row(probs, temperature=0.7, tail="drop")
    lump = temp_mod.temper_row(probs, temperature=0.7, tail="lump")
    assert np.all(drop >= lump - 1e-15)


def test_overlap_is_the_expected_acceptance():
    target_ids, target_probs = prepare_batched_row(*_make_row({1: 0.6, 2: 0.4}))
    draft_ids, draft_probs = prepare_row(*_make_row({1: 0.5, 2: 0.5}))
    beta = temp_mod.overlap(target_ids, target_probs, draft_ids, draft_probs)
    assert beta == pytest.approx(min(0.6, 0.5) + min(0.4, 0.5))
    # Identical rows overlap fully.
    same = temp_mod.overlap(target_ids, target_probs, target_ids, target_probs)
    assert same == pytest.approx(1.0)
    # Disjoint supports do not overlap at all.
    other_ids, other_probs = prepare_row(*_make_row({7: 1.0}))
    assert temp_mod.overlap(target_ids, target_probs, other_ids, other_probs) == 0.0


def test_restrict_top_k_uses_the_kernel_tie_break():
    ids = np.array([9, 3, 5, 1], dtype=np.uint32)
    values = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float32)
    probs = np.array([0.2, 0.2, 0.1, 0.5], dtype=np.float32)
    kept_ids, kept_values, kept_probs = temp_mod.restrict_top_k(ids, values, probs, 3)
    # Score descending, then id ASCENDING on the 1.0 tie -> 1, then 3, then 9.
    np.testing.assert_array_equal(kept_ids, [1, 3, 9])
    np.testing.assert_array_equal(kept_values, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(kept_probs, [0.5, 0.2, 0.2], rtol=1e-6)


def test_sweep_reports_one_row_per_temperature(tmp_path):
    path = _synthetic_log(tmp_path / "rows.npz", cycles=4)
    with np.load(path) as handle:
        data = {key: handle[key] for key in handle.files}
    rows = temp_mod.sweep(
        data, temperatures=(0.8, 1.0, 1.2), tail="lump", top_p=0.95, top_k=20
    )
    assert [row["temperature"] for row in rows] == [0.8, 1.0, 1.2]
    for row in rows:
        assert len(row["beta"]) == DEPTH
        assert all(0.0 <= value <= 1.0 for value in row["beta"])
        assert row["tokens_per_window"] == pytest.approx(row["expected_length"] + 1.0)
    text = temp_mod.report(rows, ms_per_window=37.47)
    assert "tok/s" in text and "P(alpha = 0)" in text
