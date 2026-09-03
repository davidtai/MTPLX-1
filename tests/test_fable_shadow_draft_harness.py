"""Pure-python coverage for the shadow-draft acceptance harness.

No MLX import happens here, and the first test proves none can: every
device-touching import in ``scripts/fable/shadow_draft_harness.py`` lives
inside :func:`build_replay_hooks`, so the trajectory reconstruction, the replay
orchestration, the scorer and the report are all reachable with no MLX in the
process.  The orchestration is driven with a stub hook, exactly as
``tests/test_fable_depth4_probe.py`` drives ``run_probe``.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.fable import shadow_draft_harness as harness
from scripts.fable.shadow_draft_harness import (
    ProposalVariant,
    Segment,
    ShadowRows,
    TrajectoryGapError,
    block_bootstrap_se,
    budget,
    empty_shadow_rows,
    expected_tokens,
    load_variant_module,
    lookup_prob,
    pad_row,
    parse_variant,
    reach_from_alpha,
    realised_accepts,
    replay_windows,
    report,
    request_ids,
    score,
    segment_windows,
    total_variation,
    variant_env_scope,
    window_carry,
    window_emission,
    window_record,
)

HARNESS_PATH = Path(harness.__file__)
WIDTH = 4
DEPTH = 3


# ---------------------------------------------------------------------------
# Synthetic logs and shadow rows
# ---------------------------------------------------------------------------


def _row(support, width=WIDTH):
    """``(ids, values, probs)`` for one ``{id: prob}`` support, zero-padded."""

    ids = np.zeros(width, dtype=np.uint32)
    probs = np.zeros(width, dtype=np.float64)
    for position, (token, mass) in enumerate(sorted(support.items())):
        ids[position] = np.uint32(token)
        probs[position] = float(mass)
    for position in range(len(support), width):
        ids[position] = np.uint32(0xFFFFFFFF - position)
    with np.errstate(divide="ignore"):
        values = np.where(probs > 0.0, np.log(probs), -3.0e38).astype(np.float32)
    return ids, values, probs


def make_log(windows, *, depth=DEPTH, width=WIDTH, layout="stock_prepared"):
    """A K20 log from a list of per-window dicts.

    Each window dict carries ``primary``, ``draft_tokens``, ``draft`` (one
    support per depth), ``target`` (one per depth, plus an optional bonus row),
    ``accepted``, ``selected_token``, ``selected_present``, ``uniforms``.
    """

    count = len(windows)
    rows = depth + 1
    log = {
        "layout": np.asarray(layout),
        "has_raw_logits": np.uint8(0),
        "draft_ids": np.zeros((count, depth, width), dtype=np.uint32),
        "draft_values": np.zeros((count, depth, width), dtype=np.float32),
        "draft_probs": np.zeros((count, depth, width), dtype=np.float64),
        "draft_valid": np.ones((count, depth), dtype=np.uint8),
        "target_ids": np.zeros((count, rows, width), dtype=np.uint32),
        "target_values": np.zeros((count, rows, width), dtype=np.float32),
        "target_probs": np.zeros((count, rows, width), dtype=np.float64),
        "target_valid": np.ones((count, rows), dtype=np.uint8),
        "draft_tokens": np.zeros((count, depth), dtype=np.uint32),
        "primary": np.zeros(count, dtype=np.uint32),
        "decision_uniforms": np.zeros((count, depth + 1), dtype=np.float64),
        "decision_uniforms_valid": np.full(count, depth + 1, dtype=np.uint8),
        "draft_uniforms": np.full((count, depth), np.nan, dtype=np.float64),
        "accepted": np.zeros(count, dtype=np.uint32),
        "first_reject": np.full(count, -1, dtype=np.int32),
        "selected_token": np.zeros(count, dtype=np.uint32),
        "selected_kind": np.ones(count, dtype=np.uint32),
        "selected_present": np.ones(count, dtype=np.uint8),
        "draws_used": np.zeros(count, dtype=np.uint32),
        "accept_probability": np.zeros((count, depth), dtype=np.float64),
        "accept_probability_valid": np.zeros(count, dtype=np.uint8),
        "bonus_allowed": np.ones(count, dtype=np.uint8),
        "greedy": np.zeros(count, dtype=np.uint8),
        "rng_state": np.zeros((count, 4), dtype=np.uint64),
        "stop_ids": np.zeros(0, dtype=np.uint32),
        "temperature": np.float64(1.0),
        "draft_temperature": np.float64(1.0),
        "top_p": np.float64(0.95),
        "top_k": np.int64(20),
    }
    # A PCG64 stream id per request, written into the `inc` half of
    # `rng_state` exactly as `_pcg64_state` does.  Opt-in: a window dict with
    # no "stream" leaves the column zero, which is what the PR391 lane and the
    # older stock logs look like, so the emitted-stream fallback stays covered.
    if any("stream" in window for window in windows):
        for index, window in enumerate(windows):
            stream = int(window.get("stream", 0))
            log["rng_state"][index] = (
                index + 1,
                index + 7,
                0xA5A5_0000 + stream,
                0x5A5A_0000 + stream,
            )
    # Unlogged commits.  Opt-in for the same reason: a log written before the
    # logger accounted for them has neither column.
    carries = [list(window.get("carry", ())) for window in windows]
    if any(carries):
        log["carry_len"] = np.asarray([len(c) for c in carries], dtype=np.uint32)
        carry_width = max(len(c) for c in carries)
        carry_tokens = np.zeros((count, carry_width), dtype=np.uint32)
        for index, tokens in enumerate(carries):
            carry_tokens[index, : len(tokens)] = np.asarray(tokens, dtype=np.uint32)
        log["carry_tokens"] = carry_tokens
    for index, window in enumerate(windows):
        log["primary"][index] = window["primary"]
        log["draft_tokens"][index] = window["draft_tokens"]
        log["accepted"][index] = window["accepted"]
        log["selected_token"][index] = window["selected_token"]
        log["selected_present"][index] = np.uint8(window.get("selected_present", 1))
        log["bonus_allowed"][index] = np.uint8(window.get("bonus_allowed", 1))
        log["decision_uniforms"][index] = window.get(
            "uniforms", [0.5] * (depth + 1)
        )
        if "draft_uniforms" in window:
            log["draft_uniforms"][index] = window["draft_uniforms"]
        for level, support in enumerate(window["draft"]):
            ids, values, probs = _row(support, width)
            log["draft_ids"][index, level] = ids
            log["draft_values"][index, level] = values
            log["draft_probs"][index, level] = probs
        targets = list(window["target"])
        while len(targets) < rows:
            targets.append(targets[-1])
        for level, support in enumerate(targets):
            ids, values, probs = _row(support, width)
            log["target_ids"][index, level] = ids
            log["target_values"][index, level] = values
            log["target_probs"][index, level] = probs
    return log


def shadow_from(log, supports_by_variant, *, depth=DEPTH, width=WIDTH):
    """Shadow rows from ``{variant: [[support per depth] per window]}``."""

    names = list(supports_by_variant)
    count = int(log["draft_tokens"].shape[0])
    shadow = empty_shadow_rows(
        cycles=count, variants=names, depth=depth, width=width
    )
    for position, name in enumerate(names):
        for index, per_depth in enumerate(supports_by_variant[name]):
            for level, support in enumerate(per_depth):
                ids, probs = pad_row(
                    sorted(support), [support[k] for k in sorted(support)], width=width
                )
                shadow.ids[index, position, level] = ids
                shadow.probs[index, position, level] = probs
                shadow.valid[index, position, level] = 1
                shadow.tokens[index, position, level] = log["draft_tokens"][
                    index, level
                ]
    return shadow


def simple_window(primary, drafts, selected, *, accepted=None, **extra):
    """One window whose draft rows equal its target rows: alpha == 1 at every depth."""

    support = {int(drafts[0]): 0.6, int(drafts[0]) + 100: 0.4}
    window = {
        "primary": primary,
        "draft_tokens": list(drafts),
        "draft": [dict(support) for _ in drafts],
        "target": [dict(support) for _ in range(len(drafts) + 1)],
        "accepted": len(drafts) if accepted is None else accepted,
        "selected_token": selected,
    }
    window.update(extra)
    return window


# ---------------------------------------------------------------------------
# Structure: the GPU stays in one function
# ---------------------------------------------------------------------------


def test_every_device_import_lives_inside_build_replay_hooks():
    tree = ast.parse(HARNESS_PATH.read_text())
    gpu_roots = {"mlx", "mtplx"}

    def _roots(node):
        if isinstance(node, ast.Import):
            return {alias.name.split(".")[0] for alias in node.names}
        return {(node.module or "").split(".")[0]}

    offenders = []
    for parent in ast.walk(tree):
        inside_function = isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        for node in ast.iter_child_nodes(parent):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if not (_roots(node) & gpu_roots):
                continue
            if not inside_function:
                offenders.append((getattr(parent, "name", "<module>"), node.lineno))
    assert offenders == [], (
        "a device-side import escaped a function body; importing this module "
        f"would then pull MLX into every pure scoring run: {offenders}"
    )

    source = inspect.getsource(harness.build_replay_hooks)
    assert "import mlx.core as mx" in source
    # And nothing else in the module imports mlx at all.
    mlx_functions = {
        getattr(parent, "name", "<module>")
        for parent in ast.walk(tree)
        for node in ast.iter_child_nodes(parent)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "mlx" in _roots(node)
    }
    assert mlx_functions == {"build_replay_hooks"}


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------


def test_window_emission_is_the_accepted_prefix_plus_the_selection():
    log = make_log([simple_window(7, [11, 12, 13], 20, accepted=2)])
    assert window_emission(log, 0) == [11, 12, 20]


def test_window_emission_drops_the_selection_when_a_stop_fired():
    log = make_log(
        [simple_window(7, [11, 12, 13], 0, accepted=2, selected_present=0)]
    )
    assert window_emission(log, 0) == [11, 12]


def test_segment_windows_follows_the_carry_in_across_one_request():
    log = make_log(
        [
            simple_window(7, [11, 12, 13], 20, accepted=1),
            simple_window(20, [21, 22, 23], 30, accepted=3),
        ]
    )
    segments = segment_windows(log)
    assert len(segments) == 1
    assert segments[0] == Segment(index=0, start=0, stop=2, tokens=(7, 11, 20, 21, 22, 23, 30))


def test_segment_windows_splits_on_a_broken_carry_in():
    log = make_log(
        [
            simple_window(7, [11, 12, 13], 20, accepted=1),
            simple_window(99, [21, 22, 23], 30, accepted=1),  # primary != 20
        ]
    )
    segments = segment_windows(log)
    assert [(s.start, s.stop) for s in segments] == [(0, 1), (1, 2)]
    assert segments[0].tokens == (7, 11, 20)
    assert segments[1].tokens == (99, 21, 30)


def test_segment_windows_splits_on_a_stop_token_end():
    log = make_log(
        [
            simple_window(7, [11, 12, 13], 0, accepted=1, selected_present=0),
            simple_window(0, [21, 22, 23], 30, accepted=1),
        ]
    )
    assert [(s.start, s.stop) for s in segment_windows(log)] == [(0, 1), (1, 2)]


def test_segment_windows_handles_three_seeds():
    windows = []
    for seed in range(3):
        base = 1000 * (seed + 1)
        windows.append(simple_window(base, [base + 1, base + 2, base + 3], base + 9))
        windows.append(
            simple_window(base + 9, [base + 4, base + 5, base + 6], base + 8)
        )
    segments = segment_windows(make_log(windows))
    assert [s.windows for s in segments] == [2, 2, 2]


def test_request_ids_reads_a_constant_stream_as_one_request():
    one = [simple_window(7, [11, 12, 13], 20), simple_window(20, [21, 22, 23], 30)]
    constant = [dict(window, stream=4) for window in one]
    assert list(request_ids(make_log(constant))) == [0, 0]


def test_request_ids_reads_the_pcg64_stream_id():
    windows = []
    for request in range(3):
        base = 1000 * (request + 1)
        windows.append(
            simple_window(base, [base + 1, base + 2, base + 3], base + 9, stream=request)
        )
        windows.append(
            simple_window(
                base + 9, [base + 4, base + 5, base + 6], base + 8, stream=request
            )
        )
    assert list(request_ids(make_log(windows))) == [0, 0, 1, 1, 2, 2]


def test_request_ids_declines_a_column_that_cannot_carry_the_answer():
    one = [simple_window(7, [11, 12, 13], 20), simple_window(20, [21, 22, 23], 30)]
    # All zero: the PR391 lane and every pre-stream test stub.
    assert request_ids(make_log(one)) is None
    # Interleaved: a stream that comes back is not a request id.
    interleaved = [
        simple_window(7, [11, 12, 13], 20, stream=0),
        simple_window(20, [21, 22, 23], 30, stream=1),
        simple_window(30, [31, 32, 33], 40, stream=0),
    ]
    assert request_ids(make_log(interleaved)) is None


def test_window_carry_is_empty_on_a_log_without_the_column():
    log = make_log([simple_window(7, [11, 12, 13], 20)])
    assert "carry_len" not in log
    assert window_carry(log, 0) == []


# The pattern that over-segmented the W51 shadow log (2026-09-02): a
# context-copy block round commits a slice of the prompt, its residual
# correction, and -- when it accepted the whole block -- the next freshly
# sampled primary, all WITHOUT a K20 row.  Three shapes occur, and all three
# are here: a gap that ends on a correction, a gap that ends on a fresh
# primary, and two adjacent rounds inside one gap.  The real log held 21 of
# them across 3 requests and the reconstruction read every one as a request
# boundary, reporting 24 segments.
def _copy_gap_log(*, record_carry):
    def gap(tokens):
        return {"carry": list(tokens)} if record_carry else {}

    windows = [
        # Request 0: a gap that ends on the round's own correction.
        simple_window(7, [11, 12, 13], 20, stream=0, **gap([901, 902, 903])),
        simple_window(903, [31, 32, 33], 40, stream=0),
        # Request 0 again: two adjacent copy rounds inside one gap.
        simple_window(
            40, [41, 42, 43], 50, stream=0, **gap([911, 912, 921, 922, 923])
        ),
        simple_window(923, [61, 62, 63], 70, stream=0),
        # Request 1 starts.  Its first gap ends on a freshly sampled primary
        # (the round accepted its whole block and emitted no correction).
        simple_window(100, [101, 102, 103], 120, stream=1, **gap([931, 932, 933])),
        simple_window(933, [141, 142, 143], 150, stream=1),
        # ... and request 1 is cut by max tokens: nothing is emitted after the
        # accepted prefix, which is the only genuine boundary in this log.
        simple_window(
            150, [151, 152, 153], 0, accepted=3, selected_present=0, stream=1
        ),
        # Request 2.
        simple_window(200, [201, 202, 203], 220, stream=2),
        simple_window(220, [241, 242, 243], 250, stream=2),
    ]
    return make_log(windows)


def test_segment_windows_keeps_one_request_across_a_recorded_copy_lane_gap():
    segments = segment_windows(_copy_gap_log(record_carry=True))
    assert [(s.start, s.stop) for s in segments] == [(0, 4), (4, 7), (7, 9)]
    # The gap's tokens are in the reconstructed stream, in commit order,
    # between the window's emission and the next window's primary.
    assert segments[0].tokens == (
        7, 11, 12, 13, 20, 901, 902, 903,
        31, 32, 33, 40,
        41, 42, 43, 50, 911, 912, 921, 922, 923,
        61, 62, 63, 70,
    )
    assert segments[1].tokens[:9] == (100, 101, 102, 103, 120, 931, 932, 933, 141)


def test_segment_windows_refuses_a_copy_lane_gap_the_log_never_recorded():
    log = _copy_gap_log(record_carry=False)
    with pytest.raises(TrajectoryGapError) as raised:
        segment_windows(log)
    message = str(raised.value)
    # It reports the holes, not a pile of phantom requests.
    assert "3 window(s) of 3 reconstructed request segment(s)" in message
    assert "carry_len" in message
    assert "context_copy" in message


def test_a_single_request_with_a_gap_raises_instead_of_splitting_in_two():
    # The hazard the stream id closes: ONE request whose copy round the log
    # never recorded.  Reading the token stream would call the gap a second
    # request and replay both halves from the wrong prefill, silently.
    log = make_log(
        [
            simple_window(7, [11, 12, 13], 20, stream=0),
            simple_window(903, [31, 32, 33], 40, stream=0),
        ]
    )
    with pytest.raises(TrajectoryGapError, match="1 window\\(s\\) of 1 reconstructed"):
        segment_windows(log)


def test_segment_windows_still_reads_the_stream_when_no_request_id_exists():
    # No stream column: the PR391 lane and the older stock logs.  The
    # emitted-stream heuristic is the only thing left, and it is unchanged.
    log = make_log(
        [
            simple_window(7, [11, 12, 13], 20),
            simple_window(99, [21, 22, 23], 30),
        ]
    )
    assert [(s.start, s.stop) for s in segment_windows(log)] == [(0, 1), (1, 2)]


def test_main_fails_loudly_on_a_log_that_lost_committed_tokens(tmp_path, capsys):
    log = _copy_gap_log(record_carry=False)
    path = tmp_path / "rows.npz"
    np.savez_compressed(path, **log)
    assert harness.main([str(path), "--expect-segments", "3", "--budget"]) == 1
    error = capsys.readouterr().err
    assert "does not hold the whole committed stream" in error
    assert "MTPLX_CONTEXT_COPY=0" in error


def test_main_accepts_the_same_log_once_the_carry_is_recorded(tmp_path, capsys):
    log = _copy_gap_log(record_carry=True)
    path = tmp_path / "rows.npz"
    np.savez_compressed(path, **log)
    assert harness.main([str(path), "--expect-segments", "3", "--budget"]) == 0
    assert json.loads(capsys.readouterr().out)["segments"] == 3


def test_window_record_hands_the_replay_the_carry_to_commit():
    log = _copy_gap_log(record_carry=True)
    assert window_record(log, 0)["carry"] == [901, 902, 903]
    assert window_record(log, 1)["carry"] == []


# ---------------------------------------------------------------------------
# The acceptance law
# ---------------------------------------------------------------------------


def test_alpha_is_one_when_the_proposal_matches_the_target_exactly():
    log = make_log([simple_window(7, [11, 12, 13], 20)])
    same = [[{11: 0.6, 111: 0.4}] * DEPTH]
    shadow = shadow_from(log, {"stock": same})
    result = score(log, shadow)
    for depth in range(DEPTH):
        assert result["variants"][0]["alpha"][depth][0] == pytest.approx(1.0)


def test_alpha_is_zero_on_disjoint_support():
    log = make_log([simple_window(7, [11, 12, 13], 20)])
    disjoint = [[{500: 0.5, 501: 0.5}] * DEPTH]
    shadow = shadow_from(log, {"stock": disjoint})
    result = score(log, shadow)
    for depth in range(DEPTH):
        assert result["variants"][0]["alpha"][depth][0] == pytest.approx(0.0)


def test_alpha_is_the_overlap_sum_of_min():
    # target {11: 0.6, 111: 0.4}; proposal {11: 0.25, 111: 0.75}
    # sum min = 0.25 + 0.40 = 0.65
    log = make_log([simple_window(7, [11, 12, 13], 20)])
    proposal = [[{11: 0.25, 111: 0.75}] * DEPTH]
    result = score(log, shadow_from(log, {"stock": proposal}))
    assert result["variants"][0]["alpha"][0][0] == pytest.approx(0.65)


def test_alpha_handles_an_exact_tie_without_drift():
    log = make_log(
        [
            {
                "primary": 7,
                "draft_tokens": [11, 11, 11],
                "draft": [{11: 0.5, 111: 0.5}] * DEPTH,
                "target": [{11: 0.5, 111: 0.5}] * (DEPTH + 1),
                "accepted": 3,
                "selected_token": 20,
            }
        ]
    )
    tied = [[{11: 0.5, 111: 0.5}] * DEPTH]
    result = score(log, shadow_from(log, {"stock": tied}))
    assert result["variants"][0]["alpha"][0][0] == 1.0
    assert result["fidelity"]["max_tv"] == 0.0


def test_reach_is_the_running_product_and_tokens_add_the_selection():
    alpha = np.array([[0.9, 0.8, 0.5]])
    reach = reach_from_alpha(alpha)
    assert reach[0].tolist() == pytest.approx([0.9, 0.72, 0.36])
    # sum(reach) + (1 - w_D) + w_D * bonus, bonus allowed -> sum(reach) + 1
    assert expected_tokens(reach, np.array([1.0]))[0] == pytest.approx(2.98)
    # bonus disallowed -> the all-accept branch emits nothing extra
    assert expected_tokens(reach, np.array([0.0]))[0] == pytest.approx(2.62)


def test_depth_conditional_alpha_differs_from_the_marginal_ladder():
    """The reach ladder conditions on the previous depths; alpha does not.

    With a1 = 1.0 and a2 = 0.5, the marginal alpha at depth 2 stays 0.5 while
    the *reach* through depth 2 is the product -- which is the number
    E[tokens/window] is built from.
    """

    reach = reach_from_alpha(np.array([[1.0, 0.5, 0.5]]))
    assert reach[0].tolist() == pytest.approx([1.0, 0.5, 0.25])
    assert expected_tokens(reach, np.array([1.0]))[0] == pytest.approx(2.75)


def test_realised_accepts_reads_the_uniform_tape():
    rho = np.array([[1.0, 1.0, 1.0], [0.2, 1.0, 1.0], [1.0, 0.1, 1.0]])
    uniforms = np.array([[0.5, 0.5, 0.5, 0.0]] * 3)
    assert realised_accepts(rho, uniforms).tolist() == [3, 0, 1]


def test_lookup_prob_is_exact_and_zero_off_support():
    ids = np.array([3, 9, 42], dtype=np.uint32)
    probs = np.array([0.1, 0.2, 0.7])
    assert lookup_prob(ids, probs, 9) == pytest.approx(0.2)
    assert lookup_prob(ids, probs, 10) == 0.0
    assert lookup_prob(np.zeros(0, dtype=np.uint32), np.zeros(0), 3) == 0.0


def test_total_variation_matches_one_minus_overlap_for_normalised_rows():
    left_ids = np.array([1, 2], dtype=np.uint32)
    right_ids = np.array([2, 3], dtype=np.uint32)
    distance = total_variation(
        left_ids, np.array([0.4, 0.6]), right_ids, np.array([0.5, 0.5])
    )
    # |0.4| + |0.6-0.5| + |0.5| = 1.0 -> half is 0.5; overlap is min(0.6,0.5)=0.5
    assert distance == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pairing, fidelity, arming
# ---------------------------------------------------------------------------


def _three_window_log():
    windows = [
        simple_window(7, [11, 12, 13], 20),
        simple_window(20, [21, 22, 23], 30),
        simple_window(30, [31, 32, 33], 40),
    ]
    return make_log(windows)


def _stock_supports(log):
    """The logged q rows, read straight back out -- a perfect-fidelity replay."""

    count = int(log["draft_tokens"].shape[0])
    out = []
    for index in range(count):
        per_depth = []
        for level in range(DEPTH):
            ids = log["draft_ids"][index, level]
            probs = log["draft_probs"][index, level]
            keep = probs > 0.0
            per_depth.append(
                {int(i): float(p) for i, p in zip(ids[keep], probs[keep])}
            )
        out.append(per_depth)
    return out


def _shifted(supports, first=0.7):
    """The same rows with mass moved between the two ids -- an armed candidate."""

    return [
        [{sorted(row)[0]: first, sorted(row)[1]: 1.0 - first} for row in window]
        for window in supports
    ]


def test_fidelity_passes_when_the_replay_reproduces_the_logged_rows():
    log = _three_window_log()
    result = score(log, shadow_from(log, {"stock": _stock_supports(log)}))
    assert result["fidelity"]["pass"] is True
    assert result["fidelity"]["max_tv"] == pytest.approx(0.0)
    assert "fidelity PASS" in report(result)


def test_fidelity_fails_and_the_verdict_is_withheld_when_the_replay_drifts():
    log = _three_window_log()
    supports = _stock_supports(log)
    drifted = [
        [{token: mass for token, mass in row.items()} for row in window]
        for window in supports
    ]
    first = drifted[0][0]
    keys = sorted(first)
    first[keys[0]] += 0.05
    first[keys[1]] -= 0.05
    result = score(log, shadow_from(log, {"stock": drifted}))
    assert result["fidelity"]["pass"] is False
    text = report(result)
    assert "fidelity FAIL" in text
    assert "VERDICT WITHHELD" in text


def test_a_candidate_identical_to_stock_is_reported_as_not_armed():
    log = _three_window_log()
    supports = _stock_supports(log)
    result = score(
        log, shadow_from(log, {"stock": supports, "candidate": supports})
    )
    candidate = result["variants"][1]
    assert candidate["armed"] is False
    assert "DID NOT ARM" in report(result)


def test_a_uniform_candidate_shift_gives_a_zero_variance_paired_delta():
    """A candidate whose every row is the same shift has no paired noise.

    That is the whole point of the design: the difference is taken window by
    window against the SAME p rows, so a constant effect carries a zero
    standard error rather than the +-4% of a live A/B.
    """

    log = _three_window_log()
    stock = _stock_supports(log)
    better = []
    for window in stock:
        per_depth = []
        for row in window:
            keys = sorted(row)
            per_depth.append({keys[0]: 0.7, keys[1]: 0.3})
        better.append(per_depth)
    result = score(log, shadow_from(log, {"stock": stock, "candidate": better}))
    paired = result["variants"][1]["paired"]
    assert paired["delta_tokens"][1] == pytest.approx(0.0, abs=1e-12)
    # stock alpha is 1.0 everywhere; the candidate's is min(0.6,0.7)+min(0.4,0.3)
    assert paired["delta_alpha"][0][0] == pytest.approx(0.9 - 1.0)
    assert result["variants"][1]["armed"] is True


def test_score_skips_greedy_windows_rather_than_scoring_them():
    log = _three_window_log()
    log["greedy"][1] = 1
    result = score(log, shadow_from(log, {"stock": _stock_supports(log)}))
    assert result["cycles_scored"] == 2
    assert result["cycles_skipped"] == 1


def test_score_rejects_a_non_stock_layout():
    log = _three_window_log()
    log["layout"] = np.asarray("pr391_raw")
    with pytest.raises(ValueError, match="not a stock-lane log"):
        score(log, shadow_from(log, {"stock": _stock_supports(log)}))


def test_score_rejects_shadow_rows_without_a_stock_arm():
    log = _three_window_log()
    shadow = shadow_from(log, {"candidate": _stock_supports(log)})
    with pytest.raises(ValueError, match="no 'stock' control arm"):
        score(log, shadow)


def test_per_segment_reports_each_seed_separately():
    windows = []
    for seed in range(3):
        base = 1000 * (seed + 1)
        windows.append(simple_window(base, [base + 1, base + 2, base + 3], base + 9))
        windows.append(
            simple_window(base + 9, [base + 4, base + 5, base + 6], base + 8)
        )
    log = make_log(windows)
    stock = _stock_supports(log)
    result = score(log, shadow_from(log, {"stock": stock, "candidate": stock}))
    assert [entry["segment"] for entry in result["per_segment"]] == [0, 1, 2]
    assert all(entry["windows"] == 2 for entry in result["per_segment"])


def test_divergence_uses_the_draft_tape_when_the_layout_recorded_one():
    log = _three_window_log()
    log["draft_uniforms"][:] = 0.99  # past the first id's mass, into the second
    log["layout"] = np.asarray("stock_device_k20")
    stock = _stock_supports(log)
    result = score(log, shadow_from(log, {"stock": stock, "candidate": _shifted(stock)}))
    assert result["draft_tape_used"] is True
    # every row is {drafted: 0.6, drafted+100: 0.4}; a 0.99 coin picks the
    # second id, which is not the drafted token.
    assert result["variants"][0]["diverge"][0] == pytest.approx(1.0)
    assert "logged draft tape" in report(result)


def test_divergence_falls_back_to_argmax_without_a_tape():
    log = _three_window_log()
    stock = _stock_supports(log)
    result = score(log, shadow_from(log, {"stock": stock, "candidate": _shifted(stock)}))
    assert result["draft_tape_used"] is False
    assert result["variants"][0]["diverge"][0] == pytest.approx(0.0)
    assert "argmax proxy" in report(result)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_block_bootstrap_se_is_deterministic_and_near_the_iid_se_when_iid():
    rng = np.random.default_rng(11)
    values = rng.normal(size=600)
    owners = np.repeat(np.arange(3), 200)
    first = block_bootstrap_se(values, owners, resamples=400, block=8, seed=5)
    second = block_bootstrap_se(values, owners, resamples=400, block=8, seed=5)
    assert first == second
    iid = float(np.std(values, ddof=1) / np.sqrt(values.size))
    assert first == pytest.approx(iid, rel=0.35)


def test_block_bootstrap_se_is_nan_below_two_samples():
    assert np.isnan(
        block_bootstrap_se(np.array([1.0]), np.array([0]), resamples=10)
    )


def test_budget_prices_the_marginal_chain_separately():
    result = {
        "cycles": 1110,
        "cycles_scored": 1110,
        "segments": [{}, {}, {}],
        "variants": [{}, {}],
    }
    estimate = budget(result, ms_per_window=38.7, chain_ms=5.0, prefill_s=4.0)
    assert estimate["windows"] == 1110
    assert estimate["segments"] == 3
    # 370 windows/seed: 4.0 prefill + 370*0.0387 + 370*2*0.005
    assert estimate["per_seed_s"] == pytest.approx(4.0 + 14.319 + 3.7)
    assert estimate["total_s"] == pytest.approx(3 * estimate["per_seed_s"])
    assert estimate["marginal_chain_total_s"] == pytest.approx(5.55)


# ---------------------------------------------------------------------------
# Replay orchestration, driven by a stub
# ---------------------------------------------------------------------------


class _StubHooks(harness.ReplayHooks):
    """Records the call order and hands back a row keyed by the call state."""

    def __init__(self, depth=DEPTH):
        self.depth = depth
        self.calls = []
        self.segments = []
        self.window = 0
        self.closed = False

    def start_segment(self, segment):
        self.segments.append(segment)
        self.calls.append(("start", segment.index))

    def draft_rows(self, *, variant, forced_tokens):
        self.calls.append(("draft", variant.name, tuple(forced_tokens)))
        mass = 0.9 if variant.is_stock else 0.6
        return [
            (
                np.array([int(token), int(token) + 100], dtype=np.int64),
                np.array([mass, 1.0 - mass], dtype=np.float64),
            )
            for token in forced_tokens
        ]

    def advance(self, *, window):
        self.calls.append(("advance", window["index"], window["accepted"]))

    def close(self):
        self.closed = True


def test_replay_runs_every_variant_from_the_same_state_before_advancing():
    log = _three_window_log()
    hooks = _StubHooks()
    variants = [
        ProposalVariant(name="stock"),
        ProposalVariant(name="candidate", env={"MTPLX_FABLE_INDEXER_REUSE": "1"}),
    ]
    shadow = replay_windows(log, hooks, variants)

    assert hooks.closed is True
    assert hooks.calls[0] == ("start", 0)
    # Both arms draft, then the window advances: never the other way round.
    assert hooks.calls[1][:2] == ("draft", "stock")
    assert hooks.calls[2][:2] == ("draft", "candidate")
    assert hooks.calls[3][0] == "advance"
    assert shadow.variants == ["stock", "candidate"]
    assert shadow.valid.all()
    assert shadow.variant_env == {"candidate": {"MTPLX_FABLE_INDEXER_REUSE": "1"}}


def test_replay_teacher_forces_the_logged_draft_tokens():
    log = _three_window_log()
    hooks = _StubHooks()
    replay_windows(log, hooks, [ProposalVariant(name="stock")])
    forced = [call[2] for call in hooks.calls if call[0] == "draft"]
    assert forced == [(11, 12, 13), (21, 22, 23), (31, 32, 33)]


def test_replay_requires_the_stock_arm_first():
    log = _three_window_log()
    with pytest.raises(ValueError, match="variant 0 must be the stock"):
        replay_windows(log, _StubHooks(), [ProposalVariant(name="candidate")])


def test_replay_rejects_duplicate_variant_names():
    log = _three_window_log()
    with pytest.raises(ValueError, match="duplicate variant names"):
        replay_windows(
            log,
            _StubHooks(),
            [ProposalVariant(name="stock"), ProposalVariant(name="stock")],
        )


def test_replay_rejects_a_hook_that_returns_the_wrong_depth():
    log = _three_window_log()

    class _Short(_StubHooks):
        def draft_rows(self, *, variant, forced_tokens):
            return super().draft_rows(variant=variant, forced_tokens=forced_tokens)[:1]

    with pytest.raises(RuntimeError, match="returned 1 rows"):
        replay_windows(log, _Short(), [ProposalVariant(name="stock")])


def test_replay_honours_limit():
    log = _three_window_log()
    hooks = _StubHooks()
    shadow = replay_windows(log, hooks, [ProposalVariant(name="stock")], limit=2)
    assert shadow.cycles == 2
    assert sum(1 for call in hooks.calls if call[0] == "advance") == 2


def test_replayed_rows_score_end_to_end():
    log = _three_window_log()
    shadow = replay_windows(
        log,
        _StubHooks(),
        [ProposalVariant(name="stock"), ProposalVariant(name="candidate")],
    )
    result = score(log, shadow)
    assert result["cycles_scored"] == 3
    # The stub's stock row is {token: 0.9, token+100: 0.1}; the logged one is
    # {token: 0.6, token+100: 0.4} -- so the fidelity gate must fire.
    assert result["fidelity"]["pass"] is False
    assert result["variants"][1]["armed"] is True


# ---------------------------------------------------------------------------
# Variants and shadow-row round trip
# ---------------------------------------------------------------------------


def test_parse_variant_reads_multiple_assignments():
    variant = parse_variant("reuse=MTPLX_FABLE_INDEXER_REUSE=1,MTPLX_X=0")
    assert variant.name == "reuse"
    assert variant.env == {"MTPLX_FABLE_INDEXER_REUSE": "1", "MTPLX_X": "0"}


@pytest.mark.parametrize("spec", ["", "reuse", "reuse=", "=1", "reuse=novalue"])
def test_parse_variant_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_variant(spec)


def test_parse_variant_refuses_to_shadow_the_control_arm():
    with pytest.raises(ValueError, match="reserved"):
        parse_variant("stock=MTPLX_X=1")


def test_variant_env_scope_restores_an_absent_key_to_absent(monkeypatch):
    monkeypatch.delenv("MTPLX_FABLE_TEST_KEY", raising=False)
    with variant_env_scope({"MTPLX_FABLE_TEST_KEY": "1"}):
        import os

        assert os.environ["MTPLX_FABLE_TEST_KEY"] == "1"
    import os

    assert "MTPLX_FABLE_TEST_KEY" not in os.environ


def test_variant_env_scope_restores_a_previous_value(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_TEST_KEY", "old")
    with variant_env_scope({"MTPLX_FABLE_TEST_KEY": "new"}):
        pass
    import os

    assert os.environ["MTPLX_FABLE_TEST_KEY"] == "old"


def test_load_variant_module_rejects_a_bad_spec():
    with pytest.raises(ValueError, match="dotted.module:factory"):
        load_variant_module("no_colon_here")


def test_shadow_rows_round_trip(tmp_path):
    log = _three_window_log()
    shadow = shadow_from(log, {"stock": _stock_supports(log)})
    shadow.source = "rows.npz"
    shadow.variant_env = {"candidate": {"K": "1"}}
    path = tmp_path / "shadow.npz"
    shadow.save(str(path))
    back = ShadowRows.load(str(path))
    assert back.variants == shadow.variants
    assert back.mode == "forced"
    assert back.source == "rows.npz"
    assert back.variant_env == {"candidate": {"K": "1"}}
    np.testing.assert_array_equal(back.ids, shadow.ids)
    np.testing.assert_array_equal(back.probs, shadow.probs)


def test_shadow_rows_load_names_the_missing_key(tmp_path):
    path = tmp_path / "not-shadow.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(KeyError, match="shadow_draft_harness"):
        ShadowRows.load(str(path))


def test_pad_row_rejects_an_oversized_support():
    with pytest.raises(ValueError, match="exceeds width"):
        pad_row([1, 2, 3], [0.3, 0.3, 0.4], width=2)


def test_window_record_carries_what_the_replay_needs():
    log = make_log([simple_window(7, [11, 12, 13], 20, accepted=2)])
    record = window_record(log, 0)
    assert record == {
        "index": 0,
        "primary": 7,
        "draft_tokens": [11, 12, 13],
        "accepted": 2,
        "selected_token": 20,
        "selected_present": True,
        "emission": [11, 12, 20],
        "carry": [],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_renders_every_variant_and_the_paired_line():
    log = _three_window_log()
    stock = _stock_supports(log)
    better = [
        [{sorted(row)[0]: 0.7, sorted(row)[1]: 0.3} for row in window]
        for window in stock
    ]
    result = score(log, shadow_from(log, {"stock": stock, "candidate": better}))
    text = report(result, budget_rows=budget(result))
    assert "stock" in text
    assert "candidate" in text
    assert "d(tok/win)" in text
    assert "P(diverge)" in text
    assert "capture budget (a MODEL, not a measurement)" in text
    assert "acceptance verdict only" in text


def test_report_says_so_when_there_is_no_candidate():
    log = _three_window_log()
    result = score(log, shadow_from(log, {"stock": _stock_supports(log)}))
    text = report(result)
    assert "stock only" in text
    assert "No verdict" in text


def test_result_is_json_serialisable():
    log = _three_window_log()
    stock = _stock_supports(log)
    result = score(log, shadow_from(log, {"stock": stock, "candidate": stock}))
    json.dumps(result)


# ---------------------------------------------------------------------------
# The one GPU function, checked without a GPU
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_level_functions(path):
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_methods(path):
    tree = ast.parse(path.read_text())
    return {
        method.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef)
    }


def test_build_replay_hooks_binds_names_that_actually_exist():
    """A rename in generation.py must break here, not on the first GPU run.

    ``build_replay_hooks`` cannot be exercised without a GPU, so the next best
    guard is static: every private helper it reaches into must still be a
    module-level function of ``mtplx/generation.py``, and every runtime call it
    makes must still be a method of a runtime class.  Both are checked by AST,
    with no import.
    """

    generation = _module_level_functions(REPO_ROOT / "mtplx" / "generation.py")
    for name in (
        "_distribution_from_mlx_logits",
        "_mtp_cache_offset",
        "_mtp_position_offset",
        "_rollback_mtp_cache",
        "_trim_cache_to_offset",
    ):
        assert name in generation, f"generation.py no longer defines {name}"

    runtime = _class_methods(REPO_ROOT / "mtplx" / "runtime.py")
    for name in (
        "forward_ar",
        "draft_mtp",
        "update_mtp_cache",
        "make_cache",
        "make_mtp_cache",
    ):
        assert name in runtime, f"runtime.py no longer defines {name}"

    # The chain must restore the MTP offset it found, or the two arms are not
    # paired -- the single most load-bearing line in the function.
    source = inspect.getsource(harness.build_replay_hooks)
    assert "finally:" in source and "_rollback_mtp_cache" in source


def test_production_prompt_helper_defers_its_driver_import():
    """`abba_driver` is a driver; pulling it in at import would drag its guard
    machinery into every pure scoring run."""

    source = inspect.getsource(harness._production_prompt)
    assert "from scripts.fable.abba_driver import build_production_cell" in source


def test_variant_scope_is_inert_without_env_or_call():
    with harness.variant_scope(ProposalVariant(name="stock")):
        pass


def test_variant_scope_uses_the_injected_call_when_given():
    entered = []

    class _Scope:
        def __enter__(self):
            entered.append("in")
            return self

        def __exit__(self, *exc):
            entered.append("out")
            return False

    variant = ProposalVariant(name="c", call=_Scope)
    with harness.variant_scope(variant):
        pass
    assert entered == ["in", "out"]


def test_arming_compares_against_the_stock_replay_not_the_log():
    """Two arms that drifted identically are still one arm.

    The fidelity gate and the arming check ask different questions, and the
    arming one must be answered against the replayed stock rows: a candidate
    that matches a *drifted* stock replay did not arm, even though both differ
    from the log.
    """

    log = _three_window_log()
    drifted = _shifted(_stock_supports(log), first=0.8)
    result = score(log, shadow_from(log, {"stock": drifted, "candidate": drifted}))
    assert result["fidelity"]["pass"] is False  # both drifted from the log
    assert result["variants"][1]["armed"] is False  # but they are each other
