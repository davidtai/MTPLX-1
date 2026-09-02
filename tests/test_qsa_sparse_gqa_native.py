"""CPU-only gates for the native direct-index sparse-GQA QSA kernel.

Two things are covered, neither of which dispatches Metal:

1. ``mtplx.native.qsa_sparse_gqa_unsupported_reason`` -- the shape/dtype/
   position contract.  MLX arrays are constructed but never evaluated, and
   the reason function only reads metadata, so no kernel is compiled or run.
2. ``scripts/fable/micro_qsa_sparse_gqa.py``'s scoring logic -- the
   visible-set identity checker and the bf16-ULP tolerance measure.  Those
   are pure host arithmetic and are exercised on numpy inputs, including the
   failure cases, because the whole parity story rests on them.

Numeric Metal parity belongs to the operator-controlled guarded window; see
the harness docstring for the command.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "fable" / "micro_qsa_sparse_gqa.py"

TOTAL = 16_384
CAPACITY = 32_768
ROWS = 64
TOP_K = 512
RATIO = 4
SCALE = 256 ** -0.5


def _load_harness():
    """Import the bench module without leaking its env arming into the session.

    The harness arms MTPLX_QSA_PREFILL* at module scope (its gates are read
    before ``mtplx.models.qwen4_exp`` is imported).  Importing it here must not
    change what any other test in the same pytest process sees.
    """

    before = dict(os.environ)
    spec = importlib.util.spec_from_file_location(
        "micro_qsa_sparse_gqa", HARNESS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        os.environ.clear()
        os.environ.update(before)
    module.np = np
    return module


harness = _load_harness()


# ---------------------------------------------------------------------------
# 1. binding validation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def native():
    mx = pytest.importorskip("mlx.core")
    native = pytest.importorskip("mtplx.native")
    if not native.native_qsa_available():
        pytest.skip("the native QSA extension is not built on this host")
    return mx, native


def _inputs(mx, *, rows=ROWS, ids_dtype=None, q_dtype=None):
    ids_dtype = ids_dtype or mx.int32
    q_dtype = q_dtype or mx.bfloat16
    return {
        "queries": mx.zeros((1, 24, rows, 256), q_dtype),
        "keys": mx.zeros((1, 2, CAPACITY, 256), q_dtype),
        "values": mx.zeros((1, 2, CAPACITY, 256), q_dtype),
        "block_ids": mx.zeros((rows, TOP_K), ids_dtype),
    }


def _reason(native_mod, arrays, **kwargs):
    params = {
        "pos_start": TOTAL - ROWS,
        "total_tokens": TOTAL,
        "scale": SCALE,
    }
    params.update(kwargs)
    return native_mod.qsa_sparse_gqa_unsupported_reason(
        arrays["queries"],
        arrays["keys"],
        arrays["values"],
        arrays["block_ids"],
        **params,
    )


def test_production_contract_is_accepted(native):
    mx, native_mod = native
    assert _reason(native_mod, _inputs(mx)) is None


def test_four_dimensional_block_ids_are_accepted(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = arrays["block_ids"].reshape(1, 1, ROWS, TOP_K)
    assert _reason(native_mod, arrays) is None


def test_uint32_block_ids_are_accepted(native):
    """The metallib instantiates int32 too, so no astype is forced on the lane."""

    mx, native_mod = native
    assert _reason(native_mod, _inputs(mx, ids_dtype=mx.uint32)) is None


@pytest.mark.parametrize(
    "tile,expected_ok",
    [((128, 32), True), ((256, 32), True), ((64, 64), True), ((128, 64), True),
     ((32, 32), False), ((128, 128), False), ((64, 32), False)],
)
def test_only_instantiated_tiles_are_accepted(native, tile, expected_ok):
    mx, native_mod = native
    reason = _reason(
        native_mod, _inputs(mx), key_tile=tile[0], dimension_tile=tile[1]
    )
    assert (reason is None) is expected_ok


def test_float32_queries_are_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx, q_dtype=mx.float32))
    assert reason is not None and "float16 or bfloat16" in reason


def test_mixed_dtypes_are_refused(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["keys"] = mx.zeros((1, 2, CAPACITY, 256), mx.float16)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "dtypes must match" in reason


def test_wrong_head_count_is_refused(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["queries"] = mx.zeros((1, 16, ROWS, 256), mx.bfloat16)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "[1, 24, S, 256]" in reason


def test_block_id_width_must_be_the_budget(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = mx.zeros((ROWS, 256), mx.int32)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "[S, 512]" in reason


def test_block_id_rows_must_match_queries(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = mx.zeros((ROWS + 1, TOP_K), mx.int32)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "[S, 512]" in reason


def test_float_block_ids_are_refused(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    arrays["block_ids"] = mx.zeros((ROWS, TOP_K), mx.float32)
    reason = _reason(native_mod, arrays)
    assert reason is not None and "int32 or uint32" in reason


def test_sub_crossover_context_is_refused(native):
    """2,048 tokens is exactly the boundary: 512 blocks is not yet sparse."""

    mx, native_mod = native
    reason = _reason(
        native_mod, _inputs(mx), pos_start=2048 - ROWS, total_tokens=2048
    )
    assert reason is not None and "dense/sparse boundary" in reason


def test_context_beyond_the_backing_is_refused(native):
    mx, native_mod = native
    reason = _reason(
        native_mod,
        _inputs(mx),
        pos_start=CAPACITY + 1,
        total_tokens=CAPACITY + 1 + ROWS,
    )
    assert reason is not None and "backing capacity" in reason


def test_non_causal_suffix_is_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx), pos_start=TOTAL, total_tokens=TOTAL)
    assert reason is not None and "causal suffix" in reason


def test_traced_scalars_are_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx), pos_start=mx.array(0))
    assert reason is not None and "host integers" in reason


def test_bool_positions_are_refused(native):
    mx, native_mod = native
    reason = _reason(native_mod, _inputs(mx), pos_start=True)
    assert reason is not None and "cannot be bool" in reason


def test_supported_predicate_matches_the_reason(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    assert native_mod.qsa_sparse_gqa_supported(
        arrays["queries"],
        arrays["keys"],
        arrays["values"],
        arrays["block_ids"],
        pos_start=TOTAL - ROWS,
        total_tokens=TOTAL,
        scale=SCALE,
    )
    assert not native_mod.qsa_sparse_gqa_supported(
        arrays["queries"],
        arrays["keys"],
        arrays["values"],
        arrays["block_ids"],
        pos_start=TOTAL - ROWS,
        total_tokens=TOTAL,
        scale=SCALE,
        key_tile=32,
        dimension_tile=32,
    )


def test_a_refused_call_raises_rather_than_dispatching(native):
    mx, native_mod = native
    arrays = _inputs(mx)
    with pytest.raises(ValueError, match="dense/sparse boundary"):
        native_mod.qsa_sparse_gqa(
            arrays["queries"],
            arrays["keys"],
            arrays["values"],
            arrays["block_ids"],
            pos_start=2048 - ROWS,
            total_tokens=2048,
            scale=SCALE,
        )


# ---------------------------------------------------------------------------
# 2. harness scoring logic (pure numpy)
# ---------------------------------------------------------------------------
def _selector_output(pos_start: int, rows: int, rng: np.random.Generator):
    """Reproduce what ``_select_eager`` guarantees: ascending valid prefix."""

    ids = np.zeros((rows, TOP_K), dtype=np.int64)
    ok = np.zeros((rows, TOP_K), dtype=bool)
    for r in range(rows):
        complete = (pos_start + r + 1) // RATIO
        n = min(TOP_K, complete)
        if n:
            ids[r, :n] = np.sort(rng.choice(complete, size=n, replace=False))
            ok[r, :n] = True
    return ids, ok


def _cell(pos_start: int, rows: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    ids, ok = _selector_output(pos_start, rows, rng)
    return {
        "rows": rows,
        "pos_start": pos_start,
        "block_ids": ids,
        "block_valid": ok,
    }


def test_identity_holds_on_a_well_formed_selection():
    cell = _cell(TOTAL - 256, 256)
    result = harness.check_identity(cell, sample_rows=8, strict=True)
    assert result["failures"] == []
    assert result["rows_checked"] == 256
    assert result["rows_token_sampled"] >= 8


def test_identity_holds_on_the_short_prefix_rows():
    """Rows near the crossover have fewer than 512 complete blocks."""

    cell = _cell(2040, 64)
    result = harness.check_identity(cell, sample_rows=64, strict=True)
    assert result["failures"] == []
    counts = cell["block_valid"].sum(axis=1)
    assert counts.min() < TOP_K  # the case the kernel's valid_blocks handles


def test_identity_catches_a_non_prefix_validity_mask():
    cell = _cell(TOTAL - 32, 32)
    cell["block_valid"][3, 0] = False
    cell["block_valid"][3, TOP_K - 1] = True
    with pytest.raises(SystemExit):
        harness.check_identity(cell, sample_rows=8, strict=True)


def test_identity_catches_a_wrong_valid_count():
    cell = _cell(TOTAL - 32, 32)
    cell["block_valid"][5, :] = False
    result = harness.check_identity(cell, sample_rows=8, strict=False)
    assert any(line.startswith("A1") for line in result["failures"])


def test_identity_catches_unsorted_block_ids():
    cell = _cell(TOTAL - 32, 32)
    cell["block_ids"][7, 0], cell["block_ids"][7, 1] = (
        cell["block_ids"][7, 1],
        cell["block_ids"][7, 0],
    )
    result = harness.check_identity(cell, sample_rows=8, strict=False)
    assert any(line.startswith("A3") for line in result["failures"])


def test_identity_catches_an_out_of_range_block_id():
    cell = _cell(TOTAL - 32, 32)
    row = 9
    n = int(cell["block_valid"][row].sum())
    cell["block_ids"][row, n - 1] = (TOTAL // RATIO) + 1
    result = harness.check_identity(cell, sample_rows=8, strict=False)
    assert any(line.startswith("A4") for line in result["failures"])


@pytest.mark.parametrize("pos", [2051, 2052, 4095, 4096, 16_383])
def test_kernel_and_lane_token_lists_agree(pos):
    """The two expansions of one row, materialised and compared in full."""

    rng = np.random.default_rng(pos)
    ids, ok = _selector_output(pos, 1, rng)
    lane = np.unique(harness._lane_tokens(ids[0], ok[0], pos))
    kern = np.unique(harness._kernel_tokens(ids[0], pos))
    assert np.array_equal(lane, kern)
    complete = (pos + 1) // RATIO
    expected = min(TOP_K, complete) * RATIO + (pos + 1 - complete * RATIO)
    assert lane.size == expected


def test_kernel_tokens_never_look_past_the_query_position():
    pos = 3000
    rng = np.random.default_rng(1)
    ids, _ = _selector_output(pos, 1, rng)
    kern = harness._kernel_tokens(ids[0], pos)
    assert kern.max() <= pos
    assert kern.min() >= 0


def test_kernel_tail_width_is_the_partial_block():
    """Zero to three tail tokens, generated in-kernel, never more."""

    rng = np.random.default_rng(3)
    for pos in range(4096, 4104):
        ids, _ = _selector_output(pos, 1, rng)
        complete = (pos + 1) // RATIO
        tail = [t for t in harness._kernel_tokens(ids[0], pos) if t >= complete * RATIO]
        assert len(tail) == (pos + 1) % RATIO
        assert len(tail) <= RATIO - 1


def test_parity_windows_cover_the_first_and_last_rows():
    windows = harness.parity_windows(4096, 16, 4)
    assert windows[0] == (0, 16)
    assert windows[-1] == (4080, 4096)
    assert all(b - a == 16 for a, b in windows)
    assert len(windows) == len(set(windows))


def test_parity_windows_degrade_to_the_whole_row_set():
    assert harness.parity_windows(4, 16, 4) == [(0, 4)]
    assert harness.parity_windows(1, 16, 4) == [(0, 1)]


def test_bf16_ulp_tracks_the_binade():
    values = np.array([1.0, 1.5, 2.0, 3.9, 0.25], dtype=np.float32)
    ulp = harness.bf16_ulp(values)
    assert np.allclose(ulp, np.array([2.0 ** -8, 2.0 ** -8, 2.0 ** -7,
                                      2.0 ** -7, 2.0 ** -10]))


def test_bf16_ulp_handles_zero_without_dividing_by_it():
    ulp = harness.bf16_ulp(np.zeros(4, dtype=np.float32))
    assert np.all(np.isfinite(ulp))
    assert np.all(ulp > 0)


def test_transient_estimate_grows_with_context():
    small = harness.transient_gb({"rows": 4096, "total": 16_384}, 512)
    large = harness.transient_gb({"rows": 4096, "total": 65_536}, 512)
    assert 0 < small < large
    assert large < 12.0  # the default --max-transient-gb budget


def test_every_declared_cell_has_a_causal_suffix():
    for name, (rows, total) in harness.CELLS.items():
        assert rows <= total, name
        assert (total - rows) >= 0, name
        assert total // RATIO > TOP_K, f"{name} never crosses the sparse boundary"
