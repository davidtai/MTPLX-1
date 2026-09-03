"""CPU-only gates for the split-K QSA sparse-GQA DECODE lane.

Nothing here dispatches Metal.  What is covered:

1. The split geometry -- the host arithmetic that sizes the grid and the
   partial buffer.  Every split must have work, and the splits must cover the
   tile range exactly once.
2. The VISIBLE SET.  This is the load-bearing correctness property: the
   kernel must attend exactly the keys the shipped rows-gather lane attends.
   The kernel's per-slot model and a transcription of the shipped
   ``qsa_m4_row_tokens`` closed form are compared over the interesting
   positions, and the two WRONG predicates a reader might reach for (the
   prefill kernel's leading-prefix cut, and ``candidate <= q_abs``) are shown
   to disagree, so a future edit that adopts either fails here.
3. The kernel SOURCE structure -- that the shipped predicate is the one in
   the MSL, that the normalisation really moved to the merge pass, and that
   the C++ encoder builds the kernel names the metallib instantiates.
4. The parameter-block layout the host and the device both static_assert.
5. The flag parsing.

Numeric Metal parity belongs to the operator-controlled guarded window; see
``scripts/fable/micro_qsa_sparse_decode.py`` for the command.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from mtplx.kernels import qsa_sparse_decode as lane
from mtplx.native import (
    qsa_sparse_gqa_decode_partial_shape,
    qsa_sparse_gqa_decode_split_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "native_extensions" / "qsa_sparse_gqa" / "sparse_gqa"
KERNEL_H = EXT / "steel_qsa_sparse_gqa_decode.h"
PARAMS_H = EXT / "qsa_sparse_gqa_decode_params.h"
DECODE_CPP = EXT / "qsa_sparse_gqa_decode.cpp"
METAL = EXT / "qsa_sparse_gqa.metal"

TILES = ((128, 32), (256, 32), (64, 64), (128, 64))
RATIO = lane.COMPRESS_RATIO
TOP_K = lane.TOP_K


# ---------------------------------------------------------------------------
# 1. split geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_tile", [t[0] for t in TILES])
@pytest.mark.parametrize("key_splits", [1, 2, 3, 4, 6, 8, 12, 16, 17, 33, 64])
def test_every_split_has_work(key_tile, key_splits):
    n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, key_tile, key_splits
    )
    assert n_tiles == -(-lane.SELECTED_TOKENS // key_tile)
    assert 1 <= n_splits <= n_tiles
    assert per_split >= 1
    # The last split starts inside the tile range -- no empty threadgroup.
    assert (n_splits - 1) * per_split < n_tiles
    # And the splits reach the end of it.
    assert n_splits * per_split >= n_tiles


@pytest.mark.parametrize("key_tile", [t[0] for t in TILES])
@pytest.mark.parametrize("key_splits", [1, 2, 5, 8, 16, 64])
def test_splits_partition_the_tiles_exactly_once(key_tile, key_splits):
    n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, key_tile, key_splits
    )
    covered = []
    for split in range(n_splits):
        t0 = split * per_split
        t1 = min(n_tiles, t0 + per_split)
        assert t0 < t1, "an empty split would write a dead partial state"
        covered.extend(range(t0, t1))
    assert covered == list(range(n_tiles))


def test_split_count_never_exceeds_the_tile_count():
    # 17 tiles at BK=128; asking for 64 splits must not dispatch 64.
    n_tiles, _, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, 128, 64
    )
    assert n_tiles == 17
    assert n_splits == 17


def test_partial_shape_tracks_the_split_count():
    for rows in (1, 4):
        for key_tile, _ in TILES:
            _, _, n_splits = qsa_sparse_gqa_decode_split_geometry(
                lane.SELECTED_TOKENS, key_tile, 8
            )
            shape = qsa_sparse_gqa_decode_partial_shape(rows, key_tile, 8)
            assert shape == (n_splits, lane.Q_HEADS, rows, lane.HEAD_DIM + 2)


def test_partial_buffer_is_small_next_to_what_it_replaces():
    """The split-K partial state must not cost more than the gather it kills.

    The shipped lane writes a [1, 2, rows, 2052, 256] bf16 K/V pair per
    layer; this buffer is the whole extra memory traffic the split-K
    arrangement introduces, and it has to stay an order of magnitude below.
    """

    rows = 4
    shape = qsa_sparse_gqa_decode_partial_shape(rows, 128, 8)
    partial_bytes = shape[0] * shape[1] * shape[2] * shape[3] * 4
    gathered_bytes = 2 * (1 * 2 * rows * (TOP_K * RATIO + RATIO) * 256 * 2)
    assert partial_bytes * 8 < gathered_bytes


def test_split_geometry_rejects_nonsense():
    with pytest.raises(ValueError):
        qsa_sparse_gqa_decode_split_geometry(0, 128, 8)
    with pytest.raises(ValueError):
        qsa_sparse_gqa_decode_split_geometry(2051, 0, 8)


# ---------------------------------------------------------------------------
# 2. the visible set
# ---------------------------------------------------------------------------
def _ids_for(q_abs: int, nb_total: int, seed: int = 7) -> list:
    """512 distinct block ids in an order that is deliberately NOT sorted."""

    state = seed
    pool = list(range(nb_total))
    for i in range(len(pool) - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:TOP_K]


POSITIONS = [
    2048, 2049, 2050, 2051, 2052, 2053,
    4093, 4094, 4095, 4096,
    16_383, 16_384, 16_385, 16_386,
    17_407,
]


@pytest.mark.parametrize("q_abs", POSITIONS)
def test_kernel_and_shipped_lane_attend_the_same_keys(q_abs):
    nb_total = (q_abs + 1) // RATIO + 1
    ids = _ids_for(q_abs, max(nb_total, TOP_K + 1))
    assert lane.visible_sets_agree(ids, q_abs, key_length=q_abs + 1)


@pytest.mark.parametrize("q_abs", POSITIONS)
def test_the_dropped_2052nd_slot_is_always_invalid(q_abs):
    """The kernel walks 2,051 slots; the shipped lane builds 2,052.

    The extra one is the tail's fourth, token ``((pos+1)//4)*4 + 3``, which is
    greater than ``pos`` for every residue class.  If that ever stops being
    true the kernel is dropping a visible key.
    """

    ids = _ids_for(q_abs, (q_abs + 1) // RATIO + 1)
    _, ok = lane.shipped_row_tokens(ids, q_abs)
    assert len(ok) == TOP_K * RATIO + RATIO
    assert ok[-1] is False or ok[-1] == False  # noqa: E712 - mx bool_ compat


@pytest.mark.parametrize("q_abs", POSITIONS)
def test_no_key_past_the_query_position_is_ever_attended(q_abs):
    ids = _ids_for(q_abs, (q_abs + 1) // RATIO + 1)
    for pos in lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1):
        assert pos <= q_abs


def test_the_prefix_cut_the_prefill_kernel_uses_is_wrong_here():
    """The decode selector does not sort, so a leading-prefix cut is wrong.

    ``mtplx.native.qsa_sparse_gqa`` (the prefill entry point) may take the
    first ``min(512, complete_blocks)`` slots as the valid ones, because
    ``_select_eager`` sorts its top-k ascending.  ``_select_m4`` does not.
    This test builds a row where the two disagree, so an edit that adopts the
    prefix cut in the decode kernel fails right here rather than quietly
    changing which keys the model attends.
    """

    # complete_blocks 511 < 512, so a prefix cut really cuts: it keeps the
    # first 511 SLOTS instead of the visible BLOCKS.  Slot 511 holds a
    # perfectly visible block and the cut drops it.
    q_abs = 2043
    complete = lane.visible_block_count(q_abs)
    assert complete == 511
    ids = [511] + list(range(0, 511))
    assert len(ids) == TOP_K
    assert len(set(ids)) == TOP_K
    per_slot = sorted(
        p
        for p in lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1)
        if p >= 0
    )
    prefix_cut = sorted(
        block * RATIO + within
        for slot_block, block in enumerate(ids)
        if slot_block < min(TOP_K, complete)
        for within in range(RATIO)
        if block * RATIO + within <= q_abs
    )
    assert per_slot != prefix_cut
    # And the per-slot answer is the shipped lane's.
    idx, ok = lane.shipped_row_tokens(ids, q_abs)
    shipped = sorted(t for t, good in zip(idx, ok) if good)
    assert per_slot == shipped


def test_causal_only_predicate_would_double_count_the_tail():
    """``candidate <= q_abs`` is implied by the real predicate but not equal.

    For ``block == complete_blocks`` it admits tokens of the INCOMPLETE block,
    which the tail slots already contribute -- so the softmax denominator
    would count them twice.  Pinning this stops a "simplification".
    """

    q_abs = 2049                       # (2050)//4 == 512, tail is non-empty
    complete = lane.visible_block_count(q_abs)
    ids = [complete] + list(range(0, TOP_K - 1))
    real = lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1)
    lax = []
    for slot in range(TOP_K * RATIO):
        block = ids[slot // RATIO]
        candidate = block * RATIO + (slot % RATIO)
        lax.append(candidate if candidate <= q_abs else -1)
    for within in range(RATIO - 1):
        candidate = complete * RATIO + within
        lax.append(candidate if candidate <= q_abs else -1)
    real_hits = [p for p in real if p >= 0]
    lax_hits = [p for p in lax if p >= 0]
    assert len(real_hits) == len(set(real_hits)), "the real predicate is a set"
    assert len(lax_hits) > len(set(lax_hits)), "the lax one duplicates keys"


def test_a_row_always_has_at_least_one_visible_key():
    """The merge's zero-denominator branch must be unreachable under the gate.

    The gate needs ``total // 4 > 512``, so every query row has at least one
    complete block, and the selector's top-512 of a set whose finite entries
    all outrank the -inf ones must contain at least one of them.
    """

    for q_abs in (2048, 2051, 4096, 17_407):
        complete = lane.visible_block_count(q_abs)
        assert complete >= 1
        ids = _ids_for(q_abs, complete + 1)
        hits = [
            p
            for p in lane.kernel_row_tokens(ids, q_abs, key_length=q_abs + 1)
            if p >= 0
        ]
        assert hits


def test_key_length_clamps_every_read():
    """kL is a memory-safety bound: nothing may index past it."""

    q_abs = 4096
    ids = _ids_for(q_abs, (q_abs + 1) // RATIO + 1)
    key_length = 2048  # deliberately shorter than the positions imply
    for pos in lane.kernel_row_tokens(ids, q_abs, key_length=key_length):
        assert pos < key_length


# ---------------------------------------------------------------------------
# 3. kernel source structure
# ---------------------------------------------------------------------------
def test_the_kernel_uses_the_per_slot_predicate():
    src = KERNEL_H.read_text()
    assert "raw_block < long(complete_blocks)" in src
    assert "complete_blocks = (q_abs + 1) / kCompressRatio" in src
    # The prefix cut the prefill kernel uses must NOT appear here.
    assert "valid_blocks" not in src


def test_the_normalisation_moved_to_the_merge_pass():
    src = KERNEL_H.read_text()
    split = src.split("mtplx_qsa_sparse_gqa_decode_merge")[0]
    assert "MtplxQsaDecDivOp" not in split
    assert "row_bin_op<MtplxQsaDecMulOp>" in split
    assert "prow[D] = max_score[0]" in split
    assert "prow[D + 1] = sum_score[0]" in split
    merge = src.split("mtplx_qsa_sparse_gqa_decode_merge")[1]
    assert "denom" in merge and "AccumType(1) / denom" in merge


def test_the_split_pass_walks_only_its_own_tile_range():
    src = KERNEL_H.read_text()
    assert "const int t0 = split * params->tiles_per_split;" in src
    assert "for (int ktile = t0; ktile < t1; ++ktile)" in src


def test_the_split_pass_reads_the_offset_from_a_device_buffer():
    src = KERNEL_H.read_text()
    assert "const device int* QOffset [[buffer(4)]]" in src
    assert "const int q_abs = QOffset[0] + q_pos;" in src


def test_the_online_softmax_initialises_to_finite_min_not_neg_inf():
    """finite_min, so an all-masked tile yields 0 rather than NaN."""

    src = KERNEL_H.read_text()
    assert "max_score[i] = Limits<AccumType>::finite_min;" in src
    assert "AccumType m = Limits<AccumType>::finite_min;" in src


def _expected_kernel_names() -> set:
    names = set()
    for tname in ("float16", "bfloat16"):
        for iname in ("uint32", "int32"):
            for bk, dc in TILES:
                names.add(
                    f"mtplx_qsa_sparse_gqa_decode_split_{tname}_{iname}"
                    f"_bk{bk}_dc{dc}_gqa12_hp16_d256_wm2"
                )
        names.add(f"mtplx_qsa_sparse_gqa_decode_merge_{tname}_d256")
    return names


def test_the_metallib_instantiates_every_name_the_encoder_can_ask_for():
    metal = METAL.read_text()
    # Expand the macro the same way the preprocessor will.
    instantiated = set()
    for tname in ("float16", "bfloat16"):
        for iname in ("uint32", "int32"):
            for bk, dc in TILES:
                instantiated.add(
                    f"mtplx_qsa_sparse_gqa_decode_split_{tname}_{iname}"
                    f"_bk{bk}_dc{dc}_gqa12_hp16_d256_wm2"
                )
        instantiated.add(f"mtplx_qsa_sparse_gqa_decode_merge_{tname}_d256")
    assert instantiated == _expected_kernel_names()
    # And the source really contains the macro pieces that build them.
    assert '"mtplx_qsa_sparse_gqa_decode_split_" #tname "_" #iname' in metal
    assert '"_bk" #bk "_dc" #dc "_gqa12_hp16_d256_wm2"' in metal
    assert '"mtplx_qsa_sparse_gqa_decode_merge_" #tname "_d256"' in metal
    for bk, dc in TILES:
        assert f"iname, itype, {bk}, {dc})" in metal


def test_the_encoder_builds_the_same_name_the_metallib_declares():
    cpp = DECODE_CPP.read_text()
    assert '"mtplx_qsa_sparse_gqa_decode_split_"' in cpp
    assert '"_bk", key_tile_' in cpp
    assert '"_dc", dimension_tile_' in cpp
    assert '"_gqa", kGqa, "_hp", kHeadPad, "_d",' in cpp
    assert '"mtplx_qsa_sparse_gqa_decode_merge_"' in cpp
    # The compiled-in geometry constants must match the macro's literals.
    assert re.search(r"constexpr int kGqa = 12;", cpp)
    assert re.search(r"constexpr int kHeadPad = 16;", cpp)
    assert re.search(r"constexpr int kHeadDim = 256;", cpp)
    assert re.search(r"constexpr int kWarps = 2;", cpp)


def test_the_encoder_dispatches_the_split_grid_on_the_z_axis():
    cpp = DECODE_CPP.read_text()
    assert "MTL::Size(rows, kKvHeads, n_splits)" in cpp
    assert "MTL::Size(32, kWarps, 1)" in cpp
    assert "MTL::Size(kQHeads * rows, 1, 1)" in cpp
    assert "MTL::Size(kHeadDim, 1, 1)" in cpp


def test_the_selected_token_width_agrees_across_the_three_definitions():
    cpp = DECODE_CPP.read_text()
    assert (
        "constexpr int kSelectedTokens = kTopK * kCompressRatio "
        "+ (kCompressRatio - 1);" in cpp
    )
    assert lane.SELECTED_TOKENS == TOP_K * RATIO + (RATIO - 1) == 2051
    src = KERNEL_H.read_text()
    assert "constexpr int kTail = kCompressRatio - 1;" in src
    assert (
        "const int selected_tokens = params->topk * kCompressRatio + kTail;"
        in src
    )


# ---------------------------------------------------------------------------
# 4. parameter-block layout
# ---------------------------------------------------------------------------
def _static_assert_size(header: str, struct: str) -> int:
    match = re.search(
        rf"static_assert\(sizeof\({struct}\) == (\d+)", header
    )
    assert match, f"no sizeof static_assert for {struct}"
    return int(match.group(1))


def test_decode_params_layout_is_pinned_and_arithmetically_right():
    header = PARAMS_H.read_text()
    # 11 ints + 1 float + 1 pad int, then 4 x 3 int64 at 8-byte alignment.
    ints = len(re.findall(r"^  int \w+;", header, flags=re.M))
    body = header.split("struct MtplxQsaSparseGqaDecodeParams")[1].split("};")[0]
    n_int = len(re.findall(r"\bint \w+;", body))
    n_float = len(re.findall(r"\bfloat \w+;", body))
    n_i64_arrays = len(re.findall(r"int64_t \w+\[3\];", body))
    assert (n_int, n_float, n_i64_arrays) == (11, 1, 4)
    scalar = n_int * 4 + n_float * 4
    aligned = -(-scalar // 8) * 8
    assert _static_assert_size(header, "MtplxQsaSparseGqaDecodeParams") == (
        aligned + n_i64_arrays * 24
    )
    assert ints >= 11


def test_merge_params_layout_is_pinned_and_arithmetically_right():
    header = PARAMS_H.read_text()
    body = header.split("struct MtplxQsaSparseGqaMergeParams")[1].split("};")[0]
    n_int = len(re.findall(r"\bint \w+;", body))
    n_i64_arrays = len(re.findall(r"int64_t \w+\[3\];", body))
    assert (n_int, n_i64_arrays) == (6, 1)
    scalar = n_int * 4
    aligned = -(-scalar // 8) * 8
    assert _static_assert_size(header, "MtplxQsaSparseGqaMergeParams") == (
        aligned + n_i64_arrays * 24
    )


def test_both_params_blocks_are_included_by_both_sides():
    assert '#include "sparse_gqa/qsa_sparse_gqa_decode_params.h"' in (
        KERNEL_H.read_text()
    )
    assert '#include "sparse_gqa/qsa_sparse_gqa_decode_params.h"' in (
        DECODE_CPP.read_text()
    )


def test_the_build_compiles_the_new_sources():
    cmake = (
        ROOT / "native_extensions" / "qsa_sparse_gqa" / "CMakeLists.txt"
    ).read_text()
    assert "sparse_gqa/qsa_sparse_gqa_decode.cpp" in cmake
    assert "sparse_gqa/steel_qsa_sparse_gqa_decode.h" in cmake
    assert "sparse_gqa/qsa_sparse_gqa_decode_params.h" in cmake


# ---------------------------------------------------------------------------
# 5. flags
# ---------------------------------------------------------------------------
def test_tile_flag_accepts_only_instantiated_tiles():
    from mtplx.runtime_options import _parse_sparse_decode_tile

    assert _parse_sparse_decode_tile(None) == (128, 32)
    assert _parse_sparse_decode_tile("") == (128, 32)
    for bk, dc in TILES:
        assert _parse_sparse_decode_tile(f"{bk}:{dc}") == (bk, dc)
    for bad in ("128", "128:33", "127:32", "128:32:1", "x:y"):
        with pytest.raises(ValueError):
            _parse_sparse_decode_tile(bad)


def test_splits_flag_is_bounded():
    from mtplx.runtime_options import (
        FABLE_QSA_SPARSE_DECODE_DEFAULT_SPLITS as default,
        _parse_sparse_decode_splits,
    )

    assert _parse_sparse_decode_splits(None) == default == 17
    assert _parse_sparse_decode_splits("17") == 17
    for bad in ("0", "65", "-1", "eight"):
        with pytest.raises(ValueError):
            _parse_sparse_decode_splits(bad)


def test_flags_are_off_by_default():
    from mtplx.runtime_options import env_bool

    assert env_bool("MTPLX_FABLE_QSA_SPARSE_DECODE", default=False, env={}) is False
    assert env_bool("MTPLX_FABLE_QSA_SPARSE_DRAFT", default=False, env={}) is False


def test_engagement_reports_a_pending_install_as_not_installed():
    lane.reset_for_tests()
    report = lane.engagement()
    assert report["installed"] is False
    assert report["disabled_reason"] is None
    assert report["verify_kernel"] == 0
    assert report["draft_kernel"] == 0


def test_parity_thresholds_are_stated_not_implicit():
    assert lane.PARITY_FP32_MAX_ABS_ULPS == 2.0
    assert lane.PARITY_FP32_MAX_REL_L2 == 5.0e-4
    assert lane.PARITY_SHIPPED_MAX_REL_L2 == 5.0e-2
    assert lane.PARITY_MIN_TOP1 == 0.98


# ---------------------------------------------------------------------------
# 7. what the 2026-09-02 runs settled
# ---------------------------------------------------------------------------
def test_the_tight_gate_is_tighter_than_the_measured_shipped_delta():
    """The fp32 gate must be able to FAIL if the attribution is wrong.

    Every configuration measured rel_l2 4.78e-3 against the shipped path.  If
    that delta really is the shipped path's own bf16 score and probability
    casts, the same comparison against the fp32 reference collapses to output
    rounding.  A gate set above 4.78e-3 could not tell those apart, so it
    would certify nothing.
    """

    measured_vs_shipped = 4.78e-3
    assert lane.PARITY_FP32_MAX_REL_L2 < measured_vs_shipped
    # ... while the loose bar stays an order of magnitude above it, because it
    # bounds the reference's quantisation, not the kernel's error.
    assert lane.PARITY_SHIPPED_MAX_REL_L2 > 10 * measured_vs_shipped


def test_the_reference_ladder_has_three_distinct_rungs():
    import inspect

    src = inspect.getsource(lane.reference_attention)
    # The shipped rung must keep BOTH bf16 roundings...
    assert "probs = probs.astype(queries.dtype)" in src
    # ...and the fp32 rung must remove them by UPCASTING (exact), never by
    # changing the operands.
    assert "q_view = q_view.astype(mx.float32)" in src
    assert "v_view = v_view.astype(mx.float32)" in src
    for fn, scores, probs in (
        (lane.stock_reference, False, False),
        (lane.shipped_fp32_probs_reference, False, True),
        (lane.fp32_reference, True, True),
    ):
        wrapped = inspect.getsource(fn)
        assert f"fp32_scores={scores}" in wrapped
        assert f"fp32_probs={probs}" in wrapped


def test_every_reference_rung_returns_the_query_dtype():
    """All three must be comparable element for element against bf16 output."""

    import inspect

    src = inspect.getsource(lane.reference_attention)
    assert src.rstrip().endswith(".astype(queries.dtype)")


def test_the_default_split_target_reaches_one_tile_per_threadgroup():
    """17 is the tile count at BK=128, so it is the knob's saturation point."""

    from mtplx.runtime_options import (
        FABLE_QSA_SPARSE_DECODE_DEFAULT_SPLITS as default,
    )

    n_tiles, per_split, n_splits = qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, 128, default
    )
    assert (n_tiles, per_split, n_splits) == (17, 1, 17)
    assert lane.VERIFY_ROWS * lane.KV_HEADS * n_splits == 136


@pytest.mark.parametrize("larger", [18, 32, 33, 64])
def test_split_targets_above_the_tile_count_are_the_same_configuration(larger):
    """Why the first sweep's s17 and s32 rows are one config measured twice.

    Their 5.3% spread is therefore the bench's noise floor, not a result, and
    nothing may be called a winner on a margin under it.
    """

    at_17 = qsa_sparse_gqa_decode_split_geometry(lane.SELECTED_TOKENS, 128, 17)
    assert qsa_sparse_gqa_decode_split_geometry(
        lane.SELECTED_TOKENS, 128, larger
    ) == at_17


def test_bk64_needs_33_splits_to_saturate_and_that_is_in_the_sweep():
    """BK=64 has 33 tiles, so 17 and 32 both clamp to 17 -- 33 does not."""

    at_32 = qsa_sparse_gqa_decode_split_geometry(lane.SELECTED_TOKENS, 64, 32)
    at_33 = qsa_sparse_gqa_decode_split_geometry(lane.SELECTED_TOKENS, 64, 33)
    assert at_32[2] == 17 and at_33[2] == 33
    assert lane.VERIFY_ROWS * lane.KV_HEADS * at_33[2] == 264
    harness = (ROOT / "scripts" / "fable" / "micro_qsa_sparse_decode.py").read_text()
    assert "SPLIT_TARGETS = (4, 8, 16, 17, 32, 33, 64)" in harness


def test_the_lane_and_the_native_wrapper_agree_on_the_default():
    from mtplx import native
    from mtplx.runtime_options import (
        FABLE_QSA_SPARSE_DECODE_DEFAULT_SPLITS as default,
    )

    assert native._DEFAULT_KEY_SPLITS == default
    assert native._DEFAULT_TILE == (128, 32)


def test_the_micro_refuses_the_production_gather_arm_off_the_m4_geometry():
    """kernels/qwen4_qsa_m4_fused_kv_gather.py compiles _ROWS = 4.

    Handing it a 1-row cell made it emit 4 rows anyway, and the crash landed
    three arms later at the reshape, after the M=4 results were already in.
    """

    gather = (
        ROOT / "mtplx" / "kernels" / "qwen4_qsa_m4_fused_kv_gather.py"
    ).read_text()
    assert "_ROWS = 4" in gather
    harness = (ROOT / "scripts" / "fable" / "micro_qsa_sparse_decode.py").read_text()
    assert "if rows != lane.VERIFY_ROWS:" in harness
    # and the refusal must come BEFORE the reshape that used to raise
    body = harness.split("def gather_kernel_arm")[1]
    assert body.index("if rows != lane.VERIFY_ROWS:") < body.index("swapaxes(-1, -2)")


def test_the_m1_cell_is_opt_in():
    harness = (ROOT / "scripts" / "fable" / "micro_qsa_sparse_decode.py").read_text()
    assert '"--include-m1"' in harness
    assert 'cells = [("verify-m4-16k", 4, 16_380, CAPACITY)]' in harness
    assert "if args.include_m1:" in harness


# ---------------------------------------------------------------------------
# 6. the nanobind ABI guard
#
# W68's first guarded run built the extension cleanly and then failed at the
# first call with "incompatible function arguments ... queries:
# mlx::core::array" while invoking with mlx.core.array.  The cause was not the
# stream kwarg: the extension was built against nanobind internals v19 while
# mlx.core uses v21, so the two got separate __nb_internals_<tag>_mlx__
# capsules and no array could ever be cast.  These pin the detector.
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402


def _load_abi_checker():
    path = ROOT / "scripts" / "fable" / "check_native_qsa_abi.py"
    spec = importlib.util.spec_from_file_location("check_native_qsa_abi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


abi = _load_abi_checker()


def test_internals_version_is_read_from_nb_abi_h(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nb_abi.h").write_text(
        "#ifndef NB_INTERNALS_VERSION\n#  define NB_INTERNALS_VERSION 21\n#endif\n"
    )
    assert abi.internals_version_of_nanobind(tmp_path) == 21


def test_internals_version_falls_back_to_nb_internals_h(tmp_path):
    """Older nanobind releases keep the macro in the other header."""

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nb_internals.h").write_text(
        "#define NB_INTERNALS_VERSION 17\n"
    )
    assert abi.internals_version_of_nanobind(tmp_path) == 17


def test_missing_nanobind_tree_reports_none(tmp_path):
    assert abi.internals_version_of_nanobind(tmp_path) is None


def test_abi_tag_is_extracted_from_binary_bytes(tmp_path):
    blob = tmp_path / "fake.so"
    blob.write_bytes(b"\x00junk\x00v21_system_libcpp_abi1\x00more\x00")
    assert abi.abi_tags_in_binary(blob) == {"v21_system_libcpp_abi1"}
    assert abi.abi_version_of_binary(blob) == 21


def test_two_different_tags_refuse_to_guess(tmp_path):
    blob = tmp_path / "fake.so"
    blob.write_bytes(b"v19_system_libcpp_abi1\x00v21_system_libcpp_abi1\x00")
    assert abi.abi_version_of_binary(blob) is None


def test_a_binary_with_no_tag_reports_none(tmp_path):
    blob = tmp_path / "fake.so"
    blob.write_bytes(b"nothing to see here")
    assert abi.abi_tags_in_binary(blob) == set()
    assert abi.abi_version_of_binary(blob) is None


def test_the_native_wrapper_uses_the_same_tag_regex():
    """One definition of "what a nanobind ABI tag looks like", two readers."""

    from mtplx import native

    blob = b"\x00v21_system_libcpp_abi1\x00"
    assert [m.group(1) for m in native._NB_ABI_TAG_RE.finditer(blob)] == [b"21"]


def test_the_native_wrapper_reads_a_versions_from_a_binary(tmp_path):
    from mtplx import native

    blob = tmp_path / "fake.so"
    blob.write_bytes(b"v19_system_libcpp_abi1\x00")
    assert native._nanobind_internals_version(blob) == 19
    blob.write_bytes(b"nothing")
    assert native._nanobind_internals_version(blob) is None


def test_the_cmake_guard_is_a_fatal_error_not_a_warning():
    cmake = (
        ROOT / "native_extensions" / "qsa_sparse_gqa" / "CMakeLists.txt"
    ).read_text()
    assert "MTPLX_NANOBIND_DIR" in cmake
    assert "NB_INTERNALS_VERSION" in cmake
    guard = cmake.split("if(MTPLX_NB_INTERNALS AND MTPLX_MLX_NB_INTERNALS)")[1]
    assert "FATAL_ERROR" in guard
    assert "check_native_qsa_abi.py" in guard
    # The one wrong fix must stay called out where someone would reach for it.
    assert "NEVER \"fix\" a mismatch by defining NB_INTERNALS_VERSION" in cmake


def test_the_wrappers_omit_a_none_stream():
    """Passing an explicit None was not the bug, but it is one fewer variable."""

    source = (ROOT / "mtplx" / "native" / "__init__.py").read_text()
    assert source.count("if stream is None:") == 2
    assert "stream=stream)" in source
    assert "stream=stream,\n    )" not in source


# ---------------------------------------------------------------------------
# 8. the micro's parity-gate ladder
#
# The two-gate ladder renamed PARITY_MAX_ABS_ULPS -> PARITY_FP32_MAX_ABS_ULPS
# (and PARITY_MAX_REL_L2 -> PARITY_FP32_MAX_REL_L2), and
# scripts/fable/micro_qsa_sparse_decode.py kept reading the old names in its
# report header -- so it crashed with AttributeError at startup, INSIDE a
# guarded window, after the operator had already taken the box.  These pin the
# gate-building surface so that class of drift fails here instead.
#
# Nothing below dispatches Metal: the gate builders are fed stubbed numbers,
# and the one call to the micro's own ``compare`` runs on the CPU stream.
# ---------------------------------------------------------------------------
MICRO = ROOT / "scripts" / "fable" / "micro_qsa_sparse_decode.py"


def _load_micro():
    spec = importlib.util.spec_from_file_location("micro_qsa_sparse_decode", MICRO)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


micro = _load_micro()


#: Comfortably inside every threshold, so a test can break exactly one.
PASSING_STATS = {
    "vs_fp32": {"max_abs_bf16_ulps": 1.0, "rel_l2": 1.0e-4, "top1": 1.0},
    "vs_shipped": {"max_abs_bf16_ulps": 9.0, "rel_l2": 4.78e-3, "top1": 1.0},
}


def _stats(reference=None, key=None, value=None):
    """The passing stats, optionally with ONE statistic pushed out of bounds."""

    out = {k: dict(v) for k, v in PASSING_STATS.items()}
    if reference is not None:
        out[reference][key] = value
    return out


def test_the_micro_no_longer_reads_the_pre_ladder_constant_names():
    """The exact crash: a name the lane module no longer defines."""

    source = MICRO.read_text()
    for gone in ("PARITY_MAX_ABS_ULPS", "PARITY_MAX_REL_L2"):
        assert not re.search(rf"lane\.{gone}\b", source)
        assert not hasattr(lane, gone)


def test_every_gate_names_a_threshold_the_lane_module_actually_defines():
    """Read by name at call time -- never copied, so a rename cannot drift."""

    for spec in micro.GATE_SPECS:
        assert hasattr(lane, spec.threshold_name), spec.threshold_name
        assert micro.gate_threshold(spec) == float(
            getattr(lane, spec.threshold_name)
        )


def test_the_micro_gates_are_exactly_the_checks_install_runs():
    """A rung added to the lane's ladder must not silently go unreported."""

    import inspect

    used = set(re.findall(r"PARITY_[A-Z0-9_]+", inspect.getsource(lane.install)))
    assert used == {spec.threshold_name for spec in micro.GATE_SPECS}
    # ...and both references are represented, tight and loose.
    assert {spec.reference for spec in micro.GATE_SPECS} == {
        "vs_fp32",
        "vs_shipped",
    }


def test_the_ladder_summary_states_both_rungs_before_any_measurement():
    ladder = micro.parity_gate_ladder()
    assert json.loads(json.dumps(ladder)) == ladder
    assert set(ladder["gates"]) == {spec.name for spec in micro.GATE_SPECS}
    for spec in micro.GATE_SPECS:
        row = ladder["gates"][spec.name]
        assert row["threshold_name"] == spec.threshold_name
        assert row["threshold"] == float(getattr(lane, spec.threshold_name))
        assert row["comparison"] in ("<=", ">=")
        assert row["rung"] in ("tight", "loose")
    assert ladder["gates"]["fp32_rel_l2"]["rung"] == "tight"
    assert ladder["gates"]["shipped_rel_l2"]["rung"] == "loose"


def test_each_gate_reports_threshold_observed_and_pass():
    stats = _stats()
    gates = micro.parity_gates(stats["vs_fp32"], stats["vs_shipped"])
    assert set(gates) == {spec.name for spec in micro.GATE_SPECS}
    for name, gate in gates.items():
        assert set(gate) >= {
            "threshold_name",
            "threshold",
            "comparison",
            "observed",
            "pass",
            "reference",
            "rung",
            "statistic",
        }, name
        assert gate["pass"] is True
    assert micro.gate_verdict(gates) is True
    assert micro.gate_failures(gates) == []


@pytest.mark.parametrize(
    "gate_name,reference,key,bad",
    [
        ("fp32_max_abs_ulps", "vs_fp32", "max_abs_bf16_ulps", 2.5),
        ("fp32_rel_l2", "vs_fp32", "rel_l2", 6.0e-4),
        ("fp32_top1", "vs_fp32", "top1", 0.97),
        ("shipped_rel_l2", "vs_shipped", "rel_l2", 6.0e-2),
        ("shipped_top1", "vs_shipped", "top1", 0.97),
    ],
)
def test_each_gate_fails_on_its_own_statistic(gate_name, reference, key, bad):
    stats = _stats(reference, key, bad)
    gates = micro.parity_gates(stats["vs_fp32"], stats["vs_shipped"])
    failed = {n for n, g in gates.items() if not g["pass"]}
    # top-1 is shared by both rungs, so breaking one reference's top-1 must
    # break that rung's gate and ONLY that one.
    assert failed == {gate_name}, failed
    assert micro.gate_verdict(gates) is False
    lines = micro.gate_failures(gates)
    assert len(lines) == 1
    assert gates[gate_name]["threshold_name"] in lines[0]
    assert gates[gate_name]["observed"] == bad


def test_the_loose_rung_has_no_ulp_gate():
    """The shipped path's bf16 casts move elements by many ulp by construction."""

    stats = _stats("vs_shipped", "max_abs_bf16_ulps", 1.0e6)
    gates = micro.parity_gates(stats["vs_fp32"], stats["vs_shipped"])
    assert micro.gate_verdict(gates) is True


def test_the_ulp_column_is_read_under_either_spelling():
    """``lane._compare`` says max_abs_ulps; this script says max_abs_bf16_ulps."""

    vs_fp32 = {"max_abs_ulps": 1.0, "rel_l2": 1.0e-4, "top1": 1.0}
    gates = micro.parity_gates(vs_fp32, dict(PASSING_STATS["vs_shipped"]))
    assert gates["fp32_max_abs_ulps"]["observed"] == 1.0
    assert micro.gate_verdict(gates) is True


def test_a_missing_statistic_is_a_hard_error_not_a_skipped_gate():
    with pytest.raises(KeyError, match="fp32_max_abs_ulps"):
        micro.parity_gates(
            {"rel_l2": 1.0e-4, "top1": 1.0}, dict(PASSING_STATS["vs_shipped"])
        )


def test_the_micro_compare_emits_every_statistic_the_gates_read():
    """CPU stream only -- this is the key-name contract, not a numeric check."""

    import mlx.core as mx

    with mx.stream(mx.cpu):
        ref = mx.zeros((2, 4), dtype=mx.float32)
        got = mx.zeros((2, 4), dtype=mx.float32)
        stats = micro.compare(ref, got)
    for spec in micro.GATE_SPECS:
        assert any(k in stats for k in spec.statistics), spec.name
    gates = micro.parity_gates(stats, stats)
    assert micro.gate_verdict(gates) is True


# --- the verdict roll-up, which is what the process status is --------------
def _arm(passed: bool):
    """One arm's report body, as ``run_cell`` builds it."""

    stats = _stats() if passed else _stats("vs_fp32", "rel_l2", 1.0)
    gates = micro.parity_gates(stats["vs_fp32"], stats["vs_shipped"])
    return {
        "parity": {
            "gates": gates,
            "gate": micro.gate_verdict(gates),
            "gate_failures": micro.gate_failures(gates),
        }
    }


def test_a_cell_verdict_ignores_skipped_arms_but_not_failed_ones():
    arms = {
        "native_bk128_dc32_s17": _arm(True),
        "native_bk64_dc64_s33": _arm(False),
        "native_bk256_dc32_s4": {"skipped": "tile not instantiated"},
    }
    verdict = micro.cell_verdict(arms)
    assert verdict["arms_measured"] == [
        "native_bk128_dc32_s17",
        "native_bk64_dc64_s33",
    ]
    assert verdict["arms_failed"] == ["native_bk64_dc64_s33"]
    assert verdict["pass"] is False
    assert verdict["failures"]["native_bk64_dc64_s33"]


def test_exit_code_is_zero_only_when_every_measured_arm_held():
    cells = {"verify-m4-16k": {"parity_verdict": micro.cell_verdict({"a": _arm(True)})}}
    verdict = micro.roll_up_verdict(cells)
    assert verdict == {
        "arms_measured": ["verify-m4-16k/a"],
        "arms_failed": [],
        "pass": True,
    }
    assert micro.verdict_exit_code(verdict) == 0


def test_exit_code_is_one_when_a_gate_failed():
    cells = {
        "verify-m4-16k": {
            "parity_verdict": micro.cell_verdict({"a": _arm(True), "b": _arm(False)})
        }
    }
    verdict = micro.roll_up_verdict(cells)
    assert verdict["arms_failed"] == ["verify-m4-16k/b"]
    assert verdict["pass"] is False
    assert micro.verdict_exit_code(verdict) == 1


def test_exit_code_is_three_when_nothing_was_measured():
    """No arm ran, so the ladder returned no verdict -- that is not a pass."""

    cells = {
        "verify-m4-16k": {
            "parity_verdict": micro.cell_verdict({"a": {"skipped": "no kernel"}})
        }
    }
    verdict = micro.roll_up_verdict(cells)
    assert verdict["pass"] is False
    assert micro.verdict_exit_code(verdict) == 3


def test_main_returns_the_ladder_verdict_rather_than_a_constant_zero():
    import inspect

    src = inspect.getsource(micro.main)
    assert "verdict_exit_code(" in src
    assert "return status" in src
    assert "\n    return 0\n" not in src


def test_main_builds_its_report_header_and_writes_it(tmp_path, monkeypatch):
    """The exact crash site: ``main`` reading the ladder's thresholds.

    The AttributeError that broke W68 was raised while building
    ``report[...]``, AFTER both availability guards -- i.e. inside a guarded
    window, with the box already taken.  Everything that dispatches Metal is
    stubbed here; what runs is the header build, the roll-up and the status.
    """

    monkeypatch.setattr(micro, "native_qsa_available", lambda: True)
    monkeypatch.setattr(micro.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        micro,
        "run_cell",
        lambda name, rows, q_offset, total, args: {
            "arms": {"native_bk128_dc32_s17": _arm(True)},
            "parity_verdict": micro.cell_verdict(
                {"native_bk128_dc32_s17": _arm(True)}
            ),
        },
    )
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["micro", "--out", str(out), "--tiles", "128:32", "--splits", "17"]
    )
    assert micro.main() == 0

    report = json.loads(out.read_text())
    ladder = report["parity_ladder"]["gates"]
    assert ladder["fp32_max_abs_ulps"]["threshold_name"] == "PARITY_FP32_MAX_ABS_ULPS"
    assert ladder["fp32_max_abs_ulps"]["threshold"] == lane.PARITY_FP32_MAX_ABS_ULPS
    assert ladder["fp32_rel_l2"]["threshold"] == lane.PARITY_FP32_MAX_REL_L2
    assert ladder["shipped_rel_l2"]["threshold"] == lane.PARITY_SHIPPED_MAX_REL_L2
    assert ladder["shipped_top1"]["threshold"] == lane.PARITY_MIN_TOP1
    assert report["parity_verdict"]["pass"] is True
    # ...and the old single-gate block is gone, not merely shadowed.
    assert "parity_gates" not in report


def test_main_returns_one_when_the_stubbed_arm_misses_a_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(micro, "native_qsa_available", lambda: True)
    monkeypatch.setattr(micro.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        micro,
        "run_cell",
        lambda name, rows, q_offset, total, args: {
            "arms": {"native_bk128_dc32_s17": _arm(False)},
            "parity_verdict": micro.cell_verdict(
                {"native_bk128_dc32_s17": _arm(False)}
            ),
        },
    )
    monkeypatch.setattr(sys, "argv", ["micro", "--tiles", "128:32", "--splits", "17"])
    assert micro.main() == 1
