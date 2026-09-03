"""Prefill chunk geometry: width knob, memory refusal, query-tile arithmetic.

Pure Python / NumPy.  Nothing here imports ``mlx``: the two source files that
do are checked as text (the tripwire convention in
``tests/test_qsa_prefill_runtime_gates_static.py``), and the attention
equivalence the query tile claims is proved against a NumPy reference of the
exact expression MLX's unfused SDPA fallback evaluates.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mtplx.fable_prefill_chunk import (
    ALLOW_COMPILE_ROWS_MISMATCH_ENV,
    BUDGET_ENV,
    CHUNK_SIZE_DENSE_ENV,
    CHUNK_SIZE_ENV,
    CHUNK_SIZE_REPAGE_ENV,
    COHERENCE_COMPILED,
    COHERENCE_NARROW_EAGER,
    COMPILE_ROWS_ENV,
    DEFAULT_CHUNK_SIZE,
    GUARD_ENV,
    MARGIN_ENV,
    QUERY_TILE_ENV,
    RESIDENT_ENV,
    WIRED_LIMIT_ENV,
    PrefillChunkGeometryError,
    PrefillChunkMemoryError,
    assert_prefill_chunk_coherent,
    attention_row_context_products,
    configured_full_chunk_widths,
    guard_prefill_chunk_geometry,
    plan_prefill_chunk_memory,
    query_tile_spans,
    resolve_budget_bytes,
    resolve_query_tile_rows,
    summarize_spans,
)

ROOT = Path(__file__).resolve().parents[1]

#: The production cell: 16,384 prompt tokens, 8 x 2,048.
PROMPT_TOKENS = 16_384
#: mtplx.memory_plan's dense QSA prefill model at the shipped 2,048 width:
#: 12.75 B per (chunk row x context token) x 4 live layers.
TRANSIENT_PER_TOKEN = int(12.75 * DEFAULT_CHUNK_SIZE * 4)
#: scripts/fable/abba_driver.py WIRED_LIMIT_BYTES.
DRIVER_WIRED_LIMIT = 90 * 1024**3
#: .benchmark-artifacts/fable/*.json peak_memory_bytes, all 122 rows.
CENSUS_PEAK_BYTES = 87_393_815_544


# ---------------------------------------------------------------------------
# Flag-off identity
# ---------------------------------------------------------------------------


def test_query_tile_disabled_by_default():
    assert resolve_query_tile_rows({}) == 0
    assert resolve_query_tile_rows({QUERY_TILE_ENV: ""}) == 0
    assert resolve_query_tile_rows({QUERY_TILE_ENV: "0"}) == 0
    assert resolve_query_tile_rows({QUERY_TILE_ENV: "not-a-number"}) == 0
    assert resolve_query_tile_rows({QUERY_TILE_ENV: "2048"}) == 2048


def test_query_tile_spans_empty_unless_it_would_split():
    assert query_tile_spans(2048, context_before=0, tile=0) == []
    assert query_tile_spans(2048, context_before=0, tile=2048) == []
    assert query_tile_spans(2048, context_before=0, tile=4096) == []
    assert query_tile_spans(0, context_before=0, tile=8) == []


def test_guard_inert_without_a_budget():
    assert (
        guard_prefill_chunk_geometry(
            chunk_size=16_384,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=TRANSIENT_PER_TOKEN,
            environ={},
            resident_bytes=CENSUS_PEAK_BYTES,
        )
        is None
    )


def test_guard_inert_when_disabled():
    assert (
        guard_prefill_chunk_geometry(
            chunk_size=16_384,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=TRANSIENT_PER_TOKEN,
            environ={GUARD_ENV: "0", WIRED_LIMIT_ENV: str(DRIVER_WIRED_LIMIT)},
            resident_bytes=CENSUS_PEAK_BYTES,
        )
        is None
    )


def test_guard_inert_for_families_without_a_transient_model():
    assert (
        guard_prefill_chunk_geometry(
            chunk_size=16_384,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=0,
            environ={WIRED_LIMIT_ENV: str(DRIVER_WIRED_LIMIT)},
            resident_bytes=CENSUS_PEAK_BYTES,
        )
        is None
    )


def test_budget_prefers_explicit_over_wired_limit():
    assert resolve_budget_bytes({}) is None
    assert resolve_budget_bytes({WIRED_LIMIT_ENV: "17"}) == 17
    assert resolve_budget_bytes({WIRED_LIMIT_ENV: "17", BUDGET_ENV: "23"}) == 23
    assert resolve_budget_bytes({BUDGET_ENV: "nope"}) is None


# ---------------------------------------------------------------------------
# The memory model and the refusal
# ---------------------------------------------------------------------------


def test_transient_is_linear_in_the_live_query_rows():
    base = plan_prefill_chunk_memory(
        chunk_size=DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
    )
    wide = plan_prefill_chunk_memory(
        chunk_size=2 * DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
    )
    assert base.chunks == 8
    assert wide.chunks == 4
    assert wide.transient_bytes == 2 * base.transient_bytes
    # ~1.7 GiB at the shipped geometry, which is the order of the 1.28 GB the
    # window-19 sparse-lane arm actually removed from peak_memory_bytes.
    assert 1.5 * 1024**3 < base.transient_bytes < 2.0 * 1024**3


def test_query_tile_caps_the_transient_of_a_wide_chunk():
    tiled = plan_prefill_chunk_memory(
        chunk_size=2 * DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        query_tile=DEFAULT_CHUNK_SIZE,
    )
    shipped = plan_prefill_chunk_memory(
        chunk_size=DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
    )
    assert tiled.chunks == 4
    assert tiled.live_query_rows == DEFAULT_CHUNK_SIZE
    assert tiled.transient_bytes == shipped.transient_bytes


def test_production_geometry_fits_and_4096_still_fits():
    env = {WIRED_LIMIT_ENV: str(DRIVER_WIRED_LIMIT)}
    for chunk in (DEFAULT_CHUNK_SIZE, 2 * DEFAULT_CHUNK_SIZE):
        plan = guard_prefill_chunk_geometry(
            chunk_size=chunk,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=TRANSIENT_PER_TOKEN,
            environ=env,
            # Resident excludes the transient the plan is about to add.
            resident_bytes=CENSUS_PEAK_BYTES
            - int(12.75 * DEFAULT_CHUNK_SIZE * 4 * PROMPT_TOKENS),
        )
        assert plan is not None and plan.fits
        assert plan.headroom_bytes is not None and plan.headroom_bytes > 0


def test_single_shot_prefill_is_refused_and_the_message_names_the_knob():
    env = {WIRED_LIMIT_ENV: str(DRIVER_WIRED_LIMIT)}
    with pytest.raises(PrefillChunkMemoryError) as excinfo:
        guard_prefill_chunk_geometry(
            chunk_size=PROMPT_TOKENS,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=TRANSIENT_PER_TOKEN,
            environ=env,
            resident_bytes=CENSUS_PEAK_BYTES
            - int(12.75 * DEFAULT_CHUNK_SIZE * 4 * PROMPT_TOKENS),
        )
    message = str(excinfo.value)
    assert "MTPLX_PREFILL_CHUNK_SIZE" in message
    assert QUERY_TILE_ENV in message
    assert GUARD_ENV in message


def test_query_tile_rescues_a_geometry_the_guard_would_refuse():
    env = {
        WIRED_LIMIT_ENV: str(DRIVER_WIRED_LIMIT),
        QUERY_TILE_ENV: str(DEFAULT_CHUNK_SIZE),
    }
    resident = CENSUS_PEAK_BYTES - int(
        12.75 * DEFAULT_CHUNK_SIZE * 4 * PROMPT_TOKENS
    )
    plan = guard_prefill_chunk_geometry(
        chunk_size=PROMPT_TOKENS,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        environ=env,
        resident_bytes=resident,
    )
    assert plan is not None and plan.fits
    assert plan.live_query_rows == DEFAULT_CHUNK_SIZE


def test_guard_reads_resident_override_and_margin_from_env():
    env = {
        BUDGET_ENV: str(10 * 1024**3),
        RESIDENT_ENV: str(9 * 1024**3),
        MARGIN_ENV: "0",
    }
    with pytest.raises(PrefillChunkMemoryError):
        guard_prefill_chunk_geometry(
            chunk_size=DEFAULT_CHUNK_SIZE,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=TRANSIENT_PER_TOKEN,
            environ=env,
        )
    # Same geometry, a budget that covers it.
    env[BUDGET_ENV] = str(12 * 1024**3)
    plan = guard_prefill_chunk_geometry(
        chunk_size=DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        environ=env,
    )
    assert plan is not None and plan.fits


def test_guard_never_probes_the_allocator_when_resident_is_supplied():
    def explode() -> int:  # pragma: no cover - must not run
        raise AssertionError("resident probe called despite an explicit value")

    plan = guard_prefill_chunk_geometry(
        chunk_size=DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        environ={BUDGET_ENV: str(64 * 1024**3)},
        resident_bytes=0,
        resident_probe=explode,
    )
    assert plan is not None and plan.resident_bytes == 0


def test_plan_receipt_is_json_shaped():
    plan = plan_prefill_chunk_memory(
        chunk_size=DEFAULT_CHUNK_SIZE,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        budget_bytes=DRIVER_WIRED_LIMIT,
    )
    receipt = plan.as_receipt()
    assert receipt["chunks"] == 8
    assert receipt["chunk_size"] == DEFAULT_CHUNK_SIZE
    assert set(receipt) >= {"projected_peak_bytes", "headroom_bytes", "fits"}


# ---------------------------------------------------------------------------
# The QSA graph-bank coherence refusal
# ---------------------------------------------------------------------------


def test_shipped_width_is_coherent():
    assert (
        assert_prefill_chunk_coherent(DEFAULT_CHUNK_SIZE, {})
        == COHERENCE_COMPILED
    )
    assert (
        assert_prefill_chunk_coherent(
            DEFAULT_CHUNK_SIZE, {COMPILE_ROWS_ENV: str(DEFAULT_CHUNK_SIZE)}
        )
        == COHERENCE_COMPILED
    )


def test_widening_without_the_compile_rows_knob_is_refused():
    with pytest.raises(PrefillChunkGeometryError) as excinfo:
        assert_prefill_chunk_coherent(4096, {})
    assert COMPILE_ROWS_ENV in str(excinfo.value)


def test_widening_both_knobs_together_is_accepted():
    assert (
        assert_prefill_chunk_coherent(
            4096, {COMPILE_ROWS_ENV: "4096", CHUNK_SIZE_ENV: "4096"}
        )
        == COHERENCE_COMPILED
    )


def test_narrow_chunks_take_the_eager_selector_without_a_refusal():
    """The W57 bug: warm-up rungs and tails are NOT mis-paired serves.

    ``qwen4_exp._qsa_prefill_compile_rows`` gates the compiled QSA prefill
    on ``rows == compile_rows``, so a narrower chunk already falls back by
    design.  A 100-token prompt, a GDN-boundary tail grid, and the server's
    256-token warm-up ladder chunks are all in that class.
    """

    assert assert_prefill_chunk_coherent(100, {}) == COHERENCE_NARROW_EAGER
    assert (
        assert_prefill_chunk_coherent(256, {COMPILE_ROWS_ENV: "4096"})
        == COHERENCE_NARROW_EAGER
    )
    assert (
        assert_prefill_chunk_coherent(DEFAULT_CHUNK_SIZE, {})
        == COHERENCE_COMPILED
    )


def test_warmup_ladder_chunk_passes_under_pinned_compile_rows():
    """The exact server geometry that refused on 2026-09-01.

    ``MTPLX_QSA_PREFILL_COMPILE_ROWS=4096`` pinned for real prompts, the
    background warm-up ladder passing its 256-token chunk: both rungs died
    instantly on ``PrefillChunkGeometryError``.
    """

    environ = {
        COMPILE_ROWS_ENV: "4096",
        CHUNK_SIZE_ENV: "4096",
    }
    assert (
        assert_prefill_chunk_coherent(256, environ) == COHERENCE_NARROW_EAGER
    )
    assert assert_prefill_chunk_coherent(4096, environ) == COHERENCE_COMPILED


def test_full_width_narrower_than_the_compiled_rows_is_still_refused():
    """The mis-pairing the guard exists for, in its other direction.

    ``MTPLX_QSA_PREFILL_COMPILE_ROWS=4096`` with the width left at the
    shipped 2,048 is a FULL chunk the graph bank will not serve -- every
    chunk of every real prompt demoted to the eager selector.
    """

    with pytest.raises(PrefillChunkGeometryError) as excinfo:
        assert_prefill_chunk_coherent(
            DEFAULT_CHUNK_SIZE, {COMPILE_ROWS_ENV: "4096"}
        )
    assert COMPILE_ROWS_ENV in str(excinfo.value)
    # ... and explicitly, with the width knob spelled out.
    with pytest.raises(PrefillChunkGeometryError):
        assert_prefill_chunk_coherent(
            2048, {COMPILE_ROWS_ENV: "4096", CHUNK_SIZE_ENV: "2048"}
        )


def test_narrower_full_width_under_auto_is_refused_for_both_layouts():
    """``auto`` resolves per KV layout, so BOTH per-layout keys are full."""

    environ = {
        COMPILE_ROWS_ENV: "4096",
        CHUNK_SIZE_ENV: "auto",
        CHUNK_SIZE_DENSE_ENV: "2048",
        CHUNK_SIZE_REPAGE_ENV: "1024",
    }
    assert configured_full_chunk_widths(environ) == frozenset({2048, 1024})
    for width in (2048, 1024):
        with pytest.raises(PrefillChunkGeometryError):
            assert_prefill_chunk_coherent(width, environ)
    assert (
        assert_prefill_chunk_coherent(256, environ) == COHERENCE_NARROW_EAGER
    )


def test_configured_full_chunk_widths_defaults_and_garbage():
    assert configured_full_chunk_widths({}) == frozenset({DEFAULT_CHUNK_SIZE})
    assert configured_full_chunk_widths({CHUNK_SIZE_ENV: "4096"}) == frozenset(
        {4096}
    )
    # ``_prefill_chunk_size`` falls back to 2,048 on an unparsable knob.
    assert configured_full_chunk_widths(
        {CHUNK_SIZE_ENV: "wide"}
    ) == frozenset({DEFAULT_CHUNK_SIZE})
    assert configured_full_chunk_widths({CHUNK_SIZE_ENV: "auto"}) == frozenset(
        {DEFAULT_CHUNK_SIZE}
    )


def test_short_prompt_geometry_is_admitted():
    plan = guard_prefill_chunk_geometry(
        chunk_size=100,
        total_tokens=100,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        environ={WIRED_LIMIT_ENV: str(DRIVER_WIRED_LIMIT)},
        resident_bytes=CENSUS_PEAK_BYTES,
    )
    assert plan is not None and plan.fits and plan.chunks == 1


def test_compile_rows_mismatch_can_be_waived():
    assert (
        assert_prefill_chunk_coherent(
            4096, {ALLOW_COMPILE_ROWS_MISMATCH_ENV: "1"}
        )
        == COHERENCE_COMPILED
    )
    assert (
        assert_prefill_chunk_coherent(
            DEFAULT_CHUNK_SIZE,
            {COMPILE_ROWS_ENV: "4096", ALLOW_COMPILE_ROWS_MISMATCH_ENV: "1"},
        )
        == COHERENCE_NARROW_EAGER
    )


# ---------------------------------------------------------------------------
# Work-term arithmetic: why the middle path exists
# ---------------------------------------------------------------------------


def test_widening_the_chunk_raises_the_attention_work_term():
    shipped = attention_row_context_products(PROMPT_TOKENS, DEFAULT_CHUNK_SIZE)
    wide = attention_row_context_products(PROMPT_TOKENS, 2 * DEFAULT_CHUNK_SIZE)
    assert shipped == 150_994_944
    assert wide == 167_772_160
    assert wide / shipped == pytest.approx(1.1111, rel=1e-4)


def test_query_tiling_restores_the_shipped_attention_work_term_exactly():
    shipped = attention_row_context_products(PROMPT_TOKENS, DEFAULT_CHUNK_SIZE)
    tiled = attention_row_context_products(
        PROMPT_TOKENS, 2 * DEFAULT_CHUNK_SIZE, query_tile=DEFAULT_CHUNK_SIZE
    )
    assert tiled == shipped
    # ...and a 4x wider chunk tiled back to 2,048 is the same again.
    assert (
        attention_row_context_products(
            PROMPT_TOKENS, PROMPT_TOKENS, query_tile=DEFAULT_CHUNK_SIZE
        )
        == shipped
    )


def test_span_summary():
    assert summarize_spans([]) == (0, 0)
    spans = [(0, 2048), (2048, 4096), (4096, 4100)]
    assert summarize_spans(spans) == (3, 2048)


# ---------------------------------------------------------------------------
# Query-tile equivalence against a reference of MLX's unfused SDPA fallback
# ---------------------------------------------------------------------------


def _reference_masked_sdpa(q, k, v, mask, scale, neg):
    """``softmax(where(mask, q@k.T * scale, neg), -1) @ v`` in float64.

    This is exactly MLX's ``ScaledDotProductAttention`` fallback for a bool
    mask (mlx/fast.cpp): ``where(mask, scores, finfo(dtype).min)`` then a
    precise softmax then ``@ v``.
    """

    scores = np.einsum("hqd,hkd->hqk", q, k) * scale
    scores = np.where(mask[None, :, :], scores, neg)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", weights, v)


def _tiled_masked_sdpa(q, k, v, mask, scale, neg, *, tile, context_before):
    rows = q.shape[1]
    spans = query_tile_spans(rows, context_before=context_before, tile=tile)
    assert spans, "the test asked for a split the helper refused"
    parts = [
        _reference_masked_sdpa(
            q[:, r0:r1], k[:, :keys], v[:, :keys], mask[r0:r1, :keys], scale, neg
        )
        for r0, r1, keys in spans
    ]
    return np.concatenate(parts, axis=1)


def _causal_selection_mask(rows, context_before, total_keys, rng):
    """Causal AND a random block selection -- the shape _select_eager emits."""

    qpos = context_before + np.arange(rows)
    kpos = np.arange(total_keys)
    causal = kpos[None, :] <= qpos[:, None]
    selected = rng.random((rows, total_keys)) < 0.4
    tail = kpos[None, :] >= (qpos[:, None] - 3)
    return (selected | tail) & causal


@pytest.mark.parametrize(
    "rows,context_before,tile",
    [(8, 0, 4), (8, 16, 4), (12, 20, 4), (16, 48, 8), (9, 7, 3)],
)
def test_query_tiling_sees_the_same_visible_set(rows, context_before, tile):
    rng = np.random.default_rng(20260901 + rows + context_before + tile)
    heads, head_dim = 3, 6
    total_keys = context_before + rows
    q = rng.standard_normal((heads, rows, head_dim))
    k = rng.standard_normal((heads, total_keys, head_dim))
    v = rng.standard_normal((heads, total_keys, head_dim))
    mask = _causal_selection_mask(rows, context_before, total_keys, rng)
    scale = head_dim**-0.5
    # bfloat16's finfo.min, the constant MLX substitutes for a bool mask.
    neg = -3.3895313892515355e38

    full = _reference_masked_sdpa(q, k, v, mask, scale, neg)
    tiled = _tiled_masked_sdpa(
        q, k, v, mask, scale, neg, tile=tile, context_before=context_before
    )
    assert tiled.shape == full.shape
    # Same visible set, different reduction order (shorter softmax rows and a
    # shorter P@V contraction), so this is float64 agreement, not bit
    # equality -- exactly the claim the helper's docstring makes.
    np.testing.assert_allclose(tiled, full, rtol=1e-12, atol=1e-14)


def test_dropped_columns_carry_exactly_zero_softmax_weight():
    """Why truncating the keys is a no-op and not an approximation.

    The dense lane fills masked positions with the score dtype's
    ``finfo.min``.  After the softmax's max subtraction that is
    ``exp(-3.39e38 - max)``, which underflows to a hard 0.0 -- so a key the
    tile drops contributed nothing to either the denominator or the P@V sum.
    """

    neg = -3.3895313892515355e38
    for peak in (-30.0, 0.0, 12.5, 1e3):
        assert np.exp(neg - peak) == 0.0


def test_query_tiling_is_not_a_no_op_on_the_keys_it_reads():
    """Guard against a tiling that silently reads the whole context anyway."""

    spans = query_tile_spans(4096, context_before=12_288, tile=2048)
    assert spans == [(0, 2048, 14_336), (2048, 4096, 16_384)]
    assert sum((r1 - r0) * keys for r0, r1, keys in spans) < 4096 * 16_384


# ---------------------------------------------------------------------------
# Source tripwires for the two mlx-importing call sites
# ---------------------------------------------------------------------------


def _source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_generation_guards_the_streaming_prefill_geometry():
    text = _source("mtplx/generation.py")
    assert (
        text.count(
            "_guard_prefill_chunk_geometry(\n        rt, mtp_streaming_spans, "
            "chunk_size=prefill_chunk_size\n    )"
        )
        == 1
    )
    tree = ast.parse(text)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_guard_prefill_chunk_geometry" in names
    assert "_prefill_chunk_transient_per_token" in names
    # The dense transient must not be charged to a geometry the block-sparse
    # consumer serves -- otherwise the guard refuses working 262K prefills.
    assert "_prefill_dense_transient_per_token" in names
    assert "_qsa_large_prefill_enabled" in text
    # The narrow-chunk eager fallback is counted, not silently dropped.
    assert "COHERENCE_NARROW_EAGER" in text
    assert "rt.diagnostic_counters[NARROW_EAGER_COUNTER]" in text


def test_qsa_dense_attention_is_the_only_dense_sdpa_call_site():
    text = _source("mtplx/models/qwen4_exp.py")
    assert "_qsa_dense_attention(q, k, v, mask=mask, scale=self.scale)" in text
    # The mask-fuse helper is the single door to mx.fast SDPA in that lane.
    assert text.count("def _prefill_mask_fuse_sdpa(") == 1
    assert "force_fused=True" in text


def test_server_prices_the_plan_with_the_resolved_chunk_width():
    text = _source("mtplx/server/openai.py")
    assert (
        "_plan_transient_from_config(\n            _plan_model_config, "
        "chunk_size=_plan_prefill_chunk\n        )" in text
    )


def test_chunk_width_and_gdn_prefill_keys_are_harness_reachable():
    from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS

    for key in (
        "MTPLX_PREFILL_CHUNK_SIZE",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS",
        "MTPLX_GDN_BLOCKED_PREFILL",
        "MTPLX_GDN_BLOCKED_PREFILL_FORCE_STOCK",
    ):
        assert key in MODEL_RUNTIME_ENV_OVERRIDE_KEYS, key
