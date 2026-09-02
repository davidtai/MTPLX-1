"""MTPLX_FABLE_PREFILL_MASK_FUSE -- the causal arm of the dense QSA lane.

Below the sparse crossover the QSA indexer hands attention **no** selection
whenever the chunk's whole history fits its block budget
(``last_nb <= block_topk``): top-512 of at most 512 candidate blocks is all
of them, so the visible set is exactly causal.  The lane used to materialise
a dense ``[1, 1, S, T]`` bool mask for that case anyway, which is what pushes
MLX off the fused ``steel_attention`` route at ``head_dim`` 256.  This module
pins the replacement:

* **when** the selection is trivially complete (the ``T <= (block_topk + 1) *
  ratio - 1`` frontier, 2,051 tokens on the production pack) and **when** the
  mask is causal-with-offset (chunk k sees all prior context plus a causal
  block within the chunk);
* that MLX 0.32.2's ``mask="causal"`` -- documented lower-right aligned --
  describes that offset case exactly, checked against the installed MLX
  rather than assumed;
* that the visible set of the causal string, of the dense mask the lane
  built, and of ``_qsa_blocks_to_dense_mask`` under a full selection are the
  same set;
* the counters (``mask_fuse_causal`` / ``mask_fuse_bool`` /
  ``mask_fuse_unavailable``), the query-tile composition, and the loud
  one-shot refusal when the build has no fused kernel.

Everything runs on tiny tensors on the CPU stream: the GPU on a development
box is usually holding a guarded benchmark, and ``force_fused=True`` is
resolved while the op is BUILT, so the refusal path is exercised natively
there (MLX raises "the fused kernels require a GPU (Metal) stream").  What
this cannot settle is the fused kernel's own numerics or its speed.
"""

from __future__ import annotations

import io
import contextlib

import mlx.core as mx
import numpy as np
import pytest

import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.fable_prefill_chunk import QUERY_TILE_ENV, query_tile_spans

MASK_FUSE_ENV = "MTPLX_FABLE_PREFILL_MASK_FUSE"

#: Production pack (``~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-
#: Optimized-Speed/config.json``): indexer_budget 2048, compress_ratio 4 =>
#: block_topk 512.
BLOCK_TOPK = 512
RATIO = 4
#: Largest post-update context whose selection is trivially complete:
#: ``T // ratio <= block_topk``.
TRIVIAL_FRONTIER = (BLOCK_TOPK + 1) * RATIO - 1


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # set_default_device leaks into every later-collected module (pytest
    # shares one process), so restore it.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _clean_lane_state(monkeypatch):
    """Every knob and one-shot in this lane is process-global; reset them."""

    monkeypatch.delenv(MASK_FUSE_ENV, raising=False)
    monkeypatch.delenv(QUERY_TILE_ENV, raising=False)
    qwen4_exp._prefill_mask_fuse_enabled.cache_clear()
    qwen4_exp._prefill_mask_fuse_probed.cache_clear()
    saved_counts = dict(qwen4_exp._QSA_PREFILL_COUNTS)
    saved_unavailable = dict(qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE)
    qwen4_exp._QSA_PREFILL_COUNTS.clear()
    qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE.clear()
    try:
        yield
    finally:
        qwen4_exp._prefill_mask_fuse_enabled.cache_clear()
        qwen4_exp._prefill_mask_fuse_probed.cache_clear()
        qwen4_exp._QSA_PREFILL_COUNTS.clear()
        qwen4_exp._QSA_PREFILL_COUNTS.update(saved_counts)
        qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE.clear()
        qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE.update(saved_unavailable)


def _arm(monkeypatch, value: str = "1") -> None:
    monkeypatch.setenv(MASK_FUSE_ENV, value)
    qwen4_exp._prefill_mask_fuse_enabled.cache_clear()


def _lane_mask(pos_start: int, rows: int, total: int) -> mx.array:
    """The dense mask the lane builds at ``qwen4_exp.Attention.__call__``."""

    qpos = pos_start + mx.arange(rows, dtype=mx.int32)
    tpos = mx.arange(total, dtype=mx.int32)
    return (tpos[None, :] <= qpos[:, None])[None, None]


def _lower_right_causal(rows: int, total: int) -> mx.array:
    """MLX's documented ``"causal"``: last query aligns with the last key."""

    qpos = (total - rows) + mx.arange(rows, dtype=mx.int32)
    tpos = mx.arange(total, dtype=mx.int32)
    return (tpos[None, :] <= qpos[:, None])[None, None]


# ---------------------------------------------------------------------------
# 1. WHEN is the selection trivially complete / the mask causal-with-offset
# ---------------------------------------------------------------------------


def test_indexer_short_circuits_exactly_at_the_block_budget():
    """``last_nb <= block_topk`` is the whole condition, in both routes."""

    trivial = [t for t in range(1, 4 * TRIVIAL_FRONTIER) if t // RATIO <= BLOCK_TOPK]
    assert max(trivial) == TRIVIAL_FRONTIER == 2051
    # Both the eager route (_call_rows) and the compiled route
    # (_compiled_mode -> "update_only") use this one predicate, so the two
    # cannot disagree about when attention gets no selection.
    source = qwen4_exp.__file__
    text = open(source, encoding="utf-8").read()
    assert text.count("last_nb <= self.block_topk") == 2
    assert "last_nb = T // self.ratio" in text


@pytest.mark.parametrize(
    "chunk, prompt",
    [(2048, 16_384), (4096, 16_384), (2048, 32_768), (4096, 32_768)],
)
def test_which_prefill_chunks_are_trivially_complete(chunk, prompt):
    """Only a chunk whose POST-update context is <= 2,051 gets no selection.

    At the retained prefill width (4,096) that is no chunk at all, at 16K or
    at 32K: the causal arm cannot fire there, and the whole win of the flag
    at those cells is the bool-mask arm.  At the shipped 2,048 width it is
    chunk 0 and nothing else.
    """

    trivial = [
        pos_start
        for pos_start in range(0, prompt, chunk)
        # T == pos_start + S, S == this chunk's rows
        if (pos_start + min(chunk, prompt - pos_start)) // RATIO <= BLOCK_TOPK
    ]
    assert trivial == ([0] if chunk <= TRIVIAL_FRONTIER else [])


@pytest.mark.parametrize(
    "pos_start, rows, total, expected",
    [
        (0, 2048, 2048, True),  # first chunk
        (2048, 2048, 4096, True),  # causal-with-offset: all prior + causal
        (16_384, 4096, 20_480, True),
        (0, 1, 1, True),
        (0, 2048, 4096, False),  # padded/capacity-shaped KV: NOT lower-right
        (2048, 2048, 4097, False),
        (0, 0, 0, False),
    ],
)
def test_causal_string_is_exact_only_when_last_query_is_last_key(
    pos_start, rows, total, expected
):
    assert (
        qwen4_exp._prefill_causal_mask_is_exact(
            pos_start=pos_start, rows=rows, total_keys=total
        )
        is expected
    )


# ---------------------------------------------------------------------------
# 2. The visible sets are the SAME SET (the exactness invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pos_start, rows", [(0, 8), (0, 13), (5, 8), (16, 8), (2048 - 16, 16)]
)
def test_lane_mask_equals_mlx_lower_right_causal(pos_start, rows):
    total = pos_start + rows
    lane = np.array(_lane_mask(pos_start, rows, total))
    ref = np.array(_lower_right_causal(rows, total))
    assert np.array_equal(lane, ref)
    # ... and it really is "all prior context + causal inside the chunk".
    dense = lane[0, 0]
    assert dense[:, :pos_start].all()
    within = dense[:, pos_start:]
    assert np.array_equal(within, np.tril(np.ones((rows, rows), dtype=bool)))


def test_full_block_selection_reconstructs_exactly_the_causal_mask():
    """``causal == the full-selection mask`` -- proved on the real function.

    ``_qsa_blocks_to_dense_mask`` is the lane's own reconstruction of a QSA
    selection.  Feed it every complete block (which is what the selector
    returns when ``nb_total <= block_topk``, since ``k_eff = min(block_topk,
    nb_total)``) and it must produce the causal mask, key for key.
    """

    ratio = 4
    for pos_start, rows in ((0, 16), (0, 12), (8, 8), (12, 20)):
        total = pos_start + rows
        logical_blocks = total // ratio
        topk = max(1, logical_blocks)
        # Every row selects all logical blocks; rows whose own complete-block
        # frontier is lower have the surplus ids clipped by the function's own
        # in-range guard, exactly as a real top-k of fewer candidates would.
        block_ids = mx.broadcast_to(
            mx.arange(topk, dtype=mx.int32)[None, :], (rows, topk)
        )
        block_valid = mx.ones((rows, topk), dtype=mx.bool_)
        got = qwen4_exp._qsa_blocks_to_dense_mask(
            block_ids,
            block_valid,
            pos_start=pos_start,
            total_tokens=total,
            compress_ratio=ratio,
        )
        want = _lane_mask(pos_start, rows, total)
        assert np.array_equal(np.array(got), np.array(want)), (pos_start, rows)


@pytest.mark.parametrize("pos_start, rows", [(0, 16), (12, 20), (33, 7)])
def test_mlx_causal_string_and_dense_mask_agree_numerically(pos_start, rows):
    """Against the INSTALLED MLX, not against its documentation."""

    total = pos_start + rows
    rng = np.random.default_rng(20260902)
    shape = (1, 2, rows, 16)
    kshape = (1, 2, total, 16)
    q = mx.array(rng.standard_normal(shape).astype(np.float32))
    k = mx.array(rng.standard_normal(kshape).astype(np.float32))
    v = mx.array(rng.standard_normal(kshape).astype(np.float32))
    scale = 16 ** -0.5
    string_out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=qwen4_exp._CAUSAL_MASK
    )
    dense_out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=_lane_mask(pos_start, rows, total)
    )
    mx.eval(string_out, dense_out)
    assert np.allclose(np.array(string_out), np.array(dense_out), atol=1e-6)


def test_causal_and_dense_arms_agree_within_bf16_rounding():
    """bf16 inputs: the two arms are the same visible set, one rounding class."""

    pos_start, rows, total = 24, 24, 48
    rng = np.random.default_rng(7)
    q = mx.array(rng.standard_normal((1, 2, rows, 16)).astype(np.float32)).astype(
        mx.bfloat16
    )
    kv = mx.array(rng.standard_normal((1, 2, total, 16)).astype(np.float32)).astype(
        mx.bfloat16
    )
    scale = 16 ** -0.5
    a = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=scale
    )
    b = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=scale
    )
    mx.eval(a, b)
    # bf16 has ~3 decimal digits; the visible set being identical is what
    # makes this a tolerance and not a mismatch.
    assert np.allclose(
        np.array(a.astype(mx.float32)),
        np.array(b.astype(mx.float32)),
        atol=6e-3,
        rtol=6e-3,
    )


# ---------------------------------------------------------------------------
# 3. Query-tile composition
# ---------------------------------------------------------------------------


def test_query_tile_spans_keep_every_tile_lower_right_aligned():
    """Why the string survives tiling: each tile's last query is its last key."""

    for total_keys, rows, tile in ((4096, 4096, 2048), (20_480, 4096, 1024)):
        context_before = total_keys - rows
        for r0, r1, keys in query_tile_spans(
            rows, context_before=context_before, tile=tile
        ):
            assert qwen4_exp._prefill_causal_mask_is_exact(
                pos_start=context_before + r0, rows=r1 - r0, total_keys=keys
            )


def test_tiled_causal_string_matches_untiled_and_tiled_dense(monkeypatch):
    pos_start, rows, total = 32, 32, 64
    rng = np.random.default_rng(11)
    q = mx.array(rng.standard_normal((1, 2, rows, 16)).astype(np.float32))
    kv = mx.array(rng.standard_normal((1, 2, total, 16)).astype(np.float32))
    scale = 16 ** -0.5
    untiled = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=scale
    )
    monkeypatch.setenv(QUERY_TILE_ENV, "8")
    tiled_causal = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=scale
    )
    tiled_dense = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=scale
    )
    mx.eval(untiled, tiled_causal, tiled_dense)
    assert qwen4_exp._QSA_PREFILL_COUNTS.get("query_tile") == 2
    assert np.allclose(np.array(tiled_causal), np.array(untiled), atol=1e-6)
    assert np.allclose(np.array(tiled_causal), np.array(tiled_dense), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Counters, gating and the loud refusal
# ---------------------------------------------------------------------------


class _FakeSdpa:
    """Stand-in for a build that DOES have the fused kernels."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, q, k, v, *, scale, mask=None, force_fused=False, **kw):
        kind = "causal" if isinstance(mask, str) else "bool"
        self.calls.append((kind, bool(force_fused)))
        if force_fused and kind in self.fail_on:
            raise ValueError(f"no fused kernel for {kind}")
        return mx.zeros(q.shape, dtype=q.dtype)


def _install(monkeypatch, fake):
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", fake)


def test_flag_off_never_forces_fused_and_never_builds_a_string(monkeypatch):
    fake = _FakeSdpa()
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    qwen4_exp._qsa_dense_attention(q, kv, kv, mask=_lane_mask(0, 4, 4), scale=1.0)
    assert fake.calls == [("bool", False)]
    assert "mask_fuse_causal" not in qwen4_exp._QSA_PREFILL_COUNTS
    assert "mask_fuse_bool" not in qwen4_exp._QSA_PREFILL_COUNTS


def test_counters_split_causal_and_bool(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa()
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
    )
    qwen4_exp._qsa_dense_attention(q, kv, kv, mask=_lane_mask(0, 4, 4), scale=1.0)
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_causal") == 1
    assert counts.get("mask_fuse_bool") == 1
    assert "mask_fuse_unavailable" not in counts
    # probe (2) + real call (2), all force_fused.
    assert fake.calls == [
        ("causal", True),
        ("causal", True),
        ("bool", True),
        ("bool", True),
    ]


def test_s1_decode_rows_never_take_the_forced_route(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa()
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 1, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    qwen4_exp._qsa_dense_attention(q, kv, kv, mask=_lane_mask(3, 1, 4), scale=1.0)
    assert fake.calls == [("bool", False)]


def test_refusal_is_loud_one_shot_and_per_kind(monkeypatch):
    _arm(monkeypatch)
    fake = _FakeSdpa(fail_on={"causal"})
    _install(monkeypatch, fake)
    q = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    kv = mx.zeros((1, 2, 4, 256), dtype=mx.bfloat16)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for _ in range(3):
            qwen4_exp._qsa_dense_attention(
                q, kv, kv, mask=qwen4_exp._CAUSAL_MASK, scale=1.0
            )
        qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=_lane_mask(0, 4, 4), scale=1.0
        )
    message = err.getvalue()
    assert message.count(MASK_FUSE_ENV) == 1, message
    assert "causal-mask SDPA is available" in message
    assert "head_dim 256" in message
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    assert counts.get("mask_fuse_unavailable") == 1
    assert "mask_fuse_causal" not in counts
    # The bool arm is unaffected: availability is tracked per kind.
    assert counts.get("mask_fuse_bool") == 1
    # One probe for causal (refused, sticky) + three fallbacks, then the bool
    # probe and its real call.  Never a second forced causal attempt.
    assert fake.calls == [
        ("causal", True),
        ("causal", False),
        ("causal", False),
        ("causal", False),
        ("bool", True),
        ("bool", True),
    ]


def test_probe_refuses_natively_on_a_build_without_fused_kernels(monkeypatch):
    """No monkeypatch: the CPU stream has no fused kernel, so MLX raises."""

    _arm(monkeypatch)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert (
            qwen4_exp._prefill_mask_fuse_probed("causal", 256, "bfloat16") is False
        )
        assert qwen4_exp._prefill_mask_fuse_probed("bool", 256, "bfloat16") is False
    message = err.getvalue()
    assert "require a GPU (Metal) stream" in message
    assert qwen4_exp._PREFILL_MASK_FUSE_UNAVAILABLE == {
        "causal": True,
        "bool": True,
    }
    assert qwen4_exp._QSA_PREFILL_COUNTS.get("mask_fuse_unavailable") == 2


def test_armed_flag_on_an_unavailable_build_still_returns_the_dense_answer(
    monkeypatch,
):
    _arm(monkeypatch)
    pos_start, rows, total = 8, 8, 16
    rng = np.random.default_rng(3)
    q = mx.array(rng.standard_normal((1, 2, rows, 16)).astype(np.float32))
    kv = mx.array(rng.standard_normal((1, 2, total, 16)).astype(np.float32))
    with contextlib.redirect_stderr(io.StringIO()):
        armed = qwen4_exp._qsa_dense_attention(
            q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=0.25
        )
    qwen4_exp._prefill_mask_fuse_enabled.cache_clear()
    monkeypatch.delenv(MASK_FUSE_ENV, raising=False)
    stock = qwen4_exp._qsa_dense_attention(
        q, kv, kv, mask=_lane_mask(pos_start, rows, total), scale=0.25
    )
    mx.eval(armed, stock)
    assert np.array_equal(np.array(armed), np.array(stock))


# ---------------------------------------------------------------------------
# 5. Knobs stay where the harness can reach them
# ---------------------------------------------------------------------------


def test_sparse_crossover_knob_is_documented_and_unchanged(monkeypatch):
    from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS

    assert "MTPLX_QSA_PREFILL_MIN_CONTEXT" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    assert "MTPLX_QSA_PREFILL_DEBUG" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
    monkeypatch.delenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", raising=False)
    assert qwen4_exp._qsa_prefill_min_context() == 32_768
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "8192")
    assert qwen4_exp._qsa_prefill_min_context() == 8192
    # Floor: below the 2,048-token budget the indexer has nothing to select.
    monkeypatch.setenv("MTPLX_QSA_PREFILL_MIN_CONTEXT", "512")
    assert qwen4_exp._qsa_prefill_min_context() == 2049


def test_mask_fuse_flag_rides_the_fable_raw_passthrough():
    """MTPLX_FABLE_* is exempt from the --candidate-env profile check."""

    import importlib.util
    from pathlib import Path

    root = Path(qwen4_exp.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_abba_driver_for_test", root / "scripts" / "fable" / "abba_driver.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.is_raw_env_mtplx_key(MASK_FUSE_ENV) is True
    assert driver.parse_key_values(
        [f"{MASK_FUSE_ENV}=1"], flag="--candidate-extra-env", require_mtplx=False
    ) == {MASK_FUSE_ENV: "1"}
    with pytest.raises(RuntimeError, match="belong on --candidate-env"):
        driver.parse_key_values(
            ["MTPLX_QSA_PREFILL_DEBUG=1"],
            flag="--candidate-extra-env",
            require_mtplx=False,
        )


# ---------------------------------------------------------------------------
# 6. The routing decision at the real call site
# ---------------------------------------------------------------------------

#: A miniature of the production QSA geometry.  block_topk = 8 // 2 = 4, so
#: the trivially-complete frontier is (4 + 1) * 2 - 1 = 9 tokens -- the same
#: arithmetic the 2,048-budget pack runs at 2,051.
TINY = dict(
    hidden_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=8,
    indexer_compress_ratio=2,
)
TINY_FRONTIER = (TINY["indexer_budget"] // TINY["indexer_compress_ratio"] + 1) * TINY[
    "indexer_compress_ratio"
] - 1


def _tiny_attention():
    mx.random.seed(20260902)
    layer = qwen4_exp.Attention(qwen4_exp.TextArgs(**TINY))
    layer.eval()
    mx.eval(layer.parameters())
    return layer


class _MaskRecorder(_FakeSdpa):
    """Records the mask each SDPA call receives, and returns real numbers."""

    def __call__(self, q, k, v, *, scale, mask=None, force_fused=False, **kw):
        kind = "causal" if isinstance(mask, str) else ("none" if mask is None else "bool")
        self.calls.append((kind, bool(force_fused)))
        if force_fused and kind in self.fail_on:
            raise ValueError(f"no fused kernel for {kind}")
        return _REAL_SDPA(q, k, v, scale=scale, mask=mask)


_REAL_SDPA = mx.fast.scaled_dot_product_attention


@pytest.mark.parametrize(
    "rows, expect_kind",
    [
        (TINY_FRONTIER - 1, "causal"),  # T <= frontier: no selection at all
        (TINY_FRONTIER + 1, "bool"),  # past it: a real top-k selection
    ],
)
def test_attention_routes_by_the_trivially_complete_frontier(
    monkeypatch, rows, expect_kind
):
    _arm(monkeypatch)
    layer = _tiny_attention()
    recorder = _MaskRecorder()
    _install(monkeypatch, recorder)
    x = (mx.random.normal((1, rows, TINY["hidden_size"])) * 0.3).astype(mx.bfloat16)
    cache = qwen4_exp.QSACache(compress_ratio=layer.indexer.ratio)
    with contextlib.redirect_stderr(io.StringIO()):
        mx.eval(layer(x, cache))
    kinds = [kind for kind, _ in recorder.calls]
    assert expect_kind in kinds, kinds
    counts = qwen4_exp._QSA_PREFILL_COUNTS
    if expect_kind == "causal":
        assert counts.get("mask_causal_eligible") == 1
    else:
        assert "mask_causal_eligible" not in counts


def test_flag_off_keeps_the_dense_causal_tensor_and_the_same_answer(monkeypatch):
    """Flag off is byte-identical, and the armed causal arm sees the same set."""

    layer = _tiny_attention()
    rows = TINY_FRONTIER - 1
    x = (mx.random.normal((1, rows, TINY["hidden_size"])) * 0.3).astype(mx.bfloat16)

    recorder = _MaskRecorder()
    _install(monkeypatch, recorder)
    off = layer(x, qwen4_exp.QSACache(compress_ratio=layer.indexer.ratio))
    mx.eval(off)
    assert recorder.calls == [("bool", False)]

    _arm(monkeypatch)
    recorder2 = _MaskRecorder()
    _install(monkeypatch, recorder2)
    with contextlib.redirect_stderr(io.StringIO()):
        on = layer(x, qwen4_exp.QSACache(compress_ratio=layer.indexer.ratio))
        mx.eval(on)
    # The recorder stands in for a build that HAS the fused kernel, so the
    # probe passes and the real call is forced -- both with the STRING and
    # never a tensor.  Its return value is real MLX math, so the equality
    # below is the visible-set claim, evaluated end to end through the layer.
    assert recorder2.calls == [("causal", True), ("causal", True)]
    assert np.array_equal(
        np.array(off.astype(mx.float32)), np.array(on.astype(mx.float32))
    )
