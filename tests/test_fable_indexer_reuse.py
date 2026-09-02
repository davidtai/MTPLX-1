"""MTPLX_FABLE_INDEXER_REUSE -- gating, the block-set definition, skipped work.

Row K-D2 claims depths 2 and 3 of one MTP draft chain can be handed the depth-1
QSA block selection plus whatever block the chain's own tokens completed, and
skip the query preparation, the score GEMM and the top-k that produced it.
Everything below runs on tiny tensors on the CPU stream (the GPU on a
development box is usually holding a guarded benchmark), which is enough to
settle everything except the ms/window:

* **the gate.**  Flag off, the selection is what it was before the flag
  existed -- including inside a draft-depth scope, so the ``draft_mtp``
  instrumentation alone changes nothing.
* **the definition.**  ``S_d == S_1 union {b : nb_1 <= b < nb_d}``, at a
  pooled-block boundary crossed at depth 2, at depth 3, and not at all; the
  sets are a superset chain; every selected block is complete and causal.
* **the masked duplicate.**  When no block completed, the extra slot repeats an
  already-selected id.  The rows-gather lane -- the production MTP D3 lane --
  must mark it invalid, or the softmax double-counts four tokens.
* **the skipped work.**  A counting shim over the query preparation and the
  eager scorer records one call per cycle armed against three unarmed.
* **the refusals.**  A lane the reuse cannot serve, a chain deeper than the
  single extra slot is exact for, and ``MTPLX_FABLE_COMPILED_DRAFT`` all raise.

What is NOT settled here, and needs the GPU: the cycle-time saving (ABBA) and
the acceptance cost (``scripts/fable/shadow_draft_harness.py``).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import mtplx.fable_indexer_reuse as reuse_mod
import mtplx.models.qwen4_exp as qwen4_exp
from mtplx.fable_indexer_reuse import draft_depth_scope


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
def _clean_environment(monkeypatch):
    monkeypatch.delenv(reuse_mod.ENV_FLAG, raising=False)
    monkeypatch.delenv("MTPLX_QSA_FLASH", raising=False)
    monkeypatch.delenv("MTPLX_QSA_GATHER_DECODE", raising=False)
    monkeypatch.delenv("MTPLX_FUSED_QSA_INDEXER", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_QSA_INDEXER", raising=False)
    reuse_mod.reset_indexer_reuse_counters()
    yield
    reuse_mod.reset_indexer_reuse_counters()


# The production compress ratio: four tokens per pooled block, so a 3-step
# chain crosses a block boundary on some cycles and not on others.
RATIO = 4
HIDDEN = 32
HC = 4
WIDENED = HIDDEN * HC
#: budget // ratio == 8 selected blocks, so the selector engages past 36
#: tokens of history and k_eff stays 8 across all three depths.
BLOCK_TOPK = 8


def _args(**overrides):
    fields = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=HC,
        hc_lowrank=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=BLOCK_TOPK * RATIO,
        indexer_compress_ratio=RATIO,
        partial_rotary_factor=0.5,
        full_attention_interval=2,
    )
    fields.update(overrides)
    return qwen4_exp.TextArgs(**fields)


def _head(seed: int = 7, **overrides):
    mx.random.seed(seed)
    return qwen4_exp.Qwen4ExpMTP(_args(**overrides))


def _rows(width: int, seed: int):
    mx.random.seed(seed)
    widened = mx.random.normal((1, width, WIDENED)).astype(mx.float32)
    embedding = mx.random.normal((1, width, HIDDEN)).astype(mx.float32)
    mx.eval(widened, embedding)
    return widened, embedding


class _SelectionSpy:
    """Record every decode selection the indexer returns, in order."""

    def __init__(self, indexer):
        self.indexer = indexer
        self.masks: list[mx.array] = []
        self._stock = indexer._call_decode
        indexer._call_decode = self  # type: ignore[assignment]

    def __call__(self, *args, **kwargs):
        result = self._stock(*args, **kwargs)
        self.masks.append(result)
        return result


def _prefill(head, cache, tokens: int, seed: int = 1) -> None:
    widened, embedding = _rows(tokens, seed=seed)
    mx.eval(head.fuse_and_run_history(widened, embedding, cache))


def _draft_cycle(head, cache, *, depth: int = 3, seed: int = 50):
    """One MTP draft chain: ``depth`` single-row steps in numbered scopes."""

    widened, embedding = _rows(1, seed=seed)
    for level in range(1, depth + 1):
        with draft_depth_scope(level):
            produced = head.fuse_and_run(widened, embedding, cache)
        mx.eval(produced)
        widened = produced[:, -1:, :]


def _blocks_from_mask(mask: mx.array, pos_start: int) -> set[int]:
    """The selected COMPLETE blocks a dense decode mask encodes.

    The mask is ``(selected_blocks | tail) & causal``.  Below the query's tail
    start every token is causal and outside the tail, so a complete block is
    selected exactly when all ``ratio`` of its tokens are set -- an exact
    inverse, not a heuristic.
    """

    row = mask[0, 0, 0]
    nb_q = (pos_start + 1) // RATIO
    flags = [bool(value) for value in row.tolist()]
    return {
        block
        for block in range(nb_q)
        if all(flags[block * RATIO : (block + 1) * RATIO])
    }


def _selection_sets(head, cache, spy, *, prefill_tokens: int, depth: int = 3):
    """``[(pos_start, selected_blocks), ...]`` for one drafted cycle."""

    spy.masks.clear()
    _draft_cycle(head, cache, depth=depth)
    assert len(spy.masks) == depth
    return [
        (prefill_tokens + level, _blocks_from_mask(mask, prefill_tokens + level))
        for level, mask in enumerate(spy.masks)
    ]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefill_tokens", [41, 42, 43])
def test_flag_off_selection_is_the_stock_selection(prefill_tokens):
    """Off, a draft-depth scope changes nothing: same masks, no anchor."""

    scoped_head = _head()
    scoped_cache = [qwen4_exp.QSACache(RATIO)]
    scoped_spy = _SelectionSpy(scoped_head.layers[0].self_attn.indexer)
    _prefill(scoped_head, scoped_cache, prefill_tokens)
    scoped = _selection_sets(
        scoped_head, scoped_cache, scoped_spy, prefill_tokens=prefill_tokens
    )
    scoped_masks = list(scoped_spy.masks)

    # The same chain with no scope at all -- the pre-flag code path.
    plain_head = _head()
    plain_cache = [qwen4_exp.QSACache(RATIO)]
    plain_spy = _SelectionSpy(plain_head.layers[0].self_attn.indexer)
    _prefill(plain_head, plain_cache, prefill_tokens)
    widened, embedding = _rows(1, seed=50)
    for _ in range(3):
        produced = plain_head.fuse_and_run(widened, embedding, plain_cache)
        mx.eval(produced)
        widened = produced[:, -1:, :]

    assert len(scoped_masks) == len(plain_spy.masks) == 3
    for scoped_mask, plain_mask in zip(scoped_masks, plain_spy.masks):
        assert bool(mx.array_equal(scoped_mask, plain_mask).item())
    assert scoped_head.layers[0].self_attn.indexer._indexer_reuse_anchor is None
    assert reuse_mod.indexer_reuse_counters() == {"cycles": 0, "steps_reused": 0}
    assert scoped  # the selector really engaged


def test_flag_off_outside_a_draft_scope_is_untouched_when_armed(monkeypatch):
    """Armed, a prefill or a history append still scores itself normally."""

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    head = _head()
    cache = [qwen4_exp.QSACache(RATIO)]
    _prefill(head, cache, 42)
    assert head.layers[0].self_attn.indexer._indexer_reuse_anchor is None
    assert reuse_mod.indexer_reuse_counters() == {"cycles": 0, "steps_reused": 0}


# ---------------------------------------------------------------------------
# the block-set definition
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prefill_tokens, expected_new_blocks",
    [
        # nb over the three depths: 10, 10, 11 -- the boundary is crossed by
        # the depth-2 token, so depth 3 gains block 10.
        (41, [0, 0, 1]),
        # nb: 10, 11, 11 -- crossed by the depth-1 token.
        (42, [0, 1, 1]),
        # nb: 11, 11, 11 -- no block completes inside the cycle.
        (43, [0, 0, 0]),
    ],
)
def test_reused_sets_are_the_anchor_union_the_newest_blocks(
    monkeypatch, prefill_tokens, expected_new_blocks
):
    """S_d == S_1 union {b : nb_1 <= b < nb_d}, exactly, at every depth."""

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    head = _head()
    cache = [qwen4_exp.QSACache(RATIO)]
    spy = _SelectionSpy(head.layers[0].self_attn.indexer)
    _prefill(head, cache, prefill_tokens)
    steps = _selection_sets(head, cache, spy, prefill_tokens=prefill_tokens)

    anchor_pos, anchor_blocks = steps[0]
    nb_anchor = (anchor_pos + 1) // RATIO
    assert len(anchor_blocks) == BLOCK_TOPK, "depth 1 must be the stock top-k"

    for index, (pos_start, blocks) in enumerate(steps):
        nb_now = (pos_start + 1) // RATIO
        assert nb_now - nb_anchor == expected_new_blocks[index]
        expected = anchor_blocks | set(range(nb_anchor, nb_now))
        assert blocks == expected, f"depth {index + 1}"
        # causal and complete: every id is a complete block strictly below the
        # query's own tail start.
        assert all(0 <= block < nb_now for block in blocks)

    # superset chain: nothing the chain has already attended to is dropped.
    assert steps[0][1] <= steps[1][1] <= steps[2][1]


def test_a_second_cycle_re_anchors_on_its_own_depth_one(monkeypatch):
    """The anchor is per cycle: cycle 2 never reuses cycle 1's block set."""

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    head = _head()
    cache = [qwen4_exp.QSACache(RATIO)]
    spy = _SelectionSpy(head.layers[0].self_attn.indexer)
    _prefill(head, cache, 42)

    first = _selection_sets(head, cache, spy, prefill_tokens=42)
    second = _selection_sets(head, cache, spy, prefill_tokens=45)

    assert reuse_mod.indexer_reuse_counters() == {"cycles": 2, "steps_reused": 4}
    # Cycle 2's depth 1 is a fresh top-k of a history three tokens longer, so
    # it is not obliged to equal cycle 1's -- but it IS obliged to be the
    # anchor its own depths 2 and 3 extend.
    nb_second_anchor = (second[0][0] + 1) // RATIO
    for pos_start, blocks in second[1:]:
        nb_now = (pos_start + 1) // RATIO
        assert blocks == second[0][1] | set(range(nb_second_anchor, nb_now))
    # ...and cycle 1's own depths were extensions of cycle 1's anchor, not of
    # anything cycle 2 later produced.
    nb_first_anchor = (first[0][0] + 1) // RATIO
    for pos_start, blocks in first[1:]:
        nb_now = (pos_start + 1) // RATIO
        assert blocks == first[0][1] | set(range(nb_first_anchor, nb_now))


def test_a_depth_two_against_a_different_cache_declines(monkeypatch):
    """The anchor belongs to one cache; another's depth 2 scores itself."""

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    head = _head()
    indexer = head.layers[0].self_attn.indexer
    cache_a = [qwen4_exp.QSACache(RATIO)]
    cache_b = [qwen4_exp.QSACache(RATIO)]
    _prefill(head, cache_a, 42)
    _prefill(head, cache_b, 42, seed=2)

    widened, embedding = _rows(1, seed=50)
    with draft_depth_scope(1):
        mx.eval(head.fuse_and_run(widened, embedding, cache_a))
    assert reuse_mod.indexer_reuse_counters()["cycles"] == 1

    prepared: list[int] = []
    stock_prepare = indexer._prepare_queries

    def _counted_prepare(*args, **kwargs):
        prepared.append(1)
        return stock_prepare(*args, **kwargs)

    indexer._prepare_queries = _counted_prepare  # type: ignore[assignment]
    with draft_depth_scope(2):
        mx.eval(head.fuse_and_run(widened, embedding, cache_b))
    assert prepared == [1], "a foreign cache must score its own selection"
    assert reuse_mod.indexer_reuse_counters()["steps_reused"] == 0
    assert indexer._indexer_reuse_anchor is None, "the stale anchor is dropped"


# ---------------------------------------------------------------------------
# the masked duplicate, on the production rows-gather lane
# ---------------------------------------------------------------------------
class _StubGatherCache:
    """The minimum ``_select_eager`` reads on the fixed rows-gather lane.

    The production MTP D3 cache is a ``TensorOffsetQSACache`` whose offset
    lives on the GPU; building one on the CPU stream is not possible, and the
    property under test is not about the offset's residence but about which
    slots the branch marks valid.
    """

    fixed_capacity = True
    fixed_rows_gather = True

    def __init__(self, pooled: mx.array):
        self.pooled = pooled
        self.raw_keys = mx.zeros((1, int(pooled.shape[1]) * RATIO, 1))

    def pooled_f32_view(self, nb_total: int) -> mx.array:
        return mx.swapaxes(
            self.pooled[:, :nb_total, :].astype(mx.float32), 1, 2
        )[:, None]


def _gather_lane_selection(indexer, cache, pooled, pos_start, depth):
    mx.random.seed(300 + depth)
    q = mx.random.normal((1, 1, indexer.n_heads, indexer.head_dim))
    mx.eval(q)
    with draft_depth_scope(depth):
        return indexer._select_eager(
            None if depth > 1 else q,
            pos_start,
            cache,
            pooled,
            pos_start + 1,
            rows=1,
        )


def test_rows_gather_never_double_counts_a_token(monkeypatch):
    """The duplicate slot the reuse pads with is marked invalid.

    Without this the softmax reads four tokens twice on every depth that did
    not complete a block -- silently, and only on the production lane.
    """

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    indexer = qwen4_exp.QSAIndexer(_args())
    mx.random.seed(5)
    pooled = mx.random.normal((1, 24, indexer.head_dim))
    mx.eval(pooled)
    cache = _StubGatherCache(pooled)

    # pos_start 41, 42, 43: nb = 10, 10, 11 -- no block completes for depth 2
    # (the padded duplicate case) and one completes for depth 3.
    blocks = []
    for depth, pos_start in enumerate((41, 42, 43), start=1):
        kind, token_idx, token_ok = _gather_lane_selection(
            indexer, cache, pooled, pos_start, depth
        )
        assert kind == "gather_rows"
        live = [
            int(token)
            for token, ok in zip(token_idx[0].tolist(), token_ok[0].tolist())
            if ok
        ]
        assert len(live) == len(set(live)), f"depth {depth} gathers a token twice"
        nb_now = (pos_start + 1) // RATIO
        assert max(live) <= pos_start, "the gathered set must stay causal"
        complete = {token // RATIO for token in live if token < nb_now * RATIO}
        assert len(live) == len(complete) * RATIO + (pos_start + 1 - nb_now * RATIO)
        blocks.append(complete)

    # Depth 2 padded its extra slot with a masked duplicate (no block
    # completed); depth 3's slot carried the block the depth-2 token finished.
    assert blocks[1] == blocks[0]
    assert blocks[2] == blocks[0] | {(41 + 1) // RATIO}


# ---------------------------------------------------------------------------
# the skipped work
# ---------------------------------------------------------------------------
def test_query_preparation_and_scorer_run_once_per_cycle(monkeypatch):
    """One preparation + one top-k per cycle armed; three unarmed."""

    def _counts(armed: bool) -> tuple[int, int]:
        if armed:
            monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
        else:
            monkeypatch.delenv(reuse_mod.ENV_FLAG, raising=False)
        head = _head()
        indexer = head.layers[0].self_attn.indexer
        prepared: list[int] = []
        scored: list[int] = []
        stock_prepare = indexer._prepare_queries
        stock_select = indexer._select_eager

        def _counted_prepare(*args, **kwargs):
            prepared.append(1)
            return stock_prepare(*args, **kwargs)

        def _counted_select(*args, **kwargs):
            # args[0] is `q`: None exactly when the anchor serves this depth,
            # which is also exactly when no score GEMM or top-k runs inside.
            scored.append(1 if args[0] is not None else 0)
            return stock_select(*args, **kwargs)

        indexer._prepare_queries = _counted_prepare  # type: ignore[assignment]
        indexer._select_eager = _counted_select  # type: ignore[assignment]
        cache = [qwen4_exp.QSACache(RATIO)]
        _prefill(head, cache, 42)
        prepared.clear()
        scored.clear()
        _draft_cycle(head, cache)
        return len(prepared), sum(scored)

    assert _counts(armed=False) == (3, 3)
    assert _counts(armed=True) == (1, 1)
    assert reuse_mod.indexer_reuse_counters() == {"cycles": 1, "steps_reused": 2}


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------
def test_flash_lane_raises_rather_than_reverting(monkeypatch):
    """An armed flag never reports a stock number from an unserved lane."""

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    monkeypatch.setenv("MTPLX_QSA_FLASH", "1")
    head = _head()
    cache = [qwen4_exp.QSACache(RATIO)]
    _prefill(head, cache, 42)
    with pytest.raises(RuntimeError, match="flash-skip"):
        _draft_cycle(head, cache)


def test_decode_gather_lane_raises(monkeypatch):
    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_DECODE", "1")
    head = _head()
    cache = [qwen4_exp.QSACache(RATIO)]
    _prefill(head, cache, 42)
    with pytest.raises(RuntimeError, match="decode-gather"):
        _draft_cycle(head, cache)


def test_depth_past_the_exact_bound_raises(monkeypatch):
    """One extra slot is exact only while ``depth - 1 <= ratio``."""

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    head = _head(indexer_compress_ratio=2, indexer_budget=BLOCK_TOPK * 2)
    cache = [qwen4_exp.QSACache(2)]
    widened, embedding = _rows(42, seed=1)
    mx.eval(head.fuse_and_run_history(widened, embedding, cache))
    with pytest.raises(RuntimeError, match="depth - 1 <= ratio"):
        _draft_cycle(head, cache, depth=4)


def test_ratio_one_is_refused(monkeypatch):
    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    indexer = qwen4_exp.QSAIndexer(_args(indexer_compress_ratio=1, indexer_budget=8))
    with draft_depth_scope(1):
        with pytest.raises(RuntimeError, match="compress_ratio >= 2"):
            indexer._indexer_reuse_stash(
                object(),
                rows=1,
                nb_q=mx.array([4], dtype=mx.int32),
                top_idx=mx.zeros((1, 8), dtype=mx.int32),
                k_eff=8,
            )


def test_compiled_draft_and_reuse_are_mutually_exclusive(monkeypatch):
    from mtplx.fable_compiled_draft import (
        CompiledDraftUnsupported,
        build_compiled_draft_chain,
    )

    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    with pytest.raises(CompiledDraftUnsupported, match="mutually exclusive"):
        build_compiled_draft_chain(
            rt=object(),
            mtp_cache=[],
            state_tree=[],
            mtp_hidden_variant="auto",
            selector=lambda *args: (),
            frspec_ids=None,
            depth=3,
            top_k=20,
            request_max_tokens=1024,
        )


# ---------------------------------------------------------------------------
# the gate's own contract
# ---------------------------------------------------------------------------
def test_gate_is_read_per_call_not_cached(monkeypatch):
    """The shadow harness arms a variant by scoping env around one call."""

    assert reuse_mod.indexer_reuse_enabled() is False
    monkeypatch.setenv(reuse_mod.ENV_FLAG, "1")
    assert reuse_mod.indexer_reuse_enabled() is True
    monkeypatch.setenv(reuse_mod.ENV_FLAG, "0")
    assert reuse_mod.indexer_reuse_enabled() is False


def test_draft_depth_scope_restores_and_nests():
    assert reuse_mod.current_draft_depth() is None
    with draft_depth_scope(1):
        assert reuse_mod.current_draft_depth() == 1
        with draft_depth_scope(None):
            assert reuse_mod.current_draft_depth() is None
        assert reuse_mod.current_draft_depth() == 1
    assert reuse_mod.current_draft_depth() is None
