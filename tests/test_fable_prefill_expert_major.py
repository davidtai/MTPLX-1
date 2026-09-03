"""MTPLX_FABLE_PREFILL_EXPERT_MAJOR -- planning, schedule, and exactness.

The lane claims a group of consecutive prefill chunks can be run **layer-major**
-- every chunk through layer L before any chunk enters layer L+1 -- so the
routed MoE grouped GEMM sees the group's rows in one call, and that the result
is **bit-identical** to today's chunk-major prefill.  Everything below runs on
tiny tensors on the CPU stream (the GPU on a development box is usually holding
a guarded benchmark), which is enough to settle:

* **the schedule.**  ``Qwen4ExpTextModel.forward_prefill_group`` over 4 chunks
  reproduces 4 successive ``TextModel.__call__`` chunks bitwise -- logits, the
  widened MTP stream, and every cache leaf (QSA keys/values/offset, the
  indexer's raw and pooled key banks, the GDN conv tape and recurrent state,
  and the PLE n-gram history ids).  The cache is the strong half: it proves
  the group preserved chunk order at every layer, including the PLE stage.
* **the seam.**  ``DecoderLayer.prefill_attn_half`` + ``prefill_moe_read`` +
  ``SparseMoeBlock.prefill_route``/``prefill_combine`` +
  ``prefill_moe_write``, composed, equal ``DecoderLayer.__call__`` bitwise.
  Those halves are copies of ``__call__``'s expressions; this is what keeps
  the copy honest.
* **the planner.**  Rows/expert arithmetic, the byte model, budget capping,
  span grouping (narrow tail spans and discontiguities end a group), and the
  boundary policy -- all pure python, no MLX.
* **the gating.**  Flag off is the pre-change path; an armed flag on a model
  or a request that cannot serve the schedule RAISES with the reason named
  rather than serving the control under the candidate's label.

What is NOT settled here, and needs the GPU: whether the wider GEMM is
*faster* (``scripts/fable/micro_moe_prefill_rows.py``), whether the group's
transient fits the wired limit (``scripts/fable/micro_prefill_memory_census.py``),
and bit-exactness of the *quantized* ``gather_qmm`` under a row-count change
-- the toy model is unquantized, so it exercises ``mx.gather_mm``.  The q4 path
has its own probe: ``micro_moe_prefill_rows.py --exactness``.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

import mtplx.fable_prefill_expert_major as expert_major
from mtplx.models.qwen4_exp import TextArgs, TextModel

# Chunk width x top_k must clear mlx_lm's ``do_sort = indices.size >= 64``
# threshold on a SINGLE chunk, or the group and the control would take
# different SwitchGLU routes and the comparison would measure the threshold
# rather than the schedule.  Production is never near it (20,480 per chunk).
CHUNK = 40
GROUP = 4
TOP_K = 2


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
    )


@pytest.fixture()
def tm():
    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    prev = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.set_default_device(mx.cpu)
    mx.random.seed(53)
    model = TextModel(_tiny_args())
    mx.eval(model.parameters())
    yield model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        expert_major.ENV_FLAG,
        expert_major.GROUP_ENV,
        expert_major.BUDGET_ENV,
        expert_major.MARGIN_ENV,
        expert_major.BOUNDARY_POLICY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    expert_major.reset_counters()
    yield
    expert_major.reset_counters()


def _chunks(seed: int = 7):
    mx.random.seed(seed)
    ids = mx.random.randint(0, 128, (1, CHUNK * GROUP))
    mx.eval(ids)
    return [ids[:, k * CHUNK : (k + 1) * CHUNK] for k in range(GROUP)]


def _leaf_pairs(cache):
    """Every comparable leaf of a per-layer cache, as (name, value)."""

    pairs = []
    for index, entry in enumerate(cache):
        for name in ("offset", "pooled_len"):
            if hasattr(entry, name):
                pairs.append((f"{index}.{name}", getattr(entry, name)))
        for name in ("raw_keys", "pooled", "pooled_f32_t"):
            if hasattr(entry, name):
                pairs.append((f"{index}.{name}", getattr(entry, name)))
        kv = getattr(entry, "kv", None)
        if kv is not None:
            pairs.append((f"{index}.keys", getattr(kv, "keys", None)))
            pairs.append((f"{index}.values", getattr(kv, "values", None)))
        if hasattr(entry, "__getitem__") and not hasattr(entry, "kv"):
            for slot in range(4):
                try:
                    pairs.append((f"{index}.state{slot}", entry[slot]))
                except (IndexError, KeyError, TypeError):
                    break
    return pairs


def _assert_same(expected, actual, label):
    assert type(expected) is type(actual), f"{label}: {type(expected)} vs {type(actual)}"
    if isinstance(expected, mx.array):
        assert expected.shape == actual.shape, label
        assert expected.dtype == actual.dtype, label
        assert bool(mx.array_equal(expected, actual).item()), label
    else:
        assert expected == actual, label


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


def test_group_forward_matches_chunk_major_outputs(tm):
    """4 chunks layer-major == 4 chunks chunk-major, bitwise, per chunk."""

    chunks = _chunks()

    control_cache = tm.make_cache()
    control = [
        tm(chunk, control_cache, None, return_hidden=True, emit_logits=True)
        for chunk in chunks
    ]
    mx.eval([leaf for pair in control for leaf in pair])

    group_cache = tm.make_cache()
    grouped = tm.forward_prefill_group(
        chunks, group_cache, return_hidden=True, emit_logits=True
    )
    mx.eval([leaf for pair in grouped for leaf in pair])

    assert len(grouped) == len(control) == GROUP
    for k, ((exp_logits, exp_hidden), (got_logits, got_hidden)) in enumerate(
        zip(control, grouped)
    ):
        _assert_same(exp_logits, got_logits, f"chunk{k}.logits")
        _assert_same(exp_hidden, got_hidden, f"chunk{k}.widened")


def test_group_forward_leaves_every_cache_leaf_identical(tm):
    """Chunk order inside each layer -- proved on the state, not the output.

    KV, the indexer's raw/pooled banks, the GDN conv tape and recurrent
    state, and the PLE n-gram history ids all advance once per chunk.  If the
    group had run a chunk out of order at any layer, or staged the PLE rows
    for the wrong ``prev``, this is where it would show.
    """

    chunks = _chunks(seed=11)

    control_cache = tm.make_cache()
    for chunk in chunks:
        mx.eval(tm(chunk, control_cache, None, return_hidden=True, emit_logits=False))

    group_cache = tm.make_cache()
    mx.eval(
        [
            leaf
            for pair in tm.forward_prefill_group(
                chunks, group_cache, return_hidden=True, emit_logits=False
            )
            for leaf in pair
            if leaf is not None
        ]
    )

    expected = _leaf_pairs(control_cache)
    actual = _leaf_pairs(group_cache)
    assert [name for name, _ in expected] == [name for name, _ in actual]
    compared = 0
    for (name, exp), (_, got) in zip(expected, actual):
        if exp is None and got is None:
            continue
        _assert_same(exp, got, name)
        compared += 1
    assert compared > 8, f"only {compared} leaves compared -- the walk missed the cache"


def test_group_of_one_is_the_chunk_major_path(tm):
    """G == 1 must be the control, not a degenerate special case."""

    chunk = _chunks(seed=3)[0]

    control_cache = tm.make_cache()
    exp_logits, exp_hidden = tm(
        chunk, control_cache, None, return_hidden=True, emit_logits=True
    )
    group_cache = tm.make_cache()
    (got_logits, got_hidden), = tm.forward_prefill_group(
        [chunk], group_cache, return_hidden=True, emit_logits=True
    )
    mx.eval(exp_logits, exp_hidden, got_logits, got_hidden)
    _assert_same(exp_logits, got_logits, "logits")
    _assert_same(exp_hidden, got_hidden, "widened")


def test_group_forward_honours_emit_logits(tm):
    """A cache-only group still returns the widened stream, no head matmul."""

    chunks = _chunks(seed=5)
    cache = tm.make_cache()
    out = tm.forward_prefill_group(
        chunks, cache, return_hidden=True, emit_logits=False
    )
    assert [logits for logits, _ in out] == [None] * GROUP
    assert all(hidden is not None for _, hidden in out)


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------


def test_layer_halves_compose_to_call(tm):
    """attn_half + moe_read + route/combine + moe_write == ``__call__``."""

    inner = tm.model
    layer = inner.layers[0]
    mx.random.seed(21)
    ids = mx.random.randint(0, 128, (1, CHUNK))
    hidden = mx.random.normal((1, CHUNK, inner.args.hidden_size * inner.args.hc_count))
    mx.eval(ids, hidden)

    call_cache = tm.make_cache()[0]
    expected = layer(hidden, input_ids=ids, ssm_mask=None, cache=call_cache)

    half_cache = tm.make_cache()[0]
    h = layer.prefill_attn_half(
        hidden, input_ids=ids, ssm_mask=None, cache=half_cache
    )
    mixed, hyper, inject = layer.prefill_moe_read(h)
    inds, scores = layer.mlp.prefill_route(mixed)
    routed = layer.mlp.switch_mlp(mixed, inds)
    got = layer.prefill_moe_write(
        hyper, layer.mlp.prefill_combine(routed, scores, mixed), inject
    )
    mx.eval(expected, got)
    _assert_same(expected, got, "layer halves")


def test_moe_halves_compose_to_moe_call(tm):
    """route -> switch_mlp -> combine == ``SparseMoeBlock.__call__``."""

    inner = tm.model
    mlp = inner.layers[0].mlp
    mx.random.seed(31)
    x = mx.random.normal((1, CHUNK, inner.args.hidden_size))
    mx.eval(x)

    expected = mlp(x)
    inds, scores = mlp.prefill_route(x)
    got = mlp.prefill_combine(mlp.switch_mlp(x, inds), scores, x)
    mx.eval(expected, got)
    _assert_same(expected, got, "moe halves")


def test_batched_switch_mlp_matches_per_chunk(tm):
    """The one call whose M changes: same rows, one batch or four."""

    inner = tm.model
    mlp = inner.layers[0].mlp
    mx.random.seed(41)
    xs = [mx.random.normal((1, CHUNK, inner.args.hidden_size)) for _ in range(GROUP)]
    mx.eval(xs)
    routes = [mlp.prefill_route(x) for x in xs]

    per_chunk = [mlp.switch_mlp(x, inds) for x, (inds, _) in zip(xs, routes)]
    batched = mlp.switch_mlp(
        mx.concatenate(xs, axis=1),
        mx.concatenate([inds for inds, _ in routes], axis=1),
    )
    mx.eval(per_chunk, batched)
    for k in range(GROUP):
        _assert_same(
            per_chunk[k], batched[:, k * CHUNK : (k + 1) * CHUNK], f"routed chunk{k}"
        )


# ---------------------------------------------------------------------------
# the gating
# ---------------------------------------------------------------------------


def test_group_forward_refuses_a_vision_splice(tm):
    with pytest.raises(expert_major.ExpertMajorRefusal, match="vision"):
        tm.model.forward_prefill_group(
            _chunks(), tm.make_cache(), input_embeddings=mx.zeros((1, CHUNK, 64))
        )


def test_group_forward_refuses_without_a_cache(tm):
    with pytest.raises(expert_major.ExpertMajorRefusal, match="cache"):
        tm.model.forward_prefill_group(_chunks(), None)


def test_group_forward_rejects_an_empty_group(tm):
    with pytest.raises(ValueError):
        tm.model.forward_prefill_group([], tm.make_cache())


# ---------------------------------------------------------------------------
# the planner (no MLX)
# ---------------------------------------------------------------------------


def test_flag_is_off_by_default():
    assert expert_major.enabled({}) is False
    assert expert_major.enabled({expert_major.ENV_FLAG: "0"}) is False
    assert expert_major.enabled({expert_major.ENV_FLAG: "1"}) is True


def test_rows_per_expert_matches_the_served_arithmetic():
    # 2,048 x 10 / 512 = 40 today; 4 x 4,096 gives B1's 320.
    assert expert_major.rows_per_expert(2048, 1, 10, 512) == 40
    assert expert_major.rows_per_expert(4096, 1, 10, 512) == 80
    assert expert_major.rows_per_expert(4096, 4, 10, 512) == 320
    assert expert_major.rows_per_expert(2048, 8, 10, 512) == 320


def test_routed_transient_matches_the_documented_per_row_bytes():
    per_row = expert_major.routed_transient_bytes_per_row(
        hidden=2560, moe_intermediate=640, top_k=10
    )
    assert per_row == expert_major.ROUTED_TRANSIENT_BYTES_PER_ROW == 192_000
    assert (
        expert_major.hidden_bytes_per_row(hidden=2560, hc_count=4)
        == expert_major.HIDDEN_BYTES_PER_ROW
        == 20_480
    )


def test_group_bytes_delta_is_only_the_extra_chunks():
    one = expert_major.group_bytes(chunk_rows=4096, group=1)
    assert one["delta_bytes"] == 0
    four = expert_major.group_bytes(chunk_rows=4096, group=4)
    assert four["group_rows"] == 16384
    # (G-1) x chunk x (192,000 + 20,480) = 3 x 4096 x 212,480
    assert four["delta_bytes"] == 3 * 4096 * 212_480
    assert 2.5e9 < four["delta_bytes"] < 2.7e9


def test_max_group_for_budget_is_monotone_and_never_below_one():
    per_chunk = 4096 * 212_480
    assert expert_major.max_group_for_budget(chunk_rows=4096, headroom_bytes=0) == 1
    assert (
        expert_major.max_group_for_budget(chunk_rows=4096, headroom_bytes=-(1 << 40))
        == 1
    )
    assert (
        expert_major.max_group_for_budget(
            chunk_rows=4096, headroom_bytes=3 * per_chunk
        )
        == 4
    )
    assert (
        expert_major.max_group_for_budget(
            chunk_rows=4096, headroom_bytes=3 * per_chunk - 1
        )
        == 3
    )


def _plan(**kw):
    base = dict(
        chunk_rows=4096,
        top_k=10,
        num_experts=512,
        hidden=2560,
        moe_intermediate=640,
        hc_count=4,
        environ={},
    )
    base.update(kw)
    return expert_major.plan_group(**base)


def test_plan_caps_the_group_at_the_budget_instead_of_refusing():
    """An over-budget request degrades to a smaller group, then to G=1."""

    unbounded = _plan(group=4)
    assert unbounded.group == 4
    assert unbounded.rows_per_expert == 320
    assert unbounded.headroom_bytes is None

    # 92.22 GB resident (the measured chunk-4096 peak) against a 100 GiB cap.
    plan = _plan(
        group=4,
        budget_bytes=100 * 1024**3,
        resident_bytes=92_219_331_638,
        margin_bytes=2 * 1024**3,
    )
    assert plan.group == 4, plan.as_receipt()
    assert plan.headroom_bytes > 0

    # the same peak against a 90 GiB wired limit (96.64 GB) leaves 2.27 GB
    # after the margin -- two extra chunks at 870 MB each, so G drops to 3.
    tight = _plan(
        group=4,
        budget_bytes=90 * 1024**3,
        resident_bytes=92_219_331_638,
        margin_bytes=2 * 1024**3,
    )
    assert tight.group == 3, tight.as_receipt()
    assert tight.requested_group == 4
    assert tight.headroom_bytes >= 0

    # a peak that already fills the limit degrades all the way to chunk-major
    full = _plan(
        group=4,
        budget_bytes=90 * 1024**3,
        resident_bytes=95_000_000_000,
        margin_bytes=2 * 1024**3,
    )
    assert full.group == 1
    assert full.engaged is False
    assert full.requested_group == 4


def test_plan_receipt_carries_every_decided_number():
    receipt = _plan(group=2, budget_bytes=100 * 1024**3).as_receipt()
    for key in (
        "group",
        "requested_group",
        "group_rows",
        "rows_per_expert",
        "chunk_major_rows_per_expert",
        "delta_bytes",
        "headroom_bytes",
        "boundary_policy",
        "engaged",
    ):
        assert key in receipt, key
    assert receipt["chunk_major_rows_per_expert"] == 80
    assert receipt["rows_per_expert"] == 160


def test_group_spans_groups_full_width_runs():
    spans = [(i * 4096, (i + 1) * 4096) for i in range(8)]
    groups = expert_major.group_spans(spans, 4, chunk_rows=4096)
    assert [len(g) for g in groups] == [4, 4]
    assert groups[0][0] == (0, 4096)
    assert groups[1][-1] == (28672, 32768)


def test_group_spans_isolates_narrow_tail_spans():
    """The GDN boundary tail grid must not be swallowed into a group."""

    spans = [(0, 4096), (4096, 8192), (8192, 8448), (8448, 8704)]
    groups = expert_major.group_spans(spans, 4, chunk_rows=4096)
    assert [len(g) for g in groups] == [2, 1, 1]
    assert groups[1] == [(8192, 8448)]


def test_group_spans_never_bridges_a_gap():
    spans = [(0, 4096), (4096, 8192), (12288, 16384)]
    groups = expert_major.group_spans(spans, 4, chunk_rows=4096)
    assert [len(g) for g in groups] == [2, 1]


def test_group_spans_of_one_is_one_span_each():
    spans = [(0, 4096), (4096, 8192), (8192, 8448)]
    assert expert_major.group_spans(spans, 1) == [[s] for s in spans]


def test_boundary_policy_rejects_an_unknown_value():
    assert expert_major.boundary_policy({}) == "refuse"
    assert (
        expert_major.boundary_policy({expert_major.BOUNDARY_POLICY_ENV: "group"})
        == "group"
    )
    with pytest.raises(expert_major.ExpertMajorRefusal):
        expert_major.boundary_policy({expert_major.BOUNDARY_POLICY_ENV: "maybe"})


def test_counters_are_recorded_and_resettable():
    expert_major.count("groups", 3)
    expert_major.count("groups")
    assert expert_major.snapshot_counters()["groups"] == 4
    expert_major.reset_counters()
    assert expert_major.snapshot_counters() == {}


# ---------------------------------------------------------------------------
# the generation-loop gate
# ---------------------------------------------------------------------------


def test_generation_grouping_is_inert_when_the_flag_is_off():
    from mtplx.generation import _expert_major_groups

    spans = [(0, 4096), (4096, 8192)]
    groups, plan = _expert_major_groups(
        None, spans, chunk_size=4096, capture_boundaries=False, vision_splice=None
    )
    assert plan is None
    assert groups == [[s] for s in spans]
    assert expert_major.snapshot_counters() == {}


def test_generation_grouping_keeps_boundaries_by_default(monkeypatch):
    """A boundary-capturing prefill runs chunk-major and says so."""

    from mtplx.generation import _expert_major_groups

    monkeypatch.setenv(expert_major.ENV_FLAG, "1")
    spans = [(0, 4096), (4096, 8192)]
    groups, plan = _expert_major_groups(
        None, spans, chunk_size=4096, capture_boundaries=True, vision_splice=None
    )
    assert plan is None
    assert groups == [[s] for s in spans]
    assert expert_major.snapshot_counters()["refused_boundary_capture"] == 1


def test_generation_grouping_refuses_a_vision_splice(monkeypatch):
    from mtplx.generation import _expert_major_groups

    monkeypatch.setenv(expert_major.ENV_FLAG, "1")
    with pytest.raises(expert_major.ExpertMajorRefusal, match="vision"):
        _expert_major_groups(
            None,
            [(0, 4096)],
            chunk_size=4096,
            capture_boundaries=False,
            vision_splice=object(),
        )
