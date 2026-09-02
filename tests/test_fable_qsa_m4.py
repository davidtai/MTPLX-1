"""MTPLX_FABLE_QSA_M4 -- gating, equivalence, and the dispatch-count receipt.

The lane replaces four QSA sub-chains with Metal kernels, so the kernels
themselves cannot be executed here (this suite runs on the CPU stream, and
the GPU on a development box is usually holding a guarded benchmark).  What
IS provable off-GPU, and is proved below:

* the ALGEBRA of each kernel.  Every kernel's Metal body is a closed-form
  expression; each is re-expressed here as an MLX/CPU reference with the same
  index arithmetic and the same operation order, and pinned against the stock
  chain it replaces.  For ``row_tokens`` and ``index_scores`` that pin is
  bitwise equality on random inputs -- if the Metal source and the reference
  ever disagree the microbench's differing-element count will say so, but the
  formula itself is settled here.
* the GATING.  Flag off must reproduce the pre-change code path exactly, an
  armed flag on an ineligible pack must RAISE rather than revert quietly, and
  a fixed cache's S=1 route must keep the stock chain.
* the DISPATCH COUNT.  ``test_dispatch_map_before_after`` builds each stock
  sub-chain under ``mx.compile`` on the CPU stream and counts its primitives,
  which is the before/after receipt the lane exists for.

Everything here uses tiny tensors on the CPU stream.
"""

from __future__ import annotations

import io
import math
import re

import mlx.core as mx
import pytest

import mtplx.models.qwen4_exp as qwen4_exp
import mtplx.runtime_options as runtime_options
from mtplx.kernels import qwen4_qsa_m4_indexer as m4


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


RATIO = 4
ROWS = 4
IDX_HEAD_DIM = 128
ROTARY_DIM = 64
BLOCKS = 64
TOPK = 8


def _inv_freq(rotary_dim: int, theta: float = 10_000_000.0) -> mx.array:
    return 1.0 / (
        theta ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
    )


def _indexer(**overrides):
    fields = dict(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=IDX_HEAD_DIM,
        indexer_budget=TOPK * RATIO,
        indexer_compress_ratio=RATIO,
    )
    fields.update(overrides)
    args = qwen4_exp.TextArgs(**fields)
    indexer = qwen4_exp.QSAIndexer(args)
    dtype = mx.bfloat16
    indexer.update(
        {
            "q_layernorm": {
                "weight": mx.random.uniform(
                    low=0.5, high=1.5, shape=(IDX_HEAD_DIM,)
                ).astype(dtype)
            },
            "k_layernorm": {
                "weight": mx.random.uniform(
                    low=0.5, high=1.5, shape=(IDX_HEAD_DIM,)
                ).astype(dtype)
            },
        }
    )
    return indexer


class _FixedCache:
    """The surface the fixed lane reads off an installed QSA cache."""

    fixed_capacity = True

    def __init__(self, raw_keys, pooled, offset, *, ratio=RATIO, rows=ROWS,
                 m4_rows=0):
        self.raw_keys = raw_keys
        self.pooled = pooled
        self.offset = offset
        self.ratio = ratio
        self._last_write_rows = rows
        self.fable_qsa_m4_rows = m4_rows

    def pooled_f32_view(self, nb):
        return mx.swapaxes(self.pooled.astype(mx.float32), 1, 2)[:, None, :, :nb]


def _bank(blocks=BLOCKS, dtype=mx.bfloat16):
    raw = mx.random.normal((1, blocks * RATIO, IDX_HEAD_DIM)).astype(dtype)
    pooled = mx.random.normal((1, blocks, IDX_HEAD_DIM)).astype(dtype)
    mx.eval(raw, pooled)
    return raw, pooled


# ---------------------------------------------------------------------------
# 1 -- the flag itself
# ---------------------------------------------------------------------------
def test_flag_defaults_off_and_is_read_once(monkeypatch):
    assert runtime_options.env_bool("MTPLX_FABLE_QSA_M4", default=False) is False
    monkeypatch.setenv("MTPLX_FABLE_QSA_M4", "1")
    # The module-level value is a snapshot taken at import: a mid-run env
    # change must not make two traces of one graph disagree.
    assert runtime_options.fable_qsa_m4_enabled() is runtime_options._FABLE_QSA_M4
    assert runtime_options.env_bool("MTPLX_FABLE_QSA_M4", default=False) is True


def test_flag_rejects_a_non_boolean_spelling(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_QSA_M4", "maybe")
    with pytest.raises(ValueError):
        runtime_options.env_bool("MTPLX_FABLE_QSA_M4", default=False)


# ---------------------------------------------------------------------------
# 2 -- routing: flag off is the pre-change path, and narrowing is explicit
# ---------------------------------------------------------------------------
def _arm(monkeypatch, on=True):
    monkeypatch.setattr(runtime_options, "_FABLE_QSA_M4", on)


def test_route_is_off_without_the_flag(monkeypatch):
    _arm(monkeypatch, False)
    raw, pooled = _bank()
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    assert _indexer()._m4_route(cache, 4) is False


def test_route_is_off_when_the_cache_did_not_pass_construction(monkeypatch):
    _arm(monkeypatch)
    raw, pooled = _bank()
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=0)
    assert _indexer()._m4_route(cache, 4) is False


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 8])
def test_route_narrows_to_the_verify_width(monkeypatch, rows):
    # A fixed cache also serves the S=1 D3 route; the kernels are wired for 4
    # rows, so other widths keep the stock chain. This is the ONE non-raising
    # narrowing in the lane.
    _arm(monkeypatch)
    raw, pooled = _bank()
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    assert _indexer()._m4_route(cache, rows) is False
    assert _indexer()._m4_route(cache, 4) is True


def test_flag_off_never_reaches_the_kernel(monkeypatch):
    """Default ``fused_m4=False`` must leave _extend_pooled_fixed untouched."""

    def explode(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("the M4 pooled-row kernel ran with the flag off")

    monkeypatch.setattr(m4, "qsa_m4_pooled_row", explode)
    _arm(monkeypatch, False)
    indexer = _indexer()
    raw, pooled = _bank()
    offset = mx.array(BLOCKS // 2 * RATIO, dtype=mx.int32)
    cache = _FixedCache(raw, pooled, offset)
    out = indexer._extend_pooled_fixed(cache, offset + ROWS)
    mx.eval(out)
    assert out.shape == pooled.shape


def test_flag_off_values_are_unchanged_by_the_new_keyword(monkeypatch):
    _arm(monkeypatch, False)
    indexer = _indexer()
    raw, pooled = _bank()
    offset = mx.array(BLOCKS // 2 * RATIO, dtype=mx.int32)
    a = indexer._extend_pooled_fixed(_FixedCache(raw, pooled, offset), offset + ROWS)
    b = indexer._extend_pooled(_FixedCache(raw, pooled, offset), offset + ROWS)
    mx.eval(a, b)
    assert mx.array_equal(a, b).item()


# ---------------------------------------------------------------------------
# 3 -- eligibility raises; it never reverts
# ---------------------------------------------------------------------------
def test_contract_accepts_the_production_geometry(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    raw, pooled = _bank(blocks=max(BLOCKS, indexer.block_topk))
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    indexer._require_m4_contract(cache, 4)  # must not raise


def test_contract_rejects_a_multi_kv_head_indexer(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    indexer.kv_heads = 2
    raw, pooled = _bank()
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    with pytest.raises(RuntimeError, match="single indexer KV head"):
        indexer._require_m4_contract(cache, 4)


def test_contract_rejects_a_head_dim_past_the_exact_rms_regime(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    indexer.head_dim = m4.MAX_EXACT_HEAD_DIM + 1
    raw, pooled = _bank()
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    with pytest.raises(RuntimeError, match="rms_single_row"):
        indexer._require_m4_contract(cache, 4)


def test_contract_rejects_a_step_that_completes_more_than_one_block(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    raw, pooled = _bank()
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), rows=8,
                        m4_rows=4)
    with pytest.raises(RuntimeError, match="at most one"):
        indexer._require_m4_contract(cache, 4)


def test_contract_rejects_a_bank_shallower_than_the_top_k(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    raw, pooled = _bank(blocks=indexer.block_topk - 1)
    cache = _FixedCache(raw, pooled, mx.array(4, dtype=mx.int32), m4_rows=4)
    with pytest.raises(RuntimeError, match="block_topk"):
        indexer._require_m4_contract(cache, 4)


def test_contract_rejects_a_mismatched_norm_dtype(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    raw, pooled = _bank(blocks=max(BLOCKS, indexer.block_topk))
    indexer.q_layernorm.weight = indexer.q_layernorm.weight.astype(mx.float32)
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    with pytest.raises(RuntimeError, match="query norm weight"):
        indexer._require_m4_contract(cache, 4)


def test_contract_rejects_an_fp32_bank(monkeypatch):
    _arm(monkeypatch)
    indexer = _indexer()
    raw, pooled = _bank(blocks=max(BLOCKS, indexer.block_topk), dtype=mx.float32)
    indexer.q_layernorm.weight = indexer.q_layernorm.weight.astype(mx.float32)
    indexer.k_layernorm.weight = indexer.k_layernorm.weight.astype(mx.float32)
    cache = _FixedCache(raw, pooled, mx.array(64, dtype=mx.int32), m4_rows=4)
    with pytest.raises(TypeError, match="float16/bfloat16"):
        indexer._require_m4_contract(cache, 4)


# ---------------------------------------------------------------------------
# 4 -- kernel shape validators
# ---------------------------------------------------------------------------
def test_row_tokens_validator():
    with pytest.raises(ValueError, match=r"\[rows,topk\]"):
        m4.check_row_tokens_shapes(mx.zeros((4,), dtype=mx.uint32))
    with pytest.raises(TypeError, match="uint32 or int32"):
        m4.check_row_tokens_shapes(mx.zeros((4, 8), dtype=mx.float32))
    assert m4.check_row_tokens_shapes(mx.zeros((4, 8), dtype=mx.uint32)) == (4, 8)


def test_index_scores_validator():
    with pytest.raises(ValueError, match="rows,heads,blocks"):
        m4.check_index_scores_shapes(mx.zeros((4, 4, 8), dtype=mx.float32))
    with pytest.raises(TypeError, match="fp32 reduce"):
        m4.check_index_scores_shapes(mx.zeros((1, 4, 4, 8), dtype=mx.bfloat16))
    assert m4.check_index_scores_shapes(
        mx.zeros((1, 4, 4, 8), dtype=mx.float32)
    ) == (4, 4, 8)


def test_pooled_row_validator():
    raw, pooled = _bank()
    weight = mx.ones((IDX_HEAD_DIM,), dtype=mx.bfloat16)
    inv = _inv_freq(ROTARY_DIM)
    assert m4.check_pooled_row_shapes(
        raw, pooled, weight, inv, compress_ratio=RATIO
    ) == IDX_HEAD_DIM
    with pytest.raises(TypeError, match="1-D float32"):
        m4.check_pooled_row_shapes(
            raw, pooled, weight, inv.astype(mx.float16), compress_ratio=RATIO
        )
    with pytest.raises(ValueError, match="norm_weight must be"):
        m4.check_pooled_row_shapes(
            raw,
            pooled,
            mx.ones((IDX_HEAD_DIM // 2,), dtype=mx.bfloat16),
            inv,
            compress_ratio=RATIO,
        )
    with pytest.raises(ValueError, match="head_dim must be in"):
        wide = mx.zeros((1, RATIO, 2 * m4.MAX_EXACT_HEAD_DIM), dtype=mx.bfloat16)
        m4.check_pooled_row_shapes(
            wide,
            mx.zeros((1, 1, 2 * m4.MAX_EXACT_HEAD_DIM), dtype=mx.bfloat16),
            mx.ones((2 * m4.MAX_EXACT_HEAD_DIM,), dtype=mx.bfloat16),
            inv,
            compress_ratio=RATIO,
        )


def test_f32_literal_rounds_like_mlx_weak_promotion():
    for value in (1e-12, math.sqrt(128.0), math.sqrt(256.0)):
        literal = m4._f32_literal(value)
        assert literal.endswith("f")
        narrowed = float(literal[:-1])
        # Same bits MLX gets when it narrows the python double.
        assert (
            mx.array(narrowed, dtype=mx.float32).item()
            == mx.array(value, dtype=mx.float32).item()
        )


# ---------------------------------------------------------------------------
# 5 -- ALGEBRA: CPU references of the Metal bodies vs the stock chains
# ---------------------------------------------------------------------------
def _row_tokens_reference(top_idx, offset, ratio, rows, topk):
    """The ``_SRC_ROW_TOKENS`` body, element for element."""

    width = topk * ratio + ratio
    idx = []
    ok = []
    off = int(offset.item()) if isinstance(offset, mx.array) else int(offset)
    tops = top_idx.tolist()
    for s in range(rows):
        qpos = off + s
        visible = (qpos + 1) // ratio
        row_idx, row_ok = [], []
        for c in range(width):
            if c < topk * ratio:
                block = int(tops[s][c // ratio])
                token = block * ratio + (c % ratio)
                good = block < visible
            else:
                token = visible * ratio + (c - topk * ratio)
                good = token <= qpos
            row_ok.append(good)
            row_idx.append(token if good else 0)
        idx.append(row_idx)
        ok.append(row_ok)
    return mx.array(idx, dtype=mx.int32), mx.array(ok, dtype=mx.bool_)


def _row_tokens_stock(top_idx, offset, ratio, rows, blocks):
    qpos = offset + mx.arange(rows, dtype=mx.int32)
    nb_q = (qpos + 1) // ratio
    blk = mx.arange(blocks, dtype=mx.int32)
    valid = blk[None, :] < nb_q[:, None]
    blk_ok = mx.take_along_axis(valid, top_idx.astype(mx.int64), axis=-1)
    tok_blocks = (
        top_idx.astype(mx.int32)[:, :, None] * ratio
        + mx.arange(ratio, dtype=mx.int32)
    ).reshape(rows, -1)
    blocks_ok = mx.repeat(blk_ok, ratio, axis=1)
    tail_tok = nb_q[:, None] * ratio + mx.arange(ratio, dtype=mx.int32)
    tail_ok = tail_tok <= qpos[:, None]
    token_idx = mx.concatenate([tok_blocks, tail_tok], axis=1)
    token_ok = mx.concatenate([blocks_ok, tail_ok], axis=1)
    token_idx = mx.where(token_ok, token_idx, mx.array(0, dtype=mx.int32))
    return token_idx, token_ok


@pytest.mark.parametrize("offset", [0, 3, 4, 17, 60, 252, 1024])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_row_tokens_algebra_is_bit_exact(offset, seed):
    mx.random.seed(seed)
    top_idx = (
        mx.random.uniform(shape=(ROWS, TOPK)) * BLOCKS
    ).astype(mx.uint32)
    mx.eval(top_idx)
    off = mx.array(offset, dtype=mx.int32)
    want_idx, want_ok = _row_tokens_stock(top_idx, off, RATIO, ROWS, BLOCKS)
    got_idx, got_ok = _row_tokens_reference(top_idx, off, RATIO, ROWS, TOPK)
    mx.eval(want_idx, want_ok, got_idx, got_ok)
    assert mx.array_equal(got_idx, want_idx).item()
    assert mx.array_equal(got_ok, want_ok).item()
    # The width the fused K/V gather is compiled for.
    assert int(got_idx.shape[1]) == TOPK * RATIO + RATIO


def _index_scores_reference(scores, offset, ratio, head_dim, tie=1e-12):
    """The ``_SRC_INDEX_SCORES`` body, in the kernel's operation order."""

    rows, heads, blocks = (int(s) for s in scores.shape[1:])
    off = int(offset.item()) if isinstance(offset, mx.array) else int(offset)
    flat = scores.reshape(rows, heads, blocks)
    out = []
    divisor = mx.array(math.sqrt(head_dim), dtype=mx.float32)
    step = mx.array(tie, dtype=mx.float32)
    neg = mx.array(-mx.inf, dtype=mx.float32)
    blk = mx.arange(blocks, dtype=mx.float32)
    for s in range(rows):
        acc = mx.maximum(flat[s, 0], 0.0)
        for h in range(1, heads):
            acc = acc + mx.maximum(flat[s, h], 0.0)
        value = acc / divisor
        visible = (off + s + 1) // ratio
        keep = mx.arange(blocks, dtype=mx.int32) < visible
        out.append(mx.where(keep, value, neg) - blk * step)
    return mx.stack(out, axis=0)


def _index_scores_stock(scores, offset, ratio, head_dim, blocks, rows):
    qpos = offset + mx.arange(rows, dtype=mx.int32)
    nb_q = (qpos + 1) // ratio
    blk = mx.arange(blocks, dtype=mx.int32)
    valid = blk[None, :] < nb_q[:, None]
    neg = mx.array(-mx.inf, dtype=mx.float32)
    s = mx.maximum(scores, 0.0).sum(axis=2) / math.sqrt(head_dim)
    s = s[0]
    s = mx.where(valid, s, neg)
    return s - blk.astype(mx.float32)[None, :] * 1e-12


@pytest.mark.parametrize("offset", [0, 7, 64, 4096])
@pytest.mark.parametrize("seed", [0, 1])
def test_index_scores_algebra_is_bit_exact(offset, seed):
    mx.random.seed(seed)
    # Centred on zero so roughly half the head scores are relu'd to exactly
    # 0.0 -- the tie regime the -blk*1e-12 term exists for.
    scores = mx.random.normal((1, ROWS, 4, BLOCKS)).astype(mx.float32)
    mx.eval(scores)
    off = mx.array(offset, dtype=mx.int32)
    want = _index_scores_stock(scores, off, RATIO, IDX_HEAD_DIM, BLOCKS, ROWS)
    got = _index_scores_reference(scores, off, RATIO, IDX_HEAD_DIM)
    mx.eval(want, got)
    # -inf on both sides in the masked region; compare those slots as equal.
    both_inf = mx.isinf(want) & mx.isinf(got) & (mx.sign(want) == mx.sign(got))
    diff = mx.where(both_inf, mx.zeros_like(want), mx.abs(want - got))
    assert float(mx.max(diff).item()) == 0.0


def test_index_scores_algebra_masks_exactly_the_stock_visible_set():
    scores = mx.ones((1, ROWS, 4, BLOCKS), dtype=mx.float32)
    off = mx.array(41, dtype=mx.int32)
    want = _index_scores_stock(scores, off, RATIO, IDX_HEAD_DIM, BLOCKS, ROWS)
    got = _index_scores_reference(scores, off, RATIO, IDX_HEAD_DIM)
    mx.eval(want, got)
    assert mx.array_equal(mx.isinf(want), mx.isinf(got)).item()


def _pooled_row_reference(raw, pooled, weight, inv, offset, ratio, step_rows,
                          eps):
    """The ``_SRC_POOLED_ROW`` body, in the kernel's operation order."""

    cap = int(pooled.shape[1])
    off = int(offset.item())
    block_raw = off // ratio
    block = min(block_raw, cap - 1)
    write = ((off + step_rows) // ratio) > block_raw
    fresh = raw[:, block * ratio : (block + 1) * ratio, :]
    fresh = fresh.reshape(1, 1, ratio, int(raw.shape[2]))
    mean = mx.mean(fresh.astype(mx.float32), axis=2).astype(raw.dtype)
    inv_rms = mx.rsqrt(
        mx.mean(mean.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps
    )
    normed = (mean.astype(mx.float32) * inv_rms).astype(raw.dtype) * weight
    rot = 2 * int(inv.shape[0])
    half = rot // 2
    theta = float(block * ratio) * inv
    cos, sin = mx.cos(theta), mx.sin(theta)
    x1 = normed[..., :half].astype(mx.float32)
    x2 = normed[..., half:rot].astype(mx.float32)
    lo = (x1 * cos - x2 * sin).astype(raw.dtype)
    hi = (x2 * cos + x1 * sin).astype(raw.dtype)
    candidate = mx.concatenate([lo, hi, normed[..., rot:]], axis=-1)
    old = pooled[:, block : block + 1, :]
    merged = candidate if write else old
    return mx.slice_update(pooled, merged, mx.array(block, dtype=mx.int32),
                           axes=(1,)), block, write


@pytest.mark.parametrize("offset_blocks,step", [(8, 4), (0, 4), (31, 4)])
def test_pooled_row_reference_matches_the_stock_bank_write(offset_blocks, step):
    """Same bank contents, same target row, same write condition.

    Values are compared with a tolerance: ``mx.fast.rms_norm``'s rsqrt is not
    reproducible from an MLX expression on the CPU stream, and the Metal
    kernel's bitwise claim is inherited from the shipped pool-keys kernel
    (tests/test_qsa_indexer_prepare_metal.py) and re-checked by
    scripts/fable/micro_qsa_m4.py.  What is asserted exactly here is the part
    this module actually adds: which row is written, and when.
    """

    mx.random.seed(7)
    indexer = _indexer()
    raw, pooled = _bank()
    offset = mx.array(offset_blocks * RATIO, dtype=mx.int32)
    cache = _FixedCache(raw, pooled, offset, rows=step)
    stock = indexer._extend_pooled_fixed(cache, offset + step)
    ref, block, write = _pooled_row_reference(
        raw, pooled, indexer.k_layernorm.weight, indexer._inv_freq,
        offset, RATIO, step, indexer.rms_norm_eps,
    )
    mx.eval(stock, ref)
    assert write is True
    assert block == offset_blocks
    # Every untouched row must be byte-identical.
    keep = [i for i in range(int(pooled.shape[1])) if i != block]
    assert mx.array_equal(stock[:, keep], ref[:, keep]).item()
    written = mx.abs(
        stock[:, block].astype(mx.float32) - ref[:, block].astype(mx.float32)
    )
    assert float(mx.max(written).item()) < 5e-2


def test_pooled_row_reference_holds_the_bank_when_no_block_completes():
    """ratio 8 with a 4-row step: three steps of four write nothing."""

    indexer = _indexer(indexer_compress_ratio=8)
    raw = mx.random.normal((1, BLOCKS * 8, IDX_HEAD_DIM)).astype(mx.bfloat16)
    pooled = mx.random.normal((1, BLOCKS, IDX_HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(raw, pooled)
    offset = mx.array(0, dtype=mx.int32)   # block 0 not yet complete
    ref, block, write = _pooled_row_reference(
        raw, pooled, indexer.k_layernorm.weight, indexer._inv_freq,
        offset, 8, 4, indexer.rms_norm_eps,
    )
    mx.eval(ref)
    assert write is False and block == 0
    assert mx.array_equal(ref, pooled).item()


# ---------------------------------------------------------------------------
# 5b -- the construction-time gate in graphbank
# ---------------------------------------------------------------------------
class _EagerQSAEntry:
    """The surface TensorOffsetQSACache.from_qsa_cache reads."""

    class _KV:
        step = 256

        def __init__(self, cap, dtype):
            self.keys = mx.zeros((1, 2, cap, 256), dtype)
            self.values = mx.zeros((1, 2, cap, 256), dtype)
            self.offset = cap // 2

    def __init__(self, *, ratio=4, cap=1024, dtype=mx.bfloat16):
        self.ratio = ratio
        self.kv = self._KV(cap, dtype)
        self.offset = self.kv.offset
        self.raw_keys = mx.zeros((1, cap, IDX_HEAD_DIM), dtype)
        self.pooled = mx.zeros((1, cap // ratio, IDX_HEAD_DIM), dtype)
        self.rows_gather_kv_m4 = None


def _install(monkeypatch, *, armed, fused_gather, ratio=4):
    import mtplx.graphbank as graphbank

    monkeypatch.setattr(graphbank, "fable_qsa_m4_enabled", lambda: armed)
    monkeypatch.setattr(
        graphbank, "_env_enabled",
        lambda name: fused_gather if name == "MTPLX_QSA_M4_FUSED_KV_GATHER" else True,
    )
    entry = _EagerQSAEntry(ratio=ratio)
    return graphbank.TensorOffsetQSACache.from_qsa_cache(entry, reserve_tokens=1024)


def test_graphbank_gate_refuses_an_armed_flag_without_the_fused_gather(monkeypatch):
    with pytest.raises(RuntimeError, match="MTPLX_QSA_M4_FUSED_KV_GATHER"):
        _install(monkeypatch, armed=True, fused_gather=False)


def test_graphbank_gate_refuses_a_non_ratio_4_lane(monkeypatch):
    with pytest.raises(RuntimeError, match="ratio-4"):
        _install(monkeypatch, armed=True, fused_gather=True, ratio=2)


def test_graphbank_leaves_the_lane_dark_when_the_flag_is_off(monkeypatch):
    cache = _install(monkeypatch, armed=False, fused_gather=False)
    assert cache.fable_qsa_m4 is False
    assert cache.fable_qsa_m4_rows == 0


# ---------------------------------------------------------------------------
# 6 -- the transposed-key gather contract
# ---------------------------------------------------------------------------
def test_gather_binding_advertises_its_key_layout():
    from mtplx.kernels import qwen4_qsa_m4_fused_kv_gather as g

    stock = g.bind_qwen4_qsa_m4_fused_kv_gather(capacity=16_384)
    kt = g.bind_qwen4_qsa_m4_fused_kv_gather(
        capacity=16_384, transposed_keys=True
    )
    assert stock.keys_transposed is False
    assert kt.keys_transposed is True
    assert g._SELECTED == 512 * 4 + 4
    assert g._KEYS_T_SHAPE == (1, g._H_KV, g._ROWS, g._HEAD_DIM, g._SELECTED)
    # Every selected token must be covered by the tiled transpose.
    assert g._TOKEN_TILES * g._TILE >= g._SELECTED
    assert g._DIM_TILES * g._TILE == g._HEAD_DIM


def test_rows_gather_attention_is_identical_under_either_key_layout():
    """The transposed gather is data movement only: same numbers, same order."""

    rows, h_kv, rep, head_dim, sel = 4, 2, 3, 8, 6
    heads = h_kv * rep
    mx.random.seed(3)
    q = mx.random.normal((1, heads, rows, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, h_kv, 32, head_dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, h_kv, 32, head_dim)).astype(mx.bfloat16)
    token_idx = (mx.random.uniform(shape=(rows, sel)) * 32).astype(mx.int32)
    token_ok = mx.random.uniform(shape=(rows, sel)) > 0.25
    mx.eval(q, k, v, token_idx, token_ok)

    def plain(keys, values, idx):
        return qwen4_exp._qsa_stock_rows_gather_kv(keys, values, idx)

    def transposed(keys, values, idx):
        ks, vs = qwen4_exp._qsa_stock_rows_gather_kv(keys, values, idx)
        return mx.contiguous(mx.swapaxes(ks, -1, -2)), vs

    transposed.keys_transposed = True

    a = qwen4_exp._qsa_rows_gather_attention(
        q, k, v, token_idx, token_ok, head_dim**-0.5, plain
    )
    b = qwen4_exp._qsa_rows_gather_attention(
        q, k, v, token_idx, token_ok, head_dim**-0.5, transposed
    )
    mx.eval(a, b)
    assert a.shape == b.shape
    assert mx.array_equal(a, b).item()


# ---------------------------------------------------------------------------
# 7 -- the dispatch-count receipt
# ---------------------------------------------------------------------------
_FREE = {"Reshape", "ExpandDims", "Squeeze", "Slice", "Transpose", "AsStrided",
         "Broadcast", "StopGradient"}
_EXTRA = {"DynamicSlice": 1, "DynamicSliceUpdate": 2, "ArgPartition": 4}


def _count(outputs):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    text = buffer.getvalue()
    total = 0
    for label in re.findall(r'label ="([^"]+)"', text):
        if label in _FREE:
            continue
        total += 1 + _EXTRA.get(label, 0)
    for match in re.finditer(r'\{ (\d+) \[label ="Concatenate"', text):
        node = match.group(1)
        total += max(0, len(re.findall(rf'-> {node}\b', text)) - 1)
    return total


def _compiled_count(fn, inputs):
    out = mx.compile(fn)(*inputs)
    return _count(list(out) if isinstance(out, (list, tuple)) else [out])


def test_dispatch_map_before_after(monkeypatch):
    """The receipt: compiled-lane dispatches per QSA layer, stock vs fused.

    Counted on the CPU stream from the PRODUCTION expressions.  The fused
    side is counted analytically (one custom kernel is one dispatch, and a
    custom kernel cannot be traced without Metal), which is why each stock
    number is measured and each replacement is a named constant.
    """

    # ``mx.fast.rms_norm`` is ONE Metal dispatch, but on the CPU stream MLX
    # traces its composed fallback (square/sum/rsqrt/multiply). Stand it in
    # with a single primitive so the count is the Metal count; the surrogate's
    # VALUES are irrelevant -- nothing here is evaluated for numerics.
    monkeypatch.setattr(
        mx.fast,
        "rms_norm",
        lambda x, w, eps, **kw: mx.sigmoid(x) if w is None else mx.sigmoid(x) * w,
    )
    # The baseline is the op-diet-armed path, which is what ships today.
    monkeypatch.setattr(runtime_options, "_FABLE_OPDIET", True)
    monkeypatch.setattr(
        runtime_options, "_FABLE_OPDIET_SELECTED",
        frozenset(runtime_options.FABLE_OPDIET_ITEMS),
    )
    indexer = _indexer()
    raw, pooled = _bank()
    offset = mx.array(BLOCKS // 2 * RATIO, dtype=mx.int32)
    q_idx = mx.random.normal((1, ROWS, 4, IDX_HEAD_DIM)).astype(mx.bfloat16)
    scores = mx.random.normal((1, ROWS, 4, BLOCKS)).astype(mx.float32)
    top_idx = (mx.random.uniform(shape=(ROWS, TOPK)) * BLOCKS).astype(mx.uint32)
    mx.eval(q_idx, scores, top_idx, indexer.parameters(), indexer._inv_freq)

    stock = {
        "prepare_queries": _compiled_count(
            lambda q, o: indexer._prepare_queries_eager(q, o), (q_idx, offset)
        ),
        "pooled_row": _compiled_count(
            lambda r, p, o: indexer._extend_pooled_fixed(
                _FixedCache(r, p, o), o + ROWS
            ),
            (raw, pooled, offset),
        ),
        "index_scores": _compiled_count(
            lambda s, o: _index_scores_stock(
                s, o, RATIO, IDX_HEAD_DIM, BLOCKS, ROWS
            ),
            (scores, offset),
        ),
        "row_tokens": _compiled_count(
            lambda t, o: list(_row_tokens_stock(t, o, RATIO, ROWS, BLOCKS)),
            (top_idx, offset),
        ),
    }
    # The two selection chains share their position scaffolding (qpos, nb_q,
    # blk, valid) inside one compiled graph, so their sum over-counts. This is
    # the honest combined figure the lane replaces with two kernels.
    def _glue(sc, ti, off):
        masked = _index_scores_stock(sc, off, RATIO, IDX_HEAD_DIM, BLOCKS, ROWS)
        idx, ok = _row_tokens_stock(ti, off, RATIO, ROWS, BLOCKS)
        return [masked, idx, ok]

    combined = _compiled_count(_glue, (scores, top_idx, offset))
    print("[dispatch map] combined selection glue:", combined)
    assert combined <= stock["index_scores"] + stock["row_tokens"]
    # Every stock chain must be worth replacing, and the numbers the kernel
    # module advertises must be the ones this harness measures.
    print("[dispatch map] stock per QSA layer:", stock)
    for name, measured in stock.items():
        assert measured == m4.STOCK_DISPATCHES[name], (name, measured)
        assert measured >= 9, (name, measured)

    fused = m4.FUSED_DISPATCHES
    assert set(fused) == set(stock)
    saved = sum(stock.values()) - sum(fused.values())
    print("[dispatch map] fused per QSA layer:", dict(fused), "saved:", saved)
    # 63 -> 7 per QSA layer; 12 QSA layers per verify cycle.
    assert sum(stock.values()) == 63
    assert sum(fused.values()) == 7
    assert saved == 56
    assert saved * 12 == 672
