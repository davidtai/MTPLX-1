"""QSACache must be a full citizen of the cache contract.

The QSA indexer keeps its own raw-key stream (and derived pooled block keys)
next to the attention KV. The serve loop rolls caches back after every
speculative verify round (``rollback_after_verify``: trim for trimmable
entries, snapshot-restore for the rest) and resumes banked sessions through
``state``. A raw-key stream that only ever appends desyncs from the KV on the
first rollback; once the context crosses the indexer's engage threshold the
selection mask is built from the raw-stream length while attention keys come
from the KV — the ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash
OpenCode hit live at 3.7k ctx (2026-08-27). Below the threshold the same
desync corrupts pooled blocks silently instead of crashing.

All runs are CPU (M-series GPU fp32 matmul is reduced-precision; CPU is the
parity surface).
"""

import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.graphbank as graphbank
from mtplx.cache_state import (
    rollback_after_verify,
    snapshot_untrimmable_cache,
)
from mtplx.models.qwen4_exp import Attention, QSACache, Qwen4ExpMTP, TextArgs


def _tiny_args(*, compress_ratio: int = 2) -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=compress_ratio,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


PREFILL = 12  # engage threshold with budget=8/ratio=2 is >8 visible tokens
STEP = 4  # a depth-3 verify round: 1 committed + 3 drafts


def test_rollback_then_forward_matches_fresh_run(attn):
    """A rejected verify round must leave the QSA layer exactly where a run
    that never saw the rejected tokens would be."""
    x_pre = _hidden(PREFILL, seed=1)
    x_rejected = _hidden(STEP, seed=2)
    x_next = _hidden(STEP, seed=3)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    assert cache[0].offset == PREFILL
    out = attn(x_next, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert out.shape == golden.shape
    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_state_roundtrip_resumes_identically(attn):
    """Bank restore: ``state`` must carry everything the layer needs — a
    resumed session past the engage threshold selects the same blocks and
    produces the same output as the uninterrupted run."""
    x_pre = _hidden(PREFILL, seed=4)
    x_next = _hidden(STEP, seed=5)

    live = QSACache()
    attn(x_pre, live)
    golden = attn(x_next, live)

    donor = QSACache()
    attn(x_pre, donor)
    resumed = QSACache()
    resumed.state = donor.state
    assert resumed.offset == PREFILL
    out = attn(x_next, resumed)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_trim_contract(attn):
    """QSACache is trimmable: trim rolls the layer back token-exactly,
    including through a pooled-block boundary."""
    cache = QSACache()
    assert cache.is_trimmable()

    x_pre = _hidden(PREFILL, seed=6)
    x_tail = _hidden(3, seed=7)  # odd length: trims back through a block edge
    x_next = _hidden(STEP, seed=8)

    attn(x_pre, cache)
    attn(x_tail, cache)
    assert cache.trim(3) == 3
    assert cache.offset == PREFILL
    out = attn(x_next, cache)

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_rollback_below_engage_threshold_still_exact(attn):
    """The desync is silent below the engage threshold (dense mask hides it);
    the pooled stream must still be positionally correct once the session
    grows past it."""
    x_pre = _hidden(4, seed=9)
    x_rejected = _hidden(STEP, seed=10)
    # two accepted rounds carry the session across the threshold
    x_a = _hidden(STEP, seed=11)
    x_b = _hidden(STEP, seed=12)
    x_c = _hidden(STEP, seed=13)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    for chunk in (x_a, x_b, x_c):
        out = attn(chunk, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    for chunk in (x_a, x_b, x_c):
        golden = attn(chunk, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_tensor_offset_qsa_cache_trim_matches_stock(attn):
    """The compiled-verifier cache owns fixed banks without changing QSA math."""
    x_pre = _hidden(PREFILL, seed=14)
    x_rejected = _hidden(STEP, seed=15)
    x_next = _hidden(STEP, seed=16)

    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, cache[0])
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache,
        reserve_tokens=STEP,
        initial_reserve_tokens=16,
    )

    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL

    attn(x_rejected, cache[0])
    assert cache[0].trim(STEP) == STEP
    out = attn(x_next, cache[0])

    fresh = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_tensor_offset_qsa_s1_uses_bounded_rows_gather(attn, monkeypatch):
    """The compiled D3 cache must not fall back to full-capacity attention."""

    import mtplx.models.qwen4_exp as qwen4_exp

    monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
    monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "0")
    x_pre = _hidden(PREFILL, seed=17)
    x_next = _hidden(1, seed=18)

    dense = [QSACache(compress_ratio=attn.indexer.ratio)]
    gathered = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, dense[0])
    attn(x_pre, gathered[0])
    for cache in (dense, gathered):
        promoted, failures = graphbank.promote_kv_cache_offsets(
            cache,
            reserve_tokens=4,
            initial_reserve_tokens=16,
        )
        assert promoted == 1
        assert failures == {}
        assert isinstance(cache[0], graphbank.TensorOffsetQSACache)

    dense[0].fixed_rows_gather = False
    gathered[0].fixed_rows_gather = True
    gather_rows = []
    stock_gather = qwen4_exp._qsa_rows_gather_attention

    def counted_gather(*args, **kwargs):
        gather_rows.append(int(args[0].shape[2]))
        return stock_gather(*args, **kwargs)

    monkeypatch.setattr(
        qwen4_exp,
        "_qsa_rows_gather_attention",
        counted_gather,
    )
    dense_out = attn(x_next, dense[0])
    gathered_out = attn(x_next, gathered[0])
    mx.eval(dense_out, gathered_out)

    assert gather_rows == [1]
    scale = mx.abs(dense_out.astype(mx.float32)).max().item() + 1e-6
    relative_error = (
        mx.abs(gathered_out.astype(mx.float32) - dense_out.astype(mx.float32))
        / scale
    ).max().item()
    assert relative_error < 2e-2


@pytest.mark.parametrize("accepted_count", range(4))
def test_pr391_device_mtp_replay_matches_variable_width_oracle(
    attn, accepted_count
):
    """Fixed three-row replay must preserve all five logical QSA leaves."""

    from mtplx.pr391_mtp_handoff import stage_pr391_mtp_authoritative_replay

    x_pre = _hidden(PREFILL, seed=30)
    speculative = [_hidden(1, seed=31 + row) for row in range(3)]
    authoritative = [_hidden(1, seed=41 + row) for row in range(3)]
    x_next = _hidden(1, seed=50)

    candidate = [QSACache(compress_ratio=attn.indexer.ratio)]
    oracle = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, candidate[0])
    attn(x_pre, oracle[0])
    for cache in (candidate, oracle):
        promoted, failures = graphbank.promote_kv_cache_offsets(
            cache,
            reserve_tokens=16,
            initial_reserve_tokens=16,
        )
        assert promoted == 1
        assert failures == {}
        cache[0].fixed_rows_gather = False

    # D3 stages primary, d1, and d2. Only its primary row is authoritative.
    for row in speculative:
        attn(row, candidate[0])
    attn(speculative[0], oracle[0])
    if accepted_count:
        attn(mx.concatenate(authoritative[:accepted_count], axis=1), oracle[0])

    stage_pr391_mtp_authoritative_replay(
        candidate,
        cycle_offset=PREFILL,
        accepted_count=accepted_count,
        authoritative_hidden=mx.concatenate(authoritative, axis=1),
        draft_token_ids=mx.arange(3, dtype=mx.uint32).reshape(1, 3),
        append_row=lambda hidden, _token: attn(hidden, candidate[0]),
    )

    candidate_offset = candidate[0].size()
    oracle_offset = oracle[0].size()
    assert candidate_offset == oracle_offset == PREFILL + 1 + accepted_count
    def logical_leaves(entry, offset):
        return (
            entry.kv.keys[:, :, :offset],
            entry.kv.values[:, :, :offset],
            entry.raw_keys[:, :offset],
            entry.pooled[:, : offset // entry.ratio],
        )

    candidate_leaves = logical_leaves(candidate[0], candidate_offset)
    oracle_leaves = logical_leaves(oracle[0], oracle_offset)
    mx.eval(*candidate_leaves, *oracle_leaves)
    assert all(
        mx.array_equal(actual, expected).item()
        for actual, expected in zip(candidate_leaves, oracle_leaves, strict=True)
    )

    candidate_next = attn(x_next, candidate[0])
    oracle_next = attn(x_next, oracle[0])
    mx.eval(candidate_next, oracle_next)
    assert mx.array_equal(candidate_next, oracle_next).item()


@pytest.mark.parametrize("accepted_count", range(4))
def test_pr391_capture_replay_matches_quantized_full_mtp_oracle(accepted_count):
    """Replay parity includes token fusion, MoE, hyper-connections, and QSA."""

    from mtplx.pr391_mtp_handoff import stage_pr391_mtp_authoritative_replay

    args = TextArgs(
        hidden_size=64,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )
    mx.random.seed(391)
    mtp = Qwen4ExpMTP(args)
    nn.quantize(
        mtp,
        group_size=32,
        bits=4,
        class_predicate=lambda _path, module: (
            {"group_size": 32, "bits": 4}
            if isinstance(module, nn.Linear)
            and int(module.weight.shape[-1]) % 32 == 0
            else False
        ),
    )
    mx.eval(mtp.parameters())
    token_embeddings = mx.random.normal((args.vocab_size, args.hidden_size))

    def embeddings(token_ids):
        return token_embeddings[token_ids.astype(mx.int32)]

    candidate = [QSACache(compress_ratio=4)]
    oracle = [QSACache(compress_ratio=4)]
    prefill_hidden = mx.random.normal((1, PREFILL, args.hidden_size * args.hc_count))
    prefill_tokens = mx.arange(PREFILL, dtype=mx.int32).reshape(1, -1)
    for cache in (candidate, oracle):
        mtp.fuse_and_run_history(
            prefill_hidden,
            embeddings(prefill_tokens),
            cache,
        )
        promoted, failures = graphbank.promote_kv_cache_offsets(
            cache,
            reserve_tokens=16,
            initial_reserve_tokens=16,
        )
        assert promoted == 1
        assert failures == {}
        cache[0].fixed_rows_gather = True

    # The live D3 has staged primary, d1, and d2; only primary survives.
    primary_hidden = mx.random.normal((1, 1, args.hidden_size * args.hc_count))
    primary_token = mx.array([[9]], dtype=mx.int32)
    draft_tokens = mx.array([[20, 21, 22]], dtype=mx.uint32)
    next_hidden = mtp.fuse_and_run(
        primary_hidden,
        embeddings(primary_token),
        candidate,
    )
    for token in (20, 21):
        token_row = mx.array([[token]], dtype=mx.int32)
        next_hidden = mtp.fuse_and_run(
            next_hidden[:, -1:],
            embeddings(token_row),
            candidate,
        )
    mtp.fuse_and_run(
        primary_hidden,
        embeddings(primary_token),
        oracle,
    )

    authoritative = mx.random.normal(
        (1, 3, args.hidden_size * args.hc_count)
    )
    if accepted_count:
        mtp.fuse_and_run_history(
            authoritative[:, :accepted_count],
            embeddings(draft_tokens[:, :accepted_count]),
            oracle,
        )
    stage_pr391_mtp_authoritative_replay(
        candidate,
        cycle_offset=PREFILL,
        accepted_count=accepted_count,
        authoritative_hidden=authoritative,
        draft_token_ids=draft_tokens,
        append_row=lambda hidden, tokens: mtp.fuse_and_run_history(
            hidden,
            embeddings(tokens),
            candidate,
        ),
    )

    candidate_leaves = tuple(candidate[0].state_leaves)
    oracle_leaves = tuple(oracle[0].state_leaves)
    mx.eval(*candidate_leaves, *oracle_leaves)
    mismatches = []
    for index, (actual, expected) in enumerate(
        zip(candidate_leaves, oracle_leaves, strict=True)
    ):
        if not mx.array_equal(actual, expected).item():
            logical_end = PREFILL + 1 + accepted_count
            axis = 2 if index in (0, 1) else 1
            slices = [slice(None)] * actual.ndim
            slices[axis] = slice(0, logical_end)
            prefix_actual = actual[tuple(slices)]
            prefix_expected = expected[tuple(slices)]
            mismatches.append(
                (
                    index,
                    float(
                        mx.max(
                            mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
                        ).item()
                    ),
                    float(
                        mx.max(
                            mx.abs(
                                prefix_actual.astype(mx.float32)
                                - prefix_expected.astype(mx.float32)
                            )
                        ).item()
                    ),
                )
            )
    prefix_mismatches = [item for item in mismatches if item[2] != 0.0]
    if prefix_mismatches:
        raise AssertionError(f"logical cache leaf mismatches: {prefix_mismatches!r}")

    # Dead speculative rows may remain beyond the logical offset. The next
    # D3 must overwrite them before they can affect any attention result.
    following_hidden = mx.random.normal(
        (1, 1, args.hidden_size * args.hc_count)
    )
    for token in (31, 32, 33):
        token_row = mx.array([[token]], dtype=mx.int32)
        candidate_out = mtp.fuse_and_run(
            following_hidden,
            embeddings(token_row),
            candidate,
        )
        oracle_out = mtp.fuse_and_run(
            following_hidden,
            embeddings(token_row),
            oracle,
        )
        mx.eval(candidate_out, oracle_out)
        assert mx.array_equal(candidate_out, oracle_out).item()
        following_hidden = oracle_out[:, -1:]


def test_pr391_replay_reference_preserves_live_trim_and_width_topology():
    """Reference replay must preserve the live trim and stock S0-S3 shape."""

    from mtplx.pr391_mtp_handoff import stage_pr391_mtp_authoritative_replay

    source = inspect.getsource(stage_pr391_mtp_authoritative_replay)
    assert "current_offset = int(entry.offset)" in source
    assert "entry.trim(trim)" in source
    assert "if accepted_count:" in source
    assert "authoritative_hidden[:, :accepted_count]" in source
    assert "draft_token_ids[:, :accepted_count].astype(mx.int32)" in source
    assert "variants" not in source
    assert "mx.where" not in source


@pytest.mark.parametrize("base_residue", range(4))
@pytest.mark.parametrize("accepted_count", range(4))
def test_pr391_replay_survives_next_d3_at_every_ratio4_residue(
    base_residue, accepted_count
):
    """Dead replay rows must not alter any of the following three S1 rows."""

    from mtplx.pr391_mtp_handoff import stage_pr391_mtp_authoritative_replay

    layer = Attention(_tiny_args(compress_ratio=4))
    mx.eval(layer.parameters())
    prefill = 12 + base_residue
    x_pre = _hidden(prefill, seed=60 + base_residue)
    speculative = [_hidden(1, seed=70 + row) for row in range(3)]
    authoritative = [_hidden(1, seed=80 + row) for row in range(3)]
    following = [_hidden(1, seed=90 + row) for row in range(3)]

    candidate = [QSACache(compress_ratio=4)]
    oracle = [QSACache(compress_ratio=4)]
    layer(x_pre, candidate[0])
    layer(x_pre, oracle[0])
    for cache in (candidate, oracle):
        promoted, failures = graphbank.promote_kv_cache_offsets(
            cache,
            reserve_tokens=16,
            initial_reserve_tokens=16,
        )
        assert promoted == 1
        assert failures == {}
        cache[0].fixed_rows_gather = True

    for row in speculative:
        layer(row, candidate[0])
    layer(speculative[0], oracle[0])
    if accepted_count:
        layer(
            mx.concatenate(authoritative[:accepted_count], axis=1),
            oracle[0],
        )

    stage_pr391_mtp_authoritative_replay(
        candidate,
        cycle_offset=prefill,
        accepted_count=accepted_count,
        authoritative_hidden=mx.concatenate(authoritative, axis=1),
        draft_token_ids=mx.arange(3, dtype=mx.uint32).reshape(1, 3),
        append_row=lambda hidden, _token: layer(hidden, candidate[0]),
    )

    for hidden in following:
        candidate_out = layer(hidden, candidate[0])
        oracle_out = layer(hidden, oracle[0])
        mx.eval(candidate_out, oracle_out)
        assert mx.array_equal(candidate_out, oracle_out).item()


def test_compiled_verify_bank_threads_qsa_state_without_fallback(attn):
    class TinyQSARuntime:
        def __init__(self):
            mx.random.seed(17)
            self.attn = attn
            self.embed = mx.random.normal((32, 64)).astype(mx.float32)
            self.head = mx.random.normal((64, 32)).astype(mx.float32)

        def forward_ar_capture(
            self,
            input_ids,
            *,
            cache,
            return_hidden=True,
            hidden_variant=None,
            capture_backend=None,
        ):
            del hidden_variant, capture_backend
            hidden = self.attn(self.embed[input_ids], cache[0])
            logits = hidden @ self.head
            return logits, hidden, {}

    rt = TinyQSARuntime()
    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    rt.forward_ar_capture(
        mx.arange(PREFILL, dtype=mx.int32).reshape(1, -1), cache=cache
    )
    bank = graphbank.CompiledVerifyBank(rt, request_max_tokens=16)

    bank.forward_ar_capture(mx.array([[1, 2, 3, 4]]), cache=cache)
    bank.forward_ar_capture(mx.array([[5, 6, 7, 8]]), cache=cache)

    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["traces"] == 1
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL + 8
