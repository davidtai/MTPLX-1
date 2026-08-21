from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.benchmarks.dflash2_runtime import (  # noqa: E402
    MTPLXDFlash2Bundle,
    build_deepseek_v4_dflash2_runtime_context,
    load_mtplx_deepseek_v4_dflash2_bundle,
)
from mtplx.deepseek_v4_dflash2 import (  # noqa: E402
    DeepseekV4DSparkBackend,
    DeepseekV4DSparkDraftAdapter,
    DeepseekV4TargetOps,
    generate_deepseek_v4_dflash2,
)
from mtplx.models.deepseek_v4 import DeepseekV4AffineInt4Cache  # noqa: E402
from mtplx.models.deepseek_v4_dspark import DeepseekV4DSparkCache  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _FakeDeepseekTarget:
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            model_type="deepseek_v4",
            dspark_target_layer_ids=(40, 41, 42),
        )
        self.dspark = SimpleNamespace(stages=(object(), object(), object()))
        self.model = SimpleNamespace(embed_tokens=object())
        self.calls: list[tuple[int, bool]] = []

    def make_cache(self):
        return [
            DeepseekV4AffineInt4Cache(
                window_size=128,
                compress_ratio=0,
                head_dim=512,
            )
        ]

    def __call__(
        self,
        input_ids,
        *,
        cache,
        return_hidden,
        logits_keep=None,
    ):
        self.calls.append((int(input_ids.shape[1]), logits_keep == 1))
        rows = int(input_ids.shape[1])
        logits = mx.zeros((1, 1 if logits_keep == 1 else rows, 64))
        taps = tuple(
            mx.full((1, rows, 2), float(layer_id))
            for layer_id in (40, 41, 42)
        )
        return logits, taps


def test_target_ops_uses_physical_m6_and_ordered_deepseek_taps() -> None:
    model = _FakeDeepseekTarget()
    ops = DeepseekV4TargetOps()
    cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
    )

    logits, captured = ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[29, 31, 32, 33, 34, 35]], dtype=mx.int32),
        target_cache=cache,
        capture_layer_ids={41, 42, 43},
    )
    features = ops.extract_context_feature(captured, [40, 41, 42])

    assert ops.supports_model(model)
    assert ops.family(model) == "deepseek_v4_dspark"
    assert model.calls == [(6, False)]
    assert tuple(logits.shape) == (1, 6, 64)
    assert set(captured) == {41, 42, 43}
    assert tuple(features.shape) == (1, 6, 6)
    np.testing.assert_array_equal(
        np.array(features[0, 0]),
        np.array([40, 40, 41, 41, 42, 42], dtype=np.float32),
    )


def test_target_ops_owns_affine_int4_cache_and_trims_rejected_m6_suffix() -> None:
    model = _FakeDeepseekTarget()
    ops = DeepseekV4TargetOps()
    cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
    )
    owner = cache[0]
    owner.window.append(mx.zeros((1, 6, 512), dtype=mx.bfloat16))
    owner.offset = 6

    elapsed_ns = ops.restore_after_acceptance(
        cache,
        target_len=3,
        acceptance_length=2,
        drafted_tokens=5,
    )

    assert isinstance(owner, DeepseekV4AffineInt4Cache)
    assert owner.window.bits == 4
    assert owner.window.group_size == 64
    assert owner.offset == 3
    assert len(owner.window) == 3
    assert elapsed_ns >= 0
    capabilities = ops.capabilities_for(model)
    assert capabilities.supports_dflash is True
    assert capabilities.supports_kv_trim is True
    assert capabilities.supports_target_hidden_capture is True
    assert capabilities.supports_prefix_snapshot is False
    assert capabilities.supports_tree_verify is False


class _FakeDSparkAttention:
    window_size = 8
    head_dim = 64

    def prefill_context(self, projected, cache) -> None:
        cache.prefill(projected)

    def project_kv(self, projected, positions):
        assert int(projected.shape[1]) == int(positions.shape[0])
        return projected


class _FakeDSparkOwner:
    def __init__(self) -> None:
        self.stages = tuple(
            SimpleNamespace(attn=_FakeDSparkAttention()) for _ in range(3)
        )
        self.projected_taps: tuple[mx.array, ...] | None = None
        self.proposal_positions: list[int] = []

    def make_cache(self):
        return [
            DeepseekV4DSparkCache(window_size=8, head_dim=64) for _ in range(3)
        ]

    def propose_k5(
        self,
        primary_token_ids,
        embed_tokens,
        lm_head,
        caches,
        *,
        start_pos,
    ):
        del embed_tokens, lm_head, caches
        self.proposal_positions.append(int(start_pos))
        assert int(primary_token_ids.item()) == 29
        return SimpleNamespace(
            future_tokens=mx.array([[31, 32, 33, 34, 35]], dtype=mx.uint32)
        )


class _FakeStageZero:
    def __init__(self, owner: _FakeDSparkOwner) -> None:
        self.owner = owner
        self.attn = _FakeDSparkAttention()

    def fuse_main(self, taps):
        self.owner.projected_taps = tuple(taps)
        rows = int(taps[0].shape[1])
        return mx.zeros((1, rows, 64), dtype=mx.bfloat16)


def _fake_dspark_target():
    owner = _FakeDSparkOwner()
    owner.stages = (
        _FakeStageZero(owner),
        SimpleNamespace(attn=_FakeDSparkAttention()),
        SimpleNamespace(attn=_FakeDSparkAttention()),
    )
    target = SimpleNamespace(
        args=SimpleNamespace(
            hidden_size=2,
            dspark_target_layer_ids=(40, 41, 42),
            dspark_noise_token_id=128799,
        ),
        dspark=owner,
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
    )
    return target, owner


def test_draft_adapter_advertises_m6_but_projects_three_dspark_taps() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    concatenated = mx.concatenate(
        [mx.full((1, 2, 2), value) for value in (40.0, 41.0, 42.0)],
        axis=-1,
    )

    projected = draft.project_target_hidden(concatenated)

    assert draft.block_size == 6
    assert draft.mask_token_id == 128799
    assert tuple(draft.target_layer_ids) == (40, 41, 42)
    assert draft.capabilities.default_block_tokens == 6
    assert draft.capabilities.max_block_tokens == 6
    assert draft.capabilities.supports_copyspec is False
    assert draft.capabilities.supports_ddtree is False
    assert draft.capabilities.supports_early_rollback_launch is False
    assert tuple(projected.shape) == (1, 2, 64)
    assert owner.projected_taps is not None
    assert tuple(float(tap[0, 0, 0].item()) for tap in owner.projected_taps) == (
        40.0,
        41.0,
        42.0,
    )


def test_draft_backend_appends_committed_context_once_and_returns_five_tokens() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()
    caches = backend.make_cache(
        draft_model=draft,
        sink_size=0,
        window_size=8,
        allow_full_context_layers=False,
    )
    arguments = dict(
        target_model=target,
        target_ops=DeepseekV4TargetOps(),
        draft_model=draft,
        draft_cache=caches,
        staged_first=mx.array([29], dtype=mx.uint32),
        block_len=6,
        mask_token_tail=mx.full((5,), 128799, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )

    first = backend.draft_greedy(
        **arguments,
        draft_context=mx.zeros((1, 4, 64), dtype=mx.bfloat16),
    )
    second = backend.draft_greedy(
        **arguments,
        draft_context=mx.zeros((1, 2, 64), dtype=mx.bfloat16),
    )

    assert tuple(np.array(first)) == (31, 32, 33, 34, 35)
    assert tuple(np.array(second)) == (31, 32, 33, 34, 35)
    assert owner.proposal_positions == [4, 6]
    assert [cache.prefill_length for cache in caches] == [6, 6, 6]
    assert all(cache.ring.bits == 4 for cache in caches)
    assert all(cache.ring.group_size == 64 for cache in caches)


def test_draft_backend_returns_requested_prefix_for_dflash_final_tail() -> None:
    target, owner = _fake_dspark_target()
    draft = DeepseekV4DSparkDraftAdapter(target)
    backend = DeepseekV4DSparkBackend()
    caches = backend.make_cache(
        draft_model=draft,
        sink_size=0,
        window_size=8,
        allow_full_context_layers=False,
    )

    tail = backend.draft_greedy(
        target_model=target,
        target_ops=DeepseekV4TargetOps(),
        draft_model=draft,
        draft_cache=caches,
        staged_first=mx.array([29], dtype=mx.uint32),
        draft_context=mx.zeros((1, 4, 64), dtype=mx.bfloat16),
        block_len=3,
        mask_token_tail=mx.full((5,), 128799, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
    )

    assert tuple(np.array(tail)) == (31, 32)
    assert owner.proposal_positions == [4]


def test_deepseek_bundle_reuses_mtplx_target_and_dflash2_engine_types(
    monkeypatch,
) -> None:
    from mtplx.benchmarks import dflash2_runtime

    target, _owner = _fake_dspark_target()
    target.args.model_type = "deepseek_v4"
    target.make_cache = lambda: [
        DeepseekV4AffineInt4Cache(
            window_size=128,
            compress_ratio=0,
            head_dim=512,
        )
    ]
    tokenizer = object()
    runtime = SimpleNamespace(model=target, tokenizer=tokenizer)
    calls = []
    monkeypatch.setattr(
        dflash2_runtime,
        "load_mtplx_deepseek_runtime",
        lambda path: calls.append(path) or runtime,
    )

    bundle = load_mtplx_deepseek_v4_dflash2_bundle("/models/deepseek-v4")

    assert isinstance(bundle, MTPLXDFlash2Bundle)
    assert bundle.runtime is runtime
    assert bundle.target_model is target
    assert bundle.tokenizer is tokenizer
    assert isinstance(bundle.target_ops, DeepseekV4TargetOps)
    assert isinstance(bundle.draft_model, DeepseekV4DSparkDraftAdapter)
    assert isinstance(bundle.draft_backend, DeepseekV4DSparkBackend)
    assert bundle.checkpoint_block_size == 6
    assert bundle.target_layer_ids == (40, 41, 42)
    assert bundle.draft_meta["kind"] == "deepseek_v4_dspark"
    assert calls == ["/models/deepseek-v4"]


def test_deepseek_bundle_loader_selects_dspark_at_construction(monkeypatch) -> None:
    from mtplx import runtime as runtime_module
    from mtplx.benchmarks import dflash2_runtime

    loaded = object()
    calls = []

    def fake_load(model_path, *, mtp, dspark):
        calls.append((model_path, mtp, dspark))
        return loaded

    monkeypatch.setattr(runtime_module, "load", fake_load)

    assert dflash2_runtime.load_mtplx_deepseek_runtime("model") is loaded
    assert calls == [("model", True, True)]


def test_deepseek_runtime_context_fixes_dflash_m6_without_generic_kv_quantizer() -> None:
    context = build_deepseek_v4_dflash2_runtime_context()

    assert context.runtime.verify_mode == "dflash"
    assert context.runtime.verify_len_cap == 6
    assert context.runtime.copyspec_mode == "off"
    assert context.runtime.quantize_kv_cache is False
    assert context.runtime.prefix_cache is False
    assert context.runtime.dflash_max_ctx == 0


def test_generation_adapter_translates_existing_dflash_events_without_scheduling(
    monkeypatch,
) -> None:
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=3,
        generated_token_ids=(11, 12),
        generation_tokens=2,
        accepted_from_draft=1,
        acceptance_ratio=0.5,
        cycles_completed=1,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(1,),
        peak_memory_gb=2.0,
    )
    events = [
        TokenEvent(11, 1, 0.0, 0),
        TokenEvent(12, 2, 0.5, 1),
        summary,
    ]
    calls = []

    def fake_stream(**kwargs):
        calls.append(kwargs)
        return iter(events)

    monkeypatch.setattr(adapter_module, "_stream_dflash_generate", fake_stream)
    callback_tokens = []
    bundle = SimpleNamespace(
        target_model=object(),
        target_ops=object(),
        tokenizer=SimpleNamespace(decode=lambda values: f"decoded:{values}"),
        draft_model=object(),
        draft_backend=object(),
    )

    output = generate_deepseek_v4_dflash2(
        bundle,
        [1, 2, 3],
        max_tokens=2,
        token_callback=callback_tokens.append,
        runtime_context=object(),
    )

    assert output.tokens == [11, 12]
    assert output.text == "decoded:[11, 12]"
    assert callback_tokens == [[11], [12]]
    assert output.final_state is None
    stats = output.stats
    assert stats.mode == "dspark"
    assert stats.generated_tokens == 2
    assert stats.accepted_drafts == 1
    assert stats.drafted_tokens == 1
    assert stats.rejected_drafts == 0
    assert stats.verify_calls == 1
    assert stats.speculative_depth == 5
    assert stats.decode_elapsed_s == pytest.approx(0.009)
    assert stats.decode_tok_s == pytest.approx(2 / 0.009)
    assert stats.peak_memory_bytes == 2_000_000_000
    assert stats.events == [summary.to_payload()]
    assert len(calls) == 1
    assert calls[0]["block_tokens"] == 6
    assert calls[0]["prompt_tokens_override"] == [1, 2, 3]
    assert calls[0]["quantize_kv_cache"] is False


def test_generation_adapter_stops_at_first_stop_token_and_suppresses_suffix(
    monkeypatch,
) -> None:
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=3,
        generated_token_ids=(11, 12, 13),
        generation_tokens=3,
        accepted_from_draft=2,
        acceptance_ratio=2 / 3,
        cycles_completed=1,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(2,),
        peak_memory_gb=2.0,
    )
    events = [
        TokenEvent(11, 1, 0.0, 0),
        TokenEvent(12, 2, 0.5, 1),
        TokenEvent(13, 3, 2 / 3, 1),
        summary,
    ]
    monkeypatch.setattr(
        adapter_module,
        "_stream_dflash_generate",
        lambda **_kwargs: iter(events),
    )
    callback_tokens = []
    bundle = SimpleNamespace(
        target_model=object(),
        target_ops=object(),
        tokenizer=SimpleNamespace(decode=lambda values: f"decoded:{values}"),
        draft_model=object(),
        draft_backend=object(),
    )

    output = generate_deepseek_v4_dflash2(
        bundle,
        [1, 2, 3],
        max_tokens=3,
        stop_token_ids=[12],
        token_callback=callback_tokens.append,
        runtime_context=object(),
    )

    assert output.tokens == [11]
    assert output.text == "decoded:[11]"
    assert callback_tokens == [[11]]
    assert output.finish_reason == "stop"


def test_generation_adapter_accounts_for_each_physical_tail_width(monkeypatch) -> None:
    from dflash_mlx.engine.events import SummaryEvent
    import mtplx.deepseek_v4_dflash2 as adapter_module

    summary = SummaryEvent(
        elapsed_us=10_000.0,
        prompt_token_count=3,
        generated_token_ids=tuple(range(7)),
        generation_tokens=7,
        accepted_from_draft=4,
        acceptance_ratio=4 / 7,
        cycles_completed=3,
        phase_timings_us={"prefill": 1_000.0},
        block_tokens=6,
        verify_len_cap=6,
        acceptance_history=(2, 2, 0),
    )
    monkeypatch.setattr(
        adapter_module,
        "_stream_dflash_generate",
        lambda **_kwargs: iter([summary]),
    )
    bundle = SimpleNamespace(
        target_model=object(),
        target_ops=object(),
        tokenizer=SimpleNamespace(decode=lambda values: str(values)),
        draft_model=object(),
        draft_backend=object(),
    )

    output = generate_deepseek_v4_dflash2(
        bundle,
        [1, 2, 3],
        max_tokens=7,
        runtime_context=object(),
    )

    assert output.stats.drafted_tokens == 8
    assert output.stats.rejected_drafts == 4
    assert output.stats.drafted_by_depth == [2, 2, 2, 1, 1]
