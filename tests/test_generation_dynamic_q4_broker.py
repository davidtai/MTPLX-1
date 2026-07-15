from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_state import VllmMetalPagedKVCache
from mtplx.generation import (
    PostcommitAbort,
    generate_ar,
    generate_mtp1,
    generate_mtpa,
    generate_mtpk,
)
from mtplx.memory_broker import (
    AllocatorMemorySample,
    BrokerSnapshot,
    MemoryBudget,
    UnifiedMemoryBroker,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class _TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(str(int(token)) for token in tokens)


class _RealBrokerObserver:
    """Deterministic allocator telemetry around a real memory broker."""

    def __init__(self, broker: UnifiedMemoryBroker) -> None:
        self.memory_broker = broker
        self._growth_ticket = None
        self._growth_sample = 0
        self._close_sample = 0
        self._allocation_sizes: list[int] = []
        self._pending_release_bytes = 0

    def reserve_growth(
        self,
        *,
        cache_id: str,
        steady_delta_bytes: int,
        transient_delta_bytes: int,
    ):
        ticket = self.memory_broker.plan_kv_growth(
            cache_id=cache_id,
            steady_delta_bytes=steady_delta_bytes,
            transient_delta_bytes=transient_delta_bytes,
        )
        self._growth_ticket = ticket
        self._growth_sample = 0
        return ticket

    def commit_growth(
        self,
        ticket,
        *,
        measured_physical_bytes: int,
        allocator_before,
        allocator_after,
    ):
        allocation = self.memory_broker.commit_kv_growth(
            ticket,
            allocated_physical_bytes=measured_physical_bytes,
            allocator_before=allocator_before,
            allocator_after=allocator_after,
        )
        self._allocation_sizes.append(measured_physical_bytes)
        self._growth_ticket = None
        return allocation

    def abort_growth(
        self,
        ticket,
        *,
        observed_physical_bytes: int | None,
        allocator_before,
        allocator_after,
    ) -> None:
        try:
            self.memory_broker.abort_kv_growth(
                ticket,
                observed_kv_delta_bytes=observed_physical_bytes,
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        finally:
            self._growth_ticket = None

    def sample_allocator_memory(self) -> AllocatorMemorySample:
        registered = self.memory_broker.snapshot().kv_physical_bytes
        if self._growth_ticket is not None:
            if self._growth_sample == 0:
                self._growth_sample = 1
                active = registered
            else:
                self._growth_sample = 2
                active = registered + self._growth_ticket.steady_delta_bytes
            return AllocatorMemorySample(active, 0, active)

        if self._close_sample == 0:
            self._close_sample = 1
            return AllocatorMemorySample(registered, 0, registered)

        self._close_sample = 2
        self._pending_release_bytes = self._allocation_sizes[-1]
        active = registered - self._pending_release_bytes
        return AllocatorMemorySample(active, 0, registered)

    def release_cache(
        self,
        *,
        cache_id: str,
        allocations,
        released_physical_bytes: int,
        allocator_before,
        allocator_after,
    ) -> None:
        assert released_physical_bytes == self._pending_release_bytes
        try:
            self.memory_broker.release_kv_batch(
                cache_id=cache_id,
                allocations=allocations,
                registered_kv_bytes_after=(
                    self.memory_broker.snapshot().kv_physical_bytes
                    - released_physical_bytes
                ),
                allocator_before=allocator_before,
                allocator_after=allocator_after,
            )
        finally:
            self._close_sample = 0
            self._pending_release_bytes = 0
        self._allocation_sizes.pop()


class _BrokeredTinyModel:
    def __init__(self, *, fail_after_target_calls: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.target_caches: list[VllmMetalPagedKVCache] = []
        self.mtp_caches: list[VllmMetalPagedKVCache] = []
        self.fail_after_target_calls = fail_after_target_calls
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        entry = VllmMetalPagedKVCache()
        self.target_caches.append(entry)
        return [entry]

    def make_mtp_cache(self):
        entry = VllmMetalPagedKVCache()
        self.mtp_caches.append(entry)
        return [entry]

    @staticmethod
    def _write(cache, length: int) -> None:
        if not cache:
            return
        values = mx.zeros((1, 1, int(length), 16), dtype=mx.float16)
        cache[0].update_without_fetch(values, values)

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        position_offset=None,
    ):
        self._write(mtp_cache, int(next_token_ids.shape[1]))
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        if (
            self.fail_after_target_calls is not None
            and len(self.calls) >= self.fail_after_target_calls
        ):
            raise RuntimeError("injected generation failure")
        length = int(input_ids.shape[1])
        self._write(cache, length)
        self.calls.append(
            {
                "tokens": length,
                "return_hidden": bool(return_hidden),
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
            }
        )
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        length = int(next_token_ids.shape[1])
        self._write(mtp_cache, length)
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        logits = mx.zeros((1, length, 4), dtype=mx.float32)
        logits = logits + mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits

    @property
    def physical_caches(self) -> list[VllmMetalPagedKVCache]:
        return [*self.target_caches, *self.mtp_caches]


def _runtime(
    model: _BrokeredTinyModel,
    observer: _RealBrokerObserver,
    *,
    mtp_enabled: bool,
) -> MTPLXRuntime:
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=_TinyTokenizer(),
        model_path=Path("tiny"),
        mtp_enabled=mtp_enabled,
        contract=MTPContract(),
    )
    runtime.expert_streaming = observer
    return runtime


def _broker() -> UnifiedMemoryBroker:
    return UnifiedMemoryBroker(
        budget=MemoryBudget(
            memory_limit_bytes=100_000,
        ),
        initial_snapshot=BrokerSnapshot.synthetic(charged_bytes=0),
        expert_record_bytes=16,
    )


def _configure_q4(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q4")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_TURBOQUANT", "0")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE", "4")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS", "8")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE", "4")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS", "8")
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "0")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "")


def _assert_broker_empty(broker: UnifiedMemoryBroker) -> None:
    snapshot = broker.snapshot()
    assert snapshot.owned_kv_physical_bytes == 0
    assert snapshot.unreconciled_kv_physical_bytes == 0
    assert snapshot.kv_physical_bytes == 0
    assert snapshot.charged_bytes == 0


def _assert_q4_caches_closed(
    model: _BrokeredTinyModel,
    *,
    expect_mtp: bool,
) -> None:
    assert model.target_caches
    if expect_mtp:
        assert model.mtp_caches
    assert all(cache.allocation_observer is not None for cache in model.physical_caches)
    assert all(
        cache.kv_quant_config.normalized_mode == "q4" for cache in model.physical_caches
    )
    assert all(cache._closed for cache in model.physical_caches)
    assert all(cache._kv_allocations == [] for cache in model.physical_caches)


def _generate(generator, runtime, *, max_tokens: int = 2, token_callback=None):
    kwargs = {}
    if generator is generate_ar:
        kwargs["token_callback"] = token_callback
    elif generator is generate_mtpk:
        kwargs["speculative_depth"] = 1
        kwargs["token_callback"] = token_callback
    elif generator is generate_mtpa:
        kwargs["max_depth"] = 1
    return generator(
        runtime,
        [0],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=4),
        stop_token_ids=set(),
        **kwargs,
    )


_GENERATOR_CASES = [
    pytest.param(generate_ar, False, id="ar"),
    pytest.param(generate_mtp1, True, id="mtp1"),
    pytest.param(generate_mtpk, True, id="mtpk"),
    pytest.param(generate_mtpa, True, id="mtpa"),
]


@pytest.mark.parametrize(("generator", "mtp_enabled"), _GENERATOR_CASES)
def test_generators_release_real_brokered_q4_caches_after_sequential_successes(
    monkeypatch,
    generator,
    mtp_enabled: bool,
) -> None:
    _configure_q4(monkeypatch)
    broker = _broker()
    observer = _RealBrokerObserver(broker)
    model = _BrokeredTinyModel()
    runtime = _runtime(model, observer, mtp_enabled=mtp_enabled)

    for _ in range(2):
        _generate(generator, runtime)

        _assert_q4_caches_closed(model, expect_mtp=mtp_enabled)
        _assert_broker_empty(broker)

    assert len(model.target_caches) == 2


@pytest.mark.parametrize(("generator", "mtp_enabled"), _GENERATOR_CASES)
def test_generators_release_real_brokered_q4_caches_after_forward_error(
    monkeypatch,
    generator,
    mtp_enabled: bool,
) -> None:
    _configure_q4(monkeypatch)
    broker = _broker()
    observer = _RealBrokerObserver(broker)
    model = _BrokeredTinyModel(fail_after_target_calls=1)

    with pytest.raises(RuntimeError, match="injected generation failure"):
        _generate(
            generator,
            _runtime(model, observer, mtp_enabled=mtp_enabled),
        )

    _assert_q4_caches_closed(model, expect_mtp=mtp_enabled)
    _assert_broker_empty(broker)


@pytest.mark.parametrize(
    ("generator", "mtp_enabled"),
    [
        pytest.param(generate_ar, False, id="ar"),
        pytest.param(generate_mtpk, True, id="mtpk"),
    ],
)
def test_callback_cancellation_releases_real_brokered_q4_caches(
    monkeypatch,
    generator,
    mtp_enabled: bool,
) -> None:
    _configure_q4(monkeypatch)
    broker = _broker()
    observer = _RealBrokerObserver(broker)
    model = _BrokeredTinyModel()
    callback_calls = 0

    def cancel(_tokens) -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls >= 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        _generate(
            generator,
            _runtime(model, observer, mtp_enabled=mtp_enabled),
            max_tokens=3,
            token_callback=cancel,
        )

    _assert_q4_caches_closed(model, expect_mtp=mtp_enabled)
    _assert_broker_empty(broker)


def test_mtpk_postcommit_abort_releases_real_brokered_q4_caches(monkeypatch) -> None:
    _configure_q4(monkeypatch)
    broker = _broker()
    observer = _RealBrokerObserver(broker)
    model = _BrokeredTinyModel()
    callback_calls = 0

    def abort(_tokens) -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls >= 2:
            raise PostcommitAbort("foreground_preempted_postcommit")

    with pytest.raises(PostcommitAbort, match="foreground_preempted_postcommit"):
        _generate(
            generate_mtpk,
            _runtime(model, observer, mtp_enabled=True),
            max_tokens=3,
            token_callback=abort,
        )

    _assert_q4_caches_closed(model, expect_mtp=True)
    _assert_broker_empty(broker)
