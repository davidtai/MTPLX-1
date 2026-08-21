from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx.deepseek_v4_dspark_generation import (  # noqa: E402
    DeepseekV4DSparkCycle,
    DSparkTargetVerification,
    generate_dspark,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _logits(tokens, vocab_size=64):
    logits = mx.full((1, len(tokens), vocab_size), -100.0)
    for row, token in enumerate(tokens):
        logits[:, row, token] = 100.0
    return logits


@dataclass
class _Cache:
    offset: int


def _cycle(target_predictions):
    calls = []
    target_cache = _Cache(offset=9)
    dspark_caches = [_Cache(offset=9) for _ in range(3)]

    def propose(primary, start_pos):
        calls.append(("propose", int(primary.item()), start_pos))
        return mx.array([[31, 32, 33, 34, 35]], dtype=mx.int32)

    def verify(verify_ids):
        calls.append(("verify", tuple(np.array(verify_ids)[0])))
        target_cache.offset += int(verify_ids.shape[1])
        taps = tuple(
            mx.full((1, 6, 2), layer_id, dtype=mx.float32)
            for layer_id in (40, 41, 42)
        )
        return DSparkTargetVerification(
            logits=_logits(target_predictions),
            taps=taps,
        )

    def trim_target(count):
        calls.append(("trim_target", count))
        target_cache.offset -= count

    def commit_dspark(taps, start_pos):
        width = int(taps[0].shape[1])
        calls.append(("commit_dspark", start_pos, width))
        for cache in dspark_caches:
            cache.offset += width

    return (
        DeepseekV4DSparkCycle(
            propose_k5=propose,
            verify_m6=verify,
            trim_target=trim_target,
            commit_dspark=commit_dspark,
        ),
        calls,
        target_cache,
        dspark_caches,
    )


def test_cycle_conditions_k5_on_primary_and_verifies_physical_m6() -> None:
    cycle, calls, target_cache, dspark_caches = _cycle([31, 32, 33, 34, 35, 36])

    result = cycle(_logits([29])[:, 0], start_pos=9)

    assert calls[0] == ("propose", 29, 9)
    assert calls[1] == ("verify", (29, 31, 32, 33, 34, 35))
    assert result.physical_verify_width == 6
    assert result.accepted_future_tokens == 5
    assert tuple(np.array(result.committed_tokens)[0]) == (29, 31, 32, 33, 34, 35)
    assert int(result.next_primary.item()) == 36
    assert target_cache.offset == 15
    assert [cache.offset for cache in dspark_caches] == [15, 15, 15]


def test_partial_rejection_keeps_only_primary_and_matching_future_prefix() -> None:
    cycle, calls, target_cache, dspark_caches = _cycle([31, 32, 49, 50, 51, 52])

    result = cycle(_logits([29])[:, 0], start_pos=9)

    assert result.accepted_future_tokens == 2
    assert tuple(np.array(result.committed_tokens)[0]) == (29, 31, 32)
    assert int(result.next_primary.item()) == 49
    assert calls[2] == ("trim_target", 3)
    assert calls[3] == ("commit_dspark", 9, 3)
    assert target_cache.offset == 12
    assert [cache.offset for cache in dspark_caches] == [12, 12, 12]


def test_fixed_generator_emits_one_full_k5_cycle_and_receipt() -> None:
    class _EmptyCache:
        state = ()

    class _Tokenizer:
        eos_token_id = 63

        def decode(self, tokens):
            return " ".join(str(token) for token in tokens)

    taps = tuple(mx.zeros((1, 2, 2)) for _ in range(3))
    installed = SimpleNamespace(
        make_target_cache=lambda: [_EmptyCache()],
        make_dspark_cache=lambda: [_EmptyCache(), _EmptyCache(), _EmptyCache()],
        target_prefill=lambda _ids, _cache: DSparkTargetVerification(
            logits=_logits([29]),
            taps=taps,
        ),
        prefill_dspark=lambda _taps, _cache: None,
        proposal_k5=lambda _primary, _cache, _start: mx.array(
            [[31, 32, 33, 34, 35]], dtype=mx.int32
        ),
        target_m6=lambda _ids, _cache: DSparkTargetVerification(
            logits=_logits([31, 32, 33, 34, 35, 36]),
            taps=tuple(mx.zeros((1, 6, 2)) for _ in range(3)),
        ),
        trim_target=lambda _cache, _count: None,
        commit_dspark=lambda _taps, _cache, _start: None,
    )
    rt = SimpleNamespace(
        deepseek_v4_dspark_runtime=installed,
        tokenizer=_Tokenizer(),
    )

    output = generate_dspark(rt, [7, 8], max_tokens=6)

    assert output.tokens == [29, 31, 32, 33, 34, 35]
    assert output.stats.mode == "dspark"
    assert output.stats.verify_calls == 1
    assert output.stats.events[0]["verify_width_histogram"] == {"6": 1}
    assert output.stats.events[0]["accepted_depth_histogram"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 1,
    }
