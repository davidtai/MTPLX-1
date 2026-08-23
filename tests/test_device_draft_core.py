"""Device draft core: q-construction parity, sampling law, state signature."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.fast_sampling import sparse_distribution_from_mlx_logits  # noqa: E402
from mtplx.generation import (  # noqa: E402
    _device_core_cycle_depth_eligible,
    _device_core_state_signature,
    _device_draft_q_arrays,
    _map_compact_draft_ids,
    _map_compact_host_draft,
    _require_compact_device_core,
)
from mtplx.sampling import SamplerConfig, SparseDistribution  # noqa: E402


def _host_q(logits: mx.array, config: SamplerConfig) -> dict[int, float]:
    sparse = sparse_distribution_from_mlx_logits(logits, config)
    assert sparse is not None
    return {int(t): float(p) for t, p in zip(sparse.token_ids, sparse.probs)}


def _device_q(logits: mx.array, config: SamplerConfig) -> dict[int, float]:
    ids, probs = _device_draft_q_arrays(
        logits.reshape(-1),
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    probs_np = np.asarray(probs, dtype=np.float64).reshape(-1)
    keep = probs_np > 0
    kept = probs_np[keep]
    return {int(t): float(p) for t, p in zip(ids_np[keep], kept / kept.sum())}


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("scale", [1.0, 6.0])
def test_device_q_matches_host_sparse_distribution(seed: int, scale: float) -> None:
    rng = np.random.default_rng(seed)
    logits = mx.array((rng.standard_normal(512) * scale).astype(np.float32))
    config = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)

    host = _host_q(logits, config)
    device = _device_q(logits, config)

    assert set(device) == set(host)
    for token, prob in host.items():
        assert device[token] == pytest.approx(prob, abs=2e-5), token


def test_device_q_top_p_disabled_branch() -> None:
    rng = np.random.default_rng(3)
    logits = mx.array(rng.standard_normal(256).astype(np.float32))
    config = SamplerConfig(temperature=0.8, top_p=1.0, top_k=16)

    host = _host_q(logits, config)
    device = _device_q(logits, config)

    assert set(device) == set(host)
    for token, prob in host.items():
        assert device[token] == pytest.approx(prob, abs=2e-5), token


def test_compact_draft_ids_map_back_to_target_vocabulary_on_device() -> None:
    compact_ids = mx.array([0, 3, 5], dtype=mx.int32)
    token_map = mx.array([0, 1, 2, 100, 101, 102], dtype=mx.int32)

    mapped = _map_compact_draft_ids(compact_ids, token_map)
    mx.eval(mapped)

    assert mapped.tolist() == [0, 100, 102]


def test_compact_device_core_failure_preserves_root_cause() -> None:
    cause = ValueError("compiled draft shape mismatch")

    with pytest.raises(RuntimeError, match="compact draft vocabulary") as caught:
        _require_compact_device_core(
            token_map=mx.array([0, 1], dtype=mx.int32),
            draft_core="device",
            used_device_core=False,
            device_core_error=cause,
            ineligibility_reasons=(),
            allow_host_fallback=False,
        )

    assert caught.value.__cause__ is cause


def test_compact_device_core_failure_names_ineligible_contract() -> None:
    with pytest.raises(
        RuntimeError,
        match="ineligible contract: cycle_depth, persistent_cache",
    ):
        _require_compact_device_core(
            token_map=mx.array([0, 1], dtype=mx.int32),
            draft_core="device",
            used_device_core=False,
            device_core_error=None,
            ineligibility_reasons=("cycle_depth", "persistent_cache"),
            allow_host_fallback=False,
        )


@pytest.mark.parametrize("cycle_depth", [2, 3, 4, 5])
def test_device_core_accepts_supported_cycle_depth(cycle_depth: int) -> None:
    assert _device_core_cycle_depth_eligible(cycle_depth)


@pytest.mark.parametrize("cycle_depth", [0, 1, 6])
def test_device_core_rejects_unsupported_cycle_depth(cycle_depth: int) -> None:
    assert not _device_core_cycle_depth_eligible(cycle_depth)


def test_compact_d1_host_fallback_maps_token_and_sparse_distribution() -> None:
    token_map = mx.array([10, 20, 30, 40], dtype=mx.int32)
    compact_q = SparseDistribution(
        np.array([1, 3], dtype=np.int64),
        np.array([0.75, 0.25], dtype=np.float64),
        4,
    )

    token, mapped_q = _map_compact_host_draft(
        3,
        compact_q,
        token_map=token_map,
        target_vocab_size=100,
    )

    assert token == 40
    assert mapped_q is not None
    assert mapped_q.token_ids.tolist() == [20, 40]
    assert mapped_q.probs.tolist() == pytest.approx([0.75, 0.25])
    assert mapped_q.vocab_size == 100


def test_compact_device_core_allows_only_explicit_d1_host_fallback() -> None:
    _require_compact_device_core(
        token_map=mx.array([0, 1], dtype=mx.int32),
        draft_core="device",
        used_device_core=False,
        device_core_error=None,
        ineligibility_reasons=("cycle_depth",),
        allow_host_fallback=True,
    )


def test_device_inverse_cdf_sampling_matches_q() -> None:
    # The compiled chain samples via inverse-CDF over the normalized kept
    # support; the empirical law over many keys must match q itself.
    rng = np.random.default_rng(11)
    logits = mx.array((rng.standard_normal(128) * 4.0).astype(np.float32))
    ids, q_norm = _device_draft_q_arrays(logits, temperature=0.6, top_k=20, top_p=0.95)
    cdf = mx.cumsum(q_norm, axis=-1)
    k = int(ids.shape[0])

    draws = 20_000
    keys = mx.random.split(mx.random.key(1234), draws)
    counts: dict[int, int] = {}
    picks = []
    for i in range(draws):
        u = mx.random.uniform(key=keys[i])
        picks.append(mx.minimum((cdf <= u).sum(), k - 1).astype(mx.int32))
    mx.eval(picks)
    ids_np = np.asarray(ids, dtype=np.int64)
    for pick in picks:
        token = int(ids_np[int(pick.item())])
        counts[token] = counts.get(token, 0) + 1

    probs_np = np.asarray(q_norm, dtype=np.float64)
    for i, token in enumerate(ids_np):
        expected = probs_np[i]
        if expected == 0.0:
            assert counts.get(int(token), 0) == 0
            continue
        observed = counts.get(int(token), 0) / draws
        sigma = (expected * (1 - expected) / draws) ** 0.5
        assert abs(observed - expected) < max(5 * sigma, 5e-4), (token, observed, expected)


class _FakeTensorOffsetEntry:
    def __init__(self, keys: mx.array, values: mx.array, offset: int) -> None:
        self.compile_state = [[keys, values, mx.array(offset, dtype=mx.int32)], [None, None, None]]


def test_state_signature_survives_shape_stable_swaps() -> None:
    keys_a = mx.zeros((1, 4, 32, 64), dtype=mx.float16)
    values_a = mx.zeros((1, 4, 32, 64), dtype=mx.float16)
    cache = [_FakeTensorOffsetEntry(keys_a, values_a, 3)]
    first = _device_core_state_signature(cache)

    # Same shapes, brand-new arrays (the routine eager-append swap).
    cache[0].compile_state[0][0] = mx.ones((1, 4, 32, 64), dtype=mx.float16)
    cache[0].compile_state[0][2] = mx.array(9, dtype=mx.int32)
    assert _device_core_state_signature(cache) == first

    # Capacity growth changes the traced shapes and must invalidate.
    cache[0].compile_state[0][0] = mx.zeros((1, 4, 64, 64), dtype=mx.float16)
    assert _device_core_state_signature(cache) != first
