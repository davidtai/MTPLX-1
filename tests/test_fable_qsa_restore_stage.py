"""Pure host tests for the wired restored-QSA staging flag.

No MLX, no model, no Metal.  NumPy stands in for the array runtime and the
fakes below mirror the growth semantics of the real classes byte-for-byte in
shape terms:

* ``_FakeKV`` mirrors ``mlx_lm.models.cache.KVCache.update_and_fetch``
  (``n_steps = (step + T - 1) // step`` rows appended, truncate-to-offset when
  the frontier is not step-aligned), and
* ``_FakeQSA`` mirrors ``mtplx.models.qwen4_exp.QSACache`` --
  ``_grown_cap`` doubling, positional ``write_raw`` / ``write_pooled``, the
  reservation floor, ``trim``, and the 4-leaf ``state`` contract.

The load-bearing test is :func:`test_staged_and_unstaged_restores_end_identical`:
the staged arm and an unstaged twin run the SAME synthetic suffix and must end
with identical logical state.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mtplx.fable_qsa_restore_stage import (
    COUNTERS,
    FABLE_QSA_RESTORE_STAGING_ENV,
    QSARestoreStagingUnsupported,
    qsa_restore_staging_enabled,
    reset_counters,
    reset_qsa_restore_staging_flag_cache,
    restore_staging_eligibility,
    stage_restored_suffix,
)

MODULE = Path(__file__).parents[1] / "mtplx" / "fable_qsa_restore_stage.py"

RATIO = 4
INDEX_DIM = 7
K_DIM = 3
V_DIM = 5
KV_HEADS = 2


# --------------------------------------------------------------------------
# Deterministic synthetic content, keyed to ABSOLUTE position
# --------------------------------------------------------------------------


def _raw_rows(start: int, length: int) -> np.ndarray:
    positions = np.arange(start, start + length, dtype=np.float32)[:, None]
    return (positions * 0.5 + np.arange(INDEX_DIM, dtype=np.float32)[None, :])[None]


def _pooled_rows(nb_start: int, nb_total: int) -> np.ndarray:
    blocks = np.arange(nb_start, nb_total, dtype=np.float32)[:, None]
    return (blocks * 3.25 - np.arange(INDEX_DIM, dtype=np.float32)[None, :])[None]


def _kv_rows(start: int, length: int, dim: int, sign: float) -> np.ndarray:
    positions = np.arange(start, start + length, dtype=np.float32)
    base = positions[None, None, :, None] * sign
    return base + np.arange(dim, dtype=np.float32)[None, None, None, :]


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeKV:
    step = 256

    def __init__(self, events: list[str]):
        self.keys: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.offset = 0
        self.events = events

    def update_and_fetch(self, keys: np.ndarray, values: np.ndarray):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            self.events.append("kv-grow")
            b, heads, _, k_dim = keys.shape
            v_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            new_k = np.zeros((b, heads, n_steps * self.step, k_dim), keys.dtype)
            new_v = np.zeros((b, heads, n_steps * self.step, v_dim), values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = np.concatenate([self.keys, new_k], axis=2)
                self.values = np.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @state.setter
    def state(self, value):
        self.keys, self.values = value
        self.offset = self.keys.shape[2]

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n


class _FakeQSA:
    """Growth-faithful stand-in for ``mtplx.models.qwen4_exp.QSACache``."""

    step = 256

    def __init__(self, *, ratio: int = RATIO, events: list[str] | None = None):
        self.events: list[str] = [] if events is None else events
        self.kv = _FakeKV(self.events)
        self.ratio = ratio
        self.raw_keys: np.ndarray | None = None
        self.pooled: np.ndarray | None = None
        self.pooled_len = 0
        self.pooled_f32_t: np.ndarray | None = None
        self._reserved_raw_capacity = 0
        self._reserved_pooled_capacity = 0

    @property
    def offset(self) -> int:
        return self.kv.offset

    @staticmethod
    def _grown_cap(end: int, current: int, step: int) -> int:
        cap = ((end + step - 1) // step) * step
        return max(cap, 2 * current)

    def write_raw(self, keys: np.ndarray) -> None:
        start = self.kv.offset
        end = start + keys.shape[1]
        if self.raw_keys is None or end > self.raw_keys.shape[1]:
            self.events.append("raw-grow")
            current = 0 if self.raw_keys is None else self.raw_keys.shape[1]
            cap = max(
                self._grown_cap(end, current, self.step),
                self._reserved_raw_capacity,
            )
            grown = np.zeros((1, cap, keys.shape[2]), keys.dtype)
            if self.raw_keys is not None:
                grown[:, : self.raw_keys.shape[1], :] = self.raw_keys
            self.raw_keys = grown
        self.raw_keys[:, start:end, :] = keys

    def write_pooled(self, blocks: np.ndarray, nb_start: int, nb_total: int) -> None:
        if self.pooled is None or nb_total > self.pooled.shape[1]:
            self.events.append("pooled-grow")
            current = 0 if self.pooled is None else self.pooled.shape[1]
            cap = max(
                self._grown_cap(nb_total, current, self.step),
                self._reserved_pooled_capacity,
            )
            grown = np.zeros((1, cap, blocks.shape[2]), blocks.dtype)
            if self.pooled is not None:
                grown[:, : self.pooled.shape[1], :] = self.pooled
            self.pooled = grown
        self.pooled[:, nb_start:nb_total, :] = blocks
        # Derived fp32 mirror, rebuilt from CONTENT when absent (the real
        # class's contract; zeros there would blank valid blocks).
        cap_blocks = self.pooled.shape[1]
        if self.pooled_f32_t is None:
            self.pooled_f32_t = np.swapaxes(
                self.pooled.astype(np.float32), 1, 2
            )[:, None]
        elif self.pooled_f32_t.shape[3] < cap_blocks:
            grown_t = np.zeros((1, 1, blocks.shape[2], cap_blocks), np.float32)
            grown_t[..., : self.pooled_f32_t.shape[3]] = self.pooled_f32_t
            self.pooled_f32_t = grown_t
            self.pooled_f32_t[..., nb_start:nb_total] = np.swapaxes(
                blocks.astype(np.float32), 1, 2
            )[:, None]
        else:
            self.pooled_f32_t[..., nb_start:nb_total] = np.swapaxes(
                blocks.astype(np.float32), 1, 2
            )[:, None]
        self.pooled_len = nb_total

    def reserve_indexer_capacity(
        self, *, raw_capacity: int, pooled_capacity: int
    ) -> None:
        raw_requested = int(raw_capacity)
        pooled_requested = int(pooled_capacity)
        if raw_requested < 0 or pooled_requested < 0:
            raise ValueError("QSA reserved capacities must be non-negative")
        raw_existing = 0 if self.raw_keys is None else int(self.raw_keys.shape[1])
        pooled_existing = 0 if self.pooled is None else int(self.pooled.shape[1])
        raw_target = max(raw_requested, raw_existing, self._reserved_raw_capacity)
        pooled_target = max(
            pooled_requested, pooled_existing, self._reserved_pooled_capacity
        )
        if raw_target < self.offset:
            raise ValueError("raw capacity cannot cover QSA offset")
        if pooled_target < self.pooled_len:
            raise ValueError("pooled capacity cannot truncate the valid frontier")
        self._reserved_raw_capacity = raw_target
        self._reserved_pooled_capacity = pooled_target
        if self.raw_keys is not None and raw_target > raw_existing:
            grown = np.zeros(
                (1, raw_target, self.raw_keys.shape[2]), self.raw_keys.dtype
            )
            grown[:, :raw_existing, :] = self.raw_keys
            self.raw_keys = grown
        if self.pooled is not None and pooled_target > pooled_existing:
            grown = np.zeros(
                (1, pooled_target, self.pooled.shape[2]), self.pooled.dtype
            )
            grown[:, :pooled_existing, :] = self.pooled
            self.pooled = grown

    def trim(self, n: int) -> int:
        trimmed = self.kv.trim(n)
        self.pooled_len = min(self.pooled_len, self.kv.offset // self.ratio)
        return trimmed

    @property
    def state(self):
        off = self.kv.offset
        nb = min(self.pooled_len, off // self.ratio)
        raw = None if self.raw_keys is None else self.raw_keys[:, :off, :]
        pooled = None if self.pooled is None or nb == 0 else self.pooled[:, :nb, :]
        return (*self.kv.state, raw, pooled)


class _FakeArrays:
    """Non-QSA trunk entry, carrying positional metadata staging must not touch."""

    def __init__(self, size: int = 4):
        self.state = [None] * size
        self.offset = 0
        self.positions = np.arange(16, dtype=np.int32)


def _zeros(shape, dtype):
    return np.zeros(shape, dtype)


# --------------------------------------------------------------------------
# Restore + suffix simulation
# --------------------------------------------------------------------------


def _restored_stack(prefix_tokens: int) -> tuple[list, list[str]]:
    """A trunk cache stack restored to ``prefix_tokens``, exactly as a session
    snapshot leaves it: backings sized to the STORED prefix, nothing wider."""

    events: list[str] = []
    stack: list = []
    for layer in range(3):
        if layer == 1:
            stack.append(_FakeArrays())
            continue
        cache = _FakeQSA(events=events)
        # Replay the stored prefix in one go, then shrink the backings to the
        # snapshot's logical extent -- what restore_cache(state) produces.
        cache.write_raw(_raw_rows(0, prefix_tokens))
        nb = prefix_tokens // RATIO
        if nb:
            cache.write_pooled(_pooled_rows(0, nb), 0, nb)
        cache.kv.update_and_fetch(
            _kv_rows(0, prefix_tokens, K_DIM, 1.0),
            _kv_rows(0, prefix_tokens, V_DIM, -1.0),
        )
        keys, values, raw, pooled = cache.state
        restored = _FakeQSA(events=events)
        restored.kv.state = (np.array(keys), np.array(values))
        restored.raw_keys = None if raw is None else np.array(raw)
        restored.pooled = None if pooled is None else np.array(pooled)
        restored.pooled_len = 0 if pooled is None else pooled.shape[1]
        restored.pooled_f32_t = None
        restored._reserved_raw_capacity = (
            0 if raw is None else int(raw.shape[1])
        )
        restored._reserved_pooled_capacity = (
            0 if pooled is None else int(pooled.shape[1])
        )
        stack[len(stack) :] = [restored]
    events.clear()
    return stack, events


def _run_suffix(stack, *, suffix_tokens: int, chunk: int) -> None:
    """Forward the suffix in chunks, writing exactly what a QSA layer writes."""

    done = 0
    while done < suffix_tokens:
        length = min(chunk, suffix_tokens - done)
        for cache in stack:
            if not hasattr(cache, "reserve_indexer_capacity"):
                continue
            start = cache.kv.offset
            cache.write_raw(_raw_rows(start, length))
            nb_total = (start + length) // cache.ratio
            nb_start = cache.pooled_len
            if nb_total > nb_start:
                cache.write_pooled(
                    _pooled_rows(nb_start, nb_total), nb_start, nb_total
                )
            cache.kv.update_and_fetch(
                _kv_rows(start, length, K_DIM, 1.0),
                _kv_rows(start, length, V_DIM, -1.0),
            )
        done += length


def _logical_state(stack) -> list:
    """Everything a restored cache must carry forward, as plain values."""

    snapshot = []
    for cache in stack:
        if not hasattr(cache, "reserve_indexer_capacity"):
            snapshot.append(
                ("arrays", cache.offset, np.array(cache.positions), list(cache.state))
            )
            continue
        keys, values, raw, pooled = cache.state
        snapshot.append(
            (
                "qsa",
                cache.offset,
                cache.pooled_len,
                cache.ratio,
                np.array(keys),
                np.array(values),
                None if raw is None else np.array(raw),
                None if pooled is None else np.array(pooled),
            )
        )
    return snapshot


def _assert_states_equal(left, right) -> None:
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:]):
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                assert x.shape == y.shape
                assert x.dtype == y.dtype
                assert np.array_equal(x, y)
            else:
                assert x == y


# --------------------------------------------------------------------------
# Module hygiene + gate
# --------------------------------------------------------------------------


def test_module_imports_no_mlx_or_model_at_module_scope():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "mlx" or name.startswith("mlx.") for name in imported)
    assert not any(name.startswith("mtplx.models") for name in imported)
    assert "generation" not in imported


@pytest.fixture(autouse=True)
def _clean_flag(monkeypatch):
    reset_qsa_restore_staging_flag_cache()
    reset_counters()
    monkeypatch.delenv(FABLE_QSA_RESTORE_STAGING_ENV, raising=False)
    yield
    reset_qsa_restore_staging_flag_cache()
    reset_counters()


def test_gate_defaults_off():
    assert qsa_restore_staging_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
def test_gate_accepts_truthy(monkeypatch, value):
    monkeypatch.setenv(FABLE_QSA_RESTORE_STAGING_ENV, value)
    assert qsa_restore_staging_enabled() is True


@pytest.mark.parametrize("value", ["0", "", "off", "nope"])
def test_gate_rejects_everything_else(monkeypatch, value):
    monkeypatch.setenv(FABLE_QSA_RESTORE_STAGING_ENV, value)
    assert qsa_restore_staging_enabled() is False


def test_gate_is_read_once(monkeypatch):
    monkeypatch.setenv(FABLE_QSA_RESTORE_STAGING_ENV, "1")
    assert qsa_restore_staging_enabled() is True
    monkeypatch.setenv(FABLE_QSA_RESTORE_STAGING_ENV, "0")
    assert qsa_restore_staging_enabled() is True


# --------------------------------------------------------------------------
# Eligibility -- decided once, fails loudly
# --------------------------------------------------------------------------


def test_eligibility_reasons():
    stack, _ = _restored_stack(1_024)
    assert restore_staging_eligibility(stack, suffix_tokens=512) is None
    assert restore_staging_eligibility(stack, suffix_tokens=0) == "empty-suffix"
    assert restore_staging_eligibility([], suffix_tokens=8) == "empty-cache-stack"
    assert (
        restore_staging_eligibility([_FakeArrays()], suffix_tokens=8)
        == "no-qsa-cache"
    )
    assert (
        restore_staging_eligibility("not-a-stack", suffix_tokens=8)
        == "caches-not-a-sequence"
    )


def test_ineligible_stack_raises_rather_than_falling_back():
    with pytest.raises(QSARestoreStagingUnsupported) as excinfo:
        stage_restored_suffix([_FakeArrays()], suffix_tokens=64)
    assert "no-qsa-cache" in str(excinfo.value)
    assert COUNTERS["stage_calls"] == 0


def test_disagreeing_frontiers_raise():
    stack, _ = _restored_stack(1_024)
    stack[2].kv.offset -= 1  # one layer rolled back: not a coherent restore
    with pytest.raises(QSARestoreStagingUnsupported) as excinfo:
        stage_restored_suffix(
            stack, suffix_tokens=256, allocate_zeros=_zeros,
            materialize_cache=lambda caches: None,
        )
    assert "one token frontier" in str(excinfo.value)


def test_materialize_barrier_is_mandatory_and_grouped():
    stack, _ = _restored_stack(1_024)
    calls: list[int] = []
    stage_restored_suffix(
        stack,
        suffix_tokens=4_096,
        allocate_zeros=_zeros,
        materialize_cache=lambda caches: calls.append(len(caches)),
    )
    assert calls == [len(stack)]


def test_counters_record_engagement():
    stack, _ = _restored_stack(1_024)
    report = stage_restored_suffix(
        stack,
        suffix_tokens=8_192,
        allocate_zeros=_zeros,
        materialize_cache=lambda caches: None,
    )
    assert report.qsa_entries == 2
    assert COUNTERS["stage_calls"] == 1
    assert COUNTERS["staged_layers"] == 2
    assert COUNTERS["kv_promotions"] == report.kv_promotions == 2
    assert COUNTERS["raw_promotions"] == report.raw_promotions == 2
    assert COUNTERS["pooled_promotions"] == report.pooled_promotions == 2


# --------------------------------------------------------------------------
# The load-bearing parity test
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix_tokens,suffix_tokens,chunk",
    [
        (1_024, 2_048, 256),   # block-aligned restore, multi-chunk suffix
        (4_097, 5_000, 512),   # unaligned restore point (pooled staging edge)
        (2_353, 311, 311),     # boundary-true restore, one small fused chunk
        (16_384, 1_790, 256),  # the audit's 18K divergence shape
    ],
)
def test_staged_and_unstaged_restores_end_identical(
    prefix_tokens, suffix_tokens, chunk
):
    unstaged, unstaged_events = _restored_stack(prefix_tokens)
    staged, staged_events = _restored_stack(prefix_tokens)
    _assert_states_equal(_logical_state(unstaged), _logical_state(staged))

    report = stage_restored_suffix(
        staged,
        suffix_tokens=suffix_tokens,
        allocate_zeros=_zeros,
        materialize_cache=lambda caches: None,
    )
    # Staging alone must not move the restored state.
    _assert_states_equal(_logical_state(unstaged), _logical_state(staged))
    assert report.qsa_entries == 2

    _run_suffix(unstaged, suffix_tokens=suffix_tokens, chunk=chunk)
    _run_suffix(staged, suffix_tokens=suffix_tokens, chunk=chunk)

    _assert_states_equal(_logical_state(unstaged), _logical_state(staged))
    for cache in staged:
        if hasattr(cache, "reserve_indexer_capacity"):
            assert cache.offset == prefix_tokens + suffix_tokens

    # The point of the flag: zero lazy promotions inside the suffix forward.
    assert staged_events == []
    assert unstaged_events, "the unstaged twin must actually pay the growth"


def test_staged_pooled_mirror_matches_unstaged():
    """The derived fp32 mirror is rebuilt from content, never from zeros."""

    unstaged, _ = _restored_stack(1_024)
    staged, _ = _restored_stack(1_024)
    stage_restored_suffix(
        staged,
        suffix_tokens=1_024,
        allocate_zeros=_zeros,
        materialize_cache=lambda caches: None,
    )
    _run_suffix(unstaged, suffix_tokens=1_024, chunk=256)
    _run_suffix(staged, suffix_tokens=1_024, chunk=256)
    for a, b in zip(unstaged, staged):
        if not hasattr(a, "reserve_indexer_capacity"):
            continue
        nb = a.pooled_len
        assert np.array_equal(
            a.pooled_f32_t[..., :nb], b.pooled_f32_t[..., :nb]
        )


def test_staging_survives_a_later_trim_identically():
    """A rejected verify window trims both arms to the same logical state."""

    unstaged, _ = _restored_stack(2_048)
    staged, _ = _restored_stack(2_048)
    stage_restored_suffix(
        staged,
        suffix_tokens=1_000,
        allocate_zeros=_zeros,
        materialize_cache=lambda caches: None,
    )
    _run_suffix(unstaged, suffix_tokens=1_000, chunk=256)
    _run_suffix(staged, suffix_tokens=1_000, chunk=256)
    for stack in (unstaged, staged):
        for cache in stack:
            if hasattr(cache, "reserve_indexer_capacity"):
                cache.trim(7)
    _assert_states_equal(_logical_state(unstaged), _logical_state(staged))


def test_second_stage_of_the_same_suffix_is_a_no_op():
    stack, events = _restored_stack(1_024)
    first = stage_restored_suffix(
        stack, suffix_tokens=2_048, allocate_zeros=_zeros,
        materialize_cache=lambda caches: None,
    )
    before = _logical_state(stack)
    second = stage_restored_suffix(
        stack, suffix_tokens=2_048, allocate_zeros=_zeros,
        materialize_cache=lambda caches: None,
    )
    assert first.kv_promotions == 2
    assert second.kv_promotions == 0
    assert second.raw_promotions == 0
    assert second.pooled_promotions == 0
    _assert_states_equal(before, _logical_state(stack))
    assert events == []
