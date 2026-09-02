from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRAPHBANK = ROOT / "mtplx/graphbank.py"


def _tensor_offset_qsa_cache_type():
    """Load only the cache class so this contract test never initializes MLX."""

    tree = ast.parse(GRAPHBANK.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TensorOffsetQSACache"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            class_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    # The class body reads two module-level names on its DEFAULT path (W68's
    # armed-but-inert guard). Seed them with the real functions rather than
    # stubs: ``mtplx.runtime_options`` is pure env parsing and imports no MLX,
    # so this test still never initializes the framework.
    from mtplx.runtime_options import (
        fable_qsa_sparse_decode_enabled,
        fable_qsa_sparse_draft_enabled,
    )

    namespace: dict[str, object] = {
        "fable_qsa_sparse_decode_enabled": fable_qsa_sparse_decode_enabled,
        "fable_qsa_sparse_draft_enabled": fable_qsa_sparse_draft_enabled,
    }
    exec(compile(module, str(GRAPHBANK), "exec"), namespace)
    return namespace["TensorOffsetQSACache"]


class _FakeKV:
    def __init__(self) -> None:
        self.cache = [object(), object(), object()]
        self.step = 256
        self.offset = 12
        self.trim_calls: list[int] = []

    def trim(self, count: int) -> int:
        self.trim_calls.append(count)
        return count


def test_compile_state_is_stable_and_aux_properties_rebind_slots() -> None:
    cache_type = _tensor_offset_qsa_cache_type()
    kv = _FakeKV()
    raw = object()
    pooled = object()
    cache = cache_type(
        kv,
        raw,
        pooled,
        compress_ratio=4,
        rows_gather_kv_m4=None,
    )

    installed = cache.compile_state
    assert cache.compile_state is installed
    assert installed[0] is kv.cache
    aux = installed[1]
    assert aux == [raw, pooled]

    rebound_raw = object()
    rebound_pooled = object()
    cache.raw_keys = rebound_raw
    cache.pooled = rebound_pooled

    assert cache.compile_state is installed
    assert installed[1] is aux
    assert aux[0] is rebound_raw
    assert aux[1] is rebound_pooled
    assert cache.raw_keys is rebound_raw
    assert cache.pooled is rebound_pooled
    assert cache.state_leaves == [*kv.cache, rebound_raw, rebound_pooled]

    assert cache.trim(3) == 3
    assert kv.trim_calls == [3]


def test_compiled_tensor_offset_replay_does_not_update_rollback_metadata() -> None:
    from mtplx.graphbank import TensorOffsetKVCache

    kv = TensorOffsetKVCache(
        mx.zeros((1, 1, 8, 1), dtype=mx.float32),
        mx.zeros((1, 1, 8, 1), dtype=mx.float32),
        1,
    )
    state = [kv.cache]

    def append_one(keys, values):
        kv.update_and_fetch(keys, values)
        return kv.cache[2]

    compiled = mx.compile(append_one, inputs=state, outputs=state)
    row = mx.ones((1, 1, 1, 1), dtype=mx.float32)
    first_offset = compiled(row, row)
    mx.eval(first_offset, *kv.cache)

    poison = (
        mx.array(41, dtype=mx.int32),
        mx.full((1, 1, 2, 1), 42, dtype=mx.float32),
        mx.full((1, 1, 2, 1), 43, dtype=mx.float32),
    )
    kv.rollback_state[:] = poison
    second_offset = compiled(row, row)
    mx.eval(second_offset, *kv.cache)

    assert tuple(kv.rollback_state) == poison


def test_pr391_replay_snapshot_uses_distinct_views_that_survive_compiled_d3() -> None:
    import mtplx.generation as generation

    class Entry:
        def __init__(self) -> None:
            self.kv = SimpleNamespace(
                cache=[
                    mx.zeros((1, 1, 8, 1), dtype=mx.float32),
                    mx.zeros((1, 1, 8, 1), dtype=mx.float32),
                    mx.array(1, dtype=mx.int32),
                ],
                rollback_state=[None, None, None],
            )
            self.aux = [
                mx.zeros((8, 1), dtype=mx.float32),
                mx.zeros((2, 1), dtype=mx.float32),
            ]

        @property
        def state_leaves(self):
            return [*self.kv.cache, *self.aux]

    entry = Entry()
    core = {"cache": [entry]}
    replay_state = generation._pr391_float32_d3_state(core)

    # A second Python reference to the same mx.array object does not create
    # the two-object geometry that blocks donation.  The replay snapshot must
    # therefore hold distinct zero-copy views before the compiled transition.
    for captured, live in zip(replay_state[0], entry.state_leaves, strict=True):
        assert captured is not live
        assert np.array_equal(np.asarray(captured), np.asarray(live))

    state_tree = [entry.kv.cache, entry.aux]

    def append_one(row):
        entry.kv.cache[0] = mx.slice_update(
            entry.kv.cache[0], row, entry.kv.cache[2], axes=(2,)
        )
        entry.kv.cache[1] = mx.slice_update(
            entry.kv.cache[1], row, entry.kv.cache[2], axes=(2,)
        )
        entry.kv.cache[2] = entry.kv.cache[2] + 1
        return entry.kv.cache[2]

    compiled = mx.compile(append_one, inputs=state_tree, outputs=state_tree)
    row = mx.ones((1, 1, 1, 1), dtype=mx.float32)
    result = compiled(row)
    mx.eval(result, *entry.state_leaves, *replay_state[0])

    assert int(np.asarray(replay_state[0][2])) == 1
    assert np.count_nonzero(np.asarray(replay_state[0][0])) == 0
    assert np.count_nonzero(np.asarray(replay_state[0][1])) == 0
