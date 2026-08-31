"""CPU-only contract tests for the PR391 device MTP history handoff."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mtplx" / "pr391_mtp_handoff.py"


def _load_device_replay_binder():
    """Load only the function under test, without importing MLX."""

    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "bind_pr391_mtp_device_replay"
    )
    module = ast.Module(body=[function], type_ignores=[])
    def where(condition, when_true, when_false):
        return when_true if condition.value else when_false

    def slice_array(value, start, *, axes, slice_size):
        return _Window(value, start, axes, slice_size)

    def slice_update(value, update, start, *, axes):
        return _Updated(value, update, start, axes)

    namespace = {
        "Any": Any,
        "Callable": Callable,
        "mx": SimpleNamespace(
            int32="int32",
            where=where,
            slice=slice_array,
            slice_update=slice_update,
        ),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["bind_pr391_mtp_device_replay"]


class _DeviceScalar:
    def __init__(self, value: int, dtype: str = "int32") -> None:
        self.value = value
        self.dtype = dtype

    def __sub__(self, value: int):
        return _DeviceScalar(self.value - value, self.dtype)

    def __add__(self, other):
        return _DeviceScalar(self.value + other.value, self.dtype)

    def __eq__(self, other):
        return _DeviceScalar(int(self.value == other), "bool")

    def __floordiv__(self, other):
        return _DeviceScalar(self.value // other, self.dtype)

    def reshape(self, *_shape):
        return self

    def __getitem__(self, _index):
        return self

    def astype(self, dtype):
        return _DeviceScalar(self.value, dtype)

    def __bool__(self):  # pragma: no cover - must never be reached
        raise AssertionError("device accepted_count was materialized as bool")

    def __int__(self):  # pragma: no cover - must never be reached
        raise AssertionError("device accepted_count/offset was materialized as int")


class _Rows:
    def __init__(
        self,
        name: str,
        *,
        dtype: str = "float32",
        transforms: tuple[Any, ...] = (),
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.transforms = transforms

    def __getitem__(self, index):
        return _Rows(
            self.name,
            dtype=self.dtype,
            transforms=(*self.transforms, ("slice", index)),
        )

    def reshape(self, *shape):
        return _Rows(
            self.name,
            dtype=self.dtype,
            transforms=(*self.transforms, ("reshape", shape)),
        )

    def astype(self, dtype):
        return _Rows(
            self.name,
            dtype=dtype,
            transforms=(*self.transforms, ("astype", dtype)),
        )


class _Buffer:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape


class _Window:
    def __init__(self, source, start, axes, shape) -> None:
        self.source = source
        self.start = start
        self.axes = axes
        self.shape = shape


class _Updated:
    def __init__(self, base, update, start, axes) -> None:
        self.base = base
        self.update = update
        self.start = start
        self.axes = axes
        self.shape = base.shape


class _KV:
    def __init__(self, offset: int) -> None:
        self.cache = [
            _Buffer("keys-post-d3", (1, 2, 256, 64)),
            _Buffer("values-post-d3", (1, 2, 256, 64)),
            _DeviceScalar(offset),
        ]
        self.rollback_state = ["stale-keys", "stale-values", offset - 3]

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value):
        self.cache[2] = value


class _Entry:
    fixed_capacity = True
    ratio = 4

    def __init__(self, offset: int) -> None:
        self.kv = _KV(offset)
        self.aux = [
            _Buffer("raw-post-d3", (1, 256, 64)),
            _Buffer("pooled-post-d3", (1, 64, 64)),
        ]

    @property
    def state_leaves(self):
        return [*self.kv.cache, *self.aux]


def test_device_replay_builds_exact_width_candidates_and_selects_state():
    bind = _load_device_replay_binder()

    for accepted_width in range(4):
        entry = _Entry(offset=40)
        hidden = _Rows("verify-hidden")
        tokens = _Rows("draft-token-ids", dtype="uint32")
        append_calls = []

        def append_rows(authoritative_hidden, draft_token_ids):
            width = authoritative_hidden.transforms[-1][1][1].stop
            append_calls.append(
                (
                    entry.kv.offset.value,
                    authoritative_hidden,
                    draft_token_ids,
                )
            )
            entry.kv.cache[0] = _Buffer(f"keys-s{width}", (1, 2, 256, 64))
            entry.kv.cache[1] = _Buffer(f"values-s{width}", (1, 2, 256, 64))
            entry.aux[0] = _Buffer(f"raw-s{width}", (1, 256, 64))
            entry.aux[1] = _Buffer(f"pooled-s{width}", (1, 64, 64))
            entry.kv.offset = entry.kv.offset + _DeviceScalar(width)

        replay = bind([entry], append_rows=append_rows)
        leaves = replay(_DeviceScalar(accepted_width, "uint32"), hidden, tokens)

        assert len(append_calls) == 3
        assert [call[0] for call in append_calls] == [38, 38, 38]
        assert [call[1].name for call in append_calls] == [hidden.name] * 3
        assert [call[1].transforms for call in append_calls] == [
            (("slice", (slice(None), slice(None, width), slice(None))),)
            for width in range(1, 4)
        ]
        assert [call[2].name for call in append_calls] == [tokens.name] * 3
        assert [call[2].dtype for call in append_calls] == ["int32"] * 3
        assert [call[2].transforms for call in append_calls] == [
            (
                ("reshape", (1, 3)),
                ("astype", "int32"),
                ("slice", (slice(None), slice(None, width))),
            )
            for width in range(1, 4)
        ]
        assert entry.kv.offset.value == 38 + accepted_width
        assert entry.kv.offset.dtype == "int32"
        suffix = "post-d3" if accepted_width == 0 else f"s{accepted_width}"
        for leaf, expected_name, axis, window in (
            (entry.kv.cache[0], f"keys-{suffix}", 2, 3),
            (entry.kv.cache[1], f"values-{suffix}", 2, 3),
            (entry.aux[0], f"raw-{suffix}", 1, 3),
            (entry.aux[1], f"pooled-{suffix}", 1, 1),
        ):
            assert isinstance(leaf, _Updated)
            assert isinstance(leaf.update, _Window)
            assert leaf.update.source.name == expected_name
            assert leaf.axes == (axis,)
            assert leaf.update.shape[axis] == window
        assert entry.kv.cache[0].start.value == 38
        assert entry.kv.cache[1].start.value == 38
        assert entry.aux[0].start.value == 38
        assert entry.aux[1].start.value == 9
        assert tuple(entry.kv.rollback_state) == (None, None, None)
        assert leaves == tuple(entry.state_leaves)


def test_device_replay_rejects_non_authoritative_cache_at_bind_time():
    bind = _load_device_replay_binder()

    entry = _Entry(offset=40)
    entry.fixed_capacity = False
    try:
        bind([entry], append_rows=lambda *_args: None)
    except ValueError as error:
        assert "fixed-capacity" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-fixed cache was accepted")


def test_device_replay_source_has_no_host_width_or_offset_materialization():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "bind_pr391_mtp_device_replay"
    )
    text = ast.unparse(function)

    assert "int(entry.offset)" not in text
    assert "if accepted_count" not in text
    assert ".item()" not in text
    assert "mx.eval" not in text
    assert "[:accepted_count]" not in text
    assert "mx.slice(" in text
    assert "mx.slice_update(" in text
    assert "zip(candidates[width], selected)" not in text


def test_host_selected_replay_keeps_the_fixed_d3_rewind_on_device():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "stage_pr391_mtp_authoritative_replay"
    )
    text = ast.unparse(function)

    assert "cycle_offset" not in text
    assert "int(entry.offset)" not in text
    assert "entry.trim(2)" in text
    assert ".item()" not in text
    assert "mx.eval" not in text
