"""Host-overhead trims for the fixed-M4 decode cycle.

Every test here is off-GPU: nothing evaluates an MLX array.  The three
unconditional changes (signature memoization, the n-gram row-id CSE hoist and
the hot-row stack) are pinned as *behaviour-identical* against a reference copy
of the code they replaced; the flag-gated event trims are pinned structurally,
by proving no counter or timer lives inside a ``MTPLX_FABLE_HOST_TRIMS`` guard.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

import mtplx

_WORKTREE = str(Path(__file__).resolve().parents[1])


def test_mtplx_resolves_to_this_worktree():
    assert mtplx.__file__.startswith(_WORKTREE), mtplx.__file__


# ---------------------------------------------------------------------------
# 1. runtime._signature_parameters
# ---------------------------------------------------------------------------


class _Model:
    def mtp_forward(self, hidden, ids, *, mtp_cache=None, mtp_depth=None):
        return None

    def mtp_update_cache(self, hidden, ids, *, mtp_cache=None, **kwargs):
        return None


class _Unweakrefable:
    __slots__ = ("calls",)

    def __init__(self):
        self.calls = 0

    def __call__(self, a, b=2):
        return None


def test_signature_parameters_memoizes_per_bound_method():
    from mtplx import runtime

    model = _Model()
    first = runtime._signature_parameters(model.mtp_forward)
    second = runtime._signature_parameters(model.mtp_forward)
    assert first is second
    assert list(first) == ["hidden", "ids", "mtp_cache", "mtp_depth"]


def test_signature_parameters_calls_inspect_once(monkeypatch):
    from mtplx import runtime

    model = _Model()
    # Clear any entry a sibling test left behind for this function object.
    runtime._BOUND_SIGNATURE_PARAMETERS.pop(_Model.mtp_forward, None)
    calls = []
    real = runtime.py_inspect.signature

    def counting(func, *args, **kwargs):
        calls.append(func)
        return real(func, *args, **kwargs)

    monkeypatch.setattr(runtime.py_inspect, "signature", counting)
    for _ in range(6):
        runtime._signature_parameters(model.mtp_forward)
    assert len(calls) == 1


def test_signature_parameters_separates_bound_from_plain():
    from mtplx import runtime

    model = _Model()
    bound = runtime._signature_parameters(model.mtp_forward)
    plain = runtime._signature_parameters(_Model.mtp_forward)
    assert "self" not in bound
    assert list(plain)[0] == "self"


def test_signature_parameters_matches_inline_form_for_odd_callables():
    """Same answer the replaced `try: signature(f).parameters / except: {}`."""

    from mtplx import runtime

    for func in (_Unweakrefable(), 3, "not-callable", len):
        try:
            expected = dict(runtime.py_inspect.signature(func).parameters)
        except Exception:
            expected = {}
        assert dict(runtime._signature_parameters(func)) == expected
        # Second call must agree with the first (cache hit or fallback).
        assert dict(runtime._signature_parameters(func)) == expected


def test_signature_parameters_result_is_immutable():
    from mtplx import runtime

    params = runtime._signature_parameters(3)
    with pytest.raises(TypeError):
        params["poison"] = 1  # type: ignore[index]


class _FakeRuntime:
    """Minimum surface of MTPLXRuntime for the two memoized call sites."""

    def __init__(self, model):
        from mtplx.runtime import MTPLXRuntime

        self.model = model
        self.mtp_enabled = True
        self.counters: dict[str, int] = {}
        self.contract = type(
            "C", (), {"hidden_variant": "post_norm", "concat_order": "hidden_first"}
        )()
        self.draft_mtp = MTPLXRuntime.draft_mtp.__get__(self)
        self.update_mtp_cache = MTPLXRuntime.update_mtp_cache.__get__(self)

    def _count(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1


def test_draft_and_update_do_not_re_signature_per_call(monkeypatch):
    from mtplx import runtime

    calls = []
    real = runtime.py_inspect.signature

    class _M:
        def mtp_forward(self, hidden, ids, **kwargs):
            return ("logits", "hidden")

        def mtp_update_cache(self, hidden, ids, **kwargs):
            return "hidden"

    runtime._BOUND_SIGNATURE_PARAMETERS.pop(_M.mtp_forward, None)
    runtime._BOUND_SIGNATURE_PARAMETERS.pop(_M.mtp_update_cache, None)

    def counting(func, *args, **kwargs):
        calls.append(func)
        return real(func, *args, **kwargs)

    monkeypatch.setattr(runtime.py_inspect, "signature", counting)
    monkeypatch.setattr(runtime, "mtp_adapter_depth", _null_context)
    rt = _FakeRuntime(_M())
    for _ in range(4):
        rt.draft_mtp("hidden", [1])
        rt.update_mtp_cache("hidden", [1])
    # One per distinct callable, not one per call (8 calls -> 2 signatures).
    assert len(calls) == 2


class _null_context:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# 2. _ngram_rows_np -- CSE hoist must not move a single bit
# ---------------------------------------------------------------------------


def _ngram_rows_np_reference(
    ids_np,
    prev_np,
    *,
    mult,
    sizes,
    offs,
    eos: int,
    ngram_size: int,
    heads_per_ngram: int,
):
    """Verbatim copy of the pre-hoist implementation."""

    hist = np.concatenate([prev_np, ids_np], axis=1)

    def shift(h, s):
        if s == 0:
            return h
        b, ln = h.shape
        pos = np.arange(ln, dtype=np.int64)[None, :]
        eos_pos = np.where(h == eos, pos, np.int64(-1))
        prev_incl = np.maximum.accumulate(eos_pos, axis=1)
        prev = np.concatenate(
            [np.full((b, 1), -1, dtype=np.int64), prev_incl[:, :-1]], axis=1
        )
        pos_in_seg = pos - (prev + 1)
        src = np.maximum(pos - s, 0)
        shifted = np.take_along_axis(h, src, axis=1)
        valid = (pos_in_seg >= s) & (pos - s >= 0)
        return np.where(valid, shifted, np.int64(eos))

    shifted = [shift(hist, s) for s in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * mult[0]
        for p in range(1, ngram):
            mixed = mixed ^ (shifted[p] * mult[p])
        blocks.append(mixed[..., None] % sizes[start:end] + offs[start:end])
    S = ids_np.shape[1]
    rows = np.concatenate(blocks, axis=-1)[:, -S:]
    return rows, hist[:, -(ngram_size - 1) :]


def _row_layout(rng, ngram_size, heads_per_ngram):
    heads = (ngram_size - 1) * heads_per_ngram
    mult = rng.integers(1, 2**31, size=ngram_size).astype(np.int64)
    sizes = rng.integers(1_000, 2_000_000, size=heads).astype(np.int64)
    offs = (np.arange(heads, dtype=np.int64) * 2_000_000).astype(np.int64)
    return mult, sizes, offs


def test_ngram_rows_np_matches_reference_on_random_histories():
    from mtplx.models.qwen4_exp import _ngram_rows_np

    rng = np.random.default_rng(20260901)
    eos = 151_643
    for ngram_size in (2, 3, 4, 5):
        mult, sizes, offs = _row_layout(rng, ngram_size, 8)
        kwargs = dict(
            mult=mult,
            sizes=sizes,
            offs=offs,
            eos=eos,
            ngram_size=ngram_size,
            heads_per_ngram=8,
        )
        for _ in range(40):
            batch = int(rng.integers(1, 3))
            width = int(rng.integers(1, 5))
            prev = rng.integers(0, 250_000, size=(batch, ngram_size - 1))
            ids = rng.integers(0, 250_000, size=(batch, width))
            # Sprinkle EOS so the segment-start scan is exercised.
            mask = rng.random(prev.shape) < 0.3
            prev = np.where(mask, eos, prev).astype(np.int64)
            mask = rng.random(ids.shape) < 0.3
            ids = np.where(mask, eos, ids).astype(np.int64)
            got_rows, got_hist = _ngram_rows_np(ids, prev, **kwargs)
            ref_rows, ref_hist = _ngram_rows_np_reference(ids, prev, **kwargs)
            assert np.array_equal(got_rows, ref_rows)
            assert np.array_equal(got_hist, ref_hist)
            assert got_rows.dtype == ref_rows.dtype


def test_ngram_rows_np_matches_reference_on_ple_conv_edges():
    """The ple_conv history corners: all-EOS, EOS only at 0, no EOS at all."""

    from mtplx.models.qwen4_exp import _ngram_rows_np

    rng = np.random.default_rng(7)
    eos = 151_643
    ngram_size, heads = 3, 8
    mult, sizes, offs = _row_layout(rng, ngram_size, heads)
    kwargs = dict(
        mult=mult,
        sizes=sizes,
        offs=offs,
        eos=eos,
        ngram_size=ngram_size,
        heads_per_ngram=heads,
    )
    cases = [
        # (prev, ids)
        ([[eos, eos]], [[eos, eos, eos, eos]]),
        ([[eos, eos]], [[11, 22, 33, 44]]),  # fresh prompt tail
        ([[eos, 5]], [[11, eos, 33, 44]]),
        ([[5, 6]], [[11, 22, 33, 44]]),  # no EOS anywhere
        ([[5, 6]], [[eos, 22, 33, 44]]),
        ([[5, 6]], [[11, 22, 33, eos]]),
        ([[0, 0]], [[0]]),  # width-1 (AR/staged shape)
    ]
    for prev, ids in cases:
        prev_np = np.asarray(prev, dtype=np.int64)
        ids_np = np.asarray(ids, dtype=np.int64)
        got_rows, got_hist = _ngram_rows_np(ids_np, prev_np, **kwargs)
        ref_rows, ref_hist = _ngram_rows_np_reference(ids_np, prev_np, **kwargs)
        assert np.array_equal(got_rows, ref_rows), (prev, ids)
        assert np.array_equal(got_hist, ref_hist), (prev, ids)


def test_fixed_m4_geometry_yields_64_row_ids():
    """The shape the fixed-M4 aux gathers: 4 tokens x 16 heads."""

    from mtplx.models.qwen4_exp import _ngram_rows_np

    rng = np.random.default_rng(3)
    mult, sizes, offs = _row_layout(rng, 3, 8)
    rows, hist = _ngram_rows_np(
        np.asarray([[11, 22, 33, 44]], dtype=np.int64),
        np.asarray([[7, 8]], dtype=np.int64),
        mult=mult,
        sizes=sizes,
        offs=offs,
        eos=151_643,
        ngram_size=3,
        heads_per_ngram=8,
    )
    assert rows.shape == (1, 4, 16)
    assert rows.reshape(-1).size == 64
    assert hist.shape == (1, 2)


# ---------------------------------------------------------------------------
# 3. _stack_hot_rows / _rows_matrices -- same bytes, same LRU
# ---------------------------------------------------------------------------


def test_stack_hot_rows_matches_np_stack():
    from mtplx.models.qwen4_exp import _stack_hot_rows

    rng = np.random.default_rng(11)
    for dtype, width in ((np.uint32, 20), (np.uint16, 5), (np.int64, 3)):
        rows = [
            (
                rng.integers(0, 255, size=width).astype(dtype),
                rng.integers(0, 255, size=2).astype(np.uint16),
            )
            for _ in range(64)
        ]
        for j in (0, 1):
            got = _stack_hot_rows(rows, j)
            ref = np.stack([row[j] for row in rows])
            assert np.array_equal(got, ref)
            assert got.dtype == ref.dtype
            assert got.shape == ref.shape


def _rows_matrices_reference(self, flat, names):
    """Verbatim copy of the pre-change hot-row branch."""

    uniq, inverse = np.unique(flat, return_inverse=True)
    if not (0 < len(uniq) <= self._HOT_PATH_MAX_ROWS and self._hot_cap_rows):
        if self._pool is not None and len(uniq):
            self._warm(uniq)
        return {
            name: np.ascontiguousarray(self._maps[name][0][uniq])[inverse]
            for name in names
        }
    hot = self._hot
    miss = [int(r) for r in uniq if int(r) not in hot]
    if miss:
        miss_np = np.asarray(miss, dtype=np.int64)
        if self._pool is not None:
            self._warm(miss_np)
        fetched = {
            name: np.ascontiguousarray(self._maps[name][0][miss_np])
            for name in names
        }
        for i, r in enumerate(miss):
            hot[r] = tuple(fetched[name][i] for name in names)
    self.hot_hits += len(uniq) - len(miss)
    self.hot_misses += len(miss)
    rows = []
    for r in uniq:
        key = int(r)
        rows.append(hot[key])
        hot.move_to_end(key)
    while len(hot) > self._hot_cap_rows:
        hot.popitem(last=False)
    return {
        name: np.stack([row[j] for row in rows])[inverse]
        for j, name in enumerate(names)
    }


def _make_gather(cap_rows, table_rows=4096):
    from mtplx.models.qwen4_exp import _SidecarGather

    rng = np.random.default_rng(table_rows)
    inst = object.__new__(_SidecarGather)
    inst._hot = OrderedDict()
    inst._hot_cap_rows = cap_rows
    inst._pool = None
    inst.hot_hits = 0
    inst.hot_misses = 0
    inst._maps = {
        "weight": (rng.integers(0, 2**31, size=(table_rows, 20)).astype(np.uint32), "U32"),
        "scales": (rng.integers(0, 2**15, size=(table_rows, 5)).astype(np.uint16), "BF16"),
        "biases": (rng.integers(0, 2**15, size=(table_rows, 5)).astype(np.uint16), "BF16"),
    }
    return inst


def test_rows_matrices_is_byte_and_lru_identical():
    names = ("weight", "scales", "biases")
    live = _make_gather(96)
    ref = _make_gather(96)
    rng = np.random.default_rng(99)
    for _ in range(40):
        # 64 row ids per cycle, drawn from a Zipf-ish hot set so the LRU both
        # hits and evicts.
        flat = rng.integers(0, 300, size=64).astype(np.int64)
        got = live._rows_matrices(flat, names)
        want = _rows_matrices_reference(ref, flat, names)
        for name in names:
            assert np.array_equal(got[name], want[name]), name
            assert got[name].dtype == want[name].dtype
        assert list(live._hot.keys()) == list(ref._hot.keys())
        assert live.hot_hits == ref.hot_hits
        assert live.hot_misses == ref.hot_misses
    assert live.hot_hits > 0 and live.hot_misses > 0
    assert len(live._hot) == 96  # eviction actually engaged


def test_rows_matrices_bypass_branch_untouched():
    """Prefill-sized gathers must still take the vectorized memmap path."""

    live = _make_gather(96)
    ref = _make_gather(96)
    rng = np.random.default_rng(5)
    flat = rng.integers(0, 4096, size=20_000).astype(np.int64)
    live._HOT_PATH_MAX_ROWS = 8  # force the bypass
    ref._HOT_PATH_MAX_ROWS = 8
    got = live._rows_matrices(flat, ("weight",))
    want = _rows_matrices_reference(ref, flat, ("weight",))
    assert np.array_equal(got["weight"], want["weight"])
    assert live._hot == OrderedDict()
    assert live.hot_hits == 0 and live.hot_misses == 0


# ---------------------------------------------------------------------------
# 4. MTPLX_FABLE_HOST_TRIMS -- flag default + what the guards may contain
# ---------------------------------------------------------------------------


def test_host_trims_flag_tracks_the_env_and_defaults_off():
    import os

    from mtplx import generation

    raw = os.environ.get("MTPLX_FABLE_HOST_TRIMS", "")
    expected = raw.strip().lower() in {"1", "true", "yes", "on"}
    assert generation._FABLE_HOST_TRIMS is expected
    if raw == "":
        assert generation._FABLE_HOST_TRIMS is False


def _trim_guards(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "_FABLE_HOST_TRIMS" in names:
            guards.append(node)
    return guards


def _event_subscript(node):
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "event"
    )


def test_trim_guards_touch_only_event_decoration():
    """No counter, timer or receipt field can be inside a trim guard.

    Every stats field is either a plain local mutated with ``+=`` or a
    ``_add_timing(event, ...)`` call.  If neither ever appears under a
    ``_FABLE_HOST_TRIMS`` branch, the flag cannot move a single number in
    GenerationStats or in the abba per-cycle receipt.
    """

    from mtplx import generation

    guards = _trim_guards(generation.generate_mtpk)
    assert len(guards) == 2, [ast.dump(g.test) for g in guards]
    for guard in guards:
        for branch in (guard.body, guard.orelse):
            for stmt in branch:
                for node in ast.walk(stmt):
                    assert not isinstance(node, ast.AugAssign), ast.dump(node)
                    if isinstance(node, ast.Call):
                        name = getattr(node.func, "attr", None) or getattr(
                            node.func, "id", None
                        )
                        assert name in {"append", "int", "float"}, name
                    if isinstance(node, ast.Assign):
                        assert all(
                            _event_subscript(t) for t in node.targets
                        ), ast.dump(node)
                    if isinstance(node, (ast.Return, ast.Raise, ast.Break)):
                        raise AssertionError(ast.dump(node))


def test_trim_guard_continue_skips_only_event_appends():
    """The one `continue` a trim uses must not jump over a counter.

    Guard 1 lives inside the per-depth draft loop and ends in `continue`.
    That is only safe if every statement it skips -- i.e. everything after
    the guard in its enclosing block -- is event decoration.
    """

    from mtplx import generation

    tree = ast.parse(textwrap.dedent(inspect.getsource(generation.generate_mtpk)))
    checked = 0
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            for index, stmt in enumerate(block):
                names = (
                    {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
                    if isinstance(stmt, ast.If)
                    else set()
                )
                if "_FABLE_HOST_TRIMS" not in names:
                    continue
                has_continue = any(
                    isinstance(n, ast.Continue) for n in ast.walk(stmt)
                )
                # Statements before the guard must carry every counter.
                if not has_continue:
                    continue
                checked += 1
                for later in block[index + 1 :]:
                    assert isinstance(later, ast.Expr), ast.dump(later)
                    call = later.value
                    assert isinstance(call, ast.Call)
                    assert _event_subscript(
                        getattr(call.func, "value", None)
                    ), ast.dump(later)
    assert checked == 1


def test_trimmed_draft_entry_still_serves_every_reader():
    """The trimmed per-depth dict is the one the accept path writes into."""

    # The reduced entry the trim builds ...
    entry = {"depth": 1, "token": 4242}
    # ... accept bookkeeping (generation.py, the `_pr391_depth` loop) ...
    entry["accepted"] = True
    entry["accept_probability"] = 0.5
    entry["correction"] = 4242
    # ... the fork-EV reader ...
    assert entry.get("constraint_clamped") is None
    # ... and the adaptive-dtemp reader.
    assert entry.get("accept_probability") == 0.5


def test_preserved_event_fields_are_written_outside_any_guard():
    """rejected_at_depth / timing_s must not be reachable only under a trim."""

    from mtplx import generation

    src = textwrap.dedent(inspect.getsource(generation.generate_mtpk))
    tree = ast.parse(src)
    guarded = set()
    for guard in _trim_guards(generation.generate_mtpk):
        for branch in (guard.body, guard.orelse):
            for stmt in branch:
                guarded.update(id(n) for n in ast.walk(stmt))

    saw_reject = saw_timing = False
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    _event_subscript(target)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "rejected_at_depth"
                ):
                    saw_reject = True
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_add_timing":
            saw_timing = True
    assert saw_reject, "event['rejected_at_depth'] no longer written outside a guard"
    assert saw_timing, "_add_timing no longer called outside a guard"
