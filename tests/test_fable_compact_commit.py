"""Contract tests for the compact width-a commit (MTPLX_FABLE_COMPACT_COMMIT).

Everything here runs on the CPU stream with synthetic shapes.  No Metal
kernel is dispatched: ``gated_delta_update`` falls back to its ops reference
whenever the default device is not the GPU, which is exactly what these tests
pin the default device to.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.kernels.qwen4_m4_state_handoff import (
    QWEN4_M4_VERIFY_WIDTH,
    bind_qwen4_m4_state_handoff,
)
from mtplx.qwen4_fixed_verify import (
    _bind_fixed_m4_device_commit,
    fable_compact_commit_enabled,
)


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Pin every op in this module to the CPU stream, then put it back.

    ``gated_delta_update(use_kernel=True)`` dispatches its Metal kernel only
    when the default device is the GPU; on CPU it takes the ops reference.
    Restoring the previous device keeps this module from changing the device
    for any other test that shares the process.
    """

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)

ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT / "mtplx" / "generation.py"
HANDOFF = ROOT / "mtplx" / "pr391_mtp_handoff.py"

CONV_WIDTH = 6
PLE_CONV_WIDTH = 5
HIDDEN_WIDTH = 8
AUX_WIDTH = 4
B, T, HK, DK, HV, DV = 1, QWEN4_M4_VERIFY_WIDTH, 1, 2, 1, 2


# --------------------------------------------------------------------------
# (a) flag plumbing
# --------------------------------------------------------------------------


def _read_flag(monkeypatch, value):
    import mtplx.qwen4_fixed_verify as module

    monkeypatch.setattr(module, "_FABLE_COMPACT_COMMIT", None, raising=False)
    if value is None:
        monkeypatch.delenv("MTPLX_FABLE_COMPACT_COMMIT", raising=False)
    else:
        monkeypatch.setenv("MTPLX_FABLE_COMPACT_COMMIT", value)
    return module.fable_compact_commit_enabled()


def test_compact_commit_defaults_off(monkeypatch):
    assert _read_flag(monkeypatch, None) is False
    assert _read_flag(monkeypatch, "0") is False
    assert _read_flag(monkeypatch, "") is False


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_compact_commit_accepts_the_documented_truthy_spellings(
    monkeypatch, value
):
    assert _read_flag(monkeypatch, value) is True


def test_compact_commit_env_is_read_once(monkeypatch):
    import mtplx.qwen4_fixed_verify as module

    monkeypatch.setattr(module, "_FABLE_COMPACT_COMMIT", None, raising=False)
    monkeypatch.setenv("MTPLX_FABLE_COMPACT_COMMIT", "1")
    assert module.fable_compact_commit_enabled() is True
    monkeypatch.setenv("MTPLX_FABLE_COMPACT_COMMIT", "0")
    # A measured cycle must never observe a mid-request flip.
    assert module.fable_compact_commit_enabled() is True


def test_process_flag_is_off_in_this_test_process():
    assert os.environ.get("MTPLX_FABLE_COMPACT_COMMIT") in (None, "", "0")
    assert fable_compact_commit_enabled() is False


# --------------------------------------------------------------------------
# (b) width-a commit selects the same state as all-width + mx.where
# --------------------------------------------------------------------------


class _ArraysCache:
    """Minimal stand-in for the family ``ArraysCache`` used by the commit."""

    def __init__(self, size: int) -> None:
        self.cache = [None] * int(size)
        self._mtplx_verify_rows = None
        self._mtplx_verify_ple = None
        self._mtplx_verify_compiled_aux = None

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value) -> None:
        self.cache[index] = value


class _KV:
    def __init__(self, offset: int) -> None:
        self.cache = [None, None, mx.array(offset, dtype=mx.int32)]
        self.rollback_state = ["stale", "stale", "stale"]


class _QSACache:
    def __init__(self, offset: int) -> None:
        self.kv = _KV(offset)

    @property
    def state_leaves(self):
        return [self.kv.cache[2]]


class _LinearAttn:
    """Callable GDN stub that is also the ``A_log`` / ``dt_bias`` owner."""

    def __init__(self, tag: int) -> None:
        self.tag = tag
        self.A_log = mx.zeros((HV,), dtype=mx.float32)
        self.dt_bias = mx.zeros((HV,), dtype=mx.float32)

    def __call__(self, mixed, mask, cache):
        width = int(mixed.shape[1])
        cache[0] = mx.full(
            (1, 3, CONV_WIDTH), float(300 + 10 * self.tag + width)
        )
        cache[1] = mx.full(
            (B, HV, DV, DK), float(400 + 10 * self.tag + width)
        )
        return mixed


class _PLE:
    def __call__(self, hidden, ids, cache):
        width = int(hidden.shape[1])
        cache[2] = mx.full((1, 9, PLE_CONV_WIDTH), float(100 + width))
        cache[3] = mx.full((1, 2), 200 + width, dtype=mx.int64)
        return mx.full(hidden.shape, float(width), dtype=hidden.dtype)


def _capture_rows(seed: int):
    mx.random.seed(seed)
    return (
        mx.random.normal((1, QWEN4_M4_VERIFY_WIDTH, CONV_WIDTH)),
        mx.random.normal((B, T, HK, DK)),
        mx.random.normal((B, T, HK, DK)),
        mx.random.normal((B, T, HV, DV)),
        mx.random.normal((B, T, HV)),
        mx.random.normal((B, T, HV)),
    )


def _build_fixture():
    """Return (runtime, cache, snapshot_states, verify_hidden)."""

    plain_entry = _ArraysCache(2)
    ple_entry = _ArraysCache(4)
    qsa_entry = _QSACache(40)
    cache = [plain_entry, ple_entry, qsa_entry]

    plain_entry._mtplx_verify_rows = _capture_rows(11)
    ple_entry._mtplx_verify_rows = _capture_rows(23)
    mx.random.seed(31)
    ple_entry._mtplx_verify_ple = (
        mx.random.normal((1, QWEN4_M4_VERIFY_WIDTH, HIDDEN_WIDTH)),
        mx.arange(QWEN4_M4_VERIFY_WIDTH, dtype=mx.int32).reshape(1, -1) + 7,
        mx.random.normal((1, QWEN4_M4_VERIFY_WIDTH, PLE_CONV_WIDTH)),
    )
    ple_entry._mtplx_verify_compiled_aux = mx.random.normal(
        (1, QWEN4_M4_VERIFY_WIDTH, AUX_WIDTH)
    )

    layers = [
        SimpleNamespace(is_linear=True, linear_attn=_LinearAttn(1), ple=None),
        SimpleNamespace(
            is_linear=True,
            linear_attn=_LinearAttn(2),
            ple=_PLE(),
            attn_hyper_connection=lambda hidden: (hidden, None, None),
        ),
        SimpleNamespace(is_linear=False, ple=None),
    ]
    runtime = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        _mtplx_qwen4_m4_state_handoff_binding=SimpleNamespace(
            select_windows=bind_qwen4_m4_state_handoff(),
            gdn_layer_indices=(0, 1),
            qsa_layer_indices=(2,),
            ple_layer_index=1,
        ),
    )

    mx.random.seed(101)
    snapshot_states = [
        (
            mx.random.normal((1, 3, CONV_WIDTH)),
            mx.random.normal((B, HV, DV, DK)),
        ),
        (
            mx.random.normal((1, 3, CONV_WIDTH)),
            mx.random.normal((B, HV, DV, DK)),
            mx.random.normal((1, 9, PLE_CONV_WIDTH)),
            mx.arange(2, dtype=mx.int64).reshape(1, 2),
        ),
        None,
    ]
    verify_hidden = mx.random.normal(
        (1, QWEN4_M4_VERIFY_WIDTH, HIDDEN_WIDTH)
    )
    return runtime, cache, snapshot_states, verify_hidden


def _committed(cache):
    return {
        "plain_conv": cache[0][0],
        "plain_delta": cache[0][1],
        "ple_conv": cache[1][0],
        "ple_delta": cache[1][1],
        "ple_short_conv": cache[1][2],
        "ple_history": cache[1][3],
        "qsa_offset": cache[2].kv.cache[2],
    }


def _same(left, right) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(mx.all(left == right).item())


@pytest.mark.parametrize("accepted", range(QWEN4_M4_VERIFY_WIDTH))
def test_width_commit_selects_the_all_width_state(accepted):
    runtime, cache, snapshot_states, verify_hidden = _build_fixture()
    commit = _bind_fixed_m4_device_commit(runtime, cache)
    reference_hidden, reference_roots = commit(
        mx.array([accepted], dtype=mx.uint32), snapshot_states, verify_hidden
    )
    mx.eval(reference_hidden, *reference_roots)
    reference = _committed(cache)

    runtime, cache, snapshot_states, verify_hidden = _build_fixture()
    commit = _bind_fixed_m4_device_commit(runtime, cache)
    compact_hidden, compact_roots = commit.commit_width(
        accepted, snapshot_states, verify_hidden
    )
    mx.eval(compact_hidden, *compact_roots)
    compact = _committed(cache)

    assert _same(compact_hidden, reference_hidden)
    for name in reference:
        assert _same(compact[name], reference[name]), name
    assert len(compact_roots) == len(reference_roots)


@pytest.mark.parametrize("accepted", range(QWEN4_M4_VERIFY_WIDTH))
def test_width_commit_clears_the_same_capture_attributes(accepted):
    runtime, cache, snapshot_states, verify_hidden = _build_fixture()
    commit = _bind_fixed_m4_device_commit(runtime, cache)
    _hidden, roots = commit.commit_width(
        accepted, snapshot_states, verify_hidden
    )
    mx.eval(*roots)
    for entry in (cache[0], cache[1]):
        assert entry._mtplx_verify_rows is None
    assert cache[1]._mtplx_verify_ple is None
    assert cache[1]._mtplx_verify_compiled_aux is None
    assert cache[2].kv.rollback_state == [None, None, None]


def test_width_commit_runs_one_ple_replay_at_most():
    """Below the whole window exactly one logical width is constructed."""

    for accepted in range(QWEN4_M4_VERIFY_WIDTH):
        runtime, cache, snapshot_states, verify_hidden = _build_fixture()
        widths: list[int] = []
        ple_layer = runtime.model.model.layers[1]
        inner_ple = ple_layer.ple

        def record(hidden, ids, entry, _inner=inner_ple, _sink=widths):
            _sink.append(int(hidden.shape[1]))
            return _inner(hidden, ids, entry)

        ple_layer.ple = record
        commit = _bind_fixed_m4_device_commit(runtime, cache)
        _hidden, roots = commit.commit_width(
            accepted, snapshot_states, verify_hidden
        )
        mx.eval(*roots)
        expected = [] if accepted == QWEN4_M4_VERIFY_WIDTH - 1 else [accepted + 1]
        assert widths == expected, accepted


def test_all_width_commit_still_runs_three_ple_replays():
    runtime, cache, snapshot_states, verify_hidden = _build_fixture()
    widths: list[int] = []
    ple_layer = runtime.model.model.layers[1]
    inner_ple = ple_layer.ple

    def record(hidden, ids, entry):
        widths.append(int(hidden.shape[1]))
        return inner_ple(hidden, ids, entry)

    ple_layer.ple = record
    commit = _bind_fixed_m4_device_commit(runtime, cache)
    _hidden, roots = commit(
        mx.array([0], dtype=mx.uint32), snapshot_states, verify_hidden
    )
    mx.eval(*roots)
    assert widths == [1, 2, 3]


@pytest.mark.parametrize("accepted", [-1, 4, 7])
def test_width_commit_rejects_widths_outside_the_m4_window(accepted):
    runtime, cache, snapshot_states, verify_hidden = _build_fixture()
    commit = _bind_fixed_m4_device_commit(runtime, cache)
    with pytest.raises(ValueError):
        commit.commit_width(accepted, snapshot_states, verify_hidden)


# --------------------------------------------------------------------------
# (c) MTP handoff at width a matches the three-candidate selection
# --------------------------------------------------------------------------


def _load_handoff_namespace():
    """Load both handoff entry points without importing MLX for real."""

    tree = ast.parse(HANDOFF.read_text())
    wanted = {
        "bind_pr391_mtp_device_replay",
        "stage_pr391_mtp_authoritative_replay",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(body) == 2

    def where(condition, when_true, when_false):
        return when_true if condition.value else when_false

    def slice_array(value, start, *, axes, slice_size):
        return ("window", value, axes, slice_size)

    def slice_update(value, update, start, *, axes):
        return ("updated", value, update, axes)

    namespace = {
        "Any": object,
        "Callable": object,
        "mx": SimpleNamespace(
            int32="int32",
            where=where,
            slice=slice_array,
            slice_update=slice_update,
        ),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(HANDOFF), "exec"), namespace)
    return namespace


class _Scalar:
    def __init__(self, value: int, dtype: str = "int32") -> None:
        self.value = value
        self.dtype = dtype

    def __sub__(self, other):
        return _Scalar(self.value - int(other), self.dtype)

    def __add__(self, other):
        return _Scalar(self.value + int(getattr(other, "value", other)), self.dtype)

    def __eq__(self, other):
        return _Scalar(int(self.value == other), "bool")

    def __floordiv__(self, other):
        return _Scalar(self.value // int(other), self.dtype)

    def reshape(self, *_shape):
        return self

    def __getitem__(self, _index):
        return self

    def astype(self, dtype):
        return _Scalar(self.value, dtype)

    def __int__(self):
        return self.value


class _Rows:
    def __init__(self, name, transforms=()):
        self.name = name
        self.transforms = tuple(transforms)

    def __getitem__(self, index):
        return _Rows(self.name, (*self.transforms, ("slice", index)))

    def reshape(self, *shape):
        return _Rows(self.name, (*self.transforms, ("reshape", shape)))

    def astype(self, dtype):
        return _Rows(self.name, (*self.transforms, ("astype", dtype)))


class _Buffer:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _HandoffKV:
    def __init__(self, offset):
        self.cache = [
            _Buffer("keys-post-d3", (1, 2, 256, 64)),
            _Buffer("values-post-d3", (1, 2, 256, 64)),
            _Scalar(offset),
        ]
        self.rollback_state = [None, None, None]

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value):
        self.cache[2] = value

    def trim(self, n):
        self.cache[2] = _Scalar(max(0, self.cache[2].value - int(n)))
        return int(n)


class _HandoffEntry:
    fixed_capacity = True
    ratio = 4

    def __init__(self, offset):
        self.kv = _HandoffKV(offset)
        self.aux = [
            _Buffer("raw-post-d3", (1, 256, 64)),
            _Buffer("pooled-post-d3", (1, 64, 64)),
        ]

    @property
    def state_leaves(self):
        return [*self.kv.cache, *self.aux]

    def trim(self, n):
        return self.kv.trim(n)


def _appender(entry, sink):
    def append(hidden, tokens):
        width = int(hidden.transforms[-1][1][1].stop or 0)
        sink.append(width)
        entry.kv.cache[0] = _Buffer(f"keys-s{width}", (1, 2, 256, 64))
        entry.kv.cache[1] = _Buffer(f"values-s{width}", (1, 2, 256, 64))
        entry.aux[0] = _Buffer(f"raw-s{width}", (1, 256, 64))
        entry.aux[1] = _Buffer(f"pooled-s{width}", (1, 64, 64))
        entry.kv.offset = entry.kv.offset + _Scalar(width)

    return append


@pytest.mark.parametrize("accepted", range(QWEN4_M4_VERIFY_WIDTH))
def test_host_width_handoff_matches_the_device_candidate(accepted):
    namespace = _load_handoff_namespace()

    device_entry = _HandoffEntry(offset=40)
    device_widths: list[int] = []
    replay = namespace["bind_pr391_mtp_device_replay"](
        [device_entry], append_rows=_appender(device_entry, device_widths)
    )
    replay(
        _Scalar(accepted, "uint32"),
        _Rows("verify-hidden"),
        _Rows("draft-token-ids"),
    )
    assert device_widths == [1, 2, 3]
    device_offset = device_entry.kv.offset.value
    selected = "post-d3" if accepted == 0 else f"s{accepted}"

    host_entry = _HandoffEntry(offset=40)
    host_widths: list[int] = []
    namespace["stage_pr391_mtp_authoritative_replay"](
        [host_entry],
        accepted_count=accepted,
        authoritative_hidden=_Rows("verify-hidden"),
        draft_token_ids=_Rows("draft-token-ids"),
        append_row=_appender(host_entry, host_widths),
    )

    assert host_widths == ([] if accepted == 0 else [accepted])
    assert host_entry.kv.offset.value == device_offset == 38 + accepted
    # The device path installs the selected candidate's three-row window; the
    # host path installs the same candidate directly.
    for leaf in (
        host_entry.kv.cache[0],
        host_entry.kv.cache[1],
        host_entry.aux[0],
        host_entry.aux[1],
    ):
        assert leaf.name.endswith(selected)


# --------------------------------------------------------------------------
# (a)/(d) cycle ordering and RNG accounting
# --------------------------------------------------------------------------


def _compact_branch_source():
    tree = ast.parse(GENERATION.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "_pr391_compact_commit"
        ):
            return node
    raise AssertionError("compact commit branch not found in generate_mtpk")


def _called_names(nodes):
    names: list[str] = []
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


TRACKED = (
    "commit_fixed_m4_device_window",
    "commit_fixed_m4_host_window",
    "_pr391_queue_device_verifier_mtp_replay",
    "_pr391_stage_verifier_mtp_replay",
    "_pr391_queue_device_canonical_d3",
    "_pr391_decode_float32_verifier_decision",
)


def _tracked_order(nodes):
    return [name for name in _called_names(nodes) if name in TRACKED]


def test_flag_off_branch_keeps_the_original_cycle_order():
    branch = _compact_branch_source()
    assert _tracked_order(branch.orelse) == [
        "commit_fixed_m4_device_window",
        "_pr391_queue_device_verifier_mtp_replay",
        "_pr391_queue_device_canonical_d3",
        "_pr391_decode_float32_verifier_decision",
    ]


def test_compact_branch_materializes_the_decision_first():
    branch = _compact_branch_source()
    assert _tracked_order(branch.body) == [
        "_pr391_decode_float32_verifier_decision",
        "commit_fixed_m4_host_window",
        "_pr391_stage_verifier_mtp_replay",
        "_pr391_queue_device_canonical_d3",
    ]


def test_both_branches_draw_identically_from_the_rng_tape():
    branch = _compact_branch_source()

    def rng_calls(nodes):
        calls: list[str] = []
        for node in nodes:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "rng"
                ):
                    calls.append(child.func.attr)
        return sorted(calls)

    # The only tape consumer inside either branch is the decision decode
    # itself, which commits the device-reported draws exactly once.
    assert rng_calls(branch.body) == rng_calls(branch.orelse) == []
    assert (
        _tracked_order(branch.body).count(
            "_pr391_decode_float32_verifier_decision"
        )
        == 1
    )
    assert (
        _tracked_order(branch.orelse).count(
            "_pr391_decode_float32_verifier_decision"
        )
        == 1
    )


def test_carried_d3_reservation_asserts_are_untouched():
    """The reorder must not weaken the carried-D3 cursor identity check."""

    tree = ast.parse(GENERATION.read_text())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and any(
            isinstance(arg, ast.Constant)
            and arg.value == "PR391 carried D3 RNG cursor changed"
            for arg in node.exc.args
        )
    )
    assert guard is not None
    source = GENERATION.read_text()
    assert "_pr391_carry_reservation = rng.reserve_device_choices(" in source
    assert 'int(_pr391_carry["descriptor_offset"])' in source
    # The lookahead descriptor is still the host reservation offset plus the
    # device-reported draw count, in that order.
    assert (
        "_pr391_next_descriptor_offset = (\n"
        "                int(verifier_reservation.offset)\n"
        "                + int(_pr391_verifier_decision[5])\n"
        "            )" in source
    )


def test_compact_branch_adds_no_extra_async_eval_boundary():
    """Only the decision sync moves; the enqueue boundaries stay matched."""

    branch = _compact_branch_source()

    def async_evals(nodes):
        return sum(
            1
            for node in nodes
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "async_eval"
        )

    assert async_evals(branch.body) == async_evals(branch.orelse) == 1
    # ``_pr391_stage_verifier_mtp_replay`` deliberately leaves its leaves
    # unrooted; the queue wrapper that does root them is the parity helper.
    source = GENERATION.read_text()
    assert "def _pr391_stage_verifier_mtp_replay(" in source
    stage = source.index("def _pr391_stage_verifier_mtp_replay(")
    queue = source.index("def _pr391_queue_verifier_mtp_replay(", stage)
    assert "mx.async_eval(" not in source[stage:queue]


def test_async_enqueue_guard_requires_the_width_commit_when_armed():
    tree = ast.parse(GENERATION.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pr391_require_fixed_m4_async_enqueue"
    )
    text = ast.unparse(function)
    assert "device_commit_width" in text
    assert "_pr391_compact_commit_enabled()" in text
