"""W62 MTPLX_FABLE_PLE_BOUNDARY: flags, equivalence, and engagement.

Pure Python + NumPy.  ``mtplx.ple_boundary`` imports no MLX, and the SHIPPED
reference this file compares against is not imported either: the two shipped
functions that define the behaviour (``_SidecarGather._rows_matrices`` and
``_stack_hot_rows`` in ``mtplx/models/qwen4_exp.py``, and
``_bind_fixed_m4_owned_row_prefetch`` in ``mtplx/qwen4_fixed_verify.py``) are
lifted out of the source with ``ast`` and executed in a NumPy-only namespace.

That matters for what the equivalence test is worth.  A hand-transcribed
reference proves the transcription; running the shipped source itself proves
the shipped behaviour, and it fails the moment either shipped function is
edited underneath this lane.
"""

from __future__ import annotations

import ast
import os
import textwrap
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from mtplx import ple_boundary
from mtplx import ple_row_gather

ROOT = Path(__file__).resolve().parents[1]
QWEN4_EXP = ROOT / "mtplx" / "models" / "qwen4_exp.py"
FIXED_VERIFY = ROOT / "mtplx" / "qwen4_fixed_verify.py"

FLAG = ple_boundary.ENV_FLAG
ITEMS_ENV = ple_boundary.ITEMS_ENV
PROBE_ENV = ple_boundary.PROBE_ENV

# Production PLE row geometry (config.json): ngram_size 3 x heads_per_ngram 8
# = 16 rows per token, ple_embed_dim 2560 / 16 = 160 per head, q4/g32.
ROW_SHAPES = {"weight": (20,), "scales": (5,), "biases": (5,)}
ROW_DTYPES = {"weight": np.uint32, "scales": np.uint16, "biases": np.uint16}
NAMES = ("weight", "scales", "biases")
#: One fixed-M4 decode window: 4 positions x 16 heads.
WINDOW_ROWS = 64


# --------------------------------------------------------------------------
# Lifting the shipped source (no MLX import)
# --------------------------------------------------------------------------
def _lift(path: Path, names, *, klass: str | None = None, namespace=None):
    """Compile the named top-level (or method) sources into a fresh namespace."""

    source = path.read_text()
    tree = ast.parse(source)
    scope = tree.body
    if klass is not None:
        scope = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == klass
        )
    found = {}
    for node in scope:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = textwrap.dedent(ast.get_source_segment(source, node))
    missing = [name for name in names if name not in found]
    assert not missing, f"{path.name} no longer defines {missing}"
    if namespace is None:
        namespace = {"np": np, "os": os}
    for name in names:
        exec(compile(ast.parse(found[name]), str(path), "exec"), namespace)
    return namespace


#: The shipped hot-row gather and its assembler, lifted once at import so the
#: stand-in below IS the shipped code rather than a transcription of it.
SHIPPED: dict = {"np": np, "os": os}
_lift(QWEN4_EXP, ["_stack_hot_rows"], namespace=SHIPPED)
_lift(QWEN4_EXP, ["_rows_matrices"], klass="_SidecarGather", namespace=SHIPPED)


@pytest.fixture(scope="module")
def shipped():
    """``_rows_matrices`` and ``_stack_hot_rows`` exactly as shipped."""

    return SHIPPED


@pytest.fixture(scope="module")
def shipped_prefetch():
    """``_bind_fixed_m4_owned_row_prefetch``, exactly as shipped.

    The W62 swap deliberately lives at the CALL SITE in
    ``_build_fixed_m4_compiled_verify_aux`` and not inside this binder, so
    this lift is the control arm's own code with nothing removed -- and this
    assertion is what keeps that true.
    """

    source = FIXED_VERIFY.read_text()
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_bind_fixed_m4_owned_row_prefetch"
    )
    text = ast.get_source_segment(source, node)
    assert "ple_boundary" not in text, (
        "the W62 swap moved INTO the shipped binder; the control arm's "
        "prefetch is no longer the shipped code"
    )
    namespace: dict = {"np": np, "os": os}
    exec(compile(text, str(FIXED_VERIFY), "exec"), namespace)
    return namespace["_bind_fixed_m4_owned_row_prefetch"]


def test_the_aux_builder_owns_the_primary_swap():
    """The swap is bound once, at construction, on the fixed-M4 aux path."""

    source = FIXED_VERIFY.read_text()
    assert "ple_boundary.bind_owned_row_prefetch(" in source
    assert "ple_boundary.bind_sidecar(" in source
    tree = ast.parse(source)
    builder = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_build_fixed_m4_compiled_verify_aux"
    )
    calls = {
        node.func.attr
        for node in ast.walk(builder)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"bind_sidecar", "bind_owned_row_prefetch"} <= calls


# --------------------------------------------------------------------------
# A sidecar stand-in at the production row geometry
# --------------------------------------------------------------------------
class FakeSidecar:
    """Every attribute the shipped hot-row branch and the W62 lane touch."""

    _HOT_PATH_MAX_ROWS = 4096
    #: The shipped method, so `bind_sidecar` captures the real original and a
    #: prefill-sized gather delegates to the code production would run.
    _rows_matrices = SHIPPED["_rows_matrices"]

    def __init__(self, tmp_path: Path, *, n_rows: int = 4096, hot_mb_rows: int = 4096,
                 pool: ThreadPoolExecutor | None = None, seed: int = 20260902):
        rng = np.random.default_rng(seed)
        self._path = tmp_path / "ngram-table.bin"
        self._maps = {}
        self._row_meta = []
        blobs = {}
        offset = 0
        with open(self._path, "wb") as handle:
            for name in NAMES:
                width = ROW_SHAPES[name][0]
                dtype = ROW_DTYPES[name]
                high = 2**32 if dtype is np.uint32 else 2**16
                data = rng.integers(0, high, size=(n_rows, width)).astype(dtype)
                blobs[name] = (offset, data)
                handle.write(data.tobytes())
                offset += data.nbytes
        for name in NAMES:
            start, data = blobs[name]
            memmap = np.memmap(
                self._path, mode="r", dtype=ROW_DTYPES[name],
                offset=start, shape=(n_rows, ROW_SHAPES[name][0]),
            )
            self._maps[name] = (memmap, "U32" if name == "weight" else "BF16")
            self._row_meta.append((start, ROW_SHAPES[name][0] * data.dtype.itemsize))
        self._fd = os.open(str(self._path), os.O_RDONLY)
        self._pool = pool
        self._hot: OrderedDict = OrderedDict()
        self._hot_cap_rows = hot_mb_rows
        self.hot_hits = 0
        self.hot_misses = 0
        self.prefetch_batches = 0
        self.lookahead_batches = 0
        self.vectorized_gathers = 0
        self.pread_gathers = 0
        self.warm_calls: list[list[int]] = []

    # `_warm`'s only externally visible effect is page-cache state, so the
    # stand-in records the call and does the same reads.
    def _warm(self, rows, *, counted: bool = True) -> None:
        self.warm_calls.append([int(r) for r in np.asarray(rows).reshape(-1)])
        for r in np.asarray(rows).reshape(-1):
            for base, row_bytes in self._row_meta:
                os.pread(self._fd, row_bytes, base + int(r) * row_bytes)

    def close(self):
        os.close(self._fd)


def _fresh_flags(monkeypatch, *, flag="1", items=None, probe=None):
    monkeypatch.setenv(FLAG, flag)
    if items is None:
        monkeypatch.delenv(ITEMS_ENV, raising=False)
    else:
        monkeypatch.setenv(ITEMS_ENV, items)
    if probe is None:
        monkeypatch.delenv(PROBE_ENV, raising=False)
    else:
        monkeypatch.setenv(PROBE_ENV, probe)
    for resolver in (
        ple_boundary.enabled,
        ple_boundary.items,
        ple_boundary.probe_rows,
        ple_boundary.graph_timing_enabled,
    ):
        resolver.cache_clear()
    ple_boundary.reset_receipt()


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    for resolver in (
        ple_boundary.enabled,
        ple_boundary.items,
        ple_boundary.probe_rows,
        ple_boundary.graph_timing_enabled,
    ):
        resolver.cache_clear()
    ple_boundary.reset_receipt()


# --------------------------------------------------------------------------
# Flag / item resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_flag_true_spellings(monkeypatch, value):
    _fresh_flags(monkeypatch, flag=value)
    assert ple_boundary.enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_flag_false_spellings(monkeypatch, value):
    _fresh_flags(monkeypatch, flag=value)
    assert ple_boundary.enabled() is False
    assert ple_boundary.items() == frozenset()


def test_flag_default_is_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(ITEMS_ENV, raising=False)
    for resolver in (ple_boundary.enabled, ple_boundary.items):
        resolver.cache_clear()
    assert ple_boundary.enabled() is False
    assert ple_boundary.items() == frozenset()
    assert ple_boundary.item("warm_skip") is False


def test_unparseable_flag_raises(monkeypatch):
    _fresh_flags(monkeypatch, flag="maybe")
    with pytest.raises(ValueError, match=FLAG):
        ple_boundary.enabled()


def test_flag_is_read_once(monkeypatch):
    _fresh_flags(monkeypatch, flag="1")
    assert ple_boundary.enabled() is True
    monkeypatch.setenv(FLAG, "0")
    assert ple_boundary.enabled() is True


def test_default_item_set_is_the_measured_levers_only(monkeypatch):
    """The instrument and the item that measured as noise are opt-in."""

    _fresh_flags(monkeypatch)
    assert ple_boundary.items() == frozenset(ple_boundary.DEFAULT_ITEMS)
    assert ple_boundary.DEFAULT_ITEMS == ("warm_skip", "primary_vectorized")
    assert "timing" not in ple_boundary.items()
    assert "hot_block" not in ple_boundary.items()
    assert ple_boundary.GRAPH_TIMING is False


def test_explicit_item_selection(monkeypatch):
    _fresh_flags(monkeypatch, items="warm_skip, timing")
    assert ple_boundary.items() == frozenset({"warm_skip", "timing"})
    assert ple_boundary.item("hot_block") is False
    assert ple_boundary.item("timing") is True


def test_unknown_item_raises(monkeypatch):
    _fresh_flags(monkeypatch, items="warm_skip,teleport")
    with pytest.raises(ValueError, match="teleport"):
        ple_boundary.items()


def test_unknown_item_query_raises(monkeypatch):
    _fresh_flags(monkeypatch)
    with pytest.raises(ValueError, match="teleport"):
        ple_boundary.item("teleport")


@pytest.mark.parametrize("value,expected", [("4", 4), ("64", 64)])
def test_probe_rows_parsing(monkeypatch, value, expected):
    _fresh_flags(monkeypatch, probe=value)
    assert ple_boundary.probe_rows() == expected


@pytest.mark.parametrize("value", ["0", "-3", "eight"])
def test_probe_rows_rejects_bad_values(monkeypatch, value):
    _fresh_flags(monkeypatch, probe=value)
    with pytest.raises(ValueError, match=PROBE_ENV):
        ple_boundary.probe_rows()


def test_probe_rows_default(monkeypatch):
    _fresh_flags(monkeypatch)
    assert ple_boundary.probe_rows() == ple_boundary.DEFAULT_PROBE_ROWS


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------
def test_bind_sidecar_is_inert_when_the_flag_is_off(monkeypatch, tmp_path):
    _fresh_flags(monkeypatch, flag="0")
    sidecar = FakeSidecar(tmp_path)
    try:
        assert ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"]) is None
        assert "_rows_matrices" not in vars(sidecar)
    finally:
        sidecar.close()


def test_bind_sidecar_arms_and_is_idempotent(monkeypatch, tmp_path):
    _fresh_flags(monkeypatch, items="warm_skip")
    sidecar = FakeSidecar(tmp_path)
    sidecar._rows_matrices_source = None
    try:
        line = ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"])
        assert line is not None and FLAG in line
        installed = vars(sidecar)["_rows_matrices"]
        assert ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"]) is None
        assert vars(sidecar)["_rows_matrices"] is installed
    finally:
        sidecar.close()


def test_bind_owned_row_prefetch_returns_the_shipped_objects_when_disarmed(
    monkeypatch, tmp_path
):
    _fresh_flags(monkeypatch, items="warm_skip")
    sidecar = FakeSidecar(tmp_path)
    try:
        submit = object()
        install = object()
        got_submit, got_install, line = ple_boundary.bind_owned_row_prefetch(
            sidecar, submit_primary=submit, install=install, names=NAMES
        )
        # Not "equivalent": the same objects.  The control arm's prefetch is
        # the code it was before this module existed.
        assert got_submit is submit and got_install is install and line is None
    finally:
        sidecar.close()


# --------------------------------------------------------------------------
# Equivalence: consulted rows and applied payloads
# --------------------------------------------------------------------------
def _decode_flat(rng, n_rows, *, window=4, heads=16, table_rows=4096):
    """A decode-shaped id vector: `window` positions x `heads` scattered rows."""

    del n_rows
    return rng.integers(0, table_rows, size=window * heads).astype(np.int64)


def _seed_hot(sidecar, rng, count, table_rows=4096):
    """Put `count` rows in the LRU the way an earlier cycle would have."""

    if count <= 0:
        return
    ids = np.unique(rng.integers(0, table_rows, size=count).astype(np.int64))
    fetched = {n: np.ascontiguousarray(sidecar._maps[n][0][ids]) for n in NAMES}
    for i, r in enumerate(ids.tolist()):
        sidecar._hot[int(r)] = tuple(fetched[n][i] for n in NAMES)


def _snapshot(sidecar, result):
    return (
        [(k, tuple(v.tobytes() for v in payload)) for k, payload in sidecar._hot.items()],
        sidecar.hot_hits,
        sidecar.hot_misses,
        {n: (result[n].dtype, result[n].shape, result[n].tobytes()) for n in NAMES},
    )


@pytest.mark.parametrize(
    "items",
    [
        "warm_skip",
        "hot_block",
        "warm_skip,hot_block",
        "warm_skip,hot_block,timing",
        "timing",
    ],
)
@pytest.mark.parametrize("resident", [True, False])
@pytest.mark.parametrize("preseed", [0, 8, 40])
def test_decode_gather_is_byte_identical_to_the_shipped_branch(
    monkeypatch, tmp_path, shipped, items, resident, preseed
):
    """The consulted rows and the applied payloads, over recorded traces."""

    rng = np.random.default_rng(4711 + preseed + (7 if resident else 0))
    traces = [_decode_flat(rng, None) for _ in range(12)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        control = FakeSidecar(tmp_path / "a", pool=pool)
        candidate = FakeSidecar(tmp_path / "b", pool=pool)
        try:
            seed_rng_a = np.random.default_rng(99)
            seed_rng_b = np.random.default_rng(99)
            _seed_hot(control, seed_rng_a, preseed)
            _seed_hot(candidate, seed_rng_b, preseed)

            _fresh_flags(monkeypatch, items=items)
            monkeypatch.setattr(
                ple_row_gather,
                "warm_decision",
                lambda maps, rows, sample=None: (
                    ("vectorized", 1.0) if resident else ("pread", 0.0)
                ),
            )
            assert ple_boundary.bind_sidecar(candidate, stack_hot_rows=SHIPPED["_stack_hot_rows"]) is not None
            shipped_rows = shipped["_rows_matrices"]

            for flat in traces:
                want = shipped_rows(control, flat.copy(), NAMES)
                got = candidate._rows_matrices(flat.copy(), NAMES)
                assert _snapshot(control, want) == _snapshot(candidate, got)
        finally:
            control.close()
            candidate.close()


@pytest.fixture(autouse=True)
def _make_sibling_dirs(tmp_path):
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)


def test_duplicate_ids_and_full_hits_stay_identical(monkeypatch, tmp_path, shipped):
    """The window repeats rows across positions; `inverse` must survive."""

    flat = np.array([5, 5, 7, 9, 7, 5, 11, 11] * 4, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=2) as pool:
        control = FakeSidecar(tmp_path / "a", pool=pool)
        candidate = FakeSidecar(tmp_path / "b", pool=pool)
        try:
            _fresh_flags(monkeypatch, items="warm_skip,hot_block")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, rows, sample=None: ("vectorized", 1.0),
            )
            ple_boundary.bind_sidecar(candidate, stack_hot_rows=SHIPPED["_stack_hot_rows"])
            for _ in range(3):  # second pass is an all-hit gather
                want = shipped["_rows_matrices"](control, flat.copy(), NAMES)
                got = candidate._rows_matrices(flat.copy(), NAMES)
                assert _snapshot(control, want) == _snapshot(candidate, got)
        finally:
            control.close()
            candidate.close()


def test_eviction_order_is_identical_under_a_tiny_lru(monkeypatch, tmp_path, shipped):
    rng = np.random.default_rng(31337)
    traces = [_decode_flat(rng, None) for _ in range(10)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        control = FakeSidecar(tmp_path / "a", pool=pool, hot_mb_rows=40)
        candidate = FakeSidecar(tmp_path / "b", pool=pool, hot_mb_rows=40)
        try:
            _fresh_flags(monkeypatch, items="warm_skip,hot_block")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, rows, sample=None: ("vectorized", 1.0),
            )
            ple_boundary.bind_sidecar(candidate, stack_hot_rows=SHIPPED["_stack_hot_rows"])
            for flat in traces:
                want = shipped["_rows_matrices"](control, flat.copy(), NAMES)
                got = candidate._rows_matrices(flat.copy(), NAMES)
                assert _snapshot(control, want) == _snapshot(candidate, got)
            assert len(candidate._hot) <= 40
        finally:
            control.close()
            candidate.close()


def test_a_prefill_sized_gather_delegates_to_the_shipped_method(
    monkeypatch, tmp_path
):
    """Above `_HOT_PATH_MAX_ROWS` the shipped branch owns the decision."""

    _fresh_flags(monkeypatch, items="warm_skip,hot_block")
    sidecar = FakeSidecar(tmp_path, n_rows=8192)
    try:
        seen = {}

        def spy(self, flat, names):
            seen["rows"] = int(len(np.unique(flat)))
            return {n: np.zeros((len(flat), 1), np.uint8) for n in names}

        flat = np.arange(5000, dtype=np.int64)
        out = ple_boundary._boundary_rows_matrices(
            sidecar, spy, flat, NAMES, skip_warm=True, block=True, timing=False,
        )
        assert seen["rows"] == 5000
        assert set(out) == set(NAMES)
        # Nothing entered the decode branch, so nothing was counted.
        assert ple_boundary.last_receipt()["gathers"] == 0
    finally:
        sidecar.close()


def test_zero_capacity_lru_delegates(monkeypatch, tmp_path):
    _fresh_flags(monkeypatch, items="warm_skip")
    sidecar = FakeSidecar(tmp_path, hot_mb_rows=0)
    try:
        called = {}

        def spy(self, flat, names):
            called["yes"] = True
            return {n: np.zeros((1, 1), np.uint8) for n in names}

        out = ple_boundary._boundary_rows_matrices(
            sidecar, spy, np.array([1, 2, 3], np.int64), NAMES,
            skip_warm=True, block=True, timing=False,
        )
        assert called == {"yes": True} and set(out) == set(NAMES)
    finally:
        sidecar.close()


# --------------------------------------------------------------------------
# warm_skip: what it actually removes
# --------------------------------------------------------------------------
def test_warm_skip_removes_the_pread_pass_when_the_pages_are_resident(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(5)
    flat = _decode_flat(rng, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sidecar = FakeSidecar(tmp_path, pool=pool)
        try:
            _fresh_flags(monkeypatch, items="warm_skip")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, rows, sample=None: ("vectorized", 1.0),
            )
            ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"])
            sidecar._rows_matrices(flat, NAMES)
            assert sidecar.warm_calls == []
            receipt = ple_boundary.last_receipt()
            assert receipt["warm_skipped"] == 1 and receipt["warm_taken"] == 0
            assert receipt["probes"] == 1
        finally:
            sidecar.close()


def test_warm_skip_falls_back_to_the_shipped_pass_when_cold(monkeypatch, tmp_path):
    rng = np.random.default_rng(6)
    flat = _decode_flat(rng, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sidecar = FakeSidecar(tmp_path, pool=pool)
        try:
            _fresh_flags(monkeypatch, items="warm_skip")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, rows, sample=None: ("pread", 0.0),
            )
            ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"])
            sidecar._rows_matrices(flat, NAMES)
            assert len(sidecar.warm_calls) == 1
            assert sorted(sidecar.warm_calls[0]) == sorted(
                int(r) for r in np.unique(flat)
            )
            receipt = ple_boundary.last_receipt()
            assert receipt["warm_taken"] == 1 and receipt["warm_skipped"] == 0
        finally:
            sidecar.close()


def test_an_unprobeable_table_takes_the_shipped_pass(monkeypatch, tmp_path):
    """`warm_decision` answers "pread" when mincore is unavailable."""

    rng = np.random.default_rng(7)
    flat = _decode_flat(rng, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sidecar = FakeSidecar(tmp_path, pool=pool)
        try:
            _fresh_flags(monkeypatch, items="warm_skip")
            monkeypatch.setattr(ple_row_gather._Libc, "get",
                                classmethod(lambda cls: None))
            ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"])
            sidecar._rows_matrices(flat, NAMES)
            assert len(sidecar.warm_calls) == 1
        finally:
            sidecar.close()


def test_without_a_pool_no_warm_pass_and_no_probe(monkeypatch, tmp_path):
    """MTPLX_NGRAM_PREFETCH=0: the shipped branch never warms, so neither do we."""

    rng = np.random.default_rng(8)
    flat = _decode_flat(rng, None)
    sidecar = FakeSidecar(tmp_path, pool=None)
    try:
        _fresh_flags(monkeypatch, items="warm_skip")
        ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"])
        sidecar._rows_matrices(flat, NAMES)
        assert sidecar.warm_calls == []
        assert ple_boundary.last_receipt()["probes"] == 0
    finally:
        sidecar.close()


# --------------------------------------------------------------------------
# primary_vectorized
# --------------------------------------------------------------------------
def _run_primary(sidecar, submit, install, rows):
    install(submit(rows))


@pytest.mark.parametrize("rows", [
    [3, 17, 3, 900, 41, 41, 41, 12, 5, 6, 7, 8, 9, 10, 11, 12],
    list(range(16)),
    [7] * 16,
])
def test_primary_vectorized_leaves_the_same_lru(
    monkeypatch, tmp_path, shipped_prefetch, rows
):
    """Same payload bytes, same insertion order, same eviction."""

    with ThreadPoolExecutor(max_workers=4) as pool:
        control = FakeSidecar(tmp_path / "a", pool=pool)
        candidate = FakeSidecar(tmp_path / "b", pool=pool)
        control._hot_cap_rows = 20
        candidate._hot_cap_rows = 20
        try:
            _fresh_flags(monkeypatch, items="primary_vectorized")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, ids, sample=None: ("vectorized", 1.0),
            )
            c_submit, _c_missing, c_install = shipped_prefetch(control)
            k_submit, _k_missing, k_install = shipped_prefetch(candidate)
            k_submit, k_install, line = ple_boundary.bind_owned_row_prefetch(
                candidate, submit_primary=k_submit, install=k_install,
                names=NAMES,
            )
            assert line is not None
            ids = np.asarray(rows, dtype=np.int64)
            # Two cycles, so the second exercises overwrite + move_to_end.
            for _ in range(2):
                _run_primary(control, c_submit, c_install, ids)
                _run_primary(candidate, k_submit, k_install, ids)
            assert [
                (k, tuple(v.tobytes() for v in p)) for k, p in control._hot.items()
            ] == [
                (k, tuple(v.tobytes() for v in p)) for k, p in candidate._hot.items()
            ]
            receipt = ple_boundary.last_receipt()
            assert receipt["primary_inline"] == 2 and receipt["primary_pooled"] == 0
        finally:
            control.close()
            candidate.close()


def test_primary_vectorized_payloads_are_owned_copies(monkeypatch, tmp_path):
    """A memmap VIEW in the LRU would defer the read to use time."""

    with ThreadPoolExecutor(max_workers=2) as pool:
        sidecar = FakeSidecar(tmp_path, pool=pool)
        try:
            _fresh_flags(monkeypatch, items="primary_vectorized")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, ids, sample=None: ("vectorized", 1.0),
            )
            submit, install, _line = ple_boundary.bind_owned_row_prefetch(
                sidecar,
                submit_primary=lambda rows: (),
                install=lambda pending: None,
                names=NAMES,
            )
            install(submit(np.array([1, 2, 3], dtype=np.int64)))
            for payload in sidecar._hot.values():
                for value in payload:
                    assert not isinstance(value.base, np.memmap)
                    assert not isinstance(value, np.memmap)
        finally:
            sidecar.close()


def test_primary_vectorized_falls_back_to_the_pool_when_cold(monkeypatch, tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        sidecar = FakeSidecar(tmp_path, pool=pool)
        try:
            _fresh_flags(monkeypatch, items="primary_vectorized")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, ids, sample=None: ("pread", 0.0),
            )
            seen = {}

            def shipped_submit(rows):
                seen["rows"] = list(np.asarray(rows).reshape(-1))
                return ("sentinel",)

            submit, _install, _line = ple_boundary.bind_owned_row_prefetch(
                sidecar, submit_primary=shipped_submit,
                install=lambda pending: None, names=NAMES,
            )
            assert submit(np.array([4, 5], dtype=np.int64)) == ("sentinel",)
            assert seen["rows"] == [4, 5]
            assert sidecar._hot == OrderedDict()
            assert ple_boundary.last_receipt()["primary_pooled"] == 1
        finally:
            sidecar.close()


def test_primary_vectorized_empty_row_set_delegates(monkeypatch, tmp_path):
    sidecar = FakeSidecar(tmp_path)
    try:
        _fresh_flags(monkeypatch, items="primary_vectorized")
        submit, _install, _line = ple_boundary.bind_owned_row_prefetch(
            sidecar, submit_primary=lambda rows: ("empty",),
            install=lambda pending: None, names=NAMES,
        )
        assert submit(np.array([], dtype=np.int64)) == ("empty",)
    finally:
        sidecar.close()


# --------------------------------------------------------------------------
# The lane cannot change WHICH rows are consulted
# --------------------------------------------------------------------------
def test_the_module_never_touches_the_row_arithmetic():
    """`_ngram_rows_np` decides which rows exist; this lane must not name it."""

    source = (ROOT / "mtplx" / "ple_boundary.py").read_text()
    tree = ast.parse(source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in ("_ngram_rows_np", "_rows_np", "candidate_rows", "resolve"):
        assert forbidden not in names, f"ple_boundary references {forbidden}"


def test_receipt_reports_the_armed_items(monkeypatch):
    _fresh_flags(monkeypatch, items="warm_skip,timing", probe="16")
    receipt = ple_boundary.last_receipt()
    assert receipt["items"] == "timing,warm_skip"
    assert receipt["probe_rows"] == 16
    assert receipt["gathers"] == 0


def test_graph_build_note_is_recorded(monkeypatch):
    _fresh_flags(monkeypatch, items="timing")
    ple_boundary.note_graph_build(0.0016408)
    receipt = ple_boundary.last_receipt()
    assert receipt["graph_build_calls"] == 1
    assert receipt["graph_build_ms"] == pytest.approx(1.6408, rel=1e-6)


def test_timing_item_populates_the_phase_split(monkeypatch, tmp_path):
    rng = np.random.default_rng(11)
    flat = _decode_flat(rng, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sidecar = FakeSidecar(tmp_path, pool=pool)
        try:
            _fresh_flags(monkeypatch, items="warm_skip,hot_block,timing")
            monkeypatch.setattr(
                ple_row_gather, "warm_decision",
                lambda maps, rows, sample=None: ("vectorized", 1.0),
            )
            ple_boundary.bind_sidecar(sidecar, stack_hot_rows=SHIPPED["_stack_hot_rows"])
            sidecar._rows_matrices(flat, NAMES)
            receipt = ple_boundary.last_receipt()
            assert receipt["gather_ms"] > 0.0
            assert receipt["read_ms"] > 0.0
            assert receipt["assemble_ms"] > 0.0
            assert receipt["warm_ms"] == 0.0  # skipped, so never entered
        finally:
            sidecar.close()


def test_composes_with_the_candidate_prefetch_flag(monkeypatch):
    """W56 and W62 must be armable in the same process."""

    from mtplx import ple_candidate_prefetch

    monkeypatch.setenv(ple_candidate_prefetch.ENV_FLAG, "1")
    ple_candidate_prefetch.enabled.cache_clear()
    _fresh_flags(monkeypatch)
    try:
        assert ple_candidate_prefetch.enabled() is True
        assert ple_boundary.enabled() is True
    finally:
        ple_candidate_prefetch.enabled.cache_clear()


def test_flags_are_registered_for_the_runtime():
    from mtplx import profiles

    for key in (FLAG, ITEMS_ENV, PROBE_ENV):
        assert key in profiles.MODEL_RUNTIME_ENV_OVERRIDE_KEYS
