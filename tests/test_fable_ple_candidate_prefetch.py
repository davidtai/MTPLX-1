"""MTPLX_FABLE_PLE_CANDIDATE_PREFETCH (K-P1): exactness, fallback, counters.

Pure Python + NumPy.  ``mtplx.ple_candidate_prefetch`` imports no MLX; the row
arithmetic under test is the SHIPPED ``_ngram_rows_np``, compiled out of
``mtplx/models/qwen4_exp.py`` by AST (the module itself imports MLX) exactly as
the prefill-lookahead suite does, and the gather runs against a real on-disk
memmap built in ``tmp_path`` at the production row geometry (80/10/10 bytes per
row).  No GPU, no MLX array, no model load.
"""

from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from mtplx import ple_candidate_prefetch as lane
from mtplx import ple_row_gather as row_gather
from mtplx.ple_candidate_prefetch import CandidateRowPrefetch, candidate_rows


ROOT = Path(__file__).resolve().parents[1]
MODEL_TEXT = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
VERIFY_TEXT = (ROOT / "mtplx" / "qwen4_fixed_verify.py").read_text("utf-8")
GRAPHBANK_TEXT = (ROOT / "mtplx" / "graphbank.py").read_text("utf-8")
GENERATION_TEXT = (ROOT / "mtplx" / "generation.py").read_text("utf-8")
DRIVER_TEXT = (ROOT / "scripts" / "fable" / "abba_driver.py").read_text("utf-8")
PROFILES_TEXT = (ROOT / "mtplx" / "profiles.py").read_text("utf-8")
README_TEXT = (ROOT / "scripts" / "fable" / "README.md").read_text("utf-8")

#: Production row geometry: 16 ngram heads/token, head_dim 160, q4/g32.
_MAPS = {"weight": (np.uint32, 20), "scales": (np.uint16, 5), "biases": (np.uint16, 5)}
_TABLE_ROWS = 8192
_EOS = 248_044


# ---------------------------------------------------------------------------
# The shipped row arithmetic, without importing MLX
# ---------------------------------------------------------------------------


def _bind_ngram_rows_np():
    node = next(
        n
        for n in ast.parse(MODEL_TEXT).body
        if isinstance(n, ast.FunctionDef) and n.name == "_ngram_rows_np"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"np": np}
    exec(compile(module, "<qwen4_exp>", "exec"), namespace)
    return namespace["_ngram_rows_np"]


def _rows_fn(ngram_heads: int = 16, vocab: int = 200_000, table_rows=_TABLE_ROWS):
    """The bound partial the fixed-M4 aux hands the lane.

    Head vocabularies are sized to the synthetic table so every row id the
    hash produces addresses a real row.
    """

    per_head = table_rows // ngram_heads
    return partial(
        _bind_ngram_rows_np(),
        mult=np.array([2_654_435_761, 40_503, 1_337], dtype=np.int64),
        sizes=np.full(ngram_heads, per_head, dtype=np.int64),
        offs=np.arange(ngram_heads, dtype=np.int64) * per_head,
        eos=_EOS,
        ngram_size=3,
        heads_per_ngram=ngram_heads // 2,
    )


def _window_rows(rows, previous, window_tokens):
    """What the aux itself computes for the whole 4-token verify window."""

    ids_np = np.asarray((window_tokens,), dtype=np.int64)
    prev_np = np.asarray((previous,), dtype=np.int64)
    resolved, _history = rows(ids_np, prev_np)
    return resolved


# ---------------------------------------------------------------------------
# (1) the prediction is EXACT: a candidate's rows are the window's rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", (0, 1, 2, 3))
def test_candidate_rows_equal_the_window_rows_for_every_candidate(position):
    rows = _rows_fn()
    previous = (11_001, 12_002)
    fixed = (31_003, 41_004, 51_005)[:position]
    rng = np.random.default_rng(20260902)
    candidates = rng.integers(0, 190_000, size=20, dtype=np.int64)

    predicted = candidate_rows(rows, previous, fixed, candidates)
    assert predicted.shape == (20, 16)

    for index, sampled in enumerate(candidates.tolist()):
        window = list(fixed) + [int(sampled)] + [7_777] * (3 - position)
        actual = _window_rows(rows, previous, window)[0, position, :]
        assert predicted[index].tolist() == actual.tolist(), sampled


def test_candidate_rows_are_exact_with_an_eos_inside_the_history():
    """The EOS segment scan is why the prediction keeps the FULL prefix.

    ``_ngram_rows_np`` masks a shift to EOS while ``pos_in_seg < s``, and
    ``pos_in_seg`` is measured from the last EOS at or before the position.
    Predicting from a truncated history would move that boundary and produce
    different rows; predicting from the same 2-token history plus the same
    fixed prefix cannot.
    """

    rows = _rows_fn()
    for previous in ((_EOS, 12_002), (11_001, _EOS), (_EOS, _EOS)):
        for fixed in ((), (_EOS,), (31_003, _EOS), (_EOS, 41_004)):
            candidates = np.array([5, 61_006, _EOS], dtype=np.int64)
            predicted = candidate_rows(rows, previous, fixed, candidates)
            for index, sampled in enumerate(candidates.tolist()):
                window = list(fixed) + [int(sampled)] + [7_777] * (
                    3 - len(fixed)
                )
                actual = _window_rows(rows, previous, window)[0, len(fixed), :]
                assert predicted[index].tolist() == actual.tolist()


# ---------------------------------------------------------------------------
# A synthetic table and a sidecar stub with the three attributes the lane uses
# ---------------------------------------------------------------------------


def _build_sidecar(tmp_path: Path, pool):
    """One file, three regions -- the real safetensors layout the lane preads.

    Returns ``(sidecar_stub, plain_memmaps)``: the lane reads the stub, the
    tests compare against the plain maps through ``ple_row_gather``.
    """

    path = tmp_path / "ngram.bin"
    rng = np.random.default_rng(1788)
    blobs = {}
    for name, (dtype, cols) in _MAPS.items():
        blobs[name] = rng.integers(
            0, np.iinfo(dtype).max, size=(_TABLE_ROWS, cols), dtype=dtype
        )
    offsets: dict[str, int] = {}
    with open(path, "wb") as handle:
        for name, payload in blobs.items():
            offsets[name] = handle.tell()
            handle.write(payload.tobytes())
    maps = {
        name: np.memmap(
            path,
            mode="r",
            dtype=_MAPS[name][0],
            offset=offsets[name],
            shape=(_TABLE_ROWS, _MAPS[name][1]),
        )
        for name in _MAPS
    }

    class _Sidecar:
        pass

    # Fault the whole synthetic table into core through the SAME mappings
    # mincore will be asked about.  Without this the probe's answer depends on
    # writeback timing and a warm-path test flakes into a cold decline -- the
    # lane's two regimes are exactly what these tests have to hold apart.
    for memmap in maps.values():
        int(np.asarray(memmap).view(np.uint8).sum(dtype=np.int64))

    sidecar = _Sidecar()
    sidecar._pool = pool
    sidecar._fd = os.open(str(path), os.O_RDONLY)
    sidecar._maps = {
        name: (maps[name], "U32" if name == "weight" else "BF16") for name in _MAPS
    }
    return sidecar, maps


def _shipped_gather(maps, flat):
    """Exactly the expression ``_SidecarGather._rows_matrices`` runs."""

    uniq, inverse = np.unique(np.asarray(flat, dtype=np.int64), return_inverse=True)
    return row_gather.gather_matrices(maps, uniq, inverse, tuple(_MAPS))


def _drive_cycle(prefetch, rows, previous, primary, drafts, supports):
    """One decode cycle's worth of lane traffic, in the shipped order."""

    prefetch.begin_cycle()
    prefetch.submit(
        prefix_tokens=(),
        candidate_ids=(primary,),
        completion_tokens=(0, 0, primary),
        committed_count=2,
    )
    prefix = [primary]
    for token, support in zip(drafts, supports):
        prefetch.submit(
            prefix_tokens=tuple(prefix),
            candidate_ids=support,
            completion_tokens=(0, 0, primary),
            committed_count=2,
        )
        prefix.append(token)
    window = [primary, *drafts]
    flat = _window_rows(rows, previous, window).reshape(-1)
    return flat


@pytest.fixture()
def pool():
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="test-ple")
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# (2) the consumer is byte-identical to the shipped gather
# ---------------------------------------------------------------------------


def test_resolved_rows_are_byte_identical_to_the_shipped_gather(tmp_path, pool):
    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    previous = (0, 0)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=previous)

    rng = np.random.default_rng(4242)
    supports = [rng.integers(0, 190_000, size=20, dtype=np.int64) for _ in range(3)]
    drafts = [int(support[rng.integers(0, 20)]) for support in supports]
    primary = 90_210

    flat = _drive_cycle(prefetch, rows, previous, primary, drafts, supports)
    got = prefetch.resolve(flat)
    want = _shipped_gather(maps, flat)

    assert got is not None
    for name in _MAPS:
        assert got[name].dtype == want[name].dtype
        assert got[name].shape == want[name].shape == (64, _MAPS[name][1])
        assert got[name].tobytes() == want[name].tobytes(), name
    assert lane.last_receipt()["hits"] == 1
    assert lane.last_receipt()["misses"] == 0
    os.close(sidecar._fd)


def test_cold_probe_declines_a_candidate_bucket_instead_of_preading_it(
    tmp_path, pool, monkeypatch
):
    """A cold 320-row bucket must fall back, not burn 960 GIL-held syscalls.

    Three maps x 320 rows at ~5 us of GIL-contended Python per ``os.pread``
    is ~4.8 ms against a ~50 us gather.  The decline has to REMOVE the
    bucket's index entries too, or resolve would hand back the buffer's
    uninitialised slice as if it were table rows.
    """

    lane.reset_receipt()
    monkeypatch.setattr(row_gather._Libc, "get", classmethod(lambda cls: None))
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    previous = (0, 0)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=previous)

    rng = np.random.default_rng(77)
    supports = [rng.integers(0, 190_000, size=20, dtype=np.int64) for _ in range(3)]
    drafts = [int(support[0]) for support in supports]
    flat = _drive_cycle(prefetch, rows, previous, 4_321, drafts, supports)

    assert prefetch.resolve(flat) is None
    receipt = lane.last_receipt()
    assert receipt["cold_declines"] == 3  # the three 320-row candidate buckets
    assert receipt["pread_buckets"] == 1  # the 16-row primary, within budget
    assert receipt["vectorized_buckets"] == 0
    assert receipt["misses"] == 1
    os.close(sidecar._fd)


def test_cold_primary_bucket_is_preaded_and_is_byte_identical(
    tmp_path, pool, monkeypatch
):
    """Within the pread budget the cold path still runs -- and reads right."""

    lane.reset_receipt()
    monkeypatch.setattr(row_gather._Libc, "get", classmethod(lambda cls: None))
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=(0, 0))
    prefetch.begin_cycle()
    prefetch.submit(
        prefix_tokens=(),
        candidate_ids=(4_321,),
        completion_tokens=(0, 0, 4_321),
        committed_count=2,
    )
    flat = _window_rows(rows, (0, 0), [4_321])[0, 0, :]
    got = prefetch.resolve(flat)
    want = _shipped_gather(maps, flat)
    assert got is not None
    for name in _MAPS:
        assert got[name].tobytes() == want[name].tobytes(), name
    assert lane.last_receipt()["pread_buckets"] == 1
    os.close(sidecar._fd)


def test_warm_probe_takes_the_vectorized_read(tmp_path, pool):
    """The lane's own formulation, independent of the W46 prefill flag."""

    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    previous = (0, 0)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=previous)
    rng = np.random.default_rng(1010)
    supports = [rng.integers(0, 190_000, size=20, dtype=np.int64) for _ in range(3)]
    drafts = [int(support[3]) for support in supports]
    flat = _drive_cycle(prefetch, rows, previous, 2_020, drafts, supports)

    got = prefetch.resolve(flat)
    want = _shipped_gather(maps, flat)
    assert got is not None
    for name in _MAPS:
        assert got[name].tobytes() == want[name].tobytes(), name
    receipt = lane.last_receipt()
    assert receipt["vectorized_buckets"] == 4
    assert receipt["cold_declines"] == 0
    assert receipt["pread_buckets"] == 0
    os.close(sidecar._fd)


def test_every_sampled_candidate_resolves_not_just_the_one_we_drew(tmp_path, pool):
    """The lane must cover the WHOLE support, or the win is a coin flip."""

    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    previous = (0, 0)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=previous)
    rng = np.random.default_rng(5)
    support = rng.integers(0, 190_000, size=20, dtype=np.int64)
    primary = 1_234

    for sampled in support.tolist():
        prefetch.begin_cycle()
        prefetch.submit(
            prefix_tokens=(),
            candidate_ids=(primary,),
            completion_tokens=(0, 0, primary),
            committed_count=2,
        )
        prefetch.submit(
            prefix_tokens=(primary,),
            candidate_ids=support,
            completion_tokens=(0, 0, primary),
            committed_count=2,
        )
        # Only the first two window positions are covered by this cycle's
        # submits, so resolve the two-token prefix's rows.
        flat = _window_rows(rows, previous, [primary, int(sampled)])[
            0, :2, :
        ].reshape(-1)
        got = prefetch.resolve(flat)
        assert got is not None, sampled
        want = _shipped_gather(maps, flat)
        for name in _MAPS:
            assert got[name].tobytes() == want[name].tobytes()
    assert lane.last_receipt()["hits"] == 20
    os.close(sidecar._fd)


# ---------------------------------------------------------------------------
# (3) the miss path
# ---------------------------------------------------------------------------


def test_uncovered_window_falls_back_whole(tmp_path, pool):
    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, _maps = _build_sidecar(tmp_path, pool)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=(0, 0))
    prefetch.begin_cycle()
    prefetch.submit(
        prefix_tokens=(),
        candidate_ids=(11,),
        completion_tokens=(0, 0, 11),
        committed_count=2,
    )
    covered = _window_rows(rows, (0, 0), [11, 0, 0, 0])[0, 0, :]
    # One row the lane never read is enough to send the whole window back.
    flat = np.concatenate([covered, np.array([_TABLE_ROWS - 1], dtype=np.int64)])
    assert prefetch.resolve(flat) is None
    receipt = lane.last_receipt()
    assert receipt["misses"] == 1
    assert receipt["hits"] == 0
    assert receipt["rows_missing"] == 1
    assert receipt["rows_served"] == 16
    os.close(sidecar._fd)


def test_begin_cycle_releases_the_previous_cycle(tmp_path, pool):
    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, _maps = _build_sidecar(tmp_path, pool)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=(0, 0))
    prefetch.begin_cycle()
    prefetch.submit(
        prefix_tokens=(),
        candidate_ids=(11,),
        completion_tokens=(0, 0, 11),
        committed_count=2,
    )
    flat = _window_rows(rows, (0, 0), [11])[0, 0, :]
    assert prefetch.resolve(flat) is not None
    prefetch.begin_cycle()
    assert prefetch.resolve(flat) is None
    os.close(sidecar._fd)


def test_more_buckets_than_the_window_are_declined(tmp_path, pool):
    """A cycle cannot reserve more slices than the buffer holds."""

    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, _maps = _build_sidecar(tmp_path, pool)
    prefetch = CandidateRowPrefetch(
        rows=rows, sidecar=sidecar, prompt_tail=(0, 0), window=2, top_k=20
    )
    prefetch.begin_cycle()
    reserved = [
        prefetch.submit(
            prefix_tokens=(),
            candidate_ids=(index + 1,),
            completion_tokens=(0, 0, index + 1),
            committed_count=2,
        )
        for index in range(4)
    ]
    assert reserved[:2] == [320, 320]
    assert reserved[2:] == [0, 0]
    os.close(sidecar._fd)


def test_an_over_wide_bucket_publishes_nothing(tmp_path, pool):
    """A bucket wider than its slice must not write past it, or index it."""

    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    # top_k=1 reserves 16 slots per bucket; 20 candidates need up to 320.
    prefetch = CandidateRowPrefetch(
        rows=rows, sidecar=sidecar, prompt_tail=(0, 0), window=4, top_k=1
    )
    prefetch.begin_cycle()
    rng = np.random.default_rng(9)
    support = rng.integers(0, 190_000, size=20, dtype=np.int64)
    prefetch.submit(
        prefix_tokens=(),
        candidate_ids=support,
        completion_tokens=(0, 0, 1),
        committed_count=2,
    )
    flat = _window_rows(rows, (0, 0), [int(support[0])])[0, 0, :]
    assert prefetch.resolve(flat) is None
    assert lane.last_receipt()["cold_declines"] == 1
    os.close(sidecar._fd)


def test_a_resolve_closes_the_cycle_without_an_explicit_begin(tmp_path, pool):
    """Not every fixed-M4 route calls prefetch_primary, so submit self-resets."""

    lane.reset_receipt()
    rows = _rows_fn()
    sidecar, maps = _build_sidecar(tmp_path, pool)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=(0, 0))
    for cycle in range(12):  # far more than the 4-bucket capacity
        token = 5_000 + cycle
        prefetch.submit(
            prefix_tokens=(),
            candidate_ids=(token,),
            completion_tokens=(0, 0, token),
            committed_count=2,
        )
        flat = _window_rows(rows, (0, 0), [token])[0, 0, :]
        got = prefetch.resolve(flat)
        assert got is not None, cycle
        want = _shipped_gather(maps, flat)
        for name in _MAPS:
            assert got[name].tobytes() == want[name].tobytes()
    assert lane.last_receipt()["hits"] == 12
    os.close(sidecar._fd)


def test_submit_does_no_row_arithmetic_on_the_owner_thread():
    """The lane's whole point: ~60-80 us of host work has to leave the cycle.

    Checked on the parsed AST of the class, not on its prose: the docstrings
    below name the very calls this forbids.
    """

    tree = ast.parse((ROOT / "mtplx" / "ple_candidate_prefetch.py").read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "CandidateRowPrefetch"
    )

    def _called_names(method: str) -> set[str]:
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == method
        )
        names: set[str] = set()
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            func = call.func
            names.add(
                func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            )
        return names

    submit = _called_names("submit")
    assert "candidate_rows" not in submit
    assert "unique" not in submit
    fill = _called_names("_fill")
    assert "candidate_rows" in fill
    assert "unique" in fill


# ---------------------------------------------------------------------------
# (4) the receipt
# ---------------------------------------------------------------------------


def test_receipt_carries_the_named_fields(tmp_path, pool):
    lane.reset_receipt()
    assert set(lane.last_receipt()) >= {
        "depths",
        "candidate_rows",
        "hits",
        "misses",
        "bytes",
        "worker_wait_ms",
    }
    rows = _rows_fn()
    sidecar, _maps = _build_sidecar(tmp_path, pool)
    prefetch = CandidateRowPrefetch(rows=rows, sidecar=sidecar, prompt_tail=(0, 0))
    rng = np.random.default_rng(31)
    supports = [rng.integers(0, 190_000, size=20, dtype=np.int64) for _ in range(3)]
    drafts = [int(support[1]) for support in supports]
    flat = _drive_cycle(prefetch, rows, (0, 0), 9_090, drafts, supports)
    prefetch.resolve(flat)

    receipt = lane.last_receipt()
    assert receipt["depths"] == 4  # primary + three draft depths
    assert receipt["candidate_rows"] > 0
    # 80 + 10 + 10 bytes per row across the three maps.
    assert receipt["bytes"] == receipt["candidate_rows"] * 100
    assert receipt["worker_wait_ms"] >= 0.0
    assert receipt["hits"] + receipt["misses"] == 1
    os.close(sidecar._fd)


def test_driver_records_the_lane_next_to_the_other_ple_receipts():
    assert '"ple_candidate_prefetch": _ple_candidate_prefetch_receipt()' in DRIVER_TEXT
    assert '"ple_candidate_prefetch_armed": _ple_candidate_prefetch_armed()' in (
        DRIVER_TEXT
    )
    assert "from mtplx.ple_candidate_prefetch import last_receipt" in DRIVER_TEXT
    assert "from mtplx.ple_candidate_prefetch import ENV_FLAG" in DRIVER_TEXT


# ---------------------------------------------------------------------------
# (5) the flag, and flag-off == stock
# ---------------------------------------------------------------------------


def test_flag_defaults_off_and_rejects_junk(monkeypatch):
    for value, expected in (("", False), ("0", False), ("1", True), ("on", True)):
        lane.enabled.cache_clear()
        monkeypatch.setenv(lane.ENV_FLAG, value)
        assert lane.enabled() is expected
    lane.enabled.cache_clear()
    monkeypatch.setenv(lane.ENV_FLAG, "maybe")
    with pytest.raises(ValueError):
        lane.enabled()
    lane.enabled.cache_clear()
    monkeypatch.delenv(lane.ENV_FLAG, raising=False)
    assert lane.enabled() is False
    lane.enabled.cache_clear()


def test_flag_off_leaves_the_retained_aux_byte_for_byte_unchanged():
    """The control arm's hot path must not be edited at all.

    The lane lives in its OWN aux class, so `_FixedM4SidecarAux.__call__` is
    exactly the expression the monomorphism guard in
    ``tests/test_qwen4_fixed_host_tokens_static.py`` already pins.  An A/B
    whose control arm's hot path has been touched is not a control.
    """

    shipped = VERIFY_TEXT.split("class _FixedM4SidecarAux", 1)[1].split(
        "\nclass ", 1
    )[0]
    assert "self._gather(rows.reshape(-1)).reshape(" in shipped
    assert "candidate" not in shipped
    assert "prepared" not in shipped


def test_candidate_aux_serves_from_the_buffer_and_falls_back_whole():
    lane_aux = VERIFY_TEXT.split("class _FixedM4CandidateSidecarAux", 1)[1].split(
        "\nclass ", 1
    )[0]
    assert "prepared = self._candidates.resolve(flat)" in lane_aux
    assert "if prepared is None:" in lane_aux
    assert "return self._gather(flat).reshape(1, 4, self._output_dim)" in lane_aux
    assert "self._gather(flat, prepared=prepared)" in lane_aux
    assert "self._candidates.begin_cycle()" in lane_aux
    # The shipped hot-row warm still runs, so the LRU fallback keeps its
    # coverage whether or not the buffer hits.
    assert "self._pending_warm = self._submit_warm(rows.reshape(-1))" in lane_aux


def test_lane_is_refused_on_the_experimental_aux_routes():
    """Arming against a raw/window aux would measure the control, labelled B."""

    assert "requires the retained " in VERIFY_TEXT
    builder = VERIFY_TEXT.split("def _build_fixed_m4_compiled_verify_aux", 1)[1]
    assert builder.count("ple_candidate_prefetch.enabled()") == 2
    assert "return _FixedM4CandidateSidecarAux(" in builder
    assert "return _FixedM4SidecarAux(" in builder


def test_graphbank_binds_the_hook_once_at_construction():
    assert "_fixed_m4_no_candidate_prefetch" in GRAPHBANK_TEXT
    assert '"submit_candidates_aux": submit_candidates_aux,' in GRAPHBANK_TEXT
    assert "def submit_fixed_m4_candidates(" in GRAPHBANK_TEXT


def test_draft_loop_submits_before_the_depth_is_appended():
    """The submit must precede the NEXT depth's forward, or it hides nothing."""

    body = GENERATION_TEXT.split("if _ple_candidate_submit is not None and draft_token", 1)
    assert len(body) == 2, "stock draft-loop hook missing"
    tail = body[1]
    submit_at = tail.index("_ple_candidate_submit(")
    append_at = tail.index("draft_tokens.append(draft_token)")
    assert submit_at < append_at
    assert "draft_q.token_ids" in tail[:append_at]
    assert "prefix_tokens=(int(primary), *draft_tokens)" in tail[:append_at]
    # Depth 1 also has to seed the primary's own position on a lane where
    # prefetch_fixed_m4_primary never runs, or every window misses.
    assert "if not draft_tokens:" in tail[:append_at]


def test_joint_d3_core_hook_uses_the_resolved_tokens():
    core = GENERATION_TEXT.split("core_tokens = _pr391_decode_float32_d3_tokens", 1)[1]
    head = core[: core.index("elapsed_draft = time.perf_counter()")]
    assert "_ple_candidate_submit(" in head
    assert "candidate_ids=(int(_ple_token),)" in head


def test_hook_is_resolved_once_per_request():
    """No flag read, no getattr walk, no try/except inside the decode cycle."""

    hoist = GENERATION_TEXT.split("_ple_candidate_submit = None", 1)[1]
    hoist = hoist[: hoist.index("_pr391_compact_commit =")]
    assert "submit_fixed_m4_candidates" in hoist
    assert GENERATION_TEXT.count("_ple_candidate_submit is not None") == 2
    assert "ple_candidate_prefetch.enabled()" not in GENERATION_TEXT


def test_flag_is_registered_for_validated_operator_overrides():
    assert f'"{lane.ENV_FLAG}"' in PROFILES_TEXT


def test_readme_documents_the_arm_and_both_cells():
    assert f"--candidate-extra-env {lane.ENV_FLAG}=1" in README_TEXT
    assert "cold_declines" in README_TEXT
    assert "ple-candidate-prefetch-phase2.md" in README_TEXT


def test_phase2_is_a_note_and_not_code():
    note = ROOT / "docs" / "perf" / "ple-candidate-prefetch-phase2.md"
    assert note.exists()
    assert "Nothing here is built" in note.read_text("utf-8")


def test_lane_module_never_imports_mlx():
    text = (ROOT / "mtplx" / "ple_candidate_prefetch.py").read_text("utf-8")
    assert "import mlx" not in text
    assert "import mx" not in text
