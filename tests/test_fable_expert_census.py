"""Pure-python coverage for the opt-in M4 expert-id census.

No MLX import happens here: the census only reaches for ``mlx.core`` inside
``end_cycle``, so a tiny stub with an ``eval`` recorder is enough and the
suite stays off the GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mtplx import fable_expert_census as census_mod


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "fable" / "expert_census_report.py"


class _FakeMx:
    """Stub ``mlx.core``: records every ``eval`` and touches no device."""

    def __init__(self) -> None:
        self.eval_calls: list[tuple] = []

    def eval(self, *args):
        self.eval_calls.append(args)


@pytest.fixture
def fake_mx(monkeypatch):
    stub = _FakeMx()
    package = types.ModuleType("mlx")
    package.core = stub
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", stub)
    return stub


@pytest.fixture
def census(monkeypatch):
    """Point the module singleton at a test path and restore it after."""

    def configure(path):
        census_mod._configure_for_test(str(path) if path is not None else None)
        return census_mod.census

    yield configure
    census_mod._configure_for_test(None)


def _ids(*rows):
    """Build one layer's [4, 10] selection from four ten-id rows."""

    return np.asarray(rows, dtype=np.int32)


def _layer(seed: int):
    base = np.arange(10, dtype=np.int32)
    return _ids(base + seed, base + seed, base + seed + 10, base + seed + 20)


# --------------------------------------------------------------------
# disabled path
# --------------------------------------------------------------------


def test_census_is_disabled_without_the_env_var():
    # conftest never sets MTPLX_FABLE_EXPERT_CENSUS, so the singleton built
    # at import time must be inert.
    assert census_mod._ENABLED is False
    assert census_mod.census.enabled is False


def test_disabled_census_is_a_noop(tmp_path, fake_mx):
    target = tmp_path / "census.npz"
    subject = census_mod.census
    subject.record(0, _layer(0))
    subject.end_cycle()
    assert subject.flush() is None
    assert fake_mx.eval_calls == []
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_module_level_flag_gates_every_method(tmp_path, fake_mx, census, monkeypatch):
    subject = census(tmp_path / "census.npz")
    monkeypatch.setattr(census_mod, "_ENABLED", False)
    assert subject.enabled is True  # instance flag alone must not be enough
    subject.record(0, _layer(0))
    subject.end_cycle()
    assert subject.flush() is None
    assert fake_mx.eval_calls == []


# --------------------------------------------------------------------
# enabled path
# --------------------------------------------------------------------


def test_record_is_lazy_and_end_cycle_evaluates_once(tmp_path, fake_mx, census):
    subject = census(tmp_path / "census.npz")
    for layer in range(3):
        subject.record(layer, _layer(layer))
    assert fake_mx.eval_calls == []  # nothing evaluated while recording

    subject.end_cycle()
    assert len(fake_mx.eval_calls) == 1
    (buffered,) = fake_mx.eval_calls[0]  # a single eval over the whole list
    assert len(buffered) == 3


def test_flush_writes_stacked_ids(tmp_path, fake_mx, census):
    out = tmp_path / "census.npz"
    subject = census(out)
    for _cycle in range(2):
        for layer in range(3):
            subject.record(layer, _layer(layer))
        subject.end_cycle()
    assert subject.flush() == str(out)

    with np.load(out) as data:
        ids = data["ids"]
        layer_ids = data["layer_ids"]
    assert ids.shape == (2, 3, 4, 10)
    assert ids.dtype == np.int16
    assert layer_ids.tolist() == [0, 1, 2]
    assert ids[0, 2].tolist() == _layer(2).tolist()
    assert subject.dropped_cycles == 0


def test_flush_closes_an_open_cycle(tmp_path, fake_mx, census):
    out = tmp_path / "census.npz"
    subject = census(out)
    subject.record(0, _layer(0))
    subject.record(1, _layer(1))
    subject.flush()
    with np.load(out) as data:
        assert data["ids"].shape == (1, 2, 4, 10)


def test_offshape_cycles_are_dropped(tmp_path, fake_mx, census):
    """A construction-time self-check run must not corrupt the stack.

    The self-check forwards run before any verify window exists, so they
    ride along in the first cycle the census closes.  That cycle then has
    the wrong layer roster and is dropped whole; every later cycle is a
    clean verify window.
    """

    out = tmp_path / "census.npz"
    subject = census(out)
    for layer in range(6):  # install self-check: 6 layers, no end_cycle
        subject.record(layer, _layer(layer))
    for _cycle in range(3):  # real verify windows: 3 layers each
        for layer in range(3):
            subject.record(layer, _layer(layer))
        subject.end_cycle()
    subject.flush()

    with np.load(out) as data:
        ids = data["ids"]
        layer_ids = data["layer_ids"]
    assert ids.shape == (2, 3, 4, 10)  # first cycle carried the self-check
    assert layer_ids.tolist() == [0, 1, 2]
    assert subject.dropped_cycles == 1


def test_json_path_writes_meta_beside_the_npz(tmp_path, fake_mx, census):
    out = tmp_path / "nested" / "census.json"
    subject = census(out)
    subject.record(0, _layer(0))
    subject.end_cycle()
    npz = subject.flush()

    assert npz == str(tmp_path / "nested" / "census.npz")
    assert Path(npz).exists()
    meta = json.loads(out.read_text())
    assert meta["npz"] == npz
    assert meta["cycles"] == 1
    assert meta["layers"] == 1
    assert meta["rows"] == 4
    assert meta["top_k"] == 10


def test_flush_without_data_writes_nothing(tmp_path, fake_mx, census):
    out = tmp_path / "census.npz"
    subject = census(out)
    assert subject.flush() is None
    assert not out.exists()


def test_npz_path_normalisation():
    assert census_mod._npz_path("/a/b.npz") == "/a/b.npz"
    assert census_mod._npz_path("/a/b.json") == "/a/b.npz"
    assert census_mod._npz_path("/a/b") == "/a/b.npz"


# --------------------------------------------------------------------
# report
# --------------------------------------------------------------------


def _load_report():
    spec = importlib.util.spec_from_file_location(
        "fable_expert_census_report", REPORT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_union_counts_and_jaccard():
    report = _load_report()
    base = np.arange(10, dtype=np.int32)
    # rows 0 and 1 identical, rows 2 and 3 disjoint -> 30 distinct of 40.
    layer = np.stack([base, base, base + 10, base + 20])
    ids = layer.reshape(1, 1, 4, 10)

    assert report.union_counts(ids).tolist() == [[30]]
    jaccard = report.consecutive_row_jaccard(ids)
    assert jaccard.shape == (1, 1, 3)
    assert jaccard[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_report_union_counts_all_distinct_and_all_shared():
    report = _load_report()
    distinct = np.arange(40, dtype=np.int32).reshape(1, 1, 4, 10)
    assert report.union_counts(distinct).tolist() == [[40]]

    shared = np.tile(np.arange(10, dtype=np.int32), (4, 1)).reshape(1, 1, 4, 10)
    assert report.union_counts(shared).tolist() == [[10]]
    assert report.consecutive_row_jaccard(shared).min() == pytest.approx(1.0)


def test_report_runs_and_writes_sample(tmp_path, capsys):
    report = _load_report()
    rng = np.random.default_rng(7)
    cycles, layers = 5, 4
    ids = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            rng.choice(512, size=10, replace=False)
                            for _row in range(4)
                        ]
                    )
                    for _layer in range(layers)
                ]
            )
            for _cycle in range(cycles)
        ]
    ).astype(np.int16)
    npz = tmp_path / "census.npz"
    np.savez_compressed(
        npz, ids=ids, layer_ids=np.arange(layers, dtype=np.int16)
    )
    sample_out = tmp_path / "sample.json"

    report.main([str(npz), "--sample-out", str(sample_out), "--sample-count", "7"])
    printed = capsys.readouterr().out
    assert "distinct experts per layer-cycle (U)" in printed
    assert "mean U per layer index" in printed
    assert "pairwise row overlap" in printed
    assert "implied routed-weight-bytes saving" in printed

    sample = json.loads(sample_out.read_text())
    assert len(sample) == 7
    assert np.asarray(sample).shape == (7, 4, 10)
    assert sample[0] == ids.reshape(cycles * layers, 4, 10)[0].tolist()


def test_report_saving_fraction_matches_mean_u(tmp_path, capsys):
    report = _load_report()
    base = np.arange(10, dtype=np.int16)
    layer = np.stack([base, base, base + 10, base + 20])  # U = 30 of 40
    ids = np.tile(layer, (3, 2, 1, 1)).reshape(3, 2, 4, 10)
    npz = tmp_path / "census.npz"
    np.savez_compressed(npz, ids=ids, layer_ids=np.arange(2, dtype=np.int16))

    report.main([str(npz)])
    printed = capsys.readouterr().out
    assert "= 0.2500  (25.00%)" in printed
