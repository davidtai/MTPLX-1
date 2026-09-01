"""CLI and routing-pattern invariants for the fable microbenchmarks.

Runs off-GPU: nothing here imports MLX, which is exactly the property that lets
these two scripts be argued about without holding /tmp/mtplx-gpu-exclusive.lock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

FABLE = Path(__file__).resolve().parents[1] / "scripts" / "fable"
if str(FABLE) not in sys.path:
    sys.path.insert(0, str(FABLE))

import expert_id_patterns as eip  # noqa: E402
import micro_dispatch_overhead as mdo  # noqa: E402
import micro_moe_dedup as mmd  # noqa: E402


def test_microbenchmarks_import_without_mlx():
    assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)
    assert mmd.mx is None


def test_moe_dedup_defaults():
    args = mmd.build_parser().parse_args([])
    assert args.unique == 40
    assert args.layers == 48
    assert args.reps == 20
    assert args.warmup == 3
    assert args.variants == "a,b1,b2,c,c2"
    assert args.from_census is None
    assert args.out is None
    assert args.clear_cache is False


def test_moe_dedup_rejects_unsupported_unique():
    with pytest.raises(SystemExit):
        mmd.build_parser().parse_args(["--unique", "37"])


def test_moe_dedup_dispatch_table_covers_every_variant():
    default = mmd.build_parser().parse_args([]).variants.split(",")
    assert set(default) == set(mmd.DISPATCHES)


def test_dispatch_overhead_defaults():
    args = mdo.parse_args([])
    assert args.reps == 20
    assert args.layers == 48
    assert args.chain_lengths == (100, 1000, 5000)
    assert args.ops_sweep == (50, 100, 200, 500, 1000)
    assert args.ops_sweep[0] == 50, "MLX 0.32.2 default must lead the sweep"
    assert args.child is False
    assert args.max_mb_per_buffer is None


def test_dispatch_overhead_sweep_override():
    args = mdo.parse_args(["--ops-sweep", "50,4096", "--chain-lengths", "10"])
    assert args.ops_sweep == (50, 4096)
    assert args.chain_lengths == (10,)


@pytest.mark.parametrize("unique", eip.UNIQUE_CHOICES)
def test_generated_ids_hold_the_routing_invariants(unique):
    rng = np.random.default_rng(1234)
    for _ in range(50):
        ids = eip.make_expert_ids(unique, rng=rng)
        assert ids.shape == (eip.ROWS, eip.TOP_K)
        assert eip.unique_count(ids) == unique
        for row in ids:
            assert np.unique(row).size == eip.TOP_K
        assert ids.min() >= 0 and ids.max() < eip.NUM_EXPERTS


def test_generator_is_seed_deterministic():
    a = eip.make_expert_ids(28, rng=np.random.default_rng(7))
    b = eip.make_expert_ids(28, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("unique", [0, 9, 41, 100])
def test_generator_rejects_impossible_unique_counts(unique):
    with pytest.raises(ValueError):
        eip.make_expert_ids(unique)


def test_layer_id_sets_are_independent_and_sized():
    sets = eip.make_layer_id_sets(22, 48, seed=3)
    assert len(sets) == 48
    assert all(eip.unique_count(s) == 22 for s in sets)
    assert not all(np.array_equal(sets[0], s) for s in sets[1:])


def test_validate_rejects_bad_shape_and_repeated_row():
    with pytest.raises(ValueError):
        eip.validate_ids(np.zeros((4, 9), dtype=np.int32))
    bad = eip.make_expert_ids(40, rng=np.random.default_rng(0))
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError):
        eip.validate_ids(bad)
    with pytest.raises(ValueError):
        eip.validate_ids(np.full((4, 10), eip.NUM_EXPERTS, dtype=np.int32))


def test_census_replay_reproduces_the_recorded_unique_counts(tmp_path):
    rng = np.random.default_rng(11)
    census = [eip.make_expert_ids(u, rng=rng).tolist() for u in (16, 28, 40)]
    path = tmp_path / "census.json"
    path.write_text(json.dumps(census))

    sets = eip.load_census_id_sets(path, 48, seed=5)
    assert len(sets) == 48
    recorded = {eip.unique_count(np.asarray(c)) for c in census}
    assert {eip.unique_count(s) for s in sets} <= recorded
    for s in sets:
        eip.validate_ids(s)


def test_census_rejects_empty_and_malformed(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(ValueError):
        eip.load_census_id_sets(empty, 4)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([[[0] * 10] * 4]))
    with pytest.raises(ValueError):
        eip.load_census_id_sets(bad, 4)
