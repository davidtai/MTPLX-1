"""CLI and routing-pattern invariants for the fable microbenchmarks.

Runs off-GPU: nothing here imports MLX, which is exactly the property that lets
these two scripts be argued about without holding /tmp/mtplx-gpu-exclusive.lock.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FABLE = Path(__file__).resolve().parents[1] / "scripts" / "fable"
if str(FABLE) not in sys.path:
    sys.path.insert(0, str(FABLE))

import expert_id_patterns as eip  # noqa: E402
import micro_dispatch_overhead as mdo  # noqa: E402
import micro_expert_major as mem  # noqa: E402
import micro_moe_dedup as mmd  # noqa: E402


def test_microbenchmarks_import_without_mlx():
    # In a clean interpreter, not "in whatever interpreter pytest has already
    # dragged MLX into": any earlier test module in the session imports MLX,
    # which used to make this assertion order-dependent.
    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(FABLE)!r});"
        "import expert_id_patterns, micro_dispatch_overhead, micro_moe_dedup;"
        "import micro_expert_major;"
        "leaked=[m for m in sys.modules if m == 'mlx' or m.startswith('mlx.')];"
        "print(leaked, micro_moe_dedup.mx, micro_expert_major.mx)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "[] None None", out
    assert mmd.mx is None
    assert mem.mx is None


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


def test_mixed_delta_sign_is_positive_when_slower():
    """Regression: the report printed `base - value`, so a run that got
    SLOWER (1.240 -> 1.661 ms) came out as -33.97% instead of +33.95%."""

    delta, pct = mdo.mixed_delta_vs_baseline(1.240, 1.661)
    assert delta == pytest.approx(0.421)
    assert pct == pytest.approx(100.0 * 0.421 / 1.240)
    assert pct > 0

    delta, pct = mdo.mixed_delta_vs_baseline(1.240, 0.930)
    assert delta < 0 and pct < 0

    assert mdo.mixed_delta_vs_baseline(1.240, 1.240) == (0.0, 0.0)
    assert mdo.mixed_delta_vs_baseline(0.0, 1.0) == (1.0, 0.0)


# ---------------------------------------------------------------------------
# micro_expert_major.py
# ---------------------------------------------------------------------------


def test_expert_major_defaults():
    args = mem.build_parser().parse_args([])
    assert args.unique == 28, "the census mean is the interesting operating point"
    assert args.layers == 48
    assert args.reps == 20
    assert args.warmup == 3
    assert args.variants == "a,b4,b2,c"
    assert args.from_census is None
    assert args.skip_adversarial is False
    assert args.out is None


def test_expert_major_rejects_unsupported_unique():
    with pytest.raises(SystemExit):
        mem.build_parser().parse_args(["--unique", "37"])


def test_expert_major_dispatch_table_covers_every_variant():
    default = mem.build_parser().parse_args([]).variants.split(",")
    assert set(default) == set(mem.DISPATCHES) == set(mem.VARIANTS)


def test_expert_major_kernel_variants_cost_no_extra_dispatches():
    # The whole point of recomputing the plan inside the threadgroup: a
    # stock-op plan builder would have made b4/b2 cost ~17 dispatches/layer.
    assert mem.DISPATCHES["b4"] == mem.DISPATCHES["a"] == 1
    assert mem.DISPATCHES["b2"] == 1


def test_expert_major_byte_model_matches_the_q4_g32_recipe():
    # 1280x2560 fused gate+up at 4 bits + one bf16 scale and bias per 32.
    assert mem.gu_bytes_per_expert() == 1280 * 2560 // 2 + 2 * 1280 * 80 * 2
    assert mem.gu_bytes_per_expert() == 2_048_000


def test_expert_major_only_the_kernel_variants_claim_the_dedup():
    uniques = [28] * 48
    assert mem.tiles_streamed("a", uniques) == float(eip.SLOTS)
    assert mem.tiles_streamed("c", uniques) == float(eip.SLOTS)
    assert mem.tiles_streamed("b4", uniques) == 28.0
    assert mem.tiles_streamed("b2", uniques) == 28.0


def test_expert_major_adversarial_sets_are_legal_and_hit_the_edges():
    sets = mem.adversarial_id_sets()
    assert set(sets) == {"all-distinct", "all-same", "one-expert-in-four-rows"}
    for ids in sets.values():
        eip.validate_ids(ids)
        expert, member = eip.expert_major_plan(ids)
        eip.validate_expert_major_plan(ids, expert, member)
    assert eip.unique_count(sets["all-distinct"]) == eip.SLOTS
    assert eip.unique_count(sets["all-same"]) == eip.TOP_K
    shared = sets["one-expert-in-four-rows"]
    assert eip.unique_count(shared) == eip.SLOTS - 3
    _, member = eip.expert_major_plan(shared)
    assert list(member[0]) == [0, 10, 20, 30]
