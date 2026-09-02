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
import micro_dependent_launch as mdl  # noqa: E402
import micro_dispatch_overhead as mdo  # noqa: E402
import micro_expert_major as mem  # noqa: E402
import micro_hc_read as mhr  # noqa: E402
import micro_moe_dedup as mmd  # noqa: E402


def test_microbenchmarks_import_without_mlx():
    # In a clean interpreter, not "in whatever interpreter pytest has already
    # dragged MLX into": any earlier test module in the session imports MLX,
    # which used to make this assertion order-dependent.
    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(FABLE)!r});"
        "import expert_id_patterns, micro_dispatch_overhead, micro_moe_dedup;"
        "import micro_expert_major, micro_hc_read, micro_dependent_launch;"
        "leaked=[m for m in sys.modules if m == 'mlx' or m.startswith('mlx.')"
        "        or m == 'mtplx' or m.startswith('mtplx.')];"
        "print(leaked, micro_moe_dedup.mx, micro_expert_major.mx, micro_hc_read.mx,"
        "      micro_dependent_launch.mx)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "[] None None None None", out
    assert mmd.mx is None
    assert mem.mx is None
    assert mhr.mx is None
    assert mdl.mx is None


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


# ---------------------------------------------------------------------------
# micro_hc_read.py
# ---------------------------------------------------------------------------


def test_hc_read_defaults():
    args = mhr.build_parser().parse_args([])
    assert args.rows == 4, "the fixed-M4 verifier's physical width"
    assert args.calls == 97, "2 reads per layer x 48 + the trunk mixer"
    assert args.noncombine == 1, "only the trunk mixer is use_combine=False"
    assert args.reps == 20
    assert args.warmup == 3
    assert args.variants == "a,bn,bd,b,c"
    assert args.shared_weights is False, (
        "the real cycle streams 1.28 GB of PRIVATE weights; a shared-weight "
        "default would measure an L2 hit that does not exist"
    )
    assert args.sweep is None and args.out is None
    assert args.out_per_tg is None and args.d_per_block is None


def test_hc_read_dispatch_table_covers_every_variant():
    default = mhr.build_parser().parse_args([]).variants.split(",")
    assert set(default) == set(mhr.DISPATCHES)
    assert mhr.DISPATCHES["a"] == 11, "the census counts 11 x 97 = 1,067/cycle"
    assert mhr.DISPATCHES["b"] == 3
    assert mhr.DISPATCHES["bn"] < mhr.DISPATCHES["bd"] < mhr.DISPATCHES["b"]


def test_hc_read_byte_model_matches_the_census():
    """The census correction: down is [320, 10240] = 6.55 MB, not 320x2560."""

    per = mhr.weight_bytes(True)
    assert per["down"] == 320 * 10240 * 2 == 6_553_600
    assert per["up"] == per["down"]
    assert per["inject"] == 4 * 10240 * 2
    assert per["total"] == 13_209_600
    assert mhr.weight_bytes(False)["inject"] == 0

    cyc = mhr.cycle_bytes(97, 1)
    assert cyc["total"] == 96 * 13_209_600 + 13_127_680
    assert cyc["total"] == 1_281_249_280, "~1.28 GB of mix weights per cycle"


def test_hc_read_sweep_parsing():
    assert mhr.parse_sweep("4:256:8, 6:512:16") == [(4, 256, 8), (6, 512, 16)]
    assert mhr.parse_sweep("") == []
    with pytest.raises(ValueError):
        mhr.parse_sweep("4:256")


def test_hc_read_rows_and_calls_are_overridable():
    args = mhr.build_parser().parse_args(
        ["--rows", "8", "--calls", "4", "--variants", "a,b", "--out-per-tg", "6"]
    )
    assert args.rows == 8 and args.calls == 4 and args.out_per_tg == 6


# ---------------------------------------------------------------------------
# micro_dependent_launch.py
#
# Everything below runs off-GPU.  The bench itself needs the lock; its argv
# surface, its shape derivation, its least-squares fit and its verdict do not,
# and those are the parts a reader has to be able to argue with before anyone
# spends a GPU window on it.
# ---------------------------------------------------------------------------


def test_dependent_launch_defaults():
    args = mdl.build_parser().parse_args([])
    assert args.shapes == "default"
    assert args.n_grid == "0,2,5,10,20"
    assert args.repetitions is None
    assert args.reps == 200
    assert args.warmup == 10
    assert args.arms == "A,B,C"
    assert args.min_r2 == 0.90
    assert args.max_weight_bytes == 2 * 2**30
    assert args.json is None
    assert args.print_shapes is False
    assert args.config.endswith(
        "Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed/config.json"
    )


def test_dependent_launch_resolve_defaults_need_no_mlx():
    shapes, grid, arms, budget = mdl.resolve(mdl.build_parser().parse_args([]))
    assert shapes == mdl.default_shapes()
    assert grid == (0, 2, 5, 10, 20)
    assert arms == ("A", "B", "C")
    # 36 x (26.37 + 16.71) MB has to fit beside nothing else on the box.
    assert budget["total_bytes"] == 1_550_868_480
    assert budget["total_bytes"] < 2 * 2**30


def test_dependent_launch_repetitions_override_rescales_the_budget():
    args = mdl.build_parser().parse_args(["--repetitions", "8"])
    shapes, _, _, budget = mdl.resolve(args)
    assert shapes.gdn_layers == 8
    assert budget["total_bytes"] == 8 * budget["per_repetition_bytes"]


@pytest.mark.parametrize("grid", ["", "5", "5,5", "0,-2"])
def test_dependent_launch_rejects_unfittable_grids(grid):
    """A one-point (or negative) grid cannot produce a slope, so it must not
    be allowed to reach the GPU window and produce a number anyway."""

    with pytest.raises(SystemExit):
        mdl.resolve(mdl.build_parser().parse_args(["--n-grid", grid]))


def test_dependent_launch_rejects_unknown_arm():
    with pytest.raises(SystemExit):
        mdl.resolve(mdl.build_parser().parse_args(["--arms", "A,D"]))


def test_dependent_launch_refuses_an_oversized_weight_set():
    args = mdl.build_parser().parse_args(
        ["--repetitions", "400", "--max-weight-bytes", str(2 * 2**30)]
    )
    with pytest.raises(SystemExit):
        mdl.resolve(args)


# --- shapes ---------------------------------------------------------------


def test_quantized_bytes_matches_the_hand_arithmetic():
    # q4/gs32 [16480, 2560]: 21,094,400 packed + 2 x 2,636,800 bf16 scale/bias
    assert mdl.quantized_bytes(16480, 2560, 4, 32) == 26_368_000
    # q8/gs64 [2560, 6144]: 15,728,640 packed + 2 x 491,520
    assert mdl.quantized_bytes(2560, 6144, 8, 64) == 16_711_680
    with pytest.raises(ValueError):
        mdl.quantized_bytes(16, 100, 4, 32)


def _synthetic_config(**over):
    text = {
        "hidden_size": 2560,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "num_hidden_layers": 48,
        "full_attention_interval": 4,
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(48)
        ],
    }
    text.update(over)
    return {
        "text_config": text,
        "quantization_config": {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                "bits": 4, "group_size": 32, "mode": "affine",
            },
            "language_model.model.layers.0.linear_attn.out_proj": {
                "bits": 8, "group_size": 64, "mode": "affine",
            },
        },
    }


def test_derive_shapes_reconstructs_the_fused_in_proj():
    shapes = mdl.derive_shapes_from_config(_synthetic_config())
    # qkv 10240 + z 6144 + b 48 + a 48
    assert shapes.in_proj_rows == 16480
    assert (shapes.qkv_dim, shapes.value_dim) == (10240, 6144)
    assert (shapes.in_bits, shapes.in_group) == (4, 32)
    assert (shapes.out_bits, shapes.out_group) == (8, 64)
    assert shapes.gdn_layers == 36
    assert shapes.source == "real"


def test_derive_shapes_falls_back_to_the_attention_interval():
    cfg = _synthetic_config()
    del cfg["text_config"]["layer_types"]
    assert mdl.derive_shapes_from_config(cfg).gdn_layers == 36


def test_derive_shapes_reads_a_flat_config_without_text_config():
    cfg = _synthetic_config()
    flat = dict(cfg["text_config"])
    flat["quantization_config"] = cfg["quantization_config"]
    assert mdl.derive_shapes_from_config(flat).in_proj_rows == 16480


REAL_CONFIG = Path(mdl.DEFAULT_CONFIG)


@pytest.mark.skipif(not REAL_CONFIG.is_file(), reason="served model not on this box")
def test_real_config_reproduces_the_hardcoded_defaults():
    """The staleness guard: --shapes real and --shapes default must agree.

    If this ever fails the constants at the top of the bench are stale and the
    default arm has been pricing a geometry the model no longer has.
    """

    real = mdl.derive_shapes_from_config(json.loads(REAL_CONFIG.read_text()))
    default = mdl.default_shapes()
    assert asdict_no_source(real) == asdict_no_source(default)


def asdict_no_source(shapes):
    import dataclasses

    fields = dataclasses.asdict(shapes)
    fields.pop("source")
    return fields


def test_load_shapes_reports_a_missing_config_instead_of_crashing(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        mdl.load_shapes("real", tmp_path / "nope.json")
    assert "JSON only" in str(excinfo.value)


@pytest.mark.parametrize(
    "over",
    [
        {"hidden": 0},
        {"in_group": 7},          # hidden not divisible by the group size
        {"out_group": 7},         # value_dim not divisible by the group size
        {"in_proj_rows": 4096},   # cannot hold qkv + z
        {"gdn_layers": 1},        # one point cannot fit a slope
    ],
)
def test_shapes_validate_rejects_unbuildable_geometry(over):
    import dataclasses

    fields = dataclasses.asdict(mdl.default_shapes())
    fields.update(over)
    with pytest.raises(ValueError):
        mdl.Shapes(**fields).validate()


# --- fit ------------------------------------------------------------------


def test_linear_fit_recovers_an_exact_line():
    fit = mdl.linear_fit([0, 2, 5, 10, 20], [1.0 + 0.03 * n for n in (0, 2, 5, 10, 20)])
    assert fit["slope"] == pytest.approx(0.03)
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["points"] == 5


def test_linear_fit_calls_a_flat_series_a_perfect_fit():
    """Arm B is EXPECTED to be flat; a flat series must not read as unfittable."""

    fit = mdl.linear_fit([0, 2, 5, 10, 20], [1.0] * 5)
    assert fit["slope"] == pytest.approx(0.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_linear_fit_r2_falls_on_a_noisy_series():
    fit = mdl.linear_fit([0, 1, 2, 3], [0.0, 3.0, 1.0, 4.0])
    assert 0.0 < fit["r2"] < 0.9


def test_linear_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        mdl.linear_fit([1, 1, 1], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mdl.linear_fit([1], [1.0])
    with pytest.raises(ValueError):
        mdl.linear_fit([1, 2], [1.0])


def test_per_unit_us_divides_by_the_repetition_count():
    # 0.036 ms per unit of N over 36 repetitions == 1.0 us per repetition per N
    assert mdl.per_unit_us(0.036, 36) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mdl.per_unit_us(0.036, 0)


def test_summarize_arm_fits_both_axes():
    rows = [
        {"n": n, "median_ms": 10.0 + 0.036 * n, "launches": 100 + 2 * n}
        for n in (0, 2, 5, 10, 20)
    ]
    out = mdl.summarize_arm(rows, 36, 0.90)
    assert out["us_per_op"] == pytest.approx(1.0)
    # 2 measured dispatches per unit of N -> half the per-op price each
    assert out["us_per_dispatch"] == pytest.approx(18.0)
    assert out["quotable"] is True


def test_summarize_arm_omits_the_dispatch_fit_when_counts_are_flat():
    rows = [
        {"n": n, "median_ms": 10.0 + 0.01 * n, "launches": 100}
        for n in (0, 5, 10)
    ]
    out = mdl.summarize_arm(rows, 36, 0.90)
    assert out["dispatch_fit"] is None
    assert out["us_per_dispatch"] is None


# --- verdict --------------------------------------------------------------


def test_verdict_says_fusions_pay_at_or_above_one_microsecond():
    for us in (1.0, 2.0):
        line = mdl.verdict_line(us, r2=0.99)
        assert mdl.VERDICT_FUSION in line
        assert mdl.VERDICT_BYTES not in line


def test_verdict_says_only_bytes_pay_at_or_below_half_a_microsecond():
    for us in (0.5, 0.1, 0.0):
        line = mdl.verdict_line(us, r2=0.99)
        assert mdl.VERDICT_BYTES in line
        assert mdl.VERDICT_FUSION not in line


def test_verdict_is_inconclusive_inside_the_band():
    line = mdl.verdict_line(0.75, r2=0.99)
    assert "INCONCLUSIVE" in line
    assert mdl.VERDICT_FUSION not in line and mdl.VERDICT_BYTES not in line


def test_verdict_refuses_to_quote_a_bad_fit():
    """A thermally drifted sweep must not be laundered into a per-launch price."""

    line = mdl.verdict_line(2.0, r2=0.42)
    assert "LOW-CONFIDENCE FIT" in line
    assert mdl.VERDICT_FUSION not in line
    # exactly at the threshold the fit is still quotable
    assert mdl.VERDICT_FUSION in mdl.verdict_line(2.0, r2=0.90)


# --- tiny kernel plan -----------------------------------------------------


def test_tiny_plan_cycles_the_real_gdn_kernel_set():
    assert mdl.tiny_plan(0) == []
    assert mdl.tiny_plan(5) == list(mdl.TINY_CYCLE)
    assert mdl.tiny_plan(2) == ["gate", "norm"]
    plan = mdl.tiny_plan(20)
    assert len(plan) == 20
    # 20 is 4 full cycles, so every kind is equally represented
    assert {plan.count(kind) for kind in mdl.TINY_CYCLE} == {4}
    with pytest.raises(ValueError):
        mdl.tiny_plan(-1)


def test_tiny_cycle_pushes_the_unavoidable_fusion_onto_the_wrap():
    """mx.compile fuses adjacent elementwise chains, so a run of them turns
    "N ops" into fewer than N launches and halves the reported price.

    Three of the five kinds are elementwise and three items cannot be pairwise
    separated in a cycle of five, so exactly one adjacent pair is unavoidable.
    It must sit on the WRAP, which keeps N <= 5 free of it entirely.
    """

    cycle = mdl.TINY_CYCLE
    assert set(cycle) == set(mdl.tiny_plan(len(cycle)))
    assert len(mdl.ELEMENTWISE_KINDS & set(cycle)) == 3

    def adjacent_pairs(plan):
        return [
            (a, b)
            for a, b in zip(plan, plan[1:])
            if a in mdl.ELEMENTWISE_KINDS and b in mdl.ELEMENTWISE_KINDS
        ]

    # No collision anywhere inside one pass...
    assert adjacent_pairs(mdl.tiny_plan(5)) == []
    # ...and exactly one per wrap after that, never two.
    assert len(adjacent_pairs(mdl.tiny_plan(10))) == 1
    assert len(adjacent_pairs(mdl.tiny_plan(20))) == 3


def test_print_shapes_runs_without_importing_mlx():
    probe = subprocess.run(
        [
            sys.executable,
            str(FABLE / "micro_dependent_launch.py"),
            "--print-shapes",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "in_proj [16480, 2560] q4/gs32" in probe.stdout
    assert '"total_bytes": 1550868480' in probe.stdout


def test_n_grid_is_sorted_and_deduped():
    """The table's marginal column reads against rows[0], so the grid has to
    arrive in increasing order however the user typed it."""

    _, grid, _, _ = mdl.resolve(
        mdl.build_parser().parse_args(["--n-grid", "20,0,5,5,2,10"])
    )
    assert grid == (0, 2, 5, 10, 20)


def test_command_buffer_estimate_is_a_ceiling():
    assert mdl.command_buffers(0, 50) == 0
    assert mdl.command_buffers(1, 50) == 1
    assert mdl.command_buffers(50, 50) == 1
    assert mdl.command_buffers(51, 50) == 2
    with pytest.raises(ValueError):
        mdl.command_buffers(10, 0)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, 50),
        ({"MLX_MAX_OPS_PER_BUFFER": "1000"}, 1000),
        ({"MLX_MAX_OPS_PER_BUFFER": ""}, 50),
        ({"MLX_MAX_OPS_PER_BUFFER": "nonsense"}, 50),
        ({"MLX_MAX_OPS_PER_BUFFER": "0"}, 50),
    ],
)
def test_ops_per_buffer_matches_the_mlx_default_on_anything_unset(env, expected):
    assert mdl.ops_per_buffer(env) == expected
