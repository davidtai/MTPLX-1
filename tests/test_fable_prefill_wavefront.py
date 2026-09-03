"""Wavefront prefill: schedule legality, memory model, refusals, wiring.

Two tiers, both runnable without the GPU lock:

* Pure arithmetic (no MLX at all) for the schedule, the memory plan, and the
  refusals -- including the flag-off identity that makes this whole row inert.
* One CPU-pinned wiring proof: the falsifier's own ``run_tile`` on a tiny
  model, serial versus two CPU streams, asserted BIT-IDENTICAL. That is the
  exactness claim the design note makes, exercised on the only device this
  worker may touch.

Nothing here issues Metal work. ``mx.set_default_device(mx.cpu)`` leaks into
every later-collected module (pytest shares one process), so it is restored.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mtplx.fable_prefill_chunk import (
    DEFAULT_CHUNK_SIZE,
    PrefillChunkMemoryError,
    plan_prefill_chunk_memory,
)
from mtplx.fable_prefill_wavefront import (
    DEFAULT_LANES,
    ENV_FLAG,
    LANES_ENV,
    TAIL_SOLO_ENV,
    PrefillWavefrontError,
    assert_boundary_capture_compatible,
    boundary_records_per_prompt,
    chunk_groups,
    enabled,
    guard_wavefront_geometry,
    lanes_live,
    overlappable_step_fraction,
    plan_wavefront_memory,
    resolve_lanes,
    tail_solo,
    wavefront_steps,
)

ROOT = Path(__file__).resolve().parents[1]
FABLE = ROOT / "scripts" / "fable"
SCRIPT = FABLE / "micro_two_stream_prefill.py"

#: The production cell: 16,384 prompt tokens, 8 x 2,048, 48 layers.
PROMPT_TOKENS = 16_384
LAYERS = 48
#: mtplx.memory_plan's dense QSA prefill model at the shipped 2,048 width.
TRANSIENT_PER_TOKEN = int(12.75 * DEFAULT_CHUNK_SIZE * 4)
#: scripts/fable/abba_driver.py WIRED_LIMIT_BYTES.
DRIVER_WIRED_LIMIT = 90 * 1024**3
#: .benchmark-artifacts/fable/*.json peak_memory_bytes.
CENSUS_PEAK_BYTES = 87_393_815_544
CENSUS_RESIDENT_BYTES = CENSUS_PEAK_BYTES - TRANSIENT_PER_TOKEN * PROMPT_TOKENS


# ---------------------------------------------------------------------------
# Flag-off identity: the whole row is inert until someone arms it
# ---------------------------------------------------------------------------
def test_flag_is_off_by_default():
    assert enabled({}) is False
    assert enabled({ENV_FLAG: ""}) is False
    assert enabled({ENV_FLAG: "0"}) is False
    assert enabled({ENV_FLAG: "1"}) is True


def test_lanes_and_tail_solo_defaults():
    assert resolve_lanes({}) == DEFAULT_LANES == 2
    assert resolve_lanes({LANES_ENV: "3"}) == 3
    assert resolve_lanes({LANES_ENV: "not-a-number"}) == DEFAULT_LANES
    assert resolve_lanes({LANES_ENV: "0"}) == 1
    assert tail_solo({}) is True
    assert tail_solo({TAIL_SOLO_ENV: "0"}) is False


def test_nothing_in_the_serving_path_imports_the_wavefront_module():
    """The seam is a module, not a hook: flag-off is byte-identical stock.

    Grep, not import: an import test would only prove the module loads.
    """

    importers = [
        path
        for path in sorted((ROOT / "mtplx").rglob("*.py"))
        if path.name != "fable_prefill_wavefront.py"
        and any(
            "fable_prefill_wavefront" in line
            and ("import" in line or "import_module" in line)
            for line in path.read_text().splitlines()
        )
    ]
    assert importers == [], importers


def test_live_lanes_defaults_to_one_and_changes_nothing():
    """``plan_prefill_chunk_memory`` at lanes=1 is the pre-wavefront function."""

    base = plan_prefill_chunk_memory(
        chunk_size=2048,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
    )
    explicit = plan_prefill_chunk_memory(
        chunk_size=2048,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        live_lanes=1,
    )
    assert base == explicit
    assert base.live_lanes == 1
    assert base.as_receipt()["live_lanes"] == 1
    assert base.transient_bytes == TRANSIENT_PER_TOKEN * PROMPT_TOKENS


# ---------------------------------------------------------------------------
# Schedule legality -- the thing the whole row rests on
# ---------------------------------------------------------------------------
def test_lanes_one_reproduces_the_shipped_serial_schedule():
    steps = wavefront_steps(3, 4, lanes=1)
    assert all(len(step) == 1 for step in steps)
    order = [node for step in steps for node in step]
    assert order == [(k, layer) for k in range(3) for layer in range(4)]


def test_the_two_by_two_tile_is_the_bench_tile():
    assert wavefront_steps(2, 2, lanes=2) == [
        [(0, 0)],
        [(0, 1), (1, 0)],
        [(1, 1)],
    ]


@pytest.mark.parametrize("chunks", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("layers", [1, 2, 4, 48])
@pytest.mark.parametrize("lanes", [1, 2, 3])
@pytest.mark.parametrize("tail_solo_chunk", [False, True])
def test_schedule_is_legal_and_complete(chunks, layers, lanes, tail_solo_chunk):
    """Every node runs exactly once, and never before either predecessor.

    The two edges are ``(k, L-1) -> (k, L)`` (hidden) and
    ``(k-1, L) -> (k, L)`` (cache entry L). A schedule is legal iff both
    predecessors are issued in a STRICTLY earlier step.
    """

    steps = wavefront_steps(
        chunks, layers, lanes=lanes, tail_solo_chunk=tail_solo_chunk
    )
    issued_at: dict[tuple[int, int], int] = {}
    for index, step in enumerate(steps):
        assert len(set(step)) == len(step), f"duplicate inside step {index}"
        # Independence: no two nodes in a step share a layer (== cache entry).
        assert len({layer for _, layer in step}) == len(step), step
        for node in step:
            assert node not in issued_at, f"{node} issued twice"
            issued_at[node] = index

    assert set(issued_at) == {
        (k, layer) for k in range(chunks) for layer in range(layers)
    }
    for (k, layer), when in issued_at.items():
        if layer > 0:
            assert issued_at[(k, layer - 1)] < when, (k, layer)
        if k > 0:
            assert issued_at[(k - 1, layer)] < when, (k, layer)


@pytest.mark.parametrize("lanes", [1, 2, 3, 4])
def test_lanes_live_never_exceeds_the_bound(lanes):
    assert lanes_live(8, LAYERS, lanes=lanes) == min(lanes, 8)


def test_the_unbounded_wavefront_is_what_grouping_exists_to_refuse():
    """8 chunks in flight is not a memory question, it is an OOM."""

    assert lanes_live(8, LAYERS, lanes=0) == 8
    assert lanes_live(8, LAYERS, lanes=2) == 2


def test_draining_costs_almost_nothing_at_the_production_geometry():
    frac = overlappable_step_fraction(8, LAYERS, lanes=2)
    assert frac > 0.95, frac
    # lanes=1 has nothing to overlap at all.
    assert overlappable_step_fraction(8, LAYERS, lanes=1) == 0.0


def test_groups_and_the_solo_tail_chunk():
    assert chunk_groups(8, lanes=2) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert chunk_groups(8, lanes=2, tail_solo_chunk=True) == [
        (0, 1), (2, 3), (4, 5), (6,), (7,)
    ]
    assert chunk_groups(1, lanes=2, tail_solo_chunk=True) == [(0,)]
    assert chunk_groups(0, lanes=2) == []


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------
def test_two_lanes_cost_exactly_what_a_4096_chunk_costs_in_memory():
    """...and nothing of what it costs in attention work.

    This is the single attractive property of the row, so it gets an
    assertion rather than a sentence in a doc.
    """

    wave = plan_wavefront_memory(
        chunk_size=2048,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        lanes=2,
        resident_bytes=CENSUS_RESIDENT_BYTES,
        budget_bytes=DRIVER_WIRED_LIMIT,
    )
    wide = plan_prefill_chunk_memory(
        chunk_size=4096,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        resident_bytes=CENSUS_RESIDENT_BYTES,
        budget_bytes=DRIVER_WIRED_LIMIT,
    )
    assert wave.wavefront.projected_peak_bytes == wide.projected_peak_bytes
    assert wave.serial.projected_peak_bytes == CENSUS_PEAK_BYTES
    # The attention work term is the half that does NOT come along.
    assert (
        wave.wavefront.attention_row_context_products
        == wave.serial.attention_row_context_products
    )
    assert (
        wide.attention_row_context_products
        > wave.wavefront.attention_row_context_products
    )
    assert wave.fits and wide.fits


def test_wavefront_extra_bytes_is_the_serial_transient_times_lanes_minus_one():
    wave = plan_wavefront_memory(
        chunk_size=2048,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        lanes=3,
    )
    assert wave.extra_bytes == 2 * wave.serial.transient_bytes
    assert wave.wavefront.live_lanes == 3


def test_query_tile_caps_the_per_lane_transient():
    """The escape hatch when the wavefront and a wide chunk collide."""

    plain = plan_wavefront_memory(
        chunk_size=4096,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        lanes=2,
    )
    tiled = plan_wavefront_memory(
        chunk_size=4096,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        lanes=2,
        query_tile=2048,
    )
    assert tiled.wavefront.transient_bytes * 2 == plain.wavefront.transient_bytes
    assert tiled.wavefront.live_query_rows == 2048


def test_guard_refuses_a_wavefront_that_overruns_and_names_both_projections():
    # The unbounded continuous wavefront: 8 chunks in flight at the shipped
    # 2,048 width. 99.37 GB against a 90 GiB limit -- the OOM that grouping
    # exists to prevent.
    with pytest.raises(PrefillChunkMemoryError) as excinfo:
        guard_wavefront_geometry(
            chunk_size=2048,
            total_tokens=PROMPT_TOKENS,
            transient_bytes_per_token=TRANSIENT_PER_TOKEN,
            lanes=8,
            resident_bytes=CENSUS_RESIDENT_BYTES,
            budget_bytes=DRIVER_WIRED_LIMIT,
        )
    message = str(excinfo.value)
    assert LANES_ENV in message
    assert "MTPLX_PREFILL_CHUNK_SIZE" in message
    assert "MTPLX_FABLE_PREFILL_QSA_QUERY_TILE" in message
    # The message must name BOTH projections, so the reader can see what the
    # lanes cost rather than just that something is too big. GiB, matching
    # guard_prefill_chunk_geometry's own message units.
    assert "92.55 GiB" in message, message   # 8 lanes
    assert "81.39 GiB" in message, message   # the serial schedule


def test_two_lanes_at_a_widened_chunk_fits_but_only_just():
    """2 lanes x 4,096 == 1 lane x 8,192 in this model: 1.96 GB spare.

    Not a refusal, a warning: the wavefront and a widened chunk compose only
    down to the same razor margin the 8,192 geometry already sits on.
    """

    plan = guard_wavefront_geometry(
        chunk_size=4096,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        lanes=2,
        resident_bytes=CENSUS_RESIDENT_BYTES,
        budget_bytes=DRIVER_WIRED_LIMIT,
    )
    assert plan.fits
    assert 1.9e9 < plan.wavefront.headroom_bytes < 2.0e9


def test_guard_is_inert_without_a_budget_or_a_transient_model():
    # No budget: nothing to refuse against.
    plan = guard_wavefront_geometry(
        chunk_size=4096,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=TRANSIENT_PER_TOKEN,
        lanes=8,
    )
    assert plan.lanes == 8
    # A family with no QSA transient (transient model returns 0).
    plan = guard_wavefront_geometry(
        chunk_size=4096,
        total_tokens=PROMPT_TOKENS,
        transient_bytes_per_token=0,
        lanes=8,
        budget_bytes=1,
    )
    assert plan.wavefront.transient_bytes == 0


# ---------------------------------------------------------------------------
# The correctness refusal
# ---------------------------------------------------------------------------
def test_boundary_capture_refuses_a_non_draining_wavefront():
    with pytest.raises(PrefillWavefrontError) as excinfo:
        assert_boundary_capture_compatible(
            capture_boundaries=True, lanes=2, drains_per_group=False
        )
    assert "torn" in str(excinfo.value)
    assert LANES_ENV in str(excinfo.value)


def test_boundary_capture_is_fine_when_draining_or_serial():
    assert_boundary_capture_compatible(
        capture_boundaries=True, lanes=2, drains_per_group=True
    )
    assert_boundary_capture_compatible(
        capture_boundaries=True, lanes=1, drains_per_group=False
    )
    assert_boundary_capture_compatible(
        capture_boundaries=False, lanes=8, drains_per_group=False
    )


def test_boundary_record_count_is_stated_not_discovered():
    assert boundary_records_per_prompt(8, lanes=1) == 8
    assert boundary_records_per_prompt(8, lanes=2) == 5
    assert boundary_records_per_prompt(8, lanes=2, tail_solo_chunk=False) == 4


# ---------------------------------------------------------------------------
# The falsifier script: tripwires and off-window paths
# ---------------------------------------------------------------------------
def test_script_carries_the_gpu_window_banner():
    head = SCRIPT.read_text()[:600]
    assert "GPU WINDOW REQUIRED" in head
    assert "/tmp/mtplx-gpu-exclusive.lock" in head
    assert "run_guarded.py" in head


def test_script_imports_no_mlx_and_shares_the_schedule():
    """Off-window importable, and it does not carry its own copy of the
    schedule -- a bench that disagreed with the seam about what a wavefront
    is would price the wrong thing."""

    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(FABLE)!r});"
        "import micro_two_stream_prefill as m;"
        "import mtplx.fable_prefill_wavefront as w;"
        "leaked=[k for k in sys.modules if k=='mlx' or k.startswith('mlx.')];"
        "print(leaked, m.mx, m.nn,"
        " m.wavefront_steps is w.wavefront_steps,"
        " m.lanes_live is w.lanes_live)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[] None None True True", out.stdout


def test_script_self_test_and_shapes_run_off_window():
    for argv in (["--self-test"], ["--shapes"],
                 ["--shapes", "--context-before", "14336"]):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert out.returncode == 0, out.stderr
    assert "ok" in subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout


def test_go_gate_refuses_a_win_without_a_mechanism():
    sys.path.insert(0, str(FABLE))
    import micro_two_stream_prefill as m  # noqa: E402

    go, saving, reason = m.go_verdict(100.0, 70.0, 0.4)
    assert go and abs(saving - 0.30) < 1e-9
    # Same 30% win, but arm (c) showed no concurrency: refuse it.
    go, saving, reason = m.go_verdict(100.0, 70.0, 0.0)
    assert not go and "no concurrency" in reason
    go, _, reason = m.go_verdict(100.0, 88.0, 0.4)
    assert not go and "gate is 15%" in reason
    # A ceiling under the gate makes the run INCONCLUSIVE for the row, not a
    # NO-GO for it: the tile arithmetic, not the GPU, is what missed.
    go, _, reason = m.go_verdict(100.0, 70.0, 0.4, 0.125)
    assert not go and "INCONCLUSIVE" in reason, reason
    go, _, reason = m.go_verdict(100.0, 70.0, 0.4, 0.25)
    assert go, reason


def test_bench_reuses_the_measured_indexer_transient_constant():
    sys.path.insert(0, str(FABLE))
    import micro_two_stream_prefill as m  # noqa: E402
    from mtplx.memory_plan import QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM

    assert (
        m.QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM
        == QSA_INDEXER_TRANSIENT_BYTES_PER_ELEM
    )


# ---------------------------------------------------------------------------
# CPU wiring proof: the wavefront tile is bit-identical to the serial tile
# ---------------------------------------------------------------------------
@pytest.fixture()
def cpu_device():
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(previous)


def _tiny_args():
    from mtplx.models.qwen4_exp import TextArgs

    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=4,
        hc_lowrank=16,
    )


def test_wavefront_tile_is_bit_identical_to_the_serial_tile(cpu_device):
    """The exactness claim, on the device this worker is allowed to use.

    Same four layer bodies, same issue order, same caches -- the ONLY
    difference is that ``{n1, n2}`` are annotated onto two streams. MLX
    inserts the cross-stream dependencies; scheduling must not move a value.
    """

    mx = cpu_device
    sys.path.insert(0, str(FABLE))
    import micro_two_stream_prefill as m  # noqa: E402

    m._require_mlx()
    args = _tiny_args()
    assert args.layer_types[m.GDN_LAYER_IDX] == "linear_attention"
    assert args.layer_types[m.QSA_LAYER_IDX] == "full_attention"

    rows, context_before = 4, 8
    dtype = mx.float32
    gdn, qsa, weight_bytes = m.build_bodies(
        args, share_moe=False, seed=7, quantize=False
    )
    assert weight_bytes > 0

    from mtplx.attention_context import attention_phase

    with attention_phase("prefill"):
        gdn_state = m.gdn_state_template(args, dtype=dtype)
        qsa_state = m.prime_qsa_cache(
            qsa, args, context_before=context_before, rows=rows, dtype=dtype
        )

        widened = args.hc_count * args.hidden_size
        mx.random.seed(11)
        h_c0 = mx.random.normal((1, rows, widened)).astype(dtype) * 0.3
        h_c1 = mx.random.normal((1, rows, widened)).astype(dtype) * 0.3
        mx.eval(h_c0, h_c1)

        def caches():
            return (
                m.fresh_gdn_cache(gdn_state, context_before=context_before),
                m.fresh_qsa_cache(args, qsa_state),
            )

        serial = m.run_tile(gdn, qsa, h_c0, h_c1, caches(), streams=None)
        mx.eval(serial)

        stream_a = mx.new_stream(mx.cpu)
        stream_b = mx.new_stream(mx.cpu)
        wave = m.run_tile(
            gdn, qsa, h_c0, h_c1, caches(), streams=(stream_a, stream_b)
        )
        mx.eval(wave)

    flat_serial = [serial[0], serial[1], *serial[2]]
    flat_wave = [wave[0], wave[1], *wave[2]]
    worst, differing = m.numerics(flat_wave, flat_serial)
    assert differing == 0, f"max|diff| {worst}, {differing} differing elements"
    assert worst == 0.0


def test_independent_pair_matches_the_solo_bodies(cpu_device):
    """Arm (c) must compute the same values as the bodies run alone.

    Otherwise the concurrency probe would be measuring something other than
    the two bodies, and its overlap number would be meaningless.
    """

    mx = cpu_device
    sys.path.insert(0, str(FABLE))
    import micro_two_stream_prefill as m  # noqa: E402

    m._require_mlx()
    args = _tiny_args()
    rows, context_before = 4, 8
    dtype = mx.float32
    gdn, qsa, _ = m.build_bodies(
        args, share_moe=True, seed=3, quantize=False
    )

    from mtplx.attention_context import attention_phase

    with attention_phase("prefill"):
        gdn_state = m.gdn_state_template(args, dtype=dtype)
        qsa_state = m.prime_qsa_cache(
            qsa, args, context_before=context_before, rows=rows, dtype=dtype
        )
        widened = args.hc_count * args.hidden_size
        mx.random.seed(5)
        h_gdn = mx.random.normal((1, rows, widened)).astype(dtype) * 0.3
        h_qsa = mx.random.normal((1, rows, widened)).astype(dtype) * 0.3
        mx.eval(h_gdn, h_qsa)

        def caches():
            return (
                m.fresh_gdn_cache(gdn_state, context_before=context_before),
                m.fresh_qsa_cache(args, qsa_state),
            )

        one = m.run_independent(gdn, qsa, h_gdn, h_qsa, caches(), streams=None)
        mx.eval(one)
        two = m.run_independent(
            gdn, qsa, h_gdn, h_qsa, caches(),
            streams=(mx.new_stream(mx.cpu), mx.new_stream(mx.cpu)),
        )
        mx.eval(two)

    worst, differing = m.numerics(two, one)
    assert differing == 0, f"max|diff| {worst}, {differing} differing"


def test_shared_moe_bank_is_the_same_object_and_is_counted_once(cpu_device):
    sys.path.insert(0, str(FABLE))
    import micro_two_stream_prefill as m  # noqa: E402

    m._require_mlx()
    args = _tiny_args()
    shared_gdn, shared_qsa, shared_bytes = m.build_bodies(
        args, share_moe=True, seed=1, quantize=False
    )
    assert shared_qsa.mlp is shared_gdn.mlp
    distinct_gdn, distinct_qsa, distinct_bytes = m.build_bodies(
        args, share_moe=False, seed=1, quantize=False
    )
    assert distinct_qsa.mlp is not distinct_gdn.mlp
    assert shared_bytes < distinct_bytes
    assert (
        distinct_bytes - shared_bytes
        == m.parameter_bytes(distinct_qsa.mlp)
    )
