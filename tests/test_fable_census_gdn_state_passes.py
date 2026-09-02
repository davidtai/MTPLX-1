"""Pure-Python tests for the GDN state-pass census reduction.

The tool reads a dispatch-census JSONL and never touches MLX, so it is fully
exercised here against a synthetic census built to the schema
``scripts/fable/census_retained_stack.py`` documents.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "fable" / "census_gdn_state_passes.py"


def _load():
    spec = importlib.util.spec_from_file_location("_census_gdn_passes", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load()

STEP = (
    "custom_kernel_gated_delta_step__bfloat16_t_float_128_128_16_48_"
    "bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_float_int32_ts"
)
LM_HEAD = "affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0"
FILLER = "rmsbfloat16"


def _write_census(tmp_path: Path, cycles, *, busy_ms) -> tuple[Path, int, int]:
    """One JSONL where each cycle is (step dispatches, filler dispatches)."""

    rows = []
    seq = 0
    cb = 0
    first_seq = None
    for steps, filler, busy in zip(*cycles, busy_ms):
        rows.append(
            {
                "record": "op",
                "seq": seq,
                "command_buffer_index": cb,
                "kernel_name": LM_HEAD,
                "grid": [1, 31040, 1],
            }
        )
        cycle_first = seq
        if first_seq is None:
            first_seq = seq
        seq += 1
        for _ in range(steps):
            rows.append(
                {
                    "record": "op",
                    "seq": seq,
                    "command_buffer_index": cb,
                    "kernel_name": STEP,
                    "grid": [32, 128, 48],
                }
            )
            seq += 1
        for _ in range(filler):
            rows.append(
                {
                    "record": "op",
                    "seq": seq,
                    "command_buffer_index": cb,
                    "kernel_name": FILLER,
                    "grid": [1, 1, 1],
                }
            )
            seq += 1
        rows.append(
            {
                "record": "cb",
                "command_buffer_index": cb,
                "first_op_seq": cycle_first,
                "last_op_seq": seq - 1,
                "gpu_start_ns": cb * 1_000_000_000,
                "gpu_end_ns": cb * 1_000_000_000 + int(busy * 1e6),
            }
        )
        cb += 1
    path = tmp_path / "census.jsonl"
    # The instrumented MLX build writes compact JSON (no spaces); the reducer
    # prefilters on the literal ``"record":"op"``, so the fixture must match.
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"
    )
    return path, 0, seq


@pytest.fixture
def census(tmp_path):
    # 6 cycles: [pad] all-accept, partial, all-accept, partial, copy-round,
    # [pad].  The first and last are clipped by the reducer.
    steps = [36, 36, 72, 36, 72, 180, 36]
    filler = [10, 10, 40, 10, 40, 400, 10]
    busy = [30.0, 30.0, 31.0, 30.0, 31.0, 60.0, 30.0]
    return _write_census(tmp_path, (steps, filler), busy_ms=busy)


def test_state_bytes_per_layer_pass():
    assert tool.STATE_BYTES == 3_145_728
    assert 2 * tool.STATE_BYTES == 6_291_456


def test_mtplx_fused_step_is_not_counted_as_a_library_pass():
    # mtplx_gdn_step_fused is MTPLX's own S=1 kernel: a different pass with a
    # different byte profile, and counting it would inflate the state total.
    assert tool._is_step(STEP) is True
    assert tool._is_step("custom_kernel_mtplx_gdn_step_fused_bf16_f32") is False
    assert tool._is_step("custom_kernel_mtplx_gdn_conv_norm_rows_") is False


def test_reduce_finds_every_cycle_and_drops_the_clipped_ends(census):
    path, lo, hi = census
    data = tool.reduce_census(str(path), lo, hi)
    # 7 lm_head markers -> 7 cycles, first and last dropped.
    assert len(data["cycles"]) == 5


def test_copy_round_cycles_are_excluded_not_charged_to_the_replay(census):
    path, lo, hi = census
    data = tool.reduce_census(str(path), lo, hi)
    without, with_replay, excluded = tool.split_by_replay(data)
    assert len(without) == 2
    assert len(with_replay) == 2
    assert len(excluded) == 1
    assert all(data["steps"][c] == 36 for c in without)
    assert all(data["steps"][c] == 72 for c in with_replay)
    assert all(data["steps"][c] == 180 for c in excluded)


def test_replay_delta_is_the_difference_between_the_two_cycle_kinds(census):
    path, lo, hi = census
    data = tool.reduce_census(str(path), lo, hi)
    without, with_replay, _excluded = tool.split_by_replay(data)
    a = tool.summarise(data, without)
    b = tool.summarise(data, with_replay)
    # each partial cycle carries 36 extra steps and 30 extra filler ops
    assert b["dispatches_per_cycle"] - a["dispatches_per_cycle"] == pytest.approx(66.0)
    assert b["gpu_busy_ms_mean"] - a["gpu_busy_ms_mean"] == pytest.approx(1.0)
    assert b["step_dispatches_per_cycle"] == pytest.approx(72.0)
    assert a["step_dispatches_per_cycle"] == pytest.approx(36.0)


def test_kernel_delta_reports_the_step_kernel(census):
    path, lo, hi = census
    data = tool.reduce_census(str(path), lo, hi)
    without, with_replay, _ = tool.split_by_replay(data)
    rows = dict(tool.kernel_delta(data, without, with_replay))
    key = next(name for name in rows if "gated_delta_step" in name)
    assert rows[key] == pytest.approx(36.0)


def test_cli_writes_a_report(census, tmp_path, capsys):
    path, lo, hi = census
    out = tmp_path / "report.json"
    assert (
        tool.main([str(path), "--lo", str(lo), "--hi", str(hi), "--out", str(out)])
        == 0
    )
    report = json.loads(out.read_text())
    assert report["window"]["cycles"] == 4
    assert report["window"]["excluded_copy_round_cycles"] == 1
    assert report["verify_pass_mb_per_cycle"] == pytest.approx(226.49, abs=0.01)
    assert report["replay_cost"]["p_partial"] == pytest.approx(0.5)
    assert report["replay_cost"]["gpu_busy_ms_amortised"] == pytest.approx(0.5)
    assert "gated_delta_step" in capsys.readouterr().out
