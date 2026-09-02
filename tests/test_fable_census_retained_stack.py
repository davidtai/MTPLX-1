"""Reducer and classifier tests for ``scripts/fable/census_retained_stack.py``.

Pure python on a synthetic JSONL: no MLX, no GPU, no census file on disk from a
real run.  The synthetic trace is built to have known answers for every number
the tool reports, so a regression in the apportionment, the window detection,
the gap split or the draft-chain verdict fails here rather than in a 2 GB trace
nobody re-reads.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fable" / "census_retained_stack.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("census_retained_stack", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


census = _load_module()


# --------------------------------------------------------------------------
# Synthetic census
# --------------------------------------------------------------------------
LM_HEAD = ("affine_qmv_wide_bfloat16_t_gs_64_b_8_batch_0", [1, 31040, 1])
DRAFT_HEAD = ("affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0", [1, 8192, 1])
ROUTED_GU = (
    "affine_gather_qmv_fast_bfloat16_t_gs_32_b_4_batch_0",
    [1, 160, 40],
)
PLE_DEQUANT = ("affine_dequantize_bfloat16_t_gs_32_b_4", [2560, 1, 1])
D3_COPY = ("gg1_copyuint32uint32", [20, 1, 1])
TARGET_GATHER = ("gather_frontbfloat16_int32_int_2", [2560, 1, 1])
BANK_COPY = ("vn_copybfloat16bfloat16", [2228736, 1, 1])
NORM = ("rmsbfloat16", [2560, 1, 1])


def _write_census(
    path: Path,
    cycles: int,
    *,
    draft_buffers_per_cycle: int,
    draft_gap_ns: int,
    bank_copies_per_cycle: int,
) -> None:
    """One synthetic trace with an exactly-known structure.

    Each cycle is: [lm_head] [D3 copy | PLE dequant | target gather, with a
    1 ms gap either side of the dequant] [routed MoE + norms] [the draft head,
    split over ``draft_buffers_per_cycle`` buffers separated by
    ``draft_gap_ns``] [``bank_copies_per_cycle`` x 17.8 MB copies].
    """

    rows: list[dict] = []
    seq = 0
    cb_index = 0
    clock = 1_000_000_000

    def emit(ops, duration_ns, gap_before_ns=0):
        nonlocal seq, cb_index, clock
        clock += gap_before_ns
        first = seq
        for kernel, grid in ops:
            rows.append(
                {
                    "schema_version": 1,
                    "record": "op",
                    "seq": seq,
                    "command_buffer_index": cb_index,
                    "kind": "compute",
                    "dispatch": "threads",
                    "kernel_name": kernel,
                    "grid": grid,
                    "threadgroup": [256, 1, 1],
                }
            )
            seq += 1
        rows.append(
            {
                "schema_version": 1,
                "record": "cb",
                "command_buffer_index": cb_index,
                "op_count": len(ops),
                "first_op_seq": first,
                "last_op_seq": seq - 1,
                "encode_start_ns": clock - 5_000,
                "encode_end_ns": clock - 1_000,
                "gpu_start_ns": clock,
                "gpu_end_ns": clock + duration_ns,
            }
        )
        clock += duration_ns
        cb_index += 1

    # a prefill-shaped preamble that must NOT land in the auto window
    emit([("steel_gemm_fused_nax_nt_bfloat16", [320, 8, 1])] * 4, 400_000)

    for _ in range(cycles):
        emit([LM_HEAD], 1_000_000)
        emit([D3_COPY], 20_000)
        emit([PLE_DEQUANT], 10_000, gap_before_ns=2_000_000)
        emit([TARGET_GATHER], 10_000, gap_before_ns=1_500_000)
        emit([ROUTED_GU, NORM, NORM], 500_000)
        for index in range(draft_buffers_per_cycle):
            emit(
                [DRAFT_HEAD],
                100_000,
                gap_before_ns=draft_gap_ns if index else 0,
            )
        for _ in range(bank_copies_per_cycle):
            emit([BANK_COPY], 50_000)

    rows.append(
        {
            "schema_version": 1,
            "record": "wait",
            "bucket": "sched_backpressure",
            "wait_ns": 1234,
            "at_ns": 1_000_000_100,
        }
    )
    rows.append(
        {
            "schema_version": 1,
            "record": "summary",
            "summary_seq": 0,
            "final": True,
            "ops_total": seq,
            "cbs_total": cb_index,
            "dropped_rows": 0,
            "complete": True,
            "buckets": {},
        }
    )
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )


@pytest.fixture()
def serial_census(tmp_path: Path) -> Path:
    path = tmp_path / "serial.jsonl"
    _write_census(
        path,
        cycles=8,
        draft_buffers_per_cycle=3,
        draft_gap_ns=300_000,
        bank_copies_per_cycle=6,
    )
    return path


@pytest.fixture()
def joint_census(tmp_path: Path) -> Path:
    path = tmp_path / "joint.jsonl"
    _write_census(
        path,
        cycles=8,
        draft_buffers_per_cycle=1,
        draft_gap_ns=0,
        bank_copies_per_cycle=0,
    )
    return path


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kernel", "grid", "family"),
    [
        (*LM_HEAD, "LM head"),
        (*DRAFT_HEAD, "Draft head"),
        (*ROUTED_GU, "MoE routed"),
        ("affine_qmv_wide_bfloat16_t_gs_32_b_4_batch_0", [1, 2060, 1], "GDN"),
        ("affine_qmv_wide_bfloat16_t_gs_32_b_4_batch_0", [1, 1536, 1], "QSA"),
        ("affine_qmv_wide_bfloat16_t_gs_64_b_8_batch_0", [1, 64, 1], "MoE router"),
        ("affine_qmv_wide_bfloat16_t_gs_64_b_8_batch_0", [1, 160, 1], "MoE shared"),
        ("custom_kernel_mtplx_hyper_v3_r1__bfloat16_t", [11264, 1, 1], "Hyper/residual"),
        ("gemv_wide_bfloat16_float32", [1, 2560, 1], "Hyper/residual"),
        ("gated_delta_step_mask", [4096, 1, 1], "GDN"),
        ("qsa_m4_fused_kv_gather", [1050624, 1, 1], "QSA"),
        ("sort_mbsort_uint32_uint32_bn512_tn4", [10, 1, 1], "Sampling/verify"),
        (*TARGET_GATHER, "Gather/scatter"),
        (*NORM, "Norm/elementwise"),
        (*D3_COPY, "Copy"),
        (*PLE_DEQUANT, "KV / dequant"),
        ("vvn_Addbfloat16", [5240320, 1, 1], "Norm/elementwise"),
    ],
)
def test_classify_families(kernel, grid, family):
    assert census.classify(kernel, grid)[0] == family


def test_classify_weight_bytes_use_the_verified_grid_decoding():
    # grid[1]*8 == N: lm_head [1,31040,1] -> 248,320 rows of 2,560 at q8/g64.
    _family, weight, _act = census.classify(*LM_HEAD)
    assert weight == pytest.approx(248320 * 2560 * 1.0625)
    # a routed gather is per-lane: 40 lanes of the packed gate+up block.
    _family, weight, _act = census.classify(*ROUTED_GU)
    assert weight == pytest.approx(40 * 1280 * 2560 * 0.625)


def test_classify_unmapped_qmv_still_carries_bytes():
    family, weight, _act = census.classify(
        "affine_qmv_wide_bfloat16_t_gs_32_b_4_batch_0", [1, 999, 1]
    )
    assert family == "Unknown qmv"
    assert weight == pytest.approx(999 * 8 * 2560 * 0.625)


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
def test_nnls_recovers_a_known_non_negative_solution():
    truth = [3.0, 0.5, 2.0]
    design = [[1.0, float(i), float(i * i % 7)] for i in range(1, 40)]
    target = [sum(c * x for c, x in zip(row, truth)) for row in design]
    assert census.nnls(design, target) == pytest.approx(truth, rel=1e-6)


def test_nnls_clamps_a_negative_coefficient_to_zero():
    design = [[1.0, float(i)] for i in range(1, 30)]
    target = [10.0 - 0.5 * i for i in range(1, 30)]
    solution = census.nnls(design, target)
    assert solution[1] == 0.0
    assert solution[0] > 0.0


# --------------------------------------------------------------------------
# Window and totals
# --------------------------------------------------------------------------
def test_auto_window_is_cycle_aligned_and_excludes_the_prefill_preamble(
    serial_census: Path,
):
    marks = census.find_cycle_marks(serial_census)
    assert len(marks) == 8
    lo, hi, cycles = census.auto_window(marks, 0)
    assert (lo, hi, cycles) == (marks[0], marks[-1], 7)
    buffers, _waits, _summary, _kernels = census.read_census(serial_census, lo, hi)
    kernels = {kernel for cb in buffers for kernel, _grid in cb.ops}
    assert not any("steel_gemm" in k for k in kernels)


def test_auto_window_refuses_a_census_with_no_decode_phase(tmp_path: Path):
    with pytest.raises(RuntimeError, match="no steady-state decode phase"):
        census.auto_window([1], 0)


def test_read_census_rejects_a_truncated_trace(tmp_path: Path, serial_census: Path):
    lines = serial_census.read_text().splitlines(keepends=True)
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("".join(line for line in lines if '"summary"' not in line))
    with pytest.raises(RuntimeError, match="no final summary"):
        census.read_census(truncated, 0, 10**9)


def test_read_census_rejects_a_trace_that_dropped_rows(
    tmp_path: Path, serial_census: Path
):
    lines = serial_census.read_text().splitlines()
    patched = []
    for line in lines:
        row = json.loads(line)
        if row.get("record") == "summary":
            row["dropped_rows"] = 17
        patched.append(json.dumps(row, separators=(",", ":")))
    path = tmp_path / "dropped.jsonl"
    path.write_text("\n".join(patched) + "\n")
    with pytest.raises(RuntimeError, match="INVALID trace"):
        census.read_census(path, 0, 10**9)


def test_union_busy_merges_overlapping_buffers():
    def cb(index, start, end):
        return census.CommandBuffer(index, start, end, start - 1, start, index, index, ())

    assert census.union_busy_ns([cb(0, 0, 100), cb(1, 50, 200), cb(2, 300, 400)]) == 300


def test_totals_match_the_synthetic_construction(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    assert report["window"]["cycles"] == 7
    # per cycle: 1 lm_head + 1 copy + 1 dequant + 1 gather + 3 MoE/norm
    #            + 3 draft + 6 bank copies = 16 dispatches in 14 buffers
    assert report["totals"]["dispatches_per_cycle"] == pytest.approx(16.0)
    assert report["totals"]["command_buffers_per_cycle"] == pytest.approx(14.0)


# --------------------------------------------------------------------------
# Idle-gap map
# --------------------------------------------------------------------------
def test_idle_gaps_split_host_late_from_driver():
    # the next buffer finished encoding 1 us before it ran, so 1 us of the
    # 10 us gap is driver latency and 9 us is the host being late.
    previous = census.CommandBuffer(0, 0, 1_000, 0, 0, 0, 0, (NORM,))
    following = census.CommandBuffer(1, 11_000, 12_000, 9_000, 10_000, 1, 1, (NORM,))
    (gap,) = census.idle_gaps([previous, following])
    assert gap["gap_ns"] == 10_000
    assert gap["host_late_ns"] == 9_000
    assert gap["driver_ns"] == 1_000


def test_gaps_below_the_floor_are_not_reported():
    previous = census.CommandBuffer(0, 0, 1_000, 0, 0, 0, 0, (NORM,))
    following = census.CommandBuffer(1, 9_000, 10_000, 0, 0, 1, 1, (NORM,))
    assert census.idle_gaps([previous, following]) == []


def test_ple_boundary_is_detected_and_sized(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    ple = report["idle"]["ple_boundary"]
    # 2.0 ms + 1.5 ms per cycle, on 7 cycles
    assert ple["count"] == 14
    assert ple["ms_per_cycle"] == pytest.approx(3.5, rel=1e-6)
    assert ple["share_of_idle"] > 0.5


def test_is_ple_boundary_only_matches_the_two_handoff_transitions():
    assert census.is_ple_boundary("gg1_copyuint32uint32", PLE_DEQUANT[0])
    assert census.is_ple_boundary(PLE_DEQUANT[0], "gather_frontbfloat16_int32_int_2")
    assert not census.is_ple_boundary("rmsbfloat16", "gather_frontbfloat16_int32_int_2")
    assert not census.is_ple_boundary(
        "affine_dequantize_bfloat16_t_gs_64_b_8", "gather_frontbfloat16_int32_int_2"
    )


def test_first_sync_gap_is_the_first_gap_after_the_cycle_marker(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    first = report["idle"]["first_sync_gap"]
    assert first["cycles_with_a_gap"] == 7
    assert first["mean_ms"] == pytest.approx(2.0, rel=1e-6)


def test_cycle_of_places_a_seq_in_the_right_cycle():
    marks = [10, 20, 30]
    assert census.cycle_of(5, marks) == -1
    assert census.cycle_of(10, marks) == 0
    assert census.cycle_of(15, marks) == 0
    assert census.cycle_of(25, marks) == 1
    assert census.cycle_of(999, marks) == 2


# --------------------------------------------------------------------------
# The three open questions
# --------------------------------------------------------------------------
def test_draft_chain_reads_a_serial_loop_as_serial(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    draft = report["draft_chain"]
    assert draft["draft_head_dispatches_per_cycle"] == pytest.approx(3.0)
    assert draft["command_buffers_per_cycle_mean"] == pytest.approx(3.0)
    assert draft["interior_idle_gaps_per_cycle"] == pytest.approx(2.0)
    assert draft["verdict"].startswith("SERIAL per-depth loop WITH a host sync")


def test_draft_chain_reads_a_joint_graph_as_joint(joint_census: Path):
    report = census.reduce_census(
        joint_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    draft = report["draft_chain"]
    assert draft["command_buffers_per_cycle_mean"] == pytest.approx(1.0)
    assert draft["interior_idle_gaps_per_cycle"] == 0.0
    assert draft["verdict"].startswith("JOINT compiled graph")


def test_bank_copies_are_counted_and_sized_at_17_8_MB(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    banks = report["bank_copies"]
    assert banks["total_per_cycle"] == pytest.approx(6.0)
    (row,) = banks["rows"]
    assert row["MB"] == pytest.approx(17.83, abs=0.01)


def test_bank_copies_report_absence_when_the_lane_is_fixed_capacity(
    joint_census: Path,
):
    report = census.reduce_census(
        joint_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    assert report["bank_copies"]["rows"] == []


def test_copy_bytes_uses_four_elements_per_thread_for_vector_kernels():
    # G-opdiet E3 anchor: vn_copybfloat16bfloat16[2228736] = 8,914,944 bf16.
    assert census.copy_bytes(*BANK_COPY) == pytest.approx(2228736 * 4 * 2)
    # a plain general copy is one element per thread
    assert census.copy_bytes("gg1_copybfloat16bfloat16", [3276800, 1, 1]) == pytest.approx(
        3276800 * 2
    )
    assert census.copy_bytes("gg1_copyuint32uint32", [1000, 1, 1]) == pytest.approx(4000)


# --------------------------------------------------------------------------
# Attribution and the diff
# --------------------------------------------------------------------------
def test_attribution_conserves_measured_time(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    attributed = report["attribution"]["total_ns"]
    busy = report["totals"]["gpu_busy_s"] * 1e9
    # the synthetic buffers never overlap, so attributed time == busy time
    assert attributed == pytest.approx(busy, rel=1e-9)


def test_attribution_shares_sum_to_one(serial_census: Path):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    assert sum(r["share"] for r in report["attribution"]["families"]) == pytest.approx(1.0)


def test_diff_lists_every_reference_family_even_when_absent(joint_census: Path):
    report = census.reduce_census(
        joint_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    families = {row["family"] for row in report["diff_vs_reference"]}
    assert set(census.REFERENCE["families"]) <= families
    absent = next(r for r in report["diff_vs_reference"] if r["family"] == "GDN")
    assert absent["ms_per_cycle"] is None
    assert absent["ref_ms_per_cycle"] == pytest.approx(5.157)


def test_manual_window_reproduces_the_published_window_shape(serial_census: Path):
    marks = census.find_cycle_marks(serial_census)
    report = census.reduce_census(
        serial_census, lo=marks[1], hi=marks[-1], top=5, bank_copy_min_mb=8.0
    )
    assert report["window"]["mode"] == "manual"
    assert report["window"]["cycles"] == 6


def test_print_report_runs_on_a_reduced_census(serial_census: Path, capsys):
    report = census.reduce_census(
        serial_census, lo=None, hi=None, top=5, bank_copy_min_mb=8.0
    )
    census.print_report(report)
    out = capsys.readouterr().out
    assert "diff vs D" in out
    assert "draft-chain anatomy" in out
    assert "VERDICT" in out


# --------------------------------------------------------------------------
# Capture plumbing (no GPU, no exec)
# --------------------------------------------------------------------------
def test_guarded_command_names_the_guard_and_the_timeouts(tmp_path: Path):
    argv = [
        "command",
        "--census-out",
        str(tmp_path / "census-9001.jsonl"),
        "--label",
        "w58",
    ]
    args = census.build_parser().parse_args(argv)
    args.sequence = 9001
    args.receipt_path = tmp_path / "receipt.json"
    line = census.guarded_command(args)
    assert str(census.RUN_GUARDED) in line
    assert line[line.index("--lock-timeout-seconds") + 1] == "3600"
    assert line[line.index("--child-timeout-seconds") + 1] == "3600"
    assert "run" in line


def test_build_environment_puts_the_profiler_overlay_first(tmp_path: Path):
    args = census.build_parser().parse_args(
        ["run", "--census-out", str(tmp_path / "c.jsonl")]
    )
    environment = census.build_environment(args)
    assert environment[census.CENSUS_ENV] == str(tmp_path / "c.jsonl")
    assert environment["PYTHONPATH"].split(":")[0] == str(census.PROFILER_OVERLAY)
    assert environment["PYTHONPATH"].split(":")[1] == str(census.ROOT)


def test_build_driver_argv_carries_the_retained_control_and_the_prewarm():
    args = census.build_parser().parse_args(
        ["run", "--census-out", "/tmp/does-not-exist.jsonl"]
    )
    args.sequence = 1
    args.receipt_path = Path("/tmp/receipt.json")
    argv = census.build_driver_argv(args)
    assert "--prewarm-ngram-table" in argv
    assert "--m4-stage3" in argv
    assert "--qsa-fused-kv-gather" in argv
    assert "--full-frspec" in argv
    assert "--compiled-mtp-prepare" in argv
    assert "--require-compiled-verify" in argv
    assert argv[argv.index("--guard-mode") + 1] == "attestation"
    assert "MTPLX_QWEN4_M4_ROUTED_GLU=1" in argv
    assert "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE=1" in argv
    assert "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL=1" in argv


def test_candidate_extra_env_rides_the_driver_raw_env_passthrough():
    args = census.build_parser().parse_args(
        [
            "run",
            "--census-out",
            "/tmp/does-not-exist.jsonl",
            "--candidate-extra-env",
            "MTPLX_FABLE_COMPILED_DRAFT=1",
            "--candidate-env",
            "MTPLX_QWEN4_M4_PLE_PREFIX_REUSE=1",
        ]
    )
    args.sequence = 1
    args.receipt_path = Path("/tmp/receipt.json")
    argv = census.build_driver_argv(args)
    assert argv[argv.index("MTPLX_FABLE_COMPILED_DRAFT=1") - 1] == "--env"
    assert (
        argv[argv.index("MTPLX_QWEN4_M4_PLE_PREFIX_REUSE=1") - 1] == "--candidate-env"
    )
