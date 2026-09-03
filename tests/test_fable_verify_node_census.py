"""Tests for ``scripts/fable/census_verify_nodes.py``.

Pure python on a synthetic JSONL: no MLX, no GPU, no 500 MB census on disk.
The fixture is a miniature of the production body — a fixed compiled suffix
preceded by a per-cycle-variable draft lane — so every number the tool reports
has a known answer, and the two properties the whole inventory rests on are
falsifiable here:

1. the body is found as the **longest common (kernel, grid) suffix** across
   cycles, not by a hard-coded offset;
2. the fusable groups are **disjoint**, so the ranking cannot double-count a
   dispatch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fable" / "census_verify_nodes.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("census_verify_nodes", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = _load_module()


# --------------------------------------------------------------------------
# A miniature verify body: 2 GDN blocks, 1 QSA block, 3 MoE blocks, 1 head
# --------------------------------------------------------------------------
HC = (
    ("custom_kernel_mtplx_qwen4_m4_hc_norm_bd378635__bfloat16_t_256", (4096, 1, 1)),
    ("custom_kernel_mtplx_qwen4_m4_hc_down__bfloat16_t_4_256_4_1", (20736, 1, 1)),
    ("custom_kernel_mtplx_qwen4_m4_hc_up__bfloat16_t_4_8", (40960, 1, 1)),
)
GDN_BLOCK = HC + (
    ("affine_qmv_wide_bfloat16_t_gs_32_b_4_nv_4_kl_8_batch_0", (1, 2060, 1)),
    ("g2_copybfloat16bfloat16", (10240, 4, 1)),
    ("custom_kernel_mtplx_gdn_conv_norm_rows__bfloat16_t_4", (10240, 1, 1)),
    ("gn1_Sigmoidbfloat16bfloat16", (48, 4, 1)),
    ("Ef4IAsTypeAFf4IExpEGf4INegativeFHf4IBroadcastG_VV_11160318154034397263_s", (48, 4, 1)),
    ("custom_kernel_gated_delta_step__bfloat16_t_float_128_128", (32, 128, 48)),
    ("rmsbfloat16", (6144, 1, 1)),
    ("Cf4IAsTypeADf4ISigmoidCEf4IAsTypeBFf4IMultiplyDE_VV_11160318154034397263_s", (6144, 4, 1)),
    ("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 320, 1)),
)
MOE_BLOCK = HC + (
    ("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 64, 1)),
    ("block_softmax_precise_bfloat16", (512, 1, 1)),
    ("carg_block_sort_bfloat16_uint32_bn128_tn4", (1, 4, 1)),
    ("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 160, 1)),
    ("gather_axisbfloat16uint32_intcnc", (1, 10, 4)),
    ("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 1, 1)),
    ("row_reduce_small_1_reduce_sumbfloat16", (4, 1, 1)),
    ("CV2ISigmoidADV2IMultiplyACEV2OMultiplyDB_VV_11160318154034397263_s", (640, 4, 1)),
    ("CV2IBroadcastBDV2ODivideAC_VV_11160318154034397263_s", (10, 4, 1)),
    ("v_Sigmoidbfloat16bfloat16", (4, 1, 1)),
    ("g2_copyuint32uint32", (10, 4, 1)),
    ("custom_kernel_mtplx_qwen4_m4_paired_routed_glu_bfloat16_t", (5120, 40, 1)),
    ("custom_kernel_mtplx_qwen4_m4_routed_down_reduce_bfloat16_t", (20480, 4, 1)),
)
QSA_BLOCK = HC + (
    ("compute_dynamic_offset_int32", (1, 1, 1)),
    ("gg1_dynamic_copybfloat16bfloat16", (128, 1, 1)),
    ("ss_Minimumint32", (1, 1, 1)),
    ("v_Sinfloat32float32", (32, 1, 1)),
    ("v_Cosfloat32float32", (32, 1, 1)),
    ("Ef4IAsTypeAFf4IBroadcastBGf4IMultiplyEF_VV_11160318154034397263_s", (32, 1, 1)),
    ("Ef4IBroadcastBFf4IMultiplyAEGV2INegativeC_VV_11160318154034397263_s", (32, 1, 1)),
    ("gg1_copybfloat16bfloat16", (32, 1, 1)),
    ("gg1_copybfloat16bfloat16", (64, 1, 1)),
    ("arangeint32", (4, 1, 1)),
    ("Ci4IBroadcastBDb1OLessEqualAC_VV_11160318154034397263_s", (4, 4, 1)),
    ("g2_copybool_bool_", (4, 2048, 1)),
    ("affine_qmv_wide_bfloat16_t_gs_32_b_4_nv_4_kl_8_batch_0", (1, 1536, 1)),
    ("sort_mbsort_float32_uint32_bn512_tn4", (3, 4, 1)),
    ("custom_kernel_mtplx_qwen4_qsa_m4_fused_kv_gather_c17408", (1050624, 1, 1)),
    ("gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc1_axpby0", (129, 1, 96)),
    ("block_softmax_float32", (52224, 1, 1)),
    ("affine_qmv_wide_bfloat16_t_gs_32_b_4_nv_4_kl_8_batch_0", (1, 320, 1)),
)
HEAD_BLOCK = HC[:2] + (
    ("custom_kernel_mtplx_qwen4_m4_hc_up__bfloat16_t_4_8", (40960, 1, 1)),
    ("ss_Addint32", (1, 1, 1)),
    ("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 31040, 1)),
)
PROLOGUE = (
    ("gather_frontbfloat16_int32_int_2", (20, 4, 1)),
    ("affine_dequantize_bfloat16_t_gs_64_b_8", (10240, 1, 1)),
)

BODY = (
    PROLOGUE
    + GDN_BLOCK
    + MOE_BLOCK
    + QSA_BLOCK
    + MOE_BLOCK
    + GDN_BLOCK
    + MOE_BLOCK
    + HEAD_BLOCK
)

#: The draft/sampling lane in front of the body: shapes move with accept
#: length, which is exactly why the common suffix ends where the body starts.
def _draft_lane(cycle: int):
    width = 100 + cycle
    return (
        ("affine_qmv_fast_bfloat16_t_gs_64_b_8_batch_0", (1, 8192, 1)),
        ("partition_mbsort_float32_uint32_bn512_tn4", (1, width, 1)),
        ("v_Expfloat32float32", (width, 1, 1)),
        ("gg1_copyuint32uint32", (width, 1, 1)),
    )


def _write_census(path: Path, cycles: int = 4) -> Path:
    seq = 0
    cb = 0
    rows = []
    for cycle in range(cycles):
        for name, grid in _draft_lane(cycle) + BODY:
            rows.append(
                {
                    "schema_version": 1,
                    "record": "op",
                    "seq": seq,
                    "command_buffer_index": cb,
                    "kind": "compute",
                    "dispatch": "threads",
                    "kernel_name": name,
                    "setBytes_calls": 2,
                    "setBytes_total_bytes": 16,
                    "buffer_binds": 2,
                    "grid": list(grid),
                    "threadgroup": [1024, 1, 1],
                }
            )
            seq += 1
            if seq % 17 == 0:
                cb += 1
    rows.append(
        {
            "schema_version": 1,
            "record": "summary",
            "final": True,
            "ops_total": seq,
            "dropped_rows": 0,
            "complete": True,
        }
    )
    # compact separators, exactly as the instrumented MLX build writes them
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"
    )
    return path


@pytest.fixture(scope="module")
def census(tmp_path_factory) -> Path:
    return _write_census(tmp_path_factory.mktemp("w69") / "mini-census.jsonl")


# --------------------------------------------------------------------------
# Parsing and body detection
# --------------------------------------------------------------------------
def test_iter_ops_parses_every_op_row_and_no_others(census: Path):
    ops = list(nodes.iter_ops(census))
    assert len(ops) == 4 * (len(BODY) + 4)
    assert ops[0].seq == 0
    assert ops[0].grid == (1, 8192, 1)
    assert all(len(op.grid) == 3 for op in ops)


def test_regex_and_json_fallback_agree(census: Path):
    fast = list(nodes.iter_ops(census))
    slow = []
    for line in census.read_text().splitlines():
        row = json.loads(line)
        if row.get("record") == "op":
            slow.append(
                nodes.Op(
                    row["seq"],
                    row["command_buffer_index"],
                    row["kernel_name"],
                    tuple(row["grid"]),
                )
            )
    assert fast == slow


def test_body_length_is_measured_not_assumed(census: Path):
    tails = nodes.collect_cycle_tails(census, window=200)
    # one tail per cycle mark, minus the cycles whose window is not yet full
    assert len(tails) == 3
    assert nodes.measure_body_length(tails) == len(BODY)


def test_streaming_scan_agrees_with_the_all_tails_reduction(census: Path):
    """``scan_body`` is O(window); it must give the reducer's answer exactly."""

    tails = nodes.collect_cycle_tails(census, window=200)
    reference, length, cycles = nodes.scan_body(census, window=200)
    assert length == nodes.measure_body_length(tails)
    assert cycles == len(tails)
    assert reference == tails[0]


def test_common_suffix_that_fills_the_window_is_an_error(census: Path):
    tails = nodes.collect_cycle_tails(census, window=len(BODY))
    with pytest.raises(RuntimeError, match="filled the whole"):
        nodes.measure_body_length(tails)


def test_one_cycle_cannot_measure_a_suffix(census: Path):
    with pytest.raises(RuntimeError, match="need at least 2"):
        nodes.measure_body_length(nodes.collect_cycle_tails(census, window=200)[:1])


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def test_segmentation_recovers_the_authored_blocks(census: Path):
    tails = nodes.collect_cycle_tails(census, window=200)
    body = tails[0][-nodes.measure_body_length(tails) :]
    blocks = nodes.segment(body)
    kinds = [block.kind for block in blocks]
    assert kinds == [
        "prologue",
        "GDN",
        "MoE",
        "QSA",
        "MoE",
        "GDN",
        "MoE",
        "HEAD",
    ]
    assert sum(block.size for block in blocks) == len(BODY)
    sizes = {block.kind: block.size for block in blocks}
    assert sizes["GDN"] == len(GDN_BLOCK)
    assert sizes["MoE"] == len(MOE_BLOCK)
    assert sizes["QSA"] == len(QSA_BLOCK)
    assert sizes["HEAD"] == len(HEAD_BLOCK)
    assert sizes["prologue"] == len(PROLOGUE)


def test_block_opener_must_match_the_anchor_census(census: Path):
    tails = nodes.collect_cycle_tails(census, window=200)
    body = list(tails[0][-nodes.measure_body_length(tails) :])
    # drop the first hyper-connection opener: the cut count no longer matches
    # the number of attention/MoE anchors, so segmentation must refuse rather
    # than silently fold two blocks into one.
    first_opener = next(i for i, op in enumerate(body) if "hc_norm" in op.kernel)
    stripped = body[:first_opener] + body[first_opener + 1 :]
    with pytest.raises(RuntimeError, match="no block opener matched"):
        nodes.segment(stripped)


def test_ple_block_is_labelled_separately():
    ops = [
        nodes.Op(0, 0, name, grid)
        for name, grid in MOE_BLOCK
        + (("implicit_gemm_conv_2d_bfloat16_bm32_bn8", (1, 1, 10240)),)
    ]
    assert nodes.block_kind(ops) == "MoE+PLE"


# --------------------------------------------------------------------------
# Op classes
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kernel,expected",
    [
        ("custom_kernel_mtplx_qwen4_m4_hc_up__bfloat16_t", "custom kernel"),
        ("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", "matmul/qmv"),
        ("affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_32_b_4", "matmul/qmv"),
        ("gemv_wide_bfloat16_nv4_kl32_nc0_axpby0", "matmul/gemm"),
        ("steel_gemm_fused_nax_nt_float32_float32", "matmul/gemm"),
        ("implicit_gemm_conv_2d_bfloat16_bm32", "matmul/gemm"),
        ("affine_dequantize_bfloat16_t_gs_64_b_8", "dequantize"),
        ("gg1_dynamic_copybfloat16bfloat16", "cache append/slice"),
        ("compute_dynamic_offset_int32", "cache offset"),
        ("gg2_copybfloat16bfloat16", "copy/layout"),
        ("gather_frontbfloat16_int32_int_2", "gather/scatter"),
        ("gather_axisbfloat16uint32_intcnc", "gather/scatter"),
        ("carg_block_sort_bfloat16_uint32_bn128_tn4", "sort/top-k"),
        ("merge_mbsort_float32_uint32_bn512_tn4", "sort/top-k"),
        ("block_softmax_precise_bfloat16", "softmax"),
        ("rmsbfloat16", "norm"),
        ("row_reduce_small_1_reduce_sumbfloat16", "reduce"),
        ("arangeint32", "index/arange"),
        ("v_Sigmoidbfloat16bfloat16", "elementwise"),
        ("ss_Addint32", "elementwise"),
        (
            "CV2IBroadcastBDV2ODivideAC_VV_V2V2_11160318154034397263_strided_2",
            "elementwise (fused)",
        ),
    ],
)
def test_op_class(kernel: str, expected: str):
    assert nodes.op_class(kernel) == expected


def test_every_class_is_declared():
    for kernel, _grid in BODY:
        assert nodes.op_class(kernel) in nodes.OP_CLASSES


# --------------------------------------------------------------------------
# Groups and pricing
# --------------------------------------------------------------------------
def test_groups_are_disjoint_on_the_fixture(census: Path):
    report, _body = nodes.build_report(census, window=200, body_len=None)
    # build_report raises if two groups claim the same dispatch; reaching here
    # is the assertion.  Check the counts it did produce.
    keys = {g["key"]: g for g in report["groups"]}
    # one QSA block: 2 sin/cos + 2 fused rotates + 2 concat copies
    assert keys["qsa_rope"]["matched"] == 6
    assert keys["qsa_rope"]["removable"] == 5  # 1/6 of the group is re-emitted
    # 3 HC kernels x 7 blocks = 21 matched, 2/3 removable
    assert keys["hc_triple"]["matched"] == 21
    assert keys["hc_triple"]["removable"] == 14
    # 3 MoE blocks x 8 routing-head dispatches, of which the kernel re-emits 2
    assert keys["moe_router_glue"]["matched"] == 24
    assert keys["moe_router_glue"]["removable"] == 18
    assert keys["moe_expert_id_copies"]["matched"] == 3
    # 2 GDN blocks x 5 gate-glue dispatches
    assert keys["gdn_gate_glue"]["matched"] == 10


def test_group_price_is_the_documented_arithmetic(census: Path):
    report, _body = nodes.build_report(census, window=200, body_len=None)
    for group in report["groups"]:
        assert group["host_ms"] == pytest.approx(
            group["removable"] * nodes.EXPOSED_ENCODE_US / 1000.0
        )
        assert group["gpu_ms"] == pytest.approx(
            group["dependent"] * nodes.DEPENDENT_LAUNCH_US / 1000.0, rel=1e-6, abs=1e-9
        )
        assert group["total_ms"] == pytest.approx(group["host_ms"] + group["gpu_ms"])
        assert group["tok_s"] == pytest.approx(group["total_ms"] * nodes.TOKS_PER_MS)


def test_a_new_overlapping_group_is_rejected(census: Path, monkeypatch):
    clash = nodes.Group(
        key="clash",
        title="deliberately overlaps qsa_rope",
        blocks=("QSA",),
        patterns=(("v_Sinfloat32float32", None),),
        chain_fraction=1.0,
        mechanism="-",
        exactness="-",
    )
    monkeypatch.setattr(nodes, "GROUPS", nodes.GROUPS + (clash,))
    with pytest.raises(RuntimeError, match="both claim dispatch"):
        nodes.build_report(census, window=200, body_len=None)


def test_report_totals_add_up(census: Path):
    report, _body = nodes.build_report(census, window=200, body_len=None)
    assert report["body_len_used"] == len(BODY)
    assert sum(n for _cls, n in report["by_class"]) == len(BODY)
    assert (
        sum(info["dispatches"] for info in report["blocks"].values()) == len(BODY)
    )
    assert report["opener_strategy"] == "hc_m4"


def test_stock_hc_opener_is_found_when_the_fused_read_is_absent():
    """The control stack's eager hyper-connection read segments too."""

    stock_hc = (
        ("rmsbfloat16", (10240, 1, 1)),
        ("CV2IBroadcastBDV2OMultiplyAC_VV_11160318154034397263_s", (10240, 4, 1)),
        ("gemv_wide_bfloat16_nv4_kl32_nc0_axpby0", (1, 80, 1)),
        ("gemv_wide_bfloat16_nv4_kl32_nc0_axpby0", (1, 2560, 1)),
    )
    body = [
        nodes.Op(i, 0, name, grid)
        for i, (name, grid) in enumerate(
            stock_hc
            + (("affine_qmv_wide_bfloat16_t_gs_32_b_4_nv_4_kl_8_batch_0", (1, 2060, 1)),)
            + stock_hc
            + (("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 64, 1)),)
            + stock_hc
            + (("affine_qmv_wide_bfloat16_t_gs_64_b_8_nv_4_kl_8_batch_0", (1, 31040, 1)),)
        )
    ]
    cuts, strategy = nodes.find_block_cuts(body)
    assert strategy == "stock_hc"
    assert cuts == [0, 5, 10]
    assert [block.kind for block in nodes.segment(body)] == ["GDN", "MoE", "HEAD"]


def test_neighbours_names_the_adjacent_custom_kernel(census: Path):
    tails = nodes.collect_cycle_tails(census, window=200)
    body = tails[0][-nodes.measure_body_length(tails) :]
    indices = [i for i, op in enumerate(body) if op.kernel == "v_Sinfloat32float32"]
    names = dict(nodes.neighbours(body, indices))
    assert "hc_up" in names
