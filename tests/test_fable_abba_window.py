"""Pure-Python tests for the fable ABBA harness planning and summary maths.

Nothing here imports mlx or loads a model; the module under test is import-safe
because every MLX-touching import in ``abba_window`` is deferred into ``main``.
Runs under pytest or ``python -m unittest``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import re
import statistics
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_window_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    path = ROOT / "scripts" / "fable" / "abba_window.py"
    spec = importlib.util.spec_from_file_location("_fable_abba_window", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_driver_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    path = ROOT / "scripts" / "fable" / "abba_driver.py"
    spec = importlib.util.spec_from_file_location("_fable_abba_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


window = _load_window_module()
driver = _load_driver_module()


def make_row(index, arm, seed, decode_tok_s, **overrides):
    row = {
        "index": index,
        "position_in_seed": index % 4,
        "arm": arm,
        "arm_name": window.ARM_NAMES[arm],
        "seed": seed,
        "sequence": 1000 + index,
        "decode_tok_s": decode_tok_s,
        "decode_s": 1024.0 / decode_tok_s,
        "wall_s": 2048.0 / decode_tok_s,
        "generated_tokens": 1024,
        "compiled_m4_calls": 382,
        "ms_per_compiled_window": (1024.0 / decode_tok_s) * 1000.0 / 382,
        "tokens_per_window": 1024 / 382,
        "context_copy_rounds": 10,
        "context_copy_accepted_tokens": 66,
        "context_copy_drafted_tokens": 133,
        "context_copy_active": True,
        "context_copy_cost_s": 10 * window.DEFAULT_COPY_ROUND_COST_S
        + 66 * window.DEFAULT_COPY_TOKEN_COST_S,
        "decode_s_net": (1024.0 / decode_tok_s)
        - (10 * window.DEFAULT_COPY_ROUND_COST_S
           + 66 * window.DEFAULT_COPY_TOKEN_COST_S),
        "ms_per_m4_window_net": (
            (1024.0 / decode_tok_s)
            - (10 * window.DEFAULT_COPY_ROUND_COST_S
               + 66 * window.DEFAULT_COPY_TOKEN_COST_S)
        ) * 1000.0 / 382,
        "tokens_per_m4_window": (1024 - 66) / 382,
        "accepted_by_depth": [259, 187, 120],
        "drafted_by_depth": [382, 382, 382],
        "verify_forward_s": 11.5,
        "draft_s": 1.5,
        "digest": "e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc",
        "peak_bytes": 87_393_848_312,
        "ready_c": 39.5,
        "page_cache_regime": "as-found",
        "reference_token_parity": "match",
        "ple_hot_rows": {"available": True, "hits": 10, "misses": 1},
        "per_cycle_available": False,
    }
    row.update(overrides)
    return row


class TestArmOrdering(unittest.TestCase):
    def test_known_orders(self):
        self.assertEqual(window.arm_sequence("ABBA"), ("A", "B", "B", "A"))
        self.assertEqual(window.arm_sequence("BAAB"), ("B", "A", "A", "B"))
        self.assertEqual(window.arm_sequence("AB"), ("A", "B"))

    def test_unknown_order_rejected(self):
        with self.assertRaises(ValueError):
            window.arm_sequence("AABB")

    def test_plan_runs_is_seed_major_abba(self):
        runs = window.plan_runs([11, 22], "ABBA", 500)
        self.assertEqual(len(runs), 8)
        self.assertEqual(
            [run["arm"] for run in runs],
            ["A", "B", "B", "A", "A", "B", "B", "A"],
        )
        self.assertEqual(
            [run["seed"] for run in runs], [11, 11, 11, 11, 22, 22, 22, 22]
        )
        self.assertEqual([run["index"] for run in runs], list(range(8)))
        self.assertEqual(
            [run["sequence"] for run in runs], list(range(500, 508))
        )
        self.assertEqual(
            [run["position_in_seed"] for run in runs], [0, 1, 2, 3, 0, 1, 2, 3]
        )

    def test_plan_runs_arm_names(self):
        runs = window.plan_runs([1], "AB", 0)
        self.assertEqual(
            [run["arm_name"] for run in runs], ["control", "candidate"]
        )

    def test_plan_runs_requires_distinct_seeds(self):
        with self.assertRaises(ValueError):
            window.plan_runs([7, 7], "AB", 1)

    def test_plan_runs_requires_a_seed(self):
        with self.assertRaises(ValueError):
            window.plan_runs([], "AB", 1)

    def test_default_seeds_are_the_production_triple(self):
        self.assertEqual(
            window.PRODUCTION_SEEDS, (20260829, 20260830, 20260831)
        )


class TestArgvBuilding(unittest.TestCase):
    def setUp(self):
        self.run = window.plan_runs([20260829], "ABBA", 900)[1]  # arm B, pos 1

    def test_labels_and_receipt_names_are_unique_per_run(self):
        runs = window.plan_runs([1, 2], "ABBA", 0)
        labels = {window.arm_label("p", run) for run in runs}
        names = {window.receipt_name("p", run) for run in runs}
        self.assertEqual(len(labels), len(runs))
        self.assertEqual(len(names), len(runs))

    def test_build_arm_argv_shape(self):
        argv = window.build_arm_argv(
            self.run,
            python="/py",
            driver="/d.py",
            label_prefix="fable-abba",
            receipt_dir="/out",
            common_flags=["--source", "/src"],
            arm_flags=["--m4-stage3", "--max-tokens", "1024"],
            candidate_env=["MTPLX_A=1", "MTPLX_B=2"],
            extra_env=["MLX_MAX_OPS_PER_BUFFER=8"],
        )
        self.assertEqual(argv[:2], ["/py", "/d.py"])
        self.assertIn("--guard-mode", argv)
        self.assertEqual(argv[argv.index("--guard-mode") + 1], "window")
        self.assertEqual(argv[argv.index("--seed") + 1], "20260829")
        self.assertEqual(argv[argv.index("--sequence") + 1], "901")
        self.assertEqual(
            argv[argv.index("--receipt-path") + 1],
            "/out/fable-abba-candidate-B1-s20260829-901.json",
        )
        self.assertEqual(argv.count("--candidate-env"), 2)
        self.assertEqual(argv.count("--env"), 1)
        self.assertEqual(argv[argv.index("--env") + 1], "MLX_MAX_OPS_PER_BUFFER=8")
        self.assertIn("--m4-stage3", argv)

    def test_merge_env_settings_overrides_by_key(self):
        merged = window.merge_env_settings(
            ("MTPLX_X=1", "MTPLX_Y=1"), ["MTPLX_Y=2", "MTPLX_Z=3"]
        )
        self.assertEqual(merged, ["MTPLX_X=1", "MTPLX_Y=2", "MTPLX_Z=3"])

    def test_merge_env_settings_rejects_bad_pairs(self):
        for bad in ("NOEQUALS", "=1", "KEY="):
            with self.assertRaises(ValueError):
                window.merge_env_settings((), [bad])

    def test_control_arm_matches_the_retained_paired_routed_glu_arm(self):
        self.assertEqual(
            window.CONTROL_FLAGS,
            (
                "--target-mode",
                "batched",
                "--require-compiled-verify",
                "--m4-stage3",
                "--qsa-fused-kv-gather",
                "--full-frspec",
                "--compiled-mtp-prepare",
                "--max-tokens",
                "1024",
            ),
        )
        self.assertEqual(
            sorted(window.CONTROL_CANDIDATE_ENV),
            [
                "MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE=1",
                "MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL=1",
                "MTPLX_QWEN4_M4_ROUTED_GLU=1",
            ],
        )

    def test_arm_specification_candidate_extends_control(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--candidate-env",
                "MTPLX_NEW=1",
                # argparse needs the '=' form for values that start with '--'.
                "--candidate-flag=--nax-verify",
            ]
        )
        specs = window.arm_specification(args)
        self.assertEqual(specs["A"]["flags"], list(window.CONTROL_FLAGS))
        self.assertEqual(
            specs["B"]["flags"], [*window.CONTROL_FLAGS, "--nax-verify"]
        )
        self.assertNotIn("MTPLX_NEW=1", specs["A"]["candidate_env"])
        self.assertIn("MTPLX_NEW=1", specs["B"]["candidate_env"])
        for setting in window.CONTROL_CANDIDATE_ENV:
            self.assertIn(setting, specs["B"]["candidate_env"])

    def test_control_env_applies_to_both_arms(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--control-env",
                "MTPLX_QWEN4_M4_ROUTED_GLU=0",
            ]
        )
        specs = window.arm_specification(args)
        self.assertIn("MTPLX_QWEN4_M4_ROUTED_GLU=0", specs["A"]["candidate_env"])
        self.assertIn("MTPLX_QWEN4_M4_ROUTED_GLU=0", specs["B"]["candidate_env"])

    def test_control_flag_applies_to_both_arms(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--control-flag=--nax-verify",
            ]
        )
        specs = window.arm_specification(args)
        expected = [*window.CONTROL_FLAGS, "--nax-verify"]
        self.assertEqual(specs["A"]["flags"], expected)
        self.assertEqual(specs["B"]["flags"], expected)

    def test_candidate_flag_extends_the_control_flags(self):
        """Arm B is arm A plus the candidate flags, in that order."""

        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--control-flag=--frspec-n",
                "--control-flag=32768",
                "--candidate-flag=--nax-verify",
            ]
        )
        specs = window.arm_specification(args)
        self.assertEqual(
            specs["A"]["flags"], [*window.CONTROL_FLAGS, "--frspec-n", "32768"]
        )
        self.assertEqual(
            specs["B"]["flags"],
            [*window.CONTROL_FLAGS, "--frspec-n", "32768", "--nax-verify"],
        )

    def test_control_extra_env_applies_to_both_arms(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--control-extra-env",
                "MLX_MAX_OPS_PER_BUFFER=8",
                "--candidate-extra-env",
                "MTPLX_FABLE_MOE_SORTED=1",
            ]
        )
        specs = window.arm_specification(args)
        self.assertEqual(specs["A"]["extra_env"], ["MLX_MAX_OPS_PER_BUFFER=8"])
        self.assertEqual(
            specs["B"]["extra_env"],
            ["MLX_MAX_OPS_PER_BUFFER=8", "MTPLX_FABLE_MOE_SORTED=1"],
        )

    def test_candidate_extra_env_overrides_control_without_duplicating(self):
        """The driver refuses a repeated --env key, so the merge must collapse."""

        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--control-extra-env",
                "MLX_MAX_OPS_PER_BUFFER=8",
                "--candidate-extra-env",
                "MLX_MAX_OPS_PER_BUFFER=16",
            ]
        )
        specs = window.arm_specification(args)
        self.assertEqual(specs["A"]["extra_env"], ["MLX_MAX_OPS_PER_BUFFER=8"])
        self.assertEqual(specs["B"]["extra_env"], ["MLX_MAX_OPS_PER_BUFFER=16"])

    def test_control_options_reach_both_arm_command_lines(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--control-flag=--nax-verify",
                "--control-extra-env",
                "MTPLX_FABLE_MOE_SORTED=1",
            ]
        )
        specs = window.arm_specification(args)
        runs = window.plan_runs([20260829], "AB", 1)
        for run in runs:
            argv = window.build_arm_argv(
                run,
                python="/py",
                driver="/d.py",
                label_prefix="p",
                receipt_dir="/out",
                common_flags=[],
                arm_flags=specs[run["arm"]]["flags"],
                candidate_env=specs[run["arm"]]["candidate_env"],
                extra_env=specs[run["arm"]]["extra_env"],
            )
            self.assertIn("--nax-verify", argv)
            self.assertIn("MTPLX_FABLE_MOE_SORTED=1", argv)

    def test_reserved_arm_flags_are_rejected(self):
        for reserved in ("--label", "--seed", "--receipt-path=x", "--guard-mode"):
            with self.assertRaises(ValueError):
                window.check_arm_flags([reserved])

    def test_non_reserved_arm_flags_pass(self):
        window.check_arm_flags(["--nax-verify", "--frspec-n", "32768"])

    def test_common_flags_never_disable_the_thermal_gate(self):
        args = window.build_parser().parse_args(["--sequence", "1"])
        flags = window.common_driver_flags(args)
        self.assertNotIn("--thermal-gate-max-c", flags)
        self.assertFalse(
            any("thermal" in flag and "no" in flag for flag in flags)
        )
        args = window.build_parser().parse_args(
            ["--sequence", "1", "--thermal-gate-max-c", "45"]
        )
        flags = window.common_driver_flags(args)
        self.assertEqual(
            flags[flags.index("--thermal-gate-max-c") + 1], "45"
        )


class TestExtractRunRow(unittest.TestCase):
    def _receipt(self, **row_overrides):
        row = {
            "decode_elapsed_s": 15.099174540984677,
            "decode_tok_s": 67.81827690119681,
            "wall_s": 28.893670083023608,
            "generated_tokens": 1024,
            "compiled_m4_calls": 382,
            "accepted_by_depth": [259, 187, 120],
            "drafted_by_depth": [382, 382, 382],
            "verify_forward_time_s": 11.695743768941611,
            "draft_time_s": 1.25,
            "response_token_sha256": "abc",
            "peak_memory_bytes": 87_393_848_312,
            "thermal_gate": {"ready_c": 39.57366180419922},
            "page_cache_regime": "as-found",
            "reference_token_parity": {"status": "match"},
            "ple_hot_rows": {"available": True},
            "per_cycle": {"available": False},
            "context_copy": {
                "accepted_blocks": 8,
                "accepted_tokens": 66,
                "active": True,
                "disabled_reason": None,
                "drafted_tokens": 133,
                "probes": 364,
                "rounds": 10,
                "suspensions": 1,
            },
        }
        row.update(row_overrides)
        return {"rows": [row]}

    def test_derived_window_metrics(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(self._receipt(), run)
        self.assertAlmostEqual(row["decode_tok_s"], 67.81827690119681)
        self.assertAlmostEqual(
            row["ms_per_compiled_window"], 15.099174540984677 * 1000.0 / 382
        )
        self.assertAlmostEqual(row["tokens_per_window"], 1024 / 382)
        self.assertAlmostEqual(row["ready_c"], 39.57366180419922)
        self.assertEqual(row["arm"], "A")
        self.assertEqual(row["arm_name"], "control")
        self.assertEqual(row["seed"], 20260829)

    def test_zero_compiled_calls_do_not_divide(self):
        run = window.plan_runs([1], "AB", 0)[0]
        row = window.extract_run_row(self._receipt(compiled_m4_calls=0), run)
        self.assertIsNone(row["ms_per_compiled_window"])
        self.assertIsNone(row["tokens_per_window"])

    def test_multi_row_receipt_rejected(self):
        run = window.plan_runs([1], "AB", 0)[0]
        receipt = self._receipt()
        receipt["rows"] = receipt["rows"] * 2
        with self.assertRaises(ValueError):
            window.extract_run_row(receipt, run)

    # -- context-copy corrected cycle time --------------------------------

    def test_context_copy_stats_are_surfaced(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(self._receipt(), run)
        self.assertEqual(row["context_copy_rounds"], 10)
        self.assertEqual(row["context_copy_accepted_tokens"], 66)
        self.assertEqual(row["context_copy_drafted_tokens"], 133)
        self.assertTrue(row["context_copy_active"])

    def test_net_cycle_time_removes_the_fitted_copy_budget(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(self._receipt(), run)
        expected_cost = (
            10 * window.DEFAULT_COPY_ROUND_COST_S
            + 66 * window.DEFAULT_COPY_TOKEN_COST_S
        )
        self.assertAlmostEqual(row["context_copy_cost_s"], expected_cost)
        self.assertAlmostEqual(
            row["decode_s_net"], 15.099174540984677 - expected_cost
        )
        self.assertAlmostEqual(
            row["ms_per_m4_window_net"],
            (15.099174540984677 - expected_cost) * 1000.0 / 382,
        )
        # The net cycle time is strictly cheaper than the raw one whenever a
        # copy round fired -- the copy budget only ever comes off the top.
        self.assertLess(
            row["ms_per_m4_window_net"], row["ms_per_compiled_window"]
        )

    def test_tokens_per_m4_window_excludes_copied_tokens(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(self._receipt(), run)
        self.assertAlmostEqual(row["tokens_per_m4_window"], (1024 - 66) / 382)
        self.assertAlmostEqual(row["tokens_per_window"], 1024 / 382)

    def test_cost_constants_are_overridable(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(
            self._receipt(), run, copy_round_cost_s=0.0, copy_token_cost_s=0.0
        )
        self.assertEqual(row["context_copy_cost_s"], 0.0)
        self.assertAlmostEqual(row["decode_s_net"], row["decode_s"])
        self.assertAlmostEqual(
            row["ms_per_m4_window_net"], row["ms_per_compiled_window"]
        )

    def test_missing_context_copy_block_reads_as_zero(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        receipt = self._receipt()
        del receipt["rows"][0]["context_copy"]
        row = window.extract_run_row(receipt, run)
        self.assertEqual(row["context_copy_rounds"], 0)
        self.assertEqual(row["context_copy_accepted_tokens"], 0)
        self.assertFalse(row["context_copy_active"])
        self.assertAlmostEqual(
            row["ms_per_m4_window_net"], row["ms_per_compiled_window"]
        )
        self.assertAlmostEqual(row["tokens_per_m4_window"], 1024 / 382)

    def test_zero_compiled_calls_do_not_divide_net_metrics(self):
        run = window.plan_runs([1], "AB", 0)[0]
        row = window.extract_run_row(self._receipt(compiled_m4_calls=0), run)
        self.assertIsNone(row["ms_per_m4_window_net"])
        self.assertIsNone(row["tokens_per_m4_window"])
        # The cost model itself is still recorded on the row.
        self.assertEqual(row["context_copy_rounds"], 10)


class TestProductionReceiptCostModel(unittest.TestCase):
    """Check the corrected statistic against the real w10-stack receipts.

    The receipts live outside the repo (``.benchmark-artifacts`` is not
    tracked), so this skips when they are not on the box.
    """

    RECEIPT_DIRS = (
        Path(
            "/Users/davidtai/projects/OpenSourceWTF/.worktrees/"
            "qwen38-fable-80tps/.benchmark-artifacts/fable"
        ),
        ROOT / ".benchmark-artifacts" / "fable",
    )

    def _receipt_paths(self):
        for directory in self.RECEIPT_DIRS:
            paths = sorted(directory.glob("fable-w10-stack-*.json"))
            if paths:
                return paths
        return []

    def test_real_receipts_produce_a_finite_net_cycle_time(self):
        import json

        paths = self._receipt_paths()
        if not paths:
            self.skipTest("fable-w10-stack receipts are not on this box")
        rows = []
        for index, path in enumerate(paths):
            receipt = json.loads(path.read_text())
            arm = "B" if "candidate" in path.name else "A"
            match = re.search(r"-s(\d+)-", path.name)
            assert match is not None, path.name
            seed = int(match.group(1))
            run = {
                "index": index,
                "position_in_seed": index,
                "arm": arm,
                "arm_name": window.ARM_NAMES[arm],
                "seed": seed,
                "sequence": 1788400081 + index,
            }
            row = window.extract_run_row(receipt, run)
            rows.append(row)
            source = receipt["rows"][0]["context_copy"]
            self.assertEqual(row["context_copy_rounds"], source["rounds"])
            self.assertEqual(
                row["context_copy_accepted_tokens"], source["accepted_tokens"]
            )
            # Every production run spends real time on copy rounds, so the
            # correction is never a no-op and never eats the whole window.
            self.assertGreater(row["context_copy_cost_s"], 0.0)
            self.assertLess(
                row["context_copy_cost_s"], 0.25 * row["decode_s"]
            )
            # 30-45 ms per compiled M4 window is the physical band for this
            # lane; anything outside it means the statistic is mis-wired.
            self.assertGreater(row["ms_per_m4_window_net"], 30.0)
            self.assertLess(row["ms_per_m4_window_net"], 45.0)

        summary = window.summarize(rows)
        self.assertEqual(
            summary["copy_cost_model"]["copy_round_cost_s"],
            window.DEFAULT_COPY_ROUND_COST_S,
        )
        # The corrected metric is the point: its cross-run spread must be
        # tighter than raw tok/s across the same runs.
        def spread(values):
            values = [float(v) for v in values]
            return (max(values) - min(values)) / statistics.fmean(values)

        self.assertLess(
            spread(r["ms_per_m4_window_net"] for r in rows),
            spread(r["decode_tok_s"] for r in rows),
        )


class TestSummary(unittest.TestCase):
    def _abba_rows(self):
        # One seed, ABBA: control 60 and 62, candidate 66 and 64.
        return [
            make_row(0, "A", 20260829, 60.0),
            make_row(1, "B", 20260829, 66.0),
            make_row(2, "B", 20260829, 64.0),
            make_row(3, "A", 20260829, 62.0),
        ]

    def test_arm_aggregates(self):
        summary = window.summarize(self._abba_rows())
        self.assertEqual(summary["arms"]["A"]["runs"], 2)
        self.assertEqual(summary["arms"]["B"]["runs"], 2)
        self.assertAlmostEqual(summary["arms"]["A"]["mean_decode_tok_s"], 61.0)
        self.assertAlmostEqual(summary["arms"]["B"]["mean_decode_tok_s"], 65.0)
        self.assertAlmostEqual(summary["arms"]["A"]["median_decode_tok_s"], 61.0)

    def test_per_seed_paired_delta(self):
        summary = window.summarize(self._abba_rows())
        self.assertEqual(len(summary["per_seed"]), 1)
        entry = summary["per_seed"][0]
        self.assertEqual(entry["seed"], 20260829)
        self.assertAlmostEqual(entry["delta_decode_tok_s"], 4.0)
        self.assertAlmostEqual(entry["delta_pct"], 100.0 * 4.0 / 61.0)
        self.assertTrue(entry["digests_match"])

    def test_adjacent_pairs_only_within_a_seed_and_across_arms(self):
        rows = [
            make_row(0, "A", 1, 60.0),
            make_row(1, "B", 1, 66.0),
            make_row(2, "B", 1, 64.0),
            make_row(3, "A", 1, 62.0),
            make_row(4, "A", 2, 50.0),
            make_row(5, "B", 2, 55.0),
        ]
        summary = window.summarize(rows)
        # (0,1) A->B, (1,2) same arm skipped, (2,3) B->A, (3,4) seed change
        # skipped, (4,5) A->B.
        self.assertEqual(len(summary["adjacent_pairs"]), 3)
        self.assertEqual(
            [round(entry["delta_decode_tok_s"], 6) for entry in summary["adjacent_pairs"]],
            [6.0, 2.0, 5.0],
        )

    def test_overall_deltas_and_digest_agreement(self):
        summary = window.summarize(self._abba_rows())
        overall = summary["overall"]
        self.assertAlmostEqual(overall["delta_mean_decode_tok_s"], 4.0)
        self.assertAlmostEqual(overall["delta_median_decode_tok_s"], 4.0)
        self.assertAlmostEqual(overall["paired_delta_mean_decode_tok_s"], 4.0)
        self.assertTrue(overall["all_digests_match"])

    def test_digest_mismatch_is_reported(self):
        rows = self._abba_rows()
        rows[1] = make_row(1, "B", 20260829, 66.0, digest="deadbeef")
        summary = window.summarize(rows)
        self.assertFalse(summary["overall"]["all_digests_match"])
        self.assertFalse(summary["per_seed"][0]["digests_match"])

    def test_single_arm_leaves_deltas_undefined(self):
        summary = window.summarize([make_row(0, "A", 1, 60.0)])
        self.assertEqual(summary["per_seed"], [])
        self.assertIsNone(summary["overall"]["candidate_mean_decode_tok_s"])
        self.assertIsNone(summary["overall"]["delta_mean_decode_tok_s"])

    def test_empty_summary_rejected(self):
        with self.assertRaises(ValueError):
            window.summarize([])

    # -- corrected cycle time is the PRIMARY paired statistic --------------

    def _net_rows(self):
        """One seed, ABBA, where tok/s and the net cycle time DISAGREE.

        The candidate draws twice the retrieval yield of the control, which is
        trajectory luck, not speed: on tok/s it looks faster, on the corrected
        per-M4-window cycle time it is slower.
        """

        def row(index, arm, tok_s, rounds, accepted):
            decode_s = 1024.0 / tok_s
            cost = (
                rounds * window.DEFAULT_COPY_ROUND_COST_S
                + accepted * window.DEFAULT_COPY_TOKEN_COST_S
            )
            # Tokens the copy lane emitted are tokens the M4 lane did not have
            # to produce, so a higher retrieval yield means FEWER compiled
            # windows for the same 1,024-token answer (2.49 tok/window is the
            # measured production constant).
            calls = round((1024 - accepted) / 2.49)
            return make_row(
                index,
                arm,
                20260829,
                tok_s,
                compiled_m4_calls=calls,
                ms_per_compiled_window=decode_s * 1000.0 / calls,
                tokens_per_window=1024 / calls,
                context_copy_rounds=rounds,
                context_copy_accepted_tokens=accepted,
                context_copy_cost_s=cost,
                decode_s_net=decode_s - cost,
                ms_per_m4_window_net=(decode_s - cost) * 1000.0 / calls,
                tokens_per_m4_window=(1024 - accepted) / calls,
            )

        return [
            row(0, "A", 67.0, 9, 50),
            row(1, "B", 69.0, 20, 160),
            row(2, "B", 69.0, 20, 160),
            row(3, "A", 67.0, 9, 50),
        ]

    def test_arm_aggregates_carry_the_net_cycle_time(self):
        summary = window.summarize(self._net_rows())
        for arm in ("A", "B"):
            self.assertIsNotNone(summary["arms"][arm]["mean_ms_per_m4_window_net"])
            self.assertIsNotNone(summary["arms"][arm]["mean_tokens_per_m4_window"])
        self.assertAlmostEqual(
            summary["arms"]["A"]["mean_context_copy_rounds"], 9.0
        )
        self.assertAlmostEqual(
            summary["arms"]["B"]["mean_context_copy_accepted_tokens"], 160.0
        )

    def test_net_delta_disagrees_with_tok_s_when_retrieval_yield_differs(self):
        summary = window.summarize(self._net_rows())
        entry = summary["per_seed"][0]
        # tok/s says the candidate won...
        self.assertGreater(entry["delta_decode_tok_s"], 0.0)
        # ...the corrected cycle time (a cost) says it lost.
        self.assertGreater(entry["delta_ms_per_m4_window_net"], 0.0)
        self.assertGreater(entry["delta_ms_per_m4_window_net_pct"], 0.0)
        overall = summary["overall"]
        self.assertGreater(overall["delta_mean_ms_per_m4_window_net"], 0.0)
        self.assertGreater(overall["paired_delta_mean_ms_per_m4_window_net"], 0.0)
        self.assertGreater(
            overall["adjacent_delta_mean_ms_per_m4_window_net"], 0.0
        )
        # The raw cycle time stays available alongside it.
        self.assertIsNotNone(overall["delta_mean_ms_per_compiled_window"])
        self.assertIsNotNone(entry["delta_ms_per_compiled_window"])

    def test_per_seed_reports_each_arm_retrieval_yield(self):
        entry = window.summarize(self._net_rows())["per_seed"][0]
        self.assertAlmostEqual(entry["control_mean_context_copy_rounds"], 9.0)
        self.assertAlmostEqual(entry["candidate_mean_context_copy_rounds"], 20.0)
        self.assertAlmostEqual(
            entry["control_mean_context_copy_accepted_tokens"], 50.0
        )
        self.assertAlmostEqual(
            entry["candidate_mean_context_copy_accepted_tokens"], 160.0
        )

    def test_summary_records_the_cost_model_it_used(self):
        summary = window.summarize(
            self._net_rows(), copy_round_cost_s=0.05, copy_token_cost_s=0.001
        )
        model = summary["copy_cost_model"]
        self.assertEqual(model["copy_round_cost_s"], 0.05)
        self.assertEqual(model["copy_token_cost_s"], 0.001)
        self.assertEqual(model["primary_metric"], "ms_per_m4_window_net")

    def test_net_metrics_tolerate_rows_without_them(self):
        rows = [
            make_row(0, "A", 1, 60.0, ms_per_m4_window_net=None),
            make_row(1, "B", 1, 66.0, ms_per_m4_window_net=None),
        ]
        summary = window.summarize(rows)
        self.assertIsNone(summary["arms"]["A"]["mean_ms_per_m4_window_net"])
        self.assertIsNone(
            summary["per_seed"][0]["delta_ms_per_m4_window_net"]
        )
        self.assertIsNone(
            summary["overall"]["delta_mean_ms_per_m4_window_net"]
        )
        # tok/s still resolves.
        self.assertAlmostEqual(
            summary["overall"]["delta_mean_decode_tok_s"], 6.0
        )


class TestMarkdown(unittest.TestCase):
    def test_table_has_one_row_per_run_and_all_columns(self):
        rows = [
            make_row(0, "A", 20260829, 60.0),
            make_row(1, "B", 20260829, 66.0),
        ]
        text = window.render_markdown(rows, window.summarize(rows))
        lines = text.splitlines()
        header = lines[0]
        for column in (
            "Decode tok/s",
            "Decode s",
            "ms/window",
            "ms/M4win net",
            "tok/window",
            "tok/M4win",
            "ccopy rounds",
            "ccopy accepted",
            "Accepted by depth",
            "Verify fwd s",
            "Digest",
            "Peak bytes",
            "Ready C",
            "Page cache",
        ):
            self.assertIn(column, header)
        body = [line for line in lines[2:] if line.startswith("| 0 ")]
        self.assertEqual(len(body), 1)
        self.assertIn("control (A)", body[0])
        self.assertIn("259,187,120", body[0])
        self.assertIn("as-found", body[0])
        self.assertIn("+6.000000", text)
        self.assertIn("Every arm produced the same response-token digest: yes", text)
        # The corrected statistic leads, tok/s is labelled secondary.
        self.assertIn("PRIMARY cycle-time metric: ms/M4win net", text)
        self.assertIn("PRIMARY: control ", text)
        self.assertIn("Secondary raw cycle time:", text)
        self.assertIn("Secondary throughput", text)
        self.assertIn(str(window.DEFAULT_COPY_ROUND_COST_S), text)

    def test_overridden_cost_constants_are_printed(self):
        rows = [
            make_row(0, "A", 20260829, 60.0),
            make_row(1, "B", 20260829, 66.0),
        ]
        summary = window.summarize(
            rows, copy_round_cost_s=0.05, copy_token_cost_s=0.002
        )
        text = window.render_markdown(rows, summary)
        self.assertIn("rounds*0.05", text)
        self.assertIn("accepted*0.002", text)

    def test_missing_values_render_as_na_not_a_crash(self):
        rows = [
            make_row(
                0,
                "A",
                1,
                60.0,
                ms_per_compiled_window=None,
                ms_per_m4_window_net=None,
                tokens_per_window=None,
                tokens_per_m4_window=None,
                context_copy_rounds=None,
                context_copy_accepted_tokens=None,
                ready_c=None,
                page_cache_regime=None,
            )
        ]
        text = window.render_markdown(rows, window.summarize(rows))
        self.assertIn("n/a", text)


class TestControlArmEnvironmentContract(unittest.TestCase):
    """The default control arm must survive the driver's fail-closed env check.

    Imports ``mtplx.profiles`` only, which is mlx-free.
    """

    def _driver_args(self, extra=()):
        argv = [
            "--label",
            "t",
            "--sequence",
            "1",
            "--seed",
            "20260829",
            *window.CONTROL_FLAGS,
            *extra,
        ]
        for setting in window.CONTROL_CANDIDATE_ENV:
            argv.extend(["--candidate-env", setting])
        return driver.build_parser().parse_args(argv)

    def _apply(self, args):
        from mtplx.profiles import apply_profile_env, get_profile

        overrides, candidate = driver.build_family_overrides(args)
        expected = get_profile("turbo").env_dict()
        expected.update(overrides)
        environ = {}
        # apply_profile_env announces gated env to stdout; keep tests quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            apply_profile_env(
                "turbo", environ=environ, runtime_env_overrides=overrides
            )
        observed = {key: environ.get(key) for key in expected}
        return expected, observed, overrides, candidate

    def test_control_environment_applies_without_drift(self):
        expected, observed, _, candidate = self._apply(self._driver_args())
        self.assertEqual(observed, expected)
        self.assertEqual(
            sorted(candidate), sorted(k.split("=")[0] for k in window.CONTROL_CANDIDATE_ENV)
        )

    def test_retain_events_flips_drop_events_without_drift(self):
        expected, observed, overrides, _ = self._apply(
            self._driver_args(["--retain-events"])
        )
        self.assertEqual(overrides["MTPLX_DROP_EVENTS"], "0")
        self.assertEqual(observed["MTPLX_DROP_EVENTS"], "0")
        self.assertEqual(observed, expected)

    def test_every_control_override_key_is_a_supported_runtime_override(self):
        from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS

        overrides, _ = driver.build_family_overrides(self._driver_args())
        unsupported = sorted(
            set(overrides) - set(MODEL_RUNTIME_ENV_OVERRIDE_KEYS)
        )
        self.assertEqual(unsupported, [])

    def test_thermal_gate_default_is_forty_and_cannot_be_disabled(self):
        args = self._driver_args()
        self.assertEqual(args.thermal_gate_max_c, 40.0)
        self.assertEqual(driver.DEFAULT_THERMAL_MAX_C, 40.0)
        for bad in ("0", "-1", "100", "off", "none"):
            with self.assertRaises(Exception):
                driver.thermal_gate_max_c(bad)
        self.assertNotIn(
            "--no-thermal-gate",
            {action.option_strings[0] for action in driver.build_parser()._actions if action.option_strings},
        )

    def test_default_out_dir_is_the_gitignored_fable_directory(self):
        self.assertEqual(driver.OUT_DIR, ROOT / ".benchmark-artifacts" / "fable")
        self.assertEqual(window.OUT_DIR, driver.OUT_DIR)

    def test_production_cell_constants(self):
        self.assertEqual(driver.PRODUCTION_CELL_LABEL, "coding-16k-1k-xhigh-t1")
        self.assertEqual(
            driver.EXPECTED_PROMPT,
            "b9e9acf190a37eb12bad2171dea6bbfa4e13a8f1053ce4888e5c3cb3fadfdd20",
        )
        self.assertEqual(
            driver.EXPECTED_MODEL, "29ba90f82124961d0d902a9ea9bbb1034972af2f"
        )

    def test_d3_route_is_opt_in(self):
        self.assertFalse(self._driver_args().d3_softfloat64_route)
        self.assertTrue(
            self._driver_args(["--d3-softfloat64-route"]).d3_softfloat64_route
        )


class TestDriverReceiptHelpers(unittest.TestCase):
    def test_reference_token_parity_matches_the_retained_seed(self):
        row = {
            "seed": 20260829,
            "accepted_drafts": 566,
            "drafted_tokens": 1146,
            "verify_calls": 392,
            "correction_tokens": 269,
            "bonus_tokens": 119,
            "accepted_by_depth": [259, 187, 120],
            "drafted_by_depth": [382, 382, 382],
            "response_token_sha256": (
                "e632b62c52044f544d00bed5f64350cc09806283bb19a473fe53db91157e1fdc"
            ),
        }
        parity = driver.reference_token_parity(row)
        self.assertEqual(parity["status"], "match")
        self.assertTrue(parity["match"])

        drifted = dict(row, response_token_sha256="deadbeef")
        self.assertEqual(
            driver.reference_token_parity(drifted)["status"], "drift"
        )
        self.assertEqual(
            driver.reference_token_parity(dict(row, seed=1))["status"],
            "no_reference",
        )
        self.assertEqual(
            driver.reference_token_parity(
                dict(row, accepted_by_depth=[1, 1], drafted_by_depth=[1, 1])
            )["status"],
            "depth_shape_mismatch",
        )

    def test_per_cycle_receipt_from_events(self):
        class Stats:
            events = [
                {"step": 0, "accepted": 2, "timing_s": {"draft": 0.001, "verify": 0.02}},
                {"step": 1, "accepted": 1, "timing_s": {"draft": 0.002}},
                {"step": 2, "accepted": None},
            ]

        receipt = driver.per_cycle_receipt(Stats())
        self.assertTrue(receipt["available"])
        self.assertEqual(receipt["cycles"], 3)
        self.assertEqual(receipt["step"], [0, 1, 2])
        self.assertEqual(receipt["accepted"], [2, 1, None])
        self.assertAlmostEqual(receipt["attributed_s"][0], 0.021)
        self.assertIsNone(receipt["attributed_s"][2])
        self.assertAlmostEqual(receipt["timing_totals_s"]["draft"], 0.003)

    def test_per_cycle_receipt_absent_is_marked_not_fabricated(self):
        class Stats:
            events: list = []

        receipt = driver.per_cycle_receipt(Stats())
        self.assertFalse(receipt["available"])
        self.assertIn("MTPLX_DROP_EVENTS", receipt["reason"])
        self.assertNotIn("step", receipt)

    def test_env_passthrough_rejects_the_wrong_namespace(self):
        self.assertEqual(
            driver.parse_key_values(
                ["MLX_MAX_OPS_PER_BUFFER=8"], flag="--env", require_mtplx=False
            ),
            {"MLX_MAX_OPS_PER_BUFFER": "8"},
        )
        with self.assertRaises(RuntimeError):
            driver.parse_key_values(
                ["MTPLX_X=1"], flag="--env", require_mtplx=False
            )
        with self.assertRaises(RuntimeError):
            driver.parse_key_values(
                ["X=1"], flag="--candidate-env", require_mtplx=True
            )
        with self.assertRaises(RuntimeError):
            driver.parse_key_values(
                ["MTPLX_X=1", "MTPLX_X=2"],
                flag="--candidate-env",
                require_mtplx=True,
            )


class TestOuterCommandLine(unittest.TestCase):
    def test_outer_command_names_run_guarded_and_the_plist(self):
        command = window.outer_command_line()
        self.assertIn("bench/laguna/run_guarded.py", command)
        self.assertIn("com.tea.qwen.plist", command)
        self.assertIn("--lock-timeout-seconds", command)
        self.assertIn("--child-timeout-seconds", command)
        self.assertIn(f"PYTHONPATH={window.ROOT}", command)
        self.assertIn(" -- ", command)

    def test_help_epilog_carries_the_outer_command(self):
        epilog = window.build_parser().epilog
        self.assertIn("run_guarded.py", epilog)


if __name__ == "__main__":
    unittest.main()


class PrefillOnlyTest(unittest.TestCase):
    """``--prefill-only`` / ``--max-tokens`` move the shared baseline."""

    def _specs(self, extra=()):
        args = window.build_parser().parse_args(["--sequence", "1", *extra])
        return window.arm_specification(args)

    def test_default_is_unchanged_and_equals_control_flags(self):
        specs = self._specs()
        self.assertEqual(specs["A"]["flags"], list(window.CONTROL_FLAGS))
        self.assertEqual(specs["B"]["flags"], list(window.CONTROL_FLAGS))

    def test_prefill_only_lowers_max_tokens_on_both_arms(self):
        specs = self._specs(["--prefill-only"])
        for arm in ("A", "B"):
            flags = specs[arm]["flags"]
            self.assertEqual(flags.count("--max-tokens"), 1)
            self.assertEqual(
                flags[flags.index("--max-tokens") + 1],
                str(window.PREFILL_ONLY_MAX_TOKENS),
            )

    def test_prefill_only_changes_nothing_else_about_the_control_arm(self):
        base = list(window.CONTROL_FLAGS)
        base[base.index("--max-tokens") + 1] = str(
            window.PREFILL_ONLY_MAX_TOKENS
        )
        self.assertEqual(self._specs(["--prefill-only"])["A"]["flags"], base)

    def test_explicit_max_tokens_wins_over_prefill_only(self):
        specs = self._specs(["--prefill-only", "--max-tokens", "256"])
        flags = specs["A"]["flags"]
        self.assertEqual(flags[flags.index("--max-tokens") + 1], "256")

    def test_candidate_and_control_flags_still_append_after_the_baseline(self):
        specs = self._specs(
            ["--prefill-only", "--candidate-flag=--nax-verify"]
        )
        self.assertEqual(specs["B"]["flags"][-1], "--nax-verify")
        self.assertEqual(specs["A"]["flags"], specs["B"]["flags"][:-1])

    def test_max_tokens_is_reserved_against_duplicate_arm_flags(self):
        self.assertIn("--max-tokens", window.RESERVED_ARM_FLAGS)
        with self.assertRaises(ValueError):
            window.check_arm_flags(["--max-tokens"])

    def test_resolve_max_tokens_rejects_a_non_positive_budget(self):
        with self.assertRaises(ValueError):
            window.resolve_max_tokens(0, False)

    def test_control_flags_never_duplicates_the_option(self):
        flags = window.control_flags(64)
        self.assertEqual(flags.count("--max-tokens"), 1)
        self.assertEqual(len(flags), len(window.CONTROL_FLAGS))


class TestRawEnvironmentMtplxAllowlist(unittest.TestCase):
    """MTPLX_* keys that are read straight off os.environ, not via the profile.

    ``--candidate-env`` funnels into ``mtplx.profiles.apply_profile_env``, which
    refuses any key outside MODEL_RUNTIME_ENV_OVERRIDE_KEYS; ``--env`` refuses
    MTPLX_* outright.  A knob like MTPLX_CONTEXT_COPY_K -- read by
    ``context_copy_block_k()`` with a bare ``os.environ.get`` -- was therefore
    unreachable from the harness on BOTH channels.  A named allowlist opens the
    raw channel for exactly those keys and keeps the loud refusal for the rest.
    """

    def test_allowlisted_keys_ride_the_raw_env_passthrough(self):
        parsed = driver.parse_key_values(
            ["MTPLX_CONTEXT_COPY_K=48", "MTPLX_SESSION_BANK_MAX_BYTES=4G"],
            flag="--env",
            require_mtplx=False,
        )
        self.assertEqual(
            parsed,
            {"MTPLX_CONTEXT_COPY_K": "48", "MTPLX_SESSION_BANK_MAX_BYTES": "4G"},
        )

    def test_unlisted_mtplx_keys_still_fail_loudly_on_raw_env(self):
        with self.assertRaises(RuntimeError) as caught:
            driver.parse_key_values(
                ["MTPLX_QWEN4_M4_ROUTED_GLU=1"], flag="--env", require_mtplx=False
            )
        # The message names the allowlist so the fix is obvious.
        self.assertIn("MTPLX_CONTEXT_COPY_K", str(caught.exception))
        self.assertIn("--candidate-env", str(caught.exception))

    def test_fable_diagnostic_namespace_still_rides_raw_env(self):
        self.assertEqual(
            driver.parse_key_values(
                ["MTPLX_FABLE_COMPILED_COPY_ROUND=1"],
                flag="--env",
                require_mtplx=False,
            ),
            {"MTPLX_FABLE_COMPILED_COPY_ROUND": "1"},
        )

    def test_allowlisted_keys_are_refused_on_the_override_channel(self):
        with self.assertRaises(RuntimeError) as caught:
            driver.parse_key_values(
                ["MTPLX_CONTEXT_COPY_K=48"],
                flag="--candidate-env",
                require_mtplx=True,
            )
        self.assertIn("--env", str(caught.exception))

    def test_ordinary_override_keys_are_unaffected(self):
        self.assertEqual(
            driver.parse_key_values(
                ["MTPLX_QWEN4_M4_ROUTED_GLU=1"],
                flag="--candidate-env",
                require_mtplx=True,
            ),
            {"MTPLX_QWEN4_M4_ROUTED_GLU": "1"},
        )

    def test_window_mirrors_the_driver_allowlist(self):
        """Drift here would let the window plan an arm the driver refuses."""

        self.assertEqual(window.RAW_ENV_MTPLX_KEYS, driver.RAW_ENV_MTPLX_KEYS)

    def test_window_accepts_the_block_cap_recipe_on_extra_env(self):
        window.check_env_settings(
            ["MTPLX_CONTEXT_COPY_K=48"],
            flag="--candidate-extra-env",
            mtplx=False,
        )

    def test_window_refuses_a_mis_routed_override(self):
        with self.assertRaises(ValueError) as caught:
            window.check_env_settings(
                ["MTPLX_QWEN4_M4_ROUTED_GLU=1"],
                flag="--candidate-extra-env",
                mtplx=False,
            )
        self.assertIn("--candidate-env", str(caught.exception))

    def test_window_refuses_a_raw_key_on_the_override_channel(self):
        with self.assertRaises(ValueError) as caught:
            window.check_env_settings(
                ["MTPLX_CONTEXT_COPY_K=48"], flag="--candidate-env", mtplx=True
            )
        self.assertIn("--candidate-extra-env", str(caught.exception))

    def test_window_still_requires_the_mtplx_prefix_on_the_override_channel(self):
        with self.assertRaises(ValueError):
            window.check_env_settings(
                ["MLX_MAX_OPS_PER_BUFFER=8"], flag="--candidate-env", mtplx=True
            )

    def test_arm_specification_rejects_a_mis_routed_key_before_the_gpu(self):
        args = argparse.Namespace(
            control_flag=[],
            candidate_flag=[],
            control_env=[],
            candidate_env=["MTPLX_CONTEXT_COPY_K=48"],
            control_extra_env=[],
            candidate_extra_env=[],
            max_tokens=None,
            prefill_only=False,
        )
        with self.assertRaises(ValueError):
            window.arm_specification(args)

    def test_block_cap_recipe_reaches_the_driver_command_line(self):
        """The documented recipe, end to end through the planner."""

        args = argparse.Namespace(
            control_flag=[],
            candidate_flag=[],
            control_env=[],
            candidate_env=[],
            control_extra_env=[],
            candidate_extra_env=["MTPLX_CONTEXT_COPY_K=48"],
            max_tokens=None,
            prefill_only=False,
        )
        specs = window.arm_specification(args)
        self.assertEqual(specs["A"]["extra_env"], [])
        self.assertEqual(specs["B"]["extra_env"], ["MTPLX_CONTEXT_COPY_K=48"])
        run = window.plan_runs([20260829], "AB", 900)[1]
        argv = window.build_arm_argv(
            run,
            python="py",
            driver="drv",
            label_prefix="p",
            receipt_dir="/tmp",
            common_flags=[],
            arm_flags=specs["B"]["flags"],
            candidate_env=specs["B"]["candidate_env"],
            extra_env=specs["B"]["extra_env"],
        )
        self.assertIn("--env", argv)
        self.assertEqual(argv[argv.index("--env") + 1], "MTPLX_CONTEXT_COPY_K=48")
        # ...and the driver accepts exactly that.
        self.assertEqual(
            driver.parse_key_values(
                specs["B"]["extra_env"], flag="--env", require_mtplx=False
            ),
            {"MTPLX_CONTEXT_COPY_K": "48"},
        )

    def test_both_context_copy_caps_are_allowlisted(self):
        """The documented recipe pairs them; refusing one would break it."""

        self.assertEqual(
            driver.parse_key_values(
                [
                    "MTPLX_CONTEXT_COPY_K=48",
                    "MTPLX_CONTEXT_COPY_PROBATION_K=16",
                ],
                flag="--env",
                require_mtplx=False,
            ),
            {
                "MTPLX_CONTEXT_COPY_K": "48",
                "MTPLX_CONTEXT_COPY_PROBATION_K": "16",
            },
        )

    def test_every_allowlisted_key_names_its_reader(self):
        """The receipt records where each key is read; drift would blank it."""

        self.assertEqual(
            set(driver.RAW_ENV_MTPLX_READERS), driver.RAW_ENV_MTPLX_KEYS
        )
        for key, reader in driver.RAW_ENV_MTPLX_READERS.items():
            self.assertTrue(reader, key)

    def test_compiled_copy_round_flag_rides_the_fable_namespace(self):
        """MTPLX_FABLE_COMPILED_COPY_ROUND needs no allowlist entry."""

        self.assertTrue(
            driver.is_raw_env_mtplx_key("MTPLX_FABLE_COMPILED_COPY_ROUND")
        )
        self.assertNotIn(
            "MTPLX_FABLE_COMPILED_COPY_ROUND", driver.RAW_ENV_MTPLX_KEYS
        )
