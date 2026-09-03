"""Pure-Python tests for the fable ABBA harness planning and summary maths.

Nothing here imports mlx or loads a model; the module under test is import-safe
because every MLX-touching import in ``abba_window`` is deferred into ``main``.
Runs under pytest or ``python -m unittest``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
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
    accepted_by_depth = overrides.pop("accepted_by_depth", [259, 187, 120])
    drafted_by_depth = overrides.pop("drafted_by_depth", [382, 382, 382])
    # ``None`` = the receipt never recorded the accept-probability sums, which
    # is every receipt written before the driver carried them.  Pass a list to
    # model a receipt that did.
    accept_sums = overrides.pop("accept_probability_sum_by_depth", None)
    accept_rate = window.per_depth_rates(accepted_by_depth, drafted_by_depth)
    accept_prob = window.per_depth_rates(accept_sums, drafted_by_depth)
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
        "accepted_by_depth": accepted_by_depth,
        "drafted_by_depth": drafted_by_depth,
        "accept_probability_sum_by_depth": accept_sums,
        "accept_probability_recorded": accept_sums is not None,
        "accept_rate_by_depth": accept_rate,
        "mean_accept_probability_by_depth": accept_prob,
        "conditional_accept_rate_by_depth": window.conditional_depth_ratios(
            accept_rate
        ),
        "conditional_accept_probability_by_depth": (
            window.conditional_depth_ratios(accept_prob)
        ),
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


    # -- acceptance: the realised coin and its expectation ----------------

    def test_realised_accept_rate_and_conditional_ratios(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(self._receipt(), run)
        self.assertEqual(row["accepted_by_depth"], [259, 187, 120])
        self.assertEqual(row["drafted_by_depth"], [382, 382, 382])
        for observed, expected in zip(
            row["accept_rate_by_depth"], [259 / 382, 187 / 382, 120 / 382]
        ):
            self.assertAlmostEqual(observed, expected)
        # d2|d1 first, then d3|d2.
        for observed, expected in zip(
            row["conditional_accept_rate_by_depth"], [187 / 259, 120 / 187]
        ):
            self.assertAlmostEqual(observed, expected)

    def test_accept_probability_sums_are_carried_and_divided_by_drafted(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        receipt = self._receipt(
            accept_probability_sum_by_depth=[286.5, 210.1, 133.4]
        )
        row = window.extract_run_row(receipt, run)
        self.assertTrue(row["accept_probability_recorded"])
        self.assertEqual(
            row["accept_probability_sum_by_depth"], [286.5, 210.1, 133.4]
        )
        for observed, expected in zip(
            row["mean_accept_probability_by_depth"],
            [286.5 / 382, 210.1 / 382, 133.4 / 382],
        ):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(
            row["conditional_accept_probability_by_depth"],
            [210.1 / 286.5, 133.4 / 210.1],
        ):
            self.assertAlmostEqual(observed, expected)

    def test_an_old_receipt_has_no_expected_acceptance_not_a_zero_one(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        receipt = self._receipt()
        self.assertNotIn(
            "accept_probability_sum_by_depth", receipt["rows"][0]
        )
        row = window.extract_run_row(receipt, run)
        self.assertFalse(row["accept_probability_recorded"])
        self.assertIsNone(row["accept_probability_sum_by_depth"])
        self.assertIsNone(row["mean_accept_probability_by_depth"])
        self.assertIsNone(row["conditional_accept_probability_by_depth"])
        # The realised coin is unaffected -- an old receipt still reports it.
        self.assertIsNotNone(row["accept_rate_by_depth"])
        self.assertIsNotNone(row["conditional_accept_rate_by_depth"])

    def test_conditional_ratios_divide_rates_not_raw_counts(self):
        """Depths drafted a different number of times still divide correctly.

        Counts alone would say d2|d1 = 120/200 = 0.6; the depths were drafted
        400 and 200 times, so the acceptance actually ROSE from 0.50 to 0.60.
        """

        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(
            self._receipt(
                accepted_by_depth=[200, 120, 40],
                drafted_by_depth=[400, 200, 100],
                accept_probability_sum_by_depth=[180.0, 108.0, 45.0],
            ),
            run,
        )
        self.assertAlmostEqual(row["accept_rate_by_depth"][0], 0.50)
        self.assertAlmostEqual(row["accept_rate_by_depth"][1], 0.60)
        self.assertAlmostEqual(
            row["conditional_accept_rate_by_depth"][0], 0.60 / 0.50
        )
        self.assertAlmostEqual(
            row["conditional_accept_rate_by_depth"][1], 0.40 / 0.60
        )
        self.assertAlmostEqual(
            row["conditional_accept_probability_by_depth"][0], 0.54 / 0.45
        )

    def test_a_depth_that_drafted_nothing_reads_none_not_zero(self):
        run = window.plan_runs([20260829], "ABBA", 900)[0]
        row = window.extract_run_row(
            self._receipt(
                accepted_by_depth=[259, 187, 0],
                drafted_by_depth=[382, 382, 0],
                accept_probability_sum_by_depth=[286.5, 210.1, 0.0],
            ),
            run,
        )
        self.assertIsNone(row["accept_rate_by_depth"][2])
        self.assertIsNone(row["mean_accept_probability_by_depth"][2])
        self.assertIsNone(row["conditional_accept_rate_by_depth"][1])
        self.assertIsNone(row["conditional_accept_probability_by_depth"][1])


class TestAcceptanceHelpers(unittest.TestCase):
    def test_per_depth_rates_without_a_numerator_is_none(self):
        self.assertIsNone(window.per_depth_rates(None, [1, 2, 3]))
        self.assertIsNone(window.per_depth_rates([1, 2, 3], None))

    def test_per_depth_rates_truncate_to_the_shorter_vector(self):
        self.assertEqual(window.per_depth_rates([1, 2, 3], [2, 4]), [0.5, 0.5])

    def test_conditional_ratios_of_a_single_depth_are_empty(self):
        self.assertEqual(window.conditional_depth_ratios([0.7]), [])
        self.assertIsNone(window.conditional_depth_ratios(None))

    def test_conditional_ratios_skip_an_undefined_earlier_depth(self):
        self.assertEqual(
            window.conditional_depth_ratios([None, 0.5, 0.25]), [None, 0.5]
        )


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
                # The receipt's OWN sequence: `extract_run_row` refuses a
                # receipt whose sequence is not this arm's, which is what
                # stops a stale attempt being read as this run's evidence.
                "sequence": int(
                    receipt["rows"][0].get(
                        "sequence", receipt.get("sequence", 1788400081 + index)
                    )
                ),
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


class TestAcceptanceSummary(unittest.TestCase):
    """The acceptance headline and the acceptance term behind it."""

    def _rows(self, **overrides):
        # Control accepts more per depth than the candidate; both receipts
        # carry the accept-probability sums, so both readings resolve.
        control = dict(
            accepted_by_depth=[259, 187, 120],
            drafted_by_depth=[382, 382, 382],
            accept_probability_sum_by_depth=[286.5, 210.1, 133.4],
        )
        candidate = dict(
            accepted_by_depth=[249, 177, 110],
            drafted_by_depth=[382, 382, 382],
            accept_probability_sum_by_depth=[276.5, 200.1, 123.4],
        )
        control.update(overrides)
        return [
            make_row(0, "A", 20260829, 60.0, **control),
            make_row(1, "B", 20260829, 66.0, **candidate),
            make_row(2, "B", 20260830, 64.0, **candidate),
            make_row(3, "A", 20260830, 62.0, **control),
        ]

    def test_arm_aggregates_carry_both_acceptance_readings(self):
        arms = window.summarize(self._rows())["arms"]
        self.assertTrue(arms["A"]["accept_probability_recorded"])
        for observed, expected in zip(
            arms["A"]["mean_accept_rate_by_depth"],
            [259 / 382, 187 / 382, 120 / 382],
        ):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(
            arms["B"]["mean_accept_probability_by_depth"],
            [276.5 / 382, 200.1 / 382, 123.4 / 382],
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(
            arms["A"]["conditional_accept_rate_by_depth"][0], 187 / 259
        )
        self.assertAlmostEqual(
            arms["A"]["conditional_accept_probability_by_depth"][0],
            210.1 / 286.5,
        )
        self.assertEqual(arms["A"]["mean_accepted_by_depth"], [259, 187, 120])

    def test_per_seed_and_overall_carry_the_acceptance_deltas(self):
        summary = window.summarize(self._rows())
        entry = summary["per_seed"][0]
        self.assertAlmostEqual(
            entry["control_mean_accept_rate_by_depth"][0], 259 / 382
        )
        self.assertAlmostEqual(
            entry["candidate_mean_accept_probability_by_depth"][0], 276.5 / 382
        )
        self.assertAlmostEqual(
            entry["delta_mean_accept_probability_by_depth"][0],
            (276.5 - 286.5) / 382,
        )
        overall = summary["overall"]
        self.assertTrue(overall["accept_probability_recorded"])
        self.assertAlmostEqual(
            overall["delta_mean_accept_rate_by_depth"][0], (249 - 259) / 382
        )
        self.assertAlmostEqual(
            overall["candidate_conditional_accept_probability_by_depth"][0],
            200.1 / 276.5,
        )

    def test_rows_without_the_sums_leave_the_expected_reading_unrecorded(self):
        rows = [
            make_row(0, "A", 20260829, 60.0),
            make_row(1, "B", 20260829, 66.0),
        ]
        summary = window.summarize(rows)
        self.assertFalse(summary["overall"]["accept_probability_recorded"])
        self.assertIsNone(
            summary["overall"]["control_mean_accept_probability_by_depth"]
        )
        self.assertIsNone(
            summary["overall"]["delta_mean_accept_probability_by_depth"]
        )
        self.assertIsNone(
            summary["per_seed"][0][
                "candidate_mean_accept_probability_by_depth"
            ]
        )
        # The realised coin still aggregates.
        self.assertIsNotNone(
            summary["overall"]["control_mean_accept_rate_by_depth"]
        )

    def test_a_half_upgraded_window_is_not_silently_averaged_away(self):
        """One arm from the new driver, one from the old.

        The expected reading exists only where it was recorded; the window
        still says, at the top level, that it was not recorded everywhere.
        """

        rows = [
            make_row(
                0,
                "A",
                20260829,
                60.0,
                accept_probability_sum_by_depth=[286.5, 210.1, 133.4],
            ),
            make_row(1, "B", 20260829, 66.0),
        ]
        summary = window.summarize(rows)
        self.assertTrue(summary["arms"]["A"]["accept_probability_recorded"])
        self.assertFalse(summary["arms"]["B"]["accept_probability_recorded"])
        self.assertFalse(summary["overall"]["accept_probability_recorded"])
        self.assertIsNotNone(
            summary["overall"]["control_mean_accept_probability_by_depth"]
        )
        self.assertIsNone(
            summary["overall"]["candidate_mean_accept_probability_by_depth"]
        )


class TestAcceptanceHeadline(unittest.TestCase):
    """``tok/M4win`` gets the same paired treatment as the other metrics."""

    def _rows(self):
        # Candidate emits more tokens per compiled M4 window than the control
        # at the same 1,024-token answer: a real acceptance win.
        def row(index, arm, seed, tok_s, calls):
            decode_s = 1024.0 / tok_s
            cost = (
                10 * window.DEFAULT_COPY_ROUND_COST_S
                + 66 * window.DEFAULT_COPY_TOKEN_COST_S
            )
            return make_row(
                index,
                arm,
                seed,
                tok_s,
                compiled_m4_calls=calls,
                ms_per_compiled_window=decode_s * 1000.0 / calls,
                tokens_per_window=1024 / calls,
                context_copy_cost_s=cost,
                decode_s_net=decode_s - cost,
                ms_per_m4_window_net=(decode_s - cost) * 1000.0 / calls,
                tokens_per_m4_window=(1024 - 66) / calls,
            )

        return [
            row(0, "A", 20260829, 60.0, 400),
            row(1, "B", 20260829, 66.0, 380),
            row(2, "B", 20260830, 64.0, 380),
            row(3, "A", 20260830, 62.0, 400),
        ]

    def test_per_seed_entry_carries_the_acceptance_headline(self):
        entry = window.summarize(self._rows())["per_seed"][0]
        self.assertAlmostEqual(
            entry["control_mean_tokens_per_m4_window"], 958 / 400
        )
        self.assertAlmostEqual(
            entry["candidate_mean_tokens_per_m4_window"], 958 / 380
        )
        self.assertAlmostEqual(
            entry["delta_tokens_per_m4_window"], 958 / 380 - 958 / 400
        )
        self.assertAlmostEqual(
            entry["delta_tokens_per_m4_window_pct"],
            100.0 * (958 / 380 - 958 / 400) / (958 / 400),
        )

    def test_overall_gets_the_same_statistics_as_the_other_metrics(self):
        overall = window.summarize(self._rows())["overall"]
        expected = 958 / 380 - 958 / 400
        for key in (
            "delta_mean_tokens_per_m4_window",
            "delta_median_tokens_per_m4_window",
            "paired_delta_mean_tokens_per_m4_window",
            "paired_delta_median_tokens_per_m4_window",
            "adjacent_delta_mean_tokens_per_m4_window",
        ):
            self.assertAlmostEqual(overall[key], expected, msg=key)
        self.assertAlmostEqual(
            overall["delta_mean_tokens_per_m4_window_pct"],
            100.0 * expected / (958 / 400),
        )
        self.assertAlmostEqual(
            overall["paired_delta_mean_tokens_per_m4_window_pct"],
            100.0 * expected / (958 / 400),
        )
        self.assertAlmostEqual(
            overall["control_median_tokens_per_m4_window"], 958 / 400
        )

    def test_adjacent_pairs_carry_the_acceptance_delta(self):
        adjacent = window.summarize(self._rows())["adjacent_pairs"]
        self.assertTrue(adjacent)
        for entry in adjacent:
            self.assertAlmostEqual(
                entry["delta_tokens_per_m4_window"], 958 / 380 - 958 / 400
            )

    def test_rows_without_the_acceptance_headline_stay_none(self):
        rows = [
            make_row(0, "A", 1, 60.0, tokens_per_m4_window=None),
            make_row(1, "B", 1, 66.0, tokens_per_m4_window=None),
        ]
        summary = window.summarize(rows)
        self.assertIsNone(
            summary["overall"]["delta_mean_tokens_per_m4_window"]
        )
        self.assertIsNone(
            summary["per_seed"][0]["delta_tokens_per_m4_window"]
        )
        self.assertAlmostEqual(
            summary["overall"]["delta_mean_decode_tok_s"], 6.0
        )


class TestHeadlineAgreement(unittest.TestCase):
    """Which arm each of the three headlines favours."""

    def test_all_three_favour_the_candidate(self):
        agreement = window.headline_agreement(
            {
                "delta_mean_ms_per_m4_window_net": -0.42,
                "delta_mean_ms_per_m4_window_net_pct": -1.2,
                "delta_mean_tokens_per_m4_window": 0.03,
                "delta_mean_tokens_per_m4_window_pct": 1.1,
                "delta_mean_decode_tok_s": 0.9,
                "delta_mean_pct": 1.4,
            }
        )
        self.assertEqual(agreement["cost"]["favours"], "candidate")
        self.assertEqual(agreement["acceptance"]["favours"], "candidate")
        self.assertEqual(agreement["throughput"]["favours"], "candidate")
        self.assertTrue(agreement["agree"])
        self.assertFalse(agreement["throughput_disagrees_with_m4_headlines"])

    def test_tok_s_can_point_the_other_way_from_both_m4_headlines(self):
        agreement = window.headline_agreement(
            {
                "delta_mean_ms_per_m4_window_net": -0.42,
                "delta_mean_ms_per_m4_window_net_pct": -1.2,
                "delta_mean_tokens_per_m4_window": 0.03,
                "delta_mean_tokens_per_m4_window_pct": 1.1,
                "delta_mean_decode_tok_s": -0.24,
                "delta_mean_pct": -0.3,
            }
        )
        self.assertEqual(agreement["throughput"]["favours"], "control")
        self.assertFalse(agreement["agree"])
        self.assertTrue(agreement["m4_headlines_agree"])
        self.assertTrue(agreement["throughput_disagrees_with_m4_headlines"])

    def test_cost_and_acceptance_can_split(self):
        agreement = window.headline_agreement(
            {
                "delta_mean_ms_per_m4_window_net": -1.89,
                "delta_mean_ms_per_m4_window_net_pct": -5.3,
                "delta_mean_tokens_per_m4_window": -0.09,
                "delta_mean_tokens_per_m4_window_pct": -3.3,
                "delta_mean_decode_tok_s": -0.24,
                "delta_mean_pct": -0.3,
            }
        )
        self.assertEqual(agreement["cost"]["favours"], "candidate")
        self.assertEqual(agreement["acceptance"]["favours"], "control")
        self.assertFalse(agreement["agree"])
        self.assertFalse(agreement["m4_headlines_agree"])
        self.assertFalse(agreement["throughput_disagrees_with_m4_headlines"])

    def test_a_missing_or_zero_headline_never_manufactures_a_verdict(self):
        agreement = window.headline_agreement(
            {
                "delta_mean_ms_per_m4_window_net": None,
                "delta_mean_tokens_per_m4_window": 0.0,
                "delta_mean_decode_tok_s": 0.9,
                "delta_mean_pct": 1.4,
            }
        )
        self.assertIsNone(agreement["cost"]["favours"])
        self.assertEqual(agreement["acceptance"]["favours"], "tie")
        self.assertEqual(agreement["throughput"]["favours"], "candidate")
        self.assertTrue(agreement["agree"])

    def test_the_summary_carries_the_verdict(self):
        rows = [
            make_row(0, "A", 20260829, 60.0),
            make_row(1, "B", 20260829, 66.0),
        ]
        summary = window.summarize(rows)
        self.assertIn("headline_agreement", summary)
        self.assertEqual(
            summary["headline_agreement"]["throughput"]["favours"], "candidate"
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
            "Seq",
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
            # Renamed from "Digest": the column now shows
            # `output_ids_sha256`, the digest over the raw uint32 generated
            # ids, not the older comma-joined `response_token_sha256`.
            "Output ids sha256",
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


class TestMarkdownAcceptance(unittest.TestCase):
    """The acceptance columns, the two headlines, and the disagreement note."""

    def _agreeing_rows(self):
        """Candidate wins all three headlines: cheaper, more accepted, faster."""

        def row(index, arm, seed, tok_s, calls, sums):
            decode_s = 1024.0 / tok_s
            cost = (
                10 * window.DEFAULT_COPY_ROUND_COST_S
                + 66 * window.DEFAULT_COPY_TOKEN_COST_S
            )
            return make_row(
                index,
                arm,
                seed,
                tok_s,
                compiled_m4_calls=calls,
                ms_per_compiled_window=decode_s * 1000.0 / calls,
                tokens_per_window=1024 / calls,
                context_copy_cost_s=cost,
                decode_s_net=decode_s - cost,
                ms_per_m4_window_net=(decode_s - cost) * 1000.0 / calls,
                tokens_per_m4_window=(1024 - 66) / calls,
                accept_probability_sum_by_depth=sums,
            )

        return [
            row(0, "A", 20260829, 60.0, 400, [286.5, 210.1, 133.4]),
            row(1, "B", 20260829, 66.0, 380, [296.5, 220.1, 143.4]),
        ]

    def _disagreeing_rows(self):
        """tok/s favours the candidate; both M4 headlines favour the control.

        The candidate drew twice the retrieval yield, which is trajectory luck:
        the copy lane emitted tokens the compiled windows did not have to.
        """

        def row(index, arm, tok_s, rounds, accepted):
            decode_s = 1024.0 / tok_s
            cost = (
                rounds * window.DEFAULT_COPY_ROUND_COST_S
                + accepted * window.DEFAULT_COPY_TOKEN_COST_S
            )
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

        return [row(0, "A", 67.0, 9, 50), row(1, "B", 69.0, 20, 160)]

    def test_per_run_table_carries_both_acceptance_readings(self):
        rows = [
            make_row(
                0,
                "A",
                20260829,
                60.0,
                accept_probability_sum_by_depth=[286.5, 210.1, 133.4],
            )
        ]
        text = window.render_markdown(rows, window.summarize(rows))
        header = text.splitlines()[0]
        for column in (
            "Accepted by depth",
            "Accept rate by depth",
            "Mean accept prob by depth",
            "Cond accept realised",
            "Cond accept expected",
        ):
            self.assertIn(column, header)
        body = [
            line for line in text.splitlines()[2:] if line.startswith("| 0 ")
        ]
        self.assertEqual(len(body), 1)
        # realised rates, expected probabilities, and both conditionals.
        self.assertIn("0.6780,0.4895,0.3141", body[0])
        self.assertIn("0.7500,0.5500,0.3492", body[0])
        self.assertIn("0.7220,0.6417", body[0])
        self.assertIn("0.7333,0.6349", body[0])

    def test_an_old_row_renders_the_expected_columns_as_na(self):
        rows = [make_row(0, "A", 20260829, 60.0)]
        text = window.render_markdown(rows, window.summarize(rows))
        body = [
            line for line in text.splitlines()[2:] if line.startswith("| 0 ")
        ][0]
        # Realised acceptance still renders; the expected reading is n/a, not
        # a fabricated 0.0000.
        self.assertIn("0.6780,0.4895,0.3141", body)
        self.assertIn("| n/a |", body)
        self.assertIn("Acceptance by depth: control realised", text)
        self.assertIn("(expected n/a)", text)

    def test_paired_table_gains_the_tok_per_m4win_group(self):
        rows = self._agreeing_rows()
        text = window.render_markdown(rows, window.summarize(rows))
        paired_header = next(
            line
            for line in text.splitlines()
            if line.startswith("| Seed | Control ms/M4win net")
        )
        for column in (
            "Control ms/M4win net",
            "Candidate ms/M4win net",
            "Control tok/M4win",
            "Candidate tok/M4win",
            "Delta tok/M4win",
            "Control tok/s",
            "Candidate tok/s",
            "Delta tok/s",
            "ccopy rounds A/B",
            "Digests match",
        ):
            self.assertIn(column, paired_header)
        self.assertEqual(paired_header.count("Delta %"), 3)
        seed_row = next(
            line for line in text.splitlines() if line.startswith("| 20260829 |")
        )
        cells = [cell.strip() for cell in seed_row.strip("|").split("|")]
        self.assertEqual(len(cells), len(paired_header.strip("|").split("|")))
        # Columns 5-8 are the acceptance group: control, candidate, delta, %.
        self.assertAlmostEqual(float(cells[5]), 958 / 400, places=4)
        self.assertAlmostEqual(float(cells[6]), 958 / 380, places=4)
        self.assertAlmostEqual(
            float(cells[7]), 958 / 380 - 958 / 400, places=4
        )
        # The digest column stays last: scripts read it off the end of the row.
        self.assertEqual(cells[-1], "yes")

    def test_primary_line_states_all_three_headlines(self):
        rows = self._agreeing_rows()
        text = window.render_markdown(rows, window.summarize(rows))
        primary = next(
            line for line in text.splitlines() if line.startswith("PRIMARY: ")
        )
        self.assertIn("ms/M4win net", primary)
        self.assertIn("ACCEPTANCE:", primary)
        self.assertIn("tok/M4win", primary)
        self.assertIn("THROUGHPUT:", primary)
        self.assertIn("tok/s", primary)
        self.assertIn("PRIMARY acceptance metric: tok/M4win", text)

    def test_agreeing_headlines_print_no_disagreement_note(self):
        rows = self._agreeing_rows()
        summary = window.summarize(rows)
        self.assertTrue(summary["headline_agreement"]["agree"])
        text = window.render_markdown(rows, summary)
        self.assertNotIn("HEADLINES DISAGREE", text)

    def test_disagreeing_headlines_print_one_note_naming_each(self):
        rows = self._disagreeing_rows()
        summary = window.summarize(rows)
        self.assertFalse(summary["headline_agreement"]["agree"])
        self.assertTrue(
            summary["headline_agreement"][
                "throughput_disagrees_with_m4_headlines"
            ]
        )
        text = window.render_markdown(rows, summary)
        notes = [
            line
            for line in text.splitlines()
            if line.startswith("HEADLINES DISAGREE")
        ]
        self.assertEqual(len(notes), 1)
        note = notes[0]
        self.assertIn("ms/M4win net favours control", note)
        self.assertIn("tok/M4win favours control", note)
        self.assertIn("tok/s favours candidate", note)
        self.assertIn("retrieval yield", note)

    def test_a_cost_acceptance_split_says_so_instead(self):
        # Cheaper per window but accepting less per window: the two M4
        # headlines split, so the note must not blame the retrieval yield.
        rows = [
            make_row(
                0,
                "A",
                20260829,
                60.0,
                ms_per_m4_window_net=35.5,
                tokens_per_m4_window=2.67,
            ),
            make_row(
                1,
                "B",
                20260829,
                60.0,
                ms_per_m4_window_net=33.6,
                tokens_per_m4_window=2.58,
            ),
        ]
        summary = window.summarize(rows)
        self.assertFalse(summary["headline_agreement"]["m4_headlines_agree"])
        text = window.render_markdown(rows, summary)
        note = next(
            line
            for line in text.splitlines()
            if line.startswith("HEADLINES DISAGREE")
        )
        self.assertIn("ms/M4win net favours candidate", note)
        self.assertIn("tok/M4win favours control", note)
        self.assertIn("opposite directions", note)
        self.assertNotIn("retrieval yield", note)

    def test_acceptance_line_reports_both_readings_per_arm(self):
        rows = self._agreeing_rows()
        text = window.render_markdown(rows, window.summarize(rows))
        line = next(
            l
            for l in text.splitlines()
            if l.startswith("Acceptance by depth: ")
        )
        self.assertIn("control realised 0.6780,0.4895,0.3141", line)
        self.assertIn("expected 0.7500,0.5500,0.3492", line)
        self.assertIn("conditional d2|d1,d3|d2 realised", line)

    def test_report_scraper_regex_still_finds_every_seed(self):
        r"""``scratchpad/abba_report.py`` reads the seed rows off this table.

        Its fallback digest check is
        ``re.findall(r"\| (\d{8}) \|.*\| (yes|no|NO) \|\s*$", text, re.M)``:
        the seed must stay the first column and the digest verdict the last.
        """

        rows = self._agreeing_rows()
        text = window.render_markdown(rows, window.summarize(rows))
        found = re.findall(
            r"\| (\d{8}) \|.*\| (yes|no|NO) \|\s*$", text, re.M
        )
        self.assertEqual(found, [("20260829", "yes")])


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

    def _stats_output(self, **stats_overrides):
        """The smallest ``output`` ``stats_receipt`` will accept.

        Every unset counter reads 0.0 through ``__getattr__``, so this stays a
        two-field test of the acceptance carry rather than a transcription of
        the whole stats dataclass.
        """

        class Stats:
            graphbank: dict = {}
            draft_core: dict = {}
            online_correction_cache: dict = {}
            context_copy_disabled_reason = None
            accepted_by_depth = [259, 187, 120]
            drafted_by_depth = [382, 382, 382]

            def __init__(self, **overrides):
                for key, value in overrides.items():
                    setattr(self, key, value)

            def __getattr__(self, name):
                return 0.0

        class Output:
            tokens = [11, 22, 33]

            def __init__(self, stats):
                self.stats = stats

        return Output(Stats(**stats_overrides))

    def test_stats_receipt_carries_the_accept_probability_sums(self):
        receipt = driver.stats_receipt(
            self._stats_output(
                accept_probability_sum_by_depth=[286.5, 210.1, 133.4]
            ),
            "arm-A1",
            1788400081,
            20260829,
            28.9,
        )
        self.assertEqual(
            receipt["accept_probability_sum_by_depth"], [286.5, 210.1, 133.4]
        )
        # The realised coin and its denominator stay on the receipt too --
        # the expected reading is sum/drafted, so it needs both.
        self.assertEqual(receipt["accepted_by_depth"], [259, 187, 120])
        self.assertEqual(receipt["drafted_by_depth"], [382, 382, 382])
        # JSON-safe: this is what the driver writes to disk.
        json.loads(json.dumps(receipt["accept_probability_sum_by_depth"]))

    def test_a_runtime_without_the_sums_omits_the_field_entirely(self):
        class Stats:
            graphbank: dict = {}
            draft_core: dict = {}
            online_correction_cache: dict = {}
            context_copy_disabled_reason = None
            accepted_by_depth = [259, 187, 120]
            drafted_by_depth = [382, 382, 382]
            accept_probability_sum_by_depth = None

            def __getattr__(self, name):
                return 0.0

        class Output:
            tokens = [11, 22, 33]
            stats = Stats()

        receipt = driver.stats_receipt(
            Output(), "arm-A1", 1788400081, 20260829, 28.9
        )
        # Absent, never zero-filled: the window reads a missing key as "not
        # observed" and a zero list as "observed to be zero".
        self.assertNotIn("accept_probability_sum_by_depth", receipt)

    def test_the_driver_receipt_feeds_the_window_expected_columns(self):
        """End to end: what the driver writes is what the window divides.

        The two halves of this change are in different files; this is the
        test that fails if the field is renamed on one side only.
        """

        run = window.plan_runs([20260829], "AB", 1788400081)[0]
        stats_row = driver.stats_receipt(
            self._stats_output(
                accept_probability_sum_by_depth=[286.5, 210.1, 133.4],
                decode_elapsed_s=15.1,
                generated_tokens=1024,
            ),
            "arm-A1",
            run["sequence"],
            run["seed"],
            28.9,
        )
        receipt = {
            "rows": [
                dict(
                    stats_row,
                    compiled_m4_calls=382,
                    peak_memory_bytes=87_393_848_312,
                    thermal_gate={"ready_c": 39.5},
                    page_cache_regime="as-found",
                    reference_token_parity={"status": "match"},
                    ple_hot_rows={"available": True},
                    per_cycle={"available": False},
                )
            ]
        }
        row = window.extract_run_row(receipt, run)
        self.assertTrue(row["accept_probability_recorded"])
        self.assertAlmostEqual(
            row["mean_accept_probability_by_depth"][0], 286.5 / 382
        )
        self.assertAlmostEqual(
            row["conditional_accept_probability_by_depth"][0], 210.1 / 286.5
        )
        summary = window.summarize([row])
        self.assertTrue(summary["arms"]["A"]["accept_probability_recorded"])
        text = window.render_markdown([row], summary)
        self.assertIn("0.7500,0.5500,0.3492", text)

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


class WarmGraphTest(unittest.TestCase):
    """``--prefill-only`` pays the cold first prefill chunk unmeasured.

    2026-09-01 w22: chunk 1 of a fresh process is bimodal (~1.9 s / ~4.4 s)
    on control and candidate alike, in lockstep with the throughput of the
    driver's own 29.8 GiB ``--prewarm-ngram-table`` read.  A prefill-only arm
    measures exactly one prefill, so that term has to move out of it.
    """

    def _flags(self, extra=()):
        args = window.build_parser().parse_args(["--sequence", "1", *extra])
        return window.common_driver_flags(args)

    def test_default_window_does_not_warm_the_graph(self):
        self.assertNotIn("--warm-graph", self._flags())

    def test_prefill_only_warms_the_graph(self):
        self.assertIn("--warm-graph", self._flags(["--prefill-only"]))

    def test_explicit_warm_graph_works_without_prefill_only(self):
        self.assertIn("--warm-graph", self._flags(["--warm-graph"]))

    def test_no_warm_graph_opts_back_out_of_prefill_only(self):
        flags = self._flags(["--prefill-only", "--no-warm-graph"])
        self.assertNotIn("--warm-graph", flags)

    def test_no_warm_graph_beats_an_explicit_warm_graph(self):
        flags = self._flags(["--warm-graph", "--no-warm-graph"])
        self.assertNotIn("--warm-graph", flags)

    def test_the_flag_appears_exactly_once_on_the_arm_command_line(self):
        args = window.build_parser().parse_args(
            ["--sequence", "1", "--prefill-only"]
        )
        specs = window.arm_specification(args)
        run = window.plan_runs([20260829], "ABBA", 1)[0]
        argv = window.build_arm_argv(
            run,
            python="py",
            driver="drv",
            label_prefix="p",
            receipt_dir="/out",
            common_flags=window.common_driver_flags(args),
            arm_flags=specs["A"]["flags"],
            candidate_env=specs["A"]["candidate_env"],
            extra_env=specs["A"]["extra_env"],
        )
        self.assertEqual(argv.count("--warm-graph"), 1)

    def test_warm_graph_is_reserved_against_duplicate_arm_flags(self):
        self.assertIn("--warm-graph", window.RESERVED_ARM_FLAGS)
        with self.assertRaises(ValueError):
            window.check_arm_flags(["--warm-graph"])

    def test_warm_graph_does_not_disturb_the_arm_flags(self):
        args = window.build_parser().parse_args(
            ["--sequence", "1", "--prefill-only"]
        )
        base = list(window.CONTROL_FLAGS)
        base[base.index("--max-tokens") + 1] = str(
            window.PREFILL_ONLY_MAX_TOKENS
        )
        self.assertEqual(window.arm_specification(args)["A"]["flags"], base)


class FirstChunkColdTest(unittest.TestCase):
    """The cold chunk stays in the receipt after the warm-up moves it."""

    def test_reads_the_first_chunk_wall(self):
        row = {
            "prefill_chunks": [
                {"start": 0.0, "end": 2048.0, "wall_s": 4.526, "ple_gather_s": 0.378},
                {"start": 2048.0, "end": 4096.0, "wall_s": 1.191, "ple_gather_s": 0.0},
            ]
        }
        self.assertAlmostEqual(driver.first_chunk_cold_s(row), 4.526)

    def test_missing_or_empty_is_none_not_an_error(self):
        self.assertIsNone(driver.first_chunk_cold_s(None))
        self.assertIsNone(driver.first_chunk_cold_s({}))
        self.assertIsNone(driver.first_chunk_cold_s({"prefill_chunks": []}))

    def test_malformed_chunk_is_none_not_an_error(self):
        self.assertIsNone(
            driver.first_chunk_cold_s({"prefill_chunks": [{"start": 0.0}]})
        )
        self.assertIsNone(
            driver.first_chunk_cold_s({"prefill_chunks": [{"wall_s": None}]})
        )

    def test_the_driver_parser_accepts_warm_graph(self):
        args = driver.build_parser().parse_args(
            ["--label", "x", "--sequence", "1", "--seed", "1", "--warm-graph"]
        )
        self.assertTrue(args.warm_graph)


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


class ExtraEnvRecordingTest(unittest.TestCase):
    """MTPLX_* knobs may ride --*-extra-env; the receipt must SHOW them."""

    def test_mtplx_key_on_candidate_extra_env_still_reaches_arm_b_only(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--candidate-extra-env",
                "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1",
            ]
        )
        specs = window.arm_specification(args)
        self.assertEqual(
            specs["B"]["extra_env"], ["MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1"]
        )
        self.assertEqual(specs["A"]["extra_env"], [])

    def test_driver_records_the_raw_env_passthrough_in_the_receipt(self):
        text = (
            Path(window.__file__).resolve().parent / "abba_driver.py"
        ).read_text("utf-8")
        self.assertIn('"process_environment_overrides"', text)
        self.assertIn("_EXTRA_ENVIRONMENT", text)

    def test_the_receipt_field_is_declared_exactly_once(self):
        """Two branches added this key to the same dict literal.

        Python keeps the LAST duplicate, so the merge silently dropped one
        side's value. One declaration, carrying both halves.
        """

        text = (
            Path(window.__file__).resolve().parent / "abba_driver.py"
        ).read_text("utf-8")
        self.assertEqual(text.count('"process_environment_overrides":'), 1)

    def test_the_receipt_field_covers_every_raw_setting(self):
        """Not just the allowlisted MTPLX_* ones.

        An inert candidate armed through --candidate-extra-env is exactly the
        case this field exists for, and MLX_* / MTPLX_FABLE_* keys are not on
        the allowlist -- so the comprehension must iterate the whole env.
        """

        text = (
            Path(window.__file__).resolve().parent / "abba_driver.py"
        ).read_text("utf-8")
        start = text.index("    process_environment_overrides = {")
        body = text[start:text.index("\n    }", start)]
        self.assertIn("_EXTRA_ENVIRONMENT.items()", body)
        self.assertNotIn("if key in RAW_ENV_MTPLX_KEYS", body)
        # ...while still carrying the allowlist detail per key.
        for field in ('"requested"', '"effective"', '"reader"'):
            self.assertIn(field, body)

    def test_candidate_env_remains_the_validated_channel(self):
        """A profile override on --candidate-env still reaches arm B only."""

        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--candidate-env",
                "MTPLX_QWEN4_M4_ROUTER_TOP10=1",
            ]
        )
        specs = window.arm_specification(args)
        self.assertIn(
            "MTPLX_QWEN4_M4_ROUTER_TOP10=1", specs["B"]["candidate_env"]
        )
        self.assertNotIn(
            "MTPLX_QWEN4_M4_ROUTER_TOP10=1", specs["A"]["candidate_env"]
        )

    def test_fable_key_on_candidate_env_fails_at_planning_not_after_the_lock(self):
        """The window now refuses what the driver has always refused.

        ``parse_key_values(require_mtplx=True)`` has rejected MTPLX_FABLE_*
        since before either branch (it is exempt from the MTPLX_ check
        precisely because it rides --env).  Until the window mirrored that
        rule it would happily PLAN such an arm, which then died in driver
        argument parsing -- after taking the GPU lock and starting the model
        load.  Both layers now give the same verdict, and the window's names
        the channel that works.
        """

        with self.assertRaises(RuntimeError):
            driver.parse_key_values(
                ["MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1"],
                flag="--candidate-env",
                require_mtplx=True,
            )
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--candidate-env",
                "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1",
            ]
        )
        with self.assertRaises(ValueError) as caught:
            window.arm_specification(args)
        self.assertIn("--candidate-extra-env", str(caught.exception))


# --------------------------------------------------------------------------
# --prompt-tokens: one measured cell, resized
# --------------------------------------------------------------------------


#: SHA-256 of the comma-joined default prompt ids, as produced by the builder
#: BEFORE ``--prompt-tokens`` existed:
#:
#:     list(tokenizer.apply_chat_template(
#:         [{"role": "user", "content": production_prompt_content()}],
#:         tokenize=True, add_generation_prompt=True,
#:         enable_thinking=True, reasoning_effort="xhigh"))
#:
#: Pinned so the default can never drift: every retained receipt was measured
#: on these exact ids, and a window at ``--prompt-tokens 16384`` has to stay
#: comparable to them.
PRODUCTION_PROMPT_IDS_SHA256 = (
    "049ea1d936455ae5f439113372317a2088b8e26dbdf25091e1cb1b1fe90a92cb"
)


def prompt_ids_sha256(prompt_ids):
    return hashlib.sha256(
        ",".join(str(int(value)) for value in prompt_ids).encode()
    ).hexdigest()


def fixtures_present():
    return all(
        (driver.FIXTURES / name).exists()
        for name in (
            "qwen38_generation_context.py",
            "qwen38_naturalistic_generation_patch.jsonl",
        )
    )


_REAL_TOKENIZER = []


def real_tokenizer():
    """The model pack's own tokenizer: CPU only, no MLX, no weights.

    Same tokenizer-only build ``longprompt_agreement_screen.load_tokenizer``
    does.  Skips rather than fails wherever transformers, the fixtures or the
    model pack are absent, so the rest of the file stays runnable anywhere.
    """

    if not fixtures_present():
        raise unittest.SkipTest(f"prompt fixtures not present: {driver.FIXTURES}")
    if not driver.MODEL.exists():
        raise unittest.SkipTest(f"model pack not present: {driver.MODEL}")
    if not _REAL_TOKENIZER:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"transformers not importable: {exc}") from exc
        _REAL_TOKENIZER.append(
            AutoTokenizer.from_pretrained(
                str(driver.MODEL), trust_remote_code=False
            )
        )
    return _REAL_TOKENIZER[0]


class CharTokenizer:
    """One id per character, plus a fixed chat-template overhead.

    Enough for both prompt builders without transformers: the length-targeting
    path only needs ``encode`` to be additive and the template to preserve the
    sentinel.  ``templated_length`` drives the ``tokenize=True`` path, so a
    test can hand the builder a wrong-length prompt on purpose.
    """

    PREFIX = "<|im_start|>user\n"
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

    def __init__(self, templated_length=16_384):
        self.templated_length = int(templated_length)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        reasoning_effort=None,
    ):
        if tokenize:
            return list(range(self.templated_length))
        return self.PREFIX + messages[0]["content"] + self.SUFFIX

    def encode(self, text):
        return [ord(character) % 1000 for character in text]


class PromptTokenFlagTest(unittest.TestCase):
    def test_driver_default_is_the_production_prompt_length(self):
        args = driver.build_parser().parse_args(
            ["--label", "x", "--sequence", "1", "--seed", "1"]
        )
        self.assertEqual(args.prompt_tokens, driver.DEFAULT_PROMPT_TOKENS)
        self.assertEqual(driver.DEFAULT_PROMPT_TOKENS, 16_384)

    def test_window_default_is_the_production_prompt_length(self):
        args = window.build_parser().parse_args(["--sequence", "1"])
        self.assertEqual(args.prompt_tokens, window.DEFAULT_PROMPT_TOKENS)
        self.assertEqual(
            window.DEFAULT_PROMPT_TOKENS, driver.DEFAULT_PROMPT_TOKENS
        )

    def test_window_and_driver_accept_the_same_lengths(self):
        self.assertEqual(
            tuple(window.PROMPT_TOKEN_CHOICES), tuple(driver.PROMPT_TOKEN_CHOICES)
        )
        self.assertEqual(
            tuple(driver.PROMPT_TOKEN_CHOICES),
            (1_024, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144),
        )

    def test_an_unlisted_length_is_refused_by_both_parsers(self):
        for parser, argv in (
            (window.build_parser(), ["--sequence", "1", "--prompt-tokens", "4096"]),
            (
                driver.build_parser(),
                [
                    "--label",
                    "x",
                    "--sequence",
                    "1",
                    "--seed",
                    "1",
                    "--prompt-tokens",
                    "4096",
                ],
            ),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(argv)

    def test_every_arm_carries_the_requested_length(self):
        for requested in ("8192", "16384", "262144"):
            args = window.build_parser().parse_args(
                ["--sequence", "1", "--prompt-tokens", requested]
            )
            common = window.common_driver_flags(args)
            self.assertIn("--prompt-tokens", common)
            self.assertEqual(
                common[common.index("--prompt-tokens") + 1], requested
            )
            specs = window.arm_specification(args)
            runs = window.plan_runs((20260829,), "ABBA", 1)
            for run in runs:
                argv = window.build_arm_argv(
                    run,
                    python="py",
                    driver="drv",
                    label_prefix="p",
                    receipt_dir="/tmp/nope",
                    common_flags=common,
                    arm_flags=specs[run["arm"]]["flags"],
                    candidate_env=specs[run["arm"]]["candidate_env"],
                    extra_env=specs[run["arm"]]["extra_env"],
                )
                self.assertEqual(argv.count("--prompt-tokens"), 1)
                self.assertEqual(
                    argv[argv.index("--prompt-tokens") + 1], requested
                )

    def test_the_length_is_recorded_even_at_the_default(self):
        """The default is printed, not implied.

        A receipt whose ``common_driver_flags`` omits the length cannot be
        re-run from what it recorded.
        """

        args = window.build_parser().parse_args(["--sequence", "1"])
        self.assertIn("--prompt-tokens", window.common_driver_flags(args))

    def test_prompt_tokens_is_reserved_as_an_arm_flag(self):
        self.assertIn("--prompt-tokens", window.RESERVED_ARM_FLAGS)
        args = window.build_parser().parse_args(
            ["--sequence", "1", "--candidate-flag=--prompt-tokens"]
        )
        with self.assertRaises(ValueError) as caught:
            window.arm_specification(args)
        self.assertIn("--prompt-tokens", str(caught.exception))

    def test_reference_parity_is_refused_away_from_the_default_length(self):
        """Both layers refuse it, and the window refuses it before the lock."""

        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--prompt-tokens",
                "8192",
                "--require-reference-token-parity",
            ]
        )
        with self.assertRaises(ValueError) as caught:
            window.arm_specification(args)
        self.assertIn("--require-reference-token-parity", str(caught.exception))

        driver_args = driver.build_parser().parse_args(
            [
                "--label",
                "x",
                "--sequence",
                "1",
                "--seed",
                "1",
                "--prompt-tokens",
                "8192",
                "--require-reference-token-parity",
            ]
        )
        with self.assertRaises(RuntimeError) as driver_caught:
            driver.check_prompt_tokens(driver_args)
        self.assertIn("16384", str(driver_caught.exception))

    def test_reference_parity_is_still_allowed_at_the_default_length(self):
        args = window.build_parser().parse_args(
            ["--sequence", "1", "--require-reference-token-parity"]
        )
        window.check_prompt_tokens(args)
        driver_args = driver.build_parser().parse_args(
            [
                "--label",
                "x",
                "--sequence",
                "1",
                "--seed",
                "1",
                "--require-reference-token-parity",
            ]
        )
        driver.check_prompt_tokens(driver_args)

    def test_prompt_tokens_is_refused_with_the_benchmark_matrix(self):
        args = driver.build_parser().parse_args(
            [
                "--label",
                "x",
                "--sequence",
                "1",
                "--seed",
                "1",
                "--benchmark-matrix",
                "--prompt-tokens",
                "8192",
            ]
        )
        with self.assertRaises(RuntimeError) as caught:
            driver.check_prompt_tokens(args)
        self.assertIn("--benchmark-matrix", str(caught.exception))

    def test_prefill_only_budget_does_not_depend_on_the_length(self):
        for requested in ("1024", "16384", "262144"):
            args = window.build_parser().parse_args(
                ["--sequence", "1", "--prefill-only", "--prompt-tokens", requested]
            )
            self.assertEqual(
                window.resolve_max_tokens(args.max_tokens, args.prefill_only),
                window.PREFILL_ONLY_MAX_TOKENS,
            )
            self.assertTrue(window.warm_graph_enabled(args))
            self.assertIn("--warm-graph", window.common_driver_flags(args))


class PromptConstructionTest(unittest.TestCase):
    def setUp(self):
        if not fixtures_present():
            self.skipTest(f"prompt fixtures not present: {driver.FIXTURES}")

    def test_the_warm_up_cell_prefills_the_same_prompt(self):
        cell = {"label": "arm", "prompt_ids": [1, 2, 3], "max_tokens": 64}
        warm = driver.graph_warmup_cell(cell)
        self.assertEqual(warm["label"], "arm-unmeasured-graph-warmup")
        self.assertIs(warm["prompt_ids"], cell["prompt_ids"])
        self.assertEqual(len(warm["prompt_ids"]), len(cell["prompt_ids"]))
        self.assertEqual(cell["label"], "arm")

    def test_the_length_assertion_names_the_requested_length(self):
        tokenizer = CharTokenizer(templated_length=16_000)
        with self.assertRaises(RuntimeError) as caught:
            driver.build_production_prompt_ids(tokenizer, prompt_tokens=16_384)
        self.assertEqual(
            str(caught.exception), "prompt has 16000 tokens, expected 16384"
        )

    def test_an_unlisted_length_is_refused_by_the_builder(self):
        with self.assertRaises(ValueError) as caught:
            driver.build_production_prompt_ids(
                CharTokenizer(), prompt_tokens=4_096
            )
        self.assertIn("4096", str(caught.exception))

    def test_every_length_is_exact_without_a_real_tokenizer(self):
        tokenizer = CharTokenizer()
        for target in driver.PROMPT_TOKEN_CHOICES:
            if target == driver.DEFAULT_PROMPT_TOKENS:
                continue
            prompt_ids = driver.build_production_prompt_ids(
                tokenizer, prompt_tokens=target
            )
            self.assertEqual(len(prompt_ids), target)

    def test_the_fixture_hashes_are_recorded(self):
        hashes = driver.prompt_fixture_sha256()
        self.assertEqual(
            sorted(hashes),
            [
                "qwen38_generation_context.py",
                "qwen38_naturalistic_generation_patch.jsonl",
            ],
        )
        self.assertEqual(
            hashes["qwen38_generation_context.py"],
            driver.EXPECTED_CONTEXT_SHA256,
        )
        for digest in hashes.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class RealTokenizerPromptTest(unittest.TestCase):
    """Lengths and bytes against the model pack's own tokenizer (no MLX)."""

    def test_the_default_prompt_bytes_are_unchanged(self):
        tokenizer = real_tokenizer()
        prompt_ids = driver.build_production_prompt_ids(tokenizer)
        self.assertEqual(len(prompt_ids), 16_384)
        self.assertEqual(
            prompt_ids_sha256(prompt_ids), PRODUCTION_PROMPT_IDS_SHA256
        )

    def test_the_default_matches_the_pre_flag_builder_exactly(self):
        """Not just the pinned hash: the same expression, re-evaluated."""

        tokenizer = real_tokenizer()
        before = driver.templated_ids(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": driver.production_prompt_content(),
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                reasoning_effort="xhigh",
            )
        )
        self.assertEqual(driver.build_production_prompt_ids(tokenizer), before)

    def test_short_lengths_land_exactly(self):
        tokenizer = real_tokenizer()
        for target in (1_024, 8_192, 32_768):
            prompt_ids = driver.build_production_prompt_ids(
                tokenizer, prompt_tokens=target
            )
            self.assertEqual(len(prompt_ids), target)

    def test_a_resized_prompt_is_the_matrix_cell_at_that_length(self):
        """``--prompt-tokens 65536`` reproduces ``coding-64k-1k-xhigh-t1``."""

        tokenizer = real_tokenizer()
        context = (driver.FIXTURES / "qwen38_generation_context.py").read_text()
        instruction = json.loads(
            (driver.FIXTURES / "qwen38_naturalistic_generation_patch.jsonl")
            .read_text()
            .splitlines()[0]
        )["prompt"]
        matrix = driver.build_exact_coding_prompt_ids(
            tokenizer,
            context=context,
            instruction=instruction,
            target_tokens=65_536,
            reasoning_effort="xhigh",
        )
        self.assertEqual(
            driver.build_production_prompt_ids(tokenizer, prompt_tokens=65_536),
            matrix,
        )


# --------------------------------------------------------------------------
# W76: run identity and the forensics the window summary owes the reader
# --------------------------------------------------------------------------


def make_identity_rows(
    *, control_ids, candidate_ids, control_tok_s=77.0, candidate_tok_s=77.0
):
    """Three seeds of paired rows with explicit output_ids_sha256 values."""

    rows = []
    seeds = (20260829, 20260830, 20260831)
    for position, seed in enumerate(seeds):
        rows.append(
            make_row(
                2 * position,
                "A",
                seed,
                control_tok_s,
                output_ids_sha256=control_ids[position],
                output_ids_digest_source="output_ids_sha256",
                token_sources_available=True,
                token_sources_complete=True,
            )
        )
        rows.append(
            make_row(
                2 * position + 1,
                "B",
                seed,
                candidate_tok_s,
                output_ids_sha256=candidate_ids[position],
                output_ids_digest_source="output_ids_sha256",
                token_sources_available=True,
                token_sources_complete=True,
            )
        )
    return rows


class TestReceiptIdentity(unittest.TestCase):
    """An arm LABEL repeats on every attempt; a sequence does not."""

    def _receipt(self, sequence, **row_overrides):
        row = {
            "sequence": sequence,
            "seed": 20260829,
            "arm": "fable-w76-control-A0-s20260829",
            "decode_elapsed_s": 13.0,
            "decode_tok_s": 78.0,
            "wall_s": 27.0,
            "generated_tokens": 1024,
            "compiled_m4_calls": 382,
            "accepted_by_depth": [259, 187, 120],
            "drafted_by_depth": [382, 382, 382],
            "verify_forward_time_s": 11.5,
            "draft_time_s": 1.5,
            "response_token_sha256": "a" * 64,
            "peak_memory_bytes": 87_393_848_312,
            "thermal_gate": {"ready_c": 39.5},
            "reference_token_parity": {"status": "match"},
            "context_copy": {"rounds": 10, "accepted_tokens": 66,
                             "drafted_tokens": 133, "active": True},
            "page_cache_regime": "as-found",
            "per_cycle": {"available": False},
            "ple_hot_rows": {"available": False},
            "run_id": f"fable-w76-control-A0-s20260829-{sequence}-2026-09-02T20:00:00Z",
            "attempt": 2,
            "output_ids_sha256": "b" * 64,
            "token_sources": {"available": True, "complete": True,
                              "counts": {"primary": 300}},
        }
        row.update(row_overrides)
        return {"label": "fable-w76-control-A0-s20260829", "sequence": sequence,
                "rows": [row]}

    def _run(self, sequence):
        return {
            "index": 0,
            "position_in_seed": 0,
            "arm": "A",
            "arm_name": "control",
            "seed": 20260829,
            "sequence": sequence,
        }

    def test_a_receipt_from_another_run_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            window.extract_run_row(self._receipt(1788400082), self._run(1788400081))
        self.assertIn("refusing to read another run's receipt", str(caught.exception))

    def test_run_id_and_attempt_reach_the_row(self):
        row = window.extract_run_row(
            self._receipt(1788400081), self._run(1788400081)
        )
        self.assertEqual(row["attempt"], 2)
        self.assertIn("1788400081", row["run_id"])
        self.assertEqual(row["output_ids_sha256"], "b" * 64)
        self.assertEqual(row["output_ids_digest_source"], "output_ids_sha256")
        self.assertTrue(row["token_sources_available"])

    def test_an_old_receipt_falls_back_to_the_comma_joined_digest(self):
        receipt = self._receipt(1788400081)
        del receipt["rows"][0]["output_ids_sha256"]
        del receipt["rows"][0]["token_sources"]
        row = window.extract_run_row(receipt, self._run(1788400081))
        self.assertEqual(row["output_ids_sha256"], "a" * 64)
        self.assertEqual(row["output_ids_digest_source"], "response_token_sha256")
        self.assertFalse(row["token_sources_available"])


class TestLabelUniqueness(unittest.TestCase):
    def test_receipt_filenames_carry_the_sequence(self):
        first = window.plan_runs([20260829], "ABBA", 1788400081)
        second = window.plan_runs([20260829], "ABBA", 1788500001)
        labels = {window.arm_label("fable-w76", run) for run in first}
        repeat = {window.arm_label("fable-w76", run) for run in second}
        # The LABEL is identical across the two attempts ...
        self.assertEqual(labels, repeat)
        # ... and the receipt NAME is not.
        names = {window.receipt_name("fable-w76", run) for run in first}
        repeat_names = {window.receipt_name("fable-w76", run) for run in second}
        self.assertEqual(len(names), 4)
        self.assertFalse(names & repeat_names)
        for run in first:
            self.assertTrue(
                window.receipt_name("fable-w76", run).endswith(
                    f"-{run['sequence']}.json"
                )
            )

    def test_driver_forces_the_sequence_into_a_hand_passed_path(self):
        path = Path("/tmp/fable-w76-gdn-fold-alone-control-A0-s20260829.json")
        unique = driver.unique_receipt_path(path, 1788400081)
        self.assertEqual(unique.name, path.stem + "-1788400081.json")
        # Already carrying it: unchanged, so abba_window's own naming is a
        # no-op through this function.
        self.assertEqual(driver.unique_receipt_path(unique, 1788400081), unique)

    def test_driver_auto_name_carries_the_sequence(self):
        name = driver.default_receipt_name(
            "fable-w76-control-A0-s20260829",
            1788400081,
            benchmark_matrix=False,
            natural_stop=False,
            max_tokens=1024,
        )
        self.assertTrue(name.startswith("abba-1788400081-"))
        self.assertTrue(name.endswith("seeds-16k-1k.json"))

    def test_attempt_counts_writes_at_a_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            self.assertEqual(driver.receipt_attempt(path), 1)
            path.write_text(json.dumps({"attempt": 1}))
            self.assertEqual(driver.receipt_attempt(path), 2)
            path.write_text(json.dumps({"attempt": 2}))
            self.assertEqual(driver.receipt_attempt(path), 3)
            path.write_text("not json")
            self.assertEqual(driver.receipt_attempt(path), 1)

    def test_run_id_separates_two_attempts_at_one_arm(self):
        label = "fable-w76-gdn-fold-alone-control-A0-s20260829"
        first = driver.build_run_id(label, 1788400081, "2026-09-02T18:00:00Z")
        second = driver.build_run_id(label, 1788400081, "2026-09-02T21:30:00Z")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith(label))
        self.assertIn("1788400081", first)


class TestOutputIdentitySummary(unittest.TestCase):
    """The two lines that would have saved 2026-09-02."""

    def test_identical_outputs_report_yes_from_the_id_digest(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "c2", "c3"]
        )
        text = window.render_markdown(rows, window.summarize(rows))
        self.assertIn("outputs identical per seed: yes", text)

    def test_a_differing_seed_reports_no_and_names_it(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "XX", "c3"]
        )
        text = window.render_markdown(rows, window.summarize(rows))
        self.assertIn("outputs identical per seed: no", text)
        self.assertIn("seeds that differ: 20260830", text)

    def test_the_verdict_reads_ids_not_the_text_head(self):
        # Same `digest` (the old comma-joined field), different generated
        # ids.  Reading `digest` would call this identical; the whole reason
        # this line exists is that it must not.
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "c2", "DIFF"]
        )
        self.assertEqual(len({row["digest"] for row in rows}), 1)
        summary = window.summarize(rows)
        self.assertTrue(summary["overall"]["all_digests_match"])
        self.assertFalse(summary["output_identity"]["identical_per_seed"])
        self.assertIn(
            "outputs identical per seed: no",
            window.render_markdown(rows, summary),
        )

    def test_non_engagement_when_every_seed_matches_inside_the_rounding_class(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"],
            candidate_ids=["c1", "c2", "c3"],
            control_tok_s=77.0,
            candidate_tok_s=77.02,
        )
        summary = window.summarize(rows)
        identity = summary["output_identity"]
        self.assertEqual(identity["candidate_matches_control_seeds"], 3)
        self.assertEqual(identity["paired_seeds"], 3)
        self.assertTrue(identity["in_rounding_class"])
        self.assertTrue(identity["non_engagement"])
        text = window.render_markdown(rows, summary)
        self.assertIn("candidate == control on 3/3 seeds", text)
        self.assertIn("NON-ENGAGEMENT", text)

    def test_a_bit_exact_win_is_not_called_non_engagement(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"],
            candidate_ids=["c1", "c2", "c3"],
            control_tok_s=77.0,
            candidate_tok_s=84.0,
        )
        summary = window.summarize(rows)
        self.assertFalse(summary["output_identity"]["non_engagement"])
        text = window.render_markdown(rows, summary)
        self.assertIn("candidate == control on 3/3 seeds", text)
        self.assertNotIn("NON-ENGAGEMENT", text)
        self.assertIn("bit-exact change, not an inert arm", text)

    def test_partial_match_reports_the_count(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "XX", "YY"]
        )
        summary = window.summarize(rows)
        self.assertEqual(
            summary["output_identity"]["candidate_matches_control_seeds"], 1
        )
        self.assertIn(
            "candidate == control on 1/3 seeds",
            window.render_markdown(rows, summary),
        )
        self.assertFalse(summary["output_identity"]["non_engagement"])

    def test_missing_provenance_is_reported_as_unknown_not_empty(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "c2", "c3"]
        )
        for row in rows:
            row["token_sources_available"] = False
        text = window.render_markdown(rows, window.summarize(rows))
        self.assertIn("Per-token source column: NOT recorded", text)

    def test_incomplete_provenance_is_called_out(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "c2", "c3"]
        )
        rows[3]["token_sources_complete"] = False
        text = window.render_markdown(rows, window.summarize(rows))
        self.assertIn("recorded but INCOMPLETE", text)

    def test_rounding_class_band_is_configurable(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"],
            candidate_ids=["c1", "c2", "c3"],
            control_tok_s=77.0,
            candidate_tok_s=79.0,
        )
        loose = window.summarize(rows, rounding_class_pct=10.0)
        tight = window.summarize(rows, rounding_class_pct=0.01)
        self.assertTrue(loose["output_identity"]["non_engagement"])
        self.assertFalse(tight["output_identity"]["non_engagement"])

    def test_the_table_is_keyed_by_sequence(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "c2", "c3"]
        )
        text = window.render_markdown(rows, window.summarize(rows))
        header = text.splitlines()[0]
        self.assertIn("| Seq |", header)
        body = [
            line
            for line in text.splitlines()
            if line.startswith("| ") and "control (A)" in line
        ]
        sequences = [int(line.split("|")[2].strip()) for line in body]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(set(sequences)), len(sequences))

    def test_existing_digest_line_survives_for_older_readers(self):
        rows = make_identity_rows(
            control_ids=["c1", "c2", "c3"], candidate_ids=["c1", "c2", "c3"]
        )
        text = window.render_markdown(rows, window.summarize(rows))
        # abba_report.py's fallback reads the per-seed table's last column.
        self.assertIn("Every arm produced the same response-token digest:", text)
        per_seed = [
            line
            for line in text.splitlines()
            if re.match(r"\| \d{8} \|.*\| (yes|no|NO) \|\s*$", line)
        ]
        self.assertEqual(len(per_seed), 3)
