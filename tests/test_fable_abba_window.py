"""Pure-Python tests for the fable ABBA harness planning and summary maths.

Nothing here imports mlx or loads a model; the module under test is import-safe
because every MLX-touching import in ``abba_window`` is deferred into ``main``.
Runs under pytest or ``python -m unittest``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
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
            "tok/window",
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

    def test_missing_values_render_as_na_not_a_crash(self):
        rows = [
            make_row(
                0,
                "A",
                1,
                60.0,
                ms_per_compiled_window=None,
                tokens_per_window=None,
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

    def test_candidate_env_remains_the_validated_channel(self):
        args = window.build_parser().parse_args(
            [
                "--sequence",
                "1",
                "--candidate-env",
                "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1",
            ]
        )
        specs = window.arm_specification(args)
        self.assertIn(
            "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1", specs["B"]["candidate_env"]
        )
        self.assertNotIn(
            "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD=1", specs["A"]["candidate_env"]
        )
