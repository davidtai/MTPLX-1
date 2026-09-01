"""Pure-Python tests for the fable HumanEval quality screen.

Nothing here imports mlx, loads a model, starts a server or touches the GPU
lock: every MLX-touching import in ``humaneval_screen`` is deferred into
``main``, and the pieces under test are the argv/environment construction, the
task selection, the pass@1 arithmetic and the receipt shape.

The load-bearing test is ``test_turbo_profile_cannot_stomp_the_family_env``:
it runs the REAL ``mtplx.profiles.apply_profile_env`` (which is deliberately
MLX-free) against the environment this script builds, so a profile change that
would silently move the screen's lane fails here instead of in a 40-minute
guarded window.

Runs under pytest or ``python -m unittest``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = ROOT / "scripts" / "fable" / "humaneval_screen.py"


def _load_screen():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "_fable_humaneval_screen", SCREEN_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


screen = _load_screen()


#: The subset of ``mtplx/server/openai.py:_server_runtime_env_overrides`` that
#: still fires when this screen exports ``CONTROL_FAMILY_ENV``. Every other key
#: in that function is gated on ``os.environ.get(key) is None`` (or popped when
#: the operator exported it), so an exported family value is what survives.
SERVER_OVERRIDES_WITH_FAMILY_ENV_EXPORTED = {
    # qwen4_exp + generation-mode mtp: set unconditionally.
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "0",
    # only reached because this screen refuses to export the key.
    "MTPLX_NAX_VERIFY": "0",
}


class ImportSafety(unittest.TestCase):
    def test_module_imports_without_mlx_or_mtplx(self):
        probe = (
            "import importlib.util, sys;"
            f"sys.path.insert(0, {str(ROOT)!r});"
            f"spec = importlib.util.spec_from_file_location('s', {str(SCREEN_PATH)!r});"
            "m = importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(m);"
            "leaked = sorted(k for k in sys.modules"
            " if k == 'mlx' or k.startswith('mlx.')"
            " or k == 'mtplx' or k.startswith('mtplx.'));"
            "print(leaked)"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(out, "[]", out)


class EnvSettings(unittest.TestCase):
    def test_parses_repeatable_key_values(self):
        self.assertEqual(
            screen.parse_env_settings(
                ["MTPLX_FABLE_HC_M4=1", "MTPLX_QWEN4_M4_STAGE3=0"]
            ),
            {"MTPLX_FABLE_HC_M4": "1", "MTPLX_QWEN4_M4_STAGE3": "0"},
        )

    def test_no_env_is_the_control_arm(self):
        self.assertEqual(screen.parse_env_settings([]), {})

    def test_rejects_malformed_duplicate_and_foreign_keys(self):
        for bad in (
            ["MTPLX_FABLE_HC_M4"],
            ["=1"],
            ["MTPLX_FABLE_HC_M4="],
            ["PATH=/tmp"],
            ["MTPLX_FABLE_HC_M4=1", "MTPLX_FABLE_HC_M4=0"],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    screen.parse_env_settings(bad)

    def test_refuses_the_never_export_key(self):
        with self.assertRaises(ValueError) as caught:
            screen.parse_env_settings(["MTPLX_NAX_VERIFY=0"])
        self.assertIn("MTPLX_NAX_VERIFY", str(caught.exception))


class ServerEnvironment(unittest.TestCase):
    def test_strips_inherited_mtplx_namespace_and_applies_family(self):
        base = {
            "PATH": "/usr/bin",
            "HOME": "/Users/davidtai",
            "MTPLX_FABLE_HC_M4": "1",  # leftover from a previous arm
            "MTPLX_NAX_VERIFY": "1",  # inherited trap
            "MTPLX_SOMETHING_ELSE": "9",
        }
        env = screen.build_server_env(base, {})
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["HOME"], "/Users/davidtai")
        self.assertNotIn("MTPLX_SOMETHING_ELSE", env)
        self.assertNotIn("MTPLX_NAX_VERIFY", env)
        # a leftover candidate export from a previous arm must not survive
        # into the control arm
        self.assertNotIn("MTPLX_FABLE_HC_M4", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        for key, value in screen.CONTROL_FAMILY_ENV.items():
            self.assertEqual(env[key], value, key)

    def test_candidate_env_is_applied_on_top(self):
        env = screen.build_server_env({"PATH": "/usr/bin"}, {"MTPLX_FABLE_HC_M4": "1"})
        self.assertEqual(env["MTPLX_FABLE_HC_M4"], "1")

    def test_candidate_env_cannot_smuggle_the_never_export_key(self):
        with self.assertRaises(ValueError):
            screen.build_server_env({}, {"MTPLX_NAX_VERIFY": "0"})

    def test_control_and_candidate_differ_only_by_the_candidate_keys(self):
        control = screen.build_server_env({"PATH": "/usr/bin"}, {})
        candidate = screen.build_server_env(
            {"PATH": "/usr/bin"}, {"MTPLX_FABLE_HC_M4": "1"}
        )
        difference = {
            key: (control.get(key), candidate.get(key))
            for key in set(control) | set(candidate)
            if control.get(key) != candidate.get(key)
        }
        self.assertEqual(difference, {"MTPLX_FABLE_HC_M4": (None, "1")})


class ProfileInteraction(unittest.TestCase):
    """The real profile applier, off-GPU. mtplx.profiles imports no MLX."""

    def setUp(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from mtplx import profiles

        self.profiles = profiles
        self.turbo = profiles.get_profile("turbo").env_dict()

    def test_never_export_key_really_is_a_profile_owned_trap(self):
        # If this ever stops being true the NEVER_EXPORT entry is obsolete and
        # the comment explaining it is a lie.
        self.assertEqual(self.turbo.get("MTPLX_NAX_VERIFY"), "1")
        self.assertNotIn(
            "MTPLX_NAX_VERIFY", self.profiles.PROFILE_ENV_USER_OVERRIDE_KEYS
        )

    def test_turbo_profile_cannot_stomp_the_family_env(self):
        environ = screen.build_server_env({"PATH": "/usr/bin"}, {})
        self.profiles.apply_profile_env(
            "turbo",
            environ=environ,
            runtime_env_overrides=SERVER_OVERRIDES_WITH_FAMILY_ENV_EXPORTED,
        )
        for key, value in screen.CONTROL_FAMILY_ENV.items():
            self.assertEqual(environ.get(key), value, f"{key} was stomped")
        # And the key we deliberately did not export lands on the value the
        # production server gives it.
        self.assertEqual(environ.get("MTPLX_NAX_VERIFY"), "0")

    def test_family_env_keys_are_accepted_as_runtime_overrides_or_are_free(self):
        for key in screen.CONTROL_FAMILY_ENV:
            with self.subTest(key=key):
                profile_owned = key in self.turbo
                overridable = key in self.profiles.PROFILE_ENV_USER_OVERRIDE_KEYS
                self.assertTrue(
                    (not profile_owned) or overridable,
                    f"{key} is profile-owned and not operator-overridable",
                )


class ServerArgv(unittest.TestCase):
    def test_pins_the_screen_contract(self):
        argv = screen.build_server_argv(python="/py", model="/m", port=8091)
        self.assertEqual(argv[:3], ["/py", "-m", "mtplx.server.openai"])
        pairs = dict(zip(argv, argv[1:]))
        self.assertEqual(pairs["--model"], "/m")
        self.assertEqual(pairs["--model-id"], screen.MODEL_ID)
        self.assertEqual(pairs["--port"], "8091")
        self.assertEqual(pairs["--profile"], "turbo")
        self.assertEqual(pairs["--depth"], "3")
        self.assertEqual(pairs["--generation-mode"], "mtp")
        self.assertEqual(pairs["--scheduler-mode"], "serial")
        self.assertEqual(pairs["--reasoning-mode"], "off")
        self.assertEqual(pairs["--temperature"], "0")
        self.assertEqual(pairs["--ssd-session-cache"], "off")
        self.assertIn("--load-mtp", argv)

    def test_never_targets_the_production_port(self):
        self.assertNotEqual(screen.DEFAULT_PORT, 8080)
        argv = screen.build_server_argv()
        self.assertNotIn("8080", argv)


class ChatPayload(unittest.TestCase):
    def test_reproduces_the_evalplus_openai_prompt(self):
        payload = screen.build_chat_payload("def f(x):\n    pass\n", max_tokens=768)
        self.assertEqual(payload["messages"][0]["content"], screen.SYSTEM_MESSAGE)
        self.assertEqual(
            payload["messages"][1]["content"],
            f"{screen.INSTRUCTION_PREFIX}\n```python\ndef f(x):\n    pass\n```",
        )
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["max_tokens"], 768)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertFalse(payload["stream"])

    def test_thinking_is_off_in_both_dialects(self):
        payload = screen.build_chat_payload("def f():\n    pass\n")
        self.assertIs(payload["enable_thinking"], False)
        self.assertIs(payload["chat_template_kwargs"]["enable_thinking"], False)


class TaskSelection(unittest.TestCase):
    def setUp(self):
        self.task_ids = [f"HumanEval/{index}" for index in range(164)]

    def test_full_set_and_smoke(self):
        self.assertEqual(len(screen.select_task_ids(self.task_ids, 164)), 164)
        smoke = screen.select_task_ids(self.task_ids, 20)
        self.assertEqual(smoke, self.task_ids[:20])

    def test_rejects_other_sizes_and_wrong_datasets(self):
        with self.assertRaises(ValueError):
            screen.select_task_ids(self.task_ids, 50)
        with self.assertRaises(ValueError):
            screen.select_task_ids(self.task_ids[:100], 20)


def _eval_results(statuses):
    return {
        "hash": "fe585eb4df8c88d844eeb463ea4d0302",
        "eval": {
            task_id: [
                {
                    "task_id": task_id,
                    "base_status": base,
                    "plus_status": plus,
                }
            ]
            for task_id, (base, plus) in statuses.items()
        },
    }


class Scoring(unittest.TestCase):
    def test_pass_at_1_and_per_problem_list(self):
        results = _eval_results(
            {
                "HumanEval/0": ("pass", "pass"),
                "HumanEval/1": ("pass", "fail"),
                "HumanEval/2": ("fail", "fail"),
                "HumanEval/3": ("pass", "pass"),
            }
        )
        summary = screen.summarize_scores(results, list(results["eval"]))
        self.assertEqual(summary["tasks"], 4)
        self.assertEqual(summary["humaneval"], {"passed": 3, "pass_at_1": 0.75})
        self.assertEqual(summary["humaneval_plus"], {"passed": 2, "pass_at_1": 0.5})
        self.assertEqual(summary["base_failures"], ["HumanEval/2"])
        self.assertEqual(
            summary["plus_failures"], ["HumanEval/1", "HumanEval/2"]
        )
        self.assertEqual(len(summary["per_problem"]), 4)
        self.assertEqual(
            summary["per_problem"][1],
            {"task_id": "HumanEval/1", "base_pass": True, "plus_pass": False},
        )

    def test_padding_rows_are_never_counted(self):
        results = _eval_results(
            {
                "HumanEval/0": ("pass", "pass"),
                "HumanEval/1": ("fail", "fail"),  # padded, not selected
                "HumanEval/2": ("fail", "fail"),  # padded, not selected
            }
        )
        summary = screen.summarize_scores(results, ["HumanEval/0"])
        self.assertEqual(summary["tasks"], 1)
        self.assertEqual(summary["humaneval"]["pass_at_1"], 1.0)

    def test_missing_or_duplicated_results_raise(self):
        results = _eval_results({"HumanEval/0": ("pass", "pass")})
        with self.assertRaises(KeyError):
            screen.summarize_scores(results, ["HumanEval/0", "HumanEval/1"])
        results["eval"]["HumanEval/0"].append(dict(results["eval"]["HumanEval/0"][0]))
        with self.assertRaises(ValueError):
            screen.summarize_scores(results, ["HumanEval/0"])


class ScoringFile(unittest.TestCase):
    def test_pads_to_the_full_problem_set(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            samples = directory / "samples.jsonl"
            samples.write_text(
                json.dumps({"task_id": "HumanEval/0", "solution": "def f(): pass"})
                + "\n",
                encoding="utf-8",
            )
            scored = directory / "samples_scored.jsonl"
            receipt = screen.write_scoring_file(
                samples, scored, ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
            )
            rows = [
                json.loads(line)
                for line in scored.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(receipt["scored"], 1)
        self.assertEqual(receipt["padded"], 2)
        self.assertEqual(receipt["padded_task_ids"], ["HumanEval/1", "HumanEval/2"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["solution"], "def f(): pass")
        self.assertEqual(rows[1]["solution"], "")


class ReceiptShape(unittest.TestCase):
    def test_receipt_carries_everything_the_verdict_needs(self):
        args = screen.build_parser().parse_args(
            ["--label", "hc-m4", "--n", "20", "--env", "MTPLX_FABLE_HC_M4=1"]
        )
        candidate = screen.parse_env_settings(args.env)
        scores = screen.summarize_scores(
            _eval_results({"HumanEval/0": ("pass", "fail")}), ["HumanEval/0"]
        )
        scores["dataset_hash"] = "fe585eb4df8c88d844eeb463ea4d0302"
        receipt = screen.build_receipt(
            args=args,
            guard={"mode": "attestation_fd", "lock_path": "/tmp/x"},
            candidate_env=candidate,
            server_argv=screen.build_server_argv(),
            provenance={"resolved_sha": screen.EXPECTED_MODEL_SHA},
            health={"ok": True},
            settings={"reasoning": "off", "temperature": 0.0},
            generation={"generated": 20, "wall_s": 1.0},
            scoring_file={"scored": 20, "padded": 144},
            scores=scores,
            timings={"total_s": 2.0},
            server_log={"path": "/tmp/server.log", "tail": ["line"]},
            warmup={"waited": True, "state": "done"},
        )
        for key in (
            "schema",
            "label",
            "arm",
            "flags",
            "model",
            "guard",
            "sampler",
            "dataset",
            "scores",
            "per_problem",
            "generation",
            "server_log",
            "timings_s",
        ):
            self.assertIn(key, receipt)
        self.assertEqual(receipt["arm"], "candidate")
        self.assertEqual(receipt["flags"]["candidate_env"], {"MTPLX_FABLE_HC_M4": "1"})
        self.assertEqual(
            receipt["flags"]["control_family_env"], screen.CONTROL_FAMILY_ENV
        )
        self.assertIn("MTPLX_NAX_VERIFY", receipt["flags"]["never_exported"])
        self.assertEqual(receipt["dataset"]["n"], 20)
        self.assertEqual(receipt["dataset"]["evalplus_version"], "0.3.1")
        self.assertTrue(receipt["sampler"]["greedy"])
        self.assertEqual(receipt["server_log"]["tail"], ["line"])
        # the receipt must be writable as-is
        json.dumps(receipt, sort_keys=True)

    def test_control_arm_is_labelled_control(self):
        args = screen.build_parser().parse_args(["--label", "control"])
        self.assertEqual(args.env, [])
        self.assertEqual(args.n, 164)
        self.assertEqual(args.port, 8091)


class DryRun(unittest.TestCase):
    def test_dry_run_prints_the_guarded_outer_command_and_exits_clean(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCREEN_PATH),
                "--label",
                "hc-m4",
                "--env",
                "MTPLX_FABLE_HC_M4=1",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        out = completed.stdout
        self.assertIn("run_guarded.py", out)
        self.assertIn("--plist", out)
        self.assertIn("com.tea.qwen.plist", out)
        self.assertIn("--lock-timeout-seconds", out)
        self.assertIn("mtplx.server.openai", out)
        self.assertIn("--reasoning-mode off", out)
        self.assertIn("MTPLX_FABLE_HC_M4", out)
        self.assertNotIn("8080", out)


if __name__ == "__main__":
    unittest.main()
