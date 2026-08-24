from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_final_benchmark_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_final_benchmark_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_script_bootstrap_adds_repository_root_to_import_path() -> None:
    original = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT.parents[1]]
        matrix = _module()
        assert str(matrix.ROOT) in sys.path
    finally:
        sys.path[:] = original


class CharacterTokenizer:
    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def test_scenarios_are_exact_cold_prefill_lengths_not_prefix_additions() -> None:
    matrix = _module()

    assert [(item.name, item.prompt_tokens) for item in matrix.SCENARIOS] == [
        ("burst_is_palindrome", 100),
        ("coding_cold_prefill_1k", 1_024),
        ("coding_cold_prefill_16k", 16_384),
        ("coding_cold_prefill_64k", 65_536),
        ("coding_cold_prefill_128k", 131_072),
    ]
    assert all(item.max_tokens == 1_024 for item in matrix.SCENARIOS)
    assert matrix.SCENARIOS[0].temperature == 0.0
    assert matrix.SCENARIOS[0].top_p == 1.0
    assert matrix.SCENARIOS[0].top_k == 0
    assert all(item.temperature == 0.6 for item in matrix.SCENARIOS[1:])
    assert all(item.top_p == 0.95 for item in matrix.SCENARIOS[1:])
    assert all(item.top_k == 20 for item in matrix.SCENARIOS[1:])


def test_coding_prompt_has_one_instruction_and_exact_total_token_budget() -> None:
    matrix = _module()

    prompt, token_ids = matrix.build_exact_coding_prompt(
        CharacterTokenizer(),
        target_tokens=2_048,
        context="class ExistingModule:\n    pass\n",
        instruction="Implement the production-ready scheduler.",
    )

    assert len(token_ids) == 2_048
    assert prompt.endswith("Implement the production-ready scheduler.")
    assert prompt.count("Implement the production-ready scheduler.") == 1


def test_child_command_pins_source_revision_and_disables_prefix_sessions(tmp_path) -> None:
    matrix = _module()
    args = SimpleNamespace(
        model=Path("/models/target"),
        draft=Path("/models/draft"),
        context_file=Path("/repo/mtplx/generation.py"),
        prompt_file=Path("/repo/prompt.jsonl"),
        conditioner_tokens=32,
        seed=42,
        lock=Path("/tmp/gpu.lock"),
    )
    scenario = matrix.SCENARIOS[2]

    command = matrix.child_command(
        args,
        engine="main_native_mtp",
        source_root=Path("/tmp/main"),
        source_commit="abc123",
        scenario=scenario,
        output=tmp_path / "arm.json",
    )

    assert command[command.index("--prompt-tokens") + 1] == "16384"
    assert command[command.index("--source-root") + 1] == "/tmp/main"
    assert command[command.index("--source-commit") + 1] == "abc123"
    assert "--prefix-cache" not in command
    assert "--session-id" not in command


def test_headline_uses_full_same_shape_decode_conditioning() -> None:
    matrix = _module()
    args = SimpleNamespace(
        model=Path("/models/target"),
        draft=Path("/models/draft"),
        context_file=Path("/repo/mtplx/generation.py"),
        prompt_file=Path("/repo/prompt.jsonl"),
        conditioner_tokens=32,
        seed=42,
        lock=Path("/tmp/gpu.lock"),
    )

    command = matrix.child_command(
        args,
        engine="pr_dflash2",
        source_root=Path("/repo"),
        source_commit="abc123",
        scenario=matrix.SCENARIOS[0],
        output=Path("/tmp/arm.json"),
    )

    assert command[command.index("--conditioner-tokens") + 1] == "1024"
    assert command[command.index("--conditioner-mode") + 1] == "same_prompt"
    assert "--dflash2-adaptive" in command


def test_headline_accepts_natural_eos_below_its_output_limit() -> None:
    matrix = _module()
    receipts = []
    for engine, count in (
        ("main_native_mtp", 180),
        ("pr_dflash2", 190),
        ("pr_dflash2", 190),
        ("main_native_mtp", 180),
    ):
        receipts.append(
            {
                "arm": {
                    "engine": engine,
                    "wall_s": 3.0,
                    "prefill_tps": 500.0,
                    "decode_tps": 80.0,
                    "peak_memory_gib": 20.0,
                    "prefill_s": 0.2,
                    "decode_elapsed_s": 2.5,
                    "generated_tokens": count,
                    "prompt_tokens": 100,
                    "fallback_ar": False,
                    "token_sha256": engine,
                }
            }
        )

    result = matrix.aggregate_scenario(matrix.SCENARIOS[0], receipts)

    assert result["output_limit_tokens"] == 1_024
    assert result["actual_generated_tokens_by_engine"] == {
        "main_native_mtp": [180, 180],
        "pr_dflash2": [190, 190],
    }
    assert "generated_tokens" not in result
    assert result["correctness"]["output_limit_respected"] is True
    assert result["correctness"]["exact_prompt_and_output_counts"] is True


def test_aggregate_reports_requested_metrics_and_matched_wall_delta() -> None:
    matrix = _module()
    receipts = []
    for engine, wall, prefill, decode, peak in (
        ("main_native_mtp", 20.0, 1000.0, 50.0, 20.0),
        ("pr_dflash2", 16.0, 1200.0, 64.0, 22.0),
        ("pr_dflash2", 14.0, 1400.0, 66.0, 24.0),
        ("main_native_mtp", 22.0, 1100.0, 52.0, 21.0),
    ):
        receipts.append(
            {
                "engine": engine,
                "source_commit": "main" if engine == "main_native_mtp" else "pr",
                "workload": {"prompt_tokens": 16_384, "generated_tokens": 1_024},
                "arm": {
                    "engine": engine,
                    "wall_s": wall,
                    "prefill_tps": prefill,
                    "decode_tps": decode,
                    "peak_memory_gib": peak,
                    "prefill_s": 10.0,
                    "decode_elapsed_s": wall - 10.0,
                    "generated_tokens": 1_024,
                    "prompt_tokens": 16_384,
                    "fallback_ar": False,
                    "token_sha256": engine,
                },
            }
        )

    result = matrix.aggregate_scenario(matrix.SCENARIOS[2], receipts)

    assert result["summary"]["main_native_mtp"]["wall_s"] == 21.0
    assert result["summary"]["pr_dflash2"]["wall_s"] == 15.0
    assert result["summary"]["pr_dflash2"]["prefill_tps"] == 1300.0
    assert result["summary"]["pr_dflash2"]["decode_tps"] == 65.0
    assert result["summary"]["pr_dflash2"]["peak_memory_gib"] == 23.0
    assert result["wall_time_improvement_pct"] == pytest.approx(40.0)
    assert result["cold_prefill"] is True
    assert result["prefix_cache_used"] is False
