from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_final_benchmark_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_final_benchmark_arm", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_metrics_prove_cold_prefill_and_preserve_all_requested_fields() -> None:
    arm = _module()
    stats = SimpleNamespace(
        generated_tokens=1_024,
        prompt_target_prefill_time_s=4.0,
        prompt_target_prefill_tok_s=4_096.0,
        decode_elapsed_s=16.0,
        decode_tok_s=64.0,
        elapsed_s=20.0,
        peak_memory_bytes=24 * 2**30,
        cached_tokens=0,
        new_prefill_tokens=16_384,
        session_cache_hit=False,
        cache_source="none",
        session_restore_mode="cold",
        accepted_drafts=512,
        drafted_tokens=700,
        verify_calls=300,
        speculative_depth=3,
        requested_speculative_depth=3,
        events=[],
    )
    output = SimpleNamespace(tokens=list(range(1_024)), stats=stats)

    metrics = arm.native_arm_metrics(output, prompt_tokens=16_384, wall_s=20.5)

    assert metrics["prompt_tokens"] == 16_384
    assert metrics["prefill_tps"] == 4_096.0
    assert metrics["decode_tps"] == 64.0
    assert metrics["wall_s"] == 20.5
    assert metrics["peak_memory_gib"] == 24.0
    assert metrics["cached_tokens"] == 0
    assert metrics["prefix_cache_used"] is False
    assert metrics["session_restore_mode"] == "cold"


def test_burst_prompt_is_python_palindrome_task_and_fits_exact_budget() -> None:
    arm = _module()

    assert "is_palindrome" in arm.IS_PALINDROME_PROMPT
    assert "Python" in arm.IS_PALINDROME_PROMPT
    assert "test" in arm.IS_PALINDROME_PROMPT.lower()


def test_native_draft_sampler_uses_contract_except_for_full_greedy_headline() -> None:
    arm = _module()
    contract = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

    assert arm.native_draft_sampler_values(
        temperature=0.6, top_p=0.95, top_k=20, contract=contract
    ) == (1.0, 0.95, 20)
    assert arm.native_draft_sampler_values(
        temperature=0.0, top_p=1.0, top_k=0, contract=contract
    ) == (0.0, 1.0, 0)
