from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    output = SimpleNamespace(
        tokens=list(range(1_024)),
        text="def generated():\n    return True\n",
        stats=stats,
        finish_reason="length",
    )

    metrics = arm.native_arm_metrics(output, prompt_tokens=16_384, wall_s=20.5)

    assert metrics["prompt_tokens"] == 16_384
    assert metrics["prefill_tps"] == 4_096.0
    assert metrics["decode_tps"] == 64.0
    assert metrics["wall_s"] == 20.5
    assert metrics["peak_memory_gib"] == 24.0
    assert metrics["cached_tokens"] == 0
    assert metrics["prefix_cache_used"] is False
    assert metrics["session_restore_mode"] == "cold"
    assert metrics["finish_reason"] == "length"
    assert metrics["output_text"] == "def generated():\n    return True\n"


def test_dflash_metrics_report_actual_stop_and_effective_widths() -> None:
    arm = _module()
    stats = SimpleNamespace(
        generated_tokens=102,
        prompt_eval_time_s=0.2,
        prompt_tps=500.0,
        decode_elapsed_s=0.9,
        decode_tok_s=113.0,
        elapsed_s=1.1,
        peak_memory_bytes=20 * 2**30,
        accepted_drafts=87,
        drafted_tokens=93,
        verify_calls=15,
    )
    output = SimpleNamespace(
        tokens=list(range(102)),
        text="normalized = ''.join(c.casefold() for c in text if c.isalnum())\nreturn normalized == normalized[::-1]",
        stats=stats,
        finish_reason="stop",
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(draft_block_size=8),
        telemetry=SimpleNamespace(
            adaptive_metrics={"cycles_by_block": {5: 2, 6: 3, 8: 1}}
        ),
        qwen38_feature_receipt={
            "context_route": {
                "requested_adaptive": True,
                "effective_adaptive": True,
            }
        },
    )

    metrics = arm._dflash_arm_metrics(
        output, runtime, prompt_tokens=100, wall_s=1.12
    )

    assert metrics["finish_reason"] == "stop"
    assert metrics["output_text"].startswith("normalized =")
    assert metrics["requested_width"] == 8
    assert metrics["effective_widths"] == [5, 6, 8]


def test_burst_prompt_is_python_palindrome_task_and_fits_exact_budget() -> None:
    arm = _module()

    class RecordingTokenizer:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return list(range(100))

        def decode(self, ids):
            return "rendered prompt"

    tokenizer = RecordingTokenizer()
    _text, ids = arm._build_exact_chat_prompt(
        tokenizer,
        text=arm.IS_PALINDROME_PROMPT,
        target_tokens=100,
    )

    assert "is_palindrome" in arm.IS_PALINDROME_PROMPT
    assert "Python" in arm.IS_PALINDROME_PROMPT
    assert "def is_palindrome" in arm.IS_PALINDROME_PROMPT
    assert "A man, a plan, a canal: Panama" in arm.IS_PALINDROME_PROMPT
    assert "Return only the function body" in arm.IS_PALINDROME_PROMPT
    assert len(ids) == 100
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": arm.IS_PALINDROME_PROMPT}],
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]


def test_palindrome_uses_eos_with_1024_cap_but_load_scenarios_force_1024() -> None:
    arm = _module()

    assert arm.stop_token_ids_for_prompt("is_palindrome") is None
    assert arm.stop_token_ids_for_prompt("coding") == set()


def test_native_draft_sampler_uses_contract_except_for_full_greedy_headline() -> None:
    arm = _module()
    contract = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

    assert arm.native_draft_sampler_values(
        temperature=0.6, top_p=0.95, top_k=20, contract=contract
    ) == (1.0, 0.95, 20)
    assert arm.native_draft_sampler_values(
        temperature=0.0, top_p=1.0, top_k=0, contract=contract
    ) == (0.0, 1.0, 0)


def test_candidate_adaptive_receipt_must_be_effective() -> None:
    arm = _module()

    arm.validate_candidate_adaptive_receipt(
        {
            "context_route": {
                "requested_adaptive": True,
                "effective_adaptive": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="not effectively adaptive"):
        arm.validate_candidate_adaptive_receipt(
            {
                "context_route": {
                    "requested_adaptive": True,
                    "effective_adaptive": False,
                }
            }
        )
