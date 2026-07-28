from __future__ import annotations

from scripts.bench_laguna_prefill_admission import (
    background_prompt,
    percentile,
    receipt_exit_code,
    select_candidate,
    summarize,
)


def _row(
    chunk: int,
    queue_wait_s: float,
    prefill_tok_s: float,
    decode_tok_s: float,
    *,
    m2: bool = True,
    parity: bool = True,
    simultaneous: bool = True,
    healthy: bool = True,
) -> dict[str, object]:
    return {
        "prefill_chunk_tokens": chunk,
        "cline_queue_wait_s": queue_wait_s,
        "cline_time_to_active_s": queue_wait_s + 0.01,
        "cline_ttft_client_s": queue_wait_s + 0.50,
        "background_prefill_tok_s": prefill_tok_s,
        "aggregate_decode_tok_s": decode_tok_s,
        "both_observed_m2": m2,
        "digest_parity": parity,
        "simultaneous_class_slots": simultaneous,
        "healthy_after": healthy,
    }


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([0.1, 0.2, 0.3, 0.4, 0.5], 0.95) == 0.5


def test_background_prompt_uses_requested_single_word_geometry() -> None:
    prompt = background_prompt(4096)

    assert prompt.count("package ") == 4096
    assert "final output instruction" in prompt


def test_no_promotion_is_a_successful_benchmark_result() -> None:
    assert receipt_exit_code({"selected_candidate": None}) == 0
    assert receipt_exit_code({"fatal_error": "server died"}) == 1


def test_selects_largest_candidate_clearing_every_gate() -> None:
    rows = [
        *[_row(1024, 0.75, 100.0, 75.0) for _ in range(5)],
        *[_row(512, 0.35, 99.0, 74.5) for _ in range(5)],
        *[_row(256, 0.20, 97.0, 74.0) for _ in range(5)],
        *[_row(128, 0.10, 91.0, 75.0) for _ in range(5)],
        *[_row(1024, 0.72, 98.0, 74.0) for _ in range(5)],
    ]

    selected = select_candidate(summarize(rows))

    assert selected is not None
    assert selected["prefill_chunk_tokens"] == 256


def test_rejects_missing_m2_parity_health_or_simultaneous_state() -> None:
    rows = [_row(1024, 0.75, 100.0, 75.0) for _ in range(10)]
    invalid_trials = [
        _row(256, 0.20, 98.0, 74.0, m2=False),
        _row(256, 0.20, 98.0, 74.0, parity=False),
        _row(256, 0.20, 98.0, 74.0, simultaneous=False),
        _row(256, 0.20, 98.0, 74.0, healthy=False),
        _row(256, 0.20, 98.0, 74.0),
    ]

    assert select_candidate(summarize([*rows, *invalid_trials])) is None


def test_rejects_prefill_or_decode_regression_beyond_five_percent() -> None:
    control = [_row(1024, 0.75, 100.0, 75.0) for _ in range(10)]
    slow_prefill = [_row(256, 0.20, 94.9, 75.0) for _ in range(5)]
    slow_decode = [_row(128, 0.10, 100.0, 71.2) for _ in range(5)]

    assert select_candidate(summarize([*control, *slow_prefill, *slow_decode])) is None
