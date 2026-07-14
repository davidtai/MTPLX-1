"""Benchmark runners."""

from mtplx.benchmarks.runners.hy3_dynamic_memory import (
    CONTEXT_MATRIX_TOKENS,
    run_allocator_release_probe,
    run_balanced_campaign,
    run_exclusive_hardware_window,
    run_subprocess_campaign,
)


run_hy3_dynamic_memory_campaign = run_balanced_campaign

__all__ = [
    "CONTEXT_MATRIX_TOKENS",
    "run_allocator_release_probe",
    "run_balanced_campaign",
    "run_exclusive_hardware_window",
    "run_hy3_dynamic_memory_campaign",
    "run_subprocess_campaign",
]
