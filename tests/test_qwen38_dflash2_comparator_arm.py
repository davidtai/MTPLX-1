from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_dflash2_comparator_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_dflash2_comparator_arm", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workload_mapping_preserves_current_matrix_prompt_contract() -> None:
    arm = _module()

    assert arm._workload(SimpleNamespace(prompt_kind="is_palindrome", reasoning_effort=None)) == "vanity"
    assert arm._workload(SimpleNamespace(prompt_kind="coding", reasoning_effort="low")) == "low"
    assert arm._workload(SimpleNamespace(prompt_kind="coding", reasoning_effort="xhigh")) == "xhigh"


def test_zero_token_conditioner_skips_dflash_generation() -> None:
    arm = _module()
    called = []

    result = arm._generate_or_skip(
        lambda *_args: called.append(True), object(), [1, 2, 3],
        SimpleNamespace(max_tokens=0),
    )

    assert result is None
    assert called == []
