"""MTPLX_FABLE_QSA_PREFILL_16K crossover arithmetic and wiring.

Pure Python: imports ``mtplx.qsa_prefill_16k`` (no MLX) and reads
``mtplx/models/qwen4_exp.py`` as text.  Nothing here touches a GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mtplx import qsa_prefill_16k as gate


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mtplx" / "models" / "qwen4_exp.py"
MODEL_TEXT = MODEL_PATH.read_text(encoding="utf-8")
MODEL_TREE = ast.parse(MODEL_TEXT)
PROFILES_TEXT = (ROOT / "mtplx" / "profiles.py").read_text(encoding="utf-8")

# The production cell: a 16,384-token prompt cut into 2,048-token chunks by
# mtplx/generation.py's `_iter_prefill_chunk_spans`.  The final chunk stops one
# token short -- the 16,384th token is the first decode step.
PROMPT_TOKENS = 16_384
CHUNK = 2_048


def production_chunks() -> list[tuple[int, int]]:
    """(rows, total_tokens) for each 16K prefill chunk, in order."""

    prefilled = PROMPT_TOKENS - 1
    spans = [
        (start, min(prefilled, start + CHUNK))
        for start in range(0, prefilled, CHUNK)
    ]
    return [(end - start, end) for start, end in spans]


def _top_function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in MODEL_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _source(node: ast.AST) -> str:
    value = ast.get_source_segment(MODEL_TEXT, node)
    assert value is not None
    return value


# ---------------------------------------------------------------------------
# The geometry the flag exists for
# ---------------------------------------------------------------------------


def test_production_chunk_geometry_is_eight_chunks_of_2048():
    chunks = production_chunks()
    assert len(chunks) == 8
    assert [rows for rows, _ in chunks] == [2048] * 7 + [2047]
    assert [total for _, total in chunks] == [
        2048,
        4096,
        6144,
        8192,
        10240,
        12288,
        14336,
        16383,
    ]


def test_shipped_history_form_never_reaches_the_shipped_crossover():
    """`total - rows` tops out at 14,336: the lane cannot engage at 16K."""

    histories = [
        gate.history_for_gate(rows, total, fable_16k=False)
        for rows, total in production_chunks()
    ]
    assert histories == [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336]
    assert max(histories) < 32_768


def test_shipped_floor_leaves_chunk_one_unreachable_by_env_alone():
    """Even at the 2049 floor, chunk 1's 2,048 history stays below it."""

    engaged = [
        gate.engages(
            rows, total, crossover=gate.SHIPPED_FLOOR, fable_16k=False
        )
        for rows, total in production_chunks()
    ]
    assert engaged == [False, False, True, True, True, True, True, True]


def test_flag_engages_every_chunk_whose_selection_is_not_dense():
    """total_tokens >= 2049 -- chunk 0 alone stays dense (nb == block_topk)."""

    engaged = [
        gate.engages(rows, total, crossover=gate.CROSSOVER, fable_16k=True)
        for rows, total in production_chunks()
    ]
    assert engaged == [False, True, True, True, True, True, True, True]


def test_prior_8192_receipt_engaged_only_the_last_four_chunks():
    """Reconciles rebench3-2760 (1,085 tok/s vs a matched 1,233).

    That arm set both crossovers to 8192 on a tree using the shipped history
    form, so it sparsified chunks 4-7 only -- half the prompt, and the half
    where the dense score tensor is largest.
    """

    engaged = [
        gate.engages(rows, total, crossover=8192, fable_16k=False)
        for rows, total in production_chunks()
    ]
    assert engaged == [False] * 4 + [True] * 4


# ---------------------------------------------------------------------------
# Env resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", " On "])
def test_flag_true_spellings(value):
    assert gate.flag_from_env({gate.ENV_FLAG: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "NO", "off", " "])
def test_flag_false_spellings(value):
    assert gate.flag_from_env({gate.ENV_FLAG: value}) is False


def test_flag_absent_is_off():
    assert gate.flag_from_env({}) is False


@pytest.mark.parametrize("value", ["2", "maybe", "16384", "-1"])
def test_unparseable_flag_raises_rather_than_silently_measuring_control(value):
    with pytest.raises(ValueError) as excinfo:
        gate.flag_from_env({gate.ENV_FLAG: value})
    assert gate.ENV_FLAG in str(excinfo.value)


def test_crossover_defaults_to_the_candidate_value_when_env_is_unset():
    assert (
        gate.crossover_from_env({}, gate.ENV_MIN_CONTEXT, 32_768)
        == gate.CROSSOVER
    )
    assert (
        gate.crossover_from_env({}, gate.ENV_FLASH_MIN_CONTEXT, 32_768)
        == gate.CROSSOVER
    )


def test_explicit_operator_crossover_still_wins():
    """A crossover sweep has to remain possible with the flag armed."""

    env = {gate.ENV_MIN_CONTEXT: "8192"}
    assert gate.crossover_from_env(env, gate.ENV_MIN_CONTEXT, 8192) == 8192
    # Blank/whitespace is "unset", matching the shipped `or` resolution.
    assert (
        gate.crossover_from_env(
            {gate.ENV_MIN_CONTEXT: "  "}, gate.ENV_MIN_CONTEXT, 32_768
        )
        == gate.CROSSOVER
    )


# ---------------------------------------------------------------------------
# Wiring: the shipped route must survive unchanged behind the flag
# ---------------------------------------------------------------------------


def test_shipped_history_expression_is_preserved_on_the_default_route():
    route = _source(_top_function("_qsa_large_prefill_enabled"))
    flash = _source(_top_function("_qsa_prefill_flash_attention_enabled"))
    assert "int(total_tokens) - int(rows) >= _qsa_prefill_min_context()" in route
    assert (
        "int(total_tokens) - int(rows) >= _qsa_prefill_flash_min_context()"
        in flash
    )
    assert "if _fable_qsa_prefill_16k():" in route
    assert "if _fable_qsa_prefill_16k():" in flash


def test_candidate_route_gates_on_total_tokens_and_keeps_every_other_guard():
    route = _source(_top_function("_fable_qsa_large_prefill_enabled"))
    assert "_qsa_prefill_enabled()" in route
    assert 'current_attention_phase() == "prefill"' in route
    assert "int(rows) >= _qsa_prefill_min_rows()" in route
    assert "int(total_tokens) >= _fable_qsa_prefill_min_context()" in route
    assert "- int(rows)" not in route


def test_flag_is_read_once_and_refuses_to_run_without_the_lane():
    resolver = _source(_top_function("_fable_qsa_prefill_16k"))
    assert "@lru_cache(maxsize=1)" in MODEL_TEXT
    assert "qsa_prefill_16k.flag_from_env(os.environ)" in resolver
    assert "raise RuntimeError" in resolver
    assert "_qsa_prefill_enabled()" in resolver


def test_flag_is_registered_for_validated_operator_overrides():
    assert f'"{gate.ENV_FLAG}"' in PROFILES_TEXT
