"""Compiled context-copy block round: flag semantics and the padding law.

Pure Python -- ``mtplx.context_copy`` imports nothing but ``functools`` and
``os``, so nothing here loads MLX or a model.

What is covered here is the part of the mechanism that is arithmetic: the
read-once construction-time flag, and the padding that lets one traced graph
serve every ladder rung.  The padding law is what makes the compiled round
exact, so it is tested as a law, not as an example:

  * the forward's physical width is always the installed width;
  * the LOGICAL block -- the only rows acceptance ever reads -- is a prefix of
    the padded row list and is byte-identical to what the eager round forwards;
  * padding is deterministic for a given (block, prompt, position, width).

The remaining half -- that the compiled graph's outputs equal the eager
``forward_ar`` at the same rows -- is a device claim about
``CompiledVerifyBank.forward_copy_round`` and needs a GPU window; it cannot be
asserted from here.
"""

from __future__ import annotations

import pytest

from mtplx.context_copy import (
    compiled_copy_round_enabled,
    copy_round_pad_tokens,
)


@pytest.fixture(autouse=True)
def _fresh_flag_cache():
    compiled_copy_round_enabled.cache_clear()
    yield
    compiled_copy_round_enabled.cache_clear()


# ---------------------------------------------------------------------------
# MTPLX_FABLE_COMPILED_COPY_ROUND
# ---------------------------------------------------------------------------


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("MTPLX_FABLE_COMPILED_COPY_ROUND", raising=False)
    assert compiled_copy_round_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "on", " on "])
def test_flag_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("MTPLX_FABLE_COMPILED_COPY_ROUND", value)
    compiled_copy_round_enabled.cache_clear()
    assert compiled_copy_round_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "off", "false", "yes", "2", "TRUE"])
def test_flag_everything_else_is_off(monkeypatch, value):
    """Only the three canonical spellings arm it; a typo -- or a capitalised
    spelling, matching every other flag in this module -- stays OFF."""

    monkeypatch.setenv("MTPLX_FABLE_COMPILED_COPY_ROUND", value)
    compiled_copy_round_enabled.cache_clear()
    assert compiled_copy_round_enabled() is False


def test_flag_is_read_once_per_process(monkeypatch):
    """Eligibility is construction-time: a mid-run env change must not flip it.

    A generation that switched lanes half way would leave an A/B arm partly
    compiled and partly eager, which is unmeasurable.
    """

    monkeypatch.setenv("MTPLX_FABLE_COMPILED_COPY_ROUND", "1")
    compiled_copy_round_enabled.cache_clear()
    assert compiled_copy_round_enabled() is True
    monkeypatch.setenv("MTPLX_FABLE_COMPILED_COPY_ROUND", "0")
    assert compiled_copy_round_enabled() is True


# ---------------------------------------------------------------------------
# The padding law
# ---------------------------------------------------------------------------

PROMPT = list(range(1000, 1100))


def test_flag_off_means_no_padding():
    """Width 0 is the eager lane: physical width == logical width, always."""

    for block_len in (1, 4, 8, 24, 32):
        block = PROMPT[10:10 + block_len]
        assert copy_round_pad_tokens(block, PROMPT, 10, 0) == []


def test_pad_reaches_exactly_the_installed_width():
    width = 25
    for block_len in range(1, width):
        block = PROMPT[10:10 + block_len]
        pad = copy_round_pad_tokens(block, PROMPT, 10, width)
        assert 1 + len(block) + len(pad) == width


def test_pad_is_the_prompt_continuation_past_the_proposal():
    block = PROMPT[10:18]
    pad = copy_round_pad_tokens(block, PROMPT, 10, 25)
    # rows: [primary] + block(8) + pad(16) = 25
    assert pad == PROMPT[18:34]


def test_a_block_already_at_width_needs_no_pad():
    block = PROMPT[10:34]  # 24 tokens -> 25 rows with the primary
    assert copy_round_pad_tokens(block, PROMPT, 10, 25) == []


def test_pad_repeats_the_last_token_when_the_prompt_runs_out():
    """A match near the prompt's tail still has to reach the fixed width."""

    block = PROMPT[-6:]  # positions 94..99, nothing after them
    pad = copy_round_pad_tokens(block, PROMPT, len(PROMPT) - 6, 25)
    assert len(pad) == 25 - 1 - 6
    assert set(pad) == {block[-1]}


def test_pad_mixes_continuation_then_repeat_at_the_boundary():
    block = PROMPT[90:94]
    pad = copy_round_pad_tokens(block, PROMPT, 90, 25)
    # 6 real continuation tokens (94..99), then the fill.
    assert pad[:6] == PROMPT[94:100]
    assert set(pad[6:]) == {block[-1]}
    assert len(pad) == 20


def test_padding_is_deterministic():
    block = PROMPT[90:94]
    first = copy_round_pad_tokens(block, PROMPT, 90, 25)
    second = copy_round_pad_tokens(block, PROMPT, 90, 25)
    assert first == second


def test_logical_rows_are_an_untouched_prefix_of_the_padded_rows():
    """The exactness invariant, stated directly.

    Whatever the ladder proposed, the padded forward's first ``1 + len(block)``
    rows are byte-identical to the rows the eager round would have forwarded.
    Acceptance reads only those, so it cannot observe the padding.
    """

    primary = 7
    for block_len in (1, 3, 8, 12, 16, 24):
        block = PROMPT[10:10 + block_len]
        eager_rows = [primary, *block]
        padded_rows = [
            primary,
            *block,
            *copy_round_pad_tokens(block, PROMPT, 10, 25),
        ]
        assert len(padded_rows) == 25
        assert padded_rows[: len(eager_rows)] == eager_rows


def test_a_block_wider_than_the_installed_width_is_refused():
    """No silent truncation: a block the graph cannot hold is a hard error."""

    block = PROMPT[10:40]  # 30 tokens -> 31 rows
    with pytest.raises(ValueError, match="31 rows"):
        copy_round_pad_tokens(block, PROMPT, 10, 25)


def test_an_empty_block_is_refused():
    with pytest.raises(ValueError):
        copy_round_pad_tokens([], PROMPT, 10, 25)


def test_pad_tokens_are_plain_ints():
    """The pad is spliced into an mx.array literal alongside host ints."""

    block = PROMPT[10:14]
    pad = copy_round_pad_tokens(block, PROMPT, 10, 25)
    assert all(type(token) is int for token in pad)
