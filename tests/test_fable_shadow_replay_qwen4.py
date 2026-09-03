"""The shadow-draft replay, driven against a REAL Qwen3.8 Flash-Next tree.

Everything below builds ``mtplx.models.qwen4_exp`` classes for real -- a hybrid
four-layer model with two linear-attention layers (recurrent ``ArraysCache``),
two full-attention layers (``QSACache``), a PLE layer and a real
``Qwen4ExpMTP`` head -- at toy dimensions, on the CPU stream, with no weights on
disk and no Metal.  It then runs ``replay_windows`` over a synthetic K20 log
through ``build_replay_hooks``: the same ``start_segment`` / ``draft_rows`` /
``advance`` the guarded capture runs.

It exists because three separate GPU windows were spent discovering, one at a
time, that this harness had never been run against anything but a KV-only,
Llama-shaped model:

* ``target cache would not trim to 16388`` -- an offset trim is not a prefix
  commit when half the cache is recurrent state,
* ``'DecoderLayer' object has no attribute 'input_layernorm'`` -- reached
  through ``rt.forward_ar_capture``, whose layer loop is written for a
  different family's decoder layer,
* and, found by audit rather than by a crash, a draft head with four tokens of
  history where the logged run had sixteen thousand.

Each of those three fails this module.  None of them needs a GPU to find.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP, TextArgs
from mtplx.mtp_patch import MTPContract, validate_mtp_support
from mtplx.runtime import MTPLXRuntime
from scripts.fable.shadow_draft_harness import (
    ProposalVariant,
    Segment,
    build_replay_hooks,
    commit_steps,
    replay_windows,
    segment_windows,
    trimmable_offsets,
    untrimmable_entries,
    window_record,
)
from tests.test_fable_shadow_draft_harness import make_log, simple_window


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # set_default_device leaks into every later-collected module (pytest shares
    # one process), so restore it.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _hc_m4_unarmed(monkeypatch):
    """GatedResidual REFUSES to build at any width but the production one when
    MTPLX_FABLE_HC_M4 is armed, so a tiny model cannot even be constructed with
    the session's flag on."""

    monkeypatch.delenv("MTPLX_FABLE_HC_M4", raising=False)


HIDDEN = 32
HC = 4
WIDENED = HIDDEN * HC
VOCAB = 64


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def tiny_text_args(**overrides) -> TextArgs:
    """The tiny-args field set that yields a genuinely HYBRID cache.

    ``layer_types`` is given explicitly rather than derived from
    ``full_attention_interval`` so the layout is readable:

        idx 0  linear_attention  + PLE   -> ArraysCache(size=4)
        idx 1  full_attention           -> QSACache
        idx 2  linear_attention         -> ArraysCache(size=2)
        idx 3  full_attention           -> QSACache

    ``ple_layer_ids`` is ONE-indexed (HF config convention), so ``[1]`` puts
    the PLE on layer index 0 — a linear-attention layer, which is where the
    family runs it, and which keeps index 1 free for ``Qwen4ExpMTP`` (it needs
    a full-attention layer that is NOT a PLE layer).
    """

    fields = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        # Explicit hybrid layout (full_attention_interval=2 would derive the
        # same list; spelled out so the cache classes are obvious).
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        full_attention_interval=2,
        # GatedDeltaNet. Dk=32 is the documented floor: mlx-lm's Metal
        # gated-delta kernel assigns Dk/32 values per lane, so Dk=16 builds a
        # zero-length Metal array. CPU takes gated_delta_ops and would not
        # care, but keeping the floor makes the fixture GPU-portable.
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        # MoE
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        # hyper-connections
        hc_count=HC,
        hc_lowrank=8,
        # QSA indexer: budget // ratio == 4 completed blocks, so the selector
        # engages once history passes ~10 tokens.
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        # PLE (ONE-indexed -> layer index 0, a linear-attention layer)
        ple_layer_ids=[1],
        ple_embed_dim=HIDDEN,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,  # ngram_heads = (3-1)*2 = 4 -> head dim 32/4 = 8
        ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128,
        ngram_sidecar=False,  # materialize the table; no SSD sidecar here
        # rope
        partial_rotary_factor=0.5,  # rotary_dim = head_dim * 0.5 = 4
        rope_theta=10_000.0,
        eos_token_id=0,
        tie_word_embeddings=False,
    )
    fields.update(overrides)
    return TextArgs(**fields)


# ---------------------------------------------------------------------------
# model / runtime
# ---------------------------------------------------------------------------


class TinyTokenizer:
    """The only tokenizer surface MTPLXRuntime's dataclass slot needs."""

    eos_token_id: ClassVar[int | None] = None
    eos_token_ids: ClassVar[set[int]] = set()

    def decode(self, tokens, **_kwargs) -> str:
        return " ".join(str(int(t)) for t in tokens)

    def encode(self, text, **_kwargs) -> list[int]:
        return [int(part) for part in str(text).split() if part.lstrip("-").isdigit()]


def _randomize_zero_buffers(model: Model, scale: float = 0.1) -> list[str]:
    """Give the two zero-initialized PLE buffers real values.

    ``NGramTable.weight`` and ``PLELayer.conv_weight`` are constructed as
    ``mx.zeros`` (the shipped pack overwrites them at load). Leaving them zero
    makes the whole PLE branch contribute exactly 0.0, which would hide any
    bug in it — so a fixture that means to exercise PLE has to fill them.
    """

    touched: list[str] = []
    for i, layer in enumerate(model.layers):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        table = ple.ple_embedding.ngram_embedding
        if "weight" in table:
            table.weight = mx.random.normal(table.weight.shape) * scale
            touched.append(f"layers.{i}.ple.ple_embedding.ngram_embedding.weight")
        ple.conv_weight = mx.random.normal(ple.conv_weight.shape) * scale
        touched.append(f"layers.{i}.ple.conv_weight")
    return touched


def build_tiny_model(seed: int = 0, *, args: TextArgs | None = None) -> Model:
    """A real ``qwen4_exp.Model`` with a real ``Qwen4ExpMTP`` head attached."""

    mx.random.seed(seed)
    text_args = args if args is not None else tiny_text_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(text_args)))
    # attach_mtp() reads mtp.safetensors off disk. There is no pack here, so
    # build the SAME class it would build and publish it where attach_mtp
    # publishes it — on the TextModel, which is what mtp_patch._text_model()
    # and TextModel.mtp_forward both resolve.
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    _randomize_zero_buffers(model)
    model.eval()
    mx.eval(model.parameters())
    if not validate_mtp_support(model):
        raise RuntimeError("tiny qwen4_exp MTP surface failed validation")
    return model


def build_tiny_runtime(
    seed: int = 0,
    *,
    args: TextArgs | None = None,
    set_cpu_device: bool = True,
    model_path: str | Path = "/tmp/tiny",
) -> MTPLXRuntime:
    """Construct the dataclass directly — ``runtime.load()`` needs a pack."""

    if set_cpu_device:
        mx.set_default_device(mx.cpu)
    model = build_tiny_model(seed, args=args)
    contract = MTPContract()  # defaults are the family's contract
    contract.validate()
    return MTPLXRuntime(
        model=model,
        tokenizer=TinyTokenizer(),
        model_path=Path(model_path),
        mtp_enabled=True,
        contract=contract,
    )


# ---------------------------------------------------------------------------
# introspection helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The fixture, and what it proves about itself
# ---------------------------------------------------------------------------

PROMPT = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


@pytest.fixture(scope="module")
def tiny_runtime():
    return build_tiny_runtime(seed=11, set_cpu_device=False)


def test_the_fixture_is_a_genuinely_hybrid_qwen4_tree(tiny_runtime):
    """Half the cache cannot be trimmed, which is the whole point of it."""

    cache = tiny_runtime.make_cache()
    classes = [type(entry).__name__ for entry in cache]
    assert classes == ["ArraysCache", "QSACache", "ArraysCache", "QSACache"]
    assert untrimmable_entries(cache) == [0, 2]
    # And the layer really does lack the name the old capture forward reached
    # for, which is the crash this file reproduces without a GPU.
    layer = tiny_runtime.model.layers[0]
    assert not hasattr(layer, "input_layernorm")
    assert hasattr(layer, "attn_hyper_connection")
    assert hasattr(layer, "mlp_hyper_connection")


def _replay_log():
    """Three windows and a context-copy gap, all inside the tiny vocabulary."""

    return make_log(
        [
            # depth-1 reject, then a copy round commits [50, 51, 52].
            simple_window(43, [44, 45, 46], 47, accepted=1, stream=0,
                          carry=[50, 51, 52]),
            simple_window(52, [53, 54, 55], 56, accepted=3, stream=0),
            simple_window(56, [57, 58, 59], 60, accepted=2, stream=0),
        ]
    )


def _hooks(tiny_runtime, **overrides):
    kwargs = dict(runtime=tiny_runtime, prompt_ids=PROMPT)
    kwargs.update(overrides)
    return build_replay_hooks(**kwargs)


# ---------------------------------------------------------------------------
# The replay itself
# ---------------------------------------------------------------------------


def test_the_replay_walks_a_hybrid_model_end_to_end(tiny_runtime):
    log = _replay_log()
    segments = segment_windows(log)
    assert len(segments) == 1

    hooks = _hooks(tiny_runtime)
    shadow = replay_windows(
        log, hooks, [ProposalVariant(name="stock")], segments=segments
    )

    # Every window produced a full set of shaped draft rows.
    assert shadow.cycles == 3
    assert shadow.valid.all()
    assert (shadow.probs.sum(axis=-1) > 0).all()
    # The rows are in the target id space and normalised by the draft sampler.
    assert shadow.ids.max() < 64 or (shadow.probs[shadow.ids >= 64] == 0).all()


def test_the_target_cache_lands_where_commit_steps_says_it_should(tiny_runtime):
    """The commit postcondition, on a cache half of which cannot be trimmed.

    Before 2026-09-02 `advance` committed by trimming to an absolute offset,
    which `_trim_cache_to_offset` refuses outright for a cache carrying a
    recurrent entry -- even for a no-op trim on a full accept.  This is that
    failure, at four layers instead of forty-eight.
    """

    log = _replay_log()
    segment = segment_windows(log)[0]
    hooks = _hooks(tiny_runtime)
    hooks.start_segment(segment)
    assert trimmable_offsets(hooks.cache) == [len(PROMPT), len(PROMPT)]

    offset, primary = len(PROMPT), int(log["primary"][0])
    for index in range(segment.start, segment.stop):
        window = window_record(log, index)
        hooks.advance(window=window)
        for step in commit_steps(window, offset=offset, primary=primary):
            offset, primary = step.offset, step.primary
        # Both attention entries sit exactly on the committed prefix, and the
        # recurrent ones advanced with them rather than being rolled back.
        assert trimmable_offsets(hooks.cache) == [offset, offset], index
    assert offset == len(PROMPT) + len(segment.tokens) - 1
    hooks.close()


def test_the_partial_accept_takes_the_family_commit_not_a_rollback(tiny_runtime):
    """Window 0 accepts 1 of 3, so the commit path is genuinely exercised."""

    log = _replay_log()
    segment = segment_windows(log)[0]
    hooks = _hooks(tiny_runtime)
    hooks.start_segment(segment)

    window = window_record(log, 0)
    steps = commit_steps(window, offset=len(PROMPT), primary=int(log["primary"][0]))
    assert not steps[0].full, "the first window must be a PARTIAL accept"
    assert steps[1].full, "a carry step commits everything it feeds"

    forwards = tiny_runtime.diagnostic_counters.get("forward_ar_hidden_calls", 0)
    hooks.advance(window=window)
    spent = tiny_runtime.diagnostic_counters.get("forward_ar_hidden_calls", 0) - forwards
    # One forward for the window and one for the carry.  A third would mean the
    # family commit refused and the rollback re-forwarded the kept prefix --
    # correct, but it would say the capture scope had not armed.
    assert spent == 2, f"{spent} forwards for a window plus a carry"
    hooks.close()


def test_the_draft_head_keeps_the_whole_committed_history(tiny_runtime):
    """The defect no crash would have found.

    `abba_driver` records under mtp_history_policy="committed", so the head's
    cache is the whole committed prefix: the prompt from token 1, then every
    committed token.  `advance` used to trim it to offset 0 every window.
    """

    log = _replay_log()
    segment = segment_windows(log)[0]
    hooks = _hooks(tiny_runtime)
    hooks.start_segment(segment)

    # The prompt is staged from token 1 -- paired with the hidden of the token
    # before it, which is what production's prefill does.
    assert hooks.mtp_cache[0].offset == len(PROMPT) - 1

    offset, primary = len(PROMPT), int(log["primary"][0])
    for index in range(segment.start, segment.stop):
        window = window_record(log, index)
        committed_before = offset
        hooks.advance(window=window)
        for step in commit_steps(window, offset=offset, primary=primary):
            offset, primary = step.offset, step.primary
        # It GREW by exactly the tokens this window committed.
        assert hooks.mtp_cache[0].offset == len(PROMPT) - 1 + (
            offset - len(PROMPT)
        ), index
        assert offset > committed_before
    hooks.close()


def test_the_draft_chain_leaves_the_mtp_cache_where_it_found_it(tiny_runtime):
    """Both variants must draft from the same state, so the chain rolls back."""

    log = _replay_log()
    segment = segment_windows(log)[0]
    hooks = _hooks(tiny_runtime)
    hooks.start_segment(segment)
    before = hooks.mtp_cache[0].offset
    window = window_record(log, 0)
    first = hooks.draft_rows(
        variant=ProposalVariant(name="stock"), forced_tokens=window["draft_tokens"]
    )
    assert hooks.mtp_cache[0].offset == before
    second = hooks.draft_rows(
        variant=ProposalVariant(name="stock"), forced_tokens=window["draft_tokens"]
    )
    assert hooks.mtp_cache[0].offset == before
    # Same state in, same rows out -- the pairing the fidelity gate relies on.
    assert len(first) == len(second) == 3
    for (ids_a, probs_a), (ids_b, probs_b) in zip(first, second):
        np.testing.assert_array_equal(ids_a, ids_b)
        np.testing.assert_allclose(probs_a, probs_b, rtol=0, atol=0)
    hooks.close()


def test_a_second_segment_starts_from_a_clean_state(tiny_runtime):
    log = _replay_log()
    hooks = _hooks(tiny_runtime)
    segment = segment_windows(log)[0]
    hooks.start_segment(segment)
    hooks.advance(window=window_record(log, 0))
    grown = hooks.mtp_cache[0].offset
    hooks.start_segment(Segment(index=1, start=0, stop=1, tokens=segment.tokens))
    assert trimmable_offsets(hooks.cache) == [len(PROMPT), len(PROMPT)]
    assert hooks.mtp_cache[0].offset == len(PROMPT) - 1 < grown
    hooks.close()


def test_the_rollback_fallback_lands_the_same_prefix(tiny_runtime, monkeypatch):
    """Route 4 of the ladder, the one that runs if the capture ever stops arming.

    It is the safety net for a family whose `commit_verified_window` refuses --
    a scope that did not arm, a snapshot the layers reject -- so it must leave
    the cache in exactly the state the family commit would have, and the test
    forces it rather than hoping it is never needed.
    """

    log = _replay_log()
    segment = segment_windows(log)[0]
    window = window_record(log, 0)

    def _committed(**_kwargs):
        hooks = _hooks(tiny_runtime)
        hooks.start_segment(segment)
        hooks.advance(window=window)
        offsets = trimmable_offsets(hooks.cache)
        history = hooks.mtp_cache[0].offset
        hooks.close()
        return offsets, history

    expected = _committed()
    monkeypatch.setattr(
        type(tiny_runtime.model),
        "commit_verified_window",
        lambda *args, **kwargs: False,
    )
    assert _committed() == expected


def test_the_prompt_history_is_staged_identically_however_it_is_chunked(
    tiny_runtime,
):
    """The staging loop exists to keep one 16k-row pass off the device."""

    log = _replay_log()
    segment = segment_windows(log)[0]
    offsets = []
    for chunk in (2048, 4, 1):
        hooks = _hooks(tiny_runtime, history_chunk=chunk)
        hooks.start_segment(segment)
        offsets.append(hooks.mtp_cache[0].offset)
        hooks.close()
    assert offsets == [len(PROMPT) - 1] * 3
