"""Pure host tests for scripts/fable/longprompt_agreement_screen.py.

No server, no MLX, no Metal, no network, no tokenizer. Everything under test
here is the argv/env/prompt-sizing/agreement-scoring arithmetic -- the part of
a quality gate that silently passes the wrong thing when it is wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fable" / "longprompt_agreement_screen.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("_longprompt_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lp = _load()


# --------------------------------------------------------------------------
# The env keys this screen names must still be the ones mtplx reads
# --------------------------------------------------------------------------


def test_candidate_env_keys_are_registered_runtime_overrides():
    """A renamed knob must fail here, not silently no-op inside a window."""

    source = (ROOT / "mtplx" / "profiles.py").read_text(encoding="utf-8")
    for key in (
        lp.PREFILL_CHUNK_KEY,
        lp.COMPILE_ROWS_KEY,
        "MTPLX_GDN_BLOCKED_PREFILL",
    ):
        assert f'"{key}"' in source, f"{key} is not a profiles.py key any more"


def test_prefill_chunk_size_is_operator_overridable():
    """The turbo profile sets MTPLX_PREFILL_CHUNK_SIZE=auto.

    If the key ever leaves PROFILE_ENV_USER_OVERRIDE_KEYS, apply_profile_env
    stomps the candidate arm's 4096 back to the profile value and the screen
    compares control against control.
    """

    from mtplx.profiles import PROFILE_ENV_USER_OVERRIDE_KEYS

    assert lp.PREFILL_CHUNK_KEY in PROFILE_ENV_USER_OVERRIDE_KEYS


def test_coherence_rule_matches_the_servers_own():
    source = (ROOT / "mtplx" / "fable_prefill_chunk.py").read_text(encoding="utf-8")
    assert f'COMPILE_ROWS_ENV = "{lp.COMPILE_ROWS_KEY}"' in source
    assert lp.ALLOW_MISMATCH_KEY in source


# --------------------------------------------------------------------------
# Candidate env coherence
# --------------------------------------------------------------------------


def test_chunk_size_without_compile_rows_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        lp.assert_candidate_env_coherent({lp.PREFILL_CHUNK_KEY: "4096"})


def test_compile_rows_without_chunk_size_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        lp.assert_candidate_env_coherent({lp.COMPILE_ROWS_KEY: "4096"})


def test_matched_pair_is_accepted():
    lp.assert_candidate_env_coherent(
        {lp.PREFILL_CHUNK_KEY: "4096", lp.COMPILE_ROWS_KEY: "4096"}
    )


def test_mismatch_escape_hatch_is_honoured():
    lp.assert_candidate_env_coherent(
        {lp.PREFILL_CHUNK_KEY: "4096", lp.ALLOW_MISMATCH_KEY: "1"}
    )


def test_non_numeric_width_is_not_paired():
    lp.assert_candidate_env_coherent({lp.PREFILL_CHUNK_KEY: "auto"})


def test_empty_candidate_is_the_control():
    lp.assert_candidate_env_coherent({})


def test_every_shipped_arm_is_coherent():
    for _label, settings in lp.ARMS:
        candidate = dict(
            setting.split("=", 1) for setting in settings
        )
        lp.assert_candidate_env_coherent(candidate)


# --------------------------------------------------------------------------
# Family / server environment
# --------------------------------------------------------------------------


def test_family_env_adds_the_metal_caps_to_the_abba_control_lane():
    family = lp.build_screen_family_env({"MTPLX_AR_PIPELINE": "1"})
    assert family["MTPLX_AR_PIPELINE"] == "1"
    assert family["MTPLX_WIRED_LIMIT_BYTES"] == str(90 * 1024**3)
    assert family["MTPLX_MEMORY_LIMIT_BYTES"] == str(96 * 1024**3)


def test_family_env_refuses_to_redefine_caps_the_base_already_sets():
    with pytest.raises(RuntimeError, match="Metal caps"):
        lp.build_screen_family_env({"MTPLX_WIRED_LIMIT_BYTES": "1"})


def test_metal_caps_match_the_abba_driver():
    """The screen and the speed driver must budget the same machine."""

    source = (ROOT / "scripts" / "fable" / "abba_driver.py").read_text(
        encoding="utf-8"
    )
    assert "MEMORY_LIMIT_BYTES = 96 * 1024**3" in source
    assert "WIRED_LIMIT_BYTES = 90 * 1024**3" in source
    assert lp.MEMORY_LIMIT_BYTES == 96 * 1024**3
    assert lp.WIRED_LIMIT_BYTES == 90 * 1024**3


# --------------------------------------------------------------------------
# Server argv
# --------------------------------------------------------------------------


def test_server_argv_pins_greedy_thinking_off_and_no_session_cache():
    argv = lp.build_server_argv(
        python="/py", model="/m", model_id="mid", host="127.0.0.1", port=8093
    )
    assert argv[:3] == ["/py", "-m", "mtplx.server.openai"]
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--model"] == "/m"
    assert pairs["--model-id"] == "mid"
    assert pairs["--port"] == "8093"
    assert pairs["--profile"] == "turbo"
    assert pairs["--reasoning-mode"] == "off"
    assert pairs["--temperature"] == "0"
    assert pairs["--scheduler-mode"] == "serial"
    # A cross-request session cache could serve arm B's prefill out of arm A's
    # KV, which is the one thing that would hide a broken prefill.
    assert pairs["--ssd-session-cache"] == "off"
    assert "--no-auth" in argv


def test_chat_payload_is_greedy_and_thinking_off():
    payload = lp.build_chat_payload("hello", model_id="mid")
    assert payload["temperature"] == 0.0
    assert payload["n"] == 1
    assert payload["stream"] is False
    assert payload["enable_thinking"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == lp.DEFAULT_MAX_TOKENS
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_chat_payload_carries_logprobs_only_when_asked():
    payload = lp.build_chat_payload("hello", model_id="mid", top_logprobs=5)
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5


def test_server_contract_rejects_the_wrong_model_id():
    health = {
        "model": "other",
        "generation_mode": "mtp",
        "mtp_enabled": True,
        "depth": 3,
        "profile": {"name": "turbo"},
        "scheduler": {"mode": "serial"},
    }
    settings = {
        "reasoning": "off",
        "enable_thinking": False,
        "generation_mode": "mtp",
        "depth": 3,
    }
    with pytest.raises(RuntimeError, match="health.model"):
        lp.assert_server_contract(health, settings, model_id="mid")
    health["model"] = "mid"
    lp.assert_server_contract(health, settings, model_id="mid")


def test_server_contract_rejects_thinking_on():
    health = {
        "model": "mid",
        "generation_mode": "mtp",
        "mtp_enabled": True,
        "depth": 3,
        "profile": {"name": "turbo"},
        "scheduler": {"mode": "serial"},
    }
    settings = {
        "reasoning": "xhigh",
        "enable_thinking": True,
        "generation_mode": "mtp",
        "depth": 3,
    }
    with pytest.raises(RuntimeError, match="settings.reasoning"):
        lp.assert_server_contract(health, settings, model_id="mid")


# --------------------------------------------------------------------------
# Outer guarded commands
# --------------------------------------------------------------------------


def test_outer_command_runs_the_child_under_the_guard():
    command = lp.outer_command_line(label="control", n=6, port=8093)
    assert str(lp.RUN_GUARDED) in command
    assert "--plist" in command
    assert str(lp.QWEN_PLIST) in command
    assert "longprompt_agreement_screen.py" in command
    assert "--label control" in command
    assert " -- " in command


def test_three_arms_are_control_chunk4096_and_gdnblocked():
    arms = lp.arm_command_lines()
    labels = [label for label, _ in arms]
    assert labels == ["control", "chunk4096", "gdnblocked"]
    control, chunk, gdn = (command for _, command in arms)
    assert "--env" not in control
    assert "--env MTPLX_PREFILL_CHUNK_SIZE=4096" in chunk
    assert "--env MTPLX_QSA_PREFILL_COMPILE_ROWS=4096" in chunk
    assert "--env MTPLX_GDN_BLOCKED_PREFILL=1" in gdn
    assert "--env" not in gdn.split("--env", 1)[1].replace(
        "MTPLX_GDN_BLOCKED_PREFILL=1", ""
    )


def test_score_command_names_two_receipts():
    command = lp.score_command_line()
    assert "--score" in command
    assert command.count(".json") == 2


# --------------------------------------------------------------------------
# Prompt plan and sizing
# --------------------------------------------------------------------------


def test_prompt_plan_defaults_to_six_long_plus_two_short():
    plan = lp.prompt_plan(6)
    assert len(plan) == 8
    assert [entry["seed"] for entry in plan[:6]] == list(lp.LONG_PROMPT_SEEDS)
    assert [entry["target_tokens"] for entry in plan[:6]] == [16_384] * 6
    assert [entry["target_tokens"] for entry in plan[6:]] == [9_216, 4_608]


def test_prompt_plan_can_drop_the_short_prompts():
    assert len(lp.prompt_plan(2, include_short=False)) == 2


@pytest.mark.parametrize("bad", [0, -1, 7, 99])
def test_prompt_plan_rejects_out_of_range_n(bad):
    with pytest.raises(ValueError, match="--n must be between"):
        lp.prompt_plan(bad)


def test_every_prompt_size_is_cut_differently_by_the_two_layouts():
    """The claim the whole screen rests on: at 2,048 and at 4,096 these
    prompts are chunked at DIFFERENT offsets and into a different number of
    chunks. A size that chunked identically would compare two runs of the same
    code path and pass no matter what the candidate did."""

    for entry in lp.prompt_plan(len(lp.LONG_PROMPT_SEEDS)):
        total = int(entry["target_tokens"])
        at_2048 = lp.chunk_boundaries(total, 2048)
        at_4096 = lp.chunk_boundaries(total, 4096)
        assert at_2048 != at_4096, entry["name"]
        assert len(at_2048) > len(at_4096), entry["name"]
        assert set(at_4096) < set(at_2048), entry["name"]


def test_short_prompts_end_in_a_ragged_chunk_under_both_layouts():
    for _name, _seed, target in lp.SHORT_PROMPTS:
        assert target % 2048 != 0
        assert target % 4096 != 0
    # The 16K cell divides both exactly, so on its own it never exercises a
    # partial final chunk -- which is why the short prompts exist.
    assert lp.LONG_PROMPT_TOKENS % 2048 == 0
    assert lp.LONG_PROMPT_TOKENS % 4096 == 0


def test_chunk_boundaries_arithmetic():
    assert lp.chunk_boundaries(4608, 2048) == [2048, 4096]
    assert lp.chunk_boundaries(4608, 4096) == [4096]
    assert lp.chunk_boundaries(16384, 4096) == [4096, 8192, 12288]
    assert lp.chunk_boundaries(1024, 2048) == []
    assert lp.chunk_boundaries(0, 2048) == []


def test_rotate_context_is_a_deterministic_permutation():
    context = "\n".join(f"line {index}" for index in range(50))
    first = lp.rotate_context(context, 20260829)
    again = lp.rotate_context(context, 20260829)
    other = lp.rotate_context(context, 20260830)
    assert first == again
    assert first != other
    assert sorted(first.splitlines()) == sorted(context.splitlines())


def test_rotate_context_rejects_an_empty_fixture():
    with pytest.raises(ValueError):
        lp.rotate_context("", 1)


class FakeTokenizer:
    """A character tokenizer with a fixed chat-template overhead."""

    TEMPLATE_OVERHEAD = 37

    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(int(value)) for value in ids)

    def count_templated(self, text):
        return len(text) + self.TEMPLATE_OVERHEAD


def _calls(tokenizer=None):
    tokenizer = tokenizer or FakeTokenizer()
    return {
        "encode": tokenizer.encode,
        "decode": tokenizer.decode,
        "count_templated": tokenizer.count_templated,
    }


CONTEXT = "\n".join(f"def helper_{index}(state):  return state + {index}" for index in range(400))
INSTRUCTION = "Please update the module above."


@pytest.mark.parametrize("target", [4_608, 9_216, 16_384])
def test_prompt_sizing_lands_inside_the_tolerance(target):
    calls = _calls()
    built = lp.build_prompt_text(
        context=CONTEXT,
        instruction=INSTRUCTION,
        seed=20260829,
        target_tokens=target,
        **calls,
    )
    assert abs(built["templated_tokens"] - target) <= lp.PROMPT_TOKEN_TOLERANCE
    assert built["text"].endswith("No code.")
    assert built["chars"] == len(built["text"])
    assert len(built["text_sha256"]) == 64


def test_prompt_sizing_is_reproducible_and_seed_dependent():
    calls = _calls()
    kwargs = dict(
        context=CONTEXT, instruction=INSTRUCTION, target_tokens=9_216, **calls
    )
    a = lp.build_prompt_text(seed=20260829, **kwargs)
    b = lp.build_prompt_text(seed=20260829, **kwargs)
    c = lp.build_prompt_text(seed=20260830, **kwargs)
    assert a["text_sha256"] == b["text_sha256"]
    assert a["text_sha256"] != c["text_sha256"]


def test_prompt_sizing_repeats_a_short_context_rather_than_truncating_the_target():
    calls = _calls()
    built = lp.build_prompt_text(
        context="short\ncontext\n",
        instruction=INSTRUCTION,
        seed=1,
        target_tokens=4_608,
        **calls,
    )
    assert abs(built["templated_tokens"] - 4_608) <= lp.PROMPT_TOKEN_TOLERANCE


def test_prompt_sizing_refuses_an_impossible_target():
    with pytest.raises(ValueError):
        lp.build_prompt_text(
            context=CONTEXT,
            instruction=INSTRUCTION,
            seed=1,
            target_tokens=0,
            **_calls(),
        )


def test_prompt_sizing_raises_when_it_cannot_converge():
    calls = _calls()
    calls["count_templated"] = lambda text: 1  # never moves with the budget
    with pytest.raises(RuntimeError, match="did not converge"):
        lp.build_prompt_text(
            context=CONTEXT,
            instruction=INSTRUCTION,
            seed=1,
            target_tokens=9_216,
            **calls,
        )


def test_build_prompts_names_every_row():
    prompts = lp.build_prompts(
        plan=lp.prompt_plan(2),
        context=CONTEXT,
        instruction=INSTRUCTION,
        calls=_calls(),
    )
    assert [prompt["name"] for prompt in prompts] == [
        "long-16k-s20260829",
        "long-16k-s20260830",
        "mid-9k",
        "short-4k5",
    ]


# --------------------------------------------------------------------------
# logprobs plumbing
# --------------------------------------------------------------------------


def test_extract_logprobs_sorts_top_entries_descending():
    rows = lp.extract_logprobs(
        {
            "logprobs": {
                "content": [
                    {
                        "token": "a",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "b", "logprob": -2.0},
                            {"token": "a", "logprob": -0.1},
                        ],
                    }
                ]
            }
        }
    )
    assert rows is not None
    assert [item["token"] for item in rows[0]["top"]] == ["a", "b"]


def test_extract_logprobs_returns_none_when_absent():
    assert lp.extract_logprobs({"message": {"content": "x"}}) is None


def test_probe_reports_unsupported_when_the_server_rejects_it():
    class Boom(Exception):
        reason = "Bad Request"

        def read(self):
            return b"logprobs/top_logprobs are not supported"

    def post(url, payload, **kwargs):
        assert payload["top_logprobs"] == 5
        raise Boom()

    probe = lp.probe_logprobs_support(
        "http://x", model_id="mid", post=post
    )
    assert probe["supported"] is False
    assert probe["top_logprobs"] is None
    assert "not supported" in probe["detail"]


def test_probe_reports_unsupported_when_the_server_silently_drops_it():
    def post(url, payload, **kwargs):
        return {"choices": [{"message": {"content": "READY"}}]}

    probe = lp.probe_logprobs_support("http://x", model_id="mid", post=post)
    assert probe["supported"] is False
    assert "no logprobs.content" in probe["detail"]


def test_probe_reports_supported_when_content_comes_back():
    def post(url, payload, **kwargs):
        return {
            "choices": [
                {
                    "message": {"content": "READY"},
                    "logprobs": {"content": [{"token": "R", "logprob": -0.1}]},
                }
            ]
        }

    probe = lp.probe_logprobs_support("http://x", model_id="mid", post=post)
    assert probe == {"supported": True, "top_logprobs": 5, "detail": "ok"}


# --------------------------------------------------------------------------
# Scoring primitives
# --------------------------------------------------------------------------


def test_exact_prefix_length():
    assert lp.exact_prefix_length([1, 2, 3], [1, 2, 3]) == 3
    assert lp.exact_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert lp.exact_prefix_length([1, 2, 3], [9]) == 0
    assert lp.exact_prefix_length([], []) == 0
    # A pure truncation agrees for the whole shared length.
    assert lp.exact_prefix_length([1, 2, 3], [1, 2]) == 2


def test_top2_margin_and_top1():
    row = {
        "logprob": -0.10,
        "top": [
            {"token": "a", "logprob": -0.10},
            {"token": "b", "logprob": -0.12},
        ],
    }
    assert lp.top2_margin(row) == pytest.approx(0.02)
    assert lp.top1_logprob(row) == pytest.approx(-0.10)
    assert lp.top2_margin({"top": [{"token": "a", "logprob": -1.0}]}) is None
    assert lp.top2_margin(None) is None
    assert lp.top1_logprob({"top": [{"token": "a", "logprob": -1.0}]}) == -1.0
    assert lp.top1_logprob(None) is None


def test_result_tokens_prefers_server_logprobs_then_ids_then_text():
    assert lp.result_tokens(
        {"logprobs": [{"token": "x"}], "token_ids": [1], "text": "zz"}
    ) == ["x"]
    assert lp.result_tokens({"token_ids": [1, 2], "text": "zz"}) == [1, 2]
    assert lp.result_tokens({"text": "ab"}) == ["a", "b"]


# --------------------------------------------------------------------------
# Per-prompt comparison
# --------------------------------------------------------------------------


def _row(name, tokens, *, text=None, prompt_sha="p", logprobs=None, margins=None):
    row = {
        "name": name,
        "prompt_sha256": prompt_sha,
        "prompt_tokens": 16_384,
        "target_tokens": 16_384,
        "token_ids": list(tokens),
        "token_source": "local_reencode",
        "text": text if text is not None else "".join(str(t) for t in tokens),
    }
    if logprobs is not None:
        row["logprobs"] = logprobs
    return row


def _lp_rows(tokens, *, top1=-0.1, margin=1.0):
    return [
        {
            "token": str(token),
            "logprob": top1,
            "top": [
                {"token": str(token), "logprob": top1},
                {"token": "alt", "logprob": top1 - margin},
            ],
        }
        for token in tokens
    ]


def test_identical_completions_agree_everywhere():
    row = lp.compare_prompt(_row("a", [1, 2, 3]), _row("a", [1, 2, 3]))
    assert row["identical"] is True
    assert row["prefix_length"] == 3
    assert row["divergence_position"] is None
    assert row["text_divergence"] is None


def test_divergence_reports_position_and_text_window():
    row = lp.compare_prompt(
        _row("a", [1, 2, 3], text="abc"), _row("a", [1, 2, 9], text="abz")
    )
    assert row["identical"] is False
    assert row["prefix_length"] == 2
    assert row["divergence_position"] == 2
    assert row["text_divergence"]["char_prefix_length"] == 2
    assert row["text_divergence"]["control_next"] == "c"
    assert row["text_divergence"]["candidate_next"] == "z"


def test_near_tie_divergence_is_labelled():
    control = _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], margin=0.01))
    candidate = _row("a", [1, 2, 9], logprobs=_lp_rows([1, 2, 9], margin=0.01))
    row = lp.compare_prompt(control, candidate)
    assert row["divergence_top2_margin_nats"] == pytest.approx(0.01)
    assert row["divergence_is_near_tie"] is True
    assert row["has_logprobs"] is True


def test_confident_divergence_is_labelled():
    control = _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], margin=3.0))
    candidate = _row("a", [1, 2, 9], logprobs=_lp_rows([1, 2, 9], margin=3.0))
    row = lp.compare_prompt(control, candidate)
    assert row["divergence_is_near_tie"] is False


def test_mean_top1_delta_is_taken_over_the_agreeing_prefix_only():
    control = _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], top1=-0.10))
    candidate = _row("a", [1, 2, 9], logprobs=_lp_rows([1, 2, 9], top1=-0.15))
    row = lp.compare_prompt(control, candidate)
    assert row["logprob_positions_compared"] == 2
    assert row["mean_top1_logprob_abs_delta"] == pytest.approx(0.05)


def test_a_different_prompt_voids_the_comparison():
    with pytest.raises(ValueError, match="DIFFERENT prompts"):
        lp.compare_prompt(
            _row("a", [1], prompt_sha="x"), _row("a", [1], prompt_sha="y")
        )


def test_misaligned_rows_are_refused():
    with pytest.raises(ValueError, match="not aligned"):
        lp.compare_prompt(_row("a", [1]), _row("b", [1]))


# --------------------------------------------------------------------------
# Aggregate + verdict
# --------------------------------------------------------------------------


def _receipt(label, rows):
    return {"label": label, "flags": {"candidate_env": {}}, "results": rows}


def test_summary_aggregates_prefix_lengths_and_full_agreement():
    rows = [
        lp.compare_prompt(_row("a", [1, 2, 3]), _row("a", [1, 2, 3])),
        lp.compare_prompt(
            _row("b", [1, 2, 3], text="abc"), _row("b", [1, 9, 9], text="azz")
        ),
        lp.compare_prompt(
            _row("c", [1, 2, 3], text="abc"), _row("c", [9, 9, 9], text="zzz")
        ),
    ]
    summary = lp.summarize_agreement(rows)
    assert summary["prompts"] == 3
    assert summary["min_prefix_length"] == 0
    assert summary["median_prefix_length"] == 1
    assert summary["full_agreement_prompts"] == 1
    assert summary["full_agreement_fraction"] == pytest.approx(1 / 3)
    assert summary["logprob_evidence"] == "unavailable"


def test_verdict_passes_on_full_agreement_without_logprobs():
    rows = [lp.compare_prompt(_row("a", [1, 2, 3]), _row("a", [1, 2, 3]))]
    summary = lp.summarize_agreement(rows)
    result = lp.verdict(rows, summary)
    assert result["verdict"] == "PASS"
    assert result["reasons"] == []


def test_verdict_flags_any_divergence_when_there_are_no_logprobs():
    """The strict rule: with no margin behind it, a flip cannot be excused."""

    rows = [
        lp.compare_prompt(
            _row("a", [1, 2, 3], text="abc"), _row("a", [1, 2, 9], text="abz")
        )
    ]
    summary = lp.summarize_agreement(rows)
    result = lp.verdict(rows, summary)
    assert result["verdict"] == "FLAG"
    assert "no logprobs" in result["reasons"][0]
    assert result["offenders"][0]["name"] == "a"
    assert result["offenders"][0]["position"] == 2
    assert result["offenders"][0]["text_divergence"]["control_next"] == "c"


def test_verdict_passes_a_near_tie_divergence_with_logprobs():
    rows = [
        lp.compare_prompt(
            _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], margin=0.01)),
            _row("a", [1, 2, 9], logprobs=_lp_rows([1, 2, 9], margin=0.01)),
        )
    ]
    summary = lp.summarize_agreement(rows)
    assert summary["logprob_evidence"] == "available"
    assert lp.verdict(rows, summary)["verdict"] == "PASS"


def test_verdict_flags_a_confident_divergence():
    rows = [
        lp.compare_prompt(
            _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], margin=3.0)),
            _row("a", [1, 2, 9], logprobs=_lp_rows([1, 2, 9], margin=3.0)),
        )
    ]
    result = lp.verdict(rows, lp.summarize_agreement(rows))
    assert result["verdict"] == "FLAG"
    assert "top-2 margin" in result["reasons"][0]


def test_verdict_flags_a_drifting_logprob_field_even_with_no_divergence():
    rows = [
        lp.compare_prompt(
            _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], top1=-0.10)),
            _row("a", [1, 2, 3], logprobs=_lp_rows([1, 2, 3], top1=-0.50)),
        )
    ]
    summary = lp.summarize_agreement(rows)
    assert summary["full_agreement_fraction"] == 1.0
    result = lp.verdict(rows, summary)
    assert result["verdict"] == "FLAG"
    assert "mean top-1 logprob" in result["reasons"][0]


def test_verdict_thresholds_are_the_documented_ones():
    rows = [lp.compare_prompt(_row("a", [1]), _row("a", [1]))]
    result = lp.verdict(rows, lp.summarize_agreement(rows))
    assert result["thresholds"] == {
        "near_tie_nats": 0.05,
        "max_logprob_delta": 0.02,
    }
    assert lp.DEFAULT_NEAR_TIE_NATS == 0.05
    assert lp.DEFAULT_MAX_LOGPROB_DELTA == 0.02


# --------------------------------------------------------------------------
# Receipt-level comparison
# --------------------------------------------------------------------------


def test_compare_receipts_renders_and_verdicts():
    control = _receipt("control", [_row("a", [1, 2, 3]), _row("b", [4, 5, 6])])
    candidate = _receipt("chunk4096", [_row("a", [1, 2, 3]), _row("b", [4, 5, 6])])
    report = lp.compare_receipts(control, candidate)
    assert report["verdict"]["verdict"] == "PASS"
    assert report["summary"]["full_agreement_prompts"] == 2
    text = lp.render_report(report)
    assert "control vs chunk4096" in text
    assert "VERDICT: PASS" in text


def test_compare_receipts_refuses_a_different_prompt_set():
    control = _receipt("control", [_row("a", [1])])
    candidate = _receipt("cand", [_row("a", [1]), _row("b", [2])])
    with pytest.raises(ValueError, match="same prompts"):
        lp.compare_receipts(control, candidate)


def test_compare_receipts_refuses_empty_receipts():
    with pytest.raises(ValueError, match="no results"):
        lp.compare_receipts(_receipt("c", []), _receipt("d", []))


def test_noise_floor_reports_determinism():
    a = _receipt("control", [_row("a", [1, 2, 3])])
    b = _receipt("control-2", [_row("a", [1, 2, 3])])
    floor = lp.noise_floor(a, b)
    assert floor["summary"]["deterministic"] is True
    assert "deterministic" in lp.render_noise_floor(floor)

    c = _receipt("control-2", [_row("a", [1, 2, 9], text="abz")])
    floor = lp.noise_floor(a, c)
    assert floor["summary"]["deterministic"] is False
    assert "VOID" in lp.render_noise_floor(floor)


def test_noise_floor_needs_shared_prompts():
    with pytest.raises(ValueError, match="share no prompts"):
        lp.noise_floor(
            _receipt("a", [_row("a", [1])]), _receipt("b", [_row("z", [1])])
        )


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_print_commands_needs_no_gpu_and_lists_three_arms(capsys):
    assert lp.main(["--print-commands"]) == 0
    out = capsys.readouterr().out
    assert "# control" in out
    assert "# chunk4096" in out
    assert "# gdnblocked" in out
    assert "MTPLX_GDN_BLOCKED_PREFILL=1" in out
    assert "--score" in out


def test_print_commands_validates_n():
    with pytest.raises(ValueError, match="--n must be between"):
        lp.main(["--print-commands", "--n", "99"])


def test_score_exit_code_is_one_on_flag(tmp_path, capsys):
    import json

    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    control.write_text(json.dumps(_receipt("control", [_row("a", [1, 2, 3], text="abc")])))
    candidate.write_text(
        json.dumps(_receipt("cand", [_row("a", [1, 2, 9], text="abz")]))
    )
    assert lp.main(["--score", str(control), str(candidate)]) == 1
    assert "VERDICT: FLAG" in capsys.readouterr().out
    candidate.write_text(
        json.dumps(_receipt("cand", [_row("a", [1, 2, 3], text="abc")]))
    )
    assert lp.main(["--score", str(control), str(candidate)]) == 0


def test_score_accepts_a_second_control_as_the_noise_floor(tmp_path, capsys):
    import json

    paths = []
    for name in ("control", "candidate", "control2"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_receipt(name, [_row("a", [1, 2, 3], text="abc")])))
        paths.append(str(path))
    assert lp.main(["--score", *paths]) == 0
    assert "noise floor" in capsys.readouterr().out


def test_score_rejects_the_wrong_number_of_receipts(tmp_path):
    path = tmp_path / "one.json"
    path.write_text("{}")
    with pytest.raises(SystemExit, match="CONTROL CANDIDATE"):
        lp.main(["--score", str(path)])


def test_a_run_without_a_label_is_refused():
    with pytest.raises(SystemExit, match="--label is required"):
        lp.main([])
