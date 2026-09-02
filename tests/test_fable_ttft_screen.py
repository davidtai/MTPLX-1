"""Pure host tests for scripts/fable/ttft_screen.py.

No server, no MLX, no Metal, no network. Everything under test here is the
argv/env/prompt/SSE-folding arithmetic -- the part of a benchmark harness that
silently measures the wrong thing when it is wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fable" / "ttft_screen.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("_ttft_screen_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ttft = _load()


# --------------------------------------------------------------------------
# Session headers -- the trap the audit names
# --------------------------------------------------------------------------


def test_header_names_match_the_server_resolution_order():
    """These are read out of mtplx/engine_session.py; drift breaks the lease."""

    source = (ROOT / "mtplx" / "engine_session.py").read_text(encoding="utf-8")
    for key in ttft.HEADER_SESSION_KEYS:
        assert f'"{key}"' in source, f"{key} is not a session header any more"
    assert set(ttft.SENT_SESSION_HEADERS) <= set(ttft.HEADER_SESSION_KEYS)


def test_agent_tool_names_are_recognised_by_the_server():
    source = (ROOT / "mtplx" / "server" / "openai.py").read_text(encoding="utf-8")
    start = source.index("def _anonymous_coding_agent_tool_request")
    body = source[start : start + 2_000]
    for name in ttft.AGENT_TOOL_NAMES:
        assert f'"{name}"' in body, f"{name} no longer arms the live-ref lease"


@pytest.mark.parametrize(
    "mode,expect_headers,expect_tools",
    [
        ("header", True, False),
        ("tools", False, True),
        ("both", True, True),
        ("env", False, False),
    ],
)
def test_live_ref_modes_are_exclusive_and_explicit(mode, expect_headers, expect_tools):
    headers = ttft.session_headers("sid", live_ref=mode)
    tools = ttft.request_tools(mode)
    assert bool(headers) is expect_headers
    assert bool(tools) is expect_tools
    if expect_headers:
        assert set(headers) == set(ttft.SENT_SESSION_HEADERS)
        assert set(headers.values()) == {"sid"}


def test_unknown_live_ref_mode_is_refused():
    with pytest.raises(ValueError):
        ttft.session_headers("sid", live_ref="curl")


# --------------------------------------------------------------------------
# Candidate env
# --------------------------------------------------------------------------


def _parse(settings):
    parsed = {}
    for setting in settings:
        key, value = str(setting).split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def test_control_arm_has_no_env():
    assert ttft.build_candidate_env(
        [], gdn_boundary_max=None, live_ref="header", parse=_parse
    ) == {}
    assert ttft.PRODUCTION_FAMILY_ENV == {}


def test_gdn_boundary_max_is_a_pure_env_knob():
    env = ttft.build_candidate_env(
        [], gdn_boundary_max=32, live_ref="header", parse=_parse
    )
    assert env == {ttft.GDN_BOUNDARY_MAX_KEY: "32"}


def test_gdn_boundary_max_conflict_is_refused():
    with pytest.raises(ValueError):
        ttft.build_candidate_env(
            [f"{ttft.GDN_BOUNDARY_MAX_KEY}=16"],
            gdn_boundary_max=32,
            live_ref="header",
            parse=_parse,
        )


def test_gdn_boundary_max_must_be_positive():
    with pytest.raises(ValueError):
        ttft.build_candidate_env(
            [], gdn_boundary_max=0, live_ref="header", parse=_parse
        )


def test_live_ref_env_mode_sets_the_key():
    env = ttft.build_candidate_env(
        [], gdn_boundary_max=None, live_ref="env", parse=_parse
    )
    assert env == {ttft.LIVE_REF_ENV_KEY: "1"}


def test_live_ref_key_smuggled_through_env_is_refused():
    """Setting it by hand would change the measured lane without a receipt."""

    with pytest.raises(ValueError):
        ttft.build_candidate_env(
            [f"{ttft.LIVE_REF_ENV_KEY}=1"],
            gdn_boundary_max=None,
            live_ref="header",
            parse=_parse,
        )


def test_fable_flags_pass_through():
    env = ttft.build_candidate_env(
        [
            "MTPLX_FABLE_QSA_RESTORE_STAGING=1",
            "MTPLX_FABLE_PROTECTED_TERMINAL=1",
        ],
        gdn_boundary_max=None,
        live_ref="header",
        parse=_parse,
    )
    assert env == {
        "MTPLX_FABLE_QSA_RESTORE_STAGING": "1",
        "MTPLX_FABLE_PROTECTED_TERMINAL": "1",
    }


# --------------------------------------------------------------------------
# Server argv
# --------------------------------------------------------------------------


def _argv(**kwargs):
    defaults = dict(
        python="/venv/bin/python",
        model="/models/pack",
        model_id="mtplx-flash-next-optimized-speed",
        host="127.0.0.1",
        port=8092,
        ssd_session_cache="on",
    )
    defaults.update(kwargs)
    return ttft.build_server_argv(**defaults)


def test_server_argv_pins_reasoning_off_and_greedy():
    argv = _argv()
    assert argv[argv.index("--reasoning-mode") + 1] == "off"
    assert argv[argv.index("--temperature") + 1] == "0"
    assert argv[argv.index("--port") + 1] == "8092"
    assert "--no-auth" in argv


def test_server_argv_keeps_the_production_lane():
    argv = _argv()
    assert argv[argv.index("--profile") + 1] == "turbo"
    assert argv[argv.index("--generation-mode") + 1] == "mtp"
    assert argv[argv.index("--depth") + 1] == "3"
    assert argv[argv.index("--scheduler-mode") + 1] == "serial"
    assert "--load-mtp" in argv
    # The session cache is the SUBJECT of this screen; production has it on.
    assert argv[argv.index("--ssd-session-cache") + 1] == "on"


def test_server_argv_rejects_a_bad_ssd_mode():
    with pytest.raises(ValueError):
        _argv(ssd_session_cache="maybe")


def test_default_port_is_not_production():
    assert ttft.DEFAULT_PORT == 8092
    assert ttft.DEFAULT_PORT != 8080


def test_outer_command_goes_through_the_guard():
    command = ttft.outer_command_line(label="control", gdn_boundary_max=32)
    assert "run_guarded.py" in command
    assert "ttft_screen.py" in command
    assert "--gdn-boundary-max 32" in command
    assert "launchctl" not in command


# --------------------------------------------------------------------------
# Prompt + scenarios
# --------------------------------------------------------------------------


def test_workspace_is_deterministic_per_salt():
    a = ttft.synthetic_workspace(target_tokens=512, salt="x")
    b = ttft.synthetic_workspace(target_tokens=512, salt="x")
    c = ttft.synthetic_workspace(target_tokens=512, salt="y")
    assert a == b
    assert a != c
    assert a.startswith("# workspace dump x")


def test_workspace_reaches_its_target_size():
    text = ttft.synthetic_workspace(target_tokens=16_384, salt="s")
    assert len(text) >= int(16_384 * ttft.CHARS_PER_TOKEN)


def test_rerender_changes_bytes_but_not_information():
    original = "Summary line\n\n- one\n- two\n\n```python\nx = 1\n```"
    rendered = ttft.rerender_transcript(original)
    assert rendered != original
    assert "* one" in rendered
    assert "```python" not in rendered
    assert "x = 1" in rendered


def test_rerender_of_plain_text_still_diverges():
    assert ttft.rerender_transcript("hello") != "hello"


def test_cold_scenario_is_the_opening_turn_only():
    workspace = ttft.synthetic_workspace(target_tokens=64, salt="s")
    messages = ttft.build_conversation(
        scenario="cold", workspace=workspace, assistant_turn=None
    )
    assert [message["role"] for message in messages] == ["system", "user"]


def test_matching_terminal_extends_the_cold_prompt_exactly():
    workspace = ttft.synthetic_workspace(target_tokens=64, salt="s")
    cold = ttft.build_conversation(
        scenario="cold", workspace=workspace, assistant_turn=None
    )
    warm = ttft.build_conversation(
        scenario="matching_terminal", workspace=workspace, assistant_turn="reply"
    )
    assert warm[: len(cold)] == cold
    assert [message["role"] for message in warm[len(cold) :]] == ["assistant", "user"]
    assert warm[len(cold)]["content"] == "reply"


def test_rerendered_terminal_diverges_inside_the_banked_terminal():
    """The head is identical; the divergence is the ASSISTANT turn."""

    workspace = ttft.synthetic_workspace(target_tokens=64, salt="s")
    warm = ttft.build_conversation(
        scenario="matching_terminal", workspace=workspace, assistant_turn="- a\n- b"
    )
    diverged = ttft.build_conversation(
        scenario="rerendered_terminal", workspace=workspace, assistant_turn="- a\n- b"
    )
    assert warm[:-2] == diverged[:-2]           # system + long user: exact prefix
    assert warm[-2] != diverged[-2]             # the re-rendered assistant turn
    assert warm[-1] == diverged[-1]             # the new user turn is unchanged


@pytest.mark.parametrize("blank", ["\n", "\n\n\n"])
def test_rerender_no_op_is_refused_rather_than_measured_as_arm_e(blank):
    """A whitespace-only reply re-renders to itself; that must FAIL, not pass."""

    assert ttft.rerender_transcript(blank) == blank
    with pytest.raises(ValueError, match="no-op"):
        ttft.build_conversation(
            scenario="rerendered_terminal", workspace="w", assistant_turn=blank
        )


def test_warm_scenarios_need_the_assistant_turn():
    with pytest.raises(ValueError):
        ttft.build_conversation(
            scenario="matching_terminal", workspace="w", assistant_turn=None
        )


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError):
        ttft.build_conversation(scenario="warm", workspace="w", assistant_turn="a")


def test_payload_streams_with_thinking_off():
    payload = ttft.build_chat_payload(
        [{"role": "user", "content": "hi"}], model_id="m", max_tokens=8
    )
    assert payload["stream"] is True
    assert payload["enable_thinking"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0.0
    assert "tools" not in payload


def test_payload_tools_use_agent_names():
    payload = ttft.build_chat_payload(
        [{"role": "user", "content": "hi"}],
        model_id="m",
        max_tokens=8,
        tools=ttft.AGENT_TOOL_NAMES,
    )
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert names == list(ttft.AGENT_TOOL_NAMES)


# --------------------------------------------------------------------------
# SSE folding
# --------------------------------------------------------------------------


def test_parse_sse_line_shapes():
    assert ttft.parse_sse_line(b": keep-alive\n") is None
    assert ttft.parse_sse_line(b"\n") is None
    assert ttft.parse_sse_line("data: [DONE]") == "[DONE]"
    assert ttft.parse_sse_line('data: {"a": 1}') == {"a": 1}


def _chunk(content=None, *, stats=None, usage=None, finish=None, reasoning=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if stats is not None:
        chunk["mtplx_stats"] = stats
        chunk["usage"] = usage or {}
    return chunk


def test_summarize_stream_times_the_first_CONTENT_token():
    """The first SSE frame is a role-only delta; TTFT is the first token."""

    events = [
        (0.40, _chunk(content="")),          # role/opening frame, no text
        (0.55, _chunk(content="Hel")),
        (0.60, _chunk(content="lo")),
        (
            0.70,
            _chunk(
                stats={"ttft_s": 0.52, "cached_tokens": 16_000, "session_cache_hit": True},
                usage={"prompt_tokens": 16_384, "completion_tokens": 2},
                finish="stop",
            ),
        ),
        (0.71, "[DONE]"),
    ]
    result = ttft.summarize_stream(events)
    assert result["client_first_chunk_s"] == 0.40
    assert result["client_first_token_s"] == 0.55
    assert result["client_total_s"] == 0.71
    assert result["output_sha256"] == __import__("hashlib").sha256(b"Hello").hexdigest()
    assert result["usage"]["prompt_tokens"] == 16_384
    assert result["server"]["ttft_s"] == 0.52
    assert result["server"]["cached_tokens"] == 16_000
    assert result["server"]["session_cache_hit"] is True
    assert result["finish_reason"] == "stop"
    assert result["reasoning_chars"] == 0


def test_summarize_stream_surfaces_leaked_reasoning():
    result = ttft.summarize_stream([(0.1, _chunk(reasoning="thinking..."))])
    assert result["reasoning_chars"] == len("thinking...")


def test_summarize_stream_survives_a_stream_with_no_stats():
    result = ttft.summarize_stream([(0.1, _chunk(content="x"))])
    assert result["server"]["ttft_s"] is None
    assert result["usage"] == {}


def test_summary_reports_median_and_p95_and_parity():
    rows = [
        {
            "scenario": "rerendered_terminal",
            "client_first_token_s": value,
            "server": {
                "ttft_s": value - 0.05,
                "cached_tokens": 16_000,
                "session_cache_hit": True,
                "session_restore_mode": "block_prefix_boundary_reference_lease",
                "cache_miss_reason": None,
            },
            "usage": {"prompt_tokens": 16_384},
            "output_sha256": "abc",
        }
        for value in (0.5, 0.6, 2.9)
    ]
    summary = ttft.summarize_scenarios(rows)["rerendered_terminal"]
    assert summary["repeats"] == 3
    assert summary["visible_ttft_s"]["median"] == 0.6
    assert summary["visible_ttft_s"]["max"] == 2.9
    assert summary["visible_ttft_s"]["p95"] > summary["visible_ttft_s"]["median"]
    assert summary["model_ttft_s"]["median"] == pytest.approx(0.55)
    assert summary["output_deterministic"] is True
    assert summary["cached_tokens"] == [16_000]


def test_summary_flags_non_deterministic_output():
    rows = [
        {
            "scenario": "cold",
            "client_first_token_s": 1.0,
            "server": {"ttft_s": 1.0},
            "usage": {"prompt_tokens": 10},
            "output_sha256": sha,
        }
        for sha in ("abc", "def")
    ]
    assert ttft.summarize_scenarios(rows)["cold"]["output_deterministic"] is False


# --------------------------------------------------------------------------
# Contract assertion
# --------------------------------------------------------------------------


_HEALTH = {
    "model": "m",
    "generation_mode": "mtp",
    "mtp_enabled": True,
    "depth": 3,
    "profile": {"name": "turbo"},
    "scheduler": {"mode": "serial"},
}
_SETTINGS = {
    "reasoning": "off",
    "enable_thinking": False,
    "generation_mode": "mtp",
    "depth": 3,
}


def test_contract_accepts_the_configured_server():
    ttft.assert_server_contract(_HEALTH, _SETTINGS, model_id="m")


def test_contract_refuses_a_thinking_server():
    with pytest.raises(RuntimeError, match="reasoning"):
        ttft.assert_server_contract(
            _HEALTH, {**_SETTINGS, "reasoning": "xhigh"}, model_id="m"
        )


def test_contract_refuses_the_wrong_model_id():
    with pytest.raises(RuntimeError, match="health.model"):
        ttft.assert_server_contract(_HEALTH, _SETTINGS, model_id="other")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_dry_run_touches_nothing(capsys):
    assert ttft.main(["--label", "control", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "run_guarded.py" in out
    assert "--reasoning-mode off" in out
    assert "x-mtplx-session-id" in out


def test_dry_run_prints_the_candidate_env(capsys):
    ttft.main(
        [
            "--label",
            "staging",
            "--dry-run",
            "--env",
            "MTPLX_FABLE_QSA_RESTORE_STAGING=1",
            "--gdn-boundary-max",
            "32",
        ]
    )
    out = capsys.readouterr().out
    printed = json.loads(out.split("candidate env: ")[1].splitlines()[0])
    assert printed == {
        "MTPLX_FABLE_QSA_RESTORE_STAGING": "1",
        "MTPLX_GDN_BOUNDARY_MAX": "32",
    }
