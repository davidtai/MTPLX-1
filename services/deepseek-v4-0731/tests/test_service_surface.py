"""Static safety contracts for scripts that must not be run in unit tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_candidate_is_distinct_from_live_service() -> None:
    launch = (ROOT / "launch_candidate.sh").read_text(encoding="utf-8")
    plist = (ROOT / "com.tea.deepseek-v4-0731.candidate.plist").read_text(encoding="utf-8")
    assert "com.tea.deepseek-v4-0731.candidate" in launch + plist
    assert "PORT=8081" in launch
    assert "8080" not in launch
    assert "launchctl" not in launch
    assert "/usr/bin/env -i" in launch
    assert "command environment override rejected" in launch
    assert "MTPLX_DSV4_0731_TEST_FIXTURE" in launch
    assert "candidate_entry.py" in launch
    assert "-m mtplx" not in launch


def test_candidate_config_pins_all_installation_identities() -> None:
    config = json.loads((ROOT / "candidate.json").read_text(encoding="utf-8"))
    assert config["candidate_port"] == 8081
    assert config["candidate_label"] == "com.tea.deepseek-v4-0731.candidate"
    assert config["encoding_source_revision"] == "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    for key in ("model_config_sha256", "model_index_sha256", "trusted_python_sha256"):
        assert len(config[key]) == 64
    assert config["reviewed_ref"] == "refs/tags/mtplx-dsv4-0731-reviewed"
    assert len(config["encoding_assets"]) == 9
    for relative, expected in config["encoding_assets"].items():
        assert hashlib.sha256((ROOT / "encoding" / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / "encoding/SHA256SUMS").read_bytes()).hexdigest() == config[
        "encoding_manifest_sha256"
    ]
    assert hashlib.sha256((ROOT / "candidate_entry.py").read_bytes()).hexdigest() == config[
        "candidate_entry_sha256"
    ]
    assert hashlib.sha256((ROOT / "com.tea.deepseek-v4-0731.candidate.plist").read_bytes()).hexdigest() == config[
        "candidate_plist_sha256"
    ]


def test_command_override_is_rejected_except_for_nonstarting_fixture() -> None:
    launcher = ROOT / "launch_candidate.sh"
    fixture_env = {"PATH": os.environ["PATH"], "MTPLX_DSV4_0731_TEST_FIXTURE": "1"}
    fixture = subprocess.run(
        [str(launcher), "--print-command"], env=fixture_env, check=True, capture_output=True, text=True
    )
    assert "127.0.0.1:8081" in fixture.stdout

    rejected = subprocess.run(
        [str(launcher), "--print-command"],
        env={"PATH": os.environ["PATH"], "MTPLX_DSV4_0731_EXECUTABLE": "/bin/false"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "override rejected" in rejected.stderr


def test_cutover_requires_receipts_lock_identity_and_explicit_promotion() -> None:
    source = (ROOT / "promote_cutover.py").read_text(encoding="utf-8")
    for required in (
        "LOCK_NB",
        "--promote",
        "assert_candidate_receipt",
        "assert_live_identity",
        "/v1/models",
        "finish_reason",
        "SENSITIVE_KEY",
        "finally:",
        "_bootstrap(prior_plist)",
        "candidate_model_ids",
    ):
        assert required in source


def test_server_construction_installs_verified_0731_encoder() -> None:
    from candidate_entry import install_candidate_surface
    from mtplx.server import openai as openai_server

    def stock(*_args, **_kwargs):
        return [999]

    server = SimpleNamespace(
        _encode_messages=stock,
        _parse_generated_tool_calls_or_content=lambda *_args, **_kwargs: (None, None),
        omlx_extract_tool_calls_with_thinking=lambda *_args, **_kwargs: None,
        _ToolAwareContentStreamTranslator=openai_server._ToolAwareContentStreamTranslator,
        _stream_tool_call_deltas=openai_server._stream_tool_call_deltas,
    )
    receipt = install_candidate_surface(server)
    assert server._encode_messages is not stock
    assert receipt["encoder"] == "deepseek-v4-flash-0731-official"
    assert server._template_hash(None).startswith("deepseek-v4-flash-0731-official:")
    assert server._apply_chat_template_profile(None, None) == {
        "profile": "deepseek-v4-flash-0731-official",
        "source": "official_python_encoder",
        "path": None,
        "applied": True,
        "sha256": receipt["asset_set_sha256"],
    }

    class Tokenizer:
        def __init__(self) -> None:
            self.encoded = ""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            self.encoded = text
            return list(text.encode("utf-8"))

    tokenizer = Tokenizer()
    observability: dict[str, object] = {}
    ids = server._encode_messages(
        tokenizer,
        [SimpleNamespace(role="user", content="hello", tool_calls=None)],
        enable_thinking=True,
        reasoning_effort="high",
        tools=None,
        template_observability=observability,
    )
    assert ids == list(tokenizer.encoded.encode("utf-8"))
    assert "<｜Assistant｜><think>" in tokenizer.encoded
    assert observability == {
        "backend_chat_encoding": "deepseek-v4-flash-0731-official",
        "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    }

    vector = (ROOT / "encoding/tests/test_output_1.txt").read_text(encoding="utf-8")
    marker = "<｜Assistant｜><think>"
    start = vector.find(marker) + len(marker)
    end = vector.find("<｜User｜>", start)
    thinking, regular = vector[start:end].split("</think>", 1)
    extraction = server.omlx_extract_tool_calls_with_thinking(thinking, regular, tokenizer, [])
    assert extraction.parser_source == "deepseek_v4_0731_official"
    assert extraction.tool_calls[0]["function"]["name"] == "get_weather"
    assert extraction.tool_calls[0]["id"].startswith("call_")

    translator = server._ToolAwareContentStreamTranslator(
        tools=[], argument_chunk_chars=16, tokenizer=tokenizer
    )
    midpoint = len(regular) // 2
    assert translator.feed("content", regular[:midpoint]) == []
    assert translator.feed("content", regular[midpoint:]) == []
    deltas = translator.finish()
    assert translator.suppressed_tool_markup is True
    assert translator.tool_calls[0]["function"]["name"] == "get_weather"
    assert any("tool_calls" in delta for delta in deltas)


@pytest.mark.parametrize("forbidden_key, forbidden_value", [
    ("stdout", "must never be retained"),
    ("prompt", "must never be retained"),
    ("tools", []),
    ("secret", "must never be retained"),
    ("argv", ["must never be retained"]),
    ("env", {"MUST_NEVER": "be retained"}),
    ("model_path", "/Users/davidtai/models/private"),
    ("note", '{"messages":[{"role":"user"}]}'),
    ("note", "Bearer abcdefghijklmnop"),
])
def test_candidate_receipt_rejects_sensitive_capture(forbidden_key: str, forbidden_value: object) -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from promote_cutover import PromotionError, assert_candidate_receipt  # noqa: PLC0415

    receipt = {
        "candidate_preflight": {
            "ok": True,
            "label": "com.tea.deepseek-v4-0731.candidate",
            "port": 8081,
            "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "a" * 64},
        },
        "candidate_smoke": {"ok": True, "models_ok": True, "ready": True, "finish_reason": "stop"},
        forbidden_key: forbidden_value,
    }
    with pytest.raises(PromotionError, match="sensitive"):
        assert_candidate_receipt(receipt)


def test_scrubbed_passing_candidate_receipt_is_accepted() -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from promote_cutover import assert_candidate_receipt  # noqa: PLC0415

    assert_candidate_receipt(
        {
            "schema": "mtplx.dsv4-0731-candidate.v1",
            "candidate_preflight": {
                "ok": True,
                "label": "com.tea.deepseek-v4-0731.candidate",
                "port": 8081,
                "plist_sha256": "a" * 64,
                "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
                "encoding_asset_set_sha256": "b" * 64,
                "reviewed_commit": "c" * 40,
                "model_config_sha256": "d" * 64,
                "model_index_sha256": "e" * 64,
                "promotion_target": {"label": "com.tea.deepseek-v4-0731.production", "plist_sha256": "b" * 64},
            },
            "candidate_smoke": {
                "ok": True,
                "models_ok": True,
                "ready": True,
                "finish_reason": "stop",
                "candidate_model_ids": ["deepseek-v4-0731"],
            },
        }
    )


def test_candidate_receipt_rejects_unknown_nested_fields() -> None:
    from promote_cutover import PromotionError, assert_candidate_receipt

    receipt = {
        "schema": "mtplx.dsv4-0731-candidate.v1",
        "candidate_preflight": {
            "ok": True,
            "label": "com.tea.deepseek-v4-0731.candidate",
            "port": 8081,
            "plist_sha256": "a" * 64,
            "encoding_source_revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            "encoding_asset_set_sha256": "b" * 64,
            "reviewed_commit": "c" * 40,
            "model_config_sha256": "d" * 64,
            "model_index_sha256": "e" * 64,
            "promotion_target": {
                "label": "com.tea.deepseek-v4-0731.production",
                "plist_sha256": "f" * 64,
                "unexpected": True,
            },
        },
        "candidate_smoke": {
            "ok": True,
            "models_ok": True,
            "ready": True,
            "finish_reason": "stop",
            "candidate_model_ids": ["deepseek-v4-0731"],
        },
    }
    with pytest.raises(PromotionError, match="strict receipt schema"):
        assert_candidate_receipt(receipt)
