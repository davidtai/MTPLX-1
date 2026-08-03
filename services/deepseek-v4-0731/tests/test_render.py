"""Golden contracts for the isolated DeepSeek 0731 prompt renderer."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))

from render import (  # noqa: E402
    AssetIntegrityError,
    InvalidReasoningEffort,
    render_chat,
    verify_assets,
)


def test_golden_messages_tools_tool_result_and_reasoning() -> None:
    rendered = render_chat(
        [
            {"role": "system", "content": "Be exact."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"x":"2+2"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate arithmetic.",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            }
        ],
        reasoning_effort="high",
    )
    assert hashlib.sha256(rendered.encode()).hexdigest() == "f0d541c389ee21f1a4a1f50b8624c44f06a5e30b8d65a37dc0bfdc2edd05f11e"


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_reasoning_effort_has_stable_rendering(effort: str) -> None:
    assert render_chat([{"role": "user", "content": "hi"}], reasoning_effort=effort).startswith(
        f"<｜begin▁of▁sentence｜><｜User｜>Reasoning: {effort}"
    )


def test_invalid_reasoning_effort_fails_closed() -> None:
    with pytest.raises(InvalidReasoningEffort):
        render_chat([{"role": "user", "content": "hi"}], reasoning_effort="medium")


def test_missing_or_tampered_asset_fails_closed(tmp_path: Path) -> None:
    manifest = SERVICE / "encoding" / "SHA256SUMS"
    with pytest.raises(AssetIntegrityError):
        verify_assets(tmp_path / "missing", manifest)

    asset = tmp_path / "chat_template.jinja"
    asset.write_text("tampered", encoding="utf-8")
    with pytest.raises(AssetIntegrityError):
        verify_assets(asset, manifest)
