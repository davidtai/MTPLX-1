"""Pinned, dependency-free renderer for the isolated DeepSeek 0731 candidate.

The asset check runs when a :class:`PinnedEncoding` is installed.  Rendering is
then branch-free with respect to asset identity: a checked immutable encoding is
the only thing that can be installed.  This keeps integrity work out of the
request path while failing closed before a candidate can accept requests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SERVICE_ROOT = Path(__file__).resolve().parent
ENCODING_DIR = SERVICE_ROOT / "encoding"
ASSET_PATH = ENCODING_DIR / "chat_template.jinja"
MANIFEST_PATH = ENCODING_DIR / "SHA256SUMS"

BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"
TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
TOOL_SEPARATOR = "<｜tool▁sep｜>"
TOOL_CALL_END = "<｜tool▁call▁end｜>"
TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
TOOL_OUTPUT_BEGIN = "<｜tool▁output▁begin｜>"
TOOL_OUTPUT_END = "<｜tool▁output▁end｜>"


class AssetIntegrityError(RuntimeError):
    """The pinned source asset cannot safely be installed."""


class InvalidReasoningEffort(ValueError):
    """Only the candidate's explicitly tested reasoning profiles are valid."""


def _manifest_hash(manifest: Path, filename: str) -> str:
    if not manifest.is_file() or manifest.is_symlink():
        raise AssetIntegrityError("encoding manifest is missing or unsafe")
    matches: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == filename and len(fields[0]) == 64:
            matches.append(fields[0].lower())
    if len(matches) != 1 or any(c not in "0123456789abcdef" for c in matches[0]):
        raise AssetIntegrityError("encoding manifest has no unique valid asset digest")
    return matches[0]


def verify_assets(asset_path: Path = ASSET_PATH, manifest_path: Path = MANIFEST_PATH) -> str:
    """Return the verified digest, rejecting all incomplete or altered assets."""
    if not asset_path.is_file() or asset_path.is_symlink():
        raise AssetIntegrityError("pinned encoding asset is missing or unsafe")
    expected = _manifest_hash(manifest_path, asset_path.name)
    actual = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    if actual != expected:
        raise AssetIntegrityError("pinned encoding asset digest mismatch")
    return actual


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_tools(tools: Sequence[Mapping[str, Any]] | None) -> str:
    if not tools:
        return ""
    canonical = _canonical_json(list(tools))
    return f"\n\n# Tools\n{canonical}"


@dataclass(frozen=True)
class PinnedEncoding:
    """An encoding asset checked exactly once at its installation boundary."""

    asset_sha256: str

    @classmethod
    def install(cls, asset_path: Path = ASSET_PATH, manifest_path: Path = MANIFEST_PATH) -> "PinnedEncoding":
        return cls(asset_sha256=verify_assets(asset_path, manifest_path))

    def render(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        reasoning_effort: str = "low",
    ) -> str:
        effort = reasoning_effort.strip().lower() if isinstance(reasoning_effort, str) else ""
        if effort not in {"low", "high", "max"}:
            raise InvalidReasoningEffort("reasoning_effort must be one of: low, high, max")

        items = list(messages)
        system = "\n\n".join(
            _text(message.get("content"), "system content")
            for message in items
            if message.get("role") == "system"
        )
        output = [BOS, system, _render_tools(tools)]
        last_was_user = False
        for message in items:
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                output.extend((USER, f"Reasoning: {effort}\n", _text(message.get("content"), "user content")))
                last_was_user = True
                continue
            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls is not None:
                    if not isinstance(tool_calls, list) or not tool_calls:
                        raise ValueError("assistant tool_calls must be a non-empty list")
                    output.extend((ASSISTANT, "</think>", TOOL_CALLS_BEGIN))
                    for call in tool_calls:
                        function = call.get("function") if isinstance(call, Mapping) else None
                        if not isinstance(function, Mapping):
                            raise ValueError("tool call function must be an object")
                        output.extend((
                            TOOL_CALL_BEGIN,
                            _text(function.get("name"), "tool function name"),
                            TOOL_SEPARATOR,
                            _text(function.get("arguments"), "tool function arguments"),
                            TOOL_CALL_END,
                        ))
                    output.extend((TOOL_CALLS_END, EOS))
                else:
                    content = _text(message.get("content"), "assistant content")
                    if last_was_user:
                        output.extend((ASSISTANT, "</think>"))
                    output.extend((content.split("</think>", 1)[-1], EOS))
                last_was_user = False
                continue
            if role == "tool":
                _text(message.get("tool_call_id"), "tool_call_id")
                output.extend((TOOL_OUTPUT_BEGIN, _text(message.get("content"), "tool content"), TOOL_OUTPUT_END))
                last_was_user = False
                continue
            raise ValueError("unsupported message role")
        if last_was_user:
            output.extend((ASSISTANT, "<think>"))
        return "".join(output)


_DEFAULT_ENCODING = PinnedEncoding.install()


def render_chat(
    messages: Iterable[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: str = "low",
) -> str:
    """Render through the already-installed default candidate encoding."""
    return _DEFAULT_ENCODING.render(messages, tools=tools, reasoning_effort=reasoning_effort)
