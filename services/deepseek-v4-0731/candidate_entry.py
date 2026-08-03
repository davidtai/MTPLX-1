#!/usr/bin/env python3
"""Construction-only entrypoint for the isolated V4-Flash-0731 service.

This module verifies and self-tests the official encoder before replacing the
two MTPLX request-path call sites that own prompt encoding and DSML completion
parsing. There is no tokenizer-template or stock-prompt fallback after install.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parent
ENCODING_ROOT = ROOT / "encoding"
MANIFEST = ENCODING_ROOT / "SHA256SUMS"
SOURCE_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
ENCODER_NAME = "deepseek-v4-flash-0731-official"
REQUIRED_ASSETS = (
    "encoding_dsv4.py",
    "tests/test_input_1.json",
    "tests/test_input_2.json",
    "tests/test_input_3.json",
    "tests/test_input_4.json",
    "tests/test_output_1.txt",
    "tests/test_output_2.txt",
    "tests/test_output_3.txt",
    "tests/test_output_4.txt",
)


class CandidateConstructionError(RuntimeError):
    """The isolated service cannot install its reviewed request surface."""


def _manifest_entries() -> dict[str, str]:
    if not MANIFEST.is_file() or MANIFEST.is_symlink():
        raise CandidateConstructionError("official encoding manifest is missing or unsafe")
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)", line)
        if match is None or match.group(2) in entries:
            raise CandidateConstructionError("official encoding manifest is malformed")
        entries[match.group(2)] = match.group(1)
    if tuple(entries) != REQUIRED_ASSETS:
        raise CandidateConstructionError("official encoding manifest asset set changed")
    return entries


def verify_official_assets() -> dict[str, str]:
    """Verify the exact official source and vector set once at construction."""
    entries = _manifest_entries()
    root = ENCODING_ROOT.resolve()
    for relative, expected in entries.items():
        path = ENCODING_ROOT / relative
        if not path.is_file() or path.is_symlink() or path.resolve().parent != (root / relative).parent:
            raise CandidateConstructionError("official encoding asset is missing or unsafe")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise CandidateConstructionError("official encoding asset digest mismatch")
    return entries


def _load_official_encoder() -> ModuleType:
    verify_official_assets()
    path = ENCODING_ROOT / "encoding_dsv4.py"
    module = ModuleType("mtplx_dsv4_0731_official_encoding")
    module.__file__ = str(path)
    # Compile the already-verified source bytes directly. This cannot select an
    # ignored or stale __pycache__ artifact in place of the reviewed encoder.
    code = compile(path.read_bytes(), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _self_test(encoding: ModuleType) -> None:
    """Run all four official byte vectors before installing the request path."""
    for case in range(1, 5):
        payload = json.loads((ENCODING_ROOT / f"tests/test_input_{case}.json").read_text(encoding="utf-8"))
        if case == 1:
            messages = payload["messages"]
            messages[0]["tools"] = payload["tools"]
        else:
            messages = payload
        mode = "chat" if case == 4 else "thinking"
        expected = (ENCODING_ROOT / f"tests/test_output_{case}.txt").read_text(encoding="utf-8")
        if encoding.encode_messages(messages, thinking_mode=mode) != expected:
            raise CandidateConstructionError(f"official encoding vector {case} failed")


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        value = dump(exclude_none=True)
        if isinstance(value, dict):
            return value
    value: dict[str, Any] = {}
    for field in ("role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"):
        if hasattr(message, field):
            item = getattr(message, field)
            if item is not None:
                value[field] = item
    if "role" not in value:
        raise CandidateConstructionError("request message has no role")
    return value


def _install_encoder(server: ModuleType, encoding: ModuleType):
    encode_text = server._encode_rendered_chat_text

    def encode_messages(
        tokenizer: Any,
        messages: list[Any],
        *,
        enable_thinking: bool,
        reasoning_effort: str | None = None,
        strip_assistant_reasoning_history: bool = False,
        scoped_reasoning_history: bool = False,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        tool_prompt_mode: str = "native",
        template_observability: dict[str, Any] | None = None,
    ) -> list[int]:
        del strip_assistant_reasoning_history, scoped_reasoning_history, tool_prompt_mode
        if tool_choice not in (None, "auto"):
            raise CandidateConstructionError("V4-0731 candidate does not support forced tool_choice")
        effort = reasoning_effort or "low"
        if effort not in encoding.REASONING_EFFORT_PROMPTS:
            raise CandidateConstructionError("reasoning_effort must be one of: low, high, max")
        prepared = [_message_dict(message) for message in messages]
        if not prepared:
            prepared = [{"role": "user", "content": ""}]
        if tools:
            if prepared[0].get("role") != "system":
                prepared.insert(0, {"role": "system", "content": "", "tools": tools})
            else:
                prepared[0] = {**prepared[0], "tools": tools}
        mode = "thinking" if enable_thinking else "chat"
        rendered = encoding.encode_messages(
            prepared,
            thinking_mode=mode,
            reasoning_effort=effort,
        )
        if not add_generation_prompt:
            suffix = encoding.ASSISTANT_SP_TOKEN + (
                encoding.thinking_start_token if enable_thinking else encoding.thinking_end_token
            )
            if rendered.endswith(suffix):
                rendered = rendered[: -len(suffix)]
        if template_observability is not None:
            template_observability.update(
                {
                    "backend_chat_encoding": ENCODER_NAME,
                    "encoding_source_revision": SOURCE_REVISION,
                }
            )
        return encode_text(tokenizer, rendered)

    return encode_messages


def _install_completion_parser(server: ModuleType, encoding: ModuleType):
    stock = server._parse_generated_tool_calls_or_content
    dsml_marker = f"<{encoding.dsml_token}{encoding.tool_calls_block_name}>"

    def parse_generated_tool_calls_or_content(
        text: str,
        *,
        tools: list[dict[str, Any]],
        tokenizer: Any | None = None,
        state: Any | None = None,
        response_id: str | None = None,
        stream: bool = False,
    ):
        if dsml_marker not in text:
            return stock(
                text,
                tools=tools,
                tokenizer=tokenizer,
                state=state,
                response_id=response_id,
                stream=stream,
            )
        completion = text if text.endswith(encoding.eos_token) else text + encoding.eos_token
        mode = "thinking" if encoding.thinking_end_token in completion.split(dsml_marker, 1)[0] else "chat"
        try:
            parsed = encoding.parse_message_from_completion_text(completion, thinking_mode=mode)
        except (AssertionError, ValueError) as error:
            raise CandidateConstructionError("malformed V4-0731 DSML completion") from error
        calls = parsed.get("tool_calls") or None
        return calls, None

    return parse_generated_tool_calls_or_content


def _install_actual_tool_extractor(server: ModuleType, encoding: ModuleType) -> None:
    """Install official DSML completion parsing at the live response call site."""
    from mtplx.server.omlx_bridge import ToolCallExtraction

    stock = server.omlx_extract_tool_calls_with_thinking
    dsml_marker = f"<{encoding.dsml_token}{encoding.tool_calls_block_name}>"

    def extract(
        thinking_content: str,
        regular_content: str,
        tokenizer: Any | None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ToolCallExtraction:
        combined = thinking_content + regular_content
        if dsml_marker not in combined:
            return stock(thinking_content, regular_content, tokenizer, tools)
        mode = "thinking" if thinking_content else "chat"
        completion = (
            thinking_content + encoding.thinking_end_token + regular_content
            if mode == "thinking"
            else regular_content
        )
        if not completion.endswith(encoding.eos_token):
            completion += encoding.eos_token
        try:
            parsed = encoding.parse_message_from_completion_text(completion, thinking_mode=mode)
        except (AssertionError, ValueError) as error:
            raise CandidateConstructionError("malformed V4-0731 DSML completion") from error
        calls = parsed.get("tool_calls") or None
        if calls:
            calls = [
                {**call, "id": str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")}
                for call in calls
            ]
        return ToolCallExtraction(
            cleaned_text=str(parsed.get("content") or ""),
            tool_calls=calls,
            cleaned_thinking=str(parsed.get("reasoning_content") or ""),
            parser_source="deepseek_v4_0731_official",
            status="parsed" if calls else "no_tool",
            raw_tool_markup_suppressed=True,
        )

    server.omlx_extract_tool_calls_with_thinking = extract


def _install_stream_translator(server: ModuleType, encoding: ModuleType) -> None:
    """Buffer the official DSML envelope so streaming never leaks it as text."""
    stock_class = server._ToolAwareContentStreamTranslator
    dsml_marker = f"<{encoding.dsml_token}{encoding.tool_calls_block_name}>"

    class DSV40731StreamTranslator:
        def __init__(self, *, tools, argument_chunk_chars, tokenizer=None, **kwargs) -> None:
            self._tools = tools
            self._argument_chunk_chars = argument_chunk_chars
            self._tokenizer = tokenizer
            self._stock = stock_class(
                tools=tools,
                argument_chunk_chars=argument_chunk_chars,
                tokenizer=tokenizer,
                **kwargs,
            )
            self._pending = ""
            self._mode = "undecided"
            self.tool_calls = None
            self.fallback_reason = None
            self.tool_parser_dialect = "deepseek_v4_0731_official"
            self._suppressed = False

        @property
        def has_tool_calls(self):
            return bool(self.tool_calls) if self._mode == "dsml" else self._stock.has_tool_calls

        @property
        def has_emitted_tool_deltas(self):
            return False if self._mode == "dsml" else self._stock.has_emitted_tool_deltas

        @property
        def suppressed_tool_markup(self):
            return self._suppressed or self._stock.suppressed_tool_markup

        @property
        def buffering_tool_call(self):
            return self._mode == "dsml" or self._stock.buffering_tool_call

        @property
        def tool_argument_in_progress(self):
            return self._mode == "dsml" or self._stock.tool_argument_in_progress

        @property
        def ready_to_finish_tool_turn(self):
            return False if self._mode == "dsml" else self._stock.ready_to_finish_tool_turn

        @property
        def invalid_trailing_after_tool_call(self):
            return False if self._mode == "dsml" else self._stock.invalid_trailing_after_tool_call

        def feed(self, field: str, text: str):
            if self._mode == "stock":
                return self._stock.feed(field, text)
            if field != "content":
                return self._stock.feed(field, text)
            self._pending += text
            stripped = self._pending.lstrip()
            if dsml_marker in stripped:
                self._mode = "dsml"
                self._suppressed = True
                return []
            if dsml_marker.startswith(stripped):
                return []
            self._mode = "stock"
            pending, self._pending = self._pending, ""
            return self._stock.feed(field, pending)

        def finish(self, *, defer_content_resolution: bool = False):
            if self._mode != "dsml":
                if self._pending:
                    self._stock.feed("content", self._pending)
                    self._pending = ""
                return self._stock.finish(defer_content_resolution=defer_content_resolution)
            extraction = server.omlx_extract_tool_calls_with_thinking(
                "", self._pending, self._tokenizer, self._tools
            )
            self.tool_calls = extraction.tool_calls
            self._pending = ""
            if not self.tool_calls:
                raise CandidateConstructionError("official DSML stream ended without tool calls")
            return list(
                server._stream_tool_call_deltas(
                    self.tool_calls,
                    argument_chunk_chars=self._argument_chunk_chars,
                )
            )

        def resolve_deferred_content(self, *, has_tool_calls: bool):
            if self._mode == "dsml":
                return []
            return self._stock.resolve_deferred_content(has_tool_calls=has_tool_calls)

    server._ToolAwareContentStreamTranslator = DSV40731StreamTranslator


def _install_reasoning_policy(server: ModuleType) -> None:
    def normalize(value: Any, *, default: str = "low") -> str:
        effort = str(value or default).strip().lower()
        if effort not in {"auto", "low", "high", "max"}:
            raise ValueError("reasoning_effort must be one of: auto, low, high, max")
        return effort

    def for_state(
        state: Any,
        *,
        thinking_enabled: bool,
        request_effort: str | None = None,
        allow_client_controls: bool = True,
    ) -> str | None:
        if not thinking_enabled:
            return None
        raw = request_effort if request_effort is not None and allow_client_controls else state.args.reasoning_effort
        effort = normalize(raw, default="low")
        return "low" if effort == "auto" else effort

    server._normalize_reasoning_effort = normalize
    server._reasoning_effort_for_state = for_state


def _install_construction_identity(server: ModuleType) -> None:
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    def apply_profile(_tokenizer: Any, _args: Any) -> dict[str, Any]:
        return {
            "profile": ENCODER_NAME,
            "source": "official_python_encoder",
            "path": None,
            "applied": True,
            "sha256": manifest_digest,
        }

    server._apply_chat_template_profile = apply_profile
    server._template_hash = lambda _tokenizer: f"{ENCODER_NAME}:{manifest_digest}"
    server._template_supports_scoped_reasoning = lambda _tokenizer: True


def install_candidate_surface(server: ModuleType) -> dict[str, str]:
    """Install the verified encoder/parser directly into the imported server."""
    encoding = _load_official_encoder()
    _self_test(encoding)
    if not hasattr(server, "_encode_rendered_chat_text"):
        # Unit fixture: retain the same strict no-special-token encoding contract.
        server._encode_rendered_chat_text = lambda tokenizer, text: list(
            tokenizer.encode(text, add_special_tokens=False)
        )
    server._encode_messages = _install_encoder(server, encoding)
    server._parse_generated_tool_calls_or_content = _install_completion_parser(server, encoding)
    if hasattr(server, "omlx_extract_tool_calls_with_thinking"):
        _install_actual_tool_extractor(server, encoding)
    if hasattr(server, "_ToolAwareContentStreamTranslator"):
        _install_stream_translator(server, encoding)
    _install_reasoning_policy(server)
    _install_construction_identity(server)
    server._DSV4_0731_ENCODER_INSTALLED = True
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    return {
        "encoder": ENCODER_NAME,
        "source_revision": SOURCE_REVISION,
        "asset_set_sha256": manifest_digest,
    }


def main() -> int:
    if sys.argv[1:]:
        raise CandidateConstructionError("candidate entrypoint accepts no arguments")
    from mtplx.server import openai as server
    from mtplx.cli import main as mtplx_main

    install_candidate_surface(server)
    return mtplx_main(
        [
            "serve",
            "--host", "127.0.0.1",
            "--port", "8081",
            "--model", "/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp",
            "--model-id", "deepseek-v4-0731-candidate",
            "--reasoning", "on",
            "--reasoning-effort", "low",
            "--reasoning-parser", "qwen3",
            "--tool-prompt-mode", "native",
            "--chat-template-profile", "tokenizer",
            "--warmup-tokens", "0",
            "--no-stats-footer",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
